use std::collections::HashMap;

use candle_core::{bail, Result, D};
use paged_kv_core::{CacheError, KvBackend, Prefill, Scheduler, SeqId};

use crate::model::{PagedQwen2, SeqInput};

fn cache_err(e: CacheError) -> candle_core::Error {
    candle_core::Error::Msg(format!("scheduler: {e}"))
}

struct Request {
    /// Prompt, then every token sampled so far.
    tokens: Vec<u32>,
    prompt_len: usize,
    max_new_tokens: usize,
}

/// A request that has stopped generating.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finished {
    pub id: SeqId,
    pub tokens: Vec<u32>,
}

/// What one step produced.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct StepOutput {
    /// The token each running sequence sampled, in batch order.
    pub sampled: Vec<(SeqId, u32)>,
    /// Requests that stopped this step, with everything they generated.
    pub finished: Vec<Finished>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct EngineStats {
    pub steps: usize,
    /// Includes recomputation after preemption.
    pub prefill_tokens_computed: usize,
    pub prefill_tokens_cached: usize,
    pub preemptions: usize,
    pub tokens_generated: usize,
}

/// Greedy generation driven by [`Scheduler`]: the scheduler decides which
/// sequences run and where each token's K/V goes, and the model fills it in.
///
/// After every step a running sequence's block table has one slot more than
/// it has K/V for: the slot of the token just sampled, whose K/V is written
/// when that token is fed back in on the next step.
pub struct Engine<B: KvBackend> {
    model: PagedQwen2,
    cache: B,
    scheduler: Scheduler,
    requests: HashMap<SeqId, Request>,
    stop_tokens: Vec<u32>,
    stats: EngineStats,
}

impl<B: KvBackend> Engine<B> {
    pub fn new(model: PagedQwen2, cache: B, max_batch_size: usize, prefix_caching: bool) -> Self {
        let scheduler = if prefix_caching {
            Scheduler::with_prefix_caching(cache.config(), max_batch_size)
        } else {
            Scheduler::new(cache.config(), max_batch_size)
        };
        Self {
            model,
            cache,
            scheduler,
            requests: HashMap::new(),
            stop_tokens: Vec::new(),
            stats: EngineStats::default(),
        }
    }

    /// Requests also finish when they sample any of these.
    pub fn stop_at(mut self, tokens: &[u32]) -> Self {
        self.stop_tokens = tokens.to_vec();
        self
    }

    pub fn add_request(
        &mut self,
        id: SeqId,
        prompt: Vec<u32>,
        max_new_tokens: usize,
    ) -> Result<()> {
        if prompt.is_empty() || max_new_tokens == 0 {
            bail!("a request needs a prompt and at least one token to generate");
        }
        // One spare token: the engine, not the scheduler, decides when to stop.
        self.scheduler
            .add_prompt(id, prompt.clone(), max_new_tokens + 1)
            .map_err(cache_err)?;
        self.requests.insert(
            id,
            Request {
                prompt_len: prompt.len(),
                tokens: prompt,
                max_new_tokens,
            },
        );
        Ok(())
    }

    pub fn is_idle(&self) -> bool {
        self.requests.is_empty()
    }

    pub fn stats(&self) -> EngineStats {
        self.stats
    }

    pub fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }

    /// Drops a request wherever it is, freeing its blocks, as a server does
    /// when a client disconnects. Returns whether the request existed.
    pub fn cancel(&mut self, id: SeqId) -> bool {
        if self.requests.remove(&id).is_none() {
            return false;
        }
        // Every request the engine holds is in the scheduler too.
        self.scheduler.finish_sequence(id).is_ok()
    }

    /// Schedules one step, runs the batch through the model, and samples one
    /// token for every running sequence.
    pub fn step(&mut self) -> Result<StepOutput> {
        let outcome = self.scheduler.step().map_err(cache_err)?;
        self.stats.steps += 1;
        self.stats.preemptions += outcome.preempted.len();

        let copies: Vec<_> = outcome
            .decoded
            .iter()
            .filter_map(|d| d.cow)
            .map(|c| (c.src, c.dst))
            .collect();
        if !copies.is_empty() {
            self.cache.copy_blocks(&copies).map_err(cache_err)?;
        }

        let prefills: HashMap<SeqId, Prefill> =
            outcome.prefills.iter().map(|p| (p.id, *p)).collect();
        let running = self.scheduler.running_snapshot();
        let mut starts = Vec::with_capacity(running.len());
        for r in &running {
            let Some(request) = self.requests.get(&r.id) else {
                bail!(
                    "the scheduler is running #{} but the engine has no request",
                    r.id.0
                );
            };
            if r.tokens != request.tokens.len() + 1 {
                bail!(
                    "#{} has {} slots for {} known tokens",
                    r.id.0,
                    r.tokens,
                    request.tokens.len()
                );
            }
            // Admitted this step: compute everything the prefix cache didn't
            // supply. Otherwise feed back the token sampled last step.
            starts.push(match prefills.get(&r.id) {
                Some(p) => {
                    self.stats.prefill_tokens_cached += p.cached_tokens;
                    self.stats.prefill_tokens_computed += p.computed_tokens;
                    p.cached_tokens
                }
                None => request.tokens.len() - 1,
            });
        }
        if running.is_empty() {
            return Ok(StepOutput::default());
        }

        let batch: Vec<SeqInput<'_>> = running
            .iter()
            .zip(&starts)
            .map(|(r, &start)| SeqInput {
                tokens: &self.requests[&r.id].tokens[start..],
                start,
                block_table: &r.blocks,
            })
            .collect();
        let logits = self.model.forward(&mut self.cache, &batch)?;
        let next: Vec<u32> = logits.argmax(D::Minus1)?.to_vec1()?;

        let mut out = StepOutput::default();
        for (r, token) in running.iter().zip(next) {
            out.sampled.push((r.id, token));
            let request = self.requests.get_mut(&r.id).expect("checked above");
            request.tokens.push(token);
            self.stats.tokens_generated += 1;
            let generated = request.tokens.len() - request.prompt_len;
            if generated >= request.max_new_tokens || self.stop_tokens.contains(&token) {
                self.scheduler.finish_sequence(r.id).map_err(cache_err)?;
                let request = self.requests.remove(&r.id).expect("checked above");
                out.finished.push(Finished {
                    id: r.id,
                    tokens: request.tokens[request.prompt_len..].to_vec(),
                });
            }
        }
        Ok(out)
    }
}
