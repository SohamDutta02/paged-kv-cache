use std::collections::VecDeque;

use crate::allocator::{BlockAllocator, CacheStats};
use crate::block_table::{check_ownership, BlockTable, CowCopy};
use crate::config::CacheConfig;
use crate::error::{CacheError, Result};
use crate::prefix_cache::PrefixEntry;
use crate::types::{BlockId, PhysicalSlot, SeqId};

struct PendingSeq {
    id: SeqId,
    context_len: usize,
    max_new_tokens: usize,
    /// Empty when the request was added by length only.
    prompt: Vec<u32>,
}

struct RunningSeq {
    id: SeqId,
    table: BlockTable,
    max_new_tokens: usize,
    generated: usize,
    prompt: Vec<u32>,
    cached_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockInfo {
    pub id: BlockId,
    pub ref_count: u32,
    /// Holds prefix-cached content, whether or not anyone owns it.
    pub cached: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunningInfo {
    pub id: SeqId,
    /// Prompt plus tokens generated so far.
    pub tokens: usize,
    pub generated: usize,
    pub max_new_tokens: usize,
    /// In logical order.
    pub blocks: Vec<BlockId>,
    /// Prompt tokens served from the prefix cache at its latest admission.
    pub cached_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaitingInfo {
    pub id: SeqId,
    /// Tokens to prefill on admission; a preempted request's full context.
    pub context_len: usize,
    pub max_new_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct DecodedToken {
    pub id: SeqId,
    pub slot: PhysicalSlot,
    pub cow: Option<CowCopy>,
}

/// How an admitted sequence's context was filled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Prefill {
    pub id: SeqId,
    /// Reused from the prefix cache.
    pub cached_tokens: usize,
    /// Prefilled by the model.
    pub computed_tokens: usize,
}

/// Everything that happened during one [`Scheduler::step`] call.
#[derive(Debug, Clone, Default)]
pub struct StepOutcome {
    pub admitted: Vec<SeqId>,
    /// One per admitted sequence, in the same order.
    pub prefills: Vec<Prefill>,
    pub decoded: Vec<DecodedToken>,
    /// A preempted sequence's decode from the same step is removed from
    /// `decoded`: its block was reclaimed.
    pub preempted: Vec<SeqId>,
    pub finished: Vec<SeqId>,
}

pub struct Scheduler {
    allocator: BlockAllocator,
    block_size: usize,
    max_batch_size: usize,
    waiting: VecDeque<PendingSeq>,
    running: Vec<RunningSeq>,
}

impl Scheduler {
    pub fn new(cache_config: &CacheConfig, max_batch_size: usize) -> Self {
        Self::with_allocator(
            BlockAllocator::new(cache_config.num_blocks),
            cache_config,
            max_batch_size,
        )
    }

    /// Requests added with [`add_prompt`](Self::add_prompt) reuse cached KV
    /// for any whole blocks of their prompt seen before.
    pub fn with_prefix_caching(cache_config: &CacheConfig, max_batch_size: usize) -> Self {
        Self::with_allocator(
            BlockAllocator::with_prefix_caching(cache_config.num_blocks),
            cache_config,
            max_batch_size,
        )
    }

    fn with_allocator(
        allocator: BlockAllocator,
        cache_config: &CacheConfig,
        max_batch_size: usize,
    ) -> Self {
        Self {
            allocator,
            block_size: cache_config.block_size,
            max_batch_size,
            waiting: VecDeque::new(),
            running: Vec::new(),
        }
    }

    #[inline]
    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    #[inline]
    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    #[inline]
    pub fn is_idle(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }

    #[inline]
    pub fn pool_utilization(&self) -> f64 {
        self.allocator.utilization()
    }

    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.allocator.num_blocks()
    }

    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    #[inline]
    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    #[inline]
    pub fn prefix_caching(&self) -> bool {
        self.allocator.prefix_caching()
    }

    pub fn cache_stats(&self) -> CacheStats {
        self.allocator.cache_stats()
    }

    /// `O(num_blocks)`: for metrics and dashboards, not hot loops.
    pub fn block_snapshot(&self) -> Vec<BlockInfo> {
        (0..self.allocator.num_blocks())
            .map(|i| {
                let id = BlockId(i as u32);
                let ref_count = self.allocator.ref_count(id).unwrap_or(0);
                let cached = self.allocator.is_cached(id);
                BlockInfo {
                    id,
                    ref_count,
                    cached,
                }
            })
            .collect()
    }

    pub fn running_snapshot(&self) -> Vec<RunningInfo> {
        self.running
            .iter()
            .map(|s| RunningInfo {
                id: s.id,
                tokens: s.table.num_tokens(),
                generated: s.generated,
                max_new_tokens: s.max_new_tokens,
                blocks: s.table.physical_blocks().to_vec(),
                cached_tokens: s.cached_tokens,
            })
            .collect()
    }

    /// Front of the queue first.
    pub fn waiting_snapshot(&self) -> Vec<WaitingInfo> {
        self.waiting
            .iter()
            .map(|p| WaitingInfo {
                id: p.id,
                context_len: p.context_len,
                max_new_tokens: p.max_new_tokens,
            })
            .collect()
    }

    /// A request known only by its length, so it never shares cached blocks.
    pub fn add_request(
        &mut self,
        id: SeqId,
        prompt_len: usize,
        max_new_tokens: usize,
    ) -> Result<()> {
        self.enqueue(id, prompt_len, Vec::new(), max_new_tokens)
    }

    pub fn add_prompt(&mut self, id: SeqId, prompt: Vec<u32>, max_new_tokens: usize) -> Result<()> {
        self.enqueue(id, prompt.len(), prompt, max_new_tokens)
    }

    fn enqueue(
        &mut self,
        id: SeqId,
        prompt_len: usize,
        prompt: Vec<u32>,
        max_new_tokens: usize,
    ) -> Result<()> {
        let total_capacity = self.allocator.num_blocks() * self.block_size;
        let final_context_len = prompt_len + max_new_tokens;

        if final_context_len > total_capacity {
            return Err(CacheError::OutOfBlocks {
                requested: final_context_len.div_ceil(self.block_size),
                available: self.allocator.num_blocks(),
            });
        }

        self.waiting.push_back(PendingSeq {
            id,
            context_len: prompt_len,
            max_new_tokens,
            prompt,
        });
        Ok(())
    }

    /// Admit what fits, advance every running sequence one token (preempting
    /// if blocks run out), and free finished sequences.
    pub fn step(&mut self) -> Result<StepOutcome> {
        let mut outcome = StepOutcome::default();

        self.admit(&mut outcome)?;
        self.decode(&mut outcome)?;
        self.retire_finished(&mut outcome)?;

        Ok(outcome)
    }

    /// Strict FIFO, keeping enough free blocks for every running sequence's
    /// next decode. Without that headroom the same step's decode would preempt
    /// the newest sequence, usually the one just admitted, wasting its prefill.
    fn admit(&mut self, outcome: &mut StepOutcome) -> Result<()> {
        // Over-counts when a parent and child share a tail (only the first
        // writer copies), never under-counts.
        let mut reserved = self
            .running
            .iter()
            .map(|s| s.table.blocks_needed(&self.allocator, 1))
            .sum::<Result<usize>>()?;

        while self.running.len() < self.max_batch_size {
            let Some(pending) = self.waiting.front() else {
                break;
            };

            // The last prompt token is always computed: its logits start decoding.
            let matchable = pending
                .prompt
                .len()
                .min(pending.context_len.saturating_sub(1));
            let hit = self
                .allocator
                .match_prefix(&pending.prompt[..matchable], self.block_size);
            let cached_tokens = hit.blocks.len() * self.block_size;
            let computed_tokens = pending.context_len - cached_tokens;

            // Matched blocks are full, so the rest goes into fresh blocks,
            // including the new sequence's own first decode token. Reclaiming
            // an unowned cached block also uses up a free block.
            let needed = hit.evictable + (computed_tokens + 1).div_ceil(self.block_size);
            if needed + reserved > self.allocator.num_free() {
                break;
            }

            let pending = self
                .waiting
                .pop_front()
                .expect("front() just returned Some");
            let reused = hit.blocks.len();
            for &block in &hit.blocks {
                self.allocator.acquire_cached(block)?;
            }
            let mut table = BlockTable::from_blocks(hit.blocks, self.block_size);
            table.append_many(&mut self.allocator, computed_tokens)?;
            self.cache_prompt(&table, &pending.prompt, reused, hit.last_entry)?;
            reserved += table.blocks_needed(&self.allocator, 1)?;

            outcome.admitted.push(pending.id);
            outcome.prefills.push(Prefill {
                id: pending.id,
                cached_tokens,
                computed_tokens,
            });
            self.running.push(RunningSeq {
                id: pending.id,
                table,
                max_new_tokens: pending.max_new_tokens,
                generated: 0,
                prompt: pending.prompt,
                cached_tokens,
            });
        }
        Ok(())
    }

    /// Registers the prompt's full blocks after the `skip` that were reused.
    fn cache_prompt(
        &mut self,
        table: &BlockTable,
        prompt: &[u32],
        skip: usize,
        mut parent: PrefixEntry,
    ) -> Result<()> {
        if !self.allocator.prefix_caching() {
            return Ok(());
        }
        let blocks = table.physical_blocks();
        for (i, tokens) in prompt.chunks_exact(self.block_size).enumerate().skip(skip) {
            parent = self.allocator.cache_block(blocks[i], parent, tokens)?;
        }
        Ok(())
    }

    fn decode(&mut self, outcome: &mut StepOutcome) -> Result<()> {
        let ids_to_decode: Vec<SeqId> = self.running.iter().map(|s| s.id).collect();

        for id in ids_to_decode {
            let Some(mut idx) = self.running.iter().position(|s| s.id == id) else {
                continue; // already preempted earlier in this same loop
            };

            loop {
                match self.running[idx].table.append(&mut self.allocator) {
                    Ok((slot, cow)) => {
                        outcome.decoded.push(DecodedToken { id, slot, cow });
                        self.running[idx].generated += 1;
                        break;
                    }
                    Err(CacheError::OutOfBlocks { .. }) => {
                        // The newest sequence yields, which may be `id`
                        // itself; if so there is nothing left to retry.
                        let victim_idx = self.running.len() - 1;
                        self.preempt(victim_idx, outcome)?;

                        match self.running.iter().position(|s| s.id == id) {
                            Some(new_idx) => {
                                idx = new_idx;
                                continue;
                            }
                            None => break, // `id` was the victim; done for this round
                        }
                    }
                    Err(e) => return Err(e),
                }
            }
        }
        Ok(())
    }

    fn preempt(&mut self, idx: usize, outcome: &mut StepOutcome) -> Result<()> {
        let seq = self.running.remove(idx);

        outcome.decoded.retain(|d| d.id != seq.id);
        outcome.preempted.push(seq.id);

        let context_len = seq.table.num_tokens();
        let remaining_new_tokens = seq.max_new_tokens - seq.generated;
        seq.table.free(&mut self.allocator)?;

        self.waiting.push_front(PendingSeq {
            id: seq.id,
            context_len,
            max_new_tokens: remaining_new_tokens,
            prompt: seq.prompt,
        });
        Ok(())
    }

    fn retire_finished(&mut self, outcome: &mut StepOutcome) -> Result<()> {
        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].generated >= self.running[i].max_new_tokens {
                let seq = self.running.remove(i);
                outcome.finished.push(seq.id);
                seq.table.free(&mut self.allocator)?;
            } else {
                i += 1;
            }
        }
        Ok(())
    }

    /// Retires a sequence before `max_new_tokens`, as a model does at an
    /// end-of-sequence token. A waiting sequence is dropped from the queue.
    pub fn finish_sequence(&mut self, id: SeqId) -> Result<()> {
        if let Some(i) = self.running.iter().position(|s| s.id == id) {
            let seq = self.running.remove(i);
            return seq.table.free(&mut self.allocator);
        }
        if let Some(i) = self.waiting.iter().position(|s| s.id == id) {
            self.waiting.remove(i);
            return Ok(());
        }
        Err(CacheError::UnknownSequence { id })
    }

    /// Forks a running sequence; the child shares every block until one side
    /// writes. Counts against `max_batch_size`. `child` must be an unused id;
    /// collisions are not checked.
    pub fn fork_sequence(
        &mut self,
        parent: SeqId,
        child: SeqId,
        max_new_tokens: usize,
    ) -> Result<()> {
        let idx = self
            .running
            .iter()
            .position(|s| s.id == parent)
            .ok_or(CacheError::UnknownSequence { id: parent })?;
        if self.running.len() >= self.max_batch_size {
            return Err(CacheError::BatchFull {
                max_batch_size: self.max_batch_size,
            });
        }

        let child_table = self.running[idx].table.fork(&mut self.allocator)?;
        let prompt = self.running[idx].prompt.clone();
        self.running.push(RunningSeq {
            id: child,
            table: child_table,
            max_new_tokens,
            generated: 0,
            prompt,
            cached_tokens: 0,
        });
        Ok(())
    }

    /// For tests: the batch limit holds and refcounts match the running
    /// sequences' block tables. Waiting sequences own no blocks.
    pub fn check_invariants(&self) -> std::result::Result<(), String> {
        if self.running.len() > self.max_batch_size {
            return Err(format!(
                "{} sequences running but max_batch_size is {}",
                self.running.len(),
                self.max_batch_size
            ));
        }
        check_ownership(&self.allocator, self.running.iter().map(|s| &s.table))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(num_blocks: usize, block_size: usize) -> CacheConfig {
        CacheConfig {
            num_blocks,
            block_size,
            num_kv_heads: 1,
            head_dim: 1,
            num_layers: 1,
            dtype_bytes: 4,
        }
    }

    #[test]
    fn admits_and_decodes_a_single_sequence_to_completion() {
        // block_size=4, prompt=3: admission leaves one free slot in the
        // tail, so the first decode writes in place; the second decode
        // fills the block exactly and finishes.
        let mut s = Scheduler::new(&config(8, 4), 2);
        s.add_request(SeqId(1), 3, 2).unwrap();

        let o1 = s.step().unwrap();
        assert_eq!(o1.admitted, vec![SeqId(1)]);
        assert_eq!(o1.decoded.len(), 1);
        assert_eq!(
            o1.decoded[0].slot,
            PhysicalSlot::new(o1.decoded[0].slot.block, 3)
        );
        assert!(o1.decoded[0].cow.is_none());
        assert!(o1.finished.is_empty());
        assert_eq!(s.num_running(), 1);

        let o2 = s.step().unwrap();
        assert!(o2.admitted.is_empty());
        assert_eq!(o2.decoded.len(), 1);
        assert_eq!(o2.finished, vec![SeqId(1)]);

        assert!(s.is_idle());
        assert_eq!(
            s.pool_utilization(),
            0.0,
            "finished sequence must release every block"
        );
    }

    #[test]
    fn finish_sequence_retires_running_and_waiting_sequences() {
        let mut s = Scheduler::with_prefix_caching(&config(8, 4), 1);
        s.add_prompt(SeqId(1), (0..6).collect(), 10).unwrap();
        s.add_request(SeqId(2), 3, 10).unwrap();
        s.step().unwrap();
        assert_eq!((s.num_running(), s.num_waiting()), (1, 1));

        s.finish_sequence(SeqId(1)).unwrap();
        s.finish_sequence(SeqId(2)).unwrap();
        assert!(s.is_idle());
        assert_eq!(s.pool_utilization(), 0.0);
        s.check_invariants().unwrap();
        assert_eq!(
            s.finish_sequence(SeqId(1)),
            Err(CacheError::UnknownSequence { id: SeqId(1) })
        );
    }

    #[test]
    fn batch_size_caps_concurrent_admission() {
        let mut s = Scheduler::new(&config(16, 4), 1);
        s.add_request(SeqId(1), 2, 3).unwrap();
        s.add_request(SeqId(2), 2, 1).unwrap();

        let o1 = s.step().unwrap();
        assert_eq!(o1.admitted, vec![SeqId(1)]);
        assert_eq!(s.num_running(), 1);
        assert_eq!(s.num_waiting(), 1, "seq 2 must wait: batch is full");

        let o2 = s.step().unwrap();
        assert!(
            o2.admitted.is_empty(),
            "batch still full, seq 1 not done yet"
        );
        assert_eq!(s.num_waiting(), 1);

        let o3 = s.step().unwrap();
        assert_eq!(
            o3.finished,
            vec![SeqId(1)],
            "seq 1 reaches its 3-token budget"
        );
        assert_eq!(
            s.num_waiting(),
            1,
            "seq 2 not admitted mid-step even though room just opened"
        );

        let o4 = s.step().unwrap();
        assert_eq!(
            o4.admitted,
            vec![SeqId(2)],
            "now that seq 1 is gone, seq 2 gets its turn"
        );
        assert_eq!(
            o4.finished,
            vec![SeqId(2)],
            "seq 2's budget is 1 token, finishes immediately"
        );

        assert!(s.is_idle());
    }

    #[test]
    fn oversized_request_is_rejected_up_front() {
        let mut s = Scheduler::new(&config(4, 2), 4); // total capacity = 8 tokens
        let err = s.add_request(SeqId(1), 5, 5).unwrap_err(); // needs 10, pool holds 8
        assert!(matches!(err, CacheError::OutOfBlocks { .. }));
        assert_eq!(
            s.num_waiting(),
            0,
            "rejected request must not sit in the queue"
        );
    }

    /// Bounded, so a bug fails instead of hanging. Returns finished ids in
    /// order and the number of preemptions.
    fn drain(s: &mut Scheduler) -> (Vec<SeqId>, usize) {
        let (mut finished, mut preempted) = (Vec::new(), 0);
        for _ in 0..100_000 {
            if s.is_idle() {
                break;
            }
            let o = s.step().unwrap();
            s.check_invariants().unwrap();
            assert_admissions_survive(&o);
            preempted += o.preempted.len();
            finished.extend(o.finished);
        }
        assert!(
            s.is_idle(),
            "scheduler failed to drain within the step budget"
        );
        assert_eq!(s.pool_utilization(), 0.0, "blocks leaked after draining");
        (finished, preempted)
    }

    fn assert_admissions_survive(o: &StepOutcome) {
        for id in &o.preempted {
            assert!(
                !o.admitted.contains(id),
                "{id} was admitted and preempted in the same step"
            );
        }
    }

    #[test]
    fn admission_leaves_room_for_the_next_decode() {
        // Each request fits alone but both can't grow at once. Without
        // headroom, seq 2 was admitted and preempted in the same step, twice.
        let mut s = Scheduler::new(&config(3, 2), 2);
        s.add_request(SeqId(1), 2, 2).unwrap();
        s.add_request(SeqId(2), 2, 2).unwrap();

        // Seq 1's prompt fills one block and its first decode needs a second,
        // leaving one free block, not the two seq 2 would need. Seq 2 waits.
        let o1 = s.step().unwrap();
        assert_eq!(o1.admitted, vec![SeqId(1)]);
        assert!(o1.preempted.is_empty());
        assert_eq!(s.num_waiting(), 1);
        s.check_invariants().unwrap();

        // Seq 1 decodes into the free slot of its tail and finishes.
        let o2 = s.step().unwrap();
        assert!(o2.admitted.is_empty());
        assert_eq!(o2.finished, vec![SeqId(1)]);

        let (finished, preempted) = drain(&mut s);
        assert_eq!(finished, vec![SeqId(2)]);
        assert_eq!(preempted, 0, "nothing should ever be preempted here");
    }

    #[test]
    fn growth_of_running_sequences_still_forces_preemption() {
        // Both fit at admission; by the second decode both tails are full
        // with one free block between them.
        let mut s = Scheduler::new(&config(3, 2), 2);
        s.add_request(SeqId(1), 1, 4).unwrap();
        s.add_request(SeqId(2), 1, 4).unwrap();

        let o1 = s.step().unwrap();
        assert_eq!(o1.admitted, vec![SeqId(1), SeqId(2)]);
        assert!(o1.preempted.is_empty());

        // Seq 1 takes the last block; seq 2, the newest, preempts itself and
        // its decode this step is voided.
        let o2 = s.step().unwrap();
        assert_eq!(o2.preempted, vec![SeqId(2)]);
        assert_eq!(
            o2.decoded.iter().map(|d| d.id).collect::<Vec<_>>(),
            vec![SeqId(1)]
        );
        assert_eq!(s.num_waiting(), 1);
        s.check_invariants().unwrap();

        // Seq 2 waits for seq 1 to finish, then completes on its own.
        let (finished, preempted) = drain(&mut s);
        assert_eq!(finished, vec![SeqId(1), SeqId(2)]);
        assert_eq!(preempted, 0);
    }

    /// Random arrivals and forks against a small pool, checking invariants
    /// after every operation, then draining. With caching, most prompts start
    /// with one of a few shared prefixes. Returns prompt tokens served from
    /// the cache.
    fn run_randomized(caching: bool) -> usize {
        let cfg = config(24, 4);
        let mut s = if caching {
            Scheduler::with_prefix_caching(&cfg, 6)
        } else {
            Scheduler::new(&cfg, 6)
        };
        // Deterministic, so a failure reproduces exactly.
        let mut seed: u64 = 0xD1B5_4A32_D192_ED03;
        let mut next = |bound: usize| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as usize) % bound
        };
        let prefixes: Vec<Vec<u32>> = [8u32, 12, 16]
            .iter()
            .enumerate()
            .map(|(p, &len)| (0..len).map(|t| p as u32 * 1000 + t).collect())
            .collect();
        let mut unique_token = 1_000_000u32;
        let mut next_id = 0u64;
        let mut expected = Vec::new();
        let mut finished = Vec::new();
        let mut preemptions = 0;
        let mut cached_tokens = 0;

        for step in 0..3_000 {
            if next(2) == 0 {
                let id = SeqId(next_id);
                next_id += 1;
                let mut prompt = match next(4) {
                    3 => Vec::new(),
                    p => prefixes[p % prefixes.len()].clone(),
                };
                for _ in 0..1 + next(24) {
                    prompt.push(unique_token);
                    unique_token += 1;
                }
                if s.add_prompt(id, prompt, 1 + next(30)).is_ok() {
                    expected.push(id);
                }
            }
            if next(4) == 0 {
                let running = s.running_snapshot();
                if !running.is_empty() {
                    let parent = &running[next(running.len())];
                    let remaining = parent.max_new_tokens.saturating_sub(parent.generated);
                    let child = SeqId(next_id);
                    next_id += 1;
                    if remaining > 0 && s.fork_sequence(parent.id, child, remaining).is_ok() {
                        expected.push(child);
                    }
                }
            }
            s.check_invariants()
                .unwrap_or_else(|e| panic!("step {step}, before step(): {e}"));

            let o = s.step().unwrap();
            s.check_invariants()
                .unwrap_or_else(|e| panic!("step {step}, after step(): {e}"));
            assert_admissions_survive(&o);
            assert_eq!(o.prefills.len(), o.admitted.len());
            cached_tokens += o.prefills.iter().map(|p| p.cached_tokens).sum::<usize>();
            preemptions += o.preempted.len();
            finished.extend(o.finished);
        }

        let (rest, more_preemptions) = drain(&mut s);
        finished.extend(rest);
        preemptions += more_preemptions;

        assert!(preemptions > 0, "workload too light to exercise preemption");
        expected.sort_by_key(|id| id.0);
        finished.sort_by_key(|id| id.0);
        assert_eq!(
            finished, expected,
            "every request and fork finishes exactly once"
        );
        cached_tokens
    }

    #[test]
    fn randomized_load_keeps_invariants_and_never_preempts_a_fresh_admission() {
        assert_eq!(run_randomized(false), 0);
    }

    #[test]
    fn randomized_load_with_prefix_caching_keeps_invariants() {
        assert!(
            run_randomized(true) > 0,
            "shared prefixes should hit the cache"
        );
    }

    #[test]
    fn fork_shares_the_tail_block_and_the_second_writer_pays_the_cow() {
        // The fork happens while the tail still has room, so it is shared.
        let mut s = Scheduler::new(&config(16, 8), 4);
        s.add_request(SeqId(1), 3, 5).unwrap();

        let o1 = s.step().unwrap(); // admit + first decode for the parent
        let shared_block = o1.decoded[0].slot.block;

        s.fork_sequence(SeqId(1), SeqId(2), 5).unwrap();
        assert_eq!(s.num_running(), 2);

        let o2 = s.step().unwrap();
        assert_eq!(o2.decoded.len(), 2);

        let parent = o2.decoded.iter().find(|d| d.id == SeqId(1)).unwrap();
        let child = o2.decoded.iter().find(|d| d.id == SeqId(2)).unwrap();

        // Parent is processed first (it was running before the fork) and
        // finds the tail still shared with the child: it pays the copy.
        let cow = parent
            .cow
            .expect("parent must trigger CoW on the shared tail");
        assert_eq!(cow.src, shared_block);
        assert_ne!(cow.dst, shared_block);
        assert_eq!(parent.slot.block, cow.dst);

        // By the time the child writes, it is the sole remaining owner of
        // the original block — no copy needed.
        assert!(child.cow.is_none());
        assert_eq!(child.slot.block, shared_block);
    }

    #[test]
    fn fork_of_unknown_sequence_is_rejected() {
        let mut s = Scheduler::new(&config(8, 4), 2);
        let err = s.fork_sequence(SeqId(99), SeqId(100), 1).unwrap_err();
        assert_eq!(err, CacheError::UnknownSequence { id: SeqId(99) });
    }

    #[test]
    fn fork_is_rejected_when_the_batch_is_full() {
        let mut s = Scheduler::new(&config(16, 4), 2);
        s.add_request(SeqId(1), 3, 8).unwrap();
        s.step().unwrap();
        s.fork_sequence(SeqId(1), SeqId(2), 4).unwrap(); // fills the batch

        let before = s.block_snapshot();
        let err = s.fork_sequence(SeqId(1), SeqId(3), 4).unwrap_err();
        assert_eq!(err, CacheError::BatchFull { max_batch_size: 2 });
        assert_eq!(s.num_running(), 2);
        assert_eq!(
            s.block_snapshot(),
            before,
            "a rejected fork must not take references"
        );
        s.check_invariants().unwrap();
    }

    #[test]
    fn block_snapshot_reports_full_pool_and_reflects_forking() {
        let mut s = Scheduler::new(&config(8, 4), 4);
        s.add_request(SeqId(1), 3, 5).unwrap();
        s.step().unwrap(); // admits + first decode: exactly 1 block used

        let snap = s.block_snapshot();
        assert_eq!(snap.len(), 8, "must report every block, not just used ones");
        let owned: Vec<_> = snap.iter().filter(|b| b.ref_count > 0).collect();
        assert_eq!(owned.len(), 1);
        assert_eq!(owned[0].ref_count, 1);

        s.fork_sequence(SeqId(1), SeqId(2), 5).unwrap();
        let snap2 = s.block_snapshot();
        let shared_count = snap2.iter().filter(|b| b.ref_count == 2).count();
        assert_eq!(shared_count, 1, "forked block must show ref_count 2");
    }

    #[test]
    fn running_snapshot_reports_progress_and_owned_blocks() {
        let mut s = Scheduler::new(&config(16, 4), 4);
        s.add_request(SeqId(1), 3, 5).unwrap();
        s.step().unwrap();

        let running = s.running_snapshot();
        assert_eq!(running.len(), 1);
        assert_eq!(running[0].id, SeqId(1));
        assert_eq!(running[0].generated, 1);
        assert_eq!(running[0].max_new_tokens, 5);
        assert_eq!(running[0].tokens, 4, "3 prompt + 1 generated");
        assert_eq!(running[0].blocks.len(), 1);
    }

    #[test]
    fn waiting_snapshot_lists_the_queue_in_admission_order() {
        // Batch of 1: the second and third requests stay queued.
        let mut s = Scheduler::new(&config(16, 4), 1);
        s.add_request(SeqId(1), 3, 2).unwrap();
        s.add_request(SeqId(2), 5, 6).unwrap();
        s.add_request(SeqId(3), 7, 1).unwrap();
        s.step().unwrap();

        let waiting = s.waiting_snapshot();
        assert_eq!(
            waiting,
            vec![
                WaitingInfo {
                    id: SeqId(2),
                    context_len: 5,
                    max_new_tokens: 6
                },
                WaitingInfo {
                    id: SeqId(3),
                    context_len: 7,
                    max_new_tokens: 1
                },
            ]
        );
    }

    #[test]
    fn running_snapshot_is_empty_when_idle() {
        let s = Scheduler::new(&config(8, 4), 2);
        assert!(s.running_snapshot().is_empty());
        assert_eq!(
            s.block_snapshot()
                .iter()
                .filter(|b| b.ref_count > 0)
                .count(),
            0
        );
    }

    #[test]
    fn many_small_sequences_drain_cleanly_with_no_leaks() {
        let mut s = Scheduler::new(&config(6, 2), 3);
        for i in 0..10u64 {
            s.add_request(SeqId(i), 2, 2).unwrap();
        }

        let mut finished = Vec::new();
        for _ in 0..200 {
            if s.is_idle() {
                break;
            }
            let o = s.step().unwrap();
            finished.extend(o.finished);
        }

        assert!(s.is_idle(), "did not drain within the step budget");
        finished.sort_by_key(|id| id.0);
        finished.dedup();
        assert_eq!(finished.len(), 10, "every request must finish exactly once");
        assert_eq!(s.pool_utilization(), 0.0);
    }

    fn prompt(range: std::ops::Range<u32>) -> Vec<u32> {
        range.collect()
    }

    #[test]
    fn requests_with_a_common_prefix_share_its_blocks() {
        let mut s = Scheduler::with_prefix_caching(&config(16, 4), 4);
        s.add_prompt(SeqId(1), prompt(0..10), 3).unwrap();
        let o1 = s.step().unwrap();
        assert_eq!(
            o1.prefills,
            [Prefill {
                id: SeqId(1),
                cached_tokens: 0,
                computed_tokens: 10
            }]
        );

        // Same first 8 tokens (two blocks), different ending.
        s.add_prompt(SeqId(2), [prompt(0..8), prompt(50..53)].concat(), 3)
            .unwrap();
        let o2 = s.step().unwrap();
        assert_eq!(
            o2.prefills,
            [Prefill {
                id: SeqId(2),
                cached_tokens: 8,
                computed_tokens: 3
            }]
        );

        let running = s.running_snapshot();
        assert_eq!(running[0].blocks[..2], running[1].blocks[..2]);
        assert_ne!(running[0].blocks[2], running[1].blocks[2]);
        assert_eq!(running[1].cached_tokens, 8);
        let shared = s.block_snapshot();
        assert_eq!(shared[running[0].blocks[0].index()].ref_count, 2);
        assert!(shared[running[0].blocks[0].index()].cached);
        s.check_invariants().unwrap();
    }

    #[test]
    fn an_identical_prompt_still_computes_its_last_block() {
        let mut s = Scheduler::with_prefix_caching(&config(16, 4), 4);
        s.add_prompt(SeqId(1), prompt(0..8), 4).unwrap();
        s.step().unwrap();
        s.add_prompt(SeqId(2), prompt(0..8), 4).unwrap();
        let o = s.step().unwrap();
        assert_eq!(
            o.prefills,
            [Prefill {
                id: SeqId(2),
                cached_tokens: 4,
                computed_tokens: 4
            }]
        );
    }

    #[test]
    fn a_finished_request_leaves_its_prompt_cached_for_the_next() {
        let mut s = Scheduler::with_prefix_caching(&config(16, 4), 4);
        s.add_prompt(SeqId(1), prompt(0..9), 1).unwrap();
        let o = s.step().unwrap();
        assert_eq!(o.finished, [SeqId(1)]);
        assert_eq!(s.pool_utilization(), 0.0, "cached blocks count as free");
        assert_eq!(s.cache_stats().evictable_blocks, 2);

        s.add_prompt(SeqId(2), prompt(0..12), 2).unwrap();
        let o = s.step().unwrap();
        assert_eq!(o.prefills[0].cached_tokens, 8);
        assert_eq!(s.cache_stats().evictable_blocks, 0, "both reclaimed");
        assert_eq!(s.cache_stats().cached_blocks, 3, "plus the new full block");
        s.check_invariants().unwrap();
    }

    #[test]
    fn a_preempted_request_reuses_its_cached_prompt_on_readmission() {
        // A and B fit at admission, but by the second decode both tails are
        // full with one free block between them, so B is preempted. Its full
        // prompt block stays cached while it waits: A's later growth takes
        // the empty block first instead of evicting it.
        let mut s = Scheduler::with_prefix_caching(&config(5, 2), 2);
        s.add_prompt(SeqId(1), vec![1, 2, 3], 5).unwrap();
        s.add_prompt(SeqId(2), vec![7, 8, 9], 5).unwrap();

        let mut b_prefills = Vec::new();
        let mut b_preempted = false;
        for _ in 0..20 {
            if s.is_idle() {
                break;
            }
            let o = s.step().unwrap();
            s.check_invariants().unwrap();
            b_preempted |= o.preempted.contains(&SeqId(2));
            b_prefills.extend(o.prefills.into_iter().filter(|p| p.id == SeqId(2)));
        }

        assert!(b_preempted);
        assert_eq!(
            b_prefills,
            [
                Prefill {
                    id: SeqId(2),
                    cached_tokens: 0,
                    computed_tokens: 3
                },
                Prefill {
                    id: SeqId(2),
                    cached_tokens: 2,
                    computed_tokens: 2
                },
            ]
        );
        assert!(s.is_idle());
    }

    #[test]
    fn identical_prompts_share_nothing_without_prefix_caching() {
        let mut s = Scheduler::new(&config(16, 4), 4);
        s.add_prompt(SeqId(1), prompt(0..12), 4).unwrap();
        s.add_prompt(SeqId(2), prompt(0..12), 4).unwrap();
        let o = s.step().unwrap();
        assert!(o.prefills.iter().all(|p| p.cached_tokens == 0));
        assert_eq!(s.cache_stats(), CacheStats::default());
        assert!(s.block_snapshot().iter().all(|b| !b.cached));
    }
}
