use std::collections::VecDeque;

use crate::allocator::BlockAllocator;
use crate::block_table::{BlockTable, CowCopy};
use crate::config::CacheConfig;
use crate::error::{CacheError, Result};
use crate::types::{PhysicalSlot, SeqId};

/// A request that has not yet been given any physical blocks.
///
/// `context_len` is deliberately not called `prompt_len`: after a
/// preemption, a request re-enters this queue representing a *recomputed*
/// context (original prompt plus whatever had already been generated), not a
/// fresh prompt. The scheduler and allocator don't need to know the
/// difference — both are just N tokens of context that need slots before
/// generation can continue.
struct PendingSeq {
    id: SeqId,
    context_len: usize,
    max_new_tokens: usize,
}

/// A request that currently owns physical blocks and is being decoded.
struct RunningSeq {
    id: SeqId,
    table: BlockTable,
    max_new_tokens: usize,
    generated: usize,
}

/// One token written during a [`Scheduler::step`] call.
///
/// `cow` mirrors [`BlockTable::append`]'s return value: `Some` means this
/// write landed on a block that was shared with another sequence, and the
/// caller must physically replicate `cow.src`'s bytes into `cow.dst` (a
/// `copy_blocks` kernel launch, or a host `memcpy`) before trusting anything
/// written to `slot`.
#[derive(Debug, Clone)]
pub struct DecodedToken {
    pub id: SeqId,
    pub slot: PhysicalSlot,
    pub cow: Option<CowCopy>,
}

/// Everything that happened during one [`Scheduler::step`] call.
#[derive(Debug, Clone, Default)]
pub struct StepOutcome {
    /// Sequences newly moved from waiting to running this step.
    pub admitted: Vec<SeqId>,
    /// One entry per sequence that successfully advanced by a token.
    pub decoded: Vec<DecodedToken>,
    /// Sequences forced back to the waiting queue for lack of blocks.
    /// Any decode already recorded for a preempted sequence *this same step*
    /// is removed from `decoded` — that token's block was reclaimed as part
    /// of the preemption, so it never happened as far as the caller is
    /// concerned.
    pub preempted: Vec<SeqId>,
    /// Sequences that reached their token budget and were freed this step.
    pub finished: Vec<SeqId>,
}

/// Continuous-batching scheduler over a shared [`BlockAllocator`].
///
/// This owns the admission, decoding, and eviction policy — the same role
/// vLLM's scheduler plays — but stays synchronous and dependency-free like
/// the rest of `paged-kv-core`. It never touches a [`KvBackend`](crate::KvBackend):
/// `step` returns *where* each token landed and whether a CoW copy is owed,
/// and leaves actually moving bytes (host `memcpy` or a CUDA launch) to the
/// caller. A future `paged-kv-server` wraps this in a tokio loop and wires
/// its output to a real backend; none of that plumbing needs to exist for
/// this scheduler's logic to be fully testable today.
///
/// ## Preemption policy
///
/// When a running sequence can't get the block it needs, the scheduler frees
/// the **most recently admitted** running sequence — which may be the very
/// sequence that just failed, if it is itself the newest. This is vLLM's
/// default "recompute" policy: sacrifice the arrival with the least invested
/// work, rather than the one that has made the most progress. A preempted
/// sequence doesn't lose its place in line — it's requeued at the *front* of
/// the waiting queue, representing its context up to that point, and resumes
/// as soon as blocks free up.
///
/// ## Admission ordering
///
/// Admission is strict FIFO: if the request at the front of the waiting
/// queue doesn't currently fit, the scheduler does not skip ahead to try
/// smaller requests behind it, even if those would fit. This avoids starving
/// large requests indefinitely behind a stream of small ones, at the cost of
/// occasionally leaving blocks idle that a smaller request could have used.
pub struct Scheduler {
    allocator: BlockAllocator,
    block_size: usize,
    max_batch_size: usize,
    waiting: VecDeque<PendingSeq>,
    running: Vec<RunningSeq>,
}

impl Scheduler {
    /// Build a scheduler over a fresh pool sized by `cache_config`, admitting
    /// at most `max_batch_size` sequences concurrently.
    pub fn new(cache_config: &CacheConfig, max_batch_size: usize) -> Self {
        Self {
            allocator: BlockAllocator::new(cache_config.num_blocks),
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

    /// Queue a new request. Rejected immediately, before ever touching the
    /// waiting queue, if `prompt_len + max_new_tokens` could not fit in the
    /// pool even with every other sequence evicted — such a request could
    /// never complete no matter how the scheduler juggles everyone else, so
    /// there's no reason to let it occupy a queue slot only to stall forever.
    pub fn add_request(&mut self, id: SeqId, prompt_len: usize, max_new_tokens: usize) -> Result<()> {
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
        });
        Ok(())
    }

    /// Run one scheduling iteration: admit what fits, advance every running
    /// sequence by one token, preempt whoever must yield to make that
    /// possible, and free anyone who just finished.
    pub fn step(&mut self) -> Result<StepOutcome> {
        let mut outcome = StepOutcome::default();

        self.admit(&mut outcome)?;
        self.decode(&mut outcome)?;
        self.retire_finished(&mut outcome)?;

        Ok(outcome)
    }

    fn admit(&mut self, outcome: &mut StepOutcome) -> Result<()> {
        while self.running.len() < self.max_batch_size {
            let Some(pending) = self.waiting.front() else {
                break;
            };

            let mut table = BlockTable::new(self.block_size);
            match table.append_many(&mut self.allocator, pending.context_len) {
                Ok(_) => {
                    let pending = self.waiting.pop_front().expect("front() just returned Some");
                    outcome.admitted.push(pending.id);
                    self.running.push(RunningSeq {
                        id: pending.id,
                        table,
                        max_new_tokens: pending.max_new_tokens,
                        generated: 0,
                    });
                }
                // Doesn't fit *right now* — leave it queued and stop trying
                // to admit further requests this step (see FIFO note above).
                Err(CacheError::OutOfBlocks { .. }) => break,
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    fn decode(&mut self, outcome: &mut StepOutcome) -> Result<()> {
        // Snapshot who's running before this phase starts. Sequences
        // admitted this same step are included and do get their first
        // decode step now; sequences preempted partway through this loop
        // (as someone else's victim) are skipped when their turn comes, via
        // the id lookup below coming back empty.
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
                        // Preempt the newest arrival — which may be `id`
                        // itself, if it's the newest. Either way, retry
                        // whichever sequence we were originally advancing;
                        // if it turns out that *was* the victim, there's
                        // nothing left to retry.
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

        // Any decode recorded earlier this step for the victim is void: the
        // block it landed in was just reclaimed.
        outcome.decoded.retain(|d| d.id != seq.id);
        outcome.preempted.push(seq.id);

        let context_len = seq.table.num_tokens();
        let remaining_new_tokens = seq.max_new_tokens - seq.generated;
        seq.table.free(&mut self.allocator)?;

        self.waiting.push_front(PendingSeq {
            id: seq.id,
            context_len,
            max_new_tokens: remaining_new_tokens,
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

    /// Fork a *currently running* sequence into a new one sharing its blocks
    /// — the beam-search / prefix-sharing entry point. Costs zero block
    /// copies up front: every block the parent owns gets an extra reference,
    /// and [`step`](Self::step) will transparently pay the copy for exactly
    /// the one block that diverges, the first time either side writes to a
    /// still-shared tail.
    ///
    /// `child` must be a fresh id — the scheduler does not check for
    /// collisions with existing running or waiting sequences.
    pub fn fork_sequence(&mut self, parent: SeqId, child: SeqId, max_new_tokens: usize) -> Result<()> {
        let idx = self
            .running
            .iter()
            .position(|s| s.id == parent)
            .ok_or(CacheError::UnknownSequence { id: parent })?;

        let child_table = self.running[idx].table.fork(&mut self.allocator)?;
        self.running.push(RunningSeq {
            id: child,
            table: child_table,
            max_new_tokens,
            generated: 0,
        });
        Ok(())
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
        assert_eq!(o1.decoded[0].slot, PhysicalSlot::new(o1.decoded[0].slot.block, 3));
        assert!(o1.decoded[0].cow.is_none());
        assert!(o1.finished.is_empty());
        assert_eq!(s.num_running(), 1);

        let o2 = s.step().unwrap();
        assert!(o2.admitted.is_empty());
        assert_eq!(o2.decoded.len(), 1);
        assert_eq!(o2.finished, vec![SeqId(1)]);

        assert!(s.is_idle());
        assert_eq!(s.pool_utilization(), 0.0, "finished sequence must release every block");
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
        assert!(o2.admitted.is_empty(), "batch still full, seq 1 not done yet");
        assert_eq!(s.num_waiting(), 1);

        let o3 = s.step().unwrap();
        assert_eq!(o3.finished, vec![SeqId(1)], "seq 1 reaches its 3-token budget");
        assert_eq!(s.num_waiting(), 1, "seq 2 not admitted mid-step even though room just opened");

        let o4 = s.step().unwrap();
        assert_eq!(o4.admitted, vec![SeqId(2)], "now that seq 1 is gone, seq 2 gets its turn");
        assert_eq!(o4.finished, vec![SeqId(2)], "seq 2's budget is 1 token, finishes immediately");

        assert!(s.is_idle());
    }

    #[test]
    fn oversized_request_is_rejected_up_front() {
        let mut s = Scheduler::new(&config(4, 2), 4); // total capacity = 8 tokens
        let err = s.add_request(SeqId(1), 5, 5).unwrap_err(); // needs 10, pool holds 8
        assert!(matches!(err, CacheError::OutOfBlocks { .. }));
        assert_eq!(s.num_waiting(), 0, "rejected request must not sit in the queue");
    }

    #[test]
    fn pool_pressure_forces_preemption_then_both_sequences_complete() {
        // 3 blocks of 2 tokens each = 6 tokens total capacity. Two
        // sequences with prompt=2, budget=2 tokens each (4 tokens final
        // context each) both individually fit, but not with room to spare
        // for both to run concurrently at full size.
        let mut s = Scheduler::new(&config(3, 2), 2);
        s.add_request(SeqId(1), 2, 2).unwrap();
        s.add_request(SeqId(2), 2, 2).unwrap();

        // Step 1: both admitted (1 block each, 1 free block left). Decode:
        // seq 1 grows into the spare block; seq 2 then finds the pool
        // empty and, being the newest arrival, preempts itself.
        let o1 = s.step().unwrap();
        assert_eq!(o1.admitted.len(), 2);
        assert_eq!(o1.preempted, vec![SeqId(2)]);
        assert_eq!(
            o1.decoded.iter().map(|d| d.id).collect::<Vec<_>>(),
            vec![SeqId(1)],
            "seq 2's decode this round must be voided by its own preemption"
        );
        assert_eq!(s.num_running(), 1);
        assert_eq!(s.num_waiting(), 1);

        // Step 2: seq 2 is re-admitted (its context didn't grow, so it
        // costs exactly the block it had before). Seq 1 advances into its
        // now-full tail with room to spare. Seq 2 then hits the same wall
        // and preempts itself again — but seq 1 finishes this same step,
        // fully draining the pool.
        let o2 = s.step().unwrap();
        assert_eq!(o2.admitted, vec![SeqId(2)]);
        assert_eq!(o2.preempted, vec![SeqId(2)]);
        assert_eq!(o2.finished, vec![SeqId(1)]);
        assert_eq!(s.pool_utilization(), 0.0, "seq 1 finished and seq 2 was fully evicted");

        // From here there's no more contention. Run to completion with a
        // generous bound so a real bug shows up as a failed assertion, not
        // a hang.
        let mut seen_finished = Vec::new();
        for _ in 0..20 {
            if s.is_idle() {
                break;
            }
            let o = s.step().unwrap();
            seen_finished.extend(o.finished);
        }

        assert!(s.is_idle(), "scheduler failed to drain within the step budget");
        assert_eq!(seen_finished, vec![SeqId(2)], "seq 2 must complete on its own after seq 1 finishes");
        assert_eq!(s.pool_utilization(), 0.0);
    }

    #[test]
    fn fork_shares_the_tail_block_and_the_second_writer_pays_the_cow() {
        // block_size=8 leaves plenty of room in the first block for both
        // the prompt and a couple of generated tokens, so the fork happens
        // while the tail is still shareable instead of already full.
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
        let cow = parent.cow.expect("parent must trigger CoW on the shared tail");
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
    fn many_small_sequences_drain_cleanly_with_no_leaks() {
        // Broader randomized-ish smoke test: more requests than fit at once,
        // small budgets, tight pool — run until idle and confirm every
        // block comes back.
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
}