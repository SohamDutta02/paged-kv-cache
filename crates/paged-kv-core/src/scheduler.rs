use std::collections::VecDeque;

use crate::allocator::BlockAllocator;
use crate::block_table::{check_ownership, BlockTable, CowCopy};
use crate::config::CacheConfig;
use crate::error::{CacheError, Result};
use crate::types::{BlockId, PhysicalSlot, SeqId};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockInfo {
    pub id: BlockId,
    pub ref_count: u32,
}

/// A point-in-time view of one running sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunningInfo {
    pub id: SeqId,
    /// Total tokens currently in the cache (prompt + generated so far).
    pub tokens: usize,
    pub generated: usize,
    pub max_new_tokens: usize,
    /// Physical blocks this sequence currently owns, in logical order.
    pub blocks: Vec<BlockId>,
}

/// A point-in-time view of one request in the waiting queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaitingInfo {
    pub id: SeqId,
    /// Tokens that will be prefilled on admission: the prompt for a new
    /// request, or the full context so far for a preempted one.
    pub context_len: usize,
    /// Tokens still to generate after admission.
    pub max_new_tokens: usize,
}

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

    /// Ownership state of every physical block, in block-id order.
    ///
    /// `O(num_blocks)` — fine for a metrics scrape or a dashboard poll every
    /// few hundred milliseconds, not something to call in a hot loop.
    pub fn block_snapshot(&self) -> Vec<BlockInfo> {
        (0..self.allocator.num_blocks())
            .map(|i| {
                let id = BlockId(i as u32);
                let ref_count = self.allocator.ref_count(id).unwrap_or(0);
                BlockInfo { id, ref_count }
            })
            .collect()
    }

    /// Progress and block ownership for every currently running sequence.
    pub fn running_snapshot(&self) -> Vec<RunningInfo> {
        self.running
            .iter()
            .map(|s| RunningInfo {
                id: s.id,
                tokens: s.table.num_tokens(),
                generated: s.generated,
                max_new_tokens: s.max_new_tokens,
                blocks: s.table.physical_blocks().to_vec(),
            })
            .collect()
    }

    /// The waiting queue in admission order (front first).
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

    pub fn add_request(
        &mut self,
        id: SeqId,
        prompt_len: usize,
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

    /// Admit waiting requests in FIFO order while they fit, keeping enough
    /// free blocks for every running sequence's next decode.
    ///
    /// Without that headroom, a prompt can take the last free blocks and the
    /// decode in this same step then has to preempt someone, which by the
    /// newest-first policy is usually the request just admitted, so its
    /// prefill is thrown away and it re-queues to try again next step. With
    /// it, nothing admitted in a step is preempted in that step; preemption
    /// only happens when sequences that were already running outgrow the
    /// pool.
    fn admit(&mut self, outcome: &mut StepOutcome) -> Result<()> {
        // Blocks the running sequences will take on their next decode: one
        // for each whose tail is full, or shared and due a copy-on-write.
        // Conservative when a parent and child share a tail (only the first
        // writer actually copies), never short.
        let mut reserved = self
            .running
            .iter()
            .map(|s| s.table.blocks_needed(&self.allocator, 1))
            .sum::<Result<usize>>()?;

        while self.running.len() < self.max_batch_size {
            let Some(pending) = self.waiting.front() else {
                break;
            };

            // The prompt's blocks plus its own first decode token.
            let mut table = BlockTable::new(self.block_size);
            let needed = table.blocks_needed(&self.allocator, pending.context_len + 1)?;
            if needed + reserved > self.allocator.num_free() {
                // Doesn't fit *right now*: leave it queued and stop trying
                // to admit further requests this step (strict FIFO).
                break;
            }

            table.append_many(&mut self.allocator, pending.context_len)?;
            reserved += table.blocks_needed(&self.allocator, 1)?;
            let pending = self
                .waiting
                .pop_front()
                .expect("front() just returned Some");
            outcome.admitted.push(pending.id);
            self.running.push(RunningSeq {
                id: pending.id,
                table,
                max_new_tokens: pending.max_new_tokens,
                generated: 0,
            });
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
    /// The child runs alongside its parent, so it counts against
    /// `max_batch_size`: forking into a full batch fails with
    /// [`CacheError::BatchFull`] and changes nothing.
    ///
    /// `child` must be a fresh id — the scheduler does not check for
    /// collisions with existing running or waiting sequences.
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
        self.running.push(RunningSeq {
            id: child,
            table: child_table,
            max_new_tokens,
            generated: 0,
        });
        Ok(())
    }

    /// Consistency check for tests and debug builds: at most
    /// `max_batch_size` sequences are running, and every block's refcount
    /// equals the number of running sequences that reference it (see
    /// [`check_ownership`]). Waiting sequences own no blocks, so a block
    /// held by no running sequence must be free.
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

    /// Steps until idle (bounded, so a bug fails instead of hanging),
    /// checking invariants and the no-same-step-preemption property after
    /// every step. Returns the finished ids in order and the preemption count.
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
        // 3 blocks of 2 tokens. Two requests with prompt=2, budget=2 each fit
        // the pool individually but can't both grow at once. Without decode
        // headroom, step 1 admitted both and seq 2 preempted itself in the
        // same step, wasting its prefill; on step 2 it was re-admitted and
        // preempted again.
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
        // 3 blocks of 2 tokens. Two one-token prompts with budget 4: both fit
        // at admission, but by the second decode both tails are full and
        // there is only one free block between them.
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

    #[test]
    fn randomized_load_keeps_invariants_and_never_preempts_a_fresh_admission() {
        // Random arrivals and forks against a small pool, checking invariants
        // after every operation, then draining: every request and fork must
        // finish exactly once with no leaked blocks. Deterministic LCG so a
        // failure reproduces exactly.
        let mut s = Scheduler::new(&config(24, 4), 6);
        let mut seed: u64 = 0xD1B5_4A32_D192_ED03;
        let mut next = |bound: usize| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as usize) % bound
        };
        let mut next_id = 0u64;
        let mut expected = Vec::new();
        let mut finished = Vec::new();
        let mut preemptions = 0;

        for step in 0..3_000 {
            if next(2) == 0 {
                let id = SeqId(next_id);
                next_id += 1;
                if s.add_request(id, 1 + next(40), 1 + next(30)).is_ok() {
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
