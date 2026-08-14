use crate::allocator::BlockAllocator;
use crate::config::CacheConfig;
use crate::error::{CacheError, Result};
use crate::types::{BlockId, PhysicalSlot};

/// Maps one sequence's logical token positions onto physical blocks.
///
/// This is the "page table" in the paging analogy: `blocks[i]` is the
/// physical block backing logical block `i` of this sequence. The table
/// itself owns no memory — every block it references is refcounted in the
/// shared [`BlockAllocator`], so a `BlockTable` is cheap to fork and cheap to
/// drop.
///
/// A `BlockTable` never talks to a [`KvBackend`](crate::KvBackend) directly.
/// It resolves *where* a token's KV entry lives; writing the actual K/V
/// values is the caller's job, via the backend. This split is what keeps
/// `BlockTable` testable with nothing but the allocator — no device, no
/// backend, no dtype.
#[derive(Debug, Clone)]
pub struct BlockTable {
    /// Physical block for each logical block index, in order.
    blocks: Vec<BlockId>,
    /// Tokens written so far. Determines which slot the next `append` lands
    /// in, and how many of `blocks`' trailing slots are actually live vs.
    /// merely reserved.
    num_tokens: usize,
    block_size: usize,
}

/// One block's worth of newly-forced-private data.
///
/// Returned by [`BlockTable::append`] when a write lands on a shared block:
/// the table transparently copies it to a fresh block before writing, and the
/// caller needs to know a copy happened so it can replicate the physical
/// bytes (host-side `memcpy`, or a `copy_blocks` kernel launch) before trusting
/// the new block's contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CowCopy {
    pub src: BlockId,
    pub dst: BlockId,
}

impl BlockTable {
    /// An empty table for a brand-new sequence. Allocates nothing yet — the
    /// first block is taken lazily on the first [`append`](Self::append).
    pub fn new(block_size: usize) -> Self {
        Self {
            blocks: Vec::new(),
            num_tokens: 0,
            block_size,
        }
    }

    /// Build a table for a config's block size — convenience for call sites
    /// that already have a [`CacheConfig`] in hand.
    pub fn for_config(config: &CacheConfig) -> Self {
        Self::new(config.block_size)
    }

    #[inline]
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Physical blocks backing this sequence, in logical order. What you hand
    /// to `paged_attention` as the block table for one sequence.
    #[inline]
    pub fn physical_blocks(&self) -> &[BlockId] {
        &self.blocks
    }

    /// Whether a tail block already exists and has room for one more token.
    ///
    /// `false` on an empty table: there's no tail block to have room *in*, so
    /// `append` must fall through to allocating the first one. The naive
    /// `num_tokens % block_size != 0` check gets this wrong at `num_tokens ==
    /// 0`, since `0 % n == 0` reads as "block is exactly full" instead of
    /// "no block exists yet" — hence the explicit `!blocks.is_empty()` guard.
    #[inline]
    fn last_block_has_room(&self) -> bool {
        !self.blocks.is_empty() && self.num_tokens % self.block_size != 0
    }

    /// Resolve a logical token position to its physical address, without
    /// requiring the position to already be written — used by `append` before
    /// the token count advances.
    fn slot_for_position(&self, position: usize) -> PhysicalSlot {
        let logical_block = position / self.block_size;
        let slot = position % self.block_size;
        PhysicalSlot::new(self.blocks[logical_block], slot)
    }

    /// Where logical token `position` lives physically.
    ///
    /// Returns an error if `position` has not been written yet — this
    /// resolves *live* addresses, the ones `paged_attention` would read.
    pub fn resolve(&self, position: usize) -> Result<PhysicalSlot> {
        if position >= self.num_tokens {
            return Err(CacheError::SlotOutOfRange {
                slot: position,
                block_size: self.num_tokens,
            });
        }
        Ok(self.slot_for_position(position))
    }

    /// Reserve space for the next token, growing the table by one physical
    /// block from `allocator` if the current tail block is full (or this is
    /// the first token).
    ///
    /// Returns the slot the caller should write into, and — if the tail block
    /// was shared with another sequence — a [`CowCopy`] describing the
    /// physical copy the caller must perform first.
    ///
    /// This is the one method where copy-on-write actually triggers: a shared
    /// tail block cannot be written in place, because doing so would corrupt
    /// every other sequence that still points at it. So instead of writing
    /// into the shared block, this method allocates a fresh private block,
    /// swaps it into the table in the shared block's place, and hands the
    /// caller a `(src, dst)` pair to copy before it writes the new token.
    ///
    /// Non-shared blocks (refcount 1, meaning we're the sole owner) are
    /// written in place — no copy, no new allocation, just advance the token
    /// count.
    pub fn append(&mut self, allocator: &mut BlockAllocator) -> Result<(PhysicalSlot, Option<CowCopy>)> {
        let mut cow = None;

        if self.last_block_has_room() {
            // Tail block exists and has a free slot. But it might be shared —
            // e.g. this table was just forked and hasn't written anything of
            // its own yet. Force it private before handing out the slot.
            let tail = *self.blocks.last().expect(
                "last_block_has_room() implies a tail block exists once num_tokens > 0",
            );
            if allocator.is_shared(tail)? {
                let fresh = allocator.allocate()?;
                allocator.decref(tail)?;
                *self.blocks.last_mut().unwrap() = fresh;
                cow = Some(CowCopy { src: tail, dst: fresh });
            }
        } else {
            // Tail is full (or this is the very first token): grow the table.
            let fresh = allocator.allocate()?;
            self.blocks.push(fresh);
        }

        let slot = self.slot_for_position(self.num_tokens);
        self.num_tokens += 1;
        Ok((slot, cow))
    }

    /// Append `n` tokens' worth of prompt in one call, returning every slot in
    /// order along with any CoW copies triggered along the way.
    ///
    /// This exists because prefill writes many tokens at once, and the
    /// natural call site (a loop over `append`) is easy to get wrong under
    /// partial failure — this method is all-or-nothing, matching
    /// [`BlockAllocator::allocate_many`]'s contract.
    pub fn append_many(
        &mut self,
        allocator: &mut BlockAllocator,
        n: usize,
    ) -> Result<(Vec<PhysicalSlot>, Vec<CowCopy>)> {
        // Snapshot state to roll back to on failure, so a prompt that doesn't
        // fully fit doesn't leave the table half-grown.
        let saved_blocks = self.blocks.clone();
        let saved_tokens = self.num_tokens;

        let mut slots = Vec::with_capacity(n);
        let mut copies = Vec::new();

        for _ in 0..n {
            match self.append(allocator) {
                Ok((slot, cow)) => {
                    slots.push(slot);
                    if let Some(c) = cow {
                        copies.push(c);
                    }
                }
                Err(e) => {
                    // Unwind: release anything this call allocated beyond
                    // what existed before, restore the saved table.
                    let new_blocks = &self.blocks[self.blocks.len().min(saved_blocks.len())..];
                    // Only truly *new* blocks (not CoW replacements already
                    // accounted for by their own decref) need releasing here.
                    // CoW replacements already decref'd their source; the
                    // fresh block they allocated is what we must give back.
                    for &b in new_blocks {
                        let _ = allocator.decref(b);
                    }
                    self.blocks = saved_blocks;
                    self.num_tokens = saved_tokens;
                    return Err(e);
                }
            }
        }

        Ok((slots, copies))
    }

    /// Fork this table for a new owner (a beam search branch, or a request
    /// sharing another's prefix cache).
    ///
    /// Incref's every physical block so the child shares them with the
    /// parent — no bytes are copied. Divergence is deferred until one side
    /// actually writes, at which point [`append`](Self::append) does the
    /// real copy for exactly the one block that changed.
    pub fn fork(&self, allocator: &mut BlockAllocator) -> Result<Self> {
        for &block in &self.blocks {
            allocator.incref(block)?;
        }
        Ok(Self {
            blocks: self.blocks.clone(),
            num_tokens: self.num_tokens,
            block_size: self.block_size,
        })
    }

    /// Release every block this table holds. Must be called explicitly rather
    /// than left to `Drop`, since releasing needs the shared allocator and
    /// `Drop` cannot take arguments — forgetting to call this leaks blocks
    /// rather than corrupting anything, but it does leak.
    pub fn free(mut self, allocator: &mut BlockAllocator) -> Result<()> {
        for block in self.blocks.drain(..) {
            allocator.decref(block)?;
        }
        self.num_tokens = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup(num_blocks: usize, block_size: usize) -> (BlockAllocator, BlockTable) {
        (BlockAllocator::new(num_blocks), BlockTable::new(block_size))
    }

    #[test]
    fn append_allocates_lazily_and_fills_one_block_before_growing() {
        let (mut a, mut t) = setup(8, 4);
        assert_eq!(t.num_blocks(), 0);

        for i in 0..4 {
            let (slot, cow) = t.append(&mut a).unwrap();
            assert!(cow.is_none());
            assert_eq!(slot.slot, i);
            assert_eq!(t.num_blocks(), 1, "should still be one block at token {i}");
        }

        // Fifth token must grow to a second block.
        let (slot, cow) = t.append(&mut a).unwrap();
        assert!(cow.is_none());
        assert_eq!(slot.slot, 0);
        assert_eq!(t.num_blocks(), 2);
        assert_eq!(t.num_tokens(), 5);
        assert_eq!(a.num_used(), 2);
    }

    #[test]
    fn resolve_matches_the_slots_append_handed_out() {
        let (mut a, mut t) = setup(8, 4);
        let mut slots = Vec::new();
        for _ in 0..10 {
            slots.push(t.append(&mut a).unwrap().0);
        }
        for (pos, expected) in slots.iter().enumerate() {
            assert_eq!(t.resolve(pos).unwrap(), *expected);
        }
    }

    #[test]
    fn resolve_rejects_unwritten_positions() {
        let (mut a, mut t) = setup(8, 4);
        t.append(&mut a).unwrap();
        assert!(t.resolve(0).is_ok());
        assert!(t.resolve(1).is_err(), "position 1 was never written");
    }

    #[test]
    fn fork_shares_blocks_without_copying() {
        let (mut a, mut t) = setup(8, 4);
        for _ in 0..6 {
            t.append(&mut a).unwrap(); // 2 blocks
        }
        assert_eq!(a.num_used(), 2);

        let child = t.fork(&mut a).unwrap();

        // Same physical blocks, refcount 2 each, no new allocations.
        assert_eq!(child.physical_blocks(), t.physical_blocks());
        assert_eq!(a.num_used(), 2, "fork must not allocate new blocks");
        for &b in t.physical_blocks() {
            assert_eq!(a.ref_count(b).unwrap(), 2);
        }

        t.free(&mut a).unwrap();
        child.free(&mut a).unwrap();
        assert_eq!(a.num_free(), 8);
    }

    #[test]
    fn writing_to_a_forked_sequence_triggers_cow_on_the_shared_tail_only() {
        let (mut a, mut t) = setup(8, 4);
        for _ in 0..3 {
            t.append(&mut a).unwrap(); // one partially-full block
        }
        let parent_block = t.physical_blocks()[0];

        let mut child = t.fork(&mut a).unwrap();
        assert_eq!(a.ref_count(parent_block).unwrap(), 2);

        // Child writes its 4th token — lands in the same (shared) block.
        let (slot, cow) = child.append(&mut a).unwrap();
        let cow = cow.expect("appending into a shared block must trigger CoW");
        assert_eq!(cow.src, parent_block);
        assert_ne!(cow.dst, parent_block, "CoW must produce a fresh block");
        assert_eq!(slot.block, cow.dst);
        assert_eq!(slot.slot, 3);

        // Parent's block must be untouched and now solely owned again.
        assert_eq!(t.physical_blocks()[0], parent_block);
        assert_eq!(a.ref_count(parent_block).unwrap(), 1);
        assert_eq!(a.ref_count(cow.dst).unwrap(), 1);

        // Parent can still append its own 4th token into the original block,
        // independent of the child.
        let (parent_slot, parent_cow) = t.append(&mut a).unwrap();
        assert!(parent_cow.is_none(), "sole owner should not trigger CoW");
        assert_eq!(parent_slot.block, parent_block);
        assert_eq!(parent_slot.slot, 3);

        t.free(&mut a).unwrap();
        child.free(&mut a).unwrap();
        assert_eq!(a.num_free(), 8);
    }

    #[test]
    fn cow_only_touches_the_tail_block_earlier_blocks_stay_shared() {
        let (mut a, mut t) = setup(16, 2); // block_size 2, so blocks fill fast
        for _ in 0..6 {
            t.append(&mut a).unwrap(); // 3 full blocks
        }
        let original_blocks = t.physical_blocks().to_vec();

        let mut child = t.fork(&mut a).unwrap();
        // Child grows a 4th block — new allocation, not a CoW copy, since the
        // tail (block 3) didn't exist yet for the parent.
        let (_, cow) = child.append(&mut a).unwrap();
        assert!(cow.is_none(), "growing onto a new block is not CoW");

        // The three original blocks remain shared (refcount 2) and identical
        // between parent and child.
        for (i, &b) in original_blocks.iter().enumerate() {
            assert_eq!(a.ref_count(b).unwrap(), 2, "block {i} should still be shared");
            assert_eq!(child.physical_blocks()[i], b);
        }

        t.free(&mut a).unwrap();
        child.free(&mut a).unwrap();
        assert_eq!(a.num_free(), 16);
    }

    #[test]
    fn append_many_matches_repeated_append() {
        let (mut a1, mut t1) = setup(16, 4);
        let (mut a2, mut t2) = setup(16, 4);

        let mut slots1 = Vec::new();
        for _ in 0..11 {
            slots1.push(t1.append(&mut a1).unwrap().0);
        }

        let (slots2, _) = t2.append_many(&mut a2, 11).unwrap();

        assert_eq!(slots1, slots2);
        assert_eq!(t1.physical_blocks().len(), t2.physical_blocks().len());
        assert_eq!(a1.num_used(), a2.num_used());
    }

    #[test]
    fn append_many_rolls_back_cleanly_on_exhaustion() {
        let (mut a, mut t) = setup(2, 4); // room for 8 tokens total
        let err = t.append_many(&mut a, 20).unwrap_err();
        assert!(matches!(err, CacheError::OutOfBlocks { .. }));

        // Table must be back to empty, allocator back to fully free.
        assert_eq!(t.num_tokens(), 0);
        assert_eq!(t.num_blocks(), 0);
        assert_eq!(a.num_free(), 2);
        a.check_invariants().unwrap();
    }

    #[test]
    fn free_releases_every_block_and_is_safe_on_an_empty_table() {
        let (mut a, mut t) = setup(4, 4);
        for _ in 0..9 {
            t.append(&mut a).unwrap();
        }
        assert_eq!(a.num_used(), 3);
        t.free(&mut a).unwrap();
        assert_eq!(a.num_free(), 4);

        let empty = BlockTable::new(4);
        empty.free(&mut a).unwrap(); // must not panic or error
        a.check_invariants().unwrap();
    }

    #[test]
    fn three_way_fork_shares_correctly_and_last_writer_pays_the_copy() {
        // Beam search with width 3: one parent forks into three beams that
        // all initially share everything, then each writes independently.
        let (mut a, mut t) = setup(16, 4);
        for _ in 0..4 {
            t.append(&mut a).unwrap(); // exactly one full block
        }
        let original = t.physical_blocks()[0];

        let mut beam_a = t.fork(&mut a).unwrap();
        let mut beam_b = t.fork(&mut a).unwrap();
        // t itself + beam_a + beam_b = 3 owners.
        assert_eq!(a.ref_count(original).unwrap(), 3);

        // Each beam grows a fresh block for its next token — none of these
        // touch the shared block 0, so no CoW yet.
        let (_, cow_a) = beam_a.append(&mut a).unwrap();
        let (_, cow_b) = beam_b.append(&mut a).unwrap();
        assert!(cow_a.is_none());
        assert!(cow_b.is_none());
        assert_eq!(a.ref_count(original).unwrap(), 3, "block 0 still shared 3 ways");

        t.free(&mut a).unwrap();
        beam_a.free(&mut a).unwrap();
        beam_b.free(&mut a).unwrap();
        assert_eq!(a.num_free(), 16);
    }
}
