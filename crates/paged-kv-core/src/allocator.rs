use crate::error::{CacheError, Result};
use crate::types::BlockId;

type RefCount = u16;
const MAX_REFCOUNT: RefCount = RefCount::MAX;

/// A reference-counted pool of fixed-size physical blocks. Forks share blocks
/// by reference; a block is copied only when written while shared.
#[derive(Debug, Clone)]
pub struct BlockAllocator {
    num_blocks: usize,
    /// LIFO, so recently freed (likely cache-warm) blocks are reused first.
    free_list: Vec<BlockId>,
    /// Zero means the block is on the free list.
    ref_counts: Vec<RefCount>,
    total_allocated: u64,
}

impl BlockAllocator {
    pub fn new(num_blocks: usize) -> Self {
        // Descending, so the first `allocate` returns block 0.
        let free_list = (0..num_blocks)
            .rev()
            .map(|i| BlockId(i as u32))
            .collect::<Vec<_>>();

        Self {
            num_blocks,
            free_list,
            ref_counts: vec![0; num_blocks],
            total_allocated: 0,
        }
    }

    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    #[inline]
    pub fn num_free(&self) -> usize {
        self.free_list.len()
    }

    #[inline]
    pub fn num_used(&self) -> usize {
        self.num_blocks - self.free_list.len()
    }

    #[inline]
    pub fn utilization(&self) -> f64 {
        if self.num_blocks == 0 {
            return 0.0;
        }
        self.num_used() as f64 / self.num_blocks as f64
    }

    /// High churn at flat utilization signals fragmentation or thrashing forks.
    #[inline]
    pub fn total_allocated(&self) -> u64 {
        self.total_allocated
    }

    /// Returns a block with refcount 1.
    pub fn allocate(&mut self) -> Result<BlockId> {
        let id = self.free_list.pop().ok_or(CacheError::OutOfBlocks {
            requested: 1,
            available: 0,
        })?;

        debug_assert_eq!(
            self.ref_counts[id.index()],
            0,
            "block {id} was on the free list with a nonzero refcount"
        );

        self.ref_counts[id.index()] = 1;
        self.total_allocated += 1;
        Ok(id)
    }

    /// All or nothing, so callers never unwind a partial allocation.
    pub fn allocate_many(&mut self, n: usize) -> Result<Vec<BlockId>> {
        if self.free_list.len() < n {
            return Err(CacheError::OutOfBlocks {
                requested: n,
                available: self.free_list.len(),
            });
        }

        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.allocate()?);
        }
        Ok(out)
    }

    /// Returns the new refcount.
    pub fn incref(&mut self, id: BlockId) -> Result<u32> {
        self.validate(id)?;

        let rc = &mut self.ref_counts[id.index()];
        if *rc == 0 {
            return Err(CacheError::BlockNotAllocated { id });
        }
        if *rc == MAX_REFCOUNT {
            return Err(CacheError::RefCountOverflow {
                id,
                max: MAX_REFCOUNT as u32,
            });
        }

        *rc += 1;
        Ok(*rc as u32)
    }

    /// Returns the new refcount; the block is freed at zero.
    pub fn decref(&mut self, id: BlockId) -> Result<u32> {
        self.validate(id)?;

        let rc = &mut self.ref_counts[id.index()];
        if *rc == 0 {
            // A silent no-op would hide the bug; pushing would duplicate the id.
            return Err(CacheError::BlockNotAllocated { id });
        }

        *rc -= 1;
        let new = *rc;
        if new == 0 {
            self.free_list.push(id);
        }
        Ok(new as u32)
    }

    pub fn decref_many(&mut self, ids: &[BlockId]) -> Result<()> {
        for &id in ids {
            self.decref(id)?;
        }
        Ok(())
    }

    pub fn ref_count(&self, id: BlockId) -> Result<u32> {
        self.validate(id)?;
        Ok(self.ref_counts[id.index()] as u32)
    }

    /// The copy-on-write trigger: a shared block must be copied before a write.
    pub fn is_shared(&self, id: BlockId) -> Result<bool> {
        Ok(self.ref_count(id)? > 1)
    }

    pub fn is_allocated(&self, id: BlockId) -> Result<bool> {
        Ok(self.ref_count(id)? > 0)
    }

    #[inline]
    fn validate(&self, id: BlockId) -> Result<()> {
        if id.index() >= self.num_blocks {
            return Err(CacheError::InvalidBlock {
                id,
                num_blocks: self.num_blocks,
            });
        }
        Ok(())
    }

    /// For tests: each free block is on the free list exactly once, no live
    /// block is on it at all.
    pub fn check_invariants(&self) -> std::result::Result<(), String> {
        let mut seen = vec![false; self.num_blocks];

        for &id in &self.free_list {
            let i = id.index();
            if i >= self.num_blocks {
                return Err(format!("free list holds out-of-range block {id}"));
            }
            if seen[i] {
                return Err(format!("block {id} appears twice on the free list"));
            }
            if self.ref_counts[i] != 0 {
                return Err(format!(
                    "block {id} is on the free list but has refcount {}",
                    self.ref_counts[i]
                ));
            }
            seen[i] = true;
        }

        for (i, (&rc, &free)) in self.ref_counts.iter().zip(&seen).enumerate() {
            if rc == 0 && !free {
                return Err(format!("block #{i} has refcount 0 but is not free"));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_pool_is_entirely_free() {
        let a = BlockAllocator::new(8);
        assert_eq!(a.num_free(), 8);
        assert_eq!(a.num_used(), 0);
        assert_eq!(a.utilization(), 0.0);
        a.check_invariants().unwrap();
    }

    #[test]
    fn allocate_hands_out_distinct_blocks_with_refcount_one() {
        let mut a = BlockAllocator::new(4);
        let ids: Vec<_> = (0..4).map(|_| a.allocate().unwrap()).collect();

        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 4, "allocator handed out a duplicate");

        for id in ids {
            assert_eq!(a.ref_count(id).unwrap(), 1);
        }
        assert_eq!(a.num_free(), 0);
        assert_eq!(a.utilization(), 1.0);
        a.check_invariants().unwrap();
    }

    #[test]
    fn exhausted_pool_reports_out_of_blocks() {
        let mut a = BlockAllocator::new(2);
        a.allocate().unwrap();
        a.allocate().unwrap();
        assert_eq!(
            a.allocate().unwrap_err(),
            CacheError::OutOfBlocks {
                requested: 1,
                available: 0
            }
        );
    }

    #[test]
    fn allocate_many_is_all_or_nothing() {
        let mut a = BlockAllocator::new(4);
        let err = a.allocate_many(5).unwrap_err();
        assert_eq!(
            err,
            CacheError::OutOfBlocks {
                requested: 5,
                available: 4
            }
        );
        // The failed request must not have consumed anything.
        assert_eq!(a.num_free(), 4);
        assert_eq!(a.total_allocated(), 0);
        a.check_invariants().unwrap();
    }

    #[test]
    fn decref_to_zero_returns_block_to_the_pool() {
        let mut a = BlockAllocator::new(2);
        let id = a.allocate().unwrap();
        assert_eq!(a.num_free(), 1);

        assert_eq!(a.decref(id).unwrap(), 0);
        assert_eq!(a.num_free(), 2);
        assert!(!a.is_allocated(id).unwrap());
        a.check_invariants().unwrap();
    }

    #[test]
    fn shared_block_survives_one_owner_leaving() {
        let mut a = BlockAllocator::new(2);
        let id = a.allocate().unwrap();

        // Parent forks twice: three owners total.
        assert_eq!(a.incref(id).unwrap(), 2);
        assert_eq!(a.incref(id).unwrap(), 3);
        assert!(a.is_shared(id).unwrap());

        assert_eq!(a.decref(id).unwrap(), 2);
        assert!(a.is_shared(id).unwrap());
        assert_eq!(a.num_free(), 1, "block freed while still shared");

        assert_eq!(a.decref(id).unwrap(), 1);
        assert!(!a.is_shared(id).unwrap(), "sole owner should not be shared");
        assert!(a.is_allocated(id).unwrap());

        assert_eq!(a.decref(id).unwrap(), 0);
        assert_eq!(a.num_free(), 2);
        a.check_invariants().unwrap();
    }

    #[test]
    fn double_free_is_an_error_not_a_silent_corruption() {
        let mut a = BlockAllocator::new(2);
        let id = a.allocate().unwrap();
        a.decref(id).unwrap();

        assert_eq!(
            a.decref(id).unwrap_err(),
            CacheError::BlockNotAllocated { id }
        );
        // The free list must not contain `id` twice.
        assert_eq!(a.num_free(), 2);
        a.check_invariants().unwrap();
    }

    #[test]
    fn incref_on_a_free_block_is_rejected() {
        let mut a = BlockAllocator::new(2);
        let id = BlockId(0);
        assert_eq!(
            a.incref(id).unwrap_err(),
            CacheError::BlockNotAllocated { id }
        );
    }

    #[test]
    fn out_of_range_block_is_rejected() {
        let a = BlockAllocator::new(2);
        let id = BlockId(7);
        assert_eq!(
            a.ref_count(id).unwrap_err(),
            CacheError::InvalidBlock { id, num_blocks: 2 }
        );
    }

    #[test]
    fn free_list_is_lifo() {
        let mut a = BlockAllocator::new(4);
        let ids = a.allocate_many(4).unwrap();

        a.decref(ids[1]).unwrap();
        a.decref(ids[3]).unwrap();

        // Most recently freed comes back first.
        assert_eq!(a.allocate().unwrap(), ids[3]);
        assert_eq!(a.allocate().unwrap(), ids[1]);
        a.check_invariants().unwrap();
    }

    #[test]
    fn churn_preserves_invariants() {
        // Deterministic pseudo-random allocate/fork/release churn.
        let mut a = BlockAllocator::new(16);
        let mut live: Vec<BlockId> = Vec::new();
        let mut seed: u64 = 0x243F_6A88_85A3_08D3;

        for step in 0..2_000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let roll = (seed >> 33) % 3;

            match roll {
                0 => {
                    if let Ok(id) = a.allocate() {
                        live.push(id);
                    }
                }
                1 => {
                    if !live.is_empty() {
                        let i = (seed >> 13) as usize % live.len();
                        let id = live[i];
                        if a.incref(id).is_ok() {
                            live.push(id);
                        }
                    }
                }
                _ => {
                    if !live.is_empty() {
                        let i = (seed >> 17) as usize % live.len();
                        let id = live.swap_remove(i);
                        a.decref(id).unwrap();
                    }
                }
            }

            a.check_invariants()
                .unwrap_or_else(|e| panic!("invariant broken at step {step}: {e}"));
        }

        // Drain everything; the pool must come back whole.
        for id in live {
            a.decref(id).unwrap();
        }
        assert_eq!(a.num_free(), 16, "blocks leaked during churn");
        a.check_invariants().unwrap();
    }
}
