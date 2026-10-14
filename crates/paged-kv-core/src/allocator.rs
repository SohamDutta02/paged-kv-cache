use crate::error::{CacheError, Result};
use crate::lru::Lru;
use crate::prefix_cache::{PrefixEntry, PrefixIndex};
use crate::types::BlockId;

type RefCount = u16;
const MAX_REFCOUNT: RefCount = RefCount::MAX;

/// A reference-counted pool of fixed-size physical blocks. Forks share blocks
/// by reference; a block is copied only when written while shared.
///
/// With prefix caching, a full block can be registered under its content.
/// When its last owner releases it, the block keeps its KV and becomes
/// evictable instead of free: a later request with the same prefix can
/// reclaim it, and allocation only evicts it once no empty block is left.
#[derive(Debug, Clone)]
pub struct BlockAllocator {
    num_blocks: usize,
    /// Empty blocks. LIFO, so recently freed (likely cache-warm) blocks are
    /// reused first.
    free_list: Vec<BlockId>,
    /// Zero means the block is free or evictable.
    ref_counts: Vec<RefCount>,
    total_allocated: u64,
    cache: Option<PrefixCache>,
}

#[derive(Debug, Clone)]
struct PrefixCache {
    index: PrefixIndex,
    /// Cached blocks with no owner, least recently released first.
    evictable: Lru,
    evictions: u64,
}

/// The longest cached prefix of a token sequence, in whole blocks.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PrefixMatch {
    pub blocks: Vec<BlockId>,
    /// The entry of the last matched block; the parent for caching the next.
    pub last_entry: PrefixEntry,
    /// How many matched blocks have no owner. Acquiring them uses up that
    /// many free blocks.
    pub evictable: usize,
}

impl Default for PrefixEntry {
    fn default() -> Self {
        Self::ROOT
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Blocks whose content is registered, owned or not.
    pub cached_blocks: usize,
    /// Cached blocks with no owner.
    pub evictable_blocks: usize,
    pub evictions: u64,
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
            cache: None,
        }
    }

    pub fn with_prefix_caching(num_blocks: usize) -> Self {
        Self {
            cache: Some(PrefixCache {
                index: PrefixIndex::new(num_blocks),
                evictable: Lru::new(num_blocks),
                evictions: 0,
            }),
            ..Self::new(num_blocks)
        }
    }

    #[inline]
    pub fn prefix_caching(&self) -> bool {
        self.cache.is_some()
    }

    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Includes evictable cached blocks.
    #[inline]
    pub fn num_free(&self) -> usize {
        self.free_list.len() + self.cache.as_ref().map_or(0, |c| c.evictable.len())
    }

    #[inline]
    pub fn num_used(&self) -> usize {
        self.num_blocks - self.num_free()
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

    /// Returns a block with refcount 1, evicting the least recently used
    /// cached block if no empty one is left.
    pub fn allocate(&mut self) -> Result<BlockId> {
        let id = match self.free_list.pop() {
            Some(id) => id,
            None => self.evict()?,
        };

        debug_assert_eq!(
            self.ref_counts[id.index()],
            0,
            "block {id} was on the free list with a nonzero refcount"
        );

        self.ref_counts[id.index()] = 1;
        self.total_allocated += 1;
        Ok(id)
    }

    fn evict(&mut self) -> Result<BlockId> {
        let out_of_blocks = CacheError::OutOfBlocks {
            requested: 1,
            available: 0,
        };
        let cache = self.cache.as_mut().ok_or(out_of_blocks.clone())?;
        let i = cache.evictable.pop_front().ok_or(out_of_blocks)?;
        let id = BlockId(i as u32);
        cache.index.remove(id);
        cache.evictions += 1;
        Ok(id)
    }

    /// All or nothing, so callers never unwind a partial allocation.
    pub fn allocate_many(&mut self, n: usize) -> Result<Vec<BlockId>> {
        if self.num_free() < n {
            return Err(CacheError::OutOfBlocks {
                requested: n,
                available: self.num_free(),
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
            match &mut self.cache {
                Some(cache) if cache.index.is_cached(id) => cache.evictable.push_back(id.index()),
                _ => self.free_list.push(id),
            }
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
    /// The longest run of whole blocks at the start of `tokens` whose content
    /// is cached. Empty without prefix caching.
    pub fn match_prefix(&self, tokens: &[u32], block_size: usize) -> PrefixMatch {
        let mut found = PrefixMatch::default();
        let Some(cache) = &self.cache else {
            return found;
        };
        for chunk in tokens.chunks_exact(block_size) {
            let Some((entry, block)) = cache.index.lookup(found.last_entry, chunk) else {
                break;
            };
            if cache.evictable.contains(block.index()) {
                found.evictable += 1;
            }
            found.blocks.push(block);
            found.last_entry = entry;
        }
        found
    }

    /// Takes a reference to a block from [`match_prefix`](Self::match_prefix),
    /// reclaiming it if it had no owner. Returns the new refcount.
    pub fn acquire_cached(&mut self, id: BlockId) -> Result<u32> {
        self.validate(id)?;
        if self.ref_counts[id.index()] > 0 {
            return self.incref(id);
        }
        match &mut self.cache {
            Some(cache) if cache.evictable.contains(id.index()) => {
                cache.evictable.remove(id.index());
                self.ref_counts[id.index()] = 1;
                Ok(1)
            }
            _ => Err(CacheError::BlockNotAllocated { id }),
        }
    }

    /// Registers the content of a full, owned block. Returns the entry for that
    /// content, which is an existing one if another block already holds it
    /// (this block then stays uncached). A no-op without prefix caching.
    pub fn cache_block(
        &mut self,
        id: BlockId,
        parent: PrefixEntry,
        tokens: &[u32],
    ) -> Result<PrefixEntry> {
        self.validate(id)?;
        if self.ref_counts[id.index()] == 0 {
            return Err(CacheError::BlockNotAllocated { id });
        }
        Ok(match &mut self.cache {
            Some(cache) => cache.index.insert(id, parent, tokens),
            None => PrefixEntry::ROOT,
        })
    }

    pub fn is_cached(&self, id: BlockId) -> bool {
        self.cache
            .as_ref()
            .is_some_and(|c| id.index() < self.num_blocks && c.index.is_cached(id))
    }

    pub fn cache_stats(&self) -> CacheStats {
        self.cache
            .as_ref()
            .map_or_else(CacheStats::default, |c| CacheStats {
                cached_blocks: c.index.len(),
                evictable_blocks: c.evictable.len(),
                evictions: c.evictions,
            })
    }

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

    /// For tests: every unowned block is either free or evictable, exactly
    /// once; evictable blocks are cached and free ones are not.
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
            if self.is_cached(id) {
                return Err(format!("block {id} is cached but on the free list"));
            }
            seen[i] = true;
        }

        if let Some(cache) = &self.cache {
            for i in cache.evictable.iter() {
                if seen[i] {
                    return Err(format!("block #{i} is both free and evictable"));
                }
                if self.ref_counts[i] != 0 {
                    return Err(format!(
                        "block #{i} is evictable but has refcount {}",
                        self.ref_counts[i]
                    ));
                }
                if !cache.index.is_cached(BlockId(i as u32)) {
                    return Err(format!("block #{i} is evictable but not cached"));
                }
                seen[i] = true;
            }
            cache.index.check()?;
        }

        for (i, (&rc, &free)) in self.ref_counts.iter().zip(&seen).enumerate() {
            if rc == 0 && !free {
                return Err(format!(
                    "block #{i} has refcount 0 but is neither free nor evictable"
                ));
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

    const BS: usize = 4;

    fn tokens(seed: u32) -> Vec<u32> {
        (0..BS as u32).map(|i| seed * 100 + i).collect()
    }

    #[test]
    fn released_cached_blocks_stay_matchable_and_count_as_free() {
        let mut a = BlockAllocator::with_prefix_caching(4);
        let b = a.allocate().unwrap();
        let entry = a.cache_block(b, PrefixEntry::ROOT, &tokens(1)).unwrap();
        a.decref(b).unwrap();

        assert_eq!(a.num_free(), 4);
        assert_eq!(a.utilization(), 0.0);
        let found = a.match_prefix(&tokens(1), BS);
        assert_eq!(found.blocks, [b]);
        assert_eq!(found.last_entry, entry);
        assert_eq!(found.evictable, 1);

        assert_eq!(a.acquire_cached(b).unwrap(), 1);
        assert_eq!(a.num_free(), 3);
        assert_eq!(
            a.acquire_cached(b).unwrap(),
            2,
            "a live block is just increfed"
        );
        a.check_invariants().unwrap();
    }

    #[test]
    fn allocation_uses_empty_blocks_before_evicting_the_least_recently_released() {
        let mut a = BlockAllocator::with_prefix_caching(4);
        let blocks: Vec<_> = (0..3).map(|_| a.allocate().unwrap()).collect();
        for (i, &b) in blocks.iter().enumerate() {
            a.cache_block(b, PrefixEntry::ROOT, &tokens(i as u32))
                .unwrap();
        }
        for &b in &blocks {
            a.decref(b).unwrap();
        }

        let empty = a.allocate().unwrap();
        assert!(!blocks.contains(&empty), "the never-used block goes first");
        assert_eq!(a.cache_stats().evictions, 0);

        assert_eq!(
            a.allocate().unwrap(),
            blocks[0],
            "oldest release is evicted"
        );
        assert!(a.match_prefix(&tokens(0), BS).blocks.is_empty());
        assert_eq!(a.match_prefix(&tokens(1), BS).blocks, [blocks[1]]);
        assert_eq!(
            a.cache_stats(),
            CacheStats {
                cached_blocks: 2,
                evictable_blocks: 2,
                evictions: 1
            }
        );
        a.check_invariants().unwrap();
    }

    #[test]
    fn matches_follow_the_chain_from_its_first_block() {
        let mut a = BlockAllocator::with_prefix_caching(8);
        let parent = a.allocate().unwrap();
        let child = a.allocate().unwrap();
        let p = a
            .cache_block(parent, PrefixEntry::ROOT, &tokens(1))
            .unwrap();
        a.cache_block(child, p, &tokens(2)).unwrap();

        let both = [tokens(1), tokens(2)].concat();
        assert_eq!(a.match_prefix(&both, BS).blocks, [parent, child]);
        assert_eq!(a.match_prefix(&tokens(1), BS).blocks, [parent]);
        let other_start = [tokens(3), tokens(2)].concat();
        assert!(a.match_prefix(&other_start, BS).blocks.is_empty());
        let partial = [tokens(1), tokens(2)[..BS - 1].to_vec()].concat();
        assert_eq!(
            a.match_prefix(&partial, BS).blocks,
            [parent],
            "only whole blocks match"
        );
    }

    #[test]
    fn duplicate_content_keeps_the_first_block() {
        let mut a = BlockAllocator::with_prefix_caching(4);
        let first = a.allocate().unwrap();
        let second = a.allocate().unwrap();
        let e1 = a.cache_block(first, PrefixEntry::ROOT, &tokens(1)).unwrap();
        let e2 = a
            .cache_block(second, PrefixEntry::ROOT, &tokens(1))
            .unwrap();

        assert_eq!(e1, e2);
        assert!(a.is_cached(first));
        assert!(!a.is_cached(second));
        a.decref(second).unwrap();
        assert_eq!(
            a.cache_stats().evictable_blocks,
            0,
            "the duplicate is simply freed"
        );
        a.check_invariants().unwrap();
    }

    #[test]
    fn an_evicted_parent_never_matches_its_old_children() {
        let mut a = BlockAllocator::with_prefix_caching(2);
        let parent = a.allocate().unwrap();
        let child = a.allocate().unwrap();
        let p = a
            .cache_block(parent, PrefixEntry::ROOT, &tokens(1))
            .unwrap();
        a.cache_block(child, p, &tokens(2)).unwrap();
        a.decref(parent).unwrap();
        a.decref(child).unwrap();

        // Evicts the parent; caching the same content again gets a new entry.
        let reused = a.allocate().unwrap();
        assert_eq!(reused, parent);
        let p2 = a
            .cache_block(reused, PrefixEntry::ROOT, &tokens(1))
            .unwrap();
        assert_ne!(p, p2);

        let both = [tokens(1), tokens(2)].concat();
        assert_eq!(
            a.match_prefix(&both, BS).blocks,
            [reused],
            "the orphaned child belongs to the old entry"
        );
        a.check_invariants().unwrap();
    }

    #[test]
    fn caching_is_inert_when_disabled() {
        let mut a = BlockAllocator::new(2);
        let b = a.allocate().unwrap();
        assert_eq!(
            a.cache_block(b, PrefixEntry::ROOT, &tokens(1)).unwrap(),
            PrefixEntry::ROOT
        );
        a.decref(b).unwrap();
        assert!(a.match_prefix(&tokens(1), BS).blocks.is_empty());
        assert_eq!(a.cache_stats(), CacheStats::default());
        assert!(a.acquire_cached(b).is_err());
    }

    #[test]
    fn churn_with_caching_preserves_invariants_and_content() {
        // Shadow model: what each cached block is supposed to hold.
        let mut a = BlockAllocator::with_prefix_caching(12);
        let mut live: Vec<BlockId> = Vec::new();
        let mut content: std::collections::HashMap<BlockId, Vec<u32>> = Default::default();
        let mut seed: u64 = 0xA076_1D64_78BD_642F;
        let mut next = |bound: u64| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) % bound) as usize
        };

        for step in 0..5_000 {
            match next(4) {
                0 => {
                    if let Ok(b) = a.allocate() {
                        content.retain(|&k, _| a.is_cached(k));
                        let t = tokens(next(6) as u32);
                        if a.match_prefix(&t, BS).blocks.is_empty() {
                            a.cache_block(b, PrefixEntry::ROOT, &t).unwrap();
                            content.insert(b, t);
                        }
                        live.push(b);
                    }
                }
                1 if !live.is_empty() => {
                    let b = live.swap_remove(next(live.len() as u64));
                    a.decref(b).unwrap();
                }
                _ => {
                    let t = tokens(next(6) as u32);
                    let found = a.match_prefix(&t, BS);
                    if let Some(&b) = found.blocks.first() {
                        assert_eq!(
                            content.get(&b),
                            Some(&t),
                            "step {step}: wrong block matched"
                        );
                        a.acquire_cached(b).unwrap();
                        live.push(b);
                    }
                }
            }
            a.check_invariants()
                .unwrap_or_else(|e| panic!("step {step}: {e}"));
        }
        for b in live {
            a.decref(b).unwrap();
        }
        assert_eq!(a.num_free(), 12);
        a.check_invariants().unwrap();
    }
}
