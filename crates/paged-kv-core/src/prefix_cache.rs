//! Content index for prefix caching: maps (parent, tokens) to the block that
//! holds that full block of KV, so requests with a common prefix share blocks.

use std::collections::HashMap;

use crate::types::BlockId;

/// A cached block's place in a prefix chain. Entries are never reused, so a
/// key naming an evicted parent can't match a different chain later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PrefixEntry(u64);

impl PrefixEntry {
    /// The parent of every chain's first block.
    pub const ROOT: Self = PrefixEntry(0);
}

/// Matching compares the tokens themselves, not just their hash, so a hash
/// collision can never share the wrong KV.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Key {
    parent: PrefixEntry,
    tokens: Box<[u32]>,
}

#[derive(Debug, Clone)]
pub(crate) struct PrefixIndex {
    by_key: HashMap<Key, (PrefixEntry, BlockId)>,
    by_block: Vec<Option<(PrefixEntry, Key)>>,
    next_entry: u64,
}

impl PrefixIndex {
    pub(crate) fn new(num_blocks: usize) -> Self {
        Self {
            by_key: HashMap::new(),
            by_block: vec![None; num_blocks],
            next_entry: 1,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.by_key.len()
    }

    pub(crate) fn is_cached(&self, block: BlockId) -> bool {
        self.by_block[block.index()].is_some()
    }

    pub(crate) fn lookup(
        &self,
        parent: PrefixEntry,
        tokens: &[u32],
    ) -> Option<(PrefixEntry, BlockId)> {
        let key = Key {
            parent,
            tokens: tokens.into(),
        };
        self.by_key.get(&key).copied()
    }

    /// Returns the entry for this content: the existing one if another block
    /// already holds it, in which case `block` stays uncached.
    pub(crate) fn insert(
        &mut self,
        block: BlockId,
        parent: PrefixEntry,
        tokens: &[u32],
    ) -> PrefixEntry {
        if let Some((entry, _)) = self.by_block[block.index()] {
            return entry;
        }
        let key = Key {
            parent,
            tokens: tokens.into(),
        };
        if let Some(&(entry, _)) = self.by_key.get(&key) {
            return entry;
        }
        let entry = PrefixEntry(self.next_entry);
        self.next_entry += 1;
        self.by_key.insert(key.clone(), (entry, block));
        self.by_block[block.index()] = Some((entry, key));
        entry
    }

    pub(crate) fn remove(&mut self, block: BlockId) {
        if let Some((_, key)) = self.by_block[block.index()].take() {
            self.by_key.remove(&key);
        }
    }

    pub(crate) fn check(&self) -> Result<(), String> {
        let indexed = self.by_block.iter().filter(|e| e.is_some()).count();
        if indexed != self.by_key.len() {
            return Err(format!(
                "{indexed} blocks are cached but the index holds {} keys",
                self.by_key.len()
            ));
        }
        for (i, slot) in self.by_block.iter().enumerate() {
            let Some((entry, key)) = slot else { continue };
            match self.by_key.get(key) {
                Some(&(e, b)) if e == *entry && b.index() == i => {}
                other => {
                    return Err(format!(
                        "block #{i} is cached as {entry:?} but its key maps to {other:?}"
                    ))
                }
            }
        }
        Ok(())
    }
}
