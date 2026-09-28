use std::fmt;


#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId(pub u32);

impl BlockId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// Identifier for a live sequence (one decoding request, or one beam).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SeqId(pub u64);

impl fmt::Display for SeqId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "seq{}", self.0)
    }
}

/// A fully-resolved physical address for one token's KV entry: which physical
/// block, and which slot within it.
///
/// Produced by translating a logical token position through a block table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalSlot {
    pub block: BlockId,
    pub slot: usize,
}

impl PhysicalSlot {
    #[inline]
    pub fn new(block: BlockId, slot: usize) -> Self {
        Self { block, slot }
    }
}

impl fmt::Display for PhysicalSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{}]", self.block, self.slot)
    }
}
