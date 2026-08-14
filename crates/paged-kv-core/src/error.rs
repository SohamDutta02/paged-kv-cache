use crate::types::{BlockId, SeqId};
use std::fmt;

pub type Result<T> = std::result::Result<T, CacheError>;

/// Errors surfaced by the cache layer.
///
/// Hand-written rather than derived via `thiserror` so that `paged-kv-core`
/// stays dependency-free. There are few enough variants that the boilerplate
/// is cheaper than the dependency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheError {
    /// The free list is empty. The caller must evict, preempt, or reject.
    OutOfBlocks { requested: usize, available: usize },

    /// Block id is outside `0..num_blocks`.
    InvalidBlock { id: BlockId, num_blocks: usize },

    /// Operation requires a live block but this one has refcount 0.
    BlockNotAllocated { id: BlockId },

    /// Refcount would exceed the width of the counter. In practice this means
    /// a fork storm — a beam search that never collapsed.
    RefCountOverflow { id: BlockId, max: u32 },

    /// A slot index was >= `block_size`.
    SlotOutOfRange { slot: usize, block_size: usize },

    /// A caller handed us a K or V tensor whose length does not match
    /// `num_kv_heads * head_dim`.
    ShapeMismatch { expected: usize, actual: usize },

    /// Layer index was >= `num_layers`.
    InvalidLayer { layer: usize, num_layers: usize },

    /// A scheduler operation referenced a sequence id that is neither
    /// running nor waiting.
    UnknownSequence { id: SeqId },

    /// An error surfaced by a device backend — a CUDA driver failure, an
    /// NVRTC compile error, device OOM, or similar. Carried as a message
    /// rather than a structured variant because `paged-kv-core` doesn't (and
    /// shouldn't) depend on any specific backend's error types; the backend
    /// crate is responsible for producing a message worth reading.
    Device(String),
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CacheError::OutOfBlocks {
                requested,
                available,
            } => write!(
                f,
                "out of KV blocks: requested {requested}, only {available} free"
            ),
            CacheError::InvalidBlock { id, num_blocks } => write!(
                f,
                "invalid block {id}: pool holds {num_blocks} blocks"
            ),
            CacheError::BlockNotAllocated { id } => {
                write!(f, "block {id} is not allocated (refcount 0)")
            }
            CacheError::RefCountOverflow { id, max } => {
                write!(f, "refcount overflow on block {id} (max {max})")
            }
            CacheError::SlotOutOfRange { slot, block_size } => {
                write!(f, "slot {slot} out of range for block size {block_size}")
            }
            CacheError::ShapeMismatch { expected, actual } => {
                write!(f, "shape mismatch: expected {expected} elements, got {actual}")
            }
            CacheError::InvalidLayer { layer, num_layers } => {
                write!(f, "invalid layer {layer}: model has {num_layers} layers")
            }
            CacheError::UnknownSequence { id } => {
                write!(f, "no sequence with id {id} is running or waiting")
            }
            CacheError::Device(msg) => write!(f, "device error: {msg}"),
        }
    }
}

impl std::error::Error for CacheError {}