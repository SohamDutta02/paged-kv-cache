//! # paged-kv-core
//!
//! Device-agnostic memory management for a paged KV cache, in the style of
//! vLLM's PagedAttention.
//!
//! The central idea is the same one an OS uses for virtual memory: a sequence's
//! KV cache is *logically* contiguous but *physically* scattered across
//! fixed-size blocks. A per-sequence block table maps logical block index to
//! physical block id. This removes the need to pre-reserve a contiguous slab
//! for each sequence's maximum length, which is where naive KV caching wastes
//! the overwhelming majority of its memory.
//!
//! ## Why this crate has no dependencies
//!
//! Everything here is bookkeeping: free lists, reference counts, index
//! arithmetic. None of it needs a GPU. Keeping the crate dependency-free means
//! the interesting logic — allocation policy, copy-on-write, eviction — is
//! fully testable on a laptop with no NVIDIA hardware and no CUDA toolkit.
//! Device-specific work sits behind the [`KvBackend`] trait, implemented by
//! [`CpuBackend`] here and by `paged-kv-cuda` for real hardware.

pub mod allocator;
pub mod backend;
pub mod block_table;
pub mod config;
pub mod cpu;
pub mod error;
pub mod types;

pub use allocator::BlockAllocator;
pub use backend::KvBackend;
pub use block_table::{BlockTable, CowCopy};
pub use config::CacheConfig;
pub use cpu::CpuBackend;
pub use error::{CacheError, Result};
pub use types::{BlockId, PhysicalSlot, SeqId};
