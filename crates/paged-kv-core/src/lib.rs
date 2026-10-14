//! Paged KV cache management for LLM inference, after vLLM's PagedAttention.
//!
//! The cache is a pool of fixed-size blocks. Each sequence owns a
//! [`BlockTable`] that maps its token positions onto whichever blocks it
//! holds, so memory grows a block at a time instead of being reserved for the
//! longest possible output. Blocks are refcounted: forks and requests with a
//! common prompt prefix share them, and a shared block is copied only when one
//! side writes to it.
//!
//! - [`Scheduler`] batches requests over the pool: FIFO admission, one token
//!   per running sequence per step, preemption when blocks run out, and
//!   optional prefix caching.
//! - [`BlockAllocator`] and [`BlockTable`] are the bookkeeping underneath it.
//! - [`KvBackend`] is where K/V actually lives. [`CpuBackend`] stores it in
//!   host memory and is the reference the CUDA backend (`paged-kv-cuda`) is
//!   tested against.
//!
//! None of this needs a GPU, and the crate has no dependencies.
//!
//! ```
//! use paged_kv_core::{CacheConfig, Scheduler, SeqId};
//!
//! // 64 blocks of 4 tokens; at most 8 sequences decode together.
//! let mut scheduler = Scheduler::with_prefix_caching(&CacheConfig::tiny(64), 8);
//! let system_prompt: Vec<u32> = (0..12).collect();
//! for (id, question) in [(1, 100), (2, 200)] {
//!     let mut prompt = system_prompt.clone();
//!     prompt.extend([question, question + 1]);
//!     scheduler.add_prompt(SeqId(id), prompt, 16)?;
//! }
//!
//! // Both are admitted; the second reuses the first's three full prompt blocks.
//! let step = scheduler.step()?;
//! assert_eq!(step.admitted, vec![SeqId(1), SeqId(2)]);
//! assert_eq!(step.prefills[1].cached_tokens, 12);
//!
//! // Each step then hands out one slot per running sequence for its next token.
//! let step = scheduler.step()?;
//! assert_eq!(step.decoded.len(), 2);
//! # Ok::<(), paged_kv_core::CacheError>(())
//! ```

pub mod allocator;
pub mod backend;
pub mod block_table;
pub mod config;
pub mod cpu;
pub mod error;
mod lru;
pub mod prefix_cache;
pub mod scheduler;
pub mod types;

pub use allocator::{BlockAllocator, CacheStats, PrefixMatch};
pub use backend::{check_attention, AttentionSeq, KvBackend};
pub use block_table::{check_ownership, BlockTable, CowCopy};
pub use config::CacheConfig;
pub use cpu::CpuBackend;
pub use error::{CacheError, Result};
pub use prefix_cache::PrefixEntry;
pub use scheduler::{
    BlockInfo, DecodedToken, Prefill, RunningInfo, Scheduler, StepOutcome, WaitingInfo,
};
pub use types::{BlockId, PhysicalSlot, SeqId};
