
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
pub mod scheduler;
pub mod types;

pub use allocator::BlockAllocator;
pub use backend::KvBackend;
pub use block_table::{BlockTable, CowCopy};
pub use config::CacheConfig;
pub use cpu::CpuBackend;
pub use error::{CacheError, Result};
pub use scheduler::{BlockInfo, DecodedToken, RunningInfo, Scheduler, StepOutcome};
pub use types::{BlockId, PhysicalSlot, SeqId};
