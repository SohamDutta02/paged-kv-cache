//! Runs Qwen2 models with their K/V in paged-kv-core's block cache. The
//! scheduler decides where each token's K/V goes, and attention reads it back
//! through the block tables.

mod engine;
mod model;

pub use engine::{Engine, EngineStats, Finished};
pub use model::{Config, PagedQwen2, SeqInput};
