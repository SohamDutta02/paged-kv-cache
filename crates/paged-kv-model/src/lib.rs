//! Runs Qwen2 models with their K/V in paged-kv-core's block cache. The
//! scheduler decides where each token's K/V goes, and attention reads it back
//! through the block tables. Everything but attention is candle, and the
//! tests check the output against candle's own Qwen2 token for token.
//!
//! ```no_run
//! use candle_core::{DType, Device};
//! use candle_nn::VarBuilder;
//! use paged_kv_core::{CpuBackend, SeqId};
//! use paged_kv_model::{load_config, Engine, PagedQwen2};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let dir = std::path::Path::new("Qwen2.5-0.5B-Instruct");
//! let config = load_config(&dir.join("config.json"))?;
//! let weights = [dir.join("model.safetensors")];
//! let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, DType::F32, &Device::Cpu)? };
//! let model = PagedQwen2::new(&config, vb)?;
//!
//! // 256 blocks of 16 tokens; up to 8 sequences per step.
//! let cache = CpuBackend::new(model.cache_config(256, 16));
//! let mut engine = Engine::new(model, cache, 8, true);
//! engine.add_request(SeqId(0), vec![151644, 8948, 198], 32)?;
//! while !engine.is_idle() {
//!     for finished in engine.step()? {
//!         println!("{:?}: {:?}", finished.id, finished.tokens);
//!     }
//! }
//! # Ok(())
//! # }
//! ```

mod engine;
mod model;

pub use engine::{Engine, EngineStats, Finished};
pub use model::{load_config, Config, PagedQwen2, SeqInput};
