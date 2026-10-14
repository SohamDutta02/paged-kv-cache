//! CUDA backend for `paged-kv-core`: `reshape_and_cache`, `copy_blocks`,
//! `zero_block` and `paged_attention` kernels, compiled at runtime with NVRTC
//! and launched through cudarc.
//!
//! Everything is behind the `cuda` feature, so the workspace builds without a
//! CUDA toolkit. The kernels address memory exactly like
//! `paged_kv_core::CpuBackend`, and the tests check them against it.

#![cfg_attr(not(feature = "cuda"), allow(unused_imports))]

#[cfg(feature = "cuda")]
mod backend;

#[cfg(feature = "cuda")]
pub use backend::CudaBackend;

pub const fn cuda_enabled() -> bool {
    cfg!(feature = "cuda")
}
