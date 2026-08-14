//! CUDA backend for the paged KV cache.
//!
//! Gated behind the `cuda` feature. Without it this crate compiles to nothing,
//! which is what lets the workspace build and its tests pass on a laptop with
//! no NVIDIA hardware.
//!
//! The kernels this crate will wrap, and their `KvBackend` counterparts:
//!
//! | Kernel              | Trait method                                     |
//! |---------------------|--------------------------------------------------|
//! | `reshape_and_cache` | [`write_kv`](paged_kv_core::KvBackend::write_kv) |
//! | `copy_blocks`       | [`copy_block`](paged_kv_core::KvBackend::copy_block) |
//! | `paged_attention`   | (not yet on the trait — see `backend.rs`)        |
//!
//! Note that `nvcc` compiles happily on a machine with no GPU. Kernel
//! authoring, compilation, and PTX inspection are all local work; only
//! execution and benchmarking need rented hardware.

#![cfg_attr(not(feature = "cuda"), allow(unused_imports))]

#[cfg(feature = "cuda")]
mod backend;

#[cfg(feature = "cuda")]
pub use backend::CudaBackend;

/// Whether this build can actually talk to a GPU.
pub const fn cuda_enabled() -> bool {
    cfg!(feature = "cuda")
}
