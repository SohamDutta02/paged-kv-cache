//! CUDA backend for the paged KV cache.
//!
//! Gated behind the `cuda` feature. Without it this crate compiles to nothing,
//! which is what lets the workspace build and its tests pass on a laptop with
//! no NVIDIA hardware.
//!
//! The kernels this crate wraps, and their `KvBackend` counterparts:
//!
//! | Kernel              | Trait method                                                     |
//! |---------------------|-------------------------------------------------------------------|
//! | `reshape_and_cache`  | [`write_kv_batch`](paged_kv_core::KvBackend::write_kv_batch) — `write_kv` delegates to it as a batch of one |
//! | `copy_blocks`        | [`copy_blocks`](paged_kv_core::KvBackend::copy_blocks) — `copy_block` delegates to it as a batch of one |
//! | `paged_attention`    | (not yet on the trait — see `backend.rs`'s `KvBackend` doc)      |
//!
//! Note that NVRTC (used here, see `backend.rs`) compiles CUDA source at
//! *runtime*, not build time — so this crate never needs `nvcc` or a CUDA
//! toolchain just to `cargo build`. Only actually *running* it with the
//! `cuda` feature, on a machine with an NVIDIA driver, does.
//!
//! Unlike the rest of this project, `backend.rs` has not been checked by an
//! actual compiler — see its module doc for exactly what is and isn't
//! verified, and why.

#![cfg_attr(not(feature = "cuda"), allow(unused_imports))]

#[cfg(feature = "cuda")]
mod backend;

#[cfg(feature = "cuda")]
pub use backend::CudaBackend;

/// Whether this build can actually talk to a GPU.
pub const fn cuda_enabled() -> bool {
    cfg!(feature = "cuda")
}