
#![cfg_attr(not(feature = "cuda"), allow(unused_imports))]

#[cfg(feature = "cuda")]
mod backend;

#[cfg(feature = "cuda")]
pub use backend::CudaBackend;


pub const fn cuda_enabled() -> bool {
    cfg!(feature = "cuda")
}
