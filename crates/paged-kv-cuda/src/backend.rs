//! [`CudaBackend`]: the real `KvBackend` implementation, backed by the
//! kernels in `kernels/kernels.cu`.
//!
//! ## Verification status
//!
//! Every other file in this project was checked by actually running
//! `cargo test`. This one wasn't — the authoring environment has no path to
//! a CUDA-capable `rustc` new enough for `cudarc` (see the crate's
//! `Cargo.toml` comment), so there was no compiler to check this code
//! against. What *is* verified:
//!
//!   - The addressing math (`offset`, and every kernel's index formula) is
//!     hand-derived from the identical formula in `paged_kv_core::CpuBackend`
//!     and is provably equivalent — see `kernels.cu`'s module doc.
//!   - The `cudarc` API calls (`CudaContext::new`, `launch_builder`,
//!     `.arg()`, `clone_htod`/`clone_dtoh`, `compile_ptx`, `load_module`,
//!     `load_function`) were confirmed against live documentation and a
//!     throwaway scratch crate, not recalled from memory.
//!
//! What isn't verified: that this exact file compiles. The likely failure
//! mode on first real build is a handful of small naming mismatches (a
//! method renamed between `cudarc` versions, an `Arc<>` wrapper that isn't
//! actually there) — mechanical fixes, not logic bugs. The oracle-comparison
//! tests at the bottom of this file are what actually prove correctness,
//! and they've never run.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

use paged_kv_core::{BlockId, CacheConfig, CacheError, KvBackend, PhysicalSlot, Result};

const KERNEL_SRC: &str = include_str!("../kernels/kernels.cu");
const THREADS_PER_BLOCK: u32 = 256;

/// CUDA-backed KV cache pool. One instance owns the entire device pool for a
/// given [`CacheConfig`] — construct once, reuse for the whole run.
pub struct CudaBackend {
    config: CacheConfig,
    // Held to keep the context alive for as long as this backend exists.
    // The module/stream/slices below likely already keep it alive via their
    // own internal references, but there's no cost to being explicit.
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    reshape_fn: CudaFunction,
    copy_blocks_fn: CudaFunction,
    zero_block_fn: CudaFunction,
    key_cache: cudarc::driver::CudaSlice<f32>,
    value_cache: cudarc::driver::CudaSlice<f32>,
}

fn cuda_err(e: impl std::fmt::Display) -> CacheError {
    CacheError::Device(e.to_string())
}

impl CudaBackend {
    /// Initialize on GPU 0: compile the kernels via NVRTC and allocate the
    /// full K/V pool up front, sized by `config`.
    pub fn new(config: CacheConfig) -> Result<Self> {
        let ctx = CudaContext::new(0).map_err(cuda_err)?;
        let stream = ctx.default_stream();

        let ptx = compile_ptx(KERNEL_SRC).map_err(cuda_err)?;
        let module = ctx.load_module(ptx).map_err(cuda_err)?;

        let reshape_fn = module
            .load_function("reshape_and_cache_kernel")
            .map_err(cuda_err)?;
        let copy_blocks_fn = module
            .load_function("copy_blocks_kernel")
            .map_err(cuda_err)?;
        let zero_block_fn = module
            .load_function("zero_block_kernel")
            .map_err(cuda_err)?;

        // One flat plane per K and V, covering every layer and block —
        // exactly mirroring CpuBackend's `k`/`v` Vec<f32> layout, so the two
        // backends address identically. See offset() below.
        let plane_len = config.num_layers * config.layer_elems();
        let key_cache = stream.alloc_zeros::<f32>(plane_len).map_err(cuda_err)?;
        let value_cache = stream.alloc_zeros::<f32>(plane_len).map_err(cuda_err)?;

        Ok(Self {
            config,
            ctx,
            stream,
            module,
            reshape_fn,
            copy_blocks_fn,
            zero_block_fn,
            key_cache,
            value_cache,
        })
    }

    /// Flat offset of one token's entry within a K or V plane. Identical
    /// formula to `CpuBackend::slot_offset` — this is what makes the two
    /// backends numerically comparable.
    fn offset(&self, layer: usize, at: PhysicalSlot) -> usize {
        (layer * self.config.num_blocks + at.block.index()) * self.config.block_elems()
            + at.slot * self.config.entry_elems()
    }

    fn validate(&self, layer: usize, at: PhysicalSlot) -> Result<()> {
        if layer >= self.config.num_layers {
            return Err(CacheError::InvalidLayer {
                layer,
                num_layers: self.config.num_layers,
            });
        }
        self.validate_block(at.block)?;
        if at.slot >= self.config.block_size {
            return Err(CacheError::SlotOutOfRange {
                slot: at.slot,
                block_size: self.config.block_size,
            });
        }
        Ok(())
    }

    fn validate_block(&self, block: BlockId) -> Result<()> {
        if block.index() >= self.config.num_blocks {
            return Err(CacheError::InvalidBlock {
                id: block,
                num_blocks: self.config.num_blocks,
            });
        }
        Ok(())
    }
}

impl KvBackend for CudaBackend {
    fn config(&self) -> &CacheConfig {
        &self.config
    }

    fn write_kv(&mut self, layer: usize, dst: PhysicalSlot, k: &[f32], v: &[f32]) -> Result<()> {
        // Single-token write is just a batch of one — same kernel, same
        // code path, no separate logic to keep in sync.
        self.write_kv_batch(layer, std::slice::from_ref(&dst), k, v)
    }

    fn write_kv_batch(
        &mut self,
        layer: usize,
        dsts: &[PhysicalSlot],
        k: &[f32],
        v: &[f32],
    ) -> Result<()> {
        let n = self.config.entry_elems();
        let expected = dsts.len() * n;
        if k.len() != expected {
            return Err(CacheError::ShapeMismatch { expected, actual: k.len() });
        }
        if v.len() != expected {
            return Err(CacheError::ShapeMismatch { expected, actual: v.len() });
        }
        if dsts.is_empty() {
            return Ok(()); // avoid a zero-size kernel launch
        }
        for &dst in dsts {
            self.validate(layer, dst)?;
        }

        // reshape_and_cache_kernel wants one combined index per token
        // (block * block_size + local_slot), matching vLLM's own
        // slot_mapping convention.
        let slot_mapping: Vec<i64> = dsts
            .iter()
            .map(|s| s.block.index() as i64 * self.config.block_size as i64 + s.slot as i64)
            .collect();

        let stream = self.stream.clone();
        let d_key = stream.clone_htod(k).map_err(cuda_err)?;
        let d_value = stream.clone_htod(v).map_err(cuda_err)?;
        let d_slots = stream.clone_htod(&slot_mapping).map_err(cuda_err)?;

        let layer_i32 = layer as i32;
        let num_blocks_i32 = self.config.num_blocks as i32;
        let block_size_i32 = self.config.block_size as i32;
        let entry_elems_i32 = n as i32;

        let cfg = LaunchConfig {
            grid_dim: (dsts.len() as u32, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        // Argument order here must exactly match reshape_and_cache_kernel's
        // parameter list in kernels.cu — cudarc does not check this at
        // compile time.
        let mut builder = stream.launch_builder(&self.reshape_fn);
        builder.arg(&d_key);
        builder.arg(&d_value);
        builder.arg(&mut self.key_cache);
        builder.arg(&mut self.value_cache);
        builder.arg(&d_slots);
        builder.arg(&layer_i32);
        builder.arg(&num_blocks_i32);
        builder.arg(&block_size_i32);
        builder.arg(&entry_elems_i32);
        unsafe { builder.launch(cfg) }.map_err(cuda_err)?;

        Ok(())
    }

    fn read_kv(
        &self,
        layer: usize,
        src: PhysicalSlot,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<()> {
        let n = self.config.entry_elems();
        self.validate(layer, src)?;
        if k_out.len() != n {
            return Err(CacheError::ShapeMismatch { expected: n, actual: k_out.len() });
        }
        if v_out.len() != n {
            return Err(CacheError::ShapeMismatch { expected: n, actual: v_out.len() });
        }

        let offset = self.offset(layer, src);

        // read_kv exists purely for tests/debugging (see the trait doc on
        // KvBackend) — a real serving path never calls it, paged_attention
        // reads the cache directly on device. So rather than risk getting
        // CudaSlice's partial-range view API exactly right with no compiler
        // to check it against, this pulls the whole plane back to host and
        // indexes into the Vec: definitely correct, at a cost that only
        // matters on a path that was never meant to be fast.
        let stream = self.stream.clone();
        let k_host: Vec<f32> = stream.clone_dtoh(&self.key_cache).map_err(cuda_err)?;
        let v_host: Vec<f32> = stream.clone_dtoh(&self.value_cache).map_err(cuda_err)?;
        k_out.copy_from_slice(&k_host[offset..offset + n]);
        v_out.copy_from_slice(&v_host[offset..offset + n]);
        Ok(())
    }

    fn copy_block(&mut self, src: BlockId, dst: BlockId) -> Result<()> {
        if src == dst {
            return Ok(());
        }
        self.copy_blocks(&[(src, dst)])
    }

    fn copy_blocks(&mut self, pairs: &[(BlockId, BlockId)]) -> Result<()> {
        if pairs.is_empty() {
            return Ok(());
        }
        for &(src, dst) in pairs {
            self.validate_block(src)?;
            self.validate_block(dst)?;
        }

        let flat: Vec<i64> = pairs
            .iter()
            .flat_map(|&(s, d)| [s.index() as i64, d.index() as i64])
            .collect();

        let stream = self.stream.clone();
        let d_pairs = stream.clone_htod(&flat).map_err(cuda_err)?;

        let num_pairs = pairs.len() as i32;
        let num_layers = self.config.num_layers as i32;
        let num_blocks = self.config.num_blocks as i32;
        let block_elems = self.config.block_elems() as i32;

        // Flattened (pair, layer) grid — see copy_blocks_kernel's doc in
        // kernels.cu for how blockIdx.x decodes back into the two.
        let grid = (pairs.len() * self.config.num_layers) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = stream.launch_builder(&self.copy_blocks_fn);
        builder.arg(&mut self.key_cache);
        builder.arg(&mut self.value_cache);
        builder.arg(&d_pairs);
        builder.arg(&num_pairs);
        builder.arg(&num_layers);
        builder.arg(&num_blocks);
        builder.arg(&block_elems);
        unsafe { builder.launch(cfg) }.map_err(cuda_err)?;

        Ok(())
    }

    fn zero_block(&mut self, block: BlockId) -> Result<()> {
        self.validate_block(block)?;

        let stream = self.stream.clone();
        let block_i32 = block.index() as i32;
        let num_layers = self.config.num_layers as i32;
        let num_blocks = self.config.num_blocks as i32;
        let block_elems = self.config.block_elems() as i32;

        let cfg = LaunchConfig {
            grid_dim: (self.config.num_layers as u32, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = stream.launch_builder(&self.zero_block_fn);
        builder.arg(&mut self.key_cache);
        builder.arg(&mut self.value_cache);
        builder.arg(&block_i32);
        builder.arg(&num_layers);
        builder.arg(&num_blocks);
        builder.arg(&block_elems);
        unsafe { builder.launch(cfg) }.map_err(cuda_err)?;

        Ok(())
    }

    fn device_name(&self) -> String {
        // A real implementation would query cudaGetDeviceProperties for the
        // actual GPU name; not worth guessing cudarc's exact accessor for
        // that without a compiler to check it against. Revisit once this
        // runs for the first time.
        format!("cuda:0 ({} MiB pool)", self.allocated_bytes() / (1024 * 1024))
    }
}

// These tests never ran — see the module doc at the top of this file. They
// exist so that the first thing you do on the rental GPU box is
// `cargo test -p paged-kv-cuda --features cuda` and get a real, specific
// answer about whether the kernels are correct, rather than hand-testing
// through a server that doesn't exist yet.
#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use paged_kv_core::CpuBackend;

    fn config() -> CacheConfig {
        CacheConfig::tiny(8)
    }

    #[test]
    fn constructs_and_reports_sane_metadata() {
        let cfg = config();
        let backend = CudaBackend::new(cfg.clone()).expect("CUDA device required for this test");
        assert_eq!(backend.allocated_bytes(), cfg.total_bytes());
        assert!(backend.device_name().starts_with("cuda:"));
    }

    #[test]
    fn matches_cpu_oracle_on_scatter_write_then_read() {
        let cfg = config();
        let mut cpu = CpuBackend::new(cfg.clone());
        let mut gpu = CudaBackend::new(cfg.clone()).expect("CUDA device required for this test");

        let n = cfg.entry_elems();
        let dsts = [
            PhysicalSlot::new(BlockId(0), 0),
            PhysicalSlot::new(BlockId(0), 1),
            PhysicalSlot::new(BlockId(3), 2),
        ];
        let k: Vec<f32> = (0..dsts.len() * n).map(|i| i as f32 * 0.5).collect();
        let v: Vec<f32> = (0..dsts.len() * n).map(|i| i as f32 * 0.25 + 1.0).collect();

        for (i, &dst) in dsts.iter().enumerate() {
            cpu.write_kv(1, dst, &k[i * n..(i + 1) * n], &v[i * n..(i + 1) * n])
                .unwrap();
        }
        gpu.write_kv_batch(1, &dsts, &k, &v).unwrap();

        let (mut k_cpu, mut v_cpu) = (vec![0.0; n], vec![0.0; n]);
        let (mut k_gpu, mut v_gpu) = (vec![0.0; n], vec![0.0; n]);
        for &dst in &dsts {
            cpu.read_kv(1, dst, &mut k_cpu, &mut v_cpu).unwrap();
            gpu.read_kv(1, dst, &mut k_gpu, &mut v_gpu).unwrap();
            assert_eq!(k_cpu, k_gpu, "K mismatch at {dst:?}");
            assert_eq!(v_cpu, v_gpu, "V mismatch at {dst:?}");
        }
    }

    #[test]
    fn matches_cpu_oracle_on_copy_block() {
        let cfg = config();
        let mut cpu = CpuBackend::new(cfg.clone());
        let mut gpu = CudaBackend::new(cfg.clone()).expect("CUDA device required for this test");
        let n = cfg.entry_elems();

        for layer in 0..cfg.num_layers {
            for slot in 0..cfg.block_size {
                let seed = (layer * 100 + slot) as f32;
                let k: Vec<f32> = (0..n).map(|i| seed + i as f32).collect();
                let v: Vec<f32> = (0..n).map(|i| seed + 0.5 + i as f32).collect();
                let at = PhysicalSlot::new(BlockId(2), slot);
                cpu.write_kv(layer, at, &k, &v).unwrap();
                gpu.write_kv(layer, at, &k, &v).unwrap();
            }
        }

        cpu.copy_block(BlockId(2), BlockId(5)).unwrap();
        gpu.copy_block(BlockId(2), BlockId(5)).unwrap();

        let (mut k_cpu, mut v_cpu) = (vec![0.0; n], vec![0.0; n]);
        let (mut k_gpu, mut v_gpu) = (vec![0.0; n], vec![0.0; n]);
        for layer in 0..cfg.num_layers {
            for slot in 0..cfg.block_size {
                let at = PhysicalSlot::new(BlockId(5), slot);
                cpu.read_kv(layer, at, &mut k_cpu, &mut v_cpu).unwrap();
                gpu.read_kv(layer, at, &mut k_gpu, &mut v_gpu).unwrap();
                assert_eq!(k_cpu, k_gpu, "K mismatch at layer {layer} slot {slot}");
                assert_eq!(v_cpu, v_gpu, "V mismatch at layer {layer} slot {slot}");
            }
        }
    }
}