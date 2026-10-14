use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use paged_kv_core::{BlockId, CacheConfig, CacheError, KvBackend, PhysicalSlot, Result};

const KERNEL_SRC: &str = include_str!("../kernels/kernels.cu");
const THREADS_PER_BLOCK: u32 = 256;

/// Owns the whole device pool for a [`CacheConfig`].
pub struct CudaBackend {
    config: CacheConfig,
    // Keeps the context alive for the backend's lifetime.
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

        // Same flat layout as CpuBackend, so both address identically.
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

    /// Must match `CpuBackend::slot_offset` for the oracle tests to hold.
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
            return Err(CacheError::ShapeMismatch {
                expected,
                actual: k.len(),
            });
        }
        if v.len() != expected {
            return Err(CacheError::ShapeMismatch {
                expected,
                actual: v.len(),
            });
        }
        if dsts.is_empty() {
            return Ok(()); // avoid a zero-size kernel launch
        }
        for &dst in dsts {
            self.validate(layer, dst)?;
        }

        // vLLM's slot_mapping convention: block * block_size + slot.
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

        // Must match the kernel's parameter order; cudarc can't check it.
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
            return Err(CacheError::ShapeMismatch {
                expected: n,
                actual: k_out.len(),
            });
        }
        if v_out.len() != n {
            return Err(CacheError::ShapeMismatch {
                expected: n,
                actual: v_out.len(),
            });
        }

        let offset = self.offset(layer, src);

        // Test-only path, so copying the whole plane to host is acceptable.
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

        // One thread block per (pair, layer); see copy_blocks_kernel.
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
        // TODO: report the device name once this has run on hardware.
        format!(
            "cuda:0 ({} MiB pool)",
            self.allocated_bytes() / (1024 * 1024)
        )
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use paged_kv_core::CpuBackend;

    fn config() -> CacheConfig {
        CacheConfig::tiny(8)
    }

    /// Needs NVRTC but no GPU, so CI runs it.
    #[test]
    fn kernels_compile_with_nvrtc() {
        let ptx = compile_ptx(KERNEL_SRC).expect("NVRTC failed to compile kernels.cu");
        let src = ptx.to_src();
        for name in [
            "reshape_and_cache_kernel",
            "copy_blocks_kernel",
            "zero_block_kernel",
        ] {
            assert!(
                src.contains(&format!(".entry {name}(")),
                "kernels.cu does not export `{name}`"
            );
        }
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
