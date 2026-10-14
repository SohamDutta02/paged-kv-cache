use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use paged_kv_core::{
    check_attention, AttentionSeq, BlockId, CacheConfig, CacheError, KvBackend, PhysicalSlot,
    Result,
};

const KERNEL_SRC: &str = include_str!("../kernels/kernels.cu");
const THREADS_PER_BLOCK: u32 = 256;
/// Must equal ATTN_THREADS in kernels.cu.
const ATTENTION_THREADS: u32 = 128;

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
    attention_fn: CudaFunction,
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
        let attention_fn = module
            .load_function("paged_attention_kernel")
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
            attention_fn,
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

    fn paged_attention(
        &self,
        layer: usize,
        seqs: &[AttentionSeq<'_>],
        q: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        let cfg = &self.config;
        let heads = check_attention(cfg, layer, seqs, q.len(), out.len())?;
        let rows: usize = seqs.iter().map(|s| s.num_queries).sum();
        if rows == 0 {
            return Ok(());
        }

        let max_context = seqs.iter().map(|s| s.context_len).max().unwrap_or(0);
        let max_blocks = cfg.blocks_for_tokens(max_context);
        let mut tables = vec![0i32; seqs.len() * max_blocks];
        let mut query_seq = Vec::with_capacity(rows);
        let mut query_pos = Vec::with_capacity(rows);
        for (i, s) in seqs.iter().enumerate() {
            let used = cfg.blocks_for_tokens(s.context_len);
            for (dst, b) in tables[i * max_blocks..]
                .iter_mut()
                .zip(&s.block_table[..used])
            {
                *dst = b.index() as i32;
            }
            for j in 0..s.num_queries {
                query_seq.push(i as i32);
                query_pos.push((s.context_len - s.num_queries + j) as i32);
            }
        }

        let stream = self.stream.clone();
        let d_q = stream.clone_htod(q).map_err(cuda_err)?;
        let d_tables = stream.clone_htod(&tables).map_err(cuda_err)?;
        let d_seq = stream.clone_htod(&query_seq).map_err(cuda_err)?;
        let d_pos = stream.clone_htod(&query_pos).map_err(cuda_err)?;
        let mut d_out = stream.alloc_zeros::<f32>(q.len()).map_err(cuda_err)?;
        let mut d_scratch = stream
            .alloc_zeros::<f32>(rows * heads * max_context)
            .map_err(cuda_err)?;

        let max_blocks_i32 = max_blocks as i32;
        let max_context_i32 = max_context as i32;
        let layer_i32 = layer as i32;
        let num_blocks_i32 = cfg.num_blocks as i32;
        let block_size_i32 = cfg.block_size as i32;
        let kv_heads_i32 = cfg.num_kv_heads as i32;
        let heads_i32 = heads as i32;
        let head_dim_i32 = cfg.head_dim as i32;
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();

        let launch = LaunchConfig {
            grid_dim: (rows as u32, heads as u32, 1),
            block_dim: (ATTENTION_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };

        // Must match the kernel's parameter order; cudarc can't check it.
        let mut builder = stream.launch_builder(&self.attention_fn);
        builder.arg(&mut d_out);
        builder.arg(&mut d_scratch);
        builder.arg(&d_q);
        builder.arg(&self.key_cache);
        builder.arg(&self.value_cache);
        builder.arg(&d_tables);
        builder.arg(&d_seq);
        builder.arg(&d_pos);
        builder.arg(&max_blocks_i32);
        builder.arg(&max_context_i32);
        builder.arg(&layer_i32);
        builder.arg(&num_blocks_i32);
        builder.arg(&block_size_i32);
        builder.arg(&kv_heads_i32);
        builder.arg(&heads_i32);
        builder.arg(&head_dim_i32);
        builder.arg(&scale);
        unsafe { builder.launch(launch) }.map_err(cuda_err)?;

        let host = stream.clone_dtoh(&d_out).map_err(cuda_err)?;
        out.copy_from_slice(&host);
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

    struct Lcg(u64);

    impl Lcg {
        fn below(&mut self, n: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as usize % n
        }

        fn value(&mut self) -> f32 {
            self.below(1 << 20) as f32 / 1024.0 - 512.0
        }
    }

    /// Both planes in CpuBackend's flat layout, read back slot by slot.
    fn cpu_pool(cpu: &CpuBackend) -> (Vec<f32>, Vec<f32>) {
        let cfg = cpu.config();
        let n = cfg.entry_elems();
        let (mut k, mut v) = (Vec::new(), Vec::new());
        let (mut kb, mut vb) = (vec![0.0; n], vec![0.0; n]);
        for layer in 0..cfg.num_layers {
            for block in 0..cfg.num_blocks {
                for slot in 0..cfg.block_size {
                    let at = PhysicalSlot::new(BlockId(block as u32), slot);
                    cpu.read_kv(layer, at, &mut kb, &mut vb).unwrap();
                    k.extend_from_slice(&kb);
                    v.extend_from_slice(&vb);
                }
            }
        }
        (k, v)
    }

    /// Fails at the first differing element, named by its position.
    fn assert_pools_match(cpu: &CpuBackend, gpu: &CudaBackend, when: &str) {
        let cfg = cpu.config();
        let (k_cpu, v_cpu) = cpu_pool(cpu);
        let k_gpu = gpu.stream.clone_dtoh(&gpu.key_cache).unwrap();
        let v_gpu = gpu.stream.clone_dtoh(&gpu.value_cache).unwrap();
        for (plane, cpu_vals, gpu_vals) in [("K", &k_cpu, &k_gpu), ("V", &v_cpu, &v_gpu)] {
            assert_eq!(cpu_vals.len(), gpu_vals.len());
            if let Some(i) = (0..cpu_vals.len()).find(|&i| cpu_vals[i] != gpu_vals[i]) {
                let n = cfg.entry_elems();
                let token = i / n;
                panic!(
                    "{plane} differs {when}: layer {}, block {}, slot {}, element {}: cpu {} gpu {}",
                    token / (cfg.num_blocks * cfg.block_size),
                    token / cfg.block_size % cfg.num_blocks,
                    token % cfg.block_size,
                    i % n,
                    cpu_vals[i],
                    gpu_vals[i],
                );
            }
        }
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
            "paged_attention_kernel",
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

    #[test]
    fn matches_cpu_oracle_under_random_operations() {
        // 320 values per token: more than one thread block's worth, so the
        // strided loops in every kernel run more than once.
        let cfg = CacheConfig {
            num_blocks: 64,
            block_size: 16,
            num_kv_heads: 4,
            head_dim: 80,
            num_layers: 3,
            dtype_bytes: 4,
        };
        let mut cpu = CpuBackend::new(cfg.clone());
        let mut gpu = CudaBackend::new(cfg.clone()).expect("CUDA device required for this test");
        let n = cfg.entry_elems();
        let mut rng = Lcg(7);

        for step in 0..300 {
            match rng.below(3) {
                0 => {
                    let layer = rng.below(cfg.num_layers);
                    let mut dsts: Vec<PhysicalSlot> = Vec::new();
                    for _ in 0..1 + rng.below(40) {
                        let at = PhysicalSlot::new(
                            BlockId(rng.below(cfg.num_blocks) as u32),
                            rng.below(cfg.block_size),
                        );
                        if !dsts.contains(&at) {
                            dsts.push(at);
                        }
                    }
                    let k: Vec<f32> = (0..dsts.len() * n).map(|_| rng.value()).collect();
                    let v: Vec<f32> = (0..dsts.len() * n).map(|_| rng.value()).collect();
                    for (i, &at) in dsts.iter().enumerate() {
                        let range = i * n..(i + 1) * n;
                        cpu.write_kv(layer, at, &k[range.clone()], &v[range])
                            .unwrap();
                    }
                    gpu.write_kv_batch(layer, &dsts, &k, &v).unwrap();
                }
                1 => {
                    // Copy-on-write never writes a block another pair reads.
                    let mut pairs: Vec<(BlockId, BlockId)> = Vec::new();
                    for _ in 0..1 + rng.below(6) {
                        let src = BlockId(rng.below(cfg.num_blocks) as u32);
                        let dst = BlockId(rng.below(cfg.num_blocks) as u32);
                        let clashes = src == dst
                            || pairs.iter().any(|&(s, d)| d == dst || d == src || s == dst);
                        if !clashes {
                            pairs.push((src, dst));
                        }
                    }
                    cpu.copy_blocks(&pairs).unwrap();
                    gpu.copy_blocks(&pairs).unwrap();
                }
                _ => {
                    let block = BlockId(rng.below(cfg.num_blocks) as u32);
                    cpu.zero_block(block).unwrap();
                    gpu.zero_block(block).unwrap();
                }
            }
            if step % 25 == 24 {
                assert_pools_match(&cpu, &gpu, &format!("after step {step}"));
            }
        }
    }

    #[test]
    fn paged_attention_matches_cpu_oracle() {
        let cfg = CacheConfig {
            num_blocks: 128,
            block_size: 16,
            num_kv_heads: 2,
            head_dim: 64,
            num_layers: 2,
            dtype_bytes: 4,
        };
        let heads = 6;
        let n = cfg.entry_elems();
        let mut cpu = CpuBackend::new(cfg.clone());
        let mut gpu = CudaBackend::new(cfg.clone()).expect("CUDA device required for this test");
        let mut rng = Lcg(11);

        let mut free: Vec<BlockId> = (0..cfg.num_blocks as u32).map(BlockId).collect();
        for i in (1..free.len()).rev() {
            free.swap(i, rng.below(i + 1));
        }

        // (context, queries). Contexts past 128 tokens run the kernel's
        // strided loops more than once.
        let shapes = [(1, 1), (17, 1), (300, 1), (64, 64), (500, 37), (129, 2)];
        let mut tables = Vec::new();
        for &(context, _) in &shapes {
            let table: Vec<BlockId> = (0..cfg.blocks_for_tokens(context))
                .map(|_| free.pop().unwrap())
                .collect();
            let slots: Vec<PhysicalSlot> = (0..context)
                .map(|t| PhysicalSlot::new(table[t / cfg.block_size], t % cfg.block_size))
                .collect();
            let k: Vec<f32> = (0..context * n).map(|_| rng.value() / 256.0).collect();
            let v: Vec<f32> = (0..context * n).map(|_| rng.value() / 256.0).collect();
            cpu.write_kv_batch(1, &slots, &k, &v).unwrap();
            gpu.write_kv_batch(1, &slots, &k, &v).unwrap();
            tables.push(table);
        }

        let batch: Vec<AttentionSeq<'_>> = shapes
            .iter()
            .zip(&tables)
            .map(|(&(context_len, num_queries), table)| AttentionSeq {
                block_table: table,
                context_len,
                num_queries,
            })
            .collect();
        let rows: usize = shapes.iter().map(|&(_, q)| q).sum();
        let q: Vec<f32> = (0..rows * heads * cfg.head_dim)
            .map(|_| rng.value() / 256.0)
            .collect();
        let mut out_cpu = vec![0.0; q.len()];
        let mut out_gpu = vec![0.0; q.len()];
        cpu.paged_attention(1, &batch, &q, &mut out_cpu).unwrap();
        gpu.paged_attention(1, &batch, &q, &mut out_gpu).unwrap();

        // Summation order differs, so equal only to rounding.
        for (i, (&c, &g)) in out_cpu.iter().zip(&out_gpu).enumerate() {
            let per_row = heads * cfg.head_dim;
            assert!(
                (c - g).abs() <= 1e-4 * (1.0 + c.abs()),
                "query row {}, head {}, element {}: cpu {c} gpu {g}",
                i / per_row,
                i % per_row / cfg.head_dim,
                i % cfg.head_dim,
            );
        }
    }
}
