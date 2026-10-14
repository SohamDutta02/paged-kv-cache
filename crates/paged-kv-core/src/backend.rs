use crate::config::CacheConfig;
use crate::error::{CacheError, Result};
use crate::types::{BlockId, PhysicalSlot};

/// Physical KV storage. Each method maps to one CUDA kernel, so
/// [`CpuBackend`](crate::CpuBackend) can act as an oracle: any difference
/// between it and the CUDA backend is a kernel bug.
///
/// `paged_attention` is deliberately absent until its signature (score dtype,
/// chunked-prefill softmax state) is decided.
pub trait KvBackend {
    fn config(&self) -> &CacheConfig;

    /// `k` and `v` each hold `config().entry_elems()` values, head-major.
    fn write_kv(&mut self, layer: usize, dst: PhysicalSlot, k: &[f32], v: &[f32]) -> Result<()>;

    /// Same as calling [`write_kv`](Self::write_kv) per token, but one kernel
    /// launch on device. `k` and `v` hold `dsts.len() * entry_elems()` values,
    /// in `dsts` order.
    fn write_kv_batch(
        &mut self,
        layer: usize,
        dsts: &[PhysicalSlot],
        k: &[f32],
        v: &[f32],
    ) -> Result<()> {
        let n = self.config().entry_elems();
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
        for (i, &dst) in dsts.iter().enumerate() {
            self.write_kv(layer, dst, &k[i * n..(i + 1) * n], &v[i * n..(i + 1) * n])?;
        }
        Ok(())
    }

    /// For tests: lets copy-on-write be checked without a GPU.
    fn read_kv(
        &self,
        layer: usize,
        src: PhysicalSlot,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<()>;

    /// Copies a block across every layer and both K and V planes.
    fn copy_block(&mut self, src: BlockId, dst: BlockId) -> Result<()>;

    /// One kernel launch on device; launch overhead dominates small CoW copies.
    fn copy_blocks(&mut self, pairs: &[(BlockId, BlockId)]) -> Result<()> {
        for &(src, dst) in pairs {
            self.copy_block(src, dst)?;
        }
        Ok(())
    }

    /// Not needed for correctness; makes stale data obvious when debugging.
    fn zero_block(&mut self, block: BlockId) -> Result<()>;

    fn allocated_bytes(&self) -> usize {
        self.config().total_bytes()
    }

    fn device_name(&self) -> String;
}
