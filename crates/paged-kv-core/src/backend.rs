use crate::config::CacheConfig;
use crate::error::{CacheError, Result};
use crate::types::{BlockId, PhysicalSlot};

/// Physical KV storage. Each method maps to one CUDA kernel, so
/// [`CpuBackend`](crate::CpuBackend) can act as an oracle: any difference
/// between it and the CUDA backend is a kernel bug.
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

    /// Causal attention that reads K and V through each sequence's block
    /// table. `q` holds one row of `num_heads * head_dim` values per query,
    /// sequences in order, and `out` receives the same shape. `num_heads` may
    /// be any multiple of `num_kv_heads` (grouped-query attention). Every
    /// query's own K/V must already be in the cache. Scores are scaled by
    /// `1 / sqrt(head_dim)`.
    fn paged_attention(
        &self,
        layer: usize,
        seqs: &[AttentionSeq<'_>],
        q: &[f32],
        out: &mut [f32],
    ) -> Result<()>;

    fn allocated_bytes(&self) -> usize {
        self.config().total_bytes()
    }

    fn device_name(&self) -> String;
}

/// One sequence's share of a [`KvBackend::paged_attention`] call.
#[derive(Debug, Clone, Copy)]
pub struct AttentionSeq<'a> {
    /// Physical blocks in logical order.
    pub block_table: &'a [BlockId],
    /// Tokens in the cache, including the ones being queried.
    pub context_len: usize,
    /// The queries are the last `num_queries` tokens of the context: one for
    /// decode, the uncached part of the prompt for prefill.
    pub num_queries: usize,
}

/// Validates a [`KvBackend::paged_attention`] call; returns `num_heads`.
pub fn check_attention(
    config: &CacheConfig,
    layer: usize,
    seqs: &[AttentionSeq<'_>],
    q_len: usize,
    out_len: usize,
) -> Result<usize> {
    if layer >= config.num_layers {
        return Err(CacheError::InvalidLayer {
            layer,
            num_layers: config.num_layers,
        });
    }
    for s in seqs {
        if s.num_queries > s.context_len {
            return Err(CacheError::ShapeMismatch {
                expected: s.context_len,
                actual: s.num_queries,
            });
        }
        let needed = config.blocks_for_tokens(s.context_len);
        let Some(blocks) = s.block_table.get(..needed) else {
            return Err(CacheError::ShapeMismatch {
                expected: needed,
                actual: s.block_table.len(),
            });
        };
        if let Some(&id) = blocks.iter().find(|b| b.index() >= config.num_blocks) {
            return Err(CacheError::InvalidBlock {
                id,
                num_blocks: config.num_blocks,
            });
        }
    }
    if out_len != q_len {
        return Err(CacheError::ShapeMismatch {
            expected: q_len,
            actual: out_len,
        });
    }
    let rows: usize = seqs.iter().map(|s| s.num_queries).sum();
    if rows == 0 && q_len == 0 {
        return Ok(config.num_kv_heads);
    }
    // A row must hold a whole number of KV-head groups.
    let group = config.entry_elems();
    if rows == 0 || q_len % rows != 0 || (q_len / rows) % group != 0 || q_len == 0 {
        return Err(CacheError::ShapeMismatch {
            expected: rows * group,
            actual: q_len,
        });
    }
    Ok(q_len / rows / config.head_dim)
}
