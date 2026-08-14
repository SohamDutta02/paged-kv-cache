/// Geometry of the KV cache pool.
///
/// `block_size` is measured in **tokens**, matching vLLM's convention (16 is
/// the usual default). The physical footprint of one block is therefore
/// `block_size * num_kv_heads * head_dim * dtype_bytes * 2 * num_layers`,
/// where the factor of 2 covers K and V.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheConfig {
    /// Total physical blocks in the pool.
    pub num_blocks: usize,
    /// Tokens per block.
    pub block_size: usize,
    /// KV heads (post-GQA — for Llama-3 8B this is 8, not 32).
    pub num_kv_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Transformer layers. Each layer keeps its own K and V planes.
    pub num_layers: usize,
    /// Bytes per element (4 for f32, 2 for f16/bf16, 1 for fp8).
    pub dtype_bytes: usize,
}

impl CacheConfig {
    /// Config matching Llama-3.2-1B, useful as a test and benchmark default.
    pub fn llama_3_2_1b(num_blocks: usize) -> Self {
        Self {
            num_blocks,
            block_size: 16,
            num_kv_heads: 8,
            head_dim: 64,
            num_layers: 16,
            dtype_bytes: 2,
        }
    }

    /// Small config for unit tests, where big allocations only slow things down.
    pub fn tiny(num_blocks: usize) -> Self {
        Self {
            num_blocks,
            block_size: 4,
            num_kv_heads: 2,
            head_dim: 4,
            num_layers: 2,
            dtype_bytes: 4,
        }
    }

    /// Elements in one token's K (or V) entry, across all heads of one layer.
    #[inline]
    pub fn entry_elems(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// Elements in one block's K (or V) plane for one layer.
    #[inline]
    pub fn block_elems(&self) -> usize {
        self.block_size * self.entry_elems()
    }

    /// Elements in one layer's K (or V) plane across the whole pool.
    #[inline]
    pub fn layer_elems(&self) -> usize {
        self.num_blocks * self.block_elems()
    }

    /// Total elements held by the pool, counting K and V and every layer.
    #[inline]
    pub fn total_elems(&self) -> usize {
        2 * self.num_layers * self.layer_elems()
    }

    /// Total bytes the pool occupies on device.
    #[inline]
    pub fn total_bytes(&self) -> usize {
        self.total_elems() * self.dtype_bytes
    }

    /// Bytes of KV cache consumed per token, across all layers.
    ///
    /// This is the number that decides how many concurrent sequences fit in a
    /// given amount of VRAM, so it is worth surfacing directly.
    #[inline]
    pub fn bytes_per_token(&self) -> usize {
        2 * self.num_layers * self.entry_elems() * self.dtype_bytes
    }

    /// How many blocks a sequence of `num_tokens` tokens needs.
    #[inline]
    pub fn blocks_for_tokens(&self, num_tokens: usize) -> usize {
        num_tokens.div_ceil(self.block_size)
    }

    /// Split a logical token position into (logical block index, slot).
    #[inline]
    pub fn split_position(&self, position: usize) -> (usize, usize) {
        (position / self.block_size, position % self.block_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama_bytes_per_token_is_sane() {
        let cfg = CacheConfig::llama_3_2_1b(1024);
        // 2 (K+V) * 16 layers * 8 heads * 64 dim * 2 bytes = 32768 B = 32 KiB.
        assert_eq!(cfg.bytes_per_token(), 32 * 1024);
    }

    #[test]
    fn blocks_for_tokens_rounds_up() {
        let cfg = CacheConfig::tiny(64); // block_size 4
        assert_eq!(cfg.blocks_for_tokens(0), 0);
        assert_eq!(cfg.blocks_for_tokens(1), 1);
        assert_eq!(cfg.blocks_for_tokens(4), 1);
        assert_eq!(cfg.blocks_for_tokens(5), 2);
    }

    #[test]
    fn split_position_matches_block_size() {
        let cfg = CacheConfig::tiny(64);
        assert_eq!(cfg.split_position(0), (0, 0));
        assert_eq!(cfg.split_position(3), (0, 3));
        assert_eq!(cfg.split_position(4), (1, 0));
        assert_eq!(cfg.split_position(9), (2, 1));
    }

    #[test]
    fn total_bytes_is_consistent_with_per_block_math() {
        let cfg = CacheConfig::llama_3_2_1b(1000);
        let per_block_all_layers =
            2 * cfg.num_layers * cfg.block_elems() * cfg.dtype_bytes;
        assert_eq!(cfg.total_bytes(), per_block_all_layers * cfg.num_blocks);
        // A block holds 16 tokens, so it should equal 16 * bytes_per_token.
        assert_eq!(per_block_all_layers, 16 * cfg.bytes_per_token());
    }
}
