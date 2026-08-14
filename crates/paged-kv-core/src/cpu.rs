use crate::backend::KvBackend;
use crate::config::CacheConfig;
use crate::error::{CacheError, Result};
use crate::types::{BlockId, PhysicalSlot};

/// Host-memory implementation of [`KvBackend`].
///
/// Storage is one flat `Vec<f32>` per plane (K and V), with layers, blocks,
/// slots, and heads folded into the index. The layout mirrors what the CUDA
/// backend allocates on device, so the index arithmetic exercised here is the
/// same arithmetic the kernels will perform:
///
/// ```text
/// index = ((((layer * num_blocks + block) * block_size + slot)
///           * num_kv_heads + head) * head_dim) + d
/// ```
///
/// Block-major within a layer (rather than layer-major within a block) is the
/// vLLM layout, and it is the right one: `copy_blocks` then touches one
/// contiguous run per layer instead of striding across the whole pool.
///
/// Everything is `f32` regardless of `config.dtype_bytes`. This backend exists
/// to validate *addressing and lifetime* logic, not numerics; running it in
/// f32 keeps it a clean reference against which a half-precision kernel can be
/// compared with an explicit tolerance.
#[derive(Debug, Clone)]
pub struct CpuBackend {
    config: CacheConfig,
    k: Vec<f32>,
    v: Vec<f32>,
}

impl CpuBackend {
    pub fn new(config: CacheConfig) -> Self {
        let plane = config.num_layers * config.layer_elems();
        Self {
            k: vec![0.0; plane],
            v: vec![0.0; plane],
            config,
        }
    }

    /// Offset of the start of `block`'s data within `layer`.
    #[inline]
    fn block_offset(&self, layer: usize, block: BlockId) -> usize {
        (layer * self.config.num_blocks + block.index()) * self.config.block_elems()
    }

    /// Offset of one token entry.
    #[inline]
    fn slot_offset(&self, layer: usize, at: PhysicalSlot) -> usize {
        self.block_offset(layer, at.block) + at.slot * self.config.entry_elems()
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

    fn check_len(expected: usize, actual: usize) -> Result<()> {
        if expected != actual {
            return Err(CacheError::ShapeMismatch { expected, actual });
        }
        Ok(())
    }
}

impl KvBackend for CpuBackend {
    fn config(&self) -> &CacheConfig {
        &self.config
    }

    fn write_kv(
        &mut self,
        layer: usize,
        dst: PhysicalSlot,
        k: &[f32],
        v: &[f32],
    ) -> Result<()> {
        self.validate(layer, dst)?;
        let n = self.config.entry_elems();
        Self::check_len(n, k.len())?;
        Self::check_len(n, v.len())?;

        let off = self.slot_offset(layer, dst);
        self.k[off..off + n].copy_from_slice(k);
        self.v[off..off + n].copy_from_slice(v);
        Ok(())
    }

    fn read_kv(
        &self,
        layer: usize,
        src: PhysicalSlot,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<()> {
        self.validate(layer, src)?;
        let n = self.config.entry_elems();
        Self::check_len(n, k_out.len())?;
        Self::check_len(n, v_out.len())?;

        let off = self.slot_offset(layer, src);
        k_out.copy_from_slice(&self.k[off..off + n]);
        v_out.copy_from_slice(&self.v[off..off + n]);
        Ok(())
    }

    fn copy_block(&mut self, src: BlockId, dst: BlockId) -> Result<()> {
        self.validate_block(src)?;
        self.validate_block(dst)?;
        if src == dst {
            return Ok(());
        }

        let n = self.config.block_elems();
        for layer in 0..self.config.num_layers {
            let s = self.block_offset(layer, src);
            let d = self.block_offset(layer, dst);
            // copy_within rather than a temporary buffer: source and
            // destination live in the same allocation and never overlap,
            // since src != dst.
            self.k.copy_within(s..s + n, d);
            self.v.copy_within(s..s + n, d);
        }
        Ok(())
    }

    fn zero_block(&mut self, block: BlockId) -> Result<()> {
        self.validate_block(block)?;
        let n = self.config.block_elems();
        for layer in 0..self.config.num_layers {
            let off = self.block_offset(layer, block);
            self.k[off..off + n].fill(0.0);
            self.v[off..off + n].fill(0.0);
        }
        Ok(())
    }

    fn device_name(&self) -> String {
        "cpu (host memory reference backend)".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(seed: f32, n: usize) -> Vec<f32> {
        (0..n).map(|i| seed + i as f32 * 0.25).collect()
    }

    fn backend() -> CpuBackend {
        CpuBackend::new(CacheConfig::tiny(8))
    }

    #[test]
    fn write_then_read_round_trips() {
        let mut b = backend();
        let n = b.config().entry_elems();
        let (k, v) = (entry(1.0, n), entry(100.0, n));
        let at = PhysicalSlot::new(BlockId(3), 2);

        b.write_kv(1, at, &k, &v).unwrap();

        let (mut ko, mut vo) = (vec![0.0; n], vec![0.0; n]);
        b.read_kv(1, at, &mut ko, &mut vo).unwrap();
        assert_eq!(ko, k);
        assert_eq!(vo, v);
    }

    #[test]
    fn writes_do_not_bleed_across_slots_blocks_or_layers() {
        let mut b = backend();
        let n = b.config().entry_elems();
        let (k, v) = (entry(7.0, n), entry(70.0, n));
        let at = PhysicalSlot::new(BlockId(3), 2);
        b.write_kv(1, at, &k, &v).unwrap();

        let neighbours = [
            (1, PhysicalSlot::new(BlockId(3), 1)), // slot before
            (1, PhysicalSlot::new(BlockId(3), 3)), // slot after
            (1, PhysicalSlot::new(BlockId(2), 2)), // block before
            (1, PhysicalSlot::new(BlockId(4), 2)), // block after
            (0, PhysicalSlot::new(BlockId(3), 2)), // other layer
        ];

        let (mut ko, mut vo) = (vec![0.0; n], vec![0.0; n]);
        for (layer, slot) in neighbours {
            b.read_kv(layer, slot, &mut ko, &mut vo).unwrap();
            assert!(ko.iter().all(|&x| x == 0.0), "K bled into {layer}:{slot}");
            assert!(vo.iter().all(|&x| x == 0.0), "V bled into {layer}:{slot}");
        }
    }

    #[test]
    fn copy_block_duplicates_every_slot_and_layer() {
        let mut b = backend();
        let cfg = b.config().clone();
        let n = cfg.entry_elems();

        // Fill every slot of block 0 across all layers with distinct data.
        for layer in 0..cfg.num_layers {
            for slot in 0..cfg.block_size {
                let seed = (layer * 100 + slot) as f32;
                b.write_kv(
                    layer,
                    PhysicalSlot::new(BlockId(0), slot),
                    &entry(seed, n),
                    &entry(seed + 0.5, n),
                )
                .unwrap();
            }
        }

        b.copy_block(BlockId(0), BlockId(5)).unwrap();

        let (mut ko, mut vo) = (vec![0.0; n], vec![0.0; n]);
        for layer in 0..cfg.num_layers {
            for slot in 0..cfg.block_size {
                let seed = (layer * 100 + slot) as f32;
                b.read_kv(layer, PhysicalSlot::new(BlockId(5), slot), &mut ko, &mut vo)
                    .unwrap();
                assert_eq!(ko, entry(seed, n), "K mismatch at {layer}:{slot}");
                assert_eq!(vo, entry(seed + 0.5, n), "V mismatch at {layer}:{slot}");
            }
        }
    }

    #[test]
    fn copy_is_a_snapshot_not_an_alias() {
        // This is the property copy-on-write depends on: after the copy, the
        // two blocks must be independent.
        let mut b = backend();
        let n = b.config().entry_elems();
        let at = PhysicalSlot::new(BlockId(0), 0);

        b.write_kv(0, at, &entry(1.0, n), &entry(2.0, n)).unwrap();
        b.copy_block(BlockId(0), BlockId(1)).unwrap();

        // Mutate the original.
        b.write_kv(0, at, &entry(9.0, n), &entry(9.0, n)).unwrap();

        let (mut ko, mut vo) = (vec![0.0; n], vec![0.0; n]);
        b.read_kv(0, PhysicalSlot::new(BlockId(1), 0), &mut ko, &mut vo)
            .unwrap();
        assert_eq!(ko, entry(1.0, n), "copy tracked a later write to the source");
        assert_eq!(vo, entry(2.0, n));
    }

    #[test]
    fn zero_block_clears_all_layers() {
        let mut b = backend();
        let n = b.config().entry_elems();
        for layer in 0..b.config().num_layers {
            b.write_kv(
                layer,
                PhysicalSlot::new(BlockId(2), 1),
                &entry(5.0, n),
                &entry(5.0, n),
            )
            .unwrap();
        }

        b.zero_block(BlockId(2)).unwrap();

        let (mut ko, mut vo) = (vec![0.0; n], vec![0.0; n]);
        for layer in 0..b.config().num_layers {
            b.read_kv(layer, PhysicalSlot::new(BlockId(2), 1), &mut ko, &mut vo)
                .unwrap();
            assert!(ko.iter().chain(vo.iter()).all(|&x| x == 0.0));
        }
    }

    #[test]
    fn self_copy_is_a_no_op_not_a_corruption() {
        let mut b = backend();
        let n = b.config().entry_elems();
        let at = PhysicalSlot::new(BlockId(1), 0);
        b.write_kv(0, at, &entry(3.0, n), &entry(4.0, n)).unwrap();

        b.copy_block(BlockId(1), BlockId(1)).unwrap();

        let (mut ko, mut vo) = (vec![0.0; n], vec![0.0; n]);
        b.read_kv(0, at, &mut ko, &mut vo).unwrap();
        assert_eq!(ko, entry(3.0, n));
        assert_eq!(vo, entry(4.0, n));
    }

    #[test]
    fn bad_addresses_and_shapes_are_rejected() {
        let mut b = backend();
        let cfg = b.config().clone();
        let n = cfg.entry_elems();
        let (k, v) = (entry(0.0, n), entry(0.0, n));

        assert_eq!(
            b.write_kv(99, PhysicalSlot::new(BlockId(0), 0), &k, &v)
                .unwrap_err(),
            CacheError::InvalidLayer {
                layer: 99,
                num_layers: cfg.num_layers
            }
        );
        assert_eq!(
            b.write_kv(0, PhysicalSlot::new(BlockId(99), 0), &k, &v)
                .unwrap_err(),
            CacheError::InvalidBlock {
                id: BlockId(99),
                num_blocks: cfg.num_blocks
            }
        );
        assert_eq!(
            b.write_kv(0, PhysicalSlot::new(BlockId(0), 99), &k, &v)
                .unwrap_err(),
            CacheError::SlotOutOfRange {
                slot: 99,
                block_size: cfg.block_size
            }
        );
        assert_eq!(
            b.write_kv(0, PhysicalSlot::new(BlockId(0), 0), &k[..n - 1], &v)
                .unwrap_err(),
            CacheError::ShapeMismatch {
                expected: n,
                actual: n - 1
            }
        );
    }

    #[test]
    fn allocated_bytes_tracks_config() {
        let b = CpuBackend::new(CacheConfig::llama_3_2_1b(512));
        // 512 blocks * 16 tokens * 32 KiB per token.
        assert_eq!(b.allocated_bytes(), 512 * 16 * 32 * 1024);
    }
}
