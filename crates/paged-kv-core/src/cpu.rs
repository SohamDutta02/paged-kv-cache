use crate::backend::{check_attention, AttentionSeq, KvBackend};
use crate::config::CacheConfig;
use crate::error::{CacheError, Result};
use crate::types::{BlockId, PhysicalSlot};

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

    fn write_kv(&mut self, layer: usize, dst: PhysicalSlot, k: &[f32], v: &[f32]) -> Result<()> {
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
            // Distinct blocks never overlap, so no temporary buffer is needed.
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

    fn paged_attention(
        &self,
        layer: usize,
        seqs: &[AttentionSeq<'_>],
        q: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        let heads = check_attention(&self.config, layer, seqs, q.len(), out.len())?;
        let d = self.config.head_dim;
        let bs = self.config.block_size;
        let group = heads / self.config.num_kv_heads;
        let scale = 1.0 / (d as f32).sqrt();
        let mut weights = Vec::new();
        let mut row = 0;
        for s in seqs {
            for i in 0..s.num_queries {
                let pos = s.context_len - s.num_queries + i;
                for h in 0..heads {
                    let at = (row * heads + h) * d;
                    let qh = &q[at..at + d];
                    let entry = |t: usize| {
                        let slot = PhysicalSlot::new(s.block_table[t / bs], t % bs);
                        self.slot_offset(layer, slot) + (h / group) * d
                    };

                    weights.clear();
                    weights.extend((0..=pos).map(|t| {
                        let k = &self.k[entry(t)..entry(t) + d];
                        qh.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale
                    }));
                    let max = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0;
                    for w in &mut weights {
                        *w = (*w - max).exp();
                        sum += *w;
                    }

                    let oh = &mut out[at..at + d];
                    oh.fill(0.0);
                    for (t, &w) in weights.iter().enumerate() {
                        let v = &self.v[entry(t)..entry(t) + d];
                        for (o, &x) in oh.iter_mut().zip(v) {
                            *o += w * x;
                        }
                    }
                    for o in oh.iter_mut() {
                        *o /= sum;
                    }
                }
                row += 1;
            }
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
        // Copy-on-write depends on the two blocks being independent afterwards.
        let mut b = backend();
        let n = b.config().entry_elems();
        let at = PhysicalSlot::new(BlockId(0), 0);

        b.write_kv(0, at, &entry(1.0, n), &entry(2.0, n)).unwrap();
        b.copy_block(BlockId(0), BlockId(1)).unwrap();

        b.write_kv(0, at, &entry(9.0, n), &entry(9.0, n)).unwrap();

        let (mut ko, mut vo) = (vec![0.0; n], vec![0.0; n]);
        b.read_kv(0, PhysicalSlot::new(BlockId(1), 0), &mut ko, &mut vo)
            .unwrap();
        assert_eq!(
            ko,
            entry(1.0, n),
            "copy tracked a later write to the source"
        );
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

    #[test]
    fn write_kv_batch_matches_repeated_write_kv() {
        // The reference the CUDA oracle tests compare against.
        let mut b = backend();
        let n = b.config().entry_elems();
        let dsts = [
            PhysicalSlot::new(BlockId(0), 0),
            PhysicalSlot::new(BlockId(0), 1),
            PhysicalSlot::new(BlockId(3), 2),
        ];
        let k: Vec<f32> = (0..dsts.len() * n).map(|i| i as f32 * 0.5).collect();
        let v: Vec<f32> = (0..dsts.len() * n).map(|i| i as f32 * 0.25 + 1.0).collect();

        b.write_kv_batch(1, &dsts, &k, &v).unwrap();

        let (mut ko, mut vo) = (vec![0.0; n], vec![0.0; n]);
        for (i, &dst) in dsts.iter().enumerate() {
            b.read_kv(1, dst, &mut ko, &mut vo).unwrap();
            assert_eq!(ko, k[i * n..(i + 1) * n]);
            assert_eq!(vo, v[i * n..(i + 1) * n]);
        }
    }

    #[test]
    fn write_kv_batch_rejects_mismatched_lengths_without_panicking() {
        let mut b = backend();
        let n = b.config().entry_elems();
        let dsts = [
            PhysicalSlot::new(BlockId(0), 0),
            PhysicalSlot::new(BlockId(0), 1),
        ];

        let k = vec![0.0; n];
        let v = vec![0.0; 2 * n];

        assert_eq!(
            b.write_kv_batch(0, &dsts, &k, &v).unwrap_err(),
            CacheError::ShapeMismatch {
                expected: 2 * n,
                actual: n
            }
        );
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
            self.below(1 << 16) as f32 / 32768.0 - 1.0
        }
    }

    const HEADS: usize = 4;

    /// Two KV heads, so four query heads share each in pairs.
    fn attention_config() -> CacheConfig {
        CacheConfig {
            num_blocks: 32,
            block_size: 4,
            num_kv_heads: 2,
            head_dim: 8,
            num_layers: 2,
            dtype_bytes: 4,
        }
    }

    /// One sequence with its K and V stored contiguously.
    struct Dense {
        context: usize,
        queries: usize,
        k: Vec<f32>,
        v: Vec<f32>,
        q: Vec<f32>,
    }

    fn dense_attention(seq: &Dense, cfg: &CacheConfig) -> Vec<f32> {
        let (d, kv_heads) = (cfg.head_dim, cfg.num_kv_heads);
        let group = HEADS / kv_heads;
        let mut out = vec![0.0; seq.queries * HEADS * d];
        for i in 0..seq.queries {
            let pos = seq.context - seq.queries + i;
            for h in 0..HEADS {
                let qh = &seq.q[(i * HEADS + h) * d..][..d];
                let kv = |t: usize| (t * kv_heads + h / group) * d;
                let scores: Vec<f32> = (0..=pos)
                    .map(|t| {
                        let k = &seq.k[kv(t)..kv(t) + d];
                        qh.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() / (d as f32).sqrt()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for (t, e) in exps.iter().enumerate() {
                    for j in 0..d {
                        out[(i * HEADS + h) * d + j] += e / sum * seq.v[kv(t) + j];
                    }
                }
            }
        }
        out
    }

    #[test]
    fn paged_attention_matches_dense_attention() {
        let cfg = attention_config();
        let (n, bs) = (cfg.entry_elems(), cfg.block_size);
        let mut cache = CpuBackend::new(cfg.clone());
        let mut rng = Lcg(3);

        let mut free: Vec<BlockId> = (0..cfg.num_blocks as u32).map(BlockId).collect();
        for i in (1..free.len()).rev() {
            free.swap(i, rng.below(i + 1));
        }

        // (context, queries): decode, decode filling a block, full prefill,
        // prefill after a cached prefix, a one-token prompt.
        let shapes = [(7, 1), (8, 1), (9, 9), (13, 5), (1, 1)];
        let mut seqs = Vec::new();
        let mut tables = Vec::new();
        for &(context, queries) in &shapes {
            let table: Vec<BlockId> = (0..cfg.blocks_for_tokens(context))
                .map(|_| free.pop().unwrap())
                .collect();
            let mut values = |len: usize| (0..len).map(|_| rng.value()).collect::<Vec<f32>>();
            let seq = Dense {
                context,
                queries,
                k: values(context * n),
                v: values(context * n),
                q: values(queries * HEADS * cfg.head_dim),
            };
            for t in 0..context {
                let at = PhysicalSlot::new(table[t / bs], t % bs);
                let range = t * n..(t + 1) * n;
                cache
                    .write_kv(1, at, &seq.k[range.clone()], &seq.v[range])
                    .unwrap();
                cache.write_kv(0, at, &vec![9.0; n], &vec![9.0; n]).unwrap();
            }
            for slot in context % bs..if context % bs == 0 { 0 } else { bs } {
                let at = PhysicalSlot::new(table[context / bs], slot);
                cache
                    .write_kv(1, at, &vec![f32::NAN; n], &vec![f32::NAN; n])
                    .unwrap();
            }
            seqs.push(seq);
            tables.push(table);
        }

        let batch: Vec<AttentionSeq<'_>> = seqs
            .iter()
            .zip(&tables)
            .map(|(s, table)| AttentionSeq {
                block_table: table,
                context_len: s.context,
                num_queries: s.queries,
            })
            .collect();
        let q: Vec<f32> = seqs.iter().flat_map(|s| s.q.iter().copied()).collect();
        let mut out = vec![0.0; q.len()];
        cache.paged_attention(1, &batch, &q, &mut out).unwrap();

        let mut at = 0;
        for (seq, &(context, queries)) in seqs.iter().zip(&shapes) {
            let expected = dense_attention(seq, &cfg);
            for (j, (&got, &want)) in out[at..at + expected.len()]
                .iter()
                .zip(&expected)
                .enumerate()
            {
                assert!(
                    (got - want).abs() <= 1e-5 * (1.0 + want.abs()),
                    "context {context}, {queries} queries, value {j}: got {got}, want {want}"
                );
            }
            at += expected.len();
        }
    }

    #[test]
    fn paged_attention_rejects_bad_calls() {
        let cfg = attention_config();
        let cache = CpuBackend::new(cfg.clone());
        let table = [BlockId(0), BlockId(1)];
        let seq = |context_len, num_queries| AttentionSeq {
            block_table: &table,
            context_len,
            num_queries,
        };
        let row = HEADS * cfg.head_dim;
        let q = vec![0.0; 2 * row];
        let mut out = vec![0.0; 2 * row];

        assert!(cache
            .paged_attention(1, &[seq(5, 1)], &q[..row], &mut out[..row])
            .is_ok());
        assert_eq!(
            cache.paged_attention(1, &[seq(9, 1)], &q[..row], &mut out[..row]),
            Err(CacheError::ShapeMismatch {
                expected: 3,
                actual: 2
            }),
            "block table too short for the context"
        );
        assert!(
            cache
                .paged_attention(1, &[seq(1, 2)], &q, &mut out)
                .is_err(),
            "more queries than tokens"
        );
        let three_heads = 3 * cfg.head_dim;
        assert!(
            cache
                .paged_attention(1, &[seq(5, 1)], &q[..three_heads], &mut out[..three_heads])
                .is_err(),
            "query heads not a multiple of the KV heads"
        );
        assert!(
            cache
                .paged_attention(1, &[seq(5, 1)], &q[..row], &mut out[..row - 1])
                .is_err(),
            "output shape differs from the queries"
        );
        assert_eq!(
            cache.paged_attention(2, &[seq(5, 1)], &q[..row], &mut out[..row]),
            Err(CacheError::InvalidLayer {
                layer: 2,
                num_layers: 2
            })
        );
        let outside = [BlockId(99)];
        let bad = AttentionSeq {
            block_table: &outside,
            context_len: 1,
            num_queries: 1,
        };
        assert!(matches!(
            cache.paged_attention(1, &[bad], &q[..row], &mut out[..row]),
            Err(CacheError::InvalidBlock { .. })
        ));
        cache.paged_attention(0, &[], &[], &mut []).unwrap();
    }
}
