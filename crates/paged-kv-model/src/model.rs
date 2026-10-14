use candle_core::{bail, DType, Device, Module, Result, Tensor};
use candle_nn::rotary_emb::rope;
use candle_nn::{
    embedding, linear, linear_no_bias, rms_norm, Activation, Embedding, Linear, RmsNorm, VarBuilder,
};
pub use candle_transformers::models::qwen2::Config;
use paged_kv_core::{AttentionSeq, BlockId, CacheConfig, KvBackend, PhysicalSlot};

/// One sequence's share of a [`PagedQwen2::forward`] batch.
#[derive(Debug, Clone, Copy)]
pub struct SeqInput<'a> {
    /// New tokens, at positions `start..start + tokens.len()`. Everything
    /// before `start` must already be in the cache.
    pub tokens: &'a [u32],
    pub start: usize,
    /// Must cover every position up to the last new token.
    pub block_table: &'a [BlockId],
}

struct Layer {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act: Activation,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

/// Qwen2 whose attention reads K and V through block tables. Weights and
/// numerics follow `candle_transformers::models::qwen2`, so the two can be
/// compared token for token.
pub struct PagedQwen2 {
    embed_tokens: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: Linear,
    cos: Tensor,
    sin: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    device: Device,
}

fn cache_err(e: paged_kv_core::CacheError) -> candle_core::Error {
    candle_core::Error::Msg(format!("paged cache: {e}"))
}

impl PagedQwen2 {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        if vb.dtype() != DType::F32 {
            bail!("the paged cache stores f32, so load the weights as f32");
        }
        let hidden = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = hidden / num_heads;
        let vb_m = vb.pp("model");

        let embed_tokens = embedding(cfg.vocab_size, hidden, vb_m.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let vb_l = vb_m.pp("layers").pp(i);
            let attn = vb_l.pp("self_attn");
            let mlp = vb_l.pp("mlp");
            layers.push(Layer {
                q_proj: linear(hidden, num_heads * head_dim, attn.pp("q_proj"))?,
                k_proj: linear(hidden, num_kv_heads * head_dim, attn.pp("k_proj"))?,
                v_proj: linear(hidden, num_kv_heads * head_dim, attn.pp("v_proj"))?,
                o_proj: linear_no_bias(num_heads * head_dim, hidden, attn.pp("o_proj"))?,
                gate_proj: linear_no_bias(hidden, cfg.intermediate_size, mlp.pp("gate_proj"))?,
                up_proj: linear_no_bias(hidden, cfg.intermediate_size, mlp.pp("up_proj"))?,
                down_proj: linear_no_bias(cfg.intermediate_size, hidden, mlp.pp("down_proj"))?,
                act: cfg.hidden_act,
                input_layernorm: rms_norm(hidden, cfg.rms_norm_eps, vb_l.pp("input_layernorm"))?,
                post_attention_layernorm: rms_norm(
                    hidden,
                    cfg.rms_norm_eps,
                    vb_l.pp("post_attention_layernorm"),
                )?,
            });
        }
        let norm = rms_norm(hidden, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            linear_no_bias(hidden, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            Linear::new(embed_tokens.embeddings().clone(), None)
        };

        // Same tables as the reference, so RoPE matches bit for bit.
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, head_dim / 2), vb.device())?;
        let positions = Tensor::arange(0u32, cfg.max_position_embeddings as u32, vb.device())?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let freqs = positions.matmul(&inv_freq)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            cos: freqs.cos()?,
            sin: freqs.sin()?,
            num_heads,
            num_kv_heads,
            head_dim,
            device: vb.device().clone(),
        })
    }

    /// The cache geometry this model needs.
    pub fn cache_config(&self, num_blocks: usize, block_size: usize) -> CacheConfig {
        CacheConfig {
            num_blocks,
            block_size,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            num_layers: self.layers.len(),
            dtype_bytes: 4,
        }
    }

    /// Runs every sequence's new tokens as one batch, writing their K/V into
    /// `cache`, and returns logits for each sequence's last token, shaped
    /// `(seqs.len(), vocab_size)`.
    pub fn forward(&self, cache: &mut impl KvBackend, seqs: &[SeqInput<'_>]) -> Result<Tensor> {
        let block_size = cache.config().block_size;
        let mut ids = Vec::new();
        let mut positions = Vec::new();
        let mut slots = Vec::new();
        let mut last_rows = Vec::with_capacity(seqs.len());
        for s in seqs {
            if s.tokens.is_empty() {
                bail!("every sequence in a batch needs at least one new token");
            }
            for (j, &token) in s.tokens.iter().enumerate() {
                let pos = s.start + j;
                let Some(&block) = s.block_table.get(pos / block_size) else {
                    bail!("block table has no block for position {pos}");
                };
                ids.push(token);
                positions.push(pos as u32);
                slots.push(PhysicalSlot::new(block, pos % block_size));
            }
            last_rows.push(ids.len() as u32 - 1);
        }
        let attention: Vec<AttentionSeq<'_>> = seqs
            .iter()
            .map(|s| AttentionSeq {
                block_table: s.block_table,
                context_len: s.start + s.tokens.len(),
                num_queries: s.tokens.len(),
            })
            .collect();

        let n = ids.len();
        let (heads, kv_heads, d) = (self.num_heads, self.num_kv_heads, self.head_dim);
        let positions = Tensor::new(positions.as_slice(), &self.device)?;
        // Per-token tables: every token in the batch has its own position.
        let cos = self
            .cos
            .index_select(&positions, 0)?
            .reshape((n, 1, d / 2))?;
        let sin = self
            .sin
            .index_select(&positions, 0)?
            .reshape((n, 1, d / 2))?;

        let mut xs = self
            .embed_tokens
            .forward(&Tensor::new(ids.as_slice(), &self.device)?)?;
        for (index, layer) in self.layers.iter().enumerate() {
            let h = layer.input_layernorm.forward(&xs)?;
            let q = layer.q_proj.forward(&h)?.reshape((n, heads, 1, d))?;
            let k = layer.k_proj.forward(&h)?.reshape((n, kv_heads, 1, d))?;
            let v = layer.v_proj.forward(&h)?;
            let q = rope(&q, &cos, &sin)?;
            let k = rope(&k, &cos, &sin)?;

            // Writes for the whole batch land before any attention reads, so
            // a sequence can read prefix blocks another one writes this step.
            let k: Vec<f32> = k.flatten_all()?.to_vec1()?;
            let v: Vec<f32> = v.flatten_all()?.to_vec1()?;
            cache
                .write_kv_batch(index, &slots, &k, &v)
                .map_err(cache_err)?;
            let q: Vec<f32> = q.flatten_all()?.to_vec1()?;
            let mut attn = vec![0f32; q.len()];
            cache
                .paged_attention(index, &attention, &q, &mut attn)
                .map_err(cache_err)?;
            let attn = Tensor::from_vec(attn, (n, heads * d), &self.device)?;

            let xs_attn = (layer.o_proj.forward(&attn)? + &xs)?;
            let h = layer.post_attention_layernorm.forward(&xs_attn)?;
            let gate = layer.gate_proj.forward(&h)?.apply(&layer.act)?;
            let mlp = layer
                .down_proj
                .forward(&(gate * layer.up_proj.forward(&h)?)?)?;
            xs = (xs_attn + mlp)?;
        }

        let last = Tensor::new(last_rows.as_slice(), &self.device)?;
        xs.index_select(&last, 0)?
            .apply(&self.norm)?
            .apply(&self.lm_head)
    }
}
