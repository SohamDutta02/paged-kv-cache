//! Checks the paged model against candle's own Qwen2 on a tiny model with
//! random weights, so no download is needed.

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Activation, VarBuilder};
use candle_transformers::models::qwen2::ModelForCausalLM;
use paged_kv_core::{BlockId, CpuBackend, SeqId};
use paged_kv_model::{Config, Engine, PagedQwen2, SeqInput};

const VOCAB: usize = 97;

fn config(tie_word_embeddings: bool) -> Config {
    Config {
        vocab_size: VOCAB,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        max_position_embeddings: 256,
        sliding_window: 256,
        max_window_layers: 2,
        tie_word_embeddings,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        use_sliding_window: false,
        hidden_act: Activation::Silu,
    }
}

/// Deterministic weights with Hugging Face's tensor names.
fn weights(cfg: &Config) -> Result<VarBuilder<'static>> {
    let mut seed: u64 = 0x5EED;
    let mut uniform = |shape: &[usize], scale: f32, offset: f32| -> Result<Tensor> {
        let len = shape.iter().product();
        let values: Vec<f32> = (0..len)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                offset + scale * ((seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0)
            })
            .collect();
        Tensor::from_vec(values, shape, &Device::Cpu)
    };
    let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
    let d = h / cfg.num_attention_heads;
    let kv = cfg.num_key_value_heads * d;
    let mut ts = HashMap::new();
    // A small embedding and strong attention output make the next token
    // depend on context, so wrong K/V changes what gets generated.
    ts.insert(
        "model.embed_tokens.weight".into(),
        uniform(&[VOCAB, h], 0.3, 0.0)?,
    );
    for l in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{l}");
        let w = |n: usize| 1.5 / (n as f32).sqrt();
        for (name, shape, scale, offset) in [
            ("self_attn.q_proj.weight", vec![h, h], w(h), 0.0),
            ("self_attn.q_proj.bias", vec![h], 0.1, 0.0),
            ("self_attn.k_proj.weight", vec![kv, h], w(h), 0.0),
            ("self_attn.k_proj.bias", vec![kv], 0.1, 0.0),
            ("self_attn.v_proj.weight", vec![kv, h], 3.0 * w(h), 0.0),
            ("self_attn.v_proj.bias", vec![kv], 0.1, 0.0),
            ("self_attn.o_proj.weight", vec![h, h], 3.0 * w(h), 0.0),
            ("mlp.gate_proj.weight", vec![i, h], w(h), 0.0),
            ("mlp.up_proj.weight", vec![i, h], w(h), 0.0),
            ("mlp.down_proj.weight", vec![h, i], w(i), 0.0),
            ("input_layernorm.weight", vec![h], 0.2, 1.0),
            ("post_attention_layernorm.weight", vec![h], 0.2, 1.0),
        ] {
            ts.insert(format!("{p}.{name}"), uniform(&shape, scale, offset)?);
        }
    }
    ts.insert("model.norm.weight".into(), uniform(&[h], 0.2, 1.0)?);
    if !cfg.tie_word_embeddings {
        ts.insert("lm_head.weight".into(), uniform(&[VOCAB, h], 1.0, 0.0)?);
    }
    Ok(VarBuilder::from_tensors(ts, DType::F32, &Device::Cpu))
}

fn tokens(len: usize, salt: u32) -> Vec<u32> {
    (0..len as u32)
        .map(|t| (t * 31 + salt * 17 + 5) % VOCAB as u32)
        .collect()
}

/// Logits for the last of `tokens`, continuing the reference's own cache.
fn reference_logits(m: &mut ModelForCausalLM, tokens: &[u32], offset: usize) -> Result<Vec<f32>> {
    let input = Tensor::new(tokens, &Device::Cpu)?.unsqueeze(0)?;
    m.forward(&input, offset)?.flatten_all()?.to_vec1()
}

fn greedy_reference(mut m: ModelForCausalLM, prompt: &[u32], n: usize) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    let mut logits = reference_logits(&mut m, prompt, 0)?;
    for _ in 0..n {
        let next = argmax(&logits);
        out.push(next);
        logits = reference_logits(&mut m, &[next], prompt.len() + out.len() - 1)?;
    }
    Ok(out)
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

fn assert_close(paged: &[f32], reference: &[f32], what: &str) {
    let worst = paged
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(worst < 1e-4, "{what}: logits differ by up to {worst}");
}

#[test]
fn paged_forward_matches_candle_qwen2() -> Result<()> {
    for tie in [true, false] {
        let cfg = config(tie);
        let vb = weights(&cfg)?;
        let reference = ModelForCausalLM::new(&cfg, vb.clone())?;
        let model = PagedQwen2::new(&cfg, vb)?;
        let mut cache = CpuBackend::new(model.cache_config(16, 4));

        // Scattered, out-of-order blocks with room for a few decode steps.
        let table_a = [BlockId(9), BlockId(2), BlockId(14), BlockId(5)];
        let table_b = [BlockId(0), BlockId(11), BlockId(7)];
        let mut a = tokens(11, 1);
        let mut b = tokens(5, 2);
        let mut ref_a = reference.clone();
        let mut ref_b = reference.clone();

        // Two prefills in one batch, then decode steps fed the reference's picks.
        let mut inputs = (a.clone(), 0, b.clone(), 0);
        for step in 0..4 {
            let logits = model.forward(
                &mut cache,
                &[
                    SeqInput {
                        tokens: &inputs.0,
                        start: inputs.1,
                        block_table: &table_a,
                    },
                    SeqInput {
                        tokens: &inputs.2,
                        start: inputs.3,
                        block_table: &table_b,
                    },
                ],
            )?;
            let paged_a: Vec<f32> = logits.get(0)?.to_vec1()?;
            let paged_b: Vec<f32> = logits.get(1)?.to_vec1()?;
            let want_a = reference_logits(&mut ref_a, &inputs.0, inputs.1)?;
            let want_b = reference_logits(&mut ref_b, &inputs.2, inputs.3)?;
            assert_close(
                &paged_a,
                &want_a,
                &format!("tied {tie}, step {step}, seq a"),
            );
            assert_close(
                &paged_b,
                &want_b,
                &format!("tied {tie}, step {step}, seq b"),
            );

            let (next_a, next_b) = (argmax(&want_a), argmax(&want_b));
            inputs = (vec![next_a], a.len(), vec![next_b], b.len());
            a.push(next_a);
            b.push(next_b);
        }
    }
    Ok(())
}

#[test]
fn engine_generates_the_same_tokens_as_candle_qwen2() -> Result<()> {
    let cfg = config(true);
    let vb = weights(&cfg)?;
    let reference = ModelForCausalLM::new(&cfg, vb.clone())?;
    let model = PagedQwen2::new(&cfg, vb)?;
    // Small enough that the batch runs out of blocks and preempts.
    let cache = CpuBackend::new(model.cache_config(18, 4));
    let mut engine = Engine::new(model, cache, 4, true);

    // Most prompts share a 10-token prefix, so later ones hit the cache.
    let shared = tokens(10, 9);
    let mut prompts: Vec<Vec<u32>> = (0..5)
        .map(|i| [shared.clone(), tokens(1 + i * 2, i as u32)].concat())
        .collect();
    prompts.push(tokens(7, 42));
    const NEW_TOKENS: usize = 12;
    for (i, prompt) in prompts.iter().enumerate() {
        engine.add_request(SeqId(i as u64), prompt.clone(), NEW_TOKENS)?;
    }

    let mut generated = HashMap::new();
    while !engine.is_idle() {
        for f in engine.step()?.finished {
            generated.insert(f.id, f.tokens);
        }
        engine
            .scheduler()
            .check_invariants()
            .map_err(candle_core::Error::Msg)?;
        assert!(engine.stats().steps < 500, "engine stopped making progress");
    }

    for (i, prompt) in prompts.iter().enumerate() {
        let want = greedy_reference(reference.clone(), prompt, NEW_TOKENS)?;
        assert_eq!(generated[&SeqId(i as u64)], want, "request {i}");
    }
    let stats = engine.stats();
    assert!(
        stats.prefill_tokens_cached > 0,
        "no prefix cache hits: {stats:?}"
    );
    assert!(stats.preemptions > 0, "no preemptions: {stats:?}");
    Ok(())
}

#[test]
fn engine_stops_at_a_stop_token() -> Result<()> {
    let cfg = config(true);
    let vb = weights(&cfg)?;
    let reference = ModelForCausalLM::new(&cfg, vb.clone())?;
    let prompt = tokens(6, 3);
    let full = greedy_reference(reference, &prompt, 8)?;
    let stop = full[3];
    let expected_len = full.iter().position(|&t| t == stop).unwrap() + 1;

    let model = PagedQwen2::new(&cfg, vb)?;
    let cache = CpuBackend::new(model.cache_config(16, 4));
    let mut engine = Engine::new(model, cache, 2, true).stop_at(&[stop]);
    engine.add_request(SeqId(0), prompt, 8)?;
    let mut finished = Vec::new();
    while !engine.is_idle() {
        finished.extend(engine.step()?.finished);
    }
    assert_eq!(finished[0].tokens, full[..expected_len]);
    assert_eq!(engine.scheduler().pool_utilization(), 0.0);
    Ok(())
}

#[test]
fn load_config_accepts_newer_transformers_output() -> Result<()> {
    let dir = std::env::temp_dir().join(format!("paged-kv-config-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("config.json");
    let written = serde_json::json!({
        "vocab_size": 97, "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "max_position_embeddings": 256, "sliding_window": null, "use_sliding_window": false,
        "tie_word_embeddings": true, "rms_norm_eps": 1e-6, "hidden_act": "silu",
        "rope_parameters": { "rope_theta": 10000.0, "rope_type": "default" }
    });
    std::fs::write(&path, written.to_string())?;
    let cfg = paged_kv_model::load_config(&path)?;
    assert_eq!(cfg.rope_theta, 10_000.0);
    assert_eq!(cfg.sliding_window, 256);
    assert_eq!(cfg.max_window_layers, 2);

    let mut sliding = written.clone();
    sliding["use_sliding_window"] = true.into();
    std::fs::write(&path, sliding.to_string())?;
    assert!(paged_kv_model::load_config(&path).is_err());
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}

#[test]
fn sampled_tokens_add_up_and_cancel_frees_blocks() -> Result<()> {
    let cfg = config(true);
    let model = PagedQwen2::new(&cfg, weights(&cfg)?)?;
    let cache = CpuBackend::new(model.cache_config(32, 4));
    let mut engine = Engine::new(model, cache, 4, true);
    engine.add_request(SeqId(0), tokens(9, 1), 6)?;
    engine.add_request(SeqId(1), tokens(5, 2), 20)?;

    let mut streamed: HashMap<SeqId, Vec<u32>> = HashMap::new();
    let mut finished = Vec::new();
    while finished.is_empty() {
        let out = engine.step()?;
        for (id, token) in out.sampled {
            streamed.entry(id).or_default().push(token);
        }
        finished.extend(out.finished);
    }
    assert_eq!(finished[0].id, SeqId(0));
    assert_eq!(streamed[&SeqId(0)], finished[0].tokens);

    assert!(engine.cancel(SeqId(1)));
    assert!(!engine.cancel(SeqId(1)));
    assert!(engine.is_idle());
    assert_eq!(engine.scheduler().pool_utilization(), 0.0);
    Ok(())
}

#[test]
fn dummy_weights_load_and_are_deterministic() -> Result<()> {
    let cfg = config(false);
    let run = |seed| -> Result<Vec<u32>> {
        let model = PagedQwen2::new(&cfg, paged_kv_model::dummy_weights(&cfg, seed)?)?;
        let mut engine = Engine::new(model, CpuBackend::new(model_cache(&cfg)), 2, false);
        engine.add_request(SeqId(0), tokens(6, 4), 5)?;
        let mut out = Vec::new();
        while !engine.is_idle() {
            for f in engine.step()?.finished {
                out = f.tokens;
            }
        }
        Ok(out)
    };
    assert_eq!(run(1)?, run(1)?);
    assert_eq!(run(1)?.len(), 5);
    Ok(())
}

fn model_cache(cfg: &Config) -> paged_kv_core::CacheConfig {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    paged_kv_core::CacheConfig {
        num_blocks: 16,
        block_size: 4,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim,
        num_layers: cfg.num_hidden_layers,
        dtype_bytes: 4,
    }
}
