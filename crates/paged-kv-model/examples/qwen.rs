//! Generates answers to four chat prompts that share a system prompt, with the
//! K/V in the paged cache, then checks every token against candle's own Qwen2.
//!
//!     cargo run --release -p paged-kv-model --example qwen -- <model-dir> [max-new-tokens]
//!
//! `<model-dir>` holds `config.json`, `tokenizer.json` and `model.safetensors`
//! from Qwen/Qwen2.5-0.5B-Instruct (notebooks/qwen_check.ipynb downloads them).
//! With `--features cuda` the K/V cache lives on the GPU and attention runs in
//! paged-kv-cuda's kernel; the rest of the model stays on the CPU.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::ModelForCausalLM;
use paged_kv_core::{KvBackend, SeqId};
use paged_kv_model::{Config, Engine, EngineStats, PagedQwen2};
use tokenizers::Tokenizer;

const SYSTEM: &str = "You are a concise assistant. Answer every question in one or two \
sentences, in plain language, without lists or headings.";
const QUESTIONS: [&str; 4] = [
    "What does a GPU do?",
    "Why is the sky blue?",
    "What is a hash table?",
    "Why do we sleep?",
];
const BLOCK_SIZE: usize = 16;
const NUM_BLOCKS: usize = 128;
const MAX_BATCH: usize = 4;

fn chat_prompt(question: &str) -> String {
    format!(
        "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    )
}

fn argmax(logits: &Tensor) -> candle_core::Result<u32> {
    logits.flatten_all()?.argmax(0)?.to_scalar::<u32>()
}

/// Greedy decoding with the reference model's own contiguous KV cache.
fn reference_generate(
    mut model: ModelForCausalLM,
    prompt: &[u32],
    max_new: usize,
    stop: &[u32],
) -> candle_core::Result<Vec<u32>> {
    let mut out = Vec::new();
    let mut logits = model.forward(&Tensor::new(prompt, &Device::Cpu)?.unsqueeze(0)?, 0)?;
    while out.len() < max_new {
        let next = argmax(&logits)?;
        out.push(next);
        if stop.contains(&next) {
            break;
        }
        let offset = prompt.len() + out.len() - 1;
        logits = model.forward(&Tensor::new(&[next], &Device::Cpu)?.unsqueeze(0)?, offset)?;
    }
    Ok(out)
}

type BoxError = Box<dyn std::error::Error + Send + Sync>;

fn generate<B: KvBackend>(
    model: PagedQwen2,
    cache: B,
    prompts: &[Vec<u32>],
    max_new: usize,
    stop: &[u32],
) -> Result<(HashMap<SeqId, Vec<u32>>, EngineStats), BoxError> {
    let mut engine = Engine::new(model, cache, MAX_BATCH, true).stop_at(stop);
    for (i, prompt) in prompts.iter().enumerate() {
        engine.add_request(SeqId(i as u64), prompt.clone(), max_new)?;
    }
    let mut finished = HashMap::new();
    while !engine.is_idle() {
        for f in engine.step()? {
            finished.insert(f.id, f.tokens);
        }
    }
    Ok((finished, engine.stats()))
}

fn main() -> Result<(), BoxError> {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(
        args.next()
            .ok_or("usage: qwen <model-dir> [max-new-tokens]")?,
    );
    let max_new: usize = args.next().map(|n| n.parse()).transpose()?.unwrap_or(64);

    let cfg: Config = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
    let stop: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|t| tokenizer.token_to_id(t))
        .collect();
    // The cache stores f32, so both models run in f32.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[dir.join("model.safetensors")],
            DType::F32,
            &Device::Cpu,
        )?
    };

    let prompts: Vec<Vec<u32>> = QUESTIONS
        .iter()
        .map(|q| Ok(tokenizer.encode(chat_prompt(q), false)?.get_ids().to_vec()))
        .collect::<Result<_, BoxError>>()?;

    let model = PagedQwen2::new(&cfg, vb.clone())?;
    let cache_config = model.cache_config(NUM_BLOCKS, BLOCK_SIZE);
    #[cfg(feature = "cuda")]
    let cache = paged_kv_cuda::CudaBackend::new(cache_config)?;
    #[cfg(not(feature = "cuda"))]
    let cache = paged_kv_core::CpuBackend::new(cache_config);
    println!("K/V cache on {}\n", cache.device_name());

    let started = Instant::now();
    let (paged, stats) = generate(model, cache, &prompts, max_new, &stop)?;
    let paged_secs = started.elapsed().as_secs_f64();

    let reference = ModelForCausalLM::new(&cfg, vb)?;
    let started = Instant::now();
    let mut mismatches = 0;
    for (i, (question, prompt)) in QUESTIONS.iter().zip(&prompts).enumerate() {
        let want = reference_generate(reference.clone(), prompt, max_new, &stop)?;
        let got = &paged[&SeqId(i as u64)];
        let text = tokenizer.decode(got, true)?;
        println!("Q: {question}\nA: {}\n", text.trim());
        if *got != want {
            mismatches += 1;
            let at = got.iter().zip(&want).take_while(|(a, b)| a == b).count();
            println!("   differs from the reference at generated token {at}\n");
        }
    }
    let reference_secs = started.elapsed().as_secs_f64();

    println!(
        "Prompt tokens: {} computed, {} from the prefix cache. Preemptions: {}. Steps: {}.",
        stats.prefill_tokens_computed, stats.prefill_tokens_cached, stats.preemptions, stats.steps
    );
    println!(
        "{} tokens generated in {paged_secs:.1} s paged (batched), {reference_secs:.1} s reference (one prompt at a time).",
        stats.tokens_generated
    );
    if mismatches > 0 {
        return Err(format!(
            "{mismatches} of {} answers differ from the reference",
            prompts.len()
        )
        .into());
    }
    println!(
        "All {} answers match candle's Qwen2 token for token.",
        prompts.len()
    );
    Ok(())
}
