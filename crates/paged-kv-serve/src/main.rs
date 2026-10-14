use std::error::Error;
use std::path::PathBuf;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use clap::Parser;
use paged_kv_core::KvBackend;
use paged_kv_model::{dummy_weights, load_config, Engine, PagedQwen2};
use paged_kv_serve::{router, stop_tokens, ChatTemplate, ServerOptions};
use tokenizers::Tokenizer;

/// OpenAI-compatible server for Qwen2 checkpoints, with a paged KV cache.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// Checkpoint directory: config.json, tokenizer.json and *.safetensors.
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    /// KV cache size in blocks.
    #[arg(long, default_value_t = 1024)]
    num_blocks: usize,
    #[arg(long, default_value_t = 16)]
    block_size: usize,
    #[arg(long, default_value_t = 16)]
    max_batch_size: usize,
    #[arg(long)]
    no_prefix_caching: bool,
    /// Where the K/V cache lives: cpu, or cuda in builds with the cuda feature.
    #[arg(long, default_value = "cpu")]
    device: String,
    /// Random weights in the checkpoint's shape, to benchmark without them.
    #[arg(long)]
    dummy_weights: bool,
    /// Reported in responses; defaults to the checkpoint directory's name.
    #[arg(long)]
    model_name: Option<String>,
    #[arg(long, default_value_t = 256)]
    max_in_flight: usize,
    /// Completion length when a request doesn't set max_tokens.
    #[arg(long, default_value_t = 256)]
    default_max_tokens: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let args = Args::parse();

    let config = load_config(&args.model.join("config.json"))?;
    let tokenizer = Tokenizer::from_file(args.model.join("tokenizer.json"))?;
    let chat = ChatTemplate::from_model_dir(&args.model)?.unwrap_or_else(ChatTemplate::chatml);
    let stop = stop_tokens(&args.model, &tokenizer);
    let vb = if args.dummy_weights {
        tracing::warn!("serving random weights: output is meaningless, timings are real");
        dummy_weights(&config, 0)?
    } else {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&args.model)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        // Memory-mapped: the files must not change while the server runs.
        unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, &Device::Cpu)? }
    };
    let model = PagedQwen2::new(&config, vb)?;
    let cache_config = model.cache_config(args.num_blocks, args.block_size);
    let options = ServerOptions {
        model_name: args.model_name.clone().unwrap_or_else(|| {
            args.model
                .file_name()
                .map_or("model".into(), |n| n.to_string_lossy().into_owned())
        }),
        max_context: config
            .max_position_embeddings
            .min(args.num_blocks * args.block_size),
        default_max_tokens: args.default_max_tokens,
        max_in_flight: args.max_in_flight,
    };
    let prefix_caching = !args.no_prefix_caching;

    let app = match args.device.as_str() {
        "cpu" => serve(
            Engine::new(
                model,
                paged_kv_core::CpuBackend::new(cache_config),
                args.max_batch_size,
                prefix_caching,
            ),
            tokenizer,
            chat,
            stop,
            options,
        ),
        #[cfg(feature = "cuda")]
        "cuda" => serve(
            Engine::new(
                model,
                paged_kv_cuda::CudaBackend::new(cache_config)?,
                args.max_batch_size,
                prefix_caching,
            ),
            tokenizer,
            chat,
            stop,
            options,
        ),
        other => return Err(format!("unsupported device {other:?}").into()),
    };

    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("listening on http://{addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

fn serve<B: KvBackend + Send + 'static>(
    engine: Engine<B>,
    tokenizer: Tokenizer,
    chat: ChatTemplate,
    stop: Vec<u32>,
    options: ServerOptions,
) -> axum::Router {
    tracing::info!(
        cache = %engine.scheduler().num_blocks(),
        block_size = engine.scheduler().block_size(),
        max_context = options.max_context,
        "engine ready"
    );
    router(engine, tokenizer, chat, stop, options)
}
