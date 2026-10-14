//! The HTTP API checked against the engine run directly, on a tiny model with
//! random weights and a word-level tokenizer.

use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use candle_nn::Activation;
use http_body_util::BodyExt;
use paged_kv_core::{CpuBackend, SeqId};
use paged_kv_model::{dummy_weights, Config, Engine, PagedQwen2};
use paged_kv_serve::{router, ChatTemplate, ServerOptions};
use serde_json::{json, Value};
use tokenizers::Tokenizer;
use tower::ServiceExt;

const WORDS: u32 = 97;
const EOS: u32 = 99;

fn config() -> Config {
    Config {
        vocab_size: 100,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        max_position_embeddings: 256,
        sliding_window: 256,
        max_window_layers: 2,
        tie_word_embeddings: true,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        use_sliding_window: false,
        hidden_act: Activation::Silu,
    }
}

fn engine(num_blocks: usize) -> Engine<CpuBackend> {
    let cfg = config();
    let model = PagedQwen2::new(&cfg, dummy_weights(&cfg, 7).unwrap()).unwrap();
    let cache = CpuBackend::new(model.cache_config(num_blocks, 4));
    Engine::new(model, cache, 4, true)
}

/// Words `t0`..`t96`, then ChatML's markers and an end-of-text token.
fn tokenizer() -> Tokenizer {
    let vocab: serde_json::Map<String, Value> =
        (0..WORDS).map(|i| (format!("t{i}"), json!(i))).collect();
    let special = |id: u32, content: &str| {
        json!({ "id": id, "content": content, "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true })
    };
    let spec = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [special(97, "<|im_start|>"), special(98, "<|im_end|>"), special(EOS, "<|endoftext|>")],
        "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": null,
        "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "t0" },
    });
    spec.to_string().parse().unwrap()
}

fn app(num_blocks: usize, max_in_flight: usize) -> Router {
    router(
        engine(num_blocks),
        tokenizer(),
        ChatTemplate::chatml(),
        vec![EOS],
        ServerOptions {
            model_name: "tiny-qwen2".into(),
            max_context: 128,
            default_max_tokens: 8,
            max_in_flight,
        },
    )
}

/// What the engine generates for `prompt` when driven directly.
fn direct(prompt: &[u32], max_tokens: usize) -> Vec<u32> {
    let mut engine = engine(64).stop_at(&[EOS]);
    engine
        .add_request(SeqId(0), prompt.to_vec(), max_tokens)
        .unwrap();
    loop {
        if let Some(f) = engine.step().unwrap().finished.pop() {
            let mut tokens = f.tokens;
            if tokens.last() == Some(&EOS) {
                tokens.pop();
            }
            return tokens;
        }
    }
}

fn decode(tokens: &[u32]) -> String {
    tokenizer().decode(tokens, true).unwrap()
}

async fn post(app: &Router, path: &str, body: Value) -> (StatusCode, String) {
    let request = Request::post(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

async fn get(app: &Router, path: &str) -> String {
    let request = Request::get(path).body(Body::empty()).unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8(bytes.to_vec()).unwrap()
}

/// The `data:` payloads of a server-sent event stream.
fn sse_data(body: &str) -> Vec<String> {
    body.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .map(str::to_string)
        .collect()
}

fn prompt(salt: u32, len: u32) -> Vec<u32> {
    (0..len).map(|t| (t * 31 + salt * 17 + 5) % WORDS).collect()
}

/// A sample from `/metrics`, e.g. `paged_kv_requests_total{status="finished"}`.
fn metric(text: &str, name: &str) -> Option<f64> {
    text.lines()
        .find_map(|l| l.strip_prefix(name)?.strip_prefix(' ')?.parse().ok())
}

#[tokio::test]
async fn completion_matches_the_engine() {
    let app = app(64, 16);
    let tokens = prompt(1, 9);
    let (status, body) = post(
        &app,
        "/v1/completions",
        json!({ "model": "anything", "prompt": tokens, "max_tokens": 12 }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let body: Value = serde_json::from_str(&body).unwrap();
    let expected = direct(&tokens, 12);

    assert_eq!(body["object"], "text_completion");
    assert_eq!(body["model"], "tiny-qwen2");
    assert_eq!(body["choices"][0]["text"], decode(&expected));
    let finish = if expected.len() < 12 {
        "stop"
    } else {
        "length"
    };
    assert_eq!(body["choices"][0]["finish_reason"], finish);
    assert_eq!(body["usage"]["prompt_tokens"], 9);
    assert_eq!(body["usage"]["completion_tokens"], expected.len());
}

#[tokio::test]
async fn text_prompts_are_tokenized() {
    let app = app(64, 16);
    let (status, body) = post(
        &app,
        "/v1/completions",
        json!({ "prompt": "t5 t6 t7 t8", "max_tokens": 6 }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let body: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        body["choices"][0]["text"],
        decode(&direct(&[5, 6, 7, 8], 6))
    );
}

#[tokio::test]
async fn streaming_sends_the_same_text_in_chunks() {
    let app = app(64, 16);
    let tokens = prompt(2, 7);
    let (status, body) = post(
        &app,
        "/v1/completions",
        json!({ "prompt": tokens, "max_tokens": 10, "stream": true }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let events = sse_data(&body);
    assert_eq!(events.last().map(String::as_str), Some("[DONE]"));

    let chunks: Vec<Value> = events[..events.len() - 1]
        .iter()
        .map(|e| serde_json::from_str(e).unwrap())
        .collect();
    let text: String = chunks
        .iter()
        .map(|c| c["choices"][0]["text"].as_str().unwrap())
        .collect();
    assert_eq!(text, decode(&direct(&tokens, 10)));
    assert!(
        chunks.len() > 2,
        "expected one chunk per token, got {}",
        chunks.len()
    );
    assert!(!chunks.last().unwrap()["choices"][0]["finish_reason"].is_null());
}

#[tokio::test]
async fn chat_completions_render_the_template() {
    let app = app(64, 16);
    let messages = json!([
        { "role": "system", "content": "t1 t2" },
        { "role": "user", "content": [{ "type": "text", "text": "t3 t4" }] },
    ]);
    let (status, body) = post(
        &app,
        "/v1/chat/completions",
        json!({ "messages": messages, "max_tokens": 5 }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let body: Value = serde_json::from_str(&body).unwrap();

    // ChatML around the messages; words outside the vocabulary become t0.
    let rendered = "<|im_start|>system\nt1 t2<|im_end|>\n<|im_start|>user\nt3 t4<|im_end|>\n<|im_start|>assistant\n";
    let ids = tokenizer()
        .encode(rendered, false)
        .unwrap()
        .get_ids()
        .to_vec();
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        decode(&direct(&ids, 5))
    );

    let (_, streamed) = post(
        &app,
        "/v1/chat/completions",
        json!({ "messages": messages, "max_tokens": 5, "stream": true }),
    )
    .await;
    let events = sse_data(&streamed);
    let first: Value = serde_json::from_str(&events[0]).unwrap();
    assert_eq!(first["object"], "chat.completion.chunk");
    assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
    let content: String = events[1..events.len() - 1]
        .iter()
        .map(|e| serde_json::from_str::<Value>(e).unwrap())
        .filter_map(|c| {
            c["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_string)
        })
        .collect();
    assert_eq!(content, body["choices"][0]["message"]["content"]);
}

#[tokio::test]
async fn concurrent_requests_share_batches_and_match_sequential_runs() {
    let app = app(64, 16);
    let shared: Vec<u32> = prompt(9, 8);
    let prompts: Vec<Vec<u32>> = (0..6)
        .map(|i| [shared.clone(), prompt(i, 1 + i)].concat())
        .collect();

    let mut handles = Vec::new();
    for p in &prompts {
        let app = app.clone();
        let body = json!({ "prompt": p, "max_tokens": 10 });
        handles.push(tokio::spawn(async move {
            post(&app, "/v1/completions", body).await
        }));
    }
    for (p, handle) in prompts.iter().zip(handles) {
        let (status, body) = handle.await.unwrap();
        assert_eq!(status, StatusCode::OK, "{body}");
        let body: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(body["choices"][0]["text"], decode(&direct(p, 10)));
    }

    let metrics = get(&app, "/metrics").await;
    let steps = metric(&metrics, "paged_kv_batch_size_count").unwrap();
    let sequences = metric(&metrics, "paged_kv_batch_size_sum").unwrap();
    assert!(sequences / steps > 1.5, "mean batch {}", sequences / steps);
    assert!(metric(&metrics, "paged_kv_prefill_tokens_cached_total").unwrap() > 0.0);
    assert_eq!(
        metric(&metrics, r#"paged_kv_requests_total{status="finished"}"#),
        Some(6.0)
    );
}

#[tokio::test]
async fn unsupported_or_oversized_requests_are_rejected() {
    let app = app(64, 16);
    for body in [
        json!({ "prompt": [1, 2], "temperature": 0.7 }),
        json!({ "prompt": [1, 2], "n": 2 }),
        json!({ "prompt": [1, 2], "stop": ["t5"] }),
        json!({ "prompt": [1, 2], "max_tokens": 200 }),
        json!({ "prompt": [] }),
    ] {
        let (status, text) = post(&app, "/v1/completions", body.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}: {text}");
        let error: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(error["error"]["type"], "invalid_request_error");
    }
    // Greedy-equivalent settings are fine.
    let ok = json!({ "prompt": [1, 2], "temperature": 0, "top_p": 1, "n": 1, "stop": null, "max_tokens": 2 });
    assert_eq!(post(&app, "/v1/completions", ok).await.0, StatusCode::OK);
}

#[tokio::test]
async fn a_disconnected_stream_is_cancelled_and_frees_its_blocks() {
    let app = app(64, 16);
    let request = Request::post("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({ "prompt": prompt(3, 6), "max_tokens": 100, "stream": true }).to_string(),
        ))
        .unwrap();
    let mut body = app.clone().oneshot(request).await.unwrap().into_body();
    body.frame().await.unwrap().unwrap();
    drop(body);

    for _ in 0..200 {
        let metrics = get(&app, "/metrics").await;
        let cancelled = metric(&metrics, r#"paged_kv_requests_total{status="cancelled"}"#);
        let running = metric(&metrics, "paged_kv_running_requests");
        if cancelled == Some(1.0) && running == Some(0.0) {
            let utilization = metric(&metrics, "paged_kv_kv_cache_utilization").unwrap();
            assert_eq!(utilization, 0.0);
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!(
        "the request was not cancelled:\n{}",
        get(&app, "/metrics").await
    );
}

#[tokio::test]
async fn requests_past_the_in_flight_limit_get_429() {
    let app = app(64, 1);
    let slow = Request::post("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({ "prompt": prompt(4, 6), "max_tokens": 100, "stream": true }).to_string(),
        ))
        .unwrap();
    let held = app.clone().oneshot(slow).await.unwrap();

    let (status, body) = post(&app, "/v1/completions", json!({ "prompt": [1, 2] })).await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS, "{body}");
    drop(held);
}

#[tokio::test]
async fn models_and_health() {
    let app = app(64, 16);
    let models: Value = serde_json::from_str(&get(&app, "/v1/models").await).unwrap();
    assert_eq!(models["data"][0]["id"], "tiny-qwen2");
    assert_eq!(get(&app, "/health").await, "ok");
}
