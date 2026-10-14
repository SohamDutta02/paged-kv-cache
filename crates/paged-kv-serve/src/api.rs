use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream::{self, StreamExt};
use paged_kv_core::KvBackend;
use paged_kv_model::Engine;
use serde::Deserialize;
use serde_json::{json, Value};
use tokenizers::Tokenizer;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver};

use crate::checkpoint::{ChatMessage, ChatTemplate};
use crate::metrics::Metrics;
use crate::text::Detokenizer;
use crate::worker::{self, Event, FinishReason, Submit};

pub struct ServerOptions {
    /// Reported as `model` in every response.
    pub model_name: String,
    /// Prompt plus completion, in tokens.
    pub max_context: usize,
    /// Used when a request doesn't set `max_tokens`.
    pub default_max_tokens: usize,
    /// Requests beyond this many in flight get 429.
    pub max_in_flight: usize,
}

struct Shared {
    submit: Sender<Submit>,
    tokenizer: Arc<Tokenizer>,
    chat: ChatTemplate,
    metrics: Arc<Metrics>,
    options: ServerOptions,
    in_flight: AtomicUsize,
    next_id: AtomicU64,
}

type AppState = Arc<Shared>;

/// The HTTP API. Starts the engine thread, which stops when the router is dropped.
pub fn router<B: KvBackend + Send + 'static>(
    engine: Engine<B>,
    tokenizer: Tokenizer,
    chat: ChatTemplate,
    stop_tokens: Vec<u32>,
    options: ServerOptions,
) -> Router {
    let metrics = Arc::new(Metrics::new(engine.scheduler().max_batch_size()));
    let engine = engine.stop_at(&stop_tokens);
    let submit = worker::spawn(engine, stop_tokens, metrics.clone());
    let state = Arc::new(Shared {
        submit,
        tokenizer: Arc::new(tokenizer),
        chat,
        metrics,
        options,
        in_flight: AtomicUsize::new(0),
        next_id: AtomicU64::new(0),
    });
    Router::new()
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(models))
        .route("/health", get(|| async { "ok" }))
        .route("/metrics", get(metrics_text))
        .with_state(state)
}

struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let kind = match self.status {
            StatusCode::BAD_REQUEST => "invalid_request_error",
            StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
            _ => "server_error",
        };
        let body = json!({ "error": { "message": self.message, "type": kind, "code": null } });
        (self.status, Json(body)).into_response()
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Prompt {
    Text(String),
    Tokens(Vec<u32>),
}

/// Fields every request may set. Only greedy decoding is implemented, so
/// sampling settings are accepted only at values that mean greedy.
#[derive(Deserialize)]
struct Sampling {
    temperature: Option<f32>,
    top_p: Option<f32>,
    n: Option<usize>,
    stop: Option<Value>,
}

impl Sampling {
    fn check(&self) -> Result<(), ApiError> {
        if self.temperature.is_some_and(|t| t != 0.0) {
            return Err(ApiError::invalid(
                "only greedy decoding is supported: omit temperature or set it to 0",
            ));
        }
        if self.top_p.is_some_and(|p| p != 1.0) {
            return Err(ApiError::invalid(
                "only greedy decoding is supported: omit top_p or set it to 1",
            ));
        }
        if self.n.is_some_and(|n| n != 1) {
            return Err(ApiError::invalid("n must be 1"));
        }
        let no_stop = match &self.stop {
            None | Some(Value::Null) => true,
            Some(Value::Array(list)) => list.is_empty(),
            _ => false,
        };
        if !no_stop {
            return Err(ApiError::invalid(
                "stop sequences are not supported; generation ends at the model's end-of-sequence tokens",
            ));
        }
        Ok(())
    }
}

#[derive(Deserialize)]
struct CompletionRequest {
    prompt: Prompt,
    max_tokens: Option<usize>,
    #[serde(default)]
    stream: bool,
    #[serde(flatten)]
    sampling: Sampling,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Deserialize)]
struct ContentPart {
    text: Option<String>,
}

#[derive(Deserialize)]
struct Message {
    role: String,
    content: Content,
}

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<Message>,
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    stream: bool,
    #[serde(flatten)]
    sampling: Sampling,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Completion,
    Chat,
}

async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    req.sampling.check()?;
    let prompt = match req.prompt {
        Prompt::Tokens(tokens) => tokens,
        Prompt::Text(text) => encode(&state, &text)?,
    };
    generate(state, Kind::Completion, prompt, req.max_tokens, req.stream).await
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Response, ApiError> {
    req.sampling.check()?;
    let messages: Vec<ChatMessage> = req
        .messages
        .into_iter()
        .map(|m| ChatMessage {
            role: m.role,
            content: match m.content {
                Content::Text(text) => text,
                Content::Parts(parts) => parts.into_iter().filter_map(|p| p.text).collect(),
            },
        })
        .collect();
    let text = state
        .chat
        .render(&messages)
        .map_err(|e| ApiError::invalid(format!("chat template: {e}")))?;
    let prompt = encode(&state, &text)?;
    let max_tokens = req.max_completion_tokens.or(req.max_tokens);
    generate(state, Kind::Chat, prompt, max_tokens, req.stream).await
}

fn encode(state: &Shared, text: &str) -> Result<Vec<u32>, ApiError> {
    state
        .tokenizer
        .encode(text, false)
        .map(|e| e.get_ids().to_vec())
        .map_err(|e| ApiError::invalid(format!("tokenizer: {e}")))
}

/// Decrements the in-flight count however the request ends.
struct InFlight(AppState);

impl Drop for InFlight {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

async fn generate(
    state: AppState,
    kind: Kind,
    prompt: Vec<u32>,
    max_tokens: Option<usize>,
    stream: bool,
) -> Result<Response, ApiError> {
    let options = &state.options;
    let max_tokens = max_tokens.unwrap_or(options.default_max_tokens);
    if prompt.is_empty() || max_tokens == 0 {
        return Err(ApiError::invalid(
            "the prompt and max_tokens must not be empty",
        ));
    }
    if prompt.len() + max_tokens > options.max_context {
        return Err(ApiError::invalid(format!(
            "this server's maximum context length is {} tokens, but {} were requested ({} in the prompt, {} for the completion)",
            options.max_context,
            prompt.len() + max_tokens,
            prompt.len(),
            max_tokens
        )));
    }
    if state.in_flight.fetch_add(1, Ordering::Relaxed) >= options.max_in_flight {
        state.in_flight.fetch_sub(1, Ordering::Relaxed);
        state.metrics.request_done("rejected");
        return Err(ApiError {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: format!("{} requests are already in flight", options.max_in_flight),
        });
    }
    let guard = InFlight(state.clone());

    let (events, rx) = unbounded_channel();
    let prompt_tokens = prompt.len();
    state
        .submit
        .send(Submit {
            prompt,
            max_tokens,
            events,
        })
        .map_err(|_| ApiError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "the engine has stopped".into(),
        })?;

    let reply = Reply {
        kind,
        id: format!(
            "{}-{}",
            if kind == Kind::Chat {
                "chatcmpl"
            } else {
                "cmpl"
            },
            state.next_id.fetch_add(1, Ordering::Relaxed)
        ),
        created: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()),
        model: options.model_name.clone(),
    };
    if stream {
        Ok(stream_response(state.clone(), reply, rx, guard))
    } else {
        collect_response(&state, reply, rx, prompt_tokens).await
    }
}

/// What every response and chunk repeats.
struct Reply {
    kind: Kind,
    id: String,
    created: u64,
    model: String,
}

impl Reply {
    fn object(&self, chunk: bool) -> &'static str {
        match (self.kind, chunk) {
            (Kind::Completion, _) => "text_completion",
            (Kind::Chat, false) => "chat.completion",
            (Kind::Chat, true) => "chat.completion.chunk",
        }
    }

    fn body(&self, chunk: bool, choice: Value) -> Value {
        json!({
            "id": self.id,
            "object": self.object(chunk),
            "created": self.created,
            "model": self.model,
            "choices": [choice],
        })
    }

    fn chunk(&self, text: Option<&str>, finish: Option<FinishReason>) -> Value {
        let finish = finish.map(FinishReason::as_str);
        let choice = match self.kind {
            Kind::Completion => json!({
                "index": 0, "text": text.unwrap_or(""), "logprobs": null, "finish_reason": finish,
            }),
            Kind::Chat => json!({
                "index": 0,
                "delta": text.map_or(json!({}), |t| json!({ "content": t })),
                "logprobs": null,
                "finish_reason": finish,
            }),
        };
        self.body(true, choice)
    }
}

async fn collect_response(
    state: &Shared,
    reply: Reply,
    mut rx: UnboundedReceiver<Event>,
    prompt_tokens: usize,
) -> Result<Response, ApiError> {
    let mut tokens = Vec::new();
    let finish = loop {
        match rx.recv().await {
            Some(Event::Token(t)) => tokens.push(t),
            Some(Event::Done(reason)) => break reason,
            Some(Event::Failed(message)) => return Err(ApiError::invalid(message)),
            None => {
                return Err(ApiError {
                    status: StatusCode::INTERNAL_SERVER_ERROR,
                    message: "the engine dropped the request".into(),
                })
            }
        }
    };
    let text = state
        .tokenizer
        .decode(&tokens, true)
        .map_err(|e| ApiError::invalid(format!("tokenizer: {e}")))?;
    let choice = match reply.kind {
        Kind::Completion => json!({
            "index": 0, "text": text, "logprobs": null, "finish_reason": finish.as_str(),
        }),
        Kind::Chat => json!({
            "index": 0,
            "message": { "role": "assistant", "content": text },
            "logprobs": null,
            "finish_reason": finish.as_str(),
        }),
    };
    let mut body = reply.body(false, choice);
    body["usage"] = json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": tokens.len(),
        "total_tokens": prompt_tokens + tokens.len(),
    });
    Ok(Json(body).into_response())
}

enum Phase {
    /// Chat streams open with the assistant's role.
    Role,
    Tokens,
    Done,
}

struct Streaming {
    reply: Reply,
    rx: UnboundedReceiver<Event>,
    text: Detokenizer,
    phase: Phase,
    _in_flight: InFlight,
}

/// Server-sent events in OpenAI's chunk format, ending with `[DONE]`.
/// Dropping the stream when the client disconnects cancels the request.
fn stream_response(
    state: AppState,
    reply: Reply,
    rx: UnboundedReceiver<Event>,
    in_flight: InFlight,
) -> Response {
    let phase = match reply.kind {
        Kind::Chat => Phase::Role,
        Kind::Completion => Phase::Tokens,
    };
    let start = Streaming {
        reply,
        rx,
        text: Detokenizer::new(state.tokenizer.clone()),
        phase,
        _in_flight: in_flight,
    };
    let data = |v: Value| SseEvent::default().data(v.to_string());
    let events = stream::unfold(start, move |mut s| async move {
        let events = match s.phase {
            Phase::Done => return None,
            Phase::Role => {
                s.phase = Phase::Tokens;
                let mut chunk = s.reply.chunk(Some(""), None);
                chunk["choices"][0]["delta"] = json!({ "role": "assistant", "content": "" });
                vec![data(chunk)]
            }
            Phase::Tokens => match s.rx.recv().await {
                Some(Event::Token(token)) => {
                    let delta = s.text.push(token);
                    if delta.is_empty() {
                        Vec::new()
                    } else {
                        vec![data(s.reply.chunk(Some(&delta), None))]
                    }
                }
                Some(Event::Done(reason)) => {
                    s.phase = Phase::Done;
                    vec![
                        data(s.reply.chunk(None, Some(reason))),
                        SseEvent::default().data("[DONE]"),
                    ]
                }
                Some(Event::Failed(message)) => {
                    s.phase = Phase::Done;
                    vec![data(json!({ "error": { "message": message, "type": "server_error" } }))]
                }
                None => {
                    s.phase = Phase::Done;
                    vec![data(json!({ "error": { "message": "the engine dropped the request", "type": "server_error" } }))]
                }
            },
        };
        Some((events, s))
    })
    .flat_map(|events| stream::iter(events.into_iter().map(Ok::<_, Infallible>)));
    Sse::new(events)
        .keep_alive(KeepAlive::default())
        .into_response()
}

async fn models(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.options.model_name,
            "object": "model",
            "created": 0,
            "owned_by": "paged-kv-cache",
        }],
    }))
}

async fn metrics_text(State(state): State<AppState>) -> Response {
    (
        [(
            header::CONTENT_TYPE,
            "application/openmetrics-text; version=1.0.0; charset=utf-8",
        )],
        state.metrics.encode(),
    )
        .into_response()
}
