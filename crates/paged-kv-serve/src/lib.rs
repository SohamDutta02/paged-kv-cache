//! An OpenAI-compatible HTTP server over `paged-kv-model`.
//!
//! `/v1/completions` and `/v1/chat/completions` (with `stream: true` for
//! server-sent events), `/v1/models`, `/health`, and Prometheus `/metrics`.
//! One thread owns the engine and runs a step whenever any request is live;
//! requests that arrive mid-step join the next one, which is continuous
//! batching. Handlers talk to it over channels, and a client that disconnects
//! has its request cancelled and its blocks freed.

mod api;
mod checkpoint;
mod metrics;
mod text;
mod worker;

pub use api::{router, ServerOptions};
pub use checkpoint::{stop_tokens, ChatMessage, ChatTemplate};
pub use metrics::Metrics;
