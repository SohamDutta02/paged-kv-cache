pub mod protocol;
pub mod sim;
pub mod ws;

use std::path::Path;
use std::sync::Arc;

use axum::routing::get;
use axum::Router;
use tokio::sync::Semaphore;
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;

use crate::sim::SimConfig;

#[derive(Clone)]
pub struct AppState {
    pub sim_config: SimConfig,
    /// One permit per open session.
    pub sessions: Arc<Semaphore>,
}

pub fn app(state: AppState, static_dir: &Path) -> Router {
    // Unknown paths get index.html so client-side routes survive a reload.
    let dashboard =
        ServeDir::new(static_dir).fallback(ServeFile::new(static_dir.join("index.html")));

    Router::new()
        .route("/api/ws", get(ws::upgrade))
        .route("/api/health", get(|| async { "ok" }))
        .fallback_service(dashboard)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
