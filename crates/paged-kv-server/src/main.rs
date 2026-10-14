use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use paged_kv_server::sim::SimConfig;
use paged_kv_server::{app, AppState};
use tokio::sync::Semaphore;
use tracing_subscriber::EnvFilter;

const MAX_SESSIONS: usize = 100;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    // Render sets PORT; the Docker image sets STATIC_DIR.
    let port = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);
    let static_dir = std::env::var("STATIC_DIR").map_or_else(
        |_| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/dist")),
        PathBuf::from,
    );
    if !static_dir.join("index.html").exists() {
        tracing::warn!(
            dir = %static_dir.display(),
            "dashboard not built; run `npm run build` in web/"
        );
    }

    let state = AppState {
        sim_config: SimConfig::default(),
        sessions: Arc::new(Semaphore::new(MAX_SESSIONS)),
    };
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|e| panic!("cannot bind {addr}: {e}"));
    tracing::info!(%addr, "listening");

    axum::serve(listener, app(state, &static_dir))
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server error");
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to listen for Ctrl+C");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to listen for SIGTERM")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {}
        () = terminate => {}
    }
}
