//! Systems demo server: wraps the real `paged_kv_core::Scheduler` in an HTTP
//! API and a background loop, so the actual admission / batching /
//! copy-on-write / preemption logic runs continuously and is observable from
//! outside — no model, no GPU, no real tokens. Every event this server
//! reports is genuine scheduler behavior; only the "tokens" being scheduled
//! are synthetic.
//!
//! Routes:
//!   GET  /state     — full snapshot: pool state, running sequences, recent events
//!   POST /requests  — submit a synthetic request: {"prompt_len": N, "max_new_tokens": N}
//!   /                — static demo frontend (served from ./static)
//!
//! The background loop ticks the scheduler every `TICK_INTERVAL` and, every
//! few ticks, injects a small random request on its own — so the demo stays
//! visibly alive even with nobody interacting with it.

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use paged_kv_core::{CacheConfig, Scheduler, SeqId};
use rand::Rng;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

const NUM_BLOCKS: usize = 64;
const BLOCK_SIZE: usize = 8;
const MAX_BATCH_SIZE: usize = 6;
const TICK_INTERVAL: Duration = Duration::from_millis(400);
const MAX_EVENTS: usize = 200;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum EventKind {
    Admitted,
    Preempted,
    Finished,
    Cow { src_block: u32, dst_block: u32 },
}

#[derive(Debug, Clone, Serialize)]
struct Event {
    seq_id: u64,
    tick: u64,
    #[serde(flatten)]
    kind: EventKind,
}

struct AppState {
    scheduler: Mutex<Scheduler>,
    events: Mutex<VecDeque<Event>>,
    tick: AtomicU64,
    next_seq_id: AtomicU64,
}

#[derive(Debug, Serialize)]
struct BlockDto {
    id: u32,
    ref_count: u32,
}

#[derive(Debug, Serialize)]
struct RunningDto {
    id: u64,
    tokens: usize,
    generated: usize,
    max_new_tokens: usize,
    blocks: Vec<u32>,
}

#[derive(Debug, Serialize)]
struct StateResponse {
    tick: u64,
    num_blocks: usize,
    block_size: usize,
    pool_utilization: f64,
    num_waiting: usize,
    num_running: usize,
    blocks: Vec<BlockDto>,
    running: Vec<RunningDto>,
    events: Vec<Event>,
}

#[derive(Debug, Deserialize)]
struct NewRequestBody {
    prompt_len: usize,
    max_new_tokens: usize,
}

#[derive(Debug, Serialize)]
struct NewRequestResponse {
    id: u64,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

async fn get_state(State(state): State<Arc<AppState>>) -> Json<StateResponse> {
    let sched = state.scheduler.lock().await;
    let blocks = sched
        .block_snapshot()
        .into_iter()
        .map(|b| BlockDto { id: b.id.0, ref_count: b.ref_count })
        .collect();
    let running = sched
        .running_snapshot()
        .into_iter()
        .map(|r| RunningDto {
            id: r.id.0,
            tokens: r.tokens,
            generated: r.generated,
            max_new_tokens: r.max_new_tokens,
            blocks: r.blocks.iter().map(|b| b.0).collect(),
        })
        .collect();
    let pool_utilization = sched.pool_utilization();
    let num_waiting = sched.num_waiting();
    let num_running = sched.num_running();
    let num_blocks = sched.num_blocks();
    drop(sched);

    let events: Vec<Event> = state.events.lock().await.iter().cloned().collect();

    Json(StateResponse {
        tick: state.tick.load(Ordering::Relaxed),
        num_blocks,
        block_size: BLOCK_SIZE,
        pool_utilization,
        num_waiting,
        num_running,
        blocks,
        running,
        events,
    })
}

async fn post_request(
    State(state): State<Arc<AppState>>,
    Json(body): Json<NewRequestBody>,
) -> Result<Json<NewRequestResponse>, (StatusCode, Json<ErrorResponse>)> {
    let id = SeqId(state.next_seq_id.fetch_add(1, Ordering::Relaxed));
    let mut sched = state.scheduler.lock().await;
    match sched.add_request(id, body.prompt_len, body.max_new_tokens) {
        Ok(()) => Ok(Json(NewRequestResponse { id: id.0 })),
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse { error: e.to_string() }),
        )),
    }
}

async fn background_loop(state: Arc<AppState>) {
    loop {
        tokio::time::sleep(TICK_INTERVAL).await;
        let tick = state.tick.fetch_add(1, Ordering::Relaxed) + 1;

        let mut sched = state.scheduler.lock().await;

        // Keep the demo alive on its own: inject a small synthetic request
        // every few ticks so there's always something happening, on top of
        // whatever a visitor submits through POST /requests.
        if tick % 3 == 0 {
            let (prompt_len, max_new_tokens) = {
                let mut rng = rand::thread_rng();
                (rng.gen_range(4..48), rng.gen_range(4..32))
            };
            let id = SeqId(state.next_seq_id.fetch_add(1, Ordering::Relaxed));
            let _ = sched.add_request(id, prompt_len, max_new_tokens);
        }

        // Periodically fork a random running sequence — this is the only
        // thing that ever exercises copy-on-write. Without it, the demo
        // would only ever show admission/decode/preemption and never the
        // block-sharing behavior that's the actual point of this project.
        if tick % 5 == 0 {
            let running = sched.running_snapshot();
            if !running.is_empty() {
                let parent = running[rand::thread_rng().gen_range(0..running.len())].clone();
                let remaining = parent.max_new_tokens.saturating_sub(parent.generated);
                if remaining > 0 {
                    let child_id = SeqId(state.next_seq_id.fetch_add(1, Ordering::Relaxed));
                    let _ = sched.fork_sequence(parent.id, child_id, remaining);
                }
            }
        }

        let outcome = match sched.step() {
            Ok(o) => o,
            Err(e) => {
                eprintln!("scheduler.step() error (continuing): {e}");
                continue;
            }
        };
        drop(sched);

        let mut new_events = Vec::new();
        for id in &outcome.admitted {
            new_events.push(Event { seq_id: id.0, tick, kind: EventKind::Admitted });
        }
        for d in &outcome.decoded {
            if let Some(cow) = d.cow {
                new_events.push(Event {
                    seq_id: d.id.0,
                    tick,
                    kind: EventKind::Cow { src_block: cow.src.0, dst_block: cow.dst.0 },
                });
            }
        }
        for id in &outcome.preempted {
            new_events.push(Event { seq_id: id.0, tick, kind: EventKind::Preempted });
        }
        for id in &outcome.finished {
            new_events.push(Event { seq_id: id.0, tick, kind: EventKind::Finished });
        }

        if !new_events.is_empty() {
            let mut events = state.events.lock().await;
            for e in new_events {
                events.push_back(e);
            }
            while events.len() > MAX_EVENTS {
                events.pop_front();
            }
        }
    }
}

#[tokio::main]
async fn main() {
    // num_kv_heads/head_dim/num_layers/dtype_bytes are irrelevant here —
    // this demo never touches a KvBackend, only the scheduler's block
    // bookkeeping, so only num_blocks and block_size actually matter.
    let config = CacheConfig {
        num_blocks: NUM_BLOCKS,
        block_size: BLOCK_SIZE,
        num_kv_heads: 1,
        head_dim: 1,
        num_layers: 1,
        dtype_bytes: 4,
    };
    let scheduler = Scheduler::new(&config, MAX_BATCH_SIZE);

    let state = Arc::new(AppState {
        scheduler: Mutex::new(scheduler),
        events: Mutex::new(VecDeque::new()),
        tick: AtomicU64::new(0),
        next_seq_id: AtomicU64::new(1),
    });

   tokio::spawn(background_loop(state.clone()));

    // ServeDir resolves a relative path against the process's *current
    // working directory* at runtime, not the crate's own location — so
    // `ServeDir::new("static")` silently 404s on everything unless the
    // binary happens to be launched from exactly this crate's directory.
    // Defaulting to the compile-time-known crate path fixes that for local
    // `cargo run` regardless of cwd; STATIC_DIR lets the Docker image (which
    // controls its own layout) override it explicitly.
    let static_dir = std::env::var("STATIC_DIR")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/static").to_string());

    let app = Router::new()
        .route("/state", get(get_state))
        .route("/requests", post(post_request))
        .nest_service("/", ServeDir::new(static_dir))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    println!("paged-kv-server demo listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}