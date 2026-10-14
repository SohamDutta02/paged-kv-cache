use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use futures_util::{SinkExt, StreamExt};
use paged_kv_server::sim::SimConfig;
use paged_kv_server::ws::CLOSE_TRY_AGAIN_LATER;
use paged_kv_server::{app, AppState};
use serde_json::{json, Value};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Semaphore;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};
use tower::ServiceExt;

type Client = WebSocketStream<MaybeTlsStream<TcpStream>>;

fn state(max_sessions: usize) -> AppState {
    AppState {
        // Long enough that no tick fires during a test.
        sim_config: SimConfig {
            tick: Duration::from_secs(3600),
            ..SimConfig::default()
        },
        sessions: Arc::new(Semaphore::new(max_sessions)),
    }
}

async fn serve(max_sessions: usize) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = app(state(max_sessions), Path::new("/nonexistent"));
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    addr
}

async fn connect(addr: SocketAddr) -> Client {
    connect_async(format!("ws://{addr}/api/ws"))
        .await
        .unwrap()
        .0
}

async fn recv(ws: &mut Client) -> Value {
    loop {
        let msg = tokio::time::timeout(Duration::from_secs(5), ws.next())
            .await
            .expect("timed out waiting for a message")
            .expect("connection closed")
            .unwrap();
        if let Message::Text(text) = msg {
            return serde_json::from_str(&text).unwrap();
        }
    }
}

async fn command(ws: &mut Client, id: u32, command: Value) {
    let msg = json!({ "id": id, "command": command }).to_string();
    ws.send(Message::Text(msg.into())).await.unwrap();
}

#[tokio::test]
async fn a_session_starts_with_init_and_acknowledges_commands() {
    let addr = serve(4).await;
    let mut ws = connect(addr).await;

    let init = recv(&mut ws).await;
    assert_eq!(init["type"], "init");
    assert_eq!(init["config"]["num_blocks"], 64);
    assert_eq!(init["snapshot"]["tick"], 0);

    command(
        &mut ws,
        6,
        json!({"type": "add_request", "prompt_len": 20, "max_new_tokens": 5}),
    )
    .await;
    let ack = recv(&mut ws).await;
    assert_eq!(ack["type"], "ack");
    assert_eq!(ack["id"], 6);
    assert_eq!(ack["result"]["status"], "ok");
    let seq_id = ack["result"]["seq_id"].clone();

    let state = recv(&mut ws).await;
    assert_eq!(state["type"], "state");
    assert_eq!(state["snapshot"]["waiting"][0]["id"], seq_id);
    assert_eq!(state["snapshot"]["waiting"][0]["origin"], "user");
}

#[tokio::test]
async fn an_unreadable_message_gets_an_error_and_the_session_continues() {
    let addr = serve(4).await;
    let mut ws = connect(addr).await;
    recv(&mut ws).await;

    ws.send(Message::Text("{not json".into())).await.unwrap();
    let error = recv(&mut ws).await;
    assert_eq!(error["type"], "error");

    command(&mut ws, 1, json!({"type": "pause"})).await;
    assert_eq!(recv(&mut ws).await["result"]["status"], "ok");
}

#[tokio::test]
async fn sessions_are_isolated() {
    let addr = serve(4).await;
    let mut a = connect(addr).await;
    let mut b = connect(addr).await;
    recv(&mut a).await;
    recv(&mut b).await;

    command(
        &mut a,
        1,
        json!({"type": "add_request", "prompt_len": 4, "max_new_tokens": 4}),
    )
    .await;
    recv(&mut a).await;
    assert_eq!(
        recv(&mut a).await["snapshot"]["waiting"]
            .as_array()
            .unwrap()
            .len(),
        1
    );

    command(&mut b, 1, json!({"type": "pause"})).await;
    recv(&mut b).await;
    let b_state = recv(&mut b).await;
    assert!(b_state["snapshot"]["waiting"]
        .as_array()
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn sessions_beyond_the_limit_are_turned_away() {
    let addr = serve(1).await;
    let mut first = connect(addr).await;
    recv(&mut first).await;

    let mut second = connect(addr).await;
    let msg = tokio::time::timeout(Duration::from_secs(5), second.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let Message::Close(Some(frame)) = msg else {
        panic!("expected a close frame, got {msg:?}")
    };
    assert_eq!(u16::from(frame.code), CLOSE_TRY_AGAIN_LATER);

    drop(first);
    tokio::time::sleep(Duration::from_millis(100)).await;
    let mut third = connect(addr).await;
    assert_eq!(
        recv(&mut third).await["type"],
        "init",
        "permit released on disconnect"
    );
}

fn temp_dashboard() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("paged-kv-dashboard-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("index.html"),
        "<!doctype html><title>dashboard</title>",
    )
    .unwrap();
    dir
}

async fn get(path: &str, static_dir: &Path) -> (StatusCode, String) {
    let res = app(state(1), static_dir)
        .oneshot(Request::get(path).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = res.status();
    let body = to_bytes(res.into_body(), 1 << 20).await.unwrap();
    (status, String::from_utf8(body.to_vec()).unwrap())
}

#[tokio::test]
async fn serves_health_and_the_dashboard_with_spa_fallback() {
    let dir = temp_dashboard();
    assert_eq!(
        get("/api/health", &dir).await,
        (StatusCode::OK, "ok".into())
    );

    let (status, body) = get("/", &dir).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("<title>dashboard</title>"));

    let (status, body) = get("/some/client/route", &dir).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("<title>dashboard</title>"));
}
