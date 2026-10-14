use std::time::Duration;

use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::Response;
use tokio::sync::OwnedSemaphorePermit;
use tokio::time::{interval, sleep, Instant, MissedTickBehavior};

use crate::protocol::{ClientMessage, ServerMessage};
use crate::sim::Simulation;
use crate::AppState;

const MAX_MESSAGE_BYTES: usize = 16 * 1024;
const IDLE_TIMEOUT: Duration = Duration::from_secs(30 * 60);

pub const CLOSE_INTERNAL_ERROR: u16 = 1011;
pub const CLOSE_TRY_AGAIN_LATER: u16 = 1013;
/// Application-defined (RFC 6455 reserves 4000-4999 for applications).
pub const CLOSE_IDLE: u16 = 4000;

pub async fn upgrade(ws: WebSocketUpgrade, State(state): State<AppState>) -> Response {
    let permit = state.sessions.clone().try_acquire_owned().ok();
    ws.max_message_size(MAX_MESSAGE_BYTES)
        .on_upgrade(move |socket| async move {
            match permit {
                Some(permit) => {
                    let sim = Simulation::new(state.sim_config, rand::random());
                    run(socket, sim, permit).await;
                }
                None => {
                    close(
                        socket,
                        CLOSE_TRY_AGAIN_LATER,
                        "Too many sessions are open. Try again in a minute.",
                    )
                    .await;
                }
            }
        })
}

/// The permit is held for the life of the session.
async fn run(mut socket: WebSocket, mut sim: Simulation, _permit: OwnedSemaphorePermit) {
    if send(&mut socket, &sim.init_message()).await.is_err() {
        return;
    }

    let mut ticker = interval(sim.tick_interval());
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    ticker.reset();
    let idle = sleep(IDLE_TIMEOUT);
    tokio::pin!(idle);

    loop {
        tokio::select! {
            _ = ticker.tick(), if !sim.is_paused() => {
                let msg = match sim.advance() {
                    Ok(msg) => msg,
                    Err(e) => {
                        tracing::error!(error = %e, "scheduler failed, closing session");
                        close(socket, CLOSE_INTERNAL_ERROR, "The scheduler hit an internal error.").await;
                        return;
                    }
                };
                if send(&mut socket, &msg).await.is_err() {
                    return;
                }
            }
            incoming = socket.recv() => {
                let text = match incoming {
                    Some(Ok(Message::Text(text))) => text,
                    Some(Ok(Message::Close(_)) | Err(_)) | None => return,
                    Some(Ok(_)) => continue,
                };
                idle.as_mut().reset(Instant::now() + IDLE_TIMEOUT);

                let was_paused = sim.is_paused();
                let replies = match serde_json::from_str::<ClientMessage>(&text) {
                    Ok(msg) => sim.handle(msg),
                    Err(e) => vec![ServerMessage::Error {
                        message: format!("Unreadable message: {e}"),
                    }],
                };
                // Otherwise the first tick after resuming fires immediately.
                if was_paused && !sim.is_paused() {
                    ticker.reset();
                }
                for reply in &replies {
                    if send(&mut socket, reply).await.is_err() {
                        return;
                    }
                }
            }
            () = &mut idle => {
                close(socket, CLOSE_IDLE, "Closed after 30 minutes without input.").await;
                return;
            }
        }
    }
}

async fn send(socket: &mut WebSocket, msg: &ServerMessage) -> Result<(), axum::Error> {
    let text = serde_json::to_string(msg).expect("protocol types always serialize");
    socket.send(Message::Text(text)).await
}

async fn close(mut socket: WebSocket, code: u16, reason: &'static str) {
    let frame = CloseFrame {
        code,
        reason: reason.into(),
    };
    let _ = socket.send(Message::Close(Some(frame))).await;
}
