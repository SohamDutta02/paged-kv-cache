//! Dashboard wire protocol: one JSON message per WebSocket text frame.
//!
//! `cargo test` regenerates the TypeScript copies in `web/src/api/generated/`,
//! and CI fails if they are stale. Numbers are `u32` so they map to a plain
//! TypeScript `number`.

use serde::{Deserialize, Serialize};
use ts_rs::TS;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum Origin {
    Traffic,
    User,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS)]
#[ts(export)]
pub struct PoolConfig {
    pub num_blocks: u32,
    /// Tokens per block.
    pub block_size: u32,
    pub max_batch_size: u32,
    pub max_waiting: u32,
    pub tick_ms: u32,
    /// Lengths of the shared system prompts a request can start with,
    /// indexed by system prompt id.
    pub system_prompts: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS)]
#[ts(export)]
pub struct SequenceView {
    pub id: u32,
    pub parent: Option<u32>,
    pub origin: Origin,
    pub system_prompt: Option<u32>,
    /// Prompt tokens served from the prefix cache at its latest admission.
    pub cached_tokens: u32,
    pub tokens: u32,
    pub generated: u32,
    pub max_new_tokens: u32,
    /// Physical block ids in logical order.
    pub blocks: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS)]
#[ts(export)]
pub struct WaitingView {
    pub id: u32,
    pub parent: Option<u32>,
    pub origin: Origin,
    pub system_prompt: Option<u32>,
    /// Tokens to prefill on admission; includes generated tokens if preempted.
    pub context_len: u32,
    pub max_new_tokens: u32,
    pub preempted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS)]
#[ts(export)]
pub struct Metrics {
    pub tick: u32,
    pub used_blocks: u32,
    /// Blocks with more than one owner.
    pub shared_blocks: u32,
    /// Cached blocks no sequence owns, kept for reuse until evicted.
    pub evictable_blocks: u32,
    pub running: u32,
    pub waiting: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, TS)]
#[ts(export)]
pub struct Totals {
    pub submitted: u32,
    pub admitted: u32,
    pub finished: u32,
    pub preempted: u32,
    pub forks: u32,
    pub cow_copies: u32,
    pub tokens_generated: u32,
    pub prompt_tokens_cached: u32,
    pub prompt_tokens_computed: u32,
    pub cache_evictions: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS)]
#[ts(export)]
pub struct Snapshot {
    pub tick: u32,
    pub paused: bool,
    pub traffic: bool,
    pub prefix_caching: bool,
    /// Indexed by block id.
    pub ref_counts: Vec<u32>,
    /// Blocks holding prefix-cached content, owned or not.
    pub cached_blocks: Vec<u32>,
    pub running: Vec<SequenceView>,
    /// Front of the queue first.
    pub waiting: Vec<WaitingView>,
    pub metrics: Metrics,
    pub totals: Totals,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS)]
#[serde(tag = "type", rename_all = "snake_case")]
#[ts(export)]
pub enum EventKind {
    Submitted {
        origin: Origin,
        system_prompt: Option<u32>,
        prompt_len: u32,
        max_new_tokens: u32,
    },
    Forked {
        parent: u32,
    },
    Admitted {
        cached_tokens: u32,
        computed_tokens: u32,
    },
    Cow {
        src_block: u32,
        dst_block: u32,
    },
    Preempted,
    Finished,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS)]
#[ts(export)]
pub struct Event {
    pub tick: u32,
    pub seq_id: u32,
    pub kind: EventKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS)]
#[serde(tag = "type", rename_all = "snake_case")]
#[ts(export)]
pub enum ServerMessage {
    /// Sent on connect and after a reset; replaces all client state.
    Init {
        config: PoolConfig,
        snapshot: Snapshot,
        history: Vec<Metrics>,
        events: Vec<Event>,
    },
    /// The scheduler stepped; `snapshot.metrics` is a new history point.
    Tick {
        snapshot: Snapshot,
        events: Vec<Event>,
    },
    /// State changed without a step, so no new history point.
    State {
        snapshot: Snapshot,
        events: Vec<Event>,
    },
    Ack {
        id: u32,
        result: CommandResult,
    },
    /// The client sent a message the server could not parse.
    Error {
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS)]
#[serde(tag = "status", rename_all = "snake_case")]
#[ts(export)]
pub enum CommandResult {
    /// `seq_id` is set for commands that create a sequence.
    Ok {
        seq_id: Option<u32>,
    },
    Rejected {
        message: String,
    },
}

/// `id` is chosen by the client and echoed back in the matching `Ack`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, TS)]
#[ts(export)]
pub struct ClientMessage {
    pub id: u32,
    pub command: Command,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, TS)]
#[serde(tag = "type", rename_all = "snake_case")]
#[ts(export)]
pub enum Command {
    /// `prompt_len` counts the tokens after the system prompt, if any.
    AddRequest {
        system_prompt: Option<u32>,
        prompt_len: u32,
        max_new_tokens: u32,
    },
    Fork {
        seq_id: u32,
    },
    Pause,
    Resume,
    /// Only valid while paused.
    Step,
    Reset,
    SetTraffic {
        enabled: bool,
    },
    /// Restarts the session, since it changes how the pool is managed.
    SetPrefixCaching {
        enabled: bool,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn ack_and_event_wire_format() {
        let ack = ServerMessage::Ack {
            id: 7,
            result: CommandResult::Ok { seq_id: Some(3) },
        };
        assert_eq!(
            serde_json::to_value(&ack).unwrap(),
            json!({"type": "ack", "id": 7, "result": {"status": "ok", "seq_id": 3}})
        );

        let event = Event {
            tick: 4,
            seq_id: 2,
            kind: EventKind::Cow {
                src_block: 1,
                dst_block: 9,
            },
        };
        assert_eq!(
            serde_json::to_value(&event).unwrap(),
            json!({"tick": 4, "seq_id": 2, "kind": {"type": "cow", "src_block": 1, "dst_block": 9}})
        );
    }

    #[test]
    fn client_commands_parse() {
        let msg: ClientMessage = serde_json::from_value(json!({
            "id": 1,
            "command": {"type": "add_request", "prompt_len": 12, "max_new_tokens": 8}
        }))
        .unwrap();
        assert_eq!(
            msg.command,
            Command::AddRequest {
                system_prompt: None,
                prompt_len: 12,
                max_new_tokens: 8
            },
            "system_prompt may be omitted"
        );

        let msg: ClientMessage =
            serde_json::from_value(json!({"id": 2, "command": {"type": "pause"}})).unwrap();
        assert_eq!(msg.command, Command::Pause);

        assert!(serde_json::from_value::<ClientMessage>(
            json!({"id": 3, "command": {"type": "format_disk"}})
        )
        .is_err());
    }
}
