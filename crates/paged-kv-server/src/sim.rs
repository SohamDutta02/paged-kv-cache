//! One dashboard session: a scheduler driven by synthetic traffic and client
//! commands. Synchronous and I/O-free; `ws.rs` moves messages in and out.

use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use paged_kv_core::{CacheConfig, CacheError, Scheduler, SeqId, StepOutcome};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::protocol::{
    ClientMessage, Command, CommandResult, Event, EventKind, Metrics, Origin, PoolConfig,
    SequenceView, ServerMessage, Snapshot, Totals, WaitingView,
};

pub const HISTORY_LEN: usize = 150;
pub const EVENT_LOG_LEN: usize = 200;
/// Traffic pauses at this queue depth to leave room for the user's requests.
const TRAFFIC_QUEUE_LIMIT: usize = 6;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimConfig {
    pub num_blocks: usize,
    pub block_size: usize,
    pub max_batch_size: usize,
    pub max_waiting: usize,
    pub tick: Duration,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            num_blocks: 64,
            block_size: 8,
            max_batch_size: 6,
            max_waiting: 24,
            tick: Duration::from_millis(400),
        }
    }
}

impl SimConfig {
    fn capacity_tokens(&self) -> u64 {
        (self.num_blocks * self.block_size) as u64
    }
}

#[derive(Debug, Clone, Copy)]
struct SeqMeta {
    parent: Option<u32>,
    origin: Origin,
    preempted: bool,
}

pub struct Simulation {
    config: SimConfig,
    scheduler: Scheduler,
    rng: StdRng,
    tick: u32,
    paused: bool,
    traffic: bool,
    next_id: u32,
    meta: HashMap<u32, SeqMeta>,
    totals: Totals,
    history: VecDeque<Metrics>,
    events: VecDeque<Event>,
}

impl Simulation {
    pub fn new(config: SimConfig, seed: u64) -> Self {
        let cache = CacheConfig {
            num_blocks: config.num_blocks,
            block_size: config.block_size,
            // Only block bookkeeping is simulated, so the tensor shape is unused.
            num_kv_heads: 1,
            head_dim: 1,
            num_layers: 1,
            dtype_bytes: 4,
        };
        let mut sim = Self {
            scheduler: Scheduler::new(&cache, config.max_batch_size),
            config,
            rng: StdRng::seed_from_u64(seed),
            tick: 0,
            paused: false,
            traffic: true,
            next_id: 1,
            meta: HashMap::new(),
            totals: Totals::default(),
            history: VecDeque::with_capacity(HISTORY_LEN),
            events: VecDeque::with_capacity(EVENT_LOG_LEN),
        };
        let first = sim.metrics();
        sim.history.push_back(first);
        sim
    }

    pub fn tick_interval(&self) -> Duration {
        self.config.tick
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }

    pub fn init_message(&self) -> ServerMessage {
        ServerMessage::Init {
            config: self.pool_config(),
            snapshot: self.snapshot(),
            history: self.history.iter().cloned().collect(),
            events: self.events.iter().cloned().collect(),
        }
    }

    /// Errors only on an internal scheduler inconsistency; end the session then.
    pub fn advance(&mut self) -> Result<ServerMessage, CacheError> {
        self.tick = self.tick.saturating_add(1);
        let mut events = Vec::new();
        if self.traffic {
            self.generate_traffic(&mut events);
        }
        let outcome = self.scheduler.step()?;
        self.record_step(&outcome, &mut events);

        let metrics = self.metrics();
        if self.history.len() == HISTORY_LEN {
            self.history.pop_front();
        }
        self.history.push_back(metrics);
        self.log(&events);

        Ok(ServerMessage::Tick {
            snapshot: self.snapshot(),
            events,
        })
    }

    /// The `Ack` always comes first.
    pub fn handle(&mut self, msg: ClientMessage) -> Vec<ServerMessage> {
        let id = msg.id;
        let ack = |result| ServerMessage::Ack { id, result };
        let ok = |seq_id| CommandResult::Ok { seq_id };
        let rejected = |message: String| ack(CommandResult::Rejected { message });

        match msg.command {
            Command::AddRequest {
                prompt_len,
                max_new_tokens,
            } => match self.submit(Origin::User, prompt_len, max_new_tokens) {
                Ok(event) => {
                    let seq_id = event.seq_id;
                    vec![ack(ok(Some(seq_id))), self.state_message(vec![event])]
                }
                Err(message) => vec![rejected(message)],
            },
            Command::Fork { seq_id } => match self.fork(seq_id, Origin::User) {
                Ok(event) => {
                    let child = event.seq_id;
                    vec![ack(ok(Some(child))), self.state_message(vec![event])]
                }
                Err(message) => vec![rejected(message)],
            },
            Command::Pause => {
                self.paused = true;
                vec![ack(ok(None)), self.state_message(Vec::new())]
            }
            Command::Resume => {
                self.paused = false;
                vec![ack(ok(None)), self.state_message(Vec::new())]
            }
            Command::Step => {
                if !self.paused {
                    return vec![rejected(
                        "Pause the simulation before stepping one tick at a time.".into(),
                    )];
                }
                match self.advance() {
                    Ok(tick) => vec![ack(ok(None)), tick],
                    Err(e) => vec![rejected(format!("The scheduler failed: {e}"))],
                }
            }
            Command::Reset => {
                let (paused, traffic) = (self.paused, self.traffic);
                *self = Simulation::new(self.config.clone(), self.rng.gen());
                self.paused = paused;
                self.traffic = traffic;
                vec![ack(ok(None)), self.init_message()]
            }
            Command::SetTraffic { enabled } => {
                self.traffic = enabled;
                vec![ack(ok(None)), self.state_message(Vec::new())]
            }
        }
    }

    fn state_message(&mut self, events: Vec<Event>) -> ServerMessage {
        self.log(&events);
        ServerMessage::State {
            snapshot: self.snapshot(),
            events,
        }
    }

    fn generate_traffic(&mut self, events: &mut Vec<Event>) {
        if self.tick % 3 == 0 && self.scheduler.num_waiting() < TRAFFIC_QUEUE_LIMIT {
            let prompt_len = self.rng.gen_range(4..48);
            let max_new_tokens = self.rng.gen_range(4..32);
            if let Ok(event) = self.submit(Origin::Traffic, prompt_len, max_new_tokens) {
                events.push(event);
            }
        }
        if self.tick % 5 == 0 {
            let running = self.scheduler.running_snapshot();
            if !running.is_empty() {
                let parent = &running[self.rng.gen_range(0..running.len())];
                if let Ok(event) = self.fork(seq_u32(parent.id), Origin::Traffic) {
                    events.push(event);
                }
            }
        }
    }

    fn submit(
        &mut self,
        origin: Origin,
        prompt_len: u32,
        max_new_tokens: u32,
    ) -> Result<Event, String> {
        if prompt_len == 0 {
            return Err("The prompt needs at least 1 token.".into());
        }
        if max_new_tokens == 0 {
            return Err("Generate at least 1 new token.".into());
        }
        let needed = u64::from(prompt_len) + u64::from(max_new_tokens);
        let capacity = self.config.capacity_tokens();
        if needed > capacity {
            return Err(format!(
                "That request needs {needed} tokens, but the whole pool holds {capacity}."
            ));
        }
        if self.scheduler.num_waiting() >= self.config.max_waiting {
            return Err(format!(
                "The queue is full ({} requests). Try again once some are admitted.",
                self.config.max_waiting
            ));
        }

        let id = self.next_id;
        self.scheduler
            .add_request(
                SeqId(id.into()),
                prompt_len as usize,
                max_new_tokens as usize,
            )
            .map_err(|e| e.to_string())?;
        self.next_id += 1;
        self.meta.insert(
            id,
            SeqMeta {
                parent: None,
                origin,
                preempted: false,
            },
        );
        self.totals.submitted += 1;
        Ok(Event {
            tick: self.tick,
            seq_id: id,
            kind: EventKind::Submitted {
                origin,
                prompt_len,
                max_new_tokens,
            },
        })
    }

    fn fork(&mut self, parent: u32, origin: Origin) -> Result<Event, String> {
        let running = self.scheduler.running_snapshot();
        let Some(info) = running.iter().find(|r| r.id == SeqId(parent.into())) else {
            let waiting = self
                .scheduler
                .waiting_snapshot()
                .iter()
                .any(|w| w.id == SeqId(parent.into()));
            return Err(if waiting {
                format!("#{parent} is waiting. Only running sequences can fork.")
            } else {
                format!("#{parent} isn't running.")
            });
        };
        let remaining = info.max_new_tokens.saturating_sub(info.generated);
        if remaining == 0 {
            return Err(format!("#{parent} has no tokens left to generate."));
        }

        let child = self.next_id;
        match self
            .scheduler
            .fork_sequence(SeqId(parent.into()), SeqId(child.into()), remaining)
        {
            Ok(()) => {}
            Err(CacheError::BatchFull { max_batch_size }) => {
                return Err(format!(
                    "The batch is full ({max_batch_size} running). Fork again once one finishes."
                ))
            }
            Err(e) => return Err(e.to_string()),
        }
        self.next_id += 1;
        self.meta.insert(
            child,
            SeqMeta {
                parent: Some(parent),
                origin,
                preempted: false,
            },
        );
        self.totals.forks += 1;
        Ok(Event {
            tick: self.tick,
            seq_id: child,
            kind: EventKind::Forked { parent },
        })
    }

    fn record_step(&mut self, outcome: &StepOutcome, events: &mut Vec<Event>) {
        let tick = self.tick;
        let event = |id: SeqId, kind| Event {
            tick,
            seq_id: seq_u32(id),
            kind,
        };

        for &id in &outcome.admitted {
            events.push(event(id, EventKind::Admitted));
        }
        for d in &outcome.decoded {
            if let Some(cow) = d.cow {
                events.push(event(
                    d.id,
                    EventKind::Cow {
                        src_block: cow.src.0,
                        dst_block: cow.dst.0,
                    },
                ));
                self.totals.cow_copies += 1;
            }
        }
        for &id in &outcome.preempted {
            if let Some(meta) = self.meta.get_mut(&seq_u32(id)) {
                meta.preempted = true;
            }
            events.push(event(id, EventKind::Preempted));
        }
        for &id in &outcome.finished {
            self.meta.remove(&seq_u32(id));
            events.push(event(id, EventKind::Finished));
        }

        self.totals.admitted += count(outcome.admitted.len());
        self.totals.preempted += count(outcome.preempted.len());
        self.totals.finished += count(outcome.finished.len());
        self.totals.tokens_generated += count(outcome.decoded.len());
    }

    fn log(&mut self, events: &[Event]) {
        for e in events {
            if self.events.len() == EVENT_LOG_LEN {
                self.events.pop_front();
            }
            self.events.push_back(e.clone());
        }
    }

    fn meta(&self, id: u32) -> SeqMeta {
        self.meta.get(&id).copied().unwrap_or(SeqMeta {
            parent: None,
            origin: Origin::Traffic,
            preempted: false,
        })
    }

    fn pool_config(&self) -> PoolConfig {
        PoolConfig {
            num_blocks: count(self.config.num_blocks),
            block_size: count(self.config.block_size),
            max_batch_size: count(self.config.max_batch_size),
            max_waiting: count(self.config.max_waiting),
            tick_ms: u32::try_from(self.config.tick.as_millis()).unwrap_or(u32::MAX),
        }
    }

    fn ref_counts(&self) -> Vec<u32> {
        self.scheduler
            .block_snapshot()
            .iter()
            .map(|b| b.ref_count)
            .collect()
    }

    fn metrics(&self) -> Metrics {
        let ref_counts = self.ref_counts();
        Metrics {
            tick: self.tick,
            used_blocks: count(ref_counts.iter().filter(|&&rc| rc > 0).count()),
            shared_blocks: count(ref_counts.iter().filter(|&&rc| rc > 1).count()),
            running: count(self.scheduler.num_running()),
            waiting: count(self.scheduler.num_waiting()),
        }
    }

    fn snapshot(&self) -> Snapshot {
        let running = self
            .scheduler
            .running_snapshot()
            .into_iter()
            .map(|r| {
                let id = seq_u32(r.id);
                let meta = self.meta(id);
                SequenceView {
                    id,
                    parent: meta.parent,
                    origin: meta.origin,
                    tokens: count(r.tokens),
                    generated: count(r.generated),
                    max_new_tokens: count(r.max_new_tokens),
                    blocks: r.blocks.iter().map(|b| b.0).collect(),
                }
            })
            .collect();
        let waiting = self
            .scheduler
            .waiting_snapshot()
            .into_iter()
            .map(|w| {
                let id = seq_u32(w.id);
                let meta = self.meta(id);
                WaitingView {
                    id,
                    parent: meta.parent,
                    origin: meta.origin,
                    context_len: count(w.context_len),
                    max_new_tokens: count(w.max_new_tokens),
                    preempted: meta.preempted,
                }
            })
            .collect();

        Snapshot {
            tick: self.tick,
            paused: self.paused,
            traffic: self.traffic,
            ref_counts: self.ref_counts(),
            running,
            waiting,
            metrics: self.metrics(),
            totals: self.totals.clone(),
        }
    }
}

fn seq_u32(id: SeqId) -> u32 {
    u32::try_from(id.0).expect("sequence ids are issued from a u32 counter")
}

fn count(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quiet() -> Simulation {
        let mut sim = Simulation::new(SimConfig::default(), 1);
        sim.traffic = false;
        sim
    }

    fn cmd(sim: &mut Simulation, command: Command) -> Vec<ServerMessage> {
        sim.handle(ClientMessage { id: 1, command })
    }

    fn snapshot_of(msg: &ServerMessage) -> &Snapshot {
        match msg {
            ServerMessage::Init { snapshot, .. }
            | ServerMessage::Tick { snapshot, .. }
            | ServerMessage::State { snapshot, .. } => snapshot,
            other => panic!("expected a message with a snapshot, got {other:?}"),
        }
    }

    fn add(sim: &mut Simulation, prompt_len: u32, max_new_tokens: u32) -> u32 {
        let replies = cmd(
            sim,
            Command::AddRequest {
                prompt_len,
                max_new_tokens,
            },
        );
        match &replies[0] {
            ServerMessage::Ack {
                result: CommandResult::Ok { seq_id: Some(id) },
                ..
            } => *id,
            other => panic!("request was not accepted: {other:?}"),
        }
    }

    fn rejection(replies: &[ServerMessage]) -> &str {
        match &replies[0] {
            ServerMessage::Ack {
                result: CommandResult::Rejected { message },
                ..
            } => message,
            other => panic!("expected a rejection, got {other:?}"),
        }
    }

    #[test]
    fn a_user_request_is_queued_then_admitted() {
        let mut sim = quiet();
        let replies = cmd(
            &mut sim,
            Command::AddRequest {
                prompt_len: 10,
                max_new_tokens: 4,
            },
        );
        assert_eq!(replies.len(), 2, "ack then state");
        let queued = snapshot_of(&replies[1]);
        assert_eq!(queued.waiting.len(), 1);
        assert_eq!(queued.waiting[0].origin, Origin::User);
        assert_eq!(queued.waiting[0].context_len, 10);

        let tick = sim.advance().unwrap();
        let snap = snapshot_of(&tick);
        assert!(snap.waiting.is_empty());
        assert_eq!(snap.running.len(), 1);
        assert_eq!(snap.running[0].tokens, 11);
        assert_eq!(snap.running[0].blocks.len(), 2);
        assert_eq!(snap.metrics.used_blocks, 2);
        assert_eq!(snap.totals.admitted, 1);
    }

    #[test]
    fn invalid_requests_are_rejected_with_a_reason() {
        let mut sim = quiet();
        let r = cmd(
            &mut sim,
            Command::AddRequest {
                prompt_len: 0,
                max_new_tokens: 4,
            },
        );
        assert_eq!(rejection(&r), "The prompt needs at least 1 token.");

        let r = cmd(
            &mut sim,
            Command::AddRequest {
                prompt_len: 500,
                max_new_tokens: 100,
            },
        );
        assert_eq!(
            rejection(&r),
            "That request needs 600 tokens, but the whole pool holds 512."
        );

        let r = cmd(
            &mut sim,
            Command::AddRequest {
                prompt_len: u32::MAX,
                max_new_tokens: u32::MAX,
            },
        );
        assert!(
            rejection(&r).contains("whole pool holds 512"),
            "no overflow"
        );
    }

    #[test]
    fn the_queue_has_a_limit() {
        let mut sim = quiet();
        for _ in 0..SimConfig::default().max_waiting {
            add(&mut sim, 4, 4);
        }
        let r = cmd(
            &mut sim,
            Command::AddRequest {
                prompt_len: 4,
                max_new_tokens: 4,
            },
        );
        assert!(rejection(&r).starts_with("The queue is full (24 requests)"));
    }

    #[test]
    fn forking_shares_blocks_and_records_lineage() {
        let mut sim = quiet();
        let parent = add(&mut sim, 12, 10);
        sim.advance().unwrap();

        let replies = cmd(&mut sim, Command::Fork { seq_id: parent });
        let snap = snapshot_of(&replies[1]);
        let child = snap.running.iter().find(|s| s.id != parent).unwrap();
        assert_eq!(child.parent, Some(parent));
        let parent_view = snap.running.iter().find(|s| s.id == parent).unwrap();
        assert_eq!(child.blocks, parent_view.blocks, "no copy at fork time");
        assert_eq!(snap.metrics.shared_blocks, 2);
        assert_eq!(snap.totals.forks, 1);

        // The shared tail has free slots, so the next decode copies it.
        let tick = sim.advance().unwrap();
        let ServerMessage::Tick { events, .. } = &tick else {
            panic!("expected a tick")
        };
        assert!(events
            .iter()
            .any(|e| matches!(e.kind, EventKind::Cow { .. })));
    }

    #[test]
    fn forking_explains_why_it_was_refused() {
        let mut sim = quiet();
        assert_eq!(
            rejection(&cmd(&mut sim, Command::Fork { seq_id: 99 })),
            "#99 isn't running."
        );

        let queued = add(&mut sim, 4, 4);
        assert_eq!(
            rejection(&cmd(&mut sim, Command::Fork { seq_id: queued })),
            format!("#{queued} is waiting. Only running sequences can fork.")
        );

        sim.advance().unwrap();
        for _ in 0..5 {
            cmd(&mut sim, Command::Fork { seq_id: queued });
        }
        assert_eq!(
            rejection(&cmd(&mut sim, Command::Fork { seq_id: queued })),
            "The batch is full (6 running). Fork again once one finishes."
        );
    }

    #[test]
    fn stepping_requires_pause_and_advances_one_tick() {
        let mut sim = quiet();
        assert!(rejection(&cmd(&mut sim, Command::Step)).starts_with("Pause the simulation"));

        cmd(&mut sim, Command::Pause);
        assert!(sim.is_paused());
        let replies = cmd(&mut sim, Command::Step);
        assert!(matches!(
            replies[0],
            ServerMessage::Ack {
                result: CommandResult::Ok { .. },
                ..
            }
        ));
        assert_eq!(snapshot_of(&replies[1]).tick, 1);
    }

    #[test]
    fn reset_starts_over_but_keeps_the_controls() {
        let mut sim = quiet();
        add(&mut sim, 4, 4);
        sim.advance().unwrap();
        cmd(&mut sim, Command::Pause);

        let replies = cmd(&mut sim, Command::Reset);
        let ServerMessage::Init {
            snapshot, history, ..
        } = &replies[1]
        else {
            panic!("reset should send init")
        };
        assert_eq!(snapshot.tick, 0);
        assert!(snapshot.running.is_empty() && snapshot.waiting.is_empty());
        assert_eq!(history.len(), 1);
        assert!(snapshot.paused, "pause survives reset");
        assert!(!snapshot.traffic, "traffic setting survives reset");
    }

    #[test]
    fn history_and_event_log_are_bounded() {
        let mut sim = Simulation::new(SimConfig::default(), 7);
        for _ in 0..(HISTORY_LEN * 3) {
            sim.advance().unwrap();
        }
        assert_eq!(sim.history.len(), HISTORY_LEN);
        assert!(sim.events.len() <= EVENT_LOG_LEN);
        let ServerMessage::Init { history, .. } = sim.init_message() else {
            unreachable!()
        };
        assert_eq!(history.last().unwrap().tick, sim.tick);
    }

    #[test]
    fn long_run_with_traffic_keeps_scheduler_invariants() {
        let mut sim = Simulation::new(SimConfig::default(), 42);
        let mut forks_seen = false;
        for _ in 0..2_000 {
            let tick = sim.advance().unwrap();
            sim.scheduler
                .check_invariants()
                .unwrap_or_else(|e| panic!("tick {}: {e}", sim.tick));
            let snap = snapshot_of(&tick);
            let used = snap.ref_counts.iter().filter(|&&rc| rc > 0).count();
            assert_eq!(used as u32, snap.metrics.used_blocks);
            forks_seen |= snap.running.iter().any(|s| s.parent.is_some());
            for s in &snap.running {
                assert!(sim.meta.contains_key(&s.id), "metadata for #{}", s.id);
            }
        }
        assert!(forks_seen, "traffic should fork sometimes");
        assert!(sim.totals.finished > 0);
        assert!(
            sim.meta.len() <= sim.config.max_batch_size + sim.config.max_waiting,
            "finished sequences' metadata is released"
        );
    }
}
