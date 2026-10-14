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
/// Share of traffic requests that start with a system prompt.
const TRAFFIC_SYSTEM_PROMPT_PERCENT: u32 = 75;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimConfig {
    pub num_blocks: usize,
    pub block_size: usize,
    pub max_batch_size: usize,
    pub max_waiting: usize,
    pub tick: Duration,
    pub prefix_caching: bool,
    /// Token lengths of the shared system prompts.
    pub system_prompts: Vec<usize>,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            num_blocks: 64,
            block_size: 8,
            max_batch_size: 6,
            max_waiting: 24,
            tick: Duration::from_millis(400),
            prefix_caching: true,
            system_prompts: vec![40, 56, 24],
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
    system_prompt: Option<u32>,
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
    /// Prompt tokens unique to each request are numbered from here.
    next_token: u32,
    system_prompts: Vec<Vec<u32>>,
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
        let scheduler = if config.prefix_caching {
            Scheduler::with_prefix_caching(&cache, config.max_batch_size)
        } else {
            Scheduler::new(&cache, config.max_batch_size)
        };
        // Tokens with the high byte set never collide with unique ones.
        let system_prompts = config
            .system_prompts
            .iter()
            .enumerate()
            .map(|(i, &len)| {
                (0..len as u32)
                    .map(|t| ((i as u32 + 1) << 24) | t)
                    .collect()
            })
            .collect();
        let mut sim = Self {
            scheduler,
            system_prompts,
            config,
            rng: StdRng::seed_from_u64(seed),
            tick: 0,
            paused: false,
            traffic: true,
            next_id: 1,
            next_token: 1,
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
                system_prompt,
                prompt_len,
                max_new_tokens,
            } => match self.submit(Origin::User, system_prompt, prompt_len, max_new_tokens) {
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
                self.restart();
                vec![ack(ok(None)), self.init_message()]
            }
            Command::SetTraffic { enabled } => {
                self.traffic = enabled;
                vec![ack(ok(None)), self.state_message(Vec::new())]
            }
            Command::SetPrefixCaching { enabled } => {
                self.config.prefix_caching = enabled;
                self.restart();
                vec![ack(ok(None)), self.init_message()]
            }
        }
    }

    /// Starts over with the same config, keeping the controls.
    fn restart(&mut self) {
        let (paused, traffic) = (self.paused, self.traffic);
        *self = Simulation::new(self.config.clone(), self.rng.gen());
        self.paused = paused;
        self.traffic = traffic;
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
            let with_system_prompt = !self.system_prompts.is_empty()
                && self.rng.gen_range(0..100) < TRAFFIC_SYSTEM_PROMPT_PERCENT;
            let system_prompt =
                with_system_prompt.then(|| self.rng.gen_range(0..self.system_prompts.len()) as u32);
            let prompt_len = self.rng.gen_range(4..24);
            let max_new_tokens = self.rng.gen_range(4..32);
            if let Ok(event) =
                self.submit(Origin::Traffic, system_prompt, prompt_len, max_new_tokens)
            {
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
        system_prompt: Option<u32>,
        prompt_len: u32,
        max_new_tokens: u32,
    ) -> Result<Event, String> {
        if prompt_len == 0 {
            return Err("The prompt needs at least 1 token.".into());
        }
        if max_new_tokens == 0 {
            return Err("Generate at least 1 new token.".into());
        }
        let prefix = match system_prompt {
            Some(i) => self
                .system_prompts
                .get(i as usize)
                .ok_or_else(|| format!("There is no system prompt {i}."))?
                .clone(),
            None => Vec::new(),
        };
        let needed = prefix.len() as u64 + u64::from(prompt_len) + u64::from(max_new_tokens);
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
        let mut prompt = prefix;
        let first = self.next_token;
        prompt.extend((0..prompt_len).map(|i| first.wrapping_add(i) & 0x00FF_FFFF));
        self.scheduler
            .add_prompt(SeqId(id.into()), prompt, max_new_tokens as usize)
            .map_err(|e| e.to_string())?;
        self.next_token = first.wrapping_add(prompt_len);
        self.next_id += 1;
        self.meta.insert(
            id,
            SeqMeta {
                parent: None,
                origin,
                system_prompt,
                preempted: false,
            },
        );
        self.totals.submitted += 1;
        Ok(Event {
            tick: self.tick,
            seq_id: id,
            kind: EventKind::Submitted {
                origin,
                system_prompt,
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
        let system_prompt = self.meta(parent).system_prompt;
        self.meta.insert(
            child,
            SeqMeta {
                parent: Some(parent),
                origin,
                system_prompt,
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

        for p in &outcome.prefills {
            events.push(event(
                p.id,
                EventKind::Admitted {
                    cached_tokens: count(p.cached_tokens),
                    computed_tokens: count(p.computed_tokens),
                },
            ));
            self.totals.prompt_tokens_cached += count(p.cached_tokens);
            self.totals.prompt_tokens_computed += count(p.computed_tokens);
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
        self.totals.cache_evictions =
            u32::try_from(self.scheduler.cache_stats().evictions).unwrap_or(u32::MAX);
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
            system_prompt: None,
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
            system_prompts: self.system_prompts.iter().map(|p| count(p.len())).collect(),
        }
    }

    fn metrics(&self) -> Metrics {
        let blocks = self.scheduler.block_snapshot();
        Metrics {
            tick: self.tick,
            used_blocks: count(blocks.iter().filter(|b| b.ref_count > 0).count()),
            shared_blocks: count(blocks.iter().filter(|b| b.ref_count > 1).count()),
            evictable_blocks: count(self.scheduler.cache_stats().evictable_blocks),
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
                    system_prompt: meta.system_prompt,
                    cached_tokens: count(r.cached_tokens),
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
                    system_prompt: meta.system_prompt,
                    context_len: count(w.context_len),
                    max_new_tokens: count(w.max_new_tokens),
                    preempted: meta.preempted,
                }
            })
            .collect();

        let blocks = self.scheduler.block_snapshot();
        Snapshot {
            tick: self.tick,
            paused: self.paused,
            traffic: self.traffic,
            prefix_caching: self.scheduler.prefix_caching(),
            ref_counts: blocks.iter().map(|b| b.ref_count).collect(),
            cached_blocks: blocks.iter().filter(|b| b.cached).map(|b| b.id.0).collect(),
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
                system_prompt: None,
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
                system_prompt: None,
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
                system_prompt: None,
                prompt_len: 0,
                max_new_tokens: 4,
            },
        );
        assert_eq!(rejection(&r), "The prompt needs at least 1 token.");

        let r = cmd(
            &mut sim,
            Command::AddRequest {
                system_prompt: None,
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
                system_prompt: None,
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
                system_prompt: None,
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

    fn add_with_system_prompt(sim: &mut Simulation, system_prompt: u32) -> u32 {
        let replies = cmd(
            sim,
            Command::AddRequest {
                system_prompt: Some(system_prompt),
                prompt_len: 4,
                max_new_tokens: 8,
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

    fn admitted(tick: &ServerMessage, seq_id: u32) -> (u32, u32) {
        let ServerMessage::Tick { events, .. } = tick else {
            panic!("expected a tick")
        };
        events
            .iter()
            .find_map(|e| match e.kind {
                EventKind::Admitted {
                    cached_tokens,
                    computed_tokens,
                } if e.seq_id == seq_id => Some((cached_tokens, computed_tokens)),
                _ => None,
            })
            .unwrap_or_else(|| panic!("#{seq_id} was not admitted"))
    }

    #[test]
    fn requests_with_the_same_system_prompt_reuse_its_blocks() {
        let mut sim = quiet();
        // System prompt 1 is 56 tokens: seven full blocks of 8.
        let first = add_with_system_prompt(&mut sim, 1);
        assert_eq!(admitted(&sim.advance().unwrap(), first), (0, 60));

        let second = add_with_system_prompt(&mut sim, 1);
        let tick = sim.advance().unwrap();
        assert_eq!(admitted(&tick, second), (56, 4));

        let snap = snapshot_of(&tick);
        let views: Vec<_> = snap.running.iter().collect();
        assert_eq!(views[0].blocks[..7], views[1].blocks[..7]);
        assert_eq!(views[1].cached_tokens, 56);
        assert_eq!(views[1].system_prompt, Some(1));
        assert_eq!(snap.metrics.shared_blocks, 7);
        assert!(views[0].blocks[..7]
            .iter()
            .all(|b| snap.cached_blocks.contains(b)));
        assert_eq!(snap.totals.prompt_tokens_cached, 56);
        assert_eq!(snap.totals.prompt_tokens_computed, 64);
    }

    #[test]
    fn prefix_caching_can_be_turned_off_which_restarts_the_session() {
        let mut sim = quiet();
        add_with_system_prompt(&mut sim, 0);
        sim.advance().unwrap();

        let replies = cmd(&mut sim, Command::SetPrefixCaching { enabled: false });
        let ServerMessage::Init {
            snapshot, config, ..
        } = &replies[1]
        else {
            panic!("expected init")
        };
        assert!(!snapshot.prefix_caching);
        assert_eq!(snapshot.tick, 0);
        assert_eq!(config.system_prompts, [40, 56, 24]);

        add_with_system_prompt(&mut sim, 0);
        sim.advance().unwrap();
        let second = add_with_system_prompt(&mut sim, 0);
        let tick = sim.advance().unwrap();
        assert_eq!(admitted(&tick, second), (0, 44));
        assert!(snapshot_of(&tick).cached_blocks.is_empty());
    }

    #[test]
    fn system_prompts_are_validated_and_count_toward_the_pool() {
        let mut sim = quiet();
        let r = cmd(
            &mut sim,
            Command::AddRequest {
                system_prompt: Some(9),
                prompt_len: 4,
                max_new_tokens: 4,
            },
        );
        assert_eq!(rejection(&r), "There is no system prompt 9.");

        let r = cmd(
            &mut sim,
            Command::AddRequest {
                system_prompt: Some(1),
                prompt_len: 400,
                max_new_tokens: 100,
            },
        );
        assert_eq!(
            rejection(&r),
            "That request needs 556 tokens, but the whole pool holds 512."
        );
    }

    #[test]
    fn traffic_reuses_system_prompts() {
        let mut sim = Simulation::new(SimConfig::default(), 3);
        for _ in 0..300 {
            sim.advance().unwrap();
        }
        assert!(sim.totals.prompt_tokens_cached > 0);
        assert!(sim.totals.prompt_tokens_computed > 0);
    }
}
