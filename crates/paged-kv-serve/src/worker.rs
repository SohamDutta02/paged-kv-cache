use std::collections::HashMap;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::time::Instant;

use paged_kv_core::{KvBackend, SeqId};
use paged_kv_model::{Engine, EngineStats};
use tokio::sync::mpsc::UnboundedSender;

use crate::metrics::Metrics;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// The model produced a stop token.
    Stop,
    /// `max_tokens` ran out.
    Length,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
        }
    }
}

#[derive(Debug)]
pub enum Event {
    /// A generated token. Stop tokens are not sent.
    Token(u32),
    Done(FinishReason),
    Failed(String),
}

pub struct Submit {
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub events: UnboundedSender<Event>,
}

struct Live {
    events: UnboundedSender<Event>,
    submitted: Instant,
    last_token: Option<Instant>,
}

/// Starts the thread that owns the engine. It steps whenever any request is
/// live and sleeps otherwise; dropping every sender stops it.
pub fn spawn<B: KvBackend + Send + 'static>(
    engine: Engine<B>,
    stop_tokens: Vec<u32>,
    metrics: Arc<Metrics>,
) -> Sender<Submit> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("engine".into())
        .spawn(move || Worker::new(engine, stop_tokens, metrics).run(rx))
        .expect("failed to start the engine thread");
    tx
}

struct Worker<B: KvBackend> {
    engine: Engine<B>,
    stop_tokens: Vec<u32>,
    metrics: Arc<Metrics>,
    live: HashMap<SeqId, Live>,
    next_id: u64,
    seen: EngineStats,
}

impl<B: KvBackend> Worker<B> {
    fn new(engine: Engine<B>, stop_tokens: Vec<u32>, metrics: Arc<Metrics>) -> Self {
        Self {
            engine,
            stop_tokens,
            metrics,
            live: HashMap::new(),
            next_id: 0,
            seen: EngineStats::default(),
        }
    }

    fn run(mut self, rx: Receiver<Submit>) {
        loop {
            if self.live.is_empty() {
                match rx.recv() {
                    Ok(submit) => self.admit(submit),
                    Err(_) => return,
                }
            }
            // Requests that arrived during the last step join this one.
            while let Ok(submit) = rx.try_recv() {
                self.admit(submit);
            }
            self.drop_disconnected();
            if !self.live.is_empty() {
                self.step();
            }
            // Also after a cancellation that leaves nothing to step.
            self.metrics.engine(&self.engine, &mut self.seen);
        }
    }

    fn admit(&mut self, submit: Submit) {
        let id = SeqId(self.next_id);
        self.next_id += 1;
        match self
            .engine
            .add_request(id, submit.prompt, submit.max_tokens)
        {
            Ok(()) => {
                self.live.insert(
                    id,
                    Live {
                        events: submit.events,
                        submitted: Instant::now(),
                        last_token: None,
                    },
                );
            }
            Err(e) => {
                let _ = submit.events.send(Event::Failed(e.to_string()));
                self.metrics.request_done("error");
            }
        }
    }

    /// Frees the blocks of requests whose client went away, queued or not.
    fn drop_disconnected(&mut self) {
        let gone: Vec<SeqId> = self
            .live
            .iter()
            .filter(|(_, l)| l.events.is_closed())
            .map(|(id, _)| *id)
            .collect();
        for id in gone {
            self.live.remove(&id);
            self.engine.cancel(id);
            self.metrics.request_done("cancelled");
        }
    }

    fn step(&mut self) {
        let started = Instant::now();
        let out = match self.engine.step() {
            Ok(out) => out,
            Err(e) => {
                // The engine's state is unknown after a failed step.
                for (id, live) in self.live.drain() {
                    let _ = live.events.send(Event::Failed(e.to_string()));
                    self.engine.cancel(id);
                    self.metrics.request_done("error");
                }
                return;
            }
        };
        let now = Instant::now();
        self.metrics.step(now - started, out.sampled.len());

        for (id, token) in &out.sampled {
            let Some(live) = self.live.get_mut(id) else {
                continue;
            };
            let since = live.last_token.unwrap_or(live.submitted);
            if live.last_token.is_none() {
                self.metrics.ttft.observe((now - since).as_secs_f64());
            } else {
                self.metrics
                    .inter_token
                    .observe((now - since).as_secs_f64());
            }
            live.last_token = Some(now);
            if !self.stop_tokens.contains(token) {
                // A closed channel is noticed by drop_disconnected next loop.
                let _ = live.events.send(Event::Token(*token));
            }
        }
        for finished in out.finished {
            if let Some(live) = self.live.remove(&finished.id) {
                let stopped = finished
                    .tokens
                    .last()
                    .is_some_and(|t| self.stop_tokens.contains(t));
                let reason = if stopped {
                    FinishReason::Stop
                } else {
                    FinishReason::Length
                };
                let _ = live.events.send(Event::Done(reason));
                self.metrics
                    .request_duration
                    .observe((now - live.submitted).as_secs_f64());
                self.metrics.request_done("finished");
            }
        }
    }
}
