use std::sync::atomic::AtomicU64;
use std::time::Duration;

use paged_kv_core::KvBackend;
use paged_kv_model::{Engine, EngineStats};
use prometheus_client::encoding::{text::encode, EncodeLabelSet};
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::{exponential_buckets, linear_buckets, Histogram};
use prometheus_client::registry::Registry;

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct Outcome {
    status: &'static str,
}

/// Everything `/metrics` reports, in the Prometheus/OpenMetrics text format.
pub struct Metrics {
    registry: Registry,
    requests: Family<Outcome, Counter>,
    pub(crate) ttft: Histogram,
    pub(crate) inter_token: Histogram,
    pub(crate) request_duration: Histogram,
    step_duration: Histogram,
    batch_size: Histogram,
    prefill_computed: Counter,
    prefill_cached: Counter,
    generated: Counter,
    preemptions: Counter,
    running: Gauge,
    waiting: Gauge,
    kv_utilization: Gauge<f64, AtomicU64>,
    cached_blocks: Gauge,
}

impl Metrics {
    pub fn new(max_batch_size: usize) -> Self {
        let mut registry = Registry::with_prefix("paged_kv");
        let latency = || Histogram::new(exponential_buckets(0.001, 2.0, 16));

        let requests = Family::<Outcome, Counter>::default();
        registry.register(
            "requests",
            "Requests by how they ended: finished, cancelled, rejected or error",
            requests.clone(),
        );
        let ttft = latency();
        registry.register(
            "time_to_first_token_seconds",
            "From arrival to the first generated token",
            ttft.clone(),
        );
        let inter_token = latency();
        registry.register(
            "inter_token_latency_seconds",
            "Between consecutive tokens of one request",
            inter_token.clone(),
        );
        let request_duration = latency();
        registry.register(
            "request_duration_seconds",
            "From arrival to the last token",
            request_duration.clone(),
        );
        let step_duration = latency();
        registry.register(
            "step_duration_seconds",
            "One engine step: scheduling plus a forward pass",
            step_duration.clone(),
        );
        let batch_size = Histogram::new(linear_buckets(1.0, 1.0, max_batch_size.max(1) as u16));
        registry.register(
            "batch_size",
            "Sequences in each step's batch",
            batch_size.clone(),
        );
        let prefill_computed = Counter::default();
        registry.register(
            "prefill_tokens_computed",
            "Prompt tokens run through the model, including recomputation",
            prefill_computed.clone(),
        );
        let prefill_cached = Counter::default();
        registry.register(
            "prefill_tokens_cached",
            "Prompt tokens served from the prefix cache",
            prefill_cached.clone(),
        );
        let generated = Counter::default();
        registry.register("generated_tokens", "Tokens sampled", generated.clone());
        let preemptions = Counter::default();
        registry.register(
            "preemptions",
            "Sequences evicted to free blocks",
            preemptions.clone(),
        );
        let running = Gauge::default();
        registry.register(
            "running_requests",
            "Sequences in the batch",
            running.clone(),
        );
        let waiting = Gauge::default();
        registry.register(
            "waiting_requests",
            "Requests queued for admission",
            waiting.clone(),
        );
        let kv_utilization = Gauge::<f64, AtomicU64>::default();
        registry.register(
            "kv_cache_utilization",
            "Fraction of KV blocks in use",
            kv_utilization.clone(),
        );
        let cached_blocks = Gauge::default();
        registry.register(
            "kv_cache_cached_blocks",
            "Blocks registered in the prefix cache",
            cached_blocks.clone(),
        );

        Self {
            registry,
            requests,
            ttft,
            inter_token,
            request_duration,
            step_duration,
            batch_size,
            prefill_computed,
            prefill_cached,
            generated,
            preemptions,
            running,
            waiting,
            kv_utilization,
            cached_blocks,
        }
    }

    pub fn encode(&self) -> String {
        let mut out = String::new();
        encode(&mut out, &self.registry).expect("writing to a String cannot fail");
        out
    }

    pub(crate) fn request_done(&self, status: &'static str) {
        self.requests.get_or_create(&Outcome { status }).inc();
    }

    pub(crate) fn step(&self, duration: Duration, batch_size: usize) {
        self.step_duration.observe(duration.as_secs_f64());
        self.batch_size.observe(batch_size as f64);
    }

    pub(crate) fn engine<B: KvBackend>(&self, engine: &Engine<B>, seen: &mut EngineStats) {
        let now = engine.stats();
        let delta = |new: usize, old: usize| (new - old) as u64;
        self.prefill_computed.inc_by(delta(
            now.prefill_tokens_computed,
            seen.prefill_tokens_computed,
        ));
        self.prefill_cached
            .inc_by(delta(now.prefill_tokens_cached, seen.prefill_tokens_cached));
        self.generated
            .inc_by(delta(now.tokens_generated, seen.tokens_generated));
        self.preemptions
            .inc_by(delta(now.preemptions, seen.preemptions));
        *seen = now;

        let scheduler = engine.scheduler();
        self.running.set(scheduler.num_running() as i64);
        self.waiting.set(scheduler.num_waiting() as i64);
        self.kv_utilization.set(scheduler.pool_utilization());
        self.cached_blocks
            .set(scheduler.cache_stats().cached_blocks as i64);
    }
}
