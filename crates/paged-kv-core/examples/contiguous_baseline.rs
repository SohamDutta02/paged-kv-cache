//! Runs one request stream through contiguous per-sequence KV buffers and
//! through the paged scheduler, and prints how much of the pool each one
//! turns into running sequences.
//!
//!     cargo run --release -p paged-kv-core --example contiguous_baseline

use std::collections::VecDeque;

use paged_kv_core::{CacheConfig, Scheduler, SeqId};

const STEPS: usize = 5_000;
const NUM_BLOCKS: usize = 512;
const BLOCK_SIZE: usize = 16;
const POOL_TOKENS: usize = NUM_BLOCKS * BLOCK_SIZE;
const MAX_BATCH: usize = 64;
/// The queue is topped up to this depth before every step.
const QUEUE_DEPTH: usize = 8;
const MAX_MODEL_LEN: usize = 2_048;
const PROMPT_TOKENS: (usize, usize) = (32, 512);
/// What every request asks for; the actual output stops earlier.
const MAX_NEW_TOKENS: usize = 512;

#[derive(Clone, Copy)]
struct Request {
    prompt: usize,
    output: usize,
}

struct Requests(u64);

impl Requests {
    fn draw(&mut self) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as usize
    }

    /// Output lengths are skewed short, like real stop-at-EOS traffic.
    fn next(&mut self) -> Request {
        let prompt = PROMPT_TOKENS.0 + self.draw() % (PROMPT_TOKENS.1 - PROMPT_TOKENS.0 + 1);
        let u = self.draw() % 1_001;
        let output = 8 + u * u * (MAX_NEW_TOKENS - 8) / 1_000_000;
        Request { prompt, output }
    }
}

#[derive(Clone, Copy)]
enum Policy {
    MaxModelLen,
    RequestMax,
    Oracle,
}

impl Policy {
    fn label(self) -> &'static str {
        match self {
            Policy::MaxModelLen => "Contiguous, max model length",
            Policy::RequestMax => "Contiguous, prompt + max new",
            Policy::Oracle => "Contiguous, exact (oracle)",
        }
    }

    fn reserve(self, r: Request) -> usize {
        match self {
            Policy::MaxModelLen => MAX_MODEL_LEN,
            Policy::RequestMax => r.prompt + MAX_NEW_TOKENS,
            Policy::Oracle => r.prompt + r.output,
        }
    }
}

/// First-fit over one token range: what a slab of per-sequence buffers
/// looks like, external fragmentation included.
struct Contiguous {
    /// Free ranges as (start, len), sorted by start and never adjacent.
    free: Vec<(usize, usize)>,
}

impl Contiguous {
    fn new(len: usize) -> Self {
        Self {
            free: vec![(0, len)],
        }
    }

    fn alloc(&mut self, n: usize) -> Option<usize> {
        let i = self.free.iter().position(|&(_, len)| len >= n)?;
        let (start, len) = self.free[i];
        if len == n {
            self.free.remove(i);
        } else {
            self.free[i] = (start + n, len - n);
        }
        Some(start)
    }

    fn release(&mut self, start: usize, n: usize) {
        let i = self.free.partition_point(|&(s, _)| s < start);
        self.free.insert(i, (start, n));
        if i + 1 < self.free.len() && start + n == self.free[i + 1].0 {
            let next = self.free.remove(i + 1).1;
            self.free[i].1 += next;
        }
        if i > 0 && self.free[i - 1].0 + self.free[i - 1].1 == start {
            let this = self.free.remove(i).1;
            self.free[i - 1].1 += this;
        }
    }

    fn free_tokens(&self) -> usize {
        self.free.iter().map(|&(_, len)| len).sum()
    }
}

#[derive(Default)]
struct Report {
    finished: usize,
    /// Includes prompts recomputed after preemption.
    prefilled: usize,
    preempted: usize,
    batch_total: usize,
    reserved_total: usize,
    live_total: usize,
}

impl Report {
    fn record(&mut self, batch: usize, reserved: usize, live: usize) {
        self.batch_total += batch;
        self.reserved_total += reserved;
        self.live_total += live;
    }
}

struct Running {
    start: usize,
    reserved: usize,
    tokens: usize,
    remaining: usize,
}

fn run_contiguous(policy: Policy) -> Report {
    let mut pool = Contiguous::new(POOL_TOKENS);
    let mut requests = Requests(42);
    let mut waiting: VecDeque<Request> = VecDeque::new();
    let mut running: Vec<Running> = Vec::new();
    let mut report = Report::default();

    for _ in 0..STEPS {
        while waiting.len() < QUEUE_DEPTH {
            waiting.push_back(requests.next());
        }
        // Strict FIFO, like the paged scheduler.
        while running.len() < MAX_BATCH {
            let Some(&r) = waiting.front() else { break };
            let reserved = policy.reserve(r);
            let Some(start) = pool.alloc(reserved) else {
                break;
            };
            waiting.pop_front();
            report.prefilled += r.prompt;
            running.push(Running {
                start,
                reserved,
                tokens: r.prompt,
                remaining: r.output,
            });
        }
        for s in &mut running {
            s.tokens += 1;
            s.remaining -= 1;
        }
        running.retain(|s| {
            if s.remaining == 0 {
                pool.release(s.start, s.reserved);
                report.finished += 1;
            }
            s.remaining > 0
        });
        report.record(
            running.len(),
            running.iter().map(|s| s.reserved).sum(),
            running.iter().map(|s| s.tokens).sum(),
        );
    }
    let reserved: usize = running.iter().map(|s| s.reserved).sum();
    assert_eq!(pool.free_tokens() + reserved, POOL_TOKENS, "lost a range");
    report
}

fn run_paged() -> Report {
    let config = CacheConfig {
        num_blocks: NUM_BLOCKS,
        block_size: BLOCK_SIZE,
        num_kv_heads: 1,
        head_dim: 1,
        num_layers: 1,
        dtype_bytes: 2,
    };
    let mut scheduler = Scheduler::new(&config, MAX_BATCH);
    let mut requests = Requests(42);
    let mut next_id = 0u64;
    let mut report = Report::default();

    for _ in 0..STEPS {
        while scheduler.num_waiting() < QUEUE_DEPTH {
            let r = requests.next();
            // Blocks are taken as tokens arrive, so passing the actual output
            // length is the same as stopping at EOS.
            scheduler
                .add_request(SeqId(next_id), r.prompt, r.output)
                .expect("requests fit the pool");
            next_id += 1;
        }
        let outcome = scheduler.step().expect("scheduler step");
        report.finished += outcome.finished.len();
        report.preempted += outcome.preempted.len();
        report.prefilled += outcome
            .prefills
            .iter()
            .map(|p| p.computed_tokens)
            .sum::<usize>();
        let reserved = (scheduler.pool_utilization() * POOL_TOKENS as f64).round() as usize;
        let live = scheduler.running_snapshot().iter().map(|s| s.tokens).sum();
        report.record(scheduler.num_running(), reserved, live);
    }
    report
}

fn main() {
    println!(
        "{POOL_TOKENS}-token pool, batch of up to {MAX_BATCH}, {STEPS} steps; \
         prompts of {}-{} tokens, up to {MAX_NEW_TOKENS} new tokens each\n",
        PROMPT_TOKENS.0, PROMPT_TOKENS.1
    );
    println!(
        "{:<30} {:>9} {:>11} {:>14} {:>16} {:>10} {:>18}",
        "allocator",
        "finished",
        "mean batch",
        "pool reserved",
        "reserved in use",
        "preempted",
        "prefill / finished"
    );
    let runs = [Policy::MaxModelLen, Policy::RequestMax, Policy::Oracle]
        .map(|p| (p.label(), run_contiguous(p)));
    let paged = ("Paged, 16-token blocks", run_paged());
    for (label, r) in runs.iter().chain(std::iter::once(&paged)) {
        println!(
            "{:<30} {:>9} {:>11.1} {:>13.1}% {:>15.1}% {:>10} {:>18.0}",
            label,
            r.finished,
            r.batch_total as f64 / STEPS as f64,
            100.0 * r.reserved_total as f64 / (STEPS * POOL_TOKENS) as f64,
            100.0 * r.live_total as f64 / r.reserved_total.max(1) as f64,
            r.preempted,
            r.prefilled as f64 / r.finished.max(1) as f64,
        );
    }
}
