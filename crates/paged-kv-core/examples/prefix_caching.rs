//! Runs three workloads with prefix caching off and on and prints what it
//! saves. Both runs of a workload see the identical request stream.
//!
//!     cargo run --release -p paged-kv-core --example prefix_caching

use paged_kv_core::{CacheConfig, Scheduler, SeqId};

const STEPS: usize = 5_000;
const NUM_BLOCKS: usize = 512;
const BLOCK_SIZE: usize = 16;
const MAX_BATCH: usize = 32;
/// The queue is topped up to this depth before every step.
const QUEUE_DEPTH: usize = 8;

struct Workload {
    name: &'static str,
    /// Lengths of the shared prefixes (system prompts, few-shot examples).
    shared_prefixes: &'static [usize],
    /// Percentage of requests that start with one of the shared prefixes.
    shared_percent: usize,
    user_tokens: (usize, usize),
    output_tokens: (usize, usize),
}

const WORKLOADS: &[Workload] = &[
    Workload {
        name: "Chat, 4 system prompts",
        shared_prefixes: &[220, 300, 380, 450],
        shared_percent: 90,
        user_tokens: (16, 128),
        output_tokens: (32, 256),
    },
    Workload {
        name: "Few-shot, 1 long prefix",
        shared_prefixes: &[1_000],
        shared_percent: 100,
        user_tokens: (20, 60),
        output_tokens: (1, 8),
    },
    Workload {
        name: "No shared prefixes",
        shared_prefixes: &[],
        shared_percent: 0,
        user_tokens: (200, 500),
        output_tokens: (32, 256),
    },
];

#[derive(Default)]
struct Report {
    finished: usize,
    prompt_tokens: usize,
    cached_tokens: usize,
    preemptions: usize,
    batch_total: usize,
    evictions: u64,
}

struct Lcg(u64);

impl Lcg {
    fn range(&mut self, (lo, hi): (usize, usize)) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        lo + ((self.0 >> 33) as usize) % (hi - lo + 1)
    }
}

fn run(workload: &Workload, caching: bool) -> Report {
    let config = CacheConfig {
        num_blocks: NUM_BLOCKS,
        block_size: BLOCK_SIZE,
        num_kv_heads: 1,
        head_dim: 1,
        num_layers: 1,
        dtype_bytes: 2,
    };
    let mut scheduler = if caching {
        Scheduler::with_prefix_caching(&config, MAX_BATCH)
    } else {
        Scheduler::new(&config, MAX_BATCH)
    };
    let prefixes: Vec<Vec<u32>> = workload
        .shared_prefixes
        .iter()
        .enumerate()
        .map(|(p, &len)| {
            (0..len as u32)
                .map(|t| ((p as u32 + 1) << 24) | t)
                .collect()
        })
        .collect();

    let mut rng = Lcg(42);
    let mut unique = 0u32;
    let mut next_id = 0u64;
    let mut report = Report::default();

    for _ in 0..STEPS {
        while scheduler.num_waiting() < QUEUE_DEPTH {
            // Fixed draws per request, so both runs get the same stream.
            let shared = rng.range((1, 100)) <= workload.shared_percent;
            let which = rng.range((0, prefixes.len().max(1) - 1));
            let user = rng.range(workload.user_tokens);
            let output = rng.range(workload.output_tokens);

            let mut prompt = match prefixes.get(which) {
                Some(prefix) if shared => prefix.clone(),
                _ => Vec::new(),
            };
            prompt.extend((0..user).map(|_| {
                unique += 1;
                unique
            }));
            scheduler
                .add_prompt(SeqId(next_id), prompt, output)
                .expect("requests fit the pool");
            next_id += 1;
        }

        let outcome = scheduler.step().expect("scheduler step");
        for p in &outcome.prefills {
            report.prompt_tokens += p.cached_tokens + p.computed_tokens;
            report.cached_tokens += p.cached_tokens;
        }
        report.finished += outcome.finished.len();
        report.preemptions += outcome.preempted.len();
        report.batch_total += scheduler.num_running();
    }
    report.evictions = scheduler.cache_stats().evictions;
    report
}

fn main() {
    println!("{NUM_BLOCKS} blocks of {BLOCK_SIZE} tokens, batch of {MAX_BATCH}, {STEPS} steps\n");
    println!(
        "{:<26} {:>8} {:>9} {:>12} {:>11} {:>9} {:>11} {:>10}",
        "workload",
        "caching",
        "finished",
        "prefilled",
        "from cache",
        "hit rate",
        "mean batch",
        "preempted"
    );
    for workload in WORKLOADS {
        for caching in [false, true] {
            let r = run(workload, caching);
            let computed = r.prompt_tokens - r.cached_tokens;
            println!(
                "{:<26} {:>8} {:>9} {:>12} {:>11} {:>8.1}% {:>11.1} {:>10}",
                workload.name,
                if caching { "on" } else { "off" },
                r.finished,
                computed,
                r.cached_tokens,
                100.0 * r.cached_tokens as f64 / r.prompt_tokens.max(1) as f64,
                r.batch_total as f64 / STEPS as f64,
                r.preemptions,
            );
            if caching && r.evictions > 0 {
                println!("{:<26} {:>8} evictions: {}", "", "", r.evictions);
            }
        }
    }
}
