# paged-kv-cache

A vLLM PagedAttention-inspired KV cache memory manager, built from scratch in Rust: block allocator, block table, copy-on-write for beam search, a continuous-batching scheduler, and a CUDA backend. It is split across a three-crate workspace, with a CPU reference backend that serves as a numerical oracle.

**Live demo:** [paged-kv-cache.onrender.com](https://paged-kv-cache.onrender.com) — the real scheduler running on a 64-block pool, with a live block grid and event feed. If it's been idle, the first load may take a moment while the instance wakes up.

---

## The problem

Naive KV cache implementations pre-allocate a contiguous buffer sized for each sequence's *maximum* possible length. Most of that buffer sits empty for most of a sequence's life. This was the single biggest source of wasted GPU memory in early LLM serving stacks, and it's why batch sizes stayed small even on large GPUs.

PagedAttention (vLLM, 2023) borrows the fix operating systems have used for virtual memory since the 1960s: don't reserve a contiguous slab up front. Split the cache into small fixed-size physical blocks, and give each sequence a *logical* block table mapping its token positions onto whichever physical blocks it currently owns. A sequence only consumes blocks for tokens it has actually generated. Two sequences that share a prefix (a system prompt, or a beam-search fork) can point at the same physical blocks instead of duplicating them, and only pay for a copy the moment one of them diverges.

```
 seq A block table          physical pool                seq B block table
 (parent)                   (refcounted)                 (forked from A)

 logical 0 ──────────────►  block 7   rc=2  ◄──────────── logical 0
 logical 1 ──────────────►  block 2   rc=2  ◄──────────── logical 1
 logical 2 ──────────────►  block 11  rc=1                logical 2 ─┐
                            block 4   rc=1  ◄────────────────────────┘
                                         ▲
                     B wrote into the shared tail block, so it got a
                     private copy (CoW). Earlier blocks stay shared.
```

This project implements that idea from first principles, in Rust, with a CUDA kernel layer wired in behind a device-agnostic trait.

## Architecture

```
crates/
├── paged-kv-core/          device-agnostic memory management (zero dependencies)
│   ├── types.rs            BlockId / SeqId / PhysicalSlot newtypes
│   ├── config.rs           pool geometry, stride math, memory accounting
│   ├── allocator.rs        BlockAllocator: refcounted free-list pool
│   ├── block_table.rs      BlockTable: logical→physical resolution, CoW fork
│   ├── scheduler.rs        Scheduler: admission, batching, preemption, fork
│   ├── backend.rs          KvBackend trait: the CPU/CUDA seam
│   ├── cpu.rs              CpuBackend: reference implementation, numerical oracle
│   └── error.rs            CacheError
│
├── paged-kv-cuda/          CUDA backend, feature-gated (--features cuda)
│   ├── kernels/kernels.cu  reshape_and_cache, copy_blocks, zero_block
│   └── src/backend.rs      cudarc FFI layer implementing KvBackend
│
└── paged-kv-server/        systems demo: axum server + live dashboard
    ├── src/main.rs         wraps Scheduler in an HTTP API + background loop
    └── static/index.html   live block-grid / event-feed dashboard
```

`paged-kv-core` has **zero dependencies**, on purpose. Every allocation-policy, copy-on-write, and scheduling decision is pure Rust bookkeeping. None of it needs a GPU, so all of it is testable on a laptop with no NVIDIA hardware and no CUDA toolkit. Device-specific work sits entirely behind the `KvBackend` trait.

## Design decisions worth knowing about

**Copy-on-write is lazy and block-granular, not sequence-granular.** Forking a sequence (`BlockTable::fork`) increfs every block it owns; zero bytes are copied. The copy only happens inside `append`, only for the *tail* block, and only if that block is still shared at the moment of the write. A beam-search fork that diverges late still shares almost its entire prefix in memory, and earlier blocks can stay shared indefinitely.

**Preemption sacrifices the newest arrival, not the current one.** When the scheduler can't get a block, it evicts the *most recently admitted* running sequence. That may be the sequence currently failing to get a block, if it is itself the newest. This matches vLLM's default recompute policy: the newest arrival has the least invested work to lose. A preempted sequence is requeued at the front of the waiting queue with its current context length, and resumes once blocks free up.

**`CpuBackend` is a numerical oracle, not a fallback.** It uses the exact same physical addressing formula as the CUDA kernels:

```
(((layer * num_blocks + block) * block_size + slot) * entry_elems) + i
```

Any behavioral difference between the two backends is a kernel bug, not a design choice. This is what the CUDA backend's (pending) test suite compares against.

**Admission is strict FIFO.** If the request at the front of the waiting queue doesn't fit right now, the scheduler doesn't skip ahead to smaller requests behind it, even if those would fit. This prevents large requests from starving indefinitely behind a stream of small ones, at the cost of occasionally idle blocks.

## Status

| Component | Status |
|---|---|
| `paged-kv-core` (allocator, block table, scheduler) | **45 tests passing.** Fully verified, including a 2,000-iteration randomized churn test and hand-traced preemption/CoW scenarios. |
| `paged-kv-cuda` (kernels + FFI) | **Written, unverified.** Kernel addressing is hand-proven equivalent to `CpuBackend`'s formula. The `cudarc` API surface was confirmed against live documentation and a working scratch build, but not run; the first real compile happens on rented GPU hardware. Oracle-comparison tests exist and have never executed. |
| `paged-kv-server` (demo + dashboard) | **Fully tested.** Built, run, curled, and load-tested locally during development; now live in production. |
| `paged_attention` (real attention over the cache) | **Not yet implemented.** Deliberately left off the trait: its signature (score dtype, chunked-prefill state) wasn't decided when the rest of the trait was designed. |

## Running it

```bash
# core logic: fast, no GPU needed
cargo test -p paged-kv-core

# the live demo, locally
cargo run --bin paged-kv-server
# → http://localhost:8080

# same thing, containerized
docker build -t paged-kv-demo .
docker run -p 8080:8080 paged-kv-demo

# CUDA backend: requires an NVIDIA GPU + driver
cargo test -p paged-kv-cuda --features cuda
```

The server binds to `$PORT` if set (as on Render), otherwise 8080. `STATIC_DIR` overrides where the dashboard is served from; the Docker image sets it.

### Demo HTTP API

The demo runs the real `Scheduler` over a 64-block pool (8 tokens per block, max batch of 6), stepping every 400 ms. A background loop injects a random request every 3 ticks and forks a random running sequence every 5 ticks, so you can watch admission, CoW, and preemption happen without doing anything.

| Endpoint | Description |
|---|---|
| `GET /state` | Current tick, pool utilization, per-block refcounts, running sequences with their block tables, and the last 200 events (`admitted`, `cow`, `preempted`, `finished`). |
| `POST /requests` | Submit a request. Body: `{"prompt_len": 32, "max_new_tokens": 16}`. Returns `{"id": ...}`, or `400` with an error if the scheduler rejects it. |

```bash
curl -X POST localhost:8080/requests \
  -H 'content-type: application/json' \
  -d '{"prompt_len": 32, "max_new_tokens": 16}'
```

## Roadmap

- Verify the CUDA backend against `CpuBackend` on real hardware
- Implement `paged_attention` as a fourth kernel
- Wire a real small model (candle) through the paged cache for genuine generation
- Throughput/memory benchmarks vs. a naive per-sequence buffer baseline

## License

MIT
