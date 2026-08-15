# \# paged-kv-cache

# 

# A vLLM PagedAttention-inspired KV cache memory manager, built from scratch in Rust: block allocator, block table, copy-on-write for beam search, a continuous-batching scheduler, and a CUDA backend — split across a three-crate workspace with a CPU reference backend as a numerical oracle.

# 

# \*\*Live demo:\*\* the scheduler running for real against synthetic load, with a block-level dashboard: \*\*\[paged-kv-cache.onrender.com](https://paged-kv-cache.onrender.com)\*\*

# 

# \---

# 

# \## The problem

# 

# Naive KV cache implementations pre-allocate a contiguous buffer sized for each sequence's maximum possible length. Most of that buffer sits empty for most of a sequence's life — this is the single biggest source of wasted GPU memory in early LLM serving stacks, and it's the reason batch sizes stayed small even on large GPUs.

# 

# PagedAttention (vLLM, 2023) borrows the fix operating systems have used for virtual memory since the 1960s: don't reserve a contiguous slab up front. Split the cache into small fixed-size physical blocks, and give each sequence a \*logical\* block table mapping its token positions onto whichever physical blocks it currently owns. A sequence only consumes blocks for tokens it's actually generated. Two sequences that share a prefix (a system prompt, or a beam-search fork) can point at the \*same\* physical blocks instead of duplicating them — and only pay a copy the moment one of them actually diverges.

# 

# This project reimplements that idea from first principles, in Rust, with a CUDA kernel layer wired in behind a device-agnostic trait.

# 

# \## Architecture

crates/

├── paged-kv-core/ device-agnostic memory management — zero dependencies

│ ├── types.rs BlockId / SeqId / PhysicalSlot newtypes

│ ├── config.rs pool geometry, stride math, memory accounting

│ ├── allocator.rs BlockAllocator — refcounted free-list pool

│ ├── block\_table.rs BlockTable — logical→physical resolution, CoW fork

│ ├── scheduler.rs Scheduler — admission, batching, preemption, fork

│ ├── backend.rs KvBackend trait — the CPU/CUDA seam

│ ├── cpu.rs CpuBackend — reference implementation, numerical oracle

│ └── error.rs CacheError

│

├── paged-kv-cuda/ CUDA backend, feature-gated (--features cuda)

│ ├── kernels/kernels.cu reshape\_and\_cache, copy\_blocks, zero\_block

│ └── src/backend.rs cudarc FFI layer implementing KvBackend

│

└── paged-kv-server/ systems demo: axum server + live dashboard

├── src/main.rs wraps Scheduler in an HTTP API + background loop

└── static/index.html live block-grid / event-feed dashboard





`paged-kv-core` has \*\*zero dependencies\*\*, on purpose. Every allocation-policy, copy-on-write, and scheduling decision is pure Rust bookkeeping — none of it needs a GPU, so all of it is testable on a laptop with no NVIDIA hardware and no CUDA toolkit. Device-specific work sits entirely behind the `KvBackend` trait.



\## Design decisions worth knowing about



\*\*Copy-on-write is lazy and block-granular, not sequence-granular.\*\* Forking a sequence (`BlockTable::fork`) increfs every block it owns — zero bytes copied. The copy only happens inside `append`, only for the \*tail\* block, and only if that block is still shared at the moment of the write. A beam-search fork that diverges late still shares almost its entire prefix in memory. Earlier blocks can stay shared indefinitely.



\*\*Preemption sacrifices the newest arrival, not the current one.\*\* When the scheduler can't get a block, it evicts the \*most recently admitted\* running sequence — which may be the sequence currently failing to get a block, if it's itself the newest. This matches vLLM's default recompute policy: the newest arrival has the least invested work to lose. A preempted sequence is requeued at the front of the waiting queue with its current context length, and resumes once blocks free up.



\*\*`CpuBackend` is a numerical oracle, not a fallback.\*\* It uses the exact same physical addressing formula as the CUDA kernels (`(((layer \* num\_blocks + block) \* block\_size + slot) \* entry\_elems) + i`). Any behavioral difference between the two backends is a kernel bug, not a design choice — this is what the CUDA backend's (pending) test suite compares against.



\*\*Admission is strict FIFO.\*\* If the request at the front of the waiting queue doesn't fit right now, the scheduler doesn't skip ahead to smaller requests behind it, even if those would fit. Prevents large requests from starving indefinitely behind a stream of small ones, at the cost of occasionally idle blocks.



\## Status



| Component | Status |

|---|---|

| `paged-kv-core` (allocator, block table, scheduler) | \*\*45 tests passing.\*\* Fully verified, including a 2000-iteration randomized churn test and hand-traced preemption/CoW scenarios. |

| `paged-kv-cuda` (kernels + FFI) | \*\*Written, unverified.\*\* Kernel addressing is hand-proven equivalent to `CpuBackend`'s formula. The `cudarc` API surface was confirmed against live documentation and a working scratch build, not run — first real compile happens on rented GPU hardware. Oracle-comparison tests exist and have never executed. |

| `paged-kv-server` (demo + dashboard) | \*\*Fully tested — built, run, curled, and load-tested locally\*\* during development; now live in production. |

| `paged\_attention` (real attention over the cache) | \*\*Not yet implemented.\*\* Deliberately left off the trait — its signature (score dtype, chunked-prefill state) wasn't decided when the rest of the trait was designed. |



\## Running it



```bash

\# core logic — fast, no GPU needed

cargo test -p paged-kv-core



\# the live demo, locally

cargo run --bin paged-kv-server

\# → http://localhost:8080



\# same thing, containerized

docker build -t paged-kv-demo .

docker run -p 8080:8080 paged-kv-demo



\# CUDA backend — requires an NVIDIA GPU + driver

cargo test -p paged-kv-cuda --features cuda

```



\## Why Rust



Everything here — refcounting, free lists, index arithmetic — is the kind of code where a wrong index or a use-after-free is exactly the class of bug Rust's ownership model is built to catch at compile time, and where `unsafe` is confined to a handful of well-understood spots at the FFI boundary (the CUDA kernel launches) rather than smeared across the whole allocator.



\## Roadmap



\- Verify the CUDA backend against `CpuBackend` on real hardware

\- Implement `paged\_attention` as a fourth kernel

\- Wire a real small model (candle) through the paged cache for genuine generation

\- Throughput/memory benchmarks vs. a naive per-sequence buffer baseline



\## License



MIT

