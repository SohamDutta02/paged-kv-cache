# Evaluating paged-kv-cache

Three ways in, depending on how much time you have. Each one points at output you can check for yourself.

## 2 minutes: the live demo

Open [paged-kv-cache.onrender.com](https://paged-kv-cache.onrender.com). If the instance has been idle, the first load takes a moment. Each visitor gets a private simulation running the real scheduler.

- **The block pool** is the KV cache. Blue blocks belong to one sequence, orange blocks are shared (forks, or prompts with a common system prompt), and a dashed outline marks a cached block that no sequence owns yet but a later request can reuse.
- **Click a running sequence** to trace its block table. Its tokens are contiguous to the sequence but scattered across memory, which is the point of paging.
- **The event log** shows the mechanisms at work: admissions that reuse cached tokens, copy-on-write when a fork writes to a shared block, and preemption when memory runs out.
- **Switch prefix caching off** and watch the pool fill faster with the same traffic.

## 5 minutes: read the code in this order

1. [`allocator.rs`](../crates/paged-kv-core/src/allocator.rs): refcounted blocks, the free list, and the prefix cache, which keeps unowned blocks reusable until memory is needed.
2. [`block_table.rs`](../crates/paged-kv-core/src/block_table.rs): `append` and `fork`, with copy-on-write of the tail block only when it is shared.
3. [`scheduler.rs`](../crates/paged-kv-core/src/scheduler.rs): `admit` (FIFO, decode headroom, prefix matching) and `decode` (newest-first preemption).
4. [`model.rs`](../crates/paged-kv-model/src/model.rs): where a real model writes its K/V into blocks and reads it back with `paged_attention`.
5. [`kernels.cu`](../crates/paged-kv-cuda/kernels/kernels.cu): the CUDA kernels, which address memory exactly like the CPU reference.

The [API docs](https://sohamdutta02.github.io/paged-kv-cache/) cover the public types, with a runnable example on the front page.

## 15 minutes: run it

No GPU is needed for any of this:

```bash
cargo test --workspace
cargo run --release -p paged-kv-core --example contiguous_baseline
cargo run --release -p paged-kv-core --example prefix_caching
```

The tests include randomized ones that check, after every operation, that each block's refcount equals the number of block tables pointing at it. The two benchmarks print the tables in the README's [Paged vs. contiguous buffers](../README.md#paged-vs-contiguous-buffers) and [Prefix caching](../README.md#prefix-caching) sections, and they are deterministic, so your numbers should match.

For a GPU, two Colab notebooks run on a free T4 (**Runtime → Change runtime type → T4 GPU**, then **Run all**):

- [GPU check](https://colab.research.google.com/github/SohamDutta02/paged-kv-cache/blob/main/notebooks/gpu_check.ipynb): every CUDA kernel against the CPU reference. Expect 6 tests passing.
- [Qwen2.5 check](https://colab.research.google.com/github/SohamDutta02/paged-kv-cache/blob/main/notebooks/qwen_check.ipynb): Qwen2.5-0.5B-Instruct through the cache, compared token by token with candle's own Qwen2. Expect all four answers to match, with the cache in host memory and again on the GPU.

## What is verified, and what isn't

Verified:

- Allocation, copy-on-write, prefix caching and scheduling keep every refcount consistent through thousands of random operations.
- The CUDA kernels match the CPU reference element for element on a Tesla T4, and the attention kernel matches to rounding.
- A real model generates exactly the reference implementation's tokens through the cache, including prefix-cache hits and preemption with recompute.

Not yet:

- In the Qwen check only attention runs on the GPU; the rest of the model runs on the CPU. There are no GPU throughput numbers yet.
- The attention kernel is written for clarity, with one thread block per query and head and no tiling. It is a correct reference, not a fast one.
- The benchmarks count work (tokens computed, blocks, sequences per step) in a simulation. They do not measure wall-clock time.
- One GPU, one process. There is no HTTP inference endpoint yet.

## Questions worth asking

- Why strict FIFO admission, instead of letting small requests skip ahead? See [Design decisions](../README.md#design-decisions-worth-knowing-about).
- Why preempt the newest sequence, and what does recomputing it cost? See [Paged vs. contiguous buffers](../README.md#paged-vs-contiguous-buffers).
- Why exact-match prefix keys instead of hashes? See [Prefix caching](../README.md#prefix-caching).
- How does the engine avoid attending to a token whose K/V hasn't been written? See [A real model](../README.md#a-real-model).
