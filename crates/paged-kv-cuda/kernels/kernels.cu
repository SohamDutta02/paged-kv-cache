// kernels.cu
//
// CUDA kernels backing paged-kv-cuda's CudaBackend. Compiled at *runtime* via
// NVRTC (see backend.rs), not ahead-of-time with nvcc — so building this
// crate never requires a CUDA toolchain at all. Only running it with the
// `cuda` feature enabled, on a machine with an NVIDIA driver, does.
//
// Every kernel here addresses the pool with the exact same formula as
// paged-kv-core's CpuBackend (see cpu.rs's module doc):
//
//   offset = (((layer * num_blocks + block) * block_size + slot)
//              * entry_elems) + i        for i in [0, entry_elems)
//
// where entry_elems = num_kv_heads * head_dim. Keeping the two backends'
// addressing identical is the entire point: it's what makes CpuBackend a
// valid numerical oracle. Any mismatch between what these kernels write and
// what CpuBackend computes for the same operations is a kernel bug, not a
// design difference — see backend.rs's oracle-comparison tests.
//
// All three kernels use a grid-stride loop over `entry_elems` /
// `block_elems` within each thread block, so the thread-block size (chosen
// on the Rust side, see THREADS_PER_BLOCK in backend.rs) never has to match
// those values exactly — correct whether the config is the 32-element `tiny`
// test fixture or a real model's multi-thousand-element blocks.
//
// `extern "C"` on every kernel prevents C++ name mangling: cudarc looks up
// each function by its exact string name after loading the compiled module,
// so a mangled name would make `load_function("reshape_and_cache_kernel")`
// fail to find anything.

extern "C" __global__ void reshape_and_cache_kernel(
    const float* __restrict__ key,          // [num_tokens, entry_elems], row-major
    const float* __restrict__ value,        // same shape as key
    float* __restrict__ key_cache,          // flat pool, see offset formula above
    float* __restrict__ value_cache,
    const long long* __restrict__ slot_mapping, // [num_tokens]: block*block_size + local_slot
    int layer,
    int num_blocks,
    int block_size,
    int entry_elems)
{
    // One thread block per token. Each token's slot_mapping entry is a
    // single combined index (matching vLLM's own convention) rather than a
    // separate (block, slot) pair, which keeps the per-token address
    // computation to one division and one modulo.
    const int token = blockIdx.x;
    const long long slot = slot_mapping[token];
    const long long block = slot / block_size;
    const long long local_slot = slot % block_size;

    const long long cache_base =
        (((long long)layer * num_blocks + block) * block_size + local_slot)
        * entry_elems;
    const long long src_base = (long long)token * entry_elems;

    for (int i = threadIdx.x; i < entry_elems; i += blockDim.x) {
        key_cache[cache_base + i]   = key[src_base + i];
        value_cache[cache_base + i] = value[src_base + i];
    }
}

extern "C" __global__ void copy_blocks_kernel(
    float* __restrict__ key_cache,
    float* __restrict__ value_cache,
    const long long* __restrict__ block_pairs, // [num_pairs * 2]: src0,dst0,src1,dst1,...
    int num_pairs,
    int num_layers,
    int num_blocks,
    int block_elems)  // block_size * entry_elems
{
    // Flattened (pair, layer) grid: blockIdx.x encodes both, so one launch
    // handles every pair across every layer. Batching this way — rather than
    // one launch per pair — is the entire reason copy_blocks exists as a
    // distinct operation from a loop of single-block copies: copy-on-write
    // tends to produce several small copies per scheduler step, and launch
    // overhead would otherwise dominate.
    const int idx = blockIdx.x;
    const int pair = idx / num_layers;
    const int layer = idx % num_layers;
    if (pair >= num_pairs) return;

    const long long src_block = block_pairs[pair * 2 + 0];
    const long long dst_block = block_pairs[pair * 2 + 1];

    const long long src_base = ((long long)layer * num_blocks + src_block) * block_elems;
    const long long dst_base = ((long long)layer * num_blocks + dst_block) * block_elems;

    for (int i = threadIdx.x; i < block_elems; i += blockDim.x) {
        key_cache[dst_base + i]   = key_cache[src_base + i];
        value_cache[dst_base + i] = value_cache[src_base + i];
    }
}

extern "C" __global__ void zero_block_kernel(
    float* __restrict__ key_cache,
    float* __restrict__ value_cache,
    int block,
    int num_layers,
    int num_blocks,
    int block_elems)
{
    // One thread block per layer; grid.x == num_layers.
    const int layer = blockIdx.x;
    const long long base = ((long long)layer * num_blocks + block) * block_elems;

    for (int i = threadIdx.x; i < block_elems; i += blockDim.x) {
        key_cache[base + i] = 0.0f;
        value_cache[base + i] = 0.0f;
    }
}