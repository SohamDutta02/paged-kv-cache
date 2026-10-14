// Compiled at runtime by NVRTC. Addressing must match CpuBackend exactly:
//
//   offset = (((layer * num_blocks + block) * block_size + slot)
//              * entry_elems) + i
//
// Strided loops make the kernels correct for any thread-block size.
// `extern "C"` keeps names unmangled so cudarc can load them by name.

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
    // One thread block per token.
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
    // blockIdx.x encodes (pair, layer), so one launch covers every copy.
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