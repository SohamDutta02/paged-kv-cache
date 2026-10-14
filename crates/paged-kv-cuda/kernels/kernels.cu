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
// Attention reduces across the thread block, so its thread count is fixed.
// Must equal ATTENTION_THREADS in backend.rs.
#define ATTN_THREADS 128

// Every thread gets the block-wide max or sum of `x`. `buf` holds
// ATTN_THREADS floats and is free for reuse on return.
__device__ float block_reduce(float x, float* buf, bool take_max)
{
    buf[threadIdx.x] = x;
    __syncthreads();
    for (int stride = ATTN_THREADS / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            const float other = buf[threadIdx.x + stride];
            buf[threadIdx.x] = take_max ? fmaxf(buf[threadIdx.x], other)
                                        : buf[threadIdx.x] + other;
        }
        __syncthreads();
    }
    const float result = buf[0];
    __syncthreads();
    return result;
}

// One thread block per (query, head). A query at position `pos` attends to
// positions 0..=pos of its sequence, read through the block table. Scores go
// through `scratch`, so context length isn't limited by shared memory.
extern "C" __global__ void paged_attention_kernel(
    float* __restrict__ out,                // [num_queries, num_heads, head_dim]
    float* __restrict__ scratch,            // [num_queries, num_heads, max_context]
    const float* __restrict__ q,            // same shape as out
    const float* __restrict__ key_cache,
    const float* __restrict__ value_cache,
    const int* __restrict__ block_tables,   // [num_seqs, max_blocks]
    const int* __restrict__ query_seq,      // [num_queries]: row of block_tables
    const int* __restrict__ query_pos,      // [num_queries]
    int max_blocks,
    int max_context,
    int layer,
    int num_blocks,
    int block_size,
    int num_kv_heads,
    int num_heads,
    int head_dim,
    float scale)
{
    __shared__ float buf[ATTN_THREADS];
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int pos = query_pos[row];
    const int* table = block_tables + (long long)query_seq[row] * max_blocks;
    const long long entry_elems = (long long)num_kv_heads * head_dim;
    const long long head_offset = (long long)(head / (num_heads / num_kv_heads)) * head_dim;
    const long long layer_base = (long long)layer * num_blocks;
    const float* qh = q + ((long long)row * num_heads + head) * head_dim;
    float* scores = scratch + ((long long)row * num_heads + head) * max_context;

    float local_max = -3.402823466e+38f;
    for (int t = threadIdx.x; t <= pos; t += ATTN_THREADS) {
        const float* k = key_cache + head_offset
            + ((layer_base + table[t / block_size]) * block_size + t % block_size) * entry_elems;
        float s = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            s += qh[i] * k[i];
        }
        s *= scale;
        scores[t] = s;
        local_max = fmaxf(local_max, s);
    }
    const float max_score = block_reduce(local_max, buf, true);

    float local_sum = 0.0f;
    for (int t = threadIdx.x; t <= pos; t += ATTN_THREADS) {
        const float p = expf(scores[t] - max_score);
        scores[t] = p;
        local_sum += p;
    }
    // The reduction's barriers also make every thread's scores visible.
    const float sum = block_reduce(local_sum, buf, false);

    float* oh = out + ((long long)row * num_heads + head) * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += ATTN_THREADS) {
        float acc = 0.0f;
        for (int t = 0; t <= pos; ++t) {
            const float* v = value_cache + head_offset
                + ((layer_base + table[t / block_size]) * block_size + t % block_size) * entry_elems;
            acc += scores[t] * v[i];
        }
        oh[i] = acc / sum;
    }
}
