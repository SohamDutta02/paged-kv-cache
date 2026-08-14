use crate::config::CacheConfig;
use crate::error::Result;
use crate::types::{BlockId, PhysicalSlot};

/// The seam between memory management and physical storage.
///
/// Everything above this trait — allocation policy, block tables,
/// copy-on-write, eviction, scheduling — is device-agnostic and lives in
/// `paged-kv-core`. Everything below it is a handful of memory movements that
/// happen to be CUDA kernels on real hardware and `memcpy` on a laptop.
///
/// Keeping the surface this narrow is deliberate. Each method below maps to
/// exactly one kernel in the CUDA implementation, which means the CPU backend
/// can serve as a numerical oracle: any behaviour that differs between
/// [`CpuBackend`](crate::CpuBackend) and the CUDA backend is a kernel bug, not
/// a design question.
///
/// Note the absence of a `paged_attention` method. It belongs here eventually,
/// but its signature depends on decisions not yet made — scores in f32 vs the
/// cache dtype, whether the caller supplies a partial softmax state for
/// chunked prefill. Adding a method to a trait later is cheap; unpicking a
/// signature that was guessed at is not.
pub trait KvBackend {
    /// The geometry this backend was built for.
    fn config(&self) -> &CacheConfig;

    /// Write one token's K and V into a physical slot, for one layer.
    ///
    /// `k` and `v` must each hold `config().entry_elems()` values, laid out
    /// head-major: head 0's `head_dim` values, then head 1's, and so on.
    ///
    /// CUDA counterpart: the `reshape_and_cache` kernel, which scatters a
    /// batch of these writes in one launch.
    fn write_kv(
        &mut self,
        layer: usize,
        dst: PhysicalSlot,
        k: &[f32],
        v: &[f32],
    ) -> Result<()>;

    /// Read one token's K and V back out of a physical slot.
    ///
    /// Not needed in a production serving path — attention reads the cache
    /// directly on device. It exists so that tests can assert on cache
    /// contents, which is what makes copy-on-write correctness checkable
    /// without a GPU.
    fn read_kv(
        &self,
        layer: usize,
        src: PhysicalSlot,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<()>;

    /// Copy the entire contents of one block to another, across every layer
    /// and both K and V planes.
    ///
    /// This is the expensive half of copy-on-write, invoked when a sequence
    /// writes to a block it shares with someone else.
    ///
    /// CUDA counterpart: the `copy_blocks` kernel.
    fn copy_block(&mut self, src: BlockId, dst: BlockId) -> Result<()>;

    /// Copy several blocks in one go.
    ///
    /// Provided as a method rather than a loop at the call site because on
    /// device this is a single kernel launch over all pairs, and launch
    /// overhead dominates for the small copies that CoW produces.
    fn copy_blocks(&mut self, pairs: &[(BlockId, BlockId)]) -> Result<()> {
        for &(src, dst) in pairs {
            self.copy_block(src, dst)?;
        }
        Ok(())
    }

    /// Zero a block. Not required for correctness, since live slots are always
    /// written before they are read, but it makes debugging far less painful:
    /// stale data from a previous tenant reads as plausible garbage, whereas
    /// zeros read as obviously wrong.
    fn zero_block(&mut self, block: BlockId) -> Result<()>;

    /// Bytes this backend has committed for the pool.
    fn allocated_bytes(&self) -> usize {
        self.config().total_bytes()
    }

    /// Human-readable name, for logs and benchmark output.
    fn device_name(&self) -> String;
}
