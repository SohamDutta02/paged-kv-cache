//! Scheduler and serving layer. Populated in a later step; for now it exists
//! so the workspace shape is fixed and the dependency direction is enforced
//! from the start: server -> core, cuda -> core, and never core -> anything.

use paged_kv_core::{BlockAllocator, CacheConfig, CpuBackend, KvBackend};

fn main() {
    let config = CacheConfig::llama_3_2_1b(2048);
    let allocator = BlockAllocator::new(config.num_blocks);
    let backend = CpuBackend::new(config.clone());

    println!("paged-kv-server (scaffold)");
    println!("  backend       : {}", backend.device_name());
    println!("  blocks        : {}", allocator.num_blocks());
    println!("  block size    : {} tokens", config.block_size);
    println!("  capacity      : {} tokens", config.num_blocks * config.block_size);
    println!("  bytes/token   : {} KiB", config.bytes_per_token() / 1024);
    println!("  pool footprint: {:.2} GiB", config.total_bytes() as f64 / (1 << 30) as f64);
}
