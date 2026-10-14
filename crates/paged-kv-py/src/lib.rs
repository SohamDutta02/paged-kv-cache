//! Python bindings: `paged_kv.Scheduler` runs the scheduler on its own, and
//! `paged_kv.Engine` generates with a Qwen2 checkpoint through the cache.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use paged_kv_core::{CacheConfig, CpuBackend, SeqId};
use paged_kv_model::{load_config, EngineStats, PagedQwen2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn value_error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn runtime_error(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// What one scheduler step did.
#[pyclass(module = "paged_kv", frozen, get_all)]
struct StepOutcome {
    admitted: Vec<u64>,
    /// `(id, cached_tokens, computed_tokens)` for each admission.
    prefills: Vec<(u64, usize, usize)>,
    /// `(id, block, slot)` where each running sequence's next token goes.
    decoded: Vec<(u64, u32, usize)>,
    /// `(src, dst)` block copies to make before writing: copy-on-write.
    copies: Vec<(u32, u32)>,
    preempted: Vec<u64>,
    finished: Vec<u64>,
}

#[pymethods]
impl StepOutcome {
    fn __repr__(&self) -> String {
        format!(
            "StepOutcome(admitted={:?}, decoded={}, preempted={:?}, finished={:?})",
            self.admitted,
            self.decoded.len(),
            self.preempted,
            self.finished
        )
    }
}

/// Continuous batching over a pool of fixed-size blocks, without a model.
#[pyclass(module = "paged_kv")]
struct Scheduler {
    inner: paged_kv_core::Scheduler,
}

#[pymethods]
impl Scheduler {
    #[new]
    #[pyo3(signature = (num_blocks, block_size, max_batch_size, prefix_caching = false))]
    fn new(
        num_blocks: usize,
        block_size: usize,
        max_batch_size: usize,
        prefix_caching: bool,
    ) -> PyResult<Self> {
        if num_blocks == 0 || block_size == 0 || max_batch_size == 0 {
            return Err(value_error(
                "num_blocks, block_size and max_batch_size must be positive",
            ));
        }
        // Scheduling only depends on the block geometry.
        let config = CacheConfig {
            num_blocks,
            block_size,
            num_kv_heads: 1,
            head_dim: 1,
            num_layers: 1,
            dtype_bytes: 2,
        };
        let inner = if prefix_caching {
            paged_kv_core::Scheduler::with_prefix_caching(&config, max_batch_size)
        } else {
            paged_kv_core::Scheduler::new(&config, max_batch_size)
        };
        Ok(Self { inner })
    }

    /// Queues a request by its prompt tokens, which prefix caching can match.
    fn add_prompt(&mut self, id: u64, prompt: Vec<u32>, max_new_tokens: usize) -> PyResult<()> {
        self.inner
            .add_prompt(SeqId(id), prompt, max_new_tokens)
            .map_err(value_error)
    }

    /// Queues a request known only by its prompt length.
    fn add_request(&mut self, id: u64, prompt_len: usize, max_new_tokens: usize) -> PyResult<()> {
        self.inner
            .add_request(SeqId(id), prompt_len, max_new_tokens)
            .map_err(value_error)
    }

    /// Retires a running or waiting sequence early, freeing its blocks.
    fn finish(&mut self, id: u64) -> PyResult<()> {
        self.inner.finish_sequence(SeqId(id)).map_err(value_error)
    }

    /// Admits what fits and gives every running sequence a slot for one token.
    fn step(&mut self) -> PyResult<StepOutcome> {
        let o = self.inner.step().map_err(runtime_error)?;
        Ok(StepOutcome {
            admitted: o.admitted.iter().map(|id| id.0).collect(),
            prefills: o
                .prefills
                .iter()
                .map(|p| (p.id.0, p.cached_tokens, p.computed_tokens))
                .collect(),
            decoded: o
                .decoded
                .iter()
                .map(|d| (d.id.0, d.slot.block.0, d.slot.slot))
                .collect(),
            copies: o
                .decoded
                .iter()
                .filter_map(|d| d.cow)
                .map(|c| (c.src.0, c.dst.0))
                .collect(),
            preempted: o.preempted.iter().map(|id| id.0).collect(),
            finished: o.finished.iter().map(|id| id.0).collect(),
        })
    }

    /// Each running sequence's id, token count and blocks in logical order.
    fn running(&self) -> Vec<(u64, usize, Vec<u32>)> {
        self.inner
            .running_snapshot()
            .into_iter()
            .map(|r| (r.id.0, r.tokens, r.blocks.iter().map(|b| b.0).collect()))
            .collect()
    }

    #[getter]
    fn num_running(&self) -> usize {
        self.inner.num_running()
    }

    #[getter]
    fn num_waiting(&self) -> usize {
        self.inner.num_waiting()
    }

    /// Fraction of blocks in use; cached blocks with no owner count as free.
    #[getter]
    fn utilization(&self) -> f64 {
        self.inner.pool_utilization()
    }

    #[getter]
    fn cache_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = self.inner.cache_stats();
        let d = PyDict::new(py);
        d.set_item("cached_blocks", stats.cached_blocks)?;
        d.set_item("evictable_blocks", stats.evictable_blocks)?;
        d.set_item("evictions", stats.evictions)?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "Scheduler(running={}, waiting={}, utilization={:.2})",
            self.inner.num_running(),
            self.inner.num_waiting(),
            self.inner.pool_utilization()
        )
    }
}

enum Backend {
    Cpu(paged_kv_model::Engine<CpuBackend>),
    #[cfg(feature = "cuda")]
    Cuda(paged_kv_model::Engine<paged_kv_cuda::CudaBackend>),
}

macro_rules! with_engine {
    ($backend:expr, $engine:ident => $body:expr) => {
        match $backend {
            Backend::Cpu($engine) => $body,
            #[cfg(feature = "cuda")]
            Backend::Cuda($engine) => $body,
        }
    };
}

/// Greedy generation with a Qwen2 checkpoint, K/V in the paged cache.
#[pyclass(module = "paged_kv")]
struct Engine {
    backend: Backend,
    device: String,
    next_id: u64,
}

fn safetensors_files(dir: &Path) -> PyResult<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| value_error(format!("{}: {e}", dir.display())))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(value_error(format!(
            "no .safetensors files in {}",
            dir.display()
        )));
    }
    Ok(files)
}

fn stats_dict<'py>(py: Python<'py>, s: EngineStats) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("steps", s.steps)?;
    d.set_item("prefill_tokens_computed", s.prefill_tokens_computed)?;
    d.set_item("prefill_tokens_cached", s.prefill_tokens_cached)?;
    d.set_item("preemptions", s.preemptions)?;
    d.set_item("tokens_generated", s.tokens_generated)?;
    Ok(d)
}

#[pymethods]
impl Engine {
    /// Loads `config.json` and the `.safetensors` weights from `model_dir`.
    #[new]
    #[pyo3(signature = (
        model_dir,
        num_blocks = 512,
        block_size = 16,
        max_batch_size = 8,
        prefix_caching = true,
        device = "cpu",
        stop_tokens = None,
    ))]
    fn new(
        model_dir: PathBuf,
        num_blocks: usize,
        block_size: usize,
        max_batch_size: usize,
        prefix_caching: bool,
        device: &str,
        stop_tokens: Option<Vec<u32>>,
    ) -> PyResult<Self> {
        let config = load_config(&model_dir.join("config.json")).map_err(value_error)?;
        let files = safetensors_files(&model_dir)?;
        // The weights are memory-mapped; the files must not change while loaded.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, &Device::Cpu) }
            .map_err(value_error)?;
        let model = PagedQwen2::new(&config, vb).map_err(value_error)?;
        let cache_config = model.cache_config(num_blocks, block_size);
        let stop = stop_tokens.unwrap_or_default();

        let backend = match device {
            "cpu" => Backend::Cpu(
                paged_kv_model::Engine::new(
                    model,
                    CpuBackend::new(cache_config),
                    max_batch_size,
                    prefix_caching,
                )
                .stop_at(&stop),
            ),
            #[cfg(feature = "cuda")]
            "cuda" => Backend::Cuda(
                paged_kv_model::Engine::new(
                    model,
                    paged_kv_cuda::CudaBackend::new(cache_config).map_err(runtime_error)?,
                    max_batch_size,
                    prefix_caching,
                )
                .stop_at(&stop),
            ),
            #[cfg(not(feature = "cuda"))]
            "cuda" => {
                return Err(value_error(
                    "this build has no CUDA support; build with the cuda feature",
                ))
            }
            other => return Err(value_error(format!("unknown device {other:?}"))),
        };
        Ok(Self {
            device: match &backend {
                Backend::Cpu(_) => "cpu".into(),
                #[cfg(feature = "cuda")]
                Backend::Cuda(_) => "cuda".into(),
            },
            backend,
            next_id: 0,
        })
    }

    /// Queues a prompt and returns its request id.
    fn add_request(&mut self, prompt: Vec<u32>, max_new_tokens: usize) -> PyResult<u64> {
        let id = self.next_id;
        with_engine!(&mut self.backend, e => e.add_request(SeqId(id), prompt, max_new_tokens))
            .map_err(value_error)?;
        self.next_id += 1;
        Ok(id)
    }

    /// Runs one batch through the model; returns `(id, tokens)` for each
    /// request that finished.
    fn step(&mut self, py: Python<'_>) -> PyResult<Vec<(u64, Vec<u32>)>> {
        let backend = &mut self.backend;
        let finished = py
            .detach(|| with_engine!(backend, e => e.step()))
            .map_err(runtime_error)?;
        Ok(finished.into_iter().map(|f| (f.id.0, f.tokens)).collect())
    }

    /// Generates greedily for every prompt as one continuous batch and returns
    /// each prompt's new tokens, in order. The engine must be idle.
    fn generate(
        &mut self,
        py: Python<'_>,
        prompts: Vec<Vec<u32>>,
        max_new_tokens: usize,
    ) -> PyResult<Vec<Vec<u32>>> {
        if !self.is_idle() {
            return Err(runtime_error(
                "generate needs an idle engine; finish the requests added with add_request first",
            ));
        }
        let ids = prompts
            .into_iter()
            .map(|p| self.add_request(p, max_new_tokens))
            .collect::<PyResult<Vec<u64>>>()?;

        let backend = &mut self.backend;
        let mut done: HashMap<u64, Vec<u32>> = py
            .detach(|| -> candle_core::Result<HashMap<u64, Vec<u32>>> {
                let mut done = HashMap::new();
                with_engine!(backend, e => {
                    while !e.is_idle() {
                        for f in e.step()? {
                            done.insert(f.id.0, f.tokens);
                        }
                    }
                });
                Ok(done)
            })
            .map_err(runtime_error)?;
        Ok(ids
            .iter()
            .map(|id| done.remove(id).unwrap_or_default())
            .collect())
    }

    #[getter]
    fn is_idle(&self) -> bool {
        with_engine!(&self.backend, e => e.is_idle())
    }

    /// Counters since the engine was created.
    #[getter]
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        stats_dict(py, with_engine!(&self.backend, e => e.stats()))
    }

    /// Where the K/V cache lives: `"cpu"` or `"cuda"`.
    #[getter]
    fn device(&self) -> &str {
        &self.device
    }

    fn __repr__(&self) -> String {
        format!("Engine(device={:?}, idle={})", self.device, self.is_idle())
    }
}

#[pymodule]
fn paged_kv(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Scheduler>()?;
    m.add_class::<StepOutcome>()?;
    m.add_class::<Engine>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("CUDA", cfg!(feature = "cuda"))?;
    Ok(())
}
