from os import PathLike

__version__: str
CUDA: bool
"""Whether this build can keep the K/V cache on a GPU."""

class StepOutcome:
    admitted: list[int]
    prefills: list[tuple[int, int, int]]
    """(id, cached_tokens, computed_tokens) for each admission."""
    decoded: list[tuple[int, int, int]]
    """(id, block, slot) where each running sequence's next token goes."""
    copies: list[tuple[int, int]]
    """(src, dst) block copies to make before writing: copy-on-write."""
    preempted: list[int]
    finished: list[int]

class Scheduler:
    """Continuous batching over a pool of fixed-size blocks, without a model."""

    def __init__(
        self, num_blocks: int, block_size: int, max_batch_size: int, prefix_caching: bool = False
    ) -> None: ...
    def add_prompt(self, id: int, prompt: list[int], max_new_tokens: int) -> None: ...
    def add_request(self, id: int, prompt_len: int, max_new_tokens: int) -> None: ...
    def finish(self, id: int) -> None: ...
    def step(self) -> StepOutcome: ...
    def running(self) -> list[tuple[int, int, list[int]]]:
        """(id, tokens, blocks in logical order) for each running sequence."""
    @property
    def num_running(self) -> int: ...
    @property
    def num_waiting(self) -> int: ...
    @property
    def utilization(self) -> float: ...
    @property
    def cache_stats(self) -> dict[str, int]: ...

class Engine:
    """Greedy generation with a Qwen2 checkpoint, K/V in the paged cache."""

    def __init__(
        self,
        model_dir: str | PathLike[str],
        num_blocks: int = 512,
        block_size: int = 16,
        max_batch_size: int = 8,
        prefix_caching: bool = True,
        device: str = "cpu",
        stop_tokens: list[int] | None = None,
    ) -> None: ...
    def add_request(self, prompt: list[int], max_new_tokens: int) -> int: ...
    def step(self) -> list[tuple[int, list[int]]]: ...
    def generate(self, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]: ...
    @property
    def is_idle(self) -> bool: ...
    @property
    def stats(self) -> dict[str, int]: ...
    @property
    def device(self) -> str: ...
