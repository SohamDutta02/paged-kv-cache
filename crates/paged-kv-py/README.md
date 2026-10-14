# paged-kv (Python)

Python bindings for [paged-kv-cache](https://github.com/SohamDutta02/paged-kv-cache), built with PyO3 and maturin.

- `paged_kv.Engine` loads a Qwen2 checkpoint from a directory (`config.json` plus `.safetensors`) and generates greedily, with every layer's K/V in the paged cache.
- `paged_kv.Scheduler` runs the scheduler on its own, for experimenting with batching and memory policies without a model.

```bash
pip install ./crates/paged-kv-py       # needs a Rust toolchain
```

```python
import paged_kv
from transformers import AutoTokenizer

model_dir = "Qwen2.5-0.5B-Instruct"     # a local snapshot of the Hugging Face repo
tokenizer = AutoTokenizer.from_pretrained(model_dir)
stop = [tokenizer.convert_tokens_to_ids(t) for t in ["<|im_end|>", "<|endoftext|>"]]
engine = paged_kv.Engine(model_dir, stop_tokens=stop)

def chat(question):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
    )
    return tokenizer.encode(text, add_special_tokens=False)

prompts = [chat(q) for q in ["What is a hash table?", "Why is the sky blue?"]]
for answer in engine.generate(prompts, max_new_tokens=64):
    print(tokenizer.decode(answer, skip_special_tokens=True))
print(engine.stats)
```
