"""Generates with paged_kv and with Hugging Face transformers, and compares.

    python compare_with_transformers.py <model-dir> [max-new-tokens]

<model-dir> is a local snapshot of Qwen/Qwen2.5-0.5B-Instruct. Both sides use
plain greedy decoding in float32, so they should agree token for token.
"""

import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import paged_kv

SYSTEM = (
    "You are a concise assistant. Answer every question in one or two sentences, "
    "in plain language, without lists or headings."
)
QUESTIONS = [
    "What does a GPU do?",
    "Why is the sky blue?",
    "What is a hash table?",
    "Why do we sleep?",
]


def transformers_greedy(model, prompt, max_new, stop):
    """Argmax with the model's own KV cache and no logits processors."""
    out = []
    with torch.no_grad():
        result = model(input_ids=torch.tensor([prompt]), use_cache=True)
        while len(out) < max_new:
            token = int(result.logits[0, -1].argmax())
            out.append(token)
            if token in stop:
                break
            result = model(
                input_ids=torch.tensor([[token]]),
                past_key_values=result.past_key_values,
                use_cache=True,
            )
    return out


def main():
    model_dir = sys.argv[1]
    max_new = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    stop = [tokenizer.convert_tokens_to_ids(t) for t in ["<|im_end|>", "<|endoftext|>"]]
    prompts = [
        tokenizer.encode(
            tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
            ),
            add_special_tokens=False,
        )
        for q in QUESTIONS
    ]

    engine = paged_kv.Engine(model_dir, stop_tokens=stop)
    started = time.perf_counter()
    paged = engine.generate(prompts, max_new_tokens=max_new)
    paged_secs = time.perf_counter() - started

    # The checkpoint is bfloat16; both sides upcast it to float32 the same way.
    model = AutoModelForCausalLM.from_pretrained(model_dir).float().eval()
    started = time.perf_counter()
    reference = [transformers_greedy(model, p, max_new, stop) for p in prompts]
    reference_secs = time.perf_counter() - started

    mismatches = 0
    for question, got, want in zip(QUESTIONS, paged, reference):
        print(f"Q: {question}\nA: {tokenizer.decode(got, skip_special_tokens=True).strip()}\n")
        if got != want:
            mismatches += 1
            at = next(
                (i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want))
            )
            print(f"   differs from transformers at generated token {at}\n")

    stats = engine.stats
    print(
        f"Prompt tokens: {stats['prefill_tokens_computed']} computed, "
        f"{stats['prefill_tokens_cached']} from the prefix cache."
    )
    print(
        f"{stats['tokens_generated']} tokens in {paged_secs:.1f} s with paged_kv (batched), "
        f"{reference_secs:.1f} s with transformers (one prompt at a time)."
    )
    if mismatches:
        sys.exit(f"{mismatches} of {len(prompts)} answers differ from transformers")
    print(f"All {len(prompts)} answers match Hugging Face transformers token for token.")


if __name__ == "__main__":
    main()
