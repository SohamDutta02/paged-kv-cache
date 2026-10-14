"""The engine against Hugging Face transformers on a tiny random Qwen2."""

import os

import pytest

# CI must run these, so a missing dependency fails there instead of skipping.
if os.environ.get("CI"):
    import torch
    import transformers
else:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

import paged_kv  # noqa: E402

VOCAB = 128
NEW_TOKENS = 12


@pytest.fixture(scope="module")
def tiny_qwen2(tmp_path_factory):
    config = transformers.Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = transformers.Qwen2ForCausalLM(config).eval()
    # A small embedding and strong attention output make each token depend
    # on context, so wrong K/V changes what gets generated.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.uniform_(0.8, 1.2)
            elif "embed_tokens" in name:
                p.uniform_(-0.3, 0.3)
            elif name.endswith("bias"):
                p.uniform_(-0.1, 0.1)
            else:
                scale = 1.5 / p.shape[1] ** 0.5
                if "v_proj" in name or "o_proj" in name:
                    scale *= 3
                p.uniform_(-scale, scale)
    path = tmp_path_factory.mktemp("tiny-qwen2")
    model.save_pretrained(path, safe_serialization=True)
    return model, path


def greedy(model, prompt, n):
    """Plain argmax decoding, recomputing the whole sequence each step."""
    tokens = list(prompt)
    with torch.no_grad():
        for _ in range(n):
            logits = model(input_ids=torch.tensor([tokens])).logits[0, -1]
            tokens.append(int(logits.argmax()))
    return tokens[len(prompt) :]


def prompts():
    shared = [(t * 31 + 5) % VOCAB for t in range(10)]
    tails = [[(t * 17 + i * 7 + 3) % VOCAB for t in range(1 + 2 * i)] for i in range(5)]
    return [shared + tail for tail in tails] + [[(t * 13 + 1) % VOCAB for t in range(7)]]


def test_generate_matches_transformers(tiny_qwen2):
    model, path = tiny_qwen2
    # Small enough that the batch runs out of blocks and preempts.
    engine = paged_kv.Engine(path, num_blocks=18, block_size=4, max_batch_size=4)
    batch = prompts()
    outputs = engine.generate(batch, max_new_tokens=NEW_TOKENS)

    assert outputs == [greedy(model, p, NEW_TOKENS) for p in batch]
    stats = engine.stats
    assert stats["prefill_tokens_cached"] > 0, stats
    assert stats["preemptions"] > 0, stats
    assert engine.is_idle


def test_stop_tokens_end_generation(tiny_qwen2):
    model, path = tiny_qwen2
    prompt = prompts()[0]
    full = greedy(model, prompt, NEW_TOKENS)
    stop = full[3]
    engine = paged_kv.Engine(path, num_blocks=32, block_size=4, stop_tokens=[stop])

    assert engine.generate([prompt], max_new_tokens=NEW_TOKENS) == [full[: full.index(stop) + 1]]


def test_step_by_step_matches_generate(tiny_qwen2):
    _, path = tiny_qwen2
    batch = prompts()[:3]
    expected = paged_kv.Engine(path, num_blocks=64, block_size=4).generate(batch, NEW_TOKENS)

    engine = paged_kv.Engine(path, num_blocks=64, block_size=4)
    ids = [engine.add_request(p, NEW_TOKENS) for p in batch]
    finished = {}
    while not engine.is_idle:
        finished.update(engine.step())
    assert [finished[i] for i in ids] == expected
