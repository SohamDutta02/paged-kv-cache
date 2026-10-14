"""Writes a word-level tokenizer.json with one token per id, `t0` to `t{n-1}`.

Enough to serve a model with dummy weights: prompts are sent as token ids, and
every generated token decodes to its own word, so each one streams as a chunk.
It has no special tokens, so nothing stops generation before max_tokens.

    python make_tokenizer.py <model-dir>

The vocabulary size comes from the directory's config.json.
"""

import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
size = json.loads((directory / "config.json").read_text())["vocab_size"]
spec = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [],
    "normalizer": None,
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": None,
    "decoder": None,
    "model": {"type": "WordLevel", "vocab": {f"t{i}": i for i in range(size)}, "unk_token": "t0"},
}
(directory / "tokenizer.json").write_text(json.dumps(spec))
