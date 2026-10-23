"""Shared fixtures: a tiny synthetic corpus, tokenizer and shard set, built
once per test session (takes ~1 s)."""

from pathlib import Path

import pytest

from sleuth.forge.data.synthetic import generate_corpus


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory) -> dict[str, Path]:
    from sleuth.forge.data.shard import build_shards
    from sleuth.forge.tokenizer.train_bpe import train_tokenizer

    root = tmp_path_factory.mktemp("tiny")
    generate_corpus(root / "raw", megabytes=1.0, seed=0)
    train_tokenizer([root / "raw"], vocab_size=512, out_path=root / "tok.json", log=lambda s: None)
    build_shards([root / "raw"], root / "tok.json", root / "shards", val_fraction=0.05,
                 shard_tokens=50_000, log=lambda s: None)
    return {"root": root, "raw": root / "raw", "tokenizer": root / "tok.json", "shards": root / "shards"}
