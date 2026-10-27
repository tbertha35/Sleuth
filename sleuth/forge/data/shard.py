"""Tokenize documents into memory-mapped binary shards.

Layout of an output directory:

    meta.json                 tokenizer fingerprint, dtype, per-split shard list
    train_000.bin, ...        raw token IDs, back to back (uint16 or uint32)
    train_000.idx.npy         int64 offsets where each document starts
    val_000.bin, ...

Every document is stored as [<|bos|>] + tokens + [<|eos|>]. The loader
reads random windows straight out of the .bin files via np.memmap. The OS
pages data in from the SSD on demand, so a 100 GB dataset needs no RAM.

Why uint16: with a vocab < 65,536, two bytes per token halves disk use and
I/O compared with int32. Bigger vocabs automatically switch to uint32.
"""

from __future__ import annotations

import json
import shutil
import time
from collections import deque
import multiprocessing as mp
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from ..tokenizer.bpe import Tokenizer
from .ingest import Document, find_files, is_validation, iter_input_documents


def _pool(workers: int, **kwargs):
    """A process pool using "spawn": each worker starts a fresh interpreter.
    The default on Linux ("fork") copies the parent process, which can
    deadlock if the parent has threads (e.g. the dashboard server)."""
    return mp.get_context("spawn").Pool(workers, **kwargs)

META_VERSION = 1

_worker_tok: Tokenizer | None = None


def _init_worker(tok_path: str) -> None:
    global _worker_tok
    _worker_tok = Tokenizer.load(tok_path)


def _encode_batch(texts: list[str]) -> list[np.ndarray]:
    tok = _worker_tok
    assert tok is not None
    out = []
    for t in texts:
        out.append(np.asarray(document_tokens(tok, t), dtype=np.uint32))
    return out


def doc_start_id(tok) -> int:
    """The token that starts every document in our shards (what document masking
    splits on): BOS, or for tokenizers without one (Qwen) the separator the model
    was pretrained with, <|endoftext|> (tok.start_id)."""
    return tok.bos_id if tok.bos_id is not None else tok.start_id


def document_tokens(tok, text: str) -> list[int]:
    """[start] + text + [end]. Without a BOS the start token is <|endoftext|>, which is
    also Qwen's end of text, so no separate end token: documents are separated by one."""
    start = doc_start_id(tok)
    end = [tok.eos_id] if tok.eos_id is not None and tok.eos_id != start else []
    return [start] + tok.encode_ordinary(text) + end


class _ShardWriter:
    """Appends documents to <split>_NNN.bin, starting a new file once the
    current one passes `shard_tokens` (only between documents, so no
    document is ever split across two files)."""

    def __init__(self, out_dir: Path, split: str, dtype: np.dtype, shard_tokens: int):
        self.out_dir, self.split, self.dtype, self.shard_tokens = out_dir, split, dtype, shard_tokens
        self.shards: list[dict] = []
        self.fh = None
        self.n = 0
        self.starts: list[int] = []
        self.docs = 0
        self.total = 0                     # tokens written to this split so far (all shards)
        self.sources: list[dict] = []      # where each source's documents begin, in split-wide token offsets

    def _open(self):
        name = f"{self.split}_{len(self.shards):03d}.bin"
        self.fh = open(self.out_dir / name, "wb")
        self.shards.append({"file": name, "n_tokens": 0, "n_docs": 0})
        self.n = 0
        self.starts = []

    def _close(self):
        if self.fh is None:
            return
        self.fh.close()
        self.fh = None
        np.save(self.out_dir / self.shards[-1]["file"].replace(".bin", ".idx.npy"), np.asarray(self.starts, np.int64))
        self.shards[-1]["n_tokens"] = self.n
        self.shards[-1]["n_docs"] = len(self.starts)

    def write(self, ids: np.ndarray, source: str | None = None):
        if self.fh is None:
            self._open()
        if source is not None and (not self.sources or self.sources[-1]["name"] != source):
            # Documents arrive input by input, so each source is one contiguous
            # range. Recording the ranges lets evaluation report loss per source.
            self.sources.append({"name": source, "start": self.total})
        self.starts.append(self.n)
        self.fh.write(ids.astype(self.dtype).tobytes())
        self.n += len(ids)
        self.total += len(ids)
        self.docs += 1
        if self.n >= self.shard_tokens:
            self._close()

    def finish(self) -> list[dict]:
        self._close()
        return self.shards


def build_shards(inputs: list[str | Path], tokenizer_path: str | Path, out_dir: str | Path,
                 val_fraction: float = 0.005, shard_tokens: int = 256 * 2**20, workers: int = 1,
                 text_field: str = "text", max_docs: int | None = None,
                 log: Callable[[str], None] = print) -> dict:
    tok = Tokenizer.load(tokenizer_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("*.bin")):
        raise FileExistsError(f"{out} already contains shards; use an empty directory")
    dtype = np.dtype(np.uint16 if tok.vocab_size <= 65536 else np.uint32)
    writers = {s: _ShardWriter(out, s, dtype, shard_tokens) for s in ("train", "val")}
    files = find_files(inputs)
    shutil.copy(tokenizer_path, out / "tokenizer.json")
    cfg = Path(tokenizer_path).parent / "tokenizer_config.json"   # a Hugging Face tokenizer's bos/eos names
    if cfg.exists():
        shutil.copy(cfg, out / "tokenizer_config.json")
    log(f"sharding {len(files)} files -> {out} ({dtype.name}, val_fraction={val_fraction})")

    def batches(docs: Iterable[Document], size: int = 512):
        """Yield (texts, (is_val, source) flags) batches, deciding the split up front."""
        texts, flags = [], []
        for i, d in enumerate(docs):
            if max_docs is not None and i >= max_docs:
                break
            texts.append(d.text)
            flags.append((is_validation(d.text, val_fraction), d.category))
            if len(texts) >= size:
                yield texts, flags
                texts, flags = [], []
        if texts:
            yield texts, flags

    t0 = time.time()
    total_bytes = 0
    cat_counts: dict[str, int] = {}
    cat_bytes: dict[str, int] = {}
    docs = iter_input_documents(inputs, text_field=text_field)

    def counted(docs):
        nonlocal total_bytes
        for d in docs:
            n = len(d.text.encode("utf-8"))
            total_bytes += n
            if d.category:
                cat_counts[d.category] = cat_counts.get(d.category, 0) + 1
                cat_bytes[d.category] = cat_bytes.get(d.category, 0) + n
            yield d

    batch_iter = batches(counted(docs))

    def encoded_batches():
        """Yield (encoded docs, is_val flags) in input order."""
        if workers <= 1:
            _init_worker(str(tokenizer_path))
            for texts, flags in batch_iter:
                yield _encode_batch(texts), flags
            return
        # The pool pulls texts from a background thread; we remember each
        # batch's flags in a FIFO and pair them back up. pool.imap (not
        # imap_unordered) returns results in input order, so output is
        # deterministic regardless of worker count.
        pending_flags: deque[list[bool]] = deque()

        def texts_only():
            for texts, flags in batch_iter:
                pending_flags.append(flags)
                yield texts

        with _pool(workers, initializer=_init_worker, initargs=(str(tokenizer_path),)) as pool:
            for encoded in pool.imap(_encode_batch, texts_only(), chunksize=1):
                yield encoded, pending_flags.popleft()

    for bi, (encoded, flags) in enumerate(encoded_batches()):
        for ids, (is_val, source) in zip(encoded, flags):
            writers["val" if is_val else "train"].write(ids, source)
        if bi % 200 == 0 and bi:
            log(f"  {total_bytes / 1e6:.0f} MB, {writers['train'].docs + writers['val'].docs:,} docs "
                f"({time.time() - t0:.0f}s)")

    meta = {
        "version": META_VERSION,
        # A copy of the tokenizer lives next to the shards, and meta.json names it
        # relative to them: the folder is self-contained, can move, and doesn't
        # record a local path (with a user name in it).
        "tokenizer": "tokenizer.json",
        "tokenizer_source": Path(tokenizer_path).name,
        "tokenizer_fingerprint": tok.fingerprint(),
        "vocab_size": tok.vocab_size,
        "dtype": dtype.name,
        "bos_id": tok.bos_id,
        "eos_id": tok.eos_id,
        "source_files": [str(f) for f in files],
        "source_bytes": total_bytes,
        "categories": cat_counts,
        "category_bytes": cat_bytes,       # how much text each source contributed: the actual mix
        "val_fraction": val_fraction,
        "splits": {s: w.finish() for s, w in writers.items()},
        "split_sources": {s: w.sources for s, w in writers.items()},
        "seconds": round(time.time() - t0, 1),
    }
    for s in meta["splits"]:
        meta[f"{s}_tokens"] = sum(x["n_tokens"] for x in meta["splits"][s])
    meta["bytes_per_token"] = round(total_bytes / max(1, meta["train_tokens"] + meta["val_tokens"]), 3)
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log(f"done: {meta['train_tokens']:,} train / {meta['val_tokens']:,} val tokens, "
        f"{meta['bytes_per_token']} bytes/token, {meta['seconds']}s")
    if len(cat_bytes) > 1:
        log("mix by text size: " + ", ".join(f"{c} {b / max(1, total_bytes):.1%}" for c, b in
                                              sorted(cat_bytes.items(), key=lambda kv: -kv[1])))
    return meta
