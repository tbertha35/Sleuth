"""Train a byte-level BPE tokenizer on local files.

Two phases:

1. **Count chunks** (the slow part on big corpora, so it's parallel): split
   every document with the pre-tokenizer and count how often each distinct
   chunk occurs. Real text has a few million distinct chunks even when the
   corpus is many GB, so the rest of training works on this small table
   instead of the raw text.

2. **Merge loop**: repeatedly find the most frequent adjacent token pair
   (weighted by chunk counts) and merge it into a new token. The naive way
   recounts all pairs after every merge, which is O(merges × table). We keep:
     - `pair_counts`: pair -> total count
     - `where`: pair -> set of chunk indices containing it
     - a max-heap of (count, pair), with *lazy deletion*: when a count
       changes we push the new value and simply skip stale entries when
       they surface.
   Each merge then only touches the chunks that contain the merged pair.

Ties (equal counts) break toward the smaller pair of IDs, which makes
training fully deterministic.
"""

from __future__ import annotations

import heapq
import time
from collections import Counter, defaultdict
import multiprocessing as mp
from pathlib import Path
from typing import Callable, Iterable

from ..data.ingest import find_files, iter_input_documents
from .bpe import SPECIAL_TOKENS, Tokenizer
from .pretokenize import PATTERN, compile_pattern


def _pool(workers: int, **kwargs):
    """A process pool using "spawn": each worker starts a fresh interpreter.
    The default on Linux ("fork") copies the parent process, which can
    deadlock if the parent has threads (e.g. the dashboard server)."""
    return mp.get_context("spawn").Pool(workers, **kwargs)


# --------------------------------------------------------------------------- #
# Phase 1: chunk counting
# --------------------------------------------------------------------------- #

def _count_batch(texts: list[str]) -> Counter:
    regex = compile_pattern(PATTERN)
    c: Counter = Counter()
    for t in texts:
        c.update(regex.findall(t))
    return c


def count_chunks(texts: Iterable[str], workers: int = 1, batch_chars: int = 4_000_000,
                 max_bytes: int | None = None, log: Callable[[str], None] = print) -> tuple[Counter, int]:
    """Count pre-tokenized chunks over an iterable of documents.

    Returns (counts, bytes_seen). `max_bytes` caps how much text is read:
    a tokenizer trained on a few GB is nearly identical to one trained on
    everything, and much faster to make.
    """
    total: Counter = Counter()
    seen = 0

    def batches():
        nonlocal seen
        batch, size = [], 0
        for t in texts:
            n = len(t.encode("utf-8"))
            if max_bytes is not None and seen + n > max_bytes:
                break
            batch.append(t)
            size += len(t)
            seen += n
            if size >= batch_chars:
                yield batch
                batch, size = [], 0
        if batch:
            yield batch

    t0 = time.time()
    if workers <= 1:
        for b in batches():
            total.update(_count_batch(b))
    else:
        with _pool(workers) as pool:
            for i, c in enumerate(pool.imap_unordered(_count_batch, batches())):
                total.update(c)
                if i % 20 == 0:
                    log(f"  counted {seen / 1e6:.0f} MB, {len(total):,} distinct chunks ({time.time() - t0:.0f}s)")
    return total, seen


# --------------------------------------------------------------------------- #
# Phase 2: merges
# --------------------------------------------------------------------------- #

def learn_merges(chunk_counts: dict[str, int], n_merges: int,
                 log: Callable[[str], None] = print, log_every: int = 1000) -> list[tuple[int, int]]:
    words: list[list[int]] = []
    freqs: list[int] = []
    for chunk, f in chunk_counts.items():
        b = list(chunk.encode("utf-8"))
        if len(b) >= 2:  # single bytes can never be merged; skip them
            words.append(b)
            freqs.append(f)

    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    where: dict[tuple[int, int], set[int]] = defaultdict(set)
    for wi, w in enumerate(words):
        f = freqs[wi]
        for p in zip(w, w[1:]):
            pair_counts[p] += f
            where[p].add(wi)

    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    merges: list[tuple[int, int]] = []
    t0 = time.time()
    for m in range(n_merges):
        # Pop until we find an entry whose count is still current.
        best = None
        while heap:
            negc, p = heapq.heappop(heap)
            if negc < 0 and pair_counts.get(p, 0) == -negc:
                best = p
                break
        if best is None:
            log(f"  stopped early after {m} merges: no pairs left (corpus too small for this vocab size)")
            break
        new_id = 256 + m
        merges.append(best)
        a, b = best

        touched: set[tuple[int, int]] = set()
        for wi in where.pop(best):
            w = words[wi]
            f = freqs[wi]
            # Merge best inside this word.
            nw = []
            i = 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    nw.append(new_id)
                    i += 2
                else:
                    nw.append(w[i])
                    i += 1
            # Update pair counts by the difference between old and new pairs.
            for p in zip(w, w[1:]):
                pair_counts[p] -= f
                touched.add(p)
            for p in zip(nw, nw[1:]):
                pair_counts[p] += f
                where[p].add(wi)
                touched.add(p)
            words[wi] = nw
        pair_counts.pop(best, None)
        touched.discard(best)
        for p in touched:
            c = pair_counts.get(p, 0)
            if c > 0:
                heapq.heappush(heap, (-c, p))
            elif p in pair_counts:
                del pair_counts[p]

        if log_every and (m + 1) % log_every == 0:
            log(f"  merge {m + 1}/{n_merges} ({time.time() - t0:.0f}s)")
    return merges


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def train_tokenizer(inputs: list[str | Path], vocab_size: int, out_path: str | Path,
                    workers: int = 1, max_bytes: int | None = None, text_field: str = "text",
                    log: Callable[[str], None] = print) -> Tokenizer:
    n_merges = vocab_size - 256 - len(SPECIAL_TOKENS)
    if n_merges < 0:
        raise ValueError(f"vocab_size must be at least {256 + len(SPECIAL_TOKENS)}")
    files = find_files(inputs)
    if not files:
        raise ValueError(f"no .txt/.jsonl/.parquet files found in {inputs}")
    log(f"counting chunks in {len(files)} files with {workers} workers...")
    t0 = time.time()
    # Per-input caps ("path@MB") give every source of a mix its share of the
    # tokenizer's training text.
    docs = (d.text for d in iter_input_documents(inputs, text_field=text_field))
    counts, seen = count_chunks(docs, workers=workers, max_bytes=max_bytes, log=log)
    log(f"  {seen / 1e6:.1f} MB of text, {len(counts):,} distinct chunks ({time.time() - t0:.1f}s)")

    log(f"learning {n_merges} merges...")
    t1 = time.time()
    merges = learn_merges(counts, n_merges, log=log)
    first_special = 256 + len(merges)
    specials = {s: first_special + i for i, s in enumerate(SPECIAL_TOKENS)}
    tok = Tokenizer(merges, specials, PATTERN, train_info={
        "files": [str(f) for f in files],
        "bytes_used": seen,
        "distinct_chunks": len(counts),
        "requested_vocab_size": vocab_size,
        "count_seconds": round(t1 - t0, 1),
        "merge_seconds": round(time.time() - t1, 1),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    bpt = bytes_per_token(tok, counts)
    tok.train_info["bytes_per_token"] = round(bpt, 3)
    tok.save(out_path)
    log(f"saved {out_path} (vocab {tok.vocab_size}, ~{bpt:.2f} bytes/token on its training text, "
        f"{time.time() - t0:.1f}s total)")
    return tok


def bytes_per_token(tok: Tokenizer, counts: Counter, top: int = 200_000) -> float:
    """Average bytes per token on the training text, from the chunk counts: each
    distinct chunk is encoded once and weighted by how often it occurred. Only
    the `top` most frequent chunks are used (they cover nearly all the text), so
    it's quick even for millions of distinct chunks."""
    n_bytes = n_tokens = 0
    for chunk, c in counts.most_common(top):
        n_bytes += c * len(chunk.encode("utf-8"))
        n_tokens += c * len(tok._encode_chunk(chunk))
    return n_bytes / max(1, n_tokens)
