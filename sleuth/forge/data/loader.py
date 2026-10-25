"""Read training batches out of memory-mapped shards.

Sampling is *stateless*: the batch for (seed, step, micro_step, rank) is
always the same, no matter what happened before. That makes resuming a run
trivial (just know the step number) and keeps multi-GPU ranks from seeing
the same data, with no data-loader state to checkpoint.

Windows are taken at random offsets and can span document boundaries. The
<|bos|>/<|eos|> tokens tell the model where one document ends. With
`train.doc_mask = true`, attention also stops at those boundaries:
`document_ids` numbers the documents inside each window and the model masks
attention between different numbers (see Transformer.forward).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


def document_ids(x: torch.Tensor, bos_id: int) -> torch.Tensor:
    """[B, T] token IDs -> [B, T] document numbers: 0 until the first <|bos|>
    in the window, then +1 at every <|bos|>. Every document in a shard starts
    with <|bos|> (shard.py), so this finds the same boundaries as the .idx.npy
    files, with no lookup: a window cut mid-document gets number 0 for that
    document's tail.

    Why <|bos|> and not <|eos|>: the <|eos|> is the last token of its own
    document (it's predicted from that document's text), and the <|bos|> the
    first of the next, which then sees only itself."""
    return (x == bos_id).cumsum(dim=1)


class ShardDataset:
    def __init__(self, data_dir: str | Path, split: str = "train"):
        self.dir = Path(data_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.split = split
        self.dtype = np.dtype(self.meta["dtype"])
        shards = [s for s in self.meta["splits"][split] if s["n_tokens"] > 0]
        if not shards:
            raise ValueError(f"no {split} tokens in {self.dir}")
        # mode="r" maps read-only; pages are loaded lazily by the OS.
        self.arrays = [np.memmap(self.dir / s["file"], dtype=self.dtype, mode="r") for s in shards]
        self.n_tokens = sum(len(a) for a in self.arrays)

    @property
    def vocab_size(self) -> int:
        return self.meta["vocab_size"]

    @property
    def tokenizer_path(self) -> Path:
        """The shards' tokenizer: "tokenizer.json" in the shard folder, or, for
        shards made before 2026-10-25, the absolute path meta.json recorded."""
        return self.dir / self.meta["tokenizer"]   # an absolute path stays absolute under "/"

    def batch(self, batch_size: int, seq_len: int, seed: int, step: int,
              micro_step: int = 0, rank: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Random windows of seq_len+1 tokens; x is the first seq_len, y is
        shifted by one (the "next token" each position must predict)."""
        rng = np.random.default_rng([seed, step, micro_step, rank])
        usable = np.array([max(0, len(a) - seq_len - 1) for a in self.arrays], dtype=np.float64)
        if usable.sum() == 0:
            raise ValueError(f"{self.split} split has fewer than seq_len+1={seq_len + 1} tokens per shard")
        shard_ids = rng.choice(len(self.arrays), size=batch_size, p=usable / usable.sum())
        rows = []
        for sid in shard_ids:
            off = int(rng.integers(0, usable[sid] + 1))
            rows.append(np.asarray(self.arrays[sid][off: off + seq_len + 1], dtype=np.int64))
        buf = torch.from_numpy(np.stack(rows))
        return buf[:, :-1], buf[:, 1:]

    def eval_windows(self, n_windows: int, seq_len: int) -> list[tuple[int, int, str | None]]:
        """n_windows fixed windows, spread evenly over the *whole* split:
        [(shard index, offset, source name)]. Always the same ones, so every
        evaluation is comparable.

        (The first version took windows from the start of the split. The
        documents are stored source by source, so with a mix that measured only
        the first source: FineWeb-Edu, 7% of the val set. Found by the owner's
        agent on the mix-20m run.)
        """
        # A window can start at offsets 0..len-seq_len-1 of a shard; shards too
        # short for one window are skipped.
        slots = np.array([max(0, len(a) - seq_len) for a in self.arrays], dtype=np.int64)
        if slots.sum() == 0:
            raise ValueError(f"{self.split} split has fewer than seq_len+1={seq_len + 1} tokens per shard")
        ends = np.cumsum(slots)                          # slot g lives in the shard where g < ends[sid]
        bases = np.cumsum([0] + [len(a) for a in self.arrays])
        ranges = self.meta.get("split_sources", {}).get(self.split, [])
        starts = [r["start"] for r in ranges]
        out = []
        for g in np.linspace(0, ends[-1] - 1, n_windows).astype(np.int64):
            sid = int(np.searchsorted(ends, g, side="right"))
            off = int(g - (ends[sid - 1] if sid else 0))
            src = None
            if ranges:                                   # which source this window starts in
                src = ranges[int(np.searchsorted(starts, bases[sid] + off, side="right")) - 1]["name"]
            out.append((sid, off, src))
        return out

    def window_batches(self, windows: list[tuple[int, int, str | None]], batch_size: int, seq_len: int):
        """Yield (x, y, source names) batches for the given windows."""
        for i in range(0, len(windows), batch_size):
            chunk = windows[i:i + batch_size]
            buf = torch.from_numpy(np.stack([np.asarray(self.arrays[sid][off: off + seq_len + 1], dtype=np.int64)
                                             for sid, off, _ in chunk]))
            yield buf[:, :-1], buf[:, 1:], [src for _, _, src in chunk]

    def sequential_batches(self, batch_size: int, seq_len: int, max_batches: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Non-overlapping windows from the start of the split, always the
        same ones. Used for validation so every eval is comparable."""
        made = 0
        rows = []
        for a in self.arrays:
            for off in range(0, len(a) - seq_len - 1 + 1, seq_len):
                rows.append(np.asarray(a[off: off + seq_len + 1], dtype=np.int64))
                if len(rows) == batch_size:
                    buf = torch.from_numpy(np.stack(rows))
                    yield buf[:, :-1], buf[:, 1:]
                    rows = []
                    made += 1
                    if made >= max_batches:
                        return
        if rows and made < max_batches:
            buf = torch.from_numpy(np.stack(rows))
            yield buf[:, :-1], buf[:, 1:]
