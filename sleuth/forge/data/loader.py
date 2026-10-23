"""Read training batches out of memory-mapped shards.

Sampling is *stateless*: the batch for (seed, step, micro_step, rank) is
always the same, no matter what happened before. That makes resuming a run
trivial (just know the step number) and keeps multi-GPU ranks from seeing
the same data, with no data-loader state to checkpoint.

Windows are taken at random offsets and can span document boundaries. The
<|bos|>/<|eos|> tokens tell the model where one document ends. (Masking
attention at document boundaries is a later LAB experiment; the .idx.npy
files already record where documents start.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


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
