"""N-gram memory: knowledge stored in lookup tables instead of computed by layers.

Idea (DeepSeek's "Engram", 2026; older roots in hashed n-gram embeddings): many
next-token facts are tied to the last few tokens ("the capital of" -> ...). A
transformer has to compute such associations with its layers, every time. A
table indexed by the recent n-gram can just *store* them. A lookup costs no
matmul FLOPs, so the table can be far bigger than the model: parameters that
add capacity without adding compute, and (later) that can live in RAM or on an
SSD because each token reads only a few rows.

How it works here, per position t:
  1. for each n-gram order n (e.g. 2 and 3) and each of H hash functions, hash the
     tokens x[t-n+1..t] to a row of that (order, head)'s table. Several heads make
     collisions rare: two n-grams rarely collide under every hash.
  2. concatenate the rows: m_t = [rows ...]  (dim = orders x heads x mem_dim)
  3. at each injection layer, a gate decides how much to trust the memory given
     the current hidden state h_t (Engram's "context-aware gating"):
         k = W_k m_t,  v = W_v m_t
         g = sigmoid( RMSNorm(h_t) . RMSNorm(k) / sqrt(d) )
         h_t <- h_t + g * v
     A collision or a useless n-gram then gets gated off instead of adding noise.

Positions before the start of the text (or before the start of the document,
with document masking) are a reserved "no token" value, so the n-gram never
reaches into a previous document.

Decoding: the hash needs the previous n-1 tokens, which a cached step doesn't
see. The KV cache keeps them (KVCache.token_history), on the GPU so CUDA graphs
work too.
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F
from torch import nn

NO_TOKEN = -1
_MULT = 0x9E3779B97F4A7C15 - (1 << 64)   # 64-bit golden-ratio constant, as a signed int64


class NgramMemory(nn.Module):
    def __init__(self, d_model: int, rows: int, dim: int, orders: list[int], heads: int, layers: list[int]):
        super().__init__()
        self.rows, self.dim, self.orders, self.heads, self.layers = rows, dim, list(orders), heads, list(layers)
        n_tables = len(self.orders) * heads
        self.table = nn.Embedding(n_tables * rows, dim)
        self.table.weight.no_weight_decay = True   # optim.py: decay would erase rarely-used rows
        width = n_tables * dim
        self.key = nn.ModuleList(nn.Linear(width, d_model, bias=False) for _ in self.layers)
        self.value = nn.ModuleList(nn.Linear(width, d_model, bias=False) for _ in self.layers)
        self.d_model = d_model
        # per (order, head): an odd multiplier and a seed. Fixed numbers, not buffers: they
        # are part of the table's "format" (change them and every row means something else).
        rng = random.Random(1234)
        self.hash_mul = [rng.randrange(1 << 61) * 2 + 1 for _ in range(n_tables)]
        self.hash_seed = [rng.randrange(1 << 62) for _ in range(n_tables)]

    def context(self, idx: torch.Tensor, cache=None, doc_ids: torch.Tensor | None = None) -> torch.Tensor:
        """[B, T, max_order] tokens: column j is the token j places back (NO_TOKEN
        before the start of the text or of the document)."""
        B, T = idx.shape
        n = max(self.orders)
        if cache is not None:
            hist = cache.token_history(B, idx.device)          # [B, max_len], NO_TOKEN where unwritten
            if getattr(cache, "static", False):
                pos = cache.positions                           # [T] on the GPU
                hist.index_copy_(1, pos, idx)
            else:
                pos = torch.arange(cache.pos, cache.pos + T, device=idx.device)
                hist[:, cache.pos:cache.pos + T] = idx
            back = pos[:, None] - torch.arange(n, device=idx.device)[None, :]    # [T, n]
            ctx = hist[:, back.clamp_min(0)]                                       # [B, T, n]
            return torch.where((back >= 0)[None], ctx, torch.full_like(ctx, NO_TOKEN))
        cols = [idx]
        for j in range(1, n):
            shifted = F.pad(idx[:, :-j], (j, 0), value=NO_TOKEN) if j < T else torch.full_like(idx, NO_TOKEN)
            if doc_ids is not None:
                same = F.pad(doc_ids[:, :-j], (j, 0), value=-1) == doc_ids if j < T else torch.zeros_like(idx, dtype=torch.bool)
                shifted = torch.where(same, shifted, torch.full_like(shifted, NO_TOKEN))
            cols.append(shifted)
        return torch.stack(cols, -1)

    def rows_for(self, ctx: torch.Tensor) -> torch.Tensor:
        """[B, T, n_tables] row indices into self.table."""
        out = []
        t = 0
        for order in self.orders:
            acc = torch.zeros(ctx.shape[:2], dtype=torch.long, device=ctx.device)
            for j in range(order):                 # polynomial hash of the n-gram (int64 wraps)
                acc = acc * 1000003 + (ctx[..., j] + 2)
            for _ in range(self.heads):
                h = (acc ^ self.hash_seed[t]) * self.hash_mul[t]
                h = h ^ (h >> 29)
                out.append(torch.remainder(h * _MULT, self.rows) + t * self.rows)
                t += 1
        return torch.stack(out, -1)

    def lookup(self, idx: torch.Tensor, cache=None, doc_ids: torch.Tensor | None = None) -> torch.Tensor:
        """[B, T, n_tables * dim]: the memory vector of every position."""
        rows = self.rows_for(self.context(idx, cache, doc_ids))
        return self.table(rows).flatten(-2)

    def inject(self, h: torch.Tensor, mem: torch.Tensor, slot: int) -> torch.Tensor:
        """h + gate * value, for the slot-th injection layer."""
        m = mem.to(h.dtype)
        k = self.key[slot](m)
        g = torch.sigmoid((F.rms_norm(h.float(), (self.d_model,)) * F.rms_norm(k.float(), (self.d_model,))).sum(-1, keepdim=True)
                          / math.sqrt(self.d_model))
        return h + g.to(h.dtype) * self.value[slot](m)

    def extra_repr(self) -> str:
        return (f"orders={self.orders}, heads={self.heads}, rows={self.rows:,} per table, dim={self.dim}, "
                f"layers={self.layers}")
