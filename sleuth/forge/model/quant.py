"""Layers whose weights stay quantized in memory (GGUF blocks).

Gemma 4 12B as a Q4_0 GGUF is 6.2 GiB. Dequantized to bf16 it would be
~24 GB: more than the 16 GB of an RTX 5060 Ti. So instead of converting every
weight when loading, these layers keep the file's raw blocks (4.5 bits per
weight for Q4_0) on the GPU and dequantize just before each matmul:

    QuantLinear:     y = x @ dequantize(blocks).T          (one layer at a time)
    QuantEmbedding:  dequantize only the rows of the tokens being looked up

The price is speed: every forward pass dequantizes every weight again.
(Real inference engines fuse the dequantization into the matmul kernel;
that's a possible later optimization. Here, correctness and memory first.)

Big weights are dequantized in chunks of output rows, so even the 262k-token
output layer never needs its full ~2 GB bf16 copy at once.

The quantized bytes are buffers, not Parameters: they can't be trained, and
`num_params()` counts their *logical* weights (see `n_weights`).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ...formats import gguf_quants as q

CHUNK_ELEMENTS = 1 << 25  # dequantize at most 32M weights (64 MB in bf16) at a time


def row_bytes(ggml_type: int, row_len: int) -> int:
    """Bytes per row of `row_len` values (rows must be whole blocks)."""
    per, nb = q.BLOCK[ggml_type]
    if row_len % per:
        raise ValueError(f"rows of {row_len} values aren't whole {q.TYPE_NAMES[ggml_type]} blocks of {per}")
    return row_len // per * nb


def dequantize_rows(raw: torch.Tensor, ggml_type: int, row_len: int, start: int, stop: int,
                    dtype: torch.dtype) -> torch.Tensor:
    """Rows start..stop-1 of a row-major quantized matrix -> [stop-start, row_len]."""
    rb = row_bytes(ggml_type, row_len)
    w = q.dequantize(raw[start * rb: stop * rb], ggml_type, (stop - start) * row_len, dtype)
    return w.reshape(stop - start, row_len)


class QuantLinear(nn.Module):
    """A bias-free Linear layer stored as GGUF blocks: weight [out_features, in_features]."""

    def __init__(self, in_features: int, out_features: int, ggml_type: int, device=None):
        super().__init__()
        self.in_features, self.out_features, self.ggml_type = in_features, out_features, ggml_type
        self.bias = None
        nbytes = out_features * row_bytes(ggml_type, in_features)
        self.register_buffer("raw", torch.empty(nbytes, dtype=torch.uint8, device=device))

    @property
    def n_weights(self) -> int:
        return self.in_features * self.out_features

    def dequantize_rows(self, start: int, stop: int, dtype: torch.dtype) -> torch.Tensor:
        return dequantize_rows(self.raw, self.ggml_type, self.in_features, start, stop, dtype)

    @property
    def weight(self) -> torch.Tensor:
        """The whole weight, dequantized (memory-hungry for big layers: for tests and inspection)."""
        return self.dequantize_rows(0, self.out_features, torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = max(1, CHUNK_ELEMENTS // self.in_features)
        if rows >= self.out_features:
            return F.linear(x, self.dequantize_rows(0, self.out_features, x.dtype))
        parts = [F.linear(x, self.dequantize_rows(i, min(i + rows, self.out_features), x.dtype))
                 for i in range(0, self.out_features, rows)]
        return torch.cat(parts, dim=-1)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, {q.TYPE_NAMES[self.ggml_type]}"


class QuantEmbedding(nn.Module):
    """An embedding table stored as GGUF blocks: [num_embeddings, embedding_dim]."""

    def __init__(self, num_embeddings: int, embedding_dim: int, ggml_type: int, device=None):
        super().__init__()
        self.num_embeddings, self.embedding_dim, self.ggml_type = num_embeddings, embedding_dim, ggml_type
        nbytes = num_embeddings * row_bytes(ggml_type, embedding_dim)
        self.register_buffer("raw", torch.empty(nbytes, dtype=torch.uint8, device=device))
        self.out_dtype = torch.float32  # set by Transformer.empty to the model's compute dtype

    @property
    def n_weights(self) -> int:
        return self.num_embeddings * self.embedding_dim

    @property
    def weight(self) -> torch.Tensor:
        """The whole table, dequantized (for tests and inspection)."""
        return dequantize_rows(self.raw, self.ggml_type, self.embedding_dim, 0, self.num_embeddings, torch.float32)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        rb = row_bytes(self.ggml_type, self.embedding_dim)
        rows = self.raw.view(self.num_embeddings, rb)[idx.reshape(-1)]   # gather just these rows' blocks
        w = q.dequantize(rows.reshape(-1), self.ggml_type, rows.shape[0] * self.embedding_dim, self.out_dtype)
        return w.reshape(*idx.shape, self.embedding_dim)

    def as_output_layer(self) -> QuantLinear:
        """A QuantLinear that shares this table's bytes: the tied output layer
        (logits = h @ Eᵀ), with no second copy in memory."""
        lin = QuantLinear(self.embedding_dim, self.num_embeddings, self.ggml_type, device="meta")
        lin.raw = self.raw
        return lin

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}, {q.TYPE_NAMES[self.ggml_type]}"


def quantized_modules(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, (QuantLinear, QuantEmbedding))]
