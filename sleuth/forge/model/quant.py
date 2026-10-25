"""Layers whose weights stay quantized in memory (GGUF blocks).

Gemma 4 12B as a Q4_0 GGUF is 6.2 GiB. Dequantized to bf16 it would be
~24 GB: more than the 16 GB of an RTX 5060 Ti. So instead of converting every
weight when loading, these layers keep the file's raw blocks (4.5 bits per
weight for Q4_0) on the GPU and dequantize just before each matmul:

    QuantLinear:     y = x @ dequantize(blocks).T          (one layer at a time)
    QuantEmbedding:  dequantize only the rows of the tokens being looked up

Two ways to run a QuantLinear:
  * fused (GPU + Triton, Q4_0/Q8_0): a Triton kernel unpacks the blocks inside
    the matmul (triton_kernels.py). Each weight byte is read once; nothing is
    written back. This is what makes a 12B model usable on a 16 GB card.
  * plain (everything else): dequantize to bf16 in memory, then F.linear.
    Correct everywhere, but every forward pass rewrites every weight. Big
    weights go in chunks of output rows, so even the 262k-token output layer
    never needs its full ~2 GB bf16 copy at once.

The quantized bytes are buffers, not Parameters: they can't be trained, and
`num_params()` counts their *logical* weights (see `n_weights`).

llama.cpp's activation rounding (an experiment, off by default): llama.cpp does
not multiply quantized weights by the activations as they are. On CUDA it first
rounds the activations to 8 bits too (q8_1: blocks of 32 values sharing one fp16
scale), then takes integer dot products. On CPU it does the same with q8_0/q8_K.
So llama.cpp's logits carry a rounding error ours don't. `emulate_q8_activations`
applies the same rounding to x before each quantized matmul, to test whether
that rounding explains a difference between the two (round 5: Gemma 4's first
token after "The capital of France is" is "0" here, "1" in llama.cpp, with a
0.34-logit margin that survives fp32).
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


def round_like_q8(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """x after llama.cpp's activation quantization (CUDA's quantize_q8_1):
    per block of 32 values along the last dimension, d = max|x| / 127,
    q = round(x / d) (halves away from zero, as C's roundf), and the value
    used is q · d with d stored as fp16. Returns x's shape and dtype."""
    shape, dtype = x.shape, x.dtype
    xb = x.float().reshape(*shape[:-1], shape[-1] // block, block)
    d = xb.abs().amax(-1, keepdim=True) / 127
    qv = torch.where(d > 0, xb / d.clamp_min(1e-30), torch.zeros_like(xb))
    qv = torch.sign(qv) * torch.floor(qv.abs() + 0.5)        # roundf, not round-half-to-even
    return (qv * d.half().float()).reshape(shape).to(dtype)


class QuantLinear(nn.Module):
    """A bias-free Linear layer stored as GGUF blocks: weight [out_features, in_features]."""

    emulate_q8_activations = False  # see the module docstring; set_q8_activations() turns it on

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
        from . import triton_kernels  # imported on first use: loading Triton takes a moment

        if self.emulate_q8_activations:
            x = round_like_q8(x)
        if triton_kernels.usable(x, self.ggml_type):
            return triton_kernels.qmatmul(x, self.raw, self.ggml_type, self.out_features)
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


def set_q8_activations(model: nn.Module, on: bool = True) -> int:
    """Round the input of every quantized matmul like llama.cpp does (see the
    module docstring). Returns how many layers it applies to."""
    layers = [m for m in quantized_modules(model) if isinstance(m, QuantLinear)]
    for m in layers:
        m.emulate_q8_activations = on
    return len(layers)
