"""Normalization layers.

RMSNorm (used by Llama, Qwen, Mistral...) rescales each vector to unit
root-mean-square, then multiplies by a learned per-channel gain:

    y = x / sqrt(mean(x^2) + eps) * weight

Unlike LayerNorm it doesn't subtract the mean and has no bias: cheaper, and
in practice just as good. We compute in float32 even under bf16 autocast,
because the mean of squares loses precision in bf16 (only 8 mantissa bits).
"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


def make_norm(kind: str, dim: int, eps: float) -> nn.Module:
    if kind == "rmsnorm":
        return RMSNorm(dim, eps)
    if kind == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(kind)
