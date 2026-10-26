"""Normalization layers.

RMSNorm (used by Llama, Qwen, Mistral...) rescales each vector to unit
root-mean-square, then multiplies by a learned per-channel gain:

    y = x / sqrt(mean(x^2) + eps) * weight

Unlike LayerNorm it doesn't subtract the mean and has no bias: cheaper, and
in practice just as good. We compute in float32 even under bf16 autocast,
because the mean of squares loses precision in bf16 (only 8 mantissa bits).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn


def use_fused(x: torch.Tensor) -> bool:
    """Inference on a GPU: run the one-kernel Triton versions (fused_ops.py)? The
    cheap checks come first, so training and CPU runs never import Triton."""
    if not (x.is_cuda or os.environ.get("TRITON_INTERPRET") == "1"):
        return False
    if torch.is_grad_enabled() and x.requires_grad:
        return False
    from . import fused_ops

    return fused_ops.usable(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.rms_norm computes exactly x / sqrt(mean(x²) + eps) * weight, but as one
        # fused operation instead of ~7 small GPU kernels. That matters when
        # decoding one token at a time, where launching kernels is most of the cost
        # (Gemma 4 runs 6 norms in each of its 48 layers). Still ~9 kernels with the
        # fp32 casts, so inference on a GPU uses a single Triton kernel instead.
        if use_fused(x):
            from .fused_ops import rms_norm

            return rms_norm(x, self.weight, self.eps)
        return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps).to(x.dtype)


def make_norm(kind: str, dim: int, eps: float) -> nn.Module:
    if kind == "rmsnorm":
        return RMSNorm(dim, eps)
    if kind == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(kind)
