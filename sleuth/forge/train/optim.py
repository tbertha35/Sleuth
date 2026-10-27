"""Optimizer construction.

AdamW with *decoupled* weight decay, applied only to 2-D weight matrices
(linear layers, embeddings). Norm gains and biases are excluded: decaying
a norm's gain toward 0 fights its job of setting the scale, and they're
too few parameters to overfit anyway.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import TrainConfig


def build_optimizer(model: nn.Module, cfg: TrainConfig, device_type: str) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    seen: set[int] = set()
    for p in model.parameters():  # .parameters() already yields tied weights once
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        # no_weight_decay: lookup tables most of whose rows get no gradient on a given
        # step (the n-gram memory); decay would keep shrinking the rarely-seen ones.
        (decay if p.dim() >= 2 and not getattr(p, "no_weight_decay", False) else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    # fused=True runs the whole update as a few CUDA kernels instead of a
    # loop over parameters; noticeably faster on GPU, unsupported on CPU.
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=1e-8,
                             fused=(device_type == "cuda"))
