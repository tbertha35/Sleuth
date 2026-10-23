"""Taps: named points in the model where the analyzer can look or intervene.

A `Tap` is an identity layer. In normal training it has no listeners and
just returns its input. The analyzer attaches listeners by name pattern:

    with listen(model, {"blocks.*.mlp.act": record}):
        model(tokens)

A listener is called as fn(tap_name, tensor). If it returns a tensor, that
replaces the activation. That one mechanism gives us both *observation*
(activation stats, logit lens) and *intervention* (ablation: return zeros).

Why not plain PyTorch forward hooks? Hooks attach to modules, and many
interesting tensors aren't a module's output (e.g. the SwiGLU product
silu(gate)·up, or per-head attention output before the output projection).
Explicit taps give those tensors stable names, documented in one place.
External models loaded through adapters reuse our modules, so they get the
same names for free.

Tap names (L = layer index):
    embed                      token embeddings entering block 0     [B, T, d]
    blocks.L.attn.q / .k       queries / keys after RoPE (k is per KV head) [B, H, T, hd]
    blocks.L.attn.pattern      attention probabilities               [B, H, T, S]
                               (only computed when something listens: it
                               forces the slower non-fused attention path)
    blocks.L.attn.z            per-head attention output             [B, T, H, hd]
    blocks.L.attn_out          attention block output                [B, T, d]
    blocks.L.mlp.act           MLP hidden activation ("neurons")     [B, T, ffn_hidden]
    blocks.L.mlp_out           MLP block output                      [B, T, d]
    blocks.L.resid_mid         residual stream after attention       [B, T, d]
    blocks.L.resid_post        residual stream after the block       [B, T, d]
    final_norm                 residual after the final norm         [B, T, d]
"""

from __future__ import annotations

import fnmatch
from contextlib import contextmanager
from typing import Callable, Iterator

import torch
from torch import nn

Listener = Callable[[str, torch.Tensor], "torch.Tensor | None"]


class Tap(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "<unnamed>"
        self.listeners: list[Listener] = []

    @property
    def active(self) -> bool:
        return bool(self.listeners)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fn in self.listeners:
            out = fn(self.name, x)
            if out is not None:
                x = out
        return x

    def extra_repr(self) -> str:
        return self.name


def assign_tap_names(model: nn.Module) -> None:
    """Name each Tap after its position in the module tree, dropping the
    "tap_" attribute prefix: blocks.3.mlp.tap_act -> blocks.3.mlp.act."""
    for path, m in model.named_modules():
        if isinstance(m, Tap):
            parts = path.split(".")
            parts[-1] = parts[-1].removeprefix("tap_")
            m.name = ".".join(parts)


def find_taps(model: nn.Module, pattern: str) -> list[Tap]:
    taps = [m for m in model.modules() if isinstance(m, Tap) and fnmatch.fnmatchcase(m.name, pattern)]
    if not taps:
        raise KeyError(f"no tap matches {pattern!r}; available: {', '.join(tap_names(model)[:12])} ...")
    return taps


def tap_names(model: nn.Module) -> list[str]:
    return [m.name for m in model.modules() if isinstance(m, Tap)]


@contextmanager
def listen(model: nn.Module, listeners: dict[str, Listener]) -> Iterator[None]:
    """Attach listeners for the duration of a `with` block (always removed,
    even on error, so a crashed analysis can't leave the model modified)."""
    attached: list[tuple[Tap, Listener]] = []
    try:
        for pattern, fn in listeners.items():
            for tap in find_taps(model, pattern):
                tap.listeners.append(fn)
                attached.append((tap, fn))
        yield
    finally:
        for tap, fn in attached:
            tap.listeners.remove(fn)
