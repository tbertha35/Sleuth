"""Feed-forward (MLP) blocks: where most parameters and "neurons" live.

SwiGLU (default):  down( silu(gate(x)) * up(x) )
GELU MLP:          down( gelu(up(x)) )

SwiGLU's elementwise product acts as a learned, input-dependent gate.
Since PaLM/Llama it reliably beats a plain MLP at equal parameter count.
The "neuron" activations the analyzer studies are the hidden vector before
`down`; that's what the `act` tap exposes. Note that SiLU is smooth and can
be slightly negative, so unlike ReLU there's no natural on/off: "firing"
needs a threshold (see analyzer/neurons.py).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .taps import Tap


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.hidden, bias=cfg.bias)
        self.up = nn.Linear(cfg.d_model, cfg.hidden, bias=cfg.bias)
        self.down = nn.Linear(cfg.hidden, cfg.d_model, bias=cfg.bias)
        self.tap_act = Tap()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tap_act(F.silu(self.gate(x)) * self.up(x))
        return self.down(h)


class GeluMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.hidden, bias=cfg.bias)
        self.down = nn.Linear(cfg.hidden, cfg.d_model, bias=cfg.bias)
        self.tap_act = Tap()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.tap_act(F.gelu(self.up(x))))


def make_mlp(cfg: ModelConfig) -> nn.Module:
    return SwiGLU(cfg) if cfg.ffn_type == "swiglu" else GeluMLP(cfg)
