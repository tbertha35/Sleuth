"""Feed-forward (MLP) blocks: where most parameters and "neurons" live.

SwiGLU (default):  down( silu(gate(x)) * up(x) )
GeGLU (Gemma):     down( gelu_tanh(gate(x)) * up(x) )   same shape, another activation
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
    """Gated MLP. With ffn_type "geglu" the gate uses GELU (tanh approximation,
    which is what ggml_gelu and Gemma's "gelu_pytorch_tanh" compute)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.hidden, bias=cfg.bias)
        self.up = nn.Linear(cfg.d_model, cfg.hidden, bias=cfg.bias)
        self.down = nn.Linear(cfg.hidden, cfg.d_model, bias=cfg.bias)
        self.tap_act = Tap()
        self.geglu = cfg.ffn_type == "geglu"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate(x)
        g = F.gelu(g, approximate="tanh") if self.geglu else F.silu(g)
        h = self.tap_act(g * self.up(x))
        return self.down(h)


class GeluMLP(nn.Module):
    """ffn_type "gelu": exact GELU (Pythia); "gelu_tanh": the tanh approximation
    (GPT-2's "gelu_new")."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.hidden, bias=cfg.bias)
        self.down = nn.Linear(cfg.hidden, cfg.d_model, bias=cfg.bias)
        self.tap_act = Tap()
        self.approximate = "tanh" if cfg.ffn_type == "gelu_tanh" else "none"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.tap_act(F.gelu(self.up(x), approximate=self.approximate)))


def make_mlp(cfg: ModelConfig) -> nn.Module:
    return SwiGLU(cfg) if cfg.ffn_type in ("swiglu", "geglu") else GeluMLP(cfg)
