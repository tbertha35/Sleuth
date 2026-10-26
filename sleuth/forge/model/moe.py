"""Mixture-of-experts feed-forward layers (Qwen3.6-35B-A3B, Gemma 4 26B-A4B).

Instead of one big MLP, an MoE layer has many small ones ("experts") and a router
that picks a few of them for each token. Qwen3.6-35B-A3B has 256 experts and uses
8 per token: 35B weights, but only ~3B of them are used for any one token. That
is why these models are fast for their size, and why their weights are a good
candidate for offloading: every expert has to be *stored*, but only the chosen
ones are *read*.

The computation, following llama.cpp's build_moe_ffn (src/llama-graph.cpp, commit
b2dd28a3b), which is the reference for how GGUF files of these models run:

    logits  = router(x)                          [n_experts] per token
    probs   = softmax(logits)
    chosen  = the top n_used experts by probs
    w       = probs[chosen]; w /= sum(w) (if normalized); w *= weights_scale
    out     = sum over chosen e of  w_e * down_e( act(gate_e(x)) * up_e(x) )

act is SiLU (SwiGLU experts: Qwen) or GELU-tanh (GeGLU: Gemma). Some files hold
per-expert output scales (`ffn_down_exps.scale`: one number per expert) that
multiply that projection's output.

The expert weights stay in the file's layout: one [n_experts, out, in] tensor per
projection, either plain or as quantized blocks, expert after expert (so expert
e's blocks are one contiguous slice). Only the chosen experts are computed, one
expert at a time over the tokens that chose it. That's simple and exact. It costs
a host sync per layer to learn which experts were chosen, so it doesn't fit in a
CUDA graph yet; a fused "grouped GEMM" kernel is the faster form, for later.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ...formats import gguf_quants as q
from .quant import quant_linear, row_bytes

# llama.cpp clamps the sum of the chosen weights before dividing by it: the
# smallest normal fp16 number, so the division is safe in half precision too.
MIN_WEIGHT_SUM = 6.103515625e-5


class ExpertWeights(nn.Module):
    """n_experts weight matrices of shape [out_features, in_features].

    ggml_type None: a plain Parameter [n_experts, out, in]. Otherwise a uint8
    buffer with the GGUF blocks of all experts, expert-major (like QuantLinear)."""

    def __init__(self, n_experts: int, in_features: int, out_features: int, ggml_type: int | None = None):
        super().__init__()
        self.n_experts, self.in_features, self.out_features = n_experts, in_features, out_features
        self.ggml_type = ggml_type
        if ggml_type is None:
            self.weight = nn.Parameter(torch.empty(n_experts, out_features, in_features))
        else:
            self.expert_bytes = out_features * row_bytes(ggml_type, in_features)
            self.register_buffer("raw", torch.empty(n_experts * self.expert_bytes, dtype=torch.uint8))

    @property
    def n_weights(self) -> int:
        return self.n_experts * self.out_features * self.in_features

    def forward(self, x: torch.Tensor, e: int) -> torch.Tensor:
        """x [n, in] through expert e -> [n, out]."""
        if self.ggml_type is None:
            return F.linear(x, self.weight[e].to(x.dtype))
        raw = self.raw[e * self.expert_bytes:(e + 1) * self.expert_bytes]
        return quant_linear(x, raw, self.ggml_type, self.in_features, self.out_features)

    def expert(self, e: int) -> torch.Tensor:
        """Expert e's weight, dequantized to fp32 (for tests and inspection)."""
        if self.ggml_type is None:
            return self.weight[e].float()
        raw = self.raw[e * self.expert_bytes:(e + 1) * self.expert_bytes]
        return q.dequantize(raw, self.ggml_type, self.out_features * self.in_features,
                            torch.float32).view(self.out_features, self.in_features)

    def extra_repr(self) -> str:
        kind = "plain" if self.ggml_type is None else q.TYPE_NAMES[self.ggml_type]
        return f"experts={self.n_experts}, in={self.in_features}, out={self.out_features}, {kind}"


class MoE(nn.Module):
    """Routed experts (see the module docstring). The shared expert, if the model
    has one, is the caller's business: Qwen and Gemma combine it differently.

    merged_gate_up: the file stores gate and up as one [E, 2*hidden, d] tensor
    (`ffn_gate_up_exps`, gate first) instead of two.
    down_scale: the file has per-expert scales for the down projection."""

    def __init__(self, d_model: int, n_experts: int, n_used: int, hidden: int, act: str = "silu",
                 norm_weights: bool = True, weights_scale: float = 1.0, merged_gate_up: bool = False,
                 down_scale: bool = False):
        super().__init__()
        if act not in ("silu", "gelu"):
            raise ValueError(f"act must be 'silu' or 'gelu', got {act!r}")
        self.n_experts, self.n_used, self.hidden = n_experts, n_used, hidden
        self.act, self.norm_weights, self.weights_scale = act, norm_weights, weights_scale
        self.router = nn.Linear(d_model, n_experts, bias=False)
        if merged_gate_up:
            self.gate_up = ExpertWeights(n_experts, d_model, 2 * hidden)
            self.gate = self.up = None
        else:
            self.gate_up = None
            self.gate = ExpertWeights(n_experts, d_model, hidden)
            self.up = ExpertWeights(n_experts, d_model, hidden)
        self.down = ExpertWeights(n_experts, hidden, d_model)
        self.down_scale = nn.Parameter(torch.ones(n_experts)) if down_scale else None
        # How often each expert was chosen since the last reset_counts(); the analyzer's
        # first question about an MoE model ("which experts does this text use?").
        self.register_buffer("counts", torch.zeros(n_experts, dtype=torch.long), persistent=False)

    def route(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """logits [N, E] -> (weights [N, k] fp32, chosen experts [N, k])."""
        probs = logits.float().softmax(-1)
        w, chosen = probs.topk(self.n_used, dim=-1)
        if self.norm_weights:
            w = w / w.sum(-1, keepdim=True).clamp_min(MIN_WEIGHT_SUM)
        if self.weights_scale not in (0.0, 1.0):
            w = w * self.weights_scale
        return w, chosen

    def expert_forward(self, x: torch.Tensor, e: int) -> torch.Tensor:
        if self.gate_up is not None:
            g, u = self.gate_up(x, e).split(self.hidden, dim=-1)
        else:
            g, u = self.gate(x, e), self.up(x, e)
        g = F.silu(g) if self.act == "silu" else F.gelu(g, approximate="tanh")
        y = self.down(g * u, e)
        if self.down_scale is not None:
            y = y * self.down_scale[e].to(y.dtype)
        return y

    def forward(self, x: torch.Tensor, router_input: torch.Tensor | None = None) -> torch.Tensor:
        """x [..., d] -> [..., d]. router_input: what the router sees, if not x (Gemma 4)."""
        shape = x.shape
        xf = x.reshape(-1, shape[-1])
        r = xf if router_input is None else router_input.reshape(-1, shape[-1])
        with torch.autocast(x.device.type, enabled=False):    # the router stays fp32 under autocast too
            w, chosen = self.route(self.router(r.to(self.router.weight.dtype)))
        out = torch.zeros(xf.shape, dtype=torch.float32, device=x.device)
        used = chosen.unique()
        self.counts.index_add_(0, chosen.reshape(-1), torch.ones_like(chosen.reshape(-1)))
        for e in used.tolist():
            tok, slot = (chosen == e).nonzero(as_tuple=True)
            y = self.expert_forward(xf[tok], e)
            out.index_add_(0, tok, y.float() * w[tok, slot].unsqueeze(-1))
        return out.to(x.dtype).reshape(shape)

    def reset_counts(self) -> None:
        self.counts.zero_()


def make_quantized_experts(old: ExpertWeights, ggml_type: int) -> ExpertWeights:
    """A quantized ExpertWeights of the same shape (Transformer.empty's quantized=)."""
    return ExpertWeights(old.n_experts, old.in_features, old.out_features, ggml_type)
