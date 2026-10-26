"""MTP heads for our own models: extra outputs trained to guess further ahead.

A normal language model is trained to predict only the next token. Multi-token
prediction (MTP; DeepSeek-V3 section 2.2, Gemma 4's and Qwen3.5's drafters) adds
small "heads" that learn to predict the tokens after that. At inference they make
the guesses for speculative decoding (infer/speculative.py): the big model then
checks several guesses in one pass, so it produces 1.5-2.5x more tokens per pass
with exactly the same text.

Head k (k = 1..mtp_heads) predicts the token k+1 places ahead. Like DeepSeek's MTP
modules, each head is told the token just before the one it predicts, and the
heads form a chain:

    h_0 = the model's final hidden state at position i   (after norm_f)
    h_k = head_k(h_{k-1}, embed(x[i+k]))                  (x = the token sequence)
    head_k's guess for x[i+k+1] = lm_head(h_k)            (the model's own output layer)

One head = two input norms, a projection [embed, h] (2d) -> d, one MLP block with
a residual connection, and an output norm. Unlike DeepSeek's (and Qwen's) MTP
modules, the heads have no attention: h_0 has already looked at the whole text,
and without attention a head needs no KV cache, so drafting is a few matmuls per
guess and needs no catch-up step after each check. (Hydra, Ankner et al. 2024,
showed such sequentially dependent heads draft well.) Attention in the heads is
an experiment for later.

Training: loss = main loss + mtp_weight x the mean of the heads' losses (TrainConfig).
The heads' gradients also reach the main model through h_0, which is the "extra
training signal" DeepSeek found slightly improves the main model too. Cost per
head: one MLP and one more pass through the output layer (the big one for a
small model with a 32k vocabulary).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .mlp import make_mlp
from .norm import make_norm


class MTPHead(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.enorm = make_norm(cfg.norm_type, d, cfg.norm_eps)    # the embedding of the given token
        self.hnorm = make_norm(cfg.norm_type, d, cfg.norm_eps)    # the previous hidden state
        self.proj = nn.Linear(2 * d, d, bias=cfg.bias)
        self.norm = make_norm(cfg.norm_type, d, cfg.norm_eps)
        self.mlp = make_mlp(cfg)
        self.out_norm = make_norm(cfg.norm_type, d, cfg.norm_eps)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """h [..., d] (the previous head's output, or the model's), e [..., d] the
        embedding of the token after h's position -> this head's h [..., d]."""
        x = self.proj(torch.cat([self.enorm(e), self.hnorm(h)], dim=-1))
        x = x + self.mlp(self.norm(x))
        return self.out_norm(x)


def mtp_loss(model, idx: torch.Tensor, h: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy of the heads. idx, targets [B, T] (targets[i] = x[i+1]);
    h [B, T, d] the model's final hidden states. Head k sees positions 0..T-k-1:
    h_{k-1} there and the token x[i+k] = idx[:, k:], and is scored on
    x[i+k+1] = targets[:, k:]."""
    T = idx.shape[1]
    losses = []
    hk = h
    for k, head in enumerate(model.mtp, start=1):
        if T - k <= 0:
            break
        e = model.embed(idx[:, k:])
        hk = head(hk[:, :T - k], e)
        logits = model._softcap(model.lm_head(hk))
        losses.append(F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                      targets[:, k:].reshape(-1), ignore_index=-100))
    return torch.stack(losses).mean()
