"""Turning logits into the next token.

Applied in this order (the usual order in llama.cpp / vLLM):
  1. repetition penalty: make tokens already in the context less likely
  2. temperature: divide logits by T. T < 1 sharpens (safer, more
     repetitive), T > 1 flattens (more varied, more mistakes). T = 0 means
     greedy: always pick the most likely token.
  3. top-k: keep only the k most likely tokens
  4. top-p (nucleus): keep the smallest set whose probabilities sum to >= p
  5. min-p: drop tokens less likely than min_p × (probability of the best)
  6. sample from what's left
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SamplingConfig:
    temperature: float = 0.8
    top_k: int = 0                  # 0 = off
    top_p: float = 1.0              # 1.0 = off
    min_p: float = 0.0              # 0.0 = off
    repetition_penalty: float = 1.0  # 1.0 = off; >1 discourages repeats


def apply_repetition_penalty(logits: torch.Tensor, context: torch.Tensor, penalty: float) -> torch.Tensor:
    """CTRL-style: for tokens already seen, divide positive logits by the
    penalty and multiply negative ones (both lower the probability)."""
    if penalty == 1.0 or context.numel() == 0:
        return logits
    logits = logits.clone()
    seen = logits.gather(-1, context)
    logits.scatter_(-1, context, torch.where(seen > 0, seen / penalty, seen * penalty))
    return logits


def filter_logits(logits: torch.Tensor, cfg: SamplingConfig) -> torch.Tensor:
    """logits: [B, V] (already temperature-scaled). Returns logits with
    excluded tokens set to -inf."""
    neg_inf = torch.finfo(logits.dtype).min
    if cfg.top_k > 0 and cfg.top_k < logits.size(-1):
        kth = torch.topk(logits, cfg.top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, neg_inf)
    if cfg.top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True, dim=-1)
        cum = sorted_logits.softmax(-1).cumsum(-1)
        # Remove a token if the tokens *before* it already reach top_p. This
        # always keeps the most likely token.
        remove_sorted = (cum - sorted_logits.softmax(-1)) >= cfg.top_p
        remove = remove_sorted.scatter(-1, idx, remove_sorted)
        logits = logits.masked_fill(remove, neg_inf)
    if cfg.min_p > 0.0:
        probs = logits.softmax(-1)
        logits = logits.masked_fill(probs < cfg.min_p * probs.max(-1, keepdim=True).values, neg_inf)
    return logits


def sample_next(logits: torch.Tensor, cfg: SamplingConfig, context: torch.Tensor | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """logits: [B, V] float. context: [B, S] token IDs so far. Returns [B] token IDs."""
    logits = logits.float()
    if context is not None:
        logits = apply_repetition_penalty(logits, context, cfg.repetition_penalty)
    if cfg.temperature <= 0:
        return logits.argmax(-1)
    logits = filter_logits(logits / cfg.temperature, cfg)
    probs = logits.softmax(-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)
