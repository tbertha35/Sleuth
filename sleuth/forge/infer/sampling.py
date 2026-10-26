"""Turning logits into the next token.

Applied in this order (the usual order in llama.cpp / vLLM):
  0. no-repeat n-gram: forbid any token that would repeat an n-token phrase
     already in the text (Hugging Face's no_repeat_ngram_size). A hard ban,
     so it also works with greedy decoding.
  1. repetition penalty: make tokens seen in the last `repeat_last_n`
     tokens less likely (llama.cpp's repeat_penalty / repeat_last_n)
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
    repeat_last_n: int = 64         # the penalty looks at this many recent tokens; 0 = the whole context
    no_repeat_ngram: int = 0        # 0 = off; n = never repeat an n-token phrase


class NgramBlocker:
    """Remembers every n-gram in the text so far, to ban the tokens that would
    repeat one. Small models decoded greedily fall into loops ("The capital of
    France is the capital of France is ..."): the most likely continuation of
    a phrase is the phrase again. Banning repeats of n tokens breaks the loop
    while leaving short, legitimate repeats ("of the") alone when n is 3-4.

    Incremental (a dict from the last n-1 tokens to the tokens that followed
    them), so each step costs O(1) instead of rescanning the whole context.
    The prompt counts too, as in Hugging Face's no_repeat_ngram_size."""

    def __init__(self, n: int, ids: list[int] = ()):
        self.n = n
        self.ids: list[int] = []
        self.followers: dict[tuple[int, ...], set[int]] = {}
        for t in ids:
            self.add(t)

    def add(self, token: int) -> None:
        self.ids.append(token)
        if self.n > 0 and len(self.ids) >= self.n:
            self.followers.setdefault(tuple(self.ids[len(self.ids) - self.n:-1]), set()).add(token)

    def banned(self) -> list[int]:
        """Tokens that would complete an n-gram already seen."""
        if self.n <= 0 or len(self.ids) < self.n - 1:
            return []
        key = tuple(self.ids[len(self.ids) - (self.n - 1):]) if self.n > 1 else ()
        return sorted(self.followers.get(key, ()))


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
                generator: torch.Generator | None = None, banned: list[int] | None = None) -> torch.Tensor:
    """logits: [B, V] float. context: [B, S] token IDs so far. banned: token IDs
    that must not be chosen (from NgramBlocker; batch size 1). Returns [B] token IDs."""
    logits = final_logits(logits, cfg, context, banned)
    if cfg.temperature <= 0:
        return logits.argmax(-1)
    probs = logits.softmax(-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


def final_logits(logits: torch.Tensor, cfg: SamplingConfig, context: torch.Tensor | None = None,
                 banned: list[int] | None = None) -> torch.Tensor:
    """Steps 0-5 above: the logits sample_next actually samples from (softmax
    them for the probabilities; with temperature 0, argmax them). Speculative
    decoding checks draft tokens against exactly this distribution."""
    logits = logits.float()
    if banned:
        ban = torch.tensor([b for b in banned if b < logits.size(-1)], dtype=torch.long, device=logits.device)
        masked = logits.index_fill(-1, ban, torch.finfo(logits.dtype).min)
        # If every token is banned (only possible with a tiny vocab), ignore the ban.
        logits = masked if bool((masked > torch.finfo(logits.dtype).min).any()) else logits
    if context is not None:
        if cfg.repeat_last_n > 0:
            context = context[..., -cfg.repeat_last_n:]
        logits = apply_repetition_penalty(logits, context, cfg.repetition_penalty)
    if cfg.temperature <= 0:
        return logits
    return filter_logits(logits / cfg.temperature, cfg)
