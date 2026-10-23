"""Shared helpers for analyzer views: encoding tests and scoring a model on them.

Every test case becomes   [<|bos|>] + prompt tokens + expected tokens
and we remember where the expected part starts. Prompt and expected text are
encoded separately so we know exactly which tokens are "the answer" (encoding
them together could merge a token across the boundary).

Scoring is *teacher-forced*: we feed the full sequence once and read off, at
every position, the probability the model gave to the true next token. This
is how loss is measured during training too, and it's cheap: one forward pass
per test, no sampling.

  token loss      mean cross-entropy (nats) over every next-token prediction
  expected loss   mean cross-entropy over just the expected answer's tokens
  exact match     1 if every answer token was the model's top choice, else 0
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .suites import TestCase


@dataclass
class EncodedCase:
    case: TestCase
    ids: torch.Tensor   # [1, T] on CPU
    n_prompt: int       # tokens before the answer, including <|bos|>

    @property
    def category(self) -> str:
        return self.case.category

    @property
    def answer_positions(self) -> slice:
        """Positions whose *next-token prediction* is an answer token."""
        return slice(self.n_prompt - 1, self.ids.shape[1] - 1)


def encode_cases(tokenizer, cases: list[TestCase], max_len: int) -> list[EncodedCase]:
    out = []
    for c in cases:
        prompt = [tokenizer.bos_id] + tokenizer.encode(c.prompt)
        answer = tokenizer.encode(c.expected) if c.expected else []
        ids = (prompt + answer)[:max_len]
        out.append(EncodedCase(c, torch.tensor([ids]), min(len(prompt), len(ids))))
    return out


def categories(enc: list[EncodedCase]) -> list[str]:
    return sorted({e.category for e in enc})


def layer_of(tap_name: str) -> int:
    """'blocks.3.mlp.act' -> 3"""
    return int(tap_name.split(".")[1])


@torch.no_grad()
def score_cases(model, enc: list[EncodedCase], vocab: int, device) -> dict:
    """Teacher-forced loss/accuracy per category (see module docstring)."""
    acc: dict[str, dict[str, float]] = {}
    for e in enc:
        ids = e.ids.to(device)
        logits = model(ids).logits[0, :-1, :vocab].float()
        targets = ids[0, 1:]
        nll = F.cross_entropy(logits, targets, reduction="none")
        a = acc.setdefault(e.category, {"loss_sum": 0.0, "n_tok": 0, "exp_loss_sum": 0.0, "n_exp_tok": 0,
                                        "exact": 0, "n_exp_cases": 0})
        a["loss_sum"] += float(nll.sum())
        a["n_tok"] += int(nll.numel())
        span = e.answer_positions
        if span.stop > span.start:
            a["exp_loss_sum"] += float(nll[span].sum())
            a["n_exp_tok"] += span.stop - span.start
            a["exact"] += int(bool((logits[span].argmax(-1) == targets[span]).all()))
            a["n_exp_cases"] += 1
    return {c: _finish(a) for c, a in acc.items()} | {"overall": _finish(_merge(acc.values()))}


def _merge(items) -> dict:
    out: dict[str, float] = {}
    for a in items:
        for k, v in a.items():
            out[k] = out.get(k, 0) + v
    return out


def _finish(a: dict) -> dict:
    return {
        "loss": a["loss_sum"] / max(1, a["n_tok"]),
        "expected_loss": a["exp_loss_sum"] / a["n_exp_tok"] if a["n_exp_tok"] else None,
        "exact_match": a["exact"] / a["n_exp_cases"] if a["n_exp_cases"] else None,
        "n_tokens": int(a["n_tok"]),
        "n_answers": int(a["n_exp_cases"]),
    }


def series(label: str, help: str, rows: list[str], values: list[float], fmt: str = "num") -> dict:
    """One timeline series for one checkpoint: a value per row (layer, head, ...).
    fmt: "num" or "pct" (0..1 shown as a percentage)."""
    return {"label": label, "help": help, "rows": rows, "values": [float(v) for v in values], "fmt": fmt}


def head_labels(n_layers: int, n_heads: int) -> list[str]:
    return [f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)]
