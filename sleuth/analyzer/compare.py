"""Compare two models on the same tests: do they predict the same things?

Typical uses:
  * checking a loader: the same model from .safetensors and from a Q8_0 GGUF
    should agree almost perfectly (so a disagreement means a bug, not a
    difference between the models)
  * measuring what quantization costs: Q8_0 vs Q4_K_M vs the original
  * later, comparing checkpoints or LAB variants

For every next-token prediction in the suite:
  top-1 agreement   how often both models' single most likely token is the same
  KL(A || B)        how far B's probability distribution is from A's, in nats.
                    0 = identical. Rough guide: < 0.01 is tiny (8-bit
                    quantization), ~0.05-0.2 is typical for 4-bit, > 1 means
                    the models think quite differently (or something is broken).
  loss A / loss B   each model's own cross-entropy on the true next tokens

The two models must share a tokenizer (same token IDs for the same text);
that's checked first, because comparing predictions over different
vocabularies is meaningless. Models are loaded one at a time, so comparing two
large models doesn't need memory for both.
"""

from __future__ import annotations

import gc
from typing import Callable

import torch
import torch.nn.functional as F

from .common import encode_cases


@torch.no_grad()
def _log_probs(model, enc, vocab: int, device) -> list[torch.Tensor]:
    out = []
    for e in enc:
        logits = model(e.ids.to(device)).logits[0, :-1, :vocab].float()
        out.append(F.log_softmax(logits, dim=-1).to(torch.float16).cpu())  # fp16 on CPU: compact
    return out


def compare_models(load_a: Callable, load_b: Callable, cases, device="cpu", log=print) -> dict:
    """load_a / load_b: zero-argument functions returning (model, tokenizer)."""
    model, tok_a = load_a()
    enc = encode_cases(tok_a, cases, model.cfg.max_seq_len)
    vocab = min(tok_a.vocab_size, model.cfg.padded_vocab)
    lp_a = _log_probs(model, enc, vocab, device)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model, tok_b = load_b()
    mismatched = [e.case.id for e in enc if tok_b.encode(e.case.prompt) != tok_a.encode(e.case.prompt)]
    if mismatched:
        raise ValueError(f"the tokenizers disagree on {len(mismatched)} tests (e.g. {mismatched[:3]}); "
                         f"comparing predictions needs identical token IDs")
    vocab = min(vocab, tok_b.vocab_size, model.cfg.padded_vocab)
    lp_b = _log_probs(model, enc, vocab, device)
    del model

    stats: dict[str, dict[str, float]] = {}
    for e, a, b in zip(enc, lp_a, lp_b):
        a, b = a[:, :vocab].float(), b[:, :vocab].float()
        targets = e.ids[0, 1:]
        kl = (a.exp() * (a - b)).sum(-1)                   # KL(A || B) per position
        s = stats.setdefault(e.category, {"n": 0, "agree": 0, "kl": 0.0, "kl_max": 0.0, "loss_a": 0.0, "loss_b": 0.0})
        s["n"] += targets.numel()
        s["agree"] += int((a.argmax(-1) == b.argmax(-1)).sum())
        s["kl"] += float(kl.sum())
        s["kl_max"] = max(s["kl_max"], float(kl.max()))
        s["loss_a"] -= float(a.gather(-1, targets[:, None]).sum())
        s["loss_b"] -= float(b.gather(-1, targets[:, None]).sum())

    def finish(s):
        n = max(1, s["n"])
        # KL is never negative; tiny negatives are fp16 rounding.
        return {"tokens": s["n"], "top1_agreement": s["agree"] / n, "kl_mean": max(0.0, s["kl"] / n),
                "kl_max": max(0.0, s["kl_max"]),
                "loss_a": s["loss_a"] / n, "loss_b": s["loss_b"] / n}

    total = {"n": 0, "agree": 0, "kl": 0.0, "kl_max": 0.0, "loss_a": 0.0, "loss_b": 0.0}
    for s in stats.values():
        for k in total:
            total[k] = max(total[k], s[k]) if k == "kl_max" else total[k] + s[k]
    return {c: finish(s) for c, s in sorted(stats.items())} | {"overall": finish(total)}


def format_report(result: dict, name_a: str, name_b: str) -> str:
    lines = [f"A = {name_a}", f"B = {name_b}", "",
             f"{'category':12s} {'tokens':>7s} {'top-1 agree':>12s} {'KL mean':>9s} {'KL max':>8s} {'loss A':>7s} {'loss B':>7s}"]
    for c, r in result.items():
        lines.append(f"{c:12s} {r['tokens']:7d} {r['top1_agreement']:12.1%} {r['kl_mean']:9.4f} {r['kl_max']:8.3f} "
                     f"{r['loss_a']:7.3f} {r['loss_b']:7.3f}")
    return "\n".join(lines)
