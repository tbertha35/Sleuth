"""Analyzer view: the logit lens. What would each layer predict?

The residual stream is the running sum every layer adds to (see
forge/model/transformer.py). At the end, the model turns it into
next-token predictions with the final norm + unembedding. The logit lens applies
that same final step *early*, after each layer, and asks: if the model stopped
here, how good would its prediction be?

Rows: "emb" (just the token embedding, before any layer) and "L0".."L{n-1}"
(after each block). The last row is exactly the model's real output.

Per row, over every next-token prediction in the suite:
  loss            cross-entropy of the true next token (nats; lower is better)
  top1_accuracy   how often the true next token is the single top prediction
broken down by test category.

How to read it: loss usually falls layer by layer. A layer where it drops
sharply is doing a lot of the work for that category. In small models the
early rows are often poor, because intermediate layers don't use the "output
language" yet. (A refinement called the *tuned lens* learns a small
correction per layer; it's a possible later addition.)

An example grid (row × position: top-1 guess, its probability, and the
probability of the true token) is saved for one prompt, for the dashboard.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..forge.model.taps import listen
from .common import categories, encode_cases, series

EXAMPLE_MAX_TOKENS = 40


@torch.no_grad()
def run(model, tokenizer, cases, device, **_) -> dict:
    model.eval().to(device)
    L = model.cfg.n_layers
    V = tokenizer.vocab_size
    rows = ["emb"] + [f"L{l}" for l in range(L)]
    enc = encode_cases(tokenizer, cases, model.cfg.max_seq_len)
    cats = categories(enc)
    ci = {c: i for i, c in enumerate(cats)}
    loss = torch.zeros(len(rows), len(cats), dtype=torch.float64)
    correct = torch.zeros(len(rows), len(cats), dtype=torch.float64)
    n_tok = torch.zeros(len(cats), dtype=torch.float64)

    captured: dict[int, torch.Tensor] = {}

    def grab(name, h):
        captured[0 if name == "embed" else int(name.split(".")[1]) + 1] = h

    taps = {"embed": grab, "blocks.*.resid_post": grab}

    def lens(ids):
        """Per-row logits [rows, T, V] for one sequence."""
        captured.clear()
        with listen(model, taps):
            model(ids)
        return torch.stack([model.unembed(captured[r])[0, :, :V].float() for r in range(len(rows))])

    for e in enc:
        ids = e.ids.to(device)
        if ids.shape[1] < 2:
            continue
        logits = lens(ids)[:, :-1]                          # predictions for tokens 1..T-1
        targets = ids[0, 1:].expand(len(rows), -1)
        nll = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")  # [rows, T-1]
        c = ci[e.category]
        loss[:, c] += nll.sum(-1).double().cpu()
        correct[:, c] += (logits.argmax(-1) == targets).sum(-1).double().cpu()
        n_tok[c] += targets.shape[1]

    loss_by_cat = loss / n_tok.clamp(min=1)
    acc_by_cat = correct / n_tok.clamp(min=1)
    loss_all = loss.sum(1) / n_tok.sum().clamp(min=1)
    acc_all = correct.sum(1) / n_tok.sum().clamp(min=1)

    # ---- example grid for the dashboard
    example = None
    arrays = {"loss_by_category": loss_by_cat.float().numpy(), "top1_by_category": acc_by_cat.float().numpy()}
    if enc:
        ex = max(enc, key=lambda e: min(e.ids.shape[1], EXAMPLE_MAX_TOKENS))
        ids = ex.ids[:, :EXAMPLE_MAX_TOKENS].to(device)
        probs = lens(ids)[:, :-1].softmax(-1)               # [rows, T-1, V]
        top_p, top_i = probs.max(-1)
        true_p = probs.gather(-1, ids[0, 1:].expand(len(rows), -1).unsqueeze(-1)).squeeze(-1)
        example = {
            "case_id": ex.case.id,
            "tokens": [tokenizer.decode([i]) for i in ids[0].tolist()],
            "top_tokens": [[tokenizer.decode([i]) for i in row] for row in top_i.tolist()],
        }
        arrays["example_top_p"] = top_p.cpu().numpy()
        arrays["example_true_p"] = true_p.cpu().numpy()

    summary = {
        "view": "logit_lens",
        "rows": rows,
        "categories": cats,
        "n_tokens": int(n_tok.sum()),
        "loss": [float(x) for x in loss_all],
        "top1_accuracy": [float(x) for x in acc_all],
        "loss_by_category": {c: [float(x) for x in loss_by_cat[:, i]] for c, i in ci.items()},
        "top1_by_category": {c: [float(x) for x in acc_by_cat[:, i]] for c, i in ci.items()},
        "example": example,
    }
    return {"summary": summary, "arrays": arrays}


def timeline_series(summary: dict) -> dict:
    rows = summary["rows"]
    out = {
        "top1_accuracy": series("Top-1 accuracy per layer", "How often each layer's early guess (through the final "
                                "norm + unembedding) is exactly the true next token.", rows,
                                summary["top1_accuracy"], "pct"),
        "loss": series("Loss per layer (nats)", "Cross-entropy of each layer's early guess. The last row is the "
                       "model's real loss.", rows, summary["loss"]),
    }
    for c, vals in summary["top1_by_category"].items():
        out[f"top1:{c}"] = series(f"Top-1 accuracy on “{c}”", "Same as top-1 accuracy, for this category only.",
                                  rows, vals, "pct")
    return out
