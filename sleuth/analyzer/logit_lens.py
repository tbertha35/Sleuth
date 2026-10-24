"""Analyzer view: the logit lens. What would each layer predict?

The residual stream is the running sum every layer adds to (see
forge/model/transformer.py). At the end, the model turns it into
next-token predictions with the final norm + unembedding. The logit lens applies
that same final step *early*, after each layer, and asks: if the model stopped
here, how good would its prediction be?

Rows: "emb" (just the token embedding, before any layer) and "L0".."L{n-1}"
(after each block). The last row is exactly the model's real output.

The "emb" row is a baseline, not a real prediction: with tied embeddings
(the same matrix in and out) the logit for token x is largest when x is the
*current* token, so it mostly "predicts" the input again and its loss is far
worse than a uniform guess. `emb_top1_is_input` records how often that happens.

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


LOGITS_PER_CALL = 1 << 26  # unembed up to 64M logits at once (~256 MB as float32)


@torch.no_grad()
def run(model, tokenizer, cases, device, prefix: list[int] | None = None, **_) -> dict:
    model.eval().to(device)
    L = model.cfg.n_layers
    V = tokenizer.vocab_size
    rows = ["emb"] + [f"L{l}" for l in range(L)]
    enc = encode_cases(tokenizer, cases, model.cfg.max_seq_len, prefix)
    cats = categories(enc)
    ci = {c: i for i, c in enumerate(cats)}
    loss = torch.zeros(len(rows), len(cats), dtype=torch.float64)
    correct = torch.zeros(len(rows), len(cats), dtype=torch.float64)
    n_tok = torch.zeros(len(cats), dtype=torch.float64)
    emb_is_input = 0

    captured: dict[int, torch.Tensor] = {}

    def grab(name, h):
        captured[0 if name == "embed" else int(name.split(".")[1]) + 1] = h

    taps = {"embed": grab, "blocks.*.resid_post": grab}

    def residuals(ids) -> torch.Tensor:
        """[rows, T, d]: the residual stream before any layer and after each one."""
        captured.clear()
        with listen(model, taps):
            model(ids)
        return torch.cat([captured[r] for r in range(len(rows))], dim=0)

    def row_logits(h):
        """[R, T, d] -> [R, T, V] float logits, in groups of rows.

        One call per group, not per row: unembedding is a [d x vocab] matmul,
        and with a quantized output layer (Gemma's 262k x 3840) every call
        dequantizes the whole matrix again. Groups keep memory bounded:
        all rows at once would be rows x T x V floats, gigabytes for a big vocab."""
        per = max(1, LOGITS_PER_CALL // max(1, h.shape[1] * V))
        for i in range(0, h.shape[0], per):
            yield i, model.unembed(h[i:i + per])[..., :V].float()

    for e in enc:
        ids = e.ids.to(device)
        sc = e.scored                                         # predictions of the test's own tokens
        if sc.stop <= sc.start:
            continue
        targets = ids[0, sc.start + 1: sc.stop + 1]
        c = ci[e.category]
        for r0, logits in row_logits(residuals(ids)):
            lg = logits[:, sc]                                # [g, n, V]
            g, n = lg.shape[:2]
            nll = F.cross_entropy(lg.reshape(g * n, V), targets.repeat(g), reduction="none").view(g, n)
            loss[r0:r0 + g, c] += nll.sum(1).double().cpu()
            top = lg.argmax(-1)                               # [g, n]
            correct[r0:r0 + g, c] += (top == targets).sum(1).double().cpu()
            if r0 == 0:
                emb_is_input += int((top[0] == ids[0, sc]).sum())
        n_tok[c] += targets.shape[0]

    loss_by_cat = loss / n_tok.clamp(min=1)
    acc_by_cat = correct / n_tok.clamp(min=1)
    loss_all = loss.sum(1) / n_tok.sum().clamp(min=1)
    acc_all = correct.sum(1) / n_tok.sum().clamp(min=1)

    # ---- example grid for the dashboard: the test's own tokens (plus the last
    # prefix token, whose prediction is the test's first token)
    example = None
    arrays = {"loss_by_category": loss_by_cat.float().numpy(), "top1_by_category": acc_by_cat.float().numpy()}
    if enc:
        ex = max(enc, key=lambda e: min(e.ids.shape[1] - e.n_prefix, EXAMPLE_MAX_TOKENS))
        first = ex.n_prefix - 1
        ids = ex.ids[:, :first + EXAMPLE_MAX_TOKENS].to(device)
        tops, top_ids, trues = [], [], []
        for _, logits in row_logits(residuals(ids)):
            probs = logits[:, first:-1].softmax(-1)          # [g, T', V]
            pr, i = probs.max(-1)
            tops.append(pr)
            top_ids.append(i)
            trues.append(probs.gather(-1, ids[0, first + 1:].expand(probs.shape[0], -1).unsqueeze(-1)).squeeze(-1))
        top_p, top_i, true_p = torch.cat(tops), torch.cat(top_ids), torch.cat(trues)
        example = {
            "case_id": ex.case.id,
            "tokens": [tokenizer.decode([i]) for i in ids[0, first:].tolist()],
            "top_tokens": [[tokenizer.decode([i]) for i in row] for row in top_i.tolist()],
        }
        arrays["example_top_p"] = top_p.cpu().numpy()
        arrays["example_true_p"] = true_p.cpu().numpy()

    summary = {
        "view": "logit_lens",
        "rows": rows,
        "categories": cats,
        "n_tokens": int(n_tok.sum()),
        "uniform_loss": float(torch.log(torch.tensor(float(V)))),
        "emb_top1_is_input": emb_is_input / max(1, int(n_tok.sum())),
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
