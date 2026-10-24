"""Analyzer view: ablation. Remove a part of the model and measure the damage.

Components:
  head  L,h   one attention head's output (via the `blocks.L.attn.z` tap)
  attn  L     a whole layer's attention output (`blocks.L.attn_out`)
  mlp   L     a whole layer's MLP output (`blocks.L.mlp_out`)

Two ways to "remove" a component:
  mean (default)  replace its output with its *average* output over the suite.
                  This keeps the residual stream at a normal size and only
                  removes the information that varies from token to token.
  zero            replace it with zeros. Simpler, but pushes the model into
                  states it never saw in training, which can exaggerate the
                  damage.

The <|bos|> position (or a chat prefix, see common.py) is left untouched: models often use it as an "attention
sink", and changing it would measure a different effect.

For each ablation we rerun the whole suite (teacher-forced, see common.py) and
report the change versus the intact model, per category:
  Δ loss            change in mean next-token loss (nats). Bigger = more important.
  Δ expected loss   change in loss on the expected answers only
  Δ exact match     change in the share of answers predicted exactly

Cost: (n_layers × n_heads + 2 × n_layers) suite runs per checkpoint. For a 6×6
model that's 48 runs, quick on a GPU.
"""

from __future__ import annotations

import numpy as np
import torch

from ..forge.model.taps import listen
from .common import categories, encode_cases, head_labels, score_cases, series


def _components(L: int, H: int) -> list[tuple[str, int, int]]:
    comps = [("head", l, h) for l in range(L) for h in range(H)]
    comps += [("attn", l, -1) for l in range(L)] + [("mlp", l, -1) for l in range(L)]
    return comps


def _label(kind: str, l: int, h: int) -> str:
    return f"L{l}H{h}" if kind == "head" else f"L{l} {kind}"


@torch.no_grad()
def run(model, tokenizer, cases, device, method: str = "mean", prefix: list[int] | None = None, **_) -> dict:
    if method not in ("mean", "zero"):
        raise ValueError("ablation method must be 'mean' or 'zero'")
    model.eval().to(device)
    L, H = model.cfg.n_layers, model.cfg.n_heads
    V = tokenizer.vocab_size
    enc = encode_cases(tokenizer, cases, model.cfg.max_seq_len, prefix)
    skip = {"n": 1}  # positions left untouched: <|bos|>, or the whole chat prefix
    cats = categories(enc)

    # ---- average outputs over the suite (needed for mean ablation)
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}

    def accumulate(name, t):
        x = t[0, skip["n"]:].double()             # skip <|bos|> / the prefix
        sums[name] = sums.get(name, 0) + x.sum(0)
        counts[name] = counts.get(name, 0) + x.shape[0]

    if method == "mean":
        with listen(model, {"blocks.*.attn.z": accumulate, "blocks.*.attn_out": accumulate,
                            "blocks.*.mlp_out": accumulate}):
            for e in enc:
                skip["n"] = e.n_prefix
                model(e.ids.to(device))
    means = {k: (v / max(1, counts[k])) for k, v in sums.items()}

    def replacement(name: str, t: torch.Tensor, head: int | None) -> torch.Tensor:
        t = t.clone()
        if head is None:
            t[:, skip["n"]:] = means[name].to(t.device, t.dtype) if method == "mean" else 0
        else:
            t[:, skip["n"]:, head] = means[name][head].to(t.device, t.dtype) if method == "mean" else 0
        return t

    skip["n"] = enc[0].n_prefix if enc else 1  # the same for every test
    baseline = score_cases(model, enc, V, device)
    comps = _components(L, H)
    keys = cats + ["overall"]
    d_loss = np.zeros((len(comps), len(keys)), dtype=np.float32)
    d_exp = np.full((len(comps), len(keys)), np.nan, dtype=np.float32)
    d_exact = np.full((len(comps), len(keys)), np.nan, dtype=np.float32)

    for i, (kind, l, h) in enumerate(comps):
        if kind == "head":
            tap, fn = f"blocks.{l}.attn.z", (lambda name, t, h=h: replacement(name, t, h))
        else:
            tap = f"blocks.{l}.{kind}_out"
            fn = lambda name, t: replacement(name, t, None)  # noqa: E731
        with listen(model, {tap: fn}):
            s = score_cases(model, enc, V, device)
        for j, k in enumerate(keys):
            b, a = baseline[k], s[k]
            d_loss[i, j] = a["loss"] - b["loss"]
            if b["expected_loss"] is not None:
                d_exp[i, j] = a["expected_loss"] - b["expected_loss"]
                d_exact[i, j] = a["exact_match"] - b["exact_match"]

    labels = [_label(*c) for c in comps]
    overall = d_loss[:, -1]
    ranked = sorted(range(len(comps)), key=lambda i: -overall[i])[:10]
    summary = {
        "view": "ablation",
        "method": method,
        "n_layers": L, "n_heads": H,
        "categories": cats,
        "baseline": baseline,
        "components": labels,
        "most_important": [{"component": labels[i], "delta_loss": float(overall[i])} for i in ranked],
        "head_delta_loss": [float(x) for x in overall[: L * H]],
        "attn_delta_loss": [float(x) for x in overall[L * H: L * H + L]],
        "mlp_delta_loss": [float(x) for x in overall[L * H + L:]],
    }
    arrays = {"delta_loss": d_loss, "delta_expected_loss": d_exp, "delta_exact": d_exact}
    return {"summary": summary, "arrays": arrays}


def timeline_series(summary: dict) -> dict:
    L, H = summary["n_layers"], summary["n_heads"]
    layers = [f"L{l}" for l in range(L)]
    m = summary["method"]
    return {
        "mlp_delta_loss": series(f"Δ loss when a layer's MLP is {m}-ablated", "How much worse the model gets "
                                 "without each layer's MLP. Bigger = that MLP matters more.",
                                 layers, summary["mlp_delta_loss"]),
        "attn_delta_loss": series(f"Δ loss when a layer's attention is {m}-ablated", "How much worse the model "
                                  "gets without each layer's attention.", layers, summary["attn_delta_loss"]),
        "head_delta_loss": series(f"Δ loss per {m}-ablated head", "How much worse the model gets without each "
                                  "single attention head.", head_labels(L, H), summary["head_delta_loss"]),
    }
