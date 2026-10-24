"""Analyzer view: neuron and layer activation statistics.

For one checkpoint, run every test in a suite through the model and measure,
per layer and per MLP neuron:

  * how strongly neurons respond (mean |activation|, max |activation|)
  * how often each neuron "fires", overall and per test category
  * how big each layer's contributions to the residual stream are
    (attention output, MLP output, and the stream itself)

What does "fires" mean for SwiGLU? ReLU neurons are exactly zero or not,
so "fired" is obvious. SwiGLU's silu(gate)·up is smooth and almost never
exactly zero, so we need a threshold. We use a *relative* one:

    neuron fires on a token  <=>  |activation| > k × (that layer's RMS activation)

with k = 1 by default, i.e. "noticeably above this layer's typical size".
A relative threshold keeps the numbers comparable across layers and across
checkpoints, even though raw activation scales drift during training. It
needs two passes over the suite (first measure each layer's RMS, then
count), which is cheap for suite-sized inputs.

Derived per-layer summaries (all heuristics, defined here in one place):
  * fire_rate_mean:     average fraction of tokens a neuron fires on
  * frac_low_activity:  neurons that fire on < 1% of tokens ("nearly dead")
  * frac_high_activity: neurons that fire on > 50% of tokens ("always on")
  * frac_selective:     neurons whose firing concentrates in one category:
                        selectivity = (r_top - mean of other categories' r) / r_top > 0.5,
                        and r_top > 5%. Watch this rise over training as
                        neurons specialize (e.g. code-only neurons).

The <|bos|> position is excluded: models use the first token as an
"attention sink" with huge, unrepresentative activations.
"""

from __future__ import annotations

import numpy as np
import torch

from ..forge.model.taps import listen
from .common import encode_cases, layer_of, series
from .suites import TestCase

LOW, HIGH, SELECTIVE_MIN_RATE, SELECTIVITY = 0.01, 0.5, 0.05, 0.5


@torch.no_grad()
def neuron_stats(model, tokenizer, cases: list[TestCase], k: float = 1.0, device: str | torch.device = "cpu",
                 prefix: list[int] | None = None) -> dict:
    """Returns {"summary": json-able dict, "arrays": dict of numpy arrays}."""
    model.eval().to(device)
    L = model.cfg.n_layers
    H = model.cfg.hidden
    cats = sorted({c.category for c in cases})
    inputs = [(e.category, e.ids, e.n_prefix) for e in encode_cases(tokenizer, cases, model.cfg.max_seq_len, prefix)]
    skip = {"n": 1}  # positions to leave out: <|bos|> (an attention sink), or a whole chat prefix

    # ---- pass 1: per-layer RMS of the MLP activations, plus residual-stream sizes
    sq_sum = torch.zeros(L, dtype=torch.float64)
    n_tok = 0
    comp = {name: torch.zeros(L, dtype=torch.float64) for name in ("attn_out", "mlp_out", "resid_post")}

    def p1_act(name, t):
        sq_sum[layer_of(name)] += t[0, skip["n"]:].double().pow(2).mean(-1).sum().cpu()

    def p1_comp(name, t):
        comp[name.split(".")[-1]][layer_of(name)] += t[0, skip["n"]:].double().pow(2).mean(-1).sum().cpu()

    with listen(model, {"blocks.*.mlp.act": p1_act, "blocks.*.attn_out": p1_comp,
                        "blocks.*.mlp_out": p1_comp, "blocks.*.resid_post": p1_comp}):
        for _, ids, n_prefix in inputs:
            skip["n"] = n_prefix
            model(ids.to(device))
            n_tok += ids.shape[1] - n_prefix
    if n_tok == 0:
        raise ValueError("suite produced no tokens")
    rms = (sq_sum / n_tok).sqrt()

    # ---- pass 2: per-neuron counts, per category
    ci = {c: i for i, c in enumerate(cats)}
    fire = torch.zeros(L, len(cats), H, dtype=torch.float64)
    abs_sum = torch.zeros(L, H, dtype=torch.float64)
    abs_max = torch.zeros(L, H, dtype=torch.float64)
    tok_per_cat = torch.zeros(len(cats), dtype=torch.float64)
    current = {"cat": 0}

    def p2_act(name, t):
        li = layer_of(name)
        a = t[0, skip["n"]:].double().abs().cpu()   # [T, H]
        fire[li, current["cat"]] += (a > k * rms[li]).sum(0)
        abs_sum[li] += a.sum(0)
        abs_max[li] = torch.maximum(abs_max[li], a.max(0).values)

    with listen(model, {"blocks.*.mlp.act": p2_act}):
        for cat, ids, n_prefix in inputs:
            current["cat"] = ci[cat]
            skip["n"] = n_prefix
            tok_per_cat[ci[cat]] += ids.shape[1] - n_prefix
            model(ids.to(device))

    rate_by_cat = fire / tok_per_cat.clamp(min=1)[None, :, None]   # [L, C, H]
    rate = fire.sum(1) / n_tok                                     # [L, H]
    mean_abs = abs_sum / n_tok

    top = rate_by_cat.max(1).values                                # [L, H]
    if len(cats) > 1:
        others = (rate_by_cat.sum(1) - top) / (len(cats) - 1)
        selectivity = (top - others) / top.clamp(min=1e-12)
        selective = (selectivity > SELECTIVITY) & (top > SELECTIVE_MIN_RATE)
        preferred = rate_by_cat.argmax(1)
    else:
        selective = torch.zeros_like(top, dtype=torch.bool)
        preferred = torch.zeros_like(top, dtype=torch.long)

    layers = []
    for li in range(L):
        layers.append({
            "layer": li,
            "act_rms": float(rms[li]),
            "act_mean_abs": float(mean_abs[li].mean()),
            "fire_rate_mean": float(rate[li].mean()),
            "frac_low_activity": float((rate[li] < LOW).double().mean()),
            "frac_high_activity": float((rate[li] > HIGH).double().mean()),
            "frac_selective": float(selective[li].double().mean()),
            "selective_by_category": {c: int((selective[li] & (preferred[li] == i)).sum()) for c, i in ci.items()},
            "fire_rate_by_category": {c: float(rate_by_cat[li, i].mean()) for c, i in ci.items()},
            "attn_out_rms": float((comp["attn_out"][li] / n_tok).sqrt()),
            "mlp_out_rms": float((comp["mlp_out"][li] / n_tok).sqrt()),
            "resid_rms": float((comp["resid_post"][li] / n_tok).sqrt()),
        })
    summary = {
        "view": "neuron_stats",
        "threshold_k": k,
        "categories": cats,
        "tokens_by_category": {c: int(tok_per_cat[i]) for c, i in ci.items()},
        "n_tokens": n_tok,
        "n_cases": len(cases),
        "n_layers": L,
        "n_neurons": H,
        "layers": layers,
    }
    arrays = {
        "fire_rate": rate.float().numpy(),                   # [L, H]
        "fire_rate_by_category": rate_by_cat.float().numpy(),  # [L, C, H]
        "mean_abs": mean_abs.float().numpy(),                # [L, H]
        "max_abs": abs_max.float().numpy(),                  # [L, H]
    }
    return {"summary": summary, "arrays": arrays}


def run(model, tokenizer, cases, device, k: float = 1.0, prefix: list[int] | None = None, **_) -> dict:
    """View interface (see analyzer/views.py)."""
    return neuron_stats(model, tokenizer, cases, k=k, device=device, prefix=prefix)


_HELP = {
    "frac_selective": ("Category-selective neurons", "Share of neurons whose firing concentrates in one test "
                       "category. Rising = neurons specializing.", "pct"),
    "fire_rate_mean": ("Mean firing rate", "Average fraction of tokens on which a neuron fires "
                       "(|activation| above its layer's RMS).", "pct"),
    "frac_low_activity": ("Nearly-dead neurons", "Share of neurons firing on fewer than 1% of tokens.", "pct"),
    "frac_high_activity": ("Always-on neurons", "Share of neurons firing on more than half of all tokens.", "pct"),
    "act_rms": ("MLP activation size (RMS)", "Typical magnitude of each layer's MLP hidden activations.", "num"),
    "mlp_out_rms": ("MLP output size", "How much each layer's MLP writes into the residual stream.", "num"),
    "attn_out_rms": ("Attention output size", "How much each layer's attention writes into the residual stream.",
                     "num"),
    "resid_rms": ("Residual stream size", "Magnitude of the residual stream after each layer.", "num"),
}


def timeline_series(summary: dict) -> dict:
    rows = [f"L{lay['layer']}" for lay in summary["layers"]]
    out = {k: series(label, help, rows, [lay[k] for lay in summary["layers"]], fmt)
           for k, (label, help, fmt) in _HELP.items()}
    for c in summary["categories"]:
        out[f"cat:{c}"] = series(f"Firing rate on “{c}” tests", "Average firing rate of each layer's neurons, on "
                                 "this category's tests only.", rows,
                                 [lay["fire_rate_by_category"][c] for lay in summary["layers"]], "pct")
    return out
