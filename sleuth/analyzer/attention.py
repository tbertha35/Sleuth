"""Analyzer view: what each attention head looks at.

Per head, averaged over every query position (except <|bos|> itself) in the suite:

  bos_share     attention paid to position 0. Many heads park attention on the
                first token when they have "nothing to do" (an *attention sink*).
  prev_token    attention to the immediately preceding token. "Previous-token
                heads" are a building block of induction (below).
  self          attention to the current token itself.
  entropy       how spread out the attention is, scaled to 0..1 (0 = all on one
                key, 1 = perfectly uniform over everything visible).

Induction (measured on synthetic input, not the suite):
  Feed [bos] + R random tokens + the same R tokens again. On the second copy
  a model can predict perfectly by finding where the current token appeared
  before and copying whatever came next. That's the *induction* algorithm,
  usually built from two heads: a previous-token head in an earlier layer and
  an induction head that uses it.

  induction_score  attention from a second-copy token to the position right
                   after its first occurrence. Induction heads score high (often > 0.5).
  duplicate_score  attention to the first occurrence itself.
  induction_gain   (model-level) loss on the first copy minus loss on the
                   second copy, in nats/token. Near 0 = the model can't copy;
                   large = it copies well. Induction heads tend to appear
                   suddenly during training (a "phase change"), visible in the timeline.

Random tokens are drawn from the ordinary (non-special) vocabulary with a fixed
seed, so every checkpoint sees the same sequences.

An example pattern (every layer and head, for one suite prompt) is saved for
the dashboard's pattern viewer.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..forge.model.taps import listen
from .common import encode_cases, head_labels, layer_of, series

EXAMPLE_MAX_TOKENS = 48


def induction_positions(R: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For [bos] + r_1..r_R + r_1..r_R: (queries, induction keys, duplicate keys).

    Query t is a token in the second copy. Its earlier occurrence is at
    t - R ("duplicate"), and the token that followed it is at t - R + 1,
    which is where an induction head looks.
    """
    t = torch.arange(1 + R, 1 + 2 * R)
    return t, t - R + 1, t - R


def _first_special(tokenizer) -> int:
    return min(tokenizer.special_tokens.values()) if tokenizer.special_tokens else tokenizer.vocab_size


@torch.no_grad()
def run(model, tokenizer, cases, device, induction_len: int = 40, induction_batch: int = 8, **_) -> dict:
    model.eval().to(device)
    L, H = model.cfg.n_layers, model.cfg.n_heads
    enc = encode_cases(tokenizer, cases, model.cfg.max_seq_len)

    # ---- per-head pattern statistics over the suite
    sums = {k: torch.zeros(L, H, dtype=torch.float64) for k in ("bos_share", "prev_token", "self", "entropy")}
    n_q = 0

    def on_pattern(name, p):
        li = layer_of(name)
        p = p[0].double().cpu()                    # [H, T, T]
        T = p.shape[-1]
        if T < 2:
            return
        q = p[:, 1:, :]                            # queries 1..T-1
        sums["bos_share"][li] += q[:, :, 0].sum(-1)
        sums["prev_token"][li] += p.diagonal(offset=-1, dim1=1, dim2=2).sum(-1)  # p[h, i+1, i]
        sums["self"][li] += p.diagonal(dim1=1, dim2=2)[:, 1:].sum(-1)
        # Query i can see i+1 keys; dividing by log(i+1) maps entropy to 0..1.
        ent = -(q * (q + 1e-12).log()).sum(-1)     # [H, T-1]
        visible = torch.arange(2, T + 1, dtype=torch.float64).log()
        sums["entropy"][li] += (ent / visible).sum(-1)

    with listen(model, {"blocks.*.attn.pattern": on_pattern}):
        for e in enc:
            model(e.ids.to(device))
            n_q += e.ids.shape[1] - 1
    stats = {k: (v / max(1, n_q)) for k, v in sums.items()}

    # ---- induction on repeated random tokens
    R = max(2, min(induction_len, (model.cfg.max_seq_len - 1) // 2))
    g = torch.Generator().manual_seed(1234)
    rand = torch.randint(0, _first_special(tokenizer), (induction_batch, R), generator=g)
    seq = torch.cat([torch.full((induction_batch, 1), tokenizer.bos_id), rand, rand], dim=1).to(device)
    t, k_ind, k_dup = induction_positions(R)
    ind = torch.zeros(L, H, dtype=torch.float64)
    dup = torch.zeros(L, H, dtype=torch.float64)

    def on_induction(name, p):
        li = layer_of(name)
        p = p.double().cpu()                       # [B, H, T, T]
        ind[li] = p[:, :, t, k_ind].mean(dim=(0, 2))
        dup[li] = p[:, :, t, k_dup].mean(dim=(0, 2))

    with listen(model, {"blocks.*.attn.pattern": on_induction}):
        logits = model(seq).logits[..., : tokenizer.vocab_size].float()
    nll = F.cross_entropy(logits[:, :-1].transpose(1, 2), seq[:, 1:], reduction="none")  # [B, 2R]
    # nll[:, i] predicts token i+1. First copy: tokens 2..R (token 1 is unguessable).
    # Second copy: tokens R+2..2R (token R+1 is guessable only by "the sequence restarts").
    first = nll[:, 1:R].mean().item()
    second = nll[:, R + 1:].mean().item()

    # ---- example pattern for the pattern viewer
    ex = max(enc, key=lambda e: min(e.ids.shape[1], EXAMPLE_MAX_TOKENS)) if enc else None
    example = None
    if ex is not None:
        ids = ex.ids[:, :EXAMPLE_MAX_TOKENS]
        pats = torch.zeros(L, H, ids.shape[1], ids.shape[1])

        def keep(name, p):
            pats[layer_of(name)] = p[0].float().cpu()

        with listen(model, {"blocks.*.attn.pattern": keep}):
            model(ids.to(device))
        example = {"case_id": ex.case.id, "tokens": [tokenizer.decode([i]) for i in ids[0].tolist()]}

    heads = []
    for li in range(L):
        for h in range(H):
            heads.append({"layer": li, "head": h, **{k: float(v[li, h]) for k, v in stats.items()},
                          "induction_score": float(ind[li, h]), "duplicate_score": float(dup[li, h])})
    top = sorted(heads, key=lambda x: -x["induction_score"])[:5]
    summary = {
        "view": "attention",
        "n_layers": L, "n_heads": H, "n_queries": n_q, "n_cases": len(enc),
        "induction": {"repeat_len": R, "batch": induction_batch, "loss_first_copy": first,
                      "loss_second_copy": second, "induction_gain": first - second,
                      "uniform_loss": math.log(_first_special(tokenizer))},
        "top_induction_heads": [f"L{x['layer']}H{x['head']} ({x['induction_score']:.2f})" for x in top],
        "heads": heads,
        "example": example,
    }
    arrays = {k: v.float().numpy() for k, v in stats.items()}
    arrays["induction_score"] = ind.float().numpy()
    arrays["duplicate_score"] = dup.float().numpy()
    if example is not None:
        arrays["example_pattern"] = pats.half().numpy()  # [L, H, T, T]
    return {"summary": summary, "arrays": arrays}


def timeline_series(summary: dict) -> dict:
    L, H = summary["n_layers"], summary["n_heads"]
    rows = head_labels(L, H)
    hs = summary["heads"]
    pick = lambda k: [h[k] for h in hs]  # noqa: E731
    return {
        "induction_score": series("Induction score", "Attention from a repeated token to whatever followed its "
                                  "first occurrence. Induction heads (copying machinery) score high.",
                                  rows, pick("induction_score"), "pct"),
        "induction_gain": series("Induction gain (nats/token)", "Loss on the first copy of a random sequence minus "
                                 "loss on its repeat. Rises when the model learns to copy.",
                                 ["model"], [summary["induction"]["induction_gain"]]),
        "prev_token": series("Previous-token attention", "Attention to the token just before. Previous-token "
                             "heads feed induction heads.", rows, pick("prev_token"), "pct"),
        "bos_share": series("Attention to <|bos|>", "Attention parked on the first token (an 'attention sink'): "
                            "often what a head does when it has nothing useful to attend to.",
                            rows, pick("bos_share"), "pct"),
        "entropy": series("Attention entropy (0-1)", "0 = each query looks at one token; 1 = attention spread "
                          "evenly over everything visible.", rows, pick("entropy"), "pct"),
    }
