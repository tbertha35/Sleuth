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

Which random tokens? By default they're drawn (without repeats inside a
sequence) from the tokens that occur in the suite, i.e. tokens the model has
certainly seen in training. A small model's vocabulary is full of tokens it
rarely or never saw (a TinyStories tokenizer has all 256 raw bytes, Chinese
fragments, ...); their embeddings are barely trained, so the model can't tell
two of them apart, let alone copy them. The older measurement, over the whole
ordinary vocabulary, is kept as `induction_gain_any_token` for comparison.
A fixed seed means every checkpoint sees the same sequences.

Induction on repeated *text* (`induction.text`, `text_induction_score`): each
test's own text, fed twice. Random sequences are unlike anything in training,
and a small model may copy names in a story without copying arbitrary token
strings, so this measures copying where the model is at home. The first copy is
scored like ordinary text; the gain is how much easier the repeat is.

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


def induction_positions(R: int, start: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For `start` tokens (e.g. [bos]) + r_1..r_R + r_1..r_R: (queries, induction keys, duplicate keys).

    Query t is a token in the second copy. Its earlier occurrence is at
    t - R ("duplicate"), and the token that followed it is at t - R + 1,
    which is where an induction head looks.
    """
    t = torch.arange(start + R, start + 2 * R)
    return t, t - R + 1, t - R


def _ordinary_ids(tokenizer) -> torch.Tensor:
    """Every token ID that isn't a special/added token. (Specials can sit
    anywhere: at the end for Qwen, at IDs 0-4 for Gemma.)"""
    special = set(getattr(tokenizer, "special_tokens", {}).values())
    special |= set(getattr(tokenizer, "added_tokens", {}).values())
    keep = torch.ones(tokenizer.vocab_size, dtype=torch.bool)
    keep[[i for i in special if i < tokenizer.vocab_size]] = False
    return keep.nonzero().squeeze(1)


def _repeated_random(tokenizer, pool: torch.Tensor, R: int, batch: int) -> torch.Tensor:
    """[batch, 1 + 2R]: start token + R distinct random tokens from `pool` + the same R again."""
    g = torch.Generator().manual_seed(1234)
    if len(pool) >= R:
        rand = torch.stack([pool[torch.randperm(len(pool), generator=g)[:R]] for _ in range(batch)])
    else:
        rand = pool[torch.randint(0, len(pool), (batch, R), generator=g)]
    return torch.cat([torch.full((batch, 1), tokenizer.start_id), rand, rand], dim=1)


def _induction_scores(model, seq, R, L, H, vocab, patterns: bool = True):
    """Induction and duplicate attention per head [L, H], plus the loss on the
    first and the second copy (nats/token)."""
    t, k_ind, k_dup = induction_positions(R)
    ind = torch.zeros(L, H, dtype=torch.float64)
    dup = torch.zeros(L, H, dtype=torch.float64)

    def on_induction(name, p):
        li = layer_of(name)
        p = p.double().cpu()                       # [B, H, T, T]
        ind[li] = p[:, :, t, k_ind].mean(dim=(0, 2))
        dup[li] = p[:, :, t, k_dup].mean(dim=(0, 2))

    with listen(model, {"blocks.*.attn.pattern": on_induction} if patterns else {}):
        logits = model(seq).logits[..., :vocab].float()
    nll = F.cross_entropy(logits[:, :-1].transpose(1, 2), seq[:, 1:], reduction="none")  # [B, 2R]
    # nll[:, i] predicts token i+1. First copy: tokens 2..R (token 1 is unguessable).
    # Second copy: tokens R+2..2R (token R+1 is guessable only by "the sequence restarts").
    return ind, dup, nll[:, 1:R].mean().item(), nll[:, R + 1:].mean().item()


def _text_repeat(model, enc, L, H, vocab, device) -> dict:
    """Induction on natural text: feed each test's own text twice (after its
    prefix) and compare the loss on the two copies. Unlike random tokens, the
    first copy is ordinary in-distribution text, so this asks "does the model
    copy what it has just read?" in the setting it was trained on."""
    max_len = model.cfg.max_seq_len
    ind = torch.zeros(L, H, dtype=torch.float64)
    n_q = 0
    nll_first, nll_second, n_pred, n_cases = 0.0, 0.0, 0, 0
    pos = {}

    def on_pattern(name, p):
        t, k_ind, _ = pos["t"]
        ind[layer_of(name)] += p[0][:, t, k_ind].double().sum(-1).cpu()

    for e in enc:
        text = e.ids[0, e.n_prefix:]
        R = min(len(text), (max_len - e.n_prefix) // 2)
        if R < 4:
            continue
        text = text[:R]
        seq = torch.cat([e.ids[0, :e.n_prefix], text, text])[None].to(device)
        pos["t"] = induction_positions(R, e.n_prefix)
        with listen(model, {"blocks.*.attn.pattern": on_pattern}):
            logits = model(seq).logits[0, :-1, :vocab].float()
        nll = F.cross_entropy(logits, seq[0, 1:], reduction="none")   # nll[i] predicts token i+1
        s = e.n_prefix
        nll_first += float(nll[s:s + R - 1].sum())                     # tokens 2..R of the first copy
        nll_second += float(nll[s + R:s + 2 * R - 1].sum())            # tokens 2..R of the repeat
        n_pred += R - 1
        n_q += R
        n_cases += 1
    if not n_cases:
        return {"ind": ind, "first": float("nan"), "second": float("nan"), "n_cases": 0}
    return {"ind": ind / n_q, "first": nll_first / n_pred, "second": nll_second / n_pred, "n_cases": n_cases}


@torch.no_grad()
def run(model, tokenizer, cases, device, induction_len: int = 40, induction_batch: int = 8,
        prefix: list[int] | None = None, **_) -> dict:
    model.eval().to(device)
    L, H = model.cfg.n_layers, model.cfg.n_heads
    enc = encode_cases(tokenizer, cases, model.cfg.max_seq_len, prefix)
    start = {"q": 1}  # first query position counted: after <|bos|>, or after the chat prefix

    # ---- per-head pattern statistics over the suite
    sums = {k: torch.zeros(L, H, dtype=torch.float64) for k in ("bos_share", "prev_token", "self", "entropy")}
    n_q = 0

    def on_pattern(name, p):
        li = layer_of(name)
        p = p[0].double().cpu()                    # [H, T, T]
        T, s0 = p.shape[-1], start["q"]
        if T <= s0:
            return
        q = p[:, s0:, :]                           # queries s0..T-1
        sums["bos_share"][li] += q[:, :, 0].sum(-1)
        sums["prev_token"][li] += p.diagonal(offset=-1, dim1=1, dim2=2)[:, s0 - 1:].sum(-1)  # p[h, i+1, i]
        sums["self"][li] += p.diagonal(dim1=1, dim2=2)[:, s0:].sum(-1)
        # Query i can see i+1 keys; dividing by log(i+1) maps entropy to 0..1.
        ent = -(q * (q + 1e-12).log()).sum(-1)     # [H, T-s0]
        visible = torch.arange(s0 + 1, T + 1, dtype=torch.float64).log()
        sums["entropy"][li] += (ent / visible).sum(-1)

    with listen(model, {"blocks.*.attn.pattern": on_pattern}):
        for e in enc:
            start["q"] = e.n_prefix
            model(e.ids.to(device))
            n_q += max(0, e.ids.shape[1] - e.n_prefix)
    stats = {k: (v / max(1, n_q)) for k, v in sums.items()}

    # ---- induction on repeated random tokens
    R = max(2, min(induction_len, (model.cfg.max_seq_len - 1) // 2))
    ordinary = _ordinary_ids(tokenizer)
    n_ordinary = len(ordinary)
    is_ordinary = set(ordinary.tolist())
    pool = sorted({i for e in enc for i in e.ids[0, e.n_prefix:].tolist()} & is_ordinary)
    pool_name = "suite tokens"
    if len(pool) < 2 * R:  # too few distinct tokens: fall back to the whole vocabulary
        pool, pool_name = ordinary.tolist(), "whole vocabulary"
    seq = _repeated_random(tokenizer, torch.tensor(pool), R, induction_batch).to(device)
    ind, dup, first, second = _induction_scores(model, seq, R, L, H, tokenizer.vocab_size)
    any_seq = _repeated_random(tokenizer, ordinary, R, induction_batch).to(device)
    _, _, any_first, any_second = _induction_scores(model, any_seq, R, L, H, tokenizer.vocab_size, patterns=False)
    text = _text_repeat(model, enc, L, H, tokenizer.vocab_size, device)

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
        "induction": {"repeat_len": R, "batch": induction_batch, "token_pool": pool_name, "pool_size": len(pool),
                      "loss_first_copy": first, "loss_second_copy": second, "induction_gain": first - second,
                      "uniform_loss": math.log(len(pool)),
                      "any_token": {"loss_first_copy": any_first, "loss_second_copy": any_second,
                                    "uniform_loss": math.log(n_ordinary)},
                      "induction_gain_any_token": any_first - any_second,
                      "text": {"loss_first_copy": text["first"], "loss_second_copy": text["second"],
                               "induction_gain": text["first"] - text["second"], "n_cases": text["n_cases"]}},
        "top_text_induction_heads": [f"L{li}H{h} ({float(text['ind'][li, h]):.2f})" for li, h in
                                     sorted(((li, h) for li in range(L) for h in range(H)),
                                            key=lambda x: -float(text["ind"][x]))[:5]],
        "top_induction_heads": [f"L{x['layer']}H{x['head']} ({x['induction_score']:.2f})" for x in top],
        "heads": heads,
        "example": example,
    }
    arrays = {k: v.float().numpy() for k, v in stats.items()}
    arrays["induction_score"] = ind.float().numpy()
    arrays["duplicate_score"] = dup.float().numpy()
    arrays["text_induction_score"] = text["ind"].float().numpy()
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
        "induction_gain": series("Induction gain (nats/token)", "Loss on the first copy of a random sequence of "
                                 "familiar tokens minus loss on its repeat. Rises when the model learns to copy.",
                                 ["model"], [summary["induction"]["induction_gain"]]),
        "text_induction_gain": series("Induction gain on repeated text (nats/token)", "Each test's own text, "
                                      "fed twice: loss on the first copy minus loss on the repeat.",
                                      ["model"], [summary["induction"].get("text", {}).get("induction_gain",
                                                                                           float("nan"))]),
        "induction_gain_any_token": series("Induction gain, any token (nats/token)", "The same, with random tokens "
                                           "from the whole vocabulary, including ones the model rarely or never saw.",
                                           ["model"], [summary["induction"].get("induction_gain_any_token", float("nan"))]),
        "prev_token": series("Previous-token attention", "Attention to the token just before. Previous-token "
                             "heads feed induction heads.", rows, pick("prev_token"), "pct"),
        "bos_share": series("Attention to <|bos|>", "Attention parked on the first token (an 'attention sink'): "
                            "often what a head does when it has nothing useful to attend to.",
                            rows, pick("bos_share"), "pct"),
        "entropy": series("Attention entropy (0-1)", "0 = each query looks at one token; 1 = attention spread "
                          "evenly over everything visible.", rows, pick("entropy"), "pct"),
    }
