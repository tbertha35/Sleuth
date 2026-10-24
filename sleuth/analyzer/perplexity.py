"""Perplexity on a text file, computed the way llama.cpp's `llama-perplexity` does.

Matching its procedure exactly makes our numbers directly comparable with
llama.cpp's, which is how the owner checked our Gemma 4 adapter:

  1. tokenize the whole file, with <bos> in front (when the model uses one) and
     no special tokens parsed inside the text
  2. cut it into chunks of `ctx` tokens; replace each chunk's first token with
     <bos> (when the model uses one)
  3. run each chunk on its own, and score only the second half: predictions
     at positions ctx/2 .. ctx-2. (The first half gives every scored token at
     least ctx/2 tokens of context; early positions are always worse.)
  4. perplexity = exp(mean cross-entropy over the scored tokens)

Also reported:
  * bits per byte over the scored tokens: loss / ln 2 / bytes. Unlike
    perplexity, this doesn't depend on the tokenizer, so different models compare.
  * loss by position (1-7, 8-15, 16-31, ...), over *every* prediction, which shows
    how much a model needs context. (Gemma 4 IT on raw text: ~19 nats at
    positions 1-7, ~3 by 128-255.)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def position_buckets(n: int) -> list[tuple[int, int]]:
    """[(1, 7), (8, 15), (16, 31), ...] covering predicted positions 1..n-1."""
    out, lo = [(1, min(7, n - 1))], 8
    while lo <= n - 1:
        out.append((lo, min(2 * lo - 1, n - 1)))
        lo *= 2
    return out


@torch.no_grad()
def perplexity(model, tokenizer, text: str, ctx: int = 512, max_chunks: int | None = None,
               device="cpu", log=None) -> dict:
    model.eval()
    bos = getattr(tokenizer, "bos_id", None)
    ids = ([bos] if bos is not None else []) + tokenizer.encode(text)  # common_tokenize(..., add_special=true)
    if ctx > model.cfg.max_seq_len:
        raise ValueError(f"ctx {ctx} is longer than the model's max_seq_len {model.cfg.max_seq_len}")
    n_chunks = len(ids) // ctx
    if max_chunks:
        n_chunks = min(n_chunks, max_chunks)
    if n_chunks == 0:
        raise ValueError(f"the text has {len(ids)} tokens: not enough for one chunk of {ctx}")
    first = ctx // 2
    V = tokenizer.vocab_size
    byte_len = tokenizer.token_byte_lengths()
    buckets = position_buckets(ctx)
    b_sum = [0.0] * len(buckets)
    b_n = [0] * len(buckets)
    nll_sum, n_scored, n_bytes, chunk_ppl = 0.0, 0, 0, []
    for c in range(n_chunks):
        chunk = ids[c * ctx:(c + 1) * ctx]
        if bos is not None:
            chunk = [bos] + chunk[1:]                    # as llama-perplexity: first token -> <bos>
        x = torch.tensor([chunk], device=device)
        logits = model(x).logits[0, :-1, :V].float()
        nll = F.cross_entropy(logits, x[0, 1:], reduction="none").double().cpu()   # nll[p] predicts token p+1
        scored = nll[first:]                             # predictions at positions first..ctx-2
        nll_sum += float(scored.sum())
        n_scored += scored.numel()
        n_bytes += sum(byte_len[t] for t in chunk[first + 1:])
        for i, (lo, hi) in enumerate(buckets):           # predicted positions lo..hi
            b_sum[i] += float(nll[lo - 1:hi].sum())
            b_n[i] += hi - lo + 1
        chunk_ppl.append(math.exp(nll_sum / n_scored))   # running value, like llama.cpp prints
        if log:
            log(f"[{c + 1}] {chunk_ppl[-1]:.4f}")
    loss = nll_sum / n_scored
    return {
        "tokens_in_text": len(ids), "ctx": ctx, "chunks": n_chunks, "scored_tokens": n_scored,
        "loss": loss, "perplexity": math.exp(loss),
        "bits_per_byte": nll_sum / math.log(2) / max(1, n_bytes),
        "running_perplexity": chunk_ppl,
        "by_position": [{"positions": f"{lo}-{hi}", "loss": b_sum[i] / max(1, b_n[i]), "n": b_n[i]}
                        for i, (lo, hi) in enumerate(buckets)],
    }
