"""Speculative decoding: guess several tokens, check them all in one forward pass.

Decoding a big model one token at a time is limited by reading its weights:
every step streams all of them from memory for one token's worth of math. A
forward pass over 5 tokens reads the same weights once, so it costs about as
much as a pass over 1. Speculative decoding uses that:

  1. a cheap *drafter* guesses the next k tokens;
  2. the model runs once on [last token, guess 1, ..., guess k] and so gets its
     own next-token distribution after each of them;
  3. guesses are accepted from the left while they agree with the model; at the
     first disagreement the model's own token is taken instead. If all k agree,
     the model's prediction after the last guess comes for free.

Every forward pass therefore yields between 1 and k+1 tokens, and the text is
*exactly* what the model would have produced on its own:
  * greedy: a guess is accepted only if it is the model's argmax;
  * sampling: a guess x is accepted with probability p(x) (the model's own,
    after the same penalties and filters as sample_next); on rejection the
    token is sampled from p with x removed. For a drafter that proposes one
    token with certainty, this is the standard speculative sampling rule
    (Leviathan et al. 2023; Chen et al. 2023), so the output distribution is
    unchanged.

The KV cache gets entries for all k guesses; rejected ones are dropped by
rewinding `cache.pos` (later writes overwrite them, and nothing reads past pos).

Drafters here:
  * prompt lookup (`ngram_draft`): find the last time the most recent few
    tokens appeared earlier in the text and propose what followed. Free, and
    strong whenever text gets repeated (code edits, quoting, summaries, lists).
More drafters (a small draft model, Gemma's MTP layers, trained heads) plug into
the same `draft(ids, k)` interface.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Callable

import torch

from .generate import GenerationResult, _sync
from .kv_cache import KVCache
from .sampling import NgramBlocker, SamplingConfig, final_logits

Drafter = Callable[[list[int], int], list[int]]


def ngram_draft(max_ngram: int = 3, min_ngram: int = 1) -> Drafter:
    """Prompt-lookup drafter: the tokens that followed the most recent earlier
    occurrence of the text's last n tokens (longest n first)."""

    def draft(ids: list[int], k: int) -> list[int]:
        for n in range(min(max_ngram, len(ids) - 1), min_ngram - 1, -1):
            tail = ids[-n:]
            # Search backwards, skipping the tail itself.
            for start in range(len(ids) - n - 1, -1, -1):
                if ids[start:start + n] == tail:
                    follow = ids[start + n:start + n + k]
                    if follow:
                        return follow
                    break
        return []

    return draft


@dataclass
class SpecStats:
    forwards: int = 0       # model passes after the prefill
    drafted: int = 0        # guesses proposed
    accepted: int = 0       # guesses accepted

    def as_dict(self, new_tokens: int) -> dict:
        return {"forwards": self.forwards, "drafted": self.drafted, "accepted": self.accepted,
                "acceptance": round(self.accepted / self.drafted, 3) if self.drafted else None,
                "tokens_per_forward": round(max(0, new_tokens - 1) / self.forwards, 2) if self.forwards else None}


@torch.no_grad()
def speculative_generate(model, prompt_ids: list[int], max_new_tokens: int = 200,
                         sampling: SamplingConfig | None = None, draft: Drafter | None = None, k: int = 3,
                         stop_ids: set[int] | None = None, vocab_limit: int | None = None,
                         seed: int | None = None, on_token: Callable[[int], None] | None = None,
                         cuda_graph: bool = False) -> GenerationResult:
    """Like generate(), with up to k drafted tokens checked per forward pass.
    stats: forwards, drafted, accepted, acceptance, tokens_per_forward.

    "Exactly the same text" holds as long as the check runs each row the way a
    normal decoding step does. For quantized layers that means at most 4 rows
    (triton_kernels.GEMV_MAX_ROWS), hence k = 3 by default: with k = 4 the 5-row
    check uses the matmul kernel, which rounds differently in bf16 and can flip a
    near-tie (Gemma 4 12B: ' interior' vs ' air', 0.125 logits apart).

    cuda_graph: run each check as one CUDA-graph replay (cuda_graph.py). The graph
    always takes k+1 tokens, so fewer guesses are padded; the padding's logits are
    ignored and its cache entries are overwritten later."""
    sampling = sampling or SamplingConfig()
    draft = draft or ngram_draft()
    # Drafters that read the target's hidden states (MTP, mtp.py) get them after
    # the prefill (`begin`) and after every check (`after_verify`).
    wants_h = bool(getattr(draft, "needs_hidden", False))
    stop_ids = stop_ids or set()
    if not prompt_ids:
        raise ValueError("prompt must contain at least one token (e.g. <|bos|>)")
    p = next(model.parameters())
    device = p.device
    max_len = model.cfg.max_seq_len
    if len(prompt_ids) >= max_len:
        prompt_ids = prompt_ids[-(max_len - 1):]
    gen = torch.Generator(device=device)
    gen.manual_seed(seed) if seed is not None else gen.seed()
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda" and p.dtype == torch.float32 else contextlib.nullcontext())
    decoder = None
    if cuda_graph:
        from .cuda_graph import graph_unsupported

        if why := graph_unsupported(model):
            raise NotImplementedError(why)
        if device.type != "cuda":
            raise ValueError("cuda_graph needs a model on a CUDA device")
        from .cuda_graph import decoder_for

        decoder = decoder_for(model, max_len, autocast, tokens_per_step=k + 1, with_hidden=wants_h)
        cache = decoder.cache
    elif hasattr(model, "new_cache"):  # models with their own cache (Qwen3.5: KV + recurrent states)
        cache = model.new_cache(1, max_len, torch.bfloat16 if device.type == "cuda" else p.dtype)
    else:
        cache = KVCache.for_model(model, batch=1, max_len=max_len,
                                  dtype=torch.bfloat16 if device.type == "cuda" else p.dtype)
    model.eval()
    # Recurrent layers can't simply rewind: the cache records its state after each
    # checked token, and rollback() keeps the one after the last accepted token.
    rewind = getattr(cache, "rollback", None)

    def keep(start: int, n_ok: int) -> None:
        if rewind is not None:
            rewind(start, 1 + n_ok)
        else:
            cache.pos = start + 1 + n_ok
    greedy = sampling.temperature <= 0
    plain_greedy = greedy and sampling.repetition_penalty == 1.0 and sampling.no_repeat_ngram <= 0
    blocker = NgramBlocker(sampling.no_repeat_ngram, prompt_ids) if sampling.no_repeat_ngram > 0 else None
    ids = list(prompt_ids)     # everything so far (prompt + accepted tokens)
    new: list[int] = []
    stats = SpecStats()

    def dist(logits_row: torch.Tensor) -> torch.Tensor:
        """The model's final logits for the next token after `ids`, as sample_next sees them."""
        row = logits_row[None, :vocab_limit] if vocab_limit is not None else logits_row[None]
        ctx = torch.tensor([ids], device=device)
        return final_logits(row, sampling, ctx, blocker.banned() if blocker else None)[0]

    def choose(fl: torch.Tensor, reject: int | None = None) -> int:
        if greedy:
            return int(fl.argmax())
        probs = fl.softmax(-1)
        if reject is not None:  # the residual distribution after rejecting `reject`
            probs = probs.clone()
            probs[reject] = 0
            if float(probs.sum()) <= 0:
                return reject
        return int(torch.multinomial(probs, 1, generator=gen))

    def emit(tok: int) -> bool:
        """Record a new token; True when generation must stop."""
        ids.append(tok)
        new.append(tok)
        if blocker:
            blocker.add(tok)
        if on_token:
            on_token(tok)
        return tok in stop_ids or len(new) >= max_new_tokens

    _sync(device)
    t0 = time.perf_counter()
    with autocast:
        pre = model(torch.tensor([prompt_ids], device=device), cache=cache, return_hidden=wants_h)
    logits = pre.logits[0, -1]
    if wants_h:
        draft.begin(model, cache, pre.hidden[0, -1], prompt_ids=list(prompt_ids), hidden_rows=pre.hidden[0])
    stop_reason = "max_new_tokens"
    done = emit(choose(dist(logits)))
    if new[-1] in stop_ids:
        stop_reason = "stop_token"
    t1 = time.perf_counter()
    while not done:
        pending = new[-1]  # produced, but its K/V aren't in the cache yet
        room = min(k, max_new_tokens - len(new), max_len - cache.pos - 1)
        if room < 0:
            stop_reason = "context_full"
            break
        guess = draft(ids, room)[:room] if room > 0 else []
        start = cache.pos
        hid = None
        if decoder is not None and start + k + 1 <= max_len:
            padded = [pending] + guess + [pending] * (k - len(guess))
            out = decoder.verify(torch.tensor(padded, device=device))
            hid = decoder.hidden
        else:
            if rewind is not None:
                cache.record = True
            with autocast:
                res_ = model(torch.tensor([[pending] + guess], device=device), cache=cache, return_hidden=wants_h)
            if rewind is not None:
                cache.record = False
            out, hid = res_.logits[0], (res_.hidden[0] if wants_h else None)
            if decoder is not None:
                cache.pos = start  # the normal path advanced it; the bookkeeping below sets it
        stats.forwards += 1
        stats.drafted += len(guess)
        n_ok = 0
        if plain_greedy:
            # No penalties: the model's choice at each position doesn't depend on the
            # context bookkeeping, so take all argmaxes at once (one GPU sync, not k).
            rows = out[:len(guess) + 1]
            picks = (rows[:, :vocab_limit] if vocab_limit is not None else rows).argmax(-1).tolist()
            for i, g in enumerate(guess):
                if picks[i] != g:
                    done = emit(picks[i])
                    break
                n_ok += 1
                stats.accepted += 1
                if emit(g):
                    done = True
                    break
            else:
                if not done:
                    done = emit(picks[len(guess)])
            keep(start, n_ok)
            if wants_h:  # row n_ok: the last input the target kept
                draft.after_verify(hid[n_ok].clone(), hidden_rows=hid, inputs=[pending] + guess, n_ok=n_ok)
            if new[-1] in stop_ids:
                stop_reason = "stop_token"
            if not done and cache.pos >= max_len:
                stop_reason = "context_full"
                break
            continue
        for i, g in enumerate(guess):
            fl = dist(out[i])
            if greedy:
                ok = int(fl.argmax()) == g
            else:
                ok = float(torch.rand((), generator=gen, device=device)) < float(fl.softmax(-1)[g])
            if not ok:
                done = emit(choose(fl, reject=None if greedy else g))
                break
            n_ok += 1
            stats.accepted += 1
            if emit(g):
                done = True
                break
        else:
            if not done:  # every guess accepted: the model's next token after the last one is free
                done = emit(choose(dist(out[len(guess)])))
        keep(start, n_ok)  # keep [pending] + accepted guesses; drop the rest
        if wants_h:
            draft.after_verify(hid[n_ok].clone(), hidden_rows=hid, inputs=[pending] + guess, n_ok=n_ok)
        if new[-1] in stop_ids:
            stop_reason = "stop_token"
        if not done and cache.pos >= max_len:
            stop_reason = "context_full"
            break
    _sync(device)
    t2 = time.perf_counter()
    res = GenerationResult(list(prompt_ids), new, stop_reason, t1 - t0, t2 - t1,
                           {"kv_cache_mb": round(cache.nbytes() / 1e6, 2), "device": str(device)})
    res.stats["speculative"] = stats.as_dict(len(new))
    if decoder is not None:
        res.stats["cuda_graphs"] = sorted(decoder.graphs)
    return res
