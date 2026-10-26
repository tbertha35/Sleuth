"""Our own text-generation loop.

Two phases with very different performance:
  * prefill: run the whole prompt through the model in one batched forward
    pass, filling the KV cache. Compute-bound (big matmuls), so fast per token.
  * decode: one token at a time, each step reusing the cache. Memory-bound:
    every step reads all the weights for a single token's worth of math.
We time them separately, because they respond to different optimizations
(decode speed is what quantization and speculative decoding improve).
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

import torch

from .kv_cache import KVCache
from .sampling import NgramBlocker, SamplingConfig, sample_next


@dataclass
class GenerationResult:
    prompt_ids: list[int]
    new_ids: list[int]
    stop_reason: str                    # "stop_token" | "max_new_tokens" | "context_full"
    prefill_seconds: float
    decode_seconds: float
    stats: dict = field(default_factory=dict)

    @property
    def decode_tok_per_s(self) -> float:
        n = max(0, len(self.new_ids) - 1)  # the first new token comes from the prefill pass
        return n / self.decode_seconds if self.decode_seconds > 0 else 0.0

    @property
    def prefill_tok_per_s(self) -> float:
        return len(self.prompt_ids) / self.prefill_seconds if self.prefill_seconds > 0 else 0.0


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def generate(model, prompt_ids: list[int], max_new_tokens: int = 200, sampling: SamplingConfig | None = None,
             stop_ids: set[int] | None = None, vocab_limit: int | None = None, seed: int | None = None,
             on_token: Callable[[int], None] | None = None, cuda_graph: bool = False) -> GenerationResult:
    """Generate up to max_new_tokens after prompt_ids (batch size 1).

    vocab_limit: only sample from IDs < this. The model's output layer is
    padded past the tokenizer's vocab, and those padding IDs must never be
    sampled.
    on_token: called with each new token as it's produced (for streaming).
    cuda_graph: replay each decode step as one CUDA graph (cuda_graph.py): far
    fewer kernel launches, so much faster for big models on a GPU. Needs CUDA.
    """
    sampling = sampling or SamplingConfig()
    stop_ids = stop_ids or set()
    if not prompt_ids:
        raise ValueError("prompt must contain at least one token (e.g. <|bos|>)")
    p = next(model.parameters())
    device = p.device
    max_len = model.cfg.max_seq_len
    if len(prompt_ids) >= max_len:
        prompt_ids = prompt_ids[-(max_len - 1):]  # keep the most recent context
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    else:
        gen.seed()
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

        decoder = decoder_for(model, max_len, autocast)
        cache = decoder.cache
    elif hasattr(model, "new_cache"):  # models with their own cache (Qwen3.5: KV + recurrent states)
        cache = model.new_cache(1, max_len, torch.bfloat16 if device.type == "cuda" else p.dtype)
    else:
        cache = KVCache.for_model(model, batch=1, max_len=max_len,
                                  dtype=torch.bfloat16 if device.type == "cuda" else p.dtype)
    ctx = torch.tensor([prompt_ids], device=device)
    model.eval()

    _sync(device)
    t0 = time.perf_counter()
    with autocast:
        logits = model(ctx, cache=cache).logits[:, -1]
    blocker = NgramBlocker(sampling.no_repeat_ngram, prompt_ids) if sampling.no_repeat_ngram > 0 else None
    new: list[int] = []
    stop_reason = "max_new_tokens"
    t1 = t0
    for i in range(max_new_tokens):
        if vocab_limit is not None:
            logits = logits[..., :vocab_limit]
        nxt = sample_next(logits, sampling, ctx, gen, banned=blocker.banned() if blocker else None)
        tok = int(nxt.item())  # .item() syncs, so timing below is accurate
        if blocker:
            blocker.add(tok)
        if i == 0:
            t1 = time.perf_counter()  # prefill ends once the first token exists
        new.append(tok)
        if on_token:
            on_token(tok)
        if tok in stop_ids:
            stop_reason = "stop_token"
            break
        if cache.pos >= max_len:
            stop_reason = "context_full"
            break
        ctx = torch.cat([ctx, nxt.view(1, 1)], dim=1)
        if decoder is not None:
            logits = decoder.step(nxt)  # the graph's output buffer: used before the next step
        else:
            with autocast:
                logits = model(nxt.view(1, 1), cache=cache).logits[:, -1]
    _sync(device)
    t2 = time.perf_counter()
    stats = {"kv_cache_mb": round(cache.nbytes() / 1e6, 2), "device": str(device)}
    if decoder is not None:
        stats["cuda_graphs"] = sorted(decoder.graphs)  # the cache lengths recorded
    return GenerationResult(prompt_ids, new, stop_reason, t1 - t0, t2 - t1, stats)


def stream_text(tokenizer, ids: Iterator[int]) -> Iterator[str]:
    """Decode a token stream into text pieces, holding back incomplete UTF-8
    sequences (one character can span several byte-level tokens)."""
    buf = b""
    for i in ids:
        buf += tokenizer.decode_bytes([i])
        try:
            text = buf.decode("utf-8")
        except UnicodeDecodeError as e:
            if e.start > 0:
                yield buf[: e.start].decode("utf-8")
                buf = buf[e.start:]
            continue
        buf = b""
        yield text
    if buf:
        yield buf.decode("utf-8", errors="replace")
