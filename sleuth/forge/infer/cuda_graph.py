"""CUDA-graph decoding: one GPU launch per generated token.

Decoding one token runs every layer on a single vector. Each layer is a few
small GPU kernels, and for a big model that's ~3,000 kernels per token. On the
RTX 5060 Ti box (round 7, Gemma 4 12B Q4_0) the GPU was busy only 25.7 ms of a
104 ms token: the rest was the CPU launching those kernels one by one (Python,
the PyTorch dispatcher, the driver).

A CUDA graph records the kernels of one decode step once and replays all of them
with a single call. Replays use the same memory every time, so the step must not
depend on anything that changes per token on the CPU side:
  * the input token is copied into a fixed buffer (`self.token`);
  * the position lives in a GPU tensor (StaticKVCache.pos_t);
  * attention reads a fixed-length slice of the cache and masks the rest.

Reading the whole 4k-token cache every step would waste bandwidth on short
chats, so graphs are recorded per length "bucket" (64, 128, ... max_len), each
the first time the conversation reaches it, and they share one memory pool.

Prefill (the prompt, many tokens at once) stays on the normal path: it's one
launch per kernel anyway, for all prompt tokens together.
"""

from __future__ import annotations

import contextlib

import torch

from .kv_cache import StaticKVCache


def _buckets(max_len: int, smallest: int = 64) -> list[int]:
    out, b = [], smallest
    while b < max_len:
        out.append(b)
        b *= 2
    return out + [max_len]


def graph_unsupported(model) -> str | None:
    """Why this model can't decode with CUDA graphs yet (None = it can)."""
    from ..model.moe import MoE

    if hasattr(model, "new_cache"):
        return "CUDA graphs aren't implemented for this architecture yet (it has its own cache)"
    if any(isinstance(m, MoE) for m in model.modules()):
        return "CUDA graphs aren't implemented for MoE models yet (expert choice needs a host sync)"
    return None


def decoder_for(model, max_len: int, autocast=None, tokens_per_step: int = 1,
                with_hidden: bool = False) -> "GraphDecoder":
    """The model's GraphDecoder for this context length and step size, reused
    across generate() calls, so the graphs are recorded once per model (a chat
    doesn't pay for recording every turn). Its cache is rewound for the new
    prompt: the prefill overwrites it."""
    decs = model.__dict__.setdefault("_graph_decoders", {})
    key = (max_len, tokens_per_step, with_hidden)
    if key not in decs:
        decs[key] = GraphDecoder(model, max_len, autocast, tokens_per_step, with_hidden)
    model._graph_decoder = decs[key]  # the most recent one (tests and stats look here)
    decs[key].cache.reset()
    return decs[key]


class GraphDecoder:
    """Decode with CUDA graphs. Use: `cache = dec.cache` for the prefill, then
    `dec.step(token_id_tensor)` for every new token (returns [1, vocab] logits).

    tokens_per_step > 1 (speculative decoding): `dec.verify(tokens)` runs that many
    tokens at positions cache.pos, cache.pos+1, ... and returns [T, vocab] logits;
    the caller sets cache.pos afterwards (how many of them to keep)."""

    def __init__(self, model, max_len: int, autocast=None, tokens_per_step: int = 1, with_hidden: bool = False):
        p = next(model.parameters())
        self.with_hidden = with_hidden  # also output the final-norm hidden states (MTP drafters)
        self.hidden: torch.Tensor | None = None
        if p.device.type != "cuda":
            raise ValueError("CUDA graphs need a model on a CUDA device")
        self.model, self.device, self.max_len = model, p.device, max_len
        self.autocast = autocast or contextlib.nullcontext()
        self.T = tokens_per_step
        shapes = [(model.cfg.attn_spec(i).kv_heads, model.cfg.attn_spec(i).head_dim) for i in range(model.cfg.n_layers)]
        self.cache = StaticKVCache(shapes, 1, max_len, self.device, torch.bfloat16, tokens_per_step)
        self.token = torch.zeros(1, self.T, dtype=torch.long, device=self.device)
        self.graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.buckets = _buckets(max_len)

    def _forward(self):
        with self.autocast:
            out = self.model(self.token, cache=self.cache, return_hidden=self.with_hidden)
        return (out.logits[0], out.hidden[0]) if self.with_hidden else out.logits[0]  # [T, vocab] (, [T, d])

    def _record(self, length: int):
        """Warm up on a side stream (Triton autotuning, cuBLAS workspaces), then capture."""
        self.cache.length = length
        side = torch.cuda.Stream(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(2):
                self._forward()  # writes K/V at pos_t: the same values the real step writes
        torch.cuda.current_stream(self.device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            out = self._forward()
        self.graphs[length] = (graph, out)

    @torch.no_grad()
    def _run(self, tokens: torch.Tensor) -> torch.Tensor:
        pos = self.cache.pos
        if pos + self.T > self.max_len:
            raise ValueError(f"KV cache full ({self.max_len} tokens)")
        length = next(b for b in self.buckets if b >= pos + self.T)
        self.token.copy_(tokens.view(1, self.T))
        self.cache.pos_t.fill_(pos)
        self.cache.static = True
        try:
            if length not in self.graphs:
                self._record(length)
            self.cache.length = length
            graph, out = self.graphs[length]
            graph.replay()
        finally:
            self.cache.static = False
        if self.with_hidden:
            out, self.hidden = out  # hidden: [T, d], the graph's buffer (read before the next replay)
        return out

    def step(self, token: torch.Tensor) -> torch.Tensor:
        """One decode step for `token` (an int64 tensor of one element, on the GPU)
        at position cache.pos; returns its [1, vocab] logits and advances the cache.
        The logits are the graph's output buffer: use them before the next step."""
        if self.T != 1:
            raise ValueError("step() is for one token at a time; use verify()")
        out = self._run(token)
        self.cache.pos += 1
        return out

    def verify(self, tokens: torch.Tensor) -> torch.Tensor:
        """T tokens at positions cache.pos.. -> [T, vocab] logits (the graph's output
        buffer). Doesn't move cache.pos: the caller keeps as many as it accepts."""
        return self._run(tokens)
