"""Key/value cache for fast autoregressive decoding.

Without a cache, generating token t recomputes keys and values for all t-1
earlier tokens: O(t²) work per sequence. Earlier tokens' K and V never
change (attention is causal), so we store them once and only compute the
new token's. That leaves decoding *memory-bound*: each step reads every
weight plus the whole cache, which is why decode speed tracks memory
bandwidth (see PLAN §0.1).

Memory: 2 (K and V) · n_layers · batch · n_kv_heads · max_len · head_dim ·
bytes. GQA shrinks this by n_heads / n_kv_heads. Each layer gets its own
buffer, because layers can differ in KV heads and head width (Gemma 4's
full layers: 1 KV head of 512; its sliding layers: 8 KV heads of 256).

Sliding-window layers still store every position here (simple, and the
attention mask hides the old ones). A ring buffer of `window` positions
would save memory for very long contexts; not needed yet.
"""

from __future__ import annotations

import torch


class KVCache:
    def __init__(self, shapes: list[tuple[int, int]], batch: int, max_len: int,
                 device: torch.device | str, dtype: torch.dtype):
        """shapes: (n_kv_heads, head_dim) for each layer."""
        # Preallocated once: growing tensors with torch.cat every step would
        # copy the whole cache each token.
        self.k = [torch.zeros(batch, kv, max_len, hd, device=device, dtype=dtype) for kv, hd in shapes]
        self.v = [torch.zeros_like(k) for k in self.k]
        self.max_len = max_len
        self.pos = 0  # tokens already stored (same for every layer)
        self._tokens: torch.Tensor | None = None

    def token_history(self, batch: int, device) -> torch.Tensor:
        """[batch, max_len] token ids by position (-1 = none yet), for layers that look
        at earlier tokens themselves (the n-gram memory, memory.py). Made on first use;
        the caller writes the new tokens before reading back."""
        if self._tokens is None:
            self._tokens = torch.full((batch, self.max_len), -1, dtype=torch.long, device=device)
        return self._tokens

    @classmethod
    def for_model(cls, model, batch: int, max_len: int | None = None, dtype: torch.dtype | None = None) -> "KVCache":
        cfg = model.cfg
        p = next(model.parameters())
        shapes = [(cfg.attn_spec(i).kv_heads, cfg.attn_spec(i).head_dim) for i in range(cfg.n_layers)]
        return cls(shapes, batch, max_len or cfg.max_seq_len, p.device, dtype or p.dtype)

    def update(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Store this step's K/V for `layer`; return all K/V so far: [B, K, pos+T, hd]."""
        T = k_new.shape[2]
        end = self.pos + T
        if end > self.max_len:
            raise ValueError(f"KV cache full ({self.max_len} tokens)")
        self.k[layer][:, :, self.pos:end] = k_new.to(self.k[layer].dtype)
        self.v[layer][:, :, self.pos:end] = v_new.to(self.v[layer].dtype)
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def advance(self, n: int) -> None:
        """Called by the model once all layers have stored the new tokens."""
        self.pos += n

    def reset(self) -> None:
        self.pos = 0

    def nbytes(self) -> int:
        return sum(2 * k.numel() * k.element_size() for k in self.k)


class StaticKVCache(KVCache):
    """A KVCache that one-token decode steps can use inside a CUDA graph.

    A CUDA graph replays the exact same kernels on the exact same memory, so
    nothing a decode step does may depend on a Python number that changes every
    token. Here the position lives in a GPU tensor (`pos_t`): the new K/V are
    written at that index, RoPE rows are looked up with it, and attention reads
    a fixed-length view of the cache (`length`, set per graph) with everything
    past `pos_t` masked out.

    Outside `static` mode (the prompt's prefill) it behaves like KVCache.
    """

    def __init__(self, shapes, batch, max_len, device, dtype, tokens_per_step: int = 1):
        super().__init__(shapes, batch, max_len, device, dtype)
        self.static = False
        self.length = max_len                 # how much of the cache attention reads in static mode
        self.pos_t = torch.zeros(1, dtype=torch.long, device=device)
        self._kpos = torch.arange(max_len, device=device)
        # Static steps may take several tokens at once (speculative decoding checks
        # k guesses in one pass): they go to positions pos_t, pos_t+1, ...
        self.tokens_per_step = tokens_per_step
        self._qoff = torch.arange(tokens_per_step, device=device)
        self._masks: dict[int, torch.Tensor] = {}
        self._pos: torch.Tensor | None = None

    @property
    def positions(self) -> torch.Tensor:
        """[tokens_per_step] positions of this static step's tokens (computed once per
        forward pass: every layer's K/V write uses them)."""
        if self._pos is None:
            self._pos = self.pos_t + self._qoff
        return self._pos

    def update(self, layer, k_new, v_new):
        if not self.static:
            return super().update(layer, k_new, v_new)
        # Write at pos_t (and the positions after it) without leaving the GPU.
        pos = self.positions
        self.k[layer].index_copy_(2, pos, k_new.to(self.k[layer].dtype))
        self.v[layer].index_copy_(2, pos, v_new.to(self.v[layer].dtype))
        return self.k[layer][:, :, :self.length], self.v[layer][:, :, :self.length]

    def attn_mask(self, window: int = 0) -> torch.Tensor:
        """[1, 1, T, length] bool, True = may attend: each query sees keys up to its
        own position (and, in a sliding layer, only the last `window` of them).
        Built once per forward pass for each window and shared by all layers
        (Gemma 4: 2 masks per step instead of 48); new_step() starts a new pass."""
        if window not in self._masks:
            kpos = self._kpos[:self.length][None, :]
            qpos = self.positions[:, None]
            mask = kpos <= qpos
            if window:
                mask = mask & (qpos - kpos < window)
            self._masks[window] = mask[None, None]
        return self._masks[window]

    def new_step(self, positions: torch.Tensor | None = None) -> None:
        """Called by the model at the start of each forward pass (positions may have moved).
        positions: this pass's positions, if not pos_t, pos_t+1, ... (the early-exit
        drafter's graph runs one token at a time at pos_t + j)."""
        self._masks, self._pos = {}, positions

    def advance(self, n):
        if not self.static:  # in static mode the caller moves pos and pos_t
            super().advance(n)
