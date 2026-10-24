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
