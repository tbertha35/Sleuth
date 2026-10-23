"""Key/value cache for fast autoregressive decoding.

Without a cache, generating token t recomputes keys and values for all t-1
earlier tokens: O(t²) work per sequence. Earlier tokens' K and V never
change (attention is causal), so we store them once and only compute the
new token's. That leaves decoding *memory-bound*: each step reads every
weight plus the whole cache, which is why decode speed tracks memory
bandwidth (see PLAN §0.1).

Memory: 2 (K and V) · n_layers · batch · n_kv_heads · max_len · head_dim ·
bytes. GQA shrinks this by n_heads / n_kv_heads.
"""

from __future__ import annotations

import torch


class KVCache:
    def __init__(self, n_layers: int, batch: int, n_kv_heads: int, max_len: int, head_dim: int,
                 device: torch.device | str, dtype: torch.dtype):
        shape = (n_layers, batch, n_kv_heads, max_len, head_dim)
        # Preallocated once: growing tensors with torch.cat every step would
        # copy the whole cache each token.
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_len = max_len
        self.pos = 0  # tokens already stored (same for every layer)

    @classmethod
    def for_model(cls, model, batch: int, max_len: int | None = None, dtype: torch.dtype | None = None) -> "KVCache":
        cfg = model.cfg
        p = next(model.parameters())
        return cls(cfg.n_layers, batch, cfg.kv_heads, max_len or cfg.max_seq_len, cfg.head_dim,
                   p.device, dtype or p.dtype)

    def update(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Store this step's K/V for `layer`; return all K/V so far: [B, K, pos+T, hd]."""
        T = k_new.shape[2]
        end = self.pos + T
        if end > self.max_len:
            raise ValueError(f"KV cache full ({self.max_len} tokens)")
        self.k[layer, :, :, self.pos:end] = k_new.to(self.k.dtype)
        self.v[layer, :, :, self.pos:end] = v_new.to(self.v.dtype)
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    def advance(self, n: int) -> None:
        """Called by the model once all layers have stored the new tokens."""
        self.pos += n

    def reset(self) -> None:
        self.pos = 0

    def nbytes(self) -> int:
        return 2 * self.k.numel() * self.k.element_size()
