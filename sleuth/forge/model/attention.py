"""Causal self-attention with grouped-query attention (GQA) and a KV cache.

Shapes: B batch, T new tokens, S total key length (cache + new), H query
heads, K key/value heads (K divides H), hd head dim.

GQA: H query heads share K key/value heads (each group of H/K query heads
reads the same K/V). With K = H this is ordinary multi-head attention;
with K = 1 it's multi-query attention. Fewer KV heads mean a smaller KV
cache, which is the main memory and bandwidth cost of decoding long
sequences.

Two code paths compute the same thing:
  * fused: F.scaled_dot_product_attention picks a FlashAttention-style
    kernel that never materializes the [T, S] attention matrix. Fast and
    memory-light, but it can't show us the attention pattern.
  * explicit: softmax(q·kᵀ/√hd + mask)·v, written out. It's used only when
    the analyzer listens to the `pattern` tap.

Per-layer variations (see ModelConfig.attn_spec), used by Gemma 4:
  * sliding window: query i only sees keys i-window+1 .. i, so the cost and
    the KV cache that matters stay bounded however long the text gets.
  * v_from_k: the layer has no v_proj; values are the keys before k_norm/RoPE.
  * v_norm: values are RMS-normalized (no learned gain), like q and k.
  * attn_scale: Gemma 4 uses 1 instead of 1/√hd, since q_norm and k_norm
    already control the size of the scores.

Document masking (training): `same_doc` [B, 1, T, T] is True where query and
key belong to the same document. It's ANDed with the causal mask, so the
attention matrix becomes block-diagonal: one causal triangle per document.
With a mask, SDPA can't use its FlashAttention kernel and falls back to the
"memory-efficient" one, which takes masks; the speed cost is measured on the
GPU (round 6), and FlexAttention is the fix if it's large.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .norm import RMSNorm
from .rope import apply_rope
from .taps import Tap

# SDPA can broadcast K/V heads itself from PyTorch 2.5; before that we
# repeat them explicitly (same result, a little more memory traffic).
_SDPA_HAS_GQA = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 5)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        spec = cfg.attn_spec(layer_idx)
        self.layer_idx = layer_idx
        self.kind = spec.kind
        self.n_heads, self.n_kv, self.hd = cfg.n_heads, spec.kv_heads, spec.head_dim
        self.window = spec.window
        self.scale = cfg.attn_scale or 1.0 / math.sqrt(self.hd)
        self.eps = cfg.norm_eps
        self.use_rope = cfg.pos_type == "rope"
        qkv_bias = cfg.bias or cfg.qkv_bias
        self.q_proj = nn.Linear(cfg.d_model, self.n_heads * self.hd, bias=qkv_bias)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv * self.hd, bias=qkv_bias)
        self.v_proj = None if spec.v_from_k else nn.Linear(cfg.d_model, self.n_kv * self.hd, bias=qkv_bias)
        self.o_proj = nn.Linear(self.n_heads * self.hd, cfg.d_model, bias=cfg.bias)
        self.q_norm = RMSNorm(self.hd, cfg.norm_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(self.hd, cfg.norm_eps) if cfg.qk_norm else None
        self.v_norm = cfg.v_norm
        self.tap_q = Tap()
        self.tap_k = Tap()
        self.tap_pattern = Tap()
        self.tap_z = Tap()

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache=None,
                same_doc: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)  # [B, H, T, hd]
        k = self.k_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)     # [B, K, T, hd]
        v = k if self.v_proj is None else self.v_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.v_norm:
            v = F.rms_norm(v.float(), (self.hd,), None, self.eps).to(v.dtype)
        if self.use_rope:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        q, k = self.tap_q(q), self.tap_k(k)

        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)  # now [B, K, S, hd]
        S = k.shape[2]
        offset = S - T  # how many earlier tokens are in the cache

        if self.tap_pattern.active:
            y = self._explicit(q, k, v, offset, same_doc)
        else:
            y = self._fused(q, k, v, offset, same_doc)

        y = y.transpose(1, 2)            # [B, T, H, hd]
        y = self.tap_z(y)                # per-head outputs (head ablation hooks in here)
        return self.o_proj(y.reshape(B, T, self.n_heads * self.hd))

    # ------------------------------------------------------------------ #
    def _mask(self, T: int, S: int, offset: int, device) -> torch.Tensor:
        """Boolean [T, S] mask, True = may attend. Query i (absolute position
        offset+i) sees keys at positions <= offset+i, and, in a sliding layer,
        only the last `window` of them (llama.cpp: masked if q - k >= window)."""
        qpos = torch.arange(T, device=device)[:, None] + offset
        kpos = torch.arange(S, device=device)[None, :]
        mask = kpos <= qpos
        if self.window:
            mask &= qpos - kpos < self.window
        return mask

    def _fused(self, q, k, v, offset, same_doc=None):
        T, S = q.shape[2], k.shape[2]
        kwargs = {"scale": self.scale}
        if self.n_kv != self.n_heads:
            # With a document mask, repeat K/V ourselves: the masked (memory-efficient)
            # kernel may not take enable_gqa, and SDPA would then quietly fall back to
            # the math kernel, which stores the whole [B, H, T, T] attention matrix.
            if _SDPA_HAS_GQA and same_doc is None:
                kwargs["enable_gqa"] = True
            else:
                k = k.repeat_interleave(self.n_heads // self.n_kv, dim=1)
                v = v.repeat_interleave(self.n_heads // self.n_kv, dim=1)
        windowed = self.window and S > self.window
        if same_doc is not None:
            mask = self._mask(T, S, offset, q.device) & same_doc
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, **kwargs)
        if offset == 0 and not windowed:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, **kwargs)
        if T == 1 and not windowed:
            # One new token attends to everything cached: no mask needed.
            return F.scaled_dot_product_attention(q, k, v, **kwargs)
        mask = self._mask(T, S, offset, q.device)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, **kwargs)

    def _explicit(self, q, k, v, offset, same_doc=None):
        T, S = q.shape[2], k.shape[2]
        rep = self.n_heads // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        scores = (q.float() @ k.float().transpose(-2, -1)) * self.scale  # [B, H, T, S]
        mask = self._mask(T, S, offset, q.device)
        if same_doc is not None:
            mask = mask & same_doc
        scores = scores.masked_fill(~mask, float("-inf"))
        probs = self.tap_pattern(scores.softmax(dim=-1))
        return (probs @ v.float()).to(q.dtype)
