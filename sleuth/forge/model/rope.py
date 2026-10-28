"""Rotary position embeddings (RoPE).

Instead of adding a position vector to the input, RoPE *rotates* each
query and key vector by an angle proportional to its position. Channel
pairs rotate at different speeds (frequencies theta^(-2i/d)), like the
hands of a clock. The dot product q·k then depends only on the
*difference* of positions, so attention becomes naturally relative.

Layout convention: we pair channel i with channel i + d/2 ("rotate half",
as in GPT-NeoX and Hugging Face's Llama code), not adjacent channels
(Meta's original Llama code). The two are the same up to a fixed
permutation of the q/k weight rows; external-model adapters must apply it
when loading weights stored in the other convention (e.g. GGUF).
"""

from __future__ import annotations

import torch


def rope_tables(head_dim: int, max_len: int, theta: float,
                freq_factors: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape [max_len, head_dim], computed in float32.

    freq_factors ([head_dim/2], optional) divide each pair's frequency, as in
    llama.cpp's ggml_rope_ext. Gemma 4's full-attention layers ship them as the
    rope_freqs tensor ("proportional RoPE"): a huge factor makes a pair rotate
    (almost) not at all, so those channels carry no position information.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    if freq_factors is not None:
        if freq_factors.numel() != head_dim // 2:
            raise ValueError(f"{freq_factors.numel()} RoPE frequency factors for head_dim {head_dim}")
        inv_freq = inv_freq / freq_factors.float().cpu()
    pos = torch.arange(max_len, dtype=torch.float32)
    angles = torch.outer(pos, inv_freq)            # [max_len, head_dim/2]
    angles = torch.cat([angles, angles], dim=-1)   # duplicate for the two halves
    return angles.cos(), angles.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, head_dim]; cos/sin: [T, head_dim] for the positions of x.
    Per channel pair (a, b): (a, b) -> (a·cos - b·sin, b·cos + a·sin).

    Narrower tables ([T, rope_dim], partial RoPE as in GPT-NeoX/Pythia) rotate only the
    first rope_dim channels, paired within them; the rest pass through unchanged."""
    rd = cos.shape[-1]
    if rd < x.shape[-1]:
        return torch.cat([apply_rope(x[..., :rd], cos, sin), x[..., rd:]], dim=-1)
    return (x.float() * cos + _rotate_half(x.float()) * sin).to(x.dtype)
