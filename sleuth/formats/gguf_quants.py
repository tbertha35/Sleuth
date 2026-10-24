"""Dequantizing GGUF tensors (ggml block formats), written in plain PyTorch.

Quantization stores weights as *blocks*: a group of 32 or 256 numbers kept as
small integers (2–8 bits each) plus one or more scales. To get floats back:
value = scale × integer (minus an offset or "min" in some formats).

Simple formats (32 values per block):
  Q4_0  [d:f16][16 bytes: 32 × 4-bit]          value = d × (q − 8)
  Q4_1  [d:f16][m:f16][16 bytes]                 value = d × q + m
  Q5_0  [d:f16][4 bytes: 5th bits][16 bytes]     value = d × (q − 16)
  Q5_1  [d:f16][m:f16][4 bytes][16 bytes]        value = d × q + m
  Q8_0  [d:f16][32 × int8]                       value = d × q
  IQ4_NL [d:f16][16 bytes]                       value = d × table[q]  (non-uniform levels)
In these, byte j's low 4 bits are value j and its high 4 bits are value j+16.

"K-quants" (256 values per block, split into 8 or 16 sub-blocks, each with its
own 4–8-bit scale, which in turn is scaled by a per-block f16):
  Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, and IQ4_XS (a K-style layout with IQ4_NL's levels).
The bit layouts below follow ggml's reference C code (dequantize_row_*) line
for line; comments name the struct fields.

Everything is vectorized over blocks: input is a uint8 tensor of whole
blocks, output float32. It runs on CPU or GPU, which matters later for keeping
large models quantized in VRAM and dequantizing a layer at a time.
"""

from __future__ import annotations

import torch

# ggml type ids (see ggml.h)
F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1 = 0, 1, 2, 3, 6, 7, 8, 9
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K = 10, 11, 12, 13, 14, 15
IQ4_NL, IQ4_XS, I8, I16, I32, I64, F64, BF16 = 20, 23, 24, 25, 26, 27, 28, 30

TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
    23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
    34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
}

# (values per block, bytes per block)
BLOCK = {
    F32: (1, 4), F16: (1, 2), BF16: (1, 2), F64: (1, 8), I8: (1, 1), I16: (1, 2), I32: (1, 4), I64: (1, 8),
    Q4_0: (32, 18), Q4_1: (32, 20), Q5_0: (32, 22), Q5_1: (32, 24), Q8_0: (32, 34), Q8_1: (32, 36),
    Q2_K: (256, 84), Q3_K: (256, 110), Q4_K: (256, 144), Q5_K: (256, 176), Q6_K: (256, 210), Q8_K: (256, 292),
    IQ4_NL: (32, 18), IQ4_XS: (256, 136),
}

IQ4_LEVELS = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


def tensor_nbytes(ggml_type: int, n_elements: int) -> int:
    if ggml_type not in BLOCK:
        name = TYPE_NAMES.get(ggml_type, str(ggml_type))
        raise NotImplementedError(f"GGUF tensor type {name} isn't supported yet")
    per, nb = BLOCK[ggml_type]
    if n_elements % per:
        raise ValueError(f"{n_elements} values isn't a whole number of {per}-value blocks")
    return n_elements // per * nb


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _f16(b: torch.Tensor) -> torch.Tensor:
    """[n, 2] bytes -> [n, 1] float32 (little-endian float16)."""
    return b.contiguous().view(torch.float16).float()


def _u32(b: torch.Tensor) -> torch.Tensor:
    """[n, 4] bytes -> [n] int64 (little-endian uint32)."""
    b = b.long()
    return b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16) | (b[:, 3] << 24)


def _nibbles(qs: torch.Tensor) -> torch.Tensor:
    """[n, 16] bytes -> [n, 32] values: low nibbles are 0..15, high nibbles 16..31."""
    return torch.cat([qs & 0x0F, qs >> 4], dim=1).long()


def _scale_min_k4(s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Q4_K/Q5_K: 12 bytes -> 8 six-bit scales and 8 six-bit mins (get_scale_min_k4)."""
    s = s.long()
    sc = torch.empty(s.shape[0], 8, dtype=torch.long, device=s.device)
    mn = torch.empty_like(sc)
    sc[:, :4] = s[:, 0:4] & 63
    mn[:, :4] = s[:, 4:8] & 63
    sc[:, 4:] = (s[:, 8:12] & 0x0F) | ((s[:, 0:4] >> 6) << 4)
    mn[:, 4:] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    return sc, mn


# --------------------------------------------------------------------------- #
# 32-value formats
# --------------------------------------------------------------------------- #

def _q4_0(b):
    return (_nibbles(b[:, 2:18]) - 8) * _f16(b[:, 0:2])


def _q4_1(b):
    return _nibbles(b[:, 4:20]) * _f16(b[:, 0:2]) + _f16(b[:, 2:4])


def _fifth_bits(b4: torch.Tensor) -> torch.Tensor:
    """qh (uint32): bit j is the 5th bit of value j, for j = 0..31."""
    qh = _u32(b4)
    return ((qh[:, None] >> torch.arange(32, device=b4.device)) & 1) << 4


def _q5_0(b):
    return ((_nibbles(b[:, 6:22]) | _fifth_bits(b[:, 2:6])) - 16) * _f16(b[:, 0:2])


def _q5_1(b):
    return (_nibbles(b[:, 8:24]) | _fifth_bits(b[:, 4:8])) * _f16(b[:, 0:2]) + _f16(b[:, 2:4])


def _q8_0(b):
    return b[:, 2:34].contiguous().view(torch.int8).float() * _f16(b[:, 0:2])


def _iq4_nl(b):
    levels = torch.tensor(IQ4_LEVELS, dtype=torch.float32, device=b.device)
    return levels[_nibbles(b[:, 2:18])] * _f16(b[:, 0:2])


# --------------------------------------------------------------------------- #
# 256-value K-quants
# --------------------------------------------------------------------------- #

def _q2_k(b):
    # struct { u8 scales[16]; u8 qs[64]; f16 d; f16 dmin; }
    n = b.shape[0]
    scales = b[:, 0:16].long().reshape(n, 2, 4, 2)          # [half, shift step j, group]
    qs = b[:, 16:80].long().reshape(n, 2, 1, 2, 16)         # [half, -, group, l]
    d, dmin = _f16(b[:, 80:82]), _f16(b[:, 82:84])
    shifts = (2 * torch.arange(4, device=b.device)).reshape(1, 1, 4, 1, 1)
    q = (qs >> shifts) & 3                                    # [n, half, j, group, l]
    dl = (d[:, :, None, None] * (scales & 0xF))[..., None]
    ml = (dmin[:, :, None, None] * (scales >> 4))[..., None]
    return (dl * q - ml).reshape(n, 256)


def _q3_k(b):
    # struct { u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d; }
    n = b.shape[0]
    hmask = b[:, 0:32].long().reshape(n, 1, 1, 2, 16)
    qs = b[:, 32:96].long().reshape(n, 2, 1, 2, 16)
    d = _f16(b[:, 108:110])
    # Unpack 16 six-bit scales from 12 bytes (aux[] juggling in dequantize_row_q3_K).
    a0, a1, tmp = _u32(b[:, 96:100]), _u32(b[:, 100:104]), _u32(b[:, 104:108])
    k1, k2 = 0x03030303, 0x0F0F0F0F
    aux = torch.stack([
        (a0 & k2) | (((tmp >> 0) & k1) << 4),
        (a1 & k2) | (((tmp >> 2) & k1) << 4),
        ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4),
        ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4),
    ], dim=1)                                                 # [n, 4] uint32 values
    scales = ((aux[:, :, None] >> (8 * torch.arange(4, device=b.device))) & 0xFF)  # [n, 4, 4] bytes
    scales = scales.reshape(n, 16).to(torch.int64)
    scales = torch.where(scales > 127, scales - 256, scales) - 32  # int8, then -32
    scales = scales.reshape(n, 2, 4, 2)
    j = torch.arange(4, device=b.device).reshape(1, 1, 4, 1, 1)
    half = torch.arange(2, device=b.device).reshape(1, 2, 1, 1, 1)
    q = (qs >> (2 * j)) & 3
    hbit = (hmask >> (half * 4 + j)) & 1                      # high bit set -> no "-4"
    q = q - 4 * (1 - hbit)
    return (d[:, :, None, None, None] * scales[..., None] * q).reshape(n, 256)


def _q4_k(b):
    # struct { f16 d; f16 dmin; u8 scales[12]; u8 qs[128]; }
    n = b.shape[0]
    d, dmin = _f16(b[:, 0:2]), _f16(b[:, 2:4])
    sc, mn = _scale_min_k4(b[:, 4:16])                        # [n, 8]
    qs = b[:, 16:144].long().reshape(n, 4, 32)                # 4 chunks of 64 values
    q = torch.stack([qs & 0x0F, qs >> 4], dim=2)              # [n, chunk, lo/hi, 32]
    sc, mn = sc.reshape(n, 4, 2, 1), mn.reshape(n, 4, 2, 1)
    return (d[:, :, None, None] * sc * q - dmin[:, :, None, None] * mn).reshape(n, 256)


def _q5_k(b):
    # struct { f16 d; f16 dmin; u8 scales[12]; u8 qh[32]; u8 qs[128]; }
    n = b.shape[0]
    d, dmin = _f16(b[:, 0:2]), _f16(b[:, 2:4])
    sc, mn = _scale_min_k4(b[:, 4:16])
    qh = b[:, 16:48].long().reshape(n, 1, 1, 32)
    qs = b[:, 48:176].long().reshape(n, 4, 32)
    q = torch.stack([qs & 0x0F, qs >> 4], dim=2)              # [n, chunk, lo/hi, 32]
    bit = (2 * torch.arange(4, device=b.device)).reshape(1, 4, 1, 1) + torch.arange(2, device=b.device).reshape(1, 1, 2, 1)
    q = q + ((qh >> bit) & 1) * 16
    sc, mn = sc.reshape(n, 4, 2, 1), mn.reshape(n, 4, 2, 1)
    return (d[:, :, None, None] * sc * q - dmin[:, :, None, None] * mn).reshape(n, 256)


def _q6_k(b):
    # struct { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d; }
    n = b.shape[0]
    ql = b[:, 0:128].long().reshape(n, 2, 64)
    qh = b[:, 128:192].long().reshape(n, 2, 32)
    sc = b[:, 192:208].contiguous().view(torch.int8).long().reshape(n, 2, 4, 2)  # [half, quarter, l//16]
    d = _f16(b[:, 208:210])
    A, B = ql[:, :, :32], ql[:, :, 32:]
    q = torch.stack([
        (A & 0x0F) | (((qh >> 0) & 3) << 4),
        (B & 0x0F) | (((qh >> 2) & 3) << 4),
        (A >> 4) | (((qh >> 4) & 3) << 4),
        (B >> 4) | (((qh >> 6) & 3) << 4),
    ], dim=2) - 32                                            # [n, half, quarter, 32]
    scale = sc.repeat_interleave(16, dim=3)                   # [n, half, quarter, 32]
    return (d[:, :, None, None] * scale * q).reshape(n, 256)


def _iq4_xs(b):
    # struct { f16 d; u16 scales_h; u8 scales_l[4]; u8 qs[128]; }
    n = b.shape[0]
    d = _f16(b[:, 0:2])
    sh = b[:, 2].long() | (b[:, 3].long() << 8)
    sl = b[:, 4:8].long()
    ib = torch.arange(8, device=b.device)
    lo = (sl[:, ib // 2] >> (4 * (ib % 2))) & 0x0F
    hi = (sh[:, None] >> (2 * ib)) & 3
    ls = (lo | (hi << 4)) - 32                                # [n, 8] sub-block scales
    levels = torch.tensor(IQ4_LEVELS, dtype=torch.float32, device=b.device)
    qs = b[:, 8:136].reshape(n * 8, 16)
    vals = levels[_nibbles(qs)].reshape(n, 8, 32)
    return (d[:, :, None] * ls[:, :, None] * vals).reshape(n, 256)


_DEQUANT = {Q4_0: _q4_0, Q4_1: _q4_1, Q5_0: _q5_0, Q5_1: _q5_1, Q8_0: _q8_0, IQ4_NL: _iq4_nl,
            Q2_K: _q2_k, Q3_K: _q3_k, Q4_K: _q4_k, Q5_K: _q5_k, Q6_K: _q6_k, IQ4_XS: _iq4_xs}
_PLAIN = {F32: torch.float32, F16: torch.float16, BF16: torch.bfloat16, F64: torch.float64,
          I8: torch.int8, I16: torch.int16, I32: torch.int32, I64: torch.int64}


def dequantize(raw: torch.Tensor, ggml_type: int, n_elements: int) -> torch.Tensor:
    """uint8 bytes of a whole tensor -> flat float32 tensor of n_elements."""
    nbytes = tensor_nbytes(ggml_type, n_elements)
    if raw.numel() != nbytes:
        raise ValueError(f"expected {nbytes} bytes, got {raw.numel()}")
    if ggml_type in _PLAIN:
        return raw.view(_PLAIN[ggml_type]).float()
    if ggml_type not in _DEQUANT:
        raise NotImplementedError(f"dequantizing {TYPE_NAMES.get(ggml_type, ggml_type)} isn't implemented yet")
    per, nb = BLOCK[ggml_type]
    blocks = raw.reshape(-1, nb)
    out = torch.empty(blocks.shape[0], per, dtype=torch.float32, device=raw.device)
    step = 1 << 16  # bounded chunks keep the int64 temporaries small for huge tensors
    for i in range(0, blocks.shape[0], step):
        out[i:i + step] = _DEQUANT[ggml_type](blocks[i:i + step])
    return out.reshape(-1)
