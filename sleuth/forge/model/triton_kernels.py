"""Fused "dequantize + matmul" GPU kernels, written in Triton.

Why: a kept-quantized layer (quant.py) normally unpacks its whole weight to
bf16 in memory, then multiplies. For Gemma 4 12B that writes ~24 GB of bf16
for *every token*: 0.7 tok/s, where llama.cpp manages 46 on the same card.
These kernels read the 4.5-bit blocks and do the unpacking inside the matmul,
in registers, so each weight byte crosses the memory bus once: ~6.7 GB per token
instead of ~30+.

How the kernel works (y = x @ Wᵀ, x: [M, K] activations, W: [N, K] in blocks):
  * each program computes a BLOCK_M x BLOCK_N tile of y
  * it walks K in steps of BLOCK_K: loads that slice of x, and for the same
    slice of W loads the raw bytes, unpacks the 4-bit (or 8-bit) integers,
    multiplies by each block's f16 scale, rounds to x's dtype, and adds
    x_tile @ W_tileᵀ into an fp32 accumulator (tl.dot, on tensor cores)
  * the rounding matches the plain path exactly: (q - 8)·d is computed in
    fp32 and rounded once to bf16, as dequantize() does

For 1-4 rows of x (decoding a token at a time) a second kernel is used:
tensor cores need 16 rows, so the dot-product kernel wasted most of its work,
and it reloaded each block's scale for every weight. The few-row kernel
handles whole 32-value blocks (one scale load, 16 contiguous bytes) and sums
in fp32 without rounding the weights to bf16 first (a tiny bit *more* precise).

For Q4_0 with 1-4 rows of bf16/fp16 x (decoding, speculative checks) a third kernel
is the default: x is rounded to int8 per 32 values first, as llama.cpp does, and
4 int8 products are done per instruction (dp4a). That makes a 4-row check cost
1.2x one row instead of 1.5x (Gemma 4 12B with MTP: ~72-79 -> 84 tok/s). The
price is llama.cpp's activation rounding (relative error ~5e-3 instead of ~3e-3
against exact math). SLEUTH_EXACT_ACT=1 or set_int8_activations(False) turns it off.

Triton is optional. It ships with PyTorch on Linux; on Windows it's the
`triton-windows` package (owner-approved 2026-10-24). Without it, or if the
self-check below fails, QuantLinear silently uses the plain path. Set the
environment variable SLEUTH_NO_TRITON=1 to force the plain path (to compare).

Tests run these kernels on the CPU with TRITON_INTERPRET=1 (Triton's
interpreter), which is how they were checked without a GPU.
"""

from __future__ import annotations

import os
import warnings

import torch

from ...formats import gguf_quants as q

SUPPORTED = {q.Q4_0: (0, 18), q.Q8_0: (1, 34)}  # ggml type -> (kernel code, bytes per 32-value block)

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on the machine
    triton = None


def available() -> bool:
    return triton is not None and os.environ.get("SLEUTH_NO_TRITON", "") not in ("1", "true", "yes")


if triton is not None:

    @triton.jit
    def _qmatmul_kernel(x_ptr, w_ptr, y_ptr, M, N, K, stride_xm, stride_ym, row_bytes,
                        QTYPE: tl.constexpr, IEEE: tl.constexpr,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_ok = rm < M
        n_ok = rn < N
        row = rn.to(tl.int64) * row_bytes                       # byte offset of each weight row
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            rk = k0 + tl.arange(0, BLOCK_K)
            k_ok = rk < K
            x = tl.load(x_ptr + rm[:, None] * stride_xm + rk[None, :], mask=m_ok[:, None] & k_ok[None, :], other=0.0)
            blk = rk // 32                                     # which 32-value block
            j = rk % 32                                        # position inside it
            wmask = n_ok[:, None] & k_ok[None, :]
            if QTYPE == 0:
                base = w_ptr + row[:, None] + (blk * 18)[None, :]          # Q4_0: [d: f16][16 bytes]
                byte = tl.load(base + 2 + (j % 16)[None, :], mask=wmask, other=0).to(tl.int32)
                qv = ((byte >> ((j // 16) * 4)[None, :]) & 0xF).to(tl.float32) - 8.0  # low nibble: j<16
            else:
                base = w_ptr + row[:, None] + (blk * 34)[None, :]          # Q8_0: [d: f16][32 x int8]
                qv = tl.load(base + 2 + j[None, :], mask=wmask, other=0).to(tl.int8, bitcast=True).to(tl.float32)
            d_lo = tl.load(base, mask=wmask, other=0).to(tl.uint16)
            d_hi = tl.load(base + 1, mask=wmask, other=0).to(tl.uint16)
            d = (d_lo | (d_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            w = (qv * d).to(x.dtype)                           # [BLOCK_N, BLOCK_K], rounded once
            if IEEE:
                acc += tl.dot(x, tl.trans(w), input_precision="ieee")  # fp32 inputs: no TF32 shortcuts
            else:
                acc += tl.dot(x, tl.trans(w))
        y = y_ptr + rm[:, None] * stride_ym + rn[None, :]
        tl.store(y, acc.to(y_ptr.dtype.element_ty), mask=m_ok[:, None] & n_ok[None, :])


    @triton.jit
    def _qgemv4_kernel(x_ptr, w_ptr, y_ptr, M, N, K, stride_xm, stride_ym, row_bytes,
                       QTYPE: tl.constexpr, BLOCK_N: tl.constexpr, BLOCKS: tl.constexpr, MROWS: tl.constexpr):
        """Few-row version (decoding: 1 token; speculative checks: up to 4). Program
        pid_n computes BLOCK_N outputs for every row. Each 32-value block's scale and
        bytes are loaded once and used for all rows, so 4 rows cost little more memory
        traffic than 1. Each row keeps an elementwise fp32 accumulator
        [BLOCK_N, BLOCKS, 16] over the whole K loop, reduced once at the end (reducing
        every chunk was the slow part). A row's arithmetic doesn't depend on M, so a
        speculative check computes exactly what one-token decoding computes. No tensor
        cores: with 1-4 rows of x memory speed is the limit."""
        pid_n = tl.program_id(0)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = rn < N
        row = rn.to(tl.int64) * row_bytes
        n_blocks = K // 32
        i16 = tl.arange(0, 16)
        a0 = tl.zeros((BLOCK_N, BLOCKS, 16), dtype=tl.float32)
        a1 = tl.zeros((BLOCK_N, BLOCKS, 16), dtype=tl.float32)
        a2 = tl.zeros((BLOCK_N, BLOCKS, 16), dtype=tl.float32)
        a3 = tl.zeros((BLOCK_N, BLOCKS, 16), dtype=tl.float32)
        for b0 in range(0, n_blocks, BLOCKS):
            b = b0 + tl.arange(0, BLOCKS)
            b_ok = b < n_blocks
            ok2 = n_ok[:, None] & b_ok[None, :]
            if QTYPE == 0:
                base = w_ptr + row[:, None] + (b * 18)[None, :]
                qs = tl.load(base[:, :, None] + 2 + i16[None, None, :], mask=ok2[:, :, None], other=0).to(tl.int32)
                lo = (qs & 0xF).to(tl.float32) - 8.0
                hi = (qs >> 4).to(tl.float32) - 8.0
            else:
                base = w_ptr + row[:, None] + (b * 34)[None, :]
                lo = tl.load(base[:, :, None] + 2 + i16[None, None, :], mask=ok2[:, :, None],
                             other=0).to(tl.int8, bitcast=True).to(tl.float32)
                hi = tl.load(base[:, :, None] + 18 + i16[None, None, :], mask=ok2[:, :, None],
                             other=0).to(tl.int8, bitcast=True).to(tl.float32)
            d_lo = tl.load(base, mask=ok2, other=0).to(tl.uint16)
            d_hi = tl.load(base + 1, mask=ok2, other=0).to(tl.uint16)
            d = (d_lo | (d_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)[:, :, None]   # [BN, BB, 1]
            lo = lo * d
            hi = hi * d
            xoff = (b * 32)[:, None] + i16[None, :]
            xm = b_ok[:, None]
            a0 += lo * tl.load(x_ptr + xoff, mask=xm, other=0.0).to(tl.float32)[None] + \
                hi * tl.load(x_ptr + xoff + 16, mask=xm, other=0.0).to(tl.float32)[None]
            if MROWS > 1:
                a1 += lo * tl.load(x_ptr + stride_xm + xoff, mask=xm & (M > 1), other=0.0).to(tl.float32)[None] + \
                    hi * tl.load(x_ptr + stride_xm + xoff + 16, mask=xm & (M > 1), other=0.0).to(tl.float32)[None]
            if MROWS > 2:
                a2 += lo * tl.load(x_ptr + 2 * stride_xm + xoff, mask=xm & (M > 2), other=0.0).to(tl.float32)[None] + \
                    hi * tl.load(x_ptr + 2 * stride_xm + xoff + 16, mask=xm & (M > 2), other=0.0).to(tl.float32)[None]
                a3 += lo * tl.load(x_ptr + 3 * stride_xm + xoff, mask=xm & (M > 3), other=0.0).to(tl.float32)[None] + \
                    hi * tl.load(x_ptr + 3 * stride_xm + xoff + 16, mask=xm & (M > 3), other=0.0).to(tl.float32)[None]
        out = y_ptr.dtype.element_ty
        tl.store(y_ptr + rn, tl.sum(tl.sum(a0, axis=2), axis=1).to(out), mask=n_ok)
        if MROWS > 1:
            tl.store(y_ptr + stride_ym + rn, tl.sum(tl.sum(a1, axis=2), axis=1).to(out), mask=n_ok & (M > 1))
        if MROWS > 2:
            tl.store(y_ptr + 2 * stride_ym + rn, tl.sum(tl.sum(a2, axis=2), axis=1).to(out), mask=n_ok & (M > 2))
            tl.store(y_ptr + 3 * stride_ym + rn, tl.sum(tl.sum(a3, axis=2), axis=1).to(out), mask=n_ok & (M > 3))


    # ------------------------------------------------------------------ int8 activations
    # llama.cpp's way of doing the few-row case (its CUDA "mmvq" kernels): round each
    # row of x to int8 once, in blocks of 32 sharing one scale (its q8_1 format), then
    # multiply 4 int8 pairs per instruction (dp4a) and apply both scales once per
    # block. Far fewer instructions per weight than unpacking to fp32, which is what
    # made 4 rows cost 1.46x one row: compute, not memory, was the limit.

    @triton.jit
    def _quant_q8_kernel(x_ptr, xq_ptr, dx_ptr, sx_ptr, K, stride_xm):
        """One program per (row, 32-value block), like llama.cpp's quantize_q8_1:
        d = max|x| / 127 (stored as fp16), q = roundf(x / d). Also the block's sum of q,
        for the Q4_0 offset. (llama.cpp uses the sum of the unrounded x there; with the
        rounded one the offset cancels the rounding error too: 2.5x smaller error on
        Gemma-sized matrices, 5.5e-3 vs 1.3e-2.) The 32 int8 values are packed 4 per int32."""
        r = tl.program_id(0).to(tl.int64)
        blk = tl.program_id(1)
        nb = K // 32
        i = tl.arange(0, 32)
        x = tl.load(x_ptr + r * stride_xm + blk * 32 + i).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=0)
        d = amax / 127.0
        v = tl.where(amax > 0, x / d, 0.0)
        qv = tl.where(v < 0, -tl.floor(-v + 0.5), tl.floor(v + 0.5)).to(tl.int32)   # roundf: halves away from 0
        tl.store(dx_ptr + r * nb + blk, d.to(tl.float16).to(tl.float32))
        tl.store(sx_ptr + r * nb + blk, tl.sum(qv, axis=0).to(tl.float32))
        packed = tl.sum(tl.reshape(qv & 0xFF, (8, 4)) << (tl.arange(0, 4) * 8)[None, :], axis=1)
        tl.store(xq_ptr + r * (K // 4) + blk * 8 + tl.arange(0, 8), packed)

    @triton.jit
    def _dp4a(a, b, c, ASM: tl.constexpr):
        """c + the dot product of the 4 signed bytes in a and in b (int32 each)."""
        if ASM:
            return tl.inline_asm_elementwise("dp4a.s32.s32 $0, $1, $2, $3;", "=r,r,r,r", [a, b, c],
                                             dtype=tl.int32, is_pure=True, pack=1)
        else:  # Triton's interpreter (CPU tests) can't run PTX: same arithmetic, byte by byte
            for s in tl.static_range(4):
                c += ((a << (24 - 8 * s)) >> 24) * ((b << (24 - 8 * s)) >> 24)
            return c

    @triton.jit
    def _q8_row(acc, w0, w1, dw, xq_ptr, dx_ptr, sx_ptr, r, K, b, b_ok, i4, live,
                QTYPE: tl.constexpr, ASM: tl.constexpr):
        """Add one row of x's contribution for these weight blocks. w0/w1: [BN, BB, 4]
        int32 words, the block's first and second 16 values as signed bytes."""
        nb = K // 32
        xo = r * (K // 4) + (b * 8)[:, None] + i4[None, :]
        x0 = tl.load(xq_ptr + xo, mask=b_ok[:, None] & live, other=0)
        x1 = tl.load(xq_ptr + xo + 4, mask=b_ok[:, None] & live, other=0)
        si = tl.sum(_dp4a(w1, x1[None], _dp4a(w0, x0[None], tl.zeros_like(w0), ASM), ASM), axis=2)
        dx = tl.load(dx_ptr + r * nb + b, mask=b_ok & live, other=0.0)[None, :]
        if QTYPE == 0:   # Q4_0 stores q + 8: subtract 8 x sum(q_x) once per block
            sq = tl.load(sx_ptr + r * nb + b, mask=b_ok & live, other=0.0)[None, :]
            return acc + dw * dx * (si.to(tl.float32) - 8.0 * sq)
        else:
            return acc + dw * dx * si.to(tl.float32)

    @triton.jit
    def _qgemv_q8_kernel(xq_ptr, dx_ptr, sx_ptr, w_ptr, y_ptr, M, N, K, stride_ym, row_bytes,
                         QTYPE: tl.constexpr, BLOCK_N: tl.constexpr, BLOCKS: tl.constexpr,
                         MROWS: tl.constexpr, ASM: tl.constexpr):
        """1-4 rows of int8 x (from _quant_q8_kernel) times Q4_0/Q8_0 weights, read in
        place from the GGUF blocks. Their 16 / 32 weight bytes start 2 bytes into an 18 /
        34-byte block, so they're only 2-byte aligned: loaded as uint16 pairs. As in
        _qgemv4_kernel, a row's arithmetic doesn't depend on M (speculative checks match
        one-token decoding bit for bit)."""
        pid_n = tl.program_id(0)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = rn < N
        w16 = w_ptr.to(tl.pointer_type(tl.uint16))
        row16 = rn.to(tl.int64) * (row_bytes // 2)
        nb = K // 32
        i4 = tl.arange(0, 4)
        a0 = tl.zeros((BLOCK_N, BLOCKS), dtype=tl.float32)
        a1 = tl.zeros((BLOCK_N, BLOCKS), dtype=tl.float32)
        a2 = tl.zeros((BLOCK_N, BLOCKS), dtype=tl.float32)
        a3 = tl.zeros((BLOCK_N, BLOCKS), dtype=tl.float32)
        for b0 in range(0, nb, BLOCKS):
            b = b0 + tl.arange(0, BLOCKS)
            b_ok = b < nb
            ok2 = n_ok[:, None] & b_ok[None, :]
            if QTYPE == 0:
                blk = row16[:, None] + (b * 9)[None, :]           # block start, in uint16s (18 bytes)
            else:
                blk = row16[:, None] + (b * 17)[None, :]          # 34 bytes
            dw = tl.load(w16 + blk, mask=ok2, other=0).to(tl.float16, bitcast=True).to(tl.float32)
            p = w16 + blk[:, :, None] + 1 + 2 * i4[None, None, :]
            m3 = ok2[:, :, None]
            if QTYPE == 0:
                wq = (tl.load(p, mask=m3, other=0).to(tl.uint32)
                      | (tl.load(p + 1, mask=m3, other=0).to(tl.uint32) << 16)).to(tl.int32, bitcast=True)
                w0 = wq & 0x0F0F0F0F                       # low nibbles: values 0-15 of the block
                w1 = (wq >> 4) & 0x0F0F0F0F                # high nibbles: values 16-31
            else:
                w0 = (tl.load(p, mask=m3, other=0).to(tl.uint32)
                      | (tl.load(p + 1, mask=m3, other=0).to(tl.uint32) << 16)).to(tl.int32, bitcast=True)
                w1 = (tl.load(p + 8, mask=m3, other=0).to(tl.uint32)
                      | (tl.load(p + 9, mask=m3, other=0).to(tl.uint32) << 16)).to(tl.int32, bitcast=True)
            a0 = _q8_row(a0, w0, w1, dw, xq_ptr, dx_ptr, sx_ptr, 0, K, b, b_ok, i4, True, QTYPE, ASM)
            if MROWS > 1:
                a1 = _q8_row(a1, w0, w1, dw, xq_ptr, dx_ptr, sx_ptr, 1, K, b, b_ok, i4, M > 1, QTYPE, ASM)
            if MROWS > 2:
                a2 = _q8_row(a2, w0, w1, dw, xq_ptr, dx_ptr, sx_ptr, 2, K, b, b_ok, i4, M > 2, QTYPE, ASM)
                a3 = _q8_row(a3, w0, w1, dw, xq_ptr, dx_ptr, sx_ptr, 3, K, b, b_ok, i4, M > 3, QTYPE, ASM)
        out = y_ptr.dtype.element_ty
        tl.store(y_ptr + rn, tl.sum(a0, axis=1).to(out), mask=n_ok)
        if MROWS > 1:
            tl.store(y_ptr + stride_ym + rn, tl.sum(a1, axis=1).to(out), mask=n_ok & (M > 1))
        if MROWS > 2:
            tl.store(y_ptr + 2 * stride_ym + rn, tl.sum(a2, axis=1).to(out), mask=n_ok & (M > 2))
            tl.store(y_ptr + 3 * stride_ym + rn, tl.sum(a3, axis=1).to(out), mask=n_ok & (M > 3))


GEMV_MAX_ROWS = 4  # up to this many rows of x (tokens), the few-row kernel is used
# (BLOCK_N, BLOCKS, num_warps) for _qgemv4_kernel. Round 7, three Gemma-4-12B-sized matrices:
# Q4_0 1 row 187 us / 4 rows 305 us (previous kernels: 225 / 707); Q8_0 391 / 371 (378 / 686).
GEMV4_CONFIG = {q.Q4_0: (8, 8, 2), q.Q8_0: (4, 8, 2)}
# Same for _qgemv_q8_kernel (int8 activations), measured in CUDA graphs over 600 MB of
# Gemma-sized weights (round 8): Q4_0 1 row 221 -> 213 us, 4 rows 325 -> 263 us. Not used
# for Q8_0: its fp32 kernel is already flat in M (406 / 419 us) and int8 was slower (482).
GEMV_Q8_CONFIG = {q.Q4_0: (16, 16, 4)}

# Few-row matmuls of bf16/fp16 x use int8 activations (llama.cpp's rounding) unless
# SLEUTH_EXACT_ACT=1. fp32 models always keep exact activations.
INT8_ACTIVATIONS = os.environ.get("SLEUTH_EXACT_ACT", "") not in ("1", "true", "yes")


def set_int8_activations(on: bool) -> None:
    """Switch the int8-activation few-row kernel on or off (e.g. to compare)."""
    global INT8_ACTIVATIONS
    INT8_ACTIVATIONS = on


def quantize_q8(x2: torch.Tensor):
    """Rows of x [M, K] -> (int8 values packed 4 per int32 [M, K/4], scale [M, K/32],
    block sums [M, K/32]), rounded exactly like llama.cpp's quantize_q8_1."""
    M, K = x2.shape
    xq = torch.empty(M, K // 4, dtype=torch.int32, device=x2.device)
    dx = torch.empty(M, K // 32, dtype=torch.float32, device=x2.device)
    sx = torch.empty(M, K // 32, dtype=torch.float32, device=x2.device)
    _quant_q8_kernel[(M, K // 32)](x2, xq, dx, sx, K, x2.stride(0), num_warps=1)
    return xq, dx, sx


def qmatmul(x: torch.Tensor, raw: torch.Tensor, ggml_type: int, out_features: int) -> torch.Tensor:
    """x [..., K] @ Wᵀ, where W [out_features, K] is stored as GGUF blocks in `raw` (uint8)."""
    code, nb = SUPPORTED[ggml_type]
    K = x.shape[-1]
    x2 = x.reshape(-1, K).contiguous()
    M, N = x2.shape[0], out_features
    y = torch.empty(M, N, dtype=x.dtype, device=x.device)
    if M <= GEMV_MAX_ROWS and INT8_ACTIVATIONS and x.dtype != torch.float32 and ggml_type in GEMV_Q8_CONFIG:
        xq, dx, sx = quantize_q8(x2)
        block_n, blocks, warps = GEMV_Q8_CONFIG[ggml_type]
        _qgemv_q8_kernel[(triton.cdiv(N, block_n),)](xq, dx, sx, raw, y, M, N, K, y.stride(0), K // 32 * nb,
                                                     QTYPE=code, BLOCK_N=block_n, BLOCKS=blocks,
                                                     MROWS=1 if M == 1 else (2 if M == 2 else 4),
                                                     ASM=os.environ.get("TRITON_INTERPRET") != "1",
                                                     num_warps=warps)
        return y.reshape(*x.shape[:-1], N)
    if M <= GEMV_MAX_ROWS:
        # 1-4 rows (decoding, and speculative checks of up to 3 guesses): one kernel for
        # all of them, so a row's result never depends on how many rows ran together;
        # checked tokens then match normal decoding bit for bit. Configs measured on the
        # RTX 5060 Ti (round 7) and checked for that row independence.
        block_n, blocks, warps = GEMV4_CONFIG[ggml_type]
        _qgemv4_kernel[(triton.cdiv(N, block_n),)](x2, raw, y, M, N, K, x2.stride(0), y.stride(0), K // 32 * nb,
                                                   QTYPE=code, BLOCK_N=block_n, BLOCKS=blocks,
                                                   MROWS=1 if M == 1 else (2 if M == 2 else 4), num_warps=warps)
        return y.reshape(*x.shape[:-1], N)
    block_m = 16 if M <= 16 else 64        # decoding (M = 1) wastes some tensor-core rows; memory is the limit
    block_n, block_k = 64, 64
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _qmatmul_kernel[grid](x2, raw, y, M, N, K, x2.stride(0), y.stride(0), K // 32 * nb,
                          QTYPE=code, IEEE=x.dtype == torch.float32,
                          BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k)
    return y.reshape(*x.shape[:-1], N)


_checked: dict[tuple, bool] = {}


def usable(x: torch.Tensor, ggml_type: int) -> bool:
    """Use the kernel for this input? Needs Triton, a CUDA tensor (or the
    interpreter), a supported type and dtype, and a passing one-time self-check
    against the plain path for this device/type/dtype."""
    if not available() or ggml_type not in SUPPORTED or x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if not x.is_cuda and os.environ.get("TRITON_INTERPRET") != "1":
        return False
    key = (x.device.type, x.device.index, ggml_type, x.dtype)
    if key not in _checked:
        _checked[key] = _self_check(x.device, ggml_type, x.dtype)
    return _checked[key]


def _self_check(device, ggml_type, dtype) -> bool:
    """Compare the kernel with dequantize + matmul on a small random layer.
    Any failure (compiler missing, a Triton bug on this GPU, ...) means we
    warn once and keep using the plain path, which is always correct."""
    try:
        g = torch.Generator().manual_seed(0)
        w = torch.randn(96, 160, generator=g) * 0.05
        raw = q.quantize(w, ggml_type).to(device)
        x = (torch.randn(3, 160, generator=g)).to(device, dtype)            # 3 rows: the few-row kernel
        ref = x.float() @ q.dequantize(raw, ggml_type, w.numel(), dtype).float().reshape(96, 160).T
        tol = 1e-4 if dtype == torch.float32 else 2e-2
        ok = True
        for xs in (x, x.repeat(8, 1)):                                       # 24 rows: the tensor-core kernel
            r = ref if xs is x else ref.repeat(8, 1)
            got = qmatmul(xs, raw, ggml_type, 96).float()
            ok &= bool(torch.allclose(got, r, atol=tol * r.abs().max().item(), rtol=tol))
        if not ok:
            warnings.warn(f"Triton {q.TYPE_NAMES[ggml_type]} kernel gave wrong results on {device}; "
                          f"using the plain (slower) path", stacklevel=3)
        return ok
    except Exception as e:  # noqa: BLE001 - any failure falls back to the plain path
        warnings.warn(f"Triton kernel unavailable on {device} ({type(e).__name__}: {e}); "
                      f"using the plain (slower) path", stacklevel=3)
        return False
