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


def qmatmul(x: torch.Tensor, raw: torch.Tensor, ggml_type: int, out_features: int) -> torch.Tensor:
    """x [..., K] @ Wᵀ, where W [out_features, K] is stored as GGUF blocks in `raw` (uint8)."""
    code, nb = SUPPORTED[ggml_type]
    K = x.shape[-1]
    x2 = x.reshape(-1, K).contiguous()
    M, N = x2.shape[0], out_features
    y = torch.empty(M, N, dtype=x.dtype, device=x.device)
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
        x = (torch.randn(3, 160, generator=g)).to(device, dtype)
        ref = x.float() @ q.dequantize(raw, ggml_type, w.numel(), dtype).float().reshape(96, 160).T
        got = qmatmul(x, raw, ggml_type, 96).float()
        tol = 1e-4 if dtype == torch.float32 else 2e-2
        ok = bool(torch.allclose(got, ref, atol=tol * ref.abs().max().item(), rtol=tol))
        if not ok:
            warnings.warn(f"Triton {q.TYPE_NAMES[ggml_type]} kernel gave wrong results on {device}; "
                          f"using the plain (slower) path", stacklevel=3)
        return ok
    except Exception as e:  # noqa: BLE001 - any failure falls back to the plain path
        warnings.warn(f"Triton kernel unavailable on {device} ({type(e).__name__}: {e}); "
                      f"using the plain (slower) path", stacklevel=3)
        return False
