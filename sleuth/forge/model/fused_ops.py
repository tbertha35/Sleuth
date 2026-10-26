"""Fused Triton kernels for the small per-layer steps of decoding: RMSNorm, and
q/k/v head norms + RoPE. Inference only (no autograd); the eager PyTorch code
stays the reference and the training path.

Why: decoding one token is ~25 ms for Gemma 4 12B, and the big matmuls are only
part of it. Each RMSNorm, done in fp32 the PyTorch way, is ~9 tiny GPU kernels
(cast x, cast the weight, square, mean, add eps, rsqrt, two multiplies, cast back);
RoPE is ~7 more per tensor. Gemma 4 runs 6 norms and 2 RoPEs in each of its 48
layers: thousands of kernels per token, each costing a few microseconds even
inside a CUDA graph. llama.cpp runs one kernel for each. These do the same:

    rms_norm(x, w, eps)          one kernel: y = x / sqrt(mean(x²) + eps) * w
    head_norm_rope(x, ...)       one kernel per q / k / v tensor: the per-head
                                 RMSNorm (optional gain) then NEOX RoPE (optional)

The math is fp32 throughout, like the eager path. One difference: the eager path
rounds the normalized q/k to bf16 before RoPE; the fused kernel doesn't (one
rounding instead of two, as llama.cpp, which keeps both steps in f32). Each row is
computed by its own program, so a row's result never depends on how many rows ran
together: speculative checks stay bit-identical to normal decoding.

Used when: Triton is available, the tensor is on a CUDA device (or TRITON_INTERPRET=1
for tests), autograd is off, we're not inside torch.compile, and a one-time
self-check against the eager code passed. Otherwise the eager code runs.
SLEUTH_NO_TRITON=1 turns these off too (triton_kernels.available).
"""

from __future__ import annotations

import os
import warnings

import torch
import torch.nn.functional as F

from .triton_kernels import available, triton

if triton is not None:
    import triton.language as tl

    @triton.jit
    def _rms_kernel(x_ptr, w_ptr, y_ptr, D, stride_x, stride_y, eps,
                    HAS_W: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        ok = cols < D
        x = tl.load(x_ptr + row * stride_x + cols, mask=ok, other=0.0).to(tl.float32)
        y = x * tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
        if HAS_W:
            y = y * tl.load(w_ptr + cols, mask=ok, other=0.0).to(tl.float32)
        tl.store(y_ptr + row * stride_y + cols, y.to(y_ptr.dtype.element_ty), mask=ok)

    @triton.jit
    def _head_kernel(x_ptr, w_ptr, cos_ptr, sin_ptr, y_ptr, H, T, HALF,
                     sxb, sxh, sxt, syb, syh, syt, s_cos, eps,
                     NORM: tl.constexpr, HAS_W: tl.constexpr, ROPE: tl.constexpr, BLOCK: tl.constexpr):
        # One program per (batch, head, position) row of a [B, H, T, D] tensor. The two
        # halves of the row are handled separately: NEOX RoPE pairs channel i with i + D/2.
        pid = tl.program_id(0).to(tl.int64)
        t = pid % T
        h = (pid // T) % H
        b = pid // (T * H)
        c = tl.arange(0, BLOCK)
        ok = c < HALF
        xr = x_ptr + b * sxb + h * sxh + t * sxt
        a = tl.load(xr + c, mask=ok, other=0.0).to(tl.float32)            # channels 0 .. D/2-1
        z = tl.load(xr + HALF + c, mask=ok, other=0.0).to(tl.float32)     # channels D/2 .. D-1
        if NORM:
            r = tl.rsqrt((tl.sum(a * a, axis=0) + tl.sum(z * z, axis=0)) / (2 * HALF) + eps)
            a = a * r
            z = z * r
            if HAS_W:
                a = a * tl.load(w_ptr + c, mask=ok, other=0.0).to(tl.float32)
                z = z * tl.load(w_ptr + HALF + c, mask=ok, other=0.0).to(tl.float32)
        if ROPE:
            cr = cos_ptr + t * s_cos
            sr = sin_ptr + t * s_cos
            c0 = tl.load(cr + c, mask=ok, other=0.0).to(tl.float32)
            c1 = tl.load(cr + HALF + c, mask=ok, other=0.0).to(tl.float32)
            s0 = tl.load(sr + c, mask=ok, other=0.0).to(tl.float32)
            s1 = tl.load(sr + HALF + c, mask=ok, other=0.0).to(tl.float32)
            a, z = a * c0 - z * s0, z * c1 + a * s1                    # x·cos + rotate_half(x)·sin
        yr = y_ptr + b * syb + h * syh + t * syt
        tl.store(yr + c, a.to(y_ptr.dtype.element_ty), mask=ok)
        tl.store(yr + HALF + c, z.to(y_ptr.dtype.element_ty), mask=ok)


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """RMSNorm over the last dimension, output in x's dtype (one kernel)."""
    D = x.shape[-1]
    x2 = x.reshape(-1, D)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    y = torch.empty(x2.shape, dtype=x.dtype, device=x.device)
    w = weight if weight is not None else x2      # (unused when HAS_W is False)
    _rms_kernel[(x2.shape[0],)](x2, w, y, D, x2.stride(0), y.stride(0), eps,
                                HAS_W=weight is not None, BLOCK=triton.next_power_of_2(D))
    return y.view(x.shape)


def head_norm_rope(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6,
                   cos: torch.Tensor | None = None, sin: torch.Tensor | None = None,
                   norm: bool = True) -> torch.Tensor:
    """x [B, H, T, D] (any strides with a contiguous last dimension): per-head RMSNorm
    (if norm; with gain `weight` if given), then RoPE with cos/sin [T, D] (if given).
    Returns a new tensor with x's layout."""
    B, H, T, D = x.shape
    if x.stride(-1) != 1:
        x = x.contiguous()
    y = torch.empty_like(x)                          # same strides as x (e.g. the transposed q view)
    rope = cos is not None
    if rope:
        cos, sin = cos.contiguous(), sin.contiguous()
    dummy = x
    _head_kernel[(B * H * T,)](x, weight if weight is not None else dummy, cos if rope else dummy,
                               sin if rope else dummy, y, H, T, D // 2,
                               x.stride(0), x.stride(1), x.stride(2), y.stride(0), y.stride(1), y.stride(2),
                               cos.stride(0) if rope else 0, eps,
                               NORM=norm, HAS_W=weight is not None, ROPE=rope,
                               BLOCK=triton.next_power_of_2(D // 2))
    return y


# --------------------------------------------------------------------------- when to use them
_checked: dict[tuple, bool] = {}


def usable(x: torch.Tensor) -> bool:
    """Run the fused kernels for this tensor? See the module docstring."""
    if not available() or x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if not x.is_cuda and os.environ.get("TRITON_INTERPRET") != "1":
        return False
    if torch.is_grad_enabled() and x.requires_grad:
        return False
    if torch.compiler.is_compiling():
        return False
    key = (x.device.type, x.device.index, x.dtype)
    if key not in _checked:
        _checked[key] = _self_check(x.device, x.dtype)
    return _checked[key]


def _self_check(device, dtype) -> bool:
    """Compare both kernels with the eager formulas on small random inputs."""
    try:
        g = torch.Generator().manual_seed(0)
        tol = 1e-5 if dtype == torch.float32 else 2e-2

        def close(a, b):
            return bool(torch.allclose(a.float(), b.float(), atol=tol, rtol=tol))

        x = torch.randn(5, 96, generator=g).to(device, dtype)
        w = (1 + 0.1 * torch.randn(96, generator=g)).to(device, dtype)
        ok = close(rms_norm(x, w, 1e-6), F.rms_norm(x.float(), (96,), w.float(), 1e-6))
        ok &= close(rms_norm(x, None, 1e-6), F.rms_norm(x.float(), (96,), None, 1e-6))
        q = torch.randn(2, 5, 3, 64, generator=g).to(device, dtype).transpose(1, 2)   # [B, H, T, D], strided
        wq = (1 + 0.1 * torch.randn(64, generator=g)).to(device, dtype)
        ang = torch.randn(5, 32, generator=g)
        ang = torch.cat([ang, ang], -1)
        cos, sin = ang.cos().to(device), ang.sin().to(device)
        n = F.rms_norm(q.float(), (64,), wq.float(), 1e-6)
        rot = torch.cat([-n[..., 32:], n[..., :32]], -1)
        ok &= close(head_norm_rope(q, wq, 1e-6, cos, sin), n * cos + rot * sin)
        ok &= close(head_norm_rope(q, None, 1e-6), F.rms_norm(q.float(), (64,), None, 1e-6))
        if not ok:
            warnings.warn(f"fused norm/RoPE kernels gave wrong results on {device}; using the eager path",
                          stacklevel=3)
        return ok
    except Exception as e:  # noqa: BLE001 - any failure falls back to the eager path
        warnings.warn(f"fused norm/RoPE kernels unavailable on {device} ({type(e).__name__}: {e}); "
                      f"using the eager path", stacklevel=3)
        return False
