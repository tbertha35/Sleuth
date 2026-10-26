"""Fused RMSNorm / head norm + RoPE kernels (forge/model/fused_ops.py), run in
Triton's interpreter (TRITON_INTERPRET=1 must be set before Triton is imported,
hence a subprocess; the interpreter has no bf16, which test_gpu.py covers)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("triton")

SCRIPT = r"""
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, "tests")
from sleuth.forge.model import fused_ops as fo
from sleuth.forge.model.rope import apply_rope

torch.manual_seed(0)
# 1. the kernels alone, against the eager formulas (odd sizes: masked tails)
for dt, tol in ((torch.float32, 1e-5), (torch.float16, 2e-3)):
    for D in (96, 100, 256):
        x = torch.randn(3, 5, D).to(dt)
        w = (1 + 0.1 * torch.randn(D)).to(dt)
        ref = F.rms_norm(x.float(), (D,), w.float(), 1e-6)
        assert (fo.rms_norm(x, w, 1e-6).float() - ref).abs().max() < tol * 10, (dt, D)
    for hd in (16, 48, 64):
        q = torch.randn(2, 7, 3, hd).to(dt).transpose(1, 2)         # [B, H, T, hd] as Attention makes it
        w = (1 + 0.1 * torch.randn(hd)).to(dt)
        ang = torch.randn(7, hd // 2)
        cos, sin = torch.cat([ang, ang], -1).cos(), torch.cat([ang, ang], -1).sin()
        n = F.rms_norm(q.float(), (hd,), w.float(), 1e-6)
        ref = apply_rope(n, cos, sin)
        got = fo.head_norm_rope(q, w, 1e-6, cos, sin)
        assert got.shape == q.shape and got.stride() == q.stride()
        assert (got.float() - ref).abs().max() < tol * 10, (dt, hd)
        assert (fo.head_norm_rope(q, None, 1e-6).float() - F.rms_norm(q.float(), (hd,), None, 1e-6)).abs().max() < tol * 10
        assert (fo.head_norm_rope(q, None, 0.0, cos, sin, norm=False).float() - apply_rope(q.float(), cos, sin)).abs().max() < tol * 10
    # rows are independent of how many run together (speculative checks stay exact)
    x = torch.randn(4, 96).to(dt)
    w = torch.ones(96).to(dt)
    assert torch.equal(fo.rms_norm(x, w, 1e-6)[:1], fo.rms_norm(x[:1], w, 1e-6))

# 2. whole models: fused (no grad) vs eager (SLEUTH_NO_TRITON=1 switches the kernels off)
calls = {"rms": 0, "head": 0}
_rms, _head = fo.rms_norm, fo.head_norm_rope
def rms_counted(*a, **k):
    calls["rms"] += 1
    return _rms(*a, **k)
def head_counted(*a, **k):
    calls["head"] += 1
    return _head(*a, **k)
fo.rms_norm, fo.head_norm_rope = rms_counted, head_counted

def both(m, ids):
    with torch.no_grad():
        os.environ["SLEUTH_NO_TRITON"] = "1"
        eager = m(ids).logits
        os.environ.pop("SLEUTH_NO_TRITON")
        fused = m(ids).logits
    return eager, fused

from sleuth.forge.model.config import ModelConfig
from sleuth.forge.model.transformer import Transformer
for kw in ({}, {"qk_norm": True, "n_kv_heads": 2}, {"pos_type": "none"}):
    m = Transformer(ModelConfig(vocab_size=64, d_model=64, n_layers=2, n_heads=4, max_seq_len=64, **kw)).eval()
    e, f = both(m, torch.randint(0, 64, (2, 12)))
    assert (e - f).abs().max() < 1e-4, (kw, (e - f).abs().max())

from test_gemma import IDS, make_gemma_gguf, reference_forward
from sleuth.formats.gguf_model import load_gguf_model
import tempfile, pathlib
p = pathlib.Path(tempfile.mkdtemp()) / "g.gguf"
make_gemma_gguf(p)                          # q/k norms with gains, v norm, sliding + full layers
m, _ = load_gguf_model(p, dtype=torch.float32)
e, f = both(m, torch.tensor([IDS]))
assert (e - f).abs().max() < 2e-4, (e - f).abs().max()
assert (f[0].double() - reference_forward(p, IDS)).abs().max() < 2e-4
n_layers = 6
assert calls["head"] >= 3 * n_layers and calls["rms"] >= 4 * n_layers, calls   # the fused path really ran
assert fo._checked and all(fo._checked.values())

# 3. training keeps the eager path (autograd), and still gets gradients
m = Transformer(ModelConfig(vocab_size=64, d_model=64, n_layers=1, n_heads=4, qk_norm=True, max_seq_len=64))
x = torch.randint(0, 64, (1, 8))
assert not fo.usable(m.tok_emb(x))
m(x, x).loss.backward()
assert m.blocks[0].attn.q_norm.weight.grad is not None
print("OK")
"""


def test_fused_kernels_in_interpreter():
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    env.pop("SLEUTH_NO_TRITON", None)
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-c", SCRIPT], cwd=root, env=env, capture_output=True, text=True,
                       timeout=900)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
