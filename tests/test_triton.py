"""The fused Triton dequantize+matmul kernel (forge/model/triton_kernels.py).

On a machine without a GPU, Triton can run kernels in its interpreter
(TRITON_INTERPRET=1, set before Triton is imported, hence a subprocess). The
interpreter has no bfloat16, so bf16 is checked by the GPU test in test_gpu.py.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("triton")

SCRIPT = r"""
import sys, torch
sys.path.insert(0, "tests")
from sleuth.formats import gguf_quants as q
from sleuth.forge.model import triton_kernels as tk

# 1. the kernel alone, against dequantize + matmul
for t in (q.Q4_0, q.Q8_0):
    for M in (1, 7, 40):
        for dt, tol in ((torch.float32, 1e-5), (torch.float16, 2e-3)):
            w = torch.randn(150, 288) * 0.05
            raw = q.quantize(w, t)
            x = torch.randn(M, 288).to(dt)
            ref = x.float() @ q.dequantize(raw, t, w.numel(), torch.float32).to(dt).float().reshape(150, 288).T
            got = tk.qmatmul(x, raw, t, 150).float()
            err = ((got - ref).abs().max() / ref.abs().max()).item()
            assert got.shape == (M, 150) and err < tol, (q.TYPE_NAMES[t], M, dt, err)

# 2. a whole kept-quantized model: fused path == plain path
from test_gemma import IDS, make_gemma_gguf
from sleuth.formats.gguf_model import load_gguf_model
import tempfile, pathlib, os
d = pathlib.Path(tempfile.mkdtemp())
make_gemma_gguf(d / "g.gguf", quant=q.Q4_0)
m, _ = load_gguf_model(d / "g.gguf", dtype=torch.float32, keep_quantized=True)
x = torch.tensor([IDS])
with torch.no_grad():
    fused = m(x).logits
    assert tk.usable(x.float(), q.Q4_0)
    os.environ["SLEUTH_NO_TRITON"] = "1"
    assert not tk.usable(x.float(), q.Q4_0)
    plain = m(x).logits
assert (fused - plain).abs().max().item() < 1e-4, (fused - plain).abs().max().item()
print("OK")
"""


def test_kernel_in_triton_interpreter():
    env = {**os.environ, "TRITON_INTERPRET": "1"}
    env.pop("SLEUTH_NO_TRITON", None)
    r = subprocess.run([sys.executable, "-c", SCRIPT], cwd=Path(__file__).resolve().parent.parent, env=env,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and r.stdout.strip().endswith("OK"), r.stdout[-2000:] + r.stderr[-4000:]


def test_plain_path_without_gpu_or_interpreter():
    """On a CPU without the interpreter the kernel is never used (plain path)."""
    import torch

    from sleuth.forge.model import triton_kernels as tk
    from sleuth.formats import gguf_quants as q

    if os.environ.get("TRITON_INTERPRET") == "1":
        pytest.skip("interpreter mode is on in this process")
    assert not tk.usable(torch.randn(2, 64), q.Q4_0)
