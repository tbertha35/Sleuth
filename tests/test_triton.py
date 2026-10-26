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
import sys, torch, triton
sys.path.insert(0, "tests")
from sleuth.formats import gguf_quants as q
from sleuth.forge.model import triton_kernels as tk
from sleuth.forge.model.quant import round_like_q8

# 1. the kernel alone, against dequantize + matmul. fp16 Q4_0 with 1-4 rows uses int8
# activations: its reference is x rounded like llama.cpp's q8_1 (round_like_q8), and
# with exact activations switched off it's the plain reference again.
for int8 in (True, False):
    tk.set_int8_activations(int8)
    for t in (q.Q4_0, q.Q8_0):
        for M in (1, 3, 4, 7, 40):
            for dt, tol in ((torch.float32, 1e-5), (torch.float16, 2e-3)):
                w = torch.randn(150, 288) * 0.05
                raw = q.quantize(w, t)
                x = torch.randn(M, 288).to(dt)
                xr = round_like_q8(x) if int8 and t == q.Q4_0 and M <= 4 and dt != torch.float32 else x
                ref = xr.float() @ q.dequantize(raw, t, w.numel(), torch.float32).to(dt).float().reshape(150, 288).T
                got = tk.qmatmul(x, raw, t, 150).float()
                err = ((got - ref).abs().max() / ref.abs().max()).item()
                assert got.shape == (M, 150) and err < tol, (int8, q.TYPE_NAMES[t], M, dt, err)
tk.set_int8_activations(True)

# 1b. the int8 kernel's pieces: quantize_q8 packs bytes as llama.cpp rounds them, the
# software dp4a (what the interpreter runs instead of PTX) is a signed byte dot product,
# and the Q8_0 branch (not used by qmatmul: slower than the fp32 one on the RTX 5060 Ti)
# is right too.
x = torch.randn(3, 96) * 2
xq, dx, sx = tk.quantize_q8(x)
bytes_ = xq.view(torch.int8).reshape(3, 96).float()           # little-endian: byte i = value i
assert torch.allclose(bytes_ * dx.repeat_interleave(32, 1), round_like_q8(x), atol=1e-6)
assert torch.equal(sx, bytes_.reshape(3, 3, 32).sum(-1))
for t, code in ((q.Q4_0, 0), (q.Q8_0, 1)):
    w = torch.randn(40, 96) * 0.05
    raw = q.quantize(w, t)
    y = torch.empty(3, 40)
    tk._qgemv_q8_kernel[(triton.cdiv(40, 16),)](xq, dx, sx, raw, y, 3, 40, 96, y.stride(0), 96 // 32 * tk.SUPPORTED[t][1],
                                                QTYPE=code, BLOCK_N=16, BLOCKS=2, MROWS=4, ASM=False)
    ref = round_like_q8(x) @ q.dequantize(raw, t, w.numel(), torch.float32).reshape(40, 96).T
    assert ((y - ref).abs().max() / ref.abs().max()).item() < 1e-5, q.TYPE_NAMES[t]

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


@pytest.mark.gpu
def test_few_row_kernel_rows_are_independent():
    """1-4 rows go through one kernel, and a row's result doesn't depend on how many
    rows ran with it (bit for bit, bf16). That's what lets a speculative check of k+1
    tokens give exactly the tokens normal decoding gives."""
    import torch

    from sleuth.formats import gguf_quants as q
    from sleuth.forge.model import triton_kernels as tk

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    torch.manual_seed(0)
    for t in (q.Q4_0, q.Q8_0):
        for N, K in ((1000, 3840), (3840, 1024)):
            raw = q.quantize(torch.randn(N, K) * 0.02, t).cuda()
            x = torch.randn(4, K, device="cuda").to(torch.bfloat16)
            alone = torch.cat([tk.qmatmul(x[r:r + 1], raw, t, N) for r in range(4)])
            for M in (2, 3, 4):
                assert torch.equal(tk.qmatmul(x[:M], raw, t, N), alone[:M]), (q.TYPE_NAMES[t], N, K, M)


def test_plain_path_without_gpu_or_interpreter():
    """On a CPU without the interpreter the kernel is never used (plain path)."""
    import torch

    from sleuth.forge.model import triton_kernels as tk
    from sleuth.formats import gguf_quants as q

    if os.environ.get("TRITON_INTERPRET") == "1":
        pytest.skip("interpreter mode is on in this process")
    assert not tk.usable(torch.randn(2, 64), q.Q4_0)
