"""GPU smoke tests. Skipped by default; run on a CUDA machine with:  pytest -m gpu -q

These exercise the code paths the CPU tests can't: bf16 autocast, fused
AdamW, CUDA timing, the bf16 KV cache, and the attention kernels this
PyTorch build picks for your GPU.
"""

import dataclasses

import pytest
import torch
from helpers import tiny_run_config

from sleuth.forge.infer.generate import generate
from sleuth.forge.infer.kv_cache import KVCache
from sleuth.forge.infer.sampling import SamplingConfig
from sleuth.forge.model.config import ModelConfig
from sleuth.forge.model.transformer import Transformer
from sleuth.forge.train.checkpoint import load_model
from sleuth.forge.train.trainer import train

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]


def test_bf16_training_and_generation(tiny_data, tmp_path):
    cfg = tiny_run_config(tiny_data, tmp_path, device="cuda", precision="bf16", max_steps=20, lr=3e-3)
    before = torch.get_float32_matmul_precision()
    run = train(cfg, log=lambda s: None)
    assert torch.get_float32_matmul_precision() == before  # TF32 must not leak out of train()
    model, _ = load_model(run, device="cuda")
    r = generate(model, [1, 2, 3], 20, SamplingConfig(temperature=0.8), seed=0)
    assert len(r.new_ids) > 0 and r.decode_tok_per_s > 0


def test_bf16_kv_cache_close_to_full_forward():
    torch.manual_seed(0)
    m = Transformer(ModelConfig(vocab_size=300, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
                                max_seq_len=64)).cuda().eval()
    x = torch.randint(0, 300, (1, 32), device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        full = m(x).logits.float()
        cache = KVCache.for_model(m, 1, dtype=torch.bfloat16)
        parts = [m(x[:, :16], cache=cache).logits.float()]
        parts += [m(x[:, i:i + 1], cache=cache).logits.float() for i in range(16, 32)]
    # bf16 has ~3 significant digits, so this is a loose check.
    assert (torch.cat(parts, 1) - full).abs().max().item() < 0.1


def _has_triton() -> bool:
    try:
        import triton  # noqa: F401  (torch.compile's GPU code generator)
        return True
    except ImportError:
        return False


@pytest.mark.parametrize("compile_model", [
    False,
    pytest.param(True, marks=pytest.mark.skipif(not _has_triton(), reason="torch.compile needs Triton "
                                                "(not bundled with PyTorch on Windows)")),
])
def test_compile_optional(tiny_data, tmp_path, compile_model):
    cfg = tiny_run_config(tiny_data, tmp_path, device="cuda", max_steps=4, compile=compile_model)
    train(cfg, log=lambda s: None)


def test_quantized_gemma_on_gpu(tmp_path):
    """Gemma 4-style model kept as Q4_0 blocks in VRAM: same logits as on the CPU
    (fp32), and generation with the bf16 KV cache works."""
    from test_gemma import IDS, make_gemma_gguf

    from sleuth.formats import gguf_quants as q
    from sleuth.formats.gguf_model import load_gguf_model

    path = tmp_path / "tiny-q4.gguf"
    make_gemma_gguf(path, quant=q.Q4_0)
    cpu, _ = load_gguf_model(path, dtype=torch.float32, keep_quantized=True)
    gpu, info = load_gguf_model(path, device="cuda", dtype=torch.float32, keep_quantized=True)
    assert info["kept_quantized"] > 0 and gpu.tok_emb.raw.is_cuda
    x = torch.tensor([IDS])
    with torch.no_grad():
        torch.testing.assert_close(gpu(x.cuda()).logits.cpu(), cpu(x).logits, atol=1e-3, rtol=1e-3)
    bf16, _ = load_gguf_model(path, device="cuda", dtype=torch.bfloat16, keep_quantized=True)
    r = generate(bf16, IDS[:5], 20, SamplingConfig(temperature=0.0))
    assert len(r.new_ids) == 20


def test_triton_kernel_on_gpu(tmp_path, monkeypatch):
    """The fused Triton kernel (bf16) agrees with the plain dequantize path, and
    is actually used. Skipped if Triton isn't installed (Windows: triton-windows)."""
    pytest.importorskip("triton")
    from test_gemma import IDS, make_gemma_gguf

    from sleuth.forge.model import fused_ops
    from sleuth.forge.model import triton_kernels as tk
    from sleuth.formats import gguf_quants as q
    from sleuth.formats.gguf_model import load_gguf_model

    # SLEUTH_NO_TRITON also switches off the fused norm/RoPE kernels, whose (better)
    # rounding this tiny random model magnifies ~70x over 6 layers. Keep them off in
    # both runs, so the only difference is the matmul kernel under test.
    monkeypatch.setattr(fused_ops, "usable", lambda x: False)
    path = tmp_path / "tiny-q4.gguf"
    make_gemma_gguf(path, quant=q.Q4_0)
    model, _ = load_gguf_model(path, device="cuda", dtype=torch.bfloat16, keep_quantized=True)
    x = torch.tensor([IDS], device="cuda")
    assert tk.usable(torch.zeros(1, 64, device="cuda", dtype=torch.bfloat16), q.Q4_0), "Triton self-check failed"
    with torch.no_grad():
        fused = model(x).logits.float()
        monkeypatch.setenv("SLEUTH_NO_TRITON", "1")
        plain = model(x).logits.float()
    # Same weights, same rounding; only the order of the fp32 sums differs.
    assert (fused - plain).abs().max().item() < 0.05 * plain.abs().max().item()
    assert (fused.argmax(-1) == plain.argmax(-1)).float().mean().item() > 0.9


@pytest.mark.parametrize("compile_model", [
    False,
    pytest.param(True, marks=pytest.mark.skipif(not _has_triton(), reason="torch.compile needs Triton")),
])
@pytest.mark.parametrize("kernel", ["sdpa", "flex"])
def test_doc_mask_on_gpu(tiny_data, tmp_path, compile_model, kernel):
    """Document-masked attention under bf16 autocast (and torch.compile): it
    trains, and a masked second document matches the same text on its own."""
    from sleuth.forge.data.loader import document_ids
    from sleuth.forge.model.attention import doc_block_mask

    if kernel == "flex" and not compile_model:
        pytest.skip("FlexAttention without compile is only the slow reference version (CPU-tested)")
    cfg = tiny_run_config(tiny_data, tmp_path, device="cuda", max_steps=4, doc_mask=True, compile=compile_model,
                          doc_mask_kernel=kernel)
    # Compiled FlexAttention needs heads >= 16 wide; the tiny config has 8.
    cfg.model = dataclasses.replace(cfg.model, d_model=64)
    train(cfg, log=lambda s: None)
    torch.manual_seed(0)
    m = Transformer(ModelConfig(vocab_size=300, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
                                max_seq_len=64)).cuda().eval()
    b = torch.randint(2, 300, (1, 20), device="cuda")
    b[0, 0] = 1
    both = torch.cat([torch.randint(2, 300, (1, 12), device="cuda"), b], 1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        alone = m(b).logits.float()
        doc = document_ids(both, 1)
        masked = m(both, doc_ids=doc).logits[:, 12:].float()
        flex = m(both, block_mask=doc_block_mask(doc)).logits[:, 12:].float() if kernel == "flex" else masked
    assert (masked - alone).abs().max().item() < 0.1   # bf16: a loose check
    assert (flex - alone).abs().max().item() < 0.1


def test_fused_norm_rope_on_gpu(tmp_path, monkeypatch):
    """On the GPU (the interpreter test can't do bf16): the fused kernels match the
    eager path in fp32, are at least as precise as it in bf16, and greedy decoding
    with CUDA graphs still equals plain decoding.

    Why not compare whole bf16 models, fused vs eager? The eager path rounds q/k to
    bf16 between norm and RoPE, the fused kernel doesn't, and this tiny random model
    magnifies that one rounding ~70x over 6 layers: two equally valid bf16 results
    can differ by 15-45% of the logits. So the bf16 check is on the kernel itself."""
    import sys

    import torch.nn.functional as F

    sys.path.insert(0, "tests")
    from test_gemma import IDS, make_gemma_gguf

    from sleuth.forge.model import fused_ops
    from sleuth.forge.model.rope import apply_rope, rope_tables
    from sleuth.formats import gguf_quants as q
    from sleuth.formats.gguf_model import load_gguf_model

    # 1. whole model in fp32: the fused path computes the same thing (seen: 3e-5)
    make_gemma_gguf(tmp_path / "g.gguf", quant=q.Q4_0)
    x = torch.tensor([IDS], device="cuda")
    m32, _ = load_gguf_model(tmp_path / "g.gguf", device="cuda", dtype=torch.float32, keep_quantized=True)
    with torch.no_grad():
        fused = m32(x).logits.float()
        assert fused_ops._checked.get(("cuda", 0, torch.float32)) is True
        monkeypatch.setenv("SLEUTH_NO_TRITON", "1")
        eager = m32(x).logits.float()
        monkeypatch.delenv("SLEUTH_NO_TRITON")
    assert (fused - eager).abs().max() < 1e-3 * eager.abs().max()

    # 2. bf16 at a real head size (Gemma 4: 256): fused is closer to exact fp32 math
    # than eager, which rounds twice (seen: RMS error 0.0017 vs 0.0024)
    torch.manual_seed(0)
    hd, T = 256, 256
    cos, sin = (t.cuda() for t in rope_tables(hd, T, 1e6))
    qh = (3 * torch.randn(1, T, 8, hd, device="cuda")).bfloat16().transpose(1, 2)
    w = (1 + 0.3 * torch.randn(hd, device="cuda")).bfloat16()
    exact = apply_rope(F.rms_norm(qh.float(), (hd,), w.float(), 1e-6), cos, sin)
    eager = apply_rope(F.rms_norm(qh.float(), (hd,), w.float(), 1e-6).bfloat16(), cos, sin).float()
    fused = fused_ops.head_norm_rope(qh, w, 1e-6, cos, sin).float()
    assert fused_ops._checked.get(("cuda", 0, torch.bfloat16)) is not False
    rms = lambda t: (t - exact).pow(2).mean().sqrt().item()  # noqa: E731
    assert rms(fused) <= rms(eager)

    # 3. bf16 decoding: CUDA graphs (fused kernels captured) give the same text
    m, _ = load_gguf_model(tmp_path / "g.gguf", device="cuda", dtype=torch.bfloat16, keep_quantized=True)
    greedy = SamplingConfig(temperature=0)
    a = generate(m, IDS[:5], 20, greedy)
    b = generate(m, IDS[:5], 20, greedy, cuda_graph=True)
    assert a.new_ids == b.new_ids
