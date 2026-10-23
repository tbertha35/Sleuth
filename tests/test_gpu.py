"""GPU smoke tests. Skipped by default; run on a CUDA machine with:  pytest -m gpu -q

These exercise the code paths the CPU tests can't: bf16 autocast, fused
AdamW, CUDA timing, the bf16 KV cache, and the attention kernels this
PyTorch build picks for your GPU.
"""

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
    run = train(cfg, log=lambda s: None)
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
