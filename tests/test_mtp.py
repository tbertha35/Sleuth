"""MTP drafters (forge/infer/mtp.py): a tiny Gemma-4-assistant-shaped drafter on a
tiny Gemma-style target. Whatever the drafter guesses, speculative decoding must
give exactly the target's own greedy tokens; the drafter must only read the
target's cache before its position. The real files are checked on the GPU box."""

import math

import pytest
import torch
from torch import nn

from sleuth.forge.infer.generate import generate
from sleuth.forge.infer.kv_cache import KVCache
from sleuth.forge.infer.mtp import AssistantLayer, Gemma4Assistant, MTPDrafter
from sleuth.forge.infer.sampling import SamplingConfig
from sleuth.forge.infer.speculative import speculative_generate
from sleuth.forge.model.config import ModelConfig
from sleuth.forge.model.norm import RMSNorm
from sleuth.forge.model.rope import rope_tables
from sleuth.forge.model.transformer import Transformer

D_T, D, V = 32, 16, 64


def _target():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=V, d_model=D_T, n_layers=4, n_heads=4, n_kv_heads=1, head_size=16,
                      layer_types=["sliding", "sliding", "sliding", "full"], sliding_window=6,
                      swa_head_size=8, swa_kv_heads=2, embed_scale=math.sqrt(D_T), post_norms=True,
                      qk_norm=True, max_seq_len=64, pad_vocab_to=1)
    return Transformer(cfg).eval()


def _assistant(target):
    """A random drafter with gemma4-assistant's structure, sized for `target`."""
    torch.manual_seed(1)
    m = Gemma4Assistant()
    m.n_heads, m.d, m.d_target, m.window = 4, D, D_T, target.cfg.sliding_window
    for kind, hd in (("full", 16), ("sliding", 8)):
        cos, sin = rope_tables(hd, 64, 10000.0)
        m.register_buffer(f"cos_{kind}", cos, persistent=False)
        m.register_buffer(f"sin_{kind}", sin, persistent=False)
    m.pre_proj, m.post_proj = nn.Linear(2 * D_T, D, bias=False), nn.Linear(D, D_T, bias=False)
    m.out_norm, m.lm_head = RMSNorm(D), nn.Linear(D, V, bias=False)
    m.layers = nn.ModuleList()
    for sliding in (True, True, False):
        hd = 8 if sliding else 16
        lay = AssistantLayer(sliding, 4, hd, m.window, 1e-6)
        lay.attn_norm, lay.q_norm, lay.post_attn_norm = RMSNorm(D), RMSNorm(hd), RMSNorm(D)
        lay.q_proj, lay.o_proj = nn.Linear(D, 4 * hd, bias=False), nn.Linear(4 * hd, D, bias=False)
        lay.ffn_norm, lay.post_ffn_norm = RMSNorm(D), RMSNorm(D)
        lay.ffn_gate, lay.ffn_up, lay.ffn_down = (nn.Linear(D, 32, bias=False), nn.Linear(D, 32, bias=False),
                                                 nn.Linear(32, D, bias=False))
        lay.register_buffer("scale", torch.ones(1), persistent=False)
        m.layers.append(lay)
    return m.eval()


def test_mtp_speculative_is_exact():
    t = _target()
    drafter = MTPDrafter(_assistant(t))
    prompt = [1, 5, 9, 13, 2, 7]
    greedy = SamplingConfig(temperature=0)
    plain = generate(t, prompt, 40, greedy)
    spec = speculative_generate(t, prompt, 40, greedy, draft=drafter, k=3)
    assert spec.new_ids == plain.new_ids
    s = spec.stats["speculative"]
    assert s["drafted"] > 0 and s["forwards"] < 40


def test_mtp_reads_only_the_targets_cache_before_its_position():
    t = _target()
    m = _assistant(t)
    shapes = [(t.cfg.attn_spec(i).kv_heads, t.cfg.attn_spec(i).head_dim) for i in range(t.cfg.n_layers)]
    cache = KVCache(shapes, 1, 64, "cpu", torch.float32)
    with torch.no_grad():
        h = t(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]), cache=cache, return_hidden=True).hidden[0, -1]
        a, ha = m(t, cache, torch.tensor([11]), h.view(1, -1), cache.pos)
        for k in cache.k:
            k[:, :, cache.pos:] = 1e4      # positions the target hasn't filled yet
        b, hb = m(t, cache, torch.tensor([11]), h.view(1, -1), cache.pos)
    assert torch.allclose(a, b) and torch.allclose(ha, hb)
    assert a.shape == (V,) and ha.shape == (1, D_T)


@pytest.mark.gpu
def test_mtp_with_cuda_graph_is_exact():
    """The verify step as a CUDA graph that also outputs the hidden states."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    t = _target().cuda()
    drafter = MTPDrafter(_assistant(t).cuda())
    prompt = [1, 5, 9, 13, 2, 7]
    greedy = SamplingConfig(temperature=0)
    plain = generate(t, prompt, 50, greedy)
    spec = speculative_generate(t, prompt, 50, greedy, draft=drafter, k=3, cuda_graph=True)
    assert spec.new_ids == plain.new_ids and spec.stats["cuda_graphs"]
    # The drafter's own graphs guess exactly what the drafter guesses without them.
    eager = MTPDrafter(drafter.mtp, cuda_graph=False)
    spec2 = speculative_generate(t, prompt, 50, greedy, draft=eager, k=3, cuda_graph=True)
    assert spec2.new_ids == plain.new_ids
    assert spec2.stats["speculative"]["accepted"] == spec.stats["speculative"]["accepted"]
    assert drafter._graphs is not None and drafter._graphs.graphs


def test_mtp_checks_the_target():
    t = _target()
    m = _assistant(t)
    m.d_target = 99
    with pytest.raises(ValueError, match="d_model"):
        m.check_target(t)
