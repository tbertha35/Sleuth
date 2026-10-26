"""CUDA-graph decoding (forge/infer/cuda_graph.py) and the static KV cache it uses.

The static path (position as a tensor, a fixed-length cache view with a mask)
must compute exactly what the normal path computes; that part runs on CPU. The
graph itself needs a GPU."""

import pytest
import torch

from sleuth.forge.infer.generate import generate
from sleuth.forge.infer.kv_cache import KVCache, StaticKVCache
from sleuth.forge.infer.sampling import SamplingConfig
from sleuth.forge.model.config import ModelConfig
from sleuth.forge.model.transformer import Transformer


def _model(**kw):
    torch.manual_seed(0)
    cfg = dict(vocab_size=200, d_model=64, n_layers=3, n_heads=4, n_kv_heads=2, max_seq_len=96)
    cfg.update(kw)
    return Transformer(ModelConfig(**cfg)).eval()


@pytest.mark.parametrize("kw", [
    {},                                                                  # plain GQA
    {"layer_types": ["sliding", "full", "sliding"], "sliding_window": 8},  # Gemma-style sliding layers
])
def test_static_decode_matches_normal_decode(kw):
    """Step-by-step decoding with StaticKVCache in static mode (what each CUDA
    graph replays) gives the same logits as the normal KVCache path, including
    when the cache view grows from one length bucket to the next."""
    m = _model(**kw)
    ids = torch.randint(0, 200, (1, 40))
    prompt, rest = ids[:, :5], ids[0, 5:]
    shapes = [(m.cfg.attn_spec(i).kv_heads, m.cfg.attn_spec(i).head_dim) for i in range(m.cfg.n_layers)]
    normal = KVCache(shapes, 1, 96, "cpu", torch.float32)
    static = StaticKVCache(shapes, 1, 96, "cpu", torch.float32)
    with torch.no_grad():
        a = m(prompt, cache=normal).logits[:, -1]
        b = m(prompt, cache=static).logits[:, -1]   # prefill: static is off, so it acts like KVCache
        assert torch.allclose(a, b, atol=1e-5)
        for t in rest:
            tok = t.view(1, 1)
            a = m(tok, cache=normal).logits[:, -1]
            pos = static.pos
            static.length = 16 if pos < 16 else 64      # two "buckets", as GraphDecoder picks them
            static.pos_t.fill_(pos)
            static.static = True
            b = m(tok, cache=static).logits[:, -1]
            static.static = False
            static.pos += 1
            assert torch.allclose(a, b, atol=1e-4), f"position {pos}: {(a - b).abs().max().item()}"


def test_static_mask_and_window():
    c = StaticKVCache([(1, 4)], 1, 32, "cpu", torch.float32)
    c.length, c.static = 16, True
    c.pos_t.fill_(10)
    assert c.attn_mask().view(-1).tolist() == [True] * 11 + [False] * 5
    # sliding window 4: positions 7..10 only (llama.cpp: masked if q - k >= window)
    assert c.attn_mask(4).view(-1).nonzero().view(-1).tolist() == [7, 8, 9, 10]


@pytest.mark.gpu
def test_cuda_graph_generation_matches_eager():
    """Greedy tokens with CUDA graphs = without, across two length buckets (64 and
    96), and the recorded graphs are reused by the next generate() call."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    m = _model(layer_types=["sliding", "full", "sliding"], sliding_window=8).cuda()
    greedy = SamplingConfig(temperature=0)
    prompt = [1, 5, 9, 13]
    eager = generate(m, prompt, 80, greedy)
    graph = generate(m, prompt, 80, greedy, cuda_graph=True)
    assert graph.new_ids == eager.new_ids
    assert graph.stats["cuda_graphs"] == [64, 96]
    again = generate(m, prompt, 80, greedy, cuda_graph=True)  # graphs reused: no re-recording
    assert again.new_ids == eager.new_ids and m._graph_decoder.graphs.keys() == {64, 96}


def test_static_multi_token_step_matches_normal():
    """A static step of T tokens (what speculative decoding's verify graph runs)
    gives the logits of the normal path for all T positions."""
    m = _model(layer_types=["sliding", "full", "sliding"], sliding_window=8)
    ids = torch.randint(0, 200, (1, 30))
    shapes = [(m.cfg.attn_spec(i).kv_heads, m.cfg.attn_spec(i).head_dim) for i in range(m.cfg.n_layers)]
    with torch.no_grad():
        full = m(ids).logits[0]
        st = StaticKVCache(shapes, 1, 96, "cpu", torch.float32, tokens_per_step=5)
        m(ids[:, :20], cache=st)                  # prefill 20 tokens normally
        st.length, st.static = 32, True
        st.pos_t.fill_(20)
        out = m(ids[:, 20:25], cache=st).logits[0]
        st.static = False
    assert torch.allclose(out, full[20:25], atol=1e-4)
