import pytest
import torch

from sleuth.forge.infer.kv_cache import KVCache
from sleuth.forge.model.config import ModelConfig
from sleuth.forge.model.flops import param_counts, train_flops_per_token
from sleuth.forge.model.taps import listen, tap_names
from sleuth.forge.model.transformer import Transformer

VARIANTS = [
    dict(),
    dict(n_kv_heads=2, qk_norm=True),
    dict(n_kv_heads=1),
    dict(ffn_type="gelu", norm_type="layernorm", bias=True, tie_embeddings=False),
    dict(pos_type="none"),
]


def make(**kw):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=300, d_model=64, n_layers=3, n_heads=4, max_seq_len=48, **kw)
    return Transformer(cfg).eval()


@pytest.mark.parametrize("kw", VARIANTS)
def test_shapes_and_loss(kw):
    m = make(**kw)
    x = torch.randint(0, 300, (2, 16))
    # Random targets, not x itself: with tied embeddings an untrained model
    # already favours "repeat the input token", which would lower the loss.
    out = m(x, targets=torch.randint(0, 300, (2, 16)))
    assert out.logits.shape == (2, 16, m.cfg.padded_vocab)
    assert m.cfg.padded_vocab % 64 == 0 and m.cfg.padded_vocab >= 300
    # An untrained model should be near uniform: loss ≈ ln(vocab).
    assert abs(out.loss.item() - torch.log(torch.tensor(float(m.cfg.padded_vocab))).item()) < 1.0


@pytest.mark.parametrize("kw", VARIANTS)
def test_param_count_matches_formula(kw):
    m = make(**kw)
    assert m.num_params() == param_counts(m.cfg)["total"]


@pytest.mark.parametrize("kw", VARIANTS)
def test_explicit_attention_matches_fused(kw):
    m = make(**kw)
    x = torch.randint(0, 300, (2, 20))
    fused = m(x).logits
    with listen(m, {"blocks.*.attn.pattern": lambda name, t: None}):
        explicit = m(x).logits
    torch.testing.assert_close(fused, explicit, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("kw", VARIANTS)
def test_kv_cache_matches_full_forward(kw):
    m = make(**kw)
    x = torch.randint(0, 300, (2, 24))
    full = m(x).logits
    cache = KVCache.for_model(m, batch=2)
    parts = [m(x[:, :9], cache=cache).logits, m(x[:, 9:15], cache=cache).logits]  # chunked prefill
    parts += [m(x[:, i:i + 1], cache=cache).logits for i in range(15, 24)]         # token-by-token
    torch.testing.assert_close(torch.cat(parts, 1), full, atol=1e-5, rtol=1e-4)


def test_causality():
    """Changing a future token must not change earlier logits."""
    m = make()
    x = torch.randint(0, 300, (1, 16))
    y = x.clone()
    y[0, 10] = (y[0, 10] + 1) % 300
    torch.testing.assert_close(m(x).logits[:, :10], m(y).logits[:, :10])


def test_attention_pattern_rows_sum_to_one():
    m = make()
    seen = {}
    with listen(m, {"blocks.1.attn.pattern": lambda n, t: seen.setdefault(n, t)}):
        m(torch.randint(0, 300, (1, 12)))
    p = seen["blocks.1.attn.pattern"]
    assert p.shape == (1, 4, 12, 12)
    torch.testing.assert_close(p.sum(-1), torch.ones(1, 4, 12))
    assert torch.all(p.triu(1) == 0)  # nothing attends to the future


def test_tap_names_and_ablation():
    m = make()
    names = tap_names(m)
    for expected in ("embed", "blocks.0.mlp.act", "blocks.2.resid_post", "blocks.1.attn.z", "final_norm"):
        assert expected in names
    x = torch.randint(0, 300, (1, 8))
    base = m(x).logits
    with listen(m, {"blocks.*.mlp.act": lambda n, t: torch.zeros_like(t)}):
        ablated = m(x).logits
    assert not torch.allclose(base, ablated)
    # Listeners are removed afterwards, even though they were attached by pattern.
    torch.testing.assert_close(m(x).logits, base)


def test_unembed_of_last_resid_equals_logits():
    m = make()
    x = torch.randint(0, 300, (1, 8))
    seen = {}
    with listen(m, {"blocks.2.resid_post": lambda n, t: seen.setdefault("r", t)}):
        logits = m(x).logits
    torch.testing.assert_close(m.unembed(seen["r"]), logits)


def test_bad_configs_rejected():
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=10, d_model=64, n_heads=5)
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=10, d_model=64, n_heads=4, n_kv_heads=3)
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=10, ffn_type="relu")


def test_too_long_sequence_rejected():
    m = make()
    with pytest.raises(ValueError):
        m(torch.zeros(1, 49, dtype=torch.long))


def test_flops_grow_with_seq_len():
    cfg = ModelConfig(vocab_size=1000)
    assert train_flops_per_token(cfg, 2048) > train_flops_per_token(cfg, 128) > 6 * param_counts(cfg)["matmul"] - 1
