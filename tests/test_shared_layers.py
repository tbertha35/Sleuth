"""Shared (looped) layers: model.shared_layers."""

import dataclasses

import pytest
import torch
from helpers import tiny_run_config

from sleuth.forge.infer.kv_cache import KVCache
from sleuth.forge.model.config import ModelConfig
from sleuth.forge.model.flops import param_counts
from sleuth.forge.model.transformer import Transformer

LOOP = [0, 1, 1, 1, 4]   # layer 1's weights run three times: depth 5 from 3 layers
CFG = ModelConfig(vocab_size=64, d_model=32, n_layers=5, n_heads=4, max_seq_len=64, shared_layers=LOOP)


def test_config_checks():
    for bad in ([0, 1], [0, 2, 1, 1, 1], [0, 0, 1, 1, 1]):   # wrong length; a later layer; a source that's shared
        with pytest.raises(ValueError):
            dataclasses.replace(CFG, shared_layers=bad)


def test_weights_are_shared_and_counted_once():
    torch.manual_seed(0)
    m = Transformer(CFG)
    b = m.blocks
    assert b[2].attn.q_proj.weight is b[1].attn.q_proj.weight is b[3].attn.q_proj.weight
    assert b[3].mlp.down.weight is b[1].mlp.down.weight
    assert b[2].norm1.weight is not b[1].norm1.weight          # norms stay per layer
    assert b[4].attn.q_proj.weight is not b[1].attn.q_proj.weight
    assert m.num_params() == param_counts(CFG)["total"]
    unshared = dataclasses.replace(CFG, shared_layers=[])
    per_layer = param_counts(unshared)["per_layer"]
    norms = 2 * 32
    assert param_counts(unshared)["total"] - m.num_params() == 2 * (per_layer - norms)
    # same matmul work as the unshared model: every application still runs
    assert param_counts(CFG)["matmul"] == param_counts(unshared)["matmul"]
    x = torch.randint(0, 64, (2, 16))
    m(x, x).loss.backward()
    assert b[1].attn.q_proj.weight.grad.abs().sum() > 0


def test_each_application_has_its_own_cache_slot():
    torch.manual_seed(0)
    m = Transformer(CFG).eval()
    x = torch.randint(0, 64, (1, 10))
    with torch.no_grad():
        full = m(x).logits
        cache = KVCache.for_model(m, 1, 64, torch.float32)
        m(x[:, :6], cache=cache)
        steps = [m(x[:, i:i + 1], cache=cache).logits for i in range(6, 10)]
    torch.testing.assert_close(torch.cat(steps, 1), full[:, 6:], atol=1e-5, rtol=1e-5)
    assert len(cache.k) == 5 and not torch.equal(cache.k[1], cache.k[2])


def test_snapshots_keep_the_sharing(tiny_data, tmp_path):
    from sleuth.forge.train.checkpoint import load_model
    from sleuth.forge.train.trainer import train

    cfg = tiny_run_config(tiny_data, tmp_path / "runs", max_steps=3)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, n_layers=3, shared_layers=[0, 0, 0]))
    run = train(cfg, log=lambda s: None)
    m, _ = load_model(run)
    assert m.blocks[2].mlp.up.weight is m.blocks[0].mlp.up.weight
    assert m.num_params() == param_counts(m.cfg)["total"]
