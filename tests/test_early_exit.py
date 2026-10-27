"""Early exit / self-speculation (model.exit_layer, Transformer.early_logits,
forge/infer/mtp.py: EarlyExitDrafter)."""

import dataclasses
import json

import pytest
import torch
import torch.nn.functional as F
from helpers import tiny_run_config

from sleuth.forge.infer.generate import generate
from sleuth.forge.infer.kv_cache import KVCache
from sleuth.forge.infer.mtp import EarlyExitDrafter, builtin_drafter
from sleuth.forge.infer.sampling import SamplingConfig
from sleuth.forge.infer.speculative import speculative_generate
from sleuth.forge.model.config import ModelConfig
from sleuth.forge.model.flops import param_counts
from sleuth.forge.model.transformer import Transformer

CFG = ModelConfig(vocab_size=64, d_model=32, n_layers=3, n_heads=4, max_seq_len=128, exit_layer=1)


def test_config_params_and_plain_models_unchanged():
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=64, d_model=32, n_layers=3, n_heads=4, exit_layer=3)
    torch.manual_seed(0)
    m = Transformer(CFG)
    assert m.num_params() == param_counts(CFG)["total"]
    plain = Transformer(dataclasses.replace(CFG, exit_layer=0))
    assert plain.exit_norm is None and plain.num_params() == m.num_params() - 32
    x = torch.randint(0, 64, (2, 16))
    assert plain(x, x).exit_loss is None


def test_exit_loss_is_the_next_token_loss_after_the_first_blocks():
    torch.manual_seed(0)
    m = Transformer(CFG).eval()
    x = torch.randint(0, 64, (2, 12))
    y = torch.roll(x, -1, 1)
    out = m(x, y)
    T = x.shape[1]
    h = m.embed(x)
    h = m.blocks[0](h, m.rope_cos_full[:T], m.rope_sin_full[:T])
    ref = F.cross_entropy(m.lm_head(m.exit_norm(h)).reshape(-1, m.cfg.padded_vocab), y.reshape(-1))
    torch.testing.assert_close(out.exit_loss, ref)
    out.exit_loss.backward()   # trains the first block and the exit norm, not the later blocks
    assert m.exit_norm.weight.grad is not None and m.blocks[0].attn.q_proj.weight.grad.abs().sum() > 0
    assert m.blocks[2].attn.q_proj.weight.grad is None


def test_early_logits_with_a_cache_match_one_pass_and_leave_later_layers_alone():
    torch.manual_seed(0)
    m = Transformer(CFG).eval()
    x = torch.randint(0, 64, (1, 9))
    with torch.no_grad():
        cache = KVCache.for_model(m, 1, 64, torch.float32)
        whole = m.early_logits(x, cache)
        assert cache.pos == 9
        cache2 = KVCache.for_model(m, 1, 64, torch.float32)
        m.early_logits(x[:, :5], cache2)
        steps = [m.early_logits(x[:, i:i + 1], cache2) for i in range(5, 9)]
    torch.testing.assert_close(torch.cat(steps, 1), whole[:, 5:], atol=1e-5, rtol=1e-5)
    assert cache2.k[2].abs().sum() == 0   # layer 2 is never written by drafting


def _trained():
    """A model taught a repeating sequence: both exits learn it."""
    torch.manual_seed(0)
    m = Transformer(CFG)
    cycle = torch.randperm(64)[:11]
    seq = cycle.repeat(40)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(150):
        starts = torch.randint(0, len(seq) - 33, (8,))
        x = torch.stack([seq[s:s + 32] for s in starts])
        y = torch.stack([seq[s + 1:s + 33] for s in starts])
        out = m(x, y)
        (out.loss + 0.5 * out.exit_loss).backward()
        opt.step()
        opt.zero_grad()
    return m.eval(), seq


def test_self_speculation_keeps_greedy_text_exact():
    m, seq = _trained()
    prompt, greedy = seq[:15].tolist(), SamplingConfig(temperature=0)
    plain = generate(m, prompt, 40, greedy)
    drafter = builtin_drafter(m)
    assert isinstance(drafter, EarlyExitDrafter)
    for k in (1, 3):
        spec = speculative_generate(m, prompt, 40, greedy, draft=drafter, k=k)
        assert spec.new_ids == plain.new_ids, k
        st = spec.stats["speculative"]
        assert st["accepted"] > 0.8 * st["drafted"], (k, st)


@pytest.mark.gpu
def test_self_speculation_with_cuda_graphs():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    m, seq = _trained()
    m = m.cuda()
    prompt, greedy = seq[:15].tolist(), SamplingConfig(temperature=0)
    plain = generate(m, prompt, 40, greedy, cuda_graph=True)
    drafter = builtin_drafter(m)
    spec = speculative_generate(m, prompt, 40, greedy, draft=drafter, k=3, cuda_graph=True)
    assert spec.new_ids == plain.new_ids
    assert drafter._graphs.graphs                              # the drafts ran as a graph too
    st = spec.stats["speculative"]
    assert st["accepted"] > 0.8 * st["drafted"], st            # ...and guess what the eager drafter guesses
    # An untrained model with the n-gram memory: the graph drafts in the cache's token history too.
    torch.manual_seed(1)
    mm = Transformer(dataclasses.replace(CFG, mem_rows=64, mem_dim=8, mem_layers=[0])).cuda().eval()
    eager = speculative_generate(mm, prompt, 30, greedy, draft=EarlyExitDrafter(mm), k=3)
    graphed = speculative_generate(mm, prompt, 30, greedy, draft=EarlyExitDrafter(mm), k=3, cuda_graph=True)
    assert graphed.new_ids == generate(mm, prompt, 30, greedy).new_ids
    assert graphed.stats["speculative"]["accepted"] == eager.stats["speculative"]["accepted"]


def test_trainer_logs_exit_loss(tiny_data, tmp_path):
    from sleuth.forge.train.trainer import train

    cfg = tiny_run_config(tiny_data, tmp_path / "runs", max_steps=4)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, exit_layer=1, mtp_heads=1))
    lines = []
    run = train(cfg, log=lines.append)
    recs = [json.loads(s) for s in (run / "metrics.jsonl").read_text().splitlines()]
    assert any("exit_loss" in r and "mtp_loss" in r for r in recs)
    assert any(" | exit " in s and " | mtp " in s for s in lines)
