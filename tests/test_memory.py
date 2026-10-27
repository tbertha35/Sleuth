"""N-gram memory (forge/model/memory.py, model.mem_*)."""

import dataclasses

import pytest
import torch
from helpers import tiny_run_config

from sleuth.forge.infer.generate import generate
from sleuth.forge.infer.kv_cache import KVCache
from sleuth.forge.infer.mtp import builtin_drafter
from sleuth.forge.infer.sampling import SamplingConfig
from sleuth.forge.infer.speculative import speculative_generate
from sleuth.forge.model.config import ModelConfig
from sleuth.forge.model.flops import param_counts
from sleuth.forge.model.memory import NO_TOKEN
from sleuth.forge.model.transformer import Transformer
from sleuth.forge.train.config import TrainConfig
from sleuth.forge.train.optim import build_optimizer

CFG = ModelConfig(vocab_size=64, d_model=32, n_layers=3, n_heads=4, max_seq_len=64,
                  mem_rows=1000, mem_dim=8, mem_orders=[2, 3], mem_heads=2, mem_layers=[0, 2])


def test_params_counted_and_table_not_decayed():
    torch.manual_seed(0)
    m = Transformer(CFG)
    pc = param_counts(CFG)
    assert m.num_params() == pc["total"] and pc["memory_table"] == 4 * 1000 * 8
    opt = build_optimizer(m, TrainConfig(), "cpu")
    decayed = {id(p) for g in opt.param_groups if g["weight_decay"] > 0 for p in g["params"]}
    assert id(m.memory.table.weight) not in decayed and id(m.memory.key[0].weight) in decayed
    with pytest.raises(ValueError):
        dataclasses.replace(CFG, mem_layers=[3])
    assert not CFG.llama_compatible


def test_context_and_hashing():
    m = Transformer(CFG).memory
    x = torch.tensor([[5, 6, 7, 8, 9]])
    ctx = m.context(x)
    assert ctx[0, 3].tolist() == [8, 7, 6] and ctx[0, 0].tolist() == [5, NO_TOKEN, NO_TOKEN]
    doc = torch.tensor([[0, 0, 0, 1, 1]])                  # a new document starts at position 3
    assert m.context(x, doc_ids=doc)[0, 4].tolist() == [9, 8, NO_TOKEN]
    rows = m.rows_for(ctx)
    assert rows.shape == (1, 5, 4)
    for t in range(4):                                     # each table's rows stay in its own range
        assert ((rows[..., t] >= t * 1000) & (rows[..., t] < (t + 1) * 1000)).all()
    same = m.rows_for(m.context(torch.tensor([[1, 8, 9]])))  # same last two tokens as above
    assert same[0, 2, 0] == rows[0, 4, 0] and same[0, 2, 2] != rows[0, 4, 2]   # bigram rows equal, trigram not


def test_cached_decoding_matches_one_pass():
    torch.manual_seed(0)
    m = Transformer(CFG).eval()
    with torch.no_grad():
        for p in m.memory.parameters():                     # make the memory matter
            p.normal_(0, 0.5)
        x = torch.randint(0, 64, (1, 12))
        full = m(x).logits
        cache = KVCache.for_model(m, 1, 64, torch.float32)
        m(x[:, :7], cache=cache)
        steps = [m(x[:, i:i + 1], cache=cache).logits for i in range(7, 12)]
        off = dataclasses.replace(CFG, mem_rows=0)
    torch.testing.assert_close(torch.cat(steps, 1), full[:, 7:], atol=1e-5, rtol=1e-5)
    plain = Transformer(off).eval()
    plain.load_state_dict({k: v for k, v in m.state_dict().items() if not k.startswith("memory.")})
    with torch.no_grad():
        assert (plain(x).logits - full).abs().max() > 1e-2   # the memory changes the output


def test_learns_and_drafts_exactly_with_an_early_exit():
    """With memory + early exit: training lowers the loss, and self-speculation keeps
    greedy text exact (the drafter's memory lookups go through the same cache)."""
    torch.manual_seed(0)
    cfg = dataclasses.replace(CFG, exit_layer=1, mem_layers=[0])
    m = Transformer(cfg)
    seq = torch.randperm(64)[:13].repeat(30)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(120):
        s = torch.randint(0, len(seq) - 33, (8,))
        x = torch.stack([seq[i:i + 32] for i in s])
        y = torch.stack([seq[i + 1:i + 33] for i in s])
        out = m(x, y)
        first = first if first is not None else out.loss.item()
        (out.loss + 0.5 * out.exit_loss).backward()
        opt.step()
        opt.zero_grad()
    assert out.loss.item() < 0.5 * first
    m.eval()
    prompt, greedy = seq[:10].tolist(), SamplingConfig(temperature=0)
    plain = generate(m, prompt, 30, greedy)
    spec = speculative_generate(m, prompt, 30, greedy, draft=builtin_drafter(m), k=3)
    assert spec.new_ids == plain.new_ids


@pytest.mark.gpu
def test_memory_with_cuda_graphs():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    torch.manual_seed(0)
    m = Transformer(CFG).cuda().eval()
    with torch.no_grad():
        for p in m.memory.parameters():
            p.normal_(0, 0.5)
    prompt, greedy = torch.randint(0, 64, (8,)).tolist(), SamplingConfig(temperature=0)
    a = generate(m, prompt, 30, greedy)
    b = generate(m, prompt, 30, greedy, cuda_graph=True)
    assert a.new_ids == b.new_ids


def test_trains_in_the_trainer(tiny_data, tmp_path):
    from sleuth.forge.train.trainer import train

    cfg = tiny_run_config(tiny_data, tmp_path / "runs", max_steps=3, doc_mask=True)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, mem_rows=500, mem_dim=8))
    train(cfg, log=lambda s: None)
