"""MTP heads for our own models (forge/model/mtp_heads.py) and their drafter
(forge/infer/mtp.py: HeadsDrafter)."""

import dataclasses
import json

import torch
import torch.nn.functional as F
from helpers import tiny_run_config

from sleuth.forge.infer.generate import generate
from sleuth.forge.infer.mtp import HeadsDrafter, builtin_drafter
from sleuth.forge.infer.sampling import SamplingConfig
from sleuth.forge.infer.speculative import speculative_generate
from sleuth.forge.model.config import ModelConfig
from sleuth.forge.model.flops import param_counts
from sleuth.forge.model.transformer import Transformer

CFG = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=128, mtp_heads=2)


def test_param_count_and_plain_models_unchanged():
    torch.manual_seed(0)
    m = Transformer(CFG)
    assert len(m.mtp) == 2 and m.num_params() == param_counts(CFG)["total"]
    plain = Transformer(dataclasses.replace(CFG, mtp_heads=0))
    assert plain.mtp is None and plain.num_params() < m.num_params()
    x = torch.randint(0, 64, (2, 16))
    assert plain(x, x).mtp_loss is None


def test_mtp_loss_matches_a_hand_written_chain():
    """Head k at position i: h_k = head_k(h_{k-1}[i], embed(x[i+k])), scored on x[i+k+1]."""
    torch.manual_seed(0)
    m = Transformer(CFG).eval()
    x = torch.randint(0, 64, (2, 12))
    y = torch.roll(x, -1, dims=1)
    y[:, -1] = -100                        # no target for the last position
    out = m(x, y, return_hidden=True)
    T, losses, h = x.shape[1], [], out.hidden
    for k in (1, 2):
        rows = []
        for i in range(T - k):
            rows.append(m.mtp[k - 1](h[:, i], m.tok_emb(x[:, i + k])))
        h = torch.stack(rows, dim=1)
        logits = m.lm_head(h)
        losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y[:, k:].reshape(-1), ignore_index=-100))
    torch.testing.assert_close(out.mtp_loss, torch.stack(losses).mean())
    out.mtp_loss.backward()                # the heads' loss trains the heads and the main model
    assert m.mtp[1].proj.weight.grad.abs().sum() > 0 and m.blocks[0].attn.q_proj.weight.grad.abs().sum() > 0


def test_trained_heads_draft_well_and_keep_greedy_text_exact():
    """On a repeating sequence the heads learn to look ahead: most guesses are
    accepted, and the text equals plain greedy decoding."""
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
        (out.loss + 0.5 * out.mtp_loss).backward()
        opt.step()
        opt.zero_grad()
    m.eval()
    prompt, greedy = seq[:15].tolist(), SamplingConfig(temperature=0)
    plain = generate(m, prompt, 40, greedy)
    drafter = builtin_drafter(m)
    assert isinstance(drafter, HeadsDrafter)
    for k in (1, 2, 4):                    # k > heads: the last head is reused
        spec = speculative_generate(m, prompt, 40, greedy, draft=drafter, k=k)
        assert spec.new_ids == plain.new_ids, k
        st = spec.stats["speculative"]
        assert st["accepted"] > 0.8 * st["drafted"], (k, st)


def test_trainer_logs_mtp_loss_and_runs_reload_with_heads(tiny_data, tmp_path):
    from sleuth.forge.train.trainer import train
    from sleuth.models import load_any

    cfg = tiny_run_config(tiny_data, tmp_path / "runs", max_steps=4, mtp_weight=0.3)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, mtp_heads=1))
    lines = []
    run = train(cfg, log=lines.append)
    recs = [json.loads(s) for s in (run / "metrics.jsonl").read_text().splitlines()]
    assert any("mtp_loss" in r for r in recs)
    assert any(" | mtp " in s for s in lines)
    model, tok, _ = load_any(run, device="cpu", dtype=torch.float32)
    assert model.mtp is not None and len(model.mtp) == 1
    prompt = tok.encode("Once upon a time")
    greedy = SamplingConfig(temperature=0)
    spec = speculative_generate(model, prompt, 12, greedy, draft=builtin_drafter(model), k=3)
    assert spec.new_ids == generate(model, prompt, 12, greedy).new_ids
