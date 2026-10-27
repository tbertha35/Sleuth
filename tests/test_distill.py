"""Distillation (forge/distill.py): teacher-written text, and the teacher's top-k
probabilities, with fake teacher servers."""

import dataclasses
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from helpers import tiny_run_config

from sleuth.forge.distill import (TEMPLATES, TeacherSet, distill_loss, make_prompts, parse_prompt_logprobs,
                                  score_logits, teach_text)


def test_prompts_come_from_the_corpus(tiny_data):
    ps = make_prompts([tiny_data["raw"]], 9, seed=1)
    assert len(ps) == 9 and len({p["id"] for p in ps}) == 9
    assert {p["template"] for p in ps} == set(TEMPLATES)
    assert all("Passage:\n" in p["prompt"] and len(p["prompt"]) < 2200 for p in ps)
    assert make_prompts([tiny_data["raw"]], 9, seed=1) == ps          # deterministic


def test_teach_text_writes_resumes_and_strips_thinking(tiny_data, tmp_path):
    calls = []

    def fake(url, payload, api_key_env=None):
        calls.append(payload)
        if "fail" in payload["messages"][0]["content"]:
            raise TimeoutError("teacher busy")
        return {"choices": [{"message": {"content": "<think>hmm</think>\nQ: What is 2+2?\nA: 4"}}]}

    ps = make_prompts([tiny_data["raw"]], 6, seed=2)
    ps[3]["prompt"] += " fail"
    out = tmp_path / "t.jsonl"
    n = teach_text("http://127.0.0.1:8000", "qwen", ps, out, workers=3, log=lambda s: None, request=fake)
    assert n == 5 and calls[0]["chat_template_kwargs"] == {"enable_thinking": False}
    rows = [json.loads(s) for s in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["text"] == "Q: What is 2+2?\nA: 4" and rows[0]["source"] == "teacher:qwen"
    ps[3]["prompt"] = ps[3]["prompt"].replace(" fail", "")
    calls.clear()
    assert teach_text("http://127.0.0.1:8000", "qwen", ps, out, log=lambda s: None, request=fake) == 1
    assert len(calls) == 1                                             # only the failed one again


# ---------------------------------------------------------------- option B: the teacher's probabilities


def fake_teacher(vocab: int, k: int, calls: list | None = None):
    """A 'teacher' that's sure of the true next token (rank 1) and spreads the rest
    over token ids t+1.. Answers like vLLM's /v1/completions with prompt_logprobs:
    one dict per prompt position, top k plus the actual token."""

    def request(url, payload, api_key_env=None):
        if calls is not None:
            calls.append(len(payload["prompt"]))
        ids = payload["prompt"]
        pl = [None]
        for t in range(1, len(ids)):
            entry = {str(ids[t]): {"logprob": -0.1, "rank": 1}}
            for r in range(2, k + 1):
                entry[str((ids[t] + r) % vocab)] = {"logprob": -3.0 - r, "rank": r}
            entry.setdefault(str((ids[t] + k + 5) % vocab), {"logprob": -20.0, "rank": 999})
            pl.append(entry)
        return {"choices": [{"prompt_logprobs": pl}]}

    return request


def test_parse_prompt_logprobs_keeps_the_top_k_in_rank_order():
    pl = [None, {"5": {"logprob": -9.0, "rank": 30}, "7": {"logprob": -0.5, "rank": 1},
                 "8": {"logprob": -1.5, "rank": 2}}]
    ids, lps = parse_prompt_logprobs(pl, 2)
    assert ids == [[7, 8]] and lps == [[-0.5, -1.5]]                      # the actual token (rank 30) dropped
    ids, lps = parse_prompt_logprobs(pl, 4)
    assert ids == [[7, 8, -1, -1]]                                          # padded if the server sent fewer


def test_distill_loss():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 50)
    y = torch.randint(0, 50, (2, 5))
    one_hot = torch.stack([y, torch.full_like(y, -1)], -1)                   # teacher certain of y
    lp = torch.stack([torch.zeros(2, 5), torch.full((2, 5), -1e4)], -1)
    torch.testing.assert_close(distill_loss(logits, one_hot, lp), F.cross_entropy(logits.reshape(-1, 50), y.reshape(-1)))
    ids = torch.randint(0, 50, (2, 5, 3))
    lp = torch.randn(2, 5, 3)
    p = lp.softmax(-1)
    ref = -(p * logits.log_softmax(-1).gather(-1, ids)).sum(-1).mean()
    torch.testing.assert_close(distill_loss(logits, ids, lp), ref)


def _hf_shards(tiny_data, tmp_path):
    """Shards made with a Hugging Face tokenizer.json that has no BOS (like Qwen's)."""
    from sleuth.forge.data.shard import build_shards
    from sleuth.forge.tokenizer.bpe import Tokenizer
    from sleuth.formats.hf_tokenizer import HFTokenizer, tokenizer_to_hf_json

    ours = Tokenizer.load(tiny_data["tokenizer"])
    data, cfg = tokenizer_to_hf_json(ours)
    cfg.pop("bos_token", None)
    d = tmp_path / "hf-tok"
    d.mkdir()
    (d / "tokenizer.json").write_text(json.dumps(data), encoding="utf-8")
    (d / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    assert isinstance(Tokenizer.load(d / "tokenizer.json"), HFTokenizer)
    build_shards([tiny_data["raw"]], d / "tokenizer.json", tmp_path / "hf-shards", val_fraction=0.05,
                 shard_tokens=50_000, log=lambda s: None)
    return tmp_path / "hf-shards", ours


def test_score_train_and_resume_with_a_teacher(tiny_data, tmp_path):
    from sleuth.forge.data.loader import ShardDataset
    from sleuth.forge.train.trainer import train

    shards, ours = _hf_shards(tiny_data, tmp_path)
    ds = ShardDataset(shards)
    eos = ours.eos_id
    window = np.asarray(ds.arrays[0][:300])
    assert window[0] == eos and ours.bos_id not in window                  # documents start with the separator
    calls = []
    out = tmp_path / "teacher"
    ts = score_logits("http://127.0.0.1:8000", "fake", shards, out, n_windows=6, seq_len=32, k=4,
                      workers=2, log=lambda s: None, request=fake_teacher(ds.vocab_size, 4, calls))
    assert ts.done.all() and calls == [33] * 6
    t = TeacherSet(out)
    assert (t.top_ids[:, :, 0] == t.tokens[:, 1:]).all()                    # rank 1 = the true next token
    assert np.allclose(t.top_lp[0, 0], [-0.1, -5.0, -6.0, -7.0], atol=1e-2)
    calls.clear()
    score_logits("http://127.0.0.1:8000", "fake", shards, out, 6, 32, 4, log=lambda s: None,
                 request=fake_teacher(ds.vocab_size, 4, calls))
    assert calls == []                                                      # everything already scored
    x, y, ids, lp = t.batch(3, seed=0, step=1)
    assert x.shape == (3, 32) and ids.shape == (3, 32, 4) and torch.equal(ids[..., 0], y)

    cfg = tiny_run_config({"shards": shards}, tmp_path / "runs", max_steps=3, grad_accum=2, doc_mask=True,
                          teacher=str(out), distill_fraction=0.5)
    lines = []
    run = train(cfg, log=lines.append)
    recs = [json.loads(s) for s in (run / "metrics.jsonl").read_text().splitlines()]
    assert any("distill_loss" in r for r in recs) and any(" | distill " in s for s in lines)
    assert (run / "tokenizer_config.json").exists()
    bad = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, seq_len=16))
    with pytest.raises(ValueError, match="windows of 32"):
        train(bad, log=lambda s: None)


def test_hf_tokenizer_save_round_trip(tiny_data, tmp_path):
    """`sleuth tokenizer --from-model` writes a model's tokenizer for our shards."""
    from sleuth.forge.tokenizer.bpe import Tokenizer
    from sleuth.formats.hf_tokenizer import HFTokenizer, tokenizer_to_hf_json

    data, cfg = tokenizer_to_hf_json(Tokenizer.load(tiny_data["tokenizer"]))
    a = HFTokenizer(data, cfg)
    b = Tokenizer.load(a.save(tmp_path / "t"))
    s = "Once upon a time, 12 x 12 = 144."
    assert b.encode(s) == a.encode(s) and b.fingerprint() == a.fingerprint()
    assert (b.bos_id, b.eos_id) == (a.bos_id, a.eos_id)
