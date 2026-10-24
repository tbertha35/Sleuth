"""Tests for the M3 analyzer views: scoring, attention, logit lens, ablation."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sleuth.analyzer import ablation, attention, logit_lens
from sleuth.analyzer.common import encode_cases, score_cases
from sleuth.analyzer.suites import TestCase, load_suite
from sleuth.forge.model.config import ModelConfig
from sleuth.forge.model.transformer import Transformer
from sleuth.forge.tokenizer.bpe import Tokenizer


@pytest.fixture(scope="module")
def tok(tiny_data):
    return Tokenizer.load(tiny_data["tokenizer"])


@pytest.fixture(scope="module")
def model(tok):
    torch.manual_seed(0)
    return Transformer(ModelConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=2, n_heads=4, max_seq_len=128)).eval()


CASES = [
    TestCase("a", "math", "Q: What is 2 + 3?\nA:", " 5"),
    TestCase("b", "story", "Once upon a time, there was a dog named Spot.", None),
    TestCase("c", "story", "Tom had a red ball. He gave the ball to", " Sam"),
]


def test_encode_marks_the_answer(tok):
    e = encode_cases(tok, CASES, 128)
    assert e[0].ids[0, 0].item() == tok.bos_id
    answer = e[0].ids[0, e[0].n_prompt:].tolist()
    assert tok.decode(answer) == " 5"
    assert e[1].n_prompt == e[1].ids.shape[1]  # no answer


def test_score_cases_with_a_perfect_model(tok):
    """A stub that always puts all probability on the true next token must
    score ~0 loss and 100% exact match."""
    enc = encode_cases(tok, CASES, 128)
    V = tok.vocab_size
    lookup = {tuple(e.ids[0].tolist()): e.ids for e in enc}

    def perfect(ids):
        seq = lookup[tuple(ids[0].tolist())][0]
        logits = torch.full((1, len(seq), V), -30.0)
        logits[0, torch.arange(len(seq) - 1), seq[1:]] = 30.0
        return SimpleNamespace(logits=logits)

    s = score_cases(perfect, enc, V, "cpu")
    assert s["overall"]["loss"] < 1e-6
    assert s["math"]["exact_match"] == 1.0 and s["overall"]["n_answers"] == 2
    assert s["story"]["n_answers"] == 1  # only case "c" has an answer


def test_induction_positions_point_after_the_earlier_copy():
    R = 7
    r = list(range(100, 100 + R))
    seq = [0] + r + r
    t, k_ind, k_dup = attention.induction_positions(R)
    for q, ki, kd in zip(t.tolist(), k_ind.tolist(), k_dup.tolist()):
        assert seq[kd] == seq[q] and kd < q          # duplicate key = earlier copy of the same token
        assert ki == kd + 1 and ki < q + 1           # induction key = the token that followed it


def test_attention_view_shapes_and_ranges(model, tok):
    res = attention.run(model, tok, CASES, "cpu", induction_len=10, induction_batch=2)
    s, a = res["summary"], res["arrays"]
    assert a["induction_score"].shape == (2, 4) and a["example_pattern"].shape[:2] == (2, 4)
    for k in ("bos_share", "prev_token", "self", "entropy", "induction_score", "duplicate_score"):
        assert np.all((a[k] >= -1e-6) & (a[k] <= 1 + 1e-6)), k
    rows = a["example_pattern"].astype(np.float32).sum(-1)
    np.testing.assert_allclose(rows, 1.0, atol=1e-2)  # every query's attention sums to 1
    assert math.isfinite(s["induction"]["induction_gain"])
    assert len(attention.timeline_series(s)["induction_score"]["rows"]) == 8


def test_logit_lens_last_row_is_the_model_itself(model, tok):
    res = logit_lens.run(model, tok, CASES, "cpu")
    s = res["summary"]
    assert s["rows"] == ["emb", "L0", "L1"]
    base = score_cases(model, encode_cases(tok, CASES, 128), tok.vocab_size, "cpu")
    assert s["loss"][-1] == pytest.approx(base["overall"]["loss"], rel=1e-4)
    assert res["arrays"]["example_true_p"].shape[0] == 3


@pytest.mark.parametrize("method", ["mean", "zero"])
def test_ablation(model, tok, method):
    res = ablation.run(model, tok, CASES, "cpu", method=method)
    s, a = res["summary"], res["arrays"]
    L, H = 2, 4
    assert len(s["components"]) == L * H + 2 * L
    assert a["delta_loss"].shape == (L * H + 2 * L, 3)  # math, story, overall
    assert np.isfinite(a["delta_loss"]).all()
    assert np.abs(a["delta_loss"]).max() > 0  # removing parts of the model changes something
    # Ablation must not leave the model modified.
    base = score_cases(model, encode_cases(tok, CASES, 128), tok.vocab_size, "cpu")
    assert base["overall"]["loss"] == pytest.approx(s["baseline"]["overall"]["loss"])


def test_ablating_nothing_changes_nothing(model, tok):
    """Mean-ablating a component whose output never varies has no effect. We
    fake that by ablating with the mean of a constant: zero out every
    attention head's output first, then the mean is 0 and ablation is a no-op."""
    from sleuth.forge.model.taps import listen

    with listen(model, {"blocks.*.attn.z": lambda n, t: torch.zeros_like(t)}):
        res = ablation.run(model, tok, CASES, "cpu", method="mean")
    heads = res["arrays"]["delta_loss"][: 2 * 4]
    np.testing.assert_allclose(heads, 0.0, atol=1e-6)


def test_stories_suite_loads():
    cases = load_suite("suites/stories")
    assert {c.category for c in cases} == {"names", "objects", "dialogue", "narrative"}
