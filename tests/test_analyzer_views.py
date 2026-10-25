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
    assert math.isfinite(s["induction"]["induction_gain_any_token"])
    assert s["induction"]["token_pool"] == "suite tokens" and s["induction"]["pool_size"] >= 20
    assert len(attention.timeline_series(s)["induction_score"]["rows"]) == 8


def test_induction_sequences_use_distinct_ordinary_tokens():
    tok = SimpleNamespace(vocab_size=50, start_id=2, special_tokens={"<pad>": 0, "<eos>": 1, "<bos>": 2})
    ordinary = attention._ordinary_ids(tok)
    assert ordinary.tolist() == list(range(3, 50))   # specials can sit at the start (Gemma)
    seq = attention._repeated_random(tok, ordinary, R=10, batch=3)
    assert seq.shape == (3, 21) and (seq[:, 0] == 2).all()
    assert torch.equal(seq[:, 1:11], seq[:, 11:])
    for row in seq[:, 1:11]:
        assert len(set(row.tolist())) == 10 and min(row.tolist()) >= 3


def test_logit_lens_emb_row_mostly_repeats_the_input(model, tok):
    """With tied embeddings the 'emb' row predicts the current token (see logit_lens.py)."""
    s = logit_lens.run(model, tok, CASES, "cpu")["summary"]
    assert s["emb_top1_is_input"] > 0.9 and s["loss"][0] > s["uniform_loss"]


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


def _scripted(tok, enc, after: list[int]):
    """A stub model that predicts each test's true tokens, then `after`
    (what it would write once the answer is done), then <|eos|>."""
    V = tok.vocab_size

    def model(ids):
        seq = ids[0].tolist()
        for e in enc:
            full = e.ids[0].tolist()
            if seq[:len(full)] == full:
                script = full + after + [tok.eos_id]
                break
        logits = torch.full((1, len(seq), V), -30.0)
        for t in range(len(seq)):
            logits[0, t, script[min(t + 1, len(script) - 1)]] = 30.0
        return SimpleNamespace(logits=logits)
    return model


@pytest.mark.parametrize("after, exact", [
    ("\n", True),                # "0" then a new line: the answer is over
    (" degrees", True),          # "0 degrees": still the answer 0
    ("5", False),                # "05"
    (".|5", False),              # "0.5", as two tokens: needs one more look
    (".|\n", True),              # "0." at the end of a sentence
    ("", True),                  # straight to <|eos|>
])
def test_exact_match_needs_the_answer_to_end(tok, after, exact):
    """Round 5: mix-100m's top tokens matched " 0" for "water freezes at", but it
    was about to write "0.5 degrees Fahrenheit". An answer only counts when the
    model's own next token doesn't carry on the word or number."""
    enc = encode_cases(tok, [TestCase("w", "facts", "Q: At what Celsius temperature does water freeze?\nA:", " 0")], 128)
    cont = [i for part in after.split("|") if part for i in tok.encode(part)]
    model = _scripted(tok, enc, cont)
    rows = []
    s = score_cases(model, enc, tok.vocab_size, "cpu", rows, tokenizer=tok)
    assert s["facts"]["exact_match"] == float(exact)
    # It reads only as far as it needs to decide: "0" + " d" is enough.
    assert rows[0]["after_ids"] == (cont + [tok.eos_id])[:len(rows[0]["after_ids"])]
    # Without a tokenizer (ablation), matching the answer's tokens is enough.
    assert score_cases(model, enc, tok.vocab_size, "cpu")["facts"]["exact_match"] == 1.0


def test_default_guess_note(tok):
    from sleuth.cli import _default_guesses

    space, three = tok.encode(" ")[0], tok.encode("3")[0]
    want = [space, tok.encode("7")[0]]
    rows = [{"category": "sports", "answer_loss": 1.0, "answer_ids": want, "top1_ids": [space, three],
             "exact": i == 0} for i in range(4)]
    rows += [{"category": "sports", "answer_loss": 1.0, "answer_ids": want, "top1_ids": [space, three + 1 + i],
              "exact": False} for i in range(6)]
    notes = _default_guesses(rows, tok)   # the shared " " doesn't count as a guess
    assert len(notes) == 1 and "'3' in 4 of 10" in notes[0] and "1 of them is an exact match" in notes[0]
    assert _default_guesses(rows[3:], tok) == []   # 1 of 7: not a pattern
