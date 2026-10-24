"""Chat formats (formats/chat.py) and chat-wrapped analysis (analyze --chat)."""

import json
import math

import pytest
import torch
from test_gemma import make_gemma_gguf, tiny_tokenizer

from sleuth.analyzer.common import analysis_prefix, encode_cases, score_cases
from sleuth.analyzer.suites import TestCase
from sleuth.formats.chat import chat_format, encode_chat, render_chat
from sleuth.formats.gguf_model import load_gguf_model, tokenizer_from_gguf

USER = [{"role": "user", "content": "  Hi there \n"}]


def test_gemma4_rendering_follows_the_template():
    tok = tiny_tokenizer()
    assert chat_format(tok) == "gemma4"
    # One user message; thinking off: the generation prompt carries an empty thought channel.
    assert render_chat(tok, USER) == "<|turn>user\nHi there<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    # A system message gets its own turn; content is trimmed.
    msgs = [{"role": "system", "content": " Be brief. "}] + USER
    assert render_chat(tok, msgs).startswith("<|turn>system\nBe brief.<turn|>\n<|turn>user\nHi there<turn|>\n")
    # Thinking on: <|think|> opens a system turn, and the thought channel is left to the model.
    assert render_chat(tok, USER, thinking=True) == \
        "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi there<turn|>\n<|turn>model\n"
    # Earlier model turns are written as "model", with any thinking stripped.
    hist = USER + [{"role": "assistant", "content": "<|channel>thought\nhmm<channel|>Hello!"},
                   {"role": "user", "content": "Bye"}]
    assert render_chat(tok, hist, add_generation_prompt=False) == \
        "<|turn>user\nHi there<turn|>\n<|turn>model\nHello!<turn|>\n<|turn>user\nBye<turn|>\n"


def test_encode_chat_uses_single_special_tokens():
    tok = tiny_tokenizer()
    ids = encode_chat(tok, USER)
    assert ids[0] == tok.start_id
    for marker in ("<|turn>", "<turn|>", "<|channel>", "<channel|>"):
        assert tok.special_tokens[marker] in ids, marker
    assert tok.decode(ids[1:]) == render_chat(tok, USER)


def test_chatml_and_base_models():
    from types import SimpleNamespace

    q = SimpleNamespace(special_tokens={"<|im_start|>": 1, "<|im_end|>": 2})
    assert render_chat(q, USER) == "<|im_start|>user\n  Hi there \n<|im_end|>\n<|im_start|>assistant\n"
    with pytest.raises(ValueError, match="no chat format"):
        render_chat(SimpleNamespace(special_tokens={"<|bos|>": 0}), USER)


def test_prefix_is_not_scored(tmp_path):
    path = tmp_path / "g.gguf"
    make_gemma_gguf(path)
    model, _ = load_gguf_model(path, dtype=torch.float32)
    tok = tokenizer_from_gguf(path)
    cases = [TestCase("a", "story", "the cat is an ant", " ant")]
    prefix = analysis_prefix(tok, "Continue this text.")
    (e,) = encode_cases(tok, cases, 4096, prefix)
    assert e.n_prefix == len(prefix) and e.ids[0, :len(prefix)].tolist() == prefix
    s = score_cases(model, [e], tok.vocab_size, "cpu")["overall"]
    n_text = len(tok.encode("the cat is an ant")) + len(tok.encode(" ant"))
    assert s["n_tokens"] == n_text                      # only the test's own tokens
    with torch.no_grad():
        logits = model(e.ids).logits[0, :-1, :tok.vocab_size]
    nll = torch.nn.functional.cross_entropy(logits, e.ids[0, 1:], reduction="none")
    assert s["loss"] == pytest.approx(float(nll[len(prefix) - 1:].mean()), rel=1e-5)


def test_analyze_chat_writes_a_separate_suite(tmp_path, monkeypatch):
    from sleuth.analyzer.timeline import prepare_external_run, run_timeline

    monkeypatch.setenv("SLEUTH_RUNS", str(tmp_path / "runs"))
    path = tmp_path / "g.gguf"
    make_gemma_gguf(path)
    run = prepare_external_run(path, dtype=torch.float32)
    run_timeline(run, "suites/stories", views=["neuron_stats", "attention", "logit_lens", "ablation"],
                 log=lambda s: None, chat_instruction="Continue this text.")
    for view in ("neuron_stats", "attention", "logit_lens", "ablation"):
        summary = json.loads((run / "analysis" / view / "stories-chat" / "step_0000000" / "summary.json").read_text())
        assert summary["chat_prefix"].startswith("<bos><|turn>user\nContinue this text.<turn|>")
    ll = json.loads((run / "analysis" / "logit_lens" / "stories-chat" / "step_0000000" / "summary.json").read_text())
    att = json.loads((run / "analysis" / "attention" / "stories-chat" / "step_0000000" / "summary.json").read_text())
    assert math.isfinite(ll["loss"][-1]) and att["induction"]["text"]["n_cases"] > 0
    assert not (run / "analysis" / "logit_lens" / "stories").exists()
