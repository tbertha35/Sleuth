"""Benchmarks (analyzer/bench.py) and the tokenizer pieces they need for other
families' models (Gemma 3's SentencePiece-style tokenizer.json, GPT-2's)."""

import json
import math
import random

import torch
import torch.nn.functional as F

from sleuth.analyzer.bench import _hs_clean, bits_per_byte, multiple_choice
from sleuth.formats.gemma_tokenizer import GemmaTokenizer
from sleuth.formats.hf_tokenizer import HFTokenizer, is_sentencepiece_style


def _sp_json():
    """A tiny SentencePiece-style tokenizer.json, laid out like Gemma 3's."""
    chars = ["▁", "a", "b", "c", "\n"]
    merges = [["a", "b"], ["▁", "a"], ["ab", "c"], ["▁a", "b"], ["▁ab", "c"], ["c", "c"], ["\n", "\n"]]
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3}
    for c in chars + ["".join(m) for m in merges]:
        vocab.setdefault(c, len(vocab))
    for b in range(256):
        vocab[f"<0x{b:02X}>"] = len(vocab)
    added = [{"id": vocab[t], "content": t, "special": True} for t in ("<pad>", "<eos>", "<bos>", "<unk>")]
    return {"normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
            "pre_tokenizer": {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious"},
            "added_tokens": added,
            "model": {"type": "BPE", "byte_fallback": True, "unk_token": "<unk>", "vocab": vocab, "merges": merges}}


def _slow_bpe(tok, piece):
    """The textbook loop: merge the lowest-ranked pair (leftmost first), one at a time."""
    parts = list(piece)
    while True:
        best = min(((tok.ranks[(a, b)], i) for i, (a, b) in enumerate(zip(parts, parts[1:])) if (a, b) in tok.ranks),
                   default=None)
        if best is None:
            return parts
        i = best[1]
        parts[i:i + 2] = [parts[i] + parts[i + 1]]


def test_sentencepiece_style_tokenizer_json(tmp_path):
    data = _sp_json()
    assert is_sentencepiece_style(data)
    d = tmp_path / "tok"
    d.mkdir()
    (d / "tokenizer.json").write_text(json.dumps(data), encoding="utf-8")
    (d / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<bos>", "eos_token": "<eos>"}))
    tok = HFTokenizer.load(d)
    assert isinstance(tok, GemmaTokenizer) and (tok.bos_id, tok.eos_id) == (2, 1)
    text = "abc abcc\n\nab é"
    ids = tok.encode(text)
    assert tok.decode(ids) == text                                           # é via byte fallback
    # "▁abcc": "ab" has the lowest rank, so "▁ ab c c"; no "▁"+"ab" merge, then "abc"
    assert ids[:3] == [tok.token_to_id["abc"], tok.token_to_id["▁"], tok.token_to_id["abc"]]
    random.seed(0)                                                            # heap BPE == the textbook loop
    for _ in range(200):
        piece = "".join(random.choice("▁abc") for _ in range(random.randint(1, 30)))
        assert tok._merge(piece) == _slow_bpe(tok, piece), piece


def test_gpt2_style_tokenizer_without_a_type(tiny_data):
    from sleuth.forge.tokenizer.bpe import Tokenizer
    from sleuth.formats.hf_tokenizer import tokenizer_to_hf_json

    data, cfg = tokenizer_to_hf_json(Tokenizer.load(tiny_data["tokenizer"]))
    data["model"].pop("type")                                                 # as in GPT-2's tokenizer.json
    tok = HFTokenizer(data, cfg)
    assert tok.decode(tok.encode("Once upon a time")) == "Once upon a time"


def test_hellaswag_cleanup():
    assert _hs_clean(" How to cook [title] Boil water. [step] Add salt.  ") == "How to cook. Boil water. Add salt."


class _Toy(torch.nn.Module):
    """A 'model' whose next-token distribution is fixed per previous token."""

    def __init__(self, table):
        super().__init__()
        self.table = table
        self.cfg = type("C", (), {"max_seq_len": 64, "padded_vocab": table.shape[1]})()

    def forward(self, x):
        return type("O", (), {"logits": self.table[x]})()


class _CharTok:
    vocab_size, start_id = 6, 5

    def encode(self, s):
        return [" abcd".index(c) for c in s]


def test_multiple_choice_and_bits_per_byte():
    torch.manual_seed(0)
    table = torch.randn(6, 6)
    model, tok = _Toy(table), _CharTok()
    lp = table.log_softmax(-1)
    items = [{"context": "ab", "choices": [" c", " dd", " a"], "label": 1},
             {"context": "c", "choices": [" a", " b"], "label": 0}]
    res = multiple_choice(model, tok, items, torch.device("cpu"), log=lambda s: None)

    def score(ctx, ch):
        ids = [5] + tok.encode(ctx + ch)
        n = len(ch)
        return sum(float(lp[ids[i - 1], ids[i]]) for i in range(len(ids) - n, len(ids)))

    right = sum(max(range(len(it["choices"])), key=lambda c: score(it["context"], it["choices"][c])) == it["label"]
                for it in items)
    right_norm = sum(max(range(len(it["choices"])),
                         key=lambda c: score(it["context"], it["choices"][c]) / len(it["choices"][c])) == it["label"]
                     for it in items)
    assert (res["acc"], res["acc_norm"]) == (right / 2, right_norm / 2)

    texts = ["abc dab", "dd"]
    bpb = bits_per_byte(model, tok, texts, torch.device("cpu"), seq_len=4, log=lambda s: None)
    nll = 0.0
    for t in texts:
        ids = [5] + tok.encode(t)
        nll -= sum(float(lp[ids[i - 1], ids[i]]) for i in range(1, len(ids)))
    assert math.isclose(bpb["bpb"], nll / math.log(2) / 9, rel_tol=1e-5)    # every token scored exactly once
    assert bpb["tokens"] == 9
