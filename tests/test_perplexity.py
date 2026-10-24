"""The perplexity command (llama-perplexity's chunking) and the `next` command."""

import math

import pytest
import torch
import torch.nn.functional as F
from test_gemma import make_gemma_gguf

from sleuth.analyzer.perplexity import perplexity, position_buckets
from sleuth.cli import main
from sleuth.formats.gguf_model import load_gguf_model, tokenizer_from_gguf

TEXT = "the cat is an ant. Once the cat ate an ant.\n\n" * 12


def test_position_buckets():
    assert position_buckets(32) == [(1, 7), (8, 15), (16, 31)]
    assert position_buckets(20) == [(1, 7), (8, 15), (16, 19)]


@pytest.fixture(scope="module")
def gemma(tmp_path_factory):
    path = tmp_path_factory.mktemp("ppl") / "g.gguf"
    make_gemma_gguf(path)
    return path


def test_perplexity_follows_llama_cpp_chunking(gemma):
    model, _ = load_gguf_model(gemma, dtype=torch.float32)
    tok = tokenizer_from_gguf(gemma)
    r = perplexity(model, tok, TEXT, ctx=16, max_chunks=3)
    # By hand: <bos> + text, chunks of 16 with the first token set to <bos>,
    # score predictions at positions 8..14 (the second half).
    ids = [tok.bos_id] + tok.encode(TEXT)
    total, n = 0.0, 0
    for c in range(3):
        chunk = [tok.bos_id] + ids[c * 16 + 1:(c + 1) * 16]
        x = torch.tensor([chunk])
        with torch.no_grad():
            nll = F.cross_entropy(model(x).logits[0, :-1, :tok.vocab_size], x[0, 1:], reduction="none")
        total += float(nll[8:].sum())
        n += 7
    assert r["scored_tokens"] == n and r["perplexity"] == pytest.approx(math.exp(total / n), rel=1e-5)
    assert len(r["running_perplexity"]) == 3 and r["bits_per_byte"] > 0
    assert [b["positions"] for b in r["by_position"]] == ["1-7", "8-15"]


def test_perplexity_and_next_commands(gemma, tmp_path, capsys):
    f = tmp_path / "t.txt"
    f.write_bytes(TEXT.encode("utf-8"))
    assert main(["perplexity", str(gemma), "--file", str(f), "--ctx", "16", "--chunks", "2", "--dtype", "fp32"]) == 0
    out = capsys.readouterr().out
    assert "perplexity" in out and "bits/byte" in out and "1-7" in out
    assert main(["next", str(gemma), "--prompt", "the cat", "--top", "3", "--dtype", "fp32"]) == 0
    out = capsys.readouterr().out
    assert out.count("%") >= 3 and "entropy" in out
    assert main(["next", str(gemma), "--prompt", "the cat", "--chat", "--dtype", "fp32"]) == 0
    assert main(["tokenize", "--tokenizer", str(gemma), "--chat", "the cat"]) == 0
    assert "<|turn>user" in capsys.readouterr().out


def test_generate_chat(gemma, capsys):
    assert main(["generate", str(gemma), "--prompt", "the cat", "--chat", "--system", "Be brief.",
                 "--max-new-tokens", "3", "--temperature", "0", "--dtype", "fp32"]) == 0
    assert "3 tokens" in capsys.readouterr().err
