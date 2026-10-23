import torch

from sleuth.forge.infer.generate import generate, stream_text
from sleuth.forge.infer.sampling import SamplingConfig, apply_repetition_penalty, filter_logits, sample_next
from sleuth.forge.model.config import ModelConfig
from sleuth.forge.model.transformer import Transformer
from sleuth.forge.tokenizer.bpe import Tokenizer


def tiny_model():
    torch.manual_seed(0)
    return Transformer(ModelConfig(vocab_size=200, d_model=32, n_layers=2, n_heads=4, max_seq_len=40)).eval()


def test_greedy_generation_matches_no_cache_argmax():
    m = tiny_model()
    prompt = [1, 2, 3, 4]
    r = generate(m, prompt, max_new_tokens=10, sampling=SamplingConfig(temperature=0), vocab_limit=200)
    # Reference: recompute the full sequence each step without a cache.
    seq = list(prompt)
    for _ in range(10):
        logits = m(torch.tensor([seq])).logits[0, -1, :200]
        seq.append(int(logits.argmax()))
    assert r.new_ids == seq[len(prompt):]
    assert r.stop_reason == "max_new_tokens"


def test_seeded_sampling_is_reproducible_and_respects_vocab_limit():
    m = tiny_model()
    a = generate(m, [5], 20, SamplingConfig(temperature=1.5), vocab_limit=150, seed=7)
    b = generate(m, [5], 20, SamplingConfig(temperature=1.5), vocab_limit=150, seed=7)
    assert a.new_ids == b.new_ids
    assert all(t < 150 for t in a.new_ids)


def test_stop_token_and_context_limit():
    m = tiny_model()
    first = generate(m, [5], 1, SamplingConfig(temperature=0)).new_ids[0]
    r = generate(m, [5], 10, SamplingConfig(temperature=0), stop_ids={first})
    assert r.new_ids == [first] and r.stop_reason == "stop_token"
    r = generate(m, [1] * 35, 100, SamplingConfig(temperature=0))
    assert r.stop_reason == "context_full" and 35 + len(r.new_ids) <= 41


def test_top_k_top_p_min_p():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    keep = lambda cfg: (filter_logits(logits, cfg) > -1e30).squeeze(0).tolist()  # noqa: E731
    assert keep(SamplingConfig(top_k=2)) == [True, True, False, False]
    assert keep(SamplingConfig(top_p=0.7)) == [True, True, False, False]   # 0.5 < 0.7, 0.5+0.3 >= 0.7
    assert keep(SamplingConfig(top_p=0.01)) == [True, False, False, False]  # always keeps the best
    assert keep(SamplingConfig(min_p=0.25)) == [True, True, True, False]   # >= 0.125


def test_repetition_penalty_lowers_seen_tokens():
    logits = torch.tensor([[2.0, -1.0, 0.5]])
    out = apply_repetition_penalty(logits, torch.tensor([[0, 1]]), 2.0)
    assert out.tolist() == [[1.0, -2.0, 0.5]]


def test_temperature_zero_is_argmax():
    logits = torch.tensor([[0.1, 3.0, 0.2]])
    assert sample_next(logits, SamplingConfig(temperature=0)).item() == 1


def test_stream_text_handles_split_utf8():
    tok = Tokenizer([], {"<|bos|>": 256})  # bytes only: every byte is its own token
    ids = list("héllo 🧠!".encode("utf-8"))
    pieces = list(stream_text(tok, iter(ids)))
    assert "".join(pieces) == "héllo 🧠!"
    assert all("�" not in p for p in pieces)
