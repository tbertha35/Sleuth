"""Speculative decoding (forge/infer/speculative.py): same tokens as plain
decoding for greedy, the same distribution for sampling."""

from collections import Counter

import pytest
import torch

from sleuth.forge.infer.generate import generate
from sleuth.forge.infer.sampling import SamplingConfig
from sleuth.forge.infer.speculative import ngram_draft, speculative_generate
from sleuth.forge.model.config import ModelConfig
from sleuth.forge.model.transformer import Transformer


def _model(vocab=64, **kw):
    torch.manual_seed(0)
    cfg = dict(vocab_size=vocab, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=128)
    cfg.update(kw)
    return Transformer(ModelConfig(**cfg)).eval()


def test_ngram_draft():
    d = ngram_draft(max_ngram=3)
    assert d([1, 2, 3, 4, 5, 1, 2, 3], 3) == [4, 5, 1]   # "1 2 3" seen before -> what followed
    assert d([7, 8, 9, 7, 8], 2) == [9, 7]
    assert d([1, 2, 3], 4) == []                          # nothing repeats
    assert d([5, 6, 5], 4) == [6, 5]                       # 1-gram fallback


@pytest.mark.parametrize("cfg", [
    SamplingConfig(temperature=0),
    SamplingConfig(temperature=0, repetition_penalty=1.3, repeat_last_n=16),
    SamplingConfig(temperature=0, no_repeat_ngram=3),
])
def test_greedy_speculative_matches_plain(cfg):
    """Exactly the tokens plain greedy decoding gives, with guesses accepted on the way."""
    m = _model(layer_types=["sliding", "full"], sliding_window=8)
    prompt = [1, 5, 9, 13, 5, 9, 13, 5, 9, 13, 22, 5, 9]  # repeats, so prompt lookup has something to propose
    plain = generate(m, prompt, 60, cfg, vocab_limit=60)
    spec = speculative_generate(m, prompt, 60, cfg, k=4, vocab_limit=60)
    assert spec.new_ids == plain.new_ids
    s = spec.stats["speculative"]
    assert s["forwards"] < len(spec.new_ids) - 1 or s["drafted"] == 0
    assert 0 <= s["accepted"] <= s["drafted"]


def test_greedy_with_a_bad_drafter_is_still_exact():
    """Wrong guesses only cost time: the text is unchanged."""
    m = _model()
    prompt = [1, 2, 3]
    plain = generate(m, prompt, 40, SamplingConfig(temperature=0))
    spec = speculative_generate(m, prompt, 40, SamplingConfig(temperature=0), draft=lambda ids, k: [7] * k, k=3)
    assert spec.new_ids == plain.new_ids


def test_sampling_distribution_is_unchanged():
    """Speculative sampling keeps the model's distribution. The second token is the
    one the accept/reject rule decides (the first comes from the prefill); over many
    seeds its distribution matches plain sampling's. The drafter always guesses
    token 3, so both acceptances and rejections happen. 8 real tokens (the output
    layer is padded to 64, hence vocab_limit), total variation against the noise
    level of 3000 samples."""
    m = _model(vocab=8)
    cfg = SamplingConfig(temperature=0.5, min_p=0.0)
    prompt = [1, 2, 3]
    runs = 3000

    def second(fn):
        return Counter(fn(seed).new_ids[1] for seed in range(runs))

    plain = second(lambda s: generate(m, prompt, 2, cfg, vocab_limit=8, seed=s))
    spec = second(lambda s: speculative_generate(m, prompt, 2, cfg, draft=lambda ids, k: [3] * k, k=2,
                                                 vocab_limit=8, seed=s))
    tv = 0.5 * sum(abs(plain[x] - spec[x]) for x in range(8)) / runs
    # Two plain runs with different seeds differ by ~0.035 here (measured), so 0.08
    # is about twice the noise; a wrong accept/reject rule gives far more (0.7 seen).
    assert tv < 0.08, f"total variation {tv:.3f} (plain {sorted(plain.items())}, spec {sorted(spec.items())})"


@pytest.mark.gpu
def test_speculative_with_cuda_graph_matches_plain():
    """Checks run as CUDA-graph replays of k+1 tokens (guesses padded): still the
    exact greedy tokens, including near the end of the context, where it falls
    back to the normal path."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    m = _model(layer_types=["sliding", "full"], sliding_window=8, max_seq_len=96).cuda()
    prompt = [1, 5, 9, 13, 5, 9, 13, 5, 9, 13, 22, 5, 9]
    greedy = SamplingConfig(temperature=0)
    plain = generate(m, prompt, 83, greedy, vocab_limit=60)   # fills the 96-token context
    spec = speculative_generate(m, prompt, 83, greedy, k=4, vocab_limit=60, cuda_graph=True)
    assert spec.new_ids == plain.new_ids
    assert spec.stats["cuda_graphs"] and spec.stats["speculative"]["forwards"] < len(spec.new_ids)


def test_stops_and_limits():
    m = _model()
    prompt = [1, 2, 3, 1, 2, 3, 1, 2]
    plain = generate(m, prompt, 30, SamplingConfig(temperature=0))
    stop = plain.new_ids[5]
    spec = speculative_generate(m, prompt, 30, SamplingConfig(temperature=0), stop_ids={stop}, k=4)
    assert spec.stop_reason == "stop_token" and spec.new_ids == plain.new_ids[:plain.new_ids.index(stop) + 1]
    spec = speculative_generate(m, prompt, 7, SamplingConfig(temperature=0), k=4)
    assert len(spec.new_ids) == 7 and spec.new_ids == plain.new_ids[:7]
    short = _model(max_seq_len=16)
    spec = speculative_generate(short, [1, 2, 3, 4, 5, 6, 7, 8], 50, SamplingConfig(temperature=0), k=4)
    assert spec.stop_reason == "context_full" and len(spec.new_ids) <= 16 - 8 + 1
