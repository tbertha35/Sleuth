"""Hugging Face format: regex translation, tokenizer.json, checkpoint load/export."""

import json
import random
import re

import pytest
import torch

from sleuth.forge.model.config import ModelConfig
from sleuth.forge.model.transformer import Transformer
from sleuth.forge.tokenizer.bpe import Tokenizer
from sleuth.forge.tokenizer.pretokenize import pretokenize
from sleuth.formats.hf import load_hf_model, our_name, save_hf_model
from sleuth.formats.hf_tokenizer import (GPT2_PATTERN, GPT4_PATTERN, HFTokenizer, tokenizer_to_hf_json,
                                         translate_regex)

SAMPLES = [
    "Hello world, it's 2026!", "  spaces   and\ttabs\n\nnewlines ", "café naïve 日本語 🧠 ²³",
    "under_score snake_case __init__", "x=3.14159; y = -42 # comment", "I'm you're WE'LL they'd",
]


def random_text(rng, n):
    alphabet = "abcXYZ 0123456789 .,!?'\"\n\t_-é日🧠²"
    return "".join(rng.choice(alphabet) for _ in range(n))


def test_translated_gpt4_pattern_matches_our_pretokenizer():
    rx = re.compile(translate_regex(GPT4_PATTERN))
    rng = random.Random(0)
    for s in SAMPLES + [random_text(rng, 200) for _ in range(100)]:
        assert rx.findall(s) == pretokenize(s)


def test_negated_class_translation():
    rx = re.compile(translate_regex(r"[^\s\p{L}\p{N}]"))
    for ch, expected in [("a", False), ("Z", False), ("é", False), ("日", False), ("7", False), (" ", False),
                         ("\n", False), ("_", True), ("!", True), ("-", True), ("🧠", True)]:
        assert bool(rx.fullmatch(ch)) == expected, ch


def test_gpt2_pattern_translates():
    rx = re.compile(translate_regex(GPT2_PATTERN))
    assert rx.findall("Hello world's 123 !!") == ["Hello", " world", "'s", " 123", " !!"]


def test_unknown_unicode_class_is_an_error():
    with pytest.raises(ValueError):
        translate_regex(r"\p{Lu}+")


@pytest.fixture(scope="module")
def tok(tiny_data):
    return Tokenizer.load(tiny_data["tokenizer"])


def test_tokenizer_json_roundtrip(tok, tmp_path):
    data, cfg = tokenizer_to_hf_json(tok)
    (tmp_path / "tokenizer.json").write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    hf = HFTokenizer.load(tmp_path)
    assert hf.vocab_size == tok.vocab_size
    assert (hf.bos_id, hf.eos_id) == (tok.bos_id, tok.eos_id)
    rng = random.Random(1)
    for s in SAMPLES + [random_text(rng, 300) for _ in range(50)]:
        ids = tok.encode(s)
        assert hf.encode(s) == ids  # identical token IDs
        assert hf.decode(ids) == s
    assert hf.encode("a<|eos|>b", allowed_special="all") == tok.encode("a<|eos|>b", allowed_special="all")
    assert hf.token_byte_lengths()[: 256] == tok.token_byte_lengths()[: 256]


VARIANTS = {
    "llama": dict(tie_embeddings=True),
    "qwen2": dict(qkv_bias=True, tie_embeddings=False),
    "qwen3": dict(qk_norm=True, head_size=24, tie_embeddings=False, n_kv_heads=2),
}


@pytest.mark.parametrize("kind", list(VARIANTS))
def test_export_then_load_gives_identical_outputs(kind, tmp_path):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=320, d_model=64, n_layers=3, n_heads=4, max_seq_len=64, rope_theta=50000.0,
                      norm_eps=1e-6, pad_vocab_to=64, **VARIANTS[kind])
    ours = Transformer(cfg).eval()
    # Biases start at zero; give them values so a mapping mistake would show.
    for n, p in ours.named_parameters():
        if n.endswith(".bias") or "norm" in n:
            torch.nn.init.normal_(p, 1.0 if "norm" in n else 0.0, 0.1)
    out = save_hf_model(ours, tmp_path / kind, max_shard_bytes=40_000)  # tiny limit forces several shards
    assert (out / "model.safetensors.index.json").exists()
    assert json.loads((out / "config.json").read_text())["model_type"] == kind
    loaded, info = load_hf_model(out, dtype=torch.float32)
    assert info["n_unused"] == 0
    assert loaded.cfg.qk_norm == cfg.qk_norm and loaded.cfg.qkv_bias == cfg.qkv_bias
    assert loaded.cfg.head_dim == cfg.head_dim and loaded.cfg.tie_embeddings == cfg.tie_embeddings
    x = torch.randint(0, 320, (2, 20))
    # Saved as bf16 (as real checkpoints are), so compare against our model rounded the same way.
    ours_bf16 = Transformer(cfg).eval()
    ours_bf16.load_state_dict({k: v.to(torch.bfloat16).float() for k, v in ours.state_dict().items()})
    torch.testing.assert_close(loaded(x).logits, ours_bf16(x).logits, atol=1e-4, rtol=1e-4)


def test_name_mapping():
    assert our_name("model.layers.7.self_attn.q_proj.bias") == "blocks.7.attn.q_proj.bias"
    assert our_name("model.layers.0.mlp.down_proj.weight") == "blocks.0.mlp.down.weight"
    assert our_name("model.layers.3.self_attn.k_norm.weight") == "blocks.3.attn.k_norm.weight"
    assert our_name("model.layers.0.self_attn.rotary_emb.inv_freq") is None


def test_unsupported_configs_fail_clearly(tmp_path):
    torch.manual_seed(0)
    m = Transformer(ModelConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=2, max_seq_len=16))
    out = save_hf_model(m, tmp_path / "m")
    cfg = json.loads((out / "config.json").read_text())
    for patch, msg in [({"model_type": "gemma3"}, "isn't supported"),
                       ({"rope_scaling": {"rope_type": "llama3", "factor": 8.0}}, "rope_scaling"),
                       ({"hidden_act": "gelu"}, "activation")]:
        (out / "config.json").write_text(json.dumps({**cfg, **patch}))
        with pytest.raises(NotImplementedError, match=msg):
            load_hf_model(out)
