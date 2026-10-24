"""Loading external models: GGUF (llama/qwen3) into our Transformer, the GGUF
tokenizer, and analyzing an external model end to end."""

import json

import numpy as np
import pytest
import torch
from helpers import write_gguf

from sleuth.analyzer.timeline import prepare_external_run, run_timeline
from sleuth.forge.model.config import ModelConfig
from sleuth.forge.model.transformer import Transformer
from sleuth.forge.tokenizer.bpe import Tokenizer
from sleuth.formats import gguf_quants as q
from sleuth.formats.gguf_model import load_gguf_model, permute_rope_rows, tokenizer_from_gguf
from sleuth.formats.hf import save_hf_model
from sleuth.formats.hf_tokenizer import tokenizer_to_hf_json
from sleuth.models import detect, load_any

GGUF_NAMES = [("tok_emb.weight", "token_embd.weight"), ("norm_f.weight", "output_norm.weight"),
              ("lm_head.weight", "output.weight")]
LAYER = {"norm1.weight": "attn_norm.weight", "norm2.weight": "ffn_norm.weight",
         "attn.q_proj.weight": "attn_q.weight", "attn.k_proj.weight": "attn_k.weight",
         "attn.v_proj.weight": "attn_v.weight", "attn.o_proj.weight": "attn_output.weight",
         "attn.q_norm.weight": "attn_q_norm.weight", "attn.k_norm.weight": "attn_k_norm.weight",
         "mlp.gate.weight": "ffn_gate.weight", "mlp.up.weight": "ffn_up.weight", "mlp.down.weight": "ffn_down.weight"}


def q8(x: np.ndarray) -> bytes:
    out = b""
    for blk in x.reshape(-1, 32):
        d = np.abs(blk).max() / 127 or 1.0
        out += np.float16(d).tobytes() + np.round(blk / d).astype(np.int8).tobytes()
    return out


def export_gguf(model: Transformer, tok: Tokenizer, path, arch: str, quant: bool = False):
    """Write our model the way llama.cpp's converter would (for tests)."""
    cfg = model.cfg
    tensors = {}
    for name, p in model.named_parameters():
        gname = dict(GGUF_NAMES).get(name)
        if gname is None:
            i, rest = name.split(".", 2)[1], name.split(".", 2)[2]
            gname = f"blk.{i}.{LAYER[rest]}"
        w = p.detach().float()
        if arch == "llama" and gname.endswith(("attn_q.weight", "attn_k.weight")):
            w = permute_rope_rows(w, cfg.n_heads if "attn_q" in gname else cfg.kv_heads)
        arr = w.numpy().astype(np.float32)
        if quant and arr.ndim == 2 and arr.shape[1] % 32 == 0:
            tensors[gname] = (q.Q8_0, tuple(arr.shape), q8(arr.reshape(-1)))
        else:
            tensors[gname] = (q.F32, tuple(arr.shape), arr.tobytes())
    data, _ = tokenizer_to_hf_json(tok)
    tokens = [None] * tok.vocab_size
    for s, i in data["model"]["vocab"].items():
        tokens[i] = s
    for t in data["added_tokens"]:
        tokens[t["id"]] = t["content"]
    types = [3 if i >= tok.bos_id else 1 for i in range(tok.vocab_size)]
    md = {
        "general.architecture": (8, arch),
        f"{arch}.block_count": (4, cfg.n_layers),
        f"{arch}.embedding_length": (4, cfg.d_model),
        f"{arch}.feed_forward_length": (4, cfg.hidden),
        f"{arch}.attention.head_count": (4, cfg.n_heads),
        f"{arch}.attention.head_count_kv": (4, cfg.kv_heads),
        f"{arch}.attention.key_length": (4, cfg.head_dim),
        f"{arch}.attention.layer_norm_rms_epsilon": (6, cfg.norm_eps),
        f"{arch}.rope.freq_base": (6, cfg.rope_theta),
        f"{arch}.context_length": (4, cfg.max_seq_len),
        "tokenizer.ggml.model": (8, "gpt2"),
        "tokenizer.ggml.pre": (8, "llama-bpe"),
        "tokenizer.ggml.tokens": (9, (8, tokens)),
        "tokenizer.ggml.token_type": (9, (5, types)),
        "tokenizer.ggml.merges": (9, (8, data["model"]["merges"])),
        "tokenizer.ggml.bos_token_id": (4, tok.bos_id),
        "tokenizer.ggml.eos_token_id": (4, tok.eos_id),
    }
    write_gguf(path, md, tensors)


@pytest.fixture(scope="module")
def tok(tiny_data):
    return Tokenizer.load(tiny_data["tokenizer"])


def make_model(tok, **kw):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                      max_seq_len=64, pad_vocab_to=1, norm_eps=1e-6, **kw)
    m = Transformer(cfg).eval()
    for n, p in m.named_parameters():
        if "norm" in n:
            torch.nn.init.normal_(p, 1.0, 0.1)
    return m


@pytest.mark.parametrize("arch,kw", [("qwen3", dict(qk_norm=True, tie_embeddings=False)),
                                     ("llama", dict(tie_embeddings=True))])
def test_gguf_model_matches_original(tok, tmp_path, arch, kw):
    ours = make_model(tok, **kw)
    export_gguf(ours, tok, tmp_path / "m.gguf", arch)
    loaded, info = load_gguf_model(tmp_path / "m.gguf", dtype=torch.float32)
    assert info["architecture"] == arch and info["n_unused"] == 0
    x = torch.randint(0, tok.vocab_size, (2, 20))
    torch.testing.assert_close(loaded(x).logits, ours(x).logits, atol=1e-5, rtol=1e-5)


def test_quantized_gguf_is_close(tok, tmp_path):
    ours = make_model(tok, qk_norm=True, tie_embeddings=False)
    export_gguf(ours, tok, tmp_path / "q.gguf", "qwen3", quant=True)
    loaded, info = load_gguf_model(tmp_path / "q.gguf", dtype=torch.float32)
    assert info["tensor_types"].get("Q8_0", 0) > 0
    x = torch.randint(0, tok.vocab_size, (1, 24))
    p_ours, p_q = ours(x).logits.softmax(-1), loaded(x).logits.softmax(-1)
    assert (p_ours - p_q).abs().max() < 0.02  # 8-bit weights: tiny differences only


def test_gguf_tokenizer_matches(tok, tmp_path):
    export_gguf(make_model(tok), tok, tmp_path / "m.gguf", "llama")
    gt = tokenizer_from_gguf(tmp_path / "m.gguf")
    for s in ["Once upon a time, Lily found 123 shells.", "def f(x):\n    return x * 2", "café 日本 🧠"]:
        assert gt.encode(s) == tok.encode(s)
        assert gt.decode(gt.encode(s)) == s
    assert gt.bos_id == tok.bos_id and gt.eos_id == tok.eos_id


def test_detect_and_load_any(tok, tmp_path, tiny_data):
    m = make_model(tok)
    hf_dir = save_hf_model(m, tmp_path / "hf", tokenizer=tok)
    export_gguf(m, tok, tmp_path / "m.gguf", "llama")
    assert detect(hf_dir) == "hf" and detect(tmp_path / "m.gguf") == "gguf"
    x = torch.randint(0, tok.vocab_size, (1, 12))
    a, ta, _ = load_any(hf_dir, dtype=torch.float32)
    b, tb, _ = load_any(tmp_path / "m.gguf", dtype=torch.float32)
    assert ta.encode("a story") == tb.encode("a story") == tok.encode("a story")
    # The HF export is stored in bf16, so compare loosely.
    assert (a(x).logits - b(x).logits).abs().max() < 0.05


def test_analyze_external_model(tok, tmp_path, monkeypatch):
    monkeypatch.setenv("SLEUTH_RUNS", str(tmp_path / "runs"))
    hf_dir = save_hf_model(make_model(tok), tmp_path / "hf-model", tokenizer=tok)
    run = prepare_external_run(hf_dir, dtype=torch.float32)
    assert run.name.startswith(f"ext-{tmp_path.name.lower()}-hf-model-")
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["external"]["format"] == "hf" and cfg["model"]["n_layers"] == 2
    out = run_timeline(run, "suites/basic", views=["attention", "logit_lens"], log=lambda s: None)
    t = json.loads(out["logit_lens"].read_text())
    assert t["steps"] == [0]
    assert (run / "analysis" / "attention" / "basic" / "step_0000000" / "summary.json").exists()


def test_compare_models(tok, tmp_path):
    from sleuth.analyzer.compare import compare_models
    from sleuth.analyzer.suites import load_suite

    m = make_model(tok, qk_norm=True, tie_embeddings=False)
    export_gguf(m, tok, tmp_path / "f32.gguf", "qwen3")
    export_gguf(m, tok, tmp_path / "q8.gguf", "qwen3", quant=True)
    cases = load_suite("suites/stories")[:10]

    def loader(p):
        return lambda: load_any(p, dtype=torch.float32)[:2]

    same = compare_models(loader(tmp_path / "f32.gguf"), lambda: (m, tok), cases)
    assert same["overall"]["top1_agreement"] == 1.0 and same["overall"]["kl_mean"] < 1e-4
    q = compare_models(loader(tmp_path / "f32.gguf"), loader(tmp_path / "q8.gguf"), cases)
    assert 0 < q["overall"]["kl_mean"] < 0.01 and q["overall"]["top1_agreement"] > 0.9


def test_external_run_names_are_unique(tok, tmp_path, monkeypatch):
    """Two models in folders with the same name ("hf") must not share results."""
    monkeypatch.setenv("SLEUTH_RUNS", str(tmp_path / "runs"))
    a = save_hf_model(make_model(tok), tmp_path / "qwen" / "hf", tokenizer=tok)
    b = save_hf_model(make_model(tok), tmp_path / "llama" / "hf", tokenizer=tok)
    ra, rb = prepare_external_run(a, dtype=torch.float32), prepare_external_run(b, dtype=torch.float32)
    assert ra != rb and ra.name.startswith("ext-qwen-hf-") and rb.name.startswith("ext-llama-hf-")
    assert prepare_external_run(a, dtype=torch.float32) == ra          # same model -> same folder
    with pytest.raises(ValueError, match="another model"):
        prepare_external_run(b, name=ra.name.removeprefix("ext-"), dtype=torch.float32)


def test_gguf_ties_identical_output_weight(tok, tmp_path):
    """llama.cpp's converter writes output.weight even when the model ties it to
    the embedding. If the bytes are identical, we tie them (saving vocab x d)."""
    m = make_model(tok, qk_norm=True, tie_embeddings=True)
    export_gguf(m, tok, tmp_path / "t.gguf", "qwen3", quant=True)
    from helpers import add_gguf_tensor_copy
    add_gguf_tensor_copy(tmp_path / "t.gguf", tmp_path / "t2.gguf", "token_embd.weight", "output.weight")
    loaded, info = load_gguf_model(tmp_path / "t2.gguf", dtype=torch.float32)
    assert loaded.cfg.tie_embeddings and info["tied_output"]
    assert loaded.num_params() == m.num_params()
    # A different output matrix stays separate.
    m2 = make_model(tok, qk_norm=True, tie_embeddings=False)
    export_gguf(m2, tok, tmp_path / "u.gguf", "qwen3", quant=True)
    assert not load_gguf_model(tmp_path / "u.gguf", dtype=torch.float32)[0].cfg.tie_embeddings


def test_requantize_model(tok):
    from sleuth.analyzer.compare import requantize_model

    m = make_model(tok, qk_norm=True)
    before = {n: p.clone() for n, p in m.named_parameters()}
    r = requantize_model(m, "q4_1", "blocks.*.mlp.*")
    assert r["n_tensors"] == 6 and 0.02 < r["rel_error_mean"] < 0.15
    for n, p in m.named_parameters():
        assert torch.equal(p, before[n]) == (".mlp." not in n)
    with pytest.raises(ValueError, match="choose from"):
        requantize_model(m, "q4_k")
