"""Loading GPT-2, GPT-NeoX (Pythia) and Gemma 3 checkpoints (formats/hf.py FAMILIES).

Each test writes a tiny random checkpoint in the family's Hugging Face layout and
compares our model's logits with a direct implementation of that architecture,
written here from its definition (independent of our Transformer)."""

import json
import math

import torch
import torch.nn.functional as F

from sleuth.formats.hf import load_hf_model
from sleuth.formats.safetensors import save_file

V, T = 50, 12


def _save(d, cfg: dict, tensors: dict):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg))
    save_file({k: v.contiguous() for k, v in tensors.items()}, d / "model.safetensors")


def _causal_attn(q, k, v, scale, window=0):
    """q [B,H,T,hd], k/v [B,H,T,hd] -> [B,H,T,hd]."""
    Tq = q.shape[2]
    s = (q @ k.transpose(-1, -2)) * scale
    i = torch.arange(Tq)[:, None]
    j = torch.arange(Tq)[None, :]
    ok = j <= i
    if window:
        ok &= i - j < window
    return s.masked_fill(~ok, float("-inf")).softmax(-1) @ v


def _ln(x, w, b, eps=1e-5):
    return F.layer_norm(x, (x.shape[-1],), w, b, eps)


def _rot_half(x):
    a, b = x.chunk(2, -1)
    return torch.cat([-b, a], -1)


def _rope(x, base, dim):
    """Rotary on the first `dim` channels (rotate-half within them), as GPT-NeoX / HF."""
    inv = 1.0 / base ** (torch.arange(0, dim, 2).float() / dim)
    ang = torch.outer(torch.arange(x.shape[2]).float(), inv)
    cos, sin = torch.cat([ang, ang], -1).cos(), torch.cat([ang, ang], -1).sin()
    r, rest = x[..., :dim], x[..., dim:]
    return torch.cat([r * cos + _rot_half(r) * sin, rest], -1)


def test_gpt2(tmp_path):
    torch.manual_seed(0)
    d, H, L = 32, 4, 2
    hd = d // H
    t = {"wte.weight": torch.randn(V, d) * 0.3, "wpe.weight": torch.randn(64, d) * 0.3,
         "ln_f.weight": 1 + 0.1 * torch.randn(d), "ln_f.bias": 0.1 * torch.randn(d)}
    for i in range(L):
        p = f"h.{i}."
        for ln in ("ln_1", "ln_2"):
            t[p + ln + ".weight"], t[p + ln + ".bias"] = 1 + 0.1 * torch.randn(d), 0.1 * torch.randn(d)
        for name, (i_, o_) in {"attn.c_attn": (d, 3 * d), "attn.c_proj": (d, d), "mlp.c_fc": (d, 4 * d),
                               "mlp.c_proj": (4 * d, d)}.items():
            t[p + name + ".weight"] = torch.randn(i_, o_) / math.sqrt(i_)   # Conv1D: [in, out]
            t[p + name + ".bias"] = 0.1 * torch.randn(o_)
        t[p + "attn.bias"] = torch.ones(1, 1, 64, 64)                       # the causal-mask buffer: skipped
    cfg = {"model_type": "gpt2", "n_embd": d, "n_head": H, "n_layer": L, "n_positions": 64, "vocab_size": V,
           "activation_function": "gelu_new", "layer_norm_epsilon": 1e-5}
    _save(tmp_path / "gpt2", cfg, t)

    ids = torch.randint(0, V, (2, T))
    x = t["wte.weight"][ids] + t["wpe.weight"][:T]
    for i in range(L):
        p = f"h.{i}."
        h = _ln(x, t[p + "ln_1.weight"], t[p + "ln_1.bias"])
        q, k, v = (h @ t[p + "attn.c_attn.weight"] + t[p + "attn.c_attn.bias"]).split(d, -1)
        q, k, v = (z.view(2, T, H, hd).transpose(1, 2) for z in (q, k, v))
        y = _causal_attn(q, k, v, 1 / math.sqrt(hd)).transpose(1, 2).reshape(2, T, d)
        x = x + y @ t[p + "attn.c_proj.weight"] + t[p + "attn.c_proj.bias"]
        h = _ln(x, t[p + "ln_2.weight"], t[p + "ln_2.bias"])
        m = F.gelu(h @ t[p + "mlp.c_fc.weight"] + t[p + "mlp.c_fc.bias"], approximate="tanh")
        x = x + m @ t[p + "mlp.c_proj.weight"] + t[p + "mlp.c_proj.bias"]
    ref = _ln(x, t["ln_f.weight"], t["ln_f.bias"]) @ t["wte.weight"].T

    model, info = load_hf_model(tmp_path / "gpt2", dtype=torch.float32, max_seq_len=64)
    assert model.cfg.pos_type == "learned" and info["n_unused"] == 0
    with torch.no_grad():
        torch.testing.assert_close(model(ids).logits, ref, atol=1e-4, rtol=1e-4)
        # cached decoding adds the right position vectors too
        from sleuth.forge.infer.kv_cache import KVCache

        cache = KVCache.for_model(model, 2, 64, torch.float32)
        model(ids[:, :7], cache=cache)
        torch.testing.assert_close(model(ids[:, 7:], cache=cache).logits, ref[:, 7:], atol=1e-4, rtol=1e-4)


def test_gpt_neox_pythia(tmp_path):
    torch.manual_seed(1)
    d, H, L, rot = 32, 4, 2, 0.25
    hd = d // H
    rd = int(hd * rot)
    t = {"gpt_neox.embed_in.weight": torch.randn(V, d) * 0.3, "embed_out.weight": torch.randn(V, d) * 0.3,
         "gpt_neox.final_layer_norm.weight": 1 + 0.1 * torch.randn(d),
         "gpt_neox.final_layer_norm.bias": 0.1 * torch.randn(d)}
    for i in range(L):
        p = f"gpt_neox.layers.{i}."
        for ln in ("input_layernorm", "post_attention_layernorm"):
            t[p + ln + ".weight"], t[p + ln + ".bias"] = 1 + 0.1 * torch.randn(d), 0.1 * torch.randn(d)
        for name, (o_, i_) in {"attention.query_key_value": (3 * d, d), "attention.dense": (d, d),
                               "mlp.dense_h_to_4h": (4 * d, d), "mlp.dense_4h_to_h": (d, 4 * d)}.items():
            t[p + name + ".weight"] = torch.randn(o_, i_) / math.sqrt(i_)
            t[p + name + ".bias"] = 0.1 * torch.randn(o_)
        t[p + "attention.rotary_emb.inv_freq"] = torch.ones(rd // 2)
    cfg = {"model_type": "gpt_neox", "hidden_size": d, "num_attention_heads": H, "num_hidden_layers": L,
           "intermediate_size": 4 * d, "max_position_embeddings": 64, "rotary_pct": rot, "rotary_emb_base": 10000,
           "use_parallel_residual": True, "hidden_act": "gelu", "vocab_size": V, "tie_word_embeddings": False}
    _save(tmp_path / "neox", cfg, t)

    ids = torch.randint(0, V, (2, T))
    x = t["gpt_neox.embed_in.weight"][ids]
    for i in range(L):
        p = f"gpt_neox.layers.{i}."
        h = _ln(x, t[p + "input_layernorm.weight"], t[p + "input_layernorm.bias"])
        qkv = h @ t[p + "attention.query_key_value.weight"].T + t[p + "attention.query_key_value.bias"]
        qkv = qkv.view(2, T, H, 3 * hd).transpose(1, 2)                  # HF: per head [q | k | v]
        q, k, v = qkv[..., :hd], qkv[..., hd:2 * hd], qkv[..., 2 * hd:]
        q, k = _rope(q, 10000, rd), _rope(k, 10000, rd)
        y = _causal_attn(q, k, v, 1 / math.sqrt(hd)).transpose(1, 2).reshape(2, T, d)
        a = y @ t[p + "attention.dense.weight"].T + t[p + "attention.dense.bias"]
        h2 = _ln(x, t[p + "post_attention_layernorm.weight"], t[p + "post_attention_layernorm.bias"])
        m = F.gelu(h2 @ t[p + "mlp.dense_h_to_4h.weight"].T + t[p + "mlp.dense_h_to_4h.bias"])
        m = m @ t[p + "mlp.dense_4h_to_h.weight"].T + t[p + "mlp.dense_4h_to_h.bias"]
        x = x + a + m                                                      # parallel residual
    ref = _ln(x, t["gpt_neox.final_layer_norm.weight"], t["gpt_neox.final_layer_norm.bias"]) @ t["embed_out.weight"].T

    model, _ = load_hf_model(tmp_path / "neox", dtype=torch.float32, max_seq_len=64)
    assert model.cfg.parallel_residual and model.cfg.rope_dim == rd and not model.cfg.tie_embeddings
    with torch.no_grad():
        torch.testing.assert_close(model(ids).logits, ref, atol=1e-4, rtol=1e-4)


def test_gemma3(tmp_path):
    torch.manual_seed(2)
    d, H, K, hd, L, W = 32, 2, 1, 16, 3, 4
    types = ["sliding_attention", "sliding_attention", "full_attention"]
    t = {"model.embed_tokens.weight": torch.randn(V, d) * 0.3, "model.norm.weight": 0.1 * torch.randn(d)}
    for i in range(L):
        p = f"model.layers.{i}."
        for n in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                  "post_feedforward_layernorm"):
            t[p + n + ".weight"] = 0.1 * torch.randn(d)
        t[p + "self_attn.q_norm.weight"], t[p + "self_attn.k_norm.weight"] = 0.1 * torch.randn(hd), 0.1 * torch.randn(hd)
        for n, (o_, i_) in {"q": (H * hd, d), "k": (K * hd, d), "v": (K * hd, d), "o": (d, H * hd)}.items():
            t[p + f"self_attn.{n}_proj.weight"] = torch.randn(o_, i_) / math.sqrt(i_)
        for n, (o_, i_) in {"gate": (64, d), "up": (64, d), "down": (d, 64)}.items():
            t[p + f"mlp.{n}_proj.weight"] = torch.randn(o_, i_) / math.sqrt(i_)
    cfg = {"model_type": "gemma3_text", "hidden_size": d, "num_attention_heads": H, "num_key_value_heads": K,
           "head_dim": hd, "num_hidden_layers": L, "intermediate_size": 64, "max_position_embeddings": 64,
           "rms_norm_eps": 1e-6, "rope_theta": 1e6, "rope_local_base_freq": 1e4, "query_pre_attn_scalar": 8,
           "sliding_window": W, "layer_types": types, "vocab_size": V}
    _save(tmp_path / "gemma3", cfg, t)

    def rms(x, w):   # Gemma: y = x / rms(x) * (1 + w)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * (1 + w)

    ids = torch.randint(0, V, (2, T))
    x = t["model.embed_tokens.weight"][ids] * math.sqrt(d)
    for i in range(L):
        p = f"model.layers.{i}."
        h = rms(x, t[p + "input_layernorm.weight"])
        q = (h @ t[p + "self_attn.q_proj.weight"].T).view(2, T, H, hd).transpose(1, 2)
        k = (h @ t[p + "self_attn.k_proj.weight"].T).view(2, T, K, hd).transpose(1, 2)
        v = (h @ t[p + "self_attn.v_proj.weight"].T).view(2, T, K, hd).transpose(1, 2)
        q, k = rms(q, t[p + "self_attn.q_norm.weight"]), rms(k, t[p + "self_attn.k_norm.weight"])
        sliding = types[i] == "sliding_attention"
        base = 1e4 if sliding else 1e6
        q, k = _rope(q, base, hd), _rope(k, base, hd)
        k, v = k.repeat_interleave(H // K, 1), v.repeat_interleave(H // K, 1)
        y = _causal_attn(q, k, v, 8 ** -0.5, W if sliding else 0).transpose(1, 2).reshape(2, T, H * hd)
        x = x + rms(y @ t[p + "self_attn.o_proj.weight"].T, t[p + "post_attention_layernorm.weight"])
        h = rms(x, t[p + "pre_feedforward_layernorm.weight"])
        m = F.gelu(h @ t[p + "mlp.gate_proj.weight"].T, approximate="tanh") * (h @ t[p + "mlp.up_proj.weight"].T)
        x = x + rms(m @ t[p + "mlp.down_proj.weight"].T, t[p + "post_feedforward_layernorm.weight"])
    ref = rms(x, t["model.norm.weight"]) @ t["model.embed_tokens.weight"].T

    model, _ = load_hf_model(tmp_path / "gemma3", dtype=torch.float32, max_seq_len=64)
    assert model.cfg.layer_types == ["sliding", "sliding", "full"] and model.cfg.sliding_window == W
    with torch.no_grad():
        torch.testing.assert_close(model(ids).logits, ref, atol=1e-4, rtol=1e-4)
