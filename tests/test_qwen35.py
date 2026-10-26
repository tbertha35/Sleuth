"""Qwen3.5 ("qwen35": Gated DeltaNet + gated attention), its MoE form ("qwen35moe",
Qwen3.6-35B-A3B) and their MTP drafter (formats/qwen35.py), on tiny random GGUFs. The model is checked against a plain
re-implementation of llama.cpp's formulas written here; the real 0.8B file was
checked against llama.cpp's perplexity on the owner's machine (round 8)."""

import math
import re

import torch
import torch.nn.functional as F
import pytest
from helpers import moe_reference, write_gguf

from sleuth.formats import gguf_quants as q
from sleuth.formats.qwen35 import load_qwen35, load_qwen35_drafter
from sleuth.forge.infer.generate import generate
from sleuth.forge.infer.sampling import SamplingConfig
from sleuth.forge.infer.speculative import speculative_generate

D, H, KV, HD, ROT, FFN, V = 32, 2, 1, 16, 8, 48, 50
CONV, DK, NK, NV = 4, 8, 2, 4          # 4 value heads on 2 key heads: tests the tiling
DI = NV * DK
L, EPS, THETA = 4, 1e-6, 10000.0       # layers 0-2 DeltaNet, 3 attention; block 4 = MTP
E, K_USED, EH, SH = 6, 2, 32, 32       # qwen35moe: 6 experts, 2 per token, expert / shared widths


def make_qwen35_gguf(path, seed=0, moe=False, quant=None):
    """moe: "qwen35moe" (routed + shared experts in every block, MTP included).
    quant: store the expert tensors in this GGUF type (T then holds their dequantized values)."""
    g = torch.Generator().manual_seed(seed)

    def rnd(*s, std=0.1):
        return torch.randn(*s, generator=g) * std

    def gain(n):
        return 1.0 + 0.2 * torch.randn(n, generator=g)

    T = {"token_embd.weight": rnd(V, D, std=0.3), "output_norm.weight": gain(D)}
    cd = 2 * NK * DK + DI
    for i in range(L + 1):
        b = f"blk.{i}."
        T[b + "attn_norm.weight"], T[b + "post_attention_norm.weight"] = gain(D), gain(D)
        if moe:
            T[b + "ffn_gate_inp.weight"] = rnd(E, D, std=0.5)
            T[b + "ffn_gate_exps.weight"], T[b + "ffn_up_exps.weight"] = rnd(E, EH, D, std=0.2), rnd(E, EH, D, std=0.2)
            T[b + "ffn_down_exps.weight"] = rnd(E, D, EH, std=0.2)
            T[b + "ffn_gate_inp_shexp.weight"] = rnd(D, std=0.3)
            T[b + "ffn_gate_shexp.weight"], T[b + "ffn_up_shexp.weight"] = rnd(SH, D), rnd(SH, D)
            T[b + "ffn_down_shexp.weight"] = rnd(D, SH)
        else:
            T[b + "ffn_gate.weight"], T[b + "ffn_up.weight"], T[b + "ffn_down.weight"] =                 rnd(FFN, D), rnd(FFN, D), rnd(D, FFN)
        if i < 3:
            T[b + "attn_qkv.weight"], T[b + "attn_gate.weight"] = rnd(cd, D), rnd(DI, D)
            T[b + "ssm_conv1d.weight"] = rnd(cd, CONV, std=0.4)
            T[b + "ssm_dt.bias"], T[b + "ssm_a"] = rnd(NV, std=0.5), -torch.rand(NV, generator=g) - 0.1
            T[b + "ssm_alpha.weight"], T[b + "ssm_beta.weight"] = rnd(NV, D), rnd(NV, D)
            T[b + "ssm_norm.weight"], T[b + "ssm_out.weight"] = gain(DK), rnd(D, DI)
        else:
            T[b + "attn_q.weight"], T[b + "attn_k.weight"] = rnd(H * HD * 2, D), rnd(KV * HD, D)
            T[b + "attn_v.weight"], T[b + "attn_output.weight"] = rnd(KV * HD, D), rnd(D, H * HD)
            T[b + "attn_q_norm.weight"], T[b + "attn_k_norm.weight"] = gain(HD), gain(HD)
        if i == L:
            T[b + "nextn.eh_proj.weight"] = rnd(D, 2 * D)
            T[b + "nextn.enorm.weight"], T[b + "nextn.hnorm.weight"] = gain(D), gain(D)
            T[b + "nextn.shared_head_norm.weight"] = gain(D)
    a = "qwen35moe" if moe else "qwen35"
    md = {"general.architecture": (8, a), f"{a}.block_count": (4, L + 1), f"{a}.context_length": (4, 256),
          f"{a}.embedding_length": (4, D),
          f"{a}.attention.head_count": (4, H), f"{a}.attention.head_count_kv": (4, KV),
          f"{a}.rope.freq_base": (6, THETA), f"{a}.attention.layer_norm_rms_epsilon": (6, EPS),
          f"{a}.attention.key_length": (4, HD), f"{a}.attention.value_length": (4, HD),
          f"{a}.rope.dimension_count": (4, ROT), f"{a}.rope.dimension_sections": (9, (5, [2, 1, 1, 0])),
          f"{a}.ssm.conv_kernel": (4, CONV), f"{a}.ssm.state_size": (4, DK), f"{a}.ssm.group_count": (4, NK),
          f"{a}.ssm.time_step_rank": (4, NV), f"{a}.ssm.inner_size": (4, DI),
          f"{a}.full_attention_interval": (4, 4), f"{a}.nextn_predict_layers": (4, 1)}
    if moe:
        md |= {f"{a}.expert_count": (4, E), f"{a}.expert_used_count": (4, K_USED),
               f"{a}.expert_feed_forward_length": (4, EH), f"{a}.expert_shared_feed_forward_length": (4, SH)}
    else:
        md[f"{a}.feed_forward_length"] = (4, FFN)
    tensors = {}
    for n, t in T.items():
        if quant is not None and "_exps." in n:
            raw = q.quantize(t, quant)
            T[n] = q.dequantize(raw, quant, t.numel()).view(t.shape)     # what the file really holds
            tensors[n] = (quant, tuple(t.shape), bytes(raw.numpy()))
        else:
            tensors[n] = (q.F32, tuple(t.shape), t.float().numpy().tobytes())
    write_gguf(path, md, tensors)
    return T


def _rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def reference_logits(T, ids):
    """llama.cpp's qwen35 graph, token by token, with plain loops."""
    n = len(ids)
    x = T["token_embd.weight"][ids]                                  # [n, D]
    for i in range(L):
        b = f"blk.{i}."
        h = _rms(x, T[b + "attn_norm.weight"])
        if i < 3:
            qkv = h @ T[b + "attn_qkv.weight"].T                       # [n, C]
            C = qkv.shape[1]
            padded = torch.cat([torch.zeros(CONV - 1, C), qkv])
            conv = torch.stack([(padded[t:t + CONV] * T[b + "ssm_conv1d.weight"].T).sum(0) for t in range(n)])
            conv = F.silu(conv)
            qh = conv[:, :NK * DK].view(n, NK, DK)
            kh = conv[:, NK * DK:2 * NK * DK].view(n, NK, DK)
            vh = conv[:, 2 * NK * DK:].view(n, NV, DK)
            qh = qh / qh.norm(dim=-1, keepdim=True).clamp_min(EPS) / math.sqrt(DK)
            kh = kh / kh.norm(dim=-1, keepdim=True).clamp_min(EPS)
            beta = torch.sigmoid(h @ T[b + "ssm_beta.weight"].T)
            gl = F.softplus(h @ T[b + "ssm_alpha.weight"].T + T[b + "ssm_dt.bias"]) * T[b + "ssm_a"]
            z = h @ T[b + "attn_gate.weight"].T
            out = torch.zeros(n, NV, DK)
            for hv in range(NV):
                hk = hv % NK
                M = torch.zeros(DK, DK)
                for t in range(n):
                    M = M * math.exp(gl[t, hv])
                    k, v, qq = kh[t, hk], vh[t, hv], qh[t, hk]
                    M = M + torch.outer((v - M @ k) * beta[t, hv], k)
                    out[t, hv] = M @ qq
            o = _rms(out, T[b + "ssm_norm.weight"]) * F.silu(z.view(n, NV, DK))
            a = o.reshape(n, DI) @ T[b + "ssm_out.weight"].T
        else:
            qg = (h @ T[b + "attn_q.weight"].T).view(n, H, 2 * HD)
            qv, gate = _rms(qg[..., :HD], T[b + "attn_q_norm.weight"]), qg[..., HD:]
            k = _rms((h @ T[b + "attn_k.weight"].T).view(n, KV, HD), T[b + "attn_k_norm.weight"])
            v = (h @ T[b + "attn_v.weight"].T).view(n, KV, HD)
            inv = 1.0 / THETA ** (torch.arange(0, ROT, 2).float() / ROT)

            def rope(x_):
                y = x_.clone()
                for t in range(n):
                    for j in range(ROT // 2):
                        c, s = math.cos(t * inv[j]), math.sin(t * inv[j])
                        a0, a1 = x_[t, :, j].clone(), x_[t, :, j + ROT // 2].clone()
                        y[t, :, j], y[t, :, j + ROT // 2] = a0 * c - a1 * s, a0 * s + a1 * c
                return y
            qv, k = rope(qv), rope(k)
            att = torch.zeros(n, H, HD)
            for hq in range(H):
                sc = (qv[:, hq] @ k[:, hq // (H // KV)].T) / math.sqrt(HD)
                sc = sc.masked_fill(torch.ones(n, n).triu(1).bool(), float("-inf"))
                att[:, hq] = sc.softmax(-1) @ v[:, hq // (H // KV)]
            a = (att.reshape(n, H * HD) * torch.sigmoid(gate.reshape(n, H * HD))) @ T[b + "attn_output.weight"].T
        x = x + a
        x = x + _ref_ffn(T, b, _rms(x, T[b + "post_attention_norm.weight"]))
    return _rms(x, T["output_norm.weight"]) @ T["token_embd.weight"].T


def _ref_ffn(T, b, m):
    if b + "ffn_gate_inp.weight" not in T:
        return (F.silu(m @ T[b + "ffn_gate.weight"].T) * (m @ T[b + "ffn_up.weight"].T)) @ T[b + "ffn_down.weight"].T
    # qwen35moe (llama.cpp qwen35moe.cpp build_layer_ffn): routed experts + gated shared expert
    routed = moe_reference(m, m @ T[b + "ffn_gate_inp.weight"].T, K_USED, T[b + "ffn_gate_exps.weight"],
                           T[b + "ffn_up_exps.weight"], T[b + "ffn_down_exps.weight"], F.silu)
    shared = (F.silu(m @ T[b + "ffn_gate_shexp.weight"].T) * (m @ T[b + "ffn_up_shexp.weight"].T))         @ T[b + "ffn_down_shexp.weight"].T
    return routed + torch.sigmoid(m @ T[b + "ffn_gate_inp_shexp.weight"]).unsqueeze(-1) * shared


def test_matches_reference(tmp_path):
    T = make_qwen35_gguf(tmp_path / "m.gguf")
    m, info = load_qwen35(tmp_path / "m.gguf", dtype=torch.float32)
    assert info["has_mtp"] and m.cfg.n_layers == L and [m.cfg.recurrent(i) for i in range(L)] == [1, 1, 1, 0]
    ids = [3, 17, 5, 40, 22, 9, 31, 12, 7]
    with torch.no_grad():
        got = m(torch.tensor([ids])).logits[0]
    ref = reference_logits(T, ids)
    assert torch.allclose(got, ref, atol=1e-4), (got - ref).abs().max()


def test_cached_decoding_and_speculative_are_exact(tmp_path):
    make_qwen35_gguf(tmp_path / "m.gguf")
    m, _ = load_qwen35(tmp_path / "m.gguf", dtype=torch.float32)
    prompt, greedy = [3, 17, 5, 40, 22, 9], SamplingConfig(temperature=0)
    plain = generate(m, prompt, 30, greedy)
    with torch.no_grad():
        full = m(torch.tensor([prompt + plain.new_ids[:-1]])).logits[0, len(prompt) - 1:].argmax(-1).tolist()
    assert full == plain.new_ids
    drafter = load_qwen35_drafter(tmp_path / "m.gguf", m)
    for k in (1, 3):
        spec = speculative_generate(m, prompt, 30, greedy, draft=drafter, k=k)
        assert spec.new_ids == plain.new_ids, k
        assert spec.stats["speculative"]["drafted"] > 0
    # Wrong guesses force rollbacks of the recurrent states: still exact.
    bad = speculative_generate(m, prompt, 30, greedy, draft=lambda ids, k: [1] * k, k=3)
    assert bad.new_ids == plain.new_ids


def test_mtp_catch_up_matches_a_fresh_run(tmp_path):
    """After a check, the drafter's KV for the accepted tokens equals what a fresh
    run over the true (token, previous hidden) pairs would write."""
    make_qwen35_gguf(tmp_path / "m.gguf")
    m, _ = load_qwen35(tmp_path / "m.gguf", dtype=torch.float32)
    drafter = load_qwen35_drafter(tmp_path / "m.gguf", m)
    prompt = [3, 17, 5, 40, 22, 9]
    spec = speculative_generate(m, prompt, 12, SamplingConfig(temperature=0), draft=drafter, k=3)
    seq = prompt + spec.new_ids[:-1]                       # every token the target has stored
    with torch.no_grad():
        hid = m(torch.tensor([seq]), return_hidden=True).hidden[0]
        fresh = drafter.mtp.new_cache(256, torch.float32)
        drafter.mtp(torch.tensor(seq), torch.cat([torch.zeros_like(hid[:1]), hid[:-1]]), fresh, 0)
    n = len(seq)
    assert torch.allclose(drafter.cache.k[0][:, :, :n], fresh.k[0][:, :, :n], atol=1e-5)


def test_qwen35_pretokenizer_keeps_combining_marks():
    from sleuth.formats.gguf_model import QWEN35_PATTERN
    from sleuth.formats.hf_tokenizer import translate_regex

    p = re.compile(translate_regex(QWEN35_PATTERN))
    assert p.findall("café नमस्ते 12") == ["café", " नमस्ते", " ", "1", "2"]


@pytest.mark.parametrize("quant", [None, q.Q8_0])
def test_moe_matches_reference(tmp_path, quant):
    """qwen35moe: same as the reference, with plain or quantized (kept quantized) experts."""
    from sleuth.forge.model.moe import MoE

    T = make_qwen35_gguf(tmp_path / "m.gguf", moe=True, quant=quant)
    m, info = load_qwen35(tmp_path / "m.gguf", dtype=torch.float32, keep_quantized=True)
    assert info["architecture"] == "qwen35moe"
    moes = [x for x in m.modules() if isinstance(x, MoE)]
    assert len(moes) == L and all(x.n_experts == E and x.n_used == K_USED for x in moes)
    assert (moes[0].down.ggml_type is None) == (quant is None)
    ids = [3, 17, 5, 40, 22, 9, 31, 12, 7]
    with torch.no_grad():
        got = m(torch.tensor([ids])).logits[0]
    ref = reference_logits(T, ids)
    assert torch.allclose(got, ref, atol=1e-4), (got - ref).abs().max()
    assert int(moes[0].counts.sum()) == len(ids) * K_USED          # every token picked K_USED experts
    assert (moes[0].counts > 0).sum() > K_USED                     # ... and not always the same ones


def test_moe_cached_speculative_and_mtp(tmp_path):
    """Cached decoding = full forward; MTP drafter (an MoE block too) keeps greedy text exact."""
    make_qwen35_gguf(tmp_path / "m.gguf", moe=True)
    m, _ = load_qwen35(tmp_path / "m.gguf", dtype=torch.float32)
    prompt, greedy = [3, 17, 5, 40, 22, 9], SamplingConfig(temperature=0)
    plain = generate(m, prompt, 24, greedy)
    with torch.no_grad():
        full = m(torch.tensor([prompt + plain.new_ids[:-1]])).logits[0, len(prompt) - 1:].argmax(-1).tolist()
    assert full == plain.new_ids
    drafter = load_qwen35_drafter(tmp_path / "m.gguf", m)
    spec = speculative_generate(m, prompt, 24, greedy, draft=drafter, k=3)
    assert spec.new_ids == plain.new_ids and spec.stats["speculative"]["drafted"] > 0
