"""Gemma 4 GGUF support: model loading (dequantized and kept quantized) and the tokenizer.

The model check compares our Transformer against `reference_forward` below: a
direct, plain-tensor transcription of llama.cpp's graph for this architecture
(src/models/gemma4.cpp) that uses none of our model code. The tiny test file
has the same structure as the real 12B file (`sleuth inspect` output): 5
sliding layers per full layer, full layers with wider heads, a single KV head
and no attn_v, rope_freqs, layer_output_scale, Q4_0/Q8_0 weights. With moe=True
it has the 26B-A4B's MoE layers too (merged gate_up experts, per-expert down scales).
"""

import heapq
import math
import random

import pytest
import torch
import torch.nn.functional as F
from helpers import moe_reference, write_gguf

from sleuth.forge.model.quant import QuantEmbedding, QuantLinear
from sleuth.formats import gguf_quants as q
from sleuth.formats.gemma_tokenizer import BYTE, CONTROL, NORMAL, USER_DEFINED, GemmaTokenizer
from sleuth.formats.gguf import GGUFFile
from sleuth.formats.gguf_model import load_gguf_model, tokenizer_from_gguf
from sleuth.models import load_any

# --------------------------------------------------------------------------- #
# a tiny tokenizer vocabulary in Gemma 4's style
# --------------------------------------------------------------------------- #

SPECIALS = [("<pad>", CONTROL), ("<eos>", CONTROL), ("<bos>", CONTROL), ("<unk>", 2), ("<mask>", CONTROL),
            ("<|turn>", CONTROL), ("<turn|>", CONTROL), ("<|tool>", USER_DEFINED)]
CHAT_TOKENS = ["<|channel>", "<channel|>", "<|think|>"]
CHARS = list("abcdefghijklmnopqrstuvwxyzTO.,!?'") + ["▁", "\n", "\n\n"]
MERGES = ["▁ t", "h e", "▁t he", "a n", "▁ a", "▁a n", "i s", "▁ c", "a t", "▁c at",
          "▁ ▁", "▁▁ ▁▁", "c e", "n ce", "O nce"]


def tiny_vocab():
    tokens, types = [], []
    for t, k in SPECIALS:
        tokens.append(t)
        types.append(k)
    for b in range(256):
        tokens.append(f"<0x{b:02X}>")
        types.append(BYTE)
    for c in CHARS:
        tokens.append(c)
        types.append(NORMAL)
    for m in MERGES:
        a, b = m.split(" ", 1)
        if a + b not in tokens:
            tokens.append(a + b)
            types.append(NORMAL)
    for t in CHAT_TOKENS:  # at the end, so the IDs used elsewhere in these tests don't move
        tokens.append(t)
        types.append(CONTROL)
    return tokens, types


def tiny_tokenizer():
    tokens, types = tiny_vocab()
    return GemmaTokenizer(tokens, types, MERGES, bos_id=2, eos_id=1)


def llama_cpp_bpe(word: str, ranks: dict) -> list[str]:
    """llm_tokenizer_bpe_session's merge loop (llama-vocab.cpp), transcribed:
    a priority queue of adjacent pairs ordered by (rank, left position);
    pop the best, skip it if outdated, merge, push the new neighbours."""
    syms = list(word)
    alive = [True] * len(syms)
    prev = list(range(-1, len(syms) - 1))
    nxt = list(range(1, len(syms))) + [-1]
    queue = []

    def push(l, r):
        if l == -1 or r == -1:
            return
        rank = ranks.get((syms[l], syms[r]))
        if rank is not None:
            heapq.heappush(queue, (rank, l, r, syms[l] + syms[r]))

    for i in range(1, len(syms)):
        push(i - 1, i)
    while queue:
        _, l, r, text = heapq.heappop(queue)
        if not alive[l] or not alive[r] or syms[l] + syms[r] != text or nxt[l] != r:
            continue
        syms[l] = text
        alive[r] = False
        nxt[l] = nxt[r]
        if nxt[r] != -1:
            prev[nxt[r]] = l
        push(prev[l], l)
        push(l, nxt[l])
    return [s for s, a in zip(syms, alive) if a]


def test_tokenizer_basics():
    tok = tiny_tokenizer()
    ids = tok.encode("the cat")
    assert [tok.id_to_token[i] for i in ids] == ["t", "he", "▁cat"]
    assert tok.decode(ids) == "the cat"
    # Spaces become ▁ and may merge across words; newline runs that are tokens stay whole.
    assert [tok.id_to_token[i] for i in tok.encode("an  ant\n\nOnce")] == \
        ["an", "▁▁", "an", "t", "\n\n", "Once"]
    # Characters not in the vocabulary fall back to UTF-8 byte tokens.
    ids = tok.encode("é!")
    assert [tok.id_to_token[i] for i in ids] == ["<0xC3>", "<0xA9>", "!"]
    assert tok.decode(ids) == "é!"


def test_tokenizer_special_tokens():
    tok = tiny_tokenizer()
    text = "<|turn>the cat<turn|>"
    plain = tok.encode(text)                      # control tokens not allowed: spelled out
    assert tok.special_tokens["<|turn>"] not in plain
    assert tok.decode(plain) == text
    ids = tok.encode(text, allowed_special="all")
    assert ids[0] == tok.special_tokens["<|turn>"] and ids[-1] == tok.special_tokens["<turn|>"]
    # User-defined tokens are always cut out first (as in llama.cpp).
    assert tok.encode("a<|tool>a") == [tok.token_to_id["a"], tok.user_defined["<|tool>"], tok.token_to_id["a"]]
    assert tok.start_id == 2 and tok.eos_id == 1


def test_our_merge_loop_equals_llama_cpp_queue():
    """Our BPE merges every occurrence of the best pair per pass; llama.cpp pops
    pairs from a priority queue. They should agree, including on repeats ("aaaa")."""
    rng = random.Random(0)
    alphabet = "ab▁c"
    vocab = set(alphabet)
    merges = []
    for _ in range(25):
        a, b = rng.choice(sorted(vocab)), rng.choice(sorted(vocab))
        if f"{a} {b}" not in merges:
            merges.append(f"{a} {b}")
            vocab.add(a + b)
    tokens = sorted(vocab, key=lambda t: (len(t), t))
    tok = GemmaTokenizer(tokens, [NORMAL] * len(tokens), merges, None, None)
    for _ in range(300):
        word = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 14)))
        ours = [tok.id_to_token[i] for i in tok._bpe(word)]
        assert ours == llama_cpp_bpe(word, tok.ranks), word


# --------------------------------------------------------------------------- #
# a tiny Gemma 4 GGUF
# --------------------------------------------------------------------------- #

D, H, FFN, WINDOW = 64, 4, 96, 3
PATTERN = [True, True, False, True, True, False]       # True = sliding (5:1 in the real model; 2:1 here)
HD_FULL, HD_SWA, KV_FULL, KV_SWA = 32, 16, 1, 2
BASE, BASE_SWA, CAP = 1e6, 1e4, 30.0
N_EXP, N_USED, EXP_FF = 6, 2, 32                       # MoE: 128 / 8 / 704 in the real 26B-A4B


def make_gemma_gguf(path, quant: int | None = None, seed: int = 0, moe: bool = False):
    """Write a random tiny Gemma 4 with real-looking metadata. Norm gains around 1,
    like real ones; weights scaled so activations stay sensible. quant also applies
    to the experts (not the router: F32 in the real file)."""
    g = torch.Generator().manual_seed(seed)
    tokens, types = tiny_vocab()
    V = len(tokens)

    def rnd(*shape, std=0.08):
        return torch.randn(*shape, generator=g) * std

    def gain(n):
        return 1.0 + 0.2 * torch.randn(n, generator=g)

    T = {"token_embd.weight": rnd(V, D, std=0.05), "output_norm.weight": gain(D)}
    ff = torch.ones(HD_FULL // 2)
    ff[HD_FULL // 4:] = 1e30                               # "proportional RoPE": half the pairs don't rotate
    T["rope_freqs.weight"] = ff
    for i, swa in enumerate(PATTERN):
        hd, kv = (HD_SWA, KV_SWA) if swa else (HD_FULL, KV_FULL)
        b = f"blk.{i}."
        T[b + "attn_norm.weight"] = gain(D)
        T[b + "attn_q.weight"] = rnd(H * hd, D)
        T[b + "attn_k.weight"] = rnd(kv * hd, D)
        if swa:
            T[b + "attn_v.weight"] = rnd(kv * hd, D)
        T[b + "attn_q_norm.weight"] = gain(hd)
        T[b + "attn_k_norm.weight"] = gain(hd)
        T[b + "attn_output.weight"] = rnd(D, H * hd)
        T[b + "post_attention_norm.weight"] = gain(D)
        T[b + "ffn_norm.weight"] = gain(D)
        T[b + "ffn_gate.weight"] = rnd(FFN, D)
        T[b + "ffn_up.weight"] = rnd(FFN, D)
        T[b + "ffn_down.weight"] = rnd(D, FFN)
        T[b + "post_ffw_norm.weight"] = gain(D)
        T[b + "layer_output_scale.weight"] = torch.tensor([0.5 + 0.1 * i])
        if moe:
            T[b + "ffn_gate_inp.weight"] = rnd(N_EXP, D, std=0.5)
            T[b + "ffn_gate_inp.scale"] = gain(D)
            T[b + "ffn_gate_up_exps.weight"] = rnd(N_EXP, 2 * EXP_FF, D)
            T[b + "ffn_down_exps.weight"] = rnd(N_EXP, D, EXP_FF)
            T[b + "ffn_down_exps.scale"] = 0.5 + torch.rand(N_EXP, generator=g)
            for n in ("pre_ffw_norm_2", "post_ffw_norm_1", "post_ffw_norm_2"):
                T[b + n + ".weight"] = gain(D)
    tensors = {}
    for name, t in T.items():
        if quant is not None and t.ndim in (2, 3) and "ffn_gate_inp" not in name:
            tensors[name] = (quant, tuple(t.shape), bytes(q.quantize(t, quant).numpy()))
        else:
            tensors[name] = (q.F32, tuple(t.shape), t.float().numpy().tobytes())
    a = "gemma4"
    md = {
        "general.architecture": (8, a),
        "general.name": (8, "tiny gemma4"),
        f"{a}.block_count": (4, len(PATTERN)),
        f"{a}.context_length": (4, 262144),
        f"{a}.embedding_length": (4, D),
        f"{a}.feed_forward_length": (4, FFN),
        f"{a}.attention.head_count": (4, H),
        f"{a}.attention.head_count_kv": (9, (5, [KV_SWA if s else KV_FULL for s in PATTERN])),
        f"{a}.rope.freq_base": (6, BASE),
        f"{a}.rope.freq_base_swa": (6, BASE_SWA),
        f"{a}.attention.layer_norm_rms_epsilon": (6, 1e-6),
        f"{a}.attention.key_length": (4, HD_FULL),
        f"{a}.attention.value_length": (4, HD_FULL),
        f"{a}.final_logit_softcapping": (6, CAP),
        f"{a}.attention.sliding_window": (4, WINDOW),
        f"{a}.attention.shared_kv_layers": (4, 0),
        f"{a}.embedding_length_per_layer_input": (4, 0),
        f"{a}.attention.sliding_window_pattern": (9, (7, PATTERN)),
        f"{a}.attention.key_length_swa": (4, HD_SWA),
        f"{a}.attention.value_length_swa": (4, HD_SWA),
        f"{a}.rope.dimension_count": (4, HD_FULL),
        f"{a}.rope.dimension_count_swa": (4, HD_SWA),
        **({f"{a}.expert_count": (4, N_EXP), f"{a}.expert_used_count": (4, N_USED),
            f"{a}.expert_feed_forward_length": (4, EXP_FF)} if moe else {}),
        "tokenizer.ggml.model": (8, "gemma4"),
        "tokenizer.ggml.tokens": (9, (8, tokens)),
        "tokenizer.ggml.token_type": (9, (5, types)),
        "tokenizer.ggml.merges": (9, (8, MERGES)),
        "tokenizer.ggml.bos_token_id": (4, 2),
        "tokenizer.ggml.eos_token_id": (4, 1),
        "tokenizer.ggml.eot_token_id": (4, 6),
        "tokenizer.ggml.add_bos_token": (7, True),
    }
    write_gguf(path, md, tensors)


# --------------------------------------------------------------------------- #
# reference: llama.cpp's gemma4 graph, written out with plain tensors
# --------------------------------------------------------------------------- #

def rms(x, w=None, eps=1e-6):                      # ggml_rms_norm (+ optional gain)
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y * w if w is not None else y


def rope_neox(x, base, freq_factors=None):
    """ggml_rope_ext, NEOX mode, n_dims = head width: pair (i, i + n/2) rotates
    by pos * base^(-2i/n) / freq_factor[i]. x: [T, heads, n]."""
    T, _, n = x.shape
    i = torch.arange(n // 2, dtype=torch.float64)
    theta = torch.arange(T, dtype=torch.float64)[:, None] * base ** (-2 * i / n)
    if freq_factors is not None:
        theta = theta / freq_factors.double()
    c, s = theta.cos()[:, None, :], theta.sin()[:, None, :]
    x0, x1 = x[..., : n // 2], x[..., n // 2:]
    return torch.cat([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1)


def reference_forward(path, ids: list[int]) -> torch.Tensor:
    with GGUFFile(path) as g:
        W = {n: g.tensor(n).double() for n in g.tensors}
        md = g.metadata
    a = "gemma4"
    L, d, Hq = md[f"{a}.block_count"], md[f"{a}.embedding_length"], md[f"{a}.attention.head_count"]
    pattern, kvs = md[f"{a}.attention.sliding_window_pattern"], md[f"{a}.attention.head_count_kv"]
    eps, window = md[f"{a}.attention.layer_norm_rms_epsilon"], md[f"{a}.attention.sliding_window"]
    T = len(ids)
    x = W["token_embd.weight"][ids] * math.sqrt(d)               # build_inp_embd(tok_embd, sqrtf(n_embd))
    qpos, kpos = torch.arange(T)[:, None], torch.arange(T)[None, :]
    for il in range(L):
        b = f"blk.{il}."
        swa = pattern[il]
        hd = md[f"{a}.attention.key_length_swa"] if swa else md[f"{a}.attention.key_length"]
        base = md[f"{a}.rope.freq_base_swa"] if swa else md[f"{a}.rope.freq_base"]
        ff = None if swa else W["rope_freqs.weight"]
        nkv = kvs[il]
        cur = rms(x, W[b + "attn_norm.weight"], eps)
        Q = (cur @ W[b + "attn_q.weight"].T).reshape(T, Hq, hd)
        Q = rope_neox(rms(Q, W[b + "attn_q_norm.weight"], eps), base, ff)
        K = (cur @ W[b + "attn_k.weight"].T).reshape(T, nkv, hd)
        V = (cur @ W[b + "attn_v.weight"].T).reshape(T, nkv, hd) if b + "attn_v.weight" in W else K
        K = rope_neox(rms(K, W[b + "attn_k_norm.weight"], eps), base, ff)
        V = rms(V, None, eps)
        K, V = K.repeat_interleave(Hq // nkv, dim=1), V.repeat_interleave(Hq // nkv, dim=1)
        scores = torch.einsum("thd,shd->hts", Q, K) * 1.0              # f_attention_scale = 1.0
        allowed = kpos <= qpos
        if swa:
            allowed &= ~(qpos - kpos >= window)                       # is_masked_swa: p1 - p0 >= n_swa
        scores = scores.masked_fill(~allowed, float("-inf"))
        out = torch.einsum("hts,shd->thd", scores.softmax(-1), V).reshape(T, Hq * hd)
        cur = rms(out @ W[b + "attn_output.weight"].T, W[b + "post_attention_norm.weight"], eps)
        attn_out = cur + x
        cur = rms(attn_out, W[b + "ffn_norm.weight"], eps)
        gate = F.gelu(cur @ W[b + "ffn_gate.weight"].T, approximate="tanh")   # LLM_FFN_GELU, LLM_FFN_PAR
        cur = (gate * (cur @ W[b + "ffn_up.weight"].T)) @ W[b + "ffn_down.weight"].T
        if b + "ffn_gate_inp.weight" in W:            # MoE layer (gemma4.cpp, is_moe_layer)
            cur = rms(cur, W[b + "post_ffw_norm_1.weight"], eps)          # the dense MLP = shared expert
            moe_in = rms(attn_out, W[b + "pre_ffw_norm_2.weight"], eps)
            r = rms(attn_out, None, eps) / math.sqrt(d) * W[b + "ffn_gate_inp.scale"]   # router on attn_out
            gu = W[b + "ffn_gate_up_exps.weight"]
            h = gu.shape[1] // 2                                             # gate first, then up
            routed = moe_reference(moe_in, r @ W[b + "ffn_gate_inp.weight"].T, md[f"{a}.expert_used_count"],
                                   gu[:, :h], gu[:, h:], W[b + "ffn_down_exps.weight"],
                                   lambda z: F.gelu(z, approximate="tanh"), W[b + "ffn_down_exps.scale"])
            cur = cur + rms(routed, W[b + "post_ffw_norm_2.weight"], eps)
        cur = rms(cur, W[b + "post_ffw_norm.weight"], eps) + attn_out
        x = cur * W[b + "layer_output_scale.weight"]
    x = rms(x, W["output_norm.weight"], eps)
    logits = x @ W.get("output.weight", W["token_embd.weight"]).T
    cap = md[f"{a}.final_logit_softcapping"]
    return torch.tanh(logits / cap) * cap


IDS = [2, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312]  # longer than the window


@pytest.fixture(scope="module")
def gemma_f32(tmp_path_factory):
    p = tmp_path_factory.mktemp("gemma") / "tiny-f32.gguf"
    make_gemma_gguf(p)
    return p


def test_config_from_metadata(gemma_f32):
    model, info = load_gguf_model(gemma_f32, dtype=torch.float32)
    c = model.cfg
    assert info["architecture"] == "gemma4" and info["n_unused"] == 0
    assert c.layer_types == ["sliding" if s else "full" for s in PATTERN] and c.sliding_window == WINDOW
    assert (c.attn_spec(0).head_dim, c.attn_spec(0).kv_heads, c.attn_spec(0).rope_theta) == (HD_SWA, KV_SWA, BASE_SWA)
    assert (c.attn_spec(2).head_dim, c.attn_spec(2).kv_heads, c.attn_spec(2).rope_theta) == (HD_FULL, KV_FULL, BASE)
    assert c.full_v_from_k and model.blocks[2].attn.v_proj is None and model.blocks[0].attn.v_proj is not None
    assert c.tie_embeddings and c.final_softcap == CAP and c.embed_scale == pytest.approx(8.0)
    assert c.max_seq_len == 4096  # capped, not the file's 262144


def test_matches_llama_cpp_graph(gemma_f32):
    model, _ = load_gguf_model(gemma_f32, dtype=torch.float32)
    ours = model(torch.tensor([IDS])).logits[0].double()
    ref = reference_forward(gemma_f32, IDS)
    torch.testing.assert_close(ours, ref, atol=2e-4, rtol=1e-4)


@pytest.mark.parametrize("quant", [q.Q8_0, q.Q4_0])
def test_kept_quantized_equals_dequantized(tmp_path, quant):
    path = tmp_path / "tiny-q.gguf"
    make_gemma_gguf(path, quant=quant)
    dense, i1 = load_gguf_model(path, dtype=torch.float32, keep_quantized=False)
    quantized, i2 = load_gguf_model(path, dtype=torch.float32, keep_quantized=True)
    assert i1["kept_quantized"] == 0 and i2["kept_quantized"] == 1 + 7 * len(PATTERN) - PATTERN.count(False)
    assert isinstance(quantized.tok_emb, QuantEmbedding) and isinstance(quantized.lm_head, QuantLinear)
    assert quantized.lm_head.raw is quantized.tok_emb.raw          # tied: one copy of the bytes
    assert quantized.num_params() == dense.num_params()
    x = torch.tensor([IDS])
    torch.testing.assert_close(quantized(x).logits, dense(x).logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dense(x).logits[0].double(), reference_forward(path, IDS), atol=2e-4, rtol=1e-4)
    # memory: the quantized model holds bytes, not floats
    qbytes = sum(b.numel() * b.element_size() for b in quantized.buffers() if b.dtype == torch.uint8)
    assert qbytes < 0.4 * sum(p.numel() * 4 for p in dense.parameters())


@pytest.mark.parametrize("quant", [None, q.Q8_0])
def test_moe_matches_llama_cpp_graph(tmp_path, quant):
    """Gemma 4 MoE (26B-A4B layout), plain or with experts kept quantized."""
    from sleuth.forge.model.moe import MoE

    path = tmp_path / "tiny-moe.gguf"
    make_gemma_gguf(path, quant=quant, moe=True)
    model, info = load_gguf_model(path, dtype=torch.float32, keep_quantized=quant is not None)
    assert info["n_unused"] == 0, info["unused_tensors"]
    c = model.cfg
    assert (c.n_experts, c.n_experts_used, c.expert_hidden) == (N_EXP, N_USED, EXP_FF)
    assert c.expert_gate_up_merged and c.expert_down_scale and not c.llama_compatible
    moes = [m for m in model.modules() if isinstance(m, MoE)]
    assert len(moes) == len(PATTERN) and moes[0].router.weight.dtype == torch.float32
    assert (moes[0].gate_up.ggml_type is None) == (quant is None)
    ours = model(torch.tensor([IDS])).logits[0].double()
    torch.testing.assert_close(ours, reference_forward(path, IDS), atol=2e-4, rtol=1e-4)
    assert int(moes[0].counts.sum()) == len(IDS) * N_USED and (moes[0].counts > 0).sum() > N_USED
    if quant is not None:
        dense, _ = load_gguf_model(path, dtype=torch.float32, keep_quantized=False)
        assert dense.num_params() == model.num_params()


def test_moe_generation_with_kv_cache(tmp_path):
    from sleuth.forge.infer.cuda_graph import graph_unsupported
    from sleuth.forge.infer.generate import generate
    from sleuth.forge.infer.sampling import SamplingConfig

    path = tmp_path / "tiny-moe.gguf"
    make_gemma_gguf(path, quant=q.Q4_0, moe=True)
    model, _ = load_gguf_model(path, dtype=torch.float32, keep_quantized=True)
    assert "MoE" in graph_unsupported(model)
    res = generate(model, IDS[:5], max_new_tokens=10, sampling=SamplingConfig(temperature=0.0))
    ids = IDS[:5]
    for t in res.new_ids:
        assert t == int(model(torch.tensor([ids])).logits[0, -1].argmax())
        ids = ids + [t]


def test_quantized_generation_with_kv_cache(tmp_path):
    from sleuth.forge.infer.generate import generate
    from sleuth.forge.infer.sampling import SamplingConfig

    path = tmp_path / "tiny-q4.gguf"
    make_gemma_gguf(path, quant=q.Q4_0)
    model, _ = load_gguf_model(path, dtype=torch.float32, keep_quantized=True)
    res = generate(model, IDS[:5], max_new_tokens=12, sampling=SamplingConfig(temperature=0.0))
    # Greedy decoding with the cache must equal argmax of full forwards (window 3 < length 17).
    ids = IDS[:5]
    for t in res.new_ids:
        assert t == int(model(torch.tensor([ids])).logits[0, -1].argmax())
        ids = ids + [t]


def test_tokenizer_from_gguf_and_load_any(gemma_f32):
    tok = tokenizer_from_gguf(gemma_f32)
    assert isinstance(tok, GemmaTokenizer) and tok.eot_id == 6
    assert tok.encode("the cat") == tiny_tokenizer().encode("the cat")
    model, tok2, info = load_any(gemma_f32, dtype=torch.float32)
    assert info["architecture"] == "gemma4" and tok2.start_id == 2


def test_analyzer_views_run_on_gemma(gemma_f32):
    from sleuth.analyzer import ablation, attention, logit_lens, neurons
    from sleuth.analyzer.suites import TestCase

    model, _ = load_gguf_model(gemma_f32, dtype=torch.float32)
    tok = tokenizer_from_gguf(gemma_f32)
    cases = [TestCase("a", "story", "the cat is an ant", None), TestCase("b", "story", "once the cat", " ate")]
    a = attention.run(model, tok, cases, "cpu", induction_len=4, induction_batch=2)
    assert a["arrays"]["induction_score"].shape == (len(PATTERN), H)
    ll = logit_lens.run(model, tok, cases, "cpu")["summary"]
    assert len(ll["rows"]) == len(PATTERN) + 1
    ab = ablation.run(model, tok, cases, "cpu")["summary"]
    assert len(ab["components"]) == len(PATTERN) * (H + 2)
    assert neurons.run(model, tok, cases, "cpu")["summary"]


@pytest.mark.parametrize("change,match", [
    (lambda md: md.update({"gemma4.attention.shared_kv_layers": (4, 2)}), "KV sharing"),
    (lambda md: md.update({"gemma4.embedding_length_per_layer_input": (4, 8)}), "per-layer embeddings"),
    (lambda md: md.update({"gemma4.rope.dimension_count": (4, 16)}), "partial RoPE"),
])
def test_unsupported_variants_raise(tmp_path, gemma_f32, change, match):
    with GGUFFile(gemma_f32) as g:
        kinds = {bool: 7, int: 5, float: 6, str: 8}
        md = {k: ((9, (kinds[type(v[0])], v)) if isinstance(v, list) else (kinds[type(v)], v))
              for k, v in g.metadata.items()}
        tensors = {n: (t.ggml_type, t.shape, bytes(g.raw(n).numpy())) for n, t in g.tensors.items()}
    change(md)
    write_gguf(tmp_path / "bad.gguf", md, tensors)
    with pytest.raises(NotImplementedError, match=match):
        load_gguf_model(tmp_path / "bad.gguf", dtype=torch.float32)


def test_inspect_lists_every_shape(gemma_f32):
    from sleuth.models import describe_file

    text = describe_file(gemma_f32)
    assert "shape (64, 64): layers 0-1, 3-4" in text and "shape (128, 64): layers 2, 5" in text
    assert describe_file(gemma_f32, show_key="general.name") == "tiny gemma4"
