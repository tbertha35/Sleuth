"""Load GGUF models (llama.cpp format) into our Transformer.

Supported now: architectures "llama", "qwen2", "qwen3" (the same family as
formats/hf.py) and "gemma4" (dense, text only; see gemma4_config). Run
`python -m sleuth inspect model.gguf` to see what a file contains.

Tensor names (llama.cpp's convention) -> ours:
    token_embd.weight          -> tok_emb.weight
    blk.{i}.attn_norm.weight   -> blocks.{i}.norm1.weight
    blk.{i}.attn_{q,k,v}.*     -> blocks.{i}.attn.{q,k,v}_proj.*
    blk.{i}.attn_output.weight -> blocks.{i}.attn.o_proj.weight
    blk.{i}.attn_{q,k}_norm.weight -> blocks.{i}.attn.{q,k}_norm.weight
    blk.{i}.ffn_norm.weight    -> blocks.{i}.norm2.weight
    blk.{i}.ffn_{gate,up,down}.weight -> blocks.{i}.mlp.{gate,up,down}.weight
    output_norm.weight         -> norm_f.weight
    output.weight              -> lm_head.weight (absent when tied)
  Gemma 4 adds:
    blk.{i}.post_attention_norm.weight -> blocks.{i}.post_attn_norm.weight
    blk.{i}.post_ffw_norm.weight       -> blocks.{i}.post_mlp_norm.weight
    blk.{i}.layer_output_scale.weight  -> blocks.{i}.scale
    rope_freqs.weight                  -> RoPE frequency factors of the full layers

RoPE layout: for the "llama" architecture, llama.cpp's converter *reorders*
the rows of the q and k weights from Hugging Face's "rotate half" layout
(pairs (i, i + d/2)) into the original interleaved one (pairs (2i, 2i+1)),
because llama.cpp's llama RoPE rotates adjacent pairs. We use rotate-half,
so we undo that permutation. Qwen2/Qwen3 GGUFs use NEOX-style RoPE (rotate
half) and are not permuted.

Weights are either dequantized to the requested dtype when loading, or kept
as the file's quantized blocks (forge/model/quant.py) and dequantized one
layer at a time while running. A 4-bit 12B model is ~6 GB quantized but
~24 GB in bf16, so `keep_quantized="auto"` keeps big models quantized.

The tokenizer comes from the GGUF metadata too (`tokenizer.ggml.*`). The
byte-level BPE type ("gpt2", used by Llama 3 and Qwen) maps onto our
HFTokenizer. llama.cpp doesn't store the pre-tokenizer regex; it stores a
*name* ("qwen2", "llama-bpe", ...) that we look up in PRE_PATTERNS.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import torch

from ..forge.model.config import ModelConfig
from ..forge.model.transformer import Transformer
from . import gguf_quants as q
from .gguf import GGUFFile
from .hf_tokenizer import GPT2_PATTERN, GPT4_PATTERN, HFTokenizer

SUPPORTED = {"llama", "qwen2", "qwen3", "gemma4"}
NEOX_ROPE = {"qwen2", "qwen3", "gemma4"}  # rotate-half already; no permutation to undo

QWEN2_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
                 r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
PRE_PATTERNS = {"qwen2": QWEN2_PATTERN, "llama-bpe": GPT4_PATTERN, "llama3": GPT4_PATTERN,
                "gpt-2": GPT2_PATTERN, "default": GPT2_PATTERN}

_LAYER = [
    (r"attn_norm\.weight", "norm1.weight"),
    (r"ffn_norm\.weight", "norm2.weight"),
    (r"attn_(q|k|v)\.(weight|bias)", r"attn.\1_proj.\2"),
    (r"attn_output\.weight", "attn.o_proj.weight"),
    (r"attn_(q|k)_norm\.weight", r"attn.\1_norm.weight"),
    (r"ffn_(gate|up|down)\.weight", r"mlp.\1.weight"),
    (r"post_attention_norm\.weight", "post_attn_norm.weight"),
    (r"post_ffw_norm\.weight", "post_mlp_norm.weight"),
    (r"layer_output_scale\.weight", "scale"),
]


def our_name(gguf_name: str) -> str | None:
    fixed = {"token_embd.weight": "tok_emb.weight", "output_norm.weight": "norm_f.weight", "output.weight": "lm_head.weight"}
    if gguf_name in fixed:
        return fixed[gguf_name]
    m = re.match(r"blk\.(\d+)\.(.+)$", gguf_name)
    if m:
        i, rest = m.groups()
        for pat, repl in _LAYER:
            if re.fullmatch(pat, rest):
                return f"blocks.{i}." + re.sub(pat, repl, rest)
    return None


def unpermute_rope_rows(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Undo llama.cpp's q/k row reordering (interleaved pairs -> rotate-half).
    Works for weights [n_heads*head_dim, d] and biases [n_heads*head_dim]."""
    rows = w.shape[0]
    hd = rows // n_heads
    rest = w.shape[1:]
    return w.reshape(n_heads, hd // 2, 2, *rest).transpose(1, 2).reshape(rows, *rest)


def permute_rope_rows(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """The forward reordering llama.cpp's converter applies (for tests and export)."""
    rows = w.shape[0]
    hd = rows // n_heads
    rest = w.shape[1:]
    return w.reshape(n_heads, 2, hd // 2, *rest).transpose(1, 2).reshape(rows, *rest)


def output_is_tied(g: GGUFFile) -> bool:
    """Does the output layer reuse the token embedding?

    Either output.weight is absent, or it's a byte-for-byte copy of
    token_embd.weight. Some GGUFs of tied models store the copy anyway
    (ggml-org's Qwen3-0.6B Q8_0 has an output.weight), and keeping both costs
    vocab x d extra weights: 155M for Qwen3-0.6B, 311 MB in bf16. Copies
    quantized differently (Q4_K_M stores output.weight as Q6_K) are genuinely
    different numbers, and stay separate.
    """
    return "output.weight" not in g.tensors or g.same_bytes("output.weight", "token_embd.weight")


def model_config(g: GGUFFile, max_seq_len: int = 4096) -> ModelConfig:
    arch = g.architecture
    if arch not in SUPPORTED:
        raise NotImplementedError(f"GGUF architecture {arch!r} isn't supported yet (supported: {sorted(SUPPORTED)}). "
                                  f"Run `python -m sleuth inspect <file>` and share the output to add it.")
    if arch == "gemma4":
        return gemma4_config(g, max_seq_len)
    d = int(g.arch_value("embedding_length"))
    n_heads = int(g.arch_value("attention.head_count"))
    head_dim = int(g.arch_value("attention.key_length", d // n_heads))
    vocab = g.tensors["token_embd.weight"].shape[0]
    return ModelConfig(
        vocab_size=vocab,
        d_model=d,
        n_layers=int(g.arch_value("block_count")),
        n_heads=n_heads,
        n_kv_heads=int(g.arch_value("attention.head_count_kv", n_heads)),
        head_size=head_dim if head_dim * n_heads != d else 0,
        ffn_type="swiglu",
        ffn_hidden=int(g.arch_value("feed_forward_length")),
        norm_type="rmsnorm",
        norm_eps=float(g.arch_value("attention.layer_norm_rms_epsilon", 1e-6)),
        pos_type="rope",
        rope_theta=float(g.arch_value("rope.freq_base", 10000.0)),
        max_seq_len=min(max_seq_len, int(g.arch_value("context_length", max_seq_len))),
        qk_norm="blk.0.attn_q_norm.weight" in g.tensors,
        tie_embeddings=output_is_tied(g),
        qkv_bias="blk.0.attn_q.bias" in g.tensors,
        pad_vocab_to=1,
    )


def _per_layer(g: GGUFFile, key: str, n_layers: int, default=None) -> list:
    """A hyperparameter that may be one value or one value per layer."""
    v = g.arch_value(key, default)
    if v is None:
        raise KeyError(f"GGUF metadata has no {g.architecture}.{key}")
    return list(v) if isinstance(v, list) else [v] * n_layers


def _one_per_kind(values: list, kinds: list[str], what: str) -> dict[str, object]:
    """{"full": x, "sliding": y}, checking every layer of a kind agrees."""
    out: dict[str, object] = {}
    for v, k in zip(values, kinds):
        if out.setdefault(k, v) != v:
            raise NotImplementedError(f"{what} differs between {k} layers ({out[k]} vs {v}); not supported yet")
    return out


def gemma4_config(g: GGUFFile, max_seq_len: int = 4096) -> ModelConfig:
    """Gemma 4 (dense) from GGUF metadata, following llama.cpp's gemma4 model
    (src/models/gemma4.cpp), which is the reference for how these files run:

      * embeddings × sqrt(d_model); tied output layer; final logits soft-capped
      * every block: x + post_norm(attn(norm(x))), then x + post_norm(mlp(norm(x))),
        then the whole stream × a learned per-layer scalar (layer_output_scale)
      * attention: q_norm and k_norm (RMSNorm with gain), values RMS-normalized
        without a gain, attention scale 1.0 (not 1/sqrt(head_dim)), NEOX RoPE
      * sliding layers (sliding_window_pattern = True): window of
        `attention.sliding_window` tokens, their own head width, KV heads and RoPE base
      * full layers: no attn_v in the file means values = keys; RoPE divided
        by the factors in rope_freqs.weight
      * MLP: GELU (tanh) gated, like SwiGLU with another activation

    Parts of the format we don't implement raise an error instead of loading a
    subtly wrong model: per-layer embeddings (E2B/E4B), KV sharing across
    layers, MoE layers (26B-A4B), partial RoPE.
    """
    L = int(g.arch_value("block_count"))
    d = int(g.arch_value("embedding_length"))
    if int(g.arch_value("embedding_length_per_layer_input", 0)):
        raise NotImplementedError("Gemma 4 per-layer embeddings (E2B/E4B models) aren't supported yet")
    if int(g.arch_value("attention.shared_kv_layers", 0)):
        raise NotImplementedError("Gemma 4 KV sharing between layers isn't supported yet")
    if any(n.endswith("ffn_gate_inp.weight") for n in g.tensors):
        raise NotImplementedError("Gemma 4 MoE layers (26B-A4B) aren't supported yet")
    pattern = _per_layer(g, "attention.sliding_window_pattern", L)
    kinds = ["sliding" if bool(x) else "full" for x in pattern]
    heads = _one_per_kind([int(x) for x in _per_layer(g, "attention.head_count", L)], ["all"] * L, "head_count")["all"]
    kv = _one_per_kind([int(x) for x in _per_layer(g, "attention.head_count_kv", L)], kinds, "head_count_kv")
    ffn = _one_per_kind([int(x) for x in _per_layer(g, "feed_forward_length", L)], ["all"] * L, "feed_forward_length")["all"]
    hd_full, hd_swa = int(g.arch_value("attention.key_length")), int(g.arch_value("attention.key_length_swa"))
    if (hd_full, hd_swa) != (int(g.arch_value("attention.value_length", hd_full)),
                             int(g.arch_value("attention.value_length_swa", hd_swa))):
        raise NotImplementedError("Gemma 4 with value width != key width")
    if (int(g.arch_value("rope.dimension_count", hd_full)), int(g.arch_value("rope.dimension_count_swa", hd_swa))) \
            != (hd_full, hd_swa):
        raise NotImplementedError("partial RoPE (rope.dimension_count < head width) isn't supported yet")
    v_kinds = _one_per_kind([f"blk.{i}.attn_v.weight" in g.tensors for i in range(L)], kinds, "having attn_v")
    if not v_kinds.get("sliding", True):
        raise NotImplementedError("Gemma 4 sliding layers without attn_v")
    return ModelConfig(
        vocab_size=g.tensors["token_embd.weight"].shape[0],
        d_model=d,
        n_layers=L,
        n_heads=heads,
        head_size=hd_full,
        n_kv_heads=int(kv.get("full", kv.get("sliding"))),
        ffn_type="geglu",
        ffn_hidden=int(ffn),
        norm_type="rmsnorm",
        norm_eps=float(g.arch_value("attention.layer_norm_rms_epsilon", 1e-6)),
        pos_type="rope",
        rope_theta=float(g.arch_value("rope.freq_base", 10000.0)),
        max_seq_len=min(max_seq_len, int(g.arch_value("context_length", max_seq_len))),
        qk_norm=True,
        tie_embeddings=output_is_tied(g),
        pad_vocab_to=1,
        embed_scale=math.sqrt(d),
        post_norms=True,
        v_norm=True,
        attn_scale=1.0,
        final_softcap=float(g.arch_value("final_logit_softcapping", 0.0) or 0.0),
        layer_scale="blk.0.layer_output_scale.weight" in g.tensors,
        layer_types=kinds,
        sliding_window=int(g.arch_value("attention.sliding_window", 0)) if "sliding" in kinds else 0,
        swa_head_size=hd_swa,
        swa_kv_heads=int(kv.get("sliding", 0)),
        swa_rope_theta=float(g.arch_value("rope.freq_base_swa", 0.0) or 0.0),
        full_v_from_k=not v_kinds.get("full", True),
    )


PLAIN_TYPES = {q.F32, q.F16, q.BF16, q.F64}


def dequantized_bytes(g: GGUFFile, dtype: torch.dtype) -> int:
    """How big the model gets if every weight is dequantized to `dtype`."""
    return sum(t.n_elements for t in g.tensors.values()) * torch.empty(0, dtype=dtype).element_size()


def should_keep_quantized(g: GGUFFile, device: torch.device, dtype: torch.dtype) -> bool:
    """The "auto" rule: keep weights quantized when dequantizing them would
    take more than half the GPU's memory (or more than 8 GiB on the CPU).
    Dequantized layers run faster, so small models are dequantized."""
    need = dequantized_bytes(g, dtype)
    if not any(t.ggml_type not in PLAIN_TYPES for t in g.tensors.values()):
        return False
    if device.type == "cuda":
        return need > 0.5 * torch.cuda.get_device_properties(device).total_memory
    return need > 8 * 2**30


def load_gguf_model(path: str | Path, device: str | torch.device = "cpu", dtype: torch.dtype = torch.bfloat16,
                    max_seq_len: int = 4096, keep_quantized: bool | str = "auto") -> tuple[Transformer, dict]:
    """keep_quantized: True = quantized weights stay quantized in memory
    (QuantLinear/QuantEmbedding), False = dequantize everything now,
    "auto" = see should_keep_quantized."""
    device = torch.device(device)
    with GGUFFile(path) as g:
        cfg = model_config(g, max_seq_len)
        if g.arch_value("rope.scaling.type") not in (None, "none", "linear") or g.arch_value("rope.scaling.factor", 1.0) not in (1.0, 0.0):
            raise NotImplementedError("RoPE scaling in this GGUF isn't implemented yet")
        if keep_quantized == "auto":
            keep_quantized = should_keep_quantized(g, device, dtype)
        # Which of our Linear/Embedding layers stay as quantized blocks.
        quantized: dict[str, int] = {}
        if keep_quantized:
            for name, info in g.tensors.items():
                ours = our_name(name)
                if (ours and ours.endswith(".weight") and len(info.shape) == 2 and info.ggml_type not in PLAIN_TYPES
                        and not (ours == "lm_head.weight" and cfg.tie_embeddings)):
                    quantized[ours.removesuffix(".weight")] = info.ggml_type
        model = Transformer.empty(cfg, device=device, dtype=dtype, quantized=quantized)
        params = dict(model.named_parameters())
        loaded, unused = set(), []
        permute = g.architecture not in NEOX_ROPE
        for name, info in g.tensors.items():
            ours = our_name(name)
            if name == "rope_freqs.weight" and cfg.layer_types:
                model.set_rope_freq_factors("full", g.tensor(name))
                continue
            if ours == "lm_head.weight" and cfg.tie_embeddings:
                continue  # a copy of token_embd (see output_is_tied)
            module_path = ours.removesuffix(".weight") if ours else None
            if module_path in quantized:
                _load_quantized(model, module_path, g, name, cfg, permute, device)
                loaded.add(ours)
                continue
            if ours is None or ours not in params:
                unused.append(name)
                continue
            t = g.tensor(name, dtype=torch.float32, device=device)
            if permute and re.search(r"attn\.(q|k)_proj\.", ours):
                heads = cfg.n_heads if ".q_proj." in ours else cfg.kv_heads
                t = unpermute_rope_rows(t, heads)
            p = params[ours]
            if t.numel() == p.numel() == 1:
                t = t.reshape(p.shape)  # layer_output_scale: shape (1,) either way
            if tuple(t.shape) != tuple(p.shape):
                raise ValueError(f"{name}: file shape {tuple(t.shape)} != model shape {tuple(p.shape)}")
            with torch.no_grad():
                p.copy_(t.to(dtype))
            loaded.add(ours)
        expected = set(params) | {f"{m}.weight" for m in quantized}
        missing = sorted(expected - loaded)
        if missing:
            raise KeyError(f"GGUF is missing {len(missing)} tensors, e.g. {missing[:5]}")
        types: dict[str, int] = {}
        for info in g.tensors.values():
            types[info.type_name] = types.get(info.type_name, 0) + 1
        info = {"format": "gguf", "architecture": g.architecture, "name": g.metadata.get("general.name"),
                "tied_output": cfg.tie_embeddings and "output.weight" in g.tensors,
                "kept_quantized": len(quantized),
                "tensor_types": types, "unused_tensors": unused[:20], "n_unused": len(unused)}
    model.eval()
    return model, info


def _load_quantized(model, module_path, g: GGUFFile, name: str, cfg: ModelConfig, permute: bool, device) -> None:
    """Copy a tensor's quantized blocks into a QuantLinear/QuantEmbedding as they are."""
    module = model.get_submodule(module_path)
    info = g.tensors[name]
    if module.ggml_type != info.ggml_type or module.raw.numel() != info.nbytes:
        raise ValueError(f"{name}: {info.type_name} {info.shape} doesn't fit {module_path} ({module})")
    raw = g.raw(name)
    if permute and re.search(r"attn\.(q|k)_proj$", module_path):
        # Whole rows are whole blocks, so llama.cpp's row reordering can be
        # undone on the quantized bytes directly.
        heads = cfg.n_heads if module_path.endswith("q_proj") else cfg.kv_heads
        raw = unpermute_rope_rows(raw.view(info.shape[0], -1), heads).reshape(-1)
    with torch.no_grad():
        module.raw.copy_(raw.to(device))


def tokenizer_from_gguf(path: str | Path):
    """Build a tokenizer from tokenizer.ggml.* metadata: byte-level BPE ("gpt2",
    as HFTokenizer) or Gemma 4's SentencePiece-style BPE ("gemma4")."""
    with GGUFFile(path) as g:
        md = g.metadata
    kind = md.get("tokenizer.ggml.model")
    if kind == "gemma4":
        from .gemma_tokenizer import GemmaTokenizer

        tok = GemmaTokenizer(md["tokenizer.ggml.tokens"], md.get("tokenizer.ggml.token_type"),
                             md.get("tokenizer.ggml.merges", []), md.get("tokenizer.ggml.bos_token_id"),
                             md.get("tokenizer.ggml.eos_token_id"),
                             add_bos=True)  # llama.cpp forces add_bos for Gemma 4 too
        tok.eot_id = md.get("tokenizer.ggml.eot_token_id")
        return tok
    if kind != "gpt2":
        raise NotImplementedError(f"GGUF tokenizer type {kind!r} isn't supported yet ('gpt2' and 'gemma4' are)")
    pre = md.get("tokenizer.ggml.pre", "default")
    if pre not in PRE_PATTERNS:
        raise NotImplementedError(f"pre-tokenizer {pre!r} unknown; known: {sorted(PRE_PATTERNS)}")
    tokens = md["tokenizer.ggml.tokens"]
    types = md.get("tokenizer.ggml.token_type") or [1] * len(tokens)
    # token_type: 1 normal, 2 unknown, 3 control (special), 4 user-defined, 5 unused, 6 byte
    added = [{"id": i, "content": t, "special": types[i] == 3} for i, t in enumerate(tokens) if types[i] in (3, 4)]
    vocab = {t: i for i, t in enumerate(tokens) if types[i] not in (3, 4)}
    data = {
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": PRE_PATTERNS[pre]}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "use_regex": False}]},
        "model": {"type": "BPE", "vocab": vocab, "merges": md.get("tokenizer.ggml.merges", [])},
    }
    tok = HFTokenizer(data)
    bos, eos = md.get("tokenizer.ggml.bos_token_id"), md.get("tokenizer.ggml.eos_token_id")
    if md.get("tokenizer.ggml.add_bos_token", bos is not None):
        tok.bos_id = bos
    else:
        tok.bos_id = None
    tok.eos_id = eos if eos is not None else tok.eos_id
    return tok
