"""Load GGUF models (llama.cpp format) into our Transformer.

Supported now: architectures "llama", "qwen2", "qwen3" (the same family as
formats/hf.py). Others (e.g. Gemma) need their own adapter. Run
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

RoPE layout: for the "llama" architecture, llama.cpp's converter *reorders*
the rows of the q and k weights from Hugging Face's "rotate half" layout
(pairs (i, i + d/2)) into the original interleaved one (pairs (2i, 2i+1)),
because llama.cpp's llama RoPE rotates adjacent pairs. We use rotate-half,
so we undo that permutation. Qwen2/Qwen3 GGUFs use NEOX-style RoPE (rotate
half) and are not permuted.

Weights are dequantized to the requested dtype when loading. A 4-bit 12B
model becomes ~24 GB in bf16, so big quantized models will need to stay
quantized in memory (planned).

The tokenizer comes from the GGUF metadata too (`tokenizer.ggml.*`). The
byte-level BPE type ("gpt2", used by Llama 3 and Qwen) maps onto our
HFTokenizer. llama.cpp doesn't store the pre-tokenizer regex; it stores a
*name* ("qwen2", "llama-bpe", ...) that we look up in PRE_PATTERNS.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from ..forge.model.config import ModelConfig
from ..forge.model.transformer import Transformer
from .gguf import GGUFFile
from .hf_tokenizer import GPT2_PATTERN, GPT4_PATTERN, HFTokenizer

SUPPORTED = {"llama", "qwen2", "qwen3"}
NEOX_ROPE = {"qwen2", "qwen3"}  # rotate-half already; no permutation to undo

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
    token_embd.weight. llama.cpp's converter often writes the copy even for
    tied models (ggml-org's Qwen3-0.6B Q8_0 does), and keeping both costs
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


def load_gguf_model(path: str | Path, device: str | torch.device = "cpu", dtype: torch.dtype = torch.bfloat16,
                    max_seq_len: int = 4096) -> tuple[Transformer, dict]:
    with GGUFFile(path) as g:
        cfg = model_config(g, max_seq_len)
        if g.arch_value("rope.scaling.type") not in (None, "none", "linear") or g.arch_value("rope.scaling.factor", 1.0) not in (1.0, 0.0):
            raise NotImplementedError("RoPE scaling in this GGUF isn't implemented yet")
        model = Transformer.empty(cfg, device=device, dtype=dtype)
        params = dict(model.named_parameters())
        loaded, unused = set(), []
        permute = g.architecture not in NEOX_ROPE
        for name, info in g.tensors.items():
            ours = our_name(name)
            if ours == "lm_head.weight" and cfg.tie_embeddings:
                continue  # a copy of token_embd (see output_is_tied)
            if ours is None or ours not in params:
                unused.append(name)
                continue
            t = g.tensor(name, dtype=torch.float32, device=device)
            if permute and re.search(r"attn\.(q|k)_proj\.", ours):
                heads = cfg.n_heads if ".q_proj." in ours else cfg.kv_heads
                t = unpermute_rope_rows(t, heads)
            p = params[ours]
            if tuple(t.shape) != tuple(p.shape):
                raise ValueError(f"{name}: file shape {tuple(t.shape)} != model shape {tuple(p.shape)}")
            with torch.no_grad():
                p.copy_(t.to(dtype))
            loaded.add(ours)
        missing = sorted(set(params) - loaded)
        if missing:
            raise KeyError(f"GGUF is missing {len(missing)} tensors, e.g. {missing[:5]}")
        types: dict[str, int] = {}
        for info in g.tensors.values():
            types[info.type_name] = types.get(info.type_name, 0) + 1
        info = {"format": "gguf", "architecture": g.architecture, "name": g.metadata.get("general.name"),
                "tied_output": cfg.tie_embeddings and "output.weight" in g.tensors,
                "tensor_types": types, "unused_tensors": unused[:20], "n_unused": len(unused)}
    model.eval()
    return model, info


def tokenizer_from_gguf(path: str | Path) -> HFTokenizer:
    """Build our HFTokenizer from tokenizer.ggml.* metadata (byte-level BPE only)."""
    with GGUFFile(path) as g:
        md = g.metadata
    kind = md.get("tokenizer.ggml.model")
    if kind != "gpt2":
        raise NotImplementedError(f"GGUF tokenizer type {kind!r} isn't supported yet (only byte-level BPE 'gpt2')")
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
