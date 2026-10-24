"""Load Hugging Face-format checkpoints (a directory with config.json +
*.safetensors) into our own Transformer, with no `transformers` library.

Supported architectures ("model_type" in config.json): llama, mistral, qwen2,
qwen3. All of them are pre-norm decoders with RoPE + RMSNorm + SwiGLU + GQA,
which is exactly what forge/model implements, so loading them is mostly
*renaming tensors*:

    model.embed_tokens.weight                     -> tok_emb.weight
    model.layers.{i}.input_layernorm.weight       -> blocks.{i}.norm1.weight
    model.layers.{i}.self_attn.{q,k,v,o}_proj.*   -> blocks.{i}.attn.{q,k,v,o}_proj.*
    model.layers.{i}.self_attn.{q,k}_norm.weight  -> blocks.{i}.attn.{q,k}_norm.weight  (Qwen3)
    model.layers.{i}.post_attention_layernorm.*   -> blocks.{i}.norm2.weight
    model.layers.{i}.mlp.{gate,up,down}_proj.*    -> blocks.{i}.mlp.{gate,up,down}.weight
    model.norm.weight                             -> norm_f.weight
    lm_head.weight                                -> lm_head.weight (absent when tied)

Because the external model becomes *our* Transformer, every analyzer view
(taps, ablation, logit lens, ...) works on it unchanged.

Notes:
  * RoPE: Hugging Face checkpoints already use the "rotate half" layout we use.
  * Features we don't implement (e.g. Llama 3.1's rope_scaling, sliding-window
    attention beyond max_seq_len) raise a clear error instead of loading a
    subtly wrong model.
  * Large checkpoints are split into shards listed in
    model.safetensors.index.json; we read them one tensor at a time.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from ..forge.model.config import ModelConfig
from ..forge.model.transformer import Transformer
from .safetensors import SafetensorsFile

SUPPORTED = {"llama", "mistral", "qwen2", "qwen3"}


def is_hf_dir(path: str | Path) -> bool:
    p = Path(path)
    return p.is_dir() and (p / "config.json").exists() and any(p.glob("*.safetensors"))


class Checkpoint:
    """All tensors of a (possibly sharded) safetensors checkpoint, by name."""

    def __init__(self, directory: Path):
        index = directory / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
            files = sorted(set(weight_map.values()))
        else:
            files = [f.name for f in sorted(directory.glob("*.safetensors"))]
            weight_map = None
        self.files = {f: SafetensorsFile(directory / f) for f in files}
        self.where: dict[str, SafetensorsFile] = {}
        for f, st in self.files.items():
            for k in st.keys():
                if weight_map is None or weight_map.get(k) == f:
                    self.where[k] = st

    def keys(self) -> list[str]:
        return list(self.where)

    def __contains__(self, k: str) -> bool:
        return k in self.where

    def get(self, k: str) -> torch.Tensor:
        return self.where[k].get(k)

    def close(self):
        for st in self.files.values():
            st.close()


def model_config(hf: dict, ckpt: Checkpoint, max_seq_len: int = 4096) -> ModelConfig:
    mt = hf.get("model_type")
    if mt not in SUPPORTED:
        raise NotImplementedError(f"model_type {mt!r} isn't supported yet (supported: {sorted(SUPPORTED)})")
    if hf.get("hidden_act", "silu") != "silu":
        raise NotImplementedError(f"activation {hf.get('hidden_act')!r} (only SwiGLU with SiLU is supported)")
    scaling = hf.get("rope_scaling")
    if scaling and scaling.get("rope_type", scaling.get("type")) not in (None, "default"):
        raise NotImplementedError(f"rope_scaling {scaling} isn't implemented yet")
    n_heads = hf["num_attention_heads"]
    d = hf["hidden_size"]
    head_dim = hf.get("head_dim") or d // n_heads
    window = hf.get("sliding_window") if hf.get("use_sliding_window", mt == "mistral") else None
    seq = min(max_seq_len, hf.get("max_position_embeddings", max_seq_len))
    if window and seq > window:
        seq = window  # within the window, sliding-window attention equals full attention
    return ModelConfig(
        vocab_size=hf["vocab_size"],
        d_model=d,
        n_layers=hf["num_hidden_layers"],
        n_heads=n_heads,
        n_kv_heads=hf.get("num_key_value_heads", n_heads),
        head_size=head_dim if head_dim * n_heads != d else 0,
        ffn_type="swiglu",
        ffn_hidden=hf["intermediate_size"],
        norm_type="rmsnorm",
        norm_eps=hf.get("rms_norm_eps", 1e-6),
        pos_type="rope",
        rope_theta=float(hf.get("rope_theta", 10000.0)),
        max_seq_len=seq,
        qk_norm="model.layers.0.self_attn.q_norm.weight" in ckpt,
        tie_embeddings=bool(hf.get("tie_word_embeddings", False)) or "lm_head.weight" not in ckpt,
        bias=False,
        qkv_bias="model.layers.0.self_attn.q_proj.bias" in ckpt,
        pad_vocab_to=1,  # keep the vocab exactly as the checkpoint has it
    )


_LAYER_MAP = [
    (r"input_layernorm\.weight", "norm1.weight"),
    (r"post_attention_layernorm\.weight", "norm2.weight"),
    (r"self_attn\.(q|k|v|o)_proj\.(weight|bias)", r"attn.\1_proj.\2"),
    (r"self_attn\.(q|k)_norm\.weight", r"attn.\1_norm.weight"),
    (r"mlp\.(gate|up|down)_proj\.weight", r"mlp.\1.weight"),
]
_IGNORED = re.compile(r"rotary_emb\.inv_freq$")


def our_name(hf_name: str) -> str | None:
    """Map a Hugging Face tensor name to ours (None = not part of the model we build)."""
    if hf_name == "model.embed_tokens.weight":
        return "tok_emb.weight"
    if hf_name == "model.norm.weight":
        return "norm_f.weight"
    if hf_name == "lm_head.weight":
        return "lm_head.weight"
    m = re.match(r"model\.layers\.(\d+)\.(.+)$", hf_name)
    if m:
        i, rest = m.groups()
        for pat, repl in _LAYER_MAP:
            if re.fullmatch(pat, rest):
                return f"blocks.{i}." + re.sub(pat, repl, rest)
    return None


def load_hf_model(path: str | Path, device: str | torch.device = "cpu", dtype: torch.dtype = torch.bfloat16,
                  max_seq_len: int = 4096) -> tuple[Transformer, dict]:
    d = Path(path)
    hf = json.loads((d / "config.json").read_text())
    ckpt = Checkpoint(d)
    try:
        cfg = model_config(hf, ckpt, max_seq_len)
        model = Transformer.empty(cfg, device=device, dtype=dtype)
        params = dict(model.named_parameters())  # tied weights appear once (as tok_emb.weight)
        loaded: set[str] = set()
        unused: list[str] = []
        for k in ckpt.keys():
            name = our_name(k)
            if name is None or (name == "lm_head.weight" and cfg.tie_embeddings):
                if not _IGNORED.search(k):
                    unused.append(k)
                continue
            if name not in params:
                raise KeyError(f"{k} -> {name}: no such parameter in the model we built")
            t = ckpt.get(k)
            p = params[name]
            if tuple(t.shape) != tuple(p.shape):
                raise ValueError(f"{k}: checkpoint shape {tuple(t.shape)} != model shape {tuple(p.shape)}")
            with torch.no_grad():
                p.copy_(t.to(dtype))
            loaded.add(name)
        missing = sorted(set(params) - loaded)
        if missing:
            raise KeyError(f"checkpoint is missing {len(missing)} tensors, e.g. {missing[:5]}")
    finally:
        ckpt.close()
    model.eval()
    info = {"format": "hf", "model_type": hf.get("model_type"), "architectures": hf.get("architectures"),
            "unused_tensors": unused[:20], "n_unused": len(unused)}
    return model, info


# --------------------------------------------------------------------------- #
# Export: our model -> Hugging Face format
# --------------------------------------------------------------------------- #

def save_hf_model(model: Transformer, out_dir: str | Path, tokenizer=None, max_shard_bytes: int = 2 * 2**30) -> Path:
    """Write our model as a Llama-style (or Qwen3-style, if it uses QK-norm)
    Hugging Face checkpoint: config.json + safetensors shards (+ tokenizer.json).

    That's the format llama.cpp's converter and vLLM read, so it's also the route
    to serving our models with those tools. Only Llama-compatible configs export
    (SwiGLU + RMSNorm + RoPE, no biases except q/k/v).
    """
    from .safetensors import save_file

    cfg = model.cfg
    if cfg.ffn_type != "swiglu" or cfg.norm_type != "rmsnorm" or cfg.pos_type != "rope" or cfg.bias:
        raise ValueError("only Llama-compatible models (SwiGLU, RMSNorm, RoPE, no linear biases) can be exported")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_type = "qwen3" if cfg.qk_norm else ("qwen2" if cfg.qkv_bias else "llama")
    hf_cfg = {
        "architectures": [{"llama": "LlamaForCausalLM", "qwen2": "Qwen2ForCausalLM", "qwen3": "Qwen3ForCausalLM"}[model_type]],
        "model_type": model_type,
        "vocab_size": cfg.padded_vocab,
        "hidden_size": cfg.d_model,
        "intermediate_size": cfg.hidden,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.kv_heads,
        "head_dim": cfg.head_dim,
        "hidden_act": "silu",
        "rms_norm_eps": cfg.norm_eps,
        "rope_theta": cfg.rope_theta,
        "max_position_embeddings": cfg.max_seq_len,
        "tie_word_embeddings": cfg.tie_embeddings,
        "attention_bias": cfg.qkv_bias,
        "torch_dtype": "bfloat16",
    }
    reverse = {}
    for name, _ in model.named_parameters():
        reverse[name] = _hf_name(name)
    shards: list[dict[str, torch.Tensor]] = [{}]
    size = 0
    for name, p in model.named_parameters():
        t = p.detach().to("cpu", torch.bfloat16).contiguous()
        if size and size + t.numel() * 2 > max_shard_bytes:
            shards.append({})
            size = 0
        shards[-1][reverse[name]] = t
        size += t.numel() * 2
    weight_map = {}
    for i, sh in enumerate(shards):
        fname = "model.safetensors" if len(shards) == 1 else f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(sh, out / fname, metadata={"format": "pt"})
        weight_map.update({k: fname for k in sh})
    if len(shards) > 1:
        (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1))
    (out / "config.json").write_text(json.dumps(hf_cfg, indent=2) + "\n")
    if tokenizer is not None:
        from .hf_tokenizer import tokenizer_to_hf_json

        tj, tc = tokenizer_to_hf_json(tokenizer)
        (out / "tokenizer.json").write_text(json.dumps(tj, ensure_ascii=False), encoding="utf-8")
        (out / "tokenizer_config.json").write_text(json.dumps(tc, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def _hf_name(ours: str) -> str:
    if ours == "tok_emb.weight":
        return "model.embed_tokens.weight"
    if ours == "norm_f.weight":
        return "model.norm.weight"
    if ours == "lm_head.weight":
        return "lm_head.weight"
    m = re.match(r"blocks\.(\d+)\.(.+)$", ours)
    i, rest = m.groups()
    table = {"norm1.weight": "input_layernorm.weight", "norm2.weight": "post_attention_layernorm.weight"}
    if rest in table:
        rest = table[rest]
    elif rest.startswith("attn."):
        rest = "self_attn." + rest[5:]
    elif rest.startswith("mlp."):
        rest = re.sub(r"mlp\.(gate|up|down)\.", r"mlp.\1_proj.", rest)
    return f"model.layers.{i}.{rest}"
