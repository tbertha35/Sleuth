"""One entry point for loading any model Sleuth understands.

    model, tokenizer, info = load_any(path, device="cuda")

`path` can be:
  * a Sleuth run directory or snapshot (our own models)
  * a Hugging Face model directory (config.json + *.safetensors + tokenizer.json)
  * a .gguf file (llama.cpp format)

Whatever the source, the result is our own Transformer plus a tokenizer with
the same interface, so generation and every analyzer view work the same way.
"""

from __future__ import annotations

from pathlib import Path

import torch


def detect(path: str | Path) -> str:
    """'sleuth' (our run dir or snapshot), 'hf' (Hugging Face folder) or 'gguf'."""
    from .formats.hf import is_hf_dir

    p = Path(path)
    if p.is_file() and p.suffix == ".gguf":
        return "gguf"
    if p.is_file() and p.suffix == ".safetensors":
        return "sleuth"                      # one of our snapshot files
    if p.is_dir() and (p / "snapshots").is_dir():
        return "sleuth"                      # our run directory
    if p.is_dir() and (p / "model.safetensors").exists() and not (p / "config.json").exists():
        return "sleuth"                      # our snapshot directory
    if is_hf_dir(p):
        return "hf"
    raise ValueError(f"don't know how to load {p} (expected a Sleuth run, a Hugging Face model folder, or a .gguf file)")


def default_dtype(device: torch.device) -> torch.dtype:
    """bf16 on GPU halves memory at no real cost; CPUs are often slow at bf16."""
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def load_any(path: str | Path, device: str | torch.device = "cpu", dtype: torch.dtype | None = None,
             step: int | None = None, max_seq_len: int = 4096):
    device = torch.device(device)
    kind = detect(path)
    if kind == "sleuth":
        from .forge.tokenizer.bpe import Tokenizer
        from .forge.train.checkpoint import load_model, resolve_weights

        weights = resolve_weights(path, step, nearest=True)
        model, meta = load_model(path, step=step, device=device, dtype=dtype or torch.float32, nearest=True)
        tok_path = next((d / "tokenizer.json" for d in weights.parents if (d / "tokenizer.json").exists()), None)
        if tok_path is None:
            raise FileNotFoundError("couldn't find tokenizer.json next to the checkpoint")
        return model, Tokenizer.load(tok_path), {"format": "sleuth", "step": meta.get("step")}
    if kind == "hf":
        from .formats.hf import load_hf_model
        from .formats.hf_tokenizer import HFTokenizer

        model, info = load_hf_model(path, device=device, dtype=dtype or default_dtype(device), max_seq_len=max_seq_len)
        return model, HFTokenizer.load(path), info
    from .formats.gguf_model import load_gguf_model, tokenizer_from_gguf

    model, info = load_gguf_model(path, device=device, dtype=dtype or default_dtype(device), max_seq_len=max_seq_len)
    return model, tokenizer_from_gguf(path), info


def load_tokenizer(path: str | Path):
    """Just the tokenizer (cheap: no weights are read)."""
    kind = detect(path)
    if kind == "hf":
        from .formats.hf_tokenizer import HFTokenizer

        return HFTokenizer.load(path)
    if kind == "gguf":
        from .formats.gguf_model import tokenizer_from_gguf

        return tokenizer_from_gguf(path)
    from .forge.tokenizer.bpe import Tokenizer
    from .forge.train.checkpoint import resolve_weights

    weights = resolve_weights(path)
    return Tokenizer.load(next(d / "tokenizer.json" for d in weights.parents if (d / "tokenizer.json").exists()))


def describe_file(path: str | Path, max_groups: int = 60) -> str:
    """Human-readable summary of a GGUF file or HF folder: architecture,
    hyperparameters, tokenizer, and tensors grouped by name pattern
    (blk.0..47.attn_q.weight shown once)."""
    import re
    from collections import Counter, OrderedDict

    kind = detect(path)
    lines: list[str] = []
    groups: "OrderedDict[str, list]" = OrderedDict()

    def add_tensor(name, shape, tname, nbytes):
        key = re.sub(r"\.(\d+)\.", ".{N}.", name)
        layer = re.search(r"\.(\d+)\.", name)
        g = groups.setdefault(key, [shape, Counter(), 0, set()])
        g[1][tname] += 1
        g[2] += nbytes
        if layer:
            g[3].add(int(layer.group(1)))

    if kind == "gguf":
        from .formats.gguf import GGUFFile

        with GGUFFile(path) as g:
            lines.append(f"GGUF v{g.version}, architecture: {g.architecture}, {len(g.tensors)} tensors")
            lines.append("metadata:")
            for k, v in g.metadata.items():
                if isinstance(v, list):
                    preview = ", ".join(repr(x) for x in v[:8])
                    v = f"[{len(v)} items] {preview}{', ...' if len(v) > 8 else ''}"
                elif isinstance(v, str) and len(v) > 200:
                    v = v[:200] + f"... ({len(v)} chars)"
                lines.append(f"  {k} = {v}")
            for t in g.tensors.values():
                add_tensor(t.name, t.shape, t.type_name, t.nbytes)
            total = sum(t.nbytes for t in g.tensors.values())
    elif kind == "hf":
        import json

        from .formats.hf import Checkpoint

        d = Path(path)
        lines.append(f"Hugging Face model folder: {d.name}")
        lines.append("config.json:")
        for k, v in json.loads((d / "config.json").read_text()).items():
            lines.append(f"  {k} = {v}")
        ck = Checkpoint(d)
        total = 0
        for k in ck.keys():
            info = ck.where[k].info(k)
            n = info["data_offsets"][1] - info["data_offsets"][0]
            add_tensor(k, tuple(info["shape"]), info["dtype"], n)
            total += n
        ck.close()
    else:
        raise ValueError("inspect works on .gguf files and Hugging Face model folders")
    lines.append(f"tensors ({total / 2**30:.2f} GiB), grouped:")
    for i, (key, (shape, types, nbytes, layers)) in enumerate(groups.items()):
        if i >= max_groups:
            lines.append(f"  ... {len(groups) - max_groups} more groups")
            break
        span = f" layers {min(layers)}..{max(layers)} ({len(layers)})" if layers else ""
        tstr = ", ".join(f"{t}x{c}" for t, c in types.items())
        lines.append(f"  {key}  shape {tuple(shape)}  [{tstr}]{span}  {nbytes / 2**20:.1f} MiB")
    return "\n".join(lines)
