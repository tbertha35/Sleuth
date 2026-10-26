"""Helpers shared by tests (pytest puts tests/ on sys.path)."""

import struct
from pathlib import Path


def tiny_run_config(tiny_data, runs_dir: Path, **train_overrides):
    from sleuth.forge.model.config import ModelConfig
    from sleuth.forge.train.config import DataConfig, RunConfig, TrainConfig

    tc = dict(device="cpu", micro_batch_size=4, seq_len=32, max_steps=6, lr=1e-3, warmup_steps=2,
              log_every=2, eval_every=3, eval_batches=2, snapshot_every=3, resume_every=3, keep_resume=5)
    tc.update(train_overrides)
    return RunConfig(
        name="test", runs_dir=str(runs_dir),
        model=ModelConfig(d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64),
        train=TrainConfig(**tc),
        data=DataConfig(shards=str(tiny_data["shards"])),
    )


def gguf_string(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, metadata, tensors, align=32):
    """A minimal GGUF writer, for tests only. metadata: {key: (type, value)}; tensors: {name: (ggml_type, torch_shape, raw bytes)}."""
    fmt = {0: "<B", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q"}
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for k, (t, v) in metadata.items():
        out += gguf_string(k) + struct.pack("<I", t)
        if t == 8:
            out += gguf_string(v)
        elif t == 9:
            et, items = v
            out += struct.pack("<IQ", et, len(items))
            for it in items:
                out += gguf_string(it) if et == 8 else struct.pack(fmt[et], it)
        else:
            out += struct.pack(fmt[t], v)
    offset, blobs = 0, []
    for name, (t, shape, raw) in tensors.items():
        out += gguf_string(name) + struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", d) for d in reversed(shape))
        out += struct.pack("<IQ", t, offset)
        pad = (-len(raw)) % align
        blobs.append(raw + b"\0" * pad)
        offset += len(raw) + pad
    out += b"\0" * ((-len(out)) % align)
    path.write_bytes(out + b"".join(blobs))


def add_gguf_tensor_copy(src, dst, name, new_name):
    """Rewrite a GGUF file (as written by write_gguf) with one tensor duplicated
    under a new name. Reads it back with our reader, so it's for small test files."""
    from sleuth.formats.gguf import GGUFFile

    with GGUFFile(src) as g:
        md = g.metadata
        tensors = {n: (t.ggml_type, t.shape, bytes(g.raw(n).numpy())) for n, t in g.tensors.items()}
    tensors[new_name] = tensors[name]
    kinds = {bool: 7, int: 5, float: 6, str: 8}
    meta = {}
    for k, v in md.items():
        if isinstance(v, list):
            meta[k] = (9, (kinds[type(v[0])], v))
        else:
            meta[k] = (kinds[type(v)], v)
    write_gguf(dst, meta, tensors)


def moe_reference(x, router_logits, n_used, gate, up, down, act, down_scale=None, weights_scale=0.0):
    """llama.cpp's build_moe_ffn for one layer, token by token and expert by expert,
    with plain tensors (the check for forge/model/moe.py). x [n, d]; router_logits
    [n, E]; gate/up [E, h, d]; down [E, d, h]; act: a function (silu or gelu-tanh)."""
    import torch

    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        p = router_logits[t].softmax(-1)
        top = sorted(range(p.numel()), key=lambda e: -float(p[e]))[:n_used]
        w = p[top] / max(float(p[top].sum()), 6.103515625e-5)        # norm_w
        if weights_scale not in (0.0, 1.0):
            w = w * weights_scale
        for j, e in enumerate(top):
            y = down[e] @ (act(gate[e] @ x[t]) * (up[e] @ x[t]))
            if down_scale is not None:
                y = y * down_scale[e]
            out[t] += w[j] * y
    return out
