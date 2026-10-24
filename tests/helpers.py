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
