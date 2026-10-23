"""Our own reader/writer for the .safetensors format.

The format (documented at github.com/huggingface/safetensors) is simple:

    [8 bytes]  N = header length, unsigned 64-bit little-endian
    [N bytes]  JSON header, UTF-8, padded with spaces to a multiple of 8:
               {"tensor.name": {"dtype": "BF16", "shape": [2, 3],
                                "data_offsets": [begin, end]},
                ...,
                "__metadata__": {"any": "string -> string pairs"}}
    [rest]     raw tensor bytes, little-endian, row-major (C order).
               data_offsets are relative to the start of this section.

It holds no pickled Python objects, so loading a file can't execute code
(unlike torch.load on untrusted files). The fixed layout also lets us mmap
the file and read single tensors without loading everything. That matters
for the analyzer, which may only want one layer of a 70B model.

We use it for our own checkpoints too, so one reader serves both.
"""

from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

# safetensors dtype string <-> torch dtype
DTYPES: dict[str, torch.dtype] = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
    "U8": torch.uint8, "BOOL": torch.bool,
}
for _name, _attr in (("F8_E4M3", "float8_e4m3fn"), ("F8_E5M2", "float8_e5m2"),
                     ("U16", "uint16"), ("U32", "uint32"), ("U64", "uint64")):
    if hasattr(torch, _attr):  # newer dtypes; only present in recent PyTorch
        DTYPES[_name] = getattr(torch, _attr)
TORCH_TO_ST = {v: k for k, v in DTYPES.items()}

MAX_HEADER = 100 * 2**20  # a sane cap so a corrupt file can't make us allocate GBs


def save_file(tensors: dict[str, torch.Tensor], path: str | Path, metadata: dict[str, str] | None = None) -> None:
    header: dict = {}
    offset = 0
    blobs: list[torch.Tensor] = []
    for name, t in tensors.items():
        if t.dtype not in TORCH_TO_ST:
            raise TypeError(f"{name}: dtype {t.dtype} not supported by safetensors")
        t = t.detach().to("cpu").contiguous()
        nbytes = t.numel() * t.element_size()
        header[name] = {"dtype": TORCH_TO_ST[t.dtype], "shape": list(t.shape), "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
        blobs.append(t)
    if metadata:
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()):
            raise TypeError("safetensors metadata must be str -> str")
        header["__metadata__"] = metadata
    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hjson += b" " * (-len(hjson) % 8)  # pad so tensor data starts 8-byte aligned

    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for t in blobs:
            if t.numel():
                # Reinterpret as raw bytes; works for bf16 too (numpy has no bf16).
                f.write(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    tmp.replace(path)  # atomic: a crash never leaves a half-written checkpoint


class SafetensorsFile:
    """Lazy, mmap-backed reader. Use as a context manager:

        with SafetensorsFile("model.safetensors") as f:
            print(f.keys(), f.metadata)
            w = f.get("blocks.0.attn.q_proj.weight")
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh = open(self.path, "rb")
        size = self.path.stat().st_size
        if size < 8:
            raise ValueError(f"{path}: too small to be safetensors")
        (n,) = struct.unpack("<Q", self._fh.read(8))
        if n > MAX_HEADER or 8 + n > size:
            raise ValueError(f"{path}: bad header length {n}")
        header = json.loads(self._fh.read(n).decode("utf-8"))
        self.metadata: dict[str, str] = header.pop("__metadata__", None) or {}
        self.header: dict[str, dict] = header
        self._data_start = 8 + n
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ) if size > self._data_start else None
        for name, info in header.items():
            b, e = info["data_offsets"]
            if info["dtype"] not in DTYPES:
                raise ValueError(f"{name}: unsupported dtype {info['dtype']}")
            expected = int(np.prod(info["shape"], dtype=np.int64)) * torch.empty((), dtype=DTYPES[info["dtype"]]).element_size()
            if e - b != expected or self._data_start + e > size:
                raise ValueError(f"{path}: tensor {name} has inconsistent offsets")

    def keys(self) -> list[str]:
        return list(self.header)

    def __contains__(self, name: str) -> bool:
        return name in self.header

    def info(self, name: str) -> dict:
        return self.header[name]

    def get(self, name: str) -> torch.Tensor:
        """Read one tensor (copied into normal memory, so it outlives the file)."""
        info = self.header[name]
        dtype = DTYPES[info["dtype"]]
        b, e = info["data_offsets"]
        if e == b:
            return torch.empty(info["shape"], dtype=dtype)
        raw = self._mm[self._data_start + b: self._data_start + e]  # bytes copy
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(dtype).reshape(info["shape"])

    def items(self) -> Iterator[tuple[str, torch.Tensor]]:
        for k in self.header:
            yield k, self.get(k)

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_file(path: str | Path) -> dict[str, torch.Tensor]:
    with SafetensorsFile(path) as f:
        return dict(f.items())


def read_metadata(path: str | Path) -> dict[str, str]:
    with SafetensorsFile(path) as f:
        return dict(f.metadata)
