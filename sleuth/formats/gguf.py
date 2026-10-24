"""Our own reader for GGUF, llama.cpp's model file format.

Layout (all little-endian), versions 2 and 3:

    "GGUF" magic, uint32 version, uint64 tensor_count, uint64 metadata_count
    metadata:  key (string), uint32 value type, value          x metadata_count
    tensors:   name (string), uint32 n_dims, uint64 dims[n_dims],
               uint32 ggml type, uint64 offset                  x tensor_count
    padding to `general.alignment` (default 32)
    tensor data (each tensor at data_start + offset)

    string = uint64 length + UTF-8 bytes
    value types: 0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32, 7 bool,
                 8 string, 9 array (uint32 elem type, uint64 count, items),
                 10 u64, 11 i64, 12 f64

Two things that trip people up:
  * dims are stored innermost-first (ggml's ne[0] = row length). A PyTorch
    weight of shape [out, in] appears as dims [in, out]. We reverse them.
  * most tensors are *quantized*: stored as blocks of a few bytes holding a
    scale plus small integers (see gguf_quants.py). `tensor()` dequantizes;
    `raw()` gives the blocks as-is (for keeping weights quantized in memory).

The metadata carries everything else: architecture ("general.architecture"),
hyperparameters ("<arch>.block_count", ...), and the whole tokenizer
("tokenizer.ggml.tokens", ".merges", ".scores", ...).

The file is memory-mapped, so opening a 7 GB model costs no RAM until
tensors are actually read.
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import gguf_quants as q

MAGIC = b"GGUF"
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_NP = {0: np.uint8, 1: np.int8, 2: np.uint16, 3: np.int16, 4: np.uint32, 5: np.int32, 6: np.float32,
       7: np.bool_, 10: np.uint64, 11: np.int64, 12: np.float64}


@dataclass
class TensorInfo:
    name: str
    shape: tuple[int, ...]      # PyTorch order (outermost first)
    ggml_type: int
    offset: int                 # absolute offset in the file
    nbytes: int

    @property
    def type_name(self) -> str:
        return q.TYPE_NAMES.get(self.ggml_type, f"type{self.ggml_type}")

    @property
    def n_elements(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))


class GGUFFile:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh = open(self.path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        self._pos = 0
        if self._read(4) != MAGIC:
            raise ValueError(f"{path} is not a GGUF file")
        self.version = self._unpack("<I")
        if self.version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {self.version}")
        n_tensors = self._unpack("<Q")
        n_kv = self._unpack("<Q")
        self.metadata: dict[str, object] = {}
        for _ in range(n_kv):
            key = self._string()
            self.metadata[key] = self._value(self._unpack("<I"))
        infos = []
        for _ in range(n_tensors):
            name = self._string()
            n_dims = self._unpack("<I")
            dims = [self._unpack("<Q") for _ in range(n_dims)]
            ggml_type = self._unpack("<I")
            offset = self._unpack("<Q")
            infos.append((name, tuple(reversed(dims)), ggml_type, offset))
        align = int(self.metadata.get("general.alignment", 32))
        self.data_start = (self._pos + align - 1) // align * align
        size = len(self._mm)
        self.tensors: dict[str, TensorInfo] = {}
        for name, shape, t, off in infos:
            n = int(np.prod(shape, dtype=np.int64))
            nbytes = q.tensor_nbytes(t, n)
            start = self.data_start + off
            if start + nbytes > size:
                raise ValueError(f"tensor {name} runs past the end of the file (truncated download?)")
            self.tensors[name] = TensorInfo(name, shape, t, start, nbytes)

    # ---- low-level parsing
    def _read(self, n: int) -> bytes:
        b = self._mm[self._pos:self._pos + n]
        if len(b) != n:
            raise ValueError("unexpected end of GGUF header")
        self._pos += n
        return b

    def _unpack(self, fmt: str):
        return struct.unpack(fmt, self._read(struct.calcsize(fmt)))[0]

    def _string(self) -> str:
        return self._read(self._unpack("<Q")).decode("utf-8", errors="replace")

    def _value(self, vtype: int):
        if vtype in _SCALARS:
            return self._unpack(_SCALARS[vtype])
        if vtype == 8:
            return self._string()
        if vtype == 9:
            etype = self._unpack("<I")
            count = self._unpack("<Q")
            if etype in _NP:  # numeric arrays (e.g. token scores) in one go
                arr = np.frombuffer(self._read(count * np.dtype(_NP[etype]).itemsize), dtype=_NP[etype])
                return arr.tolist()
            return [self._value(etype) for _ in range(count)]
        raise ValueError(f"unknown GGUF value type {vtype}")

    # ---- public API
    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", "unknown"))

    def arch_value(self, key: str, default=None):
        """Metadata for this architecture, e.g. arch_value('block_count')."""
        return self.metadata.get(f"{self.architecture}.{key}", default)

    def raw(self, name: str) -> torch.Tensor:
        """The tensor's bytes, unconverted (uint8, a copy owned by PyTorch)."""
        info = self.tensors[name]
        return torch.frombuffer(bytearray(self._mm[info.offset:info.offset + info.nbytes]), dtype=torch.uint8)

    def same_bytes(self, a: str, b: str, chunk: int = 1 << 26) -> bool:
        """True if tensors a and b have the same type, shape and stored bytes
        (compared in chunks, so huge tensors don't need extra RAM)."""
        ta, tb = self.tensors[a], self.tensors[b]
        if (ta.ggml_type, ta.shape) != (tb.ggml_type, tb.shape):
            return False
        for i in range(0, ta.nbytes, chunk):
            n = min(chunk, ta.nbytes - i)
            if self._mm[ta.offset + i:ta.offset + i + n] != self._mm[tb.offset + i:tb.offset + i + n]:
                return False
        return True

    def tensor(self, name: str, dtype: torch.dtype = torch.float32, device: str | torch.device = "cpu") -> torch.Tensor:
        """Dequantized tensor in PyTorch shape order."""
        info = self.tensors[name]
        flat = q.dequantize(self.raw(name).to(device), info.ggml_type, info.n_elements)
        return flat.reshape(info.shape).to(dtype)

    def close(self) -> None:
        self._mm.close()
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
