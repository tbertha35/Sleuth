import json
import struct

import pytest
import torch

from sleuth.formats.safetensors import SafetensorsFile, load_file, read_metadata, save_file


def test_roundtrip_dtypes(tmp_path):
    tensors = {
        "f32": torch.randn(3, 4),
        "bf16": torch.randn(5).to(torch.bfloat16),
        "f16": torch.randn(2, 2, 2).to(torch.float16),
        "i64": torch.arange(7),
        "u8": torch.arange(10, dtype=torch.uint8),
        "bool": torch.tensor([True, False]),
        "scalar": torch.tensor(3.5),
        "empty": torch.zeros(0, 4),
        "noncontig": torch.randn(4, 6).t(),
    }
    save_file(tensors, tmp_path / "t.safetensors", metadata={"hello": "world"})
    back = load_file(tmp_path / "t.safetensors")
    assert back.keys() == tensors.keys()
    for k, v in tensors.items():
        assert back[k].dtype == v.dtype and back[k].shape == v.shape
        assert torch.equal(back[k], v), k
    assert read_metadata(tmp_path / "t.safetensors") == {"hello": "world"}


def test_layout_follows_spec(tmp_path):
    """Check the bytes directly against the documented format."""
    save_file({"a": torch.tensor([1.0, 2.0])}, tmp_path / "t.safetensors")
    raw = (tmp_path / "t.safetensors").read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    assert (8 + n) % 8 == 0  # data section is 8-byte aligned
    header = json.loads(raw[8:8 + n])
    assert header["a"] == {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
    assert raw[8 + n:] == struct.pack("<2f", 1.0, 2.0)


def test_reads_handwritten_file(tmp_path):
    """A file written by hand (as another tool would) must load."""
    header = json.dumps({"w": {"dtype": "I32", "shape": [2, 2], "data_offsets": [0, 16]},
                         "__metadata__": {"format": "pt"}}).encode()
    p = tmp_path / "h.safetensors"
    p.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<4i", 1, 2, 3, 4))
    with SafetensorsFile(p) as f:
        assert f.metadata == {"format": "pt"}
        assert f.get("w").tolist() == [[1, 2], [3, 4]]


def test_rejects_corrupt_files(tmp_path):
    p = tmp_path / "bad.safetensors"
    p.write_bytes(struct.pack("<Q", 10**12) + b"{}")
    with pytest.raises(ValueError):
        SafetensorsFile(p)
    header = json.dumps({"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 8]}}).encode()
    p.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 8)
    with pytest.raises(ValueError):  # shape says 16 bytes, offsets say 8
        SafetensorsFile(p)
