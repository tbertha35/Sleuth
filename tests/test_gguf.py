"""GGUF reader + dequantizers.

Two independent checks for the quantization formats:
  1. simple formats: quantize real numbers with a reference quantizer written
     from the spec, then dequantize: values must come back within the quantization
     step (wrong nibble order or offsets would make the error huge).
  2. K-quants: a slow, line-by-line transcription of ggml's C reference
     decoders (dequantize_row_*) is compared with our vectorized versions on
     random blocks.
"""

import struct

import numpy as np
import pytest
import torch

from sleuth.formats import gguf_quants as q
from sleuth.formats.gguf import GGUFFile

from helpers import write_gguf

RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# 1. simple formats: reference quantizers (from ggml's quantize_row_*_ref)
# --------------------------------------------------------------------------- #

def f16b(x):
    return np.float16(x).tobytes()


def quant_q8_0(x):
    out = b""
    for blk in x.reshape(-1, 32):
        d = np.abs(blk).max() / 127 or 1.0
        out += f16b(d) + np.round(blk / d).astype(np.int8).tobytes()
    return out


def quant_q4_0(x):
    out = b""
    for blk in x.reshape(-1, 32):
        mx = blk[np.argmax(np.abs(blk))]
        d = mx / -8 or 1.0
        qv = np.clip(np.floor(blk / d + 8.5), 0, 15).astype(np.uint8)
        out += f16b(d) + (qv[:16] | (qv[16:] << 4)).astype(np.uint8).tobytes()
    return out


def quant_q4_1(x):
    out = b""
    for blk in x.reshape(-1, 32):
        lo, hi = blk.min(), blk.max()
        d = (hi - lo) / 15 or 1.0
        qv = np.clip(np.floor((blk - lo) / d + 0.5), 0, 15).astype(np.uint8)
        out += f16b(d) + f16b(lo) + (qv[:16] | (qv[16:] << 4)).astype(np.uint8).tobytes()
    return out


def quant_q5_0(x):
    out = b""
    for blk in x.reshape(-1, 32):
        mx = blk[np.argmax(np.abs(blk))]
        d = mx / -16 or 1.0
        qv = np.clip(np.floor(blk / d + 16.5), 0, 31).astype(np.uint32)
        qh = 0
        for j in range(32):
            qh |= ((int(qv[j]) >> 4) & 1) << j
        lo = (qv & 0x0F).astype(np.uint8)
        out += f16b(d) + struct.pack("<I", qh) + (lo[:16] | (lo[16:] << 4)).astype(np.uint8).tobytes()
    return out


def quant_q5_1(x):
    out = b""
    for blk in x.reshape(-1, 32):
        lo_v, hi_v = blk.min(), blk.max()
        d = (hi_v - lo_v) / 31 or 1.0
        qv = np.clip(np.floor((blk - lo_v) / d + 0.5), 0, 31).astype(np.uint32)
        qh = 0
        for j in range(32):
            qh |= ((int(qv[j]) >> 4) & 1) << j
        lo = (qv & 0x0F).astype(np.uint8)
        out += f16b(d) + f16b(lo_v) + struct.pack("<I", qh) + (lo[:16] | (lo[16:] << 4)).astype(np.uint8).tobytes()
    return out


@pytest.mark.parametrize("qtype,quant,steps", [
    (q.Q8_0, quant_q8_0, 127), (q.Q4_0, quant_q4_0, 8), (q.Q4_1, quant_q4_1, 15),
    (q.Q5_0, quant_q5_0, 16), (q.Q5_1, quant_q5_1, 31),
])
def test_simple_formats_roundtrip(qtype, quant, steps):
    x = RNG.normal(size=32 * 64).astype(np.float32)
    raw = torch.frombuffer(bytearray(quant(x)), dtype=torch.uint8)
    y = q.dequantize(raw, qtype, x.size).numpy()
    blk_max = np.abs(x.reshape(-1, 32)).max(1, keepdims=True)
    tol = 2 * blk_max / steps + 1e-3  # one quantization step, plus f16 rounding of the scale
    assert np.all(np.abs(y.reshape(-1, 32) - x.reshape(-1, 32)) <= tol)


@pytest.mark.parametrize("qtype,quant", [(q.Q8_0, quant_q8_0), (q.Q4_0, quant_q4_0), (q.Q4_1, quant_q4_1)])
def test_vectorized_quantizers_match_reference(qtype, quant):
    """Our vectorized quantizers (used by `compare --requant`) write the same bytes."""
    x = RNG.normal(size=32 * 64).astype(np.float32)
    x[:32] = 0.0  # an all-zero block takes the d = 0 branch
    ours = q.quantize(torch.from_numpy(x), qtype)
    assert bytes(ours.numpy()) == quant(x)


def test_plain_types():
    x = torch.randn(10)
    for t, dt in [(q.F32, torch.float32), (q.F16, torch.float16), (q.BF16, torch.bfloat16)]:
        raw = x.to(dt).view(torch.uint8)
        torch.testing.assert_close(q.dequantize(raw, t, 10), x.to(dt).float())


# --------------------------------------------------------------------------- #
# 2. K-quants: line-by-line transcriptions of ggml's dequantize_row_* (slow)
# --------------------------------------------------------------------------- #

def h(b, i):
    return float(np.frombuffer(bytes(b[i:i + 2]), dtype=np.float16)[0])


def i8(v):
    return v - 256 if v > 127 else v


def get_scale_min_k4(j, s):
    if j < 4:
        return s[j] & 63, s[j + 4] & 63
    return (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4), (s[j + 4] >> 4) | ((s[j] >> 6) << 4)


def ref_q2_k(b):
    scales, qs, d, dmin = b[0:16], b[16:80], h(b, 80), h(b, 82)
    y, is_, q = [], 0, 0
    for _ in range(2):
        shift = 0
        for _ in range(4):
            sc = scales[is_]; is_ += 1
            y += [d * (sc & 0xF) * ((qs[q + l] >> shift) & 3) - dmin * (sc >> 4) for l in range(16)]
            sc = scales[is_]; is_ += 1
            y += [d * (sc & 0xF) * ((qs[q + 16 + l] >> shift) & 3) - dmin * (sc >> 4) for l in range(16)]
            shift += 2
        q += 32
    return y


def ref_q3_k(b):
    hm, qs, sc12, d = b[0:32], b[32:96], b[96:108], h(b, 108)
    aux = list(struct.unpack("<3I", bytes(sc12))) + [0]
    k1, k2 = 0x03030303, 0x0F0F0F0F
    tmp = aux[2]
    aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4)
    aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4)
    aux[0] = (aux[0] & k2) | (((tmp >> 0) & k1) << 4)
    aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4)
    scales = [i8(v) for v in struct.pack("<4I", *aux)]
    y, is_, m, q = [], 0, 1, 0
    for _ in range(2):
        shift = 0
        for _ in range(4):
            dl = d * (scales[is_] - 32); is_ += 1
            y += [dl * (((qs[q + l] >> shift) & 3) - (0 if hm[l] & m else 4)) for l in range(16)]
            dl = d * (scales[is_] - 32); is_ += 1
            y += [dl * (((qs[q + 16 + l] >> shift) & 3) - (0 if hm[16 + l] & m else 4)) for l in range(16)]
            shift += 2
            m <<= 1
        q += 32
    return y


def ref_q4_k(b):
    d, dmin, s, qs = h(b, 0), h(b, 2), b[4:16], b[16:144]
    y, is_, q = [], 0, 0
    for _ in range(4):
        sc, m = get_scale_min_k4(is_, s); d1, m1 = d * sc, dmin * m
        sc, m = get_scale_min_k4(is_ + 1, s); d2, m2 = d * sc, dmin * m
        y += [d1 * (qs[q + l] & 0xF) - m1 for l in range(32)]
        y += [d2 * (qs[q + l] >> 4) - m2 for l in range(32)]
        q += 32; is_ += 2
    return y


def ref_q5_k(b):
    d, dmin, s, qh, ql = h(b, 0), h(b, 2), b[4:16], b[16:48], b[48:176]
    y, is_, q, u1, u2 = [], 0, 0, 1, 2
    for _ in range(4):
        sc, m = get_scale_min_k4(is_, s); d1, m1 = d * sc, dmin * m
        sc, m = get_scale_min_k4(is_ + 1, s); d2, m2 = d * sc, dmin * m
        y += [d1 * ((ql[q + l] & 0xF) + (16 if qh[l] & u1 else 0)) - m1 for l in range(32)]
        y += [d2 * ((ql[q + l] >> 4) + (16 if qh[l] & u2 else 0)) - m2 for l in range(32)]
        q += 32; is_ += 2; u1 <<= 2; u2 <<= 2
    return y


def ref_q6_k(b):
    ql, qh, sc, d = b[0:128], b[128:192], [i8(v) for v in b[192:208]], h(b, 208)
    y = [0.0] * 256
    for n in range(2):
        L, H, S, Y = ql[64 * n:], qh[32 * n:], sc[8 * n:], 128 * n
        for l in range(32):
            is_ = l // 16
            q1 = ((L[l] & 0xF) | (((H[l] >> 0) & 3) << 4)) - 32
            q2 = ((L[l + 32] & 0xF) | (((H[l] >> 2) & 3) << 4)) - 32
            q3 = ((L[l] >> 4) | (((H[l] >> 4) & 3) << 4)) - 32
            q4 = ((L[l + 32] >> 4) | (((H[l] >> 6) & 3) << 4)) - 32
            y[Y + l], y[Y + l + 32] = d * S[is_] * q1, d * S[is_ + 2] * q2
            y[Y + l + 64], y[Y + l + 96] = d * S[is_ + 4] * q3, d * S[is_ + 6] * q4
    return y


def ref_iq4_xs(b):
    d, sh, sl, qs = h(b, 0), b[2] | (b[3] << 8), b[4:8], b[8:136]
    y = []
    for ib in range(8):
        ls = ((sl[ib // 2] >> 4 * (ib % 2)) & 0xF) | (((sh >> 2 * ib) & 3) << 4)
        dl = d * (ls - 32)
        blk = qs[16 * ib:16 * ib + 16]
        y += [dl * q.IQ4_LEVELS[v & 0xF] for v in blk] + [dl * q.IQ4_LEVELS[v >> 4] for v in blk]
    return y


def random_blocks(qtype, n, f16_offsets):
    """Random block bytes, with the f16 scale fields set to small finite values."""
    nb = q.BLOCK[qtype][1]
    raw = RNG.integers(0, 256, size=(n, nb), dtype=np.uint8)
    for off in f16_offsets:
        raw[:, off:off + 2] = np.frombuffer(RNG.uniform(-0.1, 0.1, n).astype(np.float16).tobytes(),
                                            dtype=np.uint8).reshape(n, 2)
    return raw


@pytest.mark.parametrize("qtype,ref,f16s", [
    (q.Q2_K, ref_q2_k, [80, 82]), (q.Q3_K, ref_q3_k, [108]), (q.Q4_K, ref_q4_k, [0, 2]),
    (q.Q5_K, ref_q5_k, [0, 2]), (q.Q6_K, ref_q6_k, [208]), (q.IQ4_XS, ref_iq4_xs, [0]),
])
def test_k_quants_match_reference(qtype, ref, f16s):
    raw = random_blocks(qtype, 24, f16s)
    # Python ints, not numpy uint8: C promotes to int, numpy uint8 would wrap (2 - 4 = 254).
    expected = np.array([ref([int(v) for v in row]) for row in raw], dtype=np.float64).reshape(-1)
    got = q.dequantize(torch.from_numpy(raw.reshape(-1).copy()), qtype, 24 * 256).double().numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_unsupported_type_is_named():
    with pytest.raises(NotImplementedError, match="IQ2_XXS"):
        q.tensor_nbytes(16, 256)


# --------------------------------------------------------------------------- #
# 3. the file reader, on a tiny GGUF written here
# --------------------------------------------------------------------------- #

def test_reader_roundtrip(tmp_path):
    w = torch.randn(4, 64)
    v8 = RNG.normal(size=(2, 32)).astype(np.float32)
    write_gguf(tmp_path / "m.gguf", {
        "general.architecture": (8, "llama"),
        "llama.block_count": (4, 2),
        "llama.rope.freq_base": (6, 10000.0),
        "tokenizer.ggml.tokens": (9, (8, ["<s>", "a", "b"])),
        "tokenizer.ggml.scores": (9, (6, [0.0, -1.0, -2.0])),
        "some.flag": (7, True),
    }, {
        "w.f32": (q.F32, (4, 64), w.numpy().tobytes()),
        "w.f16": (q.F16, (4, 64), w.half().numpy().tobytes()),
        "w.q8": (q.Q8_0, (2, 32), quant_q8_0(v8.reshape(-1))),
    })
    with GGUFFile(tmp_path / "m.gguf") as f:
        assert f.architecture == "llama" and f.arch_value("block_count") == 2
        assert f.metadata["tokenizer.ggml.tokens"] == ["<s>", "a", "b"]
        assert f.metadata["tokenizer.ggml.scores"] == [0.0, -1.0, -2.0]
        assert f.metadata["some.flag"] is True
        assert f.tensors["w.f32"].shape == (4, 64)  # PyTorch order
        torch.testing.assert_close(f.tensor("w.f32"), w)
        torch.testing.assert_close(f.tensor("w.f16"), w.half().float())
        assert np.abs(f.tensor("w.q8").numpy() - v8).max() < 0.05
        assert f.tensors["w.q8"].type_name == "Q8_0"


def test_reader_rejects_truncated_file(tmp_path):
    write_gguf(tmp_path / "m.gguf", {}, {"w": (q.F32, (64,), np.zeros(64, np.float32).tobytes())})
    data = (tmp_path / "m.gguf").read_bytes()
    (tmp_path / "t.gguf").write_bytes(data[:-16])
    with pytest.raises(ValueError, match="truncated"):
        GGUFFile(tmp_path / "t.gguf")


@pytest.mark.parametrize("qtype,quant", [(q.Q4_0, quant_q4_0), (q.Q8_0, quant_q8_0)])
def test_lean_dequantizers_match_reference(qtype, quant):
    """The fast Q4_0/Q8_0 paths (used by kept-quantized layers) give exactly the
    reference values: in fp32, and rounded once in bf16 (f16 goes the slow way)."""
    x = RNG.normal(size=32 * 512).astype(np.float32) * 0.05
    raw = torch.frombuffer(bytearray(quant(x)), dtype=torch.uint8)
    ref = q._DEQUANT[qtype](raw.reshape(-1, q.BLOCK[qtype][1])).reshape(-1)
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        got = q.dequantize(raw, qtype, x.size, dt)
        assert got.dtype == dt and torch.equal(got, ref.to(dt)), dt


def test_q8_activation_rounding_matches_q8_0():
    """round_like_q8 (llama.cpp's activation rounding) gives the same values as
    quantizing to Q8_0 and back: both use d = max|x|/127 per block of 32 and
    store d as fp16."""
    from sleuth.forge.model.quant import round_like_q8

    torch.manual_seed(0)
    x = torch.randn(3, 128) * torch.logspace(-2, 1, 128)
    x[1, :32] = 0.0                                   # an all-zero block stays zero
    ref = q.dequantize(q.quantize(x, q.Q8_0), q.Q8_0, x.numel(), torch.float32).reshape(x.shape)
    got = round_like_q8(x)
    step = x.reshape(3, 4, 32).abs().amax(-1, keepdim=True).expand(3, 4, 32).reshape(3, 128) / 127
    assert ((got - ref).abs() <= step * 1.001).all()  # at most one rounding step apart (x/d vs x*(1/d))...
    assert (got == ref).float().mean() > 0.99          # ...and almost always identical
    assert (got[1, :32] == 0).all()
    # Rounding error: half a step, plus up to 127 steps times fp16's relative error in d (2^-11).
    assert ((got - x).abs() <= step * (0.5 + 127 * 2 ** -11) + 1e-7).all()


def test_set_q8_activations_rounds_the_input_of_quantized_layers():
    import torch.nn.functional as F

    from sleuth.forge.model.quant import QuantLinear, round_like_q8, set_q8_activations

    torch.manual_seed(0)
    w = torch.randn(48, 64)
    lin = QuantLinear(64, 48, q.Q4_0)
    lin.raw = q.quantize(w, q.Q4_0)
    x = torch.randn(2, 5, 64)
    plain = lin(x)
    holder = torch.nn.Sequential(lin)
    assert set_q8_activations(holder) == 1
    rounded = lin(x)
    torch.testing.assert_close(rounded, F.linear(round_like_q8(x), lin.weight))
    assert not torch.equal(rounded, plain)
    set_q8_activations(holder, False)
    torch.testing.assert_close(lin(x), plain)
