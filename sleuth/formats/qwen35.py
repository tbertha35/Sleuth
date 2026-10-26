"""Qwen3.5 / Qwen3.6 / Qwen3.8 ("qwen35" and "qwen35moe" GGUF architectures) and their MTP drafter.

A hybrid of two layer kinds, following llama.cpp's src/models/qwen35.cpp and
delta-net-base.cpp (commit b2dd28a3b, 2026-07-16), the reference for these files:

  * Gated DeltaNet ("linear attention", 3 of every 4 layers). Instead of a KV cache
    that grows with the text, each value head keeps a fixed 128x128 state matrix M:
        q, k, v  <- one projection (attn_qkv), a short causal convolution over the
                    last 4 positions (ssm_conv1d, depthwise) and SiLU
        q, k     <- L2-normalized per head; q also x 1/sqrt(head_dim)
        g        = softplus(alpha + dt_bias) * ssm_a      (ssm_a = -exp(A_log) < 0)
        beta     = sigmoid(ssm_beta x)
        M        = exp(g) M;   M += beta (v - M k) k^T;   o = M q
        out      = ssm_out( RMSNorm(o) * silu(z) ),  z = attn_gate(x)
    Key heads are tiled to the value heads (value head h uses key head h % n_k).
  * gated full attention (every 4th layer): q and a gate come interleaved per head
    from attn_q; q_norm / k_norm (RMSNorm); RoPE on the first rope.dimension_count
    channels of each head (NEOX pairs; for text, M-RoPE's sections all see the same
    position, so it is ordinary RoPE); attention; times sigmoid(gate); attn_output.
  * every block: h = x + mixer(attn_norm(x)); out = h + FFN(post_attention_norm(h)).
    FFN is a SwiGLU MLP ("qwen35"), or for "qwen35moe" (Qwen3.6-35B-A3B; llama.cpp's
    qwen35moe.cpp) routed experts (forge/model/moe.py: softmax router, top 8 of 256,
    weights renormalized) plus a shared SwiGLU expert scaled by
    sigmoid(ffn_gate_inp_shexp . x), a learned per-token gate.

The extra block after the main stack (qwen35.nextn_predict_layers = 1) is the MTP
drafter (Qwen35MTP): cur = eh_proj([enorm(embed(token)), hnorm(h)]), one gated
full-attention block with its own KV cache, then shared_head_norm and the output layer.

Not yet: CUDA graphs and the analyzer's taps for this architecture, the chunked
(parallel) DeltaNet form for long prompts (the loop over tokens is exact but slow),
offloading experts to the CPU (the 35B doesn't fit in 16 GB), and vision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..forge.model.moe import ExpertWeights, MoE
from ..forge.model.norm import RMSNorm
from ..forge.model.quant import QuantEmbedding, QuantLinear
from ..forge.model.transformer import ModelOutput
from . import gguf_quants as q

PLAIN = {q.F32, q.F16, q.BF16, q.F64}
ARCHS = ("qwen35", "qwen35moe")


@dataclass
class Qwen35Config:
    vocab_size: int
    d_model: int
    n_layers: int            # main stack (without the MTP block)
    n_heads: int
    n_kv_heads: int
    head_dim: int
    n_rot: int
    rope_theta: float
    norm_eps: float
    ffn_hidden: int
    conv_kernel: int
    k_head_dim: int          # ssm.state_size
    n_k_heads: int           # ssm.group_count
    n_v_heads: int           # ssm.time_step_rank
    d_inner: int             # ssm.inner_size = n_v_heads * v_head_dim
    full_interval: int
    n_mtp: int
    max_seq_len: int = 4096
    # qwen35moe (0 = dense)
    n_experts: int = 0
    n_experts_used: int = 0
    expert_ffn: int = 0          # expert_feed_forward_length
    shared_ffn: int = 0          # expert_shared_feed_forward_length
    expert_weights_scale: float = 0.0   # 0 or 1 = no scaling (llama.cpp's convention)

    @property
    def v_head_dim(self) -> int:
        return self.d_inner // self.n_v_heads

    @property
    def conv_dim(self) -> int:
        return 2 * self.n_k_heads * self.k_head_dim + self.d_inner

    def recurrent(self, i: int) -> bool:
        return i < self.n_layers and (i + 1) % self.full_interval != 0

    @property
    def padded_vocab(self) -> int:
        return self.vocab_size


def config_from_gguf(g, max_seq_len: int = 4096) -> Qwen35Config:
    v = g.arch_value
    if g.arch_value("attention.recurrent_layers") is not None:
        raise NotImplementedError("qwen35 GGUF with an explicit recurrent-layer list isn't supported yet")
    sections = v("rope.dimension_sections")
    if sections and len(set(sections[:3])) > 2:
        pass  # text positions are the same in every section: plain RoPE (see the module docstring)
    n_mtp = int(v("nextn_predict_layers", 0) or 0)
    total = int(v("block_count"))
    return Qwen35Config(
        vocab_size=g.tensors["token_embd.weight"].shape[0], d_model=int(v("embedding_length")),
        n_layers=total - n_mtp, n_heads=int(v("attention.head_count")), n_kv_heads=int(v("attention.head_count_kv")),
        head_dim=int(v("attention.key_length")), n_rot=int(v("rope.dimension_count")),
        rope_theta=float(v("rope.freq_base")), norm_eps=float(v("attention.layer_norm_rms_epsilon")),
        ffn_hidden=int(v("feed_forward_length", 0) or 0), conv_kernel=int(v("ssm.conv_kernel")),
        k_head_dim=int(v("ssm.state_size")), n_k_heads=int(v("ssm.group_count")),
        n_v_heads=int(v("ssm.time_step_rank")), d_inner=int(v("ssm.inner_size")),
        full_interval=int(v("full_attention_interval", 4)), n_mtp=n_mtp,
        max_seq_len=min(max_seq_len, int(v("context_length", max_seq_len))),
        n_experts=int(v("expert_count", 0) or 0), n_experts_used=int(v("expert_used_count", 0) or 0),
        expert_ffn=int(v("expert_feed_forward_length", 0) or 0),
        shared_ffn=int(v("expert_shared_feed_forward_length", 0) or 0),
        expert_weights_scale=float(v("expert_weights_scale", 0.0) or 0.0))


# --------------------------------------------------------------------------- cache
class Qwen35Cache:
    """KV for the attention layers, conv + DeltaNet states for the recurrent ones.

    Speculative decoding checks k guesses in one pass, then keeps only some of them.
    A KV cache just rewinds its position, but a recurrent state has already absorbed
    every token; so a multi-token pass records the state after each token
    (`record = True`) and `rollback` restores the one after the last kept token."""

    def __init__(self, cfg: Qwen35Config, batch: int, max_len: int, device, dtype):
        self.cfg, self.max_len, self.pos = cfg, max_len, 0
        self.k, self.v, self.conv, self.ssm = {}, {}, {}, {}
        for i in range(cfg.n_layers):
            if cfg.recurrent(i):
                self.conv[i] = torch.zeros(batch, cfg.conv_dim, cfg.conv_kernel - 1, device=device, dtype=torch.float32)
                self.ssm[i] = torch.zeros(batch, cfg.n_v_heads, cfg.v_head_dim, cfg.k_head_dim, device=device,
                                          dtype=torch.float32)
            else:
                shape = (batch, cfg.n_kv_heads, max_len, cfg.head_dim)
                self.k[i] = torch.zeros(shape, device=device, dtype=dtype)
                self.v[i] = torch.zeros(shape, device=device, dtype=dtype)
        self.record = False
        self.history: dict[int, list] = {}

    def update(self, layer, k_new, v_new):
        T = k_new.shape[2]
        end = self.pos + T
        if end > self.max_len:
            raise ValueError(f"KV cache full ({self.max_len} tokens)")
        self.k[layer][:, :, self.pos:end] = k_new.to(self.k[layer].dtype)
        self.v[layer][:, :, self.pos:end] = v_new.to(self.v[layer].dtype)
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def advance(self, n):
        self.pos += n

    def reset(self):
        self.pos = 0
        for t in list(self.conv.values()) + list(self.ssm.values()):
            t.zero_()

    def rollback(self, start: int, n_keep: int) -> None:
        """After a recorded multi-token pass that began at `start`: keep its first
        n_keep tokens (KV entries past them are just overwritten later)."""
        for i, states in self.history.items():
            conv, ssm = states[n_keep - 1]
            self.conv[i].copy_(conv)
            self.ssm[i].copy_(ssm)
        self.history = {}
        self.pos = start + n_keep

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for d in (self.k, self.v, self.conv, self.ssm) for t in d.values())


# --------------------------------------------------------------------------- layers
def _rope(x, cos, sin, n_rot):
    """NEOX RoPE on the first n_rot channels of x [B, H, T, hd]; the rest pass through."""
    r, rest = x[..., :n_rot].float(), x[..., n_rot:]
    x1, x2 = r.chunk(2, dim=-1)
    rot = torch.cat([-x2, x1], dim=-1)
    return torch.cat([(r * cos + rot * sin).to(x.dtype), rest], dim=-1)


class GatedAttention(nn.Module):
    def __init__(self, cfg: Qwen35Config, layer_idx: int):
        super().__init__()
        self.cfg, self.layer_idx = cfg, layer_idx
        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)

    def forward(self, x, cos, sin, cache=None):
        c = self.cfg
        B, T, _ = x.shape
        qg = self.q_proj(x).view(B, T, c.n_heads, 2 * c.head_dim)
        qv, gate = qg[..., :c.head_dim], qg[..., c.head_dim:]
        qv = self.q_norm(qv).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, c.n_kv_heads, c.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        qv, k = _rope(qv, cos, sin, c.n_rot), _rope(k, cos, sin, c.n_rot)
        offset = 0
        if cache is not None:
            offset = cache.pos
            k, v = cache.update(self.layer_idx, k, v)
        S = k.shape[2]
        if T == 1:
            mask = None
        else:
            qpos = torch.arange(T, device=x.device)[:, None] + offset
            mask = torch.arange(S, device=x.device)[None, :] <= qpos
        y = F.scaled_dot_product_attention(qv, k.to(qv.dtype), v.to(qv.dtype), attn_mask=mask,
                                           enable_gqa=c.n_kv_heads != c.n_heads)
        y = y.transpose(1, 2).reshape(B, T, c.n_heads * c.head_dim)
        return self.o_proj(y * torch.sigmoid(gate.reshape(B, T, -1)))


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: Qwen35Config, layer_idx: int):
        super().__init__()
        self.cfg, self.layer_idx = cfg, layer_idx
        self.norm = RMSNorm(cfg.v_head_dim, cfg.norm_eps)

    def forward(self, x, cache=None):
        c = self.cfg
        B, T, _ = x.shape
        qkv = self.qkv_proj(x).float()                     # [B, T, conv_dim]
        z = self.gate_proj(x)                               # [B, T, d_inner]
        beta = torch.sigmoid(self.beta_proj(x).float())    # [B, T, Hv]
        g = F.softplus(self.alpha_proj(x).float() + self.dt_bias) * self.a   # [B, T, Hv], < 0
        # Causal depthwise convolution over [the last K-1 inputs, this call's T inputs].
        prev = cache.conv[self.layer_idx] if cache is not None else qkv.new_zeros(B, c.conv_dim, c.conv_kernel - 1)
        inp = torch.cat([prev, qkv.transpose(1, 2)], dim=2)                       # [B, C, K-1+T]
        conv = F.silu(F.conv1d(inp, self.conv_w[:, None, :], groups=c.conv_dim))  # [B, C, T]
        kd, nk, nv = c.n_k_heads * c.k_head_dim, c.n_k_heads, c.n_v_heads
        qh = conv[:, :kd].transpose(1, 2).reshape(B, T, nk, c.k_head_dim)
        kh = conv[:, kd:2 * kd].transpose(1, 2).reshape(B, T, nk, c.k_head_dim)
        vh = conv[:, 2 * kd:].transpose(1, 2).reshape(B, T, nv, c.v_head_dim)
        eps = c.norm_eps
        qh = qh / qh.norm(dim=-1, keepdim=True).clamp_min(eps) / math.sqrt(c.k_head_dim)
        kh = kh / kh.norm(dim=-1, keepdim=True).clamp_min(eps)
        if nv != nk:  # tile key heads to the value heads (ggml_repeat): value head h uses key head h % nk
            qh, kh = qh.repeat(1, 1, nv // nk, 1), kh.repeat(1, 1, nv // nk, 1)
        M = cache.ssm[self.layer_idx].clone() if cache is not None else qkv.new_zeros(B, nv, c.v_head_dim, c.k_head_dim)
        outs, record = [], cache is not None and cache.record and T > 1
        hist = []
        for t in range(T):
            M = M * torch.exp(g[:, t])[:, :, None, None]
            kt = kh[:, t]
            d = (vh[:, t] - torch.einsum("bhij,bhj->bhi", M, kt)) * beta[:, t, :, None]
            M = M + d[:, :, :, None] * kt[:, :, None, :]
            outs.append(torch.einsum("bhij,bhj->bhi", M, qh[:, t]))
            if record:
                hist.append((inp[:, :, t + 1:t + c.conv_kernel].clone(), M.clone()))
        if cache is not None:
            cache.conv[self.layer_idx].copy_(inp[:, :, -(c.conv_kernel - 1):])
            cache.ssm[self.layer_idx].copy_(M)
            if record:
                cache.history[self.layer_idx] = hist
        o = torch.stack(outs, dim=1)                                               # [B, T, Hv, dv]
        o = self.norm(o) * F.silu(z.float().view(B, T, nv, c.v_head_dim))
        return self.out_proj(o.reshape(B, T, c.d_inner).to(x.dtype))


class SwiGLU(nn.Module):
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SharedExpertMoE(nn.Module):
    """qwen35moe's FFN: routed experts + sigmoid(gate . x) * shared SwiGLU expert."""

    def __init__(self, moe: MoE, shared: SwiGLU, shared_gate: torch.Tensor):
        super().__init__()
        self.moe, self.shared = moe, shared
        self.register_buffer("shared_gate", shared_gate, persistent=False)   # [d]: one number per token

    def forward(self, x):
        gate = torch.sigmoid(x.float() @ self.shared_gate.float()).unsqueeze(-1)
        return self.moe(x) + (self.shared(x).float() * gate).to(x.dtype)


class Block(nn.Module):
    def __init__(self, cfg: Qwen35Config, i: int):
        super().__init__()
        self.recurrent = cfg.recurrent(i)
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.post_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mixer = GatedDeltaNet(cfg, i) if self.recurrent else GatedAttention(cfg, i)
        self.mlp = SwiGLU()

    def forward(self, x, cos, sin, cache=None):
        a = self.mixer(self.attn_norm(x), cache) if self.recurrent else self.mixer(self.attn_norm(x), cos, sin, cache)
        h = x + a
        return h + self.mlp(self.post_norm(h))


def _rope_tables(cfg: Qwen35Config, device) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.n_rot, 2, dtype=torch.float32) / cfg.n_rot))
    ang = torch.outer(torch.arange(cfg.max_seq_len, dtype=torch.float32), inv)
    ang = torch.cat([ang, ang], dim=-1)
    return ang.cos().to(device), ang.sin().to(device)


class Qwen35Model(nn.Module):
    """Same calling convention as forge.model.Transformer (logits, optional hidden)."""

    def __init__(self, cfg: Qwen35Config):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)

    def new_cache(self, batch: int, max_len: int, dtype) -> Qwen35Cache:
        return Qwen35Cache(self.cfg, batch, max_len, self.rope_cos.device, dtype)

    def forward(self, idx, targets=None, cache=None, return_hidden: bool = False, **_):
        B, T = idx.shape
        offset = cache.pos if cache is not None else 0
        if offset + T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {offset + T} exceeds max_seq_len {self.cfg.max_seq_len}")
        cos, sin = self.rope_cos[offset:offset + T], self.rope_sin[offset:offset + T]
        x = self.tok_emb(idx).to(self.norm_f.weight.dtype)
        for b in self.blocks:
            x = b(x, cos, sin, cache)
        if cache is not None:
            cache.advance(T)
        h = self.norm_f(x)
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return ModelOutput(logits, loss, h if return_hidden else None)

    def num_params(self, exclude_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        n += sum({id(m.raw): m.n_weights for m in self.modules()
                  if isinstance(m, (QuantLinear, QuantEmbedding)) or getattr(m, "ggml_type", None) is not None
                  and isinstance(m, ExpertWeights)}.values())
        return n


# --------------------------------------------------------------------------- loading
def _linear(g, name, device, dtype, keep_quantized: bool) -> nn.Module:
    info = g.tensors[name]
    out_f, in_f = info.shape
    if keep_quantized and info.ggml_type not in PLAIN:
        lin = QuantLinear(in_f, out_f, info.ggml_type, device=device)
        with torch.no_grad():
            lin.raw.copy_(g.raw(name).to(device))
        return lin
    lin = nn.Linear(in_f, out_f, bias=False, device=device, dtype=dtype)
    with torch.no_grad():
        lin.weight.copy_(g.tensor(name, dtype=torch.float32, device=device).to(dtype))
    return lin


def _vec(g, name, device, dtype=torch.float32):
    return g.tensor(name, dtype=torch.float32, device=device).to(dtype)


def _set_norm(norm: RMSNorm, g, name, device, dtype):
    with torch.no_grad():
        norm.weight = nn.Parameter(_vec(g, name, device, dtype))


def _load_attention(att: GatedAttention, g, p: str, device, dtype, kq: bool):
    att.q_proj = _linear(g, p + "attn_q.weight", device, dtype, kq)
    att.k_proj = _linear(g, p + "attn_k.weight", device, dtype, kq)
    att.v_proj = _linear(g, p + "attn_v.weight", device, dtype, kq)
    att.o_proj = _linear(g, p + "attn_output.weight", device, dtype, kq)
    _set_norm(att.q_norm, g, p + "attn_q_norm.weight", device, dtype)
    _set_norm(att.k_norm, g, p + "attn_k_norm.weight", device, dtype)


def _swiglu(g, p, device, dtype, kq, suffix="") -> SwiGLU:
    mlp = SwiGLU()
    mlp.gate = _linear(g, f"{p}ffn_gate{suffix}.weight", device, dtype, kq)
    mlp.up = _linear(g, f"{p}ffn_up{suffix}.weight", device, dtype, kq)
    mlp.down = _linear(g, f"{p}ffn_down{suffix}.weight", device, dtype, kq)
    return mlp


def _experts(g, name, device, dtype, kq) -> ExpertWeights:
    """A [n_experts, out, in] tensor, kept as the file's blocks when quantized."""
    info = g.tensors[name]
    E, out_f, in_f = info.shape
    quantized = kq and info.ggml_type not in PLAIN
    if quantized:
        ex = ExpertWeights(E, in_f, out_f, info.ggml_type).to(device)
        with torch.no_grad():
            ex.raw.copy_(g.raw(name).to(device))
        return ex
    with torch.device("meta"):
        ex = ExpertWeights(E, in_f, out_f)
    ex.weight = nn.Parameter(g.tensor(name, dtype=torch.float32, device=device).to(dtype))
    return ex


def _ffn(cfg: Qwen35Config, g, p, device, dtype, kq) -> nn.Module:
    """The block's FFN: dense SwiGLU, or routed experts + gated shared expert."""
    if p + "ffn_gate_inp.weight" not in g.tensors:
        return _swiglu(g, p, device, dtype, kq)
    unsupported = [n for n in g.tensors if n.startswith(p) and n.endswith(".scale")]
    if unsupported:
        raise NotImplementedError(f"per-tensor scales ({unsupported[0]}) aren't supported for qwen35moe yet")
    merged = p + "ffn_gate_up_exps.weight" in g.tensors
    hidden = cfg.expert_ffn or g.tensors[p + "ffn_down_exps.weight"].shape[2]
    with torch.device("meta"):   # shapes only: every weight is replaced just below
        moe = MoE(cfg.d_model, cfg.n_experts, cfg.n_experts_used, hidden, act="silu", norm_weights=True,
                  weights_scale=cfg.expert_weights_scale, merged_gate_up=merged)
    moe.counts = torch.zeros(cfg.n_experts, dtype=torch.long, device=device)
    moe.router = _linear(g, p + "ffn_gate_inp.weight", device, torch.float32, False)  # small; routing in fp32
    if merged:
        moe.gate_up = _experts(g, p + "ffn_gate_up_exps.weight", device, dtype, kq)
    else:
        moe.gate = _experts(g, p + "ffn_gate_exps.weight", device, dtype, kq)
        moe.up = _experts(g, p + "ffn_up_exps.weight", device, dtype, kq)
    moe.down = _experts(g, p + "ffn_down_exps.weight", device, dtype, kq)
    if p + "ffn_up_shexp.weight" not in g.tensors:
        return moe
    shared = _swiglu(g, p, device, dtype, kq, suffix="_shexp")
    return SharedExpertMoE(moe, shared, _vec(g, p + "ffn_gate_inp_shexp.weight", device).reshape(-1))


def _embedding(g, device, dtype, kq):
    info = g.tensors["token_embd.weight"]
    if kq and info.ggml_type not in PLAIN:
        emb = QuantEmbedding(info.shape[0], info.shape[1], info.ggml_type, device=device)
        with torch.no_grad():
            emb.raw.copy_(g.raw("token_embd.weight").to(device))
        emb.out_dtype = dtype
        return emb
    emb = nn.Embedding(info.shape[0], info.shape[1], device=device, dtype=dtype)
    with torch.no_grad():
        emb.weight.copy_(g.tensor("token_embd.weight", dtype=torch.float32, device=device).to(dtype))
    return emb


def _output_layer(g, emb, device, dtype, kq):
    if "output.weight" in g.tensors and not g.same_bytes("output.weight", "token_embd.weight"):
        return _linear(g, "output.weight", device, dtype, kq)
    if isinstance(emb, QuantEmbedding):
        return emb.as_output_layer()
    lin = nn.Linear(emb.weight.shape[1], emb.weight.shape[0], bias=False, device="meta")
    lin.weight = emb.weight  # tied
    return lin


def load_qwen35(path, device="cpu", dtype=torch.bfloat16, max_seq_len: int = 4096,
                keep_quantized: bool | str = "auto") -> tuple[Qwen35Model, dict]:
    from .gguf import GGUFFile
    from .gguf_model import should_keep_quantized

    device = torch.device(device)
    with GGUFFile(path) as g:
        if g.architecture not in ARCHS:
            raise ValueError(f"{path} is a {g.architecture!r} model, not qwen35 / qwen35moe")
        cfg = config_from_gguf(g, max_seq_len)
        if "blk.0.attn_norm.weight" not in g.tensors:
            raise ValueError(f"{path} holds only an MTP block (an MTP file); load the main model instead")
        kq = should_keep_quantized(g, device, dtype) if keep_quantized == "auto" else bool(keep_quantized)
        m = Qwen35Model(cfg).to(device=device, dtype=dtype)
        m.tok_emb = _embedding(g, device, dtype, kq)
        m.lm_head = _output_layer(g, m.tok_emb, device, dtype, kq)
        _set_norm(m.norm_f, g, "output_norm.weight", device, dtype)
        for i, b in enumerate(m.blocks):
            p = f"blk.{i}."
            _set_norm(b.attn_norm, g, p + "attn_norm.weight", device, dtype)
            _set_norm(b.post_norm, g, p + "post_attention_norm.weight", device, dtype)
            b.mlp = _ffn(cfg, g, p, device, dtype, kq)
            if b.recurrent:
                d = b.mixer
                d.qkv_proj = _linear(g, p + "attn_qkv.weight", device, dtype, kq)
                d.gate_proj = _linear(g, p + "attn_gate.weight", device, dtype, kq)
                d.beta_proj = _linear(g, p + "ssm_beta.weight", device, dtype, False)
                d.alpha_proj = _linear(g, p + "ssm_alpha.weight", device, dtype, False)
                d.out_proj = _linear(g, p + "ssm_out.weight", device, dtype, kq)
                d.register_buffer("conv_w", _vec(g, p + "ssm_conv1d.weight", device), persistent=False)
                d.register_buffer("dt_bias", _vec(g, p + "ssm_dt.bias", device), persistent=False)
                d.register_buffer("a", _vec(g, p + "ssm_a", device), persistent=False)
                _set_norm(d.norm, g, p + "ssm_norm.weight", device, torch.float32)
            else:
                _load_attention(b.mixer, g, p, device, dtype, kq)
        cos, sin = _rope_tables(cfg, device)
        m.register_buffer("rope_cos", cos, persistent=False)
        m.register_buffer("rope_sin", sin, persistent=False)
        info = {"format": "gguf", "architecture": g.architecture, "name": g.metadata.get("general.name"),
                "kept_quantized": sum(isinstance(x, QuantLinear) for x in m.modules()), "has_mtp": cfg.n_mtp > 0}
    return m.eval(), info


# --------------------------------------------------------------------------- MTP
class Qwen35MTP(nn.Module):
    """The nextn block (see the module docstring), with its own KV cache."""

    @classmethod
    def load(cls, path, target: Qwen35Model, device=None, dtype=None) -> "Qwen35MTP":
        """From the main GGUF (its last block) or a separate mtp-*.gguf (qwen35, one block)."""
        from .gguf import GGUFFile

        p0 = next(target.parameters())
        device, dtype = device or p0.device, dtype or p0.dtype
        m = cls()
        with GGUFFile(path) as g:
            if g.architecture not in ARCHS:
                raise ValueError(f"{path}: {g.architecture!r}, not a qwen35 / qwen35moe MTP block")
            total, n_mtp = int(g.arch_value("block_count")), int(g.arch_value("nextn_predict_layers", 0) or 0)
            if n_mtp != 1:
                raise NotImplementedError(f"{path}: {n_mtp} MTP blocks; only 1 is supported")
            il = total - n_mtp
            p = f"blk.{il}."
            if p + "nextn.eh_proj.weight" not in g.tensors:
                raise ValueError(f"{path} has no MTP block (blk.{il}.nextn.*)")
            kq = any(g.tensors[n].ggml_type not in PLAIN for n in g.tensors if n.startswith(p))
            cfg = target.cfg
            m.cfg = cfg
            m.eh_proj = _linear(g, p + "nextn.eh_proj.weight", device, dtype, kq)
            m.enorm, m.hnorm = RMSNorm(cfg.d_model, cfg.norm_eps), RMSNorm(cfg.d_model, cfg.norm_eps)
            _set_norm(m.enorm, g, p + "nextn.enorm.weight", device, dtype)
            _set_norm(m.hnorm, g, p + "nextn.hnorm.weight", device, dtype)
            m.attn_norm, m.post_norm = RMSNorm(cfg.d_model, cfg.norm_eps), RMSNorm(cfg.d_model, cfg.norm_eps)
            _set_norm(m.attn_norm, g, p + "attn_norm.weight", device, dtype)
            _set_norm(m.post_norm, g, p + "post_attention_norm.weight", device, dtype)
            m.attn = GatedAttention(cfg, 0)
            _load_attention(m.attn, g, p, device, dtype, kq)
            m.mlp = _ffn(cfg, g, p, device, dtype, kq)
            head_norm = p + "nextn.shared_head_norm.weight" if p + "nextn.shared_head_norm.weight" in g.tensors \
                else "output_norm.weight"
            m.head_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
            _set_norm(m.head_norm, g, head_norm, device, dtype)
            if p + "nextn.shared_head_head.weight" in g.tensors:
                m.lm_head = _linear(g, p + "nextn.shared_head_head.weight", device, dtype, kq)
            else:
                m.lm_head = target.lm_head   # the main model's output layer
        m.target = [target]               # (not a submodule: its weights belong to the target)
        return m.eval()

    def new_cache(self, max_len: int, dtype) -> "_MTPCache":
        c = self.cfg
        dev = next(self.parameters()).device
        return _MTPCache(torch.zeros(1, c.n_kv_heads, max_len, c.head_dim, device=dev, dtype=dtype),
                         torch.zeros(1, c.n_kv_heads, max_len, c.head_dim, device=dev, dtype=dtype))

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, h: torch.Tensor, cache: "_MTPCache", pos: int):
        """tokens [T], h [T, d] (the target hidden state before each token) at positions
        pos..pos+T-1 -> logits [T, vocab], h_next [T, d]. Writes those KV positions."""
        target = self.target[0]
        T = tokens.numel()
        e = target.tok_emb(tokens.view(1, T)).to(h.dtype)
        cur = self.eh_proj(torch.cat([self.enorm(e), self.hnorm(h.view(1, T, -1))], dim=-1))
        cache.pos = pos
        cos, sin = target.rope_cos[pos:pos + T], target.rope_sin[pos:pos + T]
        a = self.attn(self.attn_norm(cur), cos, sin, cache)
        cache.pos = pos + T
        x = cur + a
        x = x + self.mlp(self.post_norm(x))
        hn = self.head_norm(x)
        return self.lm_head(hn)[0], hn[0]


class _MTPCache:
    def __init__(self, k, v):
        self.k, self.v, self.pos, self.max_len = {0: k}, {0: v}, 0, k.shape[2]

    def update(self, layer, k_new, v_new):
        T = k_new.shape[2]
        end = self.pos + T
        self.k[0][:, :, self.pos:end] = k_new.to(self.k[0].dtype)
        self.v[0][:, :, self.pos:end] = v_new.to(self.v[0].dtype)
        return self.k[0][:, :, :end], self.v[0][:, :, :end]


class Qwen35MTPDrafter:
    """speculative.py drafter for Qwen3.5's MTP block (llama.cpp speculative.cpp,
    draft-mtp without shared memory). Its KV entry at position p comes from the pair
    (token at p, the target's hidden state at p-1). After each check it "catches
    up": it reruns the accepted tokens with the target's real hidden states, which
    overwrites the entries its own guesses wrote."""

    needs_hidden = True

    def __init__(self, mtp: Qwen35MTP):
        self.mtp = mtp
        self.cache = None

    def begin(self, target, cache, h_last, prompt_ids=None, hidden_rows=None) -> None:
        p0 = next(self.mtp.parameters())
        self.target_cache = cache
        self.cache = self.mtp.new_cache(cache.max_len, p0.dtype)
        ids = torch.tensor(prompt_ids, device=p0.device)
        hs = torch.cat([torch.zeros_like(hidden_rows[:1]), hidden_rows[:-1]])   # h before each prompt token
        self.mtp(ids, hs, self.cache, 0)
        self.h = hidden_rows[-1:].clone()   # pairs with the next token
        self.pending_start = len(prompt_ids)

    def after_verify(self, h_row, hidden_rows=None, inputs=None, n_ok=None) -> None:
        start = self.pending_start
        kept = inputs[:n_ok + 1]
        hs = torch.cat([self.h, hidden_rows[:n_ok]])
        self.mtp(torch.tensor(kept, device=hs.device), hs, self.cache, start)
        self.h = hidden_rows[n_ok:n_ok + 1].clone()
        self.pending_start = start + n_ok + 1

    def __call__(self, ids: list[int], k: int) -> list[int]:
        if k <= 0:
            return []
        n_past = self.target_cache.pos
        tok = torch.tensor([ids[-1]], device=self.h.device)
        h, picks = self.h, []
        for i in range(k):
            logits, h = self.mtp(tok, h, self.cache, n_past + i)
            tok = logits[-1].argmax(-1).view(1)
            picks.append(tok)
        return torch.cat(picks).tolist()


def load_qwen35_drafter(path, target: Qwen35Model) -> Qwen35MTPDrafter:
    return Qwen35MTPDrafter(Qwen35MTP.load(path, target))
