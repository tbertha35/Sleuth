"""MTP ("multi-token prediction") drafters: small extra layers trained with a
model to guess the tokens after the next one. They make the guesses for
speculative decoding (speculative.py), far better than prompt lookup on fresh
text, at the cost of a few small layers per guess.

Gemma 4 assistant (`mtp-gemma-4-*.gguf`, architecture "gemma4-assistant").
Follows llama.cpp's src/models/gemma4-assistant.cpp and the draft-mtp driver in
common/speculative.cpp (commit b2dd28a3b, 2026-07-16), the reference for these files:

  input   the token t just produced, and h = the target model's hidden state
          *after its final norm* at the position before t (gemma4.cpp: t_h_nextn)
  x       = target.tok_emb(t) * sqrt(d_target)            (the target's embedding)
  cur     = pre_projection([x, h])                       2·d_target -> d (1024)
  4 layers, each: x + post_norm(attn(norm(x))), then x + post_norm(geglu(norm(x))),
          then × layer_output_scale. Attention has only a Q projection (+ q_norm,
          RoPE, scale 1.0): it reads the *target's* K/V cache, sliding layers from
          the target's layer n-2, the full layer from layer n-1 (llama-model.cpp,
          the gemma4-assistant `share` callback), for positions before t's.
  output  logits = own_embedding(norm(cur))   (tied, no soft-cap)
          h_next = post_projection(norm(cur))  d -> d_target, the h of the next guess

Every guess of one round sits at the same position (t's): the drafter shares the
target's memory and writes none of its own (speculative.cpp, is_mem_shared).
Guesses are greedy; drafting stops early when the top probability is below p_min.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ...formats import gguf_quants as q
from ..model.norm import RMSNorm
from ..model.quant import QuantLinear
from ..model.rope import apply_rope, rope_tables

PLAIN = {q.F32, q.F16, q.BF16, q.F64}


def _linear(g, name: str, device, dtype) -> nn.Module:
    """A bias-free Linear for a GGUF matrix: kept quantized (QuantLinear, Triton
    kernel on GPU) unless it's stored unquantized."""
    info = g.tensors[name]
    out_f, in_f = info.shape
    if info.ggml_type in PLAIN:
        lin = nn.Linear(in_f, out_f, bias=False, device=device, dtype=dtype)
        with torch.no_grad():
            lin.weight.copy_(g.tensor(name, dtype=torch.float32, device=device).to(dtype))
        return lin
    lin = QuantLinear(in_f, out_f, info.ggml_type, device=device)
    with torch.no_grad():
        lin.raw.copy_(g.raw(name).to(device))
    return lin


def _norm(g, name: str, eps: float, device, dtype) -> RMSNorm:
    w = g.tensor(name, dtype=torch.float32, device=device)
    n = RMSNorm(w.numel(), eps).to(device=device, dtype=dtype)
    with torch.no_grad():
        n.weight.copy_(w.to(dtype))
    return n


class AssistantLayer(nn.Module):
    def __init__(self, sliding: bool, n_heads: int, head_dim: int, window: int, eps: float):
        super().__init__()
        self.sliding, self.n_heads, self.hd, self.window, self.eps = sliding, n_heads, head_dim, window, eps

    def forward(self, x, k, v, cos, sin, n_past: int, mask=None):
        """x [1, T, d]; k, v: the target layer's cache [1, K, >=n_past, hd].
        mask: None (use the first n_past keys, sliced), or a [1, 1, 1, L] bool mask
        over k/v of length L (CUDA graphs: the position is a tensor, not an int)."""
        B, T, _ = x.shape
        h = self.attn_norm(x)
        qv = self.q_proj(h).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        qv = apply_rope(self.q_norm(qv), cos, sin)
        if mask is None:
            k, v = k[:, :, :n_past], v[:, :, :n_past]
            if self.sliding and n_past > self.window - 1:
                # The query sits at position n_past: llama.cpp masks keys with q - k >= window.
                k, v = k[:, :, n_past - (self.window - 1):], v[:, :, n_past - (self.window - 1):]
        y = F.scaled_dot_product_attention(qv, k.to(qv.dtype), v.to(qv.dtype), attn_mask=mask, scale=1.0,
                                           enable_gqa=k.shape[1] != self.n_heads)
        a = self.o_proj(y.transpose(1, 2).reshape(B, T, self.n_heads * self.hd))
        x = x + self.post_attn_norm(a)
        m = self.ffn_down(F.gelu(self.ffn_gate(self.ffn_norm(x)), approximate="tanh") * self.ffn_up(self.ffn_norm(x)))
        x = x + self.post_ffn_norm(m)
        return x * self.scale.to(x.dtype)


class Gemma4Assistant(nn.Module):
    """The gemma4-assistant drafter (see the module docstring)."""

    @classmethod
    def load(cls, path: str | Path, device="cpu", dtype=torch.bfloat16, max_seq_len: int = 4096) -> "Gemma4Assistant":
        from ...formats.gguf import GGUFFile

        m = cls()
        device = torch.device(device)
        with GGUFFile(path) as g:
            if g.architecture != "gemma4-assistant":
                raise ValueError(f"{path} is a {g.architecture!r} file, not a gemma4-assistant (MTP) drafter")
            L = int(g.arch_value("block_count"))
            eps = float(g.arch_value("attention.layer_norm_rms_epsilon", 1e-6))
            pattern = g.arch_value("attention.sliding_window_pattern")
            m.n_heads = int(g.arch_value("attention.head_count"))
            m.d = int(g.arch_value("embedding_length"))
            m.d_target = int(g.arch_value("embedding_length_out"))
            m.window = int(g.arch_value("attention.sliding_window"))
            hd = {"full": int(g.arch_value("attention.key_length")), "sliding": int(g.arch_value("attention.key_length_swa"))}
            theta = {"full": float(g.arch_value("rope.freq_base")), "sliding": float(g.arch_value("rope.freq_base_swa"))}
            factors = g.tensor("rope_freqs.weight", dtype=torch.float32) if "rope_freqs.weight" in g.tensors else None
            for kind in ("full", "sliding"):
                cos, sin = rope_tables(hd[kind], max_seq_len, theta[kind], factors if kind == "full" else None)
                m.register_buffer(f"cos_{kind}", cos.to(device), persistent=False)
                m.register_buffer(f"sin_{kind}", sin.to(device), persistent=False)
            m.pre_proj = _linear(g, "nextn.pre_projection.weight", device, dtype)
            m.post_proj = _linear(g, "nextn.post_projection.weight", device, dtype)
            m.out_norm = _norm(g, "output_norm.weight", eps, device, dtype)
            m.lm_head = _linear(g, "token_embd.weight", device, dtype)   # tied: the output layer is the embedding
            m.layers = nn.ModuleList()
            for i in range(L):
                sliding = bool(pattern[i])
                lay = AssistantLayer(sliding, m.n_heads, hd["sliding" if sliding else "full"], m.window, eps)
                p = f"blk.{i}."
                lay.attn_norm = _norm(g, p + "attn_norm.weight", eps, device, dtype)
                lay.q_proj = _linear(g, p + "attn_q.weight", device, dtype)
                lay.q_norm = _norm(g, p + "attn_q_norm.weight", eps, device, dtype)
                lay.o_proj = _linear(g, p + "attn_output.weight", device, dtype)
                lay.post_attn_norm = _norm(g, p + "post_attention_norm.weight", eps, device, dtype)
                lay.ffn_norm = _norm(g, p + "ffn_norm.weight", eps, device, dtype)
                lay.ffn_gate = _linear(g, p + "ffn_gate.weight", device, dtype)
                lay.ffn_up = _linear(g, p + "ffn_up.weight", device, dtype)
                lay.ffn_down = _linear(g, p + "ffn_down.weight", device, dtype)
                lay.post_ffn_norm = _norm(g, p + "post_ffw_norm.weight", eps, device, dtype)
                lay.register_buffer("scale", g.tensor(p + "layer_output_scale.weight", dtype=torch.float32)
                                    .reshape(1).to(device), persistent=False)
                m.layers.append(lay)
        return m.eval()

    def check_target(self, target) -> None:
        """The pairing llama.cpp enforces: the drafter's input width is the target's
        hidden size, and its layers line up with the target's last two layers."""
        cfg = target.cfg
        if cfg.d_model != self.d_target:
            raise ValueError(f"MTP drafter expects a target with d_model {self.d_target}, not {cfg.d_model}")
        if cfg.layer_type(cfg.n_layers - 2) != "sliding" or cfg.layer_type(cfg.n_layers - 1) != "full":
            raise ValueError("MTP drafter needs a target whose last two layers are sliding, then full")

    @torch.no_grad()
    def forward(self, target, cache, token: torch.Tensor, h: torch.Tensor, n_past: int):
        """token [1] (int64), h [1, d_target] -> logits [vocab], h_next [1, d_target]."""
        tc = target.cfg
        x = target.tok_emb(token.view(1, 1)) * tc.embed_scale
        cur = self.pre_proj(torch.cat([x.to(h.dtype), h.view(1, 1, -1)], dim=-1))
        for lay in self.layers:
            layer = tc.n_layers - 2 if lay.sliding else tc.n_layers - 1
            kind = "sliding" if lay.sliding else "full"
            cos = getattr(self, f"cos_{kind}")[n_past:n_past + 1]
            sin = getattr(self, f"sin_{kind}")[n_past:n_past + 1]
            cur = lay(cur, cache.k[layer], cache.v[layer], cos, sin, n_past)
        cur = self.out_norm(cur)
        return self.lm_head(cur)[0, -1], self.post_proj(cur)[0]

    def forward_static(self, target, cache, token: torch.Tensor, h: torch.Tensor, pos: torch.Tensor, length: int):
        """forward() with the position as a GPU tensor `pos` [1] and a fixed cache view
        of `length` positions (masked to the keys before pos), for CUDA graphs."""
        tc = target.cfg
        x = target.tok_emb(token.view(1, 1)) * tc.embed_scale
        cur = self.pre_proj(torch.cat([x.to(h.dtype), h.view(1, 1, -1)], dim=-1))
        kpos = torch.arange(length, device=pos.device)
        before = kpos < pos                                    # only what the target has stored
        masks = {False: before.view(1, 1, 1, -1),
                 True: (before & (pos - kpos < self.window)).view(1, 1, 1, -1)}
        for lay in self.layers:
            layer = tc.n_layers - 2 if lay.sliding else tc.n_layers - 1
            kind = "sliding" if lay.sliding else "full"
            cos = getattr(self, f"cos_{kind}").index_select(0, pos)
            sin = getattr(self, f"sin_{kind}").index_select(0, pos)
            k = cache.k[layer][:, :, :length]
            v = cache.v[layer][:, :, :length]
            cur = lay(cur, k, v, cos, sin, 0, mask=masks[lay.sliding])
        cur = self.out_norm(cur)
        return self.lm_head(cur)[0, -1], self.post_proj(cur)[0]


class _DraftGraphs:
    """The drafter's whole chain of k guesses as one CUDA graph per cache length
    (the same idea as cuda_graph.GraphDecoder): inputs are copied into fixed
    buffers, each guess's argmax feeds the next guess on the GPU, and one replay
    returns all k tokens."""

    def __init__(self, mtp: "Gemma4Assistant", target, cache, k: int):
        from .cuda_graph import _buckets

        dev = next(mtp.parameters()).device
        self.mtp, self.target, self.cache, self.k = mtp, target, cache, k
        self.token = torch.zeros(1, dtype=torch.long, device=dev)
        self.h = torch.zeros(1, mtp.d_target, dtype=next(mtp.parameters()).dtype, device=dev)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.buckets = _buckets(cache.max_len)
        self.graphs: dict[int, tuple] = {}
        self.pool = torch.cuda.graph_pool_handle()

    def _chain(self, length: int) -> torch.Tensor:
        tok, h, picks = self.token, self.h, []
        for _ in range(self.k):
            logits, h = self.mtp.forward_static(self.target, self.cache, tok, h, self.pos, length)
            tok = logits.argmax(-1).view(1)
            picks.append(tok)
        return torch.cat(picks)

    @torch.no_grad()
    def __call__(self, token: int, h: torch.Tensor, n_past: int) -> list[int]:
        length = next(b for b in self.buckets if b >= n_past)
        self.token.fill_(token)
        self.h.copy_(h.view(1, -1))
        self.pos.fill_(n_past)
        if length not in self.graphs:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._chain(length)  # warm-up (Triton autotuning); reads only, writes nothing
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                out = self._chain(length)
            self.graphs[length] = (g, out)
        g, out = self.graphs[length]
        g.replay()
        return out.tolist()


def find_mtp_file(model_path: str | Path) -> Path | None:
    """Where a GGUF model's MTP drafter is: a separate `mtp-*.gguf` in the same
    folder (Gemma 4, Qwen3.8-27B), or the model file itself when its architecture
    keeps the MTP block inside (qwen35 with nextn_predict_layers > 0)."""
    p = Path(model_path)
    if p.suffix.lower() != ".gguf":
        return None
    found = sorted(p.parent.glob("mtp-*.gguf"))
    if len(found) == 1:
        return found[0]
    from ...formats.gguf import GGUFFile

    try:
        with GGUFFile(p) as g:
            if g.architecture in ("qwen35", "qwen35moe") and int(g.arch_value("nextn_predict_layers", 0) or 0) > 0:
                return p
    except (OSError, ValueError):
        pass
    return None


def load_drafter(path: str | Path, target, p_min: float = 0.0):
    """An MTP drafter for `target` from a drafter GGUF (on the target's device):
    Gemma 4's assistant, or Qwen3.5's nextn block."""
    from ...formats.gguf import GGUFFile

    with GGUFFile(path) as g:
        arch = g.architecture
    if arch in ("qwen35", "qwen35moe"):
        from ...formats.qwen35 import load_qwen35_drafter

        return load_qwen35_drafter(path, target)
    p = next(target.parameters())
    dtype = torch.bfloat16 if p.device.type == "cuda" else torch.float32
    mtp = Gemma4Assistant.load(path, device=p.device, dtype=dtype, max_seq_len=target.cfg.max_seq_len)
    mtp.check_target(target)
    return MTPDrafter(mtp, p_min)


class MTPDrafter:
    """The speculative.py drafter interface for an MTP model: it needs the target's
    hidden states, which speculative_generate hands over (begin / after_verify)."""

    needs_hidden = True

    def __init__(self, mtp: Gemma4Assistant, p_min: float = 0.0, cuda_graph: bool = True):
        self.mtp, self.p_min, self.cuda_graph = mtp, p_min, cuda_graph
        self.target = self.cache = self.h = None
        self._graphs: _DraftGraphs | None = None

    def begin(self, target, cache, h_last: torch.Tensor, **_) -> None:
        self.mtp.check_target(target)
        self.target, self.cache, self.h = target, cache, h_last.view(1, -1)

    def after_verify(self, h_row: torch.Tensor, **_) -> None:
        """The target's hidden state at the last accepted input row."""
        self.h = h_row.view(1, -1)

    def __call__(self, ids: list[int], k: int) -> list[int]:
        out: list[int] = []
        if k <= 0:
            return out
        n_past = self.cache.pos  # where ids[-1] will go; the target's cache holds the positions before it
        if self.cuda_graph and self.p_min <= 0 and self.h.is_cuda:
            # One graph replay for all k guesses. A graph is tied to the cache buffers it
            # was recorded with, so record again if the target's cache is a new one.
            key = (self.cache.k[-1].data_ptr(), k)
            if self._graphs is None or self._graphs_key != key:
                self._graphs, self._graphs_key = _DraftGraphs(self.mtp, self.target, self.cache, k), key
            return self._graphs(ids[-1], self.h, n_past)
        tok = torch.tensor([ids[-1]], device=self.h.device)
        h = self.h
        if self.p_min <= 0:
            # No confidence cut-off: keep every guess on the GPU and read them all at
            # the end (one sync per round instead of one per guess).
            picks = []
            for _ in range(k):
                logits, h = self.mtp(self.target, self.cache, tok, h, n_past)
                tok = logits.argmax(-1).view(1)
                picks.append(tok)
            return torch.cat(picks).tolist()
        for _ in range(k):
            logits, h = self.mtp(self.target, self.cache, tok, h, n_past)
            probs = logits.float().softmax(-1)
            p, t = probs.max(-1)
            if float(p) < self.p_min:
                break
            out.append(int(t))
            tok = t.view(1)
        return out


class HeadsDrafter:
    """The drafter for our own models' MTP heads (forge/model/mtp_heads.py). Guess j
    comes from head j (the last head again past the number of heads), chained as in
    training: h_j = head_j(h_{j-1}, embed(previous guess)). The heads keep no cache,
    so there is nothing to catch up after a check: only the target's hidden state."""

    needs_hidden = True

    def __init__(self, model):
        if getattr(model, "mtp", None) is None:
            raise ValueError("this model has no MTP heads (train it with model.mtp_heads > 0)")
        self.model, self.h = model, None
        self.vocab = model.cfg.vocab_size or model.cfg.padded_vocab   # never guess a padding row

    def begin(self, target, cache, h_last: torch.Tensor, **_) -> None:
        self.h = h_last.view(1, -1)

    def after_verify(self, h_row: torch.Tensor, **_) -> None:
        self.h = h_row.view(1, -1)

    @torch.no_grad()
    def __call__(self, ids: list[int], k: int) -> list[int]:
        if k <= 0:
            return []
        m = self.model
        tok = torch.tensor([ids[-1]], device=self.h.device)
        h, picks = self.h, []
        for j in range(k):
            head = m.mtp[min(j, len(m.mtp) - 1)]
            h = head(h, m.embed(tok).to(h.dtype))
            tok = m.lm_head(h)[..., :self.vocab].argmax(-1).view(1)
            picks.append(tok)
        return torch.cat(picks).tolist()   # one sync for all k guesses


def builtin_drafter(model) -> HeadsDrafter | None:
    """The drafter a model carries itself (our runs trained with MTP heads), if any."""
    return HeadsDrafter(model) if getattr(model, "mtp", None) is not None else None
