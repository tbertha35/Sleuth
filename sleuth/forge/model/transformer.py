"""The decoder-only Transformer.

    tokens -> embedding -> [Block] x n_layers -> final norm -> lm_head -> logits

Each Block is "pre-norm":
    h = x + Attention(Norm(x))     (tokens exchange information)
    y = h + MLP(Norm(h))           (each token processed on its own)

The running sum x -> h -> y is the *residual stream*. Every layer reads
from it and adds its contribution back. Most interpretability tools
(logit lens, ablation, SAEs) are built around this stream, which is why it
has taps at every step.

Gemma-family extras (all off unless the config turns them on):
  * the embedding is multiplied by embed_scale (√d_model), so it starts at a
    size comparable to what the layers add
  * post_norms: each block output is normalized again before it's added
    ("sandwich" norm):  h = x + Norm(Attention(Norm(x)))
  * layer_scale: a learned scalar multiplies the whole stream after the block
  * final_softcap: logits = cap · tanh(logits / cap), so no logit exceeds ±cap
  * sliding and full layers use different RoPE tables (see ModelConfig)
  * MoE (n_experts > 0, Gemma 4 26B-A4B; llama.cpp gemma4.cpp): the MLP output
    becomes  post_norm_1(MLP(norm2(x))) + post_norm_2(MoE(pre_norm_2(x)))  before
    post_mlp_norm. The router sees  rms(x) / sqrt(d) * router_scale  (moe.py)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .attention import Attention
from .config import ModelConfig
from .mlp import make_mlp
from .moe import ExpertWeights, MoE
from .memory import NgramMemory
from .mtp_heads import MTPHead, mtp_loss
from .norm import make_norm
from .quant import QuantEmbedding, QuantLinear, quantized_modules
from .rope import rope_tables
from .taps import Tap, assign_tap_names


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.kind = cfg.layer_type(layer_idx)  # which RoPE table this layer uses
        self.norm1 = make_norm(cfg.norm_type, cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, layer_idx)
        self.norm2 = make_norm(cfg.norm_type, cfg.d_model, cfg.norm_eps)
        self.mlp = make_mlp(cfg)
        self.post_attn_norm = make_norm(cfg.norm_type, cfg.d_model, cfg.norm_eps) if cfg.post_norms else None
        self.post_mlp_norm = make_norm(cfg.norm_type, cfg.d_model, cfg.norm_eps) if cfg.post_norms else None
        self.scale = nn.Parameter(torch.ones(1)) if cfg.layer_scale else None
        self.moe = None
        if cfg.n_experts:
            self.moe = MoE(cfg.d_model, cfg.n_experts, cfg.n_experts_used, cfg.expert_hidden,
                           act="gelu" if cfg.ffn_type == "geglu" else "silu",
                           merged_gate_up=cfg.expert_gate_up_merged, down_scale=cfg.expert_down_scale)
            self.router_scale = nn.Parameter(torch.ones(cfg.d_model))
            self.post_mlp_norm1 = make_norm(cfg.norm_type, cfg.d_model, cfg.norm_eps)
            self.pre_moe_norm = make_norm(cfg.norm_type, cfg.d_model, cfg.norm_eps)
            self.post_moe_norm = make_norm(cfg.norm_type, cfg.d_model, cfg.norm_eps)
            self.norm_eps = cfg.norm_eps
        self.tap_attn_out = Tap()
        self.tap_resid_mid = Tap()
        self.tap_mlp_out = Tap()
        self.tap_resid_post = Tap()
        self.parallel = cfg.parallel_residual

    def forward(self, x, cos, sin, cache=None, same_doc=None):
        # attn_out / mlp_out tap exactly what gets added to the stream (after
        # the post-norm, if any), so ablating them removes the whole contribution.
        a = self.attn(self.norm1(x), cos, sin, cache, same_doc)
        if self.post_attn_norm is not None:
            a = self.post_attn_norm(a)
        if self.parallel:
            # GPT-NeoX: the MLP reads the block's input, not the stream after attention.
            # resid_mid then is only a view (input + attention), nothing reads it.
            m = self.mlp(self.norm2(x))
            x = self.tap_resid_mid(x + self.tap_attn_out(a))
            return self.tap_resid_post(x + self.tap_mlp_out(m))
        x = self.tap_resid_mid(x + self.tap_attn_out(a))
        m = self.mlp(self.norm2(x))
        if self.moe is not None:
            # The router reads the stream RMS-normalized without a gain, then scaled per channel.
            d = x.shape[-1]
            r = F.rms_norm(x.float(), (d,), eps=self.norm_eps) * (self.router_scale.float() / math.sqrt(d))
            m = self.post_mlp_norm1(m) + self.post_moe_norm(self.moe(self.pre_moe_norm(x), router_input=r))
        if self.post_mlp_norm is not None:
            m = self.post_mlp_norm(m)
        x = x + self.tap_mlp_out(m)
        if self.scale is not None:
            x = x * self.scale.to(x.dtype)
        return self.tap_resid_post(x)


@dataclass
class ModelOutput:
    logits: torch.Tensor                 # [B, T, padded_vocab]
    loss: torch.Tensor | None = None     # mean cross-entropy over targets (nats/token)
    hidden: torch.Tensor | None = None   # [B, T, d_model] after the final norm (return_hidden=True; MTP drafters)
    mtp_loss: torch.Tensor | None = None  # mean loss of the MTP heads (with targets, if the model has heads)
    exit_loss: torch.Tensor | None = None  # next-token loss of the early exit (with targets, cfg.exit_layer > 0)


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.padded_vocab, cfg.d_model)
        # GPT-2: a learned vector per position, added to the token embedding (no RoPE).
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model) if cfg.pos_type == "learned" else None
        self.tap_embed = Tap()
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm_f = make_norm(cfg.norm_type, cfg.d_model, cfg.norm_eps)
        self.tap_final_norm = Tap()
        self.lm_head = nn.Linear(cfg.d_model, cfg.padded_vocab, bias=False)
        self.mtp = nn.ModuleList(MTPHead(cfg) for _ in range(cfg.mtp_heads)) if cfg.mtp_heads else None
        # Early exit: its own norm (the residual stream mid-way has another scale than
        # at the end), then the shared output layer.
        self.exit_norm = make_norm(cfg.norm_type, cfg.d_model, cfg.norm_eps) if cfg.exit_layer else None
        self.memory = (NgramMemory(cfg.d_model, cfg.mem_rows, cfg.mem_dim, cfg.mem_orders, cfg.mem_heads,
                                   cfg.mem_layers) if cfg.mem_rows else None)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self._tie_shared_layers()

        # persistent=False: derived from the config, so not saved in checkpoints.
        # One cos/sin table per layer kind ("full", and "sliding" if used).
        self.rope_freq_factors: dict[str, torch.Tensor] = {}
        for kind, (cos, sin) in self._rope().items():
            self.register_buffer(f"rope_cos_{kind}", cos, persistent=False)
            self.register_buffer(f"rope_sin_{kind}", sin, persistent=False)

        if torch.empty(0).device.type == "meta":
            return  # building an empty shell (see Transformer.empty); weights come later
        self.apply(self._init_weights)
        # Scale down the projections that write into the residual stream.
        # Each layer adds two such contributions; without scaling, the
        # stream's variance grows with depth (GPT-2 trick).
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down.weight")):
                nn.init.normal_(p, mean=0.0, std=cfg.init_std / math.sqrt(2 * cfg.n_layers))
        assign_tap_names(self)

    def _rope(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        cfg = self.cfg
        out = {}
        for i in range(cfg.n_layers):
            spec = cfg.attn_spec(i)
            if spec.kind in out:
                continue
            if cfg.pos_type == "rope":
                out[spec.kind] = rope_tables(min(cfg.rope_dim or spec.head_dim, spec.head_dim), cfg.max_seq_len,
                                             spec.rope_theta, self.rope_freq_factors.get(spec.kind))
            else:
                z = torch.zeros(cfg.max_seq_len, spec.head_dim)
                out[spec.kind] = (z, z)
        return out

    def _refresh_rope(self) -> None:
        device = self.norm_f.weight.device
        for kind, (cos, sin) in self._rope().items():
            setattr(self, f"rope_cos_{kind}", cos.to(device))
            setattr(self, f"rope_sin_{kind}", sin.to(device))

    def set_rope_freq_factors(self, kind: str, factors: torch.Tensor) -> None:
        """Per-frequency RoPE divisors for one layer kind (Gemma 4's rope_freqs
        tensor, for full layers). They come from the model file, not the config."""
        self.rope_freq_factors[kind] = factors.detach().float().cpu()
        self._refresh_rope()

    @classmethod
    def empty(cls, cfg: ModelConfig, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32,
              quantized: dict[str, int] | None = None) -> "Transformer":
        """A model with uninitialized weights, for loading a checkpoint into.

        Building normally fills every weight with random numbers first. For a
        12B model that takes minutes and needs a full extra copy in RAM. Here
        the model is built on PyTorch's "meta" device (shapes only, no memory),
        then given real but uninitialized storage. Every weight must be loaded
        afterwards.

        quantized: {module path: ggml type}, e.g. {"blocks.0.mlp.up": Q4_0,
        "tok_emb": Q4_0}. Those Linear/Embedding layers become QuantLinear /
        QuantEmbedding (see quant.py) and get storage for the quantized bytes
        only. A quantized, tied embedding also serves as the output layer.
        """
        with torch.device("meta"):
            model = cls(cfg)
            for path, ggml_type in (quantized or {}).items():
                model._make_quantized(path, ggml_type)
        model = model.to_empty(device=device).to(dtype)  # .to(dtype) leaves the uint8 blocks alone
        for m in model.modules():
            if isinstance(m, MoE):
                m.router.float()      # routing in fp32, as llama.cpp does: bf16 logits can pick other experts
                m.reset_counts()      # to_empty left it uninitialized
        if cfg.tie_embeddings:  # to_empty creates separate tensors; re-tie them
            if isinstance(model.tok_emb, QuantEmbedding):
                model.lm_head.raw = model.tok_emb.raw
            else:
                model.lm_head.weight = model.tok_emb.weight
        if isinstance(model.tok_emb, QuantEmbedding):
            model.tok_emb.out_dtype = dtype
        model._tie_shared_layers()  # same for shared (looped) layers
        model._refresh_rope()  # buffers are derived, not loaded: recompute them
        assign_tap_names(model)
        return model

    def _tie_shared_layers(self) -> None:
        """cfg.shared_layers: layer i's Linear weights (and biases) become layer j's, the
        same Parameter objects, so they're stored, trained and counted once and get
        the gradients of every application. Norms stay per layer: a few hundred
        numbers each, and they let each application rescale its input differently."""
        for i, j in enumerate(self.cfg.shared_layers):
            if i == j:
                continue
            src = dict(self.blocks[j].named_modules())
            for name, mod in self.blocks[i].named_modules():
                if isinstance(mod, nn.Linear):
                    mod.weight = src[name].weight
                    if mod.bias is not None:
                        mod.bias = src[name].bias

    def _make_quantized(self, path: str, ggml_type: int) -> None:
        if path == "tok_emb":
            self.tok_emb = QuantEmbedding(self.cfg.padded_vocab, self.cfg.d_model, ggml_type)
            if self.cfg.tie_embeddings:
                self.lm_head = self.tok_emb.as_output_layer()
            return
        if path == "lm_head" and self.cfg.tie_embeddings:
            raise ValueError("a tied lm_head is quantized through tok_emb")
        parent_path, _, attr = path.rpartition(".")
        parent = self.get_submodule(parent_path) if parent_path else self
        old = getattr(parent, attr)
        if isinstance(old, ExpertWeights):
            setattr(parent, attr, ExpertWeights(old.n_experts, old.in_features, old.out_features, ggml_type))
            return
        if not isinstance(old, nn.Linear) or old.bias is not None:
            raise ValueError(f"{path} isn't a bias-free Linear layer; can't keep it quantized")
        setattr(parent, attr, QuantLinear(old.in_features, old.out_features, ggml_type))

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)

    # ------------------------------------------------------------------ #
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, cache=None,
                doc_ids: torch.Tensor | None = None, block_mask=None, return_hidden: bool = False) -> ModelOutput:
        """doc_ids: [B, T] integers, one per position, equal within a document
        (`document_ids` in data/loader.py). Given, a token attends only to
        earlier tokens of its own document ("document masking"); without it, a
        training window that spans two documents lets the second one read the
        first. Positions keep counting across the boundary: RoPE scores depend
        only on the distance between query and key, so a masked document behaves
        the same wherever it starts in the window.

        block_mask: the same, prepared for FlexAttention (`attention.doc_block_mask`),
        used instead of doc_ids."""
        B, T = idx.shape
        offset = cache.pos if cache is not None else 0
        same_doc = block_mask
        if (doc_ids is not None or block_mask is not None) and cache is not None:
            raise ValueError("doc_ids/block_mask are for training/scoring whole windows, not cached decoding")
        if doc_ids is not None and block_mask is None:
            same_doc = (doc_ids[:, :, None] == doc_ids[:, None, :])[:, None]  # [B, 1, T, T], shared by all layers
        if offset + T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {offset + T} exceeds max_seq_len {self.cfg.max_seq_len}")
        if getattr(cache, "static", False):
            # CUDA-graph decoding (infer/kv_cache.py: StaticKVCache): the position is a
            # GPU tensor, so look the RoPE rows up instead of slicing with a Python int.
            cache.new_step()   # its attention masks are built once per pass, shared by all layers
            pos = cache.positions
            rope = {kind: (getattr(self, f"rope_cos_{kind}").index_select(0, pos),
                           getattr(self, f"rope_sin_{kind}").index_select(0, pos)) for kind in self._rope_kinds()}
        else:
            pos = None
            rope = {kind: (getattr(self, f"rope_cos_{kind}")[offset: offset + T],
                           getattr(self, f"rope_sin_{kind}")[offset: offset + T]) for kind in self._rope_kinds()}

        x = self.tap_embed(self._add_positions(self.embed(idx), offset, pos))
        mem = self.memory.lookup(idx, cache, doc_ids) if self.memory is not None else None
        early = None
        for i, block in enumerate(self.blocks):
            if mem is not None and i in self.cfg.mem_layers:
                x = self.memory.inject(x, mem, self.cfg.mem_layers.index(i))
            x = block(x, *rope[block.kind], cache, same_doc)
            if i + 1 == self.cfg.exit_layer and targets is not None:
                early = x
        if cache is not None:
            cache.advance(T)
        h = self.tap_final_norm(self.norm_f(x))
        logits = self._softcap(self.lm_head(h))

        loss = extra = exit_loss = None
        if targets is not None:
            # float32 for the softmax: bf16 log-probs are too coarse for a good loss.
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
            if self.mtp is not None:
                extra = mtp_loss(self, idx, h, targets)
            if early is not None:
                el = self._softcap(self.lm_head(self.exit_norm(early)))
                exit_loss = F.cross_entropy(el.float().reshape(-1, el.size(-1)), targets.reshape(-1), ignore_index=-100)
        return ModelOutput(logits, loss, h if return_hidden else None, extra, exit_loss)

    def early_logits(self, idx: torch.Tensor, cache, positions: torch.Tensor | None = None) -> torch.Tensor:
        """Logits of the early exit: only the first cfg.exit_layer blocks run (they read
        and write the cache's first layers, at cache.pos, which this advances). The
        self-speculative drafter's step. A later full pass over the same positions
        recomputes those layers' K/V with the same values and overwrites them.

        In CUDA-graph (static) mode the tokens go to `positions` (a GPU tensor; default
        pos_t, pos_t+1, ...) and the cache doesn't move: the caller keeps track."""
        if not self.cfg.exit_layer:
            raise ValueError("this model has no early exit (model.exit_layer = 0)")
        T = idx.shape[1]
        offset = cache.pos
        if getattr(cache, "static", False):
            cache.new_step(positions)
            pos = cache.positions
            rope = {kind: (getattr(self, f"rope_cos_{kind}").index_select(0, pos),
                           getattr(self, f"rope_sin_{kind}").index_select(0, pos)) for kind in self._rope_kinds()}
        else:
            pos = None
            rope = {kind: (getattr(self, f"rope_cos_{kind}")[offset: offset + T],
                           getattr(self, f"rope_sin_{kind}")[offset: offset + T]) for kind in self._rope_kinds()}
        x = self._add_positions(self.embed(idx), offset, pos)
        mem = self.memory.lookup(idx, cache) if self.memory is not None else None
        for i, block in enumerate(self.blocks[:self.cfg.exit_layer]):
            if mem is not None and i in self.cfg.mem_layers:
                x = self.memory.inject(x, mem, self.cfg.mem_layers.index(i))
            x = block(x, *rope[block.kind], cache, None)
        cache.advance(T)   # a no-op in static mode
        return self._softcap(self.lm_head(self.exit_norm(x)))

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        """Token ids -> input vectors (the embedding, times embed_scale)."""
        x = self.tok_emb(idx)
        return x * self.cfg.embed_scale if self.cfg.embed_scale != 1.0 else x

    def _add_positions(self, x: torch.Tensor, offset: int, pos: torch.Tensor | None) -> torch.Tensor:
        """+ the learned position embedding (pos_type "learned"), at `pos` (a GPU tensor,
        CUDA-graph decoding) or at offset, offset+1, ..."""
        if self.pos_emb is None:
            return x
        if pos is None:
            pos = torch.arange(offset, offset + x.shape[1], device=x.device)
        return x + self.pos_emb(pos).to(x.dtype)

    def _rope_kinds(self) -> list[str]:
        return sorted({b.kind for b in self.blocks})

    def _softcap(self, logits: torch.Tensor) -> torch.Tensor:
        cap = self.cfg.final_softcap
        if not cap:
            return logits
        return (torch.tanh(logits.float() / cap) * cap).to(logits.dtype)

    def unembed(self, h: torch.Tensor) -> torch.Tensor:
        """Residual-stream vector -> logits (final norm + lm_head, + softcap).
        The logit lens applies this to every layer's residual stream."""
        return self._softcap(self.lm_head(self.norm_f(h)))

    def num_params(self, exclude_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())  # tied weights count once
        # Quantized layers hold their weights as bytes in buffers; count the
        # weights they stand for (once, if the embedding is tied to the output).
        n += sum({id(m.raw): m.n_weights for m in quantized_modules(self)}.values())
        if exclude_embedding:
            n -= self.tok_emb.n_weights if isinstance(self.tok_emb, QuantEmbedding) else self.tok_emb.weight.numel()
            if self.pos_emb is not None:
                n -= self.pos_emb.weight.numel()
        return n
