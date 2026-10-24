"""The decoder-only Transformer.

    tokens -> embedding -> [Block] x n_layers -> final norm -> lm_head -> logits

Each Block is "pre-norm":
    h = x + Attention(Norm(x))     (tokens exchange information)
    y = h + MLP(Norm(h))           (each token processed on its own)

The running sum x -> h -> y is the *residual stream*. Every layer reads
from it and adds its contribution back. Most interpretability tools
(logit lens, ablation, SAEs) are built around this stream, which is why it
has taps at every step.
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
from .norm import make_norm
from .rope import rope_tables
from .taps import Tap, assign_tap_names


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.norm1 = make_norm(cfg.norm_type, cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, layer_idx)
        self.norm2 = make_norm(cfg.norm_type, cfg.d_model, cfg.norm_eps)
        self.mlp = make_mlp(cfg)
        self.tap_attn_out = Tap()
        self.tap_resid_mid = Tap()
        self.tap_mlp_out = Tap()
        self.tap_resid_post = Tap()

    def forward(self, x, cos, sin, cache=None):
        x = self.tap_resid_mid(x + self.tap_attn_out(self.attn(self.norm1(x), cos, sin, cache)))
        x = self.tap_resid_post(x + self.tap_mlp_out(self.mlp(self.norm2(x))))
        return x


@dataclass
class ModelOutput:
    logits: torch.Tensor                 # [B, T, padded_vocab]
    loss: torch.Tensor | None = None     # mean cross-entropy over targets (nats/token)


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.padded_vocab, cfg.d_model)
        self.tap_embed = Tap()
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm_f = make_norm(cfg.norm_type, cfg.d_model, cfg.norm_eps)
        self.tap_final_norm = Tap()
        self.lm_head = nn.Linear(cfg.d_model, cfg.padded_vocab, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # persistent=False: derived from the config, so not saved in checkpoints.
        cos, sin = self._rope()
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

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

    def _rope(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        if cfg.pos_type == "rope":
            return rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        z = torch.zeros(cfg.max_seq_len, cfg.head_dim)
        return z, z

    @classmethod
    def empty(cls, cfg: ModelConfig, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> "Transformer":
        """A model with uninitialized weights, for loading a checkpoint into.

        Building normally fills every weight with random numbers first. For a
        12B model that takes minutes and needs a full extra copy in RAM. Here
        the model is built on PyTorch's "meta" device (shapes only, no memory),
        then given real but uninitialized storage. Every weight must be loaded
        afterwards.
        """
        with torch.device("meta"):
            model = cls(cfg)
        model = model.to_empty(device=device).to(dtype)
        if cfg.tie_embeddings:  # to_empty creates separate tensors; re-tie them
            model.lm_head.weight = model.tok_emb.weight
        cos, sin = model._rope()  # buffers are derived, not loaded: recompute them
        model.rope_cos = cos.to(device)
        model.rope_sin = sin.to(device)
        assign_tap_names(model)
        return model

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)

    # ------------------------------------------------------------------ #
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, cache=None) -> ModelOutput:
        B, T = idx.shape
        offset = cache.pos if cache is not None else 0
        if offset + T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {offset + T} exceeds max_seq_len {self.cfg.max_seq_len}")
        cos = self.rope_cos[offset: offset + T]
        sin = self.rope_sin[offset: offset + T]

        x = self.tap_embed(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, cos, sin, cache)
        if cache is not None:
            cache.advance(T)
        logits = self.lm_head(self.tap_final_norm(self.norm_f(x)))

        loss = None
        if targets is not None:
            # float32 for the softmax: bf16 log-probs are too coarse for a good loss.
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return ModelOutput(logits, loss)

    def unembed(self, h: torch.Tensor) -> torch.Tensor:
        """Residual-stream vector -> logits (final norm + lm_head). The
        logit lens applies this to every layer's residual stream."""
        return self.lm_head(self.norm_f(h))

    def num_params(self, exclude_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())  # tied weights count once
        if exclude_embedding:
            n -= self.tok_emb.weight.numel()
        return n
