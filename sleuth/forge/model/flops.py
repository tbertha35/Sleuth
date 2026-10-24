"""Parameter and FLOP counting, computed from the config alone.

The LAB compares variants "at equal compute", so compute must be counted
the same way for every variant, and before building the model. We follow
the PaLM paper's convention (Appendix B), the usual basis for MFU:

    training FLOPs per token = 6·N  +  12·L·d_attn·T

  * 6·N: every weight in a matmul costs 2 FLOPs per token (multiply + add)
    in the forward pass and ~4 in the backward pass. N counts *matmul*
    weights: the input embedding is a lookup, not a matmul, so it's
    excluded; the output projection (lm_head) is included even when tied.
  * 12·L·d_attn·T: the attention score and weighted-sum matmuls, whose
    cost grows with sequence length T (d_attn = n_heads·head_dim).

MFU (model FLOPs utilization) = achieved FLOP/s / hardware peak FLOP/s.
Good large runs reach 40-60%. Tiny models get far less, because their
matmuls are too small to keep the GPU busy.
"""

from __future__ import annotations

from .config import ModelConfig


def param_counts(cfg: ModelConfig) -> dict[str, int]:
    d, hd, H, K, L = cfg.d_model, cfg.head_dim, cfg.n_heads, cfg.kv_heads, cfg.n_layers
    b = 1 if cfg.bias else 0
    attn = d * H * hd + 2 * d * K * hd + H * hd * d
    qkv_b = 1 if (cfg.bias or cfg.qkv_bias) else 0
    attn_bias = qkv_b * (H * hd + 2 * K * hd) + b * d
    n_ffn_in = 2 if cfg.ffn_type == "swiglu" else 1
    mlp = n_ffn_in * d * cfg.hidden + cfg.hidden * d
    mlp_bias = b * (n_ffn_in * cfg.hidden + d)
    norm_params = d if cfg.norm_type == "rmsnorm" else 2 * d
    qk_norm = 2 * hd if cfg.qk_norm else 0
    per_layer = attn + attn_bias + mlp + mlp_bias + 2 * norm_params + qk_norm
    embedding = cfg.padded_vocab * d
    head = 0 if cfg.tie_embeddings else cfg.padded_vocab * d
    total = embedding + L * per_layer + norm_params + head
    return {
        "total": total,
        "embedding": embedding,
        "non_embedding": total - embedding,
        "per_layer": per_layer,
        # weights that take part in matmuls (what the 6N rule counts)
        "matmul": L * (attn + mlp) + cfg.padded_vocab * d,
    }


def train_flops_per_token(cfg: ModelConfig, seq_len: int) -> float:
    n = param_counts(cfg)["matmul"]
    return 6 * n + 12 * cfg.n_layers * cfg.n_heads * cfg.head_dim * seq_len


def describe(cfg: ModelConfig, seq_len: int | None = None) -> str:
    pc = param_counts(cfg)
    s = (f"{pc['total'] / 1e6:.2f}M params ({pc['non_embedding'] / 1e6:.2f}M non-embedding, "
         f"{pc['embedding'] / 1e6:.2f}M embedding)")
    if seq_len:
        s += f", {train_flops_per_token(cfg, seq_len) / 1e6:.1f} MFLOPs/token (train)"
    return s
