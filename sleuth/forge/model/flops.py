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


def _layer_params(cfg: ModelConfig, i: int) -> tuple[int, int]:
    """(parameters, matmul weights) of block i. A layer that shares another's weights
    (cfg.shared_layers) owns only its norms, but its matmuls still run."""
    spec = cfg.attn_spec(i)
    d, hd, H, K = cfg.d_model, spec.head_dim, cfg.n_heads, spec.kv_heads
    b = 1 if cfg.bias else 0
    n_kv_proj = 1 if spec.v_from_k else 2                     # Gemma 4 full layers have no v_proj
    attn = d * H * hd + n_kv_proj * d * K * hd + H * hd * d
    qkv_b = 1 if (cfg.bias or cfg.qkv_bias) else 0
    attn_bias = qkv_b * (H * hd + n_kv_proj * K * hd) + b * d
    n_ffn_in = 2 if cfg.ffn_type in ("swiglu", "geglu") else 1
    mlp = n_ffn_in * d * cfg.hidden + cfg.hidden * d
    mlp_bias = b * (n_ffn_in * cfg.hidden + d)
    norm_params = d if cfg.norm_type == "rmsnorm" else 2 * d
    n_norms = 4 if cfg.post_norms else 2
    qk_norm = 2 * hd if cfg.qk_norm else 0
    scale = 1 if cfg.layer_scale else 0
    own = n_norms * norm_params + qk_norm + scale
    if cfg.shared_layers and cfg.shared_layers[i] != i:
        return own, attn + mlp
    return attn + attn_bias + mlp + mlp_bias + own, attn + mlp


def param_counts(cfg: ModelConfig) -> dict[str, int]:
    layers = [_layer_params(cfg, i) for i in range(cfg.n_layers)]
    d = cfg.d_model
    norm_params = d if cfg.norm_type == "rmsnorm" else 2 * d
    embedding = cfg.padded_vocab * d + (cfg.max_seq_len * d if cfg.pos_type == "learned" else 0)
    head = 0 if cfg.tie_embeddings else cfg.padded_vocab * d
    # MTP heads (mtp_heads.py): 4 norms, a 2d -> d projection and an MLP each; each one
    # also runs the shared output layer again (matmul cost, no extra parameters).
    n_ffn_in = 2 if cfg.ffn_type in ("swiglu", "geglu") else 1
    mtp_matmul = 2 * d * d + n_ffn_in * d * cfg.hidden + cfg.hidden * d
    mtp_bias = (d + n_ffn_in * cfg.hidden + d) if cfg.bias else 0
    mtp = cfg.mtp_heads * (mtp_matmul + mtp_bias + 4 * norm_params)
    # The early exit adds a norm and runs the shared output layer once more.
    exit_ = 1 if cfg.exit_layer else 0
    # The n-gram memory: its tables are lookups (no matmul, like the embedding); each
    # injection layer has a key and a value projection from the looked-up rows.
    mem_table = mem_matmul = 0
    if cfg.mem_rows:
        n_tables = len(cfg.mem_orders) * cfg.mem_heads
        mem_table = n_tables * cfg.mem_rows * cfg.mem_dim
        mem_matmul = len(cfg.mem_layers) * 2 * n_tables * cfg.mem_dim * d
    total = (embedding + sum(n for n, _ in layers) + norm_params + head + mtp + exit_ * norm_params
             + mem_table + mem_matmul)
    return {
        "total": total,
        "embedding": embedding,
        "non_embedding": total - embedding - mem_table,
        "memory_table": mem_table,
        "per_layer": layers[0][0],
        # weights that take part in matmuls (what the 6N rule counts)
        "matmul": sum(m for _, m in layers) + (1 + cfg.mtp_heads + exit_) * cfg.padded_vocab * d
                  + cfg.mtp_heads * mtp_matmul + mem_matmul,
    }


def train_flops_per_token(cfg: ModelConfig, seq_len: int) -> float:
    n = param_counts(cfg)["matmul"]
    # Attention term per layer; a sliding layer only looks at `window` keys.
    attn = sum(cfg.n_heads * cfg.attn_spec(i).head_dim * min(seq_len, cfg.attn_spec(i).window or seq_len)
               for i in range(cfg.n_layers))
    return 6 * n + 12 * attn


def describe(cfg: ModelConfig, seq_len: int | None = None) -> str:
    pc = param_counts(cfg)
    s = (f"{pc['total'] / 1e6:.2f}M params ({pc['non_embedding'] / 1e6:.2f}M non-embedding, "
         f"{pc['embedding'] / 1e6:.2f}M embedding)")
    if seq_len:
        s += f", {train_flops_per_token(cfg, seq_len) / 1e6:.1f} MFLOPs/token (train)"
    return s
