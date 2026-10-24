"""Model hyperparameters.

TOML has no "null", so 0 means "use the default" for the derived sizes
(`vocab_size`, `n_kv_heads`, `ffn_hidden`).
"""

from __future__ import annotations

from dataclasses import dataclass


def round_up(x: int, multiple: int) -> int:
    return (x + multiple - 1) // multiple * multiple


@dataclass
class ModelConfig:
    vocab_size: int = 0           # tokenizer vocab; 0 = take it from the training data. The embedding
                                  #   is padded up to a multiple of pad_vocab_to (see padded_vocab)
    d_model: int = 384            # width of the residual stream
    n_layers: int = 6
    n_heads: int = 6              # query heads
    head_size: int = 0            # per-head width; 0 = d_model / n_heads (some models, e.g. Qwen3, use more)
    n_kv_heads: int = 0           # 0 = n_heads (plain multi-head). Fewer = GQA: heads share K/V,
                                  #   which shrinks the KV cache (faster decoding) at little quality cost.
    ffn_type: str = "swiglu"      # "swiglu" | "gelu"
    ffn_hidden: int = 0           # 0 = auto: 8/3·d for SwiGLU (keeps params ≈ a 4·d GELU MLP), 4·d for GELU
    norm_type: str = "rmsnorm"    # "rmsnorm" | "layernorm"
    norm_eps: float = 1e-5
    pos_type: str = "rope"        # "rope" | "none"
    rope_theta: float = 10000.0   # RoPE base; larger = slower rotation = better for long contexts
    max_seq_len: int = 1024
    qk_norm: bool = False         # normalize q and k per head: stabilizes training, esp. low precision
    tie_embeddings: bool = True   # share input embedding and output projection (saves vocab·d params)
    bias: bool = False            # biases in linear layers (modern LLMs drop them)
    qkv_bias: bool = False        # biases on just the q/k/v projections (Qwen2 has these)
    init_std: float = 0.02
    pad_vocab_to: int = 64        # GPU matmuls are faster when dimensions are multiples of 64

    def __post_init__(self):
        if not self.head_size and self.d_model % self.n_heads:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.kv_heads <= 0 or self.n_heads % self.kv_heads:
            raise ValueError(f"n_heads ({self.n_heads}) must be a multiple of n_kv_heads ({self.kv_heads})")
        if self.head_dim % 2 and self.pos_type == "rope":
            raise ValueError("RoPE needs an even head_dim")
        for name, val, ok in (("ffn_type", self.ffn_type, ("swiglu", "gelu")),
                              ("norm_type", self.norm_type, ("rmsnorm", "layernorm")),
                              ("pos_type", self.pos_type, ("rope", "none"))):
            if val not in ok:
                raise ValueError(f"{name} must be one of {ok}, got {val!r}")

    @property
    def head_dim(self) -> int:
        return self.head_size or self.d_model // self.n_heads

    @property
    def kv_heads(self) -> int:
        return self.n_kv_heads or self.n_heads

    @property
    def hidden(self) -> int:
        if self.ffn_hidden:
            return self.ffn_hidden
        if self.ffn_type == "swiglu":
            return round_up(int(8 * self.d_model / 3), 64)
        return 4 * self.d_model

    @property
    def padded_vocab(self) -> int:
        return round_up(self.vocab_size, self.pad_vocab_to)
