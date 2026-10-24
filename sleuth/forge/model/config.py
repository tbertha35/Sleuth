"""Model hyperparameters.

TOML has no "null", so 0 means "use the default" for the derived sizes
(`vocab_size`, `n_kv_heads`, `ffn_hidden`).

Most models use the same attention in every layer. Gemma 4 mixes two kinds
("layer_types"): *sliding* layers, which only look at the last
`sliding_window` tokens and are cheap, and *full* layers, which see the
whole context. The two kinds even have different head widths, KV-head
counts and RoPE bases. The plain fields (head_size, n_kv_heads, rope_theta)
describe full layers; the swa_* fields override them for sliding layers.
`attn_spec(i)` resolves what layer i actually uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
    ffn_type: str = "swiglu"      # "swiglu" | "geglu" (gated, GELU-tanh: Gemma) | "gelu"
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

    # ---- features of other model families (all off by default) ----
    embed_scale: float = 1.0      # multiply token embeddings by this (Gemma: sqrt(d_model))
    post_norms: bool = False      # also normalize each attention/MLP output before adding it to the
                                  #   residual stream ("sandwich" norm, Gemma 2-4)
    v_norm: bool = False          # RMS-normalize values per head, without a learned gain (Gemma 4)
    attn_scale: float = 0.0       # attention logit scale; 0 = 1/sqrt(head_dim). Gemma 4 uses 1.0,
                                  #   because q_norm/k_norm already fix the size of q and k
    final_softcap: float = 0.0    # logits -> cap·tanh(logits/cap): bounded logits (Gemma); 0 = off
    layer_scale: bool = False     # a learned scalar per layer multiplies its output stream (Gemma 4)
    layer_types: list[str] = field(default_factory=list)  # per layer "full" | "sliding"; [] = all full
    sliding_window: int = 0       # sliding layers see this many most recent tokens (incl. the current one)
    swa_head_size: int = 0        # sliding layers' head width / KV heads / RoPE base (0 = as full layers)
    swa_kv_heads: int = 0
    swa_rope_theta: float = 0.0
    full_v_from_k: bool = False   # full layers have no v_proj: values are the (unnormalized) keys (Gemma 4)

    def __post_init__(self):
        if not self.head_size and self.d_model % self.n_heads:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.kv_heads <= 0 or self.n_heads % self.kv_heads:
            raise ValueError(f"n_heads ({self.n_heads}) must be a multiple of n_kv_heads ({self.kv_heads})")
        if self.layer_types and len(self.layer_types) != self.n_layers:
            raise ValueError(f"layer_types has {len(self.layer_types)} entries for {self.n_layers} layers")
        if any(t not in ("full", "sliding") for t in self.layer_types):
            raise ValueError(f"layer_types entries must be 'full' or 'sliding', got {sorted(set(self.layer_types))}")
        if "sliding" in self.layer_types and self.sliding_window <= 0:
            raise ValueError("sliding layers need sliding_window > 0")
        for spec in {self.attn_spec(i) for i in range(self.n_layers)}:
            if spec.kv_heads <= 0 or self.n_heads % spec.kv_heads:
                raise ValueError(f"n_heads ({self.n_heads}) must be a multiple of the KV heads ({spec.kv_heads})")
            if spec.head_dim % 2 and self.pos_type == "rope":
                raise ValueError("RoPE needs an even head_dim")
        for name, val, ok in (("ffn_type", self.ffn_type, ("swiglu", "geglu", "gelu")),
                              ("norm_type", self.norm_type, ("rmsnorm", "layernorm")),
                              ("pos_type", self.pos_type, ("rope", "none"))):
            if val not in ok:
                raise ValueError(f"{name} must be one of {ok}, got {val!r}")

    def layer_type(self, i: int) -> str:
        return self.layer_types[i] if self.layer_types else "full"

    def attn_spec(self, i: int) -> "AttnSpec":
        """The attention shape of layer i (see the module docstring)."""
        if self.layer_type(i) == "sliding":
            return AttnSpec("sliding", self.swa_head_size or self.head_dim, self.swa_kv_heads or self.kv_heads,
                            self.swa_rope_theta or self.rope_theta, self.sliding_window, False)
        return AttnSpec("full", self.head_dim, self.kv_heads, self.rope_theta, 0, self.full_v_from_k)

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
        if self.ffn_type in ("swiglu", "geglu"):
            return round_up(int(8 * self.d_model / 3), 64)
        return 4 * self.d_model

    @property
    def llama_compatible(self) -> bool:
        """True if the model is a plain Llama/Qwen-style decoder (exportable to HF format)."""
        return (self.ffn_type == "swiglu" and self.norm_type == "rmsnorm" and self.pos_type == "rope"
                and not self.bias and self.embed_scale == 1.0 and not self.post_norms and not self.v_norm
                and not self.attn_scale and not self.final_softcap and not self.layer_scale
                and "sliding" not in self.layer_types and not self.full_v_from_k)

    @property
    def padded_vocab(self) -> int:
        return round_up(self.vocab_size, self.pad_vocab_to)


@dataclass(frozen=True)
class AttnSpec:
    kind: str           # "full" | "sliding"
    head_dim: int
    kv_heads: int
    rope_theta: float
    window: int         # 0 = sees the whole context
    v_from_k: bool      # no v_proj: values come from the keys
