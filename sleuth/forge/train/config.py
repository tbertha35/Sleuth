"""Run configuration: everything needed to reproduce a training run."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model.config import ModelConfig


@dataclass
class DataConfig:
    shards: str = "data/shards/synthetic"   # directory made by `sleuth shard`


@dataclass
class TrainConfig:
    seed: int = 1337
    device: str = "auto"            # "auto" | "cuda" | "cuda:1" | "cpu"
    precision: str = "bf16"         # "bf16" (autocast; weights stay fp32) | "fp32". bf16 is ignored on CPU.
    micro_batch_size: int = 32      # sequences per forward/backward pass
    grad_accum: int = 1             # micro-batches summed per optimizer step
    seq_len: int = 256
    max_steps: int = 2000           # optimizer steps
    lr: float = 1e-3                # peak learning rate
    min_lr_ratio: float = 0.1       # final lr = lr * min_lr_ratio
    warmup_steps: int = 100
    schedule: str = "cosine"        # "cosine" | "wsd" (warmup-stable-decay) | "constant"
    decay_fraction: float = 0.2     # wsd only: final fraction of steps spent decaying
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0          # max global gradient norm (0 = off)
    log_every: int = 10
    eval_every: int = 250
    eval_batches: int = 20          # validation windows = eval_batches x micro_batch_size...
    eval_windows: int = 0           # ...unless this is set: a fixed count, independent of the batch size
    snapshot_every: int = 500       # weight snapshots for the analyzer...
    snapshot_log_spaced: bool = True  # ...plus steps 0,1,2,4,8,... (early training changes fastest)
    snapshot_dtype: str = "bf16"    # "bf16" | "fp32"
    resume_every: int = 500         # full resumable checkpoints (weights + optimizer)
    keep_resume: int = 2            # keep only the newest N resumable checkpoints
    compile: bool = False           # torch.compile the model (faster on GPU after a warm-up)
    peak_tflops: float = 0.0        # hardware peak for MFU (from hardware/<host>.json); 0 = don't report


@dataclass
class RunConfig:
    name: str = "run"
    runs_dir: str = "runs"          # overridden by $SLEUTH_RUNS if set
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
