"""Learning-rate schedules.

All three start with a linear *warmup*. Adam's running statistics are
garbage for the first steps, and a full-size learning rate then can push
the weights somewhere they never recover from.

  cosine:   warmup, then a smooth cosine curve from lr down to min_lr.
  wsd:      warmup-stable-decay: hold lr flat, then decay over the final
            `decay_fraction` of training. Nice for checkpoint studies: a
            stable-phase checkpoint can be decayed briefly to see what a
            shorter run would have reached, without retraining.
  constant: warmup, then flat (useful for debugging).
"""

from __future__ import annotations

import math

from .config import TrainConfig


def lr_at(step: int, cfg: TrainConfig) -> float:
    peak, floor = cfg.lr, cfg.lr * cfg.min_lr_ratio
    if cfg.warmup_steps and step < cfg.warmup_steps:
        return peak * (step + 1) / cfg.warmup_steps
    if cfg.schedule == "constant":
        return peak
    if cfg.schedule == "cosine":
        span = max(1, cfg.max_steps - cfg.warmup_steps)
        progress = min(1.0, (step - cfg.warmup_steps) / span)
        return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * progress))
    if cfg.schedule == "wsd":
        decay_steps = max(1, int(cfg.max_steps * cfg.decay_fraction))
        decay_start = cfg.max_steps - decay_steps
        if step < decay_start:
            return peak
        progress = min(1.0, (step - decay_start) / decay_steps)
        return peak + (floor - peak) * progress
    raise ValueError(f"unknown schedule {cfg.schedule!r}")
