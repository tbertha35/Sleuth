"""Checkpoints: weight snapshots (for the analyzer) and resumable state.

A run directory looks like:

    runs/<run_id>/
      config.json                resolved RunConfig
      env.json                   machine, versions, git commit
      tokenizer.json             copy, so the run is self-contained
      metrics.jsonl              one JSON object per log line
      snapshots/step_0000000/model.safetensors   <- the analyzer reads these
      snapshots/step_0000500/model.safetensors
      resume/step_0000500/model.safetensors + state.pt   <- for --resume

Two kinds, because they have different jobs:
  * snapshots: weights only, bf16 by default, never deleted. Small, so we
    can keep many (log-spaced: dense early in training, where most
    change happens) for the checkpoint timeline. The model config is
    stored in the safetensors metadata, so the file alone is enough to
    rebuild the model.
  * resume: fp32 weights + optimizer state + RNG. ~4x the model size (Adam
    keeps two extra tensors per weight), so only the newest few are kept.
"""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
from pathlib import Path

import torch
from torch import nn

from ...formats.safetensors import SafetensorsFile, load_file, save_file
from ..model.config import ModelConfig
from ..model.transformer import Transformer

SNAPSHOT_FORMAT = "sleuth-snapshot-v1"
_STEP_RE = re.compile(r"step_(\d+)$")


def snapshot_steps(max_steps: int, every: int, log_spaced: bool) -> set[int]:
    steps = {0, max_steps}
    if every > 0:
        steps |= set(range(every, max_steps + 1, every))
    if log_spaced:
        s = 1
        while s < max_steps:
            steps.add(s)
            s *= 2
    return {s for s in steps if 0 <= s <= max_steps}


def _step_dir(root: Path, step: int) -> Path:
    return root / f"step_{step:07d}"


def _list_steps(root: Path) -> list[tuple[int, Path]]:
    if not root.exists():
        return []
    out = []
    for d in root.iterdir():
        m = _STEP_RE.match(d.name)
        if m and (d / "model.safetensors").exists():
            out.append((int(m.group(1)), d))
    return sorted(out)


# --------------------------------------------------------------------------- #
# Weights <-> tensors (handling tied embeddings)
# --------------------------------------------------------------------------- #

def model_tensors(model: Transformer, dtype: torch.dtype | None = None) -> dict[str, torch.Tensor]:
    """state_dict without the duplicate entry for tied weights."""
    sd = model.state_dict()
    if model.cfg.tie_embeddings:
        sd.pop("lm_head.weight", None)
    return {k: (v.to(dtype) if dtype is not None and v.is_floating_point() else v) for k, v in sd.items()}


def load_model_tensors(model: Transformer, tensors: dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(tensors, strict=False)
    if model.cfg.tie_embeddings:
        missing = [k for k in missing if k != "lm_head.weight"]
    if missing or unexpected:
        raise KeyError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")


# --------------------------------------------------------------------------- #
# Snapshots
# --------------------------------------------------------------------------- #

def save_snapshot(run_dir: Path, step: int, model: Transformer, tokens_seen: int, dtype: str = "bf16") -> Path:
    d = _step_dir(run_dir / "snapshots", step)
    d.mkdir(parents=True, exist_ok=True)
    tdtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    meta = {
        "format": SNAPSHOT_FORMAT,
        "step": str(step),
        "tokens_seen": str(tokens_seen),
        "model_config": json.dumps(dataclasses.asdict(model.cfg)),
    }
    save_file(model_tensors(model, tdtype), d / "model.safetensors", metadata=meta)
    return d


def list_snapshots(run_dir: str | Path) -> list[tuple[int, Path]]:
    return _list_steps(Path(run_dir) / "snapshots")


def resolve_weights(path: str | Path, step: int | None = None) -> Path:
    """Accept a .safetensors file, a step directory, or a run directory
    (picks `step`, or the latest snapshot)."""
    p = Path(path)
    if p.is_file():
        return p
    if (p / "model.safetensors").exists():
        return p / "model.safetensors"
    snaps = list_snapshots(p)
    if not snaps:
        raise FileNotFoundError(f"no snapshots under {p}")
    if step is None:
        return snaps[-1][1] / "model.safetensors"
    for s, d in snaps:
        if s == step:
            return d / "model.safetensors"
    raise FileNotFoundError(f"no snapshot for step {step} in {p} (have {[s for s, _ in snaps]})")


def load_model(path: str | Path, step: int | None = None, device: str | torch.device = "cpu",
               dtype: torch.dtype = torch.float32) -> tuple[Transformer, dict]:
    """Rebuild a model from a snapshot. Returns (model in eval mode, metadata)."""
    f = resolve_weights(path, step)
    with SafetensorsFile(f) as st:
        meta = dict(st.metadata)
    if meta.get("format") != SNAPSHOT_FORMAT:
        raise ValueError(f"{f} is not a Sleuth snapshot (format={meta.get('format')})")
    cfg = ModelConfig(**json.loads(meta["model_config"]))
    model = Transformer(cfg)
    load_model_tensors(model, {k: v.to(torch.float32) for k, v in load_file(f).items()})
    model.to(device=device, dtype=dtype).eval()
    meta["path"] = str(f)
    return model, meta


# --------------------------------------------------------------------------- #
# Resumable checkpoints
# --------------------------------------------------------------------------- #

def save_resume(run_dir: Path, step: int, model: Transformer, optimizer: torch.optim.Optimizer,
                extra: dict, keep: int) -> Path:
    root = run_dir / "resume"
    d = _step_dir(root, step)
    tmp = d.with_name(d.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    save_file(model_tensors(model, torch.float32), tmp / "model.safetensors")
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        **extra,
    }
    # torch.save uses pickle. Fine here: we only ever load our own files.
    torch.save(state, tmp / "state.pt")
    if d.exists():
        shutil.rmtree(d)
    tmp.rename(d)  # appear atomically, so a crash mid-save leaves the previous one intact
    for _, old in _list_steps(root)[:-keep] if keep > 0 else []:
        shutil.rmtree(old)
    return d


def latest_resume(run_dir: str | Path) -> Path | None:
    steps = _list_steps(Path(run_dir) / "resume")
    return steps[-1][1] if steps else None


def load_resume(ckpt_dir: Path, model: Transformer, optimizer: torch.optim.Optimizer) -> dict:
    load_model_tensors(model, load_file(ckpt_dir / "model.safetensors"))
    state = torch.load(ckpt_dir / "state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["rng_cpu"])
    if state.get("rng_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng_cuda"])
    return state


def unwrap(model: nn.Module) -> nn.Module:
    """torch.compile wraps the model; checkpoints must use the original."""
    return getattr(model, "_orig_mod", model)
