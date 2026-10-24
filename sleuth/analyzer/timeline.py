"""Run analyzer views over every checkpoint of a run.

Output layout (what the dashboard reads):

    runs/<run>/analysis/<view>/<suite>/
        step_0000000/summary.json, arrays.npz
        step_0000500/...
        timeline.json        every series of the view, one value per row per checkpoint

Already-analyzed checkpoints are skipped (pass force=True to redo), so you can
re-run this while training is still going and only new snapshots get
processed. Each checkpoint is loaded once and every requested view runs on it.

External models (a Hugging Face folder or a .gguf file) get a small run folder
of their own (`prepare_external_run`) with a single "checkpoint" at step 0.
Its config.json says where the model lives, so the dashboard lists it next to
our own runs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from ..forge.tokenizer.bpe import Tokenizer
from ..forge.train.checkpoint import list_snapshots, load_model
from .suites import load_suite
from .views import VIEWS


def analysis_dir(run_dir: Path, view: str, suite: Path) -> Path:
    return Path(run_dir) / "analysis" / view / Path(suite).name


def save_result(result: dict, out_dir: Path, extra: dict | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "arrays.npz", **result["arrays"])
    # summary.json last: its presence marks the checkpoint as done.
    (out_dir / "summary.json").write_text(json.dumps({**result["summary"], **(extra or {})}, indent=1) + "\n")


def external_run_name(model_path: Path) -> str:
    """A readable, unique folder name for an external model.

    The folder name alone isn't enough: Hugging Face downloads are often in
    folders with generic names ("hf", "model"), and two of them would share
    results. So: parent folder + name (or the file name for a .gguf), plus a
    short hash of the full path, e.g. "qwen3-0.6b-hf-3f9a1c".
    """
    base = model_path.stem if model_path.is_file() else f"{model_path.parent.name}-{model_path.name}"
    base = re.sub(r"[^A-Za-z0-9_.-]+", "-", base).strip("-").lower() or "model"
    digest = hashlib.sha1(os.path.normcase(str(model_path)).encode("utf-8")).hexdigest()[:6]
    return f"{base}-{digest}"


def prepare_external_run(model_path: str | Path, name: str | None = None, dtype=None, device: str = "cpu",
                         keep_quantized: bool | str = "auto") -> Path:
    """Create (or reuse) runs/ext-<name>/ for analyzing an external model."""
    from ..env import env_info, runs_root
    from ..models import detect, read_config

    model_path = Path(model_path).resolve()
    name = name or external_run_name(model_path)
    run_dir = runs_root() / f"ext-{name}"
    if (run_dir / "config.json").exists():
        old = json.loads((run_dir / "config.json").read_text()).get("external", {}).get("path")
        if old and os.path.normcase(old) != os.path.normcase(str(model_path)):
            raise ValueError(f"{run_dir} already holds results for another model ({old}); pick a different --name")
    else:
        model_cfg, info = read_config(model_path)  # the weights are loaded later, only if a view needs them
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = {"name": name, "external": {"path": str(model_path), "format": detect(model_path),
                                          "dtype": str(dtype).replace("torch.", "") if dtype else None,
                                          "keep_quantized": keep_quantized,
                                          "info": info},
               "model": dataclasses.asdict(model_cfg), "train": {"max_steps": 0}}
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        (run_dir / "env.json").write_text(json.dumps(env_info(), indent=2) + "\n")
    return run_dir


def _checkpoints(run_dir: Path, device: str):
    """[(step, loader)] plus the tokenizer, for our runs or an external model."""
    cfg = json.loads((run_dir / "config.json").read_text())
    ext = cfg.get("external")
    if ext:
        from ..models import load_any, load_tokenizer

        dtype = getattr(torch, ext["dtype"]) if ext.get("dtype") else None
        # Load the weights only if some view actually needs to run (big models take a while).
        keep = ext.get("keep_quantized", "auto")
        load = lambda: (load_any(ext["path"], device=device, dtype=dtype, keep_quantized=keep)[0],  # noqa: E731
                        {"tokens_seen": 0})
        return [(0, load)], load_tokenizer(ext["path"])
    tokenizer = Tokenizer.load(run_dir / "tokenizer.json")
    return [(s, (lambda d=d: load_model(d, device=device))) for s, d in list_snapshots(run_dir)], tokenizer


def run_timeline(run_dir: str | Path, suite: str | Path, device: str = "cpu", views: list[str] | None = None,
                 force: bool = False, steps: list[int] | None = None, options: dict | None = None,
                 log: Callable[[str], None] = print) -> dict[str, Path]:
    """Run `views` (default: all) on every checkpoint; returns {view: timeline.json path}."""
    run_dir = Path(run_dir)
    views = views or list(VIEWS)
    unknown = [v for v in views if v not in VIEWS]
    if unknown:
        raise ValueError(f"unknown view(s) {unknown}; choose from {list(VIEWS)}")
    options = options or {}
    cases = load_suite(suite)
    ckpts, tokenizer = _checkpoints(run_dir, device)
    if steps is not None:
        ckpts = [(s, f) for s, f in ckpts if s in set(steps)]
    if not ckpts:
        raise FileNotFoundError(f"no snapshots in {run_dir}")
    log(f"analyzing {len(ckpts)} checkpoints x {len(views)} views ({', '.join(views)}), {len(cases)} tests")
    for step, load in ckpts:
        step_dir = f"step_{step:07d}"
        todo = [v for v in views if force or not (analysis_dir(run_dir, v, suite) / step_dir / "summary.json").exists()]
        if not todo:
            continue
        model, meta = load()
        line = []
        for v in todo:
            t0 = time.time()
            res = VIEWS[v].run(model, tokenizer, cases, device, **options.get(v, {}))
            save_result(res, analysis_dir(run_dir, v, suite) / step_dir,
                        {"step": step, "tokens_seen": int(meta.get("tokens_seen", 0)), "suite": str(suite)})
            line.append(f"{v} {time.time() - t0:.1f}s")
        log(f"  step {step:7d}: " + ", ".join(line))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out = {v: build_timeline(analysis_dir(run_dir, v, suite), v) for v in views}
    log("timelines: " + ", ".join(str(p) for p in out.values()))
    return out


def build_timeline(out_root: Path, view: str) -> Path:
    summaries = []
    for d in sorted(out_root.glob("step_*")):
        f = d / "summary.json"
        if f.exists():
            summaries.append(json.loads(f.read_text()))
    summaries.sort(key=lambda s: s["step"])
    series: dict[str, dict] = {}
    for s in summaries:
        for name, ser in VIEWS[view].timeline_series(s).items():
            entry = series.setdefault(name, {k: ser[k] for k in ("label", "help", "rows", "fmt")} | {"values": []})
            entry["values"].append(ser["values"])
    timeline = {
        "view": view,
        "steps": [s["step"] for s in summaries],
        "tokens": [s.get("tokens_seen", 0) for s in summaries],
        "series": series,
    }
    path = out_root / "timeline.json"
    path.write_text(json.dumps(timeline) + "\n")
    return path
