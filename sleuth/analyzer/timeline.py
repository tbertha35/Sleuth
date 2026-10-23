"""Run analyzer views over every checkpoint of a run.

Output layout (what the dashboard reads):

    runs/<run>/analysis/<view>/<suite>/
        step_0000000/summary.json, arrays.npz
        step_0000500/...
        timeline.json        every series of the view, one value per row per checkpoint

Already-analyzed checkpoints are skipped (pass force=True to redo), so you can
re-run this while training is still going and only new snapshots get
processed. Each checkpoint is loaded once and every requested view runs on it.
"""

from __future__ import annotations

import json
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


def run_timeline(run_dir: str | Path, suite: str | Path, device: str = "cpu", views: list[str] | None = None,
                 force: bool = False, steps: list[int] | None = None, options: dict | None = None,
                 log: Callable[[str], None] = print) -> dict[str, Path]:
    """Run `views` (default: all) on every snapshot; returns {view: timeline.json path}."""
    run_dir = Path(run_dir)
    views = views or list(VIEWS)
    unknown = [v for v in views if v not in VIEWS]
    if unknown:
        raise ValueError(f"unknown view(s) {unknown}; choose from {list(VIEWS)}")
    options = options or {}
    tokenizer = Tokenizer.load(run_dir / "tokenizer.json")
    cases = load_suite(suite)
    snaps = list_snapshots(run_dir)
    if steps is not None:
        snaps = [(s, d) for s, d in snaps if s in set(steps)]
    if not snaps:
        raise FileNotFoundError(f"no snapshots in {run_dir}")
    log(f"analyzing {len(snaps)} checkpoints x {len(views)} views ({', '.join(views)}), {len(cases)} tests")
    for step, snap_dir in snaps:
        todo = [v for v in views if force or not (analysis_dir(run_dir, v, suite) / snap_dir.name / "summary.json").exists()]
        if not todo:
            continue
        model, meta = load_model(snap_dir, device=device)
        line = []
        for v in todo:
            t0 = time.time()
            res = VIEWS[v].run(model, tokenizer, cases, device, **options.get(v, {}))
            save_result(res, analysis_dir(run_dir, v, suite) / snap_dir.name,
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
