"""Run an analyzer view over every checkpoint of a run.

Output layout (what the dashboard reads):

    runs/<run>/analysis/<view>/<suite>/
        step_0000000/summary.json, neurons.npz
        step_0000500/...
        timeline.json        per-layer metrics vs. training step

Already-analyzed checkpoints are skipped (pass force=True to redo), so you
can re-run this while training is still going and only new snapshots get
processed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch

from ..forge.tokenizer.bpe import Tokenizer
from ..forge.train.checkpoint import list_snapshots, load_model
from .neurons import neuron_stats, save_result
from .suites import load_suite

# Per-layer metrics copied into the timeline (all from neurons.py summaries).
TIMELINE_METRICS = [
    "fire_rate_mean", "frac_low_activity", "frac_high_activity", "frac_selective",
    "act_rms", "act_mean_abs", "attn_out_rms", "mlp_out_rms", "resid_rms",
]


def analysis_dir(run_dir: Path, view: str, suite: Path) -> Path:
    return run_dir / "analysis" / view / Path(suite).name


def run_timeline(run_dir: str | Path, suite: str | Path, device: str = "cpu", k: float = 1.0,
                 force: bool = False, steps: list[int] | None = None,
                 log: Callable[[str], None] = print) -> Path:
    run_dir = Path(run_dir)
    tokenizer = Tokenizer.load(run_dir / "tokenizer.json")
    cases = load_suite(suite)
    out_root = analysis_dir(run_dir, "neuron_stats", Path(suite))
    snaps = list_snapshots(run_dir)
    if steps is not None:
        snaps = [(s, d) for s, d in snaps if s in set(steps)]
    if not snaps:
        raise FileNotFoundError(f"no snapshots in {run_dir}")
    log(f"neuron_stats: {len(cases)} tests, {len(snaps)} checkpoints -> {out_root}")
    for step, snap_dir in snaps:
        out = out_root / snap_dir.name
        if (out / "summary.json").exists() and not force:
            continue
        model, meta = load_model(snap_dir, device=device)
        res = neuron_stats(model, tokenizer, cases, k=k, device=device)
        save_result(res, out, {"step": step, "tokens_seen": int(meta.get("tokens_seen", 0)),
                               "suite": str(suite), "snapshot": meta["path"]})
        lay = res["summary"]["layers"]
        log(f"  step {step:7d}: fire rate {sum(x['fire_rate_mean'] for x in lay) / len(lay):.3f}, "
            f"low-activity {sum(x['frac_low_activity'] for x in lay) / len(lay):.1%}, "
            f"selective {sum(x['frac_selective'] for x in lay) / len(lay):.1%}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    path = build_timeline(out_root)
    log(f"timeline: {path}")
    return path


def build_timeline(out_root: Path) -> Path:
    summaries = []
    for d in sorted(out_root.glob("step_*")):
        f = d / "summary.json"
        if f.exists():
            summaries.append(json.loads(f.read_text()))
    summaries.sort(key=lambda s: s["step"])
    timeline = {
        "view": "neuron_stats",
        "steps": [s["step"] for s in summaries],
        "tokens": [s.get("tokens_seen", 0) for s in summaries],
        "n_layers": summaries[0]["n_layers"] if summaries else 0,
        "categories": summaries[0]["categories"] if summaries else [],
        "metrics": {m: [[lay[m] for lay in s["layers"]] for s in summaries] for m in TIMELINE_METRICS},
        "fire_rate_by_category": {
            c: [[lay["fire_rate_by_category"][c] for lay in s["layers"]] for s in summaries]
            for c in (summaries[0]["categories"] if summaries else [])
        },
    }
    path = out_root / "timeline.json"
    path.write_text(json.dumps(timeline) + "\n")
    return path
