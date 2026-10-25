"""What's in the runs folder, and which entries are left over from older versions.

Used by the dashboard (to hide stale runs) and by `sleuth runs` (to list them
and, with --prune, delete them).

A run is *stale* when it's an external-model results folder made before
external runs were named after their model's path (round 2): those were
called just `ext-<folder name>`, e.g. `ext-hf` for a Qwen download in a folder
named "hf". Re-analyzing the same model now writes to
`ext-qwen3-0.6b-hf-<hash>`, so the old folder lingers next to the new one.

How we recognise one: today's names end in a 6-character hash of the model's
path (`external_run_name`), and an `--name` chosen by hand is recorded with
`keep_quantized` (added in the same round as the hash, give or take one
commit). A config with neither is from before the fix. If another run holds
results for the same model file, we say which one replaces it.

Runs are never deleted automatically; `sleuth runs --prune` lists what it
would delete, and only deletes with --yes.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


def _path_hash(model_path: str) -> str:
    # Same digest as analyzer.timeline.external_run_name (kept in step by a test).
    return hashlib.sha1(os.path.normcase(model_path).encode("utf-8")).hexdigest()[:6]


def _read_config(d: Path) -> dict | None:
    try:
        return json.loads((d / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _is_legacy(name: str, cfg: dict) -> bool:
    ext = cfg.get("external") or {}
    if not ext.get("path"):
        return False  # one of our own training runs
    # Named the current way, or by hand with a current Sleuth: not legacy.
    return not (name.endswith("-" + _path_hash(ext["path"])) or "keep_quantized" in ext)


def stale_reason(name: str, cfg: dict, others: dict[str, dict]) -> str | None:
    """Why this run is stale, or None. `others`: {run name: config} of every run."""
    if not _is_legacy(name, cfg):
        return None
    path = os.path.normcase(cfg["external"]["path"])
    newer = sorted(n for n, c in others.items()
                   if not _is_legacy(n, c) and os.path.normcase((c.get("external") or {}).get("path") or "") == path)
    why = "made before external runs were named by model path (round 2)"
    return f"{why}; replaced by {newer[0]}" if newer else why


def scan(runs_dir: str | Path) -> list[dict]:
    """[{"id", "path", "config", "stale"}] for every folder with a config.json, newest name first."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    configs = {}
    for d in runs_dir.iterdir():
        if d.is_dir() and (cfg := _read_config(d)) is not None:
            configs[d.name] = cfg
    out = []
    for name in sorted(configs, reverse=True):
        out.append({"id": name, "path": runs_dir / name, "config": configs[name],
                    "stale": stale_reason(name, configs[name], configs)})
    return out


def folder_size(d: Path) -> int:
    total = 0
    for root, _, files in os.walk(d):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def prune(runs_dir: str | Path, yes: bool = False, log=print) -> list[Path]:
    """Delete stale runs (only with yes=True; otherwise just list them)."""
    stale = [r for r in scan(runs_dir) if r["stale"]]
    if not stale:
        log("nothing to prune")
        return []
    for r in stale:
        log(f"{'deleting' if yes else 'would delete'} {r['path']}  ({folder_size(r['path']) / 1e6:.1f} MB): {r['stale']}")
        if yes:
            shutil.rmtree(r["path"])
    if not yes:
        log("run again with --yes to delete these")
    return [r["path"] for r in stale]
