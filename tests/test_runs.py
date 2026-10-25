"""Stale-run detection and `sleuth runs --prune` (bug 2 from round 4)."""

import json

from sleuth.analyzer.timeline import external_run_name
from sleuth.cli import main
from sleuth.runs import prune, scan


def _make(runs, name, ext=None):
    (runs / name).mkdir(parents=True)
    cfg = {"name": name, "train": {"max_steps": 0 if ext else 10}}
    if ext:
        cfg["external"] = ext
    (runs / name / "config.json").write_text(json.dumps(cfg))


def test_old_external_runs_are_stale_and_named_ones_are_not(tmp_path):
    model = tmp_path / "models" / "hf"
    model.mkdir(parents=True)
    runs = tmp_path / "runs"
    _make(runs, "20261024-000000-mine")
    _make(runs, "ext-hf", {"path": str(model)})                                     # before the fix
    _make(runs, "ext-" + external_run_name(model), {"path": str(model)})              # today's automatic name
    _make(runs, "ext-my-pick", {"path": str(model), "keep_quantized": "auto"})        # --name, current Sleuth
    _make(runs, "ext-other", {"path": str(tmp_path / "other.gguf")})                  # old, never replaced
    got = {r["id"]: r["stale"] for r in scan(runs)}
    assert got["20261024-000000-mine"] is None and got["ext-my-pick"] is None
    assert got["ext-" + external_run_name(model)] is None
    assert "replaced by ext-" in got["ext-hf"]
    assert got["ext-other"] and "replaced" not in got["ext-other"]


def test_prune_lists_first_and_deletes_only_with_yes(tmp_path, capsys):
    runs = tmp_path / "runs"
    _make(runs, "ext-hf", {"path": str(tmp_path / "hf")})
    _make(runs, "20261024-000000-mine")
    assert main(["runs", "--runs", str(runs)]) == 0
    assert "STALE" in capsys.readouterr().out
    assert main(["runs", "--runs", str(runs), "--prune"]) == 0
    assert "would delete" in capsys.readouterr().out and (runs / "ext-hf").exists()
    assert prune(runs, yes=True, log=lambda s: None) == [runs / "ext-hf"]
    assert not (runs / "ext-hf").exists() and (runs / "20261024-000000-mine").exists()
