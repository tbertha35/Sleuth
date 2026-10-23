import json
import math

import pytest
import torch

from sleuth.forge.train.checkpoint import (latest_resume, list_snapshots, load_model, model_tensors,
                                           snapshot_steps)
from sleuth.forge.train.config import TrainConfig
from sleuth.forge.train.schedule import lr_at
from sleuth.forge.train.trainer import train

from helpers import tiny_run_config


def quiet(_):
    pass


def test_snapshot_schedule():
    assert snapshot_steps(100, 25, True) == {0, 1, 2, 4, 8, 16, 25, 32, 50, 64, 75, 100}
    assert snapshot_steps(100, 0, False) == {0, 100}


@pytest.mark.parametrize("schedule", ["cosine", "wsd", "constant"])
def test_lr_schedules(schedule):
    cfg = TrainConfig(lr=1.0, min_lr_ratio=0.1, warmup_steps=10, max_steps=100, schedule=schedule)
    lrs = [lr_at(s, cfg) for s in range(100)]
    assert lrs[0] == pytest.approx(0.1)
    assert lrs[9] == pytest.approx(1.0)
    assert all(a <= b + 1e-12 for a, b in zip(lrs[:10], lrs[1:10]))  # warmup increases
    if schedule != "constant":
        assert all(a >= b - 1e-12 for a, b in zip(lrs[10:], lrs[11:]))  # then never increases
        assert lr_at(100, cfg) == pytest.approx(0.1)
    if schedule == "wsd":
        assert lrs[50] == 1.0  # flat in the stable phase


def test_training_end_to_end(tiny_data, tmp_path):
    cfg = tiny_run_config(tiny_data, tmp_path, max_steps=12, lr=3e-3)
    run = train(cfg, log=quiet)
    assert (run / "config.json").exists() and (run / "env.json").exists() and (run / "tokenizer.json").exists()
    recs = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    train_recs = [r for r in recs if "loss" in r]
    val_recs = [r for r in recs if "val_loss" in r]
    assert train_recs[-1]["loss"] < train_recs[0]["loss"]  # it learns something
    assert all(math.isfinite(r["val_bpb"]) and r["val_bpb"] > 0 for r in val_recs)
    steps = [s for s, _ in list_snapshots(run)]
    assert steps == sorted(snapshot_steps(12, 3, True))
    model, meta = load_model(run)
    assert meta["step"] == "12"
    assert model.cfg.vocab_size == json.loads((tiny_data["shards"] / "meta.json").read_text())["vocab_size"]


def test_resume_is_exact(tiny_data, tmp_path):
    """6 steps in one go == 4 steps, stop, resume, 2 more (bit-for-bit on CPU)."""
    straight = train(tiny_run_config(tiny_data, tmp_path / "a", snapshot_dtype="fp32"), log=quiet)

    cfg = tiny_run_config(tiny_data, tmp_path / "b", snapshot_dtype="fp32")
    run = train(cfg, stop_at_step=4, log=quiet)  # 4 is not a resume_every multiple
    assert latest_resume(run).name == "step_0000004"
    train(cfg, run_dir=run, resume=True, log=quiet)

    a, _ = load_model(straight)
    b, _ = load_model(run)
    for (k, va), (_, vb) in zip(model_tensors(a).items(), model_tensors(b).items()):
        assert torch.equal(va, vb), k
    # The metrics log continues rather than restarting.
    steps = [json.loads(line)["step"] for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert steps == sorted(steps) and steps[-1] == 6
