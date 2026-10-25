"""The dashboard's job runner (sleuth/ui/jobs.py): what it accepts, how it
queues, how it stops things. Uses tests/fake_sleuth.py instead of real commands."""

import json
import sys
import time
from pathlib import Path

import pytest

from sleuth.ui.jobs import COMMANDS, GpuBusy, JobError, JobManager

FAKE = [sys.executable, "-u", str(Path(__file__).parent / "fake_sleuth.py")]


@pytest.fixture
def jm(tmp_path):
    runs = tmp_path / "runs"
    (runs / "myrun").mkdir(parents=True)
    (runs / "myrun" / "config.json").write_text(json.dumps({"name": "x", "train": {"max_steps": 5}}))
    ext = runs / "ext-model-abc123"
    ext.mkdir()
    (ext / "config.json").write_text(json.dumps({"external": {"path": str(tmp_path / "m.gguf")}}))
    (tmp_path / "m.gguf").write_bytes(b"GGUF")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "a.toml").write_text("name = 'a'\n")
    (tmp_path / "suites" / "s").mkdir(parents=True)
    (tmp_path / "suites" / "s" / "x.jsonl").write_text("{}\n")
    m = JobManager(runs, cwd=tmp_path, prefix=FAKE)
    yield m
    for p in list(m.procs.values()):
        p.kill()


def wait(jm, job, statuses, timeout=20):
    t0 = time.time()
    while jm.get(job.id).status not in statuses:
        assert time.time() - t0 < timeout, f"{job.id} stuck at {jm.get(job.id).status}"
        time.sleep(0.05)
    return jm.get(job.id)


def test_builds_argv_from_validated_fields(jm):
    argv, lane, run = jm.build("train", {"config": "configs/a.toml", "set": ["train.lr=3e-4", ""],
                                         "stop_at_step": "100"})
    assert argv == ["train", "configs/a.toml", "--set=train.lr=3e-4", "--stop-at-step=100"] and lane == "gpu"
    argv, lane, run = jm.build("generate", {"path": "myrun", "prompt": "-not an option\nsecond line",
                                            "temperature": 0, "chat": True, "device": "cpu"})
    assert argv[1] == str(jm.runs_dir / "myrun") and run == "myrun"
    assert "--prompt=-not an option\nsecond line" in argv and "--temperature=0.0" in argv and "--chat" in argv
    assert lane == "cpu"  # device=cpu doesn't need the GPU lane
    # An external model's results folder resolves to the model file itself.
    argv, _, _ = jm.build("score", {"path": "ext-model-abc123", "suite": ["suites/s"]})
    assert argv[1].endswith("m.gguf") and "--suite=suites/s" in argv
    argv, lane, _ = jm.build("tokenizer", {"input": ["configs", "suites"], "vocab_size": 512})
    assert argv[:4] == ["tokenizer", "--input", "configs", "suites"] and lane == "cpu"


@pytest.mark.parametrize("command,values,msg", [
    ("rm", {}, "unknown command"),
    ("train", {"config": "configs/a.toml", "shell": "x"}, "unknown field"),
    ("train", {}, "required"),
    ("train", {"config": "configs/nope.toml"}, "no config"),
    ("train", {"config": "configs/a.toml", "set": ["rm -rf /"]}, "KEY=VALUE"),
    ("train", {"config": "configs/a.toml", "stop_at_step": "1; ls"}, "whole number"),
    ("resume", {"run": "../etc"}, "not a run"),
    ("resume", {"run": "missing"}, "no run"),
    ("eval", {"run": "myrun", "windows": 0}, "between"),
    ("generate", {"path": "--help"}, "can't start with '-'"),
    ("generate", {"path": "no/such/model"}, "neither"),
    ("generate", {"path": "myrun", "temperature": "nan"}, "finite"),
    ("generate", {"path": "myrun", "device": "tpu"}, "choose one"),
    ("generate", {"path": "myrun", "chat": "yes"}, "true/false"),
    ("analyze", {"path": "myrun", "suite": ["suites/s"], "views": "attention;rm"}, "view names"),
    ("tokenizer", {"input": ["-x"]}, "can't start with '-'"),
])
def test_rejects_anything_off_the_allow_list(jm, command, values, msg):
    with pytest.raises(JobError, match=msg):
        jm.build(command, values)


def test_every_command_has_a_lane_and_known_field_kinds():
    kinds = {"text", "int", "float", "bool", "choice", "run", "model", "path", "config", "suite", "set"}
    for c in COMMANDS.values():
        assert c.lane in ("gpu", "cpu")
        assert all(f.kind in kinds for f in c.fields), c.name


def test_one_gpu_job_at_a_time_and_cpu_jobs_alongside(jm):
    a = jm.submit("eval", {"run": "myrun"})
    b = jm.submit("eval", {"run": "myrun"})
    c = jm.submit("eval", {"run": "myrun", "device": "cpu"})
    assert jm.get(a.id).status == "running" and jm.get(b.id).status == "queued"
    assert jm.get(c.id).status == "running"  # the CPU lane is separate
    assert "waiting for 1 GPU job" in next(j for j in jm.list() if j["id"] == b.id)["note"]
    wait(jm, a, {"done"})
    b = wait(jm, b, {"done"})
    assert b.exit_code == 0 and b.started >= jm.get(a.id).ended - 0.5
    log = jm.read_log(b.id)
    assert log["done"] and "finished" in log["text"]


def test_failure_is_recorded(jm):
    j = wait(jm, jm.submit("score", {"path": "myrun", "suite": ["suites/s"]}), {"failed"})
    assert j.exit_code == 3 and "went wrong" in jm.read_log(j.id)["text"]


def test_training_stops_with_the_stop_file_and_links_its_run(jm):
    j = jm.submit("train", {"config": "configs/a.toml"})
    t0 = time.time()
    while j.id not in jm.stop_ready:
        assert time.time() - t0 < 20
        time.sleep(0.05)
    assert jm.get(j.id).run == "fake-run"
    jm.stop(j.id)
    j = wait(jm, j, {"stopped"})
    assert j.exit_code == 0 and "stopped at step" in jm.read_log(j.id)["text"]
    assert "✓" in jm.read_log(j.id)["text"]  # UTF-8 survives


def test_other_jobs_stop_with_a_signal_and_queued_ones_cancel(jm):
    g = jm.submit("generate", {"path": "myrun"})
    q = jm.submit("eval", {"run": "myrun"})
    wait(jm, g, {"running"})
    assert jm.stop(q.id).status == "cancelled"
    jm.stop(g.id)
    g = wait(jm, g, {"stopped"})
    assert g.exit_code != 0  # killed by the signal


def test_playground_holds_the_gpu_lane(jm):
    jm.acquire_gpu("playground")
    j = jm.submit("generate", {"path": "myrun"})  # runs until stopped
    assert jm.get(j.id).status == "queued"
    assert "Playground" in next(x for x in jm.list() if x["id"] == j.id)["note"]
    jm.release_gpu("playground")
    wait(jm, j, {"running"})
    with pytest.raises(GpuBusy, match="is running"):
        jm.acquire_gpu("playground")
    jm.stop(j.id)
    wait(jm, j, {"stopped"})
    jm.acquire_gpu("playground")


def test_log_tail_and_whole_lines(jm):
    j = wait(jm, jm.submit("eval", {"run": "myrun"}), {"done"})
    full = jm.read_log(j.id)["text"]
    tail = jm.read_log(j.id, offset=-10)
    assert full.endswith(tail["text"]) and tail["text"].endswith("\n") and len(tail["text"]) <= 10


def test_history_survives_a_restart(jm, tmp_path):
    done = wait(jm, jm.submit("eval", {"run": "myrun"}), {"done"})
    jm.acquire_gpu("playground")
    queued = jm.submit("eval", {"run": "myrun"})
    again = JobManager(jm.runs_dir, cwd=tmp_path, prefix=FAKE)
    assert again.get(done.id).status == "done" and again.get(done.id).exit_code == 0
    assert again.get(queued.id).status == "cancelled"


def test_real_training_job_stops_with_a_checkpoint_and_resumes(tiny_data, tmp_path):
    """The real `python -m sleuth train`, started and stopped the way the dashboard does it."""
    from sleuth.forge.train.checkpoint import latest_resume

    jm = JobManager(tmp_path / "runs", cwd=Path(__file__).resolve().parent.parent)
    overrides = [f"data.shards={tiny_data['shards'].as_posix()}", "train.device=cpu", "train.max_steps=100000",
                 "train.log_every=1", "train.eval_every=100000", "train.resume_every=100000",
                 "model.d_model=32", "model.n_layers=1", "train.seq_len=32", "train.micro_batch_size=2"]
    j = jm.submit("train", {"config": "configs/smoke-cpu.toml", "set": overrides})
    t0 = time.time()
    while j.id not in jm.stop_ready:
        assert jm.get(j.id).status == "running", jm.read_log(j.id)["text"]
        assert time.time() - t0 < 120
        time.sleep(0.1)
    time.sleep(0.5)  # let it take a few steps
    jm.stop(j.id)
    j = wait(jm, j, {"stopped", "failed"}, timeout=120)
    log = jm.read_log(j.id)["text"]
    assert j.status == "stopped" and j.exit_code == 0, log
    assert "stopped at step" in log and j.run
    assert latest_resume(jm.runs_dir / j.run) is not None
    r = jm.submit("resume", {"run": j.run, "stop_at_step": 100000})
    t0 = time.time()
    while "resumed from" not in jm.read_log(r.id)["text"]:
        assert jm.get(r.id).status == "running", jm.read_log(r.id)["text"]
        assert time.time() - t0 < 120
        time.sleep(0.1)
    jm.stop(r.id)
    wait(jm, r, {"stopped"}, timeout=120)
