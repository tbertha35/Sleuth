"""`sleuth experiment` (forge/experiment.py): variants of one config, trained and
measured the same way, one table."""

import json

from sleuth.forge.experiment import Experiment, run_experiment


def _write(tmp_path, tiny_data):
    base = tmp_path / "base.toml"
    shards = tiny_data["shards"].as_posix()
    runs = (tmp_path / "runs").as_posix()
    # forward slashes: TOML strings built from Windows paths would need escaping
    base.write_bytes(f"""name = "tiny"
runs_dir = "{runs}"
[data]
shards = "{shards}"
[model]
d_model = 32
n_layers = 2
n_heads = 4
n_kv_heads = 2
max_seq_len = 64
[train]
device = "cpu"
micro_batch_size = 4
seq_len = 32
max_steps = 4
warmup_steps = 2
log_every = 2
eval_every = 2
eval_batches = 2
eval_windows = 8
snapshot_every = 2
resume_every = 2
compile = false
""".encode())
    exp = tmp_path / "exp.toml"
    exp.write_bytes(f"""name = "t"
base = "{base.as_posix()}"
suite = "suites/basic"
set = ["train.lr=3e-3"]

[[variant]]
name = "baseline"

[[variant]]
name = "mtp1"
set = ["model.mtp_heads=1"]
""".encode())
    return exp


def test_variants_get_the_shared_and_their_own_overrides(tmp_path, tiny_data):
    exp = Experiment.load(_write(tmp_path, tiny_data))
    a, b = (exp.run_config(v) for v in exp.variants)
    assert a.name == "t-baseline" and b.name == "t-mtp1"
    assert a.train.lr == b.train.lr == 3e-3
    assert a.model.mtp_heads == 0 and b.model.mtp_heads == 1


def test_runs_measures_and_skips_finished_variants(tmp_path, tiny_data):
    path = _write(tmp_path, tiny_data)
    calls = []

    def counting_train(cfg, **kw):
        from sleuth.forge.train.trainer import train

        calls.append(cfg.name)
        return train(cfg, log=lambda s: None, **kw)

    out = run_experiment(path, log=lambda s: None, train_fn=counting_train)
    state = json.loads((out / "results.json").read_text())
    assert calls == ["t-baseline", "t-mtp1"]
    for name in ("baseline", "mtp1"):
        r = state["results"][name]
        assert r["val_bpb"] > 0 and r["step"] == 4 and "overall" in r["knowledge"]
    assert state["results"]["mtp1"]["params"] > state["results"]["baseline"]["params"]
    md = (out / "results.md").read_text(encoding="utf-8")
    assert "| baseline |" in md and "| mtp1 |" in md
    run_experiment(path, log=lambda s: None, train_fn=counting_train)   # everything measured: nothing retrained
    assert len(calls) == 2
