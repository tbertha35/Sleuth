"""Helpers shared by tests (pytest puts tests/ on sys.path)."""

from pathlib import Path


def tiny_run_config(tiny_data, runs_dir: Path, **train_overrides):
    from sleuth.forge.model.config import ModelConfig
    from sleuth.forge.train.config import DataConfig, RunConfig, TrainConfig

    tc = dict(device="cpu", micro_batch_size=4, seq_len=32, max_steps=6, lr=1e-3, warmup_steps=2,
              log_every=2, eval_every=3, eval_batches=2, snapshot_every=3, resume_every=3, keep_resume=5)
    tc.update(train_overrides)
    return RunConfig(
        name="test", runs_dir=str(runs_dir),
        model=ModelConfig(d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64),
        train=TrainConfig(**tc),
        data=DataConfig(shards=str(tiny_data["shards"])),
    )
