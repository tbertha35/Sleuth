import pytest

from sleuth.config import apply_overrides, from_dict, load_config
from sleuth.forge.train.config import RunConfig


def test_unknown_key_suggests_fix():
    with pytest.raises(KeyError, match="did you mean 'n_layers'"):
        from_dict(RunConfig, {"model": {"n_layer": 4}})


def test_type_errors():
    with pytest.raises(TypeError):
        from_dict(RunConfig, {"train": {"lr": "3e-4"}})
    cfg = from_dict(RunConfig, {"train": {"lr": 1}})  # int is fine where a float is expected
    assert cfg.train.lr == 1.0 and isinstance(cfg.train.lr, float)


def test_overrides():
    d = apply_overrides({}, ["model.n_layers=8", "train.lr=3e-4", "name=hello", "train.compile=true"])
    cfg = from_dict(RunConfig, d)
    assert cfg.model.n_layers == 8 and cfg.train.lr == 3e-4 and cfg.name == "hello" and cfg.train.compile


def test_shipped_configs_load():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "configs"
    files = sorted(root.glob("*.toml"))
    assert files, "expected example configs in configs/"
    for f in files:
        load_config(RunConfig, f)


def test_override_paths_windows_style():
    # Bare and single-quoted Windows paths work as-is...
    assert apply_overrides({}, [r"data.shards=data\shards\x"])["data"]["shards"] == r"data\shards\x"
    assert apply_overrides({}, [r"data.shards='C:\d\shards'"])["data"]["shards"] == r"C:\d\shards"
    # ...but a broken double-quoted one is an error, not a value with quote marks in it.
    with pytest.raises(ValueError, match="single quotes"):
        apply_overrides({}, [r'data.shards="data\shards\x"'])
