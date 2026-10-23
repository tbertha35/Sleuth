"""Config files: TOML -> dataclasses, with command-line overrides.

Every run is described by one TOML file (see configs/). Loading is strict:
an unknown key is an error, with a "did you mean" hint, because a typo
like `n_layer = 12` silently doing nothing is a classic way to waste a
day of GPU time.

Overrides use dotted paths and TOML value syntax:
    --set model.n_layers=8 --set train.lr=3e-4 --set name='"test"'
(Strings may also be given bare: --set name=test.)
"""

from __future__ import annotations

import dataclasses
import difflib
import tomllib
import types
import typing
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def load_toml(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _parse_value(text: str) -> Any:
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError as e:
        if text[:1] in "\"'[{":
            # It looked like a TOML value but isn't valid; guessing would
            # silently keep the quote marks as part of the value.
            hint = (" Backslashes inside double quotes are escape sequences in TOML; "
                    "use single quotes ('C:\\data') or forward slashes.") if "\\" in text else ""
            raise ValueError(f"can't parse override value {text}: {e}.{hint}") from None
        return text  # bare string, e.g. --set data.shards=data/shards/x


def apply_overrides(d: dict, overrides: list[str]) -> dict:
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override {ov!r} must look like key.path=value")
        key, val = ov.split("=", 1)
        parts = key.strip().split(".")
        node = d
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(val.strip())
    return d


def from_dict(cls: type[T], d: dict, where: str = "") -> T:
    """Build dataclass `cls` from a dict, recursing into nested dataclasses."""
    if not isinstance(d, dict):
        raise TypeError(f"{where or cls.__name__}: expected a table, got {type(d).__name__}")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, val in d.items():
        if key not in fields:
            close = difflib.get_close_matches(key, fields, n=1)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            raise KeyError(f"unknown config key {where + key!r}{hint}")
        typ = hints[key]
        if dataclasses.is_dataclass(typ):
            val = from_dict(typ, val, where + key + ".")
        else:
            val = _coerce(val, typ, where + key)
        kwargs[key] = val
    return cls(**kwargs)


def _coerce(val: Any, typ: Any, name: str) -> Any:
    """Light type checking: catches `lr = "3e-4"` (a string) and lets an int
    stand in for a float."""
    origin = typing.get_origin(typ)
    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(typ)
        for a in args:
            try:
                return _coerce(val, a, name)
            except TypeError:
                pass
        raise TypeError(f"{name}: {val!r} doesn't match {typ}")
    if typ is float and isinstance(val, int) and not isinstance(val, bool):
        return float(val)
    if typ in (int, float, str, bool):
        if not isinstance(val, typ) or (typ is int and isinstance(val, bool)):
            raise TypeError(f"{name}: expected {typ.__name__}, got {val!r}")
        return val
    if origin is list:
        if not isinstance(val, list):
            raise TypeError(f"{name}: expected a list, got {val!r}")
        (inner,) = typing.get_args(typ) or (Any,)
        return [_coerce(v, inner, name) for v in val] if inner is not Any else val
    if typ is type(None):
        if val is not None:
            raise TypeError(f"{name}: expected None")
        return val
    return val


def to_dict(obj: Any) -> dict:
    return dataclasses.asdict(obj)


def load_config(cls: type[T], path: str | Path | None = None, overrides: list[str] | None = None) -> T:
    d = load_toml(path) if path else {}
    apply_overrides(d, overrides or [])
    return from_dict(cls, d)
