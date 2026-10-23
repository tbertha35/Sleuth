"""Load test suites: a directory of <category>.jsonl files (see suites/README.md)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestCase:
    __test__ = False  # tells pytest this isn't a test class despite the name
    id: str
    category: str
    prompt: str
    expected: str | None = None

    @property
    def text(self) -> str:
        return self.prompt + (self.expected or "")


def load_suite(path: str | Path) -> list[TestCase]:
    p = Path(path)
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(f"no .jsonl files in {p}")
    cases: list[TestCase] = []
    for f in files:
        for lineno, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            d = json.loads(line)
            if "prompt" not in d:
                raise ValueError(f"{f}:{lineno}: missing 'prompt'")
            cases.append(TestCase(
                id=d.get("id", f"{f.stem}-{lineno}"),
                category=d.get("category", f.stem),
                prompt=d["prompt"],
                expected=d.get("expected"),
            ))
    return cases
