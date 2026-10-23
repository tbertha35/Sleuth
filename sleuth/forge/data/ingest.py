"""Find and read local training documents.

Supported inputs (anything else is ignored):
  * `.jsonl`: one JSON object per line; the text is in `text_field`
    (default "text"). An optional "category" field is kept.
  * `.txt`: plain text. If the file contains lines equal to the separator
    (default "<|endoftext|>", as used by TinyStories and GPT-2-era dumps),
    each part is one document; otherwise the whole file is one document.

Files and directories whose names start with "_" or "." are skipped, so
metadata like `_source.json` can sit next to the data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

DEFAULT_TXT_SEPARATOR = "<|endoftext|>"
SUFFIXES = {".txt", ".jsonl"}


@dataclass
class Document:
    text: str
    source: str
    category: str | None = None


def find_files(inputs: Iterable[str | Path]) -> list[Path]:
    """Expand files/directories into a sorted list of data files.
    Sorting makes every run see documents in the same order."""
    files: list[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                rel = f.relative_to(p).parts
                if f.is_file() and f.suffix in SUFFIXES and not any(part[0] in "._" for part in rel):
                    files.append(f)
        else:
            raise FileNotFoundError(p)
    return files


def iter_documents(files: Iterable[Path], text_field: str = "text",
                   txt_separator: str | None = DEFAULT_TXT_SEPARATOR) -> Iterator[Document]:
    for f in files:
        if f.suffix == ".jsonl":
            yield from _iter_jsonl(f, text_field)
        else:
            yield from _iter_txt(f, txt_separator)


def _iter_jsonl(path: Path, text_field: str) -> Iterator[Document]:
    with open(path, encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: bad JSON ({e})") from None
            text = obj.get(text_field) if isinstance(obj, dict) else None
            if isinstance(text, str) and text:
                yield Document(text, str(path), obj.get("category"))


def _iter_txt(path: Path, separator: str | None) -> Iterator[Document]:
    # newline=None: universal newlines, so \r\n files become \n like the rest.
    with open(path, encoding="utf-8", errors="replace", newline=None) as fh:
        buf: list[str] = []
        for line in fh:
            if separator is not None and line.strip() == separator:
                text = "".join(buf).strip("\n")
                if text:
                    yield Document(text, str(path))
                buf = []
            else:
                buf.append(line)
        text = "".join(buf).strip("\n")
        if text:
            yield Document(text, str(path))


def is_validation(text: str, val_fraction: float) -> bool:
    """Deterministic train/val split by content hash.

    Hashing the text (rather than e.g. "every 200th document") keeps a
    document in the same split even if you add files or reorder them, so
    the validation set never leaks into training across re-shardings.
    """
    if val_fraction <= 0:
        return False
    h = int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")
    return h / 2**64 < val_fraction
