"""Find and read local training documents.

Supported inputs (anything else is ignored):
  * `.jsonl`: one JSON object per line; the text is in `text_field`
    (default "text"). An optional "category" field is kept.
  * `.txt`: plain text. If the file contains lines equal to the separator
    (default "<|endoftext|>", as used by TinyStories and GPT-2-era dumps),
    each part is one document; otherwise the whole file is one document.
  * `.parquet`: a column of texts (`text_field`), as Hugging Face datasets
    publish them (FineWeb-Edu, Wikipedia, FineMath, ...). Needs the optional
    `pyarrow` package (approved 2026-10-23 for exactly this). Read in batches,
    so a 2 GB file doesn't need 2 GB of RAM.

Files and directories whose names start with "_" or "." are skipped, so
metadata like `_source.json` can sit next to the data.

Mixing sources: an input can carry a size cap, `path@MB`, e.g.
`data/raw/fineweb-edu@3000 data/raw/finemath@600`: read at most that many
megabytes of text from it. That's how you set a data mix's proportions, and
how a tokenizer gets trained on a sample of *every* source (a single global
cap would only ever read the first one). Documents without a category of
their own are labelled with their input's folder name, so `meta.json` reports
how much of each source went in.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

DEFAULT_TXT_SEPARATOR = "<|endoftext|>"
SUFFIXES = {".txt", ".jsonl", ".parquet"}


@dataclass
class Document:
    text: str
    source: str
    category: str | None = None


def parse_input(spec: str | Path) -> tuple[Path, int | None]:
    """"data/raw/wiki@500" -> (Path("data/raw/wiki"), 500 MB in bytes); no "@N" -> no cap."""
    s = str(spec)
    if "@" in s:
        path, cap = s.rsplit("@", 1)
        try:
            return Path(path), int(float(cap) * 1e6)
        except ValueError:
            pass  # an "@" that is part of the path itself
    return Path(s), None


def find_files(inputs: Iterable[str | Path]) -> list[Path]:
    """Expand files/directories into a sorted list of data files.
    Sorting makes every run see documents in the same order."""
    files: list[Path] = []
    for inp in inputs:
        p, _ = parse_input(inp)
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
        elif f.suffix == ".parquet":
            yield from _iter_parquet(f, text_field)
        else:
            yield from _iter_txt(f, txt_separator)


def iter_input_documents(inputs: Iterable[str | Path], text_field: str = "text",
                         txt_separator: str | None = DEFAULT_TXT_SEPARATOR) -> Iterator[Document]:
    """Documents from each input in turn, honouring its `@MB` cap, labelled
    with the input's name when they have no category of their own."""
    for spec in inputs:
        path, cap = parse_input(spec)
        name = path.stem if path.is_file() else path.name
        seen = 0
        for d in iter_documents(find_files([path]), text_field, txt_separator):
            n = len(d.text.encode("utf-8"))
            if cap is not None and seen + n > cap:
                break
            seen += n
            yield d if d.category else Document(d.text, d.source, name)


def _iter_parquet(path: Path, text_field: str) -> Iterator[Document]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(f"reading {path.name} needs pyarrow: pip install pyarrow "
                          f"(or: pip install -e .[data])") from None
    pf = pq.ParquetFile(path)
    if text_field not in pf.schema_arrow.names:
        raise ValueError(f"{path}: no column {text_field!r}; it has {pf.schema_arrow.names} (use --text-field)")
    for batch in pf.iter_batches(batch_size=1024, columns=[text_field]):
        for text in batch.column(0).to_pylist():
            if isinstance(text, str) and text:
                yield Document(text, str(path))


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
