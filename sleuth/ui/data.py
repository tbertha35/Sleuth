"""What data is on disk: the dashboard's Data page and `sleuth data`.

Lists the folders under the data root (raw text, fine-tuning sets, shards,
tokenizers) with their sizes, and for shard folders the summary in their
meta.json: tokens per split, vocab, and how much each source contributed.
Read-only, and it never opens the data files themselves (some are GBs).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

SECTIONS = [
    ("raw", "Pretraining text (sleuth shard reads these)"),
    ("finetune", "Fine-tuning and memory datasets (kept apart so shard can't pick them up)"),
    ("shards", "Tokenized training shards"),
    ("tokenizers", "Tokenizers"),
]


def _walk(d: Path, limit: int = 200_000) -> tuple[int, int, float]:
    """(bytes, files, newest mtime), stopping after `limit` files."""
    size = files = 0
    newest = 0.0
    for root, _, names in os.walk(d):
        for n in names:
            try:
                st = os.stat(os.path.join(root, n))
            except OSError:
                continue
            size += st.st_size
            files += 1
            newest = max(newest, st.st_mtime)
            if files >= limit:
                return size, files, newest
    return size, files, newest


def _shard_summary(d: Path) -> dict | None:
    try:
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    cat_bytes = meta.get("category_bytes") or {}
    total = sum(cat_bytes.values()) or 1
    return {
        "train_tokens": meta.get("train_tokens"), "val_tokens": meta.get("val_tokens"),
        "vocab_size": meta.get("vocab_size"), "bytes_per_token": meta.get("bytes_per_token"),
        "tokenizer": meta.get("tokenizer_source"), "source_files": len(meta.get("source_files") or []),
        "sources": [{"name": k, "bytes": v, "share": v / total} for k, v in sorted(cat_bytes.items(), key=lambda kv: -kv[1])],
    }


def describe_data(data_dir: str | Path) -> dict:
    root = Path(data_dir)
    out = {"root": str(root), "exists": root.is_dir(), "sections": []}
    for name, title in SECTIONS:
        d = root / name
        entries = []
        if d.is_dir():
            for e in sorted(d.iterdir(), key=lambda p: p.name.lower()):
                if e.name.startswith("."):
                    continue
                if e.is_dir():
                    size, files, mtime = _walk(e)
                else:
                    st = e.stat()
                    size, files, mtime = st.st_size, 1, st.st_mtime
                item = {"name": e.name, "bytes": size, "files": files, "modified": mtime, "dir": e.is_dir()}
                if name == "shards" and e.is_dir():
                    item["shards"] = _shard_summary(e)
                if name == "tokenizers" and e.suffix == ".json":
                    try:
                        tok = json.loads(e.read_text(encoding="utf-8"))
                        item["vocab_size"] = tok.get("vocab_size") or (256 + len(tok.get("merges", [])) +
                                                                      len(tok.get("special_tokens", {})))
                    except (OSError, ValueError):
                        pass
                entries.append(item)
        out["sections"].append({"name": name, "title": title, "exists": d.is_dir(), "entries": entries})
    return out
