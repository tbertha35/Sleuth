"""Byte-level BPE tokenizer: encode, decode, save, load.

How it works:
  * Every string is UTF-8 bytes, so the base vocabulary is the 256 byte
    values (IDs 0-255). Any text can be encoded; there is no "unknown" token.
  * Training (train_bpe.py) learned an ordered list of *merges*: "the pair
    (a, b) becomes new token 256 + i". Merge i has *rank* i; lower rank means
    it was learned earlier, because it was more frequent.
  * Encoding a chunk: start from its bytes, then repeatedly merge the
    adjacent pair with the lowest rank until no learned pair is left. That
    replays the training merges in order, so encode gives the same splits
    the trainer saw.
  * Special tokens (<|bos|>, chat roles, ...) get IDs after the merges. They
    are never produced from ordinary text unless you explicitly allow it.
    That way a document containing the literal string "<|eos|>" can't
    inject a real end-of-sequence token.

File format (`tokenizer.json`, ours, human-readable):
    {"format": "sleuth-bpe", "version": 1, "pattern": "...",
     "merges": [[a, b], ...], "special_tokens": {"<|bos|>": 1234, ...},
     "train_info": {...}}
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from .pretokenize import PATTERN, compile_pattern

FORMAT_NAME = "sleuth-bpe"
FORMAT_VERSION = 1

# Reserved up front so later stages (chat SFT, tools, MTP experiments) can use
# them without retraining the tokenizer or changing the vocab size.
SPECIAL_TOKENS = [
    "<|bos|>", "<|eos|>", "<|pad|>",
    "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
] + [f"<|reserved_{i}|>" for i in range(9)]


class Tokenizer:
    def __init__(self, merges: list[tuple[int, int]], special_tokens: dict[str, int],
                 pattern: str = PATTERN, train_info: dict | None = None):
        self.merges = [tuple(m) for m in merges]
        self.pattern = pattern
        self.special_tokens = dict(special_tokens)
        self.train_info = train_info or {}
        self._regex = compile_pattern(pattern)

        # ranks: pair -> the token it merges into. The new token IDs are
        # 256, 257, ... in merge order, so lower ID == lower rank.
        self.ranks: dict[tuple[int, int], int] = {}
        self.vocab: list[bytes] = [bytes([i]) for i in range(256)]
        for i, (a, b) in enumerate(self.merges):
            self.ranks[(a, b)] = 256 + i
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.special_ids = {i: s for s, i in self.special_tokens.items()}
        expected_first = len(self.vocab)
        if self.special_tokens and min(self.special_tokens.values()) != expected_first:
            raise ValueError("special token IDs must start right after the merges")
        self._special_regex = (
            re.compile("(" + "|".join(re.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True)) + ")")
            if self.special_tokens else None
        )
        # Encoding the same word again is common (think " the"), so cache it.
        self._cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    def special(self, name: str) -> int:
        return self.special_tokens[name]

    @property
    def bos_id(self) -> int:
        return self.special_tokens["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.special_tokens["<|eos|>"]

    @property
    def pad_id(self) -> int:
        return self.special_tokens["<|pad|>"]

    # ------------------------------------------------------------------ #
    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Find the adjacent pair that was learned earliest.
            best = None
            best_rank = None
            for pair in zip(ids, ids[1:]):
                r = self.ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = pair, r
            if best is None:
                break
            ids = _merge(ids, best, best_rank)
        if len(self._cache) > 500_000:  # bound memory on huge corpora
            self._cache.clear()
        self._cache[chunk] = ids
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text, treating special-token strings as plain text."""
        out: list[int] = []
        for chunk in self._regex.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(self, text: str, allowed_special: str | Iterable[str] = "none") -> list[int]:
        """Encode text.

        allowed_special: "none" (default, safest), "all", or a set of special
        token strings that may be recognised in `text`.
        """
        if allowed_special == "none" or not self._special_regex:
            return self.encode_ordinary(text)
        allowed = set(self.special_tokens) if allowed_special == "all" else set(allowed_special)
        out: list[int] = []
        for i, part in enumerate(self._special_regex.split(text)):
            # re.split with a capture group alternates text, special, text, ...
            if i % 2 == 1 and part in allowed:
                out.append(self.special_tokens[part])
            elif part:
                out.extend(self.encode_ordinary(part))
        return out

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        parts = []
        for i in ids:
            if i < len(self.vocab):
                parts.append(self.vocab[i])
            elif i in self.special_ids:
                parts.append(self.special_ids[i].encode("utf-8"))
            else:
                raise ValueError(f"token id {i} out of range (vocab size {self.vocab_size})")
        return b"".join(parts)

    def decode(self, ids: Iterable[int]) -> str:
        # A token sequence can end mid-character (e.g. while streaming
        # generation); "replace" shows U+FFFD instead of crashing.
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def token_byte_lengths(self) -> list[int]:
        """Raw-text bytes each token stands for (0 for special tokens).
        Used to convert loss into bits-per-byte."""
        return [len(b) for b in self.vocab] + [0] * len(self.special_tokens)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "pattern": self.pattern,
            "merges": [list(m) for m in self.merges],
            "special_tokens": self.special_tokens,
            "train_info": self.train_info,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("format") != FORMAT_NAME:
            raise ValueError(f"{path} is not a {FORMAT_NAME} tokenizer file")
        if d.get("version") != FORMAT_VERSION:
            raise ValueError(f"unsupported tokenizer version {d.get('version')}")
        return cls(d["merges"], d["special_tokens"], d["pattern"], d.get("train_info"))

    def fingerprint(self) -> str:
        """Short hash of what defines the encoding (not the train_info), so
        shards can check they were made with this exact tokenizer."""
        core = json.dumps([self.pattern, self.merges, self.special_tokens], sort_keys=True)
        return hashlib.sha256(core.encode()).hexdigest()[:16]


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair`, left to right."""
    out = []
    i = 0
    n = len(ids)
    a, b = pair
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
