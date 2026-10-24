"""Gemma 4's tokenizer, as stored in a GGUF file (tokenizer.ggml.model = "gemma4").

It's BPE, but SentencePiece-flavoured rather than GPT-2-flavoured. Our
implementation follows llama.cpp's (llama-vocab.cpp, LLAMA_VOCAB_PRE_TYPE_GEMMA4):

  1. special tokens are cut out of the text first: user-defined ones always,
     control ones (<bos>, <turn|>, ...) only if allowed (see `encode`)
  2. every space becomes "▁" (U+2581), so spaces are ordinary characters
  3. the text is split into lines: runs of non-newlines, and runs of newlines.
     (No splitting into words: merges may cross word boundaries.) A run of
     newlines that is itself a token becomes that token directly.
  4. BPE on each piece, starting from single Unicode characters (not bytes, as
     in GPT-2's byte-level BPE): repeatedly merge the adjacent pair with the
     lowest rank in tokenizer.ggml.merges
  5. any piece left that isn't in the vocabulary falls back to byte tokens
     "<0xE2>", "<0x96>", ... (one per UTF-8 byte), so nothing is ever unknown

Decoding reverses it: "▁" -> space, byte tokens -> their byte.

Token types (tokenizer.ggml.token_type): 1 normal, 2 unknown, 3 control,
4 user-defined, 5 unused, 6 byte.
"""

from __future__ import annotations

import re
from typing import Iterable

SPACE = "▁"
NORMAL, UNKNOWN, CONTROL, USER_DEFINED, UNUSED, BYTE = 1, 2, 3, 4, 5, 6
_BYTE_TOKEN = re.compile(r"<0x([0-9A-Fa-f]{2})>")


class GemmaTokenizer:
    """Same interface as our Tokenizer / HFTokenizer (encode, decode, bos_id, ...)."""

    def __init__(self, tokens: list[str], types: list[int], merges: list[str],
                 bos_id: int | None, eos_id: int | None, add_bos: bool = True):
        self.id_to_token = list(tokens)
        self.types = list(types) if types else [NORMAL] * len(tokens)
        self.token_to_id = {t: i for i, t in enumerate(tokens)}
        self.ranks: dict[tuple[str, str], int] = {}
        for i, m in enumerate(merges):
            # As llama.cpp: split at the first space after the first character
            # (a merge's left part may itself be a lone space-like character).
            j = m.find(" ", 1)
            if j > 0:
                self.ranks.setdefault((m[:j], m[j + 1:]), i)
        self.byte_ids = {}
        for i, t in enumerate(tokens):
            if self.types[i] == BYTE:
                b = _BYTE_TOKEN.fullmatch(t)
                if b:
                    self.byte_ids[int(b.group(1), 16)] = i
        self.special_tokens = {t: i for i, t in enumerate(tokens) if self.types[i] in (CONTROL, UNKNOWN)}
        self.user_defined = {t: i for i, t in enumerate(tokens) if self.types[i] == USER_DEFINED}
        self.added_tokens = {**self.special_tokens, **self.user_defined}
        self.bos_id = bos_id if add_bos else None
        self.eos_id = eos_id
        self._cache: dict[str, list[int]] = {}
        self._splitters: dict[frozenset, re.Pattern | None] = {}

    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def start_id(self) -> int | None:
        return self.bos_id if self.bos_id is not None else self.eos_id

    def _splitter(self, allowed: frozenset) -> re.Pattern | None:
        """Regex matching the special tokens to cut out (longest first, as llama.cpp sorts them)."""
        if allowed not in self._splitters:
            names = sorted(set(self.user_defined) | allowed, key=len, reverse=True)
            self._splitters[allowed] = re.compile("(" + "|".join(map(re.escape, names)) + ")") if names else None
        return self._splitters[allowed]

    def encode(self, text: str, allowed_special: str | Iterable[str] = "none") -> list[int]:
        """Text -> token IDs (no <bos>; callers add `start_id` themselves, as for our other tokenizers).
        Control tokens in the text are recognized only if allowed ("all", or a set of strings)."""
        if allowed_special == "none":
            allowed = frozenset()
        elif allowed_special == "all":
            allowed = frozenset(self.special_tokens)
        else:
            allowed = frozenset(allowed_special) & frozenset(self.special_tokens)
        rx = self._splitter(allowed)
        parts = rx.split(text) if rx else [text]
        out: list[int] = []
        for i, part in enumerate(parts):
            if i % 2 == 1:
                out.append(self.added_tokens[part])
            elif part:
                out.extend(self.encode_ordinary(part))
        return out

    def encode_ordinary(self, text: str) -> list[int]:
        text = text.replace(" ", SPACE)
        out: list[int] = []
        for piece in re.findall(r"[^\n]+|\n+", text):
            out.extend(self._bpe(piece))
        return out

    def _bpe(self, piece: str) -> list[int]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        if piece.startswith("\n") and piece in self.token_to_id:
            ids = [self.token_to_id[piece]]  # a run of newlines that is one token (llama.cpp's Gemma 4 fix)
        else:
            parts = list(piece)
            while len(parts) > 1:
                best, best_rank = None, None
                for i in range(len(parts) - 1):
                    r = self.ranks.get((parts[i], parts[i + 1]))
                    if r is not None and (best_rank is None or r < best_rank):
                        best, best_rank = i, r
                if best is None:
                    break
                a, b = parts[best], parts[best + 1]
                merged, i = [], 0
                while i < len(parts):  # merge every occurrence of this pair, left to right
                    if i < len(parts) - 1 and parts[i] == a and parts[i + 1] == b:
                        merged.append(a + b)
                        i += 2
                    else:
                        merged.append(parts[i])
                        i += 1
                parts = merged
            ids = []
            for p in parts:
                tid = self.token_to_id.get(p)
                if tid is not None:
                    ids.append(tid)
                else:  # byte fallback
                    ids.extend(self.byte_ids[b] for b in p.encode("utf-8") if b in self.byte_ids)
        if len(self._cache) > 200_000:
            self._cache.clear()
        self._cache[piece] = ids
        return ids

    # ------------------------------------------------------------------ #
    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        out = bytearray()
        for i in ids:
            if not 0 <= i < len(self.id_to_token):
                raise ValueError(f"token id {i} is not in the vocabulary")
            t, kind = self.id_to_token[i], self.types[i]
            if kind == BYTE:
                b = _BYTE_TOKEN.fullmatch(t)
                out.append(int(b.group(1), 16) if b else 0)
            elif kind == NORMAL:
                out += t.replace(SPACE, " ").encode("utf-8")
            else:
                out += t.encode("utf-8")  # control / user-defined: shown as written
        return bytes(out)

    def decode(self, ids: Iterable[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def token_byte_lengths(self) -> list[int]:
        return [len(t.replace(SPACE, " ").encode("utf-8")) if k == NORMAL else (1 if k == BYTE else 0)
                for t, k in zip(self.id_to_token, self.types)]
