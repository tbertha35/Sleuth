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

import heapq
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
        self.split_lines = True   # llama.cpp's Gemma 4 pre-tokenizer; from_hf (Gemma 3) turns it off

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
        for piece in (re.findall(r"[^\n]+|\n+", text) if self.split_lines else [text]):
            out.extend(self._bpe(piece))
        return out

    def _bpe(self, piece: str) -> list[int]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        if self.split_lines and piece.startswith("\n") and piece in self.token_to_id:
            ids = [self.token_to_id[piece]]  # a run of newlines that is one token (llama.cpp's Gemma 4 fix)
        else:
            parts = self._merge(piece)
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

    def _merge(self, piece: str) -> list[str]:
        """BPE: repeatedly merge the adjacent pair with the lowest rank (leftmost first).

        A heap of candidate pairs over a linked list of symbols, as Hugging Face's
        tokenizers and SentencePiece do it: O(n log n). Re-scanning the whole piece
        after every merge is O(n^2), fine for words but not for Gemma 3's pieces,
        which are whole lines or even whole documents."""
        sym = list(piece)
        n = len(sym)
        if n < 2:
            return sym
        nxt = list(range(1, n)) + [-1]
        prv = list(range(-1, n - 1))
        heap = []
        for i in range(n - 1):
            r = self.ranks.get((sym[i], sym[i + 1]))
            if r is not None:
                heap.append((r, i, sym[i], sym[i + 1]))
        heapq.heapify(heap)
        while heap:
            r, i, a, b = heapq.heappop(heap)
            j = nxt[i]
            if sym[i] != a or j < 0 or sym[j] != b:
                continue          # stale: one side was merged into something else since
            sym[i], sym[j] = a + b, ""
            nxt[i] = nxt[j]
            if nxt[j] >= 0:
                prv[nxt[j]] = i
            for left, right in ((prv[i], i), (i, nxt[i])):
                if left >= 0 and right >= 0:
                    r2 = self.ranks.get((sym[left], sym[right]))
                    if r2 is not None:
                        heapq.heappush(heap, (r2, left, sym[left], sym[right]))
        out, i = [], 0
        while i >= 0:
            out.append(sym[i])
            i = nxt[i]
        return out

    @classmethod
    def from_hf(cls, data: dict, config: dict | None = None) -> "GemmaTokenizer":
        """A SentencePiece-style tokenizer.json (Gemma 3's: spaces -> "▁", BPE over
        characters, byte fallback). Hugging Face doesn't split the text into lines
        first, so neither do we (split_lines=False)."""
        model, config = data["model"], config or {}
        added = {t["content"]: t for t in data.get("added_tokens") or []}
        size = max(list(model["vocab"].values()) + [t["id"] for t in added.values()]) + 1
        tokens, types = [""] * size, [UNUSED] * size
        for t, i in model["vocab"].items():
            tokens[i] = t
            types[i] = BYTE if _BYTE_TOKEN.fullmatch(t) else NORMAL
        for t, a in added.items():
            tokens[a["id"]] = t
            types[a["id"]] = CONTROL if a.get("special") else USER_DEFINED
        if model.get("unk_token") in model["vocab"]:
            types[model["vocab"][model["unk_token"]]] = UNKNOWN

        def tid(name):
            name = name.get("content") if isinstance(name, dict) else name
            return tokens.index(name) if name in tokens else None

        merges = [m if isinstance(m, str) else f"{m[0]} {m[1]}" for m in model.get("merges", [])]
        tok = cls(tokens, types, merges, tid(config.get("bos_token", "<bos>")), tid(config.get("eos_token", "<eos>")),
                  add_bos=config.get("add_bos_token", True))
        tok.split_lines = False
        return tok

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
