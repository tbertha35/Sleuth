"""Our own reader for Hugging Face `tokenizer.json` files (byte-level BPE).

Covers the GPT-2-style byte-level BPE used by Qwen (2, 2.5, 3), Llama 3, GPT-2,
Mistral-NeMo and many others. SentencePiece-style tokenizers (Gemma, Llama 2)
work differently and are handled elsewhere.

How a byte-level BPE `tokenizer.json` works:
  * normalizer (optional): e.g. Unicode NFC, so "é" typed two ways encodes the same
  * pre-tokenizer: a regex splits text into chunks (like our pretokenize.py)
  * every chunk's UTF-8 bytes are mapped to printable characters by a fixed
    table (`bytes_to_unicode`, from GPT-2: byte 0x20, a space, becomes "Ġ").
    The vocab and merges are written in these characters.
  * BPE: repeatedly merge the adjacent pair with the lowest rank in `merges`
  * "added tokens" (special tokens like <|im_start|>) are matched as whole
    strings before any of this

The pre-tokenizer regexes use Unicode classes (\\p{L} letters, \\p{N}
numbers) that Python's `re` lacks. `translate_regex` rewrites them into
equivalent `re` syntax, including inside negated classes like [^\\s\\p{L}\\p{N}].
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable

# --------------------------------------------------------------------------- #
# \p{L} / \p{N} -> Python re
# --------------------------------------------------------------------------- #

LETTER = r"[^\W\d_]"     # word characters that aren't digits or underscore
NUMBER = r"\d"
_CLASS_OF = {"L": LETTER, "N": NUMBER}
_EXCLUDE_ANY = {frozenset("L"): LETTER, frozenset("N"): NUMBER, frozenset("LN"): r"[^\W_]"}


def translate_regex(pattern: str) -> str:
    """Rewrite \\p{L} and \\p{N} (inside and outside [...]) into Python re syntax."""
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        if c == "\\" and pattern.startswith("\\p{", i):
            j = pattern.index("}", i)
            out.append(_unicode_class(pattern[i + 3:j]))
            i = j + 1
        elif c == "\\":
            out.append(pattern[i:i + 2])
            i += 2
        elif c == "[":
            j, body, negated = _read_class(pattern, i)
            out.append(_rewrite_class(body, negated))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _unicode_class(name: str) -> str:
    if name not in _CLASS_OF:
        raise ValueError(f"unsupported Unicode class \\p{{{name}}} in tokenizer regex")
    return _CLASS_OF[name]


def _read_class(p: str, start: int) -> tuple[int, str, bool]:
    """Parse a [...] class starting at p[start]; return (end index, body, negated)."""
    i = start + 1
    negated = p[i:i + 1] == "^"
    if negated:
        i += 1
    body_start = i
    if p[i:i + 1] == "]":  # a literal "]" right after "[" or "[^"
        i += 1
    while p[i] != "]":
        i += 2 if p[i] == "\\" else 1
    return i + 1, p[body_start:i], negated


def _rewrite_class(body: str, negated: bool) -> str:
    props = set(re.findall(r"\\p\{(\w+)\}", body))
    rest = re.sub(r"\\p\{\w+\}", "", body)
    if not props:
        return "[" + ("^" if negated else "") + body + "]"
    for name in props:
        _unicode_class(name)  # validates
    if not negated:
        parts = ([f"[{rest}]"] if rest else []) + [_CLASS_OF[k] for k in sorted(props)]
        return "(?:" + "|".join(parts) + ")"
    # [^ rest \p{...}]: any character that is not in `rest` and not in the classes.
    excluded = _EXCLUDE_ANY[frozenset(props)]
    base = f"[^{rest}]" if rest else r"[\s\S]"
    return f"(?:(?!{excluded}){base})"


# --------------------------------------------------------------------------- #
# GPT-2 byte <-> printable character table
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's table: printable bytes map to themselves, the rest (spaces,
    control bytes, ...) to unused code points from 256 up, so every byte
    becomes a visible, non-whitespace character."""
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    table = {b: chr(b) for b in keep}
    extra = 0
    for b in range(256):
        if b not in table:
            table[b] = chr(256 + extra)
            extra += 1
    return table


GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #

class HFTokenizer:
    """Same interface as our own Tokenizer (encode/decode/bos_id/...)."""

    def __init__(self, data: dict, config: dict | None = None):
        model = data.get("model") or {}
        if model.get("type") != "BPE":
            raise ValueError(f"only BPE tokenizer.json files are supported (got {model.get('type')!r})")
        self.vocab: dict[str, int] = model["vocab"]
        merges = model.get("merges", [])
        pairs = [tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m) for m in merges]
        self.ranks = {p: i for i, p in enumerate(pairs)}
        self.ignore_merges = bool(model.get("ignore_merges", False))
        self.byte_fallback = bool(model.get("byte_fallback", False))
        self._normalize = _normalizer(data.get("normalizer"))
        self._splits = _pre_tokenizer(data.get("pre_tokenizer"))
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        added = data.get("added_tokens") or []
        self.added_tokens = {t["content"]: t["id"] for t in added}
        self.special_tokens = {t["content"]: t["id"] for t in added if t.get("special")}
        size = max(list(self.vocab.values()) + list(self.added_tokens.values())) + 1
        self.id_to_token: list[str | None] = [None] * size
        for tok, i in self.vocab.items():
            self.id_to_token[i] = tok
        for tok, i in self.added_tokens.items():
            self.id_to_token[i] = tok
        self._added_ids = set(self.added_tokens.values())
        self._added_regex = (re.compile("(" + "|".join(re.escape(t) for t in sorted(self.added_tokens, key=len, reverse=True)) + ")")
                             if self.added_tokens else None)
        self._cache: dict[str, list[int]] = {}

        config = config or {}
        self.bos_id = self._token_id(config.get("bos_token"))
        self.eos_id = self._token_id(config.get("eos_token"))
        if self.eos_id is None:
            for guess in ("<|endoftext|>", "<|end_of_text|>", "</s>"):
                if guess in self.added_tokens:
                    self.eos_id = self.added_tokens[guess]
                    break

    def _token_id(self, tok) -> int | None:
        if isinstance(tok, dict):  # tokenizer_config.json sometimes stores {"content": ...}
            tok = tok.get("content")
        if not tok:
            return None
        return self.added_tokens.get(tok, self.vocab.get(tok))

    @classmethod
    def load(cls, path: str | Path) -> "HFTokenizer":
        """Path to tokenizer.json, or to a model directory containing it (and
        optionally tokenizer_config.json, which names the bos/eos tokens)."""
        p = Path(path)
        d = p if p.is_dir() else p.parent
        f = p / "tokenizer.json" if p.is_dir() else p
        cfg_path = d / "tokenizer_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        return cls(json.loads(f.read_text(encoding="utf-8")), cfg)

    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def start_id(self) -> int | None:
        """Token the analyzer puts before every text: BOS if the model has one;
        otherwise the document separator it saw in pretraining (<|endoftext|>
        for Qwen), falling back to EOS."""
        if self.bos_id is not None:
            return self.bos_id
        for t in ("<|endoftext|>", "<|end_of_text|>"):
            if t in self.added_tokens:
                return self.added_tokens[t]
        return self.eos_id

    def _bpe(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        word = "".join(self.byte_encoder[b] for b in chunk.encode("utf-8"))
        if self.ignore_merges and word in self.vocab:
            ids = [self.vocab[word]]
        else:
            parts = list(word)
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
            ids = [self.vocab[p] for p in parts]
        if len(self._cache) > 200_000:
            self._cache.clear()
        self._cache[chunk] = ids
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        text = self._normalize(text)
        out: list[int] = []
        for piece in self._splits(text):
            out.extend(self._bpe(piece))
        return out

    def encode(self, text: str, allowed_special: str | Iterable[str] = "none") -> list[int]:
        """Like our Tokenizer: special/added token strings in `text` are only
        recognized if allowed ("all", or a set of token strings)."""
        if allowed_special == "none" or self._added_regex is None:
            return self.encode_ordinary(text)
        allowed = set(self.added_tokens) if allowed_special == "all" else set(allowed_special)
        out: list[int] = []
        for i, part in enumerate(self._added_regex.split(text)):
            if i % 2 == 1 and part in allowed:
                out.append(self.added_tokens[part])
            elif part:
                out.extend(self.encode_ordinary(part))
        return out

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        out = bytearray()
        for i in ids:
            tok = self.id_to_token[i] if 0 <= i < len(self.id_to_token) else None
            if tok is None:
                raise ValueError(f"token id {i} is not in the vocabulary")
            if i in self._added_ids:
                out += tok.encode("utf-8")
            else:
                out += bytes(self.byte_decoder.get(ch, 0) for ch in tok)
        return bytes(out)

    def decode(self, ids: Iterable[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def token_byte_lengths(self) -> list[int]:
        return [0 if (t is None or i in self._added_ids) else len(t) for i, t in enumerate(self.id_to_token)]


def _normalizer(spec: dict | None):
    if not spec:
        return lambda s: s
    kind = spec.get("type")
    if kind in ("NFC", "NFKC", "NFD", "NFKD"):
        return lambda s: unicodedata.normalize(kind, s)
    if kind == "Sequence":
        fns = [_normalizer(x) for x in spec.get("normalizers", [])]

        def seq(s):
            for f in fns:
                s = f(s)
            return s
        return seq
    if kind == "Lowercase":
        return str.lower
    raise ValueError(f"unsupported tokenizer normalizer {kind!r}")


def _pre_tokenizer(spec: dict | None):
    """Return a function text -> list of chunks."""
    if not spec:
        return lambda s: [s] if s else []
    kind = spec.get("type")
    if kind == "Sequence":
        steps = [_pre_tokenizer(x) for x in spec.get("pretokenizers", [])]

        def seq(s):
            chunks = [s]
            for step in steps:
                chunks = [c for chunk in chunks for c in step(chunk)]
            return chunks
        return seq
    if kind == "ByteLevel":
        if spec.get("add_prefix_space"):
            raise ValueError("ByteLevel add_prefix_space=true is not supported yet")
        if spec.get("use_regex", True):
            rx = re.compile(translate_regex(GPT2_PATTERN))
            return lambda s: rx.findall(s)
        return lambda s: [s] if s else []
    if kind == "Split":
        pat = spec["pattern"]
        rx = re.compile(translate_regex(pat["Regex"]) if "Regex" in pat else re.escape(pat["String"]))
        behavior = spec.get("behavior", "Isolated")
        if spec.get("invert") or behavior != "Isolated":
            raise ValueError(f"Split pre-tokenizer with behavior={behavior!r} invert={spec.get('invert')} not supported")

        def isolated(s):
            """Matches become chunks, and so does any text between them."""
            out, pos = [], 0
            for m in rx.finditer(s):
                if m.start() > pos:
                    out.append(s[pos:m.start()])
                if m.end() > m.start():
                    out.append(m.group())
                pos = m.end()
            if pos < len(s):
                out.append(s[pos:])
            return out
        return isolated
    if kind == "Digits":
        rx = re.compile(r"\d" if spec.get("individual_digits") else r"\d+")
        return lambda s: [p for p in re.split(f"({rx.pattern})", s) if p]
    raise ValueError(f"unsupported pre-tokenizer {kind!r}")


# The GPT-4 split pattern in its original \p{...} form. Our own pre-tokenizer
# (forge/tokenizer/pretokenize.py) is the same pattern already rewritten for `re`.
GPT4_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
                r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def tokenizer_to_hf_json(tok) -> tuple[dict, dict]:
    """Our Tokenizer -> (tokenizer.json, tokenizer_config.json) dicts."""
    enc = bytes_to_unicode()
    show = lambda b: "".join(enc[x] for x in b)  # noqa: E731
    vocab: dict[str, int] = {}
    for i, b in enumerate(tok.vocab):
        s = show(b)
        if s in vocab:
            raise ValueError(f"token {i} duplicates token {vocab[s]} ({b!r}); can't express in tokenizer.json")
        vocab[s] = i
    merges = [f"{show(tok.vocab[a])} {show(tok.vocab[b])}" for a, b in tok.merges]
    added = [{"id": i, "content": s, "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": False, "special": True} for s, i in tok.special_tokens.items()]
    data = {
        "version": "1.0",
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": GPT4_PATTERN}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
        ]},
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
        "model": {"type": "BPE", "vocab": vocab, "merges": merges, "ignore_merges": False,
                  "byte_fallback": False, "fuse_unk": False},
    }
    config = {"bos_token": "<|bos|>", "eos_token": "<|eos|>", "pad_token": "<|pad|>",
              "model_max_length": 1 << 30, "tokenizer_class": "PreTrainedTokenizerFast"}
    return data, config
