"""Pre-tokenization: split text into chunks before BPE ever sees it.

BPE merges are only learned *inside* a chunk, never across chunks. Splitting
first stops the tokenizer learning junk tokens like "dog." or "the cat" and
keeps word / number / punctuation / whitespace pieces separate.

The pattern follows the GPT-4 ("cl100k") rules. Those are written with
Unicode classes (\\p{L} = any letter, \\p{N} = any number), which Python's
built-in `re` doesn't support; that needs the third-party `regex` package,
which we avoid. Equivalents that `re` does support:

    letter  \\p{L}  ->  [^\\W\\d_]   "word char, but not a digit or underscore"
    number  \\p{N}  ->  \\d          (Unicode decimal digits)

The alternatives, in order (the first one that matches wins):
    1. English contractions:              's 't 're 've 'm 'll 'd
    2. an optional non-letter + letters:  " hello", "Hello", "'quoted"
    3. 1-3 digits:                        "2024" -> "202", "4"  (helps arithmetic:
                                          numbers get consistent pieces)
    4. optional space + punctuation run:  " ...", "!!", " #"
    5. whitespace ending in newlines:     "\\n\\n", "  \\n"
    6. whitespace not followed by text:   trailing spaces
    7. any other whitespace
Rule 6 is the subtle one: in "a   b", the run of spaces gives up its last
space so that " b" stays one chunk (words carry their leading space).
"""

from __future__ import annotations

import re

PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|(?:[^\r\n\w]|_)?[^\W\d_]+"
    r"|\d{1,3}"
    r"| ?(?:[^\s\w]|_)+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

_compiled: dict[str, re.Pattern] = {}


def compile_pattern(pattern: str = PATTERN) -> re.Pattern:
    if pattern not in _compiled:
        _compiled[pattern] = re.compile(pattern)
    return _compiled[pattern]


def pretokenize(text: str, pattern: str = PATTERN) -> list[str]:
    """Split text into chunks. Joining the chunks gives back the text exactly."""
    return compile_pattern(pattern).findall(text)
