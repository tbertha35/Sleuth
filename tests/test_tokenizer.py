import random

import pytest

from sleuth.forge.data.synthetic import generate_corpus
from sleuth.forge.tokenizer.bpe import SPECIAL_TOKENS, Tokenizer
from sleuth.forge.tokenizer.pretokenize import pretokenize
from sleuth.forge.tokenizer.train_bpe import learn_merges, train_tokenizer

TRICKY = [
    "",
    "Hello, world!",
    "  leading and trailing spaces  ",
    "tabs\tand\r\nwindows newlines\n\n\n",
    "numbers 1234567 and 3.14159 and -42",
    "unicode: café naïve 日本語 emoji 🧠🔍 ²³ Ⅻ",
    "code:\n    def f(x_1):\n        return x_1 ** 2  # comment",
    "contractions: I'm, you're, it's, WE'LL, they'd",
    "special-looking text <|eos|> <|bos|> should stay text",
    "a" * 1000,
]


def random_text(rng: random.Random, n: int) -> str:
    alphabet = "abc xyz ABC 0123 .,!?'\"\n\t_-é日🧠"
    return "".join(rng.choice(alphabet) for _ in range(n))


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    d = tmp_path_factory.mktemp("tok")
    generate_corpus(d / "corpus", megabytes=0.5, seed=1)
    return train_tokenizer([d / "corpus"], vocab_size=600, out_path=d / "tok.json", log=lambda s: None)


def test_pretokenize_is_lossless():
    rng = random.Random(0)
    for s in TRICKY + [random_text(rng, 300) for _ in range(50)]:
        assert "".join(pretokenize(s)) == s


def test_pretokenize_examples():
    assert pretokenize("Hello world") == ["Hello", " world"]
    assert pretokenize("12345") == ["123", "45"]
    assert pretokenize("don't") == ["don", "'t"]


def test_roundtrip(tok):
    rng = random.Random(0)
    for s in TRICKY + [random_text(rng, 500) for _ in range(50)]:
        assert tok.decode(tok.encode(s)) == s


def test_merges_compress(tok):
    s = "Once upon a time, there was a little girl named Lily."
    assert len(tok.encode(s)) < len(s.encode()) / 2


def test_special_tokens_not_injected_by_default(tok):
    ids = tok.encode("<|eos|>")
    assert tok.eos_id not in ids
    ids = tok.encode("<|eos|>", allowed_special="all")
    assert ids == [tok.eos_id]
    ids = tok.encode("hi<|user|>there", allowed_special={"<|user|>"})
    assert tok.special("<|user|>") in ids
    assert tok.decode(ids) == "hi<|user|>there"


def test_vocab_layout(tok):
    assert tok.vocab_size == 256 + len(tok.merges) + len(SPECIAL_TOKENS)
    assert tok.bos_id == 256 + len(tok.merges)
    assert len(tok.token_byte_lengths()) == tok.vocab_size


def test_save_load_identical(tok, tmp_path):
    tok.save(tmp_path / "t.json")
    t2 = Tokenizer.load(tmp_path / "t.json")
    assert t2.fingerprint() == tok.fingerprint()
    s = "There was once a dog named Spot. 123 + 456"
    assert t2.encode(s) == tok.encode(s)


def test_training_is_deterministic():
    counts = {"hello": 5, "help": 3, "hell": 2, "yellow": 4}
    assert learn_merges(counts, 10, log=lambda s: None) == learn_merges(dict(reversed(counts.items())), 10, log=lambda s: None)


def test_learn_merges_matches_naive():
    """The fast incremental trainer must produce the same merges as the
    obvious slow algorithm (recount everything after each merge)."""
    rng = random.Random(3)
    words = ["".join(rng.choice("abcab ") for _ in range(rng.randint(2, 8))) for _ in range(200)]
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + rng.randint(1, 5)

    def naive(counts, n):
        seqs = {w: list(w.encode()) for w in counts}
        merges = []
        for m in range(n):
            pc: dict = {}
            for w, s in seqs.items():
                for p in zip(s, s[1:]):
                    pc[p] = pc.get(p, 0) + counts[w]
            if not pc:
                break
            best = min(pc, key=lambda p: (-pc[p], p))
            merges.append(best)
            for w, s in seqs.items():
                out, i = [], 0
                while i < len(s):
                    if i < len(s) - 1 and (s[i], s[i + 1]) == best:
                        out.append(256 + m)
                        i += 2
                    else:
                        out.append(s[i])
                        i += 1
                seqs[w] = out
        return merges

    assert learn_merges(counts, 40, log=lambda s: None) == naive(counts, 40)
