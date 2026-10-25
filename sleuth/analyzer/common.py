"""Shared helpers for analyzer views: encoding tests and scoring a model on them.

Every test case becomes   [<|bos|>] + prompt tokens + expected tokens
and we remember where the expected part starts. (Models without a BOS token,
like Qwen, get their end-of-text token there instead, which is what they saw
between documents in training: `tokenizer.start_id`.) Prompt and expected text are
encoded separately so we know exactly which tokens are "the answer" (encoding
them together could merge a token across the boundary).

Chat models: with a `prefix` (e.g. Gemma's "<bos><|turn>user\n...<turn|>\n<|turn>model\n"),
every test becomes  prefix + prompt tokens + expected tokens. The prefix is the
same for every test, so views skip it: losses count only predictions *of the
test's own tokens* (`scored`), and statistics only positions after the prefix
(`n_prefix`). Without a prefix, the prefix is just [<|bos|>] and nothing changes.

Scoring is *teacher-forced*: we feed the full sequence once and read off, at
every position, the probability the model gave to the true next token. This
is how loss is measured during training too, and it's cheap: one forward pass
per test, no sampling.

  token loss      mean cross-entropy (nats) over every next-token prediction
  expected loss   mean cross-entropy of the expected answer's *text* per answer token
                  (both spellings, see below)
  exact match     1 if the model's own greedy answer is the expected text and ends
                  there, else 0

With a tokenizer (`score`), both answer numbers are about the text, not one token
sequence:
  * Spelling. Our tokenizer splits " 7" into " " + "7", so in a few-shot prompt
    half the example answers start with a bare " ". A model that copies that and
    writes " " + "Paris" has written the right text, but scoring only " Paris"
    called it wrong (round 6: mix-20m's top guess was " " on 54 of 70 facts).
    The answer's probability is now P(canonical tokens) + P(" " + the rest), and
    `alt_share` says how much came through the second spelling.
  * Exact match generates the answer greedily (a few tokens) and compares the
    decoded text. The answer must also *end* there: " 0" followed by ".5" is
    "0.5", not "0" (round 5: "water freezes at 0.5 degrees Fahrenheit"). See
    `greedy_match`.
Without a tokenizer (ablation, which only compares losses), it's the plain
token-level version: canonical tokens only, exact = every answer token top-1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .suites import TestCase


@dataclass
class EncodedCase:
    case: TestCase
    ids: torch.Tensor   # [1, T] on CPU
    n_prompt: int       # tokens before the answer, including <|bos|> (and any prefix)
    n_prefix: int = 1   # tokens before the test's own text: just <|bos|>, or a chat prefix

    @property
    def scored(self) -> slice:
        """Positions whose next-token prediction is one of the test's own tokens."""
        return slice(self.n_prefix - 1, self.ids.shape[1] - 1)

    @property
    def category(self) -> str:
        return self.case.category

    @property
    def answer_positions(self) -> slice:
        """Positions whose *next-token prediction* is an answer token."""
        return slice(self.n_prompt - 1, self.ids.shape[1] - 1)


def encode_cases(tokenizer, cases: list[TestCase], max_len: int, prefix: list[int] | None = None) -> list[EncodedCase]:
    """prefix: token IDs to put before every test (default: [start token])."""
    prefix = list(prefix) if prefix else [tokenizer.start_id]
    if len(prefix) >= max_len - 1:
        raise ValueError(f"the prefix ({len(prefix)} tokens) leaves no room in max_seq_len {max_len}")
    out = []
    for c in cases:
        prompt = prefix + tokenizer.encode(c.prompt)
        answer = tokenizer.encode(c.expected) if c.expected else []
        ids = (prompt + answer)[:max_len]
        out.append(EncodedCase(c, torch.tensor([ids]), min(len(prompt), len(ids)), len(prefix)))
    return out


DEFAULT_CHAT_INSTRUCTION = "Continue this text."


def analysis_prefix(tokenizer, instruction: str = DEFAULT_CHAT_INSTRUCTION) -> list[int]:
    """The chat prefix for analyzing a chat model: <bos> + a user turn holding
    `instruction` + an opened model turn. Each test's text then reads as the
    model's own reply, which is where an instruction-tuned model is at home."""
    from ..formats.chat import encode_chat

    return encode_chat(tokenizer, [{"role": "user", "content": instruction}])


def categories(enc: list[EncodedCase]) -> list[str]:
    return sorted({e.category for e in enc})


def layer_of(tap_name: str) -> int:
    """'blocks.3.mlp.act' -> 3"""
    return int(tap_name.split(".")[1])


def _ends_word(text: str) -> bool | None:
    """Does a continuation end the word/number before it? True if its first
    character that isn't punctuation is whitespace ("\n", " A:", ".\n"),
    False if it's a letter or digit ("ian" after "Paris", ".5" after "0"),
    None if it's all punctuation so far (look at one more token)."""
    for ch in text:
        if ch.isspace():
            return True
        if ch.isalnum():
            return False
    return None


def _space_spelling(tokenizer, expected: str, canonical: list[int]) -> list[int] | None:
    """The other common way to write `expected` in tokens: a bare " " and then
    the rest (" Paris" -> " " + "Paris"). None if there's no such second
    spelling (no leading space, the tokenizer has no lone-space token, or it's
    the canonical one already, as for numbers: " 7" is always " " + "7")."""
    if len(expected) < 2 or expected[0] != " " or expected[1].isspace():
        return None
    space = tokenizer.encode(" ")
    if len(space) != 1:
        return None
    alt = space + tokenizer.encode(expected[1:])
    return alt if alt != canonical else None


def _seq_logprob(model, prompt: torch.Tensor, answer: list[int], vocab: int) -> float:
    """log P(answer tokens | prompt), teacher-forced."""
    ids = torch.cat([prompt, torch.tensor([answer], device=prompt.device)], dim=1)
    logits = model(ids).logits[0, prompt.shape[1] - 1:-1, :vocab].float()
    return float(logits.log_softmax(-1).gather(1, ids[0, prompt.shape[1]:, None]).sum())


@torch.no_grad()
def greedy_match(model, prompt: torch.Tensor, expected: str, tokenizer, vocab: int,
                 max_new: int) -> tuple[bool, list[int]]:
    """Let the model answer greedily and compare the *text* with `expected`.

    It matches if the decoded answer starts with the expected text and the
    model ends the word or number there (`_ends_word`): whitespace, an end-of-
    text/turn token, or punctuation followed by whitespace. Generation stops as
    soon as the outcome is known, after at most `max_new` tokens. Uses a KV
    cache when the model has one (ours and external models alike). Returns
    (match, the generated token IDs)."""
    special = set(getattr(tokenizer, "special_tokens", {}).values())
    cfg = getattr(model, "cfg", None)
    out: list[int] = []
    room = cfg.max_seq_len - prompt.shape[1] if cfg is not None else max_new
    max_new = max(0, min(max_new, room))
    if cfg is not None:
        from ..forge.infer.kv_cache import KVCache

        cache = KVCache.for_model(model, 1, prompt.shape[1] + max_new + 1, dtype=_cache_dtype(model))
        logits = model(prompt, cache=cache).logits[0, -1, :vocab]
    else:  # a test stub: full forward passes
        cache = None
        logits = model(prompt).logits[0, -1, :vocab]
    while len(out) < max_new:
        t = int(logits.argmax())
        if t in special:
            break
        out.append(t)
        text = tokenizer.decode(out).rstrip("\ufffd")  # a character still cut in half
        n = min(len(text), len(expected))
        if text[:n] != expected[:n]:
            return False, out
        if len(text) > len(expected):
            verdict = _ends_word(text[len(expected):])
            if verdict is not None:
                return verdict, out
        if cache is not None:
            logits = model(torch.tensor([[t]], device=prompt.device), cache=cache).logits[0, -1, :vocab]
        else:
            ids = torch.cat([prompt, torch.tensor([out], device=prompt.device)], dim=1)
            logits = model(ids).logits[0, -1, :vocab]
    # Stopped (end token, or out of tokens/room): right if the whole answer is there.
    text = tokenizer.decode(out)
    return text.startswith(expected), out


def _cache_dtype(model):
    """The KV cache's dtype: the model's compute dtype (bf16 for big external models)."""
    for p in model.parameters():
        if p.is_floating_point():
            return p.dtype
    return torch.float32


@torch.no_grad()
def score_cases(model, enc: list[EncodedCase], vocab: int, device, per_case: list | None = None,
                tokenizer=None) -> dict:
    """Loss/accuracy per category (see module docstring).

    per_case: if a list is given, one dict per test is appended to it (id,
    category, answer loss, exact, the model's top-1 token IDs over the answer
    span, and with a tokenizer its greedy answer and `alt_share`), so callers
    can show what the model got wrong and compute error bars.

    tokenizer: if given, answer loss counts both spellings of the answer and
    exact match compares the model's greedy answer as text. Without one, the
    token-level version."""
    acc: dict[str, dict[str, float]] = {}
    for e in enc:
        ids = e.ids.to(device)
        logits = model(ids).logits[0, :-1, :vocab].float()
        targets = ids[0, 1:]
        nll = F.cross_entropy(logits, targets, reduction="none")
        a = acc.setdefault(e.category, {"loss_sum": 0.0, "n_tok": 0, "exp_loss_sum": 0.0, "n_exp_tok": 0,
                                        "exact": 0, "n_exp_cases": 0})
        a["loss_sum"] += float(nll[e.scored].sum())
        a["n_tok"] += int(nll[e.scored].numel())
        span = e.answer_positions
        row = {"id": e.case.id, "category": e.category, "answer_loss": None, "exact": None, "top1_ids": []}
        if span.stop > span.start:
            top1 = logits[span].argmax(-1)
            answer = targets[span].tolist()
            n = len(answer)
            ans_nll = float(nll[span].sum())
            exact = bool((top1 == targets[span]).all())
            # (Skipped if max_seq_len cut the answer short.)
            if tokenizer is not None and e.case.expected and n == len(tokenizer.encode(e.case.expected)):
                prompt = ids[:, :e.n_prompt]
                alt = _space_spelling(tokenizer, e.case.expected, answer)
                if alt is not None and prompt.shape[1] + len(alt) <= getattr(getattr(model, "cfg", None),
                                                                           "max_seq_len", 1 << 30):
                    lp_alt = _seq_logprob(model, prompt, alt, vocab)
                    total = torch.logaddexp(torch.tensor(-ans_nll), torch.tensor(lp_alt)).item()
                    row["alt_share"] = math.exp(lp_alt - total)
                    ans_nll = -total
                else:
                    row["alt_share"] = 0.0
                # Compare with the answer as this tokenizer writes it back.
                exact, greedy = greedy_match(model, prompt, tokenizer.decode(answer), tokenizer, vocab, max_new=n + 4)
                row["greedy_ids"] = greedy
            a["exp_loss_sum"] += ans_nll
            a["n_exp_tok"] += n
            a["exact"] += int(exact)
            a["n_exp_cases"] += 1
            row.update(answer_loss=ans_nll / n, exact=exact, top1_ids=top1.tolist(), answer_ids=answer)
        if per_case is not None:
            per_case.append(row)
    return {c: _finish(a) for c, a in acc.items()} | {"overall": _finish(_merge(acc.values()))}


def standard_errors(per_case: list[dict]) -> dict[str, dict[str, float]]:
    """{category (and "overall"): {"answer_loss": SE, "exact_match": SE}}.

    How far a score could move by chance, given how many tests there are.
    With n tests and an exact-match rate p, the standard error is
    sqrt(p(1-p)/n): on 40 tests at 7.5% that's ±4.2 points, so two models
    within ~8 points of each other are indistinguishable. For answer loss we
    use the spread of the per-test losses (an approximation: the reported loss
    is averaged over tokens, the SE over tests)."""
    groups: dict[str, list[dict]] = {}
    for r in per_case:
        if r["answer_loss"] is not None:
            groups.setdefault(r["category"], []).append(r)
            groups.setdefault("overall", []).append(r)
    out = {}
    for c, rows in groups.items():
        n = len(rows)
        losses = [r["answer_loss"] for r in rows]
        mean = sum(losses) / n
        var = sum((x - mean) ** 2 for x in losses) / max(1, n - 1)
        p = sum(r["exact"] for r in rows) / n
        out[c] = {"answer_loss": math.sqrt(var / n), "exact_match": math.sqrt(p * (1 - p) / n)}
    return out


def _merge(items) -> dict:
    out: dict[str, float] = {}
    for a in items:
        for k, v in a.items():
            out[k] = out.get(k, 0) + v
    return out


def _finish(a: dict) -> dict:
    return {
        "loss": a["loss_sum"] / max(1, a["n_tok"]),
        "expected_loss": a["exp_loss_sum"] / a["n_exp_tok"] if a["n_exp_tok"] else None,
        "exact_match": a["exact"] / a["n_exp_cases"] if a["n_exp_cases"] else None,
        "n_tokens": int(a["n_tok"]),
        "n_answers": int(a["n_exp_cases"]),
    }


def series(label: str, help: str, rows: list[str], values: list[float], fmt: str = "num") -> dict:
    """One timeline series for one checkpoint: a value per row (layer, head, ...).
    fmt: "num" or "pct" (0..1 shown as a percentage)."""
    return {"label": label, "help": help, "rows": rows, "values": [float(v) for v in values], "fmt": fmt}


def head_labels(n_layers: int, n_heads: int) -> list[str]:
    return [f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)]
