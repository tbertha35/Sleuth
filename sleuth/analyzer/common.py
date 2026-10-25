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
  expected loss   mean cross-entropy over just the expected answer's tokens
  exact match     1 if every answer token was the model's top choice *and* the
                  model would end the answer there, else 0

"End the answer there": with a tokenizer, an answer only counts when the model's
own next token after it doesn't carry on the same word or number. Matching the
answer's tokens alone counted " 0" as right when mix-100m was about to say
"0.5 degrees Fahrenheit" (round 5). See `answer_ends`.
"""

from __future__ import annotations

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


@torch.no_grad()
def answer_ends(model, ids: torch.Tensor, next_id: int, tokenizer, vocab: int,
                max_extra: int = 3) -> tuple[bool, list[int]]:
    """Would the model stop the answer at the end of `ids`?

    next_id is the model's top choice right after the answer (free: it comes
    out of the scoring pass). A special token (end of text, end of turn) or
    whitespace ends the answer; a letter or digit carries it on. After
    punctuation we can't tell yet ("0." could become "0.5" or end a sentence),
    so we let the model write up to `max_extra` more tokens greedily. Returns
    (ends, the continuation's token IDs)."""
    special = set(getattr(tokenizer, "special_tokens", {}).values())
    room = getattr(getattr(model, "cfg", None), "max_seq_len", None)
    cont = [int(next_id)]
    while True:
        if cont[-1] in special:
            return True, cont
        verdict = _ends_word(tokenizer.decode(cont))
        if verdict is not None:
            return verdict, cont
        if len(cont) > max_extra or (room is not None and ids.shape[1] + len(cont) > room):
            return True, cont  # nothing but punctuation ("!!!!"), or no room to look further
        seq = torch.cat([ids, torch.tensor([cont], device=ids.device)], dim=1)
        cont.append(int(model(seq).logits[0, -1, :vocab].argmax()))


@torch.no_grad()
def score_cases(model, enc: list[EncodedCase], vocab: int, device, per_case: list | None = None,
                tokenizer=None) -> dict:
    """Teacher-forced loss/accuracy per category (see module docstring).

    per_case: if a list is given, one dict per test is appended to it (id,
    category, answer loss, exact, the model's top-1 token IDs over the
    answer span and, with a tokenizer, what it would write next), so callers
    can show what the model got wrong and compute error bars.

    tokenizer: if given, an exact match also needs the model to end the
    answer there (`answer_ends`). Without one (ablation, which only compares
    losses), matching the answer's tokens is enough."""
    acc: dict[str, dict[str, float]] = {}
    for e in enc:
        ids = e.ids.to(device)
        full = model(ids).logits[0, :, :vocab].float()
        logits = full[:-1]
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
            exact = bool((top1 == targets[span]).all())
            if exact and tokenizer is not None:
                exact, after = answer_ends(model, ids, int(full[-1].argmax()), tokenizer, vocab)
                row["after_ids"] = after
            a["exp_loss_sum"] += float(nll[span].sum())
            a["n_exp_tok"] += span.stop - span.start
            a["exact"] += int(exact)
            a["n_exp_cases"] += 1
            row.update(answer_loss=float(nll[span].mean()), exact=exact, top1_ids=top1.tolist(),
                       answer_ids=targets[span].tolist())
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
    import math

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
