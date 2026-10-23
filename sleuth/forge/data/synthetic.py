"""Synthetic training text that we generate ourselves.

Why this exists:
  * It needs no download and has no licence or terms-of-service questions:
    every sentence comes from the templates in this file.
  * It's *structured on purpose*, so you can see specific abilities emerge
    in the analyzer:
      - stories repeat character names and keep pronouns consistent, so the
        model must learn to copy earlier tokens (this is what "induction
        heads" do, and you can watch them form across checkpoints);
      - math has exact answers, so you can measure accuracy, not just loss;
      - code has strict syntax plus real outputs (computed by running the
        reference implementation), so you get a third, very different
        "category" for routing / neuron-specialization views.
  * It's small-vocabulary text, a bit like TinyStories, so a ~10M model can
    actually get good at it.

Limits: it's far less varied than real text. A model trained only on this
will be fluent within its little world and useless outside it. That's fine
for a smoke test and for learning; use real data (docs/DATA.md) next.

Usage:
    python -m sleuth synth --out data/raw/synthetic --mb 50
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Stories
# --------------------------------------------------------------------------- #

GIRL_NAMES = ["Lily", "Mia", "Zoe", "Anna", "Ella", "Ruby", "Lucy", "Nora", "Ivy", "Maya", "Rosa", "Hana"]
BOY_NAMES = ["Max", "Ben", "Sam", "Tom", "Leo", "Finn", "Jack", "Omar", "Kai", "Theo", "Ravi", "Luis"]
ANIMALS = ["dog", "cat", "bunny", "bear", "fox", "bird", "frog", "duck", "mouse", "owl"]
ANIMAL_NAMES = ["Spot", "Bubbles", "Pip", "Coco", "Ziggy", "Mittens", "Biscuit", "Pebble", "Sunny", "Rocket"]

PLACES = [
    ("a small house near the forest", "the forest"),
    ("a little town by the sea", "the beach"),
    ("a big farm with red barns", "the farm"),
    ("a busy city with tall buildings", "the park"),
    ("a quiet village by the river", "the river"),
    ("a cozy cabin on a hill", "the hill"),
]
HOBBIES = [
    "paint pictures", "read books", "sing songs", "climb trees", "build towers with blocks",
    "play with a ball", "look at the stars", "bake cookies", "dance in the rain", "draw maps",
]
OBJECTS = [
    ("a shiny red ball", "the ball"), ("an old map", "the map"), ("a tiny gold key", "the key"),
    ("a blue kite", "the kite"), ("a big wooden box", "the box"), ("a soft green hat", "the hat"),
    ("a small silver bell", "the bell"), ("a funny yellow boot", "the boot"),
]
FEELINGS_GOOD = ["happy", "proud", "excited", "glad", "thankful"]
FEELINGS_BAD = ["sad", "worried", "scared", "upset", "lonely"]
WEATHER = ["sunny", "windy", "rainy", "cold", "warm"]
MORALS = [
    "Sharing makes everyone happy.",
    "It is good to help your friends.",
    "Being brave is not always easy, but it is worth it.",
    "When you work together, hard things become easy.",
    "Kind words can make a big difference.",
    "It is okay to ask for help.",
]


@dataclass
class Character:
    name: str
    kind: str  # "girl", "boy", or an animal
    he: str    # subject pronoun
    him: str   # object pronoun
    his: str   # possessive

    @property
    def He(self) -> str:
        return self.he.capitalize()

    @property
    def His(self) -> str:
        return self.his.capitalize()


def _character(rng: random.Random, avoid: set[str]) -> Character:
    roll = rng.random()
    if roll < 0.4:
        name, kind, pron = rng.choice([n for n in GIRL_NAMES if n not in avoid]), "girl", "she"
    elif roll < 0.8:
        name, kind, pron = rng.choice([n for n in BOY_NAMES if n not in avoid]), "boy", "he"
    else:
        name = rng.choice([n for n in ANIMAL_NAMES if n not in avoid])
        kind, pron = rng.choice(ANIMALS), rng.choice(["she", "he"])
    if pron == "she":
        return Character(name, kind, "she", "her", "her")
    return Character(name, kind, "he", "him", "his")


def _story_find(rng, a: Character, b: Character, place_far: str) -> list[str]:
    obj, the_obj = rng.choice(OBJECTS)
    found_where = rng.choice(["under a big rock", "behind a tree", "in the tall grass", "next to the old gate"])
    lines = [
        f"One day, {a.name} found {obj} {found_where}.",
        f'"What is this?" {a.he} asked.',
        f"{a.He} ran to show {the_obj} to {a.his} friend {b.name}.",
    ]
    if "map" in the_obj:
        lines += [
            f'{b.name} looked at it and said, "It is a map! Let\'s follow it!"',
            f"So {a.name} and {b.name} walked all the way to {place_far}.",
            f"At the end of the path, they found a box full of shiny stones.",
        ]
    else:
        lines += [
            f'{b.name} smiled and said, "Maybe someone lost it. Let\'s find who it belongs to."',
            f"They asked everyone near {place_far}. At last, a little {rng.choice(ANIMALS)} said, \"That is mine!\"",
            f"{a.name} gave {the_obj} back, and the little animal was very {rng.choice(FEELINGS_GOOD)}.",
        ]
    lines.append(f"{a.name} and {b.name} were very {rng.choice(FEELINGS_GOOD)}.")
    return lines


def _story_lost(rng, a: Character, b: Character, place_far: str) -> list[str]:
    obj, the_obj = rng.choice(OBJECTS)
    return [
        f"{a.name} had {obj}. {a.He} loved it very much.",
        f"One {rng.choice(WEATHER)} day, {a.name} took {the_obj} to {place_far}.",
        f"When it was time to go home, {the_obj} was gone!",
        f"{a.name} felt {rng.choice(FEELINGS_BAD)}. {a.He} looked everywhere, but {a.he} could not find it.",
        f'Then {b.name} came by. "Why are you {rng.choice(FEELINGS_BAD)}?" asked {b.name}.',
        f'"I lost {the_obj}," said {a.name}.',
        f'"I will help you look," said {b.name}.',
        f"They looked behind the trees and under the benches. Then {b.name} saw something.",
        f'"Here it is!" {b.he} shouted. {a.name} hugged {b.name} and said, "Thank you!"',
    ]


def _story_stuck(rng, a: Character, b: Character, place_far: str) -> list[str]:
    obj, the_obj = rng.choice([o for o in OBJECTS if "ball" in o[1] or "kite" in o[1] or "hat" in o[1]])
    return [
        f"One day, {a.name} was playing with {obj} at {place_far}.",
        f"The wind blew hard, and {the_obj} got stuck high up in a tree.",
        f"{a.name} tried to reach it, but {a.he} was too small.",
        f"{a.He} felt {rng.choice(FEELINGS_BAD)}.",
        f"Then {b.name} had an idea. \"I can stand on the bench, and you can climb on my back,\" said {b.name}.",
        f"They worked together, and {a.name} got {the_obj} down.",
        f'"We did it!" said {a.name}. {a.He} and {b.name} played together until the sun went down.',
    ]


def _story_learn(rng, a: Character, b: Character, place_far: str) -> list[str]:
    skill = rng.choice(["ride a bike", "swim", "whistle", "jump rope", "tie a knot", "fly a kite"])
    n = rng.randint(2, 5)
    return [
        f"{a.name} wanted to learn how to {skill}.",
        f"The first time {a.he} tried, {a.he} fell down. {a.He} felt {rng.choice(FEELINGS_BAD)}.",
        f'{b.name} said, "Do not give up. Try again!"',
        f"So {a.name} tried again, and again. {a.He} tried {n} more times.",
        f"At last, {a.name} could {skill}!",
        f"{a.He} was so {rng.choice(FEELINGS_GOOD)}. {a.He} thanked {b.name} for helping {a.him}.",
    ]


STORY_PLOTS = [_story_find, _story_lost, _story_stuck, _story_learn]


def make_story(rng: random.Random) -> str:
    a = _character(rng, set())
    b = _character(rng, {a.name})
    home, place_far = rng.choice(PLACES)
    article = "an" if a.kind[0] in "aeiou" else "a"
    opening = rng.choice([
        f"Once upon a time, there was {article} little {a.kind} named {a.name}.",
        f"There was once {article} {a.kind} named {a.name}.",
        f"{a.name} was {article} {rng.choice(['kind', 'curious', 'brave', 'shy', 'happy'])} little {a.kind}.",
    ])
    lines = [
        opening,
        f"{a.He} lived in {home}.",
        f"{a.name} loved to {rng.choice(HOBBIES)}.",
    ]
    lines += rng.choice(STORY_PLOTS)(rng, a, b, place_far)
    if rng.random() < 0.5:
        lines.append(rng.choice(MORALS))
    else:
        lines.append(f"From that day on, {a.name} and {b.name} were best friends.")
    return " ".join(lines)


# --------------------------------------------------------------------------- #
# Math
# --------------------------------------------------------------------------- #

ITEMS = ["apples", "books", "stickers", "marbles", "cookies", "pencils", "shells", "coins", "flowers", "cards"]


def make_math(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.4:
        # Bare arithmetic. Bigger numbers appear less often (log-uniform-ish),
        # so the model sees lots of easy cases and some hard ones.
        hi = rng.choice([10, 100, 1000])
        a, b = rng.randint(0, hi), rng.randint(0, hi)
        op = rng.choice(["+", "-", "*"])
        if op == "*":
            a, b = rng.randint(0, 20), rng.randint(0, 20)
        if op == "-" and b > a:
            a, b = b, a
        ans = {"+": a + b, "-": a - b, "*": a * b}[op]
        return f"Q: What is {a} {op} {b}?\nA: {ans}"
    if kind < 0.75:
        name = rng.choice(GIRL_NAMES + BOY_NAMES)
        he = "She" if name in GIRL_NAMES else "He"
        item = rng.choice(ITEMS)
        a, b = rng.randint(2, 30), rng.randint(2, 30)  # >= 2 so plurals stay grammatical
        form = rng.random()
        if form < 0.4:
            q = f"{name} has {a} {item}. {he} gets {b} more {item}. How many {item} does {name} have now?"
            ans = a + b
        elif form < 0.8:
            a, b = max(a, b), min(a, b)
            q = f"{name} has {a} {item}. {he} gives {b} {item} to a friend. How many {item} does {name} have left?"
            ans = a - b
        else:
            a, b = rng.randint(2, 9), rng.randint(2, 9)
            q = f"{name} has {a} bags. Each bag has {b} {item}. How many {item} does {name} have in total?"
            ans = a * b
        return f"Q: {q}\nA: {ans}"
    if kind < 0.9:
        start, n = rng.randint(0, 90), rng.randint(3, 8)
        step = rng.choice([1, 1, 2, 5, 10])
        seq = ", ".join(str(start + i * step) for i in range(n))
        return f"Count by {step} from {start}: {seq}"
    a, b = rng.sample(range(0, 1000), 2)
    word, ans = rng.choice([("bigger", max(a, b)), ("smaller", min(a, b))])
    return f"Q: Which is {word}, {a} or {b}?\nA: {ans}"


# --------------------------------------------------------------------------- #
# Code
# --------------------------------------------------------------------------- #

# Each template: (function-name choices, source with {f} {xs} {x} {acc} slots,
# a reference implementation used to compute the printed output, an input maker).
CODE_TEMPLATES = [
    (
        ["sum_list", "total", "add_all"],
        "def {f}({xs}):\n    {acc} = 0\n    for {x} in {xs}:\n        {acc} += {x}\n    return {acc}",
        lambda v: sum(v), "list",
    ),
    (
        ["find_max", "largest", "biggest"],
        "def {f}({xs}):\n    {acc} = {xs}[0]\n    for {x} in {xs}:\n        if {x} > {acc}:\n            {acc} = {x}\n    return {acc}",
        lambda v: max(v), "list",
    ),
    (
        ["count_even", "num_evens"],
        "def {f}({xs}):\n    {acc} = 0\n    for {x} in {xs}:\n        if {x} % 2 == 0:\n            {acc} += 1\n    return {acc}",
        lambda v: sum(1 for i in v if i % 2 == 0), "list",
    ),
    (
        ["squares", "square_all"],
        "def {f}({xs}):\n    return [{x} * {x} for {x} in {xs}]",
        lambda v: [i * i for i in v], "list",
    ),
    (
        ["reverse_text", "backwards"],
        "def {f}({s}):\n    return {s}[::-1]",
        lambda v: v[::-1], "word",
    ),
    (
        ["factorial", "fact"],
        "def {f}({n}):\n    {acc} = 1\n    for {x} in range(1, {n} + 1):\n        {acc} *= {x}\n    return {acc}",
        lambda v: __import__("math").factorial(v), "small_int",
    ),
    (
        ["is_even", "even"],
        "def {f}({n}):\n    return {n} % 2 == 0",
        lambda v: v % 2 == 0, "int",
    ),
    (
        ["count_vowels", "vowels"],
        "def {f}({s}):\n    {acc} = 0\n    for {x} in {s}:\n        if {x} in \"aeiou\":\n            {acc} += 1\n    return {acc}",
        lambda v: sum(1 for c in v if c in "aeiou"), "word",
    ),
]
WORDS = ["apple", "banana", "river", "planet", "sleuth", "garden", "rocket", "puzzle", "window", "orange"]


def make_code(rng: random.Random) -> str:
    names, src, ref, arg_kind = rng.choice(CODE_TEMPLATES)
    slots = {
        "f": rng.choice(names),
        "xs": rng.choice(["nums", "values", "items", "xs", "numbers"]),
        "x": rng.choice(["n", "v", "item", "x"]),
        "acc": rng.choice(["total", "result", "acc", "best", "count"]),
        "s": rng.choice(["s", "text", "word"]),
        "n": rng.choice(["n", "k", "num"]),
    }
    if slots["x"] == slots["n"]:  # avoid shadowing the argument in factorial
        slots["x"] = "i"
    body = src.format(**slots)
    calls = []
    for _ in range(rng.randint(1, 3)):
        if arg_kind == "list":
            arg = [rng.randint(0, 20) for _ in range(rng.randint(1, 6))]
        elif arg_kind == "word":
            arg = rng.choice(WORDS)
        elif arg_kind == "small_int":
            arg = rng.randint(0, 7)
        else:
            arg = rng.randint(0, 100)
        calls.append(f"print({slots['f']}({arg!r}))  # {ref(arg)!r}")
    return body + "\n\n" + "\n".join(calls)


# --------------------------------------------------------------------------- #
# Corpus writer
# --------------------------------------------------------------------------- #

GENERATORS = {"story": make_story, "math": make_math, "code": make_code}
DEFAULT_MIX = {"story": 0.7, "math": 0.15, "code": 0.15}


def generate_corpus(out_dir: Path, megabytes: float, seed: int = 0, mix: dict[str, float] | None = None) -> dict:
    """Write one jsonl file per category until ~`megabytes` of text exist.

    Each line is {"text": ..., "category": ...}. Returns counts per category.
    """
    mix = mix or DEFAULT_MIX
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    cats = list(mix)
    weights = [mix[c] for c in cats]
    files = {c: open(out_dir / f"{c}.jsonl", "w", encoding="utf-8") for c in cats}
    stats = {c: {"docs": 0, "bytes": 0} for c in cats}
    target = int(megabytes * 1_000_000)
    written = 0
    try:
        while written < target:
            cat = rng.choices(cats, weights)[0]
            text = GENERATORS[cat](rng)
            files[cat].write(json.dumps({"text": text, "category": cat}) + "\n")
            n = len(text.encode("utf-8"))
            stats[cat]["docs"] += 1
            stats[cat]["bytes"] += n
            written += n
    finally:
        for f in files.values():
            f.close()
    (out_dir / "README.txt").write_text(
        "Synthetic corpus generated by sleuth/forge/data/synthetic.py "
        f"(seed={seed}, ~{megabytes} MB). No external sources.\n"
    )
    return stats
