# Test suites

A suite is a directory of `<category>.jsonl` files. Each line is one test:

```json
{"id": "math-03", "prompt": "Q: What is 7 * 8?\nA:", "expected": " 56"}
```

`expected` is optional. The analyzer runs `prompt + expected` through the
model and records activations, broken down by category (the file name).
Later views also use `expected` to score accuracy and to measure how
ablations change it.

`basic/` matches the synthetic training corpus (stories, arithmetic, small
Python functions), plus a few out-of-distribution prompts in each category.
Add your own categories by dropping in more `.jsonl` files.

`stories/` targets TinyStories-style models:
- `names`: the answer is a name that appeared earlier (tests copying, i.e. induction)
- `objects`: remember which object or attribute was mentioned
- `dialogue`: quoted-speech conventions
- `narrative`: plain sentences without answers (for loss and activation statistics)

Some answers are one plausible continuation among several (e.g. "Yes, you" →
" can"), so exact-match scores are noisy. Compare *changes* (across checkpoints,
or with/without an ablation) rather than absolute values.

`knowledge/` asks short questions with one clear answer (220 of them: 70 facts, 50 each of
sports, math and python), for comparing models of different sizes trained on general data
(the `mix-*` configs):
- `facts`: capitals, planets, science basics
- `sports`: rules and numbers that don't change from year to year
- `math`: arithmetic and one-step word problems
- `python`: complete a line of Python, or say what it prints

A base model (no chat tuning) only *continues* text, so the questions use a
"Q: ... A:" format it will have seen in web text. Watch "answer loss" fall as
models grow: it moves long before exact matches start appearing.

The first 10 items of each file are the original 40-item suite (same IDs), so scores
on `facts-00`…`facts-09` etc. stay comparable with older runs; whole-suite numbers from
before 2026-10-25 aren't comparable with the 220-item suite. With 220 items one answer moves
exact match by 0.45 points, and `score` prints a standard error for each number.
`tests/test_suites.py` runs every Python "# prints" item to check its expected output.

`knowledge/_fewshot.txt` holds a few worked "Q: ... A: ..." examples (none of them
from the tests). `sleuth score --suite suites/knowledge --few-shot` puts them before
every test, the classic way to tell a base model "answer briefly, like this".
