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

`knowledge/` asks short questions with one clear answer, for comparing models of
different sizes trained on general data (the `mix-*` configs):
- `facts`: capitals, planets, science basics
- `sports`: rules and numbers that don't change from year to year
- `math`: arithmetic and one-step word problems
- `python`: complete a line of Python, or say what it prints

A base model (no chat tuning) only *continues* text, so the questions use a
"Q: ... A:" format it will have seen in web text. Watch "expected loss" fall as
models grow: it moves long before exact matches start appearing.

`knowledge/_fewshot.txt` holds a few worked "Q: ... A: ..." examples (none of them
from the tests). `sleuth score --suite suites/knowledge --few-shot` puts them before
every test, the classic way to tell a base model "answer briefly, like this".
