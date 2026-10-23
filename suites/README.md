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
