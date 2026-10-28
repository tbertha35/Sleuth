"""Standard benchmarks, measured the same way for every model (`sleuth bench`).

To see where our small models start beating published ones, every model, ours or
downloaded (GPT-2, Pythia, SmolLM2, Gemma 3, Qwen3), goes through exactly the
same code and the same text:

  bpb        bits per byte on documents from our own validation split (mix-3b by
             default), decoded back to text and re-encoded with each model's own
             tokenizer. Per *byte*, so tokenizers with bigger or smaller tokens
             compare fairly. Each token is predicted once, with up to seq_len - 1
             tokens of context from its document (non-overlapping windows).
  hellaswag  pick the most likely ending of an everyday scene (10,042 validation
             items, 4 endings each; chance 25%). Text cleanup as in EleutherAI's
             lm-evaluation-harness.
  arc_easy   grade-school science questions (2,376 test items, 3-5 choices),
             "Question: ...\\nAnswer: <choice>".
  knowledge  our own few-shot knowledge suite (suites/knowledge), as in experiments.

Multiple choice is scored like lm-evaluation-harness: the log-probability of each
choice's text after the context. `acc` takes the highest sum; `acc_norm` divides
by the choice's length in characters first (long endings aren't punished for
having more tokens), the number usually quoted for HellaSwag. The context starts
with the model's document-start token (BOS, or <|endoftext|>).

Published numbers use slightly different prompts and settings, so compare models
within this table, not with numbers from papers.
"""

from __future__ import annotations

import ast
import json
import math
import re
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

TASKS = ("bpb", "hellaswag", "arc_easy", "knowledge")
BENCH_DIR = Path("data/bench")


# ---------------------------------------------------------------- data

def _read_parquet(path: Path) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # the optional extra approved for reading Parquet datasets
        raise ImportError("benchmarks are stored as Parquet: pip install -e .[parquet] (pyarrow)") from e
    return pq.read_table(path).to_pylist()


def _hs_clean(text: str) -> str:
    """lm-evaluation-harness' HellaSwag cleanup (WikiHow artifacts)."""
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def _listish(v):
    """Lists come back from Parquet as lists, or (older exports) as their repr."""
    return ast.literal_eval(v) if isinstance(v, str) else v


def hellaswag_items(path: Path | None = None, limit: int = 0) -> list[dict]:
    rows = _read_parquet(path or BENCH_DIR / "hellaswag" / "validation.parquet")
    out = []
    for r in rows[:limit or None]:
        ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
        out.append({"context": _hs_clean(r["activity_label"] + ": " + ctx),
                    "choices": [" " + _hs_clean(e) for e in _listish(r["endings"])], "label": int(r["label"])})
    return out


def arc_items(path: Path | None = None, limit: int = 0) -> list[dict]:
    rows = _read_parquet(path or BENCH_DIR / "arc-easy" / "test.parquet")
    out = []
    for r in rows[:limit or None]:
        ch = _listish(r["choices"])
        labels = list(ch["label"])
        out.append({"context": f"Question: {r['question']}\nAnswer:", "choices": [" " + t for t in ch["text"]],
                    "label": labels.index(r["answerKey"])})
    return out


def val_texts(shards: Path, max_bytes: int = 2_000_000) -> list[str]:
    """Whole documents from the start of a shard set's validation split, as text,
    until max_bytes of UTF-8. The same documents for every model."""
    import numpy as np

    from ..forge.data.loader import ShardDataset
    from ..forge.data.shard import doc_start_id
    from ..forge.tokenizer.bpe import Tokenizer

    ds = ShardDataset(Path(shards), "val")
    tok = Tokenizer.load(Path(shards) / "tokenizer.json")
    start = doc_start_id(tok)
    eos = tok.eos_id
    texts, total = [], 0
    for arr in ds.arrays:
        a = np.asarray(arr)
        cuts = np.flatnonzero(a == start).tolist() + [len(a)]
        for s, e in zip(cuts[:-1], cuts[1:]):
            ids = [int(i) for i in a[s + 1:e] if int(i) not in (start, eos)]
            if not ids:
                continue
            t = tok.decode(ids)
            texts.append(t)
            total += len(t.encode("utf-8"))
            if total >= max_bytes:
                return texts
    return texts


# ---------------------------------------------------------------- scoring

def _autocast(device):
    return (torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda"
            else torch.autocast("cpu", enabled=False))


def _vocab(model, tok) -> int:
    return min(tok.vocab_size, model.cfg.padded_vocab) if hasattr(model, "cfg") else tok.vocab_size


@torch.no_grad()
def _score_batch(model, seqs: list[list[int]], n_cont: list[int], vocab: int, device) -> list[float]:
    """Sum of log-probs of the last n_cont[i] tokens of each sequence (right-padded batch)."""
    L = max(len(s) for s in seqs)
    x = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    x = x.to(device)
    with _autocast(device):
        logits = model(x).logits[:, :, :vocab]
    out = []
    for i, (s, n) in enumerate(zip(seqs, n_cont)):
        # Only the scored positions go to fp32: a whole [B, T, 262k] batch (Gemma's
        # vocabulary) in fp32 would be gigabytes.
        lp = F.log_softmax(logits[i, len(s) - n - 1:len(s) - 1].float(), -1)
        out.append(float(lp.gather(-1, x[i, len(s) - n:len(s), None]).sum()))
    return out


def _token_budget(vocab: int) -> int:
    """Tokens per batch: ~1.5G logits (3 GB in bf16), at most 16k tokens."""
    return max(1024, min(16384, int(1.5e9 // vocab)))


def multiple_choice(model, tok, items: list[dict], device, log: Callable[[str], None] = print) -> dict:
    """acc / acc_norm over items {context, choices, label} (see the module docstring)."""
    max_len = model.cfg.max_seq_len
    batch_tokens = _token_budget(_vocab(model, tok))
    start = [tok.start_id] if tok.start_id is not None else []
    reqs = []   # (item, choice, ids, n_cont)
    for qi, it in enumerate(items):
        ctx = start + tok.encode(it["context"])
        for ci, ch in enumerate(it["choices"]):
            whole = start + tok.encode(it["context"] + ch)
            # The choice's tokens, as the tokenizer splits context + choice together
            # (lm-evaluation-harness); if that doesn't extend the context, separately.
            cont = whole[len(ctx):] if whole[:len(ctx)] == ctx and len(whole) > len(ctx) else tok.encode(ch)
            ids = (ctx + cont)[-max_len:]
            reqs.append((qi, ci, ids, min(len(cont), len(ids) - 1)))
    reqs.sort(key=lambda r: -len(r[2]))
    scores: dict[tuple[int, int], float] = {}
    i, t0 = 0, time.time()
    while i < len(reqs):
        n = max(1, batch_tokens // len(reqs[i][2]))
        chunk = reqs[i:i + n]
        for r, s in zip(chunk, _score_batch(model, [r[2] for r in chunk], [r[3] for r in chunk],
                                            _vocab(model, tok), device)):
            scores[(r[0], r[1])] = s
        i += n
    right = right_norm = 0
    for qi, it in enumerate(items):
        lps = [scores[(qi, ci)] for ci in range(len(it["choices"]))]
        norm = [lp / max(1, len(ch)) for lp, ch in zip(lps, it["choices"])]
        right += int(max(range(len(lps)), key=lps.__getitem__) == it["label"])
        right_norm += int(max(range(len(norm)), key=norm.__getitem__) == it["label"])
    n = len(items)
    se = lambda p: math.sqrt(p * (1 - p) / max(1, n))  # noqa: E731
    acc, acc_norm = right / n, right_norm / n
    log(f"  {n} items in {time.time() - t0:.0f}s: acc {acc:.1%}, acc_norm {acc_norm:.1%} (±{se(acc_norm):.1%})")
    return {"n": n, "acc": acc, "acc_norm": acc_norm, "se": se(acc_norm)}


@torch.no_grad()
def bits_per_byte(model, tok, texts: list[str], device, seq_len: int = 1024,
                  log: Callable[[str], None] = print) -> dict:
    seq_len = min(seq_len, model.cfg.max_seq_len)
    start = [tok.start_id] if tok.start_id is not None else []
    vocab = _vocab(model, tok)
    batch = max(1, _token_budget(vocab) // seq_len)
    windows, n_bytes = [], 0
    for t in texts:
        ids = start + tok.encode(t)
        n_bytes += len(t.encode("utf-8"))
        for s in range(0, len(ids) - 1, seq_len - 1):   # window k predicts ids[s+1 .. s+seq_len-1]
            windows.append(ids[s:s + seq_len])
    windows.sort(key=len, reverse=True)
    nll, n_tok, t0 = 0.0, 0, time.time()
    for i in range(0, len(windows), batch):
        chunk = windows[i:i + batch]
        lps = _score_batch(model, chunk, [len(w) - 1 for w in chunk], vocab, device)
        nll -= sum(lps)
        n_tok += sum(len(w) - 1 for w in chunk)
    bpb = nll / math.log(2) / n_bytes
    log(f"  {len(texts)} documents, {n_bytes:,} bytes, {n_tok:,} tokens in {time.time() - t0:.0f}s: bpb {bpb:.4f}")
    return {"bpb": bpb, "bytes": n_bytes, "tokens": n_tok, "bytes_per_token": n_bytes / max(1, n_tok)}


def knowledge(model, tok, device, suite: str = "suites/knowledge") -> dict:
    from .common import encode_cases, score_cases
    from .suites import load_suite

    f = Path(suite) / "_fewshot.txt"
    prefix = [tok.start_id] + tok.encode(f.read_text(encoding="utf-8")) if f.exists() else None
    enc = encode_cases(tok, load_suite(suite), model.cfg.max_seq_len, prefix)
    with _autocast(device):
        res = score_cases(model, enc, _vocab(model, tok), device)
    o = res["overall"]
    return {"answer_loss": o["expected_loss"], "exact": o["exact_match"]}


# ---------------------------------------------------------------- driver

def model_name(path: str | Path) -> str:
    p = Path(path)
    return p.parent.name if p.name == "hf" else p.stem if p.is_file() else p.name


def run_bench(path: str | Path, tasks=TASKS, limit: int = 0, shards: str = "data/shards/mix-3b",
              out_dir: str | Path = "runs/bench", device: str = "auto", log: Callable[[str], None] = print,
              name: str | None = None) -> dict:
    """Run `tasks` on one model; results go to <out_dir>/<name>.json (merged with
    earlier results for the same name, so tasks can be added later)."""
    from ..models import load_any

    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                       ("cpu" if device == "auto" else device))
    name = name or model_name(path)
    out = Path(out_dir) / f"{name}.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    model, tok, info = load_any(path, device=dev, dtype=torch.float32, max_seq_len=1024)
    model.eval()
    res.update(name=name, path=str(path), params=model.num_params(),
               non_embedding=model.num_params(exclude_embedding=True), format=info.get("format"))
    log(f"{name}: {res['params'] / 1e6:.0f}M parameters ({res['non_embedding'] / 1e6:.0f}M non-embedding)")
    for task in tasks:
        log(f" {task}")
        if task == "bpb":
            res["bpb"] = bits_per_byte(model, tok, val_texts(Path(shards), 200_000 if limit else 2_000_000), dev,
                                       log=log)
        elif task == "hellaswag":
            res["hellaswag"] = multiple_choice(model, tok, hellaswag_items(limit=limit), dev, log=log)
        elif task == "arc_easy":
            res["arc_easy"] = multiple_choice(model, tok, arc_items(limit=limit), dev, log=log)
        elif task == "knowledge":
            res["knowledge"] = knowledge(model, tok, dev)
            log(f"  answer loss {res['knowledge']['answer_loss']:.3f}, exact {res['knowledge']['exact']:.1%}")
        else:
            raise ValueError(f"unknown task {task!r} (tasks: {', '.join(TASKS)})")
        res.setdefault("limited", {})[task] = limit
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1))
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return res


def table(out_dir: str | Path = "runs/bench") -> str:
    """All results in out_dir as one markdown table, smallest model first."""
    rows = [json.loads(f.read_text()) for f in sorted(Path(out_dir).glob("*.json"))]
    rows.sort(key=lambda r: r.get("non_embedding", 0))
    f = lambda v, s: s.format(v) if v is not None else "-"  # noqa: E731
    lines = ["| model | params | non-emb. | bpb | HellaSwag acc_norm | ARC-Easy acc | knowledge loss | exact |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        g = lambda t, k: (r.get(t) or {}).get(k)  # noqa: E731
        part = " (partial)" if any((r.get("limited") or {}).values()) else ""
        lines.append(f"| {r['name']}{part} | {r['params'] / 1e6:.0f}M | {r['non_embedding'] / 1e6:.0f}M | "
                     f"{f(g('bpb', 'bpb'), '{:.4f}')} | {f(g('hellaswag', 'acc_norm'), '{:.1%}')} | "
                     f"{f(g('arc_easy', 'acc'), '{:.1%}')} | {f(g('knowledge', 'answer_loss'), '{:.3f}')} | "
                     f"{f(g('knowledge', 'exact'), '{:.1%}')} |")
    return "\n".join(lines)
