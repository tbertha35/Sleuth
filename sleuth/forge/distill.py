"""Learning from a bigger model ("distillation"). Two ways, both from a teacher
served by vLLM (or any OpenAI-compatible server) on this machine or the LAN.

Option A, teacher-written text (teach_text below).

A big model (the teacher, e.g. Qwen3.8 served by vLLM on the GX10s) rewrites
snippets of our own corpus as clear explanations and question/answer sets. A
small model trained on that text learns from the teacher's knowledge and style
with *our* tokenizer: no shared vocabulary needed (the Phi / Cosmopedia recipe).

    sleuth teach text --server http://<teacher>:8000 --model <served name> \\
        --from data/raw/wikipedia data/raw/sports --n 20000 \\
        --out data/raw/teacher-qwen/part-000.jsonl
    sleuth shard data/raw/teacher-qwen ...      # as any other source

The server is any OpenAI-compatible endpoint (vLLM, llama.cpp's llama-server),
on this machine or another one on the local network: point --server at its LAN
address. Requests go through sleuth/net.py, which allows the local network and
blocks the internet.

Output lines: {"text", "source": "teacher:<model>", "template", "id"}. The file is
appended to and re-running skips ids already done, so an interrupted job resumes.

Option B, the teacher's probabilities (score_logits, TeacherSet below). Instead of
one right answer per position, the student learns the teacher's whole top-k
guess list ("Paris 91%, Lyon 3%, ..."): far more signal per token. That needs the
student to share the teacher's vocabulary, so its shards are made with the
teacher's tokenizer (a Hugging Face tokenizer.json, which Tokenizer.load reads):

    sleuth shard data/raw/... --tokenizer <teacher>/tokenizer.json --out data/shards/qwen-mix
    sleuth teach logits --server http://<teacher>:8000 --model <name> \\
        --shards data/shards/qwen-mix --windows 50000 --out data/teacher/qwen-mix
    sleuth train ... --set train.teacher=data/teacher/qwen-mix

vLLM returns, for every position of a prompt given as token ids, the top-k
log-probabilities of the next token (`prompt_logprobs`, at most the server's
--max-logprobs, 20 by default). They're stored next to the windows:
    tokens.npy  [N, T+1] uint32   the windows (inputs and targets)
    top_ids.npy [N, T, k] int32   teacher's k most likely next tokens at each position
    top_lp.npy  [N, T, k] float16 their log-probabilities
    done.npy    [N] bool          which windows are scored (resume)
Size: 120 bytes per token at k = 20 x (4 + 2): 1B tokens would be 120 GB, so
score a subset (50k windows of 1024 = 51M tokens = ~6 GB) and mix it with
ordinary next-token training.
"""

from __future__ import annotations

import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

from .data.ingest import find_files, iter_documents

# Each template turns a snippet into a different kind of teaching text. Q&A first:
# facts are our models' weakest category, and short answers teach "answer, then stop".
TEMPLATES = {
    "qa": ("Read the passage, then write 5 short question-and-answer pairs about the most important facts "
           "in it. Format each as 'Q: ...' on one line and 'A: ...' on the next, with a short, exact answer. "
           "Only use facts stated in the passage.\n\nPassage:\n{snippet}"),
    "textbook": ("Rewrite the following passage as a clear, accurate section of a textbook for a curious "
                 "high-school student. Explain the key ideas step by step and define any technical terms. "
                 "Do not mention the passage itself.\n\nPassage:\n{snippet}"),
    "example": ("Using the topic of the passage below, write one worked example or short explanation with "
                "reasoning that a student could learn from (a calculation, a cause and effect, or a "
                "comparison). Be concise and correct.\n\nPassage:\n{snippet}"),
}


def make_prompts(inputs: list[str | Path], n: int, seed: int = 0, max_chars: int = 1500,
                 templates: Iterable[str] = TEMPLATES) -> list[dict]:
    """n prompts from snippets of the corpus: about n / (number of files) documents
    from each file (a random 1 in 5 of its documents of 200+ characters), cut to
    max_chars at a paragraph or sentence end, templates in rotation."""
    names = list(templates)
    rng = random.Random(seed)
    files = find_files(inputs)
    rng.shuffle(files)
    docs, seen = [], set()
    per_file = max(1, n // max(1, len(files)) + 1)
    for keep in (0.2, 1.0):     # a second pass, taking every document, if the corpus is small
        for f in files:
            taken = 0
            for i, d in enumerate(iter_documents([f])):
                if len(d.text) < 200 or (f, i) in seen or rng.random() > keep:   # skip stubs; spread out
                    continue
                seen.add((f, i))
                docs.append((f.parent.name, d.text))
                taken += 1
                if taken >= per_file or len(docs) >= n:
                    break
            if len(docs) >= n:
                break
        if len(docs) >= n:
            break
        per_file = n
    rng.shuffle(docs)
    out = []
    for i, (source, text) in enumerate(docs[:n]):
        snippet = text[:max_chars]
        if len(text) > max_chars:   # end at the last paragraph or sentence break
            cut = max(snippet.rfind("\n\n"), snippet.rfind(". "))
            snippet = snippet[:cut + 1] if cut > max_chars // 2 else snippet
        name = names[i % len(names)]
        out.append({"id": f"{seed}-{i}", "template": name, "source": source,
                    "prompt": TEMPLATES[name].format(snippet=snippet.strip())})
    return out


_THINK = re.compile(r"<think>.*?</think>\s*", re.S)


def teach_text(server: str, model: str, prompts: list[dict], out: str | Path, max_tokens: int = 700,
               temperature: float = 0.7, workers: int = 32, api_key_env: str | None = None,
               no_thinking: bool = True, log: Callable[[str], None] = print, request=None) -> int:
    """Send every prompt not already in `out` to the teacher (in parallel: vLLM batches
    concurrent requests) and append the answers. Returns how many were written."""
    from .. import net

    request = request or net.request_json
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = [p for p in prompts if p["id"] not in done]
    log(f"{len(done)} already done, {len(todo)} to go -> {out}")
    url = server.rstrip("/") + "/v1/chat/completions"
    lock = threading.Lock()
    written = 0

    def one(p: dict) -> None:
        nonlocal written
        payload = {"model": model, "messages": [{"role": "user", "content": p["prompt"]}],
                   "max_tokens": max_tokens, "temperature": temperature}
        if no_thinking:   # Qwen-style reasoning models: answer directly (vLLM passes this to the template)
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            r = request(url, payload, api_key_env=api_key_env)
            text = _THINK.sub("", r["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:  # noqa: BLE001 - one failed request shouldn't stop the job
            log(f"{p['id']}: {type(e).__name__}: {e}")
            return
        if not text:
            return
        with lock:
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps({"text": text, "source": f"teacher:{model}", "template": p["template"],
                                    "from": p["source"], "id": p["id"]}, ensure_ascii=False) + "\n")
            written += 1
            if written % 200 == 0:
                log(f"{written} / {len(todo)}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, todo))
    log(f"wrote {written} texts")
    return written


# --------------------------------------------------------------------------- option B
class TeacherSet:
    """Scored windows on disk (see the module docstring), memory-mapped."""

    def __init__(self, path: str | Path, mode: str = "r"):
        import numpy as np

        self.dir = Path(path)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.k, self.seq_len, self.n = self.meta["k"], self.meta["seq_len"], self.meta["n_windows"]
        mm = lambda name: np.load(self.dir / name, mmap_mode=mode)  # noqa: E731
        self.tokens, self.top_ids, self.top_lp, self.done = (mm(f) for f in ("tokens.npy", "top_ids.npy",
                                                                           "top_lp.npy", "done.npy"))

    @classmethod
    def create(cls, path: str | Path, n_windows: int, seq_len: int, k: int, meta: dict) -> "TeacherSet":
        import numpy as np

        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps({"k": k, "seq_len": seq_len, "n_windows": n_windows, **meta},
                                                indent=1) + "\n")
        for name, shape, dt in (("tokens.npy", (n_windows, seq_len + 1), np.uint32),
                                ("top_ids.npy", (n_windows, seq_len, k), np.int32),
                                ("top_lp.npy", (n_windows, seq_len, k), np.float16),
                                ("done.npy", (n_windows,), np.bool_)):
            np.lib.format.open_memmap(d / name, mode="w+", dtype=dt, shape=shape)
        return cls(d, mode="r+")

    def batch(self, batch_size: int, seed: int, step: int, micro_step: int = 0):
        """Stateless random scored windows (like ShardDataset.batch): x, y [B, T] and the
        teacher's top-k ids / log-probs [B, T, k] for each position's next token."""
        import numpy as np
        import torch

        scored = np.flatnonzero(self.done)
        if not len(scored):
            raise ValueError(f"{self.dir}: no scored windows yet")
        rows = np.sort(np.random.default_rng([seed, step, micro_step, 7]).choice(scored, size=batch_size))
        buf = torch.from_numpy(self.tokens[rows].astype(np.int64))
        return (buf[:, :-1], buf[:, 1:], torch.from_numpy(self.top_ids[rows].astype(np.int64)),
                torch.from_numpy(self.top_lp[rows].astype(np.float32)))


def parse_prompt_logprobs(pl: list, k: int) -> tuple[list[list[int]], list[list[float]]]:
    """vLLM's prompt_logprobs (one {token id: {logprob, rank}} dict per prompt
    position, None for the first) -> top-k ids and log-probs predicting positions
    1..T. Each dict holds the top k plus the actual token, which is dropped unless
    it's in the top k. Missing entries (shouldn't happen) are id -1."""
    ids, lps = [], []
    for entry in pl[1:]:
        best = sorted(((v["rank"], int(t), v["logprob"]) for t, v in (entry or {}).items()))[:k]
        best = [b for b in best if b[0] <= k]
        ids.append([t for _, t, _ in best] + [-1] * (k - len(best)))
        lps.append([lp for _, _, lp in best] + [-1e4] * (k - len(best)))
    return ids, lps


def score_logits(server: str, model: str, shards: str | Path, out: str | Path, n_windows: int,
                 seq_len: int = 1024, k: int = 20, seed: int = 0, workers: int = 8,
                 api_key_env: str | None = None, log: Callable[[str], None] = print, request=None) -> TeacherSet:
    """Score n_windows random windows of the shards' train split with the teacher.
    The shards must be made with the teacher's tokenizer. Resumes an existing `out`."""
    import numpy as np

    from .. import net
    from .data.loader import ShardDataset

    request = request or net.request_json
    ds = ShardDataset(shards, "train")
    out = Path(out)
    if (out / "meta.json").exists():
        ts = TeacherSet(out, mode="r+")
        if (ts.n, ts.seq_len, ts.k) != (n_windows, seq_len, k):
            raise ValueError(f"{out} has {ts.n} windows of {ts.seq_len}, k = {ts.k}; use a new folder")
    else:
        ts = TeacherSet.create(out, n_windows, seq_len, k, {
            "model": model, "tokenizer_fingerprint": ds.meta["tokenizer_fingerprint"], "seed": seed,
            "shards": str(shards)})
        # the windows: like training batches, one window per index (stateless, reproducible)
        for i in range(n_windows):
            x, y = ds.batch(1, seq_len, seed, i, 0, rank=99)
            ts.tokens[i] = np.concatenate([x[0].numpy(), y[0, -1:].numpy()]).astype(np.uint32)
        ts.tokens.flush()
    todo = [i for i in range(n_windows) if not ts.done[i]]
    log(f"{n_windows - len(todo)} windows already scored, {len(todo)} to go -> {out}")
    url = server.rstrip("/") + "/v1/completions"
    lock = threading.Lock()
    count = 0

    def one(i: int) -> None:
        nonlocal count
        payload = {"model": model, "prompt": ts.tokens[i].tolist(), "max_tokens": 1, "prompt_logprobs": k,
                   "temperature": 1.0}
        try:
            r = request(url, payload, api_key_env=api_key_env)
            ids, lps = parse_prompt_logprobs(r["choices"][0]["prompt_logprobs"], k)
        except Exception as e:  # noqa: BLE001 - retried on the next run
            log(f"window {i}: {type(e).__name__}: {e}")
            return
        if len(ids) != seq_len:
            log(f"window {i}: {len(ids)} positions scored, expected {seq_len}")
            return
        with lock:
            ts.top_ids[i] = np.asarray(ids, dtype=np.int32)
            ts.top_lp[i] = np.asarray(lps, dtype=np.float16)
            ts.done[i] = True
            count += 1
            if count % 100 == 0:
                for a in (ts.top_ids, ts.top_lp, ts.done):
                    a.flush()
                log(f"{count} / {len(todo)}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, todo))
    for a in (ts.top_ids, ts.top_lp, ts.done):
        a.flush()
    log(f"scored {count} windows ({int(ts.done.sum())} of {n_windows} done)")
    return ts


def distill_loss(logits, top_ids, top_lp):
    """Cross-entropy of the student against the teacher's top-k distribution
    (renormalized over the k): -sum_i p_teacher(i) log p_student(i). Equal to the KL
    divergence up to a constant, so it pulls the student's probabilities toward the
    teacher's on the tokens the teacher considers likely. Positions with no teacher
    entries (id -1) are ignored."""
    import torch

    valid = top_ids >= 0
    p = torch.softmax(top_lp.float().masked_fill(~valid, -1e4), -1)
    logq = logits.float().log_softmax(-1).gather(-1, top_ids.clamp_min(0))
    per_pos = -(p * logq * valid).sum(-1)
    return per_pos[valid.any(-1)].mean()
