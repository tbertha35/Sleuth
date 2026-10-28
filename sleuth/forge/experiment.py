"""Experiments: train several variants of one config and compare them in one table.

The research ideas (MTP heads, early exits, looped blocks, memory tables, ternary
weights, distillation, ...) are all switches on ModelConfig / TrainConfig. An
experiment file names a base config, overrides shared by every variant (usually a
shorter budget), and the variants:

    name = "ideas-1"
    base = "configs/mix-20m.toml"
    set = ["train.max_steps=2500", "data.shards=data/shards/mix-3b"]

    [[variant]]
    name = "baseline"

    [[variant]]
    name = "mtp1"
    set = ["model.mtp_heads=1"]

`sleuth experiment FILE` trains the variants one after another (never two on the
GPU at once), then measures each the same way:

    val bpb                the run's whole-split validation (sleuth eval)
    knowledge few-shot     answer loss and exact match (sleuth score --few-shot)
    decode tok/s           greedy, warm, CUDA graph if possible
    + drafter              tok/s and guesses accepted with the model's own drafter
                           (MTP heads, early exit), if it has one
    train tok/s, params, wall time

Results go to runs/experiments/<name>/results.json and results.md. A variant
that already has results is skipped, so an interrupted experiment can be
restarted with the same command (a half-trained run is resumed).
"""

from __future__ import annotations

import contextlib
import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..config import apply_overrides, from_dict, load_toml, to_dict

PROMPTS = [  # decode-speed prompts: a little prose, a little code (acceptance differs)
    "Photosynthesis is the process by which",
    "def fibonacci(n):",
]


@dataclass
class Variant:
    name: str
    set: list[str] = field(default_factory=list)


@dataclass
class Experiment:
    name: str
    base: str
    set: list[str] = field(default_factory=list)
    variants: list[Variant] = field(default_factory=list)
    suite: str = "suites/knowledge"
    decode_tokens: int = 128

    @classmethod
    def load(cls, path: str | Path) -> "Experiment":
        d = load_toml(path)
        variants = [Variant(v["name"], list(v.get("set", []))) for v in d.pop("variant", [])]
        if not variants:
            raise ValueError(f"{path}: no [[variant]] tables")
        names = [v.name for v in variants]
        if len(set(names)) != len(names):
            raise ValueError(f"{path}: variant names must be unique")
        return cls(variants=variants, **d)

    def run_config(self, v: Variant):
        """The RunConfig of one variant: base file, then shared overrides, then the variant's."""
        from .train.config import RunConfig

        d = load_toml(self.base)
        apply_overrides(d, self.set + v.set + [f"name={self.name}-{v.name}"])
        return from_dict(RunConfig, d)


def decode_speed(model, tok, n_tokens: int = 128, log: Callable[[str], None] = print, name: str = "",
                 reps: int = 3) -> dict:
    """Greedy decode tok/s (CUDA graph if possible), and with the model's own drafter.
    Small models decode in well under a second, so one run is at the mercy of whatever
    else the CPU is doing: after a warm-up run (graph capture, kernel compiles) take the
    best of `reps`, then average over the prompts."""
    from .infer.cuda_graph import graph_unsupported
    from .infer.generate import generate
    from .infer.mtp import builtin_drafter
    from .infer.sampling import SamplingConfig
    from .infer.speculative import speculative_generate

    greedy = SamplingConfig(temperature=0)
    graph = graph_unsupported(model) is None
    drafter = builtin_drafter(model)
    speeds, spec_speeds, acc, drafted = [], [], 0, 0
    for p in PROMPTS:
        ids = [tok.start_id] + tok.encode(p)
        runs = [generate(model, ids, n_tokens, greedy, vocab_limit=tok.vocab_size, cuda_graph=graph)
                for _ in range(reps + 1)][1:]
        speeds.append(max(r.decode_tok_per_s for r in runs))
        if drafter is not None:
            sruns = [speculative_generate(model, ids, n_tokens, greedy, draft=drafter, k=3,
                                          vocab_limit=tok.vocab_size, cuda_graph=graph) for _ in range(reps + 1)][1:]
            s = sruns[-1]
            if s.new_ids != runs[-1].new_ids:
                log(f"warning: {name}: drafted text differs from greedy for {p!r}")
            spec_speeds.append(max(r.decode_tok_per_s for r in sruns))
            acc += s.stats["speculative"]["accepted"]
            drafted += s.stats["speculative"]["drafted"]
    out = {"decode_tok_per_s": sum(speeds) / len(speeds), "cuda_graph": graph}
    if spec_speeds:
        out["draft_tok_per_s"] = sum(spec_speeds) / len(spec_speeds)
        out["draft_acceptance"] = acc / max(1, drafted)
    return out


def measure(run_dir: Path, exp: Experiment, log: Callable[[str], None] = print) -> dict:
    """Everything in the results table for one finished run."""
    import torch

    from ..analyzer.common import encode_cases, score_cases
    from ..analyzer.suites import load_suite
    from .data.loader import ShardDataset
    from .data.shard import doc_start_id
    from .tokenizer.bpe import Tokenizer
    from .train.checkpoint import load_model
    from .train.trainer import evaluate, resolve_device

    cfg = json.loads((run_dir / "config.json").read_text())
    device = resolve_device("auto")
    gc.collect()                       # free what training left behind before loading the model again
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model, meta = load_model(run_dir, device=device, nearest=True)
    tok = Tokenizer.load(run_dir / "tokenizer.json")
    out: dict = {"run": run_dir.name, "step": int(meta["step"]), "params": model.num_params()}

    # training speed and wall time, from the metrics log
    recs = [json.loads(s) for s in (run_dir / "metrics.jsonl").read_text().splitlines() if s.strip()]
    tps = [r["tok_per_s"] for r in recs if "tok_per_s" in r]
    out["train_tok_per_s"] = sum(tps[len(tps) // 2:]) / max(1, len(tps[len(tps) // 2:]))   # second half: warm
    times = [r["time"] for r in recs if "time" in r]
    out["train_minutes"] = (times[-1] - times[0]) / 60 if len(times) > 1 else None

    # val bpb on the same windows as `sleuth eval`
    ds = ShardDataset(Path(cfg["data"]["shards"]), "val")
    byte_lens = torch.tensor(tok.token_byte_lengths() + [0] * (model.cfg.padded_vocab - tok.vocab_size), device=device)
    autocast = ((lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device.type == "cuda"
                else contextlib.nullcontext)
    masked = cfg["train"].get("doc_mask", False)
    ev = evaluate(model, ds, 8, cfg["train"]["seq_len"], cfg["train"].get("eval_windows", 640), byte_lens,
                  device, autocast, doc_start_id(tok) if masked else None)
    out["val_loss"], out["val_bpb"] = ev["val_loss"], ev["val_bpb"]
    out["val_by_source"] = ev.get("val_by_source")

    # knowledge suite, few-shot (the number that moves at these sizes)
    f = Path(exp.suite) / "_fewshot.txt"
    prefix = [tok.start_id] + tok.encode(f.read_text(encoding="utf-8")) if f.exists() else None
    enc = encode_cases(tok, load_suite(exp.suite), model.cfg.max_seq_len, prefix)
    res = score_cases(model, enc, min(tok.vocab_size, model.cfg.padded_vocab), device)
    out["knowledge"] = {c: {"answer_loss": r["expected_loss"], "exact": r["exact_match"]} for c, r in res.items()}

    if device.type == "cuda":
        out.update(decode_speed(model, tok, exp.decode_tokens, log, run_dir.name))
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def table(exp: Experiment, results: dict[str, dict]) -> str:
    """Markdown table of the results, in variant order, first variant as the reference."""
    cols = ["variant", "params", "val bpb", "Δ bpb", "few-shot ans. loss", "exact", "decode tok/s",
            "+drafter tok/s", "accepted", "train tok/s", "min"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    ref = None
    for v in exp.variants:
        r = results.get(v.name)
        if not r:
            lines.append(f"| {v.name} | (not run) |" + " |" * (len(cols) - 2))
            continue
        ref = ref if ref is not None else r["val_bpb"]
        k = r["knowledge"].get("overall", {})

        def fmt(x, f):
            return f.format(x) if x is not None else "-"

        lines.append("| " + " | ".join([
            v.name, f"{r['params'] / 1e6:.1f}M", f"{r['val_bpb']:.4f}", f"{r['val_bpb'] - ref:+.4f}",
            fmt(k.get("answer_loss"), "{:.3f}"), fmt(k.get("exact"), "{:.1%}"),
            fmt(r.get("decode_tok_per_s"), "{:.0f}"), fmt(r.get("draft_tok_per_s"), "{:.0f}"),
            fmt(r.get("draft_acceptance"), "{:.0%}"), fmt(r.get("train_tok_per_s"), "{:,.0f}"),
            fmt(r.get("train_minutes"), "{:.0f}")]) + " |")
    return "\n".join(lines)


def run_experiment(path: str | Path, only: list[str] | None = None, log: Callable[[str], None] = print,
                   train_fn=None) -> Path:
    """Train and measure every variant (or just `only`). Returns the results folder."""
    from .train.trainer import train

    train_fn = train_fn or train
    exp = Experiment.load(path)
    out_dir = Path(exp.run_config(exp.variants[0]).runs_dir) / "experiments" / exp.name
    out_dir.mkdir(parents=True, exist_ok=True)
    res_file = out_dir / "results.json"
    state = json.loads(res_file.read_text()) if res_file.exists() else {"runs": {}, "results": {}}
    for v in exp.variants:
        if only and v.name not in only:
            continue
        if v.name in state["results"]:
            log(f"[{v.name}] already measured: skipping")
            continue
        cfg = exp.run_config(v)
        run_dir = state["runs"].get(v.name)
        t0 = time.time()
        if run_dir and Path(run_dir).exists() and not (Path(run_dir) / "snapshots" / f"step_{cfg.train.max_steps:07d}").exists():
            log(f"[{v.name}] resuming {run_dir}")
            run_dir = train_fn(cfg, run_dir=Path(run_dir), resume=True)
        elif not run_dir or not Path(run_dir).exists():
            log(f"[{v.name}] training: {' '.join(v.set) or '(base config)'}")
            run_dir = train_fn(cfg)
        state["runs"][v.name] = str(run_dir)
        res_file.write_text(json.dumps(state, indent=1) + "\n")
        log(f"[{v.name}] trained in {(time.time() - t0) / 60:.0f} min; measuring")
        r = measure(Path(run_dir), exp, log)
        r["set"] = v.set
        r["config"] = to_dict(cfg)
        state["results"][v.name] = r
        res_file.write_text(json.dumps(state, indent=1) + "\n")
        (out_dir / "results.md").write_text(f"# Experiment {exp.name}\n\nBase: `{exp.base}`, shared: "
                                            f"`{' '.join(exp.set)}`\n\n{table(exp, state['results'])}\n",
                                            encoding="utf-8")
        log(table(exp, state["results"]))
    return out_dir
