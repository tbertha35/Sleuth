"""Command line: `python -m sleuth <command> ...`  (try `python -m sleuth -h`).

Typical first session:
    python -m sleuth synth                     # make a synthetic corpus
    python -m sleuth tokenizer --vocab-size 1024
    python -m sleuth shard
    python -m sleuth train configs/smoke-cpu.toml
    python -m sleuth generate runs/<run> --prompt "Once upon a time"
    python -m sleuth analyze runs/<run>
    python -m sleuth ui
or all of it at once:  python -m sleuth demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def data_root() -> Path:
    """$SLEUTH_DATA (e.g. an NVMe path) or ./data."""
    return Path(os.environ.get("SLEUTH_DATA") or "data")


def cmd_synth(a):
    from .forge.data.synthetic import generate_corpus

    stats = generate_corpus(Path(a.out), a.mb, seed=a.seed)
    for cat, s in stats.items():
        print(f"{cat:6s} {s['docs']:8,d} docs {s['bytes'] / 1e6:8.1f} MB")
    print(f"wrote {a.out}")


def cmd_tokenizer(a):
    from .forge.tokenizer.train_bpe import train_tokenizer

    train_tokenizer(a.input, a.vocab_size, a.out, workers=a.workers,
                    max_bytes=int(a.max_mb * 1e6) if a.max_mb else None, text_field=a.text_field)


def cmd_tokenize(a):
    from .forge.tokenizer.bpe import Tokenizer

    tok = Tokenizer.load(a.tokenizer)
    ids = tok.encode(a.text, allowed_special="all" if a.allow_special else "none")
    print(f"{len(a.text.encode())} bytes -> {len(ids)} tokens")
    print(" | ".join(repr(tok.decode([i]))[1:-1] for i in ids))
    print(ids)


def cmd_shard(a):
    from .forge.data.shard import build_shards

    build_shards(a.input, a.tokenizer, a.out, val_fraction=a.val_fraction, workers=a.workers,
                 shard_tokens=int(a.shard_mtokens * 2**20), text_field=a.text_field)


def cmd_train(a):
    from .config import from_dict, load_config
    from .forge.train.config import RunConfig
    from .forge.train.trainer import train

    if a.resume:
        run_dir = Path(a.resume)
        cfg = from_dict(RunConfig, json.loads((run_dir / "config.json").read_text()))
        if a.set:
            print("note: --set is ignored when resuming (the run's saved config is used)")
        train(cfg, run_dir=run_dir, resume=True, stop_at_step=a.stop_at_step)
    else:
        if not a.config:
            sys.exit("give a config file, or --resume RUN_DIR")
        cfg = load_config(RunConfig, a.config, a.set)
        train(cfg, stop_at_step=a.stop_at_step)


def cmd_info(a):
    from .config import load_config
    from .forge.model.flops import describe
    from .forge.train.config import RunConfig

    cfg = load_config(RunConfig, a.config, a.set)
    if cfg.model.vocab_size == 0:
        meta = Path(cfg.data.shards) / "meta.json"
        vocab = json.loads(meta.read_text())["vocab_size"] if meta.exists() else 8192
        import dataclasses

        cfg.model = dataclasses.replace(cfg.model, vocab_size=vocab)
        print(f"(vocab_size from {'shards' if meta.exists() else 'a placeholder of 8192'}: {vocab})")
    tc = cfg.train
    tokens = tc.max_steps * tc.micro_batch_size * tc.seq_len * tc.grad_accum
    from .forge.model.flops import param_counts, train_flops_per_token

    print(describe(cfg.model, tc.seq_len))
    print(f"head_dim {cfg.model.head_dim}, kv heads {cfg.model.kv_heads}, ffn hidden {cfg.model.hidden}, "
          f"padded vocab {cfg.model.padded_vocab}")
    print(f"training: {tokens / 1e6:.1f}M tokens = {tokens / param_counts(cfg.model)['non_embedding']:.1f} tokens per "
          f"non-embedding param, {train_flops_per_token(cfg.model, tc.seq_len) * tokens:.2e} FLOPs")


def cmd_generate(a):
    import codecs

    import torch

    from .forge.infer.generate import generate
    from .forge.infer.sampling import SamplingConfig
    from .forge.tokenizer.bpe import Tokenizer
    from .forge.train.checkpoint import load_model, resolve_weights
    from .forge.train.trainer import resolve_device

    device = resolve_device(a.device)
    model, meta = load_model(a.path, step=a.step, device=device)
    weights = resolve_weights(a.path, a.step)
    tok_path = a.tokenizer or next((p / "tokenizer.json" for p in weights.parents if (p / "tokenizer.json").exists()), None)
    if tok_path is None:
        sys.exit("couldn't find tokenizer.json next to the checkpoint; pass --tokenizer")
    tok = Tokenizer.load(tok_path)
    sampling = SamplingConfig(a.temperature, a.top_k, a.top_p, a.min_p, a.repetition_penalty)
    print(f"[step {meta['step']}, {model.num_params() / 1e6:.2f}M params, {device}]", file=sys.stderr, flush=True)
    # The first forward pass pays one-time setup (on GPU: loading kernels and
    # creating cuBLAS handles; on CPU: allocating thread pools and buffers).
    # Do it now so it doesn't distort the first sample's speed reading.
    with torch.no_grad():
        model(torch.tensor([[tok.bos_id] * 8], device=device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    for i in range(a.samples):
        prompt_ids = [tok.bos_id] + tok.encode(a.prompt)
        print(a.prompt, end="", flush=True)
        # One token can end in the middle of a multi-byte character; the
        # incremental decoder holds those bytes until the character is complete.
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        def on_token(t):
            if t != tok.eos_id:
                print(decoder.decode(tok.decode_bytes([t])), end="", flush=True)

        res = generate(model, prompt_ids, a.max_new_tokens, sampling, stop_ids={tok.eos_id},
                       vocab_limit=tok.vocab_size, seed=None if a.seed is None else a.seed + i,
                       on_token=on_token)
        print(decoder.decode(b"", final=True), flush=True)
        print(f"[{len(res.new_ids)} tokens, stop: {res.stop_reason}, prefill {res.prefill_tok_per_s:,.0f} tok/s, "
              f"decode {res.decode_tok_per_s:,.0f} tok/s]", file=sys.stderr, flush=True)
        if i + 1 < a.samples:
            print("-" * 40, flush=True)


def cmd_analyze(a):
    from .analyzer.timeline import run_timeline
    from .forge.train.trainer import resolve_device

    views = [v.strip() for v in a.views.split(",")] if a.views else None
    steps = [int(s) for s in a.steps.split(",")] if a.steps else None
    options = {"neuron_stats": {"k": a.k}, "ablation": {"method": a.ablation}}
    for suite in a.suite or ["suites/basic"]:
        run_timeline(a.run, suite, device=str(resolve_device(a.device)), views=views, force=a.force,
                     steps=steps, options=options)


def cmd_ui(a):
    from .env import runs_root
    from .ui.server import serve

    serve(a.runs or runs_root(), a.host, a.port)


def cmd_probe(a):
    import runpy

    script = Path(__file__).resolve().parent.parent / "scripts" / "probe_hardware.py"
    sys.argv = [str(script), *a.args]
    runpy.run_path(str(script), run_name="__main__")


def cmd_demo(a):
    """Synthetic data -> tokenizer -> shards -> train -> analyze, skipping steps already done."""
    from .config import load_config
    from .forge.train.config import RunConfig
    from .forge.train.trainer import train
    from .analyzer.timeline import run_timeline

    root = data_root()
    raw, tok, shards = root / "raw" / "synthetic", root / "tokenizers" / "synthetic-1k.json", root / "shards" / "synthetic"
    if not raw.exists():
        print("== 1/5 synthetic corpus")
        cmd_synth(argparse.Namespace(out=str(raw), mb=a.mb, seed=0))
    if not tok.exists():
        print("== 2/5 tokenizer")
        cmd_tokenizer(argparse.Namespace(input=[str(raw)], vocab_size=1024, out=str(tok), workers=a.workers,
                                         max_mb=None, text_field="text"))
    if not (shards / "meta.json").exists():
        print("== 3/5 shards")
        cmd_shard(argparse.Namespace(input=[str(raw)], tokenizer=str(tok), out=str(shards), val_fraction=0.01,
                                     workers=a.workers, shard_mtokens=256, text_field="text"))
    print("== 4/5 train")
    cfg = load_config(RunConfig, a.config, a.set)
    # Set the path directly rather than via a "--set" string: a Windows path
    # like data\shards inside TOML quotes would be read as escape sequences.
    cfg.data.shards = str(shards)
    run = train(cfg)
    print("== 5/5 analyze")
    run_timeline(run, "suites/basic", device="cuda" if cfg.train.device != "cpu" and _cuda() else "cpu")
    print(f"\nnext:\n  python -m sleuth generate {run} --prompt 'Once upon a time'\n  python -m sleuth ui")


def _cuda() -> bool:
    import torch

    return torch.cuda.is_available()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sleuth", description="Sleuth: build and understand LLMs, locally.")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = data_root()
    cpus = max(1, (os.cpu_count() or 2) - 1)

    s = sub.add_parser("synth", help="generate a synthetic training corpus (no downloads)")
    s.add_argument("--out", default=str(d / "raw" / "synthetic"))
    s.add_argument("--mb", type=float, default=50)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("tokenizer", help="train a byte-level BPE tokenizer")
    s.add_argument("--input", nargs="+", default=[str(d / "raw" / "synthetic")], help="files or directories")
    s.add_argument("--vocab-size", type=int, default=8192)
    s.add_argument("--out", default=None, help="default: data/tokenizers/<input>-<vocab>.json")
    s.add_argument("--workers", type=int, default=cpus)
    s.add_argument("--max-mb", type=float, default=None, help="only read this much text (faster on huge corpora)")
    s.add_argument("--text-field", default="text", help="JSON field holding the text in .jsonl files")
    s.set_defaults(fn=cmd_tokenizer)

    s = sub.add_parser("tokenize", help="show how a tokenizer splits some text")
    s.add_argument("--tokenizer", required=True)
    s.add_argument("text")
    s.add_argument("--allow-special", action="store_true")
    s.set_defaults(fn=cmd_tokenize)

    s = sub.add_parser("shard", help="tokenize documents into binary training shards")
    s.add_argument("--input", nargs="+", default=[str(d / "raw" / "synthetic")])
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--val-fraction", type=float, default=0.005)
    s.add_argument("--workers", type=int, default=cpus)
    s.add_argument("--shard-mtokens", type=float, default=256, help="tokens per shard file, in units of 2^20")
    s.add_argument("--text-field", default="text")
    s.set_defaults(fn=cmd_shard)

    s = sub.add_parser("train", help="train a model from a config (or resume a run)")
    s.add_argument("config", nargs="?")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override, e.g. --set train.lr=3e-4")
    s.add_argument("--resume", metavar="RUN_DIR")
    s.add_argument("--stop-at-step", type=int, default=None, help="pause after this step (resumable)")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("info", help="parameter count, FLOPs and token budget of a config")
    s.add_argument("config")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    s.set_defaults(fn=cmd_info)

    s = sub.add_parser("generate", help="generate text from a run or snapshot")
    s.add_argument("path", help="run dir, snapshot dir, or model.safetensors")
    s.add_argument("--prompt", default="")
    s.add_argument("--step", type=int, default=None, help="which snapshot (default: latest)")
    s.add_argument("--max-new-tokens", type=int, default=200)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--top-k", type=int, default=0)
    s.add_argument("--top-p", type=float, default=1.0)
    s.add_argument("--min-p", type=float, default=0.05)
    s.add_argument("--repetition-penalty", type=float, default=1.0)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--samples", type=int, default=1)
    s.add_argument("--device", default="auto")
    s.add_argument("--tokenizer", default=None)
    s.set_defaults(fn=cmd_generate)

    s = sub.add_parser("analyze", help="run analyzer views over every checkpoint of a run")
    s.add_argument("run")
    s.add_argument("--suite", action="append", default=None,
                   help="test suite directory (repeatable; default suites/basic). "
                        "For TinyStories models also try suites/stories")
    s.add_argument("--views", default=None,
                   help="comma-separated: neuron_stats,attention,logit_lens,ablation (default: all)")
    s.add_argument("--steps", default=None, help="only these checkpoints, e.g. 0,500,5000 (default: all)")
    s.add_argument("--device", default="auto")
    s.add_argument("--k", type=float, default=1.0, help="neuron firing threshold, in units of the layer's RMS")
    s.add_argument("--ablation", choices=["mean", "zero"], default="mean", help="how ablated parts are replaced")
    s.add_argument("--force", action="store_true", help="redo checkpoints that were already analyzed")
    s.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("ui", help="serve the local dashboard")
    s.add_argument("--runs", default=None, help="runs directory (default: $SLEUTH_RUNS or ./runs)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(fn=cmd_ui)

    s = sub.add_parser("probe", help="run scripts/probe_hardware.py (pass its options after --)")
    s.add_argument("args", nargs=argparse.REMAINDER)
    s.set_defaults(fn=cmd_probe)

    s = sub.add_parser("demo", help="synthetic data -> tokenizer -> train -> analyze, end to end")
    s.add_argument("--config", default="configs/smoke-cpu.toml")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    s.add_argument("--mb", type=float, default=20, help="size of the synthetic corpus")
    s.add_argument("--workers", type=int, default=cpus)
    s.set_defaults(fn=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    # Generated text can contain any character. A Windows console or a
    # redirected stdout may use a legacy code page that can't encode it;
    # print a replacement character instead of crashing.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    if args.cmd == "tokenizer" and args.out is None:
        name = Path(args.input[0]).name or "corpus"
        args.out = str(data_root() / "tokenizers" / f"{name}-{args.vocab_size}.json")
    try:
        args.fn(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    return 0
