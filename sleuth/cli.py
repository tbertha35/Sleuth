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
    from .models import load_tokenizer

    p = Path(a.tokenizer)
    # Our tokenizer.json, or the tokenizer of any model we can load (.gguf, HF folder, run).
    tok = Tokenizer.load(p) if p.is_file() and p.suffix == ".json" else load_tokenizer(p)
    if a.chat:
        from .formats.chat import encode_chat

        ids = encode_chat(tok, [{"role": "user", "content": a.text}])
        print(f"chat-formatted: {tok.decode(ids)!r}")
        print(f"{len(ids)} tokens")
        print(" | ".join(repr(tok.decode([i]))[1:-1] for i in ids))
        print(ids)
        return
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

    from .forge.infer.generate import generate
    from .forge.infer.sampling import SamplingConfig
    from .forge.train.trainer import resolve_device
    from .models import load_any

    device = resolve_device(a.device)
    model, tok, meta = load_any(a.path, device=device, dtype=_dtype(a.dtype), step=a.step,
                                keep_quantized=_keep_quantized(a.keep_quantized))
    if a.tokenizer:
        from .forge.tokenizer.bpe import Tokenizer

        tok = Tokenizer.load(a.tokenizer)
    sampling = SamplingConfig(a.temperature, a.top_k, a.top_p, a.min_p, a.repetition_penalty)
    if a.step is not None and meta.get("step") is not None and int(meta["step"]) != a.step:
        print(f"(no snapshot at step {a.step}; using the nearest one, step {meta['step']})", file=sys.stderr)
    where = f"step {meta['step']}" if meta.get("step") is not None else meta["format"]
    if meta.get("kept_quantized"):
        where += f", {meta['kept_quantized']} layers kept quantized"
    dt = next(model.parameters()).dtype
    shown = f"{dt} weights, bf16 autocast" if device.type == "cuda" and str(dt) == "torch.float32" else str(dt)
    print(f"[{where}, {model.num_params() / 1e6:.2f}M params, {device}, {shown}]", file=sys.stderr, flush=True)
    # The first run of anything pays one-time setup (on GPU: loading kernels,
    # creating cuBLAS handles, the KV cache and sampling kernels; on CPU:
    # thread pools and buffers). A throwaway 2-token generation exercises the
    # whole path, so the first real sample's speed reading isn't distorted.
    # (Warming up only the forward pass, as before, left the sampling path cold.)
    generate(model, [tok.start_id] * 8, 2, sampling, seed=0)
    stops = _stop_ids(tok)
    for i in range(a.samples):
        prompt_ids = _prompt_ids(tok, a)
        if not a.chat:
            print(a.prompt, end="", flush=True)
        # One token can end in the middle of a multi-byte character; the
        # incremental decoder holds those bytes until the character is complete.
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        def on_token(t):
            if t not in stops:
                print(decoder.decode(tok.decode_bytes([t])), end="", flush=True)

        res = generate(model, prompt_ids, a.max_new_tokens, sampling, stop_ids=stops,
                       vocab_limit=tok.vocab_size, seed=None if a.seed is None else a.seed + i,
                       on_token=on_token)
        print(decoder.decode(b"", final=True), flush=True)
        mem = ""
        if device.type == "cuda":
            import torch

            mem = f", peak GPU memory {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB"
        print(f"[{len(res.new_ids)} tokens, stop: {res.stop_reason}, prompt {len(prompt_ids)} tokens in "
              f"{res.prefill_seconds * 1000:.0f} ms, decode {res.decode_tok_per_s:,.1f} tok/s{mem}]",
              file=sys.stderr, flush=True)
        if i + 1 < a.samples:
            print("-" * 40, flush=True)


def _prompt_ids(tok, a) -> list[int]:
    """The prompt as token IDs: plain (start token + text), or with --chat
    wrapped in the model's chat format as a user turn (formats/chat.py)."""
    if not getattr(a, "chat", False):
        return [tok.start_id] + tok.encode(a.prompt)
    from .formats.chat import encode_chat

    messages = ([{"role": "system", "content": a.system}] if a.system else []) + [{"role": "user", "content": a.prompt}]
    return encode_chat(tok, messages, thinking=a.think)


def _add_chat_args(s):
    s.add_argument("--chat", action="store_true",
                   help="wrap the prompt as a user turn in the model's chat format (instruction-tuned models)")
    s.add_argument("--system", default=None, help="with --chat: a system message")
    s.add_argument("--think", action="store_true", help="with --chat: let the model think first (Gemma 4)")


def _add_chat_suite_arg(s):
    from .analyzer.common import DEFAULT_CHAT_INSTRUCTION

    s.add_argument("--chat", nargs="?", const=DEFAULT_CHAT_INSTRUCTION, default=None, metavar="INSTRUCTION",
                   help="chat models: put every test inside a user turn holding INSTRUCTION (default: "
                        f"{DEFAULT_CHAT_INSTRUCTION!r}) and an opened model turn; only the test's own tokens "
                        "are scored. Results go to a separate '<suite>-chat' suite")


def _keep_quantized(name: str):
    return {"auto": "auto", "yes": True, "no": False}[name]


def _add_keep_quantized(s):
    s.add_argument("--keep-quantized", choices=["auto", "yes", "no"], default="auto",
                   help="GGUF: keep quantized weights quantized in GPU/CPU memory and dequantize each layer "
                        "while running (uses ~4x less memory than bf16, but runs slower). auto = only when "
                        "dequantized weights would need more than half the GPU's memory (8 GiB on CPU)")


def _dtype(name: str | None):
    import torch

    return {None: None, "auto": None, "fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]


def _stop_ids(tok) -> set[int]:
    """End-of-text and end-of-turn tokens: generation stops at any of them."""
    stops = {tok.eos_id, getattr(tok, "eot_id", None)} - {None}
    for name in ("<|eos|>", "<|endoftext|>", "<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<end_of_turn>", "<|end|>",
                 "<eos>", "<turn|>", "<|tool_response>"):  # the last three: Gemma 4's end-of-generation tokens
        if name in tok.special_tokens:
            stops.add(tok.special_tokens[name])
    return stops


def cmd_next(a):
    """The model's top candidates for the next token, with probabilities."""
    import torch

    from .forge.train.trainer import resolve_device
    from .models import load_any

    device = resolve_device(a.device)
    model, tok, _ = load_any(a.path, device=device, dtype=_dtype(a.dtype), keep_quantized=_keep_quantized(a.keep_quantized))
    ids = _prompt_ids(tok, a)
    with torch.no_grad():
        logits = model(torch.tensor([ids], device=device)).logits[0, -1, :tok.vocab_size].float()
    probs = logits.softmax(-1)
    top = probs.topk(a.top)
    print(f"prompt: {len(ids)} tokens, last: {tok.decode(ids[-3:])!r}")
    print(f"{'rank':>4} {'id':>7} {'prob':>8} {'logit':>8}  token")
    for r, (pr, i) in enumerate(zip(top.values.tolist(), top.indices.tolist())):
        print(f"{r + 1:4d} {i:7d} {pr:8.2%} {logits[i].item():8.3f}  {tok.decode([i])!r}")
    ent = float(-(probs * probs.clamp_min(1e-30).log()).sum())
    print(f"entropy of the distribution: {ent:.3f} nats")


def cmd_eval(a):
    """Validation loss of a finished run (our models), on evenly spread windows, per source."""
    import contextlib

    import torch

    from .forge.data.loader import ShardDataset
    from .forge.tokenizer.bpe import Tokenizer
    from .forge.train.checkpoint import load_model
    from .forge.train.trainer import evaluate, resolve_device

    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text())
    shards = Path(a.shards or cfg["data"]["shards"])
    device = resolve_device(a.device)
    model, meta = load_model(run, step=a.step, device=device, nearest=True)
    ds = ShardDataset(shards, "val")
    tok = Tokenizer.load(run / "tokenizer.json")
    if tok.fingerprint() != ds.meta["tokenizer_fingerprint"]:
        raise SystemExit(f"{shards} was made with a different tokenizer than {run}")
    seq_len = cfg["train"]["seq_len"]
    byte_lens = torch.tensor(tok.token_byte_lengths() + [0] * (model.cfg.padded_vocab - tok.vocab_size), device=device)
    autocast = ((lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device.type == "cuda"
                else contextlib.nullcontext)
    ev = evaluate(model, ds, a.batch_size, seq_len, a.windows, byte_lens, device, autocast)
    print(f"{run.name} step {meta['step']}: {a.windows} windows of {seq_len} tokens from {shards}")
    print(f"val loss {ev['val_loss']:.4f} nats/token, {ev['val_bpb']:.4f} bits/byte")
    for k, v in (ev.get("val_by_source") or {}).items():
        print(f"  {k:14s} {v:.4f}")
    if "val_by_source" not in ev:
        print("  (no per-source breakdown: these shards predate it; re-run `sleuth shard` to get one)")


def cmd_score(a):
    """Per-category loss / answer loss / exact match of one model on a suite."""
    from .analyzer.common import analysis_prefix, encode_cases, score_cases
    from .analyzer.suites import load_suite
    from .forge.train.trainer import resolve_device
    from .models import load_any

    device = resolve_device(a.device)
    model, tok, meta = load_any(a.path, device=device, dtype=_dtype(a.dtype), step=a.step,
                                keep_quantized=_keep_quantized(a.keep_quantized))
    enc = []
    for suite in a.suite:
        cases = load_suite(suite)
        if a.chat:
            prefix = analysis_prefix(tok, a.chat)
        elif a.few_shot:
            # A few worked examples before each question: a base model then
            # knows the expected *format* (short answer, then stop), so the
            # score reflects what it knows rather than its rambling habits.
            f = Path(suite) / "_fewshot.txt"
            if not f.exists():
                raise SystemExit(f"--few-shot needs {f} (example questions and answers)")
            prefix = [tok.start_id] + tok.encode(f.read_text(encoding="utf-8"))
        else:
            prefix = None
        enc += encode_cases(tok, cases, model.cfg.max_seq_len, prefix)
    res = score_cases(model, enc, min(tok.vocab_size, model.cfg.padded_vocab), device)
    where = f"step {meta['step']}" if meta.get("step") is not None else meta["format"]
    print(f"{a.path} ({where}, {model.num_params() / 1e6:.1f}M params)")
    print(f"{'category':12s} {'tokens':>7s} {'loss':>7s} {'answer loss':>12s} {'exact':>7s}")
    for c, r in res.items():
        el = f"{r['expected_loss']:12.3f}" if r["expected_loss"] is not None else f"{'-':>12s}"
        ex = f"{r['exact_match']:7.1%}" if r["exact_match"] is not None else f"{'-':>7s}"
        print(f"{c:12s} {r['n_tokens']:7d} {r['loss']:7.3f} {el} {ex}")


def cmd_perplexity(a):
    import json as _json

    from .analyzer.perplexity import perplexity
    from .forge.train.trainer import resolve_device
    from .models import load_any

    device = resolve_device(a.device)
    model, tok, _ = load_any(a.path, device=device, dtype=_dtype(a.dtype), keep_quantized=_keep_quantized(a.keep_quantized),
                             max_seq_len=max(4096, a.ctx))
    text = Path(a.file).read_text(encoding="utf-8")
    if a.max_chars:
        text = text[:a.max_chars]
    r = perplexity(model, tok, text, ctx=a.ctx, max_chunks=a.chunks, device=device,
                   log=lambda line: print(line, end=" ", flush=True))
    print()
    print(f"{r['chunks']} chunks of {r['ctx']} tokens ({r['tokens_in_text']} in the text), "
          f"{r['scored_tokens']} scored (second half of each chunk)")
    print(f"perplexity {r['perplexity']:.4f}, loss {r['loss']:.4f} nats/token, {r['bits_per_byte']:.4f} bits/byte")
    print("loss by predicted position (every prediction, not just the scored half):")
    for b in r["by_position"]:
        print(f"  {b['positions']:>9s}  {b['loss']:7.3f} nats  ({b['n']} tokens)")
    if a.json:
        Path(a.json).write_text(_json.dumps(r, indent=1) + "\n")


def cmd_compare(a):
    from .analyzer.compare import compare_models, format_report
    from .analyzer.suites import load_suite
    from .forge.train.trainer import resolve_device
    from .models import load_any

    device = resolve_device(a.device)
    dtype = _dtype(a.dtype)
    cases = [c for s in (a.suite or ["suites/stories", "suites/basic"]) for c in load_suite(s)]

    def loader(path, requant=None):
        def load():
            model, tok, _ = load_any(path, device=device, dtype=dtype, keep_quantized=_keep_quantized(a.keep_quantized))
            if requant:
                from .analyzer.compare import requantize_model

                r = requantize_model(model, requant, a.requant_match)
                print(f"B: {r['n_tensors']} weights rounded through {requant.upper()}, relative error "
                      f"mean {r['rel_error_mean']:.4f}, max {r['rel_error_max']:.4f}", file=sys.stderr)
            return model, tok
        return load

    result = compare_models(loader(a.a), loader(a.b, a.requant), cases, device=device, chat_instruction=a.chat)
    name_b = a.b + (f"  (requantized: {a.requant.upper()}, weights matching {a.requant_match!r})" if a.requant else "")
    print(format_report(result, a.a, name_b))


def cmd_inspect(a):
    from .models import describe_file

    print(describe_file(a.path, max_groups=a.max_groups, show_key=a.show_key))


def cmd_analyze(a):
    from .analyzer.timeline import prepare_external_run, run_timeline
    from .forge.train.trainer import resolve_device
    from .models import detect

    device = str(resolve_device(a.device))
    run = a.run
    if detect(run) != "sleuth" or not Path(run).is_dir():
        # An external model (Hugging Face folder or .gguf): analysis results go
        # into a small run folder of their own, so the dashboard lists them.
        run = prepare_external_run(run, name=a.name, dtype=_dtype(a.dtype), device=device,
                                   keep_quantized=_keep_quantized(a.keep_quantized))
        print(f"results for this external model go to {run}")
    views = [v.strip() for v in a.views.split(",")] if a.views else None
    steps = [int(s) for s in a.steps.split(",")] if a.steps else None
    options = {"neuron_stats": {"k": a.k}, "ablation": {"method": a.ablation}}
    for suite in a.suite or ["suites/basic"]:
        run_timeline(run, suite, device=device, views=views, force=a.force, steps=steps, options=options,
                     chat_instruction=a.chat)


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
    s.add_argument("--tokenizer", required=True,
                   help="our tokenizer .json, or a model (.gguf file, Hugging Face folder, Sleuth run)")
    s.add_argument("text")
    s.add_argument("--allow-special", action="store_true")
    s.add_argument("--chat", action="store_true", help="show the text as a user turn in the model's chat format")
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
    s.add_argument("path", help="a Sleuth run/snapshot, a Hugging Face model folder, or a .gguf file")
    s.add_argument("--dtype", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
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
    _add_keep_quantized(s)
    _add_chat_args(s)
    s.set_defaults(fn=cmd_generate)

    s = sub.add_parser("analyze", help="run analyzer views over every checkpoint of a run")
    s.add_argument("run", help="a Sleuth run, a Hugging Face model folder, or a .gguf file")
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
    s.add_argument("--name", default=None, help="external models: name of the results folder under runs/")
    s.add_argument("--dtype", choices=["auto", "fp32", "bf16", "fp16"], default="auto",
                   help="external models: weight precision (auto = bf16 on GPU, fp32 on CPU)")
    _add_keep_quantized(s)
    _add_chat_suite_arg(s)
    s.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("compare", help="do two models (same tokenizer) predict the same next tokens?")
    s.add_argument("a", help="model A (Sleuth run, Hugging Face folder, or .gguf)")
    s.add_argument("b", help="model B")
    s.add_argument("--suite", action="append", default=None, help="default: suites/stories + suites/basic")
    s.add_argument("--device", default="auto")
    s.add_argument("--dtype", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    s.add_argument("--requant", default=None, metavar="TYPE",
                   help="round model B's weights through this quantizer first (Q8_0, Q4_0 or Q4_1), to measure "
                        "what that quantization alone costs. Use the same path for A and B.")
    s.add_argument("--requant-match", default="*", metavar="PATTERN",
                   help="only requantize weights whose names match, e.g. 'tok_emb.weight' or 'blocks.*.mlp.*'")
    _add_keep_quantized(s)
    _add_chat_suite_arg(s)
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("next", help="the model's top candidates for the next token after a prompt")
    s.add_argument("path", help="a Sleuth run, a Hugging Face model folder, or a .gguf file")
    s.add_argument("--prompt", default="")
    s.add_argument("--top", type=int, default=10)
    s.add_argument("--device", default="auto")
    s.add_argument("--dtype", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    _add_keep_quantized(s)
    _add_chat_args(s)
    s.set_defaults(fn=cmd_next)

    s = sub.add_parser("eval", help="validation loss of a trained run, overall and per data source")
    s.add_argument("run", help="a Sleuth run directory")
    s.add_argument("--shards", default=None, help="shard folder (default: the one the run trained on)")
    s.add_argument("--step", type=int, default=None, help="which snapshot (default: latest)")
    s.add_argument("--windows", type=int, default=640, help="number of val windows (spread over the whole split)")
    s.add_argument("--batch-size", type=int, default=8)
    s.add_argument("--device", default="auto")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("score", help="a model's loss, answer loss and exact match on a test suite, per category")
    s.add_argument("path", help="a Sleuth run, a Hugging Face model folder, or a .gguf file")
    s.add_argument("--suite", action="append", required=True, help="suite directory (repeatable)")
    s.add_argument("--step", type=int, default=None, help="our runs: which snapshot (default: latest)")
    s.add_argument("--device", default="auto")
    s.add_argument("--dtype", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    s.add_argument("--few-shot", action="store_true",
                   help="put the suite's _fewshot.txt (worked examples) before every test")
    _add_keep_quantized(s)
    _add_chat_suite_arg(s)
    s.set_defaults(fn=cmd_score)

    s = sub.add_parser("perplexity", help="perplexity on a text file, chunked like llama.cpp's llama-perplexity")
    s.add_argument("path", help="a Sleuth run, a Hugging Face model folder, or a .gguf file")
    s.add_argument("--file", required=True, help="a UTF-8 text file")
    s.add_argument("--ctx", type=int, default=512, help="tokens per chunk (llama-perplexity's -c)")
    s.add_argument("--chunks", type=int, default=None, help="at most this many chunks (--chunks)")
    s.add_argument("--max-chars", type=int, default=None, help="only read the start of the file")
    s.add_argument("--json", default=None, help="also write the results to this JSON file")
    s.add_argument("--device", default="auto")
    s.add_argument("--dtype", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    _add_keep_quantized(s)
    s.set_defaults(fn=cmd_perplexity)

    s = sub.add_parser("inspect", help="show what's inside a .gguf file or Hugging Face model folder")
    s.add_argument("path")
    s.add_argument("--max-groups", type=int, default=80, help="how many tensor groups to list")
    s.add_argument("--show-key", default=None, metavar="KEY",
                   help="print one GGUF metadata value in full, e.g. tokenizer.chat_template")
    s.set_defaults(fn=cmd_inspect)

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
