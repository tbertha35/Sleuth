"""The training loop (single device; multi-GPU / 2-node DDP arrives in M2).

One optimizer step:
    for each of grad_accum micro-batches:
        forward under bf16 autocast -> loss / grad_accum -> backward
            (gradients add up across micro-batches, so this equals one big
             batch of micro_batch_size·grad_accum sequences)
    clip the global gradient norm  (guards against rare loss spikes)
    AdamW step at the scheduled learning rate

Mixed precision: autocast runs matmuls in bf16 (much faster on tensor
cores) while the weights, gradients and optimizer state stay fp32.
bf16 has the same exponent range as fp32, so unlike fp16 it needs no loss
scaling.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import shutil
import signal
import time
from pathlib import Path
from typing import Callable

import torch

from ...config import to_dict
from ...env import env_info, runs_root
from ..data.loader import ShardDataset, document_ids
from ..model.attention import doc_block_mask
from ..model.flops import describe, train_flops_per_token
from ..model.transformer import Transformer
from ..tokenizer.bpe import Tokenizer
from .checkpoint import latest_resume, load_resume, save_resume, save_snapshot, snapshot_steps
from .config import RunConfig
from .optim import build_optimizer
from .schedule import lr_at


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


class MetricsLog:
    """Append-only JSONL: robust to crashes, trivially readable by the UI."""

    def __init__(self, path: Path):
        self.fh = open(path, "a", buffering=1)  # line-buffered: each record hits disk promptly

    def write(self, record: dict) -> None:
        self.fh.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self.fh.close()


@torch.no_grad()
def evaluate(model, ds: ShardDataset, batch_size: int, seq_len: int, n_windows: int,
             byte_lens: torch.Tensor, device: torch.device, autocast, bos_id: int | None = None) -> dict:
    """Validation loss (nats/token) and bits-per-byte, overall and per source.

    Bits-per-byte = total loss in bits / number of raw-text bytes predicted.
    It doesn't depend on the tokenizer, so models with different
    tokenizers (or vocab sizes) can be compared fairly. Lower is better.

    The windows are spread evenly over the whole val split (ShardDataset.eval_windows),
    so a mix is measured in its real proportions. Shards that record where each
    source starts also give a loss per source ("val_by_source").

    bos_id: if given, attention is document-masked (as in training with doc_mask).
    """
    was_training = model.training
    model.eval()
    loss_sum, n_tok, n_bytes = 0.0, 0, 0
    by_src: dict[str, list[float]] = {}
    windows = ds.eval_windows(n_windows, seq_len)
    for x, y, sources in ds.window_batches(windows, batch_size, seq_len):
        x, y = x.to(device), y.to(device)
        doc = document_ids(x, bos_id) if bos_id is not None else None
        with autocast():
            logits = model(x, doc_ids=doc).logits
        loss = torch.nn.functional.cross_entropy(logits.float().transpose(1, 2), y, reduction="none")  # [B, T]
        per_window = loss.sum(1).tolist()
        loss_sum += sum(per_window)
        n_tok += y.numel()
        n_bytes += int(byte_lens[y].sum().item())
        for src, w in zip(sources, per_window):
            if src is not None:
                acc = by_src.setdefault(src, [0.0, 0])
                acc[0] += w
                acc[1] += y.shape[1]
    if was_training:
        model.train()
    out = {
        "val_loss": loss_sum / max(1, n_tok),
        "val_bpb": loss_sum / math.log(2) / max(1, n_bytes),
        "val_tokens": n_tok,
    }
    if by_src:
        out["val_by_source"] = {k: v[0] / v[1] for k, v in sorted(by_src.items())}
    return out


class _StopRequested(Exception):
    """Raised by the SIGTERM / Ctrl-Break handlers: stop like Ctrl-C does (save, then exit)."""


def prepare_run_dir(cfg: RunConfig) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = runs_root(cfg.runs_dir) / f"{stamp}-{cfg.name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def train(cfg: RunConfig, run_dir: Path | None = None, resume: bool = False,
          stop_at_step: int | None = None, log: Callable[[str], None] = print) -> Path:
    """Train (or resume) a run and return its directory.

    stop_at_step: stop after this many steps of the max_steps schedule and
    save a resumable checkpoint, e.g. to train in time slices on a shared
    machine. The learning-rate schedule still follows max_steps.
    """
    device = resolve_device(cfg.train.device)
    previous = torch.get_float32_matmul_precision()
    if device.type == "cuda":
        # Let plain fp32 matmuls use TF32 tensor cores (outside autocast).
        torch.set_float32_matmul_precision("high")
    try:
        return _train(cfg, device, run_dir, resume, stop_at_step, log)
    finally:
        # The setting is process-wide. Leaving it on silently made later fp32
        # work in the same process (an exactness test, an analysis) ~1e-3 less
        # precise: TF32 keeps only 10 mantissa bits.
        torch.set_float32_matmul_precision(previous)


def _train(cfg: RunConfig, device: torch.device, run_dir: Path | None, resume: bool,
           stop_at_step: int | None, log: Callable[[str], None]) -> Path:
    tc = cfg.train
    # Config checks that would otherwise fail later, after the run folder exists.
    if tc.doc_mask_kernel not in ("sdpa", "flex"):
        raise ValueError(f"train.doc_mask_kernel must be 'sdpa' or 'flex', not {tc.doc_mask_kernel!r}")
    if tc.doc_mask and tc.doc_mask_kernel == "flex" and device.type != "cuda":
        raise ValueError("train.doc_mask_kernel = 'flex' needs a GPU (FlexAttention has no backward pass on CPU); "
                         "use 'sdpa'")
    torch.manual_seed(tc.seed)

    # ---- data + tokenizer -------------------------------------------------
    train_ds = ShardDataset(cfg.data.shards, "train")
    try:
        val_ds = ShardDataset(cfg.data.shards, "val")
    except ValueError:
        val_ds = None
        log("note: no validation split; val metrics disabled")
    data_vocab = train_ds.vocab_size
    if cfg.model.vocab_size == 0:
        cfg.model = dataclasses.replace(cfg.model, vocab_size=data_vocab)
    elif cfg.model.vocab_size != data_vocab:
        raise ValueError(f"model.vocab_size={cfg.model.vocab_size} but the shards use vocab {data_vocab} "
                         f"(set model.vocab_size = 0 to take it from the data)")
    tokenizer = Tokenizer.load(train_ds.tokenizer_path)
    if tokenizer.fingerprint() != train_ds.meta["tokenizer_fingerprint"]:
        raise ValueError("tokenizer file changed since the shards were made; re-shard or restore it")
    byte_lens = torch.tensor(tokenizer.token_byte_lengths() + [0] * (cfg.model.padded_vocab - tokenizer.vocab_size),
                             device=device)

    # ---- model + optimizer ------------------------------------------------
    model = Transformer(cfg.model).to(device)
    optimizer = build_optimizer(model, tc, device.type)
    start_step, tokens_seen = 0, 0

    if resume:
        assert run_dir is not None
        ck = latest_resume(run_dir)
        if ck is None:
            raise FileNotFoundError(f"no resumable checkpoint in {run_dir}")
        state = load_resume(ck, model, optimizer)
        start_step, tokens_seen = state["step"], state["tokens_seen"]
        log(f"resumed from {ck} at step {start_step}")
    else:
        run_dir = run_dir or prepare_run_dir(cfg)
        (run_dir / "config.json").write_text(json.dumps(to_dict(cfg), indent=2) + "\n")
        (run_dir / "env.json").write_text(json.dumps(env_info(), indent=2) + "\n")
        shutil.copy(train_ds.tokenizer_path, run_dir / "tokenizer.json")
        save_snapshot(run_dir, 0, model, 0, tc.snapshot_dtype)  # untrained baseline for the timeline

    compile_model = tc.compile
    if compile_model and device.type == "cuda":
        try:
            import triton  # noqa: F401  (torch.compile's GPU code generator)
        except ImportError:
            log("warning: compile = true needs Triton (Windows: pip install -e .[triton]); training without it")
            compile_model = False
    fwd = torch.compile(model) if compile_model else model
    use_bf16 = tc.precision == "bf16" and device.type == "cuda"

    def autocast():
        return torch.autocast(device.type, dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()

    # Document masking: the model gets each window's document numbers, so
    # attention stops at every <|bos|> (see data/loader.py: document_ids).
    mask_bos = tokenizer.bos_id if tc.doc_mask else None
    use_flex = tc.doc_mask and tc.doc_mask_kernel == "flex"
    if use_flex and compile_model and cfg.model.head_dim < 16:
        # Compiled FlexAttention needs heads at least 16 wide (Inductor: "NYI: embedding
        # dimension ... must be at least 16"). Tiny test models have 8.
        log(f"warning: doc_mask_kernel = 'flex' needs head size >= 16 under torch.compile "
            f"(this model has {cfg.model.head_dim}); using 'sdpa' instead")
        use_flex = False
    if use_flex and not compile_model:
        log("warning: doc_mask_kernel = 'flex' without torch.compile runs FlexAttention's slow reference "
            "version; set compile = true (or use 'sdpa')")

    def masks(x) -> dict:
        """The document-mask arguments for one batch (none without doc_mask)."""
        if mask_bos is None:
            return {}
        doc = document_ids(x, mask_bos)
        # The BlockMask is built here, outside the compiled model, once per micro-batch.
        return {"block_mask": doc_block_mask(doc)} if use_flex else {"doc_ids": doc}

    tokens_per_step = tc.micro_batch_size * tc.seq_len * tc.grad_accum
    flops_per_token = train_flops_per_token(cfg.model, tc.seq_len)
    snaps = snapshot_steps(tc.max_steps, tc.snapshot_every, tc.snapshot_log_spaced)
    metrics = MetricsLog(run_dir / "metrics.jsonl")
    log(f"run dir: {run_dir}")
    log(f"model: {describe(cfg.model, tc.seq_len)}")
    log(f"device: {device}, precision: {'bf16 autocast' if use_bf16 else 'fp32'}, "
        + (f"document-masked attention ({tc.doc_mask_kernel}), " if tc.doc_mask else "") +
        f"{tokens_per_step:,} tokens/step, {tc.max_steps * tokens_per_step / 1e6:.1f}M tokens total, "
        f"train tokens available: {train_ds.n_tokens / 1e6:.1f}M")
    epochs = tc.max_steps * tokens_per_step / max(1, train_ds.n_tokens)
    if epochs > 4:
        # Repeating data a few times is fine; beyond ~4 passes the model starts
        # memorizing it and validation loss gets worse (overfitting).
        log(f"warning: this run reads the training data ~{epochs:.0f} times. Expect overfitting: "
            f"watch val loss, and consider more data or fewer steps.")

    n_eval = tc.eval_windows or tc.eval_batches * tc.micro_batch_size
    stop_file = run_dir / "STOP"
    stop_file.unlink(missing_ok=True)
    log(f"to stop cleanly from anywhere: create the file {stop_file} (or press Ctrl-C here)")
    restore_signals = _install_stop_signals()

    model.train()
    completed = start_step
    t_last, tok_last = time.perf_counter(), tokens_seen
    loss_sum = torch.zeros((), device=device)
    loss_count = 0
    try:
        for step in range(start_step, tc.max_steps):
            lr = lr_at(step, tc)
            for group in optimizer.param_groups:
                group["lr"] = lr
            for micro in range(tc.grad_accum):
                x, y = train_ds.batch(tc.micro_batch_size, tc.seq_len, tc.seed, step, micro)
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with autocast():
                    loss = fwd(x, y, **masks(x)).loss
                (loss / tc.grad_accum).backward()
                # Accumulate on-device: calling .item() here would force the
                # CPU to wait for the GPU every micro-step.
                loss_sum += loss.detach()
                loss_count += 1
            if tc.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed = step + 1
            tokens_seen += tokens_per_step

            last = completed == tc.max_steps
            if completed % tc.log_every == 0 or last:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                now = time.perf_counter()
                tps = (tokens_seen - tok_last) / max(now - t_last, 1e-9)
                rec = {
                    "step": completed, "tokens": tokens_seen, "loss": (loss_sum / loss_count).item(),
                    "lr": lr, "grad_norm": float(grad_norm), "tok_per_s": round(tps, 1), "time": time.time(),
                }
                if tc.peak_tflops > 0:
                    rec["mfu"] = round(tps * flops_per_token / (tc.peak_tflops * 1e12), 4)
                if device.type == "cuda":
                    rec["mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                metrics.write(rec)
                log(f"step {completed:6d} | loss {rec['loss']:.4f} | lr {lr:.2e} | "
                    f"gnorm {rec['grad_norm']:.2f} | {tps:,.0f} tok/s" + (f" | mfu {rec['mfu']:.1%}" if "mfu" in rec else "")
                    + (f" | mem {rec['mem_gb']:.1f} GB" if "mem_gb" in rec else ""))
                loss_sum.zero_()
                loss_count = 0
                t_last, tok_last = time.perf_counter(), tokens_seen

            if val_ds is not None and (completed % tc.eval_every == 0 or last):
                t_eval = time.perf_counter()
                ev = evaluate(fwd, val_ds, tc.micro_batch_size, tc.seq_len, n_eval, byte_lens, device, autocast,
                              mask_bos)
                metrics.write({"step": completed, "tokens": tokens_seen, **ev, "time": time.time()})
                srcs = ev.get("val_by_source")
                log(f"step {completed:6d} | val loss {ev['val_loss']:.4f} | val bpb {ev['val_bpb']:.4f}"
                    + (" | " + ", ".join(f"{k} {v:.3f}" for k, v in srcs.items()) if srcs else ""))
                # Don't count eval time as training time. Shift the timer by the
                # eval's duration; resetting it would pair a full logging
                # interval's tokens with a partial interval's time whenever
                # eval_every isn't a multiple of log_every (which showed up as
                # a 2x tok/s spike after every eval).
                t_last += time.perf_counter() - t_eval

            if completed in snaps:
                save_snapshot(run_dir, completed, model, tokens_seen, tc.snapshot_dtype)
            stopping = stop_at_step is not None and completed >= stop_at_step and not last
            if stop_file.exists() and not last:
                stop_file.unlink(missing_ok=True)
                stopping = True
            if completed % tc.resume_every == 0 or last or stopping:
                save_resume(run_dir, completed, model, optimizer, {"tokens_seen": tokens_seen}, tc.keep_resume)
            if stopping:
                log(f"stopped at step {completed} as requested; resume with: python -m sleuth train --resume {run_dir}")
                return run_dir
    except (KeyboardInterrupt, _StopRequested) as e:
        if completed > start_step:
            log(f"\ninterrupted: saving resumable checkpoint at step {completed}...")
            save_resume(run_dir, completed, model, optimizer, {"tokens_seen": tokens_seen}, tc.keep_resume)
            log(f"resume with: python -m sleuth train --resume {run_dir}")
        if isinstance(e, KeyboardInterrupt):
            raise
        return run_dir
    finally:
        metrics.close()
        restore_signals()
    log(f"done. snapshots in {run_dir / 'snapshots'}")
    return run_dir


def _install_stop_signals():
    """Make SIGTERM (a `kill`, or a job scheduler) and, on Windows, Ctrl-Break
    stop training like Ctrl-C: save a resumable checkpoint first. A process run
    in the background often never sees Ctrl-C (the owner lost ~460 steps that
    way); the STOP file is the most reliable way to stop one. Returns a function
    that puts the old handlers back.

    Only the first signal raises. The dashboard re-sends Ctrl-Break while a
    job hasn't exited (Windows can drop one that arrives during start-up), and
    a second one must not interrupt the checkpoint the first one is saving."""
    requested = []

    def handler(signum, frame):
        if requested:
            return
        requested.append(signum)
        raise _StopRequested(f"signal {signum}")

    old = {}
    for name in ("SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            old[sig] = signal.signal(sig, handler)
        except ValueError:  # not the main thread (e.g. a test runner): leave signals alone
            pass

    def restore():
        for sig, h in old.items():
            signal.signal(sig, h)
    return restore
