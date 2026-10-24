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
import time
from pathlib import Path
from typing import Callable

import torch

from ...config import to_dict
from ...env import env_info, runs_root
from ..data.loader import ShardDataset
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
def evaluate(model, ds: ShardDataset, batch_size: int, seq_len: int, max_batches: int,
             byte_lens: torch.Tensor, device: torch.device, autocast) -> dict:
    """Validation loss (nats/token) and bits-per-byte.

    Bits-per-byte = total loss in bits / number of raw-text bytes predicted.
    It doesn't depend on the tokenizer, so models with different
    tokenizers (or vocab sizes) can be compared fairly. Lower is better.
    """
    was_training = model.training
    model.eval()
    loss_sum, n_tok, n_bytes = 0.0, 0, 0
    for x, y in ds.sequential_batches(batch_size, seq_len, max_batches):
        x, y = x.to(device), y.to(device)
        with autocast():
            logits = model(x).logits
        loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        loss_sum += loss.item()
        n_tok += y.numel()
        n_bytes += int(byte_lens[y].sum().item())
    if was_training:
        model.train()
    return {
        "val_loss": loss_sum / max(1, n_tok),
        "val_bpb": loss_sum / math.log(2) / max(1, n_bytes),
        "val_tokens": n_tok,
    }


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
    tokenizer = Tokenizer.load(train_ds.meta["tokenizer"])
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
        shutil.copy(train_ds.meta["tokenizer"], run_dir / "tokenizer.json")
        save_snapshot(run_dir, 0, model, 0, tc.snapshot_dtype)  # untrained baseline for the timeline

    fwd = torch.compile(model) if tc.compile else model
    use_bf16 = tc.precision == "bf16" and device.type == "cuda"

    def autocast():
        return torch.autocast(device.type, dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()

    tokens_per_step = tc.micro_batch_size * tc.seq_len * tc.grad_accum
    flops_per_token = train_flops_per_token(cfg.model, tc.seq_len)
    snaps = snapshot_steps(tc.max_steps, tc.snapshot_every, tc.snapshot_log_spaced)
    metrics = MetricsLog(run_dir / "metrics.jsonl")
    log(f"run dir: {run_dir}")
    log(f"model: {describe(cfg.model, tc.seq_len)}")
    log(f"device: {device}, precision: {'bf16 autocast' if use_bf16 else 'fp32'}, "
        f"{tokens_per_step:,} tokens/step, {tc.max_steps * tokens_per_step / 1e6:.1f}M tokens total, "
        f"train tokens available: {train_ds.n_tokens / 1e6:.1f}M")
    epochs = tc.max_steps * tokens_per_step / max(1, train_ds.n_tokens)
    if epochs > 4:
        # Repeating data a few times is fine; beyond ~4 passes the model starts
        # memorizing it and validation loss gets worse (overfitting).
        log(f"warning: this run reads the training data ~{epochs:.0f} times. Expect overfitting: "
            f"watch val loss, and consider more data or fewer steps.")

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
                    loss = fwd(x, y).loss
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
                    f"gnorm {rec['grad_norm']:.2f} | {tps:,.0f} tok/s" + (f" | mfu {rec['mfu']:.1%}" if "mfu" in rec else ""))
                loss_sum.zero_()
                loss_count = 0
                t_last, tok_last = time.perf_counter(), tokens_seen

            if val_ds is not None and (completed % tc.eval_every == 0 or last):
                t_eval = time.perf_counter()
                ev = evaluate(fwd, val_ds, tc.micro_batch_size, tc.seq_len, tc.eval_batches, byte_lens, device, autocast)
                metrics.write({"step": completed, "tokens": tokens_seen, **ev, "time": time.time()})
                log(f"step {completed:6d} | val loss {ev['val_loss']:.4f} | val bpb {ev['val_bpb']:.4f}")
                # Don't count eval time as training time. Shift the timer by the
                # eval's duration; resetting it would pair a full logging
                # interval's tokens with a partial interval's time whenever
                # eval_every isn't a multiple of log_every (which showed up as
                # a 2x tok/s spike after every eval).
                t_last += time.perf_counter() - t_eval

            if completed in snaps:
                save_snapshot(run_dir, completed, model, tokens_seen, tc.snapshot_dtype)
            stopping = stop_at_step is not None and completed >= stop_at_step and not last
            if completed % tc.resume_every == 0 or last or stopping:
                save_resume(run_dir, completed, model, optimizer, {"tokens_seen": tokens_seen}, tc.keep_resume)
            if stopping:
                log(f"stopped at step {completed} as requested; resume with: python -m sleuth train --resume {run_dir}")
                return run_dir
    except KeyboardInterrupt:
        if completed > start_step:
            log(f"\ninterrupted: saving resumable checkpoint at step {completed}...")
            save_resume(run_dir, completed, model, optimizer, {"tokens_seen": tokens_seen}, tc.keep_resume)
            log(f"resume with: python -m sleuth train --resume {run_dir}")
        raise
    finally:
        metrics.close()
    log(f"done. snapshots in {run_dir / 'snapshots'}")
    return run_dir
