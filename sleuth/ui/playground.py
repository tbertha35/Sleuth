"""Playground: talk to a model from the dashboard.

Unlike Jobs (separate processes), the Playground keeps models loaded *inside
the dashboard's process*, because loading is the slow part: Gemma 4 12B takes
a while and 7.7 GB of VRAM, so reloading it for every prompt would be painful.
At most two models stay loaded (so two can be compared side by side), the
least recently used is dropped first, and Unload frees them all.

A model on the GPU holds the jobs' GPU lane (see jobs.py), so a training job
can't start while the Playground is using the GPU, and the Playground can't
load onto a GPU that a job is using ("GPU busy"). Models on the CPU don't
touch the lane.

One request runs at a time (a lock): two generations on one GPU would only
slow each other down, and our KV cache is per call anyway.
"""

from __future__ import annotations

import codecs
import gc
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from .jobs import SAFE_RUN, GpuBusy, JobError

MAX_MODELS = 2


@dataclass
class Loaded:
    key: tuple
    label: str
    model: object
    tok: object
    meta: dict
    device: object
    load_seconds: float
    chat_format: str | None

    def public(self) -> dict:
        return {"label": self.label, "device": str(self.device), "format": self.meta.get("format"),
                "step": self.meta.get("step"), "params": self.model.num_params(),
                "load_seconds": round(self.load_seconds, 1), "chat_format": self.chat_format,
                "kept_quantized": self.meta.get("kept_quantized")}


def _num(body: dict, key: str, default, kind=float, lo=None, hi=None):
    v = body.get(key, default)
    if v is None or v == "":
        return default
    try:
        x = kind(v)
    except (TypeError, ValueError):
        raise JobError(f"{key}: expected a number") from None
    if x != x or (lo is not None and x < lo) or (hi is not None and x > hi):
        raise JobError(f"{key}: must be between {lo} and {hi}")
    return x


class Playground:
    def __init__(self, runs_dir: str | Path, jobs, max_models: int = MAX_MODELS):
        self.runs_dir = Path(runs_dir).resolve()
        self.jobs = jobs
        self.max_models = max_models
        self.models: "OrderedDict[tuple, Loaded]" = OrderedDict()
        self.lock = threading.Lock()   # one load / generation at a time
        self.busy = ""                 # what it's doing right now, for the status line

    # ------------------------------------------------------------ which model
    def _spec(self, spec) -> tuple[str, str, int | None, str, str]:
        """(path to load, label, step, device, dtype) from the page's model choice:
        {"model": run name or file/folder path, "step", "device", "dtype"}."""
        import json

        from ..forge.train.trainer import resolve_device

        if not isinstance(spec, dict):
            raise JobError("model: expected an object")
        name = str(spec.get("model") or "").strip()
        if not name:
            raise JobError("choose a model")
        step = _num(spec, "step", None, int, 0)
        device = spec.get("device") or "auto"
        if device not in ("auto", "cuda", "cpu"):
            raise JobError("device: auto, cuda or cpu")
        dtype = spec.get("dtype") or "auto"
        if dtype not in ("auto", "fp32", "bf16", "fp16"):
            raise JobError("dtype: auto, fp32, bf16 or fp16")
        run_dir = self.runs_dir / name
        if SAFE_RUN.match(name) and ".." not in name and (run_dir / "config.json").exists():
            ext = json.loads((run_dir / "config.json").read_text(encoding="utf-8")).get("external")
            path = ext["path"] if ext else str(run_dir)
            if ext:
                step = None  # external models have one "checkpoint"
        else:
            if not Path(name).exists():
                raise JobError(f"{name!r} is neither one of your runs nor an existing file or folder")
            path = name
        return path, name, step, str(resolve_device(device)), dtype

    def _key(self, spec) -> tuple:
        path, _, step, device, dtype = self._spec(spec)
        return (path, step, device, dtype)

    def _get(self, spec) -> Loaded:
        """The loaded model for this spec, loading it if needed (caller holds the lock)."""
        import torch

        from ..cli import _dtype
        from ..formats.chat import chat_format
        from ..forge.infer.generate import generate
        from ..models import load_any

        path, label, step, device, dtype = self._spec(spec)
        key = self._key(spec)
        if key in self.models:
            self.models.move_to_end(key)
            return self.models[key]
        on_gpu = device.startswith("cuda")
        if on_gpu:
            self.jobs.acquire_gpu("playground")  # raises GpuBusy before we evict anything
        while len(self.models) >= self.max_models:
            self._drop(next(iter(self.models)))
        if on_gpu:
            # Dropping the last GPU model hands the lane back (and a queued job may
            # have taken it), so claim it again.
            self.jobs.acquire_gpu("playground")
        self.busy = f"loading {label}"
        t0 = time.perf_counter()
        try:
            model, tok, meta = load_any(path, device=device, dtype=_dtype(dtype), step=step)
            model.eval()
            # Same warm-up as `sleuth generate`: a throwaway 2-token generation,
            # so the first real request's speed isn't skewed by one-time setup.
            generate(model, [tok.start_id] * 8, 2, seed=0)
        except Exception:
            if not any(str(k[2]).startswith("cuda") for k in self.models):
                self.jobs.release_gpu("playground")
            raise
        finally:
            self.busy = ""
        if meta.get("step") is not None:  # our runs: say which snapshot (it may be the nearest one)
            label += f" @ step {meta['step']}"
        loaded = Loaded(key, label, model, tok, meta, torch.device(device), time.perf_counter() - t0,
                        chat_format(tok))
        self.models[key] = loaded
        return loaded

    def _drop(self, key) -> None:
        import torch

        m = self.models.pop(key)
        del m
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not any(str(k[2]).startswith("cuda") for k in self.models):
            self.jobs.release_gpu("playground")

    def unload(self) -> dict:
        with self.lock:
            for key in list(self.models):
                self._drop(key)
        return self.state()

    def state(self) -> dict:
        return {"models": [m.public() for m in list(self.models.values())], "busy": self.busy,
                "max_models": self.max_models, "gpu_holder": self.jobs.gpu_holder}

    # ------------------------------------------------------------ prompts
    @staticmethod
    def _prompt_ids(m: Loaded, body: dict) -> list[int]:
        """Text mode: start token + prompt. Chat mode: the conversation in the
        model's chat format, ending with an opened assistant turn. Either way,
        `continuation` (text the model already wrote) is appended."""
        from ..formats.chat import encode_chat

        mode = body.get("mode", "text")
        if mode == "chat":
            if m.chat_format is None:
                raise JobError(f"{m.label} has no chat format (it's a base model): use Text mode")
            msgs = body.get("messages")
            if not isinstance(msgs, list) or not msgs:
                raise JobError("chat mode needs messages")
            clean = []
            for x in msgs:
                if not isinstance(x, dict) or x.get("role") not in ("system", "user", "assistant") \
                        or not isinstance(x.get("content"), str):
                    raise JobError("each message needs a role (system/user/assistant) and content")
                clean.append({"role": x["role"], "content": x["content"]})
            ids = encode_chat(m.tok, clean, thinking=bool(body.get("think")))
        elif mode == "text":
            prompt = body.get("prompt", "")
            if not isinstance(prompt, str):
                raise JobError("prompt: expected text")
            ids = [m.tok.start_id] + m.tok.encode(prompt)
        else:
            raise JobError("mode: text or chat")
        cont = body.get("continuation") or ""
        if not isinstance(cont, str):
            raise JobError("continuation: expected text")
        return ids + (m.tok.encode(cont) if cont else [])

    @staticmethod
    def _sampling(body: dict):
        from ..forge.infer.sampling import SamplingConfig

        return SamplingConfig(
            temperature=_num(body, "temperature", 0.8, float, 0, 10),
            top_k=_num(body, "top_k", 0, int, 0),
            top_p=_num(body, "top_p", 1.0, float, 0, 1),
            min_p=_num(body, "min_p", 0.05, float, 0, 1),
            repetition_penalty=_num(body, "repetition_penalty", 1.0, float, 0.01, 10),
            repeat_last_n=_num(body, "repeat_last_n", 64, int, 0),
            no_repeat_ngram=_num(body, "no_repeat_ngram", 0, int, 0, 64),
        )

    # ------------------------------------------------------------ the two actions
    def generate(self, body: dict, emit) -> None:
        """Stream a generation: emit("status"|"token"|"done", dict). emit raises
        when the page has gone away (Stop button), which ends the generation."""
        import torch

        from ..cli import _stop_ids
        from ..forge.infer.generate import generate

        if not self.lock.acquire(timeout=0.1):
            emit("status", {"text": "waiting for the previous request to finish…"})
            self.lock.acquire()
        try:
            if self._key(body.get("model")) not in self.models:
                emit("status", {"text": "loading the model (big models take a while)…"})
            m = self._get(body.get("model"))
            sampling = self._sampling(body)
            max_new = _num(body, "max_new_tokens", 200, int, 1, 100_000)
            seed = _num(body, "seed", None, int)
            ids = self._prompt_ids(m, body)
            emit("status", {"text": f"{m.label}: {len(ids)} prompt tokens"})
            stops = _stop_ids(m.tok)
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

            def on_token(t):
                if t not in stops:
                    piece = decoder.decode(m.tok.decode_bytes([t]))
                    if piece:
                        emit("token", {"text": piece})

            if m.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(m.device)
            self.busy = f"generating with {m.label}"
            res = generate(m.model, ids, max_new, sampling, stop_ids=stops, vocab_limit=m.tok.vocab_size,
                           seed=seed, on_token=on_token)
            tail = decoder.decode(b"", final=True)
            if tail:
                emit("token", {"text": tail})
            stats = {"model": m.label, "tokens": len(res.new_ids), "stop_reason": res.stop_reason,
                     "prompt_tokens": len(ids), "prompt_ms": round(res.prefill_seconds * 1000),
                     "tok_per_s": round(res.decode_tok_per_s, 1)}
            if m.device.type == "cuda":
                stats["peak_gb"] = round(torch.cuda.max_memory_allocated(m.device) / 2**30, 2)
            emit("done", stats)
        finally:
            self.busy = ""
            self.lock.release()

    def next_tokens(self, body: dict) -> dict:
        """The top candidates for the next token, like `sleuth next`."""
        import torch

        with self.lock:
            m = self._get(body.get("model"))
            ids = self._prompt_ids(m, body)
            top = _num(body, "top", 10, int, 1, 200)
            with torch.no_grad():
                ids = ids[-m.model.cfg.max_seq_len:]
                logits = m.model(torch.tensor([ids], device=m.device)).logits[0, -1, :m.tok.vocab_size].float()
            probs = logits.softmax(-1)
            best = probs.topk(min(top, probs.numel()))
            ent = float(-(probs * probs.clamp_min(1e-30).log()).sum())
            return {"model": m.label, "prompt_tokens": len(ids), "entropy": ent,
                    "last_tokens": [m.tok.decode([i]) for i in ids[-8:]],
                    "candidates": [{"id": i, "text": m.tok.decode([i]), "prob": p, "logit": float(logits[i])}
                                   for p, i in zip(best.values.tolist(), best.indices.tolist())]}
