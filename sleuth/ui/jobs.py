"""Jobs: start Sleuth commands from the dashboard, one GPU job at a time.

The dashboard can now *start processes*, which a read-only viewer never
could, so this module is deliberately narrow:

  * A fixed allow-list of commands (COMMANDS below). The page sends a command
    name and form values; we validate every value against that command's
    fields and build the argument list ourselves. Nothing is ever passed
    through a shell, and there's no "run this string" endpoint, so the page
    can't run anything else.
  * Every job is `python -m sleuth <command> ...` in a child process, with its
    output going to runs/_jobs/<id>.log (the page streams that file).
  * Two lanes: "gpu" (training, eval, scoring, generation...) and "cpu"
    (tokenizer, shards, and anything asked to run with device=cpu). Each lane
    runs one job at a time and queues the rest, so two trainings can never
    share the GPU (one slows the other and both can run out of memory). The
    Playground takes the GPU lane too while it has a model loaded on the GPU.
  * Stopping a training job uses the trainer's STOP file (it finishes the
    current step and saves a resumable checkpoint). Other jobs, and a training
    job still starting up, get SIGTERM (Ctrl-Break on Windows). A second Stop
    kills the process.

Jobs outlive the dashboard: closing `sleuth ui` doesn't stop a training run
(it runs in its own process group / session). After a restart those jobs
show as "orphaned", since we can no longer watch them; their log file keeps
growing and is still shown.
"""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

SAFE_RUN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
SET_ITEM = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+=[^\r\n\x00]*$")
VIEWS = re.compile(r"^[a-z_]+(,[a-z_]+)*$")
STEPS = re.compile(r"^\d+(,\d+)*$")
# Lines a job prints that name its run folder (trainer / analyze of an external model).
RUN_LINE = re.compile(rb"^(?:run dir|results for this external model go to): (.+?)\r?$", re.M)
# The trainer prints this once its loop is about to start: from then on the
# STOP file works (earlier, the trainer deletes any stale STOP file).
STOP_READY = b"to stop cleanly from anywhere"


class JobError(ValueError):
    """A request the page made that we refuse (bad value, unknown job...)."""


class GpuBusy(JobError):
    pass


@dataclass
class Field:
    name: str
    kind: str              # text | int | float | bool | choice | run | model | path | config | suite | set
    flag: str | None       # "--max-new-tokens"; None = positional argument
    label: str
    help: str = ""
    required: bool = False
    default: object = None
    choices: list | None = None
    multi: bool = False    # several values: --flag v1 --flag v2 (or for nargs="+": --flag v1 v2)
    nargs: bool = False    # with multi: one flag followed by all values
    must_exist: bool = True
    min: float | None = None
    max: float | None = None
    multiline: bool = False


@dataclass
class Command:
    name: str
    title: str
    sub: list[str]
    lane: str
    help: str
    fields: list[Field]


def _device():
    return Field("device", "choice", "--device", "Device", "auto = the GPU if there is one",
                 default="auto", choices=["auto", "cuda", "cpu"])


def _model(help="one of your runs, or the path of a .gguf file / Hugging Face folder"):
    return Field("path", "model", None, "Model", help, required=True)


def _step():
    return Field("step", "int", "--step", "Snapshot step", "empty = the latest; the nearest snapshot is used", min=0)


COMMANDS = {c.name: c for c in [
    Command("train", "Train a model", ["train"], "gpu", "Start a new training run from a config file.", [
        Field("config", "config", None, "Config", "a TOML file in configs/", required=True),
        Field("set", "set", "--set", "Overrides", "KEY=VALUE, one per line, e.g. train.max_steps=2000", multi=True),
        Field("stop_at_step", "int", "--stop-at-step", "Pause after step", "stop (resumably) at this step", min=1),
    ]),
    Command("resume", "Resume training", ["train"], "gpu", "Continue a stopped run from its last checkpoint.", [
        Field("run", "run", "--resume", "Run", required=True),
        Field("stop_at_step", "int", "--stop-at-step", "Pause after step", min=1),
    ]),
    Command("eval", "Validation loss", ["eval"], "gpu", "Validation loss of a run, overall and per data source.", [
        Field("run", "run", None, "Run", required=True), _step(),
        Field("windows", "int", "--windows", "Windows", "validation windows, spread over the whole split",
              default=640, min=1),
        _device(),
    ]),
    Command("score", "Score on a suite", ["score"], "gpu",
            "Answer loss, exact match and loss per category on test suites.", [
        _model(), Field("suite", "suite", "--suite", "Suites", required=True, multi=True, default=["suites/knowledge"]),
        _step(),
        Field("few_shot", "bool", "--few-shot", "Few-shot", "put the suite's worked examples before every test"),
        Field("chat", "bool", "--chat", "Chat-wrap", "chat models: put every test inside a user turn"),
        Field("show_items", "bool", "--show-items", "List every test", default=True),
        _device(),
    ]),
    Command("analyze", "Analyze checkpoints", ["analyze"], "gpu",
            "Run the analyzer views over every checkpoint (results appear in the Runs page).", [
        _model(), Field("suite", "suite", "--suite", "Suites", required=True, multi=True, default=["suites/basic"]),
        Field("views", "text", "--views", "Views", "comma-separated: neuron_stats,attention,logit_lens,ablation "
              "(empty = all)"),
        Field("steps", "text", "--steps", "Steps", "only these checkpoints, e.g. 0,500,5000 (empty = all)"),
        Field("chat", "bool", "--chat", "Chat-wrap", "chat models: put every test inside a user turn"),
        Field("force", "bool", "--force", "Redo finished checkpoints"),
        _device(),
    ]),
    Command("generate", "Generate text", ["generate"], "gpu",
            "Sample text (the Playground does this interactively).", [
        _model(), Field("prompt", "text", "--prompt", "Prompt", multiline=True), _step(),
        Field("max_new_tokens", "int", "--max-new-tokens", "Max new tokens", default=200, min=1, max=100_000),
        Field("temperature", "float", "--temperature", "Temperature", "0 = greedy", default=0.8, min=0, max=10),
        Field("top_k", "int", "--top-k", "Top-k", "0 = off", default=0, min=0),
        Field("top_p", "float", "--top-p", "Top-p", "1 = off", default=1.0, min=0, max=1),
        Field("repetition_penalty", "float", "--repetition-penalty", "Repetition penalty", "1 = off",
              default=1.0, min=0.01, max=10),
        Field("no_repeat_ngram", "int", "--no-repeat-ngram", "No-repeat n-gram", "0 = off", default=0, min=0, max=64),
        Field("samples", "int", "--samples", "Samples", default=1, min=1, max=100),
        Field("seed", "int", "--seed", "Seed"),
        Field("chat", "bool", "--chat", "Chat format", "wrap the prompt as a user turn (chat models)"),
        _device(),
    ]),
    Command("next", "Next-token candidates", ["next"], "gpu", "The model's top candidates for the next token.", [
        _model(), Field("prompt", "text", "--prompt", "Prompt", multiline=True),
        Field("top", "int", "--top", "How many", default=10, min=1, max=1000),
        Field("chat", "bool", "--chat", "Chat format"), _device(),
    ]),
    Command("tokenizer", "Train a tokenizer", ["tokenizer"], "cpu", "Train a byte-level BPE tokenizer.", [
        Field("input", "path", "--input", "Input folders/files", required=True, multi=True, nargs=True),
        Field("vocab_size", "int", "--vocab-size", "Vocab size", default=8192, min=256, max=1_000_000),
        Field("out", "path", "--out", "Output file", "empty = data/tokenizers/<input>-<vocab>.json", must_exist=False),
        Field("max_mb", "float", "--max-mb", "Read at most (MB)", min=0.001),
    ]),
    Command("shard", "Build shards", ["shard"], "cpu", "Tokenize documents into training shards.", [
        Field("input", "path", "--input", "Input folders/files", "per-source caps: path@MB", required=True,
              multi=True, nargs=True, must_exist=False),
        Field("tokenizer", "path", "--tokenizer", "Tokenizer", required=True),
        Field("out", "path", "--out", "Output folder", required=True, must_exist=False),
        Field("val_fraction", "float", "--val-fraction", "Validation fraction", default=0.005, min=0, max=0.5),
    ]),
]}


def _check_text(v, f: Field) -> str:
    if not isinstance(v, str):
        raise JobError(f"{f.label}: expected text")
    if "\x00" in v or len(v) > 100_000:
        raise JobError(f"{f.label}: not allowed")
    if not f.multiline and ("\n" in v or "\r" in v):
        raise JobError(f"{f.label}: must be one line")
    return v


def _not_an_option(v: str, f: Field) -> str:
    # Positional values and nargs lists can't use the --flag=value form, so a
    # value starting with "-" would be read as an option. Refuse it.
    if v.startswith("-"):
        raise JobError(f"{f.label}: can't start with '-'")
    return v


@dataclass
class Job:
    id: str
    command: str
    title: str
    argv: list[str]           # after `python -m sleuth`
    lane: str
    status: str = "queued"    # queued | running | stopping | done | failed | stopped | cancelled | orphaned
    created: float = field(default_factory=time.time)
    started: float | None = None
    ended: float | None = None
    exit_code: int | None = None
    run: str | None = None    # the run folder it works on / created (for a link)
    pid: int | None = None
    note: str = ""

    def public(self) -> dict:
        d = asdict(self)
        d["display"] = "sleuth " + " ".join(_quote(a) for a in self.argv)
        d["finished"] = self.status in FINISHED
        return d


FINISHED = {"done", "failed", "stopped", "cancelled", "orphaned"}


def _quote(a: str) -> str:
    a = a.replace("\n", "\\n")
    return a if re.match(r"^[A-Za-z0-9_./:=@,+\\-]+$", a) else repr(a)


class JobManager:
    def __init__(self, runs_dir: str | Path, cwd: str | Path | None = None, prefix: list[str] | None = None,
                 env: dict | None = None, data_dir: str | Path | None = None):
        """prefix: how to start Sleuth (tests pass a fake script instead)."""
        self.runs_dir = Path(runs_dir).resolve()
        self.dir = self.runs_dir / "_jobs"
        self.cwd = Path(cwd or os.getcwd()).resolve()
        self.prefix = prefix or [sys.executable, "-u", "-m", "sleuth"]
        self.env = env
        self.data_dir = Path(data_dir) if data_dir else None
        self.lock = threading.RLock()
        self.jobs: dict[str, Job] = {}
        self.procs: dict[str, subprocess.Popen] = {}
        self.gpu_holder: str | None = None   # "playground" while it has a model on the GPU
        self.stop_ready: set[str] = set()    # training jobs whose STOP file is now watched
        self._load_history()

    # ------------------------------------------------------------ history
    def _load_history(self) -> None:
        if not self.dir.is_dir():
            return
        for f in sorted(self.dir.glob("*.json")):
            try:
                job = Job(**json.loads(f.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                continue
            if job.status in ("running", "stopping"):
                job.status, job.note = "orphaned", ("the dashboard restarted while this job ran, so it can't be "
                                                    "watched any more; it may still be running (see its log)")
            elif job.status == "queued":
                job.status, job.note = "cancelled", "the dashboard restarted before this job started"
            self.jobs[job.id] = job
            self._save(job)

    def _save(self, job: Job) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f"{job.id}.json.tmp"
        tmp.write_text(json.dumps(asdict(job), indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, self.dir / f"{job.id}.json")

    def log_path(self, job_id: str) -> Path:
        return self.dir / f"{job_id}.log"

    # ------------------------------------------------------------ the form schema
    def specs(self) -> list[dict]:
        """Every command with its fields, plus the choices that come from disk
        (configs, suites, runs), for the page to build its forms from."""
        configs = sorted(str(p.relative_to(self.cwd)).replace("\\", "/") for p in (self.cwd / "configs").glob("*.toml"))
        suites = sorted(str(p.relative_to(self.cwd)).replace("\\", "/")
                        for p in (self.cwd / "suites").glob("*") if p.is_dir() and any(p.glob("*.jsonl")))
        out = []
        for c in COMMANDS.values():
            fields = []
            for f in c.fields:
                d = asdict(f)
                if f.kind == "config":
                    d["choices"] = configs
                elif f.kind == "suite":
                    d["choices"] = suites
                fields.append(d)
            out.append({"name": c.name, "title": c.title, "help": c.help, "lane": c.lane, "fields": fields})
        return out

    # ------------------------------------------------------------ building argv
    def _resolve_path(self, v: str) -> Path:
        p = Path(v)
        return p if p.is_absolute() else self.cwd / p

    def _run_dir(self, v, f: Field) -> Path:
        if not isinstance(v, str) or not SAFE_RUN.match(v) or ".." in v:
            raise JobError(f"{f.label}: not a run name")
        d = self.runs_dir / v
        if not (d / "config.json").exists():
            raise JobError(f"{f.label}: no run {v!r}")
        return d

    def _value(self, f: Field, v) -> list[str]:
        """One validated value -> its argv pieces (without the flag)."""
        if f.kind == "int":
            try:
                n = int(v)
            except (TypeError, ValueError):
                raise JobError(f"{f.label}: expected a whole number") from None
            if isinstance(v, float) and v != n:
                raise JobError(f"{f.label}: expected a whole number")
            self._range(f, n)
            return [str(n)]
        if f.kind == "float":
            try:
                x = float(v)
            except (TypeError, ValueError):
                raise JobError(f"{f.label}: expected a number") from None
            if not math.isfinite(x):
                raise JobError(f"{f.label}: expected a finite number")
            self._range(f, x)
            return [repr(x)]
        if f.kind == "choice":
            if v not in (f.choices or []):
                raise JobError(f"{f.label}: choose one of {f.choices}")
            return [v]
        if f.kind == "text":
            v = _check_text(v, f)
            if f.name == "views" and v and not VIEWS.match(v):
                raise JobError(f"{f.label}: comma-separated view names")
            if f.name == "steps" and v and not STEPS.match(v):
                raise JobError(f"{f.label}: comma-separated step numbers")
            return [v]
        if f.kind == "set":
            v = _check_text(v, f).strip()
            if not SET_ITEM.match(v):
                raise JobError(f"{f.label}: {v!r} isn't KEY=VALUE (e.g. train.lr=3e-4)")
            return [v]
        if f.kind == "run":
            return [str(self._run_dir(v, f))]
        v = _not_an_option(_check_text(v, f).strip(), f)
        if not v:
            raise JobError(f"{f.label}: empty")
        if f.kind == "model":
            if SAFE_RUN.match(v) and (self.runs_dir / v / "config.json").exists():
                d = self.runs_dir / v
                ext = json.loads((d / "config.json").read_text(encoding="utf-8")).get("external")
                # An external model's results folder isn't the model: use the file it points to.
                return [ext["path"] if ext else str(d)]
            if not self._resolve_path(v).exists():
                raise JobError(f"{f.label}: {v!r} is neither one of your runs nor an existing file or folder")
            return [v]
        if f.kind in ("config", "suite"):
            p = self._resolve_path(v)
            if f.kind == "config" and (p.suffix != ".toml" or not p.is_file()):
                raise JobError(f"{f.label}: no config file {v!r}")
            if f.kind == "suite" and not p.exists():
                raise JobError(f"{f.label}: no suite {v!r}")
            return [v]
        if f.kind == "path":
            check = v.rsplit("@", 1)[0] if f.name == "input" else v  # shard's per-source caps: path@MB
            if f.must_exist and not self._resolve_path(check).exists():
                raise JobError(f"{f.label}: {check!r} doesn't exist")
            return [v]
        raise JobError(f"unknown field kind {f.kind}")

    @staticmethod
    def _range(f: Field, x) -> None:
        if (f.min is not None and x < f.min) or (f.max is not None and x > f.max):
            raise JobError(f"{f.label}: must be between {f.min} and {f.max}")

    def build(self, name: str, values: dict) -> tuple[list[str], str, str | None]:
        """(argv after `python -m sleuth`, lane, run name or None)."""
        cmd = COMMANDS.get(name)
        if cmd is None:
            raise JobError(f"unknown command {name!r}")
        if not isinstance(values, dict):
            raise JobError("values must be an object")
        known = {f.name for f in cmd.fields}
        unknown = set(values) - known
        if unknown:
            raise JobError(f"unknown field(s) {sorted(unknown)} for {name}")
        argv = list(cmd.sub)
        positional: list[str] = []
        run = None
        for f in cmd.fields:
            v = values.get(f.name)
            if v is None or v == "" or v == [] or v is False:
                if f.required:
                    raise JobError(f"{f.label} is required")
                continue
            if f.kind == "bool":
                if v is not True:
                    raise JobError(f"{f.label}: expected true/false")
                argv.append(f.flag)
                continue
            vals = v if f.multi else [v]
            if not isinstance(vals, list):
                raise JobError(f"{f.label}: expected a list")
            vals = [x for x in vals if not (isinstance(x, str) and not x.strip())]
            if not vals:
                if f.required:
                    raise JobError(f"{f.label} is required")
                continue
            pieces = [p for x in vals for p in self._value(f, x)]
            if f.kind == "run":
                run = values[f.name]
            elif f.kind == "model" and SAFE_RUN.match(str(v)) and (self.runs_dir / str(v)).is_dir():
                run = str(v)
            if f.flag is None:
                positional += pieces
            elif f.nargs:
                argv += [f.flag, *pieces]
            else:
                # --flag=value, so a value that starts with "-" (a prompt, say)
                # is never mistaken for another option.
                argv += [f"{f.flag}={p}" for p in pieces]
        lane = "cpu" if values.get("device") == "cpu" else cmd.lane
        return argv[:len(cmd.sub)] + positional + argv[len(cmd.sub):], lane, run

    # ------------------------------------------------------------ queue
    def submit(self, name: str, values: dict) -> Job:
        argv, lane, run = self.build(name, values)
        cmd = COMMANDS[name]
        job_id = time.strftime("%Y%m%d-%H%M%S") + f"-{name}-{secrets.token_hex(2)}"
        job = Job(job_id, name, cmd.title, argv, lane, run=run)
        with self.lock:
            self.jobs[job.id] = job
            self._save(job)
            self._schedule()
        return job

    def _lane_busy(self, lane: str) -> bool:
        if lane == "gpu" and self.gpu_holder is not None:
            return True
        return any(j.lane == lane and j.status in ("running", "stopping") for j in self.jobs.values()
                   if j.id in self.procs)

    def _schedule(self) -> None:
        with self.lock:
            for lane in ("gpu", "cpu"):
                if self._lane_busy(lane):
                    continue
                queued = sorted((j for j in self.jobs.values() if j.status == "queued" and j.lane == lane),
                                key=lambda j: j.created)
                if queued:
                    self._start(queued[0])

    def _start(self, job: Job) -> None:
        env = dict(os.environ if self.env is None else self.env)
        # Unbuffered UTF-8 output, so the log fills as the job prints and any
        # character survives (a Windows console code page would mangle some).
        env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", SLEUTH_RUNS=str(self.runs_dir))
        if self.data_dir is not None:
            env.setdefault("SLEUTH_DATA", str(self.data_dir))
        kwargs = {}
        if os.name == "nt":
            # Its own process group: Ctrl-C in the dashboard's console doesn't
            # reach it, and we can send it Ctrl-Break to stop it.
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        self.dir.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path(job.id), "ab")
        try:
            proc = subprocess.Popen(self.prefix + job.argv, cwd=self.cwd, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, **kwargs)
        except OSError as e:
            log.close()
            job.status, job.ended, job.note = "failed", time.time(), f"couldn't start: {e}"
            self._save(job)
            return
        log.close()  # the child has its own handle
        job.status, job.started, job.pid, job.note = "running", time.time(), proc.pid, ""
        self.procs[job.id] = proc
        self._save(job)
        threading.Thread(target=self._watch, args=(job, proc), daemon=True, name=f"job-{job.id}").start()

    def _watch(self, job: Job, proc: subprocess.Popen) -> None:
        """Wait for the process; meanwhile, pick its run folder out of the log."""
        offset = 0
        while True:
            rc = proc.poll()
            offset = self._scan_log(job, offset)
            if rc is not None:
                break
            time.sleep(0.3)
        with self.lock:
            job.exit_code, job.ended = rc, time.time()
            if job.status == "stopping":
                job.status = "stopped"
            else:
                job.status = "done" if rc == 0 else "failed"
            self.procs.pop(job.id, None)
            self._save(job)
            self._schedule()

    def _scan_log(self, job: Job, offset: int) -> int:
        try:
            with open(self.log_path(job.id), "rb") as fh:
                fh.seek(offset)
                data = fh.read()
        except OSError:
            return offset
        data = data[:data.rfind(b"\n") + 1]  # whole lines only
        with self.lock:
            if job.run is None:
                for m in RUN_LINE.finditer(data):
                    p = Path(m.group(1).decode("utf-8", "replace").strip())
                    p = p if p.is_absolute() else self.cwd / p
                    if p.resolve().parent == self.runs_dir:
                        job.run = p.name
                        self._save(job)
            if STOP_READY in data:
                self.stop_ready.add(job.id)
        return offset + len(data)

    # ------------------------------------------------------------ stopping
    def stop(self, job_id: str) -> Job:
        with self.lock:
            job = self.get(job_id)
            if job.status == "queued":
                job.status, job.ended, job.note = "cancelled", time.time(), "cancelled before it started"
                self._save(job)
                return job
            proc = self.procs.get(job.id)
            if proc is None or job.status not in ("running", "stopping"):
                raise JobError(f"job {job_id} isn't running ({job.status})")
            if job.status == "stopping":
                proc.kill()  # second click: no more waiting
                job.note = "killed"
                self._save(job)
                return job
            job.status = "stopping"
            ready = job.id in self.stop_ready
            if job.command in ("train", "resume") and job.run and ready:
                # Between two steps the trainer sees this, saves a resumable
                # checkpoint and exits. Cleaner than a signal, which can land
                # in the middle of an optimizer step.
                (self.runs_dir / job.run / "STOP").touch()
                job.note = "STOP file created: stopping after the current step (with a checkpoint)"
            else:
                _interrupt(proc)
                job.note = "asked to stop"
            self._save(job)
            return job

    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id) if isinstance(job_id, str) else None
        if job is None:
            raise JobError(f"no job {job_id!r}")
        return job

    def list(self) -> list[dict]:
        # Show run paths the short way ("runs/2026...-mix-20m"), as you'd type them.
        try:
            short = os.path.relpath(self.runs_dir, self.cwd)
        except ValueError:  # another drive (Windows)
            short = str(self.runs_dir)
        with self.lock:
            out = []
            for j in sorted(self.jobs.values(), key=lambda j: j.created, reverse=True):
                d = j.public()
                if not short.startswith(".."):
                    d["display"] = d["display"].replace(str(self.runs_dir), short)
                if j.status == "queued":
                    d["note"] = self._waiting_note(j)
                out.append(d)
            return out

    def _waiting_note(self, job: Job) -> str:
        if job.lane == "gpu" and self.gpu_holder == "playground":
            return "waiting for the GPU: the Playground has a model loaded (Unload it there to continue)"
        ahead = [j for j in self.jobs.values() if j.lane == job.lane and j.id != job.id and
                 (j.status in ("running", "stopping") or (j.status == "queued" and j.created < job.created))]
        return f"waiting for {len(ahead)} {job.lane.upper()} job(s) ahead of it" if ahead else ""

    # ------------------------------------------------------------ logs
    def read_log(self, job_id: str, offset: int = 0, max_bytes: int = 256 * 1024) -> dict:
        """{"text", "offset" (where to continue), "done"}. A negative offset
        means "the last -offset bytes" (for opening a long log)."""
        job = self.get(job_id)
        path = self.log_path(job.id)
        size = path.stat().st_size if path.exists() else 0
        start = max(0, size + offset) if offset < 0 else min(offset, size)
        with self.lock:
            finished = job.status in FINISHED
        data = b""
        if size > start:
            with open(path, "rb") as fh:
                fh.seek(start)
                data = fh.read(max_bytes)
            if offset < 0 and start > 0:
                cut = data.find(b"\n") + 1  # start at a line boundary
                start, data = start + cut, data[cut:]
            if not (finished and start + len(data) >= size):
                # Send whole lines only: a line may still be being written,
                # and a cut could split a multi-byte character.
                data = data[:data.rfind(b"\n") + 1]
        nxt = start + len(data)
        return {"text": data.decode("utf-8", "replace"), "offset": nxt, "done": finished and nxt >= size}

    # ------------------------------------------------------------ the GPU lane, for the Playground
    def acquire_gpu(self, holder: str) -> None:
        with self.lock:
            if self.gpu_holder == holder:
                return
            running = [j for j in self.jobs.values() if j.lane == "gpu" and j.status in ("running", "stopping")
                       and j.id in self.procs]
            if running:
                raise GpuBusy(f"GPU busy: job {running[0].id} ({running[0].title}) is running. "
                              "Wait for it, stop it in Jobs, or use device = cpu.")
            if any(j.lane == "gpu" and j.status == "queued" for j in self.jobs.values()):
                raise GpuBusy("GPU busy: a job is waiting for it in Jobs")
            self.gpu_holder = holder

    def release_gpu(self, holder: str) -> None:
        with self.lock:
            if self.gpu_holder == holder:
                self.gpu_holder = None
                self._schedule()

    def shutdown(self) -> list[Job]:
        """Cancel queued jobs; leave running ones alone. Returns the running ones."""
        with self.lock:
            for j in self.jobs.values():
                if j.status == "queued":
                    j.status, j.note = "cancelled", "the dashboard closed before this job started"
                    self._save(j)
            return [j for j in self.jobs.values() if j.status in ("running", "stopping")]


def _interrupt(proc: subprocess.Popen) -> None:
    """The polite way to ask a process to stop: SIGTERM, or Ctrl-Break on
    Windows (both make the trainer save a checkpoint)."""
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
    except OSError:
        pass
