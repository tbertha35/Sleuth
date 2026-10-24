"""Local dashboard server: stdlib only, binds to 127.0.0.1 by default.

    python -m sleuth ui                 # then open http://127.0.0.1:8765

Serves the static app from ui/static/ (no CDN, no build step) plus a small
read-only JSON API over the runs directory:

    GET /api/runs                                    list runs
    GET /api/runs/<run>                              config, env, snapshots, analyses
    GET /api/runs/<run>/metrics                      training + validation curves
    GET /api/runs/<run>/analysis/<view>/<suite>/timeline
    GET /api/runs/<run>/analysis/<view>/<suite>/<step_dir>   one checkpoint's details

It only ever reads files under the runs directory. Every path component
from the URL is checked against a strict pattern, so a crafted URL like
/api/runs/../../etc can't escape it.
"""

from __future__ import annotations

import json
import mimetypes
import re
import sys
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np

STATIC = Path(__file__).resolve().parent / "static"
SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, msg: str):
        super().__init__(msg)
        self.status = status


def _safe(part: str) -> str:
    if not SAFE.match(part) or ".." in part:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"bad path component {part!r}")
    return part


def _read_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a line being written right now; skip it
    return out


class Api:
    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir

    def _run(self, run: str) -> Path:
        d = self.runs_dir / _safe(run)
        if not (d / "config.json").exists():
            raise ApiError(HTTPStatus.NOT_FOUND, f"no run {run!r}")
        return d

    def runs(self) -> list[dict]:
        out = []
        if not self.runs_dir.exists():
            return out
        for d in sorted(self.runs_dir.iterdir(), reverse=True):
            cfg = _read_json(d / "config.json")
            if cfg is None:
                continue
            metrics = _read_jsonl(d / "metrics.jsonl")
            last_train = next((m for m in reversed(metrics) if "loss" in m), {})
            last_val = next((m for m in reversed(metrics) if "val_bpb" in m), {})
            out.append({
                "id": d.name,
                "name": cfg.get("name"),
                "max_steps": cfg["train"]["max_steps"],
                "step": last_train.get("step", 0),
                "loss": last_train.get("loss"),
                "val_bpb": last_val.get("val_bpb"),
            })
        return out

    def run(self, run: str) -> dict:
        from ..forge.model.config import ModelConfig
        from ..forge.model.flops import param_counts

        d = self._run(run)
        cfg = _read_json(d / "config.json")
        snaps = sorted(int(p.name[5:]) for p in (d / "snapshots").glob("step_*")) if (d / "snapshots").exists() else []
        analyses = []
        for view_dir in sorted((d / "analysis").glob("*")) if (d / "analysis").exists() else []:
            for suite_dir in sorted(view_dir.glob("*")):
                steps = sorted(p.name for p in suite_dir.glob("step_*") if (p / "summary.json").exists())
                analyses.append({"view": view_dir.name, "suite": suite_dir.name, "checkpoints": steps})
        try:
            params = param_counts(ModelConfig(**cfg["model"]))
        except (TypeError, ValueError):
            params = None
        return {"id": d.name, "config": cfg, "env": _read_json(d / "env.json"), "params": params,
                "snapshots": snaps, "analyses": analyses}

    def metrics(self, run: str) -> dict:
        recs = _read_jsonl(self._run(run) / "metrics.jsonl")
        return {"train": [r for r in recs if "loss" in r], "val": [r for r in recs if "val_loss" in r]}

    def timeline(self, run: str, view: str, suite: str) -> dict:
        p = self._run(run) / "analysis" / _safe(view) / _safe(suite) / "timeline.json"
        t = _read_json(p)
        if t is None:
            raise ApiError(HTTPStatus.NOT_FOUND, "no timeline yet; run `python -m sleuth analyze`")
        return t

    def checkpoint(self, run: str, view: str, suite: str, step_dir: str) -> dict:
        d = self._run(run) / "analysis" / _safe(view) / _safe(suite) / _safe(step_dir)
        summary = _read_json(d / "summary.json")
        if summary is None:
            raise ApiError(HTTPStatus.NOT_FOUND, f"no analysis for {step_dir}")
        arrays = {}
        npz = d / "arrays.npz" if (d / "arrays.npz").exists() else d / "neurons.npz"  # older runs: neurons.npz
        if npz.exists():
            with np.load(npz) as z:
                for k in z.files:
                    arrays[k] = _to_json_array(z[k])
        return {"summary": summary, "arrays": arrays}


def _to_json_array(a: np.ndarray):
    """Rounded nested lists; NaN (e.g. "no answers in this category") becomes
    null, because JSON has no NaN and the browser's parser would reject it."""
    a = np.round(a.astype(np.float64), 4)
    if not np.isnan(a).any():
        return a.tolist()
    obj = a.astype(object)
    obj[np.isnan(a)] = None
    return obj.tolist()


class Handler(BaseHTTPRequestHandler):
    server_version = "SleuthUI/0.1"

    def __init__(self, *args, api: Api, **kwargs):
        self.api = api
        super().__init__(*args, **kwargs)

    def log_message(self, fmt, *args):  # quieter than the default (one line per request to stderr)
        if "/api/" not in (args[0] if args else ""):
            sys.stderr.write("ui: " + fmt % args + "\n")

    def _send(self, status: HTTPStatus, body: bytes, ctype: str):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        # The page may only load its own files: no external scripts, fonts or calls.
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=HTTPStatus.OK):
        self._send(status, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        try:
            if path.startswith("/api/"):
                self._json(self._route_api([p for p in path[5:].split("/") if p]))
            else:
                self._static(path)
        except ApiError as e:
            self._json({"error": str(e)}, e.status)
        except Exception as e:  # noqa: BLE001 - report errors to the page instead of dropping the connection
            self._json({"error": f"{type(e).__name__}: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _route_api(self, parts: list[str]):
        a = self.api
        if parts == ["runs"]:
            return a.runs()
        if len(parts) >= 2 and parts[0] == "runs":
            run = parts[1]
            rest = parts[2:]
            if not rest:
                return a.run(run)
            if rest == ["metrics"]:
                return a.metrics(run)
            if len(rest) == 4 and rest[0] == "analysis":
                _, view, suite, last = rest
                return a.timeline(run, view, suite) if last == "timeline" else a.checkpoint(run, view, suite, last)
        raise ApiError(HTTPStatus.NOT_FOUND, "unknown endpoint")

    def _static(self, path: str):
        name = "index.html" if path in ("/", "") else path.lstrip("/")
        if not all(SAFE.match(p) for p in name.split("/")):
            raise ApiError(HTTPStatus.NOT_FOUND, "not found")
        f = (STATIC / name).resolve()
        if STATIC not in f.parents or not f.is_file():
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            return
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        if f.suffix == ".js":
            ctype = "text/javascript"
        self._send(HTTPStatus.OK, f.read_bytes(), ctype + ("; charset=utf-8" if ctype.startswith("text") else ""))


def make_server(runs_dir: str | Path, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    api = Api(Path(runs_dir))
    return ThreadingHTTPServer((host, port), partial(Handler, api=api))


def serve(runs_dir: str | Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    httpd = make_server(runs_dir, host, port)
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"warning: listening on {host}, so other machines on your network can see this dashboard")
    print(f"Sleuth dashboard: http://{host}:{port}  (runs dir: {Path(runs_dir).resolve()})  Ctrl-C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
