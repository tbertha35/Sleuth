"""Local dashboard server: stdlib only, binds to 127.0.0.1 by default.

    python -m sleuth ui                 # then open the link it prints

Serves the static app from ui/static/ (no CDN, no build step) plus a JSON API.

Reading (no token needed):
    GET /api/runs                                    list runs (stale ones flagged, live ones marked)
    GET /api/runs/<run>                              config, env, snapshots, analyses
    GET /api/runs/<run>/metrics                      training + validation curves
    GET /api/runs/<run>/analysis/<view>/<suite>/timeline
    GET /api/runs/<run>/analysis/<view>/<suite>/<step_dir>   one checkpoint's details
    GET /api/jobs, /api/jobs/specs, /api/jobs/<id>   job list, the command forms, one job
    GET /api/jobs/<id>/log                           the job's output, streamed (Server-Sent Events)
    GET /api/playground                              loaded models
    GET /api/data                                    what's in the data folder

Doing things (every POST must carry the session token):
    POST /api/jobs                {"command", "values"}   start (or queue) a job
    POST /api/jobs/<id>/stop                              stop it (twice: kill it)
    POST /api/playground/generate                         stream a generation (Server-Sent Events)
    POST /api/playground/next                             next-token candidates
    POST /api/playground/unload                           free the loaded models

Why a token: once the dashboard can start processes, any other web page open
in the same browser could try to send it requests (the browser happily sends
a POST to 127.0.0.1). It can't read our pages, so it can't learn the random
token, which `sleuth ui` prints in its link and the page keeps for the
session. The token goes in an X-Sleuth-Token header, which a cross-site page
can't add without the server's permission (we never grant it).

Why the Host check: "DNS rebinding" makes an attacker's domain resolve to
127.0.0.1, so the browser treats our server as their site and lets them read
it. Their requests then carry their domain in the Host header, so we accept
only Host headers naming this server by its loopback address.

It only ever reads files under the runs and data directories. Every path
component from the URL is checked against a strict pattern, so a crafted URL
like /api/runs/../../etc can't escape it.
"""

from __future__ import annotations

import hmac
import json
import mimetypes
import re
import secrets
import sys
import time
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from ..runs import scan
from .jobs import GpuBusy, JobError, JobManager
from .playground import Playground

STATIC = Path(__file__).resolve().parent / "static"
SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
LIVE_SECONDS = 120   # a run whose metrics changed this recently counts as training right now
MAX_BODY = 1 << 20


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
    def __init__(self, runs_dir: Path, jobs: JobManager | None = None, data_dir: Path | None = None):
        self.runs_dir = runs_dir
        self.jobs = jobs
        self.data_dir = data_dir

    def _run(self, run: str) -> Path:
        d = self.runs_dir / _safe(run)
        if not (d / "config.json").exists():
            raise ApiError(HTTPStatus.NOT_FOUND, f"no run {run!r}")
        return d

    def _job_runs(self) -> set[str]:
        """Runs that a running job is working on."""
        if self.jobs is None:
            return set()
        return {j["run"] for j in self.jobs.list() if j["run"] and not j["finished"]}

    def runs(self) -> list[dict]:
        out = []
        busy = self._job_runs()
        for r in scan(self.runs_dir):
            d, cfg = r["path"], r["config"]
            metrics = _read_jsonl(d / "metrics.jsonl")
            last_train = next((m for m in reversed(metrics) if "loss" in m), {})
            last_val = next((m for m in reversed(metrics) if "val_bpb" in m), {})
            mf = d / "metrics.jsonl"
            updated = mf.stat().st_mtime if mf.exists() else None
            max_steps = (cfg.get("train") or {}).get("max_steps", 0)
            out.append({
                "id": d.name,
                "name": cfg.get("name"),
                "external": bool(cfg.get("external")),
                "max_steps": max_steps,
                "step": last_train.get("step", 0),
                "loss": last_train.get("loss"),
                "val_bpb": last_val.get("val_bpb"),
                "updated": updated,
                # "live": being trained right now, so the page refreshes its curves.
                "live": d.name in busy or (updated is not None and time.time() - updated < LIVE_SECONDS
                                           and last_train.get("step", 0) < max_steps),
                "stale": r["stale"],
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

    def data(self) -> dict:
        from .data import describe_data

        if self.data_dir is None:
            raise ApiError(HTTPStatus.NOT_FOUND, "no data folder configured")
        return describe_data(self.data_dir)


def _to_json_array(a: np.ndarray):
    """Rounded nested lists; NaN (e.g. "no answers in this category") becomes
    null, because JSON has no NaN and the browser's parser would reject it."""
    a = np.round(a.astype(np.float64), 4)
    if not np.isnan(a).any():
        return a.tolist()
    obj = a.astype(object)
    obj[np.isnan(a)] = None
    return obj.tolist()


class ClientGone(Exception):
    """The page closed the connection (e.g. the Stop button aborted a stream)."""


class Handler(BaseHTTPRequestHandler):
    server_version = "SleuthUI/0.2"

    def __init__(self, *args, api: Api, jobs: JobManager, playground: Playground, token: str, **kwargs):
        self.api, self.jobs, self.playground, self.token = api, jobs, playground, token
        super().__init__(*args, **kwargs)

    def log_message(self, fmt, *args):  # quieter than the default (one line per request to stderr)
        if "/api/" not in (args[0] if args else ""):
            sys.stderr.write("ui: " + fmt % args + "\n")

    def _headers(self, status: HTTPStatus, ctype: str, length: int | None = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        # The page may only load its own files: no external scripts, fonts or calls.
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")  # the link carries the token
        self.end_headers()

    def _send(self, status: HTTPStatus, body: bytes, ctype: str):
        self._headers(status, ctype, len(body))
        self.wfile.write(body)

    def _json(self, obj, status=HTTPStatus.OK):
        self._send(status, json.dumps(obj).encode(), "application/json")

    # ------------------------------------------------------------ security
    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").lower()
        bound, port = self.server.server_address[:2]
        if bound in ("0.0.0.0", "::"):
            return True  # `--host 0.0.0.0`: reachable under any name on purpose; the token still guards POSTs
        names = {"127.0.0.1", "localhost", "[::1]", str(bound).lower()}
        return host in {f"{n}:{port}" for n in names}

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        return origin is None or origin.lower() == f"http://{(self.headers.get('Host') or '').lower()}"

    def _token_ok(self) -> bool:
        return hmac.compare_digest(self.headers.get("X-Sleuth-Token") or "", self.token)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request too large")
        raw = self.rfile.read(n) if n else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            raise ApiError(HTTPStatus.BAD_REQUEST, "body must be JSON") from None
        if not isinstance(body, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "body must be a JSON object")
        return body

    # ------------------------------------------------------------ dispatch
    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def _dispatch(self, method: str):
        url = urlparse(self.path)
        path = unquote(url.path)
        try:
            if not self._host_ok():
                raise ApiError(HTTPStatus.FORBIDDEN, "unexpected Host header (open the link `sleuth ui` printed)")
            parts = [p for p in path[5:].split("/") if p] if path.startswith("/api/") else None
            if method == "POST":
                if parts is None:
                    raise ApiError(HTTPStatus.NOT_FOUND, "not found")
                if not self._origin_ok():
                    raise ApiError(HTTPStatus.FORBIDDEN, "request from another site refused")
                if not self._token_ok():
                    raise ApiError(HTTPStatus.FORBIDDEN, "missing or wrong session token: open the link that "
                                   "`sleuth ui` printed (it ends in ?token=...)")
                self._route_post(parts, self._body())
            elif parts is not None:
                self._route_get(parts, parse_qs(url.query))
            else:
                self._static(path)
        except ApiError as e:
            self._json({"error": str(e)}, e.status)
        except GpuBusy as e:
            self._json({"error": str(e)}, HTTPStatus.CONFLICT)
        except JobError as e:
            self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
        except ClientGone:
            pass
        except Exception as e:  # noqa: BLE001 - report errors to the page instead of dropping the connection
            try:
                self._json({"error": f"{type(e).__name__}: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            except OSError:
                pass

    def _route_get(self, parts: list[str], query: dict):
        a = self.api
        if parts == ["runs"]:
            return self._json(a.runs())
        if parts == ["data"]:
            return self._json(a.data())
        if parts == ["playground"]:
            return self._json(self.playground.state())
        if parts == ["jobs"]:
            return self._json(self.jobs.list())
        if parts == ["jobs", "specs"]:
            return self._json(self.jobs.specs())
        if len(parts) == 2 and parts[0] == "jobs":
            return self._json(self.jobs.get(parts[1]).public())
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "log":
            # A reconnecting EventSource sends the last offset it saw as Last-Event-ID;
            # that wins over the ?offset= it started with.
            offset = int(self.headers.get("Last-Event-ID") or (query.get("offset") or ["0"])[0])
            if (query.get("format") or [""])[0] == "json":
                return self._json(self.jobs.read_log(parts[1], offset))
            return self._stream_log(parts[1], offset)
        if len(parts) >= 2 and parts[0] == "runs":
            run = parts[1]
            rest = parts[2:]
            if not rest:
                return self._json(a.run(run))
            if rest == ["metrics"]:
                return self._json(a.metrics(run))
            if len(rest) == 4 and rest[0] == "analysis":
                _, view, suite, last = rest
                return self._json(a.timeline(run, view, suite) if last == "timeline"
                                  else a.checkpoint(run, view, suite, last))
        raise ApiError(HTTPStatus.NOT_FOUND, "unknown endpoint")

    def _route_post(self, parts: list[str], body: dict):
        if parts == ["jobs"]:
            job = self.jobs.submit(body.get("command"), body.get("values") or {})
            return self._json(job.public(), HTTPStatus.CREATED)
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "stop":
            return self._json(self.jobs.stop(parts[1]).public())
        if parts == ["playground", "unload"]:
            return self._json(self.playground.unload())
        if parts == ["playground", "next"]:
            return self._json(self.playground.next_tokens(body))
        if parts == ["playground", "generate"]:
            return self._stream(lambda emit: self.playground.generate(body, emit))
        raise ApiError(HTTPStatus.NOT_FOUND, "unknown endpoint")

    # ------------------------------------------------------------ Server-Sent Events
    # A plain HTTP response that stays open: each message is
    #   "event: <name>\ndata: <json>\n\n"
    # The browser reads them as they arrive (EventSource for GET, a streamed
    # fetch for POST). No library, no WebSocket handshake.
    def _event(self, event: str, data, id: int | None = None):
        msg = (f"id: {id}\n" if id is not None else "") + f"event: {event}\ndata: {json.dumps(data)}\n\n"
        try:
            self.wfile.write(msg.encode("utf-8"))
            self.wfile.flush()
        except OSError as e:  # broken pipe / connection reset / aborted (Windows)
            raise ClientGone() from e

    def _stream(self, work):
        self._headers(HTTPStatus.OK, "text/event-stream; charset=utf-8")
        try:
            work(self._event)
        except ClientGone:
            raise
        except GpuBusy as e:
            self._event("error", {"error": str(e), "busy": True})
        except Exception as e:  # noqa: BLE001 - the stream has started, so errors become an event
            self._event("error", {"error": str(e) if isinstance(e, JobError) else f"{type(e).__name__}: {e}"})

    def _stream_log(self, job_id: str, offset: int):
        self.jobs.get(job_id)  # 404-style error before the stream starts
        self._headers(HTTPStatus.OK, "text/event-stream; charset=utf-8")
        last_write = time.time()
        while True:
            r = self.jobs.read_log(job_id, offset)
            if r["text"] or r["offset"] != offset:
                offset = r["offset"]
                self._event("log", {"text": r["text"]}, id=offset)
                last_write = time.time()
                continue  # more may be waiting (reads are capped)
            if r["done"]:
                self._event("end", {"job": self.jobs.get(job_id).public()}, id=offset)
                return
            if time.time() - last_write > 15:
                # A comment line keeps proxies from timing out and tells us when the page has gone.
                try:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                except OSError:
                    return
                last_write = time.time()
            time.sleep(0.4)

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


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True  # open log streams must not keep the process alive on Ctrl-C
    jobs: JobManager
    playground: Playground
    token: str


def make_server(runs_dir: str | Path, host: str = "127.0.0.1", port: int = 8765, token: str | None = None,
                data_dir: str | Path | None = None, cwd: str | Path | None = None,
                job_prefix: list[str] | None = None) -> DashboardServer:
    runs_dir = Path(runs_dir)
    jobs = JobManager(runs_dir, cwd=cwd, prefix=job_prefix, data_dir=data_dir)
    playground = Playground(runs_dir, jobs)
    api = Api(runs_dir, jobs, Path(data_dir) if data_dir else None)
    token = token or secrets.token_urlsafe(24)
    httpd = DashboardServer((host, port), partial(Handler, api=api, jobs=jobs, playground=playground, token=token))
    httpd.jobs, httpd.playground, httpd.token = jobs, playground, token
    return httpd


def serve(runs_dir: str | Path, host: str = "127.0.0.1", port: int = 8765, data_dir: str | Path | None = None) -> None:
    httpd = make_server(runs_dir, host, port, data_dir=data_dir)
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"warning: listening on {host}, so other machines on your network can see this dashboard "
              "(and, with the link below, start jobs on this machine)")
    shown = "127.0.0.1" if host in ("127.0.0.1", "", "0.0.0.0") else host
    print(f"Sleuth dashboard: http://{shown}:{httpd.server_address[1]}/?token={httpd.token}")
    print(f"  runs: {Path(runs_dir).resolve()}" + (f"   data: {Path(data_dir).resolve()}" if data_dir else ""))
    print("  Open the link above: its token lets the page start jobs. Ctrl-C to stop the dashboard "
          "(running jobs keep going).")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        still = httpd.jobs.shutdown()
        httpd.server_close()
        if still:
            print(f"{len(still)} job(s) still running and will finish on their own: "
                  + ", ".join(j.id for j in still))

