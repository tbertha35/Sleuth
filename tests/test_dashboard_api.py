"""The dashboard's HTTP API beyond read-only viewing: the session token and
Host checks, jobs over HTTP, streamed logs, the Playground, the Data page,
and stale-run flags."""

import json
import sys
import threading
import urllib.request
from pathlib import Path
from urllib.error import HTTPError

import pytest
from helpers import tiny_run_config

from sleuth.forge.train.trainer import train
from sleuth.ui.server import make_server

FAKE = [sys.executable, "-u", str(Path(__file__).parent / "fake_sleuth.py")]
TOKEN = "test-token"


@pytest.fixture(scope="module")
def env(tiny_data, tmp_path_factory):
    root = tmp_path_factory.mktemp("dash")
    runs = root / "runs"
    run = train(tiny_run_config(tiny_data, runs), log=lambda s: None)
    # A stale external run from before runs were named by path hash, and its replacement.
    model = root / "hf"
    model.mkdir()
    for name, cfg in [("ext-hf", {"name": "hf", "external": {"path": str(model), "format": "hf"}}),
                      ("ext-x-hf-" + _hash(model), {"name": "x", "external": {"path": str(model), "format": "hf",
                                                                             "keep_quantized": "auto"}})]:
        (runs / name).mkdir()
        (runs / name / "config.json").write_text(json.dumps({**cfg, "model": {}, "train": {"max_steps": 0}}))
    data = root / "data"
    (data / "raw" / "tiny").mkdir(parents=True)
    (data / "raw" / "tiny" / "a.txt").write_text("hello world\n")
    (data / "shards").mkdir()
    import shutil
    shutil.copytree(tiny_data["shards"], data / "shards" / "tiny")
    httpd = make_server(runs, port=0, token=TOKEN, data_dir=data, cwd=Path.cwd(), job_prefix=FAKE)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield {"url": f"http://127.0.0.1:{httpd.server_address[1]}", "run": run, "runs": runs, "httpd": httpd}
    httpd.playground.unload()
    httpd.shutdown()


def _hash(p):
    import hashlib
    import os
    return hashlib.sha1(os.path.normcase(str(p.resolve())).encode()).hexdigest()[:6]


def call(url, body=None, token=TOKEN, headers=None):
    h = {"Content-Type": "application/json", **(headers or {})}
    if token:
        h["X-Sleuth-Token"] = token
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), headers=h,
                                 method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, r.read()


def events(raw: bytes) -> list[tuple[str, dict]]:
    out = []
    for block in raw.decode("utf-8").split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line and not line.startswith(":"))
        if "event" in lines:
            out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_posts_need_the_token_and_our_own_origin(env):
    u = env["url"] + "/api/jobs"
    body = {"command": "eval", "values": {"run": env["run"].name}}
    for kwargs in [{"token": None}, {"token": "wrong"}, {"headers": {"Origin": "http://evil.example"}}]:
        with pytest.raises(HTTPError) as e:
            call(u, body, **kwargs)
        assert e.value.code == 403
    status, raw = call(u, body, headers={"Origin": env["url"]})
    assert status == 201 and json.loads(raw)["command"] == "eval"


def test_host_header_must_be_loopback(env):
    # A DNS-rebinding page reaches us under its own name.
    with pytest.raises(HTTPError) as e:
        call(env["url"] + "/api/runs", headers={"Host": "evil.example:80"})
    assert e.value.code == 403
    port = env["url"].rsplit(":", 1)[1]
    assert call(env["url"] + "/api/runs", headers={"Host": f"localhost:{port}"})[0] == 200


def test_bad_job_requests_are_400(env):
    with pytest.raises(HTTPError) as e:
        call(env["url"] + "/api/jobs", {"command": "bash", "values": {}})
    assert e.value.code == 400 and b"unknown command" in e.value.read()


def test_job_log_streams_until_the_end(env):
    _, raw = call(env["url"] + "/api/jobs", {"command": "eval", "values": {"run": env["run"].name, "device": "cpu"}})
    job = json.loads(raw)
    _, raw = call(f"{env['url']}/api/jobs/{job['id']}/log")  # returns once the job is over
    ev = events(raw)
    assert ev[-1][0] == "end" and ev[-1][1]["job"]["status"] == "done"
    assert "finished" in "".join(d["text"] for e, d in ev if e == "log")
    listed = json.loads(call(env["url"] + "/api/jobs")[1])
    assert any(j["id"] == job["id"] and j["display"].startswith("sleuth eval") for j in listed)
    specs = json.loads(call(env["url"] + "/api/jobs/specs")[1])
    train_spec = next(s for s in specs if s["name"] == "train")
    assert "configs/tiny-10m.toml" in train_spec["fields"][0]["choices"]


def test_runs_flag_stale_and_live(env):
    runs = {r["id"]: r for r in json.loads(call(env["url"] + "/api/runs")[1])}
    assert runs["ext-hf"]["stale"] and "replaced by ext-x-hf-" in runs["ext-hf"]["stale"]
    assert runs[f"ext-x-hf-{_hash(env['runs'].parent / 'hf')}"]["stale"] is None
    assert runs[env["run"].name]["stale"] is None and runs[env["run"].name]["live"] is False  # finished


def test_data_page(env):
    d = json.loads(call(env["url"] + "/api/data")[1])
    sec = {s["name"]: s for s in d["sections"]}
    assert sec["raw"]["entries"][0]["name"] == "tiny" and sec["raw"]["entries"][0]["bytes"] == 12
    shards = sec["shards"]["entries"][0]["shards"]
    assert shards["train_tokens"] > 0 and shards["vocab_size"] == 512 and shards["sources"]
    assert not sec["finetune"]["exists"]


def test_playground_generate_streams_tokens(env):
    model = {"model": env["run"].name, "device": "cpu"}
    _, raw = call(env["url"] + "/api/playground/generate",
                  {"model": model, "mode": "text", "prompt": "Once upon a time", "max_new_tokens": 12,
                   "temperature": 0, "no_repeat_ngram": 3})
    ev = events(raw)
    kinds = [e for e, _ in ev]
    assert kinds[-1] == "done" and "token" in kinds and "error" not in kinds
    done = ev[-1][1]
    assert done["tokens"] >= 1 and done["prompt_tokens"] > 1 and "@ step" in done["model"]
    state = json.loads(call(env["url"] + "/api/playground")[1])
    assert len(state["models"]) == 1 and state["models"][0]["device"] == "cpu"
    assert state["gpu_holder"] is None  # a CPU model doesn't take the GPU lane


def test_playground_next_tokens(env):
    _, raw = call(env["url"] + "/api/playground/next",
                  {"model": {"model": env["run"].name, "device": "cpu"}, "prompt": "Once upon a", "top": 5})
    r = json.loads(raw)
    probs = [c["prob"] for c in r["candidates"]]
    assert len(probs) == 5 and probs == sorted(probs, reverse=True) and 0 < sum(probs) <= 1.0001
    assert r["entropy"] > 0


def test_playground_errors_become_events(env):
    _, raw = call(env["url"] + "/api/playground/generate",
                  {"model": {"model": env["run"].name, "device": "cpu"}, "mode": "chat",
                   "messages": [{"role": "user", "content": "hi"}]})
    (kind, data), = events(raw)[-1:]
    assert kind == "error" and "no chat format" in data["error"]
    with pytest.raises(HTTPError) as e:
        call(env["url"] + "/api/playground/next", {"model": {"model": "no/such/thing"}})
    assert e.value.code == 400


def test_playground_reports_gpu_busy(env, monkeypatch):
    # Pretend a GPU job is running and the model would go on the GPU.
    import torch

    import sleuth.forge.train.trainer as trainer
    from sleuth.ui.jobs import GpuBusy

    def busy(holder):
        raise GpuBusy("GPU busy: job x is running")

    monkeypatch.setattr(env["httpd"].jobs, "acquire_gpu", busy)
    monkeypatch.setattr(trainer, "resolve_device", lambda spec: torch.device("cuda"))
    _, raw = call(env["url"] + "/api/playground/generate",
                  {"model": {"model": env["run"].name, "device": "cuda"}, "prompt": "x"})
    kind, data = events(raw)[-1]
    assert kind == "error" and data.get("busy") and "GPU busy" in data["error"]


def test_unload(env):
    state = json.loads(call(env["url"] + "/api/playground/unload", {})[1])
    assert state["models"] == []
