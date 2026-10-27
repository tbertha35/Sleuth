"""The network gate (sleuth/net.py), and that nothing else can reach the network."""

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from sleuth import net

ROOT = Path(__file__).resolve().parents[1]


def test_policy():
    off = {}
    assert net.allowed("http://127.0.0.1:8000/v1/completions", off)
    assert net.allowed("http://localhost:8080/x", off)
    for lan_ip in ("10.0.0.5", "192.168.1.20", "172.20.0.3"):      # a vLLM server on the home network
        assert net.allowed(f"http://{lan_ip}:8000/v1", off)
    assert not net.allowed("http://8.8.8.8/x", off)
    assert not net.allowed("https://huggingface.co/x", off)
    assert not net.allowed("file:///etc/passwd", off)
    strict = {"allow_lan": False, "local_hosts": ["10.0.0.5"]}
    assert net.allowed("http://10.0.0.5:8000/v1", strict) and not net.allowed("http://10.0.0.6:8000/v1", strict)
    online = {"mode": "online", "online_hosts": ["huggingface.co"]}
    assert net.allowed("https://huggingface.co/x", online) and not net.allowed("https://example.com", online)
    with pytest.raises(net.NetworkPolicyError):
        net.request_json("http://8.8.8.8/v1/models", policy=off)


def test_loopback_request_is_made_and_logged(tmp_path):
    class Echo(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = json.dumps({"got": body, "auth": self.headers.get("Authorization")}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Echo)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        log = tmp_path / "net.log"
        r = net.request_json(f"http://127.0.0.1:{srv.server_port}/v1/x", {"a": 1}, policy={}, log_file=log)
    finally:
        srv.shutdown()
    assert r == {"got": {"a": 1}, "auth": None}
    assert "POST http://127.0.0.1" in log.read_text() and '"a"' not in log.read_text()   # no payloads in the log


def test_no_other_module_opens_connections():
    """PLAN.md §1: only net.py may use outbound network libraries. (socket is allowed
    for gethostname in env.py and the hardware probe; http.server is the dashboard's
    inbound server, bound to 127.0.0.1.)"""
    pattern = re.compile(r"^\s*(?:import|from)\s+(urllib\.request|http\.client|requests|httpx|aiohttp|socket)\b", re.M)
    socket_ok = {ROOT / "sleuth" / "env.py", ROOT / "scripts" / "probe_hardware.py"}
    bad = []
    for f in list((ROOT / "sleuth").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")):
        if f == ROOT / "sleuth" / "net.py":
            continue
        for m in pattern.finditer(f.read_text(encoding="utf-8")):
            if m.group(1) == "socket" and f in socket_ok:
                continue
            bad.append(f"{f.relative_to(ROOT)}: {m.group(0).strip()}")
    assert not bad, bad
