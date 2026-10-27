"""The only module that opens outbound network connections (PLAN.md §1, §7).

Every request goes through `request_json`, which checks the URL against the
network policy first and logs it to runs/net.log (URL, size, status; never the
payload or any API key).

Default ("offline"), with no policy file: this machine and the local network:
loopback, private addresses (10.x, 172.16-31.x, 192.168.x, link-local) and host
names that resolve only to those. So `--server http://<gx10 address>:8000` reaches
a vLLM server on another machine at home, and nothing on the internet.

Optional policy, from the repo-local `sleuth.toml` (git-ignored) or
`~/.config/sleuth/settings.toml`:

    [network]
    mode = "offline"           # or "online": also allow online_hosts
    allow_lan = true           # false: only loopback and local_hosts
    local_hosts = ["gx10-1"]   # extra hosts to treat as local
    online_hosts = []          # mode = "online" only: internet hosts to allow

API keys come from environment variables only (`api_key_env`), never from files.
"""

from __future__ import annotations

import ipaddress
import json
import os
import socket
import time
import tomllib
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


class NetworkPolicyError(PermissionError):
    pass


def load_policy(repo_root: Path | None = None) -> dict:
    """The [network] table of sleuth.toml (repo) or ~/.config/sleuth/settings.toml."""
    for f in ((repo_root or Path.cwd()) / "sleuth.toml", Path.home() / ".config" / "sleuth" / "settings.toml"):
        if f.exists():
            with open(f, "rb") as fh:
                return tomllib.load(fh).get("network", {})
    return {}


def _on_lan(host: str) -> bool:
    """True if `host` is (or resolves only to) a private-network or loopback address:
    the GX10s / a llama.cpp box on the home network, never the internet."""
    try:
        addrs = {ipaddress.ip_address(host)}
    except ValueError:      # a host name: look it up, and require every address to be local
        try:
            addrs = {ipaddress.ip_address(a[4][0].split("%")[0]) for a in socket.getaddrinfo(host, None)}
        except (OSError, ValueError):
            return False
    return bool(addrs) and all(a.is_private or a.is_loopback or a.is_link_local for a in addrs)


def allowed(url: str, policy: dict | None = None) -> bool:
    policy = load_policy() if policy is None else policy
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        return False
    host = u.hostname.lower()
    if host in LOOPBACK or host in {h.lower() for h in policy.get("local_hosts", [])}:
        return True
    if policy.get("allow_lan", True) and _on_lan(host):
        return True
    if policy.get("mode", "offline") == "online":
        return host in {h.lower() for h in policy.get("online_hosts", [])}
    return False


def _log(line: str, log_file: Path | None) -> None:
    f = log_file or Path("runs") / "net.log"
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "a", encoding="utf-8") as fh:
            fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line + "\n")
    except OSError:
        pass  # logging must never break a request


def request_json(url: str, payload: dict | None = None, api_key_env: str | None = None, timeout: float = 600,
                 policy: dict | None = None, log_file: Path | None = None) -> dict:
    """POST (payload given) or GET a JSON API and return the decoded reply."""
    if not allowed(url, policy):
        raise NetworkPolicyError(
            f"{url}: not allowed by the network policy (offline mode allows this machine and the "
            "local network; internet hosts need mode = \"online\" in sleuth.toml)")
    headers = {"Content-Type": "application/json"}
    if api_key_env and os.environ.get(api_key_env):
        headers["Authorization"] = f"Bearer {os.environ[api_key_env]}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            _log(f"{req.method} {url} -> {r.status}, {len(data or b'')} B sent, {len(body)} B in {time.time() - t0:.1f}s",
                 log_file)
    except Exception as e:
        _log(f"{req.method} {url} -> error {type(e).__name__}", log_file)
        raise
    return json.loads(body)
