"""Record where and how a run happened, for reproducibility.

Every run saves this as env.json: the machine, library versions, the git
commit of this repo (and whether there were uncommitted changes), and
which hardware-probe file describes the machine. When a result looks odd
months later, this is how you find out what was different.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def runs_root(default: str = "runs") -> Path:
    """$SLEUTH_RUNS (e.g. a fast NVMe path) wins over the config value."""
    return Path(os.environ.get("SLEUTH_RUNS") or default)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def env_info() -> dict:
    import torch

    host = socket.gethostname()
    probe = REPO_ROOT / "hardware" / f"{host}.json"
    info = {
        "hostname": host,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available() else [],
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "hardware_probe": str(probe.relative_to(REPO_ROOT)) if probe.exists() else None,
    }
    return info
