# Notes for Claude sessions working on Sleuth

Read `PLAN.md` (design + milestones) and `LOG.md` (decisions so far) first.

Hard rules from the owner:
- Fully local. Only `sleuth/net.py` (arrives in M4) may make *outbound*
  network connections, and only when the user's network policy allows it.
  No other module uses `urllib.request`, `http.client` or raw sockets for
  outbound traffic. (The dashboard's stdlib `http.server`, bound to
  127.0.0.1, is inbound-only and fine.)
- Allowed dependencies: Python stdlib, PyTorch, NumPy, pytest; plus `pyarrow`,
  approved (2026-10-23) only for reading Parquet datasets such as FineWeb-Edu, as an
  optional extra. **Ask before adding anything else**, with a justification. Never use `transformers`,
  `datasets`, `tokenizers`, `huggingface_hub`, `safetensors`, `gguf` or similar.
- Never download weights, datasets, or packages without asking. (TinyStories was
  approved as the next dataset; the owner downloads it on their machines.)
- The repo is public. Never commit personal or machine-identifying details: host
  names, usernames, home paths, LAN IPs, GPU UUIDs. Hardware probe JSON files stay
  local (git-ignored); cite anonymized numbers in LOG.md ("RTX 5060 Ti box", "GX10 #1").
- The owner's machines: two GX10s (Linux, aarch64, GB10), an RTX 5060 Ti box
  (**Windows 11**) and an RTX 3060 Ti box (OS not yet confirmed). Code must work on Windows too: no POSIX-only
  calls without a fallback, write test files with write_bytes when line endings
  matter, and no TOML strings built from Windows paths.
- Cloud sessions (no GPU): the owner approved installing PyTorch/NumPy/pytest
  for testing. download.pytorch.org may be blocked, so use PyPI:
  `python3 -m venv /home/user/venv && /home/user/venv/bin/pip install torch numpy pytest && /home/user/venv/bin/pip install -e .`
  then `/home/user/venv/bin/python -m pytest -q`.
- The owner is learning: prefer readable code, comment the *why*, and log
  decisions, results, and surprises in `LOG.md`.
- Every component gets tests (CPU-runnable with tiny configs; GPU-only tests
  marked `@pytest.mark.gpu`).
- Build in small milestones and stop for review at the end of each.
