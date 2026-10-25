"""Stands in for `python -m sleuth` in the job-runner tests: it behaves like
the real commands in the ways the runner cares about, but instantly."""

import json
import os
import sys
import time
from pathlib import Path

cmd, args = sys.argv[1], sys.argv[2:]
print("argv:", json.dumps(args), flush=True)
if cmd == "train":
    run = Path(os.environ["SLEUTH_RUNS"]) / "fake-run"
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps({"name": "fake", "train": {"max_steps": 10}}))
    print(f"run dir: {run}", flush=True)
    print(f"to stop cleanly from anywhere: create the file {run / 'STOP'}", flush=True)
    for step in range(600):
        if (run / "STOP").exists():
            (run / "STOP").unlink()
            print(f"stopped at step {step} as requested", flush=True)
            sys.exit(0)
        print(f"step {step} | loss 1.0 | ünïcödé ✓", flush=True)
        time.sleep(0.05)
    sys.exit(0)
if cmd == "score":
    print("something went wrong", flush=True)
    sys.exit(3)
if cmd == "generate":  # runs until stopped by a signal
    while True:
        time.sleep(0.05)
time.sleep(0.3)
print("finished", flush=True)
