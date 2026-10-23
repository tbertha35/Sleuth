"""Repository hygiene: every source file must actually be tracked by git.

This exists because `.gitignore` once had `data/`, which also matched the
package `sleuth/forge/data/`. Tests passed on the machine that wrote the code
but the files never reached GitHub, so every fresh clone was broken.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ["sleuth", "tests", "scripts", "configs", "suites", "docs"]


@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout")
def test_no_source_file_is_gitignored():
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--others", "--ignored", "--exclude-standard", "--", *SOURCE_DIRS],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    ignored = [f for f in out if "__pycache__" not in f and ".egg-info" not in f and not f.endswith(".pyc")]
    assert not ignored, f".gitignore hides source files from git: {ignored[:10]}"
