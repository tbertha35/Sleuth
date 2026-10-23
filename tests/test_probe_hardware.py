"""Tests for scripts/probe_hardware.py.

These exercise the parts that run anywhere (system info, disk, JSON output,
summary). The GPU sections need a real CUDA device and are covered by simply
running the probe on each machine.
"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("probe_hardware", ROOT / "scripts" / "probe_hardware.py")
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def test_system_info_has_basics():
    s = probe.probe_system()
    assert s["hostname"]
    assert s["cpu_logical_cores"] >= 1


def test_network_entries_have_no_raw_speed():
    for nic in probe.probe_network():
        assert "speed" not in nic  # converted to speed_gbit (or dropped)


def test_disk_probe_small_file(tmp_path):
    d = probe.probe_disk(tmp_path, size_gib=16 / 1024, random_seconds=0.4)
    assert d["seq_read_gbps"] > 0
    assert d["seq_write_gbps"] > 0
    assert set(d["random_4k"]) == {"qd1", "qd4", "qd16", "qd64"}
    # The probe must clean up its test file.
    assert not list(tmp_path.glob("sleuth_disk_probe_*"))


def test_disk_probe_skips_when_too_big(tmp_path):
    d = probe.probe_disk(tmp_path, size_gib=10**6, random_seconds=0.1)
    assert "skipped" in d


def test_main_writes_json_and_summary(tmp_path, capsys):
    rc = probe.main(["--out-dir", str(tmp_path), "--no-gpu", "--no-disk", "--quick", "--tag", "t"])
    assert rc == 0
    files = list(tmp_path.glob("*-t.json"))
    assert len(files) == 1
    report = json.loads(files[0].read_text())
    assert report["schema_version"] == probe.SCHEMA_VERSION
    assert report["disk"] == "skipped (--no-disk)"
    assert "saved" in capsys.readouterr().out


def test_expected_support_by_arch():
    assert probe._expected_support((8, 6)) == {"bf16": True, "tf32": True, "fp8": False, "fp4": False}  # 3060 Ti
    assert probe._expected_support((12, 0))["fp4"]  # 5060 Ti
    assert probe._expected_support((12, 1))["fp8"]  # GB10


def test_raw_file_reads_correct_bytes(tmp_path):
    """Both the uncached path (O_DIRECT here; FILE_FLAG_NO_BUFFERING on
    Windows) and the plain cached path must return the file's real bytes."""
    import mmap
    import os

    data = os.urandom(64 * 1024)
    p = tmp_path / "f.bin"
    p.write_bytes(data)
    for uncached in (True, False):
        f = probe._RawFile(str(p), uncached=uncached)
        buf = mmap.mmap(-1, 4096)
        try:
            assert f.read_into(buf, 8192) == 4096
            assert bytes(buf) == data[8192:12288]
            assert f.read_into(buf, len(data)) == 0  # end of file
        finally:
            f.close()
            buf.close()


def test_main_with_disk_path_and_private_paths(tmp_path):
    """--disk-path is a list of Paths; it must serialize, and the home
    directory must be shown as "~" in what gets written."""
    disk = tmp_path / "disk"
    rc = probe.main(["--out-dir", str(tmp_path), "--no-gpu", "--quick", "--tag", "d",
                     "--disk-path", str(disk), "--disk-size-gb", "0.01", "--random-seconds", "0.2"])
    assert rc == 0
    report = json.loads(next(tmp_path.glob("*-d.json")).read_text())
    assert isinstance(report["args"]["disk_path"], list)
    assert report["disk"]["seq_read_gbps"] > 0
    assert "uuid" not in json.dumps(report["torch"])


def test_private_path_hides_home():
    from pathlib import Path

    assert probe._private_path(Path.home() / "x" / "y.json").startswith("~")
