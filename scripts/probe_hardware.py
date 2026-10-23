#!/usr/bin/env python3
"""Sleuth hardware probe.

Measures the machine this runs on and saves the results to
``hardware/<hostname>.json`` (git-ignored: it stays on this machine) so that design decisions (batch sizes, dtypes,
Engram memory tiers, multi-node setup) can cite real numbers instead of
spec-sheet guesses.

What it reports:
  * system      - OS, CPU, RAM, Python / PyTorch / CUDA / NCCL versions
  * network     - interfaces and link speeds (matters for 2-node training)
  * gpu         - name, compute capability, memory, SM count, unified-memory flag
  * dtypes      - which of fp32/tf32/fp16/bf16/fp8/fp4 actually *run* here
  * compute     - measured matmul TFLOP/s per dtype
  * bandwidth   - GPU memory copy, host<->device (pinned and pageable), host RAM
  * attention   - which scaled_dot_product_attention backends work
  * disk        - sequential read/write and random 4 KiB read IOPS/latency
                  (the random-read numbers decide how big an SSD-backed
                  Engram table can be before it becomes the bottleneck)

Only the Python standard library is required. PyTorch is used if installed;
without it the GPU sections are skipped and say why.

Usage:
    python scripts/probe_hardware.py                     # full probe
    python scripts/probe_hardware.py --quick             # smaller, faster tests
    python scripts/probe_hardware.py --disk-path /mnt/nvme --disk-size-gb 8
    python scripts/probe_hardware.py --no-disk --no-gpu

Nothing here touches the network.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
MiB = 1024 * 1024
GiB = 1024 * MiB

# Error messages from failed kernels can be pages long; keep the JSON readable.
MAX_ERR_CHARS = 300


def _err(e: BaseException) -> str:
    msg = f"{type(e).__name__}: {e}"
    return msg if len(msg) <= MAX_ERR_CHARS else msg[:MAX_ERR_CHARS] + "..."


def _run(cmd: list[str]) -> str | None:
    """Run a command and return stdout, or None if it is missing or fails."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _private_path(p) -> str:
    """Path as text, with your home directory shown as "~" so a shared report
    doesn't include your username."""
    text = str(p)
    home = str(Path.home())
    return "~" + text[len(home):] if home and text.startswith(home) else text


def _jsonable(v):
    """Command-line values -> JSON-safe values (--disk-path gives a list of Paths)."""
    if isinstance(v, Path):
        return _private_path(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


# --------------------------------------------------------------------------- #
# System
# --------------------------------------------------------------------------- #

def _read_meminfo() -> dict[str, int]:
    """Parse /proc/meminfo into bytes. Linux only; empty dict elsewhere."""
    info: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, rest = line.split(":", 1)
                parts = rest.split()
                if parts and parts[0].isdigit():
                    mult = 1024 if len(parts) > 1 and parts[1] == "kB" else 1
                    info[key] = int(parts[0]) * mult
    except OSError:
        pass
    return info


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                # x86 says "model name"; many ARM kernels don't report a name at
                # all, so fall back to lscpu below.
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    lscpu = _run(["lscpu"])
    if lscpu:
        for line in lscpu.splitlines():
            if line.startswith("Model name:"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _ram_fallback() -> tuple[int, int | None]:
    """(total, available) bytes on systems without /proc/meminfo."""
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(st)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return st.ullTotalPhys, st.ullAvailPhys
        except (ImportError, AttributeError, OSError):
            pass
        return 0, None
    try:  # macOS and other POSIX systems
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"), None
    except (ValueError, OSError, AttributeError):
        return 0, None


def probe_system() -> dict:
    mem = _read_meminfo()
    if not mem:
        total, avail = _ram_fallback()
        mem = {"MemTotal": total, **({"MemAvailable": avail} if avail else {})}
    return {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "arch": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_logical_cores": os.cpu_count(),
        "ram_total_gib": round(mem.get("MemTotal", 0) / GiB, 2) or None,
        "ram_available_gib": round(mem.get("MemAvailable", 0) / GiB, 2) or None,
        "python": sys.version.split()[0],
    }


def probe_network() -> list[dict]:
    """List interfaces with link speed. 2-node NCCL training needs the fast one."""
    nics = []
    base = Path("/sys/class/net")
    if not base.exists():
        return nics
    for dev in sorted(base.iterdir()):
        if dev.name == "lo":
            continue
        entry: dict = {"name": dev.name}
        for field in ("speed", "mtu", "operstate"):
            try:
                val = (dev / field).read_text().strip()
                entry[field] = int(val) if val.lstrip("-").isdigit() else val
            except OSError:
                pass
        speed = entry.pop("speed", None)
        if isinstance(speed, int):
            # sysfs reports Mb/s; -1 means unknown / link down.
            entry["speed_gbit"] = speed / 1000 if speed > 0 else None
        entry["rdma_capable"] = (dev / "device" / "infiniband").exists()
        nics.append(entry)
    return nics


# --------------------------------------------------------------------------- #
# Host memory bandwidth (stdlib only)
# --------------------------------------------------------------------------- #

def probe_host_memcpy(size_mib: int, repeats: int) -> dict:
    """Single-threaded memcpy bandwidth using bytearray slice assignment.

    This is a lower bound on what the CPU can stream (one core, no SIMD
    tuning), but it is the number that matters for a host-RAM Engram tier
    fed from Python. Multi-threaded numbers come from the torch section.
    """
    n = size_mib * MiB
    src = bytearray(os.urandom(1 * MiB)) * size_mib
    dst = bytearray(n)
    dst[:] = src  # warm up: fault in all pages
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        dst[:] = src
        best = min(best, time.perf_counter() - t0)
    # A copy reads n bytes and writes n bytes.
    return {"size_mib": size_mib, "copy_gbps_read_plus_write": round(2 * n / best / 1e9, 2)}


# --------------------------------------------------------------------------- #
# GPU (PyTorch)
# --------------------------------------------------------------------------- #

def _sync(torch):
    torch.cuda.synchronize()


def _time_cuda(torch, fn, warmup: int, iters: int) -> float:
    """Median seconds per call, timed with CUDA events (host timers would
    measure kernel launch, not kernel execution)."""
    for _ in range(warmup):
        fn()
    _sync(torch)
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)
    times.sort()
    return times[len(times) // 2]


def _matmul_tflops(torch, fn, m: int, n: int, k: int, warmup: int, iters: int) -> float:
    secs = _time_cuda(torch, fn, warmup, iters)
    return round(2 * m * n * k / secs / 1e12, 2)


def probe_gpu_device(torch, idx: int) -> dict:
    p = torch.cuda.get_device_properties(idx)
    info = {
        "index": idx,
        "name": p.name,
        "compute_capability": f"{p.major}.{p.minor}",
        "sm_count": p.multi_processor_count,
        "total_memory_gib": round(p.total_memory / GiB, 2),
    }
    # Newer PyTorch exposes these; GB10 should report integrated / unified memory.
    # (The GPU's UUID is deliberately not recorded: it identifies your
    # specific card and isn't useful for any design decision.)
    for attr in ("is_integrated", "L2_cache_size"):
        if hasattr(p, attr):
            info[attr] = getattr(p, attr)
    try:
        free, total = torch.cuda.mem_get_info(idx)
        info["free_memory_gib"] = round(free / GiB, 2)
    except Exception as e:  # noqa: BLE001 - report, don't crash
        info["free_memory_gib"] = _err(e)
    return info


def _expected_support(cc: tuple[int, int]) -> dict:
    """What the architecture *should* support, per NVIDIA's generations.

    Tensor-core FP8 arrived with Ada/Hopper (sm_89/sm_90); FP4 with Blackwell
    (sm_100 datacenter, sm_12x consumer/GB10). We compare this against what
    actually runs, because a too-old PyTorch/CUDA build can lack kernels the
    hardware has.
    """
    major, minor = cc
    return {
        "bf16": major >= 8,
        "tf32": major >= 8,
        "fp8": (major, minor) >= (8, 9),
        "fp4": major >= 10,
    }


def probe_dtypes_and_compute(torch, idx: int, size: int, warmup: int, iters: int) -> dict:
    """Try each dtype for real and measure matmul throughput.

    Square matmuls of `size` are a best case: real model shapes are smaller
    and less efficient, so treat these as ceilings for MFU calculations.
    """
    dev = torch.device("cuda", idx)
    cc = torch.cuda.get_device_capability(idx)
    out: dict = {"matmul_size": size, "expected_by_arch": _expected_support(cc), "results": {}}
    res = out["results"]
    m = n = k = size

    def attempt(name, setup):
        try:
            fn = setup()
            res[name] = {"works": True, "tflops": _matmul_tflops(torch, fn, m, n, k, warmup, iters)}
        except Exception as e:  # noqa: BLE001
            res[name] = {"works": False, "error": _err(e)}
        finally:
            torch.cuda.empty_cache()

    def plain(dtype, tf32=False):
        def setup():
            torch.backends.cuda.matmul.allow_tf32 = tf32
            a = torch.randn(m, k, device=dev, dtype=dtype)
            b = torch.randn(k, n, device=dev, dtype=dtype)
            return lambda: a @ b
        return setup

    attempt("fp32", plain(torch.float32, tf32=False))
    attempt("tf32", plain(torch.float32, tf32=True))
    torch.backends.cuda.matmul.allow_tf32 = False
    attempt("fp16", plain(torch.float16))
    attempt("bf16", plain(torch.bfloat16))

    # FP8 goes through torch._scaled_mm (a private but stable-ish API). It
    # needs A row-major and B column-major, plus per-tensor float32 scales.
    def fp8_setup(fmt):
        def setup():
            if not hasattr(torch, fmt):
                raise RuntimeError(f"torch.{fmt} not available in this PyTorch build")
            dt = getattr(torch, fmt)
            a = torch.randn(m, k, device=dev).to(dt)
            b = torch.randn(n, k, device=dev).to(dt).t()  # column-major view
            one = torch.tensor(1.0, device=dev)
            return lambda: torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
        return setup

    attempt("fp8_e4m3", fp8_setup("float8_e4m3fn"))
    attempt("fp8_e5m2", fp8_setup("float8_e5m2"))

    # FP4 (NVFP4: e2m1 values, one fp8 scale per 16 elements). Support in
    # PyTorch is recent and its scale-layout requirements have changed between
    # versions, so a failure here may mean "PyTorch too old", not "no hardware
    # support" - compare with expected_by_arch and read the error.
    def fp4_setup():
        if not hasattr(torch, "float4_e2m1fn_x2"):
            raise RuntimeError("torch.float4_e2m1fn_x2 not available in this PyTorch build")
        # Two fp4 values are packed per byte; random bytes = random fp4 values.
        a = torch.randint(0, 256, (m, k // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        b = torch.randint(0, 256, (n, k // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2).t()

        def round_up(x, mult):
            return (x + mult - 1) // mult * mult

        # Block scales use a "swizzled" layout padded to 128 rows x 4 columns.
        # The values don't matter for timing, only the element count.
        sa = torch.ones(round_up(m, 128) * round_up(k // 16, 4), device=dev).to(torch.float8_e4m3fn)
        sb = torch.ones(round_up(n, 128) * round_up(k // 16, 4), device=dev).to(torch.float8_e4m3fn)
        return lambda: torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)

    attempt("fp4_nvfp4", fp4_setup)
    return out


def probe_gpu_bandwidth(torch, idx: int, size_mib: int, warmup: int, iters: int) -> dict:
    """GPU memory bandwidth and host<->device transfer speed.

    On GB10 the "device" and "host" are the same LPDDR5x pool, so expect
    H2D and D2D numbers to be much closer than on a PCIe card. That is the
    single most important fact for the Engram tier design.
    """
    dev = torch.device("cuda", idx)
    n = size_mib * MiB
    out: dict = {"size_mib": size_mib}

    try:
        x = torch.empty(n, dtype=torch.uint8, device=dev)
        y = torch.empty_like(x)
        secs = _time_cuda(torch, lambda: y.copy_(x), warmup, iters)
        out["device_copy_gbps_read_plus_write"] = round(2 * n / secs / 1e9, 1)
        del x, y
    except Exception as e:  # noqa: BLE001
        out["device_copy_gbps_read_plus_write"] = _err(e)

    for pinned in (True, False):
        tag = "pinned" if pinned else "pageable"
        try:
            h = torch.empty(n, dtype=torch.uint8, pin_memory=pinned)
            h.fill_(1)
            d = torch.empty(n, dtype=torch.uint8, device=dev)
            secs = _time_cuda(torch, lambda: d.copy_(h, non_blocking=pinned), warmup, iters)
            out[f"h2d_{tag}_gbps"] = round(n / secs / 1e9, 1)
            secs = _time_cuda(torch, lambda: h.copy_(d, non_blocking=pinned), warmup, iters)
            out[f"d2h_{tag}_gbps"] = round(n / secs / 1e9, 1)
            del h, d
        except Exception as e:  # noqa: BLE001
            out[f"h2d_{tag}_gbps"] = _err(e)
        torch.cuda.empty_cache()
    return out


def probe_attention_backends(torch, idx: int) -> dict:
    """Which fused attention kernels run here. Flash/efficient attention
    kernels are compiled per architecture, so a brand-new GPU can fall back
    to the slow 'math' path without telling you."""
    import torch.nn.functional as F

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError as e:
        return {"error": _err(e)}

    dev = torch.device("cuda", idx)
    q = torch.randn(4, 8, 1024, 64, device=dev, dtype=torch.bfloat16)
    results = {}
    for name in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "MATH"):
        backend = getattr(SDPBackend, name, None)
        if backend is None:
            results[name.lower()] = "not in this PyTorch"
            continue
        try:
            with sdpa_kernel(backend):
                F.scaled_dot_product_attention(q, q, q, is_causal=True)
            _sync(torch)
            results[name.lower()] = True
        except Exception as e:  # noqa: BLE001
            results[name.lower()] = _err(e)
    return results


def probe_torch_cpu_bandwidth(torch, size_mib: int, iters: int) -> dict:
    """Multi-threaded host RAM copy via torch (uses all cores)."""
    n = size_mib * MiB
    x = torch.empty(n, dtype=torch.uint8)
    x.fill_(1)
    y = torch.empty_like(x)
    y.copy_(x)
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        y.copy_(x)
        best = min(best, time.perf_counter() - t0)
    return {"threads": torch.get_num_threads(), "copy_gbps_read_plus_write": round(2 * n / best / 1e9, 1)}


def probe_torch(args) -> dict:
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "PyTorch not installed"}

    info: dict = {
        "available": True,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        info["cudnn"] = torch.backends.cudnn.version()
    except Exception:  # noqa: BLE001
        info["cudnn"] = None
    try:
        nccl = torch.cuda.nccl.version()
        info["nccl"] = ".".join(map(str, nccl)) if isinstance(nccl, tuple) else nccl
    except Exception:  # noqa: BLE001
        info["nccl"] = None
    try:
        import triton  # ships with PyTorch on most Linux builds; needed by torch.compile

        info["triton"] = triton.__version__
    except ImportError:
        info["triton"] = None
    if hasattr(torch.cuda, "get_arch_list"):
        # The SM architectures this PyTorch binary has kernels for. If the
        # GPU's compute capability is missing here, many ops will fail.
        info["compiled_arch_list"] = torch.cuda.get_arch_list()

    sizes = {"matmul": 4096 if args.quick else 8192, "copy_mib": 256 if args.quick else 1024}
    warmup, iters = (3, 10) if args.quick else (5, 30)

    try:
        info["host_bandwidth"] = probe_torch_cpu_bandwidth(torch, sizes["copy_mib"], 5)
    except Exception as e:  # noqa: BLE001
        info["host_bandwidth"] = _err(e)

    if args.no_gpu:
        info["gpus"] = "skipped (--no-gpu)"
        return info
    if not torch.cuda.is_available():
        info["gpus"] = "no CUDA device visible"
        return info

    gpus = []
    for idx in range(torch.cuda.device_count()):
        print(f"  probing GPU {idx}: {torch.cuda.get_device_name(idx)}", flush=True)
        g = probe_gpu_device(torch, idx)
        with torch.cuda.device(idx):
            for key, fn in (
                ("dtypes_and_compute", lambda: probe_dtypes_and_compute(torch, idx, sizes["matmul"], warmup, iters)),
                ("bandwidth", lambda: probe_gpu_bandwidth(torch, idx, sizes["copy_mib"], warmup, iters)),
                ("attention_backends", lambda: probe_attention_backends(torch, idx)),
            ):
                try:
                    g[key] = fn()
                except Exception as e:  # noqa: BLE001
                    g[key] = _err(e)
        gpus.append(g)
    info["gpus"] = gpus
    return info


def probe_nvidia_smi() -> dict | None:
    q = "name,driver_version,power.limit,clocks.max.sm,clocks.max.mem,pcie.link.gen.max,pcie.link.width.max"
    out = _run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"])
    if out is None:
        return None
    keys = q.split(",")
    return {"gpus": [dict(zip(keys, (v.strip() for v in line.split(",")))) for line in out.splitlines()]}


# --------------------------------------------------------------------------- #
# Disk
# --------------------------------------------------------------------------- #
#
# To measure the SSD rather than RAM, reads must bypass the operating
# system's file cache (right after we write the test file, all of it is
# still cached). Each OS spells this differently, so `_RawFile` hides it:
#   Linux    O_DIRECT, plus posix_fadvise(DONTNEED) as a fallback
#   Windows  CreateFileW(FILE_FLAG_NO_BUFFERING) via ctypes (stdlib)
#   macOS    fcntl F_NOCACHE
# Uncached reads need page-aligned buffers and offsets; mmap(-1, n) gives
# page-aligned memory, and all our offsets/sizes are multiples of 4 KiB.

_WIN_EOF = 38  # ERROR_HANDLE_EOF


class _RawFile:
    """A read-only file with positioned reads into a caller-owned buffer."""

    def __init__(self, path: str, uncached: bool = True):
        self.mode = "cached"
        self._fd = None
        self._handle = None
        if sys.platform == "win32":
            if uncached:
                self._open_windows(path)
            if self._handle is None:
                self._fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
            return
        flag = getattr(os, "O_DIRECT", 0)
        if uncached and flag:
            try:
                self._fd = os.open(path, os.O_RDONLY | flag)
                self.mode = "O_DIRECT"
                return
            except OSError:  # e.g. tmpfs and some network filesystems refuse O_DIRECT
                pass
        self._fd = os.open(path, os.O_RDONLY)
        if uncached and sys.platform == "darwin":
            try:
                import fcntl

                fcntl.fcntl(self._fd, fcntl.F_NOCACHE, 1)
                self.mode = "F_NOCACHE"
            except (ImportError, AttributeError, OSError):
                pass

    def _open_windows(self, path: str) -> None:
        try:
            import ctypes
            from ctypes import wintypes

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
            k32.CreateFileW.restype = wintypes.HANDLE

            class OVERLAPPED(ctypes.Structure):
                # Passing an OVERLAPPED with Offset set turns ReadFile into a
                # positioned read (like pread) on a normal synchronous handle.
                _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t),
                            ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD),
                            ("hEvent", wintypes.HANDLE)]

            k32.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                                     ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(OVERLAPPED)]
            k32.ReadFile.restype = wintypes.BOOL
            k32.CloseHandle.argtypes = [wintypes.HANDLE]
            GENERIC_READ, SHARE_RW, OPEN_EXISTING, NO_BUFFERING = 0x80000000, 0x1 | 0x2, 3, 0x20000000
            h = k32.CreateFileW(path, GENERIC_READ, SHARE_RW, None, OPEN_EXISTING, NO_BUFFERING, None)
            if h is None or h == ctypes.c_void_p(-1).value:
                return
            self._ctypes, self._k32, self._OVERLAPPED, self._DWORD = ctypes, k32, OVERLAPPED, wintypes.DWORD
            self._handle = h
            self.mode = "FILE_FLAG_NO_BUFFERING"
        except (ImportError, AttributeError, OSError):
            self._handle = None

    def read_into(self, buf: mmap.mmap, offset: int) -> int:
        """Fill `buf` from `offset`; returns bytes read (0 at end of file)."""
        if self._handle is not None:
            ct = self._ctypes
            ov = self._OVERLAPPED(0, 0, offset & 0xFFFFFFFF, offset >> 32, None)
            got = self._DWORD(0)
            # The temporary c_char view is released at the end of this
            # statement, so `buf` can still be closed later.
            addr = ct.addressof(ct.c_char.from_buffer(buf))
            if not self._k32.ReadFile(self._handle, addr, len(buf), ct.byref(got), ct.byref(ov)):
                err = ct.get_last_error()
                if err == _WIN_EOF:
                    return 0
                raise OSError(err, f"ReadFile failed (Windows error {err})")
            return got.value
        if hasattr(os, "preadv"):  # Linux, BSDs, recent macOS
            return os.preadv(self._fd, [buf], offset)
        data = os.pread(self._fd, len(buf), offset) if hasattr(os, "pread") else self._seek_read(len(buf), offset)
        buf[: len(data)] = data
        return len(data)

    def _seek_read(self, n: int, offset: int) -> bytes:
        os.lseek(self._fd, offset, os.SEEK_SET)
        return os.read(self._fd, n)

    def close(self) -> None:
        if self._handle is not None:
            self._k32.CloseHandle(self._handle)
            self._handle = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


def _drop_from_page_cache(fd: int) -> bool:
    """Linux fallback when O_DIRECT isn't allowed: ask the kernel to forget
    the file's cached pages. Works without root (unlike drop_caches)."""
    if not hasattr(os, "posix_fadvise"):
        return False
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return True
    except OSError:
        return False


def probe_disk(path: Path, size_gib: float, random_seconds: float) -> dict:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    size = int(size_gib * GiB)
    size -= size % MiB
    out: dict = {
        "path": _private_path(path.resolve()),
        "free_gib": round(usage.free / GiB, 1),
        "test_file_gib": round(size / GiB, 2),
    }
    if usage.free < size * 1.2:
        out["skipped"] = "not enough free space for the test file"
        return out

    block = 8 * MiB
    buf = mmap.mmap(-1, block)  # page-aligned, as uncached reads require
    buf.write(os.urandom(block))  # random data so compressing filesystems can't cheat

    fd_tmp, name = tempfile.mkstemp(prefix="sleuth_disk_probe_", dir=path)
    try:
        # --- sequential write (includes fsync, so it's real, not cached) ---
        t0 = time.perf_counter()
        written = 0
        while written < size:
            written += os.write(fd_tmp, buf)
        os.fsync(fd_tmp)
        out["seq_write_gbps"] = round(written / (time.perf_counter() - t0) / 1e9, 2)
        cache_dropped = _drop_from_page_cache(fd_tmp)
        os.close(fd_tmp)
        fd_tmp = -1

        # --- sequential read ---
        f = _RawFile(name)
        out["read_mode"] = f.mode if f.mode != "cached" else ("fadvise-dropped" if cache_dropped else "may-be-cached")
        try:
            t0 = time.perf_counter()
            total = 0
            while True:
                n = f.read_into(buf, total)
                if n <= 0:
                    break
                total += n
            out["seq_read_gbps"] = round(total / (time.perf_counter() - t0) / 1e9, 2)
        finally:
            f.close()

        # --- random 4 KiB reads: what an SSD-backed embedding table sees ---
        out["random_4k"] = _probe_random_reads(name, size, random_seconds)
    finally:
        if fd_tmp >= 0:
            os.close(fd_tmp)
        try:
            os.unlink(name)
        except OSError:
            pass
        buf.close()
    return out


def _probe_random_reads(name: str, size: int, seconds: float) -> dict:
    """Random aligned 4 KiB reads at several concurrency levels.

    NVMe drives only reach their rated IOPS with many requests in flight.
    The read calls release the GIL, so Python threads do give real
    parallelism here; this approximates what our async prefetcher will get.
    """
    page = 4096
    n_pages = size // page
    results = {}
    for threads in (1, 4, 16, 64):
        counts = [0] * threads
        lat_sum = [0.0] * threads
        errors: list[str] = []
        stop = time.perf_counter() + seconds / 4

        def worker(i):
            local = mmap.mmap(-1, page)
            rng = random.Random(i)
            f = None
            try:
                f = _RawFile(name)
                while time.perf_counter() < stop:
                    off = rng.randrange(n_pages) * page
                    t0 = time.perf_counter()
                    f.read_into(local, off)
                    lat_sum[i] += time.perf_counter() - t0
                    counts[i] += 1
            except Exception as e:  # noqa: BLE001 - surface it instead of silently reporting 0 IOPS
                errors.append(_err(e))
            finally:
                if f is not None:
                    f.close()
                local.close()

        ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
        t0 = time.perf_counter()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        elapsed = time.perf_counter() - t0
        total = sum(counts)
        if errors:
            results[f"qd{threads}"] = {"error": errors[0]}
            continue
        results[f"qd{threads}"] = {
            "iops": round(total / elapsed),
            "mib_per_s": round(total * page / elapsed / MiB, 1),
            "mean_latency_us": round(sum(lat_sum) / max(total, 1) * 1e6, 1),
        }
    return results


# --------------------------------------------------------------------------- #
# Summary + main
# --------------------------------------------------------------------------- #

def summarize(report: dict) -> str:
    s = report["system"]
    lines = [
        f"Host {s['hostname']} ({s['arch']}), {s['cpu_logical_cores']} cores, "
        f"{s['ram_total_gib']} GiB RAM, Python {s['python']}",
    ]
    t = report.get("torch", {})
    if not t.get("available"):
        lines.append(f"PyTorch: {t.get('reason', 'unavailable')}")
    else:
        lines.append(f"PyTorch {t['version']} (CUDA {t['cuda_build']}, NCCL {t.get('nccl')})")
        gpus = t.get("gpus")
        if isinstance(gpus, str):
            lines.append(f"GPUs: {gpus}")
        else:
            for g in gpus or []:
                lines.append(
                    f"GPU{g['index']}: {g['name']} sm_{g['compute_capability'].replace('.', '')}, "
                    f"{g['total_memory_gib']} GiB, {g['sm_count']} SMs, integrated={g.get('is_integrated', '?')}"
                )
                dc = g.get("dtypes_and_compute")
                if isinstance(dc, dict):
                    for name, r in dc["results"].items():
                        val = f"{r['tflops']} TFLOP/s" if r["works"] else "FAILED"
                        lines.append(f"    {name:10s} {val}")
                bw = g.get("bandwidth")
                if isinstance(bw, dict):
                    lines.append(
                        f"    mem copy {bw.get('device_copy_gbps_read_plus_write')} GB/s, "
                        f"H2D pinned {bw.get('h2d_pinned_gbps')} GB/s"
                    )
    disks = report.get("disk")
    for d in disks if isinstance(disks, list) else [disks]:
        if isinstance(d, dict) and "seq_read_gbps" in d:
            r = d.get("random_4k", {})
            lines.append(
                f"Disk {d['path']}: seq read {d['seq_read_gbps']} GB/s ({d['read_mode']}), "
                f"write {d['seq_write_gbps']} GB/s, 4K random "
                + ", ".join(f"{k}={v.get('iops', 'ERROR')} IOPS" for k, v in r.items())
            )
        elif isinstance(d, dict):
            lines.append(f"Disk {d.get('path')}: {d.get('skipped') or d.get('error')}")
        else:
            lines.append(f"Disk: {d}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "hardware")
    ap.add_argument("--disk-path", type=Path, action="append",
                    help="directory to benchmark (repeatable; default: the output dir)")
    ap.add_argument("--disk-size-gb", type=float, default=None,
                    help="test file size; make it larger than RAM for honest numbers if O_DIRECT is unavailable")
    ap.add_argument("--quick", action="store_true", help="smaller tests (~1 min)")
    ap.add_argument("--random-seconds", type=float, default=None,
                    help="total time for the random-read test (default 20, or 8 with --quick)")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--no-disk", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the output filename, e.g. 'after-driver-update'")
    args = ap.parse_args(argv)

    started = time.time()
    report: dict = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "args": {k: _jsonable(v) for k, v in vars(args).items()},
    }
    print("probing system...", flush=True)
    report["system"] = probe_system()
    report["network"] = probe_network()
    report["host_memcpy_single_thread"] = probe_host_memcpy(128 if args.quick else 512, 3)

    print("probing PyTorch / GPUs...", flush=True)
    report["torch"] = probe_torch(args)
    report["nvidia_smi"] = probe_nvidia_smi()

    if args.no_disk:
        report["disk"] = "skipped (--no-disk)"
    else:
        size = args.disk_size_gb if args.disk_size_gb is not None else (1.0 if args.quick else 4.0)
        disks = []
        for p in args.disk_path or [args.out_dir]:
            print(f"probing disk at {p} ({size} GiB test file)...", flush=True)
            try:
                secs = args.random_seconds if args.random_seconds is not None else (8.0 if args.quick else 20.0)
                disks.append(probe_disk(p, size, secs))
            except OSError as e:
                disks.append({"path": _private_path(p), "error": _err(e)})
        report["disk"] = disks[0] if len(disks) == 1 else disks

    report["probe_seconds"] = round(time.time() - started, 1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = report["system"]["hostname"] + (f"-{args.tag}" if args.tag else "")
    out_path = args.out_dir / f"{stem}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    print()
    print(summarize(report))
    print(f"\nsaved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
