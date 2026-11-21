"""Hardware profiling, footprint estimation, speed probes and auto-tuning.

    from . import profiler

    snap = profiler.snapshot()                       # RAM + VRAM, per-number source
    fp = profiler.estimate_model_footprint(arch, "q4", 8192)
    probe = profiler.probe_model_speed(base_url, model, cfg)
    tuned = profiler.tune_profile(cfg, snap, fp, probe)

Why this exists
---------------
Pedacito already sizes token budgets from ``vram_gb`` and the model's file
size. That estimate is deliberately conservative and only knows about the
Linux DRM sysfs counters. This module adds a *vendor-neutral* telemetry layer
that never changes the shape of :func:`servers.get_hardware_memory`:

* Linux keeps the existing ``/proc/meminfo`` + ``/sys/class/drm`` reader
  (amdgpu, Intel xe, and any other driver exposing ``mem_info_vram_*``).
* NVIDIA cards are read with ``nvidia-smi`` through an argv list, only when
  the binary resolves.
* macOS reads total RAM with ``sysctl -n hw.memsize`` and free pages with
  ``vm_stat``. Apple Silicon unified memory is counted as RAM; there is no
  separate VRAM claim.
* Windows reads ``kernel32.GlobalMemoryStatusEx`` through ``ctypes`` (the same
  API on x86-64 and ARM64).

Every reader is guarded by a platform check or ``shutil.which``, is time-boxed
to two seconds, and skips silently when absent. A missing reader is *unknown*,
never zero free memory.

The optional auto-tune path is off by default (``auto_tune = false``) and
only consumes the numbers above. It never changes a user-pinned value, never
raises concurrency for a tiered MoE server, and leaves hosted endpoints alone
apart from their profile timeout. The expensive speed probe runs only on
explicit request or once per (server, model) after a successful explicit
server load, on a daemon thread.

No third-party dependency is used here: stdlib + the existing ``requests``
client for the optional server metadata/probe paths.
"""

from __future__ import annotations

import copy
import ctypes
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import requests

from . import kvcache as kv_cache_mod
from . import servers

__all__ = [
    "CACHE_NAME", "PROBE_TTL", "GIB", "MEMORYSTATUSEX", "SNAPSHOT_TTL",
    "apply_auto_tune", "architecture_from_config", "build_report", "cache_path",
    "clear_cache", "clear_snapshot_cache", "detect_context_window",
    "estimate_model_footprint", "format_profile_report", "kv_bytes_per_token",
    "maybe_probe_async", "probe_cache_key", "probe_model_speed",
    "quant_bytes_per_param", "read_cached_probe", "reset",
    "resolve_architecture", "snapshot", "store_probe", "tune_profile",
]

_LOG = logging.getLogger("pedacito.profiler")

# The shared KV-cache model (also used by config.hardware_context_tokens).
kv_bytes_per_token = kv_cache_mod.kv_bytes_per_token

GIB = 1024 ** 3
READER_TIMEOUT = 2.0
METADATA_TIMEOUT = 2.0
PROBE_TTL = 7 * 24 * 3600.0       # a speed measurement is useful for a week
SNAPSHOT_TTL = 2.0                # burst memo; GUI polls are 15 s apart
CACHE_NAME = "profiler.json"
CACHE_VERSION = 1
HOSTED_PROBE_REFUSAL = "hosted model — refusing to probe"

# Bytes per parameter for the quantizations the footprint estimate accepts.
QUANT_BYTES = {
    "f16": 2.0, "fp16": 2.0, "bf16": 2.0,
    "f32": 4.0, "fp32": 4.0,
    "q8": 1.0, "q8_0": 1.0,
    "q6": 0.75, "q6_k": 0.75,
    "q5": 0.625, "q5_1": 0.625, "q5_0": 0.625,
    "q4": 0.5, "q4_0": 0.5, "q4_1": 0.5, "q4_k": 0.5, "q4_k_m": 0.5,
    "q2": 0.25, "q2_k": 0.25,
}
CONSERVATIVE_BYTES_PER_PARAM = 0.6

_PROBE_REPLY = "OK"
_PROBE_FILLER = (
    "The quick brown fox jumps over the lazy dog while the server measures "
    "prompt processing speed across a deliberately ordinary paragraph.\n")


# --- small helpers ------------------------------------------------------------

def _round(value, digits: int = 2) -> float:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return 0.0


def _float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _gb_from_bytes(value) -> float:
    return _round(_float(value) / GIB, 2)


def _cut(text: str, limit: int = 200) -> str:
    return str(text or "")[:limit]


def _run_argv(argv: list[str], timeout: float = READER_TIMEOUT) -> str:
    """Run an argv list with no shell and return stdout, or "" on any failure.

    This is the single subprocess seam: tests monkeypatch it with fake output
    so a Linux CI exercises the nvidia-smi and macOS readers too.
    """
    if not argv:
        return ""
    try:
        proc = subprocess.run(
            [str(part) for part in argv], shell=False, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=max(0.05, float(timeout or READER_TIMEOUT)))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        _LOG.debug("reader %s unavailable: %s", argv[0], _cut(exc))
        return ""
    if proc.returncode != 0:
        _LOG.debug("reader %s exited %s: %s", argv[0], proc.returncode,
                   _cut(proc.stderr))
        return ""
    return proc.stdout or ""


def _which(name: str) -> str | None:
    try:
        return shutil.which(name)
    except Exception:
        return None


# --- hardware snapshot --------------------------------------------------------

def _parse_nvidia_smi(text: str) -> list[dict]:
    """Parse ``index, memory.total, memory.used`` CSV rows (units: MiB)."""
    cards: list[dict] = []
    for line in str(text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        index = parts[0]
        try:
            total_mib = float(parts[1])
            used_mib = float(parts[2])
        except ValueError:
            continue
        if total_mib <= 0 and used_mib <= 0:
            continue
        total = total_mib * 1024 ** 2
        used = max(0.0, used_mib * 1024 ** 2)
        free = max(0.0, total - used)
        cards.append({
            "card": f"nvidia{index}",
            "name": f"NVIDIA GPU {index}",
            "index": _int(index, len(cards)),
            "total_gb": _gb_from_bytes(total),
            "used_gb": _gb_from_bytes(used),
            "free_gb": _gb_from_bytes(free),
            "percent": _round(used / total * 100.0, 1) if total else 0.0,
            "source": "nvidia-smi",
            "sources": {"total_gb": "nvidia-smi", "used_gb": "nvidia-smi",
                        "free_gb": "nvidia-smi (total - used)"},
        })
    return cards


def _read_nvidia_smi(run=None, which=None) -> list[dict]:
    find = which or _which
    exe = find("nvidia-smi")
    if not exe:
        return []
    out = (run or _run_argv)([
        exe, "--query-gpu=index,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ])
    try:
        return _parse_nvidia_smi(out)
    except Exception as exc:                 # pragma: no cover - defensive
        _LOG.debug("nvidia-smi parse failed: %s", _cut(exc))
        return []


_VM_STAT_LINE = re.compile(r"^([A-Za-z][A-Za-z _-]*):\s+(\d+)\.?\s*$")
_VM_PAGE_SIZE = re.compile(r"page size of\s+(\d+)\s+bytes", re.IGNORECASE)


def _parse_vm_stat(text: str) -> tuple[int, int | None]:
    """(page_size_bytes, free_bytes or None) from ``vm_stat`` output.

    macOS "available" memory is free + inactive + speculative pages; purgeable
    pages are only counted when the line exists (newer releases). The page
    size comes from the report header. None means the report was unreadable,
    which is *unknown* free memory, not zero.
    """
    page_size = 4096
    pages: dict[str, int] = {}
    for line in str(text or "").splitlines():
        m = _VM_PAGE_SIZE.search(line)
        if m:
            page_size = _int(m.group(1), page_size) or page_size
            continue
        m = _VM_STAT_LINE.match(line.strip())
        if m:
            pages[m.group(1).strip().lower()] = _int(m.group(2))
    if not pages:
        return page_size, None
    free_pages = (pages.get("pages free", 0)
                  + pages.get("pages inactive", 0)
                  + pages.get("pages speculative", 0)
                  + pages.get("pages purgeable", 0))
    return page_size, free_pages * page_size


def _read_macos_memory(run=None, which=None) -> dict | None:
    find = which or _which
    sysctl = find("sysctl")
    vm_stat = find("vm_stat")
    if not sysctl or not vm_stat:
        return None
    deadline = time.monotonic() + READER_TIMEOUT

    def left() -> float:
        return max(0.05, deadline - time.monotonic())

    total_text = (run or _run_argv)([sysctl, "-n", "hw.memsize"],
                                    timeout=left())
    try:
        total_bytes = int(str(total_text).strip())
    except ValueError:
        return None
    if total_bytes <= 0:
        return None
    stat_text = (run or _run_argv)([vm_stat], timeout=left())
    page_size, free_bytes = _parse_vm_stat(stat_text)
    if free_bytes is None:
        # vm_stat did not answer usable content: total is known, free is not.
        return {"total_bytes": total_bytes, "free_bytes": None,
                "page_size": page_size}
    return {
        "total_bytes": total_bytes,
        "free_bytes": min(max(0, free_bytes), total_bytes),
        "page_size": page_size,
    }


class MEMORYSTATUSEX(ctypes.Structure):
    """The Windows kernel32 structure, identical on x86-64 and ARM64."""

    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _read_windows_memory(kernel32=None) -> dict | None:
    """Read total/available physical memory through GlobalMemoryStatusEx."""
    if kernel32 is None:
        try:
            kernel32 = ctypes.windll.kernel32
        except (AttributeError, OSError):
            return None
    try:
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        total_bytes = int(status.ullTotalPhys)
        free_bytes = int(status.ullAvailPhys)
    except Exception as exc:                 # pragma: no cover - Windows-only
        _LOG.debug("GlobalMemoryStatusEx failed: %s", _cut(exc))
        return None
    if total_bytes <= 0:
        return None
    return {"total_bytes": total_bytes,
            "free_bytes": min(max(0, free_bytes), total_bytes)}


def _card_from_drm(card: dict) -> dict:
    """Copy one get_hardware_memory() card into the snapshot shape."""
    total = _float(card.get("total_gb"))
    used = _float(card.get("used_gb"))
    free = max(0.0, total - used)
    out = dict(card)
    out.update({
        "free_gb": _round(free),
        "source": "drm-sysfs",
        "sources": {"total_gb": "drm-sysfs", "used_gb": "drm-sysfs",
                    "free_gb": "drm-sysfs (total - used)"},
    })
    return out


def _source_summary(values: list[str]) -> str:
    unique = list(dict.fromkeys(v for v in values if v))
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    return "mixed (" + "+".join(unique) + ")"


# A short, thread-safe memo for burst callers: config.load, role derivation
# and the GUI payload can ask for the same hardware within a second, and each
# miss may spawn nvidia-smi. Caching an *unavailable* snapshot is safe: it is
# still "no reader", never zero memory, and expires quickly.
_snapshot_lock = threading.Lock()
_snapshot_cache: dict = {}


def clear_snapshot_cache() -> None:
    """Drop the short TTL snapshot memo (tests and explicit refreshes)."""
    with _snapshot_lock:
        _snapshot_cache.clear()


def _snapshot_cache_key(*, meminfo_path, drm_root, platform, which, run,
                        kernel32, include_nvidia, base_reader):
    key = (str(meminfo_path) if meminfo_path is not None else "",
           str(drm_root) if drm_root is not None else "",
           str(platform), bool(include_nvidia), os.name, which, run, kernel32,
           base_reader, _read_nvidia_smi, _read_macos_memory,
           _read_windows_memory)
    try:
        hash(key)
    except TypeError:                    # an unhashable injected callable
        return None
    return key


def snapshot(*, meminfo_path=None, drm_root=None, platform: str | None = None,
             which=None, run=None, kernel32=None,
             include_nvidia: bool = True, use_cache: bool = True) -> dict:
    """A JSON-safe hardware snapshot with a source for every number.

    Superset of :func:`servers.get_hardware_memory`: the legacy ``ram``,
    ``vram``, flat aliases and ``gpus`` keys keep their exact shape, so a GUI
    or TUI consumer can use it unchanged. Extra keys are ``sources`` (per
    number), per-card ``free_gb``/``source``, ``freshness`` and
    ``unified_memory``.

    Never raises and never requires a GPU: a missing reader contributes
    nothing and the overall ``available`` flag is False only when no reader
    returned a total. Results are memoized for ``SNAPSHOT_TTL`` seconds;
    ``use_cache=False`` recomputes, and :func:`clear_snapshot_cache` drops it.
    """
    platform = sys.platform if platform is None else str(platform)
    find = which or _which
    runner = run or _run_argv
    cache_key = _snapshot_cache_key(
        meminfo_path=meminfo_path, drm_root=drm_root, platform=platform,
        which=find, run=runner, kernel32=kernel32,
        include_nvidia=include_nvidia,
        base_reader=servers.get_hardware_memory)
    if use_cache and cache_key is not None:
        with _snapshot_lock:
            entry = _snapshot_cache.get(cache_key)
            if entry is not None and time.monotonic() - entry[0] < SNAPSHOT_TTL:
                return copy.deepcopy(entry[1])

    try:
        base = servers.get_hardware_memory(meminfo_path=meminfo_path,
                                           drm_root=drm_root)
    except Exception as exc:                 # pragma: no cover - defensive
        _LOG.debug("get_hardware_memory failed: %s", _cut(exc))
        base = {}

    ram_total = _float((base.get("ram") or {}).get("total_gb"))
    ram_free = _float((base.get("ram") or {}).get("available_gb"))
    if ram_free <= 0 and ram_total > 0:
        ram_free = max(0.0, ram_total
                       - _float((base.get("ram") or {}).get("used_gb")))
    ram_used = _float((base.get("ram") or {}).get("used_gb"))
    if ram_used <= 0 and ram_total > 0:
        ram_used = max(0.0, ram_total - ram_free)
    ram_source = "drm-sysfs" if ram_total > 0 else ""
    ram_free_source = ram_source
    unified = False
    readers: list[str] = []

    if ram_total > 0:
        readers.append("drm-sysfs")

    # macOS: sysctl + vm_stat. Apple Silicon unified memory is RAM.
    if platform == "darwin":
        mac = None
        try:
            mac = _read_macos_memory(run=runner, which=find)
        except Exception as exc:             # pragma: no cover - defensive
            _LOG.debug("macOS memory reader failed: %s", _cut(exc))
        if mac:
            ram_total = _gb_from_bytes(mac["total_bytes"])
            if mac.get("free_bytes") is None:
                ram_free, ram_used = 0.0, 0.0
                ram_free_source = ""
            else:
                ram_free = _gb_from_bytes(mac["free_bytes"])
                ram_used = max(0.0, _round(ram_total - ram_free))
                ram_free_source = "vm_stat"
            ram_source = "vm_stat"
            unified = True
            readers.append("vm_stat")

    # Windows: kernel32, no vendor-specific path.
    elif platform.startswith("win") or os.name == "nt":
        win = None
        try:
            win = _read_windows_memory(kernel32=kernel32)
        except Exception as exc:             # pragma: no cover - defensive
            _LOG.debug("Windows memory reader failed: %s", _cut(exc))
        if win:
            ram_total = _gb_from_bytes(win["total_bytes"])
            ram_free = _gb_from_bytes(win["free_bytes"])
            ram_used = max(0.0, _round(ram_total - ram_free))
            ram_source = "winapi"
            ram_free_source = "winapi"
            readers.append("winapi")

    cards: list[dict] = []
    base_vram = base.get("vram") or {}
    for card in base_vram.get("cards") or []:
        try:
            cards.append(_card_from_drm(card))
        except Exception:
            continue
    if not cards and _float(base_vram.get("total_gb")) > 0:
        # A consumer-supplied legacy snapshot may carry totals without cards;
        # keep those numbers instead of reporting zero VRAM.
        total = _float(base_vram.get("total_gb"))
        used = _float(base_vram.get("used_gb"))
        cards.append({
            "card": "gpu0", "total_gb": _round(total),
            "used_gb": _round(used), "free_gb": _round(max(0.0, total - used)),
            "percent": _round(used / total * 100.0, 1) if total else 0.0,
            "source": "drm-sysfs",
            "sources": {"total_gb": "drm-sysfs", "used_gb": "drm-sysfs",
                        "free_gb": "drm-sysfs (total - used)"},
        })

    if include_nvidia:
        try:
            nvidia = _read_nvidia_smi(run=runner, which=find)
        except Exception as exc:             # pragma: no cover - defensive
            _LOG.debug("nvidia-smi reader failed: %s", _cut(exc))
            nvidia = []
        for card in nvidia:
            if not any(c.get("card") == card.get("card") for c in cards):
                cards.append(card)
        if nvidia:
            readers.append("nvidia-smi")

    vram_total = _round(sum(_float(c.get("total_gb")) for c in cards))
    vram_used = _round(sum(_float(c.get("used_gb")) for c in cards))
    vram_free = _round(max(0.0, vram_total - vram_used))

    vram_sources = [str(c.get("source") or "") for c in cards]
    vram_total_source = _source_summary(vram_sources) if cards else ""
    vram_used_source = _source_summary(vram_sources) if cards else ""
    vram_free_source = _source_summary(
        [str((c.get("sources") or {}).get("free_gb") or c.get("source") or "")
         for c in cards]) if cards else ""

    sources = {
        "ram_total_gb": ram_source,
        "ram_free_gb": ram_free_source,
        "ram_used_gb": ram_source,
        "vram_total_gb": vram_total_source,
        "vram_used_gb": vram_used_source,
        "vram_free_gb": vram_free_source,
    }

    now = time.time()
    ram = {
        "total_gb": _round(ram_total),
        "used_gb": _round(ram_used),
        "free_gb": _round(ram_free),
        # Legacy alias: get_hardware_memory calls this "available_gb".
        "available_gb": _round(ram_free),
        "percent": _round(ram_used / ram_total * 100.0, 1) if ram_total else 0.0,
        "source": ram_source,
        "sources": {"total_gb": ram_source, "free_gb": ram_free_source,
                    "used_gb": ram_source},
    }
    vram = {
        "total_gb": vram_total,
        "used_gb": vram_used,
        "free_gb": vram_free,
        "percent": _round(vram_used / vram_total * 100.0, 1)
        if vram_total else 0.0,
        "source": vram_total_source,
        "sources": {"total_gb": vram_total_source,
                    "used_gb": vram_used_source,
                    "free_gb": vram_free_source},
        "cards": cards,
    }
    available = bool(ram_total > 0 or vram_total > 0)
    snap = {
        "ok": available,
        "available": available,
        # Legacy get_hardware_memory keys, unchanged in shape.
        "ram": ram,
        "vram": vram,
        "ram_total_gb": ram["total_gb"],
        "ram_used_gb": ram["used_gb"],
        "ram_percent": ram["percent"],
        "vram_total_gb": vram["total_gb"],
        "vram_used_gb": vram["used_gb"],
        "vram_percent": vram["percent"],
        "gpus": cards,
        # Profiler additions.
        "sources": sources,
        "platform": platform,
        "unified_memory": unified,
        "readers": list(dict.fromkeys(readers)),
        "time": now,
        "measured_at": now,
        "freshness": {"measured_at": now, "age_s": 0.0},
    }
    if cache_key is not None:
        with _snapshot_lock:
            _snapshot_cache[cache_key] = (time.monotonic(),
                                          copy.deepcopy(snap))
    return snap


# --- footprint estimation -----------------------------------------------------

def _normalise_quant(quant: str) -> str:
    raw = str(quant or "").strip().lower().replace("-", "_")
    raw = raw.replace(" ", "_")
    if not raw:
        return ""
    compact = raw.replace("_", "")
    for name in ("f16", "fp16", "bf16", "f32", "fp32", "q8", "q6",
                 "q5", "q4", "q2"):
        if compact.startswith(name) or f"_{name}" in raw or raw == name:
            return name
    # Common suffixes such as "Q4_K_M" / "4bit" / "8-bit".
    m = re.search(r"q(\d)", compact)
    if m:
        return f"q{m.group(1)}"
    m = re.search(r"(\d+)[-_ ]?bit", raw)
    if m:
        return f"q{m.group(1)}"
    return ""


def quant_bytes_per_param(quant: str) -> float:
    """Bytes per parameter for a quantization name; unknown -> 0.6 (safe)."""
    name = _normalise_quant(quant)
    return QUANT_BYTES.get(name, CONSERVATIVE_BYTES_PER_PARAM)


def _quant_from_name(name: str) -> str:
    m = re.search(r"(q\d(?:_[a-z0-9]+)*|f16|fp16|bf16|f32|fp32|"
                  r"\d+[-_ ]?bit)", str(name or "").lower())
    return _normalise_quant(m.group(1)) if m else ""


def _parse_params_from_name(name: str) -> tuple[float, float, bool]:
    """(total_b, active_b, is_moe) from a model id; (0, 0, False) if unknown."""
    text = str(name or "").lower()
    active = re.search(r"(?:^|[-_/ ])a(\d+(?:\.\d+)?)b(?:$|[-_/ .])", text)
    # 8x7b-style names: 56B total, but the active set is not in the name; the
    # caller lowers confidence. A 30B-A3B name carries both numbers.
    multi = re.search(r"(\d+)x(\d+(?:\.\d+)?)b", text)
    total = re.search(r"(?:^|[-_/ ])(\d+(?:\.\d+)?)b(?:$|[-_/ .])", text)
    is_moe = bool(active or multi or re.search(r"\bmoe\b|mixtral|a\d+b", text))
    if active:
        total_b = _float(total.group(1)) if total else 0.0
        return total_b, _float(active.group(1)), True
    if multi:
        return (_float(multi.group(1)) * _float(multi.group(2)),
                0.0, True)
    if total:
        return _float(total.group(1)), 0.0, is_moe
    return 0.0, 0.0, is_moe


def _pick(flat: dict, names: tuple[str, ...]):
    """First numeric value whose flattened key ends with one of `names`."""
    for name in names:
        for key, value in flat.items():
            if key == name or key.endswith("." + name):
                if isinstance(value, (int, float)) and not isinstance(value,
                                                                      bool):
                    return value
    return None


def _flatten_numbers(data, out: dict | None = None) -> dict:
    out = {} if out is None else out
    if isinstance(data, dict):
        for key, value in data.items():
            key_s = str(key).lower()
            if isinstance(value, dict):
                _flatten_numbers(value, out)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                out[key_s] = value
            elif isinstance(value, str):
                out[key_s] = value.lower()
    return out


def _metadata_from_payload(payload, names: tuple[str, ...] = ()) -> dict:
    """Normalise a llama.cpp/Ollama style mapping into footprint fields."""
    flat = _flatten_numbers(payload)
    meta: dict = {}

    def take(field: str, keys: tuple[str, ...], scale: float = 1.0):
        value = _pick(flat, keys)
        if value is not None:
            meta[field] = _float(value) * scale

    take("total_params_b", ("general.parameter_count", "parameter_count",
                            "n_params", "params", "parameters"))
    # A raw parameter count is usually > 1e6; a "30B" metadata field is not.
    if meta.get("total_params_b", 0) > 1e6:
        meta["total_params_b"] /= 1e9
    take("n_layers", ("block_count", "n_layers", "num_hidden_layers",
                      "layers"))
    take("n_heads", ("attention.head_count", "head_count",
                     "n_heads", "num_attention_heads"))
    take("n_kv_heads", ("attention.head_count_kv", "head_count_kv",
                        "n_kv_heads", "num_key_value_heads"))
    take("head_dim", ("attention.key_length", "key_length", "head_dim",
                      "n_embd_head"))
    take("embedding_length", ("embedding_length", "n_embd", "hidden_size"))
    take("n_experts", ("expert_count", "n_experts", "num_experts",
                       "num_local_experts"))
    take("n_experts_used", ("expert_used_count", "n_experts_used",
                            "num_experts_per_tok", "moe_top_k"))
    # Strings were flattened too (lower-case); recover the family/arch name.
    for key, value in flat.items():
        if key.endswith(("general.architecture", "architecture",
                         "model_type", "arch")) and isinstance(value, str):
            meta["architecture"] = value
            break
    if names:
        meta["name"] = names[0]
    return {k: v for k, v in meta.items() if v not in ("", None)}


def architecture_from_config(cfg) -> dict:
    """Explicit user architecture overrides from the active Config.

    Empty when the user set none. A partial override is valid: the missing
    fields lower the confidence instead of being guessed silently.
    """
    if cfg is None:
        return {}
    meta: dict = {"source": "config"}
    params = _float(getattr(cfg, "model_params_b", 0) or 0)
    active = _float(getattr(cfg, "model_active_params_b", 0) or 0)
    layers = _int(getattr(cfg, "model_layers", 0) or 0)
    kv_heads = _int(getattr(cfg, "model_kv_heads", 0) or 0)
    head_dim = _int(getattr(cfg, "model_head_dim", 0) or 0)
    quant = str(getattr(cfg, "model_quant", "") or "")
    arch = str(getattr(cfg, "model_arch", "") or "").strip().lower()
    if params > 0:
        meta["total_params_b"] = params
    if active > 0:
        meta["active_params_b"] = active
    if layers > 0:
        meta["n_layers"] = layers
    if kv_heads > 0:
        meta["n_kv_heads"] = kv_heads
    if head_dim > 0:
        meta["head_dim"] = head_dim
    if quant:
        meta["quant"] = quant
    if arch:
        meta["is_moe"] = arch in ("moe", "local-moe", "mixture")
    if len(meta) <= 1:
        return {}
    meta["name"] = str(getattr(cfg, "model", "") or "")
    if "quant" not in meta:
        inferred = _quant_from_name(meta["name"])
        if inferred:
            meta["quant"] = inferred
    return meta


def _server_roots(base_url: str) -> list[str]:
    base = str(base_url or "").rstrip("/")
    roots = [servers._root(base)]
    if base and base not in roots:
        roots.append(base)
    return [r for r in roots if r]


def _headers_for(cfg) -> dict:
    try:
        return servers._headers_for(cfg)
    except Exception:
        return {}


def _get_json(url: str, headers: dict, timeout: float = METADATA_TIMEOUT):
    try:
        resp = requests.get(url, headers=headers,
                            timeout=max(0.1, float(timeout)))
        if getattr(resp, "status_code", 0) != 200:
            return None
        return resp.json()
    except Exception:
        return None


def _metadata_from_llama(cfg) -> dict:
    """llama.cpp /props and /models metadata, when this endpoint offers it."""
    headers = _headers_for(cfg)
    model = str(getattr(cfg, "model", "") or "")
    deadline = time.monotonic() + METADATA_TIMEOUT

    def left() -> float:
        return max(0.0, deadline - time.monotonic())

    for root in _server_roots(getattr(cfg, "base_url", "")):
        if left() <= 0:
            return {}
        props = _get_json(root + "/props", headers, timeout=left())
        if isinstance(props, dict):
            meta = _metadata_from_payload(props, (model,))
            if _has_footprint_fields(meta):
                meta["source"] = "llama.cpp /props"
                return meta
    base = str(getattr(cfg, "base_url", "") or "").rstrip("/")
    candidates = [base + "/models", base + "/v1/models"]
    for root in _server_roots(getattr(cfg, "base_url", "")):
        candidates.append(root + "/models")
        candidates.append(root + "/v1/models")
        candidates.append(root + "/api/v0/models")
    for url in dict.fromkeys(candidates):
        if left() <= 0:
            return {}
        payload = _get_json(url, headers, timeout=left())
        entries = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_id = str(entry.get("id") or entry.get("model") or "")
            if model and entry_id and entry_id != model:
                continue
            meta = _metadata_from_payload(entry, (entry_id or model,))
            if _has_footprint_fields(meta):
                meta["source"] = "server /models"
                return meta
    return {}


def _metadata_from_ollama(cfg) -> dict:
    """Ollama /api/show: model_info carries block/head/expert counts."""
    roots = _server_roots(getattr(cfg, "base_url", ""))
    headers = _headers_for(cfg)
    model = str(getattr(cfg, "model", "") or "")
    if not model or not roots:
        return {}
    deadline = time.monotonic() + METADATA_TIMEOUT
    for root in roots:
        for payload in ({"model": model}, {"name": model}):
            left = deadline - time.monotonic()
            if left <= 0:
                return {}
            try:
                resp = requests.post(root + "/api/show", headers=headers,
                                     json=payload, timeout=left)
            except Exception:
                continue
            if getattr(resp, "status_code", 0) != 200:
                continue
            try:
                data = resp.json()
            except Exception:
                continue
            meta = _metadata_from_payload(data, (model,))
            if _has_footprint_fields(meta):
                meta["source"] = "ollama /api/show"
                return meta
    return {}


def _has_footprint_fields(meta: dict) -> bool:
    return bool(meta.get("total_params_b") or meta.get("n_layers")
                or meta.get("n_kv_heads") or meta.get("head_dim"))


def resolve_architecture(cfg, fetch: bool = True,
                         timeout: float = METADATA_TIMEOUT) -> dict:
    """Architecture metadata in priority order, with a provenance source.

    1. explicit `model_*` overrides in the config;
    2. llama.cpp /props and /models metadata;
    3. Ollama /api/show;
    4. model-name heuristics (size + family), plus the model file size when
       auto_fit already measured it.

    `fetch=False` is the config-load path: local only, no network calls. Any
    missing field is recorded in ``estimated_fields`` and lowers confidence
    later; nothing is presented as measured when it was guessed.
    """
    explicit = architecture_from_config(cfg)
    if explicit:
        return explicit
    local = False
    try:
        local = servers.is_local_url(str(getattr(cfg, "base_url", "") or ""))
    except Exception:
        local = False
    if cfg is not None and fetch and local:
        for reader in (_metadata_from_llama, _metadata_from_ollama):
            try:
                meta = reader(cfg)
            except Exception as exc:         # pragma: no cover - defensive
                _LOG.debug("metadata reader failed: %s", _cut(exc))
                meta = {}
            if meta:
                return meta
    model = str(getattr(cfg, "model", "") or "")
    total, active, is_moe = _parse_params_from_name(model)
    meta: dict = {"name": model, "source": "model name"}
    if total > 0:
        meta["total_params_b"] = total
    if active > 0:
        meta["active_params_b"] = active
    if is_moe:
        meta["is_moe"] = True
    quant = _quant_from_name(model)
    if quant:
        meta["quant"] = quant
    # A file size measured by auto_fit is a reasonable parameter proxy when
    # the name cannot be parsed. It is still an estimate, not metadata.
    size_bytes = _float(getattr(cfg, "model_size_bytes", 0) or 0)
    if size_bytes > 0 and not total:
        bpp = quant_bytes_per_param(quant)
        if bpp > 0:
            meta["total_params_b"] = max(0.1, size_bytes / (bpp * 1e9))
            meta["source"] = "model file size"
    if len(meta) <= 2 and not meta.get("total_params_b"):
        return {}
    return meta


def _guess_layers(params_b: float) -> int:
    if params_b <= 0:
        return 0
    if params_b <= 1.5:
        return 24
    if params_b <= 4:
        return 28
    if params_b <= 8:
        return 32
    if params_b <= 14:
        return 40
    if params_b <= 35:
        return 48
    if params_b <= 70:
        return 64
    return 80


def _guess_kv_heads(params_b: float) -> int:
    # Modern GQA checkpoints: 8 KV heads on 7-14B, 4-8 on larger models.
    if params_b <= 4:
        return 8
    if params_b <= 14:
        return 8
    if params_b <= 35:
        return 4
    return 8


def _guess_head_dim(params_b: float) -> int:
    return 128


def estimate_model_footprint(arch, quant: str, context_tokens: int, *,
                             cfg=None, model_size_bytes: int | None = None,
                             k_cache_quant: str = "",
                             v_cache_quant: str = "",
                             kv_cache_bytes_per_token: int = 0) -> dict:
    """Estimate weight residency and KV-cache size from architecture metadata.

    ``arch`` may be a metadata mapping from :func:`resolve_architecture`, a
    model id/name, or None. Returns (at least)::

        {
          "resident_gb": 7.0,             # all weights (RAM for an MoE)
          "vram_resident_gb": 7.0,        # active weights; == resident for dense
          "kv_bytes_per_token": 213623,
          "kv_gb_per_1k_tokens": 0.198929,
          "kv_gb": 1.6296,
          "confidence": 0.9,
          "arch": "dense" | "moe",
          "estimated_fields": [...]
        }

    KV sizing goes through the shared :mod:`pedacito.kvcache` helper. A known
    model file size (from `cfg.model_size_bytes`, `model_size_bytes`, or a
    config object) uses the same heuristic as `config.auto_fit`; explicit
    `k_cache_quant` / `v_cache_quant` / `kv_cache_bytes_per_token` knobs are
    honored. Without a file size it falls back to the architecture formula
    (2 * K+V elements * bytes), preserving the old profile numbers when no
    knobs are set. MoE total experts live in RAM; only the active experts are
    VRAM-resident. Missing fields are inferred conservatively and lower
    ``confidence`` instead of being reported as measured.
    """
    if cfg is None and arch is not None and hasattr(arch, "model_size_bytes"):
        # A Config passed as `arch` is also the source of the KV knobs and
        # the model file size.
        cfg = arch
    if cfg is not None:
        if model_size_bytes is None:
            model_size_bytes = _int(getattr(cfg, "model_size_bytes", 0))
        if not k_cache_quant:
            k_cache_quant = str(getattr(cfg, "k_cache_quant", "") or "")
        if not v_cache_quant:
            v_cache_quant = str(getattr(cfg, "v_cache_quant", "") or "")
        if not kv_cache_bytes_per_token:
            kv_cache_bytes_per_token = _int(
                getattr(cfg, "kv_cache_bytes_per_token", 0))
    context_tokens = max(0, _int(context_tokens))
    estimated: list[str] = []
    confidence = 1.0
    if isinstance(arch, str):
        total, active, is_moe = _parse_params_from_name(arch)
        meta: dict = {"name": arch, "source": "model name"}
        if total > 0:
            meta["total_params_b"] = total
        if active > 0:
            meta["active_params_b"] = active
        if is_moe:
            meta["is_moe"] = True
    elif isinstance(arch, dict):
        meta = dict(arch)
    elif arch is not None and hasattr(arch, "model_params_b"):
        meta = architecture_from_config(arch)
    else:
        meta = {}

    if not model_size_bytes:
        model_size_bytes = _int(meta.get("model_size_bytes") or 0)
    model_size_bytes = _int(model_size_bytes)

    source = str(meta.get("source") or "")
    if source == "config":
        confidence = 0.95
    elif source in ("llama.cpp /props", "ollama /api/show", "server /models"):
        confidence = 0.85
    elif source:
        confidence = 0.6

    total_params = _float(meta.get("total_params_b")
                          or meta.get("params_b")
                          or meta.get("params"))
    if total_params > 1e6:                  # a raw parameter count
        total_params /= 1e9
    if total_params <= 0:
        return {
            "resident_gb": 0.0, "vram_resident_gb": 0.0,
            "weights_gb": 0.0, "active_gb": 0.0,
            "kv_bytes_per_token": 0,
            "kv_gb_per_1k_tokens": 0.0, "kv_gb": 0.0,
            "context_tokens": context_tokens,
            "confidence": 0.0, "arch": "unknown", "quant": str(quant or ""),
            "bytes_per_param": quant_bytes_per_param(quant),
            "estimated_fields": ["total_params_b"],
            "source": source or "unknown",
        }

    active_params = _float(meta.get("active_params_b")
                           or meta.get("active_params"))
    if active_params > 1e6:
        active_params /= 1e9
    n_experts = _int(meta.get("n_experts"))
    n_experts_used = _int(meta.get("n_experts_used"))
    is_moe = bool(meta.get("is_moe")) or n_experts > 1 or (
        active_params > 0 and active_params < total_params)
    if is_moe and active_params <= 0 and n_experts > 1 and n_experts_used > 0:
        active_params = max(0.1, total_params * n_experts_used / n_experts)
        estimated.append("active_params_b")
        confidence -= 0.15
    if is_moe and active_params <= 0:
        active_params = max(0.1, total_params * 0.15)
        estimated.append("active_params_b")
        confidence -= 0.3

    quant_name = _normalise_quant(quant) or _normalise_quant(
        str(meta.get("quant") or ""))
    bpp = quant_bytes_per_param(quant_name)
    if not quant_name:
        estimated.append("quant")
        confidence -= 0.15

    layers = _int(meta.get("n_layers") or meta.get("layers"))
    if layers <= 0:
        layers = _guess_layers(total_params)
        if layers <= 0:
            layers = 32
        estimated.append("n_layers")
        confidence -= 0.15
    kv_heads = _int(meta.get("n_kv_heads") or meta.get("kv_heads"))
    if kv_heads <= 0:
        heads = _int(meta.get("n_heads") or meta.get("heads"))
        kv_heads = heads or _guess_kv_heads(total_params)
        if not heads:
            estimated.append("n_kv_heads")
            confidence -= 0.1
    head_dim = _int(meta.get("head_dim"))
    if head_dim <= 0:
        embedding = _int(meta.get("embedding_length"))
        heads = _int(meta.get("n_heads") or meta.get("heads"))
        if embedding > 0 and heads > 0:
            head_dim = max(1, embedding // heads)
        else:
            head_dim = _guess_head_dim(total_params)
            estimated.append("head_dim")
            confidence -= 0.1

    kv_bytes = kv_cache_mod.kv_bytes_per_token(
        kv_cache_bytes_per_token=kv_cache_bytes_per_token,
        model_size_bytes=model_size_bytes,
        n_layers=layers, n_kv_heads=kv_heads, head_dim=head_dim,
        weight_bytes_per_param=bpp,
        k_cache_quant=k_cache_quant, v_cache_quant=v_cache_quant)
    kv_gb_per_1k = kv_bytes * 1000.0 / GIB
    kv_gb = kv_gb_per_1k * context_tokens / 1000.0
    resident_gb = total_params * bpp
    active_gb = (active_params * bpp) if is_moe else resident_gb

    return {
        "resident_gb": _round(resident_gb),
        "vram_resident_gb": _round(active_gb),
        "weights_gb": _round(resident_gb),
        "active_gb": _round(active_gb),
        "kv_bytes_per_token": int(kv_bytes),
        "kv_gb_per_1k_tokens": _round(kv_gb_per_1k, 6),
        "kv_gb": _round(kv_gb, 4),
        "context_tokens": context_tokens,
        "confidence": _round(max(0.0, min(1.0, confidence)), 2),
        "arch": "moe" if is_moe else "dense",
        "quant": quant_name or "unknown",
        "bytes_per_param": bpp,
        "total_params_b": _round(total_params, 2),
        "active_params_b": _round(active_params if is_moe else total_params, 2),
        "n_layers": layers, "n_kv_heads": kv_heads, "head_dim": head_dim,
        "estimated_fields": estimated,
        "source": source or "unknown",
    }


# --- speed probe --------------------------------------------------------------

def cache_path() -> Path:
    """The user-profile measurement store, beside the user config."""
    from . import config as config_mod
    return config_mod.USER_CONFIG.parent / CACHE_NAME


def _read_store() -> dict:
    try:
        data = json.loads(cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": CACHE_VERSION, "probes": {}}
    if not isinstance(data, dict):
        return {"version": CACHE_VERSION, "probes": {}}
    probes = data.get("probes")
    if not isinstance(probes, dict):
        probes = {}
    data["version"] = CACHE_VERSION
    data["probes"] = probes
    return data


def _write_store(data: dict) -> Path:
    path = cache_path()
    tmp = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # A unique temp name (fileio.write_bytes pattern): probe threads for
        # different servers finish concurrently and must not share one
        # `.tmp` file.
        fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                   dir=str(path.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, indent=1, sort_keys=True))
        os.replace(tmp, path)
    except OSError:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    return path


def probe_cache_key(base_url: str, model: str) -> str:
    return f"{servers.normalize_url(base_url)}|{str(model or '')}"


def read_cached_probe(base_url: str, model: str,
                      now: float | None = None,
                      ttl: float = PROBE_TTL) -> dict | None:
    """A fresh cached probe for this exact (server, model), else None.

    The key includes the model id, so switching models never returns the
    previous model's measurement. TTL is a full week.
    """
    now = time.time() if now is None else float(now)
    entry = _read_store()["probes"].get(probe_cache_key(base_url, model))
    if not isinstance(entry, dict):
        return None
    if str(entry.get("model") or "") != str(model or ""):
        return None
    ts = _float(entry.get("measured_at") or entry.get("ts"))
    if ts <= 0 or ts > now + 60 or now - ts >= max(0.0, float(ttl)):
        return None
    out = dict(entry)
    out["cached"] = True
    out["age_s"] = _round(max(0.0, now - ts), 1)
    return out


def store_probe(result: dict) -> Path:
    data = _read_store()
    entry = dict(result)
    entry.setdefault("measured_at", time.time())
    entry["model"] = str(entry.get("model") or "")
    entry["base_url"] = str(entry.get("base_url") or "")
    data["probes"][probe_cache_key(entry["base_url"], entry["model"])] = entry
    return _write_store(data)


def clear_cache() -> Path:
    """Drop every cached measurement (and any tuning notes)."""
    path = cache_path()
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
    return path


def reset() -> Path:
    """Public alias for `pdc profile --reset` and the TUI command."""
    return clear_cache()


def detect_context_window(cfg, timeout: float = METADATA_TIMEOUT) -> int:
    """Best-effort server-reported context window; 0 when unavailable.

    Uses the existing LLMClient.model_info() probe with a short, local
    timeout so `pdc profile --show` never inherits a 600 s generation timeout.
    """
    if cfg is None:
        return 0
    try:
        from .config import is_hosted
        if is_hosted(cfg):
            return 0
    except Exception:
        pass
    try:
        from .llm import LLMClient
        probe_cfg = copy.copy(cfg)
        # Private provenance: LLMClient may apply local-moe defaults, and
        # those source writes must not relabel the caller's live config.
        probe_cfg.sources = dict(getattr(cfg, "sources", {}) or {})
        probe_cfg.timeout = max(0.1, float(timeout or METADATA_TIMEOUT))
        info = LLMClient(probe_cfg).model_info() or {}
        return max(0, _int(info.get("context_length")))
    except Exception:
        return 0


def _probe_prompt(target_tokens: int) -> str:
    unit = _PROBE_FILLER
    chars = max(1, int(max(64, target_tokens) * 3.6))
    repeats = max(1, chars // max(1, len(unit)))
    return (unit * repeats).rstrip() + "\n\nReply with just: OK"


def _filler_target(window: int) -> int:
    if window <= 0:
        return 512
    if window >= 4096:
        return 2048
    if window >= 1024:
        return 512
    return max(64, min(512, window // 2))


def probe_model_speed(base_url: str, model: str, config,
                      *, force: bool = False, context_window: int | None = None,
                      timeout: float | None = None) -> dict:
    """One tiny completion plus one prefill measurement.

    Runs only when the caller asked (``pdc profile --run`` / the TUI) or from
    :func:`maybe_probe_async` after a successful load. A hosted endpoint is
    refused before any client is built, whether the caller forced the probe
    or not. Every exception is swallowed: a dead server yields
    ``{"ok": False, ...}`` and nothing is cached. On success the result is
    stored in the user profile store with a timestamp and returned.
    """
    base = str(base_url or getattr(config, "base_url", "") or "")
    mid = str(model or getattr(config, "model", "") or "")
    if base and not servers.is_local_url(base):
        return {"ok": False, "refused": True, "base_url": base, "model": mid,
                "detail": HOSTED_PROBE_REFUSAL, "measured_at": time.time()}
    if not force:
        cached = read_cached_probe(base_url, model)
        if cached is not None:
            return cached
    started = time.time()
    try:
        from .config import Config
        from .llm import LLMClient, one_shot_completion, timing_record
    except Exception as exc:                 # pragma: no cover - defensive
        return {"ok": False, "base_url": str(base_url), "model": str(model),
                "detail": f"{type(exc).__name__}: {_cut(exc)}",
                "measured_at": started}

    try:
        cfg = copy.copy(config) if config is not None else Config()
        # copy.copy shares the sources dict: LLMClient construction applies
        # local-moe defaults, whose provenance writes ("local-moe default")
        # must land on the probe's private copy, never relabel the caller's
        # explicit settings (a soft source invites later auto-tune shrink).
        cfg.sources = dict(getattr(cfg, "sources", {}) or {})
        cfg.base_url = str(base_url or getattr(cfg, "base_url", ""))
        cfg.model = str(model or getattr(cfg, "model", ""))
    except Exception as exc:                 # pragma: no cover - defensive
        return {"ok": False, "base_url": str(base_url), "model": str(model),
                "detail": f"{type(exc).__name__}: {_cut(exc)}",
                "measured_at": started}
    # The probe must not wait on a long generation timeout, but a slow local
    # prefill still needs real room. 10..300 s is enough for a 2k prompt.
    probe_timeout = _float(timeout if timeout is not None
                           else getattr(cfg, "timeout", 60) or 60.0)
    cfg.timeout = min(max(probe_timeout, 10.0), 300.0)
    # A probe is allowed exactly one attempt: no "model still loading" wait,
    # no long retry loop. Any HTTP failure is a silent no-result.
    cfg.load_wait_seconds = 0
    # A probe is not part of the user's task budget and should not attach
    # speculative-decoding/prompt-cache options (or print downgrade notices).
    try:
        cfg.max_total_tokens = 0
        cfg.max_cost = 0.0
        cfg.enable_prompt_cache = False
        cfg.enable_prompt_lookup = False
    except Exception:
        pass

    window = _int(context_window) if context_window is not None else 0
    if window <= 0:
        window = detect_context_window(cfg)
    target = _filler_target(window or _int(getattr(cfg, "context_tokens", 0)))
    prompt = _probe_prompt(target)

    try:
        client = LLMClient(cfg)
        # Construction applies local-moe defaults, which raise a short
        # timeout to 600 s and would undo the probe's bounded wait; re-cap
        # it here (requests read cfg.timeout per call).
        cfg.timeout = min(max(probe_timeout, 10.0), 300.0)
        client.echo_stream = False
        client.site = "probe"
        # Keep the probe off the stdout stream.
        try:
            client.stream_callback = lambda _text: None
        except Exception:
            pass
        text = one_shot_completion(
            client,
            [{"role": "user", "content": prompt}],
            max_tokens=16, stream=True, disable_thinking=True)
    except Exception as exc:
        _LOG.info("speed probe failed for %s: %s", model, _cut(exc))
        return {"ok": False, "base_url": str(cfg.base_url),
                "model": str(cfg.model), "detail":
                f"{type(exc).__name__}: {_cut(exc)}",
                "measured_at": started}

    timing = timing_record(client)
    timings = dict(timing.last_timings or {})
    elapsed = float(timing.elapsed_s or 0.0)
    prompt_tokens = int(timing.prompt_tokens or 0)
    if prompt_tokens <= 0:
        prompt_tokens = int(timing.estimate_tokens(prompt))
    completion_tokens = int(timing.completion_tokens or 0)
    if completion_tokens <= 0:
        completion_tokens = int(timing.estimate_tokens(text or _PROBE_REPLY))

    prefill_tps = _float(timings.get("prompt_per_second")
                         or timings.get("prompt_tokens_per_second"))
    generation_tps = _float(timings.get("predicted_per_second")
                            or timings.get("predicted_tokens_per_second"))
    prefill_s = max(0.0, float(timing.prefill_s or 0.0))
    if not prefill_tps and prefill_s > 0:
        prefill_tps = prompt_tokens / prefill_s
    if not generation_tps:
        gen_s = max(0.001, elapsed - prefill_s) if elapsed > 0 else 0.0
        if gen_s > 0 and completion_tokens > 0:
            generation_tps = completion_tokens / gen_s
        else:
            generation_tps = float(timing.stream_tok_per_s or 0.0)

    result = {
        "ok": True,
        "base_url": str(cfg.base_url),
        "model": str(cfg.model),
        "prefill_tok_per_s": _round(max(0.0, prefill_tps), 1),
        "generation_tok_per_s": _round(max(0.0, generation_tps), 1),
        # Short aliases for reports and scripts.
        "prefill_tok_s": _round(max(0.0, prefill_tps), 1),
        "gen_tok_s": _round(max(0.0, generation_tps), 1),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "probe_tokens": target,
        "context_window": window,
        "elapsed_s": _round(elapsed, 3),
        "measured_at": time.time(),
        "server_timings": bool(timings),
        "cached": False,
    }
    try:
        store_probe(result)
    except Exception as exc:                 # pragma: no cover - defensive
        _LOG.info("could not cache speed probe: %s", _cut(exc))
    return result


_probe_lock = threading.Lock()
_probe_running: set[str] = set()


def maybe_probe_async(config, *, base_url: str = "", model: str = "",
                      force: bool = False):
    """Start the once-per-(server, model) probe on a daemon thread.

    Returns the thread, or None when the endpoint is hosted, a fresh cache
    entry exists, `profile_probe_on_load` is false, or a probe for this pair
    is already running. Never blocks the caller and never surfaces an
    exception.
    """
    if not bool(getattr(config, "profile_probe_on_load", True)):
        return None
    base = str(base_url or getattr(config, "base_url", "") or "")
    mid = str(model or getattr(config, "model", "") or "")
    if not base or not mid:
        return None
    # A speed probe is meaningful for loopback and private-LAN servers; a
    # cloud endpoint is never probed automatically.
    if not servers.is_local_url(base):
        return None
    key = probe_cache_key(base, mid)
    if not force and read_cached_probe(base, mid) is not None:
        return None
    with _probe_lock:
        if key in _probe_running:
            return None
        _probe_running.add(key)

    def worker():
        try:
            probe_model_speed(base, mid, config, force=force)
        except Exception as exc:             # pragma: no cover - defensive
            _LOG.info("background speed probe failed: %s", _cut(exc))
        finally:
            with _probe_lock:
                _probe_running.discard(key)

    thread = threading.Thread(target=worker, name="pdc-speed-probe",
                              daemon=True)
    thread.start()
    return thread


# --- auto-tuning --------------------------------------------------------------

# Fields whose value controls the KV-cache geometry. When any of these was
# inferred (not measured), a low-confidence footprint is a guess, and the
# tuner should propose rather than apply aggressive shrink.
_KV_GUESSED_FIELDS = frozenset(("quant", "n_layers", "n_kv_heads", "head_dim"))


def _is_explicit(cfg, key: str) -> bool:
    # Auto-tune's own writes are deliberately soft here: a later pass may
    # refine them. `config._is_explicit_limit` treats the same `auto-tune ...`
    # source as explicit, so auto_fit leaves the tuned window alone. This
    # one-way pinning is covered by tests and explained in docs/CONFIG.md.
    source = str((getattr(cfg, "sources", None) or {}).get(key, "default")
                 or "default")
    if source in ("", "default") or "preset" in source:
        return False
    return not source.startswith(("auto-fit", "auto-tune", "auto-detected",
                                  "local-moe"))


def _is_moe(cfg) -> bool:
    try:
        from .config import is_local_moe
        return bool(is_local_moe(cfg))
    except Exception:
        return False


def _floor_context(current: int) -> int:
    current = max(0, _int(current))
    if current <= 0:
        return 4096
    return min(current, 4096)


def _hardware_context_cap(cfg) -> int:
    """`config.hardware_context_tokens` for this cfg, or 0 when the model
    file size is unknown. Keeping this cap in the same path as auto_fit means
    a cache-quantization hint can never propose growth that path refuses."""
    try:
        from .config import hardware_context_tokens
        model_bytes = _int(getattr(cfg, "model_size_bytes", 0) or 0)
        if model_bytes <= 0:
            return 0
        return int(hardware_context_tokens(
            _int(getattr(cfg, "vram_gb", 0) or 0), model_bytes,
            str(getattr(cfg, "k_cache_quant", "") or ""),
            str(getattr(cfg, "v_cache_quant", "") or ""),
            _int(getattr(cfg, "kv_cache_bytes_per_token", 0) or 0)))
    except Exception:
        return 0


def _free_gb(block: dict) -> float | None:
    """Free memory from a snapshot block, deriving it from totals when absent.

    None means *unknown* (no reader), never zero.
    """
    if not isinstance(block, dict):
        return None
    value = block.get("free_gb")
    if value is not None:
        return _float(value)
    total = _float(block.get("total_gb"))
    if total <= 0:
        return None
    used = block.get("used_gb")
    if used is None:
        used = block.get("available_gb")
        if used is not None:
            return max(0.0, _float(used))
        return None
    return max(0.0, total - _float(used))


def _vram_tuning_basis(vram: dict) -> dict | None:
    """The largest card to tune against when a snapshot's cards differ in size.

    ``vram.total_gb`` remains the summed system inventory (the GUI badge and
    the ``pdc profile`` header keep showing it), but one model cannot spend an
    iGPU's 0.5 GB on top of a discrete card's 24 GB. Headroom therefore comes
    from the largest card's own total and free numbers.

    Single-card snapshots, homogeneous multi-card snapshots (where a
    multi-GPU server may genuinely pool equally sized cards), and a card whose
    free count is unknown keep the summed path exactly as before. Unknown free
    memory is never treated as zero.
    """
    cards = [card for card in (vram.get("cards") or [])
             if isinstance(card, dict)]
    sized = [(_float(card.get("total_gb")), card) for card in cards]
    sized = [(total, card) for total, card in sized if total > 0]
    if len(sized) < 2:
        return None
    largest = max(total for total, _card in sized)
    # Within rounding noise the cards are the same size class; keep pooling.
    if all(largest - total < 0.1 for total, _card in sized):
        return None
    card = next(card for total, card in sized if total == largest)
    free = _free_gb(card)
    if free is None:
        return None
    return {
        "card": str(card.get("card") or "gpu"),
        "total_gb": _round(largest),
        "free_gb": _round(free),
        "source": str((card.get("sources") or {}).get("free_gb")
                      or card.get("source") or ""),
        "ignored": [str(other.get("card") or "gpu")
                    for _total, other in sized if other is not card],
    }


def tune_profile(cfg, snapshot: dict | None, footprint: dict | None,
                 probe: dict | None, *, server_window: int | None = None) -> dict:
    """Compute an auto-tuned profile without mutating `cfg`.

    Returns the current/tuned values, a per-field ``sources`` mapping (same
    provenance convention as the local-moe block), notes, and any growth
    proposal. Callers that want the values applied use :func:`apply_auto_tune`.

    Vendor neutrality: the decision only reads numbers and their source tags.
    With no VRAM reader the profile is left untouched unless the platform
    reports unified memory (macOS), where RAM headroom is the correct pool. A
    missing reader is never interpreted as zero free memory.
    """
    if cfg is None:
        return {"ok": False, "applied": False, "reason": "no config",
                "values": {}, "config": {}, "sources": {}, "notes": []}
    fields = ("context_tokens", "max_gathered_chars", "flat_map_token_budget",
              "timeout", "max_concurrency")
    values = {key: getattr(cfg, key, 0) for key in fields}
    sources: dict[str, str] = {}
    notes: list[str] = []
    applied = False
    hosted = False
    try:
        from .config import is_hosted
        hosted = bool(is_hosted(cfg))
    except Exception:
        hosted = False

    if hosted:
        # A hosted cloud profile never gets hardware-derived tuning. The one
        # static rule is the shorter default timeout; an explicit user value
        # always wins.
        if not _is_explicit(cfg, "timeout"):
            current = _float(values["timeout"])
            if current > 30.0:
                values["timeout"] = 30.0
                sources["timeout"] = (
                    "auto-tune (hosted cloud default 30 s; set timeout "
                    "explicitly to override)")
                applied = True
                notes.append("timeout: hosted profile -> 30 s")
        return {
            "ok": True, "applied": applied, "reason": "hosted",
            "values": values, "config": dict(values), "sources": sources,
            "notes": notes,
            **values,
        }

    # Fully pinned: every tunable field has an explicit user origin.
    if all(_is_explicit(cfg, key) for key in fields):
        return {
            "ok": True, "applied": False, "reason": "user-pinned",
            "values": values, "config": dict(values), "sources": {}, "notes":
            ["all tunable fields are explicit; nothing was changed"],
            **values,
        }

    snapshot = snapshot if isinstance(snapshot, dict) else {}
    ram = snapshot.get("ram") or {}
    vram = snapshot.get("vram") or {}
    sources_map = snapshot.get("sources") or {}
    unified = bool(snapshot.get("unified_memory"))
    vram_total = _float(vram.get("total_gb"))
    vram_source = str(sources_map.get("vram_total_gb")
                      or vram.get("source") or "")
    ram_total = _float(ram.get("total_gb"))
    ram_source = str(sources_map.get("ram_total_gb")
                     or ram.get("source") or "")
    ram_free_source = str(
        sources_map.get("ram_free_gb")
        or (ram.get("sources") or {}).get("free_gb") or "")
    unified_free_known = bool(unified and ram_free_source)
    vram_free = _free_gb(vram)
    ram_free = _free_gb(ram)

    # Display totals stay summed; headroom is per-card. An iGPU's 0.5 GB does
    # not add to a 24 GB dGPU's free pool for one model.
    basis = _vram_tuning_basis(vram)
    vram_basis = None
    if basis:
        vram_total = _float(basis.get("total_gb"))
        vram_free = _float(basis.get("free_gb"))
        vram_source = str(basis.get("source") or vram_source)

    if vram_total > 0 and vram_source and vram_free is not None:
        vram_basis = basis
        free_gb = vram_free
        pool = "VRAM"
        pool_source = str(sources_map.get("vram_free_gb") or vram_source)
        if vram_basis:
            pool_source = str(vram_basis.get("source") or pool_source)
            notes.append(
                f"tuned against {vram_basis['card']} "
                f"({vram_basis['total_gb']:.1f} GB); "
                f"{', '.join(vram_basis['ignored'])} ignored "
                "(heterogeneous cards are not pooled)")
        weights = _float((footprint or {}).get("vram_resident_gb")
                         or (footprint or {}).get("active_gb")
                         or (footprint or {}).get("resident_gb"))
        reserve = max(0.5, free_gb * 0.10)
    elif (unified_free_known and ram_total > 0 and ram_source
          and ram_free is not None):
        free_gb = ram_free
        pool = "RAM (unified)"
        pool_source = ram_free_source
        # All experts live in RAM on a unified-memory machine.
        weights = _float((footprint or {}).get("resident_gb"))
        reserve = max(1.0, free_gb * 0.08)
    else:
        return {
            "ok": True, "applied": False, "reason": "no memory telemetry",
            "values": values, "config": dict(values), "sources": {}, "notes": [
                "no VRAM source and not a unified-memory platform with known "
                "free RAM; auto-tune abstained (missing telemetry is not zero "
                "memory)"],
            **values,
        }

    footprint = footprint if isinstance(footprint, dict) else {}
    guessed = sorted(_KV_GUESSED_FIELDS.intersection(
        str(field) for field in (footprint.get("estimated_fields") or [])))
    if guessed:
        notes.append(
            "metadata guessed — set model_layers/model_kv_heads/"
            "model_head_dim/model_quant or run pdc profile --run "
            f"(estimated: {', '.join(guessed)})")
    kv_per_1k = _float(footprint.get("kv_gb_per_1k_tokens"))
    confidence = _float(footprint.get("confidence"))
    confident = confidence >= 0.5
    proposal = 0
    if kv_per_1k > 0 and confidence > 0:
        headroom = free_gb - weights - reserve
        supported = int(max(0.0, headroom) / kv_per_1k * 1000.0) if \
            headroom > 0 else 0
        current = max(0, _int(values["context_tokens"]))
        floor = _floor_context(current)
        cap = _int(server_window) if server_window else 0
        target = max(floor, supported)
        if cap > 0:
            target = min(target, max(floor, cap))
        else:
            # No server-reported window: keep the growth proposal modest so a
            # provider cannot turn a heuristic into a 200k-token suggestion.
            cap = max(floor, current * 2 if current else 16384)
            target = min(target, cap)
            if target > current:
                notes.append("server window unknown; growth proposal capped "
                             "at 2x the current setting")
        # Growth is the only place the tuner can exceed auto_fit's physics:
        # cap it with the same shared KV model auto_fit uses. Shrinks already
        # come from that footprint and are not second-guessed here.
        hw_cap = _hardware_context_cap(cfg)
        if hw_cap > 0 and target > current and target > hw_cap:
            target = max(floor, hw_cap)
            notes.append(
                f"context_tokens: growth proposal capped at {target:,} by the "
                f"auto-fit VRAM estimate")
        explicit_ctx = _is_explicit(cfg, "context_tokens")
        effective_ctx = current
        if explicit_ctx:
            notes.append("context_tokens is explicit; left untouched")
        elif supported <= 0 and headroom <= 0:
            # The weights themselves do not fit the free pool. Never divide
            # by a zero/negative estimate; keep the safety floor.
            if current > floor:
                effective_ctx = floor
                if confident:
                    values["context_tokens"] = floor
                    sources["context_tokens"] = (
                        f"auto-tune ({pool} {free_gb:.1f} GB free via "
                        f"{pool_source}; {weights:.1f} GB weights leave no KV "
                        f"headroom; safety floor {floor:,})")
                    applied = True
                    notes.append(
                        f"context_tokens: {current:,} -> {floor:,} (weights do "
                        f"not fit the free {pool}; safety floor)")
                else:
                    notes.append(
                        f"context_tokens: would shrink {current:,} -> "
                        f"{floor:,} (weights do not fit the free {pool}; "
                        f"confidence {confidence:.2f} < 0.50, proposal only)")
        elif current > 0 and target < current:
            effective_ctx = target
            if confident:
                values["context_tokens"] = target
                sources["context_tokens"] = (
                    f"auto-tune ({pool} {free_gb:.1f} GB free via {pool_source}; "
                    f"weights {weights:.1f} GB; KV {kv_per_1k:.4f} GB/1k tok; "
                    f"confidence {confidence:.2f}; floor {floor:,})")
                applied = True
                notes.append(f"context_tokens: {current:,} -> {target:,} "
                             f"({pool} headroom)")
            else:
                notes.append(
                    f"context_tokens: would shrink {current:,} -> {target:,} "
                    f"({pool} headroom; confidence {confidence:.2f} < 0.50, "
                    "proposal only)")
        elif current > 0 and target > current:
            proposal = target
            notes.append(
                f"context_tokens: {current:,} fits; {target:,} also fits "
                f"({pool} headroom), proposed growth left to the user")
        else:
            notes.append(f"context_tokens: keeping {current:,} ({pool} headroom)")
        if supported <= 0 and headroom <= 0 and current <= floor:
            notes.append(f"context_tokens: keeping safety floor {floor:,}")

        tuned_ctx = max(1, _int(effective_ctx))
        moe = _is_moe(cfg)
        pressure = 0.6 if moe else 1.0
        if not _is_explicit(cfg, "max_gathered_chars"):
            previous = values["max_gathered_chars"]
            gathered_tokens = int(min(6000, max(512, tuned_ctx / 6.0
                                                * pressure)))
            gathered_chars = int(min(24000, max(2048, gathered_tokens * 4)))
            if previous != gathered_chars:
                if confident:
                    values["max_gathered_chars"] = gathered_chars
                    sources["max_gathered_chars"] = (
                        f"auto-tune (retrieval scaled to {tuned_ctx:,} tok"
                        + (", local-moe smaller window" if moe else "") + ")")
                    applied = True
                    notes.append("max_gathered_chars: "
                                 f"{previous:,} -> {gathered_chars:,}")
                else:
                    verb = "shrink" if gathered_chars < previous else "grow"
                    notes.append(
                        f"max_gathered_chars: would {verb} {previous:,} -> "
                        f"{gathered_chars:,} (confidence {confidence:.2f} < "
                        "0.50, proposal only)")
        if not _is_explicit(cfg, "flat_map_token_budget"):
            previous = values["flat_map_token_budget"]
            map_tokens = int(min(4000, max(800, tuned_ctx / 16.0 * pressure)))
            if previous != map_tokens:
                if confident:
                    values["flat_map_token_budget"] = map_tokens
                    sources["flat_map_token_budget"] = (
                        f"auto-tune (map scaled to {tuned_ctx:,} tok"
                        + (", local-moe smaller window" if moe else "") + ")")
                    applied = True
                    notes.append("flat_map_token_budget: "
                                 f"{previous:,} -> {map_tokens:,}")
                else:
                    verb = "shrink" if map_tokens < previous else "grow"
                    notes.append(
                        f"flat_map_token_budget: would {verb} {previous:,} -> "
                        f"{map_tokens:,} (confidence {confidence:.2f} < "
                        "0.50, proposal only)")
    else:
        notes.append("no usable KV-cache estimate; context left untouched")

    moe = _is_moe(cfg)
    if not _is_explicit(cfg, "timeout"):
        current_timeout = _float(values["timeout"])
        if moe:
            if current_timeout < 600.0:
                values["timeout"] = 600.0
                sources["timeout"] = (
                    "auto-tune (tiered local-moe keeps the 600 s timeout)")
                applied = True
                notes.append("timeout: tiered local-moe -> 600 s")
        elif isinstance(probe, dict) and _float(
                probe.get("prefill_tok_per_s")) > 0:
            prefill = _float(probe.get("prefill_tok_per_s"))
            ctx = max(1, _int(values["context_tokens"]))
            estimate = int(max(60.0, min(600.0, (ctx / prefill) * 1.5)))
            if abs(estimate - current_timeout) >= 1:
                values["timeout"] = float(estimate)
                sources["timeout"] = (
                    f"auto-tune (prefill {prefill:.0f} tok/s, window {ctx:,} "
                    f"tok -> {estimate} s)")
                applied = True
                notes.append(f"timeout: {current_timeout:.0f} -> {estimate} s "
                             f"(prefill estimate)")

    if moe and _int(values["max_concurrency"]) != 1:
        values["max_concurrency"] = 1
        sources["max_concurrency"] = (
            "auto-tune (tiered local-moe, single stream)")
        applied = True
        notes.append("max_concurrency: 1 (tiered local-moe)")

    return {
        "ok": True, "applied": applied, "reason": "",
        "values": values, "config": dict(values), "sources": sources,
        "notes": notes,
        "proposed_context_tokens": proposal,
        "memory": {"pool": pool, "free_gb": _round(free_gb),
                   "source": pool_source, "weights_gb": _round(weights),
                   "reserve_gb": _round(reserve),
                   "display_total_gb": _round(_float(vram.get("total_gb"))),
                   "tuning_card": (vram_basis or {}).get("card", ""),
                   "tuning_card_total_gb": (vram_basis or {}).get(
                       "total_gb", 0.0),
                   "ignored_cards": list((vram_basis or {}).get("ignored")
                                         or [])},
        "confidence": _round(confidence, 2),
        **values,
    }


def apply_auto_tune(cfg, *, snap: dict | None = None,
                    footprint: dict | None = None,
                    probe: dict | None = None,
                    server_window: int | None = None,
                    fetch_metadata: bool = False) -> dict:
    """Apply :func:`tune_profile` to `cfg` when `auto_tune` is opted in.

    Default-off: returns immediately (and touches no reader) when
    ``cfg.auto_tune`` is false. This is the config.load() wiring point, so an
    existing user's config is byte-identical to before.
    """
    if cfg is None:
        return {"ok": False, "applied": False, "reason": "no config"}
    if not bool(getattr(cfg, "auto_tune", False)):
        return {"ok": True, "applied": False, "reason": "auto_tune off",
                "values": {}, "sources": {}, "notes": []}
    try:
        hosted = False
        try:
            from .config import is_hosted
            hosted = bool(is_hosted(cfg))
        except Exception:
            hosted = False
        if hosted:
            result = tune_profile(cfg, None, None, None)
        else:
            snap = snap if isinstance(snap, dict) else snapshot()
            if footprint is None:
                meta = resolve_architecture(cfg, fetch=fetch_metadata)
                if not meta:
                    footprint = {}
                else:
                    quant = str(meta.get("quant")
                                or getattr(cfg, "model_quant", "") or "")
                    footprint = estimate_model_footprint(
                        meta, quant,
                        _int(getattr(cfg, "context_tokens", 0)), cfg=cfg)
            if probe is None:
                try:
                    probe = read_cached_probe(cfg.base_url, cfg.model)
                except Exception:
                    probe = None
            result = tune_profile(cfg, snap, footprint, probe,
                                  server_window=server_window)
        if result.get("applied"):
            values = result.get("values") or {}
            for key, value in values.items():
                try:
                    setattr(cfg, key, value)
                except Exception:
                    continue
            sources = getattr(cfg, "sources", None)
            if not isinstance(sources, dict):
                sources = {}
                try:
                    cfg.sources = sources
                except Exception:
                    sources = None
            if sources is not None:
                sources.update(result.get("sources") or {})
            try:
                cfg._auto_tune_result = result
            except Exception:
                pass
        return result
    except Exception as exc:                 # pragma: no cover - defensive
        _LOG.info("auto-tune skipped: %s", _cut(exc))
        return {"ok": False, "applied": False,
                "reason": f"{type(exc).__name__}: {_cut(exc)}",
                "values": {}, "sources": {}, "notes": []}


# --- reporting ----------------------------------------------------------------

def _age_text(seconds: float) -> str:
    seconds = max(0.0, _float(seconds))
    if seconds < 90:
        return f"{int(seconds)} s ago"
    if seconds < 7200:
        return f"{int(seconds / 60)} min ago"
    if seconds < 172800:
        return f"{int(seconds / 3600)} h ago"
    return f"{int(seconds / 86400)} d ago"


def build_report(cfg, *, snap: dict | None = None, probe: dict | None = None,
                 server_window: int | None = None,
                 fetch_metadata: bool = True) -> dict:
    """One JSON-safe report for `pdc models`, `pdc profile --show` and the TUI."""
    snap = snap if isinstance(snap, dict) else snapshot()
    meta = resolve_architecture(cfg, fetch=fetch_metadata) if cfg else {}
    quant = str((meta or {}).get("quant")
                or getattr(cfg, "model_quant", "") or "") if cfg else ""
    fp = estimate_model_footprint(
        meta, quant, _int(getattr(cfg, "context_tokens", 0)),
        cfg=cfg) if meta else {}
    probe = probe if probe is not None else (
        read_cached_probe(cfg.base_url, cfg.model) if cfg else None)
    window = server_window
    if window is None:
        window = detect_context_window(cfg) if cfg else 0
    tuning = (tune_profile(cfg, snap, fp, probe, server_window=window)
              if cfg else {})
    if cfg is not None and isinstance(tuning, dict):
        shown = dict(tuning.get("sources") or {})
        for key in ("context_tokens", "max_gathered_chars",
                    "flat_map_token_budget", "timeout", "max_concurrency"):
            source = str((getattr(cfg, "sources", None) or {}).get(key, ""))
            if key not in shown and source.startswith("auto-tune"):
                shown[key] = source
        tuning["sources"] = shown
    return {
        "ok": True,
        "snapshot": snap,
        "architecture": meta,
        "footprint": fp,
        "probe": probe,
        "server_window": window,
        "tuning": tuning,
        "current": {
            key: getattr(cfg, key, 0)
            for key in ("context_tokens", "max_gathered_chars",
                        "flat_map_token_budget", "timeout",
                        "max_concurrency")
        } if cfg else {},
        "auto_tune": bool(getattr(cfg, "auto_tune", False)) if cfg else False,
        "model": str(getattr(cfg, "model", "") or "") if cfg else "",
        "profile": str(getattr(cfg, "active_profile", "") or "") if cfg else "",
        "base_url": str(getattr(cfg, "base_url", "") or "") if cfg else "",
    }


def format_profile_report(report: dict) -> str:
    """Plain-text rendering shared by the CLI and the TUI."""
    if not isinstance(report, dict) or not report.get("ok"):
        return "Profiling report unavailable."
    snap = report.get("snapshot") or {}
    ram = snap.get("ram") or {}
    vram = snap.get("vram") or {}
    sources = snap.get("sources") or {}
    tuning = report.get("tuning") or {}
    memory = tuning.get("memory") or {}
    tuning_card = str(memory.get("tuning_card") or "")
    ignored_cards = [str(card) for card in memory.get("ignored_cards") or []]
    lines = ["Hardware snapshot"]
    lines.append(
        f"  RAM:  {ram.get('used_gb', 0):.1f} / {ram.get('total_gb', 0):.1f} "
        f"GB used, {ram.get('free_gb', 0):.1f} GB free"
        + (f"  [{sources.get('ram_total_gb')}]"
           if sources.get("ram_total_gb") else "  [unavailable]"))
    lines.append(
        f"  VRAM: {vram.get('used_gb', 0):.1f} / {vram.get('total_gb', 0):.1f} "
        f"GB used, {vram.get('free_gb', 0):.1f} GB free"
        + (f"  [{sources.get('vram_total_gb')}]"
           if sources.get("vram_total_gb") else "  [unavailable]"))
    for card in vram.get("cards") or []:
        name = str(card.get("card") or "gpu")
        free = _free_gb(card)
        marker = ""
        if name == tuning_card:
            marker = "  <- tuning basis"
        elif name in ignored_cards:
            marker = "  (ignored for tuning)"
        lines.append(
            f"    {name}: {card.get('used_gb', 0):.1f} / "
            f"{card.get('total_gb', 0):.1f} GB used"
            + (f", {free:.1f} GB free" if free is not None else "")
            + f"  [{card.get('source')}; totals only, no per-process "
            "attribution]" + marker)
    if tuning_card:
        lines.append(
            f"  tuning basis: {tuning_card} "
            f"({_float(memory.get('tuning_card_total_gb')):.1f} GB); "
            f"{', '.join(ignored_cards)} ignored "
            "(heterogeneous cards are not pooled)")
    if not snap.get("available"):
        lines.append("  (no telemetry reader answered; auto-tune abstains)")
    lines.append("")
    lines.append(f"Active model: {report.get('model') or '—'}"
                 + (f"  (profile {report.get('profile')})"
                    if report.get("profile") else ""))
    meta = report.get("architecture") or {}
    if meta:
        lines.append(f"  architecture source: {meta.get('source') or 'unknown'}")
    fp = report.get("footprint") or {}
    if fp:
        lines.append(
            f"  footprint: {fp.get('resident_gb', 0):.1f} GB resident"
            + (f", {fp.get('active_gb', 0):.1f} GB active"
               if fp.get("arch") == "moe" else "")
            + f"; KV {fp.get('kv_gb_per_1k_tokens', 0):.4f} GB / 1k tok"
            + f"; confidence {fp.get('confidence', 0):.2f}")
        if fp.get("estimated_fields"):
            lines.append("    estimated: "
                         + ", ".join(str(f) for f in fp["estimated_fields"]))
    probe = report.get("probe")
    if isinstance(probe, dict) and probe.get("ok"):
        age = _age_text(probe.get("age_s", 0)) if probe.get("cached") \
            else "this run"
        lines.append(
            f"  speed probe ({age}): prefill "
            f"{probe.get('prefill_tok_per_s', 0):.0f} tok/s, generation "
            f"{probe.get('generation_tok_per_s', 0):.1f} tok/s")
    else:
        lines.append("  speed probe: none cached (run `pedacito profile --run`)")
    lines.append("")
    lines.append("Auto-tune: "
                 + ("on" if report.get("auto_tune") else
                    "off (set auto_tune = true in the profile)"))
    if not report.get("auto_tune"):
        current = report.get("current") or tuning
        lines.append("  current: context_tokens="
                     f"{current.get('context_tokens', 0):,}, "
                     f"timeout={current.get('timeout', 0):g}s, "
                     f"max_concurrency={current.get('max_concurrency', 0)}")
        if tuning.get("notes"):
            lines.append("  with auto_tune on, it would propose:")
        for note in (tuning.get("notes") or [])[:6]:
            lines.append("  " + str(note))
    else:
        if tuning.get("proposed_context_tokens"):
            lines.append("  proposed context_tokens: "
                         f"{tuning['proposed_context_tokens']:,} "
                         "(growth is proposed, not applied)")
        for key in ("context_tokens", "max_gathered_chars",
                    "flat_map_token_budget", "timeout", "max_concurrency"):
            value = tuning.get(key, tuning.get("values", {}).get(key, 0))
            source = (tuning.get("sources") or {}).get(key, "")
            if isinstance(value, int):
                rendered = f"{value:,}"
            else:
                rendered = f"{_float(value):g}"
            lines.append(f"  {key}: {rendered}"
                         + (f"  ({source})" if source else ""))
        for note in tuning.get("notes") or []:
            lines.append("  - " + str(note))
    return "\n".join(lines)
