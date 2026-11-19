"""Context-aware auto-update: a quiet, opt-out check for newer Pedacito releases.

Local-first by construction. The ONLY network call this module ever makes is
one anonymous GET of the public PyPI JSON metadata for `pedacito` (no account,
no telemetry, no project data). It runs on a daemon thread at startup so it
can never delay startup by a request round-trip, and every failure mode --
offline, timeout, non-200, malformed JSON -- is a silent no-op.

Knobs and guarantees:
  * `check_for_updates = false` (config, or PEDACITO_CHECK_FOR_UPDATES=false)
    means ZERO network calls and zero subprocesses, everywhere: startup hooks,
    `pdc update`, the TUI's `/update` and the GUI banner all stand down.
  * A successful check is cached beside the user config (`update_check.json`)
    and re-queried at most once per 24 h; a manual check bypasses the cache.
  * Version comparison is semver-aware (pre-releases sort before their final
    release) and never reports an update when installed >= latest. Yanked
    releases are ignored.
  * The upgrade command is always an argv LIST executed with shell=False
    (Windows-safe), chosen from how Pedacito was installed (pipx, venv,
    user pip, or an editable source checkout -- which gets git instructions
    instead of a command).
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.request import urlopen

from . import __version__
from . import config as config_mod

PYPI_URL = "https://pypi.org/pypi/pedacito/json"
CHECK_TIMEOUT = 2.0          # seconds; a slow PyPI must never hold anything up
CACHE_TTL = 24 * 3600.0      # re-query at most once a day
CACHE_NAME = "update_check.json"

# --- version comparison -------------------------------------------------------

# release digits, optional pre/post suffix (PEP 440-ish spellings), optional
# +local build tag (ignored for ordering). Anything else -> unparseable.
_VERSION_RE = re.compile(
    r"^\s*v?(\d+(?:\.\d+)*)"
    r"(?:[-_.]?(a|alpha|b|beta|c|rc|pre|preview|dev|post|rev|r)[-_.]?(\d*))?"
    r"(?:\+[0-9A-Za-z.-]+)?\s*$",
    re.IGNORECASE)
_STAGE_RANK = {"dev": 0, "a": 1, "alpha": 1, "b": 2, "beta": 2,
               "c": 3, "rc": 3, "pre": 3, "preview": 3,
               "post": 5, "rev": 5, "r": 5}
_FINAL_RANK = 4


def parse_version(text: str):
    """A comparable ordering key for a version string, or None when it does
    not parse. Pre-releases sort before their final release (1.0rc1 < 1.0),
    post-releases after it (1.0 < 1.0.post1); `+build` metadata is ignored."""
    if not isinstance(text, str):
        return None
    m = _VERSION_RE.match(text)
    if m is None:
        return None
    release = tuple(int(p) for p in m.group(1).split("."))
    stage = (m.group(2) or "").lower()
    rank = _STAGE_RANK.get(stage, _FINAL_RANK)
    num = int(m.group(3)) if m.group(3) else 0
    return (release, rank, num)


def _padded(a: tuple, b: tuple) -> tuple[tuple, tuple]:
    n = max(len(a), len(b))
    return (a + (0,) * (n - len(a)), b + (0,) * (n - len(b)))


def is_newer(candidate: str, installed: str) -> bool:
    """True only when both parse and candidate > installed. An unparseable
    installed version is treated conservatively as 'do not nag'."""
    a, b = parse_version(candidate), parse_version(installed)
    if a is None or b is None:
        return False
    ra, rb = _padded(a[0], b[0])
    return (ra, a[1], a[2]) > (rb, b[1], b[2])


# --- cache ---------------------------------------------------------------------

def cache_path() -> Path:
    """The JSON cache lives beside the user config. Resolved at call time so
    tests (and XDG relocation) can move USER_CONFIG freely."""
    return config_mod.USER_CONFIG.parent / CACHE_NAME


def read_cache(now: float | None = None) -> dict | None:
    """{"checked_at": epoch, "latest": version} when fresh, else None.
    Missing, corrupt or stale files all read as 'no cache'."""
    now = time.time() if now is None else now
    try:
        with open(cache_path(), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    checked_at, latest = data.get("checked_at"), data.get("latest")
    if not isinstance(checked_at, (int, float)) or not isinstance(latest, str):
        return None
    if not latest or now - checked_at >= CACHE_TTL or checked_at > now + 60:
        return None
    return {"checked_at": float(checked_at), "latest": latest}


def write_cache(latest: str, now: float | None = None) -> None:
    """Best-effort atomic write; a read-only config dir is not an error."""
    now = time.time() if now is None else now
    path = cache_path()
    tmp = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"checked_at": now, "latest": latest}, fh)
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


# --- the one network call -------------------------------------------------------

def _fetch_payload(url: str = PYPI_URL, timeout: float = CHECK_TIMEOUT):
    """GET the public PyPI metadata and return the parsed JSON, or None on a
    non-200 answer. Any exception propagates to the caller, which treats it
    as 'offline'. `urlopen` is a module alias so tests can patch it."""
    with urlopen(url, timeout=timeout) as resp:
        status = getattr(resp, "status", None)
        if status is None:                     # file-like test doubles
            status = getattr(resp, "code", 200)
        if status != 200:
            return None
        return json.loads(resp.read())


def _latest_from_payload(payload) -> str | None:
    """The newest non-yanked release with files, from PyPI's JSON shape.
    Falls back to `info.version` when the releases map is unusable."""
    if not isinstance(payload, dict):
        return None
    best, best_key = None, None
    releases = payload.get("releases")
    if isinstance(releases, dict):
        for ver, files in releases.items():
            if not isinstance(files, list) or not files:
                continue
            if all(isinstance(f, dict) and f.get("yanked") for f in files):
                continue
            key = parse_version(ver)
            if key is None:
                continue
            if best_key is None or key > best_key:
                best, best_key = ver, key
    if best is None:
        info = payload.get("info")
        ver = info.get("version") if isinstance(info, dict) else None
        if isinstance(ver, str) and parse_version(ver) is not None:
            best = ver
    return best


# --- install-method detection ----------------------------------------------------

def _package_dirs() -> list[Path]:
    """Where this package lives on disk (tests monkeypatch this)."""
    try:
        return [Path(__file__).resolve().parent]
    except (OSError, ValueError):
        return []


def _editable_finder_active() -> bool:
    """Modern pip editable installs register an `__editable__` meta-path
    finder; legacy ones leave `*.egg-link` files on sys.path."""
    for finder in sys.meta_path:
        name = type(finder).__name__
        module = getattr(finder, "__module__", "") or ""
        if name.startswith("__editable__") or "__editable__" in module:
            return True
    for entry in sys.path:
        try:
            p = Path(entry or ".")
            if p.is_dir() and any(p.glob("*.egg-link")):
                return True
        except (OSError, ValueError):
            continue
    return False


def _pyvenv_cfg_mentions_pipx() -> bool:
    try:
        cfg = Path(sys.prefix) / "pyvenv.cfg"
        text = cfg.read_text(encoding="utf-8", errors="replace").lower()
        return "pipx" in text
    except OSError:
        return False


def _pipx_list_has_pedacito() -> bool:
    """Guarded `pipx list --json`: only when pipx is on PATH, with a short
    timeout, and every failure reads as 'not pipx'."""
    pipx = shutil.which("pipx")
    if not pipx:
        return False
    try:
        proc = subprocess.run([pipx, "list", "--json"], shell=False,
                              capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=3.0)
        if proc.returncode != 0:
            return False
        data = json.loads(proc.stdout)
        venvs = (data or {}).get("venvs") or {}
        return "pedacito" in venvs
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return False


def detect_install_method(*, query_pipx: bool = True) -> str:
    """How Pedacito was installed: pipx | editable | venv | user-pip | unknown.
    Purely local heuristics (paths, finders, pyvenv.cfg); the only subprocess
    is a guarded `pipx list --json` when pipx exists and paths were unclear.
    `query_pipx=False` skips even that (the opt-out path runs no subprocess)."""
    dirs = _package_dirs()
    if not dirs:
        return "unknown"
    for d in dirs:
        for cand in (d, d.parent):
            if (cand / ".git").exists():
                return "editable"
    if _editable_finder_active():
        return "editable"
    for d in dirs:
        parts = [p.lower() for p in d.parts]
        if "pipx" in parts or ("venvs" in parts and "pedacito" in parts):
            return "pipx"
    if sys.prefix != sys.base_prefix:
        if _pyvenv_cfg_mentions_pipx() or (query_pipx
                                           and _pipx_list_has_pedacito()):
            return "pipx"
        return "venv"
    if query_pipx and _pipx_list_has_pedacito():
        return "pipx"
    return "user-pip"


def externally_managed() -> bool:
    """PEP 668: the interpreter's stdlib carries an EXTERNALLY-MANAGED marker
    (Debian/Ubuntu/Homebrew system Pythons), so a bare `pip install --user`
    will refuse. Used only to phrase the manual instructions."""
    try:
        import sysconfig
        marker = Path(sysconfig.get_path("stdlib")) / "EXTERNALLY-MANAGED"
        return marker.exists()
    except (OSError, KeyError, ValueError):
        return False


def upgrade_command(method: str) -> tuple[list[str] | None, str]:
    """(argv list or None, human instructions) for upgrading Pedacito.
    NEVER a shell string: the caller runs the list with shell=False, which
    is the only form that is safe on Windows too. Editable/source installs
    get no command -- upgrading them is a git operation in the checkout."""
    if method == "pipx":
        pipx = shutil.which("pipx") or "pipx"
        return ([pipx, "upgrade", "pedacito"],
                "Installed with pipx. Run: pipx upgrade pedacito")
    if method == "editable":
        return (None,
                "Editable/source install: in your Pedacito checkout run "
                "`git pull && pip install -e .` (or reinstall your branch).")
    if method == "venv":
        return ([sys.executable, "-m", "pip", "install", "--upgrade",
                 "pedacito"],
                "Installed in a virtual environment. Run: "
                f"{sys.executable} -m pip install --upgrade pedacito")
    if method == "user-pip":
        argv = [sys.executable, "-m", "pip", "install", "--upgrade",
                "--user", "pedacito"]
        note = ("Installed with pip --user. Run: "
                f"{sys.executable} -m pip install --upgrade --user pedacito")
        if externally_managed():
            note += (" -- this Python is externally managed (PEP 668), so pip "
                     "may refuse; prefer `pipx upgrade pedacito` or a venv.")
        return (argv, note)
    return (None,
            "Could not tell how Pedacito was installed. Upgrade with the "
            "tool you installed it with, e.g. `pipx upgrade pedacito` or "
            "`pip install --upgrade pedacito`.")


# --- running the upgrade ----------------------------------------------------------

def run_upgrade(argv: list[str], on_line=None, timeout: float | None = None) -> int:
    """Run an upgrade argv list (shell=False) and stream its combined
    stdout/stderr line by line to `on_line`. Returns the exit code; a spawn
    failure or timeout reads as 1. Never raises. The lines are drained by a
    small reader thread so `timeout` is a real wall-clock deadline even when
    the child produces no output at all."""
    try:
        proc = subprocess.Popen(argv, shell=False, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                encoding="utf-8", errors="replace")
    except (OSError, ValueError) as exc:
        if on_line is not None:
            try:
                on_line(f"could not start the upgrade: {exc}")
            except Exception:
                pass
        return 1

    def emit(line: str) -> None:
        if on_line is not None:
            try:
                on_line(line)
            except Exception:
                pass

    lines: queue.Queue = queue.Queue()

    def reader():
        try:
            for line in proc.stdout:
                lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            lines.put(None)             # EOF sentinel

    threading.Thread(target=reader, daemon=True).start()
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        wait = None if deadline is None else max(
            0.0, deadline - time.monotonic())
        try:
            item = lines.get(timeout=wait)
        except queue.Empty:
            proc.kill()
            proc.wait()
            emit("upgrade timed out and was stopped")
            return 1
        if item is None:
            break
        emit(item.rstrip("\n"))
    try:
        # stdout is at EOF, so the child is exiting; a short grace wait only.
        return proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return 1


# --- opt-out -----------------------------------------------------------------------

def updates_enabled(cfg=None) -> bool:
    """The single opt-out gate, in config precedence order: environment
    override first, then the loaded config, then a light read of the user
    config's [universal] table. Default: enabled."""
    raw = os.environ.get("PEDACITO_CHECK_FOR_UPDATES")
    if raw is not None and raw.strip():
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if cfg is not None:
        return bool(getattr(cfg, "check_for_updates", True))
    try:
        import tomllib
        with open(config_mod.USER_CONFIG, "rb") as fh:
            data = tomllib.load(fh)
        val = (data.get("universal") or {}).get("check_for_updates")
        if isinstance(val, bool):
            return val
    except (OSError, ValueError, KeyError):
        pass
    return True


# --- the check ----------------------------------------------------------------------

def check_for_update(cfg=None, *, manual: bool = False,
                     now: float | None = None, fetch=None) -> dict:
    """Synchronous core of the feature; returns a plain dict:

      ok               a latest version is known (network or fresh cache)
      current/latest   installed and newest non-yanked versions
      update_available latest parses and is strictly newer than installed
      method/command/instructions  how to upgrade (always local knowledge)
      cached           the answer came from the 24 h cache
      detail           "" | "disabled" | "offline"

    With the opt-out set, this performs ZERO network calls and returns the
    manual upgrade instructions only. `fetch` (url, timeout) -> payload is
    injectable for tests; `now` is a fake clock for cache-TTL tests."""
    info: dict = {"ok": False, "current": __version__, "latest": None,
                  "update_available": False, "cached": False,
                  "checked_at": 0.0, "detail": ""}
    enabled = updates_enabled(cfg)
    method = detect_install_method(query_pipx=enabled)
    command, instructions = upgrade_command(method)
    info.update(method=method, command=command, instructions=instructions)
    if not enabled:
        info["detail"] = "disabled"
        return info
    now = time.time() if now is None else now
    latest = None
    if not manual:
        cache = read_cache(now=now)
        if cache is not None:
            latest = cache["latest"]
            info["cached"] = True
            info["checked_at"] = cache["checked_at"]
    if latest is None:
        try:
            payload = (fetch or _fetch_payload)(PYPI_URL, CHECK_TIMEOUT)
            latest = _latest_from_payload(payload) if payload else None
        except Exception:
            latest = None
        if latest is None:
            info["detail"] = "offline"
            return info
        write_cache(latest, now=now)
        info["checked_at"] = now
    info["ok"] = True
    info["latest"] = latest
    info["update_available"] = is_newer(latest, __version__)
    return info


_check_lock = threading.Lock()
_check_running = False


def check_for_update_async(cfg=None, fetch=None, on_done=None):
    """Fire the check on a daemon thread so startup never waits a network
    round-trip. Deduplicated: while one check is in flight, further calls
    return None. Returns the thread (or None when disabled/deduped)."""
    global _check_running
    if not updates_enabled(cfg):
        return None
    with _check_lock:
        if _check_running:
            return None
        _check_running = True

    def worker():
        global _check_running
        try:
            info = check_for_update(cfg, fetch=fetch)
            if on_done is not None:
                try:
                    on_done(info)
                except Exception:
                    pass
        finally:
            with _check_lock:
                _check_running = False

    t = threading.Thread(target=worker, name="pdc-update-check", daemon=True)
    t.start()
    return t
