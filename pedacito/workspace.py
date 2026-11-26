"""Project workspace setup: the .pedacito/ directory, ignore files, backups,
session logs, and the run log.

Everything Pedacito writes for a project lives in one place:

    <project>/.pedacito/
        index.json          the map
        summaries.json      summary cache, keyed by content hash
        backups/<stamp>/    pre-edit snapshots, mirroring your tree
        checkpoints/        time-travel tree (index.json + snapshots/)
        sessions/<stamp>.md what was asked, looked up, and changed
        log.txt             operational run log

On first run this also creates a `.pedacitoignore` file and adds `.pedacito/`
to `.gitignore`.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from . import fileio, security
from .index import BACKUP_DIR, SESSION_DIR, state_dir
from .sessions import session_dir

IGNORE_FILE = ".pedacitoignore"

# Paths Pedacito writes into a project that should never be committed: the
# workspace tree, the ignore file, and the two project-config filenames
# find_project_config() looks for.
GITIGNORE_ENTRIES = (
    ".pedacito/",
    ".pedacitoignore",
    ".pedacito.toml",
    "pedacito.toml",
)

IGNORE_TEMPLATE = """\
# Files and folders Pedacito should not index. Gitignore syntax.
# Your .gitignore is read too, along with built-in defaults for .git,
# __pycache__, .venv, node_modules, build, dist and similar.
#
# Examples:
#   deprecated/              a folder of old script versions
#   *_old.py
#   /scratch                 anchored to the project root only
#   !deprecated/still_used.py    negation; last match wins
#
# Naming a path directly on the command line always overrides these.

.pedacito/
"""


def stamp() -> str:
    """Current time as a sortable filesystem-safe string (backup and session
    names)."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


RUN_LOG_NAME = "log.txt"


def run_log_path(root: Path) -> Path:
    """Path of the operational run log for a project: `.pedacito/log.txt`."""
    return state_dir(root, RUN_LOG_NAME)


def log_event(root: Path, message: str, kind: str = "INFO") -> None:
    """Append one timestamped line to the run log: commands, phase
    transitions, statuses, and errors, but no model output (that lives in
    `.pedacito/sessions/*.md`). Appended across runs."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    p = run_log_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(f"{ts}  {kind:<6} {message}\n")


def ensure(root: Path, manage_gitignore: bool = True,
           create_ignore: bool = True) -> Path:
    """Create .pedacito/ under `root` plus the ignore scaffolding
    (.pedacitoignore, a .gitignore entry). Idempotent."""
    root = Path(root).resolve()
    d = state_dir(root)
    first_run = not d.exists()
    d.mkdir(parents=True, exist_ok=True)

    if create_ignore:
        ig = root / IGNORE_FILE
        if not ig.exists():
            ig.write_text(IGNORE_TEMPLATE, encoding="utf-8")
            print(f"  created {ig}", file=sys.stderr)
        elif ".pedacito" not in ig.read_text(encoding="utf-8", errors="replace"):
            with ig.open("a", encoding="utf-8") as fh:
                fh.write("\n.pedacito/\n")

    if manage_gitignore:
        _add_to_gitignore(root)

    if first_run:
        print(f"  state for this project lives in {d}", file=sys.stderr)
    return d


def _add_to_gitignore(root: Path) -> None:
    """Add Pedacito's entries to the project's .gitignore: create it in a git
    checkout that has none, else append the missing entries. Never creates one
    outside a repo."""
    gi = root / ".gitignore"
    if not gi.exists():
        if not (root / ".git").exists():
            return          # not a repo; leave the directory alone
        gi.write_text("# Pedacito\n" + "\n".join(GITIGNORE_ENTRIES) + "\n",
                      encoding="utf-8")
        print(f"  created {gi} with Pedacito entries", file=sys.stderr)
        return

    text = gi.read_text(encoding="utf-8", errors="replace")
    # ".pedacito" and ".pedacito/" count as the same entry
    existing = {
        ln.strip().rstrip("/")
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    }
    to_add = [e for e in GITIGNORE_ENTRIES if e.rstrip("/") not in existing]
    if not to_add:
        return

    with gi.open("a", encoding="utf-8") as fh:
        if text and not text.endswith("\n"):
            fh.write("\n")
        if text:
            fh.write("\n")           # blank separator
        fh.write("# Pedacito\n")
        for e in to_add:
            fh.write(f"{e}\n")
    n = len(to_add)
    print(f"  updated {gi} ({n} Pedacito entr"
          f"{'y' if n == 1 else 'ies'} added)", file=sys.stderr)


def new_backup_dir(root: Path, when: str | None = None,
                   unique: bool = False) -> Path:
    """Create and return a timestamped directory under .pedacito/backups/.

    With `unique`, a directory that already holds a snapshot (two runs began
    within the same second) is never reused: the new one gets a `-2`, `-3`
    suffix, so one run's originals cannot merge into another's. Without it the
    stamp's directory is shared on purpose -- every write of one run snapshots
    into the same place, first copy wins."""
    base = when or stamp()
    d = state_dir(root, BACKUP_DIR, base)
    if unique:
        n = 2
        while d.exists() and any(d.iterdir()):
            d = state_dir(root, BACKUP_DIR, f"{base}-{n}")
            n += 1
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_backups(root: Path) -> list[Path]:
    """Backup snapshot directories under .pedacito/backups/, newest first."""
    d = state_dir(root, BACKUP_DIR)
    return sorted((p for p in d.iterdir() if p.is_dir()), reverse=True) if d.exists() else []


def created_files(root: Path, snapshot: Path) -> list[str]:
    """Files the snapshot's run created that still exist (relative paths)."""
    from .fileio import created_manifest
    root = Path(root).resolve()
    try:
        names = created_manifest(snapshot).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for rel in names:
        p = (root / rel).resolve()
        if rel and _is_inside(p, root) and p.is_file():
            out.append(rel)
    return out


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def delete_created(root: Path, snapshot: Path, dry_run: bool = False) -> list[str]:
    """Remove the files the snapshot's run created (and directories left
    empty by that). Returns the relative paths removed (or that would be)."""
    root = Path(root).resolve()
    gone = created_files(root, snapshot)
    if dry_run:
        return gone
    for rel in gone:
        p = root / rel
        try:
            p.unlink()
        except OSError:
            continue
        parent = p.parent
        while parent != root and _is_inside(parent, root):
            try:
                parent.rmdir()            # only succeeds when empty
            except OSError:
                break
            parent = parent.parent
    return gone


def restore(root: Path, snapshot: Path, dry_run: bool = False) -> list[str]:
    """Copy every file in a backup snapshot back over the working tree
    byte-for-byte. Returns the relative paths restored (or that would be, if
    `dry_run`). Files created since the snapshot have no copy in it and are
    left in place unless the caller also runs `delete_created`."""
    root = Path(root).resolve()
    restored: list[str] = []
    for src in sorted(snapshot.rglob("*")):
        if not src.is_file() or src.is_symlink():
            # A hand-edited snapshot must not smuggle a link to outside data
            # into the tree (backups themselves store plain file bytes).
            continue
        rel = src.relative_to(snapshot).as_posix()
        # Containment BEFORE the write: a directory (or the file itself) on
        # the destination path may be a symlink aiming outside the project,
        # and copy_file_atomic replaces through links. resolve_safe_path
        # follows them first and refuses the escape.
        try:
            security.resolve_safe_path(root, rel)
        except (security.SecurityError, OSError, ValueError, RuntimeError):
            continue
        dest = root / rel
        restored.append(rel)
        if not dry_run:
            # Atomic replacement, never a write into the current inode: a
            # project file may be hardlinked to a ratchet checkpoint or an
            # older backup, and an in-place restore would rewrite both.
            fileio.copy_file_atomic(src, dest)
    return restored


def write_session(root: Path | None, task: str, steps: list[str], reply: str,
                  results, backup: Path | None, when: str | None = None,
                  raw_reply: str | None = None) -> Path:
    """Write a Markdown record of one task (request, lookups, reply, edits)
    to `.pedacito/sessions/<timestamp>.md`, or to the per-user
    `sessions/global/` folder when `root` is None. `raw_reply`, when given, is
    the reply as the model emitted it, before trimming or continuation rounds."""
    d = session_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    when = when or stamp()
    lines = [
        f"# {when}", "", f"**Task:** {task}", "", "## Lookups", "",
        *(f"- {s}" for s in (steps or ["(none)"])),
        "", "## Response", "", reply.strip() or "(empty)", "", "## Edits", "",
    ]
    if raw_reply is not None and raw_reply != reply:
        lines += ["", "## Raw reply (as emitted, before parsing/trimming)", "",
                  raw_reply.strip() or "(empty)"]
    if results:
        for r in results:
            lines.append(f"- {'applied' if r.ok else 'FAILED'} `{r.file}`: {r.message}")
    else:
        lines.append("- none")
    if backup is not None and root is not None and backup.exists() \
            and any(backup.rglob("*")):
        lines += ["", f"Originals saved to `{backup.relative_to(root)}`.",
                  f"Restore with `pedacito restore {root} --snapshot {backup.name}`."]
    p = d / f"{when}.md"
    n = 2
    while p.exists():             # two tasks in one second must not overwrite
        p = d / f"{when}-{n}.md"
        n += 1
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------- retention
def prune(root: Path, keep_sessions: int = 0, keep_backups: int = 0,
          keep_events: int = 0, log_max_bytes: int = 0,
          dry_run: bool = False) -> dict[str, int]:
    """Delete old sessions, backup snapshots and event files, keeping the
    newest N of each (0 = leave that kind alone), and trim `log.txt` to its
    last `log_max_bytes`. Returns how many of each were (or would be) removed.
    Backups are the undo path: only the oldest beyond the limit go."""
    import shutil
    removed = {"sessions": 0, "backups": 0, "events": 0, "scratch": 0,
               "log_bytes": 0}
    # Spilled tool-output logs are transient: `prune` always clears them.
    scratch = state_dir(root, "scratch")
    if scratch.is_dir():
        for p in scratch.glob("*.log"):
            removed["scratch"] += 1
            if not dry_run:
                p.unlink(missing_ok=True)
        if not dry_run:
            try:
                scratch.rmdir()      # remove only when it is now empty
            except OSError:
                pass
    if keep_sessions > 0:
        d = state_dir(root, SESSION_DIR)
        files = sorted(d.glob("*.md"), reverse=True) if d.is_dir() else []
        for p in files[keep_sessions:]:
            removed["sessions"] += 1
            if not dry_run:
                p.unlink(missing_ok=True)
    if keep_backups > 0:
        for snap in list_backups(root)[keep_backups:]:
            removed["backups"] += 1
            if not dry_run:
                shutil.rmtree(snap, ignore_errors=True)
                try:
                    from .fileio import created_manifest
                    created_manifest(snap).unlink()
                except OSError:
                    pass
    if keep_events > 0:
        d = state_dir(root, "events")
        files = sorted(d.glob("*.jsonl"), reverse=True) if d.is_dir() else []
        for p in files[keep_events:]:
            removed["events"] += 1
            if not dry_run:
                p.unlink(missing_ok=True)
    if log_max_bytes > 0:
        lp = run_log_path(root)
        try:
            size = lp.stat().st_size
        except OSError:
            size = 0
        if size > log_max_bytes:
            removed["log_bytes"] = size - log_max_bytes
            if not dry_run:
                data = lp.read_bytes()[-log_max_bytes:]
                nl = data.find(b"\n")
                lp.write_bytes(data[nl + 1:] if nl >= 0 else data)
    return removed


# Content under `.pedacito/` that the user authored rather than Pedacito
# generated. `reset` keeps it when `keep_config` is set.
AUTHORED_DIRS = ("rules", "skills", "prompts")
AUTHORED_FILES = ("instructions.md", "PEDACITO.md", "PEDACEITO.md")
MEMORY_FILE = "memory.md"


def _walk_files(path: Path):
    """Every regular file under `path`, without following symlinks."""
    if path.is_symlink():
        return
    if path.is_file():
        yield path
        return
    for f in path.rglob("*"):
        if f.is_file() and not f.is_symlink():
            yield f


def reset(root: Path, keep_memory: bool = False,
          keep_config: bool = False, dry_run: bool = False) -> dict:
    """Delete the project's `.pedacito/` runtime state.

    Returns ``{path, removed, kept, files, bytes}`` where `removed`/`kept` are
    top-level entry names. Code and the project's scaffolding files
    (`.pedacito.toml`, `.pedacitoignore`, `.gitignore`) are never touched.
    `keep_memory` preserves `memory.md`; `keep_config` preserves authored
    `rules/`, `skills/`, `prompts/` and instruction files. With `dry_run` the
    counts are reported but nothing is deleted.
    """
    d = state_dir(Path(root))
    keep: set[str] = set()
    if keep_memory:
        keep.add(MEMORY_FILE)
    if keep_config:
        keep.update(AUTHORED_DIRS)
        keep.update(AUTHORED_FILES)
    result = {"path": d, "removed": [], "kept": [], "files": 0, "bytes": 0}
    if d.is_symlink():
        raise OSError(f"refusing to reset {d}: it is a symlink")
    if not d.is_dir():
        return result
    for p in sorted(d.iterdir()):
        if p.name in keep:
            result["kept"].append(p.name)
            continue
        result["removed"].append(p.name)
        for f in _walk_files(p):
            result["files"] += 1
            try:
                result["bytes"] += f.stat().st_size
            except OSError:
                pass
    if dry_run:
        return result
    import shutil
    for name in result["removed"]:
        p = d / name
        try:
            if p.is_symlink() or p.is_file():
                p.unlink(missing_ok=True)
            elif p.is_dir():
                shutil.rmtree(p)
        except FileNotFoundError:
            pass                  # a concurrent prune took it; nothing to do
    d.mkdir(parents=True, exist_ok=True)
    return result


class ProjectBusy(RuntimeError):
    """Another live pedacito process is writing to this project."""


def _pid_alive_windows(pid: int) -> bool:
    """Existence check that works on Windows. `os.kill(pid, 0)` is not a
    probe there: Windows calls TerminateProcess for every signal except the
    console control events, so checking a live lock holder would kill it."""
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL,
                                     wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    ERROR_ACCESS_DENIED = 5
    STILL_ACTIVE = 259
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        # A process we may not inspect still exists (another user's run).
        return ctypes.get_last_error() == ERROR_ACCESS_DENIED
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    import os
    if os.name == "nt":
        return _pid_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


# ---------------------------------------------------------- process start time
#
# PID reuse is the failure mode the run lock must survive: a lock file left by
# a crashed run names a pid that the OS later hands to an unrelated process.
# ``_pid_alive`` alone would then declare the lock live forever. Comparing the
# recorded process start time against the current one closes the race (Linux
# /proc/<pid>/stat ticks, Windows GetProcessTimes, macOS ps -o lstart=).

_START_TIME_TOLERANCE = 2.0     # seconds; a live pid with a differing start
                                # time beyond this is a different process


def _process_start_time_windows(pid: int) -> float:
    """Creation time as seconds (100 ns intervals since 1601); 0.0 unknown."""
    import ctypes
    from ctypes import wintypes
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL,
                                         wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetProcessTimes.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME)]
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION,
                                      False, pid)
        if not handle:
            return 0.0
        try:
            born = wintypes.FILETIME()
            rest = wintypes.FILETIME()
            if not kernel32.GetProcessTimes(handle, ctypes.byref(born),
                                            ctypes.byref(rest),
                                            ctypes.byref(rest),
                                            ctypes.byref(rest)):
                return 0.0
            return ((born.dwHighDateTime << 32) | born.dwLowDateTime) / 1e7
        finally:
            kernel32.CloseHandle(handle)
    except Exception:                       # pragma: no cover - best effort
        return 0.0


def _process_start_time_ps(pid: int) -> float:
    """macOS / BSD fallback via ``ps -o lstart=`` parsed to an epoch second."""
    import subprocess
    import datetime as _dt
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)],
                             shell=False, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=2.0)
        text = (out.stdout or "").strip()
        if not text:
            return 0.0
        dt = _dt.datetime.strptime(text, "%a %b %d %H:%M:%S %Y")
        return dt.timestamp()
    except Exception:
        return 0.0


def _process_start_time(pid: int) -> float:
    """OS process creation time as a comparable float; 0.0 when unavailable.

    The unit differs by platform (seconds-since-boot on Linux, seconds-since-
    1601 on Windows, epoch seconds on macOS) but is always the value a LATER
    ``_process_start_time(pid)`` of the same live process returns unchanged."""
    if pid <= 0:
        return 0.0
    if os.name == "nt":
        return _process_start_time_windows(pid)
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(
            encoding="ascii", errors="replace")
    except OSError:
        return _process_start_time_ps(pid)
    try:
        # "pid (comm) state ppid ...": field 22 (starttime) follows the comm.
        fields = stat[stat.rfind(")") + 2:].split()
        starttime = int(fields[19])
        try:
            hz = float(os.sysconf("SC_CLK_TCK"))
        except (AttributeError, ValueError, OSError):
            hz = 100.0
        if hz <= 0:
            hz = 100.0
        return starttime / hz
    except (ValueError, IndexError, OSError):
        return 0.0


# --------------------------------------------------------------- lockfile v2
#
# `<root>/.pedacito/run.lock` is JSON {"version": 2, pid, start_time,
# owner_token, session_id, mode, created_at}. The reader stays
# backwards-compatible with the v1 single-integer `<pid>\n` format.

def _lock_path(root) -> Path:
    return state_dir(Path(root)) / "run.lock"


def _read_lockfile(path: Path) -> dict | None:
    """The parsed holder record, or None when missing/corrupt/unreadable."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "version" in data:
            return data
    except ValueError:
        pass
    try:
        pid = int(raw.split()[0])
        return {"version": 1, "pid": pid, "start_time": 0.0,
                "owner_token": "", "session_id": "", "mode": "exclusive",
                "created_at": 0.0}
    except (ValueError, IndexError):
        return None


def _is_stale(holder: dict, my_pid: int) -> bool:
    """True when `holder` no longer names a live run and may be taken over.

    A recorded start_time of 0.0 (legacy v1, or an unreadable / mocked pid) is
    tolerated: it never by itself declares the lock stale, so access-denied
    and test doubles cannot be evicted by a false positive."""
    pid = int(holder.get("pid", 0) or 0)
    if pid <= 0:
        return True
    if pid == my_pid:
        return True                       # our own pid left by a crash
    if not _pid_alive(pid):
        return True
    rec_start = float(holder.get("start_time", 0.0) or 0.0)
    if rec_start <= 0.0:
        return False
    cur_start = _process_start_time(pid)
    if cur_start <= 0.0:
        return False
    return abs(rec_start - cur_start) > _START_TIME_TOLERANCE


@dataclass(frozen=True)
class LockOwner:
    """Who holds the run lock: the process, its creation time, and the token
    that must match before ``release`` will unlink the on-disk lockfile."""
    pid: int
    start_time: float
    owner_token: str
    session_id: str
    mode: str                  # "exclusive" | "shared"
    created_at: float


class _InProcOwner:
    __slots__ = ("owner", "thread_id", "refs")

    def __init__(self, owner: LockOwner, thread_id: int):
        self.owner = owner
        self.thread_id = thread_id
        self.refs = 1


class ProjectLockManager:
    """In-process RWLock + cross-process atomic lockfile.

    In-process: multiple `shared` owners coexist; an `exclusive` owner
    conflicts with any other owner. Re-entry is allowed for the same
    `session_id` (when non-empty) or the same thread (when no session is
    given), so a session that nests acquires never deadlocks itself.

    Cross-process: the first in-process owner writes `<root>/.pedacito/
    run.lock` atomically (`O_CREAT | O_EXCL`) and every other process fails
    fast with `ProjectBusy` until it is released or detected stale.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._owners: dict[str, list[_InProcOwner]] = {}

    # ------------------------------------------------------------- acquire
    def acquire(self, root: Path | str, *, mode: str = "exclusive",
                session_id: str = "", wait_timeout: float = 0.0) -> LockOwner:
        if mode not in ("exclusive", "shared"):
            raise ValueError(f"unknown lock mode {mode!r}")
        path = _lock_path(root)
        key = str(path.resolve())
        path.parent.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        start = _process_start_time(pid)
        tid = threading.get_ident()
        deadline = (None if wait_timeout <= 0
                    else time.monotonic() + float(wait_timeout))
        with self._cond:
            while True:
                owners = self._owners.get(key, [])
                reentry = self._find_reentry(owners, session_id, tid)
                if reentry is not None:
                    reentry.refs += 1
                    return LockOwner(pid, start, reentry.owner.owner_token,
                                     session_id, reentry.owner.mode, time.time())
                if not self._conflicts(owners, mode):
                    token = uuid.uuid4().hex
                    if not owners:
                        self._write_lockfile(path, pid, start, token,
                                             session_id, mode)
                    owner = LockOwner(pid, start, token, session_id, mode,
                                      time.time())
                    self._owners.setdefault(key, []).append(
                        _InProcOwner(owner, tid))
                    return owner
                # A different session/thread holds a conflicting lock.
                reason = self._busy_reason(owners)
                if deadline is None:
                    raise ProjectBusy(reason)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ProjectBusy(reason)
                self._cond.wait(timeout=min(0.05, remaining))

    def _find_reentry(self, owners, session_id: str, tid: int):
        for o in owners:
            if session_id and o.owner.session_id == session_id:
                return o
            if not session_id and not o.owner.session_id \
                    and o.thread_id == tid:
                return o
        return None

    @staticmethod
    def _conflicts(owners, mode: str) -> bool:
        if not owners:
            return False
        if mode == "exclusive":
            return True
        return any(o.owner.mode == "exclusive" for o in owners)

    @staticmethod
    def _busy_reason(owners) -> str:
        names = sorted({o.owner.session_id or f"pid {o.owner.pid}"
                        for o in owners})
        return (f"another pedacito run is using this project "
                f"({', '.join(names)}); wait for it, or delete the run.lock "
                f"if it is stale")

    def _write_lockfile(self, path, pid, start, token, session_id, mode):
        while True:
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                pass
            except OSError as exc:
                raise RuntimeError(
                    f"could not acquire run lock {path}: {exc}") from exc
            else:
                break
            holder = _read_lockfile(path)
            if holder is not None and not _is_stale(holder, pid):
                raise ProjectBusy(
                    f"another pedacito run (pid {holder.get('pid')}) is using "
                    f"this project; wait for it, or delete {path} if it is stale")
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise RuntimeError(
                    f"could not replace stale run lock {path}: {exc}") from exc
        payload = json.dumps({"version": 2, "pid": pid, "start_time": start,
                              "owner_token": token, "session_id": session_id,
                              "mode": mode, "created_at": time.time()})
        try:
            os.write(fd, payload.encode("utf-8"))
        finally:
            os.close(fd)

    # -------------------------------------------------------------- release
    def release(self, root: Path | str, owner: LockOwner | None = None) -> None:
        path = _lock_path(root)
        key = str(path.resolve())
        with self._cond:
            owners = self._owners.get(key, [])
            if not owners:
                return
            target = self._find_target(owners, owner)
            if target is None:
                return
            target.refs -= 1
            if target.refs > 0:
                self._cond.notify_all()
                return
            owners.remove(target)
            if not owners:
                del self._owners[key]
                self._unlink_if_ours(path, target.owner)
            self._cond.notify_all()

    def _find_target(self, owners, owner):
        if owner is not None:
            for o in owners:
                if o.owner.owner_token == owner.owner_token:
                    return o
            if owner.session_id:
                for o in owners:
                    if o.owner.session_id == owner.session_id:
                        return o
            return None
        tid = threading.get_ident()
        for o in owners:
            if o.thread_id == tid:
                return o
        return None

    def _unlink_if_ours(self, path, owner: LockOwner) -> None:
        """Unlink the lockfile only when it still names this owner's token (or
        pid for a legacy v1 file) -- a file another process took over is not
        ours to remove."""
        holder = _read_lockfile(path)
        if holder is None:
            return
        if int(holder.get("version", 1) or 1) == 1:
            if int(holder.get("pid", 0) or 0) == owner.pid:
                try:
                    path.unlink()
                except OSError:
                    pass
            return
        if str(holder.get("owner_token", "")) == owner.owner_token:
            try:
                path.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------- queries
    def active_owners(self, root: Path | str) -> list[LockOwner]:
        path = _lock_path(root)
        key = str(path.resolve())
        with self._cond:
            owners = [o.owner for o in self._owners.get(key, [])]
        if owners:
            return owners
        holder = _read_lockfile(path)
        if holder is None:
            return []
        return [LockOwner(int(holder.get("pid", 0) or 0),
                          float(holder.get("start_time", 0.0) or 0.0),
                          str(holder.get("owner_token", "")),
                          str(holder.get("session_id", "")),
                          str(holder.get("mode", "exclusive")),
                          float(holder.get("created_at", 0.0) or 0.0))]

    def is_locked(self, root: Path | str) -> bool:
        path = _lock_path(root)
        key = str(path.resolve())
        with self._cond:
            if self._owners.get(key):
                return True
        holder = _read_lockfile(path)
        return holder is not None and not _is_stale(holder, os.getpid())


# Process-default manager and the legacy module-level helpers. `acquire_lock`
# keeps its historical Path return value so existing callers keep working;
# the per-thread release bookkeeping lives in a thread-local stack.
_DEFAULT_LOCK_MANAGER = ProjectLockManager()
_LOCK_TLS = threading.local()


def acquire_lock(root, *, mode: str = "exclusive", session_id: str = "",
                 wait_timeout: float = 0.0) -> Path:
    """Claim `<root>/.pedacito/run.lock` for this process (see
    `ProjectLockManager`). Two different threads or sessions cannot both hold
    an exclusive lock: the second raises `ProjectBusy` (or waits when
    `wait_timeout > 0`). Returns the lock file path for backwards
    compatibility; a stale lock (dead pid, or a live pid whose creation time
    no longer matches) is taken over."""
    path = _lock_path(root)
    owner = _DEFAULT_LOCK_MANAGER.acquire(
        root, mode=mode, session_id=session_id, wait_timeout=wait_timeout)
    held = getattr(_LOCK_TLS, "owners", None)
    if held is None:
        held = {}
        _LOCK_TLS.owners = held
    key = str(path.resolve())
    held.setdefault(key, []).append(owner)
    return path


def release_lock(root) -> None:
    """Release this thread's most recent `acquire_lock` on `root`."""
    path = _lock_path(root)
    key = str(path.resolve())
    held = getattr(_LOCK_TLS, "owners", {})
    owners = held.get(key, [])
    if not owners:
        return
    owner = owners.pop()
    if not owners:
        del held[key]
    _DEFAULT_LOCK_MANAGER.release(root, owner)


# --------------------------------------------------------- atomic state files
#
# Shared project-state files are read-modify-written by concurrent sessions;
# doing that without a per-path lock loses updates (two writers both read the
# old value, each add a different entry, and the later write clobbers the
# earlier). `atomic_update_json` / `atomic_update_text` serialise the whole
# read-modify-write under one lock and commit through `fileio.write_bytes`
# (temp file + rename), so no update is lost and no inode is ever truncated.

_FILE_LOCKS_GUARD = threading.Lock()
_FILE_LOCKS: dict[str, threading.Lock] = {}


def _file_lock(path: Path) -> threading.Lock:
    key = str(path)
    with _FILE_LOCKS_GUARD:
        lock = _FILE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _FILE_LOCKS[key] = lock
        return lock


def atomic_update_json(
    path: Path,
    updater: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    default: Callable[[], dict[str, Any]] = dict,
    root: Path | None = None,
    compact: bool = False,
) -> dict[str, Any]:
    """Atomically read-modify-write a JSON file, returning the new dict."""
    p = Path(path)
    with _file_lock(p):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = default()
        if not isinstance(data, dict):
            data = default()
        data = updater(data)
        if not isinstance(data, dict):
            raise TypeError("atomic_update_json updater must return a dict")
        p.parent.mkdir(parents=True, exist_ok=True)
        if compact:
            payload = json.dumps(data, separators=(",", ":"))
        else:
            payload = json.dumps(data, indent=1, sort_keys=True)
        fileio.write_bytes(p, payload.encode("utf-8"))
        return data


def atomic_update_text(
    path: Path,
    updater: Callable[[str], str],
    *,
    root: Path | None = None,
) -> str:
    """Atomically read-modify-write a text file, returning the new text."""
    p = Path(path)
    with _file_lock(p):
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            text = ""
        text = updater(text)
        if not isinstance(text, str):
            raise TypeError("atomic_update_text updater must return a str")
        p.parent.mkdir(parents=True, exist_ok=True)
        fileio.write_bytes(p, text.encode("utf-8"))
        return text


# ------------------------------------------------------- maintenance lifecycle
#
# `prune` / `reset` / `update` / `restore` / `undo` / `checkout` are
# destructive enough that they must refuse while another session is actively
# writing the project. The daemon (WP-4006) and the CLI both call
# `check_maintenance_safe`; a daemon that tracks live sessions registers a
# quiesce hook here so maintenance drains before it proceeds.

_QUIESCE_HOOKS: list[Callable[[Path, str], Any]] = []
_QUIESCE_LOCK = threading.Lock()


def register_quiesce_hook(hook) -> None:
    """Register `hook(root, action)` returning active session ids (or any
    truthy "active" description); maintenance waits for it to go idle."""
    with _QUIESCE_LOCK:
        if hook not in _QUIESCE_HOOKS:
            _QUIESCE_HOOKS.append(hook)


def unregister_quiesce_hook(hook) -> None:
    with _QUIESCE_LOCK:
        if hook in _QUIESCE_HOOKS:
            _QUIESCE_HOOKS.remove(hook)


def _hook_active_sessions(result) -> list[str]:
    if result is None or result is False:
        return []
    if isinstance(result, str):
        return [result] if result.strip() else []
    if isinstance(result, (list, tuple, set)):
        return [str(s) for s in result if s]
    if isinstance(result, dict):
        active = result.get("active") or result.get("sessions") or []
        if isinstance(active, (list, tuple, set)):
            return [str(s) for s in active if s]
        if active:
            return [str(active)]
        return []
    return [str(result)] if result else []


def check_maintenance_safe(
    root: Path | str,
    *,
    action: str = "",
    drain_timeout: float = 0.0,
    force: bool = False,
) -> tuple[bool, str]:
    """`(ok, reason)` for whether a destructive maintenance action may run.

    `ok` is False while a registered quiesce hook reports active sessions or
    the project run lock is held by another live run. `drain_timeout > 0`
    polls until the sessions drain or the deadline passes; `force` always
    answers `(True, "")`.
    """
    if force:
        return True, ""
    root = Path(root)
    deadline = (None if drain_timeout <= 0
                else time.monotonic() + float(drain_timeout))
    while True:
        reason = _maintenance_unsafe_reason(root, action)
        if reason == "":
            return True, ""
        if deadline is None or time.monotonic() >= deadline:
            return False, reason
        time.sleep(min(0.1, deadline - time.monotonic()))


def _maintenance_unsafe_reason(root: Path, action: str) -> str:
    with _QUIESCE_LOCK:
        hooks = list(_QUIESCE_HOOKS)
    for hook in hooks:
        try:
            active = _hook_active_sessions(hook(root, action))
        except Exception:
            continue
        if active:
            verb = action or "maintain"
            return (f"cannot {verb} now: {len(active)} active session(s) "
                    f"({', '.join(active[:4])})")
    if _DEFAULT_LOCK_MANAGER.is_locked(root):
        verb = action or "maintain"
        return f"cannot {verb} now: another pedacito run is using this project"
    # D15: a daemon in a separate process may hold live sessions for this
    # root. Ask it to quiesce over loopback with a short timeout. When no
    # daemon is running (the offline default) this is a cheap missing-file
    # read and the maintenance action proceeds immediately.
    try:
        from . import daemon as daemon_mod
        info = daemon_mod.read_daemon_info()
        if info is not None and int(info.get("pid") or 0) != os.getpid():
            status, body = daemon_mod.daemon_request(
                "POST", "/v1/quiesce",
                {"root": str(root), "timeout_s": 0.2}, timeout=1.0)
            if status == 200 and body.get("ok") is False:
                verb = action or "maintain"
                reason = str(body.get("reason") or "daemon sessions are active")
                return f"cannot {verb} now: {reason}"
    except Exception:
        pass
    return ""
