"""Background process registry: keep a dev server alive across tasks.

    from . import processes

    rec = processes.start(root, "npm run dev", name="web")
    processes.list_procs(root)
    processes.logs(root, "web")
    processes.stop(root, "web")

The registry lives under `.pedacito/procs/`: one JSON record per process and a
`.log` file with its combined stdout/stderr. PIDs are process-group leaders, so
`stop` reaps descendants too; on Windows, which has no process groups, the child
is spawned SUSPENDED and put in a named job object before it runs (see
`_job_wrap`), so every descendant it later spawns inherits membership. The job
handle is retained for the server's lifetime (`_JOB_HANDLES`): a named job's
name disappears the instant its last handle closes -- even while members still
run -- so `_job_kill` terminates through the retained handle, or reopens it by
name from another pedacito process while this one still holds one. When a job
cannot be created -- CI runners wrap every step in a job that refuses nested
assignment -- `stop` falls back to a live snapshot walk (`_windows_descendants`)
and the descendant identities `start` captured while the launcher chain was
still alive, each killed only after its exact creation time is re-validated on
the terminating handle. This is deliberately a small, local-first
manager: no daemon, no sockets -- just enough to start a server, find its
printed URL, read its log, and stop it later.
"""

from __future__ import annotations

# ctypes at module level, NOT lazily inside the Windows helpers: importing it
# while a test has patched os.name to "nt" would run ctypes' own Windows
# branch on Linux. Only the WinDLL itself stays lazy (see `_kernel32`).
import ctypes
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from ctypes import wintypes
from pathlib import Path
from urllib.parse import urlsplit

from . import browser, fileio, proc, safety, security
from .index import state_dir

PROC_DIR = "procs"
_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
# How much of the log a URL re-scan reads: enough for a banner printed long
# after `serve_wait` expired, bounded so a server that has been chatty for
# hours is never slurped whole by a GUI that polls the registry.
_LOG_TAIL_BYTES = 16384


def _dir(root) -> Path:
    return state_dir(Path(root), PROC_DIR)


def _safe_name(name: str) -> str:
    name = _NAME_RE.sub("-", (name or "").strip()).strip("-._")
    return name[:60] or "serve"


def _unique_name(root: Path, base: str) -> str:
    names = {p.stem for p in _dir(root).glob("*.json")}
    name, n = _safe_name(base), 2
    while name in names:
        name = f"{_safe_name(base)}-{n}"
        n += 1
    return name


def _record_path(root: Path, name: str) -> Path:
    return _dir(root) / f"{_safe_name(name)}.json"


def _log_path(root: Path, name: str) -> Path:
    return _dir(root) / f"{_safe_name(name)}.log"


def _read_record(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_record(root: Path, rec: dict) -> None:
    path = _record_path(root, str(rec.get("name") or "serve"))
    path.parent.mkdir(parents=True, exist_ok=True)
    fileio.write_bytes(path, json.dumps(rec, indent=1).encode("utf-8"))


# Popen objects for children THIS process started. Retaining them is what lets
# `stop` reap the child: `os.kill(pid, 0)` reports a zombie as alive, so a
# killed server stayed "running" in the registry (and `stop` burned its full
# 3-second grace period) until an unrelated Popen happened to clean up. A
# server started by one CLI process and stopped by another has no entry here;
# its parent exits, so the OS reparents and reaps it instead.
_CHILDREN_LOCK = threading.Lock()
_CHILDREN: dict[int, subprocess.Popen] = {}
_START_LOCK = threading.Lock()
# Live kernel handles to the job objects THIS process created, keyed by job
# name. A named kernel object's name is removed from the namespace the moment
# its last handle is closed -- even while member processes still run -- so
# dropping the handle also drops the name, and a later OpenJobObjectW finds
# nothing (ERROR_FILE_NOT_FOUND). Retaining the handle is what keeps the job
# terminable: `_job_kill` terminates through it directly (same process), and
# the still-registered name lets another pedacito process open it too -- but
# only while THIS process stays alive to hold the handle.
_JOB_LOCK = threading.Lock()
_JOB_HANDLES: dict[str, int] = {}


def _current_session_id() -> str:
    """The bound session id, or '' when no session is bound."""
    try:
        from . import session_ctx
        sess = session_ctx.current_session()
        return str(sess.session_id) if sess is not None else ""
    except Exception:
        return ""


def _stop_quiet(root, name: str) -> None:
    """`stop`, but never raises (teardown cleanup must stay best effort)."""
    try:
        stop(root, name)
    except Exception:
        pass


def _register_dev_server(root, name: str) -> str | None:
    """Register `stop(root, name)` as the session's dev_server cleanup."""
    try:
        from . import session_ctx
        sess = session_ctx.current_session()
    except Exception:
        return None
    if sess is None or sess.resource_registry is None:
        return None
    try:
        return sess.resource_registry.register(
            "dev_server", str(name), lambda: _stop_quiet(root, name))
    except Exception:
        return None


def _unregister_dev_server(resource_id: str | None) -> None:
    if not resource_id:
        return
    try:
        from . import session_ctx
        sess = session_ctx.current_session()
        if sess is not None and sess.resource_registry is not None:
            sess.resource_registry.unregister(resource_id)
    except Exception:
        pass


def _forget_child(pid: int) -> None:
    with _CHILDREN_LOCK:
        _CHILDREN.pop(int(pid or 0), None)


def _get_child(pid: int) -> subprocess.Popen | None:
    with _CHILDREN_LOCK:
        return _CHILDREN.get(int(pid or 0))


def _linux_proc_start_ticks(pid: int) -> int:
    """Process start-time ticks from Linux /proc/<pid>/stat, or 0 if unavailable.

    Splits on the LAST ')' because the comm field in parens can contain spaces
    and ')' characters. Field 22 (1-indexed) is starttime -> index 19 after ')'.
    """
    if os.name == "nt" or sys.platform != "linux" or pid <= 0:
        return 0
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8",
                                                  errors="replace")
    except OSError:
        return 0
    rparen = raw.rfind(")")
    if rparen < 0:
        return 0
    fields = raw[rparen + 1:].split()
    if len(fields) <= 19:
        return 0
    try:
        return int(fields[19])
    except ValueError:
        return 0


def _pid_matches_born(pid: int, born: float) -> bool:
    """True unless Linux /proc/<pid>/stat proves `pid` was recycled."""
    if os.name != "nt" and sys.platform == "linux" and born:
        cur_born = _linux_proc_start_ticks(pid)
        if cur_born and cur_born != int(born):
            return False
    return True


def _alive(pid: int) -> bool:
    pid = int(pid or 0)
    child = _get_child(pid)
    if child is not None:
        try:
            if child.poll() is not None:      # reaps the zombie
                _forget_child(pid)
                return False
        except (OSError, ValueError):
            pass
    from .workspace import _pid_alive
    return _pid_alive(pid)


def _argv(command: str) -> list[str]:
    if os.name != "nt":
        return shlex.split(command)
    return [p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'"
            else p for p in shlex.split(command, posix=False)]


def _default_name(command: str) -> str:
    try:
        argv = _argv(command)
    except ValueError:
        argv = [command]
    if not argv:
        return "serve"
    head = Path(argv[0]).stem.lower()
    if head in ("python", "python3", "node", "npm", "pnpm", "yarn", "cargo",
                "go", "deno", "bun", "uvicorn", "flask", "bundle") and len(argv) > 1:
        for arg in argv[1:]:
            if arg and not arg.startswith("-"):
                return Path(arg).stem.lower()
    return head or "serve"


def start(root, command, *, name: str = "", cfg=None,
          wait: float = 3.0) -> dict:
    """Start `command` (a string or argv list) in the background. Returns its
    registry record.

    The record is written even when the command exits immediately; `status`
    then reads `exited`. A name already used by an existing record is suffixed.
    """
    root = Path(root).resolve()
    if isinstance(command, (list, tuple)):
        argv = [str(x) for x in command]
        command_text = shlex.join(argv) if argv else ""
    else:
        command_text = str(command or "").strip()
        argv = _argv(command_text) if command_text else []
    if not argv:
        raise ValueError("empty command")
    base = name or _default_name(command_text)
    with _START_LOCK:
        name = _unique_name(root, base)
        log = _log_path(root, name)
        log.parent.mkdir(parents=True, exist_ok=True)
        # Reserve the record on disk under the lock before spawning / waiting
        # so concurrent start() calls cannot collide on the same name.
        _write_record(root, {
            "name": name, "command": command_text, "pid": 0,
            "cwd": str(root), "log": str(log), "url": "", "job": "",
            "started": time.time(), "status": "starting",
            "session_id": _current_session_id(),
        })
    env = safety.child_env(cfg)
    p = None
    job = job_note = ""
    born = 0.0
    born_ft = 0
    try:
        with log.open("ab") as fh:
            p = proc.spawn(argv, cwd=root, env=env,
                           stdout=fh, stderr=subprocess.STDOUT,
                           suspended=(os.name == "nt"))
        with _CHILDREN_LOCK:
            _CHILDREN[p.pid] = p
        # Windows: the child is frozen, so it cannot spawn anything before it
        # is in the job -- membership is inherited at CREATION, and a launcher
        # that runs first can leave its descendants outside the job forever.
        if os.name == "nt":
            k = _kernel32()
            if k is not None:
                ft = _process_created(k, p.pid)
                if ft:
                    born_ft = ft
                    born = _ft_to_unix(ft)
        elif sys.platform == "linux":
            born = _linux_proc_start_ticks(p.pid)
        job, job_note = _job_wrap(p.pid, name)
        if not _resume_child(p):
            # Could not resume (exotic): a frozen server is worse than a racy
            # job wrap -- respawn running, and wrap best effort.
            _forget_child(p.pid)
            try:
                p.kill()
                p.wait(timeout=5)
            except (OSError, subprocess.SubprocessError):
                pass
            with log.open("ab") as fh:
                p = proc.spawn(argv, cwd=root, env=env,
                               stdout=fh, stderr=subprocess.STDOUT)
            with _CHILDREN_LOCK:
                _CHILDREN[p.pid] = p
            k = _kernel32()
            if k is not None:
                ft = _process_created(k, p.pid)
                if ft:
                    born_ft = ft
                    born = _ft_to_unix(ft)
            job, job_note = _job_wrap(p.pid, name)
    except (OSError, ValueError) as e:
        rec = {"name": name, "command": command_text, "pid": 0,
               "cwd": str(root), "log": str(log), "url": "", "job": "",
               "started": time.time(), "status": "error",
               "error": str(e)[:400]}
        _write_record(root, rec)
        return rec
    rec = {"name": name, "command": command_text, "pid": p.pid,
           "cwd": str(root), "log": str(log), "url": "", "job": job,
           "job_note": job_note, "born": born,
           "started": time.time(), "status": "running",
           "session_id": _current_session_id()}
    if born_ft:
        rec["born_ft"] = born_ft
    # Windows fallback: retain descendant identities while the launcher chain
    # is still alive, so `stop` can reach them after the intermediates exit.
    win_desc: list[list[int, int]] = _merge_descendants([], p.pid, born_ft)
    _write_record(root, rec)
    resource_id = _register_dev_server(root, name)
    if resource_id:
        rec["resource_id"] = resource_id
    # Give a dev server a moment to print its URL. Never block a run for long:
    # `wait` is a ceiling, and a command that exits is reported immediately.
    deadline = time.monotonic() + max(0.0, float(wait or 0))
    while time.monotonic() < deadline:
        win_desc = _merge_descendants(win_desc, p.pid, born_ft)
        if p.poll() is not None:
            rec["status"] = "exited"
            rec["returncode"] = p.returncode
            _forget_child(p.pid)
            break
        text = _log_tail(rec, root)
        url = browser.find_server_url(text)
        if url:
            rec["url"] = url
            break
        time.sleep(0.1)
    win_desc = _merge_descendants(win_desc, p.pid, born_ft)
    if win_desc:
        rec["descendants"] = win_desc
    _write_record(root, rec)
    return rec


def _log_tail(rec: dict, root=None) -> str:
    """The last few kilobytes of a record's log, without reading the whole file.

    `root` bounds the read: the record's `log` path must resolve inside
    `.pedacito/procs/`, so a hand-edited record cannot point the reader at an
    arbitrary file (a symlink escaping the directory included)."""
    raw = str(rec.get("log") or "")
    if not raw:
        return ""
    path = Path(raw)
    if root is not None:
        try:
            security.resolve_safe_path(_dir(root), path)
        except (security.SecurityError, OSError, ValueError, RuntimeError):
            return ""
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - _LOG_TAIL_BYTES))
            data = fh.read()
    except (OSError, ValueError):
        return ""
    return data.decode("utf-8", errors="replace")


def _refresh_url(rec: dict, root=None) -> bool:
    """Recover a URL the server printed after `start` stopped waiting.

    `serve_wait` bounds how long a run may block, not how long the server has
    to boot: `npm run dev` on a cold cache (or a dev server held up by a slow
    reverse-DNS lookup inside its own framework) happily prints its URL after
    the budget is gone. The record used to stay url-less forever, so the
    registry, the GUI and the `serve` observation all kept saying nothing even
    though the address was sitting in the log. Find it on the next look.
    """
    if rec.get("url") or rec.get("status") == "error":
        return False
    url = browser.find_server_url(_log_tail(rec, root))
    if not url:
        return False
    rec["url"] = url
    return True


def list_procs(root) -> list[dict]:
    """Registry records, newest first, with dead pids marked `exited`."""
    root = Path(root).resolve()
    out: list[dict] = []
    for path in _dir(root).glob("*.json"):
        rec = _read_record(path)
        if not rec:
            continue
        stale = False
        pid = int(rec.get("pid") or 0)
        born = float(rec.get("born") or 0.0)
        if rec.get("status") == "running" and not (
                _pid_matches_born(pid, born) and _alive(pid)):
            rec["status"] = "exited"
            stale = True
            # The leader exiting is exactly when orphans may still be members:
            # release the job handle only once the job is provably empty, so a
            # later `stop` can still terminate the survivors through it.
            job = str(rec.get("job") or "")
            if _job_active_count(job) == 0:
                _forget_job(job)
        if _refresh_url(rec, root):
            stale = True
        if stale:
            _write_record(root, rec)
        out.append(rec)
    out.sort(key=lambda r: float(r.get("started") or 0), reverse=True)
    return out


def _find(root: Path, name: str) -> dict | None:
    root = Path(root).resolve()
    target = _safe_name(name)
    for rec in list_procs(root):
        if _safe_name(str(rec.get("name") or "")) == target:
            return rec
    return None


def logs(root, name: str, tail: int = 4000) -> str:
    """The process's combined output, capped to its last `tail` characters."""
    root = Path(root).resolve()
    rec = _find(root, name)
    if rec is None:
        known = ", ".join(str(r.get("name")) for r in list_procs(root)) or "(none)"
        return f"ERROR: no background process named '{name}'. Known: {known}"
    log_path = Path(str(rec.get("log") or ""))
    try:
        security.resolve_safe_path(_dir(root), log_path)
    except (security.SecurityError, OSError, ValueError, RuntimeError):
        return (f"ERROR: could not read {rec.get('log')}: "
                f"outside the process log directory")
    tail = max(200, int(tail or 4000))
    # Bounded read from EOF: a UTF-8 character is at most 4 bytes, so reading
    # 4 * tail bytes (+ margin) guarantees at least `tail` decoded characters
    # without slurping a multi-gigabyte log file into memory.
    max_bytes = tail * 4 + 64
    truncated = False
    try:
        with log_path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            if size > max_bytes:
                fh.seek(size - max_bytes)
                truncated = True
            else:
                fh.seek(0)
            data = fh.read()
    except OSError as e:
        return f"ERROR: could not read {rec['log']}: {e}"
    text = data.decode("utf-8", errors="replace")
    if truncated or len(text) > tail:
        text = f"[... earlier output omitted; last {tail} chars ...]\n" + text[-tail:]
    state = rec.get("status", "?")
    pid = int(rec.get("pid") or 0)
    born = float(rec.get("born") or 0.0)
    if state == "running" and not (_pid_matches_born(pid, born) and _alive(pid)):
        state = "exited"
    head = f"{rec.get('name')} [{state}, pid {rec.get('pid')}]"
    if rec.get("url"):
        head += f"  {rec['url']}"
    return head + "\n" + (text.strip() or "(no output yet)")


def _group_alive(pgid: int) -> bool:
    """True while a POSIX process group still has members."""
    if pgid <= 0 or os.name == "nt":
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


# --- Windows: a job object is the process group we do not have --------------
# POSIX spawns each server as a session leader, so `killpg` reaches every
# descendant -- including one whose leader already exited. Windows has no such
# id: `taskkill /T` walks a LIVE parent-child chain, so a launcher that exits
# first (`npm run dev` spawning the real server, a wrapper script, a re-exec)
# leaves its child holding the port forever while `stop` reports success. A
# named kernel job object fixes exactly that: every process the server later
# spawns inherits membership, and TerminateJobObject reaps them all. The job's
# name is gone the moment its last handle closes (even while members run), so
# the handle is retained for the server's lifetime -- see `_JOB_HANDLES`.
_JOB_TERMINATE = 0x0008
_SYNCHRONIZE = 0x00100000
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

_K32: object | None = None
_K32_STATE = "unloaded"


def _kernel32():
    """kernel32's job-object entry points, wired up once; None off Windows.

    Behind a function (not a module-level WinDLL) so this file stays importable
    and testable on POSIX, and so tests can substitute a fake.
    """
    global _K32, _K32_STATE
    if os.name != "nt":
        return None
    if _K32_STATE == "unloaded":
        _K32_STATE = "failed"
        try:
            k = ctypes.WinDLL("kernel32", use_last_error=True)
            k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            k.CreateJobObjectW.restype = wintypes.HANDLE
            k.OpenJobObjectW.argtypes = [wintypes.DWORD, wintypes.BOOL,
                                         wintypes.LPCWSTR]
            k.OpenJobObjectW.restype = wintypes.HANDLE
            k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE,
                                                   wintypes.HANDLE]
            k.AssignProcessToJobObject.restype = wintypes.BOOL
            k.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
            k.TerminateJobObject.restype = wintypes.BOOL
            k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL,
                                      wintypes.DWORD]
            k.OpenProcess.restype = wintypes.HANDLE
            k.CloseHandle.argtypes = [wintypes.HANDLE]
            k.CloseHandle.restype = wintypes.BOOL
            # Terminating a verified process by handle (no taskkill /T, which
            # could reach descendants we never validated) and counting a job's
            # live members so a dead leader's handle is only released once the
            # job is provably empty.
            k.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
            k.TerminateProcess.restype = wintypes.BOOL
            k.QueryInformationJobObject.argtypes = [
                wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                ctypes.c_void_p]
            k.QueryInformationJobObject.restype = wintypes.BOOL
            # Resuming the suspended child: Popen closes the initial thread
            # handle, so the thread is found in the system snapshot by owner.
            k.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD,
                                                   wintypes.DWORD]
            k.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
            k.Thread32First.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
            k.Thread32First.restype = wintypes.BOOL
            k.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
            k.Thread32Next.restype = wintypes.BOOL
            k.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL,
                                     wintypes.DWORD]
            k.OpenThread.restype = wintypes.HANDLE
            k.ResumeThread.argtypes = [wintypes.HANDLE]
            k.ResumeThread.restype = wintypes.DWORD
            # The orphan walk: parent pids outlive their parents on Windows.
            k.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
            k.Process32FirstW.restype = wintypes.BOOL
            k.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
            k.Process32NextW.restype = wintypes.BOOL
            k.GetProcessTimes.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME),
                ctypes.POINTER(wintypes.FILETIME),
                ctypes.POINTER(wintypes.FILETIME),
                ctypes.POINTER(wintypes.FILETIME)]
            k.GetProcessTimes.restype = wintypes.BOOL
            _K32, _K32_STATE = k, "loaded"
        except Exception:                   # pragma: no cover - no WinDLL
            _K32 = None
    return _K32


def _close_handle(k, handle) -> None:
    try:
        if handle:
            k.CloseHandle(handle)
    except Exception:                       # pragma: no cover - best effort
        pass


def _last_error(k) -> int:
    """The Win32 error behind the call that just failed (0 off Windows)."""
    try:
        return int(ctypes.get_last_error())
    except Exception:                       # pragma: no cover - best effort
        return 0


_WIN_STRUCTS: tuple | None = None


def _win_structs() -> tuple:
    """The snapshot entry structures, built once. Pure ctypes, so they can
    be constructed on any platform -- the fake-kernel32 tests rely on that."""
    global _WIN_STRUCTS
    if _WIN_STRUCTS is None:
        class THREADENTRY32(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD),
                        ("cntUsage", wintypes.DWORD),
                        ("th32ThreadID", wintypes.DWORD),
                        ("th32OwnerProcessID", wintypes.DWORD),
                        ("tpBasePri", ctypes.c_long),
                        ("tpDeltaPri", ctypes.c_long),
                        ("dwFlags", wintypes.DWORD)]

        class PROCESSENTRY32W(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD),
                        ("cntUsage", wintypes.DWORD),
                        ("th32ProcessID", wintypes.DWORD),
                        ("th32DefaultHeapID", ctypes.c_size_t),
                        ("th32ModuleID", wintypes.DWORD),
                        ("cntThreads", wintypes.DWORD),
                        ("th32ParentProcessID", wintypes.DWORD),
                        ("pcPriClassBase", ctypes.c_long),
                        ("dwFlags", wintypes.DWORD),
                        ("szExeFile", ctypes.c_wchar * 260)]

        _WIN_STRUCTS = (THREADENTRY32, PROCESSENTRY32W)
    return _WIN_STRUCTS


def _remember_job(name: str, handle: int) -> None:
    """Take ownership of a live job handle so `_job_kill` can terminate it."""
    with _JOB_LOCK:
        _JOB_HANDLES[name] = int(handle)


def _take_job(name: str) -> int:
    """Remove and return this process's live handle for `name`, or 0."""
    with _JOB_LOCK:
        return _JOB_HANDLES.pop(name, 0)


def _forget_job(name: str) -> None:
    """Close and drop the live handle for `name` (server stopped or exited)."""
    if not name:
        return
    with _JOB_LOCK:
        handle = _JOB_HANDLES.pop(name, 0)
    if handle:
        _close_handle(_kernel32(), handle)


def _ft_to_unix(ft: int) -> float:
    """FILETIME (100ns since 1601) -> Unix seconds; 0 stays 0 (unknown)."""
    return ft / 10_000_000 - 11644473600 if ft else 0.0


def _peek_job(name: str) -> int:
    """The live handle for `name` without taking ownership, or 0."""
    with _JOB_LOCK:
        return _JOB_HANDLES.get(name, 0)


_JOB_ACCOUNTING_STRUCT: type | None = None


def _job_accounting_struct():
    """JOBOBJECT_BASIC_ACCOUNTING_INFORMATION, built once (pure ctypes)."""
    global _JOB_ACCOUNTING_STRUCT
    if _JOB_ACCOUNTING_STRUCT is None:
        class JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("TotalUserTime", ctypes.c_longlong),
                ("TotalKernelTime", ctypes.c_longlong),
                ("ThisPeriodTotalUserTime", ctypes.c_longlong),
                ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
                ("TotalPageFaultCount", wintypes.DWORD),
                ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD),
                ("TotalTerminatedProcesses", wintypes.DWORD),
            ]
        _JOB_ACCOUNTING_STRUCT = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION
    return _JOB_ACCOUNTING_STRUCT


def _job_active_count(name: str) -> int:
    """Number of live processes in job `name`, or -1 when unknown.

    Peeked, not taken: the handle stays retained. -1 (handle gone or query
    failed) means the caller must keep the handle -- treating it as empty
    here would destroy a job whose members we could no longer reach.
    """
    if not name:
        return -1
    k = _kernel32()
    if k is None:
        return -1
    handle = _peek_job(name)
    if not handle:
        return -1
    info = _job_accounting_struct()()
    try:
        if not k.QueryInformationJobObject(handle, 1, ctypes.byref(info),
                                           ctypes.sizeof(info), None):
            return -1
        return int(info.ActiveProcesses)
    except Exception:                       # pragma: no cover - best effort
        return -1


def _pid_currently_alive(pid: int) -> bool:
    """True when a process with `pid` can be opened right now."""
    k = _kernel32()
    if k is None or pid <= 0:
        return False
    h = k.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return False
    _close_handle(k, h)
    return True


def _open_validated(pid: int, expected_ft: int) -> int:
    """Open a handle to `pid` only after proving it was born at `expected_ft`.

    The creation time is read on this SAME handle, so a returned handle is
    guaranteed to refer to the process whose identity matched. 0 when the pid
    cannot be opened or the identity does not match -- both fail closed. The
    caller owns the returned handle.
    """
    if not expected_ft or pid <= 0:
        return 0
    k = _kernel32()
    if k is None:
        return 0
    h = k.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION | _PROCESS_TERMINATE,
                      False, pid)
    if not h:
        return 0
    if _process_created_from_handle(k, h) != expected_ft:
        _close_handle(k, h)
        return 0
    return h


def _terminate_verified(pid: int, expected_ft: int) -> str:
    """Kill `pid` only after proving it is the process born at `expected_ft`.

    Opens ONE handle, reads the creation time on that SAME handle (so the
    check and the kill cannot straddle a pid-reuse window), compares EXACTLY,
    and terminates through that handle. Never `taskkill`/`/T`, which could
    reach descendants we never validated. Returns "reaped", "gone" (the pid
    no longer resolves to any process), or "unverified" (unknown birth, an
    unreadable creation time, a mismatch, or a failed kill -- all fail closed).
    """
    if not expected_ft or pid <= 0:
        return "unverified"
    h = _open_validated(pid, expected_ft)
    if not h:
        return "gone" if not _pid_currently_alive(pid) else "unverified"
    k = _kernel32()
    try:
        return "reaped" if k.TerminateProcess(h, 1) else "unverified"
    except Exception:                       # pragma: no cover - best effort
        return "unverified"
    finally:
        _close_handle(k, h)


def _reap_leader_and_tree(pid: int, born_ft: int, notes: list[str]) -> None:
    """Reap a validated leader and its live descendants, race-free.

    The leader is opened and validated FIRST; its tree is walked while that
    handle is still open (so the leader pid is still provably ours -- walking
    after the kill could see a pid reused by a stranger), and only then is the
    leader terminated, followed by each descendant re-validated on its own
    handle. When the leader cannot be validated the traversal is skipped and
    the caller's non-exhaustive note stands.
    """
    h = _open_validated(pid, born_ft)
    if not h:
        if _pid_currently_alive(pid):
            notes.append(f"leader pid {pid} not verified; left alone")
        return
    k = _kernel32()
    try:
        descendants = _windows_walk(pid, born_ft)  # while the leader is ours
        k.TerminateProcess(h, 1)
    finally:
        _close_handle(k, h)
    for dpid, dft in descendants:
        if _terminate_verified(dpid, dft) == "unverified":
            notes.append(f"descendant pid {dpid} not verified; left alone")


def _job_wrap(pid: int, name: str) -> tuple[str, str]:
    """Put the just-spawned (suspended) child in its own named job.

    Returns (job_name, "") on success, ("", reason) when any step failed --
    the reason lands in the registry record so a stop that cannot reap the
    tree says WHY instead of failing silently. Every step is best effort:
    without a job, `stop` still gets the validated captured-identity fallback
    plus the live snapshot walk.

    On success the job handle is RETAINED (registered in `_JOB_HANDLES`), not
    closed: a named job's name vanishes the moment its last handle closes --
    even while members still run -- so the old "close it, reopen it by name
    later" plan made `_job_kill` a no-op (ERROR_FILE_NOT_FOUND) and leaked the
    orphan. The leader is a member by the time we return, and its descendants
    inherit membership at creation.
    """
    k = _kernel32()
    if k is None or pid <= 0:
        return "", ""
    # Local\ rather than Global\: the global namespace needs a privilege we may
    # not hold, and a server started in one logon session is only ever stopped
    # from that same session.
    job = f"Local\\pedacito-{_safe_name(name)}-{pid}-{uuid.uuid4().hex[:8]}"
    hjob = hproc = None
    remembered = False
    try:
        hjob = k.CreateJobObjectW(None, job)
        if not hjob:
            return "", f"CreateJobObject failed (error {_last_error(k)})"
        hproc = k.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE,
                              False, pid)
        if not hproc:
            return "", f"OpenProcess failed (error {_last_error(k)})"
        if not k.AssignProcessToJobObject(hjob, hproc):
            err = _last_error(k)
            note = f"AssignProcessToJobObject failed (error {err})"
            if err == 5:                    # ERROR_ACCESS_DENIED
                note += ("; the child already belongs to a job that refuses "
                         "nested assignment (CI runners wrap every step in "
                         "one) -- stop falls back to the orphan walk")
            return "", note
        _remember_job(job, hjob)
        remembered = True
        return job, ""
    except Exception as e:                  # pragma: no cover - best effort
        return "", f"job setup raised {type(e).__name__}: {str(e)[:80]}"
    finally:
        # hjob ownership transfers to `_JOB_HANDLES` only on success; every
        # other path closes it here. The process handle is always transient.
        if not remembered:
            _close_handle(k, hjob)
        _close_handle(k, hproc)


def _resume_child(p) -> bool:
    """Resume a child spawned CREATE_SUSPENDED (no-op elsewhere).

    Popen closes the initial thread's handle, so the thread is found in the
    system snapshot by its owner pid; a just-created suspended child has
    exactly one. False when nothing could be resumed -- the caller respawns
    without suspension rather than leave a frozen server behind.
    """
    if os.name != "nt":
        return True
    k = _kernel32()
    if k is None:
        return False
    TH32CS_SNAPTHREAD = 0x00000004
    THREAD_SUSPEND_RESUME = 0x0002
    invalid = ctypes.c_void_p(-1).value
    threadentry32, _ = _win_structs()
    snap = k.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
    if not snap or snap == invalid:
        return False
    resumed = False
    try:
        te = threadentry32()
        te.dwSize = ctypes.sizeof(te)
        # Thread32First/Next have NO wide variant (THREADENTRY32 carries no
        # strings), unlike Process32FirstW/NextW -- don't "correct" these.
        ok = k.Thread32First(snap, ctypes.byref(te))
        while ok:
            if te.th32OwnerProcessID == p.pid:
                ht = k.OpenThread(THREAD_SUSPEND_RESUME, False,
                                  te.th32ThreadID)
                if ht:
                    if k.ResumeThread(ht) != 0xFFFFFFFF:
                        resumed = True
                    _close_handle(k, ht)
            ok = k.Thread32Next(snap, ctypes.byref(te))
    except Exception:                       # pragma: no cover - best effort
        return resumed
    finally:
        _close_handle(k, snap)
    return resumed


def _job_kill(job: str) -> bool:
    """Terminate every member of a job object created by `_job_wrap`.

    Prefers this process's retained handle (which keeps the job -- and its
    name -- alive); without one, opens it by name, which only works while the
    process that created it still holds its own handle. A returned handle is
    always closed here, so the registry never leaks a handle.
    """
    k = _kernel32()
    if k is None or not job:
        return False
    handle = _take_job(job)
    try:
        if not handle:
            handle = k.OpenJobObjectW(_JOB_TERMINATE | _SYNCHRONIZE, False, job)
            if not handle:
                # Name gone (the starter process exited) or another session.
                return False
        return bool(k.TerminateJobObject(handle, 1))
    except Exception:                   # pragma: no cover - best effort
        return False
    finally:
        _close_handle(k, handle)


def _process_created_from_handle(k, h) -> int:
    """Creation time (100ns since 1601) of the process `h` refers to; 0 unknown."""
    born = wintypes.FILETIME()
    rest = wintypes.FILETIME()
    if not k.GetProcessTimes(h, ctypes.byref(born), ctypes.byref(rest),
                             ctypes.byref(rest), ctypes.byref(rest)):
        return 0
    return (born.dwHighDateTime << 32) | born.dwLowDateTime


def _process_created(k, pid: int) -> int:
    """A process's creation time as 100ns intervals since 1601 (0 unknown)."""
    h = k.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return 0
    try:
        return _process_created_from_handle(k, h)
    except Exception:                       # pragma: no cover - best effort
        return 0
    finally:
        _close_handle(k, h)


def _windows_walk(pid: int, since_ft: int) -> list[tuple[int, int]]:
    """Descendants of `pid` as (child_pid, creation FILETIME) pairs.

    Windows keeps a process's ParentProcessId after its parent dies (there is
    no reparenting), so the chain outlives the leader -- unlike `taskkill /T`,
    which only walks LIVE parents. This is the fallback for when no job object
    could be created: a CI runner wraps every step in a job that refuses
    nested assignment, so the registry's own job is exactly what is missing
    there, and the launcher that exited first is exactly the case it leaves.

    The walk only reaches a descendant while the WHOLE intermediate chain is
    still alive: the snapshot holds live processes only, so a launcher that
    already exited hides everything below it. `start` therefore captures the
    identities it can while the chain is alive and retains them in the record;
    this function walks what is still reachable at the moment it is called.

    A pid can be reused after its holder exits, so a candidate only counts as
    a child when it was created AFTER its parent; descendants whose creation
    time cannot be read are still traversed (to reach deeper) but NOT returned
    -- an unverifiable identity must never be handed back as if it were safe.
    `since_ft` is the leader's creation FILETIME, standing in for the (dead)
    root's own time. Best effort throughout: an unreadable snapshot yields no
    descendants, never an exception.
    """
    k = _kernel32()
    if k is None or pid <= 0:
        return []
    TH32CS_SNAPPROCESS = 0x00000002
    invalid = ctypes.c_void_p(-1).value
    _, processentry32w = _win_structs()
    try:
        snap = k.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if not snap or snap == invalid:
            return []
        procs: list[tuple[int, int]] = []
        try:
            pe = processentry32w()
            pe.dwSize = ctypes.sizeof(pe)
            ok = k.Process32FirstW(snap, ctypes.byref(pe))
            while ok:
                procs.append((int(pe.th32ProcessID),
                              int(pe.th32ParentProcessID)))
                ok = k.Process32NextW(snap, ctypes.byref(pe))
        finally:
            _close_handle(k, snap)
        children: dict[int, list[int]] = {}
        for cpid, ppid in procs:
            children.setdefault(ppid, []).append(cpid)
        since_ft = int(since_ft or 0)
        born: dict[int, int] = {}

        def created(p: int) -> int:
            if p not in born:
                born[p] = _process_created(k, p)
            return born[p]

        out: list[tuple[int, int]] = []
        queue = [(pid, since_ft)]
        seen = {pid}
        while queue:
            parent, parent_born = queue.pop(0)
            for child in children.get(parent, ()):
                if child in seen:
                    continue
                seen.add(child)
                child_born = created(child)
                if child_born and parent_born and child_born < parent_born:
                    continue        # pid reuse: "child" predates its parent
                if child_born:
                    out.append((child, child_born))
                queue.append((child, child_born or parent_born))
        return out
    except Exception:                       # pragma: no cover - best effort
        return []


def _windows_descendants(pid: int, since_ft: int) -> list[int]:
    """Pids of `_windows_walk` -- see that function for the guarantees."""
    return [p for p, _ in _windows_walk(pid, since_ft)]


def _windows_descendant_identities(pid: int, since_ft: int) -> list[list[int, int]]:
    """`_windows_walk` as JSON-ready `[pid, FILETIME]` pairs for the registry."""
    return [[p, ft] for p, ft in _windows_walk(pid, since_ft)]


def _merge_descendants(acc: list[list[int, int]], pid: int,
                       born_ft: int) -> list[list[int, int]]:
    """Union the identities reachable from `pid` right now into `acc`.

    Called repeatedly while `start` waits, because the launcher chain forms
    over a few milliseconds and the snapshot only sees the part whose
    intermediates are still alive at the moment of each walk. The union is
    what survives when the intermediates later exit. No-op off Windows.
    """
    if os.name != "nt" or pid <= 0:
        return acc
    for entry in _windows_descendant_identities(pid, born_ft):
        if entry not in acc:
            acc.append(entry)
    return acc


def _terminate_pid(pid: int, job: str = "", started: float = 0.0,
                   born: float = 0.0, born_ft: int = 0,
                   win_descendants: list | None = None) -> str:
    """Terminate `pid` and its descendants; "" when clean, else a note.

    The note names what could not be verified or reaped, so `stop` can say
    "partial" instead of claiming a successful stop. Never kills a pid whose
    identity cannot be proven ours (exact FILETIME match on the terminating
    handle); `taskkill /T` is not used here because it could reach unchecked
    descendants.
    """
    if os.name == "nt":
        # The job reaps every member (including descendants spawned after
        # start); no identity check is needed -- membership IS the identity.
        if job and _job_kill(job):
            return ""
        notes: list[str] = []
        # Captured descendant identities (exact FILETIME), each validated on
        # the very handle that terminates it, so a recycled pid is never hit.
        for entry in (win_descendants or ()):
            try:
                dpid, dft = int(entry[0]), int(entry[1])
            except (TypeError, ValueError, IndexError):
                notes.append("an unreadable descendant identity was skipped")
                continue
            if _terminate_verified(dpid, dft) == "unverified":
                notes.append(f"descendant pid {dpid} not verified; left alone")
        # Leader, then its live tree -- gated on the leader's exact identity
        # and walked BEFORE the leader is terminated (see
        # `_reap_leader_and_tree`). Without `born_ft` the leader cannot be
        # proven ours, so nothing is killed by pid (old records fail closed).
        if pid > 0:
            if born_ft:
                _reap_leader_and_tree(pid, born_ft, notes)
            elif _pid_currently_alive(pid):
                notes.append(f"leader pid {pid} has no recorded birth time; "
                             "left alone")
            # Without a successful job termination the fallback is never
            # exhaustive: a descendant behind a dead intermediate (spawned
            # after start, never captured) cannot be proven absent, so this is
            # always reported partial rather than a clean stop.
            notes.append("job termination unavailable; the process tree was "
                         "not provably fully reaped")
        return "; ".join(notes)
    if pid <= 0:
        return ""
    # On Linux, verify start-time ticks when `born` was recorded so a recycled
    # PID is never killed. Legacy records without `born` keep existing behavior.
    if not _pid_matches_born(pid, born):
        return ""
    # `proc.spawn` makes the child a session/group leader, so its pid IS the
    # pgid. If the leader already exited, `getpgid(pid)` fails but orphaned
    # descendants still belong to that group: signal the group by the id we
    # started it with instead of giving up on the whole tree.
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, PermissionError, OSError):
        pgid = pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if not _alive(pid) and not _group_alive(pgid):
            return ""
        time.sleep(0.05)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    child = _get_child(pid)
    if child is not None:
        try:
            child.wait(timeout=2)
        except (OSError, subprocess.SubprocessError):
            pass
    return ""


def stop(root, name: str) -> str:
    """Stop one background process and its descendants.

    A clean stop reaps the child, drops the cleanup registration and records
    `stopped`. A partial stop (job termination unavailable and the tree not
    provably reaped) keeps the cleanup registration and child tracking so a
    later `stop` -- or session teardown -- can retry, and records
    `stop_incomplete` plus `stop_note`, never a clean `stopped`.
    """
    root = Path(root).resolve()
    rec = _find(root, name)
    if rec is None:
        known = ", ".join(str(r.get("name")) for r in list_procs(root)) or "(none)"
        return f"ERROR: no background process named '{name}'. Known: {known}"
    pid = int(rec.get("pid") or 0)
    kw: dict = {}
    if rec.get("born_ft"):
        kw["born_ft"] = int(rec["born_ft"])
    if rec.get("descendants"):
        kw["win_descendants"] = rec["descendants"]
    note = _terminate_pid(pid, str(rec.get("job") or ""),
                          started=float(rec.get("started") or 0.0),
                          born=float(rec.get("born") or 0.0), **kw)
    name = str(rec.get("name"))
    if note:
        # Partial/unverified: the tree was not provably reaped. Keep the
        # cleanup registration (resource_id) and child tracking so a retry can
        # re-attempt, and record a distinct incomplete status + note instead of
        # a clean `stopped` (no `stopped` timestamp either).
        rec["status"] = "stop_incomplete"
        rec["stop_note"] = note
        _write_record(root, rec)
        return f"stopped {name} (pid {pid}) [incomplete: {note}]"
    _unregister_dev_server(rec.get("resource_id"))
    child = _get_child(pid)
    if child is not None:
        try:
            child.wait(timeout=2)
        except (OSError, subprocess.SubprocessError):
            pass
    _forget_child(pid)
    rec["status"] = "stopped"
    rec["stopped"] = time.time()
    _write_record(root, rec)
    return f"stopped {name} (pid {pid})"


_BASE_SUFFIX_RE = re.compile(r"-\d+$")


def _proc_base_name(rec: dict) -> str:
    """Registry name without the `-2`/`-3` collision suffix `_unique_name` adds."""
    return _BASE_SUFFIX_RE.sub("", str(rec.get("name") or ""))


_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _url_key(url: str) -> tuple[str, int] | None:
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    host = parts.hostname
    if not host:
        return None
    # A restart may print a different loopback spelling of the same server.
    if host in _LOOPBACK_HOSTS:
        host = "localhost"
    port = parts.port or (443 if parts.scheme == "https" else 80)
    return (host, port)


def _same_server(a: dict, b: dict) -> bool:
    """Whether two registry records look like the same server: an identical
    command, the same host:port URL, or the same base registry name."""
    ca = " ".join(str(a.get("command") or "").split())
    cb = " ".join(str(b.get("command") or "").split())
    if ca and ca == cb:
        return True
    ka, kb = _url_key(str(a.get("url") or "")), _url_key(str(b.get("url") or ""))
    if ka and ka == kb:
        return True
    na, nb = _proc_base_name(a), _proc_base_name(b)
    return bool(na) and na == nb


def render_list(root, all: bool = False) -> str:
    """One line per registered process, or a hint when there are none.

    A stopped entry that a newer running instance of the same server
    supersedes is restart history, not live state: it is collapsed by
    default and counted in a footer. `all=True` shows every record.
    """
    recs = list_procs(root)
    if not recs:
        return ("No background processes. Start one with "
                "`pedacito serve PROJECT -- COMMAND`.")
    running = [r for r in recs if r.get("status") == "running"]
    lines, hidden = [], 0
    for rec in recs:
        if (not all and rec.get("status") == "stopped"
                and any(_same_server(rec, r)
                        and float(r.get("started") or 0)
                        >= float(rec.get("started") or 0)
                        for r in running)):
            hidden += 1
            continue
        extra = f"  {rec['url']}" if rec.get("url") else ""
        lines.append(f"{rec.get('name'):<20} [{rec.get('status', '?')}] "
                     f"pid {rec.get('pid')}{extra}  {rec.get('command', '')}")
    if hidden:
        lines.append(f"+ {hidden} superseded stopped entry(s) hidden "
                     "(use `pedacito procs --all` to show)")
    return "\n".join(lines)
