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
`_job_wrap`), so every descendant it later spawns inherits membership. When a
job cannot be created -- CI runners wrap every step in a job that refuses
nested assignment, and older Windows refuses it outright -- `stop` falls back
to `taskkill /T` plus a process-snapshot walk that reaches descendants even
after their leader exited (`_windows_descendants`). This is deliberately a
small, local-first manager: no daemon, no sockets -- just enough to start a
server, find its printed URL, read its log, and stop it later.
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
_CHILDREN: dict[int, subprocess.Popen] = {}


def _forget_child(pid: int) -> None:
    _CHILDREN.pop(int(pid or 0), None)


def _alive(pid: int) -> bool:
    pid = int(pid or 0)
    child = _CHILDREN.get(pid)
    if child is not None:
        try:
            if child.poll() is not None:      # reaps the zombie
                _CHILDREN.pop(pid, None)
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
    name = _unique_name(root, base)
    log = _log_path(root, name)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = safety.child_env(cfg)
    p = None
    job = job_note = ""
    born = 0.0
    try:
        with log.open("ab") as fh:
            p = proc.spawn(argv, cwd=root, env=env,
                           stdout=fh, stderr=subprocess.STDOUT,
                           suspended=(os.name == "nt"))
        _CHILDREN[p.pid] = p
        # Windows: the child is frozen, so it cannot spawn anything before it
        # is in the job -- membership is inherited at CREATION, and a launcher
        # that runs first can leave its descendants outside the job forever.
        if os.name == "nt":
            k = _kernel32()
            if k is not None:
                ft = _process_created(k, p.pid)
                if ft:
                    born = ft / 10_000_000 - 11644473600
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
            _CHILDREN[p.pid] = p
            k = _kernel32()
            if k is not None:
                ft = _process_created(k, p.pid)
                if ft:
                    born = ft / 10_000_000 - 11644473600
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
           "started": time.time(), "status": "running"}
    # Give a dev server a moment to print its URL. Never block a run for long:
    # `wait` is a ceiling, and a command that exits is reported immediately.
    deadline = time.monotonic() + max(0.0, float(wait or 0))
    while time.monotonic() < deadline:
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
        if rec.get("status") == "running" and not _alive(int(rec.get("pid") or 0)):
            rec["status"] = "exited"
            stale = True
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
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"ERROR: could not read {rec['log']}: {e}"
    tail = max(200, int(tail or 4000))
    if len(text) > tail:
        text = f"[... earlier output omitted; last {tail} chars ...]\n" + text[-tail:]
    state = rec.get("status", "?")
    if state == "running" and not _alive(int(rec.get("pid") or 0)):
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
# spawns inherits membership, and TerminateJobObject reaps them all -- from any
# pedacito process, because the object lives as long as it has members.
_JOB_TERMINATE = 0x0008
_SYNCHRONIZE = 0x00100000
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001

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


def _job_wrap(pid: int, name: str) -> tuple[str, str]:
    """Put the just-spawned (suspended) child in its own named job.

    Returns (job_name, "") on success, ("", reason) when any step failed --
    the reason lands in the registry record so a stop that cannot reap the
    tree says WHY instead of failing silently. Every step is best effort:
    without a job, `stop` still gets `taskkill /T` plus the orphan walk.
    """
    k = _kernel32()
    if k is None or pid <= 0:
        return "", ""
    # Local\ rather than Global\: the global namespace needs a privilege we may
    # not hold, and a server started in one logon session is only ever stopped
    # from that same session.
    job = f"Local\\pedacito-{_safe_name(name)}-{pid}-{uuid.uuid4().hex[:8]}"
    hjob = hproc = None
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
        return job, ""
    except Exception as e:                  # pragma: no cover - best effort
        return "", f"job setup raised {type(e).__name__}: {str(e)[:80]}"
    finally:
        # Dropping our own handles is safe: membership alone keeps the kernel
        # object (and its name) alive, and the leader is a member by now.
        _close_handle(k, hproc)
        _close_handle(k, hjob)


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
    """Terminate every member of a job object created by `_job_wrap`."""
    k = _kernel32()
    if k is None or not job:
        return False
    handle = None
    try:
        handle = k.OpenJobObjectW(_JOB_TERMINATE | _SYNCHRONIZE, False, job)
        if not handle:
            return False            # gone with its last member, or other session
        return bool(k.TerminateJobObject(handle, 1))
    except Exception:                   # pragma: no cover - best effort
        return False
    finally:
        _close_handle(k, handle)


def _process_created(k, pid: int) -> int:
    """A process's creation time as 100ns intervals since 1601 (0 unknown)."""
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    h = k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return 0
    try:
        born = wintypes.FILETIME()
        rest = wintypes.FILETIME()
        if not k.GetProcessTimes(h, ctypes.byref(born), ctypes.byref(rest),
                                 ctypes.byref(rest), ctypes.byref(rest)):
            return 0
        return (born.dwHighDateTime << 32) | born.dwLowDateTime
    except Exception:                       # pragma: no cover - best effort
        return 0
    finally:
        _close_handle(k, h)


def _windows_descendants(pid: int, since: float) -> list[int]:
    """Descendants of `pid` from the process snapshot, orphans included.

    Windows keeps a process's ParentProcessId after its parent dies (there is
    no reparenting), so the chain outlives the leader -- unlike `taskkill /T`,
    which only walks LIVE parents. This is the fallback for when no job object
    could be created: a CI runner wraps every step in a job that refuses
    nested assignment, so the registry's own job is exactly what is missing
    there, and the launcher that exited first is exactly the case it leaves.

    A pid can be reused after its holder exits, so a candidate only counts as
    a child when it was created AFTER its parent. For the (possibly dead)
    root, `since` -- the leader's creation time recorded at start -- stands in
    for the parent's creation time. Best effort throughout: an unreadable
    snapshot yields no descendants, never an exception.
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
        # FILETIME epoch (1601) vs Unix epoch (1970), in 100ns intervals.
        since_ft = int((max(0.0, float(since or 0.0)) + 11644473600)
                       * 10_000_000)
        born: dict[int, int] = {}

        def created(p: int) -> int:
            if p not in born:
                born[p] = _process_created(k, p)
            return born[p]

        out: list[int] = []
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
                out.append(child)
                queue.append((child, child_born or parent_born))
        return out
    except Exception:                       # pragma: no cover - best effort
        return []


def _terminate_pid(pid: int, job: str = "", started: float = 0.0,
                   born: float = 0.0) -> None:
    if os.name == "nt":
        # The job reaches the descendants its leader already orphaned; taskkill
        # is the fallback for records written without one.
        if job and _job_kill(job):
            return
        if pid > 0:
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                               capture_output=True, timeout=10, check=False)
            except (OSError, subprocess.SubprocessError):
                pass
            # taskkill /T only walks LIVE parents: when the leader already
            # exited (a launcher that spawned the real server and left), the
            # orphans need the snapshot walk. `born` is the leader's creation
            # time; `started` is the fallback for records written before it.
            for dpid in _windows_descendants(pid, born or started):
                try:
                    subprocess.run(["taskkill", "/F", "/PID", str(dpid)],
                                   capture_output=True, timeout=10,
                                   check=False)
                except (OSError, subprocess.SubprocessError):
                    pass
        return
    if pid <= 0:
        return
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
            return
        time.sleep(0.05)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def stop(root, name: str) -> str:
    """Stop one background process and its descendants."""
    root = Path(root).resolve()
    rec = _find(root, name)
    if rec is None:
        known = ", ".join(str(r.get("name")) for r in list_procs(root)) or "(none)"
        return f"ERROR: no background process named '{name}'. Known: {known}"
    pid = int(rec.get("pid") or 0)
    _terminate_pid(pid, str(rec.get("job") or ""),
                   started=float(rec.get("started") or 0.0),
                   born=float(rec.get("born") or 0.0))
    _forget_child(pid)
    rec["status"] = "stopped"
    rec["stopped"] = time.time()
    _write_record(root, rec)
    return f"stopped {rec.get('name')} (pid {pid})"


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
