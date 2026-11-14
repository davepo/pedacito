"""Background process registry: keep a dev server alive across tasks.

    from . import processes

    rec = processes.start(root, "npm run dev", name="web")
    processes.list_procs(root)
    processes.logs(root, "web")
    processes.stop(root, "web")

The registry lives under `.pedacito/procs/`: one JSON record per process and a
`.log` file with its combined stdout/stderr. PIDs are process-group leaders, so
`stop` reaps descendants too; on Windows, which has no process groups, the child
is put in a named job object that plays the same role (see `_job_wrap`). This is
deliberately a small, local-first manager: no daemon, no sockets -- just enough
to start a server, find its printed URL, read its log, and stop it later.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import time
import uuid
from pathlib import Path

from . import browser, proc, safety
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
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    tmp.replace(path)


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
    try:
        with log.open("ab") as fh:
            p = proc.spawn(argv, cwd=root, env=env,
                           stdout=fh, stderr=subprocess.STDOUT)
    except (OSError, ValueError) as e:
        rec = {"name": name, "command": command_text, "pid": 0,
               "cwd": str(root), "log": str(log), "url": "", "job": "",
               "started": time.time(), "status": "error",
               "error": str(e)[:400]}
        _write_record(root, rec)
        return rec
    _CHILDREN[p.pid] = p
    # On Windows this must happen before the child can spawn anything: a
    # descendant created later inherits job membership (see `_job_wrap`).
    job = _job_wrap(p.pid, name)
    rec = {"name": name, "command": command_text, "pid": p.pid,
           "cwd": str(root), "log": str(log), "url": "", "job": job,
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
        text = _log_tail(rec)
        url = browser.find_server_url(text)
        if url:
            rec["url"] = url
            break
        time.sleep(0.1)
    _write_record(root, rec)
    return rec


def _log_tail(rec: dict) -> str:
    """The last few kilobytes of a record's log, without reading the whole file."""
    path = Path(str(rec.get("log") or ""))
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - _LOG_TAIL_BYTES))
            data = fh.read()
    except (OSError, ValueError):
        return ""
    return data.decode("utf-8", errors="replace")


def _refresh_url(rec: dict) -> bool:
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
    url = browser.find_server_url(_log_tail(rec))
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
        if _refresh_url(rec):
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
    try:
        text = Path(rec["log"]).read_text(encoding="utf-8", errors="replace")
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
            import ctypes
            from ctypes import wintypes

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


def _job_wrap(pid: int, name: str) -> str:
    """Put the just-spawned child in its own named job; '' when that failed.

    Every step is best effort: without a job, `stop` still gets `taskkill /T`,
    which reaps the tree whenever the leader is still alive.
    """
    k = _kernel32()
    if k is None or pid <= 0:
        return ""
    # Local\ rather than Global\: the global namespace needs a privilege we may
    # not hold, and a server started in one logon session is only ever stopped
    # from that same session.
    job = f"Local\\pedacito-{_safe_name(name)}-{pid}-{uuid.uuid4().hex[:8]}"
    hjob = hproc = None
    try:
        hjob = k.CreateJobObjectW(None, job)
        if not hjob:
            return ""
        hproc = k.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE,
                              False, pid)
        if not hproc:
            return ""                       # it already exited: nothing to reap
        if not k.AssignProcessToJobObject(hjob, hproc):
            return ""
        return job
    except Exception:                       # pragma: no cover - best effort
        return ""
    finally:
        # Dropping our own handles is safe: membership alone keeps the kernel
        # object (and its name) alive, and the leader is a member by now.
        _close_handle(k, hproc)
        _close_handle(k, hjob)


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


def _terminate_pid(pid: int, job: str = "") -> None:
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
    _terminate_pid(pid, str(rec.get("job") or ""))
    _forget_child(pid)
    rec["status"] = "stopped"
    rec["stopped"] = time.time()
    _write_record(root, rec)
    return f"stopped {rec.get('name')} (pid {pid})"


def render_list(root) -> str:
    """One line per registered process, or a hint when there are none."""
    recs = list_procs(root)
    if not recs:
        return ("No background processes. Start one with "
                "`pedacito serve PROJECT -- COMMAND`.")
    lines = []
    for rec in recs:
        extra = f"  {rec['url']}" if rec.get("url") else ""
        lines.append(f"{rec.get('name'):<20} [{rec.get('status', '?')}] "
                     f"pid {rec.get('pid')}{extra}  {rec.get('command', '')}")
    return "\n".join(lines)
