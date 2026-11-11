"""Background process registry: keep a dev server alive across tasks.

    from . import processes

    rec = processes.start(root, "npm run dev", name="web")
    processes.list_procs(root)
    processes.logs(root, "web")
    processes.stop(root, "web")

The registry lives under `.pedacito/procs/`: one JSON record per process and a
`.log` file with its combined stdout/stderr. PIDs are process-group leaders, so
`stop` reaps descendants too. This is deliberately a small, local-first manager:
no daemon, no sockets -- just enough to start a server, find its printed URL,
read its log, and stop it later.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import time
from pathlib import Path

from . import browser, proc, safety
from .index import state_dir

PROC_DIR = "procs"
_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


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


def _alive(pid: int) -> bool:
    from .workspace import _pid_alive
    return _pid_alive(int(pid or 0))


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
               "cwd": str(root), "log": str(log), "url": "",
               "started": time.time(), "status": "error",
               "error": str(e)[:400]}
        _write_record(root, rec)
        return rec
    rec = {"name": name, "command": command_text, "pid": p.pid,
           "cwd": str(root), "log": str(log), "url": "",
           "started": time.time(), "status": "running"}
    # Give a dev server a moment to print its URL. Never block a run for long:
    # `wait` is a ceiling, and a command that exits is reported immediately.
    deadline = time.monotonic() + max(0.0, float(wait or 0))
    while time.monotonic() < deadline:
        if p.poll() is not None:
            rec["status"] = "exited"
            rec["returncode"] = p.returncode
            break
        try:
            text = log.read_text(encoding="utf-8", errors="replace")[-16000:]
        except OSError:
            text = ""
        url = browser.find_server_url(text)
        if url:
            rec["url"] = url
            break
        time.sleep(0.1)
    _write_record(root, rec)
    return rec


def list_procs(root) -> list[dict]:
    """Registry records, newest first, with dead pids marked `exited`."""
    root = Path(root).resolve()
    out: list[dict] = []
    for path in _dir(root).glob("*.json"):
        rec = _read_record(path)
        if not rec:
            continue
        if rec.get("status") == "running" and not _alive(int(rec.get("pid") or 0)):
            rec["status"] = "exited"
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


def _terminate_pid(pid: int) -> None:
    if pid <= 0:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           capture_output=True, timeout=10, check=False)
        except (OSError, subprocess.SubprocessError):
            pass
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if not _alive(pid):
            return
        time.sleep(0.05)
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
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
    _terminate_pid(pid)
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
