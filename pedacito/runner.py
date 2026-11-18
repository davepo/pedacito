"""Detect and run the project, and keep the last crash traceback.

The runner is deliberately small and shell-free:

* :func:`detect_runnable_targets` lists static HTML pages an iframe can show
  and the commands a project can actually run (Python entrypoints, pytest and
  a custom ``run_cmd`` in ``.pedacito/config.json``).
* :func:`run_project_command` executes one target with ``shell=False`` from
  the project root, captures both streams, enforces a timeout and extracts
  Python tracebacks so the GUI/TUI can offer a one-click repair task.

Nothing here uses a browser.  A static ``index.html`` is previewed by the GUI
through its own read-only iframe endpoint; when ``PEDACITO_NO_BROWSER=1`` (or
a test process is running) child processes also get headless display guards.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

MAX_OUTPUT = 200_000
MAX_TRACEBACK = 16_000
DEFAULT_TIMEOUT = 15

WEB_ENTRYPOINTS = ("index.html", "public/index.html", "dist/index.html")
PYTHON_ENTRYPOINTS = ("main.py", "app.py", "cli.py", "calculator.py", "game.py")
SHELL_ENTRYPOINTS = ("run.sh", "run.bat", "run.cmd")

_TRACEBACK_MARKERS = (
    "Traceback (most recent call last):",
    "Traceback (most recent call last):\r\n",
)


# --- detection ----------------------------------------------------------------

def _project_root(root) -> Path:
    return Path(str(root)).expanduser().resolve()


def _read_project_config(root: Path) -> dict:
    try:
        data = json.loads((root / ".pedacito" / "config.json")
                          .read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _python_bin() -> str:
    for name in ("python", "python3"):
        if shutil.which(name):
            return name
    return sys.executable


class Preview(dict):
    """One web preview target.

    It is a JSON-ready dict that also compares to its path and behaves like a
    path string in the common `p == "index.html"`, `"index.html" in web` and
    `Path(p)` checks, so simple callers do not need to learn the richer shape.
    """

    def __str__(self):
        return str(self.get("path") or "")

    def __fspath__(self):
        return str(self)

    def startswith(self, *args):
        return str(self).startswith(*args)

    def endswith(self, *args):
        return str(self).endswith(*args)

    def split(self, *args):
        return str(self).split(*args)

    def __eq__(self, other):
        if isinstance(other, str):
            return other in (self.get("path"), self.get("url"),
                             self.get("preview_url"))
        return dict.__eq__(self, other)

    __hash__ = None


class RunnableCommand(dict):
    """One runnable target; also compares to and prints as its command string."""

    def __str__(self):
        return str(self.get("command") or "")

    def __fspath__(self):
        return str(self)

    def startswith(self, *args):
        return str(self).startswith(*args)

    def endswith(self, *args):
        return str(self).endswith(*args)

    def split(self, *args):
        return str(self).split(*args)

    def __eq__(self, other):
        if isinstance(other, str):
            return other == self.get("command")
        return dict.__eq__(self, other)

    __hash__ = None


def _web_target(rel: str) -> Preview:
    url = "/api/runner/preview/" + quote(rel, safe="/")
    return Preview(path=rel, kind="html", url=url,
                   preview_url=url, live=True)


def _command_entry(command: str, *, kind: str, path: str = "",
                   argv=None, label: str = "", source: str = "detected",
                   cid: str = "") -> RunnableCommand:
    command = " ".join(str(command or "").split())
    if isinstance(argv, (list, tuple)):
        argv = [str(a) for a in argv]
    else:
        argv = []
    argv = argv or _split_command(command)
    return RunnableCommand(**{
        "id": cid or (kind + ("-" + path if path else "-" + command[:40])),
        "label": label or command,
        "command": command,
        "argv": argv,
        "kind": kind,
        "path": path,
        "source": source,
    })


def _python_tree(root: Path) -> tuple[bool, bool]:
    """(has Python sources, has test-looking sources or a tests/ dir)."""
    has_tests = (root / "tests").is_dir()
    has_sources = False
    skip = {".pedacito", ".git", ".hg", ".venv", "venv", "node_modules",
            "__pycache__", "dist", "build"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in skip]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            has_sources = True
            if has_tests or name.startswith("test_") or name.endswith("_test.py"):
                has_tests = True
    return has_sources, has_tests


def _custom_commands(root: Path) -> list[dict]:
    data = _read_project_config(root)
    raw = data.get("run_cmd") if "run_cmd" in data else data.get("run_cmds")
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    out: list[dict] = []
    for i, item in enumerate(items, 1):
        label = ""
        argv = None
        if isinstance(item, dict):
            label = str(item.get("label") or item.get("name") or "")
            argv = item.get("argv") or item.get("args")
            command = item.get("command") or item.get("cmd") \
                or item.get("run") or ""
            if not command and isinstance(argv, list) and argv:
                command = shlex.join([str(a) for a in argv])
        else:
            command = str(item or "")
        if not str(command).strip():
            continue
        out.append(_command_entry(
            str(command), kind="custom", path="", argv=argv,
            label=label or str(command), source="config",
            cid=f"custom-{i}"))
    return out


def detect_runnable_targets(root) -> dict:
    """Static preview pages and runnable commands for one workspace.

    The returned ``web`` list is ordered ``index.html``, ``public/index.html``,
    ``dist/index.html`` and each entry carries the iframe URL.  ``commands``
    contains Python entrypoints first, then pytest, a shell launcher and any
    ``run_cmd`` from ``.pedacito/config.json``.  Alias keys keep machine
    callers simple: ``web_previews``/``previews`` mirror ``web`` and
    ``runnable`` mirrors ``commands``.
    """
    root = _project_root(root)
    web: list[dict] = []
    if root.is_dir():
        for rel in WEB_ENTRYPOINTS:
            if (root / rel).is_file():
                web.append(_web_target(rel))

    commands: list[dict] = []
    python_paths: list[str] = []
    seen_cmds: set[str] = set()
    py = _python_bin()

    def add(entry: dict) -> None:
        key = entry["command"]
        if key and key not in seen_cmds:
            seen_cmds.add(key)
            commands.append(entry)

    if root.is_dir():
        for rel in PYTHON_ENTRYPOINTS:
            if (root / rel).is_file():
                python_paths.append(rel)
                add(_command_entry(
                    f"python {rel}", kind="python", path=rel,
                    argv=[py, rel], cid="python-" + rel.replace("/", "-")))

        # Any Python source gets the standard test command too: a project
        # without tests still answers honestly (pytest exits "no tests ran"),
        # and a generated test suite is one click away once it exists.
        if _python_tree(root)[0]:
            add(_command_entry(
                "pytest -q", kind="test", path="",
                argv=[py, "-m", "pytest", "-q"], cid="pytest-q"))

        for rel in SHELL_ENTRYPOINTS:
            if (root / rel).is_file():
                if rel.endswith(".sh"):
                    argv = ["bash", rel]
                else:
                    argv = [rel]
                add(_command_entry(rel, kind="shell", path=rel, argv=argv,
                                   cid="shell-" + rel.replace(".", "-")))

        for entry in _custom_commands(root):
            add(entry)

    custom = [c for c in commands if c["kind"] == "custom"]
    return {
        "root": str(root),
        "web": web,
        "web_previews": web,
        "previews": web,
        "html": [w["path"] for w in web],
        "commands": commands,
        "runnable": commands,
        "run_cmds": [c["command"] for c in commands],
        "python": python_paths,
        "python_entrypoints": python_paths,
        "custom": custom,
        "has_preview": bool(web),
    }


# --- execution ----------------------------------------------------------------

def _display_argv(argv: list[str]) -> str:
    try:
        if sys.platform == "win32":
            return subprocess.list2cmdline(argv)
        return shlex.join(argv)
    except (TypeError, ValueError):
        return " ".join(argv)


def _split_command(cmd) -> list[str]:
    if isinstance(cmd, (list, tuple)):
        return [str(a) for a in cmd]
    text = str(cmd or "").strip()
    if not text:
        return []
    try:
        if sys.platform != "win32":
            return shlex.split(text)
        # shlex keeps the quote characters in non-POSIX mode; strip them so
        # `python "my app.py"` becomes an argv the child can open.
        return [p.strip("\"'") for p in shlex.split(text, posix=False)]
    except ValueError:
        return text.split()


def _argv_for(cmd) -> tuple[list[str], str]:
    """(argv, display string) for a string, argv list or detected target."""
    if isinstance(cmd, dict):
        argv = cmd.get("argv")
        if isinstance(argv, (list, tuple)) and argv:
            argv = [str(a) for a in argv]
        else:
            argv = _split_command(cmd.get("command") or cmd.get("cmd"))
        display = str(cmd.get("command") or cmd.get("cmd")
                      or _display_argv(argv))
        return argv, display
    if isinstance(cmd, (list, tuple)):
        argv = [str(a) for a in cmd]
        return argv, _display_argv(argv)
    argv = _split_command(cmd)
    return argv, " ".join(str(cmd or "").split())


def _headless() -> bool:
    flag = str(os.environ.get("PEDACITO_NO_BROWSER") or "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return True
    return ("PYTEST_CURRENT_TEST" in os.environ
            or "pytest" in sys.modules)


def _child_env() -> dict:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if _headless():
        # Keep a GUI/Tk app from trying to open a window in a test or a
        # browser-less server; console apps ignore all of these.
        env.setdefault("PEDACITO_NO_BROWSER", "1")
        env.setdefault("BROWSER", "true")
        env.setdefault("MPLBACKEND", "Agg")
    return env


def _clip(text: str, limit: int = MAX_OUTPUT) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[... output truncated ...]"


def extract_traceback(*streams: str) -> str:
    """The first Python traceback in any stream (or "")."""
    for text in streams:
        text = str(text or "")
        if not text:
            continue
        starts = [text.find(marker) for marker in _TRACEBACK_MARKERS]
        starts = [i for i in starts if i >= 0]
        if not starts:
            continue
        block = text[min(starts):].strip()
        # A later log line ("[exit]") does not belong to the traceback; stop
        # at a process banner only when it clearly starts a new section.
        block = re.split(
            r"\n(?=(?:\[|>>>|Traceback \(most recent call last\):))",
            block, maxsplit=1)[0]
        return block[:MAX_TRACEBACK]
    return ""


def _failure_result(command: str, argv: list[str], *, exit_code: int,
                    stdout: str = "", stderr: str = "", detail: str = "",
                    duration_ms: int = 0, timed_out: bool = False) -> dict:
    tb = extract_traceback(stderr, stdout)
    if detail:
        stderr = (stderr + ("\n" if stderr else "") + detail).strip()
    return {
        "ok": exit_code == 0 and not tb and not timed_out,
        "exit_code": exit_code,
        "stdout": _clip(stdout),
        "stderr": _clip(stderr),
        "traceback_summary": tb,
        "command": command,
        "argv": argv,
        "duration_ms": int(duration_ms),
        "timed_out": bool(timed_out),
    }


def run_project_command(root, cmd=None, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Run one project command from `root` without a shell.

    Returns ``{ok, exit_code, stdout, stderr, traceback_summary, command,
    duration_ms, timed_out}``.  ``cmd`` may be a command string, an argv list,
    a detected target dict, or None (the first detected command is used).
    Timeouts and missing executables are reported like ordinary failures so
    the caller can always offer a repair task.
    """
    root = _project_root(root)
    if not root.is_dir():
        return _failure_result("", [], exit_code=127,
                               detail=f"workspace is not a directory: {root}")

    if cmd is None:
        targets = detect_runnable_targets(root)
        if not targets["commands"]:
            return _failure_result(
                "", [], exit_code=127,
                detail="No runnable command detected in this project.")
        cmd = targets["commands"][0]

    argv, display = _argv_for(cmd)
    if not argv:
        return _failure_result(display, [], exit_code=127,
                               detail="Empty command.")

    # `python`/`python3` means the same interpreter that is running Pedacito
    # when the PATH name does not exist (common in embedded venvs).
    if argv[0] in ("python", "python3") and not shutil.which(argv[0]):
        argv = [sys.executable] + argv[1:]
        if display.startswith(("python ", "python3 ")):
            display = "python " + " ".join(argv[1:])

    try:
        timeout_s = float(timeout)
    except (TypeError, ValueError):
        timeout_s = float(DEFAULT_TIMEOUT)
    timeout_s = max(0.1, timeout_s)
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv, shell=False, cwd=str(root), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout_s,
            env=_child_env())
    except subprocess.TimeoutExpired as exc:
        elapsed = int((time.monotonic() - started) * 1000)
        out = exc.stdout or ""
        err = exc.stderr or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", errors="replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", errors="replace")
        return _failure_result(
            display, argv, exit_code=124, stdout=str(out), stderr=str(err),
            detail=f"Command timed out after {timeout_s:g}s.",
            duration_ms=elapsed, timed_out=True)
    except OSError as exc:
        elapsed = int((time.monotonic() - started) * 1000)
        return _failure_result(
            display, argv, exit_code=127,
            detail=f"Could not run {argv[0]!r}: {exc}", duration_ms=elapsed)

    elapsed = int((time.monotonic() - started) * 1000)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    tb = extract_traceback(stderr, stdout)
    return {
        "ok": proc.returncode == 0 and not tb,
        "exit_code": int(proc.returncode),
        "stdout": _clip(stdout),
        "stderr": _clip(stderr),
        "traceback_summary": tb,
        "command": display,
        "argv": argv,
        "duration_ms": elapsed,
        "timed_out": False,
    }


def crash_prompt(result: dict, task: str = "") -> str:
    """A repair task carrying the exact command, exit code and traceback."""
    result = result or {}
    command = str(result.get("command") or "(unknown command)")
    code = result.get("exit_code", "?")
    tb = str(result.get("traceback_summary") or "").strip()
    stderr = str(result.get("stderr") or "").strip()
    stdout = str(result.get("stdout") or "").strip()
    lines = [
        "Fix this crash.",
        "",
        f"Command: {command}",
        f"Exit code: {code}",
    ]
    if result.get("timed_out"):
        lines.append("The command timed out.")
    if tb:
        lines += ["", "Traceback:", tb]
    elif stderr:
        lines += ["", "stderr:", stderr[-6000:]]
    elif stdout:
        lines += ["", "stdout:", stdout[-4000:]]
    if task:
        lines += ["", f"Context: {task}"]
    lines += [
        "",
        "Reproduce the failing command, find the cause, and fix it. Re-run "
        "the command afterwards and report the result.",
    ]
    return "\n".join(lines)
