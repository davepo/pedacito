"""Run a child process with bounded observation and process-group cleanup.

Usage
-----
    from . import proc

    rc, out, stopped, secs = proc.run(
        ["python", "main.py"], cwd=root, timeout=120, idle_timeout=10)
    # stopped is "" (normal exit), "idle", "timeout", or "error".

Why this exists: `subprocess.run` waits for the process to exit and kills only
the direct child on timeout. A game or GUI tool opens a window and never
exits, so pedacito sits at a `run` step until a person closes it -- and a
launcher can leave the visible window alive even after its parent is killed.
This module watches output, stops a process that goes silent for
`idle_timeout` seconds, and kills the whole process group (process tree on
Windows) rather than just the parent.

The idle rule is the key for GUI apps: a game usually prints a startup banner
and then nothing, while a test runner keeps printing. It is deliberately
based on observed output rather than command names, which are too varied to
classify reliably.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time


def spawn(command, *, cwd, shell: bool = False, env: dict | None = None,
          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
          ) -> subprocess.Popen:
    """Start a child in its own process group without waiting for it.

    Used by the screenshot and browser actions: launch the GUI/server,
    capture after a delay, then `terminate()` the whole tree. Output is
    discarded by default (the browser launcher passes PIPE to find the URL)."""
    kwargs = {}
    if os.name != "nt":
        kwargs["start_new_session"] = True
    return subprocess.Popen(
        command, cwd=str(cwd), stdin=subprocess.DEVNULL,
        stdout=stdout, stderr=stderr, shell=shell, env=env, **kwargs)


def terminate(proc: subprocess.Popen) -> None:
    """Kill a process started by spawn(), including its descendants."""
    _terminate(proc)


def _terminate(proc: subprocess.Popen) -> None:
    """Kill the process and its descendants, not just the direct child."""
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=10)
        except Exception:
            pass
        try:
            proc.kill()
        except OSError:
            pass
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def run(command, *, cwd, timeout: float | None = None,
        idle_timeout: float = 0, shell: bool = False,
        max_output: int = 8000, env: dict | None = None
        ) -> tuple[int | None, str, str, float]:
    """Run `command` and return (returncode, output, stopped, seconds).

    `timeout` is the hard wall-clock limit; `idle_timeout` stops the process
    after that many seconds with no output (0 disables it). `stopped` is one
    of "", "idle", "timeout", or "error" (could not start). Output is combined
    stdout+stderr, capped at about `max_output` characters by keeping the
    HEAD and the TAIL and dropping the middle: a test runner's summary and a
    traceback's final `Error:` line live at the end, which head-only
    truncation threw away exactly when the output was long enough to matter.
    """
    timeout = float(timeout or 0)
    idle_timeout = float(idle_timeout or 0)
    cap = max(200, int(max_output or 8000))
    head_cap = max(120, int(cap * 0.6))
    tail_cap = max(80, cap - head_cap)
    started = time.monotonic()
    kwargs = {}
    if os.name != "nt":
        # Its own session lets us signal the whole group on timeout.
        kwargs["start_new_session"] = True
    try:
        p = subprocess.Popen(
            command, cwd=str(cwd), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            shell=shell, env=env, **kwargs)
    except (OSError, ValueError) as e:
        return None, str(e), "error", time.monotonic() - started

    buf = bytearray()      # the first head_cap bytes
    tail = bytearray()     # rolling window: the last tail_cap bytes
    state = {"last": started, "overflow": False}
    lock = threading.Lock()

    def pump() -> None:
        pipe = p.stdout
        if pipe is None:
            return
        while True:
            try:
                chunk = pipe.read1(65536)
            except (OSError, ValueError):
                break
            if not chunk:
                break
            with lock:
                state["last"] = time.monotonic()
                if not state["overflow"]:
                    room = head_cap - len(buf)
                    if len(chunk) <= room:
                        buf.extend(chunk)
                        continue
                    buf.extend(chunk[:room])
                    state["overflow"] = True
                    chunk = chunk[room:]
                tail.extend(chunk)
                if len(tail) > tail_cap:
                    del tail[:len(tail) - tail_cap]

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()

    stopped = ""
    while True:
        rc = p.poll()
        if rc is not None:
            break
        now = time.monotonic()
        if timeout and now - started >= timeout:
            stopped = "timeout"
            _terminate(p)
            break
        if idle_timeout and now - state["last"] >= idle_timeout:
            stopped = "idle"
            _terminate(p)
            break
        time.sleep(0.05)

    try:
        rc = p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _terminate(p)
        try:
            rc = p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            rc = None
    reader.join(timeout=3)

    text = bytes(buf).decode("utf-8", "replace")
    if state["overflow"]:
        tail_text = bytes(tail).decode("utf-8", "replace")
        text = (text + "\n[... output truncated: middle omitted; kept the "
                f"first {head_cap} and last {tail_cap} chars ...]\n"
                + tail_text)
    return rc, text, stopped, time.monotonic() - started
