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
    stdout+stderr and capped at `max_output` characters.
    """
    timeout = float(timeout or 0)
    idle_timeout = float(idle_timeout or 0)
    cap = max(200, int(max_output or 8000))
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

    buf = bytearray()
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
                room = cap - len(buf)
                if room > 0:
                    buf.extend(chunk[:room])
                if len(chunk) > room:
                    state["overflow"] = True

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
    if state["overflow"] or len(text) > cap:
        text = text[:cap] + f"\n[output truncated to {cap} chars]"
    return rc, text, stopped, time.monotonic() - started
