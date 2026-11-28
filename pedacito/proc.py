"""Run a child process with bounded observation and process-group cleanup.

Usage
-----
    from . import proc

    rc, out, stopped, secs = proc.run(
        ["python", "main.py"], cwd=root, timeout=120, idle_timeout=10)
    # stopped is "" (normal exit), "idle", "timeout", or "error".

`subprocess.run` kills only the direct child on timeout and waits forever on a
GUI app that never exits. This module stops a process that goes silent for
`idle_timeout` seconds and kills the whole process group (process tree on
Windows). The idle rule is based on observed output, not command names: a
game prints a banner and then nothing, while a test runner keeps printing.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# Win32 CREATE_SUSPENDED. subprocess does not export it (it only names the
# console/process-group flags), so spell the flag out: the child is created
# frozen and the caller resumes it after wrapping it in a job object.
_CREATE_SUSPENDED = 0x00000004


def venv_bin_dirs(root) -> list[str]:
    """The project's virtualenv bin directories that exist on disk, plus the
    running interpreter's own directory (Hunt #12 §5). `bin_dir` is `Scripts`
    on Windows and `bin` on POSIX. Order: .venv, venv, then the interpreter."""
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    out: list[str] = []
    try:
        base = Path(root) if root is not None else None
    except (TypeError, ValueError):
        base = None
    if base is not None:
        for name in (".venv", "venv"):
            d = base / name / bin_dir
            if d.is_dir():
                out.append(str(d))
    exe_dir = str(Path(sys.executable).parent)
    if exe_dir and exe_dir not in out:
        out.append(exe_dir)
    return out


def with_venv_path(env: dict | None, root, *,
                   bin_dirs: tuple | list | None = None) -> dict:
    """The child environment with the project's venv bin dirs prepended to
    PATH, so a bare `pytest -q` / `ruff check .` resolves when the tool lives
    in `.venv/bin` (or Pedacito's own venv) rather than the global PATH.
    Everything else passes through untouched; an explicit env stays explicit
    (a copy, never the caller's dict).

    `bin_dirs` (from `env_resolve.ResolvedEnvironment.bin_dirs`) replaces the
    legacy `.venv`/`venv`/interpreter probe when a caller already resolved the
    project environment. Passing an empty tuple is meaningful: use exactly no
    extra directories, never silently fall back to the legacy probe."""
    base = dict(os.environ if env is None else env)
    dirs = ([str(d) for d in bin_dirs] if bin_dirs is not None
            else venv_bin_dirs(root))
    if dirs:
        path_key = "PATH" if "PATH" in base else next(
            (k for k in base if k.upper() == "PATH"), "PATH")
        path = base.get(path_key) or os.environ.get(path_key, "")
        parts = path.split(os.pathsep) if path else []
        prepend = [d for d in dirs if d not in parts]
        if prepend:
            base[path_key] = os.pathsep.join(prepend + parts)
    return base


def spawn(command, *, cwd, env: dict | None = None,
          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
          suspended: bool = False, bin_dirs: tuple | list | None = None
          ) -> subprocess.Popen:
    """Start a child in its own process group without waiting for it.

    Used by the screenshot and browser actions, which later `terminate()` the
    whole tree, and by the background-server registry. Output is discarded
    unless the caller passes PIPE. `suspended` (Windows only) starts the
    child frozen so the caller can put it in a job object before it runs a
    single instruction -- job membership is inherited at creation, so a
    child that runs first can spawn descendants that escape the job. The
    caller MUST resume it (see `processes._resume_child`)."""
    kwargs = {}
    if os.name != "nt":
        kwargs["start_new_session"] = True
    elif suspended:
        kwargs["creationflags"] = _CREATE_SUSPENDED
    return subprocess.Popen(
        command, cwd=str(cwd), stdin=subprocess.DEVNULL,
        stdout=stdout, stderr=stderr,
        env=with_venv_path(env, cwd, bin_dirs=bin_dirs), **kwargs)


def terminate(proc: subprocess.Popen) -> None:
    """Kill a process started by spawn() or run(), including its descendants."""
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
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        # Leader may already be gone while orphaned grandchildren remain in its
        # process group (spawn()/run() make proc.pid the session/group leader).
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.kill()
        except OSError:
            pass
        return
    except (PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def kill_tree(proc: subprocess.Popen) -> None:
    """Alias for terminate(): kill a spawned process and its whole tree."""
    terminate(proc)


def _register_subprocess(p: subprocess.Popen, command) -> str | None:
    """Register the live Popen as a session `subprocess` resource (no-op when
    no session is bound or the registry is absent). Returns the resource id."""
    try:
        from . import session_ctx
        sess = session_ctx.current_session()
    except Exception:
        return None
    if sess is None or sess.resource_registry is None:
        return None
    label = " ".join(str(x) for x in command)[:120]
    try:
        return sess.resource_registry.register(
            "subprocess", label, lambda: terminate(p))
    except Exception:
        return None


def _unregister_subprocess(resource_id: str | None) -> None:
    if not resource_id:
        return
    try:
        from . import session_ctx
        sess = session_ctx.current_session()
        if sess is not None and sess.resource_registry is not None:
            sess.resource_registry.unregister(resource_id)
    except Exception:
        pass


def _cancel_requested_proc(cancel_event: threading.Event | None) -> bool:
    """True when the explicit cancel_event or the bound session's is set."""
    if cancel_event is not None and cancel_event.is_set():
        return True
    try:
        from . import session_ctx
        sess = session_ctx.current_session()
        if sess is not None and sess.cancel_event.is_set():
            return True
    except Exception:
        pass
    return False


def _run_windows_sandboxed(command, *, cwd, timeout, idle_timeout, max_output,
                           env, stdin_text, cancel_event, writable_dirs=(),
                           bin_dirs=None):
    """Run `command` through the Windows Job Object + restricted-token sandbox
    and map the result onto `run()`'s (rc, output, stopped, seconds) shape."""
    from . import sandbox_win
    started = time.monotonic()
    full_env = with_venv_path(env, cwd, bin_dirs=bin_dirs)
    try:
        rc, out, _err, _elapsed, stopped = sandbox_win._run_nt(
            list(command), cwd=Path(cwd), env=full_env,
            timeout=timeout or None, idle_timeout=idle_timeout,
            max_output=max_output, stdin_text=stdin_text,
            cancel_event=cancel_event,
            writable_dirs=tuple(Path(p) for p in (writable_dirs or ())))
    except Exception as exc:            # sandbox unavailable at dispatch time
        return None, str(exc), "error", time.monotonic() - started
    return rc, out, stopped, time.monotonic() - started


def run(command, *, cwd, timeout: float | None = None,
        idle_timeout: float = 0, shell: bool = False,
        max_output: int = 8000, env: dict | None = None,
        stdin_text: str | None = None,
        cancel_event: threading.Event | None = None,
        windows_sandbox: bool = False,
        writable_dirs: tuple = (),
        bin_dirs: tuple | list | None = None,
        ) -> tuple[int | None, str, str, float]:
    """Run `command` and return (returncode, output, stopped, seconds).

    `timeout` is the hard wall-clock limit; `idle_timeout` stops the process
    after that many seconds with no output (0 disables it). `stopped` is one
    of "", "idle", "timeout", "cancelled", or "error" (could not start).
    Output is combined stdout+stderr, capped at about `max_output` characters
    by keeping the head and the tail (test summaries and tracebacks end the
    output).

    `cancel_event` is an optional ``threading.Event``; when it (or the bound
    session's ``cancel_event``) is set the child is terminated and ``stopped``
    reports "cancelled".

    `windows_sandbox` (Windows only) runs the child under the built-in Job
    Object + restricted-token sandbox instead of a plain `Popen`.
    `writable_dirs` (Windows sandbox only) lists extra directories the child
    may write; they come from `cfg.sandbox_writable`.
    `bin_dirs` overrides the legacy venv PATH probe with a resolved
    environment's bin directories (see `env_resolve.command_env`).
    """
    if windows_sandbox and os.name == "nt":
        return _run_windows_sandboxed(
            command, cwd=cwd, timeout=timeout, idle_timeout=idle_timeout,
            max_output=max_output, env=env, stdin_text=stdin_text,
            cancel_event=cancel_event, writable_dirs=writable_dirs,
            bin_dirs=bin_dirs)
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
            command, cwd=str(cwd),
            stdin=subprocess.DEVNULL if stdin_text is None else subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            shell=shell, env=with_venv_path(env, cwd, bin_dirs=bin_dirs),
            **kwargs)
    except (OSError, ValueError) as e:
        return None, str(e), "error", time.monotonic() - started
    resource_id = _register_subprocess(p, command)
    if stdin_text is not None and p.stdin is not None:
        def feed() -> None:
            try:
                p.stdin.write(stdin_text.encode("utf-8", "replace"))
            except (OSError, ValueError):
                pass
            finally:
                try:
                    p.stdin.close()
                except (OSError, ValueError):
                    pass
        threading.Thread(target=feed, daemon=True).start()

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
    try:
        while True:
            # Cancellation wins over an already-dead child: SessionContext
            # .cancel() sets the event AND terminates the subprocess, so the
            # process can be reaped in the microseconds before this poll.
            if _cancel_requested_proc(cancel_event):
                stopped = "cancelled"
                terminate(p)
                break
            rc = p.poll()
            if rc is not None:
                break
            now = time.monotonic()
            if timeout and now - started >= timeout:
                stopped = "timeout"
                terminate(p)
                break
            with lock:
                last_out = state["last"]
            if idle_timeout and now - last_out >= idle_timeout:
                stopped = "idle"
                terminate(p)
                break
            time.sleep(0.05)

        if not stopped and _cancel_requested_proc(cancel_event):
            stopped = "cancelled"

        try:
            rc = p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            terminate(p)
            try:
                rc = p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                rc = None
        reader.join(timeout=3)
    finally:
        _unregister_subprocess(resource_id)
        if p.stdout is not None:
            try:
                p.stdout.close()
            except OSError:
                pass
        if reader.is_alive():
            reader.join(timeout=1)

    with lock:
        raw_buf = bytes(buf)
        overflow = state["overflow"]
        raw_tail = bytes(tail) if overflow else b""
    text = raw_buf.decode("utf-8", "replace")
    if overflow:
        tail_text = raw_tail.decode("utf-8", "replace")
        text = (text + "\n[... output truncated: middle omitted; kept the "
                f"first {head_cap} and last {tail_cap} chars ...]\n"
                + tail_text)
    return rc, text, stopped, time.monotonic() - started
