"""RPC concurrency edge cases and process-tree cleanup.

Part 1 drives `rpc.RpcServer` over in-memory streams: permission requests
park a worker while the stdin thread stays free, steering lands mid-task,
cancel denies the parked request and keeps the warm session alive, and stdin
EOF still flushes the task's terminal `rpc_response`. Worker chatter on
`sys.stdout` must never leak onto the JSONL wire (§31).

Part 2 makes `proc.run` / `proc.spawn` kill a process tree: a Python child
starts a sleeping grandchild, and a short timeout must leave no survivor.
"""
from __future__ import annotations

import io
import json
import queue
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_sprint5_rpc import Harness  # noqa: E402

from pedacito import proc, security  # noqa: E402


@pytest.fixture(autouse=True)
def _no_lingering_permission_hook():
    yield
    security.install_permission_hook(None)


# ---------------------------------------------------------------------------
# RPC mid-task steering, permissions, cancel and EOF flush
# ---------------------------------------------------------------------------
def test_steer_and_permission_decisions_keep_stdout_pure_jsonl(
        monkeypatch):
    def run(h, task, apply, refine, plan, r):
        print("worker chatter that must never reach stdout")  # noqa: T201
        if task == "allow":
            ok, _msg = security.request_permission(
                "shell", "run ls", detail="ls -la", key="ls")
            # A step boundary after the parked permission drains steering,
            # which arrived while the worker was parked.
            try:
                steer = h.server.steer_queue().get(timeout=2)
            except queue.Empty:
                steer = ""
            r.reply = f"{ok}|{steer}"
        else:
            ok, msg = security.request_permission(
                "shell", "run rm", detail="rm x", key="rm")
            r.reply = f"{ok}|{'DENIED' in msg}"
        return r

    # The CLI rebinds sys.stdout to stderr; emulate the same discipline.
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    h = Harness(run=run)
    h.inp.feed({"id": 1, "method": "ask", "params": {"task": "allow"}})
    perm = h.wait(lambda f: f["kind"] == "permission_request",
                  "first permission_request")
    assert perm["category"] == "shell" and perm["key"] == "ls"

    h.inp.feed({"id": 2, "method": "steer", "params": {"text": "use tabs"}})
    assert h.respond(2)["ok"]
    h.inp.feed({"id": 3, "method": "permission_reply",
                "params": {"request_id": perm["request_id"],
                           "decision": "allow_once"}})
    assert h.respond(3)["result"]["decision"] == "allow_once"
    first = h.respond(1)
    assert first["ok"] and first["result"]["reply"] == "True|use tabs"

    # A denied permission is a real answer: the task continues ungated.
    h.inp.feed({"id": 4, "method": "ask", "params": {"task": "deny"}})
    perm2 = h.wait(lambda f: f["kind"] == "permission_request"
                   and f["request_id"] != perm["request_id"],
                   "second permission_request")
    h.inp.feed({"id": 5, "method": "permission_reply",
                "params": {"request_id": perm2["request_id"],
                           "decision": "deny"}})
    assert h.respond(5)["ok"]
    second = h.respond(4)
    assert second["ok"] and second["result"]["reply"] == "False|True"

    # Every byte on the protocol stream is one JSON object per line.
    for line in h.buf.getvalue().splitlines():
        parsed = json.loads(line)
        assert isinstance(parsed, dict)
    h.stop()


def test_cancel_while_parked_denies_and_keeps_the_server_warm():
    def run(h, task, apply, refine, plan, r):
        ok, msg = security.request_permission(
            "shell", "rm -rf build", detail="rm -rf build", key="rm")
        r.reply = f"{ok}|{'DENIED' in msg}"
        return r

    h = Harness(run=run)
    h.inp.feed({"id": 1, "method": "ask", "params": {"task": "danger"}})
    h.wait(lambda f: f["kind"] == "permission_request", "permission_request")

    h.inp.feed({"id": 2, "method": "cancel"})
    cancelled = h.respond(2)
    assert cancelled["ok"] and cancelled["result"]["active"] is True

    task = h.respond(1, timeout=10)
    assert not task["ok"] and task["error"] == "cancelled"
    assert task["result"]["reply"] == "False|True"   # parked call denied

    h.inp.feed({"id": 3, "method": "status"})
    status = h.respond(3)
    assert status["ok"] and status["result"]["busy"] is False
    h.stop()


def test_permission_timeout_denies_like_no_answer():
    def run(h, task, apply, refine, plan, r):
        ok, msg = security.request_permission(
            "web", "fetch docs", detail="https://example.test", key="docs")
        r.reply = f"{ok}|{'DENIED' in msg}"
        return r

    h = Harness(run=run, permission_timeout=1.0)
    h.inp.feed({"id": 1, "method": "ask", "params": {"task": "gate"}})
    h.wait(lambda f: f["kind"] == "permission_request", "permission_request")
    result = h.respond(1, timeout=10)
    assert result["ok"] and result["result"]["reply"] == "False|True"
    for line in h.buf.getvalue().splitlines():
        json.loads(line)
    h.stop()


def test_stdin_eof_mid_task_still_flushes_the_response_frame():
    def run(h, task, apply, refine, plan, r):
        ok, _msg = security.request_permission(
            "shell", "sleep", detail="sleep 1", key="sleep")
        r.reply = f"done:{ok}"
        return r

    h = Harness(run=run)
    h.inp.feed({"id": "eof-1", "method": "ask", "params": {"task": "work"}})
    h.wait(lambda f: f["kind"] == "permission_request", "permission_request")

    h.inp.close()                         # client hangs up while parked
    h.thread.join(timeout=5)
    assert not h.thread.is_alive(), "serve() deadlocked after stdin EOF"

    frames = h.frames()
    response = [f for f in frames
                if f["kind"] == "rpc_response" and f.get("id") == "eof-1"]
    assert response, frames
    assert response[-1]["ok"] and response[-1]["result"]["reply"] == "done:False"
    for line in h.buf.getvalue().splitlines():
        json.loads(line)
    h.stop()


# ---------------------------------------------------------------------------
# proc.run / proc.spawn: the whole process group dies on timeout
# ---------------------------------------------------------------------------
TREE_SCRIPT = r"""
import os
import subprocess
import sys
import time

grandchild = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(60)"])
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    fh.write(f"{os.getpid()} {grandchild.pid}\n")
time.sleep(60)
"""


def _read_pids(path: Path, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                parent, child = text.split()
                return int(parent), int(child)
        except (OSError, ValueError):
            pass
        time.sleep(0.02)
    raise AssertionError(f"the child never wrote its pid file: {path}")


def _process_state(pid: int) -> str | None:
    """Linux /proc state ('Z' = zombie, i.e. not running), None = gone."""
    try:
        return Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[2]
    except OSError:
        return None


def _dead(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = _process_state(pid)
        if state is None or state in ("Z", "X"):
            return True
        time.sleep(0.02)
    return False


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs /proc")
def test_proc_run_timeout_kills_child_and_grandchild(tmp_path):
    script = tmp_path / "tree.py"
    script.write_text(TREE_SCRIPT, encoding="utf-8")
    pidfile = tmp_path / "pids.txt"

    rc, _out, stopped, _secs = proc.run(
        [sys.executable, str(script), str(pidfile)],
        cwd=tmp_path, timeout=1.2)
    assert stopped == "timeout", (stopped, rc)
    parent, grandchild = _read_pids(pidfile)
    assert _dead(parent), f"child {parent} survived the timeout"
    assert _dead(grandchild), f"grandchild {grandchild} survived the timeout"


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs /proc")
def test_proc_spawn_terminate_kills_the_whole_tree(tmp_path):
    script = tmp_path / "tree.py"
    script.write_text(TREE_SCRIPT, encoding="utf-8")
    pidfile = tmp_path / "spawn_pids.txt"

    child = proc.spawn([sys.executable, str(script), str(pidfile)],
                       cwd=tmp_path)
    parent, grandchild = _read_pids(pidfile)
    assert parent == child.pid
    proc.terminate(child)
    child.wait(timeout=5)
    assert _dead(parent), f"child {parent} survived terminate()"
    assert _dead(grandchild), f"grandchild {grandchild} survived terminate()"


def test_proc_terminate_kills_group_when_getpgid_raises_lookup_error(monkeypatch):
    """If the group leader already exited (ProcessLookupError from getpgid),
    terminate() must still killpg(proc.pid, SIGKILL) so orphaned grandchildren
    in that process group die."""
    import os
    import signal

    if os.name == "nt":
        pytest.skip("POSIX process-group test")
    events: list[tuple] = []

    class FakeProc:
        pid = 98765

        def kill(self):
            events.append(("kill", self.pid))

    def fake_getpgid(pid):
        events.append(("getpgid", pid))
        raise ProcessLookupError

    def fake_killpg(pgid, sig):
        events.append(("killpg", pgid, sig))

    monkeypatch.setattr(proc.os, "getpgid", fake_getpgid)
    monkeypatch.setattr(proc.os, "killpg", fake_killpg)
    proc.terminate(FakeProc())
    assert events == [
        ("getpgid", 98765),
        ("killpg", 98765, signal.SIGKILL),
        ("kill", 98765),
    ]


def test_rpc_worker_finally_only_clears_matching_active_req_id():
    """A finishing worker must clear _active strictly in finally: and never
    clobber a newer task's _active req_id."""
    import threading
    from pedacito import rpc

    entered = threading.Event()
    release = threading.Event()

    def slow_run(task, apply=False, refine=False, plan=None):
        entered.set()
        release.wait(5)
        return type("R", (), {"reply": "ok", "applied": False, "results": [],
                              "ok": True, "stats": {}})()

    buf = io.StringIO()
    srv = rpc.RpcServer(
        run=slow_run, explain=lambda s: s, status=lambda: {},
        switch=lambda r, k: r, output=rpc.RpcOut(buf))
    srv._set_active("req-1")
    t = threading.Thread(
        target=srv._task_worker,
        args=("req-1", "ask", "work", False, False, None), daemon=True)
    t.start()
    assert entered.wait(5)
    # Simulate another request having become active before req-1 finishes.
    srv._set_active("req-2")
    release.set()
    t.join(timeout=5)
    assert not t.is_alive()
    assert srv._tag_for(None) == "req-2"
    assert srv._busy.is_set()
    # Matching req-2 clears cleanly.
    srv._set_active(None, expected="req-2")
    assert srv._tag_for(None) is None
    assert not srv._busy.is_set()


def test_rpc_serve_restores_previous_permission_hook():
    """serve() must restore whatever permission hook was installed before it
    started, without clobbering a different hook if someone else replaced it."""
    def prev_hook(info):
        return True, ""

    security.install_permission_hook(prev_hook)
    h = Harness()
    assert security.permission_handler() == h.server._permission_request
    h.stop()
    assert security.permission_handler() is prev_hook


def test_rpc_eof_join_grace_cancels_active_client(monkeypatch):
    """When stdin hits EOF and the worker outlives EOF_JOIN_GRACE, serve()
    must cancel self._client so a mid-stream LLM call unwinds promptly."""
    import threading
    from pedacito import rpc

    monkeypatch.setattr(rpc, "EOF_JOIN_GRACE", 0.15)
    monkeypatch.setattr(rpc, "EOF_CANCEL_GRACE", 1.0)
    cancelled = threading.Event()
    started = threading.Event()

    class CancelClient:
        def cancel(self):
            cancelled.set()

    def stuck_run(h, task, apply, refine, plan, r):
        started.set()
        cancelled.wait(5)
        r.reply = "unwound"
        return r

    h = Harness(run=stuck_run)
    h.server.set_client(CancelClient())
    h.inp.feed({"id": 1, "method": "ask", "params": {"task": "stream"}})
    assert started.wait(5)
    h.inp.close()
    h.thread.join(timeout=5)
    assert not h.thread.is_alive()
    assert cancelled.is_set()
