"""ACP bridge (``pedacito/acp.py``): JSON-RPC 2.0 over stdio on top of the
same ``rpc.RpcServer`` the JSONL transport uses.

The fake client writes requests into a queue-backed stdin and reads complete
JSON-RPC frames from a thread-safe writer, so the full bidirectional flow --
initialize, session/new, session/prompt with text/@file/image blocks, streamed
session/update notifications, a permission round-trip, cancel and EOF -- is
exercised without a model server or an editor.
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import __version__, acp, rpc, security  # noqa: E402

# A real 1x1 PNG (magic bytes included) so image validation is exercised.
PNG_1X1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAE"
    "hQGAhKmMIQAAAABJRU5ErkJggg=="
)


class LineIn:
    """Queue-backed stdin for ``AcpServer.serve()``."""

    def __init__(self):
        self.q: "queue.Queue[str]" = queue.Queue()

    def feed(self, obj) -> None:
        self.q.put(json.dumps(obj) + "\n")

    def feed_raw(self, line: str) -> None:
        self.q.put(line)

    def close(self) -> None:
        self.q.put("")

    def readline(self) -> str:
        return self.q.get()


class LineOut:
    """Thread-safe stdout that stores parsed JSON-RPC frames."""

    def __init__(self):
        self._frames: list[dict] = []
        self._lock = threading.Lock()
        self._closed = False

    def write(self, s: str) -> None:
        with self._lock:
            for line in s.splitlines():
                if line.strip():
                    self._frames.append(json.loads(line))

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self._closed = True

    def frames(self) -> list[dict]:
        with self._lock:
            return list(self._frames)


class FakeRes:
    reply = "all done"
    applied = False
    results: list = []
    steps = ["read a.py"]
    tests_ok = None
    ok = True
    stats = {"calls": 1, "prompt_tokens": 10, "completion_tokens": 5,
             "cost": 0.0, "seconds": 0.1, "session": "s-1"}


class Bridge:
    """One running ACP bridge wired to a fake engine callback."""

    def __init__(self, tmp_path: Path, run=None, *, permission_timeout=5.0):
        self.root = Path(tmp_path).resolve()
        self.inp = LineIn()
        self.out = LineOut()
        self.images: list = []
        self.profiles: list = []
        self.modes: list = []
        self.seen_tasks: list[str] = []
        self.release = threading.Event()
        self.started = threading.Event()

        def _run(task, apply=False, refine=False, plan=None):
            self.seen_tasks.append(task)
            if run is not None:
                return run(self, task, apply, refine, plan)
            return FakeRes()

        self.server = rpc.RpcServer(
            run=_run, explain=lambda q: f"about {q}",
            status=lambda: {"model": "fake"}, switch=lambda r, k: r,
            permission_timeout=permission_timeout)
        self.bridge = acp.AcpServer(
            self.server, project_root=self.root, input_stream=self.inp,
            output_stream=self.out,
            set_profile=lambda ref: self.profiles.append(ref),
            set_execution_mode=lambda mode: self.modes.append(mode),
            on_images=self.images.extend)
        self.thread = threading.Thread(target=self.bridge.serve, daemon=True)
        self.thread.start()
        # serve() installs the RPC permission hook before it reads; wait for
        # it so a prompt cannot race the hook installation.
        deadline = time.time() + 5
        while time.time() < deadline:
            if security.permission_handler() is not None:
                return
            time.sleep(0.01)
        raise AssertionError("RPC permission hook was never installed")

    # -- client helpers ------------------------------------------------- #
    def send(self, method, params=None, *, id=None):
        msg = {"jsonrpc": "2.0", "method": method}
        if id is not None:
            msg["id"] = id
        if params is not None:
            msg["params"] = params
        self.inp.feed(msg)
        return id

    def reply(self, req_id, result):
        self.inp.feed({"jsonrpc": "2.0", "id": req_id, "result": result})

    def wait(self, pred, what, timeout=6.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            got = [f for f in self.out.frames() if pred(f)]
            if got:
                return got[-1]
            time.sleep(0.02)
        raise AssertionError(f"no frame for {what}; saw {self.out.frames()}")

    def response(self, req_id, timeout=6.0):
        return self.wait(
            lambda f: f.get("id") == req_id
            and ("result" in f or "error" in f),
            f"response to {req_id}", timeout)

    def updates(self, session_id):
        return [f["params"]["update"] for f in self.out.frames()
                if f.get("method") == "session/update"
                and f.get("params", {}).get("sessionId") == session_id]

    def stop(self):
        self.inp.close()
        self.thread.join(timeout=5)
        assert not self.thread.is_alive(), "ACP serve() did not exit at EOF"


@pytest.fixture(autouse=True)
def _no_lingering_hook():
    yield
    security.install_permission_hook(None)


def _new_session(b, *, cwd=None, profile=None, mode=None):
    params = {"cwd": str(cwd or b.root)}
    if profile:
        params["profile"] = profile
    if mode:
        params["executionMode"] = mode
    b.send("session/new", params, id="new-1")
    res = b.response("new-1")
    assert "result" in res, res
    return res["result"]["sessionId"]


# ---------------------------------------------------------------------------
# initialize / session/new
# ---------------------------------------------------------------------------
def test_initialize_advertises_pedacito_and_its_capabilities(tmp_path):
    b = Bridge(tmp_path)
    b.send("initialize", {"protocolVersion": 1,
                          "clientCapabilities": {}}, id=1)
    res = b.response(1)
    result = res["result"]
    assert res["jsonrpc"] == "2.0"
    assert result["protocolVersion"] == acp.PROTOCOL_VERSION
    assert result["agentInfo"] == {"name": "pedacito", "version": __version__}
    caps = result["agentCapabilities"]
    assert caps["promptCapabilities"]["text"] is True
    assert caps["promptCapabilities"]["resources"] is True
    assert caps["promptCapabilities"]["image"] is True
    assert caps["streaming"] is True
    assert caps["permissionRequests"] is True
    b.stop()


def test_session_new_binds_cwd_profile_and_execution_mode(tmp_path):
    b = Bridge(tmp_path)
    sid = _new_session(b, profile="fast", mode="interleaved")
    assert sid
    assert b.profiles == ["fast"]
    assert b.modes == ["interleaved"]
    # the bridge is bound to one engine root: a foreign cwd is refused, not
    # silently read from
    other = tmp_path / "other"
    other.mkdir()
    b.send("session/new", {"cwd": str(other)}, id=2)
    err = b.response(2)
    assert err["error"]["code"] == -32002
    b.send("session/new", {"cwd": str(b.root), "executionMode": "bogus"},
           id=3)
    assert b.response(3)["error"]["code"] == -32602
    b.stop()


# ---------------------------------------------------------------------------
# session/prompt: blocks, notifications and the permission round-trip
# ---------------------------------------------------------------------------
def test_prompt_streams_updates_and_round_trips_permission(tmp_path):
    (tmp_path / "a.py").write_text("print('a')\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("resource body\n", encoding="utf-8")

    def run(b, task, apply, refine, plan):
        b.server.out.write(json.dumps(
            {"kind": "llm_stream", "delta_reasoning": "thinking",
             "phase": "answer"}) + "\n")
        b.server.out.write(json.dumps(
            {"kind": "llm_stream", "delta_content": "partial ",
             "phase": "answer"}) + "\n")
        b.server.out.write(json.dumps(
            {"kind": "step", "n": 1, "action": "read",
             "detail": "a.py", "obs_chars": 12}) + "\n")
        b.server.out.write(json.dumps(
            {"kind": "plan_updated",
             "path": ".pedacito/PLAN.md"}) + "\n")
        ok, _msg = security.request_permission(
            "shell", "run tests", detail="pytest -q", key="pytest")
        res = FakeRes()
        res.reply = f"done:{ok}"
        return res

    b = Bridge(tmp_path, run=run)
    sid = _new_session(b)
    b.send("session/prompt", {
        "sessionId": sid,
        "prompt": [
            {"type": "text", "text": "fix @a.py please"},
            {"type": "resource", "uri": (tmp_path / "b.txt").resolve().as_uri()},
            {"type": "image", "data": PNG_1X1, "mimeType": "image/png"},
        ],
    }, id=3)

    perm = b.wait(lambda f: f.get("method") == "session/request_permission",
                  "permission request")
    params = perm["params"]
    assert params["sessionId"] == sid
    assert [o["optionId"] for o in params["options"]] == [
        "allow_once", "allow_always", "reject_once"]
    assert params["_meta"]["requestId"]
    b.reply(perm["id"], {"outcome": {"outcome": "selected",
                                     "optionId": "allow_once"}})

    res = b.response(3)
    assert res["result"]["stopReason"] == "end_turn", res
    assert res["result"]["_meta"]["pedacito"]["session"] == "s-1"

    # Both prompt forms reached the engine: the @mention and the resource
    # block were resolved through resolve_safe_path and embedded verbatim.
    task = b.seen_tasks[-1]
    assert "fix @a.py please" in task
    assert "[file: a.py]" in task and "print('a')" in task
    assert "[file: b.txt]" in task and "resource body" in task
    # The image block became a normalized attachment for the engine.
    assert len(b.images) == 1
    assert b.images[0]["mime"] == "image/png"
    assert b.images[0]["data_b64"]

    updates = b.updates(sid)
    kinds = {u["sessionUpdate"] for u in updates}
    assert {"agent_thought_chunk", "agent_message_chunk", "tool_call",
            "tool_call_update", "plan"} <= kinds
    thoughts = [u for u in updates
                if u["sessionUpdate"] == "agent_thought_chunk"]
    assert thoughts[0]["content"]["text"] == "thinking"
    tool = next(u for u in updates if u["sessionUpdate"] == "tool_call")
    assert tool["toolCallId"] == "step-1" and tool["status"] == "in_progress"
    done = next(u for u in updates
                if u["sessionUpdate"] == "tool_call_update")
    assert done["status"] == "completed"
    # Content was streamed, so the final reply is not repeated as a chunk.
    assert all(u.get("content", {}).get("text") != "done:True"
               for u in updates)
    b.stop()


def test_prompt_without_streaming_emits_the_final_message_chunk(tmp_path):
    def run(b, task, apply, refine, plan):
        res = FakeRes()
        res.reply = "all good"
        return res

    b = Bridge(tmp_path, run=run)
    sid = _new_session(b)
    b.send("session/prompt", {
        "sessionId": sid,
        "prompt": [{"type": "text", "text": "silent run"}],
    }, id=4)
    res = b.response(4)
    assert res["result"]["stopReason"] == "end_turn"
    chunks = [u for u in b.updates(sid)
              if u["sessionUpdate"] == "agent_message_chunk"]
    assert chunks and chunks[-1]["content"]["text"] == "all good"
    b.stop()


# ---------------------------------------------------------------------------
# cancel / protocol errors / EOF
# ---------------------------------------------------------------------------
def test_cancel_maps_to_the_rpc_cancel_and_keeps_the_transport_alive(tmp_path):
    def run(b, task, apply, refine, plan):
        b.started.set()
        b.release.wait(5)
        return FakeRes()

    b = Bridge(tmp_path, run=run)
    sid = _new_session(b)
    b.send("session/prompt", {
        "sessionId": sid,
        "prompt": [{"type": "text", "text": "long"}],
    }, id=5)
    assert b.started.wait(5)
    b.send("session/cancel", {"sessionId": sid}, id=6)
    cancelled = b.response(6)
    assert cancelled["result"]["cancelled"] is True
    b.release.set()
    res = b.response(5, timeout=10)
    assert res["result"]["stopReason"] == "cancelled", res
    # The stdio process survives a cancel: another request still answers.
    b.send("initialize", {}, id=7)
    assert b.response(7)["result"]["protocolVersion"] == 1
    b.stop()


def test_unknown_method_and_unknown_session_are_jsonrpc_errors(tmp_path):
    b = Bridge(tmp_path)
    b.send("bogus/method", {}, id=1)
    err = b.response(1)
    assert err["error"]["code"] == -32601
    assert "unknown method" in err["error"]["message"]
    b.send("session/prompt", {"sessionId": "nope", "prompt": []}, id=2)
    assert b.response(2)["error"]["code"] == -32001
    b.stop()


def test_parse_error_and_invalid_request_keep_stdout_pure_json(tmp_path):
    b = Bridge(tmp_path)
    b.inp.feed_raw("this is not json\n")
    b.inp.feed_raw("[1, 2, 3]\n")
    b.send("initialize", {}, id=9)
    assert b.response(9)["result"]["protocolVersion"] == 1
    frames = b.out.frames()
    assert any(f.get("error", {}).get("code") == -32700 for f in frames)
    assert any(f.get("error", {}).get("code") == -32600 for f in frames)
    for frame in frames:
        assert frame["jsonrpc"] == "2.0"      # every stdout line is JSON-RPC
    b.stop()


def test_eof_exits_cleanly_without_waiting_on_a_pipe(tmp_path):
    b = Bridge(tmp_path)
    sid = _new_session(b)
    assert sid
    b.inp.close()
    b.thread.join(timeout=5)
    assert not b.thread.is_alive(), "EOF left the ACP loop waiting"
    for frame in b.out.frames():
        assert isinstance(frame, dict) and frame.get("jsonrpc") == "2.0"


def test_allow_always_is_cached_by_the_permission_broker(tmp_path):
    def run(b, task, apply, refine, plan):
        first, _ = security.request_permission(
            "shell", "run tests", detail="pytest -q", key="pytest")
        second, _ = security.request_permission(
            "shell", "run tests again", detail="pytest -q", key="pytest")
        res = FakeRes()
        res.reply = f"{first},{second}"
        return res

    b = Bridge(tmp_path, run=run)
    sid = _new_session(b)
    b.send("session/prompt", {
        "sessionId": sid,
        "prompt": [{"type": "text", "text": "test twice"}],
    }, id=8)
    perm = b.wait(lambda f: f.get("method") == "session/request_permission",
                  "permission request")
    b.reply(perm["id"], {"outcome": {"outcome": "selected",
                                     "optionId": "allow_always"}})
    res = b.response(8)
    assert res["result"]["stopReason"] == "end_turn"
    assert b.server.was_allowed("shell", "pytest")
    perms = [f for f in b.out.frames()
             if f.get("method") == "session/request_permission"]
    assert len(perms) == 1               # the second call came from the cache
    b.stop()


def test_reject_once_is_a_real_denial_the_task_continues(tmp_path):
    def run(b, task, apply, refine, plan):
        ok, msg = security.request_permission(
            "shell", "rm -rf build", detail="rm -rf build", key="rm")
        res = FakeRes()
        res.reply = f"{ok}|{'DENIED' in msg}"
        return res

    b = Bridge(tmp_path, run=run)
    sid = _new_session(b)
    b.send("session/prompt", {
        "sessionId": sid,
        "prompt": [{"type": "text", "text": "danger"}],
    }, id=9)
    perm = b.wait(lambda f: f.get("method") == "session/request_permission",
                  "permission request")
    b.reply(perm["id"], {"outcome": {"outcome": "selected",
                                     "optionId": "reject_once"}})
    res = b.response(9)
    assert res["result"]["stopReason"] == "end_turn"
    chunks = [u for u in b.updates(sid)
              if u["sessionUpdate"] == "agent_message_chunk"]
    assert chunks and chunks[-1]["content"]["text"] == "False|True"
    assert b.seen_tasks == ["danger"]
    b.stop()


def test_second_prompt_while_busy_is_an_error_the_first_still_finishes(tmp_path):
    def run(b, task, apply, refine, plan):
        b.started.set()
        b.release.wait(5)
        return FakeRes()

    b = Bridge(tmp_path, run=run)
    sid = _new_session(b)
    b.send("session/prompt", {
        "sessionId": sid,
        "prompt": [{"type": "text", "text": "first"}],
    }, id=20)
    assert b.started.wait(5)
    b.send("session/prompt", {
        "sessionId": sid,
        "prompt": [{"type": "text", "text": "second"}],
    }, id=21)
    busy = b.response(21)
    assert busy["result"]["stopReason"] == "error"
    assert "busy" in busy["result"]["_meta"]["error"]
    b.release.set()
    first = b.response(20, timeout=10)
    assert first["result"]["stopReason"] == "end_turn"
    b.stop()


def test_eof_while_a_prompt_runs_still_answers_before_exit(tmp_path):
    def run(b, task, apply, refine, plan):
        time.sleep(0.3)
        res = FakeRes()
        res.reply = "late but complete"
        return res

    b = Bridge(tmp_path, run=run)
    sid = _new_session(b)
    b.send("session/prompt", {
        "sessionId": sid,
        "prompt": [{"type": "text", "text": "slow"}],
    }, id=12)
    time.sleep(0.05)
    b.inp.close()
    b.thread.join(timeout=10)
    assert not b.thread.is_alive(), "EOF while a prompt ran hung the process"
    res = next((f for f in b.out.frames() if f.get("id") == 12), None)
    assert res is not None and res["result"]["stopReason"] == "end_turn"


def test_resource_traversal_is_refused(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("SECRET", encoding="utf-8")
    (tmp_path / "ok.txt").write_text("fine", encoding="utf-8")
    b = Bridge(tmp_path)
    sid = _new_session(b)
    b.send("session/prompt", {
        "sessionId": sid,
        "prompt": [{"type": "resource", "path": "../outside.txt"}],
    }, id=10)
    err = b.response(10)
    assert err["error"]["code"] == -32602
    # an @mention that escapes is simply not hydrated; the turn still runs
    b.send("session/prompt", {
        "sessionId": sid,
        "prompt": [{"type": "text", "text": "look at @../outside.txt"}],
    }, id=11)
    res = b.response(11)
    assert res["result"]["stopReason"] == "end_turn"
    assert "SECRET" not in b.seen_tasks[-1]
    b.stop()
