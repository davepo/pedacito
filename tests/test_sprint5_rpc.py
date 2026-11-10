"""Sprint 5: JSONL-RPC over stdio -- dispatch, event tagging, the steer
queue, the permission broker, cancel, and stdout purity.

The transport is the product under test: every call reaches the RpcServer
through injected fakes (a queue-backed stdin, a StringIO stdout, scripted
run/explain/status/switch callbacks). The real agent pipeline is exercised
elsewhere; here we only check the wire.
"""
from __future__ import annotations

import io
import json
import queue
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import llm, rpc, security  # noqa: E402


class LineIn:
    """stdin stand-in: readline() blocks until a line is fed (or '' = EOF)."""

    def __init__(self):
        self.q: "queue.Queue[str]" = queue.Queue()

    def feed(self, obj):
        self.q.put(json.dumps(obj) + "\n")

    def feed_raw(self, line: str):
        self.q.put(line)

    def close(self):
        self.q.put("")

    def readline(self):
        return self.q.get()


class FakeRes:
    reply = "all done"
    applied = False
    results: list = []
    steps = ["read a.py"]
    tests_ok = None
    ok = True

    @staticmethod
    def stats():
        return {}


class Harness:
    def __init__(self, run=None, **kw):
        self.inp = LineIn()
        self.buf = io.StringIO()
        box = {}
        self.out = rpc.RpcOut(self.buf,
                              lambda: box["s"]._tag_for(None)
                              if "s" in box else None)
        self.saved = []

        def _run(task, apply=False, refine=False, plan=None):
            r = FakeRes()
            r.stats = {"calls": 2, "prompt_tokens": 100, "cost": 0.001,
                       "seconds": 1.0, "session": "s-1"}
            r.applied = bool(apply)
            if apply:
                class Res:
                    path = "a.py"
                    applied = True
                    ok = True
                r.results = [Res()]
            if run is not None:
                return run(self, task, apply, refine, plan, r)
            return r

        self.checkpointed = []
        self.locked = []
        kw.setdefault("switch", lambda ref, kind: f"{kind}:{ref}")
        kw.setdefault("status", lambda: {"model": "fake-model"})
        kw.setdefault("explain", lambda q: f"about {q}")
        self.server = rpc.RpcServer(
            run=_run, ensure_lock=lambda: self.locked.append(1),
            save_checkpoint=lambda t: self.checkpointed.append(t),
            input_stream=self.inp, output=self.out,
            permission_timeout=float(kw.pop("permission_timeout", 5.0)), **kw)
        box["s"] = self.server
        self.server.set_client(type("C", (), {"cancel": lambda s: None})())
        self.thread = threading.Thread(target=self.server.serve, daemon=True)
        self.thread.start()

    # -- observation helpers ------------------------------------------------ #
    def frames(self):
        return [json.loads(l) for l in self.buf.getvalue().splitlines()
                if l.strip()]

    def wait(self, pred, what, timeout=6.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            got = [f for f in self.frames() if pred(f)]
            if got:
                return got[-1]
            time.sleep(0.02)
        raise AssertionError(f"no frame for {what}; saw {self.frames()}")

    def respond(self, req_id, timeout=6.0):
        return self.wait(
            lambda f: f["kind"] == "rpc_response" and f.get("id") == req_id,
            f"response to {req_id}", timeout)

    def stop(self):
        self.inp.close()
        self.thread.join(timeout=5)
        assert not self.thread.is_alive(), "serve() did not exit at EOF"


@pytest.fixture(autouse=True)
def _no_lingering_hook():
    yield
    security.install_permission_hook(None)


def test_pure_jsonl_and_status_unknown_errors():
    """Invalid requests still answer with parseable JSONL, never prose."""
    h = Harness()
    h.inp.feed({"id": 1, "method": "status"})
    h.inp.feed({"id": 2, "method": "bogus"})
    h.inp.feed_raw("not json at all\n")
    h.inp.feed_raw('["a list is not a request"]\n')
    f1 = h.respond(1)
    assert f1["ok"] and f1["result"]["model"] == "fake-model"
    assert f1["result"]["busy"] is False
    f2 = h.respond(2)
    assert not f2["ok"] and "unknown method" in f2["error"]
    errs = [f for f in h.frames() if f["kind"] == "rpc_response"
            and not f["ok"] and f.get("id") is None]
    assert any("invalid JSON" in f["error"] for f in errs)
    assert any("JSON object" in f["error"] for f in errs)
    for line in h.buf.getvalue().splitlines():
        json.loads(line)              # the promise: stdout is 100% JSONL
    h.stop()


def test_ask_streams_tagged_events_and_terminates():
    def run(h, task, apply, refine, plan, r):
        h.out.write(json.dumps({"t": 0.1, "kind": "step",
                                "action": "read"}) + "\n")
        return r

    h = Harness(run=run)
    h.inp.feed({"id": "r5", "method": "ask", "params": {"task": "fix it"}})
    res = h.respond("r5")
    assert res["ok"] and res["method"] == "ask"
    result = h.wait(lambda f: f["kind"] == "result", "result frame")
    assert result["req_id"] == "r5" and result["reply"] == "all done"
    assert result["cost"]["calls"] == 2 and result["session"] == "s-1"
    assert not result["applied"] and result["changed"] == []
    step = h.wait(lambda f: f["kind"] == "step", "streamed step")
    assert step["req_id"] == "r5"
    # terminal frames carry the id twice (rpc clients key on `id`, event
    # streamers on `req_id`)
    assert res["result"]["reply"] == "all done"
    kinds = [f["kind"] for f in h.frames()]
    assert kinds.index("step") < kinds.index("result")  # order preserved
    h.stop()


def test_ask_apply_locks_checkpoints_and_reports_changed():
    h = Harness()
    h.inp.feed({"id": 1, "method": "ask",
                "params": {"task": "fix it", "apply": True}})
    res = h.respond(1)
    assert res["ok"] and res["result"]["applied"]
    assert res["result"]["changed"] == ["a.py"]
    assert h.locked == [1]                       # workspace lock taken once
    assert h.checkpointed == ["fix it"]          # checkpoint per applied task
    h.stop()


def test_ask_missing_task_refused_without_killing_session():
    h = Harness()
    h.inp.feed({"id": 1, "method": "ask", "params": {}})
    f = h.respond(1)
    assert not f["ok"] and "missing params.task" in f["error"]
    h.inp.feed({"id": 2, "method": "status"})
    assert h.respond(2)["ok"]
    h.stop()


def test_busy_guard_and_warmth_after_completion():
    release = threading.Event()

    def run(h, task, apply, refine, plan, r):
        release.wait(5)
        return r

    h = Harness(run=run)
    h.inp.feed({"id": 1, "method": "ask", "params": {"task": "slow"}})
    time.sleep(0.1)                              # worker is inside run()
    h.inp.feed({"id": 2, "method": "ask", "params": {"task": "second"}})
    f2 = h.respond(2)
    assert not f2["ok"] and f2["error"].startswith("busy")
    release.set()
    assert h.respond(1)["ok"]
    h.inp.feed({"id": 3, "method": "ask", "params": {"task": "after"}})
    assert h.respond(3)["ok"]                    # session stayed warm
    h.stop()


def test_steer_lands_in_the_active_tasks_queue():
    seen: "queue.Queue[str]" = queue.Queue()

    def run(h, task, apply, refine, plan, r):
        q = h.server.steer_queue()
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                seen.put(q.get_nowait())
                break
            except queue.Empty:
                time.sleep(0.02)
        return r

    h = Harness(run=run)
    h.inp.feed({"id": 1, "method": "steer", "params": {"text": "no task"}})
    assert not h.respond(1)["ok"]                # nothing to steer yet
    h.inp.feed({"id": 2, "method": "ask", "params": {"task": "work"}})
    time.sleep(0.1)
    h.inp.feed({"id": 3, "method": "steer", "params": {"text": "use tabs"}})
    assert h.respond(3)["ok"]
    assert seen.get(timeout=5) == "use tabs"     # drained at a boundary
    assert h.respond(2)["ok"]
    h.stop()


def test_permission_roundtrip_unblocks_and_caches():
    def run(h, task, apply, refine, plan, r):
        ok1, _ = security.request_permission("shell", "run ls",
                                             detail="ls -la", key="ls")
        ok2, _ = security.request_permission("shell", "run ls again",
                                             detail="ls -la", key="ls")
        r.reply = f"{ok1},{ok2}"
        return r

    h = Harness(run=run)
    h.inp.feed({"id": 7, "method": "ask", "params": {"task": "gate"}})
    ev = h.wait(lambda f: f["kind"] == "permission_request",
                "permission_request")
    assert ev["req_id"] == "7" and ev["turn_id"] == 7
    assert ev["category"] == "shell" and ev["key"] == "ls"
    assert ev["request_id"] == "perm-1"
    assert ev["timeout_seconds"] == 5.0
    h.inp.feed({"id": 8, "method": "permission_reply",
                "params": {"request_id": ev["request_id"],
                           "decision": "allow_run"}})
    assert h.respond(8)["ok"]
    res = h.respond(7)
    assert res["ok"] and res["result"]["reply"] == "True,True"
    # second identical call was served from the cache: exactly one event
    perms = [f for f in h.frames() if f["kind"] == "permission_request"]
    assert len(perms) == 1
    assert h.server.was_allowed("shell", "ls")
    h.stop()


def test_permission_bad_request_id_and_decision_are_reported():
    def run(h, task, apply, refine, plan, r):
        ok, _ = security.request_permission("web", "fetch", detail="/x",
                                            key="example.com")
        r.reply = str(ok)
        return r

    h = Harness(run=run)
    h.inp.feed({"id": 1, "method": "ask", "params": {"task": "gate"}})
    ev = h.wait(lambda f: f["kind"] == "permission_request", "perm event")
    h.inp.feed({"id": 2, "method": "permission_reply",
                "params": {"request_id": "perm-99", "decision": "deny"}})
    f = h.respond(2)
    assert not f["ok"] and "no pending" in f["error"]
    h.inp.feed({"id": 3, "method": "permission_reply",
                "params": {"request_id": ev["request_id"],
                           "decision": "yes_please"}})
    f = h.respond(3)
    assert not f["ok"] and "decision must be" in f["error"]
    # a denial is a real answer: the run continues, ungated
    h.inp.feed({"id": 4, "method": "permission_reply",
                "params": {"request_id": ev["request_id"],
                           "decision": "deny"}})
    res = h.respond(1)
    assert res["ok"] and res["result"]["reply"] == "False"
    h.stop()


def test_permission_timeout_denies_like_no_answer():
    def run(h, task, apply, refine, plan, r):
        allowed, msg = security.request_permission("shell", "x", key="x")
        r.reply = str(allowed)
        assert msg == security.DENIED_MESSAGE
        return r

    h = Harness(run=run, permission_timeout=1.0)
    h.inp.feed({"id": 1, "method": "ask", "params": {"task": "gate"}})
    h.wait(lambda f: f["kind"] == "permission_request", "perm")
    res = h.respond(1, timeout=10)
    assert res["ok"] and res["result"]["reply"] == "False"
    h.stop()


def test_always_allow_flips_the_live_knob_via_glue():
    flipped = []

    def run(h, task, apply, refine, plan, r):
        ok, _ = security.request_permission("shell", "pytest", detail="p",
                                            key="pytest")
        r.reply = f"{ok}"
        return r

    h = Harness(run=run,
                always_apply=lambda cat, key, sk: flipped.append(
                    (cat, key, sk)))
    h.inp.feed({"id": 1, "method": "ask", "params": {"task": "gate"}})
    ev = h.wait(lambda f: f["kind"] == "permission_request", "perm")
    h.inp.feed({"id": 2, "method": "permission_reply",
                "params": {"request_id": ev["request_id"],
                           "decision": "always_allow"}})
    res = h.respond(1)
    assert res["ok"] and res["result"]["reply"] == "True"
    assert flipped == [("shell", "pytest", "")]
    # allow_always is accepted as the GUI's spelling too
    assert h.server.was_allowed("shell", "pytest")
    h.stop()


def test_cancel_denies_parked_permission_and_keeps_session_warm():
    def run(h, task, apply, refine, plan, r):
        ok, msg = security.request_permission("shell", "rm -rf", detail="x",
                                              key="rm")
        r.reply = f"{ok}:{'DENIED' in msg}"
        return r

    h = Harness(run=run)
    h.inp.feed({"id": 1, "method": "ask", "params": {"task": "gate"}})
    ev = h.wait(lambda f: f["kind"] == "permission_request", "perm")
    h.inp.feed({"id": 2, "method": "cancel"})
    assert h.respond(2)["result"]["active"] is True
    res = h.respond(1, timeout=10)
    assert not res["ok"] and res["error"] == "cancelled"
    assert res["result"]["reply"].startswith("False")  # parked call denied
    h.inp.feed({"id": 3, "method": "status"})
    assert h.respond(3)["ok"]
    h.stop()


def test_switch_profile_main_and_scout_and_guard():
    h = Harness()
    h.inp.feed({"id": 1, "method": "switch_profile",
                "params": {"profile": "big", "kind": "main"}})
    f = h.respond(1)
    assert f["ok"] and f["result"]["profile"] == "main:big"
    h.inp.feed({"id": 2, "method": "switch_profile",
                "params": {"profile": "scouty", "kind": "scout"}})
    assert h.respond(2)["result"]["profile"] == "scout:scouty"
    h.inp.feed({"id": 3, "method": "switch_profile",
                "params": {"profile": "x", "kind": "weird"}})
    assert not h.respond(3)["ok"]
    h.inp.feed({"id": 4, "method": "switch_profile", "params": {}})
    assert "missing params.profile" in h.respond(4)["error"]
    h.stop()


def test_switch_profile_refused_while_busy():
    release = threading.Event()

    def run(h, task, apply, refine, plan, r):
        release.wait(5)
        return r

    h = Harness(run=run)
    h.inp.feed({"id": 1, "method": "ask", "params": {"task": "slow"}})
    time.sleep(0.1)
    h.inp.feed({"id": 2, "method": "switch_profile",
                "params": {"profile": "big"}})
    f = h.respond(2)
    assert not f["ok"] and f["error"].startswith("busy")
    release.set()
    assert h.respond(1)["ok"]
    h.stop()


def test_explain_returns_reply_and_response_frames():
    h = Harness()
    h.inp.feed({"id": 1, "method": "explain",
                "params": {"task": "how does sync retry?"}})
    reply = h.wait(lambda f: f["kind"] == "reply", "reply frame")
    assert reply["req_id"] == "1"
    assert "about how does sync retry?" in reply["reply"]
    res = h.respond(1)
    assert res["ok"] and res["result"]["reply"] == reply["reply"]
    h.stop()


def test_llm_error_frame_leaves_the_session_warm():
    fired = {"once": False}

    def run(h, task, apply, refine, plan, r):
        if not fired["once"]:
            fired["once"] = True
            raise llm.LLMError("server died")
        return r

    h = Harness(run=run)
    h.inp.feed({"id": 1, "method": "ask", "params": {"task": "x"}})
    f = h.respond(1)
    assert not f["ok"] and "server died" in f["error"]
    h.inp.feed({"id": 2, "method": "ask", "params": {"task": "y"}})
    assert h.respond(2)["ok"]
    h.stop()


def test_worker_crash_is_contained():
    first = {"done": False}

    def run(h, task, apply, refine, plan, r):
        if not first["done"]:
            first["done"] = True
            raise RuntimeError("kaboom")
        return r

    h = Harness(run=run)
    h.inp.feed({"id": 1, "method": "ask", "params": {"task": "x"}})
    f = h.respond(1)
    assert not f["ok"] and "kaboom" in f["error"]
    h.inp.feed({"id": 2, "method": "ask", "params": {"task": "y"}})
    assert h.respond(2)["ok"]
    h.stop()


def test_eof_still_flushes_the_active_task_reply():
    def run(h, task, apply, refine, plan, r):
        time.sleep(0.6)                     # outlast stdin
        return r

    h = Harness(run=run)
    h.inp.feed({"id": 1, "method": "ask", "params": {"task": "slow"}})
    time.sleep(0.1)                         # the worker has barely begun
    h.inp.close()                           # client hangs up mid-task
    h.thread.join(timeout=15)
    assert not h.thread.is_alive(), "serve() did not exit at EOF"
    replies = [f for f in h.frames() if f["kind"] == "rpc_response"]
    assert len(replies) == 1 and replies[0]["id"] == 1 and replies[0]["ok"]
    assert any(f["kind"] == "result" and f["req_id"] == "1"
               for f in h.frames())         # engine frames flushed too


def test_rpc_out_rejects_unparseable_lines_to_stderr(capsys):
    buf = io.StringIO()
    out = rpc.RpcOut(buf, lambda: "r1")
    out.write('{"kind":"step"}\n')
    out.write("this is not json\n")            # must never reach stdout raw
    out.write('{"kind":"result"}\n{"partial":tru')
    lines = [json.loads(l) for l in buf.getvalue().splitlines()]
    assert [l["kind"] for l in lines] == ["step", "result"]
    assert all(l["req_id"] == "r1" for l in lines)
    err = capsys.readouterr().err
    assert "this is not json" in err
    out.close()
    out.write('{"kind":"late"}\n')             # closed: silent drop
    assert len(buf.getvalue().splitlines()) == 2
