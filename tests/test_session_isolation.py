"""WP-4001 tests: SessionContext, context propagation, per-session EventBus,
RunTelemetry, SessionPermissionBroker, GUI SessionRegistry/StateProxy/Tee,
and session-scoped tool caches."""

from __future__ import annotations

import contextvars
import http.client
import io
import json
import sys
import threading
import time
from concurrent.futures import as_completed
from pathlib import Path
from types import SimpleNamespace

import pytest

from pedacito import config as config_mod
from pedacito import events as events_mod
from pedacito import security, session_ctx, tools
from pedacito.gui import core as gui_core
from pedacito.gui import jobs as gui_jobs
from pedacito.gui import server as gui_server


# ---------------------------------------------------------------------------
# 1. BudgetLedger & SessionContext lifecycle
# ---------------------------------------------------------------------------


def test_budget_ledger_thread_safe_recording_and_limits():
    ledger = session_ctx.BudgetLedger(
        max_cost_usd=1.0, max_tokens=1000, max_steps=50
    )
    ok, reason = ledger.check()
    assert ok and reason == ""

    def worker():
        for _ in range(10):
            ledger.record(cost_usd=0.01, tokens=5, steps=1)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = ledger.snapshot()
    assert snap["spent_steps"] == 40
    assert snap["spent_tokens"] == 200
    assert snap["spent_cost_usd"] == pytest.approx(0.4, abs=1e-5)
    assert snap["ok"] is True

    ledger.record(steps=10)
    ok, reason = ledger.check()
    assert not ok
    assert "step budget exceeded" in reason

    cost_ledger = session_ctx.BudgetLedger(max_cost_usd=0.5)
    cost_ledger.record(cost_usd=0.5)
    ok, reason = cost_ledger.check()
    assert not ok and "cost budget exceeded" in reason

    tok_ledger = session_ctx.BudgetLedger(max_tokens=100)
    tok_ledger.record(tokens=100)
    ok, reason = tok_ledger.check()
    assert not ok and "token budget exceeded" in reason


def test_session_context_use_require_cache_cancel_close(tmp_path):
    assert session_ctx.current_session() is None
    with pytest.raises(RuntimeError):
        session_ctx.require_session()

    bus = events_mod.SessionEventBus("ses_lifecycle")
    broker = security.SessionPermissionBroker(
        "ses_lifecycle", interactive=True, timeout_s=5.0, event_bus=bus
    )
    client_cancelled = []
    lease_released = []
    resources_closed = []

    fake_client = SimpleNamespace(cancel=lambda: client_cancelled.append(True))
    fake_lease = SimpleNamespace(release=lambda: lease_released.append(True))
    fake_registry = SimpleNamespace(
        close_all=lambda: (resources_closed.append(True), 3)[1]
    )

    ctx = session_ctx.SessionContext(
        session_id="ses_lifecycle",
        root=tmp_path,
        client=fake_client,
        event_bus=bus,
        permission_broker=broker,
        model_lease=fake_lease,
        resource_registry=fake_registry,
    )
    assert ctx.created_at > 0.0
    assert ctx.root == tmp_path

    # Per-namespace caches are isolated
    c1 = ctx.cache("a")
    c2 = ctx.cache("b")
    c1["k"] = 1
    assert "k" not in c2
    assert ctx.cache("a") is c1

    # Nested use_session restores cleanly
    outer = session_ctx.SessionContext(session_id="ses_outer")
    with session_ctx.use_session(outer):
        assert session_ctx.require_session() is outer
        with session_ctx.use_session(ctx):
            assert session_ctx.require_session() is ctx
        assert session_ctx.require_session() is outer
    assert session_ctx.current_session() is None

    # Park a pending permission request, then close the session
    perm_result: dict[str, tuple[bool, str]] = {}

    def ask_perm():
        with session_ctx.use_session(ctx):
            perm_result["res"] = security.request_permission("shell", "run ls", detail="ls")

    t = threading.Thread(target=ask_perm)
    t.start()
    deadline = time.monotonic() + 3.0
    while not broker.pending() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(broker.pending()) == 1

    report = ctx.close(reason="user_closed")
    t.join(timeout=3.0)
    assert not t.is_alive()
    assert ctx.cancel_event.is_set()
    assert report["permissions_denied"] == 1
    assert report["client_cancelled"] is True
    assert report["lease_released"] is True
    assert report["resources_closed"] == 3
    assert report["closed"] is True
    assert bus.closed is True
    assert client_cancelled == [True]
    assert lease_released == [True]
    assert resources_closed == [True]
    assert perm_result["res"][0] is False
    envs, _ = bus.replay(0)
    assert any(e.kind == "session_cancelled" for e in envs)


# ---------------------------------------------------------------------------
# 2. Context propagation across threads & ContextThreadPoolExecutor
# ---------------------------------------------------------------------------


def test_spawn_thread_and_executor_propagate_isolated_contexts():
    ctx_a = session_ctx.SessionContext(session_id="ses_prop_a")
    ctx_b = session_ctx.SessionContext(session_id="ses_prop_b")

    seen_a: list[str] = []
    seen_b: list[str] = []
    errors: list[BaseException] = []

    def run_session_work(ctx: session_ctx.SessionContext, out: list[str]):
        try:
            with session_ctx.use_session(ctx):
                # Explicitly reuse one captured Context across multiple threads
                captured = contextvars.copy_context()
                barrier = threading.Barrier(4)

                def sub_thread():
                    barrier.wait(timeout=3.0)
                    out.append(session_ctx.require_session().session_id)

                sub_threads = [
                    session_ctx.spawn_thread_with_context(
                        sub_thread, ctx=captured, name=f"sub-{ctx.session_id}-{i}"
                    )
                    for i in range(4)
                ]
                for st in sub_threads:
                    st.join(timeout=3.0)

                # Also exercise ContextThreadPoolExecutor submit and map
                with session_ctx.ContextThreadPoolExecutor(max_workers=4) as pool:
                    futs = [
                        pool.submit(lambda _i: session_ctx.require_session().session_id, i)
                        for i in range(12)
                    ]
                    for fut in as_completed(futs):
                        out.append(fut.result())
                    mapped = list(
                        pool.map(
                            lambda _x: session_ctx.require_session().session_id,
                            range(8),
                        )
                    )
                    out.extend(mapped)
        except BaseException as exc:
            errors.append(exc)

    ta = threading.Thread(target=run_session_work, args=(ctx_a, seen_a))
    tb = threading.Thread(target=run_session_work, args=(ctx_b, seen_b))
    ta.start()
    tb.start()
    ta.join(timeout=5.0)
    tb.join(timeout=5.0)

    assert not errors, f"unexpected propagation errors: {errors!r}"
    assert len(seen_a) == 24 and set(seen_a) == {"ses_prop_a"}
    assert len(seen_b) == 24 and set(seen_b) == {"ses_prop_b"}


# ---------------------------------------------------------------------------
# 3. SessionEventBus, EventLog JSONL stamping & RunTelemetry isolation
# ---------------------------------------------------------------------------


def test_session_event_bus_monotonic_replay_wait_redaction_and_sink(tmp_path):
    root_log = events_mod.EventLog(tmp_path, enabled=True, redact=True)
    bus = events_mod.SessionEventBus("ses_bus_1", sink=root_log, cap=5, redact=True)

    heard: list[events_mod.EventEnvelope] = []
    bus.add_listener(heard.append)
    bus.add_listener(lambda _e: 1 / 0)  # failing listener must never break emit

    env1 = bus.emit("step", note="secret sk-abcdefghijklmnopqrstuvwxyz0123456789")
    assert env1.seq == 1
    assert env1.session_id == "ses_bus_1"
    assert "sk-abcdef" not in env1.payload["note"]
    d1 = env1.to_dict()
    assert d1["seq"] == 1 and d1["session_id"] == "ses_bus_1" and d1["kind"] == "step"

    for i in range(2, 8):
        env = bus.emit("step", idx=i)
        assert env.seq == i
    assert bus.seq == 7
    assert len(heard) == 7

    # Cap is 5 -> oldest retained seq is 3
    missed, resync = bus.replay(after_seq=1)
    assert resync is True and missed == []
    future, resync = bus.replay(after_seq=99)
    assert resync is True and future == []
    tail, resync = bus.replay(after_seq=5)
    assert resync is False and [e.seq for e in tail] == [6, 7]

    # Wait unblocks when a new event arrives
    wait_out: dict[str, object] = {}

    def waiter():
        wait_out["res"] = bus.wait(after_seq=7, timeout=3.0)

    wt = threading.Thread(target=waiter)
    wt.start()
    time.sleep(0.05)
    bus.emit("done", ok=True)
    wt.join(timeout=3.0)
    evs, resync = wait_out["res"]
    assert not resync and len(evs) == 1 and evs[0].seq == 8 and evs[0].kind == "done"

    # JSONL sink got every event ONCE with session_id and seq
    assert root_log.path is not None
    rows = [
        json.loads(line)
        for line in root_log.path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 8
    assert [r["seq"] for r in rows] == list(range(1, 9))
    assert all(r["session_id"] == "ses_bus_1" for r in rows)
    assert "sk-abcdef" not in root_log.path.read_text(encoding="utf-8")

    # Using events.get() inside SessionContext does NOT double-write sink
    ctx = session_ctx.SessionContext(session_id="ses_bus_1", root=tmp_path, event_bus=bus)
    with session_ctx.use_session(ctx):
        # Register root_log as the root's log so adapter sees bus._sink is log
        with events_mod._registry_lock:
            events_mod._registry[str(tmp_path.resolve())] = root_log
        events_mod.get(tmp_path).emit("extra", val=42)
    rows2 = [
        json.loads(line)
        for line in root_log.path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows2) == 9
    assert rows2[-1]["seq"] == 9 and rows2[-1]["session_id"] == "ses_bus_1"

    bus.remove_listener(heard.append)
    bus.close()
    assert bus.closed is True


def test_concurrent_sessions_event_streams_and_telemetry_isolated(tmp_path):
    bus_a = events_mod.SessionEventBus("ses_tel_a")
    bus_b = events_mod.SessionEventBus("ses_tel_b")
    ctx_a = session_ctx.SessionContext(session_id="ses_tel_a", root=tmp_path, event_bus=bus_a)
    ctx_b = session_ctx.SessionContext(session_id="ses_tel_b", root=tmp_path, event_bus=bus_b)

    summaries: dict[str, dict] = {}
    barrier = threading.Barrier(2)

    def run_session(ctx: session_ctx.SessionContext, tag: str, tokens_per_call: int):
        with session_ctx.use_session(ctx):
            events_mod.begin_run()
            barrier.wait(timeout=3.0)
            with session_ctx.ContextThreadPoolExecutor(max_workers=4) as pool:
                def sub_call(i: int):
                    events_mod.get(tmp_path).emit("work", tag=tag, i=i)
                    events_mod.note_call(
                        {
                            "prompt_tokens": tokens_per_call,
                            "completion_tokens": 10,
                            "total_tokens": tokens_per_call + 10,
                            "seconds": 0.1,
                            "model": f"model-{tag}",
                            "provider": "local",
                        }
                    )

                list(pool.map(sub_call, range(8)))
            summaries[tag] = events_mod.end_run()

    ta = threading.Thread(target=run_session, args=(ctx_a, "A", 100))
    tb = threading.Thread(target=run_session, args=(ctx_b, "B", 500))
    ta.start()
    tb.start()
    ta.join(timeout=5.0)
    tb.join(timeout=5.0)

    # Telemetry totals never leaked across sessions and accumulated all 8 sub-threads
    assert summaries["A"]["calls"] == 8
    assert summaries["A"]["prompt_tokens"] == 800
    assert summaries["A"]["model"] == "model-A"
    assert summaries["B"]["calls"] == 8
    assert summaries["B"]["prompt_tokens"] == 4000
    assert summaries["B"]["model"] == "model-B"

    # Event buses are strictly isolated
    evs_a, _ = bus_a.replay(0)
    evs_b, _ = bus_b.replay(0)
    assert len(evs_a) == 8 and all(e.payload["tag"] == "A" for e in evs_a)
    assert len(evs_b) == 8 and all(e.payload["tag"] == "B" for e in evs_b)
    assert [e.seq for e in evs_a] == list(range(1, 9))
    assert [e.seq for e in evs_b] == list(range(1, 9))


# ---------------------------------------------------------------------------
# 4. SessionPermissionBroker isolation & non-interactive deny
# ---------------------------------------------------------------------------


def test_permission_broker_non_interactive_and_policy_never():
    bus1 = events_mod.SessionEventBus("ses_ni_1")
    broker_ni = security.SessionPermissionBroker("ses_ni_1", event_bus=bus1)
    ctx1 = session_ctx.SessionContext(
        session_id="ses_ni_1", event_bus=bus1, permission_broker=broker_ni
    )
    with session_ctx.use_session(ctx1):
        assert security.permission_broker_active() is True
        allowed, msg = security.request_permission("shell", "run echo", detail="echo hi")
    assert allowed is False
    assert msg == security.DENIED_MESSAGE
    evs, _ = bus1.replay(0)
    assert len(evs) == 1
    assert evs[0].kind == "permission_decision"
    assert evs[0].payload["decision"] == "deny"
    assert evs[0].payload["reason"] == "non_interactive_or_policy_never"

    # Even with a responder, approval_policy="never" denies immediately
    called = []
    bus2 = events_mod.SessionEventBus("ses_ni_2")
    broker_never = security.SessionPermissionBroker(
        "ses_ni_2",
        responder=lambda req: (called.append(req) or True, ""),
        approval_policy="never",
        event_bus=bus2,
    )
    allowed2, msg2 = broker_never.request({"category": "shell", "detail": "echo hi"})
    assert allowed2 is False and msg2 == security.DENIED_MESSAGE
    assert called == []
    evs2, _ = bus2.replay(0)
    assert evs2[-1].kind == "permission_decision"
    assert evs2[-1].payload["reason"] == "non_interactive_or_policy_never"


def test_permission_broker_concurrent_sessions_and_cross_session_rejection():
    bus_a = events_mod.SessionEventBus("ses_perm_a")
    bus_b = events_mod.SessionEventBus("ses_perm_b")
    broker_a = security.SessionPermissionBroker(
        "ses_perm_a", interactive=True, timeout_s=5.0, event_bus=bus_a
    )
    broker_b = security.SessionPermissionBroker(
        "ses_perm_b", interactive=True, timeout_s=5.0, event_bus=bus_b
    )
    ctx_a = session_ctx.SessionContext(
        session_id="ses_perm_a", event_bus=bus_a, permission_broker=broker_a
    )
    ctx_b = session_ctx.SessionContext(
        session_id="ses_perm_b", event_bus=bus_b, permission_broker=broker_b
    )

    results: dict[str, tuple[bool, str]] = {}

    def ask_in(ctx: session_ctx.SessionContext, name: str):
        with session_ctx.use_session(ctx):
            results[name] = security.request_permission(
                security.CAT_SHELL, f"run {name}", detail=f"pytest {name}", key=f"k-{name}"
            )

    ta = threading.Thread(target=ask_in, args=(ctx_a, "A"))
    tb = threading.Thread(target=ask_in, args=(ctx_b, "B"))
    ta.start()
    tb.start()

    deadline = time.monotonic() + 3.0
    while (not broker_a.pending() or not broker_b.pending()) and time.monotonic() < deadline:
        time.sleep(0.01)

    req_a = broker_a.pending()[0]
    req_b = broker_b.pending()[0]
    assert req_a["session_id"] == "ses_perm_a"
    assert req_b["session_id"] == "ses_perm_b"
    # Both got "perm-1" locally — cross-answering MUST be structurally rejected!
    assert req_a["request_id"] == "perm-1"
    assert req_b["request_id"] == "perm-1"

    ok, detail = broker_a.decide("ses_perm_b", req_a["request_id"], "allow_once")
    assert ok is False and detail == "session mismatch"
    assert len(broker_a.pending()) == 1

    # Decide A=allow_run, B=deny
    ok_a, _ = broker_a.decide("ses_perm_a", req_a["request_id"], "allow_run")
    ok_b, _ = broker_b.decide("ses_perm_b", req_b["request_id"], "deny")
    assert ok_a is True and ok_b is True

    ta.join(timeout=3.0)
    tb.join(timeout=3.0)
    assert results["A"] == (True, "")
    assert results["B"] == (False, security.DENIED_MESSAGE)

    # Run grant applies ONLY to broker A, and clear_run_grants clears it
    with session_ctx.use_session(ctx_a):
        assert security.request_permission(
            security.CAT_SHELL, "run A again", detail="pytest A", key="k-A"
        ) == (True, "")
    broker_a.clear_run_grants()
    assert broker_a.pending() == []

    # Verify Session A's run grant did NOT grant anything to Session B
    broker_b_ni = security.SessionPermissionBroker("ses_perm_b")
    assert broker_b_ni.request({"category": security.CAT_SHELL, "key": "k-A"})[0] is False


def test_security_session_ack_isolated_across_sessions():
    fp = "a" * 64
    ctx_a = session_ctx.SessionContext(session_id="ses_ack_a")
    ctx_b = session_ctx.SessionContext(session_id="ses_ack_b")

    with session_ctx.use_session(ctx_a):
        security.acknowledge_findings(None, [fp])
        assert fp in security.load_acknowledged_fingerprints(None)

    with session_ctx.use_session(ctx_b):
        assert fp not in security.load_acknowledged_fingerprints(None)


# ---------------------------------------------------------------------------
# 5. GUI SessionRegistry, StateProxy, Tee routing, jobs cfg clone & server routes
# ---------------------------------------------------------------------------


def test_gui_state_proxy_and_tee_route_per_session():
    ctx_a = session_ctx.SessionContext(session_id="ses_gui_a")
    ctx_b = session_ctx.SessionContext(session_id="ses_gui_b")
    orig_streams = (sys.stdout, sys.stderr)
    fake_out = io.StringIO()
    fake_err = io.StringIO()
    sys.stdout = fake_out
    sys.stderr = fake_err
    installed = gui_core.install_tee()
    try:
        with session_ctx.use_session(ctx_a):
            gui_core.STATE.add_message("user", "hello from A")
            print("stdout-only-A")
        with session_ctx.use_session(ctx_b):
            gui_core.STATE.add_message("user", "hello from B")
            print("stdout-only-B")

        st_a = gui_core.REGISTRY.get("ses_gui_a")
        st_b = gui_core.REGISTRY.get("ses_gui_b")
        assert st_a is not None and st_b is not None and st_a is not st_b
        assert "stdout-only-A" in st_a.term
        assert "stdout-only-B" not in st_a.term
        assert "stdout-only-B" in st_b.term
        assert "stdout-only-A" not in st_b.term
        assert [m["text"] for m in st_a.messages] == ["hello from A"]
        assert [m["text"] for m in st_b.messages] == ["hello from B"]
    finally:
        gui_core.uninstall_tee(installed)
        sys.stdout, sys.stderr = orig_streams
        gui_core.REGISTRY.remove("ses_gui_a")
        gui_core.REGISTRY.remove("ses_gui_b")


def test_gui_jobs_spawn_propagates_session_and_clones_cfg(tmp_path):
    ctx_s = session_ctx.SessionContext(session_id="ses_job_clone", root=tmp_path)
    base_cfg = config_mod.Config()
    base_cfg.max_steps = 12
    base_cfg.hybrid_routing = False
    observed: dict[str, object] = {}

    try:
        with session_ctx.use_session(ctx_s):
            gui_core.STATE.root = tmp_path
            gui_core.STATE.cfg = base_cfg

            def body(job):
                sess = session_ctx.require_session()
                observed["session_id"] = sess.session_id
                observed["cfg_is_clone"] = sess.cfg is not base_cfg
                sess.cfg.max_steps = 99
                sess.cfg.hybrid_routing = True
                sess.cfg._pinned_refs = ["secret.py"]
                p = gui_jobs._project(need_index=False)
                observed["project_cfg_is_clone"] = p.cfg is not base_cfg
                p.cfg.max_steps = 77

            job = gui_jobs.spawn("task", "clone check", body)
            assert job is not None and job.thread is not None
            job.thread.join(timeout=5.0)

        assert observed["session_id"] == "ses_job_clone"
        assert observed["cfg_is_clone"] is True
        assert observed["project_cfg_is_clone"] is True
        assert base_cfg.max_steps == 12
        assert base_cfg.hybrid_routing is False
        assert not hasattr(base_cfg, "_pinned_refs")
    finally:
        gui_core.REGISTRY.remove("ses_job_clone")


def test_gui_server_events_and_permission_decide_session_scoped():
    st_a = gui_core.REGISTRY.get_or_create("ses_srv_a")
    st_b = gui_core.REGISTRY.get_or_create("ses_srv_b")
    srv = gui_server.create_server(0)
    port = srv.server_address[1]
    t_srv = threading.Thread(target=srv.serve_forever, daemon=True)
    t_srv.start()

    perm_out: dict[str, tuple[bool, str]] = {}

    def ask_in_a():
        perm_out["res"] = st_a.request_permission(
            {"category": security.CAT_SHELL, "action_summary": "run", "detail": "pytest"}
        )

    try:
        # 1. SSE /api/events?session_id=ses_srv_a receives only ses_srv_a events
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
        conn.request("GET", "/api/events?session_id=ses_srv_a&since=0")
        resp = conn.getresponse()
        assert resp.status == 200
        st_b.bus.publish("note", text="from-b")
        st_a.bus.publish("note", text="from-a")
        raw = b""
        deadline = time.monotonic() + 3.0
        while b"from-a" not in raw and time.monotonic() < deadline:
            chunk = resp.read1(1024)
            if not chunk:
                break
            raw += chunk
        conn.close()
        text = raw.decode("utf-8", errors="replace")
        assert "from-a" in text
        assert "from-b" not in text

        # Unknown session_id on /api/events -> 404
        conn404 = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
        conn404.request("GET", "/api/events?session_id=ses_does_not_exist")
        r404 = conn404.getresponse()
        r404.read()
        conn404.close()
        assert r404.status == 404

        # 2. /api/permission/decide enforces session_id matching
        tp = threading.Thread(target=ask_in_a, daemon=True)
        tp.start()
        deadline = time.monotonic() + 3.0
        while st_a.pending_permission is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert st_a.pending_permission is not None
        pid = st_a.pending_permission["id"]

        def post_json(path: str, payload: dict) -> tuple[int, dict]:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
            body = json.dumps(payload).encode("utf-8")
            c.request(
                "POST",
                path,
                body=body,
                headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
            )
            r = c.getresponse()
            data = json.loads(r.read().decode("utf-8"))
            c.close()
            return r.status, data

        # Wrong session_id (other existing session) -> 409
        code_wrong, _ = post_json(
            "/api/permission/decide",
            {"id": pid, "decision": "allow_once", "session_id": "ses_srv_b"},
        )
        assert code_wrong == 409

        # Unknown session_id -> 409 session mismatch
        code_unk, body_unk = post_json(
            "/api/permission/decide",
            {"id": pid, "decision": "allow_once", "session_id": "ses_ghost"},
        )
        assert code_unk == 409 and body_unk["detail"] == "session mismatch"

        # Matching session_id -> 200 and unblocks worker
        code_ok, body_ok = post_json(
            "/api/permission/decide",
            {"id": pid, "decision": "allow_once", "session_id": "ses_srv_a"},
        )
        assert code_ok == 200 and body_ok["ok"] is True
        tp.join(timeout=3.0)
        assert perm_out["res"] == (True, "")
    finally:
        st_a.bus.close()
        st_b.bus.close()
        srv.shutdown()
        srv.server_close()
        gui_core.REGISTRY.remove("ses_srv_a")
        gui_core.REGISTRY.remove("ses_srv_b")


# ---------------------------------------------------------------------------
# 6. tools._RUN_TEST_RUNNER_CACHE session scoping
# ---------------------------------------------------------------------------


def test_run_test_runner_cache_is_session_scoped(tmp_path, monkeypatch):
    probes: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        probes.append(list(argv))
        return 0, "pytest 8.0", False, 0.01

    monkeypatch.setattr(tools.proc, "run", fake_run)
    cfg = config_mod.Config()
    argv = [sys.executable, "-m", "pytest"]

    ctx_a = session_ctx.SessionContext(session_id="ses_cache_a")
    ctx_b = session_ctx.SessionContext(session_id="ses_cache_b")

    with session_ctx.use_session(ctx_a):
        assert tools._pytest_runner_ok(argv, tmp_path, cfg) is True
        assert tools._pytest_runner_ok(argv, tmp_path, cfg) is True
    assert len(probes) == 1  # second call in session A hit session A's cache

    with session_ctx.use_session(ctx_b):
        assert tools._pytest_runner_ok(argv, tmp_path, cfg) is True
    assert len(probes) == 2  # session B had its own empty cache

    assert str(Path(argv[0])) in ctx_a.cache("run_test_runner")
    assert str(Path(argv[0])) in ctx_b.cache("run_test_runner")
