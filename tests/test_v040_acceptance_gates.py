"""WP-4007: the twelve mandatory v0.4.0 acceptance gates, proven end to end.

Each gate is one test class (``TestGate01`` ... ``TestGate12``). The gates
compose the real engine surfaces (session contexts, the event bus, the
daemon, leases, project locks, the WAL journal, worktrees, the policy engine,
the sandbox gate and the bilingual docs) and do not stub the behaviour they
prove. Offline and platform-neutral: no network, no third-party packages.
Live Windows sandbox checks are skipped elsewhere.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import site
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import fields
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import (  # noqa: E402
    checkpoints,
    config as config_mod,
    daemon,
    edits,
    events,
    fileio,
    how,
    leases,
    policy,
    proc,
    ratchet,
    review,
    rpc,
    sandbox,
    sandbox_win,
    security,
    session_ctx,
    tools,
    worktree,
    workspace,
    workspace_overlay,
)
from pedacito import __version__  # noqa: E402
from pedacito.config import Config  # noqa: E402
from pedacito.index import Index  # noqa: E402
from pedacito.journal import (  # noqa: E402
    TaskJournal,
    find_incomplete_tasks,
    read_journal,
    rollback_incomplete_task,
)
from pedacito.session_ctx import (  # noqa: E402
    BudgetLedger,
    ContextThreadPoolExecutor,
    SessionContext,
    spawn_thread_with_context,
    use_session,
)

REPO = Path(__file__).resolve().parents[1]
LOCAL = "http://127.0.0.1:1234/v1"
MODES = ("plan", "ask", "accept-edits", "workspace-write", "danger")
CAPABILITIES = ("run", "hook", "mcp", "write", "subagent")


def _cfg(mode: str = "ask", approval: str = "ask", sandbox_on: bool = True) -> Config:
    cfg = Config()
    cfg.permission_mode = mode
    cfg.approval_policy = approval
    cfg.sandbox = sandbox_on
    return cfg


_USER_SITE = site.getusersitepackages()


def _child_env() -> dict[str, str]:
    """Environment for child interpreters. The suite isolates APPDATA, which
    hides the per-user site-packages from children, so put it back on the
    path explicitly; the child still uses the isolated pedacito config."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO), _USER_SITE, env.get("PYTHONPATH", "")) if p)
    return env


def _git(root: Path, *args: str) -> str:
    proc_out = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        timeout=60, check=True, shell=False)
    return proc_out.stdout.strip()


def _repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    (root / "app.py").write_text("x = 1\n", encoding="utf-8")
    _git(root, "add", "app.py")
    _git(root, "-c", "user.email=t@example.com", "-c", "user.name=t",
         "commit", "-q", "-m", "init")
    return root


def _wait_for(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


# ============================================================================
# Gate 1 - zero global leakage between concurrent sessions
# ============================================================================

class TestGate01ZeroGlobalLeakage:
    """Two concurrent sessions in one process share no runtime state."""

    def test_two_sessions_do_not_share_cfg_index_client_caches_budget_or_telemetry(
            self, tmp_path):
        root_a, root_b = tmp_path / "alpha", tmp_path / "beta"
        root_a.mkdir()
        root_b.mkdir()
        cfg_a = _cfg("plan", "never")
        cfg_b = _cfg("danger", "ask")
        cfg_a.max_tokens = 111
        cfg_b.max_tokens = 222
        idx_a, idx_b = Index(root_a, []), Index(root_b, [])
        client_a, client_b = object(), object()

        ctx_a = SessionContext(session_id="ses_gate1_a", root=root_a, cfg=cfg_a,
                               idx=idx_a, client=client_a,
                               budget_ledger=BudgetLedger(max_steps=3),
                               policy=policy.PolicyEngine(cfg_a, root_a),
                               mode="daemon")
        ctx_b = SessionContext(session_id="ses_gate1_b", root=root_b, cfg=cfg_b,
                               idx=idx_b, client=client_b,
                               budget_ledger=BudgetLedger(max_steps=9),
                               policy=policy.PolicyEngine(cfg_b, root_b),
                               mode="daemon")
        barrier = threading.Barrier(2)

        def worker(ctx: SessionContext, n_calls: int) -> dict:
            with use_session(ctx):
                barrier.wait(timeout=10)
                run = events.begin_run()
                for i in range(n_calls):
                    session_ctx.require_session().cache("tools")[f"k{i}"] = ctx.session_id
                    events.note_call({"prompt_tokens": 10, "completion_tokens": 5,
                                      "total_tokens": 15, "session": ctx.session_id})
                    session_ctx.require_session().budget_ledger.record(
                        steps=1, tokens=15, cost_usd=0.0)
                summary = events.end_run()
                return {"session": session_ctx.require_session().session_id,
                        "summary": summary, "run": run}

        with ContextThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(worker, ctx_a, 4)
            fut_b = pool.submit(worker, ctx_b, 7)
            res_a, res_b = fut_a.result(10), fut_b.result(10)

        assert res_a["session"] == "ses_gate1_a" and res_b["session"] == "ses_gate1_b"
        assert ctx_a.cfg is cfg_a and ctx_b.cfg is cfg_b
        assert ctx_a.cfg.max_tokens == 111 and ctx_b.cfg.max_tokens == 222
        assert ctx_a.cfg.permission_mode == "plan" and ctx_b.cfg.permission_mode == "danger"
        assert ctx_a.idx is idx_a and ctx_b.idx is idx_b
        assert ctx_a.client is client_a and ctx_b.client is client_b
        assert ctx_a.tool_caches is not ctx_b.tool_caches
        assert set(ctx_a.cache("tools").values()) == {"ses_gate1_a"}
        assert set(ctx_b.cache("tools").values()) == {"ses_gate1_b"}
        assert ctx_a.budget_ledger.snapshot() != ctx_b.budget_ledger.snapshot()
        assert ctx_a.budget_ledger.spent_steps == 4
        assert ctx_b.budget_ledger.spent_steps == 7
        assert res_a["summary"]["calls"] == 4
        assert res_b["summary"]["calls"] == 7
        assert ctx_a.policy.mode == "plan" and ctx_b.policy.mode == "danger"
        assert ctx_a.resource_registry is not ctx_b.resource_registry
        assert ctx_a.event_bus is None and ctx_b.event_bus is None

    def test_spawned_thread_and_executor_inherit_only_their_own_session(self, tmp_path):
        ctx_a = SessionContext(session_id="ses_thr_a", root=tmp_path)
        ctx_b = SessionContext(session_id="ses_thr_b", root=tmp_path)
        seen: dict[str, str] = {}

        def record(name: str) -> None:
            seen[name] = session_ctx.require_session().session_id

        with use_session(ctx_a):
            t_a = spawn_thread_with_context(record, args=("a",))
        with use_session(ctx_b):
            t_b = spawn_thread_with_context(record, args=("b",))
        t_a.join(10)
        t_b.join(10)
        assert seen == {"a": "ses_thr_a", "b": "ses_thr_b"}
        assert session_ctx.current_session() is None


# ============================================================================
# Gate 2 - event isolation (bus, daemon SSE, root JSONL)
# ============================================================================

class TestGate02EventIsolation:
    """Concurrent sessions emit into strictly their own, monotonic streams."""

    def test_bus_replay_is_session_pure_and_monotonic(self, tmp_path):
        bus_a = events.SessionEventBus("ses_ev_a")
        bus_b = events.SessionEventBus("ses_ev_b")
        barrier = threading.Barrier(2)

        def emit(bus, tag: str) -> None:
            barrier.wait(timeout=10)
            for i in range(60):
                bus.emit("token", text=f"{tag}-{i}")

        threads = [threading.Thread(target=emit, args=(bus_a, "a")),
                   threading.Thread(target=emit, args=(bus_b, "b"))]
        for th in threads:
            th.start()
        for th in threads:
            th.join(30)

        for bus, sid, tag in ((bus_a, "ses_ev_a", "a"), (bus_b, "ses_ev_b", "b")):
            envelopes, _truncated = bus.replay(0)
            assert envelopes, "bus must replay its own events"
            assert {e.session_id for e in envelopes} == {sid}
            seqs = [e.seq for e in envelopes]
            assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
            assert all(e.payload.get("text", "").startswith(tag + "-")
                       for e in envelopes if e.kind == "token")
            tail, _ = bus.replay(seqs[-1] - 5)
            assert [e.seq for e in tail] == [s for s in seqs if s > seqs[-1] - 5]

    def test_daemon_sse_and_root_jsonl_are_tagged_per_session(self, tmp_path):
        state = tmp_path / "state"
        srv = daemon.DaemonServer(state_dir=state)
        srv.start()
        try:
            root_a, root_b = tmp_path / "pa", tmp_path / "pb"
            root_a.mkdir()
            root_b.mkdir()
            ws_a = srv.create_session(root_a)
            ws_b = srv.create_session(root_b)
            barrier = threading.Barrier(2)

            def emit(ws, tag: str) -> None:
                barrier.wait(timeout=10)
                for i in range(25):
                    ws.ctx.event_bus.emit("token", text=f"{tag}{i}")

            threads = [threading.Thread(target=emit, args=(ws_a, "A")),
                       threading.Thread(target=emit, args=(ws_b, "B"))]
            for th in threads:
                th.start()
            for th in threads:
                th.join(30)

            for ws in (ws_a, ws_b):
                envs, _ = ws.ctx.event_bus.replay(0)
                assert {e.session_id for e in envs} == {ws.session_id}
                seqs = [e.seq for e in envs]
                assert seqs == sorted(seqs)
                tokens = [e.payload.get("text", "") for e in envs if e.kind == "token"]
                assert len(tokens) == 25
                assert all(t.startswith("A" if ws is ws_a else "B") for t in tokens)
        finally:
            srv.stop()

        jsonl_lines: list[dict] = []
        for root in (root_a, root_b):
            for path in (root / ".pedacito" / "events").glob("*.jsonl"):
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        jsonl_lines.append(json.loads(line))
        tagged = [r for r in jsonl_lines if r.get("session_id")]
        sids = {r["session_id"] for r in tagged}
        assert ws_a.session_id in sids and ws_b.session_id in sids
        for sid in (ws_a.session_id, ws_b.session_id):
            seqs = [r["seq"] for r in tagged if r["session_id"] == sid and "seq" in r]
            assert seqs, f"root JSONL must carry seq for {sid}"
            assert seqs == sorted(seqs)


# ============================================================================
# Gate 3 - permission isolation by (session_id, request_id)
# ============================================================================

class TestGate03PermissionIsolation:
    """Conflicting prompts route independently; background prompts deny."""

    def test_concurrent_prompts_route_by_session_and_request_id(self):
        bus = events.SessionEventBus("ses_perm_a")
        broker_a = security.SessionPermissionBroker(
            "ses_perm_a", approval_policy="ask", interactive=True,
            event_bus=bus, timeout_s=10)
        broker_b = security.SessionPermissionBroker(
            "ses_perm_b", approval_policy="ask", interactive=True,
            event_bus=events.SessionEventBus("ses_perm_b"), timeout_s=10)
        results: dict[str, tuple] = {}

        def ask(name: str, broker, key: str) -> None:
            results[name] = broker.request({
                "category": security.CAT_SHELL, "detail": f"pytest {key}",
                "key": key, "action_summary": f"run {key}", "reason": "gate3",
            })

        t_a = spawn_thread_with_context(ask, args=("a", broker_a, "conflict-cmd"))
        t_b = spawn_thread_with_context(ask, args=("b", broker_b, "conflict-cmd"))
        assert _wait_for(lambda: broker_a.pending() and broker_b.pending())
        rid_a = broker_a.pending()[0]["request_id"]
        rid_b = broker_b.pending()[0]["request_id"]

        ok, detail = broker_b.decide("ses_perm_a", rid_a, "deny")
        assert not ok and detail == "session mismatch"
        ok, detail = broker_a.decide("ses_perm_b", rid_b, "deny")
        assert not ok and detail == "session mismatch"
        assert broker_a.pending() and broker_b.pending(), \
            "a cross-session decision must not resolve either prompt"

        assert broker_a.decide("ses_perm_a", rid_a, "deny") == (True, "")
        assert broker_b.decide("ses_perm_b", rid_b, "deny") == (True, "")
        t_a.join(10)
        t_b.join(10)
        assert not results["a"][0] and not results["b"][0]
        assert broker_a.pending() == [] and broker_b.pending() == []

    def test_background_and_policy_never_prompts_deny_deterministically(self):
        bus = events.SessionEventBus("ses_headless")
        broker = security.SessionPermissionBroker(
            "ses_headless", approval_policy="never", event_bus=bus)
        started = time.monotonic()
        ok, _detail = broker.request({
            "category": security.CAT_SHELL, "detail": "rm -rf build",
            "key": "headless-cmd", "reason": "gate3"})
        assert not ok
        assert time.monotonic() - started < 2.0, "headless denial must not block"
        assert broker.pending() == []
        decisions = [e for e in bus.replay(0)[0] if e.kind == "permission_decision"]
        assert decisions and decisions[-1].payload.get("decision") == "deny"
        assert decisions[-1].session_id == "ses_headless"


# ============================================================================
# Gate 4 - model lease integrity
# ============================================================================

class TestGate04ModelLeaseIntegrity:
    """Local sessions queue, never evict each other; exclusive work drains."""

    def test_active_lease_blocks_eviction_and_eviction_skips_leased_models(
            self, monkeypatch):
        from pedacito import servers

        leases.reset_default_lease_manager()
        mgr = leases.default_lease_manager()
        held = mgr.acquire("ses_a", LOCAL, "model-a")
        assert mgr.can_evict(LOCAL, "model-a") is False
        held.release()
        assert mgr.can_evict(LOCAL, "model-a") is True

        other = "http://127.0.0.1:9100/v1"
        leased = mgr.acquire("ses_other", other, "leased-model")
        assert mgr.can_evict(other, "leased-model") is False
        states = [
            {"name": "target", "base_url": LOCAL, "online": True,
             "loaded_models": ["target-model"], "pinned_models": [],
             "active": True},
            {"name": "other", "base_url": other, "online": True,
             "loaded_models": ["leased-model", "free-model"],
             "pinned_models": [], "active": False},
        ]
        monkeypatch.setattr(servers, "list_server_states",
                            lambda config, **kw: states)
        unloaded: list[str] = []
        monkeypatch.setattr(
            servers, "unload_server_model",
            lambda server, model="", config=None, **kw:
            unloaded.append(model) or {"ok": True, "model": model})
        result = servers.evict_other_local_servers(LOCAL, None, poll=False)
        assert "leased-model" not in unloaded
        assert "free-model" in unloaded
        assert any("leased" in s for s in result["skipped"])
        leased.release()
        leases.reset_default_lease_manager()

    def test_different_model_session_queues_instead_of_evicting(self):
        mgr = leases.ModelLeaseManager()
        first = mgr.acquire("ses_a", LOCAL, "model-a")
        got: dict[str, object] = {}

        def second() -> None:
            got["lease"] = mgr.acquire("ses_b", LOCAL, "model-b", timeout_s=15)

        th = spawn_thread_with_context(second)
        time.sleep(0.3)
        assert th.is_alive() and "lease" not in got, "session B must queue"
        assert mgr.snapshot()["active_leases"], "A's lease must stay live"
        first.release()
        th.join(15)
        assert "lease" in got, "session B must acquire after A releases"
        got["lease"].release()

    def test_same_model_sessions_share_the_warm_model_without_reload(self):
        mgr = leases.ModelLeaseManager()
        first = mgr.acquire("ses_a", LOCAL, "model-a")
        started = time.monotonic()
        second = mgr.acquire("ses_b", LOCAL, "model-a", timeout_s=5)
        assert time.monotonic() - started < 2.0, "same model must not queue"
        assert len(mgr.snapshot()["active_leases"]) == 2
        first.release()
        assert mgr.can_evict(LOCAL, "model-a") is False, \
            "the warm model stays loaded while another session holds it"
        second.release()
        assert mgr.can_evict(LOCAL, "model-a") is True

    def test_exclusive_calibrate_waits_for_active_session_without_preempting(self):
        mgr = leases.ModelLeaseManager()
        session_lease = mgr.acquire("ses_chat", LOCAL, "model-a")
        got: dict[str, object] = {}

        def calibrate() -> None:
            got["lease"] = mgr.acquire(
                "calibrate", LOCAL, "model-a", exclusive=True,
                priority=leases.LeasePriority.EXCLUSIVE, timeout_s=15)

        th = spawn_thread_with_context(calibrate)
        time.sleep(0.3)
        assert th.is_alive() and "lease" not in got, \
            "exclusive lease must wait, not preempt"
        assert mgr.snapshot()["active_leases"], "session lease must survive"
        session_lease.release()
        th.join(15)
        exclusive = got["lease"]
        assert exclusive.exclusive is True
        exclusive.release()
        assert mgr.snapshot()["active_leases"] == []


# ============================================================================
# Gate 5 - project state safety
# ============================================================================

class TestGate05ProjectStateSafety:
    """Concurrent writers are refused; updates never lose; maintenance waits."""

    def test_second_writer_on_the_same_root_is_refused_with_project_busy(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        manager = workspace.ProjectLockManager()
        owner = manager.acquire(root, session_id="ses_w1")
        try:
            with pytest.raises(workspace.ProjectBusy):
                manager.acquire(root, session_id="ses_w2")
        finally:
            manager.release(root, owner)

    def test_atomic_update_json_loses_no_updates_under_contention(self, tmp_path):
        root = tmp_path / "proj"
        (root / ".pedacito").mkdir(parents=True)
        target = root / ".pedacito" / "counter.json"

        def bump(data: dict) -> dict:
            data["n"] = int(data.get("n", 0)) + 1
            return data

        def writer() -> None:
            for _ in range(20):
                workspace.atomic_update_json(target, bump, root=root)

        threads = [spawn_thread_with_context(writer) for _ in range(6)]
        for th in threads:
            th.join(60)
        assert json.loads(target.read_text(encoding="utf-8"))["n"] == 120

    def test_maintenance_refuses_while_a_session_or_lock_is_active(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        active = ["ses_live"]

        def hook(_root, _action):
            return list(active)

        workspace.register_quiesce_hook(hook)
        try:
            ok, reason = workspace.check_maintenance_safe(root, action="reset")
            assert not ok and "active session" in reason
            ok, reason = workspace.check_maintenance_safe(root, action="prune")
            assert not ok
            active.clear()
            ok, _reason = workspace.check_maintenance_safe(root, action="reset")
            assert ok
            ok, _reason = workspace.check_maintenance_safe(
                root, action="reset", force=True)
            assert ok
        finally:
            workspace.unregister_quiesce_hook(hook)

    def test_run_lock_blocks_maintenance_until_released(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        workspace.acquire_lock(root, session_id="ses_run")
        try:
            ok, reason = workspace.check_maintenance_safe(root, action="reset")
            assert not ok and reason
        finally:
            workspace.release_lock(root)
        ok, _reason = workspace.check_maintenance_safe(root, action="reset")
        assert ok


# ============================================================================
# Gate 6 - worktree safety
# ============================================================================

class TestGate06WorktreeSafety:
    """Dirty trees are refused, carried or stashed; failures leave nothing."""

    def test_dirty_tree_is_refused_by_default(self, tmp_path):
        root = _repo(tmp_path / "repo")
        (root / "app.py").write_text("x = 2\n", encoding="utf-8")
        with pytest.raises(worktree.WorktreeError):
            worktree.create_session_worktree(root, "dirty", session_id="ses_wt1")
        assert len(worktree.list_paths(root)) == 1

    def test_carry_over_moves_changes_and_stash_records_ref(self, tmp_path):
        root = _repo(tmp_path / "repo")
        (root / "app.py").write_text("x = 3\n", encoding="utf-8")
        (root / "new.txt").write_text("fresh\n", encoding="utf-8")
        carried = worktree.create_session_worktree(
            root, "carry", session_id="ses_wt2", dirty_policy="carry_over")
        try:
            assert (carried.path / "app.py").read_text(encoding="utf-8") == "x = 3\n"
            assert (carried.path / "new.txt").is_file()
            assert (root / "app.py").read_text(encoding="utf-8") == "x = 3\n", \
                "carry_over must not mutate the root tree"
        finally:
            worktree.remove_session_worktree(carried, force=True, delete_branch=True)

        stashed = worktree.create_session_worktree(
            root, "stash", session_id="ses_wt3", dirty_policy="stash")
        try:
            assert stashed.stash_ref, "stash must record its stash ref"
            assert not worktree.is_dirty(root), "stash must clean the root tree"
        finally:
            worktree.remove_session_worktree(stashed, force=True, delete_branch=True)

    def test_failed_creation_and_cancelled_session_leave_zero_stale_worktrees(
            self, tmp_path):
        root = _repo(tmp_path / "repo")
        with pytest.raises(worktree.WorktreeError):
            worktree.create_session_worktree(
                root, "bad", session_id="ses_wt4", base="no-such-ref-xyz")
        assert len(worktree.list_paths(root)) == 1
        assert _git(root, "branch", "--list", "pedacito/*") == ""

        ctx = SessionContext(session_id="ses_wt5", root=root)
        handle = worktree.create_session_worktree(
            root, "ephem", session_id=ctx.session_id, ephemeral=True)
        ctx.resource_registry.register(
            "worktree", "ephemeral", lambda: worktree.remove_session_worktree(
                handle, force=True, delete_branch=True))
        assert handle.path.exists()
        ctx.cancel("gate6")
        assert not handle.path.exists()
        assert len(worktree.list_paths(root)) == 1
        assert _git(root, "branch", "--list", "pedacito/*") == ""


# ============================================================================
# Gate 7 - crash recovery (real child process killed mid-task)
# ============================================================================

_CRASH_CHILD = """
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from pedacito import workspace
from pedacito.journal import TaskJournal
root = Path(sys.argv[2])
workspace.acquire_lock(root, session_id="ses_crash")
j = TaskJournal(root, "task-crash", "ses_crash")
j.append("task_start", goal="migrate")
j.append("edit", file="app.py")
print("READY", flush=True)
time.sleep(120)
"""


class TestGate07CrashRecovery:
    """A SIGKILLed writer leaves a parseable WAL and a recoverable lock."""

    def test_killed_child_leaves_parseable_journal_and_recoverable_lock(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        child = subprocess.Popen(
            [sys.executable, "-c", _CRASH_CHILD, str(REPO), str(root)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace",
            cwd=str(root), env=_child_env(), shell=False)
        try:
            line = child.stdout.readline().strip()
            assert line == "READY", child.stderr.read()
            child.kill()
        finally:
            child.wait(timeout=30)
            child.stdout.close()
            child.stderr.close()
        assert child.returncode != 0

        journal_files = list((root / ".pedacito").rglob("*.jsonl"))
        assert journal_files, "the WAL must be on disk"
        records = read_journal(journal_files[0])
        kinds = [r.kind for r in records]
        assert kinds[:2] == ["task_start", "edit"]

        incomplete = find_incomplete_tasks(root)
        assert [t["task_id"] for t in incomplete] == ["task-crash"]
        outcome = rollback_incomplete_task(root, incomplete[0]["wal_id"])
        assert outcome
        # No checkpoint was recorded before the crash, so rollback refuses
        # (nothing to restore) and does NOT mark the task terminal.
        assert outcome["ok"] is False
        assert [t["task_id"] for t in find_incomplete_tasks(root)] == ["task-crash"]

        lock = workspace.acquire_lock(root, session_id="ses_recovered")
        assert lock.exists()
        workspace.release_lock(root)

    def test_truncated_trailing_record_is_tolerated(self, tmp_path):
        root = tmp_path / "proj"
        j = TaskJournal(root, "task-torn", "ses_torn")
        j.append("task_start")
        path = j.path
        with path.open("a", encoding="utf-8") as fh:
            fh.write('{"seq": 9, "kind": "edit", "data"')
        assert [r.kind for r in read_journal(path)] == ["task_start"]


# ============================================================================
# Gate 8 - sensitive artifacts never leak (both engines, every store)
# ============================================================================

SENSITIVE = {
    ".env": "API_KEY=super-secret-value\n",
    "keys/id_rsa": "-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n",
    "certs/server.pem": "-----BEGIN CERTIFICATE-----\nabc\n",
    "credentials.json": '{"token": "abc"}\n',
}
ALLOWED = {
    ".env.example": "API_KEY=your-key-here\n",
    "keys/id_rsa.pub": "ssh-ed25519 AAAA public\n",
    "app.py": "x = 1\n",
}


def _sensitive_tree(root: Path) -> None:
    for rel, text in {**SENSITIVE, **ALLOWED}.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


class TestGate08SensitiveArtifactZeroLeak:
    """Secrets never reach checkpoints, ratchet, backups, mirrors or review."""

    def test_stores_exclude_secrets_and_keep_templates(self, tmp_path):
        root = tmp_path / "proj"
        _sensitive_tree(root)
        listing = {rel for rel, _ in checkpoints.iter_workspace_files(root)}
        assert not (set(SENSITIVE) & listing)
        assert {".env.example", "keys/id_rsa.pub", "app.py"} <= listing

        cp = checkpoints.create_checkpoint(root, "gate8", mode="manual")
        snap = checkpoints.snapshot_dir(root, cp["id"])
        for rel in SENSITIVE:
            assert not (snap / rel).exists(), f"checkpoint leaked {rel}"
        assert (snap / ".env.example").is_file()

        candidates = {p.relative_to(root).as_posix()
                      for p, _t in ratchet._iter_candidate_files(root, 500)}
        assert not (set(SENSITIVE) & candidates)
        info = ratchet.begin(root, 1, "gate8")
        assert not (set(SENSITIVE) & set(info["files"]))
        step = ratchet.step_dir(root, 1)
        for rel in SENSITIVE:
            assert not (step / rel).exists()

        backup = tmp_path / "backup"
        for rel in SENSITIVE:
            assert fileio.backup_copy(root / rel, backup / rel) is False
            assert not (backup / rel).exists()
        assert fileio.backup_copy(root / ".env.example",
                                  backup / ".env.example") is True

        ws = workspace_overlay.VirtualWorkspace(root, apply=False)
        try:
            with ws.exec_root() as mirror:
                for rel in SENSITIVE:
                    assert not (mirror / rel).exists(), f"mirror leaked {rel}"
                assert (mirror / "app.py").is_file()
        finally:
            ws.close()

    def test_writes_are_blocked_identically_in_two_phase_and_interleaved(self, tmp_path):
        root = tmp_path / "proj"
        _sensitive_tree(root)
        original = {rel: (root / rel).read_text(encoding="utf-8") for rel in SENSITIVE}

        ws = workspace_overlay.VirtualWorkspace(root, apply=False)
        try:
            for rel in SENSITIVE:
                res = ws.write(rel, "changed\n")
                assert not res.ok and "sensitive" in res.message.lower(), rel
            assert ws.write(".env.example", "API_KEY=changed\n").ok
        finally:
            ws.close()

        results = edits.apply_edits(
            root, [edits.Edit(".env", "API_KEY=super-secret-value", "API_KEY=x")],
            backup=False, dry_run=True)
        assert results and not results[0].ok
        assert "sensitive" in results[0].message.lower()

        cfg = Config()
        cfg.allow_write = True
        cfg.backup = False
        index = Index(root, [])
        msg = tools.tool_write(index, cfg, "credentials.json", "leaked\n")
        assert "sensitive" in msg.lower()
        for rel, text in original.items():
            assert (root / rel).read_text(encoding="utf-8") == text

    def test_review_pages_and_rows_redact_embedded_tokens(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        token = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2"
        before = "API = 'old-value-123456'\nx = 1\n"
        after = f"API = '{token}'\nx = 1\n"
        rows, added, removed, hunks = review.build_diff(before, after)
        fd = review.FileDiff("app.py", rows, added, removed, review._sha(before),
                             after, "edit", hunks=hunks, before_text=before)
        page = review.build_page([fd], f"task {token}", "model", "profile",
                                 "nonce", False, "stamp")
        assert token not in page and "[REDACTED]" in page
        env = review.FileDiff(".env", [], 1, 0, "", "SECRET=leaked\n", "edit")
        ok, detail = review.write_files(root, [env], None)
        assert not ok and "sensitive" in detail.lower()
        assert not (root / ".env").exists()


# ============================================================================
# Gate 9 - cancellation cascade reaps one session, spares its sibling
# ============================================================================

_SLEEPER = (
    "import os, sys, time\n"
    "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
    "time.sleep(float(sys.argv[2]))\n"
)


class TestGate09CancellationCascade:
    """Cancelling A reaps its child and resources; B keeps running untouched."""

    def test_cancel_reaps_session_a_and_leaves_session_b_running(self, tmp_path):
        ctx_a = SessionContext(session_id="ses_cancel_a", root=tmp_path)
        ctx_b = SessionContext(session_id="ses_cancel_b", root=tmp_path)
        pid_a_file, pid_b_file = tmp_path / "pid_a.txt", tmp_path / "pid_b.txt"
        reaped: list[str] = []
        for kind in ("dev_server", "lsp", "mcp", "temp_mirror", "worktree",
                     "review_server"):
            ctx_a.resource_registry.register(
                kind, f"a-{kind}", lambda k=kind: reaped.append(k))
            ctx_b.resource_registry.register(
                kind, f"b-{kind}", lambda: None)

        out: dict[str, tuple] = {}

        def run_child(name: str, ctx, pid_file: Path, seconds: float) -> None:
            out[name] = proc.run(
                [sys.executable, "-c", _SLEEPER, str(pid_file), str(seconds)],
                cwd=tmp_path, timeout=120, cancel_event=ctx.cancel_event)

        th_a = spawn_thread_with_context(run_child, args=("a", ctx_a, pid_a_file, 60))
        th_b = spawn_thread_with_context(run_child, args=("b", ctx_b, pid_b_file, 2))
        assert _wait_for(lambda: pid_a_file.exists() and pid_b_file.exists())
        pid_a = int(pid_a_file.read_text(encoding="utf-8"))

        report = ctx_a.cancel("gate9")
        th_a.join(60)
        assert not th_a.is_alive(), "cancelled child must stop"
        assert ctx_a.cancel_event.is_set() and not ctx_b.cancel_event.is_set()
        assert sorted(reaped) == sorted(
            ["dev_server", "lsp", "mcp", "temp_mirror", "worktree", "review_server"])
        assert len(ctx_b.resource_registry.list_resources()) == 6, \
            "sibling resources must be untouched"
        assert report["resources_closed"] is not None

        th_b.join(60)
        assert not th_b.is_alive()
        rc_b, _out_b, stopped_b, _sec_b = out["b"]
        assert rc_b == 0 and not stopped_b, "sibling child must finish on its own"
        assert not ctx_b.cancel_event.is_set()

        assert _wait_for(lambda: not workspace._pid_alive(pid_a), timeout=30), \
            "cancelled child PID must be reaped"

    def test_cancelled_temp_mirror_is_removed(self, tmp_path):
        ctx = SessionContext(session_id="ses_mirror", root=tmp_path)
        mirror = Path(tempfile.mkdtemp(prefix="pdc-gate9-"))
        (mirror / "f.txt").write_text("x", encoding="utf-8")
        ctx.resource_registry.register(
            "temp_mirror", "mirror", lambda: shutil.rmtree(mirror, ignore_errors=True))
        ctx.cancel("gate9b")
        assert not mirror.exists()


# ============================================================================
# Gate 10 - policy matrix proof
# ============================================================================

EXPECTED = {
    "plan": {"run": "deny", "hook": "deny", "mcp": "deny", "write": "deny",
             "subagent": "deny"},
    "ask": {"run": "ask", "hook": "ask", "mcp": "ask", "write": "ask",
            "subagent": "ask"},
    "accept-edits": {"run": "ask", "hook": "ask", "mcp": "ask",
                     "write": "allow", "subagent": "ask"},
    "workspace-write": {"run": "allow", "hook": "allow", "mcp": "ask",
                        "write": "allow", "subagent": "allow"},
    "danger": {"run": "allow", "hook": "allow", "mcp": "allow",
               "write": "allow", "subagent": "allow"},
}
TARGETS = {"run": "pytest -q", "hook": "pre_write", "mcp": "docs.search",
           "write": "app.py", "subagent": "worker"}
STRICT = {"deny": 3, "ask": 2, "allow": 1}


class TestGate10PolicyMatrixProof:
    """Every mode x capability cell is pinned; children only narrow."""

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("capability", CAPABILITIES)
    def test_matrix_cell(self, mode, capability):
        engine = policy.PolicyEngine(_cfg(mode, "ask"))
        decision = engine.evaluate(policy.PolicyRequest(
            capability, TARGETS[capability]))
        assert decision.decision == EXPECTED[mode][capability], (mode, capability)

    def test_no_mode_exceeds_its_declared_capabilities(self):
        for mode in ("plan", "ask", "accept-edits"):
            for cap in ("run", "hook", "mcp", "subagent", "write"):
                d = policy.PolicyEngine(_cfg(mode)).evaluate(
                    policy.PolicyRequest(cap, TARGETS[cap]))
                assert d.decision != "allow" or (
                    mode == "accept-edits" and cap == "write"), (mode, cap)
        for mode in ("workspace-write", "danger"):
            for cap in ("run", "hook", "subagent"):
                d = policy.PolicyEngine(_cfg(mode)).evaluate(
                    policy.PolicyRequest(cap, TARGETS[cap]))
                if d.decision == "allow" and cap == "run" and mode == "workspace-write":
                    assert d.requires_sandbox is True

    def test_untrusted_policy_file_cannot_widen_capabilities(self, tmp_path, monkeypatch):
        root = tmp_path / "proj"
        (root / ".pedacito").mkdir(parents=True)
        (root / ".pedacito" / "policy.toml").write_text(
            '[policy]\npermission_mode = "danger"\napproval_policy = "never"\n',
            encoding="utf-8")
        monkeypatch.setattr(config_mod, "is_trusted", lambda path: False)
        assert policy.load_policy_file(root, _cfg("ask")) == {}
        engine = policy.PolicyEngine(_cfg("ask"), root=root)
        assert engine.mode == "ask"
        assert engine.approval_policy == "ask"
        d = engine.evaluate(policy.PolicyRequest("run", "pytest -q"))
        assert d.decision == "ask"

    @pytest.mark.parametrize("parent_mode", MODES)
    @pytest.mark.parametrize("role", ["worker", "explore", "verifier"])
    def test_child_policy_strictly_narrows(self, parent_mode, role):
        parent = policy.PolicyEngine(_cfg(parent_mode, "ask"))
        child = parent.child_policy(role=role)
        for cap in CAPABILITIES:
            req = policy.PolicyRequest(cap, TARGETS[cap],
                                       metadata={"role": role} if cap == "subagent" else {})
            p = parent.evaluate(req).decision
            c = child.evaluate(req).decision
            assert STRICT[c] >= STRICT[p], (parent_mode, role, cap, p, c)
        if role in ("explore", "verifier"):
            assert child.mode == "plan"
            assert child.evaluate(policy.PolicyRequest("write", "app.py")).decision == "deny"

    def test_policy_decisions_are_emitted_to_the_session_bus(self):
        bus = events.SessionEventBus("ses_policy_bus")
        engine = policy.PolicyEngine(_cfg("plan"), event_bus=bus)
        engine.evaluate(policy.PolicyRequest("run", "pytest -q", session_id="ses_policy_bus"))
        kinds = [e.kind for e in bus.replay(0)[0]]
        assert "permission_decision" in kinds


# ============================================================================
# Gate 11 - Windows sandbox gate
# ============================================================================

class TestGate11WindowsSandboxGate:
    """Unattended execution is sandboxed or refused, never silently run bare."""

    @pytest.fixture
    def no_backend(self, monkeypatch):
        monkeypatch.setattr(sandbox, "backend_name", lambda: "")

    @pytest.fixture
    def win_backend(self, monkeypatch):
        monkeypatch.setattr(sandbox, "backend_name", lambda: "win_restricted")

    @pytest.mark.parametrize("mode", ["workspace-write", "danger"])
    def test_unattended_modes_refuse_without_a_backend(self, no_backend, mode):
        cfg = _cfg(mode, "ask", sandbox_on=True)
        cfg.sandbox_strict = True
        with pytest.raises(sandbox.SandboxRefused):
            sandbox.enforce_unattended_gate(cfg)

    def test_background_task_refuses_without_a_backend(self, no_backend):
        cfg = _cfg("ask", "ask", sandbox_on=True)
        cfg.sandbox_strict = True
        with pytest.raises(sandbox.SandboxRefused):
            sandbox.enforce_unattended_gate(cfg, background=True)

    def test_never_approval_with_sandbox_on_refuses_without_a_backend(self, no_backend):
        cfg = _cfg("workspace-write", "never", sandbox_on=True)
        with pytest.raises(sandbox.SandboxRefused):
            sandbox.enforce_unattended_gate(cfg)
        sandbox.enforce_unattended_gate(_cfg("ask", "never", sandbox_on=False))

    @pytest.mark.parametrize("mode", ["workspace-write", "danger"])
    def test_unattended_modes_pass_inside_the_win_restricted_sandbox(self, win_backend, mode):
        cfg = _cfg(mode, "ask", sandbox_on=True)
        cfg.sandbox_strict = True
        sandbox.enforce_unattended_gate(cfg)
        sandbox.enforce_unattended_gate(cfg, background=True)

    def test_interactive_ask_is_never_blocked_by_the_gate(self, no_backend):
        sandbox.enforce_unattended_gate(_cfg("ask", "ask", sandbox_on=True))

    @pytest.mark.skipif(os.name != "nt", reason="the Job Object runner is Windows-only")
    def test_sandbox_dispatch_uses_the_job_object_runner(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sandbox, "backend_name", lambda: "win_restricted")
        calls: list[list[str]] = []

        def fake_runner(command, **kwargs):
            calls.append(list(command))
            return 0, "ok", False, 0.0

        monkeypatch.setattr(proc, "_run_windows_sandboxed", fake_runner)
        cfg = _cfg("danger", "ask", sandbox_on=True)
        assert sandbox.windows_sandbox_active(cfg) is True
        argv = [sys.executable, "-c", "print('ok')"]
        rc, out, _stopped, _sec = proc.run(argv, cwd=tmp_path, timeout=30,
                                           windows_sandbox=True)
        assert rc == 0 and "ok" in out
        assert calls and calls[0] == argv

    @pytest.mark.skipif(sys.platform != "win32", reason="live Windows sandbox only")
    def test_live_windows_sandbox_runs_or_refuses_cleanly(self, tmp_path, monkeypatch):
        sandbox.reset_win_probe()
        cap = sandbox_win.probe()
        cfg = _cfg("workspace-write", "ask", sandbox_on=True)
        cfg.sandbox_strict = True
        if cap.available:
            assert sandbox.backend_name() == "win_restricted"
            sandbox.enforce_unattended_gate(cfg)
            rc, out, _err, _sec = sandbox_win.run_sandboxed(
                [sys.executable, "-c", "print('sandboxed')"],
                cwd=tmp_path, env=dict(os.environ), timeout=60)
            assert rc == 0 and "sandboxed" in out
        else:
            monkeypatch.setattr(sandbox, "backend_name", lambda: "")
            with pytest.raises(sandbox.SandboxRefused):
                sandbox.enforce_unattended_gate(cfg)
        sandbox.reset_win_probe()


# ============================================================================
# Gate 12 - versions, bilingual docs, locale parity, how resolution
# ============================================================================

def _placeholders(text: str) -> set[str]:
    return set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", text))


class TestGate12BilingualDocsAndI18nParity:
    """Versions agree, both languages document every public surface, how works."""

    def test_versions_agree(self):
        pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
        assert match is not None
        assert __version__ == match.group(1)
        assert __version__.startswith("0.4.")
        assert rpc.PROTOCOL_VERSION == "0.4.0"
        assert daemon.DAEMON_PROTOCOL_VERSION == "0.4.0"

    def test_locale_keys_and_placeholders_match(self):
        en = json.loads((REPO / "pedacito" / "locales" / "en.json").read_text(encoding="utf-8"))
        es = json.loads((REPO / "pedacito" / "locales" / "es.json").read_text(encoding="utf-8"))
        assert set(en) == set(es), set(en) ^ set(es)
        for key in en:
            assert _placeholders(str(en[key])) == _placeholders(str(es[key])), key

    def test_every_public_config_field_is_documented_in_both_languages(self):
        names = [f.name for f in fields(Config) if not f.name.startswith("_")]
        assert names
        for rel in ("docs/CONFIG.md", "docs/es/CONFIG.md"):
            text = (REPO / rel).read_text(encoding="utf-8")
            missing = [n for n in names if f"`{n}`" not in text]
            assert not missing, f"{rel} is missing: {missing}"

    def test_how_resolves_permission_mode_in_english(self):
        result = how.answer_structured("permission mode", use_model=False)
        assert result["ok"] and result["mode"] == "deterministic"
        keys = {item.get("key") for item in result["matches"]["settings"]}
        assert "permission_mode" in keys
        assert "I could not match" not in result["answer"]
        assert "permission_mode" in result["answer"] or "approval" in result["answer"].lower()

    def test_how_resolves_politica_in_spanish(self):
        result = how.answer_structured("política", use_model=False)
        assert result["ok"]
        keys = {item.get("key") for item in result["matches"]["settings"]}
        assert {"permission_mode", "approval_policy"} & keys
        assert "I could not match" not in result["answer"]

    def test_cli_rpc_describe_and_version_report_040(self):
        for argv in (["--version"], ["rpc", "--describe"]):
            res = subprocess.run([sys.executable, "-m", "pedacito", *argv],
                                 capture_output=True, text=True, timeout=120,
                                 encoding="utf-8", errors="replace",
                                 cwd=str(REPO), env=_child_env(), shell=False)
            assert res.returncode == 0, res.stderr
        described = subprocess.run([sys.executable, "-m", "pedacito", "rpc", "--describe"],
                                   capture_output=True, text=True, timeout=120,
                                   encoding="utf-8", errors="replace",
                                   cwd=str(REPO), env=_child_env(), shell=False)
        assert json.loads(described.stdout)["protocol_version"] == "0.4.0"
