"""Daemon session registry, bearer auth, SSE replay and ACP/RPC scoping."""
from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import acp, cli, daemon, rpc  # noqa: E402
from pedacito import review as review_mod  # noqa: E402
from pedacito.config import Config  # noqa: E402


def _request(info, method, path, body=None, *, token=None, timeout=4.0):
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token is not None:
        headers["X-Pedacito-Token"] = token
    req = urllib.request.Request(
        str(info["url"]).rstrip("/") + path, data=data, headers=headers,
        method=method)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as response:
            raw = response.read()
            status = int(getattr(response, "status", 200) or 200)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = int(exc.code)
    payload = json.loads(raw.decode("utf-8")) if raw else {}
    return status, payload


@pytest.fixture
def server(tmp_path):
    state = tmp_path / "state"
    srv = daemon.DaemonServer(state_dir=state)
    info = srv.start()
    yield srv, info, state
    srv.stop()


def test_rejects_non_loopback_hosts():
    for host in ("0.0.0.0", "192.168.1.5", "example.com"):
        with pytest.raises(ValueError):
            daemon.DaemonServer(host=host)


def test_requires_the_per_run_bearer_token(server):
    srv, info, _state = server
    status, payload = _request(info, "GET", "/v1/health")
    assert status == 401 and payload["error"] == "unauthorized"
    status, payload = _request(info, "GET", "/v1/health", token="wrong")
    assert status == 401
    status, payload = _request(info, "GET", "/v1/health",
                               token=info["token"])
    assert status == 200
    assert payload["protocol_version"] == "0.4.0"
    assert payload["ok"] is True


def test_discovery_file_is_created_and_removed(tmp_path):
    state = tmp_path / "state"
    srv = daemon.DaemonServer(state_dir=state)
    assert daemon.read_daemon_info(state) is None
    info = srv.start()
    path = state / "daemon.json"
    assert path.is_file()
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["protocol_version"] == "0.4.0"
    assert stored["pid"] == info["pid"]
    assert stored["token"] == info["token"]
    assert daemon.read_daemon_info(state)["url"] == info["url"]
    srv.stop()
    assert not path.exists()
    assert daemon.read_daemon_info(state) is None


def _git_repo(root: Path) -> bool:
    if not shutil.which("git"):
        return False
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
    commands = [
        ["git", "init"],
        ["git", "config", "user.email", "t@example.com"],
        ["git", "config", "user.name", "t"],
        ["git", "add", "."],
        ["git", "commit", "-m", "init"],
    ]
    for i, cmd in enumerate(commands):
        if i == 3:
            (root / "README.md").write_text("init\n", encoding="utf-8")
        result = subprocess.run(cmd, cwd=str(root), shell=False,
                                capture_output=True, text=True,
                                encoding="utf-8", env=env)
        if result.returncode != 0:
            return False
    return True


def test_sessions_are_isolated_and_replay_events(server, tmp_path):
    srv, info, _state = server
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    token = info["token"]
    code, a = _request(info, "POST", "/v1/sessions",
                       {"root": str(root_a), "client_id": "editor-a"},
                       token=token)
    assert code == 201 and a["state"] == "idle"
    code, b = _request(info, "POST", "/v1/sessions",
                       {"root": str(root_b), "client_id": "editor-b"},
                       token=token)
    assert code == 201
    assert a["session_id"] != b["session_id"]
    code, listed = _request(info, "GET", "/v1/sessions", token=token)
    assert code == 200
    ids = {s["session_id"] for s in listed["sessions"]}
    assert {a["session_id"], b["session_id"]} <= ids
    # The daemon's own registry keeps the two contexts apart.
    sa = srv.get_session(a["session_id"])
    sb = srv.get_session(b["session_id"])
    assert sa.ctx is not sb.ctx
    assert sa.ctx.root != sb.ctx.root
    assert sa.ctx.event_bus is not sb.ctx.event_bus
    # SSE-style replay is scoped to the addressed session.
    code, events = _request(
        info, "GET",
        f"/v1/sessions/{a['session_id']}/events?after=0&once=1", token=token)
    assert code == 200
    assert events["session_id"] == a["session_id"]
    assert all(e["session_id"] == a["session_id"]
               for e in events["events"])
    assert any(e["kind"] == "session_created" for e in events["events"])
    # Attaching the same id returns the same session, not a new one.
    code, attached = _request(info, "POST", "/v1/sessions",
                              {"session_id": a["session_id"],
                               "client_id": "editor-2"}, token=token)
    assert code == 200 and attached["session_id"] == a["session_id"]
    assert "editor-2" in attached["attached_clients"]


def test_worktree_session_and_resource_teardown(server, tmp_path):
    srv, info, _state = server
    root = tmp_path / "repo"
    root.mkdir()
    if not _git_repo(root):
        pytest.skip("git is not available")
    code, created = _request(
        info, "POST", "/v1/sessions",
        {"root": str(root), "worktree": "parallel"}, token=info["token"])
    assert code == 201, created
    assert created["worktree_branch"].startswith("pedacito/parallel-")
    ws = srv.get_session(created["session_id"])
    assert Path(created["root"]) != root
    assert Path(created["root"]).is_dir()
    torn: list[str] = []
    ws.ctx.resource_registry.register(
        "custom", "sentinel", lambda: torn.append("closed"))
    code, _ = _request(info, "DELETE",
                       f"/v1/sessions/{created['session_id']}",
                       token=info["token"])
    assert code == 200
    assert torn == ["closed"]
    assert ws.state == "closed"
    code, listed = _request(info, "GET", "/v1/sessions", token=info["token"])
    assert created["session_id"] not in {
        s["session_id"] for s in listed["sessions"]}


def test_permission_and_review_decisions_route_to_the_session(server, tmp_path):
    srv, info, _state = server
    root = tmp_path / "p"
    root.mkdir()
    code, created = _request(info, "POST", "/v1/sessions",
                             {"root": str(root)}, token=info["token"])
    ws = srv.get_session(created["session_id"])
    broker = ws.ctx.permission_broker
    result: dict = {}

    def ask():
        result["answer"] = broker.request({
            "category": "shell", "detail": "pytest -q", "key": "pytest"})

    worker = threading.Thread(target=ask)
    worker.start()
    deadline = time.time() + 3
    while time.time() < deadline and not broker.pending():
        time.sleep(0.01)
    pending = broker.pending()
    assert pending
    code, body = _request(
        info, "POST",
        f"/v1/sessions/{created['session_id']}/permission",
        {"request_id": pending[0]["request_id"], "decision": "allow_once"},
        token=info["token"])
    worker.join(timeout=3)
    assert code == 200 and body["ok"] is True
    assert result["answer"][0] is True

    written: dict = {}

    def writer(files=None, accepted_hunks=None):
        target = root / "a.txt"
        target.write_text("new\n", encoding="utf-8")
        written["path"] = target
        return True, "applied"

    staged = ws.ctx.review_queue.stage(
        task="write a", writer=writer,
        diffs=[review_mod.FileDiff(
            rel="a.txt", rows=[], added=1, removed=0, sha="",
            after_text="new\n", message="")])
    code, body = _request(
        info, "POST", f"/v1/sessions/{created['session_id']}/review",
        {"review_id": staged.review_id, "action": "apply"},
        token=info["token"])
    assert code == 200 and body["ok"] is True
    assert written["path"].read_text(encoding="utf-8") == "new\n"
    # An unknown review id is refused, not silently accepted.
    code, body = _request(
        info, "POST", f"/v1/sessions/{created['session_id']}/review",
        {"review_id": "rev_nope_1", "action": "apply"}, token=info["token"])
    assert code == 409


def test_quiesce_refuses_while_a_prompt_runs(tmp_path):
    state = tmp_path / "state"
    started = threading.Event()
    release = threading.Event()

    def runner(ctx, task, apply_):
        started.set()
        release.wait(5)
        return {"reply": "done"}

    srv = daemon.DaemonServer(state_dir=state, runner=runner)
    info = srv.start()
    try:
        root = tmp_path / "q"
        root.mkdir()
        code, created = _request(info, "POST", "/v1/sessions",
                                 {"root": str(root)}, token=info["token"])
        sid = created["session_id"]
        code, body = _request(
            info, "POST", f"/v1/sessions/{sid}/prompt",
            {"task": "work"}, token=info["token"])
        assert code == 202 and body["state"] == "running"
        assert started.wait(3)
        code, body = _request(info, "POST", "/v1/quiesce",
                              {"root": str(root), "timeout_s": 0.1},
                              token=info["token"])
        assert body["ok"] is False
        assert "active session" in body["reason"]
        release.set()
        deadline = time.time() + 3
        while time.time() < deadline:
            if srv.get_session(sid).state == "idle":
                break
            time.sleep(0.02)
        assert srv.get_session(sid).state == "idle"
        code, body = _request(info, "POST", "/v1/quiesce",
                              {"root": str(root), "timeout_s": 0.5},
                              token=info["token"])
        assert body["ok"] is True
    finally:
        release.set()
        srv.stop()


def test_shutdown_endpoint_stops_and_cleans_discovery(tmp_path):
    state = tmp_path / "shutdown"
    srv = daemon.DaemonServer(state_dir=state)
    info = srv.start()
    code, body = _request(info, "POST", "/v1/shutdown", {},
                          token=info["token"])
    assert code == 202 and body["stopping"] is True
    deadline = time.time() + 5
    while time.time() < deadline and daemon.read_daemon_info(state) is not None:
        time.sleep(0.02)
    assert daemon.read_daemon_info(state) is None
    srv.stop()


def test_cli_daemon_status_is_clean_offline(tmp_path, capsys):
    state = tmp_path / "offline"
    code = cli.main(["daemon", "status", "--json",
                     "--state-dir", str(state)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["running"] is False
    assert payload["protocol_version"] == "0.4.0"
    code = cli.main(["daemon", "status", "--state-dir", str(state)])
    assert code == 0
    assert "daemon not running" in capsys.readouterr().out


def test_rpc_describe_advertises_protocol_version_and_capabilities(capsys):
    code = cli.main(["rpc", "--describe"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["protocol_version"] == "0.4.0"
    assert payload["capabilities"]["sessions"] is True
    assert rpc.describe()["protocol_version"] == "0.4.0"


# --------------------------------------------------------------------------- #
# ACP/RPC session scoping and protocol handshakes
# --------------------------------------------------------------------------- #
def _fake_rpc_server(out):
    return rpc.RpcServer(
        run=lambda *a, **k: None,
        explain=lambda q: q,
        status=lambda: {"model": "fake"},
        switch=lambda ref, kind: ref,
        output=out)


def test_acp_initialize_advertises_pedacito_meta():
    out = rpc.RpcOut(io.StringIO(), lambda: None)
    bridge = acp.AcpServer(_fake_rpc_server(out),
                           input_stream=io.StringIO(),
                           output_stream=io.StringIO())
    result = bridge._initialize({})
    assert result["protocolVersion"] == 1
    assert result["_meta"]["pedacitoProtocolVersion"] == "0.4.0"
    assert result["_meta"]["capabilities"]["event_seq_replay"] is True
    assert (result["agentCapabilities"]["experimental"]["pedacito"]
            ["protocolVersion"] == "0.4.0")


def test_acp_sessions_clone_cfg_without_cross_session_mutation(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    cfg = Config()
    cfg.profiles = {
        "fast": {"model": "claude-3-5-sonnet"},
        "slow": {"model": "gemini-2.5-pro"},
    }
    cfg.active_profile = "base"
    out = rpc.RpcOut(io.StringIO(), lambda: None)
    bridge = acp.AcpServer(_fake_rpc_server(out), project_root=root,
                           input_stream=io.StringIO(),
                           output_stream=io.StringIO(), base_cfg=cfg)
    one = bridge._session_new({"cwd": str(root), "profile": "fast",
                               "executionMode": "two_phase"})
    two = bridge._session_new({"cwd": str(root), "profile": "slow",
                               "executionMode": "interleaved"})
    ctx_one = bridge._sessions[one["sessionId"]]["ctx"]
    ctx_two = bridge._sessions[two["sessionId"]]["ctx"]
    assert ctx_one is not ctx_two
    assert ctx_one.cfg is not ctx_two.cfg
    assert ctx_one.cfg.active_profile == "fast"
    assert ctx_two.cfg.active_profile == "slow"
    assert ctx_one.cfg.execution_mode == "two_phase"
    assert ctx_two.cfg.execution_mode == "interleaved"
    # The process-global config stays untouched.
    assert cfg.active_profile == "base"
    assert ctx_one.event_bus.session_id == one["sessionId"]
    assert ctx_two.event_bus.session_id == two["sessionId"]


def test_acp_session_new_refuses_cwd_outside_the_bound_root(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    out = rpc.RpcOut(io.StringIO(), lambda: None)
    bridge = acp.AcpServer(_fake_rpc_server(out), project_root=root,
                           input_stream=io.StringIO(),
                           output_stream=io.StringIO())
    with pytest.raises(acp.AcpError) as err:
        bridge._session_new({"cwd": str(outside)})
    assert err.value.code == -32002
    # The bound root itself remains valid.
    result = bridge._session_new({"cwd": str(root)})
    assert result["sessionId"]


def test_rpc_hello_and_status_include_protocol_version():
    out = rpc.RpcOut(io.StringIO(), lambda: None)
    srv = _fake_rpc_server(out)
    hello = srv.dispatch({"id": 1, "method": "hello"})
    assert hello["result"]["protocol_version"] == "0.4.0"
    assert hello["result"]["capabilities"]["permission_session_keyed"] is True
    status = srv.dispatch({"id": 2, "method": "status"})
    assert status["result"]["protocol_version"] == "0.4.0"
    assert "capabilities" in status["result"]
