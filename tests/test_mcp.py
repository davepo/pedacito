"""Opt-in MCP client and tool dispatch."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import mcp, tools
from pedacito.config import Config
from pedacito.index import Index

FAKE = Path(__file__).resolve().parent / "fake_mcp.py"


def _cfg():
    cfg = Config()
    cfg.allow_mcp = True
    cfg.mcp_servers = (f'"{sys.executable}" "{FAKE}"',)
    cfg.mcp_timeout = 10
    return cfg


def test_mcp_catalog_and_call(tmp_path):
    cfg = _cfg()
    mcp._CATALOG_CACHE.clear()
    entries = mcp.catalog(cfg)
    assert any(name.endswith("/echo") for name, _d, _s in entries)
    out = mcp.call(cfg, "fake_mcp/echo", {"x": "hi"})
    assert out.startswith("echo:") and '"x": "hi"' in out


def test_mcp_tool_is_gated_and_dispatches(tmp_path):
    idx = Index(tmp_path, [])
    assert tools.tool_mcp(idx, Config(), "fake_mcp/echo", "{}").startswith("ERROR")
    cfg = _cfg()
    mcp._CATALOG_CACHE.clear()
    obs = tools.dispatch(idx, cfg, {"action": "mcp", "symbol": "fake_mcp/echo",
                                    "content": '{"x": 1}'})
    assert obs.startswith("echo:") and '"x": 1' in obs


def test_mcp_render_is_off_by_default(tmp_path):
    assert mcp.render(Config()) == ""


def test_mcp_action_only_appears_when_enabled():
    plain = tools.schema_for(Config())
    assert "mcp" not in plain["properties"]["action"]["enum"]
    cfg = _cfg()
    schema = tools.schema_for(cfg)
    assert "mcp" in schema["properties"]["action"]["enum"]


def test_mcp_missing_server_binary_is_reported_not_raised():
    cfg = Config()
    cfg.allow_mcp = True
    cfg.mcp_servers = ("/nonexistent/pedacito-mcp-server",)
    mcp._CATALOG_CACHE.clear()
    try:
        entries = mcp.catalog(cfg)
        assert entries and entries[0][0].endswith("/(unavailable)")
        assert mcp.call(cfg, "pedacito-mcp-server/x", {}).startswith("ERROR")
    finally:
        mcp._CATALOG_CACHE.clear()


# ------------------------------------------------------------- keep-alive
def _reset():
    mcp._CATALOG_CACHE.clear()
    mcp.close_all()


def test_mcp_call_keeps_one_live_session_per_server():
    cfg = _cfg()
    cmd = cfg.mcp_servers[0]
    _reset()
    try:
        out1 = mcp.call(cfg, "fake_mcp/echo", {"x": 1})
        s1 = mcp._SESSIONS.get(cmd)
        assert s1 is not None and s1.proc.poll() is None
        out2 = mcp.call(cfg, "fake_mcp/echo", {"x": 2})
        assert mcp._SESSIONS.get(cmd) is s1        # reused, not restarted
        assert out1.startswith("echo:") and '"x": 1' in out1
        assert out2.startswith("echo:") and '"x": 2' in out2
    finally:
        _reset()


def test_mcp_call_restarts_a_dead_session():
    cfg = _cfg()
    cmd = cfg.mcp_servers[0]
    _reset()
    try:
        assert mcp.call(cfg, "fake_mcp/echo", {"x": 1}).startswith("echo:")
        s1 = mcp._SESSIONS[cmd]
        s1.proc.kill()
        s1.proc.wait(timeout=5)
        out = mcp.call(cfg, "fake_mcp/echo", {"x": 2})
        s2 = mcp._SESSIONS.get(cmd)
        assert out.startswith("echo:") and '"x": 2' in out
        assert s2 is not None and s2 is not s1     # transparent restart
        assert s2.proc.poll() is None
    finally:
        _reset()


def test_mcp_call_retries_once_on_a_fresh_session(monkeypatch):
    """A failed request on a primed session must drop it and retry exactly
    once on a fresh server, so a mid-request desync heals instead of
    failing the gather step."""
    cfg = _cfg()
    cmd = cfg.mcp_servers[0]
    _reset()
    try:
        assert mcp.call(cfg, "fake_mcp/echo", {"x": 1}).startswith("echo:")
        primed = mcp._SESSIONS[cmd]
        calls = []
        real_request = mcp._Session.request

        def flaky(self, method, params=None):
            calls.append((self, method))
            if self is primed and method == "tools/call":
                raise mcp.MCPError("desynced")
            return real_request(self, method, params)

        monkeypatch.setattr(mcp._Session, "request", flaky)
        out = mcp.call(cfg, "fake_mcp/echo", {"x": 2})
        assert out.startswith("echo:") and '"x": 2' in out
        fresh = mcp._SESSIONS.get(cmd)
        assert fresh is not None and fresh is not primed
        assert (primed, "tools/call") in calls
        assert (fresh, "tools/call") in calls
        assert primed.proc.poll() is not None      # the dead session was reaped
    finally:
        _reset()


def test_mcp_call_reports_error_when_both_attempts_fail(monkeypatch):
    cfg = _cfg()
    _reset()
    try:
        real_request = mcp._Session.request

        def broken(self, method, params=None):
            if method == "tools/call":
                raise mcp.MCPError("nope")
            return real_request(self, method, params)

        monkeypatch.setattr(mcp._Session, "request", broken)
        out = mcp.call(cfg, "fake_mcp/echo", {"x": 1})
        assert out.startswith("ERROR: MCP call failed")
        assert "nope" in out
        assert mcp._SESSIONS == {}                 # both sessions dropped
    finally:
        _reset()


def test_mcp_initialize_failure_closes_the_session(monkeypatch):
    cfg = _cfg()
    _reset()
    try:
        # pre-seed the catalog so call() gets past the name checks
        mcp._CATALOG_CACHE[tuple(cfg.mcp_servers)] = [
            ("fake_mcp/echo", "echo the arguments back", "{}")]

        def boom(self):
            raise mcp.MCPError("handshake failed")

        monkeypatch.setattr(mcp._Session, "initialize", boom)
        out = mcp.call(cfg, "fake_mcp/echo", {"x": 1})
        assert out.startswith("ERROR: MCP call failed")
        assert "handshake failed" in out
        assert mcp._SESSIONS == {}                 # no half-open session leaks
    finally:
        _reset()


def test_mcp_close_all_reaps_every_session():
    cfg = _cfg()
    cmd = cfg.mcp_servers[0]
    _reset()
    try:
        mcp.call(cfg, "fake_mcp/echo", {"x": 1})
        s = mcp._SESSIONS[cmd]
        mcp.close_all()
        assert mcp._SESSIONS == {}
        assert s.proc.poll() is not None           # the server did not outlive
    finally:
        _reset()
