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
