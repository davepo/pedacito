"""Optional LSP client and the opt-in `lsp` tool."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import lsp, tools
from pedacito.config import Config
from pedacito.index import Index

FAKE = Path(__file__).resolve().parent / "fake_lsp.py"


def _cfg():
    cfg = Config()
    cfg.allow_lsp = True
    cfg.lsp_timeout = 5
    cfg.lsp_servers = {".py": f'"{sys.executable}" "{FAKE}"'}
    return cfg


def _root(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return root


def test_lsp_definition_references_hover_and_diagnostics(tmp_path):
    root = _root(tmp_path)
    cfg = _cfg()
    lsp.close_all()
    try:
        d = lsp.result(root, cfg, "definition", "a.py", 2, 3)
        assert d.startswith("Definition:") and "a.py:5:3" in d
        r = lsp.result(root, cfg, "references", "a.py", 2, 3)
        assert r.startswith("References:") and r.count("a.py:") == 2
        h = lsp.result(root, cfg, "hover", "a.py", 2, 3)
        assert "fake hover" in h
        diag = lsp.result(root, cfg, "diagnostics", "a.py", 1, 1)
        assert "fake diagnostic" in diag
    finally:
        lsp.close_all()


def test_lsp_tool_is_gated_and_dispatches(tmp_path):
    root = _root(tmp_path)
    idx = Index(root, [])
    plain = Config()
    assert tools.tool_lsp(idx, plain, "definition", "a.py").startswith("ERROR")
    cfg = _cfg()
    lsp.close_all()
    try:
        out = tools.dispatch(idx, cfg, {
            "action": "lsp", "symbol": "definition", "file": "a.py",
            "start": 2, "end": 3, "reasoning": ""})
        assert "a.py:5:3" in out
        assert "lsp" in tools.schema_for(cfg)["properties"]["action"]["enum"]
    finally:
        lsp.close_all()


def test_lsp_reports_a_missing_server(tmp_path):
    root = _root(tmp_path)
    cfg = Config()
    cfg.allow_lsp = True
    cfg.lsp_servers = {}
    lsp.close_all()
    out = lsp.result(root, cfg, "definition", "a.py")
    assert out.startswith("ERROR: LSP definition failed")
    assert "no language server" in out
