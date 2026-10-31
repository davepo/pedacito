"""Diagnostics detection/run and the `diagnose` tool."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import diagnostics, tools
from pedacito.config import Config
from pedacito.index import Index


def test_detect_prefers_ruff_and_targets_python_files(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.setattr(diagnostics.shutil, "which",
                        lambda name: "/usr/bin/" + name
                        if name in ("ruff", "npm") else "")
    checks = diagnostics.detect(tmp_path, ["pkg/app.py", "web/app.ts"])
    assert checks
    assert checks[0].label == "ruff"
    assert "pkg/app.py" in checks[0].argv
    assert "web/app.ts" not in checks[0].argv


def test_flake8_fallback_argv_is_a_list(tmp_path, monkeypatch):
    (tmp_path / "setup.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(diagnostics.shutil, "which",
                        lambda name: "/usr/bin/flake8" if name == "flake8" else "")
    checks = diagnostics.detect(tmp_path, ["app.py"])
    assert [c.argv for c in checks] == [["flake8", "app.py"]]


def test_no_detected_checker_is_not_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: "")
    ok, report = diagnostics.run(tmp_path, Config())
    assert ok
    assert "No linter/type checker detected" in report


def test_run_reports_failures_and_respects_the_output_cap(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.setattr(diagnostics.shutil, "which",
                        lambda name: "/usr/bin/ruff" if name == "ruff" else "")

    def fake_run(argv, **kw):
        return 1, "E501 line too long\n" + "x" * 5000, "", 0.2

    monkeypatch.setattr(diagnostics.proc, "run", fake_run)
    cfg = Config()
    cfg.max_diagnostics_chars = 300
    ok, report = diagnostics.run(tmp_path, cfg, ["app.py"])
    assert not ok
    assert "CHECK ruff: FAIL" in report
    assert "E501" in report
    assert len(report) <= 360


def test_diagnose_tool_is_gated_by_allow_shell(tmp_path):
    idx = Index(tmp_path, [])
    obs = tools.tool_diagnose(idx, Config())
    assert obs.startswith("ERROR") and "allow_shell" in obs


def test_diagnose_tool_runs_detected_checks(tmp_path, monkeypatch):
    idx = Index(tmp_path, [])
    cfg = Config()
    cfg.allow_shell = True
    seen = []

    def fake_run_changed(root, cfg_, files):
        seen.extend(files or [])
        return True, "CHECK ruff: PASS"

    monkeypatch.setattr(diagnostics, "run_changed", fake_run_changed)
    obs = tools.tool_diagnose(idx, cfg)
    assert "CHECK ruff: PASS" in obs
    assert seen == []


# ---------------------------------------------- quick findings (post-edit loop)
def test_quick_findings_python_syntax_error_via_ast_fallback(tmp_path,
                                                             monkeypatch):
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: "")
    (tmp_path / "bad.py").write_text("def f(:\n    pass\n", encoding="utf-8")
    (tmp_path / "good.py").write_text("x = 1\n", encoding="utf-8")
    found = diagnostics.quick_findings(tmp_path, ["bad.py", "good.py"])
    assert found["good.py"] == []
    assert len(found["bad.py"]) == 1
    assert found["bad.py"][0].startswith("bad.py:1:")
    assert "syntax error" in found["bad.py"][0]


def test_quick_findings_prefers_an_installed_linter(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics.shutil, "which",
                        lambda name: "/usr/bin/ruff" if name == "ruff" else "")
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        return 1, "app.py:2:5: F821 undefined name 'y'\n", "", 0.1

    monkeypatch.setattr(diagnostics.proc, "run", fake_run)
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    found = diagnostics.quick_findings(tmp_path, ["app.py"])
    assert calls == [["ruff", "check", "--output-format", "concise", "app.py"]]
    assert found["app.py"] == ["app.py:2:5: F821 undefined name 'y'"]


def test_quick_findings_js_uses_tree_sitter(tmp_path):
    from pedacito.chunker_ts import TS_AVAILABLE
    (tmp_path / "app.js").write_text("function f( {\n", encoding="utf-8")
    (tmp_path / "ok.js").write_text("function f() { return 1; }\n",
                                    encoding="utf-8")
    found = diagnostics.quick_findings(tmp_path, ["app.js", "ok.js"])
    if TS_AVAILABLE:
        assert found["app.js"] and "syntax error" in found["app.js"][0]
        assert found["ok.js"] == []
    else:
        assert found["app.js"] == [] and found["ok.js"] == []


def test_quick_findings_skips_uncheckable_and_missing_files(tmp_path,
                                                            monkeypatch):
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: "")
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
    found = diagnostics.quick_findings(tmp_path,
                                       ["notes.txt", "gone.py", "", None])
    assert "notes.txt" not in found      # no cheap checker for .txt
    assert found["gone.py"] == []        # missing on disk: no crash


def test_new_findings_ignores_shifted_positions():
    baseline = {"app.py": ["app.py:3:1: E501 line too long"],
                "util.py": ["util.py: line 12 unused"]}
    current = {"app.py": ["app.py:9:1: E501 line too long",
                          "app.py:10:5: F821 undefined name 'y'"],
               "util.py": ["util.py: line 14 unused"]}
    new = diagnostics.new_findings(baseline, current)
    assert list(new) == ["app.py"]
    assert new["app.py"] == ["app.py:10:5: F821 undefined name 'y'"]


def test_new_findings_keeps_files_missing_from_the_baseline():
    new = diagnostics.new_findings({}, {"fresh.py": ["fresh.py:1:1: boom"]})
    assert new == {"fresh.py": ["fresh.py:1:1: boom"]}
    assert diagnostics.new_findings({"a.py": ["x"]}, {}) == {}


def test_render_findings_joins_and_caps():
    found = {"a.py": [f"a.py:{i}:1: problem {i}" for i in range(400)]}
    text = diagnostics.render_findings(found, cap=500)
    assert text.endswith("[... more findings truncated]")
    assert len(text) <= 500 + len("\n[... more findings truncated]")
    assert diagnostics.render_findings({}, 4000) == ""
    both = diagnostics.render_findings({"a.py": ["one"], "b.py": ["two"]})
    assert both == "one\ntwo"
