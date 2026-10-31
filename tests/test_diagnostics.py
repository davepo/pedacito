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
