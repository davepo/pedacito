"""Regression: project-wide diagnostics must not lint Pedacito's own state.

Checkpoints and backups hold full copies of old sources. Linting the state
directory reported stale errors forever, even when the live project was clean,
and made `pedacito diagnose` unusable as a gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import diagnostics
from pedacito.config import Config


def test_project_wide_diagnostics_targets_live_files_not_state(
        tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    stale = tmp_path / ".pedacito" / "backups" / "20260101-000000"
    stale.mkdir(parents=True)
    (stale / "app.py").write_text("def broken(:\n", encoding="utf-8")

    monkeypatch.setattr(diagnostics, "_which",
                        lambda name: "/usr/bin/ruff" if name == "ruff" else "")
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return 0, "", "", 0.1

    monkeypatch.setattr(diagnostics.proc, "run", fake_run)
    ok, _report = diagnostics.run(tmp_path, Config())
    assert ok
    assert calls, "ruff was not invoked"
    argv = " ".join(calls[0])
    assert "app.py" in argv
    assert ".pedacito" not in argv
