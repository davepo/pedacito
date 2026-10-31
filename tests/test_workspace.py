"""Workspace helpers: snapshot restore."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import workspace


def test_restore_reports_only_snapshot_files(tmp_path):
    (tmp_path / "a.py").write_text("new", encoding="utf-8")
    (tmp_path / "untouched.py").write_text("x", encoding="utf-8")
    snap = workspace.new_backup_dir(tmp_path, "s1")
    (snap / "a.py").write_text("old", encoding="utf-8")

    assert workspace.restore(tmp_path, snap, dry_run=True) == ["a.py"]
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "new"
    assert workspace.restore(tmp_path, snap) == ["a.py"]
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "old"
    assert (tmp_path / "untouched.py").read_text(encoding="utf-8") == "x"
