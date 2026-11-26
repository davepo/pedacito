"""Workspace helpers: snapshot restore."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import checkpoints, workspace


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


def test_acquire_lock_is_refcounted_across_threads(tmp_path):
    """A lock held by one thread (same session) must not be unlinked by
    another thread in the same process: the file survives until the LAST
    holder releases it."""
    lock = tmp_path / ".pedacito" / "run.lock"
    entered = threading.Barrier(2)
    release = threading.Barrier(2)
    first = workspace.acquire_lock(tmp_path, session_id="ses-test")
    assert first == lock and lock.is_file()

    def second_thread():
        workspace.acquire_lock(tmp_path, session_id="ses-test")  # same-session re-entry
        entered.wait()
        release.wait()
        workspace.release_lock(tmp_path)

    t = threading.Thread(target=second_thread)
    t.start()
    entered.wait()                             # the thread now holds it too
    workspace.release_lock(tmp_path)           # main thread lets go
    assert lock.is_file(), "the lock was unlinked while the thread still held it"
    release.wait()
    t.join()
    assert not lock.exists()                   # last release removed it


def test_checkpoint_id_traversal_is_rejected(tmp_path):
    for bad in ("../secret", "..", "a/b", "a\\b", "cp_../x", "cp.1"):
        with pytest.raises(checkpoints.CheckpointError):
            checkpoints.snapshot_dir(tmp_path, bad)
    assert checkpoints.snapshot_dir(tmp_path, "cp_0001_deadbe") == \
        checkpoints.checkpoint_root(tmp_path) / "snapshots" / "cp_0001_deadbe"


def test_prune_empty_dirs_does_not_climb_into_a_sibling_prefix(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    sibling = tmp_path / "proj_evil"       # shares the "proj" string prefix
    sibling.mkdir()

    checkpoints._prune_empty_dirs(root, sibling)

    assert sibling.exists()                # never removed: outside the root
    assert root.exists()
