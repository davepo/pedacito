"""Session worktree isolation (D8): collision-free branch names, per-session
state dirs, dirty-tree policies (refuse / carry_over / stash), rollback on
failure, ephemeral auto-removal, and stale pruning.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import worktree
from pedacito.session_ctx import SessionContext, use_session

pytestmark = pytest.mark.skipif(not shutil.which("git"),
                                reason="git is not installed")


def _repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "a@b.c"], cwd=tmp_path,
                   check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path,
                   check=True)
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "a.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    return tmp_path


def _git(root: Path, *args: str) -> str:
    p = subprocess.run(["git", *args], cwd=str(root), check=False,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    return (p.stdout or "") + (p.stderr or "")


def _branches(root: Path) -> list[str]:
    return [ln.strip() for ln in _git(root, "branch", "--list").splitlines()
            if ln.strip()]


def test_clean_worktree_branch_and_state_dir(tmp_path):
    root = _repo(tmp_path)
    h = worktree.create_session_worktree(
        root, "refactor", session_id="ses_abc123", base="HEAD")
    try:
        assert h.path.is_dir() and (h.path / ".git").exists()
        assert h.branch.startswith("pedacito/refactor-")
        assert h.session_id == "ses_abc123"
        assert h.state_path.is_dir()
        assert h.state_path == h.path / ".pedacito" / "sessions_state" / "ses_abc123"
        assert (h.path / "a.txt").read_text(encoding="utf-8") == "hello\n"
    finally:
        worktree.remove_session_worktree(h, force=True, delete_branch=True)


def test_collision_free_branch_names(tmp_path):
    root = _repo(tmp_path)
    h1 = worktree.create_session_worktree(root, "same", session_id="ses_one")
    h2 = worktree.create_session_worktree(root, "same", session_id="ses_two")
    try:
        assert h1.branch != h2.branch
        assert h1.path != h2.path
        assert h1.path.is_dir() and h2.path.is_dir()
    finally:
        worktree.remove_session_worktree(h1, force=True, delete_branch=True)
        worktree.remove_session_worktree(h2, force=True, delete_branch=True)


def test_dirty_refuse_raises_without_creating_worktree(tmp_path):
    root = _repo(tmp_path)
    (root / "a.txt").write_text("dirty\n", encoding="utf-8")
    before = _branches(root)
    with pytest.raises(worktree.WorktreeError) as err:
        worktree.create_session_worktree(root, "wt", session_id="ses_x")
    assert "uncommitted" in str(err.value) and "a.txt" in str(err.value)
    assert _branches(root) == before              # no branch created
    assert not (root / ".pedacito" / "worktrees").exists()


def test_dirty_carry_over_transfers_changes_without_mutating_root(tmp_path):
    root = _repo(tmp_path)
    (root / "a.txt").write_text("tracked change\n", encoding="utf-8")
    (root / "new.txt").write_text("untracked\n", encoding="utf-8")
    h = worktree.create_session_worktree(
        root, "carry", session_id="ses_c", dirty_policy="carry_over")
    try:
        # The new worktree has the tracked change AND the untracked file.
        assert (h.path / "a.txt").read_text(encoding="utf-8") == "tracked change\n"
        assert (h.path / "new.txt").read_text(encoding="utf-8") == "untracked\n"
        assert h.carried_patch is True
        # The repo root is untouched: still dirty.
        assert (root / "a.txt").read_text(encoding="utf-8") == "tracked change\n"
        assert (root / "new.txt").exists()
    finally:
        worktree.remove_session_worktree(h, force=True, delete_branch=True)


def test_dirty_stash_records_stash_ref_and_cleans_root(tmp_path):
    root = _repo(tmp_path)
    (root / "a.txt").write_text("stash me\n", encoding="utf-8")
    h = worktree.create_session_worktree(
        root, "stash", session_id="ses_s", dirty_policy="stash")
    try:
        assert h.stash_ref
        # The root is clean again (the change was stashed, not carried).
        assert not worktree.is_dirty(root)
        # The worktree has the pre-change content.
        assert (h.path / "a.txt").read_text(encoding="utf-8") == "hello\n"
    finally:
        worktree.remove_session_worktree(h, force=True, delete_branch=True)


def test_failure_rollback_leaves_zero_leftovers(tmp_path):
    root = _repo(tmp_path)
    before = _branches(root)
    with pytest.raises(worktree.WorktreeError):
        worktree.create_session_worktree(
            root, "bad", session_id="ses_b", base="no-such-ref")
    # No branch, no worktree dir, no stray entry.
    assert _branches(root) == before
    assert not (root / ".pedacito" / "worktrees" / "bad-ses_b-0").exists()
    listed = {Path(p).resolve() for p in dict(worktree.list_paths(root))}
    assert root.resolve() in listed


def test_ephemeral_worktree_removed_on_cancel(tmp_path):
    root = _repo(tmp_path)
    sess = SessionContext()
    h = None
    with use_session(sess):
        h = worktree.create_session_worktree(
            root, "eph", session_id=sess.session_id, ephemeral=True)
    assert h is not None and h.path.is_dir()
    sess.cancel()
    assert not h.path.exists()


def test_prune_stale_preserves_active_sessions(tmp_path):
    root = _repo(tmp_path)
    h_a = worktree.create_session_worktree(root, "live", session_id="ses_live")
    h_b = worktree.create_session_worktree(root, "orphan", session_id="ses_old")
    try:
        pruned = worktree.prune_stale(root, active_session_ids={"ses_live"})
        assert str(h_b.path) in pruned
        assert h_a.path.exists() and not h_b.path.exists()
        # With no active sessions left, the remaining one is pruned too.
        pruned2 = worktree.prune_stale(root)
        assert str(h_a.path) in pruned2
        assert not h_a.path.exists()
    finally:
        worktree.remove_session_worktree(h_a, force=True, delete_branch=True)
        worktree.remove_session_worktree(h_b, force=True, delete_branch=True)
