"""Git worktree helper and CLI plumbing."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import worktree

pytestmark = pytest.mark.skipif(not shutil.which("git"),
                                reason="git is not installed")


def _repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "a@b.c"], cwd=tmp_path,
                   check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path,
                   check=True)
    (tmp_path / "a.txt").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "add", "a.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path,
                   check=True)
    return tmp_path


def test_create_list_remove_worktree(tmp_path):
    root = _repo(tmp_path)
    dest = worktree.create(root, "demo")
    assert dest.is_dir() and (dest / ".git").exists()
    assert (dest / "a.txt").read_text(encoding="utf-8") == "hi\n"
    # git prints paths in its own form (forward slashes on Windows, resolved
    # /private/var paths on macOS), so compare Path objects rather than strings.
    listed = {Path(p).resolve() for p in dict(worktree.list_paths(root))}
    assert dest.resolve() in listed
    worktree.remove(root, "demo")
    assert not dest.exists()
