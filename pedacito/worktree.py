"""Isolated git worktrees for risky autonomous runs.

Usage
-----
    from . import worktree

    path = worktree.create(root, "refactor")
    print(worktree.list_paths(root))

Pedacito already snapshots every write under `.pedacito/backups/` and can
restore it. A git worktree adds the other half of the safety story: run
`pedacito auto` on a separate branch and checkout, review the diff in git, and
merge only if it is good. This module is a thin wrapper around the git CLI --
it never touches the working tree it is called from.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class WorktreeError(RuntimeError):
    pass


def _git(root: Path, args: list[str], timeout: int = 60) -> str:
    if not shutil.which("git"):
        raise WorktreeError("git is not installed or not on PATH")
    try:
        p = subprocess.run(["git", *args], cwd=str(root),
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise WorktreeError(f"git failed: {e}") from e
    out = ((p.stdout or "") + (p.stderr or "")).strip()
    if p.returncode != 0:
        raise WorktreeError(out or f"git exited {p.returncode}")
    return out


def _root(root) -> Path:
    root = Path(root).expanduser().resolve()
    try:
        return Path(_git(root, ["rev-parse", "--show-toplevel"]))
    except WorktreeError:
        raise WorktreeError(f"{root} is not inside a git repository")


def create(root, name: str, base: str = "HEAD") -> Path:
    """Create (or reuse) a worktree at .pedacito/worktrees/<name>."""
    root = _root(root)
    safe = "".join(c if c.isalnum() or c in "-_." else "-"
                   for c in (name or "").strip())
    if not safe:
        raise WorktreeError("worktree name is empty")
    dest = root / ".pedacito" / "worktrees" / safe
    if (dest / ".git").exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    branch = f"pedacito/{safe}"
    branches = _git(root, ["branch", "--list", branch])
    if branches.strip():
        _git(root, ["worktree", "add", str(dest), branch])
    else:
        _git(root, ["worktree", "add", "-b", branch, str(dest), base])
    return dest


def list_paths(root) -> list[tuple[str, str]]:
    """Return (path, branch) for every worktree of this repository."""
    root = _root(root)
    out = _git(root, ["worktree", "list", "--porcelain"])
    entries: list[tuple[str, str]] = []
    path = branch = ""
    for line in out.splitlines() + [""]:
        if line.startswith("worktree "):
            path = line[len("worktree "):].strip()
        elif line.startswith("branch "):
            branch = line[len("branch "):].strip()
        elif not line:
            if path:
                entries.append((path, branch or "(detached)"))
            path = branch = ""
    return entries


def remove(root, name: str, force: bool = False) -> None:
    root = _root(root)
    safe = "".join(c if c.isalnum() or c in "-_." else "-"
                   for c in (name or "").strip())
    dest = root / ".pedacito" / "worktrees" / safe
    args = ["worktree", "remove", str(dest)]
    if force:
        args.append("--force")
    _git(root, args)
