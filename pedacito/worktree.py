"""Isolated git worktrees for risky autonomous runs.

Usage
-----
    from . import worktree

    path = worktree.create(root, "refactor")
    print(worktree.list_paths(root))

    handle = worktree.create_session_worktree(
        root, "refactor", session_id="ses_abc123", ephemeral=True)

Lets `pedacito auto` run on a separate branch and checkout so the diff can be
reviewed in git. A thin wrapper around the git CLI; it never touches the
working tree it is called from.

`create_session_worktree` adds the D8 contract: collision-free per-session
branch names, a per-session state dir, dirty-tree handling (`refuse` /
`carry_over` / `stash`), rollback-on-failure, and session-scoped pruning.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

from . import fileio


class WorktreeError(RuntimeError):
    pass


def _safe_name(name: str) -> str:
    """Filesystem- and branch-safe worktree name."""
    return "".join(c if c.isalnum() or c in "-_." else "-"
                   for c in (name or "").strip())


def _git(root: Path, args: list[str], timeout: int = 60) -> str:
    if not shutil.which("git"):
        raise WorktreeError("git is not installed or not on PATH")
    try:
        p = subprocess.run(["git", *args], cwd=str(root), check=False,
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise WorktreeError(f"git failed: {e}") from e
    out = ((p.stdout or "") + (p.stderr or "")).strip()
    if p.returncode != 0:
        raise WorktreeError(out or f"git exited {p.returncode}")
    return out


def _git_raw(root: Path, args: list[str], timeout: int = 60) -> str:
    """`_git` without the trailing-newline strip (needed for exact patches)."""
    if not shutil.which("git"):
        raise WorktreeError("git is not installed or not on PATH")
    try:
        p = subprocess.run(["git", *args], cwd=str(root), check=False,
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise WorktreeError(f"git failed: {e}") from e
    if p.returncode != 0:
        raise WorktreeError(
            ((p.stdout or "") + (p.stderr or "")).strip()
            or f"git exited {p.returncode}")
    return p.stdout or ""


def _root(root) -> Path:
    root = Path(root).expanduser().resolve()
    try:
        return Path(_git(root, ["rev-parse", "--show-toplevel"]))
    except WorktreeError as e:
        raise WorktreeError(f"{root} is not inside a git repository") from e


def branch_name(name: str) -> str:
    """The branch a worktree create/remove pair uses for `name`."""
    safe = _safe_name(name)
    if not safe:
        raise WorktreeError("worktree name is empty")
    return f"pedacito/{safe}"


def branch_exists(root, name: str) -> bool:
    """True when the `pedacito/<name>` branch exists in this repository."""
    root = _root(root)
    return bool(_git(root, ["branch", "--list",
                            branch_name(name)]).strip())


def create(root, name: str, base: str = "HEAD",
           dirty_policy: str = "ignore") -> Path:
    """Create (or reuse) a worktree at .pedacito/worktrees/<name>.

    ``dirty_policy`` decides what happens to uncommitted user changes before
    the worktree is cut from ``base``: ``ignore`` (the historical behaviour,
    which silently leaves them behind), ``refuse`` (raise with the dirty file
    list), ``carry_over`` (apply the tracked diff and copy untracked files into
    the new worktree, leaving the root untouched), or ``stash`` (stash them
    first, so the root is clean and the worktree starts from committed HEAD).
    """
    root = _root(root)
    safe = _safe_name(name)
    if not safe:
        raise WorktreeError("worktree name is empty")
    policy = str(dirty_policy or "ignore").strip().lower()
    if policy not in ("ignore", "refuse", "carry_over", "stash"):
        raise WorktreeError(
            f"unknown dirty_policy {dirty_policy!r}; choose ignore, refuse, "
            "carry_over, or stash")
    dest = root / ".pedacito" / "worktrees" / safe
    if (dest / ".git").exists():
        return dest

    carried_diff = ""
    carried_untracked: list[str] = []
    if policy != "ignore" and is_dirty(root):
        files = dirty_files(root)
        if policy == "refuse":
            raise WorktreeError(
                "the working tree has uncommitted changes; commit or stash "
                "them first (or pass --dirty carry_over/stash). Dirty files: "
                + ", ".join(files[:20])
                + (f" (+{len(files) - 20} more)" if len(files) > 20 else ""))
        if policy == "carry_over":
            carried_diff = _git_raw(root, ["diff", "HEAD"])
            carried_untracked = _untracked_files(root)
        else:                          # stash
            _git(root, ["stash", "push", "-u", "-m",
                        f"pedacito:worktree:{safe}"])

    dest.parent.mkdir(parents=True, exist_ok=True)
    branch = f"pedacito/{safe}"
    branches = _git(root, ["branch", "--list", branch])
    created_branch = not branches.strip()
    try:
        if branches.strip():
            _git(root, ["worktree", "add", str(dest), branch])
        else:
            _git(root, ["worktree", "add", "-b", branch, str(dest), base])
        if carried_diff or carried_untracked:
            _apply_patch(dest, carried_diff)
            for rel in carried_untracked:
                src = root / rel
                if not src.is_file():
                    continue
                dst = dest / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                fileio.copy_file_atomic(src, dst)
    except BaseException:
        # Never delete a branch that existed before this call.
        _cleanup_partial(root, dest, branch if created_branch else "")
        raise
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


def _dirty_in_worktree(dest: Path) -> bool:
    """True when a worktree directory still has uncommitted user changes.

    Used only to translate git's refusal into an actionable Pedacito message,
    so any probe failure simply means "not known dirty"."""
    try:
        return dest.is_dir() and bool(dirty_files(dest))
    except (WorktreeError, OSError):
        return False


def remove(root, name: str, force: bool = False,
           delete_branch: bool = True) -> bool:
    """Remove the worktree .pedacito/worktrees/<name>.

    With ``delete_branch`` (the default) the matching ``pedacito/<name>``
    branch is deleted too, so a later ``create`` cannot silently start from a
    stale branch tip. Returns True when that branch was deleted, False when it
    was kept, absent, or could not be removed (best effort)."""
    root = _root(root)
    safe = _safe_name(name)
    if not safe:
        raise WorktreeError("worktree name is empty")
    dest = root / ".pedacito" / "worktrees" / safe
    args = ["worktree", "remove", str(dest)]
    if force:
        args.append("--force")
    try:
        _git(root, args)
    except WorktreeError as e:
        # A `--dirty carry_over` worktree (or any worktree edited in place)
        # has uncommitted work git refuses to discard. Translate the raw
        # `fatal: ... use --force` into an actionable Pedacito message --
        # and never drop the branch, so the work is still recoverable
        # (R2-BUG-03).
        if not force and ("contains modified or untracked files" in str(e)
                          or _dirty_in_worktree(dest)):
            raise WorktreeError(
                f"worktree '{safe}' has uncommitted changes (branch "
                f"'pedacito/{safe}' preserved); commit them there or "
                "pass --force to discard and remove.") from None
        raise
    if not delete_branch:
        return False
    branch = f"pedacito/{safe}"
    if not _git(root, ["branch", "--list", branch]).strip():
        return False
    try:
        _git(root, ["branch", "-D", branch])
        return True
    except WorktreeError:
        return False


# ---------------------------------------------------------------------------
# D8: dirty-tree handling, per-session state, collision-free branches
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorktreeHandle:
    """Everything needed to later remove one session worktree."""

    repo_root: Path
    path: Path
    branch: str
    name: str
    session_id: str
    state_path: Path
    stash_ref: str = ""
    carried_patch: bool = False


def _is_state_path(name: str) -> bool:
    """True for Pedacito's own bookkeeping (never counts as user work)."""
    n = name.replace("\\", "/")
    while n.startswith("./"):
        n = n[2:]
    n = n.lstrip("/")
    return (n == ".pedacito" or n.startswith(".pedacito/")
            or n == ".pedacitoignore")


def is_dirty(root: Path | str) -> bool:
    """True when the working tree has any tracked or untracked user changes."""
    return bool(dirty_files(root))


def _parse_status_line(line: str) -> str:
    """One porcelain status line -> the (possibly renamed) affected path."""
    body = line[3:].strip()
    if " -> " in body:
        # `R  old -> new`: report the NEW path (the one a worktree should carry).
        body = body.rsplit(" -> ", 1)[-1]
    # git quotes non-ASCII paths (core.quotepath); unquote the common form.
    if len(body) >= 2 and body[0] == body[-1] == '"':
        body = body[1:-1].replace('\\"', '"')
    return body


def dirty_files(root: Path | str) -> list[str]:
    """The project-relative paths the working tree has modified or added,
    excluding Pedacito's own `.pedacito/` state directory."""
    root = _root(root)
    # `_git_raw` (not `_git`): porcelain status lines carry a leading space
    # for unstaged changes, which `_git`'s strip would eat and misalign.
    out = _git_raw(root, ["status", "--porcelain"])
    files: list[str] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        name = _parse_status_line(line)
        if name and not _is_state_path(name):
            files.append(name)
    return files


def _session_short(session_id: str) -> str:
    sid = _safe_name(session_id or "nosession")
    return sid[-8:] if len(sid) > 8 else (sid or "s")


_seq_lock = threading.Lock()
_seq_counter = [0]


def _next_seq() -> int:
    with _seq_lock:
        _seq_counter[0] += 1
        return _seq_counter[0]


def _untracked_files(root: Path) -> list[str]:
    out = _git(root, ["ls-files", "--others", "--exclude-standard"])
    return [ln for ln in out.splitlines() if ln.strip()]


def _apply_patch(dest: Path, diff: str) -> None:
    """Apply a tracked diff (via stdin, preserving line endings) in `dest`."""
    if not diff.strip():
        return
    p = subprocess.run(
        ["git", "apply", "--whitespace=nowarn"],
        cwd=str(dest), input=diff, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=60)
    if p.returncode != 0:
        raise WorktreeError(
            ((p.stdout or "") + (p.stderr or "")).strip()
            or f"git apply exited {p.returncode}")


def _cleanup_partial(root: Path, dest: Path, branch: str) -> None:
    """Remove every trace of a half-created worktree and branch."""
    for args in (["worktree", "remove", "--force", str(dest)],
                 ["worktree", "prune"]):
        try:
            _git(root, args)
        except WorktreeError:
            pass
    if branch:
        try:
            _git(root, ["branch", "-D", branch])
        except WorktreeError:
            pass
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)


def _register_worktree(handle: WorktreeHandle, ephemeral: bool) -> None:
    """Register an ephemeral worktree with the bound session's registry."""
    if not ephemeral:
        return
    try:
        from . import session_ctx
        sess = session_ctx.current_session()
    except Exception:
        return
    if sess is None or sess.resource_registry is None:
        return
    if str(sess.session_id) != str(handle.session_id):
        return
    try:
        sess.resource_registry.register(
            "worktree", str(handle.path),
            lambda: _remove_handle_quiet(handle))
    except Exception:
        pass


def _remove_handle_quiet(handle: WorktreeHandle) -> None:
    try:
        remove_session_worktree(handle, force=True)
    except Exception:
        pass


def create_session_worktree(
    root: Path | str,
    name: str,
    *,
    session_id: str,
    base: str = "HEAD",
    dirty_policy: str = "refuse",     # "refuse" | "carry_over" | "stash"
    ephemeral: bool = False,
) -> WorktreeHandle:
    """Create an isolated worktree for one session, handling a dirty tree.

    Branch format: ``pedacito/<safe_name>-<session_short>-<seq>``, unique even
    when two threads create worktrees with the same ``name`` at once. A dirty
    tree is refused (default), carried over (tracked diff + untracked files), or
    stashed (recorded in ``stash_ref``) per ``dirty_policy``. Any failure during
    creation cleans up the partial worktree, branch, and directory so zero stale
    artifacts remain.
    """
    repo = _root(root)
    safe = _safe_name(name)
    if not safe:
        raise WorktreeError("worktree name is empty")
    policy = str(dirty_policy or "refuse").strip().lower()

    carried_diff = ""
    carried_untracked: list[str] = []
    stash_ref = ""
    carried = False
    if is_dirty(repo):
        if policy == "refuse":
            files = dirty_files(repo)
            raise WorktreeError(
                "the working tree has uncommitted changes; commit or stash "
                "them first (or use --worktree with carry_over/stash). Dirty "
                "files: " + ", ".join(files[:20])
                + (f" (+{len(files) - 20} more)" if len(files) > 20 else ""))
        if policy == "carry_over":
            carried = True
            carried_diff = _git_raw(repo, ["diff", "HEAD"])
            carried_untracked = _untracked_files(repo)
        elif policy == "stash":
            stash_ref = _git(
                repo, ["stash", "push", "-u", "-m",
                       f"pedacito:{session_id}:{name}"])
        else:
            raise WorktreeError(
                f"unknown dirty_policy {dirty_policy!r}; choose refuse, "
                "carry_over, or stash")

    short = _session_short(session_id)
    seq = _next_seq()
    branch = f"pedacito/{safe}-{short}-{seq}"
    # Collision-free across concurrent sessions: bump until the branch is free.
    while _git(repo, ["branch", "--list", branch]).strip():
        seq = _next_seq()
        branch = f"pedacito/{safe}-{short}-{seq}"

    dest = repo / ".pedacito" / "worktrees" / f"{safe}-{short}-{seq}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        _git(repo, ["worktree", "add", "-b", branch, str(dest), base])
        if carried:
            _apply_patch(dest, carried_diff)
            for rel in carried_untracked:
                src = repo / rel
                if not src.exists():
                    continue
                dst = dest / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                fileio.copy_file_atomic(src, dst)
        state_path = dest / ".pedacito" / "sessions_state" / str(session_id)
        state_path.mkdir(parents=True, exist_ok=True)
    except BaseException:
        _cleanup_partial(repo, dest, branch)
        raise

    handle = WorktreeHandle(
        repo_root=repo,
        path=dest,
        branch=branch,
        name=safe,
        session_id=str(session_id),
        state_path=state_path,
        stash_ref=stash_ref,
        carried_patch=carried,
    )
    _register_worktree(handle, ephemeral)
    # Record the worktree path in the bound session's task journal (if any) so
    # a crash rollback can clean it up.
    try:
        from . import session_ctx
        sess = session_ctx.current_session()
        if sess is not None:
            journal = getattr(sess, "_task_journal", None)
            if journal is not None:
                journal.append("checkpoint", worktree=str(dest))
    except Exception:
        pass
    return handle


def remove_session_worktree(
    handle: WorktreeHandle,
    *,
    force: bool = True,
    delete_branch: bool = False,
) -> None:
    """Remove one session worktree (idempotent; a missing dir is a no-op)."""
    repo = handle.repo_root
    dest = handle.path
    if dest.exists() or (dest / ".git").exists():
        args = ["worktree", "remove", str(dest)]
        if force:
            args.append("--force")
        try:
            _git(repo, args)
        except WorktreeError:
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
    try:
        _git(repo, ["worktree", "prune"])
    except WorktreeError:
        pass
    if delete_branch and handle.branch:
        try:
            _git(repo, ["branch", "-D", handle.branch])
        except WorktreeError:
            pass


def _session_ids_of(worktree_dir: Path) -> list[str]:
    ss = worktree_dir / ".pedacito" / "sessions_state"
    if not ss.is_dir():
        return []
    return sorted(p.name for p in ss.iterdir() if p.is_dir())


def prune_stale(
    root: Path | str,
    *,
    active_session_ids: set[str] | None = None,
) -> list[str]:
    """Remove pedacito-managed worktrees whose session is no longer active.

    A worktree is "session-managed" when it has a per-session state dir under
    ``.pedacito/sessions_state/``. It is pruned when NONE of its session ids is
    in ``active_session_ids`` (an empty/None set means "nothing is active", so
    every session-managed worktree is pruned). Returns the removed paths.
    """
    repo = _root(root)
    active = set(active_session_ids or ())
    base = repo / ".pedacito" / "worktrees"
    pruned: list[str] = []
    if not base.is_dir():
        return pruned
    for wt_dir in sorted(base.iterdir()):
        if not wt_dir.is_dir():
            continue
        sids = _session_ids_of(wt_dir)
        if not sids:
            continue                      # not session-managed: leave it alone
        if any(s in active for s in sids):
            continue                      # still owned by a live session
        try:
            _git(repo, ["worktree", "remove", "--force", str(wt_dir)])
            pruned.append(str(wt_dir))
        except WorktreeError:
            if wt_dir.exists():
                shutil.rmtree(wt_dir, ignore_errors=True)
                pruned.append(str(wt_dir))
    try:
        _git(repo, ["worktree", "prune"])
    except WorktreeError:
        pass
    return pruned
