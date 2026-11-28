"""VirtualWorkspace: the shadow overlay behind the interleaved agent loop.

    ws = VirtualWorkspace(root, apply=False, cfg=cfg)
    snap = ws.snapshot()
    res = ws.edit("pricing.py", search="return 0.0", replace="return 0.05")
    ws.read("pricing.py")          # the STAGED text; the file on disk is untouched
    ws.rollback(snap)              # undo everything staged since `snap`
    ws.to_file_diffs()             # review.FileDiff list for --review / the GUI

Two modes, one API:

* `apply=False` (dry run, `--review`): every write lands in an in-memory
  overlay. `read()`, `exists()`, `is_deleted()` and `changed_files()` describe
  the overlay; the real tree is never opened for writing. Commands that must
  SEE the staged edits (tests, linters) run against `exec_root()`, a temp
  mirror of the project with the overlay applied, never against the repo.
* `apply=True`: writes go to disk through `fileio` (atomic replace, the first
  copy of each file backed up into the run's backup directory) and a journal
  of the bytes they replaced makes `rollback()` exact.

Every path is resolved with `security.resolve_safe_path` (no `..`, no symlink
or absolute escape) and the model may never touch `.git/`, Pedacito state, the
protected paths from `safety.is_protected`, or secret/credential files. The
edit itself is the same `edits.apply_edits` engine the two-phase loop uses
(fuzzy anchoring, SYMBOL/LINES anchors, fragment and wipe guards, parse
validation), run against a one-file scratch directory holding the file's
CURRENT virtual text, so staged and applied edits behave identically.
"""

from __future__ import annotations

import contextlib
import dataclasses
import difflib
import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path

from . import fileio, safety, security
from .edits import Edit, Result, apply_edits

# The roadmap calls the per-file outcome `EditResult`; the engine's name for it
# is `edits.Result`. Same class.
EditResult = Result

# A dry-run mirror is a copy of the tracked project files: refuse to build one
# for a tree this large rather than fill a temp disk.
MAX_MIRROR_BYTES = 400_000_000


def _link_node_modules(link: Path, target: Path) -> None:
    """Link a project's `node_modules` into a mirror, or skip cleanly.

    An unprivileged symlink fails on Windows without the developer-mode
    ``SeCreateSymbolicLinkPrivilege`` (``OSError`` / ``WinError 1314``). Fall
    back to a directory junction (``mklink /J``, which needs no privilege), and
    finally skip without failing mirror creation -- the project copy is still
    fully usable, just without its already-installed dependencies.
    """
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError:
        pass
    if os.name == "nt":
        try:
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                capture_output=True, check=False, timeout=30)
            if link.exists():
                return
        except (OSError, subprocess.SubprocessError):
            pass
    # Last resort: leave `node_modules` absent rather than crash the mirror.


def _register_mirror(ws: "VirtualWorkspace") -> None:
    """Register `ws.close` as a `temp_mirror` resource under the bound session."""
    try:
        from . import session_ctx
        sess = session_ctx.current_session()
    except Exception:
        return
    if sess is None or sess.resource_registry is None:
        return
    try:
        ws._mirror_resource_id = sess.resource_registry.register(
            "temp_mirror", str(ws._mirror or ""), ws.close)
    except Exception:
        pass


def _unregister_mirror(ws: "VirtualWorkspace") -> None:
    rid = getattr(ws, "_mirror_resource_id", None)
    if not rid:
        return
    try:
        from . import session_ctx
        sess = session_ctx.current_session()
        if sess is not None and sess.resource_registry is not None:
            sess.resource_registry.unregister(rid)
    except Exception:
        pass
    ws._mirror_resource_id = None


class WorkspaceError(Exception):
    """A request the workspace cannot honour (unreadable, not text, too big)."""


class PathRefused(WorkspaceError):
    """A path outside the project, or one the model may never touch.
    `sensitive` marks a secret/credential file, whose READ the permission
    broker may still approve (a write is never allowed)."""

    sensitive = False


@dataclass(frozen=True)
class _Entry:
    """Everything known about one touched file. Immutable, so a snapshot is a
    shallow copy of the entries dict."""

    original: str | None            # text at first touch; None = did not exist
    original_style: fileio.TextStyle | None
    current: str | None             # None = deleted
    style: fileio.TextStyle
    edits: int = 1                  # successful edit operations folded in
    fuzzy: bool = False             # some block anchored approximately


@dataclass(frozen=True)
class Snapshot:
    """A restore point for `VirtualWorkspace.rollback`."""

    owner: int
    entries: dict
    journal_len: int
    revision: int


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unified(rel: str, before: str, after: str, deleted: bool = False) -> str:
    return "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=f"a/{rel}", tofile="/dev/null" if deleted else f"b/{rel}"))


class VirtualWorkspace:
    """A project tree plus the edits staged (or applied) on top of it."""

    def __init__(self, root, *, apply: bool = True, cfg=None,
                 backup_dir=None):
        self.root = Path(str(root)).expanduser().resolve()
        if not self.root.is_dir():
            raise WorkspaceError(f"project root is not a directory: {root}")
        self.apply = bool(apply)
        self.cfg = cfg
        self.backup_dir = Path(backup_dir) if backup_dir else None
        # Bumped on every mutation and rollback: callers key caches on it.
        self.revision = 0
        self._entries: dict[str, _Entry] = {}
        self._journal: list[tuple[str, bytes | None]] = []
        self._lock = threading.RLock()
        self._mirror: Path | None = None
        self._mirror_rev = -1
        self._mirror_touched: set[str] = set()
        self._finalizer = None
        self._mirror_resource_id: str | None = None

    # ------------------------------------------------------------ paths
    def _resolve(self, rel, *, write: bool,
                 allow_sensitive: bool = False) -> tuple[str, Path]:
        """(canonical relative POSIX path, absolute path) or PathRefused."""
        raw = fileio.normalise_rel(str(rel or ""))
        if not raw:
            raise PathRefused("empty path")
        try:
            path = security.resolve_safe_path(self.root, raw)
        except security.SecurityError as exc:
            raise PathRefused(str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise PathRefused(f"path could not be resolved: {raw} ({exc})") \
                from exc
        try:
            canon = path.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise PathRefused(f"path escapes the project root: {raw}") from exc
        if canon in ("", "."):
            raise PathRefused("that path is the project root, not a file")
        why = safety.is_protected(canon, self.cfg if write else None, write=write)
        if why:
            raise PathRefused(why)
        if write and not allow_sensitive:
            # D10: the SAME gate as the two-phase engine. Every write/edit/
            # create/delete/rename path goes through `_resolve(write=True)`,
            # so sensitive files are refused identically in both engines.
            allowed, error = security.check_sensitive_write(
                self.root, canon, action="write",
                policy=getattr(self.cfg, "policy", None))
            if not allowed:
                exc = PathRefused(error)
                exc.sensitive = True
                raise exc
        if not allow_sensitive and security.is_sensitive_path(canon):
            exc = PathRefused(
                f"`{canon}` holds secrets/credentials; it is never read or "
                "written by the model")
            exc.sensitive = True
            raise exc
        return canon, path

    def resolve(self, rel, *, write: bool = False,
                allow_sensitive: bool = False) -> str:
        """The canonical project-relative path for `rel`, or PathRefused."""
        return self._resolve(rel, write=write,
                             allow_sensitive=allow_sensitive)[0]

    # ------------------------------------------------------------ reads
    @staticmethod
    def _disk(path: Path) -> tuple[str | None, fileio.TextStyle]:
        if path.is_dir():
            raise WorkspaceError("this path is a directory, not a file")
        if not path.is_file():
            return None, fileio.TextStyle()
        try:
            return fileio.read(path, errors="strict")
        except UnicodeDecodeError as exc:
            raise WorkspaceError(
                "file is not valid UTF-8 text; it cannot be read or edited "
                "as text") from exc

    def _current(self, rel: str, path: Path
                 ) -> tuple[str | None, fileio.TextStyle]:
        if not self.apply:
            entry = self._entries.get(rel)
            if entry is not None:
                return entry.current, entry.style
        return self._disk(path)

    def read(self, rel, *, allow_sensitive: bool = False) -> str:
        """The file's current text: staged content in overlay mode, the bytes
        on disk in apply mode (which are the applied content). Raises
        PathRefused, FileNotFoundError (absent or deleted), or WorkspaceError."""
        with self._lock:
            canon, path = self._resolve(rel, write=False,
                                        allow_sensitive=allow_sensitive)
            text, _style = self._current(canon, path)
            if text is None:
                raise FileNotFoundError(canon)
            return text

    def exists(self, rel) -> bool:
        with self._lock:
            try:
                canon, path = self._resolve(rel, write=False)
                text, _ = self._current(canon, path)
            except (WorkspaceError, OSError):
                return False
            return text is not None

    def is_deleted(self, rel) -> bool:
        """True when this run deleted the file (and it is still gone)."""
        with self._lock:
            try:
                canon, path = self._resolve(rel, write=False)
            except WorkspaceError:
                return False
            entry = self._entries.get(canon)
            if entry is None or entry.current is not None:
                return False
            if self.apply:
                return not path.exists()
            return True

    def is_staged(self, rel) -> bool:
        """True when this run has touched the file (created, edited, deleted)."""
        with self._lock:
            try:
                return self.resolve(rel) in self._entries
            except WorkspaceError:
                return False

    def original_text(self, rel) -> str | None:
        """The file as it was before this run touched it (None = it did not
        exist; also None for a file never touched, ask `read` for that)."""
        with self._lock:
            entry = self._entries.get(self.resolve(rel))
            return entry.original if entry is not None else None

    def list_files(self, limit: int = 20000) -> list[str]:
        """Project-relative files of the overlay: tracked files on disk, minus
        this run's deletions, plus its creations. Secret files never appear."""
        from . import checkpoints
        with self._lock:
            found = {rel for rel, _p in checkpoints.iter_workspace_files(
                self.root)}
            for rel, entry in self._entries.items():
                if entry.current is None:
                    found.discard(rel)
                else:
                    found.add(rel)
            out = sorted(r for r in found
                         if not security.is_sensitive_path(r))
            return out[:limit]

    # ----------------------------------------------------------- journal
    def snapshot(self) -> Snapshot:
        """A restore point covering the overlay and (in apply mode) the disk."""
        with self._lock:
            return Snapshot(id(self), dict(self._entries), len(self._journal),
                            self.revision)

    def rollback(self, snap: Snapshot) -> None:
        """Undo every change made since `snap` was taken."""
        if not isinstance(snap, Snapshot) or snap.owner != id(self):
            raise WorkspaceError("that snapshot belongs to another workspace")
        with self._lock:
            failures: list[str] = []
            if self.apply:
                for rel, data in reversed(self._journal[snap.journal_len:]):
                    path = self.root / rel
                    try:
                        if data is None:
                            if path.is_file() or path.is_symlink():
                                path.unlink()
                            self._prune_empty_parents(path)
                        else:
                            fileio.write_bytes(path, data)
                    except OSError as exc:
                        failures.append(f"{rel}: {exc}")
                del self._journal[snap.journal_len:]
            self._entries = dict(snap.entries)
            self.revision += 1
            if failures:
                raise WorkspaceError(
                    "rollback could not restore: " + "; ".join(failures)
                    + " -- run `pedacito restore`")

    def _prune_empty_parents(self, path: Path) -> None:
        parent = path.parent
        while parent != self.root and self.root in parent.parents:
            try:
                parent.rmdir()
            except OSError:
                return
            parent = parent.parent

    # ----------------------------------------------------------- writing
    def _backup(self, rel: str, path: Path, existed: bool) -> None:
        if not getattr(self.cfg, "backup", True):
            return
        try:
            if self.backup_dir is None:
                from . import workspace
                self.backup_dir = workspace.new_backup_dir(self.root,
                                                           unique=True)
            if existed:
                fileio.backup_copy(path, self.backup_dir / rel)
            else:
                fileio.note_created(self.backup_dir, rel)
        except OSError:
            pass          # a failed backup never blocks the edit (best effort)

    def _scratch_apply(self, rel: str, text: str | None,
                       style: fileio.TextStyle, edit: Edit) -> Result:
        """Run `edit` through the real edit engine against a one-file scratch
        directory that holds the file's current virtual text."""
        with tempfile.TemporaryDirectory(prefix="pedacito-ovl-") as tmp:
            scratch = Path(tmp)
            if text is not None:
                target = scratch / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                fileio.write(target, text, style)
            results = apply_edits(scratch, [dataclasses.replace(edit, file=rel)],
                                  backup=False, dry_run=True, cfg=self.cfg)
        failed = [r for r in results if not r.ok]
        if failed:
            return failed[0]
        return results[0] if results else Result(
            False, rel, "the edit engine returned nothing")

    def _commit(self, rel: str, path: Path, before: str | None,
                before_style: fileio.TextStyle, after: str | None,
                after_style: fileio.TextStyle, *, fuzzy: bool = False
                ) -> str | None:
        """Record (and, in apply mode, write) one file's new content. Returns
        an error string when the disk write failed, else None."""
        if self.apply:
            pre = path.read_bytes() if path.is_file() else None
            self._backup(rel, path, existed=pre is not None)
            self._journal.append((rel, pre))
            try:
                if after is None:
                    path.unlink()
                else:
                    fileio.write(path, after, after_style)
            except OSError as exc:
                self._journal.pop()
                return f"could not write {rel}: {exc}"
        prev = self._entries.get(rel)
        if prev is None:
            self._entries[rel] = _Entry(
                before, before_style if before is not None else None,
                after, after_style, 1, fuzzy)
        else:
            self._entries[rel] = dataclasses.replace(
                prev, current=after, style=after_style,
                edits=prev.edits + 1, fuzzy=prev.fuzzy or fuzzy)
        self.revision += 1
        return None

    def apply_edit(self, edit: Edit) -> Result:
        """Apply one `edits.Edit` (search/replace, LINES or SYMBOL anchored,
        creation, deletion) to the workspace. A refused or failed edit changes
        nothing and returns `Result(ok=False)` carrying the reason."""
        with self._lock:
            try:
                rel, path = self._resolve(edit.file, write=True)
            except PathRefused as exc:
                return Result(False, str(edit.file), f"refused: {exc}")
            if edit.delete:
                return self._delete(rel, path)
            try:
                text, style = self._current(rel, path)
            except WorkspaceError as exc:
                return Result(False, rel, str(exc))
            res = self._scratch_apply(rel, text, style, edit)
            if not res.ok:
                return res
            if not res.sha:
                return res                  # "no change: already applied"
            new_text = res.after
            if new_text == text:
                return Result(True, rel,
                              "no change: the file already has this content")
            err = self._commit(rel, path, text, style, new_text, style,
                               fuzzy=res.fuzzy)
            if err:
                return Result(False, rel, err)
            res.message = res.message.replace(" (dry run)", "")
            if not self.apply:
                res.message += " [staged in memory; the file on disk is untouched]"
            return res

    def edit(self, rel, search: str = "", replace: str = "", *,
             start_line: int | None = None, end_line: int | None = None,
             symbol: str = "") -> Result:
        """Search/replace edit (exact, with the engine's whitespace/fuzzy
        tolerance), or a LINES range / SYMBOL definition replacement."""
        lines = None
        if start_line is not None:
            lines = (int(start_line),
                     int(end_line if end_line is not None else start_line))
        return self.apply_edit(Edit(str(rel), search or "", replace or "",
                                    lines=lines, symbol=symbol or ""))

    def apply_edits_batch(self, edits: list[Edit]) -> list[Result]:
        """Apply a list of Edits all-or-nothing (WP-4105C staged transaction).

        In apply mode the batch goes through :func:`edits.apply_edits_atomic`
        (preflight, durable WAL snapshot, preimage hash re-check, rollback on
        write error). In overlay mode every edit is staged in memory and a
        failure anywhere rolls the whole batch back to its snapshot."""
        if self.apply:
            from .edits import apply_edits_atomic
            return apply_edits_atomic(
                self.root, list(edits),
                backup=bool(getattr(self.cfg, "backup", True)),
                backup_dir=self.backup_dir, cfg=self.cfg)
        snap = self.snapshot()
        applied: list[Result] = []
        for i, e in enumerate(edits):
            r = self.apply_edit(e)
            if r.ok:
                applied.append(r)
                continue
            self.rollback(snap)
            held = [Result(False, rr.file,
                           "not written: another edit in this batch failed; "
                           "fix it and resend ALL the edits")
                    for rr in applied]
            tail = [Result(False, e2.file,
                           "not written: another edit in this batch failed; "
                           "fix it and resend ALL the edits")
                    for e2 in edits[i + 1:]]
            return held + [r] + tail
        return applied

    def write(self, rel, content: str) -> Result:
        """Create a file, or replace one wholesale (the edit engine's fragment
        and wipe guards still apply to an overwrite)."""
        return self.apply_edit(Edit(str(rel), "", content or ""))

    def delete(self, rel) -> Result:
        return self.apply_edit(Edit(str(rel), "", "", delete=True))

    def _delete(self, rel: str, path: Path) -> Result:
        if path.is_dir():
            return Result(False, rel, "this path is a directory, not a file")
        try:
            text, style = self._current(rel, path)
        except WorkspaceError as exc:
            return Result(False, rel, str(exc))
        if text is None:
            return Result(True, rel, "deleted: nothing to delete (already absent)")
        err = self._commit(rel, path, text, style, None, style)
        if err:
            return Result(False, rel, err)
        note = (f"deleted ({len(text.splitlines())} line(s))"
                + ("; backed up first -- undo / `pedacito restore` brings it back"
                   if self.apply else " (staged in memory; the file on disk is "
                   "untouched)"))
        return Result(True, rel, note, _unified(rel, text, "", deleted=True),
                      text, "", _sha(text), deleted=True)

    # ----------------------------------------------------------- results
    def changed_files(self) -> list[str]:
        """Files whose final content differs from what the run started with
        (a file edited and then restored, or created and deleted, is not)."""
        with self._lock:
            return sorted(rel for rel, e in self._entries.items()
                          if e.original != e.current)

    def has_changes(self) -> bool:
        return bool(self.changed_files())

    def to_edit_results(self) -> list[Result]:
        """One cumulative `edits.Result` per changed file (before = the file
        as the run found it), shaped like `apply_edits(dry_run=True)` output so
        every review/diff consumer works unchanged."""
        out: list[Result] = []
        with self._lock:
            for rel in self.changed_files():
                e = self._entries[rel]
                before = e.original or ""
                suffix = "" if self.apply else " (dry run)"
                if e.current is None:
                    out.append(Result(
                        True, rel,
                        f"deleted ({len(before.splitlines())} line(s))" + suffix,
                        _unified(rel, before, "", deleted=True), before, "",
                        _sha(before), deleted=True))
                    continue
                after = e.current
                if e.original is None:
                    note = f"created with {len(after.splitlines())} line(s)"
                else:
                    note = f"applied {e.edits} edit(s)"
                out.append(Result(
                    True, rel, note + suffix, _unified(rel, before, after),
                    before, after, _sha(before), fuzzy=e.fuzzy))
        return out

    def to_file_diffs(self) -> list:
        """`review.FileDiff` rows for the changed files (--review / GUI)."""
        from . import review
        return review.diffs_from_results(self.to_edit_results())

    def diff_text(self, rel=None, max_chars: int = 0) -> str:
        """Unified diff of everything this run changed (or of one file)."""
        want = self.resolve(rel) if rel else ""
        parts = [r.diff for r in self.to_edit_results()
                 if r.diff and (not want or r.file == want)]
        text = "".join(p if p.endswith("\n") else p + "\n" for p in parts)
        if max_chars and len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n[diff truncated]"
        return text

    def line_counts(self) -> dict[str, tuple[int, int]]:
        """file -> (lines added, lines removed) for the changed files."""
        counts: dict[str, tuple[int, int]] = {}
        for r in self.to_edit_results():
            added = removed = 0
            for ln in r.diff.splitlines():
                if ln.startswith("+") and not ln.startswith("+++"):
                    added += 1
                elif ln.startswith("-") and not ln.startswith("---"):
                    removed += 1
            counts[r.file] = (added, removed)
        return counts

    # ------------------------------------------------- running commands
    @contextlib.contextmanager
    def exec_root(self):
        """The directory commands should run in: the real root when edits are
        applied, otherwise a temp mirror with the staged edits laid over a
        copy of the tracked project files (the real tree is never touched)."""
        if self.apply:
            yield self.root
            return
        with self._lock:
            mirror = self._sync_mirror()
        yield mirror

    def scrub_paths(self, text: str) -> str:
        """Show the real project path instead of the temp mirror's."""
        if self._mirror is not None and text:
            return text.replace(str(self._mirror), str(self.root))
        return text

    def _sync_mirror(self) -> Path:
        from . import checkpoints
        if self._mirror is None:
            mirror = Path(tempfile.mkdtemp(prefix="pedacito-overlay-")).resolve()
            self._mirror = mirror
            self._finalizer = weakref.finalize(
                self, shutil.rmtree, str(mirror), True)
            _register_mirror(self)
            total = 0
            try:
                for rel, src in checkpoints.iter_workspace_files(self.root):
                    if security.is_sensitive_path(rel):
                        continue
                    total += src.stat().st_size
                    if total > MAX_MIRROR_BYTES:
                        raise WorkspaceError(
                            "this project is too large to mirror for a "
                            "dry-run command; apply the edits first (or run "
                            "with --apply) to run tests against them")
                    dest = mirror / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest)
                deps = self.root / "node_modules"
                if deps.is_dir() and not (mirror / "node_modules").exists():
                    _link_node_modules(mirror / "node_modules", deps)
            except BaseException:
                self.close()
                raise
            self._mirror_rev = -1
            self._mirror_touched = set()
        if self._mirror_rev == self.revision:
            return self._mirror
        mirror = self._mirror
        for rel in sorted(self._mirror_touched - set(self._entries)):
            if security.is_sensitive_path(rel):
                continue
            # Rolled back since the last sync: put the real file back.
            dest = mirror / rel
            src = self.root / rel
            if src.is_file():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
            elif dest.exists():
                dest.unlink()
        for rel, entry in self._entries.items():
            if security.is_sensitive_path(rel):
                continue                     # never mirror secret bytes
            dest = mirror / rel
            if entry.current is None:
                if dest.exists():
                    dest.unlink()
            else:
                fileio.write(dest, entry.current, entry.style)
        self._mirror_touched = set(self._entries)
        self._mirror_rev = self.revision
        return mirror

    def close(self) -> None:
        """Remove the dry-run mirror (idempotent; it is rebuilt on demand)."""
        _unregister_mirror(self)
        with self._lock:
            if self._finalizer is not None:
                self._finalizer()
                self._finalizer = None
            self._mirror = None
            self._mirror_rev = -1
            self._mirror_touched = set()

    def __enter__(self) -> "VirtualWorkspace":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
