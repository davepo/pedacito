"""Time-travel checkpoints: snapshots, per-file diffs, restore and branching.

Every Task, Auto step and New scaffold records a checkpoint under
``.pedacito/checkpoints/``.  A checkpoint is a full snapshot of the tracked
workspace source files plus one entry in ``index.json``::

    .pedacito/checkpoints/
        index.json                 metadata, the active head and the sequence
        snapshots/<cp_id>/...      the workspace tree at that checkpoint

The ``parent_id`` on every entry forms the tree.  Restoring an earlier
checkpoint makes it the active head, so the next checkpoint is recorded as its
child -- that is how a user branches a new run from an old checkpoint without
losing the history they already had.

Only project files are snapshotted.  State (``.pedacito``), VCS data
(``.git``), dependency/build trees and caches are skipped, as are binary
artifacts above 2 MB.  Symlinks are never followed, and every restore goes
through :mod:`pedacito.security` so a hand-edited snapshot cannot write outside
the project root.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from . import fileio, security
from .index import state_dir

CHECKPOINT_DIR = "checkpoints"
CHECKPOINTS_DIR = CHECKPOINT_DIR        # compatibility alias
INDEX_FILE = "index.json"
SNAPSHOT_DIR = "snapshots"
SNAPSHOTS_DIR = SNAPSHOT_DIR            # compatibility alias
VERSION = 1

# Directories that never belong in a source snapshot.  The list starts with
# the six the checkpoint contract names and adds the usual dependency, cache
# and build trees a project may carry.
EXCLUDED_DIRS = frozenset({
    ".git", ".hg", ".svn",
    ".pedacito",
    ".venv", "venv", "env", ".venvs", "site-packages",
    "node_modules", "__pycache__",
    "dist", "build",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".nox",
    ".idea", ".vscode", ".next", ".nuxt", ".output", ".turbo",
    ".gradle", "target", "vendor", "Pods", "Carthage",
})

BINARY_EXTS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".tif",
    ".tiff", ".pdf", ".zip", ".gz", ".bz2", ".xz", ".7z", ".rar", ".tar",
    ".exe", ".dll", ".so", ".dylib", ".a", ".o", ".obj", ".pyc", ".pyo",
    ".woff", ".woff2", ".ttf", ".otf", ".eot", ".mp3", ".mp4", ".webm",
    ".ogg", ".wav", ".mov", ".avi", ".mkv", ".sqlite", ".db", ".wasm",
})

MAX_BINARY_BYTES = 2_000_000       # "binary artifacts over 2MB" are skipped
MAX_TEXT_BYTES = 8_000_000         # sanity cap for very large text files
MAX_LABEL = 200

MODES = ("task", "auto", "new", "manual", "restore")

_LOCK = threading.RLock()


class CheckpointError(ValueError):
    """A checkpoint id, path or request is invalid."""


# --- index --------------------------------------------------------------------

def checkpoint_root(root) -> Path:
    return state_dir(Path(root), CHECKPOINT_DIR)


def index_path(root) -> Path:
    return checkpoint_root(root) / INDEX_FILE


def snapshot_dir(root, checkpoint_id: str) -> Path:
    cid = str(checkpoint_id or "")
    # A checkpoint id becomes a directory name under `.pedacito/checkpoints/
    # snapshots/`; anything with a separator or a parent component could walk
    # outside that tree, so only the ids the index actually generates are
    # accepted (`cp_0001_deadbe` -- letters, digits, `-`, `_`).
    if (not cid or "/" in cid or "\\" in cid or ".." in cid
            or not re.fullmatch(r"[A-Za-z0-9_-]+", cid)):
        raise CheckpointError(f"invalid checkpoint id {checkpoint_id!r}")
    return checkpoint_root(root) / SNAPSHOT_DIR / cid


def _empty_index() -> dict:
    return {"version": VERSION, "next_seq": 1, "head_id": None,
            "checkpoints": []}


def _load_index(root) -> dict:
    path = index_path(root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _empty_index()
    if not isinstance(data, dict):
        return _empty_index()
    out = _empty_index()
    cps = data.get("checkpoints")
    out["checkpoints"] = [c for c in cps if isinstance(c, dict)] \
        if isinstance(cps, list) else []
    try:
        out["next_seq"] = max(1, int(data.get("next_seq") or 1))
    except (TypeError, ValueError):
        out["next_seq"] = 1
    out["head_id"] = str(data.get("head_id") or "") or None
    # Repair a missing head: the newest checkpoint is the natural one.
    if out["head_id"] is None and out["checkpoints"]:
        out["head_id"] = str(out["checkpoints"][-1].get("id") or "") or None
    return out


def _save_index(root, data: dict) -> None:
    path = index_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": VERSION,
        "next_seq": int(data.get("next_seq") or 1),
        "head_id": data.get("head_id"),
        "checkpoints": list(data.get("checkpoints") or []),
    }
    fileio.write_bytes(
        path,
        (json.dumps(payload, indent=1, ensure_ascii=False) + "\n").encode(
            "utf-8"))


def _find_entry(data: dict, checkpoint_id: str) -> dict | None:
    for entry in data.get("checkpoints") or []:
        if str(entry.get("id")) == str(checkpoint_id):
            return entry
    return None


def resolve_id(root, checkpoint_id: str) -> str:
    """Resolve an exact id or a unique prefix to a full checkpoint id."""
    wanted = str(checkpoint_id or "").strip()
    if not wanted:
        raise CheckpointError("A checkpoint id is required.")
    data = _load_index(root)
    exact = _find_entry(data, wanted)
    if exact is not None:
        return str(exact["id"])
    matches = [str(c.get("id")) for c in data["checkpoints"]
               if str(c.get("id", "")).startswith(wanted)]
    if not matches:
        raise CheckpointError(f"No checkpoint matches {wanted!r}.")
    if len(matches) > 1:
        raise CheckpointError(
            f"{wanted!r} is ambiguous ({', '.join(matches[:5])}).")
    return matches[0]


def head_id(root) -> str:
    return str(_load_index(root).get("head_id") or "")


def head_checkpoint(root) -> dict | None:
    data = _load_index(root)
    entry = _find_entry(data, data.get("head_id") or "")
    if entry is None and data["checkpoints"]:
        entry = data["checkpoints"][-1]
    return dict(entry) if entry is not None else None


def active_head(root) -> dict | None:
    """The checkpoint a new run would branch from (alias of head_checkpoint)."""
    return head_checkpoint(root)


def active_head_id(root) -> str:
    return head_id(root)


def _set_head(root, checkpoint_id: str) -> None:
    with _LOCK:
        data = _load_index(root)
        data["head_id"] = checkpoint_id
        _save_index(root, data)


# --- workspace files ----------------------------------------------------------

def _is_binary_bytes(chunk: bytes) -> bool:
    return b"\x00" in chunk[:4096]


def _is_binary(path: Path) -> bool:
    if path.suffix.lower() in BINARY_EXTS:
        return True
    try:
        with path.open("rb") as fh:
            return _is_binary_bytes(fh.read(4096))
    except OSError:
        return True


def _trackable(path: Path) -> bool:
    """True when `path` is a regular source/asset file worth snapshotting."""
    try:
        if path.is_symlink() or security.is_junction_or_reparse(path) \
                or not path.is_file():
            return False
        size = path.stat().st_size
    except OSError:
        return False
    binary = _is_binary(path)
    if binary:
        return size <= MAX_BINARY_BYTES
    return size <= MAX_TEXT_BYTES


def iter_workspace_files(root) -> list[tuple[str, Path]]:
    """Every snapshot-tracked file under `root`, as (relative POSIX, path)."""
    root = Path(root).resolve()
    out: list[tuple[str, Path]] = []
    if not root.is_dir():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        dirnames[:] = sorted(
            name for name in dirnames
            if name not in EXCLUDED_DIRS and not (d / name).is_symlink()
            and not security.is_junction_or_reparse(d / name))
        for name in sorted(filenames):
            p = d / name
            try:
                rel = p.relative_to(root).as_posix()
            except ValueError:
                continue
            # Defense in depth: secret/credential material is never opened,
            # hashed or copied into `.pedacito/checkpoints/snapshots/<id>/`
            # (or anywhere else the checkpoint store can restore from).
            if security.is_sensitive_path(rel):
                continue
            if not _trackable(p):
                continue
            out.append((rel, p))
    return out


def _file_hash(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return ""


def _snapshot_map(path: Path) -> dict[str, Path]:
    """Files in one snapshot directory, keyed by project-relative path."""
    out: dict[str, Path] = {}
    if not path.is_dir():
        return out
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames.sort()
        d = Path(dirpath)
        for name in sorted(filenames):
            p = d / name
            if p.is_symlink() or not p.is_file():
                continue
            out[p.relative_to(path).as_posix()] = p
    return out


def _hash_map(files: dict[str, Path]) -> dict[str, str]:
    return {rel: _file_hash(p) for rel, p in files.items()}


def _changed_paths(old: dict[str, Path], new: dict[str, Path]) -> list[str]:
    old_h = _hash_map(old)
    new_h = _hash_map(new)
    changed = []
    for rel in sorted(set(old_h) | set(new_h)):
        if old_h.get(rel) != new_h.get(rel):
            changed.append(rel)
    return changed


# A repair round labels its checkpoint with the fix prompt's first line, which
# is boilerplate with no identifying detail ("Your last change failed this
# check:"). Fold that bare prefix to a clean, generic label so the checkpoint
# list does not show a wall of identical, uninformative titles.
_BARE_LABEL_PREFIX = "your last change failed this check"
_BARE_LABELS = {
    _BARE_LABEL_PREFIX: "repair",
    _BARE_LABEL_PREFIX + ":": "repair",
}


def _clean_label(label) -> str:
    text = " ".join(str(label or "").split())
    if not text:
        text = "checkpoint"
    low = text.lower().rstrip(":")
    if low == _BARE_LABEL_PREFIX or low.startswith(_BARE_LABEL_PREFIX + ":"):
        text = "repair"
    return text[:MAX_LABEL]


def _now_iso() -> str:
    return (datetime.now(timezone.utc).isoformat(timespec="seconds")
            .replace("+00:00", "Z"))


def _new_id(seq: int, label: str) -> str:
    seed = f"{time.time_ns()}:{seq}:{label}".encode("utf-8", "replace")
    return f"cp_{int(seq):04d}_{hashlib.sha1(seed).hexdigest()[:6]}"


# --- create / list ------------------------------------------------------------

def create_checkpoint(root, label: str, mode: str = "task",
                      parent_id: str | None = None,
                      wal_id: str = "") -> dict:
    """Snapshot the tracked workspace and append one checkpoint.

    `parent_id` defaults to the active head, which makes the normal case a
    linear history and a create-after-restore a branch.  The new checkpoint
    becomes the active head.

    `wal_id` binds the checkpoint to the WAL (task journal) that recorded it;
    recovery verifies this exact association before restoring, so a checkpoint
    created by another task can never be substituted.
    """
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if mode not in MODES:
        raise CheckpointError(
            f"Unknown checkpoint mode {mode!r}; choose from {', '.join(MODES)}.")
    label = _clean_label(label)

    with _LOCK:
        data = _load_index(root)
        parent = str(parent_id) if parent_id else None
        if parent is not None and _find_entry(data, parent) is None:
            raise CheckpointError(f"No parent checkpoint {parent!r}.")
        if parent is None:
            parent = data.get("head_id")

        seq = int(data.get("next_seq") or 1)
        cp_id = _new_id(seq, label)
        while _find_entry(data, cp_id) is not None \
                or snapshot_dir(root, cp_id).exists():
            seq += 1
            cp_id = _new_id(seq, label)

        target = snapshot_dir(root, cp_id)
        tmp = target.with_name(target.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        current = dict(iter_workspace_files(root))
        copied: dict[str, Path] = {}
        for rel, src in current.items():
            dest = tmp / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                fileio.copy_file_atomic(src, dest)
            except OSError:
                continue
            copied[rel] = dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(target, ignore_errors=True)
        os.replace(tmp, target)
        # The temporary directory no longer exists under its old name; point
        # every record at the final snapshot so the hash comparison reads the
        # copied bytes instead of empty paths.
        copied = {rel: target / rel for rel in copied}

        base_dir = snapshot_dir(root, parent) if parent else None
        base = _snapshot_map(base_dir) if base_dir is not None else {}
        changed = _changed_paths(base, copied)

        entry = {
            "id": cp_id,
            "parent_id": parent,
            "timestamp": _now_iso(),
            "mode": mode,
            "label": label,
            "wal_id": str(wal_id or ""),
            "files_changed": changed,
            # BUG-14: `file_count` used to mean "files in the snapshot", so a
            # 14-file project with 3 changed files looked like "14 files
            # changed". Keep `file_count` as the changed count and name both
            # totals explicitly.
            "file_count": len(changed),
            "changed_file_count": len(changed),
            "snapshot_file_count": len(copied),
        }
        data["checkpoints"].append(entry)
        data["head_id"] = cp_id
        data["next_seq"] = seq + 1
        _save_index(root, data)
        return dict(entry)


def ensure_baseline(root, label: str = "baseline",
                    mode: str = "task") -> dict | None:
    """Record the pre-run baseline only when no checkpoint exists yet.

    Returns the baseline entry (existing or newly created), or None when the
    workspace cannot be snapshotted.
    """
    try:
        with _LOCK:
            data = _load_index(root)
            if data["checkpoints"]:
                entry = _find_entry(data, data.get("head_id") or "")
                return dict(entry) if entry is not None else None
        return create_checkpoint(root, label, mode=mode)
    except (OSError, CheckpointError):
        return None


def ensure_pre_run_checkpoint(root, label: str = "pre-run",
                              mode: str = "task", wal_id: str = "") -> dict | None:
    """Record a pre-run checkpoint that ALWAYS matches the live workspace.

    Unlike :func:`ensure_baseline` (which only snapshots when no checkpoint
    exists yet), this re-snapshots whenever the tracked files have drifted
    from the active head since it was recorded -- a manual edit, a
    ``git commit`` or a ``pedacito rename`` between runs. The returned
    checkpoint (existing or freshly created) is then the exact ``parent_id``
    for the task's post-run checkpoint, so Undo restores the true pre-task
    state instead of an older one.

    With ``wal_id`` (the owning WAL's UUID), a *fresh* checkpoint bound to that
    UUID is always recorded -- the historical "reuse the unchanged head" fast
    path cannot bind a shared head to a new task, and rollback must be able to
    verify the exact UUID association. Without ``wal_id`` (legacy callers) the
    fast path is preserved.

    Returns None when the workspace cannot be snapshotted.
    """
    try:
        with _LOCK:
            if not wal_id:
                data = _load_index(root)
                if data["checkpoints"]:
                    head = _find_entry(data, data.get("head_id") or "")
                    if head is None:
                        head = data["checkpoints"][-1]
                    head_dir = snapshot_dir(root, str(head.get("id") or ""))
                    if head_dir.is_dir():
                        live = dict(iter_workspace_files(root))
                        if not _changed_paths(_snapshot_map(head_dir), live):
                            return dict(head)
            return create_checkpoint(root, label, mode=mode, wal_id=wal_id)
    except (OSError, CheckpointError):
        return None


class CheckpointList(list):
    """The checkpoint list, carrying the active head as an attribute.

    The list itself is chronological; every entry also gains computed
    `head`, `depth` and `children` fields so a UI can draw the tree without
    a second round trip.
    """

    @property
    def head_id(self) -> str:
        return str(getattr(self, "_head_id", "") or "")

    @property
    def active_head_id(self) -> str:
        return self.head_id


def list_checkpoints(root) -> CheckpointList:
    """Chronological checkpoints with tree annotations and the active head."""
    data = _load_index(root)
    entries = [dict(c) for c in data["checkpoints"]]
    by_id = {str(c.get("id")): c for c in entries}
    head = str(data.get("head_id") or "")

    def depth(entry: dict) -> int:
        seen = set()
        n = 0
        parent = entry.get("parent_id")
        while parent and parent in by_id and parent not in seen:
            seen.add(parent)
            n += 1
            parent = by_id[parent].get("parent_id")
        return n

    for entry in entries:
        cid = str(entry.get("id"))
        entry["head"] = cid == head
        entry["is_head"] = cid == head
        entry["depth"] = depth(entry)
        entry["children"] = [str(c.get("id")) for c in entries
                             if str(c.get("parent_id") or "") == cid]

    out = CheckpointList(entries)
    out._head_id = head
    return out


# --- diff ---------------------------------------------------------------------

def _read_text_file(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if _is_binary_bytes(data):
        return None
    return data.decode("utf-8", errors="replace")


def _diff_pair(rel: str, before: Path | None, after: Path | None) -> dict:
    old_text = _read_text_file(before) if before is not None else ""
    new_text = _read_text_file(after) if after is not None else ""
    old_missing = before is not None and old_text is None
    new_missing = after is not None and new_text is None
    binary = old_missing or new_missing
    if old_text is None:
        old_text = ""
    if new_text is None:
        new_text = ""
    if binary:
        diff = ""
        added = removed = 0
    else:
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)
        diff = "".join(difflib.unified_diff(
            old_lines, new_lines, fromfile=f"a/{rel}", tofile=f"b/{rel}",
            n=3))
        added = sum(1 for ln in diff.splitlines()
                    if ln.startswith("+") and not ln.startswith("+++"))
        removed = sum(1 for ln in diff.splitlines()
                      if ln.startswith("-") and not ln.startswith("---"))
    if before is None:
        status = "added"
    elif after is None:
        status = "deleted"
    else:
        status = "modified"
    return {"path": rel, "status": status, "diff": diff, "binary": binary,
            "added": added, "removed": removed}


def _checkpoint_entry(root, checkpoint_id: str) -> tuple[dict, dict]:
    data = _load_index(root)
    cp_id = resolve_id(root, checkpoint_id)
    entry = _find_entry(data, cp_id)
    if entry is None:
        raise CheckpointError(f"No checkpoint {checkpoint_id!r}.")
    return data, entry


def diff_checkpoint(root, checkpoint_id: str,
                    against: str = "workspace") -> dict:
    """Unified diffs between a checkpoint and its parent or the live tree.

    `against="parent"` diffs parent -> checkpoint (what that run changed).
    `against="workspace"` diffs checkpoint -> current working tree (what is
    different right now).  Returns per-file records plus a combined diff.
    """
    root = Path(root).resolve()
    against = str(against or "workspace").lower()
    if against not in ("parent", "workspace"):
        raise CheckpointError(
            "against must be 'parent' or 'workspace'.")
    data, entry = _checkpoint_entry(root, checkpoint_id)
    cp_id = str(entry["id"])
    snap = _snapshot_map(snapshot_dir(root, cp_id))

    if against == "parent":
        parent_id = str(entry.get("parent_id") or "")
        base = (_snapshot_map(snapshot_dir(root, parent_id))
                if parent_id else {})
        # Parent -> checkpoint: files present now were added, files only in
        # the parent were deleted.
        before, after = base, snap
    else:
        base = snap
        after = dict(iter_workspace_files(root))
        before = base

    changed = _changed_paths(before, after)
    files = []
    for rel in changed:
        rec = _diff_pair(rel, before.get(rel), after.get(rel))
        # The status names the move parent->checkpoint, or checkpoint->live.
        files.append(rec)

    combined = "\n".join(f["diff"] for f in files if f["diff"])
    return {
        "id": cp_id,
        "label": entry.get("label", ""),
        "against": against,
        "base_id": str(entry.get("parent_id") or "")
        if against == "parent" else cp_id,
        "files": files,
        "file_diffs": {f["path"]: f["diff"] for f in files},
        "diffs": {f["path"]: f["diff"] for f in files},
        "changed_files": [f["path"] for f in files],
        "changed": [f["path"] for f in files],
        "paths": [f["path"] for f in files],
        "diff": combined,
        "added": sum(int(f["added"]) for f in files),
        "removed": sum(int(f["removed"]) for f in files),
        "count": len(files),
    }


# --- restore ------------------------------------------------------------------

def _safe_relative(root: Path, rel) -> str:
    """Normalise `rel` and prove it stays inside `root`."""
    text = str(rel or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    if not text:
        raise CheckpointError("An empty file path cannot be restored.")
    candidate = Path(text)
    if candidate.is_absolute():
        raise CheckpointError(f"Refusing an absolute path: {rel!r}")
    try:
        security.resolve_safe_path(root, text)
    except (security.SecurityError, OSError, ValueError, RuntimeError) as exc:
        raise CheckpointError(f"Refusing unsafe path {rel!r}: {exc}") from exc
    return Path(text).as_posix()


def _prune_empty_dirs(root: Path, start: Path) -> None:
    parent = start
    resolved_root = root.resolve()
    # Path containment, not a string prefix: `/proj` must not also match the
    # sibling `/project/...` (str.startswith would), and a parent reached
    # through a symlink or `..` must not be climbed past the real root.
    while parent != root and parent.resolve().is_relative_to(resolved_root):
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def _delete_tracked(root: Path, rel: str) -> bool:
    # A restore must never unlink secret/credential material, even by accident.
    if security.is_sensitive_path(rel):
        return False
    p = root / rel
    if p.is_symlink() or security.is_junction_or_reparse(p):
        try:
            p.unlink()
            return True
        except OSError:
            return False
    if not p.is_file():
        return False
    try:
        p.unlink()
    except OSError:
        return False
    _prune_empty_dirs(root, p.parent)
    return True


def restore_checkpoint(root, checkpoint_id: str, files=None) -> dict:
    """Restore selected files, or the whole workspace, from a checkpoint.

    Any dirty changes are captured as a ``restore`` checkpoint first, so the
    one-click undo is itself undoable.  The restored checkpoint becomes the
    active head: the next run branches from it.
    """
    root = Path(root).resolve()
    # `_LOCK` is held across the whole operation (read snapshot, capture the
    # safety checkpoint, restore files, set the head) so a concurrent
    # checkpoint create/restore cannot interleave a snapshot write with this
    # restore's read-modify-write of the index. `_LOCK` is reentrant, so the
    # nested `create_checkpoint`/`_set_head` acquisitions are safe.
    with _LOCK:
        data, entry = _checkpoint_entry(root, checkpoint_id)
        cp_id = str(entry["id"])
        snap = _snapshot_map(snapshot_dir(root, cp_id))

        if files is not None:
            if isinstance(files, (str, Path)):
                files = [files]
            selected = []
            for item in files:
                if isinstance(item, dict):
                    item = item.get("path") or item.get("rel") or item.get("file")
                selected.append(_safe_relative(root, item))
            selected = sorted(set(selected))
            target = {rel: snap[rel] for rel in selected if rel in snap}
            current = {rel: root / rel for rel in selected
                       if (root / rel).is_file()
                       and not (root / rel).is_symlink()}
            affected = selected
        else:
            target = dict(snap)
            current = dict(iter_workspace_files(root))
            affected = sorted(set(target) | set(current))

        # Dirty only when an affected path differs from the checkpoint.  A
        # clean restore (same bytes) needs no safety snapshot.
        target_hashes = {rel: _file_hash(p) for rel, p in target.items()}
        current_hashes = {rel: _file_hash(p) for rel, p in current.items()}
        dirty = any(target_hashes.get(rel) != current_hashes.get(rel)
                    for rel in affected)

        safety = None
        if dirty:
            safety = create_checkpoint(
                root,
                f"Before restoring {entry.get('label') or cp_id}",
                mode="restore",
                parent_id=str(data.get("head_id")
                              or entry.get("parent_id") or "")
                or None)

        restored: list[str] = []
        deleted: list[str] = []
        skipped: list[str] = []
        identical: list[str] = []

        if files is not None:
            # Explicit files: copy when the checkpoint has them, remove when
            # the checkpoint says they did not exist yet.
            for rel in affected:
                dest = root / rel
                if rel in target:
                    if target_hashes.get(rel) == current_hashes.get(rel) \
                            and current_hashes.get(rel) is not None:
                        # The bytes on disk already match the checkpoint; the
                        # copy below would be a no-op, so say so honestly.
                        identical.append(rel)
                    try:
                        fileio.copy_file_atomic(target[rel], dest)
                        restored.append(rel)
                    except OSError:
                        skipped.append(rel)
                elif _delete_tracked(root, rel):
                    deleted.append(rel)
        else:
            for rel, src in target.items():
                try:
                    _safe_relative(root, rel)
                except CheckpointError:
                    skipped.append(rel)
                    continue
                dest = root / rel
                if target_hashes.get(rel) == current_hashes.get(rel) \
                        and current_hashes.get(rel) is not None:
                    identical.append(rel)
                try:
                    fileio.copy_file_atomic(src, dest)
                    restored.append(rel)
                except OSError:
                    skipped.append(rel)
            for rel in current:
                if rel in target:
                    continue
                try:
                    _safe_relative(root, rel)
                except CheckpointError:
                    skipped.append(rel)
                    continue
                if _delete_tracked(root, rel):
                    deleted.append(rel)

        _set_head(root, cp_id)
        return {
            "ok": True,
            "checkpoint": cp_id,
            "label": entry.get("label", ""),
            "head_id": cp_id,
            "restored": sorted(restored),
            "restored_files": sorted(restored),
            "deleted": sorted(deleted),
            "deleted_files": sorted(deleted),
            "skipped": sorted(set(skipped)),
            "identical": sorted(set(identical)),
            "files": len(restored),
            "safety": safety,
            "safety_id": str(safety.get("id")) if safety else "",
        }


def _undo_clobber_warning(root, head_entry: dict) -> str:
    """A non-empty warning when reverting to the pre-run state would overwrite
    workspace changes the last checkpoint's own delta does not cover -- a
    manual edit, a ``pedacito rename`` or a ``git commit`` made after the run.
    Returns '' when the live tree still matches the checkpoint (or drifted
    only in files the run itself changed, which undo is meant to revert)."""
    head_id = str(head_entry.get("id") or "")
    own = set(head_entry.get("files_changed") or [])
    head_dir = snapshot_dir(root, head_id)
    if not head_dir.is_dir():
        return ""
    try:
        live = dict(iter_workspace_files(root))
        changed = _changed_paths(_snapshot_map(head_dir), live)
    except OSError:
        return ""
    external = [rel for rel in changed if rel not in own]
    if not external:
        return ""
    listed = ", ".join(sorted(external)[:5])
    more = f" (+{len(external) - 5} more)" if len(external) > 5 else ""
    return (f"the working tree changed since that run in file(s) it did not "
            f"touch ({listed}{more}) -- a manual edit, rename or git commit. "
            "Undo would overwrite them.")


def undo_last_run(root, force: bool = False) -> dict:
    """Restore the state immediately before the active checkpoint's run.

    The active checkpoint's parent is the pre-run snapshot.  With a single
    checkpoint the checkpoint itself is restored (there is nothing earlier).
    Returns ``ok=False`` when no checkpoints exist, and refuses (with a
    ``clobber`` detail) when reverting would overwrite workspace changes the
    run did not make, unless ``force`` is set.
    """
    data = _load_index(root)
    if not data["checkpoints"]:
        return {"ok": False, "detail": "No checkpoints have been recorded yet."}
    head = _find_entry(data, data.get("head_id") or "")
    if head is None:
        head = data["checkpoints"][-1]
    warning = _undo_clobber_warning(root, dict(head))
    if warning and not force:
        return {
            "ok": False,
            "detail": warning,
            "clobber": True,
            "hint": "pass --force to revert anyway, or `pedacito checkout` to "
                    "restore a chosen checkpoint.",
        }
    target = str(head.get("parent_id") or head.get("id"))
    result = restore_checkpoint(root, target)
    result["undone_to"] = target
    result["from"] = str(head.get("id") or "")
    return result


# --- presentation -------------------------------------------------------------

def _render_timestamp(ts) -> str:
    """A checkpoint's stored UTC ISO timestamp, shown in local time with its
    offset (the stored JSON keeps the UTC `Z` form for compatibility)."""
    try:
        dt = datetime.fromisoformat(str(ts or "").replace("Z", "+00:00"))
        return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    except (ValueError, TypeError):
        return str(ts or "")


def render_list(root, limit: int = 0) -> str:
    """Human-readable checkpoint tree for the CLI and the TUI."""
    items = list_checkpoints(root)
    if not items:
        return ("No checkpoints yet. One is recorded before and after every "
                "Task, Auto step and New scaffold.")
    if limit and limit > 0:
        items = items[-limit:]
    lines = [f"Checkpoints for {root} (head: {head_id(root) or 'none'})"]
    for cp in items:
        mark = "\u25cf" if cp.get("head") else "\u25cb"
        parent = f" <- {cp.get('parent_id')}" if cp.get("parent_id") else ""
        n = len(cp.get("files_changed") or [])
        lines.append(
            f"  {mark} {cp.get('id')}  [{cp.get('mode')}] "
            f"{_render_timestamp(cp.get('timestamp'))}  {n} file(s){parent}")
        lines.append(f"      {cp.get('label')}")
    return "\n".join(lines)


def snapshot_size(root, checkpoint_id: str) -> int:
    """Bytes held by one checkpoint snapshot (useful for tests and the UI)."""
    cp_id = resolve_id(root, checkpoint_id)
    total = 0
    for p in _snapshot_map(snapshot_dir(root, cp_id)).values():
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total
