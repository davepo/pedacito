"""Green-ratchet step checkpoints for the autonomous loop (TECHNIQUES.md #20).

In Auto mode the loop runs task after task, each building on the previous
work. A failure late in the sequence used to leave the workspace holding the
broken step AND every later step's now-invalid dependencies — the LLM repair
microloop had already burned its rounds against the wrong baseline, and the
next task inherited a poisoned tree.

The green ratchet makes Auto mode monotonic: before step k edits anything,
every source file it might touch is snapshotted into
``.pedacito/checkpoints/step_<k>/``; when step k verifies green its snapshot
is kept as the restore point (the "green frontier"); when step k fails after
all repair rounds are exhausted, the files the step created are deleted and
the files it modified are restored from the snapshot, so the workspace is
provably back at the last verified-green state and the next task builds on
real work instead of wreckage.

Zero LLM tokens — pure file I/O — and every operation is bounded and
best-effort: a step that touched nothing rolls back to nothing, and snapshot
failures never crash the loop.

Snapshots are HARD LINKS when the filesystem allows it (same-device `os.link`
— the common case), falling back to `copy2`. A hardlinked checkpoint costs
one directory entry and zero bytes, so keeping 25 steps of history is free.
The contract this relies on: nothing ever writes *through* a project file's
inode. Every Pedacito write goes through `fileio.write`, an atomic
temp-file + `os.replace` — which swaps the path onto a fresh inode and
leaves the checkpointed inode pristine — and `rollback` restores the same
way. An in-place `open(path, "wb")` would truncate every link sharing that
inode, i.e. silently rewrite history.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


# --- configuration defaults (overridable through Config fields) ------------
DEFAULT_MAX_FILE_BYTES = 1_000_000
DEFAULT_MAX_FILES = 1_500
DEFAULT_KEEP = 25

# Directories never snapshotted or restored: state, caches, and anything a
# build/test step is expected to regenerate (mirrors spec._SKIP_DIRS).
SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".pedacito",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "venv",
        "env",
        "node_modules",
        ".venvs",
        "site-packages",
        "dist",
        "build",
        ".next",
        ".nuxt",
        ".output",
        ".turbo",
        ".gradle",
        "target",
        "vendor",
        ".bundle",
        ".pod-config",
        "Pods",
        "Carthage",
        ".terragrunt-cache",
        ".terraform",
        ".serverless",
        ".angular",
        ".cache",
        ".turbo-cache",
    }
)

# Project-relative names never snapshotted: user settings that live at the
# project root (state dirs are already skipped wholesale).
_PROTECTED = {"pedacito.json"}

_SUFFIXES = frozenset(
    {
        ".py", ".pyi", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".vue", ".svelte",
        ".rb", ".rs", ".go", ".java", ".kt", ".kts", ".scala", ".groovy", ".c", ".h",
        ".cc", ".cpp", ".hh", ".hpp", ".cs", ".fs", ".vb", ".php", ".swift", ".m", ".mm",
        ".dart", ".lua", ".pl", ".pm", ".r", ".sh", ".bash", ".zsh", ".ps1", ".html",
        ".css", ".scss", ".sass", ".less", ".sql", ".toml", ".json", ".yaml", ".yml",
        ".md", ".txt", ".cfg", ".ini",
    }
)

_STEP_PREFIX = "step_"


def checkpoint_root(root: Path) -> Path:
    return root / ".pedacito" / "checkpoints"


def step_dir(root: Path, step: int) -> Path:
    return checkpoint_root(root) / f"{_STEP_PREFIX}{max(1, int(step))}"


def _pointer_path(root: Path) -> Path:
    return checkpoint_root(root) / "green_step.txt"


# --- snapshot ---------------------------------------------------------------
def _iter_candidate_files(root: Path, max_files: int):
    """Walk the project tree, skipping state directories and huge files.

    Yields ``(path, truncated)``: exactly one path per candidate, and the
    final item's flag says whether the walk stopped at the cap (so more
    candidates exist that were never seen)."""
    count = 0
    truncated = False
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir(), reverse=True)
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir():
                    if entry.name in SKIP_DIRS or entry.name.startswith('.'):
                        continue
                    stack.append(entry)
                    continue
                if not entry.is_file():
                    continue
            except OSError:
                continue
            if entry.name.startswith('.'):
                continue
            if entry.suffix.lower() not in _SUFFIXES:
                continue
            count += 1
            if count > max_files:
                yield (entry, True)
                return
            yield (entry, False)


def _link_or_copy(src: Path, dst: Path) -> None:
    """Snapshot one file: a hard link when the filesystem permits it (same
    device, links allowed) -- free, and byte-identical by construction --
    else a full `copy2` preserving mtime (the unmodified-file fast path in
    `rollback` compares timestamps)."""
    try:
        if dst.exists():
            dst.unlink()          # a link cannot be created over an entry
        os.link(src, dst)
    except (OSError, NotImplementedError):     # EXDEV/EPERM/FAT... 
        shutil.copy2(str(src), str(dst))


def begin(
    root: Path,
    step: int,
    task: str = "",
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_files: int = DEFAULT_MAX_FILES,
    skip: Iterable[str] = (),
) -> dict[str, Any]:
    """Snapshot the project before step *step* starts editing.

    *skip* holds project-relative paths never snapshotted (the plan file:
    loop harness state that keeps ticking boxes while steps run). Returns
    ``{"files": [...], "truncated": bool, "error": str|None}``. Every
    recorded path is relative to *root*. Failures are swallowed into the
    returned dict — the loop must keep running even if the ratchet can't.
    """
    step = max(1, int(step))
    skip_set = {str(s) for s in skip} | _PROTECTED
    info: dict[str, Any] = {"files": [], "truncated": False, "error": None}
    directory = step_dir(root, step)
    try:
        # A fresh attempt for this step discards any earlier snapshot of it
        # (e.g. the previous attempt's rollback deleted its files already).
        if directory.is_dir():
            shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        info["error"] = f"cannot create checkpoint directory: {exc}"
        return info

    manifest: dict[str, dict[str, Any]] = {}
    truncated = False
    for path, over_cap in _iter_candidate_files(root, max_files):
        if over_cap:
            truncated = True
            break
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if rel.as_posix() in skip_set:
            continue
        if any(part in SKIP_DIRS for part in rel.parts[:-1]):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > max_file_bytes:
            manifest[rel.as_posix()] = {"size": size, "skip": "too-large"}
            continue
        payload = directory / rel
        try:
            payload.parent.mkdir(parents=True, exist_ok=True)
            _link_or_copy(path, payload)
        except OSError:
            manifest[rel.as_posix()] = {"size": size, "skip": "unreadable"}
            continue
        manifest[rel.as_posix()] = {"size": size}
    try:
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "task": task[:200],
                    "snapshot": datetime.now().isoformat(timespec="seconds"),
                    "skip": sorted(skip_set),
                    "max_files": max_files,
                    "files": manifest,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass
    info["files"] = sorted(k for k, v in manifest.items() if "skip" not in v)
    info["truncated"] = truncated
    if truncated:
        info["error"] = f"checkpoint truncated at {max_files} files: rollback is partial"
    return info


def green(root: Path, step: int) -> None:
    """Mark step *step* verified-green: it becomes the rollback target."""
    try:
        checkpoint_root(root).mkdir(parents=True, exist_ok=True)
        _pointer_path(root).write_text(f"{max(1, int(step))}\n", encoding="utf-8")
    except OSError:
        pass


def frontier(root: Path) -> int:
    """The last step known green (0 = none)."""
    try:
        return int(_pointer_path(root).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0


# --- rollback ---------------------------------------------------------------
def _load_manifest(root: Path, step: int):
    path = step_dir(root, step) / "manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    files = data.get("files")
    if not isinstance(files, dict):
        return None
    skip = {str(s) for s in (data.get("skip") or [])}
    try:
        max_files = int(data.get("max_files") or DEFAULT_MAX_FILES)
    except (TypeError, ValueError):
        max_files = DEFAULT_MAX_FILES
    return files, skip, max_files


def rollback(root: Path, step: int) -> dict[str, Any]:
    """Undo step *step*: restore its modified files, delete its creations.

    Returns ``{"restored": [...], "deleted": [...], "skipped": [...],
    "error": str|None}`` with project-relative paths. A file the current tree
    holds that is NOT in the snapshot (and not on the snapshot's skip list)
    was created by the step, so it goes away with the rollback. The step's
    snapshot is consumed (deleted) so a later failure can't roll back past
    its own verified-green work.
    """
    result: dict[str, Any] = {"restored": [], "deleted": [], "skipped": [],
                              "error": None}
    step = max(1, int(step))
    loaded = _load_manifest(root, step)
    if loaded is None:
        result["error"] = f"no checkpoint for step {step}: cannot roll back"
        return result
    manifest, skip_set, max_files = loaded
    directory = step_dir(root, step)
    for rel, entry in manifest.items():
        target = root / rel
        if isinstance(entry, dict) and entry.get("skip"):
            result["skipped"].append(rel)
            continue
        snapshot_file = directory / rel
        if snapshot_file.is_file():
            try:
                # (a hardlinked or copy2'd) snapshot carries the original
                # mtime: a file still matching it byte-size AND timestamp was
                # never touched by the step, so re-restoring it would only
                # inflate the rollback report.
                if (target.is_file() and not target.is_symlink()
                        and target.stat().st_size == snapshot_file.stat().st_size
                        and target.stat().st_mtime_ns
                        == snapshot_file.stat().st_mtime_ns):
                    continue
            except OSError:
                pass
            try:
                if target.is_symlink():
                    target.unlink()      # never write THROUGH a replaced link
                target.parent.mkdir(parents=True, exist_ok=True)
                # restore by REPLACE, never by copy-into: `target` may share
                # its inode with an OLDER step's snapshot (begin hardlinked
                # it while unchanged); copy2 would truncate that shared inode
                # and rewrite the checkpointed history it stands for.
                tmp = target.parent / (target.name + ".pedacito-restore")
                shutil.copy2(str(snapshot_file), str(tmp))
                os.replace(tmp, target)
                result["restored"].append(rel)
            except OSError as exc:
                try:
                    (target.parent / (target.name + ".pedacito-restore")
                     ).unlink()
                except OSError:
                    pass
                result["skipped"].append(rel)
                result["error"] = f"could not restore {rel}: {exc}"
    # Delete what the step created: current candidates absent from the
    # snapshot (the same bounded walk begin() used, so the scope agrees).
    for path, _over_cap in _iter_candidate_files(root, max_files):
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if rel in manifest or rel in skip_set:
            continue
        try:
            if path.is_symlink() or not path.is_file():
                continue
            path.unlink()
            result["deleted"].append(rel)
        except OSError as exc:
            result["error"] = f"could not remove {rel}: {exc}"
    for rel in result["deleted"]:
        _prune_empty_parent(root, rel)
    shutil.rmtree(directory, ignore_errors=True)
    result["restored"].sort()
    result["deleted"].sort()
    result["skipped"].sort()
    return result


def _prune_empty_parent(root: Path, rel: str) -> None:
    """Remove directories the rollback just emptied (a step that created a
    package then failed must not litter the tree). Only walks UP from a
    deleted file, only while the directory is empty, and never past the
    project root or through a dot directory."""
    d = (root / rel).parent
    while d != root and d.is_dir() and not d.name.startswith("."):
        try:
            next(d.iterdir())
            return                 # not empty: stop climbing
        except StopIteration:
            pass
        except OSError:
            return
        try:
            d.rmdir()
        except OSError:
            return
        d = d.parent


def prune(root: Path, keep: int = DEFAULT_KEEP) -> int:
    """Delete green step snapshots older than the last *keep*, return count removed."""
    keep = max(0, int(keep))
    base = checkpoint_root(root)
    try:
        entries = [p for p in base.iterdir() if p.is_dir() and p.name.startswith(_STEP_PREFIX)]
    except OSError:
        return 0
    steps = []
    for p in entries:
        try:
            steps.append((int(p.name[len(_STEP_PREFIX):]), p))
        except ValueError:
            continue
    steps.sort()
    if keep == 0:
        return 0
    removed = 0
    while len(steps) > keep:
        _, oldest = steps.pop(0)
        shutil.rmtree(oldest, ignore_errors=True)
        removed += 1
    return removed
