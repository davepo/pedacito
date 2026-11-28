"""Checkpoint garbage collection under a project quota (§3.7 / WP-4105F).

`Config.checkpoint_quota_mb` caps the bytes held by the three checkpoint
stores: the tree snapshots under ``.pedacito/checkpoints/snapshots/``, the
green-ratchet step snapshots (``.pedacito/checkpoints/step_<k>/``) and the
pre-edit backups (``.pedacito/backups/``).

`plan_checkpoint_gc` measures those bytes WITHOUT following links, marks every
protected item (the active head and its ancestors, an in-flight WAL/recovery
checkpoint, the green frontier, safety snapshots and authored files), and
proposes deletions oldest-first. When the quota cannot be met without deleting
protected history it says so instead of doing it. `apply_checkpoint_gc` then
performs the proposed deletions under a cross-process project lock, re-validates
every target against the plan before touching it, and journals the result so a
re-run is idempotent. Nothing is force-deleted: a target that cannot be removed
cleanly is skipped and reported.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from . import checkpoints, fileio, ratchet, security, workspace
from .index import BACKUP_DIR, state_dir

# Where the last applied GC plan/result is journaled (observability and
# idempotent re-run). Best effort.
GC_JOURNAL = "gc_journal.json"

KIND_SNAPSHOT = "snapshot"
KIND_RATCHET = "ratchet"
KIND_BACKUP = "backup"


def quota_bytes_from_cfg(cfg) -> int:
    """``checkpoint_quota_mb`` in bytes (0 = unlimited, legacy)."""
    try:
        mb = int(getattr(cfg, "checkpoint_quota_mb", 0) or 0)
    except (TypeError, ValueError):
        mb = 0
    return max(0, mb) * 1024 * 1024


# ``checkpoints.count_regular_bytes`` sums a store's bytes without following
# symlinks/junctions/reparse points; aliased here for readability.
_count_regular_bytes = checkpoints.count_regular_bytes


def _protected_snapshot_ids(root: Path) -> dict[str, str]:
    """checkpoint id -> reason, for every tree snapshot that must never go."""
    protected: dict[str, str] = {}
    data = checkpoints._load_index(root)
    entries = {str(c.get("id") or ""): c for c in data.get("checkpoints") or []}
    # The active head and its whole ancestor chain are needed for restore.
    cur = str(data.get("head_id") or "")
    seen: set[str] = set()
    while cur and cur in entries and cur not in seen:
        protected[cur] = "head-or-ancestor"
        seen.add(cur)
        cur = str(entries[cur].get("parent_id") or "")
    # Safety snapshots (a restore-mode checkpoint) are never GC'd.
    for cid, c in entries.items():
        if str(c.get("mode") or "") == "restore":
            protected.setdefault(cid, "safety-snapshot")
    # An in-flight WAL / recovery owns its bound checkpoint and its pre-run
    # checkpoint; deleting either would break rollback of an interrupted task.
    try:
        from . import journal
        inflight = journal.find_incomplete_tasks(root)
    except Exception:
        inflight = []
    inflight_wals = {str(t.get("wal_id") or "") for t in inflight
                     if t.get("wal_id")}
    inflight_cps = {str(t.get("checkpoint_id") or "") for t in inflight
                    if t.get("checkpoint_id")}
    for cid, c in entries.items():
        if cid in inflight_cps:
            protected.setdefault(cid, "in-flight-wal")
        elif str(c.get("wal_id") or "") and str(c.get("wal_id")) in inflight_wals:
            protected.setdefault(cid, "in-flight-wal")
    return protected


def _snapshot_targets(root: Path, protected: dict[str, str]
                      ) -> tuple[list[dict], list[dict]]:
    """(eligible, protected) tree-snapshot targets, oldest (index) first."""
    data = checkpoints._load_index(root)
    eligible: list[dict] = []
    kept: list[dict] = []
    for c in data.get("checkpoints") or []:
        cid = str(c.get("id") or "")
        if not cid:
            continue
        sdir = checkpoints.snapshot_dir(root, cid)
        try:
            if not sdir.is_dir() or sdir.is_symlink() \
                    or security.is_junction_or_reparse(sdir):
                continue
        except OSError:
            continue
        rec = {"kind": KIND_SNAPSHOT, "id": cid,
               "rel": sdir.relative_to(root).as_posix(),
               "bytes": _count_regular_bytes(sdir)}
        if cid in protected:
            rec["reason"] = protected[cid]
            kept.append(rec)
        else:
            eligible.append(rec)
    return eligible, kept


def _ratchet_targets(root: Path) -> tuple[list[dict], list[dict]]:
    """(eligible, protected) green-ratchet step snapshots, oldest step first."""
    base = ratchet.checkpoint_root(root)
    frontier = ratchet.frontier(root)
    eligible: list[dict] = []
    kept: list[dict] = []
    try:
        entries = sorted(base.iterdir()) if base.is_dir() else []
    except OSError:
        return eligible, kept
    for p in entries:
        if not p.name.startswith(ratchet._STEP_PREFIX):
            continue
        try:
            step = int(p.name[len(ratchet._STEP_PREFIX):])
        except ValueError:
            continue
        try:
            if not p.is_dir() or p.is_symlink() \
                    or security.is_junction_or_reparse(p):
                continue
        except OSError:
            continue
        rec = {"kind": KIND_RATCHET, "step": step,
               "rel": p.relative_to(root).as_posix(),
               "bytes": _count_regular_bytes(p)}
        # The frontier and anything not yet verified-green are in use.
        if frontier and step >= frontier:
            rec["reason"] = "green-frontier-or-later"
            kept.append(rec)
        else:
            eligible.append(rec)
    return eligible, kept


def _backup_targets(root: Path) -> tuple[list[dict], list[dict]]:
    """(eligible, protected) pre-edit backup directories, oldest first."""
    base = state_dir(root, BACKUP_DIR)
    eligible: list[dict] = []
    try:
        entries = sorted(base.iterdir()) if base.is_dir() else []
    except OSError:
        return eligible, []
    for p in entries:
        try:
            if not p.is_dir() or p.is_symlink() \
                    or security.is_junction_or_reparse(p):
                continue
        except OSError:
            continue
        eligible.append({"kind": KIND_BACKUP,
                         "rel": p.relative_to(root).as_posix(),
                         "bytes": _count_regular_bytes(p)})
    return eligible, []


def plan_checkpoint_gc(root: Path, quota_bytes: int) -> dict[str, Any]:
    """Measure every snapshot/ratchet/backup byte and propose deletions.

    Returns a dict with ``quota_bytes``, ``total_bytes``, ``protected``,
    ``eligible`` (oldest first), ``to_delete`` (the oldest-first prefix that
    would bring the store under quota, or every eligible item when the quota
    cannot be met), ``satisfiable`` and a human ``reason``. The plan is a pure
    preview: nothing is deleted.
    """
    root = Path(root).resolve()
    quota = max(0, int(quota_bytes))
    protected_ids = _protected_snapshot_ids(root)
    snap_eligible, snap_kept = _snapshot_targets(root, protected_ids)
    rat_eligible, rat_kept = _ratchet_targets(root)
    bak_eligible, _ = _backup_targets(root)
    eligible = snap_eligible + rat_eligible + bak_eligible
    kept = snap_kept + rat_kept
    total = sum(int(t["bytes"]) for t in eligible) + \
        sum(int(k["bytes"]) for k in kept)

    plan: dict[str, Any] = {
        "quota_bytes": quota,
        "total_bytes": total,
        "protected": kept,
        "eligible": eligible,
        "to_delete": [],
        "satisfiable": True,
        "reason": "",
    }
    if quota == 0:
        plan["reason"] = "quota is 0 (unlimited); nothing to delete"
        return plan
    if total <= quota:
        return plan

    remaining = total
    for t in eligible:
        plan["to_delete"].append(t)
        remaining -= int(t["bytes"])
        if remaining <= quota:
            break
    plan["satisfiable"] = remaining <= quota
    if not plan["satisfiable"]:
        plan["reason"] = (
            f"the {quota} byte checkpoint quota cannot be met safely: "
            f"{remaining - quota} bytes would remain over quota even after "
            "deleting every eligible snapshot/ratchet/backup; protected "
            "history is never deleted to satisfy a quota")
    return plan


def _rel_kind(rel: str) -> str | None:
    """The GC kind of a project-relative path, or None when it is not a
    recognized (and therefore deletable) store path."""
    parts = Path(str(rel)).parts
    if len(parts) < 2 or parts[0] != ".pedacito":
        return None
    sub = parts[1]
    if sub == "checkpoints" and len(parts) == 3 \
            and parts[2].startswith(ratchet._STEP_PREFIX):
        return KIND_RATCHET
    if sub == "checkpoints" and len(parts) >= 4 and parts[2] == "snapshots":
        return KIND_SNAPSHOT
    if sub == BACKUP_DIR and len(parts) == 3:
        return KIND_BACKUP
    return None


def _validate_target(root: Path, t: dict, protected_ids: dict[str, str],
                     frontier: int) -> tuple[bool, str]:
    """Whether ``t`` still matches the plan and may be deleted now."""
    rel = t.get("rel")
    if not isinstance(rel, str):
        return False, "missing rel"
    kind = t.get("kind")
    if _rel_kind(rel) is None or _rel_kind(rel) != kind:
        return False, "unsafe or mismatched store path"
    path = root / rel
    try:
        if path.is_symlink() or security.is_junction_or_reparse(path):
            return False, "link-or-junction"
        resolved = path.resolve()
        if not resolved.is_relative_to((root / ".pedacito").resolve()):
            return False, "escapes .pedacito"
    except OSError:
        return False, "unresolvable"
    if kind == KIND_SNAPSHOT:
        cid = str(t.get("id") or "")
        if not cid:
            return False, "missing id"
        if cid in protected_ids:
            return False, "became protected"
        return (True, "") if path.is_dir() else (False, "already-absent")
    if kind == KIND_RATCHET:
        step = t.get("step")
        if step is None:
            return False, "missing step"
        if frontier and int(step) >= int(frontier):
            return False, "became frontier-or-later"
        return (True, "") if path.is_dir() else (False, "already-absent")
    # backup
    return (True, "") if path.is_dir() else (False, "already-absent")


def apply_checkpoint_gc(root: Path, plan: dict) -> dict[str, Any]:
    """Delete the plan's ``to_delete`` targets under a cross-process lock.

    Each target is re-validated against the live store before it is touched, so
    a target that became protected (or changed identity) since planning is
    skipped, never force-deleted. ``already-absent`` targets are idempotent
    no-ops. Returns ``{deleted, skipped, error}`` and journals the result.
    """
    root = Path(root).resolve()
    result: dict[str, Any] = {"deleted": [], "skipped": [], "error": None}
    targets = list((plan or {}).get("to_delete") or [])
    if not targets:
        return result
    try:
        with workspace.recovery_lock(root):
            protected_ids = _protected_snapshot_ids(root)
            frontier = ratchet.frontier(root)
            for t in targets:
                ok, reason = _validate_target(root, t, protected_ids, frontier)
                if not ok:
                    if reason == "already-absent":
                        continue               # idempotent re-run
                    result["skipped"].append(dict(t, reason=reason))
                    continue
                path = root / str(t["rel"])
                try:
                    if path.is_dir():
                        shutil.rmtree(path)
                    elif path.is_file():
                        path.unlink()
                    else:
                        continue
                    result["deleted"].append(dict(t))
                except OSError as exc:
                    result["skipped"].append(
                        dict(t, reason=f"could not remove: {exc}"))
    except workspace.ProjectBusy as exc:
        # Another live run/thread holds the project: refuse, never force.
        result["error"] = f"project busy: {exc}"
        return result
    _journal(root, result)
    return result


def _journal(root: Path, result: dict[str, Any]) -> None:
    """Record the last applied GC outcome (best effort, idempotent overwrite)."""
    try:
        import json
        payload = {
            "deleted": [t.get("rel") for t in result["deleted"]],
            "skipped": [t.get("rel") for t in result["skipped"]],
            "error": result["error"],
        }
        path = state_dir(root, GC_JOURNAL)
        fileio.write_bytes(path, (json.dumps(payload, indent=1) + "\n").encode(
            "utf-8"))
    except OSError:
        pass


def enforce_quota(root: Path, cfg) -> tuple[bool, str]:
    """Free eligible checkpoint bytes under the configured quota, then report.

    ``(satisfiable, reason)``: ``True`` when no quota is set, the store is
    under quota, or deletion brought it under quota; ``False`` when the quota
    cannot be met without deleting protected history. The loop uses the False
    case to stop new snapshot-dependent writes and explain why.
    """
    quota = quota_bytes_from_cfg(cfg)
    if quota <= 0:
        return True, ""
    plan = plan_checkpoint_gc(root, quota)
    if plan["total_bytes"] > quota and plan["to_delete"]:
        try:
            result = apply_checkpoint_gc(root, plan)
        except Exception as exc:          # fail closed: report, never crash
            return False, f"checkpoint GC failed ({type(exc).__name__})"
        if result.get("error"):
            return False, f"checkpoint GC could not run: {result['error']}"
        plan = plan_checkpoint_gc(root, quota)
    return plan["satisfiable"], plan["reason"]
