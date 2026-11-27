"""Durable write-ahead task journal for crash / SIGKILL recovery (D7).

Every autonomous task writes a single-line JSONL journal under
``<root>/.pedacito/journal/<safe_task_id>.jsonl``. Each ``append`` is one
newline-terminated JSON object, flushed and (by default) ``fsync``-ed so a
``SIGKILL`` / ``TerminateProcess`` / power loss can only lose the trailing,
partial line -- never a committed record.

The journal is a *write-ahead* record of what a task did, so a crashed run can
be detected (``find_incomplete_tasks``) and rolled back
(``rollback_incomplete_task``) by restoring its last recorded checkpoint. All
string values pass through ``safety.redact`` before serialisation so a secret
is never persisted.

This module is deliberately import-light: it only touches ``safety`` (for
redaction) and, lazily, ``checkpoints`` / ``worktree`` during rollback.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

JOURNAL_DIR = "journal"

JOURNAL_EVENT_TYPES = (
    "task_start", "step", "model_call", "edit", "checkpoint",
    "subprocess", "permission", "review", "task_end",
)


def _safe_task_id(task_id: str) -> str:
    """A filesystem-safe journal file stem for one task id / title."""
    text = "".join(
        c if c.isalnum() or c in "-_." else "-"
        for c in str(task_id or "task").strip()
    )
    text = re.sub(r"-{2,}", "-", text).strip("-._")[:120]
    return text or "task"


def _redact(value: Any) -> Any:
    """Recursively redact string fields so secrets are never persisted."""
    from . import safety
    if isinstance(value, str):
        return safety.redact(value)
    if isinstance(value, dict):
        return {str(k): _redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(v) for v in value]
    return value


@dataclass(frozen=True)
class JournalRecord:
    """One committed journal entry."""

    seq: int
    ts: float
    task_id: str
    session_id: str
    kind: str
    data: dict[str, Any]


class TaskJournal:
    """Append-only JSONL journal for one task, thread-safe and crash-durable."""

    def __init__(
        self,
        root: Path | str,
        task_id: str,
        session_id: str = "",
        *,
        fsync: bool = True,
        redact: bool = True,
    ) -> None:
        self._lock = threading.Lock()
        self.task_id = str(task_id or "task")
        self.session_id = str(session_id or "")
        self.fsync = bool(fsync)
        self.redact = bool(redact)
        self._seq = 0
        self._closed = False
        self._enabled = False
        try:
            root_path = Path(root)
        except (TypeError, ValueError):
            root_path = None
        self._path: Path | None = None
        if root_path is not None and str(root_path):
            self._path = (root_path / ".pedacito" / JOURNAL_DIR
                          / f"{_safe_task_id(self.task_id)}.jsonl")
            self._enabled = True

    @property
    def path(self) -> Path:
        """The journal file path (exists conceptually even if not yet written)."""
        if self._path is None:
            return Path(f"{_safe_task_id(self.task_id)}.jsonl")
        return self._path

    def append(self, kind: str, **data: Any) -> JournalRecord:
        """Write one journal line and return its ``JournalRecord``.

        A missing/read-only ``root`` makes the journal a no-op that still
        returns an in-memory record, so callers never have to guard.
        """
        with self._lock:
            if self._closed:
                # The task already ended: return an in-memory record without
                # writing (so a stray append after close() can never corrupt a
                # committed journal file).
                self._seq += 1
                return JournalRecord(
                    seq=self._seq, ts=time.time(), task_id=self.task_id,
                    session_id=self.session_id, kind=str(kind),
                    data=(_redact(dict(data)) if self.redact else dict(data)))
            return self._append_locked(kind, data)

    def _append_locked(self, kind: str, data: dict[str, Any]) -> JournalRecord:
        self._seq += 1
        rec = JournalRecord(
            seq=self._seq, ts=time.time(), task_id=self.task_id,
            session_id=self.session_id, kind=str(kind),
            data=(_redact(dict(data)) if self.redact else dict(data)))
        if self._enabled and self._path is not None:
            try:
                self._write_line(rec)
            except OSError:
                # Read-only root / full disk / synthetic stream: best effort.
                pass
        return rec

    def _write_line(self, rec: JournalRecord) -> None:
        path = self._path
        assert path is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({
            "seq": rec.seq,
            "ts": rec.ts,
            "task_id": rec.task_id,
            "session_id": rec.session_id,
            "kind": rec.kind,
            "data": rec.data,
        }, ensure_ascii=True, separators=(",", ":"))
        with path.open("a", encoding="utf-8") as fh:
            fh.write(payload + "\n")
            fh.flush()
            if self.fsync:
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    # Synthetic / in-memory streams have no backing fd.
                    pass

    def close(self, status: str = "completed", **summary: Any) -> None:
        """Append a terminal ``task_end`` record (idempotent)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._append_locked("task_end",
                                {"status": str(status), **summary})


def read_journal(path: Path) -> list[JournalRecord]:
    """Parse every committed line of a journal file.

    Tolerates a truncated/corrupt trailing line left by ``SIGKILL`` /
    ``TerminateProcess`` / power loss: the partial JSON line is ignored and
    every earlier committed record is returned cleanly.
    """
    path = Path(path)
    records: list[JournalRecord] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue            # truncated/corrupt line: skip it
                if not isinstance(obj, dict) or "kind" not in obj:
                    continue
                data = obj.get("data")
                if not isinstance(data, dict):
                    data = {}
                records.append(JournalRecord(
                    seq=int(obj.get("seq") or 0),
                    ts=float(obj.get("ts") or 0.0),
                    task_id=str(obj.get("task_id") or ""),
                    session_id=str(obj.get("session_id") or ""),
                    kind=str(obj.get("kind") or ""),
                    data=data,
                ))
    except OSError:
        return []
    return records


def find_incomplete_tasks(root: Path | str) -> list[dict[str, Any]]:
    """Summaries for tasks that have started but never recorded ``task_end``."""
    root = Path(root)
    jdir = root / ".pedacito" / JOURNAL_DIR
    out: list[dict[str, Any]] = []
    if not jdir.is_dir():
        return out
    for path in sorted(jdir.glob("*.jsonl")):
        records = read_journal(path)
        if not records:
            continue
        if any(r.kind == "task_end" for r in records):
            continue
        first = records[0]
        last = records[-1]
        checkpoint_id = ""
        for r in records:
            if r.kind == "checkpoint" and r.data.get("checkpoint_id"):
                checkpoint_id = str(r.data["checkpoint_id"])
        out.append({
            "task_id": first.task_id or path.stem,
            "session_id": first.session_id,
            "path": str(path),
            "last_seq": last.seq,
            "last_kind": last.kind,
            "checkpoint_id": checkpoint_id,
            "started_at": first.ts,
            "updated_at": last.ts,
        })
    return out


def rollback_incomplete_task(root: Path | str, task_id: str) -> dict[str, Any]:
    """Roll back one incomplete task's journal.

    Restores its last recorded checkpoint (via ``checkpoints.restore`` if
    available), removes any recorded ephemeral worktree, and appends a
    ``task_end`` (``status="rolled_back"``) so the task is no longer flagged as
    incomplete. Returns a status dict; never raises.
    """
    root = Path(root)
    safe = _safe_task_id(task_id)
    path = root / ".pedacito" / JOURNAL_DIR / f"{safe}.jsonl"
    result: dict[str, Any] = {
        "ok": False,
        "task_id": str(task_id),
        "restored_checkpoint": "",
        "detail": "",
    }
    if not path.is_file():
        result["detail"] = "no journal found for this task"
        return result

    records = read_journal(path)
    if not records:
        result["detail"] = "the journal file is empty or corrupt"
        return result

    checkpoint_id = ""
    worktree_path = ""
    for r in records:
        d = r.data or {}
        if r.kind == "checkpoint" and d.get("checkpoint_id"):
            checkpoint_id = str(d["checkpoint_id"])
        if d.get("worktree"):
            worktree_path = str(d["worktree"])

    detail_bits: list[str] = []
    if checkpoint_id:
        try:
            from . import checkpoints
            restore = getattr(checkpoints, "restore", None) \
                or getattr(checkpoints, "restore_checkpoint", None)
            if callable(restore):
                restore(root, checkpoint_id)
                result["restored_checkpoint"] = checkpoint_id
                detail_bits.append(f"restored checkpoint {checkpoint_id}")
            else:
                detail_bits.append("no checkpoint restore function available")
        except Exception as exc:  # noqa: BLE001 - rollback is best effort
            detail_bits.append(f"checkpoint restore failed: {exc}")

    if worktree_path:
        try:
            wt = Path(worktree_path)
            if wt.exists():
                _remove_worktree_path(root, wt)
            detail_bits.append("removed ephemeral worktree")
        except Exception as exc:  # noqa: BLE001
            detail_bits.append(f"worktree cleanup failed: {exc}")

    try:
        journal = TaskJournal(root, task_id, records[0].session_id, redact=True)
        journal.close(status="rolled_back",
                      restored_checkpoint=result["restored_checkpoint"])
    except OSError:
        pass

    result["ok"] = True
    result["detail"] = "; ".join(detail_bits) or "rolled back (no checkpoint)"
    return result


def _remove_worktree_path(root: Path, worktree_path: Path) -> None:
    """Remove a recorded worktree directory by path (best effort)."""
    import shutil
    try:
        import subprocess
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree_path)],
            cwd=str(root), check=False, capture_output=True, timeout=30)
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=str(root), check=False, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        pass
    if worktree_path.exists():
        shutil.rmtree(worktree_path, ignore_errors=True)
