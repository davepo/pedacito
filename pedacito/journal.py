"""Durable write-ahead task journal for crash / SIGKILL recovery (D7).

Every autonomous task writes a single-line JSONL journal under
``<root>/.pedacito/journal/<wal_id>.jsonl``. Each ``append`` is one
newline-terminated JSON object, flushed and (by default) ``fsync``-ed so a
``SIGKILL`` / ``TerminateProcess`` / power loss can only lose the trailing,
partial line -- never a committed record.

The journal is a *write-ahead* record of what a task did, so a crashed run can
be detected (``find_incomplete_tasks`` / ``list_recovery``) and rolled back
(``rollback_incomplete_task`` / ``recover``) by restoring its last recorded
checkpoint.

Identity, redaction and durability (WP-4105A):

* The WAL file is named by a fresh **UUID** (``<wal_id>.jsonl``), never by the
  task title. Two same-title / simultaneous / crashed runs therefore never
  share a file, and a new run's identity is always independent. Reopening an
  interrupted run requires the explicit ``wal_id``; constructing a journal
  without one always mints a new run.
* The human title, session id and every error are **redacted** before they are
  serialised, so no secret or task prose ever reaches the journal bytes (and
  none reaches a filename).
* An ``fsync``/write failure is **surfaced**, not swallowed: ``TaskJournal``
  exposes ``durable`` and ``error``, and a best-effort ``fsync_error`` marker
  is appended so a later reader knows the durability guarantee was lost and
  refuses to claim recoverability.
* ``recover`` never marks a task terminal unless every requested operation
  succeeded, and it re-reads / revalidates the journal under the cross-process
  project lock to avoid a TOCTOU race.

Legacy (pre-v0.4.1) journals were named by a sanitised title and carry no
``version``/``wal_id``; they remain **read-only compatible**: the scanner and
``read_journal`` parse them, and recovery can still address them by title.

This module is deliberately import-light: it only touches ``safety`` (for
redaction) and, lazily, ``checkpoints`` / ``worktree`` during recovery.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

JOURNAL_DIR = "journal"

# Record schema version written by the current code. Legacy (pre-v0.4.1)
# journals carry no ``version``/``wal_id``; ``read_journal`` still parses them.
WAL_VERSION = 2

# A durable, bounded cap on the stored (redacted) title so a huge task prompt
# is never persisted verbatim into the journal metadata.
TITLE_BOUND = 400

JOURNAL_EVENT_TYPES = (
    "task_start", "step", "model_call", "edit", "checkpoint",
    "subprocess", "permission", "review", "task_end",
)

# Kinds written by the journal itself (not task events) that the recovery
# scanner must understand.
FSYNC_ERROR_KIND = "fsync_error"

# Resume is not implemented: no validated durable task-boundary, compatible
# config or known preimage exists to resume from. Never advertise it.
RESUME_NOT_IMPLEMENTED = (
    "resume is not implemented: there is no validated durable task-boundary, "
    "compatible config, or known preimage to resume from"
)


def _safe_task_id(task_id: str) -> str:
    """A filesystem-safe stem for a *legacy* title-named journal file.

    Only used to locate pre-v0.4.1 title-named WALs (read-only compatibility).
    New journals are always named by UUID, never by this stem.
    """
    text = "".join(
        c if c.isalnum() or c in "-_." else "-"
        for c in str(task_id or "task").strip()
    )
    text = re.sub(r"-{2,}", "-", text).strip("-._")[:120]
    return text or "task"


def _valid_uuid(value: Any) -> bool:
    """True when `value` is a valid UUID string (a WAL identity)."""
    try:
        uuid.UUID(str(value or ""))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def _redact(value: Any) -> Any:
    """Recursively redact string keys AND values so no secret-shaped text is
    ever persisted -- including arbitrary dictionary keys."""
    from . import safety
    if isinstance(value, str):
        return safety.redact(value)
    if isinstance(value, dict):
        return {safety.redact(str(k)): _redact(v) for k, v in value.items()}
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
    wal_id: str = ""
    durable: bool = True


def _fsync(fh) -> None:
    """``os.fsync`` indirection: the single seam for disk-error tests."""
    os.fsync(fh.fileno())


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
        wal_id: str | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self.task_id = str(task_id or "task")
        self.session_id = str(session_id or "")
        self.fsync = bool(fsync)
        # Serialization always redacts metadata and data (see ``_append_locked``
        # / ``_write_line``); the ``redact`` flag is retained only for call-site
        # compatibility and no longer changes what is written.
        self.redact = bool(redact)
        # Durable identity: a UUID separate from the title. An explicit
        # ``wal_id`` reopens an existing run; without one this is a NEW run.
        self.wal_id = str(wal_id) if wal_id and _valid_uuid(wal_id) \
            else uuid.uuid4().hex
        self.durable = True
        self.error = ""
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
                          / f"{self.wal_id}.jsonl")
            self._enabled = True
            if wal_id and _valid_uuid(wal_id):
                self._resume_from_disk()
        # A journal is durable only when it fsyncs AND writes to a real file:
        # ``fsync=False`` or a disabled journal (no root) can never claim
        # durable/terminal success.
        self.durable = self.fsync and self._enabled

    def _resume_from_disk(self) -> None:
        """Resume ``seq`` (and title) from an existing journal file.

        Only reached on an *explicit* reopen (``wal_id`` was provided): the
        filename is the WAL id, so the records belong to this exact run. A
        fresh construction (new UUID) never has an existing file to read.
        """
        path = self._path
        if path is None or not path.is_file():
            return
        records, _, _, _ = _read_journal_with_status(path)
        if not records:
            return
        self._seq = max(r.seq for r in records)
        last = records[-1]
        if last.task_id:
            self.task_id = last.task_id

    @property
    def path(self) -> Path:
        """The journal file path (exists conceptually even if not yet written)."""
        if self._path is None:
            return Path(f"{self.wal_id}.jsonl")
        return self._path

    def append(self, kind: str, **data: Any) -> JournalRecord:
        """Write one journal line and return its ``JournalRecord``.

        A missing/read-only ``root`` makes the journal a no-op that still
        returns an in-memory record, so callers never have to guard. An
        ``fsync``/write failure does not raise either: it sets ``durable`` to
        False and records ``error``, so the failure is surfaced, never hidden.
        """
        with self._lock:
            if self._closed:
                self._seq += 1
                return JournalRecord(
                    seq=self._seq, ts=time.time(), task_id=self.task_id,
                    session_id=self.session_id, kind=str(kind),
                    data=_redact(dict(data)),
                    wal_id=self.wal_id)
            return self._append_locked(kind, data)

    def _append_locked(self, kind: str, data: dict[str, Any]) -> JournalRecord:
        self._seq += 1
        rec = JournalRecord(
            seq=self._seq, ts=time.time(), task_id=self.task_id,
            session_id=self.session_id, kind=str(kind),
            data=_redact(dict(data)),
            wal_id=self.wal_id)
        if self._enabled and self._path is not None:
            try:
                self._write_line(rec)
            except OSError as exc:
                self.durable = False
                self.error = str(exc)
                self._write_fsync_error(str(exc))
        return rec

    def _write_line(self, rec: JournalRecord) -> None:
        path = self._path
        assert path is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        # Metadata is redacted unconditionally: the title, session id and kind
        # may carry secrets and must never be serialised raw.
        title = str(_redact(rec.task_id))[:TITLE_BOUND]
        payload = json.dumps({
            "version": WAL_VERSION,
            "wal_id": rec.wal_id,
            "seq": rec.seq,
            "ts": rec.ts,
            "task_id": title,
            "session_id": _redact(rec.session_id),
            "kind": _redact(rec.kind),
            "durable": bool(self.durable),
            "data": rec.data,
        }, ensure_ascii=True, separators=(",", ":"))
        with path.open("a", encoding="utf-8") as fh:
            fh.write(payload + "\n")
            fh.flush()
            if self.fsync:
                _fsync(fh)

    def _write_fsync_error(self, message: str) -> None:
        """Best-effort marker that durability was lost (no fsync: it is already
        failing). The error text is redacted before it is persisted.

        This marker is itself best-effort and may be lost in the very crash it
        records: there is no absolute guarantee it survives. The live caller
        (``durable=False`` in memory) must refuse further mutation, and the
        per-record ``durable`` flag below is the durable signal.
        """
        path = self._path
        if path is None:
            return
        try:
            self._seq += 1
            payload = json.dumps({
                "version": WAL_VERSION,
                "wal_id": self.wal_id,
                "seq": self._seq,
                "ts": time.time(),
                "task_id": str(_redact(self.task_id))[:TITLE_BOUND],
                "session_id": _redact(self.session_id),
                "kind": _redact(FSYNC_ERROR_KIND),
                "durable": False,
                "data": {"error": str(_redact(str(message)))},
            }, ensure_ascii=True, separators=(",", ":"))
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(payload + "\n")
                fh.flush()
        except OSError:
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
    every earlier committed record is returned cleanly. Legacy (unversioned)
    records without ``wal_id`` are parsed with ``wal_id=""``.
    """
    return _read_journal_with_status(path)[0]


def _read_journal_with_status(path: Path) -> tuple[list[JournalRecord], bool, bool, bool]:
    """``(records, torn, corrupt, non_durable)`` for one journal file.

    ``torn`` is True when the *trailing* line is a crash artifact: a partial
    line, or a valid JSON object that was never newline-terminated (the WAL
    contract writes ``payload + "\\n"``; a record without its terminating
    newline was never fully committed). ``corrupt`` is True for real corruption
    that must fail closed. ``non_durable`` is True when any record carries a
    genuine ``durable: false``.

    A **UUID-named** journal is a v2 WAL and is validated strictly: every
    record must have ``version == WAL_VERSION``, an exact nonempty ``wal_id``
    matching the file, an actual *boolean* ``durable`` field, a positive
    monotonic integer ``seq`` and a finite ``ts``. A missing/wrong-typed field
    (e.g. the string ``"false"`` for ``durable``) marks the record corrupt,
    never silently accepted as legacy. A **legacy title-named** journal is
    parsed leniently for read-only compatibility but is never trusted as
    recoverable, and string flags are never coerced.
    """
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError:
        return [], False, False, False
    if not raw:
        return [], False, False, False
    terminated = raw.endswith(b"\n")      # every committed record is newline-terminated
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    records: list[JournalRecord] = []
    torn = False
    corrupt = False
    non_durable = False
    n = len(lines)
    prev_seq = -1
    expected_wal_id = path.stem if _valid_uuid(path.stem) else ""
    strict = bool(expected_wal_id)        # UUID-named file -> enforce the v2 schema
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        is_last = i == n - 1
        if is_last and not terminated:
            torn = True                   # no newline: not a committed record
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            if is_last:
                torn = True               # truncated trailing line: crash artifact
            else:
                corrupt = True            # garbage mid-file: real corruption
            continue
        if not isinstance(obj, dict) or "kind" not in obj:
            corrupt = True
            continue

        # --- version (v2 requires the exact supported version) ---
        if strict and obj.get("version") != WAL_VERSION:
            corrupt = True
            continue

        # --- wal_id (v2 requires an exact nonempty match to the file) ---
        wal_raw = obj.get("wal_id")
        if strict:
            if not isinstance(wal_raw, str) or wal_raw != expected_wal_id:
                corrupt = True
                continue
            wal_id = wal_raw
        else:
            wal_id = str(wal_raw) if isinstance(wal_raw, str) else ""

        # --- seq (v2 requires a positive monotonic integer) ---
        seq_raw = obj.get("seq")
        if strict:
            if not isinstance(seq_raw, int) or isinstance(seq_raw, bool) \
                    or seq_raw <= 0:
                corrupt = True
                continue
            seq = seq_raw
        else:
            if isinstance(seq_raw, bool):
                corrupt = True
                continue
            try:
                seq = int(seq_raw or 0)
            except (TypeError, ValueError):
                corrupt = True
                continue
        if seq <= prev_seq:
            corrupt = True                # duplicate or non-monotonic sequence
            continue
        prev_seq = seq

        # --- ts (v2 requires a finite timestamp) ---
        ts_raw = obj.get("ts")
        if strict:
            if not isinstance(ts_raw, (int, float)) or isinstance(ts_raw, bool):
                corrupt = True
                continue
            ts = float(ts_raw)
        else:
            try:
                ts = float(ts_raw or 0.0)
            except (TypeError, ValueError):
                corrupt = True
                continue
        if not math.isfinite(ts):
            corrupt = True
            continue

        # --- durable (must be an actual boolean; never coerce a string) ---
        durable_raw = obj.get("durable")
        if isinstance(durable_raw, bool):
            durable = durable_raw
        elif durable_raw is None and not strict:
            durable = True                # legacy records were fsynced
        else:
            # Missing in a v2 file, or present but not a real boolean
            # (e.g. the string "false"): malformed, never coerced.
            corrupt = True
            continue
        if not durable:
            non_durable = True

        data = obj.get("data")
        if not isinstance(data, dict):
            data = {}
        records.append(JournalRecord(
            seq=seq,
            ts=ts,
            task_id=str(obj.get("task_id") or ""),
            session_id=str(obj.get("session_id") or ""),
            kind=str(obj.get("kind") or ""),
            data=data,
            wal_id=wal_id,
            durable=durable,
        ))
    return records, torn, corrupt, non_durable


@dataclass(frozen=True)
class RecoveryCandidate:
    """One recoverable (incomplete) task, as ``list_recovery`` reports it."""

    task_id: str
    session_id: str
    journal_path: Path
    checkpoint_id: str
    status: str                  # "incomplete" | "fsync_error" | "non_durable" | "torn" | "corrupt" | "no_checkpoint"
    resumable: bool
    reason: str
    wal_id: str = ""             # the recovery key ("" for legacy title-named WALs)


def _last_checkpoint(records: list[JournalRecord]) -> str:
    checkpoint_id = ""
    for r in records:
        if r.kind == "checkpoint" and r.data.get("checkpoint_id"):
            checkpoint_id = str(r.data["checkpoint_id"])
    return checkpoint_id


def _pre_run_checkpoints(records: list[JournalRecord]) -> list[str]:
    """Checkpoint ids recorded as the durable pre-run boundary (``phase="pre"``).

    Only an explicitly-identified pre-run boundary is a valid rollback target;
    a post-run checkpoint or a checkpoint of another task must never be
    restored. Returns all such ids (0 or >1 means "unsafe / ambiguous").
    """
    return [str(r.data["checkpoint_id"])
            for r in records
            if r.kind == "checkpoint" and r.data.get("checkpoint_id")
            and str(r.data.get("phase", "")) == "pre"]


def _last_worktree(records: list[JournalRecord]) -> str:
    worktree = ""
    for r in records:
        if r.data.get("worktree"):
            worktree = str(r.data["worktree"])
    return worktree


def _candidate(path: Path, records: list[JournalRecord],
               torn: bool, corrupt: bool,
               non_durable: bool) -> RecoveryCandidate | None:
    """A recovery candidate for one journal file, or None when clean/terminal.

    A torn, corrupt, fsync-error or non-durable journal is NEVER hidden, even
    when its last committed record is ``task_end``: trailing uncertainty or a
    lost durability guarantee means a new run may have begun (or the records
    may not survive a crash), so it must stay visible. Only a clean, durable,
    terminated journal yields no candidate. ``resumable`` is always False.
    """
    if not records:
        return None
    last = records[-1]
    checkpoint_id = _last_checkpoint(records)
    has_fsync_error = any(r.kind == FSYNC_ERROR_KIND for r in records)
    if corrupt:
        status, reason = ("corrupt",
                          "the journal contains corrupt records; refusing to "
                          "claim recoverability")
    elif has_fsync_error:
        status, reason = ("fsync_error",
                          "the journal lost its durability guarantee (fsync "
                          "failed); recoverability cannot be claimed")
    elif non_durable:
        status, reason = ("non_durable",
                          "the journal contains records that were not durably "
                          "persisted (fsync disabled); recoverability cannot "
                          "be claimed")
    elif torn:
        status, reason = ("torn",
                          "the journal ends with a torn line; the last "
                          "committed record may be lost")
    elif last.kind == "task_end":
        return None                     # clean durable terminal: nothing to recover
    elif not checkpoint_id:
        status, reason = ("no_checkpoint",
                          "no durable checkpoint was recorded; there is no "
                          "boundary to resume from")
    else:
        status, reason = ("incomplete", RESUME_NOT_IMPLEMENTED)
    return RecoveryCandidate(
        task_id=last.task_id or records[0].task_id or path.stem,
        session_id=last.session_id or records[0].session_id,
        journal_path=path,
        checkpoint_id=checkpoint_id,
        status=status,
        resumable=False,
        reason=reason,
        wal_id=last.wal_id or records[0].wal_id,
    )


def _is_link_or_reparse(path: Path) -> bool:
    """True when `path` is a symlink or a Windows reparse point (junction)."""
    try:
        if path.is_symlink():
            return True
        if os.name == "nt":
            st = os.lstat(path)
            if getattr(st, "st_file_attributes", 0) & 0x400:
                return True
    except OSError:
        return True
    return False


def _contained_journal_dir(root: Path) -> Path | None:
    """Return `.pedacito/journal` only when it is a real directory inside `root`."""
    try:
        root_res = Path(root).resolve()
    except OSError:
        return None
    ped_dir = root_res / ".pedacito"
    jdir = ped_dir / JOURNAL_DIR
    if not jdir.is_dir():
        return None
    if _is_link_or_reparse(ped_dir) or _is_link_or_reparse(jdir):
        return None
    try:
        jdir.resolve().relative_to(root_res)
    except (OSError, ValueError):
        return None
    return jdir


def _contained_journal_file(root: Path, jdir: Path, path: Path) -> Path | None:
    """Return `path` only when it is a regular non-link file inside `jdir`."""
    if not path.is_file() or _is_link_or_reparse(path):
        return None
    try:
        root_res = Path(root).resolve()
        jdir_res = jdir.resolve()
        path_res = path.resolve()
        path_res.relative_to(jdir_res)
        path_res.relative_to(root_res)
    except (OSError, ValueError):
        return None
    return path


def list_recovery(root: Path | str) -> list[RecoveryCandidate]:
    """Recovery candidates for every incomplete task under a project root."""
    root = Path(root)
    jdir = _contained_journal_dir(root)
    out: list[RecoveryCandidate] = []
    if jdir is None:
        return out
    for path in sorted(jdir.glob("*.jsonl")):
        safe_path = _contained_journal_file(root, jdir, path)
        if safe_path is None:
            continue
        records, torn, corrupt, non_durable = _read_journal_with_status(safe_path)
        cand = _candidate(safe_path, records, torn, corrupt, non_durable)
        if cand is not None:
            out.append(cand)
    return out


def _find_journal_path(root: Path, task_id: str) -> Path | None:
    """The journal path for `task_id`.

    A valid UUID addresses exactly the ``<wal_id>.jsonl`` file; anything else
    falls back to the legacy title-named lookup (read-only compatibility).
    There is deliberately no scan-by-title: a title can be shared by several
    UUID-named runs, and recovery must address exactly one run by its UUID.

    UUIDs are matched in both their literal and canonical-hex forms (a user may
    paste the hyphenated form while WAL filenames use ``uuid4().hex``). A valid
    UUID never falls through to the title lookup, so a title that merely looks
    like a UUID can never hijack an identity.
    """
    root = Path(root)
    jdir = _contained_journal_dir(root)
    if jdir is None:
        return None
    if _valid_uuid(task_id):
        stems = [str(task_id)]
        canonical = uuid.UUID(str(task_id)).hex
        if canonical not in stems:
            stems.append(canonical)
        for stem in stems:
            p = _contained_journal_file(root, jdir, jdir / f"{stem}.jsonl")
            if p is not None:
                return p
        return None
    stem = _safe_task_id(str(task_id))
    return _contained_journal_file(root, jdir, jdir / f"{stem}.jsonl")


RECOVERY_ACTIONS = ("resume", "rollback", "cleanup")


def _acquire_project_lock(root: Path) -> tuple[bool, str]:
    """Acquire the cross-process project write lock for a destructive recovery.

    Returns ``(True, "")`` on success, ``(False, reason)`` when another live
    run holds the project (or the lockfile cannot be written).
    """
    try:
        from . import workspace
        workspace.acquire_lock(root, mode="exclusive")
        return True, ""
    except Exception as exc:  # noqa: BLE001 - ProjectBusy and friends
        return False, f"could not acquire the project write lock: {exc}"


def _release_project_lock(root: Path) -> None:
    try:
        from . import workspace
        workspace.release_lock(root)
    except Exception:                        # noqa: BLE001 - best effort
        pass


def _append_terminal(path: Path, *, wal_id: str, title: str, session_id: str,
                     status: str, **summary: Any) -> tuple[bool, str]:
    """Append a durable ``task_end`` to an existing journal file.

    Resumes the sequence from the highest committed record (monotonic), and
    redacts every serialised field. Returns ``(durable, error)`` so the caller
    can refuse to report success when the terminal marker was not durable.
    """
    records, _, _, _ = _read_journal_with_status(path)
    seq = max((r.seq for r in records), default=0) + 1
    payload = json.dumps({
        "version": WAL_VERSION,
        "wal_id": str(wal_id or ""),
        "seq": seq,
        "ts": time.time(),
        "task_id": str(_redact(title))[:TITLE_BOUND],
        "session_id": _redact(session_id),
        "kind": _redact("task_end"),
        "durable": True,
        "data": {"status": str(status),
                 **{_redact(str(k)): _redact(v) for k, v in summary.items()}},
    }, ensure_ascii=True, separators=(",", ":"))
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(payload + "\n")
            fh.flush()
            _fsync(fh)
        return True, ""
    except OSError as exc:
        return False, str(exc)


def _checkpoint_wal_id(root: Path, checkpoint_id: str) -> str:
    """The WAL id bound to a checkpoint entry at creation, or "" (unbound)."""
    from . import checkpoints
    data = checkpoints._load_index(root)
    for entry in data.get("checkpoints") or []:
        if str(entry.get("id")) == str(checkpoint_id):
            return str(entry.get("wal_id") or "")
    return ""


def _verify_recovery_checkpoint(
    root: Path, checkpoint_id: str, journal_wal_id: str
) -> tuple[str, str, str, str]:
    """Read-only validation of a rollback target.

    Returns ``(resolved_id, detail, code, error)`` where an empty ``detail``
    means the checkpoint exists, is bound to this WAL and still has its
    snapshot directory. Every check runs in both the preview and the apply
    path, so a preview can never claim a rollback that apply would refuse.
    """
    from . import checkpoints

    try:
        resolved_id = checkpoints.resolve_id(root, checkpoint_id)
    except Exception as exc:  # noqa: BLE001 - a missing index entry is a refusal
        return ("", f"checkpoint {checkpoint_id!r} cannot be verified: {exc}",
                "checkpoint_unverified", str(exc))
    # Ownership: the checkpoint must be bound to THIS WAL's UUID at creation.
    # A legacy/unbound checkpoint or a foreign task's checkpoint (different
    # wal_id) is refused.
    if not journal_wal_id:
        return (resolved_id,
                f"checkpoint {resolved_id} is not bound to this WAL; "
                "refusing to restore it",
                "checkpoint_unbound", "")
    entry_wal_id = _checkpoint_wal_id(root, resolved_id)
    if not entry_wal_id or entry_wal_id != journal_wal_id:
        return (resolved_id,
                f"checkpoint {resolved_id} is not bound to this WAL; "
                "refusing to restore it",
                "checkpoint_unbound", "")
    # A missing, partial, altered or unmanifested snapshot would let restore
    # treat current files as deleted or copy damaged bytes; refuse instead.
    try:
        verified, reason = checkpoints.verify_recovery_snapshot(
            root, resolved_id, journal_wal_id)
    except Exception as exc:  # noqa: BLE001
        return (resolved_id,
                f"checkpoint {resolved_id} snapshot cannot be verified: {exc}",
                "checkpoint_unverified", str(exc))
    if not verified:
        return (resolved_id,
                f"checkpoint {resolved_id} cannot be verified: {reason}",
                "checkpoint_unverified", str(reason))
    return resolved_id, "", "", ""


def recover(root: Path | str, task_id: str, *, action: str,
            dry_run: bool = True) -> dict[str, Any]:
    """Preview or apply a recovery action for one task's journal.

    ``task_id`` is the recovery identity: a WAL UUID (addressing exactly that
    run's ``<wal_id>.jsonl``), or a legacy title (addressing a pre-v0.4.1
    title-named WAL).

    ``action`` is ``resume`` | ``rollback`` | ``cleanup``. Destructive actions
    default to a preview (``dry_run=True``). The task is **never** marked
    terminal unless every requested operation succeeded: a failed or partial
    restore leaves the journal incomplete so it stays visible to recovery.
    Torn / corrupt / fsync-error journals fail closed for mutations.

    ``result["code"]`` is a stable ASCII refusal/success code for surfaces that
    localize the message; ``result["detail"]`` stays the canonical English
    explanation.
    """
    root = Path(root)
    action = str(action or "").strip().lower()
    result: dict[str, Any] = {
        "ok": False, "action": action, "task_id": str(task_id),
        "dry_run": bool(dry_run), "detail": "", "code": "",
    }
    if action not in RECOVERY_ACTIONS:
        result["detail"] = f"unknown recovery action {action!r}"
        result["code"] = "unknown_action"
        return result

    path = _find_journal_path(root, str(task_id))
    if path is None:
        result["detail"] = "no journal found for this task"
        result["code"] = "no_journal"
        return result
    result["journal_path"] = str(path)
    records, torn, corrupt, non_durable = _read_journal_with_status(path)
    if not records:
        result["detail"] = "the journal file is empty or corrupt"
        result["code"] = "journal_empty"
        return result

    has_fsync_error = any(r.kind == FSYNC_ERROR_KIND for r in records)
    if records[-1].kind == "task_end":
        if torn or corrupt or non_durable or has_fsync_error:
            # A trailing torn line (or lost durability) means a newer run may
            # have started after this task_end; never present it as cleanly
            # terminal, because that would hide recoverable work.
            result["detail"] = ("refusing to treat this task as terminal: the "
                                "journal is not intact (corrupt/torn/"
                                "durability-lost)")
            result["code"] = "journal_not_intact"
        else:
            result["detail"] = "this task is already terminal; nothing to recover"
            result["code"] = "terminal"
        return result

    if action == "resume":
        cand = _candidate(path, records, torn, corrupt, non_durable)
        if cand is None:
            result["detail"] = "this task is already terminal"
            result["code"] = "terminal"
            return result
        result.update(resumable=cand.resumable, status=cand.status,
                      reason=cand.reason, checkpoint_id=cand.checkpoint_id)
        result["ok"] = cand.resumable
        result["code"] = "resume_unavailable"
        result["detail"] = f"cannot resume: {cand.reason}"
        return result

    if action == "rollback":
        if corrupt or torn or non_durable \
                or any(r.kind == FSYNC_ERROR_KIND for r in records):
            result["detail"] = ("refusing to roll back: the journal is not "
                                "intact (corrupt/torn/durability-lost)")
            result["code"] = "journal_not_intact"
            return result
        pre_ids = _pre_run_checkpoints(records)
        if not pre_ids:
            result["detail"] = ("no durable pre-run boundary recorded "
                                "(phase='pre'); refusing to roll back")
            result["code"] = "no_pre_run_boundary"
            return result
        if len(pre_ids) > 1:
            result["detail"] = ("ambiguous: multiple pre-run boundaries "
                                "recorded; refusing to roll back")
            result["code"] = "ambiguous_pre_run_boundary"
            return result
        if dry_run:
            journal_wal_id = records[-1].wal_id or records[0].wal_id
            resolved_id, detail, code, error = _verify_recovery_checkpoint(
                root, pre_ids[0], journal_wal_id)
            if detail:
                result.update(detail=detail, code=code, error=error)
                if resolved_id:
                    result["checkpoint_id"] = resolved_id
                return result
            result["ok"] = True
            result["restored_checkpoint"] = resolved_id
            result["code"] = "rollback_preview"
            result["detail"] = f"would restore checkpoint {resolved_id}"
            return result
        locked, reason = _acquire_project_lock(root)
        if not locked:
            result["detail"] = reason
            result["code"] = "project_busy"
            result["error"] = str(reason)
            return result
        try:
            # TOCTOU guard: re-read and revalidate under the lock.
            recs2, torn2, corrupt2, non_durable2 = _read_journal_with_status(path)
            if not recs2:
                result["detail"] = "the journal changed and is now empty or corrupt"
                result["code"] = "journal_changed_empty"
                return result
            has_fsync_error2 = any(
                r.kind == FSYNC_ERROR_KIND for r in recs2)
            if recs2[-1].kind == "task_end":
                if torn2 or corrupt2 or non_durable2 or has_fsync_error2:
                    result["detail"] = ("refusing to treat this task as "
                                        "terminal: the journal changed and is "
                                        "no longer intact")
                    result["code"] = "journal_not_intact"
                else:
                    result["detail"] = ("the task became terminal since the "
                                        "preview")
                    result["code"] = "terminal"
                return result
            if corrupt2 or torn2 or non_durable2 or has_fsync_error2:
                result["detail"] = ("refusing to roll back: the journal changed "
                                    "and is no longer intact")
                result["code"] = "journal_not_intact"
                return result
            pre_ids2 = _pre_run_checkpoints(recs2)
            if not pre_ids2:
                result["detail"] = "no durable pre-run boundary recorded; refusing"
                result["code"] = "no_pre_run_boundary"
                return result
            if len(pre_ids2) > 1:
                result["detail"] = "ambiguous: multiple pre-run boundaries; refusing"
                result["code"] = "ambiguous_pre_run_boundary"
                return result
            journal_wal_id = recs2[-1].wal_id or recs2[0].wal_id
            resolved_id, detail, code, error = _verify_recovery_checkpoint(
                root, pre_ids2[0], journal_wal_id)
            if detail:
                result.update(detail=detail, code=code, error=error)
                if resolved_id:
                    result["checkpoint_id"] = resolved_id
                return result

            from . import checkpoints
            restore = (getattr(checkpoints, "restore_checkpoint", None)
                       or getattr(checkpoints, "restore", None))
            if not callable(restore):
                result["detail"] = "no checkpoint restore function available"
                result["code"] = "restore_unavailable"
                return result
            restored = restore(root, resolved_id)
            if isinstance(restored, dict) and restored.get("ok") is False:
                result["detail"] = ("checkpoint restore reported failure: "
                                    f"{restored.get('detail') or restored}")
                result["code"] = "restore_failed"
                result["error"] = str(restored.get('detail') or restored)
                return result
            if isinstance(restored, dict) and restored.get("skipped"):
                skipped = len(restored["skipped"])
                result["detail"] = (f"checkpoint restore was partial "
                                    f"({skipped} file(s) skipped); "
                                    "refusing to mark the task terminal")
                result["code"] = "restore_partial"
                result["skipped"] = skipped
                return result
            if isinstance(restored, dict) and restored.get("safety_id"):
                result["safety_id"] = str(restored["safety_id"])

            # Mark terminal durably; only report success if the marker is durable.
            durable, err = _append_terminal(
                path,
                wal_id=journal_wal_id,
                title=recs2[-1].task_id or recs2[0].task_id or path.stem,
                session_id=recs2[-1].session_id or recs2[0].session_id,
                status="rolled_back", restored_checkpoint=resolved_id)
            if not durable:
                result["detail"] = (f"checkpoint restored but the terminal "
                                    f"record was not durable: {err}")
                result["code"] = "terminal_not_durable"
                result["error"] = str(err)
                result["restored_checkpoint"] = resolved_id
                return result
            result["ok"] = True
            result["restored_checkpoint"] = resolved_id
            result["code"] = "rollback_applied"
            result["detail"] = f"restored checkpoint {resolved_id}"
            return result
        except Exception as exc:  # noqa: BLE001 - recovery reports, never raises
            result["detail"] = f"checkpoint restore failed: {exc}"
            result["code"] = "restore_failed"
            result["error"] = str(exc)
            return result
        finally:
            _release_project_lock(root)

    # action == "cleanup"
    # No authoritative cross-process ownership/liveness mechanism exists in
    # this release to prove a specific recorded worktree is orphaned (a
    # caller-supplied session set is not proof that every session and OS
    # process was enumerated). Refuse rather than falsely claim orphan status.
    result["detail"] = ("cleanup requires an authoritative cross-process "
                        "ownership/liveness check for the precise worktree/"
                        "session, which is not available in this release; "
                        "refusing")
    result["code"] = "cleanup_unavailable"
    return result


def find_incomplete_tasks(root: Path | str) -> list[dict[str, Any]]:
    """Summaries for tasks that have started but never recorded ``task_end``.

    Conservative terminal detection: a journal is complete only when its LAST
    committed record is ``task_end``. A completed run followed by a crashed
    run with the same title is still incomplete (the old "any earlier
    ``task_end`` is terminal" reading was wrong)."""
    root = Path(root)
    jdir = root / ".pedacito" / JOURNAL_DIR
    out: list[dict[str, Any]] = []
    if not jdir.is_dir():
        return out
    for path in sorted(jdir.glob("*.jsonl")):
        records, torn, corrupt, non_durable = _read_journal_with_status(path)
        if not records:
            continue
        has_fsync_error = any(r.kind == FSYNC_ERROR_KIND for r in records)
        # Only a CLEAN, DURABLE terminated journal is complete; a torn/corrupt/
        # non-durable/fsync-error journal is uncertain and must stay visible
        # even if its last committed record is task_end.
        if records[-1].kind == "task_end" and not torn and not corrupt \
                and not non_durable and not has_fsync_error:
            continue
        first = records[0]
        last = records[-1]
        checkpoint_id = _last_checkpoint(records)
        resumable = False  # resume is not implemented (see _candidate)
        out.append({
            "task_id": last.task_id or first.task_id or path.stem,
            "session_id": last.session_id or first.session_id,
            "path": str(path),
            "last_seq": last.seq,
            "last_kind": last.kind,
            "checkpoint_id": checkpoint_id,
            "started_at": first.ts,
            "updated_at": last.ts,
            "wal_id": last.wal_id or first.wal_id,
            "resumable": resumable,
            "torn": torn,
            "corrupt": corrupt,
            "non_durable": non_durable,
        })
    return out


def rollback_incomplete_task(root: Path | str, task_id: str) -> dict[str, Any]:
    """Roll back one incomplete task's journal (legacy compatibility wrapper).

    Delegates to the safe :func:`recover` path (project lock, verified
    checkpoint, fail-closed on torn/corrupt journals, durable terminal append)
    and returns the legacy result shape. A failed or refused rollback returns
    ``ok=False`` and does **not** mark the task terminal.
    """
    result = recover(root, task_id, action="rollback", dry_run=False)
    return {
        "ok": bool(result.get("ok")),
        "task_id": str(task_id),
        "restored_checkpoint": str(result.get("restored_checkpoint") or ""),
        "detail": str(result.get("detail") or ""),
    }


# ---------------------------------------------------------------------------
# Atomic edit transactions (WP-4105C)
# ---------------------------------------------------------------------------
# A staged all-or-nothing multi-file edit writes its preimages and a durable
# "prepared" record BEFORE the first file changes, then marks the record
# "committed" once every write succeeded. A crash mid-commit leaves the record
# "prepared", and :func:`recover_atomic` restores the preimages on the next
# run. This is a write-ahead log for the edit batch, not a claim of
# instantaneous filesystem-wide atomic visibility: the guarantee is "all
# candidates validated and durably snapshotted before writes; roll back on
# in-process write failure; recover a crashed commit from the WAL on the next
# run".

ATOMIC_DIR = "atomic"
ATOMIC_VERSION = 1
ATOMIC_STATUS_PREPARED = "prepared"
ATOMIC_STATUS_COMMITTED = "committed"
ATOMIC_STATUS_ROLLED_BACK = "rolled_back"
ATOMIC_STATUS_RECOVERY_FAILED = "recovery_failed"

# In-process serialization only. An in-process lock is not a cross-process
# lock: concurrent *processes* are not protected, matching the rest of the
# single-writer edit path.
_ATOMIC_LOCK = threading.RLock()


@dataclass(frozen=True)
class AtomicFile:
    """One file in an atomic edit transaction."""

    rel: str            # project-relative POSIX path
    sha256: str         # full sha256 hex of the preimage bytes; "" for a new file
    existed: bool       # True: preimage bytes were snapshotted; False: new file


@dataclass(frozen=True)
class AtomicTxn:
    """An atomic edit transaction record (WAL)."""

    tx_id: str
    status: str
    files: tuple[AtomicFile, ...]


def _atomic_root(root: Path | str) -> Path:
    return Path(root).resolve() / ".pedacito" / ATOMIC_DIR


def _atomic_txn_path(root: Path | str, tx_id: str) -> Path:
    return _atomic_root(root) / f"{tx_id}.json"


def _atomic_preimage_dir(root: Path | str, tx_id: str) -> Path:
    return _atomic_root(root) / tx_id


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fsync_dir(path: Path) -> None:
    """Best-effort fsync of a directory (POSIX). Windows cannot fsync a
    directory handle; it is skipped there."""
    if os.name == "nt":
        return
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _durable_write_bytes(path: Path, data: bytes) -> None:
    """Write `data` to `path` atomically and durably (file + dir fsync)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                               dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _durable_write_text(path: Path, text: str) -> None:
    _durable_write_bytes(path, text.encode("utf-8"))


def _write_txn(root: Path, txn: AtomicTxn) -> None:
    files = [
        {"rel": str(f.rel), "sha256": str(f.sha256), "existed": bool(f.existed)}
        for f in txn.files
    ]
    payload = {
        "version": ATOMIC_VERSION,
        "tx_id": str(txn.tx_id),
        "status": str(txn.status),
        "files": files,
    }
    _durable_write_text(
        _atomic_txn_path(root, txn.tx_id),
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n")


def _read_txn(path: Path) -> AtomicTxn | None:
    """Parse and validate one transaction record file. Returns None when the
    record is missing, malformed, or carries a tx_id that does not match its
    filename (a tampered/foreign record is never trusted)."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("version") != ATOMIC_VERSION:
        return None
    tx_id = obj.get("tx_id")
    if not isinstance(tx_id, str) or not _valid_uuid(tx_id) \
            or path.stem != uuid.UUID(tx_id).hex:
        return None
    status = obj.get("status")
    if not isinstance(status, str):
        return None
    files_raw = obj.get("files")
    if not isinstance(files_raw, list):
        return None
    files: list[AtomicFile] = []
    for item in files_raw:
        if not isinstance(item, dict):
            return None
        rel = item.get("rel")
        if not isinstance(rel, str) or not _safe_atomic_rel(rel):
            return None
        sha = item.get("sha256")
        existed = item.get("existed")
        if not isinstance(sha, str) or not isinstance(existed, bool):
            return None
        if existed and not re.fullmatch(r"[0-9a-f]{64}", sha):
            return None
        if not existed and sha != "":
            return None
        files.append(AtomicFile(rel=str(rel), sha256=str(sha), existed=bool(existed)))
    return AtomicTxn(tx_id=tx_id, status=status, files=tuple(files))


def _safe_atomic_rel(rel: str) -> bool:
    """True when `rel` is a plain relative project path: no traversal, no
    absolute path, no NUL, no backslash separator."""
    if not rel or "\x00" in rel:
        return False
    r = rel.replace("\\", "/")
    if r.startswith("/"):
        return False
    return all(p and p not in (".", "..") for p in r.split("/"))


def begin_atomic(root: Path | str, preimages: dict[str, bytes],
                 created: Iterable[str] = ()) -> AtomicTxn:
    """Stage an atomic edit transaction: snapshot every preimage durably and
    write the "prepared" WAL record BEFORE the caller writes anything.

    ``preimages`` maps each *existing* affected file to its raw bytes (the
    exact bytes to restore on rollback/recovery). ``created`` names files the
    transaction will create (absent before it), so recovery can remove them.

    Raises ``OSError`` when a preimage or the record cannot be written -- a
    snapshot failure must stop the batch, never proceed without its safety net.
    """
    root = Path(root)
    tx_id = uuid.uuid4().hex
    files: list[AtomicFile] = []
    with _ATOMIC_LOCK:
        preimage_dir = _atomic_preimage_dir(root, tx_id)
        for rel in sorted(preimages):
            data = bytes(preimages[rel])
            _durable_write_bytes(preimage_dir / rel, data)
            files.append(AtomicFile(str(rel), _sha256_bytes(data), True))
        for rel in sorted(created):
            files.append(AtomicFile(str(rel), "", False))
        txn = AtomicTxn(tx_id=tx_id, status=ATOMIC_STATUS_PREPARED,
                        files=tuple(files))
        _write_txn(root, txn)
    return txn


def finish_atomic(root: Path | str, tx_id: str, status: str) -> None:
    """Durably mark a transaction terminal (``committed`` / ``rolled_back`` /
    ``recovery_failed``)."""
    with _ATOMIC_LOCK:
        txn = _read_txn(_atomic_txn_path(root, tx_id))
        if txn is None:
            return
        _write_txn(root, AtomicTxn(tx_id=str(tx_id), status=str(status),
                                   files=txn.files))


def remove_atomic(root: Path | str, tx_id: str) -> None:
    """Delete a finished transaction's record and preimage snapshot."""
    import shutil
    with _ATOMIC_LOCK:
        try:
            _atomic_txn_path(root, tx_id).unlink()
        except OSError:
            pass
        shutil.rmtree(_atomic_preimage_dir(root, tx_id), ignore_errors=True)


def _atomic_store_safe(root: Path) -> Path | None:
    """``.pedacito/atomic`` only when it is a real directory inside ``root``
    (never through a symlink or junction)."""
    from . import security
    root_res = Path(root).resolve()
    store = root_res / ".pedacito" / ATOMIC_DIR
    if not store.is_dir():
        return None
    try:
        if store.is_symlink() or security.is_junction_or_reparse(store):
            return None
        store.resolve().relative_to(root_res)
    except (OSError, ValueError):
        return None
    return store


def recover_atomic(root: Path | str) -> list[str]:
    """Roll back any incomplete atomic edit transaction left by a crash.

    A "prepared" transaction whose commit never completed is restored to its
    preimages (existing files) or removed (files the transaction created). A
    preimage that is missing or fails its recorded hash leaves the transaction
    visible (``recovery_failed``) rather than being silently discarded.
    Returns the project-relative paths that were restored or removed.
    """
    from . import fileio, security
    root = Path(root)
    store = _atomic_store_safe(root)
    restored: list[str] = []
    if store is None:
        return restored
    for path in sorted(store.glob("*.json")):
        if path.is_symlink() or security.is_junction_or_reparse(path):
            continue
        txn = _read_txn(path)
        if txn is None:
            continue
        if txn.status in (ATOMIC_STATUS_COMMITTED, ATOMIC_STATUS_ROLLED_BACK,
                          ATOMIC_STATUS_RECOVERY_FAILED):
            continue
        if txn.status != ATOMIC_STATUS_PREPARED:
            continue                       # unknown status: refuse to guess
        failed: list[str] = []
        touched: list[str] = []
        for f in txn.files:
            rel = str(f.rel)
            try:
                dest = security.resolve_safe_path(root, rel)
            except security.SecurityError:
                failed.append(rel)
                continue
            if f.existed:
                pre = _atomic_preimage_dir(root, txn.tx_id) / rel
                try:
                    data = pre.read_bytes()
                except OSError:
                    failed.append(rel)
                    continue
                if _sha256_bytes(data) != f.sha256:
                    failed.append(rel)
                    continue
                try:
                    fileio.write_bytes(dest, data)
                    touched.append(rel)
                except OSError:
                    failed.append(rel)
            else:
                try:
                    if dest.is_symlink() or dest.is_file():
                        dest.unlink()
                    touched.append(rel)
                except OSError:
                    failed.append(rel)
        if failed:
            finish_atomic(root, txn.tx_id, ATOMIC_STATUS_RECOVERY_FAILED)
        else:
            finish_atomic(root, txn.tx_id, ATOMIC_STATUS_ROLLED_BACK)
            remove_atomic(root, txn.tx_id)
        restored.extend(touched)
    return restored
