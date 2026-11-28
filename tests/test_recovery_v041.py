"""WP-4105A -- safe journal identity and recovery primitives.

Covers the v0.4.1 recovery contract (SS3.7): UUID WAL identity separate from
the title, monotonic sequence across explicit reopen, versioned records,
surfaced fsync failures, redaction of metadata/errors, conservative legacy
scanning, strict WAL parsing, pre-run-boundary ownership, and the
``RecoveryCandidate`` / ``list_recovery`` / ``recover`` primitives.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import checkpoints, journal, workspace
from pedacito import worktree as worktree_mod


# ----------------------------------------------------------------- identity


def test_wal_id_is_a_uuid_and_file_is_uuid_named(tmp_path):
    j = journal.TaskJournal(tmp_path, "Some task title", session_id="s")
    uuid.UUID(j.wal_id)  # valid UUID
    assert j.wal_id != "Some task title"
    r = j.append("task_start", task="Some task title")
    assert r.wal_id == j.wal_id
    assert j.path == tmp_path / ".pedacito" / "journal" / f"{j.wal_id}.jsonl"
    raw = json.loads(j.path.read_text(encoding="utf-8"))
    assert raw["version"] == journal.WAL_VERSION
    assert raw["wal_id"] == j.wal_id
    assert raw["task_id"] == "Some task title"     # no secret, so unchanged


def test_reopen_requires_explicit_wal_id(tmp_path):
    j = journal.TaskJournal(tmp_path, "reopen", session_id="s")
    j.append("task_start")
    j.append("step", n=1)
    # A fresh constructor WITHOUT wal_id is a NEW run, not a reopen.
    other = journal.TaskJournal(tmp_path, "reopen", session_id="s")
    assert other.wal_id != j.wal_id
    assert other.path != j.path
    # Explicit wal_id reopens the SAME run: identity + monotonic seq resume.
    j2 = journal.TaskJournal(tmp_path, "reopen", session_id="s", wal_id=j.wal_id)
    assert j2.wal_id == j.wal_id
    assert j2.path == j.path
    r = j2.append("edit", files=["a.py"])
    assert r.seq == 3
    j2.close()
    recs = journal.read_journal(j.path)
    assert [x.seq for x in recs] == [1, 2, 3, 4]


def test_same_title_runs_are_independent(tmp_path):
    a = journal.TaskJournal(tmp_path, "same", session_id="sa")
    a.append("task_start")
    a.append("checkpoint", checkpoint_id="cp_a", phase="pre")
    b = journal.TaskJournal(tmp_path, "same", session_id="sb")
    b.append("task_start")

    assert a.path != b.path
    assert a.wal_id != b.wal_id

    incomplete = {i["session_id"]: i for i in journal.find_incomplete_tasks(tmp_path)}
    assert set(incomplete) == {"sa", "sb"}
    assert incomplete["sa"]["checkpoint_id"] == "cp_a"
    assert incomplete["sb"]["checkpoint_id"] == ""   # no leak from a


def test_previous_run_checkpoint_does_not_leak_into_new_run(tmp_path):
    a = journal.TaskJournal(tmp_path, "same", session_id="sa")
    a.append("task_start")
    a.append("checkpoint", checkpoint_id="cp_a", phase="pre")
    a.close()
    b = journal.TaskJournal(tmp_path, "same", session_id="sb")
    b.append("task_start")

    result = journal.recover(tmp_path, b.wal_id, action="rollback", dry_run=False)
    assert result["ok"] is False
    assert "pre-run" in result["detail"]


def test_interrupted_run_reopen_resumes_identity(tmp_path):
    j = journal.TaskJournal(tmp_path, "crash", session_id="s")
    j.append("task_start")
    j.append("checkpoint", checkpoint_id="cp_x", phase="pre")
    j2 = journal.TaskJournal(tmp_path, "crash", session_id="s", wal_id=j.wal_id)
    assert j2.wal_id == j.wal_id
    assert j2._seq == 2
    j2.close()
    recs = journal.read_journal(j.path)
    assert [x.seq for x in recs] == [1, 2, 3]


# ----------------------------------------------------------------- durability


def test_fsync_failure_surfaces_and_disables_recoverability(tmp_path,
                                                            monkeypatch):
    j = journal.TaskJournal(tmp_path, "fsync", session_id="s")
    j.append("task_start")

    def boom(fh):
        raise OSError("disk full")

    monkeypatch.setattr(journal, "_fsync", boom)
    j.append("checkpoint", checkpoint_id="cp_x", phase="pre")

    assert j.durable is False
    assert "disk full" in j.error

    cands = journal.list_recovery(tmp_path)
    assert len(cands) == 1
    assert cands[0].status == "fsync_error"
    assert cands[0].resumable is False


def test_fsync_disabled_is_not_durable(tmp_path):
    j = journal.TaskJournal(tmp_path, "t", fsync=False)
    assert j.durable is False
    j.append("step", n=1)
    j.close()
    assert j.durable is False


def test_disabled_journal_is_not_durable():
    j = journal.TaskJournal(None, "t")
    assert j.durable is False
    j.append("task_start")
    j.close()
    assert j.durable is False


def test_fsync_disabled_records_carry_non_durable_flag(tmp_path):
    j = journal.TaskJournal(tmp_path, "t", fsync=False)
    j.append("step", n=1)
    raw = json.loads(j.path.read_text(encoding="utf-8"))
    assert raw["durable"] is False


def test_fsynced_records_carry_durable_flag(tmp_path):
    j = journal.TaskJournal(tmp_path, "t")
    j.append("step", n=1)
    raw = json.loads(j.path.read_text(encoding="utf-8"))
    assert raw["durable"] is True


def test_non_durable_journal_not_clean_terminal_after_restart(tmp_path):
    j = journal.TaskJournal(tmp_path, "t", session_id="s", fsync=False)
    j.append("task_start")
    j.close()                               # task_end, but records not fsynced
    # "restart": a fresh scanner reads the persisted file.
    cands = journal.list_recovery(tmp_path)
    assert len(cands) == 1
    assert cands[0].status == "non_durable"
    assert cands[0].resumable is False
    assert len(journal.find_incomplete_tasks(tmp_path)) == 1


def test_non_durable_journal_refuses_rollback(tmp_path):
    j = journal.TaskJournal(tmp_path, "t", session_id="s", fsync=False)
    j.append("task_start")
    j.append("checkpoint", checkpoint_id="cp_x", phase="pre")
    result = journal.recover(tmp_path, j.wal_id, action="rollback", dry_run=False)
    assert result["ok"] is False
    assert "not intact" in result["detail"]


def test_dict_keys_are_redacted(tmp_path):
    j = journal.TaskJournal(tmp_path, "t", session_id="s")
    j.append("model_call", **{"sk-" + "a" * 20: "value", "safe": "keep"})
    raw = j.path.read_text(encoding="utf-8")
    assert "sk-" not in raw
    assert "safe" in raw


def test_kind_is_redacted(tmp_path):
    j = journal.TaskJournal(tmp_path, "t", session_id="s")
    j.append("sk-" + "a" * 20)
    raw = j.path.read_text(encoding="utf-8")
    assert "sk-" not in raw


def test_title_and_error_are_redacted(tmp_path, monkeypatch):
    j = journal.TaskJournal(tmp_path, "fix sk-" + "z" * 20 + " leak",
                            session_id="s")
    j.append("task_start")

    def boom(fh):
        raise OSError("token ghp_" + "q" * 30)

    monkeypatch.setattr(journal, "_fsync", boom)
    j.append("checkpoint", checkpoint_id="cp_x", phase="pre")

    raw = j.path.read_text(encoding="utf-8")
    assert "sk-" not in raw
    assert "ghp_" not in raw


def test_redaction_is_preserved_in_versioned_records(tmp_path):
    j = journal.TaskJournal(tmp_path, "t", session_id="s")
    j.append("model_call", prompt="key is sk-" + "a" * 20,
             token="ghp_" + "b" * 30)
    raw = j.path.read_text(encoding="utf-8")
    assert "sk-" not in raw and "ghp_" not in raw


# ----------------------------------------------------------------- scanning


def test_read_journal_parses_legacy_unversioned_records(tmp_path):
    p = tmp_path / "legacy.jsonl"
    p.write_text(json.dumps({
        "seq": 1, "ts": 1.0, "task_id": "legacy", "session_id": "s",
        "kind": "task_start", "data": {},
    }) + "\n", encoding="utf-8")
    recs = journal.read_journal(p)
    assert len(recs) == 1
    assert recs[0].wal_id == ""
    assert recs[0].kind == "task_start"


def test_torn_trailing_line_marks_candidate_not_resumable(tmp_path):
    j = journal.TaskJournal(tmp_path, "torn", session_id="s")
    j.append("task_start")
    j.append("checkpoint", checkpoint_id="cp_x", phase="pre")
    with j.path.open("a", encoding="utf-8") as fh:
        fh.write('{"seq": 9, "kind": "edit", "data"')  # torn trailing line
    cands = journal.list_recovery(tmp_path)
    assert len(cands) == 1
    assert cands[0].status == "torn"
    assert cands[0].resumable is False


def test_valid_json_without_newline_is_torn(tmp_path):
    p = tmp_path / "j.jsonl"
    rec = lambda s: json.dumps({"seq": s, "ts": float(s), "task_id": "t",
                                "session_id": "s", "kind": "step", "data": {}})
    p.write_text(rec(1) + "\n" + rec(2), encoding="utf-8")  # last line no newline
    records, torn, corrupt, non_durable = journal._read_journal_with_status(p)
    assert [r.seq for r in records] == [1]
    assert torn is True
    assert corrupt is False


def test_corrupt_non_trailing_record_fails_closed(tmp_path):
    j = journal.TaskJournal(tmp_path, "corrupt", session_id="s")
    j.append("task_start")
    j.append("checkpoint", checkpoint_id="cp_x", phase="pre")
    lines = j.path.read_text(encoding="utf-8").splitlines()
    lines.insert(1, "{not json")           # corrupt line in the MIDDLE
    j.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cands = journal.list_recovery(tmp_path)
    assert len(cands) == 1
    assert cands[0].status == "corrupt"

    result = journal.recover(tmp_path, j.wal_id, action="rollback", dry_run=False)
    assert result["ok"] is False
    assert "not intact" in result["detail"]


def test_malformed_seq_is_corrupt(tmp_path):
    p = tmp_path / "j.jsonl"
    p.write_text(json.dumps({"seq": "not-a-number", "ts": 1.0, "task_id": "t",
                             "session_id": "s", "kind": "task_start",
                             "data": {}}) + "\n", encoding="utf-8")
    recs, torn, corrupt, non_durable = journal._read_journal_with_status(p)
    assert recs == []
    assert corrupt is True
    assert torn is False


def test_duplicate_seq_is_corrupt(tmp_path):
    p = tmp_path / "j.jsonl"
    rec = lambda s: json.dumps({"seq": s, "ts": float(s), "task_id": "t",
                                "session_id": "s", "kind": "step", "data": {}})
    p.write_text(rec(1) + "\n" + rec(1) + "\n", encoding="utf-8")  # duplicate
    records, torn, corrupt, non_durable = journal._read_journal_with_status(p)
    assert [r.seq for r in records] == [1]
    assert corrupt is True


def test_wal_id_mismatch_is_corrupt(tmp_path):
    wid = uuid.uuid4().hex
    other = uuid.uuid4().hex
    p = tmp_path / f"{wid}.jsonl"
    p.write_text(json.dumps({"version": 2, "wal_id": other, "seq": 1, "ts": 1.0,
                             "task_id": "t", "session_id": "s",
                             "kind": "task_start", "data": {}}) + "\n",
                 encoding="utf-8")
    records, torn, corrupt, non_durable = journal._read_journal_with_status(p)
    assert records == []
    assert corrupt is True


def _v2_rec(wal_id: str, **overrides):
    rec = {
        "version": journal.WAL_VERSION, "wal_id": wal_id, "seq": 1, "ts": 1.0,
        "task_id": "t", "session_id": "s", "kind": "task_start",
        "durable": True, "data": {},
    }
    rec.update(overrides)
    return rec


def _read_v2(tmp_path, rec):
    wid = uuid.uuid4().hex
    p = tmp_path / f"{wid}.jsonl"
    p.write_text(json.dumps(_v2_rec(wid, **rec)) + "\n", encoding="utf-8")
    return journal._read_journal_with_status(p)


def test_v2_valid_record_parses(tmp_path):
    records, torn, corrupt, non_durable = _read_v2(tmp_path, {})
    assert len(records) == 1
    assert records[0].durable is True
    assert not (torn or corrupt or non_durable)


def test_v2_string_durable_flag_is_corrupt(tmp_path):
    records, torn, corrupt, non_durable = _read_v2(tmp_path, {"durable": "false"})
    assert records == []
    assert corrupt is True


def test_v2_missing_wal_id_is_corrupt(tmp_path):
    wid = uuid.uuid4().hex
    p = tmp_path / f"{wid}.jsonl"
    rec = _v2_rec(wid)
    del rec["wal_id"]
    p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    records, torn, corrupt, non_durable = journal._read_journal_with_status(p)
    assert records == []
    assert corrupt is True


def test_v2_missing_version_is_corrupt(tmp_path):
    wid = uuid.uuid4().hex
    p = tmp_path / f"{wid}.jsonl"
    rec = _v2_rec(wid)
    del rec["version"]
    p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    records, torn, corrupt, non_durable = journal._read_journal_with_status(p)
    assert records == []
    assert corrupt is True


def test_v2_missing_durable_is_corrupt(tmp_path):
    wid = uuid.uuid4().hex
    p = tmp_path / f"{wid}.jsonl"
    rec = _v2_rec(wid)
    del rec["durable"]
    p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    records, torn, corrupt, non_durable = journal._read_journal_with_status(p)
    assert records == []
    assert corrupt is True


def test_v2_string_seq_is_corrupt(tmp_path):
    records, torn, corrupt, non_durable = _read_v2(tmp_path, {"seq": "1"})
    assert records == []
    assert corrupt is True


def test_v2_non_positive_seq_is_corrupt(tmp_path):
    records, torn, corrupt, non_durable = _read_v2(tmp_path, {"seq": 0})
    assert records == []
    assert corrupt is True


def test_v2_non_finite_ts_is_corrupt(tmp_path):
    records, torn, corrupt, non_durable = _read_v2(
        tmp_path, {"ts": float("inf")})
    assert records == []
    assert corrupt is True


def test_uncertain_journal_is_not_hidden(tmp_path):
    j = journal.TaskJournal(tmp_path, "uncertain", session_id="s")
    j.append("task_start")
    j.close()                                # clean terminal so far
    with j.path.open("a", encoding="utf-8") as fh:
        fh.write('{"seq": 9, "kind": "edit", "data"')  # a torn NEW run
    cands = journal.list_recovery(tmp_path)
    assert len(cands) == 1
    assert cands[0].status == "torn"          # not hidden behind task_end


# ----------------------------------------------------------------- candidates


def test_list_recovery_and_candidate_fields(tmp_path):
    done = journal.TaskJournal(tmp_path, "done", session_id="s1")
    done.append("task_start")
    done.close()

    pend = journal.TaskJournal(tmp_path, "pending", session_id="s2")
    pend.append("task_start", task="work")
    pend.append("checkpoint", checkpoint_id="cp_456", phase="pre")

    cands = journal.list_recovery(tmp_path)
    assert [c.task_id for c in cands] == ["pending"]
    c = cands[0]
    assert c.session_id == "s2"
    assert c.checkpoint_id == "cp_456"
    assert c.resumable is False
    assert c.status == "incomplete"
    assert c.journal_path == pend.path


def test_candidate_never_advertises_resume(tmp_path):
    j = journal.TaskJournal(tmp_path, "task", session_id="s")
    j.append("task_start")
    j.append("checkpoint", checkpoint_id="cp_x", phase="pre")
    result = journal.recover(tmp_path, j.wal_id, action="resume")
    assert result["ok"] is False
    assert result["resumable"] is False
    assert "resume is not implemented" in result["reason"]


def test_recover_resume_refuses_without_checkpoint(tmp_path):
    j = journal.TaskJournal(tmp_path, "task", session_id="s")
    j.append("task_start")
    result = journal.recover(tmp_path, j.wal_id, action="resume")
    assert result["ok"] is False
    assert result["resumable"] is False
    assert result["status"] == "no_checkpoint"


# ----------------------------------------------------------------- rollback


def _project(root: Path) -> Path:
    root.mkdir(exist_ok=True)
    return root


def test_recover_rollback_failure_does_not_mark_terminal(tmp_path):
    root = _project(tmp_path / "proj")
    j = journal.TaskJournal(root, "task", session_id="s")
    j.append("task_start")
    j.append("checkpoint", checkpoint_id="does-not-exist", phase="pre")
    result = journal.recover(root, j.wal_id, action="rollback", dry_run=False)
    assert result["ok"] is False
    assert "cannot be verified" in result["detail"]
    assert journal.find_incomplete_tasks(root)
    assert journal.list_recovery(root)


def test_recover_rollback_success_restores_and_marks_terminal(tmp_path):
    root = _project(tmp_path / "proj")
    (root / "a.py").write_text("original", encoding="utf-8")
    j = journal.TaskJournal(root, "task", session_id="s")
    cp = checkpoints.create_checkpoint(root, "baseline", wal_id=j.wal_id)
    (root / "a.py").write_text("modified", encoding="utf-8")
    j.append("task_start")
    j.append("checkpoint", checkpoint_id=cp["id"], phase="pre")

    result = journal.recover(root, j.wal_id, action="rollback", dry_run=False)
    assert result["ok"] is True
    assert result["restored_checkpoint"] == cp["id"]
    assert (root / "a.py").read_text(encoding="utf-8") == "original"
    assert journal.list_recovery(root) == []
    assert journal.find_incomplete_tasks(root) == []


def test_recover_rollback_refuses_when_project_locked(tmp_path):
    root = _project(tmp_path / "proj")
    j = journal.TaskJournal(root, "task", session_id="s")
    j.append("task_start")
    j.append("checkpoint", checkpoint_id="cp_x", phase="pre")
    workspace.acquire_lock(root, session_id="other")
    try:
        result = journal.recover(root, j.wal_id, action="rollback", dry_run=False)
        assert result["ok"] is False
        assert "lock" in result["detail"]
    finally:
        workspace.release_lock(root)


def test_recover_rollback_refuses_on_torn_journal(tmp_path):
    j = journal.TaskJournal(tmp_path, "task", session_id="s")
    j.append("task_start")
    j.append("checkpoint", checkpoint_id="cp_x", phase="pre")
    with j.path.open("a", encoding="utf-8") as fh:
        fh.write('{"seq": 9, "kind": "edit", "data"')
    result = journal.recover(tmp_path, j.wal_id, action="rollback", dry_run=False)
    assert result["ok"] is False
    assert "not intact" in result["detail"]


def test_recover_unknown_action_refused(tmp_path):
    j = journal.TaskJournal(tmp_path, "task", session_id="s")
    j.append("task_start")
    result = journal.recover(tmp_path, "task", action="explode")
    assert result["ok"] is False
    assert "unknown recovery action" in result["detail"]


def test_rollback_restores_pre_run_not_post_run(tmp_path):
    root = _project(tmp_path / "proj")
    (root / "a.py").write_text("v0", encoding="utf-8")
    j = journal.TaskJournal(root, "task", session_id="s")
    cp_pre = checkpoints.create_checkpoint(root, "pre", wal_id=j.wal_id)
    (root / "a.py").write_text("v1", encoding="utf-8")
    cp_post = checkpoints.create_checkpoint(root, "post", wal_id=j.wal_id)
    j.append("task_start")
    j.append("checkpoint", checkpoint_id=cp_pre["id"], phase="pre")
    j.append("checkpoint", checkpoint_id=cp_post["id"], phase="post")

    result = journal.recover(root, j.wal_id, action="rollback", dry_run=False)
    assert result["ok"] is True
    assert result["restored_checkpoint"] == cp_pre["id"]
    assert (root / "a.py").read_text(encoding="utf-8") == "v0"


def test_rollback_refuses_without_pre_run_boundary(tmp_path):
    root = _project(tmp_path / "proj")
    (root / "a.py").write_text("v0", encoding="utf-8")
    j = journal.TaskJournal(root, "task", session_id="s")
    cp = checkpoints.create_checkpoint(root, "baseline", wal_id=j.wal_id)
    j.append("task_start")
    j.append("checkpoint", checkpoint_id=cp["id"])   # no phase -> not a boundary
    result = journal.recover(root, j.wal_id, action="rollback", dry_run=False)
    assert result["ok"] is False
    assert "pre-run" in result["detail"]


def test_rollback_refuses_ambiguous_pre_run(tmp_path):
    j = journal.TaskJournal(tmp_path, "task", session_id="s")
    j.append("task_start")
    j.append("checkpoint", checkpoint_id="cp_a", phase="pre")
    j.append("checkpoint", checkpoint_id="cp_b", phase="pre")
    result = journal.recover(tmp_path, j.wal_id, action="rollback", dry_run=False)
    assert result["ok"] is False
    assert "ambiguous" in result["detail"]


def test_rollback_refuses_foreign_checkpoint(tmp_path):
    """A foreign task's checkpoint with a valid pre phase/mode is refused."""
    root = _project(tmp_path / "proj")
    (root / "a.py").write_text("v0", encoding="utf-8")
    cp = checkpoints.create_checkpoint(root, "foreign", mode="task",
                                       wal_id=uuid.uuid4().hex)
    j = journal.TaskJournal(root, "task", session_id="s")
    j.append("task_start")
    j.append("checkpoint", checkpoint_id=cp["id"], phase="pre")
    result = journal.recover(root, j.wal_id, action="rollback", dry_run=False)
    assert result["ok"] is False
    assert "not bound" in result["detail"]


def test_rollback_refuses_unbound_checkpoint(tmp_path):
    root = _project(tmp_path / "proj")
    (root / "a.py").write_text("v0", encoding="utf-8")
    cp = checkpoints.create_checkpoint(root, "legacy", wal_id="")
    j = journal.TaskJournal(root, "task", session_id="s")
    j.append("task_start")
    j.append("checkpoint", checkpoint_id=cp["id"], phase="pre")
    result = journal.recover(root, j.wal_id, action="rollback", dry_run=False)
    assert result["ok"] is False
    assert "not bound" in result["detail"]


def test_recover_legacy_title_named_wal(tmp_path):
    """A pre-v0.4.1 title-named WAL is readable but not rollback-safe."""
    p = tmp_path / ".pedacito" / "journal"
    p.mkdir(parents=True)
    legacy = p / "legacy-task.jsonl"
    legacy.write_text(
        json.dumps({"seq": 1, "ts": 1.0, "task_id": "legacy-task",
                    "session_id": "s", "kind": "task_start", "data": {}})
        + "\n"
        + json.dumps({"seq": 2, "ts": 2.0, "task_id": "legacy-task",
                      "session_id": "s", "kind": "checkpoint",
                      "data": {"checkpoint_id": "cp_legacy"}})
        + "\n", encoding="utf-8")

    assert journal._find_journal_path(tmp_path, "legacy-task") == legacy
    result = journal.recover(tmp_path, "legacy-task", action="rollback",
                             dry_run=False)
    assert result["ok"] is False
    assert "pre-run" in result["detail"]


# ----------------------------------------------------------------- cleanup


def test_recover_cleanup_refuses_without_authoritative_liveness(tmp_path):
    root = _project(tmp_path / "proj")
    j = journal.TaskJournal(root, "task", session_id="s")
    j.append("task_start")
    j.append("checkpoint", worktree=str(root / "wt"))
    result = journal.recover(root, j.wal_id, action="cleanup", dry_run=False)
    assert result["ok"] is False
    assert "authoritative" in result["detail"]
    assert "liveness" in result["detail"]


def test_remove_recorded_worktree_refuses_foreign_path(tmp_path):
    root = _project(tmp_path / "proj")
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    ok, detail = worktree_mod.remove_recorded_worktree(root, foreign)
    assert ok is False
    assert "outside" in detail
    assert foreign.exists()


def test_remove_recorded_worktree_refuses_without_ownership_marker(tmp_path):
    root = _project(tmp_path / "proj")
    fake = root / ".pedacito" / "worktrees" / "not-owned"
    fake.mkdir(parents=True)
    ok, detail = worktree_mod.remove_recorded_worktree(root, fake)
    assert ok is False
    assert "ownership marker" in detail
    assert fake.exists()


def test_remove_recorded_worktree_refuses_link_base(tmp_path, monkeypatch):
    root = _project(tmp_path / "proj")
    base = root / ".pedacito" / "worktrees"
    wt = base / "owned"
    (wt / ".pedacito" / "sessions_state" / "ses_x").mkdir(parents=True)

    from pedacito import security as security_mod
    monkeypatch.setattr(security_mod, "is_junction_or_reparse",
                        lambda p: Path(p) == base)
    ok, detail = worktree_mod.remove_recorded_worktree(root, wt)
    assert ok is False
    assert "base" in detail
    assert wt.exists()


def test_remove_recorded_worktree_refuses_owned_but_unproven(tmp_path):
    """Even a properly-owned worktree is refused: no authoritative cross-process
    liveness proof exists to establish it is orphaned (sibling/foreign process)."""
    root = _project(tmp_path / "proj")
    wt = root / ".pedacito" / "worktrees" / "owned"
    (wt / ".pedacito" / "sessions_state" / "ses_x").mkdir(parents=True)
    ok, detail = worktree_mod.remove_recorded_worktree(root, wt)
    assert ok is False
    assert "authoritative" in detail
    assert wt.exists()


def test_recovery_lock_is_a_cross_process_claim(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    with workspace.recovery_lock(root):
        with pytest.raises(workspace.ProjectBusy):
            workspace.acquire_lock(root, session_id="other")
    workspace.acquire_lock(root, session_id="other")
    workspace.release_lock(root)


# ----------------------------------------------------------------- git-backed
# A real Pedacito-owned worktree is still refused for lack of a liveness proof.

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


_pytest_git = pytest.mark.skipif(not shutil.which("git"), reason="git not installed")


@_pytest_git
def test_remove_recorded_worktree_refuses_even_owned_worktree(tmp_path):
    root = _repo(tmp_path)
    h = worktree_mod.create_session_worktree(root, "t", session_id="ses_dead")
    try:
        ok, detail = worktree_mod.remove_recorded_worktree(root, h.path)
        assert ok is False
        assert "authoritative" in detail
        assert h.path.exists()
    finally:
        worktree_mod.remove_session_worktree(h, force=True, delete_branch=True)
