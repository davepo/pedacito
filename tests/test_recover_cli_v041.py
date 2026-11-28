"""WP-4105B -- the user-facing ``pdc recover`` commands.

Covers the v0.4.1 recovery surface (SS3.7): read-only listing by UUID,
conservative legacy-title visibility, preview-by-default mutations, localized
EN/ES output, actionable refusals, project/maintenance locking and a safely
bound rollback that never deletes user work.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import checkpoints, cli, i18n, journal, workspace


@pytest.fixture(autouse=True)
def _isolated_recover_cli(monkeypatch, tmp_path):
    """Deterministic language and no user config or daemon dependency."""
    monkeypatch.setattr(cli, "USER_CONFIG", tmp_path / "no-user-config.toml")
    for name in ("PEDACITO_LANG", "PEDACITO_LANGUAGE"):
        monkeypatch.delenv(name, raising=False)
    # The CLI must consult this gate; the real one may probe a user daemon.
    monkeypatch.setattr(workspace, "check_maintenance_safe",
                        lambda root, **kw: (True, ""))
    i18n.set_current_lang("en")
    yield
    i18n.set_current_lang("en")


def _candidate(tmp_path, title="fix parser", session="ses_1", *,
               checkpoint=True, phase="pre", checkpoint_wal=None):
    root = tmp_path / "proj"
    root.mkdir()
    source = root / "a.py"
    source.write_text("original\n", encoding="utf-8")
    task = journal.TaskJournal(root, title, session_id=session)
    task.append("task_start", task=title)
    cp = None
    if checkpoint:
        cp = checkpoints.create_checkpoint(
            root, "baseline",
            wal_id=task.wal_id if checkpoint_wal is None else checkpoint_wal)
        task.append("checkpoint", checkpoint_id=cp["id"], phase=phase)
    source.write_text("modified\n", encoding="utf-8")
    return root, task, cp


def _journal_count(root: Path) -> int:
    return len(journal.list_recovery(root))


def _checkpoint_count(root: Path) -> int:
    return len(checkpoints.list_checkpoints(root))


# --------------------------------------------------------------------- list


def test_list_empty_is_read_only_and_needs_no_state(tmp_path, capsys):
    root = tmp_path / "proj"
    root.mkdir()

    assert cli.main(["recover", "list", str(root)]) == 0
    out = capsys.readouterr().out
    assert "No incomplete or uncertain task journals" in out
    assert not (root / ".pedacito").exists()


def test_list_shows_uuid_title_session_status_and_checkpoint(tmp_path, capsys):
    root, task, cp = _candidate(tmp_path)
    checkpoints_before = _checkpoint_count(root)

    assert cli.main(["recover", "list", str(root)]) == 0
    out = capsys.readouterr().out
    assert task.wal_id in out
    assert "fix parser" in out
    assert "ses_1" in out
    assert cp["id"] in out
    assert f".pedacito/journal/{task.wal_id}.jsonl" in out
    assert "rollback: preview available" in out
    assert "resume: unavailable" in out
    # Listing (and its rollback readiness preview) creates no checkpoint.
    assert _checkpoint_count(root) == checkpoints_before
    assert _journal_count(root) == 1


def test_list_root_flag_and_default_project(tmp_path, monkeypatch, capsys):
    root, task, _cp = _candidate(tmp_path)
    assert cli.main(["recover", "list", "--root", str(root)]) == 0
    assert task.wal_id in capsys.readouterr().out

    monkeypatch.chdir(root)
    assert cli.main(["recover", "list"]) == 0
    assert task.wal_id in capsys.readouterr().out


def test_list_marks_legacy_title_journals_read_only(tmp_path, capsys):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("original\n", encoding="utf-8")
    cp = checkpoints.create_checkpoint(root, "legacy", wal_id="")
    jdir = root / ".pedacito" / "journal"
    jdir.mkdir(parents=True)
    legacy = jdir / "legacy-task.jsonl"
    legacy.write_text(
        json.dumps({"seq": 1, "ts": 1.0, "task_id": "legacy task",
                    "session_id": "ses_old", "kind": "task_start",
                    "data": {}}) + "\n"
        + json.dumps({"seq": 2, "ts": 2.0, "task_id": "legacy task",
                      "session_id": "ses_old", "kind": "checkpoint",
                      "data": {"checkpoint_id": cp["id"],
                               "phase": "pre"}}) + "\n",
        encoding="utf-8")

    assert cli.main(["recover", "list", str(root)]) == 0
    out = capsys.readouterr().out
    assert "legacy title journal" in out
    assert "legacy task" in out
    assert "legacy journals are visible for inspection only" in out
    assert "rollback: unavailable" in out
    # A legacy title can still be addressed, but it cannot be mutated safely.
    assert cli.main(["recover", "rollback", "legacy task", str(root),
                     "--apply"]) == 1
    err = capsys.readouterr().err
    assert "not bound" in err
    assert (root / "a.py").read_text(encoding="utf-8") == "original\n"
    assert _journal_count(root) == 1


def test_list_reports_torn_journals_and_refuses_rollback(tmp_path, capsys):
    root, task, _cp = _candidate(tmp_path)
    with task.path.open("a", encoding="utf-8") as fh:
        fh.write('{"seq": 9, "kind": "edit", "data"')

    assert cli.main(["recover", "list", str(root)]) == 0
    out = capsys.readouterr().out
    assert "torn trailing record" in out
    assert "journal is not intact" in out

    assert cli.main(["recover", "rollback", task.wal_id, str(root)]) == 1
    assert "not intact" in capsys.readouterr().err
    assert (root / "a.py").read_text(encoding="utf-8") == "modified\n"


# --------------------------------------------------------------- refusals


def test_resume_and_cleanup_refuse_without_writes(tmp_path, capsys):
    root, task, _cp = _candidate(tmp_path)
    before = (root / "a.py").read_bytes()
    journals = _journal_count(root)
    checkpoints_before = _checkpoint_count(root)

    assert cli.main(["recover", "resume", task.wal_id, str(root)]) == 1
    err = capsys.readouterr().err
    assert "Resume refused" in err
    assert "resume cannot be proved" in err
    assert "No files were changed" in err

    assert cli.main(["recover", "cleanup", task.wal_id, str(root),
                     "--apply"]) == 1
    err = capsys.readouterr().err
    assert "Cleanup refused" in err
    assert "ownership or liveness proof" in err
    assert "No files were changed" in err

    assert (root / "a.py").read_bytes() == before
    assert _journal_count(root) == journals
    assert _checkpoint_count(root) == checkpoints_before


def test_rollback_refuses_without_a_pre_run_boundary(tmp_path, capsys):
    root, task, _cp = _candidate(tmp_path, checkpoint=False)

    assert cli.main(["recover", "rollback", task.wal_id, str(root)]) == 1
    err = capsys.readouterr().err
    assert "pre-run boundary" in err
    assert "No files were changed" in err
    assert (root / "a.py").read_text(encoding="utf-8") == "modified\n"
    assert _journal_count(root) == 1


def test_rollback_refuses_unbound_or_foreign_checkpoint(tmp_path, capsys):
    root, task, _cp = _candidate(
        tmp_path, checkpoint_wal=uuid.uuid4().hex)

    assert cli.main(["recover", "rollback", task.wal_id, str(root)]) == 1
    assert "not bound" in capsys.readouterr().err
    assert cli.main(["recover", "rollback", task.wal_id, str(root),
                     "--apply"]) == 1
    assert "not bound" in capsys.readouterr().err
    assert (root / "a.py").read_text(encoding="utf-8") == "modified\n"
    assert _journal_count(root) == 1


def test_rollback_refuses_a_missing_snapshot_without_deleting_files(
        tmp_path, capsys):
    root, task, cp = _candidate(tmp_path)
    shutil.rmtree(checkpoints.snapshot_dir(root, cp["id"]))

    assert cli.main(["recover", "rollback", task.wal_id, str(root)]) == 1
    err = capsys.readouterr().err
    assert "cannot be verified" in err
    assert "snapshot directory is missing" in err
    assert cli.main(["recover", "rollback", task.wal_id, str(root),
                     "--apply"]) == 1
    err = capsys.readouterr().err
    assert "cannot be verified" in err
    assert "snapshot directory is missing" in err
    # A naive restore would treat every current file as deleted.
    assert (root / "a.py").read_text(encoding="utf-8") == "modified\n"
    assert _journal_count(root) == 1


def test_unknown_task_is_actionable_and_writes_nothing(tmp_path, capsys):
    root = tmp_path / "proj"
    root.mkdir()

    assert cli.main(["recover", "rollback", "no-such-task",
                     str(root)]) == 1
    err = capsys.readouterr().err
    assert "no journal matched this identity" in err
    assert "pedacito recover list" in err
    assert not (root / ".pedacito").exists()


def test_missing_project_is_a_usage_error(tmp_path, capsys):
    missing = tmp_path / "absent"
    assert cli.main(["recover", "list", str(missing)]) == 2
    assert "project directory not found" in capsys.readouterr().err


# --------------------------------------------------------------- preview/apply


def test_rollback_preview_writes_nothing(tmp_path, capsys):
    root, task, cp = _candidate(tmp_path)
    before = (root / "a.py").read_bytes()
    journal_before = task.path.read_bytes()
    checkpoints_before = _checkpoint_count(root)

    assert cli.main(["recover", "rollback", task.wal_id, str(root)]) == 0
    out = capsys.readouterr().out
    assert "Rollback preview" in out
    assert cp["id"] in out
    assert "--apply" in out

    assert (root / "a.py").read_bytes() == before
    assert task.path.read_bytes() == journal_before
    assert _checkpoint_count(root) == checkpoints_before
    assert _journal_count(root) == 1


def test_rollback_apply_restores_boundary_and_snapshots_dirty_state(
        tmp_path, capsys):
    root, task, cp = _candidate(tmp_path)

    assert cli.main(["recover", "rollback", task.wal_id, str(root),
                     "--apply"]) == 0
    out = capsys.readouterr().out
    assert "Rolled back" in out
    assert cp["id"] in out
    assert "Safety snapshot" in out

    assert (root / "a.py").read_text(encoding="utf-8") == "original\n"
    assert _journal_count(root) == 0
    entries = checkpoints.list_checkpoints(root)
    modes = [entry.get("mode") for entry in entries]
    assert "restore" in modes
    for entry in entries:
        manifest = checkpoints.snapshot_manifest_path(root, entry["id"])
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        assert payload["complete"] is True
        assert payload["count"] == len(payload["files"])


def test_rollback_preview_survives_a_live_lock_but_apply_refuses(
        tmp_path, capsys):
    root, task, _cp = _candidate(tmp_path)
    before = (root / "a.py").read_bytes()
    workspace.acquire_lock(root, session_id="other")
    try:
        assert cli.main(["recover", "rollback", task.wal_id, str(root)]) == 0
        assert "Rollback preview" in capsys.readouterr().out

        assert cli.main(["recover", "rollback", task.wal_id, str(root),
                         "--apply"]) == 1
        err = capsys.readouterr().err
        assert "write lock" in err
        assert "No files were changed" in err
        assert (root / "a.py").read_bytes() == before
        assert _journal_count(root) == 1
    finally:
        workspace.release_lock(root)


def test_apply_obeys_the_maintenance_gate(tmp_path, monkeypatch, capsys):
    root, task, _cp = _candidate(tmp_path)
    before = (root / "a.py").read_bytes()
    calls = []

    def blocked(root_path, **kwargs):
        calls.append((root_path, kwargs))
        return False, "another pedacito run is using this project"

    monkeypatch.setattr(workspace, "check_maintenance_safe", blocked)

    assert cli.main(["recover", "rollback", task.wal_id, str(root),
                     "--apply"]) == 1
    err = capsys.readouterr().err
    assert "maintenance safety check" in err
    assert "another pedacito run" in err
    assert calls and calls[0][1]["action"] == "recover rollback"
    assert (root / "a.py").read_bytes() == before
    assert _journal_count(root) == 1


def test_hyphenated_uuid_addresses_the_same_journal(tmp_path, capsys):
    root, task, _cp = _candidate(tmp_path)
    hyphenated = str(uuid.UUID(task.wal_id))

    assert cli.main(["recover", "resume", hyphenated, str(root)]) == 1
    err = capsys.readouterr().err
    assert "no journal matched" not in err
    assert "resume cannot be proved" in err


# --------------------------------------------------------------- localized


def test_spanish_list_and_refusals_are_localized(tmp_path, capsys):
    root, task, cp = _candidate(tmp_path)

    assert cli.main(["recover", "list", str(root), "--lang", "es"]) == 0
    out = capsys.readouterr().out
    assert "Candidatos de recuperación" in out
    assert "incompleta" in out
    assert "reanudar: no disponible" in out
    assert "reversión: vista previa disponible" in out
    assert cp["id"] in out

    assert cli.main(["--lang", "es", "recover", "resume", task.wal_id,
                     str(root)]) == 1
    err = capsys.readouterr().err
    assert "Reanudación rechazada" in err
    assert "No se cambió ningún archivo" in err

    assert cli.main(["--lang", "es", "recover", "cleanup", task.wal_id,
                     str(root)]) == 1
    err = capsys.readouterr().err
    assert "Limpieza rechazada" in err
    assert "propiedad o vivacidad" in err
    assert (root / "a.py").read_text(encoding="utf-8") == "modified\n"


# ----------------------------------------------------- snapshot integrity


def _assert_refused_without_mutation(root, task, capsys, needle):
    head_before = checkpoints.head_id(root)
    before = (root / "a.py").read_bytes()
    assert cli.main(["recover", "rollback", task.wal_id, str(root),
                     "--apply"]) == 1
    err = capsys.readouterr().err
    assert needle in err
    assert (root / "a.py").read_bytes() == before
    assert checkpoints.head_id(root) == head_before
    assert _journal_count(root) == 1


def test_rollback_refuses_an_absent_manifest(tmp_path, capsys):
    root, task, cp = _candidate(tmp_path)
    checkpoints.snapshot_manifest_path(root, cp["id"]).unlink()

    _assert_refused_without_mutation(root, task, capsys,
                                     "manifest is missing")


def test_legacy_checkpoint_without_manifest_still_restores_normally(
        tmp_path):
    root, _task, cp = _candidate(tmp_path)
    checkpoints.snapshot_manifest_path(root, cp["id"]).unlink()

    result = checkpoints.restore_checkpoint(root, cp["id"])

    assert result["ok"] is True
    assert (root / "a.py").read_text(encoding="utf-8") == "original\n"


def test_rollback_refuses_an_incomplete_manifest(tmp_path, capsys):
    root, task, cp = _candidate(tmp_path)
    path = checkpoints.snapshot_manifest_path(root, cp["id"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["complete"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")

    _assert_refused_without_mutation(root, task, capsys,
                                     "not marked complete")


def test_rollback_refuses_an_altered_manifest(tmp_path, capsys):
    root, task, cp = _candidate(tmp_path)
    path = checkpoints.snapshot_manifest_path(root, cp["id"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["files"]["a.py"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    _assert_refused_without_mutation(root, task, capsys,
                                     "hash does not match")


def test_rollback_refuses_a_missing_snapshot_file(tmp_path, capsys):
    root, task, cp = _candidate(tmp_path)
    (checkpoints.snapshot_dir(root, cp["id"]) / "a.py").unlink()

    _assert_refused_without_mutation(root, task, capsys,
                                     "missing manifest file")


def test_rollback_refuses_a_changed_snapshot_file(tmp_path, capsys):
    root, task, cp = _candidate(tmp_path)
    (checkpoints.snapshot_dir(root, cp["id"]) / "a.py").write_text(
        "tampered\n", encoding="utf-8")

    _assert_refused_without_mutation(root, task, capsys,
                                     "hash does not match")


def test_rollback_refuses_a_snapshot_symlink_escape(
        tmp_path, monkeypatch, capsys):
    root, task, cp = _candidate(tmp_path)
    snapshot = checkpoints.snapshot_dir(root, cp["id"])
    target = snapshot / "a.py"
    outside = tmp_path / "outside.txt"
    outside.write_text("original\n", encoding="utf-8")
    target.unlink()
    try:
        os.symlink(outside, target)
    except OSError:
        # Windows without symlink privilege: keep the same bytes and mark the
        # path as a reparse point so the verifier must still reject it.
        target.write_text("original\n", encoding="utf-8")
        real = checkpoints.security.is_junction_or_reparse
        monkeypatch.setattr(
            checkpoints.security, "is_junction_or_reparse",
            lambda path: Path(path) == target or real(path))

    _assert_refused_without_mutation(root, task, capsys, "unsafe")
    assert outside.read_text(encoding="utf-8") == "original\n"


def test_pre_run_copy_failure_records_no_complete_checkpoint(
        tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("a\n", encoding="utf-8")
    (root / "b.py").write_text("b\n", encoding="utf-8")
    real_copy = checkpoints.fileio.copy_file_atomic

    def fail_one(src, dest, **kwargs):
        if Path(src).name == "b.py":
            raise OSError("disk full")
        return real_copy(src, dest, **kwargs)

    monkeypatch.setattr(checkpoints.fileio, "copy_file_atomic", fail_one)
    entry = checkpoints.ensure_pre_run_checkpoint(
        root, "pre-run", wal_id=uuid.uuid4().hex)

    assert entry is None
    assert checkpoints.list_checkpoints(root) == []
    assert checkpoints.head_id(root) == ""
    assert not list((checkpoints.checkpoint_root(root)
                     / "snapshots").glob("*"))


def test_genuine_empty_snapshot_is_allowed(tmp_path, capsys):
    root = tmp_path / "proj"
    root.mkdir()
    task = journal.TaskJournal(root, "empty task", session_id="ses_empty")
    task.append("task_start")
    cp = checkpoints.create_checkpoint(root, "empty", wal_id=task.wal_id)
    assert cp["snapshot_file_count"] == 0
    assert checkpoints.verify_recovery_snapshot(
        root, cp["id"], task.wal_id) == (True, "")
    unexpected = checkpoints.snapshot_dir(root, cp["id"]) / "unexpected.py"
    unexpected.write_text("not in the manifest\n", encoding="utf-8")
    ok, reason = checkpoints.verify_recovery_snapshot(
        root, cp["id"], task.wal_id)
    assert ok is False
    assert "absent from the manifest" in reason
    unexpected.unlink()
    task.append("checkpoint", checkpoint_id=cp["id"], phase="pre")

    (root / "later.py").write_text("created later\n", encoding="utf-8")
    assert cli.main(["recover", "rollback", task.wal_id, str(root)]) == 0
    assert "Rollback preview" in capsys.readouterr().out
    assert (root / "later.py").exists()

    assert cli.main(["recover", "rollback", task.wal_id, str(root),
                     "--apply"]) == 0
    assert not (root / "later.py").exists()
    assert _journal_count(root) == 0


def test_failed_safety_snapshot_aborts_before_restoring(
        tmp_path, monkeypatch, capsys):
    root, task, _cp = _candidate(tmp_path)
    head_before = checkpoints.head_id(root)
    real_create = checkpoints.create_checkpoint

    def fail_safety(root_path, label, mode="task", **kwargs):
        if mode == "restore":
            raise checkpoints.CheckpointError("disk full")
        return real_create(root_path, label, mode=mode, **kwargs)

    monkeypatch.setattr(checkpoints, "create_checkpoint", fail_safety)

    assert cli.main(["recover", "rollback", task.wal_id, str(root),
                     "--apply"]) == 1
    err = capsys.readouterr().err
    assert "did not complete" in err
    assert (root / "a.py").read_text(encoding="utf-8") == "modified\n"
    assert checkpoints.head_id(root) == head_before
    assert _journal_count(root) == 1


def test_uncertain_task_end_is_not_reported_terminal(tmp_path, capsys):
    root, task, _cp = _candidate(tmp_path)
    task.close("completed")
    with task.path.open("a", encoding="utf-8") as fh:
        fh.write('{"seq": 99, "kind": "edit", "data"')

    result = journal.recover(root, task.wal_id, action="rollback",
                             dry_run=True)
    assert result["ok"] is False
    assert result["code"] == "journal_not_intact"
    assert result["detail"].startswith("refusing to treat this task")

    assert cli.main(["recover", "rollback", task.wal_id, str(root)]) == 1
    err = capsys.readouterr().err
    assert "not intact" in err
    assert "already terminal" not in err
    assert (root / "a.py").read_text(encoding="utf-8") == "modified\n"
