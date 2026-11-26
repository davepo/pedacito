"""Durable write-ahead task journal (D7): append/redact/fsync, truncation
tolerance, incomplete-task discovery, crash recovery, and stale-lock takeover.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import journal
from pedacito import workspace   # noqa: F401  (imports requests at collection time)


def test_append_seq_redaction_and_close(tmp_path):
    j = journal.TaskJournal(tmp_path, "task1", session_id="ses_a")
    r1 = j.append("task_start", task="do the thing")
    r2 = j.append("step", n=1)
    assert r1.seq == 1 and r2.seq == 2
    assert r1.task_id == "task1" and r1.session_id == "ses_a"
    assert r1.kind == "task_start"

    # Secrets are redacted before they are persisted.
    secret = j.append("model_call",
                      prompt="key is sk-" + "a" * 20,
                      token="ghp_" + "b" * 30,
                      aws="AKIA" + "C" * 16,
                      safe="keep me")
    for bad in ("sk-", "ghp_", "AKIA"):
        assert bad not in json.dumps(secret.data)
    assert secret.data["safe"] == "keep me"

    j.close(status="completed", ok=True)
    recs = journal.read_journal(j.path)
    assert [r.kind for r in recs] == ["task_start", "step", "model_call",
                                      "task_end"]
    assert recs[-1].data["status"] == "completed"
    assert recs[-1].data["ok"] is True

    # The journal file lives under .pedacito/journal/<safe_id>.jsonl.
    assert j.path == tmp_path / ".pedacito" / "journal" / "task1.jsonl"
    # close() is idempotent: a second close does not append another task_end.
    j.close(status="completed")
    assert len(journal.read_journal(j.path)) == 4


def test_append_without_fsync_does_not_crash(tmp_path):
    j = journal.TaskJournal(tmp_path, "t", fsync=False)
    j.append("step", n=1)
    j.close()
    assert len(journal.read_journal(j.path)) == 2


def test_read_journal_tolerates_truncated_trailing_line(tmp_path):
    p = tmp_path / "j.jsonl"
    good = json.dumps({"seq": 1, "ts": 1.0, "task_id": "t", "session_id": "s",
                       "kind": "task_start", "data": {}})
    p.write_text(good + "\n" + '{"seq":2,"ts":2.0,"kind":"step","data":',
                 encoding="utf-8")            # truncated, no trailing newline
    recs = journal.read_journal(p)
    assert len(recs) == 1
    assert recs[0].kind == "task_start"


def test_read_journal_missing_file_returns_empty(tmp_path):
    assert journal.read_journal(tmp_path / "nope.jsonl") == []


def test_find_incomplete_tasks_summaries(tmp_path):
    j = journal.TaskJournal(tmp_path, "done", session_id="s1")
    j.append("task_start")
    j.append("checkpoint", checkpoint_id="cp_123")
    j.close()

    k = journal.TaskJournal(tmp_path, "pending", session_id="s2")
    k.append("task_start", task="work")
    k.append("step", n=1)
    k.append("checkpoint", checkpoint_id="cp_456")

    incomplete = journal.find_incomplete_tasks(tmp_path)
    ids = {i["task_id"]: i for i in incomplete}
    assert "done" not in ids
    assert "pending" in ids
    pend = ids["pending"]
    assert pend["session_id"] == "s2"
    assert pend["last_kind"] == "checkpoint"
    assert pend["checkpoint_id"] == "cp_456"
    assert pend["path"].endswith("pending.jsonl")


def test_rollback_incomplete_task_appends_rolled_back(tmp_path):
    k = journal.TaskJournal(tmp_path, "pending", session_id="s2")
    k.append("task_start", task="work")
    k.append("checkpoint", checkpoint_id="does-not-exist")   # restore fails cleanly

    result = journal.rollback_incomplete_task(tmp_path, "pending")
    assert result["ok"] is True
    assert result["task_id"] == "pending"
    # The checkpoint id was recorded but restore failed gracefully (best effort).
    assert result["restored_checkpoint"] == ""
    assert "checkpoint restore failed" in result["detail"]

    # The task is no longer flagged incomplete.
    assert journal.find_incomplete_tasks(tmp_path) == []
    recs = journal.read_journal(k.path)
    assert recs[-1].kind == "task_end"
    assert recs[-1].data["status"] == "rolled_back"


def test_crash_recovery_and_stale_lock_takeover(tmp_path):
    """A child writes journal entries, takes the project lock, and is SIGKILLed.
    read_journal / find_incomplete_tasks / rollback_incomplete_task /
    acquire_lock stale-takeover must all then succeed cleanly."""
    repo = Path(__file__).resolve().parents[1]
    proj = tmp_path / "proj"
    proj.mkdir()
    # The child imports ONLY stdlib-only modules (`journal`, `fileio`) so it
    # starts fast and never depends on `requests` being importable under the
    # test's scrubbed environment. It writes the lockfile directly in the same
    # v2 shape `workspace` reads, then sleeps until it is force-killed.
    code = (
        "import sys, time, json, os\n"
        "from pathlib import Path\n"
        "from pedacito import journal, fileio\n"
        f"root = Path({str(proj)!r})\n"
        "j = journal.TaskJournal(root, 'crashy', session_id='ses_crash')\n"
        "j.append('task_start', task='risky work')\n"
        "j.append('step', n=1)\n"
        "j.append('checkpoint', checkpoint_id='cp_fake')\n"
        "lock = root / '.pedacito' / 'run.lock'\n"
        "fileio.write_bytes(lock, json.dumps({'version': 2, 'pid': os.getpid(),"
        " 'start_time': 0.0, 'owner_token': '', 'session_id': 'ses_crash',"
        " 'mode': 'exclusive', 'created_at': 0.0}).encode('utf-8'))\n"
        "print('READY', flush=True)\n"
        "time.sleep(60)\n"
    )
    p = subprocess.Popen(
        [sys.executable, "-c", code], cwd=str(repo),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        line = p.stdout.readline() if p.stdout else ""
        assert "READY" in line
        p.kill()                       # force-kill (SIGKILL / TerminateProcess)
        p.wait(timeout=15)
    finally:
        if p.poll() is None:
            p.kill()
            p.wait(timeout=15)

    path = proj / ".pedacito" / "journal" / "crashy.jsonl"
    recs = journal.read_journal(path)
    assert recs and recs[0].kind == "task_start"
    assert recs[-1].kind == "checkpoint"          # no task_end: it was killed

    incomplete = journal.find_incomplete_tasks(proj)
    assert any(i["task_id"] == "crashy" for i in incomplete)

    # The stale lock left by the dead child is taken over, not refused.
    lock = workspace.acquire_lock(proj, session_id="ses_me")
    assert lock is not None
    workspace.release_lock(proj)

    result = journal.rollback_incomplete_task(proj, "crashy")
    assert result["ok"] is True
    assert journal.find_incomplete_tasks(proj) == []
