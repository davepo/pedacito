"""WP-4105C -- staged all-or-nothing edit transaction and atomic default.

Covers the v0.4.1 transaction contract (SS3.7): the explicit-vs-implicit
``atomic_edits`` precedence, staged preflight before any write, durable WAL
snapshot, preimage hash re-check, rollback on in-process write failure, crash
recovery, BOM/CRLF/mode preservation, symlink/reparse refusal and snapshot
failure stopping the batch.

All tests use local fakes; no model, network, sandbox or subprocess is needed.
"""
from __future__ import annotations

import errno
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import fileio, journal, loop, security
from pedacito import tools as tools_mod
from pedacito import verify as verify_mod
from pedacito.config import Config, atomic_edits_enabled
from pedacito.edits import Result, apply_edits_atomic, parse_edits
from pedacito.workspace_overlay import VirtualWorkspace


def _cfg(**kw) -> Config:
    cfg = Config()
    cfg.backup = False
    cfg.log_events = False
    cfg.log_sessions = False
    cfg.sources = {}
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


# ------------------------------------------------------------------- precedence

def test_explicit_true_always_wins():
    cfg = _cfg(atomic_edits=True)
    assert atomic_edits_enabled(cfg) is True


def test_explicit_false_wins_over_small_tier_default():
    cfg = _cfg(atomic_edits=False, tier="small")
    cfg.sources = {"atomic_edits": "user config (config.toml)"}
    assert atomic_edits_enabled(cfg) is False


def test_implicit_default_is_on_only_for_small_local_tiers():
    for tier in ("tiny", "small", "medium"):
        assert atomic_edits_enabled(_cfg(tier=tier)) is True
    for tier in ("large", "local-moe", "hosted", ""):
        assert atomic_edits_enabled(_cfg(tier=tier)) is False
    # active_tier (profile preset) is read the same way.
    assert atomic_edits_enabled(_cfg(active_tier="small")) is True


def test_implicit_default_is_off_for_hosted_endpoints():
    cfg = _cfg(tier="small", base_url="https://api.openai.com/v1")
    assert atomic_edits_enabled(cfg) is False


def test_default_config_is_legacy():
    assert atomic_edits_enabled(Config()) is False


# --------------------------------------------------------- staged transaction

def test_candidate_failure_writes_nothing(tmp_path):
    _write(tmp_path, "a.py", "X = 1\n")
    _write(tmp_path, "b.py", "Y = 1\n")
    reply = ("--- FILE: a.py\n<<<<<<< SEARCH\nX = 1\n=======\nX = 2\n"
             ">>>>>>> REPLACE\n"
             "--- FILE: b.py\n<<<<<<< SEARCH\nnot there\n=======\nY = 2\n"
             ">>>>>>> REPLACE\n")
    results = apply_edits_atomic(tmp_path, parse_edits(reply), cfg=_cfg())
    assert not any(r.ok for r in results)
    held = [r for r in results if r.file == "a.py"]
    assert held and "atomic_edits" in held[0].message
    assert any(r.file == "b.py" and not r.ok for r in results)
    # Nothing was written.
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "X = 1\n"
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "Y = 1\n"


def test_delete_and_create_in_one_batch(tmp_path):
    _write(tmp_path, "old.py", "OLD = 1\n")
    reply = ("--- FILE: old.py\n<<<<<<< DELETE\n>>>>>>> DELETE\n"
             "--- FILE: new.py\n<<<<<<< SEARCH\n=======\nNEW = 1\n"
             ">>>>>>> REPLACE\n")
    results = apply_edits_atomic(tmp_path, parse_edits(reply), cfg=_cfg())
    assert all(r.ok for r in results)
    assert not (tmp_path / "old.py").exists()
    assert (tmp_path / "new.py").read_text(encoding="utf-8") == "NEW = 1\n"


def test_mixed_old_and_new_files(tmp_path):
    _write(tmp_path, "a.py", "A = 1\n")
    reply = ("--- FILE: a.py\n<<<<<<< SEARCH\nA = 1\n=======\nA = 2\n"
             ">>>>>>> REPLACE\n"
             "--- FILE: b.py\n<<<<<<< SEARCH\n=======\nB = 1\n"
             ">>>>>>> REPLACE\n")
    results = apply_edits_atomic(tmp_path, parse_edits(reply), cfg=_cfg())
    assert all(r.ok for r in results)
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "A = 2\n"
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "B = 1\n"


def test_injected_disk_error_rolls_back_the_whole_batch(tmp_path, monkeypatch):
    _write(tmp_path, "a.py", "A = 1\n")
    _write(tmp_path, "b.py", "B = 1\n")
    reply = ("--- FILE: a.py\n<<<<<<< SEARCH\nA = 1\n=======\nA = 2\n"
             ">>>>>>> REPLACE\n"
             "--- FILE: b.py\n<<<<<<< SEARCH\nB = 1\n=======\nB = 2\n"
             ">>>>>>> REPLACE\n")
    real_write = fileio.write
    calls = {"n": 0}

    def flaky(path, text, style=None, **kw):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise OSError("disk full (injected)")
        return real_write(path, text, style, **kw)

    monkeypatch.setattr(fileio, "write", flaky)
    results = apply_edits_atomic(tmp_path, parse_edits(reply), cfg=_cfg())
    assert len(results) == 1 and not results[0].ok
    assert "rolled back" in results[0].message
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "A = 1\n"
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "B = 1\n"
    assert not list(journal._atomic_root(tmp_path).glob("*.json"))


def test_preimage_changed_underneath_aborts_before_commit(tmp_path, monkeypatch):
    _write(tmp_path, "a.py", "A = 1\n")
    _write(tmp_path, "b.py", "B = 1\n")
    reply = ("--- FILE: a.py\n<<<<<<< SEARCH\nA = 1\n=======\nA = 2\n"
             ">>>>>>> REPLACE\n"
             "--- FILE: b.py\n<<<<<<< SEARCH\nB = 1\n=======\nB = 2\n"
             ">>>>>>> REPLACE\n")
    real_begin = journal.begin_atomic

    # Mutate b.py after the preimages are read but before the preimage
    # re-check: the batch must abort without writing a.py.
    def mutating_begin(root, preimages, created=()):
        (tmp_path / "b.py").write_text("B = 99\n", encoding="utf-8")
        return real_begin(root, preimages, created)

    monkeypatch.setattr(journal, "begin_atomic", mutating_begin)
    results = apply_edits_atomic(tmp_path, parse_edits(reply), cfg=_cfg())
    assert len(results) == 1 and not results[0].ok
    assert "changed on disk" in results[0].message
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "A = 1\n"
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "B = 99\n"


def test_snapshot_failure_stops_the_batch(tmp_path, monkeypatch):
    _write(tmp_path, "a.py", "A = 1\n")
    reply = "--- FILE: a.py\n<<<<<<< SEARCH\nA = 1\n=======\nA = 2\n>>>>>>> REPLACE\n"

    def boom(*a, **k):
        raise OSError("snapshot disk full (injected)")

    monkeypatch.setattr(journal, "begin_atomic", boom)
    results = apply_edits_atomic(tmp_path, parse_edits(reply), cfg=_cfg())
    assert len(results) == 1 and not results[0].ok
    assert "snapshot" in results[0].message
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "A = 1\n"


# ----------------------------------------------------------- crash recovery

def test_recover_atomic_restores_preimages_and_removes_creations(tmp_path):
    _write(tmp_path, "a.py", "ORIG A\n")
    _write(tmp_path, "b.py", "ORIG B\n")
    txn = journal.begin_atomic(
        tmp_path, {"a.py": b"ORIG A\n", "b.py": b"ORIG B\n"},
        created=["c.py"])
    assert txn.status == journal.ATOMIC_STATUS_PREPARED
    # Simulate a crash mid-commit: one file written, a new file created.
    (tmp_path / "a.py").write_text("CHANGED A\n", encoding="utf-8")
    (tmp_path / "c.py").write_text("NEW C\n", encoding="utf-8")

    restored = journal.recover_atomic(tmp_path)
    assert set(restored) == {"a.py", "b.py", "c.py"}
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "ORIG A\n"
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "ORIG B\n"
    assert not (tmp_path / "c.py").exists()
    assert not list(journal._atomic_root(tmp_path).glob("*.json"))


def test_recover_atomic_ignores_committed_transactions(tmp_path):
    _write(tmp_path, "a.py", "A = 1\n")
    txn = journal.begin_atomic(tmp_path, {"a.py": b"A = 1\n"})
    journal.finish_atomic(tmp_path, txn.tx_id, journal.ATOMIC_STATUS_COMMITTED)
    journal.remove_atomic(tmp_path, txn.tx_id)
    # Nothing to recover.
    assert journal.recover_atomic(tmp_path) == []


def test_recover_atomic_refuses_a_corrupt_preimage(tmp_path):
    _write(tmp_path, "a.py", "ORIG\n")
    txn = journal.begin_atomic(tmp_path, {"a.py": b"ORIG\n"})
    # Corrupt the snapshot after the record was written.
    pre = journal._atomic_preimage_dir(tmp_path, txn.tx_id) / "a.py"
    pre.write_bytes(b"TAMPERED")
    (tmp_path / "a.py").write_text("CHANGED\n", encoding="utf-8")

    restored = journal.recover_atomic(tmp_path)
    assert restored == []
    # The transaction stays visible (recovery_failed), never silently dropped.
    remaining = [p for p in journal._atomic_root(tmp_path).glob("*.json")]
    assert remaining
    assert journal._read_txn(remaining[0]).status == \
        journal.ATOMIC_STATUS_RECOVERY_FAILED


def test_recover_atomic_refuses_an_escaping_store(tmp_path, monkeypatch):
    _write(tmp_path, "a.py", "A = 1\n")
    journal.begin_atomic(tmp_path, {"a.py": b"A = 1\n"})
    (tmp_path / "a.py").write_text("CHANGED\n", encoding="utf-8")
    monkeypatch.setattr(
        security, "is_junction_or_reparse",
        lambda p: Path(p) == journal._atomic_root(tmp_path))
    assert journal.recover_atomic(tmp_path) == []
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "CHANGED\n"


# ------------------------------------------------------------- byte fidelity

def test_bom_and_crlf_are_preserved(tmp_path):
    p = tmp_path / "a.py"
    p.write_bytes(b"\xef\xbb\xbfX = 1\r\n")
    reply = "--- FILE: a.py\n<<<<<<< SEARCH\nX = 1\n=======\nX = 2\n>>>>>>> REPLACE\n"
    results = apply_edits_atomic(tmp_path, parse_edits(reply), cfg=_cfg())
    assert all(r.ok for r in results)
    assert p.read_bytes() == b"\xef\xbb\xbfX = 2\r\n"


def test_file_mode_is_preserved(tmp_path):
    if os.name == "nt":
        pytest.skip("permission bits are not faithfully round-tripped on Windows")
    p = tmp_path / "a.py"
    p.write_text("A = 1\n", encoding="utf-8")
    os.chmod(p, 0o640)
    reply = "--- FILE: a.py\n<<<<<<< SEARCH\nA = 1\n=======\nA = 2\n>>>>>>> REPLACE\n"
    results = apply_edits_atomic(tmp_path, parse_edits(reply), cfg=_cfg())
    assert all(r.ok for r in results)
    assert (p.stat().st_mode & 0o777) == 0o640


# ------------------------------------------------------------------- escapes

def _symlink_unsupported(exc):
    if isinstance(exc, NotImplementedError):
        return True
    return isinstance(exc, OSError) and (
        getattr(exc, "winerror", None) == 1314
        or exc.errno in (errno.ENOTSUP, errno.EOPNOTSUPP))


def test_symlink_escape_is_refused(tmp_path):
    secret = tmp_path.parent / f"secret-{os.getpid()}-{tmp_path.name}.txt"
    secret.write_text("top secret\n", encoding="utf-8")
    proj = tmp_path / "proj"
    proj.mkdir()
    try:
        (proj / "link.py").symlink_to(secret)
    except (OSError, NotImplementedError) as exc:
        if not _symlink_unsupported(exc):
            raise
        pytest.skip("symlinks unsupported on this filesystem")
    reply = "--- FILE: link.py\n<<<<<<< SEARCH\n=======\nX = 1\n>>>>>>> REPLACE\n"
    results = apply_edits_atomic(proj, parse_edits(reply), cfg=_cfg())
    assert not any(r.ok for r in results)
    assert secret.read_text(encoding="utf-8") == "top secret\n"


# ----------------------------------------------------- Windows replace retry

def test_atomic_replace_retries_transient_sharing_violation(tmp_path, monkeypatch):
    import os as os_mod
    real_replace = os_mod.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            exc = PermissionError(5, "access denied")
            exc.winerror = 5
            raise exc
        return real_replace(src, dst)

    monkeypatch.setattr(os_mod, "replace", flaky)
    monkeypatch.setattr(
        fileio, "_is_transient_replace_error",
        lambda exc: getattr(exc, "winerror", 0) in (5, 32))
    monkeypatch.setattr(fileio, "_OS_REPLACE_DELAYS", (0.0, 0.0))
    fileio.write_bytes(tmp_path / "x.py", b"X = 1\n")
    assert calls["n"] == 2  # first attempt failed, retry succeeded
    assert (tmp_path / "x.py").read_bytes() == b"X = 1\n"


# --------------------------------------------------------- overlay integration

def test_overlay_batch_rolls_back_in_memory_on_failure(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    _write(proj, "a.py", "A = 1\n")
    ws = VirtualWorkspace(proj, apply=False, cfg=_cfg())
    from pedacito.edits import Edit
    good = Edit("a.py", "A = 1", "A = 2")
    bad = Edit("b.py", "no such search", "B = 2")
    results = ws.apply_edits_batch([good, bad])
    assert not any(r.ok for r in results)
    # The overlay staged a.py then rolled it back; the disk is untouched.
    assert ws.read("a.py") == "A = 1\n"
    assert (proj / "a.py").read_text(encoding="utf-8") == "A = 1\n"


def test_overlay_batch_applies_atomically_on_disk(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    _write(proj, "a.py", "A = 1\n")
    _write(proj, "b.py", "B = 1\n")
    ws = VirtualWorkspace(proj, apply=True, cfg=_cfg())
    from pedacito.edits import Edit
    results = ws.apply_edits_batch(
        [Edit("a.py", "A = 1", "A = 2"), Edit("b.py", "B = 1", "B = 2")])
    assert all(r.ok for r in results)
    assert (proj / "a.py").read_text(encoding="utf-8") == "A = 2\n"
    assert (proj / "b.py").read_text(encoding="utf-8") == "B = 2\n"


# --------------------------------------- refused gather writes (BUG-03, run_task)

PROTECTED = ".pedacito.toml"


class _GatherWriter:
    """Gather phase that appends through the real write tool. Each append is
    one gather-phase mutation; the protected config append is refused."""

    steps_log: list = []

    def __init__(self, root, cfg, appends):
        self.root, self.cfg, self.appends = root, cfg, appends

    def gather(self, task, backup_dir=None, **kw):
        self.cfg._backup_dir = backup_dir
        idx = SimpleNamespace(root=self.root, files=[])
        for rel, text in self.appends:
            tools_mod.tool_write(idx, self.cfg, rel, text, append=True)
        return "gathered source"

    def answer(self, task, gathered, **kw):
        return "Both requested edits are done."

    def apply(self, reply, **kw):
        return [], reply


def _task_cfg(**kw) -> Config:
    cfg = _cfg(**kw)
    cfg.backup = True
    cfg.allow_write = True
    cfg.verify_tests = False
    return cfg


def _project(tmp_path):
    _write(tmp_path, "currency.py", "X = 1\n")
    _write(tmp_path, PROTECTED, "max_total_model_calls = 1\n")
    return tmp_path


def _run(tmp_path, cfg, agent, task="Apply two edits"):
    return loop.run_task(tmp_path, cfg, None, agent,
                         SimpleNamespace(root=tmp_path, files=[]), task,
                         apply=True)


def test_refused_protected_append_rolls_back_partial_gather_writes(tmp_path):
    root = _project(tmp_path)
    cfg = _task_cfg(atomic_edits=True)
    agent = _GatherWriter(root, cfg, [
        ("currency.py", "# atomic probe\n"),
        (PROTECTED, "max_total_model_calls = 99\n"),
    ])
    tr = _run(root, cfg, agent)
    assert (root / "currency.py").read_text(encoding="utf-8") == "X = 1\n"
    assert (root / PROTECTED).read_text(encoding="utf-8") == \
        "max_total_model_calls = 1\n"
    assert tr.applied is False
    assert tr.ok is False
    assert tr.changed == []


def test_refused_write_without_atomic_is_still_not_ok(tmp_path):
    root = _project(tmp_path)
    cfg = _task_cfg(atomic_edits=False)
    agent = _GatherWriter(root, cfg, [
        ("currency.py", "# atomic probe\n"),
        (PROTECTED, "max_total_model_calls = 99\n"),
    ])
    tr = _run(root, cfg, agent)
    # Legacy apply keeps the change that landed, but the refusal is not a
    # clean success: the verdict says so.
    assert "# atomic probe" in (root / "currency.py").read_text(encoding="utf-8")
    assert tr.ok is False


def test_clean_gather_write_stays_ok(tmp_path):
    root = _project(tmp_path)
    cfg = _task_cfg(atomic_edits=True)
    agent = _GatherWriter(root, cfg, [("currency.py", "# atomic probe\n")])
    tr = _run(root, cfg, agent)
    assert "# atomic probe" in (root / "currency.py").read_text(encoding="utf-8")
    assert tr.applied is True
    assert tr.ok is None


def test_refusal_ledger_is_per_attempt(tmp_path):
    root = _project(tmp_path)
    cfg = _task_cfg(atomic_edits=True)
    cfg._task_refusals = ["stale-from-earlier-attempt"]
    agent = _GatherWriter(root, cfg, [("currency.py", "# atomic probe\n")])
    tr = _run(root, cfg, agent)
    assert tr.ok is None
    assert "# atomic probe" in (root / "currency.py").read_text(encoding="utf-8")


class _Requirement:
    is_edit = True
    state = verify_mod.REQ_UNLOCATED

    def __init__(self, req_id):
        self.req_id = req_id
        self.text = req_id


class _Tracker:
    def __init__(self, task, reqs):
        self.task = task
        self._reqs = reqs

    def __iter__(self):
        return iter(self._reqs)

    def clause_map(self):
        return {}

    def clause_texts(self):
        return {}


class _LegacyEditor:
    steps_log: list = []
    REPLY = ("--- FILE: currency.py\n<<<<<<< SEARCH\nX = 1\n=======\nX = 2\n"
             ">>>>>>> REPLACE\n")

    def __init__(self, root, task, reqs):
        self.root = root
        self.requirements = _Tracker(task, reqs)

    def gather(self, task, **kw):
        return "gathered source"

    def answer(self, task, gathered, **kw):
        return self.REPLY

    def apply(self, reply, **kw):
        (self.root / "currency.py").write_text("X = 2\n", encoding="utf-8")
        return [Result(True, "currency.py", "applied")], reply


def test_unaccepted_clause_without_a_refusal_keeps_the_verdict(tmp_path):
    """An applied edit whose requirement clause went unevidenced is not a
    refusal: the verdict stays as BUG-212 defined it (landed work exits 0)."""
    root = _project(tmp_path)
    cfg = _task_cfg(atomic_edits=False)
    task = "Change X and also rename Y"
    agent = _LegacyEditor(root, task, [_Requirement("R1")])
    tr = _run(root, cfg, agent, task=task)
    assert (root / "currency.py").read_text(encoding="utf-8") == "X = 2\n"
    assert tr.applied is True and tr.unaccepted == ["R1"]
    assert tr.ok is None
