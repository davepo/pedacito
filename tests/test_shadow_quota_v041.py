"""WP-4105F: shadow test isolation, ratchet safety and checkpoint quota.

Covers the §3.7 contract:

* green-ratchet snapshots are independent copies (reflink/copy2, never hard
  links) so an in-place writer cannot corrupt history, and rollback restores
  exact hashes;
* `plan_checkpoint_gc` / `apply_checkpoint_gc` count snapshot/ratchet/backup
  bytes without following links, protect the head/ancestors/in-flight
  WAL/green frontier/safety snapshots, delete oldest-eligible-first under a
  cross-process lock, and refuse (never force) when a quota cannot be met
  safely;
* shadow tests build a no-link copy with a private HOME/TMP, scrubbed env and
  separately allocated loopback ports, and never promise network isolation
  that the platform cannot provide (Windows).
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import (  # noqa: E402
    checkpoints,
    checkpoint_gc,
    proc,
    ratchet,
    sandbox,
    workspace_overlay,
)
from pedacito import config as config_mod  # noqa: E402
from pedacito.config import Config  # noqa: E402


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _py(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(text.encode("utf-8"))       # exact bytes, no newline translation
    return p


# ---------------------------------------------------------------- ratchet safety

def test_ratchet_snapshot_is_an_independent_copy(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    a = _py(root, "src/app.py", "VALUE = 1\n")
    info = ratchet.begin(root, 1, "copy audit")
    assert info["error"] is None, info
    assert "src/app.py" in info["files"]

    snap = ratchet.step_dir(root, 1) / "src/app.py"
    assert snap.is_file()
    if os.name != "nt":
        # A hard link would share the inode; the copy must not.
        assert snap.stat().st_ino != a.stat().st_ino

    # Mutate IN PLACE (no atomic replace): the old hard-link scheme would have
    # truncated the snapshot through the shared inode.
    with open(a, "wb") as fh:
        fh.write(b"VALUE = 999\n")
    assert snap.read_bytes() == b"VALUE = 1\n"      # snapshot unaffected

    report = ratchet.rollback(root, 1)
    assert report["error"] is None, report
    assert "src/app.py" in report["restored"], report
    assert _sha(a) == hashlib.sha256(b"VALUE = 1\n").hexdigest()


def test_ratchet_manifest_records_copy_or_reflink(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    _py(root, "a.py", "x = 1\n")
    ratchet.begin(root, 1)
    manifest = json.loads(
        (ratchet.step_dir(root, 1) / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["files"]["a.py"]
    assert entry["linked"] is False
    assert entry["snapshot"] in ("copy", "reflink")


def test_ratchet_copy_failure_is_reported_not_fatal(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    _py(root, "a.py", "x = 1\n")

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(ratchet, "_snapshot_file", boom)
    info = ratchet.begin(root, 1)
    assert info["error"] is None              # begin never crashes
    assert "a.py" not in info["files"]        # recorded as skipped


def test_copy_path_never_needs_symlink_privilege(tmp_path, monkeypatch):
    """WinError 1314 (no SeCreateSymbolicLinkPrivilege) never blocks a shadow
    copy: the copy path creates no symlink or junction at all."""
    def deny_symlink(*a, **k):
        e = OSError("A required privilege is not held by the client.")
        e.errno = errno.EPERM
        e.winerror = 1314
        raise e

    root = tmp_path / "proj"
    root.mkdir()
    _py(root, "src/app.py", "print(1)\n")
    monkeypatch.setattr(Path, "symlink_to", deny_symlink)
    copy = workspace_overlay.build_shadow_copy(root)
    try:
        assert (copy / "src/app.py").is_file()
    finally:
        shutil.rmtree(copy, ignore_errors=True)


# ------------------------------------------------------------- checkpoint quota

def test_config_contract_fields():
    assert "shadow_tests" in config_mod.EXEC_FIELDS
    assert Config().shadow_tests is False
    assert Config().checkpoint_quota_mb == 0


def test_plan_counts_bytes_without_following_links(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    _py(root, "a.py", "x = 1\n")
    cp = checkpoints.create_checkpoint(root, "base", mode="manual")
    snap = checkpoints.snapshot_dir(root, cp["id"])

    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"Z" * 100_000)
    try:
        (snap / "link.bin").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")

    plan = checkpoint_gc.plan_checkpoint_gc(root, 0)
    # Only the real snapshot bytes count; the symlink's 100_000 bytes do not.
    assert plan["total_bytes"] == len(b"x = 1\n")


def test_plan_protects_head_and_ancestors(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    _py(root, "a.py", "x = 1\n")
    cp1 = checkpoints.create_checkpoint(root, "one", mode="manual")
    cp2 = checkpoints.create_checkpoint(root, "two", mode="manual")
    cp3 = checkpoints.create_checkpoint(root, "three", mode="manual")
    plan = checkpoint_gc.plan_checkpoint_gc(root, 0)
    prot = {t["id"] for t in plan["protected"] if t["kind"] == "snapshot"}
    assert {cp1["id"], cp2["id"], cp3["id"]} <= prot
    elig = [t for t in plan["eligible"] if t["kind"] == "snapshot"]
    assert elig == []


def test_plan_protects_orphaned_safety_snapshot(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    _py(root, "a.py", "x = 1\n")
    cp1 = checkpoints.create_checkpoint(root, "one", mode="manual")
    cp2 = checkpoints.create_checkpoint(root, "safety", mode="restore")
    checkpoints._set_head(root, cp1["id"])      # orphan the safety snapshot
    protected = checkpoint_gc._protected_snapshot_ids(root)
    assert protected.get(cp1["id"]) == "head-or-ancestor"
    assert protected.get(cp2["id"]) == "safety-snapshot"


def test_plan_protects_green_frontier_and_marks_older_step_eligible(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    _py(root, "a.py", "x = 1\n")
    ratchet.begin(root, 1)
    ratchet.green(root, 1)
    ratchet.begin(root, 2)
    ratchet.green(root, 2)
    assert ratchet.frontier(root) == 2

    plan = checkpoint_gc.plan_checkpoint_gc(root, 0)
    rat_kept = [t for t in plan["protected"] if t["kind"] == "ratchet"]
    assert [t["step"] for t in rat_kept] == [2]
    rat_elig = [t for t in plan["eligible"] if t["kind"] == "ratchet"]
    assert [t["step"] for t in rat_elig] == [1]


def test_apply_deletes_oldest_eligible_first_and_is_idempotent(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    old = root / ".pedacito" / "backups" / "20200101-000000"
    new = root / ".pedacito" / "backups" / "20200102-000000"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (old / "big.txt").write_bytes(b"B" * 5000)
    (new / "a.py").write_bytes(b"x = 2\n")

    plan = checkpoint_gc.plan_checkpoint_gc(root, 100)
    assert plan["satisfiable"] is True
    assert [t["rel"] for t in plan["to_delete"]] == \
        [old.relative_to(root).as_posix()]

    result = checkpoint_gc.apply_checkpoint_gc(root, plan)
    assert result["error"] is None
    assert [t["rel"] for t in result["deleted"]] == \
        [old.relative_to(root).as_posix()]
    assert not old.exists()
    assert new.exists()                         # newest backup kept

    # Idempotent: a second apply of the same plan deletes nothing.
    again = checkpoint_gc.apply_checkpoint_gc(root, plan)
    assert again["deleted"] == []
    assert again["error"] is None


def test_plan_unsatisfiable_when_protected_history_blocks_quota(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    _py(root, "a.py", "x = 1\n")
    checkpoints.create_checkpoint(root, "base", mode="manual")  # head, protected
    plan = checkpoint_gc.plan_checkpoint_gc(root, 1)
    assert plan["satisfiable"] is False
    assert "cannot be met safely" in plan["reason"]
    assert plan["to_delete"] == []                  # nothing eligible to free


def test_apply_skips_a_target_that_became_protected(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    _py(root, "a.py", "x = 1\n")
    cp = checkpoints.create_checkpoint(root, "base", mode="manual")
    plan = {"to_delete": [
        {"kind": "snapshot", "id": cp["id"],
         "rel": checkpoints.snapshot_dir(root, cp["id"]).relative_to(root).as_posix(),
         "bytes": 6},
    ]}
    result = checkpoint_gc.apply_checkpoint_gc(root, plan)
    assert result["deleted"] == []
    assert any("protected" in s.get("reason", "") for s in result["skipped"])
    assert checkpoints.snapshot_dir(root, cp["id"]).is_dir()


def test_apply_rejects_an_unsafe_store_path(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    outside = tmp_path / "victim"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep\n", encoding="utf-8")
    plan = {"to_delete": [{"kind": "backup", "rel": "../victim", "bytes": 100}]}
    result = checkpoint_gc.apply_checkpoint_gc(root, plan)
    assert result["deleted"] == []
    assert (outside / "keep.txt").exists()


def test_concurrent_gc_preserves_protected_history(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    _py(root, "a.py", "x = 1\n")
    cp = checkpoints.create_checkpoint(root, "base", mode="manual")
    bak = root / ".pedacito" / "backups" / "20200101-000000"
    bak.mkdir(parents=True)
    (bak / "a.py").write_text("old\n", encoding="utf-8")

    plan = checkpoint_gc.plan_checkpoint_gc(root, 0)
    plan["to_delete"] = [
        {"kind": "backup", "rel": bak.relative_to(root).as_posix(), "bytes": 4},
        {"kind": "snapshot", "id": cp["id"],
         "rel": checkpoints.snapshot_dir(root, cp["id"]).relative_to(root).as_posix(),
         "bytes": 6},
    ]
    results = []
    barrier = threading.Barrier(4)

    def worker():
        barrier.wait()
        try:
            results.append(checkpoint_gc.apply_checkpoint_gc(root, plan))
        except Exception as exc:                    # pragma: no cover
            results.append({"error": repr(exc)})

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # The protected head survived every concurrent attempt; the eligible backup
    # was deleted by whoever held the cross-process lock.
    assert checkpoints.snapshot_dir(root, cp["id"]).is_dir()
    assert not bak.exists()
    for r in results:
        err = r.get("error")
        assert err is None or "busy" in str(err).lower(), results


def test_enforce_quota_zero_is_unlimited(tmp_path):
    cfg = Config()
    cfg.checkpoint_quota_mb = 0
    ok, reason = checkpoint_gc.enforce_quota(tmp_path, cfg)
    assert ok is True and reason == ""


def test_enforce_quota_reports_unsatisfiable(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    _py(root, "a.py", "x = 1\n")
    checkpoints.create_checkpoint(root, "base", mode="manual")
    snap = checkpoints.snapshot_dir(root, checkpoints.head_id(root))
    (snap / "big.txt").write_bytes(b"B" * (2 * 1024 * 1024))
    cfg = Config()
    cfg.checkpoint_quota_mb = 1
    ok, reason = checkpoint_gc.enforce_quota(root, cfg)
    assert ok is False
    assert "cannot be met safely" in reason


# ------------------------------------------------------------------ shadow tests

def test_build_shadow_copy_never_links_dependencies(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    _py(root, "src/app.py", "print('hi')\n")
    (root / "node_modules" / "pkg").mkdir(parents=True)
    (root / "node_modules" / "pkg" / "index.js").write_text(
        "module.exports = 1;\n", encoding="utf-8")

    copy = workspace_overlay.build_shadow_copy(root)
    try:
        assert (copy / "src/app.py").read_text(encoding="utf-8") == "print('hi')\n"
        nm = copy / "node_modules"
        assert not nm.is_symlink()
        assert not (nm / "pkg" / "index.js").exists()
        (copy / "src/app.py").write_text("print('CHANGED')\n", encoding="utf-8")
        assert (root / "src/app.py").read_text(encoding="utf-8") == "print('hi')\n"
    finally:
        shutil.rmtree(copy, ignore_errors=True)


def test_shadow_environment_has_private_home_tmp_and_scrubbed_env(tmp_path,
                                                                  monkeypatch):
    monkeypatch.setenv("SOME_API_KEY", "secret-value")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    root = tmp_path / "proj"
    root.mkdir()
    cfg = Config()
    home, tmp, env = proc.shadow_environment(root, cfg=cfg)
    try:
        assert env["HOME"] == str(home)
        assert env["TMP"] == str(tmp)
        assert env["TMPDIR"] == str(tmp)
        assert env["TEMP"] == str(tmp)
        assert str(home) != str(Path.home())
        assert "SOME_API_KEY" not in env
        assert "AWS_SECRET_ACCESS_KEY" not in env
    finally:
        shutil.rmtree(home.parent, ignore_errors=True)


def test_allocate_loopback_ports_distinct():
    ports = proc.allocate_loopback_ports(3)
    assert len(ports) == 3 and len(set(ports)) == 3
    assert all(1 <= p <= 65535 for p in ports)


def test_shadow_network_isolation_is_honest(monkeypatch):
    cfg = Config()
    monkeypatch.setattr(sandbox, "backend_name", lambda: "win_restricted")
    assert sandbox.shadow_network_isolation(cfg) is False
    monkeypatch.setattr(sandbox, "backend_name", lambda: "bwrap")
    assert sandbox.shadow_network_isolation(cfg) is True
    monkeypatch.setattr(sandbox, "backend_name", lambda: "seatbelt")
    assert sandbox.shadow_network_isolation(cfg) is True
    monkeypatch.setattr(sandbox, "backend_name", lambda: "")
    assert sandbox.shadow_network_isolation(cfg) is False
