"""Project-state concurrency & maintenance locks (WP-4002, D6/D15).

Covers the in-process RWLock + cross-process lockfile, PID-reuse stale
detection, atomic read-modify-write state files, concurrent memory updates,
and the maintenance-safety gate.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import workspace  # noqa: E402

LOCK = ".pedacito" + os.sep + "run.lock"


def _lock_file(tmp_path):
    return tmp_path / ".pedacito" / "run.lock"


# --------------------------------------------------- in-process concurrency
def test_two_threads_second_fails_fast(tmp_path):
    workspace.acquire_lock(tmp_path)
    result = {}

    def second():
        try:
            workspace.acquire_lock(tmp_path)
            result["ok"] = True
        except workspace.ProjectBusy as e:
            result["busy"] = str(e)

    t = threading.Thread(target=second)
    t.start()
    t.join(5)
    assert "busy" in result
    workspace.release_lock(tmp_path)
    assert not _lock_file(tmp_path).exists()


def test_two_threads_second_waits_when_timeout(tmp_path):
    workspace.acquire_lock(tmp_path)
    got = {}

    def second():
        got["started"] = True
        workspace.acquire_lock(tmp_path, wait_timeout=5.0)
        got["done"] = True
        workspace.release_lock(tmp_path)

    t = threading.Thread(target=second)
    t.start()
    time.sleep(0.3)
    assert got.get("started") and not got.get("done")
    workspace.release_lock(tmp_path)
    t.join(5)
    assert got.get("done")


def test_shared_and_exclusive_rwlock(tmp_path):
    mgr = workspace.ProjectLockManager()
    o1 = mgr.acquire(tmp_path, mode="shared", session_id="s1")
    o2 = mgr.acquire(tmp_path, mode="shared", session_id="s2")
    with pytest.raises(workspace.ProjectBusy):
        mgr.acquire(tmp_path, mode="exclusive", session_id="s3")
    mgr.release(tmp_path, o1)
    mgr.release(tmp_path, o2)
    o3 = mgr.acquire(tmp_path, mode="exclusive", session_id="s3")
    with pytest.raises(workspace.ProjectBusy):
        mgr.acquire(tmp_path, mode="shared", session_id="s4")
    mgr.release(tmp_path, o3)
    assert not _lock_file(tmp_path).exists()


def test_same_session_reentry_allowed(tmp_path):
    mgr = workspace.ProjectLockManager()
    o1 = mgr.acquire(tmp_path, session_id="ses")
    o2 = mgr.acquire(tmp_path, session_id="ses")      # re-entry: same session
    mgr.release(tmp_path, o1)
    assert _lock_file(tmp_path).exists()              # still held by o2
    mgr.release(tmp_path, o2)
    assert not _lock_file(tmp_path).exists()


# ------------------------------------------------------------ lockfile format
def test_lockfile_v2_json_format(tmp_path):
    lock = _lock_file(tmp_path)
    workspace.acquire_lock(tmp_path)
    data = json.loads(lock.read_text(encoding="utf-8"))
    assert data["version"] == 2
    assert data["pid"] == os.getpid()
    assert data["owner_token"]
    assert data["mode"] == "exclusive"
    assert data["session_id"] == ""
    workspace.release_lock(tmp_path)
    assert not lock.exists()


def test_lockfile_v1_backcompat_blocks_live_holder(tmp_path):
    lock = _lock_file(tmp_path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    sleeper = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        lock.write_text(f"{sleeper.pid}\n", encoding="utf-8")
        with pytest.raises(workspace.ProjectBusy):
            workspace.acquire_lock(tmp_path)
        assert sleeper.poll() is None                 # probing must not kill it
    finally:
        sleeper.kill()
        sleeper.wait()
    workspace.acquire_lock(tmp_path)                  # dead holder: takeover
    data = json.loads(lock.read_text(encoding="utf-8"))
    assert data["pid"] == os.getpid()
    workspace.release_lock(tmp_path)
    assert not lock.exists()


# ------------------------------------------------------------- stale takeover
def test_stale_takeover_dead_pid(tmp_path):
    lock = _lock_file(tmp_path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({
        "version": 2, "pid": 999999999, "start_time": 0.0,
        "owner_token": "dead", "session_id": "", "mode": "exclusive",
        "created_at": 0.0,
    }), encoding="utf-8")
    workspace.acquire_lock(tmp_path)
    assert json.loads(lock.read_text(encoding="utf-8"))["pid"] == os.getpid()
    workspace.release_lock(tmp_path)


def test_stale_takeover_pid_reuse(tmp_path):
    lock = _lock_file(tmp_path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    sleeper = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        actual = workspace._process_start_time(sleeper.pid)
        if actual <= 0.0:
            pytest.skip("process start time unavailable on this platform")
        # A live pid whose recorded start time no longer matches (pid reuse).
        lock.write_text(json.dumps({
            "version": 2, "pid": sleeper.pid, "start_time": actual + 1000.0,
            "owner_token": "reused", "session_id": "", "mode": "exclusive",
            "created_at": 0.0,
        }), encoding="utf-8")
        workspace.acquire_lock(tmp_path)              # mismatch -> stale
        assert sleeper.poll() is None
    finally:
        sleeper.kill()
        sleeper.wait()
    assert json.loads(lock.read_text(encoding="utf-8"))["pid"] == os.getpid()
    workspace.release_lock(tmp_path)


def test_live_holder_with_matching_start_time_blocks(tmp_path):
    lock = _lock_file(tmp_path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    sleeper = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        actual = workspace._process_start_time(sleeper.pid)
        if actual <= 0.0:
            pytest.skip("process start time unavailable on this platform")
        lock.write_text(json.dumps({
            "version": 2, "pid": sleeper.pid, "start_time": actual,
            "owner_token": "live", "session_id": "", "mode": "exclusive",
            "created_at": 0.0,
        }), encoding="utf-8")
        with pytest.raises(workspace.ProjectBusy):
            workspace.acquire_lock(tmp_path)
    finally:
        sleeper.kill()
        sleeper.wait()


# ----------------------------------------------------- release token matching
def test_release_refuses_foreign_owner_token(tmp_path):
    lock = _lock_file(tmp_path)
    mgr = workspace.ProjectLockManager()
    owner = mgr.acquire(tmp_path)
    data = json.loads(lock.read_text(encoding="utf-8"))
    data["owner_token"] = "someone-else"
    lock.write_text(json.dumps(data), encoding="utf-8")
    mgr.release(tmp_path, owner)                      # token mismatch: keep file
    assert lock.exists()


# ---------------------------------------------- atomic read-modify-write
def test_atomic_update_json_stress(tmp_path):
    p = tmp_path / "counter.json"

    def worker():
        for _ in range(20):
            workspace.atomic_update_json(
                p, lambda d: {"n": d.get("n", 0) + 1})

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert json.loads(p.read_text(encoding="utf-8"))["n"] == 200


def test_atomic_update_text_stress(tmp_path):
    p = tmp_path / "lines.txt"

    def worker():
        for _ in range(20):
            workspace.atomic_update_text(p, lambda s: s + "x")

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert p.read_text(encoding="utf-8") == "x" * 200


# --------------------------------------------------- concurrent memory files
def test_concurrent_memory_updates(tmp_path):
    from pedacito import config, memory

    root = tmp_path / "proj"
    root.mkdir()
    cfg = config.Config()
    cfg.micro_facts = True
    cfg.dynamic_tool_pruning = True

    # profile_stats.json: 10 threads x 20 = 200 zero-edit runs, zero lost.
    def zero():
        for _ in range(20):
            memory.note_zero_edit(root, cfg)

    threads = [threading.Thread(target=zero) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert memory.zero_edit_runs(root, cfg) == 200

    # micro_facts.json: 20 distinct commands, all retained.
    def obs(i):
        memory.observe_command(root, f"cmd-{i}", True, cfg=cfg)
        memory.observe_command(root, f"cmd-{i}", True, cfg=cfg)

    threads = [threading.Thread(target=obs, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    facts = memory.load_micro_facts(root)
    assert {f["fact_id"] for f in facts} == {
        f"cmd:{__import__('hashlib').sha1(f'cmd-{i}'.encode()).hexdigest()[:12]}"
        for i in range(20)
    }

    # memory.md: 15 distinct notes, all retained.
    def rem(i):
        memory.remember(root, f"fact number {i}", category="notes")

    threads = [threading.Thread(target=rem, args=(i,)) for i in range(15)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    notes = [e for e in memory.entries(root)
             if e[0] == "project" and "fact number" in e[2]]
    assert len(notes) == 15

    # tool_stats.json: 20 tasks, 20 distinct tools.
    cfg._track_tool_use = True

    def tool(i):
        memory.begin_tool_task(root, cfg)
        memory.record_tool_use(root, f"tool-{i}", cfg)

    threads = [threading.Thread(target=tool, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    doc = memory._tool_stats_doc(root)
    assert doc["tasks"] == 20
    assert len(doc["uses"]) == 20


# ----------------------------------------------------- maintenance safety gate
def test_check_maintenance_safe_blocks_when_locked(tmp_path):
    workspace.acquire_lock(tmp_path)
    try:
        ok, reason = workspace.check_maintenance_safe(tmp_path, action="reset")
        assert not ok and "another pedacito run" in reason
        ok2, _ = workspace.check_maintenance_safe(
            tmp_path, action="reset", force=True)
        assert ok2
    finally:
        workspace.release_lock(tmp_path)
    ok3, _ = workspace.check_maintenance_safe(tmp_path, action="reset")
    assert ok3


def test_check_maintenance_safe_quiesce_hook(tmp_path):
    active = ["ses-1"]

    def hook(root, action):
        return list(active)

    workspace.register_quiesce_hook(hook)
    try:
        ok, reason = workspace.check_maintenance_safe(tmp_path, action="prune")
        assert not ok and "active session" in reason
        active.clear()
        ok2, _ = workspace.check_maintenance_safe(tmp_path, action="prune")
        assert ok2
    finally:
        workspace.unregister_quiesce_hook(hook)


# ---------------------------------------------------- H5 lockfile hardening
def test_malformed_lockfile_does_not_crash(tmp_path):
    lock = _lock_file(tmp_path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({
        "version": 2, "pid": "abc", "start_time": "not-a-float",
        "owner_token": 123, "session_id": "", "mode": "exclusive",
        "created_at": "y",
    }), encoding="utf-8")
    # Malformed pid/start_time must never crash: treated as stale, taken over.
    workspace.acquire_lock(tmp_path)
    assert json.loads(lock.read_text(encoding="utf-8"))["pid"] == os.getpid()
    workspace.release_lock(tmp_path)
    assert not lock.exists()


def test_empty_fresh_lockfile_is_busy_not_stolen(tmp_path):
    lock = _lock_file(tmp_path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("", encoding="utf-8")   # fresh, empty (mid-create)
    with pytest.raises(workspace.ProjectBusy):
        workspace.acquire_lock(tmp_path)


def test_two_managers_in_one_process_conflict(tmp_path):
    a = workspace.ProjectLockManager()
    b = workspace.ProjectLockManager()
    oa = a.acquire(tmp_path)
    try:
        with pytest.raises(workspace.ProjectBusy):
            b.acquire(tmp_path)          # own-pid lock is NOT stale
    finally:
        a.release(tmp_path, oa)
    assert not _lock_file(tmp_path).exists()


def test_shared_reentry_upgrade_to_exclusive(tmp_path):
    mgr = workspace.ProjectLockManager()
    o1 = mgr.acquire(tmp_path, mode="shared", session_id="s1")
    o2 = mgr.acquire(tmp_path, mode="exclusive", session_id="s1")  # upgrade
    assert o2.mode == "exclusive"
    mgr.release(tmp_path, o1)
    mgr.release(tmp_path, o2)
    assert not _lock_file(tmp_path).exists()


def test_shared_reentry_upgrade_conflicts_with_other_owner(tmp_path):
    mgr = workspace.ProjectLockManager()
    o1 = mgr.acquire(tmp_path, mode="shared", session_id="s1")
    o2 = mgr.acquire(tmp_path, mode="shared", session_id="s2")
    with pytest.raises(workspace.ProjectBusy):
        mgr.acquire(tmp_path, mode="exclusive", session_id="s1")
    mgr.release(tmp_path, o1)
    mgr.release(tmp_path, o2)
    assert not _lock_file(tmp_path).exists()


def test_shared_release_cleans_up_lockfile(tmp_path):
    mgr = workspace.ProjectLockManager()
    o1 = mgr.acquire(tmp_path, mode="shared", session_id="s1")
    o2 = mgr.acquire(tmp_path, mode="shared", session_id="s2")
    mgr.release(tmp_path, o1)               # first (on-disk) owner first
    assert _lock_file(tmp_path).exists()    # still held by o2
    mgr.release(tmp_path, o2)
    assert not _lock_file(tmp_path).exists()  # no lingering run.lock


def test_check_maintenance_safe_fails_closed_on_hook_error(tmp_path):
    def boom(root, action):
        raise RuntimeError("hook exploded")

    workspace.register_quiesce_hook(boom)
    try:
        ok, reason = workspace.check_maintenance_safe(tmp_path, action="reset")
        assert not ok and "safety hook failed" in reason
    finally:
        workspace.unregister_quiesce_hook(boom)
