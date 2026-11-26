"""Model/VRAM lease manager (WP-4002, D5).

Covers the local-GPU serialisation rules: hosted bypass, cache-affinity
refcounting, different-model queuing behind an active model, cache-affinity
priority with anti-starvation aging, exclusive drain-without-preempt, the
``do_not_evict``/``pin`` protection surface, cancellation/timeout, and
``release_session``.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import leases  # noqa: E402

LOCAL = "http://127.0.0.1:1234/v1"
HOSTED = "https://api.example.com/v1"


@pytest.fixture
def mgr():
    return leases.ModelLeaseManager()


@pytest.fixture(autouse=True)
def _reset_default():
    leases.reset_default_lease_manager()
    yield
    leases.reset_default_lease_manager()


def _clock():
    """A controllable monotonic clock for aging tests."""
    state = {"t": 0.0}

    def now():
        return state["t"]

    now.state = state
    return now


# ---------------------------------------------------------------- hosted bypass
def test_hosted_url_bypasses_pool(mgr):
    h1 = mgr.acquire("s1", HOSTED, "gpt-4o")
    h2 = mgr.acquire("s2", HOSTED, "gpt-4o")
    assert h1.exclusive is False
    snap = mgr.snapshot()
    assert len(snap["active_leases"]) == 2
    assert all(rec["hosted"] for rec in snap["active_leases"])
    h1.release()
    h2.release()
    assert mgr.snapshot()["active_leases"] == []


# ----------------------------------------------------------- refcount / evict
def test_same_model_refcounts_and_blocks_eviction(mgr):
    h1 = mgr.acquire("s1", LOCAL, "model-a")
    h2 = mgr.acquire("s2", LOCAL, "model-a")          # shared: refcount++
    assert not mgr.can_evict(LOCAL, "model-a")
    h1.release()
    assert not mgr.can_evict(LOCAL, "model-a")        # s2 still holds
    h2.release()
    assert mgr.can_evict(LOCAL, "model-a")


def test_different_model_queues_behind_active(mgr):
    h_a = mgr.acquire("s1", LOCAL, "model-a")
    got = {}

    def want_b():
        got["b"] = "waiting"
        h = mgr.acquire("s2", LOCAL, "model-b", timeout_s=5)
        got["b"] = "acquired"
        h.release()

    t = threading.Thread(target=want_b)
    t.start()
    time.sleep(0.2)
    assert got["b"] == "waiting"                      # B never evicts A
    assert not mgr.can_evict(LOCAL, "model-a")        # A still resident/leased
    h_a.release()
    t.join(5)
    assert got["b"] == "acquired"


# ------------------------------------------------- cache affinity + aging
def test_cache_affinity_boosts_same_model_waiters(mgr):
    h_a = mgr.acquire("s1", LOCAL, "model-a")
    got = {}

    def want(model, name):
        h = mgr.acquire(name, LOCAL, model, timeout_s=5)
        got[name] = "acquired"
        h.release()

    tb = threading.Thread(target=want, args=("model-b", "b"))
    ta = threading.Thread(target=want, args=("model-a", "a"))
    tb.start()
    time.sleep(0.2)
    ta.start()
    time.sleep(0.2)
    h_a.release()
    tb.join(5)
    ta.join(5)
    # Both eventually acquire; the same-model waiter goes first (cache affinity).
    assert got.get("a") == "acquired"
    assert got.get("b") == "acquired"


def test_anti_starvation_aging_lets_old_waiters_win(mgr):
    clock = _clock()
    mgr._clock = clock
    h_a = mgr.acquire("s1", LOCAL, "model-a")

    # B (different model) is queued first and ages.
    b_acquired = threading.Event()

    def want_b():
        h = mgr.acquire("sb", LOCAL, "model-b", timeout_s=0.0)
        b_acquired.set()
        h.release()

    tb = threading.Thread(target=want_b)
    tb.start()
    time.sleep(0.2)
    clock.state["t"] += 2.0                          # B ages 2s (20 pts)

    # A same-model waiter arrives after B, so B's aging outranks A's affinity.
    got_a = {}

    def want_a():
        h = mgr.acquire("sa", LOCAL, "model-a", timeout_s=0.0)
        got_a["done"] = True
        h.release()

    ta = threading.Thread(target=want_a)
    ta.start()
    time.sleep(0.2)
    h_a.release()                                    # drain -> B first
    assert b_acquired.wait(5)
    ta.join(5)
    assert got_a.get("done")


# ---------------------------------------------------------------- exclusive
def test_exclusive_drains_and_blocks_new_leases(mgr):
    h1 = mgr.acquire("s1", LOCAL, "model-a")
    h2 = mgr.acquire("s2", LOCAL, "model-a")
    got = {}

    def excl():
        got["ex"] = "waiting"
        h = mgr.acquire("ex", LOCAL, "model-x", exclusive=True, timeout_s=5)
        got["ex"] = "acquired"
        h.release()

    te = threading.Thread(target=excl)
    te.start()
    time.sleep(0.2)
    assert got["ex"] == "waiting"                    # never preempts A
    h1.release()
    h2.release()
    te.join(5)
    assert got["ex"] == "acquired"


def test_exclusive_blocks_new_leases_until_released(mgr):
    h_ex = mgr.acquire("ex", LOCAL, "model-x", exclusive=True)
    got = {}

    def normal():
        got["n"] = "waiting"
        h = mgr.acquire("s1", LOCAL, "model-a", timeout_s=5)
        got["n"] = "acquired"
        h.release()

    tn = threading.Thread(target=normal)
    tn.start()
    time.sleep(0.2)
    assert got["n"] == "waiting"
    h_ex.release()
    tn.join(5)
    assert got["n"] == "acquired"


# --------------------------------------------------- do_not_evict / pin/unpin
def test_do_not_evict_and_pin_protect(mgr):
    h = mgr.acquire("s1", LOCAL, "model-a", do_not_evict=True)
    assert not mgr.can_evict(LOCAL, "model-a")
    h.release()
    assert mgr.can_evict(LOCAL, "model-a")

    mgr.pin(LOCAL, "model-b")
    assert not mgr.can_evict(LOCAL, "model-b")
    assert mgr.can_evict(LOCAL, "model-a")           # pin is model-scoped
    mgr.unpin(LOCAL, "model-b")
    assert mgr.can_evict(LOCAL, "model-b")


# ----------------------------------------------------- cancel / timeout
def test_cancel_event_aborts_queued_waiter(mgr):
    h_a = mgr.acquire("s1", LOCAL, "model-a")
    cancel = threading.Event()
    got = {}

    def want_b():
        try:
            mgr.acquire("s2", LOCAL, "model-b", timeout_s=0.0,
                        cancel_event=cancel)
            got["b"] = "acquired"
        except RuntimeError:
            got["b"] = "cancelled"

    t = threading.Thread(target=want_b)
    t.start()
    time.sleep(0.2)
    cancel.set()
    t.join(5)
    assert got["b"] == "cancelled"
    h_a.release()


def test_timeout_aborts_queued_waiter(mgr):
    h_a = mgr.acquire("s1", LOCAL, "model-a")
    got = {}

    def want_b():
        try:
            mgr.acquire("s2", LOCAL, "model-b", timeout_s=0.3)
            got["b"] = "acquired"
        except TimeoutError:
            got["b"] = "timeout"

    t = threading.Thread(target=want_b)
    t.start()
    t.join(5)
    assert got["b"] == "timeout"
    h_a.release()


# ------------------------------------------------------------ release_session
def test_release_session_releases_all_leases(mgr):
    mgr.acquire("s1", LOCAL, "model-a")
    mgr.acquire("s1", LOCAL, "model-a")
    mgr.acquire("s2", LOCAL, "model-a")
    assert not mgr.can_evict(LOCAL, "model-a")
    assert mgr.release_session("s1") == 2
    assert not mgr.can_evict(LOCAL, "model-a")       # s2 still holds
    assert mgr.release_session("s2") == 1
    assert mgr.can_evict(LOCAL, "model-a")


# --------------------------------------- servers.evict_other_local_servers
def test_evict_other_local_servers_skips_leased_and_pinned(monkeypatch):
    from pedacito import servers

    mgr = leases.default_lease_manager()
    other = "http://127.0.0.1:9000/v1"
    mgr.pin(other, "pinned-model")
    mgr.acquire("s1", other, "leased-model")

    states = [
        {"name": "target", "base_url": LOCAL, "online": True,
         "loaded_models": ["target-model"], "pinned_models": [], "active": True},
        {"name": "other", "base_url": other, "online": True,
         "loaded_models": ["pinned-model", "leased-model", "free-model"],
         "pinned_models": [], "active": False},
    ]
    monkeypatch.setattr(servers, "list_server_states",
                        lambda config, **kw: states)
    unloaded = []
    monkeypatch.setattr(servers, "unload_server_model",
                        lambda server, model="", config=None, **kw:
                        unloaded.append(model) or {"ok": True, "model": model})

    result = servers.evict_other_local_servers(LOCAL, None, poll=False)
    # Only the unleased, unpinned "free-model" is unloaded.
    assert unloaded == ["free-model"]
    assert any("leased" in s for s in result["skipped"])
    assert any("pinned" in s for s in result["skipped"])

