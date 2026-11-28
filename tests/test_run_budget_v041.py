"""WP-4105E: the session-scoped ``RunBudget`` deadline + strict cost cap.

Covers the §3.7 contract: ``remaining_seconds``, ``reserve_call`` /
``settle_call`` / ``check``, the strict fail-closed refusal of unknown pricing,
unknown usage, unbounded output and over-budget reservations, uncapped
(legacy) behavior, poisoning, JSON-safe snapshots, and concurrency safety.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito.session_ctx import (RunBudget, RunBudgetError, SessionContext,
                                  current_run_budget, set_run_budget,
                                  use_session)

_PRICES = {"input": 1.0, "output": 3.0, "cached": 0.25, "cache_write": 1.25}


# ---------------------------------------------------------------- deadline
def test_remaining_seconds_returns_none_when_uncapped():
    assert RunBudget().remaining_seconds() is None


def test_remaining_seconds_clamped_to_zero(monkeypatch):
    monkeypatch.setattr("pedacito.session_ctx.time.monotonic", lambda: 100.0)
    budget = RunBudget(deadline_monotonic=110.0)
    assert budget.remaining_seconds() == 10.0
    monkeypatch.setattr("pedacito.session_ctx.time.monotonic", lambda: 150.0)
    assert budget.remaining_seconds() == 0.0


def test_check_raises_when_deadline_passed(monkeypatch):
    monkeypatch.setattr("pedacito.session_ctx.time.monotonic", lambda: 200.0)
    budget = RunBudget(deadline_monotonic=100.0)
    with pytest.raises(RunBudgetError) as e:
        budget.check()
    assert "deadline" in str(e.value).lower()


def test_check_passes_before_deadline(monkeypatch):
    monkeypatch.setattr("pedacito.session_ctx.time.monotonic", lambda: 50.0)
    RunBudget(deadline_monotonic=100.0).check()   # does not raise


# ------------------------------------------------------------- strict cost
def test_strict_is_max_cost_positive():
    assert RunBudget(max_cost_usd=0.0).strict is False
    assert RunBudget(max_cost_usd=0.01).strict is True


def test_reserve_refuses_unknown_pricing():
    budget = RunBudget(max_cost_usd=1.0)
    with pytest.raises(RunBudgetError) as e:
        budget.reserve_call(1000, 100, {})
    assert "pricing" in str(e.value).lower()
    # A strict budget that failed to reserve is poisoned, never silently
    # downgraded to estimates.
    with pytest.raises(RunBudgetError):
        budget.check()


def test_reserve_refuses_missing_output_price():
    budget = RunBudget(max_cost_usd=1.0)
    with pytest.raises(RunBudgetError):
        budget.reserve_call(1000, 100, {"input": 1.0})


def test_reserve_refuses_unbounded_output():
    budget = RunBudget(max_cost_usd=1.0)
    with pytest.raises(RunBudgetError) as e:
        budget.reserve_call(1000, 0, _PRICES)
    assert "output" in str(e.value).lower()


def test_reserve_refuses_over_budget():
    budget = RunBudget(max_cost_usd=0.01)
    with pytest.raises(RunBudgetError) as e:
        budget.reserve_call(100_000, 10_000, _PRICES)
    assert "exceed" in str(e.value).lower()
    assert budget.reserved_usd == 0.0   # nothing was reserved


def test_reserve_computes_conservative_maximum():
    budget = RunBudget(max_cost_usd=1.0)
    token = budget.reserve_call(100_000, 10_000, _PRICES)
    assert token
    # Conservative: every input token at the most expensive input price
    # (cache_write=1.25 beats input=1.0), every output token at 3.0.
    expected = (100_000 * 1.25 + 10_000 * 3.0) / 1_000_000
    assert abs(budget.reserved_usd - expected) < 1e-9
    assert budget.spent_usd == 0.0


def test_settle_charges_actual_and_releases_reservation():
    budget = RunBudget(max_cost_usd=1.0)
    token = budget.reserve_call(100_000, 10_000, _PRICES)
    reserved = budget.reserved_usd
    budget.settle_call(token, 0.05)
    assert budget.reserved_usd == 0.0
    assert budget.spent_usd == 0.05
    assert reserved > 0.0


def test_settle_refuses_unknown_usage_and_poisons():
    budget = RunBudget(max_cost_usd=1.0)
    token = budget.reserve_call(1000, 100, _PRICES)
    with pytest.raises(RunBudgetError) as e:
        budget.settle_call(token, None)
    assert "usage" in str(e.value).lower()
    with pytest.raises(RunBudgetError):
        budget.check()                     # poisoned: no further calls


def test_settle_refuses_unknown_reservation_token():
    budget = RunBudget(max_cost_usd=1.0)
    with pytest.raises(RunBudgetError):
        budget.settle_call("res_does_not_exist", 0.0)


def test_check_raises_when_strict_budget_exhausted():
    budget = RunBudget(max_cost_usd=0.10, spent_usd=0.10)
    with pytest.raises(RunBudgetError) as e:
        budget.check()
    assert "budget" in str(e.value).lower()


# -------------------------------------------------------------- uncapped
def test_uncapped_reserve_and_settle_are_permissive():
    budget = RunBudget(max_cost_usd=0.0)
    # Unknown pricing / unbounded output do NOT refuse when uncapped.
    token = budget.reserve_call(1000, 0, {})
    assert token
    budget.settle_call(token, None)        # unknown usage is tolerated
    budget.settle_call(token, 0.0)         # double-settle is a silent no-op
    budget.check()                          # never raises (no deadline, no cap)
    assert budget.spent_usd == 0.0


def test_uncapped_still_records_known_usage():
    budget = RunBudget(max_cost_usd=0.0)
    token = budget.reserve_call(1000, 100, _PRICES)
    budget.settle_call(token, 0.42)
    assert budget.spent_usd == 0.42


# -------------------------------------------------------------- snapshot
def test_snapshot_is_json_safe_and_reflects_state(monkeypatch):
    monkeypatch.setattr("pedacito.session_ctx.time.monotonic", lambda: 5.0)
    budget = RunBudget(deadline_monotonic=15.0, max_cost_usd=2.0,
                       reserved_usd=0.25, spent_usd=0.5)
    snap = budget.snapshot()
    assert snap["remaining_seconds"] == 10.0
    assert snap["strict"] is True and snap["ok"] is True
    assert snap["reserved_usd"] == 0.25 and snap["spent_usd"] == 0.5
    import json
    json.dumps(snap)                       # no non-JSON types


# ------------------------------------------------------------- session
def test_run_budget_is_session_scoped_and_falls_back_to_cfg():
    ctx = SessionContext()
    with use_session(ctx):
        assert current_run_budget() is None
        budget = RunBudget(max_cost_usd=1.0)
        set_run_budget(budget)
        assert current_run_budget() is budget
        assert ctx.run_budget is budget
    assert current_run_budget() is None    # contextvar reset on exit


def test_headless_cfg_fallback(monkeypatch):
    class Cfg:
        pass
    cfg = Cfg()
    budget = RunBudget(max_cost_usd=0.5)
    set_run_budget(budget, cfg)
    assert current_run_budget(cfg) is budget


# ----------------------------------------------------------- concurrency
def test_concurrent_reservations_never_exceed_budget():
    budget = RunBudget(max_cost_usd=1.0)
    errors: list[Exception] = []
    settled: list[float] = []
    lock = threading.Lock()

    def worker(n: int):
        try:
            for _ in range(50):
                token = budget.reserve_call(10_000, 1_000, _PRICES)
                budget.settle_call(token, 0.0005)
                with lock:
                    settled.append(0.0005)
        except RunBudgetError as e:
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every successfully settled call was real (spent matches settled sum), and
    # the total spent never exceeded the cap. Refusals only happened because
    # the cap was genuinely exhausted (reserve raises before over-committing).
    assert abs(budget.spent_usd - sum(settled)) < 1e-6
    assert budget.spent_usd <= 1.0 + 1e-6
    assert budget.reserved_usd == 0.0


def test_poison_is_sticky_and_thread_safe():
    budget = RunBudget(max_cost_usd=1.0)
    token = budget.reserve_call(1000, 100, _PRICES)
    with pytest.raises(RunBudgetError):
        budget.settle_call(token, None)
    # After poisoning, every subsequent operation refuses, from any thread.
    results: list[bool] = []
    def probe():
        try:
            budget.check()
            results.append(True)
        except RunBudgetError:
            results.append(False)
    threads = [threading.Thread(target=probe) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [False, False, False, False]
