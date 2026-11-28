"""WP-4105E: the session-scoped ``RunBudget`` deadline + strict cost cap.

Covers the §3.7 contract: ``remaining_seconds``, ``reserve_call`` /
``settle_call`` / ``check``, the strict fail-closed refusal of unknown pricing,
unknown usage, unbounded output and over-budget reservations, uncapped
(legacy) behavior, poisoning, JSON-safe snapshots, and concurrency safety.
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
import types
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import llm as llm_mod  # noqa: E402
from pedacito import scaffold  # noqa: E402
from pedacito.config import Config  # noqa: E402
from pedacito.llm import BudgetExceeded, LLMClient, LLMError  # noqa: E402
from pedacito.session_ctx import (RunBudget, RunBudgetError, SessionContext,  # noqa: E402
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


# ------------------------------------------ bounded empty-reasoning retries
def _sse(obj) -> str:
    return "data: " + json.dumps(obj)


def _reasoning_only_stream() -> list[str]:
    """A streamed reply that spent its whole budget on hidden reasoning."""
    return [_sse({"choices": [{"delta": {"reasoning_content": "hmm " * 40},
                               "finish_reason": None}]}),
            _sse({"choices": [{"delta": {}, "finish_reason": "length"}]})]


class _Resp:
    status_code = 200
    headers: dict = {}

    def __init__(self, lines):
        self._lines = lines

    def iter_lines(self, decode_unicode=False):
        yield from self._lines

    def close(self):
        pass


def _cfg(tmp_path=None, **kw) -> Config:
    cfg = Config()
    cfg.base_url = "http://127.0.0.1:9/v1"
    cfg.model = "stub-model"
    cfg.log_sessions = False
    cfg.log_events = tmp_path is not None
    if tmp_path is not None:
        cfg._root = str(tmp_path)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _events(root: Path) -> list[dict]:
    out: list[dict] = []
    for path in sorted((root / ".pedacito" / "events").glob("*.jsonl")):
        out += [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return out


def test_reasoning_exhausted_stream_retries_twice_then_explains(monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    budgets = []

    def fake_post(url, **kw):
        budgets.append(kw["json"]["max_tokens"])
        return _Resp(_reasoning_only_stream())

    monkeypatch.setattr("pedacito.llm.requests.post", fake_post)
    with pytest.raises(LLMError) as e:
        LLMClient(_cfg()).chat([{"role": "user", "content": "hi"}], 2000,
                               stream=True, disable_thinking=True)
    assert len(budgets) == 3                    # the call plus two retries
    assert budgets == [2000, 2000, 2000]        # re-rolled, never escalated
    assert "2 retries" in str(e.value) and "finish 'length'" in str(e.value)
    assert "max_tokens_build" in str(e.value)


def test_empty_retry_never_starts_past_the_run_deadline(monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    cfg = _cfg()
    posts = []

    def fake_post(url, **kw):
        posts.append(1)
        cfg._task_deadline = time.monotonic() - 1.0   # the deadline passes here
        return _Resp(_reasoning_only_stream())

    monkeypatch.setattr("pedacito.llm.requests.post", fake_post)
    with pytest.raises(LLMError) as e:
        LLMClient(cfg).chat([{"role": "user", "content": "hi"}], 2000,
                            stream=True, disable_thinking=True)
    assert posts == [1]
    assert "time budget" in str(e.value)


def test_timeout_names_the_clamped_request_timeout_and_is_typed(tmp_path,
                                                                monkeypatch):
    def fake_post(url, **kw):
        raise requests.exceptions.Timeout("read timed out")

    monkeypatch.setattr("pedacito.llm.requests.post", fake_post)
    cfg = _cfg(tmp_path, timeout=600.0)
    cfg._task_deadline = time.monotonic() + 42.0
    with pytest.raises(LLMError) as e:
        LLMClient(cfg).chat([{"role": "user", "content": "hi"}], 100,
                            stream=False)
    msg = str(e.value)
    assert "did not answer within 600s" not in msg and "--max-minutes" in msg
    secs = float(re.search(r"within ([\d.]+)s", msg).group(1))
    assert 5 <= secs <= 42
    timeouts = [ev for ev in _events(tmp_path) if ev["kind"] == "timeout"]
    assert len(timeouts) == 1
    assert timeouts[0]["reason_code"] == "run_deadline"
    assert abs(timeouts[0]["timeout_s"] - secs) < 0.1


def test_unclamped_timeout_keeps_the_configured_value(tmp_path, monkeypatch):
    def fake_post(url, **kw):
        raise requests.exceptions.Timeout("read timed out")

    monkeypatch.setattr("pedacito.llm.requests.post", fake_post)
    with pytest.raises(LLMError) as e:
        LLMClient(_cfg(tmp_path, timeout=600.0)).chat(
            [{"role": "user", "content": "hi"}], 100, stream=False)
    assert "within 600s (`timeout` in your config)" in str(e.value)
    timeouts = [ev for ev in _events(tmp_path) if ev["kind"] == "timeout"]
    assert timeouts[0]["reason_code"] == "request_timeout"


def test_refused_call_records_budget_stop_before_any_request(tmp_path,
                                                             monkeypatch):
    def fake_post(url, **kw):
        raise AssertionError("no request may start past the cost cap")

    monkeypatch.setattr("pedacito.llm.requests.post", fake_post)
    client = LLMClient(_cfg(tmp_path, max_cost=0.000001))
    client.stats["cost"] = 0.5
    with pytest.raises(BudgetExceeded):
        client.chat([{"role": "user", "content": "hi"}], 100, stream=False)
    events = _events(tmp_path)
    assert [ev["kind"] for ev in events] == ["budget_stop"]
    assert events[0]["reason_code"] == "max_cost"


def test_strict_cost_refusal_records_budget_stop(tmp_path, monkeypatch):
    def fake_post(url, **kw):
        raise AssertionError("no request may start past the strict cap")

    monkeypatch.setattr("pedacito.llm.requests.post", fake_post)
    cfg = _cfg(tmp_path, strict_cost_budget=True, max_cost=0.000001)
    set_run_budget(RunBudget(max_cost_usd=0.000001), cfg)
    with pytest.raises(BudgetExceeded):
        LLMClient(cfg).chat([{"role": "user", "content": "hi"}], 100,
                            stream=False)
    events = _events(tmp_path)
    assert [ev["kind"] for ev in events] == ["budget_stop"]
    assert events[0]["reason_code"] == "strict_cost"


# ------------------------------------------- scaffold: bounded build + deadline
def _two_file_plan(tmp_path):
    for rel in ("a.py", "b.py"):
        (tmp_path / rel).write_text(f"# stub {rel}\n", encoding="utf-8")
    plan = scaffold.Plan(
        "p", "s",
        files=[scaffold.PlannedFile("a.py"), scaffold.PlannedFile("b.py")],
        spec=scaffold.Spec("build a and b"))
    return plan, types.SimpleNamespace(root=tmp_path)


def test_model_error_aborts_the_build_instead_of_retrying_the_next_file(
        tmp_path, monkeypatch):
    plan, idx = _two_file_plan(tmp_path)
    calls = []

    def build_pass(client, cfg, idx_, plan_, t, *a, **k):
        calls.append(t.path)
        raise LLMError("'stub' streamed no answer after 2 retries: each reply "
                       "spent the budget on hidden reasoning.")

    monkeypatch.setattr(scaffold, "build_pass", build_pass)
    summary = scaffold.build_project(None, _cfg(), idx, plan, verbose=False)
    assert calls == ["a.py"]
    assert summary["aborted"] is True and summary["done"] == []
    assert summary["failed"][0][0] == "a.py"
    assert summary["failed"][0][1].startswith("'stub' streamed no answer")
    assert summary["skipped"] == ["b.py"]


def test_build_starts_no_repair_pass_past_the_deadline(tmp_path, monkeypatch):
    plan, idx = _two_file_plan(tmp_path)
    clock = [0.0]
    monkeypatch.setattr(scaffold, "time",
                        types.SimpleNamespace(monotonic=lambda: clock[0]))
    calls = []

    def build_pass(client, cfg, idx_, plan_, t, *a, **k):
        calls.append(t.path)
        clock[0] = 20.0            # the deadline passes during this pass
        return [scaffold.Result(False, t.path, "rejected block")], ""

    monkeypatch.setattr(scaffold, "build_pass", build_pass)
    summary = scaffold.build_project(None, _cfg(), idx, plan, verbose=False,
                                     deadline=10.0)
    assert calls == ["a.py"]
    assert summary["failed"][0][0] == "a.py"


def test_verify_repair_rounds_stop_at_the_deadline(tmp_path, monkeypatch):
    from pedacito import loop
    monkeypatch.setattr(loop, "run_verify",
                        lambda root, cmd, cfg=None: (False, "AssertionError: boom"))
    monkeypatch.setattr(loop, "missing_modules", lambda out, root=None: [])
    repairs = []
    monkeypatch.setattr(scaffold, "_fix_from_output",
                        lambda *a, **k: repairs.append(1) or True)
    ok, out = scaffold.verify_and_fix(
        None, _cfg(), types.SimpleNamespace(root=tmp_path), [], "p", None,
        explicit_cmd="pytest", rounds=2, smoke=False, verbose=False,
        deadline=time.monotonic() - 1.0)
    assert ok is False and "boom" in out
    assert repairs == []


def test_run_new_closes_capped_when_the_deadline_passes_mid_build(
        tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(scaffold, "time", types.SimpleNamespace(
        monotonic=lambda: clock[0], time=time.time, sleep=time.sleep))
    monkeypatch.setattr(scaffold, "build_plan", lambda spec, client, cfg, root,
                        *a, **k: scaffold.Plan(
                            "p", "s", files=[scaffold.PlannedFile("a.py")],
                            spec=spec))
    monkeypatch.setattr(scaffold, "build_interface", lambda *a, **k: {})
    monkeypatch.setattr(scaffold, "apply_skeleton", lambda root, plan, **k: [])
    idx = types.SimpleNamespace(root=tmp_path / "proj")
    monkeypatch.setattr(scaffold, "skeleton_index",
                        lambda root, plan, cfg: idx)
    later = []

    def build_project(client, cfg, idx_, plan, **k):
        clock[0] = 100.0           # the one-minute budget runs out mid-build
        later.append(1)
        return {"done": ["a.py"], "noop": [], "failed": [], "skipped": [],
                "capped": False, "aborted": False}

    monkeypatch.setattr(scaffold, "build_project", build_project)
    monkeypatch.setattr(scaffold, "build_docs", lambda *a, **k: later.append(2)
                        or {"done": [], "noop": [], "failed": [], "skipped": []})
    code = scaffold.run_new(tmp_path / "proj", _cfg(), None,
                            scaffold.Spec("a tiny app"), apply=True,
                            verbose=False, max_minutes=1)
    assert code == 1
    assert later == [1]        # docs and verification never started
