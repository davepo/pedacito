"""WP-4105E: provider-aware retries and stream resume safety.

Covers the §3.7 contract: ``RetryPolicy`` (delta-seconds AND HTTP-date
backoff, cap, cancellable deadline-clamped sleep), the ``dedupe_stream_chunks``
resume helper, idempotency/stream-resume gating on ``ProviderCapabilities``,
and the strict RunBudget reservation integration through the LLM client
(unknown usage fails closed, known usage settles, unknown pricing refuses).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import config as config_mod
from pedacito import llm as llm_mod
from pedacito import transport_anthropic, transport_openai
from pedacito.llm import (BudgetExceeded, LLMClient, RetryPolicy,
                          dedupe_stream_chunks, parse_retry_after)
from pedacito.providers import ProviderCapabilities
from pedacito.session_ctx import (RunBudget, SessionContext, set_run_budget,
                                  use_session)


def _cfg(**kw):
    cfg = config_mod.Config()
    cfg.base_url = "http://127.0.0.1:1/v1"
    cfg.model = "m"
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


class Resp:
    def __init__(self, status=200, body=None, text="", headers=None):
        self.status_code = status
        self._body = body
        self.text = text or (json.dumps(body) if body is not None else "")
        self.headers = headers or {}

    def json(self):
        return self._body


def _ok(text="hello", usage=None):
    return Resp(200, {"choices": [{"message": {"content": text},
                                   "finish_reason": "stop"}],
                      "usage": usage or {}})


# ---------------------------------------------------------- Retry-After
def test_parse_retry_after_delta_seconds():
    assert parse_retry_after("3") == 3.0
    assert parse_retry_after("0") == 0.0
    assert parse_retry_after(" 12.5 ") == 12.5


def test_parse_retry_after_http_date_future():
    now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert parse_retry_after("Thu, 01 Jan 2026 00:00:30 GMT", now_utc=now) \
        == 30.0


def test_parse_retry_after_http_date_past_clamps_to_zero():
    now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert parse_retry_after("Wed, 31 Dec 2025 23:59:59 GMT", now_utc=now) \
        == 0.0


def test_parse_retry_after_garbage_is_none():
    assert parse_retry_after("") is None
    assert parse_retry_after("soon-ish") is None
    assert parse_retry_after(None) is None


def test_retry_after_on_a_response_parses_both_forms():
    r = Resp(429, text="", headers={"Retry-After": "3"})
    assert llm_mod._retry_after(r) == 3.0
    # A past HTTP-date parses as 0.0, not None and not an exception.
    r = Resp(429, text="", headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"})
    assert llm_mod._retry_after(r) == 0.0


# ------------------------------------------------------------ RetryPolicy
def test_retry_delay_exponential_backoff_capped():
    policy = RetryPolicy(max_attempts=5, base_s=2.0, cap_s=10.0)
    assert policy.retry_delay(1, None) == 2.0
    assert policy.retry_delay(2, None) == 4.0
    assert policy.retry_delay(3, None) == 8.0
    assert policy.retry_delay(4, None) == 10.0   # 16.0 capped to 10.0


def test_retry_delay_honours_server_delta_after():
    policy = RetryPolicy(max_attempts=3, base_s=2.0, cap_s=120.0)
    assert policy.retry_delay(1, "42") == 42.0
    assert policy.retry_delay(2, "3") == 3.0     # server wins over backoff


def test_retry_delay_honours_server_http_date():
    now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    policy = RetryPolicy(max_attempts=3, base_s=2.0, cap_s=120.0)
    assert policy.retry_delay(1, "Thu, 01 Jan 2026 00:00:15 GMT", now_utc=now) \
        == 15.0


def test_retry_delay_jitter_uses_injected_rng():
    policy = RetryPolicy(max_attempts=3, base_s=2.0, cap_s=100.0)
    class Rng:
        def uniform(self, a, b):
            return b          # the maximum jitter bound
    delay = policy.retry_delay(1, None, rng=Rng())
    assert delay == 2.0 + 0.5    # jitter = 0.25 * max(2.0, 1.0) = 0.5


def test_retry_policy_is_frozen_and_hashable():
    policy = RetryPolicy(max_attempts=3, base_s=1.0, cap_s=5.0)
    assert {policy: "x"}[policy] == "x"
    with pytest.raises(Exception):
        policy.max_attempts = 9


def test_sleep_caps_to_deadline(monkeypatch):
    policy = RetryPolicy()
    sleeps = []
    monkeypatch.setattr("pedacito.llm.time.sleep", sleeps.append)
    now = [100.0]
    monkeypatch.setattr("pedacito.llm.time.monotonic", lambda: now[0])
    slept = policy.sleep(30.0, deadline_monotonic=105.0)
    assert slept == 5.0
    # No single slice exceeds 0.25s.
    assert all(s <= 0.25 for s in sleeps)


# ----------------------------------------------------------- stream resume
def test_dedupe_stream_chunks_drops_duplicates_keeps_empty_ids():
    chunks = [("a", "hello"), ("b", " "), ("a", "dupe"), ("", "raw"),
              ("c", "world"), ("b", "dupe2")]
    new, seen = dedupe_stream_chunks(chunks)
    assert [t for _, t in new] == ["hello", " ", "raw", "world"]
    assert seen == {"a", "b", "c"}
    # A resumed feed with the same ids delivers nothing new.
    again, _ = dedupe_stream_chunks([("a", "x"), ("c", "y")], seen)
    assert again == []


def test_stream_resume_supported_defaults_false(monkeypatch):
    client = LLMClient(_cfg())
    assert client.stream_resume_supported() is False
    monkeypatch.setattr(
        LLMClient, "_capabilities",
        lambda self: ProviderCapabilities(stream_resume=True))
    assert client.stream_resume_supported() is True


# ------------------------------------------------------------- idempotency
def test_idempotency_key_requires_documented_capability(monkeypatch):
    client = LLMClient(_cfg())
    # Unknown capability (the default for a local runtime): no key.
    assert client.idempotency_key() is None

    monkeypatch.setattr(
        LLMClient, "_capabilities",
        lambda self: ProviderCapabilities(idempotency=True))
    key = client.idempotency_key()
    assert key and key.startswith("pdc-")
    assert client.idempotency_key() != key   # a fresh key per logical request


def test_transport_headers_attach_idempotency_only_when_given():
    assert "Idempotency-Key" not in transport_openai.build_headers("k")
    assert transport_openai.build_headers("k", idempotency_key="pdc-1") \
        ["Idempotency-Key"] == "pdc-1"

    assert "idempotency-key" not in transport_anthropic.build_headers("k")
    assert transport_anthropic.build_headers("k", idempotency_key="pdc-2") \
        ["idempotency-key"] == "pdc-2"


# -------------------------------------------------- strict reservation (llm)
def test_llm_reserve_and_settle_against_strict_budget(monkeypatch):
    cfg = _cfg(price_input_per_mtok=1.0, price_output_per_mtok=3.0)
    ctx = SessionContext()
    budget = RunBudget(max_cost_usd=1.0)
    set_run_budget(budget, cfg)
    with use_session(ctx):
        client = LLMClient(cfg)
        token = client._reserve_call([{"role": "user", "content": "hi"}], 100)
        assert token is not None
        assert budget.reserved_usd > 0.0
        client._settle_call(0.01)
        assert budget.spent_usd == 0.01
        assert budget.reserved_usd == 0.0


def test_llm_reserve_refuses_unknown_pricing(monkeypatch):
    cfg = _cfg()                       # no price fields
    ctx = SessionContext()
    set_run_budget(RunBudget(max_cost_usd=1.0), cfg)
    with use_session(ctx):
        client = LLMClient(cfg)
        with pytest.raises(BudgetExceeded):
            client._reserve_call([{"role": "user", "content": "hi"}], 100)


def test_llm_strict_call_with_usage_settles(monkeypatch):
    monkeypatch.setattr(
        "pedacito.llm.requests.post",
        lambda *a, **k: _ok("done", usage={"prompt_tokens": 10,
                                           "completion_tokens": 5}))
    cfg = _cfg(price_input_per_mtok=1.0, price_output_per_mtok=3.0)
    ctx = SessionContext()
    budget = RunBudget(max_cost_usd=1.0)
    set_run_budget(budget, cfg)
    with use_session(ctx):
        client = LLMClient(cfg)
        assert client.chat([{"role": "user", "content": "x"}], 10) == "done"
    assert budget.spent_usd > 0.0
    assert budget.reserved_usd == 0.0


def test_llm_strict_call_without_usage_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "pedacito.llm.requests.post",
        lambda *a, **k: _ok("done"))       # no usage reported
    cfg = _cfg(price_input_per_mtok=1.0, price_output_per_mtok=3.0)
    ctx = SessionContext()
    budget = RunBudget(max_cost_usd=1.0)
    set_run_budget(budget, cfg)
    with use_session(ctx):
        client = LLMClient(cfg)
        with pytest.raises(BudgetExceeded):
            client.chat([{"role": "user", "content": "x"}], 10)
    # The strict cap is poisoned: it will not keep running on estimates.
    with pytest.raises(Exception):
        budget.check()


def test_llm_without_strict_budget_keeps_legacy_behavior(monkeypatch):
    monkeypatch.setattr(
        "pedacito.llm.requests.post",
        lambda *a, **k: _ok("done"))       # no usage, no prices: legacy fine
    cfg = _cfg()                            # max_cost unset -> uncapped
    client = LLMClient(cfg)
    assert client.chat([{"role": "user", "content": "x"}], 10) == "done"
