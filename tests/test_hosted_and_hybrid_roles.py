"""Bug Hunt #11 crucible: hosted providers and hybrid role routing.

Pins the machinery a local-worker + hosted-planner/verifier/escalate run
depends on: hosted detection and role-profile derivation, the shared usage
ledger across role clients, provider-specific cache-token accounting and
cost math, budget stops, explicit cache markers, local-only acceleration
keys, and the escalation ladder's routing policy (reason fired -> escalate
role; green-first/red-second keeps the first; no reason -> no hosted spend).

Hermetic: scripted clients and direct accounting calls, no network.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import config as config_mod
from pedacito import llm as llm_mod
from pedacito import loop, roles
from pedacito.config import Config
from pedacito.index import Index

DEEPSEEK = "https://api.deepseek.com/v1"
LOCAL = "http://127.0.0.1:11434/v1"


def _hosted_cfg(**kw):
    cfg = Config()
    cfg.base_url = DEEPSEEK
    cfg.model = "deepseek-flash"
    cfg.sources = {}
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


class FakeClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.on_call = None
        self.site = ""

    def check(self, warn=True):
        return ""

    def warmup(self):
        pass


# ------------------------------------------------- hosted + role derivation
def test_hosted_detection_and_role_profile_derivation():
    cfg = _hosted_cfg()
    assert config_mod.is_hosted(cfg) is True
    assert config_mod.is_hosted(Config()) is False
    cfg.profiles = {"flash": {"model": "deepseek-flash",
                              "base_url": DEEPSEEK, "tier": "hosted"}}
    derived = roles.derive_config(cfg, "flash")
    assert derived is not None and derived is not cfg
    assert derived.active_profile == "flash"
    assert derived._hosted_server is True
    assert derived.json_mode == "object"      # hosted default, not inherited
    assert roles.derive_config(cfg, "ghost") is None


def test_hybrid_roles_get_distinct_clients_and_one_ledger(monkeypatch):
    monkeypatch.setattr(llm_mod, "LLMClient", FakeClient)
    worker = FakeClient(Config())
    seen = []
    worker.on_call = seen.append
    cfg = Config()
    cfg.sources = {}
    cfg.profiles = {"flash": {"model": "deepseek-flash",
                              "base_url": DEEPSEEK, "tier": "hosted"}}
    cfg.roles = {"planner": "flash", "verifier": "flash",
                 "escalate": "flash"}
    clients = [roles.client_for(cfg, r, worker)
               for r in ("planner", "verifier", "escalate")]
    for c in clients:
        assert c is not worker
        assert c.cfg.model == "deepseek-flash"
        assert c.on_call is worker.on_call     # one usage ledger
    # Cached per role; the worker config is untouched.
    assert roles.client_for(cfg, "planner", worker) is clients[0]
    assert cfg.model == Config().model
    assert roles.client_for(cfg, "summarizer", worker) is worker


def test_role_falls_back_when_profile_unknown_or_down(monkeypatch):
    monkeypatch.setattr(llm_mod, "LLMClient", FakeClient)
    worker = FakeClient(Config())
    cfg = Config()
    cfg.sources = {}
    cfg.profiles = {"flash": {"model": "m", "base_url": DEEPSEEK}}
    cfg.roles = {"planner": "ghost", "verifier": "flash"}
    assert roles.client_for(cfg, "planner", worker) is worker

    class Down(FakeClient):
        def check(self, warn=True):
            raise llm_mod.LLMError("unreachable")

    monkeypatch.setattr(llm_mod, "LLMClient", Down)
    cfg2 = Config()
    cfg2.sources = {}
    cfg2.profiles = {"flash": {"model": "m", "base_url": DEEPSEEK}}
    cfg2.roles = {"escalate": "flash"}
    assert roles.client_for(cfg2, "escalate", worker) is worker


# ------------------------------------------------------ usage + cost math
@pytest.mark.parametrize("usage,cached,written", [
    # DeepSeek: implicit hits at the top level.
    ({"prompt_tokens": 1000, "completion_tokens": 100,
      "prompt_cache_hit_tokens": 600}, 600, 0),
    # OpenAI: details sub-object.
    ({"prompt_tokens": 1000, "completion_tokens": 100,
      "prompt_tokens_details": {"cached_tokens": 400}}, 400, 0),
    # Anthropic/DashScope: explicit read + write.
    ({"prompt_tokens": 1000, "completion_tokens": 100,
      "cache_read_input_tokens": 300,
      "cache_creation_input_tokens": 200}, 300, 200),
])
def test_account_reads_provider_cache_fields(usage, cached, written):
    client = llm_mod.LLMClient(_hosted_cfg())
    rec = client._account(usage, None, 0.5, prompt_est=999)
    assert rec["cached_tokens"] == cached
    assert rec["cache_write_tokens"] == written
    assert client.stats["cached_tokens"] == cached
    assert client.stats["prompt_tokens"] == 1000


def test_cost_math_prices_cached_and_written_separately():
    cfg = _hosted_cfg(price_input_per_mtok=2.0,
                      price_output_per_mtok=8.0,
                      price_cached_per_mtok=0.2,
                      price_cache_write_per_mtok=2.5)
    client = llm_mod.LLMClient(cfg)
    rec = client._account({"prompt_tokens": 1_000_000,
                           "completion_tokens": 100_000,
                           "prompt_cache_hit_tokens": 400_000,
                           "cache_creation_input_tokens": 100_000},
                          None, 1.0, prompt_est=0)
    # fresh=500k*2 + cached=400k*0.2 + write=100k*2.5 + out=100k*8 (per Mtok)
    assert rec["cost"] == pytest.approx(
        (500_000 * 2.0 + 400_000 * 0.2 + 100_000 * 2.5
         + 100_000 * 8.0) / 1_000_000)


def test_budgets_stop_before_the_next_call():
    cfg = _hosted_cfg(max_cost=0.25)
    client = llm_mod.LLMClient(cfg)
    client._check_budget()                     # under budget: fine
    client.stats["cost"] = 0.25
    with pytest.raises(llm_mod.BudgetExceeded):
        client._check_budget()
    cfg2 = _hosted_cfg(max_total_tokens=1000)
    client2 = llm_mod.LLMClient(cfg2)
    client2.stats["prompt_tokens"] = 900
    client2.stats["completion_tokens"] = 100
    with pytest.raises(llm_mod.BudgetExceeded):
        client2._check_budget()


# ------------------------------------------------- cache markers + accel
def test_cache_markers_only_in_explicit_mode():
    msgs = [{"role": "system", "content": "SYS"},
            {"role": "user", "content": "U0"},
            {"role": "assistant", "content": "A"},
            {"role": "user", "content": "U1"}]
    explicit = llm_mod.LLMClient(_hosted_cfg(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"))
    import copy
    m = copy.deepcopy(msgs)
    assert explicit._apply_cache_markers(m) is True
    marked = [i for i, x in enumerate(m)
              if isinstance(x["content"], list)]
    assert marked == [0, 1, 3]                 # system, user0, last
    assert m[0]["content"][0]["cache_control"]["type"] == "ephemeral"

    implicit = llm_mod.LLMClient(_hosted_cfg())
    m2 = copy.deepcopy(msgs)
    assert implicit._apply_cache_markers(m2) is False
    assert m2 == msgs


def test_acceleration_keys_are_local_only():
    local = llm_mod.LLMClient(Config())
    local.cfg.base_url = LOCAL
    payload = {}
    local._apply_acceleration(payload)
    assert payload.get("cache_prompt") is True
    hosted = llm_mod.LLMClient(_hosted_cfg())
    payload2 = {}
    hosted._apply_acceleration(payload2)
    assert payload2 == {}


# ------------------------------------------------------ escalation policy
def test_escalation_reason_first_wins_and_emits_event(tmp_path):
    cfg = Config()
    cfg.log_events = True
    cfg.sources = {}
    loop._suggest_escalation(tmp_path, cfg, "first reason")
    loop._suggest_escalation(tmp_path, cfg, "second reason")
    assert cfg._task_escalation_reason == "first reason"
    files = list((tmp_path / ".pedacito" / "events").glob("*.jsonl"))
    assert files
    kinds = [json.loads(ln).get("kind")
             for ln in files[0].read_text(encoding="utf-8").splitlines()]
    assert kinds.count("escalation_suggested") == 1


def _tr(idx, applied, tests_ok, reply="did work"):
    return loop.TaskResult(results=[], reply=reply, applied=applied,
                           changed=["a.py"] if applied else [], idx=idx,
                           tests_ok=tests_ok)


def test_ladder_escalates_only_when_a_reason_fired(tmp_path, monkeypatch):
    idx = Index(tmp_path, [])
    cfg = Config()
    cfg.sources = {}
    cfg.roles = {"escalate": "flash"}
    worker, esc = FakeClient(Config()), FakeClient(Config())
    monkeypatch.setattr(roles, "client_for",
                        lambda c, role, default: esc if role == "escalate"
                        else default)
    calls = []

    def fake_once(root, cfg_, client, agent, idx_, task, **kw):
        calls.append(client)
        if client is worker:
            cfg_._task_escalation_reason = "two no-progress leases"
            return _tr(idx_, applied=False, tests_ok=None)
        return _tr(idx_, applied=True, tests_ok=True)

    monkeypatch.setattr(loop, "_run_task_once", fake_once)
    tr = loop._run_ladder(tmp_path, cfg, worker, None, idx, "fix it",
                          apply=True, review=False)
    assert calls == [worker, esc]              # reason fired -> hosted retry
    assert tr.applied is True

    # No reason + an applied edit: the escalate role is never consulted.
    calls.clear()
    cfg._task_escalation_reason = ""
    cfg._pending_escalation_reason = ""

    def green_once(root, cfg_, client, agent, idx_, task, **kw):
        calls.append(client)
        return _tr(idx_, applied=True, tests_ok=True)

    monkeypatch.setattr(loop, "_run_task_once", green_once)
    tr2 = loop._run_ladder(tmp_path, cfg, worker, None, idx, "fix it",
                           apply=True, review=False)
    assert calls == [worker] and tr2.applied is True


def test_ladder_keeps_green_first_attempt_over_red_escalation(
        tmp_path, monkeypatch):
    idx = Index(tmp_path, [])
    cfg = Config()
    cfg.sources = {}
    cfg.roles = {"escalate": "flash"}
    worker, esc = FakeClient(Config()), FakeClient(Config())
    monkeypatch.setattr(roles, "client_for",
                        lambda c, role, default: esc if role == "escalate"
                        else default)

    def fake_once(root, cfg_, client, agent, idx_, task, **kw):
        if client is worker:
            cfg_._task_escalation_reason = "semantic failure outlived repair"
            return _tr(idx_, applied=True, tests_ok=True)
        return _tr(idx_, applied=True, tests_ok=False)

    monkeypatch.setattr(loop, "_run_task_once", fake_once)
    tr = loop._run_ladder(tmp_path, cfg, worker, None, idx, "fix it",
                          apply=True, review=False)
    assert tr.tests_ok is True                 # green first wins