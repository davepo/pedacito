"""A1.4: per-phase sampling temperature (`temperature_answer`,
`temperature_repair`) and the eval row's repair/mismatch counters.

Invariants: unset means the global temperature everywhere; gather never moves;
best_of's extra candidates stay hotter than the first reply and an outer
override is never reset; replay digests ignore temperature; config writes keep
comments."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from a1_support import READ_THEN_GREP, make_project
from pedacito import config as config_mod
from pedacito import evalrun, replay
from pedacito.agent import Agent
from pedacito.config import Config, ConfigError
from pedacito.gui import settings as S
from pedacito.llm import LLMClient
from test_features_misc import EvalModel, _eval_cfg
from test_llm_transport import _cfg, _ok, _patch_post

GOOD = ("--- FILE: pricing.py\n<<<<<<< SEARCH\n    if quantity > 10:\n=======\n"
        "    if quantity >= 10:\n>>>>>>> REPLACE\n")
MISS = ("--- FILE: pricing.py\n<<<<<<< SEARCH\n    if quantity > 99:\n=======\n"
        "    if quantity >= 99:\n>>>>>>> REPLACE\n")


class TempClient(LLMClient):
    """A real LLMClient whose chat/chat_json calls are scripted; it logs the
    temperature each call would have been sent at (the override, else the
    global value) and the call site."""

    step_truncated = False
    truncated_field = ""

    def __init__(self, cfg, replies=(), steps=()):
        LLMClient.__init__(self, cfg)
        self.replies = list(replies)
        self.steps = list(steps)
        self.log: list[tuple[str, float]] = []

    def _temp(self) -> float:
        return (self._temp_override if self._temp_override is not None
                else self.cfg.temperature)

    def chat_json(self, messages, schema, max_tokens, **kw):
        self.log.append(("step", self._temp()))
        step = self.steps.pop(0) if self.steps else {"action": "done"}
        return {"reasoning": "r", **step}

    def chat(self, messages, max_tokens, stream=False, **kw):
        self.log.append((self.site or "call", self._temp()))
        self.stream_truncated = False
        return self.replies.pop(0) if self.replies else ""


def _setup(tmp_path, replies=(), steps=(), **cfg_attrs):
    _root, idx, cfg = make_project(
        tmp_path, **{"post_edit_check": False, **cfg_attrs})
    client = TempClient(cfg, replies, steps)
    return Agent(idx, client, cfg), client, cfg


def _temps(client, site):
    return [t for s, t in client.log if s == site]


# ------------------------------------------------------------- off by default
def test_defaults_are_greedy_answer_and_global_repair():
    cfg0 = Config()
    assert cfg0.temperature_answer == 0.0 and cfg0.temperature_repair is None


def test_unset_means_the_global_temperature_in_every_phase(tmp_path):
    agent, client, cfg = _setup(tmp_path, replies=[MISS, GOOD, GOOD],
                                steps=READ_THEN_GREP,
                                temperature_answer=None,
                                temperature_repair=None)
    agent.gather("Fix bulk_discount", verbose=False)
    first = agent.answer("Fix bulk_discount", "src", stream=False)
    agent.apply(first, dry_run=False, task="Fix bulk_discount", gathered="src")
    assert client.log and all(t == cfg.temperature for _s, t in client.log)
    assert {s for s, _t in client.log} >= {"step", "answer", "repair"}


def test_the_default_answer_runs_greedy_while_repair_keeps_the_global(
        tmp_path):
    agent, client, cfg = _setup(tmp_path, replies=[MISS, GOOD],
                                steps=READ_THEN_GREP)
    agent.gather("Fix bulk_discount", verbose=False)
    reply = agent.answer("Fix bulk_discount", "src", stream=False)
    agent.apply(reply, dry_run=False, task="Fix bulk_discount", gathered="src")
    assert _temps(client, "step") == [cfg.temperature] * 3
    assert _temps(client, "answer") == [0.0]
    assert _temps(client, "repair") == [cfg.temperature]


# ------------------------------------------------------------------- new path
def test_answer_and_repair_run_at_their_own_temperatures(tmp_path):
    agent, client, cfg = _setup(
        tmp_path, replies=[MISS, GOOD], steps=READ_THEN_GREP,
        temperature_answer=0.0, temperature_repair=0.05)
    agent.gather("Fix bulk_discount", verbose=False)
    reply = agent.answer("Fix bulk_discount", "src", stream=False)
    results, _ = agent.apply(reply, dry_run=False, task="Fix bulk_discount",
                             gathered="src")
    assert all(r.ok for r in results)
    assert _temps(client, "step") == [cfg.temperature] * 3     # gather: global
    assert _temps(client, "answer") == [0.0]
    assert _temps(client, "repair") == [0.05]                  # the re-anchor
    assert client._temp_override is None                       # context closed


def test_the_two_settings_are_independent(tmp_path):
    agent, client, cfg = _setup(tmp_path, replies=[MISS, GOOD],
                                temperature_answer=0.0)
    reply = agent.answer("t", "src", stream=False)
    agent.apply(reply, dry_run=False, task="t", gathered="src")
    assert _temps(client, "answer") == [0.0]
    assert _temps(client, "repair") == [cfg.temperature]
    agent, client, cfg = _setup(tmp_path / "b", replies=[MISS, GOOD],
                                temperature_answer=None, temperature_repair=0.0)
    reply = agent.answer("t", "src", stream=False)
    agent.apply(reply, dry_run=False, task="t", gathered="src")
    assert _temps(client, "answer") == [cfg.temperature]
    assert _temps(client, "repair") == [0.0]


def test_answer_temperature_also_covers_the_continue_mode(tmp_path):
    agent, client, cfg = _setup(tmp_path, replies=[GOOD], steps=READ_THEN_GREP,
                                temperature_answer=0.0, answer_mode="continue")
    agent.gather("Fix bulk_discount", verbose=False)
    agent.answer("Fix bulk_discount", "src", stream=False)
    assert agent.answer_mode_used == "continue"
    assert _temps(client, "answer") == [0.0]


def test_the_repair_re_ask_from_findings_uses_the_repair_temperature(tmp_path):
    from pedacito import verify
    agent, client, cfg = _setup(tmp_path, replies=[GOOD],
                                temperature_repair=0.0, temperature_answer=0.7)
    err = verify.Finding("placeholder", "error", "pricing.py", "stubbed")
    new = agent._repair_from_findings("t", MISS, [err], [])
    assert "SEARCH" in new
    assert client.log == [("call", 0.0)]


def test_a_role_client_uses_its_own_profile_not_the_workers(tmp_path):
    agent, main, cfg = _setup(tmp_path, replies=[GOOD], temperature_repair=0.5)
    role_cfg = Config()
    role_cfg.temperature_repair = 0.0
    role = TempClient(role_cfg, [GOOD])
    agent._role_client = lambda name: role
    from pedacito import verify
    agent._repair_from_findings(
        "t", MISS, [verify.Finding("placeholder", "error", "pricing.py", "x")], [])
    assert role.log == [("call", 0.0)] and main.log == []
    other = TempClient(Config(), [GOOD])         # a role with nothing set
    agent._role_client = lambda name: other
    agent._repair_from_findings(
        "t", MISS, [verify.Finding("placeholder", "error", "pricing.py", "x")], [])
    assert other.log == [("call", other.cfg.temperature)]


def test_the_post_edit_fix_round_is_a_repair(tmp_path):
    broken = ("--- FILE: pricing.py\n<<<<<<< SEARCH\n    return 0.0\n=======\n"
              "    return missing_name\n>>>>>>> REPLACE\n")
    fixed = ("--- FILE: pricing.py\n<<<<<<< SEARCH\n    return missing_name\n"
             "=======\n    return 0.0\n>>>>>>> REPLACE\n")
    agent, client, cfg = _setup(tmp_path, replies=[fixed],
                                temperature_repair=0.0, temperature_answer=0.7,
                                post_edit_check=True)
    agent.apply(broken, dry_run=False, task="t", gathered="src")
    assert client.log == [("repair", 0.0)]


def test_a_continuation_after_a_deferral_is_an_answer(tmp_path):
    first = "Fixed the threshold; the rest follows next turn.\n" + GOOD
    more = ("--- FILE: pricing.py\n<<<<<<< SEARCH\n    return 0.0\n=======\n"
            "    return 0.0  # none\n>>>>>>> REPLACE\n")
    agent, client, cfg = _setup(tmp_path, replies=[more, "NOTHING MORE"],
                                temperature_answer=0.0, temperature_repair=0.3)
    agent.apply(first, dry_run=False, task="t", gathered="src")
    assert client.log[0] == ("repair", 0.0)       # the completion round


# ------------------------------------------------------------------- best_of
def _best_of_setup(tmp_path, **cfg_attrs):
    return _setup(tmp_path, replies=[GOOD, GOOD, GOOD], best_of=3, **cfg_attrs)


def test_best_of_candidates_run_hotter_than_the_first_reply(tmp_path):
    agent, client, cfg = _best_of_setup(tmp_path, temperature_answer=0.0)
    first = agent.answer("Fix bulk_discount", "src", stream=False)
    agent.best_of("Fix bulk_discount", first, "src", verbose=False)
    # the first reply at 0.0; both extra candidates at 0.0 + 0.4, not reset
    assert _temps(client, "answer") == [0.0, 0.4, 0.4]


def test_best_of_offset_is_unchanged_when_unset(tmp_path):
    agent, client, cfg = _best_of_setup(tmp_path, temperature_answer=None)
    first = agent.answer("Fix bulk_discount", "src", stream=False)
    agent.best_of("Fix bulk_discount", first, "src", verbose=False)
    assert _temps(client, "answer") == [0.2, pytest.approx(0.6), pytest.approx(0.6)]


def test_best_of_never_exceeds_one(tmp_path):
    agent, client, cfg = _best_of_setup(tmp_path, temperature_answer=0.9)
    first = agent.answer("Fix bulk_discount", "src", stream=False)
    agent.best_of("Fix bulk_discount", first, "src", verbose=False)
    assert _temps(client, "answer") == [0.9, 1.0, 1.0]


# ----------------------------------------------------- clients that cannot
def test_clients_without_a_temperature_method_are_untouched(tmp_path):
    _root, idx, cfg = make_project(tmp_path, temperature_answer=0.0,
                                   temperature_repair=0.0, post_edit_check=False)
    client = EvalModel(MISS)
    assert not hasattr(client, "temperature")
    agent = Agent(idx, client, cfg)
    assert agent.answer("t", "src", stream=False) == MISS
    agent.apply(MISS, dry_run=False, task="t", gathered="src")


def test_replay_digests_ignore_temperature(tmp_path):
    _root, idx, cfg = make_project(tmp_path)
    path = tmp_path / "rec.jsonl"
    rec = replay.Recorder(EvalModel(GOOD, READ_THEN_GREP), path)
    agent = Agent(idx, rec, cfg)
    agent.gather("Fix bulk_discount", verbose=False)
    agent.answer("Fix bulk_discount", "src", stream=False)
    _r, idx2, hot = make_project(tmp_path / "hot", temperature_answer=0.0,
                                 temperature_repair=0.0)
    rep = replay.Replayer(path)
    agent2 = Agent(idx2, rep, hot)
    agent2.gather("Fix bulk_discount", verbose=False)
    assert agent2.answer("Fix bulk_discount", "src", stream=False) == GOOD
    assert rep.diverged == [] and len(rep.used) == len(rep.records)


# ------------------------------------------------------------- the real payload
def test_the_payload_carries_zero_not_the_global_value(monkeypatch):
    posts = []
    _patch_post(monkeypatch, lambda *a, **k: posts.append(k["json"]) or _ok())
    client = LLMClient(_cfg(temperature=0.2))
    with client.temperature(0.0):
        client.chat([{"role": "user", "content": "x"}], 10)
    client.chat([{"role": "user", "content": "x"}], 10)
    assert [p["temperature"] for p in posts] == [0.0, 0.2]


def test_a_phase_temperature_beats_a_sampler_temperature_key(monkeypatch):
    posts = []
    _patch_post(monkeypatch, lambda *a, **k: posts.append(k["json"]) or _ok())
    client = LLMClient(_cfg(temperature=0.2, sampler={"temperature": 0.9,
                                                      "top_k": 40}))
    with client.temperature(0.0):
        client.chat([{"role": "user", "content": "x"}], 10)
    client.chat([{"role": "user", "content": "x"}], 10)
    assert [p["temperature"] for p in posts] == [0.0, 0.9]   # unchanged outside
    assert all(p["top_k"] == 40 for p in posts)


# ------------------------------------------------------------- eval counters
class SiteModel(EvalModel):
    """EvalModel whose stats count calls by site, like LLMClient's."""

    def __init__(self, replies, steps):
        super().__init__("", steps)
        self.replies = list(replies)
        self.site = ""

    def chat(self, messages, max_tokens, **kw):
        row = self.stats["by_site"].setdefault(self.site or "call", {"calls": 0})
        row["calls"] += 1
        self.stats["calls"] += 1
        return self.replies.pop(0) if self.replies else self.reply


def _bulk_task():
    return next(t for t in evalrun.TASKS if t.id == "py-bulk-off-by-one")


SHOP_MISS = ("--- FILE: shop/pricing.py\n<<<<<<< SEARCH\n    if quantity > 99:\n"
             "=======\n    if quantity >= 99:\n>>>>>>> REPLACE\n")
SHOP_GOOD = ("--- FILE: shop/pricing.py\n<<<<<<< SEARCH\n    if quantity > 10:\n"
             "=======\n    if quantity >= 10:\n>>>>>>> REPLACE\n")
READ_SHOP = [{"action": "read", "file": "shop/pricing.py", "start": 1, "end": 30}]


def test_eval_rows_count_search_mismatches_and_repair_calls(tmp_path):
    task = _bulk_task()
    client = SiteModel([SHOP_MISS, SHOP_GOOD], READ_SHOP)
    row = evalrun.run_task(task, client, _eval_cfg(), tmp_path)
    assert row["search_mismatches"] == 1 and row["repair_calls"] == 1
    assert row["passed"] and not row["first_apply_clean"]
    clean = evalrun.run_task(task, SiteModel([SHOP_GOOD], READ_SHOP),
                             _eval_cfg(), tmp_path)
    assert clean["search_mismatches"] == 0 and clean["repair_calls"] == 0
    assert clean["first_apply_clean"]


def test_repair_calls_are_per_task_not_cumulative_over_the_client(tmp_path):
    """by_site is cumulative over a client's life; a task reports its own."""
    task = _bulk_task()
    client = SiteModel([SHOP_MISS, SHOP_GOOD, SHOP_GOOD], READ_SHOP)
    first = evalrun.run_task(task, client, _eval_cfg(), tmp_path)
    client.steps = READ_SHOP
    client.n = 0
    second = evalrun.run_task(task, client, _eval_cfg(), tmp_path)
    assert first["repair_calls"] == 1
    assert second["repair_calls"] == 0 and second["first_apply_clean"]


def test_summary_reports_the_counters_for_edit_tasks(tmp_path):
    rows = [{"id": "a", "kind": "edit", "passed": True, "repair_calls": 1,
             "search_mismatches": 2},
            {"id": "b", "kind": "edit", "passed": True, "repair_calls": 0,
             "search_mismatches": 0},
            {"id": "c", "kind": "question", "passed": True}]
    s = evalrun.summarise(rows)
    assert s["mean_repair_calls"] == 0.5 and s["mean_search_mismatches"] == 1.0
    assert "0.50 repair calls and 1.00 SEARCH mismatches per task" \
        in evalrun.render(rows, s)


# --------------------------------------------------------------- config (I4)
def test_temperatures_are_validated_and_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "USER_CONFIG", tmp_path / "none.toml")
    S.validate_text("temperature_answer = 0\ntemperature_repair = 0.1\n")
    for bad in ("-0.5", '"cold"', "true"):
        with pytest.raises(ConfigError, match="non-negative number"):
            S.validate_text(f"temperature_answer = {bad}\n")
    proj = tmp_path / "p"
    proj.mkdir()
    (tmp_path / "c.toml").write_text(
        "vram_gb = 8\ntemperature_answer = 0\n", encoding="utf-8")
    cfg = config_mod.load(root=proj, explicit=tmp_path / "c.toml",
                          use_env=False, require_hardware=False,
                          warn_budgets=False)
    assert cfg.temperature_answer == 0.0 and isinstance(cfg.temperature_answer, float)
    assert cfg.temperature_repair is None


def test_temperatures_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "USER_CONFIG", tmp_path / "none.toml")
    proj = tmp_path / "p"
    proj.mkdir()
    for raw, want in (("0", 0.0), ("0.15", 0.15), ("auto", None), ("", None)):
        monkeypatch.setenv("PEDACITO_TEMPERATURE_REPAIR", raw)
        cfg = config_mod.load(root=proj, require_hardware=False,
                              warn_budgets=False)
        assert cfg.temperature_repair == want, raw
    monkeypatch.setenv("PEDACITO_TEMPERATURE_REPAIR", "-1")
    with pytest.raises(ConfigError):
        config_mod.load(root=proj, require_hardware=False, warn_budgets=False)
    monkeypatch.setenv("PEDACITO_TEMPERATURE_REPAIR", "hot")
    with pytest.raises(ConfigError, match="not a valid number"):
        config_mod.load(root=proj, require_hardware=False, warn_budgets=False)


def test_config_writer_keeps_comments_for_the_temperatures():
    text = config_mod.TEMPLATE
    out = S.set_values(text, {"temperature_answer": 0.0, "temperature_repair": 0.0})
    S.validate_text(out)
    live = [ln for ln in out.splitlines() if ln.startswith("temperature_")]
    assert live == ["temperature_answer = 0.0", "temperature_repair = 0.0"]
    kept = [ln for ln in out.splitlines() if ln not in live]
    assert kept == text.splitlines()          # no comment or key was lost
    assert "# temperature_answer = 0.0" in out
    import tomllib
    tomllib.loads(config_mod.REFERENCE_TEMPLATE)
    names = {i["name"]: i for i in S.schema()}
    assert names["temperature_answer"]["type"] == "float"
    assert names["temperature_answer"]["default"] == 0.0
