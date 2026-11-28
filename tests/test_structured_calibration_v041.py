"""WP-4101B: structured generation and calibration.

Covers: `Config.edit_mode` (legacy | structured) and its model-card precedence;
the strict `STRUCTURED_SCHEMA`; the three-arm edit-mode scoring
(quote/anchor/structured) with a 0.05 minimum improvement and ties preferring
the current legacy arm; the structured edit-format probe; the agent's
structured answer (bounded re-ask, `schema_rejection` events without the raw
reply, `schema_enforcement` honesty) and its atomic apply; and
`LLMClient.last_schema_enforcement` recording the rung the engine actually used.

Hermetic: scripted clients and a fake HTTP transport; no network, no model.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import config as config_mod
from pedacito import index as index_mod
from pedacito import modelcard
from pedacito.agent import Agent
from pedacito.config import Config, ConfigError
from pedacito.edits import (
    STRUCTURED_ACTIONS,
    STRUCTURED_SCHEMA,
    parse_structured_reply,
)
from pedacito.llm import LLMClient

SRC = "def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n"


# --------------------------------------------------------------- config
def test_edit_mode_defaults_to_legacy():
    assert Config().edit_mode == "legacy"


def test_edit_mode_accepts_structured():
    assert config_mod._coerce("edit_mode", "structured") == "structured"
    assert Config(edit_mode="structured").edit_mode == "structured"


def test_edit_mode_rejects_unknown():
    with pytest.raises(ConfigError):
        config_mod._coerce("edit_mode", "sidebar")


def test_edit_mode_env_coerces():
    assert config_mod._parse_env("edit_mode", "structured") == "structured"
    with pytest.raises(ConfigError):
        config_mod._parse_env("edit_mode", "bogus")


# --------------------------------------------------------------- schema
def test_structured_schema_is_strict_and_matches_the_parser():
    assert STRUCTURED_SCHEMA["type"] == "object"
    assert STRUCTURED_SCHEMA["additionalProperties"] is False
    assert set(STRUCTURED_SCHEMA["required"]) == {"version", "intents", "edits"}
    assert STRUCTURED_SCHEMA["properties"]["version"] == \
        {"type": "integer", "const": 1}
    actions = STRUCTURED_SCHEMA["properties"]["edits"]["items"]["properties"][
        "edit_action"]["enum"]
    assert set(actions) == set(STRUCTURED_ACTIONS)
    # the explicit no-change reply is an empty object with empty arrays
    assert STRUCTURED_SCHEMA["properties"]["intents"]["items"]["properties"][
        "why"]["maxLength"] == 300


def test_schema_accepts_the_no_change_reply():
    # the schema has no minItems, so the no-change object validates
    intents, edits = parse_structured_reply(
        '{"version":1,"intents":[],"edits":[]}')
    assert intents == [] and edits == []


# ---------------------------------------------------- modelcard three-arm score
def _measured(quote, anchor=None, structured=None):
    m = {"tool_choice": 0.6, "json_ok": 1.0, "edit_format": quote,
         "instructions": 1, "honesty": 1, "needle": 1, "continuation": 1}
    scores = {"quote": quote, "anchor": anchor if anchor is not None else quote}
    if structured is not None:
        scores["structured"] = structured
    m["edit_mode_scores"] = scores
    return m


def test_choose_prefers_legacy_when_no_arm_improves():
    ch = modelcard.choose(_measured(0.5, structured=0.54), False)
    assert "edit_mode" not in ch and "edit_style" not in ch


def test_choose_adopts_structured_only_at_minimum_improvement():
    # 0.60 - 0.50 = 0.10 >= 0.05 and structured beats anchor: structured wins
    ch = modelcard.choose(_measured(0.5, anchor=0.5, structured=0.60), False)
    assert ch["edit_mode"] == "structured"
    # exactly the 0.05 minimum: with a 2-decimal rounding an exact 0.05 delta
    # still clears the epsilon-free comparison (improvement >= 0.05)
    ch = modelcard.choose(_measured(0.5, anchor=0.5, structured=0.55), False)
    assert ch["edit_mode"] == "structured"


def test_choose_prefers_anchor_when_structured_is_absent_or_loses():
    ch = modelcard.choose(_measured(0.5, anchor=0.60), False)
    assert ch["edit_style"] == "anchor" and "edit_mode" not in ch
    # structured present but does not beat anchor: anchor wins
    ch = modelcard.choose(_measured(0.5, anchor=0.60, structured=0.56), False)
    assert ch["edit_style"] == "anchor" and "edit_mode" not in ch


def test_choose_legacy_fallback_keeps_flat_key_gain():
    # Old cards (no edit_mode_scores) keep the historical 0.1 arm gain.
    base = {"tool_choice": 0.6, "json_ok": 1.0, "edit_format": 0.5,
            "instructions": 1, "honesty": 1, "needle": 1, "continuation": 1}
    assert "edit_style" not in modelcard.choose(base, False)
    assert modelcard.choose({**base, "edit_format_anchor": 0.7},
                            False)["edit_style"] == "anchor"
    assert "edit_style" not in modelcard.choose(
        {**base, "edit_format_anchor": 0.55}, False)


def _cal_cfg(**kw):
    cfg = Config(base_url="http://127.0.0.1:11434/v1", model="mystery-7b")
    cfg.auto_refresh_model_catalog = False
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


class _CalClient:
    """Minimal scripted client for `calibrate()`: answers the step schema and
    returns no usable edits, so every edit arm scores 0.0."""

    chars_per_token = 3.2
    stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
             "cached_tokens": 0, "cache_write_tokens": 0,
             "seconds": 0.0, "cost": 0.0, "by_site": {}}

    def chat_json(self, messages, schema, max_tokens, **kw):
        return {"action": "done", "reasoning": "r"}

    def chat(self, messages, max_tokens, **kw):
        return "ok"

    def model_info(self):
        return {}


def test_calibrate_stores_edit_mode_scores_for_all_three_arms(home):
    card = modelcard.calibrate(_CalClient(), _cal_cfg(), verbose=False,
                               quick=True)
    scores = card["measured"]["edit_mode_scores"]
    assert set(scores) == {"quote", "anchor", "structured"}
    assert scores["quote"] == card["measured"]["edit_format"]
    assert scores["anchor"] == card["measured"]["edit_format_anchor"]
    assert scores["structured"] == card["measured"]["edit_format_structured"]


def test_apply_fills_edit_mode_only_when_unset(home):
    cfg = Config(base_url="http://127.0.0.1:11434/v1", model="m")
    modelcard.save_card({
        "version": modelcard.CARD_VERSION,
        "key": modelcard.card_key(cfg), "ts": time.time(),
        "model": "m", "url": cfg.base_url,
        "measured": {"overall": 0.9}, "choices": {"edit_mode": "structured"},
        "notes": [], "repeats": 1, "seconds": 1.0})
    fresh = Config(base_url="http://127.0.0.1:11434/v1", model="m")
    fresh.sources = {}
    modelcard.apply(fresh)
    assert fresh.edit_mode == "structured"
    assert fresh.sources["edit_mode"] == "model card"
    # an explicit user value wins
    tuned = Config(base_url="http://127.0.0.1:11434/v1", model="m")
    tuned.sources = {"edit_mode": "profile 'x' (via --profile; explicit)"}
    tuned.edit_mode = "legacy"
    modelcard.apply(tuned)
    assert tuned.edit_mode == "legacy"


# ------------------------------------------------ the structured edit probe
_FIXTURE = ("def is_leech(self, lapses):\n    return lapses >= 8\n\n"
            "def next_due(self):\n    return self.queue[0]\n")


def _sr(intents, edits, version=1):
    return json.dumps({"version": version, "intents": intents, "edits": edits})


def _intent(file="src/m.py", symbol="", why="fix"):
    return {"file": file, "symbol": symbol, "why": why}


def _edit(action, file="src/m.py", symbol="", search="", replacement=""):
    return {"edit_action": action, "file": file, "symbol": symbol,
            "search": search, "replacement": replacement}


class _StructuredModel:
    """Answers the six paired tasks as valid strict structured JSON."""

    def __init__(self, fail=False):
        self.fail = fail

    def chat(self, messages, max_tokens, **kw):
        assert kw.get("json_schema") is STRUCTURED_SCHEMA
        if self.fail:
            return "not json at all"
        last = messages[-1]["content"]
        if "threshold from 8 to self.threshold" in last:
            return _sr([_intent("src/m.py", "is_leech", "use self.threshold")],
                       [_edit("search_replace", symbol="is_leech",
                              search="lapses >= 8",
                              replacement="lapses >= self.threshold")])
        if "return None when self.queue is empty" in last:
            return _sr([_intent("src/m.py", "next_due", "return None")],
                       [_edit("replace_symbol", symbol="next_due",
                              replacement="def next_due(self):\n"
                              "    return self.queue[0] if self.queue else None")])
        if "Rename is_leech" in last:
            return _sr([_intent("src/m.py", "is_leech", "rename"),
                        _intent("src/m.py", "", "add the new name")],
                       [_edit("delete_symbol", symbol="is_leech"),
                        _edit("append", replacement="def is_due(self, lapses):"
                              "\n    return lapses >= 8")])
        if "IndexError" in last:
            return _sr([_intent("src/m.py", "next_due", "raise IndexError")],
                       [_edit("replace_symbol", symbol="next_due",
                              replacement="def next_due(self):\n"
                              "    if not self.queue:\n"
                              "        raise IndexError('empty queue')\n"
                              "    return self.queue[0]")])
        if "docstring" in last:
            return _sr([_intent("src/m.py", "next_due", "document it")],
                       [_edit("replace_symbol", symbol="next_due",
                              replacement="def next_due(self):\n"
                              "    \"\"\"Return the next due item.\"\"\"\n"
                              "    return self.queue[0]")])
        if "THRESHOLD" in last:
            return _sr([_intent("src/m.py", "is_leech", "use the constant")],
                       [_edit("search_replace", symbol="is_leech",
                              search="lapses >= 8",
                              replacement="lapses >= THRESHOLD")])
        return _sr([_intent("src/m.py", "is_leech", "fix")],
                   [_edit("search_replace", symbol="is_leech",
                          search="lapses >= 8", replacement="lapses >= 10")])


def test_structured_edit_probe_scores_a_strict_model():
    score, notes = modelcard._t_edit_format_structured(
        _StructuredModel(), Config())
    assert score == 1.0 and notes == []


def test_structured_edit_probe_rejects_non_json():
    score, notes = modelcard._t_edit_format_structured(
        _StructuredModel(fail=True), Config())
    assert score == 0.0 and notes


# ----------------------------------------------- agent structured apply
class _Events:
    def __init__(self):
        self.emitted = []

    def emit(self, kind, **fields):
        self.emitted.append((kind, fields))


def _agent(tmp_path, **cfg_attrs):
    root = tmp_path / "proj"
    root.mkdir(parents=True, exist_ok=True)
    (root / "app.py").write_text(SRC, encoding="utf-8")
    cfg = Config(edit_mode="structured")
    cfg.sources = {}
    cfg.auto_allowlist = False
    for k, v in cfg_attrs.items():
        setattr(cfg, k, v)
    idx = index_mod.build([str(root)], cfg, client=None, summarise=False,
                          root=root, verbose=False)
    agent = Agent(idx, None, cfg)
    agent._read_spans = {"app.py": [(1, 7)]}
    agent.events = _Events()
    return root, agent, cfg


def _replace_beta(replacement="def beta():\n    return 22"):
    return _sr([_intent("app.py", "beta", "fix it")],
               [_edit("replace_symbol", "app.py", "beta",
                      replacement=replacement)])


def test_apply_structured_applies_a_valid_reply(tmp_path):
    root, agent, cfg = _agent(tmp_path)
    results, reply = agent.apply(_replace_beta(), dry_run=False, task="t")
    assert any(r.ok for r in results)
    text = (root / "app.py").read_text(encoding="utf-8")
    assert "return 22" in text
    assert "    return 2\n" not in text


def test_apply_structured_dry_run_writes_nothing(tmp_path):
    root, agent, cfg = _agent(tmp_path)
    results, reply = agent.apply(_replace_beta(), dry_run=True, task="t")
    assert any(r.ok for r in results)
    assert (root / "app.py").read_text(encoding="utf-8") == SRC


def test_apply_structured_no_change_reply_is_empty(tmp_path):
    root, agent, cfg = _agent(tmp_path)
    results, reply = agent.apply('{"version":1,"intents":[],"edits":[]}',
                                 dry_run=False, task="t")
    assert results == []
    assert (root / "app.py").read_text(encoding="utf-8") == SRC


def test_apply_structured_rejects_bad_json_without_applying(tmp_path):
    root, agent, cfg = _agent(tmp_path)
    raw = "sure, here: " + _replace_beta()      # prose-wrapped, no salvage
    results, reply = agent.apply(raw, dry_run=False, task="t")
    assert results and not any(r.ok for r in results)
    assert (root / "app.py").read_text(encoding="utf-8") == SRC
    # a schema_rejection event, never the raw reply
    kinds = [k for k, _f in agent.events.emitted]
    assert "schema_rejection" in kinds
    for _k, fields in agent.events.emitted:
        assert raw not in json.dumps(fields)


def test_apply_structured_rejects_an_invalid_edit(tmp_path):
    root, agent, cfg = _agent(tmp_path)
    # targets a symbol that does not exist: local validation must refuse it
    reply = _sr([_intent("app.py", "gamma", "fix it")],
                [_edit("replace_symbol", "app.py", "gamma",
                       replacement="def gamma():\n    return 3")])
    results, reply2 = agent.apply(reply, dry_run=False, task="t")
    assert results and not any(r.ok for r in results)
    assert (root / "app.py").read_text(encoding="utf-8") == SRC


# --------------------------------------------- agent structured answer
class _AnswerClient:
    """Returns a scripted sequence of raw replies; records the schema rung."""

    def __init__(self, replies, enforcement="strict"):
        self.replies = list(replies)
        self.calls = 0
        self.last_schema_enforcement = enforcement
        self.chars_per_token = 0.0

    def chat(self, messages, max_tokens, json_schema=None, stream=False, **kw):
        self.calls += 1
        assert json_schema is STRUCTURED_SCHEMA
        if self.calls <= len(self.replies):
            return self.replies[self.calls - 1]
        return self.replies[-1]


def test_answer_structured_returns_a_valid_reply_on_first_try(tmp_path):
    root, agent, cfg = _agent(tmp_path)
    agent.client = _AnswerClient([_replace_beta()])
    out = agent.answer("Fix beta", "SOURCE\n" + SRC, stream=False)
    assert "return 22" in out
    assert agent.client.calls == 1
    kinds = [k for k, _f in agent.events.emitted]
    assert "answer" in kinds and "schema_rejection" not in kinds
    ans = [f for k, f in agent.events.emitted if k == "answer"][0]
    assert ans["mode"] == "structured"
    assert ans["schema_enforcement"] == "strict"


def test_answer_structured_reasks_boundedly_on_rejection(tmp_path):
    root, agent, cfg = _agent(tmp_path)
    agent.client = _AnswerClient(["not json", _replace_beta()])
    out = agent.answer("Fix beta", "SOURCE\n" + SRC, stream=False)
    assert "return 22" in out
    assert agent.client.calls == 2
    kinds = [k for k, _f in agent.events.emitted]
    assert kinds.count("schema_rejection") == 1
    # the rejection reason never persists the raw reply text
    for k, fields in agent.events.emitted:
        assert "not json" not in json.dumps(fields)


def test_answer_structured_gives_up_after_the_reask_budget(tmp_path):
    root, agent, cfg = _agent(tmp_path)
    agent.client = _AnswerClient(["bad"] * 10)
    out = agent.answer("Fix beta", "SOURCE\n" + SRC, stream=False)
    # exhausted: the explicit no-change object, never a half-validated reply
    assert json.loads(out) == {"version": 1, "intents": [], "edits": []}
    # 1 initial + _MAX_STRUCTURED_REASKS re-asks, all rejected
    assert agent.client.calls == 1 + 2


# ------------------------------------------------ schema_enforcement honesty
def _cfg(**kw):
    cfg = Config()
    cfg.base_url = "http://127.0.0.1:1/v1"
    cfg.model = "m"
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


class _Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = json.dumps(body) if body is not None else ""
        self.headers = {}
        self.closed = False

    def json(self):
        return self._body

    def close(self):
        self.closed = True


def _ok(content):
    return _Resp(200, {"choices": [{"message": {"content": content,
                                                "finish_reason": "stop"}}],
                       "usage": {}})


def _patch_post(monkeypatch, fn):
    monkeypatch.setattr("pedacito.llm.requests.post", fn)


def test_last_schema_enforcement_records_the_real_rung(monkeypatch):
    captured = {}

    def fake(url, **kw):
        captured["payload"] = kw.get("json") or {}
        return _ok('{"version":1,"intents":[],"edits":[]}')

    _patch_post(monkeypatch, fake)
    c = LLMClient(_cfg(json_mode="schema"))
    c.chat([{"role": "user", "content": "x"}], 200,
           json_schema=STRUCTURED_SCHEMA, stream=False)
    assert c.last_schema_enforcement == "strict"
    assert captured["payload"]["response_format"]["type"] == "json_schema"

    # json_object mode records json_object, never "sampler-enforced"
    c2 = LLMClient(_cfg(json_mode="object"))
    c2.chat([{"role": "user", "content": "x"}], 200,
            json_schema=STRUCTURED_SCHEMA, stream=False)
    assert c2.last_schema_enforcement == "json_object"

    # opt-out / prompt-only is recorded honestly as prompt_only
    c3 = LLMClient(_cfg(json_mode="schema", enforce_json_schema=False))
    c3.chat([{"role": "user", "content": "x"}], 200,
            json_schema=STRUCTURED_SCHEMA, stream=False)
    assert c3.last_schema_enforcement == "prompt_only"


# ----------------------------------------------- a tiny 'home' card fixture
@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(modelcard, "_path",
                        lambda: tmp_path / "home" / "model_cards.json")
    (tmp_path / "home").mkdir()
    return tmp_path / "home"
