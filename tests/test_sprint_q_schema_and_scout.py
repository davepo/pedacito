"""Sprint Q: engine-enforced JSON grammar (response_format ladder with a
per-endpoint downgrade) and dual-model scout/architect gather routing.

Invariants: `build_gather_json_schema` is strict and tracks the dynamically
pruned tool set exactly; `schema_for` still builds the byte-identical gather
schema (prompt-cache stability); llm.py asks the ENGINE to constrain sampling
on every schema'd call, and a server that 400s the constraint is downgraded
json_schema -> json_object -> plain, once, immediately, remembered per
(base_url, model). `cfg.scout_profile` moves only the gather loop to another
profile's client; unknown/unreachable scouts fall back softly."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from a1_support import READ_THEN_GREP, Scripted, make_project
from pedacito import config as config_mod
from pedacito import llm as llm_mod
from pedacito import roles as roles_mod
from pedacito import tools
from pedacito.agent import Agent
from pedacito.config import Config
from pedacito.gui import settings as S
from pedacito.llm import LLMClient, LLMError
from test_llm_transport import Resp, _cfg, _ok, _patch_post, _sse


@pytest.fixture(autouse=True)
def _fresh_json_caps():
    """The capability cache is process-wide; keep tests independent."""
    llm_mod._JSON_CAPS.clear()
    yield
    llm_mod._JSON_CAPS.clear()


def _step_schema(**kw):
    return tools.build_gather_json_schema(
        ["read", "grep", "done"], kw.pop("reasoning_first", True), **kw)


def _done_reply():
    return _ok('{"action": "done", "file": "", "start": 0, "end": 0, '
               '"pattern": "", "symbol": "", "reasoning": "r"}')


# ------------------------------------------------------- the schema builder
def test_enum_matches_the_allowed_actions_exactly():
    actions = ["read", "grep", "done"]
    s = tools.build_gather_json_schema(actions, True)
    assert s["properties"]["action"]["enum"] == actions
    assert s["type"] == "object"
    assert s["additionalProperties"] is False


def test_strict_shape_requires_every_present_field():
    for rf in (True, False):
        s = tools.build_gather_json_schema(["read", "write", "done"], rf)
        assert set(s["required"]) == set(s["properties"])


def test_reasoning_position_follows_the_flag():
    a = tools.build_gather_json_schema(["read", "done"], False)
    assert list(a["properties"])[0] == "action"
    assert a["required"][-1] == "reasoning"
    b = tools.build_gather_json_schema(["read", "done"], True)
    assert list(b["properties"])[0] == "reasoning"
    assert b["required"][0] == "reasoning"


def test_unused_fields_are_dropped_and_required_follows():
    s = tools.build_gather_json_schema(["read", "done"], True)
    assert "to" not in s["properties"] and "command" not in s["properties"]
    assert "to" not in s["required"] and "command" not in s["required"]
    w = tools.build_gather_json_schema(["run", "done"], True)
    assert "command" in w["properties"] and "command" in w["required"]


def test_web_and_run_test_fields_splice_in_only_with_their_actions():
    base = tools.build_gather_json_schema(["read", "done"], True)
    assert "url" not in base["properties"] and "target" not in base["properties"]
    w = tools.build_gather_json_schema(
        ["read", tools.FETCH_URL_ACTION, tools.WEB_SEARCH_ACTION, "done"], True)
    assert "url" in w["properties"] and "query" in w["properties"]
    assert set(w["required"]) == set(w["properties"])
    t = tools.build_gather_json_schema(
        ["read", tools.RUN_TEST_ACTION, "done"], True)
    assert "target" in t["properties"] and "filter" in t["properties"]


def test_compact_keeps_only_action_and_reasoning_required():
    s = tools.build_gather_json_schema(["read", "write", "done"], True,
                                       compact=True)
    assert sorted(s["required"]) == ["action", "reasoning"]
    assert len(s["properties"]) > 2


def test_builder_does_not_mutate_STEP_SCHEMA():
    before = json.dumps(tools.STEP_SCHEMA, sort_keys=True)
    tools.build_gather_json_schema(["read", "done"], True)
    tools.build_gather_json_schema(["run", "done"], False, compact=True)
    assert json.dumps(tools.STEP_SCHEMA, sort_keys=True) == before


def _drop_for(cfg):
    d = set()
    if not cfg.allow_write:
        d.add("to")
    if not (cfg.allow_shell or cfg.allow_serve or cfg.allow_screenshots
            or cfg.allow_browser):
        d.add("command")
    return d


@pytest.mark.parametrize("attrs", [
    {},
    {"allow_write": True, "backup": True},
    {"allow_shell": True, "allow_serve": True},
    {"enable_web": True},
    {"allow_fetch": True, "allow_subagents": True},
    {"step_reasoning_first": False},
    {"compact_step_schema": True},
])
def test_schema_for_delegates_byte_identically(attrs):
    cfg = Config()
    cfg.sources = {}
    for k, v in attrs.items():
        setattr(cfg, k, v)
    s = tools.schema_for(cfg)
    built = tools.build_gather_json_schema(
        s["properties"]["action"]["enum"], tools.reasoning_first(cfg),
        compact=bool(getattr(cfg, "compact_step_schema", False)),
        drop=_drop_for(cfg))
    assert json.dumps(built) == json.dumps(s)


# -------------------------------------------------- engine ladder, buffered
def test_schema_call_attaches_strict_response_format(monkeypatch):
    seen = {}

    def fake(url, **kw):
        seen.update(kw["json"])
        return _done_reply()

    _patch_post(monkeypatch, fake)
    c = LLMClient(_cfg())
    out = c.chat_json([{"role": "user", "content": "go"}], _step_schema(), 200)
    rf = seen["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["name"] == "pedacito_action"
    assert rf["json_schema"]["schema"] == _step_schema()
    assert out["action"] == "done"


def test_plain_calls_never_carry_a_response_format(monkeypatch):
    posts = []
    _patch_post(monkeypatch,
                 lambda *a, **k: posts.append(dict(k["json"])) or _ok("hi"))
    c = LLMClient(_cfg())
    assert c.chat([{"role": "user", "content": "x"}], 20) == "hi"
    assert "response_format" not in posts[0]


def test_json_schema_refusal_downgrades_to_object(monkeypatch):
    posts = []

    def fake(url, **kw):
        p = dict(kw["json"])
        posts.append(p)
        if (p.get("response_format") or {}).get("type") == "json_schema":
            return Resp(400, text='{"error": "json_schema is not supported '
                                  'by this server"}')
        return _done_reply()

    _patch_post(monkeypatch, fake)
    c = LLMClient(_cfg())
    out = c.chat_json([{"role": "user", "content": "go"}], _step_schema(), 200)
    assert out["action"] == "done"
    assert len(posts) == 2                     # retried immediately
    assert posts[0]["response_format"]["type"] == "json_schema"
    assert posts[1]["response_format"] == {"type": "json_object"}
    caps = llm_mod._json_caps(c.cfg.base_url, c.cfg.model)
    assert caps["schema"] is False and caps["object"] is True


def test_double_refusal_ends_at_plain_prompting(monkeypatch):
    posts = []

    def fake(url, **kw):
        p = dict(kw["json"])
        posts.append(p)
        rf = (p.get("response_format") or {}).get("type")
        if rf == "json_schema":
            return Resp(400, text='{"error": "json_schema is not supported"}')
        if rf == "json_object":
            return Resp(400, text='{"error": "unknown field response_format"}')
        return _done_reply()

    _patch_post(monkeypatch, fake)
    c = LLMClient(_cfg())
    out = c.chat_json([{"role": "user", "content": "go"}], _step_schema(), 200)
    assert out["action"] == "done"
    assert len(posts) == 3
    assert "response_format" not in posts[2]
    assert "Reply with one JSON object only" in posts[2]["messages"][-1]["content"]
    caps = llm_mod._json_caps(c.cfg.base_url, c.cfg.model)
    assert caps == {"schema": False, "object": False}
    # remembered for the rest of the process: the next call starts at plain
    posts.clear()
    fake2 = lambda url, **kw: (posts.append(dict(kw["json"])) or _done_reply())
    _patch_post(monkeypatch, fake2)
    c.chat_json([{"role": "user", "content": "go"}], _step_schema(), 200)
    assert len(posts) == 1 and "response_format" not in posts[0]


def test_refusal_is_cached_per_base_url_and_model(monkeypatch):
    posts = []

    def fake(url, **kw):
        p = dict(kw["json"])
        posts.append(p)
        rf = (p.get("response_format") or {}).get("type")
        if rf == "json_schema" and p["model"] == "m":
            return Resp(400, text='{"error": "json_schema is not supported"}')
        return _done_reply()

    _patch_post(monkeypatch, fake)
    c1 = LLMClient(_cfg())                       # model "m" refuses
    c1.chat_json([{"role": "user", "content": "go"}], _step_schema(), 200)
    c1.chat_json([{"role": "user", "content": "go"}], _step_schema(), 200)
    c2 = LLMClient(_cfg(model="other"))          # untouched endpoint
    c2.chat_json([{"role": "user", "content": "go"}], _step_schema(), 200)
    tiers = [(p.get("response_format") or {}).get("type") for p in posts]
    assert tiers == ["json_schema", "json_object", "json_object", "json_schema"]


def test_object_mode_refusal_drops_to_plain(monkeypatch):
    posts = []

    def fake(url, **kw):
        p = dict(kw["json"])
        posts.append(p)
        if (p.get("response_format") or {}).get("type") == "json_object":
            return Resp(400, text='{"error": "response_format unsupported"}')
        return _done_reply()

    _patch_post(monkeypatch, fake)
    c = LLMClient(_cfg(json_mode="object"))
    c.chat_json([{"role": "user", "content": "go"}], _step_schema(), 200)
    assert posts[0]["response_format"] == {"type": "json_object"}
    assert len(posts) == 2 and "response_format" not in posts[1]
    caps = llm_mod._json_caps(c.cfg.base_url, c.cfg.model)
    assert caps["object"] is False and caps["schema"] is True


def test_unrelated_400_on_plain_does_not_loop(monkeypatch):
    posts = []

    def fake(url, **kw):
        p = dict(kw["json"])
        posts.append(p)
        return Resp(400, text='{"error": "response_format is not valid here"}')

    _patch_post(monkeypatch, fake)
    c = LLMClient(_cfg())
    with pytest.raises(LLMError):
        c.chat_json([{"role": "user", "content": "go"}], _step_schema(), 200)
    # one downgrade rung per tier, then the plain failure is raised as-is
    assert len(posts) == 3
    assert [(p.get("response_format") or {}).get("type") for p in posts] \
        == ["json_schema", "json_object", None]


def test_enforce_off_never_asks_the_engine(monkeypatch):
    posts = []
    _patch_post(monkeypatch,
                 lambda url, **kw: posts.append(dict(kw["json"]))
                 or _done_reply())
    c = LLMClient(_cfg(enforce_json_schema=False))
    assert c.cfg.enforce_json_schema is False
    c.chat_json([{"role": "user", "content": "go"}], _step_schema(), 200)
    assert "response_format" not in posts[0]
    assert "Reply with one JSON object only" in posts[0]["messages"][-1]["content"]


def test_default_config_enforces(monkeypatch):
    assert Config().enforce_json_schema is True


# --------------------------------------------------- engine ladder, streaming
def test_streaming_structured_call_downgrades(monkeypatch):
    posts = []
    lines = [
        _sse({"choices": [{"delta": {"content": "hi"},
                           "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        "data: [DONE]",
    ]

    def fake(url, **kw):
        p = dict(kw["json"])
        posts.append(p)
        if (p.get("response_format") or {}).get("type") == "json_schema":
            return Resp(400, text='{"error": "json_schema is not supported"}')
        return Resp(200, lines=lines)

    _patch_post(monkeypatch, fake)
    c = LLMClient(_cfg())
    out = c.chat([{"role": "user", "content": "go"}], 50,
                 schema=_step_schema(), stream=True)
    assert out == "hi"
    assert len(posts) == 2
    assert posts[1]["response_format"] == {"type": "json_object"}


# ------------------------------------------------------------ scout routing
class _ScoutBase(Scripted):
    """Scripted model plus the little server-hygiene the real client offers
    (check/warmup) and a mutable `site`, so roles.scout_client accepts it."""

    def check(self, warn=True):
        return True

    def warmup(self):
        return None


def _profile_cfg(**kw):
    cfg = Config()
    cfg.sources = {}
    cfg.log_events = False
    cfg.log_sessions = False
    cfg.profiles = {"scout": {"model": "tiny-fast",
                              "base_url": "http://127.0.0.1:9/v1"},
                    "arch": {"model": "big-slow"}}
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_scout_client_soft_cases(monkeypatch):
    made = []
    monkeypatch.setattr(llm_mod, "LLMClient",
                        lambda c: made.append(c) or _ScoutBase([]))
    d = object()
    cfg = _profile_cfg()
    assert roles_mod.scout_client(cfg, d) is d          # nothing configured
    cfg.scout_profile = "arch"
    cfg.active_profile = "arch"
    assert roles_mod.scout_client(cfg, d) is d          # scout IS the active
    cfg.scout_profile = "ghost"
    assert roles_mod.scout_client(cfg, d) is d          # unknown profile
    assert made == []                                   # never contacted it


def test_scout_client_resolves_and_caches(monkeypatch):
    made = []

    def factory(c):
        obj = _ScoutBase([])
        obj.derived = c
        made.append(obj)
        return obj

    monkeypatch.setattr(llm_mod, "LLMClient", factory)
    cfg = _profile_cfg(scout_profile="scout")
    d = object()
    a = roles_mod.scout_client(cfg, d)
    b = roles_mod.scout_client(cfg, d)
    assert a is b and len(made) == 1
    assert a.derived.model == "tiny-fast"
    assert a.derived.base_url == "http://127.0.0.1:9/v1"


def test_scout_client_unreachable_falls_back_once(monkeypatch):
    class Dead(_ScoutBase):
        def check(self, warn=True):
            raise LLMError("no route to host")

    made = []

    def factory(c):
        made.append(c)
        return Dead([])

    monkeypatch.setattr(llm_mod, "LLMClient", factory)
    cfg = _profile_cfg(scout_profile="scout")
    d = object()
    assert roles_mod.scout_client(cfg, d) is d
    assert roles_mod.scout_client(cfg, d) is d          # verdict cached
    assert len(made) == 1


def test_gather_runs_on_scout_answer_tools_on_architect(tmp_path, monkeypatch):
    scout = _ScoutBase(READ_THEN_GREP)
    derived = []

    def factory(c):
        derived.append(c)
        return scout

    monkeypatch.setattr(llm_mod, "LLMClient", factory)
    arch = Scripted([])
    _root, idx, cfg = make_project(
        tmp_path, scout_profile="scout",
        profiles={"scout": {"model": "tiny-fast",
                            "base_url": "http://127.0.0.1:9/v1"}})
    agent = Agent(idx, arch, cfg)
    agent.gather("Fix bulk_discount", verbose=False)
    assert derived and derived[0].model == "tiny-fast"
    assert len(scout.seen) == 3                         # every step on scout
    assert arch.seen == []                              # architect: no steps
    # the scout's steps actually ran the tools against the shared agent state
    assert any("grep" in line for line in agent.steps_log)


def test_gather_falls_back_when_scout_is_dead(tmp_path, monkeypatch):
    class Dead(_ScoutBase):
        def check(self, warn=True):
            raise LLMError("connection refused")

    monkeypatch.setattr(llm_mod, "LLMClient", lambda c: Dead([]))
    arch = Scripted(READ_THEN_GREP)
    _root, idx, cfg = make_project(
        tmp_path, scout_profile="scout",
        profiles={"scout": {"model": "tiny-fast",
                            "base_url": "http://127.0.0.1:9/v1"}})
    agent = Agent(idx, arch, cfg)
    agent.gather("Fix bulk_discount", verbose=False)
    # soft fallback: the architect carried the whole loop
    assert len(arch.seen) == 3


def test_gather_without_scout_profile_is_unchanged(tmp_path):
    arch = Scripted(READ_THEN_GREP)
    _root, idx, cfg = make_project(tmp_path)
    agent = Agent(idx, arch, cfg)
    assert agent._gather_client() is arch
    agent.gather("Fix bulk_discount", verbose=False)
    assert len(arch.seen) == 3


# ------------------------------------------------------------------ config
def test_new_fields_round_trip_through_toml(tmp_path):
    text = S.set_values(config_mod.TEMPLATE,
                        {"scout_profile": "scout",
                         "enforce_json_schema": False})
    flat, _names = S.validate_text(text, str(tmp_path / "c.toml"))
    assert flat["scout_profile"] == "scout"
    assert flat["enforce_json_schema"] is False
    p = tmp_path / "c.toml"
    p.write_text(text, encoding="utf-8")
    cfg = config_mod.load(root=tmp_path, explicit=p, use_env=False,
                          require_hardware=False, warn_budgets=False)
    assert cfg.scout_profile == "scout"
    assert cfg.enforce_json_schema is False
    assert cfg.sources["scout_profile"] != "default"


def test_env_overrides_apply(tmp_path, monkeypatch):
    monkeypatch.setenv("PEDACITO_SCOUT_PROFILE", "scout")
    monkeypatch.setenv("PEDACITO_ENFORCE_JSON_SCHEMA", "false")
    cfg = config_mod.load(root=tmp_path, require_hardware=False,
                          warn_budgets=False)
    assert cfg.scout_profile == "scout"
    assert cfg.enforce_json_schema is False
    assert "env" in cfg.sources["scout_profile"]


# -------------------------------------------------------------------- GUI
def test_settings_schema_exposes_the_fields():
    items = {i["name"]: i for i in S.schema()}
    ej = items["enforce_json_schema"]
    assert ej["type"] == "bool" and ej["default"] is True
    sp = items["scout_profile"]
    assert sp["type"] == "str" and sp["default"] == ""
    # curated labels/descriptions reach the All settings page
    assert sp["label"] == S.FIELD_META["scout_profile"].label
    assert ej["label"] == S.FIELD_META["enforce_json_schema"].label


def test_status_reports_effective_scout_profile():
    from pedacito.gui import core
    st = core.State()
    cfg = _profile_cfg()
    st.cfg = cfg
    cfg.active_profile = "arch"
    cfg.scout_profile = "scout"
    assert st.status()["scout_profile"] == "scout"
    cfg.scout_profile = "arch"                            # == active: no-op
    assert st.status()["scout_profile"] == ""
    cfg.scout_profile = "ghost"                           # not a real profile
    assert st.status()["scout_profile"] == ""
    cfg.scout_profile = ""
    assert st.status()["scout_profile"] == ""


def test_settings_js_wires_the_scout_controls():
    js = (Path(__file__).resolve().parents[1] / "pedacito" / "gui" / "static"
          / "settings.js").read_text(encoding="utf-8")
    assert "scout_profile" in js and "enforce_json_schema" in js
    assert "None (use active profile)" in js
    app = (Path(__file__).resolve().parents[1] / "pedacito" / "gui" / "static"
           / "app.js").read_text(encoding="utf-8")
    assert "scoutChip" in app and "Scout: " in app
