"""A1.2: the reason-first gather step schema (`step_reasoning_first`).

Invariants this flag could break: the truncation salvage that assumes
`reasoning` is the last field (llm.repair_truncated_json / _cut_field), the
guards that refuse a cut or cap-closed payload, the byte-stability of the
gather prompt, replay digests, card-vs-user precedence and config writes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from a1_support import READ_THEN_GREP, Scripted, gather, make_project
from pedacito import agent as agent_mod
from pedacito import config as config_mod
from pedacito import modelcard, replay, tools
from pedacito.agent import Agent
from pedacito.config import Config, ConfigError
from pedacito.gui import settings as S
from pedacito.llm import LLMClient, LLMError, repair_truncated_json

OLD_ORDER = ["action", "file", "start", "end", "pattern", "symbol", "content",
             "reasoning"]
NEW_ORDER = ["reasoning", "action", "file", "start", "end", "pattern", "symbol",
             "content"]


def _rf_cfg(**kw):
    cfg = Config()
    cfg.step_reasoning_first = True
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


# ------------------------------------------------------------- off by default
def test_default_schema_is_unchanged():
    assert Config().step_reasoning_first is None
    for cfg in (Config(), _rf_cfg(step_reasoning_first=False),
                _rf_cfg(step_reasoning_first=None)):
        s = tools.schema_for(cfg)
        assert list(s["properties"]) == OLD_ORDER
        assert s["required"] == OLD_ORDER
        assert s["properties"]["reasoning"]["description"] == \
            "Under 15 words: what you still need. Not a plan."
    assert tools.STEP_SCHEMA["properties"]["reasoning"]["description"] == \
        "Under 15 words: what you still need. Not a plan."


def test_default_gather_prompt_and_schema_are_the_old_ones(tmp_path):
    _a, client = gather(tmp_path, READ_THEN_GREP)
    assert all(list(schema["properties"]) == OLD_ORDER
               for _msgs, schema in client.seen)


# ------------------------------------------------------------------- new path
def test_reason_first_schema_moves_reasoning_to_the_front():
    s = tools.schema_for(_rf_cfg())
    assert list(s["properties"]) == NEW_ORDER
    assert s["required"] == NEW_ORDER
    r = s["properties"]["reasoning"]
    assert r["description"] == "Under 20 words: what you need next and why this action."
    assert r["maxLength"] == 160
    # the shared template is never mutated
    assert tools.STEP_SCHEMA["properties"]["reasoning"]["maxLength"] == 160
    assert list(tools.STEP_SCHEMA["properties"])[-1] == "reasoning"
    # the enum is narrowed exactly as before
    assert s["properties"]["action"]["enum"] == \
        tools.schema_for(Config())["properties"]["action"]["enum"]


def test_reason_first_composes_with_the_compact_schema_and_write_actions():
    cfg = _rf_cfg(compact_step_schema=True, allow_write=True, allow_shell=True)
    s = tools.schema_for(cfg)
    assert next(iter(s["properties"])) == "reasoning"
    assert s["required"] == ["reasoning", "action"]
    assert list(s["properties"])[-1] == "content"


def test_worked_examples_follow_the_schema_order():
    from pedacito.llm import schema_example_with
    ex = json.loads(schema_example_with(tools.schema_for(_rf_cfg()),
                                        action="grep", reasoning="why"))
    assert list(ex)[0] == "reasoning" and ex["action"] == "grep"


# ------------------------------------------------------------ truncation (I5)
class _Fake(LLMClient):
    """A client that replays (text, finish_reason) pairs through the real
    chat_json salvage logic."""

    def __init__(self, replies, cfg=None):
        LLMClient.__init__(self, cfg or Config())
        self.replies = list(replies)
        self.n = 0
        self.asked: list[list[dict]] = []

    def chat(self, messages, max_tokens, schema=None, stream=False, **kw):
        self.asked.append(list(messages))
        text, finish = self.replies[min(self.n, len(self.replies) - 1)]
        self.n += 1
        self.last_finish_reason = finish
        return text


RF = tools.schema_for(_rf_cfg(allow_browser=True, allow_shell=True,
                              allow_write=True))
MSGS = [{"role": "user", "content": "t"}]
TAIL = '"start":0,"end":0,"pattern":"","symbol":"","to":"","command":"","content":""'


def test_cut_mid_reasoning_leaves_no_action_and_reasks():
    cut = '{"reasoning":"I need to see how the discount threshold is compu'
    assert json.loads(repair_truncated_json(cut)) == {
        "reasoning": "I need to see how the discount threshold is compu"}
    good = ('{"reasoning":"read it","action":"read","file":"a.py",' + TAIL + "}")
    c = _Fake([(cut, "length"), (good, "stop")])
    obj = c.chat_json(MSGS, RF, max_tokens=700)
    assert obj["action"] == "read" and c.n == 2
    assert c.step_truncated is False and c.truncated_field == ""
    nudge = c.asked[1][-1]["content"]
    assert "cut off" in nudge and "'reasoning' kept under 10 words" in nudge
    assert c.asked[1][-2] == {"role": "assistant", "content": cut}


def test_cut_mid_reasoning_every_time_raises_instead_of_acting():
    cut = '{"reasoning":"Let me think about every file in the pro'
    c = _Fake([(cut, "length")])
    with pytest.raises(LLMError, match="token limit"):
        c.chat_json(MSGS, RF, max_tokens=700)


def test_cut_right_after_reasoning_has_no_action_either():
    cut = '{"reasoning":"read the file",'
    c = _Fake([(cut, "length")])
    with pytest.raises(LLMError, match="token limit"):
        c.chat_json(MSGS, RF, max_tokens=700)


def test_cut_mid_content_names_the_content_field():
    cut = ('{"reasoning":"write it","action":"write","file":"a.py","start":0,'
           '"end":0,"pattern":"","symbol":"","to":"","command":"",'
           '"content":"def f():\\n    return 1\\ndef g(')
    c = _Fake([(cut, "length")])
    obj = c.chat_json(MSGS, RF, max_tokens=700)
    assert obj["action"] == "write" and obj["content"].endswith("def g(")
    assert c.step_truncated is True and c.truncated_field == "content"


def test_cut_mid_pattern_names_the_pattern_field():
    cut = ('{"reasoning":"measure","action":"browser","file":"","start":2,'
           '"end":0,"pattern":"press Enter; eval new Promise(res => { const r = img[')
    c = _Fake([(cut, "length")])
    obj = c.chat_json(MSGS, RF, max_tokens=700)
    assert obj["action"] == "browser"
    assert c.step_truncated is True and c.truncated_field == "pattern"


def test_cut_inside_a_half_typed_key_after_the_action_is_not_benign():
    """Reason-first: the cut can land between arguments, where no `"key":` was
    opened yet. The first required argument missing from the salvage is named,
    so gather refuses to run `grep` with an empty pattern."""
    cut = '{"reasoning":"find it","action":"grep","file":"","start":0,"end":0,"pat'
    c = _Fake([(cut, "length")])
    obj = c.chat_json(MSGS, RF, max_tokens=700)
    assert obj["action"] == "grep"
    assert c.step_truncated is True and c.truncated_field == "pattern"


def test_cut_inside_the_action_value_names_the_action():
    c = _Fake([('{"reasoning":"go","action":"gre', "length")])
    obj = c.chat_json(MSGS, RF, max_tokens=700)
    assert c.truncated_field == "action" and obj["action"] == "gre"


def test_cut_after_every_field_but_the_closing_brace_is_benign():
    cut = ('{"reasoning":"done","action":"done","file":"",' + TAIL)
    c = _Fake([(cut, "length")])
    obj = c.chat_json(MSGS, RF, max_tokens=700)
    assert obj["action"] == "done" and c.truncated_field == ""


def test_a_complete_reason_first_reply_flags_nothing():
    good = ('{"reasoning":"ok","action":"done","file":"",' + TAIL + "}")
    c = _Fake([(good, "stop")])
    assert c.chat_json(MSGS, RF, max_tokens=700)["action"] == "done"
    assert c.step_truncated is False and c.truncated_field == ""


def test_default_order_cut_behaviour_is_unchanged():
    """The default schema still treats a cut inside the trailing `reasoning`
    as benign (the guard for the flag-off path)."""
    default = tools.schema_for(Config())
    cut = ('{"action":"read","file":"a.py","start":1,"end":9,"pattern":"",'
           '"symbol":"","content":"","reasoning":"See the im')
    c = _Fake([(cut, "length")])
    obj = c.chat_json(MSGS, default, max_tokens=700)
    assert obj["action"] == "read" and c.truncated_field == "reasoning"


# ----------------------------------------- gather guards, with real salvage
def _gather_with_replies(tmp_path, monkeypatch, replies, **cfg_attrs):
    """Real chat_json salvage feeding the real gather loop; returns the list
    of dispatched actions and the agent."""
    _root, idx, cfg = make_project(tmp_path, step_reasoning_first=True,
                                   **cfg_attrs)
    dispatched = []
    monkeypatch.setattr(agent_mod, "dispatch",
                        lambda index, cfg_, step, client=None:
                        dispatched.append(step.get("action")) or "ok")
    client = _Fake(replies, cfg)
    agent = Agent(idx, client, cfg)
    agent.gather("Fix bulk_discount", verbose=False)
    return dispatched, agent


def test_gather_refuses_a_payload_cut_at_the_token_limit(tmp_path, monkeypatch):
    cut = ('{"reasoning":"measure","action":"browser","file":"","start":2,'
           '"end":0,"pattern":"eval new Promise(res => { const r = img[')
    dispatched, agent = _gather_with_replies(
        tmp_path, monkeypatch, [(cut, "length")], allow_browser=True)
    assert dispatched == []
    discarded = [s for s in agent.steps_log if "(discarded" in s]
    assert len(discarded) == agent_mod._MAX_PAYLOAD_CUTS


def test_gather_refuses_a_step_cut_between_arguments(tmp_path, monkeypatch):
    cut = '{"reasoning":"find it","action":"grep","file":"","start":0,"end":0,"pat'
    dispatched, agent = _gather_with_replies(
        tmp_path, monkeypatch, [(cut, "length")])
    assert dispatched == []
    assert any("`pattern` payload cut off" in s for s in agent.steps_log)


def test_gather_refuses_a_string_closed_at_its_cap(tmp_path, monkeypatch):
    cap = tools.schema_for(_rf_cfg(allow_browser=True))["properties"]["pattern"]["maxLength"]
    reply = json.dumps({"reasoning": "measure", "action": "browser", "file": "",
                        "start": 0, "end": 0, "pattern": "x" * cap, "symbol": "",
                        "content": "", "command": "python main.py"})
    dispatched, agent = _gather_with_replies(
        tmp_path, monkeypatch, [(reply, "stop")], allow_browser=True)
    assert dispatched == []
    assert len([s for s in agent.steps_log if "(discarded" in s]) \
        == agent_mod._MAX_PAYLOAD_CUTS


def test_reasoning_closed_at_its_cap_is_not_refused(tmp_path, monkeypatch):
    """`reasoning` is the one field a cap-closed string cannot hurt."""
    cap = RF["properties"]["reasoning"]["maxLength"]
    reply = json.dumps({"reasoning": "r" * cap, "action": "grep", "file": "",
                        "start": 0, "end": 0, "pattern": "bulk", "symbol": "",
                        "content": ""})
    dispatched, agent = _gather_with_replies(
        tmp_path, monkeypatch, [(reply, "stop"),
                                ('{"reasoning":"x","action":"done","file":"",'
                                 '"start":0,"end":0,"pattern":"","symbol":"",'
                                 '"content":""}', "stop")])
    assert dispatched == ["grep"]
    assert not any("(discarded" in s for s in agent.steps_log)


def test_gather_recovers_from_a_reason_first_step_that_never_finished(
        tmp_path, monkeypatch):
    """Cut mid-reasoning on every attempt: gather re-asks with its own
    recovery prompt, then moves on; nothing is dispatched blind."""
    cut = '{"reasoning":"Let me think about every file in the proj'
    dispatched, agent = _gather_with_replies(
        tmp_path, monkeypatch, [(cut, "length")])
    assert dispatched == []


# ------------------------------------------------- byte stability (I1), replay (I3)
def test_gather_prompt_is_byte_identical_across_steps_and_flag_states(tmp_path):
    runs = {}
    for flag in (False, True):
        sub = tmp_path / str(flag)
        sub.mkdir()
        _a, client = gather(sub, READ_THEN_GREP,
                            {"step_reasoning_first": flag})
        heads = {tuple(msgs[:2]) for msgs, _s in client.seen}
        assert len(client.seen) == 3 and len(heads) == 1
        runs[flag] = heads.pop()
        first_prop = next(iter(client.seen[0][1]["properties"]))
        assert first_prop == ("reasoning" if flag else "action")
    assert runs[False] == runs[True]            # the flag only moves the schema


def test_reason_first_does_not_change_the_replay_digest(tmp_path):
    """Recording with the flag on replays cleanly, and the digest (which hashes
    the messages, not the schema) is the same with the flag off."""
    digests = {}
    for flag in (False, True):
        sub = tmp_path / str(flag)
        sub.mkdir()
        _root, idx, cfg = make_project(sub, step_reasoning_first=flag)
        rec_path = sub / "rec.jsonl"
        rec = replay.Recorder(Scripted(READ_THEN_GREP), rec_path)
        Agent(idx, rec, cfg).gather("Fix bulk_discount", verbose=False)
        lines = [json.loads(ln) for ln in rec_path.read_text().splitlines()]
        digests[flag] = [r["digest"] for r in lines if r["kind"] == "chat_json"]
        rep = replay.Replayer(rec_path)
        Agent(idx, rep, cfg).gather("Fix bulk_discount", verbose=False)
        assert rep.diverged == [] and len(rep.used) == 3
    assert digests[False] == digests[True]


# ------------------------------------------------------- calibration (I7, I2)
class _OrderSensitive:
    """Picks the right action only when `reasoning` comes first. Records the
    first schema property of every call."""

    chars_per_token = 3.2

    def __init__(self, rf_good=True):
        self.rf_good = rf_good
        self.first_props: list[str] = []

    def chat_json(self, messages, schema, max_tokens, **kw):
        first = next(iter(schema["properties"]))
        self.first_props.append(first)
        text = messages[1]["content"]
        done = "RESULT:" in messages[-1]["content"]
        if first != "reasoning" or not self.rf_good:
            return {"reasoning": "r", "action": "outline", "file": "x"}
        pick = ("done" if done else "callers" if "calls the retry" in text
                else "grep" if "MAX_RETRIES" in text
                else "outline" if "List every function" in text
                else "find" if "authentication" in text else "read")
        return {"reasoning": "r", "action": pick, "file": "src/scheduler.py",
                "start": 82, "end": 90, "pattern": "MAX_RETRIES",
                "symbol": "retry"}

    def chat(self, messages, max_tokens, **kw):
        return "ok"


def test_calibration_adopts_reason_first_only_when_it_measures_better():
    good = _OrderSensitive(rf_good=True)
    cfg = Config()
    card = modelcard.calibrate(good, cfg, verbose=False, quick=True)
    m = card["measured"]
    assert m["tool_choice_rf"] == 1.0 and m["json_ok_rf"] == 1.0
    assert m["tool_choice"] < 0.5                 # the baseline arm: action first
    assert card["choices"]["step_reasoning_first"] is True
    # the plain arm (and the few-shot arm, which keeps the default order) are
    # action-first, the reason-first arm reasoning-first
    assert good.first_props.count("action") == 20
    assert good.first_props.count("reasoning") == 10

    flat = _OrderSensitive(rf_good=False)
    card2 = modelcard.calibrate(flat, Config(), verbose=False, quick=True)
    assert "step_reasoning_first" not in card2["choices"]


def test_the_baseline_arm_ignores_a_flag_already_set():
    """A card must be about the model: a user flag cannot turn the baseline
    arm into the reason-first arm."""
    cfg = Config()
    cfg.step_reasoning_first = True
    probe = _OrderSensitive()
    modelcard._t_tool_choice(probe, cfg)
    assert set(probe.first_props) == {"action"}
    modelcard._t_tool_choice(probe, cfg, True)
    assert probe.first_props[-1] == "reasoning"


BASE = {"tool_choice": 0.6, "json_ok": 1.0, "edit_format": 1, "instructions": 1,
        "honesty": 1, "needle": 1, "continuation": 1}


def test_choose_applies_one_rule_to_native_and_reason_first():
    def pick(arm, score, js=1.0):
        m = {**BASE, f"tool_choice_{arm}": score, f"json_ok_{arm}": js}
        ch = modelcard.choose(m, False)
        return ch["tool_mode"] == "native" if arm == "native" \
            else ch.get("step_reasoning_first") is True

    for arm in ("native", "rf"):
        assert pick(arm, 0.9)                     # a clear gain
        assert pick(arm, 0.7)                     # exactly +0.1 counts
        assert not pick(arm, 0.65)                # less than +0.1
        assert not pick(arm, 0.9, js=0.8)         # lost well-formed steps
    assert "step_reasoning_first" not in modelcard.choose(dict(BASE), False)


def test_the_gain_threshold_survives_float_rounding():
    """0.7 + 0.1 is 0.7999999999999999, which used to reject a 0.8 arm."""
    m = {**BASE, "tool_choice": 0.7, "tool_choice_rf": 0.8, "json_ok_rf": 1.0}
    assert modelcard.choose(m, False)["step_reasoning_first"] is True
    m = {**BASE, "tool_choice": 0.7, "tool_choice_native": 0.8,
         "json_ok_native": 1.0}
    assert modelcard.choose(m, False)["tool_mode"] == "native"


def test_card_fills_reason_first_only_when_the_user_left_it_alone(tmp_path,
                                                                  monkeypatch):
    monkeypatch.setattr(config_mod, "USER_CONFIG", tmp_path / "home" / "c.toml")
    (tmp_path / "home").mkdir()
    cfg = Config()
    cfg.sources = {}
    card = modelcard.calibrate(_OrderSensitive(), cfg, verbose=False, quick=True)
    assert card["choices"]["step_reasoning_first"] is True
    modelcard.save_card(card)

    fresh = Config()
    fresh.sources = {"step_reasoning_first": "default"}
    modelcard.apply(fresh)
    assert fresh.step_reasoning_first is True
    assert fresh.sources["step_reasoning_first"] == "model card"
    assert next(iter(tools.schema_for(fresh)["properties"])) == "reasoning"

    for user_value in (False, True):
        explicit = Config()
        explicit.sources = {"step_reasoning_first": "config file"}
        explicit.step_reasoning_first = user_value
        modelcard.apply(explicit)
        assert explicit.step_reasoning_first is user_value
        assert explicit.sources["step_reasoning_first"] == "config file"


# --------------------------------------------------------------- config (I4)
def test_step_reasoning_first_is_validated_and_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "USER_CONFIG", tmp_path / "none.toml")
    S.validate_text("step_reasoning_first = true\n")
    S.validate_text("[profiles.x]\nmodel = 'm'\nstep_reasoning_first = false\n")
    for bad in ('"yes"', "1", "0.5"):
        with pytest.raises(ConfigError, match="true or false"):
            S.validate_text(f"step_reasoning_first = {bad}\n")
    proj = tmp_path / "p"
    proj.mkdir()
    (tmp_path / "cfg.toml").write_text(
        "vram_gb = 8\nstep_reasoning_first = true\n", encoding="utf-8")
    cfg = config_mod.load(root=proj, explicit=tmp_path / "cfg.toml",
                          use_env=False, require_hardware=False,
                          warn_budgets=False)
    assert cfg.step_reasoning_first is True
    assert cfg.sources["step_reasoning_first"].startswith("--config")


def test_step_reasoning_first_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "USER_CONFIG", tmp_path / "none.toml")
    proj = tmp_path / "p"
    proj.mkdir()
    for raw, want in (("1", True), ("true", True), ("0", False),
                      ("off", False), ("auto", None), ("", None)):
        monkeypatch.setenv("PEDACITO_STEP_REASONING_FIRST", raw)
        cfg = config_mod.load(root=proj, require_hardware=False,
                              warn_budgets=False)
        assert cfg.step_reasoning_first is want, raw


def test_config_writer_keeps_comments_and_other_keys():
    text = config_mod.TEMPLATE
    out = S.set_values(text, {"step_reasoning_first": True})
    S.validate_text(out)
    assert "step_reasoning_first = true" in out
    # every original line survives, in order
    kept = [ln for ln in out.splitlines() if ln != "step_reasoning_first = true"]
    assert kept == text.splitlines()
    again = S.set_values(out, {"step_reasoning_first": False})
    live = [ln for ln in again.splitlines()
            if ln.startswith("step_reasoning_first")]
    assert live == ["step_reasoning_first = false"]
    assert "# step_reasoning_first = true   # unset: the model card decides" \
        in again                                # the commented example stays
    prof = S.set_values(text, {"step_reasoning_first": True}, profile="default")
    assert "step_reasoning_first = true" in prof.split("\n[profiles.default]\n")[1]


def test_settings_schema_lists_the_flag_with_its_default():
    item = next(i for i in S.schema() if i["name"] == "step_reasoning_first")
    assert item["type"] == "bool" and item["default"] is None
    assert "reasoning" in item["help"].lower()
