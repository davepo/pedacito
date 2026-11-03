"""A1.5: project few-shot step examples in the first user turn
(`fewshot_steps`), shown under grammar and native mode too.

Invariants: off means byte-identical user0; with the flag on the examples sit
once in user0 (never appended to later turns) and user0 stays byte-identical
across steps and runs; the card fills the flag only when the user left it
alone; replay still matches."""
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
from pedacito.llm import LLMClient, estimate_tokens
from test_llm_transport import _cfg, _ok, _patch_post

HEADER = agent_mod.FEWSHOT_HEADER


def _user0(client) -> str:
    return client.seen[0][0][1]


def _example_lines(user0: str) -> list[str]:
    text = user0.split(HEADER, 1)[1].split("\n\nTASK\n", 1)[0]
    return text.split("\n")[1:]


# ------------------------------------------------------------- off by default
def test_off_by_default_user0_has_no_examples(tmp_path):
    assert Config().fewshot_steps is None
    for flag in (None, False):
        sub = tmp_path / str(flag)
        sub.mkdir()
        _a, client = gather(sub, READ_THEN_GREP, {"fewshot_steps": flag})
        user0 = _user0(client)
        assert HEADER not in user0
        assert "\n\nTASK\nFix bulk_discount" in user0


def test_off_user0_equals_the_old_assembly(tmp_path):
    agent, client = gather(tmp_path, READ_THEN_GREP)
    repo_map, _tier = agent._map("Fix bulk_discount")
    lead = agent._lead_notes()
    want = (lead + "\n\n" if lead else "") \
        + f"PROJECT MAP\n\n{repo_map}\n\nTASK\nFix bulk_discount"
    assert _user0(client).startswith(want)          # + the step-1 nudge


# ------------------------------------------------------------------- new path
def test_three_examples_sit_between_the_map_and_the_task(tmp_path):
    agent, client = gather(tmp_path, READ_THEN_GREP, {"fewshot_steps": True})
    user0 = _user0(client)
    assert user0.count(HEADER) == 1
    assert user0.index("PROJECT MAP") < user0.index(HEADER) < user0.index("\nTASK\n")
    lines = _example_lines(user0)
    assert len(lines) == 3
    shots = [json.loads(ln) for ln in lines]
    assert [s["action"] for s in shots] == ["grep", "read", "done"]
    # a real definition of this project, with a range inside the file
    assert shots[0]["pattern"] == "add_tax"
    assert shots[1]["file"] == "pricing.py"
    n_lines = len(Path(agent.index.root, "pricing.py").read_text().splitlines())
    assert 1 <= shots[1]["start"] <= shots[1]["end"] <= n_lines
    # unused fields are left out; reasoning is present in each
    assert all("reasoning" in s and "content" not in s for s in shots)


def test_examples_are_about_a_hundred_tokens(tmp_path):
    (tmp_path / "on").mkdir()
    (tmp_path / "off").mkdir()
    _a, on = gather(tmp_path / "on", READ_THEN_GREP, {"fewshot_steps": True})
    _a, off = gather(tmp_path / "off", READ_THEN_GREP)
    extra = estimate_tokens(_user0(on)) - estimate_tokens(_user0(off))
    assert 40 <= extra <= 90


def test_the_read_example_is_short_but_the_fallback_keeps_its_span(tmp_path):
    """A copied read example sits in every later call, so user0's is short;
    the no-grammar fallback (flag off) keeps its 40 lines."""
    root = tmp_path / "big"
    root.mkdir()
    body = "".join(f"    x{i} = {i}\n" for i in range(80))
    (root / "m.py").write_text("def long_one():\n" + body, encoding="utf-8")
    from pedacito import index as index_mod
    cfg = Config()
    cfg.log_events = False
    cfg.sources = {}
    idx = index_mod.build([str(root)], cfg, client=None, summarise=False,
                          root=root, verbose=False)
    agent = Agent(idx, Scripted(READ_THEN_GREP), cfg)
    short = agent._fewshot_shots(agent_mod.FEWSHOT_USER0_READ_LINES)[1]
    assert short["end"] - short["start"] == agent_mod.FEWSHOT_USER0_READ_LINES
    wide = agent._fewshot_shots()[1]
    assert wide["end"] - wide["start"] == 40
    client = LLMClient(_cfg())
    off = Agent(idx, client, cfg)
    off._prime_fewshot()                          # flag unset: the fallback's
    assert client.fewshot[1]["end"] - client.fewshot[1]["start"] == 40
    cfg.fewshot_steps = True
    off._prime_fewshot()                          # flag on: user0 has them
    assert client.fewshot == []


@pytest.mark.parametrize("over", [
    {}, {"tool_mode": "native"}, {"json_mode": "object"}, {"schema_off": True}])
def test_every_request_carries_the_examples_once_in_the_first_user_turn(
        tmp_path, monkeypatch, over):
    """The real client, in every transport mode: each lookup request holds the
    block exactly once, in message 1 and nowhere else, byte-identical from
    step to step (prompt tokens grow by that block per call, never more)."""
    over = dict(over)
    schema_off = over.pop("schema_off", False)
    bodies = []

    def post(url, **k):
        p = k["json"]
        bodies.append(p)
        step = {"action": "grep", "pattern": "bulk_discount",
                "reasoning": "r"} if len(bodies) == 1 else {
            "action": "done", "reasoning": "r"}
        if "tools" in p:
            msg = {"content": None, "tool_calls": [
                {"function": {"arguments": json.dumps(step)}}]}
            return type(_ok())(200, {"choices": [{"message": msg,
                                                  "finish_reason": "stop"}],
                                     "usage": {}})
        return _ok(json.dumps(step))

    _patch_post(monkeypatch, post)
    _root, idx, cfg = make_project(tmp_path, fewshot_steps=True, **over)
    client = LLMClient(cfg)
    client._schema_supported = not schema_off
    Agent(idx, client, cfg).gather("Fix bulk_discount", verbose=False)
    assert len(bodies) >= 2
    first = [b["messages"][1]["content"] for b in bodies]
    for b in bodies:
        texts = [m["content"] for m in b["messages"]]
        assert sum(t.count(HEADER) for t in texts) == 1
        assert HEADER in texts[1]
    # the stored first user turn does not change as the gather goes on
    assert len({f.split("\n\nTASK\n")[0] for f in first}) == 1


def test_examples_follow_the_step_schema_order(tmp_path):
    _a, client = gather(tmp_path, READ_THEN_GREP,
                        {"fewshot_steps": True, "step_reasoning_first": True})
    lines = _example_lines(_user0(client))
    assert all(list(json.loads(ln))[0] == "reasoning" for ln in lines)
    shot = json.loads(lines[1])
    assert list(shot) == ["reasoning", "action", "file", "start", "end"]


def test_user0_is_byte_identical_across_steps_and_runs(tmp_path):
    seen = []
    for i in range(2):
        sub = tmp_path / str(i)
        sub.mkdir()
        _a, client = gather(sub, READ_THEN_GREP, {"fewshot_steps": True})
        heads = {tuple(msgs[:2]) for msgs, _s in client.seen}
        assert len(client.seen) == 3 and len(heads) == 1      # across steps
        seen.append(heads.pop())
        # the examples are never appended to a later turn
        for msgs, _s in client.seen:
            assert sum(m.count(HEADER) for m in msgs) == 1
            assert HEADER not in msgs[-1] or len(msgs) == 2
    # across runs of the same project: the system prompt and the examples are
    # the prefix a server can keep cached
    assert seen[0][0] == seen[1][0]
    assert _example_lines(seen[0][1]) == _example_lines(seen[1][1])


def test_the_system_prompt_is_unchanged_by_the_flag(tmp_path):
    systems = []
    for flag in (False, True):
        sub = tmp_path / str(flag)
        sub.mkdir()
        _a, client = gather(sub, READ_THEN_GREP, {"fewshot_steps": flag})
        systems.append(client.seen[0][0][0])
    assert systems[0] == systems[1]


def test_a_project_without_code_gets_no_block(tmp_path):
    from pedacito import index as index_mod
    root = tmp_path / "docs"
    root.mkdir()
    (root / "notes.txt").write_text("hello\n")
    cfg = Config()
    cfg.log_events = False
    cfg.log_sessions = False
    cfg.fewshot_steps = True
    cfg.sources = {}
    idx = index_mod.build([str(root)], cfg, client=None, summarise=False,
                          root=root, verbose=False)
    client = Scripted(READ_THEN_GREP)
    agent = Agent(idx, client, cfg)
    assert agent._fewshot_shots() == []
    agent.gather("what is in notes", verbose=False)
    assert HEADER not in client.seen[0][0][1]
    assert "\n\nTASK\nwhat is in notes" in client.seen[0][0][1]


# ----------------------------------- the no-grammar fallback does not repeat it
def test_the_fallback_path_is_not_shown_the_examples_twice(tmp_path, monkeypatch):
    posts = []
    _patch_post(monkeypatch, lambda url, **k: posts.append(k["json"])
                or _ok('{"action": "done", "reasoning": "r"}'))
    for flag in (False, True):
        _root, idx, cfg = make_project(tmp_path / str(flag), fewshot_steps=flag)
        client = LLMClient(_cfg(fewshot_steps=flag))
        client._schema_supported = False
        client.cfg.log_events = False
        agent = Agent(idx, client, cfg)
        agent._prime_fewshot()
        assert bool(client.fewshot) is (not flag)
        client.chat_json([{"role": "user", "content": "go"}],
                         tools.schema_for(cfg), 100)
        last = posts[-1]["messages"][-1]["content"]
        assert ("Worked examples (for THIS project" in last) is (not flag)
        assert "Example of a well-formed reply" in last or not flag


def test_native_mode_still_sees_them(tmp_path):
    agent, client = gather(tmp_path, READ_THEN_GREP,
                           {"fewshot_steps": True, "tool_mode": "native"})
    assert HEADER in _user0(client)


# ------------------------------------------------------------------- replay
def test_replay_with_the_examples_on_and_the_divergence_when_off(tmp_path):
    path = tmp_path / "rec.jsonl"
    _root, idx, cfg = make_project(tmp_path / "on", fewshot_steps=True)
    rec = replay.Recorder(Scripted(READ_THEN_GREP), path)
    Agent(idx, rec, cfg).gather("Fix bulk_discount", verbose=False)
    rep = replay.Replayer(path)
    Agent(idx, rep, cfg).gather("Fix bulk_discount", verbose=False)
    assert rep.diverged == [] and len(rep.used) == 3
    # a recording made without the examples reports the changed prompt
    path2 = tmp_path / "rec2.jsonl"
    _r, idx2, cfg2 = make_project(tmp_path / "off", fewshot_steps=False)
    rec2 = replay.Recorder(Scripted(READ_THEN_GREP), path2)
    Agent(idx2, rec2, cfg2).gather("Fix bulk_discount", verbose=False)
    rep2 = replay.Replayer(path2)
    Agent(idx, rep2, cfg).gather("Fix bulk_discount", verbose=False)
    assert rep2.diverged


# ----------------------------------------------------------- calibration arm
class _ExampleSensitive:
    """Picks the right action only when the first user turn carries worked
    examples; records whether each call saw them."""

    chars_per_token = 3.2

    def __init__(self, learns=True):
        self.learns = learns
        self.saw: list[bool] = []

    def chat_json(self, messages, schema, max_tokens, **kw):
        text = messages[1]["content"]
        shown = HEADER in text
        self.saw.append(shown)
        done = "RESULT:" in messages[-1]["content"]
        if not (shown and self.learns):
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


def test_calibration_adopts_the_examples_only_when_they_measure_better():
    probe = _ExampleSensitive(learns=True)
    card = modelcard.calibrate(probe, Config(), verbose=False, quick=True)
    m = card["measured"]
    assert m["tool_choice_fs"] == 1.0 and m["json_ok_fs"] == 1.0
    assert m["tool_choice"] < 0.5 and m["tool_choice_rf"] < 0.5
    assert card["choices"]["fewshot_steps"] is True
    assert "step_reasoning_first" not in card["choices"]      # independent arm
    # the arms: plain, reason-first, examples -- ten scenarios each
    assert probe.saw.count(True) == 10 and probe.saw.count(False) == 20

    flat = _ExampleSensitive(learns=False)
    card2 = modelcard.calibrate(flat, Config(), verbose=False, quick=True)
    assert "fewshot_steps" not in card2["choices"]


def test_the_baseline_arm_never_shows_examples_even_if_the_flag_is_set():
    cfg = Config()
    cfg.fewshot_steps = True
    probe = _ExampleSensitive()
    modelcard._t_tool_choice(probe, cfg)
    assert set(probe.saw) == {False}
    modelcard._t_tool_choice(probe, cfg, False, True)
    assert probe.saw[-1] is True


BASE = {"tool_choice": 0.6, "json_ok": 1.0, "edit_format": 1, "instructions": 1,
        "honesty": 1, "needle": 1, "continuation": 1}


def test_choose_applies_the_native_rule_to_the_examples():
    def adopted(score, js=1.0):
        m = {**BASE, "tool_choice_fs": score, "json_ok_fs": js}
        return modelcard.choose(m, False).get("fewshot_steps") is True

    assert adopted(0.9) and adopted(0.7)             # a gain of 0.1 or more
    assert not adopted(0.65)                         # less than 0.1
    assert not adopted(0.9, js=0.8)                  # lost well-formed steps
    m = {**BASE, "tool_choice": 0.7, "tool_choice_fs": 0.8, "json_ok_fs": 1.0}
    assert modelcard.choose(m, False)["fewshot_steps"] is True   # float edge
    assert "fewshot_steps" not in modelcard.choose(dict(BASE), False)


def test_card_fills_the_examples_only_when_the_user_left_them_alone(
        tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "USER_CONFIG", tmp_path / "home" / "c.toml")
    (tmp_path / "home").mkdir()
    cfg = Config()
    cfg.sources = {}
    card = modelcard.calibrate(_ExampleSensitive(), cfg, verbose=False,
                               quick=True)
    modelcard.save_card(card)

    fresh = Config()
    fresh.sources = {"fewshot_steps": "default"}
    modelcard.apply(fresh)
    assert fresh.fewshot_steps is True
    assert fresh.sources["fewshot_steps"] == "model card"
    assert agent_mod.fewshot_enabled(fresh)

    for user_value in (False, True):
        explicit = Config()
        explicit.sources = {"fewshot_steps": "profile 'x' (explicit)"}
        explicit.fewshot_steps = user_value
        modelcard.apply(explicit)
        assert explicit.fewshot_steps is user_value
    # and an end-to-end gather follows whichever won
    on = Config()
    on.sources = {}
    modelcard.apply(on)
    _r, idx, _c = make_project(tmp_path / "e2e")
    client = Scripted(READ_THEN_GREP)
    Agent(idx, client, on).gather("Fix bulk_discount", verbose=False)
    assert HEADER in client.seen[0][0][1]


# --------------------------------------------------------------- config (I4)
def test_fewshot_steps_is_validated_loaded_and_written(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "USER_CONFIG", tmp_path / "none.toml")
    S.validate_text("fewshot_steps = true\n")
    for bad in ('"yes"', "1"):
        with pytest.raises(ConfigError, match="true or false"):
            S.validate_text(f"fewshot_steps = {bad}\n")
    proj = tmp_path / "p"
    proj.mkdir()
    monkeypatch.setenv("PEDACITO_FEWSHOT_STEPS", "on")
    cfg = config_mod.load(root=proj, require_hardware=False, warn_budgets=False)
    assert cfg.fewshot_steps is True
    assert cfg.sources["fewshot_steps"] == "env PEDACITO_FEWSHOT_STEPS"
    monkeypatch.setenv("PEDACITO_FEWSHOT_STEPS", "auto")
    assert config_mod.load(root=proj, require_hardware=False,
                           warn_budgets=False).fewshot_steps is None
    text = config_mod.TEMPLATE
    out = S.set_values(text, {"fewshot_steps": True})
    S.validate_text(out)
    live = [ln for ln in out.splitlines() if ln.startswith("fewshot_steps")]
    assert live == ["fewshot_steps = true"]
    assert [ln for ln in out.splitlines() if ln not in live] == text.splitlines()
    item = next(i for i in S.schema() if i["name"] == "fewshot_steps")
    assert item["type"] == "bool" and item["default"] is None
