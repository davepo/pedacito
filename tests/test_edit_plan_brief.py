"""A1.3: the edit-plan brief (`edit_plan_brief`).

Invariants: off means byte-identical briefs; a prompt-pack override bypasses
the flag; the gather prompt is untouched; replay still matches; and the no-op
and promise detectors do not misread a plan line."""
from __future__ import annotations

import difflib
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from a1_support import READ_THEN_GREP, Scripted, gather, make_project
from pedacito import config as config_mod
from pedacito import evalrun, loop, prompts, replay
from pedacito.agent import EDIT_FORMAT, Agent
from pedacito.config import Config
from pedacito.gui import settings as S

PLAN = ("I'll change `bulk_discount` in pricing.py, lines 4-7, because the "
        "threshold is off by one.")
BLOCK = ("--- FILE: pricing.py\n<<<<<<< SEARCH\n    if quantity > 10:\n=======\n"
         "    if quantity >= 10:\n>>>>>>> REPLACE\n")
BLOCK_NO_DIGITS = ("--- FILE: pricing.py\n<<<<<<< SEARCH\n    return old_name\n"
                   "=======\n    return new_name\n>>>>>>> REPLACE\n")


@pytest.fixture(autouse=True)
def _no_user_prompts(tmp_path, monkeypatch):
    """Keep a developer's real ~/.config/pedacito/prompts out of the tests."""
    monkeypatch.setattr(config_mod, "USER_CONFIG", tmp_path / "home" / "c.toml")


def _cfg(**kw):
    cfg = Config()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _changed_lines(old: str, new: str) -> tuple[list[str], list[str]]:
    gone, added = [], []
    for ln in difflib.ndiff(old.splitlines(), new.splitlines()):
        if ln.startswith("- "):
            gone.append(ln[2:])
        elif ln.startswith("+ "):
            added.append(ln[2:])
    return gone, added


# -------------------------------------------------------------- on by default
def test_on_by_default_the_briefs_are_reworded():
    assert Config().edit_plan_brief is True
    for pack in ("full", "small", "tiny"):
        cfg = _cfg(prompt_pack=pack)
        want = prompts._TINY_EDIT if pack == "tiny" else EDIT_FORMAT
        assert prompts.edit_format(cfg, EDIT_FORMAT) != want
        assert prompts.PLAN_LINE in prompts.edit_format(cfg, EDIT_FORMAT)
    # the flag-off path stays byte-identical to the original briefs
    for pack in ("full", "small", "tiny"):
        off = _cfg(prompt_pack=pack, edit_plan_brief=False)
        want = prompts._TINY_EDIT if pack == "tiny" else EDIT_FORMAT
        assert prompts.edit_format(off, EDIT_FORMAT) == want
        assert "nothing else around them" in want


# ------------------------------------------------------------------- new path
@pytest.mark.parametrize("pack", ["full", "tiny"])
def test_the_brief_rewords_exactly_two_places(pack):
    off = prompts.edit_format(_cfg(prompt_pack=pack, edit_plan_brief=False),
                              EDIT_FORMAT)
    on = prompts.edit_format(_cfg(prompt_pack=pack, edit_plan_brief=True),
                             EDIT_FORMAT)
    assert "nothing else around them" not in on
    assert "no other text between or inside blocks" in on.replace("\n", " ")
    assert prompts.PLAN_LINE in on
    gone, added = _changed_lines(off, on)
    assert gone and added
    # the example, rules and everything else are untouched
    assert on.count("<<<<<<< SEARCH") == off.count("<<<<<<< SEARCH")
    assert len(gone) <= 3 and len(added) <= 3


def test_the_plan_line_replaces_the_old_closing_line():
    on = prompts.edit_format(_cfg(edit_plan_brief=True), EDIT_FORMAT)
    assert "short plain explanation" not in on
    assert on.rstrip().endswith("output no blocks.")
    assert "Before the blocks, write 1-3 lines: the file, the lines or symbol " \
        "you will change, and why." in on
    tiny = prompts.edit_format(_cfg(prompt_pack="tiny", edit_plan_brief=True),
                               EDIT_FORMAT)
    assert "Write a short explanation first." not in tiny
    assert prompts.PLAN_LINE in tiny


def test_a_callers_own_brief_is_left_alone():
    cfg = _cfg(edit_plan_brief=True)
    assert prompts.edit_format(cfg, "CUSTOM FORMAT") == "CUSTOM FORMAT"
    assert prompts.plan_brief("nothing to reword") == "nothing to reword"


def test_anchor_style_rewords_the_brief_and_quote_is_the_default():
    """A2.5: edit_style 'anchor' prefers SYMBOL anchors; 'quote' (and unset)
    keep the quote-first wording."""
    assert Config().edit_style is None
    on = prompts.edit_format(_cfg(edit_style="anchor"), EDIT_FORMAT)
    assert prompts.ANCHOR_HINT in on
    assert "Prefer an anchor when quoting the old text exactly is" not in on
    for style in ("quote", None):
        off = prompts.edit_format(_cfg(edit_style=style), EDIT_FORMAT)
        assert "Prefer an anchor when quoting the old text exactly is" in off
        assert prompts.ANCHOR_HINT not in off
    # the tiny brief is reworded too
    tiny = prompts.edit_format(_cfg(prompt_pack="tiny", edit_style="anchor"),
                               EDIT_FORMAT)
    assert "Prefer SYMBOL over LINES" in tiny


def test_prompt_overrides_bypass_the_flag(tmp_path):
    root = tmp_path / "proj"
    (root / ".pedacito" / "prompts").mkdir(parents=True)
    (root / ".pedacito" / "prompts" / "edit_format.md").write_text(
        "MY OWN FORMAT: nothing else around them.")
    cfg = _cfg(edit_plan_brief=True)
    cfg._root = str(root)
    assert prompts.edit_format(cfg, EDIT_FORMAT) == \
        "MY OWN FORMAT: nothing else around them."
    (root / ".pedacito" / "prompts" / "answer.md").write_text("MY ANSWER BRIEF")
    assert prompts.answer_system(cfg, EDIT_FORMAT, "t") == "MY ANSWER BRIEF"


def test_the_answer_phase_sends_the_new_brief(tmp_path):
    for flag in (False, True):
        sub = tmp_path / str(flag)
        sub.mkdir()
        agent, client = gather(sub, READ_THEN_GREP,
                               {"edit_plan_brief": flag}, client=Scripted(
                                   READ_THEN_GREP, reply=PLAN + "\n" + BLOCK))
        agent.answer("Fix bulk_discount", "gathered source", stream=False)
        system = client.chats[-1][0]["content"]
        assert (prompts.PLAN_LINE in system) is flag
        assert ("nothing else around them" in system) is (not flag)


# ---------------------------------------------- the gather prompt is untouched
def test_gather_messages_do_not_depend_on_the_flag(tmp_path):
    heads = {}
    for flag in (False, True):
        sub = tmp_path / str(flag)
        sub.mkdir()
        _a, client = gather(sub, READ_THEN_GREP, {"edit_plan_brief": flag})
        distinct = {tuple(msgs[:2]) for msgs, _s in client.seen}
        assert len(distinct) == 1
        heads[flag] = distinct.pop()
    assert heads[False] == heads[True]


def test_replay_of_gather_and_answer_with_the_brief_on(tmp_path):
    _root, idx, cfg = make_project(tmp_path, edit_plan_brief=True)
    path = tmp_path / "rec.jsonl"
    rec = replay.Recorder(Scripted(READ_THEN_GREP, reply=PLAN + "\n" + BLOCK), path)
    agent = Agent(idx, rec, cfg)
    agent.gather("Fix bulk_discount", verbose=False)
    agent.answer("Fix bulk_discount", "src", stream=False)
    rep = replay.Replayer(path)
    agent2 = Agent(idx, rep, cfg)
    agent2.gather("Fix bulk_discount", verbose=False)
    assert agent2.answer("Fix bulk_discount", "src", stream=False) \
        == PLAN + "\n" + BLOCK
    assert rep.diverged == [] and len(rep.used) == len(rep.records)
    # the recorded digest of the answer call carries the new brief, so a
    # recording made with the flag off reports the divergence instead of
    # silently feeding the old answer
    _r2, idx2, cfg2 = make_project(tmp_path / "off", edit_plan_brief=False)
    rep2 = replay.Replayer(path)
    a3 = Agent(idx2, rep2, cfg2)
    a3.gather("Fix bulk_discount", verbose=False)
    a3.answer("Fix bulk_discount", "src", stream=False)
    assert rep2.diverged


# ------------------------------------------------ detectors (plan lines, I9)
def test_a_plan_line_with_blocks_is_not_an_unexecuted_promise():
    assert not loop._looks_like_unexecuted_promise(PLAN + "\n\n" + BLOCK)
    # no digits anywhere: the numbers in the old evidence test could not save it
    plain = "I'll rename `old_name` in pricing.py because the task asks for it."
    assert not loop._looks_like_unexecuted_promise(plain + "\n\n"
                                                   + BLOCK_NO_DIGITS)
    for opener in ("Let me fix", "I will fix", "I'm going to fix"):
        assert not loop._looks_like_unexecuted_promise(
            f"{opener} `old_name` in pricing.py.\n\n{BLOCK_NO_DIGITS}")


def test_a_bare_promise_is_still_a_promise():
    assert loop._looks_like_unexecuted_promise(
        "I'll run the game and report what I see.")
    assert loop._looks_like_unexecuted_promise(
        "Let me check the render path in pricing.py")
    assert not loop._looks_like_unexecuted_promise(
        "I'll run it. VERDICT: PASSED, the page rendered 640 pixels.")


def test_evidence_check_accepts_a_plan_line_before_blocks():
    plain = "I'll rename `old_name` in pricing.py because the task asks for it."
    tr = types.SimpleNamespace(reply=plain + "\n\n" + BLOCK_NO_DIGITS, steps=[],
                               live_ok=True)
    assert loop._evidence_failure("Verify pricing.py rename works", tr) == ""
    tr_bare = types.SimpleNamespace(reply="I'll run the check next.", steps=[],
                                    live_ok=True)
    assert "only promises" in loop._evidence_failure(
        "Verify pricing.py rename works", tr_bare)


def test_ordinary_plan_lines_are_not_no_ops():
    for plan in (PLAN,
                 "pricing.py, `bulk_discount` (L4-7): the guard excludes 10.",
                 "I'll rename `old_name` to `new_name` in pricing.py so the "
                 "callers match the spec."):
        assert not loop._is_noop(plan + "\n\n" + BLOCK)
        assert not loop._declares_incomplete(plan + "\n\n" + BLOCK)
        assert not loop._edit_refusal_head(plan + "\n\n" + BLOCK)
    # a genuine declaration is still one
    assert loop._is_noop("No changes needed: bulk_discount is already correct.")


# ----------------------------------------------------- end to end through eval
def test_eval_scores_a_plan_line_reply_as_a_clean_first_apply(tmp_path):
    task = next(t for t in evalrun.TASKS if t.id == "py-bulk-off-by-one")
    reply = (PLAN.replace("pricing.py", "shop/pricing.py") + "\n\n"
             "--- FILE: shop/pricing.py\n<<<<<<< SEARCH\n    if quantity > 10:\n"
             "=======\n    if quantity >= 10:\n>>>>>>> REPLACE\n")
    from test_features_misc import EvalModel, _eval_cfg
    cfg = _eval_cfg()
    cfg.edit_plan_brief = True
    row = evalrun.run_task(task, EvalModel(reply, [
        {"action": "read", "file": "shop/pricing.py", "start": 1, "end": 30}]),
        cfg, tmp_path)
    assert row["passed"] and row["first_apply_clean"] and row["failure"] == ""


# ------------------------------------------------------------------- config
def test_the_flag_is_a_plain_bool_in_config_and_the_gui(tmp_path):
    S.validate_text("edit_plan_brief = true\n")
    with pytest.raises(config_mod.ConfigError):
        S.validate_text('edit_plan_brief = "yes"\n')
    item = next(i for i in S.schema() if i["name"] == "edit_plan_brief")
    assert item["type"] == "bool" and item["default"] is True
    out = S.set_values(config_mod.TEMPLATE, {"edit_plan_brief": True})
    S.validate_text(out)
    kept = [ln for ln in out.splitlines() if ln != "edit_plan_brief = true"]
    assert kept == config_mod.TEMPLATE.splitlines()


def test_edit_style_is_a_closed_vocabulary_that_defaults_to_the_card():
    S.validate_text('edit_style = "anchor"\n')
    S.validate_text('edit_style = "quote"\n')
    with pytest.raises(config_mod.ConfigError):
        S.validate_text('edit_style = "sidebar"\n')
    item = next(i for i in S.schema() if i["name"] == "edit_style")
    assert item["type"] == "str" and item["default"] is None
    assert item["choices"] == ["quote", "anchor"]
