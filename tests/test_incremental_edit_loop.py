"""Symbol-anchored incremental edit-verify microloops.

The harness plans `[{file, symbol, intended_change}]`, then takes ONE verified
SEARCH/REPLACE hunk per symbol. Each hunk's exact source is appended at the
prompt tail, dry-applied, checked by verify.py + the syntax gate, and only then
written; the temporary source is replaced by a one-line receipt so the next
symbol's prompt stays small. The full impacted-test/lint pass runs once after
the last hunk.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from a1_support import make_project

from pedacito import config as config_mod
from pedacito import diagnostics, loop, modelcard, testsel
from pedacito import index as index_mod
from pedacito.agent import Agent, parse_edit_intents
from pedacito.config import Config
from pedacito.gui import settings as S


class SeqModel:
    """Scripted answer-phase client: one chat() reply per call."""

    def __init__(self, replies, steps=None):
        self.replies = list(replies)
        self.steps = list(steps or [])
        self.chat_calls: list[list[dict]] = []
        self.json_calls = 0
        self.stats = {}
        self.stream_truncated = False

    def chat_json(self, messages, schema, max_tokens, **kw):
        self.json_calls += 1
        if self.json_calls <= len(self.steps):
            return {"reasoning": "r", **self.steps[self.json_calls - 1]}
        return {"reasoning": "r", "action": "done"}

    def chat(self, messages, max_tokens, **kw):
        self.chat_calls.append([dict(m) for m in messages])
        return self.replies.pop(0)


def _two_file_project(tmp_path, **cfg_attrs):
    root, idx, cfg = make_project(tmp_path, **cfg_attrs)
    (root / "cart.py").write_text(
        "def total(items):\n    return sum(items)\n", encoding="utf-8")
    idx = index_mod.reindex_file(idx, "cart.py", cfg, save=False)
    return root, idx, cfg


HUNK_ADD_TAX = (
    "--- FILE: pricing.py SYMBOL add_tax\n<<<<<<< SEARCH\n=======\n"
    "def add_tax(price, rate):\n"
    "    return round(price * (1 + rate), 3)\n>>>>>>> REPLACE\n")
HUNK_TOTAL = (
    "--- FILE: cart.py SYMBOL total\n<<<<<<< SEARCH\n=======\n"
    "def total(items):\n    return sum(items) + 1\n>>>>>>> REPLACE\n")
INTENTS = [
    {"file": "pricing.py", "symbol": "add_tax",
     "intended_change": "round tax to three decimals"},
    {"file": "cart.py", "symbol": "total",
     "intended_change": "add one to the sum"},
]


# --------------------------------------------------------------------- config
def test_flag_defaults_on_and_is_visible_to_the_settings_editor():
    assert Config().incremental_edit_loop is True
    item = next(i for i in S.schema() if i["name"] == "incremental_edit_loop")
    assert item["type"] == "bool" and item["default"] is True
    S.validate_text("incremental_edit_loop = true\n")
    with pytest.raises(config_mod.ConfigError):
        S.validate_text('incremental_edit_loop = "yes"\n')


def test_model_card_enables_the_microloop_for_weak_edit_models():
    base = {"tool_choice": 0.6, "json_ok": 1.0, "edit_format": 1.0,
            "instructions": 1.0, "honesty": 1.0, "needle": 1.0,
            "continuation": 1.0}
    assert "incremental_edit_loop" not in modelcard.choose(base, False)
    weak = dict(base, edit_format=0.5)
    assert modelcard.choose(weak, False)["incremental_edit_loop"] is True


def test_intent_parser_accepts_prose_fences_and_wrappers():
    text = ("Plan:\n```json\n"
            '{"edits": [{"file": "a.py", "symbol": "f", '
            '"intended_change": "fix it"}]}\n```')
    assert parse_edit_intents(text) == [
        {"file": "a.py", "symbol": "f", "intended_change": "fix it"}]
    assert parse_edit_intents("no json here") == []


# ------------------------------------------------------- sequential microloop
def test_symbols_are_edited_one_hunk_at_a_time_with_source_popped(tmp_path):
    root, idx, cfg = _two_file_project(tmp_path)
    client = SeqModel([HUNK_ADD_TAX, HUNK_TOTAL])
    agent = Agent(idx, client, cfg)

    results, reply, findings = agent.apply_edit_intents(
        "Change add_tax and total", "gathered source", INTENTS, verbose=False)

    assert [r.ok for r in results] == [True, True]
    assert not [f for f in findings if f.severity == "error"]
    assert "rate), 3" in (root / "pricing.py").read_text(encoding="utf-8")
    assert "sum(items) + 1" in (root / "cart.py").read_text(encoding="utf-8")
    assert reply.splitlines()[1:] == [
        "[EDITED pricing.py:add_tax]", "[EDITED cart.py:total]"]

    assert len(client.chat_calls) == 2
    first, second = client.chat_calls
    # Each call appends only the CURRENT symbol's exact source at the tail.
    assert "def add_tax" in first[-1]["content"]
    assert "def total" not in first[-1]["content"]
    assert "def total" in second[-1]["content"]
    # The first symbol's source is gone; its receipt remains.
    assert "def add_tax" not in second[-1]["content"]
    assert "[EDITED pricing.py:add_tax]" in second[-1]["content"]


def test_an_invalid_hunk_is_rejected_before_the_next_symbol(tmp_path):
    root, idx, cfg = _two_file_project(tmp_path)
    bad = HUNK_ADD_TAX.replace("    return round(price * (1 + rate), 3)\n",
                               "    return round(price * (1 + rate), 3\n")
    client = SeqModel([bad, HUNK_ADD_TAX, HUNK_TOTAL])
    agent = Agent(idx, client, cfg)

    results, _reply, _findings = agent.apply_edit_intents(
        "Change add_tax and total", "gathered source", INTENTS, verbose=False)

    assert [r.ok for r in results] == [True, True]
    # The bad syntax never reached disk; the repaired hunk did.
    assert "rate), 3" in (root / "pricing.py").read_text(encoding="utf-8")
    assert len(client.chat_calls) == 3
    assert "WAS REJECTED" in client.chat_calls[1][-1]["content"]
    assert "syntax error" in client.chat_calls[1][-1]["content"]


# ----------------------------------------------------------- loop integration
def test_loop_runs_tests_and_lint_after_all_symbol_hunks(tmp_path, monkeypatch):
    root, idx, cfg = _two_file_project(
        tmp_path, verify_tests=True, allow_shell=True, post_edit_check=True,
        refresh_summaries=0)
    client = SeqModel(
        [
            ('[{"file": "pricing.py", "symbol": "add_tax", '
             '"intended_change": "round to three decimals"}, '
             '{"file": "cart.py", "symbol": "total", '
             '"intended_change": "add one"}]'),
            HUNK_ADD_TAX,
            HUNK_TOTAL,
        ],
        steps=[{"action": "read", "file": "pricing.py", "start": 1, "end": 8},
               {"action": "done"}],
    )
    test_runs: list[tuple[list[str], bool, bool]] = []

    def fake_impacted(_idx, _cfg, files):
        test_runs.append((
            list(files),
            "rate), 3" in (root / "pricing.py").read_text(encoding="utf-8"),
            "sum(items) + 1" in (root / "cart.py").read_text(encoding="utf-8")))
        return True, "", "fake impacted tests"

    lint_runs: list[tuple[list[str], bool, bool]] = []

    def fake_quick(_root, files):
        lint_runs.append((
            list(files),
            "rate), 3" in (root / "pricing.py").read_text(encoding="utf-8"),
            "sum(items) + 1" in (root / "cart.py").read_text(encoding="utf-8")))
        return {}

    monkeypatch.setattr(testsel, "run_impacted", fake_impacted)
    monkeypatch.setattr(diagnostics, "quick_findings", fake_quick)

    agent = Agent(idx, client, cfg)
    tr = loop.run_task(
        root, cfg, client, agent, idx,
        "Change add_tax in pricing.py and total in cart.py", apply=True)

    assert tr.applied and sorted(tr.changed) == ["cart.py", "pricing.py"]
    assert [r.ok for r in tr.results] == [True, True]
    # One baseline run before the hunks, one full run after BOTH landed.
    assert test_runs[0][1:] == (False, False)
    assert test_runs[-1][1:] == (True, True)
    assert lint_runs[0][1:] == (False, False)
    assert lint_runs[-1][1:] == (True, True)


# ------------------------------------------- coverage-seeded second pass (R4)
TWO_CLAUSE_TASK = ("Add apply_discount in pricing.py and add compute_shipping "
                   "in shipping.py")
HUNK_DISCOUNT = (
    "--- FILE: pricing.py SYMBOL apply_discount\n<<<<<<< SEARCH\n=======\n"
    "def apply_discount(price, pct):\n"
    "    return round(price * (1 - pct), 2)\n>>>>>>> REPLACE\n")
HUNK_SHIPPING = (
    "--- FILE: shipping.py SYMBOL compute_shipping\n<<<<<<< SEARCH\n=======\n"
    "def compute_shipping(weight):\n"
    "    return round(weight * 0.5, 2)\n>>>>>>> REPLACE\n")
INTENT_DISCOUNT = [{"file": "pricing.py", "symbol": "apply_discount",
                    "intended_change": "add the discount helper"}]


def _two_clause_project(tmp_path, **cfg_attrs):
    root, idx, cfg = make_project(tmp_path, **cfg_attrs)
    (root / "pricing.py").write_text(
        (root / "pricing.py").read_text(encoding="utf-8")
        + "\ndef apply_discount(price, pct):\n    return price\n",
        encoding="utf-8")
    (root / "shipping.py").write_text(
        "def compute_shipping(weight):\n    return weight\n",
        encoding="utf-8")
    idx = index_mod.reindex_file(idx, "pricing.py", cfg, save=False)
    idx = index_mod.reindex_file(idx, "shipping.py", cfg, save=False)
    return root, idx, cfg


def test_unaddressed_clause_seeds_one_bounded_followup_pass(tmp_path):
    """The first batch answers only clause one; the requirement ledger flags
    clause two and seeds a second pass whose intent names it -- no planner
    call, no third pass."""
    root, idx, cfg = _two_clause_project(tmp_path)
    client = SeqModel([HUNK_DISCOUNT, HUNK_SHIPPING])
    agent = Agent(idx, client, cfg)
    events = []
    agent._emit = lambda name, **kw: events.append((name, kw))

    results, reply, findings = agent.apply_edit_intents(
        TWO_CLAUSE_TASK, "gathered source", INTENT_DISCOUNT, verbose=False)

    assert [r.ok for r in results] == [True, True]
    assert not [f for f in findings if f.severity == "error"]
    assert "def apply_discount" in (root / "pricing.py").read_text(encoding="utf-8")
    assert "def compute_shipping" in (root / "shipping.py").read_text(encoding="utf-8")
    assert "Coverage follow-up" in reply
    assert "[EDITED shipping.py:compute_shipping]" in reply
    # Exactly one follow-up pass, and its target message names clause two.
    followups = [kw for name, kw in events if name == "coverage_followup"]
    assert len(followups) == 1
    assert any("compute_shipping" in c for c in followups[0]["clauses"])
    assert len(client.chat_calls) == 2
    second = client.chat_calls[1][-1]["content"]
    assert "shipping.py" in second
    assert "compute_shipping" in second


def test_followup_pass_does_not_recurse(tmp_path):
    """Even when the follow-up hunk never lands, the pass is bounded: one
    follow-up with its usual retry ceiling, then the batch ends -- no third
    pass, and the failed target is reported as failed."""
    root, idx, cfg = _two_clause_project(tmp_path)
    bad = HUNK_SHIPPING.replace("    return round(weight * 0.5, 2)\n",
                                "    return round(weight * 0.5, 2\n")
    client = SeqModel([HUNK_DISCOUNT, bad, bad, bad])
    agent = Agent(idx, client, cfg)
    events = []
    agent._emit = lambda name, **kw: events.append((name, kw))

    results, reply, _findings = agent.apply_edit_intents(
        TWO_CLAUSE_TASK, "", INTENT_DISCOUNT, verbose=False)

    assert [r.ok for r in results] == [True, False]
    assert "return weight" in (root / "shipping.py").read_text(encoding="utf-8")
    assert sum(1 for name, _ in events if name == "coverage_followup") == 1
    # one hunk for clause one, then the follow-up's three bounded attempts
    assert len(client.chat_calls) == 4
    assert "[FAILED shipping.py:compute_shipping" in reply


def test_no_followup_when_every_clause_is_addressed(tmp_path):
    root, idx, cfg = _two_clause_project(tmp_path)
    client = SeqModel([HUNK_DISCOUNT, HUNK_SHIPPING])
    agent = Agent(idx, client, cfg)
    events = []
    agent._emit = lambda name, **kw: events.append((name, kw))
    intents = INTENT_DISCOUNT + [
        {"file": "shipping.py", "symbol": "compute_shipping",
         "intended_change": "add the shipping helper"}]

    results, reply, _findings = agent.apply_edit_intents(
        TWO_CLAUSE_TASK, "", intents, verbose=False)

    assert [r.ok for r in results] == [True, True]
    assert not [name for name, _ in events if name == "coverage_followup"]
    assert "Coverage follow-up" not in reply
    assert len(client.chat_calls) == 2
