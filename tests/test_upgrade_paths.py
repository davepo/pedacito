"""The upgrade paths from PEDACITO_VALIDATION_REPORT_2.md, exercised end to
end at the deterministic layer (no live model):

  U1  escalation policy   -- repeat suppression, two-lease strikes, the
                             semantic-failure signal, per-profile zero-edit
                             statistics, and the ladder's reason-based retry.
  U2  context-window edit discipline -- the whole-file marker, the oversized
                             rewrite gate, and the small-window answer note.
  U3  numeric-SEARCH auto-reanchor.
  U4  prose-vs-edit gap naming.
  U5  shrinking micro-loop retry budgets.

The headless DOM smoke check (U6) lives in test_domcheck.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_agent_context import Scripted, _agent  # noqa: E402

from pedacito import edits, loop, memory, prompts, verify  # noqa: E402
from pedacito.config import Config  # noqa: E402

SRC = {
    "pkg/__init__.py": "",
    "pkg/core.py": "def apply_discount(price):\n    return price * 0.9\n",
    "pkg/shop.py": ("from pkg.core import apply_discount\n\n\n"
                    "def checkout(p):\n    return apply_discount(p)\n"),
}


def _block(rel, search, replace, header=""):
    return (f"--- FILE: {rel}{header}\n<<<<<<< SEARCH\n{search}\n"
            f"=======\n{replace}\n>>>>>>> REPLACE\n")


# --- U3: numeric-SEARCH auto-reanchor ----------------------------------------

def test_numeric_search_reanchors_to_the_line(tmp_path):
    (tmp_path / "f.txt").write_text("alpha\nbeta\ngamma\ndelta\n")
    reply = _block("f.txt", "3", "GAMMA")
    res = edits.apply_edits(tmp_path, edits.parse_edits(reply), backup=False)
    assert res[0].ok, res[0].message
    assert (tmp_path / "f.txt").read_text() == "alpha\nbeta\nGAMMA\ndelta\n"
    assert "auto-reanchored" in res[0].message


def test_numeric_search_that_occurs_lit_keeps_text_semantics(tmp_path):
    (tmp_path / "f.txt").write_text("version 3\nother\n")
    reply = _block("f.txt", "3", "4")
    res = edits.apply_edits(tmp_path, edits.parse_edits(reply), backup=False)
    assert res[0].ok, res[0].message
    # matched the literal "3" inside "version 3", NOT line 3
    assert (tmp_path / "f.txt").read_text() == "version 4\nother\n"
    assert "auto-reanchored" not in res[0].message


def test_numeric_search_out_of_range_refuses_with_teaching(tmp_path):
    (tmp_path / "f.txt").write_text("alpha\nbeta\n")
    reply = _block("f.txt", "99", "X")
    res = edits.apply_edits(tmp_path, edits.parse_edits(reply), backup=False)
    assert not res[0].ok
    assert "99" in res[0].message and "line number" in res[0].message
    # nothing was written
    assert (tmp_path / "f.txt").read_text() == "alpha\nbeta\n"


# --- U2: context-window edit discipline --------------------------------------

def test_empty_search_overwrite_marks_whole_file(tmp_path):
    big = "\n".join(f"line{i}" for i in range(200)) + "\n"
    (tmp_path / "big.txt").write_text(big)
    # a genuine same-size rewrite (no copied lines) passes the shrink guard
    fresh = "\n".join(f"newline{i}" for i in range(200)) + "\n"
    reply = _block("big.txt", "", fresh)
    res = edits.apply_edits(tmp_path, edits.parse_edits(reply),
                            backup=False, dry_run=True)
    assert res[0].ok, res[0].message
    assert res[0].whole_file


def _small_cfg(**kw):
    cfg = Config()
    cfg.small_context_tokens = 16384
    cfg.context_tokens = 8192
    cfg.max_whole_file_lines = 120
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_oversized_whole_file_rewrite_refused_on_small_window():
    agent = SimpleNamespace(cfg=_small_cfg())
    before = "\n".join(f"line{i}" for i in range(200))
    r = edits.Result(True, "big.py", "applied", before=before, whole_file=True)
    found = verify.check_oversized_rewrite(agent, [r], task="fix the bug")
    assert found and found[0].kind == "oversized_rewrite"
    assert found[0].severity == "error"


def test_oversized_rewrite_warns_when_task_asks_for_that_rewrite():
    agent = SimpleNamespace(cfg=_small_cfg())
    before = "\n".join(f"line{i}" for i in range(200))
    r = edits.Result(True, "big.py", "applied", before=before, whole_file=True)
    found = verify.check_oversized_rewrite(
        agent, [r], task="rewrite big.py from scratch")
    assert found and found[0].severity == "warn"


def test_oversized_rewrite_off_for_large_window_or_small_file():
    before = "\n".join(f"line{i}" for i in range(200))
    r = edits.Result(True, "big.py", "applied", before=before, whole_file=True)
    # large window: no gate
    big_window = _small_cfg(context_tokens=100000)
    assert verify.check_oversized_rewrite(
        SimpleNamespace(cfg=big_window), [r], task="x") == []
    # small file under the ceiling: no gate
    small_file = edits.Result(True, "s.py", "applied",
                              before="a\nb\nc\n", whole_file=True)
    assert verify.check_oversized_rewrite(
        SimpleNamespace(cfg=_small_cfg()), [small_file], task="x") == []
    # a non-whole-file edit is never gated
    hunk = edits.Result(True, "big.py", "applied", before=before,
                        whole_file=False)
    assert verify.check_oversized_rewrite(
        SimpleNamespace(cfg=_small_cfg()), [hunk], task="x") == []


def test_small_window_note_names_large_files_and_is_off_when_roomy():
    note = prompts.small_window_note(_small_cfg(), ["big.py", "huge.py"])
    assert "CONTEXT PRESSURE" in note and "big.py" in note
    assert prompts.small_window_note(_small_cfg(context_tokens=100000)) == ""
    # the discipline is opt-out via small_context_tokens = 0
    assert prompts.small_window_note(_small_cfg(small_context_tokens=0)) == ""


# --- U4: prose-vs-edit gap naming --------------------------------------------

def test_prose_only_gap_is_named_as_description_not_implementation():
    task = "Add apply_discount in pricing.py and add compute_tax in tax.py"
    tracker = verify.RequirementTracker(task)
    reply = ("I added apply_discount. Next I will add compute_tax to compute "
             "the tax.\n\n" + _block("pricing.py", "",
                                     "def apply_discount(p, pct):\n"
                                     "    return p * (1 - pct)\n"))
    eds = verify.parse_edits(reply)
    found = verify.check_coverage(task, reply, eds, tracker=tracker)
    msgs = [f.message for f in found]
    assert any("compute_tax" in m and "described" in m for m in msgs), msgs
    # the implemented clause is not flagged
    assert not any("apply_discount" in m for m in msgs), msgs


def test_unaddressed_clause_absent_from_prose_keeps_generic_wording():
    task = ("Add apply_discount in pricing.py and add compute_shipping in "
            "shipping.py")
    tracker = verify.RequirementTracker(task)
    reply = ("Done.\n\n" + _block("pricing.py", "",
                                  "def apply_discount(p, pct):\n"
                                  "    return p * (1 - pct)\n"))
    eds = verify.parse_edits(reply)
    found = verify.check_coverage(task, reply, eds, tracker=tracker)
    msgs = [f.message for f in found]
    assert any("compute_shipping" in m and "no edit or sentence" in m
               for m in msgs), msgs
    assert not any("described" in m for m in msgs), msgs


# --- U5: shrinking micro-loop retry budgets ----------------------------------

def test_incremental_retry_budgets_shrink(tmp_path, monkeypatch):
    agent = _agent(tmp_path, Scripted(), {"m.py": "def f():\n    return 1\n"},
                   max_tokens_answer=8000, repair_rounds=3)
    budgets = []

    def fake_chat(setting, messages, client=None, **kw):
        budgets.append(kw.get("max_tokens"))
        return "no edit block here"      # rejected -> forces the next attempt

    monkeypatch.setattr(agent, "_chat_at", fake_chat)
    intents = [{"file": "m.py", "symbol": "f", "intended_change": "change f"}]
    agent.apply_edit_intents("change f", "", intents)
    # attempts = 1 + min(repair_rounds, 3) = 4
    assert len(budgets) == 4, budgets
    assert budgets[0] == 8000
    assert budgets[1] == max(2048, 8000 // 2)
    assert budgets[2] == max(2048, 8000 // 4)
    assert budgets[3] == max(2048, 8000 // 6)
    assert budgets == sorted(budgets, reverse=True)


# --- U1: repeat suppression and lease strikes --------------------------------

def test_identical_readonly_repeat_is_suppressed(tmp_path):
    client = Scripted(steps=[
        {"action": "grep", "pattern": "checkout"},
        {"action": "grep", "pattern": "checkout"},
        {"action": "done"}])
    agent = _agent(tmp_path, client, SRC)
    agent.gather("look around", verbose=False)
    seen = "\n".join(m["content"] for m in client.json_calls[-1])
    assert "already ran and its result is above" in seen


def test_two_lease_strikes_suggest_escalation(tmp_path):
    agent = _agent(tmp_path, Scripted(), {"a.py": "x = 1\n"})
    agent._note_lease_strike("stuck task")
    assert not getattr(agent.cfg, "_task_escalation_reason", "")
    agent._note_lease_strike("stuck task")
    reason = getattr(agent.cfg, "_task_escalation_reason", "")
    assert reason and "leases" in reason
    # a third strike does not overwrite the first reason
    agent._note_lease_strike("stuck task")
    assert getattr(agent.cfg, "_task_escalation_reason", "") == reason


def test_semantic_failure_regex_excludes_syntax_errors():
    assert loop._SEMANTIC_FAILURE_RE.search("TypeError: x is not a function")
    assert loop._SEMANTIC_FAILURE_RE.search("AssertionError: expected 3")
    assert loop._SEMANTIC_FAILURE_RE.search("RangeError: maximum call stack")
    assert not loop._SEMANTIC_FAILURE_RE.search("SyntaxError: unexpected token")


# --- U1: per-profile zero-edit statistics ------------------------------------

def test_zero_edit_stats_count_per_profile_and_clear(tmp_path):
    small = Config(); small.active_profile = "small"
    big = Config(); big.active_profile = "big"
    assert memory.zero_edit_runs(tmp_path, small) == 0
    assert memory.note_zero_edit(tmp_path, small) == 1
    assert memory.note_zero_edit(tmp_path, small) == 2
    assert memory.note_zero_edit(tmp_path, big) == 1
    assert memory.zero_edit_runs(tmp_path, small) == 2
    memory.clear_zero_edit(tmp_path, small)
    assert memory.zero_edit_runs(tmp_path, small) == 0
    assert memory.zero_edit_runs(tmp_path, big) == 1


def _zero_edit_tr():
    return loop.TaskResult([], "I looked around but produced no usable edit",
                           False, [], None)


def test_run_task_queues_escalation_after_zero_edit_threshold(tmp_path,
                                                              monkeypatch):
    cfg = Config(); cfg.active_profile = "small"
    cfg.escalation_zero_edit_runs = 3
    monkeypatch.setattr(loop, "_run_ladder", lambda *a, **k: _zero_edit_tr())
    for _ in range(2):
        loop.run_task(tmp_path, cfg, "client", object(), None, "t")
    assert not getattr(cfg, "_pending_escalation_reason", "")
    loop.run_task(tmp_path, cfg, "client", object(), None, "t")
    reason = getattr(cfg, "_pending_escalation_reason", "")
    assert reason and "zero-edit" in reason and "small" in reason


def test_run_task_success_clears_the_profile_counter(tmp_path, monkeypatch):
    cfg = Config(); cfg.active_profile = "small"
    memory.note_zero_edit(tmp_path, cfg)
    memory.note_zero_edit(tmp_path, cfg)
    ok = loop.TaskResult([SimpleNamespace(ok=True)], "done", True, ["f.py"],
                         None)
    monkeypatch.setattr(loop, "_run_ladder", lambda *a, **k: ok)
    loop.run_task(tmp_path, cfg, "client", object(), None, "t")
    assert memory.zero_edit_runs(tmp_path, cfg) == 0


# --- U1: the ladder's reason-based escalation --------------------------------

class _LadderTR:
    def __init__(self, client):
        self.results = [SimpleNamespace(ok=True)]
        self.applied = True
        self.changed = ["f.py"]
        self.reply = "done"
        self.idx = None
        self.tests_ok = False if client == "weak" else True


def _ladder_patches(monkeypatch, calls):
    from pedacito import roles

    def fake_once(root, cfg, client, agent, idx, task, **kw):
        calls.append(client)
        return _LadderTR(client)

    monkeypatch.setattr(loop, "_run_task_once", fake_once)
    monkeypatch.setattr(roles, "client_for",
                        lambda cfg, role, c: "strong" if role == "escalate"
                        else c)
    monkeypatch.setattr("pedacito.agent.Agent", lambda idx, c, cfg: object())


def test_ladder_escalates_on_reason_even_when_edits_landed(tmp_path,
                                                           monkeypatch):
    calls = []
    _ladder_patches(monkeypatch, calls)
    cfg = Config()
    cfg._pending_escalation_reason = "semantic failure after repair rounds"
    tr = loop._run_ladder(tmp_path, cfg, "weak", object(), None, "t")
    assert calls == ["weak", "strong"]
    assert tr.tests_ok is True          # the stronger model's green result wins


def test_ladder_keeps_green_first_attempt_over_red_escalation(tmp_path,
                                                              monkeypatch):
    calls = []
    from pedacito import roles

    def fake_once(root, cfg, client, agent, idx, task, **kw):
        calls.append(client)
        tr = _LadderTR(client)
        tr.tests_ok = True if client == "weak" else False
        return tr

    monkeypatch.setattr(loop, "_run_task_once", fake_once)
    monkeypatch.setattr(roles, "client_for",
                        lambda cfg, role, c: "strong" if role == "escalate"
                        else c)
    monkeypatch.setattr("pedacito.agent.Agent", lambda idx, c, cfg: object())
    cfg = Config()
    cfg._pending_escalation_reason = "two no-progress leases"
    tr = loop._run_ladder(tmp_path, cfg, "weak", object(), None, "t")
    assert calls == ["weak", "strong"]
    assert tr.tests_ok is True          # first attempt was green: keep it


def test_ladder_reason_escalation_respects_auto_escalate_off(tmp_path,
                                                             monkeypatch):
    calls = []
    _ladder_patches(monkeypatch, calls)
    cfg = Config(); cfg.auto_escalate = False
    cfg._pending_escalation_reason = "semantic failure"
    loop._run_ladder(tmp_path, cfg, "weak", object(), None, "t")
    assert calls == ["weak"]            # no retry when escalation is disabled


def test_ladder_consumes_pending_reason_once(tmp_path, monkeypatch):
    calls = []
    _ladder_patches(monkeypatch, calls)
    cfg = Config()
    cfg._pending_escalation_reason = "zero-edit statistic"
    loop._run_ladder(tmp_path, cfg, "weak", object(), None, "t")
    # the reason was consumed into the task slot and the pending slot cleared
    assert getattr(cfg, "_pending_escalation_reason", "") == ""
    # a second task with no new reason does not escalate again
    calls.clear()
    loop._run_ladder(tmp_path, cfg, "weak", object(), None, "t2")
    assert calls == ["weak"]