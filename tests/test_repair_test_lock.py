"""The repair-phase test-file lock (TECHNIQUES.md #46).

A gate repair may not rewrite a PRE-EXISTING test file to turn it green --
the prose TEST_RULES alone did not stop small models from weakening the red
assertion instead of fixing the code. The lock is deterministic, one-shot
(armed by the repair re-ask, consumed by the next real apply), and exempt
when the task or an edit requirement explicitly asks for test changes, when
the task itself created the test file, or when a deterministic gate proved
the file broken.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_agent_context import Scripted, _agent  # noqa: E402

from pedacito import loop, prompts, verify  # noqa: E402
from pedacito.agent import _gate_flagged_files  # noqa: E402
from pedacito.config import Config  # noqa: E402
from pedacito.edits import Result  # noqa: E402

CORE = "def discount(price):\n    return price * 0.9\n"
TEST = ("from pkg.core import discount\n\n\n"
        "def test_discount():\n"
        "    assert discount(100) == 90.0\n")
FILES = {
    "pkg/__init__.py": "",
    "pkg/core.py": CORE,
    "tests/test_core.py": TEST,
}
TASK = "fix the discount rounding in pkg/core.py"


def _block(rel, search, replace):
    return (f"--- FILE: {rel}\n<<<<<<< SEARCH\n{search}\n"
            f"=======\n{replace}\n>>>>>>> REPLACE\n")


CODE_FIX = _block("pkg/core.py", "return price * 0.9",
                  "return round(price * 0.9, 2)")
TEST_WEAKEN = _block("tests/test_core.py", "assert discount(100) == 90.0",
                     "assert discount(100) >= 0")


# --- the exemption predicate: explicit test-change requests only -------------

@pytest.mark.parametrize("task", [
    "add a regression test for the parser",
    "fix the failing test",
    "write unit tests for the cache",
    "update tests/test_cart.py to the new API",
    "fix test_cart.py",
    "update the code and tests",
    "improve test coverage",
])
def test_wants_test_changes_accepts_explicit_requests(task):
    assert prompts.wants_test_changes(task)


@pytest.mark.parametrize("task", [
    "make the tests pass",
    "run the tests",
    "fix the cart total so tests pass",
    "fix the bug so the tests pass",
    "ensure tests pass",
    "the tests should pass after the change",
    "fix the discount rounding",
])
def test_wants_test_changes_rejects_make_them_pass_requests(task):
    assert not prompts.wants_test_changes(task)


def test_gate_flagged_files_excludes_failing_test_findings():
    contract = verify.Finding(
        "contract", "error", "",
        "ContractError in tests/test_core.py:5: calls discount() with 2 args")
    failing = verify.Finding(
        "tests", "error", "",
        "tests/test_core.py:5: AssertionError\n"
        "FAILED tests/test_core.py::test_discount")
    assert "tests/test_core.py" in _gate_flagged_files([contract])
    # A red assertion is NOT proof the test is wrong: no exemption from it.
    assert _gate_flagged_files([failing]) == set()


# --- the lock itself ----------------------------------------------------------

def _locked_agent(tmp_path, task=TASK, files=None):
    agent = _agent(tmp_path, Scripted(), files or FILES)
    agent._arm_repair_test_lock(task)
    return agent


def test_repair_refuses_weakening_a_pre_existing_test(tmp_path):
    agent = _locked_agent(tmp_path)
    res = agent._apply_once(CODE_FIX + TEST_WEAKEN, dry_run=False)
    by_file = {r.file: r for r in res}
    assert by_file["pkg/core.py"].ok
    assert not by_file["tests/test_core.py"].ok
    assert "refused" in by_file["tests/test_core.py"].message
    # the code fix landed; the assertion is untouched on disk
    assert "round" in (tmp_path / "pkg/core.py").read_text(encoding="utf-8")
    assert "assert discount(100) == 90.0" in \
        (tmp_path / "tests/test_core.py").read_text(encoding="utf-8")


def test_lock_is_one_shot(tmp_path):
    agent = _locked_agent(tmp_path)
    first = agent._apply_once(TEST_WEAKEN, dry_run=False)
    assert not first[0].ok
    # no re-arm: the next apply is an ordinary one again
    second = agent._apply_once(TEST_WEAKEN, dry_run=False)
    assert all(r.ok for r in second)


def test_dry_run_filters_without_consuming_the_lock(tmp_path):
    agent = _locked_agent(tmp_path)
    dry = agent._apply_once(TEST_WEAKEN, dry_run=True)
    assert not dry[0].ok and "refused" in dry[0].message
    real = agent._apply_once(TEST_WEAKEN, dry_run=False)
    assert not real[0].ok
    assert "assert discount(100) == 90.0" in \
        (tmp_path / "tests/test_core.py").read_text(encoding="utf-8")


def test_task_asking_for_test_changes_is_exempt(tmp_path):
    agent = _locked_agent(tmp_path, task="add a test for the discount")
    res = agent._apply_once(TEST_WEAKEN, dry_run=False)
    assert all(r.ok for r in res)


def test_edit_requirement_asking_for_tests_is_exempt(tmp_path):
    agent = _locked_agent(tmp_path)
    tracker = verify.RequirementTracker("update the tests for the new API")
    tracker.task = TASK
    agent.requirements = tracker
    res = agent._apply_once(TEST_WEAKEN, dry_run=False)
    assert all(r.ok for r in res)


def test_test_file_created_by_this_task_is_not_locked(tmp_path):
    files = dict(FILES)
    files["tests/test_new.py"] = "def test_new():\n    assert 1 == 1\n"
    agent = _agent(tmp_path, Scripted(), files)
    # the ladder's snapshot predates the task's own creation
    agent.cfg._task_preexisting = set(FILES)
    agent._arm_repair_test_lock(TASK)
    reply = _block("tests/test_new.py", "assert 1 == 1", "assert 1 == 2")
    res = agent._apply_once(reply, dry_run=False)
    assert all(r.ok for r in res)


def test_gate_flagged_test_file_is_exempt(tmp_path):
    agent = _locked_agent(tmp_path)
    agent._arm_repair_test_lock(TASK, exempt=["tests/test_core.py"])
    res = agent._apply_once(TEST_WEAKEN, dry_run=False)
    assert all(r.ok for r in res)


def test_lock_normalises_model_path_spellings(tmp_path):
    agent = _locked_agent(tmp_path)
    reply = _block("./tests/test_core.py", "assert discount(100) == 90.0",
                   "assert True")
    res = agent._apply_once(reply, dry_run=False)
    assert not res[0].ok and "refused" in res[0].message


def test_non_test_files_are_never_locked(tmp_path):
    agent = _locked_agent(tmp_path)
    res = agent._apply_once(CODE_FIX, dry_run=False)
    assert all(r.ok for r in res)


# --- arming: the repair re-ask owns the lock ----------------------------------

def test_repair_re_ask_arms_the_lock(tmp_path):
    client = Scripted(replies=[CODE_FIX + TEST_WEAKEN])
    agent = _agent(tmp_path, client, FILES)
    err = verify.Finding("tests", "error", "", "the suite fails")
    fixed = agent._repair_from_findings(TASK, "original reply", [err], [])
    assert fixed
    assert agent._repair_lock_task == TASK
    res = agent._apply_once(fixed, dry_run=False)
    by_file = {r.file: r for r in res}
    assert by_file["pkg/core.py"].ok
    assert not by_file["tests/test_core.py"].ok


def test_preflight_repair_does_not_arm_the_lock(tmp_path):
    client = Scripted(replies=[CODE_FIX])
    agent = _agent(tmp_path, client, FILES)
    err = verify.Finding("placeholder", "error", "pkg/core.py", "stubbed")
    agent._repair_from_findings(TASK, "original reply", [err], [],
                                test_lock=False)
    assert agent._repair_lock_task is None


# --- end to end: the syntax-gate repair seat ----------------------------------

def test_post_edit_repair_seat_cannot_weaken_the_test(tmp_path, monkeypatch):
    cfg = Config()
    cfg.verify_tests = True
    cfg.allow_shell = True
    cfg.repair_rounds = 1
    cfg.log_events = False
    cfg.backup = False
    # isolate the syntax seat: every later gate is green, tests pass
    for name in ("_new_static_errors", "_new_frontend_errors",
                 "_new_contract_errors", "_new_dom_errors"):
        monkeypatch.setattr(loop, name, lambda *a, **k: [])
    monkeypatch.setattr(loop, "_run_tests",
                        lambda *a, **k: (True, "1 passed", "pytest -q"))
    broken = CORE.replace("return price * 0.9", "return price * (")
    client = Scripted(replies=[CODE_FIX + TEST_WEAKEN])
    agent = _agent(tmp_path, client, FILES)
    results = [Result(True, "pkg/core.py", "edit applied",
                      before=CORE, after=broken)]
    out, reply, tests_ok, _preexisting = loop._post_edit_tests(
        tmp_path, cfg, agent, agent.index, TASK, results, "original reply",
        None, None, ["pkg/core.py"])
    assert tests_ok is True
    by_file = {r.file: r for r in out}
    assert not by_file["tests/test_core.py"].ok
    assert "refused" in by_file["tests/test_core.py"].message
    assert "assert discount(100) == 90.0" in \
        (tmp_path / "tests/test_core.py").read_text(encoding="utf-8")
    assert "round" in (tmp_path / "pkg/core.py").read_text(encoding="utf-8")