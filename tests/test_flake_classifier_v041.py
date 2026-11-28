"""WP-4105D: flake and infrastructure classification, bounded node rerun,
quarantine metadata and repair routing. Every subprocess is a local fake."""
from __future__ import annotations

import datetime
import json
import sys
import textwrap
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import events, loop, test_classify as tc, testsel  # noqa: E402
from pedacito.config import Config  # noqa: E402
from pedacito.edits import Result  # noqa: E402

BASELINE = textwrap.dedent("""\
    FAILED tests/test_a.py::test_old - assert 1 == 2
    1 failed in 0.10s
    """)
CURRENT = textwrap.dedent("""\
    FAILED tests/test_a.py::test_old - assert 1 == 2
    FAILED tests/test_c.py::test_new - assert 5 == 6
    2 failed in 0.20s
    """)
NEW_ONLY = "FAILED tests/test_c.py::test_new - assert 5 == 6\n1 failed in 0.10s\n"
PASS_RERUN = "tests/test_c.py::test_new PASSED\n1 passed in 0.05s\n"
RERUN_SAME = "FAILED tests/test_c.py::test_new - assert 5 == 6\n1 failed in 0.10s\n"
TIMEOUT_OUT = "`python -m pytest -q` was stopped (timeout) after 300s\n"
COLLECTION_OUT = ("ERROR collecting tests/test_c.py\n"
                  "E   ImportError: cannot import name 'x'\n"
                  "!!! Interrupted: 1 error during collection !!!\n")


def _result(before: str, after: str, file: str = "a.py") -> Result:
    return Result(True, file, "edit applied", before=before, after=after)


# --------------------------------------------------------- output kinds
@pytest.mark.parametrize("text, kind", [
    (TIMEOUT_OUT, "timeout"),
    ("`pytest` was stopped (idle) after 10s", "timeout"),
    (COLLECTION_OUT, "collection_error"),
    ("ModuleNotFoundError: No module named 'pygame'", "missing_dependency"),
    ("pytest is not installed for this interpreter", "missing_dependency"),
    ("could not run `pytest -q`: sandbox refused", "infra"),
    ("ok 1 passed", ""),
])
def test_output_kind_names_infrastructure_without_node_failures(text, kind):
    assert tc.classify_output_kind(text) == kind


def test_node_failure_suppresses_missing_dependency_but_not_timeout():
    traceback = "FAILED tests/test_c.py::test_new - ModuleNotFoundError: No module named 'x'"
    assert tc.classify_output_kind(traceback, has_node_failures=True) == ""
    assert tc.classify_output_kind(TIMEOUT_OUT + traceback,
                                   has_node_failures=True) == "timeout"


# ------------------------------------------------------- node ids and reruns
@pytest.mark.parametrize("node, ok", [
    ("tests/test_c.py::test_new", True),
    ("tests/test_c.py::TestA::test_b[1-2]", True),
    ("test_x.py::test_y", True),
    ("../secret.py::test_x", False),
    ("tests/../test_c.py::test_new", False),
    ("/abs/test_c.py::test_new", False),
    ("tests/test_c.py::test x", False),
    ("tests/test_c.py::t; rm -rf /", False),
    ("tests/test_c.py", False),
    ("node::name", False),
    ("", False),
])
def test_valid_node_id_accepts_only_safe_pytest_ids(node, ok):
    assert tc.valid_node_id(node) is ok


def test_rerun_command_uses_only_the_original_pytest_runner():
    assert tc.rerun_command("python -m pytest -q", "tests/test_c.py::test_new") \
        == "python -m pytest -q tests/test_c.py::test_new"
    assert tc.rerun_command("npm test", "tests/test_c.py::test_new") == ""
    assert tc.rerun_command("node --test", "node::name") == ""
    assert tc.rerun_command("cargo test", "tests/test_c.py::test_new") == ""
    assert tc.rerun_command("python -m unittest discover", "tests/test_c.py::t") == ""


def test_node_file_must_exist_inside_the_project(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_c.py").write_text("def test_new(): pass\n", encoding="utf-8")
    assert tc.node_file_inside(tmp_path, "tests/test_c.py::test_new")
    assert not tc.node_file_inside(tmp_path, "tests/missing.py::test_new")
    assert not tc.node_file_inside(tmp_path, "../elsewhere.py::test_x")


def test_impacted_base_is_the_runner_run_impacted_would_use(tmp_path):
    cfg = Config()
    cfg.test_command = "pytest -q"
    assert testsel.impacted_base(tmp_path, cfg, ["a.py"]) == "pytest -q"
    assert testsel.impacted_base(tmp_path, Config(), ["a.py"]) == ""


# ------------------------------------------------------------ classification
def _rerun(passed: bool, output: str) -> tc.NodeRerun:
    return tc.NodeRerun("tests/test_c.py::test_new", passed, output)


def test_passing_rerun_with_evidence_is_a_flake():
    verdict = tc.classify_test_failure(CURRENT, _rerun(True, PASS_RERUN), BASELINE)
    assert verdict.kind == "flake"
    assert verdict.node_id == "tests/test_c.py::test_new"
    assert verdict.baseline == "not_failing"
    assert verdict.rerun == "passed"


def test_same_assertion_on_rerun_is_a_real_failure():
    verdict = tc.classify_test_failure(CURRENT, _rerun(False, RERUN_SAME), BASELINE)
    assert verdict.kind == "real_failure"
    assert verdict.rerun == "failed"
    assert verdict.kind in tc.REPAIRABLE_KINDS


def test_changed_assertion_on_rerun_stays_unknown():
    changed = "FAILED tests/test_c.py::test_new - assert 9 == 1\n1 failed in 0.1s\n"
    verdict = tc.classify_test_failure(CURRENT, _rerun(False, changed), BASELINE)
    assert verdict.kind == "unknown"


def test_exit_zero_without_pass_evidence_is_not_a_flake():
    verdict = tc.classify_test_failure(CURRENT, _rerun(True, "no tests ran\n"), BASELINE)
    assert verdict.kind == "unknown"
    assert verdict.rerun == "inconclusive"


def test_no_rerun_stays_unknown():
    verdict = tc.classify_test_failure(NEW_ONLY, None, BASELINE)
    assert verdict.kind == "unknown"
    assert verdict.rerun == "not_run"


def test_failure_already_in_the_baseline_is_not_attributable():
    base = "FAILED tests/test_c.py::test_new - assert 5 == 6\n1 failed in 0.1s\n"
    verdict = tc.classify_test_failure(NEW_ONLY, _rerun(False, RERUN_SAME), base)
    assert verdict.kind == "unknown"
    assert verdict.baseline == "failed"
    assert "pre-existing" in verdict.reason


@pytest.mark.parametrize("initial", [TIMEOUT_OUT, COLLECTION_OUT])
def test_infrastructure_stops_are_never_reran(initial):
    verdict = tc.classify_test_failure(initial, None, BASELINE)
    assert verdict.kind in {"timeout", "collection_error"}
    assert verdict.rerun == "not_run"


def test_timeout_during_the_rerun_is_reported_as_timeout():
    verdict = tc.classify_test_failure(
        NEW_ONLY, _rerun(False, TIMEOUT_OUT), BASELINE)
    assert verdict.kind == "timeout"
    assert verdict.rerun == "infra"


def test_missing_dependency_without_node_failure_is_not_repairable():
    verdict = tc.classify_test_failure(
        "ModuleNotFoundError: No module named 'pygame'", None, "")
    assert verdict.kind == "missing_dependency"
    assert verdict.kind not in tc.REPAIRABLE_KINDS


def test_ambiguous_multiple_failures_without_a_rerun_target_stay_unknown():
    verdict = tc.classify_test_failure(CURRENT, None, BASELINE)
    assert verdict.kind == "unknown"
    assert verdict.node_id == ""


# ------------------------------------------------------------------ quarantine
def _write_flakes(root: Path, body) -> None:
    (root / ".pedacito").mkdir(exist_ok=True)
    text = body if isinstance(body, str) else json.dumps(body)
    (root / ".pedacito" / "flakes.json").write_text(text, encoding="utf-8")


def test_quarantine_loads_only_entries_with_an_unexpired_date(tmp_path):
    today = datetime.date(2026, 11, 27)
    _write_flakes(tmp_path, {"quarantine": [
        {"test_id": "tests/test_c.py::test_new", "reason": "flaky #12",
         "expires": "2027-01-01"},
        {"test_id": "tests/test_d.py::test_old", "reason": "old",
         "expires": "2026-01-01"},
        {"test_id": "tests/test_e.py::test_x", "reason": "no expiry"},
        {"test_id": "tests/test_f.py::test_y", "expires": "2027-01-01"},
    ]})
    active, warnings = tc.load_quarantine(tmp_path, today=today)
    assert set(active) == {"tests/test_c.py::test_new"}
    assert active["tests/test_c.py::test_new"].expires == "2027-01-01"
    assert len(warnings) == 3
    assert any("expired" in w for w in warnings)


def test_quarantine_file_absent_or_malformed_is_empty_with_a_warning(tmp_path):
    assert tc.load_quarantine(tmp_path) == ({}, ())
    _write_flakes(tmp_path, "{not json")
    active, warnings = tc.load_quarantine(tmp_path)
    assert active == {} and warnings and "unreadable" in warnings[0]
    _write_flakes(tmp_path, {"quarantine": "oops"})
    assert tc.load_quarantine(tmp_path)[0] == {}


def test_quarantine_file_that_escapes_the_project_is_refused(tmp_path):
    outside = tmp_path / "outside"
    project = tmp_path / "project"
    outside.mkdir()
    project.mkdir()
    (outside / "flakes.json").write_text(json.dumps({"quarantine": [
        {"test_id": "tests/test_c.py::test_new", "reason": "x",
         "expires": "2099-01-01"}]}), encoding="utf-8")
    (project / ".pedacito").mkdir()
    link = project / ".pedacito" / "flakes.json"
    try:
        link.symlink_to(outside / "flakes.json")
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable here: {exc}")
    active, warnings = tc.load_quarantine(project)
    assert active == {} and warnings


# ------------------------------------------------------------ events
class _Log:
    def __init__(self):
        self.calls = []

    def emit(self, kind, _full=False, session_id="", seq=0, **fields):
        self.calls.append((kind, fields))


def test_test_category_event_is_additive_and_bounded():
    log = _Log()
    events.emit_test_category(log, round_=1, category="flake",
                              node_id="tests/test_c.py::test_new" + "x" * 400,
                              baseline="not_failing", rerun="passed",
                              reason="r" * 500, quarantined=False)
    kind, fields = log.calls[0]
    assert kind == events.TEST_CATEGORY_KIND == "test_category"
    assert fields["category"] == "flake" and fields["round"] == 1
    assert len(fields["node_id"]) <= 300 and len(fields["reason"]) <= 300
    assert "kind" not in fields


# ---------------------------------------------------- loop routing (fakes)
def _cfg() -> Config:
    cfg = Config()
    cfg.verify_tests = True
    cfg.allow_shell = True
    cfg.repair_rounds = 1
    cfg.log_events = False
    cfg.static_gate = False
    cfg.frontend_contracts = False
    cfg.interface_lock = False
    cfg.test_command = "pytest -q"
    return cfg


class _Agent:
    def __init__(self):
        self.calls = []

    def _repair_from_findings(self, task, reply, errors, results):
        self.calls.append(list(errors))
        return "--- FILE: a.py\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\n"

    def _apply_once(self, reply, dry_run, backup_dir=None):
        return [_result("x = 1\n", "x = 2\n")]


def _project(tmp_path) -> None:
    (tmp_path / "tests").mkdir(exist_ok=True)
    (tmp_path / "tests" / "test_c.py").write_text(
        "def test_new():\n    assert 5 == 6\n", encoding="utf-8")


def _spy_rerun(monkeypatch, outputs):
    calls: list[str] = []

    def fake_run(root, cfg, command):
        calls.append(command)
        return outputs.pop(0)

    monkeypatch.setattr(testsel, "run_command", fake_run)
    return calls


def test_flake_is_not_repaired_and_the_run_stays_red(tmp_path, monkeypatch):
    _project(tmp_path)
    reruns = _spy_rerun(monkeypatch, [(True, PASS_RERUN)])
    monkeypatch.setattr(loop, "_run_tests", lambda *a: (False, CURRENT, "pytest -q"))
    agent, collect = _Agent(), []
    _res, _reply, tests_ok, preexisting = loop._post_edit_tests(
        tmp_path, _cfg(), agent, types.SimpleNamespace(root=tmp_path), "task",
        [_result("x = 1\n", "x = 1\n")], "reply", None,
        (False, BASELINE, "pytest -q"), ["a.py"], collect=collect)
    assert reruns == ["pytest -q tests/test_c.py::test_new"]
    assert agent.calls == [], "a flake must never reach model repair"
    assert tests_ok is False, "a flaky failure must not turn the run green"
    assert preexisting is False
    assert any("NOT verified" in f.message and "flake" in f.message for f in collect)
    assert all(f.severity == "error" for f in collect if f.kind == "tests")


def test_real_failure_is_repaired_only_with_its_own_packet(tmp_path, monkeypatch):
    _project(tmp_path)
    _spy_rerun(monkeypatch, [(False, RERUN_SAME)])
    runs = [(False, CURRENT, "pytest -q"), (True, "2 passed", "pytest -q")]
    monkeypatch.setattr(loop, "_run_tests", lambda *a: runs.pop(0))
    agent = _Agent()
    loop._post_edit_tests(
        tmp_path, _cfg(), agent, types.SimpleNamespace(root=tmp_path), "task",
        [_result("x = 1\n", "x = 1\n")], "reply", None,
        (False, BASELINE, "pytest -q"), ["a.py"])
    assert len(agent.calls) == 1
    packet = agent.calls[0][0].message
    assert "tests/test_c.py::test_new" in packet
    assert "tests/test_a.py::test_old" in packet and "still failing" in packet


def test_timeout_consumes_no_rerun_and_is_not_repaired(tmp_path, monkeypatch):
    _project(tmp_path)
    reruns = _spy_rerun(monkeypatch, [])
    monkeypatch.setattr(loop, "_run_tests", lambda *a: (False, TIMEOUT_OUT, "pytest -q"))
    agent, collect = _Agent(), []
    _res, _reply, tests_ok, preexisting = loop._post_edit_tests(
        tmp_path, _cfg(), agent, types.SimpleNamespace(root=tmp_path), "task",
        [_result("x = 1\n", "x = 1\n")], "reply", None,
        (False, BASELINE, "pytest -q"), ["a.py"], collect=collect)
    assert reruns == [] and agent.calls == []
    assert tests_ok is False and preexisting is False
    assert any("timeout" in f.message for f in collect)


def test_collection_error_is_not_rerun_as_a_node(tmp_path, monkeypatch):
    _project(tmp_path)
    reruns = _spy_rerun(monkeypatch, [])
    monkeypatch.setattr(loop, "_run_tests",
                        lambda *a: (False, COLLECTION_OUT, "pytest -q"))
    agent = _Agent()
    loop._post_edit_tests(
        tmp_path, _cfg(), agent, types.SimpleNamespace(root=tmp_path), "task",
        [_result("x = 1\n", "x = 1\n")], "reply", None,
        (False, BASELINE, "pytest -q"), ["a.py"])
    assert reruns == [] and agent.calls == []


def test_quarantined_failure_is_visible_but_neither_reran_nor_repaired(
        tmp_path, monkeypatch):
    _project(tmp_path)
    _write_flakes(tmp_path, {"quarantine": [
        {"test_id": "tests/test_c.py::test_new", "reason": "tracked in #12",
         "expires": "2099-01-01"}]})
    reruns = _spy_rerun(monkeypatch, [])
    monkeypatch.setattr(loop, "_run_tests", lambda *a: (False, CURRENT, "pytest -q"))
    agent, collect = _Agent(), []
    _res, _reply, tests_ok, _pre = loop._post_edit_tests(
        tmp_path, _cfg(), agent, types.SimpleNamespace(root=tmp_path), "task",
        [_result("x = 1\n", "x = 1\n")], "reply", None,
        (False, BASELINE, "pytest -q"), ["a.py"], collect=collect)
    assert reruns == [] and agent.calls == []
    assert tests_ok is False, "quarantine must never turn a red run green"
    assert any("quarantined" in f.message for f in collect)


def test_each_node_is_reran_at_most_once_across_repair_rounds(tmp_path, monkeypatch):
    _project(tmp_path)
    cfg = _cfg()
    cfg.repair_rounds = 3
    reruns = _spy_rerun(monkeypatch, [(False, RERUN_SAME)])
    runs = [(False, CURRENT, "pytest -q")] * 4
    monkeypatch.setattr(loop, "_run_tests", lambda *a: runs.pop(0))
    agent = _Agent()
    loop._post_edit_tests(
        tmp_path, cfg, agent, types.SimpleNamespace(root=tmp_path), "task",
        [_result("x = 1\n", "x = 1\n")], "reply", None,
        (False, BASELINE, "pytest -q"), ["a.py"])
    assert reruns == ["pytest -q tests/test_c.py::test_new"]
    assert len(agent.calls) >= 1


def test_non_pytest_runner_is_not_rerun_with_invented_syntax(tmp_path, monkeypatch):
    _project(tmp_path)
    cfg = _cfg()
    cfg.test_command = "npm test"
    reruns = _spy_rerun(monkeypatch, [])
    monkeypatch.setattr(loop, "_run_tests", lambda *a: (
        False, "✖ adds two numbers\n", "npm test"))
    agent = _Agent()
    loop._post_edit_tests(
        tmp_path, cfg, agent, types.SimpleNamespace(root=tmp_path), "task",
        [_result("x = 1\n", "x = 1\n")], "reply", None, None, ["a.py"])
    assert reruns == [] and agent.calls == []
