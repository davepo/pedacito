"""Immediate syntax gate and failure-identity repair packets."""
from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import loop, verify
from pedacito.chunker_ts import TS_AVAILABLE
from pedacito.config import Config
from pedacito.edits import Result


def _result(before: str, after: str, file: str = "mod.py") -> Result:
    return Result(True, file, "edit applied", before=before, after=after)


# ------------------------------------------------------------ syntax gate
def test_syntax_gate_rejects_a_new_python_error_with_ten_lines():
    before = "\n".join(f"x{i} = {i}" for i in range(20)) + "\n"
    after = before.replace("x9 = 9", "x9 = (")
    findings = verify.check_syntax([_result(before, after)])

    assert len(findings) == 1
    finding = findings[0]
    assert finding.kind == "syntax" and finding.severity == "error"
    assert finding.file == "mod.py" and finding.line == 10
    assert "ast.parse:" in finding.message
    assert "lines 5-14:" in finding.message

    window = [ln for ln in finding.message.splitlines()
              if re.match(r"^(?:>>|  )\s*\d+\|", ln)]
    assert len(window) == 10
    assert any(">>" in ln and "x9 = (" in ln for ln in window)


def test_syntax_gate_ignores_valid_edits_and_already_broken_families():
    before = "\n".join(f"x{i} = {i}" for i in range(20)) + "\n"
    broken = before.replace("x9 = 9", "x9 = (")
    assert verify.check_syntax([_result(before, before + "x20 = 20\n")]) == []
    # A file that was already unparseable may still be edited (recovery path).
    assert verify.check_syntax([_result(broken, broken + "# still broken\n")]) == []


@pytest.mark.skipif(not TS_AVAILABLE, reason="tree-sitter is not installed")
def test_syntax_gate_uses_tree_sitter_for_other_languages():
    before = "function ok() {\n  return 1;\n}\n"
    after = "function ok() {\n  return 1;\n"
    findings = verify.check_syntax([_result(before, after, "app.js")])

    assert len(findings) == 1
    assert findings[0].kind == "syntax"
    assert "tree-sitter javascript" in findings[0].message
    assert "return 1;" in findings[0].message


def test_post_edit_syntax_gate_skips_subprocess_until_the_patch_parses(
        tmp_path, monkeypatch):
    cfg = Config()
    cfg.verify_tests = True
    cfg.allow_shell = True
    cfg.repair_rounds = 1
    cfg.log_events = False

    before = "\n".join(f"x{i} = {i}" for i in range(20)) + "\n"
    broken = before.replace("x9 = 9", "x9 = (")
    calls = []

    def fake_run_tests(idx, cfg_, files):
        calls.append(list(files))
        return True, "1 passed", "pytest"

    monkeypatch.setattr(loop, "_run_tests", fake_run_tests)

    class FakeAgent:
        errors = None

        def _repair_from_findings(self, task, reply, errors, results):
            self.errors = list(errors)
            return "--- FILE: mod.py\n<<<<<<< SEARCH\nx9 = (\n=======\nx9 = 9\n>>>>>>> REPLACE\n"

        def _apply_once(self, reply, dry_run, backup_dir=None):
            return [_result(before, before)]

    agent = FakeAgent()
    results, reply, tests_ok, _preexisting = loop._post_edit_tests(
        tmp_path, cfg, agent, object(), "fix the syntax", [_result(before, broken)],
        "old reply", None, None, ["mod.py"])

    assert agent.errors and agent.errors[0].kind == "syntax"
    assert "lines 5-14:" in agent.errors[0].message
    assert calls == [["mod.py"]], "the subprocess ran only after the fix parsed"
    assert tests_ok is True
    assert all(r.after != broken for r in results)


# ------------------------------------------------- failure identity diffing
BASELINE = textwrap.dedent("""\
    =========================== short test summary info ============================
    FAILED tests/test_a.py::test_old - assert 1 == 2
    FAILED tests/test_b.py::test_same - assert 3 == 4
    2 failed in 0.10s
    """)


CURRENT = textwrap.dedent("""\
    =========================== short test summary info ============================
    FAILED tests/test_b.py::test_same - assert 3 == 4
    FAILED tests/test_c.py::test_new - assert 5 == 6
    2 failed in 0.10s
    """)


def test_failure_identity_diff_sees_one_fixed_and_one_new_at_equal_counts():
    assert loop._failure_score(BASELINE) == loop._failure_score(CURRENT) == 2
    diff = loop.classify_failures(BASELINE, CURRENT)

    assert [f.test_id for f in diff.new] == ["tests/test_c.py::test_new"]
    assert [f.test_id for f in diff.fixed] == ["tests/test_a.py::test_old"]
    assert [f.test_id for f in diff.still_failing] == [
        "tests/test_b.py::test_same"]
    assert {f.test_id for f in diff.pre_existing} == {
        "tests/test_a.py::test_old", "tests/test_b.py::test_same"}
    assert diff.has_new


def test_failure_signature_is_normalised_across_run_specific_details():
    first = ("FAILED tests/test_x.py::test_x - AssertionError: assert "
             "0xdeadbeef == 1 at /tmp/run-1/app.py:12")
    second = ("FAILED tests/test_x.py::test_x - AssertionError: assert "
              "0xfeedface == 1 at /tmp/run-2/app.py:99")
    a = loop.failure_identities(first)["tests/test_x.py::test_x"].signature
    b = loop.failure_identities(second)["tests/test_x.py::test_x"].signature
    assert a == b
    assert "0x<addr>" in a and "app.py:12" not in a


def test_traceback_packet_quotes_project_frames_and_drops_venv(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "app" / "calc.py").write_text(
        "def divide(a, b):\n    return a / b\n", encoding="utf-8")
    (tmp_path / "tests" / "test_calc.py").write_text(
        "from app.calc import divide\n\n\ndef test_divide():\n"
        "    assert divide(2, 0) == 1\n", encoding="utf-8")
    venv = tmp_path / ".venv" / "lib" / "python3.14" / "site-packages"
    venv.mkdir(parents=True)
    (venv / "dep.py").write_text("def helper():\n    return 0\n",
                                 encoding="utf-8")

    output = textwrap.dedent(f"""\
        =================================== FAILURES ===================================
        __________________________________ test_divide __________________________________
        Traceback (most recent call last):
          File "{venv / 'dep.py'}", line 2, in helper
          File "{tmp_path / 'app' / 'calc.py'}", line 2, in divide
            return a / b
        tests/test_calc.py:4: in test_divide
            assert divide(2, 0) == 1
        E   ZeroDivisionError: division by zero
        tests/test_calc.py:4: ZeroDivisionError
        =========================== short test summary info ============================
        FAILED tests/test_calc.py::test_divide - ZeroDivisionError: division by zero
        1 failed in 0.10s
        """)
    diff = loop.classify_failures("", output)
    packet = loop.traceback_repair_packet(
        tmp_path, diff, current=loop.failure_identities(output))

    assert "TRACEBACK REPAIR PACKET" in packet
    assert "tests/test_calc.py::test_divide" in packet
    assert "ZeroDivisionError: division by zero" in packet
    assert "ASSERTION WINDOW (tests/test_calc.py:" in packet
    assert "PROJECT TRACEBACK FRAME (app/calc.py:2, in divide)" in packet
    assert "return a / b" in packet
    assert "site-packages" not in packet and ".venv" not in packet


# ------------------------------------------ loop integration: flat count repairs
def test_post_edit_tests_repairs_new_failure_even_when_total_is_flat(
        tmp_path, monkeypatch):
    cfg = Config()
    cfg.verify_tests = True
    cfg.allow_shell = True
    cfg.repair_rounds = 1
    cfg.log_events = False

    runs = [(False, CURRENT, "pytest"), (True, "2 passed", "pytest")]
    monkeypatch.setattr(loop, "_run_tests",
                        lambda idx, cfg_, files: runs.pop(0))

    class FakeAgent:
        errors = None

        def _repair_from_findings(self, task, reply, errors, results):
            self.errors = list(errors)
            return "--- FILE: a.py\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\n"

        def _apply_once(self, reply, dry_run, backup_dir=None):
            return [_result("x = 1\n", "x = 2\n", "a.py")]

    agent = FakeAgent()
    results = [_result("x = 1\n", "x = 1\n", "a.py")]
    out, reply, tests_ok, _preexisting = loop._post_edit_tests(
        tmp_path, cfg, agent, object(), "make tests pass", results, "old reply",
        None, (False, BASELINE, "pytest"), ["a.py"])

    assert tests_ok is True
    assert agent.errors, "a new failure must trigger repair at an equal count"
    packet = agent.errors[0].message
    assert "these tests fail (they did not before)" in packet
    assert "TRACEBACK REPAIR PACKET" in packet
    assert "tests/test_c.py::test_new" in packet
    assert "tests/test_a.py::test_old" in packet


def test_post_edit_suppresses_repair_for_a_recorded_pre_existing_failure(
        tmp_path, monkeypatch):
    """R2: with no in-run baseline (the recovery path), the cross-run command
    history is the only baseline there is. When run_impacted labelled the
    failure pre-existing, the repair loop must not chase it."""
    import types

    from pedacito import testsel

    cfg = Config()
    cfg.verify_tests = True
    cfg.allow_shell = True
    cfg.repair_rounds = 2
    cfg.log_events = False
    cfg.static_gate = False
    cfg.frontend_contracts = False
    cfg.interface_lock = False

    failing = ("FAILED tests/test_x.py::test_a - assert 0\n"
               "1 failed in 0.10s")
    desc = f"`pytest -q` -- {testsel.PRE_EXISTING_LABEL}"
    runs = []
    monkeypatch.setattr(loop, "_run_tests",
                        lambda idx, cfg_, files: (runs.append(1)
                                                  or (False, failing, desc)))

    class FakeAgent:
        def __init__(self):
            self.calls = []

        def _repair_from_findings(self, task, reply, errors, results):
            self.calls.append([e.kind for e in errors])
            return "REPAIR"

        def _apply_once(self, reply, dry_run, backup_dir=None):
            return []

    agent = FakeAgent()
    results = [_result("x = 1\n", "x = 2\n", "a.py")]
    out, reply, tests_ok, _preexisting = loop._post_edit_tests(
        tmp_path, cfg, agent, types.SimpleNamespace(root=tmp_path),
        "task", results, "reply", None, None, ["a.py"])

    assert tests_ok is False             # the suite is honestly still red
    assert agent.calls == []             # ...but the failure pre-dates the edit
    assert len(runs) == 1                # no repair round re-ran the tests


def test_post_edit_still_repairs_when_an_in_run_baseline_shows_a_new_failure(
        tmp_path, monkeypatch):
    """R2 guard-rail: an in-run baseline stays authoritative. Even if the
    description carries the pre-existing label, a baseline diff that proves a
    NEW failure must still repair (the label only suppresses when there is no
    baseline to diff against)."""
    import types

    from pedacito import testsel

    cfg = Config()
    cfg.verify_tests = True
    cfg.allow_shell = True
    cfg.repair_rounds = 1
    cfg.log_events = False
    cfg.static_gate = False
    cfg.frontend_contracts = False
    cfg.interface_lock = False

    desc = f"`pytest -q` -- {testsel.PRE_EXISTING_LABEL}"
    runs = [(False, CURRENT, desc), (True, "2 passed", desc)]
    monkeypatch.setattr(loop, "_run_tests",
                        lambda idx, cfg_, files: runs.pop(0))

    class FakeAgent:
        def __init__(self):
            self.errors = None

        def _repair_from_findings(self, task, reply, errors, results):
            self.errors = list(errors)
            return "REPAIR"

        def _apply_once(self, reply, dry_run, backup_dir=None):
            return [_result("x = 1\n", "x = 2\n", "a.py")]

    agent = FakeAgent()
    results = [_result("x = 1\n", "x = 1\n", "a.py")]
    out, reply, tests_ok, _preexisting = loop._post_edit_tests(
        tmp_path, cfg, agent, types.SimpleNamespace(root=tmp_path),
        "task", results, "reply", None, (False, BASELINE, "pytest"), ["a.py"])

    assert tests_ok is True
    assert agent.errors, "the baseline diff proves a new failure: repair it"
