"""WP-4104C: index-backed plan validation and conservative impact.

Every test is offline: the index is built statically with no model server, and
the CLI path only loads or builds that index. The checks pinned here are the
parts a live Auto run depends on: containment, security, deliberate creations,
dependency order, indexed callers/tests, and the exact exit-code contract.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import cli, graph, index as index_mod, loop, plan, scaffold  # noqa: E402
from pedacito.config import Config  # noqa: E402


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_project(tmp_path: Path) -> tuple[Path, object]:
    root = tmp_path / "project"
    root.mkdir()
    _write(root, "pkg/__init__.py", "")
    _write(root, "pkg/core.py",
           "def helper():\n"
           "    return 1\n\n"
           "def main():\n"
           "    return helper()\n")
    _write(root, "pkg/use.py",
           "from pkg.core import helper\n\n"
           "def run():\n"
           "    return helper()\n")
    _write(root, "tests/test_core.py",
           "from pkg.core import helper\n\n"
           "def test_helper():\n"
           "    assert helper() == 1\n")
    idx = index_mod.build([str(root)], Config(), client=None, root=root,
                          summarise=False, verbose=False)
    return root, idx


def _plan(root: Path, text: str) -> Path:
    path = root / "PLAN.md"
    path.write_text("# Plan\n\n" + text.strip() + "\n", encoding="utf-8")
    return path


def _codes(issues) -> set[str]:
    return {issue.code for issue in issues}


def _severity(issues, code: str) -> str:
    return next(issue.severity for issue in issues if issue.code == code)


# --------------------------------------------------------------- contracts ---

def test_shared_dataclasses_are_frozen_and_keep_the_exact_fields():
    assert [f.name for f in dataclasses.fields(plan.PlanIssue)] == [
        "code", "severity", "task_index", "file", "symbol", "message"]
    assert [f.name for f in dataclasses.fields(plan.PlanImpact)] == [
        "files", "symbols", "callers", "tests", "unknowns"]
    issue = plan.PlanIssue("x", plan.WARNING, 1, "a.py", "", "m")
    impact = plan.PlanImpact((), (), (), (), ())
    with pytest.raises(dataclasses.FrozenInstanceError):
        issue.code = "y"       # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        impact.files = ("a.py",)  # type: ignore[misc]


# ------------------------------------------------------------ clean impact ---

def test_clean_plan_resolves_explicit_symbols_callers_and_tests(tmp_path):
    root, idx = _build_project(tmp_path)
    path = _plan(root, "- [ ] Update pkg/core.py: helper")
    issues, impact = plan.check_plan(root, path, idx)

    assert plan.blocking_issues(issues) == []
    assert plan.check_plan_exit_code(issues) == 0
    assert "pkg/core.py" in impact.files
    assert impact.symbols == ("helper",)
    assert "pkg/use.py" in impact.callers
    assert "tests/test_core.py" in impact.tests
    # Static edges are evidence, not a completeness proof.
    assert any("static import" in item for item in impact.unknowns)


def test_bare_backticked_symbol_contributes_its_files_and_callers(tmp_path):
    root, idx = _build_project(tmp_path)
    path = _plan(root, "- [ ] Refactor `helper` across the project")
    issues, impact = plan.check_plan(root, path, idx)
    assert plan.blocking_issues(issues) == []
    assert "pkg/core.py" in impact.files
    assert impact.symbols == ("helper",)
    assert "pkg/use.py" in impact.callers


def test_missing_unmarked_edit_is_definite_but_investigation_is_a_warning(
        tmp_path):
    root, idx = _build_project(tmp_path)
    path = _plan(root, "- [ ] Update missing.py with the helper")
    issues, _impact = plan.check_plan(root, path, idx)
    assert _severity(issues, "missing_file") == plan.ERROR
    assert plan.check_plan_exit_code(issues) == 1

    path = _plan(root, "- [ ] Inspect missing.py for the helper")
    issues, _impact = plan.check_plan(root, path, idx)
    assert _severity(issues, "missing_file") == plan.WARNING
    assert plan.check_plan_exit_code(issues) == 0


# --------------------------------------------------------- creations/paths ---

def test_deliberate_creation_is_valid_when_the_parent_is_contained(tmp_path):
    root, idx = _build_project(tmp_path)
    path = _plan(root, "- [ ] Create pkg/newmod.py with a build() helper")
    issues, impact = plan.check_plan(root, path, idx)
    assert plan.blocking_issues(issues) == []
    assert "pkg/newmod.py" in impact.files
    assert any("planned creation" in item for item in impact.unknowns)


def test_a_later_edit_of_an_earlier_creation_is_valid(tmp_path):
    root, idx = _build_project(tmp_path)
    path = _plan(root,
                 "- [ ] Create src/newmod.py with a helper()\n"
                 "- [ ] Update src/newmod.py: helper")
    issues, impact = plan.check_plan(root, path, idx)
    assert plan.blocking_issues(issues) == []
    assert "src/newmod.py" in impact.files
    assert any("created by an earlier task" in item for item in impact.unknowns)


def test_creation_parent_that_is_a_file_is_a_definite_error(tmp_path):
    root, idx = _build_project(tmp_path)
    _write(root, "blocker", "not a directory\n")
    path = _plan(root, "- [ ] Create blocker/new.py")
    issues, _impact = plan.check_plan(root, path, idx)
    assert "creation_parent_not_directory" in _codes(issues)
    assert plan.check_plan_exit_code(issues) == 1


def test_path_traversal_is_a_definite_error_and_never_touches_the_file(
        tmp_path):
    root, idx = _build_project(tmp_path)
    path = _plan(root, "- [ ] Update ../escape.py")
    issues, _impact = plan.check_plan(root, path, idx)
    assert "path_escape" in _codes(issues)
    assert plan.check_plan_exit_code(issues) == 1
    assert not (root.parent / "escape.py").exists()


def test_sensitive_and_protected_targets_are_definite_errors(tmp_path):
    root, idx = _build_project(tmp_path)
    path = _plan(root, "- [ ] Create .env\n- [ ] Update .git/config")
    issues, _impact = plan.check_plan(root, path, idx)
    assert "sensitive_path" in _codes(issues)
    assert "protected_path" in _codes(issues)
    assert plan.check_plan_exit_code(issues) == 1


def test_plan_path_must_be_contained_and_existing(tmp_path):
    root, idx = _build_project(tmp_path)
    outside = tmp_path / "outside.md"
    outside.write_text("- [ ] anything\n", encoding="utf-8")
    issues, impact = plan.check_plan(root, outside, idx)
    assert "invalid_plan_path" in _codes(issues)
    assert plan.check_plan_exit_code(issues) == 2
    assert impact.files == ()

    issues, _impact = plan.check_plan(root, root / "missing.md", idx)
    assert "invalid_plan_path" in _codes(issues)
    assert plan.check_plan_exit_code(issues) == 2


# ------------------------------------------------------------ order/cycles ---

def test_dependency_cycle_and_later_task_dependency_are_errors(tmp_path):
    root, idx = _build_project(tmp_path)
    path = _plan(root,
                 "- [ ] first (depends on 2)\n"
                 "- [ ] second [depends_on: 1]")
    issues, _impact = plan.check_plan(root, path, idx)
    assert "task_order" in _codes(issues)
    assert "task_dependency_cycle" in _codes(issues)
    assert plan.check_plan_exit_code(issues) == 1


def test_dependency_cycle_names_every_participating_task(tmp_path):
    root, idx = _build_project(tmp_path)
    # Two cycles share task 2; the finding must name the whole group, not
    # only one DFS back-edge (the QA report saw a partial "between tasks 2").
    path = _plan(root,
                 "- [ ] first (depends on 2)\n"
                 "- [ ] second (depends on 1, 3)\n"
                 "- [ ] third (depends on 2)")
    issues, _impact = plan.check_plan(root, path, idx)
    cycle = next(i for i in issues if i.code == "task_dependency_cycle")
    assert cycle.message == "dependency cycle between tasks 1, 2, 3"


def test_a_two_task_cycle_names_both_tasks(tmp_path):
    root, idx = _build_project(tmp_path)
    path = _plan(root,
                 "- [ ] first\n"
                 "- [ ] second (depends on 3)\n"
                 "- [ ] third (depends on 2)")
    issues, _impact = plan.check_plan(root, path, idx)
    cycle = next(i for i in issues if i.code == "task_dependency_cycle")
    assert cycle.message == "dependency cycle between tasks 2 and 3"


def test_missing_dependency_number_is_an_error(tmp_path):
    root, idx = _build_project(tmp_path)
    path = _plan(root, "- [ ] only task depends_on: 3")
    issues, _impact = plan.check_plan(root, path, idx)
    assert "task_dependency_missing" in _codes(issues)
    assert plan.check_plan_exit_code(issues) == 1


def test_import_cycle_helper_is_deterministic_and_a_warning_in_a_plan(tmp_path):
    root = tmp_path / "cycle"
    root.mkdir()
    _write(root, "pkg/__init__.py", "")
    _write(root, "pkg/a.py", "from pkg import b\n\ndef a():\n    return 1\n")
    _write(root, "pkg/b.py", "from pkg import a\n\ndef b():\n    return 2\n")
    idx = index_mod.build([str(root)], Config(), client=None, root=root,
                          summarise=False, verbose=False)
    assert idx.graph.import_cycles() == [["pkg/a.py", "pkg/b.py"]]
    assert graph.Graph.import_cycles(idx.graph) == [["pkg/a.py", "pkg/b.py"]]
    path = _plan(root, "- [ ] Update pkg/a.py: a")
    issues, _impact = plan.check_plan(root, path, idx)
    assert _severity(issues, "import_cycle") == plan.WARNING
    assert plan.check_plan_exit_code(issues) == 0


# -------------------------------------------------------- dynamic/unknowns ---

def test_dynamic_or_unlinked_evidence_is_unknown_not_a_failure(tmp_path):
    root = tmp_path / "dynamic"
    root.mkdir()
    _write(root, "dyn.py", "def run():\n    return 1\n")
    idx = index_mod.build([str(root)], Config(), client=None, root=root,
                          summarise=False, verbose=False)
    path = _plan(root, "- [ ] Update dyn.py: run")
    issues, impact = plan.check_plan(root, path, idx)
    assert plan.blocking_issues(issues) == []
    assert impact.symbols == ("run",)
    assert any("dynamic imports" in item for item in impact.unknowns)
    assert any("no indexed test" in item for item in impact.unknowns)


def test_dynamic_python_symbol_is_unknown_not_a_definite_error(tmp_path):
    root, _idx = _build_project(tmp_path)
    _write(root, "pkg/dynamic.py",
           "def __getattr__(name):\n"
           "    return lambda: name\n")
    idx = index_mod.build([str(root)], Config(), client=None, root=root,
                          summarise=False, verbose=False)
    path = _plan(root, "- [ ] Update pkg/dynamic.py: generated")
    issues, _impact = plan.check_plan(root, path, idx)
    assert _severity(issues, "missing_symbol") == plan.WARNING
    assert plan.check_plan_exit_code(issues) == 0


def test_missing_python_symbol_is_definite_but_creation_marker_softens_it(
        tmp_path):
    root, idx = _build_project(tmp_path)
    path = _plan(root, "- [ ] Update pkg/core.py: no_such_symbol")
    issues, _impact = plan.check_plan(root, path, idx)
    assert _severity(issues, "missing_symbol") == plan.ERROR
    assert plan.check_plan_exit_code(issues) == 1

    path = _plan(root, "- [ ] Add pkg/core.py: new_symbol")
    issues, _impact = plan.check_plan(root, path, idx)
    assert _severity(issues, "missing_symbol") == plan.WARNING
    assert plan.check_plan_exit_code(issues) == 0


# ----------------------------------------------------- case-insensitive OS ---

def test_case_only_collision_is_an_error_on_windows_and_a_warning_elsewhere(
        tmp_path, monkeypatch):
    root, idx = _build_project(tmp_path)
    _write(root, "README.md", "# readme\n")
    # Rebuild so the index knows the exact casing.
    idx = index_mod.build([str(root)], Config(), client=None, root=root,
                          summarise=False, verbose=False)
    path = _plan(root, "- [ ] Create readme.md")

    monkeypatch.setattr(plan, "_is_windows", lambda: True)
    issues, _impact = plan.check_plan(root, path, idx)
    assert _severity(issues, "case_collision") == plan.ERROR
    assert plan.check_plan_exit_code(issues) == 1

    monkeypatch.setattr(plan, "_is_windows", lambda: False)
    issues, impact = plan.check_plan(root, path, idx)
    assert _severity(issues, "case_collision") == plan.WARNING
    assert plan.check_plan_exit_code(issues) == 0
    assert "readme.md" in impact.files


# ------------------------------------------------------------- execution ----

def test_run_loop_stops_on_a_definite_plan_error_before_any_task(
        tmp_path, monkeypatch):
    root, idx = _build_project(tmp_path)
    path = _plan(root,
                 "- [ ] first (depends on 2)\n"
                 "- [ ] second (depends on 1)")
    calls = []
    monkeypatch.setattr(loop, "run_task",
                        lambda *a, **k: calls.append(1))
    cfg = Config()
    cfg.auto_green_ratchet = False
    cfg.backup = False
    code = loop.run_loop(root, cfg, None, idx, path, apply=True)
    assert code == 1
    assert calls == []


def test_run_loop_proceeds_through_a_warning_plan(tmp_path, monkeypatch):
    root, idx = _build_project(tmp_path)
    path = _plan(root, "- [ ] Mention the docs/guide.md file")
    calls = []

    def fake_run_task(root_, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None, is_repair=False):
        calls.append(task)
        return loop.TaskResult(
            [types.SimpleNamespace(ok=True, file="docs/guide.md")],
            "done", True, ["docs/guide.md"], idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    cfg = Config()
    cfg.auto_green_ratchet = False
    cfg.backup = False
    code = loop.run_loop(root, cfg, None, idx, path, apply=True)
    assert code == 0
    assert len(calls) == 1


def test_gate_plan_returns_the_exit_code_contract(tmp_path):
    root, idx = _build_project(tmp_path)
    bad = _plan(root, "- [ ] Update missing.py")
    assert plan.gate_plan(root, bad, idx) == 1
    outside = tmp_path / "outside.md"
    outside.write_text("- [ ] anything\n", encoding="utf-8")
    assert plan.gate_plan(root, outside, idx) == 2
    assert plan.gate_plan(root, _plan(root, "- [ ] Update pkg/core.py: helper"),
                          idx) is None


def test_scaffold_complex_gate_stops_before_run_loop(tmp_path, monkeypatch):
    root = tmp_path / "newproject"
    root.mkdir()
    plan_obj = scaffold.Plan("demo", "a demo",
                             [scaffold.PlannedFile("main.py", "entry")],
                             spec=scaffold.Spec("a demo", name_hint="demo"))
    spec = plan_obj.spec
    cfg = Config()
    cfg.auto_plan_file = "PLAN.md"
    calls = []
    monkeypatch.setattr(
        plan, "check_plan",
        lambda *a, **k: ([plan.PlanIssue("blocked", plan.ERROR, 1, "main.py",
                                         "", "blocked for the test")],
                         plan.PlanImpact((), (), (), (), ())))
    monkeypatch.setattr(loop, "run_loop",
                        lambda *a, **k: calls.append(1) or 0)
    code = scaffold._run_complex(root, cfg, None, None, plan_obj, spec,
                                 verbose=False)
    assert code == 1
    assert calls == []


# -------------------------------------------------------------------- CLI ---

def _cli(argv: list[str], capsys) -> tuple[int, str, str]:
    code = cli.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_cli_plan_check_clean_errors_and_invalid_exit_codes(tmp_path, capsys):
    root, _idx = _build_project(tmp_path)
    _plan(root, "- [ ] Update pkg/core.py: helper")
    code, out, _err = _cli(["plan", "--check", "PLAN.md", str(root),
                            "--json", "--lang", "en"], capsys)
    assert code == 0
    payload = json.loads(out)
    assert payload["exit_code"] == 0 and payload["issues"] == []
    assert "pkg/core.py" in payload["impact"]["files"]

    _plan(root, "- [ ] Update missing.py")
    code, _out, _err = _cli(["plan", "--check", "PLAN.md", str(root),
                             "--lang", "en"], capsys)
    assert code == 1

    outside = tmp_path / "outside.md"
    outside.write_text("- [ ] anything\n", encoding="utf-8")
    code, _out, _err = _cli(["plan", "--check", str(outside), str(root),
                             "--lang", "en"], capsys)
    assert code == 2


def test_cli_impact_prints_the_conservative_preview(tmp_path, capsys):
    root, _idx = _build_project(tmp_path)
    _plan(root, "- [ ] Update pkg/core.py: helper")
    code, out, _err = _cli(["plan", "--check", "PLAN.md", str(root),
                            "--impact", "--lang", "en"], capsys)
    assert code == 0
    assert "Impact (conservative" in out
    assert "pkg/use.py" in out
    assert "tests/test_core.py" in out
