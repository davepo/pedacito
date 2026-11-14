"""Regression tests for the tenth bug-hunting report (Bug Hunt #10, Nov 2026).

Each section pins the deterministic half of one reported finding found while
driving Pedacito as a real user on a polyglot Python + ES-module chiptune
tracker build. See /mnt/hdd/projects/testing/bug_hunting_10/BUG_HUNT_10_REPORT.md
for the symptom each fix addresses.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import loop  # noqa: E402
from pedacito.config import Config  # noqa: E402


def _cfg(**kw) -> Config:
    c = Config()
    c.log_sessions = False
    c.log_events = False
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# -- BUG-1002: a local module name is not a pip dependency -------------------

_COLLECT_ERROR = (
    "==================== ERRORS ====================\n"
    "___________ ERROR collecting tests/test_synth.py ___________\n"
    "ImportError while importing test module "
    "'/work/tests/test_synth.py'.\n"
    "tests/test_synth.py:1: in <module>\n"
    "    from pattern_store import PatternBank\n"
    "E   ModuleNotFoundError: No module named 'pattern_store'\n")


def test_missing_modules_reports_local_name_without_root(tmp_path):
    """Old behaviour is still the no-root contract (scaffold classification
    tests rely on it)."""
    (tmp_path / "pattern_store.py").write_text("x = 1\n", encoding="utf-8")
    assert loop.missing_modules(_COLLECT_ERROR) == ["pattern_store"]


def test_missing_modules_excludes_project_source(tmp_path):
    """BUG-1002: `auto --verify pytest -q` on a freshly built project printed
    'Install with pip install pattern_store' for the module the build itself
    had just written, then refused to repair. With a root, a name that
    resolves to project source is a broken import path -- code repair, not
    a dependency (and with --auto-install, a dependency-squatting vector)."""
    (tmp_path / "pattern_store.py").write_text("x = 1\n", encoding="utf-8")
    pkg = tmp_path / "mylib"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("y = 2\n", encoding="utf-8")
    out = _COLLECT_ERROR + ("E   ModuleNotFoundError: No module named "
                            "'mylib.core'\n"
                            "E   ModuleNotFoundError: No module named "
                            "'requests'\n")
    assert loop.missing_modules(out, root=tmp_path) == ["requests"]


def test_missing_modules_local_package_dir_needs_init(tmp_path):
    """A bare directory without __init__.py is not an importable module, so
    the name may legitimately be a namespace package from elsewhere."""
    (tmp_path / "data").mkdir()
    out = "E   ModuleNotFoundError: No module named 'data'\n"
    assert loop.missing_modules(out, root=tmp_path) == ["data"]
    (tmp_path / "data" / "__init__.py").write_text("", encoding="utf-8")
    assert loop.missing_modules(out, root=tmp_path) == []


# -- BUG-1003: the enablement message must describe what will happen ----------


def test_infer_task_checks_honest_without_shell(tmp_path, capsys):
    """verify_tests is ANDed with allow_shell at every consumer, so on a
    shell-less run the assignment enables nothing -- the harness used to
    print "enabling verify_tests for this run" and then retract it with
    "verify_tests/allow_shell is off"."""
    cfg = _cfg(allow_shell=False, verify_tests=False, verify_dom=False)
    loop._infer_task_checks(tmp_path, cfg, "make pytest pass")
    loop._infer_task_checks(tmp_path, cfg, "run the domcheck smoke suite")
    err = capsys.readouterr().err
    assert cfg.verify_tests and cfg.verify_dom   # the flags themselves flip
    assert "enabling verify_tests for this run" not in err
    assert "enabling verify_dom for this run" not in err
    assert "needs allow_shell" in err


def test_infer_task_checks_announce_when_effective(tmp_path, capsys):
    cfg = _cfg(allow_shell=True, verify_tests=False, verify_dom=False)
    loop._infer_task_checks(tmp_path, cfg, "make pytest pass")
    err = capsys.readouterr().err
    assert "enabling verify_tests for this run" in err


# -- BUG-1004: `ask --verify-dom` alone must arm the DOM gate -----------------


def test_dom_wanted_needs_boundary_or_user_consent():
    """README/QUICKSTART both document `ask --verify-dom` as sufficient
    ("set verify_dom and allow_shell, or ask --verify-dom"), but _dom_wanted
    required allow_shell even for the explicit per-run flag, so the gate the
    user paid for silently never ran."""
    cfg = _cfg(verify_dom=True, allow_shell=False)
    assert not loop._dom_wanted(cfg)            # config-only: boundary holds
    cfg._dom_user_consent = True                # what `ask --verify-dom` sets
    assert loop._dom_wanted(cfg)                # explicit flag: gate armed
    cfg2 = _cfg(verify_dom=False, allow_shell=True, verify_tests=True)
    assert not loop._dom_wanted(cfg2)           # consent without want: off


# -- BUG-1005: a dep-blocked verify failure still rolls the step back ---------


def test_run_loop_dep_blocked_verify_rolls_back_ratchet(tmp_path, monkeypatch,
                                                        capsys):
    """The green ratchet (#20) promises: a step that fails its check is undone
    to the last verified-green state. But `auto` treated a verify failure
    blocked by a missing third-party package as a special early exit that
    skipped undo_step entirely -- the failed step's files stayed in the tree
    unverified, and a resumed run built on exactly the wreckage the ratchet
    exists to prevent. (Observed live: task 1 of the chiptune build was
    dep-blocked and left pattern_store.py plus a repair-invented test file on
    disk with no ROLLED BACK telemetry.)"""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "build it", ["Add helper module",
                                       "Add second module"])
    idx = types.SimpleNamespace(root=tmp_path)

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None, is_repair=False):
        (tmp_path / "helper.py").write_text("import requests\n",
                                            encoding="utf-8")
        return loop.TaskResult(
            [types.SimpleNamespace(ok=True, file="helper.py")],
            "reply", True, ["helper.py"], idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    monkeypatch.setattr(
        loop, "run_verify",
        lambda root, cmd, *a, **kw: (
            False,
            "E   ModuleNotFoundError: No module named 'requests'"))

    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         verify="pytest -q", repair_rounds=0)
    err = capsys.readouterr().err
    assert code == 1
    assert "verification blocked by missing dependencies: requests" in err
    # The failed step is undone, not abandoned: the file it created is gone
    # and its consumed checkpoint left no restore point behind.
    assert not (tmp_path / "helper.py").exists()
    assert not (tmp_path / ".pedacito" / "checkpoints" / "step_1").exists()
