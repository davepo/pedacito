"""Regression: --verify-final must run the whole-goal command exactly once.

The plan loop runs it after the last task (feeding failures to repair); auto's
post-run verification pass then ran the same command a second time, doubling a
full test suite and making a "run once" flag observably run twice.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import cli, loop, scaffold
from pedacito.config import Config


def _args(plan: Path) -> SimpleNamespace:
    return SimpleNamespace(
        task=None, spec=None, plan=str(plan), refine=False, force=False,
        apply=True, verify="sh -c 'exit 0'", verify_final=True, repair=0,
        max_tasks=0, max_minutes=0, keep_going=True, diagnostics=False,
        worktree="", auto_install=False, strict=False)


def test_auto_skips_post_run_verify_when_final_already_ran(tmp_path,
                                                           monkeypatch):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal", ["one"])
    cfg = Config()
    cfg.auto_smoke = False
    cfg._verify_final_ran = True
    cfg._verify_final_ok = True
    called: list[str] = []

    monkeypatch.setattr(loop, "run_loop", lambda *a, **k: 0)
    monkeypatch.setattr(scaffold, "verify_and_fix",
                        lambda *a, **k: called.append(k.get("explicit_cmd"))
                        or (True, ""))

    code = cli._cmd_auto(_args(plan), cfg, None,
                         SimpleNamespace(root=tmp_path), tmp_path)
    assert code == 0
    assert called == []


def test_auto_still_repairs_a_red_final_check(tmp_path, monkeypatch):
    """A final check that ended RED keeps the post-run repair pass.

    The loop's "for the record" path (a run stopped by a cap before the last
    task) runs the whole-goal command WITHOUT repair and leaves repair and
    exit-code responsibility to auto's post-run pass. Skipping that pass on
    the ran-flag alone let a failed whole-goal check exit 0 unrepaired.
    """
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal", ["one"])
    cfg = Config()
    cfg.auto_smoke = False
    cfg._verify_final_ran = True
    cfg._verify_final_ok = False
    called: list[str] = []

    monkeypatch.setattr(loop, "run_loop", lambda *a, **k: 0)
    monkeypatch.setattr(scaffold, "verify_and_fix",
                        lambda *a, **k: called.append(k.get("explicit_cmd"))
                        or (False, "boom"))

    code = cli._cmd_auto(_args(plan), cfg, None,
                         SimpleNamespace(root=tmp_path), tmp_path)
    assert called == ["sh -c 'exit 0'"]
    assert code == 1


def test_auto_keeps_the_smoke_run_when_final_ran_green(tmp_path, monkeypatch):
    """--verify-final replaces the verify COMMAND, not the smoke run."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal", ["one"])
    cfg = Config()
    cfg.auto_smoke = True
    cfg._verify_final_ran = True
    cfg._verify_final_ok = True
    seen: list[dict] = []

    monkeypatch.setattr(loop, "run_loop", lambda *a, **k: 0)
    monkeypatch.setattr(
        scaffold, "verify_and_fix",
        lambda *a, **k: seen.append({"cmd": k.get("explicit_cmd"),
                                     "smoke": k.get("smoke")})
        or (True, ""))

    code = cli._cmd_auto(_args(plan), cfg, None,
                         SimpleNamespace(root=tmp_path), tmp_path)
    assert code == 0
    assert seen == [{"cmd": "", "smoke": True}]
