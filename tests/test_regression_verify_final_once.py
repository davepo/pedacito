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
    called: list[str] = []

    monkeypatch.setattr(loop, "run_loop", lambda *a, **k: 0)
    monkeypatch.setattr(scaffold, "verify_and_fix",
                        lambda *a, **k: called.append(k.get("explicit_cmd"))
                        or (True, ""))

    code = cli._cmd_auto(_args(plan), cfg, None,
                         SimpleNamespace(root=tmp_path), tmp_path)
    assert code == 0
    assert called == []
