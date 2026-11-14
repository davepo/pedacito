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


# -- BUG-1006: a user-granted test command is consent to RUN it ---------------


def test_test_run_consent_from_granted_commands_only():
    """allow_shell remains the boundary for commands inferred from TASK
    text; but a test command the user granted -- `test_command`, or the
    `--verify` stamp the CLI sets -- is consent too. The chiptune run proved
    the old reading wrong: `auto --apply --verify "pytest -q"` executed that
    pytest from the step gate and recorded it in the ledger, while the same
    ladder printed "no test evidence for it" minutes earlier."""
    assert not loop._test_run_consent(_cfg(allow_shell=False))
    assert loop._test_run_consent(_cfg(test_command="pytest -q"))
    assert loop._test_run_consent(_cfg(test_command="  ")) is False
    c = _cfg()
    c._tests_user_consent = True          # what `auto --verify pytest` sets
    assert loop._test_run_consent(c)


def test_infer_task_checks_enablement_with_granted_test_command(
        tmp_path, capsys):
    cfg = _cfg(allow_shell=False, verify_tests=False,
               test_command="pytest -q")
    loop._infer_task_checks(tmp_path, cfg, "make pytest pass")
    err = capsys.readouterr().err
    assert "enabling verify_tests for this run" in err
    assert "needs allow_shell" not in err


def test_post_edit_tests_runs_granted_suite_without_allow_shell(
        tmp_path, monkeypatch):
    from pedacito import testsel

    calls: list = []

    def fake_impacted(index, cfg, changed):
        calls.append(testsel.detect_test_command(index.root, cfg))
        return True, "1 passed in 0.01s", "`pytest -q` (fake)"

    monkeypatch.setattr(testsel, "detect_test_command",
                        lambda root, cfg=None: "pytest -q")
    monkeypatch.setattr(testsel, "run_impacted", fake_impacted)
    agent = types.SimpleNamespace(_repair_from_findings=lambda *a, **k: "")

    refused = loop._post_edit_tests(
        tmp_path, _cfg(verify_tests=True, allow_shell=False), agent,
        types.SimpleNamespace(root=tmp_path), "make pytest pass",
        [], "", None, None, [])
    assert calls == []                             # boundary still holds
    assert refused[2] is None                      # no test verdict

    granted = loop._post_edit_tests(
        tmp_path, _cfg(verify_tests=True, allow_shell=False,
                       test_command="pytest -q"), agent,
        types.SimpleNamespace(root=tmp_path), "make pytest pass",
        [], "", None, None, [])
    assert calls == ["pytest -q"]                  # user command: consented
    assert granted[2] is True                      # ... and it has a verdict


def test_auto_cli_stamps_test_consent(tmp_path, monkeypatch):
    """`pedacito auto --verify "pytest -q"` must reach the loop with the
    consent stamp; a `--verify` that names no test command must not smuggle
    any (that would make `--verify "rm -rf"` imply test consent)."""
    from pedacito import cli

    (tmp_path / "PLAN.md").write_text("Goal\n\n- [ ] Add helper\n",
                                      encoding="utf-8")
    seen: dict = {}

    def spy(root, cfg, *a, **kw):
        seen["consent"] = getattr(cfg, "_tests_user_consent", False)
        return 0

    monkeypatch.setattr(loop, "run_loop", spy)
    from pedacito import scaffold
    # The post-run smoke/verify stage drives the model; this test pins the
    # consent stamp that reaches the loop, so stub the stage out.
    monkeypatch.setattr(scaffold, "verify_and_fix",
                        lambda *a, **k: (True, ""))
    idx = types.SimpleNamespace(root=tmp_path)
    args = types.SimpleNamespace(
        task=None, spec=None, plan=None, force=False, apply=True,
        refine=False, verify="pytest -q", verify_final=False, repair=None,
        max_tasks=None, max_minutes=None, auto_install=False,
        keep_going=False, diagnostics=False, worktree="")
    cli._cmd_auto(args, _cfg(), None, idx, tmp_path)
    assert seen.get("consent") is True

    seen.clear()
    args.verify = "make build"
    cli._cmd_auto(args, _cfg(), None, idx, tmp_path)
    assert seen.get("consent", False) is False


# -- BUG-1007: a run's own fresh failures are not a baseline ------------------


def test_command_was_failing_scoped_to_run_start(tmp_path):
    import time as _time

    from pedacito import memory
    cmd = "make test"
    run_start = _time.time()
    memory.observe_command(tmp_path, cmd, False, _cfg(micro_facts=True),
                           kind="test")
    # The red outcome was recorded DURING this run: not pre-existing.
    assert not memory.command_was_failing(tmp_path, cmd, before=run_start)
    # The next run legitimately inherits it as a baseline.
    assert memory.command_was_failing(tmp_path, cmd,
                                      before=_time.time() + 1)
    # Unscoped reads keep the old contract (eval, legacy callers).
    assert memory.command_was_failing(tmp_path, cmd)


def test_second_failure_in_same_run_not_laundered(tmp_path, monkeypatch):
    """Chiptune run 3, task 1: attempt 2 ran the worker-invented `make test`
    once (red), then called the SAME red outcome "pre-existing (recorded
    before this run)" and skipped repairing it -- the task's own failure
    laundered into a baseline by the run's own ledger write."""
    import time as _time

    from pedacito import memory, testsel

    idx = types.SimpleNamespace(root=tmp_path)
    cfg = _cfg(micro_facts=True, test_command="pytest -q")
    cfg._run_started = _time.time()
    monkeypatch.setattr(testsel, "impacted_tests", lambda i, ch: [])
    monkeypatch.setattr(testsel, "narrow_command",
                        lambda *a, **k: ("pytest -q", False))

    def fake_run(root, cfg_, command):
        memory.observe_command(root, command, False, cfg_, kind="test")
        return False, "1 failed"

    monkeypatch.setattr(testsel, "run_command", fake_run)
    _ok1, _out1, desc1 = testsel.run_impacted(idx, cfg, ["app.py"])
    ok2, _out2, desc2 = testsel.run_impacted(idx, cfg, ["app.py"])
    assert ok2 is False
    assert "pre-existing" not in desc1 and "pre-existing" not in desc2
    # Without a run stamp (eval flows) the old baseline reading survives.
    cfg2 = _cfg(micro_facts=True, test_command="pytest -q")
    _ok3, _out3, desc3 = testsel.run_impacted(idx, cfg2, ["app.py"])
    assert "pre-existing" in desc3


# -- BUG-1008: "NOT RUN" must blame the knob that is actually missing ---------


def test_granted_but_undetected_command_says_so(tmp_path, monkeypatch,
                                                capsys):
    from pedacito import testsel

    monkeypatch.setattr(testsel, "detect_test_command",
                        lambda root, cfg=None: "")
    agent = types.SimpleNamespace(_repair_from_findings=lambda *a, **k: "")
    loop._post_edit_tests(
        tmp_path, _cfg(verify_tests=True, allow_shell=False,
                       test_command="pytest -q"), agent,
        types.SimpleNamespace(root=tmp_path), "make pytest pass",
        [], "", None, None, [])
    err = capsys.readouterr().err
    assert "tests: NOT RUN" in err and "no test command was detected" in err
    assert "allow_shell is off" not in err


# -- BUG-1009: an unlaunchable check is retired, not repaired ------------------


def test_run_loop_retires_unlaunchable_verify(tmp_path, monkeypatch, capsys):
    """`auto --verify "pytest -q"` run OUTSIDE the venv: `/bin/sh: line 1:
    pytest: command not found` reached the repair ladder, burned model repair
    rounds per task, and the ratchet rolled valid steps back over a PATH
    problem no edit can fix (run 3, task 1 -- the model itself proposed the
    right answer: the binary is missing)."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "build it", ["Add helper", "Add second module"])
    idx = types.SimpleNamespace(root=tmp_path)

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None, is_repair=False):
        (tmp_path / "helper2.py").write_text("x = 1\n", encoding="utf-8")
        return loop.TaskResult(
            [types.SimpleNamespace(ok=True, file="helper2.py")],
            "reply", True, ["helper2.py"], idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    calls: list = []

    def fake_verify(root, cmd, *a, **kw):
        calls.append(cmd)
        return False, "/bin/sh: line 1: pytest: command not found"

    monkeypatch.setattr(loop, "run_verify", fake_verify)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         verify="pytest -q", repair_rounds=2)
    err = capsys.readouterr().err
    assert code == 0
    assert "verification check retired" in err
    assert len(calls) == 1                      # not re-broken per task
    assert (tmp_path / "helper2.py").exists()   # step NOT rolled back


def test_scaffold_command_not_found_is_environment(tmp_path):
    from pedacito import scaffold
    assert scaffold._environment_issue(
        "/bin/sh: line 1: pytest: command not found")
    assert scaffold._environment_issue(
        "'pytest' is not recognized as an internal or external command")
    assert not scaffold._environment_issue("AssertionError: 1 == 2")


# -- BUG-1010: a named-cause 400 is not an acceleration rejection -------------


def test_accel_denial_respects_the_named_cause():
    """`test_unrelated_400_on_plain_does_not_loop` passed only when another
    test had warmed the endpoint-keyed denial cache: standalone, the plain
    rung's 400 -- whose body named `response_format`, not `cache_prompt` --
    was charged to the acceleration keys, flipped the cap and re-sent the
    request (4 calls, not the documented 3)."""
    from pedacito.config import Config
    from pedacito.llm import LLMClient

    cfg = Config()
    cfg.model = "bug1010-isolated"
    c = LLMClient(cfg)
    named = RuntimeError('server returned 400: {"error": "response_format '
                         'is not valid here"}')
    assert not c._deny_accel_caps({"cache_prompt": True}, named)
    ours = RuntimeError('server returned 400: {"error": "unexpected key '
                        'cache_prompt"}')
    assert c._deny_accel_caps({"cache_prompt": True}, ours)


# -- BUG-1011: the config upgrader's own backup is not project content -------


def test_config_upgrade_backup_is_harness_owned():
    """`pedacito config --upgrade --path .pedacito.toml` writes
    `.pedacito.toml.bak` (documented, expected), and the next documented step
    `pedacito new .` refused the directory as "not empty
    (.pedacito.toml.bak)". The freshness guard asks whether the folder holds a
    PROJECT; a backup of harness config is not one."""
    from pedacito import scaffold

    assert scaffold.is_harness_owned(".pedacito.toml")
    assert scaffold.is_harness_owned(".pedacito.toml.bak")
    assert scaffold.is_harness_owned(".pedacito.toml.bak-20261114-071300")
    assert scaffold.is_harness_owned("pedacito.toml.bak")
    assert not scaffold.is_harness_owned("index.html")
    assert not scaffold.is_harness_owned("main.py.bak")


# -- BUG-1014: the documented qualified rename works from the CLI ------------


def test_rename_accepts_qualified_query_and_follows_receivers(tmp_path):
    """README's "Renaming a symbol across the project" documents
    `pedacito rename Store.save save_record`; the CLI rejected every dotted
    name as "not a valid identifier" before reaching the engine. The engine
    now takes the qualified spelling and renames only that class's member:
    another class's `save`, an untyped receiver and string text stay put."""
    from pedacito import refactor

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg/store.py").write_text(
        "class Store:\n"
        "    def save(self, item):\n"
        "        return self.save(item)\n"
        "\n"
        "class Other:\n"
        "    def save(self, item):\n"
        "        return item\n", encoding="utf-8")
    (tmp_path / "pkg/app.py").write_text(
        "from pkg.store import Other, Store\n"
        "\n"
        "def run(z):\n"
        "    x = Store()\n"
        "    y = Other()\n"
        "    return x.save(1 if 'save' else 0), y.save(2), z.save(3)\n",
        encoding="utf-8")

    res = refactor.rename_symbol_across_project(
        tmp_path, "Store.save", "save_record", dry_run=True)
    assert res["ok"], res
    assert res["per_file"] == {"pkg/store.py": 2, "pkg/app.py": 1}, res
    assert any("untyped receiver" in s for s in res["not_renamed"]), \
        res["not_renamed"]

    res = refactor.rename_symbol_across_project(
        tmp_path, "Store.save", "save_record")
    assert res["ok"], res
    store = (tmp_path / "pkg/store.py").read_text(encoding="utf-8")
    app = (tmp_path / "pkg/app.py").read_text(encoding="utf-8")
    assert "def save_record(self, item)" in store
    assert "self.save_record(item)" in store
    assert "class Other:\n    def save(self" in store   # foreign member
    assert "x.save_record(1" in app
    assert "y.save(2)" in app and "z.save(3)" in app
    assert "'save'" in app                              # string untouched


def test_cli_rename_refreshes_the_index_in_place(tmp_path):
    """TECHNIQUES.md #26 promises the applied files are re-indexed so the next
    search does not answer from a stale symbol table; the CLI wrote the files
    and returned, leaving the old identifier in `.pedacito/index.json`."""
    import argparse

    from pedacito import cli, index as index_mod

    (tmp_path / "app.py").write_text(
        "def load(x):\n    return x\n\n\ndef use():\n    return load(1)\n",
        encoding="utf-8")
    cfg = _cfg()
    idx = index_mod.build([str(tmp_path)], cfg, client=None, root=tmp_path,
                          summarise=False)
    assert any(c.qualname.split(".")[-1] == "load" for c in idx.chunks)

    args = argparse.Namespace(old="load", new="read_all", file=None,
                              strings=False, apply=True, root=str(tmp_path),
                              config=None, profile=None, paths=[],
                              no_gitignore=False, exclude=[], include=[])
    assert cli._cmd_rename(args) == 0
    assert "read_all(1)" in (tmp_path / "app.py").read_text(encoding="utf-8")

    fresh = index_mod.Index.load(tmp_path)
    names = {c.qualname.split(".")[-1] for c in fresh.chunks}
    assert "read_all" in names and "load" not in names


# -- BUG-1012: the post-edit suite must use an interpreter that HAS pytest ---


def test_detected_pytest_uses_the_pedacito_interpreter(tmp_path, monkeypatch):
    """A fresh project has `tests/` but no installed pytest for the PATH
    `python`. Detection emitted `python -m pytest -q` anyway, so every
    post-edit gate recorded `tests: FAILED` with ModuleNotFoundError while the
    user's `--verify "/venv/python -m pytest -q"` on the same suite passed --
    two commands, two verdicts. With no `pytest` on PATH, name the interpreter
    that runs Pedacito (pytest is a core dependency)."""
    import shlex
    import sys

    from pedacito import testsel

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_app.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8")
    monkeypatch.setattr(testsel.shutil, "which", lambda name: None)
    cmd = testsel.detect_test_command(tmp_path, _cfg())
    assert cmd == f"{shlex.quote(sys.executable)} -m pytest -q"
    assert testsel.narrow_command(cmd, ["tests/test_app.py"])[1] is True
