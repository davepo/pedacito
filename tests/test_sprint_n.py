"""The fast static analysis gate and the green-ratchet checkpoints.

Part 1 -- the static gate (`spec.run_fast_static_gate`, `verify.check_static`,
`loop._post_edit_tests`, `loop._gated_check`): undefined names, bad
`__all__` entries and syntax breakage are caught in milliseconds (ruff with
five rules, or a pure-AST scope resolver when ruff cannot run), and only
errors NEW since the step started may trigger the cheap repair microloop --
the test subprocess is never started against a patch that cannot resolve its
own names.

Part 2 -- the green ratchet (`pedacito/ratchet.py`, wired into `run_loop`):
every `auto` step snapshots the project first; a step that fails its check
after all repair rounds is UNDONE (modified files restored, created files
deleted) so later steps can never build on wreckage.

The AST engine is forced in most tests (`spec._ruff_command` -> []) so they
behave identically on machines with and without ruff; one test exercises the
real ruff engine, one exercises the timeout fallback.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from pedacito import fileio, loop, ratchet, spec, verify
from pedacito.config import Config
from pedacito.edits import Result

BROKEN = "def total(n):\n    return jsonx.dumps(n)\n"
GOOD = "import json\n\n\ndef total(n):\n    return json.dumps(n)\n"


def _pw(path, text: str) -> None:
    """Write a project file the way the loop does -- via fileio.write's
    atomic replace, so a ratchet checkpoint hardlinked to the previous
    inode keeps its bytes."""
    fileio.write(Path(path), text, fileio.TextStyle())


def _force_ast(monkeypatch):
    """Make the gate use the pure-AST engine (no ruff subprocess)."""
    monkeypatch.setattr(spec, "_ruff_command", lambda: [])


# ------------------------------------------------------------- AST fallback
def test_ast_fallback_flags_missing_import_with_hint():
    errs = spec.ast_static_errors("m.py", BROKEN)
    assert len(errs) == 1
    assert "F821" in errs[0] and "jsonx" in errs[0]
    assert errs[0].startswith("m.py:2:12: ")


def test_ast_fallback_hints_common_modules():
    errs = spec.ast_static_errors("m.py", "print(json.dumps(1))\n")
    assert "F821" in errs[0] and "import json" in errs[0]


def test_ast_fallback_clean_on_real_python_patterns():
    src = '''
import json, os
from typing import Optional


class Widget:
    LIMIT = 5

    def render(self, items):
        seen = [i for i in items if i not in seen_in_call(None)]
        return json.dumps({"seen": len(seen), "cwd": os.getcwd()})


def seen_in_call(x: Optional[list]):
    total = 0

    def bump():
        nonlocal total
        total += 1
        return total
    for part in (x or []):
        bump()
    with open(__file__) as fh:
        line = fh.read()
    try:
        int(line)
    except ValueError as exc:
        return -1
    return total


ROWS = [[c for c in range(r)] for r in range(Widget.LIMIT)]
fn = lambda q: q + 1
if (n := len(ROWS)) > 2:
    print(n, fn, seen_in_call, ROWS)
'''
    assert spec.ast_static_errors("m.py", src) == []


def test_ast_fallback_skips_star_imports_and_exec():
    assert spec.ast_static_errors("m.py",
                                  "from os import *\nprint(anything)\n") == []
    assert spec.ast_static_errors("m.py",
                                  "exec('x=1')\nprint(undefined_name)\n") == []


def test_ast_fallback_catches_class_attribute_from_method():
    src = "class C:\n    attr = 1\n    def m(self):\n        return attr\n"
    errs = spec.ast_static_errors("m.py", src)
    assert any("F821" in e and "attr" in e for e in errs)


def test_ast_fallback_syntax_error():
    errs = spec.ast_static_errors("m.py", "def f(:\n    pass\n")
    assert len(errs) == 1 and "E999" in errs[0]


def test_ast_fallback_all_entries():
    assert spec.ast_static_errors(
        "m.py", "def f():\n    pass\n__all__ = ['f']\n") == []
    errs = spec.ast_static_errors(
        "m.py", "def f():\n    pass\n__all__ = ['f', 'ghost']\n")
    assert len(errs) == 1 and "F822" in errs[0] and "ghost" in errs[0]


def test_ast_fallback_use_before_local_binding():
    src = "x = 1\n\n\ndef f():\n    print(x)\n    x = 2\n"
    errs = spec.ast_static_errors("m.py", src)
    assert any("F823" in e for e in errs)


def test_ast_fallback_import_shadowed_by_loop_and_late_future():
    errs = spec.ast_static_errors("m.py", "import os\nfor os in [1]:\n    pass\n")
    assert any("F402" in e for e in errs)
    errs = spec.ast_static_errors("m.py", "import os\nfrom __future__ import x\n")
    assert any("F404" in e for e in errs)


def test_ast_fallback_finds_no_false_positives_in_pedacito_itself():
    """The whole real repo parses through the fallback with zero errors --
    the conservative rule: anything ambiguous stays quiet."""
    root = Path(spec.__file__).resolve().parents[1]
    files = spec.list_python_files(root, cap=300)
    files += spec.list_python_files(root / "pedacito", cap=300)
    errors = []
    for p in files:
        errors += spec.ast_static_errors(p.name, p.read_text(encoding="utf-8"))
    assert errors == [], errors[:4]


# ------------------------------------------------------------- gate engines
def test_gate_uses_ruff_when_available(tmp_path):
    if not spec._ruff_command():
        pytest.skip("ruff not installed here")
    (tmp_path / "m.py").write_text(BROKEN)
    errs = spec.run_fast_static_gate(tmp_path, [tmp_path / "m.py"])
    assert any("F821" in e and "jsonx" in e for e in errs), errs
    assert errs[0].startswith("m.py:")


def test_gate_falls_back_to_ast_without_ruff(tmp_path, monkeypatch):
    _force_ast(monkeypatch)
    (tmp_path / "m.py").write_text(BROKEN)
    errs = spec.run_fast_static_gate(tmp_path, [tmp_path / "m.py"])
    assert any("F821" in e for e in errs), errs


def test_gate_falls_back_when_ruff_times_out(tmp_path, monkeypatch):
    monkeypatch.setattr(
        spec, "_ruff_command",
        lambda: [sys.executable, "-c", "import time; time.sleep(30)"])
    (tmp_path / "m.py").write_text(BROKEN)
    errs = spec.run_fast_static_gate(tmp_path, [tmp_path / "m.py"],
                                     timeout=0.4)
    assert any("F821" in e for e in errs), errs


def test_gate_ignores_missing_and_non_python(tmp_path, monkeypatch):
    _force_ast(monkeypatch)
    (tmp_path / "notes.md").write_text("anything here")
    errs = spec.run_fast_static_gate(
        tmp_path, [tmp_path / "nope.py", tmp_path / "notes.md"])
    assert errs == []


def test_static_error_key_strips_position():
    a = "m.py:12:5: F821 Undefined name `json` -- did you forget `import json`?"
    b = "m.py:99:1: F821 Undefined name `json` -- did you forget `import json`?"
    assert spec.static_error_key(a) == spec.static_error_key(b)
    c = "m.py:12:5: F821 Undefined name `os`"
    assert spec.static_error_key(c) != spec.static_error_key(a)


# ------------------------------------------------------------ verify buffers
def _buf_results(before, after):
    return [Result(True, "m.py", "applied", before=before, after=after)]


def test_check_static_flags_new_buffer_error():
    findings = verify.check_static(
        _buf_results(GOOD, "def f():\n    return jsonx\n"))
    assert len(findings) == 1
    f = findings[0]
    assert f.kind == "static" and f.severity == "error" and f.file == "m.py"
    assert "jsonx" in f.message


def test_check_static_quiet_on_preexisting_and_clean():
    assert verify.check_static(_buf_results(BROKEN, BROKEN)) == []
    assert verify.check_static(_buf_results(GOOD, GOOD)) == []
    assert verify.check_static(_buf_results("", "")) == []


def test_check_pipeline_includes_static_findings(tmp_path):
    agent = types.SimpleNamespace(cfg=Config(), root=tmp_path)
    findings = verify.check(agent, "touch m.py", "",
                            _buf_results(GOOD, BROKEN))
    assert any(f.kind == "static" for f in findings)

    off = agent.cfg
    off.static_gate = False
    findings = verify.check(agent, "touch m.py", "",
                            _buf_results(GOOD, BROKEN))
    assert not any(f.kind == "static" for f in findings)


# ---------------------------------------------------- _post_edit_tests gate
class _FakeAgent:
    def __init__(self, fixed_src):
        self.fixed_src = fixed_src
        self.calls = []

    def _repair_from_findings(self, task, reply, errors, results):
        self.calls.append([e.kind for e in errors])
        return "REPAIR"

    def _apply_once(self, reply, dry_run, backup_dir):
        # The repair lands: rewrite the real file and return its Result.
        (Path(".") / "m.py").write_text(self.fixed_src)
        return [Result(True, "m.py", "applied", before=BROKEN,
                       after=self.fixed_src)]


def test_post_edit_static_repairs_before_any_test_subprocess(tmp_path,
                                                             monkeypatch):
    cfg = Config()
    cfg.verify_tests = True
    cfg.allow_shell = True
    cfg.repair_rounds = 1
    cfg._static_baseline = set()
    _force_ast(monkeypatch)
    (tmp_path / "m.py").write_text(BROKEN)
    monkeypatch.chdir(tmp_path)
    agent = _FakeAgent(GOOD)
    runs = []
    monkeypatch.setattr(loop, "_run_tests",
                        lambda idx, cfg_, files: (runs.append(list(files))
                                                  or (True, "ok", "pytest")))
    results = [Result(True, "m.py", "applied", before="", after=BROKEN)]
    out_results, reply, tests_ok = loop._post_edit_tests(
        tmp_path, cfg, agent, types.SimpleNamespace(root=tmp_path), "task",
        results, "reply", None, None, ["m.py"])
    # The static error went to repair first; the test subprocess ran exactly
    # once -- AFTER the patch was already green again.
    assert agent.calls == [["static"]]
    assert len(runs) == 1 and runs[0] == ["m.py"]
    assert tests_ok is True
    assert any(r.file == "m.py" and r.after == GOOD for r in out_results)


def test_post_edit_preexisting_static_errors_do_not_fire(tmp_path, monkeypatch):
    cfg = Config()
    cfg.verify_tests = True
    cfg.allow_shell = True
    _force_ast(monkeypatch)
    from pedacito import spec as sp
    cfg._static_baseline = {sp.static_error_key(e)
                            for e in sp.ast_static_errors("m.py", BROKEN)}
    (tmp_path / "m.py").write_text(BROKEN)
    agent = _FakeAgent(GOOD)
    runs = []
    monkeypatch.setattr(loop, "_run_tests",
                        lambda idx, cfg_, files: (runs.append(1)
                                                  or (True, "ok", "pytest")))
    results = [Result(True, "m.py", "applied", before=BROKEN, after=BROKEN)]
    loop._post_edit_tests(tmp_path, cfg, agent, types.SimpleNamespace(
        root=tmp_path), "task", results, "reply", None, None, ["m.py"])
    assert agent.calls == []          # pre-existing: never repaired
    assert len(runs) == 1             # straight to the tests


def test_static_gate_off_skips_everything(tmp_path, monkeypatch):
    cfg = Config()
    cfg.static_gate = False
    (tmp_path / "m.py").write_text(BROKEN)
    assert loop._new_static_errors(tmp_path, cfg, ["m.py"]) == []
    cfg._static_baseline = None
    assert loop._static_errors(tmp_path, cfg, ["m.py"]) == []


# ------------------------------------------------------------- ratchet core
def test_ratchet_begin_green_rollback_roundtrip(tmp_path):
    # The loop writes through fileio.write (atomic replace), so snapshots
    # -- hard links when the fs allows -- keep their own inode. Editing a.py
    # must never truncate the checkpointed bytes.
    _pw(tmp_path / "a.py", "GOOD = 1\n")
    info = ratchet.begin(tmp_path, 1, "step one")
    assert "a.py" in info["files"]
    # The step edits a.py and creates new.py:
    _pw(tmp_path / "a.py", "BAD = jsonx\n")
    (tmp_path / "new.py").write_text("x = 1\n")
    ratchet.green(tmp_path, 1)
    assert ratchet.frontier(tmp_path) == 1

    info = ratchet.begin(tmp_path, 2, "step two")
    _pw(tmp_path / "a.py", "WRECKED\n")
    (tmp_path / "c.py").write_text("created in step two\n")
    back = ratchet.rollback(tmp_path, 2)
    assert back["restored"] == ["a.py"]
    assert back["deleted"] == ["c.py"]
    assert (tmp_path / "a.py").read_text() == "BAD = jsonx\n"
    assert not (tmp_path / "c.py").exists()
    assert ratchet.frontier(tmp_path) == 1     # green target still stands
    # The failed step's snapshot is consumed:
    assert not ratchet.step_dir(tmp_path, 2).exists()


def test_ratchet_rollback_skips_untouched_files(tmp_path):
    (tmp_path / "a.py").write_text("GOOD = 1\n")
    ratchet.begin(tmp_path, 1)
    (tmp_path / "b.py").write_text("new\n")
    back = ratchet.rollback(tmp_path, 1)
    assert back["restored"] == []
    assert back["deleted"] == ["b.py"]


def test_ratchet_rollback_never_undoes_the_plan_file(tmp_path):
    plan = tmp_path / "PLAN.md"
    plan.write_text("- [ ] work\n")
    ratchet.begin(tmp_path, 1, skip=["PLAN.md"])
    plan.write_text("- [x] work\n")           # the loop ticks it mid-step
    back = ratchet.rollback(tmp_path, 1)
    assert plan.read_text() == "- [x] work\n"
    assert back["restored"] == [] and back["deleted"] == []


def test_ratchet_rollback_without_checkpoint_reports_error(tmp_path):
    back = ratchet.rollback(tmp_path, 7)
    assert back["error"] and "no checkpoint" in back["error"]


def test_ratchet_skips_state_and_huge_files(tmp_path):
    (tmp_path / ".pedacito").mkdir()
    (tmp_path / ".pedacito" / "state.json").write_text("{}")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.py").write_text("x\n")
    big = tmp_path / "big.py"
    big.write_text("x = 1\n" + "# pad\n" * 2000)
    info = ratchet.begin(tmp_path, 1, max_file_bytes=50)
    assert not any(f.startswith(".pedacito") for f in info["files"])
    assert not any("node_modules" in f for f in info["files"])
    assert "big.py" not in info["files"]        # recorded, not copied
    assert "state.json" not in "\n".join(info["files"])


def test_ratchet_prune_keeps_the_latest(tmp_path):
    for i in range(1, 6):
        ratchet.begin(tmp_path, i)
    assert ratchet.prune(tmp_path, keep=2) == 3
    assert ratchet.step_dir(tmp_path, 4).is_dir()
    assert ratchet.step_dir(tmp_path, 5).is_dir()
    assert not ratchet.step_dir(tmp_path, 1).exists()


# ------------------------------------------------------- run_loop end to end
def _cfg():
    c = Config()
    c.log_sessions = False
    return c


def test_run_loop_rolls_back_the_failed_step(tmp_path, monkeypatch):
    """Task 1 lands green work; task 2 wrecks a.py and adds new.py, then its
    check fails with zero repair rounds left -- the workspace must return to
    exactly the post-task-1 state."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal", ["Build module one", "Build module two"])
    idx = types.SimpleNamespace(root=tmp_path)
    cfg = _cfg()
    cfg.static_gate = False

    def fake_run_task(root, cfg_, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None, is_repair=False):
        title = task.split("\n\n")[0]
        if title == "Build module one":
            _pw(root / "a.py", "GOOD = 1\n")
            changed = ["a.py"]
        else:
            _pw(root / "a.py", "BAD = wrecked\n")
            (root / "new.py").write_text("orphan\n")
            changed = ["a.py", "new.py"]
        return loop.TaskResult([Result(True, "a.py", "applied")],
                               "done the work", True, changed, idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    verdicts = iter([True, False])   # task 1 verifies green, task 2 fails
    monkeypatch.setattr(
        loop, "run_verify",
        lambda root, cmd: (next(verdicts, False), "boom"))
    code = loop.run_loop(tmp_path, cfg, None, idx, plan, apply=True,
                         verify="pytest -q", repair_rounds=0,
                         stop_on_fail=False)
    assert code == 1
    assert (tmp_path / "a.py").read_text() == "GOOD = 1\n"   # restored
    assert not (tmp_path / "new.py").exists()                # deleted
    text = plan.read_text()
    assert "- [x] Build module one" in text
    assert "- [!] Build module two" in text
    assert ratchet.frontier(tmp_path) == 1     # step 1 is still the target


def test_run_loop_ratchet_off_leaves_the_wreckage(tmp_path, monkeypatch):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["Build module one", "Build module two"])
    idx = types.SimpleNamespace(root=tmp_path)
    cfg = _cfg()
    cfg.static_gate = False
    cfg.auto_green_ratchet = False

    def fake_run_task(root, cfg_, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None, is_repair=False):
        title = task.split("\n\n")[0]
        if title == "Build module one":
            _pw(root / "a.py", "GOOD = 1\n")
            changed = ["a.py"]
        else:
            _pw(root / "a.py", "BAD = wrecked\n")
            changed = ["a.py"]
        return loop.TaskResult([Result(True, "a.py", "applied")],
                               "done", True, changed, idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    monkeypatch.setattr(loop, "run_verify", lambda root, cmd: (False, "boom"))
    loop.run_loop(tmp_path, cfg, None, idx, plan, apply=True,
                  verify="pytest -q", repair_rounds=0, stop_on_fail=False)
    assert (tmp_path / "a.py").read_text() == "BAD = wrecked\n"
    assert ratchet.frontier(tmp_path) == 0


def test_run_loop_keeps_checkpoints_when_checks_pass(tmp_path, monkeypatch):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["one", "two"])
    idx = types.SimpleNamespace(root=tmp_path)
    cfg = _cfg()
    cfg.static_gate = False

    def fake_run_task(root, cfg_, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None, is_repair=False):
        _pw(root / "f.py", task.split("\n\n")[0] + "\n")
        return loop.TaskResult([Result(True, "f.py", "applied")],
                               "done", True, ["f.py"], idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    monkeypatch.setattr(loop, "run_verify", lambda root, cmd: (True, ""))
    code = loop.run_loop(tmp_path, cfg, None, idx, plan, apply=True,
                         verify="pytest -q")
    assert code == 0
    assert ratchet.frontier(tmp_path) == 2
    assert ratchet.step_dir(tmp_path, 2).is_dir()    # green restore point kept


def test_run_loop_step_static_gate_blocks_the_verify_subprocess(
        tmp_path, monkeypatch):
    """With the static gate on, a step that introduces an undefined name
    fails fast: the --verify subprocess never starts, the task fails, and the
    green ratchet rolls the step back -- the two features compounding."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["Add the helper"])
    idx = types.SimpleNamespace(root=tmp_path)
    cfg = _cfg()
    _force_ast(monkeypatch)

    def fake_run_task(root, cfg_, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None, is_repair=False):
        (root / "h.py").write_text(BROKEN)
        return loop.TaskResult([Result(True, "h.py", "applied")],
                               "done", True, ["h.py"], idx_)

    verify_runs = []
    monkeypatch.setattr(loop, "run_task", fake_run_task)
    monkeypatch.setattr(
        loop, "run_verify",
        lambda root, cmd: verify_runs.append(1) or (True, ""))
    code = loop.run_loop(tmp_path, cfg, None, idx, plan, apply=True,
                         verify="pytest -q", repair_rounds=0)
    assert code == 1
    assert verify_runs == []                 # gate short-circuited the shell
    assert not (tmp_path / "h.py").exists()  # and the ratchet undid the step


def test_config_flags_defaults_and_taskresult_fields():
    c = Config()
    assert c.static_gate is True and c.static_gate_timeout == 3.0
    assert c.auto_green_ratchet is True
    assert c.auto_checkpoint_max_file_bytes > 0
    assert c.auto_checkpoint_max_files > 0 and c.auto_keep_checkpoints > 0
    tr = loop.TaskResult([], "r", False, [], None)
    assert tr.rolled_back is False and tr.rollback_info == {}
