"""Offline tests for the autonomous outer loop (`pedacito auto`).

Covers the plan-file checklist helpers, the goal->task planning call, and the
Ralph loop driver itself (task ordering, marking done/failed, verification
repair round-trips, and the stop-on-fail / keep-going paths). No LLM server
required: the inner loop is monkeypatched away so the OUTER loop logic is what
is under test.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import loop
from pedacito.config import Config


# ------------------------------------------------------------------- plan file
def test_write_and_read_plan_roundtrip(tmp_path):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "build a todo CLI", ["task one", "task two"])
    text = plan.read_text()
    assert "# Plan" in text and "Goal: build a todo CLI" in text
    assert "- [ ] task one" in text and "- [ ] task two" in text
    assert loop.read_plan(plan) == text.split("\n")


def test_next_task_and_mark_done(tmp_path):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["first", "second", "third"])
    lines = loop.read_plan(plan)
    idx, task = loop.next_task(lines)
    assert task == "first"
    assert loop.mark_done(plan, "first")
    idx, task = loop.next_task(loop.read_plan(plan))
    assert task == "second"
    # A task already done is never re-picked, and [x] boxes are not unchecked.
    assert "- [x] first" in plan.read_text()


def test_mark_failed_is_skipped_not_retried(tmp_path):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["doomed", "next"])
    assert loop.mark_failed(plan, "doomed")
    idx, task = loop.next_task(loop.read_plan(plan))
    assert task == "next", "a failed task must not be re-picked by the loop"
    assert "- [!] doomed" in plan.read_text()


def test_mark_unknown_task_returns_false(tmp_path):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["known"])
    assert loop.mark_done(plan, "not in the plan") is False


# ----------------------------------------------------------- task kinds
def test_investigation_tasks_are_recognised():
    assert loop._is_investigation_task(
        "Reproduce the black screen by running main.py")
    assert loop._is_investigation_task("Locate the cause of the crash")
    assert loop._is_investigation_task("Check the render path")
    assert loop._is_investigation_task(
        "Audit the current game code to identify which of the seven requested "
        "enhancements are already implemented and working")
    assert loop._is_investigation_task("Assess which endpoints are covered")
    assert loop._is_investigation_task("Report what the scheduler does")
    assert not loop._is_investigation_task("Fix the black screen")
    assert not loop._is_investigation_task("Reproduce and fix the crash")
    assert not loop._is_investigation_task("Implement the parser")
    assert not loop._is_investigation_task("Audit and fix the collision logic")


def test_tool_call_only_answer_is_detected():
    assert loop._looks_like_tool_calls_only(
        '<tool_call>{"name": "run", "arguments": {}}</tool_call>')
    assert not loop._looks_like_tool_calls_only(
        "Likely cause: the surface is never cleared. "
        "The run produced no output, so the loop is spun before drawing.")


def test_run_loop_marks_investigation_task_done_without_edits(tmp_path,
                                                             monkeypatch):
    """The reported failure: a reproduce/diagnose task is a success even when
    it changes no code. It must be ticked and the next fix task must still
    run, instead of stopping the whole loop on 'no applied edits'."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "fix the black screen",
                    ["Reproduce the black screen",
                     "Fix the rendering bug"])
    idx = types.SimpleNamespace(root=tmp_path)
    calls = []

    def fake_investigate(root, cfg, client, agent, idx_, task,
                         answer_system=None):
        calls.append(("investigate", task))
        return loop.TaskResult(
            [], "Reproduced: the window opens black and the process produces "
            "no output.", True, [], idx_)

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None):
        calls.append(("edit", task))
        return _result(idx_)

    monkeypatch.setattr(loop, "run_investigation_task", fake_investigate)
    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
    assert code == 0
    assert [kind for kind, _ in calls] == ["investigate", "edit"]
    fix_task = calls[1][1]
    assert "CONTEXT FROM PRIOR INVESTIGATION" in fix_task
    assert "window opens black" in fix_task
    assert "- [x] Reproduce the black screen" in plan.read_text()
    assert "- [x] Fix the rendering bug" in plan.read_text()


# ------------------------------------------------------------------- planning
def test_generate_tasks_extracts_titles():
    class Stub:
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            assert schema["required"] == ["tasks"]
            return {"goal": "g", "tasks": [
                {"title": "Implement the parser"},
                {"title": "Wire the CLI"},
                {"title": ""},                      # dropped: empty title
                {"nonsense": "x"},                  # dropped: not a dict/field
            ]}
    titles = loop.generate_tasks("build a tool", Stub(), Config())
    assert titles == ["Implement the parser", "Wire the CLI"]


def test_generate_tasks_retries_once_when_empty():
    """A model handed a bug-fix goal sometimes returns zero tasks. The planner
    must ask once more (with an explicit nudge) before giving up."""
    calls = {"n": 0}

    class Stub:
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"tasks": []}
            return {"tasks": [{"title": "Reproduce the crash"},
                              {"title": "Fix the level loader"}]}

    titles = loop.generate_tasks("main.py crashes", Stub(), Config())
    assert titles == ["Reproduce the crash", "Fix the level loader"]
    assert calls["n"] == 2


def test_generate_tasks_falls_back_to_a_single_task():
    """If the model stays empty even after the retry, `auto` still makes an
    attempt instead of aborting the run outright."""
    class Stub:
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            return {"tasks": []}

    titles = loop.generate_tasks(
        "main.py crashes: FileNotFoundError in level_loader", Stub(), Config())
    assert titles == ["Address: main.py crashes: FileNotFoundError in level_loader"]


def test_generate_tasks_tolerates_plain_string_list():
    class Stub:
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            return {"tasks": ["First step", "Second step"]}
    assert loop.generate_tasks("do it", Stub(), Config()) == \
        ["First step", "Second step"]


# ---------------------------------------------------------------- run_loop
def _ok(file="a.py"):
    return types.SimpleNamespace(ok=True, file=file)


def _result(idx, applied=True, ok=True):
    return loop.TaskResult([_ok()] if ok else [],
                           "reply", applied, ["a.py"] if ok else [], idx)


def _cfg():
    c = Config()
    c.log_sessions = False
    return c


def test_run_loop_implements_each_task_in_order(tmp_path, monkeypatch):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal", ["task A", "task B", "task C"])
    idx = types.SimpleNamespace(root=tmp_path)
    calls = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True, review=False, answer_system=None):
        calls.append(task)
        return _result(idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
    assert code == 0
    assert calls == ["task A", "task B", "task C"]
    text = plan.read_text()
    assert "- [x] task A" in text and "- [x] task B" in text and "- [x] task C" in text


def test_run_loop_respects_max_tasks(tmp_path, monkeypatch):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["a", "b", "c"])
    idx = types.SimpleNamespace(root=tmp_path)
    calls = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True, review=False, answer_system=None):
        calls.append(task)
        return _result(idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True, max_tasks=2)
    assert calls == ["a", "b"]
    assert "- [ ] c" in plan.read_text()          # never reached


def test_run_loop_respects_max_minutes_between_tasks(tmp_path, monkeypatch):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["a", "b"])
    idx = types.SimpleNamespace(root=tmp_path)
    calls = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None):
        calls.append(task)
        return _result(idx_)

    # start, first-task check, second-task check (already over 1 minute).
    clock = iter([0.0, 0.0, 61.0])
    monkeypatch.setattr(loop.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         max_minutes=1)
    assert code == 0
    assert calls == ["a"]
    text = plan.read_text()
    assert "- [x] a" in text
    assert "- [ ] b" in text              # not started after the time limit


def test_run_loop_stops_on_failed_task(tmp_path, monkeypatch):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["broken", "never reached"])
    idx = types.SimpleNamespace(root=tmp_path)

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True, review=False, answer_system=None):
        return _result(idx_, ok=False)            # no applied edits

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
    assert code == 1
    assert "- [ ] broken" in plan.read_text()     # not marked (we stopped)


def test_run_loop_keep_going_flags_and_continues(tmp_path, monkeypatch):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["broken", "good"])
    idx = types.SimpleNamespace(root=tmp_path)
    calls = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True, review=False, answer_system=None):
        calls.append(task)
        return _result(idx_, ok=(task != "broken"))

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         stop_on_fail=False)
    assert calls == ["broken", "good"]
    assert code == 1
    text = plan.read_text()
    assert "- [!] broken" in text                 # flagged, never re-picked
    assert "- [x] good" in text


def test_run_loop_treats_noop_reply_as_done(tmp_path, monkeypatch):
    """A task the model reports as already complete (no edit blocks, but an
    explicit 'no changes needed') must be marked done and the loop moved on,
    not treated as a failure that halts the whole run."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["already done", "real work"])
    idx = types.SimpleNamespace(root=tmp_path)
    calls = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True, review=False, answer_system=None):
        calls.append(task)
        if task == "already done":
            return loop.TaskResult(
                [], "No changes are needed; the file is already correct.",
                True, [], idx_)
        return _result(idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
    assert code == 0
    assert calls == ["already done", "real work"]
    text = plan.read_text()
    assert "- [x] already done" in text
    assert "- [x] real work" in text


def test_run_loop_verification_repair_round_trip(tmp_path, monkeypatch):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["needs tests"])
    idx = types.SimpleNamespace(root=tmp_path)
    tasks = []
    verifies = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True, review=False, answer_system=None):
        tasks.append(task)
        return _result(idx_)

    def fake_verify(root, command):
        verifies.append(command)
        # Fail the first check, pass the second (after the model "fixes" it).
        return (len(verifies) >= 2), "boom" if len(verifies) < 2 else ""

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    monkeypatch.setattr(loop, "run_verify", fake_verify)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         verify="pytest -q", repair_rounds=2)
    assert code == 0
    assert len(tasks) == 2                         # original + one repair
    assert tasks[0] == "needs tests"
    assert "failed this check" in tasks[1]
    assert "- [x] needs tests" in plan.read_text()


def test_run_loop_verification_exhausts_repair_rounds(tmp_path, monkeypatch):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["always failing"])
    idx = types.SimpleNamespace(root=tmp_path)
    tasks = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True, review=False, answer_system=None):
        tasks.append(task)
        return _result(idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    monkeypatch.setattr(loop, "run_verify", lambda root, cmd: (False, "boom"))
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         verify="pytest -q", repair_rounds=2)
    assert code == 1
    assert len(tasks) == 3                         # initial + two repairs
    assert "- [ ] always failing" in plan.read_text()   # never marked done


def test_run_loop_stops_when_no_tasks(tmp_path, monkeypatch):
    plan = tmp_path / "PLAN.md"
    plan.write_text("# Plan\n\n- [x] already done\n", encoding="utf-8")
    idx = types.SimpleNamespace(root=tmp_path)
    calls = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True, review=False, answer_system=None):
        calls.append(task)
        return _result(idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
    assert code == 0
    assert calls == []


# ------------------------------------------------------ missing dependencies
def test_missing_modules_and_pip_names():
    out = ("Traceback (most recent call last):\n"
           "ModuleNotFoundError: No module named 'pygame'\n"
           "ImportError: No module named 'foo.bar'")
    assert loop.missing_modules(out) == ["pygame", "foo"]
    assert loop.missing_modules("nothing here") == []
    assert loop.pip_name("PIL") == "Pillow"
    assert loop.pip_name("sklearn") == "scikit-learn"
    assert loop.pip_name("pygame") == "pygame"


def test_run_loop_does_not_code_repair_missing_dependency(tmp_path, monkeypatch):
    """A verify failure caused by an absent package must NOT spawn repair
    tasks (editing source cannot install it); it is reported and the task
    fails."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["implement main"])
    idx = types.SimpleNamespace(root=tmp_path)
    tasks = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True, review=False, answer_system=None):
        tasks.append(task)
        return _result(idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    monkeypatch.setattr(loop, "run_verify", lambda root, cmd: (
        False, "ModuleNotFoundError: No module named 'pygame'"))
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         verify="python -c 'import main'", repair_rounds=3,
                         auto_install=False)
    assert code == 1
    assert tasks == ["implement main"], "must not spawn code-repair tasks"
    assert "- [ ] implement main" in plan.read_text()


def test_run_loop_auto_installs_missing_dependency(tmp_path, monkeypatch):
    """With auto_install on, the missing package is installed and the verify
    is retried; a pass then marks the task done."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["implement main"])
    idx = types.SimpleNamespace(root=tmp_path)
    installs = []
    verify_calls = {"n": 0}

    def fake_verify(root, cmd):
        verify_calls["n"] += 1
        if verify_calls["n"] == 1:
            return False, "ModuleNotFoundError: No module named 'pygame'"
        return True, ""

    monkeypatch.setattr(loop, "run_task", lambda *a, **k: _result(idx))
    monkeypatch.setattr(loop, "run_verify", fake_verify)
    monkeypatch.setattr(loop, "install_modules",
                        lambda pkgs, root: (installs.append(list(pkgs)), (True, ""))[1])
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         verify="python main.py", auto_install=True)
    assert code == 0
    assert installs == [["pygame"]]
    assert "- [x] implement main" in plan.read_text()


def test_run_verify_runs_in_project_root(tmp_path):
    ok, _ = loop.run_verify(tmp_path, f"{sys.executable} -c 'import sys; sys.exit(0)'")
    assert ok is True
    bad, out = loop.run_verify(tmp_path,
                               f"{sys.executable} -c 'import sys; sys.exit(3)'")
    assert bad is False


# ---------------------------------------------------------------- run_task
def test_run_task_applies_edits_and_reindexes(tmp_path):
    """The shared single-task executor must walk gather -> answer -> apply and
    actually write (the path `ask`/`chat`/`auto` all share after the refactor),
    using an offline stub client in place of a live server."""
    from pedacito import index as index_mod
    from pedacito.agent import Agent

    (tmp_path / "old_name.py").write_text('print("hi")\n', encoding="utf-8")
    cfg = Config()
    cfg.log_sessions = False
    idx = index_mod.build([str(tmp_path)], cfg, client=None, summarise=False, root=tmp_path)

    REPLY = (
        "Renaming the file.\n\n"
        "--- FILE: new_name.py\n"
        "<<<<<<< SEARCH\n"
        "=======\n"
        'print("hi")\n'
        ">>>>>>> REPLACE\n\n"
        "--- FILE: old_name.py\n"
        "<<<<<<< SEARCH\n"
        'print("hi")\n'
        "=======\n"
        ">>>>>>> REPLACE\n"
    )

    class StubClient:
        stream_truncated = False
        step_truncated = False
        def __init__(self):
            self.n = 0
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            if self.n == 0:
                self.n += 1
                return {"action": "read", "file": "old_name.py", "pattern": "",
                        "symbol": "", "start": 1, "end": 200, "content": "",
                        "reasoning": "read the file"}
            return {"action": "done", "file": "", "pattern": "", "symbol": "",
                    "start": 0, "end": 0, "content": "", "reasoning": "done"}
        def chat(self, messages, max_tokens, schema=None, stream=False, **kw):
            return REPLY

    client = StubClient()
    tr = loop.run_task(tmp_path, cfg, client, Agent(idx, client, cfg), idx,
                       "rename old_name.py to new_name.py", apply=True,
                       review=False)
    assert tr.applied, "shared executor did not apply the edit"
    assert "new_name.py" in tr.changed           # reindexed after the write
    assert (tmp_path / "new_name.py").is_file()
    assert not (tmp_path / "old_name.py").exists()
    assert 'print("hi")' in (tmp_path / "new_name.py").read_text()
    # The operational run log records the task and its result.
    from pedacito import workspace
    log = workspace.run_log_path(tmp_path).read_text()
    assert "task: rename old_name.py to new_name.py" in log
    assert "applied new_name.py" in log


def test_read_requests_parses_both_tool_call_dialects():
    """Answer-phase replies that ask for reads come in at least two dialects:
    the `<tool_call>`/`<function=read>` form and the Anthropic
    `<invoke name="read">` form. Both must yield their file paths."""
    reply = (
        "<tool_call>\n<function=read>\n<parameter=file_path>\nsrc/game.py\n"
        "</parameter>\n</function>\n</tool_call>\n"
        '<invoke name="read">\n<parameter name="file_path">src/ui.py</parameter>\n'
        "</invoke>\n"
        '<invoke name="write">\n<parameter name="file_path">src/x.py</parameter>\n'
        "</invoke>\n")
    assert loop._read_requests(reply) == ["src/game.py", "src/ui.py"]
    assert loop._read_requests("just prose, no tool calls") == []


def test_honour_reads_refuses_paths_outside_root(tmp_path):
    """Files requested by the model are read relative to the project root;
    anything outside it (or missing) is reported as unreadable instead of
    leaking the host filesystem into the prompt."""
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    out = loop._honour_reads(tmp_path, ["a.py", "../../etc/passwd", "nope.py"])
    assert "x = 1" in out
    assert "### ../../etc/passwd" in out and "not readable" in out
    assert "### nope.py" in out and "not readable" in out


def test_run_task_recovers_after_tool_call_reply(tmp_path):
    """The reported failure: the answer phase emitted `<tool_call>` reads for
    files gather never touched, so no edit blocks parsed and the task died.
    run_task must honour those reads from disk and re-ask -- the re-ask
    carries the requested source and the task then applies cleanly."""
    from pedacito import index as index_mod
    from pedacito.agent import Agent

    (tmp_path / "game.py").write_text("def tick():\n    pass\n", encoding="utf-8")
    (tmp_path / "player.py").write_text("x = 1\n", encoding="utf-8")
    cfg = Config()
    cfg.log_sessions = False
    idx = index_mod.build([str(tmp_path)], cfg, client=None, summarise=False,
                          root=tmp_path)

    TOOL_REPLY = (
        "<tool_call>\n<function=read>\n<parameter=file_path>\ngame.py\n"
        "</parameter>\n</function>\n</tool_call>\n")
    EDIT_REPLY = (
        "Implementing player.\n\n"
        "--- FILE: player.py\n"
        "<<<<<<< SEARCH\nx = 1\n=======\n"
        "def update():\n    pass\n>>>>>>> REPLACE\n")

    class StubClient:
        stream_truncated = False
        step_truncated = False

        def __init__(self):
            self.answers = 0

        def chat_json(self, messages, schema, max_tokens=0, **kw):
            return {"action": "done", "file": "", "pattern": "", "symbol": "",
                    "start": 0, "end": 0, "content": "", "reasoning": "done"}

        def chat(self, messages, max_tokens, schema=None, stream=False, **kw):
            self.answers += 1
            if self.answers == 1:
                return TOOL_REPLY
            user = "\n".join(m["content"] for m in messages
                             if m["role"] == "user")
            assert "EXTRA SOURCE READ FOR YOU" in user
            assert "def tick()" in user
            return EDIT_REPLY

    client = StubClient()
    tr = loop.run_task(tmp_path, cfg, client, Agent(idx, client, cfg), idx,
                       "implement player.py", apply=True, review=False)
    assert tr.applied, "recovery did not apply the re-asked edit"
    assert "def update()" in (tmp_path / "player.py").read_text()
    assert client.answers == 2


def test_run_log_records_events(tmp_path):
    from pedacito import workspace
    workspace.log_event(tmp_path, "hello", "INFO")
    workspace.log_event(tmp_path, "boom", "ERROR")
    log = workspace.run_log_path(tmp_path).read_text()
    assert "hello" in log and "boom" in log
    assert "INFO" in log and "ERROR" in log
    # one line per event, with a timestamp and a padded kind column
    lines = log.splitlines()
    assert len(lines) == 2
    assert "  INFO   " in lines[0] and "  ERROR  " in lines[1]


# ------------------------------------------------------ auto plan regeneration
def test_auto_regenerates_when_plan_is_complete(tmp_path):
    """`auto -t "new goal"` on a project whose PLAN.md is all-checked must
    start a fresh plan from the new goal, not silently re-read the finished
    checklist and report 'all tasks complete' without doing anything."""
    from pedacito import cli
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "old goal", ["old task"])
    assert loop.mark_done(plan, "old task")

    class Stub:
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            return {"tasks": [{"title": "fix the bug"}, {"title": "add a test"}]}

    args = types.SimpleNamespace(task="fix the bug", spec=None, plan=None,
                                 apply=False, force=False, verify=None,
                                 repair=None, max_tasks=None, keep_going=False)
    code = cli._cmd_auto(args, Config(), Stub(), None, tmp_path)
    assert code == 0
    text = plan.read_text()
    assert "- [ ] fix the bug" in text
    assert "- [ ] add a test" in text
    assert "old task" not in text


def test_auto_resumes_plan_with_open_tasks(tmp_path):
    """The inverse: a plan that still has unchecked work is resumed, not
    overwritten, when the same goal (or any -t) is given without --force."""
    from pedacito import cli
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal", ["done task", "open task"])
    assert loop.mark_done(plan, "done task")

    class Stub:
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            raise AssertionError("generate_tasks must not be called to resume")

    args = types.SimpleNamespace(task="anything", spec=None, plan=None,
                                 apply=False, force=False, verify=None,
                                 repair=None, max_tasks=None, keep_going=False)
    code = cli._cmd_auto(args, Config(), Stub(), None, tmp_path)
    assert code == 0
    text = plan.read_text()
    assert "- [x] done task" in text
    assert "- [ ] open task" in text


def test_auto_runs_post_verify_after_apply(tmp_path, monkeypatch):
    """`auto --apply` must exercise the project once after the checklist is
    done and feed a failure back, rather than declaring success blindly."""
    from pedacito import cli, scaffold
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal", ["only task"])
    seen = {}

    monkeypatch.setattr(loop, "run_loop", lambda *a, **k: 0)
    monkeypatch.setattr(scaffold, "verify_and_fix",
                        lambda *a, **k: seen.update(k) or (True, ""))

    args = types.SimpleNamespace(task=None, spec=None, plan=None, apply=True,
                                 force=False, verify=None, repair=None,
                                 max_tasks=None, keep_going=False,
                                 auto_install=False)
    code = cli._cmd_auto(args, Config(), None, None, tmp_path)
    assert code == 0
    assert "smoke" in seen, "auto --apply did not run the post-verify"


# ------------------------------------------------------ `new --complex` chaining
def test_plan_tasks_lists_one_task_per_source_file():
    from pedacito import scaffold
    plan = scaffold.Plan("md2html", "y", files=[
        scaffold.PlannedFile("main.py", "entry"),
        scaffold.PlannedFile("parser.py", "parse"),
        scaffold.PlannedFile("game/__init__.py", "package"),    # skipped
        scaffold.PlannedFile("README.md", "docs"),              # excluded
        scaffold.PlannedFile("tests/test_parser.py", "t"),      # excluded
    ], spec=scaffold.Spec("y"))
    assert scaffold.plan_tasks(plan) == [
        "Implement main.py -- entry",
        "Implement parser.py -- parse",
    ]


def test_plan_tasks_splits_oversized_files(tmp_path):
    """A source file whose stub is longer than auto_split_lines must be split
    into several tasks (one per group of top-level definitions) so no single
    whole-file reply overflows the model's output budget."""
    from pedacito import scaffold

    stub = "".join(
        f"def f{i}():\n    \"\"\"doc\"\"\"\n    pass\n\n" for i in range(10)
    )
    (tmp_path / "big.py").write_text(stub, encoding="utf-8")
    cfg = Config()
    plan = scaffold.Plan("x", "y", files=[scaffold.PlannedFile("big.py", "big")],
                         spec=scaffold.Spec("y"))
    idx = scaffold.skeleton_index(tmp_path, plan, cfg)
    tasks = scaffold.plan_tasks(plan, idx=idx, max_lines=8)
    assert len(tasks) > 1, f"expected split, got {tasks}"
    assert all("big.py:" in t and "lines " in t for t in tasks), tasks
    # every function is covered exactly once across the split tasks
    names = " ".join(tasks)
    for i in range(10):
        assert f"f{i}" in names

    # A small file is left whole.
    (tmp_path / "small.py").write_text("def f():\n    pass\n", encoding="utf-8")
    small_plan = scaffold.Plan("x", "y",
                               files=[scaffold.PlannedFile("small.py", "s")],
                               spec=scaffold.Spec("y"))
    idx2 = scaffold.skeleton_index(tmp_path, small_plan, cfg)
    assert scaffold.plan_tasks(small_plan, idx=idx2, max_lines=120) == \
        ["Implement small.py -- s"]


def test_new_complex_chains_to_auto_loop(tmp_path, monkeypatch):
    """`new --complex --apply` must scaffold the skeleton, write a resumable
    task checklist, and hand implementation to the outer loop with the
    stub-implementing answer brief."""
    from pedacito import scaffold

    calls = {}
    def fake_run_loop(root, cfg, client, idx, plan_path, **kw):
        calls["root"], calls["plan_path"], calls["kw"] = root, plan_path, kw
        return 0
    monkeypatch.setattr(loop, "run_loop", fake_run_loop)

    class Stub:
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            if "project_name" in schema["properties"]:
                return {"project_name": "md2html", "summary": "s",
                        "files": [{"path": "main.py", "purpose": "entry"},
                                  {"path": "parser.py", "purpose": "parse"}]}
            return {"files": [
                {"path": "main.py", "code": "def main():\n    pass\n"},
                {"path": "parser.py",
                 "code": "def parse(text):\n    raise NotImplementedError\n"},
            ]}
        def chat(self, *a, **k):
            return ""

    cfg = Config()
    code = scaffold.run_new(tmp_path, cfg, Stub(), scaffold.Spec("md2html"),
                            apply=True, complex=True, verbose=False)
    assert code == 0
    assert calls["kw"]["answer_system"] == scaffold.AUTO_IMPLEMENT_SYSTEM
    assert calls["plan_path"].name == "PLAN.md"
    # The checklist has one task per source file.
    text = "\n".join(loop.read_plan(calls["plan_path"]))
    assert "Implement main.py -- entry" in text
    assert "Implement parser.py -- parse" in text
    # The skeleton (interface stubs) is on disk before the loop runs.
    assert (tmp_path / "main.py").is_file()
    assert "def parse(text)" in (tmp_path / "parser.py").read_text()


def test_new_complex_dry_run_writes_nothing(tmp_path, monkeypatch):
    """`new --complex` WITHOUT --apply stays a dry run: nothing on disk, no
    loop, and the task list is only previewed."""
    from pedacito import scaffold

    ran = []
    def fake_run_loop(root, cfg, client, idx, plan_path, **kw):
        ran.append(plan_path)
        return 0
    monkeypatch.setattr(loop, "run_loop", fake_run_loop)

    class Stub:
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            return {"project_name": "x", "summary": "s",
                    "files": [{"path": "main.py", "purpose": "entry"}]}
        def chat(self, *a, **k):
            return ""

    code = scaffold.run_new(tmp_path, Config(), Stub(), scaffold.Spec("x"),
                            complex=True, verbose=False)
    assert code == 0
    assert ran == []
    assert list(tmp_path.iterdir()) == []


def _complex_stub():
    class Stub:
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            if "project_name" in schema["properties"]:
                return {"project_name": "x", "summary": "s",
                        "files": [{"path": "main.py", "purpose": "entry"}]}
            return {"files": [{"path": "main.py",
                               "code": "def main():\n    pass\n"}]}
        def chat(self, *a, **k):
            return ""
    return Stub()


def test_new_complex_runs_import_repair(tmp_path, monkeypatch):
    """Complex mode implements each file in a fresh context, so cross-file
    names can drift. It must run the same import-repair pass the simple `new`
    build does, or it reports success while shipping an unimportable tree."""
    from pedacito import scaffold

    calls = {"fix": 0}
    seq = {"n": 0}

    def fake_first_error(root, plan):
        seq["n"] += 1
        # A code-level mismatch (not a missing package), then clean.
        return ("ImportError: cannot import name 'x' from 'y'"
                if seq["n"] == 1 else None)

    monkeypatch.setattr(loop, "run_loop", lambda *a, **k: 0)
    monkeypatch.setattr(scaffold, "fix_import_errors",
                        lambda *a, **k: calls.__setitem__("fix", calls["fix"] + 1) or 0)
    monkeypatch.setattr(scaffold, "_first_import_error", fake_first_error)

    code = scaffold.run_new(tmp_path, Config(), _complex_stub(),
                            scaffold.Spec("x"), apply=True, complex=True,
                            verbose=False)
    assert calls["fix"] == 1, "complex mode skipped the import-repair pass"
    assert code == 0


def test_new_complex_handles_missing_dependency(tmp_path, monkeypatch):
    """A missing third-party package must not be sent through the code-repair
    loop (editing source cannot install it). With --auto-install it is
    installed and the import re-checked."""
    from pedacito import scaffold

    state = {"installed": [], "fix": 0, "checks": 0}

    def fake_first_error(root, plan):
        state["checks"] += 1
        if state["installed"]:
            return None
        return "ModuleNotFoundError: No module named 'pygame'"

    monkeypatch.setattr(loop, "run_loop", lambda *a, **k: 0)
    monkeypatch.setattr(scaffold, "_first_import_error", fake_first_error)
    monkeypatch.setattr(scaffold, "fix_import_errors",
                        lambda *a, **k: state.__setitem__("fix", state["fix"] + 1) or 0)

    def fake_install(pkgs, root):
        state["installed"].extend(pkgs)
        return True, ""
    monkeypatch.setattr(loop, "install_modules", fake_install)

    code = scaffold.run_new(tmp_path, Config(), _complex_stub(),
                            scaffold.Spec("x"), apply=True, complex=True,
                            verbose=False, auto_install=True)
    assert state["installed"] == ["pygame"]
    assert state["fix"] == 0, "a missing dep must not trigger code repair"
    assert code == 0


def test_new_complex_reports_missing_dependency_without_install(tmp_path, monkeypatch):
    from pedacito import scaffold

    state = {"fix": 0}
    monkeypatch.setattr(loop, "run_loop", lambda *a, **k: 0)
    monkeypatch.setattr(scaffold, "_first_import_error",
                        lambda root, plan: "ModuleNotFoundError: No module named 'pygame'")
    monkeypatch.setattr(scaffold, "fix_import_errors",
                        lambda *a, **k: state.__setitem__("fix", state["fix"] + 1) or 0)

    code = scaffold.run_new(tmp_path, Config(), _complex_stub(),
                            scaffold.Spec("x"), apply=True, complex=True,
                            verbose=False, auto_install=False)
    assert state["fix"] == 0, "a missing dep must not trigger code repair"
    assert code == 1


def test_new_complex_flags_still_broken_import(tmp_path, monkeypatch):
    """If the tree still cannot import after repair, the run must report a
    non-zero exit instead of quietly claiming success."""
    from pedacito import scaffold

    monkeypatch.setattr(loop, "run_loop", lambda *a, **k: 0)
    monkeypatch.setattr(scaffold, "fix_import_errors", lambda *a, **k: 0)
    monkeypatch.setattr(scaffold, "_first_import_error",
                        lambda root, plan: "ImportError: cannot import name 'x'")

    code = scaffold.run_new(tmp_path, Config(), _complex_stub(),
                            scaffold.Spec("x"), apply=True, complex=True,
                            verbose=False)
    assert code == 1


def test_fix_api_mismatches_repairs_a_call(tmp_path):
    """The API repair pass must turn a wrong-arity call into an applied edit,
    driving the tree back to consistency."""
    from pedacito import scaffold
    from pedacito import index as index_mod

    (tmp_path / "main.py").write_text(
        "from core import Game\n\ndef main():\n    Game()\n", encoding="utf-8")
    (tmp_path / "core.py").write_text(
        "class Game:\n    def __init__(self, x):\n        self.x = x\n",
        encoding="utf-8")
    idx = index_mod.Index(tmp_path, [])
    plan = scaffold.Plan("x", "y", files=[
        scaffold.PlannedFile("main.py"), scaffold.PlannedFile("core.py")],
        spec=scaffold.Spec("y"))

    class Stub:
        def chat(self, messages, max_tokens, schema=None, stream=False, **kw):
            return ("--- FILE: main.py\n<<<<<<< SEARCH\n    Game()\n"
                    "=======\n    Game(1)\n>>>>>>> REPLACE\n")

    n = scaffold.fix_api_mismatches(Stub(), Config(), idx, plan, None,
                                    verbose=False)
    assert n == 1
    assert "Game(1)" in (tmp_path / "main.py").read_text()
    from pedacito import apicheck
    assert apicheck.find_mismatches(
        scaffold._py_sources(tmp_path, ["main.py", "core.py"])) == []


# ------------------------------------------------------- runtime verification
def test_smoke_run_clean_exit_passes(tmp_path):
    from pedacito import scaffold
    (tmp_path / "main.py").write_text("print('ok')\n", encoding="utf-8")
    ok, _ = scaffold.smoke_run(tmp_path, "main.py", timeout=10)
    assert ok


def test_smoke_run_crash_fails_with_traceback(tmp_path):
    from pedacito import scaffold
    (tmp_path / "main.py").write_text("raise SystemExit('boom')\n",
                                      encoding="utf-8")
    ok, out = scaffold.smoke_run(tmp_path, "main.py", timeout=10)
    assert not ok and "boom" in out


def test_smoke_run_timeout_means_started(tmp_path):
    """A game/server that keeps running is a SUCCESS: it started."""
    from pedacito import scaffold
    (tmp_path / "main.py").write_text("import time\ntime.sleep(5)\n",
                                      encoding="utf-8")
    ok, out = scaffold.smoke_run(tmp_path, "main.py", timeout=1)
    assert ok and "still running" in out


def test_verify_and_fix_repairs_smoke_failure(tmp_path, monkeypatch):
    """A smoke-run crash is fed back to the model and the check retried."""
    from pedacito import scaffold
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    idx = types.SimpleNamespace(root=tmp_path)
    calls = {"n": 0}

    def fake_smoke(root, entry="main.py", timeout=20):
        calls["n"] += 1
        return (calls["n"] >= 2), ("TypeError: boom" if calls["n"] < 2 else "")

    fixes = []
    monkeypatch.setattr(scaffold, "smoke_run", fake_smoke)
    monkeypatch.setattr(scaffold, "_fix_from_output",
                        lambda *a, **k: fixes.append(1) or 1)
    ok, _ = scaffold.verify_and_fix(None, Config(), idx, ["main.py"], "x", None,
                                    smoke=True, rounds=2)
    assert ok
    assert len(fixes) == 1, "a failing smoke run must trigger a repair round"


def test_verify_and_fix_ignores_headless_display_error(tmp_path, monkeypatch):
    """No display is the machine's problem, not the code's: do not repair it,
    do not fail the run."""
    from pedacito import scaffold
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    idx = types.SimpleNamespace(root=tmp_path)
    monkeypatch.setattr(scaffold, "smoke_run",
                        lambda root, entry="main.py", timeout=20: (
                            False, "pygame.error: No available video device"))
    fixes = []
    monkeypatch.setattr(scaffold, "_fix_from_output",
                        lambda *a, **k: fixes.append(1) or 1)
    ok, out = scaffold.verify_and_fix(None, Config(), idx, ["main.py"], "x", None,
                                      smoke=True)
    assert ok and fixes == []
    assert "environment" in out


def test_verify_and_fix_does_not_code_repair_missing_dependency(tmp_path, monkeypatch):
    from pedacito import scaffold
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    idx = types.SimpleNamespace(root=tmp_path)
    monkeypatch.setattr(scaffold, "smoke_run",
                        lambda root, entry="main.py", timeout=20: (
                            False, "ModuleNotFoundError: No module named 'pygame'"))
    fixes = []
    monkeypatch.setattr(scaffold, "_fix_from_output",
                        lambda *a, **k: fixes.append(1) or 1)
    ok, _ = scaffold.verify_and_fix(None, Config(), idx, ["main.py"], "x", None,
                                    smoke=True)
    assert not ok
    assert fixes == [], "a missing dep must not be sent to the repair loop"


# ------------------------------------------------- custom answer brief + format
def test_custom_answer_system_still_carries_edit_format(tmp_path):
    """A custom answer brief (scaffold stub implementation, resume) must still
    carry the exact search/replace FORMAT, or a small model emits fenced
    blocks without a `--- FILE:` header and the parser finds nothing."""
    from pedacito.agent import Agent, ANSWER_SYSTEM, EDIT_FORMAT

    captured = {}

    class StubClient:
        stream_truncated = False
        step_truncated = False
        def chat(self, messages, max_tokens, schema=None, stream=False, **kw):
            captured["system"] = messages[0]["content"]
            return "ok"

    idx = types.SimpleNamespace(
        root=tmp_path,
        map_for_prompt=lambda cfg: ("MAP", "flat"),
    )
    cfg = Config()
    agent = Agent(idx, StubClient(), cfg)
    agent.answer("do a thing", "some source", system="IMPLEMENT THE STUBS",
                 stream=False)
    sys_prompt = captured["system"]
    assert "IMPLEMENT THE STUBS" in sys_prompt
    assert "--- FILE:" in sys_prompt
    assert ">>>>>>> REPLACE" in sys_prompt
    # The default brief is unchanged: intro + the shared format.
    assert ANSWER_SYSTEM.startswith("You are a Python code assistant")
    assert ANSWER_SYSTEM.endswith(EDIT_FORMAT)
