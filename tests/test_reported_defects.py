"""Regression tests for the externally reported defect list (Nov 2026 triage).

Each section pins the fixed behavior of one reported defect; the docstrings
name the report entry so the table in the validation report and the suite stay
linked. The companion agent-side sections (nested-symbol anchors, forced-read
matching, repeated reads / NEED) live at the bottom.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import cli, config as config_mod, fileio, index as index_mod
from pedacito import llm as llm_mod
from pedacito import loop, tools, workspace
from pedacito.agent import Agent
from pedacito.config import Config
from pedacito.edits import Result
from pedacito.llm import LLMClient


def _cfg(**kw) -> Config:
    c = Config()
    c.log_sessions = False
    c.static_gate = False
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _pw(path, text: str) -> None:
    """Write like the loop does (atomic replace) so ratchet hardlinks hold."""
    fileio.write(Path(path), text, fileio.TextStyle())


# ------------------------------------------------- bug 1: silent partial edits
def test_partial_edit_reported_at_task_and_run_level(tmp_path, monkeypatch,
                                                      capsys):
    """A task whose reply applies SOME blocks but fails others stays ticked
    (by design, see test_loop.py's strict test), but the unapplied blocks must
    surface: at the moment it happens, in the run log, and in the finish line."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal", ["Patch two files"])
    idx = types.SimpleNamespace(root=tmp_path)
    cfg = _cfg()

    def fake_run_task(root, cfg_, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None, is_repair=False):
        _pw(root / "a.py", "GOOD = 1\n")
        return loop.TaskResult(
            [Result(True, "a.py", "applied"),
             Result(False, "b.py", "SEARCH block did not match")],
            "done", True, ["a.py"], idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    monkeypatch.setattr(loop, "run_verify", lambda root, cmd: (True, ""))
    code = loop.run_loop(tmp_path, cfg, None, idx, plan, apply=True,
                         verify="pytest -q")
    assert code == 0                       # unchanged: still not a failure
    err = capsys.readouterr().err
    assert "partial edit" in err and "b.py" in err
    log = workspace.run_log_path(tmp_path).read_text(encoding="utf-8")
    assert "partial edit on task" in log


def test_partial_edit_final_summary_mentions_partial_tasks(tmp_path,
                                                            monkeypatch,
                                                            capsys):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal", ["Patch two files"])
    idx = types.SimpleNamespace(root=tmp_path)
    cfg = _cfg()

    def fake_run_task(root, cfg_, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None, is_repair=False):
        _pw(root / "a.py", "GOOD = 1\n")
        return loop.TaskResult(
            [Result(True, "a.py", "applied"),
             Result(False, "b.py", "SEARCH block did not match")],
            "done", True, ["a.py"], idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    monkeypatch.setattr(loop, "run_verify", lambda root, cmd: (True, ""))
    loop.run_loop(tmp_path, cfg, None, idx, plan, apply=True,
                  verify="pytest -q")
    err = capsys.readouterr().err
    assert "ticked done with blocks that did not apply" in err
    log = workspace.run_log_path(tmp_path).read_text(encoding="utf-8")
    assert "partially applied task" in log


# --------------------------------------- bug 3: verify_final (whole-goal mode)
def _two_task_ratchet(tmp_path, monkeypatch):
    """Plan where each task fixes one half; the check only passes with BOTH.
    Returns the list of run_verify call counts."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal", ["Fix module one", "Fix module two"])
    idx = types.SimpleNamespace(root=tmp_path)
    cfg = _cfg()

    def fake_run_task(root, cfg_, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None, is_repair=False):
        title = task.split("\n\n")[0]
        if title == "Fix module one":
            _pw(root / "a.py", "OK\n")
            changed = ["a.py"]
        else:
            _pw(root / "b.py", "OK\n")
            changed = ["b.py"]
        return loop.TaskResult([Result(True, "x.py", "applied")],
                               "done", True, changed, idx_)

    calls: list[str] = []

    def fake_verify(root, cmd):
        calls.append(cmd)
        ok = ((root / "a.py").is_file() and (root / "b.py").is_file())
        return ok, "" if ok else "half the fix is missing"

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    monkeypatch.setattr(loop, "run_verify", fake_verify)
    return plan, idx, cfg, calls


def test_per_task_verify_rolls_back_valid_partial_fix(tmp_path, monkeypatch):
    """DOCUMENTS THE REPORTED DEFECT: with the default per-task gate, task 1's
    valid half is rolled back because the whole-goal check cannot pass yet."""
    plan, idx, cfg, calls = _two_task_ratchet(tmp_path, monkeypatch)
    code = loop.run_loop(tmp_path, cfg, None, idx, plan, apply=True,
                         verify="pytest -q", repair_rounds=0,
                         stop_on_fail=False)
    assert calls == ["pytest -q", "pytest -q"]      # gate ran after EVERY task
    assert code == 1
    assert not (tmp_path / "a.py").exists()          # valid work undone
    assert not (tmp_path / "b.py").exists()


def test_verify_final_defers_the_shell_check_to_the_last_task(tmp_path,
                                                              monkeypatch,
                                                              capsys):
    """Opt-in mode: cheap gates stay per-step, the shell check runs once the
    plan's LAST unchecked task lands -- so partial valid fixes accumulate."""
    plan, idx, cfg, calls = _two_task_ratchet(tmp_path, monkeypatch)
    code = loop.run_loop(tmp_path, cfg, None, idx, plan, apply=True,
                         verify="pytest -q", repair_rounds=0,
                         stop_on_fail=False, verify_final=True)
    assert calls == ["pytest -q"]                    # ran once, on the last task
    assert code == 0
    assert (tmp_path / "a.py").is_file() and (tmp_path / "b.py").is_file()
    text = plan.read_text()
    assert "- [x] Fix module one" in text and "- [x] Fix module two" in text
    err = capsys.readouterr().err
    assert "final-only" in err                       # the deferral is announced


def test_verify_final_rolls_back_only_the_last_step_when_it_fails(tmp_path,
                                                                  monkeypatch):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal", ["Fix module one", "Break module two"])
    idx = types.SimpleNamespace(root=tmp_path)
    cfg = _cfg()

    def fake_run_task(root, cfg_, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None, is_repair=False):
        title = task.split("\n\n")[0]
        if title == "Fix module one":
            _pw(root / "a.py", "GOOD = 1\n")
            changed = ["a.py"]
        else:
            _pw(root / "b.py", "BAD = wrecked\n")
            changed = ["b.py"]
        return loop.TaskResult([Result(True, "x.py", "applied")],
                               "done", True, changed, idx_)

    calls: list[str] = []

    def fake_verify(root, cmd):
        calls.append(cmd)
        ok = not (root / "b.py").exists()
        return ok, "" if ok else "b.py is broken"

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    monkeypatch.setattr(loop, "run_verify", fake_verify)
    code = loop.run_loop(tmp_path, cfg, None, idx, plan, apply=True,
                         verify="pytest -q", repair_rounds=0,
                         stop_on_fail=False, verify_final=True)
    assert calls == ["pytest -q"]                    # only the real final check
    assert code == 1
    assert (tmp_path / "a.py").read_text() == "GOOD = 1\n"   # step 1 survives
    assert not (tmp_path / "b.py").exists()                   # step 2 rolled back
    text = plan.read_text()
    assert "- [x] Fix module one" in text
    assert "- [!] Break module two" in text


def test_verify_final_runs_the_check_for_the_record_on_early_stop(tmp_path,
                                                                  monkeypatch):
    """max_tasks stopped the loop before the last task: the whole-goal check
    must still be paid for ONCE, and reported (no silent green exit)."""
    plan, idx, cfg, calls = _two_task_ratchet(tmp_path, monkeypatch)
    code = loop.run_loop(tmp_path, cfg, None, idx, plan, apply=True,
                         verify="pytest -q", max_tasks=1,
                         stop_on_fail=False, verify_final=True)
    assert len(calls) == 1        # deferred mid-run, then once for the record
    log = workspace.run_log_path(tmp_path).read_text(encoding="utf-8")
    assert "final-only" in log    # and the log says why it ran there


# -------------------------------- bug 4: ad-hoc backgrounding in `run` is inert
def test_run_rejects_shell_backgrounding(tmp_path):
    """`run` executes argv WITHOUT a shell: `&` / `nohup` do not background,
    and a quiet server dies to the idle kill. The error must redirect to the
    supported registered-process flow (`serve`, `.pedacito/procs`)."""
    cfg = _cfg()
    cfg.allow_shell = True
    cfg.shell_allowlist = ()
    out = tools.tool_run(types.SimpleNamespace(root=tmp_path), cfg,
                         "python3 -m http.server 8000 &")
    assert out.startswith("ERROR")
    assert "serve" in out and "procs" in out
    out2 = tools.tool_run(types.SimpleNamespace(root=tmp_path), cfg,
                          "nohup python3 -m http.server 8000")
    assert out2.startswith("ERROR") and "serve" in out2
    out3 = tools.tool_run(types.SimpleNamespace(root=tmp_path), cfg,
                          "setsid sleep 30")
    assert out3.startswith("ERROR") and "serve" in out3


def test_run_still_runs_ordinary_commands(tmp_path):
    """The backgrounding guard must not swallow quoted ampersands or normal
    invocations."""
    cfg = _cfg()
    cfg.allow_shell = True
    cfg.shell_allowlist = ()
    _pw(tmp_path / "main.py", "print('x & y')\n")
    out = tools.tool_run(types.SimpleNamespace(root=tmp_path), cfg,
                         f'{sys.executable} -c "print(chr(38))"')
    assert "&" in out and not out.startswith("ERROR: `run` cannot background")


# ------------------------------------- bug 5: answer-phase repetition loops
class Resp:
    def __init__(self, status=200, body=None, text="", lines=None,
                 headers=None, raise_after=None):
        self.status_code = status
        self._body = body
        self.text = text or (json.dumps(body) if body is not None else "")
        self._lines = lines or []
        self.headers = headers or {}
        self.closed = False
        self._raise_after = raise_after

    def json(self):
        return self._body

    def iter_lines(self, decode_unicode=False):
        yield from self._lines
        if self._raise_after is not None:
            raise self._raise_after

    def close(self):
        self.closed = True


def _sse(obj):
    return "data: " + json.dumps(obj)


def _chunk(text="", finish=None):
    return _sse({"choices": [{"delta": {"content": text} if text else {},
                              "finish_reason": finish}]})


def _llm_cfg():
    cfg = Config()
    cfg.base_url = "http://127.0.0.1:1/v1"
    cfg.model = "m"
    return cfg


_PARA = ("The cache opens before the socket is validated, so a late timeout "
         "never reaches the retry path and the request hangs there. ")


def test_repetition_loop_stops_the_stream(monkeypatch, capsys):
    """A degenerate paragraph loop no longer streams until the context dies:
    the guard cuts the reply, flags it truncated, and says so."""
    lines = [_chunk(_PARA) for _ in range(12)] + [_chunk(finish="stop")]
    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **k:
                        Resp(200, lines=lines))
    client = LLMClient(_llm_cfg())
    text = client.chat([{"role": "user", "content": "explain"}], 8000,
                       stream=True)
    assert client.stream_repetition is True
    assert client.stream_truncated is True       # re-ask machinery engages
    assert len(text) < len(_PARA) * 12           # the loop was cut, not streamed
    err = capsys.readouterr().err
    assert "repeating" in err


def test_repetition_guard_leaves_varied_streams_alone(monkeypatch):
    """Non-repeating prose (and short refrain-like cycles under the floor)
    must stream through untouched."""
    lines = [_chunk(f"Observation {i}: the value differs here. ")
             for i in range(60)] + [_chunk(finish="stop")]
    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **k:
                        Resp(200, lines=lines))
    client = LLMClient(_llm_cfg())
    text = client.chat([{"role": "user", "content": "list"}], 8000,
                       stream=True)
    assert client.stream_repetition is False
    assert client.stream_truncated is False
    assert text.startswith("Observation 0:") and "Observation 59" in text


# ------------------------- bug 6: the goal survives write_plan and a resume
_GOAL = ("Build a report generator.\n\n"
         "Constraints:\n"
         "- no third-party deps\n"
         "- python 3.11 only\n\n"
         + "Export CSV and JSON. " * 30)      # past the old 300/char cut


def test_write_plan_persists_the_full_goal(tmp_path):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, _GOAL, ["Task one", "Task two"])
    text = plan.read_text()
    assert "Export CSV and JSON." in text          # nothing truncated
    # continuation lines never parse as tasks:
    unchecked = loop.next_task(loop.read_plan(plan))
    assert unchecked and unchecked[1].startswith("Task one")
    assert loop.read_plan_goal(plan).strip() == _GOAL.strip()


def test_write_plan_single_line_goal_format_unchanged(tmp_path):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "build a todo CLI", ["x"])
    assert "Goal: build a todo CLI" in plan.read_text(encoding="utf-8")
    assert loop.read_plan_goal(plan) == "build a todo CLI"


def test_auto_resume_carries_the_persisted_goal(tmp_path, monkeypatch):
    """`pedacito auto` resuming a plan with no -t/--spec must hand the workers
    the ORIGINAL goal again -- read it back from the plan file."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, _GOAL, ["Task one", "Task two"])
    seen: dict = {}

    def fake_run_loop(root, cfg_, client, idx, plan_path, **kw):
        seen.update(kw)
        return 0

    monkeypatch.setattr(loop, "run_loop", fake_run_loop)
    args = types.SimpleNamespace(task=None, spec=None, refine=False,
                                 force=False, apply=True, plan=None,
                                 verify=None, repair=None, max_tasks=None,
                                 max_minutes=None, auto_install=False,
                                 keep_going=False, strict=False,
                                 worktree="", diagnostics=False)
    cfg = _cfg()
    idx = types.SimpleNamespace(root=tmp_path)
    rc = cli._cmd_auto(args, cfg, None, idx, tmp_path)
    assert rc == 0
    assert "Export CSV and JSON." in seen.get("goal", "")


# ------------------------------------------- bug 2: nested/ambiguous symbols
def _files(tmp_path, spec: dict[str, str]):
    for rel, text in spec.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    cfg = _cfg()
    idx = index_mod.build([str(tmp_path)], cfg, client=None, summarise=False,
                          root=tmp_path, verbose=False)
    return Agent(idx, None, cfg), cfg, idx


def test_unindexed_nested_symbol_falls_back_to_whole_file(tmp_path):
    """The chunkers deliberately index only top-level/class-member names, so
    a def nested in a def or a function inside an IIFE has no symbol anchor.
    The intent must fall back to the whole-file target (plain quoted hunks)
    instead of failing the edit with 'no definition by that name'."""
    agent, _cfg_, idx = _files(tmp_path, {
        "a.py": "def outer():\n    def helper():\n        return 1\n"
                "    return helper()\n",
        "app.js": "(function () {\n  function greet(name) {\n"
                  "    return 'hi ' + name;\n  }\n  greet('x');\n})();\n"})
    assert idx.find_symbol("helper") == []       # (confirms the index gap)
    assert idx.find_symbol("greet") == []
    target, failure = agent._resolve_edit_intent(
        {"file": "a.py", "symbol": "helper", "intended_change": "x"})
    assert failure is None and target is not None
    assert target["file"] == "a.py" and target["symbol"] == ""
    assert target["start"] == 1 and target["end"] == 4
    target2, failure2 = agent._resolve_edit_intent(
        {"file": "app.js", "symbol": "greet", "intended_change": "y"})
    assert failure2 is None and target2 is not None
    assert target2["symbol"] == "" and target2["end"] >= 5


def test_truly_absent_symbol_still_fails(tmp_path):
    """The fallback is for EXISTING files only: an intent whose file does not
    exist and whose symbol cannot resolve anywhere stays a hard failure."""
    agent, _cfg_, idx = _files(tmp_path, {"a.py": "def top():\n    return 1\n"})
    target, failure = agent._resolve_edit_intent(
        {"file": "missing.py", "symbol": "ghost", "intended_change": "x"})
    assert target is None and failure is not None


# ------------------------------------------ bug 7: forced-read path matching
def test_full_path_token_does_not_match_same_named_root_file(tmp_path):
    """A task naming `web/app.py` used to ALSO force-read a root `app.py`
    (the reverse suffix match), and once web/app.py had been read in full
    the stale nudge about app.py never cleared."""
    agent, cfg, idx = _files(tmp_path, {
        "app.py": "import os\nX = 1\n",
        "web/app.py": "def page():\n    return 2\n"})
    agent._read_spans = {"web/app.py": [(1, 3)]}   # the real target, fully read
    missing = agent._compute_unread("Fix the timeout in web/app.py")
    assert missing == []


def test_bare_name_token_still_forces_every_same_named_file(tmp_path):
    """The short->deep direction is kept: `app.py` in a task title names both
    the root file and `web/app.py`, and unread ones keep the nudge."""
    agent, cfg, idx = _files(tmp_path, {
        "app.py": "import os\nX = 1\n",
        "web/app.py": "def page():\n    return 2\n"})
    agent._read_spans = {}
    missing = agent._compute_unread("Tidy app.py")
    assert any(m.startswith("app.py ") for m in missing)
    assert any(m.startswith("web/app.py ") for m in missing)


def test_prefetch_named_target_shares_the_single_match_rule(tmp_path):
    """The post-gather prefetch trigger matched with the same bidirectional
    rule; it must not chase the root `app.py` once the named deep file was
    read. (Regression guard against re-introducing a second copy.)"""
    agent, cfg, idx = _files(tmp_path, {
        "app.py": "import os\nX = 1\n",
        "web/app.py": "def page():\n    return 2\n"})
    src = inspect.getsource(Agent._file_token_match)
    assert 'r == w or r.endswith("/" + w)' in src
    # the reverse rule must appear nowhere in the coverage bookkeeping code:
    body = inspect.getsource(Agent._compute_unread) \
        + inspect.getsource(Agent._post_gather_prefetch) \
        + inspect.getsource(Agent._prefetch_step0)
    assert 'endswith("/" + r)' not in body


# ------------------------------- bug 8: repeated reads and NEED in answers
class _StepClient:
    """Pops scripted gather steps; answers via a plain callable."""
    stream_truncated = False
    stream_repetition = False
    step_truncated = False

    def __init__(self, steps):
        self.steps = list(steps)
        self.json_calls = 0
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def chat_json(self, messages, schema, max_tokens, **kw):
        self.json_calls += 1
        return self.steps.pop(0) if self.steps else {"action": "done"}

    def chat(self, messages, max_tokens, **kw):
        raise AssertionError("answer phase should not run in this test")


def test_repeated_identical_reads_are_cut_off(tmp_path):
    """Reported symptom: the same `read` re-ran dozens of times. The gather
    loop counts exact repeats and stops (this test mirrors the existing
    lock-in and is the answer-phase counterpart to bug 8's report)."""
    _files(tmp_path, {"big.py": "x = 1\n" * 40})
    cfg = _cfg()
    idx = index_mod.build([str(tmp_path)], cfg, client=None, summarise=False,
                          root=tmp_path, verbose=False)
    read = {"action": "read", "file": "big.py", "start": 1, "end": 10}
    client = _StepClient([read] * 30)
    agent = Agent(idx, client, cfg)
    gathered = agent.gather("What does big.py contain?", verbose=False)
    assert client.json_calls < 10, (
        f"gather ran the identical read {client.json_calls} times")
    assert "x = 1" in gathered


def test_need_line_in_answer_refetches_from_disk(tmp_path):
    """Reported symptom: the answer says `NEED file:lines` instead of the
    code. That directive is honoured IN the answer phase: the harness
    re-reads the span from disk and re-asks, bounded by
    max_rehydration_rounds. (test_evidence_rehydration.py locks the
    mechanism; this is the defect-report-side regression.)"""
    agent, cfg, idx = _files(
        tmp_path, {"loop.py": "for i in range(3):\n    print(i)\n"})
    calls: list[list[dict]] = []

    def chat(msgs):
        calls.append([dict(m) for m in msgs])
        if len(calls) == 1:
            return "Let me quote it.\nNEED loop.py:1-2"
        return "FINAL: prints 0, 1, 2"

    msgs = [{"role": "system", "content": "SYS"},
            {"role": "user", "content": "what does loop.py do?"}]
    reply = agent._chat_needs(chat, msgs)
    assert reply == "FINAL: prints 0, 1, 2"
    assert len(calls) == 2
    assert "for i in range(3):" in msgs[-1]["content"]
    assert not any("NEED path/to/file" in m["content"] for m in msgs[1:])


# -------------------------------------------------- bug 9a: XDG config hint
@pytest.mark.skipif(
    os.name == "nt",
    reason="XDG_CONFIG_HOME is POSIX-only; Windows uses %APPDATA%")
def test_profile_error_names_xdg_ignored_config(tmp_path, monkeypatch):
    """With XDG_CONFIG_HOME set, ~/.config/pedacito/config.toml exists but is
    ignored -- the error must say so, or the user edits a dead file."""
    home = tmp_path / "home"
    xdg = tmp_path / "xdg"
    (home / ".config" / "pedacito").mkdir(parents=True)
    dead = home / ".config" / "pedacito" / "config.toml"
    dead.write_text("[profiles.big]\nmodel = 'm'\n", encoding="utf-8")
    xdg.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.setattr(config_mod, "USER_CONFIG",
                        xdg / "pedacito" / "config.toml")
    with pytest.raises(config_mod.ConfigError) as ei:
        config_mod.load(root=tmp_path, profile="ghost")
    msg = str(ei.value)
    assert "Looked in" in msg
    assert str(dead) in msg and "XDG_CONFIG_HOME" in msg
