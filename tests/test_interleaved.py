"""The interleaved agent loop (pedacito/interleaved.py).

The model is always a scripted double: `ScriptedTools` speaks the native
`chat_tools` channel, `JsonChat` only has `chat()` (the JSON text protocol),
and a real `LLMClient` with a patched transport covers the OpenAI-native and
Anthropic-native request builders.
"""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import events as events_mod
from pedacito import index as index_mod
from pedacito import interleaved, review, security
from pedacito.agent import Agent
from pedacito.config import Config
from pedacito.interleaved import (InterleavedResult, build_tool_schemas,
                                  parse_text_tool_calls, run_interleaved)
from pedacito.llm import LLMClient, LLMError

PRICING = (
    "def bulk_discount(quantity):\n"
    "    if quantity > 10:\n"
    "        return 0.1\n"
    "    return 0.0\n"
    "\n"
    "\n"
    "def total(price, quantity):\n"
    "    return price * quantity * (1 - bulk_discount(quantity))\n"
)
TEST_PRICING = (
    "from pricing import bulk_discount, total\n"
    "\n"
    "\n"
    "def test_threshold():\n"
    "    assert bulk_discount(10) == 0.1\n"
    "\n"
    "\n"
    "def test_total():\n"
    "    assert total(10, 10) == 90.0\n"
)
CALC = (
    "def old():\n"
    "    return undefined_before()\n"
    "\n"
    "\n"
    "def add(a, b):\n"
    "    return a + b\n"
)
BASE_FILES = {
    "pricing.py": PRICING,
    "conftest.py": "",
    "tests/test_pricing.py": TEST_PRICING,
    "calc.py": CALC,
}


def make_agent(tmp_path, client, files=None, cfg=None, **cfg_attrs):
    root = tmp_path / "proj"
    root.mkdir(parents=True, exist_ok=True)
    for rel, text in (BASE_FILES if files is None else files).items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    cfg = cfg or Config()
    cfg.log_events = False
    cfg.log_sessions = False
    cfg.sources = {}
    for k, v in cfg_attrs.items():
        setattr(cfg, k, v)
    idx = index_mod.build([str(root)], cfg, client=None, summarise=False,
                          root=root, verbose=False)
    return root, Agent(idx, client, cfg)


class ScriptedTools:
    """A model double on the native `chat_tools` channel. Each step is a list
    of `(tool, arguments)` calls, a dict (`calls`, `content`, `usage`), a bare
    string (JSON text), or a callable `(messages, client) -> step`. Every
    request is recorded (deep-copied) in `self.requests`."""

    def __init__(self, steps, *, loop_last=False):
        self.steps = list(steps)
        self.requests: list[tuple[list, list]] = []
        self.loop_last = loop_last

    def chat_tools(self, messages, tools, max_tokens, **kw):
        n = len(self.requests)
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        if n < len(self.steps):
            step = self.steps[n]
        elif self.loop_last:
            step = self.steps[-1]
        else:
            step = [("finish", {"summary": "script exhausted"})]
        if callable(step):
            step = step(messages, self)
        if isinstance(step, str):
            return step
        usage = {"completion_tokens": 20}
        content = ""
        if isinstance(step, dict):
            usage = step.get("usage", usage)
            content = step.get("content", "")
            step = step.get("calls", [])
        return {"content": content, "usage": usage, "tool_calls": [
            {"id": f"call_{n}_{i}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}
            for i, (name, args) in enumerate(step)]}


class JsonChat:
    """A model double with only `chat()`: the JSON text protocol."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests: list[list] = []

    def chat(self, messages, max_tokens, **kw):
        n = len(self.requests)
        self.requests.append(copy.deepcopy(messages))
        if n < len(self.replies):
            return self.replies[n]
        return '{"tool": "finish", "arguments": {"summary": "done"}}'


def tool_msgs(result: InterleavedResult, name: str | None = None) -> list[str]:
    return [m["content"] for m in result.transcript
            if m["role"] == "tool" and (name is None or m.get("name") == name)]


def tree(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): (p.read_bytes() if p.is_file()
                                             else None)
            for p in sorted(root.rglob("*"))}


def edit(file, search, replace, **kw):
    return ("edit_file", {"file": file, "search": search, "replace": replace,
                          **kw})


# =============================================================== the flow
FIX_FLOW = [
    [("read", {"file": "pricing.py"})],
    [edit("pricing.py", "    if quantity > 10:\n        return 0.1",
          "    if quantity >= 10:\n        return 0.2")],
    [("run_test", {})],
    [("read", {"file": "pricing.py", "start_line": 1, "end_line": 4})],
    [edit("pricing.py", "return 0.2", "return 0.1")],
    [("run_test", {"target": "tests/test_pricing.py"})],
    [("finish", {"summary": "bulk_discount now applies from 10 units"})],
]


def test_read_edit_test_traceback_edit_test_pass_finish_applied(tmp_path):
    client = ScriptedTools(FIX_FLOW)
    root, agent = make_agent(tmp_path, client)
    res = run_interleaved(agent, "fix the bulk discount threshold", apply=True)
    assert res.ok, res.error
    assert res.stop_reason == "finish"
    assert res.summary == "bulk_discount now applies from 10 units"
    assert res.turns == 7 and res.tool_calls == 7
    assert res.changed_files == ["pricing.py"]
    assert res.tests_run == 2 and res.tests_green is True
    assert "if quantity >= 10:" in (root / "pricing.py").read_text(encoding="utf-8")
    assert "return 0.2" not in (root / "pricing.py").read_text(encoding="utf-8")
    # the traceback of the wrong first attempt reached the model on turn 4
    first_test = tool_msgs(res, "run_test")[0]
    assert "FAILED" in first_test and "assert 0.2 == 0.1" in first_test
    assert "OK" in tool_msgs(res, "run_test")[1]
    # and so did the fresh numbered source, including its own edit
    assert ">= 10" in tool_msgs(res, "read")[1]
    # results are engine objects usable by review
    assert res.edit_results and res.edit_results[0].file == "pricing.py"
    assert [fd.rel for fd in res.file_diffs] == ["pricing.py"]
    assert res.applied and res.workspace.apply


def test_dry_run_stages_edits_but_never_touches_disk(tmp_path):
    client = ScriptedTools(FIX_FLOW)
    root, agent = make_agent(tmp_path, client)
    before = tree(root)
    res = run_interleaved(agent, "fix the bulk discount threshold", apply=False)
    assert res.ok, res.error
    assert tree(root) == before                        # 100% untouched
    assert not res.applied and not res.workspace.apply
    # the second read saw the STAGED edit; the tests ran against it too
    assert ">= 10" in tool_msgs(res, "read")[1]
    assert "assert 0.2 == 0.1" in tool_msgs(res, "run_test")[0]
    assert res.tests_green is True
    # the workspace replays the staged work as review rows
    assert res.changed_files == ["pricing.py"]
    (fd,) = res.file_diffs
    assert isinstance(fd, review.FileDiff) and (fd.added, fd.removed) == (1, 1)
    assert "if quantity >= 10:" in fd.after_text
    assert "return 0.1" in fd.after_text and "0.2" not in fd.after_text
    assert res.workspace.read("pricing.py") == fd.after_text
    assert "(dry run)" in res.edit_results[0].message


def test_review_flag_forces_the_overlay_and_signoff_applies_it(tmp_path):
    client = ScriptedTools(FIX_FLOW)
    root, agent = make_agent(tmp_path, client)
    before = tree(root)
    res = run_interleaved(agent, "fix it", apply=True, review=True)
    assert not res.applied and tree(root) == before
    ok, msg = review.write_files(root, res.file_diffs, tmp_path / "bk")
    assert ok, msg
    assert "if quantity >= 10:" in (root / "pricing.py").read_text(encoding="utf-8")
    assert (tmp_path / "bk" / "pricing.py").read_text(encoding="utf-8") == PRICING


def test_events_and_transcript_are_structured(tmp_path):
    client = ScriptedTools(FIX_FLOW)
    root, agent = make_agent(tmp_path, client)
    log = events_mod.EventLog(root, enabled=False)
    seen_by_log = []
    log.listeners.append(lambda kind, f: seen_by_log.append(kind))
    agent.events = log
    got: list[tuple[str, dict]] = []
    res = run_interleaved(agent, "fix it", apply=False,
                          on_event=lambda k, f: got.append((k, f)))
    kinds = [k for k, _ in got]
    assert kinds[0] == "interleaved_turn"
    assert {"interleaved_turn", "tool_call", "tool_result", "edit_staged",
            "finish"} <= set(kinds)
    assert kinds[-1] == "finish"
    assert seen_by_log == kinds                       # agent.events gets them too
    staged = [f for k, f in got if k == "edit_staged"]
    assert [f["file"] for f in staged] == ["pricing.py", "pricing.py"]
    assert staged[0]["applied"] is False and staged[0]["added"] == 2
    calls = [f for k, f in got if k == "tool_call"]
    assert [c["name"] for c in calls] == [c[0][0] for c in FIX_FLOW]
    done = got[-1][1]
    assert done["ok"] is True and done["stop_reason"] == "finish"
    assert done["changed_files"] == ["pricing.py"]
    # transcript: ChatML/OpenAI shape, JSON-serialisable, ids paired
    json.dumps(res.transcript)
    roles = [m["role"] for m in res.transcript]
    assert roles[:2] == ["system", "user"]
    assert set(roles) == {"system", "user", "assistant", "tool"}
    ids = [tc["id"] for m in res.transcript if m["role"] == "assistant"
           for tc in m["tool_calls"]]
    assert ids == [m["tool_call_id"] for m in res.transcript
                   if m["role"] == "tool"]
    assert all("name" in m for m in res.transcript if m["role"] == "tool")


def test_run_interleaved_never_raises_on_model_failure(tmp_path):
    class Boom:
        def chat_tools(self, *a, **k):
            raise LLMError("server exploded")

    root, agent = make_agent(tmp_path, Boom())
    res = run_interleaved(agent, "x", apply=False)
    assert not res.ok and res.stop_reason == "error"
    assert "server exploded" in res.error
    assert res.turns == 1 and res.changed_files == []


# ================================================================== gates
SYNTAX_FLOW = [
    [("read", {"file": "calc.py"})],
    [edit("calc.py", "return a + b", "return a +")],
    [edit("calc.py", "return a + b", "return a + b + 1")],
    [("finish", {"summary": "added one"})],
]


@pytest.mark.parametrize("apply", [False, True])
def test_syntax_error_is_rejected_rolled_back_and_fixed_next_turn(tmp_path,
                                                                  apply):
    disk_after_bad_edit = {}

    def watch(messages, client):
        # the model's turn right after the rejected edit: the file on disk is
        # exactly what it was (dry run: never touched; apply: rolled back)
        disk_after_bad_edit["text"] = (tmp_path / "proj" / "calc.py").read_text(encoding="utf-8")
        return SYNTAX_FLOW[2]

    steps = [SYNTAX_FLOW[0], SYNTAX_FLOW[1], watch, SYNTAX_FLOW[3]]
    client = ScriptedTools(steps)
    root, agent = make_agent(tmp_path, client)
    got = []
    res = run_interleaved(agent, "add one", apply=apply,
                          on_event=lambda k, f: got.append((k, f)))
    bad = tool_msgs(res, "edit_file")[0]
    assert bad.startswith("ERROR") and "REJECTED" in bad
    assert "syntax error" in bad and "unchanged" in bad
    assert disk_after_bad_edit["text"] == CALC
    gate = [f for k, f in got if k == "gate_error"][0]
    assert gate["gate"] == "syntax" and gate["rolled_back"] is True
    # exactly one request spent repairing: the very next model call
    assert len(client.requests) == 4
    good = tool_msgs(res, "edit_file")[1]
    assert good.startswith(("STAGED", "APPLIED")) and "GATES" in good
    assert res.ok and res.changed_files == ["calc.py"]
    assert "a + b + 1" in res.workspace.read("calc.py")
    on_disk = (root / "calc.py").read_text(encoding="utf-8")
    assert ("a + b + 1" in on_disk) is apply


def test_defensive_gate_rolls_back_a_staged_edit(tmp_path, monkeypatch):
    """Even if the edit engine lets a broken file through, the post-stage
    syntax gate rolls the edit back (apply mode restores the bytes)."""
    real = interleaved._Gates.syntax

    def picky(self, rel, before, after):
        if "BOOM" in after:
            return ["boom: broken on purpose"]
        return real(self, rel, before, after)

    monkeypatch.setattr(interleaved._Gates, "syntax", picky)
    client = ScriptedTools([
        [edit("calc.py", "return a + b", "return a + b  # BOOM")],
        [("finish", {"summary": "gave up"})]])
    root, agent = make_agent(tmp_path, client)
    res = run_interleaved(agent, "x", apply=True)
    msg = tool_msgs(res, "edit_file")[0]
    assert "ROLLED BACK" in msg and "boom: broken on purpose" in msg
    assert (root / "calc.py").read_text(encoding="utf-8") == CALC
    assert res.changed_files == []


def test_pre_existing_syntax_problem_is_not_blamed(tmp_path):
    files = {"bad.py": "def ok():\n    return 1\n\n\ndef broken(:\n    pass\n"}
    client = ScriptedTools([
        [edit("bad.py", "return 1", "return 2")],
        [("finish", {"summary": "s"})]])
    root, agent = make_agent(tmp_path, client, files=files)
    res = run_interleaved(agent, "x", apply=False)
    out = tool_msgs(res, "edit_file")[0]
    # still broken, but it was before this edit too: applied, never rejected
    assert not out.startswith("ERROR") and "return 2" in res.workspace.read("bad.py")


def test_write_file_creation_with_syntax_error_is_aborted(tmp_path):
    client = ScriptedTools([
        [("write_file", {"file": "pkg/new.py", "content": "def f(:\n  pass\n"})],
        [("write_file", {"file": "pkg/new.py", "content": "def f():\n    return 1\n"})],
        [("finish", {"summary": "s"})]])
    root, agent = make_agent(tmp_path, client)
    res = run_interleaved(agent, "x", apply=False)
    first, second = tool_msgs(res, "write_file")
    assert first.startswith("ERROR") and "syntax error" in first
    assert second.startswith("STAGED")
    assert res.changed_files == ["pkg/new.py"]
    assert not (root / "pkg").exists()


F821_FLOW = [
    [edit("calc.py", "    return a + b", "    return json.dumps(a + b)")],
    [edit("calc.py", "def old():", "import json\n\n\ndef old():")],
    [("finish", {"summary": "json result"})],
]


@pytest.mark.parametrize("apply", [False, True])
def test_undefined_name_is_reported_immediately_and_fixed_next_turn(tmp_path,
                                                                    apply):
    client = ScriptedTools(F821_FLOW)
    root, agent = make_agent(tmp_path, client)
    got = []
    res = run_interleaved(agent, "x", apply=apply,
                          on_event=lambda k, f: got.append((k, f)))
    first, second = tool_msgs(res, "edit_file")
    assert first.startswith(("STAGED", "APPLIED"))        # stays in place
    assert "GATE ERROR" in first and "F821" in first and "`json`" in first
    assert "import json" in first                          # the hint
    assert "undefined_before" not in first                 # pre-existing: not blamed
    assert "GATES: syntax, undefined names" in second and "GATE ERROR" not in second
    gate = [f for k, f in got if k == "gate_error"]
    assert len(gate) == 1 and gate[0]["gate"] == "static"
    assert gate[0]["rolled_back"] is False
    assert res.ok and not res.gate_errors
    assert "json.dumps" in res.workspace.read("calc.py")


def test_pre_existing_undefined_names_never_block_an_edit(tmp_path):
    client = ScriptedTools([
        [edit("calc.py", "return a + b", "return a + b + 1")],
        [("finish", {"summary": "s"})]])
    root, agent = make_agent(tmp_path, client)
    res = run_interleaved(agent, "x", apply=False)
    assert "GATE ERROR" not in tool_msgs(res, "edit_file")[0]
    assert res.ok


def test_finish_is_refused_once_while_gate_errors_are_open(tmp_path):
    client = ScriptedTools([
        [edit("calc.py", "    return a + b", "    return json.dumps(a + b)")],
        [("finish", {"summary": "done?"})],
        [("finish", {"summary": "done anyway"})]])
    root, agent = make_agent(tmp_path, client)
    res = run_interleaved(agent, "x", apply=False)
    refused, final = tool_msgs(res, "finish")
    assert refused.startswith("NOT FINISHED") and "F821" in refused
    assert final == "Finished."
    assert res.stop_reason == "finish" and res.turns == 3
    assert not res.ok and "unresolved gate errors" in res.error
    assert any("F821" in e for e in res.gate_errors)
    assert res.summary == "done anyway"


FRONTEND = {
    "index.html": ('<html><body><div id="app"></div><div id="go"></div>'
                   '<script src="app.js"></script></body></html>\n'),
    "app.js": ("const el = document.getElementById('app');\n"
               "el.textContent = 'hi';\n"
               "const old = document.getElementById('legacy');\n"),
}


def test_frontend_contract_gate_blames_only_new_breakage(tmp_path):
    client = ScriptedTools([
        # touches the file that ALREADY has a broken '#legacy' query: no blame
        [edit("app.js", "'hi'", "'hello'")],
        # introduces a new query for an id the HTML never defines
        [edit("app.js", "'hello';", "'hello';\ndocument.getElementById('nope');")],
        [edit("app.js", "getElementById('nope')", "getElementById('go')")],
        [("finish", {"summary": "s"})]])
    root, agent = make_agent(tmp_path, client, files=FRONTEND)
    before = tree(root)
    got = []
    res = run_interleaved(agent, "x", apply=False,
                          on_event=lambda k, f: got.append((k, f)))
    first, second, third = tool_msgs(res, "edit_file")
    assert "GATE ERROR" not in first, first
    assert "GATE ERROR" in second and "#nope" in second
    assert "FrontendContractError" in second
    assert "GATE ERROR" not in third and "legacy" not in third
    assert [f["gate"] for k, f in got if k == "gate_error"] == ["static"]
    assert tree(root) == before
    assert res.ok


# =============================================================== containment
@pytest.mark.parametrize("apply", [False, True])
def test_path_traversal_and_git_are_refused_through_every_tool(tmp_path, apply):
    steps = [[
        ("read", {"file": "../outside.py"}),
        edit("../outside.py", "a", "b"),
        ("write_file", {"file": "../outside.py", "content": "x = 1\n"}),
        ("delete_file", {"file": "../outside.py"}),
        ("write_file", {"file": "sub/../../outside.py", "content": "x = 1\n"}),
        ("write_file", {"file": ".git/config", "content": "[evil]\n"}),
        ("delete_file", {"file": ".git/HEAD"}),
        ("read", {"file": ".git/config"}),
    ], [
        ("write_file", {"file": ".pedacito/index.json", "content": "{}"}),
        ("write_file", {"file": ".env", "content": "TOKEN=1\n"}),
        ("read", {"file": ".env"}),
    ], [("finish", {"summary": "s"})]]
    client = ScriptedTools(steps)
    root, agent = make_agent(tmp_path, client)
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (root / ".env").write_text("TOKEN=real\n", encoding="utf-8")
    before = tree(root)
    res = run_interleaved(agent, "x", apply=apply)
    outs = [m["content"] for m in res.transcript if m["role"] == "tool"][:-1]
    assert len(outs) == 8 + 3
    assert [o for o in outs if not o.startswith("ERROR")] == []
    assert "traversal" in outs[0] and "traversal" in outs[2]
    assert ".git" in outs[5] and "refused" in outs[5]
    assert "sensitive" in outs[-1] or "secrets" in outs[-1]
    assert not (tmp_path / "outside.py").exists()
    assert res.changed_files == []
    pre = {k: v for k, v in before.items() if not k.startswith(".pedacito/")}
    post = {k: v for k, v in tree(root).items()
            if not k.startswith(".pedacito/")}
    assert post == pre


def test_secret_read_may_be_approved_by_the_permission_broker(tmp_path):
    client = ScriptedTools([[("read", {"file": ".env"})],
                            [("finish", {"summary": "s"})]])
    root, agent = make_agent(tmp_path, client)
    (root / ".env").write_text("TOKEN=real\n", encoding="utf-8")
    asked = []
    security.install_permission_hook(lambda info: (asked.append(info) or True,
                                                   ""))
    try:
        res = run_interleaved(agent, "x", apply=False)
    finally:
        security.install_permission_hook(None)
    assert "TOKEN=real" in tool_msgs(res, "read")[0]
    assert asked and asked[0]["category"] == security.CAT_SENSITIVE_FILE


# ================================================================== leases
def grep_forever(n_label="x"):
    counter = {"n": 0}

    def step(messages, client):
        counter["n"] += 1
        return [("grep", {"pattern": f"{n_label}{counter['n']}"})]

    return step


def test_max_total_model_calls_bounds_the_loop(tmp_path):
    client = ScriptedTools([grep_forever()], loop_last=True)
    root, agent = make_agent(tmp_path, client, max_total_model_calls=5,
                             max_no_progress_steps=0)
    res = run_interleaved(agent, "x", apply=False)
    assert len(client.requests) == 5 and res.turns == 5
    assert res.stop_reason == "max_total_model_calls" and not res.ok
    assert "max_total_model_calls" in res.error


def test_max_no_progress_steps_stops_a_spinning_model(tmp_path):
    client = ScriptedTools([[("read", {"file": "pricing.py"})]],
                           loop_last=True)
    root, agent = make_agent(tmp_path, client, max_no_progress_steps=3,
                             max_total_model_calls=50)
    res = run_interleaved(agent, "x", apply=False)
    # turn 1 is new; turns 2-4 repeat it verbatim and add nothing
    assert len(client.requests) == 4
    assert res.stop_reason == "max_no_progress_steps"
    assert "no progress" in res.error or "made no progress" in res.error


def test_failing_calls_count_as_no_progress_but_edits_reset_it(tmp_path):
    bad = [("read", {"file": "missing_a.py"})]
    good = [edit("calc.py", "return a + b", "return a + b + 1")]
    client = ScriptedTools([
        [("read", {"file": "nope1.py"})], [("read", {"file": "nope2.py"})],
        good,                                   # progress resets the counter
        [("read", {"file": "nope3.py"})], [("read", {"file": "nope4.py"})],
        [("finish", {"summary": "s"})]])
    root, agent = make_agent(tmp_path, client, max_no_progress_steps=3)
    res = run_interleaved(agent, "x", apply=False)
    assert res.stop_reason == "finish" and res.turns == 6
    assert bad and res.changed_files == ["calc.py"]


def test_cancel_event_before_start_sends_nothing(tmp_path):
    client = ScriptedTools([[("read", {"file": "pricing.py"})]])
    root, agent = make_agent(tmp_path, client)
    ev = threading.Event()
    ev.set()
    res = run_interleaved(agent, "x", apply=False, cancel_event=ev)
    assert client.requests == [] and res.turns == 0
    assert res.stop_reason == "cancelled" and not res.ok


def test_cancel_event_mid_run_skips_the_remaining_calls(tmp_path):
    ev = threading.Event()

    def cancel_now(messages, client):
        ev.set()
        return [("read", {"file": "pricing.py"}),
                edit("calc.py", "return a + b", "return 0")]

    client = ScriptedTools([[("grep", {"pattern": "bulk"})], cancel_now,
                            [("finish", {"summary": "never"})]])
    root, agent = make_agent(tmp_path, client)
    res = run_interleaved(agent, "x", apply=True, cancel_event=ev)
    assert len(client.requests) == 2
    assert res.stop_reason == "cancelled" and res.error == "cancelled"
    assert "cancelled" in tool_msgs(res)[-1]
    assert (root / "calc.py").read_text(encoding="utf-8") == CALC          # edit never ran
    ids = [tc["id"] for m in res.transcript if m["role"] == "assistant"
           for tc in m["tool_calls"]]
    assert len(ids) == len(tool_msgs(res))                  # history stays valid


def test_cancel_predicate_is_honoured(tmp_path):
    state = {"n": 0}

    def pred():
        state["n"] += 1
        return state["n"] > 2

    client = ScriptedTools([grep_forever()], loop_last=True)
    root, agent = make_agent(tmp_path, client, max_no_progress_steps=0)
    res = run_interleaved(agent, "x", apply=False, cancel_event=pred)
    assert res.stop_reason == "cancelled" and len(client.requests) <= 2


def test_max_task_seconds_lease(tmp_path, monkeypatch):
    class Clock:
        now = 1000.0

        def time(self):
            return self.now

    clock = Clock()
    monkeypatch.setattr(interleaved, "time", clock)

    def slow(messages, client):
        clock.now += 1000
        return [("grep", {"pattern": "slow"})]

    client = ScriptedTools([[("grep", {"pattern": "fast"})], slow,
                            [("finish", {"summary": "late"})]])
    root, agent = make_agent(tmp_path, client, max_task_seconds=900)
    res = run_interleaved(agent, "x", apply=False)
    assert res.stop_reason == "max_task_seconds" and not res.ok
    assert len(client.requests) == 2


def test_max_task_completion_tokens_lease(tmp_path):
    step = {"calls": [("grep", {"pattern": "a"})], "usage":
            {"completion_tokens": 600}}
    steps = [dict(step, calls=[("grep", {"pattern": f"p{i}"})])
             for i in range(6)]
    client = ScriptedTools(steps)
    root, agent = make_agent(tmp_path, client, max_task_completion_tokens=1000,
                             max_no_progress_steps=0)
    res = run_interleaved(agent, "x", apply=False)
    assert len(client.requests) == 2
    assert res.stop_reason == "max_task_completion_tokens"
    assert res.completion_tokens == 1200


def test_staged_work_survives_a_lease_stop(tmp_path):
    client = ScriptedTools([
        [edit("calc.py", "return a + b", "return a + b + 1")],
        grep_forever()], loop_last=True)
    root, agent = make_agent(tmp_path, client, max_total_model_calls=4,
                             max_no_progress_steps=0)
    res = run_interleaved(agent, "x", apply=False)
    assert not res.ok and res.stop_reason == "max_total_model_calls"
    assert res.changed_files == ["calc.py"] and res.file_diffs
    assert "Staged changes to 1 file" in res.summary


# ============================================================ KV stability
def test_system_prompt_tools_and_history_prefix_are_byte_identical(tmp_path):
    client = ScriptedTools(FIX_FLOW)
    root, agent = make_agent(tmp_path, client)
    res = run_interleaved(agent, "fix it", apply=False)
    assert res.ok
    reqs = client.requests
    assert len(reqs) == 7
    first_msgs, first_tools = reqs[0]
    sys_bytes = json.dumps(first_msgs[0], sort_keys=True)
    user_bytes = json.dumps(first_msgs[1], sort_keys=True)
    tool_bytes = json.dumps(first_tools, sort_keys=True)
    for msgs, tools in reqs:
        assert json.dumps(msgs[0], sort_keys=True) == sys_bytes
        assert json.dumps(msgs[1], sort_keys=True) == user_bytes
        assert json.dumps(tools, sort_keys=True) == tool_bytes
    # append-only: every request begins with the whole previous request
    for (prev, _), (cur, _) in zip(reqs, reqs[1:]):
        assert len(cur) > len(prev)
        assert [json.dumps(m, sort_keys=True) for m in cur[:len(prev)]] == \
            [json.dumps(m, sort_keys=True) for m in prev]


def test_old_tool_results_fold_once_the_window_is_nearly_full(tmp_path):
    big = "\n".join(f"line {i:03d} " + "x" * 40 for i in range(1, 400)) + "\n"
    reads = [[("read", {"file": "big.txt", "start_line": 1 + 30 * i,
                        "end_line": 30 + 30 * i})] for i in range(10)]
    client = ScriptedTools(reads + [[("finish", {"summary": "s"})]])
    files = dict(BASE_FILES, **{"big.txt": big})
    root, agent = make_agent(tmp_path, client, files=files,
                             context_tokens=2600, reserve_tokens=200,
                             max_no_progress_steps=0, max_total_model_calls=50)
    res = run_interleaved(agent, "x", apply=False)
    assert res.ok and res.folds >= 1
    reqs = client.requests
    sys0, user0 = reqs[0][0][0], reqs[0][0][1]
    last_msgs = reqs[-1][0]
    assert last_msgs[0] == sys0 and last_msgs[1] == user0   # head never moves
    tools_final = [m for m in last_msgs if m["role"] == "tool"]
    assert any(m["content"].startswith("[folded") for m in tools_final[:4])
    assert not tools_final[-1]["content"].startswith("[folded")
    assert not tools_final[-2]["content"].startswith("[folded")
    # call structure intact: every assistant call still has its result
    asst_ids = [tc["id"] for m in last_msgs if m["role"] == "assistant"
                for tc in m["tool_calls"]]
    assert asst_ids == [m["tool_call_id"] for m in tools_final]
    # the transcript keeps the FULL, unfolded results
    assert not any(m["content"].startswith("[folded")
                   for m in res.transcript if m["role"] == "tool")


# ============================================================ JSON protocol
def test_json_text_protocol_for_clients_without_native_tools(tmp_path):
    replies = [
        'I will look first.\n```json\n{"tool": "read", "arguments": '
        '{"file": "calc.py"}}\n```',
        '{"name": "edit_file", "arguments": {"file": "calc.py", "search": '
        '"return a + b", "replace": "return a * b"}}',
        '{"tool_calls": [{"tool": "diff", "arguments": {}}, {"tool": '
        '"finish", "arguments": {"summary": "multiplied"}}]}',
    ]
    client = JsonChat(replies)
    root, agent = make_agent(tmp_path, client)
    before = tree(root)
    res = run_interleaved(agent, "x", apply=False)
    assert res.ok and res.summary == "multiplied"
    assert res.changed_files == ["calc.py"] and tree(root) == before
    system = client.requests[0][0]["content"]
    assert "TOOL PROTOCOL" in system and "edit_file(file: string" in system
    assert "finish(summary: string" in system
    for msgs in client.requests:
        roles = [m["role"] for m in msgs]
        assert roles[0] == "system"
        assert all(a != b for a, b in zip(roles[1:], roles[2:]))   # alternate
    assert client.requests[1][-1]["content"].startswith("RESULT (read):")
    assert "return a * b" in res.workspace.read("calc.py")
    # both calls of the last reply ran, in order, answered in ONE user turn
    last = res.transcript[-1]
    assert last["role"] == "user"
    assert last["content"].index("RESULT (diff)") < last["content"].index(
        "RESULT (finish)")
    assert last["content"].endswith("Finished.")


def test_native_channel_accepts_json_text_replies_gracefully(tmp_path):
    client = ScriptedTools([
        '{"tool": "read", "arguments": {"file": "calc.py"}}',
        {"content": '```json\n[{"tool":"finish","arguments":{"summary":"ok"}}]'
                    '\n```', "calls": []},
    ])
    root, agent = make_agent(tmp_path, client)
    res = run_interleaved(agent, "x", apply=False)
    assert res.ok and res.summary == "ok" and res.turns == 2
    assert "def add" in tool_msgs(res, "read")[0]
    # history records the call natively even though it arrived as text
    assert res.transcript[2]["tool_calls"][0]["function"]["name"] == "read"


def test_parse_text_tool_calls_shapes():
    valid = frozenset({"read", "edit_file", "finish", "run_test"})
    one = parse_text_tool_calls('{"tool":"read","arguments":{"file":"a.py"}}',
                                valid)
    assert [(c.name, c.arguments) for c in one] == [("read", {"file": "a.py"})]
    flat = parse_text_tool_calls('{"action": "read", "file": "a.py", '
                                 '"reasoning": "look"}', valid)
    assert flat[0].arguments == {"file": "a.py"}
    fn = parse_text_tool_calls(
        '{"function": {"name": "finish", "arguments": "{\\"summary\\": \\"x\\"}"}}',
        valid)
    assert fn[0].name == "finish" and fn[0].arguments == {"summary": "x"}
    aliased = parse_text_tool_calls('{"tool": "edit", "arguments": {}}', valid)
    assert aliased[0].name == "edit_file"
    many = parse_text_tool_calls(
        'noise {"tool_calls": [{"name": "read", "arguments": {"file": "a"}}, '
        '{"name": "nope"}, {"name": "run_test"}]} trailing', valid)
    assert [c.name for c in many] == ["read", "run_test"]
    assert parse_text_tool_calls("just prose, no json", valid) == []
    assert parse_text_tool_calls('{"tool": "unknown_tool"}', valid) == []
    bad = parse_text_tool_calls('{"tool": "read", "arguments": "not json"}',
                                valid)
    assert bad[0].error


def test_prose_replies_get_one_nudge_then_end_the_run(tmp_path):
    client = ScriptedTools([
        {"content": "Let me think about this.", "calls": []},
        {"content": "The bug is the > in bulk_discount.", "calls": []}])
    root, agent = make_agent(tmp_path, client)
    res = run_interleaved(agent, "explain the bug", apply=False)
    assert res.stop_reason == "no_tool_call" and res.ok
    assert res.summary == "The bug is the > in bulk_discount."
    nudge = client.requests[1][0][-1]
    assert nudge["role"] == "user" and "call `finish`" in nudge["content"]


def test_silent_model_is_not_a_success(tmp_path):
    client = ScriptedTools([{"content": "", "calls": []}] * 3)
    root, agent = make_agent(tmp_path, client)
    res = run_interleaved(agent, "x", apply=False)
    assert not res.ok and "neither a tool call nor text" in res.error


def test_unknown_tools_and_bad_arguments_are_answered_not_fatal(tmp_path):
    client = ScriptedTools([
        [("teleport", {"to": "mars"}), ("read", {}), ("edit_file", {"file": "calc.py"}),
         ("edit_file", {"file": "calc.py", "replace": "x"}),
         ("run_test", {"target": "../../etc/passwd"})],
        [("finish", {"summary": "s"})]])
    root, agent = make_agent(tmp_path, client)
    res = run_interleaved(agent, "x", apply=False)
    outs = tool_msgs(res)
    assert "unknown tool `teleport`" in outs[0]
    assert "read needs a `file`" in outs[1]
    assert "needs `replace`" in outs[2]
    assert "needs `search`" in outs[3] and "write_file" in outs[3]
    assert outs[4].lstrip("[").count("ERROR") >= 1 or "refused" in outs[4]
    assert res.ok and res.changed_files == []


def test_a_crashing_tool_is_reported_not_fatal(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("index exploded")

    monkeypatch.setattr(interleaved.tools_mod, "tool_grep", boom)
    client = ScriptedTools([[("grep", {"pattern": "x"})],
                            [("finish", {"summary": "s"})]])
    root, agent = make_agent(tmp_path, client)
    res = run_interleaved(agent, "x", apply=False)
    out = tool_msgs(res, "grep")[0]
    assert out.startswith("ERROR") and "index exploded" in out
    assert res.ok and res.stop_reason == "finish"


def test_calls_per_turn_are_capped(tmp_path):
    many = [("grep", {"pattern": f"p{i}"}) for i in range(12)]
    client = ScriptedTools([many, [("finish", {"summary": "s"})]])
    root, agent = make_agent(tmp_path, client)
    res = run_interleaved(agent, "x", apply=False)
    outs = tool_msgs(res)
    skipped = [o for o in outs if o.startswith("[skipped")]
    assert len(skipped) == 12 - interleaved.MAX_CALLS_PER_TURN
    assert len(outs) == 12 + 1                      # every call id answered


# ================================================================= tools
def test_staged_files_are_visible_to_grep_outline_and_find(tmp_path):
    client = ScriptedTools([
        [("write_file", {"file": "fresh.py", "content":
                         "class Fresh:\n    def shiny(self):\n        return 7\n"})],
        [("grep", {"pattern": "shiny"}), ("outline", {"file": "fresh.py"}),
         ("find", {"pattern": "fresh"}), ("grep", {"pattern": "return 7",
                                                  "path": "fresh.py"})],
        [("finish", {"summary": "s"})]])
    root, agent = make_agent(tmp_path, client)
    res = run_interleaved(agent, "x", apply=False)
    grep, outline, find, scoped = tool_msgs(res)[1:5]
    assert "fresh.py:2" in grep and "def shiny" in grep
    assert "class Fresh" in outline and "def Fresh.shiny" in outline or \
        "shiny" in outline
    assert "fresh.py" in find
    assert "fresh.py:3" in scoped
    assert not (root / "fresh.py").exists()


def test_delete_write_and_diff_tools(tmp_path):
    client = ScriptedTools([
        [("delete_file", {"file": "calc.py"}),
         ("write_file", {"file": "docs/readme.md",
                         "content": "# hi\n\nreal text\n"}),
         ("diff", {}), ("diff", {"file": "calc.py"}),
         ("read", {"file": "calc.py"})],
        [("finish", {"summary": "s"})]])
    root, agent = make_agent(tmp_path, client)
    res = run_interleaved(agent, "x", apply=False)
    deleted, wrote, diff_all, diff_one, read = tool_msgs(res)[:5]
    assert "STAGED" in deleted and "calc.py" in deleted
    assert "docs/readme.md" in wrote
    assert "+# hi" in diff_all and "/dev/null" in diff_all
    assert "docs/readme.md" not in diff_one
    assert read.startswith("ERROR")                    # deleted: gone for read
    assert sorted(res.changed_files) == ["calc.py", "docs/readme.md"]
    assert (root / "calc.py").exists()
    fds = {fd.rel: fd for fd in res.file_diffs}
    assert fds["calc.py"].deleted and not fds["docs/readme.md"].deleted


def test_auto_allowlist_requires_reading_existing_files_first(tmp_path):
    client = ScriptedTools([
        [edit("pricing.py", "return 0.1", "return 0.2"),
         ("read", {"file": "pricing.py"}),
         edit("pricing.py", "return 0.1", "return 0.2"),
         ("write_file", {"file": "brand_new.py", "content": "N = 1\n"}),
         edit("calc.py", "return a + b", "return 0"),        # named in the task
         ("delete_file", {"file": "tests/test_pricing.py"})],
        [("finish", {"summary": "s"})]])
    root, agent = make_agent(tmp_path, client, auto_allowlist=True)
    res = run_interleaved(agent, "tweak calc.py please", apply=False)
    blocked, _read, ok1, new, named, deleted = tool_msgs(res)[:6]
    assert "write allowlist" in blocked and blocked.startswith("ERROR")
    assert ok1.startswith("STAGED") and new.startswith("STAGED")
    assert named.startswith("STAGED")
    assert "write allowlist" in deleted
    assert sorted(res.changed_files) == ["brand_new.py", "calc.py", "pricing.py"]


def test_duplicate_or_missing_call_ids_are_made_unique(tmp_path):
    class SameIds:
        def chat_tools(self, messages, tools, max_tokens, **kw):
            if len(messages) > 3:
                return {"tool_calls": [{"function": {
                    "name": "finish", "arguments": '{"summary": "s"}'}}]}
            return {"tool_calls": [
                {"id": "dup", "function": {"name": "grep",
                                           "arguments": '{"pattern": "a"}'}},
                {"id": "dup", "function": {"name": "grep",
                                           "arguments": '{"pattern": "b"}'}}]}

    root, agent = make_agent(tmp_path, SameIds())
    res = run_interleaved(agent, "x", apply=False)
    ids = [tc["id"] for m in res.transcript if m["role"] == "assistant"
           for tc in m["tool_calls"]]
    assert len(ids) == len(set(ids)) == 3
    assert ids == [m["tool_call_id"] for m in res.transcript
                   if m["role"] == "tool"]


def test_comment_only_stub_files_are_refused_with_a_clear_message(tmp_path):
    client = ScriptedTools([
        [("write_file", {"file": "docs/stub.md", "content": "# todo\n"})],
        [("finish", {"summary": "s"})]])
    root, agent = make_agent(tmp_path, client)
    res = run_interleaved(agent, "x", apply=False)
    out = tool_msgs(res, "write_file")[0]
    assert out.startswith("ERROR") and "stub" in out and "rename" not in out
    assert res.changed_files == []


def test_read_pages_long_files(tmp_path):
    big = "\n".join(f"row {i}" for i in range(1, 501)) + "\n"
    client = ScriptedTools([
        [("read", {"file": "big.txt"}),
         ("read", {"file": "big.txt", "start_line": 495}),
         ("read", {"file": "big.txt", "start_line": 900})],
        [("finish", {"summary": "s"})]])
    root, agent = make_agent(tmp_path, client, files={"big.txt": big},
                             max_read_lines=100)
    res = run_interleaved(agent, "x", apply=False)
    page1, tail, past = tool_msgs(res)[:3]
    assert page1.startswith("big.txt lines 1-100 of 500:")
    assert "start_line=101" in page1 and "row 100" in page1
    assert "row 101" not in page1.split("[showing")[0]
    assert tail.startswith("big.txt lines 495-500 of 500:")
    assert past.startswith("ERROR") and "500 line" in past


def test_run_tool_is_gated_by_allow_shell_deny_rules_and_the_broker(tmp_path):
    steps = [[("run", {"command": "echo hi"})],
             [("run", {"command": "python -c \"print(6 * 7)\""})],
             [("finish", {"summary": "s"})]]
    # allow_shell off + no broker: the tool is not even offered
    client = ScriptedTools(steps)
    root, agent = make_agent(tmp_path, client)
    res = run_interleaved(agent, "x", apply=False)
    assert "run" not in [t["function"]["name"] for t in client.requests[0][1]]
    assert "unknown tool `run`" in tool_msgs(res)[0]

    # allow_shell on, but shell_deny wins over it
    client = ScriptedTools(steps)
    root2, agent2 = make_agent(tmp_path / "b", client, allow_shell=True,
                               shell_deny=("echo *",))
    res = run_interleaved(agent2, "x", apply=False)
    assert "shell_deny" in tool_msgs(res)[0]
    assert "EXIT: 0" in tool_msgs(res)[1] and "42" in tool_msgs(res)[1]
    assert "BEGIN UNTRUSTED RUN OUTPUT" in tool_msgs(res)[1]

    # a live broker makes `run` available and its denial is the tool result
    asked = []
    security.install_permission_hook(
        lambda info: (asked.append(info["category"]) or False, "denied by user"))
    try:
        client = ScriptedTools(steps)
        root3, agent3 = make_agent(tmp_path / "c", client)
        res = run_interleaved(agent3, "x", apply=False)
    finally:
        security.install_permission_hook(None)
    assert "run" in [t["function"]["name"] for t in client.requests[0][1]]
    assert "denied by user" in tool_msgs(res)[0]
    assert asked and asked[0] == security.CAT_SHELL


def test_run_in_a_dry_run_sees_staged_edits_in_the_mirror(tmp_path):
    cmd = "python -c \"print(open('calc.py').read())\""
    client = ScriptedTools([
        [edit("calc.py", "return a + b", "return a * 41")],
        [("run", {"command": cmd})],
        [("finish", {"summary": "s"})]])
    root, agent = make_agent(tmp_path, client, allow_shell=True)
    before = tree(root)
    res = run_interleaved(agent, "x", apply=False)
    out = tool_msgs(res, "run")[0]
    assert "return a * 41" in out
    assert tree(root) == before


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_git_tool_is_read_only_and_flags_staged_edits(tmp_path):
    client = ScriptedTools([
        [("git", {"subcommand": "status"})],
        [edit("calc.py", "return a + b", "return 0")],
        [("git", {"subcommand": "status"}), ("git", {"subcommand": "push"}),
         ("git", {"subcommand": "log", "args": "-n 1"})],
        [("finish", {"summary": "s"})]])
    root, agent = make_agent(tmp_path, client)

    def sh(*a):
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *a],
                       cwd=root, check=True, capture_output=True)

    sh("init", "-q")
    sh("add", "-A")
    sh("commit", "-q", "-m", "seed")
    res = run_interleaved(agent, "x", apply=False)
    status0 = tool_msgs(res, "git")[0]
    status1, push, log = tool_msgs(res, "git")[1:4]
    assert "BEGIN UNTRUSTED GIT OUTPUT" in status0
    assert "staged in memory" in status1 and "use the `diff` tool" in status1
    assert push.startswith("ERROR") and "status, diff, log or blame" in push
    assert "seed" in log
    assert (root / "calc.py").read_text(encoding="utf-8") == CALC


# ============================================================ tool schemas
def names(schemas):
    return [s["function"]["name"] for s in schemas]


def test_default_schema_lists_the_core_tools_in_openai_format():
    cfg = Config()
    schemas = build_tool_schemas(cfg)
    assert names(schemas) == ["read", "grep", "find", "outline", "edit_file",
                              "write_file", "delete_file", "run_test", "git",
                              "diff", "finish"]
    for s in schemas:
        assert s["type"] == "function"
        fn = s["function"]
        assert fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert set(params["required"]) <= set(params["properties"])
    by = {s["function"]["name"]: s["function"]["parameters"] for s in schemas}
    assert set(by["read"]["properties"]) == {"file", "start_line", "end_line"}
    assert by["read"]["required"] == ["file"]
    assert {"file", "search", "replace", "symbol"} <= set(
        by["edit_file"]["properties"])
    assert by["write_file"]["required"] == ["file", "content"]
    assert by["finish"]["required"] == ["summary"]
    assert by["git"]["properties"]["subcommand"]["enum"] == [
        "status", "diff", "log", "blame"]
    assert json.dumps(schemas) == json.dumps(build_tool_schemas(Config()))


def test_optional_tools_follow_their_settings(monkeypatch):
    assert "run" in names(build_tool_schemas(Config(allow_shell=True)))
    assert "run" not in names(build_tool_schemas(Config()))
    assert "git" not in names(build_tool_schemas(Config(allow_git=False)))
    assert "lsp" in names(build_tool_schemas(Config(allow_lsp=True)))
    cfg = Config()
    cfg.enable_lsp = True                       # the roadmap's spelling
    assert "lsp" in names(build_tool_schemas(cfg))
    assert "lsp" not in names(build_tool_schemas(Config()))
    assert "mcp_call" in names(build_tool_schemas(Config(allow_mcp=True)))
    assert "mcp_call" not in names(build_tool_schemas(Config()))
    monkeypatch.setattr(interleaved.tools_mod, "_image_gen_offered",
                        lambda cfg: True)
    assert "generate_image" in names(build_tool_schemas(Config()))
    # a dry run never writes binary files
    assert "generate_image" not in names(build_tool_schemas(Config(),
                                                            apply=False))
    # finish is always last, so the prefix is stable as tools are added
    assert names(build_tool_schemas(Config(allow_shell=True,
                                           allow_mcp=True)))[-1] == "finish"


# ====================================================== checkpoint handling
def test_checkpoint_argument_forms(tmp_path):
    from pedacito import checkpoints

    seen = []
    client = ScriptedTools([[("finish", {"summary": "s"})]])
    root, agent = make_agent(tmp_path, client)
    res = run_interleaved(agent, "x", apply=True, checkpoint=True)
    assert res.checkpoint_id.startswith("cp_")
    assert len(checkpoints.list_checkpoints(root)) == 1
    # a dry run has nothing to protect: no snapshot is taken
    root2, agent2 = make_agent(tmp_path / "d", ScriptedTools([]))
    res2 = run_interleaved(agent2, "x", apply=False, checkpoint=True)
    assert res2.checkpoint_id == ""
    assert not (root2 / ".pedacito" / "checkpoints").exists()
    res3 = run_interleaved(agent, "x", apply=False, checkpoint="cp_0042_abc")
    assert res3.checkpoint_id == "cp_0042_abc"
    res4 = run_interleaved(
        agent, "x", apply=False,
        checkpoint=lambda ws: seen.append(ws.apply) or {"id": "cp_cb"})
    assert res4.checkpoint_id == "cp_cb" and seen == [False]
    res5 = run_interleaved(agent, "x", apply=False,
                           checkpoint=lambda ws: 1 / 0)    # never fatal
    assert res5.checkpoint_id == ""


# ================================================= real LLMClient transports
def json_response(obj) -> requests.Response:
    r = requests.Response()
    r.status_code = 200
    r._content = json.dumps(obj).encode("utf-8")
    r._content_consumed = True
    r.headers["Content-Type"] = "application/json"
    r.encoding = "utf-8"
    return r


def openai_tool_reply(*calls, usage=None):
    return {"choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_{i}", "type": "function",
             "function": {"name": n, "arguments": json.dumps(a)}}
            for i, (n, a) in enumerate(calls)]}}],
        "usage": usage or {"prompt_tokens": 300, "completion_tokens": 40}}


def test_real_llmclient_native_tool_calls_over_the_openai_channel(tmp_path):
    cfg = Config()
    cfg.base_url = "https://api.example.com/v1"
    cfg.model = "frontier-1"
    cfg.tool_mode = "native"
    client = LLMClient(cfg)
    payloads = []
    replies = [
        openai_tool_reply(("read", {"file": "calc.py"})),
        openai_tool_reply(("finish", {"summary": "looked"}),
                          usage={"prompt_tokens": 500, "completion_tokens": 9}),
    ]

    def fake_post(payload, stream=False):
        payloads.append(copy.deepcopy(payload))
        return json_response(replies[len(payloads) - 1])

    client._post = fake_post
    root, agent = make_agent(tmp_path, client, cfg=cfg)
    res = run_interleaved(agent, "look at calc", apply=False)
    assert res.ok and res.summary == "looked" and res.turns == 2
    p0, p1 = payloads
    assert p0["tool_choice"] == "auto" and p0["stream"] is False
    assert [t["function"]["name"] for t in p0["tools"]][:2] == ["read", "grep"]
    assert p0["tools"] == p1["tools"]
    assert p0["messages"][0]["role"] == "system"
    # the second request replays the call and its result in native shape
    assert p1["messages"][2]["tool_calls"][0]["function"]["name"] == "read"
    assert p1["messages"][3]["role"] == "tool"
    assert p1["messages"][3]["tool_call_id"] == "call_0"
    assert "def add" in p1["messages"][3]["content"]
    # usage flowed into the client's accounting
    assert client.stats["calls"] == 2
    assert client.stats["completion_tokens"] == 49
    assert res.completion_tokens == 49


def test_real_llmclient_falls_back_to_json_when_tools_are_rejected(tmp_path):
    cfg = Config()
    cfg.base_url = "https://api.example.com/v1"
    cfg.model = "no-tools-model"
    cfg.tool_mode = "native"
    client = LLMClient(cfg)
    payloads = []

    def fake_post(payload, stream=False):
        payloads.append(copy.deepcopy(payload))
        if "tools" in payload:
            raise LLMError("Server returned 400: this model does not support "
                           "tools")
        text = ('{"tool": "finish", "arguments": {"summary": "json mode"}}')
        return json_response({"choices": [{"finish_reason": "stop", "message":
                                           {"content": text}}], "usage": {}})

    client._post = fake_post
    root, agent = make_agent(tmp_path, client, cfg=cfg)
    res = run_interleaved(agent, "x", apply=False)
    assert res.ok and res.summary == "json mode"
    assert client._native_ok is False
    assert "tools" in payloads[0] and "tools" not in payloads[1]
    assert "TOOL PROTOCOL" in payloads[1]["messages"][0]["content"]


def test_real_llmclient_local_grammar_mode_uses_the_json_protocol(tmp_path):
    cfg = Config()
    cfg.base_url = "http://127.0.0.1:1234/v1"
    cfg.model = "local-model"
    client = LLMClient(cfg)
    payloads = []

    def fake_post(payload, stream=False):
        payloads.append(copy.deepcopy(payload))
        text = '{"tool": "finish", "arguments": {"summary": "local"}}'
        return json_response({"choices": [{"finish_reason": "stop", "message":
                                           {"content": text}}], "usage": {}})

    client._post = fake_post
    root, agent = make_agent(tmp_path, client, cfg=cfg)
    res = run_interleaved(agent, "x", apply=False)
    assert res.ok and res.summary == "local"
    assert "tools" not in payloads[0]


def test_real_llmclient_anthropic_native_tool_use(tmp_path):
    cfg = Config()
    cfg.base_url = "https://api.anthropic.com/v1"
    cfg.model = "claude-test"
    client = LLMClient(cfg)
    payloads = []
    replies = [
        {"content": [{"type": "text", "text": "reading"},
                     {"type": "tool_use", "id": "tu_1", "name": "read",
                      "input": {"file": "calc.py"}}],
         "stop_reason": "tool_use",
         "usage": {"input_tokens": 200, "output_tokens": 30}},
        {"content": [{"type": "tool_use", "id": "tu_2", "name": "finish",
                      "input": {"summary": "claude says done"}}],
         "stop_reason": "tool_use",
         "usage": {"input_tokens": 260, "output_tokens": 12}},
    ]

    def fake_anthropic_post(payload, headers, stream=False):
        payloads.append(copy.deepcopy(payload))
        return json_response(replies[len(payloads) - 1])

    client._anthropic_post = fake_anthropic_post
    root, agent = make_agent(tmp_path, client, cfg=cfg)
    res = run_interleaved(agent, "x", apply=False)
    assert res.ok and res.summary == "claude says done"
    p0, p1 = payloads
    assert [t["name"] for t in p0["tools"]][:2] == ["read", "grep"]
    assert "input_schema" in p0["tools"][0]
    assert "thinking" not in p0                 # tool turns run thinking-off
    assert p0["system"][0]["text"].startswith("You are Pedacito's coding agent")
    last = p1["messages"][-1]
    assert last["role"] == "user" and last["content"][0]["type"] == "tool_result"
    assert last["content"][0]["tool_use_id"] == "tu_1"
    assert any(b.get("type") == "tool_use" for b in p1["messages"][-2]["content"])
    assert client.stats["completion_tokens"] == 42


# ========================================================== full-tree proof
def test_dry_run_leaves_the_whole_tree_byte_identical(tmp_path):
    steps = FIX_FLOW[:6] + [
        [("write_file", {"file": "docs/new.md", "content": "# new\n\nbody\n"}),
         ("delete_file", {"file": "calc.py"})],
        [("finish", {"summary": "s"})]]
    client = ScriptedTools(steps)
    root, agent = make_agent(tmp_path, client, allow_shell=True)
    before = tree(root)
    mtimes = {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
    res = run_interleaved(agent, "x", apply=False)
    assert res.ok, res.error
    assert sorted(res.changed_files) == ["calc.py", "docs/new.md", "pricing.py"]
    assert tree(root) == before
    assert {p: p.stat().st_mtime_ns for p in root.rglob("*")
            if p.is_file()} == mtimes
    assert not (root / "docs").exists()


def test_interleaved_result_defaults():
    r = InterleavedResult()
    assert not r.ok and r.changed_files == [] and r.turns == 0
    assert r.workspace is None and r.error == ""
