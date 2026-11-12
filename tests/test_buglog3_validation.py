"""Regression tests for the third bug-hunting report
(`PEDACITO_BUG_LOG_3.md`, PulseForge-16 run, Nov 2026).

Like `test_buglog_validation.py` and `test_buglog2_validation.py`, each
section pins the deterministic half of one reported finding: the gate that
did not exist, the verdict that lied, the ledger that over-credited.
Findings whose cause is purely model behaviour are exercised through the
helper the fix added, not by re-running a live model. BUG-327 is pinned as
verified-not-a-bug (the ledger behaviour the report suspected is correct).
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import (config as config_mod, domcheck, edits, events,  # noqa: E402
                      llm, loop, memory, plan, refactor, scaffold, testsel,
                      verify, webcheck)
from pedacito import index as index_mod  # noqa: E402
from pedacito.agent import Agent, _delivered_end  # noqa: E402
from pedacito.config import Config  # noqa: E402
from pedacito.edits import Result, apply_edits, parse_edits  # noqa: E402
from pedacito.llm import LLMClient, LLMError  # noqa: E402


def _cfg(**kw) -> Config:
    c = Config()
    c.log_sessions = False
    c.log_events = False
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _w(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _files(tmp_path, tree: dict[str, str], **cfg_kw):
    for rel, text in tree.items():
        _w(tmp_path, rel, text)
    cfg = _cfg(**cfg_kw)
    idx = index_mod.build([str(tmp_path)], cfg, root=tmp_path,
                          summarise=False, verbose=False)
    agent = Agent(idx, None, cfg)
    return agent, cfg, idx


def _edit(file, search, replace):
    return (f"--- FILE: {file}\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n"
            ">>>>>>> REPLACE\n")


class Scripted:
    """chat_json pops scripted gather steps; chat pops scripted replies."""
    step_truncated = False
    stream_truncated = False

    def __init__(self, steps=(), replies=("ok",)):
        self.steps = list(steps)
        self.replies = list(replies)
        self.json_calls: list[list[dict]] = []
        self.chat_calls: list[list[dict]] = []
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def chat_json(self, messages, schema, max_tokens=0, **kw):
        self.json_calls.append([dict(m) for m in messages])
        step = self.steps.pop(0) if self.steps else {"action": "done"}
        return {"reasoning": "r", **step}

    def chat(self, messages, max_tokens=0, stream=False, **kw):
        self.chat_calls.append([dict(m) for m in messages])
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def _gather_agent(tmp_path, client, tree, **cfg_kw):
    for rel, text in tree.items():
        _w(tmp_path, rel, text)
    cfg = _cfg(backup=False, stuck_recovery=False, **cfg_kw)
    idx = index_mod.build([str(tmp_path)], cfg, root=tmp_path,
                          summarise=False, verbose=False)
    return Agent(idx, client, cfg), cfg, idx


# ------------------------------------- BUG-301: new never ran the suite --
# `new --apply` wrote a test suite and exited 0 without running it; the red
# verdict only surfaced when the user ran npm test themselves.

class _PyBuild:
    """Plans and builds one Python file; nothing else to say."""
    stats = {"calls": 0, "completion_tokens": 0, "seconds": 0}

    def __init__(self, files=None):
        self.files = files or [{"path": "a.py", "purpose": "core"}]
        self.built: list[str] = []

    def chat_json(self, messages, schema, max_tokens=0, **kw):
        return {"project_name": "demo", "summary": "a demo",
                "files": self.files}

    def chat(self, messages, max_tokens=0, **kw):
        user = messages[-1]["content"]
        for f in self.files:
            rel = f["path"]
            if rel in user and rel not in self.built:
                self.built.append(rel)
                if rel.endswith(".md"):
                    return _edit(rel, "", "# Demo\n\nA demo project.\n")
                return _edit(rel, "", "def a():\n    return 1\n")
        return ""


def _run_new(tmp_path, client, monkeypatch, *, verify_cmd="", detect="npm test",
             verdict=(True, ""), **cfg_kw):
    seen = {}

    def fake_detect(root, cfg=None):
        seen["detect"] = True
        return detect

    def fake_verify(client_, cfg_, idx_, rels, name, backup_dir, **kw):
        seen["verify"] = kw
        return verdict

    monkeypatch.setattr(testsel, "detect_test_command", fake_detect)
    monkeypatch.setattr(scaffold, "verify_and_fix", fake_verify)
    code = scaffold.run_new(tmp_path, _cfg(**cfg_kw), client,
                            scaffold.Spec("a demo project"),
                            apply=True, verbose=False, verify=verify_cmd)
    return code, seen


def test_new_runs_the_detected_suite_and_a_red_verdict_reaches_exit(
        tmp_path, monkeypatch, capsys):
    code, seen = _run_new(tmp_path, _PyBuild(), monkeypatch,
                          verdict=(False, "3 failing tests / 6 assertions"))
    err = capsys.readouterr().err
    assert seen["verify"]["explicit_cmd"] == "npm test"
    assert code == 1, (code, err)
    assert "BUILD INCOMPLETE" in err and "npm test" in err


def test_new_green_suite_exits_zero(tmp_path, monkeypatch, capsys):
    code, seen = _run_new(tmp_path, _PyBuild(), monkeypatch,
                          verdict=(True, ""))
    err = capsys.readouterr().err
    assert code == 0, (code, err)
    assert "tests: passed (npm test)" in err
    assert "BUILD INCOMPLETE" not in err


def test_new_without_a_test_command_skips_verification(tmp_path, monkeypatch):
    code, seen = _run_new(tmp_path, _PyBuild(), monkeypatch, detect="")
    assert code == 0
    assert "verify" not in seen                 # nothing to run, nothing ran


def test_new_explicit_verify_command_wins_over_detection(tmp_path,
                                                         monkeypatch):
    code, seen = _run_new(tmp_path, _PyBuild(), monkeypatch,
                          verify_cmd="make check")
    assert code == 0
    assert seen["verify"]["explicit_cmd"] == "make check"
    assert "detect" not in seen


# ------------------------------------- BUG-302: planned docs stay stubs --
# Plan.sources excludes .md, so a planned README was skeletonised and never
# built; the summary could not even report the stub.

def _docs_plan():
    return scaffold.Plan("demo", "a demo", files=[
        scaffold.PlannedFile("a.py", "core"),
        scaffold.PlannedFile("README.md", "project documentation"),
    ], spec=scaffold.Spec("a demo project"))


def test_build_docs_writes_the_planned_readme(tmp_path):
    _w(tmp_path, "a.py", "def a():\n    return 1\n")
    p = _docs_plan()
    scaffold.apply_skeleton(tmp_path, p, verbose=False)
    idx = scaffold.skeleton_index(tmp_path, p, Config())
    stub = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "project documentation" in stub      # the three-line header
    client = Scripted(replies=[_edit("README.md", "",
                                     "# Demo\n\nRun `python a.py`.\n")])
    summary = scaffold.build_docs(client, Config(), idx, p, verbose=False)
    assert summary["done"] == ["README.md"] and not summary["failed"]
    text = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert text.startswith("# Demo") and "Run `python a.py`." in text


def test_build_docs_no_block_is_a_failure_not_a_noop(tmp_path):
    _w(tmp_path, "a.py", "def a():\n    return 1\n")
    p = _docs_plan()
    scaffold.apply_skeleton(tmp_path, p, verbose=False)
    idx = scaffold.skeleton_index(tmp_path, p, Config())
    client = Scripted(replies=["I cannot write that."])
    summary = scaffold.build_docs(client, Config(), idx, p, verbose=False)
    assert summary["done"] == [] and len(summary["failed"]) == 1
    assert summary["failed"][0][0] == "README.md"
    assert len(client.chat_calls) == 2          # one nudge re-ask, then give up


def test_build_docs_skips_a_missing_file(tmp_path):
    p = _docs_plan()
    scaffold.apply_skeleton(tmp_path, p, verbose=False)
    idx = scaffold.skeleton_index(tmp_path, p, Config())
    (tmp_path / "README.md").unlink()
    summary = scaffold.build_docs(Scripted(), Config(), idx, p, verbose=False)
    assert summary["skipped"] == ["README.md"] and not summary["done"]


def test_new_merges_docs_into_the_summary_and_exit(tmp_path, monkeypatch,
                                                   capsys):
    client = _PyBuild(files=[{"path": "a.py", "purpose": "core"},
                             {"path": "README.md", "purpose": "docs"}])
    code, _seen = _run_new(tmp_path, client, monkeypatch, detect="")
    captured = capsys.readouterr()
    assert code == 0, (code, captured.err)
    # summarise_result goes to stdout; the README must be counted, not hidden
    assert "2 file(s) written" in captured.out
    assert (tmp_path / "README.md").read_text(
        encoding="utf-8").startswith("# Demo")


# ------------------------------------- BUG-303: new emits no event stream --

def test_new_write_path_emits_session_start_and_call_events(tmp_path):
    client = _PyBuild()
    client.on_call = None                       # a real client's usage hook
    cfg = _cfg(log_events=True)
    code = scaffold.run_new(tmp_path, cfg, client, scaffold.Spec("a demo"),
                            plan_only=True, verbose=False)
    assert code == 0
    evs = events.load(tmp_path)
    starts = [e for e in evs if e.get("kind") == "session_start"]
    assert starts and starts[0].get("cmd") == "new"
    assert callable(client.on_call)             # attached before the first call
    client.on_call({"site": "plan", "prompt_tokens": 10, "completion_tokens": 2})
    assert any(e.get("kind") == "call" and e.get("site") == "plan"
               for e in events.load(tmp_path))


def test_new_dry_run_writes_no_events(tmp_path):
    code = scaffold.run_new(tmp_path, _cfg(log_events=True), _PyBuild(),
                            scaffold.Spec("a demo"), verbose=False)
    assert code == 0
    assert not (tmp_path / ".pedacito" / "events").exists()


# ------------------------------------- BUG-304: duplicate whole-file targets --
# Three describe() titles that are not definitions each fell back to the same
# whole-file target; the satisfied target was re-asked to exhaustion and the
# batch ended "3 of 4" although every intended test was green.

class _OneHunk:
    stream_truncated = False

    def __init__(self, reply):
        self.reply = reply
        self.calls: list[list[dict]] = []

    def chat(self, messages, max_tokens=0, **kw):
        self.calls.append(messages)
        return self.reply


def test_duplicate_resolved_targets_are_covered_not_reasked(tmp_path):
    tree = {"suite.py": "def keep():\n    return 1\n"}
    agent, cfg, idx = _files(tmp_path, tree)
    reply = _edit("suite.py", "def keep():\n    return 1\n",
                  "def keep():\n    return 2\n")
    client = _OneHunk(reply)
    agent.client = client
    intents = [{"file": "suite.py", "symbol": s,
                "intended_change": "fix the block"}
               for s in ("describe: one", "describe: two", "describe: three")]
    results, receipts, findings = agent.apply_edit_intents(
        "Fix the three describe blocks in suite.py", "", intents,
        verbose=False)
    assert len(client.calls) == 1               # targets 2+3 never re-asked
    assert all(r.ok for r in results)
    assert not [f for f in findings if f.severity == "error"]
    assert receipts.count("[EDITED suite.py]") == 3
    assert (tmp_path / "suite.py").read_text().count("return 2") == 1


# ------------------------------------- BUG-305/330: all-to-all provenance --
# Every file was tagged with every clause in the run because generic terms
# and read-span overlap counted as evidence.

PROV_TASK = (
    "1. In src/presets.js add the PRESETS bank and applyPreset.\n"
    "2. In src/ui.js populate the preset select options.\n"
    "3. In src/main.js wire the onPresetChange handler.")


def test_clause_provenance_is_per_file_not_all_to_all():
    tracker = verify.RequirementTracker(PROV_TASK)
    assert [r.req_id for r in tracker.items] == ["R1", "R2", "R3"]
    tracker.observe_edit([
        Result(True, "src/presets.js", "applied",
               after="export const PRESETS = { warm: [] };\n"
                     "export function applyPreset(pattern, bank) {\n"
                     "  pattern.tracks = bank.rows;\n}\n"),
        Result(True, "src/ui.js", "applied",
               after="const sel = document.getElementById('preset-bank');\n"
                     "for (const name of Object.keys(bank)) {\n"
                     "  sel.options.add(new Option(name, name));\n}\n"),
        Result(True, "src/main.js", "applied",
               after="buildTransport(transport, "
                     "{ onPresetChange: loadPreset });\n"),
    ])
    assert tracker.clause_map() == {
        "src/presets.js": ["R1"],
        "src/ui.js": ["R2"],
        "src/main.js": ["R3"],
    }


def test_generic_terms_alone_do_not_badge_a_file():
    task = ("1. Make npm test pass for the public tests. "
            "2. Replace the stub README.md with real documentation.")
    tracker = verify.RequirementTracker(task)
    tracker.observe_edit([
        Result(True, "README.md", "applied",
               after="# PulseForge\n\nReal documentation.\n"),
        Result(True, "tests/theory.test.js", "applied",
               after="test('offset', () => { strictEqual(offset, 62.5); });\n"),
    ])
    # R1's vocabulary is all generic (npm/tests/public/pass): it cannot badge
    # a file it does not name; the README serves only the README clause.
    assert tracker.clause_map() == {"README.md": ["R2"]}


def test_five_clause_three_file_run_is_not_all_to_all():
    task = ("1. In src/presets.js define PRESETS as four rows of sixteen.\n"
            "2. In src/presets.js implement applyPreset over tracks.\n"
            "3. In src/ui.js fill the preset dropdown.\n"
            "4. In src/ui.js return the pattern from the loader.\n"
            "5. In src/main.js pass onPresetChange to buildTransport.")
    tracker = verify.RequirementTracker(task)
    tracker.observe_edit([
        Result(True, "src/presets.js", "applied",
               after="export const PRESETS = [];\n"
                     "export function applyPreset(p, b) { p.tracks = b; }\n"),
        Result(True, "src/ui.js", "applied",
               after="fillDropdown();\nreturn pattern;\n"),
        Result(True, "src/main.js", "applied",
               after="buildTransport(t, { onPresetChange: load });\n"),
    ])
    cmap = tracker.clause_map()
    assert cmap.get("src/presets.js") == ["R1", "R2"]
    assert cmap.get("src/ui.js") == ["R3", "R4"]
    assert cmap.get("src/main.js") == ["R5"]


# ------------------------------------- BUG-306: constraints as requirements --
# "Change nothing else" became an unaddressed edit requirement and could seed
# a pointless R4 follow-up.

def test_negative_constraint_is_not_a_coverage_requirement():
    task = "Fix the offset in theory.js. Change nothing else."
    reqs = verify._requirements_from_task(task)
    kinds = [r.kind for r in reqs]
    assert "constraint" in kinds
    constraint = next(r for r in reqs if r.kind == "constraint")
    assert constraint.final                     # never chased by gather
    ledger = verify.requirement_ledger(task)
    assert len(ledger) == 1 and "Change nothing else" not in ledger[0][0]
    edits = [Result(True, "theory.js", "applied",
                    after="const offset = 62.5;\n")]
    assert verify.check_coverage(task, "done", edits) == []
    assert verify.coverage_seeds(task, "done", edits) == []


def test_preservation_clause_is_a_constraint():
    task = ("Add the delay node to synth.js. Keep every public signature "
            "and do not weaken unrelated tests.")
    reqs = verify._requirements_from_task(task)
    assert [r.kind for r in reqs] == ["edit", "constraint"]
    assert verify.requirement_ledger(task)[0][0].startswith("Add the delay")
    edits = [Result(True, "synth.js", "applied",
                    after="export const delay = new DelayNode(ctx);\n")]
    assert verify.check_coverage(task, "done", edits) == []


# ------------------------------------- BUG-307: ungrounded README accepted --

def _grounding_agent(tmp_path):
    _w(tmp_path, "package.json", json.dumps({
        "name": "pulseforge-16",
        "scripts": {"test": "node --test tests/*.test.js"}}))
    return types.SimpleNamespace(index=types.SimpleNamespace(root=tmp_path))


def test_readme_contradicting_the_manifest_is_flagged(tmp_path):
    agent = _grounding_agent(tmp_path)
    r = Result(True, "README.md", "applied",
               after="# Pedacito\n\nTests are written in Jest.\n")
    found = verify.check_doc_grounding(agent, [r])
    assert len(found) == 1 and found[0].severity == "warn"
    assert "Pedacito" in found[0].message and "jest" in found[0].message.lower()


def test_grounded_readme_passes(tmp_path):
    agent = _grounding_agent(tmp_path)
    r = Result(True, "README.md", "applied",
               after="# PulseForge-16\n\nRun the tests with node --test.\n")
    assert verify.check_doc_grounding(agent, [r]) == []


# ------------------------------------- BUG-308: failed edits as evidence --
# A rejected hunk's attempted replace text satisfied the clause term match,
# so R4 never fired in the exact case it exists for.

def test_rejected_hunk_does_not_satisfy_its_clause():
    # Coverage reasons over multi-clause tasks; the rejected hunk names every
    # symbol of its clause, and that text must not count as evidence.
    task = ("1. Add generateEuclidean to src/pattern.js.\n"
            "2. Add a delay node to src/synth.js.")
    attempted = "export function generateEuclidean(steps, pulses) {\n" \
                "  return [];\n}\n"
    failed = [Result(False, "src/pattern.js",
                     "edit produced a syntax error near line 149; not written",
                     after=attempted),
              Result(True, "src/synth.js", "applied",
                     after="export const delay = new DelayNode(ctx);\n")]
    found = verify.check_coverage(task, "I tried.", failed)
    assert any("generateEuclidean" in f.message for f in found)
    seeds = verify.coverage_seeds(task, "I tried.", failed)
    assert len(seeds) == 1                      # R4 has its trigger back
    assert "generateEuclidean" in seeds[0].text
    applied = [Result(True, "src/pattern.js", "applied", after=attempted),
               failed[1]]
    assert verify.check_coverage(task, "done", applied) == []
    assert verify.coverage_seeds(task, "done", applied) == []


# ------------------------------------- BUG-309: the over-credited ledger --
# A range recorded for lines a budget cut removed turned every later read of
# them into a false "already read" note.

def test_delivered_end_counts_only_shown_lines():
    obs = ("f.py lines 1-60:\n 1 | alpha\n 2 | beta\n"
           "[truncated: context budget reached]")
    # The cut can land mid-line: the last numbered line is only a prefix, so
    # neither the ledger nor the evidence store may credit it.
    assert _delivered_end(obs, 1) == 1
    assert _delivered_end("f.py lines 5-60:\n 5 | x\n", 5) == 5
    assert _delivered_end("ERROR: no such file", 1) == 0
    assert _delivered_end("f.py lines 1-60:\n[truncated]", 1) == 0


def test_char_budget_cut_credits_only_delivered_lines(tmp_path):
    tree = {"f.py": "".join(f"line_{i:03d} = {i}\n" for i in range(1, 61))}
    client = Scripted([{"action": "read", "file": "f.py",
                        "start": 1, "end": 60}])
    agent, cfg, idx = _gather_agent(tmp_path, client, tree,
                                    max_gathered_chars=400)
    agent.gather("read f.py", verbose=False)
    spans = agent._read_spans.get("f.py") or []
    assert spans, "the delivered prefix must still be credited"
    start, end = spans[0]
    assert start == 1 and end < 60              # NOT the requested range
    shown = agent.evidence.spans[-1]
    assert shown.end == end                     # evidence agrees with ledger


def test_subagent_absorption_credits_only_rendered_lines(tmp_path):
    tree = {"big.py": "".join(f"v{i} = {i}\n" for i in range(1, 31)),
            "other.py": "".join(f"w{i} = {i}\n" for i in range(1, 31))}
    agent, cfg, idx = _gather_agent(tmp_path, Scripted(), tree,
                                    max_read_tokens=10)   # ~35 chars
    child = Agent(idx, None, cfg)
    child.evidence.record("big.py", 1, 30, tree["big.py"], 1)
    child.evidence.record("other.py", 1, 30, tree["other.py"], 2)
    selected = agent.select_subagent_evidence(child, limit=2)
    rendered = agent._render_subagent_evidence(selected)
    copied = agent.absorb_subagent_evidence(child, limit=2)
    assert copied, "the first span fits partially"
    first = copied[0]
    assert first.end < 30                       # clipped, not the full span
    # the ledger credits exactly the lines the render delivers
    assert f"{first.file} lines 1-{first.end}:" in rendered
    ledger = agent._read_spans.get(first.file)
    assert ledger and ledger[0] == (1, first.end)
    # a span the render dropped entirely is not credited anywhere
    dropped = "big.py" if first.file == "other.py" else "other.py"
    assert len(copied) == 1 and dropped not in agent._read_spans
    assert dropped not in rendered


# ------------------------------------- BUG-310: Python-only static gate --
# A JS hunk that deleted the node:test / node:assert imports while still
# calling describe / strictEqual passed "applied and verified".

JS_BEFORE = ("// tests/theory.test.js -- Node test suite\n"
             "import { describe, test } from 'node:test';\n"
             "import { strictEqual, ok } from 'node:assert/strict';\n"
             "import { midiToHz } from '../src/theory.js';\n"
             "describe('theory', () => {\n"
             "  test('a4', () => { strictEqual(midiToHz(69), 440); });\n"
             "});\n")
JS_AFTER = ("import { midiToHz } from '../src/theory.js';\n"
            "describe('theory', () => {\n"
            "  test('a4', () => { strictEqual(midiToHz(69), 440); });\n"
            "});\n")


def test_js_lost_import_gate_flags_deleted_imports():
    r = Result(True, "tests/theory.test.js", "applied",
               before=JS_BEFORE, after=JS_AFTER)
    found = verify.check_static_js([r])
    assert len(found) == 1 and found[0].severity == "error"
    assert "describe" in found[0].message and "strictEqual" in found[0].message


def test_js_lost_import_gate_quiet_when_redeclared_or_unused():
    rede = JS_AFTER + "function describe(name, fn) { fn(); }\n" \
                      "const test = (name, fn) => fn();\n" \
                      "const strictEqual = (a, b) => a === b;\n"
    assert verify.check_static_js([Result(True, "t.js", "ok",
                                          before=JS_BEFORE, after=rede)]) == []
    unused = "import { midiToHz } from '../src/theory.js';\nmidiToHz(69);\n"
    assert verify.check_static_js([Result(True, "t.js", "ok",
                                          before=JS_BEFORE,
                                          after=unused)]) == []
    assert verify.check_static_js([Result(False, "t.js", "rejected",
                                          before=JS_BEFORE,
                                          after=JS_AFTER)]) == []


def test_verify_check_runs_the_js_gate(tmp_path):
    agent, cfg, idx = _files(tmp_path, {"t.js": JS_BEFORE})
    reply = _edit("t.js", JS_BEFORE, JS_AFTER)
    r = Result(True, "t.js", "applied", before=JS_BEFORE, after=JS_AFTER)
    found = verify.check(agent, "update the imports", reply, [r])
    assert any(f.kind == "static" and "describe" in f.message for f in found)


# ------------------------------------- BUG-311: vanishing gate blockers --
# A gate whose repair produced nothing broke out of the loop without
# unresolved(), so task.findings / RPC blockers showed a clean run.

class _NoRepair:
    def _repair_from_findings(self, *a, **k):
        return ""


def _broken_page(tmp_path):
    _w(tmp_path, "index.html",
       '<!doctype html><html><body><div id="stage"></div>'
       '<script type="module" src="main.js"></script></body></html>')
    _w(tmp_path, "main.js", "document.getElementById('board');\n")


def test_frontend_blocker_survives_a_zero_repair(tmp_path):
    _broken_page(tmp_path)
    cfg = _cfg(frontend_contracts=True, static_gate=False,
               interface_lock=False, repair_rounds=1)
    idx = types.SimpleNamespace(root=tmp_path)
    collected: list = []
    loop._post_edit_tests(
        tmp_path, cfg, _NoRepair(), idx, "wire the page",
        [Result(True, "main.js", "applied")], "reply", None, None,
        ["main.js"], collect=collected)
    assert any(f.kind == "frontend" and f.severity == "error"
               for f in collected), [f.message for f in collected]
    assert any("board" in f.message for f in collected)


# ------------------------------------- BUG-312: the umbrella plan step --
# The leading goal-summary sentence became step 1 and implemented the whole
# plan; every concrete step was then a no-op.

UMBRELLA_TASK = (
    "Bring the Phase-2 feature to a working, tested state.\n"
    "1. In tests/theory.test.js restore the node:test imports.\n"
    "2. In src/pattern.js add generateEuclidean.\n"
    "3. In src/pattern.js add a setStep method.\n"
    "4. In src/ui.js import Pattern and wire the sliders.\n"
    "5. In src/synth.js add the delay node.\n"
    "6. In src/main.js connect the transport.")


def test_umbrella_sentence_is_not_a_plan_step():
    titles = plan.requirement_titles(UMBRELLA_TASK)
    assert len(titles) == 6                     # the cap still holds...
    assert titles[0].startswith("In tests/theory.test.js")
    assert not any(t.startswith("Bring") for t in titles)
    assert titles[-1].startswith("In src/main.js")   # ...and no slot was eaten


def test_umbrella_only_task_has_no_plan():
    assert plan.requirement_titles(
        "Bring the feature to a working, tested state.") == []


def test_single_clause_task_has_no_plan():
    assert plan.requirement_titles("Fix the offset in theory.js.") == []


# ------------------------------------- BUG-313: contradictory finish line --
# "[6 done, 0 failed] / All tasks complete." printed next to exit 1 under
# --strict; the summary now names the strict rejection.

def test_strict_unclean_finish_is_not_all_tasks_complete(tmp_path,
                                                         monkeypatch, capsys):
    _w(tmp_path, "a.py", "GOOD = 1\n")
    plan_path = tmp_path / "PLAN.md"
    loop.write_plan(plan_path, "goal", ["task A"])
    idx = types.SimpleNamespace(root=tmp_path)

    def fake_run_task(root, cfg_, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None, is_repair=False):
        return loop.TaskResult(
            [Result(True, "a.py", "applied"),
             Result(False, "a.py", "search block not found")],
            "reply", True, ["a.py"], idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    cfg = _cfg(static_gate=False, frontend_contracts=False,
               interface_lock=False, use_memory=False, backup=False)
    code = loop.run_loop(tmp_path, cfg, None, idx, plan_path, apply=True,
                         strict=True)
    err = capsys.readouterr().err
    assert code == 1
    assert "All tasks complete." not in err
    assert "NOT clean under --strict" in err
    assert "1 task(s) done, 0 failed" in err    # the honest counters stay
    assert "--strict: task(s) ticked done" in err


# ------------------------------------- BUG-314: repair prompts as requirements --
# The injected diagnostic ("Your last change failed this check...") was mined
# into coverage clauses and could seed R4 edits.

REPAIR_TASK = ("Your last change failed this check:\n\nFRONTEND DOM CONTRACT "
               "GATE\nmain.js queries #board which no template defines.\n\n"
               "Fix these first. Emit search/replace blocks for the files "
               "that need changing.")


def test_repair_gather_mines_no_requirements(tmp_path):
    client = Scripted([{"action": "done"}])
    agent, cfg, idx = _gather_agent(tmp_path, client, {"a.py": "X = 1\n"})
    agent.gather(REPAIR_TASK, verbose=False, is_repair=True)
    assert agent._is_repair is True
    assert agent.requirements.items == []
    assert agent.requirements.task == ""


def test_repair_prompt_gets_no_coverage_verdict(tmp_path):
    agent, cfg, idx = _files(tmp_path, {"main.js": "x();\n"})
    agent._is_repair = True
    r = Result(True, "main.js", "applied", after="y();\n")
    found = verify.check(agent, REPAIR_TASK, "fixed it", [r])
    assert not [f for f in found if f.kind == "coverage"]
    agent._is_repair = False
    found = verify.check(agent, REPAIR_TASK, "fixed it", [r])
    assert [f for f in found if f.kind == "coverage"]   # the old pollution


def test_run_task_forwards_is_repair(tmp_path, monkeypatch):
    seen = {}

    def fake_ladder(root, cfg_, client, agent, idx_, task, **kw):
        seen.update(kw)
        return loop.TaskResult([], "noop", False, [], idx_)

    monkeypatch.setattr(loop, "_run_ladder", fake_ladder)
    cfg = _cfg(use_memory=False)
    idx = types.SimpleNamespace(root=tmp_path)
    loop.run_task(tmp_path, cfg, None, None, idx, REPAIR_TASK, is_repair=True)
    assert seen["is_repair"] is True


# ------------------------------------- BUG-315: drifted LINES anchor --
# A LINES block whose quoted SEARCH still located uniquely was applied at the
# stale range, unbalancing the file; the quoted text is the real anchor.

JS_TAIL = ("describe('tail', () => {\n"
           "  test('works', () => { ok(true); });\n"
           "});\n")
JS_DOC = ("import { x } from './x.js';\n"
          "\n"
          "function head() {\n"
          "  return x;\n"
          "}\n"
          "\n" + JS_TAIL)
APPEND = ("describe('added', () => {\n"
          "  test('new', () => { ok(true); });\n"
          "});\n")


def test_lines_anchor_relocates_to_a_unique_search(tmp_path):
    _w(tmp_path, "f.js", JS_DOC)
    block = (f"--- FILE: f.js LINES 1-2\n<<<<<<< SEARCH\n{JS_TAIL}"
             f"=======\n{JS_TAIL}{APPEND}>>>>>>> REPLACE\n")
    results = apply_edits(tmp_path, parse_edits(block), backup=False,
                          cfg=_cfg())
    assert len(results) == 1 and results[0].ok, results[0].message
    text = (tmp_path / "f.js").read_text(encoding="utf-8")
    assert text.startswith("import { x }")      # the head survived
    assert text.endswith(APPEND)                # the append landed at the tail
    assert text.count("describe(") == 2


def test_lines_anchor_without_search_keeps_the_range(tmp_path):
    _w(tmp_path, "f.js", JS_DOC)
    block = ("--- FILE: f.js LINES 1-2\n<<<<<<< SEARCH\n=======\n"
             "const y = 2;\n>>>>>>> REPLACE\n")
    results = apply_edits(tmp_path, parse_edits(block), backup=False,
                          cfg=_cfg())
    assert results[0].ok, results[0].message
    text = (tmp_path / "f.js").read_text(encoding="utf-8")
    assert text.startswith("const y = 2;\n")
    assert "function head()" in text            # the rest is untouched


# ------------------------------------- BUG-316: always-thinking empty retry --
# The retry path set disable_thinking on a family that cannot stop thinking;
# the budget runaway burned the run instead of escalating or failing fit.

@pytest.mark.parametrize("model", [
    "lfm2.5-24b-a2b", "LFM2.5-24B-A2B-Turbo-Brilliance",
    "lfm2_5-24b-a2b-turbo-brilliance", "lfm-2.5-24b-a2b",
    "LiquidAI_lfm2.5-24b-a2b-instruct",
])
def test_always_thinks_recognises_every_lfm_spelling(model):
    assert llm._always_thinks(model) is True


def test_always_thinks_leaves_other_models_alone():
    assert llm._always_thinks("qwen3-8b") is False
    assert llm._always_thinks("lfm2-7b") is False


def _sse(obj):
    return "data: " + json.dumps(obj)


def _empty_thinking_lines():
    return [_sse({"choices": [{"delta": {"reasoning_content": "hmm " * 40},
                               "finish_reason": None}]}),
            _sse({"choices": [{"delta": {}, "finish_reason": "length"}]})]


class _Resp:
    status_code = 200
    headers: dict = {}

    def __init__(self, lines):
        self._lines = lines

    def iter_lines(self, decode_unicode=False):
        yield from self._lines

    def close(self):
        pass


def _lfm_cfg(**kw):
    cfg = Config()
    cfg.base_url = "http://127.0.0.1:11434/v1"
    cfg.model = "lfm2_5-24b-a2b-turbo-brilliance"
    cfg.log_events = False
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_empty_answer_escalates_the_budget_instead_of_disabling_thinking(
        monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    payloads = []
    answers = iter([_empty_thinking_lines(),
                    [_sse({"choices": [{"delta": {"content": "ANSWER"},
                                        "finish_reason": "stop"}]})]])

    def fake_post(url, **kw):
        payloads.append(kw["json"])
        return _Resp(next(answers))

    monkeypatch.setattr("pedacito.llm.requests.post", fake_post)
    cfg = _lfm_cfg(context_tokens=32768, reserve_tokens=8192,
                   max_tokens_answer=4000)
    out = LLMClient(cfg).chat([{"role": "user", "content": "hi"}], 1500,
                              stream=True)
    assert out == "ANSWER"
    assert len(payloads) == 2
    assert payloads[1]["max_tokens"] == 3000    # doubled, not re-rolled
    for p in payloads:                          # never the futile off switch
        assert p.get("chat_template_kwargs", {}).get("enable_thinking") \
            is not False
        assert p.get("think") is not False


def test_tiny_window_raises_a_model_fit_error(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    payloads = []

    def fake_post(url, **kw):
        payloads.append(kw["json"])
        return _Resp(_empty_thinking_lines())

    monkeypatch.setattr("pedacito.llm.requests.post", fake_post)
    cfg = _lfm_cfg(context_tokens=5408, reserve_tokens=1352,
                   max_tokens_answer=1500)
    with pytest.raises(LLMError) as e:
        LLMClient(cfg).chat([{"role": "user", "content": "hi"}], 1500,
                            stream=True)
    assert "always reasons" in str(e.value)
    assert "context window is too small" in str(e.value)
    budgets = [p["max_tokens"] for p in payloads]
    assert budgets == sorted(budgets) and len(set(budgets)) > 1
    for p in payloads:
        assert p.get("chat_template_kwargs", {}).get("enable_thinking") \
            is not False
        assert p.get("think") is not False


# ------------------------------------- BUG-317: raw tool-call sentinels --

def test_strip_reasoning_removes_tool_call_sentinels():
    raw = ("<|tool_call_start|>[read(path='src/pattern.js'), "
           "read(path='src/theory.js')]<|tool_call_end|>")
    assert llm.strip_reasoning(raw) == ""
    mixed = "Here is the plan.\n" + raw + "\nDone."
    assert llm.strip_reasoning(mixed) == "Here is the plan.\n\nDone."
    unclosed = "text\n<|tool_call_start|>[read(path='x.py')"
    assert llm.strip_reasoning(unclosed) == "text"
    stray = "a <|tool_call_end|> b"
    assert llm.strip_reasoning(stray) == "a  b"
    assert llm.strip_reasoning("plain answer") == "plain answer"


# ------------------------------------- BUG-318: unnamed frontend violations --

def test_frontend_violations_are_named_in_events_and_log(tmp_path):
    _broken_page(tmp_path)
    cfg = _cfg(frontend_contracts=True, static_gate=False,
               interface_lock=False, repair_rounds=0, log_events=True)
    idx = types.SimpleNamespace(root=tmp_path)
    loop._post_edit_tests(
        tmp_path, cfg, None, idx, "wire the page",
        [Result(True, "main.js", "applied")], "reply", None, None,
        ["main.js"], collect=[])
    evs = [e for e in events.load(tmp_path)
           if e.get("kind") == "tests" and e.get("frontend")]
    assert evs and evs[0].get("violations"), evs
    assert any("board" in v for v in evs[0]["violations"])
    log = (tmp_path / ".pedacito" / "log.txt").read_text(encoding="utf-8")
    assert "frontend contract:" in log and "board" in log


# ------------------------------------- BUG-319: stale describe labels --
# Correct-by-contract (strings are never rewritten), but the leftovers are
# now reported so the test report's lie is visible.

def test_rename_reports_string_leftovers_without_touching_them(tmp_path):
    _w(tmp_path, "src/pattern.js",
       "export class Pattern {\n  toggleStep(i) { this.steps[i] ^= 1; }\n}\n")
    _w(tmp_path, "tests/theory.test.js",
       "import { Pattern } from '../src/pattern.js';\n"
       "describe('pattern: toggleStep', () => {\n"
       "  const p = new Pattern();\n  p.toggleStep(0);\n});\n")
    res = refactor.rename_symbol_across_project(tmp_path, "toggleStep",
                                                "flipStep")
    assert res["per_file"].get("src/pattern.js") == 1
    assert res["per_file"].get("tests/theory.test.js") == 1
    leftovers = res["string_matches"]
    assert any(s.startswith("tests/theory.test.js:2:")
               and "pattern: toggleStep" in s for s in leftovers)
    text = (tmp_path / "tests/theory.test.js").read_text(encoding="utf-8")
    assert "describe('pattern: toggleStep'" in text   # string untouched
    assert "p.flipStep(0);" in text                   # code renamed


def test_tool_rename_symbol_surfaces_the_note(tmp_path):
    _w(tmp_path, "src/pattern.js",
       "export class Pattern {\n  toggleStep(i) { this.steps[i] ^= 1; }\n}\n")
    _w(tmp_path, "tests/theory.test.js",
       "describe('pattern: toggleStep', () => {});\n")
    cfg = _cfg()
    idx = index_mod.build([str(tmp_path)], cfg, root=tmp_path,
                          summarise=False, verbose=False)
    out = refactor.tool_rename_symbol(idx, cfg, "toggleStep", "flipStep",
                                      dry_run=True)
    assert "NOTE:" in out and "string literal" in out
    assert "pattern: toggleStep" in out


# ------------------------------------- BUG-320: failing reads loop forever --
# Identical failing read-only steps were exempt from repeat suppression, so
# one bad read ran to the no-progress lease.

def test_identical_failing_reads_are_suppressed(tmp_path):
    # Interleaved repeats: consecutive identical steps already hit the
    # BUG-208 dedup; the failing-read cache must catch the same failed step
    # picked up again after other work, without dispatching it.
    steps = [{"action": "read", "file": "missing.js", "start": 150, "end": 200},
             {"action": "read", "file": "a.py"},
             {"action": "read", "file": "missing.js", "start": 150, "end": 200},
             {"action": "read", "file": "a.py"},
             {"action": "read", "file": "missing.js", "start": 150, "end": 200},
             {"action": "done"}]
    client = Scripted(steps)
    agent, cfg, idx = _gather_agent(tmp_path, client, {"a.py": "X = 1\n"},
                                    log_events=True)
    agent.gather("read missing.js", verbose=False)
    evs = events.load(tmp_path)
    cached = [e for e in evs if e.get("kind") == "cached_failure"]
    assert len(cached) == 2                     # repeats 2+3 never dispatched
    dispatched = [e for e in evs if e.get("kind") == "step"
                  and e.get("action") == "read"
                  and e.get("detail") == "missing.js"]
    assert len(dispatched) == 1                 # only the first one ran
    last_prompt = client.json_calls[-1]
    assert any("already failed at step" in m["content"] for m in last_prompt)


def test_a_mutation_rearms_a_failed_read(tmp_path):
    steps = [{"action": "read", "file": "missing.js", "start": 1, "end": 5},
             {"action": "write", "file": "missing.js",
              "content": "export const x = 1;\n"},
             {"action": "read", "file": "missing.js", "start": 1, "end": 5},
             {"action": "done"}]
    client = Scripted(steps)
    agent, cfg, idx = _gather_agent(tmp_path, client, {"a.py": "X = 1\n"},
                                    log_events=True, allow_write=True)
    gathered = agent.gather("create and read missing.js", verbose=False)
    evs = events.load(tmp_path)
    assert not [e for e in evs if e.get("kind") == "cached_failure"]
    assert "export const x = 1;" in gathered    # the re-read really ran


# ------------------------------------- BUG-321: null step arguments --

def test_null_step_fields_do_not_become_literal_none(tmp_path, capsys):
    client = Scripted([{"action": "read", "file": None, "start": None,
                        "end": None}, {"action": "done"}])
    agent, cfg, idx = _gather_agent(tmp_path, client, {"a.py": "X = 1\n"})
    agent.gather("read the overview", verbose=True)
    err = capsys.readouterr().err
    assert "read None LNone-None" not in err
    assert "(no file)" in err
    last_prompt = client.json_calls[-1]
    assert any("ERROR" in m["content"] for m in last_prompt)


def test_null_range_defaults_are_shown_honestly(tmp_path, capsys):
    client = Scripted([{"action": "read", "file": "a.py", "start": None,
                        "end": None}, {"action": "done"}])
    agent, cfg, idx = _gather_agent(tmp_path, client, {"a.py": "X = 1\n"})
    agent.gather("read a.py", verbose=True)
    err = capsys.readouterr().err
    assert "a.py L1-end" in err and "LNone-None" not in err


# ------------------------------------- BUG-322: Node failures unclassified --
# failure_identities was pytest-shaped; a red node --test suite classified
# as 0/0/0/0 and R2's identity diff was blind.

NODE_TAP = """TAP version 13
# Subtest: legacy tuning
not ok 1 - legacy: wrong A4 tuning assumption
  ---
  error: 'Expected 442 to equal 440'
  location: 'tests/legacy.test.js:5:3'
  ...
# fail 1
"""

NODE_SPEC = """\u2716 legacy: wrong A4 tuning assumption (1.2ms)
  AssertionError [ERR_ASSERTION]: Expected values to be strictly equal:
  442 !== 440
      at TestContext.<anonymous> (tests/legacy.test.js:5:3)
"""


def test_node_tap_failures_get_identities():
    infos = loop.failure_identities(NODE_TAP)
    assert list(infos) == ["node::legacy: wrong A4 tuning assumption"]
    info = next(iter(infos.values()))
    assert "442" in info.signature
    assert info.file == "tests/legacy.test.js" and info.line == 5


def test_node_spec_failures_classify_as_pre_existing():
    diff = loop.classify_failures(NODE_SPEC, NODE_SPEC)
    assert len(diff.pre_existing) == 1 and len(diff.still_failing) == 1
    assert diff.new == [] and diff.fixed == []
    worse = NODE_SPEC + "\n\u2716 another: new failure (0.5ms)\n" \
                        "  AssertionError: boom\n"
    diff2 = loop.classify_failures(NODE_SPEC, worse)
    assert len(diff2.new) == 1 and len(diff2.still_failing) == 1


def test_failure_score_reads_the_node_fail_counter():
    assert loop._failure_score("# fail 2\n") == 2
    assert loop._failure_score("\u2139 fail 1\n") == 1
    assert loop._failure_score("3 failed, 1 error\n") == 4   # pytest intact


# ------------------------------------- BUG-323: red suite absent from findings --

def _red_suite_cfg(tmp_path, **kw):
    _w(tmp_path, "fail.py",
       "print('not ok 1 - legacy: wrong A4')\n"
       "print('# fail 1')\nraise SystemExit(1)\n")
    _w(tmp_path, "a.py", "X = 1\n")
    cfg = _cfg(verify_tests=True, allow_shell=True, static_gate=False,
               frontend_contracts=False, interface_lock=False,
               use_memory=False, backup=False,
               test_command=f"{sys.executable} fail.py", **kw)
    idx = index_mod.build([str(tmp_path)], cfg, root=tmp_path,
                          summarise=False, verbose=False)
    return cfg, idx


def test_pre_existing_red_suite_is_a_warning_finding(tmp_path):
    cfg, idx = _red_suite_cfg(tmp_path, repair_rounds=0)
    out = "not ok 1 - legacy: wrong A4\n# fail 1\n"
    collected: list = []
    loop._post_edit_tests(
        tmp_path, cfg, None, idx, "append a sentence",
        [Result(True, "a.py", "applied")], "reply", None, (False, out),
        ["a.py"], collect=collected)
    tests = [f for f in collected if f.kind == "tests"]
    assert len(tests) == 1 and tests[0].severity == "warn"
    assert "pre-existing" in tests[0].message


def test_new_red_suite_is_an_error_finding(tmp_path):
    cfg, idx = _red_suite_cfg(tmp_path, repair_rounds=0)
    collected: list = []
    loop._post_edit_tests(
        tmp_path, cfg, None, idx, "append a sentence",
        [Result(True, "a.py", "applied")], "reply", None, None,
        ["a.py"], collect=collected)
    tests = [f for f in collected if f.kind == "tests"]
    assert len(tests) == 1 and tests[0].severity == "error"
    assert "not ok 1" in tests[0].message       # the output tail rides along


# ------------------------------------- BUG-324: the empty dropdown passes --

def test_empty_select_is_flagged(tmp_path):
    _w(tmp_path, "index.html",
       '<!doctype html><html><body><select id="preset-bank"></select>'
       '<script type="module" src="app.js"></script></body></html>')
    _w(tmp_path, "app.js", "export const x = 1;\n")
    found = webcheck.run(tmp_path)
    empty = [f for f in found if "option" in f.message]
    assert len(empty) == 1 and "preset-bank" in empty[0].message


def test_runtime_option_population_suppresses_the_finding(tmp_path):
    _w(tmp_path, "index.html",
       '<!doctype html><html><body><select id="preset-bank"></select>'
       '<script type="module" src="app.js"></script></body></html>')
    _w(tmp_path, "app.js",
       "const sel = document.getElementById('preset-bank');\n"
       "sel.options.add(new Option('warm', 'warm'));\n")
    found = webcheck.run(tmp_path)
    assert not [f for f in found if "no <option> children" in f.message]


def test_select_with_options_is_clean(tmp_path):
    _w(tmp_path, "index.html",
       '<!doctype html><html><body>'
       '<select id="preset-bank"><option value="warm">warm</option></select>'
       '<script type="module" src="app.js"></script></body></html>')
    _w(tmp_path, "app.js", "export const x = 1;\n")
    found = webcheck.run(tmp_path)
    assert not [f for f in found if "no <option> children" in f.message]


# ------------------------------------- BUG-325: domcheck's false "bound" --
# addEventListener(type, undefined) recorded a binding; a listener on a node
# later removed masked an unbound control. Live checks need Playwright.

_HAS_PW = domcheck.available()[0]
requires_pw = pytest.mark.skipif(
    not _HAS_PW, reason="Playwright + Chromium not installed")

_DOM_INDEX = ('<!doctype html><html><head><title>t</title></head>'
              '<body><button id="go">Go</button><div id="box"></div>'
              '<script type="module" src="app.js"></script></body></html>')


def _dom_project(tmp_path, app_js):
    (tmp_path / "index.html").write_text(_DOM_INDEX)
    (tmp_path / "app.js").write_text(app_js)
    return tmp_path


@requires_pw
def test_undefined_handler_is_not_a_binding(tmp_path):
    _dom_project(tmp_path,
                 "const handlers = {};\n"
                 "document.getElementById('go')"
                 ".addEventListener('click', handlers.onPlay);\n")
    res = domcheck.check(tmp_path)
    assert not res.error, res.error
    assert any(f.kind == "unbound_control" and "go" in f.message
               for f in res.findings), [str(f) for f in res.findings]


@requires_pw
def test_real_handler_binds_the_control(tmp_path):
    _dom_project(tmp_path,
                 "document.getElementById('go')"
                 ".addEventListener('click', () => {});\n")
    res = domcheck.check(tmp_path)
    assert not res.error, res.error
    assert not [f for f in res.findings if f.kind == "unbound_control"]


@requires_pw
def test_detached_listener_does_not_mask_an_unbound_control(tmp_path):
    _dom_project(tmp_path,
                 "const box = document.getElementById('box');\n"
                 "const b = document.createElement('button');\n"
                 "b.addEventListener('click', () => {});\n"
                 "box.appendChild(b);\n"
                 "box.innerHTML = '';\n")     # the bound node is now detached
    res = domcheck.check(tmp_path)
    assert not res.error, res.error
    assert any(f.kind == "unbound_control" and "go" in f.message
               for f in res.findings), [str(f) for f in res.findings]


# ------------------------------------- BUG-326: partial profile replacement --

def test_partial_overlay_profile_merges_over_the_user_profile(tmp_path,
                                                              monkeypatch):
    user = tmp_path / "user.toml"
    user.write_text(
        '[profiles.p]\n'
        'base_url = "http://127.0.0.1:11434/v1"\n'
        'model = "user-model"\n'
        'vram_gb = 24\n'
        'repair_rounds = 2\n', encoding="utf-8")
    monkeypatch.setattr(config_mod, "USER_CONFIG", user)
    overlay = tmp_path / "overlay.toml"
    overlay.write_text('[profiles.p]\nrepair_rounds = 0\n', encoding="utf-8")
    cfg = config_mod.load(explicit=overlay, use_env=False, profile="p",
                          require_hardware=False, warn_budgets=False)
    assert cfg.base_url == "http://127.0.0.1:11434/v1"
    assert cfg.model == "user-model"
    assert cfg.repair_rounds == 0               # the overlay key still wins


def test_full_overlay_profile_still_replaces_keys(tmp_path, monkeypatch):
    user = tmp_path / "user.toml"
    user.write_text('[profiles.p]\nbase_url = "http://127.0.0.1:1/v1"\n'
                    'model = "user-model"\nvram_gb = 24\n', encoding="utf-8")
    monkeypatch.setattr(config_mod, "USER_CONFIG", user)
    overlay = tmp_path / "overlay.toml"
    overlay.write_text('[profiles.p]\nmodel = "other-model"\n',
                       encoding="utf-8")
    cfg = config_mod.load(explicit=overlay, use_env=False, profile="p",
                          require_hardware=False, warn_budgets=False)
    assert cfg.model == "other-model"
    assert cfg.base_url == "http://127.0.0.1:1/v1"


# ------------------------------------- BUG-327: verified NOT a bug --
# observe_command returns after retiring the contradicted fact; the pending
# row appears only on a LATER call, and command_was_failing scans all three
# lists, so R2's baseline verdict is never ambiguous.

def test_retire_and_pending_are_sequential_not_duplicate(tmp_path):
    cmd = "npm test"
    assert memory.observe_command(tmp_path, cmd, True) is None
    promoted = memory.observe_command(tmp_path, cmd, True)
    assert promoted is not None                 # two successes promote
    fact_id = promoted["fact_id"]
    assert memory.observe_command(tmp_path, cmd, False) is None
    doc = json.loads(memory.micro_facts_file(tmp_path).read_text(
        encoding="utf-8"))
    retired = [f for f in doc["retired"] if f["fact_id"] == fact_id]
    assert len(retired) == 1
    assert retired[0]["contradictions"] == 1 and retired[0]["last_ok"] is False
    assert not [p for p in doc["pending"] if p["fact_id"] == fact_id]
    assert memory.command_was_failing(tmp_path, cmd) is True
    # the NEXT failure opens the pending trail -- one row, not a duplicate
    memory.observe_command(tmp_path, cmd, False)
    doc = json.loads(memory.micro_facts_file(tmp_path).read_text(
        encoding="utf-8"))
    pending = [p for p in doc["pending"] if p["fact_id"] == fact_id]
    assert len(pending) == 1 and pending[0]["failures"] == 1
    assert memory.command_was_failing(tmp_path, cmd) is True


# ------------------------------------- BUG-328: the clipped clause tail --
# The ledger kept only the first 140 chars and 8 occurrence-order terms, so
# the handler contract at the tail of a long clause was never a requirement.

LONG_CLAUSE = (
    "In src/main.js remove the duplicate second window keydown listener and "
    "its pianoKeys map so each key plays once, pass onPlay onStop "
    "onBpmChange onSwingChange onFilterChange onDelayChange onPresetChange "
    "to buildTransport, add presets: Object.values(PRESETS), and drop the "
    "stale lookup")


def test_long_clause_keeps_its_full_text_and_tail_terms():
    reqs = verify._requirements_from_task(LONG_CLAUSE)
    assert len(reqs) == 1
    req = reqs[0]
    assert req.text == LONG_CLAUSE              # no 140-char clip
    assert not req.text.endswith(" ...")
    assert "onPresetChange" in req.terms        # the distinctive tail survives
    assert "pianoKeys" in req.terms and "buildTransport" in req.terms


def test_plan_titles_keep_the_full_clause_for_the_step():
    task = (f"1. {LONG_CLAUSE}.\n2. In src/ui.js populate the preset "
            "dropdown options from PRESETS.")
    titles = plan.requirement_titles(task)
    assert len(titles) == 2
    assert titles[0] == LONG_CLAUSE             # display clip is print-only
    assert len(titles[0]) > 140


# ------------------------------------- BUG-329: the auto-read bug log --
# A generic term located a clause in an unrelated reference doc and the
# requirement assist pulled it into the gather context.

def test_requirement_assist_skips_unrelated_reference_docs(tmp_path):
    tree = {"BUG_LOG.md": "# Bug log\n\nSome comment about the run.\n",
            "src/app.py": "# a comment\ndef encodePattern():\n    pass\n"}
    agent, cfg, idx = _files(tmp_path, tree)
    tracker = verify.RequirementTracker(
        "Update the comment above encodePattern")
    obs = ("BUG_LOG.md:1: Some comment about the run.\n"
           "src/app.py:1: # a comment\n")
    tracker.observe_search("grep", {"pattern": "comment"}, obs, idx)
    action, req = tracker.next_action(idx)
    assert action is not None
    assert action.get("file") != "BUG_LOG.md"   # the prose log stays out
    assert action.get("file") == "src/app.py"


def test_requirement_assist_reads_a_doc_the_task_names(tmp_path):
    tree = {"BUG_LOG.md": "# Bug log\n\nSome comment about the run.\n"}
    agent, cfg, idx = _files(tmp_path, tree)
    tracker = verify.RequirementTracker(
        "Read BUG_LOG.md and summarize the comment")
    obs = "BUG_LOG.md:1: Some comment about the run.\n"
    tracker.observe_search("grep", {"pattern": "comment"}, obs, idx)
    action, req = tracker.next_action(idx)
    assert action is not None and action.get("file") == "BUG_LOG.md"