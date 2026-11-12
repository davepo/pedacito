"""Regression tests for the second bug-hunting report
(`PEDACITO_BUG_LOG_2.md`, PulseForge-16 run, Nov 2026).

Like `test_buglog_validation.py`, each section pins the deterministic half of
one reported finding: the gate that did not exist, the verdict that lied, the
budget that was never enforced. Findings whose cause is purely model behaviour
with no harness-side determinant are exercised through the helper the fix
added, not by re-running a live model.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import (llm, loop, memory, rpc, scaffold, spec, tools,  # noqa: E402
                      verify, webcheck)
from pedacito.agent import Agent  # noqa: E402
from pedacito.config import Config  # noqa: E402


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


def _files(tmp_path, tree: dict[str, str]):
    from pedacito import index as index_mod
    for rel, text in tree.items():
        _w(tmp_path, rel, text)
    cfg = _cfg()
    idx = index_mod.build([str(tmp_path)], cfg, root=tmp_path,
                          summarise=False, verbose=False)
    agent = Agent(idx, None, cfg)
    return agent, cfg, idx


def _edit(file, search, replace):
    return (f"--- FILE: {file}\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n"
            ">>>>>>> REPLACE\n")


# ------------------------------------------- BUG-201: failed build repair --
# A per-file build pass whose edit was rejected used to leave the stub behind
# with no second chance; only web-contract errors got a repair pass.

def test_build_repair_note_carries_diagnostic_and_window(tmp_path):
    _w(tmp_path, "src/theory.js",
       "export function midiToHz(midi) {\n"
       "  throw new Error(\"not implemented\");\n}\n")
    target = scaffold.PlannedFile("src/theory.js", "theory")
    bad = [types.SimpleNamespace(
        file="src/theory.js",
        message="edit produced a syntax error near line 2; not written")]
    note = scaffold._build_repair_note(target, bad, tmp_path)
    assert "REPAIR" in note
    assert "syntax error near line 2" in note
    assert "current file around line 2" in note
    assert "not implemented" in note          # the actual file window


def test_build_project_reasks_once_then_reports_failed(tmp_path, monkeypatch):
    _w(tmp_path, "a.py", "def a():\n    return 1\n")
    plan = scaffold.Plan("demo", "s", [scaffold.PlannedFile("a.py", "a")])
    idx = scaffold.skeleton_index(tmp_path, plan, Config())
    calls = {"n": 0, "notes": []}

    def fake_build_pass(client, cfg, idx_, plan_, target, done, verbose=True,
                        backup_dir=None, repair_note=""):
        calls["n"] += 1
        calls["notes"].append(repair_note)
        # always rejected: the loop must re-ask exactly once, then give up
        return [scaffold.Result(False, "a.py", "syntax error near line 1")], ""

    monkeypatch.setattr(scaffold, "build_pass", fake_build_pass)
    summary = scaffold.build_project(None, Config(), idx, plan, verbose=False)
    assert calls["n"] == 2                     # one build + one repair re-ask
    assert calls["notes"][0] == ""             # first attempt: no note
    assert "REPAIR" in calls["notes"][1]       # second attempt: the brief
    assert summary["failed"] and summary["done"] == []


def test_build_project_repair_can_succeed(tmp_path, monkeypatch):
    _w(tmp_path, "a.py", "def a():\n    return 1\n")
    plan = scaffold.Plan("demo", "s", [scaffold.PlannedFile("a.py", "a")])
    idx = scaffold.skeleton_index(tmp_path, plan, Config())
    state = {"n": 0}

    def fake_build_pass(client, cfg, idx_, plan_, target, done, verbose=True,
                        backup_dir=None, repair_note=""):
        state["n"] += 1
        if state["n"] == 1:
            return [scaffold.Result(False, "a.py", "syntax error")], ""
        return [scaffold.Result(True, "a.py", "1 block applied")], ""

    monkeypatch.setattr(scaffold, "build_pass", fake_build_pass)
    summary = scaffold.build_project(None, Config(), idx, plan, verbose=False)
    assert summary["done"] == ["a.py"] and summary["failed"] == []


# ------------------------------------------- BUG-202: the clause splitter --
# A `;` inside a parenthetical became two plan tasks; a verbless fragment
# became a nonsense step. Depth/quote tracking + a verb-aware merge fix it,
# without swallowing an explicit bulleted requirement.

def test_splitter_keeps_parenthetical_semicolon_together():
    reqs = verify.requirement_ledger(
        "Parse pitch (sharps allowed; return null for bad input)")
    assert len(reqs) == 1
    assert "sharps allowed; return null" in reqs[0][0]


def test_splitter_keeps_bracket_and_quote_semicolons_together():
    reqs = verify.requirement_ledger(
        'Set the array [1; 2] and the label "a; b" before drawing')
    assert len(reqs) == 1


def test_bulleted_requirements_stay_separate_even_with_unknown_verb():
    titles = plan_titles(
        "- add retries to sync\n- log every retry\n- document it in the README")
    assert len(titles) == 3
    assert "document it in the README" in titles


def test_numbered_requirements_stay_separate():
    titles = plan_titles(
        "1. add retries\n2. document the change\n3. log every retry")
    assert len(titles) == 3


def test_single_change_is_not_planned():
    assert plan_titles("fix the typo in utils.py") == []


def test_verbless_soft_fragment_merges_into_previous():
    # a soft-boundary fragment with no verb of its own is a continuation,
    # not a requirement of its own
    reqs = verify.requirement_ledger("Add sharps support; with a null fallback")
    assert len(reqs) == 1


def plan_titles(task):
    from pedacito import plan
    return plan.requirement_titles(task)


# ------------------------------------------- BUG-203: tests actually run ---
# The task named `node --test` but run_test was pytest-only, and the final
# verdict never said the tests were skipped.

def test_task_named_tests_detects_node_and_pytest():
    assert "node --test" in loop._task_named_tests(
        "Make node --test tests/theory.test.js pass.")
    assert loop._task_named_tests("run pytest -q") != ""
    assert loop._task_named_tests("npm test") != ""
    assert loop._task_named_tests("add a button") == ""


def test_note_tests_not_run_emits_event(tmp_path, capsys):
    cfg = _cfg()
    loop._note_tests_not_run(tmp_path, cfg,
                             "Make node --test tests/theory.test.js pass.")
    err = capsys.readouterr().err
    assert "tests: NOT RUN" in err and "node --test" in err


def test_note_tests_not_run_silent_when_no_test_named(tmp_path, capsys):
    loop._note_tests_not_run(tmp_path, _cfg(), "add a button")
    assert "NOT RUN" not in capsys.readouterr().err


def _dispatch_run_test(root, **kw):
    cfg = _cfg(sandbox=False)
    cfg._root = str(root)
    idx = types.SimpleNamespace(root=root)
    return tools.dispatch(idx, cfg, {"action": "run_test", **kw})


@pytest.mark.skipif(not __import__("shutil").which("node"),
                    reason="node not on PATH")
def test_run_test_runs_a_node_suite(tmp_path):
    _w(tmp_path, "tests/theory.test.js",
       'const {test}=require("node:test");const assert=require("node:assert");\n'
       'test("good",()=>{assert.equal(1,1);});\n'
       'test("bad",()=>{assert.equal(1,2);});\n')
    out = _dispatch_run_test(tmp_path, target="tests/theory.test.js")
    assert "EXIT: 1 (FAILED)" in out
    assert "SUMMARY:" in out and "fail 1" in out
    assert "node --test" in out.splitlines()[0]


@pytest.mark.skipif(not __import__("shutil").which("node"),
                    reason="node not on PATH")
def test_run_test_node_green_and_python_not_hijacked(tmp_path):
    _w(tmp_path, "tests/ok.test.js",
       'const {test}=require("node:test");const assert=require("node:assert");\n'
       'test("good",()=>{assert.equal(1,1);});\n')
    out = _dispatch_run_test(tmp_path, target="tests/ok.test.js")
    assert "EXIT: 0 (OK)" in out and "PASSED" in out


def test_run_test_python_project_still_uses_pytest(tmp_path):
    _w(tmp_path, "pyproject.toml", '[project]\nname="d"\nversion="0"\n')
    _w(tmp_path, "tests/test_math.py", "def test_ok():\n    assert 1==1\n")
    out = _dispatch_run_test(tmp_path)
    assert "EXIT: 0 (OK)" in out and "1 passed" in out


@pytest.mark.skipif(not __import__("shutil").which("node"),
                    reason="node not on PATH")
def test_run_test_node_refuses_traversal(tmp_path):
    _w(tmp_path, "a.test.js", "test('x',()=>{});\n")
    out = _dispatch_run_test(tmp_path, target="../../etc/passwd")
    assert out.startswith("ERROR")


# ------------------------------------------- BUG-204: UTF-8 at the source --
# requests defaults a charset-less reply to ISO-8859-1, turning ▶ into â¶.

def test_llm_post_forces_utf8_decoding(monkeypatch):
    import requests
    cfg = _cfg()
    client = llm.LLMClient(cfg)
    resp = requests.Response()
    resp.status_code = 200
    resp._content = "▶ Play ⏹ Stop".encode("utf-8")
    resp.headers["Content-Type"] = "text/event-stream"   # no charset
    resp.encoding = "ISO-8859-1"                        # the wrong default
    monkeypatch.setattr(client, "_http_post", lambda *a, **k: resp)
    monkeypatch.setattr(client, "_check_circuit", lambda: None)
    monkeypatch.setattr(client, "_record_success", lambda: None)
    r = client._post({"model": "m"})
    assert r.encoding == "utf-8"
    assert r.text == "▶ Play ⏹ Stop"


# ------------------------------------------- BUG-205: test-clause evidence --
# "Add unit tests" demands a declaration-bearing edit; an import-only hunk to
# a test file is not evidence, and "make tests pass" is an implementation job.

def test_test_author_clause_recognised():
    assert verify._is_test_author_clause("Add unit tests for generateEuclidean")
    assert verify._is_test_author_clause("write a test for the parser")
    # "make ... pass" is NOT authoring: it asks for implementation evidence
    assert not verify._is_test_author_clause("make node --test tests/x.js pass")


def test_test_decl_regex_needs_a_declaration():
    assert verify._TEST_DECL_RE.search("def test_euclid():")
    assert verify._TEST_DECL_RE.search('test("euclid", () => {});')
    assert verify._TEST_DECL_RE.search("assert x == 1")
    # an import line mentions the symbol but declares no test
    assert not verify._TEST_DECL_RE.search(
        'import { generateEuclidean } from "./pattern.js";')


# ------------------------------------------- BUG-206: two blocks, one file --
# Appending tests is the commonest two-block shape; rejecting it stranded the
# feature half-implemented.

def test_incremental_accepts_two_blocks_for_same_file(tmp_path):
    agent, cfg, idx = _files(tmp_path, {
        "src/pattern.js": "export function f() {\n  return 1;\n}\n"})
    reply = (_edit("src/pattern.js", "  return 1;", "  return 2;")
             + _edit("src/pattern.js", "export function f() {",
                     "export function f() {\n  // appended"))
    results, findings = agent._check_incremental_hunk(
        "task", "", "src/pattern.js", "", reply)
    assert results and all(r.ok for r in results), \
        [r.message for r in results]


def test_incremental_still_refuses_other_file(tmp_path):
    agent, cfg, idx = _files(tmp_path, {
        "src/pattern.js": "export function f() {\n  return 1;\n}\n",
        "src/other.js": "export const x = 1;\n"})
    # a block for the target AND one for another file: the microloop owns one
    # target at a time, so the stray other-file block is refused
    reply = (_edit("src/pattern.js", "  return 1;", "  return 2;")
             + _edit("src/other.js", "export const x = 1;", "export const x = 2;"))
    results, _ = agent._check_incremental_hunk(
        "task", "", "src/pattern.js", "", reply)
    assert results and not results[0].ok
    assert "current symbol is in" in results[0].message


# ------------------------------------------- BUG-207: one finding per clause --
# The same coverage clause surfaced once per hunk and again at the final gate.

def test_dedupe_findings_by_kind_file_message():
    F = verify.Finding
    a = F("coverage", "warn", "a.js", "no edit addresses 'Add tests'")
    b = F("coverage", "warn", "a.js", "no edit addresses 'Add tests'")
    c = F("coverage", "warn", "b.js", "no edit addresses 'Add tests'")
    d = F("syntax", "error", "a.js", "boom")
    out = verify.dedupe_findings([a, b, c, d, a])
    assert len(out) == 3
    assert out[0] is a and out[1] is c and out[2] is d


def test_dedupe_findings_empty_and_none():
    assert verify.dedupe_findings([]) == []
    assert verify.dedupe_findings(None) == []


# ------------------------------------------- BUG-209: honest context budget --
# With no VRAM headroom the fallback window is a guess; clamp it to the
# server-reported window so source budgets stay honest.

def test_config_defaults_for_task_budget():
    c = Config()
    assert c.max_task_completion_tokens == 60000
    assert c.max_task_seconds == 900


# ------------------------------------------- BUG-210: const reassignment ---
# Parse-clean JS that throws `TypeError: Assignment to constant variable`.

def _spec_run(tmp_path, files):
    for rel, text in files.items():
        _w(tmp_path, rel, text)
    return (spec._js_const_reassign_errors(tmp_path),
            spec._js_instance_method_errors(tmp_path),
            spec.verify_frontend_contracts(tmp_path))


def test_spec_flags_const_reassignment(tmp_path):
    c, i, fe = _spec_run(tmp_path, {"src/pattern.js":
        "export function gen(steps, pulses) {\n"
        "  const spaces = steps - pulses;\n"
        "  let remainder = spaces % pulses;\n"
        "  spaces -= remainder;\n"
        "  return spaces;\n}\n"})
    assert len(c) == 1 and "spaces" in c[0][2]
    assert any("spaces" in e for e in fe)


def test_spec_const_forms_and_shadowing(tmp_path):
    c, _, _ = _spec_run(tmp_path, {"a.js":
        "const x = 1;\nx++;\n"
        "const ys = [1, 2];\nfor (const y of ys) { y = 3; }\n"
        "let z = 0; z = 5;\n"
        "function f(x) { x = 2; return x; }\n"})
    # x++ and the for-of write; z is let, the inner x is a parameter
    assert len(c) == 2


def test_webcheck_flags_const_reassignment(tmp_path):
    _w(tmp_path, "a.js", "const x = 1; x = 2;\n")
    f = webcheck.run(tmp_path)
    assert any("'x'" in str(x) and "const" in str(x) for x in f)


def test_webcheck_const_compound_and_quiet_cases(tmp_path):
    _w(tmp_path, "a.js",
       "const spaces = 5;\nspaces -= 1;\n"
       "let y = 1; y = 2;\n"
       "const o = {}; o.x = 3;\n")
    f = webcheck.run(tmp_path)
    assert any("spaces" in str(x) for x in f)
    assert not any("'y'" in str(x) for x in f)
    assert not any("o.x" in str(x) for x in f)


# ------------------------------------------- BUG-211: missing instance method --
# `engine.setBpm()` on a class that defines no setBpm throws on first use.

_SYNTH = ("export class SynthEngine {\n"
          "  constructor() { this.ctx = null; }\n"
          "  setCutoff(freq) { return freq; }\n"
          "  setResonance(q) { return q; }\n"
          "  setDelayFeedback(amount) { return amount; }\n}\n")


def test_spec_flags_missing_instance_method(tmp_path):
    c, i, fe = _spec_run(tmp_path, {
        "src/synth.js": _SYNTH,
        "src/ui.js": ("import { SynthEngine } from \"./synth.js\";\n"
                      "const engine = new SynthEngine();\n"
                      "export function wire() {\n"
                      "  engine.setBpm(120);\n"
                      "  engine.setCutoff(800);\n}\n"),
        "index.html": ('<html><body><script type="module" '
                       'src="src/ui.js"></script></body></html>'),
    })
    assert len(i) == 1 and "setBpm" in i[0][2] and "SynthEngine" in i[0][2]
    assert any("setBpm" in e for e in fe)


def test_spec_instance_this_x_and_ambiguity(tmp_path):
    _, i, _ = _spec_run(tmp_path, {"a.js":
        "class Engine {\n"
        "  constructor() { this.core = new Core(); }\n"
        "  run() { this.core.spin(); this.core.missing(); }\n}\n"
        "class Core { spin() {} }\n"})
    assert len(i) == 1 and "missing" in i[0][2]
    _, i2, _ = _spec_run(tmp_path / "sub", {"a.js":
        "class A { m() {} }\n"
        "let x = new A(); x = new A(); x.nope();\n"
        "const y = new A(); function f(y) { y.nope(); }\n"
        "y.toString();\n"})
    assert i2 == []


def test_webcheck_flags_missing_instance_method(tmp_path):
    _w(tmp_path, "src/synth.js", _SYNTH)
    _w(tmp_path, "src/main.js",
       "import { SynthEngine } from './synth.js';\n"
       "const engine = new SynthEngine();\n"
       "engine.setBpm(120);\nengine.setCutoff(800);\n")
    f = webcheck.run(tmp_path)
    hit = [x for x in f if "setBpm" in str(x)]
    assert hit and "SynthEngine" in str(hit[0])
    assert hit[0].peer == "src/synth.js"
    assert not any("calls engine.setCutoff" in str(x) for x in f)


def test_webcheck_instance_quiet_shapes(tmp_path):
    _w(tmp_path, "a.js",
       "class A extends B { m() {} }\n"
       "const x = new A(); x.unknown();\n"
       "class C { [k]() {} m() {} }\n"
       "const c = new C(); c.whatever();\n")
    f = webcheck.run(tmp_path)
    assert not any("no such method" in str(x) for x in f)


def test_webcheck_instance_param_receiver_quiet(tmp_path):
    # a receiver passed through a parameter is a dataflow question the regex
    # level cannot answer; it stays quiet (spec.py does too)
    _w(tmp_path, "src/synth.js", _SYNTH)
    _w(tmp_path, "src/ui.js",
       "export function wireTransport(container, engine, pattern) {\n"
       "  engine.setBpm(120);\n}\n")
    f = webcheck.run(tmp_path)
    assert not any("no such method" in str(x) for x in f)


# ------------------------------------------- BUG-212: an honest exit code ---
# A run whose every edit block failed used to exit 0 reporting `applied`.

def test_is_noop_matches_fabricated_marker():
    assert loop._is_noop("[No changes: the function is already correct]")
    assert loop._is_noop("No change needed; already implemented.")
    assert not loop._is_noop("I added the retry handling to sync.")


def test_rpc_summary_applied_only_when_edits_landed():
    res = types.SimpleNamespace(
        reply="done", applied=False, ok=True, steps=1, tests_ok=None,
        stats={}, results=[types.SimpleNamespace(file="a.py", ok=False)],
        findings=[])
    s = rpc.RpcServer._summary(res)
    assert s["applied"] is False and s["changed"] == []


# ------------------------------------------- BUG-214: edit-only evidence ----
# Coverage terms must match the edit's content, not the whole reply prose.

def test_edit_evidence_text_is_file_plus_patch():
    e = types.SimpleNamespace(file="a.js", search="old line", replace="new line")
    text = verify._edit_evidence_text(e)
    assert "a.js" in text and "new line" in text and "old line" in text


# ------------------------------------------- BUG-215: provenance in the RPC --
# The clause map / legend / unaccepted clauses used to reach only the review
# page; machine clients got bare edits.

def test_task_result_carries_provenance_fields():
    tr = loop.TaskResult([], "reply", False, [], None)
    assert tr.clauses == {} and tr.clause_texts == {} and tr.unaccepted == []


def test_rpc_summary_carries_provenance():
    res = types.SimpleNamespace(
        reply="done", applied=True, ok=True, steps=1, tests_ok=None,
        stats={}, results=[types.SimpleNamespace(file="src/a.js", ok=True)],
        findings=[],
        clauses={"src/a.js": ["R1"]},
        clause_texts={"R1": "Add retries to sync"},
        unaccepted=["R2"])
    s = rpc.RpcServer._summary(res)
    assert s["clause_texts"] == {"R1": "Add retries to sync"}
    assert s["unaccepted"] == ["R2"]
    assert s["edits"][0]["clauses"] == ["R1"]


def test_rpc_summary_provenance_defaults_for_old_shapes():
    res = types.SimpleNamespace(reply="x", applied=True, ok=True, steps=1,
                                stats={}, results=[], findings=[])
    s = rpc.RpcServer._summary(res)
    assert s["clause_texts"] == {} and s["unaccepted"] == []


# ------------------------------------------- BUG-216: lost wiring ----------
# A whole-file rewrite that drops imported-and-called helpers deletes working
# wiring; error by default, warn only when the task asks to rewrite that file.

def _lost_wiring_agent(tmp_path):
    before = ("import { buildGrid } from './ui.js';\n"
              "import { SynthEngine } from './synth.js';\n"
              "const engine = new SynthEngine();\n"
              "buildGrid(document.body, engine);\n")
    after = "export const placeholder = 1;\n"
    agent, cfg, idx = _files(tmp_path, {
        "src/main.js": before,
        "src/ui.js": "export function buildGrid(el, e) { return el; }\n",
        "src/synth.js": "export class SynthEngine { start() {} }\n"})
    r = types.SimpleNamespace(ok=True, file="src/main.js", before=before,
                              after=after, message="replaced contents")
    return agent, r


def test_lost_wiring_is_error_by_default(tmp_path):
    agent, r = _lost_wiring_agent(tmp_path)
    f = verify.check_lost_wiring(agent, [r], "add a share button")
    assert f and f[0].kind == "lost_wiring" and f[0].severity == "error"
    assert "buildGrid" in f[0].message


def test_lost_wiring_warns_when_task_asks_rewrite(tmp_path):
    agent, r = _lost_wiring_agent(tmp_path)
    f = verify.check_lost_wiring(
        agent, [r], "rewrite src/main.js from scratch to simplify it")
    assert f and f[0].severity == "warn"


def test_lost_wiring_ignores_non_rewrite_results(tmp_path):
    agent, r = _lost_wiring_agent(tmp_path)
    r.message = "1 block applied"          # not a whole-file replacement
    assert verify.check_lost_wiring(agent, [r], "add a share button") == []


# ------------------------------------------- BUG-217: honest pending facts --
# A pending micro-fact must not assert the command works; a failure streak is
# the opposite of evidence.

def test_pending_fact_unconfirmed_and_names_failures():
    t = memory._pending_fact_text("node --test", "test", 0, 3)
    assert "unconfirmed" in t and "failed 3 times" in t
    assert "may work" not in t
    t1 = memory._pending_fact_text("node --test", "test", 0, 1)
    assert "failed 1 time" in t1
    t0 = memory._pending_fact_text("pytest", "test", 0, 0)
    assert "unconfirmed" in t0 and "0 runs" in t0


def test_pending_fact_with_success_still_unconfirmed():
    t = memory._pending_fact_text("pytest", "test", 1, 0)
    assert "may work" in t and "not yet confirmed" in t


# ------------------------------------------- BUG-218: a per-task budget -----
# A 4B model burned 836s / 74k output tokens on a zero-edit run; the task now
# has a wall-clock and output-token ceiling.

def _budget_agent(tmp_path, **cfgkw):
    agent, cfg, idx = _files(tmp_path, {"a.py": "x = 1\n"})
    for k, v in cfgkw.items():
        setattr(agent.cfg, k, v)
    return agent


def test_task_budget_exceeded_on_tokens(tmp_path):
    agent = _budget_agent(tmp_path, max_task_completion_tokens=100,
                          max_task_seconds=0)
    client = types.SimpleNamespace(stats={"completion_tokens": 500})
    agent._task_t0 = __import__("time").time()
    agent._task_clients = [client]
    agent._task_tokens0 = 0
    assert agent._task_budget_exceeded("answer") is True
    # the verdict is sticky: a second call does not re-print
    assert agent._task_budget_exceeded("answer") is True


def test_task_budget_exceeded_on_wall_clock(tmp_path):
    agent = _budget_agent(tmp_path, max_task_completion_tokens=0,
                          max_task_seconds=1)
    agent._task_t0 = __import__("time").time() - 5
    agent._task_clients = [types.SimpleNamespace(stats={})]
    agent._task_tokens0 = 0
    assert agent._task_budget_exceeded("gather") is True


def test_task_budget_off_without_snapshot(tmp_path):
    agent = _budget_agent(tmp_path, max_task_completion_tokens=1,
                          max_task_seconds=1)
    agent._task_t0 = 0.0                 # no task snapshot
    assert agent._task_budget_exceeded("answer") is False


def test_task_budget_within_limits(tmp_path):
    agent = _budget_agent(tmp_path, max_task_completion_tokens=10000,
                          max_task_seconds=900)
    agent._task_t0 = __import__("time").time()
    agent._task_clients = [types.SimpleNamespace(
        stats={"completion_tokens": 10})]
    agent._task_tokens0 = 0
    assert agent._task_budget_exceeded("answer") is False
