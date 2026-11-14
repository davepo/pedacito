"""Regression tests for the seventh bug-hunting report
(`PEDACITO_BUG_LOG_7.md`, Nov 2026).

Like the earlier buglog validation files, each section pins the deterministic
half of one reported finding: the gate that did not exist, the verdict that
lied, the count that inflated. Findings whose cause is purely model behaviour
are exercised through the helper the fix added, not by re-running a live
model.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import (  # noqa: E402
    llm, loop, plan, refactor, scaffold, testsel, verify, webcheck)
from pedacito import index as index_mod  # noqa: E402
from pedacito.agent import Agent  # noqa: E402
from pedacito.config import Config  # noqa: E402
from pedacito.edits import Result  # noqa: E402
from pedacito.llm import LLMClient  # noqa: E402


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
        self.chat_calls: list[list[dict]] = []

    def chat_json(self, messages, schema, max_tokens=0, **kw):
        step = self.steps.pop(0) if self.steps else {"action": "done"}
        return {"reasoning": "r", **step}

    def chat(self, messages, max_tokens=0, stream=False, **kw):
        self.chat_calls.append([dict(m) for m in messages])
        return self.replies.pop(0) if self.replies else "ok"


# --------------------------------------------------------------------------
# BUG-201 / BUG-202: the web-contract repair loop counted block rows, re-ran
# degenerate repeats, and shipped a "fix" that deleted a required export.
# --------------------------------------------------------------------------

def _web_plan(tmp_path, tree):
    for rel, text in tree.items():
        _w(tmp_path, rel, text)
    cfg = _cfg(backup=False)
    idx = index_mod.build([str(tmp_path)], cfg, root=tmp_path,
                          summarise=False, verbose=False)
    plan = scaffold.Plan(
        project_name="p", summary="s",
        files=[scaffold.PlannedFile(path=r) for r in tree])
    return cfg, idx, plan


class _Seq:
    """chat pops scripted replies in order; records the calls and messages."""
    stream_truncated = False

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.msgs: list = []

    def chat(self, messages, max_tokens=0, **kw):
        self.calls += 1
        self.msgs.append([dict(m) for m in messages])
        return self.replies.pop(0) if self.replies else "ok"


def test_contract_fix_counts_files_not_block_rows(tmp_path):
    cfg, idx, pl = _web_plan(tmp_path, {
        "index.html": '<!doctype html><body><div id="stage"></div>'
                      '<script type="module" src="main.js"></script></body>',
        "main.js": "document.getElementById('board');\n",
    })
    # Two DISTINCT blocks land in the same file: the count is one file.
    reply = (_edit("main.js", "document.getElementById('board');",
                   "document.getElementById('stage');")
             + _edit("main.js", "document.getElementById('stage');",
                     "document.getElementById('stage'); // wired"))
    client = _Seq([reply])
    fixed, errors = scaffold.fix_web_contracts(
        client, cfg, idx, pl, None, verbose=False, rounds=3)
    assert fixed == 1
    assert not errors


def test_contract_fix_stops_when_no_error_resolves(tmp_path):
    cfg, idx, pl = _web_plan(tmp_path, {
        "index.html": '<!doctype html><body><div id="stage"></div>'
                      '<script type="module" src="main.js"></script></body>',
        "main.js": "document.getElementById('board');\n",
    })
    # A real block that lands but resolves no contract error: one pass only.
    reply = _edit("main.js", "document.getElementById('board');",
                  "document.getElementById('board'); // note")
    client = _Seq([reply, reply, reply])
    fixed, errors = scaffold.fix_web_contracts(
        client, cfg, idx, pl, None, verbose=False, rounds=3)
    assert client.calls == 1            # no progress -> stop, do not spin
    assert errors                       # the violation still stands


def test_contract_fix_reverts_an_export_deleting_fix(tmp_path):
    cfg, idx, pl = _web_plan(tmp_path, {
        "index.html": '<!doctype html><body><div id="stage"></div>'
                      '<script type="module" src="main.js"></script></body>',
        "main.js": "import { helper } from './util.js';\n"
                   "document.getElementById('board');\nhelper();\n",
        "util.js": "export function helper() { return 1; }\n",
    })
    util_before = (tmp_path / "util.js").read_text(encoding="utf-8")
    # The "fix" replaces util.js wholesale and drops the `helper` export the
    # importer still needs: a NEW contract error naming util.js.
    reply = ("--- FILE: util.js\n<<<<<<< SEARCH\n=======\n"
             "export function other() { return 2; }\n>>>>>>> REPLACE\n")
    client = _Seq([reply])
    fixed, errors = scaffold.fix_web_contracts(
        client, cfg, idx, pl, None, verbose=False, rounds=2)
    assert fixed == 0                                   # reverted, not counted
    assert (tmp_path / "util.js").read_text(encoding="utf-8") == util_before  # bytes restored


# --------------------------------------------------------------------------
# BUG-203 / BUG-223: a bare number is not distinctive evidence; it tied every
# hunk to every clause that mentioned the same digit.
# --------------------------------------------------------------------------

def test_bare_numbers_are_not_specific_terms():
    assert verify._specific_terms(["16", "4", "pattern", "TRACK_IDS"]) == \
        ["pattern", "TRACK_IDS"]


def test_a_number_does_not_tie_a_hunk_to_a_clause():
    task = ("1. In src/pattern.js add generateEuclidean with 16 steps.\n"
            "2. In src/theory.js correct the offset table.")
    tracker = verify.RequirementTracker(task)
    tracker.observe_edit([
        Result(True, "src/theory.js", "applied",
               after="export const offset = 16;\n"),
    ])
    assert "R1" not in tracker.clause_map().get("src/theory.js", [])


# --------------------------------------------------------------------------
# BUG-204 / BUG-212 / BUG-205: the five/six-slot cap truncated a multi-clause
# brief, and a list introduction ("...contracts:") ate a concrete slot.
# --------------------------------------------------------------------------

EIGHT_CLAUSES = "\n".join(
    f"{i}. In src/f{i}.js add the widget{i} handler." for i in range(1, 9))


def test_requirement_ledger_holds_eight_clauses():
    reqs = verify.requirement_ledger(EIGHT_CLAUSES)
    assert len(reqs) == 8                       # the old cap of five cut it


def test_list_introduction_is_not_a_requirement():
    task = ("Required final public API and behavior:\n"
            "1. In src/a.js export alpha.\n"
            "2. In src/b.js export beta.\n"
            "3. In src/c.js export gamma.")
    reqs = verify.requirement_ledger(task)
    assert len(reqs) == 3
    assert not any(t.rstrip().endswith(":") for t, _ in reqs)
    assert reqs[0][0].startswith("In src/a.js")


def test_plan_titles_match_the_ledger_cap_and_drop_the_intro():
    task = ("Repair the following contracts:\n" + EIGHT_CLAUSES)
    titles = plan.requirement_titles(task)
    assert len(titles) == 8                     # the old cap of six cut it
    assert not any(t.rstrip().endswith(":") for t in titles)
    assert titles[0].startswith("In src/f1.js")


# --------------------------------------------------------------------------
# BUG-206: a recorded baseline describes the OLD contents of a test file this
# run rewrote; its failures cannot be assumed pre-existing.
# --------------------------------------------------------------------------

def test_rewritten_test_file_baseline_is_not_pre_existing(monkeypatch):
    idx = types.SimpleNamespace(root=Path("/tmp"))
    monkeypatch.setattr(testsel, "detect_test_command", lambda *a: "npm test")
    monkeypatch.setattr(testsel, "impacted_tests",
                        lambda *a: [("tests/a.test.js", "imports")])
    monkeypatch.setattr(testsel, "narrow_command",
                        lambda *a, **k: ("npm test tests/a.test.js", True))
    monkeypatch.setattr(testsel, "_was_already_failing", lambda *a: True)
    monkeypatch.setattr(testsel, "run_command", lambda *a: (False, "fail"))
    # The changed file IS the narrowed test: verdict stays edit-caused.
    _ok, _out, desc = testsel.run_impacted(idx, _cfg(), ["tests/a.test.js"])
    assert testsel.PRE_EXISTING_LABEL not in desc
    # An unrelated change keeps the recorded pre-existing verdict.
    _ok, _out, desc2 = testsel.run_impacted(idx, _cfg(), ["src/other.js"])
    assert testsel.PRE_EXISTING_LABEL in desc2


# --------------------------------------------------------------------------
# BUG-222: a deleted test file still rides in `changed` and the stale index;
# narrowing to it runs zero tests and exits 0 (a false green).
# --------------------------------------------------------------------------

def test_deleted_test_file_is_not_selected(tmp_path):
    _w(tmp_path, "src/a.js", "export function a() { return 1; }\n")
    _w(tmp_path, "tests/a.test.js",
       "import { a } from '../src/a.js';\ntest('a', () => a());\n")
    cfg = _cfg()
    idx = index_mod.build([str(tmp_path)], cfg, root=tmp_path,
                          summarise=False, verbose=False)
    (tmp_path / "tests/a.test.js").unlink()     # deleted, index now stale
    picked = testsel.impacted_tests(idx, ["src/a.js", "tests/a.test.js"])
    assert "tests/a.test.js" not in [t for t, _ in picked]
    assert all((tmp_path / t).is_file() for t, _ in picked)


# --------------------------------------------------------------------------
# BUG-207: a gather-phase shell mutation already made the edit; the model's
# no-op block confirms it. Honor that instead of failing the target.
# --------------------------------------------------------------------------

def test_shell_satisfied_needs_mutation_and_noop_confirmation(tmp_path):
    agent, cfg, idx = _files(tmp_path, {"src/a.js": "x();\n"})
    noop = _edit("src/a.js", "x();", "x();")
    real = _edit("src/a.js", "x();", "y();")
    cfg._task_mutations = ["src/a.js"]
    assert agent._shell_satisfied("src/a.js", noop) is True
    assert agent._shell_satisfied("src/a.js", real) is False   # not a confirm
    cfg._task_mutations = []
    assert agent._shell_satisfied("src/a.js", noop) is False   # no mutation
    cfg._task_mutations = ["src/other.js"]
    assert agent._shell_satisfied("src/a.js", noop) is False   # other file


# --------------------------------------------------------------------------
# BUG-208: a later hunk must not clobber a definition an earlier whole-file
# hunk appended to the same file.
# --------------------------------------------------------------------------

def test_later_hunk_cannot_drop_an_earlier_append(tmp_path):
    agent, cfg, idx = _files(tmp_path, {
        "a.js": "export function keep() { return 1; }\n"})
    # Intent 1 (whole-file) appends `helper`.
    full = ("--- FILE: a.js\n<<<<<<< SEARCH\n=======\n"
            "export function keep() { return 1; }\n"
            "export function helper() { return 2; }\n>>>>>>> REPLACE\n")
    cur = ("export function keep() { return 1; }\n"
           "export function helper() { return 2; }")
    # Intent 2 (symbol keep): a SEARCH spanning keep+helper whose REPLACE drops
    # helper -- a valid symbol hunk that nonetheless clobbers the append.
    clobber = _edit("a.js", cur, "export function keep() { return 9; }")
    kept = _edit("a.js", cur,
                 "export function keep() { return 9; }\n"
                 "export function helper() { return 2; }")
    client = _Seq([full, clobber, kept])
    agent.client = client
    intents = [
        {"file": "a.js", "symbol": "", "intended_change": "add helper"},
        {"file": "a.js", "symbol": "keep", "intended_change": "tweak keep"},
    ]
    results, receipts, findings = agent.apply_edit_intents(
        "Add helper to a.js and tweak keep.", "", intents, verbose=False)
    assert client.calls == 3                    # the clobber was re-asked
    assert all(r.ok for r in results)
    final = (tmp_path / "a.js").read_text(encoding="utf-8")
    assert "export function helper()" in final  # the append survived
    assert "return 9" in final                  # the tweak landed too
    # The re-ask named the lost definition so the model could keep it.
    assert any("helper" in str(m) for m in client.msgs[-1])


# --------------------------------------------------------------------------
# BUG-209: a `read` with neither file nor symbol is rejected before dispatch.
# --------------------------------------------------------------------------

def test_read_without_file_or_symbol_is_rejected(tmp_path):
    client = Scripted([{"action": "read"}, {"action": "done"}])
    agent, cfg, idx = _files(tmp_path, {"a.py": "X = 1\n"})
    agent.client = client
    agent.gather("Inspect a.py.", verbose=False)
    assert any("rejected: no file or symbol" in s for s in agent.steps_log)


# --------------------------------------------------------------------------
# BUG-210: a spec that forbids extra files gets no starter README/.gitignore/
# packaging.
# --------------------------------------------------------------------------

def test_spec_forbids_extras_detection():
    assert scaffold._spec_forbids_extras(
        "Create exactly these 10 files. Do not add README or extra files.")
    assert scaffold._spec_forbids_extras("no README or additional files")
    assert not scaffold._spec_forbids_extras("Build a small web app.")


def test_no_extras_plan_writes_no_starter_files(tmp_path):
    pl = scaffold.Plan(project_name="p", summary="s",
                       files=[scaffold.PlannedFile(path="main.py")],
                       no_extras=True)
    scaffold.apply_skeleton(tmp_path, pl, verbose=False)
    assert (tmp_path / "main.py").is_file()
    assert not (tmp_path / "README.md").exists()
    assert not (tmp_path / ".gitignore").exists()
    assert not (tmp_path / "pyproject.toml").exists()


def test_default_plan_still_writes_starter_files(tmp_path):
    pl = scaffold.Plan(project_name="p", summary="s",
                       files=[scaffold.PlannedFile(path="main.py")])
    scaffold.apply_skeleton(tmp_path, pl, verbose=False)
    assert (tmp_path / "README.md").is_file()
    assert (tmp_path / ".gitignore").is_file()


# --------------------------------------------------------------------------
# BUG-211 / BUG-216: a task that names its test/DOM command owes that check.
# --------------------------------------------------------------------------

def test_task_naming_tests_and_domcheck_enables_both(tmp_path):
    cfg = _cfg(verify_tests=False, verify_dom=False)
    loop._infer_task_checks(tmp_path, cfg, "Fix the bug. Run npm test and "
                                           "domcheck.")
    assert cfg.verify_tests is True
    assert cfg.verify_dom is True


def test_task_naming_only_pytest_enables_tests(tmp_path):
    cfg = _cfg(verify_tests=False, verify_dom=False)
    loop._infer_task_checks(tmp_path, cfg, "Repair the parser; run pytest.")
    assert cfg.verify_tests is True
    assert cfg.verify_dom is False


def test_task_naming_no_check_leaves_flags_off(tmp_path):
    cfg = _cfg(verify_tests=False, verify_dom=False)
    loop._infer_task_checks(tmp_path, cfg, "Refactor the helper module.")
    assert cfg.verify_tests is False
    assert cfg.verify_dom is False


# --------------------------------------------------------------------------
# BUG-213: the ordinary answer path runs the same coverage verdict + bounded
# follow-up the incremental microloop runs.
# --------------------------------------------------------------------------

def test_coverage_pass_flags_unaddressed_clause(tmp_path):
    agent, cfg, idx = _files(tmp_path, {"a.py": "X = 1\n", "b.py": "Y = 2\n"})
    task = "1. In a.py add alpha.\n2. In b.py add beta."
    agent.requirements = verify.RequirementTracker(task)
    r = Result(True, "a.py", "applied", after="X = 1\nalpha = 1\n")
    findings: list = []
    _res, _reply, findings = agent._coverage_pass(
        task, "", [r], "applied a.py", findings, verbose=False)
    assert any(f.kind == "coverage" for f in findings)


def test_ordinary_answer_path_emits_coverage_followup(tmp_path):
    client = Scripted([{"action": "done"}],
                      [_edit("a.py", "X = 1\n", "X = 1\nalpha = 1\n"), "ok"])
    agent, cfg, idx = _files(tmp_path, {"a.py": "X = 1\n", "b.py": "Y = 2\n"},
                           incremental_edit_loop=False, use_memory=False,
                           backup=False)
    agent.client = client
    emitted: list = []
    orig = agent._emit

    def spy(kind, **kw):
        emitted.append(kind)
        return orig(kind, **kw)

    agent._emit = spy
    task = "1. In a.py add alpha.\n2. In b.py add beta."
    loop._run_task_once(tmp_path, cfg, client, agent, idx, task, apply=True)
    assert "coverage_followup" in emitted


# --------------------------------------------------------------------------
# BUG-214: a whole-file replacement that drops an export another file still
# imports breaks the module graph.
# --------------------------------------------------------------------------

def test_lost_export_gate_flags_a_dropped_imported_export(tmp_path):
    agent, cfg, idx = _files(tmp_path, {
        "src/pattern.js": "export const TRACK_IDS = [1];\n"
                          "export function generateEuclidean() { return 1; }\n",
        "src/main.js": "import { TRACK_IDS } from './pattern.js';\n"
                       "console.log(TRACK_IDS);\n",
    })
    before = (tmp_path / "src/pattern.js").read_text(encoding="utf-8")
    after = "export function generateEuclidean() { return 1; }\n"
    r = Result(True, "src/pattern.js", "replaced contents (2 -> 1 line(s))",
               before=before, after=after, whole_file=True)
    found = verify.check_lost_exports(agent, [r])
    assert any(f.kind == "lost_exports" and f.severity == "error"
               and "TRACK_IDS" in f.message for f in found)


def test_lost_export_gate_quiet_when_no_importer_needs_it(tmp_path):
    agent, cfg, idx = _files(tmp_path, {
        "src/pattern.js": "export const TRACK_IDS = [1];\n",
        "src/main.js": "console.log('hi');\n",
    })
    before = (tmp_path / "src/pattern.js").read_text(encoding="utf-8")
    r = Result(True, "src/pattern.js", "replaced contents (1 -> 1 line(s))",
               before=before, after="export const OTHER = [2];\n",
               whole_file=True)
    assert verify.check_lost_exports(agent, [r]) == []


# --------------------------------------------------------------------------
# BUG-215: the same completed edit block re-emitted under changing FILE
# headers is a degenerate loop the verbatim tail test cannot see.
# --------------------------------------------------------------------------

class _Resp:
    def __init__(self, lines):
        self.status_code = 200
        self.text = ""
        self.headers = {}
        self._lines = lines

    def iter_lines(self, decode_unicode=False):
        yield from self._lines

    def close(self):
        pass


def _sse(text="", finish=None):
    return "data: " + json.dumps(
        {"choices": [{"delta": {"content": text} if text else {},
                      "finish_reason": finish}]})


def _llm_cfg():
    cfg = Config()
    cfg.base_url = "http://127.0.0.1:1/v1"
    cfg.model = "m"
    return cfg


_DUP_BODY = ("function generateEuclidean(pulses, steps) {\n"
             "  const pattern = [];\n"
             "  for (let i = 0; i < steps; i++) {\n"
             "    pattern.push(Math.floor((i * pulses) / steps));\n"
             "  }\n  return pattern;\n}\n")


def test_duplicate_block_loop_stops_the_stream(monkeypatch, capsys):
    lines = []
    for i in range(6):
        block = (f"--- FILE: src/pattern.js L{10 + i}-{20 + i}\n"
                 "<<<<<<< SEARCH\n=======\n" + _DUP_BODY + ">>>>>>> REPLACE\n")
        if i == 2:                       # closing marker split across chunks
            lines.append(_sse(block[:-8]))
            lines.append(_sse(block[-8:]))
        else:
            lines.append(_sse(block))
    lines.append(_sse(finish="stop"))
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: _Resp(lines))
    client = LLMClient(_llm_cfg())
    text = client.chat([{"role": "user", "content": "edit"}], 8000, stream=True)
    assert client.stream_repetition is True
    assert client.stream_truncated is True
    assert text.count(">>>>>>> REPLACE") <= 3
    assert "re-emitting" in capsys.readouterr().err


def test_short_repeated_blocks_are_exempt(monkeypatch):
    lines = []
    for i in range(4):
        lines.append(_sse(
            f"Step {i}: add the shared import.\n--- FILE: src/f{i}.js\n"
            "<<<<<<< SEARCH\n=======\nimport {x} from \"./y\";\n"
            ">>>>>>> REPLACE\n"))
    lines.append(_sse(finish="stop"))
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: _Resp(lines))
    client = LLMClient(_llm_cfg())
    text = client.chat([{"role": "user", "content": "edit"}], 8000, stream=True)
    assert client.stream_repetition is False
    assert client.stream_truncated is False
    assert "Step 3" in text


def test_distinct_large_blocks_do_not_trip(monkeypatch):
    lines = []
    for i in range(3):
        body = _DUP_BODY.replace("generateEuclidean", f"variant{i}")
        lines.append(_sse(f"--- FILE: src/p{i}.js\n<<<<<<< SEARCH\n=======\n"
                          + body + ">>>>>>> REPLACE\n"))
    lines.append(_sse(finish="stop"))
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: _Resp(lines))
    client = LLMClient(_llm_cfg())
    text = client.chat([{"role": "user", "content": "edit"}], 8000, stream=True)
    assert client.stream_repetition is False
    assert text.count(">>>>>>> REPLACE") == 3


# --------------------------------------------------------------------------
# BUG-217: Int8Array / Int16Array / Int32Array are built-ins, not undefined
# project names.
# --------------------------------------------------------------------------

def test_typed_arrays_are_known_globals():
    for name in ("Int8Array", "Int16Array", "Int32Array"):
        assert name in webcheck.JS_GLOBAL_NAMES


# --------------------------------------------------------------------------
# BUG-218: the post-write frontend gate says what is true -- the edits landed
# and violations were found; it does not claim the write was rejected.
# --------------------------------------------------------------------------

class _NoRepair:
    def _repair_from_findings(self, *a, **k):
        return ""


def test_frontend_gate_wording_is_honest_about_the_write(tmp_path):
    _w(tmp_path, "index.html",
       '<!doctype html><html><body><div id="stage"></div>'
       '<script type="module" src="main.js"></script></body></html>')
    _w(tmp_path, "main.js", "document.getElementById('board');\n")
    cfg = _cfg(frontend_contracts=True, static_gate=False,
               interface_lock=False, repair_rounds=0, log_events=True)
    idx = types.SimpleNamespace(root=tmp_path)
    loop._post_edit_tests(
        tmp_path, cfg, _NoRepair(), idx, "wire the page",
        [Result(True, "main.js", "applied")], "reply", None, None,
        ["main.js"], collect=[])
    log = (tmp_path / ".pedacito" / "events.jsonl")
    text = log.read_text(encoding="utf-8") if log.exists() else ""
    if text:
        assert "found violations in the applied edits" in text
        assert "rejected the patch" not in text


# --------------------------------------------------------------------------
# BUG-219: rename --strings moves whole-word occurrences inside string
# literals (test labels, docs); comments and code stay untouched.
# --------------------------------------------------------------------------

def test_rename_default_leaves_strings_and_reports_them(tmp_path):
    _w(tmp_path, "src/t.py",
       'def toggleStep(x):\n    return x\nLABEL = "toggleStep adds step"\n')
    res = refactor.rename_symbol_across_project(
        tmp_path, "toggleStep", "toggleStepAt", dry_run=True)
    assert res["ok"]
    assert res["string_replacements"] == 0
    assert res["string_matches"]                 # the leftover note still fires


def test_rename_strings_moves_labels_not_comments(tmp_path):
    _w(tmp_path, "src/t.py",
       'def toggleStep(x):\n    return x\n'
       'LABEL = "toggleStep adds step"\n'
       'OTHER = "toggleStepify is unrelated"\n'
       '# toggleStep comment stays\n')
    _w(tmp_path, "tests/t.test.js",
       'import { toggleStep } from "../src/t.js";\n'
       'test("toggleStep adds step", () => { toggleStep(1); });\n'
       '// toggleStep comment stays\n')
    _w(tmp_path, "src/t.js",
       'export function toggleStep(x) { return x; }\n')
    res = refactor.rename_symbol_across_project(
        tmp_path, "toggleStep", "toggleStepAt", dry_run=False,
        include_strings=True)
    assert res["ok"]
    assert res["string_replacements"] >= 2
    assert not res["string_matches"]
    py = (tmp_path / "src/t.py").read_text(encoding="utf-8")
    js = (tmp_path / "tests/t.test.js").read_text(encoding="utf-8")
    assert 'LABEL = "toggleStepAt adds step"' in py
    assert '"toggleStepify is unrelated"' in py      # whole-word only
    assert "# toggleStep comment stays" in py        # comments untouched
    assert 'test("toggleStepAt adds step"' in js
    assert "// toggleStep comment stays" in js


def test_rename_strings_handles_fstring(tmp_path):
    _w(tmp_path, "m.py",
       'def toggleStep(x):\n    return x\n'
       'msg = f"call {toggleStep} now: toggleStep label"\n')
    res = refactor.rename_symbol_across_project(
        tmp_path, "toggleStep", "toggleStepAt", dry_run=False,
        include_strings=True)
    assert res["ok"]
    out = (tmp_path / "m.py").read_text(encoding="utf-8")
    assert 'f"call {toggleStepAt} now: toggleStepAt label"' in out


# --------------------------------------------------------------------------
# BUG-220: a purely pre-existing red suite is not a task failure under
# --strict.
# --------------------------------------------------------------------------

def _strict_run(tmp_path, monkeypatch, *, preexisting):
    _w(tmp_path, "a.py", "GOOD = 1\n")
    plan_path = tmp_path / "PLAN.md"
    loop.write_plan(plan_path, "goal", ["task A"])
    idx = types.SimpleNamespace(root=tmp_path)

    def fake_run_task(root, cfg_, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None, is_repair=False):
        return loop.TaskResult(
            [Result(True, "a.py", "applied")], "reply", True, ["a.py"], idx_,
            tests_ok=False, tests_preexisting_only=preexisting)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    cfg = _cfg(static_gate=False, frontend_contracts=False,
               interface_lock=False, use_memory=False, backup=False)
    return loop.run_loop(tmp_path, cfg, None, idx, plan_path, apply=True,
                         strict=True)


def test_strict_ignores_purely_pre_existing_red(tmp_path, monkeypatch):
    assert _strict_run(tmp_path, monkeypatch, preexisting=True) == 0


def test_strict_still_fails_a_new_red(tmp_path, monkeypatch):
    assert _strict_run(tmp_path, monkeypatch, preexisting=False) == 1


# --------------------------------------------------------------------------
# BUG-221: node failure identities -- the spec summary header is not a test,
# a file-level TAP wrapper stands for its subtests, and a YAML block scalar
# error carries its message on the following lines.
# --------------------------------------------------------------------------

def test_node_spec_summary_header_is_not_a_failure():
    text = ("✖ failing tests:\n"
            "✖ adds step (1ms)\n"
            "  error: boom\n")
    ids = loop._node_failure_identities(text)
    assert "node::adds step" in ids
    assert not any("failing tests" in k for k in ids)


def test_node_file_wrapper_expands_to_subtests():
    text = ("not ok 1 - tests/foo.test.js\n"
            "  not ok 1 - adds step\n"
            "    error: 'expected 1'\n"
            "  not ok 2 - removes step\n"
            "    error: 'expected 2'\n")
    ids = loop._node_failure_identities(text)
    assert "node::adds step" in ids
    assert "node::removes step" in ids
    assert "node::tests/foo.test.js" not in ids


def test_node_yaml_block_scalar_error_message():
    text = ("not ok 1 - adds step\n"
            "  error: |-\n"
            "    expected 1\n"
            "    got 2\n"
            "  ...\n")
    ids = loop._node_failure_identities(text)
    sig = ids["node::adds step"].signature
    assert "expected 1" in sig
    assert "|-" not in sig