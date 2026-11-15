"""Consolidated Follow-Up & Hardening Sweep (Bug Hunts #7-#13).

One section per numbered item of the sweep, plus regression-only pins for the
Hunt #7 findings that Hunts #8-#12 had already mitigated (BUG-206/207/208/215/
217/222). Every test is hermetic: scripted clients, monkeypatched card storage
and subprocess boundaries -- no network, no real model server, no browser.

Items already covered in depth by the per-hunt validation files are pinned
here at the specific gap THIS sweep filled, so the whole sweep can be re-run
as one unit.
"""
from __future__ import annotations

import json
import os
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import (  # noqa: E402
    agent as agent_mod, browser, cli, config as config_mod, edits, fileio,
    how_index, index as index_mod, llm, loop, modelcard, plan, processes,
    proc, providers, refactor, roles, scaffold, spec, testsel, verify,
    webcheck)
from pedacito.agent import Agent  # noqa: E402
from pedacito.config import Config  # noqa: E402
from pedacito.edits import Result  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Shared helpers (same shape as test_buglog7_validation.py)
# ---------------------------------------------------------------------------

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
        self.json_calls: list[list[dict]] = []

    def chat_json(self, messages, schema, max_tokens=0, **kw):
        self.json_calls.append([dict(m) for m in messages])
        step = self.steps.pop(0) if self.steps else {"action": "done"}
        return {"reasoning": "r", **step}

    def chat(self, messages, max_tokens=0, stream=False, **kw):
        self.chat_calls.append([dict(m) for m in messages])
        return self.replies.pop(0) if self.replies else "ok"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolated model-card storage (never the real user's cards)."""
    monkeypatch.setattr(modelcard, "_path",
                        lambda: tmp_path / "home" / "model_cards.json")
    (tmp_path / "home").mkdir()
    return tmp_path / "home"


def _save_card(model, url="http://127.0.0.1:11434/v1", measured=None):
    card = {"version": modelcard.CARD_VERSION,
            "key": f"{url.rstrip('/')}|{model}", "ts": time.time(),
            "model": model, "url": url,
            "measured": dict(measured or {}),
            "choices": {}, "notes": [], "repeats": 3, "seconds": 1.0}
    modelcard.save_card(card)
    return card


# ===========================================================================
# Hunt #7 regressions (already mitigated in Hunts #8-#12; pinned here)
# ===========================================================================

# --- BUG-217: TypedArrays are built-ins, not undefined project names --------

def test_bug217_typed_arrays_are_known_globals():
    for name in ("Int8Array", "Int16Array", "Int32Array", "Uint8Array",
                 "Uint16Array", "Uint32Array", "Float32Array", "Float64Array",
                 "BigInt64Array", "BigUint64Array", "Uint8ClampedArray"):
        assert name in webcheck.JS_GLOBAL_NAMES, name


def test_bug217_typed_array_use_is_not_an_unbound_name(tmp_path):
    _w(tmp_path, "main.js",
       "const samples = new Float32Array(8);\n"
       "const bytes = new Uint8Array(samples.buffer);\n"
       "console.log(bytes.length);\n")
    findings = webcheck.run(tmp_path)
    assert not [f for f in findings
                if "Float32Array" in f.message or "Uint8Array" in f.message]


# --- BUG-222: impacted_tests selects only files that exist on disk ----------

def test_bug222_deleted_test_file_is_not_selected(tmp_path):
    _w(tmp_path, "src/app.py", "def go():\n    return 1\n")
    _w(tmp_path, "tests/test_app.py", "from src.app import go\n"
                                      "def test_go():\n    assert go() == 1\n")
    cfg = _cfg()
    idx = index_mod.build([str(tmp_path)], cfg, root=tmp_path,
                          summarise=False, verbose=False)
    picked = testsel.impacted_tests(idx, ["src/app.py"])
    assert any(t == "tests/test_app.py" for t, _why in picked)
    # Delete the test file but keep it in the (now stale) index and `changed`:
    # narrowing to it would run zero tests and exit 0 -- a false green.
    (tmp_path / "tests/test_app.py").unlink()
    picked = testsel.impacted_tests(idx, ["src/app.py", "tests/test_app.py"])
    assert all(t != "tests/test_app.py" for t, _why in picked)


# --- BUG-206: a rewritten test file invalidates the recorded baseline -------

def _patch_runner(monkeypatch, *, cmd="pytest -q", picked=None,
                  was_failing=True, ok=False, out="1 failed"):
    monkeypatch.setattr(testsel, "detect_test_command",
                        lambda root, cfg=None: cmd)
    monkeypatch.setattr(testsel, "impacted_tests",
                        lambda index, changed, limit=12:
                        list(picked if picked is not None
                             else [("tests/test_x.py", "changed itself")]))
    monkeypatch.setattr(testsel, "_language_base",
                        lambda root, changed, cfg, base: base)
    monkeypatch.setattr(testsel, "narrow_command",
                        lambda base, tests, changed=None: (base, True))
    monkeypatch.setattr(testsel, "_was_already_failing",
                        lambda root, c, before=None: was_failing)
    monkeypatch.setattr(testsel, "run_command",
                        lambda root, cfg, c, timeout=None: (ok, out))


def test_bug206_rewritten_test_loses_the_preexisting_label(tmp_path,
                                                          monkeypatch):
    _patch_runner(monkeypatch)
    idx = SimpleNamespace(root=tmp_path)
    ok, _out, desc = testsel.run_impacted(idx, _cfg(), ["tests/test_x.py"])
    assert ok is False
    assert testsel.PRE_EXISTING_LABEL not in desc


def test_bug206_untouched_test_keeps_the_preexisting_label(tmp_path,
                                                          monkeypatch):
    _patch_runner(monkeypatch)
    idx = SimpleNamespace(root=tmp_path)
    ok, _out, desc = testsel.run_impacted(idx, _cfg(), ["src/x.py"])
    assert ok is False
    assert testsel.PRE_EXISTING_LABEL in desc


# --- BUG-215: the stream block census counts blocks, not FILE headers -------

def test_bug215_file_headers_are_not_block_completions():
    body = ("function f() {\n  const a = 1;\n  const b = 2;\n"
            "  return a + b;\n}\n")
    text = ""
    for i in range(3):
        text += (f"--- FILE: src/f{i}.js\n<<<<<<< SEARCH\n=======\n"
                 + body + ">>>>>>> REPLACE\n")
    matches = list(llm._STREAM_BLOCK_DONE_RE.finditer(text))
    assert len(matches) == 3           # one per block, none per FILE header
    assert all("--- FILE:" not in m.group(0) for m in matches)


def test_bug215_symbol_blocks_still_count():
    text = ("--- FILE: a.py\n<<<<<<< SYMBOL def go\n"
            "def go():\n    return 1\n>>>>>>> END_SYMBOL\n")
    assert len(list(llm._STREAM_BLOCK_DONE_RE.finditer(text))) == 1


# --- BUG-207: a gather shell mutation + no-op confirmation is satisfied -----

def test_bug207_shell_mutation_confirmed_by_noop_blocks(tmp_path):
    agent, cfg, idx = _files(tmp_path, {"a.py": "X = 1\n"})
    cfg._task_mutations = ["a.py"]
    noop = _edit("a.py", "X = 1\n", "X = 1\n")
    assert agent._shell_satisfied("a.py", noop) is True
    # A real block is not a confirmation: the normal apply path owns it.
    real = _edit("a.py", "X = 1\n", "X = 2\n")
    assert agent._shell_satisfied("a.py", real) is False
    # No mutation ledger entry: nothing to honor.
    cfg._task_mutations = []
    assert agent._shell_satisfied("a.py", noop) is False


def test_bug207_ladder_resets_the_mutation_ledger_per_task(tmp_path):
    cfg = _cfg()
    cfg._task_mutations = ["stale.py"]
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(loop, "_run_task_once",
                       lambda *a, **k: loop.TaskResult([], "noop", False,
                                                       [], None))
        loop._run_ladder(tmp_path, cfg, "client", object(), None, "t")
        assert cfg._task_mutations == []
    finally:
        monkey.undo()


# --- BUG-208: appended names survive later hunks (ledger + name census) -----

def test_bug208_top_level_names_census(tmp_path):
    agent, cfg, idx = _files(tmp_path, {
        "m.py": "def alpha():\n    return 1\n\n"
                "class Box:\n    def beta(self):\n        return 2\n",
    })
    names = agent._top_level_names("m.py")
    # The census is of TOP-LEVEL units: a method lives inside its class's
    # span, so the append ledger only owes names a later hunk can drop whole.
    assert names == {"alpha", "Box"}


def test_bug208_hunk_clobbering_an_append_is_rejected(tmp_path):
    # The incremental microloop records the names a whole-file hunk appends;
    # a later hunk that drops them is refused and the pre-hunk bytes restored.
    # The second target must be a DIFFERENT symbol: identical file+symbol
    # targets dedupe as "already covered" before any hunk is asked for.
    agent, cfg, idx = _files(tmp_path, {
        "m.py": "def alpha():\n    return 1\n",
    }, incremental_edit_loop=True, use_memory=False, backup=False,
        repair_rounds=2)
    task = "1. In m.py add beta.\n2. In m.py tweak alpha."
    appended_reply = _edit("m.py", "", "def alpha():\n    return 1\n\n"
                                       "def beta():\n    return 2\n")
    # A whole-file re-quote that tweaks alpha but drops the appended beta.
    clobber_reply = _edit("m.py",
                          "def alpha():\n    return 1\n\n"
                          "def beta():\n    return 2\n",
                          "def alpha():\n    return 99\n")
    # Every retry re-emits the clobber: the batch reports the LAST attempt's
    # rejection, so the "appended" verdict must survive all three.
    client = Scripted([{"action": "done"}],
                      [appended_reply] + [clobber_reply] * 3)
    agent.client = client
    intents = [{"file": "m.py", "symbol": ""},
               {"file": "m.py", "symbol": "alpha"}]
    results, _reply, _findings = agent.apply_edit_intents(
        task, "", intents, verbose=False)
    final = (tmp_path / "m.py").read_text(encoding="utf-8")
    assert "def beta():" in final          # the append survived
    assert any(not r.ok and "appended" in r.message for r in results)


# ===========================================================================
# Item 1: the scaffold repair count is of RESOLVED violations, not block rows
# ===========================================================================

def _web_plan(tmp_path, tree):
    for rel, text in tree.items():
        _w(tmp_path, rel, text)
    cfg = _cfg(backup=False)
    idx = index_mod.build([str(tmp_path)], cfg, root=tmp_path,
                          summarise=False, verbose=False)
    pl = scaffold.Plan(project_name="p", summary="s",
                       files=[scaffold.PlannedFile(path=r) for r in tree])
    return cfg, idx, pl


class _Seq:
    stream_truncated = False

    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages, max_tokens=0, **kw):
        return self.replies.pop(0) if self.replies else "ok"


def test_item1_count_is_resolved_violations_not_blocks(tmp_path):
    cfg, idx, pl = _web_plan(tmp_path, {
        "index.html": '<!doctype html><body><div id="stage"></div>'
                      '<script type="module" src="main.js"></script></body>',
        "main.js": "document.getElementById('board');\n",
    })
    # Two blocks land in one file and clear the ONE standing violation.
    reply = (_edit("main.js", "document.getElementById('board');",
                   "document.getElementById('stage');")
             + _edit("main.js", "document.getElementById('stage');",
                     "document.getElementById('stage'); // wired"))
    fixed, errors = scaffold.fix_web_contracts(
        _Seq([reply]), cfg, idx, pl, None, verbose=False, rounds=3)
    assert fixed == 1
    assert not errors


def test_item1_landed_block_that_resolves_nothing_counts_zero(tmp_path):
    cfg, idx, pl = _web_plan(tmp_path, {
        "index.html": '<!doctype html><body><div id="stage"></div>'
                      '<script type="module" src="main.js"></script></body>',
        "main.js": "document.getElementById('board');\n",
    })
    # A real edit lands, but the violation stands: it is not a fix.
    reply = _edit("main.js", "document.getElementById('board');",
                  "document.getElementById('board'); // touched")
    fixed, errors = scaffold.fix_web_contracts(
        _Seq([reply]), cfg, idx, pl, None, verbose=False, rounds=1)
    assert fixed == 0
    assert errors


# ===========================================================================
# Items 2 / 14: requirement caps raised to 16; venv PATH for child processes
# ===========================================================================

def test_item2_sixteen_requirements_survive_the_ledger():
    task = "\n".join(f"{i}. In file{i}.py add feature{i}."
                     for i in range(1, 15))
    ledger = verify.requirement_ledger(task)
    assert len(ledger) == 14                 # the old 12-cap truncated this
    titles = plan.requirement_titles(task)
    assert len(titles) == 14


def test_item2_abbreviation_is_not_a_clause_boundary():
    text = ("Add support for e.g. sharps and flats. "
            "Return null for bad input.")
    clauses = verify._split_clauses(text)
    assert "e.g. sharps" in clauses[0][0]
    assert len(clauses) == 2


def test_item2_generic_terms_do_not_prove_provenance():
    for word in ("update", "error", "fix", "change", "ensure", "handle",
                 "support", "using", "work"):
        assert word in verify._GENERIC_TERMS, word


def test_item14_venv_bin_dirs_order(tmp_path):
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / "venv" / "bin").mkdir(parents=True)
    dirs = proc.venv_bin_dirs(tmp_path)
    assert dirs[0] == str(tmp_path / ".venv" / "bin")
    assert dirs[1] == str(tmp_path / "venv" / "bin")
    assert str(Path(sys.executable).parent) in dirs


def test_item14_with_venv_path_prepends_and_copies(tmp_path):
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    env = {"PATH": "/usr/bin", "OTHER": "keep"}
    out = proc.with_venv_path(env, tmp_path)
    assert out["PATH"].split(os.pathsep)[0] == str(tmp_path / ".venv" / "bin")
    assert out["OTHER"] == "keep"
    assert env["PATH"] == "/usr/bin"         # the caller's dict is untouched


@pytest.mark.skipif(os.name == "nt", reason="POSIX shebang executable")
def test_item14_run_resolves_a_tool_from_the_project_venv(tmp_path):
    bin_dir = tmp_path / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    tool = bin_dir / "pedacito-fake-tool"
    tool.write_text("#!/bin/sh\necho venv-tool-ran\n", encoding="utf-8")
    tool.chmod(0o755)
    rc, out, stopped, _secs = proc.run(["pedacito-fake-tool"], cwd=tmp_path,
                                       timeout=30)
    assert stopped == "" and rc == 0
    assert "venv-tool-ran" in out


# ===========================================================================
# Item 7: the browser session runs in a bounded worker subprocess
# ===========================================================================

def test_item7_fake_playwright_stays_in_process(monkeypatch):
    fake = types.ModuleType("playwright")    # no __file__: a test double
    monkeypatch.setitem(sys.modules, "playwright", fake)
    seen = {}

    def inproc(url, *, actions, wait, out, timeout, headless):
        seen["url"] = url
        return True, {"title": "ok"}

    monkeypatch.setattr(browser, "_playwright_inproc", inproc)
    ok, info = browser._run_playwright("http://x/", actions="", wait=0,
                                       out=None, timeout=5, headless=True)
    assert ok is True and info == {"title": "ok"}
    assert seen["url"] == "http://x/"


def test_item7_real_playwright_spawns_a_bounded_worker(monkeypatch):
    fake = types.ModuleType("playwright")
    fake.__file__ = "/fake/playwright/__init__.py"
    monkeypatch.setitem(sys.modules, "playwright", fake)
    captured = {}

    def fake_run(cmd, *, cwd, timeout, max_output):
        captured["cmd"] = cmd
        captured["timeout"] = timeout
        return 0, browser._WORKER_MARK + json.dumps(
            {"ok": True, "info": {"title": "page"}}), "", 0.1

    monkeypatch.setattr(browser.proc, "run", fake_run)
    ok, info = browser._run_playwright(
        "http://x/", actions="wait 2; hold #b 3", wait=1,
        out=None, timeout=20, headless=True)
    assert ok is True and info == {"title": "page"}
    assert "--worker" in captured["cmd"]
    # cap = timeout + scripted dwell (wait 1 + wait 2 + hold 3) + 15 grace
    assert captured["timeout"] == 20 + 1 + 2 + 3 + 15


def test_item7_worker_timeout_is_an_honest_failure(monkeypatch):
    fake = types.ModuleType("playwright")
    fake.__file__ = "/fake/playwright/__init__.py"
    monkeypatch.setitem(sys.modules, "playwright", fake)
    monkeypatch.setattr(browser.proc, "run",
                        lambda *a, **k: (None, "", "timeout", 99.0))
    ok, info = browser._run_playwright("http://x/", actions="", wait=0,
                                       out=None, timeout=5, headless=True)
    assert ok is False
    assert "NOT verified" in info


# ===========================================================================
# Item 8: --strict exempts a purely pre-existing red suite (CLI gate)
# ===========================================================================

def test_item8_strict_skips_preexisting_only_failure():
    args = SimpleNamespace(strict=True)
    tr = SimpleNamespace(results=[], tests_ok=False,
                         tests_preexisting_only=True)
    assert cli._strict_task_error(args, tr) == ""
    tr.tests_preexisting_only = False
    assert "still fail" in cli._strict_task_error(args, tr)


def test_item8_strict_still_fails_a_bad_edit_block():
    args = SimpleNamespace(strict=True)
    tr = SimpleNamespace(results=[Result(False, "a.py", "search not found")],
                         tests_ok=True, tests_preexisting_only=True)
    assert "edit block(s) failed" in cli._strict_task_error(args, tr)
    assert cli._strict_task_error(SimpleNamespace(strict=False), tr) == ""


# ===========================================================================
# Item 9: task text naming a browser check enables verify_dom
# ===========================================================================

@pytest.mark.parametrize("task", [
    "Run a dom check on the page.",
    "Open it in a headless browser and confirm the chart renders.",
    "Verify the counter in the browser.",
    "Test the layout in a browser.",
    "Use browser verification for the landing page.",
])
def test_item9_dom_phrases_enable_verify_dom(tmp_path, task):
    cfg = _cfg(verify_dom=False, allow_shell=True)
    loop._infer_task_checks(tmp_path, cfg, task)
    assert cfg.verify_dom is True


def test_item9_unrelated_browser_prose_does_not_enable(tmp_path):
    cfg = _cfg(verify_dom=False, allow_shell=True)
    loop._infer_task_checks(tmp_path, cfg,
                            "Add a browser history panel to the settings page.")
    assert cfg.verify_dom is False


# ===========================================================================
# Item 10: empty JS test files are flagged; node's 0-tests pass is noted
# ===========================================================================

def test_item10_empty_test_file_is_a_warning():
    r = Result(True, "x.test.js", "applied",
               before="test('a', () => {});\n",
               after="// helpers only\nfunction setup() {}\n")
    found = verify.check_empty_js_tests([r])
    assert len(found) == 1
    assert found[0].kind == "empty_tests" and found[0].severity == "warn"


def test_item10_real_cases_and_exclusions():
    ok = Result(True, "x.test.js", "applied", before="a\n",
                after="test('a', () => {});\n")
    assert verify.check_empty_js_tests([ok]) == []
    # `function test(` is a helper declaration, `.test(` a method call,
    # comments are not cases -- none of them counts as a test.
    for body in ("function test(name, fn) { fn(); }\n",
                 "const re = /a/;\nre.test('a');\n",
                 "// test('commented out', () => {});\n"):
        r = Result(True, "x.test.js", "applied", before="a\n", after=body)
        assert len(verify.check_empty_js_tests([r])) == 1
    # An unchanged file is not this edit's responsibility.
    same = Result(True, "x.test.js", "applied", before="a\n", after="a\n")
    assert verify.check_empty_js_tests([same]) == []


def test_item10_wired_into_verify_check(tmp_path):
    agent, cfg, idx = _files(tmp_path, {"x.test.js": "// empty\n"})
    r = Result(True, "x.test.js", "applied",
               before="test('a', () => {});\n", after="// empty\n")
    found = verify.check(agent, "add tests", "done", [r])
    assert any(f.kind == "empty_tests" for f in found)


def test_item10_node_zero_tests_note(tmp_path, monkeypatch):
    _patch_runner(monkeypatch, cmd="node --test", ok=True,
                  out="# tests 0\n# pass 0\n")
    idx = SimpleNamespace(root=tmp_path)
    ok, _out, desc = testsel.run_impacted(idx, _cfg(), ["x.test.js"])
    assert ok is True
    assert "0 tests" in desc


# ===========================================================================
# Item 11: wipe vetoes on every whole-file shape; lost exports in all idioms
# ===========================================================================

def test_item11_empty_replacement_is_refused_at_any_size():
    assert edits.fragment_refusal("a.js", "const x = 1;\n", "")
    assert edits.fragment_refusal("a.js", "const x = 1;\n", "   \n")


def test_item11_export_surface_wipe_is_refused():
    before = "export function go() { return 1; }\nexport const N = 2;\n"
    after = "function go() { return 1; }\nconst N = 2;\n"
    why = edits.fragment_refusal("a.js", before, after)
    assert why and "export" in why


def test_item11_empty_search_wipe_vetoed_by_apply(tmp_path):
    _w(tmp_path, "a.js", "export function go() { return 1; }\n")
    # Empty SEARCH + empty REPLACE is a parse-time no-op (discarded, never
    # applied); a whitespace-only REPLACE is the wipe shape that reaches apply.
    discards: list = []
    blocks = edits.parse_edits(
        "--- FILE: a.js\n<<<<<<< SEARCH\n=======\n>>>>>>> REPLACE\n",
        discards=discards)
    assert not blocks and len(discards) == 1
    blocks = edits.parse_edits(
        "--- FILE: a.js\n<<<<<<< SEARCH\n=======\n   \n>>>>>>> REPLACE\n")
    results = edits.apply_edits(tmp_path, blocks, backup=False)
    assert len(results) == 1 and not results[0].ok
    assert "erase the whole file" in results[0].message
    assert (tmp_path / "a.js").read_text(encoding="utf-8").startswith("export")


def test_item11_full_quote_wipe_vetoed_by_apply(tmp_path):
    before = "export function go() { return 1; }\n"
    _w(tmp_path, "a.js", before)
    blocks = edits.parse_edits(_edit("a.js", before, ""))
    results = edits.apply_edits(tmp_path, blocks, backup=False)
    assert not results[0].ok
    assert (tmp_path / "a.js").read_text(encoding="utf-8") == before


def test_item11_whole_file_flag_set_on_empty_search_overwrite(tmp_path):
    _w(tmp_path, "a.js", "export const N = 1;\n")
    blocks = edits.parse_edits(
        "--- FILE: a.js\n<<<<<<< SEARCH\n=======\n"
        "export const N = 2;\n>>>>>>> REPLACE\n")
    results = edits.apply_edits(tmp_path, blocks, backup=False)
    assert results[0].ok and results[0].whole_file is True


def test_item11_lost_exports_cjs_form(tmp_path):
    agent, cfg, idx = _files(tmp_path, {
        "lib.js": "module.exports = { go: go };\nfunction go() { return 1; }\n",
        "use.js": "const { go } = require('./lib.js');\ngo();\n",
    })
    before = (tmp_path / "lib.js").read_text(encoding="utf-8")
    r = Result(True, "lib.js", "replaced contents", before=before,
               after="function go() { return 1; }\n", whole_file=True)
    found = verify.check_lost_exports(agent, [r])
    assert any(f.kind == "lost_exports" and "go" in f.message for f in found)


def test_item11_lost_exports_window_form(tmp_path):
    agent, cfg, idx = _files(tmp_path, {
        "lib.js": "window.go = function () { return 1; };\n",
        "use.js": "window.go();\n",
    })
    before = (tmp_path / "lib.js").read_text(encoding="utf-8")
    r = Result(True, "lib.js", "replaced contents", before=before,
               after="const go = function () { return 1; };\n",
               whole_file=True)
    found = verify.check_lost_exports(agent, [r])
    assert any(f.kind == "lost_exports" and "go" in f.message for f in found)


def test_item11_lost_exports_esm_warns_when_task_asks_rewrite(tmp_path):
    agent, cfg, idx = _files(tmp_path, {
        "lib.js": "export function go() { return 1; }\n",
        "use.js": "import { go } from './lib.js';\ngo();\n",
    })
    before = (tmp_path / "lib.js").read_text(encoding="utf-8")
    r = Result(True, "lib.js", "replaced contents", before=before,
               after="export const other = 1;\n", whole_file=True)
    found = verify.check_lost_exports(agent, [r], task="rewrite lib.js")
    assert found and found[0].severity == "warn"


# ===========================================================================
# Item 12: the spec architect gets the repo's real runner and entries
# ===========================================================================

def test_item12_refine_system_names_the_rules():
    assert "use it verbatim" in spec.REFINE_SYSTEM
    assert "never invent a different runner" in spec.REFINE_SYSTEM
    assert "TOP-LEVEL ENTRIES" in spec.REFINE_SYSTEM
    assert "never invent domain restrictions" in spec.REFINE_SYSTEM


def test_item12_snapshot_leads_with_runner_and_entries(tmp_path):
    _w(tmp_path, "package.json",
       json.dumps({"name": "demo", "scripts": {"test": "node --test"}}))
    _w(tmp_path, "index.html", "<!doctype html><title>demo</title>")
    _w(tmp_path, "style.css", "body { margin: 0; }")
    _w(tmp_path, "main.js", "console.log('demo');\n")
    seen = {}

    class Spy:
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            seen["messages"] = messages
            return {}                      # useless reply -> fallback path

    out = spec.refine_prompt("Add a counter to the page.", tmp_path,
                             _cfg(), Spy())
    assert out["ok"] is True
    user = seen["messages"][1]["content"]
    # A real package.json test script wins over `node --test` discovery:
    # the snapshot names the runner the project actually invokes.
    assert "TEST RUNNER: npm test" in user
    assert "TOP-LEVEL ENTRIES:" in user
    assert "style.css" in user and "main.js" in user
    # The facts sit at the head of the snapshot, before the file list.
    snap = user.split("TEST RUNNER:", 1)[1]
    assert snap.index("TOP-LEVEL ENTRIES:") < snap.index("EXISTING PROJECT")


def test_item12_fallback_verification_uses_the_detected_runner():
    snapshot = ("TEST RUNNER: node --test\n"
                "EXISTING PROJECT FILES:\n  main.js\n")
    data = spec._fallback_spec("Add a counter.", snapshot, "auto")
    assert "node --test" in data["verification"]
    assert "npm test" not in data["verification"]


def test_item12_fallback_without_runner_does_not_fabricate():
    data = spec._fallback_spec("Add a counter.",
                               "TEST RUNNER: (none detected)\n", "auto")
    assert "node --test" not in data["verification"]
    assert "npm test" not in data["verification"]


# ===========================================================================
# Item 13: provider price defaults + the cost-blind budget warning
# ===========================================================================

def test_item13_model_price_table():
    p = providers.default_prices("https://api.deepseek.com/v1",
                                 "deepseek-reasoner")
    assert p == {"price_input_per_mtok": 0.55,
                 "price_cached_per_mtok": 0.14,
                 "price_output_per_mtok": 2.19}
    p = providers.default_prices("https://api.deepseek.com/v1", "deepseek-chat")
    assert p["price_input_per_mtok"] == 0.27
    p = providers.default_prices("https://dashscope.aliyuncs.com/v1", "qwen-max")
    assert p["price_output_per_mtok"] == 9.60
    p = providers.default_prices("https://api.openai.com/v1", "gpt-4o")
    assert p["price_input_per_mtok"] == 2.50


def test_item13_host_fallback_and_local_zero():
    # An unnamed model on a known host falls back to the host's flagship price.
    p = providers.default_prices("https://api.deepseek.com/v1", "some-alias")
    assert p["price_input_per_mtok"] == 0.27
    # Local servers stay unpriced for models the table does not name. (The
    # name table itself is host-independent -- locality gates the config
    # fill, pinned in test_item13_fill_only_when_all_defaults_and_remote.)
    assert providers.default_prices("http://127.0.0.1:11434/v1",
                                    "local-model") == {}
    assert providers.default_prices("https://unknown.example/v1", "m") == {}


def test_item13_fill_only_when_all_defaults_and_remote():
    cfg = _cfg(base_url="https://api.deepseek.com/v1", model="deepseek-chat")
    cfg.sources = {}
    config_mod.fill_default_prices(cfg)
    assert cfg.price_input_per_mtok == 0.27
    assert "provider price table" in cfg.sources["price_input_per_mtok"]
    # Any explicit price wins: the whole fill is skipped.
    cfg2 = _cfg(base_url="https://api.deepseek.com/v1", model="deepseek-chat")
    cfg2.sources = {"price_input_per_mtok": "user"}
    cfg2.price_input_per_mtok = 99.0
    config_mod.fill_default_prices(cfg2)
    assert cfg2.price_input_per_mtok == 99.0
    assert cfg2.price_output_per_mtok == 0.0
    # Local endpoints are never filled.
    cfg3 = _cfg(base_url="http://localhost:1234/v1", model="deepseek-chat")
    cfg3.sources = {}
    config_mod.fill_default_prices(cfg3)
    assert cfg3.price_input_per_mtok == 0.0


def test_item13_refill_after_url_model_override():
    # load() priced the config file's local endpoint (nothing to fill);
    # --url/--model then point at a hosted one: the cli re-fills.
    cfg = _cfg(base_url="http://localhost:1234/v1", model="local")
    cfg.sources = {}
    config_mod.fill_default_prices(cfg)
    assert cfg.price_input_per_mtok == 0.0
    cfg.base_url = "https://api.openai.com/v1"
    cfg.model = "gpt-4o"
    config_mod.fill_default_prices(cfg)
    assert cfg.price_input_per_mtok == 2.50


def test_item13_cost_blind_warning():
    cfg = _cfg(max_cost=5.0)
    assert "can never trigger" in config_mod.cost_blind_warning(cfg)
    cfg.price_input_per_mtok = 0.27
    assert config_mod.cost_blind_warning(cfg) == ""
    cfg2 = _cfg(max_cost=0.0)
    assert config_mod.cost_blind_warning(cfg2) == ""
    # Kept out of budget_warnings, which early-returns for hosted APIs.
    cfg3 = _cfg(base_url="https://api.deepseek.com/v1", max_cost=5.0)
    assert config_mod.budget_warnings(cfg3) == []


# ===========================================================================
# BUG-209: consecutive malformed gather steps hit a bounded breaker
# ===========================================================================

def test_bug209_malformed_streak_breaks_gather(tmp_path):
    client = Scripted([{"action": "read"} for _ in range(8)])
    agent, cfg, idx = _files(tmp_path, {"a.py": "X = 1\n"})
    agent.client = client
    agent.gather("Inspect a.py.", verbose=False)
    rejected = [s for s in agent.steps_log
                if "rejected: no file or symbol" in s]
    assert len(rejected) == agent_mod._MAX_MALFORMED_STEPS == 3


def test_bug209_rejection_names_a_copyable_example(tmp_path):
    client = Scripted([{"action": "read"}, {"action": "done"}])
    agent, cfg, idx = _files(tmp_path, {"a.py": "X = 1\n"})
    agent.client = client
    agent.gather("Inspect a.py.", verbose=False)
    # The second chat_json call carries the rejection observation. The
    # example is compact json.dumps output; accept either colon spacing.
    obs = client.json_calls[1][-1]["content"]
    assert "Example:" in obs
    assert '"file":"a.py"' in obs or '"file": "a.py"' in obs


def test_bug209_valid_step_resets_the_streak(tmp_path):
    steps = [{"action": "read"},
             {"action": "read", "file": "a.py", "start": 1, "end": 1},
             {"action": "read"}, {"action": "read"}, {"action": "read"},
             {"action": "done"}]
    client = Scripted(steps)
    agent, cfg, idx = _files(tmp_path, {"a.py": "X = 1\n"})
    agent.client = client
    agent.gather("Inspect a.py.", verbose=False)
    rejected = [s for s in agent.steps_log
                if "rejected: no file or symbol" in s]
    # 1 + 3 rejections: the valid read reset the counter mid-run.
    assert len(rejected) == 4


# ===========================================================================
# BUG-213: the coverage guard tolerates an agent without _coverage_pass
# ===========================================================================

class _BareAgent:
    """The fake test agent: gather/answer/apply, no coverage machinery."""

    def __init__(self, idx):
        self.index = idx
        self.steps_log = []

    def gather(self, task, backup_dir=None, is_repair=False):
        return "gathered"

    def answer(self, task, gathered, system=None):
        return _edit("a.py", "X = 1\n", "X = 2\n")

    def apply(self, reply, dry_run=False, task="", gathered="",
              backup_dir=None):
        return [Result(True, "a.py", "applied")], reply


def test_bug213_bare_agent_runs_the_answer_path(tmp_path):
    _w(tmp_path, "a.py", "X = 1\n")
    cfg = _cfg(static_gate=False, frontend_contracts=False,
               interface_lock=False, use_memory=False, backup=False,
               verify_tests=False, post_edit_check=False)
    idx = index_mod.build([str(tmp_path)], cfg, root=tmp_path,
                          summarise=False, verbose=False)
    agent = _BareAgent(idx)
    assert not hasattr(agent, "_coverage_pass")
    tr = loop._run_task_once(tmp_path, cfg, None, agent, idx,
                             "Change X in a.py.", apply=True)
    assert tr.applied is True
    assert any(r.ok for r in tr.results)


# ===========================================================================
# BUG-218: rejected files go back to pre-task bytes at the failure exit
# ===========================================================================

def test_bug218_restore_modified_and_remove_created(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    _w(root, "mod.js", "BROKEN = true;\n")
    _w(root, "new.js", "CREATED = true;\n")
    backup = tmp_path / "snap"
    backup.mkdir()
    _w(backup, "mod.js", "ORIGINAL = true;\n")   # pre-task bytes
    fileio.note_created(backup, "new.js")
    cfg = _cfg()
    idx = index_mod.build([str(root)], cfg, root=root,
                          summarise=False, verbose=False)
    restored, deleted, idx = loop._restore_rejected_files(
        root, backup, ["mod.js", "new.js"], idx, cfg)
    assert restored == ["mod.js"] and deleted == ["new.js"]
    assert (root / "mod.js").read_text(encoding="utf-8") == "ORIGINAL = true;\n"
    assert not (root / "new.js").exists()


def test_bug218_none_backup_is_a_graceful_noop(tmp_path):
    _w(tmp_path, "mod.js", "BROKEN = true;\n")
    cfg = _cfg()
    idx = index_mod.build([str(tmp_path)], cfg, root=tmp_path,
                          summarise=False, verbose=False)
    restored, deleted, _idx = loop._restore_rejected_files(
        tmp_path, None, ["mod.js"], idx, cfg)
    assert restored == [] and deleted == []
    assert (tmp_path / "mod.js").read_text(encoding="utf-8") == "BROKEN = true;\n"


def test_bug218_unresolved_frontend_gate_rolls_back(tmp_path):
    # The contract gate ends unresolved (no repair client): the rejected file
    # must not stay broken in the tree.
    _w(tmp_path, "index.html",
       '<!doctype html><body><div id="stage"></div>'
       '<script type="module" src="main.js"></script></body>')
    _w(tmp_path, "main.js", "document.getElementById('stage');\n")
    cfg = _cfg(frontend_contracts=True, static_gate=False,
               interface_lock=False, repair_rounds=0, backup=True)
    idx = index_mod.build([str(tmp_path)], cfg, root=tmp_path,
                          summarise=False, verbose=False)
    backup = tmp_path / "snap"
    backup.mkdir()
    _w(backup, "main.js", "document.getElementById('stage');\n")
    # The task broke the wiring; the snapshot holds the working bytes.
    _w(tmp_path, "main.js", "document.getElementById('board');\n")
    results = [Result(True, "main.js", "applied")]
    _res, _reply, tests_ok, _pre = loop._post_edit_tests(
        tmp_path, cfg, _NoRepair(), idx, "wire the page", results, "reply",
        backup, None, ["main.js"], collect=[])
    assert (tmp_path / "main.js").read_text(encoding="utf-8") == \
        "document.getElementById('stage');\n"


class _NoRepair:
    def _repair_from_findings(self, *a, **k):
        return ""


# ===========================================================================
# Item 15: editing specialties are gated on the measured edit skill
# ===========================================================================

def test_item15_gate_strips_editing_tags_below_threshold():
    measured = {"edit_format": 0.6, "tool_choice": 0.9, "svg_ui": 0.95,
                "math_algo": 0.95, "testing_qa": 0.95, "sql_data": 0.95}
    tags = modelcard.specialty_tags_from_measurements(
        "qwen2.5-coder-14b", measured)
    for t in ("coding", "frontend_ui", "3d_graphics_svg", "math_algorithms",
              "testing_qa", "sql_data"):
        assert t not in tags, t


def test_item15_gate_passes_at_threshold():
    measured = {"edit_format": 0.7, "tool_choice": 0.6, "svg_ui": 0.95}
    tags = modelcard.specialty_tags_from_measurements("some-model", measured)
    assert "frontend_ui" in tags and "3d_graphics_svg" in tags


def test_item15_tool_choice_floor():
    measured = {"edit_format": 0.9, "tool_choice": 0.4, "svg_ui": 0.95}
    tags = modelcard.specialty_tags_from_measurements("some-model", measured)
    assert "frontend_ui" not in tags


def test_item15_translation_is_ungated():
    measured = {"edit_format": 0.0, "tool_choice": 0.0, "translation": 0.95}
    tags = modelcard.specialty_tags_from_measurements("aya-35b", measured)
    assert tags == ["translation_i18n"]


def test_item15_strong_pair_earns_coding_and_testing():
    measured = {"edit_format": 0.8, "tool_choice": 0.8}
    tags = modelcard.specialty_tags_from_measurements("plain-model", measured)
    assert "coding" in tags and "testing_qa" in tags
    weak = {"edit_format": 0.4, "tool_choice": 0.9}
    tags = modelcard.specialty_tags_from_measurements("coder-model", weak)
    assert "coding" not in tags


@pytest.mark.parametrize("name", ["ocr-large", "router-7b", "orchestrator-x",
                                  "embed-base"])
def test_item15_utility_names_never_edit(name):
    measured = {"edit_format": 0.99, "tool_choice": 0.99, "svg_ui": 0.99}
    assert modelcard.specialty_tags_from_measurements(name, measured) == \
        ["general"]


def test_item15_saved_cards_are_refiltered_on_load(home):
    cfg = _cfg(base_url="http://127.0.0.1:11434/v1", model="coder-14b")
    _save_card("coder-14b", measured={
        "overall": 0.8, "edit_format": 0.2, "tool_choice": 0.9,
        "specialties": ["coding", "testing_qa"]})
    card = modelcard.load(cfg)
    assert card is not None
    assert "coding" not in card["measured"]["specialties"]
    # The re-derivation persisted, so the next reader sees the gated tags.
    raw = json.loads((home / "model_cards.json").read_text(encoding="utf-8"))
    stored = next(iter(raw.values()))
    assert "coding" not in stored["measured"]["specialties"]


def test_item15_refilter_persists_only_on_change(home):
    cfg = _cfg(base_url="http://127.0.0.1:11434/v1", model="coder-14b")
    _save_card("coder-14b", measured={
        "overall": 0.8, "edit_format": 0.9, "tool_choice": 0.9,
        "specialties": ["coding", "testing_qa"]})
    p = home / "model_cards.json"
    before = p.read_text(encoding="utf-8")
    modelcard.load(cfg)
    assert p.read_text(encoding="utf-8") == before


# ===========================================================================
# Item 16: composite skill scoring + bounded inertia in specialist routing
# ===========================================================================

LOCAL = "http://127.0.0.1:11434/v1"


def _routed_cfg(**kw):
    cfg = Config()
    cfg.base_url = LOCAL
    cfg.model = "worker-8b"
    cfg.vram_gb = 24
    cfg.sources = {}
    cfg.active_profile = "worker"
    cfg.auto_route_specialties = True
    cfg.profiles = {
        "worker": {"model": "worker-8b", "tier": "small",
                   "specialties": ["frontend_ui"]},
        "ui": {"model": "ui-27b", "tier": "medium",
               "specialties": ["frontend_ui"]},
    }
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_item16_composite_skill_formula(home):
    cfg = _routed_cfg()
    _save_card("ui-27b", measured={"svg_ui": 0.8, "edit_format": 0.6,
                                   "overall": 0.7})
    skill = roles._specialist_skill(cfg, cfg.profiles, "ui", "frontend_ui")
    assert abs(skill - (0.40 * 0.8 + 0.35 * 0.6 + 0.25 * 0.7 + 0.02)) < 1e-9
    # An uncalibrated profile scores only its tier bonus.
    assert roles._specialist_skill(cfg, cfg.profiles, "worker",
                                   "frontend_ui") == 0.01


def test_item16_inertia_keeps_a_close_task(home):
    cfg = _routed_cfg()
    _save_card("worker-8b", measured={"svg_ui": 0.5, "edit_format": 0.5,
                                      "overall": 0.5})
    _save_card("ui-27b", measured={"svg_ui": 0.55, "edit_format": 0.5,
                                   "overall": 0.5})
    name, tag = roles.select_specialist_profile(
        "[specialty: frontend_ui] draw the layout", None, cfg)
    assert (name, tag) == ("", "")           # lead < 0.08: worker keeps it


def test_item16_clear_lead_overrides_inertia(home):
    cfg = _routed_cfg()
    _save_card("worker-8b", measured={"svg_ui": 0.5, "edit_format": 0.5,
                                      "overall": 0.5})
    _save_card("ui-27b", measured={"svg_ui": 0.9, "edit_format": 0.9,
                                   "overall": 0.9})
    name, tag = roles.select_specialist_profile(
        "[specialty: frontend_ui] draw the layout", None, cfg)
    assert name == "ui" and tag == "frontend_ui"


def test_item16_router_named_active_never_hoards(home):
    cfg = _routed_cfg()
    cfg.profiles["worker"]["model"] = "orchestrator-8b"
    cfg.model = "orchestrator-8b"
    # The specialist lives on its own server: once the hoarding override
    # lifts the inertia, the separate-server sort key puts it first even
    # though the orchestrator's card is stronger.
    local2 = "http://127.0.0.1:11435/v1"
    cfg.profiles["ui"]["base_url"] = local2
    _save_card("orchestrator-8b", measured={"svg_ui": 0.9, "edit_format": 0.9,
                                            "overall": 0.9})
    _save_card("ui-27b", url=local2,
               measured={"svg_ui": 0.5, "edit_format": 0.5, "overall": 0.5})
    name, tag = roles.select_specialist_profile(
        "[specialty: frontend_ui] draw the layout", None, cfg)
    assert name == "ui"                      # even with a weaker card
    # Control: the SAME cards under a normal worker name keep the task --
    # the skill gap is far below the inertia margin, so only the router
    # name made the difference above.
    cfg.profiles["worker"]["model"] = "worker-8b"
    cfg.model = "worker-8b"
    _save_card("worker-8b", measured={"svg_ui": 0.9, "edit_format": 0.9,
                                      "overall": 0.9})
    name, tag = roles.select_specialist_profile(
        "[specialty: frontend_ui] draw the layout", None, cfg)
    assert (name, tag) == ("", "")


def test_item16_tier_bonus_never_outvotes_skill(home):
    cfg = _routed_cfg()
    cfg.profiles["ui"]["tier"] = "hosted"    # the biggest bonus: 0.03
    cfg.profiles["ui"]["base_url"] = LOCAL   # ...but keep it in the local pool
    _save_card("worker-8b", measured={"svg_ui": 0.9, "edit_format": 0.9,
                                      "overall": 0.9})
    _save_card("ui-27b", measured={"svg_ui": 0.1, "edit_format": 0.1,
                                   "overall": 0.1})
    name, _tag = roles.select_specialist_profile(
        "[specialty: frontend_ui] draw the layout", None, cfg)
    assert name == ""                        # skill lead beats the tier bonus


# ===========================================================================
# Item 17: the no-extras spec detection covers the imperative forms
# ===========================================================================

@pytest.mark.parametrize("text", [
    "Do not add README or extra files.",
    "Create exactly these 10 files.",
    "no README or additional files",
    "without a readme",
    "only create main.js",
    "create only main.js and index.html",
    "Generate exactly 3 files.",
    "omit any extra files",
    "never add a README",
])
def test_item17_forbidding_forms_match(text):
    assert scaffold._spec_forbids_extras(text) is True


@pytest.mark.parametrize("text", [
    "Build a small web app.",
    "The function only creates a cache entry.",   # descriptive, not a constraint
    "Add a README describing the API.",
])
def test_item17_neutral_forms_do_not_match(text):
    assert scaffold._spec_forbids_extras(text) is False


# ===========================================================================
# Item 18: --include-labels moves test-label strings only
# ===========================================================================

def test_item18_label_helper_moves_only_test_labels():
    text = ('test("oldName works", () => { oldName(); });\n'
            'it.skip("oldName edge", () => {});\n'
            'describe("oldName suite", () => {});\n'
            'const msg = "oldName is mentioned in prose";\n'
            '// oldName comment\n')
    out, n = refactor._rename_in_test_labels(text, "oldName", "newName", ".js")
    assert n == 3
    assert 'test("newName works"' in out
    assert 'it.skip("newName edge"' in out
    assert 'describe("newName suite"' in out
    assert '"oldName is mentioned in prose"' in out
    assert "// oldName comment" in out


def test_item18_rename_include_labels_end_to_end(tmp_path):
    _w(tmp_path, "src/t.js",
       "export function oldName(x) { return x; }\n")
    _w(tmp_path, "tests/t.test.js",
       'import { oldName } from "../src/t.js";\n'
       'test("oldName adds step", () => { oldName(1); });\n'
       'const doc = "oldName in prose stays";\n')
    res = refactor.rename_symbol_across_project(
        tmp_path, "oldName", "newName", include_labels=True)
    assert res["ok"]
    assert res["label_replacements"] == 1
    js = (tmp_path / "tests/t.test.js").read_text(encoding="utf-8")
    assert 'test("newName adds step"' in js
    assert '"oldName in prose stays"' in js
    assert "newName(1)" in js


def test_item18_default_leaves_labels_alone(tmp_path):
    _w(tmp_path, "src/t.js",
       "export function oldName(x) { return x; }\n")
    _w(tmp_path, "tests/t.test.js",
       'import { oldName } from "../src/t.js";\n'
       'test("oldName adds step", () => { oldName(1); });\n')
    res = refactor.rename_symbol_across_project(tmp_path, "oldName", "newName")
    assert res["ok"] and res["label_replacements"] == 0
    js = (tmp_path / "tests/t.test.js").read_text(encoding="utf-8")
    assert 'test("oldName adds step"' in js


def test_item18_labels_are_a_subset_of_strings(tmp_path):
    _w(tmp_path, "src/t.js",
       "export function oldName(x) { return x; }\n")
    _w(tmp_path, "tests/t.test.js",
       'import { oldName } from "../src/t.js";\n'
       'test("oldName adds step", () => { oldName(1); });\n'
       'const doc = "oldName in prose";\n')
    res = refactor.rename_symbol_across_project(
        tmp_path, "oldName", "newName", include_strings=True,
        include_labels=True)
    assert res["ok"]
    # --strings already moved every string: the label pass finds nothing left.
    assert res["label_replacements"] == 0
    assert res["string_replacements"] >= 2
    js = (tmp_path / "tests/t.test.js").read_text(encoding="utf-8")
    assert '"newName in prose"' in js


def test_item18_cli_flag_is_wired():
    parser = cli.build_parser()
    args = parser.parse_args(["rename", "old", "new", "--include-labels"])
    assert args.include_labels is True


# ===========================================================================
# Item 19: procs list collapses superseded stopped entries
# ===========================================================================

def _proc_record(root, name, status, started, command="npm run dev",
                 url="", pid=0):
    processes._write_record(root, {
        "name": name, "command": command, "pid": pid, "cwd": str(root),
        "log": str(root / f"{name}.log"), "url": url, "job": "",
        "started": started, "status": status,
    })


def test_item19_stopped_entry_superseded_by_running_match(tmp_path):
    now = time.time()
    _proc_record(tmp_path, "dev", "stopped", now - 100)
    _proc_record(tmp_path, "dev-2", "running", now, pid=os.getpid())
    out = processes.render_list(tmp_path)
    assert "dev-2" in out
    assert not any(ln.startswith("dev ") for ln in out.splitlines())
    assert "+ 1 superseded stopped entry" in out
    assert "--all" in out
    full = processes.render_list(tmp_path, all=True)
    assert any(ln.startswith("dev ") for ln in full.splitlines())
    assert "superseded" not in full


def test_item19_url_match_across_loopback_spellings(tmp_path):
    now = time.time()
    _proc_record(tmp_path, "web", "stopped", now - 10,
                 command="npm run dev", url="http://127.0.0.1:3000")
    _proc_record(tmp_path, "frontend", "running", now, command="vite",
                 url="http://localhost:3000/", pid=os.getpid())
    out = processes.render_list(tmp_path)
    assert "web" not in out and "+ 1 superseded" in out


def test_item19_different_port_is_a_different_server(tmp_path):
    now = time.time()
    _proc_record(tmp_path, "web", "stopped", now - 10,
                 url="http://localhost:3000")
    _proc_record(tmp_path, "other", "stopped", now - 5,
                 command="vite --port 3001", url="http://localhost:3001")
    _proc_record(tmp_path, "frontend", "running", now, command="vite",
                 url="http://localhost:3000", pid=os.getpid())
    out = processes.render_list(tmp_path)
    assert "other" in out                    # not superseded: different port


def test_item19_exited_entries_are_never_collapsed(tmp_path):
    now = time.time()
    _proc_record(tmp_path, "dev", "exited", now - 100)
    _proc_record(tmp_path, "dev-2", "running", now, pid=os.getpid())
    out = processes.render_list(tmp_path)
    assert any(ln.startswith("dev ") for ln in out.splitlines())


def test_item19_stopped_newer_than_the_running_match_stays(tmp_path):
    now = time.time()
    _proc_record(tmp_path, "dev", "running", now - 100, pid=os.getpid())
    _proc_record(tmp_path, "dev-2", "stopped", now)
    out = processes.render_list(tmp_path)
    assert "dev-2" in out


def test_item19_cli_all_flag_is_wired():
    parser = cli.build_parser()
    args = parser.parse_args(["procs", "--all"])
    assert args.all is True
    args = parser.parse_args(["procs"])
    assert args.all is False


# ===========================================================================
# Item 20: the docs name the new behavior (drift pins; the full guards live
# in test_doc_sync.py)
# ===========================================================================

def test_item20_readme_fetch_row_names_the_loopback_allowlist():
    text = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    assert 'fetch_allowed_hosts = ["127.0.0.1", "localhost"]' in text


def test_item20_model_guide_hardening_notes():
    text = " ".join((ROOT / "MODEL_GUIDE.md").read_text(encoding="utf-8").split())
    assert "edit_format >= 0.65" in text
    assert "4B limits" in text
    assert "does not reserve VRAM" in text


def test_item20_procs_all_documented():
    for doc in ("README.md", "QUICKSTART.md"):
        text = " ".join((ROOT / doc).read_text(encoding="utf-8").split())
        assert "superseded stopped" in text, doc
    cards = {c["name"]: c for c in how_index.CLI_COMMANDS}
    assert "--all" in cards["procs"]["usage"]
    assert any(f["flag"] == "--all" for f in cards["procs"]["flags"])