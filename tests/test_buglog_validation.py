"""Regression tests for the bug-hunting report (`PEDACITO_BUG_LOG.md`, Nov 2026).

Each section pins the deterministic half of one reported finding: the check
that did not exist, the message that was missing, or the exit code that lied.
Findings whose cause is model behaviour with no harness-side determinant
(naming drift, prose contradictions, a plan that asks for the wrong thing)
are deliberately NOT represented here -- the validation report lists them as
outstanding rather than dressing them up as tests.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import agent as agent_mod, cli, config as config_mod, rpc  # noqa: E402
from pedacito import scaffold, spec, tools, verify, webcheck  # noqa: E402
from pedacito.agent import Agent  # noqa: E402
from pedacito.config import Config  # noqa: E402
from pedacito.scaffold import Plan, PlannedFile, Spec  # noqa: E402


def _cfg(**kw) -> Config:
    c = Config()
    c.log_sessions = False
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _files(tmp_path, tree: dict[str, str]):
    """A real index + Agent over an in-memory project (no LLM)."""
    from pedacito import index as index_mod
    for rel, text in tree.items():
        _w(tmp_path, rel, text)
    cfg = _cfg()
    idx = index_mod.build([str(tmp_path)], cfg, root=tmp_path,
                          summarise=False, verbose=False)
    agent = Agent(idx, None, cfg)
    return agent, cfg, idx


def _w(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


# ------------------------------------------- BUG-004: a name bound twice ----
# The incremental loop appends `import { WAYPOINTS }` to a file that already
# binds it. The file then never evaluates: `SyntaxError: Identifier ... has
# already been declared`, and every module importing it fails too. A parse of
# the single file is valid, so no existing check saw it.

def test_gate_flags_a_module_level_name_bound_twice(tmp_path):
    _w(tmp_path, "index.html",
       '<!doctype html><script type="module" src="src/waves.js"></script>')
    _w(tmp_path, "src/waves.js",
       "import { WAYPOINTS } from './path.js';\n"
       "import { WAYPOINTS } from './path2.js';\n"
       "export function fire() { return WAYPOINTS.length; }\n")
    errs = spec.verify_frontend_contracts(tmp_path, None)
    assert any("WAYPOINTS" in e and "already been declared" in e
               for e in errs), errs


def test_gate_flags_an_import_that_rebinds_a_local(tmp_path):
    """Mixed lexical/import binding is the same fatal SyntaxError."""
    _w(tmp_path, "index.html",
       '<!doctype html><script type="module" src="a.js"></script>')
    _w(tmp_path, "a.js",
       "const score = 0;\nimport { score } from './b.js';\n")
    errs = spec.verify_frontend_contracts(tmp_path, None)
    assert any("score" in e and "already" in e for e in errs), errs


def test_two_hoisted_functions_are_one_function(tmp_path):
    """In a CLASSIC SCRIPT re-declaring `function f` is legal (the last wins)
    -- the gate that reports a SyntaxError here would be a false positive on
    working code. (In an ES module the same pair is a SyntaxError; see the
    module tests below.)"""
    _w(tmp_path, "index.html",
       '<!doctype html><script src="a.js"></script>')
    _w(tmp_path, "a.js",
       "function f() { return 1; }\nfunction f() { return 2; }\nf();\n")
    assert spec.verify_frontend_contracts(tmp_path, None) == []


def _no_node(monkeypatch):
    """Hide node so the binding gate is exercised without `node --check`
    (which would report the same SyntaxError first and mask the gate)."""
    import shutil
    real = shutil.which
    monkeypatch.setattr(shutil, "which",
                        lambda name, *a, **k: None if name == "node"
                        else real(name, *a, **k))


def test_module_refuses_two_functions_with_one_name(tmp_path, monkeypatch):
    """An ES module may not re-declare `function f`: V8 raises SyntaxError and
    the whole module graph dies. The binding gate must catch it even when
    node is not installed."""
    _no_node(monkeypatch)
    _w(tmp_path, "index.html",
       '<!doctype html><script type="module" src="a.js"></script>')
    _w(tmp_path, "a.js",
       "function f() { return 1; }\nfunction f() { return 2; }\n"
       "export { f };\n")
    errs = spec.verify_frontend_contracts(tmp_path, None)
    assert any("f" in e and "already" in e for e in errs), errs
    found = webcheck.run(tmp_path)
    assert any("'f'" in f.message and "already" in f.message for f in found), found


def test_module_refuses_a_function_next_to_a_var(tmp_path, monkeypatch):
    """`function f` + `var f` is legal in a script but a SyntaxError in a
    module, in either order."""
    _no_node(monkeypatch)
    _w(tmp_path, "index.html",
       '<!doctype html><script type="module" src="a.js"></script>')
    _w(tmp_path, "a.js",
       "var f = 1;\nfunction f() { return 2; }\nexport { f };\n")
    errs = spec.verify_frontend_contracts(tmp_path, None)
    assert any("f" in e and "already" in e for e in errs), errs


def test_module_allows_two_vars_with_one_name(tmp_path, monkeypatch):
    """`var f; var f;` is the one repeat an ES module allows -- no finding."""
    _no_node(monkeypatch)
    _w(tmp_path, "index.html",
       '<!doctype html><script type="module" src="a.js"></script>')
    _w(tmp_path, "a.js", "var f = 1;\nvar f = 2;\nexport { f };\n")
    assert spec.verify_frontend_contracts(tmp_path, None) == []
    assert webcheck.run(tmp_path) == []


# ---------------------------------- BUG-009/010/018: unreachable call -------
# A module's scope holds only what it declares or imports. A call to a name
# bound nowhere is a ReferenceError on the first execution of that line, and
# it is invisible to a parser. Three shapes, three different fixes.

def _exported_elsewhere(tmp_path):
    _w(tmp_path, "index.html",
       '<!doctype html><script type="module" src="src/main.js"></script>')
    _w(tmp_path, "src/audio.js", "export function playMortar() { return 1; }\n")
    _w(tmp_path, "src/main.js",
       "import { init } from './audio.js';\n"
       "init();\nplayMortar();\n")


def test_gate_names_the_module_that_exports_the_name(tmp_path):
    _exported_elsewhere(tmp_path)
    errs = spec.verify_frontend_contracts(tmp_path, None)
    hit = [e for e in errs if "never imports it" in e]
    assert hit, errs
    assert "playMortar" in hit[0] and "src/audio.js" in hit[0]
    assert "import { playMortar }" in hit[0]      # the exact fix
    assert "ReferenceError" in hit[0]
    # the bad `init` import is the OTHER check's contract, and it is still
    # reported beside this one
    assert any("does not export it" in e for e in errs), errs


def test_gate_flags_a_declaration_that_was_never_exported(tmp_path):
    """Both files load as modules: `fmt` exists, but its module keeps it
    private, so the call in main.js can only ever throw."""
    _w(tmp_path, "index.html",
       '<!doctype html><script type="module" src="main.js"></script>'
       '<script type="module" src="helpers.js"></script>')
    _w(tmp_path, "helpers.js",
       "export const VERSION = 1;\nfunction fmt(n) { return n; }\n")
    _w(tmp_path, "main.js", "fmt(1);\n")
    errs = spec.verify_frontend_contracts(tmp_path, None)
    assert any("fmt" in e and "never exports" in e for e in errs), errs


def test_gate_flags_a_call_no_file_defines(tmp_path):
    _w(tmp_path, "index.html",
       '<!doctype html><script type="module" src="main.js"></script>')
    _w(tmp_path, "main.js", "spawnEnemy(3);\n")
    errs = spec.verify_frontend_contracts(tmp_path, None)
    assert any("spawnEnemy" in e and "ReferenceError" in e
               for e in errs), errs


def test_host_and_cdn_names_are_never_reported(tmp_path):
    """requestAnimationFrame is a browser global and `THREE` may arrive from a
    script tag: a gate that flags them vetoes working code."""
    _w(tmp_path, "index.html",
       '<!doctype html><script src="https://cdn/3.js"></script>'
       '<script type="module" src="main.js"></script>')
    _w(tmp_path, "main.js",
       "requestAnimationFrame(loop);\nnew THREE.Vector3();\n"
       "document.body.append('x');\nfunction loop(t) { return t; }\n")
    assert spec.verify_frontend_contracts(tmp_path, None) == []


def test_one_vocabulary_for_both_readers():
    """The gate used to keep its own suppression list, and it had grown HTML
    tag names: a call to a `Menu()` nobody defined became unspeakable. There
    is one list now, owned by webcheck and curated for the gate's sake, so
    the advisor and the veto cannot drift apart again."""
    assert spec._JS_GLOBAL_NAMES is webcheck.JS_GLOBAL_NAMES
    assert spec._JS_LIBRARY_GLOBALS is webcheck.JS_LIBRARY_GLOBALS
    assert spec._JS_TEST_GLOBALS is webcheck.JS_TEST_GLOBALS
    assert spec._JS_NON_NAMES is webcheck.JS_NON_NAMES
    assert spec._JS_GLOBAL_PREFIXES == webcheck.JS_GLOBAL_PREFIXES
    for name in ("Menu", "Table", "Form", "Title", "Data", "Slot", "Style"):
        assert not webcheck._js_is_host_name(name), name


def test_a_tag_shaped_helper_name_is_still_a_missing_helper(tmp_path):
    """Merged in: the window methods the gate stayed quiet about and the dump
    missed. Left out: the tag names, which belong to no scope at all."""
    _w(tmp_path, "index.html",
       '<!doctype html><script type="module" src="main.js"></script>')
    _w(tmp_path, "main.js",
       "addEventListener('keydown', () => {});\n"
       "Menu();\n")
    errs = spec.verify_frontend_contracts(tmp_path, None)
    assert any("Menu" in e for e in errs), errs
    assert not any("addEventListener" in e for e in errs), errs
    found = webcheck.run(tmp_path)
    assert any("Menu" in f.message for f in found), found
    assert not any("addEventListener" in f.message for f in found), found


def test_a_name_bound_by_import_or_destructure_is_not_free(tmp_path):
    _w(tmp_path, "index.html",
       '<!doctype html><script type="module" src="main.js"></script>')
    _w(tmp_path, "lib.js",
       "export const cfg = {a: 1};\nexport default function go() {}\n")
    _w(tmp_path, "main.js",
       "import go, { cfg as c } from './lib.js';\n"
       "const { draw } = c;\ngo(); draw(); c();\n")
    assert spec.verify_frontend_contracts(tmp_path, None) == []


def test_classic_script_globals_stay_visible(tmp_path):
    """A classic <script> top level really is window scope: main.js may be a
    module and legacy.js a classic script that defines its globals."""
    _w(tmp_path, "index.html",
       '<!doctype html><script src="legacy.js"></script>'
       '<script type="module" src="main.js"></script>')
    _w(tmp_path, "legacy.js", "function legacyInit() { return 1; }\n")
    _w(tmp_path, "main.js", "legacyInit();\n")
    assert spec.verify_frontend_contracts(tmp_path, None) == []


def test_comment_text_is_not_code(tmp_path):
    """Blanking must never be applied to import/export parsing (the specifier
    lives in the quotes) and a name only mentioned in a comment is not a call."""
    _w(tmp_path, "index.html",
       '<!doctype html><script type="module" src="main.js"></script>')
    _w(tmp_path, "a.js", "export function real() {}\n")
    _w(tmp_path, "main.js",
       "// TODO import { real } from './a.js'; and call ghost();\n"
       "import { real } from './a.js';   // real() is called below\n"
       "export function use() { return real(); }\n")
    assert spec.verify_frontend_contracts(tmp_path, None) == []


# --------------------------------------- webcheck parity (notices side) -----
# The same two contracts are published as notices by `pedacito webcheck`. The
# vocabulary is owned by webcheck and shared by the gate: one fixture, both
# readers, the same names.

def test_webcheck_and_the_gate_flag_the_same_names(tmp_path):
    _w(tmp_path, "index.html",
       '<!doctype html><script type="module" src="main.js"></script>')
    _w(tmp_path, "waves.js", "export const WAYPOINTS = [1];\n")
    _w(tmp_path, "main.js",
       "import { WAYPOINTS } from './waves.js';\n"
       "import { WAYPOINTS } from './waves.js';\n"
       "unreachableFn(WAYPOINTS);\n")
    errs = spec.verify_frontend_contracts(tmp_path, None)
    found = webcheck.run(tmp_path)
    assert any("WAYPOINTS" in e and "already" in e for e in errs), errs
    assert any("WAYPOINTS" in f.message for f in found), found
    assert any("unreachableFn" in f.message for f in found), found
    gate_names = {n for n in ("WAYPOINTS", "unreachableFn")
                  if any(n in e for e in errs)}
    notice_names = {n for n in ("WAYPOINTS", "unreachableFn")
                    if any(n in f.message for f in found)}
    assert gate_names == notice_names == {"WAYPOINTS", "unreachableFn"}


def test_the_gate_stays_quiet_where_the_blanker_once_was_not(tmp_path):
    """The gate reads the grammar, so template nesting and destructuring were
    never its problem -- pin that, because the two readers must not disagree
    in the direction that vetoes working code."""
    _w(tmp_path, "index.html",
       '<!doctype html><script type="module" src="app.js"></script>')
    _w(tmp_path, "lit.js",
       "export function html(strings, ...values) { return strings; }\n")
    _w(tmp_path, "app.js",
       "import { html } from './lit.js';\n"
       "const [open, setOpen] = [false, () => {}];\n"
       "export function panel() {\n"
       "  setOpen(!open);\n"
       "  return html`<p>when a server is reachable (falls back)</p>`;\n"
       "}\n")
    assert spec.verify_frontend_contracts(tmp_path, None) == []
    assert webcheck.run(tmp_path) == []


def test_webcheck_is_quiet_on_a_project_that_links(tmp_path):
    """Ground truth from the bug hunt: a finished browser game must produce
    zero findings from either reader (a false positive costs a repair round
    on working code)."""
    _w(tmp_path, "index.html",
       '<!doctype html><html><body><canvas id="c"></canvas>'
       '<script type="module" src="src/main.js"></script></body></html>')
    _w(tmp_path, "src/audio.js",
       "export function playMortar() { return 1; }\n")
    _w(tmp_path, "src/main.js",
       "import { playMortar } from './audio.js';\n"
       "const ctx = document.getElementById('c').getContext('2d');\n"
       "function frame(t) { playMortar(); return t; }\n"
       "requestAnimationFrame(frame);\n")
    assert spec.verify_frontend_contracts(tmp_path, None) == []
    assert webcheck.run(tmp_path) == []


# ------------------------------------ BUG-013: stack-aware EDGE CASES -------
# The generic fallback line asked a browser game to "fail with a clear
# traceback", which no browser prints. The fallback now reads the stack the
# file list names.

def _fallback_edge_cases(root, files):
    got = spec.refine_prompt("build the thing", root, None, None)
    return got["refined"]


def test_js_plan_gets_browser_edge_cases(tmp_path):
    _w(tmp_path, "src/game.js", "export function step() {}\n")
    refined = _fallback_edge_cases(tmp_path, ["src/game.js"])
    tail = refined[refined.index("EDGE CASES"):]
    tail = tail[:tail.index("VERIFICATION")]
    assert "DOM node" in tail and "devicePixelRatio" in tail
    assert "traceback" not in tail          # nothing here throws one


def test_python_plan_keeps_the_traceback_case(tmp_path):
    _w(tmp_path, "svc.py", "def run(cfg): return cfg\n")
    refined = _fallback_edge_cases(tmp_path, ["svc.py"])
    tail = refined[refined.index("EDGE CASES"):][:
        refined[refined.index("EDGE CASES"):].index("VERIFICATION")]
    assert "traceback" in tail


def test_mixed_plan_says_both(tmp_path):
    _w(tmp_path, "svc.py", "def run(cfg): return cfg\n")
    _w(tmp_path, "ui.js", "export function draw() {}\n")
    refined = _fallback_edge_cases(tmp_path, ["svc.py", "ui.js"])
    tail = refined[refined.index("EDGE CASES"):][:
        refined[refined.index("EDGE CASES"):].index("VERIFICATION")]
    assert "DOM node" in tail and "traceback" in tail


# ----------------------------------------- BUG-005: mid-token requirement ---

def test_clip_clause_lands_on_a_word_boundary():
    long = ("The restart button must map to the state machine's reset "
            "transition and clear the score before the first frame draws "
            "anything at all on the canvas")
    assert len(long) > 140
    out = verify._clip_clause(long)
    assert len(out) <= 144
    assert out.endswith(" ...")
    assert " " in out                      # never one cut-off token
    assert not out[:-4].endswith("ma")     # ... the old `p[:140]` landed here
    assert " ".join(out[:-4].split()) in long       # still a real prefix
    short = "Writes the header row."
    assert verify._clip_clause(short) == short      # untouched


def test_clip_clause_survives_a_first_word_longer_than_the_limit():
    """A single long token (a path, a minified identifier) has no boundary to
    fall back on: keep the head and mark it rather than returning nothing."""
    out = verify._clip_clause("x" * 300, limit=140)
    assert out == "x" * 140 + " ..."


def test_requirement_ledger_uses_the_word_boundary_clip():
    """The clip helper existed, but the ledger still cut with `p[:140]`, so a
    plan step inherited a requirement title ending mid-word."""
    clause = ("The restart button must map to the state machine's reset "
              "transition and clear the score before the first frame draws "
              "anything at all on the canvas")
    reqs = verify.requirement_ledger(clause)
    assert reqs, "the clause must survive splitting"
    text = reqs[0][0]
    assert text == verify._clip_clause(clause)
    assert text.endswith(" ...")
    assert " ".join(text[:-4].split()) in clause   # a real prefix, whole words


# --------------------------------------------------- BUG-019: run_test scope --

def test_run_test_refuses_a_standalone_script_with_guidance(tmp_path):
    """`pedacito new` writes `def check(...)` verification scripts. Pointed at
    one, pytest exits 5 -- and one campaign renamed working functions to
    `test_*` to chase the code."""
    _w(tmp_path, "pyproject.toml", '[project]\nname = "d"\nversion = "0"\n')
    _w(tmp_path, "scripts/check_build.py",
       "def check():\n    print('ok')\n\n\nif __name__ == '__main__':\n"
       "    check()\n")
    cfg = _cfg(sandbox=False)
    cfg._root = str(tmp_path)
    idx = types.SimpleNamespace(root=tmp_path)
    out = tools.dispatch(idx, cfg, {"action": "run_test",
                                    "target": "scripts/check_build.py"})
    assert out.startswith("ERROR:")
    assert "collects nothing" in out and "exits 5" in out
    assert "Do NOT rename" in out
    assert "python scripts/check_build.py" in out     # the working way


def test_run_test_labels_an_empty_collection(tmp_path):
    """When pytest really does exit 5 the verdict line says NOT COLLECTED and
    explains that nothing about the code under test was measured. A directory
    with no test files is the shape the pre-scan cannot pre-judge."""
    _w(tmp_path, "pyproject.toml", '[project]\nname = "d"\nversion = "0"\n')
    _w(tmp_path, "src/__init__.py", "")
    _w(tmp_path, "src/app.py", "def helper():\n    return 1\n")
    cfg = _cfg(sandbox=False)
    cfg._root = str(tmp_path)
    idx = types.SimpleNamespace(root=tmp_path)
    out = tools.dispatch(idx, cfg, {"action": "run_test", "target": "src"})
    assert "EXIT: 5 (NOT COLLECTED)" in out, out
    assert "EXIT CODE MEANS" in out and "NOTHING WAS COLLECTED" in out


# ------------------------------------------------ BUG-008: whole-file anchor --

BIG = ("".join(f"def keep_{i}():\n    return {i}\n" for i in range(1, 41))
       + "class MortarShell:\n    def fire(self):\n        return 2\n"
       + "".join(f"def tail_{i}():\n    return {i}\n" for i in range(1, 21)))


def test_whole_file_lines_anchor_rejected_for_a_symbol_target(tmp_path):
    """The loop asked for one definition. `LINES 1-N` of an N-line file is the
    whole file wearing a range anchor: the answer token cap cuts it mid-body
    and the class is gone. Reject with the form that fits."""
    assert len(BIG.splitlines()) == 123
    agent, _cfg_, _idx = _files(tmp_path, {"mortar.py": BIG})
    reply = ("--- FILE: mortar.py LINES 1-123\n"
             "<<<<<<< SEARCH\n=======\n"
             "def keep_1():\n    return 1\n"
             ">>>>>>> REPLACE\n")
    results, _findings = agent._check_incremental_hunk(
        "fix fire()", "", "mortar.py", "MortarShell.fire", reply)
    assert not results[0].ok
    assert "WHOLE file" in results[0].message
    assert "SYMBOL mortar.py :: MortarShell.fire" in results[0].message


def test_a_small_range_on_a_big_file_is_allowed(tmp_path):
    agent, _cfg_, _idx = _files(tmp_path, {"mortar.py": BIG})
    reply = ("--- FILE: mortar.py LINES 81-83\n"
             "<<<<<<< SEARCH\n=======\n"
             "class MortarShell:\n    def fire(self):\n"
             "        return 99\n>>>>>>> REPLACE\n")
    results, findings = agent._check_incremental_hunk(
        "fix fire()", "", "mortar.py", "MortarShell.fire", reply)
    assert results[0].ok, results[0].message


def test_a_short_file_may_still_be_replaced_wholesale(tmp_path):
    """The guard is about cutting a large file mid-definition. A 6-line file
    fits in any answer, so `LINES 1-6` stays legal."""
    agent, _cfg_, _idx = _files(tmp_path, {"tiny.py": "def a():\n    return 1\n"
                                                      "def b():\n    return 2\n"
                                                      "def c():\n    return 3\n"})
    reply = ("--- FILE: tiny.py LINES 1-6\n<<<<<<< SEARCH\n=======\n"
             "def a():\n    return 10\n"
             "def b():\n    return 2\n"
             "def c():\n    return 3\n>>>>>>> REPLACE\n")
    results, _f = agent._check_incremental_hunk("t", "", "tiny.py", "a", reply)
    assert results[0].ok, results[0].message


# ------------------------------------------------------ BUG-007: mixed state --

def test_partial_incremental_batch_says_what_is_on_disk(tmp_path):
    """Ten of eleven hunks landed. Nothing rolls back (the landed hunks were
    each verified), but a summary that reads "applied" hides that the feature
    is half-implemented -- and the run that lost `MortarShell` did exactly
    that."""
    events = []
    agent, cfg, idx = _files(tmp_path, {"a.py": "def f():\n    return 1\n"})
    agent._emit = lambda name, **kw: events.append((name, kw))
    good = ("--- FILE: a.py SYMBOL f\n<<<<<<< SEARCH\n"
            "def f():\n    return 1\n=======\n"
            "def f():\n    return 2\n>>>>>>> REPLACE\n")
    replies = [good, ""]

    class Scripted:
        stats = {}
        stream_truncated = False

        def chat(self, *a, **k):
            return replies.pop(0) if replies else ""

    agent.client = Scripted()
    intents = [{"file": "a.py", "symbol": "f", "intended_change": "x"},
               {"file": "ghost.py", "symbol": "gone", "intended_change": "y"}]
    results, _reply, _findings = agent.apply_edit_intents(
        "task", "", intents, verbose=False)
    assert any(r.ok for r in results) and any(not r.ok for r in results), results
    kinds = [e[0] for e in events]
    assert "incremental_partial" in kinds, events
    part = [e[1] for e in events if e[0] == "incremental_partial"][0]
    assert part["applied"] == 1 and part["failed"] == 1
    assert "ghost.py" in part["files"]


def test_partial_batch_overflow_counts_files_not_blocks(tmp_path, capsys):
    """The banner lists at most six failed FILES; the '(+N more)' overflow
    must count the files it did not list, not the failed blocks. Eight ghost
    targets in eight files used to print '(+8 more)' after six names."""
    agent, cfg, idx = _files(tmp_path, {"a.py": "def f():\n    return 1\n"})
    good = ("--- FILE: a.py SYMBOL f\n<<<<<<< SEARCH\n"
            "def f():\n    return 1\n=======\n"
            "def f():\n    return 2\n>>>>>>> REPLACE\n")
    replies = [good]

    class Scripted:
        stats = {}
        stream_truncated = False

        def chat(self, *a, **k):
            return replies.pop(0) if replies else ""

    agent.client = Scripted()
    intents = [{"file": "a.py", "symbol": "f", "intended_change": "x"}]
    intents += [{"file": f"ghost{i}.py", "symbol": "gone",
                 "intended_change": "y"} for i in range(8)]
    results, _reply, _findings = agent.apply_edit_intents(
        "task", "", intents, verbose=True)
    assert sum(1 for r in results if r.ok) == 1
    assert sum(1 for r in results if not r.ok) == 8
    err = capsys.readouterr().err
    assert "PARTIALLY APPLIED" in err
    assert "(+2 more)" in err, err
    assert "(+8 more)" not in err, err


# ------------------------------------------------------------- BUG-017: why --

def test_repair_collapse_event_carries_the_reason(tmp_path):
    """The event said 12 messages were dropped; nothing said why, so the only
    way to learn was to re-read the transcript."""
    events = []
    agent, cfg, idx = _files(tmp_path, {"a.py": "def f():\n    return 1\n"})
    agent._emit = lambda name, **kw: events.append((name, kw))
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "bad draft"},
            {"role": "user", "content": "diagnostic"}]
    removed = agent._collapse_repair_turns(
        msgs, 2, receipt="f() updated",
        reason="gate: NameError in a.py:1 undefined name 'g'")
    assert removed == 2
    assert len(msgs) == 3                     # back to the snapshot + receipt
    hits = [e for e in events if e[0] == "repair_collapse"]
    assert hits and "NameError" in hits[0][1]["reason"], events


# -------------------------------------- BUG-023: the anchor that cannot miss --

def test_failed_anchor_retry_offers_the_lines_anchor(tmp_path):
    """A quoted SEARCH that will not match (mojibake, tabs, CRLF) has one
    form left that cannot miss: the range the numbered SOURCE already shows.
    The re-anchor prompt used to offer only "copy the text exactly again"."""
    agent, cfg, idx = _files(tmp_path, {"a.py": "def f():\n    return 1\n"})
    prompts: list[list[dict]] = []
    doomed = ("--- FILE: a.py\n<<<<<<< SEARCH\n"
              "    def f():\n        return 1\n=======\n"
              "def f():\n    return 2\n>>>>>>> REPLACE\n")

    class Scripted:
        stats = {}
        stream_truncated = False

        def chat(self, messages, *a, **k):
            prompts.append(list(messages))
            return ""

    agent.client = Scripted()
    results, _reply = agent.apply(doomed, dry_run=True, repair_rounds=1,
                                  task="change f", gathered="")
    assert len(prompts) == 1, "no re-anchor retry was asked for"
    tail = "\n".join(m["content"] for m in prompts[0])
    assert "could not be applied" in tail          # the failure is quoted back
    assert "SMALL FIXES (LINES ANCHOR)" in tail    # ... and the form that fits
    assert "LINES 40-46" in tail                   # the shape, with numbers
    assert not results[0].ok                       # nothing landed


# ------------------------------------------------------------ BUG-006: gates --
# With the default `allow_shell=False` the whole per-task check chain used to
# be skipped as "no tests wanted", so the syntax / static / frontend /
# contract gates -- every one of them in-process, zero subprocess -- never
# ran. A broken ES module graph sailed through and `ask` reported success.

def test_gates_run_without_a_shell(tmp_path):
    """`verify_tests=False, allow_shell=False` still vetoes: the frontend
    gate needs neither flag, and its finding reaches the caller."""
    from pedacito import loop
    from pedacito.edits import Result
    _w(tmp_path, "index.html",
       '<!doctype html><script type="module" src="main.js"></script>')
    _w(tmp_path, "main.js", "import { a } from './nope.js';\n")
    cfg = _cfg(verify_tests=False, allow_shell=False, static_gate=False,
               frontend_contracts=True, interface_lock=False, repair_rounds=0)
    idx = types.SimpleNamespace(root=tmp_path)
    collected: list = []
    _results, _reply, tests_ok = loop._post_edit_tests(
        tmp_path, cfg, None, idx, "wire the modules",
        [Result(True, "main.js", "applied")], "reply", None, None,
        ["main.js"], collect=collected)
    assert tests_ok is False                # the gate vetoed the step
    assert collected, "the gate chain was skipped with the shell off"
    assert any(f.kind == "frontend" and "nope.js" in f.message
               for f in collected), [f.message for f in collected]


def test_the_contract_gate_reaches_the_caller_too(tmp_path):
    from pedacito import loop
    from pedacito.edits import Result
    _w(tmp_path, "lib.py", "def make(a):\n    return a\n")
    _w(tmp_path, "use.py", "from lib import make\nmake(1, 2, 3)\n")
    cfg = _cfg(verify_tests=False, allow_shell=False, static_gate=False,
               frontend_contracts=False, interface_lock=True,
               repair_rounds=0)
    idx = types.SimpleNamespace(root=tmp_path)
    collected: list = []
    loop._post_edit_tests(
        tmp_path, cfg, None, idx, "change make",
        [Result(True, "use.py", "applied")], "reply", None, None,
        ["use.py", "lib.py"], collect=collected)
    assert any("contract" in (f.kind + f.message) and "make" in f.message
               for f in collected), [f.message for f in collected]


# ---------------------------------------------------------- BUG-016: blockers --

def test_rpc_summary_lists_blockers():
    """`applied: true` with a module graph that cannot link is not a success
    a caller can see. The gate findings now travel in their own field, and
    `applied` keeps meaning what it says."""
    res = types.SimpleNamespace(
        reply="done", applied=True, ok=True, steps=3, tests_ok=None,
        stats={"calls": 1}, results=[types.SimpleNamespace(file="main.js",
                                                           ok=True)],
        findings=[types.SimpleNamespace(kind="frontend", severity="error",
                                        message="ContractError in main.js:1: "
                                                "'./nope.js' does not exist",
                                        file="main.js"),
                  types.SimpleNamespace(kind="webcheck", severity="notice",
                                        message="harmless style notice",
                                        file="main.js")])
    s = rpc.RpcServer._summary(res)
    assert s["applied"] is True
    assert len(s["blockers"]) == 1
    assert "frontend" in s["blockers"][0] and "does not exist" in s["blockers"][0]


def test_rpc_summary_blockers_empty_when_clean():
    res = types.SimpleNamespace(reply="x", applied=True, ok=True, steps=1,
                                stats={}, results=[], findings=[])
    assert rpc.RpcServer._summary(res)["blockers"] == []


# ---------------------------------------------- BUG-002/003: the build gate --
# `pedacito new --apply` wrote each file in its own call and never ran the
# cross-file contract gate over the result: the tree passed, the page 404'd.

def _html_plan(name="dash"):
    return Plan(name, "a browser game", files=[
        PlannedFile("index.html", "shell with #c"),
        PlannedFile("src/main.js", "game loop"),
    ], spec=Spec("a browser game"))


class _MismatchedPage:
    """Builds index.html + main.js whose DOM queries do not match, and has
    nothing to say when asked to repair them: each file is individually
    valid, so nothing in the per-file check could see the pair."""

    stats = {"calls": 0, "completion_tokens": 0, "seconds": 0}

    def __init__(self, queried="#board"):
        self.built: list[str] = []
        self.queried = queried

    def chat_json(self, messages, schema, max_tokens=0, **kw):
        return {"project_name": "dash", "summary": "a canvas game",
                "files": [{"path": "index.html", "purpose": "shell"},
                          {"path": "main.js", "purpose": "game loop"}]}

    def _answer(self, rel: str, body: str) -> str:
        self.built.append(rel)
        return (f"--- FILE: {rel}\n<<<<<<< SEARCH\n=======\n"
                f"{body}\n>>>>>>> REPLACE\n")

    def chat(self, messages, max_tokens=0, **kw):
        if len(self.built) == 0:
            return self._answer("index.html",
                                "<!doctype html><canvas id='stage'></canvas>"
                                "<script src='main.js'></script>")
        if len(self.built) == 1:
            return self._answer("main.js",
                                f"document.getElementById('{self.queried}');")
        return ""          # the repair rounds get nothing useful


def test_web_contract_errors_sees_a_page_that_cannot_run(tmp_path):
    _w(tmp_path, "index.html",
       '<!doctype html><canvas id="stage"></canvas>'
       '<script type="module" src="src/main.js"></script>')
    _w(tmp_path, "src/main.js",
       "document.getElementById('board');\n"
       "import { spawn } from './missing.js';\n")
    errs = scaffold.web_contract_errors(tmp_path)
    assert any("board" in e for e in errs), errs
    assert any("missing.js" in e for e in errs), errs


def test_web_contract_errors_is_quiet_without_web_files(tmp_path):
    _w(tmp_path, "main.py", "print(1)\n")
    assert scaffold.web_contract_errors(tmp_path) == []


def test_new_apply_exits_nonzero_when_the_page_cannot_link(tmp_path, capsys):
    """The build's own gate, with a client that cannot repair it: the exit
    code and stderr must say the page is broken, not `done`."""
    client = _MismatchedPage(queried="board")
    code = scaffold.run_new(tmp_path, _cfg(static_gate=True,
                                           frontend_contracts=True),
                            client, scaffold.Spec("a canvas game"),
                            apply=True, review=False, plan_only=False,
                            verbose=False)
    err = capsys.readouterr().err
    assert client.built[:2] == ["index.html", "main.js"]   # it did build
    assert code == 1, (code, err)
    assert "BUILD INCOMPLETE" in err
    assert "board" in err


def test_new_apply_is_green_when_the_page_links(tmp_path, capsys):
    """Same shape, one id that exists: the gate must not bark, and the exit
    code stays 0. This is the false-positive bar -- a working build."""
    client = _MismatchedPage(queried="stage")
    code = scaffold.run_new(tmp_path, _cfg(static_gate=True,
                                           frontend_contracts=True),
                            client, scaffold.Spec("a canvas game"),
                            apply=True, review=False, plan_only=False,
                            verbose=False)
    err = capsys.readouterr().err
    assert client.built[:2] == ["index.html", "main.js"]
    assert code == 0, (code, err)
    assert "BUILD INCOMPLETE" not in err


# ---------------------------------------------------- BUG-020: README --run ---

def test_readme_run_command_serves_a_html_only_project():
    """ES modules are refused over `file://`; a README that says
    `python main.py` (a file the project does not contain) leaves whoever
    opens it with a CORS error and no hint."""
    html_only = Plan("dash", "canvas game", files=[
        PlannedFile("index.html", "shell"),
        PlannedFile("src/main.js", "loop"),
    ], spec=Spec("canvas game"))
    cmd = scaffold._readme_run_command(html_only)
    assert "http.server" in cmd and "http://localhost:8000/index.html" in cmd
    assert "python main.py" not in cmd


def test_readme_run_command_keeps_the_entrypoint_for_a_python_project():
    py = Plan("md2html", "cli", files=[PlannedFile("main.py", "CLI entry")],
              spec=Spec("markdown to html"))
    assert scaffold._readme_run_command(py) == "python main.py"


def test_readme_run_command_covers_both_stacks(tmp_path):
    both = Plan("svc", "server with a page", files=[
        PlannedFile("main.py", "server"),
        PlannedFile("index.html", "page"),
    ], spec=Spec("server with a page"))
    cmd = scaffold._readme_run_command(both)
    assert cmd.startswith("python main.py")
    assert "http://localhost:8000/index.html" in cmd
    assert "file://" not in cmd or "CORS" in cmd


def test_skeleton_writes_the_serve_hint(tmp_path):
    root = tmp_path / "dash"
    scaffold.apply_skeleton(root, _html_plan("dash"), dry_run=False,
                            verbose=False)
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "python -m http.server" in readme
    assert "python main.py" not in readme


# --------------------------------------------------------- BUG-001: shadowed --

def test_xdg_shadow_note_names_the_ignored_file(tmp_path, monkeypatch):
    home, xdg = tmp_path / "home", tmp_path / "xdg"
    (home / ".config" / "pedacito").mkdir(parents=True)
    dead = home / ".config" / "pedacito" / "config.toml"
    dead.write_text("[profiles.a]\nmodel = 'm'\n", encoding="utf-8")
    xdg.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.setattr(config_mod, "USER_CONFIG",
                        xdg / "pedacito" / "config.toml")
    note = config_mod.xdg_shadow_note()
    assert str(dead) in note and "NOT read" in note
    assert "XDG_CONFIG_HOME" in note

    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setattr(config_mod, "USER_CONFIG",
                        home / ".config" / "pedacito" / "config.toml")
    assert config_mod.xdg_shadow_note() == ""      # nothing shadowed


def test_profiles_empty_note_explains_a_shadowed_config(tmp_path, monkeypatch):
    home, xdg = tmp_path / "home", tmp_path / "xdg"
    (home / ".config" / "pedacito").mkdir(parents=True)
    (home / ".config" / "pedacito" / "config.toml").write_text(
        "[profiles.a]\nmodel = 'm'\n", encoding="utf-8")
    xdg.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.setattr(config_mod, "USER_CONFIG",
                        xdg / "pedacito" / "config.toml")
    monkeypatch.setattr(cli.config, "USER_CONFIG",
                        xdg / "pedacito" / "config.toml")
    out = cli._profiles_empty_note()
    assert "NOT read" in out and str(home / ".config") in out

    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setattr(cli.config, "USER_CONFIG",
                        home / ".config" / "pedacito" / "config.toml")
    assert "No profiles defined" in cli._profiles_empty_note()
