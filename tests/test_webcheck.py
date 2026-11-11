"""P1: `pedacito.webcheck` -- static contract checks for web projects.

Every test builds a tiny project on disk and asserts on findings: the
checks are regex-level, so the fixtures are the specification of exactly
which patterns are (and are not) flagged.
"""
from __future__ import annotations

import types
from pathlib import Path

import pytest

from pedacito import cli, loop, webcheck, workspace
from pedacito.config import Config


def _w(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _hit(found, rel=None, needle=None, line=None) -> bool:
    return any((rel is None or f.rel == rel)
               and (needle is None or needle in f.message)
               and (line is None or f.line == line)
               for f in found)


# ----------------------------------------------------------------- DOM ids
def test_dom_id_refs_are_checked_against_every_page(tmp_path):
    _w(tmp_path, "index.html",
       "<body>\n"
       '<div id="app"></div>\n'
       '<script src="game.js"></script>\n'
       "</body>\n")
    _w(tmp_path, "game.js",
       'const el = document.getElementById("app");\n'
       'const hud = document.getElementById("hud");\n'
       'const row = document.getElementById("row-" + i);\n'
       'const q = document.querySelector("#score");\n'
       'const ok = document.querySelector("#app .bar");\n'
       'const dyn = document.querySelector("#" + kind);\n')
    found = webcheck.run(tmp_path)
    assert _hit(found, "game.js", "'hud'", line=2)
    assert _hit(found, "game.js", "'score'", line=4)
    # existing ids, a concatenated prefix, a complex selector whose leading
    # id exists, and a fully dynamic selector all stay quiet
    assert not _hit(found, needle="'app'")
    assert not _hit(found, needle="row-")
    assert len(found) == 2


def test_ids_built_by_js_count_as_existing(tmp_path):
    _w(tmp_path, "index.html", '<script src="game.js"></script>\n')
    _w(tmp_path, "game.js",
       'const d = document.createElement("div");\n'
       'd.id = "hud";\n'
       'document.body.appendChild(d);\n'
       'const el = document.getElementById("hud");\n')
    assert webcheck.run(tmp_path) == []


def test_a_guarded_lookup_is_not_reported_as_a_throw(tmp_path):
    """Every missing id used to be reported as "returns null and the next
    property access throws" -- including `getElementById('x') || fallback`,
    `?.` and `if (!el) return`, where nothing throws. The finding stays (the
    id is still rendered nowhere, so the branch it feeds is dead), but the
    consequence is worded for the call site in front of the model: a notice
    caught overstating once gets ignored everywhere else."""
    _w(tmp_path, "index.html",
       '<body>\n<div id="app"></div>\n<script src="game.js"></script>\n'
       "</body>\n")
    _w(tmp_path, "game.js",
       'const bare = document.getElementById("bare");\n'      # 1 unguarded
       'bare.textContent = "x";\n'                            # 2
       'const fb = document.getElementById("fb") || '
       'document.querySelector(".fallback");\n'                # 3 fallback
       'const opt = document.getElementById("opt")?.textContent;\n'   # 4 ?.
       'const guarded = document.getElementById("guarded");\n'  # 5 assigned
       'if (!guarded) { return; }\n'                          # 6 ...then checked
       'if (document.getElementById("cond")) { go(); }\n'      # 7 condition
       'const q = document.querySelector("#qs");\n'            # 8 unguarded
       'q.style.color = "red";\n'
       'const t = document.querySelector("#tern");\n'          # 10 assigned
       'const label = t ? t.textContent : "";\n')              # 11 ...ternary
    found = webcheck.run(tmp_path)
    for name in ("bare", "fb", "opt", "guarded", "cond", "qs", "tern"):
        assert _hit(found, "game.js", f"'{name}'"), name
    assert len(found) == 7
    for line in (1, 8):                      # nothing handles the null
        assert _hit(found, "game.js", "next property access throws",
                    line=line), line
    for line in (3, 4, 5, 7, 10):            # a fallback, ?., or a null check
        assert _hit(found, "game.js", "handles the null", line=line), line
        assert not _hit(found, "game.js", "next property access throws",
                        line=line), line


# ---------------------------------------------------- ESM behind a classic
def test_esm_file_needs_type_module_on_the_script_tag(tmp_path):
    _w(tmp_path, "util.js", "export const x = 1;\n")
    _w(tmp_path, "game.js",
       'import { x } from "./util.js";\nconsole.log(x);\n')
    _w(tmp_path, "index.html",
       '<head>\n<script src="game.js"></script>\n</head>\n')
    found = webcheck.run(tmp_path)
    assert _hit(found, "game.js", 'type="module"', line=1)
    assert any(f.peer == "index.html" for f in found)
    # with the attribute set, the contract is fine
    _w(tmp_path, "index.html",
       '<head>\n<script type="module" src="game.js"></script>\n</head>\n')
    assert webcheck.run(tmp_path) == []


def test_inline_module_syntax_in_a_classic_script(tmp_path):
    _w(tmp_path, "index.html",
       "<body>\n<script>\nimport x from './y.js';\nx();\n</script>\n"
       "</body>\n")
    found = webcheck.run(tmp_path)
    assert _hit(found, "index.html", 'type="module"', line=3)


# ---------------------------------------------------- duplicate globals
def test_duplicate_globals_only_conflict_on_a_shared_page(tmp_path):
    _w(tmp_path, "a.js", "function draw() {}\ndraw();\n")
    _w(tmp_path, "b.js", "const draw = () => {};\n")
    _w(tmp_path, "index.html",
       '<script src="a.js"></script>\n<script src="b.js"></script>\n')
    found = webcheck.run(tmp_path)
    assert _hit(found, "b.js", "'draw'", line=1)
    assert any(f.rel == "b.js" and "a.js:1" in f.message for f in found)
    assert _hit(found, "b.js", "SyntaxError")     # const redeclaration
    # on separate pages there is no shared global scope, so no conflict
    _w(tmp_path, "index.html", '<script src="a.js"></script>\n')
    _w(tmp_path, "other.html", '<script src="b.js"></script>\n')
    assert not _hit(webcheck.run(tmp_path), needle="already declared")


def test_modules_are_exempt_from_the_globals_check(tmp_path):
    _w(tmp_path, "a.js", "function draw() {}\nexport { draw };\n")
    _w(tmp_path, "b.js", "function draw() {}\nexport { draw };\n")
    _w(tmp_path, "index.html",
       '<script type="module" src="a.js"></script>\n'
       '<script type="module" src="b.js"></script>\n')
    assert not _hit(webcheck.run(tmp_path), needle="already declared")


# -------------------------------------------------------- dangling src
def test_dangling_script_src_is_a_finding(tmp_path):
    _w(tmp_path, "index.html", '<script src="js/missing.js"></script>\n')
    found = webcheck.run(tmp_path)
    assert _hit(found, "index.html", "does not exist", line=1)
    # an external CDN URL is nobody's business
    _w(tmp_path, "index.html",
       '<script src="https://cdn.example/lib.js"></script>\n')
    assert webcheck.run(tmp_path) == []


# --------------------------------------------------------- bare imports
def test_bare_imports_need_a_package_json_entry(tmp_path):
    _w(tmp_path, "package.json",
       '{"dependencies": {"leftpad": "^1.0.0"},'
       ' "devDependencies": {"mocha": "^10"}}')
    _w(tmp_path, "tests/helpers.js",
       'import leftpad from "leftpad";\n'        # declared
       'import assert from "node:assert";\n'     # builtin prefix
       'import fs from "fs";\n'                  # builtin
       'import sinon from "sinon";\n'            # NOT declared
       'const q = require("qs");\n'              # NOT declared
       'const m = require("mocha");\n'           # declared
       'import rel from "./local.js";\n'         # relative
       'import t from "@org/tools";\n')          # scoped, NOT declared
    found = webcheck.run(tmp_path)
    assert _hit(found, "tests/helpers.js", "'sinon'", line=4)
    assert _hit(found, "tests/helpers.js", "'qs'", line=5)
    assert _hit(found, "tests/helpers.js", "'@org/tools'", line=8)
    assert len(found) == 3


def test_bare_import_check_needs_a_package_json(tmp_path):
    _w(tmp_path, "a.js", 'import sinon from "sinon";\n')
    assert webcheck.run(tmp_path) == []


# -------------------------------------------------------- member calls
def test_member_calls_checked_against_the_single_definer(tmp_path):
    _w(tmp_path, "game.js",
       "const Game = {\n"
       "  start() { this.tick(); },\n"
       "  tick() {},\n"
       "};\n"
       "window.Game = Game;\n")
    _w(tmp_path, "index.html",
       '<script src="game.js"></script>\n'
       "<script>\n"
       "Game.start();\n"
       "Game.pasue();\n"
       "</script>\n"
       '<button onclick="Game.reset()">go</button>\n')
    found = webcheck.run(tmp_path)
    assert _hit(found, "index.html", "pasue", line=4)
    assert _hit(found, "index.html", "reset", line=6)
    assert not _hit(found, needle="start")
    assert any(f.peer == "game.js" for f in found)


def test_member_check_skips_ambiguous_globals(tmp_path):
    _w(tmp_path, "a.js", "const Game = { start() {} };\n")
    _w(tmp_path, "b.js", "const Game = { pasue() {} };\n")
    _w(tmp_path, "index.html",
       '<script src="a.js"></script>\n<script src="b.js"></script>\n'
       "<script>Game.whatever();</script>\n")
    # two definers: no way to know which one the page means at runtime
    assert not _hit(webcheck.run(tmp_path), needle="appears nowhere")


# ------------------------------------------------------- files filter
def test_files_filter_keeps_either_side_of_a_contract(tmp_path):
    _w(tmp_path, "index.html", '<script src="game.js"></script>\n')
    _w(tmp_path, "game.js", 'document.getElementById("hud");\n')
    all_found = webcheck.run(tmp_path)
    assert len(all_found) == 1 and all_found[0].rel == "game.js"
    assert webcheck.run(tmp_path, files=["game.js"]) == all_found
    # editing the PAGE that loads the script is editing the same contract
    assert webcheck.run(tmp_path, files=["index.html"]) == all_found
    assert webcheck.run(tmp_path, files=["unrelated.py"]) == []


# ------------------------------------------------------------ notice()
def test_notice_prints_warnings_and_never_raises(tmp_path, capsys,
                                                 monkeypatch):
    _w(tmp_path, "index.html", '<script src="game.js"></script>\n')
    _w(tmp_path, "game.js", 'document.getElementById("hud");\n')
    assert webcheck.notice(tmp_path) == 1
    err = capsys.readouterr().err
    assert "[webcheck]" in err and "hud" in err

    def boom(*a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(webcheck, "run", boom)
    assert webcheck.notice(tmp_path) == 0


# ------------------------------------------------------------------- CLI
def test_cli_webcheck_reports_and_gates(tmp_path, capsys):
    _w(tmp_path, "index.html", '<script src="game.js"></script>\n')
    _w(tmp_path, "game.js", 'document.getElementById("hud");\n')
    assert cli.main(["webcheck", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "game.js:1:" in out and "hud" in out
    _w(tmp_path, "index.html",
       '<div id="hud"></div>\n<script src="game.js"></script>\n')
    assert cli.main(["webcheck", str(tmp_path)]) == 0
    assert "no web contract problems" in capsys.readouterr().out


# ----------------------------------------------------- auto-loop wiring
def _cfg():
    c = Config()
    c.log_sessions = False
    return c


def test_run_loop_notices_webcheck_findings(tmp_path, monkeypatch, capsys):
    """A task that touched web files gets the contract lint as a notice
    board (stderr + WARN in the run log) -- never as a task failure."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal", ["Add the score display"])
    idx = types.SimpleNamespace(root=tmp_path)
    ok_block = types.SimpleNamespace(ok=True, file="index.html",
                                     message="applied")
    tr = loop.TaskResult([ok_block], "done", True, ["index.html"], idx)
    monkeypatch.setattr(loop, "run_task", lambda *a, **k: tr)
    fake = [webcheck.Finding("game.js", 4, "boom", peer="index.html")]
    seen_files = []
    monkeypatch.setattr(
        webcheck, "run",
        lambda root, files=None: seen_files.append(files) or fake)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
    assert code == 0                       # notice only: the task stays done
    assert seen_files == [["index.html"]]  # scoped to the changed files
    err = capsys.readouterr().err
    assert "[webcheck]" in err and "boom" in err
    assert "webcheck:" in workspace.run_log_path(tmp_path).read_text()
    assert "- [x] Add the score display" in plan.read_text()


def test_run_loop_skips_webcheck_for_non_web_changes(tmp_path, monkeypatch):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal", ["Fix the parser"])
    idx = types.SimpleNamespace(root=tmp_path)
    ok_block = types.SimpleNamespace(ok=True, file="main.py",
                                     message="applied")
    tr = loop.TaskResult([ok_block], "done", True, ["main.py"], idx)
    monkeypatch.setattr(loop, "run_task", lambda *a, **k: tr)
    monkeypatch.setattr(webcheck, "run",
                        lambda *a, **k: pytest.fail("must not run"))
    assert loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True) == 0


# ------------------------------------------------ blanks: what is not code --
# Checks 6 and 7 read names out of a blanked copy of the file, so every false
# positive they could produce is a bug in the blanker or in the binding pool.
# All three shapes below came from running the checker over real code.

def test_nested_template_literal_text_is_not_code(tmp_path):
    """Lit-html nests a template inside an interpolation. A string regex stops
    at the first inner backtick and hands the closing markup back as code --
    which reads as `reachable (falls back`, a call to nothing."""
    _w(tmp_path, "index.html",
       '<!doctype html><script type="module" src="app.js"></script>')
    _w(tmp_path, "app.js",
       "import { html } from './lit.js';\n"
       "export function panel(props) {\n"
       "  return html`<div class=${props.cls}>\n"
       "    ${props.open ? html`<span onClick=${() => props.toggle()}>x</span>`"
       " : null}\n"
       "    Use it when a server is reachable (falls back to the\n"
       "    built-in reference automatically)</label>`;\n"
       "}\n")
    found = webcheck.run(tmp_path)
    assert not _hit(found, needle="reachable"), [str(f) for f in found]
    assert not _hit(found, needle="built"), [str(f) for f in found]


def test_destructured_bindings_are_bound_names(tmp_path):
    """`const [open, setOpen] = useState()` binds both names. Reading the
    payload as one token reports `setOpen()` as an unreachable call."""
    _w(tmp_path, "index.html",
       '<!doctype html><script type="module" src="app.js"></script>')
    _w(tmp_path, "app.js",
       "import { useState } from './react.js';\n"
       "const [open, setOpen] = useState(false);\n"
       "const { draw, fill: doFill } = canvas;\n"
       "export function use() { setOpen(!open); draw(); doFill(); }\n")
    found = webcheck.run(tmp_path)
    for name in ("setOpen", "draw", "doFill"):
        assert not _hit(found, needle=name), [str(f) for f in found]


def test_a_quote_in_a_regex_or_comment_does_not_swallow_the_file(tmp_path):
    """/['"]/ and // don't must not open a string: once they do, the rest of
    the file is blanked and a real ReferenceError goes unreported."""
    _w(tmp_path, "index.html",
       '<!doctype html><script type="module" src="app.js"></script>')
    _w(tmp_path, "app.js",
       "const QUOTE = /['\"]/;\n"                # a character class of quotes
       "// don't treat this as a string either\n"
       "export function go() { return spawnEnemy(3); }\n")
    found = webcheck.run(tmp_path)
    assert _hit(found, rel="app.js", needle="spawnEnemy"), \
        [str(f) for f in found]


def test_the_blanker_keeps_offsets_and_line_numbers(tmp_path):
    """Findings are routed to a repair prompt by file:line; blanking that
    shifts a line sends the repair to the wrong place."""
    _w(tmp_path, "index.html",
       '<!doctype html><script type="module" src="app.js"></script>')
    src = ("import { a } from './x.js';\n"
           "const text = `\n  multi\n  line\n`;\n"
           "export function go() { return nope(); }\n")
    _w(tmp_path, "app.js", src)
    assert len(webcheck._js_code(src)) == len(src)
    assert webcheck._js_code(src).count("\n") == src.count("\n")
    found = webcheck.run(tmp_path)
    assert _hit(found, rel="app.js", needle="nope", line=6), \
        [str(f) for f in found]
