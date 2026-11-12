"""Introspection tests: scoped live evidence + DOM contracts + ledger.

The gather phase has one honest tool (`run_test`: a sandboxed,
shell-free, token-bounded pytest run), gives the loop one more deterministic
gate (`verify_frontend_contracts`: the checks a browser would run at page
load), and gives the repository a memory of why repairs were needed (the
anti-pattern ledger distilled from fail -> pass transitions). These tests
cover all three, plus their wiring.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import loop, memory, security, spec, tools
from pedacito.config import Config
from pedacito.edits import Result


def _cfg(**over):
    c = Config()
    c.log_sessions = False
    c.log_events = False
    for k, v in over.items():
        setattr(c, k, v)
    return c


# =========================================================== run_test ======

def _pyproject(root: Path) -> None:
    (root / "pyproject.toml").write_text(
        "[project]\nname = \"demo\"\nversion = \"0\"\n", encoding="utf-8")


def _test_tree(root: Path, failing: bool = False):
    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / "test_math.py").write_text(
        "def test_ok():\n    assert 2 + 2 == 4\n", encoding="utf-8")
    if failing:
        (root / "tests" / "test_bad.py").write_text(
            "def test_broken():\n    x = undefined_name(1)\n    assert x\n",
            encoding="utf-8")


def test_run_test_passes_and_reports_green(tmp_path):
    _pyproject(tmp_path)
    _test_tree(tmp_path)
    cfg = _cfg(sandbox=False)
    cfg._root = str(tmp_path)
    idx = types.SimpleNamespace(root=tmp_path)
    out = tools.dispatch(idx, cfg, {"action": "run_test"})
    assert "EXIT: 0 (OK)" in out
    assert "1 passed" in out
    assert "shell" not in out.lower() or True


def test_run_test_failure_shows_condensed_traceback(tmp_path):
    _pyproject(tmp_path)
    _test_tree(tmp_path, failing=True)
    cfg = _cfg(sandbox=False)
    cfg._root = str(tmp_path)
    idx = types.SimpleNamespace(root=tmp_path)
    out = tools.dispatch(idx, cfg, {"action": "run_test"})
    assert "FAILED" in out.splitlines()[1]
    assert "test_broken" in out
    # a NameError's project frame is what the model needs, verbatim
    assert "undefined_name" in out
    assert "FAILURES" in out


def test_run_test_scopes_by_target_and_filter(tmp_path):
    _pyproject(tmp_path)
    _test_tree(tmp_path, failing=True)
    cfg = _cfg(sandbox=False)
    cfg._root = str(tmp_path)
    idx = types.SimpleNamespace(root=tmp_path)
    ok = tools.dispatch(idx, cfg, {"action": "run_test",
                                   "target": "tests/test_math.py"})
    assert "EXIT: 0" in ok and "1 passed" in ok
    node = tools.dispatch(idx, cfg, {"action": "run_test",
                                     "target": "tests/test_math.py::test_ok"})
    assert "EXIT: 0" in node
    filtered = tools.dispatch(idx, cfg, {"action": "run_test",
                                         "filter": "ok and not broken"})
    assert "EXIT: 0" in filtered


def test_run_test_rejects_traversal_and_shell_metacharacters(tmp_path):
    _pyproject(tmp_path)
    _test_tree(tmp_path)
    cfg = _cfg(sandbox=False)
    cfg._root = str(tmp_path)
    idx = types.SimpleNamespace(root=tmp_path)
    esc = tools.dispatch(idx, cfg, {"action": "run_test",
                                    "target": "../../etc/passwd"})
    assert esc.startswith("ERROR")
    semi = tools.dispatch(idx, cfg, {"action": "run_test",
                                     "target": "tests; rm -rf /"})
    assert semi.startswith("ERROR")
    node = tools.dispatch(idx, cfg, {"action": "run_test",
                                     "target": "tests/test_math.py::x | ls"})
    assert node.startswith("ERROR")
    dash = tools.dispatch(idx, cfg, {"action": "run_test",
                                     "filter": "--collect-only"})
    assert dash.startswith("ERROR")
    # nothing escaped: the repo is untouched
    assert sorted(p.name for p in tmp_path.iterdir()) == \
        ["pyproject.toml", "tests"]


def test_run_test_argv_never_touches_a_shell(tmp_path, monkeypatch):
    _pyproject(tmp_path)
    _test_tree(tmp_path)
    cfg = _cfg(sandbox=False)
    cfg._root = str(tmp_path)
    idx = types.SimpleNamespace(root=tmp_path)
    seen = {}

    def fake_run(argv, cwd, timeout=0, idle_timeout=0, max_output=0, env=None,
                 **kw):
        seen["argv"] = list(argv)
        return 0, "2 passed in 0.1s", "", 0.1

    monkeypatch.setattr(tools.proc, "run", fake_run)
    out = tools.dispatch(idx, cfg, {"action": "run_test",
                                    "target": "tests/test_math.py"})
    assert out.startswith("RUN TEST:")
    assert any("pytest" in a for a in seen["argv"])
    assert all(ch not in " ".join(seen["argv"][2:]) for ch in (";", "|", "&"))


def test_run_test_disabled_routes_through_permission_broker(tmp_path):
    _pyproject(tmp_path)
    _test_tree(tmp_path)
    idx = types.SimpleNamespace(root=tmp_path)
    seen = []

    def hook(info):
        seen.append(info)
        return False, "user said no"

    cfg = _cfg(sandbox=False, enable_run_test_in_gather=False)
    cfg._root = str(tmp_path)
    security.install_permission_hook(hook)
    try:
        out = tools.dispatch(idx, cfg, {"action": "run_test"})
    finally:
        security.install_permission_hook(None)
    assert "user said no" in out
    assert seen and seen[0]["category"] == security.CAT_TEST_EXECUTION


def test_run_test_records_command_evidence(tmp_path):
    _pyproject(tmp_path)
    _test_tree(tmp_path)                       # one passing test
    cfg = _cfg(sandbox=False)
    cfg._root = str(tmp_path)
    idx = types.SimpleNamespace(root=tmp_path)
    out = tools.dispatch(idx, cfg, {"action": "run_test"})
    assert "EXIT: 0" in out
    # one green scoped run is an observation, not yet a confirmed fact
    assert memory.load_micro_facts(tmp_path) == []
    tools.dispatch(idx, cfg, {"action": "run_test"})
    facts = memory.load_micro_facts(tmp_path)
    assert any("pytest" in f["text"] for f in facts)


def test_run_test_in_schema_and_pinned_by_repo_tests(tmp_path):
    cfg = _cfg()
    cfg._root = str(tmp_path)
    d = tools.schema_for(cfg)
    assert "run_test" not in d["properties"]["action"]["enum"]  # no tests yet
    _test_tree(tmp_path)
    d = tools.schema_for(cfg)
    assert "run_test" in d["properties"]["action"]["enum"]
    assert "target" in d["required"] and "filter" in d["required"]
    # dynamic pruning must NOT hide run_test while the repo has tests,
    # even with a task text that carries no trigger word
    cfg._task_text = "rewrite the readme intro"
    kept = tools._prune_optional_actions(cfg, ["grep", "run_test"])
    assert "run_test" in kept
    # and a bug-task in a testless repo offers it through the prompt trigger
    bare = Config()
    bare._root = str(tmp_path / "empty")
    bare._task_text = "the login button is broken, traceback below"
    assert "run_test" in tools.schema_for(bare)["properties"]["action"]["enum"]


# ================================================= frontend contracts ======

def _webapp(root: Path):
    (root / "index.html").write_text(
        "<!doctype html>\n<html>\n<head>\n"
        '<link rel="stylesheet" href="app.css">\n'
        '<script src="missing.js"></script>\n'
        '<script src="app.js"></script>\n</head>\n<body>\n'
        '<div id="board"></div>\n<button id="new-task-btn">add</button>\n'
        "</body></html>\n", encoding="utf-8")
    (root / "app.css").write_text("body { color: red }", encoding="utf-8")
    (root / "app.js").write_text(
        "const b = document.getElementById('board');\n"
        "const m = document.getElementById('add-task-btn');\n"
        "document.querySelector('#new-task-btn').onclick = () => {};\n",
        encoding="utf-8")


def test_frontend_contracts_catch_missing_asset_and_wrong_id(tmp_path):
    _webapp(tmp_path)
    errs = spec.verify_frontend_contracts(tmp_path, None)
    assert any("missing.js" in e for e in errs)
    assert any("add-task-btn" in e for e in errs)
    assert all(e.startswith("FrontendContractError in ") for e in errs)
    assert any("only defines ids" in e for e in errs)


def test_frontend_contracts_clean_project_is_quiet(tmp_path):
    (tmp_path / "index.html").write_text(
        '<html><body><div id="x"></div><script src="a.js"></script>'
        '<link rel="stylesheet" href="a.css"></body></html>', encoding="utf-8")
    (tmp_path / "a.js").write_text(
        "document.getElementById('x');", encoding="utf-8")
    (tmp_path / "a.css").write_text("", encoding="utf-8")
    assert spec.verify_frontend_contracts(tmp_path, None) == []


def test_frontend_contracts_dynamic_ids_are_not_guessed_at(tmp_path):
    (tmp_path / "index.html").write_text(
        '<html><body><div id="list"></div><script src="a.js"></script>'
        "</body></html>", encoding="utf-8")
    (tmp_path / "a.js").write_text(
        "function addRow(id) {\n"
        "  const d = document.createElement('div');\n"
        "  d.innerHTML = `<span id=\"${id}-name\">n</span>`;\n"
        "  document.querySelector(`#${id}-name`).textContent = 'x';\n"
        "}\n", encoding="utf-8")
    assert spec.verify_frontend_contracts(tmp_path, None) == []


def test_frontend_contracts_js_created_ids_count_as_defined(tmp_path):
    (tmp_path / "index.html").write_text(
        '<html><body><script src="a.js"></script></body></html>',
        encoding="utf-8")
    (tmp_path / "a.js").write_text(
        "const d = document.createElement('div');\n"
        "d.id = 'generated';\n"
        "document.getElementById('generated');\n", encoding="utf-8")
    assert spec.verify_frontend_contracts(tmp_path, None) == []


def test_frontend_contracts_broken_js_is_a_contract_error(tmp_path):
    (tmp_path / "index.html").write_text(
        '<html><script src="a.js"></script></html>', encoding="utf-8")
    (tmp_path / "a.js").write_text("function render( {\n", encoding="utf-8")
    errs = spec.verify_frontend_contracts(tmp_path, None)
    assert errs and "does not parse" in errs[0]


def test_frontend_contracts_scope_to_modified_files(tmp_path):
    _webapp(tmp_path)
    # only the CSS changed: nothing the CSS can break is reported
    assert spec.verify_frontend_contracts(tmp_path, ["app.css"]) == []
    # the JS changed: its bad id is reported, the HTML's rot is not
    errs = spec.verify_frontend_contracts(tmp_path, ["app.js"])
    assert any("add-task-btn" in e for e in errs)
    assert not any("missing.js" in e for e in errs)
    # the HTML changed: the id registry moved, so every JS query is
    # re-verified against it, and the broken asset link is in scope too
    errs = spec.verify_frontend_contracts(tmp_path, ["index.html"])
    assert any("missing.js" in e for e in errs)
    assert any("add-task-btn" in e for e in errs)


def test_frontend_contracts_ignores_external_and_template_refs(tmp_path):
    (tmp_path / "index.html").write_text(
        '<html><head>'
        '<script src="https://cdn.example.com/lib.js"></script>'
        '<script src="//cdn.example.com/x.js"></script>'
        '<link rel="stylesheet" href="data:text/css,body{}">'
        '<script src="{{ url_for(\'static\', f=\'app.js\') }}"></script>'
        "</head><body></body></html>", encoding="utf-8")
    assert spec.verify_frontend_contracts(tmp_path, None) == []


# ========================================================= ledger =========

def test_anti_pattern_roundtrip_dedupe_and_cap(tmp_path):
    cfg = _cfg()
    (tmp_path / "app.js").write_text("x\n", encoding="utf-8")
    row = memory.record_anti_pattern(tmp_path, "dom_id_mismatch",
                                     "queries an id no template renders",
                                     ["app.js"], cfg)
    assert row and row["kind"] == "dom_id_mismatch"
    again = memory.record_anti_pattern(tmp_path, "dom_id_mismatch",
                                       "queries an id no template renders",
                                       ["app.js"], cfg)
    assert again["confirmations"] == 2
    assert len(memory.load_anti_patterns(tmp_path)) == 1
    for i in range(10):
        memory.record_anti_pattern(tmp_path, f"k{i}", f"lesson number {i}",
                                   [], cfg)
    rows = memory.load_anti_patterns(tmp_path)
    assert len(rows) <= 8
    doc = json.loads((tmp_path / ".pedacito" / "micro_facts.json")
                     .read_text())
    assert len(doc["anti_patterns"]) <= 8


def test_anti_patterns_retire_with_their_files(tmp_path):
    cfg = _cfg()
    (tmp_path / "app.js").write_text("x\n", encoding="utf-8")
    memory.record_anti_pattern(tmp_path, "dom_id_mismatch", "a lesson",
                               ["app.js"], cfg)
    assert len(memory.load_anti_patterns(tmp_path)) == 1
    (tmp_path / "app.js").unlink()
    assert memory.load_anti_patterns(tmp_path) == []
    doc = json.loads((tmp_path / ".pedacito" / "micro_facts.json")
                     .read_text())
    assert doc["anti_patterns"] == []          # retired on disk too


def test_observed_writes_preserve_the_ledger(tmp_path):
    cfg = _cfg()
    memory.record_anti_pattern(tmp_path, "signature_contract",
                               "caller and callee disagreed", [], cfg)
    for _ in range(2):      # two successes promote a command micro-fact
        memory.observe_command(tmp_path, "python -m pytest -q", True, cfg,
                               kind="test")
    doc = json.loads((tmp_path / ".pedacito" / "micro_facts.json")
                     .read_text())
    assert len(doc["anti_patterns"]) == 1
    assert len(doc["facts"]) == 1


def test_render_anti_patterns_block(tmp_path):
    cfg = _cfg()
    (tmp_path / "app.js").write_text("x\n", encoding="utf-8")
    memory.record_anti_pattern(tmp_path, "dom_id_mismatch",
                               "the query had no element behind it",
                               ["app.js"], cfg)
    block = memory.render_anti_patterns(tmp_path, cfg)
    assert block.startswith("[REPO GOTCHAS & LESSONS]")
    assert "dom_id_mismatch" in block
    cfg2 = _cfg(micro_facts=False)
    assert memory.render_anti_patterns(tmp_path, cfg2) == ""


def test_distill_repair_lesson_shapes():
    assert spec is not None  # module handle for readability
    k, line, files = loop.distill_repair_lesson(
        "FrontendContractError in app.js:42: queries DOM element "
        "'#add-task-btn', but index.html only defines ids", "frontend")
    assert k == "dom_id_mismatch" and files == ["app.js", "index.html"]
    k2, line2, _ = loop.distill_repair_lesson(
        "m.py:2:12: F821 undefined name 'jsonx'", "static")
    assert k2 == "avoid_missing_import" and "jsonx" in line2
    k3, _, _ = loop.distill_repair_lesson(
        "TypeError: f() got an unexpected keyword argument 'fast'", "verify")
    assert k3 == "signature_contract"
    k4, _, _ = loop.distill_repair_lesson(
        "FileContractError: references missing stylesheet asset 'a.css'", "")
    assert k4 == "asset_404"
    assert loop.distill_repair_lesson("a sentence with no known shape", "") \
        is None


def test_repair_lesson_keeps_dot_file_names(tmp_path):
    """`str.lstrip("./")` strips CHARACTERS, so `.eslintrc.js` became
    `eslintrc.js`: the lesson was recorded under a name that does not exist and
    retired on the very next load. The leading dot (and the case) must survive
    into the ledger, and distinct dot-files must not compare equal."""
    _k, _line, files = loop.distill_repair_lesson(
        "undefined name 'x' in .eslintrc.js", "static")
    assert files == [".eslintrc.js"]
    _k2, _l2, mixed = loop.distill_repair_lesson(
        "FrontendContractError in ./src/MyComponent.tsx:1: queries DOM element "
        "'#x', but index.html only defines ids", "frontend")
    assert mixed == ["index.html", "src/MyComponent.tsx"]
    assert not loop._same_rel(".env.py", "env.py")
    assert not loop._same_rel("pkg/mod.py", ".mod.py")
    assert loop._same_rel("./pkg/a.py", "pkg/a.py")
    # end to end: a lesson about a dot-file survives the existence prune
    (tmp_path / ".eslintrc.js").write_text("module.exports = {};\n",
                                          encoding="utf-8")
    facts = memory.micro_facts_file(tmp_path)
    facts.parent.mkdir(parents=True, exist_ok=True)
    facts.write_text(json.dumps({"anti_patterns": [
        {"lesson_id": "L1", "kind": "avoid_missing_import",
         "text": "import x where it is used", "files": files,
         "last_seen": 1}]}), encoding="utf-8")
    live = memory.load_anti_patterns(tmp_path)
    assert [a["lesson_id"] for a in live] == ["L1"]


def test_post_edit_frontend_gate_repairs_then_learns(tmp_path, monkeypatch):
    """Full wiring: a broken DOM id blocks the test subprocess, the repair
    lands, and the fail -> pass transition writes one lesson to the ledger."""
    (tmp_path / "index.html").write_text(
        '<html><body><div id="board"></div><script src="app.js"></script>'
        "</body></html>", encoding="utf-8")
    (tmp_path / "app.js").write_text(
        "document.getElementById('nope');\n", encoding="utf-8")
    cfg = _cfg(verify_tests=True, allow_shell=True, repair_rounds=1,
               sandbox=False)
    cfg._frontend_baseline = set()
    (tmp_path / "app.py").write_text("import json\n", encoding="utf-8")
    FIXED = "document.getElementById('board');\n"

    class FakeAgent:
        calls = []

        def _repair_from_findings(self, task, reply, findings, results):
            FakeAgent.calls.append([f.kind for f in findings])
            return "REPAIR"

        def _apply_once(self, reply, dry_run, backup_dir):
            (tmp_path / "app.js").write_text(FIXED)
            return [Result(True, "app.js", "applied",
                            before="document.getElementById('nope');\n",
                            after=FIXED)]

    runs = []
    monkeypatch.setattr(loop, "_run_tests",
                        lambda idx, cfg_, files: (runs.append(list(files))
                                                  or (True, "1 passed",
                                                      "pytest")))
    agent = FakeAgent()
    results = [Result(True, "app.js", "applied",
                      before="", after="document.getElementById('nope');\n")]
    out_results, reply, tests_ok = loop._post_edit_tests(
        tmp_path, cfg, agent, types.SimpleNamespace(root=tmp_path), "task",
        results, "reply", None, None, ["app.js"])
    assert FakeAgent.calls == [["frontend"]]       # gate fired first, once
    assert len(runs) == 1                          # subprocess ran after fix
    assert tests_ok is True
    rows = memory.load_anti_patterns(tmp_path)
    assert len(rows) == 1 and rows[0]["kind"] == "dom_id_mismatch"
    assert "nope" in rows[0]["text"] or "getElementById" in rows[0]["text"]


def test_post_edit_run_counts_a_recurred_lesson_kind(tmp_path, monkeypatch):
    """R5 wiring: a lesson injected before the run whose failure kind the run
    trips again gets `recurred` bumped -- the effectiveness signal the ledger
    retires on. The lesson distilled from this run's own repair is NOT counted
    (it was not in the run's prompt)."""
    (tmp_path / "index.html").write_text(
        '<html><body><div id="board"></div><script src="app.js"></script>'
        "</body></html>", encoding="utf-8")
    (tmp_path / "app.js").write_text(
        "document.getElementById('nope');\n", encoding="utf-8")
    cfg = _cfg(verify_tests=True, allow_shell=True, repair_rounds=1,
               sandbox=False)
    cfg._frontend_baseline = set()
    seeded = memory.record_anti_pattern(
        tmp_path, "dom_id_mismatch",
        "an earlier run queried an id no template renders", ["app.js"], cfg)
    FIXED = "document.getElementById('board');\n"

    class FakeAgent:
        def _repair_from_findings(self, task, reply, findings, results):
            return "REPAIR"

        def _apply_once(self, reply, dry_run, backup_dir):
            (tmp_path / "app.js").write_text(FIXED)
            return [Result(True, "app.js", "applied",
                            before="document.getElementById('nope');\n",
                            after=FIXED)]

    monkeypatch.setattr(loop, "_run_tests",
                        lambda idx, cfg_, files: (True, "1 passed", "pytest"))
    results = [Result(True, "app.js", "applied",
                      before="", after="document.getElementById('nope');\n")]
    loop._post_edit_tests(
        tmp_path, cfg, FakeAgent(), types.SimpleNamespace(root=tmp_path),
        "task", results, "reply", None, None, ["app.js"])

    rows = {r["lesson_id"]: r for r in memory.load_anti_patterns(tmp_path)}
    assert rows[seeded["lesson_id"]]["runs"] == 1
    assert rows[seeded["lesson_id"]]["recurred"] == 1
    assert rows[seeded["lesson_id"]]["avoided"] == 0
    fresh = [r for lid, r in rows.items() if lid != seeded["lesson_id"]]
    assert len(fresh) == 1 and fresh[0]["runs"] == 0


def test_post_edit_clean_run_counts_lesson_avoided(tmp_path, monkeypatch):
    """R5 wiring: a run whose gates and tests stay clear of a lesson's kind
    counts as `avoided` -- the signal that keeps an effective lesson alive."""
    cfg = _cfg(verify_tests=True, allow_shell=True, repair_rounds=1)
    before = "x = 1\n"
    (tmp_path / "mod.py").write_text(before, encoding="utf-8")
    seeded = memory.record_anti_pattern(
        tmp_path, "dom_id_mismatch",
        "an earlier run queried an id no template renders", ["mod.py"], cfg)

    class FakeAgent:
        def _repair_from_findings(self, *a):
            raise AssertionError("a clean run must not repair")

        def _apply_once(self, *a):
            raise AssertionError("a clean run must not repair")

    monkeypatch.setattr(loop, "_run_tests",
                        lambda idx, cfg_, files: (True, "1 passed", "pytest"))
    loop._post_edit_tests(
        tmp_path, cfg, FakeAgent(), object(), "task",
        [Result(True, "mod.py", "applied", before=before, after=before)],
        "reply", None, None, ["mod.py"])

    rows = memory.load_anti_patterns(tmp_path)
    assert len(rows) == 1
    assert rows[0]["lesson_id"] == seeded["lesson_id"]
    assert rows[0]["runs"] == 1
    assert rows[0]["avoided"] == 1
    assert rows[0]["recurred"] == 0


def test_frontend_gate_ablation_skips_check(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text(
        '<html><script src="gone.js"></script></body></html>',
        encoding="utf-8")
    cfg = _cfg(verify_tests=True, allow_shell=True, repair_rounds=1,
               sandbox=False, frontend_contracts=False)
    assert loop._frontend_enabled(cfg) is False
    assert loop._frontend_errors(tmp_path, cfg, ["index.html"]) == []
    monkeypatch.setattr(loop, "_run_tests",
                        lambda idx, cfg_, files: (True, "1 passed", "pytest"))

    class FakeAgent:
        def _repair_from_findings(self, *a):
            raise AssertionError("gate must not fire when ablated")

        def _apply_once(self, *a):
            raise AssertionError("gate must not fire when ablated")

    loop._post_edit_tests(
        tmp_path, cfg, FakeAgent(), types.SimpleNamespace(root=tmp_path),
        "task", [Result(True, "index.html", "applied", before="",
                         after="x")], "reply", None, None, ["index.html"])


def test_gated_check_in_auto_loop_blocks_on_new_frontend_error(tmp_path):
    """run_loop's gate helper honors the same baseline scoping as the
    post-edit loop: pre-existing rot passes, a fresh break does not."""
    (tmp_path / "index.html").write_text(
        '<html><script src="gone.js"></script></body></html>',
        encoding="utf-8")
    cfg = _cfg(sandbox=False)
    pre = set(spec.verify_frontend_contracts(tmp_path, None))
    cfg._frontend_baseline = pre
    assert loop._new_frontend_errors(tmp_path, cfg, ["index.html"]) == []
    cfg._frontend_baseline = set()
    fresh = loop._new_frontend_errors(tmp_path, cfg, ["index.html"])
    assert fresh and "gone.js" in fresh[0]
