"""Regression tests for the Bug Hunt #9 self-healing session (BUG-901..).

BUG-901: the project starter (`pedacito config --init --path
    <project>/.pedacito.toml`) is generated from the user `TEMPLATE`, and
    `project_template()` only comments the four USER-scope keys. Every
    execution-capable setting therefore stayed ACTIVE at its default:
    `allow_write`, `allow_shell`, `allow_serve`, `allow_screenshots`,
    `allow_browser`, `mcp_servers`, `allow_lsp`, `auto_verify`. Two
    consequences, both live in Bug Hunt #9 Phase 0:
      1. Every `pedacito config` / run in the workspace printed
         "[config] ignoring allow_write, allow_shell, ..." -- a trust
         warning about a file that asked for nothing.
      2. Once trusted, those explicit `false` values layer OVER the user
         config and silently revoke capabilities the user had granted.

BUG-902: `scaffold.HARNESS_OWNED` excludes `.gitignore`, but
    `workspace.ensure` creates and appends to it on every run, and
    `config --init` calls it. A directory therefore stopped counting as
    fresh the moment Pedacito touched it, and `new` refused its own
    `.gitignore` with "is not empty".

BUG-903: `_run_complex` called `plan_tasks(..., include_tests=design)`, so
    every planned test file was dropped from the autonomous loop unless
    `--design` was passed. `Plan.tests` documents that they are built AFTER
    the sources, and nothing after the loop built them: `new --complex`
    reported "10 task(s) done, 0 failed" over two untouched test stubs.
"""

import tomllib
import types
from pathlib import Path

import pytest

from pedacito import config
from pedacito import loop as loop_mod
from pedacito import scaffold
from pedacito import testsel

EXEC_STARTER_KEYS = ("allow_write", "allow_shell", "allow_serve",
                     "allow_screenshots", "allow_browser", "mcp_servers",
                     "allow_lsp", "auto_verify")


@pytest.fixture()
def iso_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".config" / "pedacito").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    user_cfg = config._user_config_dir() / "config.toml"
    monkeypatch.setattr(config, "USER_CONFIG", user_cfg)
    return home


def _active_keys(text: str) -> set[str]:
    """Every key a TOML reader would actually apply, across all tables."""
    data = tomllib.loads(text)
    keys: set[str] = set()
    for name, value in data.items():
        if name.startswith("profiles"):
            continue
        if isinstance(value, dict):
            keys.update(value)
        else:
            keys.add(name)
    return keys


def test_project_starter_does_not_activate_execution_fields(iso_home):
    text = config.project_template()
    active = _active_keys(text)
    offenders = sorted(active.intersection(EXEC_STARTER_KEYS))
    assert offenders == [], (
        "the project starter actively sets execution fields, which both warns "
        f"on every load and can revoke user capabilities once trusted: {offenders}")


def test_project_starter_keeps_them_visible_as_guidance(iso_home):
    text = config.project_template()
    for key in EXEC_STARTER_KEYS:
        assert key in text, f"{key} disappeared from the starter entirely"
        assert any(line.lstrip().startswith("# " + key)
                   for line in text.splitlines()), (
            f"{key} is neither active nor shown as a commented example")


def test_untrusted_project_starter_loads_without_a_trust_warning(
        iso_home, tmp_path, capsys):
    proj = tmp_path / "proj"
    proj.mkdir()
    config.write_template(proj / ".pedacito.toml")
    capsys.readouterr()
    config._warned_untrusted.clear()
    config.load(root=proj, require_hardware=False)
    err = capsys.readouterr().err
    assert "ignoring" not in err, (
        "a freshly written project starter should not trigger the trust "
        f"warning; stderr was: {err}")


def test_trusted_project_starter_does_not_downgrade_user_capabilities(
        iso_home, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    config.USER_CONFIG.write_text(
        "[universal]\n"
        "allow_shell = true\n"
        "allow_write = true\n"
        "allow_serve = true\n"
        "allow_browser = true\n",
        encoding="utf-8",
    )
    cfg_path = proj / ".pedacito.toml"
    config.write_template(cfg_path)
    config.trust_project(cfg_path)
    cfg = config.load(root=proj, require_hardware=False)
    assert cfg.allow_shell is True
    assert cfg.allow_write is True
    assert cfg.allow_serve is True
    assert cfg.allow_browser is True


# --------------------------------------------------------------------------
# BUG-902: `.gitignore` is harness-managed, so it cannot count as content
# --------------------------------------------------------------------------

def _empty_plan(spec, client, cfg, root, *a, **k):
    return scaffold.Plan("Emberfall", "a tiny game")


def test_gitignore_alone_does_not_make_a_directory_occupied(
        tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(scaffold, "build_plan", _empty_plan)
    root = tmp_path / "clean"
    root.mkdir()
    (root / ".git").mkdir()
    (root / ".pedacito.toml").write_text("# project config\n", encoding="utf-8")
    (root / ".gitignore").write_text(".pedacito/\n__pycache__/\n",
                                     encoding="utf-8")
    capsys.readouterr()
    code = scaffold.run_new(root, config.load(require_hardware=False), None,
                            scaffold.Spec("a tiny game"), verbose=False)
    assert code != 2
    assert "is not empty" not in capsys.readouterr().err


def test_gitignore_does_not_hide_a_real_project_file(
        tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(scaffold, "build_plan", _empty_plan)
    root = tmp_path / "occupied"
    root.mkdir()
    (root / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
    (root / "main.py").write_text("print('hello')\n", encoding="utf-8")
    capsys.readouterr()
    code = scaffold.run_new(root, config.load(require_hardware=False), None,
                            scaffold.Spec("a tiny game"), verbose=False)
    assert code == 2
    err = capsys.readouterr().err
    assert "is not empty" in err
    assert "main.py" in err


# --------------------------------------------------------------------------
# BUG-903: complex mode must implement the planned test files too
# --------------------------------------------------------------------------

def test_complex_mode_plans_the_test_files(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    plan = scaffold.Plan("Emberfall", "a tiny game")
    plan.files = [scaffold.PlannedFile(p) for p in
                  ("index.html", "src/app.js", "tests/engine.test.js")]
    seen = {}

    def fake_run_loop(root_, cfg, client, idx, plan_file, **kw):
        seen["plan"] = Path(plan_file).read_text(encoding="utf-8")
        return 0

    monkeypatch.setattr(loop_mod, "run_loop", fake_run_loop)
    monkeypatch.setattr(scaffold, "_web_contract_gate", lambda *a, **k: [])
    monkeypatch.setattr(scaffold, "verify_and_fix", lambda *a, **k: (True, ""))
    cfg = config.load(require_hardware=False)
    cfg.log_sessions = False
    cfg.backup = False
    code = scaffold._run_complex(root, cfg, None, None, plan,
                                 scaffold.Spec("a tiny game"), verbose=False)
    assert code == 0
    text = seen["plan"]
    assert "Implement src/app.js" in text
    assert "Implement tests/engine.test.js" in text, (
        "complex mode dropped the planned test file from the task list; "
        f"plan was:\n{text}")


def test_complex_mode_still_runs_tests_after_sources(tmp_path, monkeypatch):
    root = tmp_path / "proj2"
    root.mkdir()
    plan = scaffold.Plan("Emberfall", "a tiny game")
    plan.files = [scaffold.PlannedFile(p) for p in
                  ("tests/engine.test.js", "src/app.js")]
    seen = {}

    def fake_run_loop(root_, cfg, client, idx, plan_file, **kw):
        seen["plan"] = Path(plan_file).read_text(encoding="utf-8")
        return 0

    monkeypatch.setattr(loop_mod, "run_loop", fake_run_loop)
    monkeypatch.setattr(scaffold, "_web_contract_gate", lambda *a, **k: [])
    monkeypatch.setattr(scaffold, "verify_and_fix", lambda *a, **k: (True, ""))
    cfg = config.load(require_hardware=False)
    cfg.log_sessions = False
    cfg.backup = False
    scaffold._run_complex(root, cfg, None, None, plan,
                          scaffold.Spec("a tiny game"), verbose=False)
    text = seen["plan"]
    assert text.index("Implement src/app.js") < \
        text.index("Implement tests/engine.test.js")


# --------------------------------------------------------------------------
# BUG-904: the headless DOM check must bound a page that never yields
# --------------------------------------------------------------------------

def test_domcheck_returns_a_result_for_a_quiet_page(tmp_path):
    pytest.importorskip("playwright.sync_api")
    from pedacito import domcheck
    ok, hint = domcheck.available()
    if not ok:
        pytest.skip(hint)
    (tmp_path / "index.html").write_text(
        "<!doctype html><html><body><div id='x'>Hello</div></body></html>",
        encoding="utf-8")
    res = domcheck.check(tmp_path, timeout=10)
    assert not res.error, res.error
    assert res.snapshot is not None
    assert res.snapshot.get("body_children")


def test_domcheck_bounds_a_page_that_loops_forever(tmp_path):
    import time
    pytest.importorskip("playwright.sync_api")
    from pedacito import domcheck
    ok, hint = domcheck.available()
    if not ok:
        pytest.skip(hint)
    (tmp_path / "index.html").write_text(
        "<!doctype html><html><body><div id='x'>spin</div>"
        "<script>while (true) {}</script></body></html>",
        encoding="utf-8")
    started = time.monotonic()
    res = domcheck.check(tmp_path, timeout=5)
    elapsed = time.monotonic() - started
    assert elapsed < 40, (
        "domcheck.check blocked on a runaway page script; page.evaluate has "
        "no timeout and the check must be bounded")
    assert res.error, "a page that never yields cannot be reported as verified"
    assert "did not finish" in res.error or "looping" in res.error


# --------------------------------------------------------------------------
# BUG-905: a single-parameter arrow callback binds its parameter
# --------------------------------------------------------------------------

def test_webcheck_accepts_a_single_parameter_arrow_callback(tmp_path):
    from pedacito import webcheck
    (tmp_path / "index.html").write_text(
        "<!doctype html><html><body><div id='x'></div>"
        "<script type='module' src='main.js'></script></body></html>",
        encoding="utf-8")
    (tmp_path / "main.js").write_text(
        "const rows = [{ name: 'Shade' }];\n"
        "rows.forEach(m => {\n"
        "  if (m.name.includes('Shade')) {\n"
        "    document.getElementById('x').textContent = m.name;\n"
        "  }\n"
        "});\n",
        encoding="utf-8")
    messages = [f.message for f in webcheck.run(tmp_path)]
    assert not any("calls m()" in m for m in messages), (
        "an arrow callback's own parameter is a binding, not a call to an "
        f"undefined function; findings were: {messages}")


# --------------------------------------------------------------------------
# BUG-906: a Python-only change in a mixed tree must run pytest
# --------------------------------------------------------------------------

def _mixed_project(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "package.json").write_text(
        '{"type": "module", "scripts": {"test": "node --test tests/*.test.js"}}',
        encoding="utf-8")
    (tmp_path / "tests" / "engine.test.js").write_text("// js test\n",
                                                       encoding="utf-8")
    (tmp_path / "tests" / "test_server.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8")
    (tmp_path / "src" / "app.js").write_text("export const a = 1;\n",
                                             encoding="utf-8")


def _record_command(monkeypatch):
    seen = {}

    def fake_run(root, cfg, cmd):
        seen["cmd"] = cmd
        return True, "1 passed"

    monkeypatch.setattr(testsel, "run_command", fake_run)
    monkeypatch.setattr(testsel, "_was_already_failing",
                        lambda root, cmd: False)
    return seen


def test_mixed_project_runs_pytest_for_a_python_change(tmp_path, monkeypatch):
    _mixed_project(tmp_path)
    idx = types.SimpleNamespace(root=tmp_path)
    monkeypatch.setattr(testsel, "detect_test_command",
                        lambda root, cfg=None: "npm test")
    monkeypatch.setattr(testsel, "impacted_tests",
                        lambda i, ch: [("tests/test_server.py",
                                        "changed itself")])
    seen = _record_command(monkeypatch)
    ok, _out, _desc = testsel.run_impacted(
        idx, config.load(require_hardware=False), ["tests/test_server.py"])
    assert ok
    assert "pytest" in seen["cmd"], (
        "a Python-only change must be verified by pytest, not by the JS "
        f"runner; command was {seen['cmd']}")
    assert "test_server.py" in seen["cmd"]


def test_mixed_project_keeps_npm_for_a_js_change(tmp_path, monkeypatch):
    _mixed_project(tmp_path)
    idx = types.SimpleNamespace(root=tmp_path)
    monkeypatch.setattr(testsel, "detect_test_command",
                        lambda root, cfg=None: "npm test")
    monkeypatch.setattr(testsel, "impacted_tests",
                        lambda i, ch: [("tests/engine.test.js",
                                        "changed itself")])
    seen = _record_command(monkeypatch)
    testsel.run_impacted(idx, config.load(require_hardware=False),
                         ["src/app.js"])
    assert seen["cmd"] == "npm test -- tests/engine.test.js"


def test_explicit_test_command_is_never_overridden(tmp_path, monkeypatch):
    _mixed_project(tmp_path)
    idx = types.SimpleNamespace(root=tmp_path)
    monkeypatch.setattr(testsel, "detect_test_command",
                        lambda root, cfg=None: "npm test")
    monkeypatch.setattr(testsel, "impacted_tests",
                        lambda i, ch: [("tests/test_server.py",
                                        "changed itself")])
    seen = _record_command(monkeypatch)
    cfg = config.load(require_hardware=False)
    cfg.test_command = "npm test"
    testsel.run_impacted(idx, cfg, ["tests/test_server.py"])
    assert seen["cmd"].startswith("npm test")


# --------------------------------------------------------------------------
# BUG-907: traceback-relative paths must not block lesson promotion
# --------------------------------------------------------------------------

def test_lesson_scope_survives_traceback_relative_paths():
    from pedacito import memory
    files = ["../src/combat.js", "combat.js", "src/combat.js",
             "src/dungeon.js", "../src/fov.js", "src/main.js"]
    assert memory._lesson_scope(files) == "src"


def test_repeated_lesson_promotes_to_a_path_scoped_rule(tmp_path):
    from pedacito import memory
    cfg = config.load(require_hardware=False)
    text = ("an edit left a JS file unparsable -- a browser aborts the whole "
            "file on one syntax error; re-check the parse before calling a JS "
            "edit done")
    files = ["../src/combat.js", "combat.js", "src/combat.js", "src/main.js"]
    first = memory.record_anti_pattern(tmp_path, "js_syntax", text, files, cfg)
    assert first["confirmations"] == 1
    assert not (tmp_path / ".pedacito" / "rules").exists()
    memory.record_anti_pattern(tmp_path, "js_syntax", text, files, cfg)
    rule = tmp_path / ".pedacito" / "rules" / "auto_src.md"
    assert rule.exists(), (
        "a lesson confirmed twice whose files all resolve under src/ must "
        "promote into .pedacito/rules/auto_src.md")
    body = rule.read_text(encoding="utf-8")
    assert "auto_generated: true" in body
    assert "paths: src/*" in body
    assert "js_syntax" in body
    doc = memory._facts_doc(tmp_path)
    assert not doc["anti_patterns"]
    assert doc["promoted"]


# --------------------------------------------------------------------------
# BUG-908: an optional single positional path must not resolve the root to /
# --------------------------------------------------------------------------

def test_infer_root_accepts_a_single_path_string(tmp_path):
    from pedacito import index as index_mod
    assert index_mod.infer_root(str(tmp_path)) == tmp_path.resolve()


def test_rpc_path_flag_resolves_to_the_given_project(tmp_path):
    from pedacito import cli
    from pedacito import index as index_mod
    args = types.SimpleNamespace(root=None, paths=str(tmp_path))
    assert cli._root(args) == tmp_path.resolve()
    assert index_mod.infer_root(str(tmp_path)) == tmp_path.resolve()


# --------------------------------------------------------------------------
# BUG-909: the post-run receipt must name what actually ran
# --------------------------------------------------------------------------

def test_verify_receipt_names_a_smoke_run(tmp_path):
    (tmp_path / "main.py").write_text("print('ok')\n", encoding="utf-8")
    note, ran = scaffold.verify_receipt(tmp_path, "", True, True, "")
    assert ran
    assert note.startswith("verify: smoke passed")


def test_verify_receipt_does_not_claim_a_pass_when_nothing_ran(tmp_path):
    note, ran = scaffold.verify_receipt(tmp_path, "", True, True, "")
    assert not ran
    assert "not run" in note
    assert "passed" not in note


def test_verify_receipt_names_an_explicit_command(tmp_path):
    note, ran = scaffold.verify_receipt(tmp_path, "pytest -q", False,
                                        True, "")
    assert ran
    assert note.startswith("verify: passed")
    assert "pytest -q" not in note  # the command is named by the caller


# --------------------------------------------------------------------------
# FEAT-901: a read-only task does not need the edit-protocol re-ask
# --------------------------------------------------------------------------

class _RecordingAgent:
    def __init__(self):
        self.answers = 0

    def answer(self, task, src, system=None):
        self.answers += 1
        return "No changes: this task asks for no edits."

    def apply(self, raw, dry_run=True, task=None, gathered="", backup_dir=None):
        return [], raw


def test_read_only_task_skips_the_edit_reask(tmp_path):
    agent = _RecordingAgent()
    out = loop_mod._recover_no_edit_reply(
        tmp_path, agent,
        "Count the exported functions in src/fov.js and report only. "
        "Do not edit any file.",
        "", "src/fov.js has one exported function.", dry_run=False,
        backup_dir=None)
    assert out is None
    assert agent.answers == 0, (
        "a task that forbids edits must not spend a model call re-asking "
        "for edit blocks")


def test_edit_task_still_gets_the_reask(tmp_path):
    agent = _RecordingAgent()
    loop_mod._recover_no_edit_reply(
        tmp_path, agent, "Add a retry to the sync client.", "",
        "I would add a retry.", dry_run=False, backup_dir=None)
    assert agent.answers > 0, "an editing task keeps the recovery round"


def test_report_only_directive_skips_the_reask(tmp_path):
    agent = _RecordingAgent()
    out = loop_mod._recover_no_edit_reply(
        tmp_path, agent, "Report only: list the exported functions.", "",
        "src/fov.js exports one function.", dry_run=False, backup_dir=None)
    assert out is None
    assert agent.answers == 0


def test_and_report_only_skips_the_reask(tmp_path):
    agent = _RecordingAgent()
    out = loop_mod._recover_no_edit_reply(
        tmp_path, agent, "Count the exported functions and report only.",
        "", "There is one exported function.", dry_run=False,
        backup_dir=None)
    assert out is None
    assert agent.answers == 0


def test_edit_task_mentioning_read_only_still_gets_the_reask(tmp_path):
    agent = _RecordingAgent()
    loop_mod._recover_no_edit_reply(
        tmp_path, agent, "Add a read-only mode to the settings page.", "",
        "I would add a read-only mode.", dry_run=False, backup_dir=None)
    assert agent.answers > 0, (
        "an edit task that merely describes a read-only feature must keep "
        "the recovery round; the phrase is not a task-level prohibition")


def test_edit_task_with_a_scoped_without_clause_still_gets_the_reask(tmp_path):
    agent = _RecordingAgent()
    loop_mod._recover_no_edit_reply(
        tmp_path, agent,
        "Add caching to the sync client without editing the database schema.",
        "", "I would add caching.", dry_run=False, backup_dir=None)
    assert agent.answers > 0, (
        "'without editing <one specific thing>' still asks for edits "
        "elsewhere; the recovery round must run")


def test_scoped_prohibition_still_gets_the_reask(tmp_path):
    agent = _RecordingAgent()
    loop_mod._recover_no_edit_reply(
        tmp_path, agent,
        "Fix the parser. Do not edit any file other than src/parser.py.",
        "", "I would fix the parser.", dry_run=False, backup_dir=None)
    assert agent.answers > 0, (
        "a prohibition scoped to 'any file other than X' still asks for "
        "edits to X; the recovery round must run")
