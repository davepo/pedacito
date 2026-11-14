"""Regression tests for the Bug Hunt #8 self-healing session (BUG-801..808).

BUG-801: a `.pedacito.toml` inside an untrusted project could grant
    execution capabilities to itself: `allow_shell` / `allow_write` /
    `allow_browser` / `allow_screenshots` were not in ``config.EXEC_FIELDS``,
    so ``_drop_untrusted`` left them alone and every command the project
    config wanted to run was silently approved.

BUG-802: ``config --init --path <project>/.pedacito.toml`` wrote the *user*
    starter (``default_profile``, placeholder ``base_url``, ``vram_gb``).
    Because a project config layers over the user one, those keys silently
    shadowed a working user profile with a server that does not exist.

BUG-803: the ``config --init`` advice told a project file to fill exactly
    the keys BUG-802 refuses to put there.

BUG-804: ``new`` in a directory holding only harness-owned files
    (``.git``, ``.pedacito*``, the project config) demanded ``--force``,
    refusing its own litter as if it were a project.

BUG-805: ``testsel.narrow_command`` forwarded Python test paths to the
    node/npm runners, which count an unparsable file as one more failing
    test; the impacted subset could never go green (proved live: a green
    JS suite reported "11 still failing" over a .py path).

BUG-806: ``new --complex`` skipped the frontend-contract gate ``new --apply``
    runs, so the autonomous loop could ship a page whose ids and imports do
    not answer, with the tests skipped, and still exit 0.

BUG-807: the coverage follow-up in ``agent._coverage_intents`` turned an
    unaddressed CODE clause into an edit intent against whichever indexed
    file matched its words -- including prose documents that merely QUOTE
    the task text (the plan, a bug report, notes). Live in Bug Hunt #8: a
    ``style.css`` task rewrote a table row of the session's own bug report.

BUG-808: a verification gate that ends a run RED -- DOM contract
    violations standing, the test subprocess never spawned -- was
    answered with EXIT 0 and NO final word in either non-planned face:
    ``ask --apply`` printed "[frontend gate: ... skipped until they are
    fixed]" twice (violations 7, then 8) and exited silently; the
    ``run_loop`` finish line counted the task in "[N done, 0 failed]".
    Only --strict or the RPC blocker told the truth. The fix echoes the
    last gate verdict where the caller reads it: a red-verification note
    on the ask path, and named red tasks in the loop's finish line, the
    way partial applications already are.
"""

import inspect
import types

import pytest

from pedacito import config
from pedacito import scaffold
from pedacito import testsel
from pedacito import verify as verify_mod
from pedacito import webcheck

CAP_KEYS = ("allow_shell", "allow_write", "allow_browser", "allow_screenshots")


@pytest.fixture()
def iso_home(tmp_path, monkeypatch):
    """Point the user layer (config, trust ledger, --init defaults) at a
    private tmp home so no test reads or writes the real one."""
    home = tmp_path / "home"
    (home / ".config" / "pedacito").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    user_cfg = config._user_config_dir() / "config.toml"
    monkeypatch.setattr(config, "USER_CONFIG", user_cfg)
    return home


# --------------------------------------------------------------------------
# BUG-801: capabilities are execution fields; a project file cannot grant them
# --------------------------------------------------------------------------

def test_capability_toggles_are_execution_fields():
    # the one-line root cause: these were missing from EXEC_FIELDS
    for key in CAP_KEYS:
        assert key in config.EXEC_FIELDS, key


def test_untrusted_project_config_cannot_grant_capabilities(tmp_path,
                                                             monkeypatch,
                                                             capsys):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / ".pedacito.toml").write_text(
        "allow_shell = true\nallow_write = true\n"
        "allow_browser = true\nallow_screenshots = true\n",
        encoding="utf-8",
    )
    cfg = config.load(root=proj, require_hardware=False)
    for key in CAP_KEYS:
        assert getattr(cfg, key) is False, key
    err = capsys.readouterr().err
    assert "ignoring" in err and ".pedacito.toml" in err
    # and `pedacito trust` has a file to review: the same one load() read
    assert config.find_project_config(proj) == proj / ".pedacito.toml"


def test_trusted_project_config_grants_capabilities(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(config, "USER_CONFIG",
                        tmp_path / "home" / ".config" / "pedacito" / "config.toml")
    proj = tmp_path / "proj"
    proj.mkdir()
    path = proj / ".pedacito.toml"
    path.write_text("allow_shell = true\nallow_write = true\n", encoding="utf-8")
    config.trust_project(path)
    cfg = config.load(root=proj, require_hardware=False)
    assert cfg.allow_shell is True
    assert cfg.allow_write is True
    # editing the file after approval revokes it again
    path.write_text("allow_shell = false\nallow_write = true\n", encoding="utf-8")
    cfg = config.load(root=proj, require_hardware=False)
    assert cfg.allow_write is False


def test_capability_toggles_in_a_project_profile_are_dropped(tmp_path,
                                                             monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / ".pedacito.toml").write_text(
        "[profiles.team]\nallow_browser = true\nallow_screenshots = true\n",
        encoding="utf-8",
    )
    cfg = config.load(root=proj, require_hardware=False, profile="team")
    assert cfg.allow_browser is False
    assert cfg.allow_screenshots is False


def test_user_config_still_grants_capabilities(iso_home, monkeypatch):
    config.USER_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    config.USER_CONFIG.write_text(
        "allow_shell = true\nallow_browser = true\n", encoding="utf-8",
    )
    cfg = config.load(require_hardware=False)
    assert cfg.allow_shell is True
    assert cfg.allow_browser is True


def test_explicit_config_file_grants_capabilities(tmp_path, monkeypatch):
    # --config <file> is a first-party layer; capabilities must survive
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    explicit = tmp_path / "hunt.toml"
    explicit.write_text("allow_shell = true\nverify_tests = true\n",
                        encoding="utf-8")
    cfg = config.load(explicit=explicit, require_hardware=False)
    assert cfg.allow_shell is True
    assert cfg.verify_tests is True


# --------------------------------------------------------------------------
# BUG-802: config --init --path <project file> must write a PROJECT starter
# --------------------------------------------------------------------------

def test_project_starter_does_not_shadow_the_user_profile(iso_home, monkeypatch):
    config.USER_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    config.USER_CONFIG.write_text(
        'default_profile = "qwen3_8-27b-gsq-rco"\n'
        'base_url = "http://127.0.0.1:11434/v1"\n'
        "vram_gb = 24\n"
        '[profiles.qwen3_8-27b-gsq-rco]\n'
        'model = "qwen3.8-27b:gsq-rco"\n',
        encoding="utf-8",
    )
    proj = iso_home / "proj"
    proj.mkdir()
    text = config.write_template(proj / ".pedacito.toml").read_text(
        encoding="utf-8"
    )
    # not one live user-scope line in the project starter
    for key in ("default_profile", "base_url", "vram_gb", "model"):
        for ln in text.splitlines():
            assert not ln.startswith(key + " ="), ln

    cfg = config.load(root=proj, require_hardware=False)
    assert cfg.base_url == "http://127.0.0.1:11434/v1"
    assert cfg.default_profile == "qwen3_8-27b-gsq-rco"


def test_user_starter_keeps_the_mandatory_keys(iso_home):
    text = config.write_template(config.USER_CONFIG).read_text(encoding="utf-8")
    assert "default_profile = " in text
    assert "base_url = " in text


def test_user_path_is_not_project_scoped(iso_home, tmp_path):
    # the shape of the user dir (~/.config/pedacito or <proj>/.pedacito) must
    # not be confused with a project file: only the user's own config is user
    assert config.is_project_config_path(tmp_path / "proj" / ".pedacito.toml")
    assert config.is_project_config_path(
        tmp_path / "proj" / ".pedacito" / "config.toml")
    assert not config.is_project_config_path(config.USER_CONFIG)


def test_state_config_path_is_also_project_scoped(iso_home):
    target = iso_home / "proj" / ".pedacito" / "config.toml"
    text = config.write_template(target).read_text(encoding="utf-8")
    for ln in text.splitlines():
        assert not ln.startswith("default_profile ="), ln
        assert not ln.startswith("base_url ="), ln


# --------------------------------------------------------------------------
# BUG-803: config --init advice must match the scope it just wrote
# --------------------------------------------------------------------------

def test_cli_init_advice_matches_the_scope_it_wrote(iso_home, capsys):
    from pedacito import cli
    proj = iso_home / "proj"
    proj.mkdir()

    rc = cli.main(["config", "--init", "--path", str(proj / ".pedacito.toml")])
    assert rc == 0
    out = capsys.readouterr().err
    assert "Project scope:" in out
    assert "Mandatory settings:" not in out

    rc = cli.main(["config", "--init"])
    assert rc == 0
    out = capsys.readouterr().err
    assert "Mandatory settings:" in out
    assert "Project scope:" not in out


# --------------------------------------------------------------------------
# BUG-804: new in a directory that holds only harness-owned files
# --------------------------------------------------------------------------

def _empty_plan(spec, client, cfg, root, *a, **k):
    return scaffold.Plan("Pond", "a tiny game")


def test_harness_files_alone_do_not_make_a_directory_occupied(tmp_path,
                                                              monkeypatch,
                                                              capsys):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(scaffold, "build_plan", _empty_plan)
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / ".pedacito.toml").write_text("# from an earlier review round\n",
                                          encoding="utf-8")
    (clean / ".pedacitoignore").write_text("dist/\n", encoding="utf-8")
    (clean / ".git").mkdir()

    capsys.readouterr()
    code = scaffold.run_new(clean, config.load(require_hardware=False), None,
                            scaffold.Spec("a tiny pond game"), verbose=False)
    assert code != 2
    assert "is not empty" not in capsys.readouterr().err


def test_real_project_files_still_refuse_a_new_scaffold(tmp_path, monkeypatch,
                                                        capsys):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / ".pedacito.toml").write_text("# harness file only\n", encoding="utf-8")
    (busy / "index.html").write_text("<h1>real project</h1>\n", encoding="utf-8")

    assert scaffold.is_harness_owned(".pedacito.toml")
    assert not scaffold.is_harness_owned("index.html")
    capsys.readouterr()
    code = scaffold.run_new(busy, config.load(require_hardware=False), None,
                            scaffold.Spec("a tiny pond game"), verbose=False)
    assert code == 2
    assert "is not empty (index.html)" in capsys.readouterr().err
    # the refusal wrote nothing
    assert sorted(p.name for p in busy.iterdir()) == [".pedacito.toml",
                                                      "index.html"]


# --------------------------------------------------------------------------
# BUG-805: the node/npm runners must never be handed a Python test path
# --------------------------------------------------------------------------

def test_npm_narrowing_never_forwards_a_python_test_to_the_js_runner():
    cmd, narrowed = testsel.narrow_command("npm test", ["tests/test_server.py"])
    assert narrowed is False
    assert cmd == "npm test"
    assert "test_server" not in cmd


def test_npm_narrowing_falls_back_to_the_full_suite_when_nothing_is_js():
    cmd, narrowed = testsel.narrow_command("pnpm test --silent",
                                           ["tests/a.py", "server/api.py"])
    assert (cmd, narrowed) == ("pnpm test --silent", False)


def test_npm_narrowing_keeps_the_typescript_and_plain_js_shapes():
    cmd, narrowed = testsel.narrow_command(
        "npm test", ["tests/ui.spec.ts", "tests/test_server.py"])
    assert narrowed and cmd == "npm test -- tests/ui.spec.ts"
    cmd, narrowed = testsel.narrow_command("yarn test", ["ui.test.js", "x.py"])
    assert narrowed and cmd == "yarn test ui.test.js"


def test_node_test_narrowing_drops_python_entries():
    cmd, narrowed = testsel.narrow_command(
        "node --test tests", ["tests/ui.test.js", "tests/test_server.py"])
    assert narrowed and cmd == "node --test tests tests/ui.test.js"


def test_impacted_verdict_never_names_a_command_the_project_cannot_run(
        tmp_path, monkeypatch):
    idx = types.SimpleNamespace(root=tmp_path)
    monkeypatch.setattr(testsel, "detect_test_command",
                        lambda root, cfg=None: "npm test")
    monkeypatch.setattr(testsel, "impacted_tests",
                        lambda i, ch: [("tests/ui.test.js", "imports"),
                                       ("tests/test_server.py", "imports")])
    monkeypatch.setattr(testsel, "_was_already_failing",
                        lambda root, cmd: False)
    seen = {}

    def fake_run(root, cfg, cmd):
        seen["cmd"] = cmd
        return True, "2 passing"

    monkeypatch.setattr(testsel, "run_command", fake_run)
    ok, out, desc = testsel.run_impacted(idx, config.load(require_hardware=False),
                                         ["src/main.js"])
    assert ok
    assert seen["cmd"] == "npm test -- tests/ui.test.js"   # not the .py path
    assert "test_server" not in seen["cmd"]
    assert "impacted tests" in desc


# --------------------------------------------------------------------------
# BUG-806: new --complex must run the same frontend-contract gate --apply does
# --------------------------------------------------------------------------

def _stub_gate_env(monkeypatch, violations):
    """the gate's collaborators: page checks + one repair round that cannot
    clear them + silent notice board"""
    monkeypatch.setattr(scaffold, "web_contract_errors",
                        lambda root: list(violations))
    monkeypatch.setattr(scaffold, "fix_web_contracts",
                        lambda client, cfg, idx, plan, backup_dir, verbose=True:
                        (0, list(violations)))
    monkeypatch.setattr(webcheck, "notice", lambda root: "")


def test_the_gate_reports_violations_and_vetoes(tmp_path, monkeypatch, capsys):
    _stub_gate_env(monkeypatch, ["#btn-start has no handler", "main.js "
                                 "imports ./audio.js which does not export "
                                 "play"])
    clean = tmp_path / "g"
    clean.mkdir()
    errors = scaffold._web_contract_gate(
        clean, config.load(require_hardware=False), None, None,
        scaffold.Plan("Pond", "s"), None, verbose=True)
    assert len(errors) == 2
    err = capsys.readouterr().err
    assert "BUILD INCOMPLETE" in err
    assert "pedacito ask" in err          # and it says how to recover

    _stub_gate_env(monkeypatch, [])
    assert scaffold._web_contract_gate(
        clean, config.load(require_hardware=False), None, None,
        scaffold.Plan("Pond", "s"), None, verbose=False) == []
    assert "BUILD INCOMPLETE" not in capsys.readouterr().err


def test_complex_build_runs_the_gate(tmp_path, monkeypatch, capsys):
    from pedacito import loop
    _stub_gate_env(monkeypatch, [])
    calls = []

    def gate(root, cfg, client, idx, plan, backup_dir, *, verbose=True):
        calls.append(root)
        return ["index.html queries #btn-fire, nothing binds it"]

    monkeypatch.setattr(scaffold, "_web_contract_gate", gate)
    monkeypatch.setattr(loop, "write_plan", lambda *a, **k: None)
    monkeypatch.setattr(loop, "run_loop",
                        lambda *a, **k: 0)           # the loop thinks it won
    clean = tmp_path / "cx"
    clean.mkdir()
    cfg = config.load(require_hardware=False)
    monkeypatch.setattr(cfg, "backup", False, raising=False)
    capsys.readouterr()
    code = scaffold._run_complex(clean, cfg, None, None,
                                 scaffold.Plan("Pond", "s"),
                                 scaffold.Spec("a tiny pond game"),
                                 verbose=False)
    assert calls == [clean], "--complex must reach the shared gate"
    assert code == 1, "a complex build that leaves new DOM violations " \
                      "must not exit 0"

    # a page with no violations lets the loop's own verdict stand
    monkeypatch.setattr(scaffold, "_web_contract_gate",
                        lambda *a, verbose=True, **k: [])
    assert scaffold._run_complex(clean, cfg, None, None,
                                 scaffold.Plan("Pond", "s"),
                                 scaffold.Spec("a tiny pond game"),
                                 verbose=False) == 0


def test_both_build_modes_reach_the_same_gate():
    # one shared function, called from both paths -- not per-mode logic
    # that one mode can forget
    assert "_web_contract_gate(" in inspect.getsource(scaffold.run_new)
    assert "_web_contract_gate(" in inspect.getsource(scaffold._run_complex)
    assert "contract_failed" in inspect.getsource(scaffold.run_new)


# --------------------------------------------------------------------------
# BUG-807: a coverage follow-up must not rewrite prose that merely quotes
# the task
# --------------------------------------------------------------------------

def _chunk(qualname, kind="function", start=1, end=4):
    return types.SimpleNamespace(qualname=qualname, kind=kind, start=start,
                                 end=end, scope_end=end, nlines=end - start + 1,
                                 members=[])


class _FakeIdx:
    """just the surface `_coverage_intents` uses: in_file, find_symbol,
    resolve_file, files"""

    def __init__(self, files, symbols):
        self.files = files
        self.symbols = symbols

    def in_file(self, rel):
        return self.symbols.get(rel, [])

    def find_symbol(self, term):
        return [c for cs in self.symbols.values()
                for c in cs if c.qualname == term]

    def resolve_file(self, name):
        for f in self.files:
            if f == name or f.endswith("/" + name):
                return f
        return None


def _resolve(reqs, files, symbols):
    from pedacito import agent as A
    probe = types.SimpleNamespace(index=_FakeIdx(files, symbols))
    return A.Agent._coverage_intents(probe, reqs)


def test_coverage_followup_does_not_target_a_quoting_document():
    files = ["src/game.py", "NOTES.md"]
    symbols = {"src/game.py": [_chunk("spin")]}

    # the live shape: auto-grep located the clause in the note, because the
    # note quotes the task verbatim -- that is not an implementation site
    clause = ("Neon arcade styling for the cabinet frame and "
              "reduced-motion behavior for spin().")
    req = verify_mod.Requirement(text=clause, terms=["spin", "game.py"])
    req.add_anchor("NOTES.md", 3, 3)
    files_out = [i["file"] for i in _resolve([req], files, symbols)]
    assert "NOTES.md" not in files_out, files_out
    assert "src/game.py" in files_out, files_out   # the clause lands on code


def test_coverage_followup_still_targets_a_named_document():
    files = ["src/game.py", "NOTES.md"]
    symbols = {}
    req = verify_mod.Requirement(
        text="Record the decision in NOTES.md.", terms=["NOTES.md"])
    req.add_anchor("NOTES.md", 1, 3)
    assert [i["file"] for i in _resolve([req], files, symbols)] == ["NOTES.md"]


def test_coverage_followup_pages_are_not_prose():
    # a web project's index.html is a deliverable, not a prose document to be
    # shielded; the shield is exactly {md, rst, txt} and the name test is
    # exactly "the clause writes the file's name (with its extension)"
    from pedacito import agent as A
    assert not A._is_prose_doc("index.html")
    assert not A._is_prose_doc("css/style.css")
    assert A._is_prose_doc("notes.md") and A._is_prose_doc("NOTES.MD")
    assert A._is_prose_doc("pkg/specs.rst") and A._is_prose_doc("a.txt")
    assert not A._clause_names_file(
        "NOTES.md", "Neon arcade styling, reduced-motion behavior for spin()")
    assert A._clause_names_file("NOTES.md", "Record it in NOTES.md today")
    assert A._clause_names_file("docs/NOTES.md", "see docs/NOTES.md")


# --------------------------------------------------------------------------
# BUG-808: the last gate verdict was RED -- the run has to say so
# --------------------------------------------------------------------------

def _fake_tr(**kw):
    defaults = dict(tests_ok=False, tests_preexisting_only=False, findings=[])
    defaults.update(kw)
    return types.SimpleNamespace(**defaults)


def test_red_verdict_note_names_the_blockers_and_stays_honest():
    from pedacito.cli import _red_verdict_note
    F = verify_mod.Finding
    tr = _fake_tr(findings=[
        F("frontend", "error", "",
          "'vitest' is imported but package.json declares no such "
          "dependency"),
        F("frontend", "error", "",
          "src/main.js:124: calls audio() but nothing defines it"),
        F("frontend", "error", "", "third blocker"),
        F("frontend", "error", "", "fourth blocker"),
        F("coverage", "warn", "", "a mere warning"),
    ])
    note = _red_verdict_note(types.SimpleNamespace(strict=False), tr)
    assert "verification: RED" in note
    assert "'vitest' is imported" in note          # blockers named, not just counted
    assert "third blocker" in note                 # three are shown
    assert "fourth blocker" not in note
    assert "(+1 more blocker(s))" in note          # the rest counted
    assert "a mere warning" not in note            # warnings are not blockers
    # the note promises nothing false: exit stays 0 without --strict,
    # so the text itself must say so and point at a repair route
    assert "--strict" in note and "domcheck" in note


def test_red_verdict_note_is_silent_when_it_would_lie():
    from pedacito.cli import _red_verdict_note
    N = types.SimpleNamespace
    # green, untested, pre-existing red, --strict (which already exits 1
    # with its own ERROR line), and no task at all are NOT red verdicts
    assert _red_verdict_note(N(strict=False), _fake_tr(tests_ok=True)) == ""
    assert _red_verdict_note(N(strict=False), _fake_tr(tests_ok=None)) == ""
    assert _red_verdict_note(
        N(strict=False), _fake_tr(tests_preexisting_only=True)) == ""
    assert _red_verdict_note(N(strict=True), _fake_tr()) == ""
    assert _red_verdict_note(N(strict=False), None) == ""


def _gate_red_loop_run(tmp_path, monkeypatch, *, tests_ok, preexisting=False):
    from pedacito import loop as loop_mod
    from pedacito.config import Config
    from pedacito.edits import Result
    (tmp_path / "a.py").write_text("GOOD = 1\n", encoding="utf-8")
    plan_path = tmp_path / "PLAN.md"
    loop_mod.write_plan(plan_path, "goal", ["task A"])
    idx = types.SimpleNamespace(root=tmp_path)

    def fake_run_task(root, cfg_, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None, is_repair=False):
        return loop_mod.TaskResult(
            [Result(True, "a.py", "applied")], "reply", True, ["a.py"], idx_,
            tests_ok=tests_ok, tests_preexisting_only=preexisting)

    monkeypatch.setattr(loop_mod, "run_task", fake_run_task)
    cfg = Config()
    cfg.log_sessions = cfg.log_events = False
    cfg.static_gate = cfg.frontend_contracts = cfg.interface_lock = False
    cfg.use_memory = cfg.backup = False
    code = loop_mod.run_loop(tmp_path, cfg, None, idx, plan_path, apply=True)
    return code


def test_run_loop_finish_line_names_gate_red_tasks(tmp_path, monkeypatch,
                                                   capsys):
    # the live defect: "[8 done, 0 failed]" after two frontend-gate
    # "skipped until they are fixed" rounds -- the skipped checks were
    # never fixed and the finish line said nothing. Exit 0 is the
    # documented non-strict contract and stays; SILENCE was the bug.
    code = _gate_red_loop_run(tmp_path, monkeypatch, tests_ok=False)
    err = capsys.readouterr().err
    assert code == 0
    assert "verification red" in err                       # per task
    assert "ticked done while the last verification gate " \
           "was RED" in err                                # finish line
    assert "1 task(s)" in err


def test_run_loop_stays_quiet_when_the_round_was_not_red(tmp_path,
                                                         monkeypatch,
                                                         capsys):
    # a green verdict, and a red suite that purely predates the task
    # (BUG-220: inherited red is not this task's gate), must not alarm
    code = _gate_red_loop_run(tmp_path, monkeypatch, tests_ok=True)
    assert code == 0
    assert "verification gate was RED" not in capsys.readouterr().err
    code = _gate_red_loop_run(tmp_path, monkeypatch, tests_ok=False,
                              preexisting=True)
    assert code == 0
    assert "verification gate was RED" not in capsys.readouterr().err


# --------------------------------------------------------------------------
# BUG-809: a protective tail ("Touch nothing else.") merged into its edit
# clause must not reclassify the whole clause as a pure constraint. The live
# run: `ask --review` on "Change index.html title ... Touch nothing else."
# produced clause_map:{} -- the file diff carried no provenance badge, the
# coverage seed was empty, and a partial accept could not name what it left
# unaccepted, because the only requirement had become kind=constraint.
# --------------------------------------------------------------------------

def test_prohibition_tail_does_not_erase_the_edit_clause():
    reqs = verify_mod._requirements_from_task(
        "Change index.html. Touch nothing else.")
    assert len(reqs) == 1                     # tail merged, not a second R
    assert reqs[0].kind == "edit" and reqs[0].is_edit


def test_verbless_prohibition_still_feeds_the_coverage_ledger():
    # requirement_ledger drops constraint clauses; before the fix the fused
    # clause was one, so the whole edit task left an empty ledger.
    assert [t for t, _ in verify_mod.requirement_ledger(
        "Change index.html. Touch nothing else.")] == \
        ["Change index.html Touch nothing else"]


def test_leading_prohibition_is_still_a_constraint_bug_306():
    # the fix must not swallow the BUG-306 cases: a clause whose OWN head is
    # the prohibition binds the model but is never a step, a badge or a seed.
    for c in ("Do not modify game.js.",
              "Keep index.html unchanged.",
              "Touch nothing else.",
              "Change index.html. Do not modify game.js."):
        reqs = verify_mod._requirements_from_task(c)
        poisoned = [r for r in reqs
                    if "Do not" in r.text or "unchanged" in r.text.lower()
                    or r.text.lower().startswith("touch")]
        assert poisoned and all(r.kind == "constraint" and not r.is_edit
                                for r in poisoned), c


def test_prohibition_tail_edit_still_yields_review_provenance():
    # the R-badge path: clause_map is built only from is_edit clauses with
    # recorded edit spans (gui/core.py review payload).
    tr = verify_mod.RequirementTracker(
        "Change index.html. Touch nothing else.")
    changed = tr.observe_edit(file="index.html",
                              search="<title>Neon Vortex DX</title>",
                              replace="<title>NeonVortex-DX</title>")
    assert changed and changed[0].req_id == "R1"
    assert tr.clause_map() == {"index.html": ["R1"]}
    assert "R1" in tr.clause_texts()


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
