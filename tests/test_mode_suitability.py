"""Task-vs-Auto mode suitability: the deterministic classifier, the CLI/TUI
interactive nudges, and the GUI's live banner + pre-run choice.

The classifier (`loop.evaluate_mode_suitability`) is zero-LLM: it flags
single-pass Task work (1-2 existing files, localized edit verbs, tiny
projects) so a 5-part HTML/CSS tweak never spins up a 14-step Auto loop.
Explicit planning/architecture asks always keep Auto.
"""
from __future__ import annotations

# ruff: noqa: F811  (imported pytest fixtures are re-bound as test parameters)

import io
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import cli, loop  # noqa: E402
from pedacito import tui as tui_mod  # noqa: E402
from pedacito.config import Config  # noqa: E402
from pedacito.gui import jobs  # noqa: E402
from pedacito.tui import TuiSession  # noqa: E402
from test_gui import gui, project, fake_port  # noqa: F401,E402 (shared fixtures)
from test_e2e_cli import env  # noqa: F401,E402 (isolated CLI project)

ROOT = Path(__file__).resolve().parents[1]
FAKE = Path(__file__).resolve().parent / "fake_llm.py"

# The 5-part index.html / input.css content+styling edit: one cohesive pass.
FIVE_PART_PROMPT = (
    "In index.html remove the old hero banner, change the headline to "
    "'Fresh Cakes', stack the pricing cards vertically, add a link to the "
    "menu page, and in input.css restyle the buttons with rounded corners "
    "and update the color palette."
)
BIG_PROMPT = (
    "Build a full FastAPI backend with SQLite migrations, auth middleware, "
    "CLI client, and pytest suite from scratch"
)


@pytest.fixture
def web_root(tmp_path):
    (tmp_path / "index.html").write_text(
        "<html><body><h1>Cakes</h1></body></html>\n", encoding="utf-8")
    (tmp_path / "input.css").write_text(
        "body { margin: 0; }\n", encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------- classifier --

def test_multi_clause_html_css_edit_is_flagged_for_task(web_root):
    verdict = loop.evaluate_mode_suitability(FIVE_PART_PROMPT, web_root,
                                             "auto")
    assert verdict["recommended_mode"] == "ask"
    assert verdict["mismatch"] is True
    assert "Task" in verdict["reason"] and verdict["reason"].endswith(".")
    # The same prompt without a disk root still flags on the edit verbs.
    assert loop.evaluate_mode_suitability(
        FIVE_PART_PROMPT, None, "auto")["mismatch"] is True


def test_large_multi_module_architecture_prompt_keeps_auto(tmp_path):
    for root in (tmp_path, None):
        verdict = loop.evaluate_mode_suitability(BIG_PROMPT, root, "auto")
        assert verdict["recommended_mode"] == "auto"
        assert verdict["mismatch"] is False


def test_explicit_planning_phrases_override_the_nudge(web_root):
    for phrase in ("create a plan", "multi-step plan", "from scratch"):
        prompt = f"Please {phrase}: restyle the buttons in index.html"
        verdict = loop.evaluate_mode_suitability(prompt, web_root, "auto")
        assert verdict["mismatch"] is False, phrase
        assert verdict["recommended_mode"] == "auto"


def test_four_or_more_new_files_or_modules_keep_auto(tmp_path):
    four_files = ("Build auth.py, models.py, cli.py and test_api.py for "
                  "the new service")
    assert loop.evaluate_mode_suitability(
        four_files, tmp_path, "auto")["mismatch"] is False
    four_modules = ("Build a backend with auth middleware, a database "
                    "layer, a CLI client and a dashboard")
    assert loop.evaluate_mode_suitability(
        four_modules, tmp_path, "auto")["mismatch"] is False
    # ...but the SAME four filenames already on disk are a multi-file edit,
    # not a new build: the verb branch still recommends one Task pass.
    for name in ("auth.py", "models.py", "cli.py", "test_api.py"):
        (tmp_path / name).write_text("x = 1\n", encoding="utf-8")
    verdict = loop.evaluate_mode_suitability(
        "update auth.py, models.py, cli.py and test_api.py", tmp_path,
        "auto")
    assert verdict["recommended_mode"] == "ask"


def test_localized_edit_verbs_flag_without_any_file_mention(tmp_path):
    verdict = loop.evaluate_mode_suitability(
        "fix the typo in the header and hide the debug banner", tmp_path,
        "auto")
    assert verdict["mismatch"] is True
    assert verdict["recommended_mode"] == "ask"


def test_tiny_project_content_tweak_flags_and_scaffolding_does_not(tmp_path):
    for name in ("index.html", "app.js", "style.css"):
        (tmp_path / name).write_text("/* empty */\n", encoding="utf-8")
    tweak = ("the landing page content and styling should feel warmer: "
             "a new palette and bigger fonts")
    verdict = loop.evaluate_mode_suitability(tweak, tmp_path, "auto")
    assert verdict["mismatch"] is True
    assert "3 source file" in verdict["reason"]
    scaffold = "scaffold a FastAPI backend with a pytest test suite"
    assert loop.evaluate_mode_suitability(
        scaffold, tmp_path, "auto")["mismatch"] is False


def test_ignored_directories_do_not_grow_the_project(tmp_path):
    (tmp_path / "index.html").write_text("<html></html>\n", encoding="utf-8")
    junk = tmp_path / "node_modules" / "junk"
    junk.mkdir(parents=True)
    for i in range(8):
        (junk / f"lib{i}.js").write_text("module.exports = {};\n",
                                         encoding="utf-8")
    hidden = tmp_path / ".pedacito"
    hidden.mkdir()
    (hidden / "state.py").write_text("x = 1\n", encoding="utf-8")
    verdict = loop.evaluate_mode_suitability(
        "the page content and styling need a warmer palette", tmp_path,
        "auto")
    assert verdict["mismatch"] is True
    assert "1 source file" in verdict["reason"]


def test_three_existing_files_without_edit_verbs_keep_auto(tmp_path):
    for name in ("a.py", "b.py", "c.py"):
        (tmp_path / name).write_text("x = 1\n", encoding="utf-8")
    verdict = loop.evaluate_mode_suitability(
        "a.py, b.py and c.py are the release candidates", tmp_path, "auto")
    assert verdict["mismatch"] is False


def test_ask_request_and_empty_prompt_never_mismatch(web_root):
    verdict = loop.evaluate_mode_suitability(FIVE_PART_PROMPT, web_root,
                                             "ask")
    assert verdict == {"recommended_mode": "ask", "mismatch": False,
                       "reason": ""}
    assert loop.evaluate_mode_suitability(
        "   ", web_root, "auto")["mismatch"] is False


# ----------------------------------------------------------------- CLI nudge --

class _FakeTTY(io.StringIO):
    def isatty(self):
        return True


class _FakePipe(io.StringIO):
    def isatty(self):
        return False


def _nudge_args(**over):
    base = dict(cmd="auto", task="fix the typo in index.html",
                force_auto=False, yes=False)
    base.update(over)
    return SimpleNamespace(**base)


def test_cli_non_tty_auto_never_blocks_on_stdin(web_root, monkeypatch,
                                                capsys):
    monkeypatch.setattr(sys, "stdin", _FakePipe())

    def boom(*a, **k):
        raise AssertionError("input() must never run on a non-TTY stdin")

    monkeypatch.setattr("builtins.input", boom)
    assert cli._auto_mode_nudge(_nudge_args(), web_root) is False
    err = capsys.readouterr().err
    assert "single-pass" in err and err.count("\n") == 1


def test_cli_tty_nudge_default_yes_and_answers(web_root, monkeypatch,
                                               capsys):
    monkeypatch.setattr(sys, "stdin", _FakeTTY())
    seen = []

    def answer(text):
        def _input(prompt=""):
            seen.append(prompt)
            return text
        return _input

    for text, want in (("", True), ("y", True), ("yes", True),
                       ("n", False), ("no", False)):
        seen.clear()
        monkeypatch.setattr("builtins.input", answer(text))
        assert cli._auto_mode_nudge(_nudge_args(), web_root) is want, text
        assert seen == ["Run as a single-pass Task instead? [Y/n]: "]
    err = capsys.readouterr().err
    assert ("\U0001f4a1 [pedacito] This request looks better suited for "
            "single-pass Task mode than multi-step Auto mode:") in err


def test_cli_force_auto_and_yes_bypass_the_nudge(web_root, monkeypatch,
                                                 capsys):
    monkeypatch.setattr(sys, "stdin", _FakeTTY())
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail(
        "the nudge must not ask when bypassed"))
    assert cli._auto_mode_nudge(_nudge_args(force_auto=True),
                                web_root) is False
    assert cli._auto_mode_nudge(_nudge_args(yes=True), web_root) is False
    assert cli._auto_mode_nudge(_nudge_args(task=""), web_root) is False
    assert capsys.readouterr().err == ""


def test_cli_nudge_eof_keeps_auto(web_root, monkeypatch):
    monkeypatch.setattr(sys, "stdin", _FakeTTY())

    def eof(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert cli._auto_mode_nudge(_nudge_args(), web_root) is False


def test_auto_parser_accepts_force_auto_and_yes():
    parser = cli.build_parser()
    args = parser.parse_args(["auto", ".", "-t", "x", "--force-auto"])
    assert args.force_auto is True and args.yes is False
    args = parser.parse_args(["auto", ".", "-t", "x", "-y"])
    assert args.yes is True and args.force_auto is False


def test_cmd_auto_itself_never_reads_stdin(web_root, monkeypatch):
    """The loop entry point stays pipe-safe: plan generation for a
    mismatching goal must not touch input() (the nudge lives in main)."""
    class Stub:
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            return {"tasks": [{"title": "fix the typo"}]}

    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail(
        "_cmd_auto must never block on stdin"))
    args = SimpleNamespace(task="fix the typo in index.html", spec=None,
                           plan=None, apply=False, force=False, verify=None,
                           repair=None, max_tasks=None, keep_going=False)
    assert cli._cmd_auto(args, Config(), Stub(), None, web_root) == 0
    assert (web_root / "PLAN.md").exists()


# ------------------------------------------- CLI end to end (fake server) ----

LIMIT_REPLY = ("Raised the limit.\n\n--- FILE: app.py\n<<<<<<< SEARCH\n"
               "LIMIT = 10\n=======\nLIMIT = 25\n>>>>>>> REPLACE\n")


@pytest.fixture(scope="module")
def cli_fake_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    env_vars = dict(os.environ, FAKE_PORT=str(port), FAKE_REPLY=LIMIT_REPLY,
                    FAKE_READ_FILE="app.py")
    env_vars.pop("FAKE_ACTIONS", None)
    proc = subprocess.Popen([sys.executable, str(FAKE)], env=env_vars,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                socket.create_connection(("127.0.0.1", port),
                                         timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError("fake server did not start")
        yield port
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def _auto_argv(port, proj, *extra):
    return ["auto", str(proj), "-t",
            "fix the header comment in app.py and update the LIMIT note",
            "--url", f"http://127.0.0.1:{port}/v1",
            "--model", "fake-model", *extra]


@pytest.mark.network
def test_auto_nudge_yes_runs_one_task_pass(cli_fake_port, env, monkeypatch,
                                           capsys):
    monkeypatch.setattr(sys, "stdin", _FakeTTY())
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    code = cli.main(_auto_argv(cli_fake_port, env, "--apply"))
    err = capsys.readouterr().err
    assert code == 0
    assert "better suited for single-pass Task mode" in err
    assert "[running as a single-pass Task (ask)]" in err
    # The single pass applied the edit; no multi-step checklist was born.
    assert "LIMIT = 25" in (env / "app.py").read_text(encoding="utf-8")
    assert not (env / "PLAN.md").exists()


@pytest.mark.network
def test_auto_nudge_no_keeps_the_loop(cli_fake_port, env, monkeypatch,
                                      capsys):
    monkeypatch.setattr(sys, "stdin", _FakeTTY())
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    seen = []
    monkeypatch.setattr(cli, "_cmd_auto",
                        lambda args, cfg, client, idx, root:
                        seen.append(args.task) or 0)
    code = cli.main(_auto_argv(cli_fake_port, env))
    capsys.readouterr()
    assert code == 0
    assert seen == ["fix the header comment in app.py and update the "
                    "LIMIT note"]


@pytest.mark.network
def test_force_auto_skips_the_question(cli_fake_port, env, monkeypatch,
                                       capsys):
    monkeypatch.setattr(sys, "stdin", _FakeTTY())
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail(
        "--force-auto must not ask"))
    seen = []
    monkeypatch.setattr(cli, "_cmd_auto",
                        lambda args, cfg, client, idx, root:
                        seen.append(args.task) or 0)
    assert cli.main(_auto_argv(cli_fake_port, env, "--force-auto")) == 0
    capsys.readouterr()
    assert len(seen) == 1


# ----------------------------------------------------------------- TUI nudge --

def _tui(tmp_path, script=""):
    cfg = Config()
    cfg.model = "fake-model"
    return TuiSession(tmp_path, cfg, client=None, headless=True,
                      stdin=io.StringIO(script), stdout=io.StringIO(),
                      width=100, height=24)


def test_tui_headless_auto_logs_a_note_and_never_waits(tmp_path,
                                                       monkeypatch):
    (tmp_path / "index.html").write_text("<html></html>\n", encoding="utf-8")
    session = _tui(tmp_path, "/auto fix the typo in index.html\n/quit\n")
    calls = []
    monkeypatch.setattr(session, "run_auto",
                        lambda goal: calls.append(("auto", goal)))
    monkeypatch.setattr(session, "run_task",
                        lambda task: calls.append(("task", task)))
    try:
        assert session.run() == 0
    finally:
        session.close()
    assert calls == [("auto", "fix the typo in index.html")]
    text = session.stdout.getvalue()
    assert "better suited for single-pass Task mode" in text
    assert "non-interactive" in text


def test_tui_interactive_nudge_default_yes_runs_task(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<html></html>\n", encoding="utf-8")
    session = _tui(tmp_path)
    session.headless = False               # exercise the interactive branch
    calls = []
    monkeypatch.setattr(session, "run_auto",
                        lambda goal: calls.append(("auto", goal)))
    monkeypatch.setattr(session, "run_task",
                        lambda task: calls.append(("task", task)))
    try:
        for answer, want in (("", "task"), ("y", "task"), ("n", "auto")):
            calls.clear()
            monkeypatch.setattr(session, "_read_answer_line",
                                lambda a=answer: a)
            session.handle_line("/auto fix the typo in index.html")
            assert calls == [(want, "fix the typo in index.html")], answer
        transcript = "\n".join(t for _k, t in session.transcript)
        assert "Run as a single-pass Task instead? [Y/n]:" in transcript
    finally:
        session.close()


def test_tui_auto_bang_forces_the_loop(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<html></html>\n", encoding="utf-8")
    session = _tui(tmp_path)
    calls = []
    monkeypatch.setattr(session, "run_auto",
                        lambda goal: calls.append(("auto", goal)))
    monkeypatch.setattr(session, "run_task",
                        lambda task: calls.append(("task", task)))
    try:
        session.handle_line("/auto! fix the typo in index.html")
        assert calls == [("auto", "fix the typo in index.html")]
        transcript = "\n".join(t for _k, t in session.transcript)
        assert "better suited" not in transcript
        # A big architecture goal skips the question entirely.
        calls.clear()
        session.handle_line(f"/auto {BIG_PROMPT}")
        assert calls == [("auto", BIG_PROMPT)]
    finally:
        session.close()


def test_tui_help_documents_auto_bang():
    assert "/auto! GOAL" in tui_mod.HELP_TEXT
    assert "/auto!" in tui_mod._SLASH_COMMANDS


# ------------------------------------------------------------- GUI backend --

def test_mode_check_endpoint_classifies_without_a_job(gui):
    status, j = gui.post("/api/mode-check",
                         {"text": FIVE_PART_PROMPT, "mode": "auto"})
    assert status == 200 and j["ok"] is True
    assert j["recommended_mode"] == "ask" and j["mismatch"] is True
    assert j["reason"]
    # The check is synchronous: it never claims the single job slot.
    assert gui.get("/api/state")[1]["job"] is None
    status, j = gui.post("/api/mode-check", {"text": BIG_PROMPT,
                                             "mode": "auto"})
    assert status == 200 and j["mismatch"] is False
    status, j = gui.post("/api/mode-check", {"text": FIVE_PART_PROMPT,
                                             "mode": "task"})
    assert status == 200 and j["recommended_mode"] == "ask"
    assert j["mismatch"] is False


@pytest.mark.network
def test_mode_check_endpoint_uses_the_open_project(project):
    status, j = project.post("/api/mode-check",
                             {"text": "fix the typo in app.py",
                              "mode": "auto"})
    assert status == 200 and j["mismatch"] is True
    assert j["recommended_mode"] == "ask"


@pytest.mark.network
def test_api_auto_accepts_force_auto(project, monkeypatch):
    seen = {}

    def fake_auto(ns, cfg, client, idx, root):
        seen["force_auto"] = getattr(ns, "force_auto", False)
        return 0

    monkeypatch.setattr(jobs.cli_mod, "_cmd_auto", fake_auto)
    status, _ = project.post("/api/auto", {"text": "fix the typo in app.py",
                                           "force_auto": True})
    assert status == 202 and project.wait_idle(30)
    assert seen["force_auto"] is True
    status, _ = project.post("/api/auto", {"text": "fix the typo in app.py"})
    assert status == 202 and project.wait_idle(30)
    assert seen["force_auto"] is False


# ------------------------------------------------------------ GUI frontend --

def test_gui_html_serves_the_mode_helpers_and_recommendation_handler(gui):
    status, _h, html = gui.req("GET", "/")
    assert status == 200
    assert b"/static/app.js" in html
    status, _h, js = gui.req("GET", "/static/app.js")
    assert status == 200
    text = js.decode("utf-8")
    # The four mode helper descriptions (Task/ask, Auto, Plan, New).
    assert "Single-pass edit" in text
    assert "reads & applies all parts of your prompt together in 1 turn" \
        in text
    assert "Autonomous multi-step loop" in text
    assert "splits big goals into a PLAN.md checklist" in text
    assert "Generates or updates PLAN.md without editing code yet." in text
    assert "Scaffolds a brand-new project in a target directory." in text
    # Tooltips on every mode button.
    assert "title=${MODES[k].title}" in text
    # The live recommendation banner + pre-run choice handler.
    assert "/api/mode-check" in text
    assert "Looks like a single-pass edit" in text
    assert "Switch to Task mode (Recommended)" in text
    assert "Continue with Auto" in text
    assert "Run as a single-pass Task instead?" in text
    assert "Run as Task (Recommended)" in text
    assert "Run Auto Anyway" in text
    assert "force_auto" in text
    assert "scheduleModeCheck" in text
    status, _h, css = gui.req("GET", "/static/app.css")
    assert status == 200
    assert b"#modeNudge" in css and b"#modeConfirm" in css


# ------------------------------------------------------------- docs in sync --

@pytest.mark.parametrize("doc", ["README.md", "QUICKSTART.md"])
def test_force_auto_is_documented(doc):
    paths = [ROOT / doc]
    if doc == "README.md":
        paths += sorted((ROOT / "docs").glob("*.md"))
    text = " ".join(" ".join(p.read_text(encoding="utf-8") for p in paths).split())
    assert "--force-auto" in text
    assert "Run as a single-pass Task instead?" in text


def test_how_cards_cover_the_nudge():
    from pedacito import how_index
    cards = {c["name"]: c for c in how_index.CLI_COMMANDS}
    auto = cards["auto"]
    assert "--force-auto" in auto["usage"]
    assert any(f["flag"] == "--force-auto" for f in auto["flags"])
    assert any(f["flag"] == "-y, --yes" for f in auto["flags"])
    assert "Task-vs-Auto" in " ".join(auto.get("notes", []))
    tui_notes = " ".join(cards["tui"]["notes"])
    assert "/auto!" in tui_notes
    assert "single-pass Task" in tui_notes
