"""v0.4.0 bug-hunt wave 1 (WP-B4): TUI width wrapping, `/map` and `/files`,
complete Spanish TUI strings, and Spanish deterministic `how` answers."""

from __future__ import annotations

import io
import os
import site
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pedacito import config as config_mod  # noqa: E402
from pedacito import how, i18n, security, tui  # noqa: E402
from pedacito.chunker import Chunk  # noqa: E402
from pedacito.index import Index  # noqa: E402

# Captured before the autouse fixture moves APPDATA, so children keep the
# per-user site-packages that hold this environment's dependencies.
_USER_SITE = site.getusersitepackages()

LONG_ASSISTANT = (
    "This is a deliberately long assistant reply that keeps going with many "
    "words so the headless transcript has to wrap it cleanly to the width "
    "without overflowing into the next row of output. " * 3)
LONG_TOOL = ("tool_result: " + "src/pedacito/some_deeply_nested_module_name.py "
             "* 4 ") * 2
WIDTH_COMMANDS = ["/help", "/status", "/plan", "/map", "/files", "/expand",
                  "/quit"]


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO), _USER_SITE, env.get("PYTHONPATH", "")) if p)
    return env


def _session(tmp_path, *, lang: str, width: int, idx=None):
    stdout = io.StringIO()
    session = tui.TuiSession(
        tmp_path, config_mod.Config(), headless=True, stdin=io.StringIO(""),
        stdout=stdout, width=width, height=24, lang=lang, idx=idx)
    return session, stdout


def _drive(tmp_path, commands, *, lang: str, width: int, idx=None) -> str:
    session, stdout = _session(tmp_path, lang=lang, width=width, idx=idx)
    try:
        session._redraw(force=True)
        session.append(LONG_ASSISTANT, "assistant")
        session.append(LONG_TOOL, "system")
        session._redraw(force=True)
        for line in commands:
            session.handle_line(line)
            session._redraw(force=True)
    finally:
        session.close()
        i18n.set_current_lang("en")
    return stdout.getvalue()


def _indexed_project(root: Path) -> Index:
    chunks = [
        Chunk(file="app/main.py", qualname="main", kind="function",
              start=1, end=3),
        Chunk(file="app/main.py", qualname="Server", kind="class",
              start=5, end=20),
        Chunk(file="app/main.py", qualname="Server.start", kind="method",
              start=6, end=9),
        Chunk(file="config/.env", qualname="API_TOKEN", kind="function",
              start=1, end=1),
        Chunk(file="config/credentials.json", qualname="load_creds",
              kind="function", start=1, end=2),
    ]
    return Index(root, chunks)


# ------------------------------------------------------------ BUG-07 wrapping --

def test_headless_output_never_exceeds_width(tmp_path):
    for lang in ("en", "es"):
        for width in (80, 100):
            text = _drive(tmp_path, WIDTH_COMMANDS, lang=lang, width=width,
                          idx=_indexed_project(tmp_path))
            lines = text.splitlines()
            assert len(lines) > 40
            too_long = [line for line in lines if len(line) > width]
            assert not too_long, (lang, width, too_long[:3])


def test_headless_rows_wrap_with_continuation_indent():
    rows = tui._headless_rows("system", "  " + "word " * 60, 40)
    assert len(rows) > 1
    assert all(len(row) <= 40 for row in rows)
    assert rows[1].startswith(" " * len("system: "))
    assert "".join(rows).count("word") == 60


def test_headless_rows_clamp_tiny_widths():
    rows = tui._headless_rows("activity", "x" * 200, 5)
    assert all(len(row) <= 20 for row in rows)


def test_cli_headless_tui_respects_width_in_english_and_spanish():
    for lang in ("en", "es"):
        proc = subprocess.run(
            [sys.executable, "-m", "pedacito", "tui", "--headless",
             "--no-apply", "--width", "80", "--height", "24", "--lang", lang],
            input=b"/help\n/status\n/quit\n", capture_output=True,
            cwd=str(REPO), env=_child_env(), timeout=600)
        assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
        lines = proc.stdout.decode("utf-8").splitlines()
        assert max(len(line) for line in lines) <= 80, lang


# ------------------------------------------------------------ BUG-12 /map /files --

def test_map_and_files_list_indexed_files_and_hide_sensitive(tmp_path):
    assert security.is_sensitive_path("config/.env")
    assert security.is_sensitive_path("config/credentials.json")
    text = _drive(tmp_path, ["/map", "/files"], lang="en", width=80,
                  idx=_indexed_project(tmp_path))
    assert "Unknown command" not in text
    assert "Project map: 1 files, 3 symbols" in text
    assert "app/main.py (3 symbols): main, Server, Server.start" in text
    assert "Indexed files: 1" in text
    assert "Sensitive files hidden: 2" in text
    assert "API_TOKEN" not in text
    assert "load_creds" not in text
    assert "credentials.json" not in text
    assert ".env" not in text


def test_map_without_index_explains_and_does_not_fail(tmp_path):
    text = _drive(tmp_path, ["/map", "/files"], lang="en", width=80)
    assert "Unknown command" not in text
    assert "This project has no index yet" in text
    assert "No indexed files yet" in text


def test_map_and_files_are_listed_in_help():
    assert "/map" in tui.HELP_TEXT and "/files" in tui.HELP_TEXT
    assert "/map" in tui.HELP_TEXT_ES and "/files" in tui.HELP_TEXT_ES


# ---------------------------------------------------------- BUG-09 Spanish TUI --

def test_spanish_tui_welcome_has_no_english_leftovers(tmp_path):
    session, _stdout = _session(tmp_path, lang="es", width=80)
    try:
        rows = "\n".join(session._all_transcript_rows(80))
        assert "Escribe una tarea de programación" in rows
        assert "PgUp/PgDn o la rueda del ratón" in rows
        assert "Type a coding task" not in rows
    finally:
        session.close()
        i18n.set_current_lang("en")


def test_spanish_tui_plan_and_quit_have_no_english_leftovers(tmp_path):
    text = _drive(tmp_path, ["/plan", "/help", "/quit"], lang="es", width=80)
    for english in ("Type a coding task", "No plan tasks at", "Goodbye."):
        assert english not in text
    assert "No hay tareas de plan en" in text
    assert "Hasta luego." in text
    assert "Comandos de Pedacito TUI" in text


def test_wp_b4_locale_keys_exist_in_both_catalogs():
    en = i18n.load_catalog("en")
    es = i18n.load_catalog("es")
    for key in ("tui.welcome", "tui.goodbye", "tui.plan.none", "tui.map.row",
                "tui.files.skipped", "tui.unknown_command", "how.banner",
                "how.settings_heading", "how.when_choices"):
        assert key in en and key in es, key
        assert en[key] != es[key], key


# ------------------------------------------------------------ BUG-08 Spanish how --

def test_how_offline_spanish_answer_is_spanish():
    result = how.answer_structured("política", lang="es", use_model=False)
    answer = result["answer"]
    assert result["lang"] == "es"
    assert result["mode"] == "deterministic"
    assert "Referencia integrada" in answer
    assert "Configuración relevante" in answer
    assert "En la interfaz gráfica (GUI)" in answer
    assert "Built-in reference" not in answer
    assert "Relevant Settings" not in answer
    # Descriptions come from docs/es/CONFIG.md, not the English schema help.
    assert "La matriz de capacidades de cinco modos" in answer
    assert "por defecto" in answer


def test_how_english_default_is_unchanged():
    result = how.answer_structured("política", lang="en", use_model=False)
    answer = result["answer"]
    assert "**Built-in reference**" in answer
    assert "**Relevant Settings (`config.toml`)**" in answer
    assert "Referencia integrada" not in answer


def test_cli_how_offline_spanish_prints_spanish_reference():
    proc = subprocess.run(
        [sys.executable, "-m", "pedacito", "how", "--offline", "--lang", "es",
         "política"],
        capture_output=True, cwd=str(REPO), env=_child_env(), timeout=600)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    out = proc.stdout.decode("utf-8")
    assert "Referencia integrada" in out
    assert "Configuración relevante" in out
    assert "Built-in reference" not in out
