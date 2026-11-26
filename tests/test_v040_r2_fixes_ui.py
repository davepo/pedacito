"""v0.4.0 round 2 remediation (track R2-C): GUI settings range validation,
malformed JSON bodies on the settings and review routes, Spanish GUI modal
buttons, Spanish `how` cards, and Spanish/plural TUI strings."""

from __future__ import annotations

import io
import json
import os
import site
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from pedacito import config as config_mod  # noqa: E402
from pedacito import config_editor, how, i18n, tui  # noqa: E402
from pedacito.chunker import Chunk  # noqa: E402
from pedacito.config import ConfigError  # noqa: E402
from pedacito.index import Index  # noqa: E402
from test_gui import gui as _shared_gui  # noqa: E402,F401  (pytest re-export)
from test_v040_tui_and_how_i18n import _drive, _indexed_project  # noqa: E402

_USER_SITE = site.getusersitepackages()

NEW_GUI_KEYS = (
    "settings.scan.local", "settings.scan.again", "settings.scan.scanning",
    "settings.profile.calibrate", "settings.profile.calibrate_title",
    "settings.profile.calibrate_hint", "settings.profile.default",
    "settings.profile.set_default", "settings.profile.rename_title",
    "settings.profile.new", "settings.profile.add_hosted",
    "settings.profile.add_hosted_title", "settings.profile.calibrate_all_title",
    "settings.calall.title", "settings.calall.run",
)


@pytest.fixture
def gui(_shared_gui):  # noqa: F811
    return _shared_gui


def _user_config(gui) -> str:
    gui.cfg_path.parent.mkdir(parents=True, exist_ok=True)
    gui.cfg_path.write_text("[universal]\nvram_gb = 8\n", encoding="utf-8")
    return gui.cfg_path.read_text(encoding="utf-8")


# ------------------------------------------------ R2-UI-01 range validation --

def test_validate_text_rejects_out_of_range_whole_numbers():
    for text, key in (("daemon_port = -1\n", "daemon_port"),
                      ("daemon_port = 70000\n", "daemon_port"),
                      ("subagent_max_parallel = 0\n", "subagent_max_parallel"),
                      ("subagent_max_parallel = 33\n", "subagent_max_parallel")):
        with pytest.raises(ConfigError, match=f"{key} must be a whole number"):
            config_editor.validate_text(text)
    for text in ("daemon_port = 0\n", "daemon_port = 65535\n",
                 "subagent_max_parallel = 1\n", "subagent_max_parallel = 32\n"):
        flat, _profiles = config_editor.validate_text(text)
        assert flat


def test_profile_scoped_range_is_checked_too():
    with pytest.raises(ConfigError, match="subagent_max_parallel must be"):
        config_editor.validate_text(
            "[profiles.fast]\nmodel = \"m\"\nsubagent_max_parallel = 0\n")


def test_settings_set_refuses_out_of_range_and_writes_nothing(gui):
    before = _user_config(gui)
    for changes in ({"daemon_port": -1}, {"daemon_port": 70000},
                    {"subagent_max_parallel": 0}):
        status, body = gui.post("/api/settings/set",
                                {"file": "user", "changes": changes})
        assert status == 400 and body["ok"] is False, changes
        assert "must be a whole number" in body["detail"], body
    assert gui.cfg_path.read_text(encoding="utf-8") == before


def test_settings_set_accepts_in_range_values(gui):
    _user_config(gui)
    for changes in ({"daemon_port": 0}, {"daemon_port": 8080},
                    {"subagent_max_parallel": 4}):
        status, body = gui.post("/api/settings/set",
                                {"file": "user", "changes": changes})
        assert status == 200 and body["ok"], (changes, body)
    text = gui.cfg_path.read_text(encoding="utf-8")
    assert "daemon_port = 8080" in text
    assert "subagent_max_parallel = 4" in text


# ---------------------------------------- R2-UI-02 / R2-UI-09 malformed JSON --

def test_malformed_json_body_is_400_on_settings_and_review(gui):
    for path in ("/api/settings/set", "/api/review/decide"):
        status, _headers, data = gui.req("POST", path, raw=b"{not json")
        assert status == 400, (path, data)
        assert json.loads(data) == {"ok": False, "error": "invalid JSON body"}


def test_empty_body_is_not_treated_as_malformed(gui):
    _user_config(gui)
    status, _headers, data = gui.req("POST", "/api/settings/set", raw=b"")
    assert b"invalid JSON body" not in data
    assert status != 500


# ------------------------------------------ R2-UI-03 Spanish GUI modal labels --

def test_new_gui_keys_exist_in_both_catalogs_and_differ():
    en = i18n.load_catalog("en")
    es = i18n.load_catalog("es")
    for key in NEW_GUI_KEYS:
        assert key in en and key in es, key
        assert en[key] != es[key], key
        assert sorted(_placeholders(en[key])) == sorted(_placeholders(es[key])), key


def test_settings_modal_buttons_use_locale_keys():
    source = (REPO / "pedacito" / "gui" / "static" / "settings.js").read_text(
        encoding="utf-8")
    for key in NEW_GUI_KEYS:
        assert f"'{key}'" in source, key
    for literal in ("text: 'Scan local servers'", "text: 'Calibrate',",
                    "text: 'Set default'", "text: '+ New profile'",
                    "text: '+ Add Hosted Provider\\u2026'"):
        assert literal not in source, literal


def _placeholders(text: str) -> set[str]:
    import re
    return set(re.findall(r"\{([A-Za-z0-9_]+)\}", text))


# ---------------------------------- R2-UI-04 / BUG-08 Spanish how answers --

def test_how_spanish_setting_card_has_banner():
    result = how.answer_structured("sandbox", lang="es", use_model=False)
    assert result["setting"] is not None
    assert "Referencia integrada" in result["answer"]


def test_how_cli_spanish_sandbox_has_banner():
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO), _USER_SITE, env.get("PYTHONPATH", "")) if p)
    proc = subprocess.run(
        [sys.executable, "-m", "pedacito", "how", "--offline", "--lang", "es",
         "sandbox"],
        capture_output=True, cwd=str(REPO), env=env, timeout=300)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert "Referencia integrada" in proc.stdout.decode("utf-8")


@pytest.mark.parametrize("question", ["política", "permisos"])
def test_how_spanish_permission_cards_are_spanish(question):
    answer = how.answer_structured(question, lang="es",
                                   use_model=False)["answer"]
    assert "Referencia integrada" in answer
    assert "Permitir una vez" in answer
    assert "Denegar y redirigir" in answer
    for english in ("Allow Once", "Deny & Redirect", "Always Allow",
                    "Allow for This Run", "Read the badge"):
        assert english not in answer, english


def test_how_spanish_daemon_card_is_spanish():
    answer = how.answer_structured("daemon", lang="es",
                                   use_model=False)["answer"]
    assert "Referencia integrada" in answer
    assert "archivo de descubrimiento" in answer
    assert "every request needs" not in answer


def test_how_english_permission_card_is_unchanged():
    answer = how.answer_structured("permissions", lang="en",
                                   use_model=False)["answer"]
    assert "Allow Once: this single request only" in answer
    assert "Built-in reference" in answer


# --------------------------------- R2-UI-05 / R2-UI-07 TUI Spanish and plural --

def _tiny_index(root: Path, count: int) -> Index:
    chunks = [Chunk(file="app/x.py", qualname=f"fn{i}", kind="function",
                    start=i + 1, end=i + 1) for i in range(count)]
    return Index(root, chunks)


def test_tui_expand_without_cards_is_spanish(tmp_path):
    text = _drive(tmp_path, ["/expand"], lang="es", width=80,
                  idx=_indexed_project(tmp_path))
    flat = " ".join(text.split())
    assert "No tool-call cards in this session yet." not in flat
    assert "Todavía no hay tarjetas de herramientas en esta sesión." in flat


def test_tui_expand_without_cards_is_english_in_english(tmp_path):
    text = _drive(tmp_path, ["/expand"], lang="en", width=80,
                  idx=_indexed_project(tmp_path))
    assert "No tool-call cards in this session yet." in text


@pytest.mark.parametrize("width", [60, 80])
def test_tui_spanish_footer_and_expand_never_overflow(tmp_path, width):
    text = _drive(tmp_path, ["/expand", "/map", "/status"], lang="es",
                  width=width, idx=_indexed_project(tmp_path))
    lines = text.splitlines()
    assert lines
    too_long = [line for line in lines if len(line) > width]
    assert not too_long, too_long[:3]
    assert "Turno:" in text and "Sesión:" in text


def test_tui_map_singular_symbol_english(tmp_path):
    text = _drive(tmp_path, ["/map"], lang="en", width=80,
                  idx=_tiny_index(tmp_path, 1))
    assert "app/x.py (1 symbol): fn0" in text
    assert "1 symbols" not in text
    assert "1 symbol)" in text


def test_tui_map_plural_and_singular_spanish(tmp_path):
    one = _drive(tmp_path, ["/map"], lang="es", width=80,
                 idx=_tiny_index(tmp_path, 1))
    assert "(1 símbolo) · fn0" in one
    many = _drive(tmp_path, ["/map"], lang="es", width=80,
                  idx=_tiny_index(tmp_path, 2))
    assert "(2 símbolos)" in many


def test_tui_symbol_count_helper_uses_plural_keys(tmp_path):
    session, _stdout = _session(tmp_path, lang="en")
    try:
        assert session._symbol_count(1) == "1 symbol"
        assert session._symbol_count(0) == "0 symbols"
        assert session._symbol_count(5) == "5 symbols"
    finally:
        session.close()
        i18n.set_current_lang("en")


def _session(tmp_path, *, lang: str):
    stdout = io.StringIO()
    session = tui.TuiSession(
        tmp_path, config_mod.Config(), headless=True, stdin=io.StringIO(""),
        stdout=stdout, width=80, height=24, lang=lang)
    return session, stdout
