"""Tests for WP-02: GUI main-page switcher, TUI F3/slash switcher, and CLI i18n."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from pedacito import cli, config as config_mod, i18n, tui
from pedacito.gui import core as gui_core, server as gui_server, settings_api


@pytest.fixture
def isolated_user_config(tmp_path, monkeypatch):
    """Redirect USER_CONFIG and reset GUI STATE/i18n for hermetic tests."""
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setattr(config_mod, "USER_CONFIG", cfg_path)
    monkeypatch.setattr(gui_core, "USER_CONFIG", cfg_path)
    monkeypatch.setattr(settings_api, "USER_CONFIG", cfg_path)
    monkeypatch.delenv("PEDACITO_LANG", raising=False)
    monkeypatch.delenv("PEDACITO_LANGUAGE", raising=False)
    i18n.set_current_lang("en")
    with gui_core.STATE.lock:
        old_cfg = gui_core.STATE.cfg
        old_ucfg = gui_core.STATE._user_cfg
        old_ustamp = gui_core.STATE._user_cfg_stamp
        old_gcfg = gui_core.STATE._global_cfg
        old_gstamp = gui_core.STATE._global_cfg_stamp
        gui_core.STATE.cfg = None
        gui_core.STATE._user_cfg = None
        gui_core.STATE._user_cfg_stamp = None
        gui_core.STATE._global_cfg = None
        gui_core.STATE._global_cfg_stamp = None
    yield cfg_path
    i18n.set_current_lang("en")
    with gui_core.STATE.lock:
        gui_core.STATE.cfg = old_cfg
        gui_core.STATE._user_cfg = old_ucfg
        gui_core.STATE._user_cfg_stamp = old_ustamp
        gui_core.STATE._global_cfg = old_gcfg
        gui_core.STATE._global_cfg_stamp = old_gstamp


# ----------------------------------------------------------------- Web GUI --

def test_api_i18n_and_settings_language(isolated_user_config):
    code, body = gui_server.GET_ROUTES["/api/i18n"]({})
    assert code == 200
    assert body["ok"] is True
    assert body["lang"] == "en"
    assert body["supported"] == ["en", "es"]
    assert "en" in body["catalogs"] and "es" in body["catalogs"]
    assert body["catalogs"]["es"]["gui.settings"] == "Configuración"

    # Switch to Spanish
    code, res = gui_server.POST_ROUTES["/api/settings/language"]({}, {"language": "es"})
    assert code == 200
    assert res["ok"] is True
    assert res["language"] == "es"
    assert res["persisted"] is True
    assert "catalogs" in res
    assert isolated_user_config.exists()
    assert 'language = "es"' in isolated_user_config.read_text(encoding="utf-8")

    # GET /api/i18n and STATE.status() reflect the new language immediately
    code, after = gui_server.GET_ROUTES["/api/i18n"]({})
    assert code == 200
    assert after["lang"] == "es"
    status = gui_core.STATE.status()
    assert status["language"] == "es"
    assert status["llm_reply_language"] == "match_ui"

    # Reject invalid language
    code, err = gui_server.POST_ROUTES["/api/settings/language"]({}, {"language": "fr"})
    assert code == 400
    assert err["ok"] is False


def test_index_html_injects_lang_and_preserves_theme(isolated_user_config):
    raw = b'<!doctype html>\n<html lang="en">\n<head></head><body></body></html>'
    assert b'<html lang="en">' in gui_server.index_html(raw)

    settings_api.save_language("es")
    rendered = gui_server.index_html(raw).decode("utf-8")
    assert '<html lang="es">' in rendered

    settings_api.save_theme("nord")
    rendered_both = gui_server.index_html(raw).decode("utf-8")
    assert '<html lang="es" data-theme="nord">' in rendered_both


def test_gui_static_has_lang_switch_and_i18n_hooks():
    static_dir = Path(gui_server.STATIC)
    app_js = (static_dir / "app.js").read_text(encoding="utf-8")
    settings_js = (static_dir / "settings.js").read_text(encoding="utf-8")
    theme_js = (static_dir / "theme.js").read_text(encoding="utf-8")
    app_css = (static_dir / "app.css").read_text(encoding="utf-8")

    assert 'id="langSwitch"' in app_js
    assert "PDC.setLang" in app_js
    assert "/api/i18n" in app_js
    assert "/api/settings/language" in app_js
    assert "PDC.rerenderSettings" in settings_js
    assert "llm_reply_language" in settings_js
    assert "pedacito_lang" in theme_js
    assert ".lang-switch" in app_css


# --------------------------------------------------------------------- TUI --

def test_tui_f3_parse_and_toggle(tmp_path, isolated_user_config):
    assert tui.parse_escape_sequence("\x1bOR") == ("F3", 3)
    assert tui.parse_escape_sequence("\x1b[13~") == ("F3", 5)
    assert tui._windows_key_sequence("=") == "\x1bOR"

    cfg = config_mod.Config()
    session = tui.TuiSession(tmp_path, cfg, headless=True,
                             stdin=io.StringIO(""), stdout=io.StringIO(""))
    try:
        assert session.lang == "en"
        en_frame = session.render_frame(width=120, height=30)
        assert "lang: [EN|es]" in en_frame
        assert "F3: [EN|es]" in en_frame
        assert "TRANSCRIPT" in en_frame
        assert "ACTIVITY & PLAN" in en_frame
        assert session._prompt_prefix() == f"pdc tui ({session.profile}) > "

        # Press F3 -> switches to Spanish and persists
        assert session.handle_key("F3") is True
        assert session.lang == "es"
        assert session.cfg.language == "es"
        assert 'language = "es"' in isolated_user_config.read_text(encoding="utf-8")

        es_frame = session.render_frame(width=120, height=30)
        assert "idioma: [en|ES]" in es_frame
        assert "F3: [en|ES]" in es_frame
        assert "TRANSCRIPCIÓN" in es_frame
        assert "ACTIVIDAD Y PLAN" in es_frame

        # Press F3 again -> back to English
        assert session.handle_key("F3") is True
        assert session.lang == "en"
        assert 'language = "en"' in isolated_user_config.read_text(encoding="utf-8")
    finally:
        session.close()


def test_tui_slash_lang_command(tmp_path, isolated_user_config):
    cfg = config_mod.Config()
    session = tui.TuiSession(tmp_path, cfg, headless=True,
                             stdin=io.StringIO(""), stdout=io.StringIO(""))
    try:
        session.handle_line("/lang es")
        assert session.lang == "es"
        assert 'language = "es"' in isolated_user_config.read_text(encoding="utf-8")

        session.handle_line("/help")
        assert any("Comandos de Pedacito TUI" in text for _r, text in session.transcript)

        session.handle_line("/lang en")
        assert session.lang == "en"
        assert 'language = "en"' in isolated_user_config.read_text(encoding="utf-8")
    finally:
        session.close()


# --------------------------------------------------------------------- CLI --

def test_cli_lang_flag_and_env_help(isolated_user_config, capsys, monkeypatch):
    # `pdc --lang es --help`
    with pytest.raises(SystemExit) as exc:
        cli.main(["--lang", "es", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "asistente de programación" in out
    assert "ejecutar una tarea" in out

    # `pdc ask --lang es --help`
    with pytest.raises(SystemExit) as exc:
        cli.main(["ask", "--lang", "es", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Ejecuta una tarea de programación" in out
    assert "escribir los cambios en disco" in out

    # `PEDACITO_LANG=es pdc --help`
    monkeypatch.setenv("PEDACITO_LANG", "es")
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "asistente de programación" in out
