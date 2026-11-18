"""Shared color-scheme engine, theme persistence and both front ends.

The tests isolate ``themes.THEMES_FILE`` in a temp directory, so neither the
developer's real ``~/.config/pedacito/themes.json`` nor a project's config is
touched.
"""
from __future__ import annotations

import http.client
import io
import json
import re
import threading

import pytest

from pedacito import config as config_mod
from pedacito import themes
from pedacito.config import Config
from pedacito.gui import core, server
from pedacito.tui import TuiSession, run_tui

HEX = re.compile(r"^#[0-9a-f]{6}$")


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A private theme store and config path for every test."""
    path = tmp_path / "cfg" / "themes.json"
    monkeypatch.setattr(themes, "THEMES_FILE", path)
    monkeypatch.setattr(config_mod, "USER_CONFIG",
                        tmp_path / "cfg" / "config.toml")
    monkeypatch.delenv("PEDACITO_THEMES_FILE", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    return path


def _tokens(overrides=None):
    tokens = dict(themes.BUILTIN_THEMES["pedacito-dark"])
    tokens.update(overrides or {})
    return tokens


class _Client:
    def __init__(self, port: int):
        self.port = port

    def _req(self, method: str, path: str, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {"Content-Type": "application/json"}
        data = json.dumps(body) if body is not None else None
        conn.request(method, path, body=data, headers=headers)
        response = conn.getresponse()
        payload = response.read()
        status = response.status
        conn.close()
        return status, json.loads(payload or b"{}")

    def get(self, path: str):
        return self._req("GET", path)

    def post(self, path: str, body=None):
        return self._req("POST", path, body or {})


@pytest.fixture
def gui(store, tmp_path):
    core.STATE.__init__()
    core.STATE.root = tmp_path / "project"
    core.STATE.root.mkdir()
    srv = server.create_server(0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield _Client(srv.server_address[1])
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(5)
        core.STATE.__init__()


# --------------------------------------------------------------- built-ins

def test_six_builtin_presets_are_complete_valid_hex_palettes():
    assert len(themes.BUILTIN_THEMES) >= 6
    expected = {
        "pedacito-dark", "catppuccin-mocha", "tokyo-night",
        "gruvbox-dark", "nord", "solarized-dark",
    }
    assert expected <= set(themes.BUILTIN_THEMES)
    for name, tokens in themes.BUILTIN_THEMES.items():
        assert set(tokens) == set(themes.THEME_KEYS), name
        for key in themes.THEME_KEYS:
            assert HEX.fullmatch(tokens[key]), (name, key, tokens[key])
    assert themes.BUILTIN_THEMES["catppuccin-mocha"]["bg"] == "#1e1e2e"
    assert themes.BUILTIN_THEMES["tokyo-night"]["bg"] == "#1a1b26"
    assert themes.BUILTIN_THEMES["nord"]["bg"] == "#2e3440"
    assert themes.BUILTIN_THEMES["solarized-dark"]["bg"] == "#002b36"
    assert themes.BUILTIN_THEMES["gruvbox-dark"]["bg"] == "#282828"


# --------------------------------------------------------- custom store ----


def test_save_custom_theme_normalizes_and_persists(store):
    saved = themes.save_custom_theme("my-theme", _tokens({
        "bg": "#abc", "surface": "#FED", "accent": "12f",
    }))
    assert saved["bg"] == "#aabbcc"
    assert saved["surface"] == "#ffeedd"
    assert saved["accent"] == "#1122ff"
    assert store.is_file()
    written = json.loads(store.read_text(encoding="utf-8"))
    assert written["custom_themes"]["my-theme"]["bg"] == "#aabbcc"

    data = themes.list_themes()
    assert data["themes"]["my-theme"] == saved
    assert data["custom_names"] == ["my-theme"]
    assert "my-theme" not in data["builtin_names"]

    # The active selection is written beside the custom palettes.
    themes.set_active_theme("my-theme")
    assert themes.get_active_theme_name() == "my-theme"
    reloaded = json.loads(store.read_text(encoding="utf-8"))
    assert reloaded["active_theme"] == "my-theme"
    assert themes.get_active_theme()["bg"] == "#aabbcc"


def test_save_custom_theme_rejects_bad_tokens_and_builtin_names(store):
    with pytest.raises(ValueError, match="missing color token"):
        themes.save_custom_theme("short", {"bg": "#000000"})
    with pytest.raises(ValueError, match="unknown theme color token"):
        themes.save_custom_theme("extra", _tokens({"nope": "#000000"}))
    with pytest.raises(ValueError, match="invalid hex color"):
        themes.save_custom_theme("bad", _tokens({"bg": "#12345"}))
    with pytest.raises(ValueError, match="invalid hex color"):
        themes.save_custom_theme("bad", _tokens({"accent": "not-a-color"}))
    with pytest.raises(ValueError, match="built-in"):
        themes.save_custom_theme("nord", _tokens())
    with pytest.raises(ValueError, match="theme name is required"):
        themes.save_custom_theme("  ", _tokens())
    assert not store.exists()


def test_delete_protects_builtins_and_resets_the_active_theme(store):
    themes.save_custom_theme("temporary", _tokens())
    themes.set_active_theme("temporary")
    assert themes.delete_custom_theme("nord") is False
    assert themes.delete_custom_theme("Nord") is False
    assert themes.delete_custom_theme("missing") is False
    assert themes.delete_custom_theme("Temporary") is True
    assert "temporary" not in themes.list_themes()["themes"]
    assert themes.get_active_theme_name() == themes.DEFAULT_THEME


def test_project_config_override_wins_and_preserves_other_keys(store, tmp_path):
    project = tmp_path / "project"
    cfg_path = project / ".pedacito" / "config.json"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(json.dumps({"theme": "nord", "keep": {"x": 1}}),
                        encoding="utf-8")
    assert themes.get_active_theme_name(project) == "nord"
    assert themes.get_active_theme(project)["bg"] == "#2e3440"

    themes.set_active_theme("gruvbox-dark", project)
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert data["theme"] == "gruvbox-dark"
    assert data["keep"] == {"x": 1}
    assert themes.get_active_theme_name(project) == "gruvbox-dark"
    # The global selection is untouched while a project override exists.
    assert themes.get_active_theme_name() == themes.DEFAULT_THEME

    assert themes.clear_project_theme(project) is True
    assert themes.get_active_theme_name(project) == themes.DEFAULT_THEME
    assert themes.clear_project_theme(project) is False


# ------------------------------------------------------------------ ANSI ----


def test_hex_to_rgb_and_truecolor_sequences():
    assert themes.hex_to_rgb("#abc") == (170, 187, 204)
    assert themes.hex_to_rgb("#FF0000") == (255, 0, 0)
    assert themes.ansi_fg("#ff0000") == "\x1b[38;2;255;0;0m"
    assert themes.ansi_bg("#336699") == "\x1b[48;2;51;102;153m"


def test_ansi_256_fallbacks_and_style_text():
    assert themes.ansi_fg("#ff0000", truecolor=False) == "\x1b[38;5;196m"
    assert themes.ansi_bg("#ffffff", truecolor=False) == "\x1b[48;5;231m"
    styled = themes.style_text("hello", fg="#ff0000", bg="#000000", bold=True)
    assert styled == ("\x1b[1m\x1b[38;2;255;0;0m"
                      "\x1b[48;2;0;0;0mhello\x1b[0m")
    assert themes.style_text("plain") == "plain"


def test_no_color_disables_styling(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert themes.style_text("plain", fg="#ff0000", bold=True) == "plain"


def test_truecolor_detection_reads_colorterm_and_term(monkeypatch):
    assert themes.supports_truecolor({"COLORTERM": "truecolor", "TERM": "xterm"})
    assert themes.supports_truecolor({"COLORTERM": "24bit", "TERM": "xterm"})
    assert themes.supports_truecolor({"TERM": "xterm-truecolor"})
    assert not themes.supports_truecolor({"TERM": "xterm-256color"})
    monkeypatch.setenv("COLORTERM", "truecolor")
    assert themes.ansi_fg("#00ff00", truecolor=None) == "\x1b[38;2;0;255;0m"
    monkeypatch.delenv("COLORTERM")
    monkeypatch.setenv("TERM", "xterm-256color")
    assert themes.ansi_fg("#00ff00", truecolor=None) == "\x1b[38;5;46m"


# ------------------------------------------------------------------- GUI ----


def test_gui_theme_endpoints(gui, store):
    status, body = gui.get("/api/themes")
    assert status == 200 and body["ok"] is True
    assert body["active_theme"] == themes.DEFAULT_THEME
    assert set(themes.BUILTIN_THEMES) <= set(body["builtin_names"])
    assert set(themes.THEME_KEYS) == set(body["themes"]["nord"])

    status, body = gui.post("/api/themes/select", {"name": "nord"})
    assert status == 200 and body["active_theme"] == "nord"
    assert gui.get("/api/themes")[1]["active_theme"] == "nord"

    custom = _tokens({"bg": "#abc", "accent": "#0f0"})
    status, body = gui.post("/api/themes/save",
                            {"name": "gui-custom", "tokens": custom})
    assert status == 200 and body["active_theme"] == "gui-custom"
    assert body["themes"]["gui-custom"]["bg"] == "#aabbcc"
    assert body["themes"]["gui-custom"]["accent"] == "#00ff00"
    assert store.is_file()

    status, body = gui.post("/api/themes/delete", {"name": "gui-custom"})
    assert status == 200 and body["active_theme"] == themes.DEFAULT_THEME
    assert "gui-custom" not in body["themes"]

    assert gui.post("/api/themes/delete", {"name": "nord"})[0] == 400
    assert gui.post("/api/themes/select", {"name": "nope"})[0] == 400
    bad = _tokens({"bg": "#zzzzzz"})
    assert gui.post("/api/themes/save",
                    {"name": "bad", "tokens": bad})[0] == 400


# ------------------------------------------------------------------- TUI ----


def test_tui_theme_command_lists_and_switches(store, tmp_path):
    script = "\n".join(["/theme", "/theme nord", "/theme", "/quit"]) + "\n"
    out = io.StringIO()
    code = run_tui(tmp_path, Config(), headless=True,
                   stdin=io.StringIO(script), stdout=out)
    assert code == 0
    text = out.getvalue()
    assert "Color themes" in text
    for name in ("pedacito-dark", "catppuccin-mocha", "tokyo-night",
                 "gruvbox-dark", "nord", "solarized-dark"):
        assert name in text
    assert "* nord" in text
    assert "Theme: nord (saved)" in text
    assert themes.get_active_theme_name() == "nord"
    assert themes.get_active_theme()["bg"] == "#2e3440"

    # The same command reports an unknown name without changing the choice.
    out = io.StringIO()
    run_tui(tmp_path, Config(), headless=True,
            stdin=io.StringIO("/theme not-a-theme\n/quit\n"), stdout=out)
    assert "unknown theme" in out.getvalue().lower()
    assert themes.get_active_theme_name() == "nord"


def test_tui_uses_active_palette_and_respects_no_color(store, tmp_path,
                                                       monkeypatch):
    themes.set_active_theme("nord")
    session = TuiSession(tmp_path, Config(), headless=True, color=True,
                         width=100, height=28, stdout=io.StringIO())
    try:
        frame = session.render_frame(100, 28)
        # Nord's accent is #88c0d0: the prompt/header/banner use 24-bit SGR.
        assert "\x1b[38;2;136;192;208m" in frame
        assert "\x1b[1m" in frame
    finally:
        session.close()

    monkeypatch.setenv("NO_COLOR", "1")
    session = TuiSession(tmp_path, Config(), headless=True, color=True,
                         width=100, height=28, stdout=io.StringIO())
    try:
        assert "\x1b[" not in session.render_frame(100, 28)
    finally:
        session.close()
