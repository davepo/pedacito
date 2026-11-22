"""The GUI's two theme families must paint as one reconciled choice.

The nine CSS-native ``gui_theme`` schemes carry full crafted palettes in
app.css (ansi colors, terminal, code and diff surfaces included); the unified
engine (themes.json) carries presets and custom palettes as inline ten-token
derivations. Regression for the rollback where the engine's unconditional
boot application flattened every crafted scheme into the active preset's
generic derivation -- a config with ``gui_theme = "jazz_solo"`` painted as
whatever preset sat in themes.json -- and a Settings gui_theme edit never
repainted while an engine palette was active.

The live checks drive the real GUI in headless Chromium and are skipped when
Playwright is absent; the text pins run everywhere.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pedacito import config as config_mod
from pedacito import domcheck, themes
from pedacito.gui import server
from test_gui import gui as _shared_gui  # noqa: F401  (pytest re-export)

STATIC = Path(server.__file__).resolve().parent / "static"
ENUM = list(config_mod._ENUM_FIELDS["gui_theme"])


@pytest.fixture
def gui(_shared_gui):  # noqa: F811  (re-export the fixture imported above)
    return _shared_gui


def _scheme_list(src):
    match = re.search(r"CSS_SCHEMES\s*=\s*\[(.*?)\]", src, re.S)
    assert match, "CSS_SCHEMES list not found"
    return re.findall(r"'([^']+)'", match.group(1))


# --------------------------------------------------------------- text pins ----

def test_scheme_lists_match_the_config_enum():
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    theme = (STATIC / "theme.js").read_text(encoding="utf-8")
    assert _scheme_list(app) == ENUM
    assert _scheme_list(theme) == ENUM
    # every scheme carries a label and swatch for the Theme Manager grid
    meta = re.search(r"CSS_SCHEME_META\s*=\s*\{(.*?)\n\};", app, re.S)
    assert meta, "CSS_SCHEME_META not found"
    for name in ENUM:
        assert re.search(r"\b" + name + r":\s*\{", meta.group(1)), name


def test_css_scheme_selection_clears_the_engine_tokens():
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    body = app.split("function applyTheme(t, userPick) {", 1)[1]
    body = body.split("\n}", 1)[0]
    assert "clearThemeTokens();" in body
    # boot resolution defers to a CSS-native marker or server attribute
    assert "function applyThemePayload(payload, explicitEngine)" in app
    assert "if (isCssScheme(marker))" in app
    assert "if (!marker && isCssScheme(attr))" in app
    # a built-in preset paints as its crafted CSS block, not a derivation
    assert "if (isCssScheme(active))" in app


def test_theme_grid_does_not_duplicate_the_builtin_schemes():
    # The engine presets ARE the crafted schemes: one pill per name, the
    # engine list only adds custom palettes on top.
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    assert ".filter((n) => st.cssSchemes.indexOf(n) < 0)" in app


def test_settings_theme_edits_repaint_after_the_first_poll():
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    assert ("const follow = lastServerTheme === null ? local === 'slate' : true;"
            in app)


def test_prepaint_skips_the_engine_cache_for_css_schemes():
    theme = (STATIC / "theme.js").read_text(encoding="utf-8")
    assert "var serverCss = !t && attr && CSS_SCHEMES.indexOf(attr) >= 0;" in theme
    assert "if (!cssNative && !serverCss)" in theme


def test_themes_payload_offers_the_css_schemes(gui, tmp_path, monkeypatch):
    monkeypatch.setattr(themes, "THEMES_FILE", tmp_path / "cfg" / "themes.json")
    status, body = gui.get("/api/themes")
    assert status == 200 and body["ok"] is True
    assert body["css_schemes"] == ENUM


def test_gui_theme_survives_unrelated_config_error(gui, tmp_path, monkeypatch):
    """A broken `default_profile` (deleted profile) makes `config.load()` raise
    `ConfigError`, which previously caused `user_theme()` to fall back to
    `slate` and reset the GUI theme on every poll/reload."""
    from pedacito.gui import core as gui_core

    cfg_path = tmp_path / "broken_user_config.toml"
    cfg_path.write_text(
        '[universal]\n'
        'default_profile = "deleted-profile-name"\n'
        'gui_theme = "jazz_solo"\n'
        'language = "es"\n'
        '\n'
        '[profiles.real]\n'
        'model = "qwen3-8b"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(config_mod, "USER_CONFIG", cfg_path)
    monkeypatch.setattr(gui_core, "USER_CONFIG", cfg_path)
    monkeypatch.setattr(themes, "THEMES_FILE", tmp_path / "themes.json")
    with gui_core.STATE.lock:
        gui_core.STATE.cfg = None
        gui_core.STATE._user_cfg = None
        gui_core.STATE._user_cfg_stamp = None
        gui_core.STATE._user_raw_prefs = None

    assert gui_core.STATE.user_theme() == "jazz_solo"
    assert gui_core.STATE.user_language() == "es"
    status, state = gui.get("/api/state")
    assert status == 200
    assert state["gui_theme"] == "jazz_solo"
    html = server.index_html(b'<!doctype html><html lang="en"></html>').decode("utf-8")
    assert 'data-theme="jazz_solo"' in html


def test_gui_theme_falls_back_to_themes_json_builtin_when_unset(
        gui, tmp_path, monkeypatch):
    """If `config.toml` has no explicit `gui_theme`, a built-in scheme saved in
    `themes.json` (e.g. via TUI `/theme jazz_solo`) is honored instead of
    resetting to `slate`."""
    from pedacito.gui import core as gui_core

    cfg_path = tmp_path / "no_theme_config.toml"
    cfg_path.write_text(
        '[universal]\nvram_gb = 16\n\n[profiles.default]\nmodel = "qwen3-8b"\n',
        encoding="utf-8",
    )
    tfile = tmp_path / "themes.json"
    tfile.write_text(json.dumps({"active_theme": "jazz_solo", "custom_themes": {}}),
                     encoding="utf-8")
    monkeypatch.setattr(config_mod, "USER_CONFIG", cfg_path)
    monkeypatch.setattr(gui_core, "USER_CONFIG", cfg_path)
    monkeypatch.setattr(themes, "THEMES_FILE", tfile)
    with gui_core.STATE.lock:
        gui_core.STATE.cfg = None
        gui_core.STATE._user_cfg = None
        gui_core.STATE._user_cfg_stamp = None
        gui_core.STATE._user_raw_prefs = None

    assert gui_core.STATE.user_theme() == "jazz_solo"


# ------------------------------------------------------- live browser pins ----

requires_pw = pytest.mark.skipif(
    not domcheck.available()[0], reason="Playwright + Chromium not installed")

_READ_VARS = """() => {
  const cs = getComputedStyle(document.documentElement);
  const inline = document.documentElement.style;
  return {
    bg: cs.getPropertyValue('--bg').trim(),
    term: cs.getPropertyValue('--term-bg').trim(),
    ansi5: cs.getPropertyValue('--ansi-5').trim(),
    inlineBg: inline.getPropertyValue('--bg'),
    attr: document.documentElement.getAttribute('data-theme'),
  };
}"""


def _wait_scheme(pg, name):
    pg.wait_for_function(
        "(n) => localStorage.getItem('pedacito_theme') === n",
        arg=name, timeout=15000)


def _launch(gui):
    from playwright.sync_api import sync_playwright
    try:
        pw = sync_playwright().start()
        chromium = pw.chromium.launch(headless=True)
    except Exception as e:                       # pragma: no cover - env
        pytest.skip(f"Chromium is unavailable: {e}")
    pg = chromium.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(f"http://127.0.0.1:{gui.port}/")
    return pg, errors, chromium, pw


@pytest.fixture
def page(gui, tmp_path, monkeypatch):
    """The reported broken state: gui_theme=jazz_solo in the user config and
    a different (custom) palette active in the engine store, plus a headless
    browser."""
    if not domcheck.available()[0]:
        pytest.skip("Playwright + Chromium not installed")
    monkeypatch.setattr(themes, "THEMES_FILE", tmp_path / "cfg" / "themes.json")
    (tmp_path / "cfg").mkdir(exist_ok=True)
    (tmp_path / "cfg" / "themes.json").write_text(json.dumps({
        "active_theme": "engine-pick",
        "custom_themes": {"engine-pick": {
            "bg": "#1e1e2e", "surface": "#181825", "surface_alt": "#313244",
            "text": "#cdd6f4", "muted": "#a6adc8", "accent": "#cba6f7",
            "border": "#45475a", "success": "#a6e3a1", "warning": "#fab387",
            "danger": "#f38ba8",
        }},
    }), encoding="utf-8")
    status, body = gui.post("/api/settings/theme", {"gui_theme": "jazz_solo"})
    assert status == 200 and body["ok"]
    pg, errors, chromium, pw = _launch(gui)
    try:
        yield pg, errors
    finally:
        chromium.close()
        pw.stop()


@pytest.fixture
def plain_page(gui, tmp_path, monkeypatch):
    """A fresh install: no user config (gui_theme defaults to slate) and no
    engine store (the active palette defaults to slate)."""
    if not domcheck.available()[0]:
        pytest.skip("Playwright + Chromium not installed")
    monkeypatch.setattr(themes, "THEMES_FILE", tmp_path / "cfg" / "themes.json")
    pg, errors, chromium, pw = _launch(gui)
    try:
        yield pg, errors
    finally:
        chromium.close()
        pw.stop()


@requires_pw
def test_crafted_gui_theme_survives_the_engine_and_repaints(page, gui):
    pg, errors = page

    # 1. The crafted jazz_solo palette paints -- not the engine store's
    #    active derivation: no inline tokens, full crafted terminal/ansi
    #    surfaces.
    _wait_scheme(pg, "jazz_solo")
    v = pg.evaluate(_READ_VARS)
    assert v["attr"] == "jazz_solo"
    assert v["inlineBg"] == ""
    assert v["bg"] == "#f4f1ea"
    assert v["term"] == "#fbf8f0"
    assert v["ansi5"] == "#8e24aa"

    # 1b. A browser coming from the broken version carries a stale engine
    #     cache: it must not overpaint the crafted scheme, and must be cleared.
    pg.evaluate("""() => localStorage.setItem('pedacito_theme_css',
        JSON.stringify({'--bg': '#1e1e2e', '--term-bg': '#1e1e2e',
                        '--ansi-5': '#cba6f7'}))""")
    pg.reload()
    _wait_scheme(pg, "jazz_solo")
    v = pg.evaluate(_READ_VARS)
    assert v["inlineBg"] == ""
    assert v["bg"] == "#f4f1ea"
    assert pg.evaluate(
        "() => localStorage.getItem('pedacito_theme_css')") is None

    # 2. An explicit Theme Manager custom-palette pick applies inline tokens
    #    and survives a reload. (Built-in presets are the crafted schemes
    #    themselves; only custom palettes paint as engine derivations.)
    status, body = gui.post("/api/themes/save", {
        "name": "tokyo-custom",
        "tokens": {"bg": "#1a1b26", "surface": "#16161e",
                   "surface_alt": "#24283b", "text": "#c0caf5",
                   "muted": "#565f89", "accent": "#7aa2f7",
                   "border": "#292e42", "success": "#9ece6a",
                   "warning": "#e0af68", "danger": "#f7768e"},
    })
    assert status == 200 and body["ok"]
    pg.click("#themeBtn")
    pg.wait_for_selector('.theme-pill[title="tokyo-custom"]', timeout=10000)
    # Each built-in scheme appears exactly once -- no second engine pill.
    assert pg.evaluate(
        "() => document.querySelectorAll('.theme-pill[title=\"nord\"]').length"
    ) == 1
    pg.click('.theme-pill[title="tokyo-custom"]')
    _wait_scheme(pg, "tokyo-custom")
    v = pg.evaluate(_READ_VARS)
    assert v["inlineBg"] == "#1a1b26"
    assert v["bg"] == "#1a1b26"
    assert v["attr"] is None
    pg.reload()
    _wait_scheme(pg, "tokyo-custom")
    v = pg.evaluate(_READ_VARS)
    assert v["bg"] == "#1a1b26"
    assert v["attr"] is None

    # 3. A server-side Settings gui_theme edit repaints live even while the
    #    engine palette is active (the poll follows real server changes).
    status, body = gui.post("/api/settings/theme", {"gui_theme": "midnight"})
    assert status == 200 and body["ok"]
    _wait_scheme(pg, "midnight")
    v = pg.evaluate(_READ_VARS)
    assert v["bg"] == "#05070a"
    assert v["inlineBg"] == ""
    assert v["attr"] == "midnight"

    # 4. The Theme Manager's crafted-scheme pills switch back and persist.
    pg.click("#themeBtn")
    pg.wait_for_selector('.theme-pill[title="jazz_solo"]', timeout=10000)
    pg.click('.theme-pill[title="jazz_solo"]')
    _wait_scheme(pg, "jazz_solo")
    pg.reload()
    _wait_scheme(pg, "jazz_solo")
    v = pg.evaluate(_READ_VARS)
    assert v["bg"] == "#f4f1ea"
    assert v["term"] == "#fbf8f0"
    assert v["inlineBg"] == ""
    assert v["attr"] == "jazz_solo"
    assert not errors, errors


@requires_pw
def test_default_slate_paints_the_crafted_palette_not_the_derivation(plain_page):
    pg, errors = plain_page
    # No gui_theme preference and the default preset: the crafted slate block
    # paints, not the engine's flattened derivation (terminal = page bg,
    # ansi blue/purple/cyan all = accent). Either boot order -- the engine
    # resolution or the first state poll -- lands on the same crafted CSS.
    pg.wait_for_function(
        "() => localStorage.getItem('pedacito_theme') !== null", timeout=15000)
    v = pg.evaluate(_READ_VARS)
    assert v["inlineBg"] == ""
    assert v["attr"] is None
    assert v["bg"] == "#0d1117"
    assert v["term"] == "#010409"
    assert v["ansi5"] == "#d2a8ff"
    assert not errors, errors
