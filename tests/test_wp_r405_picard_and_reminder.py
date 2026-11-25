"""WP-R405: Captain Picard Mode (`picard_mode`) and Active-Development Update
Reminder (`show_update_reminder` + `update.active_dev_reminder()`).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pedacito import __version__
from pedacito import config as config_mod
from pedacito import tui as tui_mod
from pedacito import update as update_mod
from pedacito.config import Config, ConfigError
from pedacito.gui import settings as gui_settings
from pedacito.tui import TuiSession, run_tui
from test_gui import gui as _shared_gui  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def gui(_shared_gui):  # noqa: F811
    return _shared_gui


# --- Feature 1: Captain Picard Mode -------------------------------------------

def test_picard_mode_config_defaults_and_coercion(tmp_path, monkeypatch):
    cfg = Config()
    assert cfg.picard_mode is False
    assert cfg.show_update_reminder is True

    # Commented defaults exist in TEMPLATE and REFERENCE_TEMPLATE
    assert "# picard_mode = false" in config_mod.TEMPLATE
    assert "# picard_mode = false" in config_mod.REFERENCE_TEMPLATE
    assert "# show_update_reminder = true" in config_mod.TEMPLATE
    assert "# show_update_reminder = true" in config_mod.REFERENCE_TEMPLATE

    # Concise help text is present in schema
    schema_by_name = {item["name"]: item for item in gui_settings.schema()}
    assert "picard_mode" in schema_by_name
    assert "Make it so." in schema_by_name["picard_mode"]["help"]
    assert "show_update_reminder" in schema_by_name

    # Coercion and TOML loading
    assert config_mod._coerce("picard_mode", True) is True
    assert config_mod._coerce("picard_mode", False) is False
    with pytest.raises(ConfigError):
        config_mod._coerce("picard_mode", "yes")

    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        "[universal]\npicard_mode = true\nshow_update_reminder = false\n",
        encoding="utf-8",
    )
    loaded = config_mod.load(
        root=tmp_path, explicit=cfg_file,
        require_hardware=False, warn_budgets=False,
        upgrade_user_config=False,
    )
    assert loaded.picard_mode is True
    assert loaded.show_update_reminder is False

    # Env var override
    monkeypatch.setenv("PEDACITO_PICARD_MODE", "false")
    monkeypatch.setenv("PEDACITO_SHOW_UPDATE_REMINDER", "true")
    loaded_env = config_mod.load(
        root=tmp_path, explicit=cfg_file,
        require_hardware=False, warn_budgets=False,
        upgrade_user_config=False,
    )
    assert loaded_env.picard_mode is False
    assert loaded_env.show_update_reminder is True


def test_tui_picard_slash_command_header_and_confirm_cues(tmp_path):
    cfg = Config()
    cfg.model = "qwen3-coder"
    cfg.active_profile = "default"
    session = TuiSession(tmp_path, cfg, headless=True, stdout=io.StringIO())
    try:
        assert "/picard" in tui_mod._SLASH_COMMANDS
        assert "/picard" in tui_mod.HELP_TEXT
        assert "/picard" in tui_mod.HELP_TEXT_ES

        # Default off: no [Picard] indicator
        frame_off = session.render_frame(120, 32)
        assert "[Picard]" not in frame_off
        assert "Enter: send" in frame_off

        # Tab completion for /pic -> /picard
        session.input_buf = "/pic"
        session.cursor_pos = 4
        session._complete()
        assert session.input_buf == "/picard "
        session.input_buf = ""
        session.cursor_pos = 0

        # Turn on via /picard on
        session.handle_line("/picard on")
        assert session.cfg.picard_mode is True
        transcript = "\n".join(line for _k, line in session.transcript)
        assert "Make it so." in transcript

        frame_on = session.render_frame(120, 32)
        assert "[Picard]" in frame_on
        assert "Enter: Make it so." in frame_on

        # Confirm prompt relabels to Make it so? [y/N]
        session.stdin = io.StringIO("n\n")
        ans = session._confirm("Upgrade to 9.9.9 now? (y/N):")
        assert ans is False
        assert any("Make it so? [y/N]" in line for _k, line in session.transcript)

        # Turn off via /picard off
        session.handle_line("/picard off")
        assert session.cfg.picard_mode is False
        frame_off_again = session.render_frame(120, 32)
        assert "[Picard]" not in frame_off_again

        # Toggle without argument
        session.handle_line("/picard")
        assert session.cfg.picard_mode is True
        session.handle_line("/picard")
        assert session.cfg.picard_mode is False
    finally:
        session.close()


def test_gui_state_and_settings_expose_picard_mode_and_reminder(gui):
    status, st = gui.get("/api/state")
    assert status == 200
    assert st["picard_mode"] is False
    assert st["show_update_reminder"] is True

    # Create user config and toggle picard_mode via /api/settings/set
    assert gui.post("/api/settings/setup", {
        "base_url": "http://localhost:1234/v1",
        "model": "qwen3-8b",
        "vram_gb": "8",
    })[0] == 200

    status, res = gui.post("/api/settings/set", {
        "file": "user",
        "profile": None,
        "changes": {"picard_mode": True, "show_update_reminder": False},
        "removes": [],
    })
    assert status == 200 and res["ok"] is True
    assert res["picard_mode"] is True
    assert res["show_update_reminder"] is False
    assert res["resolved"]["picard_mode"]["value"] is True
    assert res["resolved"]["show_update_reminder"]["value"] is False

    status, st2 = gui.get("/api/state")
    assert status == 200
    assert st2["picard_mode"] is True
    assert st2["show_update_reminder"] is False


def test_gui_static_picard_text_and_reminder_wiring():
    app_js = (ROOT / "pedacito" / "gui" / "static" / "app.js").read_text(encoding="utf-8")
    settings_js = (ROOT / "pedacito" / "gui" / "static" / "settings.js").read_text(encoding="utf-8")

    # Helper definition & tooltip semantic preservation
    assert "PDC.picardText" in app_js
    assert "PDC.picardTitle" in app_js
    assert "'Make it so.'" in app_js

    # Primary buttons wired through picardText
    assert "PDC.picardText(sendSem)" in app_js          # Composer Run / Send
    assert "PDC.picardText(semLabel)" in app_js         # PDC.confirm & PDC.prompt
    assert "PDC.picardText(acceptLabel)" in app_js      # Review Apply / Accept
    assert "PDC.picardText('Commit')" in app_js         # Git Drawer Commit
    assert "PDC.picardText('Allow Once')" in app_js     # Permission Broker Allow

    # Settings Save changes / Create config / Trust buttons
    assert "picardText('Create config')" in settings_js
    assert "picardText(trustLbl)" in settings_js
    assert "picardText('Validate & save')" in settings_js
    assert "picardText(sem)" in settings_js

    # Subtle dismissible reminder in Welcome chat area and Settings -> Updates
    assert "activeDevReminder" in app_js
    assert "pedacito_update_reminder_dismissed_at" in app_js
    assert "gui.update.active_dev_tip" in app_js
    assert "gui.update.active_dev_tip" in settings_js


# --- Feature 2: Active-Development Update Reminder ---------------------------

def test_active_dev_reminder_normal_and_cached_update(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg" / "config.toml"
    monkeypatch.setattr(config_mod, "USER_CONFIG", cfg_path)

    cfg = Config()
    # Normal active-dev tip when no cache exists (or updates disabled)
    en_tip = update_mod.active_dev_reminder(cfg, lang="en")
    assert en_tip == (
        "Tip: Pedacito is in active development with frequent bug fixes and "
        "updates -- run `pedacito update` regularly to stay current."
    )
    es_tip = update_mod.active_dev_reminder(cfg, lang="es")
    assert es_tip == (
        "Consejo: Pedacito está en desarrollo activo con correcciones y mejoras "
        "frecuentes -- ejecuta `pedacito update` con regularidad para mantenerte "
        "al día."
    )

    # Cached newer version in update_check.json when updates are enabled
    monkeypatch.setenv("PEDACITO_CHECK_FOR_UPDATES", "true")
    update_mod.write_cache("9.9.9")
    en_avail = update_mod.active_dev_reminder(cfg, lang="en")
    assert f"{__version__} -> 9.9.9" in en_avail
    assert "pedacito update --run" in en_avail

    es_avail = update_mod.active_dev_reminder(cfg, lang="es")
    assert f"{__version__} -> 9.9.9" in es_avail
    assert "pedacito update --run" in es_avail

    # Opt-out via show_update_reminder = False returns ""
    cfg.show_update_reminder = False
    assert update_mod.active_dev_reminder(cfg, lang="en") == ""
    assert update_mod.active_dev_reminder(cfg, lang="es") == ""


def test_headless_tui_never_emits_exit_reminder(tmp_path):
    cfg = Config()
    cfg.show_update_reminder = True
    out = io.StringIO()
    code = run_tui(
        tmp_path, cfg,
        headless=True,
        stdin=io.StringIO("/quit\n"),
        stdout=out,
    )
    assert code == 0
    text = out.getvalue()
    assert "active development" not in text
    assert "desarrollo activo" not in text
