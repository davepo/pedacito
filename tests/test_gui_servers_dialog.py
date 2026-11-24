"""The Servers & Memory dialog titles rows after the server, not a profile.

Every profile pointing at one local endpoint shares a single registry row
(``servers.configured_local_servers``). The row used to be titled after the
first profile that pointed there -- with model-named profiles that read like
a model sitting unloaded on the server ("is that supposed to say Lemonade
Server?"). The title is now kind + host, the profile names ride as a
secondary line, and a hint explains that load/unload is a server-side
operation that creates no profile. The actions must keep routing by the
registry name (``s.name``), never by the display title.

The live check drives the real GUI in headless Chromium and is skipped when
Playwright is absent; the text pins run everywhere. No real model server is
contacted: the registry, the hardware snapshot and the load call are fakes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pedacito import domcheck, profiler as profiler_mod, servers  # noqa: E402
from test_gui import gui as _shared_gui  # noqa: F401  (pytest re-export)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def gui(_shared_gui):  # noqa: F811  (re-export the fixture imported above)
    return _shared_gui


def _dialog_src():
    src = (ROOT / "pedacito" / "gui" / "static" / "app.js").read_text(
        encoding="utf-8")
    return src[src.index("function serversDialog"):
               src.index("\nfunction procsDialog")]


def _memory():
    return {
        "ram": {"total_gb": 64.0, "used_gb": 32.0, "percent": 50.0},
        "vram": {"total_gb": 24.0, "used_gb": 20.0, "percent": 83.3,
                 "cards": []},
        "ram_total_gb": 64.0, "ram_used_gb": 32.0,
        "vram_total_gb": 24.0, "vram_used_gb": 20.0,
        "available": True,
    }


def _lemonade_state():
    """The reported setup: many model-named profiles share one Lemonade URL,
    so the registry collapses them into a single row named after the first
    profile. The live catalog also carries a model no profile references."""
    return {
        "name": "qwen3_8-27b-gsq-rco",
        "kind": "lemonade", "provider": "lemonade",
        "base_url": "http://127.0.0.1:11434/v1",
        "online": True,
        "models": ["Qwen3.8-27B-GSQ-RCO-GGUF",
                   "Qwen3-Coder-30B-A3B-Instruct-GGUF",
                   "LFM2.5-24B-A2B-Turbo-Brilliance-GGUF",
                   "Downloaded-But-Unprofiled-GGUF"],
        "loaded_models": [], "loading_models": [],
        "profiles": ["qwen3_8-27b-gsq-rco", "qwen3_8-27b",
                     "qwen3-coder-30b-a3b-instruct",
                     "lfm2_5-24b-a2b-turbo-brilliance", "qwen-qwen3_5-9b"],
        "active": False,
    }


# --------------------------------------------------------------- text pins ----

def test_row_title_is_the_server_not_the_first_profile():
    dialog = _dialog_src()
    assert "function serverTitle(s)" in dialog
    assert "h('b', { text: serverTitle(s) })" in dialog
    # The old profile-name title is gone.
    assert "text: s.name || '?'" not in dialog
    assert "lemonade: 'Lemonade'" in dialog


def test_profile_names_ride_as_a_secondary_line():
    dialog = _dialog_src()
    assert "s.profiles.slice(0, 3).join(', ')" in dialog
    assert "' + ' + (s.profiles.length - 3) + ' more'" in dialog
    # The full list stays reachable as a tooltip.
    assert "title: s.profiles.join(', ')" in dialog


def test_load_unload_pin_still_route_by_the_registry_name():
    # The relabel is display-only: every action must keep sending the
    # registry key (s.name), which _resolve_server matches -- never the
    # kind + host display title. Load, unload, Pin and Unpin: four payloads.
    dialog = _dialog_src()
    assert dialog.count("server: s.name") == 4


def test_dialog_explains_that_load_creates_no_profile():
    dialog = _dialog_src()
    assert "creates no profile and switches nothing" in dialog
    assert "live catalog the server reports" in dialog
    assert "including models no profile references" in dialog


def test_auth_required_servers_show_the_key_hint():
    dialog = _dialog_src()
    assert "s.auth_required === true" in dialog
    assert "API key required" in dialog
    assert ("Set api_key or STRATA_API_KEY in Settings -> Profiles to "
            "load/unload models") in dialog


# ------------------------------------------------------- live browser pin ----

requires_pw = pytest.mark.skipif(
    not domcheck.available()[0], reason="Playwright + Chromium not installed")


@requires_pw
def test_dialog_titles_the_server_and_load_routes_by_registry_name(
        gui, monkeypatch):
    monkeypatch.setattr(profiler_mod, "snapshot",
                        lambda *a, **k: _memory())
    monkeypatch.setattr(servers, "list_server_states",
                        lambda *a, **k: [_lemonade_state()])
    calls = []

    def fake_load(target, model="", config=None, evict_others=False, kind=""):
        calls.append((target, model))
        return {"ok": True, "action": "load", "server": target,
                "model": model, "detail": "loaded"}

    monkeypatch.setattr(servers, "load_server_model", fake_load)

    from playwright.sync_api import sync_playwright
    try:
        pw = sync_playwright().start()
        chromium = pw.chromium.launch(headless=True)
    except Exception as e:                       # pragma: no cover - env
        pytest.skip(f"Chromium is unavailable: {e}")
    pg = chromium.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    # With no model configured the boot poll auto-opens Settings; this test
    # is about the Servers dialog, so mark the setup prompt already shown.
    pg.add_init_script("sessionStorage.setItem('pedacito_setup_shown', '1')")
    try:
        pg.goto(f"http://127.0.0.1:{gui.port}/")
        pg.wait_for_selector("#serversBtn", timeout=15000)
        pg.click("#serversBtn")
        pg.wait_for_selector(".modal .note-box", timeout=10000)

        # The row is titled after the server, not the qwen-finetune profile.
        title = pg.text_content(".modal .note-box b")
        assert title == "Lemonade \u00b7 127.0.0.1:11434"
        assert "qwen" not in title.lower()

        # The profile names ride as a secondary line (first three + count).
        prof = pg.text_content(".modal .note-box .muted.small-text")
        assert prof == ("profiles: qwen3_8-27b-gsq-rco, qwen3_8-27b, "
                        "qwen3-coder-30b-a3b-instruct + 2 more")

        # The dropdown is the server's live catalog -- including the model
        # no profile references -- since nothing is resident.
        opts = pg.eval_on_selector_all(
            ".modal .note-box select option", "els => els.map(e => e.value)")
        assert opts == _lemonade_state()["models"]

        # The hint explains the server-side semantics.
        body = pg.text_content(".modal-body")
        assert "creates no profile and switches nothing" in body
        assert "no model resident" in body

        # Load still routes by the registry name, not the display title.
        pg.select_option(".modal .note-box select",
                         "Qwen3-Coder-30B-A3B-Instruct-GGUF")
        pg.click(".modal .note-box button.primary")
        pg.wait_for_function(
            "() => document.querySelector('.modal-body')"
            ".textContent.includes('Loaded ')", timeout=10000)
        assert calls == [("qwen3_8-27b-gsq-rco",
                          "Qwen3-Coder-30B-A3B-Instruct-GGUF")]
        assert not errors, errors
    finally:
        chromium.close()
        pw.stop()
