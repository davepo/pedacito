"""Deterministic tests for WP-3303: Rich Zero-Dep TUI Upgrade & Round 3 fixes.

Covers:
1. Inline syntax-highlighted unified diffs (ANSI green/red/cyan/bold vs
   headless/NO_COLOR plain text with zero stray ANSI escape bytes).
2. Collapsible tool-call cards (`[1] ✓ read src/app.py L10-60 · 42ms`,
   `Ctrl+O` hotkey toggle, `/expand [N|all]` and `/collapse [N|all]`).
3. Interactive `@` fuzzy picker popup (6-row live dropdown over indexed files,
   symbols, and local images; Up/Down selection, Tab/Enter insertion, Esc
   dismissal, and pure helper functions).
4. Mouse-clickable SGR zones (sidebar tabs, `[EN|es]` language badge,
   collapsible tool-card rows, scrollbar clicks, and wheel scrolling).
5. Round 3 TUI fixes:
   - R3-BUG-06: `/status` telemetry labels localized in Spanish under `/lang es`
     via `i18n.t("tui.status.*")` with 100% EN/ES catalog parity.
   - R3-BUG-08: When a task names `pytest`/tests and `allow_shell` is off,
     display localized `"tests: skipped (needs allow_shell)"` on the final
     verify receipt instead of only a green `✓ verify: 0 error(s), 0 warning(s)`.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import config as config_mod
from pedacito import i18n
from pedacito import tui as tui_mod
from pedacito.chunker import Chunk
from pedacito.config import Config
from pedacito.index import Index
from pedacito.tui import (
    TuiSession,
    extract_at_query,
    format_unified_diff,
    fuzzy_rank_picker_candidates,
    parse_escape_sequence,
    style_unified_diff_line,
)

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _cfg(**overrides) -> Config:
    cfg = Config()
    cfg.model = "qwen3-coder"
    cfg.active_profile = "default"
    cfg.context_tokens = 32768
    cfg.auto_route_specialties = True
    cfg._card = {"measured": {"overall": 8.4},
                 "choices": {}, "version": 2, "ts": 0}
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


@pytest.fixture(autouse=True)
def _reset_lang(tmp_path, monkeypatch):
    cfg_path = tmp_path / "user_config.toml"
    monkeypatch.setattr(config_mod, "USER_CONFIG", cfg_path)
    monkeypatch.delenv("NO_COLOR", raising=False)
    i18n.set_current_lang("en")
    yield
    i18n.set_current_lang("en")


# =========================================================================
# 1. Inline Syntax-Highlighted Unified Diffs
# =========================================================================

def test_inline_diff_ansi_styling_and_headless_degradation(tmp_path, monkeypatch):
    sample_diff = "\n".join([
        "diff --git a/src/app.py b/src/app.py",
        "--- a/src/app.py",
        "+++ b/src/app.py",
        "@@ -10,4 +10,5 @@ def compute(x):",
        "     y = x + 1",
        "-    return y",
        "+    z = y * 2",
        "+    return z",
    ])

    # Pure helper: color=True applies bold headers, cyan hunk, red '-', green '+'
    colored_lines = format_unified_diff(sample_diff, color=True)
    assert colored_lines[0].startswith("\x1b[1m")      # diff --git bold
    assert colored_lines[1].startswith("\x1b[1m")      # --- bold
    assert colored_lines[2].startswith("\x1b[1m")      # +++ bold
    assert colored_lines[3].startswith("\x1b[36m")     # @@ cyan
    assert "\x1b[" not in colored_lines[4]             # context line unstyled
    assert colored_lines[5].startswith("\x1b[31m")     # - removed red
    assert colored_lines[6].startswith("\x1b[32m")     # + added green
    assert colored_lines[7].startswith("\x1b[32m")     # + added green

    # Pure helper: color=False has ZERO ANSI escape bytes
    plain_lines = format_unified_diff(sample_diff, color=False)
    for ln in plain_lines:
        assert "\x1b" not in ln

    # TUI session with color=True renders ANSI diff colors inside transcript
    color_session = TuiSession(
        tmp_path, _cfg(), headless=True, color=True,
        width=100, height=24, stdout=io.StringIO(),
    )
    try:
        color_session.append(sample_diff, "diff")
        frame_color = color_session.render_frame()
        assert "\x1b[1m  diff --git a/src/app.py b/src/app.py\x1b[0m" in frame_color
        assert "\x1b[36m  @@ -10,4 +10,5 @@ def compute(x):\x1b[0m" in frame_color
        assert "\x1b[31m  -    return y\x1b[0m" in frame_color
        assert "\x1b[32m  +    z = y * 2\x1b[0m" in frame_color
    finally:
        color_session.close()

    # TUI session in default --headless mode (color=False) has ZERO stray ANSI bytes
    headless_out = io.StringIO()
    headless_session = TuiSession(
        tmp_path, _cfg(), headless=True,
        width=100, height=24, stdout=headless_out,
    )
    try:
        headless_session.append(sample_diff, "diff")
        headless_session.append("```diff\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n```", "assistant")
        frame_plain = headless_session.render_frame()
        headless_session._redraw(force=True)
        assert "\x1b" not in frame_plain
        assert "\x1b" not in headless_out.getvalue()
        assert "+    z = y * 2" in frame_plain
        assert "-    return y" in frame_plain
    finally:
        headless_session.close()

    # NO_COLOR environment variable forces plain text even when color=True
    monkeypatch.setenv("NO_COLOR", "1")
    assert "\x1b" not in style_unified_diff_line("+added line", color=True)


# =========================================================================
# 2. Collapsible Tool-Call Cards
# =========================================================================

def test_collapsible_tool_cards_ctrl_o_and_slash_commands(tmp_path):
    assert parse_escape_sequence("\x0f") == ("CTRL_O", 1)

    session = TuiSession(
        tmp_path, _cfg(), headless=True,
        width=110, height=28, stdout=io.StringIO(),
    )
    try:
        # Step events create compact 1-line tool cards with sequential IDs, collapsed by default
        session._apply_event("step", {
            "n": 1,
            "action": "read",
            "detail": "src/app.py L10-60",
            "duration_ms": 42,
            "output": "def main():\n    return 42",
        })
        session._apply_event("step", {
            "n": 2,
            "action": "grep",
            "detail": "compute",
            "duration_ms": 15,
            "error": True,
            "output": "ERROR: pattern timed out",
        })

        assert len(session.tool_cards) == 2
        assert session.tool_cards[0]["id"] == 1
        assert session.tool_cards[0]["expanded"] is False
        assert session.tool_cards[1]["id"] == 2
        assert session.tool_cards[1]["ok"] is False

        frame_collapsed = session.render_frame()
        assert "[1] \u2713 read src/app.py L10-60 \u00b7 42ms" in frame_collapsed
        assert "[2] \u2717 grep compute \u00b7 15ms" in frame_collapsed
        # Verbose outputs are hidden while collapsed
        assert "def main():" not in frame_collapsed
        assert "ERROR: pattern timed out" not in frame_collapsed

        # Ctrl+O toggles the latest tool card ([2])
        session.handle_key("CTRL_O")
        assert session.tool_cards[1]["expanded"] is True
        frame_ctrl_o = session.render_frame()
        assert "ERROR: pattern timed out" in frame_ctrl_o
        assert "def main():" not in frame_ctrl_o

        # Second Ctrl+O collapses card [2] again
        session.handle_key("CTRL_O")
        assert session.tool_cards[1]["expanded"] is False
        assert "ERROR: pattern timed out" not in session.render_frame()

        # /expand 1 expands card 1 only
        session.handle_line("/expand 1")
        assert session.tool_cards[0]["expanded"] is True
        assert session.tool_cards[1]["expanded"] is False
        assert "def main():" in session.render_frame()

        # /collapse 1 collapses card 1
        session.handle_line("/collapse 1")
        assert session.tool_cards[0]["expanded"] is False
        assert "def main():" not in session._render_tool_card_text(session.tool_cards[0])

        # /expand all expands both cards
        session.handle_line("/expand all")
        assert all(c["expanded"] for c in session.tool_cards)
        frame_all = session.render_frame()
        assert "def main():" in frame_all
        assert "ERROR: pattern timed out" in frame_all

        # /collapse all collapses all cards
        session.handle_line("/collapse all")
        assert not any(c["expanded"] for c in session.tool_cards)
    finally:
        session.close()


def test_headless_expand_emits_card_detail_into_transcript(tmp_path):
    out = io.StringIO()
    session = TuiSession(
        tmp_path, _cfg(), headless=True,
        width=110, height=28, stdout=out,
    )
    try:
        session._apply_event("step", {
            "n": 1,
            "action": "read",
            "detail": "src/app.py L10-60",
            "duration_ms": 42,
            "output": "def main():\n    return 42",
        })
        session._redraw(force=True)
        out.seek(0)
        out.truncate()

        session.handle_line("/expand 1")
        session._redraw(force=True)
        emitted = out.getvalue()
        assert "Tool card(s) [1] expanded." in emitted
        assert "    \u2502 def main():" in emitted
        assert "    \u2502     return 42" in emitted
    finally:
        session.close()


# =========================================================================
# 3. Interactive `@` Fuzzy Picker Popup
# =========================================================================

def test_at_fuzzy_picker_filtering_navigation_insertion_and_dismiss(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text(
        "class Engine:\n    def run_fast(self):\n        return 1\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "utils.py").write_text(
        "def helper():\n    return 2\n",
        encoding="utf-8",
    )
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    chunks = [
        Chunk(file="src/app.py", qualname="Engine", kind="class", start=1, end=3),
        Chunk(file="src/app.py", qualname="Engine.run_fast", kind="method", start=2, end=3),
        Chunk(file="src/utils.py", qualname="helper", kind="function", start=1, end=2),
    ]
    idx = Index(tmp_path, chunks)

    # Pure helper: extract_at_query & fuzzy_rank_picker_candidates
    assert extract_at_query("check @app", 10) == (6, 10, "app")
    assert extract_at_query("email user@example.com", 22) is None
    ranked = fuzzy_rank_picker_candidates("run_fast", [
        {"kind": "file", "value": "src/app.py", "label": "src/app.py", "detail": "file"},
        {"kind": "symbol", "value": "src/app.py:Engine.run_fast",
         "label": "src/app.py:Engine.run_fast", "detail": "method"},
    ])
    assert len(ranked) == 1
    assert ranked[0]["value"] == "src/app.py:Engine.run_fast"

    session = TuiSession(
        tmp_path, _cfg(), idx=idx, headless=True,
        width=110, height=28, stdout=io.StringIO(),
    )
    try:
        # Typing `@` opens the 6-row inline fuzzy dropdown
        session.feed_keys(" please inspect @")
        state = session.picker_state()
        assert state["open"] is True
        assert state["query"] == ""
        assert 1 <= len(state["items"]) <= 6
        kinds = {it["kind"] for it in state["items"]}
        assert "file" in kinds
        assert "image" in kinds or "symbol" in kinds

        # Popup renders inline in the frame
        frame_popup = session.render_frame()
        assert "@src/app.py" in frame_popup

        # Live filtering for symbol `run_fast`
        session.feed_keys("run_fast")
        state_sym = session.picker_state()
        assert state_sym["open"] is True
        assert state_sym["query"] == "run_fast"
        assert state_sym["selected"]["value"] == "src/app.py:Engine.run_fast"

        # Tab inserts `@src/app.py:Engine.run_fast ` into the prompt buffer
        session.handle_key("TAB")
        assert session.picker_state()["open"] is False
        assert "@src/app.py:Engine.run_fast " in session.input_buf

        # Typing another `@` for image/file selection and navigating with Down/Up + Enter
        session.feed_keys("@")
        assert session.picker_state()["open"] is True
        first_val = session.picker_state()["selected"]["value"]
        session.handle_key("DOWN")
        assert session.picker_state()["index"] == 1
        second_val = session.picker_state()["selected"]["value"]
        assert second_val != first_val
        session.handle_key("UP")
        assert session.picker_state()["index"] == 0

        # Filter to `logo` and accept with Enter (must insert, NOT submit prompt)
        session.feed_keys("logo")
        assert session.picker_state()["selected"]["value"] == "assets/logo.png"
        session.handle_key("ENTER")
        assert session.picker_state()["open"] is False
        assert "@assets/logo.png" in session.input_buf

        # Esc dismisses the popup cleanly without modifying the typed buffer
        session.feed_keys("@ut")
        assert session.picker_state()["open"] is True
        before_buf = session.input_buf
        session.handle_key("ESC")
        assert session.picker_state()["open"] is False
        assert session.input_buf == before_buf
    finally:
        session.close()


# =========================================================================
# 4. SGR Mouse-Clickable Zones
# =========================================================================

def test_sgr_mouse_click_zones_sidebar_lang_toolcards_and_scrollbar(tmp_path):
    # SGR left button press parses to MOUSE_CLICK
    assert parse_escape_sequence("\x1b[<0;40;2M") == ("MOUSE_CLICK", 10)
    # SGR release is swallowed cleanly as UNKNOWN
    assert parse_escape_sequence("\x1b[<0;40;2m") == ("UNKNOWN", 10)

    (tmp_path / "PLAN.md").write_text("- [ ] First plan task\n", encoding="utf-8")
    session = TuiSession(
        tmp_path, _cfg(), headless=True,
        width=120, height=30, stdout=io.StringIO(),
    )
    try:
        session.activity.append("read src/app.py")
        session.modified.append("src/app.py")
        session.add_tool_card(
            "read", "src/app.py L1-20", ok=True, duration_ms=25,
            output="line 1\nline 2",
        )

        # 1) Click the `[EN|es]` language badge on header row 2 -> toggles EN -> ES
        raw_h2 = _ANSI_ESCAPE_RE.sub("", session.header_lines(120)[1])
        pill_col = raw_h2.index("[EN|es]") + 2
        zone = session.handle_mouse_click(col=pill_col, row=2)
        assert zone == "lang_badge"
        assert session.lang == "es"

        # Click `[en|ES]` via raw SGR sequence in feed_keys -> toggles ES -> EN
        raw_h2_es = _ANSI_ESCAPE_RE.sub("", session.header_lines(120)[1])
        pill_col_es = raw_h2_es.index("[en|ES]") + 2
        session.feed_keys([f"\x1b[<0;{pill_col_es};2M"])
        assert session.lang == "en"

        # 2) Click tool card row in transcript pane -> toggles expand/collapse
        frame_lines = session.render_frame(120, 30).splitlines()
        card_row_1based = next(
            idx + 1 for idx, ln in enumerate(frame_lines)
            if "[1] \u2713 read src/app.py L1-20" in ln
        )
        assert session.tool_cards[0]["expanded"] is False
        zone_card = session.handle_mouse_click(col=10, row=card_row_1based)
        assert zone_card == "tool_card:1"
        assert session.tool_cards[0]["expanded"] is True
        assert "line 1" in session.render_frame(120, 30)

        # Click again to collapse
        assert session.handle_mouse_click(col=10, row=card_row_1based) == "tool_card:1"
        assert session.tool_cards[0]["expanded"] is False

        # 3) Click sidebar section headers to filter/switch sidebar tabs
        # Row 3 in the right sidebar is `ACTIVITY & PLAN`
        zone_side = session.handle_mouse_click(col=105, row=3)
        assert zone_side == "sidebar_tab:plan"
        assert session.sidebar_tab == "plan"
        side_plan_only = session.render_frame(120, 30)
        assert "First plan task" in side_plan_only
        assert "MODIFIED FILES" not in side_plan_only

        # Cycle through activity -> files -> all
        assert session.handle_mouse_click(col=105, row=3) == "sidebar_tab:activity"
        assert session.handle_mouse_click(col=105, row=3) == "sidebar_tab:files"
        assert session.handle_mouse_click(col=105, row=3) == "sidebar_tab:all"

        # 4) Scrollbar click when transcript has scrollback
        for i in range(60):
            session.append(f"scroll line {i}", "system")
        assert session.scroll_offset == 0
        side_width = min(46, max(32, 120 // 3))
        left_width = 120 - side_width - 1
        assert session.handle_mouse_click(col=left_width, row=5) == "scrollbar_up"
        assert session.scroll_offset > 0
        assert session.handle_mouse_click(col=left_width, row=22) == "scrollbar_down"
    finally:
        session.close()


# =========================================================================
# 5. Round 3 TUI Fixes: R3-BUG-06 (/status i18n) & R3-BUG-08 (skipped tests)
# =========================================================================

def test_r3_bug_06_status_report_spanish_localization(tmp_path):
    en_cat = i18n.load_catalog("en")
    es_cat = i18n.load_catalog("es")
    assert set(en_cat) == set(es_cat)
    status_keys = [k for k in en_cat if k.startswith("tui.status.")]
    assert len(status_keys) >= 15

    session = TuiSession(
        tmp_path, _cfg(), headless=True,
        width=110, height=28, stdout=io.StringIO(),
    )
    try:
        en_report = session.status_report()
        assert "Pedacito TUI telemetry" in en_report
        assert "profile :" in en_report
        assert "writes  : ON" in en_report
        assert "per-call-site tokens:" in en_report

        session.handle_line("/lang es")
        assert session.lang == "es"
        es_report = session.status_report()
        assert "Telemetr\u00eda de Pedacito TUI" in es_report
        assert "perfil" in es_report
        assert "modelo" in es_report
        assert "escritura: ACTIVA" in es_report
        assert "velocidad:" in es_report
        assert "cach\u00e9" in es_report
        assert "pliegues" in es_report
        assert "ruteo" in es_report
        assert "ficha de modelo: 8.4 / 10 (calibrado)" in es_report
        assert "especialidades:" in es_report
        assert "tokens por etapa:" in es_report
    finally:
        session.close()


def test_r3_bug_08_skipped_tests_receipt_when_allow_shell_off(tmp_path, monkeypatch):
    cfg = _cfg(allow_shell=False)

    def fake_run_task(root, cfg_ref, client, agent, idx, task, **kwargs):
        # Simulate preflight verify passing while tests were skipped due to allow_shell=False
        session._apply_event("verify", {"round": 0, "errors": 0, "warnings": 0})
        return SimpleNamespace(
            reply="Fixed bug.",
            applied=True,
            changed=["app.py"],
            results=[SimpleNamespace(ok=True, file="app.py", message="applied", diff="")],
            idx=idx,
            steps=["1. read app.py"],
            stats={},
            tests_ok=None,
            findings=[],
            rolled_back=False,
        )

    monkeypatch.setattr(tui_mod.loop, "run_task", fake_run_task)

    session = TuiSession(
        tmp_path, cfg, headless=True,
        width=110, height=28, stdout=io.StringIO(),
    )
    try:
        session.run_task("fix the parser and run pytest")
        receipts = [text for kind, text in session.transcript if kind == "receipt"]
        combined = "\n".join(receipts)
        assert "tests: skipped (needs allow_shell)" in combined
        assert any("tests: skipped (needs allow_shell)" in act for act in session.activity)

        # Also verify Spanish localization of the skipped-tests receipt
        session.handle_line("/lang es")
        session.run_task("arregla el parser y ejecuta pytest")
        es_receipts = "\n".join(text for kind, text in session.transcript if kind == "receipt")
        assert "pruebas: omitidas (requiere allow_shell)" in es_receipts
    finally:
        session.close()
