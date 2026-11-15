"""Deterministic tests for the split-screen terminal UI.

No PTY and no model server are required: `TuiSession` renders complete frames
to an in-memory stream and `run_tui(..., headless=True)` consumes a scripted
input stream. The same tests run on POSIX and Windows CI.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import cli, events, security, workspace
from pedacito import tui as tui_mod
from pedacito.config import Config
from pedacito.tui import TuiSession, parse_escape_sequence, run_tui


class FakeClient:
    """The telemetry fields a real LLMClient exposes, without a network."""

    def __init__(self):
        self.last_prompt_tokens = 8192
        self.last_completion_tokens = 240
        self.last_tok_per_s = 48.5
        self.avg_tok_per_s = 45.0
        self.context_tokens = 32768
        self.active_model_role = "qwen3-coder (coding)"
        self.stats = {
            "calls": 3,
            "prompt_tokens": 14200,
            "completion_tokens": 1900,
            "cached_tokens": 12212,
            "cache_write_tokens": 0,
            "seconds": 42.2,
            "cost": 0.0,
            "by_site": {
                "gather": {"calls": 2, "prompt_tokens": 11000,
                           "completion_tokens": 1200, "cached_tokens": 9400,
                           "cache_write_tokens": 0},
                "answer": {"calls": 1, "prompt_tokens": 3200,
                           "completion_tokens": 700, "cached_tokens": 2812,
                           "cache_write_tokens": 0},
            },
        }

    @property
    def context_utilization_pct(self) -> float:
        return round(self.last_prompt_tokens / self.context_tokens * 100, 1)

    @property
    def cache_hit_pct(self) -> float:
        total = int(self.stats.get("prompt_tokens") or 0)
        return round(int(self.stats.get("cached_tokens") or 0) / total * 100, 1) \
            if total else 0.0

    def telemetry(self) -> dict:
        return {
            "last_prompt_tokens": self.last_prompt_tokens,
            "last_completion_tokens": self.last_completion_tokens,
            "last_tok_per_s": self.last_tok_per_s,
            "avg_tok_per_s": self.avg_tok_per_s,
            "context_tokens": self.context_tokens,
            "context_utilization_pct": self.context_utilization_pct,
            "session_prompt_tokens": self.stats["prompt_tokens"],
            "session_completion_tokens": self.stats["completion_tokens"],
            "cache_hit_pct": self.cache_hit_pct,
            "active_model_role": self.active_model_role,
        }


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


def test_tui_frame_and_telemetry_bar_rendering(tmp_path):
    cfg = _cfg()
    fake = FakeClient()
    session = TuiSession(tmp_path, cfg, client=fake, headless=True,
                         stdout=io.StringIO())
    try:
        wide = session.render_frame(120, 32)
        assert len(wide.splitlines()) == 32
        # Zone 1: project, profile, calibration and routing state.
        assert tmp_path.name in wide
        assert "profile: default" in wide
        assert "\u2713 Calibrated" in wide
        assert "Auto-Route: ON" in wide
        assert "Scout: none" in wide
        # Zone 2: the split viewport and its sidebar titles.
        assert "TRANSCRIPT" in wide
        assert "ACTIVITY & PLAN" in wide
        # Zone 3: context meter, turn burn, speed and active brain.
        assert "25.0%" in wide
        assert "8,192 / 32,768" in wide
        assert "Turn: +8,192 in / +240 out" in wide
        assert "48.5 tok/s" in wide
        assert "(avg 45.0 tok/s)" in wide
        assert "qwen3-coder (coding)" in wide
        assert "KV Cache: 86%" in wide
        # Zone 4: the prompt bar.
        assert "pdc tui (default) >" in wide
        assert "PgUp/PgDn/Wheel: scroll" in wide
        assert "Tab: complete" in wide
        assert "/help | /quit" in wide

        narrow = session.render_frame(78, 24)
        assert len(narrow.splitlines()) == 24
        assert "TRANSCRIPT" in narrow
        assert "ACTIVITY & PLAN" not in narrow       # auto-collapsed
        assert "sidebar: off" in narrow
        # Required values survive the compact footer.
        assert "25.0%" in narrow
        assert "8,192 / 32,768" in narrow
        assert "240" in narrow
        assert "48.5 tok/s" in narrow
        assert "qwen3-coder (coding)" in narrow

        fake.last_prompt_tokens = 30000
        high = session.render_frame(120, 32)
        assert "91.6%" in high
        assert "HIGH" in high
        assert "31" in tui_mod._meter(91.6)[1]      # red ANSI color code
    finally:
        session.close()


def test_tui_slash_commands_end_to_end(tmp_path):
    cfg = _cfg()
    script = "\n".join([
        "/help",
        "/status",
        "/auto-route on",
        "/think off",
        "/skills",
        "/memory add Prefer pure stdlib",
        "/memory list",
        "/consolidate",
        "/sidebar",
        "/quit",
    ]) + "\n"
    out = io.StringIO()
    code = run_tui(tmp_path, cfg, headless=True, stdin=io.StringIO(script),
                   stdout=out)
    assert code == 0
    text = out.getvalue()
    assert "Pedacito TUI commands" in text          # /help
    assert "Pedacito TUI telemetry" in text         # /status
    assert "Auto-Route: ON" in text                 # /auto-route on
    assert "Thinking: OFF" in text                  # /think off
    assert "fastapi-rest-endpoint" in text          # /skills
    assert "remembered" in text                     # /memory add
    assert "Prefer pure stdlib" in text             # /memory list
    assert "consolidated:" in text                  # /consolidate
    assert "Sidebar:" in text                       # /sidebar
    assert "Goodbye." in text                       # /quit

    memory_file = tmp_path / ".pedacito" / "memory.md"
    consolidated = tmp_path / ".pedacito" / "consolidated_session.md"
    assert memory_file.is_file()
    assert "Prefer pure stdlib" in memory_file.read_text(encoding="utf-8")
    assert consolidated.is_file()
    assert "Consolidated session handoff" in consolidated.read_text(
        encoding="utf-8")
    assert cfg.disable_thinking is True
    assert cfg.auto_route_specialties is True


def test_tui_task_execution_updates_activity_and_telemetry(
        tmp_path, monkeypatch):
    cfg = _cfg()
    fake = FakeClient()
    source = tmp_path / "app.py"
    original = "def f():\n    return 1\n"
    changed = "def f():\n    return 2\n"
    source.write_text(original, encoding="utf-8")
    snapshot = workspace.new_backup_dir(tmp_path, when="20250101-000000")
    (snapshot / "app.py").write_text(original, encoding="utf-8")

    def fake_run_task(root, cfg_ref, client, agent, idx, task, **kwargs):
        elog = events.get(root, cfg_ref)
        elog.emit("step", n=1, action="read", detail="app.py")
        elog.emit("step", n=2, action="grep", detail="retry")
        elog.emit("specialist_routed", profile="coder", model="qwen3-coder",
                  specialty="coding")
        elog.emit("model_swap", unloaded="old-model", loaded="qwen3-coder")
        elog.emit("verify", round=0, errors=0, warnings=1)
        source.write_text(changed, encoding="utf-8")
        client.last_prompt_tokens = 12000
        client.last_completion_tokens = 300
        client.last_tok_per_s = 50.0
        client.stats["prompt_tokens"] = 12000
        client.stats["completion_tokens"] = 300
        return SimpleNamespace(
            reply="Updated f().", applied=True, changed=["app.py"],
            results=[SimpleNamespace(ok=True, file="app.py",
                                     message="applied", diff="")],
            idx=idx, steps=["1. read app.py"], stats={}, tests_ok=True,
            findings=[], rolled_back=False)

    monkeypatch.setattr(tui_mod.loop, "run_task", fake_run_task)

    out = io.StringIO()
    session = TuiSession(
        tmp_path, cfg, client=fake, headless=True,
        stdin=io.StringIO("change f\n/undo\n/quit\n"), stdout=out)
    try:
        assert session.run() == 0
    finally:
        session.close()
    text = out.getvalue()
    assert "running apply on qwen3-coder" in text
    assert "read app.py" in text
    assert "grep retry" in text
    assert "specialist_routed" in text and "coder" in text
    assert "model_swap" in text and "old-model" in text
    assert "verify" in text
    assert "36.6%" in text                       # 12,000 / 32,768
    assert "50.0 tok/s" in text
    assert "Updated f()." in text
    assert "applied app.py" in text
    assert "tests: GREEN" in text
    assert "restored 1 file(s)" in text
    assert source.read_text(encoding="utf-8") == original
    assert "app.py" in session.modified


def test_tui_permission_handler_allow_and_deny(tmp_path):
    cfg = _cfg()
    fake = FakeClient()
    out = io.StringIO()
    session = TuiSession(tmp_path, cfg, client=fake, headless=True,
                         stdin=io.StringIO("y\nn\n"), stdout=out)
    try:
        allowed, message = security.request_permission(
            security.CAT_SHELL, "run the test suite", detail="pytest -q")
        assert allowed is True
        assert "Allowed" in message
        denied, denied_message = security.request_permission(
            security.CAT_SENSITIVE_FILE, "read a secret",
            detail=".env")
        assert denied is False
        assert denied_message == security.DENIED_MESSAGE
        text = out.getvalue()
        assert "Permission Required" in text
        assert "shell" in text
        assert "sensitive_file" in text
        assert "y = once / a = session / n = deny" in text
    finally:
        session.close()
    assert security.permission_handler() is None


def test_cli_tui_subcommand(tmp_path, monkeypatch):
    parser = cli.build_parser()
    args = parser.parse_args(["tui", str(tmp_path), "--headless"])
    assert args.cmd == "tui"
    assert args.project == str(tmp_path)
    assert args.headless is True
    assert args.apply is True
    no_apply = parser.parse_args(["tui", str(tmp_path), "--no-apply"])
    assert no_apply.apply is False
    no_think = parser.parse_args(["tui", str(tmp_path), "--no-think"])
    assert no_think.no_think is True

    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert cli.main(["tui", str(tmp_path), "--headless"]) == 0


def test_vt_mode_is_safe_off_windows(monkeypatch):
    monkeypatch.setattr(tui_mod.os, "name", "posix")
    assert tui_mod.enable_vt_mode() is True


def test_interactive_alt_screen_is_restored(tmp_path):
    class TtyInput(io.StringIO):
        def isatty(self):
            return True

    class TtyOutput(io.StringIO):
        def isatty(self):
            return True

    cfg = _cfg()
    out = TtyOutput()
    session = TuiSession(tmp_path, cfg, client=FakeClient(),
                         stdin=TtyInput("/quit\n"), stdout=out)
    assert session.interactive is True
    assert session.run() == 0
    text = out.getvalue()
    assert "\x1b[?1049h" in text
    assert "\x1b[?1049l" in text
    assert "\x1b[r" in text and "\x1b[?25h" in text


def test_call_events_alone_move_the_meter(tmp_path):
    cfg = _cfg()
    session = TuiSession(tmp_path, cfg, headless=True, stdout=io.StringIO())
    try:
        session._apply_event("call", {
            "site": "gather", "prompt_tokens": 16000,
            "completion_tokens": 120, "cached_tokens": 8000,
            "seconds": 4.0})
        t = session.telemetry()
        assert t["last_prompt_tokens"] == 16000
        assert t["last_completion_tokens"] == 120
        assert t["last_tok_per_s"] == 30.0
        assert t["context_utilization_pct"] == 48.8
        assert t["cache_hit_pct"] == 50.0
    finally:
        session.close()


def test_consolidate_resets_the_context_meter(tmp_path):
    cfg = _cfg()
    fake = FakeClient()
    fake.last_prompt_tokens = 24000
    session = TuiSession(tmp_path, cfg, client=fake, headless=True,
                         stdout=io.StringIO())
    try:
        assert session.telemetry()["context_utilization_pct"] == 73.2
        session.handle_line("/consolidate")
        reset = session.telemetry()
        assert reset["last_prompt_tokens"] < 24000
        assert reset["context_utilization_pct"] < 20.0
        assert (tmp_path / ".pedacito" /
                "consolidated_session.md").is_file()
    finally:
        session.close()


@pytest.mark.parametrize("pct,color,state", [
    (10.0, "32", "ok"),
    (75.0, "33", "warn"),
    (95.0, "31", "high"),
])
def test_context_meter_thresholds(pct, color, state):
    bar, got_color, got_state = tui_mod._meter(pct)
    assert got_color == color and got_state == state
    assert len(bar) == 16


def test_escape_sequence_and_mouse_wheel_parser():
    assert parse_escape_sequence("\x1b[5~") == ("PAGE_UP", 4)
    assert parse_escape_sequence("\x1b[6~") == ("PAGE_DOWN", 4)
    assert parse_escape_sequence("\x1b[1;2A") == ("SHIFT_UP", 6)
    assert parse_escape_sequence("\x1b[1;2B") == ("SHIFT_DOWN", 6)
    assert parse_escape_sequence("\x1b[5;3~") == ("ALT_PAGE_UP", 6)
    assert parse_escape_sequence("\x1b[6;3~") == ("ALT_PAGE_DOWN", 6)
    assert parse_escape_sequence("\x1b[H") == ("HOME", 3)
    assert parse_escape_sequence("\x1b[1~") == ("HOME", 4)
    assert parse_escape_sequence("\x1b[F") == ("END", 3)
    assert parse_escape_sequence("\x1b[4~") == ("END", 4)
    assert parse_escape_sequence("\x1b[3~") == ("DELETE", 4)
    assert parse_escape_sequence("\x1b[<64;20;10M") == ("WHEEL_UP", 12)
    assert parse_escape_sequence("\x1b[<65;20;10M") == ("WHEEL_DOWN", 12)
    assert parse_escape_sequence("\x1b[A") == ("UP", 3)
    assert parse_escape_sequence("\x1b[B") == ("DOWN", 3)
    assert parse_escape_sequence("\x1b[A\x1b[A\x1b[A") == ("WHEEL_UP", 9)
    assert parse_escape_sequence("\x1b[B\x1b[B") == ("WHEEL_DOWN", 6)
    assert parse_escape_sequence("\x00I") == ("PAGE_UP", 2)
    assert parse_escape_sequence("\xe0Q") == ("PAGE_DOWN", 2)
    assert parse_escape_sequence("\x1bOQ") == ("F2", 3)
    assert parse_escape_sequence("\x7f") == ("BACKSPACE", 1)
    assert parse_escape_sequence("\x17") == ("CTRL_W", 1)
    assert parse_escape_sequence("\x1b[") == ("", 0)      # incomplete


def test_viewport_scrollback_pgup_pgdn_and_stationary_lock(tmp_path):
    cfg = _cfg()
    session = TuiSession(tmp_path, cfg, client=FakeClient(), headless=True,
                         width=100, height=24, stdout=io.StringIO())
    try:
        for i in range(1, 101):
            session.append(f"line {i:03d}", "system")
        bottom = session.render_frame()
        assert "line 100" in bottom
        assert "Scrolled back" not in bottom

        session.handle_key("PAGE_UP")
        assert session.scroll_offset > 0
        page = session.render_frame()
        assert "Scrolled back" in page
        assert "PgDn / End / Esc" in page
        numbers = [int(m.group(1)) for m in
                   re.finditer(r"line (\d{3})", page)]
        assert numbers and max(numbers) < 100

        before = session.scroll_offset
        session.handle_key("WHEEL_UP")
        assert session.scroll_offset == before + 3
        before_text = session.render_frame()
        first_line = next(ln for ln in before_text.splitlines()
                          if "line " in ln)
        # New background output must not yank the reader back to the bottom.
        session.append("line 101 new background output", "system")
        session.append("line 102 more background output", "system")
        after_text = session.render_frame()
        first_after = next(ln for ln in after_text.splitlines()
                           if "line " in ln)
        assert first_after == first_line
        assert session.scroll_offset > before

        # END, ESC and /scroll bottom all snap back to the live bottom.
        session.handle_key("END")
        assert session.scroll_offset == 0
        session.handle_key("PAGE_UP")
        assert session.scroll_offset > 0
        session.handle_key("ESC")
        assert session.scroll_offset == 0
        session.handle_key("PAGE_UP")
        session.handle_line("/scroll bottom")
        assert session.scroll_offset == 0

        # /scroll top/N are deterministic scriptable drivers too.
        session.handle_line("/scroll top")
        top = session.render_frame()
        assert "line 001" in top
        session.handle_line("/scroll 5")
        assert session.scroll_offset == 5
    finally:
        session.close()


def test_input_buffer_editing_history_and_tab_completion(tmp_path):
    cfg = _cfg()
    cfg.profiles = {"default": {}, "fast": {}}
    session = TuiSession(tmp_path, cfg, client=FakeClient(), headless=True,
                         width=100, height=24, stdout=io.StringIO())
    try:
        session.feed_keys("hello world")
        assert session.input_buf == "hello world"
        assert session.cursor_pos == 11
        for _ in range(5):
            session.handle_key("LEFT")
        assert session.cursor_pos == 6
        session.handle_key("CTRL_W")
        assert session.input_buf == "world"
        assert session.cursor_pos == 0

        session.command_history.append("previous task")
        session._history_pos = len(session.command_history)
        session.handle_key("UP")
        assert session.input_buf == "previous task"
        assert session.cursor_pos == len("previous task")
        session.handle_key("DOWN")
        assert session.input_buf == "world"

        session.input_buf = ""
        session.cursor_pos = 0
        session.feed_keys("/con")
        session.handle_key("TAB")
        assert session.input_buf == "/consolidate "
        assert session.cursor_pos == len("/consolidate ")

        session.input_buf = ""
        session.cursor_pos = 0
        session.feed_keys("/pro")
        session.handle_key("TAB")
        assert session.input_buf == "/profile "

        session.input_buf = ""
        session.cursor_pos = 0
        session.feed_keys("/aut")
        session.handle_key("TAB")
        assert session.input_buf == "/auto-route "

        # Profile-name completion after the command.
        session.input_buf = "/profile fa"
        session.cursor_pos = len(session.input_buf)
        session.handle_key("TAB")
        assert session.input_buf == "/profile fast "

        # Home/End and Delete follow cursor_pos exactly.
        session.input_buf = "abcd"
        session.cursor_pos = 4
        session.handle_key("HOME")
        assert session.cursor_pos == 0
        session.handle_key("DELETE")
        assert session.input_buf == "bcd"
        session.handle_key("END")
        assert session.cursor_pos == 3
    finally:
        session.close()


def test_vt_enter_and_exit_restores_mode_1007_and_1049(tmp_path):
    class TtyInput(io.StringIO):
        def isatty(self):
            return True

    class TtyOutput(io.StringIO):
        def isatty(self):
            return True

    cfg = _cfg()
    out = TtyOutput()
    session = TuiSession(tmp_path, cfg, client=FakeClient(),
                         stdin=TtyInput("/quit\n"), stdout=out)
    assert session.run() == 0
    text = out.getvalue()
    assert "\x1b[?1049h" in text
    assert "\x1b[?1007h" in text
    assert "\x1b[?1000h" not in text          # native selection default
    assert "\x1b[?1007l" in text
    assert "\x1b[?1000l" in text
    assert "\x1b[?1006l" in text
    assert "\x1b[?1049l" in text
    assert "\x1b[?2026h" in text              # synchronized output
    assert "\x1b[?2026l" in text


def test_mouse_toggle_and_sidebar_wheel_scrolling(tmp_path):
    cfg = _cfg()
    session = TuiSession(tmp_path, cfg, client=FakeClient(), headless=True,
                         width=120, height=32, stdout=io.StringIO())
    try:
        for i in range(60):
            session.activity.append(f"activity {i}")
        session.handle_key("WHEEL_UP", col=110)   # over the right sidebar
        assert session.scroll_offset == 0
        assert session.sidebar_scroll_offset == 3
        session.handle_key("ALT_PAGE_DOWN")
        assert session.sidebar_scroll_offset == 0
        assert session.mouse_sgr is False
        session.handle_key("F2")
        assert session.mouse_sgr is True
        session.handle_line("/mouse off")
        assert session.mouse_sgr is False
        frame = session.render_frame()
        assert "pdc tui (default) >" in frame
    finally:
        session.close()


def test_synchronized_redraw_keeps_partial_prompt_and_cursor(tmp_path):
    class TtyInput(io.StringIO):
        def isatty(self):
            return True

    class TtyOutput(io.StringIO):
        def isatty(self):
            return True

    cfg = _cfg()
    out = TtyOutput()
    session = TuiSession(tmp_path, cfg, client=FakeClient(),
                         stdin=TtyInput(""), stdout=out, width=80, height=24)
    try:
        session._prepare_terminal()
        session.input_buf = "partial"
        session.cursor_pos = 3
        session._redraw(force=True)
        text = out.getvalue()
        assert "\x1b[?2026h" in text and "\x1b[?2026l" in text
        assert "partial" in text
        # prefix "pdc tui (default) > " + cursor 3 -> column 24, last row 24
        assert "\x1b[24;24H" in text
    finally:
        session.close()
