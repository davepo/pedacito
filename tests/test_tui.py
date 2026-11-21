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
from pedacito import config as config_mod
from pedacito import profiler
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


class _TrackingLock:
    """The session's RLock with an entry counter (the append path is what a
    background worker races with the renderer)."""

    def __init__(self, lock):
        self._lock = lock
        self.entries = 0

    def __enter__(self):
        self.entries += 1
        return self._lock.__enter__()

    def __exit__(self, *exc):
        return self._lock.__exit__(*exc)


def test_tui_append_takes_the_write_lock(tmp_path):
    session = TuiSession(tmp_path, _cfg(), headless=True,
                         stdout=io.StringIO())
    tracking = _TrackingLock(session._write_lock)
    session._write_lock = tracking
    try:
        session._append("worker line")
        assert tracking.entries >= 1
        assert session.transcript[-1] == ("system", "worker line")
    finally:
        session.close()


def test_tui_append_after_close_is_a_safe_noop(tmp_path):
    """A startup update check or probe daemon can finish after Ctrl+C
    teardown; it must not mutate the torn-down transcript."""
    session = TuiSession(tmp_path, _cfg(), headless=True,
                         stdout=io.StringIO())
    session.close()
    before = list(session.transcript)
    session._append("late worker line")
    session._append_many("late worker line")
    assert session.transcript == before
    assert session.activity == []


def test_tui_effort_command_sets_reports_aliases_and_rejects(tmp_path):
    """`/effort` dispatch + handler: no argument reports the live value,
    valid levels set the config and source label, aliases collapse to
    default, and a bad level prints usage without changing anything."""
    cfg = _cfg(reasoning_effort="medium")
    session = TuiSession(tmp_path, cfg, headless=True, stdout=io.StringIO())
    text = lambda: "\n".join(line for _kind, line in session.transcript)
    try:
        session.handle_line("/effort")
        assert "Reasoning effort: medium" in text()

        session.handle_line("/effort high")
        assert cfg.reasoning_effort == "high"
        assert cfg.sources["reasoning_effort"] == "tui /effort"
        assert "sent as reasoning_effort" in text()

        session.handle_line("/effort off")
        assert cfg.reasoning_effort == "default"
        assert "no field is sent at default" in text()

        before = cfg.reasoning_effort
        session.handle_line("/effort banana")
        assert cfg.reasoning_effort == before
        assert "Usage: /effort default|low|medium|high" in text()
    finally:
        session.close()


def test_tui_servers_pin_parses_target_model_and_state(tmp_path, monkeypatch):
    session = TuiSession(tmp_path, _cfg(), headless=True,
                         stdout=io.StringIO())
    calls = []

    def fake_pin(target, model="", pinned=True, config=None):
        calls.append((target, model, pinned))
        return {"ok": True, "action": "pin" if pinned else "unpin",
                "server": target, "model": model or "resident",
                "pinned": pinned, "detail": "done"}

    monkeypatch.setattr(tui_mod.servers_mod, "pin_server_model", fake_pin)
    try:
        session._servers("pin lemonade qwen3-8b")
        session._servers("unpin lemonade")
        assert calls == [("lemonade", "qwen3-8b", True),
                         ("lemonade", "", False)]
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


def test_tui_verify_detects_sets_reports_and_clears(tmp_path):
    (tmp_path / "package.json").write_text(
        '{"scripts": {"test": "node --test"}}\n', encoding="utf-8")
    cfg = _cfg()
    session = TuiSession(tmp_path, cfg, headless=True,
                         stdout=io.StringIO())
    try:
        def transcript():
            return "\n".join(line for _kind, line in session.transcript)

        session._slash("/verify status")
        text = transcript()
        assert "Post-edit verify (auto_verify): OFF" in text
        assert "Static gate (static_gate): ON" in text
        assert "Repair rounds (auto_repair_rounds): 2" in text
        assert "Scoped tests:" in text and "test_timeout (300 s)" in text
        assert "Green ratchet (auto_green_ratchet): ON" in text
        assert "interface lock (interface_lock): ON" in text

        session._slash("/verify on")
        assert cfg.auto_verify == "npm test"
        assert cfg.sources["auto_verify"] == "tui /verify"
        assert "detected npm test" in transcript()

        session._slash("/verify status")
        assert "Post-edit verify (auto_verify): ON \u2014 npm test" in transcript()

        session._slash("/verify off")
        assert cfg.auto_verify == ""
        assert cfg.sources["auto_verify"] == "tui /verify"
        assert "OFF \u2014 auto_verify cleared" in transcript()

        # Registered for completion and documented in /help.
        assert "/verify" in tui_mod._SLASH_COMMANDS
        assert "/verify [on|off|status]" in tui_mod.HELP_TEXT
    finally:
        session.close()


def test_tui_verify_on_without_a_detectable_command_stays_empty(tmp_path):
    cfg = _cfg()
    session = TuiSession(tmp_path, cfg, headless=True,
                         stdout=io.StringIO())
    try:
        session._slash("/verify on")
        text = "\n".join(line for _kind, line in session.transcript)
        assert cfg.auto_verify == ""
        assert cfg.sources["auto_verify"] == "tui /verify"
        assert "no test command detected" in text
        assert "syntax/AST" in text
        assert "static and interface-contract gates" in text
        assert "scoped impacted tests" in text
    finally:
        session.close()


def test_tui_verify_never_writes_config_files(tmp_path):
    config_file = tmp_path / ".pedacito.toml"
    config_file.write_text(
        '[universal]\nmodel = "unchanged"\n', encoding="utf-8")
    before = config_file.read_bytes()
    cfg = _cfg()
    session = TuiSession(tmp_path, cfg, headless=True,
                         stdout=io.StringIO())
    try:
        session._slash("/verify on")
        session._slash("/verify status")
        session._slash("/verify off")
    finally:
        session.close()
    assert config_file.read_bytes() == before


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


def test_alternate_scroll_mode_sequence_and_wheel_burst():
    """The terminal enables ?1007h (alternate scroll) so a wheel becomes an
    arrow-key burst. The parser must swallow the mode CSI whole -- never leak
    '[' / '1' / '0' into the prompt -- and still read the burst as WHEEL."""
    assert parse_escape_sequence("\x1b[?1007h") == ("UNKNOWN", 8)
    assert parse_escape_sequence("\x1b[?1007l") == ("UNKNOWN", 8)
    # one drain: mode set + wheel burst, then a real key after it
    buf = "\x1b[?1007h\x1b[A\x1b[A\x1b[Aq"
    key, used = parse_escape_sequence(buf)
    assert key == "UNKNOWN" and used == 8
    key, used2 = parse_escape_sequence(buf[used:])
    assert key == "WHEEL_UP" and used2 == 9
    assert buf[used + used2] == "q"
    # a single arrow stays a history key, not a wheel event
    assert parse_escape_sequence("\x1b[A") == ("UP", 3)
    assert parse_escape_sequence("\x1b[<64;15;10M") == ("WHEEL_UP", 12)
    assert parse_escape_sequence("\x1b[<65;15;10M") == ("WHEEL_DOWN", 12)


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


def test_tui_profile_run_refuses_hosted_endpoint(tmp_path, monkeypatch):
    from pedacito import profiler
    cfg = _cfg(base_url="https://api.deepseek.com/v1",
               model="deepseek-chat")
    calls = []
    monkeypatch.setattr(profiler, "probe_model_speed",
                        lambda *a, **k: calls.append((a, k)))
    session = TuiSession(tmp_path, cfg, headless=True, stdout=io.StringIO())
    try:
        session.handle_line("/profile run")
        assert calls == []
        system = [text for kind, text in session.transcript
                  if kind == "system"]
        assert any("hosted model" in text and "refusing to probe" in text
                   for text in system)
        assert not any(kind == "error" for kind, _ in session.transcript)
    finally:
        session.close()


def test_tui_profile_run_probes_local_endpoint(tmp_path, monkeypatch):
    from pedacito import profiler
    cfg = _cfg(base_url="http://127.0.0.1:1234/v1", model="local-model")
    calls = []

    def fake_probe(base_url, model, config, **kw):
        calls.append((base_url, model, kw.get("force")))
        return {"ok": True, "base_url": base_url, "model": model,
                "prefill_tok_per_s": 800.0, "generation_tok_per_s": 40.0,
                "cached": False, "measured_at": 0.0}

    monkeypatch.setattr(profiler, "probe_model_speed", fake_probe)
    monkeypatch.setattr(
        profiler, "build_report",
        lambda cfg, **kw: {"ok": True, "model": cfg.model, "snapshot": {},
                           "current": {}, "tuning": {}, "auto_tune": False})
    session = TuiSession(tmp_path, cfg, headless=True, stdout=io.StringIO())
    try:
        session.handle_line("/profile run")
        assert calls == [("http://127.0.0.1:1234/v1", "local-model", True)]
        assert any("Hardware snapshot" in text
                   for _, text in session.transcript)
    finally:
        session.close()


def test_tui_profile_reset_writes_auto_tune_false(tmp_path, monkeypatch):
    """R3.7: /profile reset must write the same file change as the CLI."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        'default_profile = "x"\n'
        "[profiles.x]\n"
        'base_url = "http://127.0.0.1:1234/v1"\n'
        'model = "local-model"\n'
        "vram_gb = 24\n"
        "auto_tune = true\n", encoding="utf-8")
    monkeypatch.setattr(profiler, "apply_auto_tune",
                        lambda *a, **k: {"ok": True, "applied": False})
    monkeypatch.setattr(profiler, "reset",
                        lambda: tmp_path / "profiler.json")
    cfg = config_mod.load(root=tmp_path, explicit=cfg_path, profile="x",
                          require_hardware=False, warn_budgets=False)
    assert cfg.auto_tune is True
    assert cfg._config_path == str(cfg_path)
    session = TuiSession(tmp_path, cfg, client=FakeClient(), headless=True,
                         stdout=io.StringIO())
    try:
        session.handle_line("/profile reset")
        text = "\n".join(line for _kind, line in session.transcript)
        assert f"Set auto_tune = false in {cfg_path}" in text
        assert "auto_tune is off for this session." in text
        assert cfg.auto_tune is False
        assert "auto_tune = false" in cfg_path.read_text(encoding="utf-8")
        again = config_mod.load(root=tmp_path, explicit=cfg_path, profile="x",
                                require_hardware=False, warn_budgets=False)
        assert again.auto_tune is False
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


def test_tui_alt_screen_off_resets_scroll_region_before_leaving_alt_screen(tmp_path):
    """DECSTBM (`\\x1b[r`) always homes the cursor to (1, 1); emitting it after
    `\\x1b[?1049l` clobbers the cursor position restored on the main screen."""
    seq = tui_mod._ALT_SCREEN_OFF
    assert "\x1b[r" in seq
    assert "\x1b[?1049l" in seq
    assert seq.index("\x1b[r") < seq.index("\x1b[?1049l")
    assert seq.index("\x1b[?2026l") < seq.index("\x1b[?1049l")
    assert seq.index("\x1b[0m") < seq.index("\x1b[?1049l")

    after = seq.split("\x1b[?1049l", 1)[1]
    # Only cursor visibility restore (`?25h`) may follow leaving the alt screen;
    # nothing after `?1049l` may move the cursor or reset scroll margins.
    assert after == "\x1b[?25h"
    assert not re.search(r"\x1b\[[0-9;]*[HfABCDEFGJKr]", after)

    class TtyInput(io.StringIO):
        def isatty(self):
            return True

    class TtyOutput(io.StringIO):
        def isatty(self):
            return True

    out = TtyOutput()
    session = TuiSession(tmp_path, _cfg(), client=FakeClient(),
                         stdin=TtyInput(""), stdout=out, width=80, height=24)
    session._prepare_terminal()
    out.truncate(0)
    out.seek(0)
    session.close()
    teardown = out.getvalue()
    assert "\x1b[?1049l" in teardown
    assert teardown.index("\x1b[r") < teardown.index("\x1b[?1049l")
    assert teardown.split("\x1b[?1049l", 1)[1] == "\x1b[?25h"


def test_bracketed_paste_sequences_and_handle_paste(tmp_path):
    assert "\x1b[?2004h" in tui_mod._ALT_SCREEN_ON
    assert "\x1b[?2004l" in tui_mod._ALT_SCREEN_OFF
    assert (tui_mod._ALT_SCREEN_OFF.index("\x1b[?2004l")
            < tui_mod._ALT_SCREEN_OFF.index("\x1b[?1049l"))
    assert parse_escape_sequence("\x1b[200~") == ("PASTE_START", 6)
    assert parse_escape_sequence("\x1b[201~") == ("PASTE_END", 6)

    submitted: list[str] = []
    session = TuiSession(tmp_path, _cfg(), client=FakeClient(), headless=True,
                         width=100, height=24, stdout=io.StringIO())
    session.handle_line = lambda line: (submitted.append(line) or True)
    try:
        # Multi-line paste normalizes \r\n and \r, keeps \t and \n, strips escapes,
        # and never triggers submit.
        raw_paste = "def foo():\r\n\t\x1b[31mreturn\x1b[0m \x1b]0;evil\x071\r# done"
        session.handle_paste(raw_paste)
        assert submitted == []
        assert session.input_buf == "def foo():\n\treturn 1\n# done"
        assert session.cursor_pos == len(session.input_buf)

        # Bracketed paste through feed_keys also inserts without submitting.
        session.input_buf = ""
        session.cursor_pos = 0
        session.feed_keys(["\x1b[200~alpha\r\nbeta\x1b[201~"])
        assert submitted == []
        assert session.input_buf == "alpha\nbeta"
    finally:
        session.close()


def test_bracketed_paste_in_interactive_reader_loop(tmp_path):
    class TtyInput(io.StringIO):
        def isatty(self):
            return True

    class TtyOutput(io.StringIO):
        def isatty(self):
            return True

    submitted: list[str] = []
    out = TtyOutput()
    session = TuiSession(tmp_path, _cfg(), client=FakeClient(),
                         stdin=TtyInput(""), stdout=out, width=80, height=24)

    def capture(line: str) -> bool:
        submitted.append(line)
        session._quit = True
        return False

    session.handle_line = capture
    # Pre-seed raw buffer with a bracketed multi-line paste followed by Enter.
    session._raw_buf = "\x1b[200~first line\r\nsecond line\x1b[201~\r"
    try:
        assert session.run() == 0
        assert submitted == ["first line\nsecond line"]
        text = out.getvalue()
        assert "\x1b[?2004h" in text
        assert "\x1b[?2004l" in text
        assert text.index("\x1b[?2004l") < text.index("\x1b[?1049l")
    finally:
        session.close()


def test_multiline_keys_and_backslash_continuation(tmp_path):
    assert parse_escape_sequence("\x1b[13;2u") == ("SHIFT_ENTER", 7)
    assert parse_escape_sequence("\x1b[27;2;13~") == ("SHIFT_ENTER", 10)
    assert parse_escape_sequence("\x1b\r") == ("ALT_ENTER", 2)
    assert parse_escape_sequence("\x1b\n") == ("ALT_ENTER", 2)
    assert parse_escape_sequence("\n") == ("CTRL_J", 1)
    assert parse_escape_sequence("\r") == ("ENTER", 1)

    class TtyInput(io.StringIO):
        def isatty(self):
            return True

    class TtyOutput(io.StringIO):
        def isatty(self):
            return True

    submitted: list[str] = []
    session = TuiSession(tmp_path, _cfg(), client=FakeClient(),
                         stdin=TtyInput(""), stdout=TtyOutput(),
                         width=100, height=24)
    session.handle_line = lambda line: (submitted.append(line) or True)
    try:
        session.feed_keys("line1")
        session.handle_key("SHIFT_ENTER")
        session.feed_keys("line2")
        session.handle_key("ALT_ENTER")
        session.feed_keys("line3")
        session.handle_key("CTRL_J")
        session.feed_keys("line4\\")
        # Lone trailing backslash + ENTER in interactive mode inserts newline.
        session.handle_key("ENTER")
        assert submitted == []
        assert session.input_buf == "line1\nline2\nline3\nline4\n"

        session.feed_keys("line5")
        session.handle_key("ENTER")
        assert submitted == ["line1\nline2\nline3\nline4\nline5"]
        assert session.input_buf == ""
    finally:
        session.close()


def test_multiline_navigation_home_end_and_line_kills(tmp_path):
    session = TuiSession(tmp_path, _cfg(), client=FakeClient(), headless=True,
                         width=100, height=24, stdout=io.StringIO())
    try:
        session.command_history = ["older cmd"]
        session._history_pos = 1
        session.input_buf = "long first line\nhi\nanother long line"
        # Place cursor at col 10 on line 2 ("another lo|ng line")
        session.cursor_pos = len("long first line\nhi\n") + 10

        # UP moves to line 1 ("hi"), clamping to end of line 1 (col 2),
        # while remembering preferred col 10.
        session.handle_key("UP")
        assert session.cursor_pos == len("long first line\n") + 2

        # Second UP moves to line 0, restoring target col 10!
        session.handle_key("UP")
        assert session.cursor_pos == 10

        # UP on logical line 0 steps into command history.
        session.handle_key("UP")
        assert session.input_buf == "older cmd"

        # DOWN restores the multi-line draft.
        session.handle_key("DOWN")
        assert session.input_buf == "long first line\nhi\nanother long line"

        # HOME / END operate on the current logical line.
        session.cursor_pos = len("long first line\n") + 1  # middle of "hi"
        session.handle_key("HOME")
        assert session.cursor_pos == len("long first line\n")
        session.handle_key("END")
        assert session.cursor_pos == len("long first line\nhi")

        # CTRL_U kills to start of current logical line only.
        session.handle_key("CTRL_U")
        assert session.input_buf == "long first line\n\nanother long line"
        assert session.cursor_pos == len("long first line\n")

        # CTRL_K kills to end of current logical line only.
        session.cursor_pos = 4  # "long| first line"
        session.handle_key("CTRL_K")
        assert session.input_buf == "long\n\nanother long line"
        assert session.cursor_pos == 4
    finally:
        session.close()


def test_multiline_rendering_height_scroll_and_cursor_position(tmp_path):
    session = TuiSession(tmp_path, _cfg(), client=FakeClient(), headless=True,
                         width=80, height=24, stdout=io.StringIO())
    try:
        # 3-line draft: frame height stays exactly 24, cursor on row 2.
        session.input_buf = "first\nsecond\nthird"
        # Put cursor on line index 1 ("se|cond", col 2 -> offset 6 + 2 = 8).
        session.cursor_pos = 8
        frame3 = session.render_frame(80, 24)
        lines3 = frame3.splitlines()
        assert len(lines3) == 24
        assert lines3[21].startswith("pdc tui (default) > first")
        assert lines3[22].startswith("  \u2502 second")
        assert lines3[23].startswith("  \u2502 third")
        # Prompt occupies 1-based rows 22, 23, 24; row 2 of prompt is row 23.
        # Prefix "  │ " has len 4, cursor at col 2 -> 1-based col 7.
        assert session._cursor_screen_position(80, 24) == (23, 7)

        # 8-line draft: capped at MAX_PROMPT_ROWS (6), frame height still 24,
        # and scroll indicator [L8/8] shown when cursor is at the end.
        session.input_buf = "\n".join(f"row{i}" for i in range(1, 9))
        session.cursor_pos = len(session.input_buf)
        frame8 = session.render_frame(80, 24)
        lines8 = frame8.splitlines()
        assert len(lines8) == 24
        prompt_rows, cur_row, _cur_col = session._prompt_lines(80)
        assert len(prompt_rows) == tui_mod.MAX_PROMPT_ROWS == 6
        assert cur_row == 5
        assert "[L8/8]" in lines8[18]
        assert "row8" in lines8[23]

        # Moving cursor to line 1 scrolls the prompt window to the top.
        session.cursor_pos = 0
        frame8_top = session.render_frame(80, 24)
        lines8_top = frame8_top.splitlines()
        assert len(lines8_top) == 24
        assert "[L1/8]" in lines8_top[18]
        assert "row1" in lines8_top[18]

        # Help and keys text invariants.
        assert ("Web: https://www.pedacito.dev \u00b7 "
                "Contact: contact@pedacito.dev") in tui_mod.HELP_TEXT
        assert "Shift+Enter" in tui_mod.KEYS_TEXT
        assert "Bracketed paste" in tui_mod.KEYS_TEXT
        assert "Shift+Enter" in tui_mod.KEYS_TEXT_ES
    finally:
        session.close()
