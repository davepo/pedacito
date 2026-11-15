"""Deterministic tests for the split-screen terminal UI.

No PTY and no model server are required: `TuiSession` renders complete frames
to an in-memory stream and `run_tui(..., headless=True)` consumes a scripted
input stream. The same tests run on POSIX and Windows CI.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import cli, events, security, workspace
from pedacito import tui as tui_mod
from pedacito.config import Config
from pedacito.tui import TuiSession, run_tui


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
        assert "/help: commands" in wide

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
