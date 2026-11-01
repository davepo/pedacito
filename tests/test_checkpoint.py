"""Conversation checkpoints used by ask --continue / chat --resume."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import checkpoint
from pedacito.config import Config


def test_save_latest_render_and_list(tmp_path):
    cfg = Config()
    cfg.log_sessions = True
    msgs = [{"role": "user", "content": "find the retry bug"},
            {"role": "assistant", "content": "read src/sync.py"}]
    p = checkpoint.save(tmp_path, "chat", "find the retry bug", msgs,
                        "fixed it in sync.py", cfg,
                        extra={"steps": ["read src/sync.py"]})
    assert p is not None and p.is_file()
    cp = checkpoint.latest(tmp_path, "chat")
    assert cp["task"] == "find the retry bug"
    assert cp["reply"] == "fixed it in sync.py"
    assert cp["steps"] == ["read src/sync.py"]
    rendered = checkpoint.render(cp)
    assert "PRIOR CONVERSATION" in rendered
    assert "read src/sync.py" in rendered
    assert "Previous reply" in rendered
    assert "chat" in checkpoint.render_list(tmp_path)


def test_checkpoints_are_not_written_when_session_logging_is_off(tmp_path):
    cfg = Config()
    cfg.log_sessions = False
    assert checkpoint.save(tmp_path, "ask", "t", [], "r", cfg) is None
    assert checkpoint.latest(tmp_path) is None


def test_latest_filters_by_command(tmp_path):
    cfg = Config()
    cfg.log_sessions = True
    checkpoint.save(tmp_path, "ask", "a", [], "a reply", cfg)
    checkpoint.save(tmp_path, "chat", "c", [], "c reply", cfg)
    assert checkpoint.latest(tmp_path, "ask")["task"] == "a"
    assert checkpoint.latest(tmp_path, "chat")["task"] == "c"
    assert checkpoint.load(tmp_path, "missing.json") is None


def test_secrets_are_redacted_in_checkpoints(tmp_path):
    cfg = Config()
    cfg.log_sessions = True
    cfg.redact_secrets = True
    checkpoint.save(tmp_path, "ask", "token=sk-abcdefghijklmnopqrstuvwxyz",
                    [], "use sk-abcdefghijklmnopqrstuvwxyz", cfg)
    text = checkpoint.latest(tmp_path)["reply"]
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in text


def test_render_is_bounded(tmp_path):
    cfg = Config()
    cfg.log_sessions = True
    checkpoint.save(tmp_path, "ask", "t", [
        {"role": "user", "content": "x" * 5000}], "y" * 5000, cfg)
    rendered = checkpoint.render(checkpoint.latest(tmp_path), max_chars=500)
    assert len(rendered) <= 520
    assert "truncated" in rendered
