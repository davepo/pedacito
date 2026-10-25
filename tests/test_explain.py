"""Tests for `pedacito explain` (prose answers about an indexed project)."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import explain
from pedacito import index as index_mod
from pedacito.agent import Agent, EDIT_FORMAT
from pedacito.cli import main
from pedacito.config import Config

FAKE = Path(__file__).resolve().parent / "fake_llm.py"


class _CapturingClient:
    """Records the messages sent to `chat` and returns a fixed reply."""

    def __init__(self, reply: str = "ok"):
        self.reply = reply
        self.calls: list[list[dict]] = []

    def chat(self, messages, max_tokens, stream=False, schema=None,
             disable_thinking=None):
        self.calls.append(messages)
        return self.reply


def _make_agent(tmp_path: Path):
    (tmp_path / "app.py").write_text(
        "def f(x):\n    return x + 1\n", encoding="utf-8")
    cfg = Config()
    cfg.base_url = "http://127.0.0.1:1234/v1"
    cfg.model = "fake-model"
    idx = index_mod.build([str(tmp_path)], cfg, client=None, summarise=False,
                          root=tmp_path)
    client = _CapturingClient()
    return Agent(idx, client, cfg), client, cfg


def test_no_index_prints_direction_and_returns_zero(tmp_path, monkeypatch,
                                                    capsys):
    monkeypatch.chdir(tmp_path)
    code = main(["explain", ".", "-q", "how does this work?"])
    assert code == 0
    err = capsys.readouterr().err
    assert "No index found" in err
    assert "pedacito index" in err


def test_answer_omits_edit_format(tmp_path):
    agent, client, _cfg = _make_agent(tmp_path)
    agent.answer("what does app.py do?", "def f(x):\n    return x + 1\n",
                 system=explain.EXPLAIN_SYSTEM, include_edit_format=False,
                 stream=False)
    system_prompt = client.calls[0][0]["content"]
    assert "SEARCH" not in system_prompt
    assert EDIT_FORMAT.splitlines()[0].strip() not in system_prompt


def test_answer_includes_edit_format_by_default(tmp_path):
    agent, client, _cfg = _make_agent(tmp_path)
    agent.answer("change app.py", "def f(x):\n    return x + 1\n",
                 system="You are a code assistant.",
                 stream=False)
    system_prompt = client.calls[0][0]["content"]
    assert "SEARCH" in system_prompt


@pytest.fixture(scope="module")
def fake_port():
    port = _free_port()
    env = dict(os.environ, FAKE_PORT=str(port),
               FAKE_REPLY="app.py defines f, which adds one to its argument.")
    env.pop("FAKE_ACTIONS", None)
    proc = subprocess.Popen([sys.executable, str(FAKE)], env=env,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), 0.5):
                    break
            except OSError:
                if proc.poll() is not None:
                    pytest.fail("fake_llm exited early")
                time.sleep(0.1)
        else:
            pytest.fail("fake_llm did not start")
        yield port
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.mark.network
def test_explain_answer_end_to_end(tmp_path, fake_port):
    (tmp_path / "old_name.py").write_text('print("hi")', encoding="utf-8")
    cfg = Config()
    cfg.base_url = f"http://127.0.0.1:{fake_port}/v1"
    cfg.model = "fake-model"
    cfg.max_steps = 6
    idx = index_mod.build([str(tmp_path)], cfg, client=None, summarise=False,
                          root=tmp_path)
    from pedacito.llm import LMStudio
    agent = Agent(idx, LMStudio(cfg), cfg)
    reply = explain.answer(agent, "what does old_name.py do?", verbose=False)
    assert reply.strip()
    assert "--- FILE:" not in reply
