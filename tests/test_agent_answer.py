"""Answer-prompt trimming for hosted APIs."""
from __future__ import annotations

from pedacito import index as index_mod
from pedacito.agent import Agent
from pedacito.config import Config


class _Client:
    def __init__(self):
        self.calls = []

    def chat(self, messages, max_tokens, stream=False, **kw):
        self.calls.append(messages)
        return "ok"


def _agent(tmp_path, base_url):
    (tmp_path / "app.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    cfg = Config()
    cfg.base_url = base_url
    cfg.model = "m"
    idx = index_mod.build([str(tmp_path)], cfg, client=None, summarise=False,
                          root=tmp_path)
    client = _Client()
    return Agent(idx, client, cfg), client


def test_hosted_answer_trims_oldest_source_and_terminates(tmp_path):
    agent, client = _agent(tmp_path, "https://api.example.com/v1")
    gathered = "OLDEST-MARK\n" + "x" * 400_000 + "\nNEWEST-MARK"
    agent.answer("task", gathered, stream=False)
    user = client.calls[0][1]["content"]
    assert "NEWEST-MARK" in user
    assert "OLDEST-MARK" not in user
    assert "older gathered source trimmed" in user


def test_hosted_answer_trim_terminates_when_head_alone_exceeds_ceiling(tmp_path):
    agent, client = _agent(tmp_path, "https://api.example.com/v1")
    agent.cfg.max_prompt_tokens = 10
    agent.answer("task", "y" * 4100, stream=False)
    assert client.calls


def test_repair_source_survives_non_utf8_file(tmp_path):
    agent, _client = _agent(tmp_path, "http://127.0.0.1:1234/v1")
    (tmp_path / "bad.py").write_bytes(b"x = '\xe9'\n")

    class _F:
        file = "bad.py"
        message = "search block not found"

    out = agent._repair_source([_F()])
    assert "source unavailable" in out
