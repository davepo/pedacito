"""Checklist scratchpad and its prompt injection."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import todo, tools
from pedacito.agent import Agent
from pedacito.config import Config
from pedacito.index import Index


def test_todo_round_trip_and_cap(tmp_path):
    text = "- [x] inspect\n- [ ] implement"
    todo.write(tmp_path, text)
    assert todo.read(tmp_path) == text
    rendered = todo.render(tmp_path, Config(), limit=8)
    assert "CURRENT TASK CHECKLIST" in rendered
    assert "truncated" in rendered
    todo.clear(tmp_path)
    assert todo.read(tmp_path) == ""


def test_todo_tool_writes_and_reads(tmp_path):
    idx = Index(tmp_path, [])
    obs = tools.dispatch(idx, Config(), {
        "action": "todo", "content": "- [ ] one step"})
    assert "CHECKLIST UPDATED" in obs
    assert "one step" in todo.read(tmp_path)
    read_back = tools.dispatch(idx, Config(), {"action": "todo", "content": ""})
    assert "one step" in read_back


def test_agent_injects_the_checklist_into_answer_prompt(tmp_path):
    todo.write(tmp_path, "- [ ] finish the parser")
    idx = Index(tmp_path, [])
    cfg = Config()

    class Stub:
        def __init__(self):
            self.seen = []

        def chat(self, messages, max_tokens=0, **kw):
            self.seen.append(messages[-1]["content"])
            return "No changes: done."

    client = Stub()
    agent = Agent(idx, client, cfg)
    agent.answer("do it", "")
    assert any("finish the parser" in text for text in client.seen)
