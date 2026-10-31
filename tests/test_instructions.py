"""Project instruction files (PEDACEITO.md / AGENTS.md / CLAUDE.md)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import instructions
from pedacito.agent import Agent
from pedacito.config import Config
from pedacito.index import Index


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_first_instruction_file_wins_and_is_labelled(tmp_path):
    _write(tmp_path / "AGENTS.md", "Use ruff.\nNever edit generated/.\n")
    _write(tmp_path / "PEDACEITO.md", "Project-specific rule.\n")
    rendered = instructions.render(tmp_path, Config())
    assert "from PEDACEITO.md" in rendered
    assert "Project-specific rule." in rendered
    assert "Use ruff." not in rendered


def test_instruction_text_is_capped(tmp_path):
    _write(tmp_path / "AGENTS.md", "x" * 5000)
    cfg = Config()
    cfg.max_instruction_chars = 200
    rendered = instructions.render(tmp_path, cfg)
    assert len(rendered) < 500
    assert "truncated" in rendered


def test_instructions_can_be_disabled(tmp_path):
    _write(tmp_path / "AGENTS.md", "Rule.\n")
    cfg = Config()
    cfg.use_project_instructions = False
    assert instructions.render(tmp_path, cfg) == ""


def test_agent_injects_instructions_into_gather_and_answer(tmp_path):
    _write(tmp_path / "AGENTS.md", "ALWAYS use the project's build command.\n")
    idx = Index(tmp_path, [])
    cfg = Config()
    cfg.log_sessions = False

    class Stub:
        stream_truncated = False
        step_truncated = False

        def __init__(self):
            self.seen = []

        def chat_json(self, messages, schema, max_tokens=0, **kw):
            self.seen.append(messages[0]["content"])
            return {"action": "done", "reasoning": "nothing to read"}

        def chat(self, messages, max_tokens=0, **kw):
            self.seen.append(messages[-1]["content"])
            return "No changes: nothing to do."

    client = Stub()
    agent = Agent(idx, client, cfg)
    agent.gather("do the task", verbose=False)
    agent.answer("do the task", "")
    assert any("ALWAYS use the project's build command." in text
               for text in client.seen)


def test_planner_prompt_includes_project_instructions(tmp_path):
    from pedacito import loop

    _write(tmp_path / "AGENTS.md", "The planner must respect generated/.\n")
    seen = []

    class Stub:
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            seen.append(messages[0]["content"])
            return {"tasks": [{"title": "do the task"}]}

    titles = loop.generate_tasks("a goal", Stub(), Config(), root=tmp_path)
    assert titles == ["do the task"]
    assert "must respect generated/" in seen[0]

