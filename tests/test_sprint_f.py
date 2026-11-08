"""Sprint F: deterministic gather tools + TokenEconomizer techniques.

Covers the compact `outline` TOC, `usages` call/import lookup, `git_diff`,
repository micro-facts (confirmation and contradiction retirement), dynamic
tool schema pruning, and the pre-lease loop-guard redirect.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import memory, tools
from pedacito.agent import Agent
from pedacito.chunker import chunk_source
from pedacito.config import Config
from pedacito.index import Index


APP = '''\
"""demo module"""


class Server:
    def start(self):
        return helper(1)

    def stop(self):
        return helper(2)


def helper(value):
    return value + 1


def run():
    s = Server()
    return s.start()
'''


def _index(tmp_path: Path, files: dict[str, str] | None = None) -> Index:
    files = files or {"pkg/app.py": APP}
    chunks = []
    for rel, src in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src, encoding="utf-8")
        chunks.extend(chunk_source(rel, src, 60))
    return Index(tmp_path, chunks)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.email=a@b.c",
                    "-c", "user.name=tester", *args], cwd=root, check=True,
                   capture_output=True)


# --------------------------------------------------------------- new tools
def test_outline_is_a_compact_toc_with_exact_ranges(tmp_path):
    idx = _index(tmp_path)
    out = tools.tool_outline(idx, Config(), "pkg/app.py")
    assert "### pkg/app.py" in out
    assert "class Server [4-9]" in out
    assert "def start(self) [5-6]" in out
    assert "def helper(value) [12-13]" in out
    assert "return helper(1)" not in out          # bodies stay out of the TOC
    assert "L5-6" not in out                     # exact start-end, not L-prefixed


def test_usages_returns_deduped_call_sites_and_imports(tmp_path):
    idx = _index(tmp_path, {
        "pkg/app.py": APP,
        "pkg/use.py": ("from pkg.app import Server, helper\n\n"
                       "def go():\n"
                       "    s = Server()\n"
                       "    return helper(s.start())\n"),
    })
    out = tools.tool_usages(idx, Config(), "helper")
    assert "defined at pkg/app.py:12-13" in out
    assert "pkg/app.py:6  return helper(1)" in out
    assert "pkg/app.py:9  return helper(2)" in out
    assert "pkg/use.py:5  return helper(s.start())" in out
    assert "pkg/use.py:1  from pkg.app import Server, helper" in out
    assert out.count("pkg/use.py:5") == 1          # deduplicated
    assert "imports (" in out.lower()


def test_usages_missing_symbol_gives_a_next_step(tmp_path):
    idx = _index(tmp_path)
    out = tools.tool_usages(idx, Config(), "never_defined")
    assert "No definition" in out and "grep" in out


def test_git_diff_reports_status_and_uncommitted_diff(tmp_path):
    idx = _index(tmp_path)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "initial")
    (tmp_path / "pkg/app.py").write_text(APP + "\n# changed\n", encoding="utf-8")

    out = tools.dispatch(idx, Config(), {"action": "git_diff"})
    assert "STATUS (git status --short)" in out
    assert "M pkg/app.py" in out
    assert "+# changed" in out
    assert "read-only" in out.lower()


def test_git_diff_can_narrow_to_one_file_and_is_token_capped(tmp_path):
    idx = _index(tmp_path, {
        "pkg/app.py": "x = 1\n",
        "pkg/other.py": "y = 2\n",
        "pkg/big.py": "\n".join(f"value_{i} = {i}" for i in range(300)) + "\n",
    })
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "initial")
    (tmp_path / "pkg/app.py").write_text("x = 2\n", encoding="utf-8")
    (tmp_path / "pkg/other.py").write_text("y = 3\n", encoding="utf-8")
    (tmp_path / "pkg/big.py").write_text(
        "\n".join(f"changed_{i} = {i}" for i in range(300)) + "\n",
        encoding="utf-8")

    out = tools.dispatch(idx, Config(), {"action": "git_diff",
                                         "file": "pkg/app.py"})
    assert "M pkg/app.py" in out
    assert "pkg/other.py" not in out and "pkg/big.py" not in out

    cfg = Config()
    cfg.max_read_tokens = 20                       # ~80 chars, floor at 400
    capped = tools.dispatch(idx, cfg, {"action": "git_diff"})
    assert "[diff capped at" in capped


# ------------------------------------------------------------ micro-facts
def test_micro_fact_promotes_after_failure_then_success_and_retires(tmp_path):
    cfg = Config()
    command = "python -m pytest -q"
    assert memory.render_micro_facts(tmp_path, cfg) == ""
    memory.observe_command(tmp_path, command, False, cfg, kind="test")
    assert memory.render_micro_facts(tmp_path, cfg) == ""      # still pending
    memory.observe_command(tmp_path, command, True, cfg, kind="test")

    facts = memory.load_micro_facts(tmp_path)
    assert len(facts) == 1
    fact = facts[0]
    assert {"fact_id", "text", "confirmations", "contradictions",
            "last_seen"} <= set(fact)
    assert fact["confirmations"] == 1 and fact["contradictions"] == 0
    rendered = memory.render_micro_facts(tmp_path, cfg)
    assert command in rendered and "REPO MICRO-FACTS" in rendered

    # A later failure contradicts the stored fact: retire it immediately.
    memory.observe_command(tmp_path, command, False, cfg, kind="test")
    assert memory.load_micro_facts(tmp_path) == []
    retired = memory.load_micro_facts(tmp_path, include_retired=True)
    assert retired and retired[0]["contradictions"] == 1
    assert command not in memory.render_micro_facts(tmp_path, cfg)


def test_micro_fact_promotes_after_two_successful_runs(tmp_path):
    cfg = Config()
    command = "ruff check ."
    memory.observe_command(tmp_path, command, True, cfg)
    assert memory.load_micro_facts(tmp_path) == []
    memory.observe_command(tmp_path, command, True, cfg)
    facts = memory.load_micro_facts(tmp_path)
    assert len(facts) == 1
    assert facts[0]["confirmations"] == 2
    assert command in memory.render_micro_facts(tmp_path, cfg)


def test_micro_facts_respect_the_token_cap(tmp_path):
    cfg = Config()
    cfg.max_micro_fact_tokens = 20
    for i in range(8):
        command = f"python -m pytest tests/suite_{i}.py -q"
        memory.observe_command(tmp_path, command, True, cfg, kind="test")
        memory.observe_command(tmp_path, command, True, cfg, kind="test")
    text = memory.render_micro_facts(tmp_path, cfg)
    assert text
    assert len(text) <= 20 * 4 + 120       # head + bounded rows


# ------------------------------------------------------ schema pruning
def test_dynamic_tool_pruning_drops_stale_optional_tools(tmp_path):
    cfg = Config()
    cfg._root = str(tmp_path)
    cfg._task_text = "polish the parser"
    cfg.allow_fetch = True
    cfg.allow_subagents = True
    for _ in range(10):
        memory.begin_tool_task(tmp_path, cfg)
    # Core navigation survives even though none of it was ever used.
    schema = tools.schema_for(cfg)
    enum = schema["properties"]["action"]["enum"]
    assert {"read", "grep", "find", "outline", "usages", "done"} <= set(enum)
    assert "fetch" not in enum
    assert "task" not in enum
    assert "git_diff" not in enum

    # A trigger word brings the optional tool back for this task.
    cfg._task_text = "fetch https://example.com and review uncommitted changes"
    enum = tools.schema_for(cfg)["properties"]["action"]["enum"]
    assert "fetch" in enum and "git_diff" in enum

    # A recently used tool is not pruned even without a trigger word.
    cfg._task_text = "polish the parser"
    memory.record_tool_use(tmp_path, "fetch", cfg)
    enum = tools.schema_for(cfg)["properties"]["action"]["enum"]
    assert "fetch" in enum


def test_dynamic_tool_pruning_can_be_disabled(tmp_path):
    cfg = Config()
    cfg._root = str(tmp_path)
    cfg._task_text = "polish the parser"
    cfg.allow_fetch = True
    cfg.dynamic_tool_pruning = False
    for _ in range(12):
        memory.begin_tool_task(tmp_path, cfg)
    assert "fetch" in tools.schema_for(cfg)["properties"]["action"]["enum"]


# -------------------------------------------------------- loop guard
class _Scripted:
    step_truncated = False

    def __init__(self, steps):
        self.steps = list(steps)
        self.json_calls: list[list[dict]] = []
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def chat_json(self, messages, schema, max_tokens, **kw):
        self.json_calls.append([dict(m) for m in messages])
        if self.steps:
            return {"reasoning": "r", **self.steps.pop(0)}
        return {"action": "done", "reasoning": "enough"}


def test_loop_guard_injects_one_redirect_before_the_no_progress_lease(tmp_path):
    idx = _index(tmp_path)
    cfg = Config()
    cfg.log_events = False
    cfg.stuck_recovery = False
    cfg.max_total_model_calls = 0
    cfg.max_no_progress_steps = 3
    cfg.max_gathered_chars = 10 ** 7
    client = _Scripted([
        {"action": "grep", "pattern": "no_such_term_1"},
        {"action": "grep", "pattern": "no_such_term_2"},
        {"action": "grep", "pattern": "no_such_term_3"},
    ])
    Agent(idx, client, cfg).gather("look at the app", verbose=False)

    texts = [m["content"] for call in client.json_calls for m in call]
    guards = [t for t in texts if "[LOOP GUARD:" in t]
    assert len(guards) == 1
    assert "2 steps without adding new source lines" in guards[0]
    assert "outline/usages" in guards[0]
    assert "emit 'done' now" in guards[0]
    # The redirect is seen by the model, then the hard lease still ends gather.
    assert len(client.json_calls) == 3


def test_agent_gather_injects_confirmed_micro_facts_in_the_stable_prefix(tmp_path):
    idx = _index(tmp_path)
    cfg = Config()
    cfg.log_events = False
    cfg.stuck_recovery = False
    cfg.max_total_model_calls = 0
    command = "python -m pytest -q"
    memory.observe_command(tmp_path, command, True, cfg, kind="test")
    memory.observe_command(tmp_path, command, True, cfg, kind="test")

    client = _Scripted([{"action": "done"}])
    Agent(idx, client, cfg).gather("look at the app", verbose=False)
    system = client.json_calls[0][0]["content"]
    assert "REPO MICRO-FACTS" in system
    assert command in system
