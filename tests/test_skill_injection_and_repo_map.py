"""Manual skill injection and the repo-map / KV-prefix regression pins.

The shipped repo map is task-conditioned, deterministic and lives in user0
(the first user turn); the shipped skills library loads a full checklist only
when the model asks for it. These tests pin those contracts and the one added
surface: `/skill NAME` (TUI) and "Use in next task" (GUI), which ride the
existing explicit-pinned-context path.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import config as config_mod
from pedacito import index as index_mod
from pedacito import skills, tools
from pedacito import tui as tui_mod
from pedacito.agent import Agent, _trim_gathered
from pedacito.config import Config
from pedacito.llm import estimate_tokens
from pedacito.tui import TuiSession


BIG = {f"mods/m{i}.py": f"def unused_{i}():\n    return {i}\n"
       for i in range(24)}
BIG.update({
    "core/engine.py": "from core import util\n\n"
                      "def run_engine():\n    return util.helper()\n",
    "core/util.py": "def helper():\n    return 'leech threshold'\n",
    "core/sched.py": ('"""Scheduler."""\n\n'
                      "def is_leech(lapses):\n    return lapses >= 8\n"),
    "tests/test_sched.py": ("from core.sched import is_leech\n\n"
                            "def test_x():\n    assert is_leech(9)\n"),
})


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _project(tmp_path: Path, files: dict[str, str], cfg: Config | None = None):
    root = tmp_path / "proj"
    root.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        _write(root, rel, text)
    return root, index_mod.build([str(root)], cfg or Config(), client=None,
                                 summarise=False, root=root, verbose=False)


class Scripted:
    """One scripted gather step per call; records every message list."""

    step_truncated = False
    stream_truncated = False

    def __init__(self, steps=(), replies=("ok",)):
        self.steps = list(steps)
        self.replies = list(replies)
        self.json_calls: list[list[dict]] = []
        self.chat_calls: list[list[dict]] = []
        self.stats = {"calls": 0, "prompt_tokens": 0,
                      "completion_tokens": 0}

    def chat_json(self, messages, schema, max_tokens, **kw):
        self.json_calls.append([dict(m) for m in messages])
        step = self.steps.pop(0) if self.steps else {"action": "done"}
        return {"reasoning": "r", **step}

    def chat(self, messages, max_tokens, stream=False, **kw):
        self.chat_calls.append([dict(m) for m in messages])
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def _agent(index, client, **cfg_kw):
    cfg = Config()
    cfg.backup = False
    cfg.log_events = False
    cfg.max_gathered_chars = 10 ** 7
    cfg.stuck_recovery = False
    for key, value in cfg_kw.items():
        setattr(cfg, key, value)
    return Agent(index, client, cfg)


# --------------------------------------------------- repo-map regression pins
def test_map_is_byte_identical_for_the_same_index_and_task(tmp_path):
    idx = _project(tmp_path, BIG)[1]
    cfg = Config()
    cfg.flat_map_token_budget = 400                  # exercises the ranked tier
    task = "fix is_leech in core/sched.py"
    first, tier = idx.map_for_prompt(cfg, task)
    assert tier == "ranked"
    # A cache eviction must not change the bytes: ranking is deterministic.
    idx._map_cache.clear()
    again, tier_again = idx.map_for_prompt(cfg, task)
    assert again == first and tier_again == tier
    # A fresh index over the same files produces the same map too.
    idx2 = _project(tmp_path, BIG)[1]
    fresh, fresh_tier = idx2.map_for_prompt(cfg, task)
    assert fresh == first and fresh_tier == tier
    # A different task legitimately re-ranks, but the same task is stable.
    assert idx.map_for_prompt(cfg, task) == (first, tier)
    assert idx.map_for_prompt(
        cfg, task="improve run_engine in core/engine.py")[0] != first


def test_system_and_user0_stay_byte_identical_across_gather_steps(tmp_path):
    idx = _project(tmp_path, {"app.py": "def add(a, b):\n    return a + b\n"})[1]
    client = Scripted([
        {"action": "read", "file": "app.py", "start": 1, "end": 2},
        {"action": "read", "file": "app.py", "start": 1, "end": 2},
        {"action": "done"},
    ])
    agent = _agent(idx, client)
    agent.gather("make add handle negative values", verbose=False)
    assert len(client.json_calls) >= 2
    first_head = client.json_calls[0][:2]
    assert [m["role"] for m in first_head] == ["system", "user"]
    assert "PROJECT MAP" in first_head[1]["content"]
    assert "TASK\nmake add handle negative values" in first_head[1]["content"]
    for call in client.json_calls:
        assert call[:2] == first_head          # no prompt-prefix churn


def test_flat_map_estimate_never_exceeds_flat_map_token_budget(tmp_path):
    # A project that fits the default window: the returned map is the flat
    # tier and its estimate is under the absolute key.
    small = _project(tmp_path / "small", {
        "a.py": "def a():\n    return 1\n",
        "b.py": "def b():\n    return 2\n",
    })[1]
    cfg = Config()
    text, tier = small.map_for_prompt(cfg, task="change a")
    assert tier == "flat" and estimate_tokens(text) <= cfg.flat_map_token_budget

    idx = _project(tmp_path, BIG)[1]
    flat_tokens = estimate_tokens(idx.render())
    # When the flat tier is rich enough to fit, its actual render obeys the
    # absolute key; when it cannot fit, the estimator never hands back a flat
    # map over budget instead of falling back to cards/ranked/dirs.
    for budget in (flat_tokens, flat_tokens + 200, flat_tokens * 4):
        cfg = Config()
        cfg.flat_map_token_budget = budget
        text, tier = idx.map_for_prompt(cfg, task="fix is_leech in core/sched.py")
        if tier == "flat":
            assert estimate_tokens(text) <= budget
    tight = Config()
    tight.flat_map_token_budget = max(200, flat_tokens // 3)
    text, tier = idx.map_for_prompt(tight, task="fix is_leech")
    assert tier != "flat"
    # The compact directory tier is itself budget-bounded.
    tiny = Config()
    tiny.flat_map_token_budget = max(1, estimate_tokens(idx.render_dirs()))
    text, tier = idx.map_for_prompt(tiny, task="fix is_leech")
    assert tier == "dirs" and estimate_tokens(text) <= tiny.flat_map_token_budget


def test_index_excludes_venv_node_modules_state_and_ignore_rules(tmp_path):
    root = tmp_path / "proj"
    _write(root, ".pedacitoignore", "generated/\n")
    _write(root, ".gitignore", "vendor/\n")
    _write(root, "src/app.py", "def app():\n    return 1\n")
    _write(root, ".venv/lib/site-packages/dep.py", "x = 1\n")
    _write(root, "node_modules/pkg/index.js", "module.exports = 1\n")
    _write(root, ".pedacito/state.md", "# state\n")
    _write(root, "generated/out.py", "generated = True\n")
    _write(root, "vendor/lib.py", "vendored = True\n")
    idx = index_mod.build([str(root)], Config(), client=None, summarise=False,
                          root=root, verbose=False)
    files = set(idx.files)
    assert "src/app.py" in files
    for junk in (".venv/lib/site-packages/dep.py",
                 "node_modules/pkg/index.js", ".pedacito/state.md",
                 "generated/out.py", "vendor/lib.py"):
        assert junk not in files


# ------------------------------------------------------- the skills library
def test_skill_precedence_project_over_user_over_bundled(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setattr(config_mod, "USER_CONFIG", home / "config.toml")
    project = tmp_path / "proj"
    project.mkdir()
    user_skill = home / "skills" / "add-a-test.md"
    user_skill.write_text(
        "---\nname: add-a-test\ndescription: User test recipe\n---\n"
        "USER BODY\n", encoding="utf-8")
    project_skill = project / ".pedacito" / "skills" / "add-a-test.md"
    project_skill.parent.mkdir(parents=True)
    project_skill.write_text(
        "---\nname: add-a-test\ndescription: Project test recipe\n---\n"
        "PROJECT BODY\n", encoding="utf-8")
    cfg = Config()
    assert "PROJECT BODY" in skills.load(project, cfg, "add-a-test")[0]
    rows = {r["name"]: r for r in skills.list_skills(project, cfg)}
    assert rows["add-a-test"]["scope"] == "project"
    project_skill.unlink()
    assert "USER BODY" in skills.load(project, cfg, "add-a-test")[0]
    assert {r["name"]: r for r in skills.list_skills(project, cfg)
            }["add-a-test"]["scope"] == "user"
    user_skill.unlink()
    text, info = skills.load(project, cfg, "add-a-test")
    assert info["source"] == "bundled"
    assert skills.BUNDLED["add-a-test"][1].strip() in text


def test_skill_tool_returns_the_full_body(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    body = ("1. Read the component.\n2. Wire the handler.\n"
            "3. Test both themes with a real click.\n")
    path = skills.save_skill(root, name="deploy-notes",
                             description="Release checklist",
                             triggers=["deploy"], body=body)
    idx = index_mod.build([str(root)], Config(), client=None, summarise=False,
                          root=root, verbose=False)
    out = tools.dispatch(idx, Config(), {"action": "skill",
                                         "symbol": "deploy-notes"})
    assert out == f"SKILL deploy-notes ({path}):\n{body.strip()}"


# ------------------------------------- manual injection through pinned context
def test_tui_skill_command_pins_the_full_body_for_the_next_task(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    cfg = Config()
    cfg._root = str(root)
    out = io.StringIO()
    session = TuiSession(root, cfg, headless=True, stdout=out)
    try:
        session._slash("/skill")
        text = "\n".join(line for _kind, line in session.transcript)
        assert "Usage: /skill <name>" in text
        assert session.pins == []

        session._slash("/skill definitely-not-a-skill")
        text = "\n".join(line for _kind, line in session.transcript)
        assert "no skill named" in text and "add-a-test" in text
        assert session.pins == []

        session._slash("/skill add-a-test")
        text = "\n".join(line for _kind, line in session.transcript)
        assert "Skill 'add-a-test' pinned for the next task" in text
        assert session.pins == ["skill:add-a-test"]
        assert cfg._pinned_refs == ["skill:add-a-test"]

        # Registered for completion and documented in /help.
        assert "/skill" in tui_mod._SLASH_COMMANDS
        assert "/skill NAME" in tui_mod.HELP_TEXT

        session._slash("/pin remove skill:add-a-test")
        assert session.pins == [] and cfg._pinned_refs == []
    finally:
        session.close()


def test_skill_pin_lands_in_pinned_context_and_survives_pruning(tmp_path):
    idx = _project(tmp_path, {"app.py": "def add(a, b):\n    return a + b\n"})[1]
    client = Scripted([
        {"action": "read", "file": "app.py", "start": 1, "end": 2},
        {"action": "done"},
    ])
    agent = _agent(idx, client)
    agent.cfg._pinned_refs = ["skill:add-a-test"]
    agent.gather("add a focused test for add", verbose=False)

    body = skills.BUNDLED["add-a-test"][1].strip()
    assert "### Explicitly Pinned Context" in client.json_calls[0][1]["content"]
    assert "--- pinned skill add-a-test" in agent.pinned_context
    assert body in agent.pinned_context
    assert body in client.json_calls[0][1]["content"]
    # The pin is classified like @file/@symbol and never evicted wholesale.
    kept, dropped = _trim_gathered(agent.gathered_source,
                                   lambda text: len(text) > 300)
    assert body in kept and dropped >= 1


def test_gui_pin_payload_reaches_the_same_pinned_refs(tmp_path):
    from pedacito.gui import jobs

    cfg = Config()
    jobs._set_pinned_refs(cfg, {"pinned": ["skill:add-a-test", "@app.py"]})
    assert cfg._pinned_refs == ["skill:add-a-test", "app.py"]
    js = (Path(__file__).resolve().parents[1] / "pedacito" / "gui" /
          "static" / "app.js").read_text(encoding="utf-8")
    assert "Use in next task" in js
    assert "Pinned for the next task" in js
    assert "skillRef(r.name)" in js
