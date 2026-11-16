"""The skills library & creator and session consolidation upgrades.

Hermetic: scripted draft clients, tmp project roots, monkeypatched user-config
and state dirs, and the shared GUI server fixture -- no network, no real model
server. Covers select_skills caps and trigger semantics, save/delete/draft,
the `skills` and `memory --consolidate` CLI, the chat slash commands, the RPC
`consolidate`/`compact`/`op` dispatch, and the GUI skills + consolidate
endpoints.
"""
# `gui` below is test_gui's server fixture: pytest hands it to each test as a
# parameter, which ruff otherwise reads as a redefinition.
# ruff: noqa: F811
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pedacito import cli, memory, rpc, sessions, skills, workspace
from pedacito import config as config_mod
from pedacito import llm as llm_mod
from pedacito.config import Config, ConfigError
from pedacito.gui import core
from pedacito.llm import estimate_messages, estimate_tokens
from test_gui import gui  # noqa: F401  (the shared server fixture)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolated user config AND state dirs (skills, memory, global sessions)."""
    monkeypatch.setattr(config_mod, "USER_CONFIG",
                        tmp_path / "home" / "config.toml")
    monkeypatch.setattr(config_mod, "USER_STATE_DIR", tmp_path / "state")
    (tmp_path / "home").mkdir()
    return tmp_path / "home"


@pytest.fixture
def proj(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    return root


class FakeDraftClient:
    """Scripted chat() reply for draft_skill_with_llm and the GUI endpoint."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def chat(self, messages, **kw):
        self.calls.append((messages, kw))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _draft_payload(**over):
    data = {"name": "alpine-dark-mode",
            "description": "Build Alpine.js components with a dark-mode toggle",
            "triggers": ["alpine", "dark mode", "x-data", "toggle"],
            "body": "1. Read the component.\n2. Wire x-data.\n3. Test both themes."}
    data.update(over)
    return data


# ------------------------------------------------------- select_skills caps

def test_select_skills_caps_at_two_and_1200_chars(proj, home):
    task = "fix the failing test and add a test with pytest coverage"
    picked = skills.select_skills(proj, task)
    assert [r["name"] for r in picked] == ["add-a-test", "fix-a-failing-test"]
    assert picked[0]["score"] > picked[1]["score"]      # phrase + words beat one
    assert len(picked) <= skills.MAX_SELECTED_SKILLS == 2
    assert sum(len(r["body"]) for r in picked) <= skills.MAX_SELECTED_CHARS == 1200
    # A tighter char budget truncates bodies instead of dropping the contract.
    tight = skills.select_skills(proj, task, max_chars=300)
    assert sum(len(r["body"]) for r in tight) <= 300
    assert skills.select_skills(proj, task, limit=1)[0]["name"] == "add-a-test"
    assert skills.select_skills(proj, task, limit=0) == []


def test_trigger_scoring_semantics(proj, home):
    # Whole word beats nothing; short triggers never match inside a word.
    assert skills._trigger_score(("ast",), "build an ast for the parser") == 1.0
    assert skills._trigger_score(("ast",), "add a fastapi endpoint") == 0.0
    assert skills._trigger_score(("sql",), "mysql query") == 0.0
    # Multi-word phrases are the strongest signal.
    assert skills._trigger_score(("race condition",),
                                 "fix a race condition in the pool") == 2.0
    # Long triggers may match camelCase identifiers as substrings.
    assert skills._trigger_score(("requestanimationframe",),
                                 "the requestanimationframeloop stacks") == 0.5
    picked = skills.select_skills(proj, "add a fastapi endpoint")
    assert [r["name"] for r in picked] == ["fastapi-rest-endpoint"]
    assert "parser-lexer-state-machine" not in [r["name"] for r in picked]
    picked = skills.select_skills(proj, "fix a race condition in the worker")
    assert picked[0]["name"] == "async-concurrency-safety"


def test_select_skills_off_empty_and_unmatched(proj, home):
    off = Config()
    off.use_skills = False
    assert skills.select_skills(proj, "add a test", off) == []
    assert skills.select_skills(proj, "") == []
    assert skills.select_skills(proj, "   ") == []
    assert skills.select_skills(proj, "zzz qqq nothing matches here") == []
    block = skills.render_selected(proj, "fix the failing test please")
    assert block.startswith("SKILLS FOR THIS TASK")
    assert "## fix-a-failing-test" in block
    assert len(block) <= skills.MAX_SELECTED_CHARS
    assert skills.render_selected(proj, "zzz qqq nothing") == ""


# ------------------------------------------------- save / delete management

def test_save_skill_project_and_user_scopes(proj, home):
    path = skills.save_skill(proj, name="deploy-notes",
                             description="Release checklist",
                             triggers=["deploy", "release"],
                             body="1. Tag the release.\n2. Publish.")
    assert path == proj / ".pedacito" / "skills" / "deploy-notes.md"
    text, info = skills.load(proj, Config(), "deploy-notes")
    assert "Tag the release." in text and info["source"] == str(path)
    rows = {r["name"]: r for r in skills.list_skills(proj)}
    assert rows["deploy-notes"]["scope"] == "project"
    assert rows["deploy-notes"]["triggers"] == ["deploy", "release"]
    assert rows["add-a-test"]["scope"] == "bundled"

    gpath = skills.save_skill(proj, name="deploy-notes",
                              description="Global release checklist",
                              triggers=["deploy"], body="1. Global steps.",
                              scope="user")
    assert gpath == home / "skills" / "deploy-notes.md"
    # The project copy wins the name; the user copy is still listed.
    rows = {r["name"]: r for r in skills.list_skills(proj)}
    assert rows["deploy-notes"]["scope"] == "project"
    assert "Global release checklist" in gpath.read_text(encoding="utf-8")
    # "global" is an accepted alias for the user scope.
    assert skills._scope_dir(proj, "global") == home / "skills"


def test_save_skill_validation(proj, home):
    kw = dict(description="d", triggers=["t"], body="1. step")
    for bad in ("Bad Name", "-lead", "_lead", "a" * 42, "", "has space",
                "drop/tables"):
        with pytest.raises(ValueError):
            skills.save_skill(proj, name=bad, **kw)
    assert skills._validate_slug("  My-Skill  ") == "my-skill"
    with pytest.raises(ValueError):
        skills.save_skill(proj, name="ok-name", description="  ",
                          triggers=[], body="1. step")
    with pytest.raises(ValueError):
        skills.save_skill(proj, name="ok-name", description="d",
                          triggers=[], body="   ")
    with pytest.raises(ValueError):
        skills.save_skill(proj, name="ok-name", description="d",
                          triggers=[], body="x" * (skills.MAX_SKILL_CHARS + 1))
    with pytest.raises(ValueError):
        skills.save_skill(None, name="ok-name", scope="project", **kw)
    with pytest.raises(ValueError):
        skills.save_skill(proj, name="ok-name", scope="bogus", **kw)
    # Long descriptions are trimmed to the catalog's one-line budget.
    path = skills.save_skill(proj, name="long-desc", description="d" * 300,
                             triggers=[], body="1. step")
    assert len(skills._scan(proj, Config())["long-desc"]["description"]) <= 120
    assert path.is_file()


def test_save_skill_updates_folder_layout_in_place(proj, home):
    folder = proj / ".pedacito" / "skills" / "deploy"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        "---\nname: deploy\ndescription: old\n---\nold body\n",
        encoding="utf-8")
    path = skills.save_skill(proj, name="deploy", description="new desc",
                             triggers=["deploy"], body="1. new body")
    assert path == folder / "SKILL.md"          # layout preserved
    assert not (folder.parent / "deploy.md").exists()
    assert "new body" in path.read_text(encoding="utf-8")
    assert skills.delete_skill(proj, "deploy") is True
    assert not folder.exists()                  # empty folder cleaned up


def test_delete_skill_refuses_builtins_and_unshadows(proj, home):
    assert skills.delete_skill(proj, "add-a-test") is False
    assert skills.delete_skill(proj, "no-such-skill") is False
    # A custom file shadowing a built-in is deletable; the built-in returns.
    skills.save_skill(proj, name="add-a-test", description="custom override",
                      triggers=["custom"], body="1. custom steps")
    row = {r["name"]: r for r in skills.list_skills(proj)}["add-a-test"]
    assert row["scope"] == "project" and row["description"] == "custom override"
    assert skills.delete_skill(proj, "add-a-test") is True
    row = {r["name"]: r for r in skills.list_skills(proj)}["add-a-test"]
    assert row["scope"] == "bundled"
    with pytest.raises(ValueError):
        skills.delete_skill(proj, "Bad Name")


# ------------------------------------------------------------------- draft

def test_draft_skill_with_llm_plain_json(proj, home):
    client = FakeDraftClient(json.dumps(_draft_payload()))
    draft = skills.draft_skill_with_llm(client, "Alpine.js dark-mode components")
    assert draft == _draft_payload()
    messages, kw = client.calls[0]
    assert kw == dict(max_tokens=700, stream=False, disable_thinking=True)
    assert messages[0]["role"] == "system"
    assert "Alpine.js dark-mode components" in messages[1]["content"]
    # The draft saves cleanly through the normal path.
    path = skills.save_skill(proj, scope="project", **draft)
    assert path.is_file()


def test_draft_skill_fenced_json_slug_and_fallbacks(proj, home):
    payload = _draft_payload(name="Alpine Dark Mode!!",
                             triggers="alpine, dark mode, toggle")
    client = FakeDraftClient("```json\n" + json.dumps(payload) + "\n```")
    draft = skills.draft_skill_with_llm(client, "dark mode")
    assert draft["name"] == "alpine-dark-mode"
    assert draft["triggers"] == ["alpine", "dark mode", "toggle"]
    # No name in the reply -> slugified prompt; no description -> generated.
    client = FakeDraftClient(json.dumps({"body": "1. step", "triggers": []}))
    draft = skills.draft_skill_with_llm(client, "Deploy the release safely")
    assert draft["name"] == "deploy-the-release-safely"
    assert draft["description"].startswith("Procedure for:")
    # Unslugifiable prompt -> the stable fallback name.
    client = FakeDraftClient(json.dumps({"body": "1. step"}))
    assert skills.draft_skill_with_llm(client, "!!!")["name"] == "custom-skill"


def test_draft_skill_rejects_bad_input_and_replies(proj, home):
    with pytest.raises(ValueError):
        skills.draft_skill_with_llm(FakeDraftClient("{}"), "   ")
    for reply in ("no json here", "{broken json", "",
                  json.dumps({"name": "x", "description": "d", "body": "  "})):
        with pytest.raises(ValueError):
            skills.draft_skill_with_llm(FakeDraftClient(reply), "a skill")
    # Over-long bodies are trimmed to the crisp built-in budget.
    long_body = "\n".join(f"{i}. step number {i} with padding text"
                          for i in range(1, 40))
    client = FakeDraftClient(json.dumps(_draft_payload(body=long_body)))
    draft = skills.draft_skill_with_llm(client, "long skill")
    assert len(draft["body"]) <= skills.MAX_BUILTIN_BODY_CHARS == 550


# --------------------------------------------------------------------- CLI

def _run(argv, capsys):
    code = cli.main(argv)
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def test_cli_skills_list_show_add_rm(proj, home, capsys):
    code, out, _ = _run(["skills", "list", str(proj)], capsys)
    assert code == 0
    assert "add-a-test" in out and "[bundled]" in out
    assert "fastapi-rest-endpoint" in out and "performance-memory-profiling" in out

    code, out, _ = _run(["skills", "add", "deploy-notes", str(proj),
                         "--description", "Release checklist",
                         "--triggers", "deploy, release",
                         "--body", "1. Tag the release."], capsys)
    assert code == 0 and "Saved skill 'deploy-notes' (project)" in out
    assert (proj / ".pedacito" / "skills" / "deploy-notes.md").is_file()

    code, out, _ = _run(["skills", "show", "deploy-notes", str(proj)], capsys)
    assert code == 0 and "[project]" in out
    assert "deploy, release" in out and "Tag the release." in out
    code, _, err = _run(["skills", "show", "nope", str(proj)], capsys)
    assert code == 1 and "No skill named 'nope'" in err

    code, _, err = _run(["skills", "rm", "add-a-test", str(proj)], capsys)
    assert code == 1 and "built-in" in err
    code, out, _ = _run(["skills", "rm", "deploy-notes", str(proj)], capsys)
    assert code == 0 and "Deleted skill 'deploy-notes'" in out
    assert not (proj / ".pedacito" / "skills" / "deploy-notes.md").exists()

    code, _, err = _run(["skills", "add", "Bad Name", str(proj),
                         "--description", "d", "--body", "1. x"], capsys)
    assert code == 2 and "invalid skill name" in err
    code, _, err = _run(["skills", "show"], capsys)
    assert code == 2 and "Usage" in err

    # --global writes the user library; a bare project path still lists.
    code, out, _ = _run(["skills", "add", "global-note", str(proj), "--global",
                         "--description", "d", "--body", "1. x"], capsys)
    assert code == 0 and "(user)" in out
    assert (home / "skills" / "global-note.md").is_file()
    code, out, _ = _run(["skills", str(proj)], capsys)
    assert code == 0 and "global-note" in out and "[user" in out


def test_cli_skills_draft(proj, home, capsys, monkeypatch):
    client = FakeDraftClient(json.dumps(_draft_payload()))
    monkeypatch.setattr(cli, "_config", lambda args: Config())
    monkeypatch.setattr(cli, "LLMClient", lambda cfg: client)
    code, out, _ = _run(["skills", "draft", "Alpine.js dark-mode components",
                         str(proj)], capsys)
    assert code == 0 and "Drafted skill 'alpine-dark-mode' (project)" in out
    assert (proj / ".pedacito" / "skills" / "alpine-dark-mode.md").is_file()
    code, _, err = _run(["skills", "draft"], capsys)
    assert code == 2 and "Usage" in err
    # A model that returns prose fails cleanly, leaving no file behind.
    monkeypatch.setattr(cli, "LLMClient",
                        lambda cfg: FakeDraftClient("sorry, no JSON"))
    code, _, err = _run(["skills", "draft", "another skill", str(proj)], capsys)
    assert code == 1 and "skills draft:" in err


def test_cli_memory_consolidate(proj, home, capsys):
    _write_transcripts(proj)
    code, out, _ = _run(["memory", str(proj), "--consolidate"], capsys)
    assert code == 0
    assert "Consolidated session:" in out and "Checkpoint:" in out
    for head in memory.HANDOFF_SECTIONS:
        assert f"## {head}" in out
    assert (proj / ".pedacito" / "consolidated_session.md").is_file()
    assert "Pinned" in out                       # the decision reached memory.md


def test_chat_consolidate_and_compact_commands(proj, home, capsys):
    _write_transcripts(proj)
    args = SimpleNamespace(paths=[str(proj)], root=None)
    ref = {"cfg": Config(), "resume_context": ""}
    assert cli._chat_command("/consolidate", args, ref["cfg"], ref) is True
    err = capsys.readouterr().err
    assert "consolidated:" in err and "handoff" in err
    assert ref["resume_context"].startswith("CONSOLIDATED SESSION HANDOFF")
    assert "## Goal & Context" in ref["resume_context"]
    # The alias shares the same live buffer.
    ref["resume_context"] = ""
    assert cli._chat_command("/compact", args, ref["cfg"], ref) is True
    assert ref["resume_context"].startswith("CONSOLIDATED SESSION HANDOFF")
    assert "/consolidate" in cli.CHAT_HELP and "/compact" in cli.CHAT_HELP


# ------------------------------------------------------ consolidate_session

def _write_transcripts(root):
    """Three session logs with edits, a decision, a test result and a next
    step -- enough text that consolidation must shrink the token count.
    app.py exists on disk so the history is current: a transcripts-only
    project is a clean slate, whose handoff is archived instead of seeded."""
    (root / "app.py").write_text("print('app')\n", encoding="utf-8")
    reply = ("Done. We decided to keep the flat skill layout.\n"
             "12 passed in 0.42s (pytest).\n"
             "Next step: wire the GUI button.\n"
             + "Filler detail line about the change.\n" * 30)
    results = [SimpleNamespace(ok=True, file="app.py", message="ok"),
               SimpleNamespace(ok=False, file="old.py", message="anchor missing")]
    for i in range(3):
        workspace.write_session(root, f"task number {i}", ["read app.py"],
                                reply, results, None, f"2026010{i}-000000")


def test_session_task_skips_the_auto_goal_preamble():
    """`auto` writes a two-line Task field; the listing and the handoff must
    show the goal's own heading, not the harness preamble."""
    text = ("# 20260101-000000\n\n"
            "**Task:** ORIGINAL USER GOAL (background -- the CURRENT TASK "
            "below is the step to do now; every constraint in the goal still "
            "binds it):\n"
            "# sensor_vault - 4-task specification\n\n## Lookups\n")
    assert sessions._task(text) == "# sensor_vault - 4-task specification"
    # an ordinary one-line task is untouched, even one ending in ':'
    plain = "**Task:** fix the login timeout\n"
    assert sessions._task(plain) == "fix the login timeout"
    colon = "**Task:** reproduce the failure:\n\n## Lookups\n"
    assert sessions._task(colon) == "reproduce the failure:"


def test_consolidate_from_transcripts(proj, home):
    _write_transcripts(proj)
    res = memory.consolidate_session(proj, Config())
    assert set(res) == {"before_tokens", "after_tokens", "summary", "path",
                        "pinned"}
    assert res["before_tokens"] > res["after_tokens"]
    assert res["after_tokens"] == estimate_tokens(res["summary"]) < 400
    assert len(res["summary"]) <= memory.MAX_HANDOFF_CHARS
    for head in memory.HANDOFF_SECTIONS:
        assert f"## {head}" in res["summary"]
    assert "task number 2" in res["summary"]            # newest task first
    assert "app.py" in res["summary"] and "old.py" in res["summary"]
    assert "12 passed" in res["summary"]
    assert "flat skill layout" in res["summary"]
    assert "wire the GUI button" in res["summary"]
    path = Path(res["path"])
    assert path == proj / ".pedacito" / "consolidated_session.md"
    assert path.read_text(encoding="utf-8") == res["summary"]
    # The durable decision was pinned to project memory.
    assert res["pinned"] >= 1
    assert any("flat skill layout" in text
               for _sc, _cat, text in memory.entries(proj))
    assert memory.handoff_prompt(res["summary"]).startswith(
        "CONSOLIDATED SESSION HANDOFF")


def test_consolidate_live_messages(proj, home):
    msgs = [
        {"role": "user", "content": "PROJECT MAP (truncated)\nTASK\n"
                                    "fix the login timeout"},
        {"role": "assistant", "content":
            "applied `auth.py`\nWe decided to raise the timeout to 30s.\n"
            "3 passed (pytest).\nNext step: add a regression test."},
        {"role": "user", "text": "now add the regression test"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "applied `tests/test_auth.py`"}]},
    ]
    res = memory.consolidate_session(proj, Config(), msgs)
    assert res["before_tokens"] == estimate_messages(msgs)
    assert "fix the login timeout" in res["summary"]
    assert "now add the regression test" in res["summary"]
    assert "auth.py" in res["summary"] and "tests/test_auth.py" in res["summary"]
    assert "timeout to 30s" in res["summary"]
    assert res["after_tokens"] < 400
    # Observation turns are not tasks.
    res2 = memory.consolidate_session(proj, Config(),
                                      [{"role": "user", "content": "RESULT: ok"}])
    assert "RESULT: ok" not in res2["summary"]


def test_consolidate_empty_history_and_cap(proj, home):
    res = memory.consolidate_session(proj, Config())
    assert res["before_tokens"] >= 1 and res["after_tokens"] < 400
    assert "No prior task recorded" in res["summary"]
    for head in memory.HANDOFF_SECTIONS:
        assert f"## {head}" in res["summary"]
    # A huge history still yields the four sections under the char cap.
    big = [{"role": "user", "content": f"task {i} " + "x" * 800}
           for i in range(40)]
    big += [{"role": "assistant",
             "content": "We decided to keep it simple. " * 60}]
    res = memory.consolidate_session(proj, Config(), big)
    assert len(res["summary"]) <= memory.MAX_HANDOFF_CHARS
    assert res["after_tokens"] < 400
    for head in memory.HANDOFF_SECTIONS:
        assert f"## {head}" in res["summary"]


def test_consolidate_model_rewrite_and_fallback(proj, home):
    _write_transcripts(proj)
    refined = ("## Goal & Context\n- keep the flat layout\n"
               "## Files Modified & Key Decisions\n- app.py\n"
               "## Verified Test Status\n- 12 passed\n"
               "## Open Next Steps\n- wire the GUI button\n")
    res = memory.consolidate_session(proj, Config(),
                                     client=FakeDraftClient(refined))
    assert res["summary"] == refined.strip()
    # A broken or off-format model reply keeps the deterministic draft.
    for reply in (RuntimeError("model down"), "just prose, no headings", ""):
        res = memory.consolidate_session(proj, Config(),
                                         client=FakeDraftClient(reply))
        assert res["summary"].startswith("# Consolidated session handoff")
        assert "## Goal & Context" in res["summary"]


def test_consolidate_pin_switches_and_projectless(proj, home):
    _write_transcripts(proj)
    res = memory.consolidate_session(proj, Config(), pin_memory=False)
    assert res["pinned"] == 0 and not memory.project_file(proj).exists()
    off = Config()
    off.use_memory = False
    assert memory.consolidate_session(proj, off)["pinned"] == 0
    # No project root: the checkpoint and sessions live in per-user state.
    workspace.write_session(None, "global task", [], "reply", [], None,
                            "20260101-000000")
    res = memory.consolidate_session(None, Config())
    assert res["path"] == str(config_mod.USER_STATE_DIR
                              / "consolidated_session.md")
    assert Path(res["path"]).is_file()
    assert res["pinned"] == 0
    assert "global task" in res["summary"]


# --------------------------------------------------------------------- RPC

def _rpc_server(**kw):
    return rpc.RpcServer(
        run=lambda *a, **k: None, explain=lambda q: "", status=lambda: {},
        switch=lambda ref, kind: ref, input_stream=io.StringIO(""),
        output=rpc.RpcOut(io.StringIO(), lambda: None), **kw)


def test_rpc_consolidate_dispatch_aliases_and_errors():
    payload = {"before_tokens": 900, "after_tokens": 120,
               "summary": "## Goal & Context\n- x", "path": "/tmp/x.md",
               "pinned": 1}
    calls = []

    def consolidate():
        calls.append(1)
        return payload

    s = _rpc_server(consolidate=consolidate)
    for msg in ({"id": 1, "method": "consolidate"},
                {"id": 2, "method": "compact"},
                {"id": 3, "op": "consolidate"},
                {"id": 4, "op": "compact"}):
        frame = s.dispatch(msg)
        assert frame["ok"] is True and frame["result"] == payload
        assert frame["method"] == "consolidate"
    assert len(calls) == 4

    # Busy tasks refuse consolidation; unwired sessions say so.
    s._busy.set()
    frame = s.dispatch({"id": 5, "method": "consolidate"})
    assert not frame["ok"] and "busy: cannot consolidate mid-task" in frame["error"]
    s._busy.clear()
    frame = _rpc_server().dispatch({"id": 6, "method": "consolidate"})
    assert not frame["ok"] and "not wired" in frame["error"]

    def boom():
        raise OSError("disk full")

    frame = _rpc_server(consolidate=boom).dispatch({"id": 7, "op": "compact"})
    assert not frame["ok"] and "disk full" in frame["error"]


# --------------------------------------------------------------- GUI: skills

def test_gui_skills_list_and_match_preview(gui):
    status, j = gui.get("/api/skills")
    assert status == 200 and j["ok"] is True
    names = {r["name"] for r in j["skills"]}
    assert len(names) == 18 and "fastapi-rest-endpoint" in names
    assert all(r["scope"] == "bundled" for r in j["skills"])
    status, j = gui.get("/api/skills?task=fix%20the%20failing%20test")
    assert status == 200
    assert j["matches"][0]["name"] == "fix-a-failing-test"
    assert j["matches"][0]["score"] > 0


def test_gui_skills_save_and_delete_project(gui):
    root = gui.tmp / "proj"
    root.mkdir()
    with core.STATE.lock:
        core.STATE.root = root
    status, j = gui.post("/api/skills", {
        "name": "deploy-notes", "description": "Release checklist",
        "triggers": "deploy, release", "body": "1. Tag the release.",
        "scope": "project"})
    assert status == 200 and j["ok"] is True and j["scope"] == "project"
    assert Path(j["saved"]) == root / ".pedacito" / "skills" / "deploy-notes.md"
    row = {r["name"]: r for r in j["skills"]}["deploy-notes"]
    assert row["scope"] == "project" and row["triggers"] == ["deploy", "release"]

    status, j = gui.post("/api/skills/delete", {"name": "deploy-notes"})
    assert status == 200 and j["deleted"] == "deploy-notes"
    assert not (root / ".pedacito" / "skills" / "deploy-notes.md").exists()
    status, j = gui.post("/api/skills/delete", {"name": "add-a-test"})
    assert status == 400 and "built-in" in j["detail"]
    status, j = gui.post("/api/skills/delete", {"name": "ghost"})
    assert status == 400 and "No project skill" in j["detail"]


def test_gui_skills_scope_guards_and_validation(gui, monkeypatch):
    # No project open: project scope lands in the scratchpad, global works.
    scratchpad = gui.tmp / "state"
    monkeypatch.setattr(config_mod, "USER_STATE_DIR", scratchpad)
    status, j = gui.post("/api/skills", {"name": "x-note", "description": "d",
                                         "body": "1. x", "scope": "project"})
    assert status == 200 and j["scope"] == "project"
    assert Path(j["saved"]) == (scratchpad / "scratchpad" / ".pedacito"
                                / "skills" / "x-note.md")
    status, j = gui.post("/api/skills", {"name": "x-note", "description": "d",
                                         "triggers": ["x"], "body": "1. x",
                                         "scope": "global"})
    assert status == 200 and j["scope"] == "user"
    assert Path(j["saved"]) == gui.cfg_path.parent / "skills" / "x-note.md"
    status, j = gui.post("/api/skills", {"name": "Bad Name", "description": "d",
                                         "body": "1. x", "scope": "user"})
    assert status == 400 and "invalid skill name" in j["detail"]
    status, j = gui.post("/api/skills", {"name": "ok-name", "description": "d",
                                         "body": "1. x", "scope": "bogus"})
    assert status == 400 and "unknown scope" in j["detail"]
    status, j = gui.post("/api/skills/delete", {"name": "x-note",
                                                "scope": "project"})
    assert status == 200 and j["deleted"] == "x-note"


def test_gui_skills_draft_with_active_and_fallback_client(gui, monkeypatch):
    client = FakeDraftClient(json.dumps(_draft_payload()))
    with core.STATE.lock:
        core.STATE.client = client
    status, j = gui.post("/api/skills/draft",
                         {"prompt": "Alpine.js dark-mode components"})
    assert status == 200 and j["ok"] is True
    assert j["draft"]["name"] == "alpine-dark-mode"
    assert client.calls and client.calls[0][1]["stream"] is False
    # Drafting never saves: the library is unchanged until an explicit save.
    assert all(r["scope"] == "bundled" for r in gui.get("/api/skills")[1]["skills"])
    status, j = gui.post("/api/skills/draft", {"prompt": "  "})
    assert status == 400 and j["ok"] is False
    with core.STATE.lock:
        core.STATE.client = FakeDraftClient("not JSON at all")
    status, j = gui.post("/api/skills/draft", {"prompt": "a skill"})
    assert status == 200 and j["ok"] is False and "draft failed" in j["detail"]

    # Project-less fallback: the user config's model drafts instead.
    with core.STATE.lock:
        core.STATE.client = None
    fallback = FakeDraftClient(json.dumps(_draft_payload(name="fallback-skill")))
    monkeypatch.setattr(config_mod, "load", lambda **kw: Config())
    monkeypatch.setattr(llm_mod, "LLMClient", lambda cfg: fallback)
    status, j = gui.post("/api/skills/draft", {"prompt": "a skill"})
    assert status == 200 and j["ok"] is True
    assert j["draft"]["name"] == "fallback-skill" and fallback.calls
    # No usable user config -> a clear 400, never a crash.
    def no_config(**kw):
        raise ConfigError("no model configured")
    monkeypatch.setattr(config_mod, "load", no_config)
    status, j = gui.post("/api/skills/draft", {"prompt": "a skill"})
    assert status == 400 and "No model configured" in j["detail"]


# -------------------------------------------------------- GUI: consolidation

def _gui_chat(root):
    with core.STATE.lock:
        core.STATE.root = root
        core.STATE.cfg = Config()
        core.STATE.client = SimpleNamespace(stats={}, last_prompt_tokens=9999)
    core.STATE.add_message("user", "fix the login timeout in auth.py")
    core.STATE.add_message("assistant",
                           "applied `auth.py`\nWe decided to raise the timeout "
                           "to 30s.\n3 passed (pytest).\n"
                           "Next step: add a regression test.")


def test_gui_consolidate_resets_transcript_and_meter(gui):
    root = gui.tmp / "proj"
    root.mkdir()
    _gui_chat(root)
    status, j = gui.post("/api/session/consolidate")
    assert status == 200 and j["ok"] is True
    for head in memory.HANDOFF_SECTIONS:
        assert f"## {head}" in j["summary"]
    assert j["after_tokens"] < 400
    assert Path(j["path"]) == root / ".pedacito" / "consolidated_session.md"
    assert Path(j["path"]).is_file()
    assert "status" in j and j["status"]["root"] == str(root)
    # The live buffer is now exactly one handoff card...
    snap = core.STATE.messages_snapshot()
    assert len(snap) == 1
    assert snap[0]["role"] == "system" and snap[0]["kind"] == "handoff"
    assert snap[0]["consolidated"] is True
    assert snap[0]["text"] == j["summary"]
    # ...and the Context Meter follows immediately.
    with core.STATE.lock:
        assert core.STATE.client.last_prompt_tokens == j["after_tokens"]
    state = gui.get("/api/state")[1]
    assert len(state["messages"]) == 1
    assert state["messages"][0]["kind"] == "handoff"


def test_gui_consolidate_busy_and_projectless(gui, monkeypatch):
    with core.STATE.lock:
        core.STATE.job = SimpleNamespace(active=True, thread=None,
                                         cancelled=False)
    status, j = gui.post("/api/session/consolidate")
    assert status == 409 and "job is running" in j["detail"]
    with core.STATE.lock:
        core.STATE.job = None
    # Project-less mode: the checkpoint lands in the persistent scratchpad
    # workspace, not a project and not a refusal.
    monkeypatch.setattr(config_mod, "USER_STATE_DIR", gui.tmp / "state")
    core.STATE.add_message("user", "a project-less question")
    core.STATE.add_message("assistant", "an answer")
    status, j = gui.post("/api/session/consolidate")
    assert status == 200 and j["ok"] is True
    scratchpad = gui.tmp / "state" / "scratchpad"
    assert Path(j["path"]) == scratchpad / ".pedacito" / "consolidated_session.md"
    assert Path(j["path"]).is_file()
    assert j["pinned"] == 0
    assert len(core.STATE.messages_snapshot()) == 1
