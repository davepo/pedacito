"""The GUI without an open project: global profiles, profile switching,
starting a New project, and the standalone Ask/scratchpad path.

The ``gui`` fixture below is test_gui's local server on a free port with an
isolated config directory. Model calls are replaced with tiny fakes, so these
tests are deterministic and never touch a real model server.
"""
# `gui` is test_gui's server fixture, handed to each test as a parameter.
# ruff: noqa: F811
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pedacito import config as config_mod
from pedacito import llm as llm_mod
from pedacito.gui import jobs
from test_gui import gui  # noqa: F401  (the shared server fixture)

CONFIG = '''\
default_profile = "fast"

[universal]
vram_gb = 8
base_url = "http://127.0.0.1:1/v1"
model = "default-model"
auto_route_specialties = true
disable_thinking = true

[profiles.fast]
model = "fast-model"
tier = "small"

[profiles.smart]
model = "smart-model"
tier = "large"
specialties = ["coding", "testing"]
'''


def _write(client, text=CONFIG):
    client.cfg_path.parent.mkdir(parents=True, exist_ok=True)
    client.cfg_path.write_text(text, encoding="utf-8")


class FakeClient:
    """The parts of LLMClient the project/Ask paths touch, without a server."""

    created: list = []

    def __init__(self, cfg):
        self.cfg = cfg
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "cost": 0.0, "seconds": 0.0}
        self.stream_callback = None
        self.echo_stream = False
        self.site = ""
        FakeClient.created.append(self)

    def check(self, warn=True):
        return self.cfg.model

    def warmup(self):
        return None

    def cancel(self):
        return None

    def chat(self, messages, max_tokens, **kw):
        self.stats["calls"] += 1
        self.stats["prompt_tokens"] += 10
        self.stats["completion_tokens"] += 5
        self.last_messages = messages
        return "Scratchpad answer."


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Keep every test off a real model server, and skip auto-fit probing."""
    FakeClient.created = []
    monkeypatch.setattr(jobs, "LLMClient", FakeClient)

    def config_for(args):
        return config_mod.load(
            root=getattr(args, "root", None) or None,
            profile=getattr(args, "profile", None),
            require_hardware=False, warn_budgets=False,
            upgrade_user_config=False)

    monkeypatch.setattr(jobs.cli_mod, "_config", config_for)


# ------------------------------------------------------------ global profiles

def test_state_without_a_project_lists_global_profiles_and_settings(gui):
    _write(gui)
    status, st = gui.get("/api/state")
    assert status == 200
    assert st["root"] == "" and st["scratchpad"] is True
    assert st["workspace_label"] == "Workspace: Scratchpad (no project open)"
    assert st["profiles"] == ["fast", "smart"]
    assert st["profile"] == "fast" and st["model"] == "fast-model"
    assert st["auto_route_specialties"] is True
    assert st["disable_thinking"] is True

    details = st["profile_catalog"]
    assert set(details) == {"fast", "smart"}
    assert details["fast"]["active"] is True
    assert details["smart"]["model"] == "smart-model"
    assert details["smart"]["tier"] == "large"
    assert details["smart"]["specialties"] == ["coding", "testing"]
    # Badge fields are present even when no card/file has been measured.
    for key in ("calibrated", "calibrated_at", "card_score", "is_hosted"):
        assert key in details["smart"]
    assert details["smart"]["calibrated"] is False
    assert details["smart"]["is_hosted"] is False
    assert [p["name"] for p in st["profile_details"]] == ["fast", "smart"]

    # The Settings dialog reads the same global config, not bare defaults.
    status, settings = gui.get("/api/settings")
    assert status == 200
    assert settings["active_profile"] == "fast"
    assert {"fast", "smart"} <= set(settings["user"]["profiles"])


def test_profile_switch_and_toggles_work_without_a_project(gui):
    _write(gui)
    status, j = gui.post("/api/profile", {"profile": "smart"})
    assert status == 200 and j["ok"] is True
    assert j["profile"] == "smart" and j["model"] == "smart-model"

    st = gui.get("/api/state")[1]
    assert st["profile"] == "smart" and st["model"] == "smart-model"

    # A typo is still refused rather than silently ignored.
    status, j = gui.post("/api/profile", {"profile": "ghost"})
    assert status == 400 and "no profile named" in j["detail"]

    # The two topbar toggles persist with no project open too.
    status, j = gui.post("/api/profile", {
        "profile": "smart", "auto_route_specialties": False,
        "disable_thinking": False})
    assert status == 200 and j["ok"] is True
    assert j["auto_route_specialties"] is False
    assert j["disable_thinking"] is False
    st = gui.get("/api/state")[1]
    assert st["auto_route_specialties"] is False
    assert st["disable_thinking"] is False
    assert "auto_route_specialties = false" in gui.cfg_path.read_text(
        encoding="utf-8")


# -------------------------------------------------------------- new project

def test_new_project_from_no_project_uses_the_selected_profile_and_binds(
        gui, monkeypatch):
    _write(gui)
    target = gui.tmp / "brand-new-app"
    calls = []

    def fake_run_new(root, cfg, client, spec, **kw):
        calls.append({"root": Path(root), "model": cfg.model,
                      "profile": cfg.active_profile, "apply": kw.get("apply")})
        (Path(root) / "app.py").write_text("print('hello')\n",
                                           encoding="utf-8")
        return 0

    monkeypatch.setattr(jobs.scaffold, "run_new", fake_run_new)

    # Select the non-default profile first, exactly as the topbar does.
    assert gui.post("/api/profile", {"profile": "smart"})[0] == 200
    status, _ = gui.post("/api/run", {
        "mode": "new", "text": "a tiny hello-world app",
        "project": str(target), "build": "build", "profile": "smart"})
    assert status == 202 and gui.wait_idle(30)

    assert target.is_dir()
    assert calls and calls[0]["model"] == "smart-model"
    assert calls[0]["profile"] == "smart" and calls[0]["apply"] is True
    assert (target / "app.py").is_file()

    st = gui.get("/api/state")[1]
    assert st["root"] == str(target.resolve())
    assert st["profile"] == "smart" and st["model"] == "smart-model"
    assert st["scratchpad"] is False


# --------------------------------------------------------------- ask / chat

def test_ask_without_a_project_answers_on_the_selected_profile(gui):
    _write(gui)
    assert gui.post("/api/profile", {"profile": "smart"})[0] == 200

    status, _ = gui.post("/api/run", {
        "mode": "explain", "text": "what is a context window?"})
    assert status == 202 and gui.wait_idle(30)

    answers = [m for m in gui.get("/api/state")[1]["messages"]
               if m.get("kind") == "answer"]
    assert answers and answers[-1]["text"] == "Scratchpad answer."
    st = gui.get("/api/state")[1]
    assert st["root"] == "" and st["scratchpad"] is True
    # The answer ran on the profile selected in the topbar, not the default.
    used = [c for c in FakeClient.created if hasattr(c, "last_messages")]
    assert used and used[-1].cfg.model == "smart-model"


# ------------------------------------------------- prompt refinement without one

class FakeSpecClient:
    """The one schema-constrained architect call, without a model server."""

    created: list = []

    def __init__(self, cfg):
        self.cfg = cfg
        self.messages = None
        FakeSpecClient.created.append(self)

    def chat_json(self, messages, schema, max_tokens=None,
                  disable_thinking=False):
        self.messages = messages
        return {
            "objective": "Build a CLI calculator with +, -, *, /.",
            "files": [{"path": "calc.py", "action": "create",
                       "responsibility": "argument parsing and arithmetic"}],
            "contracts": ["def calculate(expression: str) -> float"],
            "requirements": ["1. `calc.py \"2 + 3\"` prints 5.",
                             "2. A malformed expression exits non-zero."],
            "edge_cases": ["Division by zero reports an error."],
            "verification": "Run `python calc.py \"2 + 3\"` and expect 5.",
        }


def test_refine_without_a_project_uses_the_selected_profile(gui, monkeypatch):
    _write(gui)
    state = gui.tmp / "state"
    monkeypatch.setattr(config_mod, "USER_STATE_DIR", state)
    FakeSpecClient.created = []
    monkeypatch.setattr(llm_mod, "LLMClient", FakeSpecClient)

    assert gui.post("/api/profile", {"profile": "smart"})[0] == 200
    # Both spellings are the same synchronous architect route.
    status, j = gui.post("/api/refine", {
        "prompt": "build a cli calculator", "mode": "task"})
    assert status == 200 and j["ok"] is True, j
    assert j["original"] == "build a cli calculator"
    assert "OBJECTIVE" in j["refined"]
    assert "def calculate(expression: str) -> float" in j["refined"]
    assert isinstance(j["contracts"], list)

    used = FakeSpecClient.created[-1]
    assert used.cfg.model == "smart-model"          # topbar selection wins
    prompt = used.messages[-1]["content"]
    # No repo facts are invented for a project that does not exist.
    assert "TEST RUNNER:" not in prompt
    assert "TOP-LEVEL ENTRIES:" not in prompt
    assert "(empty project)" in prompt

    # The long spelling still answers identically.
    status, j = gui.post("/api/spec/refine", {"prompt": "a tiny web page"})
    assert status == 200 and j["ok"] is True


# ------------------------------------------- skills, MCP and workspace routes

class FakeDraftClient:
    created: list = []

    def __init__(self, cfg):
        self.cfg = cfg
        FakeDraftClient.created.append(self)

    def chat(self, messages, max_tokens=None, **kw):
        return json.dumps({
            "name": "cli-calculator",
            "description": "Build and test a small CLI calculator",
            "triggers": ["calculator", "cli", "arithmetic"],
            "body": "1. Parse argv.\n2. Evaluate safely.\n3. Test errors.",
        })


def test_skills_list_draft_and_save_without_a_project(gui, monkeypatch):
    _write(gui)
    state = gui.tmp / "state"
    monkeypatch.setattr(config_mod, "USER_STATE_DIR", state)
    FakeDraftClient.created = []
    monkeypatch.setattr(llm_mod, "LLMClient", FakeDraftClient)
    assert gui.post("/api/profile", {"profile": "smart"})[0] == 200

    status, j = gui.get("/api/skills")
    assert status == 200 and j["ok"] is True
    assert len(j["skills"]) == 19                 # every built-in is present
    assert all(r["scope"] == "bundled" for r in j["skills"])

    status, j = gui.post("/api/skills/draft",
                         {"prompt": "a CLI calculator checklist"})
    assert status == 200 and j["ok"] is True, j
    assert j["draft"]["name"] == "cli-calculator"
    assert FakeDraftClient.created[-1].cfg.model == "smart-model"

    # No scope means the global library when no project is open.
    status, j = gui.post("/api/skills", {
        "name": "deploy-notes", "description": "Release checklist",
        "triggers": "deploy, release", "body": "1. Tag the release."})
    assert status == 200 and j["scope"] == "user"
    assert Path(j["saved"]) == gui.cfg_path.parent / "skills" / "deploy-notes.md"

    # An explicit project scope saves into the scratchpad project.
    status, j = gui.post("/api/skills", {
        "name": "scratch-note", "description": "Scratchpad checklist",
        "body": "1. Do the thing.", "scope": "project"})
    assert status == 200 and j["scope"] == "project"
    expected = state / "scratchpad" / ".pedacito" / "skills" / "scratch-note.md"
    assert Path(j["saved"]) == expected and expected.is_file()
    names = {r["name"] for r in gui.get("/api/skills")[1]["skills"]}
    assert {"deploy-notes", "scratch-note"} <= names

    status, j = gui.post("/api/skills/delete",
                         {"name": "scratch-note", "scope": "project"})
    assert status == 200 and j["deleted"] == "scratch-note"
    assert not expected.exists()


def test_mcp_servers_save_globally_without_a_project(gui):
    _write(gui)
    status, j = gui.get("/api/mcp/servers")
    assert status == 200 and j["ok"] is True
    assert j["user_config"] == str(gui.cfg_path)

    status, j = gui.post("/api/mcp/servers", {
        "name": "docs", "transport": "http",
        "url": "http://127.0.0.1:9/mcp", "enabled": False})
    assert status == 200 and j["ok"] is True, j
    assert j["path"] == str(gui.cfg_path)
    assert "[mcp_servers.docs]" in gui.cfg_path.read_text(encoding="utf-8")

    # Asking for the project file with no project open falls back to global
    # instead of refusing the Save button.
    status, j = gui.post("/api/mcp/servers", {
        "name": "local-notes", "transport": "http",
        "url": "http://127.0.0.1:9/notes", "enabled": False,
        "file": "project"})
    assert status == 200 and j["ok"] is True, j
    assert j["path"] == str(gui.cfg_path)
    assert "[mcp_servers.local-notes]" in gui.cfg_path.read_text(
        encoding="utf-8")


def test_session_memory_reset_and_procs_fall_back_to_the_scratchpad(
        gui, monkeypatch):
    _write(gui)
    state = gui.tmp / "state"
    monkeypatch.setattr(config_mod, "USER_STATE_DIR", state)
    scratchpad = state / "scratchpad"

    gui.post("/api/clear", {})
    gui.post("/api/run", {"mode": "explain", "text": "what is a token?"})
    assert gui.wait_idle(30)

    status, j = gui.post("/api/session/consolidate", {})
    assert status == 200 and j["ok"] is True, j
    handoff = scratchpad / ".pedacito" / "consolidated_session.md"
    assert Path(j["path"]) == handoff and handoff.is_file()

    status, j = gui.get("/api/memory")
    assert status == 200 and j["ok"] is True
    status, j = gui.post("/api/memory", {
        "action": "add", "text": "scratchpad facts survive a reload",
        "scope": "project"})
    assert status == 200 and j["stored"] is True, j
    assert (scratchpad / ".pedacito" / "memory.md").is_file()
    assert any("scratchpad facts" in e["text"]
               for e in gui.get("/api/memory")[1]["entries"])

    status, j = gui.get("/api/procs")
    assert status == 200 and j["ok"] is True and j["processes"] == []
    status, j = gui.post("/api/procs", {"action": "list"})
    assert status == 200 and j["ok"] is True

    status, j = gui.post("/api/project/prune", {"dry_run": True})
    assert status == 200 and j["ok"] is True, j

    status, j = gui.post("/api/project/reset", {
        "keep_memory": True, "keep_config": True, "dry_run": True})
    assert status == 200 and j["ok"] is True and j["dry_run"] is True, j
    status, j = gui.post("/api/project/reset", {
        "keep_memory": True, "keep_config": True})
    assert status == 200 and j["ok"] is True, j
    assert (scratchpad / ".pedacito" / "memory.md").is_file()
    # The undo spelling reaches the same workspace route and is not refused
    # for lacking a project.
    status, j = gui.post("/api/undo", {})
    assert status == 400 and "No snapshots" in j["detail"]


def test_refine_and_workspace_controls_are_not_project_gated():
    """Static guard: the controls unlocked above must not regain a root gate."""
    src = (Path(__file__).resolve().parents[1] / "pedacito" / "gui"
           / "static" / "app.js").read_text(encoding="utf-8")
    for anchor in ("function memoryDialog()", "function procsDialog()",
                   "function pruneDialog()", "function resetDialog()"):
        start = src.index(anchor)
        assert "_needRoot()" not in src[start:start + 160]
    start = src.index('<button id="refineBtn"')
    assert "!s.root" not in src[start:start + 600]
    start = src.index("Save to Project")
    assert "disabled" not in src[start:start + 260]
