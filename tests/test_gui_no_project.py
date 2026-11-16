"""The GUI without an open project: global profiles, profile switching,
starting a New project, and the standalone Ask/scratchpad path.

The ``gui`` fixture below is test_gui's local server on a free port with an
isolated config directory. Model calls are replaced with tiny fakes, so these
tests are deterministic and never touch a real model server.
"""
# `gui` is test_gui's server fixture, handed to each test as a parameter.
# ruff: noqa: F811
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pedacito import config as config_mod
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
