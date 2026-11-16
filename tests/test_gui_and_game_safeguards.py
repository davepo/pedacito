"""Desktop GUI & game safeguards: Builder-mode detection, the zero-install /
zero-hang skill, and the AST hazard gate.

1. is_creative_or_design_task covers desktop GUIs, interactive utilities and
   games, and the Builder brief carries the target's craft checklist
   (palette/keyboard/grid for GUIs, loop/state-machine/juice for games).
2. The desktop-gui-and-games skill activates on calculator and game prompts
   and keeps the zero-install, non-blocking and geometry contracts.
3. check_python_gui_hazards flags an import-time mainloop(), pack/grid
   collisions and uninstalled third-party GUI imports; check_gui_hazards
   reports only NEW hazards, and verify.check wires the gate in.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_agent_context import Scripted, _agent  # noqa: E402

from pedacito import prompts, skills, verify  # noqa: E402
from pedacito.config import Config  # noqa: E402
from pedacito.edits import Result  # noqa: E402


# --- 1. Builder mode for GUIs and games ---------------------------------------

@pytest.mark.parametrize("task", [
    "Build a scientific calculator with a GUI",
    "Make an Asteroids game with particle effects",
    "build a tkinter notepad",
    "create a PyQt6 settings dialog",
    "make a snake game in pygame",
    "write a tetris clone with a score HUD",
    "build a currency converter desktop app",
    "add a stopwatch widget",
    "make a pong game with screen shake",
    "build a three.js visualizer",
    "write a customtkinter keypad dialog",
])
def test_gui_and_game_tasks_are_creative(task):
    assert prompts.is_creative_or_design_task(task) is True


@pytest.mark.parametrize("task", [
    "build a parser for ini files",
    "fix the off-by-one in bulk_discount",
    "refactor the database layer",
    "update the contribution guide",     # "gui" must not fire inside "guide"
    "run the full gamut of checks",      # "game" must not fire inside "gamut"
])
def test_plain_code_tasks_stay_non_creative(task):
    assert prompts.is_creative_or_design_task(task) is False


def test_builder_brief_carries_the_target_craft():
    gui = prompts.creative_craft_directives("build a calculator gui")
    assert "1995" in gui                 # never the default gray
    assert "<Return>" in gui and "<BackSpace>" in gui and "<Escape>" in gui
    assert "rowconfigure/columnconfigure" in gui
    game = prompts.creative_craft_directives("make an asteroids game")
    assert "requestAnimationFrame" in game
    assert "fixed-timestep" in game and "game-over" in game
    assert "score/HUD" in game
    # Web work needs no extra rules -- the persona covers it.
    assert prompts.creative_craft_directives("build a landing page") == ""


def test_creative_answer_system_adapts_to_gui_and_game():
    cfg = Config()
    gui_sys = prompts.answer_system(cfg, "EDIT FORMAT",
                                    "build a calculator gui", creative=True)
    assert "Senior Product Engineer & Visual Designer" in gui_sys
    assert "rowconfigure/columnconfigure" in gui_sys
    game_sys = prompts.answer_system(cfg, "EDIT FORMAT",
                                     "make a snake game", creative=True)
    assert "requestAnimationFrame" in game_sys
    web_sys = prompts.answer_system(cfg, "EDIT FORMAT",
                                    "build a landing page", creative=True)
    assert "requestAnimationFrame" not in web_sys
    # Without creative mode the craft rules stay out of the brief.
    plain = prompts.answer_system(cfg, "EDIT FORMAT", "build a calculator gui")
    assert "rowconfigure" not in plain


def test_gui_and_game_briefs_skip_the_web_static_site_rules(tmp_path):
    # A tkinter/pygame task is creative now, but the web-specific stylesheet
    # /Tailwind-CDN rule would be nonsense for it -- the craft checklist
    # replaces it. (The deterministic build-file guard still applies.)
    cfg = Config()
    cfg._root = str(tmp_path)
    gui_sys = prompts.answer_system(cfg, "EDIT FORMAT",
                                    "build a calculator gui", creative=True)
    assert "rowconfigure/columnconfigure" in gui_sys
    assert "cdn.tailwindcss.com" not in gui_sys
    game_sys = prompts.answer_system(cfg, "EDIT FORMAT",
                                     "make a snake game", creative=True)
    assert "requestAnimationFrame" in game_sys
    assert "cdn.tailwindcss.com" not in game_sys
    # A plain web design task still gets the static-site rule.
    web_sys = prompts.answer_system(cfg, "EDIT FORMAT",
                                    "make a landing page", creative=True)
    assert "cdn.tailwindcss.com" in web_sys


def test_agent_marks_gui_and_game_tasks_creative(tmp_path):
    from a1_support import make_project
    from pedacito.agent import Agent
    from test_phase_temperature import TempClient
    _root, idx, cfg = make_project(tmp_path)
    agent = Agent(idx, TempClient(cfg), cfg)
    assert agent._creative_for("Build a scientific calculator with a GUI")
    assert agent._creative_for("Make an Asteroids game with particle effects")


# --- 2. the desktop-gui-and-games skill ---------------------------------------

def test_desktop_gui_and_games_skill_contract():
    _desc, body, triggers = skills.BUNDLED["desktop-gui-and-games"]
    assert len(body) <= 580
    assert len(body) <= skills.MAX_BUILTIN_BODY_CHARS
    for token in ("mainloop()", "__main__", "app.press(key)", "game.step(dt)",
                  ".pack()", ".grid()", "TclError", "tkinter", "tk.Canvas",
                  "<canvas>", "PyQt6", "customtkinter", "pygame",
                  "destroy()", "rowconfigure", "columnconfigure"):
        assert token in body, token
    for trigger in ("calculator", "tkinter", "pyqt", "pyside", "gui", "game",
                    "pygame", "arcade", "snake", "pong", "tetris", "breakout",
                    "asteroids", "sprite", "keypad"):
        assert trigger in triggers, trigger


@pytest.mark.parametrize("task", [
    "Build a scientific calculator with a GUI",
    "Make an Asteroids game with particle effects",
    "write a snake game in pygame",
])
def test_skill_activates_on_calculator_and_game_prompts(tmp_path, task):
    picked = skills.select_skills(tmp_path, task, Config())
    assert any(p["name"] == "desktop-gui-and-games" for p in picked), picked
    text, info = skills.load(tmp_path, Config(), "desktop-gui-and-games")
    assert info is not None and info["source"] == "bundled"
    assert text.startswith("SKILL desktop-gui-and-games")


def test_skill_injected_into_the_task_brief(tmp_path):
    block = skills.render_selected(tmp_path, "build a calculator gui",
                                   Config())
    assert "desktop-gui-and-games" in block
    assert "mainloop()" in block and "TclError" in block


# --- 3. the AST hazard gate ----------------------------------------------------

MAINLOOP_BAD = "import tkinter as tk\nroot = tk.Tk()\nroot.mainloop()\n"
MAINLOOP_GUARDED = (
    "import tkinter as tk\n"
    "def main():\n    root = tk.Tk()\n    root.mainloop()\n"
    "if __name__ == '__main__':\n    main()\n")
MAINLOOP_INLINE_GUARD = (
    "import tkinter as tk\nroot = tk.Tk()\n"
    "if __name__ == '__main__':\n    root.mainloop()\n")
MAINLOOP_REVERSED_GUARD = (
    "import tkinter as tk\nroot = tk.Tk()\n"
    "if '__main__' == __name__:\n    root.mainloop()\n")
MAINLOOP_METHOD = (
    "import tkinter as tk\n"
    "class App:\n    def run(self):\n        self.root.mainloop()\n")
MAINLOOP_TRY = (
    "import tkinter as tk\ntry:\n    tk.Tk().mainloop()\n"
    "except Exception:\n    pass\n")
MAINLOOP_DEFERRED = (
    "import tkinter as tk\nroot = tk.Tk()\n"
    "start = lambda: root.mainloop()\n"
    "gen = (root.mainloop() for _ in range(1))\n")

GEOM_MIXED_PARENT = (
    "import tkinter as tk\n"
    "def build(root):\n"
    "    lbl = tk.Label(root)\n    lbl.pack()\n"
    "    btn = tk.Button(root)\n    btn.grid(row=0)\n")
GEOM_SAME_WIDGET = (
    "import tkinter as tk\n"
    "def build(root):\n"
    "    f = tk.Frame(root)\n    f.pack()\n    f.grid(row=0)\n")
GEOM_CLEAN = (
    "import tkinter as tk\n"
    "def build(root):\n"
    "    lbl = tk.Label(root)\n    lbl.grid(row=0)\n"
    "    btn = tk.Button(root)\n    btn.grid(row=1)\n"
    "    root.columnconfigure(0, weight=1)\n")
GEOM_DIFFERENT_PARENTS = (
    "import tkinter as tk\n"
    "def build(root):\n"
    "    top = tk.Frame(root)\n    top.pack()\n"
    "    lbl = tk.Label(top)\n    lbl.grid(row=0)\n")
GEOM_MODULE_LEVEL = (
    "import tkinter as tk\nroot = tk.Tk()\n"
    "a = tk.Label(root)\na.pack()\n"
    "b = tk.Label(root)\nb.grid()\n")


def test_toplevel_mainloop_is_flagged_unless_guarded(tmp_path):
    p = tmp_path / "calc.py"
    hits = verify.check_python_gui_hazards(p, MAINLOOP_BAD)
    assert len(hits) == 1
    assert hits[0].startswith("line 3:")
    assert "mainloop()" in hits[0] and "import time" in hits[0]
    assert "__main__" in hits[0] and "app.press(key)" in hits[0]
    for safe in (MAINLOOP_GUARDED, MAINLOOP_INLINE_GUARD,
                 MAINLOOP_REVERSED_GUARD, MAINLOOP_METHOD,
                 MAINLOOP_DEFERRED):
        assert verify.check_python_gui_hazards(p, safe) == [], safe
    # A top-level try does not make the loop import-safe.
    assert verify.check_python_gui_hazards(p, MAINLOOP_TRY)


def test_pack_grid_collision_is_flagged(tmp_path):
    p = tmp_path / "gui.py"
    hits = verify.check_python_gui_hazards(p, GEOM_MIXED_PARENT)
    assert len(hits) == 1
    assert "mix geometry managers" in hits[0] and "TclError" in hits[0]
    hits = verify.check_python_gui_hazards(p, GEOM_SAME_WIDGET)
    assert len(hits) == 1 and "both .pack()" in hits[0]
    assert verify.check_python_gui_hazards(p, GEOM_CLEAN) == []
    assert verify.check_python_gui_hazards(p, GEOM_DIFFERENT_PARENTS) == []
    assert verify.check_python_gui_hazards(p, GEOM_MODULE_LEVEL)


def test_uninstalled_gui_import_is_flagged_with_escape_hatches(tmp_path,
                                                               monkeypatch):
    p = tmp_path / "game.py"
    src = "import pygame\n"
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    hits = verify.check_python_gui_hazards(p, src)
    assert len(hits) == 1
    assert "pygame" in hits[0] and "tkinter" in hits[0]
    assert hits[0].startswith("line 1:")
    # from-imports count too
    assert verify.check_python_gui_hazards(p, "from pygame import locals\n")
    # Declared in a manifest: the install is intended, stay quiet.
    (tmp_path / "requirements.txt").write_text("pygame>=2.5\n",
                                               encoding="utf-8")
    assert verify.check_python_gui_hazards(p, src) == []
    (tmp_path / "requirements.txt").unlink()
    (tmp_path / "pyproject.toml").write_text(
        '[project]\ndependencies = ["pygame"]\n', encoding="utf-8")
    assert verify.check_python_gui_hazards(p, src) == []
    (tmp_path / "pyproject.toml").unlink()
    # The task asked for the library: stay quiet.
    assert verify.check_python_gui_hazards(
        p, src, task="make a pygame game") == []
    # Installed: stay quiet.
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert verify.check_python_gui_hazards(p, src) == []
    # An uncertain probe never flags.
    def boom(name):
        raise ValueError("no spec")
    monkeypatch.setattr(importlib.util, "find_spec", boom)
    assert verify.check_python_gui_hazards(p, src) == []


def test_broken_buffers_and_stdlib_are_left_alone(tmp_path):
    p = tmp_path / "x.py"
    assert verify.check_python_gui_hazards(p, "def broken(:\n") == []
    assert verify.check_python_gui_hazards(p, "") == []
    assert verify.check_python_gui_hazards(
        p, "import tkinter\nfrom tkinter import ttk\n") == []


def _result(rel, before, after):
    return Result(True, rel, "applied", before=before, after=after)


def test_check_gui_hazards_reports_only_new_hazards(tmp_path):
    bad = MAINLOOP_BAD
    r = _result("calc.py", "", bad)
    found = verify.check_gui_hazards([r], root=tmp_path)
    assert len(found) == 1
    f = found[0]
    assert f.kind == "gui_hazard" and f.severity == "error"
    assert f.file == "calc.py" and f.line == 3
    # The same hazard already in the before buffer: quiet, even when the
    # edit shifts its line number.
    assert verify.check_gui_hazards(
        [_result("calc.py", bad, bad + "# comment\n")], root=tmp_path) == []
    assert verify.check_gui_hazards(
        [_result("calc.py", bad, "# header\n" + bad)], root=tmp_path) == []
    # Fixing the hazard: quiet.
    fixed = bad.replace("root.mainloop()",
                        "if __name__ == '__main__':\n    root.mainloop()")
    assert verify.check_gui_hazards(
        [_result("calc.py", bad, fixed)], root=tmp_path) == []
    # Non-Python files are not this gate's business.
    assert verify.check_gui_hazards(
        [_result("page.html", "", bad)], root=tmp_path) == []


def test_uninstalled_import_is_a_warning_finding(tmp_path, monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    found = verify.check_gui_hazards(
        [_result("game.py", "", "import pygame\n")], root=tmp_path)
    assert len(found) == 1
    assert found[0].severity == "warn" and found[0].kind == "gui_hazard"


def test_verify_check_wires_the_gui_hazard_gate(tmp_path):
    agent = _agent(tmp_path, Scripted(), {"main.py": "print('hi')\n"})
    reply = ("--- FILE: calc.py\n<<<<<<< SEARCH\n\n=======\n"
             + MAINLOOP_BAD + ">>>>>>> REPLACE\n")
    results = agent._apply_once(reply, dry_run=True)
    found = verify.check(agent, "build a calculator", reply, results)
    hits = [f for f in found if f.kind == "gui_hazard"]
    assert len(hits) == 1 and hits[0].severity == "error"
    assert "mainloop()" in hits[0].message
    # The static gate's off switch covers this gate too.
    agent.cfg.static_gate = False
    found = verify.check(agent, "build a calculator", reply, results)
    assert not [f for f in found if f.kind == "gui_hazard"]
