"""Desktop GUI & game safeguards: Builder-mode detection, the zero-install /
zero-hang skill, and the AST hazard gate.

1. is_creative_or_design_task covers desktop GUIs, interactive utilities and
   games, and the Builder brief carries the target's craft checklist
   (palette/keyboard/grid for GUIs, loop/state-machine/juice for games).
2. The desktop-gui-and-games skill activates on calculator and game prompts
   and keeps the zero-install, non-blocking and geometry contracts.
3. check_python_gui_hazards flags an import-time mainloop(), pack/grid
   collisions, unguarded tk.StringVar() when the module sets tk = None on
   ImportError, and uninstalled third-party GUI imports; check_gui_hazards
   reports only NEW hazards, and verify.check wires the gate in.
4. run_new executes the detected suite after scaffolding and feeds a failing
   test's traceback into repair rounds until green (live CachyOS sci-calc:
   PEP 668 kept pytest out of reach, so the suite was silently skipped).
5. Test-file build prompts carry the EXACT implementation interfaces on disk
   (module-level defs, class methods, constants/mappings), so a test never
   guesses ClassName.main for a module-level def or confuses log with ln.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_agent_context import Scripted, _agent  # noqa: E402

from pedacito import prompts, scaffold, skills, testsel, verify  # noqa: E402
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
    # This skill alone runs a 580-char budget: the handler-test and safe-ast
    # contracts joined the headless-fallback ones (the global 550 draft trim
    # in skills.MAX_BUILTIN_BODY_CHARS stays for model-drafted skills).
    assert len(body) <= 580
    for token in ("mainloop()", "__main__", "app.press(key)", "game.step(dt)",
                  ".pack()", ".grid()", "TclError", "tkinter", "tk.Canvas",
                  "<canvas>", "PyQt6", "customtkinter", "pygame",
                  "destroy()", "rowconfigure", "columnconfigure",
                  # The headless-fallback contract (live CachyOS run):
                  "tk = None", "ImportError", "tk.StringVar()",
                  "if tk is not None:", ".get()/.set()", "log/ln", "main",
                  # End-to-end handler tests + safe expression parsing:
                  "on_button_press", "on_equals", "sin", "sqrt",
                  'ast.parse(expr, mode="eval")', "eval()", "set(...)"):
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


# --- 4. tk = None headless fallback: unguarded tk.StringVar() in __init__ ------
# On a box without libtk, `import tkinter` raises ImportError and the module's
# fallback sets tk = None. An unconditional tk.StringVar() in __init__ then
# crashes headless verification with AttributeError (live CachyOS sci-calc).

TK_NONE_UNGUARDED = (
    "try:\n    import tkinter as tk\nexcept ImportError:\n    tk = None\n\n"
    "class App:\n"
    "    def __init__(self, root):\n"
    "        self.root = root\n"
    "        self.var = tk.StringVar(value='0')\n")
TK_NONE_GUARDED = TK_NONE_UNGUARDED.replace(
    "        self.var = tk.StringVar(value='0')",
    "        if tk is not None:\n"
    "            self.var = tk.StringVar(value='0')\n"
    "        else:\n"
    "            self.var = None")
TK_NONE_GUARDED_ELSE = TK_NONE_UNGUARDED.replace(
    "        self.var = tk.StringVar(value='0')",
    "        if tk is None:\n"
    "            self.var = None\n"
    "        else:\n"
    "            self.var = tk.StringVar(value='0')")
TK_NONE_GUARDED_TRUTHY = TK_NONE_UNGUARDED.replace(
    "        self.var = tk.StringVar(value='0')",
    "        if tk:\n"
    "            self.var = tk.StringVar(value='0')")
TK_NONE_GUARDED_REVERSED = TK_NONE_UNGUARDED.replace(
    "        self.var = tk.StringVar(value='0')",
    "        if None is not tk:\n"
    "            self.var = tk.StringVar(value='0')")
TK_NONE_GUARDED_TERNARY = TK_NONE_UNGUARDED.replace(
    "        self.var = tk.StringVar(value='0')",
    "        self.var = tk.StringVar(value='0') if tk is not None else None")
TK_NONE_GUARDED_TERNARY_REV = TK_NONE_UNGUARDED.replace(
    "        self.var = tk.StringVar(value='0')",
    "        self.var = None if tk is None else tk.StringVar(value='0')")


def test_tk_none_stringvar_in_init_is_flagged(tmp_path):
    p = tmp_path / "calc.py"
    hits = verify.check_python_gui_hazards(p, TK_NONE_UNGUARDED)
    assert len(hits) == 1
    assert hits[0].startswith("line 9:")
    for token in ("StringVar", "if tk is not None:", "ImportError",
                  "AttributeError", "__init__"):
        assert token in hits[0], token


@pytest.mark.parametrize("attr", ["StringVar", "IntVar", "BooleanVar",
                                  "DoubleVar"])
def test_all_tk_var_types_are_flagged(tmp_path, attr):
    p = tmp_path / "calc.py"
    src = TK_NONE_UNGUARDED.replace("tk.StringVar(", f"tk.{attr}(")
    hits = verify.check_python_gui_hazards(p, src)
    assert len(hits) == 1 and attr in hits[0]


def test_guarded_tk_none_calls_are_clean(tmp_path):
    p = tmp_path / "calc.py"
    for safe in (TK_NONE_GUARDED, TK_NONE_GUARDED_ELSE,
                 TK_NONE_GUARDED_TRUTHY, TK_NONE_GUARDED_REVERSED,
                 TK_NONE_GUARDED_TERNARY, TK_NONE_GUARDED_TERNARY_REV):
        assert verify.check_python_gui_hazards(p, safe) == [], safe


def test_tk_none_hazard_needs_the_fallback_and_init_scope(tmp_path):
    p = tmp_path / "calc.py"
    # No tk = None fallback: an unconditional StringVar is not this hazard.
    plain = ("import tkinter as tk\n"
             "class App:\n"
             "    def __init__(self, root):\n"
             "        self.var = tk.StringVar()\n")
    assert verify.check_python_gui_hazards(p, plain) == []
    # The fallback exists, but the call lives outside __init__: out of scope.
    other_method = TK_NONE_UNGUARDED.replace(
        "    def __init__(self, root):", "    def build(self, root):")
    assert verify.check_python_gui_hazards(p, other_method) == []


def test_tk_none_hazard_recognises_import_error_handler_variants(tmp_path):
    p = tmp_path / "calc.py"
    for handler in ("except ModuleNotFoundError:",
                    "except (ImportError, ModuleNotFoundError):"):
        src = TK_NONE_UNGUARDED.replace("except ImportError:", handler)
        assert verify.check_python_gui_hazards(p, src), handler


def test_tk_none_hazard_is_an_error_finding_only_when_new(tmp_path):
    found = verify.check_gui_hazards(
        [_result("calc.py", "", TK_NONE_UNGUARDED)], root=tmp_path)
    assert len(found) == 1
    f = found[0]
    assert f.kind == "gui_hazard" and f.severity == "error"
    assert f.file == "calc.py" and f.line == 9
    # Already present before the edit: quiet. Guarded before, broken after:
    # a new hazard, reported.
    assert verify.check_gui_hazards(
        [_result("calc.py", TK_NONE_UNGUARDED, TK_NONE_UNGUARDED)],
        root=tmp_path) == []
    assert len(verify.check_gui_hazards(
        [_result("calc.py", TK_NONE_GUARDED, TK_NONE_UNGUARDED)],
        root=tmp_path)) == 1


# --- 5. run_new runs the scaffolded suite and repairs it to green --------------
# The live sci-calc run: a 2-file project whose unittest suite failed (ln was
# log10), while detection named a pytest the fresh folder could not launch --
# so verification was skipped as an "environment issue" and new exited 0.

class _CalcBuild:
    """Plans calculator.py + test_calculator.py, builds a module whose ln()
    is wrongly log10, and repairs it when the verify traceback arrives."""
    stats = {"calls": 0, "completion_tokens": 0, "seconds": 0}

    MODULE = (
        "import math\n\n\n"
        "class ScientificCalculator:\n"
        "    def ln(self, x):\n"
        "        return math.log10(x)\n\n"
        "    @staticmethod\n"
        "    def main():\n"
        "        return 0\n")
    TESTS = (
        "import math\nimport unittest\n\n"
        "from calculator import ScientificCalculator\n\n\n"
        "class TestCalc(unittest.TestCase):\n"
        "    def test_ln_e(self):\n"
        "        self.assertAlmostEqual("
        "ScientificCalculator().ln(math.e), 1.0)\n")
    FIX = ("--- FILE: calculator.py\n<<<<<<< SEARCH\n"
           "        return math.log10(x)\n=======\n"
           "        return math.log(x)\n>>>>>>> REPLACE\n")

    def __init__(self, repair_reply=None):
        self.built: list[str] = []
        self.repair_prompts: list[str] = []
        self.repair_reply = self.FIX if repair_reply is None else repair_reply

    def chat_json(self, messages, schema, max_tokens=0, **kw):
        return {"project_name": "sci-calc", "summary": "a calculator",
                "files": [{"path": "calculator.py", "purpose": "core"},
                          {"path": "test_calculator.py", "purpose": "tests"}]}

    def chat(self, messages, max_tokens=0, **kw):
        user = messages[-1]["content"]
        if "make this check pass" in user:
            self.repair_prompts.append(user)
            return self.repair_reply
        for rel, text in (("calculator.py", self.MODULE),
                          ("test_calculator.py", self.TESTS)):
            if rel in user and rel not in self.built:
                self.built.append(rel)
                return (f"--- FILE: {rel}\n<<<<<<< SEARCH\n\n=======\n"
                        f"{text}>>>>>>> REPLACE\n")
        return ""


def _pep668(monkeypatch):
    """pytest is neither on PATH nor importable: the fresh-folder scenario."""
    real_which = testsel.shutil.which
    monkeypatch.setattr(
        testsel.shutil, "which",
        lambda name: None if name == "pytest" else real_which(name))
    real_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a, **k: None if name == "pytest"
        else real_spec(name, *a, **k))


def _new_cfg() -> Config:
    cfg = Config()
    cfg.log_sessions = False
    cfg.log_events = False
    cfg.backup = False
    return cfg


def test_run_new_repairs_a_failing_unit_test_to_green(tmp_path, monkeypatch,
                                                      capsys):
    _pep668(monkeypatch)
    client = _CalcBuild()
    code = scaffold.run_new(tmp_path, _new_cfg(), client,
                            scaffold.Spec("a scientific calculator"),
                            apply=True, verbose=False)
    err = capsys.readouterr().err
    assert code == 0, err
    # The failing suite reached repair, traceback in hand -- not skipped.
    assert client.repair_prompts, err
    assert "AssertionError" in client.repair_prompts[0]
    assert "unittest" in client.repair_prompts[0]
    # The repair prompt carries the implementation source, so the fix reads
    # the real ln/log bodies instead of guessing at them.
    assert "class ScientificCalculator" in client.repair_prompts[0]
    assert "math.log10(x)" in client.repair_prompts[0]
    assert "tests: passed" in err and "unittest" in err
    fixed = (tmp_path / "calculator.py").read_text(encoding="utf-8")
    assert "math.log(x)" in fixed and "log10" not in fixed


def test_run_new_unrepaired_suite_reaches_the_exit_code(tmp_path, monkeypatch,
                                                        capsys):
    _pep668(monkeypatch)
    client = _CalcBuild(repair_reply="I cannot fix this.")
    code = scaffold.run_new(tmp_path, _new_cfg(), client,
                            scaffold.Spec("a scientific calculator"),
                            apply=True, verbose=False)
    err = capsys.readouterr().err
    assert code == 1, err
    assert client.repair_prompts, "the failure was skipped, not repaired"
    assert "BUILD INCOMPLETE" in err


# --- 6. test-file prompts carry the exact implementation interfaces -----------
# The live sci-calc run: the test pass guessed ScientificCalculator.main when
# main() was module-level, and ln vs log semantics drifted, because the compact
# contract map truncates entries and omits methods/constants entirely.

CALC_IMPL = (
    "import math\n\n"
    "OPS = {'log': math.log10, 'ln': math.log}\n\n\n"
    "def main():\n"
    "    return 0\n\n\n"
    "def log(x):\n"
    "    return math.log10(x)\n\n\n"
    "def ln(x):\n"
    "    return math.log(x)\n\n\n"
    "class ScientificCalculator:\n"
    "    def press(self, key):\n"
    "        self.last = key\n\n"
    "    def equals(self):\n"
    "        return 0\n")


class _PromptRecorder:
    """Records every build prompt; replies with no edits (the test inspects
    the prompt, not the build result)."""
    stats = {"calls": 0, "completion_tokens": 0, "seconds": 0}

    def __init__(self):
        self.prompts: list[str] = []

    def chat(self, messages, max_tokens=0, **kw):
        self.prompts.append(messages[-1]["content"])
        return ""


def _calc_plan(tmp_path):
    from pedacito import index as index_mod
    (tmp_path / "calculator.py").write_text(CALC_IMPL, encoding="utf-8")
    (tmp_path / "test_calculator.py").write_text("# tests stub\n",
                                                 encoding="utf-8")
    plan = scaffold.Plan(
        project_name="sci-calc", summary="a calculator",
        files=[scaffold.PlannedFile("calculator.py", "core"),
               scaffold.PlannedFile("test_calculator.py", "tests")],
        spec=scaffold.Spec("a scientific calculator"))
    cfg = _new_cfg()
    idx = index_mod.build([str(tmp_path)], cfg, client=None, root=tmp_path,
                          summarise=False, verbose=False)
    return plan, cfg, idx


def test_build_pass_for_a_test_file_injects_exact_interfaces(tmp_path):
    plan, cfg, idx = _calc_plan(tmp_path)
    client = _PromptRecorder()
    scaffold.build_pass(client, cfg, idx, plan, plan.tests[0],
                        ["calculator.py"], verbose=False)
    assert client.prompts
    prompt = client.prompts[0]
    assert "exact symbols the implementation files expose on disk" in prompt
    # module-level defs with their real signatures -- main() is NOT a method
    assert "def main()" in prompt
    assert "def log(x)" in prompt and "def ln(x)" in prompt
    # class names AND method names, not just __init__
    assert "class ScientificCalculator" in prompt
    assert "press(key)" in prompt and "equals()" in prompt
    # key constants/mappings survive, so log vs ln is read, not guessed
    assert "OPS = " in prompt and "'ln': math.log" in prompt
    # the explicit anti-guessing instruction rides along
    assert "never ClassName.main" in prompt
    assert "log vs ln" in prompt


def test_build_pass_for_a_source_file_keeps_the_contract_map(tmp_path):
    plan, cfg, idx = _calc_plan(tmp_path)
    client = _PromptRecorder()
    scaffold.build_pass(client, cfg, idx, plan, plan.sources[0], [],
                        verbose=False)
    prompt = client.prompts[0]
    assert "signatures every file imports by" in prompt
    assert "exact symbols the implementation files expose" not in prompt


def test_test_interface_map_prefers_the_linked_module(tmp_path):
    from pedacito import index as index_mod
    plan, cfg, idx = _calc_plan(tmp_path)
    (tmp_path / "zz_other.py").write_text("def unrelated():\n    pass\n",
                                          encoding="utf-8")
    plan.files.append(scaffold.PlannedFile("zz_other.py", "misc"))
    idx = index_mod.reindex_file(idx, "zz_other.py", cfg)
    # A tight budget keeps the module the test is named after, not the tail.
    text = scaffold._test_interface_map(idx, plan, cfg,
                                        "test_calculator.py", max_chars=160)
    assert "calculator.py:" in text
    assert "def main()" in text
    assert "zz_other.py" not in text


def test_js_test_target_gets_the_exact_interface_section(tmp_path):
    from pedacito import index as index_mod
    (tmp_path / "calc.js").write_text(
        "export function ln(x) { return Math.log(x); }\n"
        "export const OPS = { log: Math.log10, ln: Math.log };\n",
        encoding="utf-8")
    (tmp_path / "calc.test.js").write_text("// tests stub\n", encoding="utf-8")
    plan = scaffold.Plan(
        project_name="calc-js", summary="a calculator",
        files=[scaffold.PlannedFile("calc.js", "core"),
               scaffold.PlannedFile("calc.test.js", "tests")],
        spec=scaffold.Spec("a calculator"))
    cfg = _new_cfg()
    idx = index_mod.build([str(tmp_path)], cfg, client=None, root=tmp_path,
                          summarise=False, verbose=False)
    client = _PromptRecorder()
    scaffold.build_pass(client, cfg, idx, plan, plan.tests[0], ["calc.js"],
                        verbose=False)
    prompt = client.prompts[0]
    assert "exact symbols the implementation files expose on disk" in prompt
    assert "function ln(x)" in prompt
    assert "OPS = " in prompt
