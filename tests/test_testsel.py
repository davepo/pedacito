"""PEP 668 test-runner detection: `pedacito new` on Arch/CachyOS lands in a
fresh folder with no .venv of its own, and pytest lives only inside
Pedacito's venv -- not on PATH. A detected command that cannot run there
("pytest: command not found") is classified as an environment issue and the
suite is silently skipped. These tests pin that detection always names a
runner that can actually execute: the project's own venv pytest, the stdlib
unittest runner for unittest-only suites (both test naming conventions),
a pytest we can really launch, or a clean unittest fallback.
"""
from __future__ import annotations

import importlib.util
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import loop, scaffold, testsel  # noqa: E402
from pedacito.config import Config  # noqa: E402

UNITTEST_TEST = (
    "import unittest\n\n\n"
    "class T(unittest.TestCase):\n"
    "    def test_ok(self):\n"
    "        self.assertTrue(True)\n")
PYTEST_TEST = "def test_ok():\n    assert True\n"

_PY = "python" if sys.platform == "win32" else "python3"
UNITTEST_CMD = f'{_PY} -m unittest discover -p "*test*.py"'


def _w(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _no_pytest(monkeypatch):
    """Simulate the PEP 668 box: pytest is neither on PATH nor importable
    by the running interpreter. Other names keep their real answers."""
    real_which = testsel.shutil.which
    monkeypatch.setattr(
        testsel.shutil, "which",
        lambda name: None if name == "pytest" else real_which(name))
    real_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a, **k: None if name == "pytest"
        else real_spec(name, *a, **k))


# --- unittest-only suites need no pytest at all -------------------------------

def test_unittest_only_suite_uses_the_stdlib_runner(tmp_path, monkeypatch):
    _w(tmp_path, "test_calc.py", UNITTEST_TEST)
    _no_pytest(monkeypatch)
    assert testsel.detect_test_command(tmp_path, Config()) == UNITTEST_CMD


def test_unittest_only_wins_even_when_pytest_is_available(tmp_path):
    # The suite does not need pytest; running it on the stdlib runner keeps
    # the verdict independent of whatever venv happens to be active.
    _w(tmp_path, "test_calc.py", UNITTEST_TEST)
    assert testsel.detect_test_command(tmp_path, Config()) == UNITTEST_CMD


def test_unittest_only_requires_testcase_and_no_pytest_refs(tmp_path,
                                                           monkeypatch):
    # import unittest but only bare functions: pytest-style, not unittest.
    _w(tmp_path, "test_a.py",
       "import unittest\n\ndef test_ok():\n    assert True\n")
    monkeypatch.setattr(testsel.shutil, "which",
                        lambda name: "/usr/bin/pytest" if name == "pytest"
                        else None)
    assert testsel.detect_test_command(tmp_path, Config()) == "pytest -q"
    # A pytest reference anywhere means the suite may need pytest.
    _w(tmp_path, "test_a.py",
       "import unittest\nimport pytest\n\n"
       "class T(unittest.TestCase):\n    def test_ok(self):\n        pass\n")
    assert testsel.detect_test_command(tmp_path, Config()) == "pytest -q"


# --- both naming conventions are discovered ------------------------------------

def test_suffix_convention_files_are_detected(tmp_path, monkeypatch):
    _no_pytest(monkeypatch)
    for rel in ("calc_test.py", "tests/unit_test.py",
                "tests/deep/nested_test.py"):
        _w(tmp_path, rel, PYTEST_TEST)
        assert testsel.detect_test_command(tmp_path, Config()), rel
        # Clean up so each placement is tested on its own.
        (tmp_path / rel).unlink()
    assert testsel.detect_test_command(tmp_path, Config()) == ""


def test_prefix_convention_still_detected(tmp_path, monkeypatch):
    _no_pytest(monkeypatch)
    _w(tmp_path, "tests/deep/test_a.py", PYTEST_TEST)
    assert testsel.detect_test_command(tmp_path, Config())


# --- runner preference order ----------------------------------------------------

def test_project_venv_pytest_is_preferred(tmp_path, monkeypatch):
    _w(tmp_path, "test_a.py", PYTEST_TEST)
    _no_pytest(monkeypatch)
    venv_pytest = tmp_path / ".venv" / "bin" / "pytest"
    _w(tmp_path, ".venv/bin/pytest", "#!/bin/sh\n")
    assert testsel.detect_test_command(tmp_path, Config()) == \
        f"{shlex.quote(str(venv_pytest))} -q"
    # The Windows layout is recognised too.
    venv_pytest.unlink()
    scripts_pytest = tmp_path / ".venv" / "Scripts" / "pytest.exe"
    _w(tmp_path, ".venv/Scripts/pytest.exe", "")
    assert testsel.detect_test_command(tmp_path, Config()) == \
        f"{shlex.quote(str(scripts_pytest))} -q"


def test_path_pytest_used_when_available(tmp_path, monkeypatch):
    _w(tmp_path, "test_a.py", PYTEST_TEST)
    monkeypatch.setattr(testsel.shutil, "which",
                        lambda name: "/usr/bin/pytest" if name == "pytest"
                        else None)
    assert testsel.detect_test_command(tmp_path, Config()) == "pytest -q"


def test_interpreter_pytest_when_no_path_pytest(tmp_path, monkeypatch):
    # BUG-1012 stays pinned: with no PATH pytest, name the interpreter that
    # runs Pedacito (pytest is a core dependency and importable here).
    _w(tmp_path, "test_a.py", PYTEST_TEST)
    monkeypatch.setattr(testsel.shutil, "which", lambda name: None)
    assert testsel.detect_test_command(tmp_path, Config()) == \
        f"{shlex.quote(sys.executable)} -m pytest -q"


def test_clean_unittest_fallback_when_no_pytest_anywhere(tmp_path,
                                                         monkeypatch):
    # pytest-style tests, but nothing can launch pytest: fall back to a
    # runnable stdlib command instead of a dead "pytest -q".
    _w(tmp_path, "test_a.py", PYTEST_TEST)
    _no_pytest(monkeypatch)
    assert testsel.detect_test_command(tmp_path, Config()) == UNITTEST_CMD


def test_explicit_test_command_always_wins(tmp_path):
    _w(tmp_path, "test_a.py", UNITTEST_TEST)
    cfg = Config()
    cfg.test_command = "make check"
    assert testsel.detect_test_command(tmp_path, cfg) == "make check"


# --- the fallback really runs, and failures really surface ----------------------

def test_detected_unittest_command_runs_and_reports(tmp_path, monkeypatch):
    _w(tmp_path, "test_calc.py", UNITTEST_TEST)
    _no_pytest(monkeypatch)
    cmd = testsel.detect_test_command(tmp_path, Config())
    ok, out = loop.run_verify(tmp_path, cmd)
    assert ok, out
    assert "Ran 1 test" in out
    # A failing assertion is a CODE failure, not an environment excuse:
    # verify_and_fix must feed it to repair, not skip it.
    _w(tmp_path, "test_calc.py", UNITTEST_TEST.replace("assertTrue(True)",
                                                       "assertTrue(False)"))
    ok, out = loop.run_verify(tmp_path, cmd)
    assert not ok
    assert "AssertionError" in out
    assert not scaffold._environment_issue(out)
    assert not loop.missing_modules(out, root=tmp_path)


def test_unittest_discovery_covers_both_conventions(tmp_path, monkeypatch):
    _w(tmp_path, "test_a.py", UNITTEST_TEST)
    _w(tmp_path, "b_test.py", UNITTEST_TEST.replace("class T(", "class U("))
    _no_pytest(monkeypatch)
    cmd = testsel.detect_test_command(tmp_path, Config())
    ok, out = loop.run_verify(tmp_path, cmd)
    assert ok, out
    assert "Ran 2 tests" in out


# --- mixed-tree language routing stays consistent -------------------------------

def test_language_base_keeps_the_unittest_runner(tmp_path):
    _w(tmp_path, "test_a.py", UNITTEST_TEST)
    cfg = Config()
    base = testsel.detect_test_command(tmp_path, cfg)
    assert testsel._language_base(tmp_path, ["a.py"], cfg, base) == base


def test_language_base_switches_a_py_change_off_npm(tmp_path, monkeypatch):
    _w(tmp_path, "package.json", '{"scripts": {"test": "jest"}}')
    _w(tmp_path, "test_a.py", UNITTEST_TEST)
    _no_pytest(monkeypatch)
    cfg = Config()
    base = testsel.detect_test_command(tmp_path, cfg)
    assert base == "npm test"
    assert testsel._language_base(tmp_path, ["a.py"], cfg, base) == \
        UNITTEST_CMD


def test_language_base_switches_a_js_change_off_unittest(tmp_path):
    # A JS-only change belongs to the JS runner even when the base command
    # names the Python suite (the BUG _language_base fixes, unittest side).
    _w(tmp_path, "test_a.py", UNITTEST_TEST)
    _w(tmp_path, "app.test.js", "require('node:test').test('x', () => {});\n")
    cfg = Config()
    assert testsel._language_base(tmp_path, ["app.js"], cfg,
                                  UNITTEST_CMD) == "node --test"
