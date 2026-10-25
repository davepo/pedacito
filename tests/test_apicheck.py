"""Tests for the static cross-file API consistency checker.

The `--complex` build implements each file in a fresh context, so calls can
drift from the definitions they target. These tests pin the checker's exact
behaviour (including what it must NOT flag).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import apicheck


def test_catches_missing_required_argument():
    """The exact bug that shipped: core.py calls Level() while level.py
    defines Level.__init__(self, level_data)."""
    src = {
        "main.py": ("from src.core import Game\n\ndef main():\n"
                    "    game = Game()\n"),
        "src/core.py": ("from src.level import Level\n\nclass Game:\n"
                        "    def __init__(self):\n"
                        "        self.level = Level()\n"),
        "src/level.py": ("class Level:\n"
                         "    def __init__(self, level_data):\n"
                         "        self.data = level_data\n"),
    }
    problems = apicheck.find_mismatches(src)
    assert len(problems) == 1
    p = problems[0]
    assert p.rel == "src/core.py" and p.name == "Level"
    assert "level_data" in p.message
    assert "src/level.py" in p.defined_in


def test_correct_call_is_clean():
    src = {
        "core.py": ("from level import Level\n\nclass Game:\n"
                    "    def __init__(self):\n"
                    "        self.level = Level('main')\n"),
        "level.py": ("class Level:\n"
                     "    def __init__(self, level_data):\n"
                     "        self.data = level_data\n"),
    }
    assert apicheck.find_mismatches(src) == []


def test_default_parameters_are_optional():
    src = {
        "a.py": "def f(x, y=1):\n    return x\n\ndef g():\n    return f(2)\n",
    }
    assert apicheck.find_mismatches(src) == []


def test_too_many_positional_arguments():
    src = {
        "a.py": "def f(x):\n    return x\n\ndef g():\n    return f(1, 2)\n",
    }
    problems = apicheck.find_mismatches(src)
    assert problems and "at most 1" in problems[0].message


def test_class_without_init_takes_no_arguments():
    src = {
        "a.py": "class A:\n    pass\n\ndef g():\n    return A()\n",
    }
    assert apicheck.find_mismatches(src) == []


def test_method_calls_are_ignored():
    """`self.run()` / `obj.run()` must never be matched against a top-level
    `run` function: resolving attribute receivers needs a type checker."""
    src = {
        "a.py": ("def run():\n    return 1\n\nclass C:\n"
                 "    def go(self):\n        return self.run()\n"),
    }
    assert apicheck.find_mismatches(src) == []


def test_star_args_calls_are_skipped():
    src = {
        "a.py": "def f(x):\n    return x\n\ndef g(*a):\n    return f(*a)\n",
    }
    assert apicheck.find_mismatches(src) == []


def test_ambiguous_name_is_skipped():
    """The same name defined twice with different arities is not resolved."""
    src = {
        "a.py": "def f(x):\n    return x\n",
        "b.py": "def f(x, y):\n    return x\n",
        "c.py": "def g():\n    return f()\n",
    }
    assert apicheck.find_mismatches(src) == []


def test_keyword_argument_satisfies_required():
    src = {
        "a.py": "def f(x, y):\n    return x\n\ndef g():\n    return f(1, y=2)\n",
    }
    assert apicheck.find_mismatches(src) == []
