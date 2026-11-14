"""The three built-in small-model procedural skills."""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import skills, tools
from pedacito.config import Config

NEW = {
    "surgical-symbol-edit": (
        "SYMBOL", "AFTER_SYMBOL", "LINES", "outline"),
    "frontend-dom-audio-debug": (
        "getElementById", "AudioContext", "ctx.destination", "hold",
        "eval"),
    "safe-multi-file-rename": (
        "usages", "rename_symbol", "ClassName.method", "run_test"),
}


def test_catalog_lists_all_eight_with_one_line_summaries(tmp_path):
    cat = skills.catalog(tmp_path, Config())
    for name in NEW:
        assert f"- {name}:" in cat
    assert len(skills.names(tmp_path, Config())) == 8
    # One line per skill: the body never leaks into the catalog.
    assert all(line.startswith("- ") for line in cat.splitlines()[1:])
    assert "SEARCH" not in cat


def test_new_skills_load_through_dispatch(tmp_path):
    cfg = Config()
    idx = types.SimpleNamespace(root=tmp_path)
    for name, words in NEW.items():
        out = tools.dispatch(idx, cfg, {"action": "skill", "symbol": name})
        assert out.startswith(f"SKILL {name}")
        for word in words:
            assert word in out, (name, word)


def test_unknown_skill_lists_the_new_names(tmp_path):
    text, info = skills.load(tmp_path, Config(), "nope")
    assert info is None and "ERROR" in text
    for name in NEW:
        assert name in text
