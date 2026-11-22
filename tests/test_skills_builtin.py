"""The built-in procedural skills: the original small-model trio plus the
domain checklists, all loadable through the catalog and the `skill` action."""
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


def test_catalog_lists_all_nineteen_with_one_line_summaries(tmp_path):
    cat = skills.catalog(tmp_path, Config(), limit=10000)
    for name in skills.BUNDLED:
        assert f"- {name}:" in cat
    assert len(skills.names(tmp_path, Config())) == 19
    # One line per skill: the body never leaks into the catalog.
    assert all(line.startswith("- ") for line in cat.splitlines()[1:])
    assert "SEARCH" not in cat


def test_modern_web_design_ui_skill_has_the_zero_build_playbook():
    desc, body, triggers = skills.BUNDLED["modern-web-design-ui"]
    assert len(body) <= 580
    for token in ("#0b0f17", "clamp()", "65ch", "translateY(-1px)",
                  "@tailwind", "cdn.tailwindcss.com", "hamburger"):
        assert token in body, token
    for trigger in ("website", "webpage", "landing page", "personal site",
                    "bio", "portfolio", "styling", "modern", "sleek",
                    "appealing", "css", "layout", "theme", "hero", "card"):
        assert trigger in triggers, trigger


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


def test_safe_git_commit_loads_selects_and_appears_in_the_catalog(tmp_path):
    text, info = skills.load(tmp_path, Config(), "safe-git-commit")
    assert info is not None and info["source"] == "bundled"
    assert text.startswith("SKILL safe-git-commit (bundled):")
    out = tools.dispatch(
        types.SimpleNamespace(root=tmp_path), Config(),
        {"action": "skill", "symbol": "safe-git-commit"})
    assert out.startswith("SKILL safe-git-commit (bundled):")
    picked = {row["name"] for row in skills.select_skills(
        tmp_path, "commit my changes with a good message", Config())}
    assert "safe-git-commit" in picked
    cat = skills.catalog(tmp_path, Config(), limit=10000)
    assert "- safe-git-commit:" in cat


def test_catalog_and_selection_stay_empty_with_skills_disabled(tmp_path):
    cfg = Config()
    cfg.use_skills = False
    assert skills.catalog(tmp_path, cfg, limit=10000) == ""
    assert skills.select_skills(tmp_path, "commit my changes", cfg) == []
