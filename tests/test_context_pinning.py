"""Explicit context pinning (`@file` / `@symbol`) and project rules.

The unit-level contracts here are deliberately model-free: prompt parsing is
deterministic, the gathered-source packer must never evict a pin, and the GUI
routes are pure functions over ``STATE``. The TUI commands run on a headless
session with an in-memory stdout, like test_tui.py.
"""
from __future__ import annotations

import io
import threading
from pathlib import Path

import pytest

from pedacito import prompts, select
from pedacito.agent import _trim_gathered
from pedacito.chunker import chunk_source
from pedacito.config import Config
from pedacito.index import Index
from pedacito.tui import TuiSession


CALC = '''\
class ScientificCalculator:
    def evaluate_expression(self, expr):
        return eval(expr)
'''


def _workspace(tmp_path: Path) -> tuple[Path, Index]:
    """A tiny project + its exact index (no model, no summaries)."""
    (tmp_path / "pedacito").mkdir()
    (tmp_path / "pedacito" / "llm.py").write_text(
        "def complete(prompt):\n    return prompt\n", encoding="utf-8")
    (tmp_path / "gui.py").write_text(
        "def draw():\n    return 'ui'\n", encoding="utf-8")
    (tmp_path / "calc.py").write_text(CALC, encoding="utf-8")
    chunks = []
    for rel in ("pedacito/llm.py", "gui.py", "calc.py"):
        chunks.extend(chunk_source(rel, (tmp_path / rel).read_text(
            encoding="utf-8")))
    return tmp_path, Index(tmp_path, chunks)


def test_parse_pinned_references_resolves_files_symbols_and_ignores_noise(
        tmp_path):
    root, idx = _workspace(tmp_path)
    prompt = (
        "Refactor @pedacito/llm.py and update "
        "@ScientificCalculator.evaluate_expression, then tweak @gui.py.\n"
        "Email name@domain.com about it.\n"
        "```python\n"
        "@decorator\n"
        "def f():\n"
        "    return 1\n"
        "```\n"
    )
    pins = select.parse_pinned_references(prompt, idx)
    by_raw = {p.raw_token: p for p in pins}

    assert set(by_raw) == {
        "@pedacito/llm.py", "@ScientificCalculator.evaluate_expression",
        "@gui.py",
    }
    assert by_raw["@pedacito/llm.py"].kind == "file"
    assert by_raw["@pedacito/llm.py"].path == "pedacito/llm.py"
    assert "def complete" in by_raw["@pedacito/llm.py"].content
    assert by_raw["@gui.py"].path == "gui.py"

    symbol = by_raw["@ScientificCalculator.evaluate_expression"]
    assert symbol.kind == "symbol"
    assert symbol.path == "calc.py"
    assert symbol.symbol_name == "ScientificCalculator.evaluate_expression"
    assert (symbol.start_line, symbol.end_line) == (2, 3)
    assert "def evaluate_expression" in symbol.content
    assert "class ScientificCalculator" not in symbol.content

    assert all("@domain" not in p.raw_token for p in pins)
    assert all(p.raw_token != "@decorator" for p in pins)


def test_pinned_items_are_never_evicted_ahead_of_unpinned_source():
    pinned = ("--- pinned file pinned.py (L1-2)\n"
              "PINNED_DEFINITION = True\n")
    unpinned = "\n\n".join(
        f"--- read f{i}.py L1-40\n" + ("UNPINNED_BODY line\n" * 200)
        for i in range(4))
    source = pinned + "\n\n" + unpinned
    kept, dropped = _trim_gathered(source, lambda text: len(text) > 900)

    assert "PINNED_DEFINITION" in kept
    assert "UNPINNED_BODY" not in kept
    assert dropped >= 1


def test_project_rules_load_save_and_ride_the_static_answer_prefix(tmp_path):
    root = tmp_path / "rules-project"
    root.mkdir()
    assert prompts.load_project_rules(root) == ""

    saved = prompts.save_project_rules(root, "- Use pathlib only\n"
                                             "- Tests live under tests/\n")
    assert saved == root / ".pedacito" / "rules.md"
    assert saved.is_file()
    assert "pathlib" in prompts.load_project_rules(root)

    cfg = Config()
    cfg._root = str(root)
    cfg.active_tier = "full"
    system = prompts.answer_system(cfg, "EDIT FORMAT", task="add a helper")
    assert "PROJECT RULES" in system
    assert "- Use pathlib only" in system
    # Rules belong to the stable prefix, before the task-shaped edit brief.
    assert system.index("- Use pathlib only") < system.index("EDIT FORMAT")

    gather = prompts.gather_system(cfg, "GATHER BASE", "")
    assert "- Use pathlib only" in gather

    prompts.save_project_rules(root, "")
    assert not saved.exists()
    assert prompts.load_project_rules(root) == ""


def test_gui_context_suggest_and_rules_endpoints(tmp_path, monkeypatch):
    from pedacito.gui import server

    root, idx = _workspace(tmp_path)

    class FakeState:
        def __init__(self):
            self.root = root
            self.idx = idx
            self.lock = threading.RLock()

    monkeypatch.setattr(server, "STATE", FakeState())

    status, payload = server._context_suggest({"q": ["evaluate"]})
    assert status == 200 and payload["ok"] is True
    labels = [x["label"] for x in payload["items"]]
    assert "ScientificCalculator.evaluate_expression" in labels
    assert all(x["insert"].startswith("@") for x in payload["items"])
    assert any(x["kind"] == "symbol" for x in payload["items"])

    status, payload = server._context_suggest({"q": ["llm"]})
    assert status == 200
    assert any(x["kind"] == "file" and x["label"] == "pedacito/llm.py"
               for x in payload["items"])

    status, payload = server._rules_get({})
    assert status == 200 and payload["text"] == ""

    status, payload = server._rules_post(
        {}, {"text": "- Never touch generated files\n"})
    assert status == 200 and payload["exists"] is True
    assert (root / ".pedacito" / "rules.md").read_text(
        encoding="utf-8") == "- Never touch generated files\n"

    status, payload = server._rules_get({})
    assert "Never touch generated files" in payload["text"]


def test_tui_pin_and_rules_commands_operate_cleanly(tmp_path):
    root, idx = _workspace(tmp_path)
    cfg = Config()
    cfg._root = str(root)
    out = io.StringIO()
    session = TuiSession(root, cfg, idx=idx, headless=True, stdout=out)
    try:
        session.handle_line("/pin add gui.py")
        assert "gui.py" in session.pins
        assert cfg._pinned_refs == ["gui.py"]

        session.handle_line("/pin add ScientificCalculator.evaluate_expression")
        assert "ScientificCalculator.evaluate_expression" in session.pins

        session.handle_line("/pin list")
        shown = "\n".join(text for _kind, text in session.transcript)
        assert "Pinned context (2)" in shown
        assert "gui.py" in shown

        session.handle_line("/rules set Use pathlib only")
        rules = root / ".pedacito" / "rules.md"
        assert rules.is_file()
        assert "Use pathlib only" in rules.read_text(encoding="utf-8")

        session.handle_line("/rules")
        shown = "\n".join(text for _kind, text in session.transcript)
        assert "Project rules" in shown
        assert "Use pathlib only" in shown

        session.handle_line("/rules clear")
        assert not rules.exists()

        session.handle_line("/pin clear")
        assert session.pins == []
        assert cfg._pinned_refs == []
    finally:
        session.close()


@pytest.mark.parametrize("reference", ["gui.py", "@gui.py", "gui"])
def test_resolve_pinned_reference_accepts_optional_at(reference, tmp_path):
    _root, idx = _workspace(tmp_path)
    pins = select.resolve_pinned_references([reference], idx)
    assert pins
    assert pins[0].kind == "file"
    assert pins[0].path == "gui.py"
