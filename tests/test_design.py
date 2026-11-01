"""Spec-driven design: contract generation, validation, and saved artifacts."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import design
from pedacito.config import Config


def _design():
    return {
        "project_name": "todo",
        "summary": "A small todo CLI.",
        "modules": [
            {"path": "app.py", "responsibility": "CLI entry point",
             "interfaces": ["main(argv) -> int"], "depends_on": ["store.py"],
             "estimated_lines": 60},
            {"path": "store.py", "responsibility": "persist tasks",
             "interfaces": ["Store.add(task)"], "depends_on": [],
             "estimated_lines": 80},
        ],
        "data_shapes": [{"name": "Task", "fields": ["id: int", "title: str"]}],
        "tests": [{"path": "test_store.py", "covers": ["store.py"],
                   "cases": ["add then list", "empty store"]}],
        "risks": ["file locking"],
    }


def test_validate_accepts_a_clean_design(tmp_path):
    assert design.validate(_design(), Config()) == []


def test_validate_reports_cycles_missing_deps_and_width():
    d = _design()
    d["modules"][0]["depends_on"] = ["store.py", "missing.py"]
    d["modules"][1]["depends_on"] = ["app.py"]
    d["modules"][1]["interfaces"] = [f"f{i}()" for i in range(13)]
    d["modules"][1]["estimated_lines"] = 999
    warnings = design.validate(d, Config())
    assert any("dependency cycle" in w for w in warnings)
    assert any("unknown module missing.py" in w for w in warnings)
    assert any("auto_split_lines" in w for w in warnings)
    assert any("exposes 13 interfaces" in w for w in warnings)


def test_generate_saves_json_and_markdown(tmp_path):
    class Stub:
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            assert "SPECIFICATION" in messages[-1]["content"]
            return _design()

    cfg = Config()
    out = design.generate(Stub(), cfg, type("S", (), {"text": "todo CLI"})(),
                          tmp_path)
    assert out["project_name"] == "todo"
    assert (tmp_path / ".pedacito" / "design.json").is_file()
    md = (tmp_path / ".pedacito" / "design.md").read_text(encoding="utf-8")
    assert "## Modules" in md and "store.py" in md
    assert "todo" in design.planner_context(out)
    assert design.load(tmp_path)["project_name"] == "todo"
    assert "DESIGN BRIEF" in design.brief(tmp_path)
