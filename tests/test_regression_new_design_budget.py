"""Regression tests for BUG-010 (DeepSeek bug hunt, v0.3.1).

``new --design --apply --max-minutes N`` ignored the cap outside --complex:
a small model burned the whole budget on design retries and the run was
killed by the external timeout with no sources and no closing message.
The plain new path now carries a wall-clock deadline (checked between
calls and between files, never mid-call -- the same pattern as the auto
loop), and designed Python test files are normalized to pytest-discoverable
``test_*.py`` names (the 27B retry exited 1 over ``todo_store.test.py``).
"""

import time
import types

import pytest

from pedacito import design, llm as llm_mod, scaffold
from pedacito.config import Config
from pedacito.edits import Result
from pedacito.llm import LLMClient, LLMError


def _cfg(**kw):
    cfg = Config()
    cfg.base_url = "http://127.0.0.1:9/v1"
    cfg.model = "stub-model"
    cfg.log_sessions = False
    cfg.log_events = False
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


# ------------------------------------------------- chat_json deadline gate
def test_chat_json_stops_retrying_once_the_deadline_passed(monkeypatch):
    calls = []

    def post(*a, **k):
        calls.append(1)
        raise AssertionError("no call may start past the deadline")

    monkeypatch.setattr("pedacito.llm.requests.post", post)
    client = LLMClient(_cfg())
    with pytest.raises(LLMError) as e:
        client.chat_json([{"role": "user", "content": "x"}], {"type": "object"},
                         max_tokens=100, deadline=time.monotonic() - 1)
    assert "time budget" in str(e.value)
    assert calls == []


def test_chat_json_deadline_expiring_between_attempts(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(llm_mod.time, "monotonic", lambda: clock[0])
    client = LLMClient(_cfg())
    attempts = []

    def fake_chat(messages, max_tokens=0, **kw):
        attempts.append(1)
        clock[0] += 100.0          # this call spends the whole budget
        client.last_finish_reason = "stop"
        return "sorry, no JSON today"

    monkeypatch.setattr(client, "chat", fake_chat)
    with pytest.raises(LLMError) as e:
        client.chat_json([{"role": "user", "content": "x"}], {"type": "object"},
                         max_tokens=100, deadline=50.0)
    assert "time budget" in str(e.value)
    assert attempts == [1]         # the first call ran; the re-ask did not


def test_chat_json_without_a_deadline_keeps_its_retries(monkeypatch):
    client = LLMClient(_cfg())
    attempts = []

    def fake_chat(messages, max_tokens=0, **kw):
        attempts.append(1)
        client.last_finish_reason = "stop"
        return "sorry, no JSON today"

    monkeypatch.setattr(client, "chat", fake_chat)
    with pytest.raises(LLMError) as e:
        client.chat_json([{"role": "user", "content": "x"}], {"type": "object"},
                         max_tokens=100)
    assert "would not produce valid JSON" in str(e.value)
    assert len(attempts) == 3


# ------------------------------------------------- build_project file gate
def _two_file_plan(tmp_path):
    for rel in ("a.py", "b.py"):
        (tmp_path / rel).write_text(f"# stub {rel}\n", encoding="utf-8")
    plan = scaffold.Plan(
        "p", "s",
        files=[scaffold.PlannedFile("a.py"), scaffold.PlannedFile("b.py")],
        spec=scaffold.Spec("build a and b"))
    idx = types.SimpleNamespace(root=tmp_path)
    return plan, idx


def test_build_project_stops_between_files_at_the_deadline(tmp_path,
                                                           monkeypatch):
    plan, idx = _two_file_plan(tmp_path)
    calls = []
    monkeypatch.setattr(scaffold, "build_pass",
                        lambda *a, **k: calls.append(1) or ([], ""))
    summary = scaffold.build_project(
        None, _cfg(), idx, plan, verbose=False,
        deadline=time.monotonic() - 1)
    assert calls == []
    assert summary["capped"] is True
    assert summary["skipped"] == ["a.py", "b.py"]
    assert summary["done"] == [] and summary["failed"] == []


def test_build_project_without_a_deadline_builds_every_file(tmp_path,
                                                            monkeypatch):
    plan, idx = _two_file_plan(tmp_path)
    monkeypatch.setattr(
        scaffold, "build_pass",
        lambda client, cfg, idx_, plan_, t, *a, **k: (
            [Result(True, t.path, "applied")], "ok"))
    monkeypatch.setattr(scaffold.index_mod, "reindex_file",
                        lambda idx_, rel, cfg: idx_)
    summary = scaffold.build_project(None, _cfg(), idx, plan, verbose=False)
    assert summary["done"] == ["a.py", "b.py"]
    assert not summary.get("capped")


# ------------------------------------------------- run_new threads the cap
def test_run_new_gives_the_design_call_a_deadline(tmp_path, monkeypatch):
    seen = {}

    def fake_generate(client, cfg, spec, root, save_it=True, **kw):
        seen.update(kw)
        seen["max_task_seconds"] = getattr(cfg, "max_task_seconds", None)
        return design._norm({"project_name": "p", "summary": "s"})

    monkeypatch.setattr(design, "generate", fake_generate)
    monkeypatch.setattr(scaffold, "build_plan",
                        lambda spec, client, cfg, root, *a, **k:
                        scaffold.Plan("p", "s", spec=spec))
    cfg = _cfg()
    before = time.monotonic()
    code = scaffold.run_new(tmp_path / "proj", cfg, None,
                            scaffold.Spec("a tiny todo app"),
                            verbose=False, design=True, max_minutes=5)
    assert code == 0
    assert "deadline" in seen, "the design call must carry the run's deadline"
    assert before + 300 <= seen["deadline"] <= time.monotonic() + 300
    assert seen["max_task_seconds"] == 300


# ------------------------------------------------- pytest-discoverable tests
def test_design_system_teaches_the_pytest_naming_rule():
    assert "test_" in design.DESIGN_SYSTEM
    assert ".test.py" in design.DESIGN_SYSTEM


def test_norm_renames_undiscoverable_python_test_paths():
    d = design._norm({
        "project_name": "todo", "summary": "s",
        "modules": [
            {"path": "todo_store.py", "responsibility": "store",
             "interfaces": [], "depends_on": [], "estimated_lines": 40},
            {"path": "cli.py", "responsibility": "cli",
             "interfaces": [], "depends_on": ["todo_store.py"],
             "estimated_lines": 40},
        ],
        "data_shapes": [],
        "tests": [
            {"path": "todo_store.test.py", "covers": ["todo_store.py"],
             "cases": ["add and list"]},
            {"path": "tests/api.spec.py", "covers": ["cli.py"],
             "cases": ["help"]},
            {"path": "test_cli.py", "covers": ["cli.py"], "cases": ["run"]},
            {"path": "store_test.py", "covers": ["todo_store.py"],
             "cases": ["load"]},
            {"path": "app.test.js", "covers": [], "cases": ["renders"]},
        ],
        "risks": [],
    })
    paths = [t["path"] for t in d["tests"]]
    assert paths == ["test_todo_store.py", "tests/test_api.py",
                     "test_cli.py", "store_test.py", "app.test.js"]


def test_norm_remaps_references_to_a_renamed_module():
    d = design._norm({
        "project_name": "p", "summary": "s",
        "modules": [
            {"path": "store.test.py", "responsibility": "r",
             "interfaces": [], "depends_on": [], "estimated_lines": 10},
            {"path": "cli.py", "responsibility": "r",
             "interfaces": [], "depends_on": ["store.test.py"],
             "estimated_lines": 10},
        ],
        "data_shapes": [],
        "tests": [{"path": "store.test.py", "covers": ["store.test.py"],
                   "cases": ["c"]}],
        "risks": [],
    })
    assert [m["path"] for m in d["modules"]] == ["test_store.py", "cli.py"]
    assert d["modules"][1]["depends_on"] == ["test_store.py"]
    assert d["tests"][0]["path"] == "test_store.py"
    assert d["tests"][0]["covers"] == ["test_store.py"]
