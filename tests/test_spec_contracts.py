"""Offline tests for the Prompt/Spec Architect and the contract-first
interface lock (`pedacito/spec.py`), plus their integration into `run_loop`
(task injection + the cross-file gate) and the GUI (`/api/spec/refine`).

No LLM server is needed: the architect's schema call is stubbed with plain
duck-typed objects, and every failure path must land on the deterministic
fallback rather than an error. The loop tests monkeypatch `run_task` exactly
like test_loop.py so only the OUTER contract behaviour is under test."""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import loop, spec  # noqa: E402
from pedacito.config import Config  # noqa: E402


# ------------------------------------------------------------- contract map --

def test_extract_project_contracts_lists_exact_signatures(tmp_path):
    (tmp_path / "parser.py").write_text(
        "def parse_line(text, sep=','):\n    return text.split(sep)\n\n"
        "class TableParser:\n    def __init__(self, encoding='utf-8'):\n"
        "        self.encoding = encoding\n"
        "    def feed(self, rows):\n        return len(rows)\n", encoding="utf-8")
    (tmp_path / "ui.js").write_text(
        "function renderTable(rows, opts = {}) {\n  return rows.length\n}\n", encoding="utf-8")
    (tmp_path / "test_parser.py").write_text(
        "def test_nothing():\n    assert True\n", encoding="utf-8")
    got = spec.extract_project_contracts(tmp_path)
    assert "def parse_line(text, sep=',')" in got
    assert "class TableParser" in got and "__init__(encoding='utf-8')" in got
    assert "renderTable(rows, opts = {})" in got or "function renderTable" in got
    assert "test_parser" not in got          # test files are not public API


def test_contract_map_block_wraps_map_in_lock_header(tmp_path):
    (tmp_path / "a.py").write_text("def one(x):\n    return x\n", encoding="utf-8")
    block = spec.contract_map_block(tmp_path)
    assert block.startswith(spec.CONTRACT_HEADER)
    assert block.endswith(spec.CONTRACT_FOOTER)
    assert "def one(x)" in block
    assert "DO NOT BREAK SIGNATURES" in block
    # a near-empty project must not inject anything
    assert spec.contract_map_block(tmp_path / "missing") == ""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert spec.contract_map_block(empty) == ""


def test_map_caps_size_and_prefers_modified(tmp_path):
    for i in range(60):
        (tmp_path / f"mod{i:02d}.py").write_text(
            f"def fn_{i}(a, b, c, d, e, f, g):\n    return a\n", encoding="utf-8")
    (tmp_path / "mod59.py").write_text("def touched_symbol(z):\n    return z\n", encoding="utf-8")
    block = spec.contract_map_block(tmp_path, modified=["mod59.py"])
    assert "touched_symbol" in block          # freshest contracts survive
    assert len(block) < 4000                  # ~300 tokens, never a dump


# ------------------------------------------------------------ the verifier ---

def test_verify_flags_missing_symbol_and_bad_arity(tmp_path):
    (tmp_path / "parser.py").write_text(
        "def parse_line(text, sep=','):\n    return text\n\n"
        "class TableParser:\n    def __init__(self, encoding='utf-8'):\n"
        "        pass\n", encoding="utf-8")
    (tmp_path / "cli.py").write_text(
        "from parser import parse_line, TableParser\n"
        "from parser import missing_fn\n"
        "import json\n"
        "from dataclasses import dataclass\n\n"
        "def main():\n"
        "    TableParser('utf-8', 'extra')\n"
        "    parse_line('a')\n"
        "    json.loads('{}')\n", encoding="utf-8")
    errs = spec.verify_cross_file_contracts(tmp_path)
    assert any("missing_fn" in e and "only defines" in e and
               "parse_line" in e for e in errs)
    assert any("TableParser" in e and "positional" in e for e in errs)
    assert all("json" not in e and "dataclass" not in e for e in errs)
    assert all(e.startswith("ContractError in cli.py:") for e in errs)


def test_verify_clean_project_reports_nothing(tmp_path):
    (tmp_path / "a.py").write_text("def f(x, y=1):\n    return x + y\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("from a import f\nprint(f(1), f(2, 3))\n", encoding="utf-8")
    assert spec.verify_cross_file_contracts(tmp_path) == []


def test_verify_relative_imports(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("def boot():\n    return 1\n", encoding="utf-8")
    (pkg / "m2.py").write_text("def go(a):\n    return a\n", encoding="utf-8")
    (pkg / "main.py").write_text(
        "from . import m2, boot, missing\n"
        "from .m2 import gone\n"
        "from .m2 import go\n"
        "m2.boot()\ngo(1)\n", encoding="utf-8")
    errs = spec.verify_cross_file_contracts(tmp_path)
    assert any("missing" in e for e in errs)
    assert any("gone" in e for e in errs)
    assert not any("'m2'" in e or "'boot'" in e.split("but")[0]
                   for e in errs)      # valid submodule/attr imports pass


def test_verify_packages_and_reexports(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from .core import Engine\n", encoding="utf-8")
    (pkg / "core.py").write_text("class Engine:\n    def run(self):\n"
                                 "        return 'vroom'\n", encoding="utf-8")
    (tmp_path / "use.py").write_text(
        "from pkg import Engine\nfrom pkg import absent\n"
        "Engine().run()\n", encoding="utf-8")
    errs = spec.verify_cross_file_contracts(tmp_path)
    assert any("absent" in e for e in errs)
    assert not any("imports 'Engine'" in e for e in errs)


def test_verify_trusts_ambiguous_imports(tmp_path):
    (tmp_path / "a.py").write_text("def f(x):\n    return x\n", encoding="utf-8")
    (tmp_path / "b.py").write_text(
        "from a import *          # star: unknown surface, skip\n"
        "import numpy as np       # third party: never guessed\n"
        "from sklearn import model_selection\n"
        "f(np, model_selection)\n", encoding="utf-8")
    assert spec.verify_cross_file_contracts(tmp_path) == []


def test_verify_shadowed_names_are_trusted(tmp_path):
    (tmp_path / "a.py").write_text("def f(x):\n    return x\n", encoding="utf-8")
    (tmp_path / "b.py").write_text(
        "from a import f\n\ndef f(x, y, z):\n    return x\n\nf(1, 2, 3)\n", encoding="utf-8")
    assert spec.verify_cross_file_contracts(tmp_path) == []


# ----------------------------------------------------------- refine_prompt ---

class _OK:
    def __init__(self, reply):
        self.reply = reply

    def chat_json(self, messages, schema, max_tokens=None,
                  disable_thinking=False):
        assert messages[0]["role"] == "system"
        assert "objective" in schema["properties"]
        return self.reply


class _Boom:
    def chat_json(self, *a, **k):
        raise RuntimeError("connection refused")


def test_refine_prompt_fallback_shape(tmp_path):
    (tmp_path / "parser.py").write_text("def parse_line(t):\n    return t\n", encoding="utf-8")
    got = spec.refine_prompt("add a CSV export", tmp_path, None, None)
    assert got["ok"] is True
    assert got["original"] == "add a CSV export"
    for section in ("OBJECTIVE", "FILES TO CREATE OR MODIFY",
                    "INTERFACE CONTRACTS", "REQUIREMENTS", "EDGE CASES",
                    "VERIFICATION", "ORIGINAL REQUEST (verbatim)"):
        assert section in got["refined"]
    assert isinstance(got["contracts"], list)
    # the repo snapshot informs the fallback even without any model
    assert "parser.py" in got["refined"]


def test_refine_prompt_with_model_reply(tmp_path):
    reply = {"objective": "Ship CSV export in cli.py.",
             "files": [{"path": "./cli.py", "action": "modify",
                        "responsibility": "add --csv flag"},
                       {"path": "cli.py", "action": "modify",
                        "responsibility": "duplicate entry"},
                       {"path": "../escape.py", "action": "create",
                        "responsibility": "path traversal attempt"}],
             "contracts": ["def export_csv(rows, path) -> str"],
             "requirements": ["1. Empty input writes only the header."],
             "edge_cases": [],
             "verification": "pedacito ask ... --apply"}
    got = spec.refine_prompt("add csv", tmp_path, Config(), _OK(reply))
    assert "def export_csv(rows, path) -> str" in got["refined"]
    assert got["contracts"] == ["def export_csv(rows, path) -> str"]
    assert got["refined"].count("cli.py") >= 1
    assert "escape.py" not in got["refined"]        # path sanitised
    assert "add csv" in got["refined"]              # original is preserved


def test_refine_prompt_survives_client_errors(tmp_path):
    for bad in (_Boom(), object(), None):
        got = spec.refine_prompt("just do it", tmp_path, None, bad
                                 if bad is not None else None)
        assert got["ok"] is True and "OBJECTIVE" in got["refined"]


def test_refine_prompt_uses_planner_max_tokens(tmp_path):
    seen = {}

    class Probe:
        def chat_json(self, messages, schema, max_tokens=None,
                      disable_thinking=False):
            seen["max_tokens"] = max_tokens
            seen["disable_thinking"] = disable_thinking
            return {"objective": "x", "files": [], "contracts": [],
                    "requirements": ["r"], "verification": "v"}
    cfg = Config()
    cfg.max_tokens_answer = 1234
    spec.refine_prompt("do it", tmp_path, cfg, Probe())
    assert seen == {"max_tokens": 1234, "disable_thinking": True}


# ------------------------------------------------------------ run_loop lock --

def _cfg():
    c = Config()
    c.log_sessions = False
    return c


def _ok(file="app.py"):
    return types.SimpleNamespace(ok=True, file=file)


def _result(idx, changed):
    return loop.TaskResult([_ok()], "reply", True, changed, idx)


def test_run_loop_injects_contract_map(tmp_path, monkeypatch):
    (tmp_path / "lib.py").write_text("def helper(a, b=1):\n    return a\n", encoding="utf-8")
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal", ["task A", "task B"])
    idx = types.SimpleNamespace(root=tmp_path)
    calls = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None, is_repair=False):
        calls.append(task)
        return _result(idx_, ["lib.py"])

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
    assert code == 0
    assert len(calls) == 2
    for c in calls:
        assert spec.CONTRACT_HEADER in c
        assert "def helper(a, b=1)" in c
        assert c.endswith("[END INTERFACE CONTRACTS]") or "helper" in c
        # the existing invariant: the title still leads the prompt
        assert c.split("\n\n")[0] in ("task A", "task B")


def test_run_loop_injection_off_when_interface_lock_false(tmp_path,
                                                          monkeypatch):
    (tmp_path / "lib.py").write_text("def helper(a):\n    return a\n", encoding="utf-8")
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal", ["task A"])
    idx = types.SimpleNamespace(root=tmp_path)
    calls = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None, is_repair=False):
        calls.append(task)
        return _result(idx_, ["lib.py"])

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    cfg = _cfg()
    cfg.interface_lock = False
    assert loop.run_loop(tmp_path, cfg, None, idx, plan, apply=True) == 0
    assert calls == ["task A"]          # untouched: no map, no gate note


def test_run_loop_gate_feeds_contract_errors_into_repair(tmp_path,
                                                         monkeypatch):
    (tmp_path / "lib.py").write_text("def helper(a):\n    return a\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("from lib import helper, vanished\n"
                                     "helper(1)\n", encoding="utf-8")
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal", ["task A"])
    idx = types.SimpleNamespace(root=tmp_path)
    seen = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None, is_repair=False):
        seen.append(task)
        return _result(idx_, ["app.py"])

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
    assert code == 1                        # the task is reported failed
    assert len(seen) > 1                    # repair rounds were attempted
    assert any("CROSS-FILE INTERFACE CONTRACT VIOLATIONS" in t
               and "vanished" in t for t in seen[1:])
    assert spec.CONTRACT_HEADER in seen[0]  # locked prompt was sent first


def test_run_loop_gate_passes_clean_project(tmp_path, monkeypatch):
    (tmp_path / "lib.py").write_text("def helper(a):\n    return a\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("from lib import helper\nhelper(1)\n", encoding="utf-8")
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal", ["task A"])
    idx = types.SimpleNamespace(root=tmp_path)

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None, is_repair=False):
        return _result(idx_, ["app.py"])

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    assert loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True) == 0


# ------------------------------------------------------------- GUI route ----

def test_gui_refine_route(tmp_path, monkeypatch):
    from pedacito.gui import core, server

    monkeypatch.setattr(core.STATE, "root", tmp_path)
    monkeypatch.setattr(core.STATE, "cfg", Config())
    monkeypatch.setattr(core.STATE, "client", None)   # offline: fallback
    route = server.POST_ROUTES["/api/spec/refine"]

    (tmp_path / "svc.py").write_text("def run(cfg): \n    return cfg\n", encoding="utf-8")
    status, j = route({}, {"prompt": "add a health endpoint to svc.py"})
    assert status == 200 and j["ok"] is True
    assert "OBJECTIVE" in j["refined"]
    assert j["original"] == "add a health endpoint to svc.py"
    assert isinstance(j["contracts"], list)

    status, j = route({}, {"prompt": "   "})
    assert status == 400 and j["ok"] is False

    monkeypatch.setattr(core.STATE, "root", None)
    status, j = route({}, {"prompt": "anything"})
    assert status == 400 and "project" in j["detail"].lower()
