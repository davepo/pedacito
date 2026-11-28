"""WP-4104B: deterministic autopsy and the tool availability inspector.

Autopsy is a pure function over typed event logs; the inspector must report
exactly the action set `schema_for` would send, and `--allow-tool` may only
restore a tool that dynamic pruning removed.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import autopsy, events, i18n, memory, tools  # noqa: E402
from pedacito import cli  # noqa: E402
from pedacito.config import Config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _write_log(root: Path, records: list[dict], name: str = "20260101-000000-1.jsonl") -> Path:
    d = root / ".pedacito" / events.EVENT_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / name
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _ev(kind: str, seq: int, **fields) -> dict:
    return {"t": float(seq), "kind": kind, "session_id": "s1", "seq": seq, **fields}


# ----------------------------------------------------------------- autopsy --

def test_autopsy_result_is_frozen_and_categories_are_exact():
    result = autopsy.Autopsy("unknown", (), (), True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.category = "timeout"  # type: ignore[misc]
    assert autopsy.CATEGORIES == (
        "unread_edit", "format_error", "truncation", "schema_rejection",
        "test_regression", "flake", "api_hallucination", "context_overflow",
        "permission_denial", "timeout", "budget_stop", "partial_batch", "unknown",
    )
    assert set(autopsy.SEVERITY) == set(autopsy.CATEGORIES) - {"unknown"}


def test_missing_logs_are_unknown_and_incomplete(tmp_path):
    result = autopsy.autopsy(tmp_path, "last")
    assert result.category == "unknown"
    assert result.evidence == ()
    assert result.incomplete is True


def test_no_typed_evidence_is_unknown_not_a_diagnosis(tmp_path):
    _write_log(tmp_path, [_ev("step", 1, action="read", n=1),
                          _ev("exit", 2, code=0)])
    result = autopsy.autopsy(tmp_path, "last")
    assert result.category == "unknown"
    assert result.incomplete is True


def test_selection_is_severity_first_then_earliest_seq(tmp_path):
    _write_log(tmp_path, [
        _ev("gate_error", 1, gate="syntax", file="a.py"),
        _ev("task_budget", 2, phase="gather", reason="wall clock"),
        _ev("step", 3, action="run", obs_code="timeout", n=3),
        _ev("step", 4, action="run", obs_code="timeout", n=4),
        _ev("exit", 5, code=1),
    ])
    result = autopsy.autopsy(tmp_path, "last")
    assert result.category == "timeout"
    assert [item["seq"] for item in result.evidence] == [3, 4]
    assert result.incomplete is False


def test_ties_break_on_earliest_seq_then_name(tmp_path):
    _write_log(tmp_path, [
        _ev("payload_cut", 1, field="content"),
        _ev("gate_error", 2, gate="syntax", file="a.py"),
        _ev("exit", 3, code=1),
    ])
    result = autopsy.autopsy(tmp_path, "last")
    # truncation (3) is less severe than format_error (5): format_error wins.
    assert result.category == "format_error"
    _write_log(tmp_path, [
        _ev("step_rejected", 1, reason="malformed", n=1),
        _ev("gate_error", 2, gate="syntax", file="a.py"),
        _ev("exit", 3, code=1),
    ], name="20260101-000001-1.jsonl")
    again = autopsy.autopsy(tmp_path, "last")
    assert again.category == "schema_rejection"
    assert again == autopsy.autopsy(tmp_path, "last")


def test_evidence_is_redacted_and_bounded(tmp_path):
    secret = "sk-" + "a" * 40
    _write_log(tmp_path, [
        _ev("gate_error", 1, gate="syntax", file=f"src/{secret}.py",
            reason=f"token {secret}"),
        _ev("exit", 2, code=1),
    ])
    result = autopsy.autopsy(tmp_path, "last")
    blob = json.dumps(result.evidence)
    assert secret not in blob
    assert "reason" not in result.evidence[0]
    assert result.evidence[0]["seq"] == 1
    assert result.evidence[0]["kind"] == "gate_error"


def test_terminal_missing_or_gap_marks_incomplete(tmp_path):
    _write_log(tmp_path, [_ev("gate_error", 1, gate="syntax", file="a.py")])
    assert autopsy.autopsy(tmp_path, "last").incomplete is True
    _write_log(tmp_path, [
        _ev("gate_error", 1, gate="syntax", file="a.py"),
        _ev("exit", 4, code=1),
    ], name="20260101-000002-1.jsonl")
    assert autopsy.autopsy(tmp_path, "last").incomplete is True


def test_run_id_selects_session_or_file(tmp_path):
    _write_log(tmp_path, [
        {"kind": "task_budget", "session_id": "old", "seq": 1, "phase": "gather"},
        {"kind": "exit", "session_id": "old", "seq": 2, "code": 1},
    ], name="20250101-000000-1.jsonl")
    _write_log(tmp_path, [
        {"kind": "gate_error", "session_id": "new", "seq": 1, "gate": "syntax"},
        {"kind": "exit", "session_id": "new", "seq": 2, "code": 1},
    ], name="20260101-000000-1.jsonl")
    assert autopsy.autopsy(tmp_path, "last").category == "format_error"
    assert autopsy.autopsy(tmp_path, "old").category == "budget_stop"
    assert autopsy.autopsy(tmp_path, "nope").category == "unknown"


def test_real_event_log_round_trip(tmp_path):
    log = events.EventLog(tmp_path, True, True, 20)
    log.emit("step", action="write", file="a.py", n=2, error=True,
             obs_code="unread_edit")
    log.emit("gate_error", gate="syntax", file="a.py")
    log.emit("exit", code=1)
    result = autopsy.autopsy(tmp_path, "last")
    # format_error (severity 5) outranks unread_edit (4) even though it is later.
    assert result.category == "format_error"
    assert result.incomplete is False
    assert result.evidence[0]["file"] == "a.py"
    assert result.evidence[0]["seq"] is None


@pytest.mark.parametrize("text,category", [
    ("ERROR: this task never read the file a.py. `read` it first", "unread_edit"),
    ("ERROR: denied by policy (rule-1): no", "permission_denial"),
    ("RUN TEST: x\nSTOPPED: timed out after 15s (killed)", "timeout"),
    ("some output\n[truncated: context budget reached]", "truncation"),
    ("plain successful output", ""),
])
def test_observation_markers_are_deterministic(text, category):
    assert autopsy.classify_observation(text) == category


def test_typed_category_maps_only_typed_fields():
    assert autopsy.typed_category({"kind": "verify", "errors": 1,
                                   "kinds": ["unread"]}) == "unread_edit"
    assert autopsy.typed_category({"kind": "verify", "errors": 0,
                                   "kinds": ["unread"]}) == ""
    assert autopsy.typed_category({"kind": "test_result", "ok": False,
                                   "stopped": ""}) == "test_regression"
    assert autopsy.typed_category({"kind": "test_result", "ok": False,
                                   "stopped": "timeout"}) == "timeout"
    assert autopsy.typed_category({"kind": "llm_error",
                                   "code": "context_overflow"}) == "context_overflow"
    assert autopsy.typed_category({"kind": "llm_error",
                                   "code": "raw body text"}) == ""
    assert autopsy.typed_category({"kind": "step", "obs_code": "budget"}) == ""
    assert autopsy.typed_category({"kind": "tool_result",
                                   "obs_code": "timeout"}) == "timeout"


def test_payload_and_rendering_are_plain(tmp_path):
    _write_log(tmp_path, [_ev("payload_cut", 1, field="content"),
                          _ev("exit", 2, code=1)])
    result = autopsy.autopsy(tmp_path, "last")
    body = autopsy.payload(result)
    assert body["category"] == "truncation"
    assert body["incomplete"] is False
    assert "field=content" in autopsy.render_evidence(result.evidence[0])


def test_every_category_and_step_has_locale_text():
    en = json.loads((ROOT / "pedacito" / "locales" / "en.json").read_text(encoding="utf-8"))
    es = json.loads((ROOT / "pedacito" / "locales" / "es.json").read_text(encoding="utf-8"))
    keys = [f"autopsy.category.{c}" for c in autopsy.CATEGORIES]
    keys += [f"autopsy.next.{s}" for steps in autopsy._NEXT_STEPS.values() for s in steps]
    keys += ["autopsy.next.check_run_log_completeness"]
    for key in keys:
        assert key in en and key in es, key


def test_cli_autopsy_reports_and_exits_zero(tmp_path, capsys):
    _write_log(tmp_path, [_ev("task_budget", 1, phase="gather", reason="x"),
                          _ev("exit", 2, code=1)])
    assert cli.main(["autopsy", "--last", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert i18n.t("autopsy.category.budget_stop") in out
    assert "task_budget" in out


def test_cli_autopsy_json(tmp_path, capsys):
    _write_log(tmp_path, [_ev("gate_error", 1, gate="syntax", file="a.py"),
                          _ev("exit", 2, code=1)])
    assert cli.main(["autopsy", "--last", "--json", str(tmp_path)]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["category"] == "format_error"
    assert body["incomplete"] is False
    assert body["evidence"][0]["file"] == "a.py"


def test_gui_autopsy_route_is_read_only(tmp_path, monkeypatch):
    from pedacito.gui import server

    _write_log(tmp_path, [_ev("step", 1, action="run", obs_code="timeout", n=1),
                          _ev("exit", 2, code=1)])
    monkeypatch.setattr(server, "_project_root", lambda: str(tmp_path))
    status, body = server.GET_ROUTES["/api/autopsy"]({})
    assert status == 200
    assert body["ok"] is True
    assert body["category"] == "timeout"
    assert body["label"]
    assert body["next_labels"]
    assert "/api/autopsy" not in server.POST_ROUTES
    monkeypatch.setattr(server, "_project_root", lambda: None)
    status, body = server.GET_ROUTES["/api/autopsy"]({})
    assert status == 400 and body["ok"] is False


# ------------------------------------------- controlled budget and timeouts --

def test_budget_stop_and_timeout_kinds_map_deterministically(tmp_path):
    _write_log(tmp_path, [_ev("budget_stop", 1, reason_code="max_cost"),
                          _ev("exit", 2, code=1)])
    result = autopsy.autopsy(tmp_path, "last")
    assert result.category == "budget_stop" and result.incomplete is False
    _write_log(tmp_path, [
        _ev("llm_start", 1, model="m"),
        _ev("timeout", 2, timeout_s=52.6, reason_code="run_deadline"),
        _ev("exit", 3, code=1),
    ], name="20260101-000001-1.jsonl")
    result = autopsy.autopsy(tmp_path, "last")
    assert result.category == "timeout" and result.incomplete is False
    assert result.evidence[0]["timeout_s"] == 52.6
    assert autopsy.typed_category({"kind": "budget_stop"}) == "budget_stop"
    assert autopsy.typed_category({"kind": "timeout"}) == "timeout"


def _real_cfg(tmp_path, **overrides):
    cfg = Config()
    cfg.base_url = "http://127.0.0.1:9/v1"
    cfg.model = "stub-model"
    cfg.log_sessions = False
    cfg.log_events = True
    cfg._root = str(tmp_path)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_strict_cost_refusal_autopsies_as_budget_stop(tmp_path, monkeypatch):
    from pedacito import session_ctx
    from pedacito.llm import BudgetExceeded, LLMClient

    def no_request(url, **kw):
        raise AssertionError("no request may start past the strict cap")

    monkeypatch.setattr("pedacito.llm.requests.post", no_request)
    cfg = _real_cfg(tmp_path, strict_cost_budget=True, max_cost=0.000001)
    session_ctx.set_run_budget(session_ctx.RunBudget(max_cost_usd=0.000001), cfg)
    events.get(tmp_path, cfg).emit("session_start", cmd="ask")
    with pytest.raises(BudgetExceeded):
        LLMClient(cfg).chat([{"role": "user", "content": "hi"}], 100, stream=False)
    events.get(tmp_path, cfg).emit("exit", code=1)
    result = autopsy.autopsy(tmp_path, "last")
    assert result.category == "budget_stop"
    assert result.incomplete is False


def test_clamped_timeout_autopsies_as_timeout(tmp_path, monkeypatch):
    import time

    import requests

    from pedacito.llm import LLMClient, LLMError

    def timed_out(url, **kw):
        raise requests.exceptions.Timeout("read timed out")

    monkeypatch.setattr("pedacito.llm.requests.post", timed_out)
    cfg = _real_cfg(tmp_path, timeout=600.0)
    cfg._task_deadline = time.monotonic() + 30.0
    events.get(tmp_path, cfg).emit("session_start", cmd="ask")
    with pytest.raises(LLMError):
        LLMClient(cfg).chat([{"role": "user", "content": "hi"}], 100, stream=False)
    events.get(tmp_path, cfg).emit("exit", code=1)
    result = autopsy.autopsy(tmp_path, "last")
    assert result.category == "timeout"
    assert result.incomplete is False
    assert 5 <= result.evidence[0]["timeout_s"] <= 30


# --------------------------------------------------------------- inspector --

@pytest.fixture
def fresh_cfg(tmp_path, monkeypatch):
    """A config with a project root, pruning on, and every stale tool unused."""
    monkeypatch.setattr(memory, "tool_unused_tasks", lambda root, action: 999)

    def make(**overrides):
        cfg = Config()
        cfg._root = str(tmp_path)
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    return make


def _active_names(rows) -> list[str]:
    return [row.name for row in rows if row.active]


@pytest.mark.parametrize("task", ["", "fix the failing test", "read https://example.com"])
@pytest.mark.parametrize("overrides", [
    {},
    {"allow_shell": True, "allow_write": True, "backup": True},
    {"allow_write": True, "allow_browser": True, "allow_screenshots": True,
     "allow_serve": True, "allow_lsp": True},
    {"enable_web": True, "allow_subagents": True, "allow_mcp": True},
])
def test_inspector_matches_schema_exactly(fresh_cfg, task, overrides):
    cfg = fresh_cfg(**overrides)
    cfg._task_text = task
    schema_enum = tools.schema_for(cfg)["properties"]["action"]["enum"]
    rows = tools.inspect_tools(cfg, None, task)
    assert _active_names(rows) == schema_enum
    assert cfg._task_text == task


def test_inspector_does_not_leave_task_text_behind(fresh_cfg):
    cfg = fresh_cfg()
    assert not hasattr(cfg, "_task_text")
    tools.inspect_tools(cfg, None, "some task")
    assert not hasattr(cfg, "_task_text")


def test_reasons_are_from_the_contract(fresh_cfg):
    cfg = fresh_cfg(allow_shell=False, allow_serve=False)
    rows = {row.name: row for row in tools.inspect_tools(cfg, None, "")}
    assert rows["read"].active and rows["read"].reason == "active"
    assert rows["run"].reason == "config_disabled" and not rows["run"].active
    assert rows["run"].permission == "allow_shell"
    assert rows["undo"].reason == "config_disabled"
    assert rows["done"].permission == "none"
    assert rows["fetch_url"].reason == "not_relevant"
    assert rows["generate_image"].reason in ("missing_dependency", "config_disabled")
    for row in rows.values():
        assert row.reason in tools.TOOL_REASONS
        assert isinstance(row, tools.ToolAvailability)
        assert dataclasses.is_dataclass(row)
        with pytest.raises(dataclasses.FrozenInstanceError):
            row.active = True  # type: ignore[misc]


def test_pruned_tools_are_reported_and_forced_back(fresh_cfg):
    cfg = fresh_cfg(allow_shell=True)
    rows = {row.name: row for row in tools.inspect_tools(cfg, None, "")}
    assert rows["run"].active is False and rows["run"].reason == "pruned"
    forced = {row.name: row for row in tools.inspect_tools(
        cfg, None, "", force=("run",))}
    assert forced["run"].active is True and forced["run"].reason == "forced"
    cfg._force_tools = ("run",)
    assert "run" in tools.schema_for(cfg)["properties"]["action"]["enum"]


def test_allow_tool_never_overrides_config_or_policy(fresh_cfg):
    cfg = fresh_cfg(allow_serve=False, allow_shell=False)
    forced = {row.name: row for row in tools.inspect_tools(
        cfg, None, "", force=("serve", "run"))}
    assert forced["serve"].active is False
    assert forced["serve"].reason == "config_disabled"
    assert forced["run"].active is False
    assert forced["run"].reason == "config_disabled"
    cfg._force_tools = ("serve", "run")
    enum = tools.schema_for(cfg)["properties"]["action"]["enum"]
    assert "serve" not in enum and "run" not in enum


def test_session_force_resets_when_cleared(fresh_cfg):
    cfg = fresh_cfg(allow_shell=True)
    cfg._force_tools = ("run",)
    assert "run" in tools.schema_for(cfg)["properties"]["action"]["enum"]
    cfg._force_tools = ()
    assert "run" not in tools.schema_for(cfg)["properties"]["action"]["enum"]


def test_permission_table_covers_every_action(fresh_cfg):
    cfg = fresh_cfg(allow_shell=True, allow_write=True, backup=True,
                    allow_serve=True, allow_screenshots=True, allow_browser=True,
                    allow_lsp=True, allow_mcp=True, allow_subagents=True,
                    enable_web=True, allow_fetch=True, use_memory=True)
    names = {row.name for row in tools.inspect_tools(cfg, None, "")}
    assert names == set(tools.ACTION_PERMISSION)


def test_cli_tools_json_and_unknown_allow_tool(tmp_path, capsys):
    assert cli.main(["tools", str(tmp_path), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {"name", "active", "reason", "permission"} <= set(rows[0])
    assert cli.main(["tools", str(tmp_path), "--allow-tool", "nope"]) == 2


def test_cli_ask_and_auto_accept_allow_tool():
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    for name in ("ask", "auto"):
        ns = sub.choices[name].parse_args(
            ["--allow-tool", "run", "--allow-tool", "write", "-t", "task text", "."])
        assert ns.allow_tool == ["run", "write"]


def test_tui_tools_command_is_registered():
    from pedacito import tui
    assert "/tools" in tui._SLASH_COMMANDS
    assert "/tools" in tui.HELP_TEXT and "/tools" in tui.HELP_TEXT_ES


def test_schema_default_is_unchanged_when_no_force(fresh_cfg):
    cfg = fresh_cfg()
    base = tools.schema_for(cfg)
    cfg._force_tools = ()
    assert tools.schema_for(cfg) == base
