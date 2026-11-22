"""execution_mode wiring (Job 8): config resolution, loop dispatch, the
interleaved call floor, the CLI flag, the TUI /mode command and the GUI
activity feed.

The interleaved engine itself is covered by test_interleaved.py; here the
model doubles and fixture repos are reused to prove the WIRING: which engine
run_task picks, that two_phase stays byte-for-byte the default path for small
local models, and that every surface (CLI, TUI, GUI) exposes the knob.
"""
from __future__ import annotations

import http.client
import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import cli  # noqa: E402
from pedacito import config as config_mod  # noqa: E402
from pedacito import loop, review  # noqa: E402
from pedacito import tui as tui_mod  # noqa: E402
from pedacito.config import Config, ConfigError  # noqa: E402
from pedacito.gui import core  # noqa: E402
from pedacito.tui import TuiSession  # noqa: E402

from test_gui import gui as _shared_gui  # noqa: F401,E402  (pytest re-export)
from test_interleaved import (  # noqa: E402
    FIX_FLOW,
    ScriptedTools,
    edit,
    make_agent,
    tree,
)


@pytest.fixture
def gui(_shared_gui):  # noqa: F811  (re-export the fixture imported above)
    return _shared_gui


TASK = "fix the bulk discount threshold"


def _source_tree(root: Path) -> dict:
    """The project's own files: run_task legitimately writes harness state
    (checkpoints, backups, review pages) under .pedacito even for a dry run,
    so the "untouched" assertions compare only the source tree."""
    return {rel: data for rel, data in tree(root).items()
            if not rel.startswith(".pedacito")}


def _ns(**kw):
    """A minimal config-like object for resolve_execution_mode."""
    base = dict(execution_mode="auto", active_tier="", active_profile="",
                model="", base_url="http://localhost:1234/v1", provider="",
                profiles={}, json_mode="schema", _hosted_server=False,
                sources={})
    base.update(kw)
    return SimpleNamespace(**base)


NATIVE_CARD = {"choices": {"tool_mode": "native"},
               "measured": {"tool_choice_native": 0.95}}


# =============================================================== resolution
@pytest.mark.parametrize("mode", ["two_phase", "interleaved"])
def test_explicit_modes_pass_through_unchanged(mode):
    # Even a hosted endpoint cannot override an explicit choice.
    cfg = _ns(execution_mode=mode, base_url="https://api.example.com/v1")
    assert config_mod.resolve_execution_mode(cfg) == mode


@pytest.mark.parametrize("kw", [
    dict(base_url="https://api.deepseek.com/v1"),      # hosted endpoint
    dict(active_tier="hosted"),                        # hosted tier
    dict(json_mode="object"),                          # hosted-only mode
    dict(_hosted_server=True),                         # detected hosted
    dict(model="qwen35b-a3b"),                         # local-MoE pattern
    dict(provider="strata"),                           # local-MoE runtime
    dict(active_profile="local-moe"),                  # profile label
])
def test_auto_picks_interleaved_for_hosted_and_local_moe(kw):
    assert config_mod.resolve_execution_mode(_ns(**kw)) == "interleaved"


@pytest.mark.parametrize("model", [
    "qwen2.5-coder-1.5b",       # tiny
    "llama-3.2-3b",             # small
    "qwen2.5-coder-7b",         # small
    "some-14b-model",           # medium
    "llama-3.3-70b",            # large but UNCALIBRATED (no card on disk)
    "mystery-model",            # unclassifiable
    "",                         # nothing selected yet
])
def test_auto_stays_two_phase_for_small_and_uncalibrated_local(model):
    assert config_mod.resolve_execution_mode(_ns(model=model)) == "two_phase"


def test_auto_interleaves_large_local_only_with_a_native_card():
    large = dict(model="llama-3.3-70b")
    # calibrated + native choice + measured score above the bar
    assert config_mod.resolve_execution_mode(
        _ns(_card=NATIVE_CARD, **large)) == "interleaved"
    # an explicit large tier works the same way
    assert config_mod.resolve_execution_mode(
        _ns(active_tier="large", _card=NATIVE_CARD)) == "interleaved"
    # no measured score: the card's native choice is trusted
    no_score = {"choices": {"tool_mode": "native"}, "measured": {}}
    assert config_mod.resolve_execution_mode(
        _ns(_card=no_score, **large)) == "interleaved"
    # measured score below 0.85: the native channel is not reliable enough
    low = {"choices": {"tool_mode": "native"},
           "measured": {"tool_choice_native": 0.5}}
    assert config_mod.resolve_execution_mode(
        _ns(_card=low, **large)) == "two_phase"
    # the card chose the grammar channel
    grammar = {"choices": {"tool_mode": "grammar"}, "measured": {}}
    assert config_mod.resolve_execution_mode(
        _ns(_card=grammar, **large)) == "two_phase"
    # a stale card must be recalibrated before it can steer the engine
    stale = dict(NATIVE_CARD, _needs_recalibrate=True)
    assert config_mod.resolve_execution_mode(
        _ns(_card=stale, **large)) == "two_phase"


def test_unknown_mode_value_falls_back_to_two_phase():
    assert config_mod.resolve_execution_mode(
        _ns(execution_mode="bogus")) == "two_phase"
    assert config_mod.resolve_execution_mode(_ns(execution_mode="")) == "two_phase"


def test_config_field_default_validation_and_enum():
    cfg = Config()
    assert cfg.execution_mode == "two_phase"
    assert config_mod._ENUM_FIELDS["execution_mode"] == (
        "two_phase", "interleaved", "auto")
    assert config_mod._coerce("execution_mode", " INTERLEAVED ") == "interleaved"
    with pytest.raises(ConfigError) as exc:
        config_mod._coerce("execution_mode", "bogus")
    assert "two_phase" in str(exc.value)


# =============================================================== dispatch
def _sentinel_result(agent):
    return loop.TaskResult([], "done", False, [], agent.index)


@pytest.mark.parametrize("mode,model", [
    ("two_phase", ""),                    # the untouched default
    ("two_phase", "qwen2.5-coder-7b"),
    ("auto", "qwen2.5-coder-7b"),         # auto resolves small -> two_phase
    ("auto", "llama-3.3-70b"),            # large but uncalibrated
])
def test_two_phase_path_is_untouched_for_small_local_models(
        tmp_path, monkeypatch, mode, model):
    client = ScriptedTools(FIX_FLOW)
    root, agent = make_agent(tmp_path, client, execution_mode=mode, model=model)
    seen = {"ladder": 0}

    def fake_ladder(*a, **k):
        seen["ladder"] += 1
        return _sentinel_result(agent)

    def no_interleaved(*a, **k):
        raise AssertionError("interleaved engine ran for a two_phase config")

    monkeypatch.setattr(loop, "_run_ladder", fake_ladder)
    monkeypatch.setattr(loop, "_run_interleaved_task", no_interleaved)
    tr = loop.run_task(root, agent.cfg, client, agent, agent.index, TASK,
                       apply=True)
    assert seen["ladder"] == 1
    assert tr.reply == "done"


def test_explicit_interleaved_dispatches_even_for_a_small_model(
        tmp_path, monkeypatch):
    client = ScriptedTools(FIX_FLOW)
    root, agent = make_agent(tmp_path, client, execution_mode="interleaved",
                             model="qwen2.5-coder-7b")
    seen = {}

    def fake_interleaved(root_, cfg, client_, agent_, idx, task, **kw):
        seen.update(kw)
        return _sentinel_result(agent)

    def no_ladder(*a, **k):
        raise AssertionError("two_phase ladder ran for an interleaved config")

    monkeypatch.setattr(loop, "_run_interleaved_task", fake_interleaved)
    monkeypatch.setattr(loop, "_run_ladder", no_ladder)
    loop.run_task(root, agent.cfg, client, agent, agent.index, TASK, apply=True)
    assert seen["apply"] is True and seen["review"] is False


def test_interleaved_apply_end_to_end_through_run_task(tmp_path):
    """auto + local-MoE dispatches the real engine: the edit lands, the test
    verdict and the synthesized step log ride the TaskResult contract."""
    client = ScriptedTools(FIX_FLOW)
    root, agent = make_agent(tmp_path, client, execution_mode="auto",
                             model="qwen35b-a3b")
    tr = loop.run_task(root, agent.cfg, client, agent, agent.index, TASK,
                       apply=True)
    assert tr.applied
    assert tr.changed == ["pricing.py"]
    text = (root / "pricing.py").read_text(encoding="utf-8")
    assert "if quantity >= 10:" in text and "return 0.2" not in text
    assert tr.tests_ok is True
    assert tr.reply == "bulk_discount now applies from 10 units"
    # the familiar numbered step lines, synthesized from the transcript
    assert tr.steps[0] == "1. read pricing.py"
    assert any(ln.startswith("3. run_test") for ln in tr.steps)
    assert tr.steps[-1].startswith("7. finish")
    # live evidence: run_test observations were seen
    assert tr.live_ok is True and "run_test" in (tr.live_kinds or ())
    # the engine ran exactly one conversation (7 scripted turns)
    assert len(client.requests) == 7


def test_interleaved_dry_run_never_touches_disk(tmp_path):
    client = ScriptedTools(FIX_FLOW)
    root, agent = make_agent(tmp_path, client, execution_mode="interleaved")
    before = _source_tree(root)
    tr = loop.run_task(root, agent.cfg, client, agent, agent.index, TASK,
                       apply=False)
    assert not tr.applied and tr.changed == []
    assert _source_tree(root) == before
    assert tr.results and "(dry run)" in tr.results[0].message
    assert tr.tests_ok is True          # tests ran against the staged overlay


def test_interleaved_review_hands_off_to_the_same_signoff_page(
        tmp_path, monkeypatch):
    client = ScriptedTools(FIX_FLOW)
    root, agent = make_agent(tmp_path, client, execution_mode="interleaved")
    before = _source_tree(root)
    seen = {}

    def fake_run(page, nonce, writer, **kw):
        # The tree must still be pristine when the review page is served:
        # the engine staged everything in the overlay.
        seen["at_signoff"] = _source_tree(root)
        ok, msg = writer()
        assert ok, msg
        return "applied", msg

    monkeypatch.setattr(review, "run", fake_run)
    tr = loop.run_task(root, agent.cfg, client, agent, agent.index, TASK,
                       apply=True, review=True)
    assert seen["at_signoff"] == before
    assert tr.applied
    assert "if quantity >= 10:" in (root / "pricing.py").read_text(encoding="utf-8")
    assert all("applied via review" in r.message for r in tr.results if r.ok)


def test_interleaved_review_rejection_writes_nothing(tmp_path, monkeypatch):
    client = ScriptedTools(FIX_FLOW)
    root, agent = make_agent(tmp_path, client, execution_mode="interleaved")
    before = _source_tree(root)

    def fake_run(page, nonce, writer, **kw):
        return "rejected", "the user declined"

    monkeypatch.setattr(review, "run", fake_run)
    tr = loop.run_task(root, agent.cfg, client, agent, agent.index, TASK,
                       apply=True, review=True)
    assert not tr.applied
    assert _source_tree(root) == before
    assert all("not written (rejected)" in r.message for r in tr.results if r.ok)


def test_interleaved_trajectory_is_recorded(tmp_path):
    client = ScriptedTools(FIX_FLOW)
    root, agent = make_agent(tmp_path, client, execution_mode="interleaved",
                             record_trajectories=True)
    tr = loop.run_task(root, agent.cfg, client, agent, agent.index, TASK,
                       apply=True)
    assert tr.applied
    files = list((root / ".pedacito" / "trajectories").glob("*.json"))
    assert files, "no trajectory written for the interleaved run"
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    roles = [m.get("role") for m in payload["messages"]]
    assert "tool" in roles and "assistant" in roles
    assert payload["metadata"]["verify_passed"] is True


# =============================================================== call floor
SHORT_FLOW = [
    [("read", {"file": "pricing.py"})],
    [edit("pricing.py", "    if quantity > 10:", "    if quantity >= 10:")],
    [("finish", {"summary": "threshold fixed"})],
]


def _cap_probe_flow(seen):
    def probe(messages, client):
        seen["cap"] = client.cfg.max_total_model_calls
        seen["source"] = (client.cfg.sources or {}).get("max_total_model_calls")
        return SHORT_FLOW[0]
    return [probe] + SHORT_FLOW[1:]


def test_default_call_cap_gets_the_interleaved_floor_and_is_restored(tmp_path):
    seen = {}
    client = ScriptedTools(_cap_probe_flow(seen))
    root, agent = make_agent(tmp_path, client, execution_mode="interleaved")
    client.cfg = agent.cfg
    assert agent.cfg.max_total_model_calls == 30      # the gather-era default
    tr = loop.run_task(root, agent.cfg, client, agent, agent.index, TASK,
                       apply=True)
    assert tr.applied
    assert seen["cap"] == loop._INTERLEAVED_CALL_FLOOR
    assert seen["source"] == "interleaved default"
    # restored afterwards: the next two_phase task keeps its own budget
    assert agent.cfg.max_total_model_calls == 30
    assert agent.cfg.sources.get("max_total_model_calls") in (None, "default")


def test_preset_call_cap_is_also_raised_to_the_floor(tmp_path):
    seen = {}
    client = ScriptedTools(_cap_probe_flow(seen))
    root, agent = make_agent(tmp_path, client, execution_mode="interleaved")
    client.cfg = agent.cfg
    agent.cfg.sources["max_total_model_calls"] = "preset tier"
    loop.run_task(root, agent.cfg, client, agent, agent.index, TASK, apply=True)
    assert seen["cap"] == loop._INTERLEAVED_CALL_FLOOR
    assert agent.cfg.max_total_model_calls == 30
    assert agent.cfg.sources["max_total_model_calls"] == "preset tier"


def test_explicit_call_cap_is_strictly_respected(tmp_path):
    seen = {}
    client = ScriptedTools(_cap_probe_flow(seen))
    root, agent = make_agent(tmp_path, client, execution_mode="interleaved",
                             max_total_model_calls=50)
    client.cfg = agent.cfg
    agent.cfg.sources["max_total_model_calls"] = "config file"
    tr = loop.run_task(root, agent.cfg, client, agent, agent.index, TASK,
                       apply=True)
    assert tr.applied
    assert seen["cap"] == 50                          # never raised
    assert seen["source"] == "config file"
    assert agent.cfg.max_total_model_calls == 50
    assert agent.cfg.sources["max_total_model_calls"] == "config file"


# =============================================================== CLI
@pytest.mark.parametrize("cmd,base", [
    ("ask", ["ask", ".", "-t", "x"]),
    ("auto", ["auto", ".", "-t", "x"]),
    ("new", ["new", "proj"]),
    ("eval", ["eval"]),
])
def test_cli_execution_mode_flag_on_every_task_command(cmd, base):
    parser = cli.build_parser()
    args = parser.parse_args(base + ["--execution-mode", "interleaved"])
    assert args.execution_mode == "interleaved"
    assert parser.parse_args(base).execution_mode is None


def test_cli_execution_mode_rejects_unknown_values():
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["ask", ".", "-t", "x", "--execution-mode", "bogus"])
    assert exc.value.code == 2


def test_cli_config_applies_the_execution_mode_override(tmp_path):
    cfg_file = tmp_path / "user.toml"
    cfg_file.write_text(
        'vram_gb = 24\n'
        'model = "base-model"\n'
        'base_url = "http://127.0.0.1:9/v1"\n'
        'execution_mode = "two_phase"\n',
        encoding="utf-8")
    args = SimpleNamespace(
        cmd="ask", config=str(cfg_file), profile=None, root=str(tmp_path),
        paths=[str(tmp_path)], execution_mode="interleaved")
    cfg = cli._config(args)
    assert cfg.execution_mode == "interleaved"
    assert cfg.sources["execution_mode"] == "--execution-mode"
    # without the flag the file value stands
    args.execution_mode = None
    cfg = cli._config(args)
    assert cfg.execution_mode == "two_phase"


# =============================================================== TUI
def _tui_cfg(**overrides):
    cfg = Config()
    cfg.model = "qwen3-coder"
    cfg.active_profile = "default"
    cfg.sources = {}
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_tui_mode_command_sets_reports_aliases_and_rejects(tmp_path):
    cfg = _tui_cfg()
    session = TuiSession(tmp_path, cfg, headless=True, stdout=io.StringIO())
    text = lambda: "\n".join(line for _kind, line in session.transcript)
    try:
        session.handle_line("/mode")
        assert "Execution mode: two_phase (resolved: two_phase)" in text()

        session.handle_line("/mode interleaved")
        assert cfg.execution_mode == "interleaved"
        assert cfg.sources["execution_mode"] == "tui /mode"
        assert "Execution mode: interleaved (resolved: interleaved)" in text()

        # the alias and the dash spelling both work
        session.handle_line("/exec-mode auto")
        assert cfg.execution_mode == "auto"
        session.handle_line("/mode two-phase")
        assert cfg.execution_mode == "two_phase"

        before = cfg.execution_mode
        session.handle_line("/mode banana")
        assert cfg.execution_mode == before
        assert "Usage: /mode two_phase|interleaved|auto" in text()

        # /status shows the configured and the resolved mode
        session.handle_line("/status")
        assert "mode    : two_phase \u2192 two_phase" in text()
    finally:
        session.close()


def test_tui_mode_is_registered_for_help_and_completion():
    assert "/mode" in tui_mod._SLASH_COMMANDS
    assert "/exec-mode" in tui_mod._SLASH_COMMANDS
    assert tui_mod._TAB_PREFIX["/mod"] == "/mode "
    assert "/mode [MODE]" in tui_mod.help_text("en")
    assert "/mode [MODO]" in tui_mod.help_text("es")


# =============================================================== GUI
def test_gui_state_note_event_builds_the_interleaved_feed():
    state = core.State()
    try:
        state.note_event("interleaved_turn",
                         {"turn": 1, "calls": ["read"], "tokens": 40,
                          "mode": "native", "text": False})
        state.note_event("tool_call",
                         {"turn": 1, "name": "read", "id": "c1",
                          "args": {"file": "pricing.py"}})
        state.note_event("tool_result",
                         {"turn": 1, "name": "read", "id": "c1", "ok": True,
                          "chars": 120, "preview": "1: def ...",
                          "gate_errors": 0})
        state.note_event("edit_staged",
                         {"turn": 2, "file": "pricing.py", "added": 1,
                          "removed": 1, "applied": True, "diff": "@@ ..."})
        state.note_event("finish",
                         {"ok": True, "stop_reason": "finish",
                          "summary": "done", "changed_files": ["pricing.py"],
                          "turns": 2, "tool_calls": 2, "applied": True})
        events, _resync = state.bus.wait(0, 0.01)
        feed = [e for e in events if e["type"] == "interleaved"]
        assert [e["kind"] for e in feed] == [
            "interleaved_turn", "tool_call", "tool_result", "edit_staged",
            "finish"]
        assert feed[0]["calls"] == ["read"] and feed[0]["text"] is False
        assert feed[2]["gate_errors"] == 0
        eng = state.engine()
        assert [r["kind"] for r in eng["interleaved"]] == [
            "interleaved_turn", "tool_call", "tool_result", "edit_staged",
            "finish"]
        assert eng["phase"] == "edit"        # edit_staged moved the phase
        # a fresh run clears the feed
        job = state.begin_job("task", "t")
        assert state.engine()["interleaved"] == []
        state.end_job(job)
    finally:
        state.bus.close()


def test_gui_state_and_sse_expose_execution_mode(gui):
    # No project open: the global default config reports two_phase.
    status, state = gui.get("/api/state")
    assert status == 200
    assert state["execution_mode"] == "two_phase"
    assert state["resolved_execution_mode"] == "two_phase"
    assert state["engine"]["interleaved"] == []

    # An open project with an explicit interleaved config is reflected live.
    cfg = Config()
    cfg.execution_mode = "interleaved"
    with core.STATE.lock:
        core.STATE.cfg = cfg
    state = gui.get("/api/state")[1]
    assert state["execution_mode"] == "interleaved"
    assert state["resolved_execution_mode"] == "interleaved"

    # auto resolves by model class: a small local model stays two_phase.
    cfg.execution_mode = "auto"
    cfg.model = "qwen2.5-coder-7b"
    state = gui.get("/api/state")[1]
    assert state["execution_mode"] == "auto"
    assert state["resolved_execution_mode"] == "two_phase"

    # SSE: an interleaved event reaches the stream without a state poll.
    conn = http.client.HTTPConnection("127.0.0.1", gui.port, timeout=10)
    conn.request("GET", "/events?since=" + str(core.STATE.bus.seq))
    resp = conn.getresponse()
    assert resp.status == 200
    core.STATE.note_event("tool_call",
                          {"turn": 1, "name": "read", "id": "c9",
                           "args": {"file": "app.py"}})
    seen = []
    deadline = time.time() + 5
    while time.time() < deadline:
        line = resp.fp.readline().decode()
        if line.startswith("data:"):
            e = json.loads(line[5:])
            if e["type"] == "interleaved":
                seen.append(e)
                break
    conn.close()
    assert seen and seen[0]["name"] == "read" and seen[0]["id"] == "c9"
    # and the authoritative poll carries the same record
    feed = gui.get("/api/state")[1]["engine"]["interleaved"]
    assert any(r["kind"] == "tool_call" and r["id"] == "c9" for r in feed)