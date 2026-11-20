"""Split-stage hybrid routing and the 2-model sandbox arena (v0.3.0).

Hybrid: a strong ARCHITECT plans (Stage 1) and a fast EDITOR implements the
blueprint (Stage 2). These tests lock the stage handoff, the blueprint riding
ahead of the editor's context, the VRAM eviction between local servers, the
combined + per-stage telemetry, and the graceful editor-only fallback.

Arena: one prompt runs on two contestants, each in an isolated sandbox copy of
the workspace. The real project is never written until a winner is applied
between two checkpoints. These tests lock sandbox isolation, the scorecard,
persistence, the apply/undo contract, baseline-deletion safety, and the three
GUI arena endpoints against the fake OpenAI-compatible server.
"""
# The `gui` / `fake_port` / `project` fixtures are test_gui's, imported below.
# ruff: noqa: F811
from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pedacito import arena as arena_mod  # noqa: E402
from pedacito import checkpoints as cp_mod  # noqa: E402
from pedacito import events as events_mod  # noqa: E402
from pedacito import llm as llm_mod  # noqa: E402
from pedacito import loop, roles as roles_mod  # noqa: E402
from pedacito import servers as servers_mod  # noqa: E402
from pedacito import tui as tui_mod  # noqa: E402
from pedacito.config import Config  # noqa: E402
from pedacito.llm import LLMError  # noqa: E402
from test_agent_context import Scripted, _agent  # noqa: E402
from test_gui import APP, EDIT_REPLY, fake_port, gui, project  # noqa: F401,E402
from test_tui import FakeClient  # noqa: E402

# The fast, deterministic single-model path: no gates, no memory, no backups.
FAST = dict(verify_tests=False, post_edit_check=False, static_gate=False,
            interface_lock=False, frontend_contracts=False, verify_dom=False,
            use_memory=False, backup=False, log_sessions=False,
            refresh_summaries=0, stuck_recovery=False, verify_answer="off",
            log_events=False)

BLUEPRINT_REPLY = (
    "Let me reason about this.\n"
    "HYBRID_BLUEPRINT\n"
    "1. ROOT CAUSE: greet prints instead of returning the text.\n"
    "2. TARGET FILES: app.py -- the only definition of greet.\n"
    "3. EXACT SIGNATURES & CONTRACTS: greet(name: str) -> str.\n"
    "4. EDGE CASES: keep the trailing greet(\"x\") call working.\n"
    "END_BLUEPRINT\n"
)


class Architect(Scripted):
    """Stage-1 double: the LLMClient surface without a server. Warmup usage is
    counted BEFORE the plan snapshot so the per-stage delta must exclude it."""

    def __init__(self, cfg, replies):
        super().__init__(replies=list(replies))
        self.cfg = cfg
        self.site = ""
        self.on_call = None

    def check(self, warn=True):
        return None

    def warmup(self):
        self.stats["calls"] += 1
        self.stats["prompt_tokens"] += 10
        self.stats["completion_tokens"] += 1

    def chat(self, messages, max_tokens, stream=False, **kw):
        self.stats["calls"] += 1
        self.stats["prompt_tokens"] += 100
        self.stats["completion_tokens"] += 50
        return super().chat(messages, max_tokens, stream=stream, **kw)


class Contestant(Scripted):
    """An editor / arena contestant double that counts answer-phase usage."""

    def __init__(self, cfg, replies=(EDIT_REPLY,), steps=()):
        super().__init__(steps=steps, replies=list(replies))
        self.cfg = cfg
        self.site = ""
        self.on_call = None

    def check(self, warn=True):
        return None

    def warmup(self):
        return None

    def chat(self, messages, max_tokens, stream=False, **kw):
        self.stats["calls"] += 1
        self.stats["prompt_tokens"] += 200
        self.stats["completion_tokens"] += 80
        return super().chat(messages, max_tokens, stream=stream, **kw)


@pytest.fixture(autouse=True)
def _fresh_arena_memory(monkeypatch):
    """The arena keeps an in-memory copy beside the on-disk record; isolate it
    per test so a stale comparison never leaks across roots."""
    monkeypatch.setattr(arena_mod, "_LAST", None)


def _listen(tmp_path, cfg):
    """Attach a recorder to the root's event log (removed by the caller)."""
    cfg.log_events = True
    elog = events_mod.get(tmp_path, cfg)
    seen: list[tuple[str, dict]] = []
    elog.listeners.append(lambda kind, fields: seen.append((kind, fields)))
    return elog, seen


def _hybrid_agent(tmp_path, *, architect_provider="", replies=(BLUEPRINT_REPLY,)):
    editor = Contestant(Config(), replies=[EDIT_REPLY])
    agent = _agent(tmp_path, editor, {"app.py": APP}, **FAST)
    cfg = agent.cfg
    cfg.model = "editor-fast"
    cfg.base_url = "http://127.0.0.1:8001/v1"
    cfg.active_profile = "default"
    cfg.hybrid_routing = True
    cfg.architect_provider = architect_provider
    cfg.architect_model = "architect-big"
    made: list[Architect] = []

    def factory(new_cfg):
        client = Architect(new_cfg, replies)
        made.append(client)
        return client

    return agent, editor, cfg, made, factory


# --------------------------------------------------------------- hybrid -----

def test_hybrid_active_requires_a_distinct_armed_model():
    cfg = Config()
    cfg.model = "editor-fast"
    assert loop.hybrid_active(cfg) is False              # toggle off
    cfg.hybrid_routing = True
    assert loop.hybrid_active(cfg) is False              # no architect model
    cfg.architect_model = "editor-fast"
    assert loop.hybrid_active(cfg) is False              # same brain twice
    cfg.architect_model = "architect-big"
    assert loop.hybrid_active(cfg) is True


def test_extract_blueprint_takes_the_body_or_the_whole_reply():
    assert loop._extract_blueprint(BLUEPRINT_REPLY).startswith("1. ROOT CAUSE")
    assert "END_BLUEPRINT" not in loop._extract_blueprint(BLUEPRINT_REPLY)
    # A small architect that forgot the fence still planned: take it whole.
    assert loop._extract_blueprint("just a plan") == "just a plan"
    assert loop._extract_blueprint("   ") == ""


def test_hybrid_stage1_plans_then_editor_implements_the_blueprint(
        tmp_path, monkeypatch):
    agent, editor, cfg, made, factory = _hybrid_agent(tmp_path)
    monkeypatch.setattr(llm_mod, "LLMClient", factory)
    cfg.answer_mode = "continue"
    elog, seen = _listen(tmp_path, cfg)
    try:
        tr = loop.run_task(tmp_path, cfg, editor, agent, agent.index,
                           "make greet return its greeting", apply=True)
    finally:
        elog.listeners.clear()

    # Stage 1 ran on the architect model with the planning system prompt.
    assert len(made) == 1 and made[0].cfg.model == "architect-big"
    arch_msgs = made[0].chat_calls[0]
    assert arch_msgs[0]["content"] == loop._HYBRID_SYSTEM
    assert "TASK:" in arch_msgs[1]["content"]
    assert "make greet return its greeting" in arch_msgs[1]["content"]
    assert "HYBRID_BLUEPRINT" in arch_msgs[1]["content"]

    # Stage 2: the blueprint rides ahead of the editor's gathered context,
    # markers stripped, and the edit lands.
    editor_prompt = editor.chat_calls[0][1]["content"]
    assert "HYBRID_BLUEPRINT (from the architect model" in editor_prompt
    assert "TARGET FILES: app.py" in editor_prompt
    assert "END_BLUEPRINT" not in editor_prompt
    assert tr.applied
    assert 'return f"hello {name}"' in (tmp_path / "app.py").read_text(
        encoding="utf-8")

    # The exact stage labels, in order, plus the closing breakdown event.
    stages = [f for k, f in seen if k == "hybrid_stage"]
    assert [f["stage"] for f in stages] == [1, 2]
    assert stages[0]["label"] == "Hybrid Stage 1 (Architect): architect-big"
    assert stages[1]["label"] == "Hybrid Stage 2 (Editor): editor-fast"
    assert any(k == "hybrid" for k, _ in seen)

    # Combined telemetry = editor + architect; warmup usage stays excluded.
    h = tr.stats["hybrid"]
    a, e = h["architect"], h["editor"]
    assert a["model"] == "architect-big" and e["model"] == "editor-fast"
    assert a["calls"] == 1 and a["prompt_tokens"] == 100
    assert a["completion_tokens"] == 50
    for k in ("calls", "prompt_tokens", "completion_tokens"):
        assert tr.stats[k] == a[k] + e[k]
    assert tr.stats["completion_tokens"] > e["completion_tokens"]

    # Session state is restored: the forced rebuild and the blueprint are gone.
    assert cfg.answer_mode == "continue"
    assert cfg._hybrid_blueprint == ""


def test_hybrid_architect_messages_enter_the_trajectory(
        tmp_path, monkeypatch):
    """The editor's trajectory must include the architect's planning exchange,
    without counting its tokens twice or changing the run summary."""
    agent, editor, cfg, made, factory = _hybrid_agent(tmp_path)
    monkeypatch.setattr(llm_mod, "LLMClient", factory)
    cfg.record_trajectories = True

    tr = loop.run_task(tmp_path, cfg, editor, agent, agent.index,
                       "make greet return its greeting", apply=True)

    files = sorted((tmp_path / ".pedacito" / "trajectories").glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    messages = data["messages"]
    assert any(m.get("role") == "system"
               and m.get("content") == loop._HYBRID_SYSTEM
               for m in messages)
    assert any(m.get("role") == "assistant"
               and "HYBRID_BLUEPRINT" in str(m.get("content") or "")
               and "ROOT CAUSE" in str(m.get("content") or "")
               for m in messages)
    # One architect call, counted once: the transcript capture is passive.
    assert tr.stats["hybrid"]["architect"]["calls"] == 1
    assert data["metadata"]["tokens"]["calls"] == tr.stats["calls"]
    assert data["metadata"]["tokens"]["completion_tokens"] == \
        tr.stats["completion_tokens"]


def test_hybrid_evicts_between_local_servers_at_each_handoff(
        tmp_path, monkeypatch):
    agent, editor, cfg, made, factory = _hybrid_agent(
        tmp_path, architect_provider="http://127.0.0.1:8002/v1")
    monkeypatch.setattr(llm_mod, "LLMClient", factory)
    evicted: list[str] = []
    monkeypatch.setattr(
        servers_mod, "evict_other_local_servers",
        lambda target, c, **kw: evicted.append(servers_mod.normalize_url(target))
        or {"ok": True, "evicted": []})
    tr = loop.run_task(tmp_path, cfg, editor, agent, agent.index,
                       "make greet return its greeting", apply=True)
    assert tr.applied
    # editor -> architect before Stage 1, then architect -> editor before 2.
    assert evicted == ["http://127.0.0.1:8002", "http://127.0.0.1:8001"]


def test_hybrid_evict_only_between_different_local_servers(monkeypatch):
    cfg = Config()
    cfg.auto_evict_servers = True
    calls: list[str] = []
    monkeypatch.setattr(
        servers_mod, "evict_other_local_servers",
        lambda target, c, **kw: calls.append(target) or {"ok": True})
    loop._hybrid_evict(cfg, "http://127.0.0.1:8001/v1", "http://127.0.0.1:8002/v1")
    assert calls == ["http://127.0.0.1:8002/v1"]
    calls.clear()
    loop._hybrid_evict(cfg, "http://127.0.0.1:8001/v1", "http://127.0.0.1:8001/v1")
    loop._hybrid_evict(cfg, "http://127.0.0.1:8001/v1", "https://api.example.com/v1")
    loop._hybrid_evict(cfg, "https://api.example.com/v1", "http://127.0.0.1:8002/v1")
    assert calls == []                                   # same / hosted: coexist
    cfg.auto_evict_servers = False
    loop._hybrid_evict(cfg, "http://127.0.0.1:8001/v1", "http://127.0.0.1:8002/v1")
    assert calls == []                                   # the switch is off


def test_hybrid_unreachable_architect_falls_back_to_the_editor(
        tmp_path, monkeypatch, capsys):
    agent, editor, cfg, _made, _factory = _hybrid_agent(tmp_path)
    calls: list[str] = []

    def dead(new_cfg):
        calls.append(new_cfg.model)
        raise LLMError("connection refused")

    monkeypatch.setattr(llm_mod, "LLMClient", dead)
    elog, seen = _listen(tmp_path, cfg)
    try:
        tr = loop.run_task(tmp_path, cfg, editor, agent, agent.index,
                           "make greet return its greeting", apply=True)
    finally:
        elog.listeners.clear()
    assert tr.applied                                    # editor ran alone
    assert 'return f"hello {name}"' in (tmp_path / "app.py").read_text(
        encoding="utf-8")
    assert not [f for k, f in seen if k == "hybrid_stage"]
    assert "HYBRID_BLUEPRINT (from the architect model" not in \
        editor.chat_calls[0][1]["content"]
    assert "unavailable" in capsys.readouterr().err
    # The dead verdict is cached: a second ask never retries the server.
    n = len(calls)
    assert loop._architect_client(cfg, editor) is None
    assert len(calls) == n


def test_hybrid_empty_blueprint_falls_back_to_the_editor(tmp_path, monkeypatch):
    agent, editor, cfg, made, factory = _hybrid_agent(tmp_path, replies=("   ",))
    monkeypatch.setattr(llm_mod, "LLMClient", factory)
    elog, seen = _listen(tmp_path, cfg)
    try:
        tr = loop.run_task(tmp_path, cfg, editor, agent, agent.index,
                           "make greet return its greeting", apply=True)
    finally:
        elog.listeners.clear()
    assert tr.applied and len(made) == 1
    # Stage 1 was attempted (its event fires before the plan is read), but an
    # empty blueprint means no Stage 2 handoff and no combined breakdown.
    assert not [f for k, f in seen
                if k == "hybrid_stage" and f.get("stage") == 2]
    assert not [f for k, f in seen if k == "hybrid"]
    assert "HYBRID_BLUEPRINT (from the architect model" not in \
        editor.chat_calls[0][1]["content"]


def test_resolve_model_config_shared_by_hybrid_and_arena():
    cfg = Config()
    cfg.model = "editor-fast"
    cfg.base_url = "http://127.0.0.1:8001/v1"
    cfg.profiles = {"coder": {"model": "big-model",
                              "base_url": "http://127.0.0.1:9001/v1"}}
    assert roles_mod.resolve_model_config(cfg, "", "") is None
    bare = roles_mod.resolve_model_config(cfg, "", "tiny")
    assert bare.model == "tiny" and bare.base_url == cfg.base_url
    prof = roles_mod.resolve_model_config(cfg, "coder", "")
    assert prof is None                                  # profile, no model id
    prof = roles_mod.resolve_model_config(cfg, "coder", "big-model")
    assert prof.model == "big-model"
    assert prof.base_url == "http://127.0.0.1:9001/v1"
    url = roles_mod.resolve_model_config(cfg, "http://127.0.0.1:9002/v1", "m")
    assert url.base_url == "http://127.0.0.1:9002/v1" and url.model == "m"


# ---------------------------------------------------------------- arena -----

def _arena_cfg(tmp_path, **extra):
    cfg = Config()
    for k, v in FAST.items():
        setattr(cfg, k, v)
    cfg.base_url = "http://127.0.0.1:8001/v1"
    cfg.model = "editor-fast"
    cfg.active_profile = "default"
    for k, v in extra.items():
        setattr(cfg, k, v)
    return cfg


def _workspace(tmp_path, extra=None):
    (tmp_path / "app.py").write_text(APP, encoding="utf-8")
    for rel, text in (extra or {}).items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


def _patch_contestants(monkeypatch, steps_by_model=None, fail_models=(),
                       cancel_models=()):
    steps_by_model = steps_by_model or {}
    made: list[Contestant] = []

    def factory(new_cfg):
        model = new_cfg.model
        if model in cancel_models:
            raise llm_mod.CancelledError("stopped by user")
        if model in fail_models:
            raise LLMError(f"{model} server refused the connection")
        client = Contestant(new_cfg, replies=[EDIT_REPLY],
                            steps=steps_by_model.get(model, ()))
        made.append(client)
        return client

    monkeypatch.setattr(llm_mod, "LLMClient", factory)
    return made


def test_arena_runs_in_sandboxes_and_never_touches_the_workspace(
        tmp_path, monkeypatch):
    _workspace(tmp_path, {
        "extra.py": "keep me\n",
        ".git/config": "[core]\n",
        "node_modules/pkg/index.js": "module.exports = 1\n",
        ".venv/lib/x.py": "x = 1\n",
        "dist/out.js": "compiled\n",
        "build/out.o": "obj\n",
    })
    cfg = _arena_cfg(tmp_path)
    _patch_contestants(monkeypatch)
    events: list[tuple[str, dict]] = []
    state = arena_mod.run_arena_comparison(
        tmp_path, "make greet return its greeting", "big-model", "small-model",
        cfg, on_event=lambda kind, fields: events.append((kind, fields)))

    # The real project is byte-for-byte untouched.
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == APP
    assert (tmp_path / "extra.py").read_text(encoding="utf-8") == "keep me\n"

    assert state["arena_id"].startswith("arena_")
    assert state["winner"] is None
    for side, model in (("a", "big-model"), ("b", "small-model")):
        rec = state["contestants"][side]
        assert rec["model"] == model and rec["ok"] is True
        assert rec["files_changed"] == ["app.py"]
        assert "-    print(\"hello\", name)" in rec["diff"]
        assert "+    return f\"hello {name}\"" in rec["diff"]
        assert rec["summary"] == "Make greet return the text."
        assert rec["elapsed_sec"] >= 0 and rec["tok_per_sec"] >= 0
        assert rec["total_tokens"] == (rec["stats"]["prompt_tokens"]
                                       + rec["stats"]["completion_tokens"])
        assert rec["stats"]["completion_tokens"] == 80
        sandbox = Path(state["sandboxes"][side])
        assert 'return f"hello {name}"' in (sandbox / "app.py").read_text(
            encoding="utf-8")
        # The snapshot walker's exclusions hold inside the sandbox too.
        for junk in (".git", "node_modules", ".venv", "dist", "build"):
            assert not (sandbox / junk).exists()

    # Only source files are baselined; the junk trees never are.
    assert "app.py" in state["baseline_files"]
    assert "extra.py" in state["baseline_files"]
    assert not any(f.startswith((".git/", "node_modules/", "dist/"))
                   for f in state["baseline_files"])

    # Persisted beside the in-memory state, and the stage events fired.
    on_disk = arena_mod.last_arena(tmp_path)
    assert on_disk["arena_id"] == state["arena_id"]
    assert (tmp_path / ".pedacito" / "arena_last.json").is_file()
    phases = [f.get("phase") for k, f in events if k == "arena_stage"]
    assert "arena_start" in phases and "arena_done" in phases
    assert phases.count("start") == 2 and phases.count("done") == 2


def test_arena_apply_writes_between_checkpoints_and_is_undoable(
        tmp_path, monkeypatch):
    _workspace(tmp_path)
    cfg = _arena_cfg(tmp_path)
    _patch_contestants(monkeypatch)
    state = arena_mod.run_arena_comparison(
        tmp_path, "make greet return its greeting", "big-model", "small-model",
        cfg)
    arena_id = state["arena_id"]
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == APP

    out = arena_mod.apply_arena_winner(tmp_path, arena_id, "a")
    assert out["ok"] and out["winner"] == "a" and out["model"] == "big-model"
    assert out["files_changed"] == ["app.py"]
    assert 'return f"hello {name}"' in (tmp_path / "app.py").read_text(
        encoding="utf-8")

    labels = [c["label"] for c in cp_mod.list_checkpoints(tmp_path)]
    assert f"Arena pre-apply ({arena_id})" in labels
    assert "Arena winner: big-model" in labels
    head = cp_mod.list_checkpoints(tmp_path).head_id
    assert head == out["post_checkpoint"]

    stored = arena_mod.last_arena(tmp_path)
    assert stored["winner"] == "a"
    assert stored["applied"]["checkpoint"] == out["post_checkpoint"]

    # The pre-apply checkpoint is a clean undo back to the original workspace.
    cp_mod.restore_checkpoint(tmp_path, out["pre_checkpoint"])
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == APP


def test_arena_apply_deletes_only_baseline_files_the_winner_removed(
        tmp_path, monkeypatch):
    _workspace(tmp_path, {"extra.py": "remove me\n"})
    cfg = _arena_cfg(tmp_path, allow_write=True)
    _patch_contestants(
        monkeypatch,
        steps_by_model={"big-model": ({"action": "delete", "file": "extra.py"},)})
    state = arena_mod.run_arena_comparison(
        tmp_path, "make greet return its greeting and drop extra.py",
        "big-model", "small-model", cfg)
    sandbox_a = Path(state["sandboxes"]["a"])
    sandbox_b = Path(state["sandboxes"]["b"])
    assert not (sandbox_a / "extra.py").exists()         # A deleted it
    assert (sandbox_b / "extra.py").exists()             # B kept it
    assert "extra.py" in state["contestants"]["a"]["files_changed"]

    # A file created in the real workspace AFTER the arena must survive apply.
    (tmp_path / "late.py").write_text("created later\n", encoding="utf-8")
    out = arena_mod.apply_arena_winner(tmp_path, state["arena_id"], "a")
    assert out["deleted"] == ["extra.py"]
    assert not (tmp_path / "extra.py").exists()
    assert (tmp_path / "late.py").read_text(encoding="utf-8") == "created later\n"
    assert 'return f"hello {name}"' in (tmp_path / "app.py").read_text(
        encoding="utf-8")


def test_arena_evicts_between_local_contestants(tmp_path, monkeypatch):
    _workspace(tmp_path)
    cfg = _arena_cfg(tmp_path)
    _patch_contestants(monkeypatch)
    evicted: list[str] = []
    monkeypatch.setattr(
        servers_mod, "evict_other_local_servers",
        lambda target, c, **kw: evicted.append(servers_mod.normalize_url(target))
        or {"ok": True, "evicted": []})
    arena_mod.run_arena_comparison(
        tmp_path, "make greet return its greeting",
        "http://127.0.0.1:8001/v1:big-model",
        "http://127.0.0.1:8002/v1:small-model", cfg)
    # One handoff: contestant A's server gives the VRAM to contestant B's.
    assert evicted == ["http://127.0.0.1:8002"]


def test_arena_prunes_older_sandboxes_on_a_new_run(tmp_path, monkeypatch):
    _workspace(tmp_path)
    cfg = _arena_cfg(tmp_path)
    _patch_contestants(monkeypatch)
    first = arena_mod.run_arena_comparison(
        tmp_path, "make greet return its greeting", "big-model", "small-model",
        cfg)
    second = arena_mod.run_arena_comparison(
        tmp_path, "make greet return its greeting", "big-model", "small-model",
        cfg)
    assert first["arena_id"] != second["arena_id"]
    base = tmp_path / ".pedacito" / "arena"
    assert not (base / first["arena_id"]).exists()
    assert (base / second["arena_id"]).is_dir()
    assert arena_mod.last_arena(tmp_path)["arena_id"] == second["arena_id"]


def test_arena_failed_contestant_is_a_row_not_an_abort(tmp_path, monkeypatch):
    _workspace(tmp_path)
    cfg = _arena_cfg(tmp_path)
    _patch_contestants(monkeypatch, fail_models=("broken-model",))
    state = arena_mod.run_arena_comparison(
        tmp_path, "make greet return its greeting", "broken-model",
        "small-model", cfg)
    a, b = state["contestants"]["a"], state["contestants"]["b"]
    assert a["ok"] is False and a["summary"].startswith("run failed:")
    assert b["ok"] is True and b["files_changed"] == ["app.py"]


def test_arena_user_stop_aborts_the_whole_comparison(tmp_path, monkeypatch):
    _workspace(tmp_path)
    cfg = _arena_cfg(tmp_path)
    _patch_contestants(monkeypatch, cancel_models=("big-model",))
    with pytest.raises(llm_mod.CancelledError):
        arena_mod.run_arena_comparison(
            tmp_path, "make greet return its greeting", "big-model",
            "small-model", cfg)


def test_arena_rejects_bad_requests_and_missing_sandboxes(tmp_path, monkeypatch):
    _workspace(tmp_path)
    cfg = _arena_cfg(tmp_path)
    with pytest.raises(arena_mod.ArenaError):
        arena_mod.run_arena_comparison(tmp_path, "  ", "a-model", "b-model", cfg)
    with pytest.raises(arena_mod.ArenaError):
        arena_mod.run_arena_comparison(tmp_path, "task", "", "b-model", cfg)
    with pytest.raises(arena_mod.ArenaError):
        arena_mod.run_arena_comparison(tmp_path, "task", "a-model", "", cfg)

    _patch_contestants(monkeypatch)
    state = arena_mod.run_arena_comparison(
        tmp_path, "make greet return its greeting", "big-model", "small-model",
        cfg)
    arena_id = state["arena_id"]
    with pytest.raises(arena_mod.ArenaError):
        arena_mod.apply_arena_winner(tmp_path, arena_id, "c")
    with pytest.raises(arena_mod.ArenaError):
        arena_mod.apply_arena_winner(tmp_path, "arena_nope", "a")
    shutil.rmtree(state["sandboxes"]["a"])
    with pytest.raises(arena_mod.ArenaError):
        arena_mod.apply_arena_winner(tmp_path, arena_id, "a")


# ------------------------------------------------------------------ GUI -----

@pytest.mark.network
def test_gui_arena_last_empty_and_apply_without_run(project, monkeypatch):
    monkeypatch.setattr(arena_mod, "_LAST", None)
    status, last = project.get("/api/arena/last")
    assert status == 200 and last == {"ok": True, "arena": None}
    status, out = project.post("/api/arena/apply", {"winner": "a"})
    assert status == 400 and out["ok"] is False


@pytest.mark.network
def test_gui_arena_run_last_and_apply(project, fake_port, monkeypatch):
    gui = project
    # Keep the handshake deterministic: no real server probing in the worker.
    monkeypatch.setattr(servers_mod, "evict_other_local_servers",
                        lambda *a, **k: {"ok": True, "evicted": []})
    status, _body = gui.post("/api/arena/run", {
        "text": "make greet return its greeting",
        "model_a": "fake-model", "model_b": "fake-model-b"})
    assert status == 202
    assert gui.wait_idle(60)

    # Both contestants ran in sandboxes; the real workspace is untouched.
    assert (gui.root / "app.py").read_text(encoding="utf-8") == APP
    status, last = gui.get("/api/arena/last")
    assert status == 200 and last["ok"] and last["arena"]
    arena = last["arena"]
    assert arena["contestants"]["a"]["ok"] is True
    assert arena["contestants"]["b"]["ok"] is True
    assert arena["contestants"]["a"]["files_changed"] == ["app.py"]
    assert arena["contestants"]["a"]["model"] == "fake-model"
    assert arena["contestants"]["b"]["model"] == "fake-model-b"
    assert any(m.get("kind") == "arena" for m in gui.get("/api/state")[1]["messages"])

    # Apply winner A: the edit lands between two labeled checkpoints.
    status, out = gui.post("/api/arena/apply",
                           {"arena_id": arena["arena_id"], "winner": "a"})
    assert status == 200 and out["ok"], out
    assert out["files_changed"] == ["app.py"]
    assert 'return f"hello {name}"' in (gui.root / "app.py").read_text(
        encoding="utf-8")
    labels = [c["label"] for c in cp_mod.list_checkpoints(gui.root)]
    assert "Arena winner: fake-model" in labels
    assert any(l.startswith("Arena pre-apply") for l in labels)
    assert gui.get("/api/arena/last")[1]["arena"]["winner"] == "a"


# ------------------------------------------------------------------ TUI -----

def test_tui_hybrid_command_arms_and_disarms(tmp_path):
    cfg = Config()
    cfg.model = "editor-fast"
    cfg.active_profile = "default"
    cfg.profiles = {"arch": {"model": "architect-big",
                             "base_url": "http://127.0.0.1:9/v1"}}
    script = "/hybrid arch\n/hybrid\n/hybrid off\n/quit\n"
    out = io.StringIO()
    code = tui_mod.run_tui(tmp_path, cfg, headless=True,
                           stdin=io.StringIO(script), stdout=out,
                           client=FakeClient())
    assert code == 0
    text = out.getvalue()
    assert "Hybrid routing: ON" in text
    assert "architect-big" in text
    assert "Hybrid routing: OFF" in text
    assert cfg.hybrid_routing is False                   # /hybrid off won
    assert cfg.architect_model == "architect-big"        # selection remembered


def test_tui_hybrid_rejects_the_same_model_and_renders_the_scorecard(tmp_path):
    cfg = Config()
    cfg.model = "editor-fast"
    cfg.active_profile = "default"
    session = tui_mod.TuiSession(tmp_path, cfg, client=FakeClient(),
                                 headless=True, stdout=io.StringIO())
    try:
        session._hybrid("editor-fast")                   # architect == editor
        assert any("equals the editor model" in line
                   for _kind, line in session.transcript)
        assert cfg.hybrid_routing is False

        state = {"arena_id": "arena_demo", "prompt": "make greet return",
                 "winner": "a",
                 "contestants": {
                     "a": {"provider": "local", "model": "big-model", "ok": True,
                           "files_changed": ["app.py"], "elapsed_sec": 1.5,
                           "tok_per_sec": 40.0, "total_tokens": 280,
                           "summary": "Return the greeting."},
                     "b": {"provider": "", "model": "small-model", "ok": False,
                           "files_changed": [], "elapsed_sec": 0.5,
                           "tok_per_sec": 0.0, "total_tokens": 0,
                           "summary": "run failed: boom"}}}
        card = session._arena_scorecard(state)
        assert "arena_demo" in card
        assert "local/big-model" in card and "small-model" in card
        assert "280 tokens" in card
        assert "winner applied: A" in card
    finally:
        session.close()
