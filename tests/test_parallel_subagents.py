"""Bounded parallel subagents: disjoint scopes, isolation and verification."""
from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import session_ctx, subagent  # noqa: E402
from pedacito.config import Config  # noqa: E402


class FakeBus:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def emit(self, kind, **fields):
        self.events.append((kind, fields))


def _parent(tmp_path, monkeypatch, **cfg_values):
    root = tmp_path / "proj"
    root.mkdir(exist_ok=True)
    cfg = Config(**cfg_values)
    ctx = session_ctx.SessionContext(root=root, cfg=cfg, mode="interactive")
    ctx.event_bus = FakeBus()
    return ctx


# --------------------------------------------------------------------------- #
# scope validation
# --------------------------------------------------------------------------- #
def test_validate_disjoint_scopes_accepts_partitions():
    subagent.validate_disjoint_scopes([
        subagent.SubagentSpec("explore", "left", ("src/a/**",)),
        subagent.SubagentSpec("explore", "right", ("src/b/**",)),
        subagent.SubagentSpec("general", "write", ("docs/**",)),
        subagent.SubagentSpec("verifier", "check"),
    ])
    subagent.validate_disjoint_scopes([])
    subagent.validate_disjoint_scopes([
        subagent.SubagentSpec("verifier", "only verifier"),
    ])


@pytest.mark.parametrize("left,right", [
    ("src/", "src/api/"),
    ("a.py", "a.py"),
    ("src/**", "src/foo.py"),
    ("src/*.py", "src/app.py"),
])
def test_validate_disjoint_scopes_rejects_overlap(left, right):
    with pytest.raises(ValueError):
        subagent.validate_disjoint_scopes([
            subagent.SubagentSpec("explore", "a", (left,)),
            subagent.SubagentSpec("general", "b", (right,)),
        ])


def test_validate_disjoint_scopes_rejects_empty_sibling_scope():
    with pytest.raises(ValueError):
        subagent.validate_disjoint_scopes([
            subagent.SubagentSpec("explore", "a", ("src/**",)),
            subagent.SubagentSpec("general", "b"),
        ])


# --------------------------------------------------------------------------- #
# model families
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model,family", [
    ("claude-3-5-sonnet", "claude"),
    ("openrouter/anthropic/claude-opus", "claude"),
    ("gemini-2.5-pro", "gemini"),
    ("gemma2:9b", "gemma"),
    ("deepseek-chat", "deepseek"),
    ("qwen2.5-coder", "qwen"),
    ("gpt-4o", "openai"),
    ("o1-mini", "openai"),
    ("o3", "openai"),
    ("o4-mini", "openai"),
    ("llama3:8b", "llama"),
    ("mistral-small", "mistral"),
    ("phi-3-mini", "phi"),
    ("", ""),
])
def test_model_family(model, family):
    assert subagent.model_family(model) == family


# --------------------------------------------------------------------------- #
# concurrency + per-worker isolation
# --------------------------------------------------------------------------- #
def test_parallel_workers_are_bounded_and_isolated(tmp_path, monkeypatch):
    parent = _parent(tmp_path, monkeypatch, subagent_max_parallel=2)
    lock = threading.Lock()
    active = {"now": 0, "max": 0}
    seen: list[tuple[str, str, str]] = []

    def runner(ctx, spec):
        with lock:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        seen.append((ctx.session_id, spec.role, spec.brief))
        assert session_ctx.current_session() is ctx
        assert ctx.parent_session_id == parent.session_id
        assert ctx.mode == "subagent"
        assert ctx.policy.mode == "plan"          # explore is read-only
        time.sleep(0.12)
        with lock:
            active["now"] -= 1
        return subagent.SubagentOutcome(spec.role, ctx.session_id, True,
                                        f"{spec.role}:{spec.brief}", [])

    specs = [
        subagent.SubagentSpec("explore", f"q{i}", (f"src/{i}/**",))
        for i in range(4)
    ]
    outcomes = subagent.run_parallel_subagents(parent, specs,
                                               runner=runner)
    assert len(outcomes) == 4
    assert active["max"] <= 2                     # cfg subagent_max_parallel
    assert all(o.ok for o in outcomes)
    assert len({o.session_id for o in outcomes}) == 4
    assert parent.session_id not in {o.session_id for o in outcomes}
    assert {brief for _sid, _role, brief in seen} == {f"q{i}" for i in range(4)}


def _git_repo(root: Path) -> bool:
    if not shutil.which("git"):
        return False
    (root / "README.md").write_text("init\n", encoding="utf-8")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
    for cmd in (["git", "init"],
                ["git", "config", "user.email", "t@example.com"],
                ["git", "config", "user.name", "t"],
                ["git", "add", "."],
                ["git", "commit", "-m", "init"]):
        result = subprocess.run(cmd, cwd=str(root), shell=False,
                                capture_output=True, text=True,
                                encoding="utf-8", env=env)
        if result.returncode != 0:
            return False
    return True


def test_general_worker_gets_an_isolated_worktree(tmp_path, monkeypatch):
    parent = _parent(tmp_path, monkeypatch)
    if not _git_repo(parent.root):
        pytest.skip("git is not available")

    def runner(ctx, spec):
        assert ctx.root != parent.root
        assert (ctx.root / ".git").exists() or (ctx.root / ".git").is_file()
        (ctx.root / "new.txt").write_text("child work\n", encoding="utf-8")
        return subagent.SubagentOutcome(spec.role, ctx.session_id, True,
                                        "wrote new.txt", [])

    outcome = subagent.run_parallel_subagents(
        parent, [subagent.SubagentSpec("general", "write", ("src/",))],
        runner=runner)[0]
    assert outcome.ok
    assert outcome.worktree_branch.startswith("pedacito/sub-1-")
    assert not (parent.root / "new.txt").exists()      # parent untouched


def test_verifier_cross_family_detection(tmp_path, monkeypatch):
    parent = _parent(tmp_path, monkeypatch)
    parent.cfg.model = "claude-3-5-sonnet"
    parent.cfg.profiles = {"gemini": {"model": "gemini-2.5-pro"},
                           "claude-b": {"model": "claude-3-opus"}}

    def runner(ctx, spec):
        return subagent.SubagentOutcome(spec.role, ctx.session_id, True,
                                        "checked", [])

    cross = subagent.run_parallel_subagents(
        parent,
        [subagent.SubagentSpec("verifier", "verify", profile="gemini")],
        runner=runner)[0]
    assert cross.cross_family is True
    assert cross.verifier_family == "gemini"

    parent.event_bus.events.clear()
    same = subagent.run_parallel_subagents(
        parent,
        [subagent.SubagentSpec("verifier", "verify", profile="claude-b")],
        runner=runner)[0]
    assert same.cross_family is False
    assert same.verifier_family == "claude"
    kinds = [kind for kind, _fields in parent.event_bus.events]
    assert "verifier_same_family" in kinds


def test_verifier_is_read_only(tmp_path, monkeypatch):
    parent = _parent(tmp_path, monkeypatch)

    def runner(ctx, spec):
        assert ctx.policy.mode == "plan"
        assert ctx.cfg.allow_write is False
        return subagent.SubagentOutcome(spec.role, ctx.session_id, True,
                                        "ok", [])

    outcome = subagent.run_parallel_subagents(
        parent, [subagent.SubagentSpec("verifier", "check")],
        runner=runner)[0]
    assert outcome.ok


def test_runner_failure_is_contained(tmp_path, monkeypatch):
    parent = _parent(tmp_path, monkeypatch)

    def runner(ctx, spec):
        raise RuntimeError("boom")

    outcome = subagent.run_parallel_subagents(
        parent, [subagent.SubagentSpec("explore", "q", ("src/**",))],
        runner=runner)[0]
    assert outcome.ok is False
    assert "boom" in outcome.error
