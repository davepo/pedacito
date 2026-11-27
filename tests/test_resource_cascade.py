"""Session Resource Registry & cancellation cascade (D13).

Unit tests for registration/priority-grouped-LIFO teardown/error-resilience/
idempotency, plus an end-to-end isolation test that cancelling Session A reaps
only A's resources while a sibling Session B keeps every one of its own alive.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito.resources import SessionResourceRegistry
from pedacito.session_ctx import SessionContext, use_session


def test_register_unregister_and_list():
    reg = SessionResourceRegistry("ses_a")
    rid = reg.register("subprocess", "python server", lambda: None)
    assert rid
    assert [r["kind"] for r in reg.list_resources()] == ["subprocess"]
    reg.unregister(rid)
    assert reg.list_resources() == []


def test_teardown_priority_grouped_lifo():
    reg = SessionResourceRegistry("s")
    order: list[str] = []

    def make(name):
        return lambda: order.append(name)

    reg.register("worktree", "w", make("worktree"))        # phase 2, oldest
    reg.register("temp_mirror", "m", make("temp_mirror"))  # phase 2
    reg.register("subprocess", "p", make("subprocess"))    # phase 1
    reg.register("lsp", "l", make("lsp"))                  # phase 1, newest
    res = reg.teardown_all()
    assert res["errors"] == []
    # Phase 1 (processes) before phase 2 (filesystem), LIFO within each phase.
    assert order == ["lsp", "subprocess", "temp_mirror", "worktree"]


def test_teardown_never_raises_and_is_idempotent():
    reg = SessionResourceRegistry("s")

    def boom():
        raise RuntimeError("boom")

    reg.register("custom", "c", boom)
    rid = reg.register("subprocess", "p", lambda: None)
    res = reg.teardown_all()
    assert res["torn_down"] == [rid]
    assert len(res["errors"]) == 1
    assert res["errors"][0]["kind"] == "custom"
    # A second teardown is a clean no-op (entries were cleared).
    assert reg.teardown_all() == {"torn_down": [], "errors": []}


def test_cancellation_cascade_isolates_sibling_sessions(tmp_path):
    """Cancelling Session A reaps A's subprocess and every A resource, while a
    sibling Session B's subprocess, LSP/MCP cleanups and temp mirror stay
    completely alive and unaffected."""
    from pedacito import proc

    repo = Path(__file__).resolve().parents[1]
    a = SessionContext()
    b = SessionContext()

    flags: dict[str, bool] = {}

    def mark(name):
        def _fn():
            flags[name] = True
        return _fn

    # A and B each own a real child subprocess (sleeping), a temp mirror, and
    # mock LSP / MCP / dev-server cleanups.
    def bind(sess, prefix):
        with use_session(sess):
            reg = sess.resource_registry
            reg.register("lsp", "lsp", mark(f"{prefix}_lsp"))
            reg.register("mcp", "mcp", mark(f"{prefix}_mcp"))
            reg.register("dev_server", "srv", mark(f"{prefix}_srv"))
            mirror_dir = tmp_path / f"{prefix}_mirror"
            mirror_dir.mkdir()
            reg.register("temp_mirror", str(mirror_dir),
                         lambda: _rmdir(mirror_dir))
            return mirror_dir

    mirror_a = bind(a, "a")
    mirror_b = bind(b, "b")

    results: dict[str, str] = {}

    def run_sleep(sess, label):
        with use_session(sess):
            _rc, _out, stopped, _secs = proc.run(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                cwd=str(repo), timeout=None, cancel_event=sess.cancel_event)
            results[label] = stopped

    ta = threading.Thread(target=run_sleep, args=(a, "a"))
    tb = threading.Thread(target=run_sleep, args=(b, "b"))
    ta.start()
    tb.start()
    time.sleep(0.6)                       # let both subprocesses register

    assert any(r["kind"] == "subprocess" for r in a.resource_registry.list_resources())
    assert any(r["kind"] == "subprocess" for r in b.resource_registry.list_resources())

    a.cancel()

    ta.join(timeout=15)
    assert not ta.is_alive()              # A's subprocess was terminated promptly
    assert results.get("a") == "cancelled"
    # A's phase-1 resources (lsp/mcp/dev_server) and phase-2 mirror are reaped.
    assert flags.get("a_lsp") is True
    assert flags.get("a_mcp") is True
    assert flags.get("a_srv") is True
    assert not mirror_a.exists()

    # Session B is completely untouched.
    assert tb.is_alive()
    assert any(r["kind"] == "subprocess" for r in b.resource_registry.list_resources())
    assert flags.get("b_lsp") is not True
    assert flags.get("b_mcp") is not True
    assert flags.get("b_srv") is not True
    assert mirror_b.exists()

    b.cancel()
    tb.join(timeout=15)
    assert not tb.is_alive()
    assert not mirror_b.exists()


def test_node_modules_symlink_fallback_skips_cleanly(tmp_path, monkeypatch):
    """A Windows `OSError(WinError 1314)` on the node_modules symlink must not
    crash mirror creation: `_link_node_modules` falls back (junction) and,
    when that fails too, skips cleanly."""
    from pedacito import workspace_overlay as wo

    root = tmp_path / "proj"
    root.mkdir()
    (root / "node_modules").mkdir()
    (root / "node_modules" / "x.js").write_text("// x\n", encoding="utf-8")
    (root / "a.py").write_text("x = 1\n", encoding="utf-8")

    def boom(self, target, target_is_directory=False):
        raise OSError(1314, "A required privilege is not held by the client")

    monkeypatch.setattr(wo.Path, "symlink_to", boom)
    monkeypatch.setattr(
        wo.subprocess, "run",
        lambda *a, **k: (_ for _ in ()).throw(OSError("junction failed")))

    link = tmp_path / "nm_link"
    target = tmp_path / "nm_real"
    target.mkdir()
    wo._link_node_modules(link, target)   # must not raise
    assert not link.exists()

    # And a full dry-run mirror still builds fine with a node_modules present.
    ws = wo.VirtualWorkspace(root, apply=False)
    try:
        with ws.exec_root() as mirror:
            assert (mirror / "a.py").exists()
    finally:
        ws.close()


def _rmdir(path: Path) -> None:
    import shutil
    shutil.rmtree(path, ignore_errors=True)
