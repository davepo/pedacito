"""Time-travel checkpoints and the one-click project runner.

Covers the checkpoint tree (snapshot, per-file diff, single-file/full restore,
branch-after-restore, undo), the automatic pre/post checkpoints in the task
loop, target detection, sandboxed command execution with traceback capture,
and the GUI endpoints -- including the preview path-traversal gate.
"""
from __future__ import annotations

import http.client
import io
import json
import shlex
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import checkpoints, cli, config as config_mod, loop, runner  # noqa: E402
from pedacito.config import Config  # noqa: E402
from pedacito.gui import core, server, settings_api  # noqa: E402
from pedacito.tui import TuiSession  # noqa: E402


# --- checkpoint tree ----------------------------------------------------------

def test_create_list_diff_restore_and_branch(tmp_path):
    (tmp_path / "a.txt").write_text("one\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "b.py").write_text("x = 1\n", encoding="utf-8")

    first = checkpoints.create_checkpoint(tmp_path, "initial state")
    assert first["id"].startswith("cp_0001_")
    assert first["parent_id"] is None
    assert first["mode"] == "task"
    assert first["timestamp"].endswith("Z")
    assert {"a.txt", "src/b.py"} <= set(first["files_changed"])

    (tmp_path / "a.txt").write_text("two\n", encoding="utf-8")
    (tmp_path / "new.txt").write_text("created\n", encoding="utf-8")
    second = checkpoints.create_checkpoint(tmp_path, "second step")
    assert second["parent_id"] == first["id"]
    assert set(second["files_changed"]) == {"a.txt", "new.txt"}

    listing = checkpoints.list_checkpoints(tmp_path)
    assert [cp["id"] for cp in listing] == [first["id"], second["id"]]
    assert listing.head_id == second["id"]
    assert listing[1]["head"] is True and listing[0]["head"] is False
    assert listing[1]["depth"] == 1
    assert listing[0]["children"] == [second["id"]]

    parent_diff = checkpoints.diff_checkpoint(
        tmp_path, second["id"], against="parent")
    assert set(parent_diff["changed_files"]) == {"a.txt", "new.txt"}
    by_path = {f["path"]: f for f in parent_diff["files"]}
    assert by_path["a.txt"]["status"] == "modified"
    assert "-one" in by_path["a.txt"]["diff"]
    assert "+two" in by_path["a.txt"]["diff"]
    assert by_path["new.txt"]["status"] == "added"
    assert "+created" in parent_diff["diff"]

    # A live edit shows up in the workspace diff, in the right direction.
    (tmp_path / "a.txt").write_text("three\n", encoding="utf-8")
    live_diff = checkpoints.diff_checkpoint(
        tmp_path, second["id"], against="workspace")
    live = {f["path"]: f for f in live_diff["files"]}
    assert live["a.txt"]["status"] == "modified"
    assert "+three" in live["a.txt"]["diff"]

    # Restore one file only: the workspace-created file stays.
    single = checkpoints.restore_checkpoint(tmp_path, first["id"],
                                            files=["a.txt"])
    assert single["restored"] == ["a.txt"]
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "one\n"
    assert (tmp_path / "new.txt").is_file()
    assert checkpoints.head_id(tmp_path) == first["id"]

    # A checkpoint created now branches from `first`, not the old head.
    branch = checkpoints.create_checkpoint(tmp_path, "branch")
    assert branch["parent_id"] == first["id"]
    assert branch["parent_id"] != second["id"]

    # A full restore matches the checkpoint exactly: later files disappear.
    checkpoints.restore_checkpoint(tmp_path, first["id"])
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "one\n"
    assert not (tmp_path / "new.txt").exists()
    assert (tmp_path / "src" / "b.py").read_text(encoding="utf-8") == "x = 1\n"


def test_restore_creates_a_safety_snapshot_and_works_from_selected_files(tmp_path):
    (tmp_path / "app.py").write_text("v1\n", encoding="utf-8")
    base = checkpoints.create_checkpoint(tmp_path, "base")
    (tmp_path / "app.py").write_text("v2\n", encoding="utf-8")
    checkpoints.create_checkpoint(tmp_path, "v2")
    (tmp_path / "app.py").write_text("dirty\n", encoding="utf-8")

    result = checkpoints.restore_checkpoint(tmp_path, base["id"])
    assert result["ok"] is True
    assert result["safety_id"]
    safety = next(cp for cp in checkpoints.list_checkpoints(tmp_path)
                  if cp["id"] == result["safety_id"])
    assert safety["mode"] == "restore"
    assert "app.py" in safety["files_changed"]
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "v1\n"


def test_undo_restores_the_state_before_the_last_run(tmp_path, capsys):
    (tmp_path / "app.py").write_text("before\n", encoding="utf-8")
    checkpoints.ensure_baseline(tmp_path, "baseline")
    (tmp_path / "app.py").write_text("after\n", encoding="utf-8")
    checkpoints.create_checkpoint(tmp_path, "the run")

    assert checkpoints.undo_last_run(tmp_path)["ok"] is True
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "before\n"

    # The CLI command is the same operation, with the same outcome.
    (tmp_path / "app.py").write_text("again\n", encoding="utf-8")
    checkpoints.create_checkpoint(tmp_path, "another run")
    code = cli.main(["undo", "--root", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "before\n"
    assert "undo: back to" in capsys.readouterr().err

    # The TUI slash command falls back to checkpoints before backups.
    (tmp_path / "app.py").write_text("tui dirty\n", encoding="utf-8")
    checkpoints.create_checkpoint(tmp_path, "tui run")
    session = TuiSession(tmp_path, Config(), headless=True,
                         stdout=io.StringIO())
    try:
        session.handle_line("/undo")
        assert (tmp_path / "app.py").read_text(encoding="utf-8") == "before\n"
        session.handle_line("/checkpoints")
        text = "\n".join(line for _kind, line in session.transcript)
        assert "Checkpoints for" in text
    finally:
        session.close()


def test_large_binary_artifacts_are_not_snapshotted(tmp_path):
    (tmp_path / "keep.py").write_text("print('ok')\n", encoding="utf-8")
    big = tmp_path / "blob.bin"
    big.write_bytes(b"\x00" + b"x" * (checkpoints.MAX_BINARY_BYTES + 10))
    cp = checkpoints.create_checkpoint(tmp_path, "binary skip")
    assert "keep.py" in cp["files_changed"]
    assert "blob.bin" not in cp["files_changed"]


# --- automatic checkpointing --------------------------------------------------

def test_task_run_records_a_baseline_and_a_post_run_checkpoint(tmp_path,
                                                               monkeypatch):
    (tmp_path / "app.py").write_text("before\n", encoding="utf-8")
    cfg = Config()
    cfg.use_memory = False
    cfg._auto_step = 1                        # run as an Auto step
    wrote = []

    def fake_ladder(root, cfg_ref, client, agent, idx, task, **kwargs):
        (Path(root) / "app.py").write_text("after\n", encoding="utf-8")
        wrote.append(task)
        return loop.TaskResult(
            results=[], reply="done", applied=True, changed=["app.py"],
            idx=idx, steps=[])

    monkeypatch.setattr(loop, "_run_ladder", fake_ladder)
    tr = loop.run_task(tmp_path, cfg, object(), None, None, "change app")
    assert tr.reply == "done"
    items = checkpoints.list_checkpoints(tmp_path)
    assert len(items) == 2
    assert items[0]["label"].startswith("baseline before")
    assert items[1]["mode"] == "auto"
    assert items[1]["parent_id"] == items[0]["id"]
    assert "app.py" in items[1]["files_changed"]


def test_task_run_memory_path_records_exactly_one_post_checkpoint(tmp_path,
                                                                  monkeypatch):
    """The DEFAULT run_task path (use_memory on) records ONE post-run
    checkpoint. A second empty post snapshot duplicates the timeline entry
    and halves the effective auto_keep_checkpoints retention; the
    use_memory=False test above only exercises the early-return branch."""
    (tmp_path / "app.py").write_text("before\n", encoding="utf-8")
    cfg = Config()
    assert cfg.use_memory is True
    cfg._auto_step = 1                        # run as an Auto step

    def fake_ladder(root, cfg_ref, client, agent, idx, task, **kwargs):
        (Path(root) / "app.py").write_text("after\n", encoding="utf-8")
        return loop.TaskResult(
            results=[], reply="done", applied=True, changed=["app.py"],
            idx=idx, steps=[])

    monkeypatch.setattr(loop, "_run_ladder", fake_ladder)
    tr = loop.run_task(tmp_path, cfg, object(), None, None, "change app")
    assert tr.reply == "done"
    items = checkpoints.list_checkpoints(tmp_path)
    assert len(items) == 2                    # baseline + ONE post, not two
    assert items[0]["label"].startswith("baseline before")
    assert items[1]["mode"] == "auto"
    assert items[1]["parent_id"] == items[0]["id"]
    assert "app.py" in items[1]["files_changed"]


# --- runner detection / execution ---------------------------------------------

def test_detect_runnable_targets_finds_web_and_python_entrypoints(tmp_path):
    (tmp_path / "index.html").write_text("<canvas></canvas>", encoding="utf-8")
    (tmp_path / "public").mkdir()
    (tmp_path / "public" / "index.html").write_text("<h1>p</h1>",
                                                     encoding="utf-8")
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "index.html").write_text("<h1>d</h1>",
                                                   encoding="utf-8")
    (tmp_path / "main.py").write_text("print('main')\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("print('app')\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8")
    (tmp_path / ".pedacito").mkdir()
    (tmp_path / ".pedacito" / "config.json").write_text(
        json.dumps({"run_cmd": "python main.py --smoke"}),
        encoding="utf-8")
    # The state file's `run_cmd` is only honoured once the project is trusted
    # (a cloned repo must not be able to run arbitrary commands from the Run
    # tab). This test trusts the project so the custom command is listed.
    (tmp_path / ".pedacito.toml").write_text("# approved\n", encoding="utf-8")
    config_mod.trust_project(tmp_path / ".pedacito.toml")

    targets = runner.detect_runnable_targets(tmp_path)
    assert [p["path"] for p in targets["web"]] == [
        "index.html", "public/index.html", "dist/index.html"]
    assert targets["has_preview"] is True
    assert targets["web"][0]["url"] == "/api/runner/preview/index.html"
    commands = [c["command"] for c in targets["commands"]]
    assert "python main.py" in commands
    assert "python app.py" in commands
    assert "pytest -q" in commands
    assert "python main.py --smoke" in commands
    assert targets["python_entrypoints"] == ["main.py", "app.py"]
    assert "python main.py" in targets["commands"]      # compares like a string
    assert targets["web"][0] == "index.html"


def test_detect_runnable_targets_ignores_untrusted_run_cmd(tmp_path, capsys):
    """A cloned repo can ship `.pedacito/config.json` with a `run_cmd`; it is
    never listed (or run) until the project is trusted."""
    (tmp_path / "main.py").write_text("print('main')\n", encoding="utf-8")
    (tmp_path / ".pedacito").mkdir()
    (tmp_path / ".pedacito" / "config.json").write_text(
        json.dumps({"run_cmd": "python main.py --pwned"}), encoding="utf-8")

    targets = runner.detect_runnable_targets(tmp_path)
    commands = [c["command"] for c in targets["commands"]]
    assert "python main.py --pwned" not in commands
    assert "not trusted" in capsys.readouterr().err


def test_run_project_command_captures_streams_exit_and_traceback(tmp_path):
    (tmp_path / "ok.py").write_text(
        "import sys\nprint('hello out')\nprint('hello err', file=sys.stderr)\n",
        encoding="utf-8")
    ok = runner.run_project_command(tmp_path, [sys.executable, "ok.py"])
    assert ok["ok"] is True and ok["exit_code"] == 0
    # `proc.run` merges stderr into stdout (one bounded stream).
    assert "hello out" in ok["stdout"]
    assert "hello err" in ok["stdout"]
    assert ok["traceback_summary"] == ""
    assert ok["duration_ms"] >= 0

    (tmp_path / "bad.py").write_text(
        "import sys\nprint('partial')\nprint('problem', file=sys.stderr)\n"
        "sys.exit(3)\n", encoding="utf-8")
    bad = runner.run_project_command(tmp_path, [sys.executable, "bad.py"])
    assert bad["ok"] is False and bad["exit_code"] == 3
    assert "partial" in bad["stdout"]
    assert "problem" in bad["stdout"]

    (tmp_path / "crash.py").write_text(
        "def explode():\n    raise ValueError('boom')\n\nexplode()\n",
        encoding="utf-8")
    crash = runner.run_project_command(tmp_path, [sys.executable, "crash.py"])
    assert crash["ok"] is False and crash["exit_code"] != 0
    assert "Traceback (most recent call last):" in crash["traceback_summary"]
    assert "ValueError: boom" in crash["traceback_summary"]
    prompt = runner.crash_prompt(crash)
    assert "ValueError: boom" in prompt
    assert "crash.py" in prompt and "Exit code:" in prompt

    timeout = runner.run_project_command(
        tmp_path, [sys.executable, "-c",
                   "import time; time.sleep(2)"], timeout=0.2)
    assert timeout["ok"] is False and timeout["timed_out"] is True


# --- GUI endpoints ------------------------------------------------------------

class GuiClient:
    def __init__(self, port: int, tmp: Path):
        self.port = port
        self.tmp = tmp

    def req(self, method: str, path: str, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        data = None if body is None else json.dumps(body)
        conn.request(method, path, body=data,
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        raw = response.read()
        status = response.status
        conn.close()
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = raw
        return status, payload

    def get(self, path):
        return self.req("GET", path)

    def post(self, path, body=None):
        return self.req("POST", path, body or {})


@pytest.fixture
def gui(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg" / "config.toml"
    from pedacito import config as config_mod
    monkeypatch.setattr(config_mod, "USER_CONFIG", cfg_path)
    monkeypatch.setattr(core, "USER_CONFIG", cfg_path)
    monkeypatch.setattr(settings_api, "USER_CONFIG", cfg_path)
    monkeypatch.setattr(core, "RECENT_FILE", tmp_path / "cfg" / "recent.json")
    core.STATE.__init__()
    srv = server.create_server(0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True,
                              name="pedacito-test-checkpoints-gui")
    thread.start()
    try:
        yield GuiClient(srv.server_address[1], tmp_path)
    finally:
        core.STATE.bus.close()
        srv.shutdown()
        srv.server_close()
        thread.join(5)
        core.STATE.__init__()


def test_tui_run_remembers_the_crash_and_fix_sends_it(monkeypatch, tmp_path):
    (tmp_path / "main.py").write_text("print('ok')\n", encoding="utf-8")
    session = TuiSession(tmp_path, Config(), headless=True,
                         stdout=io.StringIO())
    failure = {
        "ok": False, "exit_code": 1,
        "stdout": "partial\n", "stderr": "",
        "traceback_summary": "Traceback (most recent call last):\n"
                             "RuntimeError: tui boom",
        "command": "python main.py", "duration_ms": 4, "timed_out": False,
    }
    monkeypatch.setattr(runner, "run_project_command",
                        lambda root, cmd=None, timeout=15: dict(failure))
    try:
        session.handle_line("/run")
        assert "RuntimeError: tui boom" in session.last_traceback
        captured = {}

        def fake_task(prompt):
            captured["prompt"] = prompt

        monkeypatch.setattr(session, "run_task", fake_task)
        session.handle_line("/fix")
        assert "python main.py" in captured["prompt"]
        assert "RuntimeError: tui boom" in captured["prompt"]
        assert "Exit code: 1" in captured["prompt"]
    finally:
        session.close()


def test_gui_checkpoint_endpoints_and_restore(gui):
    root = gui.tmp / "project"
    root.mkdir()
    (root / "app.py").write_text("v1\n", encoding="utf-8")
    (root / "notes.txt").write_text("keep\n", encoding="utf-8")
    core.STATE.root = root

    status, created = gui.post("/api/checkpoints/create",
                               {"label": "manual baseline"})
    assert status == 200 and created["ok"] is True
    cp = created["checkpoint"]
    assert cp["mode"] == "manual"
    assert cp["parent_id"] is None

    status, listing = gui.get("/api/checkpoints")
    assert status == 200 and listing["ok"] is True
    assert listing["head_id"] == cp["id"]
    assert [item["id"] for item in listing["checkpoints"]] == [cp["id"]]

    (root / "app.py").write_text("v2\n", encoding="utf-8")
    status, created2 = gui.post("/api/checkpoints/create",
                                {"label": "second"})
    assert status == 200
    second = created2["checkpoint"]
    assert second["parent_id"] == cp["id"]

    status, diff = gui.get(
        f"/api/checkpoints/diff?id={second['id']}&against=parent")
    assert status == 200 and diff["changed_files"] == ["app.py"]
    assert "+v2" in diff["diff"]

    status, restored = gui.post("/api/checkpoints/restore",
                                {"id": cp["id"], "files": ["app.py"]})
    assert status == 200 and restored["ok"] is True
    assert restored["restored"] == ["app.py"]
    assert (root / "app.py").read_text(encoding="utf-8") == "v1\n"

    # Full restore also removes files created after the checkpoint.
    (root / "later.txt").write_text("new\n", encoding="utf-8")
    status, restored = gui.post("/api/checkpoints/restore", {"id": cp["id"]})
    assert status == 200
    assert "later.txt" in restored["deleted"]
    assert not (root / "later.txt").exists()

    # Path traversal in a selected file is refused before any write.
    status, bad = gui.post("/api/checkpoints/restore",
                           {"id": cp["id"], "files": ["../secret.txt"]})
    assert status == 404 and bad["ok"] is False


def test_gui_runner_targets_preview_exec_and_traversal(gui):
    root = gui.tmp / "webapp"
    root.mkdir()
    (root / "index.html").write_text(
        "<!doctype html><canvas id=c></canvas>", encoding="utf-8")
    (root / "assets").mkdir()
    (root / "assets" / "app.css").write_text("canvas{display:block}",
                                              encoding="utf-8")
    (root / "main.py").write_text("print('ran')\n", encoding="utf-8")
    secret = gui.tmp / "secret.txt"
    secret.write_text("do not serve\n", encoding="utf-8")
    core.STATE.root = root

    status, targets = gui.get("/api/runner/targets")
    assert status == 200 and targets["ok"] is True
    assert [p["path"] for p in targets["web"]] == ["index.html"]
    assert "python main.py" in [c["command"] for c in targets["commands"]]

    status, body = gui.get("/api/runner/preview/index.html")
    assert status == 200 and b"<canvas" in body
    status, body = gui.get("/api/runner/preview/assets/app.css")
    assert status == 200 and b"canvas{" in body

    assert gui.get("/api/runner/preview/../secret.txt")[0] == 404
    assert gui.get("/api/runner/preview/%2e%2e%2fsecret.txt")[0] == 404
    assert gui.get("/api/runner/preview/..%5csecret.txt")[0] == 404

    command = f'{shlex.quote(sys.executable)} main.py'
    status, result = gui.post("/api/runner/exec", {"command": command})
    assert status == 200 and result["ok"] is True
    assert "ran" in result["stdout"]

    (root / "crash.py").write_text(
        "raise RuntimeError('gui boom')\n", encoding="utf-8")
    status, result = gui.post(
        "/api/runner/exec",
        {"command": f"{shlex.quote(sys.executable)} crash.py"})
    assert status == 200 and result["ok"] is False
    assert "RuntimeError: gui boom" in result["traceback_summary"]

    status, result = gui.post(
        "/api/runner/exec",
        {"command": [sys.executable, "-c", "print('json argv')"]})
    assert status == 200 and result["ok"] is True
    assert "json argv" in result["stdout"]


def test_gui_runner_endpoints_need_a_project(gui):
    status, body = gui.get("/api/runner/targets")
    assert status == 400 and body["ok"] is False
    status, body = gui.get("/api/checkpoints")
    assert status == 400 and body["ok"] is False
