"""CLI end to end against the fake server: gather -> preflight -> apply ->
events/usage, the Ctrl-C-safe chat loop, and the budget stop."""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import cli, config as config_mod, events   # noqa: E402

FAKE = Path(__file__).resolve().parent / "fake_llm.py"
REPLY = ("Raised the limit.\n\n--- FILE: app.py\n<<<<<<< SEARCH\nLIMIT = 10\n"
         "=======\nLIMIT = 25\n>>>>>>> REPLACE\n")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def fake_port():
    port = _free_port()
    env = dict(os.environ, FAKE_PORT=str(port), FAKE_REPLY=REPLY,
               FAKE_READ_FILE="app.py")
    env.pop("FAKE_ACTIONS", None)
    proc = subprocess.Popen([sys.executable, str(FAKE)], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), 0.5):
                    break
            except OSError:
                time.sleep(0.1)
        yield port
    finally:
        proc.terminate()
        proc.wait(timeout=5)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "USER_CONFIG", tmp_path / "home" / "config.toml")
    monkeypatch.setenv("PEDACITO_VRAM_GB", "8")
    monkeypatch.delenv("PEDACITO_PROFILE", raising=False)
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "app.py").write_text("LIMIT = 10\n\n\ndef check(n):\n    return n < LIMIT\n",
                                 encoding="utf-8")
    return proj


def _args(port, proj, *extra):
    return ["ask", str(proj), "-t", "raise LIMIT in app.py to 25", "--url",
            f"http://127.0.0.1:{port}/v1", "--model", "fake-model", *extra]


@pytest.mark.network
def test_ask_apply_runs_the_whole_pipeline_and_logs_events(fake_port, env, capsys):
    code = cli.main(_args(fake_port, env, "--apply"))
    err = capsys.readouterr().err
    assert code == 0
    assert "LIMIT = 25" in (env / "app.py").read_text()
    assert "[task:" in err and "cache hit" in err           # per-task usage line
    log = events.load(env)
    kinds = [e["kind"] for e in log]
    assert {"session_start", "step", "gather_done", "answer", "task", "call"} <= set(kinds)
    task = next(e for e in log if e["kind"] == "task")
    assert task["edits_ok"] == 1 and task["answer_mode"] == "rebuild"
    assert any(e["kind"] == "call" and e.get("site") == "gather" for e in log)
    assert any(e["kind"] == "call" and e.get("site") == "answer" for e in log)
    # nothing secret-shaped or protected was written
    assert not (env / ".pedacito" / "PLAN.md").exists()


@pytest.mark.network
def test_dry_run_writes_nothing(fake_port, env, capsys):
    assert cli.main(_args(fake_port, env)) == 0
    assert "LIMIT = 10" in (env / "app.py").read_text()
    assert "dry run" in capsys.readouterr().err


@pytest.mark.network
def test_token_budget_stops_the_run_cleanly(fake_port, env, capsys):
    code = cli.main(_args(fake_port, env, "--apply", "--max-tokens-budget", "1"))
    err = capsys.readouterr().err
    assert code == 1 and "Token budget spent" in err
    assert "LIMIT = 10" in (env / "app.py").read_text()


@pytest.mark.network
def test_chat_survives_ctrl_c_during_a_task(fake_port, env, monkeypatch, capsys):
    calls = {"n": 0}
    real = cli.loop.run_task

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt
        return real(*a, **k)

    monkeypatch.setattr(cli.loop, "run_task", flaky)
    lines = iter(["raise LIMIT in app.py to 25", "raise LIMIT in app.py to 25", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    code = cli.main(["chat", str(env), "--apply", "--url",
                     f"http://127.0.0.1:{fake_port}/v1", "--model", "fake-model"])
    err = capsys.readouterr().err
    assert code == 0 and calls["n"] == 2
    assert "cancelled; the session is still open" in err
    assert "LIMIT = 25" in (env / "app.py").read_text()


@pytest.mark.network
def test_cost_and_status_slash_commands(fake_port, env, monkeypatch, capsys):
    lines = iter(["/status", "/cost", "/memory", "/skills", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    code = cli.main(["chat", str(env), "--url", f"http://127.0.0.1:{fake_port}/v1",
                     "--model", "fake-model"])
    err = capsys.readouterr().err
    assert code == 0
    assert "prompts : pack full" in err and "window  :" in err
    assert "calls" in err and "add-a-test" in err


@pytest.mark.network
def test_jsonl_streams_events_on_stdout_only(fake_port, env, capsys):
    import json
    assert cli.main(_args(fake_port, env, "--jsonl")) == 0
    out = capsys.readouterr().out
    recs = [json.loads(ln) for ln in out.splitlines()]      # every line is JSON
    kinds = [r["kind"] for r in recs]
    assert kinds[0] == "session_start" and kinds[-1] == "exit"
    res = next(r for r in recs if r["kind"] == "result")
    assert "LIMIT = 25" in res["edits"][0]["diff"] and res["reply"]
    assert "LIMIT = 10" in (env / "app.py").read_text()      # still a dry run


@pytest.mark.network
def test_plan_mode_dry_run_lists_one_step_per_requirement(fake_port, env, capsys):
    task = "- raise LIMIT in app.py to 25\n- keep check() working\n"
    args = ["ask", str(env), "-t", task, "--url", f"http://127.0.0.1:{fake_port}/v1",
            "--model", "fake-model", "--plan"]
    assert cli.main(args) == 0
    err = capsys.readouterr().err
    assert "plan (requirements; 2 step(s))" in err and "1. " in err and "2. " in err
    assert "LIMIT = 10" in (env / "app.py").read_text()
    assert list((env / ".pedacito" / "plans").glob("*.md"))
