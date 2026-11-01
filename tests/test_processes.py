"""Background process registry and the opt-in `serve` tool."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import processes, tools
from pedacito.config import Config
from pedacito.index import Index


def _script(root: Path, port: str = "54321") -> Path:
    p = root / "server.py"
    p.write_text(
        "import time\n"
        f"print('running on http://127.0.0.1:{port}', flush=True)\n"
        "time.sleep(60)\n",
        encoding="utf-8")
    return p


def _cmd(script: Path) -> str:
    return f'"{sys.executable}" "{script}"'


def test_start_list_logs_and_stop(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    script = _script(root)
    rec = processes.start(root, _cmd(script), name="web", wait=3)
    try:
        assert rec["status"] == "running"
        assert rec["url"] == "http://127.0.0.1:54321"
        assert Path(rec["log"]).is_file()
        listed = processes.list_procs(root)
        assert listed and listed[0]["name"] == "web"
        assert "running on http://127.0.0.1:54321" in processes.logs(root, "web")
    finally:
        out = processes.stop(root, "web")
    assert out.startswith("stopped web")
    assert processes.list_procs(root)[0]["status"] in ("stopped", "exited")
    # stopping an unknown name is an error, not a traceback
    assert processes.stop(root, "missing").startswith("ERROR")


def test_serve_action_is_gated_and_dispatches(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    script = _script(root, "54322")
    idx = Index(root, [])
    plain = Config()
    obs = tools.dispatch(idx, plain, {"action": "serve", "symbol": "list",
                                      "reasoning": ""})
    assert obs.startswith("ERROR: `serve` is disabled")

    cfg = Config()
    cfg.allow_serve = True
    cfg.serve_wait = 3
    try:
        started = tools.dispatch(idx, cfg, {
            "action": "serve", "symbol": "start", "command": _cmd(script),
            "file": "web", "reasoning": ""})
        assert started.startswith("started web")
        assert "url:" in started
        listing = tools.dispatch(idx, cfg, {"action": "serve",
                                            "symbol": "list",
                                            "reasoning": ""})
        assert "web" in listing
        logs = tools.dispatch(idx, cfg, {"action": "serve", "symbol": "logs",
                                         "file": "web", "reasoning": ""})
        assert "running on http://127.0.0.1:54322" in logs
    finally:
        tools.dispatch(idx, cfg, {"action": "serve", "symbol": "stop",
                                  "file": "web", "reasoning": ""})


def test_serve_action_appears_only_when_enabled():
    plain = tools.schema_for(Config())
    assert "serve" not in plain["properties"]["action"]["enum"]
    cfg = Config()
    cfg.allow_serve = True
    assert "serve" in tools.schema_for(cfg)["properties"]["action"]["enum"]
