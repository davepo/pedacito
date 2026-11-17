"""Regression suite for the New-project lifecycle, the live LLM heartbeat,
hosted-token telemetry, actionable READMEs and always-creative refinement.

Covers the five deliverable scenarios end to end:
  1. a multi-module prompt auto-enables Complex scaffolding (complex=False);
  2. POST /api/new switches the workspace BEFORE scaffolding (see also
     test_gui.py) and the frontend resets the one-shot Complex checkbox;
  3. hosted usage {"prompt_tokens": 500, "completion_tokens": 300} populates
     the run telemetry across BOTH Ask and New;
  4. the README a build ships is real documentation whose commands match the
     generated CLI's actual flags;
  5. refine_prompt always samples creatively (0.6 local, native hosted) and
     returns clean spec text.
Plus the TUI parity: /new switches the active root first, llm_start shows a
waiting line, and every run closes with the same `[run: ...]` aggregate the
CLI prints.
"""
# ruff: noqa: F811  (imported pytest fixtures are re-bound as test parameters)
from __future__ import annotations

import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import config, events, loop, scaffold, spec  # noqa: E402
from pedacito import llm as llm_mod  # noqa: E402
from pedacito import tui as tui_mod  # noqa: E402
from pedacito.config import Config  # noqa: E402
from pedacito.gui import core, jobs  # noqa: E402
from pedacito.local_server import LocalHTTPServer  # noqa: E402
from test_gui import gui, fake_port  # noqa: F401,E402 (shared fixtures)

FOUR_MODULE_PROMPT = ("Build a task tracker across cli.py, storage.py, "
                      "models.py and report.py")


def _read_events(root: Path) -> list[dict]:
    out = []
    for path in sorted((root / ".pedacito" / "events").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# 1. Auto-Complex: a multi-module prompt upgrades itself, an explicit check
#    is never overridden, and a single-file prompt stays standard.
# ---------------------------------------------------------------------------

def test_detect_complex_scaffold_multi_module():
    assert loop.detect_complex_scaffold(FOUR_MODULE_PROMPT)
    assert loop.detect_complex_scaffold(
        "a multi-module architecture with separate layers")
    assert not loop.detect_complex_scaffold("fix the typo in README.md")
    assert not loop.detect_complex_scaffold("")


def test_mode_suitability_keeps_auto_for_multi_module(tmp_path):
    verdict = loop.evaluate_mode_suitability(FOUR_MODULE_PROMPT, tmp_path,
                                             "auto")
    assert verdict["mismatch"] is False
    assert verdict["recommended_mode"] == "auto"


def _tracker_plan(spec, client, cfg, root, *a, **k):
    return scaffold.Plan(
        "Tracker", "a task tracker",
        [scaffold.PlannedFile(p, "") for p in
         ("cli.py", "storage.py", "models.py", "report.py")],
        spec=spec)


def test_run_new_auto_enables_complex_and_emits_the_exact_notice(
        tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_complex(root, cfg, client, idx, plan, spec, **kw):
        seen["complex"] = True
        return 0

    monkeypatch.setattr(scaffold, "build_plan", _tracker_plan)
    monkeypatch.setattr(scaffold, "build_interface", lambda *a, **k: {})
    monkeypatch.setattr(scaffold, "_run_complex", fake_complex)
    cfg = config.load(require_hardware=False)
    cfg.log_sessions = False
    root = tmp_path / "tracker"
    code = scaffold.run_new(root, cfg, None,
                            scaffold.Spec(FOUR_MODULE_PROMPT),
                            apply=True, complex=False, verbose=False)
    assert code == 0
    assert seen.get("complex") is True
    assert scaffold.COMPLEX_AUTO_NOTICE == (
        "Auto-enabled Complex scaffolding "
        "(detected multi-module architecture).")
    assert scaffold.COMPLEX_AUTO_NOTICE in capsys.readouterr().err
    # the notice landed in the NEW project's event log, verbatim
    records = _read_events(root)
    assert any(r.get("kind") == "complex_auto"
               and r.get("text") == scaffold.COMPLEX_AUTO_NOTICE
               for r in records), records


def test_explicit_complex_is_never_overridden_or_re_announced(
        tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_complex(root, cfg, client, idx, plan, spec, **kw):
        seen["complex"] = True
        return 0

    monkeypatch.setattr(scaffold, "build_plan", _tracker_plan)
    monkeypatch.setattr(scaffold, "build_interface", lambda *a, **k: {})
    monkeypatch.setattr(scaffold, "_run_complex", fake_complex)
    cfg = config.load(require_hardware=False)
    cfg.log_sessions = False
    root = tmp_path / "explicit"
    # a SINGLE-file prompt with the checkbox ON: still Complex, but no
    # auto-upgrade notice (the user already chose it).
    code = scaffold.run_new(root, cfg, None,
                            scaffold.Spec("a tiny cli in main.py"),
                            apply=True, complex=True, verbose=False)
    assert code == 0 and seen.get("complex") is True
    assert "Auto-enabled" not in capsys.readouterr().err
    assert not any(r.get("kind") == "complex_auto"
                   for r in _read_events(root))


def test_single_file_prompt_stays_standard(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_build(client, cfg, idx, plan, **kw):
        seen["standard"] = True
        return {"done": [f.path for f in plan.sources], "noop": [],
                "failed": [], "skipped": []}

    def fake_complex(*a, **k):
        seen["complex"] = True
        return 0

    monkeypatch.setattr(scaffold, "build_plan",
                        lambda spec, client, cfg, root, *a, **k: scaffold.Plan(
                            "Tiny", "a tiny cli",
                            [scaffold.PlannedFile("main.py", "entry")],
                            spec=spec))
    monkeypatch.setattr(scaffold, "build_project", fake_build)
    monkeypatch.setattr(scaffold, "build_interface", lambda *a, **k: {})
    monkeypatch.setattr(scaffold, "fix_import_errors", lambda *a, **k: 0)
    monkeypatch.setattr(scaffold, "_web_contract_gate", lambda *a, **k: [])
    monkeypatch.setattr(scaffold, "verify_and_fix",
                        lambda *a, **k: (True, ""))
    monkeypatch.setattr(scaffold, "_run_complex", fake_complex)
    cfg = config.load(require_hardware=False)
    cfg.log_sessions = False
    root = tmp_path / "tiny"
    code = scaffold.run_new(root, cfg, None,
                            scaffold.Spec("a tiny cli in main.py"),
                            apply=True, complex=False, verbose=False)
    assert code == 0
    assert seen.get("standard") is True
    assert "complex" not in seen
    assert "Auto-enabled" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 2/3. Hosted usage: the run collector, the shared aggregate line, and the
#      GUI end-to-end across Ask AND New against a usage-reporting server.
# ---------------------------------------------------------------------------

HOSTED_USAGE = {"prompt_tokens": 500, "completion_tokens": 300,
                "total_tokens": 800,
                "completion_tokens_details": {"reasoning_tokens": 120}}


def test_run_collector_aggregates_hosted_usage_and_speed():
    cfg = Config()
    cfg.model = "deepseek-reasoner"
    cfg.base_url = "https://api.deepseek.com/v1"
    client = llm_mod.LLMClient(cfg)
    events.begin_run()
    try:
        rec = client._account(dict(HOSTED_USAGE), None, 2.0, 0)
    finally:
        summary = events.end_run()
    assert rec["total_tokens"] == 800
    assert rec["reasoning_tokens"] == 120
    assert rec["tok_per_sec"] == 150.0
    assert rec["provider"] == "DeepSeek"
    assert summary["calls"] == 1
    assert summary["prompt_tokens"] == 500
    assert summary["completion_tokens"] == 300
    assert summary["total_tokens"] == 800
    assert summary["reasoning_tokens"] == 120
    assert summary["elapsed_s"] == 2.0
    assert summary["tok_per_sec"] == 150.0
    assert summary["model"] == "deepseek-reasoner"
    assert summary["provider"] == "DeepSeek"
    # the shared CLI/TUI aggregate line tells the whole story in one line
    line = events.run_line(summary)
    assert line.startswith("[run: 1 calls, 500 in / 300 out")
    assert "800 total" in line and "120 reasoning" in line
    assert "150.0 tok/s" in line
    assert "DeepSeek/deepseek-reasoner" in line
    assert events.run_line({}) == ""
    assert events.run_line(events.end_run()) == ""


def test_server_timings_win_over_wall_clock_speed():
    cfg = Config()
    client = llm_mod.LLMClient(cfg)
    rec = client._account({"prompt_tokens": 100, "completion_tokens": 50},
                          {"predicted_per_second": 42.5}, 10.0, 0)
    assert rec["tok_per_sec"] == 42.5


class _UsageHandler(BaseHTTPRequestHandler):
    """An OpenAI-compatible endpoint that reports REAL hosted usage on every
    reply (the stock fake_llm answers schema calls with `usage: {}`)."""

    def log_message(self, *a):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._json({"object": "list", "data": [{"id": "fake-model"}]})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        usage = dict(self.server.usage)
        rf = req.get("response_format") or {}
        if rf.get("type") in ("json_schema", "json_object"):
            action = {"action": "done", "file": "", "pattern": "",
                      "symbol": "", "start": 0, "end": 0, "content": "",
                      "reasoning": "done"}
            self._json({"choices": [{"message": {"content": json.dumps(action)},
                                     "finish_reason": "stop"}],
                        "usage": usage})
            return
        text = "A tiny demo answer."
        if req.get("stream"):
            gate = getattr(self.server, "gate", None)
            if gate is not None:
                gate.wait(15)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            frame = {"choices": [{"delta": {"content": text},
                                  "finish_reason": None}]}
            self.wfile.write(("data: " + json.dumps(frame) + "\n\n").encode())
            self.wfile.write(("data: " + json.dumps(
                {"choices": [{"delta": {}, "finish_reason": "stop"}]})
                + "\n\n").encode())
            if (req.get("stream_options") or {}).get("include_usage"):
                self.wfile.write(("data: " + json.dumps(
                    {"choices": [], "usage": usage}) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        self._json({"choices": [{"message": {"content": text},
                                 "finish_reason": "stop"}], "usage": usage})


@pytest.fixture
def usage_server():
    srv = LocalHTTPServer(("127.0.0.1", 0), _UsageHandler)
    srv.usage = dict(HOSTED_USAGE)
    srv.gate = None
    thread = threading.Thread(target=srv.serve_forever, daemon=True,
                              name="usage-fake-server")
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(5)


def _setup(gui, port):
    gui.post("/api/settings/setup",
             {"base_url": f"http://127.0.0.1:{port}/v1",
              "model": "fake-model", "vram_gb": "8"})


@pytest.mark.network
def test_hosted_usage_populates_telemetry_for_ask_and_new(
        gui, usage_server, monkeypatch):
    port = usage_server.server_address[1]
    _setup(gui, port)
    # warmup's probe call would double the aggregate; this test is about
    # usage propagation, so keep the run to exactly the answer/build call.
    monkeypatch.setattr(llm_mod.LLMClient, "warmup", lambda self: None)

    # ASK (the project-less scratchpad): one streamed answer call.
    status, _ = gui.post("/api/ask", {"text": "what is 2+2?"})
    assert status == 202 and gui.wait_idle(30)
    state = gui.get("/api/state")[1]
    assert not state["error"], state["messages"]
    t = state["telemetry"]
    assert t["last_prompt_tokens"] == 500
    assert t["last_completion_tokens"] == 300
    assert t["last_total_tokens"] == 800
    assert t["last_reasoning_tokens"] == 120
    assert t["last_tok_per_s"] > 0
    answer = [m for m in state["messages"] if m.get("kind") == "answer"][-1]
    stats = answer["stats"]
    assert stats["calls"] == 1
    assert stats["prompt_tokens"] == 500
    assert stats["completion_tokens"] == 300
    assert stats["total_tokens"] == 800
    assert stats["reasoning_tokens"] == 120
    assert stats["tok_per_sec"] > 0
    assert stats["model"] == "fake-model"
    assert stats["provider"] == "local server"

    # NEW: the build's calls aggregate into the same run-scoped telemetry.
    target = gui.tmp / "hosted-new"

    def fake_build(root, cfg, client, spec, **kw):
        client.site = "build"
        client.chat([{"role": "user", "content": "build it"}],
                    max_tokens=100)
        (root / "main.py").write_text("def main():\n    return 0\n",
                                      encoding="utf-8")
        return 0

    monkeypatch.setattr(jobs.scaffold, "run_new", fake_build)
    status, _ = gui.post("/api/new", {"text": "a tiny demo",
                                      "project": str(target),
                                      "build": "build"})
    assert status == 202 and gui.wait_idle(30)
    state = gui.get("/api/state")[1]
    assert not state["error"], state["messages"]
    result = [m for m in state["messages"] if m.get("kind") == "result"][-1]
    stats = result["stats"]
    assert stats["calls"] == 1
    assert stats["prompt_tokens"] == 500
    assert stats["completion_tokens"] == 300
    assert stats["total_tokens"] == 800
    assert stats["reasoning_tokens"] == 120
    assert stats["elapsed_s"] > 0
    assert stats["tok_per_sec"] > 0
    assert stats["model"] == "fake-model"
    # the per-call event carried the hosted numbers into the session log too
    records = _read_events(target)
    calls = [r for r in records if r.get("kind") == "call"]
    assert calls and calls[-1]["prompt_tokens"] == 500
    assert calls[-1]["completion_tokens"] == 300
    assert calls[-1]["total_tokens"] == 800


@pytest.mark.network
def test_llm_heartbeat_lights_up_while_thinking_and_clears_on_call(
        gui, usage_server, monkeypatch):
    port = usage_server.server_address[1]
    _setup(gui, port)
    monkeypatch.setattr(llm_mod.LLMClient, "warmup", lambda self: None)
    gate = threading.Event()
    usage_server.gate = gate
    try:
        status, _ = gui.post("/api/ask", {"text": "take your time"})
        assert status == 202

        def thinking(s):
            return s if s.get("llm_call", {}).get("started") else None

        state = gui.wait_for(thinking, 15)
        assert state, "the heartbeat never lit up while the model thought"
        call = state["llm_call"]
        assert call["provider"] == "local server"
        assert call["model"] == "fake-model"
        assert call["phase"] == "answer"
        assert call["started"] > 0
        # the job is still running: the indicator belongs to a live request
        assert state["job"]
    finally:
        gate.set()
    assert gui.wait_idle(30)
    state = gui.get("/api/state")[1]
    assert state["llm_call"] == {}          # the matching call cleared it
    assert not state["error"], state["messages"]


def test_gui_state_heartbeat_lifecycle():
    """llm_start sets the pulse payload; the matching call clears it; a new
    job never inherits a stale indicator."""
    core.STATE.note_event("llm_start", {"provider": "DeepSeek",
                                        "model": "deepseek-reasoner",
                                        "phase": "plan"})
    call = core.STATE.status()["llm_call"]
    assert call["provider"] == "DeepSeek"
    assert call["model"] == "deepseek-reasoner"
    assert call["phase"] == "plan"
    assert call["started"] > 0
    core.STATE.note_event("call", {"prompt_tokens": 500,
                                   "completion_tokens": 300})
    assert core.STATE.status()["llm_call"] == {}
    core.STATE.note_event("llm_start", {"provider": "OpenAI", "model": "gpt"})
    job = core.STATE.begin_job("task", "fresh run")
    assert job is not None
    assert core.STATE.status()["llm_call"] == {}
    core.STATE.end_job(job)


def test_frontend_heartbeat_and_one_shot_complex_reset():
    js = (Path(__file__).resolve().parents[1] / "pedacito" / "gui" / "static"
          / "app.js").read_text(encoding="utf-8")
    # the pulsing indicator + its 1s elapsed timer
    assert "function LlmHeartbeat()" in js
    assert "setInterval(() => tick((x) => x + 1), 1000)" in js
    assert "is thinking & generating..." in js
    assert "s elapsed)" in js
    # a real New build follows the server's immediate switch and resets the
    # Complex checkbox (it never persists into the next run)
    assert "S.mode === 'new' && body.build !== 'preview'" in js
    assert "S.opts.new.complex = false;" in js
    assert "saveOpts();" in js
    css = (Path(__file__).resolve().parents[1] / "pedacito" / "gui" / "static"
           / "app.css").read_text(encoding="utf-8")
    assert ".llm-heartbeat" in css and ".llm-pulse" in css


# ---------------------------------------------------------------------------
# 4. The README a build ships is real documentation, grounded in the CLI
#    source that was actually generated.
# ---------------------------------------------------------------------------

CLI_SOURCE = '''\
"""Todo CLI."""
import argparse


def build_parser():
    parser = argparse.ArgumentParser(prog="todo")
    sub = parser.add_subparsers(dest="cmd")
    add = sub.add_parser("add", help="add a task")
    add.add_argument("text")
    add.add_argument("--priority", default="normal")
    lst = sub.add_parser("list", help="list tasks")
    lst.add_argument("--limit", type=int, default=10)
    done = sub.add_parser("done", help="mark a task done")
    done.add_argument("id", type=int)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.cmd:
        parser.print_help()
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

README_REPLY = (
    "--- FILE: README.md\n"
    "<<<<<<< SEARCH\n"
    "=======\n"
    "# Todo CLI\n"
    "\n"
    "## Overview & Architecture\n"
    "\n"
    "A tiny task tracker. `cli.py` is the entry point (argparse subcommands "
    "`add`, `list`, `done`); `test_cli.py` holds the unittest suite.\n"
    "\n"
    "## Quick Start\n"
    "\n"
    "```bash\n"
    "python3 cli.py add \"Buy milk\" --priority high\n"
    "python3 cli.py list --limit 5\n"
    "python3 cli.py done 1\n"
    "```\n"
    "\n"
    "Running with no arguments prints the help.\n"
    "\n"
    "## Running Tests\n"
    "\n"
    "```bash\n"
    "python3 -m unittest -v\n"
    "```\n"
    ">>>>>>> REPLACE\n"
)


class _DocsClient:
    """Answers the docs pass with a full-file README block; records prompts."""

    def __init__(self, reply=README_REPLY):
        self.reply = reply
        self.calls = []

    def chat(self, messages, **kw):
        self.calls.append(messages)
        return self.reply


def _docs_project(tmp_path):
    root = tmp_path / "todo"
    root.mkdir()
    plan = scaffold.Plan(
        "Todo", "a tiny task cli",
        [scaffold.PlannedFile("cli.py", "entry point"),
         scaffold.PlannedFile("test_cli.py", "unittest suite"),
         scaffold.PlannedFile("README.md", "project README")],
        spec=scaffold.Spec("a tiny todo cli"))
    scaffold.apply_skeleton(root, plan, dry_run=False, verbose=False)
    (root / "cli.py").write_text(CLI_SOURCE, encoding="utf-8")
    (root / "test_cli.py").write_text(
        "import unittest\n\n\nclass TestCli(unittest.TestCase):\n"
        "    def test_help(self):\n"
        "        from cli import main\n"
        "        self.assertEqual(main([]), 0)\n", encoding="utf-8")
    cfg = config.load(require_hardware=False)
    cfg.backup = False
    idx = scaffold.skeleton_index(root, plan, cfg)
    return root, plan, cfg, idx


def test_build_docs_replaces_the_stub_with_a_real_readme(tmp_path):
    root, plan, cfg, idx = _docs_project(tmp_path)
    stub = (root / "README.md").read_text(encoding="utf-8")
    client = _DocsClient()
    summary = scaffold.build_docs(client, cfg, idx, plan, verbose=False)
    assert summary["done"] == ["README.md"]
    assert not summary["failed"]
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert readme != stub
    # the three mandated sections, with copy-pasteable commands
    for section in ("## Overview & Architecture", "## Quick Start",
                    "## Running Tests"):
        assert section in readme
    assert "python3 -m unittest -v" in readme
    # every example flag is a REAL flag of the generated cli.py
    assert "--priority" in CLI_SOURCE and "--priority" in readme
    assert "--limit" in CLI_SOURCE and "--limit" in readme
    for sub in ("add", "list", "done"):
        assert f"cli.py {sub}" in readme
    # the docs prompt was grounded in the actual CLI source, not invention
    assert client.calls, "the docs pass never asked the model"
    prompt = "\n".join(m["content"] for m in client.calls[0])
    assert "CLI SOURCE" in prompt
    assert 'add_parser("--priority"' in prompt or "--priority" in prompt
    assert "add_parser" in prompt


def test_docs_and_build_prompts_mandate_the_usability_contract():
    for section in ("Overview & Architecture", "Quick Start",
                    "Running Tests"):
        assert section in scaffold.DOCS_SYSTEM
    assert "EVERY CLI subcommand" in scaffold.DOCS_SYSTEM
    assert "python3 -m unittest -v" in scaffold.DOCS_SYSTEM
    # zero-arg CLI prints help instead of crashing -- in BOTH build paths
    assert "parser.print_help()" in scaffold.BUILD_SYSTEM
    assert "parser.print_help()" in scaffold.AUTO_IMPLEMENT_SYSTEM


def test_complex_build_documents_the_extra_readme_too(tmp_path, monkeypatch):
    """The Complex loop implements sources in fresh contexts; the README the
    skeleton added as an EXTRA (not in the plan) still gets the docs pass."""
    root = tmp_path / "complex-todo"
    root.mkdir()
    plan = scaffold.Plan(
        "Todo", "a tiny task cli",
        [scaffold.PlannedFile("cli.py", "entry point"),
         scaffold.PlannedFile("test_cli.py", "unittest suite")],
        spec=scaffold.Spec("a tiny todo cli"))
    scaffold.apply_skeleton(root, plan, dry_run=False, verbose=False)
    (root / "cli.py").write_text(CLI_SOURCE, encoding="utf-8")
    (root / "test_cli.py").write_text(
        "import unittest\n\n\nclass TestCli(unittest.TestCase):\n"
        "    def test_help(self):\n"
        "        from cli import main\n"
        "        self.assertEqual(main([]), 0)\n", encoding="utf-8")
    cfg = config.load(require_hardware=False)
    cfg.backup = False
    cfg.log_sessions = False
    idx = scaffold.skeleton_index(root, plan, cfg)
    stub = (root / "README.md").read_text(encoding="utf-8")

    monkeypatch.setattr(loop, "run_loop", lambda *a, **k: 0)
    monkeypatch.setattr(scaffold, "_web_contract_gate", lambda *a, **k: [])
    monkeypatch.setattr(scaffold, "verify_and_fix",
                        lambda *a, **k: (True, ""))
    client = _DocsClient()
    code = scaffold._run_complex(root, cfg, client, idx, plan, plan.spec,
                                 verbose=False)
    assert code == 0
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert readme != stub
    assert "## Quick Start" in readme
    assert "python3 -m unittest -v" in readme
    assert client.calls, "the extra README never reached the docs pass"


def test_failed_docs_reach_the_complex_exit_code(tmp_path, monkeypatch):
    root = tmp_path / "complex-bad-docs"
    root.mkdir()
    plan = scaffold.Plan(
        "Todo", "a tiny task cli",
        [scaffold.PlannedFile("cli.py", "entry point")],
        spec=scaffold.Spec("a tiny todo cli"))
    scaffold.apply_skeleton(root, plan, dry_run=False, verbose=False)
    (root / "cli.py").write_text(CLI_SOURCE, encoding="utf-8")
    cfg = config.load(require_hardware=False)
    cfg.backup = False
    cfg.log_sessions = False
    idx = scaffold.skeleton_index(root, plan, cfg)

    class _NoDocs:
        def chat(self, messages, **kw):
            return "I cannot write documentation."

    monkeypatch.setattr(loop, "run_loop", lambda *a, **k: 0)
    monkeypatch.setattr(scaffold, "_web_contract_gate", lambda *a, **k: [])
    monkeypatch.setattr(scaffold, "verify_and_fix",
                        lambda *a, **k: (True, ""))
    code = scaffold._run_complex(root, cfg, _NoDocs(), idx, plan, plan.spec,
                                 verbose=False)
    assert code == 1, "a stub README must not ship under a green exit code"


# ---------------------------------------------------------------------------
# 5. refine_prompt: ALWAYS creative sampling, architect persona, clean text.
# ---------------------------------------------------------------------------

REFINED_REPLY = {
    "objective": "A tiny todo CLI that stores tasks in tasks.json.",
    "files": [{"path": "cli.py", "action": "create",
               "responsibility": "argparse entry point"}],
    "contracts": ["def main(argv=None) -> int"],
    "requirements": ["1. `add TEXT` appends a task.",
                     "2. No arguments prints the help."],
    "edge_cases": ["Empty task text is rejected with a clear message."],
    "verification": "python3 -m unittest -v",
}


def _capture_post(monkeypatch, captured):
    def fake_post(url, **kw):
        captured.append(kw.get("json"))
        data = {"choices": [{"message": {"content": json.dumps(REFINED_REPLY)},
                             "finish_reason": "stop"}], "usage": {}}
        return SimpleNamespace(status_code=200, encoding="utf-8",
                               headers={}, json=lambda: data,
                               close=lambda: None)

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)


def test_refine_samples_creatively_on_local_even_with_toggle_off(
        tmp_path, monkeypatch):
    cfg = Config()
    cfg.base_url = "http://127.0.0.1:8080/v1"
    cfg.model = "qwen3-coder"
    cfg.creative_mode = False          # the toggle is OFF...
    cfg.temperature = 0.0              # ...and the base sampling is greedy
    captured = []
    _capture_post(monkeypatch, captured)
    client = llm_mod.LLMClient(cfg)
    out = spec.refine_prompt("a tiny todo cli in cli.py", tmp_path, cfg,
                             client)
    assert out["ok"] is True
    assert captured, "refine never called the model"
    # ...yet the refine call sampled at 0.6, and the override was restored
    assert captured[0]["temperature"] == 0.6
    assert client._temp_override is None
    system = captured[0]["messages"][0]["content"]
    assert "Principal Product Architect" in system
    assert "VERBATIM" in system
    # clean spec text: no JSON wrapper, no schema keys, real content
    refined = out["refined"]
    assert refined.strip()
    assert not refined.lstrip().startswith("{")
    assert "objective" in refined.lower() or "Objective" in refined
    assert "cli.py" in refined


def test_refine_omits_temperature_for_hosted_native_sampling(
        tmp_path, monkeypatch):
    cfg = Config()
    cfg.base_url = "https://api.deepseek.com/v1"
    cfg.model = "deepseek-reasoner"
    cfg.creative_mode = False
    captured = []
    _capture_post(monkeypatch, captured)
    client = llm_mod.LLMClient(cfg)
    out = spec.refine_prompt("a tiny todo cli in cli.py", tmp_path, cfg,
                             client)
    assert out["ok"] is True
    assert captured
    # hosted APIs tune their own sampling: no temperature key at all
    assert "temperature" not in captured[0]
    assert client._temp_override is None


def test_refine_fallback_is_clean_spec_text(tmp_path):
    # no model at all: the deterministic architect still lands a usable spec
    out = spec.refine_prompt("a tiny todo cli in cli.py", tmp_path, Config(),
                             None)
    assert out["ok"] is True
    assert out["original"] == "a tiny todo cli in cli.py"
    refined = out["refined"]
    assert refined.strip()
    assert not refined.lstrip().startswith("{")
    assert "cli.py" in refined


def test_refine_system_carries_the_domain_defaults():
    # the prompt wraps at ~76 cols; compare on collapsed whitespace
    system = " ".join(spec.REFINE_SYSTEM.split())
    assert "Principal Product Architect" in system
    # Web/Canvas: zero-build, no unrequested package.json
    assert "index.html" in system and "styles.css" in system
    assert "script.js" in system
    assert "no package.json" in system
    # Desktop GUI: stdlib tkinter, dark flat, keyboard, headless-safe tests
    assert "tkinter" in system and "dark flat" in system
    assert "keyboard" in system and "headless-safe" in system
    # Multi-file: exact modules, --help subcommands, README matching flags
    assert "--help" in system
    assert "README examples that match the real flags" in system


# ---------------------------------------------------------------------------
# TUI parity: /new switches first, llm_start shows a waiting line, and every
# run closes with the same `[run: ...]` aggregate the CLI prints.
# ---------------------------------------------------------------------------

class _TuiFakeClient:
    """Enough of LLMClient for the TUI's lazy backend, without a network."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "cached_tokens": 0, "cache_write_tokens": 0,
                      "seconds": 0.0, "cost": 0.0, "by_site": {}}
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_tok_per_s = 0.0
        self.avg_tok_per_s = 0.0
        self.context_tokens = 32768
        self.active_model_role = "worker"

    def check(self, warn=True):
        return ""

    def warmup(self):
        pass

    def cancel(self):
        pass

    def telemetry(self):
        return {"last_prompt_tokens": self.last_prompt_tokens,
                "last_completion_tokens": self.last_completion_tokens,
                "last_tok_per_s": self.last_tok_per_s,
                "avg_tok_per_s": self.avg_tok_per_s,
                "context_tokens": self.context_tokens,
                "context_utilization_pct": 0.0,
                "session_prompt_tokens": self.stats["prompt_tokens"],
                "session_completion_tokens": self.stats["completion_tokens"],
                "cache_hit_pct": 0.0,
                "active_model_role": self.active_model_role}


def _tui_cfg():
    cfg = Config()
    cfg.model = "deepseek-reasoner"
    cfg.base_url = "https://api.deepseek.com/v1"
    cfg.context_tokens = 32768
    cfg.log_sessions = False
    return cfg


def _simulate_call(root, cfg):
    """One finished hosted call, exactly as llm._account would fold it."""
    events.get(root, cfg).emit(
        "llm_start", provider="DeepSeek", model="deepseek-reasoner",
        phase="build")
    events.note_call({"site": "build", "prompt_tokens": 500,
                      "completion_tokens": 300, "total_tokens": 800,
                      "reasoning_tokens": 120, "seconds": 2.0,
                      "tok_per_sec": 150.0, "model": "deepseek-reasoner",
                      "provider": "DeepSeek"})


def test_tui_new_switches_root_before_scaffolding(tmp_path, monkeypatch):
    cfg = _tui_cfg()
    client = _TuiFakeClient(cfg)
    target = tmp_path / "brand-new"
    seen = {}

    def fake_run_new(root, cfg_, client_, spec_, **kw):
        # while the build runs, the session must ALREADY live in the target
        seen["root"] = root
        seen["apply"] = kw.get("apply")
        seen["complex"] = kw.get("complex")
        seen["session_root"] = session.root
        seen["elog_root"] = Path(session.elog.root)
        seen["index_files"] = len(session.idx.files) if session.idx else -1
        _simulate_call(root, cfg_)
        (root / "main.py").write_text("def main():\n    return 0\n",
                                      encoding="utf-8")
        return 0

    monkeypatch.setattr(scaffold, "run_new", fake_run_new)
    out = io.StringIO()
    session = tui_mod.TuiSession(tmp_path, cfg, client=client, headless=True,
                                 stdout=out)
    try:
        session.handle_line(f"/new {target} a tiny demo cli")
        assert session.root == target.resolve()
        assert seen["root"] == target.resolve()
        assert seen["session_root"] == target.resolve()
        assert seen["elog_root"] == target.resolve()
        assert seen["apply"] is True
        assert seen["complex"] is False      # auto-detection decides, not a toggle
        assert seen["index_files"] == 0      # empty index BEFORE the build
        # the event log rebound: the build's events land in the NEW project
        assert any(r.get("kind") == "llm_start"
                   for r in _read_events(target))
        text = out.getvalue()
        assert f"switched workspace to {target.resolve()}" in text
        assert "[llm] Waiting for DeepSeek/deepseek-reasoner..." in text
        assert "Project created at" in text
        # the closing aggregate matches the CLI's run line
        assert "[run: 1 calls, 500 in / 300 out" in text
        assert "800 total" in text and "120 reasoning" in text
        assert "150.0 tok/s" in text
        assert "DeepSeek/deepseek-reasoner" in text
        # the post-build index sees the scaffolded file
        assert session.idx is not None
        assert "main.py" in session.idx.files
    finally:
        session.close()


def test_tui_new_refuses_a_non_empty_target(tmp_path):
    cfg = _tui_cfg()
    target = tmp_path / "occupied"
    target.mkdir()
    (target / "main.py").write_text("print('hi')\n", encoding="utf-8")
    out = io.StringIO()
    session = tui_mod.TuiSession(tmp_path, cfg, client=_TuiFakeClient(cfg),
                                 headless=True, stdout=out)
    try:
        session.handle_line(f"/new {target} a tiny demo cli")
        assert session.root == tmp_path.resolve()    # never switched
        assert "is not empty" in out.getvalue()
        session.handle_line("/new")
        assert "Usage: /new <dir> <prompt>" in out.getvalue()
    finally:
        session.close()


def test_tui_task_and_auto_close_with_the_run_aggregate(tmp_path, monkeypatch):
    cfg = _tui_cfg()
    client = _TuiFakeClient(cfg)

    def fake_run_task(root, cfg_, client_, agent, idx, task, **kw):
        _simulate_call(root, cfg_)
        return None

    def fake_generate_tasks(goal, client_, cfg_, root=None, **kw):
        return ["Implement cli.py"]

    def fake_run_loop(root, cfg_, client_, idx, plan_file, **kw):
        _simulate_call(root, cfg_)
        return 0

    monkeypatch.setattr(tui_mod.loop, "run_task", fake_run_task)
    monkeypatch.setattr(tui_mod.loop, "generate_tasks", fake_generate_tasks)
    monkeypatch.setattr(tui_mod.loop, "run_loop", fake_run_loop)
    out = io.StringIO()
    session = tui_mod.TuiSession(tmp_path, cfg, client=client, headless=True,
                                 stdout=out)
    try:
        session.handle_line("change something")
        text = out.getvalue()
        assert "[run: 1 calls, 500 in / 300 out" in text
        assert "150.0 tok/s" in text
        out.truncate(0)
        out.seek(0)
        session.handle_line("/auto! build a tiny todo cli")
        text = out.getvalue()
        assert "auto loop finished" in text
        assert "[run: 1 calls, 500 in / 300 out" in text
        assert "DeepSeek/deepseek-reasoner" in text
    finally:
        session.close()


def test_tui_llm_start_and_complex_auto_events(tmp_path):
    cfg = _tui_cfg()
    session = tui_mod.TuiSession(tmp_path, cfg, headless=True,
                                 stdout=io.StringIO())
    try:
        session._apply_event("llm_start", {"provider": "DeepSeek",
                                           "model": "deepseek-reasoner",
                                           "phase": "plan"})
        assert ("[llm] Waiting for DeepSeek/deepseek-reasoner... (plan)"
                in session.activity)
        # repeated heartbeats are never deduped away (every request shows)
        session._apply_event("llm_start", {"provider": "DeepSeek",
                                           "model": "deepseek-reasoner",
                                           "phase": "plan"})
        assert sum("[llm] Waiting" in a for a in session.activity) == 2
        session._apply_event("complex_auto",
                             {"text": scaffold.COMPLEX_AUTO_NOTICE})
        assert scaffold.COMPLEX_AUTO_NOTICE in session.activity
        assert any(scaffold.COMPLEX_AUTO_NOTICE in text
                   for _kind, text in session.transcript)
    finally:
        session.close()


def test_tui_help_lists_the_new_command(tmp_path):
    assert "/new" in tui_mod._SLASH_COMMANDS
    assert "/new DIR PROMPT" in tui_mod.HELP_TEXT
