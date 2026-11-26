"""`pdc ask --json` / `--verify-cmd`: the headless single-object verdict.

Every run below is stubbed (`cli.LLMClient` and `_get_index`), so no model
server and no network are involved. The stub answers through the same
`run_task` pipeline the real CLI uses, including the edit application,
telemetry aggregation and workspace diff.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import cli, events, index as index_mod  # noqa: E402
from pedacito.config import Config  # noqa: E402

KEYS = {"ok", "model", "provider", "turns", "elapsed_sec", "tokens",
        "files_changed", "verify_passed", "diff", "diff_truncated",
        "sandbox_effective"}
TOKEN_KEYS = {"prompt", "completion", "tok_per_sec"}

REPLY = ("Raised the limit.\n\n--- FILE: app.py\n<<<<<<< SEARCH\n"
         "LIMIT = 10\n=======\nLIMIT = 25\n>>>>>>> REPLACE\n")
SECRET = "sk-proj-abcdefghijklmnopqrstuvwx"


class StubClient:
    """The subset of LLMClient the gather/answer pipeline uses, plus the
    per-call usage record a real client folds into RunTelemetry."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.on_call = None
        self.on_start = None
        self.site = ""
        self.fewshot: list = []
        self.chars_per_token = 0.0
        self.truncated_field = ""
        self.stream_truncated = False
        self.step_truncated = False
        self.partial_text = ""
        self.actions = 0
        self.calls = 0
        self.reply = REPLY
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0, "reasoning_tokens": 0,
                      "cached_tokens": 0, "cache_write_tokens": 0,
                      "seconds": 0.0, "cost": 0.0}

    def check(self, warn: bool = True):
        return ""

    def warmup(self):
        return None

    def cancel(self):
        return None

    def chat_json(self, messages, schema, max_tokens=0, **kw):
        action = ({"action": "read", "file": "app.py", "pattern": "",
                   "symbol": "", "start": 1, "end": 200, "content": "",
                   "reasoning": "read it"}
                  if self.actions == 0 else
                  {"action": "done", "file": "", "pattern": "", "symbol": "",
                   "start": 0, "end": 0, "content": "", "reasoning": "done"})
        self.actions += 1
        self._record("gather")
        return action

    def chat(self, messages, max_tokens, schema=None, stream=False, **kw):
        self._record("answer")
        return self.reply

    def _record(self, site: str) -> None:
        self.calls += 1
        rec = {"site": site, "prompt_tokens": 100, "completion_tokens": 20,
               "total_tokens": 120, "reasoning_tokens": 0, "cached_tokens": 0,
               "cache_write_tokens": 0, "seconds": 0.25, "tok_per_sec": 80.0,
               "cost": 0.0, "model": self.cfg.model,
               "provider": "stub-provider", "finish": "stop"}
        events.note_call(rec)
        if self.on_call is not None:
            self.on_call(rec)
        self.stats["calls"] += 1
        self.stats["prompt_tokens"] += rec["prompt_tokens"]
        self.stats["completion_tokens"] += rec["completion_tokens"]
        self.stats["total_tokens"] += rec["total_tokens"]
        self.stats["seconds"] += rec["seconds"]


@pytest.fixture
def project(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "app.py").write_text(
        "LIMIT = 10\n\n\ndef check(n):\n    return n < LIMIT\n",
        encoding="utf-8")
    return proj


@pytest.fixture
def cfg_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PEDACITO_VRAM_GB", "8")
    path = tmp_path / "user.toml"
    path.write_text(
        'vram_gb = 8\nrefresh_summaries = 0\nlog_sessions = false\n'
        'show_telemetry = false\n', encoding="utf-8")
    return path


def _run(project, cfg_file, monkeypatch, *extra, client_holder=None,
         reply=None):
    cfg = Config()
    cfg.base_url = "http://127.0.0.1:9/v1"
    cfg.model = "stub-model"
    cfg.refresh_summaries = 0
    cfg.log_sessions = False
    cfg.show_telemetry = False
    idx = index_mod.build([str(project)], cfg, client=None, summarise=False,
                          root=project)

    def make_client(run_cfg):
        client = StubClient(run_cfg)
        if reply is not None:
            client.reply = reply
        if client_holder is not None:
            client_holder.append(client)
        return client

    monkeypatch.setattr(cli, "LLMClient", make_client)
    monkeypatch.setattr(cli, "_get_index", lambda *a, **k: idx)
    return cli.main(["ask", str(project), "-t", "raise LIMIT to 25",
                     "--config", str(cfg_file), "--model", "stub-model",
                     "--url", "http://127.0.0.1:9/v1", *extra])


def _one_object(out: str) -> dict:
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 1, f"stdout is not one JSON object: {out!r}"
    return json.loads(lines[0])


def test_ask_json_emits_exactly_one_complete_object(project, cfg_file,
                                                    monkeypatch, capsys):
    clients: list = []
    code = _run(project, cfg_file, monkeypatch, "--apply", "--json",
                client_holder=clients)
    captured = capsys.readouterr()
    assert code == 0
    obj = _one_object(captured.out)
    assert set(obj) == KEYS
    assert set(obj["tokens"]) == TOKEN_KEYS
    assert obj["ok"] is True
    assert obj["model"] == "stub-model"
    assert obj["provider"] == "stub-provider"
    assert obj["turns"] == clients[0].calls > 0
    assert isinstance(obj["elapsed_sec"], float) and obj["elapsed_sec"] > 0
    assert obj["tokens"]["prompt"] == 100 * clients[0].calls
    assert obj["tokens"]["completion"] == 20 * clients[0].calls
    assert obj["tokens"]["tok_per_sec"] > 0
    assert obj["files_changed"] == ["app.py"]
    assert obj["verify_passed"] is None          # no verification ran
    assert "+LIMIT = 25" in obj["diff"]
    assert obj["diff_truncated"] is False
    # Human output moved to stderr; the file was applied.
    assert "looking things up" in captured.err
    assert "LIMIT = 25" in (project / "app.py").read_text(encoding="utf-8")


def test_ask_json_refine_calls_count_toward_the_run(project, cfg_file,
                                                    monkeypatch, capsys):
    """--refine runs one spec-architect model call BEFORE the task. A single
    run-scoped collector must cover refine + task, so the JSON turns/tokens
    equal every call the client actually made. A second begin_run() reset
    after refine silently dropped the architect call from the totals."""
    clients: list = []
    code = _run(project, cfg_file, monkeypatch, "--apply", "--json",
                "--refine", "--no-plan", client_holder=clients)
    captured = capsys.readouterr()
    assert code == 0
    assert "prompt refined by the spec architect" in captured.err
    obj = _one_object(captured.out)
    assert clients[0].calls >= 2                 # refine + at least one task call
    assert obj["turns"] == clients[0].calls
    assert obj["tokens"]["prompt"] == 100 * clients[0].calls
    assert obj["tokens"]["completion"] == 20 * clients[0].calls


def test_ask_json_implies_no_review(project, cfg_file, monkeypatch, capsys):
    seen = {}
    real = cli.loop.run_task

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(cli.loop, "run_task", spy)
    assert _run(project, cfg_file, monkeypatch, "--apply", "--json",
                "--review") == 0
    capsys.readouterr()
    assert seen.get("review") is False


def test_verify_cmd_overrides_auto_verify_without_touching_config(
        project, cfg_file, monkeypatch, capsys):
    cfg_file.write_text(cfg_file.read_text(encoding="utf-8")
                        + 'auto_verify = "false"\n', encoding="utf-8")
    before = cfg_file.read_bytes()
    code = _run(project, cfg_file, monkeypatch, "--apply", "--json",
                "--verify-cmd", f'"{sys.executable}" --version')
    captured = capsys.readouterr()
    assert code == 0
    obj = _one_object(captured.out)
    assert obj["verify_passed"] is True            # `true`, not config `false`
    assert "[verify: passed]" in captured.err
    assert cfg_file.read_bytes() == before          # never written


def test_verify_cmd_records_a_sources_entry_in_memory():
    cfg = Config()
    cfg.auto_verify = "pytest"
    cfg.sources = {}
    cmd = cli._ask_verify_command(SimpleNamespace(verify_cmd="make check"), cfg)
    assert cmd == "make check"
    assert cfg.auto_verify == "make check"
    assert cfg.sources["auto_verify"] == "--verify-cmd"


def test_verify_passed_null_when_no_verification_ran(project, cfg_file,
                                                     monkeypatch, capsys):
    # verify_tests is off by default and no verify command is configured.
    assert _run(project, cfg_file, monkeypatch, "--apply", "--json") == 0
    obj = _one_object(capsys.readouterr().out)
    assert obj["verify_passed"] is None


def test_ask_json_dry_run_reports_no_applied_changes(project, cfg_file,
                                                     monkeypatch, capsys):
    assert _run(project, cfg_file, monkeypatch, "--json") == 0
    obj = _one_object(capsys.readouterr().out)
    assert obj["ok"] is True
    assert obj["files_changed"] == []
    assert obj["diff"] == ""
    assert obj["diff_truncated"] is False
    assert "LIMIT = 10" in (project / "app.py").read_text(encoding="utf-8")


def test_verify_cmd_failure_marks_ok_false_but_keeps_exit_code(
        project, cfg_file, monkeypatch, capsys):
    code = _run(project, cfg_file, monkeypatch, "--apply", "--json",
                "--verify-cmd", "false")
    assert code == 0                      # exit semantics are unchanged
    obj = _one_object(capsys.readouterr().out)
    assert obj["verify_passed"] is False
    assert obj["ok"] is False


def test_json_and_jsonl_are_rejected(project, cfg_file, capsys):
    code = cli.main(["ask", str(project), "-t", "x", "--config",
                     str(cfg_file), "--json", "--jsonl"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "--json and --jsonl are mutually exclusive" in captured.err


def test_diff_cap_sets_truncation():
    cfg = Config()
    cfg.redact_secrets = True
    report = cli._ask_json_report(
        cfg, {"calls": 1}, ok=True, files_changed=[], diff="x" * 250_000,
        verify_passed=None)
    assert report["diff_truncated"] is True
    assert len(report["diff"]) == cli.ASK_JSON_DIFF_LIMIT
    small = cli._ask_json_report(cfg, {}, ok=True, files_changed=[],
                                 diff="small", verify_passed=None)
    assert small["diff_truncated"] is False and small["turns"] == 0


def test_planted_secret_never_reaches_stdout(project, cfg_file, monkeypatch,
                                             capsys):
    # The task text carries the secret and the stubbed model echoes it into
    # the applied edit, so the only place it could surface is the JSON diff.
    reply = (f"Adding the token.\n\n--- FILE: app.py\n"
             "<<<<<<< SEARCH\nLIMIT = 10\n=======\n"
             f'TOKEN = "{SECRET}"\nLIMIT = 25\n>>>>>>> REPLACE\n')
    code = _run(project, cfg_file, monkeypatch, "--apply", "--json",
                "-t", f"add the API token {SECRET} to app.py", reply=reply)
    assert code == 0
    captured = capsys.readouterr()
    assert SECRET not in captured.out
    obj = _one_object(captured.out)
    assert "[REDACTED]" in obj["diff"]
    assert SECRET in (project / "app.py").read_text(encoding="utf-8")
