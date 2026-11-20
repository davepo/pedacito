"""Opt-in trajectory export: default off, one scrubbed ChatML file per run.

The direct tests drive `loop.run_task` with a scripted client (no server) and
use the same tiny indexed project as the replay/checkpoint suites; the CLI
test exercises the `pdc ask --save-trajectory` flag through `cli.main`.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import cli, events, index as index_mod, loop, security  # noqa: E402
from pedacito.agent import Agent  # noqa: E402
from pedacito.config import Config  # noqa: E402

TASK = "raise LIMIT in app.py"
SECRET = "sk-proj-abcdefghijklmnopqrstuvwx"
FILE_TEXT = (
    "LIMIT = 10\n"
    f'API_KEY = "{SECRET}"\n'
    "\n"
    "def check(n):\n"
    "    return n < LIMIT\n"
)
REPLY = "No changes are needed; the limit is intentional.\n"

META_KEYS = {"verify_passed", "harness_version", "edit_format", "model_id",
             "provider", "task_hash", "started_at", "elapsed_s", "tokens"}
TOKEN_KEYS = {"calls", "prompt_tokens", "completion_tokens", "total_tokens",
              "reasoning_tokens"}


class StubClient:
    """The gather/answer subset, with usage records folded into events just
    like the real client. Reads app.py once, then says `done`."""

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
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0, "reasoning_tokens": 0,
                      "cached_tokens": 0, "cache_write_tokens": 0,
                      "seconds": 0.0, "cost": 0.0}

    def check(self, warn: bool = True):
        return ""

    def warmup(self):
        return None

    def loaded_models(self):
        return None

    def cancel(self):
        return None

    def provider_label(self):
        return "stub-provider"

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text or "") // 4)

    def chat_json(self, messages, schema, max_tokens=0, **kw):
        if self.actions == 0:
            step = {"action": "read", "file": "app.py", "pattern": "",
                    "symbol": "", "start": 1, "end": 200, "content": "",
                    "reasoning": "read it"}
        else:
            step = {"action": "done", "file": "", "pattern": "", "symbol": "",
                    "start": 0, "end": 0, "content": "",
                    "reasoning": "done"}
        self.actions += 1
        self._record("gather")
        return step

    def chat(self, messages, max_tokens, schema=None, stream=False, **kw):
        self._record("answer")
        return REPLY

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


def _make_run(tmp_path, *, record=False):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "app.py").write_text(FILE_TEXT, encoding="utf-8")
    cfg = Config()
    cfg.log_events = False
    cfg.log_sessions = False
    cfg.show_telemetry = False
    cfg.sources = {}
    cfg.record_trajectories = record
    idx = index_mod.build([str(root)], cfg, client=None, summarise=False,
                          root=root, verbose=False)
    client = StubClient(cfg)
    agent = Agent(idx, client, cfg)
    return root, idx, cfg, client, agent


def _files(root: Path) -> list[Path]:
    return sorted((root / ".pedacito" / "trajectories").glob("*.json"))


def test_default_off_writes_nothing(tmp_path):
    root, idx, cfg, client, agent = _make_run(tmp_path)
    tr = loop.run_task(root, cfg, client, agent, idx, TASK, apply=False)
    assert tr is not None
    assert _files(root) == []
    assert not (root / ".pedacito" / "trajectories").exists()


def test_config_knob_captures_without_a_flag(tmp_path):
    root, idx, cfg, client, agent = _make_run(tmp_path, record=True)
    loop.run_task(root, cfg, client, agent, idx, TASK, apply=False)

    files = _files(root)
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert set(data) == {"metadata", "messages"}
    meta = data["metadata"]
    assert META_KEYS <= set(meta)
    assert meta["verify_passed"] is None
    assert meta["harness_version"] == "0.3.1"
    assert meta["edit_format"] == "quote"
    assert meta["model_id"] == cfg.model
    assert meta["provider"] == "stub-provider"
    assert meta["task_hash"] == hashlib.sha256(
        TASK.encode("utf-8")).hexdigest()
    assert TASK not in json.dumps(meta)
    assert isinstance(meta["elapsed_s"], float)
    assert meta["started_at"].endswith("+00:00")
    assert TOKEN_KEYS <= set(meta["tokens"])
    assert meta["tokens"]["calls"] == client.calls

    messages = data["messages"]
    assert messages and all(isinstance(m, dict) for m in messages)
    assert all(m.get("role") in {"system", "user", "assistant"}
               for m in messages)
    assert all("content" in m for m in messages)
    roles = {m["role"] for m in messages}
    assert {"system", "user", "assistant"} <= roles
    contents = [m.get("content") for m in messages
                if isinstance(m.get("content"), str)]
    assert any(c.startswith("RESULT:") for c in contents), (
        "the tool result was not part of the transcript")
    assert any("read app.py" in c or "app.py" in c for c in contents)
    # The final answer is present exactly once as the last assistant turn.
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == REPLY


def test_planted_api_key_in_tool_result_is_never_persisted(tmp_path):
    root, idx, cfg, client, agent = _make_run(tmp_path, record=True)
    loop.run_task(root, cfg, client, agent, idx, TASK, apply=False)

    files = _files(root)
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    data = json.loads(text)
    # The read really carried the key into a tool result...
    joined = "\n".join(m.get("content") or "" for m in data["messages"]
                       if isinstance(m.get("content"), str))
    assert "RESULT:" in joined
    assert "API_KEY" in joined
    # ...but the file has neither the secret nor any scanner finding.
    assert SECRET not in text
    assert security.scan_text_for_secrets(text) == []
    assert "[REDACTED]" in text


def test_capture_failure_does_not_fail_the_task(tmp_path):
    root, idx, cfg, client, agent = _make_run(tmp_path, record=True)
    # A regular file where the directory must go makes the capture mkdir fail.
    state = root / ".pedacito"
    state.mkdir(parents=True, exist_ok=True)
    (state / "trajectories").write_text("not a directory", encoding="utf-8")

    tr = loop.run_task(root, cfg, client, agent, idx, TASK, apply=False)
    assert tr is not None
    assert (state / "trajectories").is_file()          # unchanged
    assert _files(root) == []


# --------------------------------------------------------------- CLI flag ---

def _cli_run(project: Path, cfg_file: Path, monkeypatch, *extra):
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
        client.cfg = run_cfg
        return client

    monkeypatch.setattr(cli, "LLMClient", make_client)
    monkeypatch.setattr(cli, "_get_index", lambda *a, **k: idx)
    return cli.main(["ask", str(project), "-t", TASK,
                     "--config", str(cfg_file), "--model", "stub-model",
                     "--url", "http://127.0.0.1:9/v1", *extra])


def test_ask_save_trajectory_flag_writes_one_valid_file(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "app.py").write_text(FILE_TEXT, encoding="utf-8")
    cfg_file = tmp_path / "user.toml"
    cfg_text = ("vram_gb = 8\nrefresh_summaries = 0\nlog_sessions = false\n"
                "show_telemetry = false\n")
    cfg_file.write_text(cfg_text, encoding="utf-8")

    code = _cli_run(project, cfg_file, monkeypatch, "--save-trajectory")
    assert code == 0

    files = sorted((project / ".pedacito" / "trajectories").glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert set(data) == {"metadata", "messages"}
    assert META_KEYS <= set(data["metadata"])
    assert data["metadata"]["task_hash"] == hashlib.sha256(
        TASK.encode("utf-8")).hexdigest()
    assert data["messages"][-1]["role"] == "assistant"
    assert security.scan_text_for_secrets(
        files[0].read_text(encoding="utf-8")) == []
    # The opt-in flag forces capture in memory only; the file is untouched.
    assert cfg_file.read_text(encoding="utf-8") == cfg_text


def test_config_file_knob_captures_without_the_flag(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "app.py").write_text(FILE_TEXT, encoding="utf-8")
    cfg_file = tmp_path / "user.toml"
    cfg_file.write_text(
        "vram_gb = 8\nrefresh_summaries = 0\nlog_sessions = false\n"
        "show_telemetry = false\nrecord_trajectories = true\n",
        encoding="utf-8")

    code = _cli_run(project, cfg_file, monkeypatch)
    assert code == 0
    files = sorted((project / ".pedacito" / "trajectories").glob("*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text(encoding="utf-8"))["metadata"][
        "task_hash"] == hashlib.sha256(TASK.encode("utf-8")).hexdigest()


def test_default_off_ask_writes_no_trajectory(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "app.py").write_text(FILE_TEXT, encoding="utf-8")
    cfg_file = tmp_path / "user.toml"
    cfg_file.write_text(
        "vram_gb = 8\nrefresh_summaries = 0\nlog_sessions = false\n"
        "show_telemetry = false\n", encoding="utf-8")

    code = _cli_run(project, cfg_file, monkeypatch)
    assert code == 0
    assert not (project / ".pedacito" / "trajectories").exists()
