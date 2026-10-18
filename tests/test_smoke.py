"""Offline smoke tests: no LLM server required.

Covers the config-simplification surface (templates parse, tier presets
apply, setup error carries a hint), the edit-block parser/renamer bugs
("nothing to review" on rename tasks), and the coverage ledger in agent.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import config as config_mod
from pedacito import edits
from pedacito.agent import Agent
from pedacito.llm import LMStudio, LLMError


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


FAKE = Path(__file__).resolve().parent / "fake_llm.py"


# --------------------------------------------------------------------- config
def test_templates_are_valid_toml():
    for name in ("TEMPLATE", "REFERENCE_TEMPLATE"):
        text = getattr(config_mod, name)
        data = tomllib.loads(text)          # must not raise
        assert isinstance(data, dict)
    # The starter template must stay short -- that was the whole point of
    # the simplification.
    assert len(config_mod.TEMPLATE.splitlines()) < 80
    assert "base_url" in config_mod.TEMPLATE
    assert "model" in config_mod.TEMPLATE


def test_tier_preset_applies_without_profiles(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        '[server]\nbase_url = "http://localhost:9/v1"\n'
        'model = "m"\ntier = "small"\n', encoding="utf-8")
    cfg = config_mod.load(root=tmp_path, explicit=p, use_env=False)
    assert cfg.max_steps == config_mod.PRESETS["small"]["max_steps"]
    assert cfg.active_tier == "small"
    assert "tier 'small' preset" in cfg.sources["max_steps"]


def test_bad_tier_fails_loudly(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text('tier = "huge"\n', encoding="utf-8")
    with pytest.raises(config_mod.ConfigError):
        config_mod.load(root=tmp_path, explicit=p, use_env=False)


def test_default_profile_missing_gives_setup_hint(tmp_path, monkeypatch):
    fake = tmp_path / "user.toml"
    fake.write_text('default_profile = "gpu"\n', encoding="utf-8")
    monkeypatch.setattr(config_mod, "USER_CONFIG", fake)
    with pytest.raises(config_mod.ConfigError) as e:
        config_mod.load(root=tmp_path, use_env=False)
    assert "--setup" in str(e.value)


def test_discover_servers_and_interactive_setup_exist():
    assert callable(config_mod.discover_servers)
    assert callable(config_mod.interactive_setup)


# --------------------------------------------------------------------- edits
RENAME_REPLY = """\
Renaming the module:

--- FILE: new_name.py
<<<<<<< SEARCH
=======
print("hi")
>>>>>>> REPLACE

--- FILE: old_name.py
<<<<<<< SEARCH
print("hi")
=======
>>>>>>> REPLACE
"""


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_parse_edits_roundtrip():
    blocks = edits.parse_edits(RENAME_REPLY)
    assert [b.file for b in blocks] == ["new_name.py", "old_name.py"]
    assert blocks[0].search == "" and "hi" in blocks[0].replace
    assert "hi" in blocks[1].search and blocks[1].replace.strip() == ""


def test_rename_pair_applies(tmp_path):
    """The exact regression behind 'nothing to review' on rename tasks:
    deleting the source file's full content (EOF block without trailing
    newline in SEARCH) must pair with the create block. The protocol moves
    the source's REAL bytes into the new path and removes the old path, so
    the source contributes no Result of its own -- asserting on disk state
    is the honest check."""
    _write(tmp_path / "old_name.py", 'print("hi")')     # no trailing \n!
    blocks = edits.parse_edits(RENAME_REPLY)
    results = edits.apply_edits(tmp_path, blocks, backup=False, dry_run=True)
    by_file = {r.file: r for r in results}
    assert by_file["new_name.py"].ok, by_file["new_name.py"].message
    assert by_file["new_name.py"].renames_from == ("old_name.py",), \
        "rename pairing dropped -- this is the 'nothing to review' bug"
    # And for real this time: the move happens, old path is removed.
    results = edits.apply_edits(tmp_path, blocks, backup=False, dry_run=False)
    assert all(r.ok for r in results), [r.message for r in results]
    assert (tmp_path / "new_name.py").is_file()
    assert not (tmp_path / "old_name.py").exists()
    assert 'print("hi")' in (tmp_path / "new_name.py").read_text()


def test_search_block_matches_at_eof_with_newline_tail(tmp_path):
    _write(tmp_path / "a.py", "x = 1\n")
    reply = ('--- FILE: a.py\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n'
             '>>>>>>> REPLACE\n')
    results = edits.apply_edits(tmp_path, edits.parse_edits(reply),
                                backup=False, dry_run=False)
    assert results[0].ok, results[0].message
    assert (tmp_path / "a.py").read_text() == "x = 2\n"


# ------------------------------------------------------------------ budgets
def test_default_budgets_fit_small_windows():
    """The historical failure: defaults sized for 32k+ models made every
    step overflow an 8k local server (and DeepSeek's real window). The
    default context_tokens must be small enough that the per-step prompt
    budget is sane without any user tuning."""
    cfg = config_mod.Config()
    assert cfg.context_tokens <= 8192
    usable = cfg.context_tokens - cfg.reserve_tokens
    assert cfg.max_tokens_answer <= usable
    assert cfg.max_tokens_step <= usable
    # Every shipped tier preset must survive its own sanity check.
    for name, vals in config_mod.PRESETS.items():
        c = config_mod.Config()
        for k, v in vals.items():
            setattr(c, k, v)
        assert not config_mod.budget_warnings(c), f"preset {name} is over-budget"


def test_truncated_json_gets_token_limit_error():
    """A thinking model that spends its whole budget on reasoning returns
    no JSON at all and finish_reason='length'. chat_json must report THAT
    (with actionable advice), not the generic 'would not produce valid JSON'
    message that sends users hunting for a broken grammar.

    The salvage path is covered by test_truncated_action_is_recovered: when
    enough of the object survived to repair, the step is reused instead of
    costing another round trip."""
    p2 = _free_port()
    # No FAKE_ACTIONS: the server's default script is read-then-done, so the
    # first reply is truncated-but-recoverable; the retry escalation then
    # meets the plain `done` object, which parses fine. That proves the
    # recovery path end to end without needing a second server.
    env = dict(os.environ, FAKE_PORT=str(p2),
               FAKE_ACTIONS=json.dumps([{"action": "read", "file": "a.txt",
                                         "start": 1, "end": 2,
                                         "content": "trunc",
                                         "_finish": "length"},
                                        {"action": "done", "reasoning": ""}]),
               FAKE_REPLY="")
    proc = subprocess.Popen([sys.executable, str(FAKE)], env=env,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", p2), 0.5):
                    break
            except OSError:
                if proc.poll() is not None:
                    pytest.fail("fake_llm exited early")
                time.sleep(0.1)
        cfg = config_mod.Config()
        cfg.base_url = f"http://127.0.0.1:{p2}/v1"
        cfg.model = "fake-model"
        client = LMStudio(cfg)
        obj = client.chat_json([{"role": "user", "content": "go"}],
                               {"type": "object"}, max_tokens=64)
        assert obj.get("action") == "read", \
            f"salvage failed: {obj!r}"
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_unrecoverable_truncation_raises_token_limit_error():
    """When NOTHING parseable survives (a thinking model that emitted only
    prose before the cutoff), chat_json must raise the token-limit LLMError
    with 'token limit' and 'max_tokens_step' in the message -- not the
    generic grammar-failure text."""
    p2 = _free_port()
    s = socket.socket()
    s.bind(("127.0.0.1", p2))
    s.listen(8)
    import threading

    def serve():
        while True:
            try:
                conn, _ = s.accept()
            except OSError:
                return
            try:
                data = b""
                while b"\r\n\r\n" not in data:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                body_len = 0
                head, _, rest = data.partition(b"\r\n\r\n")
                for line in head.split(b"\r\n"):
                    if line.lower().startswith(b"content-length:"):
                        body_len = int(line.split(b":")[1])
                while len(rest) < body_len:
                    rest += conn.recv(4096)
                payload = {
                    "choices": [{"message": {
                        "content": "I need to think about this very carefully "
                                   "because the task requires reading several "
                                   "files and considering many options..."},
                        "finish_reason": "length"}], "usage": {}}
                b_ = json.dumps(payload).encode()
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json"
                             b"\r\nContent-Length: " + str(len(b_)).encode()
                             + b"\r\n\r\n" + b_)
            finally:
                conn.close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    try:
        cfg = config_mod.Config()
        cfg.base_url = f"http://127.0.0.1:{p2}/v1"
        cfg.model = "fake-model"
        client = LMStudio(cfg)
        with pytest.raises(LLMError) as exc:
            client.chat_json([{"role": "user", "content": "go"}],
                             {"type": "object"}, max_tokens=64)
        msg = str(exc.value)
        assert "token limit" in msg.lower(), msg
        assert "max_tokens_step" in msg, msg
    finally:
        s.close()


# --------------------------------------------------------------------- agent
class FakeIndex:
    def __init__(self, root, files):
        self.root = root
        self.files = list(files)


def test_coverage_ledger_normalises_paths(tmp_path):
    """A read whose output head carries a './'-prefixed or differently-cased
    path must still count as coverage for the index key 'pkg/mod.py' -- the
    ledger/lookup mismatch here made tasks look 'unread' and killed the
    review stage."""
    from pedacito import tools
    from pedacito.agent import _READ_HEAD_RX

    _write(tmp_path / "pkg" / "mod.py", "a = 1\nb = 2\n")
    index = FakeIndex(tmp_path, ["pkg/mod.py"])
    index.resolve_file = lambda f: "pkg/mod.py"   # minimal Index stub
    cfg = config_mod.Config()

    agent = Agent(index, client=None, cfg=cfg)
    agent._read_spans = {}
    obs = tools.tool_read(index, cfg, "./PKG/mod.py", 1, 2)
    m = _READ_HEAD_RX.match(obs)
    assert m, f"tool_read output not recognised: {obs!r}"
    key = m.group(1).strip().lower().lstrip("./")
    agent._read_spans[key] = [(int(m.group(2)), int(m.group(3)))]
    assert agent._compute_unread("edit pkg/mod.py") == []


def test_compute_unread_reports_missing(tmp_path):
    _write(tmp_path / "pkg" / "mod.py", "a = 1\nb = 2\n")
    agent = Agent(FakeIndex(tmp_path, ["pkg/mod.py"]), client=None,
                  cfg=config_mod.Config())
    agent._read_spans = {}
    missing = agent._compute_unread("edit pkg/mod.py")
    assert len(missing) == 1 and "not yet read" in missing[0]
