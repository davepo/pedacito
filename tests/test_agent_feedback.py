"""Post-edit feedback loop (F1), compacted-lookup stubs + the read digest
(F5), and the task write allowlist ledgers gather arms (F7)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import diagnostics
from pedacito import index as index_mod
from pedacito.agent import Agent, _compact_result_marker
from pedacito.config import Config


def _content_findings(marker: str = "BROKEN"):
    """quick_findings stand-in: a file is 'broken' while it contains the
    marker, so findings follow the edits instead of a real linter."""
    def fake(root, files):
        out = {}
        for rel in files or []:
            try:
                text = (Path(root) / rel).read_text(encoding="utf-8")
            except OSError:
                out[rel] = []
                continue
            out[rel] = ([f"{rel}:1:1: E999 {marker} present"]
                        if marker in text else [])
        return out
    return fake


class _FixClient:
    """Scripted answer-phase client: pops one reply per chat() call and
    records the user prompt it was sent."""
    stream_truncated = False

    def __init__(self, replies=()):
        self.replies = list(replies)
        self.prompts = []

    def chat(self, messages, max_tokens, stream=False, **kw):
        self.prompts.append(messages[-1]["content"])
        return self.replies.pop(0) if self.replies else "NOTHING MORE"


def _agent(tmp_path, client, files=None, **cfg_kw) -> Agent:
    files = files or {"app.py": "def f():\n    return 1\n"}
    for rel, src in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src, encoding="utf-8")
    cfg = Config()
    cfg.backup = False
    for k, v in cfg_kw.items():
        setattr(cfg, k, v)
    idx = index_mod.build([str(tmp_path)], cfg, client=None, summarise=False,
                          root=tmp_path)
    return Agent(idx, client, cfg)


_BROKEN_REPLY = ("--- FILE: app.py\n<<<<<<< SEARCH\ndef f():\n    return 1\n"
                 "=======\nBROKEN = 1\n>>>>>>> REPLACE\n")


def _block(search: str, replace: str) -> str:
    return (f"--- FILE: app.py\n<<<<<<< SEARCH\n{search}\n=======\n"
            f"{replace}\n>>>>>>> REPLACE\n")


# ------------------------------------------------------- post-edit feedback
def test_post_edit_check_asks_for_a_fix_and_applies_it(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "quick_findings", _content_findings())
    fix = _block("BROKEN = 1", "FIXED = 1")
    client = _FixClient([fix])
    agent = _agent(tmp_path, client)
    results, final = agent.apply(_BROKEN_REPLY, dry_run=False,
                                 repair_rounds=2, task="break it")
    assert all(r.ok for r in results)
    assert "FIXED = 1" in (tmp_path / "app.py").read_text()
    assert len(client.prompts) == 1
    prompt = client.prompts[0]
    assert "NEW problems" in prompt
    assert "app.py:1:1: E999 BROKEN present" in prompt   # rendered finding
    assert "FILE: app.py" in prompt                      # source window
    assert fix.strip() in final                          # fix joined the reply


def test_post_edit_check_stops_on_nothing_more(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "quick_findings", _content_findings())
    client = _FixClient(["NOTHING MORE"])
    agent = _agent(tmp_path, client)
    results, _final = agent.apply(_BROKEN_REPLY, dry_run=False,
                                  repair_rounds=3, task="t")
    assert all(r.ok for r in results)
    assert "BROKEN" in (tmp_path / "app.py").read_text()
    assert len(client.prompts) == 1        # the loop did not re-ask


def test_post_edit_check_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "quick_findings", _content_findings())
    client = _FixClient([])
    agent = _agent(tmp_path, client, post_edit_check=False)
    results, _final = agent.apply(_BROKEN_REPLY, dry_run=False,
                                  repair_rounds=2, task="t")
    assert all(r.ok for r in results)
    assert client.prompts == []
    assert "BROKEN" in (tmp_path / "app.py").read_text()


def test_post_edit_check_never_rebaselines_its_own_damage(tmp_path,
                                                          monkeypatch):
    """First-capture-wins: had a fix round adopted the current (broken) state
    as its baseline, the damage would launder away and the loop would stop
    after one ask. Instead it keeps asking, bounded by repair_rounds."""
    monkeypatch.setattr(diagnostics, "quick_findings", _content_findings())
    client = _FixClient([_block("BROKEN = 1", "BROKEN = 11"),
                         _block("BROKEN = 11", "BROKEN = 111")])
    agent = _agent(tmp_path, client)
    agent.apply(_BROKEN_REPLY, dry_run=False, repair_rounds=2, task="t")
    assert len(client.prompts) == 2
    assert "BROKEN = 111" in (tmp_path / "app.py").read_text()


def test_post_edit_check_skips_dry_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "quick_findings", _content_findings())
    client = _FixClient([])
    agent = _agent(tmp_path, client)
    agent.apply(_BROKEN_REPLY, dry_run=True, repair_rounds=2, task="t")
    assert client.prompts == []
    assert "BROKEN" not in (tmp_path / "app.py").read_text()


# ----------------------------------------------------- fix-prompt sourcing
def test_diagnostics_source_shows_numbered_windows(tmp_path):
    src = "".join(f"line_{i} = {i}\n" for i in range(1, 21))
    agent = _agent(tmp_path, _FixClient([]), files={"app.py": src})
    out = agent._diagnostics_source({"app.py": ["app.py:10:3: boom"]})
    assert out.startswith("FILE: app.py")
    assert "... lines 6-14 ..." in out
    assert "   10| line_10 = 10" in out


def test_diagnostics_source_without_a_position_shows_the_head(tmp_path):
    agent = _agent(tmp_path, _FixClient([]))
    out = agent._diagnostics_source({"app.py": ["weird finding, no position"]})
    assert "... lines 1-3 ..." in out      # 2 lines + the trailing split item
    assert "    1| def f():" in out


def test_diagnostics_source_caps_the_file_count(tmp_path):
    files = {f"f{i}.py": "x = 1\n" for i in range(8)}
    agent = _agent(tmp_path, _FixClient([]), files=files)
    found = {f"f{i}.py": [f"f{i}.py:1:1: boom"] for i in range(8)}
    out = agent._diagnostics_source(found)
    assert out.count("FILE: f") == 6


def test_diagnostics_source_survives_a_missing_file(tmp_path):
    agent = _agent(tmp_path, _FixClient([]))
    assert agent._diagnostics_source({"gone.py": ["gone.py:1:1: x"]}) == \
        "(source unavailable)"


# ------------------------------------------------------ compaction stubs
def test_compact_result_marker_stubs_a_read():
    content = "RESULT:\npkg/app.py lines 4-8:\n    4| def helper(value):\n"
    assert _compact_result_marker(content) == \
        "RESULT:\n[compacted earlier lookup: read pkg/app.py:4-8]"


def test_compact_result_marker_keeps_the_head_of_other_results():
    content = "RESULT:\n3 definition(s) matching 'serv'\n  a.py:1-2 [x]\n"
    assert _compact_result_marker(content) == (
        "RESULT:\n[compacted earlier lookup: 3 definition(s) matching "
        "'serv']")


def test_compact_result_marker_clips_long_heads():
    out = _compact_result_marker("RESULT:\n" + "y" * 400 + "\n")
    head = out.split("lookup: ", 1)[1]
    assert head.endswith("...]")
    assert len(head) <= 161                  # 160 chars + the closing bracket


def test_compact_result_marker_of_an_empty_result():
    assert _compact_result_marker("RESULT:\n") == ""


# ------------------------------------------------------------ read digest
_SRC = ("def helper(value):\n    return value + 1\n\n\n"
        "class Server:\n    def start(self):\n        return 1\n\n\n"
        "def run():\n    return 2\n")


def test_read_digest_folds_spans_and_names_symbols(tmp_path):
    agent = _agent(tmp_path, _FixClient([]), files={"pkg/app.py": _SRC})
    digest = agent._read_digest({"pkg/app.py": [(1, 2), (5, 7)]})
    assert digest.startswith("--- digest (files read during gather)")
    assert "- pkg/app.py lines 1-2, 5-7 (helper, Server)" in digest
    assert "run" not in digest               # outside every read span


def test_read_digest_caps_ranges_and_handles_empty(tmp_path):
    agent = _agent(tmp_path, _FixClient([]), files={"pkg/app.py": _SRC})
    assert agent._read_digest({}) == ""
    many = [(i * 4 + 1, i * 4 + 2) for i in range(10)]
    digest = agent._read_digest({"pkg/app.py": many})
    assert ", ..." in digest                 # more than 8 folded ranges


# ------------------------------------------------------ write allowlist
def test_write_allowlist_reflects_config_and_ledgers(tmp_path):
    agent = _agent(tmp_path, _FixClient([]), files={"pkg/app.py": _SRC})
    assert agent._write_allowlist() is None          # feature off
    agent.cfg.auto_allowlist = True
    assert agent._write_allowlist() is None          # gather never armed it
    agent.cfg._task_reads = {"pkg/app.py": [(1, 2)]}
    agent.cfg._task_allowlist = {"README.md"}
    assert agent._write_allowlist() == {"pkg/app.py", "README.md"}


def test_gather_arms_allowlist_ledgers_and_digests_reads(tmp_path):
    class Scripted:
        step_truncated = False

        def __init__(self):
            self.n = 0

        def chat_json(self, messages, schema, max_tokens, **kw):
            self.n += 1
            if self.n == 1:
                return {"action": "read", "file": "pkg/app.py",
                        "start": 1, "end": 2, "reasoning": "look"}
            return {"action": "done", "reasoning": "finished"}

    client = Scripted()
    agent = _agent(tmp_path, client, files={"pkg/app.py": _SRC})
    gathered = agent.gather("tidy up pkg/app.py", verbose=False)
    assert agent.cfg._task_reads == {"pkg/app.py": [(1, 2)]}
    assert "pkg/app.py" in agent.cfg._task_allowlist     # mined from title
    assert "--- digest (files read during gather)" in gathered
    assert "- pkg/app.py lines 1-2 (helper)" in gathered
