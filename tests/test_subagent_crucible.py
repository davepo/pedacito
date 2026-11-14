"""Bug Hunt #11 crucible: subagent delegation, prefix fork, evidence handoff.

Stress cases for the bounded read-only helper (TECHNIQUES #28 and the
subagent handoff): the child's isolation matrix, the single-slot shared-prefix
fork shape, depth caps per endpoint kind, and the BUG-309 parity rule -- the
token-bounded evidence block, the parent's read ledger and the content hash
must all agree on exactly the lines the render delivers.

Hermetic: scripted clients only, a two-file project in tmp_path.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from a1_support import Scripted, make_project

from pedacito import subagent, verify
from pedacito.agent import Agent, sha256_text
from pedacito.config import Config

LOCAL_URL = "http://127.0.0.1:11434/v1"
HOSTED_URL = "https://api.deepseek.com/v1"

CRC_BODY = "".join(f"def f{i}(x):\n    return x + {i}\n" for i in range(40))


def _project(tmp_path, **cfg_attrs):
    root, idx, cfg = make_project(tmp_path, allow_subagents=True,
                                  log_events=False, log_sessions=False,
                                  **cfg_attrs)
    (root / "big.py").write_text(CRC_BODY, encoding="utf-8")
    from pedacito import index as index_mod
    idx = index_mod.reindex_file(idx, "big.py", cfg, save=False)
    return root, idx, cfg


# ------------------------------------------------------------- isolation
def test_child_config_is_read_only_and_leaves_parent_alone(tmp_path):
    _root, _idx, cfg = _project(
        tmp_path, allow_write=True, allow_shell=True, allow_browser=True,
        allow_screenshots=True, allow_mcp=True, enable_mcp=True,
        allow_fetch=True, enable_web=True, base_url=LOCAL_URL)
    cfg._backup_dir = "/tmp/x"
    cfg._task_reads = {"a.py": [(1, 2)]}
    child = subagent.child_config(cfg)
    for flag in ("allow_write", "allow_shell", "allow_browser",
                 "allow_screenshots", "allow_mcp", "enable_mcp",
                 "allow_fetch", "enable_web"):
        assert getattr(child, flag) is False, flag
    assert child.answer_mode == "rebuild"
    assert child._depth == 1 and child._interactive is False
    for scratch in ("_backup_dir", "_task_reads", "_task_allowlist",
                    "_task_mutations", "_task_preexisting", "_agent"):
        assert scratch not in child.__dict__, scratch
    # The parent keeps every capability and its scratch state.
    assert cfg.allow_write and cfg.enable_mcp and cfg.allow_shell
    assert cfg._backup_dir == "/tmp/x" and cfg._task_reads


def test_depth_caps_local_one_hosted_two(tmp_path):
    _root, idx, cfg = _project(tmp_path, base_url=LOCAL_URL)
    assert subagent.max_depth(cfg) == 1
    cfg._depth = 1
    out = subagent.tool_task(idx, cfg, Scripted([]), "anything")
    assert out.startswith("ERROR") and "depth" in out

    hosted = Config()
    hosted.base_url = HOSTED_URL
    hosted.model = "deepseek-flash"
    assert subagent.max_depth(hosted) == 2


# ----------------------------------------------------------- prefix fork
def test_tiny_local_child_forks_parent_prefix_byte_identical(tmp_path):
    _root, idx, cfg = _project(tmp_path, active_tier="tiny",
                               base_url=LOCAL_URL)
    client = Scripted(
        [{"action": "read", "file": "pricing.py", "start": 1, "end": 3},
         {"action": "done"}],
        "pricing.py:1-3 defines add_tax.")
    parent = Agent(idx, client, cfg)
    prefix = [{"role": "system", "content": "PARENT STABLE SYSTEM"},
              {"role": "user", "content": "PARENT STABLE USER0"}]
    parent._messages = [dict(m) for m in prefix]

    obs = subagent.tool_task(idx, cfg, client, "who defines add_tax?",
                             parent=parent)

    first = client.seen[0][0]
    assert first[:2] == [m["content"] for m in prefix]
    assert first[2] == "I will handle this bounded, read-only lookup."
    assert first[3].startswith("HELPER TASK")
    assert "who defines add_tax?" in first[3]
    # The parent's own conversation is untouched by the child's turns.
    assert parent._messages == prefix
    assert obs.startswith("HELPER RESULT")


def test_hosted_and_scout_and_large_do_not_fork(tmp_path):
    _root, idx, cfg = _project(tmp_path, base_url=HOSTED_URL,
                               model="deepseek-flash")
    assert subagent._shared_prefix_eligible(cfg) is False
    client = Scripted([{"action": "done"}], "nothing read")
    parent = Agent(idx, client, cfg)
    parent._messages = [{"role": "system", "content": "HOSTED SYSTEM"},
                        {"role": "user", "content": "HOSTED USER0"}]
    subagent.tool_task(idx, cfg, client, "q?", parent=parent)
    assert client.seen[0][0][0] != "HOSTED SYSTEM"

    _root2, idx2, cfg2 = _project(tmp_path / "s", active_tier="tiny",
                                  base_url=LOCAL_URL, scout_profile="scout")
    assert subagent._shared_prefix_eligible(cfg2) is False

    _root3, idx3, cfg3 = _project(tmp_path / "l", active_tier="large",
                                  base_url=LOCAL_URL)
    assert subagent._shared_prefix_eligible(cfg3) is False


# ------------------------------------------- BUG-309: clip/ledger parity
def test_clipped_span_credits_only_delivered_lines(tmp_path):
    _root, idx, cfg = _project(tmp_path, active_tier="tiny",
                               base_url=LOCAL_URL, max_read_tokens=60)
    client = Scripted(
        [{"action": "read", "file": "big.py", "start": 1, "end": 80},
         {"action": "done"}],
        "big.py:1-80 read.")
    parent = Agent(idx, client, cfg)
    parent.requirements = verify.RequirementTracker("fix f0 in big.py")

    obs = subagent.tool_task(idx, cfg, client, "read big.py fully",
                             parent=parent)

    copied = [s for s in parent.evidence.spans if s.file == "big.py"]
    assert copied, obs
    span = copied[-1]
    kept_lines = len(span.exact_text.split("\n"))
    assert span.end < 80                          # the budget really clipped
    assert span.end - span.start + 1 == kept_lines
    assert (span.start, span.end) in parent._read_spans["big.py"]
    assert span.content_sha == sha256_text(span.exact_text)
    rendered = parent._render_subagent_evidence([span])
    assert f"lines {span.start}-{span.end}:" in rendered
    assert "EXACT EVIDENCE ALREADY READ BY THE HELPER" in obs


def test_span_that_does_not_fit_gets_no_ledger_credit(tmp_path):
    _root, idx, cfg = _project(tmp_path, active_tier="tiny",
                               base_url=LOCAL_URL, max_read_tokens=40)
    parent = Agent(idx, Scripted([]), cfg)
    child = Agent(idx, Scripted([]), cfg)
    # Two spans: the first consumes the whole budget, the second must drop.
    child.evidence.record("big.py", 1, 40, CRC_BODY, 1)
    child.evidence.record("pricing.py", 1, 7,
                          (Path(_root) / "pricing.py")
                          .read_text(encoding="utf-8"), 2)
    copied = parent.absorb_subagent_evidence(child)
    assert [s.file for s in copied] == ["big.py"]
    assert "pricing.py" not in parent._read_spans
    assert not [s for s in parent.evidence.spans if s.file == "pricing.py"]


# ------------------------------------------------------- digest boundaries
def test_long_digest_is_clipped(tmp_path):
    _root, idx, cfg = _project(tmp_path)
    client = Scripted([{"action": "done"}], "x" * 9000)
    obs = subagent.tool_task(idx, cfg, client, "q?", parent=None)
    assert "[digest clipped]" in obs
    assert len(obs) < 9000


def test_child_crash_never_kills_the_parent(tmp_path):
    _root, idx, cfg = _project(tmp_path)

    class Boom:
        site = ""

        def chat_json(self, *a, **k):
            raise RuntimeError("boom")

        def chat(self, *a, **k):
            raise RuntimeError("boom")

    obs = subagent.tool_task(idx, cfg, Boom(), "q?", parent=None)
    assert obs.startswith("ERROR: helper agent failed")


def test_task_gates_disabled_empty_and_no_client(tmp_path):
    _root, idx, cfg = _project(tmp_path)
    assert subagent.tool_task(idx, cfg, Scripted([]), "")\
        .startswith("ERROR")
    assert subagent.tool_task(idx, cfg, None, "q?").startswith("ERROR")
    cfg.allow_subagents = False
    assert subagent.tool_task(idx, cfg, Scripted([]), "q?")\
        .startswith("ERROR")