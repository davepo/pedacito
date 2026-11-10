"""Repair-trajectory collapse and constrained subagent handoff.

The first half proves a successful repair pops the failed draft/diagnostic
turns back to the snapshot taken before the hunk request. The second half
proves a `task` child's exact evidence crosses back into the parent's evidence
store + read ledger, so the parent can edit those lines without re-reading.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from a1_support import Scripted, make_project

from pedacito import index as index_mod
from pedacito import roles as roles_mod
from pedacito import subagent, tools, verify
from pedacito.agent import Agent, sha256_text
from pedacito.config import Config
from pedacito.edits import parse_edits


HUNK_ADD_TAX = (
    "--- FILE: pricing.py SYMBOL add_tax\n<<<<<<< SEARCH\n=======\n"
    "def add_tax(price, rate):\n"
    "    return round(price * (1 + rate), 3)\n>>>>>>> REPLACE\n")
HUNK_TOTAL = (
    "--- FILE: cart.py SYMBOL total\n<<<<<<< SEARCH\n=======\n"
    "def total(items):\n    return sum(items) + 1\n>>>>>>> REPLACE\n")
INTENTS = [
    {"file": "pricing.py", "symbol": "add_tax",
     "intended_change": "round tax to three decimals"},
    {"file": "cart.py", "symbol": "total",
     "intended_change": "add one to the sum"},
]


def _two_file_project(tmp_path, **cfg_attrs):
    root, idx, cfg = make_project(tmp_path, **cfg_attrs)
    (root / "cart.py").write_text(
        "def total(items):\n    return sum(items)\n", encoding="utf-8")
    idx = index_mod.reindex_file(idx, "cart.py", cfg, save=False)
    return root, idx, cfg


class RefModel:
    """Scripted answer-phase model that keeps the actual message-list refs."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []
        self.snapshots: list[list[dict]] = []
        self.stream_truncated = False

    def chat(self, messages, max_tokens, **kw):
        self.calls.append(messages)
        self.snapshots.append([dict(m) for m in messages])
        return self.replies.pop(0)


# ---------------------------------------------- repair trajectory collapse
def test_successful_hunk_repair_pops_failed_turns_back_to_p(tmp_path):
    _root, idx, cfg = _two_file_project(tmp_path)
    bad = HUNK_ADD_TAX.replace(
        "    return round(price * (1 + rate), 3)\n",
        "    return round(price * (1 + rate), 3\n")
    client = RefModel([bad, HUNK_ADD_TAX, HUNK_TOTAL])
    agent = Agent(idx, client, cfg)

    results, reply, findings = agent.apply_edit_intents(
        "Change add_tax and total", "gathered source", INTENTS, verbose=False)

    assert [r.ok for r in results] == [True, True]
    assert not [f for f in findings if f.severity == "error"]
    assert "[EDITED pricing.py:add_tax (repaired 1x)]" in reply
    assert "[EDITED cart.py:total]" in reply
    # The rejected draft + diagnostic were popped from the successful call's
    # live list, leaving the two-message stable prefix plus the compact receipt.
    repaired_call = client.calls[1]
    assert len(repaired_call) == 3
    assert repaired_call[0]["content"] == client.calls[0][0]["content"]
    assert repaired_call[1]["content"] == client.calls[0][1]["content"]
    assert repaired_call[-1]["content"] == \
        "[EDITED pricing.py:add_tax (repaired 1x)]"
    assert not any("WAS REJECTED" in m["content"] for m in repaired_call)
    # The next symbol never sees the broken draft or its parser diagnostic.
    next_prompt = client.snapshots[2][-1]["content"]
    assert "WAS REJECTED" not in next_prompt
    assert "return round(price * (1 + rate), 3\n" not in next_prompt
    assert "[EDITED pricing.py:add_tax (repaired 1x)]" in next_prompt
    # Evidence/read-ledger state is untouched by the message-tail collapse.
    assert agent.evidence.spans
    assert agent._read_spans.get("pricing.py")
    assert agent._read_spans.get("cart.py")


def test_collapse_repair_turns_can_be_disabled(tmp_path):
    _root, idx, cfg = _two_file_project(tmp_path, collapse_repair_turns=False)
    bad = HUNK_ADD_TAX.replace(
        "    return round(price * (1 + rate), 3)\n",
        "    return round(price * (1 + rate), 3\n")
    client = RefModel([bad, HUNK_ADD_TAX, HUNK_TOTAL])
    agent = Agent(idx, client, cfg)

    agent.apply_edit_intents("Change add_tax and total", "", INTENTS,
                             verbose=False)

    # Ablation keeps the historical failed turns in the live list.
    failed_call = client.calls[1]
    assert any("WAS REJECTED" in m["content"] for m in failed_call)


# ------------------------------------- EvidenceSpan + read-ledger handoff
def test_subagent_evidence_credits_parent_ledger_and_skips_second_read(tmp_path):
    _root, idx, cfg = make_project(tmp_path, allow_subagents=True,
                                   log_events=False, log_sessions=False)
    parent = Agent(idx, Scripted([]), cfg)
    parent.requirements = verify.RequirementTracker(
        "edit add_tax in pricing.py")
    parent.read_any = False
    child_client = Scripted(
        [{"action": "read", "file": "pricing.py", "start": 1, "end": 7},
         {"action": "done"}],
        "pricing.py:1-3 defines add_tax.")

    obs = subagent.tool_task(
        idx, cfg, child_client,
        "read add_tax in pricing.py and report its exact body", parent=parent)

    assert "HELPER RESULT" in obs and "pricing.py:1-3" in obs
    assert "EXACT EVIDENCE ALREADY READ BY THE HELPER" in obs
    assert "def add_tax(price, rate):" in obs
    # Copied, content-addressed span + credited ledger range.
    copied = [s for s in parent.evidence.spans if s.file == "pricing.py"]
    assert copied and copied[-1].start == 1
    assert copied[-1].content_sha == sha256_text(copied[-1].exact_text)
    assert any(a <= 1 and 3 <= b
               for a, b in parent._read_spans.get("pricing.py", []))
    # The matching task clause advanced to LOCATED/READ without a model call.
    assert parent.requirements.items[0].state in (
        verify.REQ_LOCATED, verify.REQ_READ)

    # A parent edit quoting those lines now passes the unread check without a
    # second `read`.
    reply = (
        "--- FILE: pricing.py\n<<<<<<< SEARCH\n"
        "def add_tax(price, rate):\n"
        "    return round(price * (1 + rate), 2)\n"
        "=======\n"
        "def add_tax(price, rate):\n"
        "    return round(price * (1 + rate), 3)\n"
        ">>>>>>> REPLACE\n")
    findings = verify.check_unread(parent, parse_edits(reply), [])
    assert not [f for f in findings if f.kind == "unread"]


# ------------------------------------------- constrained-hardware routing
def test_task_routes_to_scout_when_configured(monkeypatch, tmp_path):
    _root, idx, cfg = make_project(
        tmp_path, allow_subagents=True, scout_profile="scout",
        profiles={"scout": {"model": "tiny-fast",
                            "base_url": "http://127.0.0.1:9/v1"}},
        log_events=False, log_sessions=False)
    architect = Scripted([])
    scout = Scripted(
        [{"action": "read", "file": "pricing.py", "start": 1, "end": 3},
         {"action": "done"}],
        "pricing.py:1-3 defines add_tax.")
    calls = []

    def fake_scout(cfg_, default):
        calls.append((cfg_, default))
        return scout

    monkeypatch.setattr(roles_mod, "scout_client", fake_scout)
    obs = subagent.tool_task(idx, cfg, architect, "who defines add_tax?",
                             parent=None)

    assert calls and calls[0][1] is architect
    assert "pricing.py:1-3" in obs
    assert scout.seen and scout.chats
    assert architect.seen == [] and architect.chats == []


def test_tiny_prunes_task_unless_prompt_asks_for_it_or_scout_is_active():
    cfg = Config()
    cfg.sources = {}
    cfg.allow_subagents = True
    cfg.active_tier = "tiny"
    cfg._task_text = "find the callers of add_tax"
    assert "task" not in tools.schema_for(cfg)["properties"]["action"]["enum"]

    cfg._task_text = "delegate this research to a subagent"
    assert "task" in tools.schema_for(cfg)["properties"]["action"]["enum"]

    cfg._task_text = "find the callers of add_tax"
    cfg.scout_profile = "scout"
    assert "task" in tools.schema_for(cfg)["properties"]["action"]["enum"]


def test_local_tiny_subagent_forks_the_parent_stable_prefix(tmp_path):
    _root, idx, cfg = make_project(
        tmp_path, allow_subagents=True, active_tier="tiny",
        base_url="http://127.0.0.1:9/v1",
        log_events=False, log_sessions=False)
    client = Scripted(
        [{"action": "read", "file": "pricing.py", "start": 1, "end": 3},
         {"action": "done"}],
        "pricing.py:1-3 defines add_tax.")
    parent = Agent(idx, client, cfg)
    prefix = [{"role": "system", "content": "PARENT STABLE SYSTEM"},
              {"role": "user", "content": "PARENT STABLE USER0"}]
    parent._messages = [dict(m) for m in prefix]

    subagent.tool_task(idx, cfg, client, "who defines add_tax?", parent=parent)

    first_child_call = client.seen[0][0]
    assert first_child_call[:2] == [m["content"] for m in prefix]
    assert "HELPER TASK" in first_child_call[-1]
    assert parent._messages == prefix
