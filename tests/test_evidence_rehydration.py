"""Sprint 2: content-addressed evidence store and append-only exact-span
rehydration (TECHNIQUES.md section 11).

Invariants: every displayed read/context/prefetch span is hashed into
`EvidenceStore` recording exactly what the model was SHOWN (post clip); the
`NEED path:120-168` directive re-fetches only that span and APPENDS at the
conversation tail -- history bytes never change, so the prefix cache holds --
bounded by `max_rehydration_rounds`; and repair auto-rehydration fires only
for unread/fuzzy findings covered by, or within +/-30 lines of, recorded
evidence."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from a1_support import READ_THEN_GREP, Scripted, gather, make_project
from pedacito.agent import (Agent, EvidenceStore, sha256_text,
                            _READ_STEP_REFETCH)
from pedacito.config import Config
from pedacito.edits import Edit, parse_edits
from pedacito.verify import (Finding, check_fuzzy, check_unread,
                             evidence_rehydrations, find_needs)

PRICING_SRC = (
    "def add_tax(price, rate):\n"
    "    return round(price * (1 + rate), 2)\n"
    "\n"
    "def bulk_discount(quantity):\n"
    "    if quantity > 10:\n"
    "        return 0.1\n"
    "    return 0.0\n"
)


# ------------------------------------------------------- EvidenceStore units
def test_record_displayed_hashes_the_shown_text():
    store = EvidenceStore()
    # header + numbered lines exactly as delivered (post clip)
    obs = ("pricing.py lines 2-3:\n"
           "    2 |     return round(price * (1 + rate), 2)\n"
           "    3 |\n"
           "    4 | def bulk_discount(quantity):")
    span = store.record_displayed(obs, read_step=5)
    assert span is not None
    assert span.file == "pricing.py"
    assert (span.start, span.end) == (2, 4)
    assert span.exact_text == ("    return round(price * (1 + rate), 2)\n\n"
                               "def bulk_discount(quantity):")
    assert span.content_sha == sha256_text(span.exact_text)
    assert span.read_step == 5


def test_record_displayed_ignores_non_source():
    store = EvidenceStore()
    assert store.record_displayed("ERROR: no such file", 1) is None
    assert store.record_displayed("pricing.py outline: 2 defs", 1) is None
    assert store.record_displayed(
        "pricing.py lines 1-7: [already read; this range contains no new "
        "source.]", 3) is None
    assert store.spans == []


def test_covering_and_nearby_proximity_rule():
    store = EvidenceStore()
    store.record("pkg/a.py", 1, 10, "x" * 10, 1)
    assert store.covering("pkg/a.py", 3, 8)
    assert not store.covering("pkg/a.py", 3, 12)
    # within the +/-30 window: region 15-45 touches span 1-10 from afar
    assert store.nearby("pkg/a.py", 15, 45)
    assert not store.nearby("pkg/a.py", 100, 120)
    # the two spellings a small model alternates between are the same file
    assert len(store.nearby("./pkg/a.py", 5, 6)) == 1
    assert store.exact("pkg/a.py", 1, 10) is not None
    assert store.exact("pkg/a.py", 2, 10) is None


def test_gather_records_displayed_spans(tmp_path):
    agent, _client = gather(tmp_path, READ_THEN_GREP)
    spans = agent.evidence.for_file("pricing.py")
    assert spans, "a delivered read must leave an evidence span"
    s = spans[0]
    assert s.start == 1 and s.end == 7
    assert s.exact_text in PRICING_SRC
    assert s.content_sha == sha256_text(s.exact_text)
    assert s.read_step == 1                      # shown by gather step 1
    # grep/done steps display no source: only the read recorded
    assert len(spans) == 1


def test_prefetch_records_evidence_at_step_zero(tmp_path):
    agent, _client = gather(tmp_path, READ_THEN_GREP,
                            {"prefetch_targets": 1}, task="Fix bulk_discount")
    prefetch = [s for s in agent.evidence.for_file("pricing.py")
                if s.read_step == 0]
    assert prefetch, "the prefetch-displayed lines must be recorded"
    assert prefetch[0].exact_text in PRICING_SRC


# ------------------------------------------------------------ NEED parsing
def test_find_needs_parses_exact_span_directives():
    reply = ("I will edit it.\n"
             "NEED shop/pricing.py:120-168\n"
             "- NEED a/b.py: 5\n"                     # space after colon
             "need c.py 30 to 40\n"                   # lowercase + `to`
             "NEED `d.py:7-9`\n"                      # backticks
             "NEED e.py:168-120\n"                    # reversed -> swapped
             "I need help: 3\n"                       # not a path
             "pricing.py:2 says the rate is wrong\n"  # citation, not NEED
             "NEED shop/pricing.py:120-168\n")        # exact duplicate
    assert find_needs(reply) == [
        ("shop/pricing.py", 120, 168), ("a/b.py", 5, 5), ("c.py", 30, 40),
        ("d.py", 7, 9), ("e.py", 120, 168)]


# --------------------------------------- findings carry their flagged region
def test_unread_finding_records_its_region(tmp_path):
    _root, idx, cfg = make_project(tmp_path)
    agent = Agent(idx, Scripted([]), cfg)
    agent._read_spans = {"pricing.py": [(1, 3)]}
    agent.read_any = True
    edits = [Edit(file="pricing.py", search="x", replace="y", lines=(5, 6))]
    found = check_unread(agent, edits, [])
    assert found and found[0].kind == "unread"
    assert found[0].lines == (5, 6) and found[0].line == 5


def test_fuzzy_finding_records_the_changed_region():
    before = PRICING_SRC
    after = before.replace("if quantity > 10:", "if quantity >= 10:")
    result = type("R", (), {"fuzzy": True, "file": "pricing.py",
                            "before": before, "after": after})()
    found = check_fuzzy([result])
    assert len(found) == 1
    assert found[0].lines == (5, 5)          # the edited line in `before`


# -------------------------------------------------- evidence-gated rehydration
def _agent_with_read(tmp_path):
    agent, _ = gather(tmp_path, READ_THEN_GREP)   # shows pricing.py 1-7
    return agent


def test_rehydrations_require_nearby_evidence(tmp_path):
    agent = _agent_with_read(tmp_path)
    near = Finding("unread", "error", "pricing.py", "guess", line=4,
                   lines=(4, 5))
    assert evidence_rehydrations(agent, [near]) == [("pricing.py", 4, 5)]
    # a region beyond the +/-30 window has no recorded evidence: no dump
    far = Finding("unread", "error", "pricing.py", "guess", line=100,
                  lines=(100, 120))
    assert evidence_rehydrations(agent, [far]) == []
    # a never-read-file finding carries no region: nothing to re-fetch
    whole = Finding("unread", "error", "pricing.py", "never read")
    assert evidence_rehydrations(agent, [whole]) == []
    # other kinds never rehydrate, even with a line inside recorded evidence
    stale = Finding("stale", "error", "pricing.py", "gone", line=4,
                    lines=(4, 5))
    assert evidence_rehydrations(agent, [stale]) == []


def test_rehydrations_dedupe_and_cap(tmp_path):
    agent = _agent_with_read(tmp_path)
    findings = [Finding("fuzzy", "warn", "pricing.py", "approx", line=4,
                        lines=(4, 5))] * 2 + \
        [Finding("unread", "error", "pricing.py", "g", line=2 + i,
                 lines=(2 + i, 2 + i)) for i in range(6)]
    targets = evidence_rehydrations(agent, findings)
    assert len(targets) == 4                       # bounded (#3)
    assert targets[0] == ("pricing.py", 4, 5)      # deduped, order kept


def test_auto_rehydrate_appends_fresh_span_and_credits_ledger(tmp_path):
    agent = _agent_with_read(tmp_path)
    before = len(agent._read_spans["pricing.py"])
    block = agent._auto_rehydrate([
        Finding("unread", "error", "pricing.py", "guess", line=4,
                lines=(4, 6))])
    assert "FRESH SOURCE FOR THE FLAGGED REGIONS" in block
    assert "if quantity > 10:" in block            # exact disk bytes, shown
    assert "4 |" in block                          # numbered like a `read`
    assert len(agent._read_spans["pricing.py"]) == before + 1
    refetched = agent.evidence.spans[-1]
    assert (refetched.file, refetched.start, refetched.end) == (
        "pricing.py", 4, 6)
    assert refetched.read_step == _READ_STEP_REFETCH
    assert agent.rehydration_rounds == 1
    # far from any evidence: nothing appended, ledger untouched
    assert agent._auto_rehydrate([
        Finding("unread", "error", "pricing.py", "guess", line=100,
                lines=(100, 120))]) == ""


# ------------------------------------------------------ explicit NEED rounds
def test_need_rehydration_appends_at_tail_and_reasks(tmp_path):
    agent = _agent_with_read(tmp_path)
    calls = []

    def chat(msgs):
        calls.append([dict(m) for m in msgs])
        return ("I need the exact lines.\nNEED pricing.py:4-5\n"
                if len(calls) == 1 else "FINAL ANSWER")

    head = {"role": "system", "content": "SYSTEM TEXT"}
    tail = {"role": "user", "content": "TASK TEXT"}
    msgs = [head, tail]
    reply = agent._chat_needs(chat, msgs)
    assert reply == "FINAL ANSWER"
    assert len(calls) == 2
    # history was never rewritten: the first two bytes are the same message
    assert calls[1][:2] == calls[0]
    # the fetched span arrived appended as assistant + user turns at the tail
    assert len(msgs) == 4
    assert msgs[2]["role"] == "assistant" and "NEED pricing.py:4-5" in \
        msgs[2]["content"]
    assert msgs[3]["role"] == "user"
    assert "EXACT SOURCE FETCHED" in msgs[3]["content"]
    assert "def bulk_discount(quantity):" in msgs[3]["content"]
    assert "if quantity > 10:" in msgs[3]["content"]
    # ledger credited, evidence recorded, round counted
    assert (4, 5) in agent._read_spans["pricing.py"]
    assert agent.evidence.spans[-1].read_step == _READ_STEP_REFETCH
    assert agent.rehydration_rounds == 1


def test_need_rehydration_is_bounded(tmp_path):
    agent = _agent_with_read(tmp_path)
    calls = []

    def chat(msgs):
        calls.append(len(msgs))
        return "NEED pricing.py:4-5"          # never learns

    msgs = [{"role": "user", "content": "TASK"}]
    reply = agent._chat_needs(chat, msgs)
    assert reply == "NEED pricing.py:4-5"
    # default max_rehydration_rounds = 2: the first pass + two rehydrated re-asks
    assert len(calls) == 3
    assert agent.rehydration_rounds == 2
    assert Config().max_rehydration_rounds == 2


def test_need_rehydration_disabled_by_zero_rounds(tmp_path):
    agent, _ = gather(tmp_path, READ_THEN_GREP)
    agent.cfg.max_rehydration_rounds = 0
    calls = []

    def chat(msgs):
        calls.append(1)
        return "NEED pricing.py:4-5"

    msgs = [{"role": "user", "content": "TASK"}]
    assert agent._chat_needs(chat, msgs) == "NEED pricing.py:4-5"
    assert len(calls) == 1                    # a single pass, no re-ask
    assert len(msgs) == 1                     # nothing appended


def test_fetch_exact_span_reports_content_drift(tmp_path):
    agent = _agent_with_read(tmp_path)
    (tmp_path / "proj" / "pricing.py").write_text(
        PRICING_SRC.replace("if quantity > 10:", "if quantity > 9:"),
        encoding="utf-8")
    obs = agent._fetch_exact_span("pricing.py", 4, 6)
    assert "if quantity > 9:" in obs          # fresh bytes from disk
    assert "content changed since it was last shown" in obs
    new = agent.evidence.spans[-1]
    assert new.content_sha == sha256_text(new.exact_text)
    # re-fetching the same unchanged range now says "unchanged"
    obs2 = agent._fetch_exact_span("pricing.py", 4, 6)
    assert "unchanged since step" in obs2
    # source the model was never shown arrives with no compare note at all
    _root, idx2, cfg2 = make_project(tmp_path / "second")
    fresh = Agent(idx2, Scripted([]), cfg2)
    obs3 = fresh._fetch_exact_span("pricing.py", 4, 5)
    assert "def bulk_discount(quantity):" in obs3
    assert "unchanged" not in obs3 and "content changed" not in obs3


# --------------------------------------------------------- repair wiring (2B)
class _Sequenced(Scripted):
    """A client whose chat() walks a scripted list of replies."""

    def __init__(self, replies):
        super().__init__()
        self.replies = list(replies)

    def chat(self, messages, max_tokens, **kw):
        self.chats.append([dict(m) for m in messages])
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def test_repair_from_findings_rehydrates_and_honours_need(tmp_path):
    agent = _agent_with_read(tmp_path)
    block = ("--- FILE: pricing.py\n<<<<<<< SEARCH\n    if quantity > 10:\n"
             "=======\n    if quantity >= 10:\n>>>>>>> REPLACE\n")
    client = _Sequenced(["Prose.\n" + block + "\nNEED pricing.py:6-7",
                         "Prose.\n" + block])
    agent.client = client
    errors = [Finding("unread", "error", "pricing.py",
                      "lines 4-5 were not (fully) among the lines read",
                      line=4, lines=(4, 5))]
    fixed = agent._repair_from_findings("Fix the threshold", block, errors, [])
    assert parse_edits(fixed), "the corrected reply must survive the rounds"
    assert len(client.chats) == 2                   # NEED bought one re-ask
    # implicit: fresh flagged source was appended at the TAIL of the repair
    # prompt before the first generation (the prefix instruction stays put)
    assert "FRESH SOURCE FOR THE FLAGGED REGIONS" in \
        client.chats[0][-1]["content"]
    assert "if quantity > 10:" in client.chats[0][-1]["content"]
    assert "NEED path/to/file.py:START-END" in client.chats[0][-1]["content"]
    # explicit: the NEED span arrived as appended turns, not a rewrite
    assert client.chats[1][:2] == client.chats[0][:2]
    assert "EXACT SOURCE FETCHED" in client.chats[1][-1]["content"]
    assert "    return 0.0" in client.chats[1][-1]["content"]
    # and both fetches are ledger-credited, so the fix is not a fresh guess
    assert (4, 5) in agent._read_spans["pricing.py"]
    assert (6, 7) in agent._read_spans["pricing.py"]
    assert agent.rehydration_rounds == 2            # auto + one NEED round
