"""Requirement evidence state machine + utility source packing.

The tracker is deterministic and free: search results locate a requirement,
displayed reads make it READ, clean patches EDIT it, and passing verification
seals VERIFIED (or REPORTED for an explain/investigation clause). The answer
packer then pins those exact spans instead of dropping the oldest source FIFO.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import verify
from pedacito.agent import _trim_gathered


def _read_item(rel: str, lines, marker: str) -> str:
    body = "\n".join(f"{i:>4} | {line}" for i, line in enumerate(lines, 1))
    return (f"--- read {rel}\n{rel} lines 1-{len(lines)}:\n{body}\n{marker}")


def _src(*items: str) -> str:
    return ("\n\n".join(items)
            + "\n\n--- digest (files read during gather)\n- see above")


def _marks(*markers: str, keep: int = 1):
    return lambda text: sum(m in text for m in markers) > keep


# --------------------------------------------------------- state transitions
def test_edit_requirement_walks_located_read_edited_verified():
    tracker = verify.RequirementTracker(
        "Fix the retry logic in client.py and update README.md")
    req = tracker.items[0]
    assert req.kind == "edit" and req.state == verify.REQ_UNLOCATED

    tracker.observe_search(
        "grep", {"pattern": "retry", "file": "client.py"},
        "client.py:10: [retry] retry_request()")
    assert req.state == verify.REQ_LOCATED
    assert ("client.py", 10, 10) in req.anchors

    tracker.observe_read(
        "client.py", 10, 12,
        "client.py lines 10-12:\n  10 | def retry_request():\n"
        "  11 |     return backoff()\n  12 | # retry logic")
    assert req.state == verify.REQ_READ

    tracker.observe_edit([{
        "file": "client.py",
        "search": "def retry_request():",
        "replace": "def retry_request():  # fixed",
    }])
    assert req.state == verify.REQ_EDITED

    tracker.observe_verified(["client.py"])
    assert req.state == verify.REQ_VERIFIED and req.final


def test_report_requirement_walks_located_read_reported():
    tracker = verify.RequirementTracker(
        "Explain how the retry backoff works in client.py")
    req = tracker.items[0]
    assert req.kind == "report" and req.state == verify.REQ_UNLOCATED

    tracker.observe_search("grep", {"pattern": "retry"},
                           "client.py:10: [retry] retry_request()")
    assert req.state == verify.REQ_LOCATED

    tracker.observe_read(
        "client.py", 10, 12,
        "client.py lines 10-12:\n  10 | def retry_request():\n"
        "  11 |     return backoff()\n  12 | # retry backoff")
    assert req.state == verify.REQ_READ

    tracker.observe_report("The retry backoff doubles after each failure.")
    assert req.state == verify.REQ_REPORTED and req.final


def test_coverage_dry_run_does_not_advance_the_live_tracker():
    """check_coverage judges dry-applied drafts and scored candidates; it
    must not promote the run's authoritative evidence. A rejected draft used
    to mark its requirements EDITED, and the next coverage check then skipped
    them as addressed. Only the agent's post-apply observe_edit advances the
    live tracker."""
    task = "Fix the retry logic in client.py and update README.md"
    tracker = verify.RequirementTracker(task)
    before = tracker.states()
    reply = ("--- FILE: client.py\n<<<<<<< SEARCH\n"
             "def retry_request():\n=======\n"
             "def retry_request():  # fixed\n>>>>>>> REPLACE\n")
    edits = verify.parse_edits(reply)
    findings = verify.check_coverage(task, reply, edits, tracker=tracker)
    assert tracker.states() == before, "a dry run advanced the live tracker"
    # the judgement itself is unchanged: README is still unaddressed
    assert any("README" in f.message for f in findings), findings
    # and the tracker still advances when the edit really lands
    tracker.observe_edit(edits)
    assert tracker.items[0].state == verify.REQ_EDITED


def test_coverage_seeds_only_edit_clauses_and_stays_a_dry_run():
    """R4: the follow-up trigger is the same judgement check_coverage renders,
    narrowed to edit clauses (a report clause is answered by prose, not another
    hunk) -- and like check_coverage it never advances the live tracker."""
    task = "Add apply_discount in pricing.py; explain how bulk_discount works"
    tracker = verify.RequirementTracker(task)
    assert [r.kind for r in tracker.items] == ["edit", "report"]
    reply = ("--- FILE: pricing.py SYMBOL apply_discount\n<<<<<<< SEARCH\n"
             "=======\ndef apply_discount(price, pct):\n"
             "    return round(price * (1 - pct), 2)\n>>>>>>> REPLACE\n")
    edits = verify.parse_edits(reply)
    before = tracker.states()

    seeds = verify.coverage_seeds(task, reply, edits, tracker=tracker)
    assert seeds == [] or all(r.is_edit for r in seeds)
    # the edit clause is addressed by the patch; the report clause is not,
    # but it must not seed an edit intent
    findings = verify.check_coverage(task, reply, edits, tracker=tracker)
    assert any("bulk_discount" in f.message for f in findings), findings
    assert all(r.is_edit for r in seeds)
    assert tracker.states() == before, "a seed pass advanced the live tracker"

    # with nothing applied, the edit clause IS a seed
    seeds = verify.coverage_seeds(task, "", [], tracker=tracker)
    assert [r.text for r in seeds] == [tracker.items[0].text]
    assert tracker.states() == before


def test_oldest_unmet_requirement_selects_read_then_search():
    tracker = verify.RequirementTracker(
        "Fix the retry logic in client.py and update README.md")
    first, second = tracker.items

    tracker.observe_search(
        "grep", {"pattern": "retry", "file": "client.py"},
        "client.py:10-12: [retry] retry_request()")
    plan, chosen = tracker.next_action()
    assert chosen is first
    assert plan == {"action": "read", "file": "client.py",
                    "start": 10, "end": 12}

    tracker.observe_read(
        "client.py", 10, 12,
        "client.py lines 10-12:\n  10 | def retry_request():\n"
        "  11 |     return backoff()\n  12 | # retry logic")
    assert first.state == verify.REQ_READ

    plan, chosen = tracker.next_action()
    assert chosen is second
    assert plan["action"] == "grep"
    assert plan["pattern"].lower() == "readme"


def test_located_line_expands_to_its_indexed_definition():
    tracker = verify.RequirementTracker(
        "Fix bulk_discount in pricing.py and update README.md")
    tracker.observe_search(
        "grep", {"pattern": "bulk_discount"},
        "pricing.py:5: [bulk_discount] if quantity > 10:")

    class _Unit:
        kind = "function"
        start, end, scope_end, nlines = 4, 7, None, 4

    class _Index:
        def in_file(self, rel):
            return [_Unit()] if rel == "pricing.py" else []

    plan, req = tracker.next_action(_Index())
    assert req is tracker.items[0]
    assert plan == {"action": "read", "file": "pricing.py",
                    "start": 4, "end": 7}


def test_verified_only_advances_a_patch_that_was_edited():
    tracker = verify.RequirementTracker(
        "Fix the retry logic in client.py and update README.md")
    first = tracker.items[0]
    tracker.observe_verified(["client.py"])
    assert first.state == verify.REQ_UNLOCATED       # nothing to verify yet

    tracker.observe_edit(file="client.py")
    assert first.state == verify.REQ_EDITED
    tracker.observe_verified(["other.py"])
    assert first.state == verify.REQ_EDITED          # wrong file: no promotion
    tracker.observe_verified(["client.py"])
    assert first.state == verify.REQ_VERIFIED


# ------------------------------------------------------ utility source packing
def test_packer_pins_located_requirement_span_instead_of_fifo():
    old = _read_item("anchor.py", ["def target():", "    return 1"], "ANCHOR_MARK")
    mid = _read_item("middle.py", ["x = 1"], "MID_MARK")
    new = _read_item("newer.py", ["y = 2"], "NEW_MARK")
    tracker = verify.RequirementTracker(
        "Fix target in anchor.py and update README.md")
    tracker.observe_search(
        "grep", {"pattern": "target", "file": "anchor.py"},
        "anchor.py:1-2: def target():")

    text, dropped = _trim_gathered(
        _src(old, mid, new), _marks("ANCHOR_MARK", "MID_MARK", "NEW_MARK"),
        requirements=tracker)

    assert "ANCHOR_MARK" in text                 # pinned, though it is oldest
    assert "MID_MARK" not in text and "NEW_MARK" not in text
    assert dropped == 2 and "middle.py" in text and "newer.py" in text


def test_packer_pins_active_failing_diagnostic_span():
    broken = _read_item(
        "data.py", ["VALUE = 1", "result = missing_name + 1"], "DIAG_MARK")
    other = _read_item("unrelated.py", ["sprite = 1"], "OTHER_MARK")

    text, dropped = _trim_gathered(
        _src(broken, other), _marks("DIAG_MARK", "OTHER_MARK"),
        diagnostics={"data.py": ["data.py:2: undefined name 'missing_name'"]})

    assert "DIAG_MARK" in text and "OTHER_MARK" not in text
    assert dropped == 1


def test_packer_penalizes_near_duplicates_and_keeps_new_requirement_info():
    total = _read_item("billing.py", ["def total():", "    return 1"],
                       "TOTAL_MARK")
    duplicate = _read_item("billing_copy.py", ["def total():", "    return 1"],
                           "DUP_MARK")
    docs = _read_item("notes.md", ["invoice", "test", "totals"], "DOC_MARK")
    game = _read_item("game.py", ["sprite", "pixels"], "GAME_MARK")
    tracker = verify.RequirementTracker(
        "Fix the billing total in billing.py and add a test for invoice totals")
    tracker.observe_search(
        "grep", {"pattern": "billing", "file": "billing.py"},
        "billing.py:1-2: def total():")

    text, _dropped = _trim_gathered(
        _src(total, duplicate, docs, game),
        _marks("TOTAL_MARK", "DUP_MARK", "DOC_MARK", "GAME_MARK", keep=2),
        requirements=tracker)

    assert "DOC_MARK" in text                    # distinct second requirement
    assert "GAME_MARK" not in text               # unrelated
    assert not ("TOTAL_MARK" in text and "DUP_MARK" in text)  # token Jaccard


def test_packer_prefers_definition_and_caller_chain_over_scattered_read():
    definition = _read_item("lib.py", ["def calc():", "    return 1"],
                            "DEF_MARK")
    caller = _read_item("use.py", ["calc()"], "CALL_MARK")
    scattered = _read_item("other.py", ["unrelated = True"], "OTHER_MARK")
    tracker = verify.RequirementTracker(
        "Fix calc in lib.py and update README.md")
    tracker.observe_search(
        "grep", {"pattern": "calc", "file": "lib.py"},
        "lib.py:1-2: def calc():")

    class _Graph:
        def neighbors(self, rel):
            if rel == "lib.py":
                return {"use.py"}
            if rel == "use.py":
                return {"lib.py"}
            return set()

    class _Index:
        graph = _Graph()

    text, _dropped = _trim_gathered(
        _src(definition, caller, scattered),
        _marks("DEF_MARK", "CALL_MARK", "OTHER_MARK", keep=2),
        requirements=tracker, index=_Index())

    assert "DEF_MARK" in text and "CALL_MARK" in text
    assert "OTHER_MARK" not in text
