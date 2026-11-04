"""A3: shadow-mode prose verification -- findings logged, never acted on."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import verify


def _agent(read_spans, symbols=(), files=()):
    class Idx:
        def resolve_file(self, name):
            return name if name in files else None

        def find_symbol(self, symbol):
            return [symbol] if symbol in symbols else []

    class A:
        _read_spans = read_spans
        index = Idx()

    return A()


def test_citations_report_unresolvable_paths_and_unread_lines():
    a = _agent({"shop/pricing.py": [(1, 20)]}, files={"shop/pricing.py"})
    reply = "See ghost.py:5 and shop/pricing.py:99 for the details."
    fs = verify.check_citations(a, reply)
    assert any("does not resolve" in f.message for f in fs)
    assert any("not among the lines" in f.message for f in fs)


def test_citations_pass_a_cited_line_that_was_read():
    a = _agent({"shop/pricing.py": [(1, 20)]}, files={"shop/pricing.py"})
    reply = "The tax rate lives at shop/pricing.py:10."
    assert verify.check_citations(a, reply) == []


def test_ungrounded_identifiers_flag_invented_symbols_only():
    a = _agent({}, symbols={"process_order"})
    reply = "It calls process_order and then retry_with_backoff."
    fs = verify.check_ungrounded_identifiers(a, reply, "explain process_order")
    assert len(fs) == 1 and "retry_with_backoff" in fs[0].message


def test_prose_shadow_findings_are_always_warnings():
    a = _agent({})
    reply = "It calls frobnicate_widgets and reports at ghost.py:3."
    fs = verify.prose_shadow(a, "do the thing", reply)
    assert fs and all(f.severity == "warn" for f in fs)
