"""A2.3: the step-0 harness prefetch (`prefetch_targets`).

Invariants: off means no extra turn; with the flag on, the task-named files /
symbols are read before the first model step through the normal tool path (so
the read ledger and SOURCE block update), the step-0 turns are byte-identical
across runs, and prefetch reads never count toward `max_steps`."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from a1_support import READ_THEN_GREP, Scripted, gather, make_project
from pedacito.agent import Agent
from pedacito.config import Config


def _first_call(client):
    """The message contents of the first model step's chat_json call."""
    return client.seen[0][0]


def test_prefetch_is_off_by_default(tmp_path):
    assert Config().prefetch_targets == 0
    _a, client = gather(tmp_path, READ_THEN_GREP, task="Fix bulk_discount")
    # user0 (message 1) carries no prefetch turn
    assert not any("prefetch" in m for m in _first_call(client))


def test_prefetch_reads_the_task_symbol_before_step_one(tmp_path):
    _a, client = gather(tmp_path, READ_THEN_GREP, {"prefetch_targets": 1},
                        task="Fix bulk_discount")
    msgs = _first_call(client)
    # the prefetch read observation rides between user0 and the first step
    assert any("RESULT:" in m and "bulk_discount" in m for m in msgs)
    assert any("'action': 'read'" in m and "prefetch" in m for m in msgs)
    # and the read landed in the ledger, so the source is genuinely gathered
    assert "pricing.py" in _a._read_spans


def test_prefetch_turns_are_byte_identical_across_runs(tmp_path):
    runs = []
    for i in range(2):
        sub = tmp_path / str(i)
        sub.mkdir()
        _a, client = gather(sub, READ_THEN_GREP, {"prefetch_targets": 2},
                            task="Fix bulk_discount")
        runs.append([list(m) for m in _first_call(client)])
    assert runs[0] == runs[1]


def test_prefetch_does_not_consume_max_steps(tmp_path):
    _root, idx, cfg = make_project(tmp_path, prefetch_targets=2, max_steps=1)
    client = Scripted(READ_THEN_GREP)
    agent = Agent(idx, client, cfg)
    agent.gather("Fix bulk_discount", verbose=False)
    # one real model step (max_steps=1), prefetch reads logged as step 0
    assert len(client.seen) == 1
    assert any(ln.startswith("0. read") and "(prefetch)" in ln
               for ln in agent.steps_log)
