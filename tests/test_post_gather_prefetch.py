"""A4.2 (P6): one bounded post-gather prefetch when named targets stay unread."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from a1_support import Scripted, make_project
from pedacito.agent import Agent
from pedacito.config import Config


def test_post_gather_prefetch_is_off_by_default(tmp_path):
    _root, idx, cfg = make_project(tmp_path)          # prefetch_targets = 0
    client = Scripted([{"action": "grep", "pattern": "bulk_discount"},
                       {"action": "done"}])
    agent = Agent(idx, client, cfg)
    agent.gather("Explain the pricing logic.", verbose=False)
    # no post-gather read: the ranked file stays unread
    assert "pricing.py" not in agent._read_spans


def test_post_gather_prefetch_reads_a_top_file(tmp_path):
    # The task names no file or symbol, so step-0 prefetch reads nothing; the
    # model locates but never reads, so the trigger (zero reads of the ranked
    # top file) fires one post-gather read.
    _root, idx, cfg = make_project(tmp_path, prefetch_targets=1, max_steps=4)
    client = Scripted([{"action": "grep", "pattern": "bulk_discount"},
                       {"action": "done"}])
    agent = Agent(idx, client, cfg)
    agent.gather("Explain the pricing logic.", verbose=False)
    assert "pricing.py" in agent._read_spans
    assert any("(prefetch)" in ln and "post read" in ln
               for ln in agent.steps_log)
