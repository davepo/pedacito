"""Global gather bounds:

A. the action-independent no-progress lease and the model-call ceiling, which
   bound gather even when `max_steps = 0` removes the fixed lookup cap;
B. the tokenizer-aware read token budget, which clips `read`/`context`
   observations at whole-line boundaries and credits the span ledger with
   ONLY the lines the model was actually shown.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import index as index_mod
from pedacito.agent import Agent
from pedacito.config import Config
from a1_support import Scripted, make_project

BIG = ("\n".join(f"value_{i:04d} = {i}  # padding to widen the line"
                 for i in range(200)) + "\n")


def _project(tmp_path, files, **cfg_attrs):
    """(root, idx, cfg) for a small on-disk project; `cfg_attrs` set on config."""
    root = tmp_path / "proj"
    root.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    cfg = Config()
    cfg.log_events = False
    cfg.log_sessions = False
    cfg.sources = {}
    for k, v in cfg_attrs.items():
        setattr(cfg, k, v)
    idx = index_mod.build([str(root)], cfg, client=None, summarise=False,
                          root=root, verbose=False)
    return root, idx, cfg


def _last_user(messages) -> str:
    return [m["content"] for m in messages if m["role"] == "user"][-1]


# ------------------------------------------------------------------ defaults
def test_new_global_bounds_have_defaults():
    cfg = Config()
    assert cfg.max_total_model_calls == 30
    assert cfg.max_no_progress_steps == 4
    assert cfg.max_read_tokens == 1200


# ------------------------------------------------------- A: no-progress lease
def test_no_progress_lease_stops_a_wandering_gather(tmp_path):
    """Four consecutive steps that add no unread span and resolve no symbol
    end gather with a harness note -- even with max_steps = 0 (no lookup cap).
    Varied greps dodge the exact-repeat and duplicate-read guards on purpose."""
    _root, idx, cfg = make_project(tmp_path)

    class Wanderer:
        step_truncated = False

        def __init__(self):
            self.n = 0

        def chat_json(self, messages, schema, max_tokens, **kw):
            self.n += 1
            return {"action": "grep", "file": "pricing.py",
                    "pattern": f"needle_{self.n}", "reasoning": "hunt"}

    client = Wanderer()
    agent = Agent(idx, client, cfg)
    agent.gather("Fix bulk_discount", verbose=False)
    assert client.n == cfg.max_no_progress_steps
    assert "[Harness: gather stopped after" in _last_user(agent._messages)
    assert "no unread line span" in _last_user(agent._messages)


def test_reads_reset_the_no_progress_streak(tmp_path):
    """A read of new lines is progress: three greps between reads never
    reaches the lease of four."""
    steps = []
    for i in range(1, 8):
        steps.append({"action": "read", "file": "pricing.py",
                      "start": i, "end": i})
        steps += [{"action": "grep", "file": "pricing.py", "pattern": f"p{i}"},
                  {"action": "grep", "file": "pricing.py", "pattern": f"q{i}"},
                  {"action": "grep", "file": "pricing.py", "pattern": f"r{i}"}]
    steps.append({"action": "done"})
    _root, idx, cfg = make_project(tmp_path, max_total_model_calls=0)
    client = Scripted(steps)
    Agent(idx, client, cfg).gather("Fix bulk_discount", verbose=False)
    assert client.n == len(steps)   # never cut by the lease
    every = "\n".join(c for call, _schema in client.seen for c in call)
    assert "[Harness: gather stopped" not in every


def test_resolving_a_new_symbol_counts_as_progress(tmp_path):
    """The lease resets when a step resolves a symbol the ledger has never
    seen -- here a symbol read whose lines are already covered, so only the
    resolution can be what keeps gather alive."""
    steps = [
        {"action": "read", "file": "pricing.py", "start": 1, "end": 7},
        {"action": "grep", "file": "pricing.py", "pattern": "a"},
        {"action": "grep", "file": "pricing.py", "pattern": "b"},
        {"action": "read", "file": "", "symbol": "bulk_discount"},
        {"action": "grep", "file": "pricing.py", "pattern": "c"},
        {"action": "grep", "file": "pricing.py", "pattern": "d"},
        {"action": "done"},
    ]
    _root, idx, cfg = make_project(tmp_path)
    client = Scripted(steps)
    agent = Agent(idx, client, cfg)
    agent.gather("Fix bulk_discount", verbose=False)
    assert client.n == len(steps), "resolving a new symbol did not reset the lease"


# ---------------------------------------------------- A: model-call ceiling
def test_model_call_ceiling_stops_gather(tmp_path):
    """cfg.max_total_model_calls is a hard ceiling on LLM invocations per
    task: cycling fresh single-line reads keeps retrieval progress so ONLY
    the call ceiling can end the loop."""
    _root, idx, cfg = make_project(tmp_path, max_total_model_calls=3)

    class ForeverReader:
        step_truncated = False

        def __init__(self):
            self.n = 0

        def chat_json(self, messages, schema, max_tokens, **kw):
            self.n += 1
            line = (self.n % 7) + 1
            return {"action": "read", "file": "pricing.py",
                    "start": line, "end": line, "reasoning": "more"}

    agent = Agent(idx, ForeverReader(), cfg)
    agent.gather("Fix bulk_discount", verbose=False)
    assert agent.client.n == 3
    note = _last_user(agent._messages)
    assert "[Harness: gather stopped" in note and "model calls" in note


# ------------------------------------------- B: ledger-accurate read clipping
def _gather_one_clip(tmp_path, **cfg_attrs):
    _root, idx, cfg = _project(tmp_path, {"data.py": BIG}, **cfg_attrs)
    steps = [{"action": "read", "file": "data.py", "start": 1, "end": 200},
             {"action": "done"}]
    client = Scripted(steps)
    agent = Agent(idx, client, cfg)
    gathered = agent.gather("change value_0001 to 100", verbose=False)
    return agent, gathered


def test_read_clip_credits_the_ledger_with_only_shown_lines(tmp_path):
    """The header used to promise the REQUESTED range; now the budget clips
    before the ledger, the header says what was delivered, and the span is
    (start, actual_end) -- truncated-away lines are not credited as read."""
    agent, gathered = _gather_one_clip(tmp_path, max_read_tokens=60)
    m = re.search(r"data\.py lines 1-(\d+):", gathered)
    assert m, gathered[:300]
    end = int(m.group(1))
    assert 1 <= end < 200, "the 200-line read was not clipped"
    # the ledger matches the clipped header, not the requested range
    assert agent._read_spans["data.py"] == [(1, end)]
    # the continuation hint uses the exact promised wording
    assert (f"[CLIPPED at line {end} due to token budget. Use "
            f"next_start={end + 1} to continue reading.]") in gathered
    # whole-line boundaries: every surviving numbered line is complete
    shown = [int(pm.group(1)) for pm in
             (re.match(r"\s*(\d+) \|", ln) for ln in gathered.split("\n")) if pm]
    assert shown == list(range(1, end + 1))
    assert ("# padding to widen the line" in gathered
            and "truncated: context budget reached" not in gathered)
    # the first clipped-away line is gone from source AND from the ledger
    assert f"value_{end:04d} = {end}  # padding" not in gathered


def test_read_clip_is_tokenizer_aware(tmp_path):
    """A calibrated client tokenizer (chars_per_token) measures instead of
    the ~3.5 heuristic: the same token budget delivers more lines when the
    tokenizer says the text is cheaper per character."""
    _agent1, g1 = _gather_one_clip(tmp_path, max_read_tokens=60)
    end_default = int(re.search(r"data\.py lines 1-(\d+):", g1).group(1))

    _root, idx, cfg = _project(tmp_path, {"data2.py": BIG},
                               max_read_tokens=60)
    # unique file name so the second gather builds its own index
    steps = [{"action": "read", "file": "data2.py", "start": 1, "end": 200},
             {"action": "done"}]
    client = Scripted(steps)
    client.chars_per_token = 7.0          # a calibrated tokenizer
    agent2 = Agent(idx, client, cfg)
    g2 = agent2.gather("change value_0001 to 100", verbose=False)
    end_cal = int(re.search(r"data2\.py lines 1-(\d+):", g2).group(1))
    assert end_cal > end_default, (end_cal, end_default)
    assert agent2._read_spans["data2.py"] == [(1, end_cal)]


def test_read_budget_off_restores_the_full_range(tmp_path):
    """max_read_tokens = 0 disables the clip: the ledger again credits the
    full delivered range."""
    agent, gathered = _gather_one_clip(tmp_path, max_read_tokens=0)
    assert re.search(r"data\.py lines 1-200:", gathered)
    assert agent._read_spans["data.py"] == [(1, 200)]
