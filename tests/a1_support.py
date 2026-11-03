"""Shared fixtures for the Phase A1 tests (reason-first steps, the edit-plan
brief, per-phase temperature, project few-shot steps): a tiny indexed project
and an `EvalModel` that records what each call was shown."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import index as index_mod
from pedacito.agent import Agent
from pedacito.config import Config
from test_features_misc import EvalModel

PRICING = (
    "def add_tax(price, rate):\n"
    "    return round(price * (1 + rate), 2)\n"
    "\n"
    "def bulk_discount(quantity):\n"
    "    if quantity > 10:\n"
    "        return 0.1\n"
    "    return 0.0\n"
)


def make_project(tmp_path, **cfg_attrs):
    """(root, idx, cfg) for a one-file project; `cfg_attrs` set on the config."""
    root = tmp_path / "proj"
    root.mkdir(exist_ok=True)
    (root / "pricing.py").write_text(PRICING, encoding="utf-8")
    cfg = Config()
    cfg.log_events = False
    cfg.log_sessions = False
    cfg.sources = {}
    for k, v in cfg_attrs.items():
        setattr(cfg, k, v)
    idx = index_mod.build([str(root)], cfg, client=None, summarise=False,
                          root=root, verbose=False)
    return root, idx, cfg


class Scripted(EvalModel):
    """EvalModel that records the message contents and schema of every step
    call, and the temperature each chat() ran at."""

    def __init__(self, steps=None, reply=""):
        super().__init__(reply, steps)
        self.seen: list[tuple[list[str], dict]] = []
        self.temps: list[float | None] = []
        self._override = None

    def chat_json(self, messages, schema, max_tokens, **kw):
        self.seen.append(([m["content"] for m in messages], schema))
        return super().chat_json(messages, schema, max_tokens, **kw)


def gather(tmp_path, steps, cfg_attrs=None, client=None, task="Fix bulk_discount"):
    """Run one gather over the tiny project; returns (agent, client)."""
    _root, idx, cfg = make_project(tmp_path, **(cfg_attrs or {}))
    client = client or Scripted(steps)
    agent = Agent(idx, client, cfg)
    agent.gather(task, verbose=False)
    return agent, client


READ_THEN_GREP = [
    {"action": "read", "file": "pricing.py", "start": 1, "end": 7},
    {"action": "grep", "pattern": "bulk_discount"},
    {"action": "done"},
]
