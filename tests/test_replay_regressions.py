"""Frozen replay regression gates for five multi-step bug-hunt fixes.

Each test scripts the model's next step (through `replay.Replayer` or the
deterministic `Scripted` client) and asserts the harness-side decision, with
no live LLM server. Together they run in well under a second.

  BUG-208  identical read-only repeats are suppressed, the deterministic
           requirement assist fires, and two no-progress leases suggest
           escalation (§40).
  BUG-320  a repeated failing read hits `cached_failure`, never dispatch.
  BUG-314  a repair diagnostic prompt mines no requirements and seeds no
           bogus coverage follow-up.
  BUG-807  coverage seeds never target prose documents unless the clause
           names that document (§35).
  §46/#49  a repair cannot weaken a pre-existing test file, and a reason with
           no `escalate` role buys one flipped-edit-style retry.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_agent_context import Scripted, _agent  # noqa: E402

from pedacito import edits as edits_mod, loop, replay, verify  # noqa: E402
from pedacito.config import Config  # noqa: E402

CORE = "def retry_request():\n    return 1\n"
FILES = {
    "README.md": "# Readme\n\nretry notes\n",
    "pkg/__init__.py": "",
    "pkg/core.py": CORE,
}
TASK = "fix retry_request in pkg/core.py and update README.md"


class _Events:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def emit(self, kind, **fields):
        self.events.append((kind, fields))

    def __getattr__(self, _name):
        return lambda *a, **k: None

    def kinds(self):
        return [k for k, _f in self.events]


def _recording(path: Path, steps) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for n, step in enumerate(steps):
            fh.write(json.dumps({"kind": "chat_json", "n": n,
                                 "out": {"reasoning": "r", **step}}) + "\n")
    return path


def _read(file, a, b):
    return {"action": "read", "file": file, "start": a, "end": b}


def test_bug208_repeat_suppression_assist_and_lease_strikes(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    recording = _recording(tmp_path / "run.jsonl", [
        _read("pkg/core.py", 1, 2),
        _read("pkg/core.py", 1, 2),
        _read("pkg/core.py", 1, 2),
        {"action": "done"},
    ])
    agent = _agent(project, replay.Replayer(recording), FILES)
    events = _Events()
    agent.events = events

    agent.gather(TASK, verbose=False)

    # 1st repeat is suppressed; the deterministic requirement assist runs for
    # the still-unread README clause; the third identical step is a loop guard.
    assert any("already ran and its result is above" in m["content"]
               for m in agent._messages), agent._messages[-3:]
    assert "requirement assist" in "\n".join(agent.steps_log), agent.steps_log
    assert "repeat" in [f.get("signal") for k, f in events.events
                        if k == "recovery"]

    # Two no-progress leases on one task are the escalation signal (§40).
    agent._note_lease_strike(TASK)
    assert "escalation_suggested" not in events.kinds()
    agent._note_lease_strike(TASK)
    assert "escalation_suggested" in events.kinds()
    reason = getattr(agent.cfg, "_task_escalation_reason", "")
    assert "leases" in reason, reason


def test_bug320_repeated_failing_read_is_cached(tmp_path):
    steps = [
        {"action": "read", "file": "missing.py", "start": 1, "end": 9},
        _read("pkg/core.py", 1, 2),
        {"action": "read", "file": "missing.py", "start": 1, "end": 9},
        _read("pkg/core.py", 1, 2),
        {"action": "read", "file": "missing.py", "start": 1, "end": 9},
        {"action": "done"},
    ]
    agent = _agent(tmp_path, Scripted(steps), FILES)
    events = _Events()
    agent.events = events
    agent.gather("read missing.py", verbose=False)

    assert events.kinds().count("cached_failure") == 2, events.kinds()
    dispatched = [f for k, f in events.events
                  if k == "step" and f.get("action") == "read"
                  and f.get("detail") == "missing.py"]
    assert len(dispatched) == 1, dispatched      # only the first attempt ran
    assert any("already failed at step" in m["content"]
               for m in agent._messages)


def test_bug314_repair_prompt_mines_no_requirements(tmp_path):
    diagnostic = (
        "Your last change failed this check:\n"
        "- F821: `json` is not defined in src/app.py\n"
        "- ContractError in src/app.py:2: calls run() with 2 args\n"
        "\n"
        "Fix these first:\n"
        "1. import json\n"
        "2. correct the call signature\n"
    )
    agent = _agent(tmp_path, Scripted([{"action": "done"}]),
                   {"src/app.py": "def run():\n    return 1\n"})
    events = _Events()
    agent.events = events
    agent.gather(diagnostic, verbose=False, is_repair=True)

    assert agent._is_repair is True
    assert agent.requirements.items == []

    # The batch pass must not append a coverage verdict or follow-up pass.
    def _boom(*_a, **_k):
        raise AssertionError("coverage pass ran for a repair prompt")

    agent._coverage_pass = _boom
    results, _reply, findings = agent.apply_edit_intents(diagnostic, "", [])
    assert results == [] and findings == []
    assert agent._coverage_intents([]) == []
    assert "coverage_followup" not in events.kinds()


def test_bug807_prose_documents_are_not_coverage_targets(tmp_path):
    files = {
        "src/app.py": "def compute_tax(amount):\n    return amount * 0.2\n",
        "PLAN.md": "# Plan\n\nImplement compute_tax in the checkout path.\n",
        "NOTES.md": "- compute_tax must round\n- reduced-motion behavior\n",
        "report.txt": "compute_tax appears in this report verbatim\n",
    }
    agent = _agent(tmp_path, Scripted(), files)

    # A code clause whose words appear verbatim in prose, but whose anchor
    # points at the note: the prose document is refused and the code target
    # is chosen instead.
    req = verify.Requirement("Implement compute_tax in src/app.py",
                             ["compute_tax", "src/app.py"], "edit",
                             req_id="R1")
    req.add_anchor("PLAN.md", 1, 1)
    req.add_anchor("report.txt", 1, 1)
    intents = agent._coverage_intents([req])
    assert intents, intents
    assert all(not i["file"].endswith((".md", ".rst", ".txt"))
               for i in intents), intents
    assert intents[0]["file"] == "src/app.py", intents

    # A clause that explicitly names the document still owes that edit.
    named = verify.Requirement("update PLAN.md with the compute_tax section",
                               ["PLAN.md", "compute_tax"], "edit",
                               req_id="R2")
    named.add_anchor("PLAN.md", 1, 1)
    named_intents = agent._coverage_intents([named])
    assert any(i["file"] == "PLAN.md" for i in named_intents), named_intents


def _repair_reply():
    return (
        "--- FILE: src/app.py\n<<<<<<< SEARCH\n"
        "def f():\n    return 1\n=======\n"
        "def f():\n    return 2\n>>>>>>> REPLACE\n"
        "--- FILE: tests/test_app.py\n<<<<<<< SEARCH\n"
        "    assert True\n=======\n"
        "    assert False\n>>>>>>> REPLACE\n"
    )


def test_46_repair_test_lock_refuses_weakening_and_applies_code_fix(tmp_path):
    files = {
        "src/app.py": "def f():\n    return 1\n",
        "tests/test_app.py": "def test_f():\n    assert True\n",
    }
    agent = _agent(tmp_path, Scripted(), files)
    agent.cfg._task_preexisting = set(files)
    task = "fix f in src/app.py"
    agent._arm_repair_test_lock(task)

    parsed = edits_mod.parse_edits(_repair_reply())
    kept, refused = agent._filter_repair_test_edits(parsed, task)
    assert {e.file for e in kept} == {"src/app.py"}
    assert len(refused) == 1 and refused[0].file == "tests/test_app.py"
    assert "pre-existing test file" in refused[0].message
    assert "assert True" in (tmp_path / "tests/test_app.py").read_text(encoding="utf-8")

    applied = edits_mod.apply_edits(tmp_path, kept, backup=False)
    assert all(r.ok for r in applied), applied
    assert "return 2" in (tmp_path / "src/app.py").read_text(encoding="utf-8")
    assert "assert True" in (tmp_path / "tests/test_app.py").read_text(encoding="utf-8")


class _TR:
    def __init__(self, tests_ok, landed=True, reply="done"):
        self.results = [SimpleNamespace(ok=landed)]
        self.applied = landed
        self.changed = ["f.py"] if landed else []
        self.reply = reply
        self.idx = None
        self.tests_ok = tests_ok


def test_49_one_strategy_shift_when_no_escalate_role(tmp_path, monkeypatch):
    calls = []
    events = []
    verdicts = [(False, True, "done"), (True, True, "done")]
    state = {"n": 0}

    def fake_once(root, cfg, client, agent, idx, task, **kw):
        i = state["n"]
        state["n"] += 1
        calls.append((client, getattr(cfg, "edit_style", None)))
        return _TR(*verdicts[min(i, len(verdicts) - 1)])

    from pedacito import roles
    monkeypatch.setattr(loop, "_run_task_once", fake_once)
    monkeypatch.setattr(roles, "client_for", lambda cfg, role, c: c)
    monkeypatch.setattr("pedacito.agent.Agent", lambda idx, c, cfg: object())
    monkeypatch.setattr(loop.workspace, "log_event",
                        lambda root, msg, level="INFO": events.append(msg))

    cfg = Config()
    cfg.edit_style = "quote"
    cfg._pending_escalation_reason = "semantic failure after repair rounds"
    result = loop._run_ladder(tmp_path, cfg, "weak", object(), None, "t")

    assert calls == [("weak", "quote"), ("weak", "anchor")], calls
    assert result.tests_ok is True
    assert cfg.edit_style == "quote"            # restored after the retry
    assert any(m.startswith("strategy_shift:") for m in events), events
