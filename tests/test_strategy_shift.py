"""One bounded strategy shift when escalation has nowhere to go (#49).

When an escalation reason fires (no-progress leases, a semantic failure that
outlived repair, the zero-edit statistic) but NO distinct `escalate` role is
configured, the ladder used to give up: the same model retried nothing. The
strategy shift gives the task ONE more attempt on the SAME client and budgets
with the edit style flipped (anchor <-> quote, unset -> anchor) -- a different
mechanical approach, not a different model. Green runs never reach it; the
retry must land to win; both-proven-red keeps the first attempt.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import loop, workspace  # noqa: E402
from pedacito.config import Config  # noqa: E402

REASON = "semantic failure after repair rounds"


class _ShiftTR:
    def __init__(self, tests_ok, landed=True, reply="done"):
        self.results = [SimpleNamespace(ok=landed)]
        self.applied = landed
        self.changed = ["f.py"] if landed else []
        self.reply = reply
        self.idx = None
        self.tests_ok = tests_ok
        self.attempt = -1


def _shift_patches(monkeypatch, calls, verdicts):
    """verdicts: (tests_ok, landed, reply) per attempt; the last one repeats.
    calls records (client, edit_style) as seen at each attempt."""
    from pedacito import roles
    n = [0]

    def fake_once(root, cfg, client, agent, idx, task, **kw):
        i = n[0]
        n[0] += 1
        calls.append((client, getattr(cfg, "edit_style", None)))
        ok, landed, reply = verdicts[min(i, len(verdicts) - 1)]
        tr = _ShiftTR(ok, landed=landed, reply=reply)
        tr.attempt = i
        return tr

    monkeypatch.setattr(loop, "_run_task_once", fake_once)
    # no distinct escalate role: every role resolves to the same client
    monkeypatch.setattr(roles, "client_for", lambda cfg, role, c: c)
    monkeypatch.setattr("pedacito.agent.Agent", lambda idx, c, cfg: object())


def _cfg(reason=REASON, style=None):
    cfg = Config()
    cfg.edit_style = style
    if reason:
        cfg._pending_escalation_reason = reason
    return cfg


def test_red_landed_task_gets_one_style_shift_retry(tmp_path, monkeypatch):
    calls = []
    _shift_patches(monkeypatch, calls,
                   [(False, True, "done"), (True, True, "done")])
    cfg = _cfg(style="anchor")
    tr = loop._run_ladder(tmp_path, cfg, "weak", object(), None, "t")
    assert calls == [("weak", "anchor"), ("weak", "quote")]
    assert tr.attempt == 1 and tr.tests_ok is True
    assert cfg.edit_style == "anchor"      # restored after the retry


@pytest.mark.parametrize("style,flipped", [
    ("quote", "anchor"),
    (None, "anchor"),
])
def test_shift_flip_direction_and_restore(tmp_path, monkeypatch, style,
                                          flipped):
    calls = []
    _shift_patches(monkeypatch, calls,
                   [(False, True, "done"), (True, True, "done")])
    cfg = _cfg(style=style)
    loop._run_ladder(tmp_path, cfg, "weak", object(), None, "t")
    assert calls[1] == ("weak", flipped)
    assert cfg.edit_style == style


@pytest.mark.parametrize("verdict", [(True, True, "done"),
                                     (None, True, "done")])
def test_green_or_unverdicted_landing_never_shifts(tmp_path, monkeypatch,
                                                   verdict):
    calls = []
    _shift_patches(monkeypatch, calls, [verdict])
    tr = loop._run_ladder(tmp_path, _cfg(), "weak", object(), None, "t")
    assert len(calls) == 1
    assert tr.attempt == 0


@pytest.mark.parametrize("verdict", [(None, False, "I produced no usable edit"),
                                     (False, False, "done")])
def test_zero_edit_failure_gets_a_shift(tmp_path, monkeypatch, verdict):
    calls = []
    _shift_patches(monkeypatch, calls, [verdict, (True, True, "done")])
    tr = loop._run_ladder(tmp_path, _cfg(), "weak", object(), None, "t")
    assert len(calls) == 2
    assert tr.attempt == 1


def test_coherent_noop_is_left_alone(tmp_path, monkeypatch):
    calls = []
    _shift_patches(monkeypatch, calls, [(None, False, "nothing to change")])
    tr = loop._run_ladder(tmp_path, _cfg(), "weak", object(), None, "t")
    assert len(calls) == 1
    assert tr.attempt == 0


def test_retry_must_land_to_win(tmp_path, monkeypatch):
    calls = []
    _shift_patches(monkeypatch, calls,
                   [(False, True, "done"), (None, False, "no usable edit")])
    tr = loop._run_ladder(tmp_path, _cfg(), "weak", object(), None, "t")
    assert len(calls) == 2
    assert tr.attempt == 0                 # the retry landed nothing: keep first


def test_both_proven_red_keeps_the_first_attempt(tmp_path, monkeypatch):
    calls = []
    _shift_patches(monkeypatch, calls,
                   [(False, True, "done"), (False, True, "done")])
    tr = loop._run_ladder(tmp_path, _cfg(), "weak", object(), None, "t")
    assert len(calls) == 2
    assert tr.attempt == 0


def test_unverdicted_retry_wins_over_red_first(tmp_path, monkeypatch):
    calls = []
    _shift_patches(monkeypatch, calls,
                   [(False, True, "done"), (None, True, "done")])
    tr = loop._run_ladder(tmp_path, _cfg(), "weak", object(), None, "t")
    assert tr.attempt == 1


def test_at_most_one_shift_per_task(tmp_path, monkeypatch):
    calls = []
    _shift_patches(monkeypatch, calls,
                   [(False, True, "done"), (False, True, "done"),
                    (True, True, "done")])
    loop._run_ladder(tmp_path, _cfg(), "weak", object(), None, "t")
    assert len(calls) == 2                 # never a third attempt


def test_no_reason_means_no_shift(tmp_path, monkeypatch):
    calls = []
    _shift_patches(monkeypatch, calls, [(False, True, "done")])
    tr = loop._run_ladder(tmp_path, _cfg(reason=""), "weak", object(), None, "t")
    assert len(calls) == 1
    assert tr.attempt == 0


def test_auto_escalate_off_disables_the_shift(tmp_path, monkeypatch):
    calls = []
    _shift_patches(monkeypatch, calls, [(False, True, "done")])
    cfg = _cfg()
    cfg.auto_escalate = False
    loop._run_ladder(tmp_path, cfg, "weak", object(), None, "t")
    assert len(calls) == 1


def test_dry_run_never_shifts(tmp_path, monkeypatch):
    calls = []
    _shift_patches(monkeypatch, calls, [(False, True, "done")])
    loop._run_ladder(tmp_path, _cfg(), "weak", object(), None, "t", apply=False)
    assert len(calls) == 1


def test_review_mode_never_shifts(tmp_path, monkeypatch):
    calls = []
    _shift_patches(monkeypatch, calls, [(False, True, "done")])
    loop._run_ladder(tmp_path, _cfg(), "weak", object(), None, "t", review=True)
    assert len(calls) == 1


def test_distinct_escalate_role_takes_priority(tmp_path, monkeypatch):
    from pedacito import roles
    calls = []

    def fake_once(root, cfg, client, agent, idx, task, **kw):
        calls.append((client, getattr(cfg, "edit_style", None)))
        return _ShiftTR(False if client == "weak" else True)

    monkeypatch.setattr(loop, "_run_task_once", fake_once)
    monkeypatch.setattr(roles, "client_for",
                        lambda cfg, role, c: "strong" if role == "escalate"
                        else c)
    monkeypatch.setattr("pedacito.agent.Agent", lambda idx, c, cfg: object())
    tr = loop._run_ladder(tmp_path, _cfg(), "weak", object(), None, "t")
    # the stronger model handles the reason; no style flip on either attempt
    assert calls == [("weak", None), ("strong", None)]
    assert tr.tests_ok is True


def test_strategy_shift_event_is_logged(tmp_path, monkeypatch):
    calls = []
    _shift_patches(monkeypatch, calls,
                   [(False, True, "done"), (True, True, "done")])
    events = []
    monkeypatch.setattr(workspace, "log_event",
                        lambda root, msg, level="INFO": events.append(
                            (msg, level)))
    loop._run_ladder(tmp_path, _cfg(), "weak", object(), None, "t")
    assert any("strategy_shift" in m for m, _lvl in events)


def test_style_restored_even_when_the_retry_raises(tmp_path, monkeypatch):
    from pedacito import roles
    n = [0]

    def fake_once(root, cfg, client, agent, idx, task, **kw):
        n[0] += 1
        if n[0] == 2:
            raise RuntimeError("boom")
        return _ShiftTR(False)

    monkeypatch.setattr(loop, "_run_task_once", fake_once)
    monkeypatch.setattr(roles, "client_for", lambda cfg, role, c: c)
    monkeypatch.setattr("pedacito.agent.Agent", lambda idx, c, cfg: object())
    cfg = _cfg(style="anchor")
    with pytest.raises(RuntimeError):
        loop._run_ladder(tmp_path, cfg, "weak", object(), None, "t")
    assert cfg.edit_style == "anchor"
