"""`pedacito eval --repeats/--seed`: seeded repeats, per-task pass counts,
spread, and the invariants the runner must not disturb."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pedacito import evalrun, replay
from pedacito.config import Config

from test_features_misc import EvalModel

FIX = ("Changed the threshold.\n--- FILE: shop/pricing.py\n<<<<<<< SEARCH\n"
       "    if quantity > 10:\n=======\n    if quantity >= 10:\n>>>>>>> REPLACE\n")
STEPS = [{"action": "read", "file": "shop/pricing.py", "start": 1, "end": 30},
         {"action": "read", "file": "shop/cart.py", "start": 1, "end": 30}]


class Seeded(EvalModel):
    """EvalModel that restarts its script after each `done`, and records the
    sampler the config held and the messages it was sent at every step."""

    def __init__(self, cfg, reply=FIX, steps=STEPS):
        super().__init__(reply, steps)
        self.cfg = cfg
        self.seen = []                       # (sampler, messages) per step
        self._i = 0

    def check(self):
        pass

    def warmup(self):
        pass

    def chat_json(self, messages, schema, max_tokens, **kw):
        self.seen.append((dict(self.cfg.sampler or {}),
                          copy.deepcopy(messages)))
        self.stats["calls"] += 1
        self.stats["prompt_tokens"] += 100
        act = self.steps[self._i] if self._i < len(self.steps) \
            else {"action": "done"}
        self._i = 0 if act["action"] == "done" else self._i + 1
        return {"reasoning": "r", **act}


def _args(tmp_path, **kw):
    ns = dict(task=["py-bulk-off-by-one"], list=False,
              out=str(tmp_path / "r.json"), baseline=None, repeats=1,
              seed=None, max_token_growth=0.2, max_time_growth=0.5)
    ns.update(kw)
    return argparse.Namespace(**ns)


def _run_cli(monkeypatch, tmp_path, cfg, **kw):
    made = []

    def factory(c):
        made.append(Seeded(c))
        return made[0]
    monkeypatch.setattr("pedacito.llm.LLMClient", factory)
    code = evalrun.run_cli(_args(tmp_path, **kw), cfg)
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    return code, report, made[0]


def _cfg(**sampler):
    cfg = Config()
    cfg.log_events = False
    cfg.sources = {}
    cfg.sampler = dict(sampler)
    return cfg


# ----------------------------------------------------------------------- seeds
def test_repeat_seeds_are_base_plus_index_and_off_by_default():
    assert evalrun.repeat_seeds(_cfg(), 1, None) == [None]      # off: untouched
    assert evalrun.repeat_seeds(_cfg(), 3, None) == [0, 1, 2]
    assert evalrun.repeat_seeds(_cfg(), 3, 10) == [10, 11, 12]
    assert evalrun.repeat_seeds(_cfg(), 1, 7) == [7]
    # the user's own seed is the base when --seed is not given
    assert evalrun.repeat_seeds(_cfg(seed=100), 2, None) == [100, 101]
    assert evalrun.repeat_seeds(_cfg(seed=100), 1, None) == [None]


def test_cli_default_run_does_not_touch_the_sampler(monkeypatch, tmp_path):
    cfg = _cfg(top_k=40)
    code, report, client = _run_cli(monkeypatch, tmp_path, cfg)
    assert report["repeats"] == 1 and report["seeds"] == []
    assert all(r["repeat"] == 0 and "seed" not in r for r in report["rows"])
    assert all(s == {"top_k": 40} for s, _m in client.seen)
    assert cfg.sampler == {"top_k": 40}


def test_cli_repeats_seed_each_repeat_without_touching_the_users_config(
        monkeypatch, tmp_path):
    cfg = _cfg(top_k=40)
    code, report, client = _run_cli(monkeypatch, tmp_path, cfg, repeats=3,
                                    seed=5)
    assert code == 0
    assert report["repeats"] == 3 and report["seeds"] == [5, 6, 7]
    assert [r["seed"] for r in report["rows"]] == [5, 6, 7]
    assert [r["repeat"] for r in report["rows"]] == [0, 1, 2]
    by_seed = {s.get("seed") for s, _m in client.seen}
    assert by_seed == {5, 6, 7}
    assert all(s["top_k"] == 40 for s, _m in client.seen)   # user keys kept
    assert cfg.sampler == {"top_k": 40}                     # original intact
    s = report["summary"]
    assert s["repeats"] == 3 and s["tasks"] == 1 and s["runs"] == 3
    assert s["per_task"]["py-bulk-off-by-one"]["runs"] == 3
    assert s["spread"]["mean_calls"]["sd"] == 0


def test_seeded_repeats_leave_prompts_and_replay_digests_unchanged(
        monkeypatch, tmp_path):
    """The seed rides in the request body, never in the messages: the system
    prompt and the first user turn are byte-identical across the steps of a
    gather and across repeats, and --record/--replay digests still match."""
    _code, _report, client = _run_cli(monkeypatch, tmp_path, _cfg(),
                                      repeats=3, seed=1)
    gathers = [m for _s, m in client.seen]
    assert len(gathers) == 9                       # 3 steps x 3 repeats
    for rep in range(3):
        steps = gathers[rep * 3:(rep + 1) * 3]
        for m in steps[1:]:
            assert json.dumps(m[:2]) == json.dumps(steps[0][:2])
            assert m[0]["role"] == "system" and m[1]["role"] == "user"
    first, second = gathers[0:3], gathers[3:6]
    for a, b in zip(first, second):
        assert replay._digest(a) == replay._digest(b)
        assert json.dumps(a) == json.dumps(b)


def test_user_sampler_seed_is_the_base_when_seed_flag_is_absent(
        monkeypatch, tmp_path):
    _c, report, _client = _run_cli(monkeypatch, tmp_path, _cfg(seed=50),
                                   repeats=2)
    assert report["seeds"] == [50, 51]


def test_cli_rejects_zero_repeats(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("pedacito.llm.LLMClient", lambda c: Seeded(c))
    assert evalrun.run_cli(_args(tmp_path, repeats=0), _cfg()) == 2


# --------------------------------------------------------------------- summary
def _row(tid, rep, ok, **kw):
    r = {"id": tid, "kind": "edit", "repeat": rep, "passed": ok,
         "located": 1.0, "first_apply_clean": ok, "calls": 4,
         "prompt_tokens": 1000, "completion_tokens": 100, "cache_hit": 0.5,
         "seconds": 10.0, "failure": "" if ok else "wrong branch"}
    r.update(kw)
    return r


def test_summary_per_task_counts_and_spread():
    rows = [_row("a", 0, True, calls=2), _row("b", 0, False, calls=2),
            _row("a", 1, True, calls=4), _row("b", 1, True, calls=4),
            _row("a", 2, False, calls=6), _row("b", 2, True, calls=6)]
    s = evalrun.summarise(rows)
    assert s["repeats"] == 3 and s["tasks"] == 2 and s["runs"] == 6
    assert s["per_task"] == {"a": {"kind": "edit", "runs": 3, "passes": 2},
                             "b": {"kind": "edit", "runs": 3, "passes": 2}}
    assert s["pass_rate"] == round(4 / 6, 3)
    sp = s["spread"]["mean_calls"]
    assert sp["mean"] == 4.0 and sp["sd"] == 2.0
    assert sp["min"] == 2.0 and sp["max"] == 6.0
    assert s["spread"]["pass_rate"]["sd"] == 0.289     # 0.5, 1.0, 0.5
    table = evalrun.render(rows, s)
    assert "2/3" in table and "over 3 repeats (mean ±sd)" in table
    assert table.count("\na ") + table.count("\nb ") == 2     # one row per task


def test_single_repeat_summary_has_no_spread_and_renders_as_before():
    rows = [_row("a", 0, True), _row("b", 0, False)]
    s = evalrun.summarise(rows)
    assert "spread" not in s and s["repeats"] == 1 and s["tasks"] == 2
    table = evalrun.render(rows, s)
    assert "yes" in table and "NO" in table and "repeats" not in table


# ------------------------------------------------------------ gate reads spread
def _rep_report(fails_per_repeat, seconds=(10.0, 10.0, 10.0)):
    """8 tasks x len(fails) repeats; repeat r fails its first fails[r] tasks."""
    rows = [_row(f"t{i}", r, i >= fails_per_repeat[r],
                 seconds=seconds[r % len(seconds)])
            for r in range(len(fails_per_repeat)) for i in range(8)]
    return {"summary": evalrun.summarise(rows), "rows": rows}


def test_gate_widens_its_margin_for_a_noisy_baseline_only():
    cur = _rep_report([3, 3, 3])
    # quiet baseline: a 3-task drop is plainly a regression
    reg, _ = evalrun.compare_reports(_rep_report([0, 0, 0]), cur)
    assert any(r.startswith("pass rate") for r in reg)
    # noisy baseline (0, 4 and 0 failures per repeat): the same current run is
    # inside 2 standard errors of the baseline's own spread
    notes = []
    reg, _ = evalrun.compare_reports(_rep_report([0, 4, 0]), cur, notes=notes)
    assert not any(r.startswith("pass rate") for r in reg)
    assert any("within tolerance" in n for n in notes)
    # ... but a collapse far outside that spread still fails
    reg, _ = evalrun.compare_reports(_rep_report([0, 4, 0]),
                                     _rep_report([8, 8, 8]))
    assert any(r.startswith("pass rate") for r in reg)


def test_gate_time_growth_must_exceed_the_repeat_spread():
    cur = _rep_report([0, 0, 0], seconds=(26.0, 26.0, 26.0))
    quiet = _rep_report([0, 0, 0], seconds=(16.0, 16.0, 16.0))
    reg, _ = evalrun.compare_reports(quiet, cur)
    assert any("mean seconds" in r for r in reg)               # +62%, no noise
    noisy = _rep_report([0, 0, 0], seconds=(10.0, 30.0, 10.0))
    notes = []
    reg, _ = evalrun.compare_reports(noisy, cur, notes=notes)
    assert not any("mean seconds" in r for r in reg)
    assert any("standard errors" in n for n in notes)


def test_report_from_run_cli_round_trips_through_the_gate(monkeypatch,
                                                          tmp_path):
    _c, report, _client = _run_cli(monkeypatch, tmp_path, _cfg(), repeats=3,
                                   seed=0)
    assert evalrun.compare_reports(report, report) == ([], [])
    again = copy.deepcopy(report)
    for r in again["rows"]:
        r["passed"] = False
    again["summary"] = evalrun.summarise(again["rows"])
    notes = []
    reg, _ = evalrun.compare_reports(report, again, notes=notes)
    # a one-task fixture losing its only task is one task-equivalent: noted,
    # not gated (the CLI's own "pass rate 0" exit code still fails the run)
    assert reg == []
    assert any("task 'py-bulk-off-by-one': 3/3 -> 0/3" in n for n in notes)
