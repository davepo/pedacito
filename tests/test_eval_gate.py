"""CI A/B gate over eval reports and strict replay."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import evalrun, replay


def _report(**kw):
    base = {"pass_rate": 0.8, "locate_rate": 0.9, "first_apply_clean": 0.7,
            "mean_prompt_tokens": 1000, "mean_completion_tokens": 200,
            "mean_calls": 4, "mean_seconds": 10, "injection_followed": 0,
            "failures": {}}
    base.update(kw)
    return {"summary": base, "rows": []}


def test_ab_gate_passes_identical_reports():
    reg, imp = evalrun.compare_reports(_report(), _report())
    assert reg == [] and imp == []
    assert "PASS" in evalrun.render_compare(reg, imp)


def test_ab_gate_flags_quality_token_and_injection_regressions():
    bad = _report(pass_rate=0.7, locate_rate=0.8, first_apply_clean=0.6,
                  mean_prompt_tokens=1500, mean_completion_tokens=400,
                  mean_calls=8, mean_seconds=20, injection_followed=2,
                  failures={"injection followed": 2})
    reg, _imp = evalrun.compare_reports(_report(), bad)
    text = "\n".join(reg)
    assert "pass rate" in text
    assert "locate rate" in text
    assert "first-apply-clean" in text
    assert "mean prompt tokens" in text
    assert "injection followed" in text
    assert "failure 'injection followed'" in text
    assert "FAIL" in evalrun.render_compare(reg, [])


def test_ab_gate_reports_improvements():
    better = _report(pass_rate=0.9, mean_prompt_tokens=700)
    reg, imp = evalrun.compare_reports(_report(), better)
    assert reg == []
    text = "\n".join(imp)
    assert "pass rate" in text and "mean prompt tokens" in text


def test_ab_gate_flags_a_thinking_setting_split():
    """P5: token/time growth across a disable_thinking change measures the
    setting, not the harness -- the gate hard-flags the split and skips the
    growth comparison. Reports without the key (older --out files) compare
    exactly as before."""
    base = _report()
    base["disable_thinking"] = False
    hot = _report(mean_completion_tokens=900, mean_seconds=40)
    hot["disable_thinking"] = True
    reg, _imp = evalrun.compare_reports(base, hot)
    assert any("disable_thinking differs" in r for r in reg)
    assert not any("completion tokens" in r for r in reg)
    assert "FAIL" in evalrun.render_compare(reg, [])
    # the same setting on both sides: growth compares as usual
    base["disable_thinking"] = True
    reg, _ = evalrun.compare_reports(base, hot)
    assert not any("differs" in r for r in reg)
    assert any("mean completion tokens" in r for r in reg)
    # older reports without the key: no flag, growth still compared
    reg, _ = evalrun.compare_reports(_report(), _report(mean_seconds=40))
    assert not any("differs" in r for r in reg)
    assert any("mean seconds" in r for r in reg)


def _runs(passes, kind="edit", failure="wrong branch", **extra):
    """Report built from rows. `passes` maps a task id to one bool per repeat."""
    rows = []
    for tid, outcomes in passes.items():
        for rep, ok in enumerate(outcomes):
            rows.append({"id": tid, "kind": kind, "repeat": rep, "passed": ok,
                         "located": 1.0, "first_apply_clean": ok,
                         "failure": "" if ok else failure})
    rep = {"summary": evalrun.summarise(rows), "rows": rows}
    rep.update(extra)
    return rep


def _eight(flip=(), repeats=1):
    """Eight tasks that all pass in every repeat, except the (task, repeat)
    pairs in `flip`."""
    return _runs({f"t{i}": [(i, r) not in flip for r in range(repeats)]
                  for i in range(8)})


def test_gate_new_failure_label_is_info_not_a_regression():
    """Pass 4's repro: splitting 'wrong answer' into a finer label with the
    same count must not fail the gate."""
    base = _report(failures={"wrong answer": 1})
    cur = _report(failures={"ungrounded citation": 1})
    notes = []
    reg, imp = evalrun.compare_reports(base, cur, notes=notes)
    assert reg == [] and imp == []
    assert any("new failure label 'ungrounded citation'" in n for n in notes)
    assert "PASS" in evalrun.render_compare(reg, imp, notes)
    assert "ungrounded citation" in evalrun.render_compare(reg, imp, notes)
    # the 2-tuple API still works without a notes list
    assert evalrun.compare_reports(base, cur) == ([], [])


def test_gate_label_present_in_both_is_compared_per_run_with_one_task_margin():
    base = _report(failures={"wrong file": 1})
    reg, _ = evalrun.compare_reports(base, _report(failures={"wrong file": 2}))
    assert reg == []                                    # one more task: noise
    reg, _ = evalrun.compare_reports(base, _report(failures={"wrong file": 3}))
    assert any("failure 'wrong file': 1 -> 3" in r for r in reg)
    # counts pool every repeat, so 3 repeats of one failing task are 1 per run
    b3 = _report(failures={"wrong file": 3})
    b3["summary"]["repeats"] = 3
    c3 = _report(failures={"wrong file": 4})
    c3["summary"]["repeats"] = 3
    assert evalrun.compare_reports(b3, c3)[0] == []


def test_gate_a_single_task_flip_is_within_tolerance():
    notes = []
    reg, _ = evalrun.compare_reports(_eight(), _eight(flip={(3, 0)}),
                                     notes=notes)
    assert reg == []
    assert any("within tolerance" in n for n in notes)
    # one flip in one repeat out of three: smaller still
    reg, _ = evalrun.compare_reports(_eight(repeats=3),
                                     _eight(flip={(3, 1)}, repeats=3))
    assert reg == []


def test_gate_more_than_one_task_equivalent_is_a_regression():
    reg, _ = evalrun.compare_reports(_eight(), _eight(flip={(1, 0), (2, 0)}))
    assert any(r.startswith("pass rate: 100% -> 75%") for r in reg)
    assert any(r.startswith("first-apply-clean") for r in reg)
    # across 3 repeats a flip spread over 2 tasks' worth of passes also fails
    flips = {(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)}
    reg, _ = evalrun.compare_reports(_eight(repeats=3),
                                     _eight(flip=flips, repeats=3))
    assert any(r.startswith("pass rate") for r in reg)


def test_gate_a_task_flipping_in_every_repeat_is_one_task_equivalent():
    """Plan A0.1: a single-task flip across 3 repeats does not fail the gate;
    it is noted. A second task flipping as well is beyond the tolerance."""
    base = _eight(repeats=3)
    notes = []
    reg, _ = evalrun.compare_reports(
        base, _eight(flip={(0, 0), (0, 1), (0, 2)}, repeats=3), notes=notes)
    assert reg == []
    assert any("task 't0': 3/3 -> 0/3" in n for n in notes)
    reg, _ = evalrun.compare_reports(
        base, _eight(flip={(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)},
                     repeats=3))
    assert any(r.startswith("pass rate") for r in reg)
    # and a single repeat of one task flipping is never enough
    assert evalrun.compare_reports(_eight(), _eight(flip={(0, 0)}))[0] == []


def test_gate_compares_only_the_tasks_both_reports_ran():
    base = _eight()
    cur = _runs({"t0": [True], "t1": [True]})
    notes = []
    reg, _ = evalrun.compare_reports(base, cur, notes=notes)
    assert reg == [] and any("task sets differ" in n for n in notes)
    bad = _runs({"t0": [False], "t1": [False]})
    reg, _ = evalrun.compare_reports(base, bad)
    assert any(r.startswith("pass rate: 100% -> 0%") for r in reg)


def test_gate_injection_is_strict_per_run():
    base = _report(injection_followed=0)
    base["summary"]["repeats"] = 3
    cur = _report(injection_followed=1)
    cur["summary"]["repeats"] = 3
    reg, _ = evalrun.compare_reports(base, cur)
    assert any("injection followed" in r for r in reg)       # 1 of 3 runs
    same = _report(injection_followed=3)
    same["summary"]["repeats"] = 3
    base3 = _report(injection_followed=3)
    base3["summary"]["repeats"] = 3
    assert evalrun.compare_reports(base3, same)[0] == []


def test_gate_label_known_to_the_baseline_taxonomy_is_not_waved_through():
    """A label the old taxonomy could emit and that jumps is a regression even
    though it is absent from the baseline's failures."""
    base = _eight()
    cur = _runs({f"t{i}": [i >= 3] for i in range(8)}, failure="format error")
    cur["summary"]["failures"] = {"format error": 3}
    reg, _ = evalrun.compare_reports(base, cur)
    assert any("failure 'format error': 0 -> 3" in r for r in reg)


def test_gate_reads_old_reports_whose_rows_have_no_repeat_key():
    base, cur = _eight(), _eight(flip={(2, 0)})
    for rep in (base, cur):
        for r in rep["rows"]:
            r.pop("repeat")
    assert evalrun.compare_reports(base, cur)[0] == []
    cur = _eight(flip={(2, 0), (3, 0)})
    for r in cur["rows"]:
        r.pop("repeat")
    assert evalrun.compare_reports(base, cur)[0] != []


def test_replayer_marks_a_divergent_prompt(tmp_path):
    rec = tmp_path / "run.jsonl"
    rec.write_text(json.dumps({
        "kind": "chat", "n": 0, "digest": "not-this", "preview": "",
        "out": "recorded answer", "flags": {},
    }) + "\n", encoding="utf-8")
    client = replay.Replayer(rec)
    assert client.chat([{"role": "user", "content": "new prompt"}]) \
        == "recorded answer"
    assert client.diverged == [0]
    assert "differed" in client.summary()
