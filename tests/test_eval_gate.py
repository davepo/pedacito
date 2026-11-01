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
