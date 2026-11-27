"""Eval v2 scorecards (WP-4106A): the stratified corpus skeleton, schema-v2
validation, the `eval --scorecard` path and the comparison gate.

Everything runs offline. Rows are synthetic test fixtures built here (they
are not measurements), or come from a scripted fake model. No model server,
network, Node or LLM is needed, and nothing here reports a measured result.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pedacito import evalrun
from pedacito.config import Config

from test_features_misc import EvalModel

FIX = ("Changed the threshold.\n--- FILE: shop/pricing.py\n<<<<<<< SEARCH\n"
       "    if quantity > 10:\n=======\n    if quantity >= 10:\n>>>>>>> REPLACE\n")
STEPS = [{"action": "read", "file": "shop/pricing.py", "start": 1, "end": 30}]
PLACEHOLDER_STRATA = ("api", "environment", "flake")
RUNNABLE_STRATA = ("edit_format", "multi_file", "security", "context")


def _meta(seeds=(1, 2), **kw) -> dict:
    meta = {"harness_version": "0.4.0", "model": "fake-model",
            "provider": "fake", "profile_hash": "a" * 64,
            "tokenizer": evalrun.TOKEN_PROVENANCE,
            "environment": {"os": "test", "python": "3.11"},
            "seeds": list(seeds), "disable_thinking": False,
            "taxonomy_version": evalrun.TAXONOMY_VERSION,
            "metrics_version": evalrun.METRICS_VERSION}
    meta.update(kw)
    return meta


def _row(task, repeat, seed, *, passed=True, label="") -> dict:
    """One row shaped like `evalrun.run_task` returns it."""
    row = {"id": task.id, "fixture": task.fixture, "kind": task.kind,
           "hard": task.hard, "repeat": repeat, "seed": seed,
           "passed": passed, "failure": "" if passed else label,
           "calls": 3, "prompt_tokens": 1000, "completion_tokens": 200,
           "cache_hit": 0.0, "seconds": 1.5, "located": 1.0,
           "repair_calls": 0}
    if task.kind == "edit":
        row["first_apply_clean"] = passed
        if task.test_cmd:
            row["tests_ok"] = bool(passed)
        else:
            row["tests_ok"] = None
    return row


def _corpus_rows(seeds=(1, 2), fail=lambda task: None) -> list[dict]:
    """A full synthetic run over every task. `fail(task)` returns a failure
    label for a task that should fail, or None for one that passes."""
    rows = []
    for rep, seed in enumerate(seeds):
        for task in evalrun.TASKS:
            label = fail(task)
            rows.append(_row(task, rep, seed, passed=label is None,
                             label=label or ""))
    return rows


def _card(seeds=(1, 2), rows=None, **meta) -> dict:
    rows = _corpus_rows(seeds) if rows is None else rows
    return evalrun.make_scorecard(rows, metadata=_meta(seeds, **meta))


def _with_task(rows, task_id, key, value) -> list[dict]:
    out = copy.deepcopy(rows)
    for r in out:
        if r["id"] == task_id:
            r[key] = value
    return out


# ------------------------------------------------------------- the skeleton --
def test_skeleton_covers_every_stratum_with_entries():
    seen = {e.stratum for e in evalrun.SCORECARD_SKELETON}
    assert seen == set(evalrun.STRATA)
    assert len(evalrun.STRATA) == 7


def test_runnable_entries_are_exactly_the_eval_tasks():
    runnable = {e.id for e in evalrun.SCORECARD_SKELETON if e.runnable}
    assert runnable == {t.id for t in evalrun.TASKS}


def test_runnable_strata_are_backed_by_runnable_tasks():
    for stratum in RUNNABLE_STRATA:
        entries = [e for e in evalrun.SCORECARD_SKELETON
                   if e.stratum == stratum]
        assert any(e.runnable for e in entries), stratum


def test_placeholder_strata_are_marked_non_runnable_with_a_reason():
    for stratum in PLACEHOLDER_STRATA:
        entries = [e for e in evalrun.SCORECARD_SKELETON
                   if e.stratum == stratum]
        assert entries, stratum
        assert all(not e.runnable for e in entries), stratum
        assert all("no fixture yet" in e.note for e in entries), stratum


def test_placeholders_never_produce_rows_and_report_not_measured():
    card = _card()
    by = card["summary"]["by_stratum"]
    for stratum in PLACEHOLDER_STRATA:
        assert by[stratum]["runnable"] is False
        assert by[stratum]["rows"] == 0
        assert by[stratum]["pass"] == {"numerator": 0, "denominator": 0,
                                       "rate": None}
    for stratum in RUNNABLE_STRATA:
        assert by[stratum]["runnable"] is True
        assert by[stratum]["rows"] > 0


def test_corpus_fingerprint_is_stable_and_changes_with_the_corpus(monkeypatch):
    first = evalrun.corpus_fingerprint()
    assert first == evalrun.corpus_fingerprint()
    assert len(first) == 64
    changed = [dataclasses.replace(t, prompt=t.prompt + " ")
               if t.id == "py-cart-remove" else t for t in evalrun.TASKS]
    monkeypatch.setattr(evalrun, "TASKS", changed)
    assert evalrun.corpus_fingerprint() != first


# --------------------------------------------- one synthetic fixture per stratum
@pytest.mark.parametrize("stratum", RUNNABLE_STRATA)
def test_each_runnable_stratum_has_a_validated_fixture(stratum):
    """A synthetic run where only this stratum's tasks lose. The card must
    validate and the stratum's own numerator must fall."""
    members = {e.id for e in evalrun.SCORECARD_SKELETON
               if e.stratum == stratum and e.runnable}
    label = "injection followed" if stratum == "security" else "wrong file"

    def fail(task):
        if task.id in members:
            return label if task.kind == "edit" or stratum == "security" \
                else "retrieval miss"
        return None

    good = _card()
    bad = _card(rows=_corpus_rows(fail=fail))
    assert evalrun.validate_scorecard(bad) == []
    b = good["summary"]["by_stratum"][stratum]["pass"]
    c = bad["summary"]["by_stratum"][stratum]["pass"]
    assert c["numerator"] < b["numerator"]
    notes: list[str] = []
    evalrun.compare_scorecards(good, bad, notes=notes)
    assert any(n.startswith(f"stratum {stratum}: pass") for n in notes)


@pytest.mark.parametrize("stratum", PLACEHOLDER_STRATA)
def test_placeholder_stratum_rows_are_refused(stratum):
    card = _card()
    ghost = copy.deepcopy(card["rows"][0])
    ghost["id"] = next(e.id for e in evalrun.SCORECARD_SKELETON
                       if e.stratum == stratum)
    ghost["stratum"] = stratum
    card["rows"].append(ghost)
    errors = evalrun.validate_scorecard(card)
    assert any("not a runnable" in e or "no runnable" in e or "stratum" in e
               for e in errors), errors


# ------------------------------------------------------ build and validate --
def test_make_scorecard_produces_a_valid_schema_v2_card():
    card = _card()
    assert card["schema_version"] == 2
    assert card["corpus_id"] == evalrun.CORPUS_ID
    assert card["corpus_sha256"] == evalrun.corpus_fingerprint()
    assert card["task_ids"] == sorted(t.id for t in evalrun.TASKS)
    assert card["seeds"] == [1, 2]
    assert len(card["rows"]) == 2 * len(evalrun.TASKS)
    assert evalrun.validate_scorecard(card) == []
    assert set(card["summary"]) >= {"pass", "first_apply_clean",
                                    "repair_calls", "prompt_tokens",
                                    "completion_tokens", "seconds",
                                    "failure_labels", "by_stratum"}
    assert card["summary"]["pass"]["denominator"] == len(card["rows"])


def test_summary_rates_carry_denominators():
    card = _card()
    p = card["summary"]["pass"]
    assert p["numerator"] == p["denominator"] == len(card["rows"])
    assert p["rate"] == 1.0
    fac = card["summary"]["first_apply_clean"]
    edit_rows = [r for r in card["rows"] if r["kind"] == "edit"]
    assert fac["denominator"] == len(edit_rows)


def test_null_means_unavailable_and_is_left_out_of_means():
    rows = _corpus_rows()
    rows[0] = dict(rows[0], prompt_tokens=None, seconds=None)
    card = _card(rows=rows)
    assert evalrun.validate_scorecard(card) == []
    assert card["summary"]["prompt_tokens"]["count"] == len(rows) - 1
    assert card["rows"][0]["prompt_tokens"] is None


def test_crash_row_is_a_failure_with_null_tokens():
    rows = _corpus_rows()
    crash = dict(rows[0], passed=False, failure="crash", error="boom")
    crash.pop("prompt_tokens")
    crash.pop("seconds")
    crash.pop("first_apply_clean", None)
    rows[0] = crash
    card = _card(rows=rows)
    assert evalrun.validate_scorecard(card) == []
    row = card["rows"][0]
    assert row["passed"] is False and row["failure_labels"] == ["crash"]
    assert row["prompt_tokens"] is None and row["first_apply_clean"] is None


def test_make_scorecard_refuses_missing_rows_instead_of_inventing_passes():
    rows = _corpus_rows()[:-1]
    with pytest.raises(evalrun.ScorecardError, match="missing row"):
        _card(rows=rows)


def test_make_scorecard_refuses_missing_metadata():
    meta = _meta()
    del meta["model"]
    with pytest.raises(evalrun.ScorecardError, match="model"):
        evalrun.make_scorecard(_corpus_rows(), metadata=meta)


def test_make_scorecard_refuses_a_row_outside_the_seeds():
    rows = _corpus_rows()
    rows[0] = dict(rows[0], repeat=5)
    with pytest.raises(evalrun.ScorecardError, match="outside"):
        _card(rows=rows)


MUTATIONS = {
    "schema v1": lambda c: c.update(schema_version=1),
    "missing model": lambda c: c.pop("model"),
    "bool as taxonomy": lambda c: c.update(taxonomy_version=True),
    "wrong corpus hash": lambda c: c.update(corpus_sha256="0" * 64),
    "old metrics": lambda c: c.update(metrics_version=1),
    "blank provider": lambda c: c.update(provider="  "),
    "environment not strings": lambda c: c["environment"].update(python=3),
    "seeds empty": lambda c: c.update(seeds=[]),
    "seed not int": lambda c: c.update(seeds=[1, "2"]),
    "task list truncated": lambda c: c.update(task_ids=c["task_ids"][1:]),
    "dropped row": lambda c: c["rows"].pop(),
    "duplicate row": lambda c: c["rows"].append(copy.deepcopy(c["rows"][0])),
    "unknown task": lambda c: c["rows"][0].update(id="no-such-task"),
    "wrong stratum": lambda c: c["rows"][0].update(stratum="api"),
    "wrong kind": lambda c: c["rows"][0].update(kind="question"),
    "wrong seed": lambda c: c["rows"][0].update(seed=99),
    "repeat out of range": lambda c: c["rows"][0].update(repeat=7),
    "passed as string": lambda c: c["rows"][0].update(passed="yes"),
    "negative tokens": lambda c: c["rows"][0].update(prompt_tokens=-1),
    "float repair count": lambda c: c["rows"][0].update(repair_calls=1.5),
    "unknown label": lambda c: c["rows"][0].update(
        passed=False, failure_labels=["made up"]),
    "passing row with label": lambda c: c["rows"][0].update(
        passed=True, failure_labels=["wrong file"]),
    "question with first-apply": lambda c: next(
        r for r in c["rows"] if r["kind"] == "question").update(
            first_apply_clean=True),
    "fabricated pass count": lambda c: c["summary"]["pass"].update(
        numerator=c["summary"]["pass"]["numerator"] + 1),
    "stale summary": lambda c: c["summary"]["failure_labels"].update(x=1),
    "seconds NaN": lambda c: c["rows"][0].update(seconds=float("nan")),
    "seconds infinite": lambda c: c["rows"][0].update(seconds=float("inf")),
    "located above one": lambda c: c["rows"][0].update(located=1.5),
    "located NaN": lambda c: c["rows"][0].update(located=float("nan")),
    "located as string": lambda c: c["rows"][0].update(located="1.0"),
    "pass without verified test": lambda c: next(
        r for r in c["rows"] if r["id"] == "py-bulk-off-by-one").update(
            tests_ok=None),
    "tests_ok on a question": lambda c: next(
        r for r in c["rows"] if r["kind"] == "question").update(tests_ok=True),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_validation_rejects_each_corruption(name):
    card = copy.deepcopy(_card())
    MUTATIONS[name](card)
    errors = evalrun.validate_scorecard(card)
    assert errors, name


def test_validation_rejects_non_objects():
    assert evalrun.validate_scorecard([]) == ["scorecard must be a JSON object"]
    assert evalrun.validate_scorecard("x") == ["scorecard must be a JSON object"]


def test_validation_is_deterministic():
    card = copy.deepcopy(_card())
    card["rows"].pop()
    assert evalrun.validate_scorecard(card) == evalrun.validate_scorecard(card)


def test_scorecard_round_trips_through_json_and_still_validates():
    card = json.loads(json.dumps(_card()))
    assert evalrun.validate_scorecard(card) == []


def test_scorecard_dataclass_matches_the_card_shape():
    card = _card()
    assert set(card) == {f.name for f in dataclasses.fields(evalrun.Scorecard)}


# ----------------------------------------------------------- the comparison --
def test_identical_scorecards_pass_with_no_regression():
    card = _card()
    reg, imp = evalrun.compare_scorecards(card, copy.deepcopy(card))
    assert reg == [] and imp == []


def test_a_collapsed_stratum_is_a_regression():
    base = _card()
    cur = _card(rows=_corpus_rows(
        fail=lambda t: "wrong file" if t.kind == "edit" else None))
    reg, _ = evalrun.compare_scorecards(base, cur)
    assert any("pass rate" in r or "first-apply" in r for r in reg)


def test_injection_success_is_a_security_regression():
    base = _card()
    cur = _card(rows=_corpus_rows(
        fail=lambda t: "injection followed" if t.id == "py-injection-summary"
        else None))
    reg, _ = evalrun.compare_scorecards(base, cur)
    assert any("injection followed" in r for r in reg)


def test_prompt_growth_is_refused_by_default_and_allowed_when_asked():
    base = _card()
    rows = _corpus_rows()
    for r in rows:
        r["prompt_tokens"] += 1
    cur = _card(rows=rows)
    reg, _ = evalrun.compare_scorecards(base, cur)
    assert any("prompt tokens" in r for r in reg)
    reg, _ = evalrun.compare_scorecards(base, cur, max_prompt_growth=0.5)
    assert not any("prompt tokens" in r for r in reg)


def test_time_growth_uses_its_own_tolerance():
    base = _card()
    rows = _corpus_rows()
    for r in rows:
        r["seconds"] = 4.5
    cur = _card(rows=rows)
    reg, _ = evalrun.compare_scorecards(base, cur)
    assert any("seconds" in r for r in reg)
    reg, _ = evalrun.compare_scorecards(base, cur, max_time_growth=5.0)
    assert not any("seconds" in r for r in reg)


def test_unmeasured_prompt_tokens_cannot_pass_the_growth_gate():
    base = _card()
    rows = _corpus_rows()
    for r in rows:
        r["prompt_tokens"] = None
    cur = _card(rows=rows)
    reg, _ = evalrun.compare_scorecards(base, cur)
    assert any("not measured" in r for r in reg)


def test_unmeasured_first_apply_on_an_edit_row_is_a_regression():
    base = _card()
    rows = _corpus_rows()
    edit = next(r for r in rows if r["kind"] == "edit")
    edit["first_apply_clean"] = None
    cur = _card(rows=rows)
    reg, _ = evalrun.compare_scorecards(base, cur)
    assert any("first-apply-clean not measured" in r for r in reg)


@pytest.mark.parametrize("field,value", [
    ("seeds", [5, 6]),
    ("model", "other-model"),
    ("provider", "other"),
    ("profile_hash", "b" * 64),
    ("tokenizer", "exact:other"),
    ("disable_thinking", True),
    ("taxonomy_version", 1),
])
def test_incomparable_scorecards_are_refused(field, value):
    base = _card()
    if field == "seeds":
        cur = _card(seeds=(5, 6), rows=_corpus_rows(seeds=(5, 6)))
    else:
        cur = _card(**{field: value})
    with pytest.raises(evalrun.ScorecardError, match=field):
        evalrun.compare_scorecards(base, cur)


def test_a_card_from_a_different_corpus_is_refused():
    base = _card()
    cur = copy.deepcopy(base)
    cur["corpus_sha256"] = "f" * 64
    with pytest.raises(evalrun.ScorecardError, match="invalid current"):
        evalrun.compare_scorecards(base, cur)


def test_an_invalid_baseline_is_refused():
    base = copy.deepcopy(_card())
    base["rows"].pop()
    with pytest.raises(evalrun.ScorecardError, match="invalid baseline"):
        evalrun.compare_scorecards(base, _card())


def test_notes_carry_per_stratum_rates_and_are_not_gated():
    base = _card()
    reg, imp = evalrun.compare_scorecards(base, _card())
    notes: list[str] = []
    evalrun.compare_scorecards(base, _card(), notes=notes)
    for stratum in RUNNABLE_STRATA:
        assert any(n.startswith(f"stratum {stratum}: pass") for n in notes)
    for stratum in PLACEHOLDER_STRATA:
        assert f"stratum {stratum}: not measured (non-runnable placeholder)" \
            in notes
    assert reg == []


# --------------------------------------------- eval -> scorecard, offline ----
class _Fake(EvalModel):
    def check(self):
        pass

    def warmup(self):
        pass


def _eval_args(tmp_path, **kw):
    ns = dict(task=[], list=False, out=None, baseline=None, repeats=1,
              seed=None, max_token_growth=0.2, max_time_growth=0.5,
              scorecard=None)
    ns.update(kw)
    return argparse.Namespace(**ns)


def _cfg() -> Config:
    cfg = Config()
    cfg.log_events = False
    cfg.sources = {}
    cfg.sampler = {}
    return cfg


def test_eval_without_scorecard_keeps_the_legacy_report_shape(
        monkeypatch, tmp_path):
    monkeypatch.setattr(evalrun, "TASKS",
                        [t for t in evalrun.TASKS
                         if t.id == "py-bulk-off-by-one"])
    monkeypatch.setattr("pedacito.llm.LLMClient", lambda c: _Fake(FIX, STEPS))
    out = tmp_path / "r.json"
    code = evalrun.run_cli(_eval_args(tmp_path, out=str(out)), _cfg())
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert set(report) == {"model", "pack", "disable_thinking", "taxonomy",
                           "metrics", "repeats", "seeds", "summary", "rows"}
    assert not list(tmp_path.glob("*scorecard*"))


def test_scorecard_needs_an_explicit_seed(monkeypatch, tmp_path):
    monkeypatch.setattr("pedacito.llm.LLMClient", lambda c: _Fake(FIX, STEPS))
    target = tmp_path / "card.json"
    args = _eval_args(tmp_path, scorecard=str(target),
                      task=["py-bulk-off-by-one"])
    assert evalrun.run_cli(args, _cfg()) == 2
    assert not target.exists()


def test_scorecard_refuses_a_task_subset_before_running(monkeypatch, tmp_path):
    monkeypatch.setattr("pedacito.llm.LLMClient",
                        lambda c: pytest.fail("no model may be built"))
    target = tmp_path / "card.json"
    args = _eval_args(tmp_path, scorecard=str(target), seed=1,
                      task=["py-bulk-off-by-one"])
    assert evalrun.run_cli(args, _cfg()) == 2
    assert not target.exists()


def test_eval_writes_a_valid_scorecard_from_a_real_offline_run(
        monkeypatch, tmp_path):
    _one_edit(monkeypatch, True)
    target = tmp_path / "card.json"
    args = _eval_args(tmp_path, scorecard=str(target), seed=4)
    assert evalrun.run_cli(args, _cfg()) == 0
    card = json.loads(target.read_text(encoding="utf-8"))
    assert evalrun.validate_scorecard(card) == []
    assert card["seeds"] == [4]
    assert card["task_ids"] == ["py-bulk-off-by-one"]
    (row,) = card["rows"]
    assert row["seed"] == 4 and row["repeat"] == 0
    assert row["stratum"] == "edit_format"
    assert row["prompt_tokens"] and row["prompt_tokens"] > 0
    assert card["summary"]["prompt_tokens"]["count"] == 1
    assert card["provider"] == "unspecified"
    assert len(card["profile_hash"]) == 64
    assert "api_key" not in json.dumps(card).lower()


def test_eval_scorecard_is_comparable_with_itself(monkeypatch, tmp_path):
    _one_edit(monkeypatch, True)
    target = tmp_path / "card.json"
    evalrun.run_cli(_eval_args(tmp_path, scorecard=str(target), seed=4), _cfg())
    card = json.loads(target.read_text(encoding="utf-8"))
    reg, _ = evalrun.compare_scorecards(card, copy.deepcopy(card))
    assert reg == []


def test_scorecard_metadata_hashes_profile_and_never_stores_the_endpoint():
    cfg = _cfg()
    cfg.model = "m"
    cfg.base_url = "http://user:secret@example.invalid/v1"
    meta = evalrun._scorecard_metadata(cfg, [1])
    assert "secret" not in json.dumps(meta)
    assert "example.invalid" not in json.dumps(meta)
    assert meta["seeds"] == [1]
    other = _cfg()
    other.model = "m"
    other.sampler = {"temperature": 0.9}
    assert evalrun._scorecard_metadata(other, [1])["profile_hash"] \
        != meta["profile_hash"]


# ----------------------------------------- review fixes: pairs, types, prompts --
def test_duplicate_pair_is_refused_by_make_scorecard():
    rows = _corpus_rows() + [copy.deepcopy(_corpus_rows()[0])]
    with pytest.raises(evalrun.ScorecardError, match="duplicate row"):
        _card(rows=rows)


def test_duplicate_pair_is_refused_even_with_a_recomputed_summary():
    card = copy.deepcopy(_card())
    card["rows"].append(copy.deepcopy(card["rows"][0]))
    card["summary"] = evalrun._summarize_rows(card["rows"])
    errors = evalrun.validate_scorecard(card)
    assert any("duplicate row" in e for e in errors), errors


def test_a_duplicate_that_inflates_passes_is_refused_when_a_failure_is_dropped():
    """Swap a failing row for a copy of a passing one, then recompute the
    summary so the totals look self-consistent. Every pair must still be
    exactly one row, so this is refused."""
    card = copy.deepcopy(_card())
    victim = next(i for i, r in enumerate(card["rows"])
                  if r["kind"] == "edit" and r["repeat"] == 0 and i > 0)
    card["rows"][victim] = copy.deepcopy(card["rows"][0])
    card["summary"] = evalrun._summarize_rows(card["rows"])
    errors = evalrun.validate_scorecard(card)
    assert any("duplicate row" in e for e in errors), errors
    assert any("missing row" in e for e in errors), errors


def test_every_pair_must_have_exactly_one_row_with_a_recomputed_summary():
    card = copy.deepcopy(_card())
    card["rows"].pop()
    card["rows"].append(copy.deepcopy(card["rows"][0]))
    card["summary"] = evalrun._summarize_rows(card["rows"])
    errors = evalrun.validate_scorecard(card)
    assert any("duplicate row" in e for e in errors)
    assert any("missing row" in e for e in errors)


@pytest.mark.parametrize("key,bad", [
    ("passed", "false"),          # the string 'false' must not become a pass
    ("passed", 0),
    ("passed", None),
    ("first_apply_clean", "yes"),
    ("repair_calls", "3"),
    ("repair_calls", True),
    ("prompt_tokens", 1.5),
    ("prompt_tokens", True),
    ("completion_tokens", "200"),
    ("seconds", "1.5"),
    ("seconds", True),
    ("failure", ["wrong file"]),
    ("kind", "essay"),
    ("repeat", None),
    ("repeat", "0"),
])
def test_malformed_runner_values_are_refused_not_coerced(key, bad):
    rows = _corpus_rows()
    target = next(i for i, r in enumerate(rows) if r["kind"] == "edit") \
        if key == "first_apply_clean" else 0
    rows[target] = dict(rows[target], **{key: bad})
    with pytest.raises(evalrun.ScorecardError):
        _card(rows=rows)


def test_string_false_is_not_turned_into_a_pass():
    rows = _corpus_rows()
    rows[0] = dict(rows[0], passed="false", failure="")
    with pytest.raises(evalrun.ScorecardError, match="passed must be a boolean"):
        _card(rows=rows)


def test_contradictory_pass_and_label_are_refused():
    rows = _corpus_rows()
    rows[0] = dict(rows[0], passed=True, failure="wrong file")
    with pytest.raises(evalrun.ScorecardError, match="passed but carries"):
        _card(rows=rows)


def test_a_failure_without_a_label_is_refused():
    rows = _corpus_rows()
    rows[0] = dict(rows[0], passed=False, failure="")
    with pytest.raises(evalrun.ScorecardError, match="without a failure label"):
        _card(rows=rows)


def test_question_row_with_first_apply_is_refused():
    rows = _corpus_rows()
    q = next(i for i, r in enumerate(rows) if r["kind"] == "question")
    rows[q] = dict(rows[q], first_apply_clean=True)
    with pytest.raises(evalrun.ScorecardError, match="only for edit rows"):
        _card(rows=rows)


def test_a_non_object_row_is_refused():
    rows = _corpus_rows()
    rows[0] = "not a row"
    with pytest.raises(evalrun.ScorecardError, match="must be an object"):
        _card(rows=rows)


def test_missing_measurements_are_null_not_zero():
    rows = _corpus_rows()
    rows[0] = {k: v for k, v in rows[0].items()
               if k not in ("prompt_tokens", "seconds")}
    card = _card(rows=rows)
    assert card["rows"][0]["prompt_tokens"] is None
    assert card["rows"][0]["seconds"] is None
    assert evalrun.validate_scorecard(card) == []


def test_tokenizer_is_labelled_unverified_never_exact():
    meta = evalrun._scorecard_metadata(_cfg(), [1])
    assert meta["tokenizer"] == evalrun.TOKEN_PROVENANCE
    assert meta["tokenizer"].startswith("unverified")
    assert "exact" not in meta["tokenizer"]
    assert "endpoint-reported" not in json.dumps(meta)
    assert evalrun.validate_scorecard(_card()) == []


def _prompt_nulled(card_rows, indexes):
    rows = copy.deepcopy(card_rows)
    for i in indexes:
        rows[i]["prompt_tokens"] = None
    return rows


def test_one_missing_prompt_count_on_current_fails_closed_against_growth():
    """Current grew by 50% on every row it did measure, and one row is
    unmeasured. Unequal denominators must not hide the growth."""
    base = _card()
    rows = _corpus_rows()
    for r in rows[1:]:
        r["prompt_tokens"] = int(r["prompt_tokens"] * 1.5)
    rows[0]["prompt_tokens"] = None
    cur = _card(rows=rows)
    reg, _ = evalrun.compare_scorecards(base, cur, max_prompt_growth=1.0)
    assert any("not measured for 1 of" in r for r in reg), reg


def test_one_missing_prompt_count_on_baseline_fails_closed():
    base = _card(rows=_prompt_nulled(_corpus_rows(), [0]))
    cur = _card()
    reg, _ = evalrun.compare_scorecards(base, cur, max_prompt_growth=10.0)
    assert any("baseline: prompt tokens not measured for 1 of" in r
               for r in reg), reg


def test_a_crash_row_without_prompt_count_fails_the_growth_gate():
    rows = _corpus_rows()
    rows[0] = dict(rows[0], passed=False, failure="crash")
    rows[0].pop("prompt_tokens")
    base = _card()
    cur = _card(rows=rows)
    reg, _ = evalrun.compare_scorecards(base, cur, max_prompt_growth=10.0)
    assert any("not measured for 1 of" in r for r in reg), reg


def test_complete_paired_prompt_counts_still_compare():
    base = _card()
    reg, _ = evalrun.compare_scorecards(base, copy.deepcopy(base))
    assert not any("prompt" in r for r in reg)


# ---------------------------- review fixes: verified tests, locate, finiteness --
def _one_edit(monkeypatch, tests_result):
    """One edit task through the real `run_task`, with the test runner
    replaced by `tests_result` (True, False, or None for unavailable)."""
    monkeypatch.setattr(evalrun, "TASKS",
                        [t for t in evalrun.TASKS
                         if t.id == "py-bulk-off-by-one"])
    monkeypatch.setattr(evalrun, "_run_tests",
                        lambda root, task: tests_result)
    monkeypatch.setattr("pedacito.llm.LLMClient",
                        lambda c: _Fake(FIX, STEPS))


def test_unavailable_test_runner_stays_a_legacy_pass_but_is_not_a_card(
        monkeypatch, tmp_path, capsys):
    _one_edit(monkeypatch, None)
    out = tmp_path / "legacy.json"
    assert evalrun.run_cli(_eval_args(tmp_path, out=str(out)), _cfg()) == 0
    row = json.loads(out.read_text(encoding="utf-8"))["rows"][0]
    assert row["passed"] is True and row["tests_ok"] is None   # legacy kept
    target = tmp_path / "card.json"
    assert evalrun.run_cli(_eval_args(tmp_path, scorecard=str(target), seed=4),
                           _cfg()) == 2
    assert not target.exists()
    assert "required test" in capsys.readouterr().err


def test_verified_test_success_produces_a_valid_card(monkeypatch, tmp_path):
    _one_edit(monkeypatch, True)
    target = tmp_path / "card.json"
    assert evalrun.run_cli(_eval_args(tmp_path, scorecard=str(target), seed=4),
                           _cfg()) == 0
    card = json.loads(target.read_text(encoding="utf-8"))
    assert evalrun.validate_scorecard(card) == []
    (row,) = card["rows"]
    assert row["passed"] is True and row["tests_ok"] is True


def test_failed_test_run_is_recorded_as_a_failure_not_a_pass(
        monkeypatch, tmp_path):
    _one_edit(monkeypatch, False)
    target = tmp_path / "card.json"
    assert evalrun.run_cli(_eval_args(tmp_path, scorecard=str(target), seed=4),
                           _cfg()) == 1          # legacy: pass rate 0 exits 1
    card = json.loads(target.read_text(encoding="utf-8"))
    assert evalrun.validate_scorecard(card) == []
    (row,) = card["rows"]
    assert row["passed"] is False and row["tests_ok"] is False
    assert row["failure_labels"] == ["test regression"]


def test_make_scorecard_refuses_a_pass_without_its_required_test():
    rows = _corpus_rows()
    i = next(i for i, r in enumerate(rows) if r["id"] == "py-bulk-off-by-one")
    rows[i] = dict(rows[i], tests_ok=None)
    with pytest.raises(evalrun.ScorecardError, match="required test is unverified"):
        _card(rows=rows)


def test_a_locate_rate_decline_is_caught_by_the_gate():
    base = _card()
    assert evalrun._legacy_view(base)["rows"][0]["located"] == 1.0
    rows = _corpus_rows()
    for r in rows:
        r["located"] = 0.5
    reg, _ = evalrun.compare_scorecards(base, _card(rows=rows))
    assert any(r.startswith("locate rate") for r in reg), reg


def test_locate_rate_is_recorded_in_the_summary_with_a_denominator():
    card = _card()
    loc = card["summary"]["located"]
    assert loc["count"] == len(card["rows"])
    assert loc["mean"] == 1.0


def test_an_unmeasured_locate_rate_fails_closed():
    base = _card()
    rows = _corpus_rows()
    rows[0]["located"] = None
    reg, _ = evalrun.compare_scorecards(base, _card(rows=rows))
    assert any("locate rate not measured for 1 of" in r for r in reg), reg


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -0.1])
def test_compare_refuses_non_finite_or_negative_tolerances(bad):
    card = _card()
    with pytest.raises(evalrun.ScorecardError, match="finite"):
        evalrun.compare_scorecards(card, card, max_prompt_growth=bad)
    with pytest.raises(evalrun.ScorecardError, match="finite"):
        evalrun.compare_scorecards(card, card, max_time_growth=bad)


def test_make_scorecard_refuses_a_non_finite_measurement():
    rows = _corpus_rows()
    rows[0] = dict(rows[0], seconds=float("nan"))
    with pytest.raises(evalrun.ScorecardError, match="finite"):
        _card(rows=rows)
