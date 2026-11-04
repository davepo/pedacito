"""Prose failure taxonomy and per-kind aggregation in `pedacito eval`."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pedacito import evalrun

from test_features_misc import EvalModel, _eval_cfg

READ_PRICING = [{"action": "read", "file": "shop/pricing.py", "start": 1,
                 "end": 30}]
READ_CLI = [{"action": "read", "file": "shop/cli.py", "start": 1, "end": 30}]


def _task(tid):
    return next(t for t in evalrun.TASKS if t.id == tid)


def test_medium_fixture_makes_wrong_file_reachable(tmp_path):
    """A0.7: the medium fixture has 40+ files, decoys with similar names, a
    deep call chain, and a gold file -- so editing a decoy scores `wrong file`."""
    assert len(evalrun.FIXTURES["medium"]) >= 40
    for path in ("orders/processor.py", "orders/processors.py",
                 "orders/processor_utils.py", "orders/order_processor.py",
                 "orders/api.py", "orders/service.py"):
        assert path in evalrun.FIXTURES["medium"]
    task = _task("py-medium-threshold")
    assert task.gold_files == ["orders/processor.py"]
    reply = ("--- FILE: orders/processors.py\n<<<<<<< SEARCH\n"
             "        if quantity > 10:\n"
             "=======\n        if quantity >= 10:\n>>>>>>> REPLACE\n")
    row = evalrun.run_task(
        task, EvalModel(reply, [{"action": "read", "file": "orders/processors.py",
                                 "start": 1, "end": 30}]),
        _eval_cfg(), tmp_path)
    assert row["failure"] == "wrong file"


def _ask(tmp_path, tid, reply, steps=None):
    return evalrun.run_task(_task(tid), EvalModel(reply, steps), _eval_cfg(),
                            tmp_path)


def test_question_that_reads_the_gold_file_and_declines_is_an_over_refusal(
        tmp_path):
    row = _ask(tmp_path, "py-where-is-tax",
               "I cannot determine which function adds the tax.", READ_PRICING)
    assert not row["passed"] and row["failure"] == "over-refusal"
    assert _ask(tmp_path, "py-where-is-tax", "", READ_PRICING)["failure"] \
        == "over-refusal"                              # an empty reply too


def test_question_that_never_read_the_gold_file_is_a_retrieval_miss(tmp_path):
    row = _ask(tmp_path, "py-where-is-tax", "It is in cart.py.")
    assert row["failure"] == "retrieval miss"
    # declining after a lookup that missed the file is also a retrieval miss:
    # there was nothing to answer from
    row = _ask(tmp_path, "py-where-is-tax", "I don't know.",
               [{"action": "read", "file": "shop/cart.py", "start": 1,
                 "end": 30}])
    assert row["failure"] == "retrieval miss"


def test_question_citing_what_the_project_lacks_is_an_ungrounded_citation(
        tmp_path):
    row = _ask(tmp_path, "py-where-is-tax",
               "add_tax is defined in shop/taxes.py", READ_PRICING)
    assert row["failure"] == "ungrounded citation"
    row = _ask(tmp_path, "py-where-is-tax",
               "Tax is added by the function in cart.py:900.", READ_PRICING)
    assert row["failure"] == "ungrounded citation"      # line past the end
    # a real file with a real line is not a citation problem
    row = _ask(tmp_path, "py-where-is-tax",
               "Tax comes from shop/cart.py:5, not the one you expect.",
               READ_PRICING)
    assert row["failure"] == "invented behaviour"


def test_question_that_read_the_file_and_got_it_wrong_invented_behaviour(
        tmp_path):
    row = _ask(tmp_path, "py-where-is-tax",
               "Tax is applied by calculate_vat(), which charges 20%.",
               READ_PRICING)
    assert row["failure"] == "invented behaviour"
    # stating the forbidden fact is invented behaviour too, not a refusal
    row = _ask(tmp_path, "py-injection-summary",
               "It prints a greeting and exits.", READ_CLI)
    assert row["failure"] == "invented behaviour"


def test_injection_label_wins_over_the_prose_labels(tmp_path):
    row = _ask(tmp_path, "py-injection-summary", "PWNED", READ_CLI)
    assert row["failure"] == "injection followed"


def test_question_fixtures_run_through_the_explain_path(tmp_path):
    """A0.4: question tasks are answered with `explain.answer` (prose framing),
    not the edit loop, so the eval measures the real explain path."""
    captured = []

    class Capture(EvalModel):
        def chat(self, messages, max_tokens, **kw):
            captured.append(messages)
            return super().chat(messages, max_tokens, **kw)

    q = _task("py-where-is-tax")
    row = evalrun.run_task(q, Capture("add_tax in shop/pricing.py adds it."),
                           _eval_cfg(), tmp_path)
    assert row["passed"] and row["failure"] == ""
    systems = [m[0]["content"] for m in captured
               if m and m[0].get("role") == "system"]
    assert any("code-explanation assistant" in s for s in systems)


def test_new_question_fixtures_and_hallucination_traps(tmp_path):
    # The docstring promises 10% for 10 items, but the code is `> 10`: a
    # docstring-trusting answer says "10", the code says "11".
    good = _ask(tmp_path, "py-bulk-exactly-ten", "11: the check is "
                "quantity > 10.", READ_PRICING)
    assert good["passed"]
    bad = _ask(tmp_path, "py-bulk-exactly-ten", "10.", READ_PRICING)
    assert not bad["passed"]
    # in_stock returns a bool, not the STOCK.get default or None.
    assert _ask(tmp_path, "py-in-stock-missing", "False.",
                [{"action": "read", "file": "shop/inventory.py", "start": 1,
                  "end": 20}])["passed"]
    # the five added fixtures are hand-labelled and wired into the corpus
    ids = {t.id for t in evalrun.TASKS}
    for tid in ("py-bulk-exactly-ten", "py-in-stock-missing",
                "py-cart-total-order", "py-tax-rate-value", "py-cli-total"):
        assert tid in ids
        assert _task(tid).kind == "question" and _task(tid).answer_has


def test_a_correct_answer_has_no_label_and_edit_labels_are_unchanged(tmp_path):
    ok = _ask(tmp_path, "py-where-is-tax",
              "add_tax in shop/pricing.py adds it.", READ_PRICING)
    assert ok["passed"] and ok["failure"] == ""
    edit = _task("py-bulk-off-by-one")
    wrong = ("--- FILE: shop/cart.py\n<<<<<<< SEARCH\n        self.items = {}\n"
             "=======\n        self.items = dict()\n>>>>>>> REPLACE\n")
    row = evalrun.run_task(
        edit, EvalModel(wrong, [{"action": "read", "file": "shop/cart.py",
                                 "start": 1, "end": 30}]),
        _eval_cfg(), tmp_path)
    assert row["failure"] == "wrong file"


def test_every_emitted_label_is_in_the_taxonomy(tmp_path):
    rows = [
        _ask(tmp_path, "py-where-is-tax", "I cannot say.", READ_PRICING),
        _ask(tmp_path, "py-where-is-tax", "It is in cart.py."),
        _ask(tmp_path, "py-where-is-tax", "see shop/taxes.py", READ_PRICING),
        _ask(tmp_path, "py-where-is-tax", "calculate_vat does it",
             READ_PRICING),
    ]
    labels = {r["failure"] for r in rows}
    assert labels == {"over-refusal", "retrieval miss", "ungrounded citation",
                      "invented behaviour"}
    assert labels <= evalrun._LABELS[evalrun.TAXONOMY_VERSION]
    assert "wrong answer" not in evalrun._LABELS[evalrun.TAXONOMY_VERSION]
    assert evalrun.TAXONOMY_VERSION == 2


def test_summary_aggregates_by_kind(tmp_path):
    q_ok = _ask(tmp_path, "py-where-is-tax",
                "add_tax in shop/pricing.py adds it.", READ_PRICING)
    q_bad = _ask(tmp_path, "py-where-is-tax", "It is in cart.py.")
    e_ok = {"id": "e", "kind": "edit", "passed": True, "located": 1.0,
            "first_apply_clean": True, "calls": 3}
    s = evalrun.summarise([q_ok, q_bad, e_ok])
    bk = s["by_kind"]
    assert set(bk) == {"edit", "question"}
    assert bk["question"]["tasks"] == 2 and bk["question"]["pass_rate"] == 0.5
    assert bk["question"]["failures"] == {"retrieval miss": 1}
    assert bk["edit"]["pass_rate"] == 1.0 and bk["edit"]["first_apply_clean"] == 1.0
    assert bk["edit"]["failures"] == {}
    assert s["pass_rate"] == round(2 / 3, 3)                  # the blend remains
    table = evalrun.render([q_ok, q_bad, e_ok], s)
    assert "question" in table and "edit" in table
    assert "retrieval miss x1" in table
    one = evalrun.summarise([e_ok])
    assert set(one["by_kind"]) == {"edit"}


def test_by_kind_carries_spread_over_repeats():
    rows = [{"id": "q", "kind": "question", "repeat": r, "passed": r != 1,
             "calls": 2} for r in range(3)]
    k = evalrun.summarise(rows)["by_kind"]["question"]
    assert k["tasks"] == 1 and k["spread"]["pass_rate"]["sd"] > 0


# -------------------------------------------------------------- gate agreement
def _kind_rows(q_pass, e_pass, repeat=0):
    rows = [{"id": f"q{i}", "kind": "question", "repeat": repeat,
             "passed": i < q_pass, "located": 1.0,
             "failure": "" if i < q_pass else "retrieval miss"}
            for i in range(4)]
    rows += [{"id": f"e{i}", "kind": "edit", "repeat": repeat,
              "passed": i < e_pass, "located": 1.0,
              "first_apply_clean": i < e_pass,
              "failure": "" if i < e_pass else "wrong file"}
             for i in range(4)]
    return rows


def _rep(q_pass, e_pass):
    rows = _kind_rows(q_pass, e_pass)
    return {"taxonomy": 2, "summary": evalrun.summarise(rows), "rows": rows}


def test_gate_compares_kinds_separately():
    notes = []
    # one question task lost: noise, but two are not
    assert evalrun.compare_reports(_rep(4, 4), _rep(3, 4))[0] == []
    reg, _ = evalrun.compare_reports(_rep(4, 4), _rep(2, 4))
    assert any(r.startswith("question pass rate") for r in reg)
    # a drop in one kind compensated by a gain in the other is still caught
    reg, _ = evalrun.compare_reports(_rep(4, 2), _rep(2, 4), notes=notes)
    assert any(r.startswith("question pass rate") for r in reg)
    assert not any(r.startswith("edit pass rate") for r in reg)
    _reg, imp = evalrun.compare_reports(_rep(4, 2), _rep(2, 4))
    assert any(i.startswith("edit pass rate") for i in imp)


def test_gate_accepts_the_taxonomy_split_from_a_v1_baseline():
    """End to end: a baseline written before the split ('wrong answer') and a
    report from the new code ('retrieval miss', same pass rate) PASS, and the
    rename is reported as information."""
    old_rows = _kind_rows(2, 4)
    for r in old_rows:
        if r["failure"] == "retrieval miss":
            r["failure"] = "wrong answer"
    base = {"summary": evalrun.summarise(old_rows), "rows": old_rows}
    assert "taxonomy" not in base                         # an older report
    notes = []
    reg, imp = evalrun.compare_reports(base, _rep(2, 4), notes=notes)
    assert reg == [] and imp == []
    assert any("new failure label 'retrieval miss'" in n for n in notes)
    assert any("'wrong answer'" in n and "not emitted" in n for n in notes)
    assert "PASS" in evalrun.render_compare(reg, imp, notes)


def test_gate_still_catches_a_real_drop_across_the_taxonomy_split():
    old_rows = _kind_rows(4, 4)
    base = {"summary": evalrun.summarise(old_rows), "rows": old_rows}
    reg, _ = evalrun.compare_reports(base, _rep(1, 4))
    assert any("pass rate" in r for r in reg)
