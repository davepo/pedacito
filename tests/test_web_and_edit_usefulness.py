"""D1-D4: file-deletion edits, removal-clause verification, stale-plan
precedence, and the webcheck form-control notices.

Each gap is pinned at the level it lives at: parse/apply for the DELETE
syntax, the requirement machinery for fuzzy removals and stale checklists,
and regex-level webcheck fixtures for honeypots and unlabelled fields.
"""
from __future__ import annotations

from pathlib import Path

from pedacito import (checkpoint, edits, memory, prompts, review, verify,
                      webcheck, workspace)
from pedacito import index as index_mod
from pedacito.agent import EDIT_FORMAT, Agent
from pedacito.config import Config


def _w(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _hit(found, rel=None, needle=None, line=None) -> bool:
    return any((rel is None or f.rel == rel)
               and (needle is None or needle in f.message)
               and (line is None or f.line == line)
               for f in found)


TAGLINE = '<p class="tagline">Consultant, builder, and lifelong tinkerer.</p>'
REMOVE_TASK = ('Remove the "Consultant, builder, lifelong tinker" tagline '
               "from index.html. Do not change anything else.")


# ------------------------------------------------------- D1: deleting a file
def test_delete_block_and_delete_file_directive_parse():
    reply = ("--- FILE: old.html\n<<<<<<< DELETE\n>>>>>>> DELETE\n"
             "DELETE_FILE: gone.html\n")
    parsed = edits.parse_edits(reply)
    assert [(e.file, e.delete) for e in parsed] == [
        ("old.html", True), ("gone.html", True)]
    # The body between the markers is ignored, and a stray anchor suffix on
    # the header never becomes part of the path.
    parsed = edits.parse_edits(
        "--- FILE: old.html LINES 1-9\n<<<<<<< DELETE\nquote anything\n"
        ">>>>>>> DELETE\n")
    assert [(e.file, e.delete) for e in parsed] == [("old.html", True)]


def test_delete_quoted_inside_another_block_is_documentation():
    # A doc block SHOWING the delete syntax is text, not a deletion: the
    # DELETE_FILE line inside the REPLACE body belongs to the outer block.
    reply = ("--- FILE: docs.md\n<<<<<<< SEARCH\nold doc\n=======\n"
             "Use the one-line form:\nDELETE_FILE: path/to/file.html\n"
             ">>>>>>> REPLACE\n")
    parsed = edits.parse_edits(reply)
    assert [(e.file, e.delete) for e in parsed] == [("docs.md", False)]
    # Same for a full DELETE example quoted in a symbol-splice body.
    reply = ("<<<<<<< SYMBOL docs.md :: example\n"
             "--- FILE: path\n<<<<<<< DELETE\n>>>>>>> DELETE\n"
             ">>>>>>> END_SYMBOL\n")
    parsed = edits.parse_edits(reply)
    assert [(e.file, e.delete) for e in parsed] == [("docs.md", False)]


def test_delete_applies_backs_up_and_restore_brings_it_back(tmp_path):
    _w(tmp_path, "old.html", "<p>bye</p>\n")
    _w(tmp_path, "gone.html", "<p>gone</p>\n")
    _w(tmp_path, "keep.html", "<p>stay</p>\n")
    reply = ("--- FILE: old.html\n<<<<<<< DELETE\n>>>>>>> DELETE\n"
             "DELETE_FILE: gone.html\n")
    snap = workspace.new_backup_dir(tmp_path)
    res = edits.apply_edits(tmp_path, edits.parse_edits(reply),
                            backup=True, backup_dir=snap)
    assert len(res) == 2 and all(r.ok and r.deleted for r in res)
    assert not (tmp_path / "old.html").exists()
    assert not (tmp_path / "gone.html").exists()
    assert (tmp_path / "keep.html").exists()
    # The snapshot holds the exact pre-delete bytes; restore brings them back.
    assert (snap / "old.html").read_text(encoding="utf-8") == "<p>bye</p>\n"
    assert sorted(workspace.restore(tmp_path, snap)) == ["gone.html", "old.html"]
    assert (tmp_path / "old.html").read_text(encoding="utf-8") == "<p>bye</p>\n"
    assert (tmp_path / "gone.html").read_text(encoding="utf-8") == "<p>gone</p>\n"


def test_delete_result_shape_feeds_the_review_gates(tmp_path):
    _w(tmp_path, "old.html", "<p>bye</p>\n")
    res = edits.apply_edits(
        tmp_path, edits.parse_edits("DELETE_FILE: old.html\n"))
    r = res[0]
    assert r.ok and r.deleted and not r.whole_file
    assert r.message.startswith("deleted")
    assert r.before == "<p>bye</p>\n" and r.after == "" and r.sha
    assert "-<p>bye</p>" in r.diff


def test_delete_refuses_sensitive_traversal_protected_and_directories(tmp_path):
    _w(tmp_path, ".env", "SECRET=1\n")
    _w(tmp_path, "keep.html", "<p>stay</p>\n")
    (tmp_path / "sub").mkdir()
    reply = ("DELETE_FILE: .env\n"
             "DELETE_FILE: ../escape.html\n"
             "DELETE_FILE: .pedacito/state.json\n"
             "DELETE_FILE: sub\n")
    res = edits.apply_edits(tmp_path, edits.parse_edits(reply))
    assert len(res) == 4 and not any(r.ok for r in res)
    by_file = {r.file: r.message for r in res}
    assert "secrets/credentials" in by_file[".env"]
    assert "traversal" in by_file["../escape.html"]
    assert "VCS/Pedacito state" in by_file[".pedacito/state.json"]
    assert "directory" in by_file["sub"]
    assert (tmp_path / ".env").exists() and (tmp_path / "sub").is_dir()


def test_delete_of_an_absent_file_is_a_noop_success(tmp_path):
    res = edits.apply_edits(
        tmp_path, edits.parse_edits("DELETE_FILE: missing.html\n"))
    # Nothing was deleted (no backup, no unlink): a successful no-op.
    assert res[0].ok and not res[0].deleted
    assert "already absent" in res[0].message


def test_delete_dry_run_leaves_the_file(tmp_path):
    _w(tmp_path, "old.html", "<p>bye</p>\n")
    res = edits.apply_edits(
        tmp_path, edits.parse_edits("DELETE_FILE: old.html\n"), dry_run=True)
    assert res[0].ok and res[0].deleted and "dry run" in res[0].message
    assert (tmp_path / "old.html").exists()


def test_delete_without_a_snapshot_writes_a_bak_beside_the_source(tmp_path):
    _w(tmp_path, "old.html", "<p>bye</p>\n")
    res = edits.apply_edits(
        tmp_path, edits.parse_edits("DELETE_FILE: old.html\n"), backup=True)
    assert res[0].ok and not (tmp_path / "old.html").exists()
    assert (tmp_path / "old.html.bak").read_text(encoding="utf-8") == "<p>bye</p>\n"


def test_empty_replace_refusal_points_at_the_delete_syntax(tmp_path):
    _w(tmp_path, "old.html", "<p>bye</p>\n")
    reply = ("--- FILE: old.html\n<<<<<<< SEARCH\n<p>bye</p>\n=======\n"
             ">>>>>>> REPLACE\n")
    res = edits.apply_edits(tmp_path, edits.parse_edits(reply))
    assert not res[0].ok and (tmp_path / "old.html").exists()
    assert "DELETE_FILE: old.html" in res[0].message


def test_review_signoff_deletes_with_backup_and_shows_the_deletion(tmp_path):
    _w(tmp_path, "old.html", "<p>bye</p>\n")
    preview = edits.apply_edits(
        tmp_path, edits.parse_edits("DELETE_FILE: old.html\n"), dry_run=True)
    diffs = review.diffs_from_results(preview)
    assert len(diffs) == 1 and diffs[0].deleted and diffs[0].removed > 0
    snap = workspace.new_backup_dir(tmp_path)
    ok, msg = review.write_files(tmp_path, diffs, snap)
    assert ok and "deleted: old.html" in msg
    assert not (tmp_path / "old.html").exists()
    assert (snap / "old.html").read_text(encoding="utf-8") == "<p>bye</p>\n"
    # Reopening the review from the snapshot shows the deletion, not a skip.
    again = review.diffs_from_backup(tmp_path, snap)
    assert [(d.rel, d.deleted) for d in again] == [("old.html", True)]
    # A second sign-off on the already-gone file is not stale.
    ok, _msg = review.write_files(tmp_path, diffs, snap)
    assert ok


# ------------------------- D2: coalescing prompt + fuzzy removal verification
def test_prompts_teach_section_coalescing_and_delete():
    for text in (EDIT_FORMAT, prompts._TINY_EDIT):
        assert "ONE HTML section" in text or "ONE section" in text
        assert "enclosing `<section>`" in text
        assert "DELETE_FILE:" in text
        assert "<<<<<<< DELETE" in text


def test_fuzzy_removal_matches_the_inflected_tagline():
    # The clause quotes from memory; the page carries an inflected form.
    assert verify._literal_fuzzy_in(
        TAGLINE, "Consultant, builder, lifelong tinker")
    assert not verify._literal_in(
        TAGLINE, "Consultant, builder, lifelong tinker")
    # Unrelated copy, and a single common word, never fuzzy-match.
    assert not verify._literal_fuzzy_in(
        "We sell android devices.", "lifelong tinker")
    assert not verify._literal_fuzzy_in("a tinkerer at heart", "tinker")
    # Token bounds: exact or a bounded prefix, never a short-word accident.
    assert verify._copy_token_match("tinker", "tinkerer")
    assert not verify._copy_token_match("and", "android")
    assert not verify._copy_token_match("cat", "catch")


def test_removal_still_present_in_after_text_is_not_addressed():
    still = edits.Result(True, "index.html", "replaced 1 block(s)",
                         after=TAGLINE + "\n")
    found = verify.check_coverage(REMOVE_TASK, "Removed it.", [still])
    assert any("still present" in f.message
               and "Consultant, builder, lifelong tinker" in f.message
               for f in found)
    gone = edits.Result(True, "index.html", "replaced 1 block(s)",
                        after="<p>Welcome.</p>\n")
    assert verify.check_coverage(REMOVE_TASK, "Removed it.", [gone]) == []


def test_insertion_literals_stay_exact():
    # The inflected tagline fuzzy-matches the quoted copy, but an INSERTION
    # is only satisfied by the exact string: fuzzy tolerance is removal-only.
    task = ('Add the tagline "Consultant, builder, lifelong tinker" to '
            "index.html. Do not change anything else.")
    near = edits.Result(True, "index.html", "replaced 1 block(s)",
                        after=TAGLINE + "\n")
    found = verify.check_coverage(task, "Added it.", [near])
    assert any("missing" in f.message
               and "Consultant, builder, lifelong tinker" in f.message
               for f in found)


def test_unmet_requirements_names_the_exact_disk_line(tmp_path):
    _w(tmp_path, "index.html", f"<h1>Jane</h1>\n{TAGLINE}\n")
    # No edit touched the file: the removal is still owed, and the disk check
    # names the exact line instead of re-guessing.
    unmet = verify.unmet_requirements(tmp_path, REMOVE_TASK, [])
    assert len(unmet) == 1
    req = unmet[0]
    assert req.missing_literals == ["Consultant, builder, lifelong tinker"]
    assert req.literal_hits == [("index.html", 2, TAGLINE)]
    # Once the copy is really gone from disk, nothing is owed.
    _w(tmp_path, "index.html", "<h1>Jane</h1>\n<p>Welcome.</p>\n")
    assert verify.unmet_requirements(tmp_path, REMOVE_TASK, []) == []


def test_merge_unmet_annotates_the_seed_with_the_disk_line(tmp_path):
    _w(tmp_path, "index.html", f"{TAGLINE}\n")
    tracker = verify.RequirementTracker(REMOVE_TASK)
    seeds = verify.coverage_seeds(REMOVE_TASK, "Removed it.", [],
                                  tracker=tracker)
    unmet = verify.unmet_requirements(tmp_path, REMOVE_TASK, [],
                                      tracker=tracker)
    merged = verify.merge_unmet(seeds, unmet)
    assert len(merged) == 1
    assert "still on disk at index.html:1: " + TAGLINE in merged[0].text
    assert merged[0].literal_hits == [("index.html", 1, TAGLINE)]
    # The run's authoritative tracker is never annotated by the merge.
    assert all("still on disk" not in r.text for r in tracker.items)


def test_coverage_followup_carries_the_disk_location(tmp_path, monkeypatch):
    _w(tmp_path, "index.html", f"{TAGLINE}\n")
    cfg = Config()
    cfg.log_sessions = False
    cfg.log_events = False
    idx = index_mod.build([str(tmp_path)], cfg, root=tmp_path,
                          summarise=False, verbose=False)
    agent = Agent(idx, None, cfg)
    agent.requirements = verify.RequirementTracker(REMOVE_TASK)
    # The follow-up hunk itself is not under test here: only that the merged
    # disk evidence reaches the coverage_followup receipt and the final error.
    monkeypatch.setattr(agent, "apply_edit_intents",
                        lambda *a, **k: ([], "", []))
    emitted: list = []
    orig = agent._emit

    def spy(kind, **kw):
        emitted.append((kind, kw))
        return orig(kind, **kw)

    monkeypatch.setattr(agent, "_emit", spy)
    findings: list = []
    _res, _reply, findings = agent._coverage_pass(
        REMOVE_TASK, "", [], "Removed it.", findings, verbose=False)
    followups = [kw for kind, kw in emitted if kind == "coverage_followup"]
    assert followups, "the still-present removal must seed a follow-up"
    assert any("still on disk at index.html:1" in c
               for kw in followups for c in kw["clauses"])
    assert any(f.severity == "error" and f.file == "index.html"
               and "still present" in f.message
               and "still on disk at index.html:1" in f.message
               for f in findings)


# ------------------------------ D3: stale plans never bind the current task
def test_completed_checklist_items_are_not_requirements():
    task = ("- [x] Add the tagline to index.html\n"
            "- [x] Wire the contact form\n"
            "1. [x] Ship the old footer\n"
            "- [ ] Polish the footer text\n")
    reqs = verify._requirements_from_task(task)
    assert [r.text for r in reqs] == ["Polish the footer text"]
    assert [t for t, _terms in verify.requirement_ledger(task)] == [
        "Polish the footer text"]
    # An all-completed checklist mines NOTHING: a standalone Ask run whose
    # injected context is finished history has no mandatory requirements.
    done = "[x] Add the tagline to index.html\n[x] Wire the contact form"
    assert verify._requirements_from_task(done) == []
    assert verify.requirement_ledger(done) == []


def test_task_marker_splits_the_stale_handoff_off():
    stale = ('CONSOLIDATED SESSION HANDOFF (earlier turns compacted):\n'
             '[x] Add the tagline "Consultant, builder, and lifelong '
             'tinkerer." to index.html\n'
             '[x] Wire the contact form\n')
    current = "Remove the tagline from index.html. Do not change anything else."
    for marker in ("NEW TASK", "CURRENT TASK", "TASK"):
        blob = stale + "\n" + marker + "\n" + current
        reqs = verify._requirements_from_task(blob)
        assert [r.text for r in reqs] == [
            "Remove the tagline from index.html", "Do not change anything else"]
        seeds = verify.coverage_seeds(blob, "Done.", [])
        assert all("Wire the contact form" not in r.text for r in seeds)
        assert all("[x]" not in r.text for r in seeds)


def test_handoff_and_checkpoint_carry_the_precedence_rule():
    rule = memory.TASK_PRECEDENCE_RULE
    assert "supersedes" in rule and "unconditionally" in rule
    handoff = memory.handoff_prompt("added the tagline last session")
    assert handoff.startswith("CONSOLIDATED SESSION HANDOFF")
    assert handoff.endswith(rule)
    # The rule survives even a checkpoint truncated to almost nothing.
    cp = {"task": "add the tagline", "reply": "done " * 2000}
    rendered = checkpoint.render(cp, max_chars=300)
    assert rendered.endswith(rule)
    assert checkpoint.render(None) == ""


# ----------------------------- D4: honeypots and unlabelled form controls
def test_unhidden_gotcha_honeypot_is_flagged(tmp_path):
    _w(tmp_path, "index.html",
       '<form action="/send" method="post">\n'
       '<label for="email">Email</label>\n'
       '<input type="email" id="email" name="email">\n'
       '<input type="text" name="_gotcha" aria-hidden="true">\n'
       "</form>\n")
    found = webcheck.run(tmp_path)
    assert _hit(found, "index.html", "honeypot", line=4)
    assert _hit(found, "index.html", 'name="_gotcha"', line=4)
    # aria-hidden is not hiding: the visible trap also has no label.
    assert _hit(found, "index.html", "no label", line=4)
    # The labelled email field stays quiet.
    assert not _hit(found, needle='name="email"')


def test_honeypot_class_and_aria_hidden_signals_are_flagged(tmp_path):
    _w(tmp_path, "index.html",
       '<input type="text" name="website" class="honeypot">\n'
       '<input type="text" name="fax" aria-hidden="true">\n')
    found = webcheck.run(tmp_path)
    assert _hit(found, "index.html", 'class="honeypot"', line=1)
    assert _hit(found, "index.html", 'aria-hidden="true"', line=2)


def test_hidden_honeypot_variants_are_quiet(tmp_path):
    _w(tmp_path, "style.css", ".honeypot { display: none; }\n")
    _w(tmp_path, "index.html",
       '<link rel="stylesheet" href="style.css">\n'
       "<form>\n"
       '<input type="text" name="_gotcha" hidden>\n'
       '<input type="text" name="g2" class="honeypot" style="display:none">\n'
       '<input type="text" name="g3" class="honeypot" '
       'style="visibility:hidden">\n'
       '<input type="text" name="g4" class="honeypot">\n'   # hidden by the CSS
       '<input type="hidden" name="g5">\n'
       "</form>\n")
    assert webcheck.run(tmp_path) == []


def test_inline_style_block_hides_the_honeypot_class(tmp_path):
    _w(tmp_path, "index.html",
       "<style>.honeypot { display: none; }</style>\n"
       '<form><input type="text" name="_gotcha" class="honeypot"></form>\n')
    assert webcheck.run(tmp_path) == []


def test_unlabelled_visible_form_fields_are_flagged(tmp_path):
    _w(tmp_path, "index.html",
       "<form>\n"
       '<input type="text" name="a">\n'
       '<input type="email" name="b">\n'
       '<textarea name="c"></textarea>\n'
       "</form>\n"
       '<input type="text" name="outside">\n')
    found = webcheck.run(tmp_path)
    assert _hit(found, "index.html", "no label", line=2)
    assert _hit(found, "index.html", "no label", line=3)
    assert _hit(found, "index.html", "no label", line=4)
    # Outside a <form> the label check does not apply.
    assert len(found) == 3


def test_labelled_form_fields_are_quiet(tmp_path):
    _w(tmp_path, "index.html",
       "<form>\n"
       '<label for="a">A</label><input type="text" id="a" name="a">\n'
       '<label>B <input type="email" name="b"></label>\n'
       '<input type="text" name="c" aria-label="C">\n'
       '<input type="text" name="d" aria-labelledby="dl">\n'
       "</form>\n")
    assert webcheck.run(tmp_path) == []


def test_exempt_and_non_text_input_types_are_not_label_checked(tmp_path):
    _w(tmp_path, "index.html",
       "<form>\n"
       '<input type="submit" value="Send">\n'
       '<input type="button" value="Click">\n'
       '<input type="reset">\n'
       '<input type="hidden" name="tok">\n'
       '<input type="checkbox" name="agree">\n'
       '<input type="password" name="pw">\n'
       "</form>\n")
    assert webcheck.run(tmp_path) == []
