"""Hunk-level diff review sign-off (v0.3.2 M1.2 / WP-3203).

A file's change is segmented into hunks the reviewer can accept
independently: `build_diff` tags every row, `reconstruct_from_hunks`
rebuilds the after-text from the accepted groups, and
`apply_hunk_selection` refuses (without mutating anything) a partial
acceptance that would leave the file syntactically broken. Covers the
standalone review page's /apply payload, the GUI core's decide_review /
review_file / review_rediff contracts, and 100% backwards compatibility
with file-level sign-off.
"""
from __future__ import annotations

import http.client
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import review  # noqa: E402
from pedacito.gui import core  # noqa: E402


# Two changes 30 equal lines apart (> 2*CONTEXT): two hunks. The full after
# text is valid Python (x = [1, 2]), but accepting only one hunk leaves an
# unbalanced bracket -- invalid Python.
PAD30 = "".join(f"# pad {i}\n" for i in range(30))
BRACKET_BEFORE = "x = 1\n" + PAD30 + "y = 2\n"
BRACKET_AFTER = "x = [1,\n" + PAD30 + "     2]\n"

# Two changes 10 equal lines apart: two hunks, each valid on its own.
PAD10 = "".join(f"# pad {i}\n" for i in range(10))
SAFE_BEFORE = "a = 1\n" + PAD10 + "b = 2\n"
SAFE_AFTER = "a = 10\n" + PAD10 + "b = 20\n"


def _fd(rel: str, before: str, after: str) -> review.FileDiff:
    rows, added, removed, hunks = review.build_diff(before, after)
    return review.FileDiff(rel, rows, added, removed, review._sha(before),
                           after, "edit", hunks=hunks, before_text=before)


# ------------------------------------------------------------- segmentation ---

def test_build_diff_segments_hunks_and_tags_rows():
    rows, added, removed, hunks = review.build_diff(BRACKET_BEFORE,
                                                    BRACKET_AFTER)
    assert (added, removed) == (2, 2)
    assert len(hunks) == 2
    assert [h.index for h in hunks] == [0, 1]
    # 1-based starts, context included, ranges disjoint.
    assert hunks[0].a_start == 1 and hunks[0].b_start == 1
    assert hunks[0].a_start + hunks[0].a_count <= hunks[1].a_start
    assert hunks[0].b_start + hunks[0].b_count <= hunks[1].b_start
    assert (hunks[0].added, hunks[0].removed) == (1, 1)
    assert (hunks[1].added, hunks[1].removed) == (1, 1)
    # Every row belongs to its hunk (or to no hunk: folded/skip context).
    for h in hunks:
        assert h.rows
        for ri in h.rows:
            assert rows[ri].hunk == h.index
    for row in rows:
        assert row.hunk in (-1, 0, 1)
        if row.kind == "skip":
            assert row.hunk == -1
    # The changed lines themselves are in the hunks, not in the gap.
    changed = [r for r in rows if r.kind == "replace"]
    assert [r.hunk for r in changed] == [0, 1]


def test_build_rows_stays_compatible_and_close_changes_merge():
    rows, added, removed = review.build_rows(BRACKET_BEFORE, BRACKET_AFTER)
    rows2, added2, removed2, _hunks = review.build_diff(BRACKET_BEFORE,
                                                        BRACKET_AFTER)
    assert (rows, added, removed) == (rows2, added2, removed2)
    # Adjacent changes (no wide equal gap) stay ONE hunk.
    _r, _a, _rm, hunks = review.build_diff("a = 1\nb = 2\n", "a = 9\nb = 8\n")
    assert len(hunks) == 1
    # No change: no hunks, only equal rows.
    rows3, a3, rm3, hunks3 = review.build_diff("same\n", "same\n")
    assert (a3, rm3, hunks3) == (0, 0, [])
    assert all(r.kind == "equal" for r in rows3)


def test_reconstruct_from_hunks_all_none_first_second():
    from pedacito import edits
    # Fixture sanity: the whole change is valid Python, each half is not.
    assert edits.syntax_error("m.py", BRACKET_AFTER) is None
    assert edits.syntax_error("m.py", BRACKET_BEFORE) is None
    ids = [0, 1]
    assert review.reconstruct_from_hunks(
        BRACKET_BEFORE, BRACKET_AFTER, ids) == BRACKET_AFTER
    assert review.reconstruct_from_hunks(
        BRACKET_BEFORE, BRACKET_AFTER, []) == BRACKET_BEFORE
    assert review.reconstruct_from_hunks(
        BRACKET_BEFORE, BRACKET_AFTER, [0]) == "x = [1,\n" + PAD30 + "y = 2\n"
    assert review.reconstruct_from_hunks(
        BRACKET_BEFORE, BRACKET_AFTER, [1]) == "x = 1\n" + PAD30 + "     2]\n"
    # A safe partial keeps the other change's before-lines.
    assert review.reconstruct_from_hunks(
        SAFE_BEFORE, SAFE_AFTER, [0]) == "a = 10\n" + PAD10 + "b = 2\n"
    assert review.reconstruct_from_hunks(
        SAFE_BEFORE, SAFE_AFTER, [1]) == "a = 1\n" + PAD10 + "b = 20\n"


# ------------------------------------------------------- apply_hunk_selection ---

def test_partial_acceptance_that_breaks_syntax_is_refused_without_mutation():
    fd = _fd("m.py", BRACKET_BEFORE, BRACKET_AFTER)
    ok, detail = review.apply_hunk_selection([fd], {"m.py": [0]})
    assert not ok
    assert "m.py" in detail and "invalid syntax" in detail
    assert fd.after_text == BRACKET_AFTER      # untouched: retryable
    assert fd.hunk_selection is None
    # The whole-file acceptance still works right after the refusal.
    ok, detail = review.apply_hunk_selection([fd], {"m.py": [0, 1]})
    assert ok and detail == ""
    assert fd.after_text == BRACKET_AFTER
    assert fd.hunk_selection == (0, 1)


def test_valid_partial_acceptance_reconstructs_the_after_text():
    fd = _fd("s.py", SAFE_BEFORE, SAFE_AFTER)
    ok, _ = review.apply_hunk_selection([fd], {"s.py": [0]})
    assert ok
    assert fd.after_text == "a = 10\n" + PAD10 + "b = 2\n"
    assert fd.hunk_selection == (0,)


def test_selection_map_semantics_absent_empty_and_unknown():
    fd = _fd("s.py", SAFE_BEFORE, SAFE_AFTER)
    # Absent entry = whole file, untouched.
    ok, _ = review.apply_hunk_selection([fd], {})
    assert ok and fd.after_text == SAFE_AFTER and fd.hunk_selection is None
    # Every file fully deselected = nothing to accept.
    ok, detail = review.apply_hunk_selection([fd], {"s.py": []})
    assert not ok and detail == "Select at least one file or hunk to accept."
    assert fd.after_text == SAFE_AFTER
    # Out-of-range indices are refused.
    ok, detail = review.apply_hunk_selection([fd], {"s.py": [7]})
    assert not ok and "unknown hunk index" in detail
    # A legacy diff without hunks is kept whole when selected.
    legacy = review.FileDiff("old.py", [], 0, 0, "", "z = 1\n", "legacy")
    ok, _ = review.apply_hunk_selection([legacy], {"old.py": [0]})
    assert ok and legacy.after_text == "z = 1\n"


def test_write_files_refuses_a_hand_set_broken_hunk_selection(tmp_path):
    # Defense in depth: even a caller that bypassed apply_hunk_selection
    # cannot write a partial acceptance that no longer parses.
    (tmp_path / "m.py").write_text(BRACKET_BEFORE, encoding="utf-8")
    fd = _fd("m.py", BRACKET_BEFORE, BRACKET_AFTER)
    fd.after_text = "x = [1,\n" + PAD30 + "y = 2\n"
    fd.hunk_selection = (0,)
    ok, detail = review.write_files(tmp_path, [fd], None)
    assert not ok and "invalid syntax" in detail
    assert (tmp_path / "m.py").read_text(encoding="utf-8") == BRACKET_BEFORE


# --------------------------------------------------------------- build_page ---

def test_build_page_renders_hunk_and_file_checkboxes_only_when_interactive():
    fd = _fd("m.py", BRACKET_BEFORE, BRACKET_AFTER)
    page = review.build_page([fd], "task", "model", "profile", "nonce",
                             interactive=True, stamp="s")
    assert 'class="file-box"' in page
    assert page.count('class="hunk-box"') == 2
    assert 'data-hunk="0"' in page and 'data-hunk="1"' in page
    assert "Hunk 1" in page and "Hunk 2" in page
    assert "accepted_hunks" in page            # the Apply POST carries them
    assert "indeterminate" in page             # partial file-box state
    readonly = review.build_page([fd], "task", "model", "profile", "nonce",
                                 interactive=False, stamp="s")
    assert '<input type="checkbox" class="hunk-box"' not in readonly
    assert '<input type="checkbox" class="file-box"' not in readonly


def test_build_page_deleted_file_has_no_hunk_checkboxes():
    fd = _fd("gone.py", "x = 1\n", "")
    fd.deleted = True
    page = review.build_page([fd], "t", "m", "p", "n", interactive=True,
                             stamp="s")
    assert '<input type="checkbox" class="hunk-box"' not in page
    assert 'class="file-box"' in page          # whole-file decision only


# ------------------------------------------------- standalone /apply endpoint ---

def _serve(page, nonce, writer, diffs):
    result = {}
    urls = []
    ready = threading.Event()

    def on_url(u):
        urls.append(u)
        ready.set()

    t = threading.Thread(target=lambda: result.update(
        out=review.run(page, nonce, writer, open_browser=False, timeout=10,
                       on_url=on_url, diffs=diffs)))
    t.start()
    assert ready.wait(5)
    port = int(urls[0].rsplit(":", 1)[1].rstrip("/"))
    return t, result, port


def _post(port, path, body):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("POST", path, body=json.dumps(body),
                 headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    data = r.read()
    conn.close()
    return r.status, json.loads(data)


def test_standalone_apply_refuses_broken_partial_then_accepts_retry(tmp_path):
    (tmp_path / "m.py").write_text(BRACKET_BEFORE, encoding="utf-8")
    diffs = [_fd("m.py", BRACKET_BEFORE, BRACKET_AFTER)]
    page = review.build_page(diffs, "t", "m", "p", "tok", interactive=True,
                             stamp="s")

    def writer():
        return review.write_files(tmp_path, diffs, None)

    t, result, port = _serve(page, "tok", writer, diffs)
    decided = False
    try:
        status, j = _post(port, "/apply", {
            "nonce": "tok", "accepted": ["m.py"],
            "accepted_hunks": {"m.py": [0]}})
        assert status == 200 and not j["ok"]
        assert "invalid syntax" in j["detail"]
        assert (tmp_path / "m.py").read_text(encoding="utf-8") == BRACKET_BEFORE
        assert t.is_alive()                    # refused != decided: retryable
        status, j = _post(port, "/apply", {
            "nonce": "tok", "accepted": ["m.py"],
            "accepted_hunks": {"m.py": [0, 1]}})
        decided = True
        assert status == 200 and j["ok"]
    finally:
        if not decided:
            _post(port, "/cancel", {"nonce": "tok"})
        t.join(10)
    assert result["out"][0] == "applied"
    assert (tmp_path / "m.py").read_text(encoding="utf-8") == BRACKET_AFTER


def test_standalone_apply_file_filter_and_legacy_nonce_only(tmp_path):
    (tmp_path / "a.py").write_text(SAFE_BEFORE, encoding="utf-8")
    (tmp_path / "b.py").write_text(SAFE_BEFORE, encoding="utf-8")
    diffs = [_fd("a.py", SAFE_BEFORE, SAFE_AFTER),
             _fd("b.py", SAFE_BEFORE, SAFE_AFTER)]
    page = review.build_page(diffs, "t", "m", "p", "tok", interactive=True,
                             stamp="s")

    def writer():
        return review.write_files(tmp_path, diffs, None)

    t, result, port = _serve(page, "tok", writer, diffs)
    try:
        status, j = _post(port, "/apply", {"nonce": "tok",
                                           "accepted": ["a.py"]})
        assert status == 200 and j["ok"]
    finally:
        t.join(10)
    assert result["out"][0] == "applied"
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == SAFE_AFTER
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == SAFE_BEFORE

    # Legacy payload (nonce only, no selection keys): the writer runs as-is.
    (tmp_path / "a.py").write_text(SAFE_BEFORE, encoding="utf-8")
    diffs2 = [_fd("a.py", SAFE_BEFORE, SAFE_AFTER)]
    page2 = review.build_page(diffs2, "t", "m", "p", "tok", interactive=True,
                              stamp="s")
    t, result, port = _serve(page2, "tok",
                             lambda: review.write_files(tmp_path, diffs2, None),
                             diffs2)
    try:
        status, j = _post(port, "/apply", {"nonce": "tok"})
        assert status == 200 and j["ok"]
    finally:
        t.join(10)
    assert result["out"][0] == "applied"
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == SAFE_AFTER


def test_standalone_writer_with_required_param_gets_the_accepted_map(tmp_path):
    diffs = [_fd("a.py", SAFE_BEFORE, SAFE_AFTER)]
    page = review.build_page(diffs, "t", "m", "p", "tok", interactive=True,
                             stamp="s")
    seen = {}

    def writer(accepted):
        seen["map"] = accepted
        return True, "ok"

    t, result, port = _serve(page, "tok", writer, diffs)
    try:
        status, j = _post(port, "/apply", {
            "nonce": "tok", "accepted": ["a.py"],
            "accepted_hunks": {"a.py": [1]}})
        assert status == 200 and j["ok"]
    finally:
        t.join(10)
    assert result["out"] == ("applied", "ok")
    assert seen["map"] == {"a.py": "a = 1\n" + PAD10 + "b = 20\n"}


# ------------------------------------------------------------- GUI core ---

def _state_with(diffs):
    st = core.State()                          # no root: fully in-memory
    writes = []

    def writer():
        writes.append([d.rel for d in diffs])
        return True, "ok"

    rv = core.Review(1, "task", diffs, writer, partial_ok=True)
    st.review = rv
    return st, rv, writes


def test_gui_decide_review_hunk_refusal_keeps_review_retryable():
    m = _fd("m.py", BRACKET_BEFORE, BRACKET_AFTER)
    s = _fd("s.py", SAFE_BEFORE, SAFE_AFTER)
    st, rv, writes = _state_with([m, s])
    ok, detail = st.decide_review(1, "apply", ["m.py"],
                                  accepted_hunks={"m.py": [0]})
    assert not ok and "invalid syntax" in detail
    assert [d.rel for d in rv.diffs] == ["m.py", "s.py"]
    assert m.after_text == BRACKET_AFTER and m.hunk_selection is None
    assert rv.outcome == "" and writes == []
    # Retry with both hunks: applied, the unselected file dropped.
    ok, detail = st.decide_review(1, "apply", ["m.py"],
                                  accepted_hunks={"m.py": [0, 1]})
    assert ok and detail == "ok"
    assert [d.rel for d in rv.diffs] == ["m.py"]
    assert m.after_text == BRACKET_AFTER
    assert rv.outcome == "applied" and writes == [["m.py"]]


def test_gui_decide_review_valid_partial_and_empty_selection_errors():
    s = _fd("s.py", SAFE_BEFORE, SAFE_AFTER)
    st, rv, writes = _state_with([s])
    ok, _ = st.decide_review(1, "apply", ["s.py"],
                             accepted_hunks={"s.py": [0]})
    assert ok
    assert s.after_text == "a = 10\n" + PAD10 + "b = 2\n"
    assert s.hunk_selection == (0,)
    assert rv.outcome == "applied"

    # Legacy file-only empty selection keeps its exact historical error.
    s2 = _fd("s.py", SAFE_BEFORE, SAFE_AFTER)
    st2, rv2, _ = _state_with([s2])
    ok, detail = st2.decide_review(1, "apply", [])
    assert not ok and detail == "Select at least one file to accept."
    assert rv2.outcome == ""
    # Hunk-aware empty selection gets the hunk variant.
    ok, detail = st2.decide_review(1, "apply", [],
                                   accepted_hunks={"s.py": []})
    assert not ok and detail == "Select at least one file or hunk to accept."
    assert rv2.outcome == ""


def test_gui_review_payloads_carry_hunks_and_rediff_resegments():
    m = _fd("m.py", BRACKET_BEFORE, BRACKET_AFTER)
    st, rv, _ = _state_with([m])
    meta = st.review_meta()
    assert meta["files"][0]["hunks"][0]["index"] == 0
    assert len(meta["files"][0]["hunks"]) == 2
    brief = st.review_meta_brief()
    assert len(brief["files"][0]["hunks"]) == 2
    payload = st.review_file(1, "m.py")
    assert len(payload["hunks"]) == 2
    assert all(len(r) == 9 for r in payload["rows"] if r[0] != "skip")
    assert payload["rows"][0][8] == 0          # row -> hunk index
    # A hand edit re-segments and drops any stale hunk selection.
    m.hunk_selection = (0,)
    edited = "x = 5\n" + PAD30 + "y = 2\n"
    out = st.review_rediff(1, "m.py", edited)
    assert out["after"] == edited
    assert len(out["hunks"]) == 1
    assert m.hunk_selection is None
    assert rv.edits["m.py"] == edited
    # decide_review applies the hand edit over the staged after-text.
    ok, _ = st.decide_review(1, "apply", ["m.py"])
    assert ok and m.after_text == edited
