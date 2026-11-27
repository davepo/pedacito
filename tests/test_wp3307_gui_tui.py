"""WP-3307 regression suite: GUI Term reload null-deref, partial hunk-review
provenance, thread chat-card persistence & legacy fallback, narrow-viewport
responsive CSS, deterministic `/compact` handoffs, silent HTTP disconnect
handling, Settings Profiles ellipsis/score chips, and `domcheck --baseline`
confirmation output.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import checkpoints, cli, domcheck, memory, review  # noqa: E402
from pedacito import config as config_mod  # noqa: E402
from pedacito.gui import core, server, settings_api  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parents[1] / "pedacito" / "gui" / "static"
APP_JS = STATIC_DIR / "app.js"
SETTINGS_JS = STATIC_DIR / "settings.js"
APP_CSS = STATIC_DIR / "app.css"

# Capture the real Playwright browsers path before conftest redirects
# LOCALAPPDATA to a per-test temporary folder on Windows.
if os.name == "nt" and "PLAYWRIGHT_BROWSERS_PATH" not in os.environ:
    _real_pw = Path.home() / "AppData" / "Local" / "ms-playwright"
    if _real_pw.is_dir():
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(_real_pw)
_HAS_PW = domcheck.available()[0]
requires_pw = pytest.mark.skipif(
    not _HAS_PW, reason="Playwright + Chromium not installed")

PAD10 = "".join(f"# pad {i}\n" for i in range(10))
TWO_HUNK_BEFORE = (
    "def apply_discount(price, pct):\n"
    "    return price * (1 - pct)\n"
    + PAD10
    + "def compute_shipping(weight):\n"
    "    return weight * 2\n"
)
TWO_HUNK_AFTER = (
    "def apply_discount(price, pct):\n"
    "    pct = max(0, min(pct, 1))\n"
    "    return price * (1 - pct)\n"
    + PAD10
    + "def compute_shipping(weight, express=False):\n"
    "    base = weight * 2\n"
    "    return base + 5 if express else base\n"
)


def _two_hunk_fd(rel: str = "pricing.py",
                 clauses: tuple[str, ...] = ("R1", "R2")) -> review.FileDiff:
    rows, added, removed, hunks = review.build_diff(
        TWO_HUNK_BEFORE, TWO_HUNK_AFTER)
    return review.FileDiff(rel, rows, added, removed,
                           review._sha(TWO_HUNK_BEFORE), TWO_HUNK_AFTER,
                           "edit", clauses=clauses, hunks=hunks,
                           before_text=TWO_HUNK_BEFORE)


# ------------------------------------ 1. R3-BUG-01: Term.newLine() & reload ---

def test_term_newline_assigns_cur_and_has_defensive_null_guards():
    src = APP_JS.read_text(encoding="utf-8")
    m = re.search(
        r"Term\.prototype\.newLine\s*=\s*function\s*\(\)\s*\{([\s\S]*?)\n\};",
        src)
    assert m, "Term.prototype.newLine not found in app.js"
    body = m.group(1)
    assert "this.el.appendChild(div);" in body
    assert "this.cur = div;" in body
    assert body.index("this.el.appendChild(div);") < body.index("this.cur = div;")

    # Defensive null guards in span(), write(), reset()
    span_m = re.search(
        r"Term\.prototype\.span\s*=\s*function\s*\(text\)\s*\{([\s\S]*?)\n\};",
        src)
    assert span_m and "!this.cur" in span_m.group(1)
    write_m = re.search(
        r"Term\.prototype\.write\s*=\s*function\s*\(text\)\s*\{([\s\S]*?)\n\};",
        src)
    assert write_m and "!this.el" in write_m.group(1)
    reset_m = re.search(
        r"Term\.prototype\.reset\s*=\s*function\s*\(text\)\s*\{([\s\S]*?)\n\};",
        src)
    assert reset_m and "!this.el" in reset_m.group(1)


@requires_pw
def test_gui_reload_after_terminal_output_does_not_crash_and_meets_narrow_layout(
        tmp_path, monkeypatch):
    from playwright.sync_api import sync_playwright

    cfg_path = tmp_path / "cfg" / "config.toml"
    monkeypatch.setattr(config_mod, "USER_CONFIG", cfg_path)
    monkeypatch.setattr(core, "USER_CONFIG", cfg_path)
    monkeypatch.setattr(settings_api, "USER_CONFIG", cfg_path)
    monkeypatch.setattr(core, "RECENT_FILE", tmp_path / "cfg" / "gui_recent.json")
    for var in [k for k in os.environ if k.startswith("PEDACITO_")]:
        monkeypatch.delenv(var)

    core.STATE.__init__()
    core.STATE.term_write("\x1b[32mhello from terminal\x1b[0m\nline two\n")
    fd = _two_hunk_fd()
    core.STATE.stage_review({
        "task": "Two-hunk task",
        "diffs": [fd],
        "writer": lambda: (True, "written"),
        "partial_ok": True,
        "clause_texts": {
            "R1": "Clamp the discount in apply_discount.",
            "R2": "Add express handling to compute_shipping.",
        },
    })

    srv = server.create_server(0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    page_errors: list[str] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 800})
                page.on("pageerror", lambda err: page_errors.append(str(err)))
                page.goto(url, wait_until="domcontentloaded", timeout=15000)
                page.wait_for_selector("#term .tl", state="attached", timeout=10000)
                # Reload after terminal output is already in snapshot.term.
                # The snapshot can resolve before WorkspacePane mounts, so the
                # page must retain the text and render it once the pane exists.
                page.reload(wait_until="domcontentloaded", timeout=15000)
                page.wait_for_selector("#reviewBar", timeout=10000)
                # The snapshot is applied asynchronously after reload; wait for
                # the restored terminal text rather than sampling a possibly
                # pre-effect DOM. This is bounded and still asserts the text.
                page.wait_for_function(
                    "() => { const el = document.querySelector('#term'); "
                    "if (!el) return false; const text = el.innerText || ''; "
                    "return text.includes('hello from terminal') "
                    "&& text.includes('line two'); }",
                    timeout=10000)
                assert page_errors == [], f"Uncaught page errors on reload: {page_errors}"
                term_text = page.locator("#term").inner_text()
                assert "hello from terminal" in term_text
                assert "line two" in term_text

                # R3-BUG-04: narrow viewport 420x800 must have zero horizontal overflow
                # and keep topbar controls reachable.
                page.set_viewport_size({"width": 420, "height": 800})
                page.wait_for_timeout(150)
                overflow = page.evaluate(
                    "document.documentElement.scrollWidth - window.innerWidth")
                assert overflow <= 0, f"Horizontal overflow at 420x800: {overflow}px"
                for sel in ("#projMenuBtn", "#settingsBtn", "#themeBtn",
                            "#langSwitch", "#exitBtn"):
                    box = page.locator(sel).bounding_box()
                    assert box is not None, f"{sel} not visible at 420x800"
                    assert box["x"] >= 0 and box["x"] + box["width"] <= 421, (
                        f"{sel} out of 420px viewport bounds: {box}")
            finally:
                browser.close()
    finally:
        core.STATE.bus.close()
        srv.shutdown()
        srv.server_close()


# --------------------------- 2. R3-BUG-02: Partial Hunk-Review Provenance -----

def test_partial_hunk_review_persists_accepted_hunks_line_stats_and_unaccepted_clauses():
    st = core.State()
    fd = _two_hunk_fd("pricing.py", clauses=("R1", "R2"))
    assert len(fd.hunks) == 2
    assert fd.hunks[0].added == 1 and fd.hunks[0].removed == 0
    assert fd.hunks[1].added == 3 and fd.hunks[1].removed == 2
    assert fd.added == 4 and fd.removed == 2

    clause_texts = {
        "R1": "Clamp the discount in apply_discount in pricing.py.",
        "R2": "Add express handling to compute_shipping in pricing.py.",
    }
    rv = st.stage_review({
        "task": "Clamp apply_discount and add express to compute_shipping.",
        "diffs": [fd],
        "writer": lambda: (True, "1 file(s) written: pricing.py"),
        "partial_ok": True,
        "clause_texts": clause_texts,
    })

    # Accept only hunk 0 (serves R1; hunk 1 serving R2 is rejected)
    ok, detail = st.decide_review(
        rv.id, "apply", ["pricing.py"], accepted_hunks={"pricing.py": [0]})
    assert ok
    assert "R2" in detail

    meta = st.review_meta()
    assert meta["outcome"] == "applied"
    assert meta["accepted_hunks"] == {"pricing.py": [0]}
    assert meta["unaccepted"] == ["R2"]
    fmeta = meta["files"][0]
    assert fmeta["accepted_hunks"] == [0]
    # Line stats reflect only the accepted hunk (hunk 0: +1 / -0), not the full +4 / -2
    assert fmeta["added"] == 1
    assert fmeta["removed"] == 0

    brief = st.review_meta_brief()
    assert brief["accepted_hunks"] == {"pricing.py": [0]}
    assert brief["unaccepted"] == ["R2"]
    assert brief["files"][0]["added"] == 1
    assert brief["files"][0]["removed"] == 0

    # Review message in transcript records hunks_accepted and hunks_total
    rev_msg = next(m for m in st.messages if m["role"] == "review")
    assert rev_msg["hunks_accepted"] == 1
    assert rev_msg["hunks_total"] == 2

    # App.js displays concise partial-apply counts
    app_js = APP_JS.read_text(encoding="utf-8")
    assert "accepted ' + acc + ' of ' + tot + ' hunks" in app_js


# ----------------------- 3. R3-BUG-03: Thread Persistence for Chat Cards ------

def test_thread_save_and_load_preserves_card_metadata_and_legacy_fallback(tmp_path):
    st = core.State()
    st.set_root(tmp_path)
    tid = st.thread_id

    st.add_message("question", "Which database?", qid=7, answered=True,
                   answer="sqlite")
    st.add_message("review", "Add retry logic", review_id=3, files=2,
                   outcome="applied", detail="1 file(s) written",
                   hunks_accepted=1, hunks_total=2)
    st.add_message("assistant", "Applied changes", kind="review_applied",
                   review_id=3, files=["client.py"])
    st.save_active_thread(force=True)

    thread_path = core.threads_dir(tmp_path) / f"{tid}.json"
    raw = json.loads(thread_path.read_text(encoding="utf-8"))
    by_role = {m["role"]: m for m in raw["messages"]}
    assert by_role["question"]["qid"] == 7
    assert by_role["question"]["answered"] is True
    assert by_role["question"]["answer"] == "sqlite"
    assert by_role["review"]["files"] == 2
    assert by_role["review"]["outcome"] == "applied"
    assert by_role["review"]["review_id"] == 3
    assert by_role["review"]["detail"] == "1 file(s) written"
    assert by_role["review"]["hunks_accepted"] == 1
    assert by_role["review"]["hunks_total"] == 2
    assert by_role["assistant"]["kind"] == "review_applied"

    loaded = core.read_thread_file(thread_path)
    assert loaded is not None
    l_by_role = {m["role"]: m for m in loaded["messages"]}
    assert l_by_role["question"]["qid"] == 7
    assert l_by_role["question"]["answered"] is True
    assert l_by_role["review"]["files"] == 2
    assert l_by_role["review"]["outcome"] == "applied"
    assert l_by_role["review"]["review_id"] == 3
    assert l_by_role["review"]["detail"] == "1 file(s) written"
    assert l_by_role["review"]["hunks_accepted"] == 1
    assert l_by_role["review"]["hunks_total"] == 2

    # Legacy thread without files/outcome on review card:
    legacy_path = core.threads_dir(tmp_path) / "legacy01.json"
    legacy_path.write_text(json.dumps({
        "id": "legacy01",
        "title": "legacy thread",
        "created_at": 10.0,
        "updated_at": 20.0,
        "messages": [{"role": "review", "content": "old task", "ts": 12.0}],
    }), encoding="utf-8")
    leg = core.read_thread_file(legacy_path)
    assert leg is not None
    assert "files" not in leg["messages"][0]

    # Verify ReviewCard in app.js guards against undefined files and stale Open review
    app_js = APP_JS.read_text(encoding="utf-8")
    rc_match = re.search(
        r"function ReviewCard\(props\)\s*\{([\s\S]*?)\n\}", app_js)
    assert rc_match, "ReviewCard not found in app.js"
    rc_body = rc_match.group(1)
    assert "liveMatch" in rc_body
    assert "fileCount != null" in rc_body
    assert "activeRev && activeRev.pending" in rc_body


# ---------------------- 4. R3-BUG-04: Narrow-Viewport Responsive CSS ----------

def test_narrow_viewport_css_has_flex_wrap_and_min_width_rules():
    css = APP_CSS.read_text(encoding="utf-8")
    assert "flex-wrap: wrap" in css
    assert "max-width: 100%" in css
    assert "min-width: 0" in css
    assert "@media (max-width: 900px)" in css


# ---------------------- 5. R3-BUG-05: Deterministic /compact Handoffs ---------

def test_handoff_material_collects_tui_receipts_and_ignores_numbered_source_docstrings(
        tmp_path):
    msgs = [
        {"role": "user", "content": "Add retry logic to client.py"},
        # Numbered source lines from a read observation mentioning "pytest unit tests"
        {"role": "user", "content": (
            'RESULT:\n'
            '   1 | """Client module tested by pytest unit tests."""\n'
            '   2 | def fetch():\n'
            '   3 |     return 1\n'
        )},
        {"role": "assistant", "content": (
            'I updated the docstring `"""Run pytest unit tests for client."""` '
            'and decided to use exponential backoff.\n'
            'Next step: wire timeout config.'
        )},
        {"role": "receipt", "content": "\u2713 applied client.py: 1 block applied"},
        {"role": "receipt", "content": "\u2717 failed old_client.py: anchor not found"},
        {"role": "receipt", "content": "tests: GREEN"},
    ]
    _before, facts = memory._handoff_material(tmp_path, msgs)
    assert "client.py" in facts["applied"]
    assert "old_client.py" in facts["failed"]
    # Docstring / numbered source lines mentioning pytest must NOT leak into tests
    assert all("1 |" not in t and '"""' not in t for t in facts["tests"])
    assert any("tests: GREEN" in t for t in facts["tests"])

    rendered = memory._render_handoff(facts)
    assert "- Applied: client.py" in rendered
    assert "- FAILED edits: old_client.py" in rendered
    assert "- tests: GREEN" in rendered
    assert '1 | """' not in rendered


def test_handoff_material_falls_back_to_checkpoints_or_explicit_files_changed(
        tmp_path):
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    checkpoints.create_checkpoint(tmp_path, "baseline", mode="task")
    (tmp_path / "app.py").write_text("x = 2\n", encoding="utf-8")
    checkpoints.create_checkpoint(tmp_path, "update x", mode="task")

    msgs = [
        {"role": "user", "content": "Update x in app.py"},
        {"role": "assistant", "content": "Done updating x."},
    ]
    # Without receipts in messages, fallback to checkpoints.list_checkpoints(root)
    _before, facts = memory._handoff_material(tmp_path, msgs)
    assert "app.py" in facts["applied"]

    # Explicit files_changed parameter on consolidate_session / _handoff_material
    _before2, facts2 = memory._handoff_material(
        None, msgs, files_changed=["explicit/mod.py"])
    assert "explicit/mod.py" in facts2["applied"]


# ---------------------- 6. P3 Polish: R3-BUG-07, R3-BUG-09, domcheck ----------

def test_gui_server_catches_client_disconnect_silently(capsys):
    for exc_cls in (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
        handler = server._Handler.__new__(server._Handler)
        handler.path = "/api/state"
        handler.headers = {"Host": "127.0.0.1:8000"}
        handler.server = SimpleNamespace(server_address=("127.0.0.1", 8000))
        handler.rfile = io.BytesIO(b"")

        class BrokenWriter:
            def write(self, _data):
                raise exc_cls(10053, "An established connection was aborted")

            def flush(self):
                raise exc_cls(10053, "An established connection was aborted")

        handler.wfile = BrokenWriter()
        handler.send_response = lambda *a, **k: None
        handler.send_header = lambda *a, **k: None
        handler.end_headers = lambda *a, **k: None

        # _send, do_GET, do_POST, and _events must return cleanly without stderr noise
        handler._send(200, b"ok", "text/plain")
        handler.do_GET()
        handler.headers["Content-Type"] = "application/json"
        handler.do_POST()
        handler._events({})

    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "10053" not in captured.err


def test_settings_profiles_list_has_ellipsis_title_and_score_chip():
    sjs = SETTINGS_JS.read_text(encoding="utf-8")
    css = APP_CSS.read_text(encoding="utf-8")
    assert "pname grow ellip" in sjs
    assert "scoreChip" in sjs
    assert "Number(cal.card_score || 0).toFixed(2)" in sjs
    pname_m = re.search(r"\.pname\s*\{([^}]*)\}", css)
    assert pname_m
    pbody = pname_m.group(1)
    for prop in ("min-width: 0", "white-space: nowrap",
                 "overflow: hidden", "text-overflow: ellipsis"):
        assert prop in pbody


def test_domcheck_baseline_prints_unchanged_confirmation(tmp_path, monkeypatch,
                                                         capsys):
    snap = {"elements": [{"tag": "div", "id": "app", "classes": ""}],
            "controls": [], "text_len": 100}
    base_file = tmp_path / "base.json"
    domcheck.save_snapshot(base_file, snap)

    monkeypatch.setattr(
        domcheck, "check",
        lambda root, entry="", timeout=45, baseline=None: domcheck.DomResult(
            entry="index.html", findings=[], snapshot=snap))

    code = cli.main(["domcheck", str(tmp_path), "--baseline", str(base_file)])
    out = capsys.readouterr().out
    assert code == 0
    assert "domcheck: ok (unchanged vs baseline)" in out
