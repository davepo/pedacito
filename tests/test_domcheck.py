"""U6: the headless DOM smoke check (pedacito/domcheck.py).

The deterministic helpers (entry detection, snapshot diffing, finding keys,
the missing-install path) are tested without a browser. The live checks load
real pages in headless Chromium and are skipped when Playwright is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import domcheck  # noqa: E402

_HAS_PW = domcheck.available()[0]
requires_pw = pytest.mark.skipif(
    not _HAS_PW, reason="Playwright + Chromium not installed")


# --- deterministic helpers ---------------------------------------------------

def test_find_entry_prefers_root_index(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "index.html").write_text("<html></html>")
    (tmp_path / "index.html").write_text("<html></html>")
    assert domcheck.find_entry(tmp_path) == "index.html"


def test_find_entry_prefers_module_page_over_plain(tmp_path):
    (tmp_path / "about.html").write_text("<html><body>hi</body></html>")
    (tmp_path / "app.html").write_text(
        "<html><script type=\"module\" src=\"a.js\"></script></html>")
    assert domcheck.find_entry(tmp_path) == "app.html"


def test_find_entry_override_and_missing(tmp_path):
    (tmp_path / "page.html").write_text("<html></html>")
    assert domcheck.find_entry(tmp_path, "page.html") == "page.html"
    assert domcheck.find_entry(tmp_path, "nope.html") == ""
    assert domcheck.find_entry(tmp_path) == "page.html"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert domcheck.find_entry(empty) == ""


def test_finding_key_ignores_shifting_digits():
    a = domcheck.DomFinding("page_error", "error", "boom at app.js:12")
    b = domcheck.DomFinding("page_error", "error", "boom at app.js:99")
    assert a.key == b.key
    c = domcheck.DomFinding("load_error", "error", "boom at app.js:12")
    assert a.key != c.key


def _snap(elements, controls=(), text_len=100):
    return {"elements": elements, "controls": list(controls),
            "text_len": text_len}


def test_diff_mounted_flags_lost_named_elements_and_controls():
    before = _snap(
        [{"tag": "div", "id": "app", "classes": ""},
         {"tag": "button", "id": "go", "classes": ""},
         {"tag": "span", "id": "label", "classes": ""}],
        controls=[{"tag": "button", "id": "go", "classes": ""}])
    after = _snap([{"tag": "div", "id": "app", "classes": ""}])
    found = domcheck._diff_mounted(before, after)
    assert found and found[0].kind == "mount_regression"
    assert found[0].severity == "error"
    assert "go" in found[0].message


def test_diff_mounted_ignores_anonymous_churn_and_additions():
    before = _snap([{"tag": "div", "id": "app", "classes": ""},
                    {"tag": "p", "id": "", "classes": "x"}])
    # the named element stays; only an anonymous <p> is swapped, and a new
    # element is added -- not a regression
    after = _snap([{"tag": "div", "id": "app", "classes": ""},
                   {"tag": "p", "id": "", "classes": "y"},
                   {"tag": "section", "id": "", "classes": ""}])
    assert domcheck._diff_mounted(before, after) == []


def test_diff_mounted_flags_text_collapse():
    before = _snap([{"tag": "div", "id": "app", "classes": ""}], text_len=800)
    after = _snap([{"tag": "div", "id": "app", "classes": ""}], text_len=100)
    found = domcheck._diff_mounted(before, after)
    assert found and "text collapsed" in found[0].message


def test_check_reports_missing_install_without_a_browser(monkeypatch):
    monkeypatch.setattr(domcheck, "available", lambda: (False, "install it"))
    res = domcheck.check(Path("/nonexistent"))
    assert res.error == "install it" and not res.ok


def test_check_reports_no_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(domcheck, "available", lambda: (True, ""))
    res = domcheck.check(tmp_path)
    assert "no HTML entry" in res.error and not res.ok


def test_snapshot_roundtrip(tmp_path):
    snap = _snap([{"tag": "div", "id": "app", "classes": ""}])
    p = tmp_path / "snap.json"
    domcheck.save_snapshot(p, snap)
    assert domcheck.load_snapshot(p) == snap
    assert domcheck.load_snapshot(tmp_path / "missing.json") is None


# --- live headless checks ----------------------------------------------------

INDEX = ('<!doctype html><html><head><title>t</title></head>'
         '<body><div id="app"></div>'
         '<script type="module" src="app.js"></script></body></html>')


def _project(tmp_path, app_js):
    (tmp_path / "index.html").write_text(INDEX)
    (tmp_path / "app.js").write_text(app_js)
    return tmp_path


@requires_pw
def test_clean_mounted_page_passes(tmp_path):
    _project(tmp_path, (
        "const app = document.getElementById('app');\n"
        "const b = document.createElement('button');\n"
        "b.id = 'go'; b.textContent = 'Go';\n"
        "b.addEventListener('click', () => {});\n"
        "app.appendChild(b);\n"))
    res = domcheck.check(tmp_path)
    assert not res.error, res.error
    assert res.ok, [str(f) for f in res.findings]
    assert res.snapshot and res.snapshot["body_children"] >= 1


@requires_pw
def test_uncaught_page_error_is_reported(tmp_path):
    _project(tmp_path, "throw new Error('boom');\n")
    res = domcheck.check(tmp_path)
    assert not res.error
    assert any(f.kind == "page_error" and f.severity == "error"
               for f in res.findings), [str(f) for f in res.findings]
    assert not res.ok


@requires_pw
def test_missing_module_is_a_load_error(tmp_path):
    (tmp_path / "index.html").write_text(
        '<!doctype html><html><body><div id="app"></div>'
        '<script type="module" src="missing.js"></script></body></html>')
    res = domcheck.check(tmp_path)
    assert not res.error
    assert any(f.kind == "load_error" for f in res.findings), \
        [str(f) for f in res.findings]
    assert not res.ok


@requires_pw
def test_unbound_control_is_a_warning_and_delegation_suppresses_it(tmp_path):
    # a lone button with no listener anywhere -> unbound (warn, not error)
    (tmp_path / "index.html").write_text(
        '<!doctype html><html><body><button id="lonely">x</button>'
        '</body></html>')
    res = domcheck.check(tmp_path)
    assert not res.error
    assert any(f.kind == "unbound_control" and f.severity == "warn"
               for f in res.findings), [str(f) for f in res.findings]
    assert res.ok                        # warnings do not gate

    # a document-level (delegated) listener suppresses the finding
    (tmp_path / "index.html").write_text(
        '<!doctype html><html><body><button id="lonely">x</button>'
        '<script>document.addEventListener("click", () => {});</script>'
        '</body></html>')
    res = domcheck.check(tmp_path)
    assert not any(f.kind == "unbound_control" for f in res.findings), \
        [str(f) for f in res.findings]


@requires_pw
def test_snapshot_diff_catches_a_replaced_entry_module(tmp_path):
    # BUG-216 shape: the entry module mounts UI, then is replaced by one that
    # mounts nothing; the mounted-surface diff must flag the regression.
    _project(tmp_path, (
        "const app = document.getElementById('app');\n"
        "app.innerHTML = '<button id=\"go\">Go</button>"
        "<span id=\"label\">hi</span>';\n"
        "document.getElementById('go').addEventListener('click', () => {});\n"))
    before = domcheck.check(tmp_path)
    assert before.ok and before.snapshot
    # replace the entry module with one that no longer mounts the UI
    (tmp_path / "app.js").write_text("// replaced module: mounts nothing\n")
    after = domcheck.check(tmp_path, baseline=before.snapshot)
    assert any(f.kind == "mount_regression" and f.severity == "error"
               for f in after.findings), [str(f) for f in after.findings]
    assert not after.ok