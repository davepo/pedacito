"""Bug Hunt #12 crucible: headless browser capture and JS-eval diagnostics.

The browser action exists to catch what a screenshot alone cannot: a 404
asset that leaves the page half-rendered, a console error, a key held across
real animation frames, and a model-authored eval that must report its value
instead of hanging the loop forever.
"""
from __future__ import annotations

import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import browser, tools
from pedacito.local_server import LocalHTTPServer
from pedacito.config import Config
from pedacito.index import Index


# --------------------------------------------------------------------- scanners

def test_unclosed_brackets_names_each_opener_with_its_offset():
    assert "unclosed" in browser._unclosed_brackets("foo(1, 2")
    assert "`(` from char 3" in browser._unclosed_brackets("foo(1, 2")

    hint = browser._unclosed_brackets("var a = [1, 2")
    assert "unclosed" in hint and "`[` from char 8" in hint

    hint = browser._unclosed_brackets("if (x) {")
    assert "`{` from char 7" in hint

    # Several openers: the report names the innermost three and counts the rest.
    deep = "((([[{" + "["
    hint = browser._unclosed_brackets(deep)
    assert "unclosed" in hint and "more" in hint


def test_unclosed_brackets_ignores_strings_comments_and_balanced_code():
    assert browser._unclosed_brackets("(x + [1, {a: 2}])") == ""
    assert browser._unclosed_brackets("var s = '})'; s + \"(\"") == ""
    assert browser._unclosed_brackets("var t = `a{b(`; t") == ""
    assert browser._unclosed_brackets("// } ] )\nvar x = 1; x") == ""
    assert browser._unclosed_brackets("/* ( [ { */ 1 + 1") == ""
    # A bracket-looking char escaped inside a string stays inside it.
    assert browser._unclosed_brackets("var s = 'it\\'s ok'; s") == ""


def test_unclosed_brackets_reports_crossing_and_unterminated_strings():
    crossed = browser._unclosed_brackets("([)]")
    # `)` at char 2 meets `[` from char 1, while `(` from char 0 is older.
    assert "brackets cross at char 2" in crossed
    assert "`[` from char 1" in crossed
    assert "([)]" in crossed or "([]" in crossed

    hint = browser._unclosed_brackets("var s = 'abc")
    assert "string is still open" in hint
    assert "'" in hint


def test_eval_syntax_error_states_the_intact_length_and_the_tail():
    expr = "var a = [1, 2"
    text = browser._eval_syntax_error(
        expr, "SyntaxError: Unexpected end of input\n  at page.evaluate")
    assert "eval SYNTAX ERROR" in text
    assert f"{len(expr)} chars" in text
    assert "nothing was truncated in transit" in text
    assert "`[` from char 8" in text
    assert expr[-70:] in text


# ------------------------------------------------------------------ raced eval

def test_raced_eval_wraps_in_promise_race_and_keeps_the_expression_verbatim():
    expr = "const a = [1, 2]; a.map(x => x * 2)"
    src = browser._raced_eval(expr, 3)
    assert "Promise.race" in src
    assert "setTimeout" in src
    assert json.dumps(expr) in src
    assert "typeof v === 'function'" in src
    assert "3000" in src


@pytest.fixture()
def page():
    if not browser.available():
        pytest.skip("Playwright is not installed")
    from playwright.sync_api import sync_playwright
    try:
        pw = sync_playwright().start()
        chromium = pw.chromium.launch(headless=True)
    except Exception as e:                       # pragma: no cover - env
        pytest.skip(f"Chromium is unavailable: {e}")
    pg = chromium.new_page()
    pg.goto("about:blank")
    try:
        yield pg
    finally:
        chromium.close()
        pw.stop()


def test_raced_eval_resolves_values_promises_and_bare_functions(page):
    assert page.evaluate(browser._raced_eval("1 + 1", 2)) == 2
    assert page.evaluate(browser._raced_eval("Promise.resolve(7)", 2)) == 7
    assert page.evaluate(
        browser._raced_eval("(function(){ return 9; })", 2)) == 9
    # A never-settling promise becomes a rejection, not a hung evaluate.
    with pytest.raises(Exception) as exc:
        page.evaluate(browser._raced_eval("new Promise(function(){})", 1))
    assert "did not settle" in str(exc.value)


# --------------------------------------------------------------------- parser

def test_parse_actions_complex_chain_and_eval_consumes_the_rest():
    spec = ("press Enter; hold ArrowUp 1.5; click #start; fill #name Ada; "
            "type hello; wait 0.5; "
            "eval const a = 1; const b = 2; a + b")
    actions = browser.parse_actions(spec)
    assert actions == [
        ("press", "Enter"),
        ("hold", ("ArrowUp", 1.5)),
        ("click", "#start"),
        ("fill", ("#name", "Ada")),
        ("type", "hello"),
        ("wait", 0.5),
        ("eval", "const a = 1; const b = 2; a + b"),
    ]
    # The eval is ONE action, not three shredded fragments.
    assert actions[-1][0] == "eval"
    assert actions[-1][1].count(";") == 2
    assert [a for a, _ in actions].count("press") == 1


def test_parse_actions_hold_clamps_and_bad_values_are_safe():
    assert browser.parse_actions("hold ArrowRight 0") == [
        ("hold", ("ArrowRight", 0.05))]
    assert browser.parse_actions("hold ArrowRight -3") == [
        ("hold", ("ArrowRight", 0.05))]
    assert browser.parse_actions("hold ArrowRight 99") == [
        ("hold", ("ArrowRight", 30.0))]
    assert browser.parse_actions("hold ArrowRight nope") == [
        ("hold", ("ArrowRight", 1.0))]
    assert browser.parse_actions("hold") == []
    assert browser.parse_actions("eval") == []
    assert browser.parse_actions("js 1 + 1") == [("eval", "1 + 1")]


# ------------------------------------------------------- live capture over HTTP

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Dash</title></head>
<body tabindex="0">
<canvas id="status" width="320" height="120"></canvas>
<div id="ticks">0</div>
<script>
window.__ticks = 0;
var down = false, ticks = 0;
function frame() {
  if (down) {
    ticks += 1;
    window.__ticks = ticks;
    document.getElementById('ticks').textContent = String(ticks);
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
addEventListener('keydown', function (e) {
  if (e.key === 'ArrowRight') { down = true; e.preventDefault(); }
});
addEventListener('keyup', function (e) {
  if (e.key === 'ArrowRight') { down = false; }
});
fetch('/missing-asset.js').then(function (r) {
  if (!r.ok) { console.error('asset missing: ' + r.status); }
})["catch"](function () { console.error('asset request failed'); });
</script>
</body></html>
"""


class _DashHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = _PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.send_header("Content-Type", "text/javascript")
        self.end_headers()
        self.wfile.write(b"console.error('missing');")

    def log_message(self, *args):
        pass


@pytest.fixture()
def live_page():
    httpd = LocalHTTPServer(("127.0.0.1", 0), _DashHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)


def test_live_capture_records_404_console_error_and_held_key(
        live_page, tmp_path):
    if not browser.available():
        pytest.skip("Playwright is not installed")
    ok, info = browser.run(
        live_page, actions="hold ArrowRight 0.3; eval window.__ticks",
        wait=0.2, out=tmp_path / "shot.png", timeout=15)
    if not ok and ("Chromium" in str(info) or "executable doesn't exist"
                   in str(info).lower()):
        pytest.skip(str(info))
    assert ok, info
    assert info["method"].startswith("Playwright")
    assert info["canvas"] == "320x120"
    assert any("404" in e and "missing-asset.js" in e
               for e in info["network_errors"]), info["network_errors"]
    assert any("asset missing" in e for e in info["console"]), info["console"]
    assert any(a.startswith("held ArrowRight") for a in info["actions"])
    evals = [a for a in info["actions"] if a.startswith("eval'd window.__ticks")]
    assert evals, info["actions"]
    ticks = int(re.search(r"->\s*(\d+)", evals[0]).group(1))
    assert ticks >= 3, (ticks, info["actions"])
    assert (tmp_path / "shot.png").is_file()


# ------------------------------------------------------------ host restriction

def _fake_run(monkeypatch, seen):
    def fake(url, **kwargs):
        seen.append((url, kwargs.get("actions")))
        return True, {"method": "fake", "url": url, "title": "f",
                      "text": "body", "canvas": "", "console": [],
                      "network_errors": [], "actions": [], "action_errors": []}
    monkeypatch.setattr(browser, "run", fake)


def test_browser_refuses_eval_on_a_non_allowlisted_remote_host(
        tmp_path, monkeypatch):
    seen: list[tuple[str, str]] = []
    _fake_run(monkeypatch, seen)
    cfg = Config()
    cfg.allow_browser = True
    cfg.browser_eval_hosts = ("localhost", "127.0.0.1", "::1")
    idx = Index(tmp_path, [])
    out = tools.tool_browser(idx, cfg, None, command="https://example.com/x",
                             pattern="eval document.cookie")
    assert out.startswith("ERROR:")
    assert "eval" in out and "only allowed on local pages" in out
    assert not seen, "the browser was launched before the host check"


def test_browser_allows_eval_on_loopback_and_named_hosts(tmp_path, monkeypatch):
    seen: list[tuple[str, str]] = []
    _fake_run(monkeypatch, seen)
    cfg = Config()
    cfg.allow_browser = True
    idx = Index(tmp_path, [])

    for url in ("http://127.0.0.1:8123/", "http://localhost:8123/"):
        out = tools.tool_browser(idx, cfg, None, command=url,
                                 pattern="eval window.__ticks")
        assert "BROWSER:" in out, out
    assert len(seen) == 2
    for url, actions in seen:
        host = urlparse(url).hostname
        assert host in ("127.0.0.1", "localhost")
        assert actions == "eval window.__ticks"

    seen.clear()
    cfg.browser_eval_hosts = ("example.com",)
    out = tools.tool_browser(idx, cfg, None, command="https://example.com/a",
                             pattern="eval 1 + 1")
    assert "BROWSER:" in out, out
    assert seen and urlparse(seen[0][0]).hostname == "example.com"
