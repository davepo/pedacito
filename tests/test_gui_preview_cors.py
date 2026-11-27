"""BUG-06: the Live Preview iframe is sandboxed without `allow-same-origin`, so
its origin is opaque and every module script it loads is a CORS request with
`Origin: null`. `/preview/*` must answer that request so ES modules run, while
the sandbox isolation (no `allow-same-origin`) and the `/api/*` and `/` headers
stay exactly as they were."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito.gui import core  # noqa: E402
from test_gui import gui as _shared_gui  # noqa: E402,F401  (pytest re-export)


@pytest.fixture
def gui(_shared_gui):  # noqa: F811  (re-export the fixture imported above)
    return _shared_gui


APP_HTML = (
    "<!doctype html><meta charset=utf-8><title>Module app</title>"
    '<p id="out">waiting</p>'
    '<button id="go">go</button>'
    '<script type="module" src="app.js"></script>'
)
APP_JS = (
    'const out = document.getElementById("out");\n'
    'out.textContent = "module ran";\n'
    'document.getElementById("go").addEventListener("click", () => {\n'
    '  out.textContent = "clicked";\n'
    "});\n"
)


def _module_project(tmp_path):
    root = tmp_path / "site"
    (root / "web").mkdir(parents=True)
    (root / "web" / "index.html").write_bytes(APP_HTML.encode("utf-8"))
    (root / "web" / "app.js").write_bytes(APP_JS.encode("utf-8"))
    core.STATE.root = root
    return root


def _headers(resp):
    return {k.lower(): v for k, v in resp[1].items()}


def test_preview_module_script_answers_the_opaque_origin(gui, tmp_path):
    _module_project(tmp_path)
    status, hdrs, body = gui.req("GET", "/preview/web/app.js",
                                 headers={"Origin": "null"})
    assert status == 200
    h = _headers((status, hdrs, body))
    assert h["access-control-allow-origin"] == "null"
    assert "origin" in h["vary"].lower()
    assert body == APP_JS.encode("utf-8")
    assert "javascript" in h["content-type"]


def test_preview_page_is_also_readable_from_the_opaque_origin(gui, tmp_path):
    _module_project(tmp_path)
    resp = gui.req("GET", "/preview/web/index.html?v=1", headers={"Origin": "null"})
    assert resp[0] == 200
    assert _headers(resp)["access-control-allow-origin"] == "null"


def test_preview_never_echoes_a_real_website_origin(gui, tmp_path):
    """Project files must not become readable by arbitrary websites."""
    _module_project(tmp_path)
    for origin in ("https://evil.example", "http://127.0.0.1:9999", "*"):
        resp = gui.req("GET", "/preview/web/app.js", headers={"Origin": origin})
        assert resp[0] == 200
        assert "access-control-allow-origin" not in _headers(resp), origin
    resp = gui.req("GET", "/preview/web/app.js")
    assert "access-control-allow-origin" not in _headers(resp)


def test_preview_keeps_the_sandbox_without_allow_same_origin(gui, tmp_path):
    _module_project(tmp_path)
    for path in ("/preview/web/index.html", "/preview/web/app.js"):
        csp = _headers(gui.req("GET", path))["content-security-policy"]
        directives = [d.strip() for d in csp.split(";")]
        sandbox = [d for d in directives if d.startswith("sandbox")]
        assert sandbox and "allow-same-origin" not in sandbox[0], path
        assert "allow-scripts" in sandbox[0]
        assert "unsafe-eval" not in csp


@pytest.mark.parametrize("path", ["/api/state", "/api/settings", "/"])
def test_cors_header_is_never_sent_outside_preview(gui, tmp_path, path):
    _module_project(tmp_path)
    resp = gui.req("GET", path, headers={"Origin": "null"})
    assert "access-control-allow-origin" not in _headers(resp), path


def _playwright_browsers(monkeypatch):
    """conftest points LOCALAPPDATA/HOME-ish paths at a temp folder, which
    hides the installed browsers: pin Playwright to the real cache."""
    import os
    if os.environ.get("PLAYWRIGHT_BROWSERS_PATH"):
        return
    if os.name == "nt":
        real = Path(os.path.expanduser("~")) / "AppData" / "Local" / "ms-playwright"
    else:
        real = Path(os.path.expanduser("~")) / ".cache" / "ms-playwright"
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(real))


def test_preview_module_app_runs_in_a_sandboxed_iframe(gui, tmp_path, monkeypatch):
    """Live check: a real Chromium loads the sandboxed preview iframe and the
    ES module must execute (no CORS or CSP console errors)."""
    sync_api = pytest.importorskip("playwright.sync_api")
    _module_project(tmp_path)
    _playwright_browsers(monkeypatch)
    port = gui.port
    parent_url = f"http://127.0.0.1:{port}/__preview_test_parent"
    parent = (
        "<!doctype html><title>parent</title>"
        f'<iframe id="pv" sandbox="allow-scripts allow-forms allow-modals" '
        f'src="http://127.0.0.1:{port}/preview/web/index.html?v=1" '
        'width="600" height="400"></iframe>'
    )
    with sync_api.sync_playwright() as pw:
        try:
            browser = pw.chromium.launch()
        except Exception as exc:  # no browser binary on this machine
            pytest.skip(f"chromium unavailable: {exc}")
        try:
            page = browser.new_page()
            errors = []
            page.on("console", lambda m: errors.append(m.text)
                    if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.route(parent_url, lambda route: route.fulfill(
                status=200, content_type="text/html", body=parent))
            page.goto(parent_url)
            frame = None
            for _ in range(150):
                frame = page.frame(url=lambda u: "/preview/" in u)
                if frame is not None:
                    break
                page.wait_for_timeout(100)
            assert frame is not None
            frame.wait_for_function(
                "document.getElementById('out') && "
                "document.getElementById('out').textContent === 'module ran'",
                timeout=15000)
            frame.click("#go")
            frame.wait_for_function(
                "document.getElementById('out').textContent === 'clicked'",
                timeout=5000)
            assert not [e for e in errors if "CORS" in e or "Content Security" in e
                        or "blocked" in e.lower()], errors
        finally:
            browser.close()
