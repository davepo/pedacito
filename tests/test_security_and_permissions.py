"""Security hardening and the live human-in-the-loop Permission Broker.

Covers `pedacito/security.py` (workspace containment, sensitive-file guard,
GUI CSRF origin guard) and the broker end to end: a gated tool call parks its
worker thread, `/api/state` reports `pending_permission`, and `/api/permission/
decide` resolves allow_once / allow_run / always / deny cleanly.
"""
import http.client
import json
import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import ClassVar

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import config as config_mod  # noqa: E402
from pedacito import index as index_mod  # noqa: E402
from pedacito import security, tools, webfetch  # noqa: E402
from pedacito.config import Config  # noqa: E402
from pedacito.gui import core, server, settings_api  # noqa: E402


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Client:
    def __init__(self, port):
        self.port = port

    def req(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        hdrs = {"Content-Type": "application/json", **(headers or {})}
        data = None if body is None and method == "GET" else json.dumps(body or {})
        conn.request(method, path, body=data, headers=hdrs)
        r = conn.getresponse()
        payload = r.read()
        out = (r.status, json.loads(payload) if payload[:1] in (b"{", b"[") else payload)
        conn.close()
        return out

    def get(self, path, headers=None):
        return self.req("GET", path, headers=headers)

    def post(self, path, body=None, headers=None):
        return self.req("POST", path, body or {}, headers=headers)


@pytest.fixture
def gui_broker(tmp_path, monkeypatch):
    """A fresh GUI server (isolated config) WITH the permission broker
    installed -- exactly what `pedacito gui` runs."""
    cfg_path = tmp_path / "cfg" / "config.toml"
    monkeypatch.setattr(config_mod, "USER_CONFIG", cfg_path)
    monkeypatch.setattr(core, "USER_CONFIG", cfg_path)
    monkeypatch.setattr(settings_api, "USER_CONFIG", cfg_path)
    monkeypatch.setattr(core, "RECENT_FILE", tmp_path / "cfg" / "gui_recent.json")
    for var in [k for k in os.environ if k.startswith("PEDACITO_")]:
        monkeypatch.delenv(var)
    core.STATE.__init__()
    srv = server.create_server(0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    core.install_permission_broker()
    c = Client(srv.server_address[1])
    c.state = core.STATE
    yield c
    core.uninstall_permission_broker()
    core.STATE.bus.close()
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def bare_broker(tmp_path, monkeypatch):
    """The broker without any HTTP server: hook + STATE only."""
    monkeypatch.setattr(core, "RECENT_FILE", tmp_path / "gui_recent.json")
    core.STATE.__init__()
    core.install_permission_broker()
    yield core.STATE
    core.uninstall_permission_broker()
    core.STATE.bus.close()


def _wait_pending(st, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        with st.lock:
            pp = dict(st.pending_permission or {})
        if pp:
            return pp
        time.sleep(0.02)
    raise AssertionError("no pending permission request appeared")


# ----------------------------------------------------------- path containment ---

def test_resolve_safe_path_blocks_traversal_and_symlink_escape(tmp_path):
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("x = 1\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "passwd").write_text("root:x:0:0\n")

    ok = security.resolve_safe_path(root, "src/app.py")
    assert ok == (root / "src" / "app.py").resolve()

    for bad in ("../../etc/passwd", "../outside/passwd", "src/../../escape",
                "/etc/passwd", str(outside / "passwd")):
        with pytest.raises(security.SecurityError):
            security.resolve_safe_path(root, bad)

    # a symlink inside the project that points outside must be refused
    link = root / "escape"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")
    with pytest.raises(security.SecurityError):
        security.resolve_safe_path(root, "escape/passwd")
    # and backslashes are normalised, never trusted
    with pytest.raises(security.SecurityError):
        security.resolve_safe_path(root, "..\\..\\windows\\system32")


# ---------------------------------------------------------- sensitive guard ---

@pytest.mark.parametrize("name", [
    ".env", ".env.local", ".env.production", "config/.env", "prod.env",
    "id_rsa", "id_ed25519", "server.pem", "tls.key", "deploy.key",
    ".npmrc", ".pypirc", ".netrc", "credentials.json", ".git-credentials",
    ".ssh/config", ".ssh/id_ed25519", ".aws/credentials",
])
def test_is_sensitive_path_blocks_secrets(name):
    assert security.is_sensitive_path(Path("proj") / name), name


@pytest.mark.parametrize("name", [
    ".env.example", ".env.sample", ".env.template", "app.py", "README.md",
    "id_rsa.pub", "server.crt", "config.json", "main.go", "keyboard.json",
])
def test_is_sensitive_path_allows_ordinary_files(name):
    assert not security.is_sensitive_path(Path("proj") / name), name


def test_sensitive_file_never_indexed(tmp_path):
    (tmp_path / ".env").write_text("API_KEY=super-secret\n")
    (tmp_path / "app.py").write_text("x = 1\n")
    idx = index_mod.Index(tmp_path, [])
    index_mod.reindex_file(idx, ".env", Config(), save=False)
    assert ".env" not in idx.files
    assert ".env" not in getattr(idx, "file_hashes", {})
    # the model-side adoption path must not fold it in either
    assert tools.adopt_if_on_disk(idx, Config(), ".env") is None
    # ordinary files still adopt normally
    assert tools.adopt_if_on_disk(idx, Config(), "app.py") == "app.py"


# ----------------------------------------------------- GUI server CSRF guard ---

def test_post_from_foreign_origin_is_403(gui_broker):
    # Origin: some website driving the loopback agent
    status, body = gui_broker.post("/api/clear", {},
                                   headers={"Origin": "https://evil.example.com"})
    assert status == 403 and body["ok"] is False
    # Referer: a classic cross-site form/post fingerprint
    status, body = gui_broker.post("/api/clear", {},
                                   headers={"Referer": "https://evil.example.com/pwn"})
    assert status == 403 and body["ok"] is False
    # both foreign
    status, _ = gui_broker.post("/api/shutdown", {}, headers={
        "Origin": "http://evil.example.com",
        "Referer": "http://evil.example.com/"})
    assert status == 403


def test_post_from_loopback_origin_referer_still_works(gui_broker):
    port = gui_broker.port
    for hdrs in ({"Origin": f"http://127.0.0.1:{port}"},
                 {"Origin": f"http://localhost:{port}"},
                 {"Referer": f"http://localhost:{port}/"},
                 {"Referer": f"http://[::1]:{port}/"},
                 {}):
        status, body = gui_broker.post("/api/clear", {}, headers=hdrs)
        assert status == 200 and body["ok"] is True, hdrs


# ------------------------------------------------------------- the broker ---

def _ask(st, **kw):
    """Call `STATE.request_permission` on a worker thread; returns the
    (thread, results) pair -- the thread parks until the modal answers."""
    kw.setdefault("category", security.CAT_SHELL)
    kw.setdefault("action_summary", "Run command: `pytest -q`")
    kw.setdefault("detail", "pytest -q")
    kw.setdefault("key", kw.get("detail", ""))
    out = {}

    def work():
        out["value"] = st.request_permission(
            {"category": kw["category"], "action_summary": kw["action_summary"],
             "detail": kw["detail"], "risk_level": kw.get("risk_level", "high"),
             "reason": kw.get("reason", ""), "key": kw["key"]})
    t = threading.Thread(target=work, daemon=True)
    t.start()
    return t, out


def test_broker_pauses_worker_and_resolves_allow_once(bare_broker):
    st = bare_broker
    t, out = _ask(st)
    pp = _wait_pending(st)
    assert pp["category"] == "shell" and "pytest" in pp["detail"]
    assert "created_at" in pp and pp["risk_level"] == "high"
    # the modal's payload reaches /api/state-shaped status too
    assert st.status()["pending_permission"]["id"] == pp["id"]
    ok, _detail = st.decide_permission(pp["id"], "allow_once")
    assert ok
    t.join(5)
    assert out["value"][0] is True
    with st.lock:
        assert st.pending_permission is None
    # allow_once is not a standing grant: the next ask is pending again
    t2, out2 = _ask(st)
    pp2 = _wait_pending(st)
    st.decide_permission(pp2["id"], "deny")
    t2.join(5)
    assert out2["value"][0] is False


def test_broker_allow_run_whitelists_the_pattern(bare_broker):
    st = bare_broker
    t, out = _ask(st, detail="pytest -q tests", key="pytest -q tests")
    pp = _wait_pending(st)
    st.decide_permission(pp["id"], "allow_run")
    t.join(5)
    assert out["value"] == (True, "")
    # second identical request: answered from the run cache, never pends
    t2, out2 = _ask(st, detail="pytest -q tests", key="pytest -q tests")
    t2.join(5)
    assert out2["value"] == (True, "")
    with st.lock:
        assert st.pending_permission is None
    # a DIFFERENT command is a new request
    t3, out3 = _ask(st, detail="pytest -q other", key="pytest -q other")
    pp3 = _wait_pending(st)
    st.decide_permission(pp3["id"], "deny")
    t3.join(5)
    assert out3["value"][0] is False
    # ... and a new job clears the run scope
    st.begin_job("task", "next run")
    t4, out4 = _ask(st, detail="pytest -q tests", key="pytest -q tests")
    pp4 = _wait_pending(st)
    st.decide_permission(pp4["id"], "deny")
    t4.join(5)
    assert out4["value"][0] is False


def test_broker_deny_returns_clean_redirect(bare_broker):
    st = bare_broker
    t, out = _ask(st, detail="rm -rf /var", key="rm -rf /var")
    pp = _wait_pending(st)
    ok, _ = st.decide_permission(pp["id"], "deny")
    assert ok
    t.join(5)
    allowed, msg = out["value"]
    assert allowed is False
    assert msg == security.DENIED_MESSAGE
    assert "PERMISSION DENIED BY USER" in msg


def test_broker_allow_always_updates_the_live_config(bare_broker, tmp_path):
    st = bare_broker
    cfg = Config()
    cfg.allow_fetch = False
    with st.lock:
        st.cfg = cfg
    t, out = _ask(st, category=security.CAT_WEB, detail="https://docs.example.org/x",
                  key="docs.example.org")
    pp = _wait_pending(st)
    st.decide_permission(pp["id"], "allow_always")
    t.join(5)
    assert out["value"][0] is True
    assert cfg.allow_fetch is True
    assert "docs.example.org" in tuple(cfg.fetch_allowed_hosts)
    # subsequent asks for that host are answered without pending again
    t2, out2 = _ask(st, category=security.CAT_WEB,
                    detail="https://docs.example.org/y", key="docs.example.org")
    t2.join(5)
    assert out2["value"][0] is True
    with st.lock:
        assert st.pending_permission is None


def test_broker_times_out_into_clean_denial(bare_broker, monkeypatch):
    st = bare_broker
    monkeypatch.setattr(core, "PERMISSION_TIMEOUT", 0.4)
    t, out = _ask(st, detail="sleep infinity")
    _wait_pending(st)
    t.join(10)
    assert out["value"] == (False, security.DENIED_MESSAGE)
    with st.lock:
        assert st.pending_permission is None


def test_permission_decide_endpoint_round_trip(gui_broker):
    st = gui_broker.state
    t, out = _ask(st, detail="pip install requests",
                  category=security.CAT_PACKAGE_INSTALL,
                  key="pip install requests")
    # the browser sees it in the polled state...
    end = time.time() + 5
    pp = None
    while time.time() < end and not pp:
        status, body = gui_broker.get("/api/state")
        assert status == 200
        pp = body.get("pending_permission")
        time.sleep(0.05)
    assert pp and pp["category"] == "package_install"
    # ... and answers through the documented route
    status, body = gui_broker.post("/api/permission/decide",
                                   {"id": pp["id"], "decision": "allow_once"})
    assert status == 200 and body["ok"] is True
    t.join(5)
    assert out["value"][0] is True
    # unknown decision -> 400; stale id -> 409
    status, _ = gui_broker.post("/api/permission/decide",
                                {"id": pp["id"], "decision": "yolo"})
    assert status == 400
    status, _ = gui_broker.post("/api/permission/decide",
                                {"id": "nope", "decision": "deny"})
    assert status == 409


# ------------------------------------------------- tools wired to the broker ---

def _run_via_thread(st, cfg, root, command, results):
    def work():
        idx = index_mod.Index(root, [])
        results["obs"] = tools.tool_run(idx, cfg, command)
    t = threading.Thread(target=work, daemon=True)
    t.start()
    return t


def test_tool_run_asks_then_executes_with_allow_once(bare_broker, tmp_path):
    st = bare_broker
    cfg = Config()                      # allow_shell = False (the default)
    cmd = f"{sys.executable} -c \"print('perm-approved-run')\""
    results = {}
    t = _run_via_thread(st, cfg, tmp_path, cmd, results)
    pp = _wait_pending(st)
    assert pp["category"] == "shell"
    assert cmd in pp["detail"]
    st.decide_permission(pp["id"], "allow_run")
    t.join(60)
    assert "perm-approved-run" in results["obs"]
    assert results["obs"].startswith("RUN:")
    # second identical run: cached, no new modal, still executes
    results2 = {}
    t2 = _run_via_thread(st, cfg, tmp_path, cmd, results2)
    t2.join(60)
    assert "perm-approved-run" in results2["obs"]
    with st.lock:
        assert st.pending_permission is None


def test_tool_run_deny_pivots_and_never_executes(bare_broker, tmp_path):
    st = bare_broker
    cfg = Config()
    marker = tmp_path / "did-it"
    cmd = f"{sys.executable} -c \"open({str(marker)!r},'w').write('x')\""
    results = {}
    t = _run_via_thread(st, cfg, tmp_path, cmd, results)
    pp = _wait_pending(st)
    st.decide_permission(pp["id"], "deny")
    t.join(30)
    assert results["obs"] == security.DENIED_MESSAGE
    assert not marker.exists()


def test_tool_run_pip_install_needs_its_own_approval(bare_broker, tmp_path):
    st = bare_broker
    cfg = Config()
    cfg.allow_shell = True              # shell allowed -- install still gated
    results = {}
    t = _run_via_thread(st, cfg, tmp_path, "pip install somepkg", results)
    pp = _wait_pending(st)
    assert pp["category"] == "package_install"
    st.decide_permission(pp["id"], "deny")
    t.join(30)
    assert results["obs"] == security.DENIED_MESSAGE


def test_tool_run_without_broker_keeps_legacy_errors(tmp_path):
    security.install_permission_hook(None)
    cfg = Config()
    obs = tools.tool_run(index_mod.Index(tmp_path, []), cfg, "pytest -q")
    assert obs.startswith("ERROR") and "allow_shell" in obs


def test_sensitive_read_blocked_without_broker(tmp_path):
    security.install_permission_hook(None)
    (tmp_path / ".env").write_text("API_KEY=super-secret\n")
    cfg = Config()
    cfg.allow_shell = True
    obs = tools.tool_read(index_mod.Index(tmp_path, []), cfg, ".env", 1, 0)
    assert obs.startswith("ERROR") and "sensitive" in obs
    assert "super-secret" not in obs


def test_sensitive_read_with_broker_allow_once(bare_broker, tmp_path):
    st = bare_broker
    (tmp_path / ".env").write_text("API_KEY=super-secret\n")
    cfg = Config()
    results = {}

    def work():
        results["obs"] = tools.tool_read(index_mod.Index(tmp_path, []), cfg,
                                         ".env", 1, 0)
    t = threading.Thread(target=work, daemon=True)
    t.start()
    pp = _wait_pending(st)
    assert pp["category"] == "sensitive_file" and pp["key"] == ".env"
    st.decide_permission(pp["id"], "allow_once")
    t.join(30)
    assert "super-secret" in results["obs"]


def test_fetch_disabled_asks_the_broker(bare_broker, tmp_path):
    st = bare_broker
    cfg = Config()                       # allow_fetch = False
    results = {}

    def work():
        results["obs"] = webfetch.tool_fetch(index_mod.Index(tmp_path, []), cfg,
                                             "https://docs.example.org/json")
    t = threading.Thread(target=work, daemon=True)
    t.start()
    pp = _wait_pending(st)
    assert pp["category"] == "web" and pp["key"] == "docs.example.org"
    st.decide_permission(pp["id"], "deny")
    t.join(30)
    assert results["obs"] == security.DENIED_MESSAGE
    # allow_once proceeds to the (stubbed) request
    calls = []

    class _Resp:
        url = "https://docs.example.org/json"
        status_code = 200
        headers: ClassVar = {"Content-Type": "text/plain"}

        def iter_content(self, _n):
            yield b"hello from the web"

        def close(self):
            pass

    def fake_get(url, **kw):
        calls.append(url)
        return _Resp()

    orig = webfetch.requests.get
    webfetch.requests.get = fake_get
    try:
        t2 = threading.Thread(target=work, daemon=True)
        t2.start()
        pp2 = _wait_pending(st)
        st.decide_permission(pp2["id"], "allow_once")
        t2.join(30)
        assert "EXTERNAL CONTENT from https://docs.example.org/json" in results["obs"]
        assert calls == ["https://docs.example.org/json"]
    finally:
        webfetch.requests.get = orig


def test_install_command_detection():
    yes = ["pip install requests", "pip3 -q install pkg", "python -m pip install x",
           "uv pip install ruff", "npm i left-pad", "yarn add react",
           "poetry add black", "cargo add serde", "go get example.com/mod",
           "conda install numpy"]
    no = ["pytest -q", "python app.py", "pip list", "ruff check .",
          "git status", "python -m venv .venv", "ls -la"]
    for cmd in yes:
        argv = cmd.split()
        assert security.is_install_command(argv), cmd
    for cmd in no:
        argv = cmd.split()
        assert not security.is_install_command(argv), cmd


def test_uninstall_releases_any_parked_worker(bare_broker):
    st = bare_broker
    t, out = _ask(st, detail="make the coffee")
    _wait_pending(st)
    core.uninstall_permission_broker()
    t.join(5)
    assert out["value"] == (False, security.DENIED_MESSAGE)
    core.install_permission_broker()      # the fixture teardown re-cleans


def test_gui_status_reports_no_pending_permission_by_default(gui_broker):
    status, body = gui_broker.get("/api/state")
    assert status == 200 and body["pending_permission"] is None


# ------------------------------------------------- built-in web tools (fetch_url /
#                                                  web_search, enable_web) ---

WEB_DENIED_EXACT = ("[PERMISSION DENIED BY USER: Do not retry fetching "
                    "external URLs. Proceed using local project files only.]")
FETCH_DESC = ("Fetch and read the text content of an external http/https "
              "web page (harness tool; requires user permission when "
              "enable_web is off)")
SEARCH_DESC = ("Search the web for external documentation or information "
               "(harness tool; requires user permission when enable_web is "
               "off)")


class _FakeResp:
    """Enough of http.client.HTTPResponse for tools._web_open."""

    def __init__(self, body, url, ctype="text/html; charset=utf-8", status=200):
        self._body = body if isinstance(body, bytes) else body.encode()
        self._url = url
        self.status = status
        self.headers = {"Content-Type": ctype}
        self.closed = False

    def read(self, n=-1):
        return self._body if n is None or n < 0 else self._body[:n]

    def geturl(self):
        return self._url

    def close(self):
        self.closed = True


def _serve(monkeypatch, resp_factory):
    """Patch urllib.request.urlopen; records (url, timeout, user_agent)."""
    seen = []

    def fake_urlopen(req, timeout=None, **kw):
        seen.append((req.full_url, timeout, req.get_header("User-agent")))
        return resp_factory(req.full_url)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return seen


def _run_in_thread(st, fn):
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("r", fn()), daemon=True)
    t.start()
    return t, box


def test_enable_web_defaults_off_and_ships_in_template():
    assert Config().enable_web is False
    assert "enable_web = false" in config_mod.TEMPLATE


def test_enable_web_gui_metadata_exact():
    from pedacito.gui import settings
    item = {i["name"]: i for i in settings.schema()}["enable_web"]
    assert item["label"] == "Enable Web Fetch & Search"
    assert item["help"] == (
        "Allow the agent to fetch URLs or search documentation online. "
        "When disabled, web tool attempts trigger an interactive "
        "permission prompt.")


def test_web_tools_enter_schema_only_when_enabled_or_prompt_reaches_web():
    plain = tools.schema_for(Config())
    assert "fetch_url" not in plain["properties"]["action"]["enum"]
    assert "web_search" not in plain["properties"]["action"]["enum"]
    assert "url" not in plain["properties"] and "query" not in plain["properties"]

    cfg = Config()
    cfg._task_text = "summarize https://apnews.com/world-news-council"
    s = tools.schema_for(cfg)
    assert "fetch_url" in s["properties"]["action"]["enum"]
    assert "web_search" in s["properties"]["action"]["enum"]
    assert "url" in s["properties"] and "query" in s["properties"]
    assert "url" in s["required"] and "query" in s["required"]
    keys = list(s["properties"])
    assert keys.index("symbol") < keys.index("url") < keys.index("query") \
        < keys.index("content")                                # order intact

    on = Config()
    on.enable_web = True
    assert "fetch_url" in tools.schema_for(on)["properties"]["action"]["enum"]


def test_prompt_states_exact_web_tool_descriptions():
    from pedacito import prompts
    bullets = dict(prompts._EXTRA_BULLETS)
    assert FETCH_DESC in " ".join(bullets["fetch_url"].split())
    assert SEARCH_DESC in " ".join(bullets["web_search"].split())
    cfg = Config()
    cfg.enable_web = True
    compact = " ".join(prompts._compact_tool_help(cfg).split())
    assert FETCH_DESC in compact and SEARCH_DESC in compact
    assert FETCH_DESC in " ".join(tools.TOOL_HELP.split())


def test_fetch_url_broker_denies_with_exact_redirect(bare_broker):
    st = bare_broker
    cfg = Config()
    with st.lock:
        st.cfg = cfg
    url = "https://example.com/article"
    t, box = _run_in_thread(st, lambda: tools.fetch_url(cfg, url))
    pp = _wait_pending(st)
    assert pp["category"] == "web_access"
    assert pp["action_summary"] == f"Fetch external URL: {url}"
    assert pp["detail"] == url and pp["key"] == url
    assert pp["setting_key"] == "enable_web"
    st.decide_permission(pp["id"], "deny")
    t.join(15)
    assert box["r"] == WEB_DENIED_EXACT


def test_fetch_url_allow_once_fetches_strips_scripts(bare_broker, monkeypatch):
    st = bare_broker
    cfg = Config()
    with st.lock:
        st.cfg = cfg
    html = ("<html><head><title>Example Article</title>"
            "<script>SECRET_SCRIPT_LEAK()</script></head>"
            "<body><h1>Hello World</h1><p>Body text here.</p>"
            "<pre>  keep    indent</pre></body></html>")
    seen = _serve(monkeypatch, lambda u: _FakeResp(html, u))
    url = "https://example.com/article"

    t, box = _run_in_thread(st, lambda: tools.fetch_url(cfg, url))
    pp = _wait_pending(st)
    st.decide_permission(pp["id"], "allow_once")
    t.join(15)
    out = box["r"]
    assert seen == [(url, 8, "Pedacito/1.0")]
    assert "EXTERNAL CONTENT from https://example.com/article" in out
    assert "Title: Example Article" in out
    assert "Hello World" in out and "Body text here." in out
    assert "SECRET_SCRIPT_LEAK" not in out
    assert "untrusted" in out.lower()


def test_fetch_url_allow_always_flips_enable_web(bare_broker, monkeypatch):
    st = bare_broker
    cfg = Config()
    with st.lock:
        st.cfg = cfg
    _serve(monkeypatch, lambda u: _FakeResp("<p>ok</p>", u))
    url = "https://example.com/docs"
    t, box = _run_in_thread(st, lambda: tools.fetch_url(cfg, url))
    pp = _wait_pending(st)
    st.decide_permission(pp["id"], "allow_always")
    t.join(15)
    assert cfg.enable_web is True
    # the setting itself flipped: a second fetch never parks again
    t2, box2 = _run_in_thread(st, lambda: tools.fetch_url(cfg, url))
    t2.join(15)
    assert "ok" in box2["r"]
    with st.lock:
        assert st.pending_permission is None


def test_web_search_broker_then_top_results(bare_broker, monkeypatch):
    st = bare_broker
    cfg = Config()
    with st.lock:
        st.cfg = cfg
    page = """
    <div class="result">
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.pytest.org%2Fen%2Ffixtures&amp;rut=x">Pytest fixtures</a>
      <td class="result__snippet">Explicit <b>fixtures</b> remove noise.</td>
    </div>
    <div class="result">
      <a class="result__a" href="https://example.org/guide">A guide</a>
      <td class="result__snippet">Second &amp; snippet.</td>
    </div>"""
    _serve(monkeypatch, lambda u: _FakeResp(page, u))
    q = "pytest fixture scopes"
    t, box = _run_in_thread(st, lambda: tools.web_search(cfg, q))
    pp = _wait_pending(st)
    assert pp["category"] == "web_access"
    assert pp["action_summary"] == f"Web search: {q}"
    assert pp["setting_key"] == "enable_web"
    st.decide_permission(pp["id"], "allow_once")
    t.join(15)
    out = box["r"]
    assert "WEB SEARCH RESULTS for 'pytest fixture scopes'" in out
    assert "https://docs.pytest.org/en/fixtures" in out   # uddg unwrapped
    assert "https://example.org/guide" in out
    assert "Second & snippet." in out


def test_fetch_url_ssrf_refused_before_any_prompt(bare_broker, monkeypatch):
    st = bare_broker
    cfg = Config()
    cfg.enable_web = True                     # even enabled: SSRF still holds
    with st.lock:
        st.cfg = cfg
    seen = _serve(monkeypatch, lambda u: _FakeResp("<p>nope</p>", u))
    for bad in ("http://127.0.0.1:8000/admin", "http://localhost/x",
                "http://169.254.169.254/latest/meta-data/",
                "http://[::1]/x", "http://10.0.0.5/internal",
                "ftp://example.com/file"):
        out = tools.fetch_url(cfg, bad)
        assert out.startswith("ERROR:"), bad
        assert st.status()["pending_permission"] is None
    assert seen == []
    # ... unless the user named that host in fetch_allowed_hosts
    cfg.fetch_allowed_hosts = ("localhost",)
    out = tools.fetch_url(cfg, "http://localhost:8000/devdocs")
    assert "EXTERNAL CONTENT" in out


def test_fetch_url_clips_to_max_read_tokens(bare_broker, monkeypatch):
    st = bare_broker
    cfg = Config()
    cfg.enable_web = True
    cfg.max_read_tokens = 100
    with st.lock:
        st.cfg = cfg
    body = "<p>" + ("lorem ipsum dolor sit amet " * 300) + "</p>"
    _serve(monkeypatch, lambda u: _FakeResp(body, u))
    out = tools.fetch_url(cfg, "https://example.com/long")
    assert "clipped at max_read_tokens" in out
    page = out.split("\n\n", 1)[1].split("\n[... clipped")[0]
    assert len(page) <= 400


def test_dispatch_routes_web_actions_with_field_fallback():
    cfg = Config()
    cfg.enable_web = False
    # no broker installed here: security falls back to a clean denial
    assert tools.dispatch(None, cfg, {"action": "fetch_url",
                                      "url": "https://example.com/x"}
                          ) == WEB_DENIED_EXACT
    assert tools.dispatch(None, cfg, {"action": "fetch_url",
                                      "pattern": "https://example.com/x"}
                          ) == WEB_DENIED_EXACT
    assert tools.dispatch(None, cfg, {"action": "web_search",
                                      "query": "docs"}
                          ) == WEB_DENIED_EXACT
    assert tools.dispatch(None, cfg, {"action": "web_search",
                                      "pattern": "docs"}
                          ) == WEB_DENIED_EXACT
