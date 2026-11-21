"""Bug Hunt #12 crucible: the outbound fetch guard (SSRF + DNS rebinding).

The fetch tool is the one place a prompt-injected URL reaches the network, so
its guard has to be true at three layers: the URL as written, every address
its name resolves to, and every redirect hop. The last two are where a literal
hostname check fails.
"""
from __future__ import annotations

import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import security, tools, webfetch
from pedacito.local_server import LocalHTTPServer
from pedacito.config import Config
from pedacito.index import Index


PRIVATE_URLS = [
    "http://127.0.0.1/",
    "http://localhost/",
    "http://10.0.0.1/",
    "http://192.168.1.1/",
    "http://169.254.169.254/latest/meta-data/",
    "http://[::1]/",
    "http://0.0.0.0/",
]


def _host(url: str) -> str:
    return urlparse(url).hostname or ""


# ----------------------------------------------------------- direct refusals

@pytest.mark.parametrize("url", PRIVATE_URLS)
def test_direct_private_targets_are_refused_without_an_explicit_name(url):
    blocked = security.ssrf_error(url, ())
    assert blocked, url
    err, ips = security.resolve_checked(_host(url), ())
    assert err and ips == [], (url, err, ips)


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "ftp://example.com/x",
    "javascript:alert(1)",
    "not a url",
])
def test_non_http_schemes_and_garbage_never_reach_the_resolver(url):
    blocked = security.ssrf_error(url, ("*",))
    assert blocked, url


def test_empty_hostname_fails_closed():
    err, ips = security.resolve_checked("", ())
    assert err and ips == []
    assert security.ssrf_error("http:///nohost", ("*",))


# ------------------------------------------------------- wildcard is not a pass

@pytest.mark.parametrize("url", PRIVATE_URLS)
def test_wildcard_still_refuses_private_addresses(url):
    assert security.ssrf_error(url, ("*",)), url
    err, ips = security.resolve_checked(_host(url), ("*",))
    assert err and ips == [], (url, err, ips)


def test_wildcard_allows_a_public_ip_but_never_a_private_one():
    assert security.ssrf_error("http://93.184.216.34/x", ("*",)) == ""
    err, ips = security.resolve_checked("93.184.216.34", ("*",))
    assert err == "" and ips == ["93.184.216.34"]
    # Naming a private host explicitly IS the user's decision.
    assert security.ssrf_error("http://10.1.2.3/x", ("10.1.2.3",)) == ""
    err, ips = security.resolve_checked("10.1.2.3", ("10.1.2.3",))
    assert err == "" and ips == ["10.1.2.3"]


# ------------------------------------------------------------ DNS rebinding

def _resolver(mapping):
    def fake(host, port, *args, **kwargs):
        if host not in mapping:
            raise socket.gaierror(f"no answer for {host}")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))
                for ip in mapping[host]]
    return fake


@pytest.mark.parametrize("private_ip", ["127.0.0.1", "169.254.169.254",
                                         "10.0.0.7", "192.168.0.5"])
def test_one_private_answer_refuses_the_whole_name(monkeypatch, private_ip):
    monkeypatch.setattr(socket, "getaddrinfo", _resolver({
        "evil-rebind.example.com": ["93.184.216.34", private_ip]}))
    err, ips = security.resolve_checked("evil-rebind.example.com", ("*",))
    assert err and ips == []
    assert private_ip in err


def test_rebinding_name_is_refused_before_any_socket_is_opened(
        monkeypatch, tmp_path):
    monkeypatch.setattr(socket, "getaddrinfo", _resolver({
        "evil-rebind.example.com": ["93.184.216.34", "127.0.0.1"]}))

    def no_connect(*args, **kwargs):
        raise AssertionError("a socket was opened for a rebinding name")

    monkeypatch.setattr(socket, "create_connection", no_connect)
    monkeypatch.setattr(socket.socket, "connect", no_connect)

    cfg = Config()
    cfg.enable_web = True
    cfg.fetch_allowed_hosts = ("*",)
    out = tools.fetch_url(cfg, "http://evil-rebind.example.com/admin")
    assert out.startswith("ERROR:")
    assert "127.0.0.1" in out


def test_fully_public_resolution_is_accepted_ipv4_first(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _resolver({
        "good.example": ["2606:2800:220:1:248:1893:25c8:1946",
                         "93.184.216.34"]}))
    err, ips = security.resolve_checked("good.example", ("*",))
    assert err == ""
    assert ips[0] == "93.184.216.34"


# ------------------------------------------------------- redirect revalidation

class _Redirector(BaseHTTPRequestHandler):
    hits: list[str] = []

    def do_GET(self):
        type(self).hits.append(self.path)
        self.send_response(302)
        self.send_header("Location", "http://169.254.169.254/secret")
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture()
def redirector():
    _Redirector.hits = []
    httpd = LocalHTTPServer(("127.0.0.1", 0), _Redirector)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/start"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)


class _RebindRedirector(_Redirector):
    def do_GET(self):
        type(self).hits.append(self.path)
        self.send_response(302)
        self.send_header("Location", "http://evil-rebind.example.com/secret")
        self.end_headers()


@pytest.fixture()
def rebinding_redirector():
    _Redirector.hits = []
    httpd = LocalHTTPServer(("127.0.0.1", 0), _RebindRedirector)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/start"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)


def test_opener_disables_automatic_redirects():
    handler = tools._NoAutoRedirect()
    assert handler.redirect_request(
        None, None, 302, "Found", {}, "http://169.254.169.254/x") is None


def test_fetch_url_revalidates_each_redirect_hop(redirector):
    cfg = Config()
    cfg.enable_web = True
    # Hop 1 is named explicitly: the loopback redirector itself is allowed.
    cfg.fetch_allowed_hosts = ("127.0.0.1",)
    out = tools.fetch_url(cfg, redirector)
    assert out.startswith("ERROR:"), out
    assert "redirect" in out.lower()
    assert "169.254.169.254" in out
    assert _Redirector.hits == ["/start"], (
        "the redirect target must never be requested")


def test_webfetch_revalidates_each_redirect_hop(redirector, tmp_path, monkeypatch):
    cfg = Config()
    cfg.allow_fetch = True
    cfg.fetch_allowed_hosts = ("127.0.0.1",)
    idx = Index(tmp_path, [])
    out = webfetch.tool_fetch(idx, cfg, redirector)
    assert out.startswith("ERROR:"), out
    assert "169.254.169.254" in out
    assert _Redirector.hits == ["/start"]


def test_redirect_hop_is_re_resolved_and_blocks_a_rebinding_name(
        rebinding_redirector, monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _resolver({
        "127.0.0.1": ["127.0.0.1"],
        "evil-rebind.example.com": ["93.184.216.34", "127.0.0.1"]}))
    cfg = Config()
    cfg.enable_web = True
    cfg.fetch_allowed_hosts = ("127.0.0.1", "*")
    out = tools.fetch_url(cfg, rebinding_redirector)
    assert out.startswith("ERROR:"), out
    assert "evil-rebind.example.com" in out
    assert "127.0.0.1" in out
    assert _Redirector.hits == ["/start"]


def test_webfetch_redirect_hop_is_re_resolved_too(rebinding_redirector,
                                                  tmp_path, monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _resolver({
        "127.0.0.1": ["127.0.0.1"],
        "evil-rebind.example.com": ["127.0.0.1"]}))
    cfg = Config()
    cfg.allow_fetch = True
    cfg.fetch_allowed_hosts = ("127.0.0.1", "*")
    idx = Index(tmp_path, [])
    out = webfetch.tool_fetch(idx, cfg, rebinding_redirector)
    assert out.startswith("ERROR:"), out
    assert "evil-rebind.example.com" in out
    assert _Redirector.hits == ["/start"]


def test_explicitly_named_private_host_is_fetchable():
    """The user may still fetch a named local service; only the redirect to a
    DIFFERENT un-named private address is refused."""
    # Point the redirector at a harmless local second hop by asking for a path
    # it serves itself: use a tiny non-redirecting page instead.
    class OK(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"<html><body>local ok</body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = LocalHTTPServer(("127.0.0.1", 0), OK)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        cfg = Config()
        cfg.enable_web = True
        cfg.fetch_allowed_hosts = ("127.0.0.1",)
        out = tools.fetch_url(cfg, f"http://127.0.0.1:{httpd.server_address[1]}/")
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)
    assert "local ok" in out
    assert out.startswith("EXTERNAL CONTENT")


# ------------------------------------------- WP-03: RFC 6598 / trailing-dot / v6 wrappers

@pytest.mark.parametrize("host", [
    "100.64.0.1",
    "100.100.100.200",
    "fec0::1",
    "localhost.",
    "foo.local.",
    "foo.localhost.",
    "::ffff:127.0.0.1",
    "::ffff:169.254.169.254",
    "::ffff:100.64.0.1",
    "2002:7f00:0001::",
    "2001:0::",
])
def test_extended_private_host_spellings_and_ranges_blocked(host):
    assert security.is_private_host(host), host
    url = f"http://[{host}]/x" if ":" in host else f"http://{host}/x"
    assert security.ssrf_error(url, ("*",)), host
    err, ips = security.resolve_checked(host, ("*",))
    assert err and ips == [], (host, err, ips)


def test_webfetch_refuses_ssrf_before_asking_permission_broker(tmp_path):
    asked = []
    security.install_permission_hook(
        lambda info: asked.append(info) or (True, ""))
    try:
        cfg = Config()
        cfg.allow_fetch = False
        idx = Index(tmp_path, [])
        for bad in ("http://169.254.169.254/latest/meta-data/",
                    "http://100.100.100.200/latest/meta-data/",
                    "http://localhost./secret",
                    "http://[::ffff:127.0.0.1]/"):
            out = webfetch.tool_fetch(idx, cfg, bad)
            assert out.startswith("ERROR:"), bad
        assert asked == []
    finally:
        security.install_permission_hook(None)


def test_pinned_get_closes_session_even_on_error(monkeypatch):
    closed = []

    class FakeSession:
        def __init__(self):
            self.trust_env = True

        def mount(self, *_a, **_k):
            pass

        def get(self, *_a, **_k):
            raise webfetch.requests.exceptions.ConnectTimeout("boom")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(webfetch.requests, "Session", FakeSession)
    with pytest.raises(webfetch.requests.exceptions.ConnectTimeout):
        webfetch._pinned_get("https://example.com/", "example.com", "93.184.216.34")
    assert closed == [True]
