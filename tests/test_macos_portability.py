"""macOS CI portability guards that run deterministically on Linux.

The GitHub macOS runner fails where the Linux one cannot see it. Its first
pick was a stall: `http.server.HTTPServer.server_bind()` names itself with
`socket.getfqdn(host)`, and for the bind-all addresses a dev server likes to
use (`0.0.0.0`, `::`) that is a reverse lookup of the runner's own hostname --
which macOS answers over mDNS, slowly. "Tens of seconds" is what
`pedacito/local_server.py` documents, and Bug Hunt #12 paid for it twice: the
background-server fixtures wrote their readiness file *after* the bind, so the
test saw a running process with no port file and no log line, and CI blamed the
registry for a lookup the stdlib performed.

The same lesson applies to a child process pedacito cannot keep alive: a
registry server outlives the CLI run that started it, so the only DNS-free
server class available to it is one it builds itself out of `socket`.

Zero extra dependencies: stdlib re, pathlib, socket and pytest only.
"""
from __future__ import annotations

import re
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito.local_server import LocalHTTPServer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = (REPO_ROOT / "pedacito", REPO_ROOT / "tests")

# `HTTPServer(...)` / `ThreadingHTTPServer(...)`: a construction, not a mention.
# The lookbehind leaves `LocalHTTPServer(` alone -- that is the class to use.
_SERVER_CALL = re.compile(r"(?<![\w.])(?:Threading)?HTTPServer\s*\(")
# One class has to subclass the stdlib server for everyone else.
_ALLOWED = {(REPO_ROOT / "pedacito" / "local_server.py").resolve()}
# ...and one test has to *call* it, to prove the stall this rule is about.
_EXEMPT = "# dns-lookup-expected"


class _Quiet(BaseHTTPRequestHandler):
    def log_message(self, *args):       # keep pytest output clean
        pass


def _sources():
    for base in SCAN_DIRS:
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            yield path.resolve(), path.relative_to(REPO_ROOT).as_posix()


def test_nothing_builds_a_server_that_asks_dns_to_bind():
    """Every local test/dev server must be a `LocalHTTPServer`.

    In-process servers get it from `pedacito.local_server`; a server pedacito
    spawns and cannot supervise (the background `serve` registry) builds one
    from `socket`, because `getfqdn` stalls there just the same. The child
    scripts in this suite are strings in these files, so this one scan covers
    both shapes -- and catches the next fixture that reaches for
    `http.server` before a macOS runner does.
    """
    hits = []
    for path, label in _sources():
        if path in _ALLOWED:
            continue
        for no, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            if _EXEMPT in line:
                continue
            code = line.split("#", 1)[0]
            if _SERVER_CALL.search(code):
                hits.append(f"{label}:{no}: {code.strip()}")
    assert not hits, (
        "constructing an http.server class binds with socket.getfqdn(), a "
        "reverse-DNS lookup that stalls macOS runners:\n"
        + "\n".join(hits)
        + "\nUse pedacito.local_server.LocalHTTPServer (in-process) or a bare "
          "socket (a spawned fake).")


def test_local_server_binds_without_a_single_lookup(monkeypatch):
    """The replacement class must never ask, on either bind spelling."""
    def _no_dns(*_args, **_kwargs):
        raise AssertionError("reverse DNS lookup during bind")

    monkeypatch.setattr(socket, "getfqdn", _no_dns)
    for host in ("127.0.0.1", "0.0.0.0"):
        srv = LocalHTTPServer((host, 0), _Quiet)
        try:
            assert srv.server_port > 0
            assert srv.server_name == host      # the bind address, not a name
        finally:
            srv.server_close()


def test_the_stdlib_server_is_what_stalls(monkeypatch):
    """The guard above is not decorative: the class it bans really does ask.

    If CPython ever drops the lookup, this test says so instead of leaving the
    rule explaining itself to nobody.
    """
    def _no_dns(*_args, **_kwargs):
        raise AssertionError("reverse DNS lookup during bind")

    monkeypatch.setattr(socket, "getfqdn", _no_dns)
    with pytest.raises(AssertionError, match="reverse DNS"):
        ThreadingHTTPServer(  # dns-lookup-expected
            ("0.0.0.0", 0), _Quiet).server_close()
