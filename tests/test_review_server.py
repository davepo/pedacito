"""Loopback review/GUI servers: request guards and the apply/cancel flow."""
import http.client
import io
import json
import socket
import threading
import time

import pytest

from pedacito import review


def _serve(writer, on_discard=None):
    result = {}
    urls = []
    ready = threading.Event()

    def on_url(u):
        urls.append(u)
        ready.set()

    t = threading.Thread(target=lambda: result.update(
        out=review.run("<html>page</html>", "tok", writer, open_browser=False,
                       timeout=10, on_url=on_url, on_discard=on_discard)))
    t.start()
    assert ready.wait(5)
    port = int(urls[0].rsplit(":", 1)[1].rstrip("/"))
    return t, result, port


def _req(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    conn.request(method, path, body=json.dumps(body or {}), headers=hdrs)
    r = conn.getresponse()
    data = r.read()
    conn.close()
    return r.status, data


def _raw_http(port, payload, timeout=6.0, half_close=False):
    """Send raw bytes and read until the server closes; (reply, elapsed)."""
    started = time.monotonic()
    sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    chunks = []
    try:
        sock.sendall(payload)
        if half_close:
            sock.shutdown(socket.SHUT_WR)
        while True:
            try:
                chunk = sock.recv(4096)
            except (TimeoutError, OSError):
                break
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        sock.close()
    return b"".join(chunks), time.monotonic() - started


def _status_line(reply: bytes) -> bytes:
    return reply.split(b"\r\n", 1)[0]


class _FakeBodyHandler:
    """Minimal handler surface for the shared body helpers."""

    def __init__(self, headers, body=b""):
        self.headers = headers
        self.rfile = io.BytesIO(body)
        self.close_connection = False


def test_request_body_helpers_fail_closed_on_bad_lengths_and_cap_refusals():
    partial = _FakeBodyHandler({"Content-Length": "100"}, b"{}")
    with pytest.raises(review.RequestBodyError) as exc:
        review.read_request_body(partial, timeout=0.2)
    assert exc.value.status == 400
    assert partial.close_connection

    for bad in ("-1", "abc", "", "+5", "1.5", "0x10"):
        handler = _FakeBodyHandler({"Content-Length": bad}, b"{}")
        with pytest.raises(review.RequestBodyError) as exc:
            review.read_request_body(handler, timeout=0.2)
        assert exc.value.status == 400, bad
        assert handler.close_connection, bad
        assert handler.rfile.tell() == 0, bad      # never read a bad body

    # A refusal is capped: it consumes only the explicit limit and reports
    # that the body was not fully drained.
    capped = _FakeBodyHandler({"Content-Length": "4096"}, b"x" * 4096)
    assert review.discard_request_body(capped, limit=1024, timeout=0.2) is False
    assert capped.rfile.tell() == 1024
    assert capped.close_connection

    # The normal reader accepts a body at its limit and returns it whole.
    whole = _FakeBodyHandler({"Content-Length": "2"}, b"{}")
    assert review.read_request_body(whole, limit=2, timeout=0.2) == b"{}"
    assert not whole.close_connection


def test_review_server_rejects_foreign_host_and_cross_site_post():
    t, result, port = _serve(lambda: (True, "ok"))
    applied = False
    try:
        assert _req(port, "GET", "/")[0] == 200
        assert _req(port, "GET", "/", headers={"Host": "evil.example"})[0] == 403
        status, _ = _req(port, "POST", "/apply", {"nonce": "tok"},
                         headers={"Content-Type": "text/plain"})
        assert status == 403
        status, _ = _req(port, "POST", "/apply", {"nonce": "tok"},
                         headers={"Origin": "http://evil.example"})
        assert status == 403
        assert _req(port, "POST", "/apply", {"nonce": "wrong"})[0] == 403
        assert t.is_alive()
        status, data = _req(port, "POST", "/apply", {"nonce": "tok"})
        applied = True
        assert status == 200 and json.loads(data)["ok"]
    finally:
        if not applied:
            _req(port, "POST", "/cancel", {"nonce": "tok"})
        t.join(10)
    assert result["out"] == ("applied", "ok")


def test_review_server_survives_non_object_body():
    t, result, port = _serve(lambda: (True, "ok"))
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", "/apply", body=b"[1, 2]",
                     headers={"Content-Type": "application/json"})
        assert conn.getresponse().status == 403
        conn.close()
        assert _req(port, "POST", "/cancel", {"nonce": "wrong"})[0] == 403
        assert t.is_alive()
        assert _req(port, "POST", "/cancel", {"nonce": "tok"})[0] == 200
    finally:
        t.join(10)
    assert result["out"][0] == "cancelled"


def test_review_server_requires_cancel_nonce_sets_headers_and_oneshot():
    from pedacito.local_server import LocalHTTPServer

    t, result, port = _serve(lambda: (True, "ok"))
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/")
        resp = conn.getresponse()
        assert resp.status == 200
        hdrs = {k.lower(): v for k, v in resp.getheaders()}
        resp.read()
        conn.close()
        assert hdrs.get("x-frame-options") == "DENY"
        assert hdrs.get("x-content-type-options") == "nosniff"

        # /cancel without nonce is refused (403) and server stays alive
        assert _req(port, "POST", "/cancel", {})[0] == 403
        assert _req(port, "POST", "/cancel", {"nonce": "bad"})[0] == 403
        assert t.is_alive()

        # Valid /cancel decides once
        assert _req(port, "POST", "/cancel", {"nonce": "tok"})[0] == 200
    finally:
        t.join(10)
    assert result["out"][0] == "cancelled"

    # One-shot guard & serialization: once decided, further /apply or /cancel gets 409
    calls = []
    entered = threading.Event()
    release = threading.Event()

    def slow_writer():
        calls.append(1)
        entered.set()
        release.wait(5)
        return True, "ok"

    srv = LocalHTTPServer(("127.0.0.1", 0), review._Handler)
    srv.page = "<html></html>"
    srv.nonce = "tok"
    srv.writer = slow_writer
    srv.on_discard = None
    srv.lock = threading.Lock()
    srv.done = threading.Event()
    srv.outcome = ("timeout", "")
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        p2 = srv.server_address[1]
        r1 = {}
        t1 = threading.Thread(
            target=lambda: r1.update(
                res=_req(p2, "POST", "/apply", {"nonce": "tok"})))
        t1.start()
        assert entered.wait(5)
        r2 = {}
        t2 = threading.Thread(
            target=lambda: r2.update(
                res=_req(p2, "POST", "/cancel", {"nonce": "tok"})))
        t2.start()
        release.set()
        t1.join(5)
        t2.join(5)
        assert r1["res"][0] == 200
        assert r2["res"][0] == 409
        assert _req(p2, "POST", "/apply", {"nonce": "tok"})[0] == 409
        assert len(calls) == 1
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(5)

    page = review.build_page([], "t", "m", "p", "tok123", True, "s")
    assert "post('/cancel', { nonce: window.NONCE })" in page


def test_review_write_files_and_diffs_from_backup_block_escape_and_symlinks(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")

    # Traversal target in FileDiff is refused before any write
    bad_diff = review.FileDiff("../outside.txt", [], 1, 0, "", "pwn\n", "new")
    ok, msg = review.write_files(root, [bad_diff], None)
    assert not ok and "unsafe path" in msg.lower()
    assert outside.read_text(encoding="utf-8") == "secret\n"

    # Symlink target inside root pointing outside root is refused
    link = root / "escape.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    sym_diff = review.FileDiff("escape.txt", [], 1, 0, "", "pwn\n", "edit")
    ok, msg = review.write_files(root, [sym_diff], None)
    assert not ok and "unsafe path" in msg.lower()
    assert outside.read_text(encoding="utf-8") == "secret\n"

    # diffs_from_backup skips symlinks in backup_dir and root
    bak = tmp_path / "bak"
    bak.mkdir()
    (bak / "link_in_bak.txt").symlink_to(outside)
    (bak / "real.txt").write_text("old\n", encoding="utf-8")
    (root / "real.txt").write_text("new\n", encoding="utf-8")
    diffs = review.diffs_from_backup(root, bak)
    assert [d.rel for d in diffs] == ["real.txt"]


def test_write_files_keeps_first_backup_on_repeat_apply(tmp_path):
    (tmp_path / "a.txt").write_text("one\n", encoding="utf-8")
    from pedacito import edits
    reply = ("--- FILE: a.txt\n<<<<<<< SEARCH\none\n=======\ntwo\n"
             ">>>>>>> REPLACE\n")
    results = edits.apply_edits(tmp_path, edits.parse_edits(reply),
                                backup=False, dry_run=True)
    diffs = review.diffs_from_results(results, root=tmp_path)
    bak = tmp_path / "bak"
    bak.mkdir()
    assert review.write_files(tmp_path, diffs, bak)[0]
    assert review.write_files(tmp_path, diffs, bak)[0]
    assert (bak / "a.txt").read_text(encoding="utf-8") == "one\n"
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "two\n"


def test_gui_handler_guards_and_stale_review_ids():
    from pedacito.gui import core, server
    from pedacito.local_server import LocalHTTPServer
    srv = LocalHTTPServer(("127.0.0.1", 0), server._Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        assert _req(port, "GET", "/api/state")[0] == 200
        assert _req(port, "GET", "/api/state",
                    headers={"Host": "evil.example"})[0] == 403
        assert _req(port, "POST", "/api/run", {"mode": "task", "text": "x"},
                    headers={"Content-Type": "text/plain"})[0] == 403
        core.STATE.review = None
        assert _req(port, "POST", "/api/review/decide",
                    {"id": 0, "action": "apply"})[0] == 409
    finally:
        srv.shutdown()
        srv.server_close()


def test_review_server_drains_get_body_and_rejection_paths():
    """Every refusal (and a GET that carries a body) is answered only after the
    body is consumed: Windows otherwise resets the connection (10053) and the
    client never sees the 403/404."""
    t, result, port = _serve(lambda: (True, "ok"))
    try:
        assert _req(port, "GET", "/", {"unused": True})[0] == 200
        assert _req(port, "GET", "/", {"unused": True},
                    headers={"Host": "evil.example"})[0] == 403
        assert _req(port, "GET", "/missing", {"unused": True})[0] == 404
        assert _req(port, "POST", "/missing", {"nonce": "tok"})[0] == 404
        assert _req(port, "POST", "/apply", {"nonce": "tok"},
                    headers={"Content-Type": "text/plain"})[0] == 403
        assert _req(port, "POST", "/apply", {"nonce": "tok"},
                    headers={"Origin": "http://evil.example"})[0] == 403
        assert t.is_alive()
        assert _req(port, "POST", "/cancel", {"nonce": "tok"})[0] == 200
    finally:
        t.join(10)
    assert result["out"][0] == "cancelled"


def test_review_server_bounds_incomplete_and_oversized_bodies(monkeypatch):
    monkeypatch.setattr(review, "MAX_JSON_BODY_BYTES", 64 * 1024)
    t, result, port = _serve(lambda: (True, "ok"))
    host = f"127.0.0.1:{port}"

    def post(headers: bytes, body: bytes = b"",
             host_header: str | None = None) -> tuple[bytes, float]:
        payload = (b"POST /cancel HTTP/1.1\r\nHost: "
                   + (host_header or host).encode()
                   + b"\r\nContent-Type: application/json\r\n" + headers
                   + b"\r\n" + body)
        return _raw_http(port, payload)

    try:
        # An incomplete body is refused (400) at the deadline, not parsed as
        # an empty/partial request.
        reply, elapsed = post(b"Content-Length: 400\r\n", b'{"nonce":')
        assert b"400" in _status_line(reply), reply
        assert elapsed < 4.0, elapsed

        # A body just over the JSON cap is drained within the explicit cap and
        # deadline, and gets the real 413.
        cap = review.MAX_JSON_BODY_BYTES
        big = f"Content-Length: {cap + 17}\r\n".encode()
        reply, elapsed = post(big, b"x" * (cap + 17))
        assert b"413" in _status_line(reply), reply
        assert elapsed < 3.0, elapsed

        # Expect: 100-continue on an accepted route is granted, then the
        # missing body is still bounded and refused.
        reply, elapsed = post(b"Content-Length: 100\r\n"
                              b"Expect: 100-continue\r\n")
        assert b"100 Continue" in reply and b"400" in reply, reply
        assert elapsed < 2.0, elapsed

        # A rejected request must not be invited to send its body.
        reply, elapsed = post(b"Content-Length: 100\r\n"
                              b"Expect: 100-continue\r\n",
                              host_header="evil.example")
        assert b"403" in _status_line(reply), reply
        assert b"100 Continue" not in reply, reply
        assert elapsed < 2.0, elapsed

        assert t.is_alive()
        assert _req(port, "POST", "/cancel", {"nonce": "tok"})[0] == 200
    finally:
        t.join(10)
    assert result["out"][0] == "cancelled"


def test_review_server_rejects_mismatched_and_invalid_lengths_without_mutation():
    calls = []

    def writer():
        calls.append(1)
        return True, "ok"

    t, result, port = _serve(writer)
    host = f"127.0.0.1:{port}"

    def post(declared: bytes, body: bytes = b"") -> bytes:
        payload = (b"POST /apply HTTP/1.1\r\nHost: " + host.encode()
                   + b"\r\nContent-Type: application/json\r\nContent-Length: "
                   + declared + b"\r\n\r\n" + body)
        reply, _elapsed = _raw_http(port, payload, half_close=True)
        return reply

    try:
        # Fully valid JSON, but the declared length says more follows. The old
        # reader returned this prefix on EOF and applied the review anyway.
        body = b'{"nonce":"tok"}'
        declared = str(len(body) + 50).encode()
        assert b"400" in _status_line(post(declared, body))
        assert calls == [] and "out" not in result

        for bad in (b"-1", b"12abc", b"", b"+5", b"1.5"):
            assert b"400" in _status_line(post(bad)), bad
            assert calls == [] and "out" not in result, bad

        # The server is still usable and a complete request applies once.
        assert _req(port, "POST", "/apply", {"nonce": "tok"})[0] == 200
    finally:
        t.join(10)
    assert result["out"] == ("applied", "ok")
    assert calls == [1]


def test_review_server_survives_a_client_disconnect_mid_body():
    t, result, port = _serve(lambda: (True, "ok"))
    try:
        sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        sock.sendall(
            f"POST /cancel HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
            "Content-Type: application/json\r\nContent-Length: 200\r\n\r\n"
            '{"nonce":'.encode())
        sock.close()                    # vanish before the body completes
        time.sleep(0.1)
        assert _req(port, "POST", "/cancel", {"nonce": "tok"})[0] == 200
    finally:
        t.join(10)
    assert result["out"][0] == "cancelled"


def test_gui_handler_drains_get_body_and_rejection_paths():
    from pedacito.gui import server
    from pedacito.local_server import LocalHTTPServer
    srv = LocalHTTPServer(("127.0.0.1", 0), server._Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        assert _req(port, "GET", "/api/state", {"unused": True})[0] == 200
        assert _req(port, "GET", "/api/state", {"unused": True},
                    headers={"Host": "evil.example"})[0] == 403
        assert _req(port, "GET", "/api/nope", {"unused": True})[0] == 404
        assert _req(port, "POST", "/api/nope", {"unused": True})[0] == 404
        assert _req(port, "POST", "/api/run", {"mode": "task"},
                    headers={"Content-Type": "text/plain"})[0] == 403
        assert _req(port, "GET", "/api/state")[0] == 200
    finally:
        srv.shutdown()
        srv.server_close()


def test_gui_handler_bounds_incomplete_and_oversized_bodies(monkeypatch):
    from pedacito.gui import server
    from pedacito.local_server import LocalHTTPServer
    monkeypatch.setattr(server, "MAX_BODY", 64 * 1024)
    srv = LocalHTTPServer(("127.0.0.1", 0), server._Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    host = f"127.0.0.1:{port}"
    try:
        # A GET body shorter than its declared length is refused, not treated
        # as an empty successful GET.
        reply, elapsed = _raw_http(
            port,
            (f"GET /api/state HTTP/1.1\r\nHost: {host}\r\n"
             "Content-Length: 100\r\n\r\n{").encode())
        assert b"400" in _status_line(reply), reply
        assert elapsed < 4.0, elapsed

        # POST over MAX_BODY is drained within the cap/deadline and gets 413.
        cap = server.MAX_BODY
        body = b"x" * (cap + 17)
        reply, elapsed = _raw_http(
            port,
            (f"POST /api/run HTTP/1.1\r\nHost: {host}\r\n"
             "Content-Type: application/json\r\n"
             f"Content-Length: {len(body)}\r\n\r\n").encode() + body)
        assert b"413" in _status_line(reply), reply
        assert elapsed < 3.0, elapsed

        # Malformed/incomplete JSON on a strict route is a 400, not a hang.
        reply, elapsed = _raw_http(
            port,
            (f"POST /api/review/decide HTTP/1.1\r\nHost: {host}\r\n"
             "Content-Type: application/json\r\n"
             "Content-Length: 200\r\n\r\n{\"id\"").encode())
        assert b"400" in _status_line(reply), reply
        assert elapsed < 4.0, elapsed

        # Expect: 100-continue on an accepted route is granted; the missing
        # body is still bounded and refused.
        reply, elapsed = _raw_http(
            port,
            (f"POST /api/run HTTP/1.1\r\nHost: {host}\r\n"
             "Content-Type: application/json\r\n"
             "Expect: 100-continue\r\nContent-Length: 100\r\n\r\n").encode())
        assert b"100 Continue" in reply and b"400" in reply, reply
        assert elapsed < 2.0, elapsed

        # A rejected (foreign Host) Expect request gets no 100 Continue.
        reply, elapsed = _raw_http(
            port,
            (b"POST /api/run HTTP/1.1\r\nHost: evil.example\r\n"
             b"Content-Type: application/json\r\n"
             b"Expect: 100-continue\r\nContent-Length: 100\r\n\r\n"))
        assert b"403" in _status_line(reply), reply
        assert b"100 Continue" not in reply, reply
        assert elapsed < 2.0, elapsed

        assert _req(port, "GET", "/api/state")[0] == 200
    finally:
        srv.shutdown()
        srv.server_close()


def test_gui_accepts_a_large_complete_body_without_truncation(monkeypatch):
    from pedacito.gui import server
    from pedacito.local_server import LocalHTTPServer
    route = "/api/__test_body_echo"
    monkeypatch.setitem(
        server.POST_ROUTES, route,
        lambda q, body: (200, {"ok": True, "n": len(body.get("payload", ""))}))
    srv = LocalHTTPServer(("127.0.0.1", 0), server._Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    host = f"127.0.0.1:{port}"
    try:
        size = 4 * 1024 * 1024
        body = json.dumps({"payload": "x" * size}).encode()
        reply, elapsed = _raw_http(
            port,
            (f"POST {route} HTTP/1.1\r\nHost: {host}\r\n"
             "Content-Type: application/json\r\n"
             f"Content-Length: {len(body)}\r\n\r\n").encode() + body,
            timeout=20)
        assert b"200" in _status_line(reply), reply[:200]
        assert f'"n": {size}'.encode() in reply
        assert elapsed < 10.0, elapsed

        # The documented 60 MiB GUI body gets a practical, finite deadline.
        assert review._body_read_timeout(server.MAX_BODY) >= 8.0
        assert review._body_read_timeout(server.MAX_BODY) <= \
            review.BODY_READ_MAX_TIMEOUT
    finally:
        srv.shutdown()
        srv.server_close()


def test_gui_handler_survives_a_client_disconnect_mid_body():
    from pedacito.gui import server
    from pedacito.local_server import LocalHTTPServer
    srv = LocalHTTPServer(("127.0.0.1", 0), server._Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        sock.sendall(
            f"GET /api/state HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
            "Content-Length: 200\r\n\r\n{".encode())
        sock.close()
        time.sleep(0.1)
        assert _req(port, "GET", "/api/state")[0] == 200
    finally:
        srv.shutdown()
        srv.server_close()


def test_review_build_writer_returns_ok_tuple(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from pedacito import scaffold
    (tmp_path / "bak").mkdir()
    (tmp_path / "bak" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")
    cfg = SimpleNamespace(model="m", active_profile="", review_port=0,
                          open_browser=False, review_timeout=1)
    seen = {}

    def fake_run(page, nonce, writer, **kw):
        seen["res"] = writer()
        return "applied", "kept"

    monkeypatch.setattr(review, "run", fake_run)
    plan = scaffold.Plan("p", "s")
    out = scaffold._review_build(tmp_path, plan, {"done": ["a.py"]}, cfg,
                                 tmp_path / "bak", "stamp")
    assert out == []
    ok, _detail = seen["res"]
    assert ok


def test_empty_partial_selection_keeps_the_review_applicable():
    """An apply with no files selected must be refused without touching the
    staged diffs: truncating them first left the review permanently
    un-approvable (every retry saw an empty selection)."""
    from types import SimpleNamespace

    from pedacito.gui import core

    st = core.State()                       # no root: fully in-memory
    writes = []

    def writer():
        writes.append(1)
        return True, "ok"

    diffs = [SimpleNamespace(rel="a.py", removed_source=False, after_text="x\n"),
             SimpleNamespace(rel="b.py", removed_source=False, after_text="y\n"),
             SimpleNamespace(rel="old.py", removed_source=True, after_text="")]
    rv = core.Review(1, "task", diffs, writer, partial_ok=True)
    st.review = rv

    ok, detail = st.decide_review(1, "apply", [])
    assert not ok
    assert detail == "Select at least one file to accept."
    assert [d.rel for d in rv.diffs] == ["a.py", "b.py", "old.py"]
    assert rv.outcome == "" and writes == []

    ok, detail = st.decide_review(1, "apply", ["a.py"])
    assert ok and detail == "ok"
    assert [d.rel for d in rv.diffs] == ["a.py", "old.py"]
    assert rv.outcome == "applied" and writes == [1]
