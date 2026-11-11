"""Loopback review/GUI servers: request guards and the apply/cancel flow."""
import http.client
import json
import threading

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
            _req(port, "POST", "/cancel")
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
        assert _req(port, "POST", "/cancel")[0] == 200
    finally:
        t.join(10)
    assert result["out"][0] == "cancelled"


def test_write_files_keeps_first_backup_on_repeat_apply(tmp_path):
    (tmp_path / "a.txt").write_text("one\n")
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
    assert (bak / "a.txt").read_text() == "one\n"
    assert (tmp_path / "a.txt").read_text() == "two\n"


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


def test_review_build_writer_returns_ok_tuple(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from pedacito import scaffold
    (tmp_path / "bak").mkdir()
    (tmp_path / "bak" / "a.py").write_text("x = 1\n")
    (tmp_path / "a.py").write_text("x = 2\n")
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
