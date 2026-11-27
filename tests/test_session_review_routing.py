"""WP-4004 (D9): per-session review routing.

Two concurrent sessions stage reviews into independent `ReviewQueue`s; ids,
pending lists and decisions never cross. `SessionContext.cancel()` cancels only
the session's own review, the GUI routes `/api/review*` by
`(session_id, review_id)`, and the standalone loopback server registers itself
as a `review_server` resource.
"""
from __future__ import annotations

import http.client
import json
import sys
import threading
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import review, session_ctx  # noqa: E402
from pedacito.gui import core as gui_core  # noqa: E402
from pedacito.gui import server as gui_server  # noqa: E402
from pedacito.local_server import LocalHTTPServer  # noqa: E402


def _fd(rel: str, before: str, after: str) -> review.FileDiff:
    rows, added, removed, hunks = review.build_diff(before, after)
    return review.FileDiff(rel, rows, added, removed, review._sha(before),
                           after, "edit", hunks=hunks, before_text=before)


PAD = "".join(f"# pad {i}\n" for i in range(10))


def _two_file_review():
    return [
        _fd("a.py", "one = 1\n", "one = 10\n"),
        _fd("b.py", "two = 2\n", "two = 20\n"),
    ]


def _hunk_review():
    return [_fd("safe.py", "a = 1\n" + PAD + "b = 2\n",
                "a = 10\n" + PAD + "b = 20\n")]


def test_queues_isolate_ids_pending_and_decisions():
    a = session_ctx.SessionContext(session_id="ses_alpha1111")
    b = session_ctx.SessionContext(session_id="ses_beta2222")
    qa, qb = a.review_queue, b.review_queue
    assert qa is not None and qb is not None

    writes: dict[str, dict] = {}

    def writer_for(key):
        def writer(accepted):
            writes[key] = dict(accepted)
            return True, "ok"
        return writer

    ra = qa.stage(task="A", diffs=_two_file_review(),
                  writer=writer_for("a"), clause_texts={"R1": "clause"})
    rb = qb.stage(task="B", diffs=_hunk_review(),
                  writer=writer_for("b"))
    assert ra.review_id.startswith("rev_")
    assert rb.review_id.startswith("rev_")
    assert ra.review_id != rb.review_id
    assert ra.session_id == a.session_id and rb.session_id == b.session_id
    assert [r["review_id"] for r in qa.list_pending()] == [ra.review_id]
    assert [r["review_id"] for r in qb.list_pending()] == [rb.review_id]
    assert qa.get(rb.review_id) is None
    assert qb.get(ra.review_id) is None

    # Per-file selection on A.
    ok, detail = qa.decide(ra.review_id, "apply", files=["a.py"])
    assert ok and detail == "ok"
    assert set(writes["a"]) == {"a.py"}
    assert ra.outcome == "applied" and ra.done.is_set()
    assert qa.list_pending() == []
    # B is untouched.
    assert [r["review_id"] for r in qb.list_pending()] == [rb.review_id]

    # Per-hunk selection on B (hunk 0 only): only the first change applies.
    ok, detail = qb.decide(rb.review_id, "apply",
                           accepted_hunks={"safe.py": [0]})
    assert ok and detail == "ok"
    text = writes["b"]["safe.py"]
    assert "a = 10" in text and "b = 2" in text and "b = 20" not in text
    assert rb.outcome == "applied"


def test_ttl_expiry_and_cancel_all_call_discard():
    q = review.ReviewQueue("ses_ttl")
    discarded: list[str] = []

    def writer(accepted):
        return True, "ok"

    r = q.stage(task="ttl", diffs=_two_file_review(), writer=writer,
                on_discard=lambda: discarded.append("expired"), ttl_s=1)
    assert q.expire_stale(now=r.created_at + 2) == 1
    assert r.outcome == "expired" and r.done.is_set()
    assert discarded == ["expired"]
    assert not q.decide(r.review_id, "apply")[0]

    r2 = q.stage(task="cancel", diffs=_two_file_review(), writer=writer,
                 on_discard=lambda: discarded.append("cancelled"))
    assert q.cancel_all("session gone") == 1
    assert r2.outcome == "cancelled" and r2.detail == "session gone"
    assert discarded[-1] == "cancelled"
    assert q.cancel_all() == 0


def test_session_cancel_isolates_sibling_review():
    a = session_ctx.SessionContext(session_id="ses_cancel_a")
    b = session_ctx.SessionContext(session_id="ses_cancel_b")
    applied: dict[str, dict] = {}

    def writer_for(key):
        def writer(accepted):
            applied[key] = dict(accepted)
            return True, "ok"
        return writer

    ra = a.review_queue.stage(task="A", diffs=_two_file_review(),
                              writer=writer_for("a"))
    rb = b.review_queue.stage(task="B", diffs=_two_file_review(),
                              writer=writer_for("b"))

    report = a.cancel("test cancel")
    assert report["reviews_cancelled"] == 1
    assert ra.outcome == "cancelled" and ra.done.is_set()
    assert rb.outcome == "" and not rb.done.is_set()
    assert [r["review_id"] for r in b.review_queue.list_pending()] \
        == [rb.review_id]

    assert b.review_queue.decide(rb.review_id, "apply", files=["b.py"])[0]
    assert set(applied["b"]) == {"b.py"}


# --------------------------------------------------------------- GUI routing ---


def _req(port, method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request(method, path, body=json.dumps(body or {}),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data


def test_gui_routes_reviews_by_session_and_review_id():
    sid_a = "ses_gui_route_a"
    sid_b = "ses_gui_route_b"
    state_a = gui_core.REGISTRY.get_or_create(sid_a)
    state_b = gui_core.REGISTRY.get_or_create(sid_b)
    applied: dict[str, dict] = {}

    def writer(accepted):
        applied["a"] = dict(accepted)
        return True, "ok"

    ra = state_a.review_queue.stage(task="A", diffs=_two_file_review(),
                                     writer=writer)
    rb = state_b.review_queue.stage(task="B", diffs=_two_file_review(),
                                     writer=lambda accepted: (True, "ok"))

    srv = LocalHTTPServer(("127.0.0.1", 0), gui_server._Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        status, data = _req(port, "GET",
                            f"/api/review?session_id={sid_a}&review_id={ra.review_id}")
        assert status == 200
        meta = json.loads(data)["review"]
        assert meta["review_id"] == ra.review_id
        assert meta["session_id"] == sid_a
        assert [f["rel"] for f in meta["files"]] == ["a.py", "b.py"]

        # A review id from another session is not visible / is rejected.
        status, _ = _req(
            port, "GET",
            f"/api/review?session_id={sid_b}&review_id={ra.review_id}")
        assert status in (403, 404)
        status, _ = _req(
            port, "POST", "/api/review/decide",
            {"session_id": sid_b, "review_id": ra.review_id,
             "action": "apply"})
        assert status in (403, 404)
        assert applied == {}

        status, data = _req(
            port, "POST", "/api/review/decide",
            {"session_id": sid_a, "review_id": ra.review_id,
             "action": "apply", "files": ["a.py"]})
        assert status == 200 and json.loads(data)["ok"]
        assert set(applied["a"]) == {"a.py"}
        assert ra.outcome == "applied"
        assert rb.outcome == ""
    finally:
        srv.shutdown()
        srv.server_close()
        gui_core.REGISTRY.remove(sid_a)
        gui_core.REGISTRY.remove(sid_b)


# ------------------------------------------------------- standalone resource ---


def test_standalone_review_registers_and_releases_review_server():
    ctx = session_ctx.SessionContext(session_id="ses_review_server")
    out: dict = {}
    ready = threading.Event()

    def worker():
        with session_ctx.use_session(ctx):
            out["res"] = review.run(
                "<html>page</html>", "tok", lambda: (True, "ok"),
                open_browser=False, timeout=10,
                on_url=lambda _url: ready.set())

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    assert ready.wait(5)
    kinds = [r["kind"] for r in ctx.resource_registry.list_resources()]
    assert "review_server" in kinds

    ctx.cancel("test teardown")
    thread.join(10)
    assert not thread.is_alive()
    assert out["res"][0] == "cancelled"
    assert [r["kind"] for r in ctx.resource_registry.list_resources()] == []
