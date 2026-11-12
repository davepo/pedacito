"""R8: review hunk provenance. Each proposed file carries the task clause(s)
its edits serve, and a partial accept flags any clause left unaccepted.

The unit tests are deterministic (no model); the end-to-end test drives the
fake OpenAI-compatible server with a two-clause task and asserts the review
rows carry clause ids and accepting one file leaves the other clause flagged.
"""
import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import config as config_mod  # noqa: E402
from pedacito import edits, review, verify  # noqa: E402
from pedacito.gui import core, server, settings_api  # noqa: E402

FAKE = Path(__file__).resolve().parent / "fake_llm.py"

TASK = ("Clamp the discount in apply_discount in pricing.py. "
        "Add express handling to compute_shipping in shipping.py.")

PRICING = "def apply_discount(price, pct):\n    return price * (1 - pct)\n"
SHIPPING = "def compute_shipping(weight):\n    return weight * 2\n"

REPLY = (
    "Clamp the discount and add express handling.\n\n"
    "--- FILE: pricing.py\n"
    "<<<<<<< SEARCH\n"
    "def apply_discount(price, pct):\n"
    "    return price * (1 - pct)\n"
    "=======\n"
    "def apply_discount(price, pct):\n"
    "    pct = max(0, min(pct, 1))\n"
    "    return price * (1 - pct)\n"
    ">>>>>>> REPLACE\n\n"
    "--- FILE: shipping.py\n"
    "<<<<<<< SEARCH\n"
    "def compute_shipping(weight):\n"
    "    return weight * 2\n"
    "=======\n"
    "def compute_shipping(weight, express=False):\n"
    "    base = weight * 2\n"
    "    return base + 5 if express else base\n"
    ">>>>>>> REPLACE\n"
)


# --- deterministic unit tests -------------------------------------------------

def _tracker_with_edits():
    t = verify.RequirementTracker(TASK)
    t.observe_edit([{"file": "pricing.py", "search": "def apply_discount",
                     "replace": "def apply_discount", "ok": True}])
    t.observe_edit([{"file": "shipping.py", "search": "def compute_shipping",
                     "replace": "def compute_shipping", "ok": True}])
    return t


def test_requirements_carry_stable_clause_ids():
    t = verify.RequirementTracker(TASK)
    assert [r.req_id for r in t] == ["R1", "R2"]
    assert all(r.is_edit for r in t)


def test_clause_map_and_diffs_carry_clause_ids(tmp_path):
    (tmp_path / "pricing.py").write_text(PRICING)
    (tmp_path / "shipping.py").write_text(SHIPPING)
    results = edits.apply_edits(tmp_path, edits.parse_edits(REPLY),
                                backup=False, dry_run=True)
    assert all(r.ok for r in results)
    t = _tracker_with_edits()
    cmap = t.clause_map()
    assert cmap == {"pricing.py": ["R1"], "shipping.py": ["R2"]}
    diffs = review.diffs_from_results(results, root=tmp_path, clauses=cmap)
    by_rel = {d.rel: d for d in diffs}
    assert by_rel["pricing.py"].clauses == ("R1",)
    assert by_rel["shipping.py"].clauses == ("R2",)
    # without a clause map the field stays empty (backwards compatible)
    plain = review.diffs_from_results(results, root=tmp_path)
    assert all(d.clauses == () for d in plain)


def test_build_page_renders_clause_badges(tmp_path):
    (tmp_path / "pricing.py").write_text(PRICING)
    (tmp_path / "shipping.py").write_text(SHIPPING)
    results = edits.apply_edits(tmp_path, edits.parse_edits(REPLY),
                                backup=False, dry_run=True)
    t = _tracker_with_edits()
    diffs = review.diffs_from_results(results, root=tmp_path,
                                      clauses=t.clause_map())
    page = review.build_page(diffs, TASK, "m", "", "nonce",
                             interactive=True, stamp="s",
                             clause_texts=t.clause_texts())
    assert 'class="clause"' in page
    assert ">R1</span>" in page and ">R2</span>" in page
    # the badge is titled with the clause text so a reviewer sees the intent
    assert "Clamp the discount in apply_discount" in page


def test_partial_accept_flags_the_unaccepted_clause():
    st = core.State()                       # no root: fully in-memory
    writes = []

    def writer():
        writes.append(1)
        return True, "ok"

    diffs = [review.FileDiff("pricing.py", [], 1, 0, "sha", "x\n", "m",
                             clauses=("R1",)),
             review.FileDiff("shipping.py", [], 1, 0, "sha", "y\n", "m",
                             clauses=("R2",))]
    rv = core.Review(1, TASK, diffs, writer, partial_ok=True,
                     clause_texts=_tracker_with_edits().clause_texts())
    st.review = rv

    ok, detail = st.decide_review(1, "apply", ["pricing.py"])
    assert ok and writes == [1]
    assert rv.unaccepted == ["R2"]
    assert "R2" in detail
    meta = st.review_meta()
    assert meta["unaccepted"] == ["R2"]
    assert meta["clause_texts"]["R2"].startswith("Add express handling")
    by_rel = {f["rel"]: f for f in meta["files"]}
    assert by_rel["pricing.py"]["clauses"] == ["R1"]
    # a chat line records the clause left open
    assert any(m.get("kind") == "review_clauses_unaccepted"
               for m in st.messages)


def test_full_accept_flags_no_clause():
    st = core.State()
    diffs = [review.FileDiff("pricing.py", [], 1, 0, "sha", "x\n", "m",
                             clauses=("R1",)),
             review.FileDiff("shipping.py", [], 1, 0, "sha", "y\n", "m",
                             clauses=("R2",))]
    rv = core.Review(1, TASK, diffs, lambda: (True, "ok"), partial_ok=True)
    st.review = rv
    ok, _detail = st.decide_review(1, "apply", ["pricing.py", "shipping.py"])
    assert ok and rv.unaccepted == []


# --- end to end with the fake server ------------------------------------------

def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Client:
    def __init__(self, port):
        self.port = port

    def req(self, method, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        conn.request(method, path,
                     body=None if body is None and method == "GET"
                     else json.dumps(body or {}),
                     headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        data = r.read()
        conn.close()
        return r.status, (json.loads(data) if data[:1] in (b"{", b"[") else data)

    def get(self, path):
        return self.req("GET", path)

    def post(self, path, body=None):
        return self.req("POST", path, body or {})

    def wait_idle(self, timeout=30):
        end = time.time() + timeout
        while time.time() < end:
            if not self.get("/api/state")[1]["job"]:
                return True
            time.sleep(0.1)
        return False

    def wait_for(self, fn, timeout=30):
        end = time.time() + timeout
        while time.time() < end:
            got = fn(self.get("/api/state")[1])
            if got:
                return got
            time.sleep(0.1)
        return None


@pytest.fixture
def gui(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg" / "config.toml"
    monkeypatch.setattr(config_mod, "USER_CONFIG", cfg_path)
    monkeypatch.setattr(core, "USER_CONFIG", cfg_path)
    monkeypatch.setattr(settings_api, "USER_CONFIG", cfg_path)
    monkeypatch.setattr(core, "RECENT_FILE", tmp_path / "cfg" / "gui_recent.json")
    for var in [k for k in os.environ if k.startswith("PEDACITO_")]:
        monkeypatch.delenv(var)
    core.STATE.__init__()
    srv = server.create_server(0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    core.install_runner()
    c = Client(srv.server_address[1])
    c.tmp = tmp_path
    yield c
    core.STATE.bus.close()
    srv.shutdown()
    srv.server_close()
    core.uninstall_runner()
    job = core.STATE.job
    if job is not None and job.thread is not None:
        job.thread.join(5)


@pytest.fixture
def fake_port():
    port = _free_port()
    actions = [
        {"action": "read", "file": "pricing.py", "start": 1, "end": 200},
        {"action": "read", "file": "shipping.py", "start": 1, "end": 200},
        {"action": "done"},
    ]
    env = dict(os.environ, FAKE_PORT=str(port), FAKE_REPLY=REPLY,
               FAKE_ACTIONS=json.dumps(actions))
    proc = subprocess.Popen([sys.executable, str(FAKE)], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError("fake server did not start")
        yield port
    finally:
        proc.terminate()
        proc.wait(timeout=5)


@pytest.mark.network
def test_two_clause_review_carries_ids_and_partial_accept_flags(gui, fake_port):
    gui.post("/api/settings/setup",
             {"base_url": f"http://127.0.0.1:{fake_port}/v1",
              "model": "fake-model", "vram_gb": "8"})
    root = gui.tmp / "proj"
    root.mkdir()
    (root / "pricing.py").write_text(PRICING)
    (root / "shipping.py").write_text(SHIPPING)
    assert gui.post("/api/project", {"path": str(root), "profile": ""})[0] == 202
    assert gui.wait_idle(30)
    assert gui.post("/api/index", {"summarise": False})[0] == 202
    assert gui.wait_idle(30)

    originals = core.install_tee()
    try:
        status, _ = gui.post("/api/run", {"mode": "task", "text": TASK,
                                          "write": "review", "plan": "never"})
        assert status == 202
        state = gui.wait_for(
            lambda s: s if s["review"] and s["review"]["pending"] else None, 30)
        assert state, core.STATE.messages
        # nothing written yet
        assert (root / "pricing.py").read_text() == PRICING
        assert (root / "shipping.py").read_text() == SHIPPING

        meta = gui.get("/api/review")[1]["review"]
        by_rel = {f["rel"]: f for f in meta["files"]}
        assert set(by_rel) == {"pricing.py", "shipping.py"}
        # each review row carries the clause id it serves
        assert by_rel["pricing.py"]["clauses"] == ["R1"]
        assert by_rel["shipping.py"]["clauses"] == ["R2"]
        assert meta["clause_texts"]["R1"].startswith("Clamp the discount")
        assert meta["clause_texts"]["R2"].startswith("Add express handling")

        # accept only pricing.py: shipping.py's clause stays flagged
        status, body = gui.post("/api/review/decide",
                                {"id": meta["id"], "action": "apply",
                                 "files": ["pricing.py"]})
        assert status == 200 and body["ok"]
        assert gui.wait_idle(30)
        assert "pct = max(0, min(pct, 1))" in (root / "pricing.py").read_text()
        assert (root / "shipping.py").read_text() == SHIPPING

        after = gui.get("/api/review")[1]["review"]
        assert after["outcome"] == "applied"
        assert after["unaccepted"] == ["R2"]
        assert any(m.get("kind") == "review_clauses_unaccepted"
                   for m in gui.get("/api/state")[1]["messages"])
    finally:
        core.uninstall_tee(originals)