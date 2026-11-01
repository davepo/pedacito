"""The browser GUI's backend: request guards, the event stream, terminal
capture, settings endpoints, job control, and an end-to-end task review against
the fake OpenAI-compatible server (no model or GPU needed)."""
import http.client
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import config as config_mod  # noqa: E402
from pedacito import review as review_mod  # noqa: E402
from pedacito.gui import (  # noqa: E402
    core,
    jobs,
    server,
    settings_api,  # noqa: E402
)

FAKE = Path(__file__).resolve().parent / "fake_llm.py"

EDIT_REPLY = (
    "Make greet return the text.\n\n"
    "--- FILE: app.py\n"
    "<<<<<<< SEARCH\n"
    "def greet(name):\n"
    "    print(\"hello\", name)\n"
    "=======\n"
    "def greet(name):\n"
    "    return f\"hello {name}\"\n"
    ">>>>>>> REPLACE\n"
)
APP = 'def greet(name):\n    print("hello", name)\n\n\ngreet("x")\n'


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Client:
    def __init__(self, port):
        self.port = port

    def req(self, method, path, body=None, headers=None, raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        hdrs = {"Content-Type": "application/json", **(headers or {})}
        data = raw if raw is not None else (
            None if body is None and method == "GET" else json.dumps(body or {}))
        conn.request(method, path, body=data, headers=hdrs)
        r = conn.getresponse()
        payload = r.read()
        out = (r.status, dict(r.getheaders()), payload)
        conn.close()
        return out

    def get(self, path):
        status, _h, data = self.req("GET", path)
        return status, (json.loads(data) if data[:1] in (b"{", b"[") else data)

    def post(self, path, body=None):
        status, _h, data = self.req("POST", path, body or {})
        return status, json.loads(data)

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
    """A fresh GUI server on a free port with an isolated config directory."""
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
    core.install_runner()
    c = Client(srv.server_address[1])
    c.cfg_path = cfg_path
    c.tmp = tmp_path
    yield c
    core.STATE.bus.close()
    srv.shutdown()
    srv.server_close()
    core.uninstall_runner()
    # let any job thread notice the shutdown before the next test resets STATE
    job = core.STATE.job
    if job is not None and job.thread is not None:
        job.thread.join(5)


# --- security ------------------------------------------------------------------

def test_static_pages_and_security_headers(gui):
    status, headers, body = gui.req("GET", "/")
    assert status == 200 and b"pedacito" in body
    assert "default-src 'self'" in headers["Content-Security-Policy"]
    assert headers["X-Frame-Options"] == "DENY"
    for name in ("app.css", "app.js", "panels.js", "settings.js", "main.js"):
        assert gui.req("GET", "/static/" + name)[0] == 200
    assert gui.req("GET", "/static/../gui/server.py")[0] == 404
    assert gui.req("GET", "/static/nope.js")[0] == 404


def test_foreign_host_origin_and_non_json_posts_are_refused(gui):
    assert gui.req("GET", "/api/state", headers={"Host": "evil.example"})[0] == 403
    assert gui.req("GET", "/events", headers={"Host": "evil.example"})[0] == 403
    status, _h, _b = gui.req("POST", "/api/clear", raw="{}",
                             headers={"Content-Type": "text/plain"})
    assert status == 403
    status, _h, _b = gui.req("POST", "/api/clear", {},
                             headers={"Origin": "http://evil.example"})
    assert status == 403
    assert gui.post("/api/clear")[0] == 200


def test_malformed_and_oversized_bodies_do_not_crash(gui):
    status, _h, data = gui.req("POST", "/api/run", raw="[1, 2]")
    assert status == 400 and json.loads(data)["ok"] is False
    status, _h, data = gui.req("POST", "/api/run", raw="not json")
    assert status == 400
    assert gui.get("/api/state")[0] == 200


# --- the event bus and stream ---------------------------------------------------

def test_bus_replays_after_a_sequence_number():
    bus = core.Bus()
    for i in range(3):
        bus.publish("x", n=i)
    events, resync = bus.wait(0, 0.01)
    assert [e["n"] for e in events] == [0, 1, 2] and not resync
    events, _ = bus.wait(2, 0.01)
    assert [e["n"] for e in events] == [2]
    assert bus.wait(3, 0.01) == ([], False)


def test_bus_flags_a_gap_when_the_buffer_overflowed():
    bus = core.Bus(cap=3)
    for i in range(10):
        bus.publish("x", n=i)
    _events, resync = bus.wait(2, 0.01)
    assert resync


def test_payload_id_never_clobbers_the_sequence_number():
    bus = core.Bus()
    bus.publish("msg_append", id=7, text="a")
    bus.publish("msg_append", id=7, text="b")
    events, _ = bus.wait(0, 0.01)
    assert [e["seq"] for e in events] == [1, 2]
    assert [e["id"] for e in events] == [7, 7]
    assert [e["seq"] for e in bus.wait(1, 0.01)[0]] == [2]


def test_sse_streams_new_events_with_ids(gui):
    conn = http.client.HTTPConnection("127.0.0.1", gui.port, timeout=10)
    conn.request("GET", "/events?since=0")
    resp = conn.getresponse()
    assert resp.status == 200
    assert resp.getheader("Content-Type") == "text/event-stream"
    core.STATE.add_message("system", "hello")
    core.STATE.append_message(core.STATE.messages[-1]["id"], " world")
    seen = []
    deadline = time.time() + 5
    while time.time() < deadline and len(seen) < 2:
        line = resp.fp.readline().decode()
        if line.startswith("data:"):
            seen.append(json.loads(line[5:]))
    conn.close()
    assert [e["type"] for e in seen] == ["msg_new", "msg_append"]
    assert seen[1]["seq"] == seen[0]["seq"] + 1
    assert seen[1]["id"] == seen[0]["msg"]["id"]


def test_snapshot_rebuilds_a_reloaded_page(gui):
    core.STATE.add_message("user", "hi", mode="task")
    core.STATE.term_write("some output\n")
    status, snap = gui.get("/api/snapshot")
    assert status == 200
    assert snap["messages"][-1]["text"] == "hi"
    assert "some output" in snap["term"]
    assert snap["seq"] == core.STATE.bus.seq


# --- terminal capture -------------------------------------------------------------

def test_tee_mirrors_output_to_the_terminal_pane_and_the_real_stream(capsys):
    core.STATE.__init__()
    orig = core.install_tee()
    try:
        print("to the pane", file=sys.stderr)
        print("and stdout")
    finally:
        core.uninstall_tee(orig)
    assert "to the pane" in core.STATE.term and "and stdout" in core.STATE.term
    captured = capsys.readouterr()
    assert "to the pane" in captured.err and "and stdout" in captured.out


def test_tee_survives_a_broken_real_stream():
    class Broken:
        def write(self, s):
            raise OSError("closed")

        def flush(self):
            raise OSError("closed")

    core.STATE.__init__()
    tee = core.Tee(Broken(), core.STATE.term_write)
    tee.write("still captured")
    tee.flush()
    assert "still captured" in core.STATE.term


# --- files, folders, plans ---------------------------------------------------------

def test_folder_browser_lists_dirs_hides_dotfiles_and_flags_projects(tmp_path):
    (tmp_path / "proj" / ".git").mkdir(parents=True)
    (tmp_path / "plain").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "f.txt").write_text("x")
    out = server.list_dir(str(tmp_path), show_files=False, hidden=False)
    names = {e["name"]: e for e in out["entries"]}
    assert set(names) == {"proj", "plain"}
    assert names["proj"]["project"] and not names["plain"]["project"]
    both = server.list_dir(str(tmp_path), show_files=True, hidden=True)
    assert {"f.txt", ".hidden"} <= {e["name"] for e in both["entries"]}
    # a missing path falls back to its nearest existing parent
    out = server.list_dir(str(tmp_path / "nope" / "deeper"), False, False)
    assert out["path"] == str(tmp_path.resolve())


def test_file_viewer_stays_inside_the_project(gui):
    root = gui.tmp / "p"
    root.mkdir()
    # The viewer returns the file's bytes verbatim; write LF explicitly so
    # Windows' text-mode newline translation does not turn this into CRLF.
    (root / "a.py").write_bytes(b"x = 1\n")
    (root / "b.bin").write_bytes(b"\x00\x01\x02")
    (gui.tmp / "secret.txt").write_text("nope")
    core.STATE.root = root
    assert gui.get("/api/file?path=a.py")[1]["text"] == "x = 1\n"
    assert gui.get("/api/file?path=b.bin")[1]["binary"] is True
    assert gui.get("/api/file?path=../secret.txt")[0] == 400
    assert gui.get("/api/file?path=missing.py")[0] == 400


def test_parse_plan_statuses():
    text = ("# Plan\n\nGoal: x\n\n- [ ] one\n- [x] two\n- [!] three\n- [-] four\n"
            "not a task\n")
    tasks = jobs.parse_plan(text)
    assert [t["status"] for t in tasks] == ["todo", "done", "failed", "skipped"]
    assert [t["title"] for t in tasks] == ["one", "two", "three", "four"]


# --- job control -----------------------------------------------------------------

def test_only_one_job_runs_at_a_time_and_stop_interrupts_it(gui):
    started = threading.Event()

    def spin(job):
        started.set()
        while True:
            time.sleep(0.02)

    job = jobs.spawn("task", "spinning", spin)
    assert job is not None and started.wait(5)
    assert jobs.spawn("task", "second", lambda j: None) is None
    status, body = gui.post("/api/run", {"mode": "explain", "text": "x"})
    assert status in (400, 409)
    assert jobs.stop_job() is True
    assert gui.wait_idle(10)
    assert job.cancelled
    assert any(m.get("kind") == "cancelled" for m in core.STATE.messages)
    # the slot is free again
    assert jobs.spawn("task", "again", lambda j: None) is not None
    assert gui.wait_idle(10)


def test_a_failing_job_reports_the_error_in_the_chat(gui):
    def boom(job):
        raise RuntimeError("kaboom")

    jobs.spawn("task", "failing", boom)
    assert gui.wait_idle(10)
    assert core.STATE.messages[-1]["role"] == "error"
    assert "kaboom" in core.STATE.messages[-1]["text"]
    assert "kaboom" in gui.get("/api/state")[1]["error"]


def test_model_questions_block_until_answered(gui):
    got = {}

    def ask(job):
        got["answer"] = core.STATE.ask_user("Which one?")

    jobs.spawn("task", "asking", ask)
    assert gui.wait_for(lambda s: s["question"], 10)
    qid = next(m["qid"] for m in core.STATE.messages if m["role"] == "question")
    assert gui.post("/api/answer", {"qid": qid + 99, "text": "x"})[0] == 409
    assert gui.post("/api/answer", {"qid": qid, "text": "the first"})[0] == 200
    assert gui.wait_idle(10)
    assert got["answer"] == "the first"
    q = next(m for m in core.STATE.messages if m["role"] == "question")
    assert q["answered"] and q["answer"] == "the first"


def test_steer_requires_a_running_job(gui):
    status, body = gui.post("/api/steer", {"text": "go left"})
    assert status == 409 and not body["ok"]


# --- staged review (no model needed) --------------------------------------------------

def _staged(gui, applied):
    root = gui.tmp / "r"
    root.mkdir()
    (root / "a.txt").write_text("one\ntwo\n")
    (root / "b.txt").write_text("alpha\n")
    from pedacito import edits
    reply = ("--- FILE: a.txt\n<<<<<<< SEARCH\none\n=======\nONE\n>>>>>>> REPLACE\n"
             "--- FILE: b.txt\n<<<<<<< SEARCH\nalpha\n=======\nBETA\n>>>>>>> REPLACE\n")
    results = edits.apply_edits(root, edits.parse_edits(reply), dry_run=True)
    diffs = review_mod.diffs_from_results(results, root)

    def writer():
        applied.append([d.rel for d in diffs])
        return review_mod.write_files(root, diffs, None)

    out = {}
    thread = threading.Thread(target=lambda: out.update(
        res=review_mod.run("<html>", "n", writer, diffs=diffs, task="t")))
    thread.start()
    assert gui.wait_for(lambda s: s["review"] and s["review"]["pending"], 10)
    return root, thread, out


def test_review_partial_accept_writes_only_selected_files_and_honours_edits(gui):
    applied = []
    core.STATE.job = core.Job(1, "task", "t")        # a job is waiting on the page
    root, thread, out = _staged(gui, applied)
    meta = gui.get("/api/review")[1]["review"]
    assert [f["rel"] for f in meta["files"]] == ["a.txt", "b.txt"]
    assert meta["partial_ok"] and meta["pending"]
    rid = meta["id"]
    status, f = gui.get(f"/api/review/file?id={rid}&rel=a.txt")
    assert status == 200 and f["added"] == 1 and "ONE" in f["after"]
    assert gui.get(f"/api/review/file?id={rid + 5}&rel=a.txt")[0] == 404
    # hand-edit a.txt before accepting
    status, d = gui.post("/api/review/rediff",
                         {"id": rid, "rel": "a.txt", "text": "uno\ntwo\n"})
    assert status == 200 and d["added"] == 1
    # a stale id is refused and leaves the review open
    assert gui.post("/api/review/decide",
                    {"id": rid + 1, "action": "apply"})[0] == 409
    status, body = gui.post("/api/review/decide",
                            {"id": rid, "action": "apply", "files": ["a.txt"]})
    thread.join(10)
    assert status == 200 and body["ok"]
    assert out["res"][0] == "applied"
    assert (root / "a.txt").read_text() == "uno\ntwo\n"      # edited text
    assert (root / "b.txt").read_text() == "alpha\n"         # left out
    assert applied == [["a.txt"]]
    # deciding twice is refused
    assert gui.post("/api/review/decide", {"id": rid, "action": "reject"})[0] == 409
    meta = gui.get("/api/review")[1]["review"]
    assert meta["outcome"] == "applied" and not meta["pending"]


def test_review_reject_changes_nothing_and_unblocks_the_job(gui):
    core.STATE.job = core.Job(1, "task", "t")
    root, thread, out = _staged(gui, [])
    rid = gui.get("/api/review")[1]["review"]["id"]
    assert gui.post("/api/review/decide", {"id": rid, "action": "reject"})[0] == 200
    thread.join(10)
    assert out["res"][0] == "cancelled"
    assert (root / "a.txt").read_text() == "one\ntwo\n"


def test_review_apply_with_no_files_selected_is_refused(gui):
    core.STATE.job = core.Job(1, "task", "t")
    root, thread, out = _staged(gui, [])
    rid = gui.get("/api/review")[1]["review"]["id"]
    status, body = gui.post("/api/review/decide",
                            {"id": rid, "action": "apply", "files": []})
    assert status == 409 and "at least one" in body["detail"]
    gui.post("/api/review/decide", {"id": rid, "action": "reject"})
    thread.join(10)


# --- settings endpoints ----------------------------------------------------------------

def test_settings_first_run_creates_a_working_config(gui):
    status, p = gui.get("/api/settings")
    assert status == 200 and p["user"]["exists"] is False
    assert any(i["name"] == "vram_gb" for i in p["schema"])
    assert gui.get("/api/state")[1]["needs_setup"] is True
    status, body = gui.post("/api/settings/setup",
                            {"base_url": "http://x:1/v1", "model": "m", "vram_gb": "16"})
    assert status == 200 and body["user"]["exists"]
    assert gui.get("/api/state")[1]["needs_setup"] is False
    flat, profiles = config_mod._read(gui.cfg_path)
    assert flat["vram_gb"] == 16 and profiles["default"]["model"] == "m"
    # not twice
    assert gui.post("/api/settings/setup",
                    {"base_url": "u", "model": "m"})[0] == 400
    # and the connection is required
    gui.cfg_path.unlink()
    assert gui.post("/api/settings/setup", {"base_url": "", "model": ""})[0] == 400


def _config(gui):
    gui.post("/api/settings/setup",
             {"base_url": "http://x:1/v1", "model": "m", "vram_gb": "8"})


def test_settings_set_profile_edits_keep_comments_and_make_a_backup(gui):
    _config(gui)
    before = gui.cfg_path.read_text()
    status, body = gui.post("/api/settings/set", {
        "file": "user", "profile": "default",
        "changes": {"model": "bigger", "tier": "medium", "api_key": "sk-secret"}})
    assert status == 200
    text = gui.cfg_path.read_text()
    assert re.search(r'^model\s*= "bigger"', text, re.M)
    assert 'tier = "medium"' in text
    assert "# Pedacito configuration." in text
    assert Path(str(gui.cfg_path) + ".bak").read_text() == before
    prof = body["user"]["profiles"]["default"]
    assert prof["api_key"] == "" and "api_key" in prof["_secret_set"]   # masked
    # the structured views never carry it; only the raw editor shows the file
    assert "sk-secret" not in json.dumps(body["user"]["profiles"])
    assert "sk-secret" not in json.dumps(body["user"]["settings"])
    assert "sk-secret" not in json.dumps(body["resolved"])
    # a blank secret means "unchanged"
    gui.post("/api/settings/set", {"file": "user", "profile": "default",
                                   "changes": {"api_key": ""}})
    assert 'api_key = "sk-secret"' in gui.cfg_path.read_text()


def test_settings_invalid_values_are_rejected_and_the_file_is_untouched(gui):
    _config(gui)
    before = gui.cfg_path.read_text()
    status, body = gui.post("/api/settings/set", {
        "file": "user", "changes": {"vram_gb": -4}})
    assert status == 400 and "positive whole number" in body["detail"]
    status, body = gui.post("/api/settings/set", {
        "file": "user", "profile": "default", "changes": {"tier": "gigantic"}})
    assert status == 400
    status, body = gui.post("/api/settings/set", {
        "file": "user", "changes": {"temprature": 1}})
    assert status == 400 and "temprature" in body["detail"]
    status, body = gui.post("/api/settings/raw", {"file": "user", "text": "[oops"})
    assert status == 400 and "invalid TOML" in body["detail"]
    assert gui.cfg_path.read_text() == before


def test_settings_profile_lifecycle(gui):
    _config(gui)
    assert gui.post("/api/settings/profile", {
        "file": "user", "action": "create", "name": "fast",
        "values": {"base_url": "http://y/v1", "model": "tiny"}})[0] == 200
    status, body = gui.post("/api/settings/profile", {
        "file": "user", "action": "duplicate", "name": "fast2", "source": "fast"})
    assert status == 200
    assert {"default", "fast", "fast2"} <= set(body["user"]["profiles"])
    assert body["user"]["profiles"]["fast2"]["model"] == "tiny"
    assert gui.post("/api/settings/profile", {
        "file": "user", "action": "create", "name": "fast", "values": {}})[0] == 400
    status, body = gui.post("/api/settings/profile", {
        "file": "user", "action": "delete", "name": "fast2"})
    assert status == 200 and "fast2" not in body["user"]["profiles"]
    assert gui.post("/api/settings/profile", {
        "file": "user", "action": "delete", "name": "ghost"})[0] == 400


def test_peek_lists_profiles_without_opening_the_project(gui):
    _config(gui)
    gui.post("/api/settings/profile", {"file": "user", "action": "create",
                                       "name": "fast", "values": {"model": "t"}})
    status, out = gui.post("/api/peek", {"path": str(gui.tmp)})
    assert out["ok"] and {"default", "fast"} <= set(out["profiles"])
    assert gui.post("/api/peek", {"path": str(gui.tmp / "missing")})[1]["ok"] is False
    assert core.STATE.root is None                      # nothing was opened


def test_settings_raw_save_and_project_file_needs_a_project(gui):
    _config(gui)
    text = gui.cfg_path.read_text() + "\n# a note\n"
    assert gui.post("/api/settings/raw", {"file": "user", "text": text})[0] == 200
    assert gui.cfg_path.read_text().endswith("# a note\n")
    status, p = gui.get("/api/settings")
    assert p["project"]["available"] is False
    root = gui.tmp / "proj"
    root.mkdir()
    core.STATE.root = root
    status, body = gui.post("/api/settings/set", {
        "file": "project", "changes": {"max_steps": 6}})
    assert status == 200
    assert (root / ".pedacito.toml").read_text().strip().endswith("max_steps = 6")
    assert body["project"]["exists"]


def test_settings_probe_server_reports_models_or_the_error(gui):
    status, out = gui.post("/api/settings/models", {"base_url": ""})
    assert out["ok"] is False
    status, out = gui.post("/api/settings/models",
                           {"base_url": f"http://127.0.0.1:{_free_port()}/v1"})
    assert out["ok"] is False and out["error"]


# --- end to end with the fake server ---------------------------------------------------

@pytest.fixture(scope="module")
def fake_port():
    port = _free_port()
    env = dict(os.environ, FAKE_PORT=str(port), FAKE_REPLY=EDIT_REPLY,
               FAKE_READ_FILE="app.py")
    env.pop("FAKE_ACTIONS", None)
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


@pytest.fixture
def project(gui, fake_port):
    gui.post("/api/settings/setup", {"base_url": f"http://127.0.0.1:{fake_port}/v1",
                                     "model": "fake-model", "vram_gb": "8"})
    root = gui.tmp / "proj"
    root.mkdir()
    (root / "app.py").write_text(APP)
    status, body = gui.post("/api/project", {"path": str(root), "profile": ""})
    assert status == 202
    assert gui.wait_idle(30)
    state = gui.get("/api/state")[1]
    assert state["root"] == str(root.resolve()), core.STATE.messages
    assert state["model_ok"] is True and state["has_index"] is False
    status, _ = gui.post("/api/index", {"summarise": False})
    assert status == 202 and gui.wait_idle(30)
    assert gui.get("/api/state")[1]["has_index"] is True
    gui.root = root
    return gui


def _review_pending(state):
    return state if state["review"] and state["review"]["pending"] else None


@pytest.mark.network
def test_open_project_lists_files_and_remembers_it(project):
    status, files = project.get("/api/files")
    assert files["files"] == ["app.py"]
    assert project.get("/api/state")[1]["recent"][0] == str(project.root.resolve())
    assert "greet" in project.get("/api/file?path=app.py")[1]["text"]


@pytest.mark.network
def test_task_in_review_mode_shows_a_diff_and_accept_writes_it(project):
    originals = core.install_tee()      # pytest swaps stderr between phases
    try:
        _review_accept_flow(project)
    finally:
        core.uninstall_tee(originals)


def _review_accept_flow(project):
    status, body = project.post("/api/run", {
        "mode": "task", "text": "make greet return its greeting",
        "write": "review", "plan": "never"})
    assert status == 202
    state = project.wait_for(_review_pending, 30)
    assert state, core.STATE.messages
    assert (project.root / "app.py").read_text() == APP       # nothing written yet
    meta = project.get("/api/review")[1]["review"]
    assert [f["rel"] for f in meta["files"]] == ["app.py"]
    status, f = project.get(f"/api/review/file?id={meta['id']}&rel=app.py")
    kinds = [r[0] for r in f["rows"]]
    assert "replace" in kinds and f["added"] == 1 and f["removed"] == 1
    status, body = project.post("/api/review/decide",
                                {"id": meta["id"], "action": "apply",
                                 "files": ["app.py"]})
    assert status == 200 and body["ok"]
    assert project.wait_idle(30)
    assert 'return f"hello {name}"' in (project.root / "app.py").read_text()
    result = [m for m in core.STATE.messages if m.get("kind") == "result"][-1]
    assert result["head"] == "Wrote 1 file(s)."
    assert result["results"][0]["file"] == "app.py"
    assert any(m["role"] == "activity" for m in core.STATE.messages)
    assert any(m["role"] == "stream" for m in core.STATE.messages)
    assert "looking things up" in core.STATE.term


@pytest.mark.network
def test_task_review_reject_leaves_the_file_alone(project):
    project.post("/api/run", {"mode": "task", "text": "change greet",
                              "write": "review", "plan": "never"})
    state = project.wait_for(_review_pending, 30)
    assert state
    rid = state["review"]["id"]
    assert project.post("/api/review/decide",
                        {"id": rid, "action": "reject"})[0] == 200
    assert project.wait_idle(30)
    assert (project.root / "app.py").read_text() == APP
    card = [m for m in core.STATE.messages if m["role"] == "review"][-1]
    assert card["outcome"] == "cancelled"


@pytest.mark.network
def test_direct_write_then_restore_from_the_backup(project):
    status, _ = project.post("/api/run", {
        "mode": "task", "text": "make greet return its greeting",
        "write": "apply", "plan": "never"})
    assert status == 202 and project.wait_idle(30)
    assert 'return f"hello {name}"' in (project.root / "app.py").read_text()
    assert not (project.root / ".pedacito" / "run.lock").exists()   # released
    snaps = project.get("/api/state")[1]["snapshots"]
    assert snaps
    status, body = project.post("/api/restore", {"snapshot": snaps[0]})
    assert status == 200 and body["ok"]
    assert (project.root / "app.py").read_text() == APP
    assert project.post("/api/restore", {"snapshot": "nope"})[0] == 400


@pytest.mark.network
def test_explain_streams_into_the_chat_as_an_answer(project):
    status, _ = project.post("/api/run", {"mode": "explain",
                                          "text": "how does greet work?"})
    assert status == 202 and project.wait_idle(30)
    answer = [m for m in core.STATE.messages if m.get("kind") == "answer"][-1]
    assert answer["role"] == "assistant" and answer["text"].strip()
    assert (project.root / "app.py").read_text() == APP      # read-only


@pytest.mark.network
def test_running_without_an_index_is_refused_with_guidance(gui, fake_port):
    gui.post("/api/settings/setup", {"base_url": f"http://127.0.0.1:{fake_port}/v1",
                                     "model": "fake-model", "vram_gb": "8"})
    root = gui.tmp / "p2"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    gui.post("/api/project", {"path": str(root), "profile": ""})
    assert gui.wait_idle(30)
    gui.post("/api/run", {"mode": "task", "text": "do it"})
    assert gui.wait_idle(30)
    last = core.STATE.messages[-1]
    assert last["role"] == "error" and "index" in last["text"]


@pytest.mark.network
def test_new_project_preview_writes_nothing(gui, fake_port):
    gui.post("/api/settings/setup", {"base_url": f"http://127.0.0.1:{fake_port}/v1",
                                     "model": "fake-model", "vram_gb": "8"})
    target = gui.tmp / "brand-new"
    status, _ = gui.post("/api/run", {"mode": "new", "text": "a demo cli",
                                      "project": str(target), "build": "preview"})
    assert status == 202 and gui.wait_idle(30)
    assert not target.exists()
    # and it insists on a folder and a description
    gui.post("/api/run", {"mode": "new", "text": "a demo", "project": ""})
    assert gui.wait_idle(30)
    assert "directory" in core.STATE.messages[-1]["text"]
    gui.post("/api/run", {"mode": "new", "text": "", "project": str(target)})
    assert gui.wait_idle(30)
    assert "Describe" in core.STATE.messages[-1]["text"]
