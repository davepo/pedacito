"""Per-project chat threads and friendly default names.

The transcript lives in `<project>/.pedacito/gui/threads/<id>.json` (one file
per conversation, written atomically). A new chat opens named by the clock
(`Nov 08, 09:20 PM`), takes its first user question as its title, and a manual
rename (`title_custom`) is never overwritten again. Threads are restored on
restart and switched from the Chat head via
`/api/threads/new|select|delete|rename`. No model and no server round-trip
needed for the pure `core.State` tests; the endpoint tests reuse `gui` from
test_gui."""
# `gui` below is test_gui's server fixture: pytest hands it to each test as a
# parameter, which ruff otherwise reads as a redefinition.
# ruff: noqa: F811
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pedacito.gui import core  # noqa: E402
from test_gui import gui  # noqa: F401  (the shared server fixture)


def _file(root, tid):
    return core.threads_dir(root) / (tid + ".json")


# ---------------------------------------------------------------- the basics

def test_thread_is_written_per_message_and_titled_from_the_first_question(tmp_path):
    st = core.State()
    st.set_root(tmp_path)
    tid = st.thread_id
    st.add_message("user", "fix the flaky login test")
    st.add_message("assistant", "the fixture raced; here is the patch")
    assert st.threads[tid]["title"] == "fix the flaky login test"
    data = json.loads(_file(tmp_path, tid).read_text(encoding="utf-8"))
    assert data["id"] == tid and data["title"] == "fix the flaky login test"
    assert [m["role"] for m in data["messages"]] == ["user", "assistant"]
    assert data["messages"][0]["content"] == "fix the flaky login test"
    assert data["created_at"] <= data["updated_at"]
    # the next user message must not steal the title
    st.add_message("user", "and run the tests")
    assert st.threads[tid]["title"] == "fix the flaky login test"
    st.delete_thread(tid)


def test_thread_title_is_one_collapsed_short_line():
    assert core.thread_title_from("  do   this\n   then that  ") == "do this then that"
    # blank input yields "" -- the caller then keeps the timestamp title
    assert core.thread_title_from("") == ""
    assert core.thread_title_from("   \n  ") == ""
    long = core.thread_title_from("x" * 200)
    assert len(long) == core.THREAD_TITLE_CHARS + 1      # the ellipsis
    assert long.endswith("\u2026")


def test_default_thread_title_is_a_friendly_timestamp():
    # "Nov 08, 09:20 PM": month/day and 12-hour clock, no year, no seconds
    from datetime import datetime
    assert core.default_thread_title(datetime(2026, 11, 8, 21, 20)) == "Nov 08, 09:20 PM"
    assert re.fullmatch(r"[A-Z][a-z]{2} \d{2}, \d{2}:\d{2} [AP]M",
                        core.default_thread_title())


def test_threads_survive_a_restart(tmp_path):
    st = core.State()
    st.set_root(tmp_path)
    tid = st.thread_id
    st.add_message("user", "persist me")
    # a brand-new State (what the GUI process does on start) reads the folder
    fresh = core.State()
    fresh.set_root(tmp_path)
    assert fresh.thread_id == tid                 # the newest thread reopens
    assert [m["text"] for m in fresh.messages] == ["persist me"]
    assert fresh.threads[tid]["title"] == "persist me"
    # and ids keep counting up from the restored transcript, no collisions
    m = fresh.add_message("user", "still here")
    assert m["id"] == 2
    assert len(json.loads(_file(tmp_path, tid).read_text())["messages"]) == 2
    fresh.delete_thread(tid)


def test_threads_are_per_project(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    st = core.State()
    st.set_root(a)
    tid_a = st.thread_id
    st.add_message("user", "in project a")
    st.set_root(b)                                 # switching flushes a's thread
    assert st.thread_id != tid_a and st.messages == []
    st.add_message("user", "in project b")
    assert _file(a, tid_a).exists() and not _file(a, st.thread_id).exists()
    assert _file(b, st.thread_id).exists()
    # back to a: a's thread (and only a's) comes back
    st.set_root(a)
    assert st.thread_id == tid_a
    assert [m["text"] for m in st.messages] == ["in project a"]
    st.delete_thread(tid_a)


def test_reopening_the_same_root_keeps_the_current_thread(tmp_path):
    st = core.State()
    st.set_root(tmp_path)
    tid = st.thread_id
    st.add_message("user", "one")
    st.set_root(tmp_path)                          # e.g. a profile switch
    assert st.thread_id == tid
    assert [m["text"] for m in st.messages] == ["one"]
    st.delete_thread(tid)


def test_new_select_and_delete_threads(tmp_path):
    st = core.State()
    st.set_root(tmp_path)
    t1 = st.thread_id
    st.add_message("user", "first chat")
    t2 = st.new_thread()
    assert t2 != t1 and st.messages == []
    assert st.threads[t1]["title"] == "first chat"    # the old one stays saved
    assert _file(tmp_path, t1).exists() and _file(tmp_path, t2).exists()
    assert st.select_thread(t1) and [m["text"] for m in st.messages] == ["first chat"]
    assert not st.select_thread("no-such-id")
    # deleting the active thread opens the most recent survivor
    assert st.delete_thread(t1)
    assert st.thread_id == t2 and st.messages == []
    assert not _file(tmp_path, t1).exists()
    assert st.delete_thread(t2)
    # nothing left: a fresh thread takes over, still inside this project
    assert len(st.threads) == 1 and st.messages == []
    assert st.delete_thread("no-such-id") is False
    fresh = next(iter(st.threads))
    st.delete_thread(fresh)


def test_new_threads_are_timed_stamped_then_titled_then_renamed(tmp_path):
    """Thread naming contract: timestamp -> first question -> hand-picked name.
    A rename marks the thread custom and survives a full reload."""
    from datetime import datetime
    before = core.default_thread_title(datetime.now())
    st = core.State()
    st.set_root(tmp_path)
    tid = st.thread_id
    after = core.default_thread_title(datetime.now())
    # (the two endpoints cover a minute rolling over between the calls)
    assert st.threads[tid]["title"] in {before, after}
    assert st.threads[tid]["title_custom"] is False
    # the first user question becomes the title (still not "custom")
    st.add_message("user", "why is the ci red?")
    assert st.threads[tid]["title"] == "why is the ci red?"
    assert st.threads[tid]["title_custom"] is False
    # a manual rename collapses whitespace and sticks as custom
    assert st.rename_thread(tid, "  login   bug \n hunt  ") == "login bug hunt"
    assert st.threads[tid]["title_custom"] is True
    # further messages never touch a custom title
    st.add_message("assistant", "the lint job flakes")
    assert st.threads[tid]["title"] == "login bug hunt"
    # persisted to the thread file
    data = json.loads(_file(tmp_path, tid).read_text(encoding="utf-8"))
    assert data["title"] == "login bug hunt" and data["title_custom"] is True
    # ... and survives a fresh State (a GUI restart)
    fresh = core.State()
    fresh.set_root(tmp_path)
    assert fresh.threads[tid]["title"] == "login bug hunt"
    assert fresh.threads[tid]["title_custom"] is True
    # clear_messages resets a plain title to the timestamp, never a custom one
    fresh.clear_messages()
    assert fresh.threads[tid]["title"] == "login bug hunt"
    # renames are capped and blanks are refused, for every existing thread too
    long = st.rename_thread(tid, "y" * 300)
    assert long == "y" * core.THREAD_RENAME_CHARS
    assert st.rename_thread(tid, "   ") is None
    assert st.rename_thread(tid, "login bug hunt") == "login bug hunt"
    assert st.rename_thread("no-such-thread", "x") is None
    # renaming another (inactive) thread works and persists as well
    t2 = st.new_thread()
    st.add_message("user", "second chat")
    assert st.select_thread(tid)                         # back to the first one
    assert st.rename_thread(t2, "quiet side conversation")
    assert st.threads[t2]["title_custom"] is True
    assert st.thread_id == tid                           # renaming did not switch
    again = core.State()
    again.set_root(tmp_path)
    assert again.threads[t2]["title"] == "quiet side conversation"
    st.delete_thread(tid)
    st.delete_thread(t2)


def test_clear_empties_the_thread_but_keeps_its_file(tmp_path):
    st = core.State()
    st.set_root(tmp_path)
    tid = st.thread_id
    st.add_message("user", "wipe me")
    before = core.default_thread_title()
    st.clear_messages()
    after = core.default_thread_title()
    assert st.messages == [] and st.thread_id == tid
    # the question-derived title goes back to a fresh timestamp
    assert st.threads[tid]["title"] in {before, after}
    assert _file(tmp_path, tid).exists()
    assert json.loads(_file(tmp_path, tid).read_text())["messages"] == []
    st.delete_thread(tid)


def test_corrupt_thread_files_are_skipped_with_a_warning(tmp_path, capsys):
    d = core.threads_dir(tmp_path)
    d.mkdir(parents=True)
    good = {"id": "good1", "title": "keep me", "created_at": 5.0,
            "updated_at": 5.0, "messages": [{"role": "user", "content": "ok", "ts": 1.0}]}
    (d / "good1.json").write_text(json.dumps(good), encoding="utf-8")
    (d / "broken.json").write_text("{not json", encoding="utf-8")
    (d / "notamsg.json").write_text(json.dumps({"id": "n1", "messages": "no"}),
                                    encoding="utf-8")
    (d / "weird.json").write_text(json.dumps({"id": "bad id!!", "messages": []}),
                                  encoding="utf-8")
    (d / "notes.json").write_text("hello", encoding="utf-8")   # not a thread dir file? .json -> checked
    recs = core.load_thread_records(tmp_path)
    assert [r["id"] for r in recs] == ["good1"]
    err = capsys.readouterr().err
    assert "ignoring thread broken.json" in err
    assert "ignoring thread notamsg.json" in err
    st = core.State()
    st.set_root(tmp_path)
    assert st.thread_id == "good1"                 # the readable thread opens
    assert [m["text"] for m in st.messages] == ["ok"]
    st.delete_thread("good1")


def test_load_caps_at_the_newest_threads(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "MAX_THREADS", 2)
    d = core.threads_dir(tmp_path)
    d.mkdir(parents=True)
    for i, ts in enumerate((100.0, 300.0, 200.0)):
        (d / f"t{i}.json").write_text(json.dumps(
            {"id": f"t{i}", "title": "x", "created_at": ts, "updated_at": ts,
             "messages": []}), encoding="utf-8")
    recs = core.load_thread_records(tmp_path)
    assert [r["id"] for r in recs] == ["t1", "t2"]
    # an empty project loads no threads and still gets a usable chat
    st = core.State()
    st.set_root(tmp_path / "nowhere")
    assert len(st.threads) == 1 and st.messages == []
    st.new_thread()                                # nothing was clobbered
    assert st.delete_thread(next(iter(st.threads)))


# ---------------------------------------------------------------- endpoints

def test_thread_endpoints_round_trip(gui):
    st = core.STATE
    st.set_root(gui.tmp)                           # threads need a project folder
    t1 = st.thread_id
    st.add_message("user", "please fix the login bug")
    status, j = gui.post("/api/threads/new", {})
    assert status == 200 and j["ok"] and j["id"] != t1
    t2 = j["id"]
    assert st.messages == []
    status, j = gui.post("/api/threads/select", {"id": t1})
    assert status == 200 and j["ok"] and j["id"] == t1
    assert [m["text"] for m in st.messages] == ["please fix the login bug"]
    status, j = gui.post("/api/threads/select", {"id": "ghost"})
    assert status == 400 and j["ok"] is False
    status, state = gui.get("/api/state")
    assert state["active_thread_id"] == t1
    assert state["thread_title"] == "please fix the login bug"
    assert {t["id"] for t in state["threads"]} == {t1, t2}
    active = [t for t in state["threads"] if t["active"]]
    assert [t["id"] for t in active] == [t1]
    assert active[0]["message_count"] == 1
    status, j = gui.post("/api/threads/delete", {"id": t1})
    assert status == 200 and j["ok"] and j["id"] != t1
    assert st.messages == []                       # the empty survivor opens
    assert not _file(st.root, t1).exists()
    status, j = gui.post("/api/threads/delete", {"id": t1})
    assert status == 400 and j["ok"] is False


def test_clear_endpoint_keeps_working(gui):
    tid = core.STATE.thread_id
    core.STATE.add_message("user", "hi")
    status, j = gui.post("/api/clear", {})
    assert status == 200 and j["ok"] is True
    assert core.STATE.messages == []
    assert tid in core.STATE.threads               # same chat, blank slate
    _status, state = gui.get("/api/state")
    assert state["active_thread_id"] == tid
    # "hi" is gone: the thread carries a fresh timestamp name again
    assert re.fullmatch(r"[A-Z][a-z]{2} \d{2}, \d{2}:\d{2} [AP]M",
                        state["thread_title"])
    assert state["thread_title_custom"] is False


def test_rename_endpoint_round_trip(gui):
    st = core.STATE
    st.set_root(gui.tmp)                           # threads need a project folder
    tid = st.thread_id
    st.add_message("user", "rename me properly")
    status, j = gui.post("/api/threads/rename", {"id": tid, "title": "  gui   notes \n" })
    assert status == 200 and j["ok"] and j["title"] == "gui notes"
    assert st.threads[tid]["title_custom"] is True
    assert any(t["id"] == tid and t["title_custom"] for t in j["threads"])
    _status, state = gui.get("/api/state")
    assert state["thread_title"] == "gui notes"
    assert state["thread_title_custom"] is True
    # persisted to the thread file itself
    data = json.loads(_file(gui.tmp, tid).read_text(encoding="utf-8"))
    assert data["title"] == "gui notes" and data["title_custom"] is True
    # a later prompt cannot undo a custom name
    st.add_message("user", "and check the tests")
    assert st.threads[tid]["title"] == "gui notes"
    # blanks and unknown ids are refused
    status, j = gui.post("/api/threads/rename", {"id": tid, "title": "   "})
    assert status == 400 and j["ok"] is False
    status, j = gui.post("/api/threads/rename", {"id": "ghost", "title": "x"})
    assert status == 400 and j["ok"] is False
    assert st.threads[tid]["title"] == "gui notes"   # nothing changed
    st.delete_thread(tid)
