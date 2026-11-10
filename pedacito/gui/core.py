"""Shared state for the GUI server: the event bus, terminal capture, the chat
transcript, the staged review, and the live permission broker (a gated tool
call pauses its worker until the browser modal answers).

Locking: every read or write of `STATE` fields goes through `STATE.lock` (an
RLock), held only around state mutations -- never around model or filesystem
work. Workers snapshot what they need, work unlocked, then swap results in.

Events reach the browser over Server-Sent Events (`Bus`). The chat transcript
and terminal text are also kept here, so a reloaded page rebuilds itself from
`snapshot()` and then follows the stream from the sequence number it holds.
"""

from __future__ import annotations

import json
import os
import queue
import re
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from .. import review as review_mod
from .. import security, workspace
from ..config import USER_CONFIG
from ..events import cache_hit_ratio
from ..index import INDEX_VERSION

RECENT_FILE = USER_CONFIG.parent / "gui_recent.json"
MAX_RECENT = 8
MAX_MESSAGES = 600
MAX_TERM_CHARS = 200_000
QUESTION_TIMEOUT = 900.0
# How long a gated tool call waits for a human answer before the run is
# allowed to continue with a clean denial (the modal never leaves a worker
# hanging forever -- an unattended server must not freeze a job).
PERMISSION_TIMEOUT = 120.0

# Job kinds that drive the Engine V2 loop (and thus the activity telemetry).
RUN_KINDS = ("task", "explain", "auto", "new")
# Harness event kind -> the phase the ACTIVITY & PLAN panel should show.
EVENT_PHASE = {"step": "gather", "gather_done": "edit", "answer": "edit",
               "edit_intents": "edit", "incremental_edit": "edit",
               "verify": "verify", "tests": "verify",
               "rehydrate": "gather", "fold": "gather"}
# client.site -> phase, while a run is live (the call site is authoritative).
SITE_PHASE = {"gather": "gather", "compaction": "gather", "answer": "edit",
              "verify": "verify", "repair": "repair"}

# Chat threads live inside the project: <root>/.pedacito/gui/threads/<id>.json,
# one readable file per conversation, kept so the GUI can restore the last chat
# after a restart. A fresh thread opens under a friendly local timestamp
# ("Nov 08, 09:20 PM"); once the first request lands, the title becomes that
# request (one line, short) -- no extra model call, and the title stays honest
# because it is literally what the user asked. A manual rename sets
# `title_custom`, and then no automatic update ever touches the name again.
THREADS_SUBPATH = Path(".pedacito") / "gui" / "threads"
THREAD_TITLE_CHARS = 48
THREAD_TITLE_FORMAT = "%b %d, %I:%M %p"
# A hand-typed rename is not truncated to a first-message title's 48 chars, but
# it still gets a sane ceiling (the switcher and the file both stay readable).
THREAD_RENAME_CHARS = 120
# Only this many most-recent threads load into memory per project; older files
# stay on disk untouched.
MAX_THREADS = 100

_THREAD_ID_RE = re.compile(r"\A[A-Za-z0-9_-]{1,64}\Z")


def threads_dir(root) -> Path:
    return Path(root) / THREADS_SUBPATH


def default_thread_title(when=None) -> str:
    """The name a brand-new conversation carries until its first prompt (or a
    rename): a clean local timestamp, like `Nov 08, 09:20 PM`."""
    return (when or datetime.now()).strftime(THREAD_TITLE_FORMAT)


def thread_title_from(text) -> str:
    """The auto-title for a conversation: its first request, one line, short.
    "" for nothing usable -- the caller then keeps the timestamp title."""
    collapse = re.sub(r"\s+", " ", str(text or "")).strip()
    if not collapse:
        return ""
    if len(collapse) <= THREAD_TITLE_CHARS:
        return collapse
    return collapse[:THREAD_TITLE_CHARS].rstrip() + "…"


def read_thread_file(path):
    """One thread JSON -> a normalized record, or None (with a warning) when
    the file is not a thread. A corrupt thread must never stop the GUI from
    opening the project."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        sys.stderr.write("[gui] ignoring thread %s: %s\n" % (path.name, exc))
        return None
    if not isinstance(data, dict):
        sys.stderr.write("[gui] ignoring thread %s: not an object\n" % path.name)
        return None
    tid = data.get("id")
    if not (isinstance(tid, str) and _THREAD_ID_RE.match(tid)):
        sys.stderr.write("[gui] ignoring thread %s: bad id\n" % path.name)
        return None
    raw_msgs = data.get("messages")
    if not isinstance(raw_msgs, list):
        sys.stderr.write("[gui] ignoring thread %s: bad messages\n" % path.name)
        return None
    messages = []
    for i, m in enumerate(raw_msgs):
        if not isinstance(m, dict):
            continue
        try:
            ts = float(m.get("ts") or 0)
        except (TypeError, ValueError):
            ts = 0.0
        # the file stores {role, content, ts}; the runtime transcript is
        # {id, role, text, time, ...}
        messages.append({"id": i + 1, "role": str(m.get("role") or "user"),
                         "text": str(m.get("content") or ""), "time": ts})
    try:
        created = float(data.get("created_at") or 0)
        updated = float(data.get("updated_at") or created or 0)
    except (TypeError, ValueError):
        created = updated = time.time()
    title = str(data.get("title") or "").strip() or default_thread_title()
    return {"id": tid, "title": title, "created_at": created, "updated_at": updated,
            "profile": str(data.get("profile") or ""),
            "title_custom": bool(data.get("title_custom")),
            "messages": messages}


def load_thread_records(root):
    """The project's saved threads, newest first, capped at MAX_THREADS."""
    try:
        paths = list(threads_dir(root).glob("*.json"))
    except OSError:
        return []
    records = [rec for rec in (read_thread_file(p) for p in paths) if rec]
    records.sort(key=lambda r: r["updated_at"], reverse=True)
    return records[:MAX_THREADS]


class Bus:
    """A bounded, replayable event stream. Each event gets an increasing `seq`
    (kept apart from payload fields such as a message `id`); a consumer asks
    for everything after the seq it has seen."""

    def __init__(self, cap: int = 4000):
        self._cond = threading.Condition()
        self._events: deque[dict] = deque(maxlen=cap)
        self.seq = 0
        self.closed = False

    def publish(self, type_: str, **data) -> int:
        with self._cond:
            self.seq += 1
            self._events.append({**data, "seq": self.seq, "type": type_})
            self._cond.notify_all()
            return self.seq

    def wait(self, after: int, timeout: float = 15.0) -> tuple[list[dict], bool]:
        """Events newer than `after`, blocking up to `timeout` seconds when
        there are none. The flag is True when `after` fell off the buffer, so
        the consumer must resynchronise from a snapshot."""
        with self._cond:
            if not self.closed and self.seq <= after:
                self._cond.wait(timeout)
            if not self._events:
                return [], False
            oldest = self._events[0]["seq"]
            if after < oldest - 1:
                return [], True
            return [e for e in self._events if e["seq"] > after], False

    def close(self) -> None:
        with self._cond:
            self.closed = True
            self._cond.notify_all()


class Tee:
    """Stands in for sys.stdout/sys.stderr: everything still reaches the real
    stream (the terminal that launched the GUI) and also feeds the page's
    terminal pane."""

    def __init__(self, real, sink):
        self._real = real
        self._sink = sink

    def write(self, s):
        try:
            n = self._real.write(s)
        except (OSError, ValueError):
            n = len(s) if isinstance(s, str) else 0
        if isinstance(s, str) and s:
            try:
                self._sink(s)
            except Exception:
                pass
        return n

    def flush(self):
        try:
            self._real.flush()
        except (OSError, ValueError):
            pass

    def __getattr__(self, name):
        return getattr(self._real, name)


def args_for(root_str: str, profile: str | None) -> SimpleNamespace:
    """Stand-in for the argparse Namespace that cli.py's _config/_matcher/
    _root/_refresh helpers expect."""
    return SimpleNamespace(
        paths=[root_str], root=root_str, config=None, profile=profile or None,
        exclude=[], include=[], ext=[], no_refs=False, no_gitignore=False,
        url=None, model=None, steps=None,
    )


INDEX_SUBPATH = Path(".pedacito") / "index.json"


def detect_index_meta(root) -> dict:
    """Read just enough of `.pedacito/index.json` to describe it (count of
    files and chunks, modified time) without building the in-memory Index.
    Runs off the STATE lock; a missing or unreadable file is simply 'no
    index', since `jobs.open_project` still loads the full index later."""
    meta = {"indexed": False, "files": 0, "chunks": 0, "time": 0.0}
    if root is None:
        return meta
    p = Path(root) / INDEX_SUBPATH
    try:
        data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        if int(data.get("version", 1)) > INDEX_VERSION:
            return meta          # unusable: same verdict as Index.load
        chunks = data.get("chunks") or []
        names = set(data.get("file_hashes") or {})
        names.update(c.get("file") for c in chunks
                     if isinstance(c, dict) and c.get("file"))
        names.discard(None)
        meta = {"indexed": True, "files": len(names), "chunks": len(chunks),
                "time": p.stat().st_mtime}
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return meta


def load_recent() -> list[str]:
    """The most-recently-opened project paths, newest first."""
    try:
        data = json.loads(RECENT_FILE.read_text(encoding="utf-8",
                                                errors="replace"))
        return [str(p) for p in data][:MAX_RECENT]
    except (OSError, ValueError):
        return []


def remember_recent(root: str) -> None:
    recent = [root] + [r for r in load_recent() if r != root]
    try:
        RECENT_FILE.parent.mkdir(parents=True, exist_ok=True)
        RECENT_FILE.write_text(json.dumps(recent[:MAX_RECENT]),
                               encoding="utf-8")
    except OSError:
        pass  # a lost recents list is not worth failing a request over


class Job:
    """The one running background job."""

    def __init__(self, jid: int, kind: str, title: str):
        self.id = jid
        self.kind = kind
        self.title = title
        self.started = time.time()
        self.thread: threading.Thread | None = None
        self.active = True
        self.cancelled = False
        self.client = None          # set by jobs.session, for Stop


class Review:
    """A set of proposed file changes waiting for the user's decision.

    `done` is set when a job thread is blocked on the answer (the loop's
    review step); a deferred review has no waiter and applies directly."""

    def __init__(self, rid: int, task: str, diffs: list, writer,
                 on_discard=None, partial_ok: bool = True):
        self.id = rid
        self.task = task
        self.diffs = diffs
        self.writer = writer
        self.on_discard = on_discard
        self.partial_ok = partial_ok
        self.outcome = ""           # "", "applied", "cancelled", "timeout"
        self.detail = ""
        self.done = threading.Event()
        self.edits: dict[str, str] = {}
        self.mid = 0                # the chat message announcing this review
        self.created = time.time()


def before_text(fd) -> str:
    """The file's text before the change, rebuilt from the diff rows."""
    return "\n".join(r.a for r in fd.rows if r.kind not in ("insert", "skip"))


def pack_rows(rows) -> list:
    """Compact JSON form of diff rows for the browser."""
    out = []
    for r in rows:
        if r.kind == "skip":
            out.append(["skip", r.hidden, r.fold])
        else:
            out.append([r.kind, r.a_no, r.a, r.b_no, r.b,
                        [list(s) for s in r.a_spans],
                        [list(s) for s in r.b_spans], r.fold])
    return out


class State:
    """Everything held between HTTP requests (one active project)."""

    def __init__(self):
        self.lock = threading.RLock()
        self.bus = Bus()
        self.root: Path | None = None
        self.args: SimpleNamespace | None = None
        self.cfg = None
        self.client = None
        self.idx = None
        self.agent = None
        self.profile_request = ""
        self.index_gen = 0
        # auto-detected `.pedacito/index.json` metadata (see `set_root`):
        # {"indexed": bool, "files": n, "chunks": n, "time": mtime}
        self.index_meta: dict = detect_index_meta(self.root)

        self.job: Job | None = None
        self._job_counter = 0
        self.error = ""
        self.note = ""

        self.model_checked = False
        self.model_status = ""
        self.model_ok: bool | None = None

        self.messages: list[dict] = []
        self._mid = 0
        self.term = ""

        # Per-project chat threads (see `save_active_thread`). `messages` IS
        # the active record's list while a thread is attached, so every turn
        # that lands in the transcript lands in the thread too.
        self.threads: dict[str, dict] = {}
        self.thread_id = ""
        self._thread_io = threading.Lock()
        self._last_thread_save = 0.0
        self._load_threads()

        self.review: Review | None = None
        self._rid = 0
        self.apply_lock = threading.Lock()

        # Live Permission Broker (Sprint L): a gated tool call (web fetch,
        # pip install, unapproved shell command, sensitive file read) parks
        # its worker thread here until the browser modal answers, the
        # request matches an "Allow for Run" pattern, or it times out into a
        # clean denial. `_perm_slots` maps request id -> waiter;
        # `pending_permission` is the modal's payload (the oldest waiter).
        self.pending_permission: dict | None = None
        self._perm_slots: dict[str, dict] = {}
        self._perm_seq = 0
        # "Allow for Run" grants, cleared when a new job claims the slot.
        self.run_allow: set[tuple[str, str]] = set()
        # "Always Allow" for categories without a config knob (sensitive
        # file paths): remembered for the life of this GUI process.
        self.session_allow: set[tuple[str, str]] = set()

        self.steer_queue: queue.Queue[str] = queue.Queue()
        self._qid = 0
        self._answers: dict[int, dict] = {}
        self.question_mid = 0

        self.last_backup_dir: Path | None = None
        self.last_auto: dict = {}
        self.plan_path: Path | None = None
        # Most recent built-in benchmark result (see pedacito.benchmarks and
        # gui.jobs.run_benchmark). Kept in memory so the Benchmarks dialog can
        # show the scorecard and the Open Generated Project button without a
        # second model call.
        self.benchmark: dict | None = None

        # Engine V2 telemetry for the ACTIVITY & PLAN panel (see
        # `note_event` and `engine`): the live phase, the last gather step
        # count, and the most recent harness tool actions.
        self.phase = ""
        self.gather_steps = 0
        self.actions: deque[dict] = deque(maxlen=60)

    # --- project & index auto-detect --------------------------------------------

    def set_root(self, root: Path | str | None) -> None:
        """Adopt a project folder and immediately look for an existing
        `.pedacito/index.json`, so `/api/state` reports the index (count and
        timestamp) the moment the switch starts -- not only after the whole
        open job (config, model probe, index load) has finished. The disk
        read happens outside the lock; only the swap holds it.

        Chat threads follow the project: the outgoing thread is flushed to
        the folder being left, then that folder's saved threads are loaded
        and the most recently updated one becomes the active chat. Opening
        the same root again (a profile switch) keeps the current thread."""
        meta = detect_index_meta(root)
        new_root = Path(root) if root is not None else None
        with self.lock:
            same_root = new_root == self.root
        if not same_root:
            self.save_active_thread(force=True)   # flush to the folder being left
        records = load_thread_records(new_root) if new_root else []
        with self.lock:
            self.root = new_root
            self.index_meta = meta
            if same_root:
                return
            self.threads = {r["id"]: r for r in records}
            if records:
                self._attach_thread(records[0])
            else:
                rec = self._new_thread_locked()
                self.threads[rec["id"]] = rec
                self._attach_thread(rec)

    def apply_index(self, idx) -> None:
        """Refresh the reported index metadata after `STATE.idx` changed.
        Call with the new (or None) index; disk stat runs outside the lock."""
        with self.lock:
            root = self.root
        if idx is not None:
            time = 0.0
            if root is not None:
                try:
                    time = (Path(root) / INDEX_SUBPATH).stat().st_mtime
                except OSError:
                    pass
            meta = {"indexed": True, "files": len(idx.files),
                    "chunks": len(idx.chunks), "time": time}
        else:
            meta = detect_index_meta(root)
        with self.lock:
            self.index_meta = meta

    # --- chat transcript ----------------------------------------------------

    def add_message(self, role: str, text: str = "", **extra) -> dict:
        with self.lock:
            self._mid += 1
            msg = {"id": self._mid, "role": role, "text": text,
                   "time": time.time(), **extra}
            self.messages.append(msg)
            del self.messages[:-MAX_MESSAGES]
            snap = dict(msg)
            rec = self.threads.get(self.thread_id)
            if role == "user" and rec is not None and \
                    not rec.get("title_custom") and \
                    not any(m["role"] == "user" for m in self.messages[:-1]):
                title = thread_title_from(text)
                if title:
                    rec["title"] = title
        self.bus.publish("msg_new", msg=snap)
        # a whole new message is always worth persisting; only streaming
        # appends (`append_message`) ride the throttle
        self.save_active_thread(force=True)
        return snap

    def append_message(self, mid: int, text: str) -> None:
        if not text:
            return
        with self.lock:
            for m in reversed(self.messages):
                if m["id"] == mid:
                    m["text"] += text
                    break
            else:
                return
        self.bus.publish("msg_append", id=mid, text=text)
        self.save_active_thread()          # throttled: streaming appends are many

    def set_message(self, mid: int, **fields) -> None:
        with self.lock:
            for m in reversed(self.messages):
                if m["id"] == mid:
                    m.update(fields)
                    break
            else:
                return
        self.bus.publish("msg_set", id=mid, fields=fields)
        self.save_active_thread()

    def message_text(self, mid: int) -> str:
        """The current text of one message ("" when it is gone)."""
        with self.lock:
            for m in reversed(self.messages):
                if m["id"] == mid:
                    return str(m.get("text") or "")
        return ""

    def messages_snapshot(self) -> list[dict]:
        """A consistent, JSON-safe copy of the chat transcript."""
        with self.lock:
            return [dict(m) for m in self.messages]

    def clear_messages(self) -> None:
        """Empty the current conversation. The thread file stays (this is the
        same chat, restarted from a blank slate); only switching with `new_thread`
        opens another one."""
        with self.lock:
            self.messages.clear()
            self._mid = 0
            rec = self.threads.get(self.thread_id)
            if rec is not None and not rec.get("title_custom"):
                rec["title"] = default_thread_title()
        self.bus.publish("msgs_clear")
        self.save_active_thread(force=True)
        self.bus.publish("status")

    # --- chat threads (per-project persistence) --------------------------------

    def _new_thread_locked(self, profile: str = "") -> dict:
        """Build a fresh unsaved thread record. Caller holds `self.lock`."""
        now = time.time()
        return {"id": uuid.uuid4().hex, "title": default_thread_title(),
                "created_at": now, "updated_at": now, "profile": profile,
                "title_custom": False, "messages": []}

    def _attach_thread(self, rec: dict) -> None:
        """Point the live transcript at a thread record. Caller holds `self.lock`."""
        self.thread_id = rec["id"]
        self.messages = rec["messages"]
        self._mid = max((m.get("id", 0) for m in self.messages), default=0)

    def _load_threads(self) -> None:
        """Restore the project's saved threads (used at startup); falls back
        to one fresh in-memory thread when there is nothing to load."""
        records = load_thread_records(self.root) if self.root else []
        with self.lock:
            self.threads = {r["id"]: r for r in records}
            if records:
                self._attach_thread(records[0])
            else:
                rec = self._new_thread_locked()
                self.threads[rec["id"]] = rec
                self._attach_thread(rec)

    def _thread_payload_locked(self, rec: dict) -> dict:
        """The JSON file content for one thread record. Caller holds `lock`
        (the active record's `messages` IS the live transcript list)."""
        return {
            "id": rec["id"],
            "title": rec["title"],
            "created_at": rec["created_at"],
            "updated_at": rec["updated_at"],
            "profile": rec.get("profile", ""),
            "title_custom": bool(rec.get("title_custom")),
            "messages": [{"role": m.get("role"),
                          "content": str(m.get("text") or ""),
                          "ts": m.get("time")}
                         for m in rec["messages"]],
        }

    def _write_thread_file(self, payload: dict) -> None:
        """Persist one thread payload atomically (temp file + rename, so a
        crash mid-save cannot truncate a conversation)."""
        if not self.root:
            return
        path = threads_dir(self.root) / (payload["id"] + ".json")
        with self._thread_io:            # serialize writers; never hold `lock` during I/O
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(payload["id"] + ".json.tmp")
                tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                               encoding="utf-8")
                os.replace(tmp, path)
            except OSError as exc:
                sys.stderr.write("[gui] could not save thread %s: %s\n" % (path, exc))

    def save_active_thread(self, force: bool = False) -> None:
        """Persist the active thread to `<root>/.pedacito/gui/threads/<id>.json`.
        Streaming appends are throttled to ~1/s; the save on job end is forced.
        The thread follows the project folder: switching to another project
        never moves these files."""
        rec = self.threads.get(self.thread_id)
        if rec is None or not self.root:
            return
        now = time.monotonic()
        if not force and (now - self._last_thread_save) < 1.0:
            return
        with self.lock:
            self._last_thread_save = now
            rec["updated_at"] = time.time()
            payload = self._thread_payload_locked(rec)
        self._write_thread_file(payload)

    def rename_thread(self, tid: str, title: str) -> str | None:
        """Give a thread a hand-picked name and mark it `title_custom`, so no
        automatic title update ever overwrites it again. Persists immediately
        (also for a thread that is not the active chat). Returns the new title,
        or None when the thread does not exist / the title is blank."""
        title = re.sub(r"\s+", " ", str(title or "")).strip()
        if len(title) > THREAD_RENAME_CHARS:
            title = title[:THREAD_RENAME_CHARS].rstrip()
        if not title:
            return None
        with self.lock:
            rec = self.threads.get(tid)
            if rec is None:
                return None
            rec["title"] = title
            rec["title_custom"] = True
            payload = self._thread_payload_locked(rec)
        self._write_thread_file(payload)
        self.bus.publish("status")
        return title

    def _threads_summary_locked(self) -> list:
        """The project's threads for the switcher, newest updated first.
        Caller holds `self.lock`."""
        items = [{
            "id": rec["id"], "title": rec["title"],
            "created_at": rec["created_at"], "updated_at": rec["updated_at"],
            "profile": rec.get("profile", ""),
            "title_custom": bool(rec.get("title_custom")),
            "message_count": len(rec["messages"]),
            "active": rec["id"] == self.thread_id,
        } for rec in self.threads.values()]
        items.sort(key=lambda it: it["updated_at"], reverse=True)
        return items

    def threads_summary(self) -> list:
        with self.lock:
            return self._threads_summary_locked()

    def new_thread(self, profile: str = "") -> str:
        """Start a fresh conversation in this project and return its id. The
        previous thread stays saved in the folder."""
        self.save_active_thread(force=True)
        with self.lock:
            rec = self._new_thread_locked(profile)
            self.threads[rec["id"]] = rec
            self._attach_thread(rec)
        self.save_active_thread(force=True)
        self.bus.publish("msgs_clear")
        self.bus.publish("status")
        return rec["id"]

    def select_thread(self, tid: str) -> bool:
        """Make a saved thread of this project the active conversation.
        False when there is no thread with that id."""
        with self.lock:
            if tid == self.thread_id:
                return True
        self.save_active_thread(force=True)
        with self.lock:
            rec = self.threads.get(tid)
            if rec is None:
                return False
            self._attach_thread(rec)
        self.bus.publish("msgs_clear")
        self.bus.publish("status")
        return True

    def delete_thread(self, tid: str) -> bool:
        """Forget a thread and remove its file. Deleting the active one opens
        the most recent survivor, or a fresh chat when nothing is left."""
        path = None
        with self.lock:
            rec = self.threads.pop(tid, None)
            if rec is None:
                return False
            if self.root:
                path = threads_dir(self.root) / (tid + ".json")
            was_active = tid == self.thread_id
            if was_active:
                if self.threads:
                    self._attach_thread(max(self.threads.values(),
                                            key=lambda r: r["updated_at"]))
                else:
                    fresh = self._new_thread_locked()
                    self.threads[fresh["id"]] = fresh
                    self._attach_thread(fresh)
        if path is not None:
            with self._thread_io:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    sys.stderr.write("[gui] could not delete thread %s: %s\n"
                                     % (path, exc))
        if was_active:
            self.bus.publish("msgs_clear")
        self.bus.publish("status")
        return True

    # --- terminal -----------------------------------------------------------

    def term_write(self, text: str) -> None:
        if not text:
            return
        with self.lock:
            self.term = (self.term + text)[-MAX_TERM_CHARS:]
        self.bus.publish("term", text=text)

    def term_clear(self) -> None:
        with self.lock:
            self.term = ""
        self.bus.publish("term_clear")

    # --- jobs ---------------------------------------------------------------

    def begin_job(self, kind: str, title: str) -> Job | None:
        """Claim the single job slot; None while another job is running."""
        with self.lock:
            if self.job is not None and self.job.active:
                return None
            self._job_counter += 1
            self.job = Job(self._job_counter, kind, title)
            self.error, self.note = "", ""
            # "Allow for Run" grants are scoped to one run; a leftover modal
            # from a cancelled job must never block the next one.
            self.run_allow.clear()
            stale = list(self._perm_slots)
            for pid in stale:
                slot = self._perm_slots.pop(pid)
                slot["decision"] = "deny"
                slot["event"].set()
            self.pending_permission = None
            if kind in RUN_KINDS:      # a fresh run starts the telemetry afresh
                self.phase = "gather"
                self.gather_steps = 0
                self.actions.clear()
            job = self.job
        self.bus.publish("job", job=job.id, state="start", kind=kind,
                         title=title)
        return job

    def end_job(self, job: Job, error: str = "") -> None:
        with self.lock:
            job.active = False
            if error:
                self.error = error
            if job.kind in RUN_KINDS:
                self.phase = "done"    # the panel keeps the finished run's feed
        self.bus.publish("job", job=job.id, state="end", kind=job.kind,
                         error=error)

    # --- engine telemetry ---------------------------------------------------------

    def note_event(self, kind: str, f: dict) -> None:
        """Fold one harness event (the agent's EventLog listener) into the
        activity telemetry: the live phase and a recent tool-action feed.
        Only a phase change or a recorded action pings the page."""
        action = None
        step_n = None
        if kind == "step":
            action = {"tool": str(f.get("action") or "step"),
                      "detail": str(f.get("detail") or "")[:120],
                      "ok": not f.get("error")}
            step_n = f.get("n")
        elif kind == "incremental_edit":
            tail = "" if f.get("ok") else \
                " -- " + str(f.get("reason") or "")[:80]
            action = {"tool": "edit-hunk",
                      "detail": f"{f.get('target', '?')}/{f.get('of', '?')}"
                                + tail,
                      "ok": bool(f.get("ok"))}
        elif kind == "edit_intents":
            action = {"tool": "edit-intents",
                      "detail": ", ".join(str(x)
                                          for x in (f.get("files") or []))[:120],
                      "ok": True}
        elif kind == "verify":
            action = {"tool": "verify",
                      "detail": f"round {f.get('round', '?')}: "
                                f"{f.get('errors', 0)} error(s), "
                                f"{f.get('warnings', 0)} warning(s)",
                      "ok": not f.get("errors")}
        elif kind == "tests":
            action = {"tool": "tests",
                      "detail": "passed" if f.get("ok") else "failed",
                      "ok": bool(f.get("ok"))}
        ping = kind in EVENT_PHASE or action is not None
        with self.lock:
            if kind in EVENT_PHASE:
                self.phase = EVENT_PHASE[kind]
            if step_n is not None:
                try:
                    self.gather_steps = max(self.gather_steps, int(step_n))
                except (TypeError, ValueError):
                    pass
            if action is not None:
                self.actions.append(action)
        if ping:
            self.bus.publish("status")

    def engine(self) -> dict:
        """Engine V2 state for the ACTIVITY & PLAN panel: the live phase,
        step/call counters, read spans in the ledger, the requirement ledger,
        and the recent gather/microloop actions. The shared agent is read
        best-effort: its per-run attributes survive the run, so the ledger
        keeps showing what the last GUI-driven task/explain run touched."""
        with self.lock:
            agent, client = self.agent, self.client
            job = self.job if self.job is not None and self.job.active else None
            actions = [dict(a) for a in self.actions][-12:]
            steps = self.gather_steps
            stats = dict(client.stats) if client is not None else {}
            if job is None:
                phase = self.phase or "idle"
            else:
                phase = self.phase or (
                    "gather" if job.kind in RUN_KINDS else "idle")
                site = str(getattr(client, "site", "") or "") if client else ""
                phase = SITE_PHASE.get(site, phase)
        read_spans = 0
        requirements: list[dict] = []
        try:
            ledger = getattr(agent, "_read_spans", None) or {}
            read_spans = sum(len(v) for v in ledger.values())
        except (TypeError, AttributeError, RuntimeError):
            pass
        try:
            tracker = getattr(agent, "requirements", None)
            for req in list(getattr(tracker, "items", None) or []):
                requirements.append({
                    "text": str(req.text)[:90],
                    "state": str(req.state),
                    "kind": str(req.kind),
                    "reads": len(req.read_spans),
                    "edits": len(req.edit_spans),
                    "anchors": len(req.anchors),
                })
        except (TypeError, AttributeError):
            pass
        return {"phase": phase, "steps": steps, "calls": stats.get("calls", 0),
                "read_spans": read_spans,
                "requirements": requirements[:8], "actions": actions}

    # --- live permission broker (Sprint L) ---------------------------------------

    def _refresh_pending_locked(self) -> None:
        """`pending_permission` mirrors the oldest unanswered request (the
        modal). Caller holds `self.lock`."""
        if not self._perm_slots:
            self.pending_permission = None
            return
        oldest = min(self._perm_slots.values(),
                     key=lambda s: s["req"]["created_at"])
        self.pending_permission = dict(oldest["req"])

    def request_permission(self, info: dict) -> tuple[bool, str]:
        """The broker hook (installed into `pedacito.security`). Blocks the
        calling worker thread until the user answers the GUI modal.

        Order of checks: the "Allow for Run" cache, then the live request to
        the browser (timeout `PERMISSION_TIMEOUT`, interruptible by Stop).
        Returns `(allowed, message for the model)`."""
        category = str(info.get("category") or "")
        key = str(info.get("key") or info.get("detail") or "")
        setting_key = str(info.get("setting_key") or "")
        with self.lock:
            if (category, key) in self.run_allow \
                    or (category, key) in self.session_allow:
                return True, ""
            self._perm_seq += 1
            pid = f"perm-{self._perm_seq}"
            req = {"id": pid, "category": category,
                   "action_summary": str(info.get("action_summary") or ""),
                   "detail": str(info.get("detail") or ""),
                   "risk_level": str(info.get("risk_level") or "medium"),
                   "reason": str(info.get("reason") or ""),
                   "setting_key": setting_key,
                   "key": key, "created_at": time.time()}
            slot = {"event": threading.Event(), "decision": "", "req": req}
            self._perm_slots[pid] = slot
            self._refresh_pending_locked()
            job = self.job
        self.bus.publish("permission", request=dict(req))
        self.bus.publish("status")
        deadline = time.time() + PERMISSION_TIMEOUT
        # Short waits keep the thread interruptible by Stop (same rhythm as
        # `ask_user`); a cancelled job is a clean denial, not a crash.
        while not slot["event"].wait(0.5):
            if time.time() > deadline or (job is not None and job.cancelled):
                break
        decision = slot["decision"]
        with self.lock:
            self._perm_slots.pop(pid, None)
            self._refresh_pending_locked()
        if decision == "allow_once":
            self.bus.publish("status")
            return True, ""
        if decision == "allow_run":
            with self.lock:
                self.run_allow.add((category, key))
            self.bus.publish("status")
            return True, ""
        if decision == "allow_always":
            self._always_allow(category, key, setting_key)
            self.bus.publish("status")
            return True, ""
        self.bus.publish("status")
        if not decision:               # timeout or Stop: deny in the safe direction
            self.add_message(
                "system",
                f"Permission request ({security.CATEGORY_LABELS.get(category, category)}) "
                "went unanswered and was denied; the run continues without it.",
                kind="permission_timeout")
        return False, security.DENIED_MESSAGE

    def _always_allow(self, category: str, key: str,
                      setting_key: str = "") -> None:
        """Apply an "Always Allow" decision: flip the matching config knob on
        the live config (so this run never asks again) and persist it to the
        user config when one exists. A request may name its own knob through
        `setting_key` (the web tools set `enable_web`); otherwise the knob
        comes from the category table. Categories without a knob (a specific
        sensitive file) are remembered for the GUI session."""
        setting = setting_key or {security.CAT_WEB: "allow_fetch",
                                  security.CAT_SHELL: "allow_shell",
                                  security.CAT_PACKAGE_INSTALL:
                                      "auto_install_deps",
                                  security.CAT_TEST_EXECUTION:
                                      "enable_run_test_in_gather",
                                  }.get(category)
        with self.lock:
            cfg = self.cfg
        if cfg is not None and setting:
            try:
                setattr(cfg, setting, True)
                if category == security.CAT_WEB and key:
                    hosts = tuple(getattr(cfg, "fetch_allowed_hosts", ()) or ())
                    if "*" not in hosts and key not in hosts:
                        cfg.fetch_allowed_hosts = tuple(hosts) + (key,)
            except (TypeError, AttributeError):
                pass
        changes = {setting: True} if setting else {}
        if category == security.CAT_WEB and key and cfg is not None:
            hosts = tuple(getattr(cfg, "fetch_allowed_hosts", ()) or ())
            if "*" not in hosts and key not in hosts:
                changes["fetch_allowed_hosts"] = list(hosts) + [key]
        if changes:
            try:                                    # best effort: in-memory wins
                from . import settings_api
                settings_api.apply_changes("user", None, changes, [])
            except Exception:
                pass
        with self.lock:
            self.session_allow.add((category, key))

    def decide_permission(self, pid: str, decision: str) -> tuple[bool, str]:
        """Answer a pending request from the browser. Returns (ok, detail)."""
        with self.lock:
            slot = self._perm_slots.get(pid)
            if slot is None:
                return False, "This permission request is no longer pending."
            slot["decision"] = decision
            self._perm_slots.pop(pid, None)
            self._refresh_pending_locked()
        slot["event"].set()
        self.bus.publish("permission_decided", id=pid, decision=decision)
        self.bus.publish("status")
        return True, ""

    def deny_all_permissions(self, reason: str = "") -> int:
        """Answer every open request with a denial (server shutdown, job
        teardown). Returns how many waiters were released."""
        with self.lock:
            slots = list(self._perm_slots.values())
            self._perm_slots.clear()
            self._refresh_pending_locked()
        for slot in slots:
            slot["decision"] = "deny"
            slot["event"].set()
        if slots:
            self.bus.publish("status")
        return len(slots)

    # --- questions and steering -----------------------------------------------

    def ask_user(self, text: str) -> str:
        """The model's `question` action: show it in the chat and block until
        the user answers (an empty answer lets the model choose)."""
        with self.lock:
            self._qid += 1
            qid = self._qid
            slot = {"event": threading.Event(), "answer": ""}
            self._answers[qid] = slot
            job = self.job
        msg = self.add_message("question", text, qid=qid, answered=False)
        with self.lock:
            self.question_mid = msg["id"]
        self.bus.publish("status")
        deadline = time.time() + QUESTION_TIMEOUT
        # Short waits keep the thread interruptible by Stop.
        while not slot["event"].wait(0.5):
            if time.time() > deadline or (job is not None and job.cancelled):
                break
        with self.lock:
            self._answers.pop(qid, None)
            self.question_mid = 0
        if not slot["event"].is_set():
            self.set_message(msg["id"], answered=True, answer="")
        self.bus.publish("status")
        return slot["answer"]

    def answer_question(self, qid: int, text: str) -> bool:
        with self.lock:
            slot = self._answers.get(qid)
            mid = self.question_mid
        if slot is None:
            return False
        slot["answer"] = text.strip()
        self.set_message(mid, answered=True, answer=slot["answer"])
        slot["event"].set()
        return True

    def steer(self, text: str) -> None:
        self.steer_queue.put(text.strip()[:500])
        self.add_message("user", text.strip(), kind="steer")

    # --- review -------------------------------------------------------------

    def stage_review(self, rv_args: dict) -> Review:
        with self.lock:
            self._rid += 1
            rv = Review(self._rid, **rv_args)
            self.review = rv
        n = len([d for d in rv.diffs if not d.removed_source])
        msg = self.add_message("review", rv.task, review_id=rv.id, files=n,
                               outcome="")
        rv.mid = msg["id"]
        # A plain transcript line next to the card, so the chat history of a
        # run reads: prompt -> ... -> "Staged changes across N file(s)".
        self.add_message("assistant",
                         f"Staged changes across {n} file(s) for review.",
                         kind="review_staged", review_id=rv.id, files=n)
        self.bus.publish("review", id=rv.id)
        return rv

    def review_meta(self, rv: Review | None = None) -> dict | None:
        with self.lock:
            rv = rv or self.review
            if rv is None:
                return None
            return {
                "id": rv.id, "task": rv.task, "outcome": rv.outcome,
                "detail": rv.detail, "partial_ok": rv.partial_ok,
                "pending": not rv.outcome,
                "files": [{
                    "rel": d.rel, "added": d.added, "removed": d.removed,
                    "message": d.message, "source": d.removed_source,
                    "renames_from": list(d.renames_from),
                    "edited": d.rel in rv.edits,
                } for d in rv.diffs],
            }

    def _find_diff(self, rv: Review, rel: str):
        for d in rv.diffs:
            if d.rel == rel:
                return d
        return None

    def review_file(self, rid: int, rel: str) -> dict | None:
        with self.lock:
            rv = self.review
            if rv is None or rv.id != rid:
                return None
            fd = self._find_diff(rv, rel)
            if fd is None:
                return None
            after = rv.edits.get(rel, fd.after_text)
            return {"rel": fd.rel, "rows": pack_rows(fd.rows),
                    "after": after, "added": fd.added,
                    "removed": fd.removed, "message": fd.message}

    def review_rediff(self, rid: int, rel: str, text: str) -> dict | None:
        """Store the user's hand-edited version of a proposed file and return
        its diff against the original."""
        with self.lock:
            rv = self.review
            if rv is None or rv.id != rid or rv.outcome or not rv.partial_ok:
                return None
            fd = self._find_diff(rv, rel)
            if fd is None or fd.removed_source:
                return None
            before = before_text(fd)
            text = text.replace("\r\n", "\n")
            rv.edits[rel] = text
        rows, added, removed = review_mod.build_rows(before, text)
        with self.lock:
            fd.rows, fd.added, fd.removed = rows, added, removed
            fd.after_text = text
        return {"rel": rel, "rows": pack_rows(rows), "after": text,
                "added": added, "removed": removed, "message": fd.message}

    def decide_review(self, rid: int, action: str, files: list[str] | None):
        """Apply or reject the staged review. Returns (ok, detail)."""
        with self.apply_lock:
            with self.lock:
                rv = self.review
                if rv is None or rv.id != rid or rv.outcome:
                    return False, "This review is no longer valid."
            if action == "reject":
                self._finish_review(rv, "cancelled", "No files were changed.",
                                    discard=True)
                self.add_message("assistant", "Rejected staged changes",
                                 kind="review_rejected", review_id=rv.id)
                return True, rv.detail
            if rv.partial_ok and files is not None:
                keep = set(files)
                with self.lock:
                    rv.diffs[:] = [d for d in rv.diffs
                                   if d.removed_source or d.rel in keep]
                    for d in rv.diffs:
                        if d.rel in rv.edits:
                            d.after_text = rv.edits[d.rel]
                if not any(not d.removed_source for d in rv.diffs):
                    return False, "Select at least one file to accept."
            elif rv.edits:
                for d in rv.diffs:
                    if d.rel in rv.edits:
                        d.after_text = rv.edits[d.rel]
            ok, detail = rv.writer()
            if not ok:
                return False, detail
            self._finish_review(rv, "applied", detail)
            written = [d.rel for d in rv.diffs if not d.removed_source]
            self.add_message("assistant",
                             f"Applied changes to {len(written)} file(s)",
                             kind="review_applied", review_id=rv.id,
                             files=written)
            self.reindex_written(written)
            return True, detail

    def reindex_written(self, rels: list[str]) -> None:
        """Fold freshly accepted files back into the open index so the Files
        tab and the next run see the new contents right after a review is
        applied. When a job thread was blocked on the review (Task/Edit), the
        loop re-indexes and swaps the result itself; only a review accepted
        with no job running needs to do it here."""
        if not rels:
            return
        with self.lock:
            idx, cfg = self.idx, self.cfg
            busy = self.job is not None and self.job.active
        if busy or idx is None or cfg is None:
            return
        from .. import index as index_mod  # lazy: core loads before index
        for rel in rels:
            try:
                index_mod.reindex_file(idx, rel, cfg)
            except Exception:
                pass        # a moved/deleted file re-indexes on the next run
        with self.lock:
            self.index_gen += 1
        self.apply_index(idx)
        self.bus.publish("status")

    def _finish_review(self, rv: Review, outcome: str, detail: str,
                       discard: bool = False) -> None:
        if discard and rv.on_discard is not None:
            try:
                rv.on_discard()
            except Exception as e:      # never hang the waiting job
                detail = f"Discard failed: {e}"
        with self.lock:
            rv.outcome, rv.detail = outcome, detail
        self.set_message(rv.mid, outcome=outcome, detail=detail)
        rv.done.set()
        self.bus.publish("review_done", id=rv.id, outcome=outcome)

    # --- snapshots for the browser ---------------------------------------------

    def snapshot(self) -> dict:
        with self.lock:
            return {"seq": self.bus.seq,
                    "messages": self.messages_snapshot(),
                    "term": self.term, "status": self.status()}

    def files_signature(self) -> str:
        with self.lock:
            idx = self.idx
            return f"{self.index_gen}:{len(idx.files) if idx else 0}"

    def status(self) -> dict:
        """The JSON payload the page polls to update itself."""
        with self.lock:
            cfg, client, idx = self.cfg, self.client, self.idx
            meta = self.index_meta
            indexed = idx is not None or meta["indexed"]
            job = self.job if self.job is not None and self.job.active else None
            stats = dict(client.stats) if client is not None else {}
            profile_models = {}
            if cfg:
                for name, body in (cfg.profiles or {}).items():
                    model = body.get("model") if isinstance(body, dict) else ""
                    profile_models[name] = str(model or cfg.model or "")
            return {
                "root": str(self.root) if self.root else "",
                "profile": cfg.active_profile if cfg else "",
                "profiles": sorted(cfg.profiles) if cfg else [],
                "profile_models": profile_models,
                "default_profile": cfg.default_profile if cfg else "",
                # scout/architect split (Sprint Q): the gather-phase scout
                # profile as the topbar badge should show it -- "" when unset,
                # when it names no real profile, or when it equals the active
                # profile (the architect then already runs gather itself).
                "scout_profile": (cfg.scout_profile
                                  if cfg and cfg.scout_profile
                                  and cfg.scout_profile in (cfg.profiles or {})
                                  and cfg.scout_profile != cfg.active_profile
                                  else ""),
                "model": cfg.model if cfg else "",
                "base_url": cfg.base_url if cfg else "",
                "review_default": bool(cfg.review) if cfg else True,
                "has_index": indexed,
                "indexed": indexed,
                "index_files": (len(idx.files) if idx is not None
                                else meta["files"]),
                "index_chunks": (len(idx.chunks) if idx is not None
                                 else meta["chunks"]),
                "index_time": meta["time"],
                "files_sig": self.files_signature(),
                "job": ({"id": job.id, "kind": job.kind, "title": job.title,
                         "started": job.started,
                         "cancelled": job.cancelled} if job else None),
                "error": self.error,
                "note": self.note,
                "model_checked": self.model_checked,
                "model_status": self.model_status,
                "model_ok": self.model_ok,
                "review": self.review_meta_brief(),
                "question": bool(self.question_mid),
                # live permission broker: the modal's payload (or null)
                "pending_permission": (dict(self.pending_permission)
                                        if self.pending_permission else None),
                "recent": load_recent(),
                # the chat thread currently open, plus the project's history
                # (see `_load_threads`), so the page can offer "switch back"
                "active_thread_id": self.thread_id,
                "thread_title": (self.threads.get(self.thread_id) or {}).get(
                    "title") or default_thread_title(),
                "thread_title_custom": bool((self.threads.get(self.thread_id) or
                                             {}).get("title_custom")),
                "threads": self._threads_summary_locked(),
                "needs_setup": not USER_CONFIG.exists(),
                "context_tokens": cfg.context_tokens if cfg else 0,
                "context_origin": (cfg.sources.get("context_tokens", "default")
                                   if cfg else ""),
                "json_mode": cfg.json_mode if cfg else "",
                "gui_theme": cfg.gui_theme if cfg else "slate",
                "disable_thinking": bool(cfg.disable_thinking) if cfg else False,
                "allow_steering": bool(cfg.allow_steering) if cfg else False,
                "usage": {
                    "calls": stats.get("calls", 0),
                    "tokens_in": stats.get("prompt_tokens", 0),
                    "tokens_out": stats.get("completion_tokens", 0),
                    "cache_hit": cache_hit_ratio(stats) if stats else 0.0,
                    "cost": stats.get("cost", 0.0),
                    "seconds": stats.get("seconds", 0.0),
                },
                "snapshots": [p.name for p in
                              (workspace.list_backups(self.root)[:8]
                               if self.root else [])],
                "plan_file": (cfg.auto_plan_file if cfg else "PLAN.md"),
                "last_auto": dict(self.last_auto),
                "engine": self.engine(),
            }

    def review_meta_brief(self) -> dict | None:
        """The staged review as `/api/state` reports it: enough for the page
        (and a polling client) to show the review panel and its per-file diff
        payload without a second round trip. `/api/review/file` carries the
        row-level rows; the brief carries the per-file summary."""
        rv = self.review
        if rv is None:
            return None
        return {"id": rv.id, "pending": not rv.outcome, "outcome": rv.outcome,
                "task": rv.task, "partial_ok": rv.partial_ok,
                "files": [{"rel": d.rel, "added": d.added, "removed": d.removed,
                           "source": d.removed_source,
                           "edited": d.rel in rv.edits} for d in rv.diffs]}


STATE = State()


def install_runner() -> None:
    """Route reviews opened by the loop/scaffold code into the page, blocking
    the job thread until the user decides."""
    def runner(*, page, nonce, writer, timeout, on_discard, diffs, task):
        diffs = diffs if diffs is not None else []
        rv = STATE.stage_review({
            "task": task, "diffs": diffs, "writer": writer,
            "on_discard": on_discard, "partial_ok": on_discard is None})
        job = STATE.job
        deadline = time.time() + float(timeout or 1800)
        while not rv.done.wait(0.5):
            if time.time() > deadline:
                STATE._finish_review(
                    rv, "timeout",
                    "Review window timed out; nothing was written.",
                    discard=on_discard is not None)
                break
            if job is not None and job.cancelled:
                STATE._finish_review(rv, "cancelled",
                                     "Cancelled; nothing was written.",
                                     discard=on_discard is not None)
                break
        return rv.outcome, rv.detail

    review_mod.RUNNER = runner


def uninstall_runner() -> None:
    review_mod.RUNNER = None


def install_permission_broker() -> None:
    """Route gated tool calls (web fetch, package install, unapproved shell
    command, sensitive file read) into the page: the worker thread parks in
    `STATE.request_permission` until the modal answers. Installed alongside
    the review runner while the GUI server serves; the CLI never installs a
    hook, so headless runs keep their historical "disabled in config"
    errors instead of waiting on a human who is not there."""
    security.install_permission_hook(STATE.request_permission)


def uninstall_permission_broker() -> None:
    """Deactivate the broker and release any thread still waiting (a denied
    run pivots on `DENIED_MESSAGE`; the job ends anyway during shutdown)."""
    security.install_permission_hook(None)
    STATE.deny_all_permissions("server shutting down")


def install_tee() -> tuple:
    """Mirror stdout/stderr into the terminal pane; returns the originals."""
    orig = (sys.stdout, sys.stderr)
    sys.stdout = Tee(orig[0], STATE.term_write)
    sys.stderr = Tee(orig[1], STATE.term_write)
    return orig


def uninstall_tee(orig: tuple) -> None:
    sys.stdout, sys.stderr = orig
