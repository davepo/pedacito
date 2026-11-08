"""Shared state for the GUI server: the event bus, terminal capture, the chat
transcript, and the staged review.

Locking: every read or write of `STATE` fields goes through `STATE.lock` (an
RLock), held only around state mutations -- never around model or filesystem
work. Workers snapshot what they need, work unlocked, then swap results in.

Events reach the browser over Server-Sent Events (`Bus`). The chat transcript
and terminal text are also kept here, so a reloaded page rebuilds itself from
`snapshot()` and then follows the stream from the sequence number it holds.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

from .. import review as review_mod
from .. import workspace
from ..config import USER_CONFIG
from ..events import cache_hit_ratio
from ..index import INDEX_VERSION

RECENT_FILE = USER_CONFIG.parent / "gui_recent.json"
MAX_RECENT = 8
MAX_MESSAGES = 600
MAX_TERM_CHARS = 200_000
QUESTION_TIMEOUT = 900.0


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

        self.review: Review | None = None
        self._rid = 0
        self.apply_lock = threading.Lock()

        self.steer_queue: queue.Queue[str] = queue.Queue()
        self._qid = 0
        self._answers: dict[int, dict] = {}
        self.question_mid = 0

        self.last_backup_dir: Path | None = None
        self.last_auto: dict = {}
        self.plan_path: Path | None = None

    # --- project & index auto-detect --------------------------------------------

    def set_root(self, root: Path | str | None) -> None:
        """Adopt a project folder and immediately look for an existing
        `.pedacito/index.json`, so `/api/state` reports the index (count and
        timestamp) the moment the switch starts -- not only after the whole
        open job (config, model probe, index load) has finished. The disk
        read happens outside the lock; only the swap holds it."""
        meta = detect_index_meta(root)
        with self.lock:
            self.root = Path(root) if root is not None else None
            self.index_meta = meta

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
        self.bus.publish("msg_new", msg=snap)
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

    def set_message(self, mid: int, **fields) -> None:
        with self.lock:
            for m in reversed(self.messages):
                if m["id"] == mid:
                    m.update(fields)
                    break
            else:
                return
        self.bus.publish("msg_set", id=mid, fields=fields)

    def clear_messages(self) -> None:
        with self.lock:
            self.messages = []
        self.bus.publish("msgs_clear")

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
            job = self.job
        self.bus.publish("job", job=job.id, state="start", kind=kind,
                         title=title)
        return job

    def end_job(self, job: Job, error: str = "") -> None:
        with self.lock:
            job.active = False
            if error:
                self.error = error
        self.bus.publish("job", job=job.id, state="end", kind=job.kind,
                         error=error)

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
        msg = self.add_message(
            "review", rv.task, review_id=rv.id,
            files=len([d for d in rv.diffs if not d.removed_source]),
            outcome="")
        rv.mid = msg["id"]
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
            return True, detail

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
            return {"seq": self.bus.seq, "messages": list(self.messages),
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
                "recent": load_recent(),
                "needs_setup": not USER_CONFIG.exists(),
                "context_tokens": cfg.context_tokens if cfg else 0,
                "context_origin": (cfg.sources.get("context_tokens", "default")
                                   if cfg else ""),
                "json_mode": cfg.json_mode if cfg else "",
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
            }

    def review_meta_brief(self) -> dict | None:
        rv = self.review
        if rv is None:
            return None
        return {"id": rv.id, "pending": not rv.outcome, "outcome": rv.outcome}


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


def install_tee() -> tuple:
    """Mirror stdout/stderr into the terminal pane; returns the originals."""
    orig = (sys.stdout, sys.stderr)
    sys.stdout = Tee(orig[0], STATE.term_write)
    sys.stderr = Tee(orig[1], STATE.term_write)
    return orig


def uninstall_tee(orig: tuple) -> None:
    sys.stdout, sys.stderr = orig
