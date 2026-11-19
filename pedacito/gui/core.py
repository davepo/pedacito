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

from .. import config as config_mod
from .. import review as review_mod
from .. import security, workspace
from .. import update as update_mod
from ..config import USER_CONFIG
from ..events import cache_hit_ratio
from ..index import INDEX_VERSION

RECENT_FILE = USER_CONFIG.parent / "gui_recent.json"
MAX_RECENT = 8
MAX_MESSAGES = 600
MAX_TERM_CHARS = 200_000
# Live reasoning/content buffers for the stream drawer: keep the newest tail
# only, so a runaway generation cannot grow process memory without bound.
MAX_LIVE_CHARS = 200_000
# Cap for the upgrade drawer's subprocess output (pedacito.update): keep the
# newest tail so a chatty pip/pipx cannot grow process memory without bound.
MAX_UPDATE_OUTPUT = 100_000
QUESTION_TIMEOUT = 900.0
# How long a gated tool call waits for a human answer before the run is
# allowed to continue with a clean denial (the modal never leaves a worker
# hanging forever -- an unattended server must not freeze a job).
PERMISSION_TIMEOUT = 120.0

# Job kinds that drive the Engine V2 loop (and thus the activity telemetry).
RUN_KINDS = ("task", "explain", "auto", "new", "arena")
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
# after a restart. A chat opened with no project (the How and New modes) lives
# in the per-user state folder instead: <USER_STATE_DIR>/threads/global/<id>.json
# (see config.USER_STATE_DIR). A fresh thread opens under a friendly local
# timestamp ("Nov 08, 09:20 PM"); once the first request lands, the title
# becomes that request (one line, short) -- no extra model call, and the title
# stays honest because it is literally what the user asked. A manual rename
# sets `title_custom`, and then no automatic update ever touches the name
# again.
THREADS_SUBPATH = Path(".pedacito") / "gui" / "threads"
GLOBAL_THREADS_SUBDIR = Path("threads") / "global"
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
    """Where this chat scope keeps its thread files: the project's
    `.pedacito/gui/threads/`, or the per-user `threads/global/` folder when
    no project is open (root is None)."""
    if root is None:
        return config_mod.USER_STATE_DIR / GLOBAL_THREADS_SUBDIR
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
    """The scope's saved threads, newest first, capped at MAX_THREADS. `root`
    None means the per-user project-less folder."""
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
        # Instant stream cancellation: Stop sets this token, and the LLM
        # client checks it on every SSE line (and closes the live socket).
        self.cancel_event = threading.Event()
        self.stop_reason = ""
        # Live streaming buffer for the Activity drawer. `llm_stream` events
        # append deltas here; /api/state carries the snapshot on every poll.
        self.live_reasoning = ""
        self.live_content = ""
        self.live_tokens = 0
        self.live_tok_per_sec = 0.0
        self.stream_phase = ""
        self.live_updated = 0.0
        # A completed `new` job reports the project it created, so the page
        # can adopt the workspace without a manual refresh (see end_job).
        self.project_root = ""
        # The run's aggregated LLM telemetry (events.end_run), filled when
        # the worker finishes: tokens, elapsed, speed, model, provider.
        self.telemetry: dict = {}


class Review:
    """A set of proposed file changes waiting for the user's decision.

    `done` is set when a job thread is blocked on the answer (the loop's
    review step); a deferred review has no waiter and applies directly."""

    def __init__(self, rid: int, task: str, diffs: list, writer,
                 on_discard=None, partial_ok: bool = True, clause_texts=None):
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
        # R8 provenance: clause id -> text for the legend, and the clause ids
        # a partial accept left unaccepted (filled by decide_review).
        self.clause_texts: dict[str, str] = dict(clause_texts or {})
        self.unaccepted: list[str] = []


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


def profile_catalog(cfg, calibration: dict | None = None) -> list[dict]:
    """Every configured profile with the badges the topbar and profile lists
    need, sorted by name. `profiles` in `/api/state` stays a list of names for
    older clients; this is the richer companion keyed by the same names.

    `tier` prefers the explicit profile key, then the active profile's resolved
    tier, then the same name heuristic `config.load` uses. `specialties` is the
    explicit list, or what the model card detected when the config left it
    empty."""
    calibration = calibration or {}
    out: list[dict] = []
    for name in sorted((cfg.profiles or {}) if cfg is not None else {}):
        body = (cfg.profiles or {}).get(name) or {}
        body = body if isinstance(body, dict) else {}
        cal = calibration.get(name) or {}
        tier = str(body.get("tier") or "")
        if not tier and cfg is not None and name == cfg.active_profile:
            tier = str(getattr(cfg, "active_tier", "") or "")
        if not tier:
            try:
                tier = str(config_mod._guess_tier(name, body) or "")
            except Exception:
                tier = ""
        specialties = body.get("specialties") or ()
        if not specialties:
            choices = cal.get("card_choices") or {}
            specialties = choices.get("specialties") or ()
        out.append({
            "name": name,
            "model": str(body.get("model")
                         or (cfg.model if cfg is not None else "") or ""),
            "base_url": str(body.get("base_url")
                            or (cfg.base_url if cfg is not None else "") or ""),
            "tier": tier,
            "specialties": [str(t) for t in specialties],
            "is_hosted": bool(cal.get("is_hosted")),
            "calibrated": bool(cal.get("calibrated")),
            "calibrated_at": str(cal.get("calibrated_at") or ""),
            "calibrated_age_days": int(cal.get("calibrated_age_days") or 0),
            "calibrated_stale": bool(cal.get("calibrated_stale")),
            "card_score": float(cal.get("card_score") or 0.0),
            "active": bool(cfg is not None and name == cfg.active_profile),
        })
    return out


class State:
    """Everything held between HTTP requests (one active project)."""

    def __init__(self):
        self.lock = threading.RLock()
        self.bus = Bus()
        self.root: Path | None = None
        self.args: SimpleNamespace | None = None
        self.cfg = None
        # Theme fallback for a project-less session: the user config, cached
        # by (mtime, size) so a polling /api/state does not re-read the file.
        self._user_cfg = None
        self._user_cfg_stamp = None
        # The full global config for a project-less session (profiles, model,
        # toggles). Separate from `_user_cfg` because that one is cosmetic and
        # may be a partial load; this is what /api/state reports and what a
        # standalone Ask runs on. Cached by (mtime, size, selected profile).
        self._global_cfg = None
        self._global_cfg_stamp = None
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
        # Root of the last completed `new` job: /api/state carries it so a
        # page that missed the SSE job-end event still adopts the project.
        self.new_project_root = ""

        self.model_checked = False
        self.model_status = ""
        self.model_ok: bool | None = None

        self.messages: list[dict] = []
        self._mid = 0
        self.term = ""

        # Per-scope chat threads (see `save_active_thread`): the open project's
        # `.pedacito/gui/threads/`, or the per-user global folder while no
        # project is open. `messages` IS the active record's list while a
        # thread is attached, so every turn that lands in the transcript lands
        # in the thread too.
        self.threads: dict[str, dict] = {}
        self.thread_id = ""
        self._thread_io = threading.Lock()
        self._last_thread_save = 0.0
        self._load_threads()

        self.review: Review | None = None
        self._rid = 0
        self.apply_lock = threading.Lock()

        # Live Permission Broker: a gated tool call (web fetch,
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

        # Context-aware auto-update (pedacito.update): the startup check's
        # notice (None until a check lands) and the live upgrade subprocess.
        # The banner is data-driven from these; nothing here blocks startup.
        self.update: dict | None = None
        self.update_run: dict = {"running": False, "ok": None, "code": None,
                                 "output": "", "command": [], "detail": ""}

        # Engine V2 telemetry for the ACTIVITY & PLAN panel (see
        # `note_event` and `engine`): the live phase, the last gather step
        # count, and the most recent harness tool actions.
        self.phase = ""
        self.gather_steps = 0
        self.actions: deque[dict] = deque(maxlen=60)
        # Specialist routing (#50): the profile/model/tag of the most recent
        # `specialist_routed` event; cleared when a model swap hands the card
        # back to the worker's own model.
        self.specialist: dict = {}
        # Context folds in the current run (the telemetry footer's counter);
        # reset when a fresh run claims the job slot.
        self.folds = 0
        # Live LLM heartbeat: {provider, model, phase, started} while a
        # request is in flight (set by the `llm_start` event, cleared by
        # `call`); the page renders a pulsing indicator with an elapsed timer.
        self.llm_call: dict = {}

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
        the same root again (a profile switch) keeps the current thread.
        `None` switches to the project-less scope in the per-user state
        folder (`threads/global/`)."""
        meta = detect_index_meta(root)
        new_root = Path(root) if root is not None else None
        with self.lock:
            same_root = new_root == self.root
            rec = self.threads.get(self.thread_id)
        if not same_root:
            # Flush to the scope being left. An untouched project-less chat
            # stays unpersisted: opening a project must not litter the state
            # folder with empty conversations.
            if self.root is not None or (rec is not None and rec.get("messages")):
                self.save_active_thread(force=True)
        records = load_thread_records(new_root)
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

    def user_theme(self) -> str:
        """The saved `gui_theme`, even with no project open.

        A project-less GUI has no cfg until something loads one, and the
        topbar picker used to reset to the built-in default on every reload.
        Load USER_CONFIG (no hardware probe, no budget warnings, no additive
        upgrade write) and cache it by (mtime, size); `gui_theme` is purely
        cosmetic, so this never touches STATE.cfg or a running job.
        """
        with self.lock:
            cfg = self.cfg
        if cfg is not None:
            return str(getattr(cfg, "gui_theme", "") or "slate")
        try:
            st = USER_CONFIG.stat()
            stamp = (st.st_mtime_ns, st.st_size)
        except OSError:
            return "slate"
        with self.lock:
            if self._user_cfg is not None and self._user_cfg_stamp == stamp:
                return str(getattr(self._user_cfg, "gui_theme",
                                   "") or "slate")
        try:
            cfg = config_mod.load(root=None, require_hardware=False,
                                  warn_budgets=False,
                                  upgrade_user_config=False)
        except (config_mod.ConfigError, OSError):
            cfg = None
        with self.lock:
            self._user_cfg = cfg
            self._user_cfg_stamp = stamp
        return str(getattr(cfg, "gui_theme", "") or "slate") if cfg \
            else "slate"

    def set_profile(self, name: str | None) -> None:
        """Remember the profile picked in the topbar for a project-less GUI.

        With a project open `profile_request` is the profile `open_project`
        re-opens with; without one it is the profile the global config (and a
        standalone Ask) resolves. Invalidate the cached global config so the
        next /api/state poll reflects the choice immediately."""
        with self.lock:
            self.profile_request = str(name or "").strip()
            self._global_cfg_stamp = None

    def global_config(self, profile: str | None = None):
        """The user's global config, loaded even with no project open.

        `profile` defaults to the topbar selection (`profile_request`); an
        empty string means "let default_profile decide". No hardware probe, no
        budget warnings and no additive config write: /api/state polls this
        once a second. A broken config (or a profile deleted since it was
        selected) falls back to a bare Config instead of taking the GUI down;
        an open project's config still wins via `self.cfg`."""
        with self.lock:
            if self.root is not None and self.cfg is not None:
                return self.cfg
            requested = (self.profile_request if profile is None
                         else str(profile or "").strip())
        try:
            st = USER_CONFIG.stat()
            stamp = (st.st_mtime_ns, st.st_size, requested)
        except OSError:
            stamp = (0, 0, requested)
        with self.lock:
            if (self._global_cfg is not None
                    and self._global_cfg_stamp == stamp):
                return self._global_cfg
        try:
            cfg = config_mod.load(root=None, profile=requested or None,
                                  require_hardware=False, warn_budgets=False,
                                  upgrade_user_config=False)
        except (config_mod.ConfigError, OSError):
            try:
                cfg = config_mod.load(root=None, require_hardware=False,
                                      warn_budgets=False,
                                      upgrade_user_config=False)
            except (config_mod.ConfigError, OSError):
                cfg = config_mod.Config()
        with self.lock:
            self._global_cfg = cfg
            self._global_cfg_stamp = stamp
        return cfg

    def scratchpad_path(self) -> Path:
        """A per-user, project-less workspace for New-mode prompt refinement.

        Created on demand; holding it separate from the checkout keeps a
        standalone Refine Prompt (New) from describing whatever folder
        happened to be the process's cwd."""
        p = config_mod.USER_STATE_DIR / "scratchpad"
        p.mkdir(parents=True, exist_ok=True)
        return p

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

    def carry_transcript(self, messages: list[dict], title: str = "") -> None:
        """Adopt a conversation from the scope being left into the active
        thread of the scope just opened: a New build's creation turn must
        survive the project switch, so the user sees what was just built in
        the NEW project's session log. The outgoing scope keeps its own
        flushed copy (`set_root` saved it before loading this one).

        Into an empty thread the messages keep their ids (the live SSE stream
        already used them); into a thread that has history they are
        renumbered after it, so ids never collide."""
        msgs = [dict(m) for m in messages if isinstance(m, dict)]
        if not msgs:
            return
        with self.lock:
            rec = self.threads.get(self.thread_id)
            if rec is None:
                return
            if rec["messages"]:
                base = self._mid
                for m in msgs:
                    base += 1
                    m["id"] = base
                self._mid = base
            else:
                self._mid = max((m.get("id", 0) for m in msgs), default=0)
            rec["messages"].extend(msgs)      # IS self.messages (attached)
            if title and not rec.get("title_custom"):
                named = thread_title_from(title)
                if named:
                    rec["title"] = named
        self.save_active_thread(force=True)
        self.bus.publish("status")

    def reset_threads(self) -> None:
        """Forget every in-memory chat after `pedacito reset` deleted
        `.pedacito/`. A single fresh thread opens; nothing is written until
        the next message, so the reset leaves the state folder clean."""
        with self.lock:
            rec = self._new_thread_locked()
            self.threads = {rec["id"]: rec}
            self._attach_thread(rec)
        self.bus.publish("msgs_clear")
        self.bus.publish("status")

    # --- chat threads (per-project, with a project-less global scope) ----------

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
        """Restore the current scope's saved threads (used at startup);
        falls back to one fresh in-memory thread when there is nothing to
        load. With no project open the scope is the per-user global folder."""
        records = load_thread_records(self.root)
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
        crash mid-save cannot truncate a conversation). Writes to the scope
        that is current at call time: the project's folder, or the per-user
        `threads/global/` when no project is open."""
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
        """Persist the active thread to `<root>/.pedacito/gui/threads/<id>.json`,
        or to the per-user `threads/global/` folder while no project is open.
        Streaming appends are throttled to ~1/s; the save on job end is forced.
        The thread follows the scope: switching projects, or opening one from
        the project-less scope, never moves these files."""
        rec = self.threads.get(self.thread_id)
        if rec is None:
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
        """Forget a thread and remove its file (project or project-less).
        Deleting the active one opens the most recent survivor, or a fresh
        chat when nothing is left."""
        path = None
        with self.lock:
            rec = self.threads.pop(tid, None)
            if rec is None:
                return False
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
                self.folds = 0
                self.llm_call = {}
            job = self.job
        self.bus.publish("job", job=job.id, state="start", kind=kind,
                         title=title)
        return job

    def end_job(self, job: Job, error: str = "") -> None:
        with self.lock:
            job.active = False
            self.llm_call = {}      # no request outlives its job's indicator
            if job.cancelled:
                job.stop_reason = job.stop_reason or "Stopped by user"
                job.stream_phase = "stopped"
            if error:
                self.error = error
            if job.kind in RUN_KINDS:
                self.phase = "done"    # the panel keeps the finished run's feed
            if job.kind == "new" and job.project_root:
                self.new_project_root = job.project_root
        self.bus.publish("job", job=job.id, state="end", kind=job.kind,
                         error=error, project_root=job.project_root)

    # --- engine telemetry ---------------------------------------------------------

    def note_llm_stream(self, f: dict) -> None:
        """Fold one throttled `llm_stream` delta into the active job's live
        drawer and push it to SSE listeners. The full snapshot also rides on
        /api/state, so a polling client and a late page load both see it."""
        d_reasoning = str(f.get("delta_reasoning") or "")
        d_content = str(f.get("delta_content") or "")
        try:
            tokens = int(f.get("tokens_so_far") or 0)
        except (TypeError, ValueError):
            tokens = 0
        try:
            speed = float(f.get("tok_per_sec") or 0.0)
        except (TypeError, ValueError):
            speed = 0.0
        phase = str(f.get("phase") or "")
        with self.lock:
            job = self.job if self.job is not None and self.job.active else None
            if job is None:
                return
            if d_reasoning:
                job.live_reasoning = (
                    job.live_reasoning + d_reasoning)[-MAX_LIVE_CHARS:]
            if d_content:
                job.live_content = (
                    job.live_content + d_content)[-MAX_LIVE_CHARS:]
            if phase:
                job.stream_phase = phase
            if tokens:
                job.live_tokens = tokens
            if speed:
                job.live_tok_per_sec = speed
            job.live_updated = time.time()
            payload = {
                "job": job.id,
                "phase": job.stream_phase,
                "delta_reasoning": d_reasoning,
                "delta_content": d_content,
                "tokens": job.live_tokens,
                "tok_per_sec": round(job.live_tok_per_sec, 1),
                "updated": job.live_updated,
            }
        self.bus.publish("stream", **payload)

    def note_event(self, kind: str, f: dict) -> None:
        """Fold one harness event (the agent's EventLog listener) into the
        activity telemetry: the live phase and a recent tool-action feed.
        Only a phase change or a recorded action pings the page."""
        if kind == "llm_stream":
            self.note_llm_stream(f)
            return
        if kind == "llm_start":
            # Heartbeat on: a request just left for the model. The page's
            # timer ticks from `started` until the matching `call` lands.
            with self.lock:
                self.llm_call = {
                    "provider": str(f.get("provider") or ""),
                    "model": str(f.get("model") or ""),
                    "phase": str(f.get("phase") or ""),
                    "started": time.time()}
                job = (self.job if self.job is not None and self.job.active
                       else None)
                if job is not None:
                    # Each request gets a fresh drawer buffer; the previous
                    # call's reasoning/output is already in the transcript.
                    job.live_reasoning = ""
                    job.live_content = ""
                    job.live_tokens = 0
                    job.live_tok_per_sec = 0.0
                    job.stream_phase = ""
                    job.live_updated = time.time()
            self.bus.publish("status")
            return
        if kind == "call":
            with self.lock:
                self.llm_call = {}
            self.bus.publish("status")
            return
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
        elif kind == "pinned_context":
            action = {"tool": "pinned-context",
                      "detail": str(f.get("summary") or "")[:120],
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
        elif kind == "specialist_routed":
            action = {"tool": "specialist",
                      "detail": f"{f.get('profile', '?')} "
                                f"({f.get('model', '?')}) <- "
                                f"{f.get('specialty', '?')}",
                      "ok": True}
        elif kind == "model_swap":
            action = {"tool": "model-swap",
                      "detail": f"{f.get('unloaded') or '-'} -> "
                                f"{f.get('loaded', '?')} "
                                f"({f.get('elapsed_s', 0)}s)",
                      "ok": True}
        ping = kind in EVENT_PHASE or action is not None
        with self.lock:
            if kind == "fold":
                self.folds += 1
            if kind in EVENT_PHASE:
                self.phase = EVENT_PHASE[kind]
            if step_n is not None:
                try:
                    self.gather_steps = max(self.gather_steps, int(step_n))
                except (TypeError, ValueError):
                    pass
            if action is not None:
                self.actions.append(action)
            if kind == "specialist_routed":
                self.specialist = {
                    "profile": str(f.get("profile") or ""),
                    "model": str(f.get("model") or ""),
                    "specialty": str(f.get("specialty") or "")}
            elif kind == "model_swap" and self.cfg is not None \
                    and str(f.get("loaded") or "") == str(self.cfg.model or ""):
                self.specialist = {}   # the worker got its card back
        if ping:
            self.bus.publish("status")

    # --- auto-update notice & upgrade run (pedacito.update) ---------------------

    def note_update(self, info: dict | None) -> None:
        """Publish the startup/manual check's result; the banner renders from
        `status().update`. A disabled or offline check simply clears it."""
        with self.lock:
            self.update = dict(info) if info else None
            payload = dict(self.update) if self.update else None
        self.bus.publish("update", update=payload)

    def update_run_start(self, command: list[str]) -> None:
        with self.lock:
            self.update_run = {"running": True, "ok": None, "code": None,
                               "output": "", "command": list(command),
                               "detail": ""}
            payload = dict(self.update_run)
        self.bus.publish("update_run", **payload)

    def note_update_output(self, line: str) -> None:
        """One subprocess line into the upgrade drawer (capped tail buffer)."""
        with self.lock:
            out = self.update_run["output"] + line + "\n"
            if len(out) > MAX_UPDATE_OUTPUT:
                out = out[-MAX_UPDATE_OUTPUT:]
            self.update_run["output"] = out
        self.bus.publish("update_output", line=line)

    def update_run_done(self, code: int, detail: str = "") -> None:
        with self.lock:
            self.update_run["running"] = False
            self.update_run["code"] = code
            self.update_run["ok"] = code == 0
            self.update_run["detail"] = detail
            payload = dict(self.update_run)
        self.bus.publish("update_run", **payload)

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
            specialist = dict(self.specialist)
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
                "requirements": requirements[:8], "actions": actions,
                "specialist": specialist}

    # --- live permission broker ----------------------------------------------

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
        # Publish the transcript before exposing `self.review`. A polling
        # client treats "review is pending" as "the transcript already says
        # so": if the review became visible first, `/api/state` could return
        # the card without the "Staged changes..." line (the two are read by
        # separate calls in the state handler).
        with self.lock:
            self._rid += 1
            rv = Review(self._rid, **rv_args)
        n = len([d for d in rv.diffs if not d.removed_source])
        msg = self.add_message("review", rv.task, review_id=rv.id, files=n,
                               outcome="")
        rv.mid = msg["id"]
        # A plain transcript line next to the card, so the chat history of a
        # run reads: prompt -> ... -> "Staged changes across N file(s)".
        self.add_message("assistant",
                         f"Staged changes across {n} file(s) for review.",
                         kind="review_staged", review_id=rv.id, files=n)
        with self.lock:
            self.review = rv
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
                "clause_texts": rv.clause_texts,
                "unaccepted": rv.unaccepted,
                "files": [{
                    "rel": d.rel, "added": d.added, "removed": d.removed,
                    "message": d.message, "source": d.removed_source,
                    "renames_from": list(d.renames_from),
                    "edited": d.rel in rv.edits,
                    "clauses": list(getattr(d, "clauses", ())),
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
                    all_clauses = {c for d in rv.diffs if not d.removed_source
                                   for c in getattr(d, "clauses", ())}
                    kept = [d for d in rv.diffs
                            if d.removed_source or d.rel in keep]
                    if not any(not d.removed_source for d in kept):
                        return False, "Select at least one file to accept."
                    kept_clauses = {c for d in kept if not d.removed_source
                                    for c in getattr(d, "clauses", ())}
                    # R8: a clause none of whose files was accepted stays
                    # flagged, so the sign-off records what is still open.
                    rv.unaccepted = sorted(all_clauses - kept_clauses)
                    rv.diffs[:] = kept
                    for d in rv.diffs:
                        if d.rel in rv.edits:
                            d.after_text = rv.edits[d.rel]
            elif rv.edits:
                for d in rv.diffs:
                    if d.rel in rv.edits:
                        d.after_text = rv.edits[d.rel]
            ok, detail = rv.writer()
            if not ok:
                return False, detail
            if rv.unaccepted:
                detail += ("; clause(s) left unaccepted: "
                           + ", ".join(rv.unaccepted))
            self._finish_review(rv, "applied", detail)
            written = [d.rel for d in rv.diffs if not d.removed_source]
            self.add_message("assistant",
                             f"Applied changes to {len(written)} file(s)",
                             kind="review_applied", review_id=rv.id,
                             files=written)
            if rv.unaccepted:
                self.add_message(
                    "assistant",
                    "Clause(s) " + ", ".join(rv.unaccepted)
                    + " left unaccepted by the partial review.",
                    kind="review_clauses_unaccepted", review_id=rv.id,
                    clauses=list(rv.unaccepted))
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
        status = self.status()          # resolves the project-less theme first
        with self.lock:
            return {"seq": self.bus.seq,
                    "messages": self.messages_snapshot(),
                    "term": self.term, "status": status}

    def files_signature(self) -> str:
        with self.lock:
            idx = self.idx
            return f"{self.index_gen}:{len(idx.files) if idx else 0}"

    def status(self) -> dict:
        """The JSON payload the page polls to update itself."""
        theme = self.user_theme()       # disk read/cache outside the lock
        with self.lock:
            cfg, client, idx = self.cfg, self.client, self.idx
            root = self.root
        if cfg is None and root is None:
            # No project open: the topbar still offers every global profile,
            # the selected model, and the auto-route/thinking toggles.
            cfg = self.global_config()
        # Workspace security is a disk scan, so it stays outside the state
        # lock and behind a short cache; a run/ack/fix clears the cache.
        try:
            security_payload = security.security_status(
                root, cfg, use_cache=True)
        except Exception:
            security_payload = security.security_status(None, cfg)
        with self.lock:
            meta = self.index_meta
            indexed = idx is not None or meta["indexed"]
            job = self.job if self.job is not None and self.job.active else None
            if client is None and self.job is not None \
                    and self.job.client is not None:
                # The scratchpad Ask answers on its own throwaway client
                # (no project, so STATE.client stays None); the panel keeps
                # showing that client's last-call numbers after the job.
                client = self.job.client
            stats = dict(client.stats) if client is not None else {}
            # Live telemetry for the footer: the worker client's snapshot,
            # overridden by the specialist's own last-call numbers while a
            # specialist holds the card (the shared ledger keeps the session
            # totals identical either way).
            telemetry: dict = {}
            if client is not None and hasattr(client, "telemetry"):
                try:
                    telemetry = client.telemetry()
                except Exception:
                    telemetry = {}
                spec_name = str((self.specialist or {}).get("profile") or "")
                spec = None
                if cfg is not None and spec_name:
                    spec = (cfg.__dict__.get("_specialist_clients") or {}) \
                        .get(spec_name)
                if spec is not None and hasattr(spec, "telemetry"):
                    try:
                        st = spec.telemetry()
                        for k in ("last_prompt_tokens",
                                  "last_completion_tokens", "last_elapsed_s",
                                  "last_tok_per_s", "context_tokens",
                                  "context_utilization_pct",
                                  "active_model_role"):
                            telemetry[k] = st[k]
                    except Exception:
                        pass
            calibration: dict = {}
            if cfg is not None:
                try:
                    from .. import modelcard
                    calibration = modelcard.calibration_map(cfg)
                    active = cfg.active_profile or ""
                    if active not in calibration:
                        calibration[active] = modelcard.profile_calibration(cfg)
                except Exception:
                    calibration = {}
            profile_models = {}
            if cfg:
                for name, body in (cfg.profiles or {}).items():
                    model = body.get("model") if isinstance(body, dict) else ""
                    profile_models[name] = str(model or cfg.model or "")
            details = profile_catalog(cfg, calibration)
            no_project = self.root is None
            return {
                "root": str(self.root) if self.root else "",
                "profile": cfg.active_profile if cfg else "",
                "profiles": sorted(cfg.profiles) if cfg else [],
                "profile_models": profile_models,
                # Rich per-profile view (badges, model, tier, specialties)
                # for the no-project topbar and profile lists. The plain
                # `profiles` names stay for older clients.
                "profile_details": details,
                "profile_catalog": {p["name"]: p for p in details},
                "profile_tiers": {p["name"]: p["tier"] for p in details},
                "workspace_label": ("Workspace: Scratchpad (no project open)"
                                    if no_project else f"Workspace: {self.root}"),
                "scratchpad": no_project,
                "suggested_new_project": ("" if not no_project else str(
                    Path.home() / "pedacito-projects" / "my-project")),
                "default_profile": cfg.default_profile if cfg else "",
                # scout/architect split: the gather-phase scout
                # profile as the topbar badge should show it -- "" when unset,
                # when it names no real profile, or when it equals the active
                # profile (the architect then already runs gather itself).
                "scout_profile": (cfg.scout_profile
                                  if cfg and cfg.scout_profile
                                  and cfg.scout_profile in (cfg.profiles or {})
                                  and cfg.scout_profile != cfg.active_profile
                                  else ""),
                # Split-stage hybrid routing: the Options bar's Architect
                # toggle + selector read these (the model itself rides in
                # profile_models, like every other profile's model).
                "hybrid_routing": bool(cfg.hybrid_routing) if cfg else False,
                "architect_provider": (cfg.architect_provider if cfg else ""),
                "architect_model": (cfg.architect_model if cfg else ""),
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
                         "cancelled": job.cancelled,
                         "stop_reason": job.stop_reason,
                         "project_root": job.project_root,
                         # Live reasoning/output drawer snapshot. Fields stay
                         # present (empty/0) when the job has not streamed yet.
                         "live_reasoning": job.live_reasoning,
                         "live_content": job.live_content,
                         "live_tokens": job.live_tokens,
                         "live_tok_per_sec": round(job.live_tok_per_sec, 1),
                         "stream_phase": job.stream_phase,
                         "live_updated": job.live_updated} if job else None),
                # Live LLM heartbeat ({provider, model, phase, started} while
                # a request is in flight, else {}): the pulsing indicator.
                "llm_call": dict(self.llm_call),
                # The last completed New project: the page's fallback hook
                # for adopting the workspace when the SSE end event was missed.
                "new_project_root": self.new_project_root,
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
                "gui_theme": theme,
                "disable_thinking": bool(cfg.disable_thinking) if cfg else False,
                "reasoning_effort": (str(getattr(cfg, "reasoning_effort",
                                                 "default") or "default")
                                     if cfg else "default"),
                "model_class": (config_mod.capability_profile(cfg)
                                if cfg else ""),
                "reasoning_capable": bool(cfg) and (
                    config_mod.is_local_moe(cfg)
                    or str(getattr(cfg, "reasoning_effort", "default"))
                    in ("low", "medium", "high")),
                "max_concurrency": (int(getattr(cfg, "max_concurrency", 0) or 0)
                                    if cfg else 0),
                "auto_evict_servers": (bool(cfg.auto_evict_servers)
                                       if cfg else True),
                "allow_steering": bool(cfg.allow_steering) if cfg else False,
                # Non-blocking security banner: unacknowledged secrets and
                # unignored .env files, masked and fingerprint-only.
                "security": security_payload,
                # Context-aware auto-update: the check's notice (or null),
                # the live upgrade run, and the effective opt-out state.
                "update": dict(self.update) if self.update else None,
                "update_run": dict(self.update_run),
                "check_for_updates": update_mod.updates_enabled(cfg),
                # specialist routing (#50): the topbar toggle's state
                "auto_route_specialties": (bool(cfg.auto_route_specialties)
                                           if cfg else False),
                "usage": {
                    "calls": stats.get("calls", 0),
                    "tokens_in": stats.get("prompt_tokens", 0),
                    "tokens_out": stats.get("completion_tokens", 0),
                    "cache_hit": cache_hit_ratio(stats) if stats else 0.0,
                    "cost": stats.get("cost", 0.0),
                    "seconds": stats.get("seconds", 0.0),
                },
                # telemetry footer + drawer, calibration badges, and the
                # planner profile the topbar's orchestrator badge names
                "telemetry": telemetry,
                "calibration": calibration,
                "folds": self.folds,
                "planner_profile": (str((getattr(cfg, "roles", None) or {})
                                        .get("planner") or "")
                                    if cfg else ""),
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
                "clause_texts": rv.clause_texts, "unaccepted": rv.unaccepted,
                "files": [{"rel": d.rel, "added": d.added, "removed": d.removed,
                           "source": d.removed_source,
                           "edited": d.rel in rv.edits,
                           "clauses": list(getattr(d, "clauses", ()))}
                          for d in rv.diffs]}


STATE = State()


def install_runner() -> None:
    """Route reviews opened by the loop/scaffold code into the page, blocking
    the job thread until the user decides."""
    def runner(*, page, nonce, writer, timeout, on_discard, diffs, task,
               clause_texts=None):
        diffs = diffs if diffs is not None else []
        rv = STATE.stage_review({
            "task": task, "diffs": diffs, "writer": writer,
            "on_discard": on_discard, "partial_ok": on_discard is None,
            "clause_texts": clause_texts})
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
