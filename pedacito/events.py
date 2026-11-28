"""Structured, local-only telemetry: one JSON object per line.

    from . import events
    log = events.get(root, cfg)              # shared per project root
    log.emit("step", action="read", file="a.py", obs_tokens=212)
    events.attach(client, log)               # per-call usage, cache hits, cost

Files live in `<root>/.pedacito/events/<stamp>-<pid>.jsonl`, one per process,
rotated by `prune()`. Nothing is sent anywhere. String fields are passed
through `safety.redact` so a secret that surfaced in a tool argument never
lands on disk. `log.txt` stays the human-readable summary; this is the
machine-readable record `pedacito eval` and `config --cost` read.
"""

from __future__ import annotations

import contextvars
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from . import safety, session_ctx

EVENT_DIR = "events"


def _clean_value(v: Any, *, redact: bool, cap: int) -> Any:
    if isinstance(v, str):
        return safety.redact(v)[:cap] if redact else v[:cap]
    if isinstance(v, dict):
        return {str(k): _clean_value(x, redact=redact, cap=cap) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean_value(x, redact=redact, cap=cap) for x in list(v)[:50]]
    return v


@dataclass(frozen=True)
class EventEnvelope:
    """One sequenced, session-tagged event on a `SessionEventBus`."""

    seq: int
    session_id: str
    t: float
    kind: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "seq": self.seq,
            "session_id": self.session_id,
            "t": self.t,
            "kind": self.kind,
        }
        for k, v in self.payload.items():
            if k not in ("seq", "session_id", "t", "kind"):
                out[k] = v
        return out


class SessionEventBus:
    """Per-session monotonic event stream with bounded replay + optional JSONL sink."""

    def __init__(
        self,
        session_id: str,
        sink: EventLog | None = None,
        *,
        cap: int = 4000,
        redact: bool = True,
    ) -> None:
        self.session_id = str(session_id)
        self._sink = sink
        self._cap = max(1, int(cap))
        self.redact = bool(redact)
        self._cond = threading.Condition()
        self._events: deque[EventEnvelope] = deque(maxlen=self._cap)
        self._seq = 0
        self._t0 = time.time()
        self.closed = False
        self._listeners: list[Callable[[EventEnvelope], None]] = []
        self._listeners_lock = threading.Lock()

    @property
    def seq(self) -> int:
        with self._cond:
            return self._seq

    def emit(self, kind: str, _full: bool = False, **fields: Any) -> EventEnvelope:
        cap = 60000 if _full else 4000
        cleaned = {
            str(k): _clean_value(v, redact=self.redact, cap=cap)
            for k, v in fields.items()
        }
        with self._cond:
            self._seq += 1
            seq = self._seq
            env = EventEnvelope(
                seq=seq,
                session_id=self.session_id,
                t=round(time.time() - self._t0, 3),
                kind=str(kind),
                payload=cleaned,
            )
            self._events.append(env)
            self._cond.notify_all()

        with self._listeners_lock:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(env)
            except Exception:
                pass

        if self._sink is not None:
            try:
                self._sink._emit_direct(
                    str(kind),
                    _full=_full,
                    session_id=self.session_id,
                    seq=seq,
                    **fields,
                )
            except Exception:
                pass
        return env

    def replay(self, after_seq: int = 0) -> tuple[list[EventEnvelope], bool]:
        """Return `(events_after, resync_needed)` without blocking."""
        with self._cond:
            if after_seq > self._seq:
                return [], True
            if not self._events:
                return [], False
            oldest = self._events[0].seq
            if after_seq < oldest - 1:
                return [], True
            return [e for e in self._events if e.seq > after_seq], False

    def wait(
        self, after_seq: int, timeout: float = 15.0
    ) -> tuple[list[EventEnvelope], bool]:
        """Block up to `timeout` seconds until events newer than `after_seq` arrive."""
        with self._cond:
            if after_seq > self._seq:
                return [], True
            if not self.closed and self._seq <= after_seq:
                self._cond.wait(timeout)
            if after_seq > self._seq:
                return [], True
            if not self._events:
                return [], False
            oldest = self._events[0].seq
            if after_seq < oldest - 1:
                return [], True
            return [e for e in self._events if e.seq > after_seq], False

    def add_listener(self, fn: Callable[[EventEnvelope], None]) -> None:
        with self._listeners_lock:
            if fn not in self._listeners:
                self._listeners.append(fn)

    def remove_listener(self, fn: Callable[[EventEnvelope], None]) -> None:
        with self._listeners_lock:
            try:
                self._listeners.remove(fn)
            except ValueError:
                pass

    def close(self) -> None:
        with self._cond:
            self.closed = True
            self._cond.notify_all()


class EventLog:
    """Append-only JSONL writer for one project root. Disabled logs are
    no-ops, and a write error disables the log rather than the run."""

    def __init__(self, root, enabled: bool = True, redact: bool = True,
                 keep: int = 20):
        self.root = Path(root)
        self.enabled = bool(enabled)
        self.redact = bool(redact)
        self.keep = int(keep or 0)
        self.path: Path | None = None
        self._lock = threading.Lock()
        self._t0 = time.time()
        self._counts: dict[str, int] = {}
        # `--jsonl`: also write every event, one per line, to this stream
        # (the real stdout, while human text is redirected to stderr).
        self.tee = None
        # Live observers, called as fn(kind, fields) for every emit (the GUI
        # streams steps from here). A failing listener is ignored.
        self.listeners: list = []

    def _open(self) -> Path | None:
        if self.path is not None:
            return self.path
        d = self.root / ".pedacito" / EVENT_DIR
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.enabled = False
            return None
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = d / f"{stamp}-{os.getpid()}.jsonl"
        if self.keep:
            prune(self.root, self.keep, current=self.path)
        return self.path

    def _emit_direct(
        self,
        kind: str,
        _full: bool = False,
        session_id: str = "",
        seq: int = 0,
        **fields,
    ) -> None:
        """Write one event directly to listeners/tee/JSONL without re-entering
        a SessionEventBus."""
        for fn in list(self.listeners):
            try:
                fn(kind, fields)
            except Exception:
                pass
        if not self.enabled:
            return
        try:
            rec: dict[str, Any] = {"t": round(time.time() - self._t0, 3), "kind": kind}
            if session_id:
                rec["session_id"] = str(session_id)
            if seq:
                rec["seq"] = int(seq)
            for k, v in fields.items():
                rec[k] = self._clean(v, 60000 if _full else 600)
            line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
            with self._lock:
                self._counts[kind] = self._counts.get(kind, 0) + 1
                if self.tee is not None:
                    try:
                        self.tee.write(line)
                        self.tee.flush()
                    except (OSError, ValueError):
                        self.tee = None
                path = self._open()
                if path is None:
                    return
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
        except Exception:
            self.enabled = False

    def emit(
        self,
        kind: str,
        _full: bool = False,
        session_id: str = "",
        seq: int = 0,
        **fields,
    ) -> None:
        """Record one event (`_full`: keep long strings, for answers and
        diffs). Never raises."""
        sess = session_ctx.current_session()
        eff_sid = str(session_id or (sess.session_id if sess is not None else ""))
        eff_seq = int(seq or 0)
        if not eff_seq and sess is not None and sess.event_bus is not None:
            bus = sess.event_bus
            if not session_id or session_id == sess.session_id:
                try:
                    env = bus.emit(kind, _full=_full, **fields)
                    eff_seq = env.seq
                    if getattr(bus, "_sink", None) is self:
                        return
                except Exception:
                    pass
        self._emit_direct(
            kind, _full=_full, session_id=eff_sid, seq=eff_seq, **fields
        )

    def _clean(self, v, cap: int = 600):
        return _clean_value(v, redact=self.redact, cap=cap)

    def count(self, kind: str) -> int:
        """Events of `kind` emitted by this process so far."""
        return self._counts.get(kind, 0)


class _SessionEventAdapter:
    """Session-tagged view over a root `EventLog` (forwarding into the active
    `SessionEventBus` when bound without duplicate JSONL sink writes)."""

    def __init__(
        self,
        log: EventLog,
        *,
        session_id: str = "",
        bus: SessionEventBus | None = None,
    ) -> None:
        object.__setattr__(self, "_log", log)
        object.__setattr__(self, "_session_id", str(session_id or ""))
        object.__setattr__(self, "_bus", bus)

    def emit(
        self,
        kind: str,
        _full: bool = False,
        session_id: str = "",
        seq: int = 0,
        **fields,
    ) -> None:
        eff_sid = str(session_id or self._session_id)
        eff_seq = int(seq or 0)
        bus = self._bus
        if bus is not None and not eff_seq:
            if not session_id or session_id == getattr(bus, "session_id", ""):
                try:
                    env = bus.emit(kind, _full=_full, **fields)
                    eff_seq = env.seq
                    if getattr(bus, "_sink", None) is self._log:
                        return
                except Exception:
                    pass
        self._log._emit_direct(
            kind, _full=_full, session_id=eff_sid, seq=eff_seq, **fields
        )

    def count(self, kind: str) -> int:
        return self._log.count(kind)

    def __getattr__(self, name: str):
        return getattr(self._log, name)

    def __setattr__(self, name: str, value) -> None:
        setattr(self._log, name, value)


_NULL = EventLog(".", enabled=False)
_registry: dict[str, EventLog] = {}
_registry_lock = threading.Lock()


def _get_root_log(root, cfg=None) -> EventLog:
    if root is None:
        return _NULL
    if cfg is not None and not getattr(cfg, "log_events", True):
        return _NULL
    key = str(Path(root).resolve())
    with _registry_lock:
        log = _registry.get(key)
        if log is None:
            log = EventLog(
                key,
                True,
                bool(getattr(cfg, "redact_secrets", True)),
                int(getattr(cfg, "keep_events", 20) or 0),
            )
            _registry[key] = log
    return log


def get(root, cfg=None, *, session_id: str | None = None) -> EventLog | _SessionEventAdapter:
    """The EventLog (or session-tagged adapter) for `root` (disabled when
    `log_events` is off or `root` is None)."""
    log = _get_root_log(root, cfg)
    sess = session_ctx.current_session()
    if session_id is not None:
        eff_sid = str(session_id)
    elif sess is not None:
        eff_sid = sess.session_id
    else:
        eff_sid = ""
    bus = None
    if sess is not None and sess.event_bus is not None:
        if not eff_sid or eff_sid == sess.session_id:
            bus = sess.event_bus
    if not eff_sid and bus is None:
        return log
    return _SessionEventAdapter(log, session_id=eff_sid, bus=bus)


def attach(client, log: EventLog | _SessionEventAdapter) -> None:
    """Route a client's per-call usage records into `log` (no-op for clients
    without an `on_call` hook, such as test doubles)."""
    if client is None or not hasattr(client, "on_call"):
        return
    client.on_call = lambda rec: log.emit("call", **rec)
    if hasattr(client, "on_start"):
        client.on_start = lambda info: log.emit("llm_start", **info)


def emit_prompt_report(log, report) -> None:
    """Emit one bounded `prompt_report` event for a PromptReport.

    The payload carries run/section IDs, counts, hashes and dispositions --
    never section text, source or credentials. Inspection failures must never
    break a run, so every error here is swallowed.
    """
    if log is None or report is None:
        return
    try:
        from . import prompt_inspect
        payload = prompt_inspect.event_payload(report)
    except Exception:
        return
    try:
        log.emit("prompt_report", **payload)
    except Exception:
        pass


class RunTelemetry:
    """Aggregated LLM usage for ONE run (a /new, /auto or task job), kept
    session/context-scoped so concurrent jobs never mix their numbers while
    sub-threads of the same session accumulate safely under a lock.
    `begin_run()` opens one in the active context; every `_account` folds into
    it via `note_call`; `end_run()` returns the summary and closes it."""

    def __init__(self):
        self._lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.reasoning_tokens = 0
        self.seconds = 0.0
        self.model = ""
        self.provider = ""

    def note(self, rec: dict) -> None:
        with self._lock:
            self.calls += 1
            self.prompt_tokens += int(rec.get("prompt_tokens") or 0)
            self.completion_tokens += int(rec.get("completion_tokens") or 0)
            self.total_tokens += int(rec.get("total_tokens") or 0)
            self.reasoning_tokens += int(rec.get("reasoning_tokens") or 0)
            self.seconds += float(rec.get("seconds") or 0.0)
            if rec.get("model"):
                self.model = str(rec["model"])
            if rec.get("provider"):
                self.provider = str(rec["provider"])

    def summary(self) -> dict:
        with self._lock:
            tok_s = (self.completion_tokens / self.seconds
                     if self.seconds > 0 else 0.0)
            return {"calls": self.calls,
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                    "total_tokens": self.total_tokens,
                    "reasoning_tokens": self.reasoning_tokens,
                    "elapsed_s": round(self.seconds, 3),
                    "tok_per_sec": round(tok_s, 1),
                    "model": self.model,
                    "provider": self.provider}


_CURRENT_RUN: contextvars.ContextVar[RunTelemetry | None] = (
    contextvars.ContextVar("pedacito_current_run", default=None)
)


def begin_run() -> RunTelemetry:
    """Open a fresh run collector in the active context/session (replacing any
    stale one)."""
    run = RunTelemetry()
    _CURRENT_RUN.set(run)
    sess = session_ctx.current_session()
    if sess is not None:
        with sess._lock:
            sess.run_telemetry = run
    return run


def current_run() -> RunTelemetry | None:
    sess = session_ctx.current_session()
    if sess is not None:
        with sess._lock:
            if sess.run_telemetry is not None:
                return sess.run_telemetry
    return _CURRENT_RUN.get()


def end_run() -> dict:
    """Close this context/session's collector and return its aggregated summary
    ({} when no run was open)."""
    run: RunTelemetry | None = None
    sess = session_ctx.current_session()
    if sess is not None:
        with sess._lock:
            run = sess.run_telemetry
            sess.run_telemetry = None
    cv_run = _CURRENT_RUN.get()
    _CURRENT_RUN.set(None)
    if run is None:
        run = cv_run
    return run.summary() if run is not None else {}


def note_call(rec: dict) -> None:
    """Fold one finished LLM call into the active run, if any. Never raises."""
    run = current_run()
    if run is not None:
        try:
            run.note(rec)
        except Exception:
            pass


def run_line(summary: dict) -> str:
    """The one-line aggregate of a finished run (`new`, `auto`, a task).
    Shared verbatim by the CLI stderr and the TUI transcript so both tell
    the same tokens/speed story. Empty string when the run made no calls."""
    if not summary or not int(summary.get("calls") or 0):
        return ""
    pt = int(summary.get("prompt_tokens") or 0)
    ct = int(summary.get("completion_tokens") or 0)
    bits = [f"{int(summary['calls'])} calls", f"{pt:,} in / {ct:,} out"]
    total = int(summary.get("total_tokens") or 0)
    reasoning = int(summary.get("reasoning_tokens") or 0)
    if total or reasoning:
        extra = f"{total:,} total" if total else ""
        if reasoning:
            extra += (", " if extra else "") + f"{reasoning:,} reasoning"
        bits.append(extra)
    bits.append(f"{float(summary.get('elapsed_s') or 0):.1f}s")
    tok_s = float(summary.get("tok_per_sec") or 0)
    if tok_s > 0:
        bits.append(f"{tok_s:.1f} tok/s")
    model = str(summary.get("model") or "")
    provider = str(summary.get("provider") or "")
    if model:
        bits.append(f"{provider}/{model}" if provider else model)
    elif provider:
        bits.append(provider)
    return "[run: " + ", ".join(bits) + "]"


def prune(root, keep: int, current: Path | None = None) -> int:
    """Delete all but the newest `keep` event files; returns how many."""
    d = Path(root) / ".pedacito" / EVENT_DIR
    if keep <= 0 or not d.is_dir():
        return 0
    files = sorted((p for p in d.glob("*.jsonl") if p != current),
                   key=lambda p: p.name, reverse=True)
    removed = 0
    for p in files[max(0, keep - (1 if current else 0)):]:
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def load(root, limit_files: int = 5) -> list[dict]:
    """Events from the newest `limit_files` files, oldest first (for eval
    reports and `config --cost`)."""
    d = Path(root) / ".pedacito" / EVENT_DIR
    if not d.is_dir():
        return []
    out: list[dict] = []
    for p in sorted(d.glob("*.jsonl"), key=lambda p: p.name)[-limit_files:]:
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
        except OSError:
            continue
    return out


def cache_hit_ratio(stats: dict) -> float:
    """Fraction of prompt tokens the server served from its cache."""
    pt = int(stats.get("prompt_tokens") or 0)
    return (int(stats.get("cached_tokens") or 0) / pt) if pt else 0.0
