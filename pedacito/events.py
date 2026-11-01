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

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from . import safety

EVENT_DIR = "events"


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

    def emit(self, kind: str, _full: bool = False, **fields) -> None:
        """Record one event (`_full`: keep long strings, for answers and
        diffs). Never raises."""
        for fn in list(self.listeners):
            try:
                fn(kind, fields)
            except Exception:
                pass
        if not self.enabled:
            return
        try:
            rec = {"t": round(time.time() - self._t0, 3), "kind": kind}
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

    def _clean(self, v, cap: int = 600):
        if isinstance(v, str):
            return safety.redact(v)[:cap] if self.redact else v[:cap]
        if isinstance(v, dict):
            return {str(k): self._clean(x, cap) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [self._clean(x, cap) for x in list(v)[:50]]
        return v

    def count(self, kind: str) -> int:
        """Events of `kind` emitted by this process so far."""
        return self._counts.get(kind, 0)


_NULL = EventLog(".", enabled=False)
_registry: dict[str, EventLog] = {}


def get(root, cfg=None) -> EventLog:
    """The shared EventLog for `root` (a disabled one when `log_events` is
    off or `root` is None)."""
    if root is None:
        return _NULL
    if cfg is not None and not getattr(cfg, "log_events", True):
        return _NULL
    key = str(Path(root).resolve())
    log = _registry.get(key)
    if log is None:
        log = EventLog(key, True, bool(getattr(cfg, "redact_secrets", True)),
                       int(getattr(cfg, "keep_events", 20) or 0))
        _registry[key] = log
    return log


def attach(client, log: EventLog) -> None:
    """Route a client's per-call usage records into `log` (no-op for clients
    without an `on_call` hook, such as test doubles)."""
    if client is None or not hasattr(client, "on_call"):
        return
    client.on_call = lambda rec: log.emit("call", **rec)


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
