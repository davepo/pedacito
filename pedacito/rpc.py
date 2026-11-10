"""Interactive bidirectional JSONL-RPC over stdio - the machine-client transport.

This is the same socket protocol the GUI and IDE extensions speak, but riding
on the CLI process's stdin/stdout so *any* client (editor, IDE, Web UI, another
agent) can drive one warm ``pedacito`` session over a process boundary:
newline-delimited JSON requests in, newline-delimited JSON events out, all
human-facing diagnostics on stderr so stdout stays 100% parseable.

The win over ``pedacito ask`` is what survives between requests: the project
index (symbols, call graph, embeddings), the session's token/cost ledger and
the conversation history all stay in memory, so a client can hold one process
open and issue many ``ask``/``explain`` calls with no per-request warm-up.

Wire protocol (stdin, one JSON object per line)::

    {"id": 1, "method": "ask", "params": {"task": "...", "apply": true,
                                          "refine": true}}
    {"id": 2, "method": "steer", "params": {"text": "..."}}
    {"id": 3, "method": "permission_reply",
     "params": {"request_id": "a1b2c3d4", "decision": "allow_run"}}
    {"id": 4, "method": "cancel"}
    {"id": 5, "method": "status"}

Every stdout line is a JSON event. Events streamed while a task runs carry
``"req_id"`` of the request that produced them; every request gets exactly one
terminal ``{"kind": "rpc_response", "id": ...}`` frame. ``ask`` additionally
streams the engine's live events and ends with a compact
``{"kind": "result", "req_id": ...}`` summary frame (reply, applied files,
steps, cost) before its ``rpc_response``.

Methods:

* ``ask`` / ``explain`` - run one agent turn with the warm index; ``ask``
  params: ``task`` (required), ``apply`` (default off - the safe default for
  machine clients), ``refine`` (expand the brief into a sectioned spec before
  planning), ``plan`` (true forces planning, false skips it).
* ``steer`` - push a mid-run steering message into the active task's queue;
  drained at a step boundary, which never rewrites already-sent history, so
  the provider's prompt cache is preserved.
* ``permission_reply`` - answer a ``permission_request`` event the server
  emitted on stdout (mirrors the GUI's decision broker; ``params.request_id``
  + ``params.decision`` of ``allow_once|allow_run|always_allow|deny``).
* ``cancel`` - abort the active task; the process stays warm.
* ``switch_profile`` - change the model profile for subsequent requests,
  keeping the index.
* ``status`` - active profile, scout profile, session ledger, index health.

The server runs each task on a worker thread while the main thread keeps
reading stdin - that is what makes ``steer``/``permission_reply``/``cancel``
able to reach a busy task, exactly like the GUI's socket client.
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
import uuid
from typing import Any, Callable, Optional, TextIO

from . import llm, security

# Default cap on a single stdin line - one JSON object per line; a longer
# line is refused rather than trusted to fit memory.
MAX_LINE = 10_000_000

# How long a worker waits for a permission decision before the request is
# denied as if nobody answered (same policy as the GUI's broker).
PERMISSION_TIMEOUT = 120.0

# stdin closed while a task runs: how long its worker gets to finish and
# flush its frames, and how long after a forced cancel before the process
# stops waiting on it regardless.
EOF_JOIN_GRACE = 120.0
EOF_CANCEL_GRACE = 10.0

_TASK_METHODS = ("ask", "explain")

# The decision vocabulary, identical to the GUI's modal except `always_allow`
# accepted as an alias of `allow_always` (both spellings feel natural to
# clients that read our `permission_request` events).
_DECISIONS = ("allow_once", "allow_run", "allow_always", "always_allow", "deny")

_METHODS = _TASK_METHODS + ("steer", "permission_reply", "cancel",
                            "switch_profile", "status")


def _turn_id(value: Any) -> Optional[int]:
    """The numeric turn id a client's request maps to, when it carries one.

    The GUI keys permission popups and live rows by an integer turn id; RPC
    clients either send plain int ids or prefixed ones (``"r11"``, ``"req11"``)
    - both spellings resolve here so a permission event can name its turn.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    s = str(value).strip()
    for prefix in ("r", "req", "turn"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


class RpcOut:
    """Thread-safe JSONL writer for stdout - the one stream every frame obeys.

    Doubles as ``EventLog.tee``'s consumer: the engine writes each event as a
    JSON line, we parse it back, tag it with the active request's ``req_id``
    and forward it, counting terminal ``result`` events so the server knows
    whether the engine already spoke for itself. A line we cannot round-trip
    is dropped to stderr, never written raw onto stdout.
    """

    def __init__(self, real_out: TextIO,
                 tag: Callable[[], Optional[str]] = lambda: None,
                 suppress: tuple = ()):
        self._out = real_out
        self._tag = tag
        self._suppress = tuple(suppress)
        self._lock = threading.Lock()
        self._buf = ""
        self._closed = False
        self.results = 0  # `kind:"result"` events let through - see _task_worker

    # -- EventLog.tee duck-type ------------------------------------------ #
    def write(self, s: str) -> Optional[int]:
        if self._closed:
            return 0
        n = len(s)
        self._buf += s
        while True:
            nl = self._buf.find("\n")
            if nl < 0:
                break
            line, self._buf = self._buf[:nl], self._buf[nl + 1:]
            self._emit_line(line)
        return n

    def flush(self) -> None:
        pass  # frames go out whole; the buffer holds at most one partial line

    def close(self) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def _emit_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            event = json.loads(line)
        except ValueError:
            self._to_stderr(line)
            return
        if not isinstance(event, dict):
            self._to_stderr(line)
            return
        if event.get("kind") in self._suppress:
            return
        req = self._tag()
        if req is not None:
            event["req_id"] = req
        if event.get("kind") == "result":
            self.results += 1
        self.emit(event)

    def emit(self, obj: dict) -> None:
        """Serialize + write one frame under the lock (atomic JSONL line)."""
        try:
            line = json.dumps(obj, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return
        with self._lock:
            if self._closed:
                return
            try:
                self._out.write(line + "\n")
                self._out.flush()
            except (ValueError, OSError):
                self._closed = True

    def _to_stderr(self, line: str) -> None:
        with self._lock:
            try:
                sys.stderr.write(f"[rpc:dropped] {line[:500]}\n")
            except (ValueError, OSError):
                pass


class RpcServer:
    """The stdin loop + the method dispatcher.

    All engine/CLI knowledge arrives as injected callables so the dispatch,
    tagging and permission-broker logic is unit-testable with plain fakes -
    the transport is the product here, not the task execution.
    """

    def __init__(
        self,
        *,
        run: Callable[..., Any],            # (task, apply, refine, plan) -> TaskResult
        explain: Callable[[str], str],      # symbol -> human report
        status: Callable[[], dict],         # -> status payload
        switch: Callable[[str, str], str],  # (ref, kind) -> profile name for the session
        input_stream: Optional[TextIO] = None,
        output: Optional[RpcOut] = None,
        ensure_lock: Optional[Callable[[], Any]] = None,
        save_checkpoint: Optional[Callable[[], None]] = None,
        always_apply: Optional[Callable[[str, str, str], None]] = None,
        permission_timeout: float = PERMISSION_TIMEOUT,
    ):
        self._run = run
        self._explain = explain
        self._status = status
        self._switch = switch
        self._ensure_lock = ensure_lock
        self._save_checkpoint = save_checkpoint
        self._in = input_stream if input_stream is not None else sys.stdin
        self._perm_timeout = max(1.0, float(permission_timeout))
        self.out = output or RpcOut(sys.stdout, lambda: self._tag_for(None))
        self._mu = threading.Lock()
        self._active: Optional[str] = None      # req_id of the running task
        self._seq = 0
        self._busy = threading.Event()
        self._cancel = threading.Event()
        self._client = None                     # LLMClient, for mid-stream cancel
        self._worker: Optional[threading.Thread] = None
        self._steer_q: "queue.Queue[str]" = queue.Queue()

        # -- permission broker state (mirrors gui/core.py STATE) ------------
        # one slot per outstanding request; the worker parks on its event, the
        # stdin thread fills it from permission_reply
        self._perm: dict[str, dict] = {}
        self._perm_seq = 0
        self._allowed: set = set()          # (category, key) -> session-allowed
        # glue callback: (category, key, setting_key) -> flip live cfg knobs
        self._always_apply = always_apply

    # ------------------------------------------------------------------ #
    # event tagging
    # ------------------------------------------------------------------ #
    def _tag_for(self, explicit: Optional[str]) -> Optional[str]:
        if explicit is not None:
            return explicit
        with self._mu:
            return self._active

    def _set_active(self, req_id: Optional[str]) -> None:
        with self._mu:
            self._active = req_id
        if req_id is None:
            self._busy.clear()
        else:
            self._busy.set()

    def _frame(self, req_id: Any, method: str, *, result: Any = None,
               error: Any = None) -> dict:
        frame = {"kind": "rpc_response", "id": req_id, "method": method,
                 "ok": error is None}
        if result is not None:
            frame["result"] = result
        if error is not None:
            frame["error"] = str(error)
        if result is None and error is None:
            frame["result"] = {"ok": True}
        return frame

    # ------------------------------------------------------------------ #
    # dispatch (runs on the stdin thread)
    # ------------------------------------------------------------------ #
    def dispatch(self, msg: dict) -> Optional[dict]:
        """Handle one decoded request; return the frame to emit, or None when
        the method streams its own frames from a worker thread."""
        req_id = msg.get("id")
        if req_id is None or req_id == "":
            self._seq += 1
            req_id = f"srv-{uuid.uuid4().hex[:8]}-{self._seq}"
        method = str(msg.get("method") or "")
        params = msg.get("params")
        if not isinstance(params, dict):
            params = {}

        if method in _TASK_METHODS:
            if self._busy.is_set():
                return self._frame(req_id, method,
                                   error="busy: one task at a time; steer or "
                                         "cancel the active one first")
            return self._start_task(req_id, method, params)

        if method == "switch_profile":
            if self._busy.is_set():
                return self._frame(req_id, method,
                                   error="busy: cannot switch profiles mid-task")
            ref = str(params.get("profile") or params.get("ref") or "")
            kind = str(params.get("kind") or "main")
            if kind not in ("main", "scout"):
                return self._frame(req_id, method,
                                   error="kind must be 'main' or 'scout'")
            if not ref:
                return self._frame(req_id, method, error="missing params.profile")
            try:
                name = self._switch(ref, kind)
            except SystemExit:
                return self._frame(req_id, method,
                                   error=f"unknown {kind} profile {ref!r}")
            except (ValueError, KeyError, OSError) as exc:
                return self._frame(req_id, method, error=str(exc))
            return self._frame(req_id, method,
                               result={"profile": name, "kind": kind})

        if method == "status":
            try:
                payload = dict(self._status())
            except (ValueError, KeyError, OSError) as exc:
                return self._frame(req_id, method, error=str(exc))
            payload["busy"] = self._busy.is_set()
            return self._frame(req_id, method, result=payload)

        if method == "permission_reply":
            rid = str(params.get("request_id") or params.get("id") or "")
            decision = str(params.get("decision") or "")
            with self._mu:
                slot = self._perm.get(rid)
            if slot is None:
                return self._frame(req_id, method,
                                   error=f"no pending permission request {rid!r}")
            if decision not in _DECISIONS:
                return self._frame(req_id, method,
                                   error=f"decision must be one of {list(_DECISIONS)}")
            slot["decision"] = decision
            slot["event"].set()
            return self._frame(req_id, method,
                               result={"request_id": rid, "decision": decision})

        if method == "steer":
            if not self._busy.is_set():
                return self._frame(req_id, method,
                                   error="no active task to steer")
            text = str(params.get("text") or "")
            if not text.strip():
                return self._frame(req_id, method, error="missing params.text")
            with self._mu:
                q = self._steer_q
            q.put(text)
            return self._frame(req_id, method, result={"queued": True})

        if method == "cancel":
            active = self._tag_for(None)
            self._cancel.set()
            # nudge a mid-stream LLM call so it raises CancelledError now
            # rather than at the next step boundary
            with self._mu:
                client = self._client
            if client is not None:
                try:
                    client.cancel()
                except Exception:
                    pass
            with self._mu:
                for slot in self._perm.values():
                    if not slot["decision"]:
                        slot["decision"] = "deny"
                        slot["event"].set()
            return self._frame(req_id, method,
                               result={"cancelled": True,
                                       "active": active is not None})

        return self._frame(req_id, method,
                           error=f"unknown method {method!r}; expected one of "
                                 f"{list(_METHODS)}")

    # ------------------------------------------------------------------ #
    # task worker
    # ------------------------------------------------------------------ #
    def _start_task(self, req_id: Any, method: str, params: dict) -> Optional[dict]:
        """Launch the worker thread; it emits all frames for this request.
        Returns a frame only when the request is refused before launching."""
        task = params.get("task") or params.get("message") or params.get("query")
        if method == "explain":
            task = params.get("symbol") or task
        if not str(task or "").strip():
            return self._frame(req_id, method, error="missing params.task")
        apply_ = bool(params.get("apply", False))
        refine = params.get("refine")
        plan = params.get("plan")
        if apply_ and self._ensure_lock is not None:
            try:
                self._ensure_lock()
            except SystemExit as exc:
                return self._frame(req_id, method, error=str(exc))
        self._cancel.clear()
        with self._mu:
            self._steer_q = queue.Queue()
        self._set_active(str(req_id))
        worker = threading.Thread(
            target=self._task_worker,
            args=(req_id, method, str(task), apply_, refine, plan),
            daemon=True, name="pedacito-rpc")
        self._worker_start(worker)
        return None

    def _worker_start(self, worker: threading.Thread) -> None:
        self._worker = worker
        worker.start()

    def _task_worker(self, req_id: Any, method: str, task: str,
                     apply_: bool, refine: Any, plan: Any) -> None:
        try:
            if method == "explain":
                reply = self._explain(task)
                self._set_active(None)  # untether the tee before terminal frames
                self.out.emit({"kind": "reply", "req_id": str(req_id),
                               "reply": reply})
                self.out.emit(self._frame(req_id, method, result={"reply": reply}))
                return
            before = self.out.results
            res = self._run(task, apply=apply_, refine=refine, plan=plan)
            cancelled = self._cancel.is_set()
            self._set_active(None)
            payload = self._summary(res)
            if not cancelled and payload["applied"] and self._save_checkpoint:
                self._save_checkpoint(task)
            # the engine's own `result` event already streamed through the
            # tee (tagged, full detail) unless the run ended before it spoke
            # (planned path, early error, cancel) - only duplicate then
            if self.out.results == before:
                frame = {"kind": "result", "id": req_id, "req_id": str(req_id),
                         **payload}
                if cancelled:
                    frame["cancelled"] = True
                self.out.emit(frame)
            if cancelled:
                self.out.emit(self._frame(req_id, method, error="cancelled",
                                          result=payload))
                return
            self.out.emit(self._frame(req_id, method, result=payload))
        except (llm.LLMError, llm.CancelledError, llm.BudgetExceeded) as exc:
            self._set_active(None)
            self.out.emit(self._frame(req_id, method, error=str(exc)))
        except OSError as exc:
            self._set_active(None)
            self.out.emit(self._frame(req_id, method, error=str(exc)))
        except Exception as exc:  # a crashed worker must not kill the session
            self._set_active(None)
            self.out.emit(self._frame(req_id, method,
                                      error=f"internal error: {exc!r}"))
        finally:
            self._set_active(None)
            with self._mu:
                for slot in self._perm.values():
                    if not slot["decision"]:
                        slot["decision"] = "deny"
                        slot["event"].set()

    @staticmethod
    def _summary(res: Any) -> dict:
        stats = getattr(res, "stats", None) or {}
        applied = bool(getattr(res, "applied", False))
        results = getattr(res, "results", None) or []
        changed = sorted({str(getattr(r, "path", "")) for r in results
                          if getattr(r, "applied", False)}) if applied else []
        return {
            "reply": getattr(res, "reply", None),
            "applied": applied,
            "changed": changed,
            "ok": bool(getattr(res, "ok", True)),
            "steps": getattr(res, "steps", None),
            "tests_ok": getattr(res, "tests_ok", None),
            "cost": {k: stats.get(k) for k in
                     ("calls", "prompt_tokens", "completion_tokens",
                      "cached_tokens", "cost", "seconds") if k in stats},
            "session": stats.get("session"),
        }

    # ------------------------------------------------------------------ #
    # permission broker (called from the worker thread via the tee)
    # ------------------------------------------------------------------ #
    def _permission_request(self, info: dict) -> tuple[bool, str]:
        """The `security.install_permission_hook` broker: publish one
        ``permission_request`` event, park, and answer the gated call.

        Mirrors `gui/core.py`'s `STATE.request_permission` check order: the
        run/session allow cache first, then the live ask. The worker thread
        parks; the stdin thread stays free to deliver ``permission_reply`` or
        ``cancel``. Timeout / cancel / closed stream all deny in the safe
        direction - "no answer is no".
        """
        category = str(info.get("category") or "")
        key = str(info.get("key") or info.get("detail") or "")
        setting_key = str(info.get("setting_key") or "")
        with self._mu:
            if (category, key) in self._allowed:
                return True, ""
            self._perm_seq += 1
            rid = f"perm-{self._perm_seq}"
            ev = threading.Event()
            self._perm[rid] = {"event": ev, "decision": "", "info": info}
        req_id = self._tag_for(None)
        self.out.emit({
            "kind": "permission_request", "req_id": req_id, "request_id": rid,
            "turn_id": _turn_id(req_id), "category": category, "key": key,
            "action_summary": str(info.get("action_summary") or ""),
            "detail": str(info.get("detail") or ""),
            "risk_level": str(info.get("risk_level") or "medium"),
            "reason": str(info.get("reason") or ""),
            "setting_key": setting_key,
            "timeout_seconds": self._perm_timeout,
        })
        deadline = time.time() + self._perm_timeout
        # 0.5s beats keep the wait interruptible by cancel, same rhythm as
        # the GUI's broker.
        while not ev.wait(0.5):
            if time.time() > deadline or self._cancel.is_set():
                break
        with self._mu:
            slot = self._perm.pop(rid, None)
        decision = (slot or {}).get("decision") or ""
        if decision == "allow_once":
            return True, ""
        if decision in ("allow_run", "allow_always", "always_allow"):
            with self._mu:
                self._allowed.add((category, key))
            if decision != "allow_run" and self._always_apply is not None:
                try:
                    self._always_apply(category, key, setting_key)
                except Exception:
                    pass  # in-memory grant already recorded still stands
            return True, ""
        return False, security.DENIED_MESSAGE

    def was_allowed(self, category: str, key: str) -> bool:
        """True when this (category, key) is in the session allow cache."""
        with self._mu:
            return (category, key) in self._allowed

    def set_client(self, client) -> None:
        """The live LLMClient, so `cancel` can interrupt a streamed call (and
        a profile switch can re-point it)."""
        with self._mu:
            self._client = client

    def steer_queue(self) -> "queue.Queue[str]":
        """The queue the active task drains at step boundaries; the wiring
        assigns it to `cfg._steer` when a task starts (`steer` puts into it)."""
        with self._mu:
            return self._steer_q

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #
    def serve(self) -> int:
        """Read stdin until EOF / a broken pipe. Returns the process exit code."""
        security.install_permission_hook(self._permission_request)
        self._set_active(None)
        try:
            while True:
                try:
                    line = self._in.readline()
                except (EOFError, KeyboardInterrupt, ValueError):
                    break
                if line == "":
                    break  # stdin closed
                if len(line) > MAX_LINE:
                    self.out.emit(self._frame(None, None,
                                              error=f"request line exceeds "
                                                    f"{MAX_LINE} characters"))
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError as exc:
                    self.out.emit(self._frame(None, None,
                                              error=f"invalid JSON: {exc}"))
                    continue
                if not isinstance(msg, dict):
                    self.out.emit(self._frame(None, None,
                                              error="request must be a JSON object"))
                    continue
                try:
                    frame = self.dispatch(msg)
                except Exception as exc:  # dispatcher bug != dead session
                    frame = self._frame(msg.get("id"),
                                        str(msg.get("method") or ""),
                                        error=f"internal error: {exc!r}")
                if frame is not None:
                    self.out.emit(frame)
        finally:
            # EOF: stdin is gone, but the active task's reply still belongs on
            # stdout. Let the worker run out (any permission it parks on is
            # denied -- nobody is left to answer it) and flush its frames
            # before closing the stream. A task that outstays EOF_JOIN_GRACE
            # is cancelled so the process can still exit.
            deadline = time.monotonic() + EOF_JOIN_GRACE
            forced = False
            worker = self._worker
            while worker is not None and worker.is_alive():
                with self._mu:
                    for slot in self._perm.values():
                        if not slot["decision"]:
                            slot["decision"] = "deny"
                            slot["event"].set()
                if time.monotonic() > deadline:
                    if forced:
                        break                 # wedged worker: let the process go
                    self._cancel.set()        # grace spent: wind the task down
                    forced = True
                    deadline = time.monotonic() + EOF_CANCEL_GRACE
                worker.join(timeout=0.5)
            self._cancel.set()
            with self._mu:
                for slot in self._perm.values():
                    if not slot["decision"]:
                        slot["decision"] = "deny"
                        slot["event"].set()
            self.out.close()
        return 0


__all__ = ["RpcOut", "RpcServer", "MAX_LINE", "PERMISSION_TIMEOUT",
           "EOF_JOIN_GRACE", "EOF_CANCEL_GRACE"]
