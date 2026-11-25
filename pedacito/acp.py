"""Agent Client Protocol (ACP) bridge over stdin/stdout.

ACP is JSON-RPC 2.0 for editors and agent hosts: the client asks the agent to
create a session, then sends prompts whose content blocks (text, `@file`
resources, images) stream back as ``session/update`` notifications. This module
is a *translation layer only*: every task runs through the exact same
``rpc.RpcServer`` used by ``pedacito rpc`` - same warm index, same permission
broker, same steering/cancel machinery, same result summary - so the two
transports can never drift apart.

Wire shape (one JSON object per line; stdout carries nothing else)::

    client -> agent   {"jsonrpc":"2.0","id":1,"method":"initialize","params":{...}}
    agent  -> client   {"jsonrpc":"2.0","id":1,"result":{...}}
    client -> agent   {"jsonrpc":"2.0","id":2,"method":"session/new",
                       "params":{"cwd":"/project"}}
    client -> agent   {"jsonrpc":"2.0","id":3,"method":"session/prompt",
                       "params":{"sessionId":"...","prompt":[
                         {"type":"text","text":"fix the retry"},
                         {"type":"resource","uri":"file:///project/a.py"},
                         {"type":"image","data":"<base64>","mimeType":"image/png"}]}}
    agent  -> client   {"jsonrpc":"2.0","method":"session/update",
                        "params":{"sessionId":"...","update":{
                          "sessionUpdate":"tool_call", ...}}}
    agent  -> client   {"jsonrpc":"2.0","id":"acp-perm-1",
                        "method":"session/request_permission","params":{...}}
    client -> agent   {"jsonrpc":"2.0","id":"acp-perm-1",
                       "result":{"outcome":{"outcome":"selected",
                                            "optionId":"allow_once"}}}
    agent  -> client   {"jsonrpc":"2.0","id":3,"result":{"stopReason":"end_turn"}}

Design notes:

* The bridge owns stdin and feeds RPC requests into a queue-backed stream;
  ``RpcServer.serve()`` remains the one dispatcher, worker-thread manager and
  permission broker. Nothing about task execution is reimplemented here.
* A prompt runs on a watcher thread so the read loop stays free for
  ``session/cancel`` and permission responses while a task is in flight.
* ``session/request_permission`` is the bidirectional half: the RPC broker's
  ``permission_request`` frame becomes a JSON-RPC *request* to the client, and
  the client's response is turned back into ``permission_reply``. A timeout or
  cancel still denies in the safe direction, exactly like the GUI.
* Human diagnostics go to stderr; EOF closes stdin, drains the active worker
  and exits 0 without leaving a pipe wait behind on Windows.
"""
from __future__ import annotations

import json
import queue
import re
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Optional, TextIO

from . import __version__
from . import rpc, security

JSONRPC = "2.0"
PROTOCOL_VERSION = 1

#: ACP option ids the bridge offers, mapped to the RPC broker's vocabulary.
PERMISSION_OPTIONS = (
    {"optionId": "allow_once", "name": "Allow once", "kind": "allow_once"},
    {"optionId": "allow_always", "name": "Always allow", "kind": "allow_always"},
    {"optionId": "reject_once", "name": "Reject", "kind": "reject_once"},
)

#: Cap on one embedded file so a 100 MB log cannot be pasted into a prompt.
MAX_RESOURCE_CHARS = 60_000
MAX_IMAGES = 4

# `@src/a.py` mentions inside a text block (same convention the engine's
# own gather phase understands). A capture without an extension is harmless:
# it only hydrates when the token names an existing project file.
_MENTION_RE = re.compile(r"(?<![\w@])@([A-Za-z0-9_./\\-]+)")

_TRAILING = ".,:;!?)]}'\""


class AcpError(Exception):
    """A JSON-RPC error with a protocol code (invalid params, unknown method)."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = int(code)
        self.message = str(message)
        self.data = data


class _Prompt:
    """One in-flight ``session/prompt`` request."""

    __slots__ = ("jsonrpc_id", "session_id", "rpc_id", "event", "frame",
                 "streamed")

    def __init__(self, jsonrpc_id: Any, session_id: str, rpc_id: str):
        self.jsonrpc_id = jsonrpc_id
        self.session_id = session_id
        self.rpc_id = rpc_id
        self.event = threading.Event()
        self.frame: Optional[dict] = None
        self.streamed = False


class AcpOut(rpc.RpcOut):
    """An ``RpcOut`` that translates RPC frames into ACP notifications.

    Reuses the base class's line buffering, JSON validation, result counting
    and close semantics; only the final ``emit`` step changes. Engine events
    arrive through ``write()`` (the EventLog tee), terminal frames through
    ``emit()`` - both land in ``AcpServer._on_frame``.
    """

    def __init__(self, bridge: "AcpServer", real_out: TextIO):
        super().__init__(real_out, lambda: None)
        self._bridge = bridge

    def emit(self, obj: dict) -> None:  # noqa: D401 - interface method
        self._bridge._on_frame(obj)


class _QueueInput:
    """A queue-backed ``stdin`` stand-in for ``RpcServer.serve()``."""

    def __init__(self):
        self._q: "queue.Queue[str]" = queue.Queue()

    def feed(self, obj: dict) -> None:
        self._q.put(json.dumps(obj, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._q.put("")

    def readline(self) -> str:  # noqa: D401 - file protocol
        return self._q.get()


class AcpServer:
    """JSON-RPC 2.0 ACP front end wrapping one warm ``rpc.RpcServer``."""

    def __init__(
        self,
        server: rpc.RpcServer,
        *,
        project_root: Optional[Path] = None,
        input_stream: Optional[TextIO] = None,
        output_stream: Optional[TextIO] = None,
        set_profile: Optional[Callable[[str], Any]] = None,
        set_execution_mode: Optional[Callable[[str], Any]] = None,
        on_images: Optional[Callable[[list], None]] = None,
        version: str = __version__,
    ):
        self.server = server
        self._in = input_stream if input_stream is not None else sys.stdin
        self._out = output_stream if output_stream is not None else sys.stdout
        self.version = str(version)
        self.project_root = (Path(project_root).expanduser().resolve()
                             if project_root is not None else None)
        self._set_profile = set_profile
        self._set_execution_mode = set_execution_mode
        self._on_images = on_images

        self._rpc_in = _QueueInput()
        server.bind_input(self._rpc_in)
        self.out = AcpOut(self, self._out)
        server.out = self.out

        self._write_lock = threading.Lock()
        self._lock = threading.RLock()
        self._sessions: dict[str, dict] = {}
        self._prompts: dict[str, _Prompt] = {}
        self._active: Optional[_Prompt] = None
        self._last_session: str = ""
        self._pending_perms: dict[str, tuple[str, str]] = {}
        self._perm_seq = 0
        self._req_seq = 0
        self._watchers: list[threading.Thread] = []
        self._rpc_thread: Optional[threading.Thread] = None
        self._initialized = False

    # ------------------------------------------------------------------ #
    # output
    # ------------------------------------------------------------------ #
    @property
    def closed(self) -> bool:
        return self.out.closed

    def _write(self, obj: dict) -> None:
        """Serialize exactly one JSON-RPC frame, atomically."""
        try:
            line = json.dumps(obj, ensure_ascii=False, default=str) + "\n"
        except (TypeError, ValueError):
            return
        with self._write_lock:
            if self.out.closed:
                return
            try:
                self._out.write(line)
                self._out.flush()
            except (ValueError, OSError):
                self.out.close()

    def _notify(self, method: str, params: dict) -> None:
        self._write({"jsonrpc": JSONRPC, "method": method, "params": params})

    def _respond(self, req_id: Any, result: Any) -> None:
        self._write({"jsonrpc": JSONRPC, "id": req_id, "result": result})

    def _fail(self, req_id: Any, code: int, message: str,
              data: Any = None) -> None:
        err: dict = {"code": int(code), "message": str(message)}
        if data is not None:
            err["data"] = data
        self._write({"jsonrpc": JSONRPC, "id": req_id, "error": err})

    def _session_update(self, session_id: str, update: dict) -> None:
        self._notify("session/update",
                     {"sessionId": session_id, "update": update})

    # ------------------------------------------------------------------ #
    # ACP capabilities
    # ------------------------------------------------------------------ #
    def _initialize(self, params: dict) -> dict:
        self._initialized = True
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "agentInfo": {"name": "pedacito", "version": self.version},
            "agentCapabilities": {
                "loadSession": False,
                "promptCapabilities": {
                    "text": True,
                    "resources": True,
                    "embeddedContext": True,
                    "image": True,
                    "audio": False,
                },
                "mcpCapabilities": {"http": False, "sse": False},
                "streaming": True,
                "permissionRequests": True,
            },
        }

    # ------------------------------------------------------------------ #
    # sessions
    # ------------------------------------------------------------------ #
    def _session_new(self, params: dict) -> dict:
        cwd = params.get("cwd") or params.get("project")
        root = self.project_root
        if cwd:
            try:
                candidate = Path(str(cwd)).expanduser().resolve()
            except OSError as exc:
                raise AcpError(-32602, f"invalid cwd {cwd!r}: {exc}") from exc
            if not candidate.is_dir():
                raise AcpError(-32602, f"cwd is not a directory: {cwd}")
            if root is not None and candidate != root:
                raise AcpError(
                    -32002,
                    f"this agent is bound to {root}; start another "
                    f"`pedacito acp` process for {candidate}")
            root = candidate
        if root is None:
            root = Path.cwd().resolve()

        profile = str(params.get("profile") or "").strip()
        if profile:
            self._apply_profile(profile)
        mode = str(params.get("executionMode")
                   or params.get("execution_mode") or "").strip().lower()
        if mode:
            self._apply_execution_mode(mode)

        sid = uuid.uuid4().hex
        with self._lock:
            self._sessions[sid] = {"id": sid, "root": root,
                                   "profile": profile or None,
                                   "execution_mode": mode or None}
            self._last_session = sid
        return {"sessionId": sid}

    def _apply_profile(self, profile: str) -> None:
        if self._set_profile is None:
            return
        try:
            self._set_profile(profile)
        except SystemExit as exc:
            raise AcpError(-32602, f"unknown profile {profile!r}: {exc}") from exc
        except (ValueError, KeyError, OSError, RuntimeError) as exc:
            raise AcpError(-32602, f"cannot use profile {profile!r}: {exc}") from exc

    def _apply_execution_mode(self, mode: str) -> None:
        allowed = ("two_phase", "interleaved", "auto")
        if mode not in allowed:
            raise AcpError(-32602,
                           f"executionMode must be one of {list(allowed)}")
        if self._set_execution_mode is None:
            return
        try:
            self._set_execution_mode(mode)
        except (ValueError, KeyError, OSError, RuntimeError) as exc:
            raise AcpError(-32602,
                           f"cannot use executionMode {mode!r}: {exc}") from exc

    def _session(self, params: dict) -> dict:
        sid = params.get("sessionId") or params.get("session_id") or ""
        session = self._sessions.get(str(sid))
        if session is None:
            raise AcpError(-32001, f"session not found: {sid!r}")
        return session

    # ------------------------------------------------------------------ #
    # prompt content
    # ------------------------------------------------------------------ #
    def _build_prompt(self, root: Path, blocks: Any) -> tuple[str, list]:
        if isinstance(blocks, str):
            blocks = [{"type": "text", "text": blocks}]
        if not isinstance(blocks, list) or not blocks:
            raise AcpError(-32602,
                           "params.prompt must be a non-empty array of "
                           "content blocks")
        texts: list[str] = []
        images: list[dict] = []
        for block in blocks:
            if not isinstance(block, dict):
                raise AcpError(-32602, "each prompt content block must be an "
                                       "object")
            kind = str(block.get("type") or "text")
            if kind == "text":
                text = block.get("text")
                if not isinstance(text, str):
                    raise AcpError(-32602, "text content block needs a "
                                           "string 'text'")
                # Hydrate @mentions on the caller's own text only; an
                # embedded resource's body must not pull in further files.
                texts.append(self._hydrate_mentions(root, text))
            elif kind in ("resource", "resource_link"):
                texts.append(self._resource_block(root, block))
            elif kind == "image":
                images.append(self._image_block(block, len(images) + 1))
            else:
                raise AcpError(-32602,
                               f"unsupported content block type {kind!r}")
        if len(images) > MAX_IMAGES:
            raise AcpError(-32602,
                           f"too many image blocks ({len(images)}; max {MAX_IMAGES})")
        text = "\n\n".join(t for t in texts if t).strip()
        return text, images

    def _resource_block(self, root: Path, block: dict) -> str:
        resource = block.get("resource")
        if not isinstance(resource, dict):
            resource = block
        raw_path = (resource.get("path") or block.get("path")
                    or resource.get("uri") or block.get("uri") or "")
        embedded = resource.get("text") if "text" in resource else None
        if not raw_path:
            if isinstance(embedded, str):
                return embedded[:MAX_RESOURCE_CHARS]
            raise AcpError(-32602, "resource block needs a path or uri")
        rel, body = self._read_resource(root, raw_path, embedded)
        return f"[file: {rel}]\n{body}"

    def _read_resource(self, root: Path, raw_path: Any,
                       embedded: Any = None) -> tuple[str, str]:
        target = _uri_to_path(raw_path)
        if embedded is None:
            try:
                path = security.resolve_safe_path(root, target)
            except security.SecurityError as exc:
                raise AcpError(-32602, str(exc)) from exc
            except OSError as exc:
                raise AcpError(-32602, f"cannot read {target!r}: {exc}") from exc
            if not path.is_file():
                raise AcpError(-32602, f"resource not found: {target}")
            try:
                body = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                raise AcpError(-32602, f"cannot read {target!r}: {exc}") from exc
        else:
            if not isinstance(embedded, str):
                raise AcpError(-32602, "resource 'text' must be a string")
            try:
                path = security.resolve_safe_path(root, target)
                rel = path.relative_to(root).as_posix()
            except (security.SecurityError, OSError, ValueError) as exc:
                raise AcpError(-32602, str(exc)) from exc
            body = embedded
        body = body[:MAX_RESOURCE_CHARS]
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = path.name
        return rel, body

    def _hydrate_mentions(self, root: Path, text: str) -> str:
        """Embed `@path` mentions the same safe way resource blocks are."""
        if not text or "@" not in text:
            return text
        seen: set[str] = set()
        additions: list[str] = []
        for match in _MENTION_RE.finditer(text):
            raw = match.group(1).rstrip(_TRAILING)
            if not raw or raw in seen:
                continue
            seen.add(raw)
            try:
                rel, body = self._read_resource(root, raw, embedded=None)
            except AcpError:
                # A mention that is not a readable project file is left for
                # the engine's own @mention parser (which reports it in-turn).
                continue
            additions.append(f"[file: {rel}]\n{body}")
        if not additions:
            return text
        return text + "\n\n" + "\n\n".join(additions)

    @staticmethod
    def _image_block(block: dict, n: int) -> dict:
        data = block.get("data")
        mime = str(block.get("mimeType") or block.get("mime")
                   or "image/png").strip()
        if not isinstance(data, str) or not data.strip():
            raise AcpError(-32602, "image content block needs base64 'data'")
        from . import imagegen
        try:
            return imagegen.normalize_image_attachment({
                "name": f"acp-image-{n}",
                "mime": mime,
                "data_b64": data,
            })
        except (ValueError, TypeError) as exc:
            raise AcpError(-32602, f"invalid image block: {exc}") from exc

    # ------------------------------------------------------------------ #
    # prompt lifecycle
    # ------------------------------------------------------------------ #
    def _prompt(self, req_id: Any, params: dict) -> None:
        """Start a prompt on a watcher thread; the response is sent later."""
        session = self._session(params)
        root = session["root"]
        text, images = self._build_prompt(root, params.get("prompt"))
        if not text.strip():
            raise AcpError(-32602, "prompt is empty")
        if self._on_images is not None:
            try:
                self._on_images(images)
            except Exception as exc:       # a transport bug must not deadlock
                raise AcpError(-32602,
                               f"cannot attach images: {exc}") from exc
        with self._lock:
            self._req_seq += 1
            rpc_id = f"acp-{session['id'][:8]}-{self._req_seq}"
            prompt = _Prompt(req_id, session["id"], rpc_id)
            self._prompts[rpc_id] = prompt
            if self._active is None:
                self._active = prompt
        msg = {
            "id": rpc_id,
            "method": "ask",
            "params": {
                "task": text,
                "apply": rpc._flag(params.get("apply"), default=False),
                "events": True,
            },
        }
        watcher = threading.Thread(
            target=self._prompt_worker, args=(prompt, msg),
            daemon=True, name="pedacito-acp-prompt")
        with self._lock:
            self._watchers.append(watcher)
        watcher.start()

    def _prompt_worker(self, prompt: _Prompt, msg: dict) -> None:
        try:
            frame = self.server.dispatch(msg)
        except Exception as exc:           # dispatcher bug != dead session
            frame = {"kind": "rpc_response", "ok": False,
                     "error": f"internal error: {exc!r}"}
        if frame is not None:
            # Refused before launching (busy, missing task, ...). ACP still
            # answers with a stopReason the client can act on.
            prompt.frame = frame
            prompt.event.set()
        prompt.event.wait()
        self._finish_prompt(prompt)

    def _finish_prompt(self, prompt: _Prompt) -> None:
        with self._lock:
            self._prompts.pop(prompt.rpc_id, None)
            was_active = self._active is prompt
            if was_active:
                self._active = None
                self._last_session = prompt.session_id
                for rid, (_, sid) in list(self._pending_perms.items()):
                    if sid == prompt.session_id:
                        self._pending_perms.pop(rid, None)
        frame = prompt.frame or {}
        stop, error = _stop_reason(frame)
        result = frame.get("result") if isinstance(frame.get("result"),
                                                   dict) else {}
        if stop == "end_turn" and not prompt.streamed:
            reply = str(result.get("reply") or "")
            if reply:
                self._session_update(prompt.session_id, {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": reply},
                })
        payload: dict = {"stopReason": stop}
        meta: dict = {}
        if error:
            meta["error"] = error
        summary = {k: result.get(k) for k in
                   ("applied", "changed", "blockers", "cost", "steps",
                    "tests_ok", "session") if k in result}
        if summary:
            meta["pedacito"] = summary
        if meta:
            payload["_meta"] = meta
        self._respond(prompt.jsonrpc_id, payload)

    def _cancel(self, req_id: Any, params: dict, *, notify: bool) -> None:
        sid = params.get("sessionId") or params.get("session_id")
        if sid:
            self._session(params)                # validate the session id
        elif not (self._active is not None or self._last_session):
            raise AcpError(-32001, "session not found: no active session")
        frame = self.server.dispatch({
            "id": f"acp-cancel-{uuid.uuid4().hex[:6]}", "method": "cancel"})
        if not notify:
            result = (frame or {}).get("result") or {"cancelled": True}
            self._respond(req_id, result)

    # ------------------------------------------------------------------ #
    # frame translation (worker thread)
    # ------------------------------------------------------------------ #
    def _active_session_id(self) -> str:
        with self._lock:
            if self._active is not None:
                return self._active.session_id
            return self._last_session

    def _on_frame(self, frame: Any) -> None:
        try:
            self._translate(frame)
        except Exception as exc:           # telemetry must never break a task
            try:
                print(f"[acp:frame-error] {exc!r}", file=sys.stderr)
            except (ValueError, OSError):
                pass

    def _translate(self, frame: Any) -> None:
        if not isinstance(frame, dict):
            return
        kind = frame.get("kind")
        if kind == "permission_request":
            self._ask_permission(frame)
        elif kind == "step":
            self._tool_call(frame)
        elif kind == "llm_stream":
            self._stream_chunk(frame)
        elif kind == "plan_updated":
            self._plan_update(frame)
        elif kind == "rpc_response":
            sid = str(frame.get("id"))
            with self._lock:
                prompt = self._prompts.get(sid)
            if prompt is not None:
                prompt.frame = frame
                prompt.event.set()

    def _stream_chunk(self, frame: dict) -> None:
        with self._lock:
            prompt = self._active
        reasoning = str(frame.get("delta_reasoning") or "")
        content = str(frame.get("delta_content") or "")
        if not (reasoning or content):
            return
        sid = prompt.session_id if prompt is not None else self._active_session_id()
        if not sid:
            return
        if reasoning:
            self._session_update(sid, {
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": reasoning},
            })
        if content:
            if prompt is not None:
                prompt.streamed = True
            self._session_update(sid, {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": content},
            })

    def _tool_call(self, frame: dict) -> None:
        sid = self._active_session_id()
        if not sid:
            return
        step = frame.get("n")
        tool_id = f"step-{step}" if step is not None else \
            f"step-{uuid.uuid4().hex[:6]}"
        action = str(frame.get("action") or "tool")
        detail = str(frame.get("detail") or "")
        title = f"{action}: {detail}" if detail else action
        raw = {"action": action, "detail": detail, "step": step,
               "obs_chars": frame.get("obs_chars")}
        self._session_update(sid, {
            "sessionUpdate": "tool_call",
            "toolCallId": tool_id,
            "title": title,
            "kind": action,
            "status": "in_progress",
            "rawInput": raw,
        })
        self._session_update(sid, {
            "sessionUpdate": "tool_call_update",
            "toolCallId": tool_id,
            "title": title,
            "kind": action,
            "status": "completed",
            "rawOutput": {"obs_chars": frame.get("obs_chars")},
        })

    def _plan_update(self, frame: dict) -> None:
        sid = self._active_session_id()
        if not sid:
            return
        path = str(frame.get("path") or "")
        entries = [{"content": path or "plan updated", "priority": "medium",
                    "status": "in_progress"}]
        self._session_update(sid, {"sessionUpdate": "plan",
                                   "entries": entries})

    # ------------------------------------------------------------------ #
    # permissions (agent -> client request, client response back to RPC)
    # ------------------------------------------------------------------ #
    def _ask_permission(self, frame: dict) -> None:
        with self._lock:
            prompt = self._prompts.get(str(frame.get("req_id")))
            sid = prompt.session_id if prompt is not None \
                else self._active_session_id()
            self._perm_seq += 1
            req_id = f"acp-perm-{self._perm_seq}"
            request_id = str(frame.get("request_id") or "")
            self._pending_perms[req_id] = (request_id, sid)
        params = {
            "sessionId": sid,
            "toolCall": {
                "toolCallId": request_id,
                "title": str(frame.get("action_summary")
                             or frame.get("key") or "permission request"),
                "kind": str(frame.get("category") or "tool"),
                "status": "pending",
                "rawInput": {"detail": frame.get("detail") or "",
                             "key": frame.get("key") or ""},
            },
            "options": [dict(o) for o in PERMISSION_OPTIONS],
            "_meta": {
                "requestId": request_id,
                "category": frame.get("category") or "",
                "riskLevel": frame.get("risk_level") or "medium",
                "reason": frame.get("reason") or "",
                "timeoutSeconds": frame.get("timeout_seconds"),
            },
        }
        self._write({"jsonrpc": JSONRPC, "id": req_id,
                     "method": "session/request_permission", "params": params})

    def _handle_response(self, msg: dict) -> None:
        rid = msg.get("id")
        with self._lock:
            pending = self._pending_perms.pop(rid, None)
        if pending is None:
            return                        # late/unknown response: just ignore
        request_id, _sid = pending
        decision = _decision_from(msg)
        self.server.dispatch({
            "id": f"acp-pr-{uuid.uuid4().hex[:6]}",
            "method": "permission_reply",
            "params": {"request_id": request_id, "decision": decision},
        })

    # ------------------------------------------------------------------ #
    # read loop
    # ------------------------------------------------------------------ #
    def handle(self, msg: Any) -> None:
        """Process one decoded JSON-RPC message (request, response or
        notification)."""
        if not isinstance(msg, dict):
            self._fail(None, -32600, "request must be a JSON object")
            return
        if "method" not in msg:
            if "id" in msg and ("result" in msg or "error" in msg):
                self._handle_response(msg)
                return
            rid = msg.get("id")
            if rid is not None:
                self._fail(rid, -32600, "invalid request: missing method")
            return
        if msg.get("jsonrpc") not in (None, JSONRPC):
            rid = msg.get("id")
            if rid is not None:
                self._fail(rid, -32600,
                           f"jsonrpc must be {JSONRPC!r}")
            return
        method = msg.get("method")
        if not isinstance(method, str) or not method:
            rid = msg.get("id")
            if rid is not None:
                self._fail(rid, -32600, "invalid request: bad method")
            return
        notify = "id" not in msg
        req_id = msg.get("id")
        params = msg.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            if not notify:
                self._fail(req_id, -32602, "params must be an object")
            return
        try:
            if method == "initialize":
                result = self._initialize(params)
            elif method == "session/new":
                result = self._session_new(params)
            elif method == "session/prompt":
                if notify:
                    return
                self._prompt(req_id, params)
                return
            elif method == "session/cancel":
                self._cancel(req_id, params, notify=notify)
                return
            else:
                raise AcpError(-32601, f"unknown method {method!r}")
        except AcpError as exc:
            if not notify:
                self._fail(req_id, exc.code, exc.message, exc.data)
            return
        except Exception as exc:           # never kill the transport
            if not notify:
                self._fail(req_id, -32603, f"internal error: {exc!r}")
            return
        if not notify:
            self._respond(req_id, result)

    def _handle_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        if len(line) > rpc.MAX_LINE:
            self._fail(None, -32600,
                       f"request line exceeds {rpc.MAX_LINE} characters")
            return
        try:
            msg = json.loads(line)
        except ValueError as exc:
            self._fail(None, -32700, f"parse error: {exc}")
            return
        self.handle(msg)

    def serve(self) -> int:
        """Read ACP messages until stdin EOF, then drain and exit 0."""
        self._rpc_thread = threading.Thread(
            target=self._rpc_loop, daemon=True, name="pedacito-acp-rpc")
        self._rpc_thread.start()
        self.server.wait_ready(5.0)          # hook installed before any prompt
        try:
            while True:
                try:
                    line = self._in.readline()
                except (EOFError, KeyboardInterrupt, ValueError, OSError):
                    break
                if line == "":
                    break
                self._handle_line(line)
        finally:
            self._shutdown()
        return 0

    def _rpc_loop(self) -> None:
        try:
            self.server.serve()
        except Exception as exc:           # a broken loop must not hang EOF
            try:
                print(f"[acp:rpc-loop] {exc!r}", file=sys.stderr)
            except (ValueError, OSError):
                pass

    def _shutdown(self) -> None:
        """EOF: stop the RPC reader, let terminal frames flush, close stdout."""
        self._rpc_in.close()
        if self._rpc_thread is not None:
            self._rpc_thread.join(timeout=rpc.EOF_JOIN_GRACE + 5.0)
        with self._lock:
            watchers = list(self._watchers)
        # The RPC thread waited for the active worker and emitted its terminal
        # frame, so the watcher resolves on its own; give it a bounded grace
        # first, then force-answer anything still wedged so the process exits.
        for watcher in watchers:
            watcher.join(timeout=5.0)
        with self._lock:
            stuck = [p for p in self._prompts.values() if not p.event.is_set()]
        for prompt in stuck:
            prompt.event.set()
        for watcher in watchers:
            if watcher.is_alive():
                watcher.join(timeout=2.0)
        with self._lock:
            self._prompts.clear()
            self._pending_perms.clear()
        self.out.close()


def main(server: rpc.RpcServer, **kwargs) -> int:
    """Wrap `server` in an ACP bridge and serve until stdin EOF."""
    return AcpServer(server, **kwargs).serve()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _uri_to_path(raw: Any) -> str:
    """A `file://` URI or a plain path as a path string."""
    text = str(raw or "").strip()
    if text.startswith("file:"):
        text = text[5:]
        if text.startswith("//"):
            # file:///C:/x -> C:/x on Windows; file://host/x keeps the host
            text = text[2:]
            if text.startswith("localhost/"):
                text = text[len("localhost/"):]
        # Percent escapes in a URI path.
        try:
            from urllib.parse import unquote
            text = unquote(text)
        except (ValueError, TypeError):
            pass
        if re.match(r"^/[A-Za-z]:", text):
            text = text[1:]
    return text


def _decision_from(msg: dict) -> str:
    """Map an ACP permission response to the RPC broker's decision string."""
    if msg.get("error"):
        return "deny"
    result = msg.get("result")
    if not isinstance(result, dict):
        return "deny"
    outcome = result.get("outcome")
    option = ""
    if isinstance(outcome, dict):
        if str(outcome.get("outcome") or "") == "cancelled":
            return "deny"
        option = str(outcome.get("optionId") or "")
    if not option:
        option = str(result.get("optionId") or result.get("decision") or "")
    if option == "allow_once":
        return "allow_once"
    if option in ("allow_always", "always_allow", "allow_run"):
        return "allow_always"
    return "deny"


def _stop_reason(frame: dict) -> tuple[str, str]:
    """ACP stopReason plus the error text, from one terminal RPC frame."""
    if not isinstance(frame, dict):
        return "error", "internal error: no terminal frame"
    result = frame.get("result")
    res = result if isinstance(result, dict) else {}
    if frame.get("ok"):
        return "end_turn", ""
    error = str(frame.get("error") or "")
    low = error.lower()
    if error == "cancelled" or res.get("cancelled") \
            or "cancelled" in low or "canceled" in low \
            or "stopped by user" in low:
        return "cancelled", ""
    return "error", error or "task failed"


__all__ = ["AcpError", "AcpOut", "AcpServer", "PROTOCOL_VERSION",
           "PERMISSION_OPTIONS", "MAX_RESOURCE_CHARS", "JSONRPC", "main"]
