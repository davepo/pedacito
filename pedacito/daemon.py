"""Headless daemon: a loopback-only session registry with warm index sharing.

The daemon owns *sessions*, not one project. Each session gets its own
`SessionContext` (config clone, event bus, permission broker, policy engine,
review queue, resource registry) while the parsed index and the model-lease
manager stay warm and shared. Editors and machine clients attach over a tiny
HTTP/SSE control surface:

    GET    /v1/health
    GET    /v1/sessions
    POST   /v1/sessions                 create or attach
    GET    /v1/sessions/<id>
    DELETE /v1/sessions/<id>
    GET    /v1/sessions/<id>/events     SSE (or JSON with once=1)
    POST   /v1/sessions/<id>/prompt
    POST   /v1/sessions/<id>/cancel
    POST   /v1/sessions/<id>/permission
    POST   /v1/sessions/<id>/review
    POST   /v1/quiesce
    POST   /v1/shutdown

Security is deliberately narrow for this milestone: the server binds only a
loopback address, every endpoint requires the per-run bearer token
(``X-Pedacito-Token`` or ``Authorization: Bearer``), and the discovery file is
written atomically under the per-user state folder with 0o600 where the
platform honours it. There is no WAN exposure and no remote bind option.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from . import fileio, workspace
from .session_ctx import SessionContext, spawn_thread_with_context, use_session

DAEMON_PROTOCOL_VERSION = "0.4.0"
DISCOVERY_FILENAME = "daemon.json"

_LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")
_LOOPBACK_SET = set(_LOOPBACK_HOSTS)


def _state_path(state_dir: Path | str | None = None) -> Path:
    if state_dir is not None:
        return Path(state_dir).expanduser()
    from . import config as config_mod
    return Path(config_mod.USER_STATE_DIR)


def _discovery_path(state_dir: Path | str | None = None) -> Path:
    return _state_path(state_dir) / DISCOVERY_FILENAME


def _format_url(host: str, port: int) -> str:
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{int(port)}"


@dataclass
class WorkspaceSession:
    """One daemon-owned session and its live lifecycle state."""

    ctx: SessionContext
    state: str = "idle"                 # idle | running | draining | closed
    attached_clients: set[str] = field(default_factory=set)
    last_activity: float = field(default_factory=time.time)
    project_root: Path | None = None
    worktree_branch: str = ""
    error: str = ""
    workers: list[threading.Thread] = field(default_factory=list, repr=False)

    @property
    def session_id(self) -> str:
        return self.ctx.session_id

    def metadata(self) -> dict[str, Any]:
        cfg = self.ctx.cfg
        return {
            "session_id": self.ctx.session_id,
            "root": str(self.ctx.root or ""),
            "project_root": str(self.project_root or self.ctx.root or ""),
            "state": self.state,
            "attached_clients": sorted(self.attached_clients),
            "worktree_branch": self.worktree_branch,
            "profile": str(getattr(cfg, "active_profile", "") or ""),
            "created_at": self.ctx.created_at,
            "last_activity": self.last_activity,
        }


class DaemonServer:
    """A loopback HTTP/SSE daemon that owns isolated workspace sessions."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        state_dir: Path | None = None,
        cfg: Any = None,
        runner: Callable[..., Any] | None = None,
    ) -> None:
        clean_host = str(host or "").strip().lower()
        if clean_host not in _LOOPBACK_SET:
            raise ValueError(
                f"daemon host must be loopback ({', '.join(_LOOPBACK_HOSTS)}); "
                f"refusing {host!r}")
        self.host = clean_host
        self.port = int(port or 0)
        self.state_dir = Path(state_dir).expanduser() if state_dir else None
        self._token = secrets.token_urlsafe(32)
        self._cfg_template = cfg
        self._runner = runner
        self._lock = threading.RLock()
        self._sessions: dict[str, WorkspaceSession] = {}
        self._indexes: dict[str, Any] = {}
        self._http: Any = None
        self._thread: threading.Thread | None = None
        self._started = False
        self._stopped = threading.Event()
        self._stop_guard = threading.Lock()
        self._quiesce_hook = self._hook_active_sessions
        from . import leases as leases_mod
        self._leases = leases_mod.default_lease_manager()

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    @property
    def url(self) -> str:
        if self._http is not None:
            host, port = self._http.server_address[:2]
            return _format_url(str(host), int(port))
        return _format_url(self.host, self.port)

    @property
    def token(self) -> str:
        return self._token

    def start(self) -> dict[str, Any]:
        """Bind, serve on a background thread, and publish the discovery file."""
        with self._lock:
            if self._started:
                return self._info()
            from .local_server import LocalHTTPServer

            daemon = self

            class _Handler(BaseHTTPRequestHandler):
                protocol_version = "HTTP/1.1"

                def log_message(self, *_args: Any) -> None:  # noqa: D401
                    return

                def do_GET(self) -> None:       # noqa: N802 - http.server API
                    daemon._handle(self, "GET")

                def do_POST(self) -> None:      # noqa: N802
                    daemon._handle(self, "POST")

                def do_DELETE(self) -> None:    # noqa: N802
                    daemon._handle(self, "DELETE")

            self._http = LocalHTTPServer((self.host, self.port), _Handler)
            self._thread = spawn_thread_with_context(
                self._http.serve_forever,
                name="pedacito-daemon-http",
                daemon=True)
            workspace.register_quiesce_hook(self._quiesce_hook)
            self._started = True
            self._stopped.clear()
            self._write_discovery()
            return self._info()

    def stop(self, *, drain_timeout_s: float = 10.0) -> None:
        """Drain workers, close every session and socket, and clean discovery."""
        with self._stop_guard:
            self._stop_impl(drain_timeout_s=drain_timeout_s)

    def _stop_impl(self, *, drain_timeout_s: float = 10.0) -> None:
        with self._lock:
            if not self._started and self._http is None:
                self._remove_discovery()
                return
            self._started = False
            sessions = list(self._sessions.values())
        deadline = time.monotonic() + max(0.0, float(drain_timeout_s or 0.0))
        for ws in sessions:
            if ws.state != "closed":
                ws.state = "draining"
        while time.monotonic() < deadline:
            if not any(ws.state in ("running", "draining") and ws.workers
                       for ws in sessions):
                break
            time.sleep(0.02)
        for ws in sessions:
            self.close_session(ws.session_id, reason="daemon shutdown")
            for worker in list(ws.workers):
                try:
                    worker.join(timeout=max(0.0, deadline - time.monotonic()))
                except RuntimeError:
                    pass
        httpd = self._http
        thread = self._thread
        self._http = None
        self._thread = None
        if httpd is not None:
            try:
                httpd.shutdown()
            except Exception:
                pass
            try:
                httpd.server_close()
            except Exception:
                pass
        if thread is not None and thread is not threading.current_thread():
            try:
                thread.join(timeout=max(0.1, deadline - time.monotonic()))
            except RuntimeError:
                pass
        workspace.unregister_quiesce_hook(self._quiesce_hook)
        self._remove_discovery()
        self._stopped.set()
        with self._lock:
            self._sessions.clear()
            self._indexes.clear()

    # ------------------------------------------------------------------ #
    # sessions
    # ------------------------------------------------------------------ #
    def create_session(
        self,
        root: Path | str,
        *,
        profile: str = "",
        permission_mode: str = "",
        worktree_name: str = "",
        dirty_policy: str = "refuse",
    ) -> WorkspaceSession:
        """Create an isolated session; the index and lease manager stay warm."""
        from . import events as events_mod
        from . import policy as policy_mod
        from . import security

        root_path = Path(root).expanduser().resolve()
        cfg = self._load_cfg(root_path, profile)
        if permission_mode:
            cfg.permission_mode = str(permission_mode).strip().lower()
        approval = str(getattr(cfg, "approval_policy", "ask") or "ask").lower()
        idx = self._index_for(root_path)
        ctx = SessionContext(root=root_path, cfg=cfg, idx=idx, mode="daemon")
        bus = events_mod.SessionEventBus(
            ctx.session_id, sink=self._root_log(root_path, cfg))
        ctx.event_bus = bus
        ctx.permission_broker = security.SessionPermissionBroker(
            ctx.session_id,
            approval_policy=approval,
            cancel_event=ctx.cancel_event,
            event_bus=bus,
            interactive=approval != "never",
        )
        ctx.policy = policy_mod.PolicyEngine(
            cfg, root=root_path, event_bus=bus,
            budget_ledger=ctx.budget_ledger)
        if ctx.review_queue is not None:
            try:
                ctx.review_queue.set_listener(
                    lambda review: bus.emit(
                        "review_staged",
                        review_id=review.review_id,
                        files=[d.rel for d in review.diffs
                               if not d.removed_source]))
            except Exception:
                pass

        ws = WorkspaceSession(ctx=ctx, state="idle",
                              project_root=root_path,
                              last_activity=time.time())
        if worktree_name:
            from . import worktree as worktree_mod
            handle = worktree_mod.create_session_worktree(
                root_path, worktree_name,
                session_id=ctx.session_id,
                dirty_policy=dirty_policy)
            ctx.worktree = handle
            ctx.root = handle.path
            ws.worktree_branch = handle.branch
        with self._lock:
            self._sessions[ctx.session_id] = ws
        bus.emit("session_created", root=str(ctx.root),
                 project_root=str(root_path),
                 worktree_branch=ws.worktree_branch)
        return ws

    def get_session(self, session_id: str) -> WorkspaceSession | None:
        with self._lock:
            return self._sessions.get(str(session_id))

    def close_session(self, session_id: str, *, reason: str = "closed") -> bool:
        with self._lock:
            ws = self._sessions.pop(str(session_id), None)
            if ws is None:
                return False
            ws.state = "closed"
        try:
            ws.ctx.close(reason=reason)
        except Exception:
            pass
        bus = getattr(ws.ctx, "event_bus", None)
        emit = getattr(bus, "emit", None)
        if callable(emit):
            try:
                emit("session_closed", reason=str(reason))
            except Exception:
                pass
        try:
            self._leases.release_session(ws.ctx.session_id)
        except Exception:
            pass
        return True

    def active_sessions_for(self, root: Path | str) -> list[dict[str, Any]]:
        wanted = Path(root).expanduser().resolve()
        out: list[dict[str, Any]] = []
        for ws in self._all_sessions():
            if ws.state == "closed":
                continue
            base = ws.project_root or ws.ctx.root
            try:
                same = Path(base).resolve() == wanted
            except OSError:
                same = False
            if same:
                out.append(ws.metadata())
        return out

    def quiesce(self, root: Path | str | None = None,
                *, timeout_s: float = 10.0) -> tuple[bool, str]:
        """Drain matching sessions; refuse while a worker is still running."""
        wanted = Path(root).expanduser().resolve() if root is not None else None
        targets: list[WorkspaceSession] = []
        for ws in self._all_sessions():
            if ws.state == "closed":
                continue
            if wanted is not None:
                base = ws.project_root or ws.ctx.root
                try:
                    if Path(base).resolve() != wanted:
                        continue
                except OSError:
                    continue
            targets.append(ws)
        if not targets:
            return True, ""
        for ws in targets:
            if ws.state == "running":
                ws.state = "draining"
        deadline = time.monotonic() + max(0.0, float(timeout_s or 0.0))
        while True:
            running = [ws for ws in targets
                       if ws.state in ("running", "draining")]
            if not running:
                break
            if time.monotonic() >= deadline:
                verb = f" for {root}" if root is not None else ""
                return (False,
                        f"cannot maintain now: {len(running)} active "
                        f"session(s){verb}")
            time.sleep(0.02)
        for ws in targets:
            self.close_session(ws.session_id, reason="quiesced")
        return True, ""

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _load_cfg(self, root: Path, profile: str) -> Any:
        from . import config as config_mod
        cfg = None
        if self._cfg_template is not None:
            from .session_ctx import clone_config
            cfg = clone_config(self._cfg_template)
        if cfg is None:
            try:
                cfg = config_mod.load(
                    root=root, profile=profile or None,
                    require_hardware=False, warn_budgets=False)
            except Exception:
                cfg = config_mod.Config()
        if profile:
            profiles = getattr(cfg, "profiles", {}) or {}
            entry = profiles.get(profile) if isinstance(profiles, dict) else None
            if isinstance(entry, dict):
                for key, value in entry.items():
                    if hasattr(cfg, key):
                        try:
                            setattr(cfg, key, value)
                        except Exception:
                            pass
            try:
                cfg.active_profile = profile
            except Exception:
                pass
        return cfg

    def _index_for(self, root: Path) -> Any:
        key = str(root)
        with self._lock:
            cached = self._indexes.get(key)
        if cached is not None:
            return cached
        idx = None
        try:
            from .index import Index
            idx = Index.load(root) or Index(root, [])
        except Exception:
            idx = None
        with self._lock:
            self._indexes.setdefault(key, idx)
            return self._indexes.get(key)

    def _root_log(self, root: Path, cfg: Any) -> Any:
        try:
            from . import events as events_mod
            log = events_mod.get(root, cfg)
            if hasattr(log, "_emit_direct"):
                return log
            return getattr(log, "_log", None)
        except Exception:
            return None

    def _hook_active_sessions(self, root: Path, _action: str) -> list[str]:
        out: list[str] = []
        wanted = Path(root).expanduser().resolve()
        for ws in self._all_sessions():
            if ws.state not in ("running", "draining"):
                continue
            base = ws.project_root or ws.ctx.root
            try:
                if Path(base).resolve() != wanted:
                    continue
            except OSError:
                continue
            out.append(ws.session_id)
        return out

    def _info(self) -> dict[str, Any]:
        info = {
            "protocol_version": DAEMON_PROTOCOL_VERSION,
            "pid": os.getpid(),
            "start_time": workspace._process_start_time(os.getpid()),
            "url": self.url,
            "host": self.host,
            "port": int(self.port or 0),
            "token": self._token,
            "started_at": time.time(),
        }
        if self._http is not None:
            host, port = self._http.server_address[:2]
            info.update(url=_format_url(str(host), int(port)),
                        host=str(host), port=int(port))
        return info

    def _write_discovery(self) -> None:
        path = _discovery_path(self.state_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._info(), indent=1, sort_keys=True).encode(
            "utf-8")
        fileio.write_bytes(path, payload)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass                     # Windows: best-effort only

    def _remove_discovery(self) -> None:
        path = _discovery_path(self.state_dir)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        try:
            if int(data.get("pid") or 0) != os.getpid():
                return
        except (TypeError, ValueError):
            return
        try:
            path.unlink()
        except OSError:
            pass

    def _request_stop(self) -> None:
        def _stop() -> None:
            time.sleep(0.05)
            self.stop()

        spawn_thread_with_context(_stop, name="pedacito-daemon-stop")

    def _all_sessions(self) -> list[WorkspaceSession]:
        with self._lock:
            return list(self._sessions.values())

    # ------------------------------------------------------------------ #
    # HTTP dispatch
    # ------------------------------------------------------------------ #
    def _handle(self, handler: BaseHTTPRequestHandler, method: str) -> None:
        try:
            parsed = urlparse(handler.path)
            if not self._authorized(handler):
                self._json(handler, 401, {"error": "unauthorized"})
                return
            query = parse_qs(parsed.query)
            if method == "GET":
                parts = [p for p in parsed.path.split("/") if p]
                if (len(parts) == 4 and parts[:2] == ["v1", "sessions"]
                        and parts[3] == "events"):
                    self._stream_events(handler, query)
                    return
            status, payload = self._route(method, parsed.path, query,
                                          self._body(handler))
            self._json(handler, status, payload)
        except Exception as exc:      # never kill the daemon on one request
            try:
                self._json(handler, 500,
                           {"error": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass

    def _authorized(self, handler: BaseHTTPRequestHandler) -> bool:
        supplied = str(handler.headers.get("X-Pedacito-Token") or "")
        if not supplied:
            auth = str(handler.headers.get("Authorization") or "")
            if auth.lower().startswith("bearer "):
                supplied = auth[7:].strip()
        return bool(supplied) and hmac.compare_digest(supplied, self._token)

    @staticmethod
    def _body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
        try:
            length = int(handler.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            length = 0
        if length <= 0:
            return {}
        raw = handler.rfile.read(length)
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _json(handler: BaseHTTPRequestHandler, status: int,
              payload: dict[str, Any]) -> None:
        data = json.dumps(payload, default=str).encode("utf-8")
        handler.send_response(int(status))
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(data)
        handler.close_connection = True

    def _route(self, method: str, path: str, query: dict[str, list[str]],
               body: dict[str, Any]) -> tuple[int, Any]:
        if path == "/v1/health" and method == "GET":
            return 200, {
                "ok": True,
                "protocol_version": DAEMON_PROTOCOL_VERSION,
                "pid": os.getpid(),
                "sessions": len([ws for ws in self._all_sessions()
                                 if ws.state != "closed"]),
            }
        if path == "/v1/sessions" and method == "GET":
            return 200, {"sessions": [
                ws.metadata() for ws in self._all_sessions()
                if ws.state != "closed"]}
        if path == "/v1/sessions" and method == "POST":
            return self._create_or_attach(body)
        if path == "/v1/quiesce" and method == "POST":
            root = body.get("root")
            try:
                timeout_s = float(body.get("timeout_s",
                                           body.get("timeout", 10.0)))
            except (TypeError, ValueError):
                timeout_s = 10.0
            ok, reason = self.quiesce(root, timeout_s=timeout_s)
            return 200, {"ok": bool(ok), "reason": reason}
        if path == "/v1/shutdown" and method == "POST":
            self._request_stop()
            return 202, {"ok": True, "stopping": True}

        parts = [p for p in path.split("/") if p]
        if len(parts) >= 3 and parts[0] == "v1" and parts[1] == "sessions":
            sid = parts[2]
            ws = self.get_session(sid)
            if ws is None:
                return 404, {"error": f"session not found: {sid}"}
            if len(parts) == 3 and method == "GET":
                return 200, ws.metadata()
            if len(parts) == 3 and method == "DELETE":
                ok = self.close_session(sid, reason="deleted")
                return (200 if ok else 404), {
                    "ok": bool(ok), "session_id": sid}
            if len(parts) == 4 and method == "POST":
                action = parts[3]
                if action == "prompt":
                    return self._prompt(ws, body)
                if action == "cancel":
                    return self._cancel(ws)
                if action == "permission":
                    return self._permission(ws, body)
                if action == "review":
                    return self._review(ws, body)
        return 404, {"error": f"not found: {method} {path}"}

    def _create_or_attach(self, body: dict[str, Any]) -> tuple[int, Any]:
        sid = str(body.get("session_id") or body.get("sessionId") or "")
        client = str(body.get("client_id") or body.get("clientId") or "http")
        if sid:
            ws = self.get_session(sid)
            if ws is None:
                return 404, {"error": f"session not found: {sid}"}
            ws.attached_clients.add(client)
            ws.last_activity = time.time()
            return 200, {"ok": True, "attached": True, **ws.metadata()}
        root = body.get("root") or body.get("project") or ""
        if not str(root).strip():
            return 400, {"error": "root is required to create a session"}
        try:
            ws = self.create_session(
                Path(str(root)),
                profile=str(body.get("profile") or ""),
                permission_mode=str(body.get("permission_mode")
                                    or body.get("permissionMode") or ""),
                worktree_name=str(body.get("worktree")
                                  or body.get("worktree_name") or ""),
                dirty_policy=str(body.get("dirty_policy") or "refuse"),
            )
        except (OSError, ValueError, RuntimeError) as exc:
            return 400, {"error": f"{type(exc).__name__}: {exc}"}
        ws.attached_clients.add(client)
        return 201, {"ok": True, "attached": False, **ws.metadata()}

    def _prompt(self, ws: WorkspaceSession,
                body: dict[str, Any]) -> tuple[int, Any]:
        task = str(body.get("task") or body.get("prompt")
                   or body.get("message") or "").strip()
        if not task:
            return 400, {"error": "task is required"}
        if ws.state == "running":
            return 409, {"error": "session is busy"}
        apply_ = bool(body.get("apply", False))
        client = str(body.get("client_id") or body.get("clientId") or "http")
        ws.attached_clients.add(client)
        ws.state = "running"
        ws.last_activity = time.time()
        worker = spawn_thread_with_context(
            self._prompt_worker, args=(ws.session_id, task, apply_),
            name=f"pedacito-daemon-{ws.session_id[-6:]}", daemon=True)
        ws.workers.append(worker)
        return 202, {"ok": True, "session_id": ws.session_id,
                     "state": ws.state}

    def _prompt_worker(self, session_id: str, task: str,
                       apply_: bool) -> None:
        ws = self.get_session(session_id)
        if ws is None or ws.state == "closed":
            return
        ctx = ws.ctx
        bus = ctx.event_bus
        try:
            if bus is not None:
                bus.emit("prompt_started", task=task[:4000], apply=bool(apply_))
            with use_session(ctx):
                if self._runner is not None:
                    try:
                        result = self._runner(ctx, task, apply_)
                    except TypeError:
                        result = self._runner(ctx, task)
                else:
                    result = None
            reply = ""
            if isinstance(result, dict):
                reply = str(result.get("reply") or result.get("digest") or "")
            elif result is not None:
                reply = str(result)
            if bus is not None:
                bus.emit("prompt_finished", ok=True, reply=reply[:8000])
        except Exception as exc:
            ws.error = f"{type(exc).__name__}: {exc}"
            if bus is not None:
                bus.emit("prompt_error", ok=False, error=ws.error[:1000])
        finally:
            ws.state = "idle"
            ws.last_activity = time.time()

    def _cancel(self, ws: WorkspaceSession) -> tuple[int, Any]:
        try:
            ws.ctx.cancel("cancelled")
        except Exception:
            pass
        ws.state = "idle"
        ws.last_activity = time.time()
        return 200, {"ok": True, "session_id": ws.session_id,
                     "cancelled": True}

    def _permission(self, ws: WorkspaceSession,
                    body: dict[str, Any]) -> tuple[int, Any]:
        broker = ws.ctx.permission_broker
        request_id = str(body.get("request_id") or body.get("requestId") or "")
        decision = str(body.get("decision") or "")
        if broker is None:
            return 404, {"error": "no permission broker for this session"}
        if not request_id or not decision:
            return 400, {"error": "request_id and decision are required"}
        ok, detail = broker.decide(ws.session_id, request_id, decision)
        ws.last_activity = time.time()
        return (200 if ok else 409), {"ok": bool(ok), "detail": detail}

    def _review(self, ws: WorkspaceSession,
                body: dict[str, Any]) -> tuple[int, Any]:
        queue = ws.ctx.review_queue
        review_id = str(body.get("review_id") or body.get("reviewId") or "")
        action = str(body.get("action") or "")
        if queue is None:
            return 404, {"error": "no review queue for this session"}
        if not review_id or not action:
            return 400, {"error": "review_id and action are required"}
        files = body.get("files")
        if files is not None and not isinstance(files, list):
            files = None
        hunks = body.get("accepted_hunks") or body.get("acceptedHunks")
        if hunks is not None and not isinstance(hunks, dict):
            hunks = None
        ok, detail = queue.decide(review_id, action, files=files,
                                  accepted_hunks=hunks)
        ws.last_activity = time.time()
        return (200 if ok else 409), {"ok": bool(ok), "detail": detail}

    def _stream_events(self, handler: BaseHTTPRequestHandler,
                       query: dict[str, list[str]]) -> None:
        parts = [p for p in urlparse(handler.path).path.split("/") if p]
        ws = self.get_session(parts[2]) if len(parts) == 4 else None
        if ws is None:
            self._json(handler, 404, {"error": "session not found"})
            return
        try:
            after = int((query.get("after") or ["0"])[0])
        except (TypeError, ValueError):
            after = 0
        once = str((query.get("once") or ["0"])[0]).lower() in (
            "1", "true", "yes")
        bus = getattr(ws.ctx, "event_bus", None)
        if bus is None:
            self._json(handler, 404, {"error": "session has no event bus"})
            return
        if once:
            events, resync = bus.replay(after)
            self._json(handler, 200, {
                "session_id": ws.session_id,
                "after": after,
                "resync": bool(resync),
                "events": [e.to_dict() for e in events],
            })
            return
        try:
            timeout = float((query.get("timeout") or ["5"])[0])
        except (TypeError, ValueError):
            timeout = 5.0
        timeout = max(0.0, min(timeout, 30.0))
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.close_connection = True
        deadline = time.monotonic() + timeout
        current = after
        try:
            while True:
                events, resync = bus.replay(current)
                if resync:
                    frame = {"seq": current, "kind": "resync",
                             "session_id": ws.session_id}
                    handler.wfile.write(
                        ("id: resync\nevent: resync\ndata: "
                         + json.dumps(frame) + "\n\n").encode("utf-8"))
                    handler.wfile.flush()
                    current = bus.seq
                    events = []
                if events:
                    for event in events:
                        payload = json.dumps(event.to_dict(), default=str)
                        handler.wfile.write(
                            (f"id: {event.seq}\nevent: {event.kind}\ndata: "
                             f"{payload}\n\n").encode("utf-8"))
                        current = event.seq
                    handler.wfile.flush()
                remaining = deadline - time.monotonic()
                if remaining <= 0 or ws.state == "closed":
                    break
                waiting, _ = bus.wait(current, timeout=min(0.5, remaining))
                if waiting:
                    for event in waiting:
                        payload = json.dumps(event.to_dict(), default=str)
                        handler.wfile.write(
                            (f"id: {event.seq}\nevent: {event.kind}\ndata: "
                             f"{payload}\n\n").encode("utf-8"))
                        current = event.seq
                    handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError, ValueError):
            pass


# --------------------------------------------------------------------------- #
# client discovery helpers
# --------------------------------------------------------------------------- #

def read_daemon_info(state_dir: Path | str | None = None) -> dict[str, Any] | None:
    """The live daemon's discovery record, or None when absent/stale."""
    path = _discovery_path(state_dir)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        pid = int(data.get("pid") or 0)
    except (TypeError, ValueError):
        return None
    if pid <= 0 or not workspace._pid_alive(pid):
        return None
    recorded = float(data.get("start_time") or 0.0)
    if recorded > 0.0:
        current = workspace._process_start_time(pid)
        if current > 0.0 and abs(current - recorded) > 2.0:
            return None
    return data


def _requester(url: str, token: str, method: str, path: str,
               body: dict[str, Any] | None, timeout: float) -> tuple[int, dict]:
    parsed = urlparse(str(url or ""))
    if (parsed.hostname or "").lower() not in _LOOPBACK_SET:
        return 0, {"error": "daemon discovery URL is not loopback"}
    data = None
    headers = {"X-Pedacito-Token": str(token),
               "Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url.rstrip("/") + path, data=data,
                                     headers=headers, method=method)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read()
            status = int(getattr(response, "status", 200) or 200)
    except urllib.error.HTTPError as exc:
        raw = exc.read() if hasattr(exc, "read") else b""
        status = int(exc.code)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return 0, {"error": f"{type(exc).__name__}: {exc}"}
    if not raw:
        return status, {}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return status, {"error": "response was not JSON"}
    return status, payload if isinstance(payload, dict) else {"result": payload}


def daemon_request(method: str, path: str, body: dict[str, Any] | None = None,
                   *, state_dir: Path | str | None = None,
                   timeout: float = 5.0) -> tuple[int, dict[str, Any]]:
    """Call the running daemon; returns ``(status, payload)`` (0 on no daemon)."""
    info = read_daemon_info(state_dir)
    if info is None:
        return 0, {"error": "daemon not running"}
    return _requester(str(info.get("url") or ""),
                      str(info.get("token") or ""),
                      method.upper(), path, body, timeout)


def daemon_status(state_dir: Path | str | None = None) -> dict[str, Any]:
    """A JSON-safe status record that is always safe to call offline."""
    info = read_daemon_info(state_dir)
    if info is None:
        return {"running": False, "protocol_version": DAEMON_PROTOCOL_VERSION}
    status, health = daemon_request("GET", "/v1/health",
                                    state_dir=state_dir, timeout=2.0)
    if status != 200:
        return {"running": False, "protocol_version": DAEMON_PROTOCOL_VERSION,
                "reason": health.get("error") or f"health returned {status}",
                "info": info}
    return {"running": True, "protocol_version": DAEMON_PROTOCOL_VERSION,
            "info": info, "health": health}


__all__ = [
    "DAEMON_PROTOCOL_VERSION",
    "DaemonServer",
    "WorkspaceSession",
    "daemon_request",
    "daemon_status",
    "read_daemon_info",
]
