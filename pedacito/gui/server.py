"""HTTP routes for the GUI: static files, JSON endpoints, and the SSE stream.

The server binds to 127.0.0.1 only and refuses foreign Host/Origin headers
(`review.request_allowed`), since it can write files. POSTs must be JSON, which
a cross-site page cannot send without a CORS preflight that this server never
grants.
"""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .. import review as review_mod
from .. import workspace
from ..config import ConfigError
from ..local_server import LocalHTTPServer
from ..select import Matcher, is_venv_dir
from . import jobs, settings_api
from .core import STATE, install_runner, install_tee, uninstall_runner, uninstall_tee

STATIC = Path(__file__).parent / "static"
MAX_BODY = 12 * 1024 * 1024
MAX_VIEW_BYTES = 600_000
MAX_SCAN_FILES = 4000          # workspace scan cap for the Files tab
MAX_BACKUPS = 20

# Explicit allowlist of served types (everything under STATIC/ is still
# directory-contained below). vendor.js is the vendored preact+htm bundle and
# must arrive as a JS MIME for `import` to accept it.
STATIC_TYPES = {".html": "text/html", ".css": "text/css",
                ".js": "application/javascript", ".mjs": "application/javascript",
                ".json": "application/json", ".svg": "image/svg+xml",
                ".woff2": "font/woff2", ".png": "image/png",
                ".ico": "image/x-icon"}

CSP = ("default-src 'self'; style-src 'self' 'unsafe-inline'; "
       "script-src 'self'; connect-src 'self'; img-src 'self' data:; "
       "frame-ancestors 'none'; base-uri 'none'; form-action 'none'")

_SERVER: LocalHTTPServer | None = None


# --- helpers ------------------------------------------------------------------

def _within(root: Path, rel: str) -> Path | None:
    """`rel` resolved under `root`, or None when it escapes the project."""
    try:
        p = (root / rel).resolve()
        p.relative_to(root.resolve())
        return p
    except (ValueError, OSError):
        return None


def list_dir(path: str, show_files: bool, hidden: bool) -> dict:
    """One directory for the folder browser."""
    home = Path.home()
    start = path or (str(STATE.root.parent) if STATE.root else str(home))
    p = Path(start).expanduser()
    try:
        p = p.resolve()
    except OSError:
        p = home
    while not p.is_dir() and p != p.parent:
        p = p.parent
    entries, truncated = [], False
    try:
        with os.scandir(p) as it:
            raw = sorted(it, key=lambda e: (not e.is_dir(follow_symlinks=False),
                                            e.name.lower()))
    except OSError as e:
        return {"path": str(p), "parent": str(p.parent), "home": str(home),
                "entries": [], "error": str(e)}
    for e in raw:
        if not hidden and e.name.startswith("."):
            continue
        try:
            is_dir = e.is_dir()
        except OSError:
            continue
        if not is_dir and not show_files:
            continue
        item = {"name": e.name, "dir": is_dir}
        if is_dir and len(entries) < 400:
            d = Path(e.path)
            item["project"] = (d / ".git").exists() or (d / ".pedacito").exists()
        entries.append(item)
        if len(entries) >= 2000:
            truncated = True
            break
    return {"path": str(p),
            "parent": str(p.parent) if p.parent != p else "",
            "home": str(home), "entries": entries, "truncated": truncated}


def read_project_file(rel: str) -> dict:
    root = STATE.root
    if root is None:
        raise ConfigError("No project is open.")
    p = _within(root, rel)
    if p is None or not p.is_file():
        raise ConfigError(f"No such file in the project: {rel}")
    size = p.stat().st_size
    data = p.read_bytes()[:MAX_VIEW_BYTES]
    if b"\x00" in data[:4096]:
        return {"path": rel, "binary": True, "size": size, "text": ""}
    return {"path": rel, "binary": False, "size": size,
            "text": data.decode("utf-8", errors="replace"),
            "truncated": size > MAX_VIEW_BYTES}


def project_files() -> dict:
    """The FILES tab list: the index's files when one is loaded, unioned with
    a bounded workspace scan (honouring `.pedacitoignore`/`.gitignore`) so
    files written since the last index still show up -- and the list works at
    all before the first index is built. `indexed` marks which entries the
    index knows."""
    with STATE.lock:
        root, idx, args = STATE.root, STATE.idx, STATE.args
        indexed = sorted(idx.files) if idx is not None else []
        sig = STATE.files_signature()
    out = {"sig": sig, "indexed": indexed, "files": indexed, "truncated": False}
    if root is None or not root.is_dir():
        return out
    matcher = Matcher.from_files(root)
    if args is not None:
        matcher.extend(args.exclude or [])
        matcher.extend("!" + p for p in (args.include or []))
    found, scanned, stop = set(indexed), 0, False
    for dirpath, dirnames, filenames in os.walk(root):
        if stop:
            break
        d = Path(dirpath)
        base = "" if d == root else d.relative_to(root).as_posix()
        dirnames[:] = [x for x in dirnames
                       if not is_venv_dir(d / x)
                       and not matcher.prunable_dir(f"{base}/{x}" if base else x)]
        for name in sorted(filenames):
            rel = f"{base}/{name}" if base else name
            if rel in found or matcher.excluded(rel):
                continue
            found.add(rel)
            scanned += 1
            if scanned >= MAX_SCAN_FILES:
                out["truncated"] = True
                stop = True
                break
    out["files"] = sorted(found)
    return out


def backup_snapshots() -> list:
    """The BACKUPS tab list: snapshot directories newest first, with a file
    count and timestamp for each."""
    with STATE.lock:
        root = STATE.root
    out = []
    if root is None:
        return out
    for d in workspace.list_backups(root)[:MAX_BACKUPS]:
        try:
            n = sum(1 for p in d.rglob("*") if p.is_file())
            when = d.stat().st_mtime
        except OSError:
            n, when = 0, 0.0
        out.append({"name": d.name, "files": n, "time": when})
    return out


def plan_payload() -> dict:
    with STATE.lock:
        path, root = STATE.plan_path, STATE.root
    if path is None and root is not None and STATE.cfg is not None:
        cand = root / STATE.cfg.auto_plan_file
        path = cand if cand.exists() else None
    if path is None:
        return {"path": "", "exists": False, "text": "", "tasks": []}
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"path": str(path), "exists": False, "text": "", "tasks": []}
    shown = str(path)
    if root is not None:
        try:
            shown = Path(path).resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return {"path": shown, "exists": True, "text": text,
            "tasks": jobs.parse_plan(text)}


# --- routes ---------------------------------------------------------------------

GET_ROUTES: dict = {}
POST_ROUTES: dict = {}


def get(path):
    def deco(fn):
        GET_ROUTES[path] = fn
        return fn
    return deco


def post(path):
    def deco(fn):
        POST_ROUTES[path] = fn
        return fn
    return deco


def _busy_reply(job):
    return (202, {"ok": True, "job": job.id}) if job else (
        409, {"ok": False, "detail": "Another job is running. Wait for it "
              "or press Stop."})


@get("/api/state")
def _state(q, body=None):
    # `status` carries the header/engine telemetry; the chat transcript rides
    # alongside it so a polling client can reconcile messages in one call.
    # (`messages` is the same field name /api/snapshot uses.)
    return 200, {**STATE.status(), "messages": STATE.messages_snapshot()}


@get("/api/engine")
def _engine(q, body=None):
    """Engine V2 telemetry for the ACTIVITY & PLAN panel. `status.engine`
    carries the same payload; this route is the cheap standalone poll."""
    return 200, STATE.engine()


@get("/api/snapshot")
def _snapshot(q, body=None):
    return 200, STATE.snapshot()


@get("/api/files")
def _files(q, body=None):
    return 200, project_files()


@get("/api/backups")
def _backups(q, body=None):
    """Backup snapshots for the WORKSPACE panel's BACKUPS tab."""
    return 200, {"backups": backup_snapshots()}


@get("/api/file")
def _file(q, body=None):
    return 200, read_project_file(q.get("path", [""])[0])


@get("/api/fs")
def _fs(q, body=None):
    return 200, list_dir(q.get("path", [""])[0],
                         q.get("files", ["0"])[0] == "1",
                         q.get("hidden", ["0"])[0] == "1")


@get("/api/review")
def _review(q, body=None):
    return 200, {"review": STATE.review_meta()}


@get("/api/review/file")
def _review_file(q, body=None):
    try:
        rid = int(q.get("id", ["0"])[0])
    except ValueError:
        rid = 0
    out = STATE.review_file(rid, q.get("rel", [""])[0])
    return (200, out) if out else (404, {"ok": False, "detail": "no such file"})


@get("/api/plan")
def _plan(q, body=None):
    return 200, plan_payload()


@get("/api/settings")
def _settings(q, body=None):
    return 200, settings_api.payload()


@post("/api/peek")
def _peek(q, body):
    """Profile names for a folder, without opening it: reads the config files
    only (no model-server contact, no warnings printed)."""
    from .. import config as config_mod
    p = Path(str(body.get("path", ""))).expanduser()
    if not p.is_dir():
        return 200, {"ok": False, "detail": f"not a directory: {p}"}
    try:
        cfg = config_mod.load(root=p.resolve(), require_hardware=False,
                              warn_budgets=False, use_env=False)
    except ConfigError as e:
        return 200, {"ok": False, "detail": str(e)}
    return 200, {"ok": True, "profiles": sorted(cfg.profiles),
                 "default": cfg.default_profile}


@post("/api/project")
def _project(q, body):
    path, profile = str(body.get("path", "")), str(body.get("profile", ""))
    if not path.strip():
        return 400, {"ok": False, "detail": "Enter a project path."}
    return _busy_reply(jobs.spawn(
        "project", "opening project",
        lambda job: jobs.open_project(path, profile)))


@post("/api/profile")
def _profile_switch(q, body):
    """Switch the active model profile and re-open the current project with
    it, so subsequent runs use the chosen profile."""
    name = str(body.get("profile", "")).strip()
    with STATE.lock:
        root = STATE.root
        known = sorted(STATE.cfg.profiles) if STATE.cfg else []
    if root is None:
        return 400, {"ok": False, "detail": "Open a project first."}
    if name and known and name not in known:
        return 400, {"ok": False, "detail": f"no profile named {name!r}"}
    return _busy_reply(jobs.spawn(
        "project", f"switching profile to {name or 'default'}",
        lambda job: jobs.open_project(str(root), name)))


@post("/api/index")
def _index(q, body):
    summarise = bool(body.get("summarise", True))
    return _busy_reply(jobs.spawn(
        "index", "building index" + ("" if summarise else " (no summaries)"),
        lambda job: jobs.build_index(job, summarise)))


@post("/api/check_model")
def _check_model(q, body):
    return _busy_reply(jobs.spawn("check", "checking model", jobs.check_model))


@post("/api/run")
def _run(q, body):
    mode = body.get("mode", "")
    fns = {"task": (jobs.run_task, "running task"),
           "explain": (jobs.run_explain, "answering"),
           "auto": (jobs.run_auto, "running auto"),
           "new": (jobs.run_new, "creating project")}
    if mode not in fns:
        return 400, {"ok": False, "detail": f"unknown mode {mode!r}"}
    fn, title = fns[mode]
    opts = {**body, "_echoed": True}
    if mode == "auto":
        # Auto applies by default (like `pedacito auto --apply`); only an
        # explicit "Plan only" request keeps it read-only.
        opts["plan_only"] = jobs.auto_plan_only(opts)
        opts["apply"] = not opts["plan_only"]
    return _busy_reply(jobs.spawn(
        mode, title, lambda job: fn(job, opts),
        echo=lambda: jobs.echo_submission(mode, body)))


@post("/api/stop")
def _stop(q, body):
    return 200, {"ok": jobs.stop_job()}


@post("/api/steer")
def _steer(q, body):
    text = str(body.get("text", "")).strip()
    job = STATE.job
    if not text or job is None or not job.active:
        return 409, {"ok": False, "detail": "Nothing is running."}
    if STATE.cfg is None or not STATE.cfg.allow_steering:
        return 409, {"ok": False, "detail": "allow_steering is off."}
    STATE.steer(text)
    return 200, {"ok": True}


@post("/api/answer")
def _answer(q, body):
    ok = STATE.answer_question(int(body.get("qid") or 0),
                               str(body.get("text", "")))
    return (200, {"ok": True}) if ok else (409, {"ok": False,
                                                  "detail": "No open question."})


@post("/api/review/decide")
def _decide(q, body):
    files = body.get("files")
    ok, detail = STATE.decide_review(
        int(body.get("id") or 0), str(body.get("action", "")),
        list(files) if isinstance(files, list) else None)
    return (200 if ok else 409), {"ok": ok, "detail": detail}


@post("/api/review/rediff")
def _rediff(q, body):
    out = STATE.review_rediff(int(body.get("id") or 0), str(body.get("rel", "")),
                              str(body.get("text", "")))
    return (200, out) if out else (409, {"ok": False,
                                         "detail": "This file can't be edited."})


@post("/api/restore")
def _restore(q, body):
    ok, detail = jobs.restore_snapshot(str(body.get("snapshot", "")))
    return (200 if ok else 400), {"ok": ok, "detail": detail}


@post("/api/plan")
def _plan_save(q, body):
    job = STATE.job
    if job is not None and job.active:
        return 409, {"ok": False, "detail": "Wait for the running job first."}
    info = plan_payload()
    if not info["exists"]:
        return 404, {"ok": False, "detail": "There is no plan file yet."}
    with STATE.lock:
        path = STATE.plan_path or (STATE.root / STATE.cfg.auto_plan_file)
    Path(path).write_text(str(body.get("text", "")), encoding="utf-8")
    return 200, {"ok": True}


@post("/api/clear")
def _clear(q, body):
    STATE.clear_messages()
    return 200, {"ok": True}


# chat threads: the conversation history stored in the project itself ------

@post("/api/threads/new")
def _thread_new(q, body):
    """Start a fresh conversation in this project. The current thread stays
    saved under `.pedacito/gui/threads/`, so nothing is lost."""
    tid = STATE.new_thread(str(body.get("profile") or ""))
    return 200, {"ok": True, "id": tid}


@post("/api/threads/select")
def _thread_select(q, body):
    """Open one of this project's saved threads (switching chats)."""
    if not STATE.select_thread(str(body.get("id") or "")):
        return 400, {"ok": False, "detail": "No such saved thread."}
    return 200, {"ok": True, "id": STATE.thread_id}


@post("/api/threads/delete")
def _thread_delete(q, body):
    """Forget a thread and delete its file. If it was the active chat, the
    most recently updated survivor opens in its place."""
    if not STATE.delete_thread(str(body.get("id") or "")):
        return 400, {"ok": False, "detail": "No such saved thread."}
    return 200, {"ok": True, "id": STATE.thread_id}


@post("/api/term_clear")
def _term_clear(q, body):
    STATE.term_clear()
    return 200, {"ok": True}


@post("/api/shutdown")
def _shutdown(q, body):
    threading.Thread(target=shutdown_server, daemon=True).start()
    return 200, {"ok": True}


# settings ---------------------------------------------------------------------

def _reload_after_settings() -> dict:
    """Re-resolve the open project's config so a saved change applies now."""
    with STATE.lock:
        root, profile = STATE.root, STATE.profile_request
    if root is None:
        return {"reloaded": False}
    job = jobs.spawn("project", "applying settings",
                     lambda j: jobs.open_project(str(root), profile,
                                                 announce=False))
    return {"reloaded": bool(job),
            "reload_pending": job is None}


def _settings_write(fn):
    def route(q, body):
        try:
            data = fn(body)
        except ConfigError as e:
            return 400, {"ok": False, "detail": str(e)}
        return 200, {"ok": True, **data, **_reload_after_settings()}
    return route


POST_ROUTES["/api/settings/set"] = _settings_write(
    lambda b: settings_api.apply_changes(
        b.get("file", "user"), b.get("profile") or None,
        dict(b.get("changes") or {}), list(b.get("removes") or [])))
POST_ROUTES["/api/settings/profile"] = _settings_write(
    lambda b: settings_api.profile_action(
        b.get("file", "user"), b.get("action", ""), str(b.get("name", "")),
        dict(b.get("values") or {}), str(b.get("source", ""))))
POST_ROUTES["/api/settings/raw"] = _settings_write(
    lambda b: settings_api.save_raw(b.get("file", "user"),
                                    str(b.get("text", ""))))
POST_ROUTES["/api/settings/setup"] = _settings_write(
    lambda b: settings_api.create_user_config(
        str(b.get("base_url", "")).strip(), str(b.get("model", "")).strip(),
        b.get("vram_gb")))
POST_ROUTES["/api/settings/trust"] = _settings_write(
    lambda b: settings_api.trust_project_config())


@post("/api/settings/models")
def _models(q, body):
    return 200, settings_api.probe_server(
        str(body.get("base_url", "")), str(body.get("api_key", "")),
        str(body.get("file", "user")), str(body.get("profile", "")))


@post("/api/settings/discover")
def _discover(q, body):
    from ..discovery import discover_servers
    found = discover_servers(timeout=1.5)
    return 200, {"servers": [{"label": a, "base_url": b, "models": c}
                             for a, b, c in found]}


@post("/api/settings/scan")
def _scan(q, body):
    return 200, settings_api.scan_servers()


@post("/api/settings/reset")
def _settings_reset(q, body):
    """Wipe all profiles/roles and restore the recommended baseline defaults,
    after backing the current user config up as config.toml.reset.<stamp>.bak."""
    try:
        return 200, {"ok": True, **settings_api.reset_to_baseline()}
    except ConfigError as e:
        return 400, {"ok": False, "detail": str(e)}


@post("/api/models/scan")
def _models_scan(q, body):
    """Discover local model servers (and any explicit URLs) and list every
    importable model WITHOUT writing anything to the config."""
    urls = body.get("urls") or body.get("url") or []
    if isinstance(urls, str):
        urls = [urls]
    try:
        out = settings_api.scan_models(
            urls, vram_gb=body.get("vram_gb") or 0)
    except ConfigError as e:
        return 400, {"ok": False, "detail": str(e)}
    return 200, {"ok": True, **out}


@post("/api/models/import")
def _models_import(q, body):
    """Write only the profiles the user selected in the discovery table."""
    try:
        return 200, {"ok": True,
                     **settings_api.import_models(body.get("models") or [])}
    except ConfigError as e:
        return 400, {"ok": False, "detail": str(e)}


@post("/api/models/calibrate")
def _models_calibrate(q, body):
    """Run `pedacito calibrate`'s suite against one profile as a background
    job (it takes about a minute; progress streams to the terminal pane)."""
    name = str(body.get("profile", "")).strip()
    mode = "full" if str(body.get("mode", "")) == "full" else "quick"
    if not name:
        return 400, {"ok": False, "detail": "Pick a profile to calibrate."}
    known = settings_api.known_profiles()
    if name not in known:
        hint = ", ".join(known) if known else "none configured yet"
        return 400, {"ok": False,
                     "detail": f"no profile named {name!r} (profiles: {hint})"}
    job = jobs.spawn("calibrate", f"calibrating profile {name}",
                     lambda j: jobs.run_calibrate(j, {"profile": name,
                                                      "mode": mode}))
    return _busy_reply(job)


@get("/api/profiles")
def _profiles(q, body=None):
    return 200, settings_api.list_profiles()


# --- the handler ------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    server_version = "PedacitoGUI"

    def log_message(self, *a):
        pass

    def _headers(self, code: int, ctype: str, length: int | None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", CSP)
        self.end_headers()

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self._headers(code, ctype, len(body))
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, default=str).encode(),
                   "application/json")

    def _static(self, name: str) -> None:
        if name in ("", "/"):
            name = "index.html"
        name = name.lstrip("/")
        p = (STATIC / name).resolve()
        try:
            p.relative_to(STATIC.resolve())
        except ValueError:
            self._send(404, b"not found", "text/plain")
            return
        if not p.is_file():
            self._send(404, b"not found", "text/plain")
            return
        ctype = (STATIC_TYPES.get(p.suffix.lower())
                 or mimetypes.guess_type(p.name)[0]
                 or "application/octet-stream")
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        self._send(200, p.read_bytes(), ctype)

    def do_GET(self):
        if not review_mod.request_allowed(self, post=False):
            self._json(403, {"ok": False})
            return
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if url.path == "/events":
            self._events(query)
            return
        route = GET_ROUTES.get(url.path)
        if route is None:
            if url.path.startswith("/api/"):
                self._json(404, {"ok": False})
            else:
                self._static("index.html" if url.path == "/"
                             else url.path.replace("/static/", "", 1))
            return
        self._call(route, query, None)

    def do_POST(self):
        if not review_mod.request_allowed(self, post=True):
            self._json(403, {"ok": False})
            return
        url = urlparse(self.path)
        route = POST_ROUTES.get(url.path)
        if route is None:
            self._json(404, {"ok": False})
            return
        try:
            n = max(0, int(self.headers.get("Content-Length") or 0))
        except ValueError:
            n = 0
        if n > MAX_BODY:
            self._json(413, {"ok": False, "detail": "Request too large."})
            return
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            body = {}
        if not isinstance(body, dict):
            body = {}
        self._call(route, parse_qs(url.query), body)

    def _call(self, route, query, body) -> None:
        try:
            code, obj = route(query, body) if body is not None else route(query)
        except ConfigError as e:
            code, obj = 400, {"ok": False, "detail": str(e)}
        except Exception as e:
            print(f"[gui] {self.path}: {type(e).__name__}: {e}", file=sys.stderr)
            code, obj = 500, {"ok": False, "detail": f"{type(e).__name__}: {e}"}
        self._json(code, obj)

    def _events(self, query) -> None:
        """Server-Sent Events: everything after `since` (or Last-Event-ID)."""
        try:
            last = int(self.headers.get("Last-Event-ID")
                       or query.get("since", ["0"])[0])
        except ValueError:
            last = 0
        self._headers(200, "text/event-stream", None)
        self.close_connection = True
        try:
            self.wfile.write(b"retry: 1500\n\n")
            self.wfile.flush()
            while not STATE.bus.closed:
                events, resync = STATE.bus.wait(last, 15.0)
                if resync:
                    self.wfile.write(b"event: resync\ndata: {}\n\n")
                    self.wfile.flush()
                    return
                if not events:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                chunk = []
                for e in events:
                    last = e["seq"]
                    chunk.append(f"id: {e['seq']}\nevent: {e['type']}\n"
                                 f"data: {json.dumps(e, default=str)}\n\n")
                self.wfile.write("".join(chunk).encode())
                self.wfile.flush()
        except (OSError, ValueError):
            return


def shutdown_server() -> None:
    """Stop serve_forever() from a thread other than the one running it."""
    STATE.bus.close()
    if _SERVER is not None:
        _SERVER.shutdown()


def create_server(port: int = 0) -> LocalHTTPServer:
    """A bound (not yet serving) GUI server; `run` and the tests build on it."""
    global _SERVER
    srv = LocalHTTPServer(("127.0.0.1", port), _Handler)
    _SERVER = srv
    return srv


def run(port: int = 0, open_browser: bool = True, project: str | None = None,
        profile: str | None = None) -> None:
    """Start the GUI server; block until Ctrl-C or the page's Exit button.
    `project` (and `profile`) open that folder as soon as the server is up."""
    srv = create_server(port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"pedacito gui running at {url}  (Ctrl-C to stop)", file=sys.stderr)
    originals = install_tee()
    install_runner()
    if project:
        path = str(Path(project).expanduser().resolve())
        STATE.set_root(path)     # report any existing index before the open job
        jobs.spawn("project", "opening project",
                   lambda job: jobs.open_project(path, profile or ""))
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print(file=sys.stderr)
    finally:
        uninstall_runner()
        uninstall_tee(originals)
        STATE.bus.close()
        srv.shutdown()
        srv.server_close()
