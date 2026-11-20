"""HTTP routes for the GUI: static files, JSON endpoints, and the SSE stream.

The server binds to 127.0.0.1 only and refuses foreign Host/Origin headers
(`review.request_allowed`), since it can write files. POSTs must be JSON, which
a cross-site page cannot send without a CORS preflight that this server never
grants. State-mutating POSTs additionally pass `security.post_origin_allowed`:
when a request carries an `Origin` or `Referer`, its hostname must be loopback
(or the bound host), so a website open in the same browser cannot drive this
server (CSRF).
"""

from __future__ import annotations

import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlparse

from .. import __version__ as PED_VERSION
from .. import checkpoints as checkpoints_mod
from .. import review as review_mod
from .. import prompts as prompts_mod
from .. import runner as runner_mod
from .. import security, themes as themes_mod, workspace
from .. import update as update_mod
from ..config import ConfigError
from ..local_server import LocalHTTPServer
from ..select import Matcher, is_venv_dir, suggest_context
from . import jobs, settings_api
from .core import (STATE, install_permission_broker, install_runner,
                   install_tee, uninstall_permission_broker,
                   uninstall_runner, uninstall_tee)

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

# The GUI frames project pages with `sandbox="allow-scripts"` (no
# `allow-same-origin`), so a previewed page runs in an opaque origin. Its CSP
# is separate from the app's own: inline scripts still work (static pages keep
# them inline) but eval is refused. A POST from the sandbox arrives with
# `Origin: null`, which `security.post_origin_allowed` rejects.
PREVIEW_CSP = ("default-src 'self' data: blob:; "
               "style-src 'self' 'unsafe-inline'; "
               "script-src 'self' 'unsafe-inline'; "
               "img-src 'self' data: blob:; "
               "media-src 'self' data: blob:; "
               "connect-src 'self' data: blob:; "
               "frame-ancestors 'self'")

# The cake emoji as an SVG favicon. The page declares it as a data: URI, but
# browsers still probe /favicon.ico on their own; answering with the same SVG
# keeps the console free of 404s.
FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
    '<text y=".9em" font-size="90">🍰</text></svg>'
).encode("utf-8")

_SERVER: LocalHTTPServer | None = None
# The args `run()` was started with, so the post-upgrade Restart button can
# re-exec the same GUI (same port, project and profile) in place.
_RUN_ARGS = SimpleNamespace(port=0, open_browser=True, project=None,
                            profile=None)


# --- helpers ------------------------------------------------------------------

def _within(root: Path, rel: str) -> Path | None:
    """`rel` resolved under `root`, or None when it escapes the project.
    Symlinks are resolved on both sides first (`security.resolve_safe_path`),
    so a link inside the tree pointing at `/etc/passwd` is refused too, and a
    literal `..` never survives the gate."""
    try:
        return security.resolve_safe_path(root, rel)
    except (security.SecurityError, OSError, ValueError):
        return None


def index_html(data: bytes) -> bytes:
    """The page HTML with the saved theme already on <html>, so a reload
    with no project open paints in the user's colorscheme instead of
    flashing the default until /api/state answers. theme.js still lets a
    browser-local pick override it."""
    theme = STATE.user_theme()
    if not theme or theme == "slate":
        return data
    text = data.decode("utf-8", errors="replace")
    marker = '<html lang="en">'
    if marker not in text:
        return data
    # `theme` is a closed enum from config.py; still escape defensively.
    safe = (str(theme).replace("&", "&amp;").replace('"', "&quot;")
            .replace("<", "&lt;").replace(">", "&gt;"))
    text = text.replace(marker, f'<html lang="en" data-theme="{safe}">', 1)
    return text.encode("utf-8")


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
    if security.is_sensitive_path(p):
        # Same gate as the model's `read` tool: secret/credential material
        # only reaches a viewer after an explicit human approval.
        if security.permission_broker_active():
            allowed, msg = security.request_permission(
                category=security.CAT_SENSITIVE_FILE,
                action_summary=f"Read sensitive file `{rel}`",
                detail=rel, risk_level="high", key=rel)
            if not allowed:
                raise ConfigError(msg)
        else:
            raise ConfigError(
                f"`{rel}` is a sensitive file (secret/credential material is "
                "never previewed without an explicit approval).")
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
        return {"path": "", "exists": False, "text": "", "tasks": [],
                "mtime": 0}
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        mtime = Path(path).stat().st_mtime_ns
    except OSError:
        return {"path": str(path), "exists": False, "text": "", "tasks": [],
                "mtime": 0}
    shown = str(path)
    if root is not None:
        try:
            shown = Path(path).resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return {"path": shown, "exists": True, "text": text,
            "tasks": jobs.parse_plan(text), "mtime": mtime}


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
    # (`messages` is the same field name /api/snapshot uses.) `plan` is the
    # live checklist: an Auto/Plan run ticks boxes on disk mid-job, and the
    # page must see those ticks immediately instead of waiting for the run's
    # final SSE transition.
    return 200, {**STATE.status(),
                 "messages": STATE.messages_snapshot(),
                 "plan": plan_payload()}


@get("/api/engine")
def _engine(q, body=None):
    """Engine V2 telemetry for the ACTIVITY & PLAN panel. `status.engine`
    carries the same payload; this route is the cheap standalone poll."""
    return 200, STATE.engine()


@get("/api/snapshot")
def _snapshot(q, body=None):
    return 200, STATE.snapshot()


# --- auto-update (pedacito.update): notice, manual check, upgrade, restart ----

# A pipx/pip upgrade that has not finished in ten minutes is stuck; the GUI
# kills it rather than leaving the drawer spinning forever.
UPDATE_RUN_TIMEOUT = 600.0


@get("/api/update")
def _update_get(q, body=None):
    with STATE.lock:
        info = dict(STATE.update) if STATE.update else None
        run = dict(STATE.update_run)
        cfg = STATE.cfg
    return 200, {"ok": True, "update": info, "update_run": run,
                 "check_for_updates": update_mod.updates_enabled(cfg),
                 "current": PED_VERSION}


@post("/api/update/check")
def _update_check(q, body=None):
    """Manual 'check now': bypasses the 24 h cache. Synchronous but bounded
    (2 s network timeout); with the opt-out set it makes no network call."""
    with STATE.lock:
        cfg = STATE.cfg
    info = update_mod.check_for_update(cfg, manual=True)
    STATE.note_update(info)
    return 200, {"ok": True, "update": info}


@post("/api/update/run")
def _update_run(q, body=None):
    """Run the detected upgrade command in the background, streaming its
    output into the drawer. Editable installs and the opt-out get the manual
    instructions instead (ok=False); a second click while running is a 409."""
    with STATE.lock:
        cfg = STATE.cfg
        if STATE.update_run["running"]:
            return 409, {"ok": False,
                         "detail": "An upgrade is already running."}
        info = STATE.update
    if not update_mod.updates_enabled(cfg):
        method = update_mod.detect_install_method(query_pipx=False)
        _cmd, instructions = update_mod.upgrade_command(method)
        return 200, {"ok": False, "detail": instructions}
    method = (info or {}).get("method") or update_mod.detect_install_method()
    command, instructions = update_mod.upgrade_command(method)
    if command is None:
        return 200, {"ok": False, "detail": instructions}
    cancel = STATE.update_run_start(command)

    def worker():
        started = time.monotonic()
        code = update_mod.run_upgrade(
            command, on_line=STATE.note_update_output,
            timeout=UPDATE_RUN_TIMEOUT, cancel=cancel)
        detail = ""
        if code != 0:
            if cancel.is_set():
                why = "Upgrade cancelled"
            elif time.monotonic() - started >= UPDATE_RUN_TIMEOUT:
                why = "Upgrade timed out"
            else:
                why = f"Upgrade failed (exit {code})"
            detail = f"{why}. {instructions}"
        STATE.update_run_done(code, detail)

    threading.Thread(target=worker, name="pedacito-gui-update-run",
                     daemon=True).start()
    return 202, {"ok": True, "running": True, "command": command}


@post("/api/update/cancel")
def _update_cancel(q, body=None):
    """Stop the running upgrade, including any children it spawned. A stale
    click after the upgrade finished is a 409, not a silent no-op."""
    if not STATE.cancel_update_run():
        return 409, {"ok": False, "detail": "No upgrade is running."}
    return 200, {"ok": True, "cancelling": True}


def _restart_argv() -> list[str]:
    """The same GUI on the same port, project and profile."""
    args = _RUN_ARGS
    argv = [sys.executable, "-m", "pedacito", "gui", "--port", str(args.port)]
    if not args.open_browser:
        argv.append("--no-open")
    if args.project:
        argv.append(str(args.project))
        if args.profile:
            argv += ["--profile", str(args.profile)]
    return argv


def _restart_process(argv: list[str]) -> None:
    """Replace this GUI with the freshly upgraded one. Windows has no execv
    semantics (the call returns and the old interpreter keeps running), so
    spawn the replacement and hard-exit; POSIX replaces the process in place."""
    time.sleep(0.5)          # let the HTTP answer reach the page first
    shutdown_server()
    if os.name == "nt":
        subprocess.Popen(argv)
        os._exit(0)
        return
    os.execv(sys.executable, argv)


@post("/api/update/restart")
def _update_restart(q, body=None):
    """After a successful upgrade, re-exec this GUI on the SAME port so the
    open page reconnects to the new version. Only offered when the run's
    exit code was 0."""
    with STATE.lock:
        ok = bool(STATE.update_run.get("ok"))
    if not ok:
        return 409, {"ok": False,
                     "detail": "No successful upgrade to restart into."}
    argv = _restart_argv()
    threading.Thread(target=_restart_process, args=(argv,),
                     name="pedacito-gui-restart", daemon=True).start()
    return 200, {"ok": True, "detail": "Restarting Pedacito…"}


@get("/api/files")
def _files(q, body=None):
    return 200, project_files()


def _active_context():
    """(root, index) for read-only context endpoints, lock-safe when present."""
    lock = getattr(STATE, "lock", None)
    if lock is not None:
        with lock:
            return getattr(STATE, "root", None), getattr(STATE, "idx", None)
    return getattr(STATE, "root", None), getattr(STATE, "idx", None)


@get("/api/context/suggest")
def _context_suggest(q, body=None):
    """Fast `@` autocomplete: workspace files + indexed AST symbols.

    `?q=` is the partial token after `@`. The endpoint is synchronous and
    zero-LLM (the composer calls it on every keystroke), so it must answer
    from the warm index rather than rebuilding or probing anything.
    """
    query = (q.get("q") or q.get("query") or [""])[0]
    root, idx = _active_context()
    items = suggest_context(idx if idx is not None else root, query, limit=20)
    return 200, {
        "ok": True,
        "query": query,
        "items": items,
        "suggestions": items,
        "files": [x for x in items if x.get("kind") == "file"],
        "symbols": [x for x in items if x.get("kind") == "symbol"],
    }


@get("/api/rules")
def _rules_get(q, body=None):
    """Read the active project's persistent rules (`.pedacito/rules.md`)."""
    root, _idx = _active_context()
    if root is None:
        raise ConfigError("No project is open.")
    text = prompts_mod.load_project_rules(root)
    path = prompts_mod.project_rules_path(root)
    return 200, {
        "ok": True,
        "path": ".pedacito/rules.md",
        "exists": bool(path is not None and path.is_file()),
        "text": text,
    }


@post("/api/rules")
def _rules_post(q, body):
    """Save (or clear, with empty text) `.pedacito/rules.md`."""
    root, _idx = _active_context()
    if root is None:
        raise ConfigError("No project is open.")
    body = body or {}
    text = body.get("text")
    if text is None:
        text = body.get("content")
    try:
        path = prompts_mod.save_project_rules(root, str(text or ""))
    except (OSError, ValueError) as e:
        raise ConfigError(f"Could not save project rules: {e}") from e
    return 200, {
        "ok": True,
        "path": ".pedacito/rules.md",
        "exists": path.is_file(),
        "text": prompts_mod.load_project_rules(root),
    }


@get("/api/backups")
def _backups(q, body=None):
    """Backup snapshots for the WORKSPACE panel's BACKUPS tab."""
    return 200, {"backups": backup_snapshots()}


# --- time-travel checkpoints --------------------------------------------------

def _reindex_after_restore(restored, deleted) -> None:
    """Keep an open index in step with files a checkpoint restore touched."""
    try:
        from .. import index as index_mod
        with STATE.lock:
            idx, cfg = STATE.idx, STATE.cfg
        if idx is None or cfg is None:
            return
        for rel in restored:
            idx = index_mod.reindex_file(idx, rel, cfg, save=False)
        for rel in deleted:
            idx.forget(rel)
        idx.save()
        with STATE.lock:
            STATE.idx = idx
        STATE.apply_index(idx)
    except Exception:
        pass


@get("/api/checkpoints")
def _checkpoints_list(q, body=None):
    """The checkpoint tree for the active workspace."""
    root = _project_root()
    if root is None:
        return 400, {"ok": False, "detail": "No project is open."}
    try:
        items = checkpoints_mod.list_checkpoints(root)
        head = checkpoints_mod.head_id(root)
    except OSError as e:
        return 400, {"ok": False, "detail": str(e)}
    return 200, {
        "ok": True,
        "root": str(root),
        "head_id": head,
        "checkpoints": [dict(cp) for cp in items],
        "count": len(items),
    }


@get("/api/checkpoints/diff")
def _checkpoints_diff(q, body=None):
    """Unified diff for one checkpoint, against its parent or the live tree."""
    root = _project_root()
    if root is None:
        return 400, {"ok": False, "detail": "No project is open."}
    checkpoint_id = str(q.get("id", [""])[0] or "")
    against = str(q.get("against", ["workspace"])[0] or "workspace").lower()
    if not checkpoint_id:
        return 400, {"ok": False, "detail": "An id is required."}
    if against not in ("parent", "workspace"):
        return 400, {"ok": False,
                     "detail": "against must be 'parent' or 'workspace'."}
    try:
        out = checkpoints_mod.diff_checkpoint(root, checkpoint_id,
                                              against=against)
    except checkpoints_mod.CheckpointError as e:
        return 404, {"ok": False, "detail": str(e)}
    except OSError as e:
        return 400, {"ok": False, "detail": str(e)}
    out["ok"] = True
    return 200, out


@post("/api/checkpoints/create")
def _checkpoints_create(q, body):
    """Record a manual named checkpoint (the header's pin button)."""
    root = _project_root()
    if root is None:
        return 400, {"ok": False, "detail": "No project is open."}
    label = str(body.get("label") or body.get("name")
                or "manual checkpoint")
    try:
        cp = checkpoints_mod.create_checkpoint(root, label, mode="manual")
    except (checkpoints_mod.CheckpointError, OSError) as e:
        return 400, {"ok": False, "detail": str(e)}
    STATE.add_message("system", f"Checkpoint {cp['id']} recorded.",
                      kind="checkpoint")
    return 200, {"ok": True, "checkpoint": cp}


@post("/api/checkpoints/restore")
def _checkpoints_restore(q, body):
    """Restore the whole workspace, or named files, from one checkpoint."""
    root = _project_root()
    if root is None:
        return 400, {"ok": False, "detail": "No project is open."}
    checkpoint_id = str(body.get("id") or body.get("checkpoint") or "")
    if not checkpoint_id:
        return 400, {"ok": False, "detail": "A checkpoint id is required."}
    files = body.get("files")
    if files is None and body.get("file"):
        files = [body.get("file")]
    if isinstance(files, str):
        files = [files]
    try:
        out = checkpoints_mod.restore_checkpoint(root, checkpoint_id,
                                                 files=files)
    except checkpoints_mod.CheckpointError as e:
        return 404, {"ok": False, "detail": str(e)}
    except OSError as e:
        return 400, {"ok": False, "detail": str(e)}
    _reindex_after_restore(out.get("restored") or [],
                           out.get("deleted") or [])
    detail = (f"Restored {out['files']} file(s) from {out['checkpoint']}."
              + (f" Removed {len(out['deleted'])} created file(s)."
                 if out.get("deleted") else ""))
    STATE.add_message("system", detail, kind="restored")
    out["detail"] = detail
    return 200, out


# --- 2-model sandbox arena ------------------------------------------------------

@post("/api/arena/run")
def _arena_run(q, body):
    """Start an arena comparison: {"text", "model_a", "model_b"} (contestants
    are profile names, 'provider:model' strings or {provider, model} dicts).
    Both runs happen in isolated sandboxes; the workspace is not touched."""
    body = body or {}
    return _busy_reply(jobs.spawn(
        "arena", "arena comparison",
        lambda job: jobs.run_arena(job, {**body, "_echoed": True}),
        echo=lambda: jobs.echo_submission("arena", body)))


@get("/api/arena/last")
def _arena_last(q, body=None):
    """The stored comparison (scorecard + diffs) for the open project."""
    from .. import arena as arena_mod
    root = _project_root()
    if root is None:
        return 400, {"ok": False, "detail": "No project is open."}
    return 200, {"ok": True, "arena": arena_mod.last_arena(root)}


@post("/api/arena/apply")
def _arena_apply(q, body):
    """Adopt a winner's sandbox into the workspace between two checkpoints.
    Synchronous (file copies only, no model calls); refuses while a job runs."""
    from .. import arena as arena_mod
    root = _project_root()
    if root is None:
        return 400, {"ok": False, "detail": "No project is open."}
    body = body or {}
    with STATE.lock:
        job = STATE.job
    if job is not None and job.active:
        return 409, {"ok": False, "detail": "A job is running. Wait for it "
                     "or press Stop."}
    arena_id = str(body.get("arena_id") or "").strip()
    if not arena_id:
        state = arena_mod.last_arena(root)
        arena_id = str((state or {}).get("arena_id") or "")
    if not arena_id:
        return 400, {"ok": False, "detail": "No arena to apply -- run one first."}
    winner = str(body.get("winner") or "a").strip().lower()
    try:
        out = arena_mod.apply_arena_winner(root, arena_id, winner)
    except arena_mod.ArenaError as e:
        return 400, {"ok": False, "detail": str(e)}
    except (checkpoints_mod.CheckpointError, OSError) as e:
        return 400, {"ok": False, "detail": str(e)}
    deleted = out.get("deleted") or []
    _reindex_after_restore([f for f in out.get("files_changed") or []
                            if f not in deleted], deleted)
    detail = (f"Arena winner applied: {out['model']} "
              f"({len(out['files_changed'])} file(s) changed).")
    STATE.add_message("system", detail, kind="arena")
    out["detail"] = detail
    return 200, out


# --- live runner / preview ----------------------------------------------------

@get("/api/runner/targets")
def _runner_targets(q, body=None):
    """Detected web previews and runnable commands for the workspace."""
    root = _project_root()
    if root is None:
        return 400, {"ok": False, "detail": "No project is open."}
    try:
        targets = runner_mod.detect_runnable_targets(root)
    except OSError as e:
        return 400, {"ok": False, "detail": str(e)}
    return 200, {"ok": True, **targets}


@post("/api/runner/exec")
def _runner_exec(q, body):
    """Run one detected target or a custom command; never uses a shell."""
    root = _project_root()
    if root is None:
        return 400, {"ok": False, "detail": "No project is open."}
    command = body.get("command") or body.get("cmd")
    target_id = str(body.get("id") or body.get("target") or "")
    if not command and target_id:
        try:
            targets = runner_mod.detect_runnable_targets(root)
        except OSError as e:
            return 400, {"ok": False, "detail": str(e)}
        for target in targets["commands"]:
            if target.get("id") == target_id:
                command = target
                break
        if not command:
            return 404, {"ok": False,
                         "detail": f"No runnable target {target_id!r}."}
    timeout = body.get("timeout") or runner_mod.DEFAULT_TIMEOUT
    try:
        result = runner_mod.run_project_command(root, command,
                                                timeout=timeout)
    except (OSError, ValueError) as e:
        return 400, {"ok": False, "detail": str(e)}
    return 200, result


@get("/api/file")
def _file(q, body=None):
    return 200, read_project_file(q.get("path", [""])[0])


@get("/api/preview")
def _preview_list(q, body=None):
    """The HTML pages the Live Preview tab can frame ([] when no project is
    open). Reuses the Files-tab listing so newly written pages show up too."""
    if STATE.root is None:
        return 200, {"files": []}
    try:
        listing = project_files()
    except Exception:
        listing = {"files": []}
    files = [f for f in listing.get("files", [])
             if str(f).lower().endswith((".html", ".htm"))]
    return 200, {"files": sorted(files)}


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


# --- non-blocking secret & credential guard ------------------------------------

def _security_cfg():
    """The live config (global fallback when no project is open)."""
    return _how_config()


@get("/api/security/status")
def _security_status(q, body=None):
    """Unacknowledged secrets and unignored sensitive files, masked."""
    root = _project_root()
    return 200, security.security_status(root, _security_cfg(),
                                         use_cache=False)


@post("/api/security/scan")
def _security_scan(q, body):
    """Preflight for an outbound prompt.

    The page asks this before a run: `needs_confirmation` is True only when
    the endpoint is external, the workspace policy is "warn", and the text
    carries unacknowledged secrets. Localhost/LAN providers never trigger a
    pause even when the prompt contains a key.
    """
    from .. import llm as llm_mod
    root = _project_root()
    cfg = _security_cfg()
    parts = [str(body.get("text") or "")]
    # Attachments are composed into the same outbound prompt; scan them too so
    # a pasted `.env` cannot bypass the preflight.
    for attachment in body.get("attachments") or []:
        if isinstance(attachment, dict):
            attached = str(attachment.get("text") or "")
            if attached:
                parts.append(attached)
    text = "\n\n".join(part for part in parts if part)
    prompt = security.unacknowledged_findings(
        security.scan_text_for_secrets(text, "<prompt>"), root)
    status = security.security_status(root, cfg, use_cache=False)
    external = llm_mod.is_external_provider(cfg)
    mode = security.outbound_mode(cfg)
    out = {
        **status,
        "ok": True,
        "external_provider": external,
        "workspace_findings": status.get("findings", []),
        "findings": [finding.to_dict() for finding in prompt],
        "prompt_findings": [finding.to_dict() for finding in prompt],
        "needs_confirmation": bool(prompt) and external and mode == "warn",
    }
    return 200, out


@post("/api/security/ack")
def _security_ack(q, body):
    """Persist SHA-256 fingerprints only; the API never sees a raw secret."""
    root = _project_root()
    if root is None:
        return 400, {"ok": False, "detail": "No project is open."}
    cfg = _security_cfg()
    values = body.get("fingerprints") or []
    if not isinstance(values, list):
        values = [values]
    if body.get("all"):
        status = security.security_status(root, cfg, use_cache=False)
        values = values + [f["fingerprint"] for f in status.get("findings", [])]
    acknowledged = security.acknowledge_findings(root, values)
    security.clear_security_cache(root)
    return 200, {
        "ok": True,
        "acknowledged": sorted(acknowledged),
        **security.security_status(root, cfg, use_cache=False),
    }


@post("/api/security/gitignore")
def _security_gitignore(q, body):
    """Append the missing `.env` patterns to the workspace `.gitignore`."""
    root = _project_root()
    if root is None:
        return 400, {"ok": False, "detail": "No project is open."}
    patterns = body.get("patterns")
    if not isinstance(patterns, list) or not patterns:
        patterns = [".env", ".env.*"]
    added = security.ensure_gitignore_entries(root, patterns)
    security.clear_security_cache(root)
    cfg = _security_cfg()
    return 200, {
        "ok": True,
        "added": added,
        **security.security_status(root, cfg, use_cache=False),
    }


@get("/api/mcp/tools")
def _mcp_tools(q, body=None):
    """The named-MCP-server tool inventory for the GUI: every listed tool
    with its input schema and trust flag. A down server lists nothing (5s
    budget each); the `servers` key still names what is configured."""
    from .. import mcp
    cfg = _how_config()
    servers = mcp.named_servers(cfg)
    tools = mcp.list_mcp_tools(cfg)
    return 200, {"enabled": bool(getattr(cfg, "enable_mcp", True)),
                 "servers": sorted(servers),
                 "trusted": {n: bool(s.get("trusted"))
                             for n, s in servers.items()},
                 "tools": tools}


def _how_config():
    """The live config, or a harmless default when no project is open."""
    from .. import config as config_mod
    with STATE.lock:
        root, cfg = STATE.root, STATE.cfg
    if cfg is not None:
        return cfg
    if root is None:
        return STATE.global_config()
    try:
        return config_mod.load(root=root,
                               require_hardware=False, warn_budgets=False)
    except ConfigError:
        return config_mod.Config()


def _how_reply(query: str, use_model: bool):
    """Shared GET/POST implementation for the How / Help tab."""
    from .. import how as how_mod
    query = " ".join(str(query or "").split())
    if not query:
        return 400, {"ok": False, "detail": "Ask a question first."}
    try:
        return 200, how_mod.answer_structured(
            query, cfg=_how_config(), use_model=use_model)
    except how_mod.HowError as e:
        return 400, {"ok": False, "detail": str(e)}


@get("/api/how")
def _how_get(q, body=None):
    raw = q.get("use_model", ["0"])[0].lower()
    return _how_reply(q.get("q", [""])[0],
                      raw in ("1", "true", "yes", "on"))


@post("/api/how")
def _how_post(q, body):
    raw = body.get("use_model", False)
    if isinstance(raw, str):
        use_model = raw.strip().lower() in ("1", "true", "yes", "on")
    else:
        use_model = bool(raw)
    return _how_reply(str(body.get("query", "")), use_model)


# --- named MCP servers ---------------------------------------------------------

@get("/api/mcp/servers")
def _mcp_servers(q, body=None):
    """Every configured [mcp_servers.<name>] entry plus its live tools."""
    return 200, {"ok": True, **settings_api.mcp_servers_payload()}


@post("/api/mcp/servers")
def _mcp_server_save(q, body):
    try:
        return 200, {"ok": True, **settings_api.save_mcp_server(body)}
    except ConfigError as e:
        return 400, {"ok": False, "detail": str(e)}


@post("/api/mcp/servers/delete")
def _mcp_server_delete(q, body):
    try:
        return 200, {"ok": True, **settings_api.delete_mcp_server(body)}
    except ConfigError as e:
        return 400, {"ok": False, "detail": str(e)}


@post("/api/mcp/servers/test")
def _mcp_server_test(q, body):
    try:
        result = settings_api.test_mcp_server(body)
    except ConfigError as e:
        return 400, {"ok": False, "detail": str(e)}
    return 200, {"ok": True, **result}


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


# --- project reset, memory, background servers and on-demand checks ------------

def _project_root():
    with STATE.lock:
        return STATE.root


def _effective_root() -> Path:
    """The workspace a route should act on: the open project, or the
    persistent per-user scratchpad when no project is open.

    Session history, memory, background servers and cleanup all need a
    directory to write to. Refusing them with "Open a project first" left the
    GUI's own buttons dead before the user opened anything, even though Ask
    already runs there. The scratchpad is created on demand and shared across
    reloads, so a project-less chat keeps its memory, threads and state."""
    root = _project_root()
    return root if root is not None else STATE.scratchpad_path()


@post("/api/project/reset")
def _project_reset(q, body):
    """Wipe this workspace's `.pedacito/` state exactly like `pedacito reset`:
    running background servers are stopped first, code and `.pedacito.toml`
    are untouched, and `reindex` rebuilds the index afterwards as a job.
    The auto-plan file (PLAN.md) is removed too -- a fresh start must not
    inherit the old checklist. Source files are never touched.

    With no project open this resets the scratchpad instead of refusing."""
    from .. import processes
    with STATE.lock:
        project_open = STATE.root is not None
    root = _effective_root()
    with STATE.lock:
        busy = STATE.job is not None and STATE.job.active
    if busy:
        return 409, {"ok": False, "detail": "A job is running. Stop it before "
                                            "resetting the project state."}
    dry_run = bool(body.get("dry_run"))
    keep_memory = bool(body.get("keep_memory"))
    keep_config = bool(body.get("keep_config"))
    running = [str(r.get("name")) for r in processes.list_procs(root)
               if r.get("status") == "running"]
    stopped: list[str] = []
    lock_held = False
    if not dry_run:
        try:
            workspace.acquire_lock(root)
            lock_held = True
        except workspace.ProjectBusy as e:
            return 409, {"ok": False, "detail": str(e)}
    try:
        if not dry_run:
            for name in running:
                try:
                    processes.stop(root, name)
                    stopped.append(name)
                except Exception:
                    pass
        try:
            plan = workspace.reset(root, keep_memory=keep_memory,
                                   keep_config=keep_config, dry_run=dry_run)
        except OSError as e:
            return 400, {"ok": False, "detail": str(e)}
        # The auto-plan checklist rides with the session state: a reset
        # that left PLAN.md behind would resume the old plan on the next
        # auto run. Only a regular file directly named by the setting is
        # removed -- never a symlink or an escaping path.
        plan_name = str(getattr(STATE.cfg, "auto_plan_file", "") or "PLAN.md")
        plan_name = plan_name.replace("\\", "/").strip()
        while plan_name.startswith("./"):
            plan_name = plan_name[2:]
        plan_path = root / plan_name
        plan_removed = False
        if (plan_name and not Path(plan_name).is_absolute()
                and ".." not in plan_name.split("/")
                and plan_path.is_file() and not plan_path.is_symlink()):
            if dry_run:
                plan_removed = True
            else:
                try:
                    plan_path.unlink()
                    plan_removed = True
                except OSError:
                    pass
    finally:
        if lock_held:
            workspace.release_lock(root)
    out = {"ok": True, "dry_run": dry_run, "path": str(plan.get("path", "")),
           "removed": list(plan.get("removed") or []),
           "kept": list(plan.get("kept") or []),
           "files": int(plan.get("files") or 0),
           "bytes": int(plan.get("bytes") or 0),
           "running": running, "stopped": stopped,
           "plan_file": plan_name, "plan_removed": plan_removed}
    cleared = ["session history"]
    if not keep_memory:
        cleared.append("memory")
    if plan_removed:
        cleared.append(plan_name)
    joined = (", ".join(cleared[:-1]) + ", and " + cleared[-1]
              if len(cleared) > 2 else " and ".join(cleared))
    out["summary"] = (("Would clear" if dry_run else "Cleared")
                      + f" {joined} — ready for a fresh start.")
    if dry_run:
        out["would_stop"] = running
        return 200, out
    with STATE.lock:
        STATE.idx = None
        STATE.plan_path = None
        STATE.last_backup_dir = None
    STATE.apply_index(None)
    STATE.reset_threads()
    out["reindex_started"] = False
    # A scratchpad has no project index to rebuild; `build_index` would refuse
    # the job anyway because STATE.root is still None.
    if body.get("reindex") and project_open:
        job = jobs.spawn("index", "rebuilding index after reset",
                         lambda j: jobs.build_index(j, True))
        out["reindex_started"] = job is not None
        if job is not None:
            out["job"] = job.id
    return 200, out


def _memory_payload(root) -> dict:
    from .. import memory
    entries = [{"scope": sc, "category": cat, "text": text}
               for sc, cat, text in memory.entries(root)]
    return {
        "ok": True,
        "entries": entries,
        "facts": memory.load_micro_facts(root),
        "lessons": memory.load_anti_patterns(root, prune=False),
        "promoted": memory.load_promoted_lessons(root),
        "retired": memory.load_retired_lessons(root),
        "project_file": str(memory.project_file(root)),
    }


@get("/api/memory")
def _memory_get(q, body=None):
    return 200, _memory_payload(_effective_root())


@post("/api/memory")
def _memory_post(q, body):
    from .. import memory
    root = _effective_root()
    action = str(body.get("action") or "add").strip().lower()
    out: dict = {"ok": True}
    if action == "add":
        stored, msg = memory.remember(
            root, str(body.get("text") or ""),
            category=str(body.get("category") or ""),
            scope=str(body.get("scope") or "project"))
        out.update(stored=stored, detail=msg)
        if not stored:
            out["ok"] = False
    elif action == "forget":
        needle = str(body.get("text") or body.get("needle") or "").strip()
        if not needle:
            return 400, {"ok": False, "detail": "Say which fact to forget."}
        out["forgot"] = memory.forget(root, needle,
                                      scope=str(body.get("scope")
                                                or "project"))
    elif action in ("forget_lesson", "delete_lesson"):
        lesson_id = str(body.get("id") or body.get("lesson_id") or "").strip()
        if not lesson_id:
            return 400, {"ok": False, "detail": "Name the lesson to delete."}
        out["forgot"] = memory.forget_lesson(root, lesson_id)
    else:
        return 400, {"ok": False, "detail": f"unknown memory action {action!r}"}
    ok = bool(out.get("ok", True))
    out.update(_memory_payload(root))
    out["ok"] = ok
    return 200, out


# --- skills library & creator -------------------------------------------------

def _skills_payload(root, task: str = "") -> dict:
    from .. import skills
    out: dict = {"ok": True, "skills": skills.list_skills(root)}
    if task.strip():
        out["matches"] = [
            {"name": r["name"], "score": r["score"]}
            for r in skills.select_skills(root, task)]
    return out


@get("/api/skills")
def _skills_get(q, body=None):
    """Every skill (bundled / user / project) with its scope badge; an
    optional `task` query previews which checklists select_skills injects.

    With no project open this reads the built-in library plus the user's
    global folder (and any skill saved into the scratchpad project)."""
    return 200, _skills_payload(_effective_root(), str(q.get("task", "")))


@post("/api/skills")
def _skills_save(q, body):
    """Save a custom skill: {name, description, triggers, body, scope}.

    `scope` is "project" or "user"/"global". With no project open the default
    is the user's global library; an explicit project scope saves into the
    scratchpad project instead of refusing."""
    from .. import skills
    root = _project_root()
    raw_scope = str(body.get("scope") or "").strip().lower()
    if not raw_scope:
        scope = "project" if root is not None else "user"
    elif raw_scope in ("user", "global"):
        scope = "user"
    elif raw_scope == "project":
        scope = "project"
    else:
        return 400, {"ok": False, "detail": f"unknown scope {raw_scope!r}"}
    target_root = _effective_root() if scope == "project" else root
    triggers = body.get("triggers") or []
    if isinstance(triggers, str):
        triggers = [t.strip() for t in triggers.split(",") if t.strip()]
    try:
        path = skills.save_skill(
            target_root, name=str(body.get("name") or ""),
            description=str(body.get("description") or ""),
            triggers=[str(t) for t in triggers],
            body=str(body.get("body") or ""), scope=scope)
    except ValueError as e:
        return 400, {"ok": False, "detail": str(e)}
    out = _skills_payload(_effective_root())
    out.update(saved=str(path), scope=scope)
    return 200, out


@post("/api/skills/draft")
def _skills_draft(q, body):
    """Auto-draft a skill from one sentence with the active model. Returns
    the draft for the form; saving stays an explicit second step."""
    from .. import skills
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        return 400, {"ok": False, "detail": "Describe what the skill should "
                                            "teach first."}
    with STATE.lock:
        cfg, client, profile = STATE.cfg, STATE.client, STATE.profile_request
    if client is None:
        # Project-less GUI: draft on the profile selected in the topbar.
        from .. import config as config_mod
        from ..llm import LLMClient
        try:
            cfg = config_mod.load(root=None, profile=profile or None,
                                  require_hardware=False, warn_budgets=False,
                                  upgrade_user_config=False)
            client = LLMClient(cfg)
        except (ConfigError, OSError, ValueError) as e:
            return 400, {"ok": False,
                         "detail": f"No model configured for drafting: {e}"}
    try:
        draft = skills.draft_skill_with_llm(client, prompt)
    except Exception as e:
        return 200, {"ok": False, "detail": f"draft failed: {e}"}
    return 200, {"ok": True, "draft": draft}


@post("/api/skills/delete")
def _skills_delete(q, body):
    """Delete a custom skill (built-ins are immutable). A project-scope
    delete falls back to the scratchpad when no project is open."""
    from .. import skills
    root = _project_root()
    name = str(body.get("name") or "")
    raw_scope = str(body.get("scope") or "").strip().lower()
    if not raw_scope:
        scope = "project" if root is not None else "user"
    elif raw_scope in ("user", "global"):
        scope = "user"
    elif raw_scope == "project":
        scope = "project"
    else:
        return 400, {"ok": False, "detail": f"unknown scope {raw_scope!r}"}
    target_root = _effective_root() if scope == "project" else root
    try:
        deleted = skills.delete_skill(target_root, name, scope=scope)
    except ValueError as e:
        return 400, {"ok": False, "detail": str(e)}
    if not deleted:
        detail = (f"'{name}' is a built-in skill and cannot be deleted."
                  if name.strip().lower() in skills.BUILTIN_SKILLS
                  else f"No {scope} skill named '{name}' to delete.")
        return 400, {"ok": False, "detail": detail}
    out = _skills_payload(_effective_root())
    out["deleted"] = name
    return 200, out


# --- session consolidation ------------------------------------------------------

@post("/api/session/consolidate")
def _session_consolidate(q, body):
    """Compact the live chat transcript into a structured handoff checkpoint
    (.pedacito/consolidated_session.md), then reset the transcript to that
    handoff alone so the next turn starts small. The Context Meter follows
    immediately via the client's last-call token numbers."""
    from .. import memory
    with STATE.lock:
        cfg, client = STATE.cfg, STATE.client
        job = STATE.job if STATE.job is not None and STATE.job.active else None
    root = _effective_root()
    if job is not None:
        return 409, {"ok": False,
                     "detail": "A job is running. Stop it before "
                               "consolidating the session."}
    msgs = [{"role": m.get("role"), "content": str(m.get("text") or "")}
            for m in STATE.messages_snapshot()
            if m.get("role") in ("user", "assistant") and m.get("text")]
    try:
        res = memory.consolidate_session(root, cfg, msgs or None)
    except OSError as e:
        return 400, {"ok": False, "detail": f"consolidate failed: {e}"}
    # Reset the live buffer to the handoff: one visible system card replaces
    # the compacted transcript (the thread file keeps the full history until
    # this save, so nothing is lost that wasn't distilled).
    STATE.clear_messages()
    STATE.add_message("system", res["summary"], kind="handoff",
                      consolidated=True)
    if client is not None:
        try:
            client.last_prompt_tokens = int(res["after_tokens"])
        except (AttributeError, TypeError, ValueError):
            pass
    STATE.bus.publish("status")
    return 200, {"ok": True, **res, "status": STATE.status()}


@get("/api/procs")
def _procs_get(q, body=None):
    from .. import processes
    root = _effective_root()
    return 200, {"ok": True, "processes": processes.list_procs(root)}


@post("/api/procs")
def _procs_post(q, body):
    from .. import processes
    root = _effective_root()
    action = str(body.get("action") or "list").strip().lower()
    if action == "start":
        command = body.get("command") or ""
        with STATE.lock:
            cfg = STATE.cfg
        try:
            rec = processes.start(
                root, command, name=str(body.get("name") or ""), cfg=cfg,
                wait=float(getattr(cfg, "serve_wait", 3.0) or 3.0))
        except (OSError, ValueError) as e:
            return 400, {"ok": False, "detail": str(e)}
        return 200, {"ok": True, "process": rec,
                     "processes": processes.list_procs(root)}
    if action == "stop":
        name = str(body.get("name") or "").strip()
        if not name:
            return 400, {"ok": False, "detail": "Name the process to stop."}
        detail = processes.stop(root, name)
        return 200, {"ok": not detail.startswith("ERROR"), "detail": detail,
                     "processes": processes.list_procs(root)}
    if action == "logs":
        name = str(body.get("name") or "").strip()
        if not name:
            return 400, {"ok": False, "detail": "Name the process for logs."}
        try:
            tail = int(body.get("tail") or 4000)
        except (TypeError, ValueError):
            tail = 4000
        return 200, {"ok": True, "name": name,
                     "logs": processes.logs(root, name, tail=tail)}
    if action == "list":
        return 200, {"ok": True, "processes": processes.list_procs(root)}
    return 400, {"ok": False, "detail": f"unknown procs action {action!r}"}


# --- model-server lifecycle and hardware ------------------------------------

def _servers_config():
    """The config the server manager should inspect: the open project's live
    config, else the user's global selection (never a hardware probe)."""
    with STATE.lock:
        cfg = STATE.cfg
    return cfg if cfg is not None else STATE.global_config()


def _profile_availability(config, states) -> dict:
    """Profile name -> available|unavailable|offline|unknown.

    Pure bookkeeping over the server states the caller just fetched: this
    adds no probe and never contacts a hosted endpoint. A profile is matched
    to a state by its normalized effective base_url; a hosted (non-local)
    profile is always "unknown", as is a local profile whose URL is not a
    configured server or whose server returned no model list. Only a positive
    answer with a non-empty list that lacks the model becomes "unavailable" --
    the state the GUI disables and the switch guard refuses.
    """
    from .. import servers
    out: dict[str, str] = {}
    if config is None:
        return out
    by_url: dict[str, dict] = {}
    for state in states or []:
        if not isinstance(state, dict):
            continue
        url = servers.normalize_url(str(state.get("base_url") or ""))
        if not url:
            continue
        # Duplicate spellings are already deduped by configured_local_servers;
        # if a caller hands us both, an online row wins over an offline one.
        if url not in by_url or (state.get("online")
                                 and not by_url[url].get("online")):
            by_url[url] = state
    for name, body in (getattr(config, "profiles", None) or {}).items():
        body = body if isinstance(body, dict) else {}
        base = str(body.get("base_url")
                   or getattr(config, "base_url", "") or "")
        if not base or not servers.is_local_url(base):
            out[str(name)] = "unknown"
            continue
        model = str(body.get("model")
                    or getattr(config, "model", "") or "").strip()
        state = by_url.get(servers.normalize_url(base))
        if state is None:
            out[str(name)] = "unknown"
        elif not state.get("online"):
            out[str(name)] = "offline"
        else:
            models = state.get("models")
            if not isinstance(models, (list, tuple, set)):
                models = []
            if not model or not models:
                out[str(name)] = "unknown"
            elif model in [str(m) for m in models]:
                out[str(name)] = "available"
            else:
                out[str(name)] = "unavailable"
    return out


def _server_payload(config, memory=None) -> dict:
    from .. import profiler
    from .. import servers
    # snapshot() is a superset of get_hardware_memory(): same RAM/VRAM shape,
    # plus nvidia-smi / vm_stat / winapi numbers and a source per number.
    # That is what fills the NVIDIA and macOS/Windows live telemetry bar.
    snap = memory if memory is not None else profiler.snapshot()
    out = servers.server_summary(config, memory=snap)
    out["memory"] = snap
    out["hardware"] = snap
    out["vram"] = snap["vram"]
    out["ram"] = snap["ram"]
    out["badge"] = servers.memory_badge(snap)
    out["snapshot"] = snap
    # Availability rides this existing poll: no extra interval, no extra
    # network call, and hosted endpoints are classified before any matching.
    availability = _profile_availability(config, out.get("servers") or [])
    out["profile_availability"] = availability
    with STATE.lock:
        STATE.profile_availability = dict(availability)
    return out


@get("/api/servers")
def _servers_get(q, body=None):
    """Live VRAM/RAM plus every configured local server's online state and
    resident model(s). Never errors when a server is asleep."""
    try:
        return 200, _server_payload(_servers_config())
    except Exception as e:
        from .. import profiler
        return 200, {"ok": False, "detail": f"{type(e).__name__}: {e}",
                     "memory": profiler.snapshot(),
                     "servers": []}


def _server_target(body):
    return (body.get("server") or body.get("name")
            or body.get("profile") or body.get("base_url") or "")


@post("/api/servers/load")
def _servers_load(q, body):
    """Load a model on one configured local server and return fresh
    hardware + server telemetry."""
    from .. import servers
    config = _servers_config()
    target = _server_target(body)
    model = str(body.get("model") or "").strip()
    result = servers.load_server_model(target, model, config)
    payload = _server_payload(config)
    payload.update(result)
    return (200 if result.get("ok") else 502), payload


@post("/api/servers/unload")
def _servers_unload(q, body):
    """Unload a model (or every resident model) from one server, or from all
    local servers when no target is named. Returns updated telemetry."""
    from .. import servers
    config = _servers_config()
    target = _server_target(body)
    model = str(body.get("model") or "").strip()
    if target:
        result = servers.unload_server_model(target, model, config)
        results = [result]
    else:
        results = servers.unload_all_local_servers(config)
        result = {"ok": all(r.get("ok") for r in results),
                  "action": "unload", "server": "all",
                  "detail": f"{sum(1 for r in results if r.get('ok'))} of "
                            f"{len(results)} server(s) acknowledged"}
    payload = _server_payload(config)
    payload.update(result)
    payload["results"] = results
    return (200 if result.get("ok") or not results else 502), payload


@post("/api/servers/pin")
def _servers_pin(q, body):
    """Pin or unpin a loaded Lemonade model and return fresh telemetry."""
    from .. import servers
    config = _servers_config()
    target = _server_target(body)
    model = str(body.get("model") or "").strip()
    pinned = bool(body.get("pinned", True))
    kind = str(body.get("kind") or "").strip()
    result = servers.pin_server_model(target, model, pinned, config, kind=kind)
    payload = _server_payload(config)
    payload.update(result)
    return (200 if result.get("ok") else 502), payload


@post("/api/servers/auto-evict")
def _servers_auto_evict(q, body):
    """Toggle the auto-unload-other-servers handshake and apply it live."""
    enabled = bool(body.get("enabled", body.get("auto_evict_servers", True)))
    with STATE.lock:
        cfg = STATE.cfg
        if cfg is not None:
            cfg.auto_evict_servers = enabled
    try:
        settings_api.apply_changes("user", None,
                                   {"auto_evict_servers": enabled}, [])
    except ConfigError:
        pass
    return 200, {"ok": True, "auto_evict_servers": enabled}


@post("/api/effort")
def _effort(q, body):
    """Set reasoning_effort for the live run and persist it as a user setting.

    Local-MoE and reasoning-capable hosted models both read the key; the
    transport omits it for "default" and for models without the control."""
    from .. import config as config_mod
    level = str(body.get("level",
                         body.get("reasoning_effort", "default"))).strip().lower()
    if level not in ("default", "low", "medium", "high"):
        return 400, {"ok": False,
                     "detail": "reasoning effort must be default, low, "
                               "medium or high"}
    with STATE.lock:
        cfg = STATE.cfg
        if cfg is not None:
            cfg.reasoning_effort = level
            cfg.sources["reasoning_effort"] = "GUI Effort selector"
    try:
        settings_api.apply_changes("user", None,
                                   {"reasoning_effort": level}, [])
    except ConfigError:
        # A scratchpad/no-config session still gets the live switch; the
        # setting is persisted as soon as a config file exists.
        pass
    return 200, {"ok": True, "reasoning_effort": level,
                 "model_class": (config_mod.capability_profile(cfg)
                                 if cfg is not None else "")}


@post("/api/project/check")
def _project_check(q, body):
    """Run one on-demand project check: diagnose, webcheck or domcheck."""
    from .. import domcheck, diagnostics, webcheck
    root = _project_root()
    if root is None:
        return 409, {"ok": False, "detail": "Open a project first."}
    kind = str(body.get("kind") or "").strip().lower()
    with STATE.lock:
        cfg = STATE.cfg
    cfg = cfg if cfg is not None else _how_config()
    if kind == "diagnose":
        passed, report = diagnostics.run(root, cfg)
        return 200, {"ok": True, "kind": kind, "passed": bool(passed),
                     "report": report, "findings": []}
    if kind == "webcheck":
        files = body.get("files")
        found = webcheck.run(root, files=list(files)
                             if isinstance(files, list) and files else None)
        rows = [{"rel": f.rel, "line": f.line, "message": f.message,
                 "peer": f.peer} for f in found[:500]]
        report = "\n".join(str(f) for f in found[:500])
        if len(found) > 500:
            report += f"\n[... {len(found) - 500} more finding(s)]"
        if not report:
            report = ("no web contract problems found (looked at every "
                      ".html/.js outside node_modules and friends)")
        return 200, {"ok": True, "kind": kind, "passed": not found,
                     "report": report, "findings": rows,
                     "count": len(found)}
    if kind == "domcheck":
        entry = str(body.get("entry") or getattr(cfg, "dom_entry", "") or "")
        try:
            timeout = int(body.get("timeout")
                          or getattr(cfg, "dom_timeout", 45) or 45)
        except (TypeError, ValueError):
            timeout = 45
        res = domcheck.check(root, entry=entry, timeout=timeout)
        rows = [{"kind": f.kind, "severity": f.severity,
                 "message": f.message} for f in res.findings]
        report = (res.error
                  or "\n".join(str(f) for f in res.findings)
                  or "no DOM problems found (page loaded, module graph "
                     "mounted, controls bound)")
        return 200, {"ok": True, "kind": kind, "passed": bool(res.ok),
                     "entry": res.entry, "error": res.error,
                     "report": report, "findings": rows}
    return 400, {"ok": False, "detail":
                 "kind must be diagnose, webcheck or domcheck"}


@post("/api/project/prune")
def _project_prune(q, body):
    """Retention for sessions, backups, events and log.txt.

    With no project open this prunes the scratchpad workspace."""
    root = _effective_root()
    with STATE.lock:
        cfg = STATE.cfg
    cfg = cfg if cfg is not None else _how_config()

    def _int(key, fallback):
        raw = body.get(key)
        if raw in (None, ""):
            return int(getattr(cfg, key, fallback) or 0)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise ConfigError(f"{key} must be a whole number")
        if value < 0:
            raise ConfigError(f"{key} must be zero or greater")
        return value

    keep_sessions = _int("keep_sessions", 0)
    keep_backups = _int("keep_backups", 0)
    dry_run = bool(body.get("dry_run"))
    removed = workspace.prune(
        root, keep_sessions, keep_backups,
        int(getattr(cfg, "keep_events", 20) or 0),
        int(getattr(cfg, "log_max_bytes", 2_000_000) or 0),
        dry_run=dry_run)
    verb = "would remove" if dry_run else "removed"
    return 200, {
        "ok": True, "dry_run": dry_run, "removed": removed,
        "keep_sessions": keep_sessions, "keep_backups": keep_backups,
        "summary": (f"{verb}: {removed['sessions']} session(s), "
                    f"{removed['backups']} backup snapshot(s), "
                    f"{removed['events']} event file(s), "
                    f"{removed['log_bytes']:,} byte(s) of log.txt"),
    }


def _profile_switch_refusal(cfg, name: str):
    """409 when the cached /api/servers map proved this profile's model gone.

    Only "unavailable" is refused. "unknown" (no cache, hosted profile, an
    unconfigured URL or an empty model list) and "offline" fail open, so a
    sleeping server or a not-yet-polled page can never lock the picker."""
    if not name or cfg is None:
        return None
    with STATE.lock:
        state = (STATE.profile_availability or {}).get(name)
    if state != "unavailable":
        return None
    from .. import servers
    body = (cfg.profiles or {}).get(name) or {}
    body = body if isinstance(body, dict) else {}
    model = str(body.get("model") or getattr(cfg, "model", "") or "")
    base = str(body.get("base_url") or getattr(cfg, "base_url", "") or "")
    server = base or "its server"
    if base:
        try:
            for candidate in servers.configured_local_servers(cfg):
                if servers.normalize_url(
                        candidate.get("base_url") or "") == \
                        servers.normalize_url(base):
                    server = str(candidate.get("name") or base)
                    break
        except Exception:
            pass
    return 409, {"ok": False, "detail": (
        f"model {model or '(none)'} is not available on {server} \u2014 "
        "it may have been deleted; update the profile in Settings")}


@post("/api/profile")
def _profile_switch(q, body):
    """Switch the active model profile.

    With a project open this re-opens it on the chosen profile. With no
    project it still switches the global selection (and accepts the two
    topbar toggles), so the user can pick a model before opening or creating
    anything."""
    name = str(body.get("profile", "")).strip()
    toggles = {k: body[k] for k in ("auto_route_specialties",
                                    "disable_thinking") if k in body}
    with STATE.lock:
        root = STATE.root
        cfg = STATE.cfg
    if root is None:
        try:
            cfg = STATE.global_config()
            known = sorted(cfg.profiles or {})
        except Exception:
            known = []
        if name and known and name not in known:
            return 400, {"ok": False, "detail": f"no profile named {name!r}"}
        refusal = _profile_switch_refusal(cfg, name)
        if refusal:
            return refusal
        if toggles:
            try:
                settings_api.apply_changes("user", None, toggles, [])
            except ConfigError as e:
                return 400, {"ok": False, "detail": str(e)}
        if "profile" in body:
            STATE.set_profile(name)
        cfg = STATE.global_config()
        return 200, {"ok": True, "profile": cfg.active_profile,
                     "model": cfg.model,
                     "profiles": sorted(cfg.profiles or {}),
                     "auto_route_specialties": bool(
                         cfg.auto_route_specialties),
                     "disable_thinking": bool(cfg.disable_thinking)}
    known = sorted(cfg.profiles) if cfg else []
    if name and known and name not in known:
        return 400, {"ok": False, "detail": f"no profile named {name!r}"}
    refusal = _profile_switch_refusal(cfg, name)
    if refusal:
        return refusal
    if toggles:
        try:
            settings_api.apply_changes("user", None, toggles, [])
        except ConfigError as e:
            return 400, {"ok": False, "detail": str(e)}
    # SYNCHRONOUS: swap config/client/index wiring in place. This route must
    # never claim the single job slot -- a profile switch is not a task run,
    # and a background "project" job made the Run button show Stop.
    return 200, jobs.switch_profile(name)


@post("/api/index")
def _index(q, body):
    summarise = bool(body.get("summarise", True))
    # A workspace with 0 indexable files is indexed synchronously: a valid
    # empty index is written and the project marked indexed, answered 200 with
    # a clear status. The old background job died silently on the empty case
    # (SystemExit is not an Exception) and left the page stuck on "no index".
    empty = jobs.empty_workspace_index()
    if empty is not None:
        return 200, empty
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


# Named aliases for the Ask, Auto and New modes, for API clients that address
# the mode directly. Each accepts the same body as POST /api/run.
@post("/api/ask")
def _run_ask(q, body):
    return _run(q, {**body, "mode": "explain"})


@post("/api/auto")
def _run_auto(q, body):
    return _run(q, {**body, "mode": "auto"})


@post("/api/new")
def _run_new(q, body):
    return _run(q, {**body, "mode": "new"})


@post("/api/mode-check")
def _mode_check(q, body):
    """Deterministic Task-vs-Auto suitability check for the composer.

    `{"text": ..., "mode": "auto"}` -> the classifier verdict
    (`recommended_mode`, `mismatch`, `reason`). Synchronous and zero-LLM:
    the GUI polls it (debounced) while the user types an Auto prompt, so it
    must never claim the job slot."""
    from .. import loop as loop_mod
    text = str(body.get("text") or "")
    mode = str(body.get("mode") or "auto")
    requested = "ask" if mode in ("ask", "task") else "auto"
    verdict = loop_mod.evaluate_mode_suitability(text, STATE.root, requested)
    return 200, {"ok": True, **verdict}


@post("/api/refactor/rename")
def _refactor_rename(q, body):
    """Deterministic symbol rename: AST/tokenize, zero model tokens.

    `{"old": ..., "new": ..., "file": ..., "apply": true|false}`. Without
    `apply` this is a preview: per-file diffs, nothing written. Applying
    snapshots each original first, so the Changes tab's Restore undoes it.
    Refuses while an agent run owns the workspace."""
    from .. import refactor as refactor_mod
    old = str(body.get("old") or "").strip()
    new = str(body.get("new") or "").strip()
    if not old or not new:
        return 400, {"ok": False, "detail": "Both `old` and `new` are required."}
    root = STATE.root
    if root is None:
        return 409, {"ok": False, "detail": "Open a workspace first."}
    if STATE.job is not None and STATE.job.active:
        return 409, {"ok": False, "detail": "An agent run is active; stop it "
                       "before renaming."}
    scope = str(body.get("file") or "").strip() or None
    if scope is not None and _within(root, scope) is None:
        return 400, {"ok": False, "detail": f"`{scope}` escapes the workspace."}
    apply_it = bool(body.get("apply"))
    state = {"dir": None}

    def _snapshot(rel: str, path: Path):
        from .. import fileio
        if state["dir"] is None:
            state["dir"] = workspace.new_backup_dir(root, unique=True)
        fileio.backup_copy(path, state["dir"] / rel)
        return None

    res = refactor_mod.rename_symbol_across_project(
        root, old, new, scope_file=scope, dry_run=not apply_it,
        before_write=_snapshot if apply_it else None)
    if not res.get("ok"):
        return 400, res
    if state["dir"] is not None:
        res["backup"] = str(state["dir"].relative_to(root))
    if apply_it and res["files_modified"] and STATE.idx is not None:
        # keep the GUI's index truthful after the rename (the next agent run
        # rebuilds anyway; this only prevents stale search results)
        from ..index import reindex_file
        for rel in res["files_modified"]:
            reindex_file(STATE.idx, rel, STATE.cfg, save=False)
    return 200, res


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


@post("/api/permission/decide")
def _permission_decide(q, body):
    """Answer a live permission request: decision is one of allow_once,
    allow_run, allow_always, deny. Unblocks the parked worker thread."""
    pid = str(body.get("id") or "")
    decision = str(body.get("decision") or "")
    if decision not in ("allow_once", "allow_run", "allow_always", "deny"):
        return 400, {"ok": False, "detail": f"unknown decision {decision!r}"}
    ok, detail = STATE.decide_permission(pid, decision)
    return (200 if ok else 409), {"ok": ok, "detail": detail}


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


# The Changes tab and scripts call this "undo"; it restores a backup snapshot
# (the newest when none is named), exactly like /api/restore.
POST_ROUTES["/api/undo"] = _restore


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


@post("/api/spec/refine")
def _spec_refine(q, body):
    """Prompt/Spec Architect: expand a brief composer prompt into a concrete,
    repo-aware spec (objective / files / interface contracts / requirements /
    edge cases / verification). Synchronous -- the server is threaded and no
    run job is disturbed -- and `refine_prompt` always answers: if the model
    is unreachable the deterministic architect shapes the same sections, so
    the button never dead-ends."""
    text = str(body.get("prompt", ""))
    if not text.strip():
        return 400, {"ok": False, "detail": "Type what to build first."}
    mode = str(body.get("mode") or "auto")
    profile = str(body.get("profile") or "")
    with STATE.lock:
        root, cfg, client = STATE.root, STATE.cfg, STATE.client
    if root is None:
        # No project is open: refine against the empty scratchpad on the
        # selected global profile. There is no repo snapshot or detected test
        # runner to inject, so the architect turns the idea into a standalone
        # spec the user can send to New or Ask. `refine_prompt` always
        # answers, even when the model is unreachable.
        from ..llm import LLMClient
        root = STATE.scratchpad_path()
        try:
            if cfg is None or (profile
                               and profile != getattr(cfg, "active_profile",
                                                      "")):
                cfg = STATE.global_config(profile=profile or None)
            if client is None or (profile and cfg is not None
                                  and profile != getattr(
                                      getattr(client, "cfg", None),
                                      "active_profile", "")):
                client = LLMClient(cfg) if cfg is not None else None
        except Exception:
            cfg, client = None, None
    elif profile and profile != getattr(STATE, "profile_request", ""):
        # Refine with the chosen profile's model without re-opening the
        # session: build a throwaway cfg+client for this one call.
        try:
            from .. import cli as cli_mod
            from ..llm import LLMClient
            from .core import args_for
            cfg = cli_mod._config(args_for(str(root), profile))
            client = LLMClient(cfg)
        except Exception:
            pass                       # keep the session's cfg+client
    from .. import spec as spec_mod
    try:
        got = spec_mod.refine_prompt(text, root, cfg, client, mode=mode)
    except Exception as e:
        return 200, {"ok": False, "detail": f"refine failed: {e}"}
    return 200, {"ok": True, "refined": got["refined"],
                 "original": got["original"], "contracts": got["contracts"]}


# The shorter spelling used by scripts and older clients.
POST_ROUTES["/api/refine"] = _spec_refine


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


@post("/api/threads/rename")
def _thread_rename(q, body):
    """Give a thread a hand-picked title (marked custom, so the automatic
    first-message title never overwrites it). Persisted to its thread file;
    the response carries the refreshed thread list and full status."""
    title = STATE.rename_thread(str(body.get("id") or ""),
                                str(body.get("title") or ""))
    if title is None:
        return 400, {"ok": False,
                     "detail": "No such thread (or the title was empty)."}
    return 200, {"ok": True, "id": str(body.get("id") or ""), "title": title,
                 "threads": STATE.threads_summary(), "status": STATE.status()}


@post("/api/term_clear")
def _term_clear(q, body):
    STATE.term_clear()
    return 200, {"ok": True}


@post("/api/shutdown")
def _shutdown(q, body):
    threading.Thread(target=shutdown_server, daemon=True,
                     name="pedacito-gui-shutdown").start()
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
POST_ROUTES["/api/profiles/rename"] = _settings_write(
    lambda b: settings_api.rename_profile(
        b.get("file", "user"), str(b.get("old_name", "")),
        str(b.get("new_name", ""))))
POST_ROUTES["/api/settings/scout"] = _settings_write(
    lambda b: settings_api.configure_scout(dict(b or {})))


@post("/api/settings/theme")
def _theme(q, body):
    """Persist the GUI colorscheme in the user config. Cosmetic only: no
    project reload, no jobs touched."""
    try:
        return 200, {"ok": True, **settings_api.save_theme(
            str(body.get("gui_theme", "")))}
    except ConfigError as e:
        return 400, {"ok": False, "detail": str(e)}


def _themes_payload() -> dict:
    data = themes_mod.list_themes(STATE.root)
    return {"ok": True, **data}


@get("/api/themes")
def _themes(q, body=None):
    """Every built-in and custom palette, plus the active theme's name."""
    return 200, _themes_payload()


@post("/api/themes/select")
def _themes_select(q, body):
    """Activate a theme. A project with an override updates that override."""
    name = str((body or {}).get("name") or "").strip()
    try:
        themes_mod.set_active_theme(name, STATE.root)
    except ValueError as e:
        raise ConfigError(str(e)) from e
    return 200, _themes_payload()


@post("/api/themes/save")
def _themes_save(q, body):
    """Validate, store and activate a user-created theme."""
    body = body or {}
    try:
        themes_mod.save_custom_theme(str(body.get("name") or ""),
                                     body.get("tokens"), STATE.root)
        themes_mod.set_active_theme(str(body.get("name") or ""), STATE.root)
    except ValueError as e:
        raise ConfigError(str(e)) from e
    return 200, _themes_payload()


@post("/api/themes/delete")
def _themes_delete(q, body):
    """Delete a custom theme; built-in presets are immutable."""
    name = str((body or {}).get("name") or "").strip()
    if not name:
        raise ConfigError("theme name is required")
    if not themes_mod.delete_custom_theme(name):
        if name in themes_mod.BUILTIN_THEMES:
            raise ConfigError(f"{name!r} is a built-in theme and cannot be deleted")
        raise ConfigError(f"no custom theme named {name!r}")
    return 200, _themes_payload()


@post("/api/settings/autotune")
def _autotune(q, body):
    """Auto-Tune Universal Settings: preview the recommendations for four
    questions (writes nothing), or apply them to the user config when
    `apply` is true (then the live config reloads in place)."""
    try:
        return 200, {"ok": True, **settings_api.autotune(
            dict(body or {}), bool(body.get("apply")))}
    except ConfigError as e:
        return 400, {"ok": False, "detail": str(e)}


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
    job (it takes about a minute; progress streams to the terminal pane).
    Hosted provider profiles are refused: calibration is local-only."""
    name = str(body.get("profile", "")).strip()
    mode = "full" if str(body.get("mode", "")) == "full" else "quick"
    if not name:
        return 400, {"ok": False, "detail": "Pick a profile to calibrate."}
    known = settings_api.known_profiles()
    if name not in known:
        hint = ", ".join(known) if known else "none configured yet"
        return 400, {"ok": False,
                     "detail": f"no profile named {name!r} (profiles: {hint})"}
    hosted_err = settings_api.hosted_profile_error(name)
    if hosted_err:
        return 400, {"ok": False, "detail": hosted_err}
    job = jobs.spawn("calibrate", f"calibrating profile {name}",
                     lambda j: jobs.run_calibrate(j, {"profile": name,
                                                      "mode": mode}))
    return _busy_reply(job)


# The settings modal posts the shorter spelling; same handler.
POST_ROUTES["/api/calibrate"] = _models_calibrate


@post("/api/calibrate/all")
def _calibrate_all(q, body):
    """Calibrate every LOCAL profile in turn as one background job. Hosted
    profiles are excluded; `recalibrate` re-measures fresh cards too. The
    modal polls /api/state's `calibration` map for per-profile progress."""
    recalibrate = bool(body.get("recalibrate"))
    mode = "full" if str(body.get("mode", "")) == "full" else "quick"
    job = jobs.spawn("calibrate", "calibrating all local profiles",
                     lambda j: jobs.run_calibrate_all(
                         j, {"recalibrate": recalibrate, "mode": mode}))
    return _busy_reply(job)


@get("/api/profiles")
def _profiles(q, body=None):
    return 200, settings_api.list_profiles()


@get("/api/providers/presets")
def _provider_presets(q, body=None):
    """The quick-add catalog: major hosted providers with their base URLs,
    env-var names and curated model ids (no secrets, no probing). Carries
    `ok` like the POST sibling: the UI treats a truthful `ok` as "the server
    knows this endpoint" and replaces its built-in fallback catalog."""
    return 200, {"ok": True, **settings_api.provider_presets()}


@post("/api/providers/add")
def _provider_add(q, body):
    """Create one hosted provider profile from a preset: only an API key (or
    the name of the env var holding it) is needed. Writes the profile, reloads
    the live config, and returns the fresh settings payload -- the topbar
    Profile dropdown can select it immediately."""
    try:
        return 200, {"ok": True, **settings_api.add_provider_preset(body)}
    except ConfigError as e:
        return 400, {"ok": False, "detail": str(e)}


@post("/api/providers/probe")
def _provider_probe(q, body):
    """Verify a hosted provider key live: one GET to `<base_url>/models` with
    the pasted key (or the env var that holds it), Anthropic headers included
    for Claude. Returns the provider's real model list -- each id paired with
    a profile name derived from the model -- or, when authentication/network
    access fails, `{"ok": False, "error": ...}` plus the curated fallback so
    the dialog stays usable. A probe reads nothing and writes nothing."""
    try:
        return 200, settings_api.probe_provider(body or {})
    except ConfigError as e:
        return 400, {"ok": False, "detail": str(e)}


@post("/api/providers/import")
def _provider_import(q, body):
    """Import one or many models from the hosted-provider dialog: each becomes
    its own profile named after the MODEL (never the vendor), sharing the
    dialog's base URL and key/env var and the preset's hosted budgets."""
    try:
        return 200, {"ok": True, **settings_api.import_hosted_models(body or {})}
    except ConfigError as e:
        return 400, {"ok": False, "detail": str(e)}


# --- built-in model benchmarks -------------------------------------------------

@get("/api/benchmarks")
def _benchmarks(q, body=None):
    """The five built-in diagnostic benchmarks, the profiles that may be
    tested, the default output root, recent on-disk runs, and the most recent
    result from this GUI session (for scorecard polling)."""
    from .. import benchmarks
    with STATE.lock:
        cfg, root, last = STATE.cfg, STATE.root, STATE.benchmark
        job = STATE.job if STATE.job is not None and STATE.job.active else None
        job_info = ({"id": job.id, "kind": job.kind, "title": job.title}
                    if job else None)
    base = benchmarks.default_benchmarks_root(cfg, current_root=root)
    recent = benchmarks.recent_runs(base)
    if last and last.get("project_dir"):
        recent = [r for r in recent
                  if str(r.get("project_dir")) != str(last["project_dir"])]
        recent.insert(0, last)
    catalog = benchmarks.list_benchmarks()
    return 200, {
        "ok": True,
        "benchmarks": catalog,
        "catalog": catalog,
        "profiles": settings_api.known_profiles(),
        "default_base_dir": str(base),
        "recent": recent[:12],
        "runs": recent[:12],
        "recent_runs": recent[:12],
        "last_run": last,
        "job": job_info,
    }


@post("/api/benchmarks/run")
def _benchmarks_run(q, body):
    """Spawn one benchmark job. Poll GET /api/benchmarks: its `last_run`
    carries the completed report and generated project path once the job
    ends (progress streams to Activity/Chat meanwhile)."""
    from .. import benchmarks
    benchmark_id = str(body.get("benchmark_id") or "").strip()
    profile = str(body.get("profile") or "").strip()
    base_dir = str(body.get("base_dir") or "").strip()
    try:
        benchmarks.get_benchmark(benchmark_id)
    except ConfigError as e:
        return 400, {"ok": False, "detail": str(e)}
    known = settings_api.known_profiles()
    if profile and profile not in known:
        return 400, {"ok": False,
                     "detail": f"no profile named {profile!r} "
                               f"(profiles: {', '.join(known) or 'none configured'})"}
    if base_dir:
        p = Path(base_dir).expanduser()
        if p.exists() and not p.is_dir():
            return 400, {"ok": False,
                         "detail": f"output parent is not a directory: {p}"}
        base_dir = str(p)
    opts = {"profile": profile, "benchmark_id": benchmark_id,
            "base_dir": base_dir}
    job = jobs.spawn(
        "benchmark", f"benchmark {benchmark_id} \u00b7 {profile or 'default'}",
        lambda j: jobs.run_benchmark(j, opts))
    if job is None:
        return 409, {"ok": False,
                     "detail": "Another job is running. Wait for it or stop it."}
    # The page polls GET /api/benchmarks (so it can render progress while the
    # job runs). API clients may send {"wait": true} to block until the report
    # is ready and receive it directly in this response.
    if body.get("wait"):
        try:
            timeout = float(body.get("timeout") or 1800)
        except (TypeError, ValueError):
            timeout = 1800.0
        if job.thread is not None:
            job.thread.join(max(1.0, timeout))
        with STATE.lock:
            result = STATE.benchmark
        if result and result.get("job_id") == job.id:
            return 200, {**result, "ok": True, "job": job.id}
        return 202, {"ok": True, "job": job.id, "pending": True,
                     "detail": "The benchmark is still running; poll "
                               "/api/benchmarks for its last_run."}
    return 202, {"ok": True, "job": job.id, "benchmark_id": benchmark_id,
                 "profile": profile, "base_dir": base_dir}


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
        data = p.read_bytes()
        if name == "index.html":
            data = index_html(data)
        self._send(200, data, ctype)

    def _preview(self, rel: str) -> None:
        """Serve one project file for the Live Preview iframe: read-only and
        directory-contained (`_within`), with framing headers the app's own
        CSP forbids -- a previewed page must be embeddable by the same origin
        and may run its own inline scripts/styles. Sensitive files are never
        previewed. Relative asset links inside the page resolve back through
        /preview/ as well."""
        self._serve_project_asset(rel)

    def _runner_preview(self, rel: str) -> None:
        """The Run tab's preview endpoint, with the same path-traversal gate
        as /preview/ but no HTML-page restriction: stylesheets, scripts,
        images, fonts, JSON, wasm and media all load from the project root."""
        self._serve_project_asset(rel)

    def _serve_project_asset(self, rel: str) -> None:
        root = STATE.root
        if root is None:
            self._send(404, b"no project open", "text/plain")
            return
        # unquote once, then resolve through the security gate: a literal
        # `..`, an encoded `%2e%2e`, a symlink out of the tree and an
        # absolute path all fail here before any byte is read.
        raw = unquote(str(rel or ""))
        if "\x00" in raw:
            self._send(404, b"not found", "text/plain")
            return
        p = _within(root, raw)
        if p is None or not p.is_file():
            self._send(404, b"not found", "text/plain")
            return
        if security.is_sensitive_path(p):
            self._send(403, b"sensitive file", "text/plain")
            return
        ctype = (STATIC_TYPES.get(p.suffix.lower())
                 or mimetypes.guess_type(p.name)[0]
                 or "application/octet-stream")
        if ctype.startswith("text/") or ctype.endswith("javascript") \
                or ctype in ("image/svg+xml", "application/json",
                             "application/wasm"):
            ctype += "; charset=utf-8"
        try:
            data = p.read_bytes()
        except OSError:
            self._send(404, b"not found", "text/plain")
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header("Content-Security-Policy", PREVIEW_CSP)
        self.end_headers()
        self.wfile.write(data)

    def _discard_request_body(self) -> None:
        """Consume a POST body we refuse before parsing it.

        A handler that answers and closes while the client is still sending a
        body makes Windows abort the connection (WinError 10053) before the
        client reads the 403/404/413 response. Draining the body first keeps
        the refusal graceful on every platform; the cap matches MAX_BODY so a
        hostile Content-Length cannot make us read forever."""
        try:
            n = max(0, int(self.headers.get("Content-Length") or 0))
        except ValueError:
            return
        if (self.headers.get("Expect") or "").lower() == "100-continue":
            # The client is waiting for a 100 before sending; closing with the
            # refusal is already graceful and reading here would block.
            return
        remaining = min(n, MAX_BODY)
        while remaining > 0:
            try:
                chunk = self.rfile.read(min(65536, remaining))
            except OSError:
                return
            if not chunk:
                return
            remaining -= len(chunk)

    def do_GET(self):
        if not review_mod.request_allowed(self, post=False):
            self._json(403, {"ok": False})
            return
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if url.path == "/events":
            self._events(query)
            return
        if url.path.startswith("/api/runner/preview/"):
            self._runner_preview(url.path[len("/api/runner/preview/"):])
            return
        if url.path.startswith("/preview/"):
            self._preview(url.path[len("/preview/"):])
            return
        if url.path == "/favicon.ico":
            self._send(200, FAVICON_SVG, "image/svg+xml")
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
            self._discard_request_body()
            self._json(403, {"ok": False})
            return
        # CSRF: a browser-driven POST carries Origin/Referer -- both must name
        # a loopback host (or this server's bound address). Foreign values
        # mean some website is trying to drive the local agent: refuse.
        bound = (getattr(self.server, "server_address", ("127.0.0.1", 0)) or
                 ("127.0.0.1", 0))[0]
        if not security.post_origin_allowed(self.headers, bound_host=bound):
            self._discard_request_body()
            self._json(403, {"ok": False,
                             "detail": "Origin/Referer is not loopback"})
            return
        url = urlparse(self.path)
        route = POST_ROUTES.get(url.path)
        if route is None:
            self._discard_request_body()
            self._json(404, {"ok": False})
            return
        try:
            n = max(0, int(self.headers.get("Content-Length") or 0))
        except ValueError:
            n = 0
        if n > MAX_BODY:
            self._discard_request_body()
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
    global _RUN_ARGS
    srv = create_server(port)
    _RUN_ARGS = SimpleNamespace(port=srv.server_address[1],
                                open_browser=open_browser,
                                project=project, profile=profile)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"pedacito gui running at {url}  (Ctrl-C to stop)", file=sys.stderr)
    originals = install_tee()
    install_runner()
    install_permission_broker()
    # Quiet release check on a daemon thread: never delays startup by a
    # network round-trip, silent when offline or opted out (pedacito.update).
    update_mod.check_for_update_async(on_done=STATE.note_startup_update)
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
        uninstall_permission_broker()
        uninstall_runner()
        uninstall_tee(originals)
        STATE.bus.close()
        srv.shutdown()
        srv.server_close()
