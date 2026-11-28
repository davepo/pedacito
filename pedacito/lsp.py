"""Optional Language Server Protocol client.

    from . import lsp
    print(lsp.definition(root, cfg, "src/app.py", line=12, column=5))

Pedacito does not need an LSP: the import-aware graph and tree-sitter chunker
answer most location questions. When `allow_lsp = true` and a server is
installed, this module adds exact definitions, references, hover text, and
diagnostics through the standard stdio JSON-RPC protocol.

Servers are detected from PATH by default (pyright-langserver, pylsp,
jedi-language-server, typescript-language-server, rust-analyzer, gopls) or
configured per extension:

    [profiles.local]
    allow_lsp = true
    lsp_servers = { ".py" = "pyright-langserver --stdio" }

Everything is opt-in and local: no network, no daemon, one process per
(root, command) with an atexit reaper.

The diagnostics gate (`lsp_diagnostics_gate`, also needs `allow_lsp`) is
`collect_gate_diagnostics`: version-matched diagnostics for the files before
and after an edit, drawn from one monotonic budget, then `diagnostics_delta`
keeps only new findings on changed lines. Anything but "ok" is unverified.
"""

from __future__ import annotations

import atexit
import difflib
import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from urllib.parse import unquote, urlparse

from . import env_resolve, proc, security

GATE_BUDGET_SECONDS = 2.9
_REAP_GRACE = 0.3
_GATE_STATUSES = ("ok", "unavailable", "timeout", "stale", "denied")
_STATUS_RANK = {"ok": 0, "stale": 1, "timeout": 2, "unavailable": 3, "denied": 4}

_LANGUAGE_ID = {
    ".py": "python", ".pyi": "python",
    ".ts": "typescript", ".tsx": "typescriptreact",
    ".js": "javascript", ".jsx": "javascriptreact",
    ".rs": "rust", ".go": "go", ".rb": "ruby",
    ".java": "java", ".c": "c", ".h": "cpp", ".cc": "cpp",
    ".cpp": "cpp", ".cs": "csharp", ".php": "php",
    ".kt": "kotlin", ".swift": "swift", ".scala": "scala", ".lua": "lua",
}

# (command, args) candidates, in preference order.
_DEFAULT_SERVERS = {
    ".py": [
        ("pyright-langserver", ["--stdio"]),
        ("pylsp", []),
        ("jedi-language-server", []),
    ],
    ".pyi": [("pyright-langserver", ["--stdio"]), ("pylsp", [])],
    ".ts": [("typescript-language-server", ["--stdio"])],
    ".tsx": [("typescript-language-server", ["--stdio"])],
    ".js": [("typescript-language-server", ["--stdio"])],
    ".jsx": [("typescript-language-server", ["--stdio"])],
    ".rs": [("rust-analyzer", [])],
    ".go": [("gopls", [])],
    ".rb": [("solargraph", ["stdio"])],
}

_CLIENTS_LOCK = threading.Lock()
_CLIENTS: dict[tuple[str, str, str, str], LSPClient] = {}
_EOF = object()
_MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB frame cap
_MAX_NOTES = 200
_REGISTERED_SESSIONS: set[str] = set()


def _sid() -> str:
    """The bound session id ('' when no session is bound): the process-level
    fallback key, so session-less callers share one client per (root, cmd)."""
    try:
        from . import session_ctx
        sess = session_ctx.current_session()
        return str(sess.session_id) if sess is not None else ""
    except Exception:
        return ""


def _register_lsp_cleanup(sid: str) -> None:
    """Register ONE lsp cleanup per session (idempotent), so cancelling a
    session closes only ITS language-server clients."""
    if not sid or sid in _REGISTERED_SESSIONS:
        return
    try:
        from . import session_ctx
        sess = session_ctx.current_session()
    except Exception:
        return
    if sess is None or sess.resource_registry is None:
        return
    _REGISTERED_SESSIONS.add(sid)
    try:
        sess.resource_registry.register(
            "lsp", f"lsp:{sid}", lambda: close_session(sid))
    except Exception:
        _REGISTERED_SESSIONS.discard(sid)


class LSPError(RuntimeError):
    pass


class LSPTimeout(LSPError):
    """A server request or startup ran out of its time budget."""


def _split(command: str) -> list[str]:
    import shlex
    if os.name != "nt":
        return shlex.split(command)
    return [p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'"
            else p for p in shlex.split(command, posix=False)]


def command_for(root, rel: str, cfg=None, *, environment=None) -> list[str]:
    """Configured or PATH-detected LSP command for this file, or []."""
    ext = Path(rel).suffix.lower()
    configured = getattr(cfg, "lsp_servers", {}) or {}
    if isinstance(configured, dict):
        spec = configured.get(ext) or configured.get(_LANGUAGE_ID.get(ext, ""))
        if spec:
            return _split(str(spec))
    if environment is None:
        environment = env_resolve.resolve_environment(Path(root))
    for exe, args in _DEFAULT_SERVERS.get(ext, []):
        found = env_resolve.find_executable(environment, exe)
        if found:
            return [str(found)] + list(args)
    return []


def _to_uri(path: Path) -> str:
    """file:// URI for an absolute path; handles Windows drives and UNC shares
    through pathlib (`file:///C:/...`, `file://server/share/...`)."""
    return Path(path).resolve().as_uri()


_uri = _to_uri


def _from_uri(uri: str) -> str:
    """Filesystem path for a file:// URI. Strips the leading slash of a Windows
    drive (`/C:/x` -> `C:\\x`) and rebuilds UNC paths from the authority."""
    try:
        p = urlparse(uri)
    except ValueError:
        return uri
    if p.scheme.lower() != "file":
        return uri
    path = unquote(p.path)
    if p.netloc and p.netloc.lower() != "localhost":
        return str(PureWindowsPath("//" + p.netloc + path))
    if len(path) >= 3 and path[0] == "/" and path[1].isalpha() and path[2] == ":":
        return str(PureWindowsPath(path[1:]))
    return path


def _uri_key(uri: str) -> str:
    """Comparison key for a URI or path: resolved, case-normalised on Windows,
    so `file:///c%3A/x` and `file:///C:/x` name the same file."""
    text = _from_uri(uri) if str(uri).startswith("file:") else str(uri)
    try:
        key = str(Path(text).resolve())
    except (OSError, RuntimeError, ValueError):
        key = text
    return os.path.normcase(key)


def _rel(root: Path, path: str) -> str:
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return path


class LSPClient:
    """One stdio LSP process with a request/notification reader thread."""

    def __init__(self, root, command: list[str], cfg=None, timeout: float = 10,
                 *, environment=None, startup_timeout: float | None = None):
        self.root = Path(root).resolve()
        self.command = list(command)
        self.timeout = max(1.0, float(timeout or 10))
        self._io_lock = threading.Lock()
        self._req_lock = threading.RLock()
        self._closed = False
        self._q: queue.Queue = queue.Queue()
        self._notes: list[dict] = []
        self._diagnostics: dict[str, list[dict]] = {}
        self._diag_versions: dict[str, int | None] = {}
        self._versions: dict[str, int] = {}
        self._id = 0
        if environment is None:
            environment = env_resolve.resolve_environment(self.root)
        self.environment = environment
        kwargs = {}
        if os.name != "nt":
            kwargs["start_new_session"] = True
        try:
            self.proc = subprocess.Popen(
                self.command, cwd=str(self.root), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                env=env_resolve.command_env(environment, cfg), **kwargs)
        except (OSError, ValueError) as e:
            raise LSPError(f"could not start {self.command[0]}: {e}") from e
        threading.Thread(target=self._pump, daemon=True).start()
        self._initialized = False
        legacy_timeout = self.timeout
        if startup_timeout is not None:
            self.timeout = max(0.05, float(startup_timeout))
        try:
            self.initialize()
        except (LSPError, OSError, ValueError, TypeError):
            self.close()
            raise
        finally:
            self.timeout = legacy_timeout

    def _pump(self) -> None:
        pipe = self.proc.stdout
        try:
            if pipe is None:
                return
            while True:
                msg = self._read_message(pipe)
                if msg is None:
                    return
                self._q.put(msg)
        except Exception:
            pass
        finally:
            self._q.put(_EOF)

    @staticmethod
    def _read_message(pipe) -> dict | None:
        headers = {}
        while True:
            line = pipe.readline()
            if not line:
                return None
            if line in (b"\r\n", b"\n"):
                break
            try:
                key, val = line.decode("ascii", "replace").split(":", 1)
            except ValueError:
                continue
            headers[key.strip().lower()] = val.strip()
        try:
            n = int(headers.get("content-length", "0"))
        except ValueError:
            return None
        if n <= 0 or n > _MAX_CONTENT_LENGTH:
            return None
        body = pipe.read(n)
        if not body or len(body) < n:
            return None
        try:
            msg = json.loads(body.decode("utf-8", "replace"))
        except ValueError:
            return None
        return msg if isinstance(msg, dict) else None

    def _send_unlocked(self, obj: dict) -> None:
        if self.proc.stdin is None:
            raise LSPError("language server stdin is closed")
        body = json.dumps(obj).encode("utf-8")
        self.proc.stdin.write(
            f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
        self.proc.stdin.flush()

    def _send(self, obj: dict) -> None:
        with self._io_lock:
            if self._closed:
                raise LSPError("language server stdin is closed")
            self._send_unlocked(obj)

    def _record_note(self, msg: dict) -> None:
        if msg.get("method") == "textDocument/publishDiagnostics":
            params = msg.get("params") or {}
            uri = params.get("uri")
            if uri:
                key = _uri_key(uri)
                self._diagnostics[key] = params.get("diagnostics") or []
                ver = params.get("version")
                self._diag_versions[key] = (
                    ver if isinstance(ver, int) and not isinstance(ver, bool)
                    else None)
            return
        self._notes.append(msg)
        if len(self._notes) > _MAX_NOTES:
            del self._notes[:len(self._notes) - _MAX_NOTES]

    def _is_publish_for(self, msg: dict, key: str) -> bool:
        if msg.get("method") != "textDocument/publishDiagnostics":
            return False
        uri = (msg.get("params") or {}).get("uri")
        return bool(uri) and _uri_key(uri) == key

    def _request(self, method: str, params: dict | None = None) -> dict:
        with self._req_lock:
            self._id += 1
            rid = self._id
            self._send({"jsonrpc": "2.0", "id": rid, "method": method,
                        "params": params or {}})
            deadline = time.monotonic() + self.timeout
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise LSPTimeout(f"{method} timed out after {self.timeout:.0f}s")
                try:
                    msg = self._q.get(timeout=left)
                except queue.Empty:
                    raise LSPTimeout(f"{method} timed out after {self.timeout:.0f}s")
                if msg is _EOF:
                    self._q.put(_EOF)
                    raise LSPError(f"{method} failed: language server closed stdout")
                if msg.get("id") == rid and "method" not in msg:
                    if "error" in msg:
                        raise LSPError(str(msg["error"])[:400])
                    return msg.get("result") or {}
                if "id" in msg and msg.get("method"):
                    # A server-initiated request (registerCapability, progress,
                    # configuration, ...): acknowledge with null so it never blocks
                    # the response we are waiting for.
                    try:
                        self._send({"jsonrpc": "2.0", "id": msg["id"],
                                    "result": None})
                    except (OSError, ValueError, LSPError):
                        pass
                    continue
                self._record_note(msg)

    def notify(self, method: str, params: dict | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def initialize(self) -> None:
        result = self._request("initialize", {
            "processId": os.getpid(),
            "rootUri": _uri(self.root),
            "capabilities": {
                "textDocument": {
                    "definition": {"linkSupport": True},
                    "references": {},
                    "hover": {"contentFormat": ["markdown", "plaintext"]},
                    "publishDiagnostics": {},
                },
            },
            "workspaceFolders": [{"uri": _uri(self.root),
                                  "name": self.root.name}],
        })
        self.capabilities = result.get("capabilities") or {}
        self.notify("initialized", {})
        self._initialized = True

    def open(self, rel: str) -> Path:
        try:
            path = security.resolve_safe_path(self.root, rel)
        except security.SecurityError as e:
            raise LSPError(f"could not open {rel}: {e}") from e
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise LSPError(f"could not read {rel}: {e}") from e
        uri = _uri(path)
        with self._req_lock:
            prev_ver = self._versions.get(uri)
            if prev_ver is None:
                self._versions[uri] = 1
                self.notify("textDocument/didOpen", {
                    "textDocument": {
                        "uri": uri,
                        "languageId": _LANGUAGE_ID.get(path.suffix.lower(),
                                                       "plaintext"),
                        "version": 1,
                        "text": text,
                    }})
            else:
                new_ver = prev_ver + 1
                self._versions[uri] = new_ver
                self.notify("textDocument/didChange", {
                    "textDocument": {
                        "uri": uri,
                        "version": new_ver,
                    },
                    "contentChanges": [{"text": text}],
                })
        return path

    def request_for(self, op: str, rel: str, line: int = 1,
                    column: int = 1) -> dict:
        with self._req_lock:
            path = self.open(rel)
            pos = {"textDocument": {"uri": _uri(path)},
                   "position": {"line": max(0, int(line) - 1),
                                "character": max(0, int(column) - 1)}}
            if op == "definition":
                return self._request("textDocument/definition", pos)
            if op == "references":
                pos["context"] = {"includeDeclaration": True}
                return self._request("textDocument/references", pos)
            if op == "hover":
                return self._request("textDocument/hover", pos)
            if op == "diagnostics":
                diags, fresh = self._wait_diagnostics(_to_uri(path))
                return {"_diagnostics": diags, "_fresh": fresh}
            raise LSPError(f"unknown LSP operation '{op}'")

    def _send_ack(self, msg: dict) -> None:
        try:
            self._send({"jsonrpc": "2.0", "id": msg["id"], "result": None})
        except (OSError, ValueError, LSPError):
            pass

    def _verdict(self, key: str, want: int | None) -> tuple[list[dict], bool] | None:
        """Cached diagnostics for `key` when they answer the current version.
        An unversioned publish cannot prove freshness, so it is returned as
        unverified at once rather than waited on."""
        if key not in self._diagnostics:
            return None
        ver = self._diag_versions.get(key)
        if ver is None:
            return list(self._diagnostics[key]), False
        if want is not None and ver == want:
            return list(self._diagnostics[key]), True
        return None

    def _wait_diagnostics(self, uri: str) -> tuple[list[dict], bool]:
        """Diagnostics for `uri` after its latest didOpen/didChange, and whether
        they are fresh (published for the version just sent)."""
        key = _uri_key(uri)
        want = self._versions.get(uri)
        with self._req_lock:
            # Drain queued notifications first so fresher publishDiagnostics
            # overwrite stale entries.
            eof = False
            while True:
                try:
                    msg = self._q.get_nowait()
                except queue.Empty:
                    break
                if msg is _EOF:
                    self._q.put(_EOF)
                    eof = True
                    break
                if "id" in msg and msg.get("method"):
                    self._send_ack(msg)
                    continue
                self._record_note(msg)
            verdict = self._verdict(key, want)
            if verdict is not None:
                return verdict
            if eof:
                return list(self._diagnostics.get(key, [])), False
            deadline = time.monotonic() + min(self.timeout, 5.0)
            while time.monotonic() < deadline:
                try:
                    msg = self._q.get(
                        timeout=max(0.05, deadline - time.monotonic()))
                except queue.Empty:
                    break
                if msg is _EOF:
                    self._q.put(_EOF)
                    break
                if "id" in msg and msg.get("method"):
                    self._send_ack(msg)
                    continue
                self._record_note(msg)
                if self._is_publish_for(msg, key):
                    verdict = self._verdict(key, want)
                    if verdict is not None:
                        return verdict
            return list(self._diagnostics.get(key, [])), False

    def gate_diagnostics(self, rel: str, deadline: float) -> tuple[str, int | None, list[dict]]:
        """Open `rel` at its current disk content and wait until the server
        publishes diagnostics for exactly that version. Older or delayed
        publishes are ignored. Returns (status, version, diagnostics) with
        status "ok", "stale" or "timeout"; the wait ends at `deadline`
        (monotonic). Raises LSPError when the server exits."""
        path = self.open(rel)
        uri = _to_uri(path)
        key = _uri_key(uri)
        with self._req_lock:
            want = self._versions.get(uri)
            saw_other = False
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    return ("stale" if saw_other else "timeout"), want, []
                try:
                    msg = self._q.get(timeout=left)
                except queue.Empty:
                    continue
                if msg is _EOF:
                    self._q.put(_EOF)
                    raise LSPError("language server closed stdout")
                if "id" in msg and msg.get("method"):
                    self._send_ack(msg)
                    continue
                self._record_note(msg)
                if not self._is_publish_for(msg, key):
                    continue
                ver = self._diag_versions.get(key)
                if ver is None:
                    return "stale", want, []
                if ver == want:
                    return "ok", want, list(self._diagnostics.get(key, []))
                saw_other = True

    def close(self) -> None:
        with self._io_lock:
            self._closed = True
            try:
                self._send_unlocked({"jsonrpc": "2.0", "id": self._id + 1,
                                     "method": "shutdown", "params": {}})
                self._send_unlocked({"jsonrpc": "2.0", "method": "exit"})
            except (OSError, ValueError, LSPError):
                pass
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
            except OSError:
                pass
        try:
            proc.terminate(self.proc)
            self.proc.wait(timeout=3)
        except (OSError, subprocess.SubprocessError):
            try:
                self.proc.kill()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=2)
            except (OSError, subprocess.SubprocessError):
                pass
        try:
            if self.proc.stdout:
                self.proc.stdout.close()
        except OSError:
            pass


def close_all() -> None:
    with _CLIENTS_LOCK:
        clients = list(_CLIENTS.values())
        _CLIENTS.clear()
    for c in clients:
        try:
            c.close()
        except (OSError, subprocess.SubprocessError):
            pass


def close_session(session_id: str) -> None:
    """Close and forget every LSP client owned by one session, leaving the
    clients of any sibling session untouched."""
    sid = str(session_id)
    with _CLIENTS_LOCK:
        clients = [c for k, c in _CLIENTS.items() if k[0] == sid]
        for key in [k for k in _CLIENTS if k[0] == sid]:
            _CLIENTS.pop(key, None)
    _REGISTERED_SESSIONS.discard(sid)
    for c in clients:
        try:
            c.close()
        except (OSError, subprocess.SubprocessError):
            pass


atexit.register(close_all)


def client_for(root, rel: str, cfg=None, *, environment=None,
               startup_timeout: float | None = None) -> LSPClient:
    if environment is None:
        environment = env_resolve.resolve_environment(Path(root))
    command = command_for(root, rel, cfg, environment=environment)
    if not command:
        raise LSPError(
            f"no language server configured or found for {rel}. Install one "
            "(pyright-langserver/pylsp/typescript-language-server/"
            "rust-analyzer/gopls) or set lsp_servers in the profile.")
    env_key = ";".join(str(d) for d in environment.bin_dirs)
    key = (_sid(), str(Path(root).resolve()), " ".join(command), env_key)
    stale: LSPClient | None = None
    with _CLIENTS_LOCK:
        c = _CLIENTS.get(key)
        if c is not None and c.proc.poll() is None:
            return c
        if c is not None:
            stale = _CLIENTS.pop(key, None)
    if stale is not None:
        try:
            stale.close()
        except (OSError, subprocess.SubprocessError):
            pass
    c = LSPClient(root, command, cfg,
                  timeout=float(getattr(cfg, "lsp_timeout", 10) or 10),
                  environment=environment, startup_timeout=startup_timeout)
    prev: LSPClient | None = None
    with _CLIENTS_LOCK:
        prev = _CLIENTS.get(key)
        _CLIENTS[key] = c
    if prev is not None and prev is not c:
        try:
            prev.close()
        except (OSError, subprocess.SubprocessError):
            pass
    _register_lsp_cleanup(key[0])
    return c


@dataclass(frozen=True)
class Diagnostic:
    """One redacted, bounded diagnostic. `file` is project-relative POSIX;
    `line`/`end_line` are 1-based and inclusive; `version` is the document
    version the server reported, or None when it reported none."""

    file: str
    line: int
    end_line: int
    code: str
    message: str
    severity: int
    version: int | None


def diff_line_maps(old: str, new: str) -> tuple[set[int], dict[int, int]]:
    """Changed NEW-side line numbers, and a map from each unchanged OLD line
    to its NEW line (both 1-based). Deleted lines have no mapping."""
    a = str(old).splitlines()
    b = str(new).splitlines()
    changed: set[int] = set()
    mapping: dict[int, int] = {}
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                mapping[i1 + k + 1] = j1 + k + 1
        else:
            changed.update(range(j1 + 1, j2 + 1))
    return changed, mapping


def _pre_existing(d: Diagnostic, before: tuple[Diagnostic, ...],
                  line_maps: dict[str, dict[int, int]] | None) -> bool:
    """Whether `d` has a baseline twin. Without line maps any same-message
    baseline finding counts. With them, the baseline line must map onto
    `d.line`; a baseline line that was deleted or rewritten counts as a match,
    so uncertainty never turns an old finding into a repair target."""
    maps = (line_maps or {}).get(d.file, {}) if line_maps is not None else None
    for b in before:
        if b.file != d.file or b.code != d.code or b.message != d.message:
            continue
        if maps is None:
            return True
        mapped = maps.get(b.line)
        if mapped is None or mapped == d.line:
            return True
    return False


def diagnostics_delta(before: tuple[Diagnostic, ...],
                      after: tuple[Diagnostic, ...],
                      changed_lines: dict[str, set[int]], *,
                      line_maps: dict[str, dict[int, int]] | None = None,
                      ) -> tuple[Diagnostic, ...]:
    """The actionable findings: fresh (versioned) after-diagnostics that
    intersect a changed new-file line, multiline ranges included, and have no
    baseline twin. `line_maps` (per file, old line -> new line) is what
    `diff_line_maps` returns; leave it None when no mapping is known."""
    out: list[Diagnostic] = []
    for d in after:
        changed = changed_lines.get(d.file)
        if not changed or d.version is None:
            continue
        if not any(n in changed for n in range(d.line, max(d.line, d.end_line) + 1)):
            continue
        if _pre_existing(d, before, line_maps):
            continue
        out.append(d)
    return tuple(out)


def _norm_rel(raw) -> str:
    return str(raw or "").strip().replace("\\", "/").lstrip("/")


def _gate_diagnostic(rel: str, d, version: int | None) -> Diagnostic | None:
    if not isinstance(d, dict):
        return None
    r = d.get("range") or {}
    start = r.get("start") or {}
    end = r.get("end") or start
    try:
        line = int(start.get("line", 0)) + 1
        end_line = max(line, int(end.get("line", start.get("line", 0))) + 1)
        sev = int(d.get("severity") or 0)
    except (TypeError, ValueError):
        return None
    code = "" if d.get("code") is None else str(d.get("code"))[:80]
    message = security.redact_secrets_in_text(str(d.get("message") or ""))
    return Diagnostic(rel, line, end_line, code, message[:300], sev, version)


def _evict(client: LSPClient) -> None:
    """Forget and close one client (a timed-out server is not reused)."""
    with _CLIENTS_LOCK:
        for key in [k for k, v in _CLIENTS.items() if v is client]:
            _CLIENTS.pop(key, None)
    try:
        client.close()
    except (OSError, subprocess.SubprocessError):
        pass


def _worse(a: str, b: str) -> str:
    return a if _STATUS_RANK[a] >= _STATUS_RANK[b] else b


def _gate_one(root: Path, rel: str, cfg, environment,
              deadline: float) -> tuple[str, tuple[Diagnostic, ...]]:
    if not command_for(root, rel, cfg, environment=environment):
        return "unavailable", ()
    client: LSPClient | None = None
    try:
        client = client_for(root, rel, cfg, environment=environment,
                            startup_timeout=deadline - _REAP_GRACE - time.monotonic())
        status, version, raw = client.gate_diagnostics(rel, deadline - _REAP_GRACE)
    except LSPTimeout:
        return "timeout", ()
    except (LSPError, OSError, ValueError, TypeError, KeyError, IndexError):
        return "unavailable", ()
    if status == "timeout" and client is not None:
        _evict(client)
    diags = tuple(d for d in (_gate_diagnostic(rel, x, version) for x in raw)
                  if d is not None)
    return status, diags


def collect_gate_diagnostics(root: Path, files: list[str], *, cfg,
                             environment, deadline: float,
                             ) -> tuple[str, tuple[Diagnostic, ...]]:
    """Current fresh diagnostics for the existing `files`, all drawn from one
    monotonic `deadline` (startup, waits and reaping included; no per-file
    reset). Returns (status, diagnostics) with status ok, unavailable,
    timeout, stale or denied; any status other than ok means the result
    cannot be used as proof. Missing paths have no diagnostics here: a file the
    edit created has an empty baseline, and a deleted file has nothing to check.
    Never raises for server or budget failures."""
    if not (getattr(cfg, "lsp_diagnostics_gate", False)
            and getattr(cfg, "allow_lsp", False)):
        return "denied", ()
    root = Path(root).resolve()
    status = "ok"
    out: list[Diagnostic] = []
    seen: set[str] = set()
    for raw in files:
        raw_str = str(raw or "").strip()
        if not raw_str:
            status = _worse(status, "denied")
            continue
        try:
            path = security.resolve_safe_path(root, raw_str)
        except security.SecurityError:
            status = _worse(status, "denied")
            continue
        rel = path.relative_to(root).as_posix()
        if rel in seen or not path.is_file():
            continue
        seen.add(rel)
        if deadline - time.monotonic() - _REAP_GRACE <= 0.05:
            return _worse(status, "timeout"), tuple(out)
        st, diags = _gate_one(root, rel, cfg, environment, deadline)
        status = _worse(status, st)
        out.extend(diags)
        if st == "timeout":
            break
    return status, tuple(out)


def _fmt_location(root: Path, loc: dict) -> str:
    if not isinstance(loc, dict):
        return ""
    uri = loc.get("uri") or loc.get("targetUri") or ""
    rng = loc.get("range") or loc.get("targetRange") or {}
    start = (rng.get("start") or {}) if isinstance(rng, dict) else {}
    if not uri:
        return ""
    return (f"{_rel(root, _from_uri(uri))}:{int(start.get('line', 0)) + 1}:"
            f"{int(start.get('character', 0)) + 1}")


def _fmt_hover(result) -> str:
    contents = (result or {}).get("contents") if isinstance(result, dict) else None
    if isinstance(contents, str):
        return contents
    if isinstance(contents, dict):
        return str(contents.get("value") or contents.get("text") or "")
    if isinstance(contents, list):
        bits = []
        for c in contents:
            if isinstance(c, str):
                bits.append(c)
            elif isinstance(c, dict):
                bits.append(str(c.get("value") or c.get("text") or ""))
        return "\n".join(b for b in bits if b)
    return ""


def _fmt_diagnostics(root: Path, rel: str, diags) -> str:
    if not diags:
        return f"No diagnostics for {rel}."
    lines = [f"Diagnostics for {rel}:"]
    for d in diags[:40]:
        if not isinstance(d, dict):
            continue
        r = d.get("range") or {}
        start = r.get("start") or {}
        sev = {1: "error", 2: "warning", 3: "info", 4: "hint"}.get(
            d.get("severity"), "diagnostic")
        lines.append(f"  {rel}:{int(start.get('line', 0)) + 1}:"
                     f"{int(start.get('character', 0)) + 1}: {sev}: "
                     f"{str(d.get('message') or '')[:300]}")
    return "\n".join(lines)


def result(root, cfg, op: str, rel: str, line: int = 1,
           column: int = 1, *, environment=None) -> str:
    """Run one LSP operation and return model/CLI-friendly text."""
    root = Path(root).resolve()
    rel = (rel or "").replace("\\", "/").lstrip("/")
    if not rel:
        return "ERROR: lsp needs a file."
    try:
        c = client_for(root, rel, cfg, environment=environment)
        data = c.request_for(op, rel, line, column)
    except LSPError as e:
        return f"ERROR: LSP {op} failed: {e}"
    except (OSError, ValueError, KeyError, TypeError, IndexError) as e:
        return f"ERROR: LSP {op} failed: {str(e)[:300]}"
    if op in ("definition", "references"):
        if isinstance(data, dict) and "result" in data:
            data = data["result"]
        items = data if isinstance(data, list) else (
            [data] if isinstance(data, dict) and data else [])
        locs = [x for x in (_fmt_location(root, item) for item in items) if x]
        if not locs:
            return f"No {op} found for {rel}:{line}:{column}."
        label = "Definition" if op == "definition" else "References"
        return label + ":\n" + "\n".join(f"  {x}" for x in locs[:60])
    if op == "hover":
        text = _fmt_hover(data).strip()
        return text or f"No hover information for {rel}:{line}:{column}."
    if op == "diagnostics":
        text = _fmt_diagnostics(root, rel, data.get("_diagnostics")
                                if isinstance(data, dict) else [])
        if isinstance(data, dict) and data.get("_fresh") is False:
            text += ("\n(unverified: no diagnostics matched the current file "
                     "version; they may be stale)")
        return text
    return f"ERROR: unknown LSP operation '{op}'."
