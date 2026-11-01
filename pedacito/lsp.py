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
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import proc, safety

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

_CLIENTS: dict[tuple[str, str], LSPClient] = {}


class LSPError(RuntimeError):
    pass


def _split(command: str) -> list[str]:
    import shlex
    if os.name != "nt":
        return shlex.split(command)
    return [p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'"
            else p for p in shlex.split(command, posix=False)]


def command_for(root, rel: str, cfg=None) -> list[str]:
    """Configured or PATH-detected LSP command for this file, or []."""
    ext = Path(rel).suffix.lower()
    configured = getattr(cfg, "lsp_servers", {}) or {}
    if isinstance(configured, dict):
        spec = configured.get(ext) or configured.get(_LANGUAGE_ID.get(ext, ""))
        if spec:
            return _split(str(spec))
    for exe, args in _DEFAULT_SERVERS.get(ext, []):
        found = shutil.which(exe)
        if found:
            return [found] + list(args)
    return []


def _uri(path: Path) -> str:
    return path.resolve().as_uri()


def _from_uri(uri: str) -> str:
    try:
        p = urlparse(uri)
        if p.scheme == "file":
            return unquote(p.path)
    except ValueError:
        pass
    return uri


def _rel(root: Path, path: str) -> str:
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return path


class LSPClient:
    """One stdio LSP process with a request/notification reader thread."""

    def __init__(self, root, command: list[str], cfg=None, timeout: float = 10):
        self.root = Path(root).resolve()
        self.command = list(command)
        self.timeout = max(1.0, float(timeout or 10))
        self._q: queue.Queue = queue.Queue()
        self._notes: list[dict] = []
        self._id = 0
        kwargs = {}
        if os.name != "nt":
            kwargs["start_new_session"] = True
        try:
            self.proc = subprocess.Popen(
                self.command, cwd=str(self.root), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                env=safety.child_env(cfg), **kwargs)
        except (OSError, ValueError) as e:
            raise LSPError(f"could not start {self.command[0]}: {e}") from e
        threading.Thread(target=self._pump, daemon=True).start()
        self._initialized = False
        try:
            self.initialize()
        except (LSPError, OSError, ValueError, TypeError):
            self.close()
            raise

    def _pump(self) -> None:
        pipe = self.proc.stdout
        if pipe is None:
            return
        while True:
            msg = self._read_message(pipe)
            if msg is None:
                return
            self._q.put(msg)

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
        if n <= 0:
            return None
        body = pipe.read(n)
        if not body:
            return None
        try:
            msg = json.loads(body.decode("utf-8", "replace"))
        except ValueError:
            return None
        return msg if isinstance(msg, dict) else None

    def _send(self, obj: dict) -> None:
        if self.proc.stdin is None:
            raise LSPError("language server stdin is closed")
        body = json.dumps(obj).encode("utf-8")
        self.proc.stdin.write(
            f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
        self.proc.stdin.flush()

    def _request(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        rid = self._id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method,
                    "params": params or {}})
        deadline = time.monotonic() + self.timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise LSPError(f"{method} timed out after {self.timeout:.0f}s")
            try:
                msg = self._q.get(timeout=left)
            except queue.Empty:
                raise LSPError(f"{method} timed out after {self.timeout:.0f}s")
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
                except (OSError, ValueError):
                    pass
                continue
            self._notes.append(msg)

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
        path = (self.root / rel).resolve()
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise LSPError(f"could not read {rel}: {e}") from e
        self.notify("textDocument/didOpen", {
            "textDocument": {
                "uri": _uri(path),
                "languageId": _LANGUAGE_ID.get(path.suffix.lower(), "plaintext"),
                "version": 1,
                "text": text,
            }})
        return path

    def request_for(self, op: str, rel: str, line: int = 1,
                    column: int = 1) -> dict:
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
            return {"_diagnostics": self._wait_diagnostics(_uri(path))}
        raise LSPError(f"unknown LSP operation '{op}'")

    def _wait_diagnostics(self, uri: str) -> list[dict]:
        # LSP pushes diagnostics as a notification after didOpen/didChange.
        for note in self._notes:
            if note.get("method") == "textDocument/publishDiagnostics" \
                    and (note.get("params") or {}).get("uri") == uri:
                return (note.get("params") or {}).get("diagnostics") or []
        deadline = time.monotonic() + min(self.timeout, 5.0)
        while time.monotonic() < deadline:
            try:
                msg = self._q.get(timeout=max(0.05, deadline - time.monotonic()))
            except queue.Empty:
                break
            if msg.get("method") == "textDocument/publishDiagnostics" \
                    and (msg.get("params") or {}).get("uri") == uri:
                return (msg.get("params") or {}).get("diagnostics") or []
            if "id" in msg and msg.get("method"):
                try:
                    self._send({"jsonrpc": "2.0", "id": msg["id"],
                                "result": None})
                except (OSError, ValueError):
                    pass
                continue
            self._notes.append(msg)
        return []

    def close(self) -> None:
        try:
            self._send({"jsonrpc": "2.0", "id": self._id + 1,
                        "method": "shutdown", "params": {}})
            self._send({"jsonrpc": "2.0", "method": "exit"})
        except (OSError, ValueError):
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


def close_all() -> None:
    for key in list(_CLIENTS):
        try:
            _CLIENTS.pop(key).close()
        except (OSError, subprocess.SubprocessError):
            pass


atexit.register(close_all)


def client_for(root, rel: str, cfg=None) -> LSPClient:
    command = command_for(root, rel, cfg)
    if not command:
        raise LSPError(
            f"no language server configured or found for {rel}. Install one "
            "(pyright-langserver/pylsp/typescript-language-server/"
            "rust-analyzer/gopls) or set lsp_servers in the profile.")
    key = (str(Path(root).resolve()), " ".join(command))
    c = _CLIENTS.get(key)
    if c is not None and c.proc.poll() is None:
        return c
    if c is not None:
        try:
            c.close()
        except (OSError, subprocess.SubprocessError):
            pass
        _CLIENTS.pop(key, None)
    c = LSPClient(root, command, cfg,
                  timeout=float(getattr(cfg, "lsp_timeout", 10) or 10))
    _CLIENTS[key] = c
    return c


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
           column: int = 1) -> str:
    """Run one LSP operation and return model/CLI-friendly text."""
    root = Path(root).resolve()
    rel = (rel or "").replace("\\", "/").lstrip("/")
    if not rel:
        return "ERROR: lsp needs a file."
    try:
        c = client_for(root, rel, cfg)
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
        return _fmt_diagnostics(root, rel, data.get("_diagnostics")
                                if isinstance(data, dict) else [])
    return f"ERROR: unknown LSP operation '{op}'."
