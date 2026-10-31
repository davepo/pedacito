"""Optional MCP (Model Context Protocol) client.

MCP is deliberately opt-in (`allow_mcp = false` by default): arbitrary external
tools are the opposite of a small, predictable surface for a 9B-27B model.
When enabled, Pedacito exposes one `mcp` action and injects a short catalog of
the servers' tools into the prompt. The model calls:

    action=mcp, symbol="server/tool", content='{"argument": "value"}'

Servers are configured as command strings:

    [mcp]
    allow_mcp = true
    mcp_servers = ["python /path/to/server.py", "node /path/to/server.js"]

Each call starts the server, performs the MCP initialize handshake, invokes one
tool, and stops it. That is slower than keeping a session open, but it keeps
lifecycle, crashes, and orphan processes out of the agent loop -- important
when the server is untrusted and the run may be long.
"""

from __future__ import annotations

import json
import os
import queue
import shlex
import subprocess
import threading
from pathlib import Path

from . import proc

PROTOCOL_VERSION = "2024-11-05"
_CATALOG_CACHE: dict[tuple[str, ...], list[tuple[str, str, str]]] = {}


class MCPError(RuntimeError):
    pass


def _server_name(cmd: str) -> str:
    try:
        argv = _split_command(cmd)
    except ValueError:
        argv = [cmd]
    if not argv:
        return "mcp"
    # `python /path/server.py` and `node server.js` should be named after the
    # server script, not after the interpreter.
    script = next((a for a in argv[1:]
                   if not a.startswith("-")
                   and a.lower().endswith((".py", ".js", ".mjs", ".cjs",
                                           ".ts", ".rb", ".jar", ".go"))),
                  "")
    if script:
        return Path(script).stem.lower()
    return Path(argv[0]).name.lower()


def _split_command(cmd: str) -> list[str]:
    if os.name != "nt":
        return shlex.split(cmd)
    return [p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'"
            else p for p in shlex.split(cmd, posix=False)]


class _Session:
    def __init__(self, command: str, timeout: int = 30):
        self.command = command
        self.timeout = max(1, timeout)
        self._q: queue.Queue = queue.Queue()
        kwargs = {}
        if os.name != "nt":
            kwargs["start_new_session"] = True
        self.proc = subprocess.Popen(
            _split_command(command), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            **kwargs)
        self._next_id = 1
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        try:
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._q.put(json.loads(line))
                except ValueError:
                    continue
        except Exception:
            pass

    def _send(self, obj: dict):
        if self.proc.stdin is None:
            raise MCPError("server stdin is closed")
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: dict | None = None) -> dict:
        rid = self._next_id
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": rid, "method": method,
                    "params": params or {}})
        while True:
            try:
                msg = self._q.get(timeout=self.timeout)
            except queue.Empty:
                raise MCPError(f"{method} timed out after {self.timeout}s")
            if msg.get("id") == rid:
                if "error" in msg:
                    raise MCPError(str(msg["error"])[:400])
                return msg.get("result") or {}

    def notify(self, method: str):
        self._send({"jsonrpc": "2.0", "method": method})

    def initialize(self):
        self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "pedacito", "version": "0.1"},
        })
        self.notify("notifications/initialized")

    def close(self):
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except OSError:
            pass
        try:
            proc.terminate(self.proc)
            self.proc.wait(timeout=3)
        except Exception:
            try:
                self.proc.kill()
            except OSError:
                pass


def _servers(cfg) -> tuple[str, ...]:
    if not getattr(cfg, "allow_mcp", False):
        return ()
    return tuple(str(c) for c in (getattr(cfg, "mcp_servers", ()) or ())
                 if str(c).strip())


def catalog(cfg) -> list[tuple[str, str, str]]:
    """Return [(server/tool, description, schema_json)] for enabled servers."""
    servers = _servers(cfg)
    if not servers:
        return []
    key = tuple(servers)
    if key in _CATALOG_CACHE:
        return _CATALOG_CACHE[key]
    out: list[tuple[str, str, str]] = []
    for cmd in servers:
        name = _server_name(cmd)
        s = _Session(cmd, int(getattr(cfg, "mcp_timeout", 30) or 30))
        try:
            s.initialize()
            result = s.request("tools/list")
            for tool in result.get("tools") or []:
                tname = str(tool.get("name") or "").strip()
                if not tname:
                    continue
                desc = str(tool.get("description") or "").strip()
                schema = json.dumps(tool.get("inputSchema") or {},
                                    ensure_ascii=True)
                out.append((f"{name}/{tname}", desc, schema))
        except Exception as e:
            out.append((f"{name}/(unavailable)",
                        f"server failed: {str(e)[:200]}", "{}"))
        finally:
            s.close()
    _CATALOG_CACHE[key] = out
    return out


def render(cfg, limit: int = 2500) -> str:
    """A compact catalog for the project prompt, or '' when MCP is off."""
    entries = catalog(cfg)
    if not entries:
        return ""
    lines = ["MCP TOOLS (call with action=mcp, symbol=server/tool, "
             "content=<JSON args>):"]
    for name, desc, _schema in entries:
        lines.append(f"  {name} -- {desc[:160]}")
    text = "\n".join(lines)
    if len(text) > limit:
        text = text[:limit].rstrip() + "\n[... MCP catalog truncated]"
    return text


def call(cfg, target: str, args: dict | None = None) -> str:
    """Invoke one MCP tool. `target` is "server/tool" or a bare tool name."""
    entries = catalog(cfg)
    if not entries:
        return "ERROR: MCP is disabled or no MCP servers are configured."
    names = [e[0] for e in entries]
    if not target:
        return "MCP tools:\n" + "\n".join(f"  {n}" for n in names)
    if "/" not in target:
        hits = [n for n in names if n.rsplit("/", 1)[-1] == target]
        if len(hits) != 1:
            return (f"ERROR: unknown or ambiguous MCP tool '{target}'. "
                    f"Available: {', '.join(names[:20])}")
        target = hits[0]
    if target not in names:
        return (f"ERROR: unknown MCP tool '{target}'. "
                f"Available: {', '.join(names[:20])}")
    server_name, tool_name = target.split("/", 1)
    cmd = next((c for c in _servers(cfg)
                if _server_name(c) == server_name), "")
    if not cmd:
        return f"ERROR: MCP server '{server_name}' is not configured."
    s = _Session(cmd, int(getattr(cfg, "mcp_timeout", 30) or 30))
    try:
        s.initialize()
        result = s.request("tools/call", {
            "name": tool_name, "arguments": args or {},
        })
    except Exception as e:
        return f"ERROR: MCP call failed: {str(e)[:400]}"
    finally:
        s.close()
    content = result.get("content") or []
    bits = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            bits.append(str(item.get("text") or ""))
        elif isinstance(item, dict):
            bits.append(json.dumps(item, ensure_ascii=True)[:2000])
    text = "\n".join(b for b in bits if b).strip()
    if result.get("isError"):
        return "ERROR: MCP tool reported failure:\n" + (text or "(no detail)")
    return text or "(MCP tool returned no text content)"
