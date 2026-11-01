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

Tool calls keep one live session per server command (a module-level cache):
the server is started and initialized on first use and reused afterwards,
which matters because a gather loop may call the same tool many times. A dead
or desynced session is dropped and the call retried once on a fresh server;
`close_all()` (registered with atexit) reaps every session when the process
ends, so no orphan server outlives the run. The catalog listing still uses
one throwaway session per server, since it is fetched at most once.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import shlex
import subprocess
import threading
import time
from pathlib import Path

from . import proc

PROTOCOL_VERSION = "2024-11-05"
_CATALOG_CACHE: dict[tuple[str, ...], list[tuple[str, str, str]]] = {}
_CATALOG_TIME: dict[tuple[str, ...], float] = {}
# A catalog holding an "(unavailable)" entry (a server that was down at
# startup) is retried after this many seconds instead of poisoning the whole
# run: servers come up late, and a restart should not need a new session.
_RETRY_UNAVAILABLE_AFTER = 30.0
# Live keep-alive sessions by server command string (see module docstring).
_SESSIONS: dict[str, "_Session"] = {}


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


def server_names(servers) -> dict[str, str]:
    """{unique name: command}. Two servers named after the same script get a
    numeric suffix, so a call routes to the right one."""
    out: dict[str, str] = {}
    for cmd in servers:
        base = _server_name(cmd)
        name, n = base, 2
        while name in out:
            name = f"{base}-{n}"
            n += 1
        out[name] = cmd
    return out


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
                    msg = json.loads(line)
                except ValueError:
                    continue
                if isinstance(msg, dict):
                    self._q.put(msg)
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
            if msg.get("id") == rid and "method" not in msg:
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


# Per-command (consecutive failures, retry-not-before): a server that will not
# start is skipped for a growing cool-down instead of costing a timeout per call.
_COOLDOWN: dict[str, tuple[int, float]] = {}


def _cooling(cmd: str) -> float:
    """Seconds left before `cmd` may be started again (0 = now)."""
    n, until = _COOLDOWN.get(cmd, (0, 0.0))
    return max(0.0, until - time.time()) if n >= 2 else 0.0


def _note_failure(cmd: str) -> None:
    n = _COOLDOWN.get(cmd, (0, 0.0))[0] + 1
    _COOLDOWN[cmd] = (n, time.time() + min(300, 15 * 2 ** (n - 1)))


def _live_session(cmd: str, timeout: int) -> "_Session":
    """The cached initialised session for `cmd`, restarted when the cached
    process has exited. Callers `_drop_session` after a failed request so the
    next call starts a fresh server."""
    s = _SESSIONS.get(cmd)
    if s is not None:
        if s.proc.poll() is None:
            return s
        s.close()
        _SESSIONS.pop(cmd, None)
    wait = _cooling(cmd)
    if wait:
        raise MCPError(f"server keeps failing to start; retrying in {wait:.0f}s")
    s = None
    try:
        s = _Session(cmd, timeout)
        s.initialize()
    except Exception:
        if s is not None:
            s.close()
        _note_failure(cmd)
        raise
    _COOLDOWN.pop(cmd, None)
    _SESSIONS[cmd] = s
    return s


def _drop_session(cmd: str) -> None:
    """Close and forget `cmd`'s cached session after a failure."""
    s = _SESSIONS.pop(cmd, None)
    if s is not None:
        s.close()


def close_all() -> None:
    """Close every cached MCP session. Registered with atexit so a kept-alive
    server never outlives the process; tests call it between cases."""
    for cmd in list(_SESSIONS):
        _drop_session(cmd)
    _COOLDOWN.clear()


atexit.register(close_all)


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
        cached = _CATALOG_CACHE[key]
        down = any("(unavailable)" in e[0] for e in cached)
        if not down or time.time() - _CATALOG_TIME.get(key, 0.0) \
                < _RETRY_UNAVAILABLE_AFTER:
            return cached
        _CATALOG_CACHE.pop(key, None)
    out: list[tuple[str, str, str]] = []
    for name, cmd in server_names(servers).items():
        try:
            s = _Session(cmd, int(getattr(cfg, "mcp_timeout", 30) or 30))
        except (OSError, ValueError) as e:
            out.append((f"{name}/(unavailable)",
                        f"server failed: {str(e)[:200]}", "{}"))
            continue
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
    _CATALOG_TIME[key] = time.time()
    return out


_TYPE_NAMES = {"string": str, "integer": int, "number": (int, float),
               "boolean": bool, "array": list, "object": dict}


def signature(schema_json: str) -> str:
    """`(path: string, limit?: integer, mode?: a|b)` from a JSON input schema,
    so the model sees WHAT arguments a tool takes, not just its name."""
    try:
        schema = json.loads(schema_json or "{}")
    except ValueError:
        return "()"
    props = schema.get("properties") or {}
    required = set(schema.get("required") or ())
    parts = []
    for k, v in list(props.items())[:6]:
        v = v if isinstance(v, dict) else {}
        t = v.get("type", "any")
        if v.get("enum"):
            t = "|".join(str(e) for e in v["enum"][:4])
        parts.append(f"{k}{'' if k in required else '?'}: {t}")
    if len(props) > 6:
        parts.append("...")
    return "(" + ", ".join(parts) + ")"


def validate_args(schema_json: str, args: dict) -> str:
    """'' when `args` fit the tool's input schema, else an ERROR text naming
    the missing/mistyped arguments and the expected signature."""
    try:
        schema = json.loads(schema_json or "{}")
    except ValueError:
        return ""
    if not isinstance(schema, dict) or not schema:
        return ""
    props = schema.get("properties") or {}
    problems = []
    missing = [k for k in (schema.get("required") or []) if k not in args]
    if missing:
        problems.append("missing required argument(s): " + ", ".join(missing))
    for k, v in args.items():
        spec = props.get(k)
        if spec is None:
            if schema.get("additionalProperties") is False:
                problems.append(f"unknown argument `{k}`")
            continue
        want = _TYPE_NAMES.get(spec.get("type"))
        if want and not isinstance(v, want):
            problems.append(f"`{k}` should be {spec.get('type')}")
        elif spec.get("type") in ("integer", "number") and isinstance(v, bool):
            problems.append(f"`{k}` should be {spec.get('type')}")
        elif spec.get("enum") and v not in spec["enum"]:
            problems.append(f"`{k}` must be one of {spec['enum'][:6]}")
    if not problems:
        return ""
    return ("ERROR: arguments do not match the tool's schema: "
            + "; ".join(problems) + f". Expected {signature(schema_json)}.")


def render(cfg, limit: int = 2500) -> str:
    """A compact catalog for the project prompt, or '' when MCP is off."""
    entries = catalog(cfg)
    if not entries:
        return ""
    lines = ["MCP TOOLS (call with action=mcp, symbol=server/tool, "
             "content=<JSON args>):"]
    for name, desc, schema in entries:
        sig = signature(schema) if schema and schema != "{}" else ""
        lines.append(f"  {name}{sig} -- {desc[:160]}")
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
    cmd = server_names(_servers(cfg)).get(server_name, "")
    if not cmd:
        return f"ERROR: MCP server '{server_name}' is not configured."
    schema = next((e[2] for e in entries if e[0] == target), "{}")
    bad = validate_args(schema, args or {})
    if bad:
        return bad
    timeout = int(getattr(cfg, "mcp_timeout", 30) or 30)
    params = {"name": tool_name, "arguments": args or {}}
    result: dict = {}
    for attempt in (1, 2):
        try:
            s = _live_session(cmd, timeout)
            result = s.request("tools/call", params)
            break
        except Exception as e:
            # The session may be dead or desynced mid-request: drop it, and on
            # the first failure retry once on a fresh server.
            _drop_session(cmd)
            if attempt == 2 or _cooling(cmd):
                return f"ERROR: MCP call failed: {str(e)[:400]}"
    content = result.get("content") or []
    bits = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            bits.append(str(item.get("text") or ""))
        elif isinstance(item, dict):
            bits.append(json.dumps(item, ensure_ascii=True)[:2000])
    text = "\n".join(b for b in bits if b).strip()
    cap = max(500, int(getattr(cfg, "max_shell_output_chars", 8000) or 8000))
    if len(text) > cap:
        text = text[:cap] + f"\n[MCP result truncated at {cap} chars]"
    if result.get("isError"):
        return "ERROR: MCP tool reported failure:\n" + (text or "(no detail)")
    return text or "(MCP tool returned no text content)"
