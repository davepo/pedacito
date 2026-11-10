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

Sprint R adds a second, pruned path: named `[mcp_servers.<name>]` tables
(command + args for a stdio server, or url for a Streamable HTTP one, plus
env/enabled/trusted). `list_mcp_tools` returns descriptors,
`call_mcp_tool` invokes one named tool and caps the result at
`cfg.max_read_tokens` tokens, and the tools layer injects only the handful of
tools that match the current task (`relevant_tools`), asking the permission
broker first for servers not marked `trusted`. Everything here is stdlib only.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import re
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import proc

PROTOCOL_VERSION = "2024-11-05"
_CATALOG_CACHE: dict[tuple[str, ...], list[tuple[str, str, str]]] = {}
_CATALOG_TIME: dict[tuple[str, ...], float] = {}
# Prompts and resources are listed with the same short retry window as tools.
_EXTRA_CACHE: dict[tuple[str, tuple[str, ...]],
                   list[tuple[str, str, str]]] = {}
_EXTRA_TIME: dict[tuple[str, tuple[str, ...]], float] = {}
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
    def __init__(self, command: str, timeout: int = 30, *, argv=None,
                 env: dict | None = None):
        self.command = command
        self.timeout = max(1, timeout)
        self._q: queue.Queue = queue.Queue()
        kwargs = {}
        if os.name != "nt":
            kwargs["start_new_session"] = True
        if env:
            child = dict(os.environ)
            for k, v in env.items():
                if str(k):
                    child[str(k)] = str(v)
            kwargs["env"] = child
        self.proc = subprocess.Popen(
            list(argv) if argv else _split_command(command),
            stdin=subprocess.PIPE,
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
    _EXTRA_CACHE.clear()
    _EXTRA_TIME.clear()
    # Named (Sprint R) sessions and their tool listings.
    for name in list(_NAMED_SESSIONS):
        _drop_named_session(name)
    _NAMED_TOOLS.clear()
    _NAMED_TOOLS_TIME.clear()


atexit.register(close_all)


def _servers(cfg) -> tuple[str, ...]:
    if not getattr(cfg, "allow_mcp", False):
        return ()
    raw = getattr(cfg, "mcp_servers", ()) or ()
    if isinstance(raw, dict):
        # Named [mcp_servers.<name>] tables belong to the pruned `mcp_call`
        # path (named_servers / list_mcp_tools), not the legacy catalog.
        return ()
    return tuple(str(c) for c in raw if str(c).strip())


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


def _prompt_signature(args_json: str) -> str:
    """`(file: string, style?: string)` from a prompts/list arguments array."""
    try:
        args = json.loads(args_json or "[]")
    except ValueError:
        return "()"
    parts = []
    for arg in (args or [])[:6]:
        if not isinstance(arg, dict):
            continue
        name = str(arg.get("name") or "").strip()
        if name:
            parts.append(f"{name}{'' if arg.get('required') else '?'}")
    if len(args or []) > 6:
        parts.append("...")
    return "(" + ", ".join(parts) + ")"


def _extra(cfg, kind: str, method: str, parse) -> list[tuple[str, str, str]]:
    """Cached prompts/resources listing. Servers that do not implement the
    method (or are down) contribute nothing, and the result is retried after
    the same short window as the tool catalog."""
    servers = _servers(cfg)
    if not servers:
        return []
    key = (kind, tuple(servers))
    if key in _EXTRA_CACHE:
        if time.time() - _EXTRA_TIME.get(key, 0.0) < _RETRY_UNAVAILABLE_AFTER:
            return _EXTRA_CACHE[key]
        _EXTRA_CACHE.pop(key, None)
    out: list[tuple[str, str, str]] = []
    timeout = int(getattr(cfg, "mcp_timeout", 30) or 30)
    for name, cmd in server_names(servers).items():
        s = None
        try:
            s = _Session(cmd, timeout)
            s.initialize()
            result = s.request(method)
            out.extend(parse(name, result))
        except Exception:
            pass                    # method unsupported or server unavailable
        finally:
            if s is not None:
                s.close()
    _EXTRA_CACHE[key] = out
    _EXTRA_TIME[key] = time.time()
    return out


def prompts(cfg) -> list[tuple[str, str, str]]:
    """[(server/name, description, arguments_json)] for enabled servers."""
    def parse(server, result):
        out = []
        for p in (result or {}).get("prompts") or []:
            if not isinstance(p, dict):
                continue
            name = str(p.get("name") or "").strip()
            if name:
                out.append((f"{server}/{name}",
                            str(p.get("description") or "").strip(),
                            json.dumps(p.get("arguments") or [],
                                       ensure_ascii=True)))
        return out
    return _extra(cfg, "prompts", "prompts/list", parse)


def resources(cfg) -> list[tuple[str, str, str]]:
    """[(server/uri, description, mimeType)] for enabled servers."""
    def parse(server, result):
        out = []
        for r in (result or {}).get("resources") or []:
            if not isinstance(r, dict):
                continue
            uri = str(r.get("uri") or "").strip()
            if uri:
                desc = str(r.get("description") or r.get("name") or "").strip()
                out.append((f"{server}/{uri}", desc,
                            str(r.get("mimeType") or "").strip()))
        return out
    return _extra(cfg, "resources", "resources/list", parse)


def _resolve_extra(entries, target: str) -> str:
    """Resolve `server/name` or a bare name/URI to a catalog entry."""
    names = [e[0] for e in entries]
    if target in names:
        return target
    tail = target.rsplit("/", 1)[-1]
    hits = [n for n in names if n.rsplit("/", 1)[-1] == tail]
    return hits[0] if len(hits) == 1 else ""


def _content_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if value.get("type") == "text":
            return str(value.get("text") or "")
        return ""
    if isinstance(value, list):
        return "\n".join(_content_text(v) for v in value if _content_text(v))
    return ""


def _cap(cfg, text: str) -> str:
    cap = max(500, int(getattr(cfg, "max_shell_output_chars", 8000) or 8000))
    if len(text) > cap:
        return text[:cap] + f"\n[MCP result truncated at {cap} chars]"
    return text


def prompt(cfg, target: str, args: dict | None = None) -> str:
    """Load one MCP prompt. `target` is `server/name` or a bare name."""
    entries = prompts(cfg)
    if not entries:
        return "ERROR: no MCP prompts are available (allow_mcp and mcp_servers)."
    full = _resolve_extra(entries, target)
    if not full:
        return (f"ERROR: unknown or ambiguous MCP prompt '{target}'. "
                f"Available: {', '.join(e[0] for e in entries[:20])}")
    server, name = full.split("/", 1)
    cmd = server_names(_servers(cfg)).get(server, "")
    if not cmd:
        return f"ERROR: MCP server '{server}' is not configured."
    timeout = int(getattr(cfg, "mcp_timeout", 30) or 30)
    try:
        s = _live_session(cmd, timeout)
        result = s.request("prompts/get",
                           {"name": name, "arguments": args or {}})
    except Exception as e:
        _drop_session(cmd)
        return f"ERROR: MCP prompt failed: {str(e)[:400]}"
    bits = []
    desc = str(result.get("description") or "").strip()
    if desc:
        bits.append(desc)
    for msg in result.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")
        text = _content_text(msg.get("content"))
        if text:
            bits.append(f"[{role}] {text}")
    return _cap(cfg, "\n".join(bits).strip() or "(prompt returned no text)")


def resource(cfg, target: str, args: dict | None = None) -> str:
    """Read one MCP resource. `target` is `server/uri` or a bare URI."""
    entries = resources(cfg)
    if not entries:
        return ("ERROR: no MCP resources are available (allow_mcp and "
                "mcp_servers).")
    full = _resolve_extra(entries, target)
    if not full:
        return (f"ERROR: unknown or ambiguous MCP resource '{target}'. "
                f"Available: {', '.join(e[0] for e in entries[:20])}")
    server, uri = full.split("/", 1)
    cmd = server_names(_servers(cfg)).get(server, "")
    if not cmd:
        return f"ERROR: MCP server '{server}' is not configured."
    timeout = int(getattr(cfg, "mcp_timeout", 30) or 30)
    try:
        s = _live_session(cmd, timeout)
        result = s.request("resources/read", {"uri": uri})
    except Exception as e:
        _drop_session(cmd)
        return f"ERROR: MCP resource read failed: {str(e)[:400]}"
    bits = []
    for item in result.get("contents") or []:
        if not isinstance(item, dict):
            continue
        if item.get("text") is not None:
            bits.append(str(item.get("text") or ""))
        elif item.get("blob") is not None:
            n = len(str(item.get("blob") or ""))
            bits.append(f"[binary resource: {item.get('mimeType') or 'unknown'}"
                        f", base64 length {n}]")
    return _cap(cfg, "\n".join(b for b in bits if b).strip()
                or "(resource returned no text)")


def render(cfg, limit: int = 2500) -> str:
    """A compact catalog for the project prompt, or '' when MCP is off."""
    entries = catalog(cfg)
    prompt_entries = prompts(cfg)
    resource_entries = resources(cfg)
    if not (entries or prompt_entries or resource_entries):
        return ""
    lines = []
    if entries:
        lines.append("MCP TOOLS (call with action=mcp, symbol=server/tool, "
                     "content=<JSON args>):")
        for name, desc, schema in entries:
            sig = signature(schema) if schema and schema != "{}" else ""
            lines.append(f"  {name}{sig} -- {desc[:160]}")
    if prompt_entries:
        lines.append("MCP PROMPTS (action=mcp, symbol=prompt:server/name, "
                     "content=<JSON args>):")
        for name, desc, args in prompt_entries:
            lines.append(f"  prompt:{name}{_prompt_signature(args)} -- "
                         f"{desc[:160]}")
    if resource_entries:
        lines.append("MCP RESOURCES (action=mcp, symbol=resource:server/uri):")
        for name, desc, mime in resource_entries:
            tail = f" [{mime}]" if mime else ""
            lines.append(f"  resource:{name}{tail} -- {desc[:160]}")
    text = "\n".join(lines)
    if len(text) > limit:
        text = text[:limit].rstrip() + "\n[... MCP catalog truncated]"
    return text


def call(cfg, target: str, args: dict | None = None) -> str:
    """Invoke one MCP tool, prompt, or resource.

    Targets: `server/tool`, `prompt:server/name`, `resource:server/uri`, or a
    bare tool/prompt name. A bare resource URI is resolved by suffix."""
    target = (target or "").strip()
    if target.startswith("prompt:"):
        return prompt(cfg, target[len("prompt:"):].strip(), args)
    if target.startswith("resource:"):
        return resource(cfg, target[len("resource:"):].strip(), args)
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


# ---------------------------------------------------------------------------
# Sprint R: named servers, pruned injection, permission-gated calls
# ---------------------------------------------------------------------------
# The legacy path above exposes EVERY tool of EVERY server to the model at
# once. The named-table path instead lists each configured server's tools
# with a short (5s) budget, caches the listing for the session, and hands
# the prompt layer a relevance filter so only the few tools matching the
# current task are ever shown. A call runs through the same JSON-RPC plumbing
# but routes by server name, and the result is capped at `max_read_tokens`
# tokens so one verbose server cannot flood the gather context.

_NAMED_SESSIONS: dict[str, "_Session | _HttpSession"] = {}
_NAMED_TOOLS: dict[str, list[dict]] = {}
_NAMED_TOOLS_TIME: dict[str, float] = {}
_NAMED_LIST_TIMEOUT = 5.0      # seconds per server for a tools/list listing


def named_servers(cfg) -> dict[str, dict]:
    """{name: spec} for the `[mcp_servers.<name>]` table format. Returns {}
    for the legacy list-of-commands format, for `enable_mcp = false`, and for
    entries that are disabled or name no transport. Each spec is normalized to
    command/args/url/env/enabled/trusted with defaults applied."""
    if not getattr(cfg, "enable_mcp", True):
        return {}
    raw = getattr(cfg, "mcp_servers", None)
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict] = {}
    for name, body in raw.items():
        if not isinstance(body, dict) or not str(name).strip():
            continue
        if not bool(body.get("enabled", True)):
            continue
        command = str(body.get("command") or "").strip()
        url = str(body.get("url") or "").strip()
        if not command and not url:
            continue
        args = body.get("args")
        if isinstance(args, (list, tuple)):
            args = [str(a) for a in args]
        elif isinstance(args, str) and args.strip():
            args = _split_command(args)
        else:
            args = []
        env = body.get("env")
        out[str(name)] = {
            "command": command,
            "args": args,
            "url": url,
            "env": dict(env) if isinstance(env, dict) else {},
            "enabled": True,
            "trusted": bool(body.get("trusted", False)),
        }
    return out


class _HttpSession:
    """MCP Streamable HTTP client on stdlib urllib: one JSON-RPC message per
    POST, the server's `Mcp-Session-Id` echoed back on every later request,
    and an SSE body reduced to the message carrying our request id."""

    def __init__(self, url: str, timeout: int = 30, env: dict | None = None):
        self.url = url
        self.timeout = max(1, timeout)
        self.env = env or {}
        self._next_id = 1
        self.session_id = ""

    def _post(self, obj: dict, want_reply: bool):
        headers = {"Content-Type": "application/json",
                   "Accept": "application/json, text/event-stream"}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        # A server table's env is the natural place for an auth token; the two
        # conventional spellings become headers, everything else is ignored.
        for k, v in self.env.items():
            key = str(k)
            if key.lower() in ("authorization", "x_api_key", "x-api-key"):
                headers[key.replace("_", "-")] = str(v)
        data = json.dumps(obj).encode("utf-8")
        req = urllib.request.Request(self.url, data=data, headers=headers,
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read().decode("utf-8", "replace")
                sid = r.headers.get("Mcp-Session-Id") or ""
                ctype = r.headers.get("Content-Type") or ""
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            raise MCPError(f"HTTP {e.code}: {detail or e.reason}") from e
        except (urllib.error.URLError, OSError) as e:
            raise MCPError(f"HTTP MCP server unreachable: {str(e)[:200]}") from e
        if sid:
            self.session_id = sid
        if not want_reply:
            return {}
        return self._pick_reply(raw, ctype, obj.get("id"))

    @staticmethod
    def _pick_reply(raw: str, ctype: str, rid) -> dict:
        if "text/event-stream" in ctype or raw.lstrip().startswith("event:"):
            for line in raw.splitlines():
                if not line.startswith("data:"):
                    continue
                try:
                    msg = json.loads(line[5:].strip())
                except ValueError:
                    continue
                if (isinstance(msg, dict) and msg.get("id") == rid
                        and "method" not in msg):
                    if "error" in msg:
                        raise MCPError(str(msg["error"])[:400])
                    return msg.get("result") or {}
            raise MCPError("no matching reply in the SSE stream")
        try:
            msg = json.loads(raw or "{}")
        except ValueError as e:
            raise MCPError(f"malformed JSON reply: {str(e)[:200]}") from e
        if not isinstance(msg, dict):
            raise MCPError("unexpected JSON reply shape")
        if "error" in msg:
            raise MCPError(str(msg["error"])[:400])
        return msg.get("result") or {}

    def request(self, method: str, params: dict | None = None) -> dict:
        rid = self._next_id
        self._next_id += 1
        return self._post({"jsonrpc": "2.0", "id": rid, "method": method,
                           "params": params or {}}, want_reply=True)

    def notify(self, method: str):
        try:
            self._post({"jsonrpc": "2.0", "method": method}, want_reply=False)
        except MCPError:
            pass       # notifications legitimately reply 202-empty

    def initialize(self):
        self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "pedacito", "version": "0.1"},
        })
        self.notify("notifications/initialized")

    def close(self):
        pass


def _named_session_key(name: str, spec: dict) -> str:
    """Identity of a server definition: changing command/args/url restarts the
    cached session instead of reusing a stale one."""
    return json.dumps([name, spec.get("command", ""), spec.get("args", []),
                       spec.get("url", ""),
                       sorted((str(k), str(v))
                               for k, v in (spec.get("env") or {}).items())],
                      sort_keys=True)


def _named_session(name: str, spec: dict, timeout: int):
    """The cached initialised session for one named server (stdio process or
    HTTP endpoint), restarted when the cached process has exited."""
    s = _NAMED_SESSIONS.get(name)
    if s is not None:
        if isinstance(s, _Session) and s.proc.poll() is not None:
            s.close()
            _NAMED_SESSIONS.pop(name, None)
        else:
            return s
    if spec.get("url"):
        s = _HttpSession(spec["url"], timeout, spec.get("env"))
    else:
        argv = [spec["command"], *spec.get("args", [])]
        s = _Session(f"mcp:{name}", timeout, argv=argv, env=spec.get("env"))
    try:
        s.initialize()
    except Exception:
        s.close()
        raise
    _NAMED_SESSIONS[name] = s
    return s


def _drop_named_session(name: str) -> None:
    """Close and forget one named server's session after a failure."""
    s = _NAMED_SESSIONS.pop(name, None)
    if s is not None:
        s.close()


def _named_tools(cfg, name: str, spec: dict) -> list[dict]:
    """One server's tool listing [{name, description, input_schema}], cached
    for the session; a down or slow server contributes nothing and is retried
    after the same cool-down window as the legacy catalog."""
    cached = _NAMED_TOOLS.get(name)
    if cached is not None:
        if cached or time.time() - _NAMED_TOOLS_TIME.get(name, 0.0) \
                < _RETRY_UNAVAILABLE_AFTER:
            return cached
        _NAMED_TOOLS.pop(name, None)
    out: list[dict] = []
    try:
        s = _named_session(name, spec, int(_NAMED_LIST_TIMEOUT))
        result = s.request("tools/list")
    except Exception:
        _drop_named_session(name)
        _NAMED_TOOLS[name] = []
        _NAMED_TOOLS_TIME[name] = time.time()
        return []
    for tool in result.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        tname = str(tool.get("name") or "").strip()
        if not tname:
            continue
        schema = tool.get("inputSchema")
        out.append({"name": tname,
                    "description": str(tool.get("description") or "").strip(),
                    "input_schema": schema if isinstance(schema, dict) else {}})
    _NAMED_TOOLS[name] = out
    _NAMED_TOOLS_TIME[name] = time.time()
    return out


def list_mcp_tools(cfg) -> list[dict]:
    """Descriptors {server, name, description, input_schema, trusted} for every
    enabled named server. Tool listings run with a 5s budget per server and
    are cached for the session; an unreachable server lists nothing."""
    out: list[dict] = []
    for name, spec in named_servers(cfg).items():
        trusted = bool(spec.get("trusted"))
        for tool in _named_tools(cfg, name, spec):
            out.append({"server": name, "name": tool["name"],
                        "description": tool["description"],
                        "input_schema": tool["input_schema"],
                        "trusted": trusted})
    return out


def relevant_tools(descriptors: list[dict], task: str,
                   limit: int = 4) -> list[dict]:
    """The few injected descriptors whose server/tool/description words
    appear in the task text, most matches first; ties keep listing order. An
    empty task or no matches returns nothing: the point of the prune is that
    the model never sees a tool the task cannot need."""
    words = {w for w in re.findall(r"[a-z0-9_]{4,}", (task or "").lower())}
    if not words:
        return []
    scored: list[tuple[int, int, dict]] = []
    for i, d in enumerate(descriptors):
        hay = " ".join((str(d.get("server", "")), str(d.get("name", "")),
                        str(d.get("description", "")))).lower()
        toks = set(re.findall(r"[a-z0-9_]+", hay))
        score = len(words & toks)
        if score:
            scored.append((-score, i, d))
    scored.sort()
    return [d for _, _, d in scored[:limit]]


def compact_line(d: dict) -> str:
    """One prompt line for an injected tool: `server.name(args) -- summary`."""
    sig = signature(json.dumps(d.get("input_schema") or {}))
    desc = re.sub(r"\s+", " ", str(d.get("description") or ""))[:120]
    return f"  {d['server']}.{d['name']}{sig} -- {desc}"


def call_mcp_tool(cfg, server_name: str, tool_name: str,
                  arguments: dict | None = None,
                  timeout: float = 15) -> str:
    """Invoke one named server's tool and return its text result, capped at
    roughly `cfg.max_read_tokens` tokens. The tool must be one the server
    actually lists (the listing doubles as the argument check), a dead session
    is restarted and the call retried once, and every failure comes back as an
    ERROR string instead of an exception."""
    servers = named_servers(cfg)
    spec = servers.get((server_name or "").strip())
    if spec is None:
        return (f"ERROR: MCP server '{server_name}' is not configured "
                "(check [mcp_servers.<name>] and enable_mcp).")
    name = server_name.strip()
    tools = {t["name"]: t for t in _named_tools(cfg, name, spec)}
    tname = (tool_name or "").strip()
    if not tname:
        return (f"MCP tools on '{name}': "
                + (", ".join(tools) if tools else "(none listed)"))
    if tname not in tools:
        return (f"ERROR: unknown MCP tool '{name}.{tname}'. Available: "
                + (", ".join(tools) if tools else "(none listed)"))
    args = arguments if isinstance(arguments, dict) else {}
    bad = validate_args(json.dumps(tools[tname].get("input_schema") or {}),
                        args)
    if bad:
        return bad
    params = {"name": tname, "arguments": args}
    result: dict = {}
    t = max(1, int(timeout))
    for attempt in (1, 2):
        try:
            s = _named_session(name, spec, t)
            result = s.request("tools/call", params)
            break
        except Exception as e:
            _drop_named_session(name)
            if attempt == 2:
                return f"ERROR: MCP call failed: {str(e)[:400]}"
    bits = []
    for item in result.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "text":
            bits.append(str(item.get("text") or ""))
        elif isinstance(item, dict):
            bits.append(json.dumps(item, ensure_ascii=True)[:2000])
    text = "\n".join(b for b in bits if b).strip()
    tokens = max(200, int(getattr(cfg, "max_read_tokens", 1200) or 1200))
    cap = int(tokens * 3.5)      # chars per token, as in the other tool caps
    if len(text) > cap:
        text = text[:cap] + (f"\n[MCP result capped at about {tokens} tokens;"
                             f" {len(text) - cap} more characters dropped]")
    if result.get("isError"):
        return "ERROR: MCP tool reported failure:\n" + (text or "(no detail)")
    return text or "(MCP tool returned no text content)"
