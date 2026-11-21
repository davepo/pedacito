"""Bug Hunt #11 crucible: the optional LSP bridge against a mock server.

Pins the coordinate math (1-based harness <-> 0-based protocol), the push
diagnostics path, clean timeout + subprocess reaping when a language server
hangs, the server-initiated-request ack, and the opt-in / root-boundary gates.
The mock is a pure-Python stdio JSON-RPC script written into tmp_path; no real
language server is needed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import lsp, safety, tools
from pedacito.config import Config
from pedacito.index import Index

# Modes: normal | hang | server-request (asks workspace/configuration first).
MOCK_LSP = '''
import json, sys, time
MODE = sys.argv[1] if len(sys.argv) > 1 else "normal"
def read():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\\r\\n", b"\\n"):
            break
        k, v = line.decode("ascii", "replace").split(":", 1)
        headers[k.strip().lower()] = v.strip()
    n = int(headers.get("content-length", "0"))
    body = sys.stdin.buffer.read(n) if n > 0 else b""
    return json.loads(body.decode()) if body else None
def send(obj):
    body = json.dumps(obj).encode()
    sys.stdout.buffer.write(b"Content-Length: " + str(len(body)).encode()
                            + b"\\r\\n\\r\\n" + body)
    sys.stdout.buffer.flush()
while True:
    msg = read()
    if msg is None:
        break
    method, rid = msg.get("method"), msg.get("id")
    if method == "exit":
        break
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": rid, "result": {"capabilities": {
            "definitionProvider": True, "referencesProvider": True,
            "hoverProvider": True, "textDocumentSync": 1}}})
        continue
    if method in ("shutdown", "initialized"):
        if rid is not None:
            send({"jsonrpc": "2.0", "id": rid, "result": None})
        continue
    if method == "textDocument/didOpen":
        uri = msg["params"]["textDocument"]["uri"]
        send({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
              "params": {"uri": uri, "diagnostics": [{
                  "range": {"start": {"line": 9, "character": 4},
                            "end": {"line": 9, "character": 12}},
                  "severity": 1, "message": "crucible diagnostic"}]}})
        continue
    if MODE == "hang" and method.startswith("textDocument/"):
        time.sleep(120)
        continue
    if MODE == "server-request" and method == "textDocument/hover":
        send({"jsonrpc": "2.0", "id": 9001,
              "method": "workspace/configuration", "params": {"items": []}})
    params = msg.get("params") or {}
    pos = params.get("position") or {}
    uri = (params.get("textDocument") or {}).get("uri", "")
    if method == "textDocument/definition":
        send({"jsonrpc": "2.0", "id": rid, "result": [{
            "uri": uri, "range": {"start": {"line": 14, "character": 3},
                                  "end": {"line": 14, "character": 18}}}]})
    elif method == "textDocument/references":
        send({"jsonrpc": "2.0", "id": rid, "result": [
            {"uri": uri, "range": {"start": {"line": 1, "character": 0},
                                   "end": {"line": 1, "character": 6}}},
            {"uri": uri, "range": {"start": {"line": 19, "character": 4},
                                   "end": {"line": 19, "character": 10}}}]})
    elif method == "textDocument/hover":
        send({"jsonrpc": "2.0", "id": rid, "result": {"contents": {
            "kind": "markdown",
            "value": "pos=(%s,%s)" % (pos.get("line"), pos.get("character"))}}})
    elif rid is not None:
        send({"jsonrpc": "2.0", "id": rid, "result": None})
'''


def _setup(tmp_path, mode="normal", timeout=5):
    root = tmp_path / "proj"
    root.mkdir(exist_ok=True)
    (root / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    script = tmp_path / "mock_lsp.py"
    script.write_text(MOCK_LSP, encoding="utf-8")
    cfg = Config()
    cfg.allow_lsp = True
    cfg.lsp_timeout = timeout
    cfg.sources = {}
    cfg.lsp_servers = {".py": f'"{sys.executable}" "{script}" {mode}'}
    lsp.close_all()
    return root, cfg


def test_coordinate_math_both_directions(tmp_path):
    root, cfg = _setup(tmp_path)
    try:
        # Hover echoes the position the client SENT: harness line 3 col 7
        # must arrive 0-based as (2, 6).
        h = lsp.result(root, cfg, "hover", "a.py", 3, 7)
        assert "pos=(2,6)" in h
        # Returned 0-based locations print 1-based: line 14 char 3 -> 15:4.
        d = lsp.result(root, cfg, "definition", "a.py", 12, 5)
        assert d.startswith("Definition:") and "a.py:15:4" in d
        r = lsp.result(root, cfg, "references", "a.py", 12, 5)
        assert "a.py:2:1" in r and "a.py:20:5" in r
        # Push diagnostics after didOpen: 0-based 9:4 -> 10:5.
        g = lsp.result(root, cfg, "diagnostics", "a.py", 1, 1)
        assert "a.py:10:5: error: crucible diagnostic" in g
    finally:
        lsp.close_all()


def test_hanging_server_times_out_and_is_reaped(tmp_path):
    root, cfg = _setup(tmp_path, mode="hang", timeout=1)
    try:
        t0 = time.monotonic()
        out = lsp.result(root, cfg, "definition", "a.py", 2, 1)
        dt = time.monotonic() - t0
        assert out.startswith("ERROR") and "timed out" in out
        assert dt < 8                          # bounded by lsp_timeout
        clients = list(lsp._CLIENTS.values())
        assert clients
        procs = [c.proc for c in clients]
    finally:
        lsp.close_all()
    time.sleep(0.2)
    assert all(p.poll() is not None for p in procs)


def test_server_initiated_request_is_acked_not_blocking(tmp_path):
    root, cfg = _setup(tmp_path, mode="server-request")
    try:
        h = lsp.result(root, cfg, "hover", "a.py", 2, 3)
        assert "pos=(1,2)" in h
    finally:
        lsp.close_all()


def test_lsp_gates_disabled_outside_root_and_unknown_op(tmp_path):
    root, cfg = _setup(tmp_path)
    idx = Index(root, [])
    try:
        off = Config()
        assert tools.tool_lsp(idx, off, "definition", "a.py")\
            .startswith("ERROR")
        assert tools.tool_lsp(idx, cfg, "definition", "../outside.py")\
            .startswith("ERROR")
        assert lsp.result(root, cfg, "teleport", "a.py").startswith("ERROR")
        # The tool dispatches and its output is fenced as external data.
        out = tools.dispatch(idx, cfg, {"action": "lsp",
                                        "symbol": "definition",
                                        "file": "a.py", "start": 2, "end": 3})
        assert "a.py:15:4" in out
        framed = safety.frame("lsp", out, cfg)
        assert framed.startswith("[BEGIN UNTRUSTED LSP OUTPUT")
    finally:
        lsp.close_all()


def test_no_server_configured_is_a_clear_error(tmp_path):
    root, cfg = _setup(tmp_path)
    cfg.lsp_servers = {}
    try:
        out = lsp.result(root, cfg, "definition", "a.py")
        assert out.startswith("ERROR") and "no language server" in out
    finally:
        lsp.close_all()


def test_read_message_caps_content_length_at_16mb():
    """Oversized Content-Length frames must be rejected without allocating."""
    import io

    oversized = io.BytesIO(
        f"Content-Length: {lsp._MAX_CONTENT_LENGTH + 1}\r\n\r\n{{}}".encode("ascii"))
    assert lsp.LSPClient._read_message(oversized) is None


def test_lsp_didopen_once_then_didchange_with_incrementing_versions(tmp_path):
    """Repeated open() calls on the same file send didOpen (v1) once, then
    didChange (v2, v3, ...), and only the latest diagnostics per URI are kept."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n", encoding="utf-8")
    log_path = tmp_path / "seen.jsonl"
    server = tmp_path / "track_lsp.py"
    server.write_text(
        "import json, sys\n"
        f"LOG = {str(log_path)!r}\n"
        "def read():\n"
        "    headers = {}\n"
        "    while True:\n"
        "        line = sys.stdin.buffer.readline()\n"
        "        if not line:\n"
        "            return None\n"
        "        if line in (b'\\r\\n', b'\\n'):\n"
        "            break\n"
        "        k, v = line.decode('ascii', 'replace').split(':', 1)\n"
        "        headers[k.strip().lower()] = v.strip()\n"
        "    n = int(headers.get('content-length', '0'))\n"
        "    return json.loads(sys.stdin.buffer.read(n).decode()) if n > 0 else None\n"
        "def send(obj):\n"
        "    body = json.dumps(obj).encode()\n"
        "    sys.stdout.buffer.write(b'Content-Length: ' + str(len(body)).encode() + b'\\r\\n\\r\\n' + body)\n"
        "    sys.stdout.buffer.flush()\n"
        "while True:\n"
        "    msg = read()\n"
        "    if msg is None or msg.get('method') == 'exit':\n"
        "        break\n"
        "    m, rid = msg.get('method'), msg.get('id')\n"
        "    if m in ('textDocument/didOpen', 'textDocument/didChange'):\n"
        "        td = msg['params']['textDocument']\n"
        "        with open(LOG, 'a', encoding='utf-8') as fh:\n"
        "            fh.write(json.dumps({'method': m, 'version': td['version']}) + '\\n')\n"
        "        send({'jsonrpc': '2.0', 'method': 'textDocument/publishDiagnostics',\n"
        "              'params': {'uri': td['uri'], 'diagnostics': [{\n"
        "                  'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}},\n"
        "                  'severity': 1, 'message': f'v{td[\"version\"]}-diag'}]}})\n"
        "    elif m == 'initialize':\n"
        "        send({'jsonrpc': '2.0', 'id': rid, 'result': {'capabilities': {}}})\n"
        "    elif m == 'textDocument/hover':\n"
        "        send({'jsonrpc': '2.0', 'id': rid, 'result': {'contents': 'ok'}})\n"
        "    elif rid is not None:\n"
        "        send({'jsonrpc': '2.0', 'id': rid, 'result': None})\n",
        encoding="utf-8")
    cfg = Config()
    cfg.allow_lsp = True
    cfg.lsp_timeout = 5
    cfg.sources = {}
    cfg.lsp_servers = {".py": f'"{sys.executable}" "{server}"'}
    lsp.close_all()
    try:
        assert "ok" in lsp.result(root, cfg, "hover", "a.py", 1, 1)
        (root / "a.py").write_text("x = 2\n", encoding="utf-8")
        assert "ok" in lsp.result(root, cfg, "hover", "a.py", 1, 1)
        diag = lsp.result(root, cfg, "diagnostics", "a.py", 1, 1)
        assert "v3-diag" in diag or "v2-diag" in diag
        import json as _json
        events = [_json.loads(ln) for ln in
                  log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert events == [
            {"method": "textDocument/didOpen", "version": 1},
            {"method": "textDocument/didChange", "version": 2},
            {"method": "textDocument/didChange", "version": 3},
        ]
        # Only latest diagnostics stored per URI, and _notes stays bounded.
        client = next(iter(lsp._CLIENTS.values()))
        assert len(client._diagnostics) == 1
        for i in range(lsp._MAX_NOTES + 50):
            client._record_note({"method": "window/logMessage", "n": i})
        assert len(client._notes) <= lsp._MAX_NOTES
    finally:
        lsp.close_all()


def test_lsp_fails_fast_on_server_eof(tmp_path):
    """A language server that closes stdout mid-request must fail fast via the
    EOF sentinel instead of hanging for lsp_timeout."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n", encoding="utf-8")
    server = tmp_path / "crash_lsp.py"
    server.write_text(
        "import json, sys\n"
        "def read():\n"
        "    headers = {}\n"
        "    while True:\n"
        "        line = sys.stdin.buffer.readline()\n"
        "        if not line:\n"
        "            return None\n"
        "        if line in (b'\\r\\n', b'\\n'):\n"
        "            break\n"
        "        k, v = line.decode('ascii', 'replace').split(':', 1)\n"
        "        headers[k.strip().lower()] = v.strip()\n"
        "    n = int(headers.get('content-length', '0'))\n"
        "    return json.loads(sys.stdin.buffer.read(n).decode()) if n > 0 else None\n"
        "def send(obj):\n"
        "    body = json.dumps(obj).encode()\n"
        "    sys.stdout.buffer.write(b'Content-Length: ' + str(len(body)).encode() + b'\\r\\n\\r\\n' + body)\n"
        "    sys.stdout.buffer.flush()\n"
        "while True:\n"
        "    msg = read()\n"
        "    if msg is None:\n"
        "        break\n"
        "    if msg.get('method') == 'initialize':\n"
        "        send({'jsonrpc': '2.0', 'id': msg['id'], 'result': {'capabilities': {}}})\n"
        "    elif msg.get('method') == 'textDocument/hover':\n"
        "        sys.exit(1)\n",
        encoding="utf-8")
    cfg = Config()
    cfg.allow_lsp = True
    cfg.lsp_timeout = 15
    cfg.sources = {}
    cfg.lsp_servers = {".py": f'"{sys.executable}" "{server}"'}
    lsp.close_all()
    try:
        t0 = time.monotonic()
        out = lsp.result(root, cfg, "hover", "a.py", 1, 1)
        assert out.startswith("ERROR") and "closed stdout" in out
        assert time.monotonic() - t0 < 5.0
    finally:
        lsp.close_all()