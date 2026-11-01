"""Tiny stdio LSP server used by tests (JSON-RPC with Content-Length frames)."""
import json
import sys


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        key, val = line.decode("ascii", "replace").split(":", 1)
        headers[key.strip().lower()] = val.strip()
    n = int(headers.get("content-length", "0"))
    return json.loads(sys.stdin.buffer.read(n).decode("utf-8"))


def send(obj):
    body = json.dumps(obj).encode("utf-8")
    sys.stdout.buffer.write(
        f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
    sys.stdout.buffer.flush()


while True:
    msg = read_message()
    if msg is None:
        break
    method = msg.get("method")
    rid = msg.get("id")
    if method == "exit":
        break
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": rid, "result": {
            "capabilities": {"definitionProvider": True,
                             "referencesProvider": True,
                             "hoverProvider": True}}})
    elif method == "shutdown":
        send({"jsonrpc": "2.0", "id": rid, "result": None})
    elif method == "textDocument/didOpen":
        uri = (msg.get("params") or {}).get("textDocument", {}).get("uri", "")
        send({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
              "params": {"uri": uri, "diagnostics": [{
                  "range": {"start": {"line": 0, "character": 0},
                            "end": {"line": 0, "character": 1}},
                  "severity": 2,
                  "message": "fake diagnostic",
              }]}})
    elif method == "textDocument/definition":
        uri = (msg.get("params") or {}).get("textDocument", {}).get("uri", "")
        send({"jsonrpc": "2.0", "id": rid, "result": [{
            "uri": uri,
            "range": {"start": {"line": 4, "character": 2},
                      "end": {"line": 4, "character": 8}},
        }]})
    elif method == "textDocument/references":
        uri = (msg.get("params") or {}).get("textDocument", {}).get("uri", "")
        send({"jsonrpc": "2.0", "id": rid, "result": [
            {"uri": uri, "range": {"start": {"line": 1, "character": 0},
                                   "end": {"line": 1, "character": 3}}},
            {"uri": uri, "range": {"start": {"line": 7, "character": 4},
                                   "end": {"line": 7, "character": 7}}},
        ]})
    elif method == "textDocument/hover":
        send({"jsonrpc": "2.0", "id": rid, "result": {
            "contents": {"kind": "markdown", "value": "**fake hover**"}}})
    elif rid is not None:
        send({"jsonrpc": "2.0", "id": rid, "result": None})
