"""Tiny MCP server used by tests: newline-delimited JSON-RPC on stdio."""
import json
import sys

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except ValueError:
        continue
    method = msg.get("method")
    rid = msg.get("id")
    if rid is None:
        continue                       # notification
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {},
                  "serverInfo": {"name": "fake", "version": "0"}}
    elif method == "tools/list":
        result = {"tools": [{
            "name": "echo",
            "description": "echo the arguments back",
            "inputSchema": {"type": "object"},
        }]}
    elif method == "tools/call":
        args = (msg.get("params") or {}).get("arguments") or {}
        result = {"content": [{"type": "text",
                               "text": "echo:" + json.dumps(args,
                                                            sort_keys=True)}]}
    elif method == "ping":
        result = {}
    else:
        result = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": rid,
                                 "result": result}) + "\n")
    sys.stdout.flush()
