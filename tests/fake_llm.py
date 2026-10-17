"""Minimal OpenAI-compatible server for testing pedacito end to end.

Behavior is driven by env vars:
  FAKE_ACTIONS  - JSON list of gather-loop actions (default: read file then done)
  FAKE_REPLY    - the answer-phase reply text
"""
import json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ACTIONS = json.loads(os.environ.get("FAKE_ACTIONS", "[]"))
REPLY = os.environ.get("FAKE_REPLY", "")
_state = {"i": 0}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._send({"object": "list", "data": [{"id": "fake-model"}]})
        else:
            self.send_response(404)
            self.end_headers()

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        rf = req.get("response_format") or {}
        msgs = req.get("messages", [])
        last = msgs[-1]["content"] if msgs else ""
        DONE = {"action": "done", "file": "", "pattern": "", "symbol": "",
                "start": 0, "end": 0, "content": "", "reasoning": "done"}
        if rf.get("type") in ("json_schema", "json_object"):
            i = _state["i"]
            _state["i"] += 1
            content = json.dumps(ACTIONS[i]) if i < len(ACTIONS) else json.dumps(DONE)
            self._send({"choices": [{"message": {"content": content},
                                     "finish_reason": "stop"}], "usage": {}})
            return
        if "single word: ok" in last:
            self._send({"choices": [{"message": {"content": "ok"},
                                     "finish_reason": "stop"}], "usage": {}})
            return
        self._send({"choices": [{"message": {"content": REPLY},
                                 "finish_reason": "stop"}], "usage": {}})


port = int(os.environ.get("FAKE_PORT", "8931"))
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
