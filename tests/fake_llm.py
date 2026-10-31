"""Minimal OpenAI-compatible server for testing pedacito end to end.

Behavior is driven by env vars:
  FAKE_ACTIONS  - JSON list of gather-loop actions. With no list configured the
                  built-in default is "read FAKE_READ_FILE (lines 1-200), then
                  done" -- FAKE_READ_FILE defaults to old_name.py so the e2e
                  rename fixture works out of the box.
  FAKE_REPLY    - the answer-phase reply text
  FAKE_FINISH   - optional JSON list of finish_reason overrides applied to the
                  schema-gated responses in order (["length"] = truncated reply)
  FAKE_PORT     - listen port (default 8931)

FAKE_ACTIONS entries may carry "_finish": "length" to mark that single step as
truncated -- this keeps per-test scenarios independent of one shared counter.
The action cursor resets whenever a NON-schema request arrives, i.e. when the
agent moves from its gather loop to the answer phase, so every test's gather
loop starts from action 0 even though the module-scoped server is reused.
"""
import json, os, sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

# When run as a script, sys.path[0] is tests/, not the checkout root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito.local_server import LocalHTTPServer

ACTIONS = json.loads(os.environ.get("FAKE_ACTIONS", "[]"))
REPLY = os.environ.get("FAKE_REPLY", "")
FINISHES = json.loads(os.environ.get("FAKE_FINISH", "[]"))
_state = {"i": 0}

# Default gather script when FAKE_ACTIONS is not set: read one file, then
# done. The file name comes from FAKE_READ_FILE so each test can point the
# fake model at whatever fixture project it built (the module-scoped server
# is shared and cannot know the per-test tmp paths otherwise).
_DEFAULT_FILE = os.environ.get("FAKE_READ_FILE", "old_name.py")
DEFAULT_ACTION = {"action": "read", "file": _DEFAULT_FILE, "pattern": "",
                  "symbol": "", "start": 1, "end": 200, "content": "",
                  "reasoning": "read the file named in the task"}
DONE = {"action": "done", "file": "", "pattern": "", "symbol": "",
        "start": 0, "end": 0, "content": "", "reasoning": "done"}


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
        if rf.get("type") in ("json_schema", "json_object"):
            i = _state["i"]
            _state["i"] += 1
            if ACTIONS:
                action = ACTIONS[i] if i < len(ACTIONS) else DONE
            else:
                # No script configured: read the target file once, then done.
                action = DEFAULT_ACTION if i == 0 else DONE
            fin = action.get("_finish") or (
                FINISHES[i] if i < len(FINISHES) else "stop")
            content = {k: v for k, v in action.items() if not k.startswith("_")}
            text = json.dumps(content)
            if fin == "length":
                # Simulate a real cutoff: drop the last few characters so the
                # JSON is genuinely unterminated -- repair_truncated_json must
                # not be handed a complete object and asked to "salvage" it.
                text = text[:-5]
            self._send({"choices": [{"message": {"content": text},
                                     "finish_reason": fin}], "usage": {}})
            return
        # Non-schema request: the agent has left the gather loop. Reset the
        # cursor so the next test's gather starts at action 0.
        _state["i"] = 0
        if "single word: ok" in last:
            self._send({"choices": [{"message": {"content": "ok"},
                                     "finish_reason": "stop"}], "usage": {}})
            return
        self._send({"choices": [{"message": {"content": REPLY},
                                 "finish_reason": "stop"}], "usage": {}})


port = int(os.environ.get("FAKE_PORT", "8931"))
LocalHTTPServer(("127.0.0.1", port), H).serve_forever()
