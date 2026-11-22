"""Native Anthropic /v1/messages transport (WP-12).

Unit tests for the pure translation layer plus live loopback fake-server tests
for buffered and SSE-streaming responses, outbound secret scanning, and the
``anthropic_native = false`` fallback to the OpenAI-compatible path.

The loopback fakes are built on ``pedacito.local_server.LocalHTTPServer`` (the
DNS-free class required by ``test_macos_portability``), never the stdlib
``HTTPServer``.
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import config as config_mod
from pedacito import security as security_mod
from pedacito import transport_anthropic as ta
from pedacito.local_server import LocalHTTPServer  # noqa: E402
from pedacito.llm import LLMClient, LLMError


GENERIC_SECRET = "AbCdEf0123456789XYZ"

# ------------------------------------------------------------------ unit layer


def _cfg(**kw):
    cfg = config_mod.Config()
    cfg.base_url = "https://api.anthropic.com/v1"
    cfg.model = "claude-fake"
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _cc_count(system_blocks, conversation) -> int:
    total = sum(1 for b in system_blocks if "cache_control" in b)
    for m in conversation:
        for b in m["content"]:
            if "cache_control" in b:
                total += 1
    return total


def test_is_anthropic_native():
    assert ta.is_anthropic_native(_cfg())
    assert ta.is_anthropic_native(
        _cfg(base_url="http://127.0.0.1:1234/v1", provider="anthropic"))
    assert not ta.is_anthropic_native(_cfg(anthropic_native=False))
    assert not ta.is_anthropic_native(
        _cfg(base_url="https://api.openai.com/v1"))
    assert not ta.is_anthropic_native(
        _cfg(base_url="http://127.0.0.1:1234/v1"))  # loopback, no provider


def test_messages_endpoint_maps_all_spellings():
    assert ta.messages_endpoint("https://api.anthropic.com") \
        == "https://api.anthropic.com/v1/messages"
    assert ta.messages_endpoint("https://api.anthropic.com/v1") \
        == "https://api.anthropic.com/v1/messages"
    assert ta.messages_endpoint("https://api.anthropic.com/v1/messages") \
        == "https://api.anthropic.com/v1/messages"
    assert ta.messages_endpoint("http://127.0.0.1:9999/v1") \
        == "http://127.0.0.1:9999/v1/messages"


def test_build_headers_adds_beta_only_for_one_hour_ttl():
    h = ta.build_headers("k", ttl="5m")
    assert h["x-api-key"] == "k"
    assert h["anthropic-version"] == "2023-06-01"
    assert h["content-type"] == "application/json"
    assert "anthropic-beta" not in h
    h1 = ta.build_headers("k", ttl="1h")
    assert h1["anthropic-beta"] == "extended-cache-ttl-2025-04-11"


def test_translate_extracts_system_and_tags_last_block():
    system, conv = ta.translate_messages([
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
    ])
    assert [b["text"] for b in system] == ["SYS"]
    assert system[-1]["cache_control"] == {"type": "ephemeral"}
    assert all(m["role"] != "system" for m in conv)
    assert conv[0]["content"] == [
        {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]


def test_cache_control_ttl_5m_vs_1h():
    system, conv = ta.translate_messages(
        [{"role": "system", "content": "SYS"},
         {"role": "user", "content": "hi"}],
        cache_ttl="5m")
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    system1, conv1 = ta.translate_messages(
        [{"role": "system", "content": "SYS"},
         {"role": "user", "content": "hi"}],
        cache_ttl="1h")
    assert system1[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert conv1[0]["content"][0]["cache_control"] == {
        "type": "ephemeral", "ttl": "1h"}


def test_four_breakpoint_cap_and_penultimate_user_turn():
    msgs = [{"role": "system", "content": "SYS"}]
    for i in range(5):
        msgs.append({"role": "user", "content": f"U{i}"})
        msgs.append({"role": "assistant", "content": f"A{i}"})
    system, conv = ta.translate_messages(msgs, cache_enabled=True)
    assert _cc_count(system, conv) == 3          # system + user0 + penultimate
    assert _cc_count(system, conv) <= 4
    # the LAST user turn is left uncached (so it can vary each call)
    last_user = [m for m in conv if m["role"] == "user"][-1]
    assert "cache_control" not in last_user["content"][-1]


def test_two_turns_have_no_penultimate_breakpoint():
    system, conv = ta.translate_messages(
        [{"role": "system", "content": "S"},
         {"role": "user", "content": "U0"},
         {"role": "assistant", "content": "A0"},
         {"role": "user", "content": "U1"}])
    assert _cc_count(system, conv) == 2          # system + user0 only


def test_cache_disabled_means_no_markers():
    system, conv = ta.translate_messages(
        [{"role": "system", "content": "S"},
         {"role": "user", "content": "U0"}],
        cache_enabled=False)
    assert _cc_count(system, conv) == 0


def test_image_url_becomes_native_image_block():
    system, conv = ta.translate_messages([
        {"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64,AA=="}},
        ]}], cache_enabled=False)
    blocks = conv[0]["content"]
    assert blocks[0] == {"type": "text", "text": "look"}
    assert blocks[1] == {"type": "image", "source": {
        "type": "base64", "media_type": "image/png", "data": "AA=="}}


def test_tool_calls_and_tool_results_are_translated():
    system, conv = ta.translate_messages([
        {"role": "user", "content": "run the step"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "step",
                          "arguments": '{"action": "read", "file": "a.py"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "RESULT:\nline"},
    ], cache_enabled=False)
    user, assistant, tool = conv
    assert assistant["role"] == "assistant"
    assert assistant["content"] == [{"type": "tool_use", "id": "call_1",
                                     "name": "step",
                                     "input": {"action": "read", "file": "a.py"}}]
    assert tool["role"] == "user"               # tool -> user, tool_result
    assert tool["content"] == [{"type": "tool_result", "tool_use_id": "call_1",
                                "content": "RESULT:\nline"}]


def test_translate_tools_and_forced_choice():
    tools, choice = ta.translate_tools(
        [{"type": "function", "function": {
            "name": "step", "description": "Do it",
            "parameters": {"type": "object", "properties": {}}}}],
        {"type": "function", "function": {"name": "step"}})
    assert tools == [{"name": "step", "description": "Do it",
                      "input_schema": {"type": "object", "properties": {}}}]
    assert choice == {"type": "tool", "name": "step"}
    # forced tool_choice is downgraded to auto when thinking is on
    _, choice_think = ta.translate_tools(
        [{"type": "function", "function": {"name": "step", "parameters": {}}}],
        {"type": "function", "function": {"name": "step"}}, thinking_on=True)
    assert choice_think == {"type": "auto"}
    assert ta.translate_tools(None) == (None, None)


def test_build_payload_thinking_budget_and_temperature_omission():
    cfg = _cfg(reasoning_effort="high")
    body = ta.build_payload(cfg, [{"role": "user", "content": "hi"}],
                            max_tokens=100, temperature=0.5)
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 12288}
    assert "temperature" not in body
    assert body["max_tokens"] > 12288              # budget < max_tokens

    low = ta.build_payload(_cfg(reasoning_effort="low"),
                           [{"role": "user", "content": "hi"}], 500)
    assert low["thinking"] == {"type": "enabled", "budget_tokens": 2048}

    off = ta.build_payload(_cfg(reasoning_effort="high", disable_thinking=True),
                           [{"role": "user", "content": "hi"}], 100,
                           temperature=0.7)
    assert "thinking" not in off
    assert off["temperature"] == 0.7              # temperature returns


def test_parse_json_response_normalises_usage_and_tool_calls():
    data = {
        "content": [
            {"type": "text", "text": "answer"},
            {"type": "thinking", "thinking": "reasoning"},
            {"type": "tool_use", "id": "tu_1", "name": "step",
             "input": {"action": "done"}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 100, "output_tokens": 20,
                  "cache_read_input_tokens": 30,
                  "cache_creation_input_tokens": 15},
    }
    got = ta.parse_json_response(data)
    assert got["text"] == "answer"
    assert got["reasoning"] == "reasoning"
    assert got["tool_calls"] == [{"id": "tu_1", "name": "step",
                                  "arguments_json": '{"action": "done"}'}]
    assert got["finish_reason"] == "tool_calls"
    assert got["usage"] == {"prompt_tokens": 145, "completion_tokens": 20,
                            "cached_tokens": 30, "cache_write_tokens": 15}


def test_parse_sse_stream_handles_all_delta_types():
    lines = [
        "event: message_start",
        'data: {"type":"message_start","message":{"usage":'
        '{"input_tokens":100,"cache_read_input_tokens":30,'
        '"cache_creation_input_tokens":15}}}',
        "event: content_block_start",
        'data: {"type":"content_block_start","index":0,"content_block":'
        '{"type":"text","text":""}}',
        'data: {"type":"content_block_delta","index":0,"delta":'
        '{"type":"text_delta","text":"Hel"}}',
        'data: {"type":"content_block_delta","index":0,"delta":'
        '{"type":"text_delta","text":"lo"}}',
        "event: content_block_start",
        'data: {"type":"content_block_start","index":1,"content_block":'
        '{"type":"thinking","thinking":""}}',
        'data: {"type":"content_block_delta","index":1,"delta":'
        '{"type":"thinking_delta","thinking":"reasoning..."}}',
        "event: content_block_start",
        'data: {"type":"content_block_start","index":2,"content_block":'
        '{"type":"tool_use","id":"tu_1","name":"step","input":{}}}',
        'data: {"type":"content_block_delta","index":2,"delta":'
        '{"type":"input_json_delta","partial_json":"{\\"action\\": \\"d"}}',
        'data: {"type":"content_block_delta","index":2,"delta":'
        '{"type":"input_json_delta","partial_json":"one\\"}"}}',
        "event: message_delta",
        'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},'
        '"usage":{"output_tokens":20}}',
        "event: message_stop",
        'data: {"type":"message_stop"}',
    ]

    class Resp:
        def __init__(self, lines):
            self._lines = lines

        def iter_lines(self, decode_unicode=False):
            yield from self._lines

    seen_text, seen_reasoning = [], []
    got = ta.parse_sse_stream(
        Resp(lines), on_text=seen_text.append, on_reasoning=seen_reasoning.append)
    assert got["text"] == "Hello"
    assert got["reasoning"] == "reasoning..."
    assert got["tool_calls"] == [{"id": "tu_1", "name": "step",
                                  "arguments_json": '{"action": "done"}'}]
    assert got["finish_reason"] == "tool_calls"
    assert got["usage"] == {"prompt_tokens": 145, "completion_tokens": 20,
                            "cached_tokens": 30, "cache_write_tokens": 15}
    assert seen_text == ["Hel", "lo"]
    assert seen_reasoning == ["reasoning..."]


# ------------------------------------------------------------ live fake server

class _AnthropicHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep pytest output clean
        pass

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _event(self, name, data):
        self.wfile.write(f"event: {name}\n".encode())
        self.wfile.write(("data: " + json.dumps(data) + "\n\n").encode())

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._send_json({"data": [{"id": "claude-fake"}]})
        else:
            self._send_json({"error": {"type": "not_found_error"}}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        self.server.last_request = req
        self.server.last_headers = dict(self.headers)
        if self.server.stream:
            self._stream()
        else:
            self._buffered()

    def _buffered(self):
        content = []
        if self.server.text is not None:
            content.append({"type": "text", "text": self.server.text})
        if self.server.thinking is not None:
            content.append({"type": "thinking",
                            "thinking": self.server.thinking})
        if self.server.tool is not None:
            content.append({"type": "tool_use", "id": self.server.tool["id"],
                            "name": self.server.tool["name"],
                            "input": self.server.tool["input"]})
        self._send_json({
            "id": "msg_1", "type": "message", "role": "assistant",
            "content": content, "stop_reason": self.server.stop_reason,
            "usage": dict(self.server.usage),
        })

    @staticmethod
    def _chunks(text, size=3):
        return [text[i:i + size] for i in range(0, len(text), size)]

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        u = self.server.usage
        self._event("message_start", {"type": "message_start", "message": {
            "usage": {"input_tokens": u.get("input_tokens", 0),
                      "output_tokens": 0,
                      "cache_read_input_tokens": u.get(
                          "cache_read_input_tokens", 0),
                      "cache_creation_input_tokens": u.get(
                          "cache_creation_input_tokens", 0)}}})
        idx = 0
        if self.server.text is not None:
            self._event("content_block_start",
                        {"type": "content_block_start", "index": idx,
                         "content_block": {"type": "text", "text": ""}})
            for ch in self._chunks(self.server.text):
                self._event("content_block_delta",
                            {"type": "content_block_delta", "index": idx,
                             "delta": {"type": "text_delta", "text": ch}})
            idx += 1
        if self.server.thinking is not None:
            self._event("content_block_start",
                        {"type": "content_block_start", "index": idx,
                         "content_block": {"type": "thinking",
                                           "thinking": ""}})
            for ch in self._chunks(self.server.thinking):
                self._event("content_block_delta",
                            {"type": "content_block_delta", "index": idx,
                             "delta": {"type": "thinking_delta",
                                       "thinking": ch}})
            idx += 1
        if self.server.tool is not None:
            self._event("content_block_start",
                        {"type": "content_block_start", "index": idx,
                         "content_block": {"type": "tool_use",
                                           "id": self.server.tool["id"],
                                           "name": self.server.tool["name"],
                                           "input": {}}})
            raw = json.dumps(self.server.tool["input"])
            for ch in self._chunks(raw):
                self._event("content_block_delta",
                            {"type": "content_block_delta", "index": idx,
                             "delta": {"type": "input_json_delta",
                                       "partial_json": ch}})
            idx += 1
        self._event("message_delta",
                    {"type": "message_delta",
                     "delta": {"stop_reason": self.server.stop_reason},
                     "usage": {"output_tokens": u.get("output_tokens", 0)}})
        self._event("message_stop", {"type": "message_stop"})
        self.wfile.flush()


USAGE = {"input_tokens": 100, "output_tokens": 20,
         "cache_read_input_tokens": 30, "cache_creation_input_tokens": 15}


@pytest.fixture
def anthropic_server():
    srv = LocalHTTPServer(("127.0.0.1", 0), _AnthropicHandler)
    srv.stream = False
    srv.text = "Hello, world"
    srv.thinking = None
    srv.tool = None
    srv.stop_reason = "end_turn"
    srv.usage = dict(USAGE)
    srv.last_request = None
    srv.last_headers = None
    thread = threading.Thread(target=srv.serve_forever, daemon=True,
                              name="anthropic-fake")
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(5)


def _client(srv, **kw):
    cfg = config_mod.Config()
    cfg.base_url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    cfg.model = "claude-fake"
    cfg.provider = "anthropic"
    cfg.api_key = "sk-ant-test"
    cfg.api_key_env = ""
    cfg.prompt_cache = "explicit"       # so the native cache markers engage
    for k, v in kw.items():
        setattr(cfg, k, v)
    return LLMClient(cfg)


def test_live_non_streaming_records_cache_tokens(anthropic_server):
    client = _client(anthropic_server)
    assert client.chat([{"role": "system", "content": "SYS"},
                        {"role": "user", "content": "hi"}], 100) == "Hello, world"
    assert client.stats["cached_tokens"] == 30
    assert client.stats["cache_write_tokens"] == 15
    assert client.stats["prompt_tokens"] == 145
    assert client.stats["completion_tokens"] == 20
    # the request that left was a native /v1/messages body, not OpenAI shape
    req = anthropic_server.last_request
    assert req["system"] == [{"type": "text", "text": "SYS",
                              "cache_control": {"type": "ephemeral"}}]
    assert req["messages"][0]["role"] == "user"
    assert req["messages"][0]["content"][0]["type"] == "text"
    assert anthropic_server.last_headers["x-api-key"] == "sk-ant-test"


def test_live_streaming_records_cache_and_reasoning(anthropic_server):
    anthropic_server.stream = True
    anthropic_server.text = "Hello, world"
    anthropic_server.thinking = "Let me think about it."
    client = _client(anthropic_server)
    deltas = []
    client.stream_callback = deltas.append
    text = client.chat([{"role": "user", "content": "hi"}], 100, stream=True)
    assert text == "Hello, world"
    assert "".join(deltas) == "Hello, world"
    assert client.last_stream_reasoning == "Let me think about it."
    assert client.stats["cached_tokens"] == 30
    assert client.stats["cache_write_tokens"] == 15
    assert client.stats["completion_tokens"] == 20


def test_live_native_tool_call_via_buffered_json(anthropic_server):
    anthropic_server.text = None
    anthropic_server.thinking = None
    anthropic_server.tool = {"id": "tu_1", "name": "step",
                             "input": {"action": "done", "reasoning": "r"}}
    anthropic_server.stop_reason = "tool_use"
    client = _client(anthropic_server)
    from pedacito.tools import schema_for
    out = client.chat_json([{"role": "user", "content": "go"}],
                           schema_for(client.cfg), 200)
    assert out["action"] == "done"
    req = anthropic_server.last_request
    assert req["tools"][0]["name"] == "step"
    assert req["tool_choice"] == {"type": "tool", "name": "step"}


def test_outbound_secret_scanner_fires_before_anthropic_request(monkeypatch):
    """A secret in an OpenAI-shaped message is redacted before translation to a
    native /v1/messages body leaves (external provider + redact mode)."""
    class Resp:
        def __init__(self, body):
            self._body = body
            self.status_code = 200
            self.headers = {}
            self.encoding = "utf-8"
            self.text = json.dumps(body)

        def json(self):
            return self._body

        def close(self):
            pass

    captured = {}

    def fake_post(url, **kw):
        captured["url"] = url
        captured["json"] = kw["json"]
        return Resp({"content": [{"type": "text", "text": "ok"}],
                     "stop_reason": "end_turn",
                     "usage": dict(USAGE)})

    monkeypatch.setattr("pedacito.llm.requests.post", fake_post)
    cfg = config_mod.Config()
    cfg.base_url = "https://api.anthropic.com/v1"
    cfg.model = "claude-fake"
    cfg.api_key = "sk-ant-test"
    cfg.security_outbound_mode = "redact"
    client = LLMClient(cfg)
    text = f'api_key = "{GENERIC_SECRET}"'
    assert client.chat([{"role": "user", "content": text}], 10) == "ok"
    sent = json.dumps(captured["json"])
    assert GENERIC_SECRET not in sent
    assert security_mod.REDACTED_SECRET in sent
    assert captured["url"].endswith("/v1/messages")


def test_anthropic_native_false_falls_back_to_openai_path(monkeypatch):
    class Resp:
        def __init__(self, body):
            self._body = body
            self.status_code = 200
            self.headers = {}
            self.encoding = "utf-8"
            self.text = json.dumps(body)

        def json(self):
            return self._body

        def close(self):
            pass

    captured = {}

    def fake_post(url, **kw):
        captured["url"] = url
        captured["json"] = kw["json"]
        return Resp({"choices": [{"message": {"content": "ok"},
                                  "finish_reason": "stop"}], "usage": {}})

    monkeypatch.setattr("pedacito.llm.requests.post", fake_post)
    cfg = config_mod.Config()
    cfg.base_url = "https://api.anthropic.com/v1"
    cfg.model = "claude-fake"
    cfg.api_key = "sk-ant-test"
    cfg.anthropic_native = False
    cfg.prompt_cache = "off"
    client = LLMClient(cfg)
    assert client.chat([{"role": "user", "content": "hi"}], 10) == "ok"
    assert captured["url"].endswith("/v1/chat/completions")
    body = captured["json"]
    assert "system" not in body
    assert body["messages"][0] == {"role": "user", "content": "hi"}


def test_auth_diagnostics_name_x_api_key_for_anthropic(monkeypatch):
    class ProbeResp:
        status_code = 401
        headers = {}
        text = '{"error": {"type": "authentication_error"}}'

        def raise_for_status(self):
            import requests
            raise requests.exceptions.HTTPError("401", response=self)

    monkeypatch.setattr("pedacito.llm.requests.get",
                        lambda *a, **k: ProbeResp())
    cfg = config_mod.Config()
    cfg.base_url = "https://api.anthropic.com/v1"
    cfg.model = "claude-fake"
    with pytest.raises(LLMError) as e:
        LLMClient(cfg).check()
    msg = str(e.value)
    assert "x-api-key" in msg
    assert "ANTHROPIC_API_KEY" in msg
