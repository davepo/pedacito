"""Native Google Gemini /v1beta transport tests (offline + loopback)."""

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
from pedacito import transport_gemini as tg
from pedacito.llm import LLMClient, LLMError, RunCancelledError
from pedacito.local_server import LocalHTTPServer  # noqa: E402

GENERIC_SECRET = "AbCdEf0123456789XYZ"


def _cfg(**kw):
    cfg = config_mod.Config()
    cfg.base_url = "https://generativelanguage.googleapis.com/v1beta"
    cfg.model = "gemini-2.5-pro"
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_is_gemini_native():
    assert tg.is_gemini_native(_cfg())
    assert tg.is_gemini_native(
        _cfg(base_url="https://generativelanguage.googleapis.com"))
    # OpenAI-compat path stays OpenAI unless provider == "gemini"
    assert not tg.is_gemini_native(
        _cfg(base_url="https://generativelanguage.googleapis.com/v1beta/openai",
             provider="openai"))
    assert tg.is_gemini_native(
        _cfg(base_url="http://127.0.0.1:9999/v1beta", provider="gemini"))
    assert not tg.is_gemini_native(_cfg(gemini_native=False))
    assert not tg.is_gemini_native(
        _cfg(base_url="https://api.openai.com/v1", provider="openai"))


def test_endpoint_and_models_endpoint_spellings():
    assert tg.endpoint("https://generativelanguage.googleapis.com", "gemini-2.5-pro") \
        == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
    assert tg.endpoint("https://generativelanguage.googleapis.com/v1beta", "models/gemini-2.5-flash") \
        == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    assert tg.endpoint("https://generativelanguage.googleapis.com/v1beta/", "gemini-2.5-pro", stream=True) \
        == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:streamGenerateContent?alt=sse"
    assert tg.endpoint("http://127.0.0.1:8080/v1beta", "gemini-fake", stream=True) \
        == "http://127.0.0.1:8080/v1beta/models/gemini-fake:streamGenerateContent?alt=sse"
    assert tg.models_endpoint("https://generativelanguage.googleapis.com/v1beta/openai") \
        == "https://generativelanguage.googleapis.com/v1beta/models"


def test_build_headers():
    h = tg.build_headers("AIzaFakeKey")
    assert h == {
        "x-goog-api-key": "AIzaFakeKey",
        "Content-Type": "application/json",
    }


def test_translate_messages_system_images_and_tool_loop():
    sys_inst, contents = tg.translate_messages([
        {"role": "system", "content": "SYS1"},
        {"role": "system", "content": "SYS2"},
        {"role": "user", "content": [
            {"type": "text", "text": "inspect"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        ]},
        {"role": "assistant", "content": "checking", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {
                "name": "step", "arguments": '{"action": "read", "file": "a.py"}',
            }},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "RESULT:\nok"},
    ])
    assert sys_inst == {"parts": [{"text": "SYS1\n\nSYS2"}]}
    assert contents[0] == {
        "role": "user",
        "parts": [
            {"text": "inspect"},
            {"inlineData": {"mimeType": "image/png", "data": "AA=="}},
        ],
    }
    assert contents[1] == {
        "role": "model",
        "parts": [
            {"text": "checking"},
            {"functionCall": {"name": "step", "args": {"action": "read", "file": "a.py"}}},
        ],
    }
    assert contents[2] == {
        "role": "user",
        "parts": [
            {"functionResponse": {"name": "step", "response": {"result": "RESULT:\nok"}}},
        ],
    }


def test_build_request_thinking_schema_and_cached_content():
    cfg = _cfg(reasoning_effort="medium")
    schema = {
        "type": "object",
        "properties": {"action": {"type": "string"}},
        "required": ["action"],
        "additionalProperties": False,
    }
    body = tg.build_request(
        cfg,
        [{"role": "user", "content": "go"}],
        1024,
        temperature=0.2,
        response_schema=schema,
        cached_content="cachedContents/abc123",
    )
    assert body["cachedContent"] == "cachedContents/abc123"
    gc = body["generationConfig"]
    assert gc["maxOutputTokens"] == 1024
    assert gc["temperature"] == 0.2
    assert gc["thinkingConfig"] == {"thinkingBudget": 8192, "includeThoughts": True}
    assert gc["responseMimeType"] == "application/json"
    assert "additionalProperties" not in gc["responseSchema"]

    off = tg.build_request(
        _cfg(reasoning_effort="high", disable_thinking=True),
        [{"role": "user", "content": "go"}],
        200,
    )
    assert off["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}


def test_parse_response_thoughts_tools_finish_and_usage():
    data = {
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {"text": "let me think", "thought": True},
                    {"text": "final answer"},
                    {"functionCall": {"name": "step", "args": {"action": "done"}}},
                ],
            },
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": 120,
            "candidatesTokenCount": 30,
            "thoughtsTokenCount": 15,
            "cachedContentTokenCount": 80,
            "totalTokenCount": 165,
        },
    }
    parsed = tg.parse_response(data)
    assert parsed["reasoning"] == "let me think"
    assert parsed["text"] == "final answer"
    assert parsed["tool_calls"] == [{
        "id": "call_0",
        "name": "step",
        "arguments_json": '{"action": "done"}',
    }]
    assert parsed["finish_reason"] == "tool_calls"
    assert parsed["usage"] == {
        "prompt_tokens": 120,
        "completion_tokens": 45,
        "cached_tokens": 80,
        "completion_tokens_details": {"reasoning_tokens": 15},
        "total_tokens": 165,
    }

    assert tg.parse_response({"candidates": [{"finishReason": "MAX_TOKENS"}]})["finish_reason"] == "length"
    assert tg.parse_response({"candidates": [{"finishReason": "SAFETY"}]})["finish_reason"] == "content_filter"


def test_parse_sse_stream_and_cancel_and_error():
    lines = [
        'data: {"candidates":[{"content":{"parts":[{"text":"hmm","thought":true}]}}]}',
        'data: {"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}',
        'data: {"candidates":[{"content":{"parts":[{"text":"lo"}]},"finishReason":"STOP"}],'
        '"usageMetadata":{"promptTokenCount":50,"candidatesTokenCount":10,'
        '"thoughtsTokenCount":5,"cachedContentTokenCount":20}}',
    ]

    class Resp:
        def __init__(self, seq):
            self._seq = seq

        def iter_lines(self, decode_unicode=False):
            yield from self._seq

    texts, thoughts = [], []
    got = tg.parse_sse_stream(
        Resp(lines), on_text=texts.append, on_reasoning=thoughts.append)
    assert texts == ["Hel", "lo"]
    assert thoughts == ["hmm"]
    assert got["text"] == "Hello"
    assert got["reasoning"] == "hmm"
    assert got["finish_reason"] == "stop"
    assert got["usage"]["cached_tokens"] == 20
    assert got["usage"]["completion_tokens"] == 15

    with pytest.raises(tg.StreamCancelled):
        tg.parse_sse_stream(Resp(lines), cancel_check=lambda: True)

    with pytest.raises(tg.StreamError):
        tg.parse_sse_stream(
            Resp(['data: {"error":{"code":429,"status":"RESOURCE_EXHAUSTED","message":"quota"}}']))


# ------------------------------------------------------------ live fake server

class _GeminiHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.server.last_headers = dict(self.headers)
        if self.path.endswith("/models"):
            self._send_json({
                "models": [
                    {"name": "models/gemini-2.5-pro", "inputTokenLimit": 1048576,
                     "outputTokenLimit": 65536},
                ]
            })
        else:
            self._send_json({"error": {"message": "not found"}}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        self.server.last_path = self.path
        self.server.last_request = req
        self.server.last_headers = dict(self.headers)
        if self.server.stream or "streamGenerateContent" in self.path:
            self._stream()
        else:
            self._buffered()

    def _buffered(self):
        parts = []
        if self.server.thought is not None:
            parts.append({"text": self.server.thought, "thought": True})
        if self.server.text is not None:
            parts.append({"text": self.server.text})
        if self.server.tool is not None:
            parts.append({"functionCall": dict(self.server.tool)})
        self._send_json({
            "candidates": [{
                "content": {"role": "model", "parts": parts},
                "finishReason": self.server.finish_reason,
            }],
            "usageMetadata": dict(self.server.usage),
        })

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        if self.server.thought:
            chunk = {"candidates": [{"content": {"parts": [
                {"text": self.server.thought, "thought": True}]}}]}
            self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
        text = self.server.text or ""
        for i in range(0, len(text), 4):
            chunk = {"candidates": [{"content": {"parts": [
                {"text": text[i:i + 4]}]}}]}
            self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
        final = {
            "candidates": [{"content": {"parts": []},
                            "finishReason": self.server.finish_reason}],
            "usageMetadata": dict(self.server.usage),
        }
        self.wfile.write(("data: " + json.dumps(final) + "\n\n").encode())
        self.wfile.flush()


@pytest.fixture
def gemini_server():
    srv = LocalHTTPServer(("127.0.0.1", 0), _GeminiHandler)
    srv.stream = False
    srv.text = "Hello from Gemini"
    srv.thought = None
    srv.tool = None
    srv.finish_reason = "STOP"
    srv.usage = {
        "promptTokenCount": 100,
        "candidatesTokenCount": 20,
        "thoughtsTokenCount": 5,
        "cachedContentTokenCount": 40,
    }
    srv.last_request = None
    srv.last_headers = None
    srv.last_path = ""
    t = threading.Thread(target=srv.serve_forever, daemon=True, name="gemini-fake")
    t.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(5)


def _client(srv, **kw):
    cfg = config_mod.Config()
    cfg.base_url = f"http://127.0.0.1:{srv.server_address[1]}/v1beta"
    cfg.model = "gemini-2.5-pro"
    cfg.provider = "gemini"
    cfg.api_key = "AIza-test-key"
    cfg.api_key_env = ""
    for k, v in kw.items():
        setattr(cfg, k, v)
    return LLMClient(cfg)


def test_live_gemini_buffered_and_check(gemini_server):
    client = _client(gemini_server)
    assert client.check(warn=False) == "gemini-2.5-pro"
    assert gemini_server.last_headers["x-goog-api-key"] == "AIza-test-key"

    reply = client.chat([
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
    ], 256)
    assert reply == "Hello from Gemini"
    assert gemini_server.last_path == "/v1beta/models/gemini-2.5-pro:generateContent"
    assert gemini_server.last_request["systemInstruction"] == {"parts": [{"text": "SYS"}]}
    assert client.stats["cached_tokens"] == 40
    assert client.stats["reasoning_tokens"] == 5
    assert client.stats["completion_tokens"] == 25
    assert client.stats["prompt_tokens"] == 100


def test_live_gemini_streaming_and_cancel(gemini_server):
    gemini_server.stream = True
    gemini_server.thought = "analyzing..."
    gemini_server.text = "Streamed answer"
    client = _client(gemini_server)
    deltas = []
    client.stream_callback = deltas.append
    text = client.chat([{"role": "user", "content": "go"}], 200, stream=True)
    assert text == "Streamed answer"
    assert "".join(deltas) == "Streamed answer"
    assert client.last_stream_reasoning == "analyzing..."
    assert client.stats["cached_tokens"] == 40
    assert client.stats["reasoning_tokens"] == 5

    # Pre-cancelled event aborts cleanly
    ev = threading.Event()
    ev.set()
    with pytest.raises(RunCancelledError):
        client.chat([{"role": "user", "content": "go"}], 200, stream=True, cancel_event=ev)


def test_live_gemini_native_tool_and_schema(gemini_server):
    gemini_server.text = None
    gemini_server.tool = {"name": "step", "args": {"action": "done", "reasoning": "ok"}}
    client = _client(gemini_server, tool_mode="native")
    from pedacito.tools import schema_for
    out = client.chat_json([{"role": "user", "content": "go"}], schema_for(client.cfg), 200)
    assert out["action"] == "done"
    assert "tools" in gemini_server.last_request
    assert gemini_server.last_request["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"


def test_outbound_secret_guard_runs_before_gemini_translation(monkeypatch):
    class Resp:
        status_code = 200
        headers = {}
        encoding = "utf-8"

        def json(self):
            return {
                "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1},
            }

        def close(self):
            pass

    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None, stream=False):
        captured["url"] = url
        captured["body"] = json
        return Resp()

    monkeypatch.setattr("pedacito.llm.requests.post", fake_post)
    cfg = _cfg(security_outbound_mode="redact", api_key="k")
    client = LLMClient(cfg)
    client.chat([{"role": "user", "content": f'api_key = "{GENERIC_SECRET}"'}], 50)
    sent_text = captured["body"]["contents"][0]["parts"][0]["text"]
    assert GENERIC_SECRET not in sent_text
    assert security_mod.REDACTED_SECRET in sent_text


def test_gemini_native_false_falls_back_to_openai_path(monkeypatch):
    class Resp:
        status_code = 200
        headers = {}
        encoding = "utf-8"

        def json(self):
            return {
                "choices": [{"message": {"content": "compat ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            }

        def close(self):
            pass

    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None, stream=False):
        seen["url"] = url
        return Resp()

    monkeypatch.setattr("pedacito.llm.requests.post", fake_post)
    cfg = _cfg(gemini_native=False, api_key="k")
    assert LLMClient(cfg).chat([{"role": "user", "content": "hi"}], 20) == "compat ok"
    assert seen["url"].endswith("/chat/completions")


def test_gemini_401_raises_actionable_llmerror(monkeypatch):
    class ErrResp:
        status_code = 401
        headers = {}
        encoding = "utf-8"
        text = '{"error":{"message":"API key not valid"}}'

        def close(self):
            pass

    monkeypatch.setattr("pedacito.llm.requests.post", lambda *a, **k: ErrResp())
    with pytest.raises(LLMError) as exc:
        LLMClient(_cfg(api_key="bad")).chat([{"role": "user", "content": "hi"}], 20)
    assert "GEMINI_API_KEY" in str(exc.value)
