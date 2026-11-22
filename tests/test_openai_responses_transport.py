"""OpenAI Responses API (/v1/responses) transport tests (offline + loopback)."""

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
from pedacito import transport_openai as to
from pedacito.llm import LLMClient, LLMError, RunCancelledError
from pedacito.local_server import LocalHTTPServer  # noqa: E402

GENERIC_SECRET = "AbCdEf0123456789XYZ"


def _cfg(**kw):
    cfg = config_mod.Config()
    cfg.base_url = "https://api.openai.com/v1"
    cfg.model = "gpt-5.4"
    cfg.openai_responses = True
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_is_openai_responses():
    assert not to.is_openai_responses(config_mod.Config())
    assert to.is_openai_responses(_cfg())
    assert to.is_openai_responses(
        config_mod.Config(base_url="http://127.0.0.1:9999/v1",
                          provider="openai-responses"))


def test_responses_endpoint_spellings():
    assert to.responses_endpoint("https://api.openai.com") \
        == "https://api.openai.com/v1/responses"
    assert to.responses_endpoint("https://api.openai.com/v1") \
        == "https://api.openai.com/v1/responses"
    assert to.responses_endpoint("https://api.openai.com/v1/responses") \
        == "https://api.openai.com/v1/responses"
    assert to.responses_endpoint("http://127.0.0.1:8080/v1") \
        == "http://127.0.0.1:8080/v1/responses"
    assert to.models_endpoint("https://api.openai.com/v1/responses") \
        == "https://api.openai.com/v1/models"


def test_build_request_translates_instructions_input_tools_reasoning_and_cache():
    cfg = _cfg(reasoning_effort="high")
    tools = [{"type": "function", "function": {
        "name": "step", "description": "Pick step",
        "parameters": {"type": "object", "properties": {"action": {"type": "string"}}},
    }}]
    choice = {"type": "function", "function": {"name": "step"}}
    enc_item = {
        "type": "reasoning",
        "id": "rs_1",
        "encrypted_content": "ENC_BLOB",
        "summary": [{"type": "summary_text", "text": "thought"}],
    }
    body = to.build_request(
        cfg,
        [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            ]},
            {"role": "assistant", "content": "calling", "reasoning_items": [enc_item],
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "step", "arguments": '{"action":"read"}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "file body"},
        ],
        4096,
        tools=tools,
        tool_choice=choice,
        temperature=0.4,
        prompt_cache_key="proj-hash-123",
        previous_response_id="resp_prev",
    )
    assert body["instructions"] == "SYS"
    assert body["max_output_tokens"] == 4096
    assert body["reasoning"] == {"effort": "high"}
    assert body["include"] == ["reasoning.encrypted_content"]
    assert "temperature" not in body  # omitted while reasoning is active
    assert body["prompt_cache_key"] == "proj-hash-123"
    assert body["previous_response_id"] == "resp_prev"
    assert body["tools"] == [{
        "type": "function", "name": "step", "description": "Pick step",
        "parameters": {"type": "object", "properties": {"action": {"type": "string"}}},
    }]
    assert body["tool_choice"] == {"type": "function", "name": "step"}
    assert body["input"][0]["content"] == [
        {"type": "input_text", "text": "look"},
        {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
    ]
    assert body["input"][1] == enc_item
    assert body["input"][2]["type"] == "message"
    assert body["input"][3] == {
        "type": "function_call", "call_id": "call_1",
        "name": "step", "arguments": '{"action":"read"}',
    }
    assert body["input"][4] == {
        "type": "function_call_output", "call_id": "call_1", "output": "file body",
    }


def test_parse_response_extracts_reasoning_tools_and_usage():
    data = {
        "id": "resp_42",
        "status": "completed",
        "output": [
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "ENC",
             "summary": [{"type": "summary_text", "text": "planned it"}]},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "Done."}]},
            {"type": "function_call", "call_id": "fc_1", "name": "step",
             "arguments": '{"action":"done"}'},
        ],
        "usage": {
            "input_tokens": 200,
            "output_tokens": 50,
            "total_tokens": 250,
            "input_tokens_details": {"cached_tokens": 120},
            "output_tokens_details": {"reasoning_tokens": 18},
        },
    }
    got = to.parse_response(data)
    assert got["id"] == "resp_42"
    assert got["text"] == "Done."
    assert got["reasoning"] == "planned it"
    assert got["reasoning_items"][0]["encrypted_content"] == "ENC"
    assert got["tool_calls"] == [{
        "id": "fc_1", "name": "step", "arguments_json": '{"action":"done"}',
    }]
    assert got["finish_reason"] == "tool_calls"
    assert got["usage"] == {
        "prompt_tokens": 200,
        "completion_tokens": 50,
        "cached_tokens": 120,
        "reasoning_tokens": 18,
        "completion_tokens_details": {"reasoning_tokens": 18},
        "total_tokens": 250,
    }


def test_parse_sse_stream_handles_deltas_tools_and_cancel():
    lines = [
        'data: {"type":"response.created","response":{"id":"resp_99"}}',
        'data: {"type":"response.reasoning_summary_text.delta","delta":"think "}',
        'data: {"type":"response.output_item.done","item":{"type":"reasoning","id":"rs_9","encrypted_content":"BLOB"}}',
        'data: {"type":"response.output_text.delta","delta":"Hel"}',
        'data: {"type":"response.output_text.delta","delta":"lo"}',
        'data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","call_id":"c1","name":"step"}}',
        'data: {"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\\"action\\":\\"done\\"}"}',
        'data: {"type":"response.completed","response":{"id":"resp_99","status":"completed",'
        '"usage":{"input_tokens":100,"output_tokens":30,'
        '"input_tokens_details":{"cached_tokens":60},'
        '"output_tokens_details":{"reasoning_tokens":12}}}}',
    ]

    class Resp:
        def __init__(self, seq):
            self._seq = seq

        def iter_lines(self, decode_unicode=False):
            yield from self._seq

    texts, reasonings = [], []
    got = to.parse_sse_stream(
        Resp(lines), on_text=texts.append, on_reasoning=reasonings.append)
    assert got["id"] == "resp_99"
    assert texts == ["Hel", "lo"]
    assert reasonings == ["think "]
    assert got["text"] == "Hello"
    assert got["reasoning_items"] == [
        {"type": "reasoning", "id": "rs_9", "encrypted_content": "BLOB"}
    ]
    assert got["tool_calls"] == [{
        "id": "c1", "name": "step", "arguments_json": '{"action":"done"}',
    }]
    assert got["finish_reason"] == "tool_calls"
    assert got["usage"]["cached_tokens"] == 60
    assert got["usage"]["reasoning_tokens"] == 12

    with pytest.raises(to.StreamCancelled):
        to.parse_sse_stream(Resp(lines), cancel_check=lambda: True)

    with pytest.raises(to.StreamError):
        to.parse_sse_stream(
            Resp(['data: {"type":"error","error":{"code":"rate_limit_exceeded","message":"slow"}}']))


# ------------------------------------------------------------ live fake server

class _ResponsesHandler(BaseHTTPRequestHandler):
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
        if self.path.endswith("/models"):
            self._send_json({"data": [{"id": "gpt-5.4"}, {"id": "o3"}]})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        self.server.requests.append(req)
        self.server.last_path = self.path
        self.server.last_headers = dict(self.headers)
        if self.server.stream:
            self._stream()
        else:
            self._buffered()

    def _buffered(self):
        turn = len(self.server.requests)
        output = [
            {"type": "reasoning", "id": f"rs_{turn}", "encrypted_content": f"ENC_{turn}",
             "summary": [{"type": "summary_text", "text": self.server.reasoning}]},
        ]
        if self.server.tool is not None:
            output.append({"type": "function_call", "call_id": f"fc_{turn}",
                           "name": self.server.tool["name"],
                           "arguments": json.dumps(self.server.tool["arguments"])})
        else:
            output.append({"type": "message", "role": "assistant",
                           "content": [{"type": "output_text", "text": self.server.text}]})
        self._send_json({
            "id": f"resp_{turn}",
            "status": "completed",
            "output": output,
            "usage": dict(self.server.usage),
        })

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        events = [
            {"type": "response.created", "response": {"id": "resp_stream_1"}},
            {"type": "response.reasoning_summary_text.delta", "delta": self.server.reasoning},
            {"type": "response.output_item.done",
             "item": {"type": "reasoning", "id": "rs_s", "encrypted_content": "ENC_S"}},
        ]
        for i in range(0, len(self.server.text), 4):
            events.append({"type": "response.output_text.delta",
                           "delta": self.server.text[i:i + 4]})
        events.append({
            "type": "response.completed",
            "response": {"id": "resp_stream_1", "status": "completed",
                         "usage": dict(self.server.usage)},
        })
        for ev in events:
            self.wfile.write(("data: " + json.dumps(ev) + "\n\n").encode())
        self.wfile.flush()


@pytest.fixture
def responses_server():
    srv = LocalHTTPServer(("127.0.0.1", 0), _ResponsesHandler)
    srv.stream = False
    srv.text = "Responses ok"
    srv.reasoning = "thinking..."
    srv.tool = None
    srv.usage = {
        "input_tokens": 150,
        "output_tokens": 40,
        "input_tokens_details": {"cached_tokens": 90},
        "output_tokens_details": {"reasoning_tokens": 14},
    }
    srv.requests = []
    srv.last_path = ""
    srv.last_headers = {}
    t = threading.Thread(target=srv.serve_forever, daemon=True, name="oai-resp-fake")
    t.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(5)


def _client(srv, **kw):
    cfg = config_mod.Config()
    cfg.base_url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    cfg.model = "o3"
    cfg.provider = "openai-responses"
    cfg.openai_responses = True
    cfg.api_key = "sk-test"
    cfg.api_key_env = ""
    cfg.reasoning_effort = "medium"
    cfg.prompt_cache_key = "my-cache-key"
    for k, v in kw.items():
        setattr(cfg, k, v)
    return LLMClient(cfg)


def test_live_responses_buffered_multi_turn_preserves_encrypted_reasoning(responses_server):
    client = _client(responses_server)
    assert "o3" in client.check(warn=False)
    r1 = client.chat([{"role": "user", "content": "first"}], 200)
    assert r1 == "Responses ok"
    assert responses_server.last_path == "/v1/responses"
    req1 = responses_server.requests[0]
    assert req1["reasoning"] == {"effort": "medium"}
    assert client.last_response_id == "resp_1"
    assert client.stats["cached_tokens"] == 90
    assert client.stats["reasoning_tokens"] == 14

    # Turn 2 replays assistant message; client attaches cached encrypted reasoning item
    client.chat([
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": r1},
        {"role": "user", "content": "second"},
    ], 200)
    req2 = responses_server.requests[1]
    assert any(
        item.get("type") == "reasoning" and item.get("encrypted_content") == "ENC_1"
        for item in req2["input"]
    )


def test_live_responses_streaming_and_cancel(responses_server):
    responses_server.stream = True
    client = _client(responses_server)
    deltas = []
    client.stream_callback = deltas.append
    out = client.chat([{"role": "user", "content": "hi"}], 200, stream=True)
    assert out == "Responses ok"
    assert "".join(deltas) == "Responses ok"
    assert client.last_stream_reasoning == "thinking..."
    assert client.stats["cached_tokens"] == 90
    assert client.stats["reasoning_tokens"] == 14

    ev = threading.Event()
    ev.set()
    with pytest.raises(RunCancelledError):
        client.chat([{"role": "user", "content": "hi"}], 200, stream=True, cancel_event=ev)


def test_outbound_secret_guard_runs_before_openai_responses(monkeypatch):
    class Resp:
        status_code = 200
        headers = {}
        encoding = "utf-8"

        def json(self):
            return {
                "id": "r_1", "status": "completed",
                "output": [{"type": "message", "role": "assistant",
                            "content": [{"type": "output_text", "text": "ok"}]}],
                "usage": {"input_tokens": 5, "output_tokens": 1},
            }

        def close(self):
            pass

    captured = {}
    monkeypatch.setattr(
        "pedacito.llm.requests.post",
        lambda url, headers=None, json=None, timeout=None, stream=False: (
            captured.update({"url": url, "body": json}) or Resp()
        ),
    )
    cfg = _cfg(security_outbound_mode="redact", api_key="sk-test",
               prompt_cache_key="my-cache-key")
    LLMClient(cfg).chat([{"role": "user", "content": f'api_key = "{GENERIC_SECRET}"'}], 50)
    assert captured["body"]["prompt_cache_key"] == "my-cache-key"
    sent_text = captured["body"]["input"][0]["content"]
    assert GENERIC_SECRET not in sent_text
    assert security_mod.REDACTED_SECRET in sent_text


def test_openai_responses_401_raises_llmerror(monkeypatch):
    class ErrResp:
        status_code = 401
        headers = {}
        encoding = "utf-8"
        text = '{"error":{"message":"invalid api key"}}'

        def close(self):
            pass

    monkeypatch.setattr("pedacito.llm.requests.post", lambda *a, **k: ErrResp())
    with pytest.raises(LLMError) as exc:
        LLMClient(_cfg(api_key="bad")).chat([{"role": "user", "content": "hi"}], 20)
    assert "OPENAI_API_KEY" in str(exc.value)
