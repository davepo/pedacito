"""Regression tests for v0.3.0 live streaming.

Covers the SSE client (delta.reasoning_content, inline <think> split across
chunks, final usage telemetry, instant cancel_event), the transparent JSON
fallback for non-streaming mocks, and the GUI's live drawer state plus the
Stop endpoint wiring.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import config as config_mod
from pedacito.gui import core, jobs, server
from pedacito.llm import LLMClient, RunCancelledError


def _cfg(**kw):
    cfg = config_mod.Config()
    cfg.base_url = "http://127.0.0.1:1/v1"
    cfg.model = "tiny"
    for key, value in kw.items():
        setattr(cfg, key, value)
    return cfg


class Resp:
    def __init__(self, lines=None, body=None,
                 ctype="text/event-stream"):
        self.status_code = 200
        self._lines = list(lines or [])
        self._body = body
        self.headers = {"Content-Type": ctype} if ctype else {}
        self.text = ""
        self.closed = False
        self.consumed = 0

    def iter_lines(self, decode_unicode=False):
        for line in self._lines:
            self.consumed += 1
            yield line

    def json(self):
        return self._body

    def close(self):
        self.closed = True


def _sse(obj) -> str:
    return "data: " + json.dumps(obj)


def _chunk(text: str = "", reasoning: str = "", finish=None) -> str:
    delta = {}
    if text:
        delta["content"] = text
    if reasoning:
        delta["reasoning_content"] = reasoning
    return _sse({"choices": [{"delta": delta, "finish_reason": finish}]})


def _usage(prompt=100, completion=7) -> str:
    return _sse({"choices": [], "usage": {
        "prompt_tokens": prompt, "completion_tokens": completion}})


def _patch_post(monkeypatch, response):
    monkeypatch.setattr("pedacito.llm.requests.post",
                        lambda *a, **k: response)


def test_reasoning_and_content_deltas_assemble_and_account_usage(monkeypatch):
    lines = [
        _chunk(reasoning="I should "),
        _chunk(reasoning="think."),
        _chunk("Hello "),
        _chunk("world"),
        _chunk(finish="stop"),
        _usage(100, 7),
        "data: [DONE]",
    ]
    response = Resp(lines)
    _patch_post(monkeypatch, response)
    client = LLMClient(_cfg())
    client.echo_stream = False
    deltas, events = [], []
    client.stream_callback = deltas.append
    client.on_stream = events.append

    text = client.chat([{"role": "user", "content": "hi"}], 50, stream=True)

    assert text == "Hello world"
    assert deltas == ["Hello ", "world"]
    assert response.closed
    # Final usage wins over the counted estimate.
    assert (client.stats["prompt_tokens"], client.stats["completion_tokens"]) \
        == (100, 7)
    assert client.last_completion_tokens == 7
    assert client.last_stream_reasoning == "I should think."
    assert client.last_stream_tokens == 7
    # Reasoning and content are distinct phases, and the final event carries
    # the reported token count.
    phases = {event["phase"] for event in events}
    assert {"reasoning", "content"} <= phases
    assert all(event["kind"] == "llm_stream" for event in events)
    assert events[-1]["tokens_so_far"] == 7
    assert any(event["delta_reasoning"] for event in events)
    assert any(event["delta_content"] for event in events)


@pytest.mark.parametrize("stream", [True, False])
def test_mid_stream_usage_frame_updates_live_token_count(monkeypatch, stream):
    """A usage frame can arrive before the last deltas. Live llm_stream
    events emitted after it must carry the server's reported completion
    count, not the char estimate -- in the primary streaming loop and in
    the buffered-SSE reader alike."""
    lines = [_chunk("start "), _usage(100, 50)]
    lines.extend(_chunk("x") for _ in range(16))
    lines.append(_chunk(finish="stop"))
    lines.append("data: [DONE]")
    response = Resp(lines)
    _patch_post(monkeypatch, response)
    client = LLMClient(_cfg())
    client.echo_stream = False
    events = []
    client.on_stream = events.append

    text = client.chat([{"role": "user", "content": "hi"}], 50, stream=stream)

    assert text == "start " + "x" * 16
    # The chunk-count flush after the usage frame reports 50 tokens; the
    # char estimate for ~22 chars would be ~6. The final event is excluded
    # so a mid-stream event must carry the reported count.
    assert any(e["tokens_so_far"] == 50 for e in events[:-1]), \
        [e["tokens_so_far"] for e in events]


def test_inline_think_tags_split_across_chunks(monkeypatch):
    lines = [
        _chunk("pre<thi"),
        _chunk("nk>hidden "),
        _chunk("plan</think>visible"),
        _chunk(finish="stop"),
        "data: [DONE]",
    ]
    _patch_post(monkeypatch, Resp(lines))
    client = LLMClient(_cfg())
    client.echo_stream = False
    deltas, events = [], []
    client.stream_callback = deltas.append
    client.on_stream = events.append

    text = client.chat([{"role": "user", "content": "hi"}], 50, stream=True)

    assert text == "previsible"          # the <think> block is hidden
    assert deltas == ["pre", "visible"]  # never the raw tags
    assert client.last_reasoning_chars == len("hidden plan")
    assert client.last_stream_reasoning == "hidden plan"
    reasoning_events = [e for e in events if e["phase"] == "reasoning"]
    content_events = [e for e in events if e["phase"] == "content"]
    assert reasoning_events and content_events
    assert "hidden plan" in "".join(e["delta_reasoning"]
                                    for e in reasoning_events)
    assert "<think>" not in "".join(e["delta_content"]
                                    for e in content_events)


def test_cancel_event_aborts_mid_stream_immediately(monkeypatch):
    response = Resp([
        _chunk("one"),
        _chunk("two"),
        _chunk("three"),
        "data: [DONE]",
    ])
    _patch_post(monkeypatch, response)
    client = LLMClient(_cfg())
    client.echo_stream = False
    cancel = threading.Event()
    client.stream_callback = lambda delta: cancel.set()

    with pytest.raises(RunCancelledError):
        client.chat([{"role": "user", "content": "hi"}], 50, stream=True,
                    cancel_event=cancel)

    assert cancel.is_set()
    assert response.closed
    # The first delta triggered the cancel; the reader stopped before the
    # remaining chunks were consumed.
    assert response.consumed == 1
    assert client.partial_text == "one"


@pytest.mark.parametrize("stream", [True, False])
def test_cancel_before_response_registration_stops_before_any_read(
        monkeypatch, stream):
    """`cancel()` can land between `_post` returning and the response being
    registered as `_active_response`. Both readers must notice the pending
    stop immediately after registering, close the response and raise without
    consuming a line (and without accounting a partial completion)."""
    cancel = threading.Event()
    response = Resp([_chunk("late")])
    _patch_post(monkeypatch, response)
    client = LLMClient(_cfg())
    client.echo_stream = False

    def cancelling_post(payload, stream=False):
        cancel.set()
        return response

    client._post = cancelling_post
    with pytest.raises(RunCancelledError):
        client.chat([{"role": "user", "content": "hi"}], 50,
                    stream=stream, cancel_event=cancel)

    assert response.consumed == 0
    assert response.closed
    assert client.stats["completion_tokens"] == 0


def test_buffered_sse_transport_error_on_cancel_is_a_stop(monkeypatch):
    """A non-streaming request answered with SSE is read by
    _consume_stream_response. When cancel() closes the socket from another
    thread and the read surfaces as a transport error, the stop must raise
    RunCancelledError -- not return the partial reply as if it completed."""
    cancel = threading.Event()

    class CancelResp(Resp):
        def iter_lines(self, decode_unicode=False):
            self.consumed += 1
            yield _chunk("part")
            cancel.set()
            raise requests.exceptions.ConnectionError("closed by cancel")

    response = CancelResp()
    _patch_post(monkeypatch, response)
    client = LLMClient(_cfg())
    client.echo_stream = False
    # stream=False keeps the request buffered; the SSE content-type then
    # routes the reply through _consume_stream_response.
    with pytest.raises(RunCancelledError):
        client.chat([{"role": "user", "content": "hi"}], 50, stream=False,
                    cancel_event=cancel)
    assert client.partial_text == "part"
    assert response.closed
    # A stop is not a completed reply: nothing was accounted as generated.
    assert client.stats["completion_tokens"] == 0


def test_non_streaming_json_responses_stay_transparent(monkeypatch):
    body = {
        "choices": [{"message": {"content": "plain answer"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }
    _patch_post(monkeypatch, Resp(body=body, ctype="application/json"))
    client = LLMClient(_cfg())
    assert client.chat([{"role": "user", "content": "hi"}], 20) \
        == "plain answer"

    # And a server/test double that ignores `stream: true` still works.
    client2 = LLMClient(_cfg())
    assert client2.chat([{"role": "user", "content": "hi"}], 20,
                        stream=True) == "plain answer"


class ConsumedResp(Resp):
    """Faithful requests.Response behaviour: once iter_lines has consumed
    the body, .json() raises. The plain Resp double (working .json()) masks
    the real fallback path."""

    def iter_lines(self, decode_unicode=False):
        self.consumed += 1
        yield from self._lines

    def json(self):
        if self.consumed:
            raise RuntimeError(
                "The content for this response was already consumed")
        return self._body


def test_stream_json_fallback_parses_the_buffered_body(monkeypatch):
    """A server that ignores `stream: true` and labels its JSON body with a
    non-JSON content-type must still be understood: the streaming reader
    buffers the raw lines and parses them when r.json() fails."""
    body = json.dumps({
        "choices": [{"message": {"content": "plain answer"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }, indent=1)                              # several raw lines, one object
    response = ConsumedResp(lines=body.splitlines(), ctype="text/plain")
    _patch_post(monkeypatch, response)
    client = LLMClient(_cfg())
    client.echo_stream = False
    assert client.chat([{"role": "user", "content": "hi"}], 20,
                       stream=True) == "plain answer"
    assert client.last_completion_tokens == 2
    assert response.closed


def test_gui_status_exposes_live_fields_and_stop_sets_cancel_event(monkeypatch,
                                                                 tmp_path):
    state = core.State()
    monkeypatch.setattr(jobs, "STATE", state)
    try:
        job = state.begin_job("task", "streaming test")
        assert job is not None
        state.note_llm_stream({
            "phase": "reasoning", "delta_reasoning": "thinking",
            "delta_content": "", "tokens_so_far": 12,
            "tok_per_sec": 8.5,
        })
        state.note_llm_stream({
            "phase": "content", "delta_reasoning": "",
            "delta_content": "print('hi')", "tokens_so_far": 18,
            "tok_per_sec": 9.25,
        })
        payload = state.status()["job"]
        assert payload["live_reasoning"] == "thinking"
        assert payload["live_content"] == "print('hi')"
        assert payload["live_tokens"] == 18
        assert payload["live_tok_per_sec"] == 9.2
        assert payload["stream_phase"] == "content"

        class FakeClient:
            def __init__(self):
                self.cancelled = False

            def cancel(self):
                self.cancelled = True

        client = FakeClient()
        job.client = client
        code, reply = server.POST_ROUTES["/api/stop"]({}, {})
        assert code == 200 and reply["ok"] is True
        assert job.cancel_event.is_set()
        assert client.cancelled
        assert job.stop_reason == "Stopped by user"
    finally:
        with state.lock:
            if state.job is not None:
                state.job.active = False
        state.bus.close()
