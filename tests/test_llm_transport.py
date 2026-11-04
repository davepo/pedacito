"""Transport hardening: error contract, rate limits, streaming (callback,
usage, cache hits, cancel, dropped connections), budgets, sampler protection."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import config as config_mod
from pedacito import llm as llm_mod
from pedacito.llm import (BudgetExceeded, CancelledError, CircuitOpenError,
                          LLMClient, LLMError, estimate_messages,
                          extract_content, strip_reasoning)


def _cfg(**kw):
    cfg = config_mod.Config()
    cfg.base_url = "http://127.0.0.1:1/v1"
    cfg.model = "m"
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


class Resp:
    def __init__(self, status=200, body=None, text="", lines=None, headers=None,
                 raise_after=None):
        self.status_code = status
        self._body = body
        self.text = text or (json.dumps(body) if body is not None else "")
        self._lines = lines or []
        self.headers = headers or {}
        self.closed = False
        self._raise_after = raise_after

    def json(self):
        return self._body

    def iter_lines(self, decode_unicode=False):
        yield from self._lines
        if self._raise_after is not None:
            raise self._raise_after

    def close(self):
        self.closed = True


def _ok(text="hello", usage=None):
    return Resp(200, {"choices": [{"message": {"content": text},
                                   "finish_reason": "stop"}],
                      "usage": usage or {}})


def _sse(obj):
    return "data: " + json.dumps(obj)


def _chunk(text="", finish=None, usage=None):
    d = {"choices": [{"delta": {"content": text} if text else {},
                      "finish_reason": finish}]}
    if usage is not None:
        d = {"choices": [], "usage": usage}
    return _sse(d)


def _patch_post(monkeypatch, fn):
    monkeypatch.setattr("pedacito.llm.requests.post", fn)


# --------------------------------------------------------------- error contract
@pytest.mark.parametrize("body,expected", [
    ("the request exceeds the available context size (n_ctx=4096)", True),
    ("This model's maximum context length is 8192 tokens", True),
    ("error: context_length_exceeded", True),
    ("too many tokens in the prompt", True),
    ("Trying to keep the first 4096 tokens when context length is only 2048", True),
    ("temperature 5 exceeds the maximum allowed value of 2", False),
    ("top_k value is out of range", False),
    ("invalid request: exceeds rate", False),
])
def test_context_overflow_detection_is_specific(body, expected):
    assert LLMClient(_cfg())._context_too_long(body) is expected


def test_timeout_and_request_errors_become_llmerror(monkeypatch):
    def timeout(*a, **k):
        raise requests.exceptions.ReadTimeout("slow")

    _patch_post(monkeypatch, timeout)
    with pytest.raises(LLMError) as e:
        LLMClient(_cfg(timeout=7.0)).chat([{"role": "user", "content": "x"}], 10)
    assert "did not answer within 7s" in str(e.value)

    def boom(*a, **k):
        raise requests.exceptions.InvalidURL("bad")

    _patch_post(monkeypatch, boom)
    with pytest.raises(LLMError) as e:
        LLMClient(_cfg()).chat([{"role": "user", "content": "x"}], 10)
    assert "InvalidURL" in str(e.value)


def test_error_kinds_form_a_taxonomy():
    assert llm_mod.ModelLoadError.kind == "load"
    assert llm_mod.TemplateError.kind == "template"
    assert BudgetExceeded.kind == "budget" and CancelledError.kind == "cancelled"


def test_stale_keepalive_connection_is_retried_once(monkeypatch):
    client = LLMClient(_cfg())
    calls = []

    class Sess:
        def __init__(self, fail):
            self.fail = fail

        def post(self, url, **kw):
            calls.append(self.fail)
            if self.fail:
                raise requests.exceptions.ConnectionError("closed by peer")
            return _ok("fine")

        def close(self):
            pass

    client._session = Sess(True)
    client._session_used = True
    monkeypatch.setattr(requests, "Session", lambda: Sess(False))
    assert client.chat([{"role": "user", "content": "x"}], 10) == "fine"
    assert calls == [True, False]
    # a genuinely dead server still reports as unreachable
    client = LLMClient(_cfg())
    client._session = type("S", (), {"post": lambda s, u, **k: (_ for _ in ()).throw(
        requests.exceptions.ConnectionError("down")), "close": lambda s: None})()
    monkeypatch.setattr(requests, "Session", lambda: client._session)
    with pytest.raises(LLMError) as e:
        client.chat([{"role": "user", "content": "x"}], 10)
    assert "Cannot reach the server" in str(e.value)


# ------------------------------------------------------------------ rate limits
def test_429_honours_retry_after_and_ignores_the_model_load_deadline(monkeypatch):
    sleeps = []
    monkeypatch.setattr("pedacito.llm.time.sleep", sleeps.append)
    monkeypatch.setattr("pedacito.llm.random.uniform", lambda a, b: 0.0)
    seq = [Resp(429, text="slow down", headers={"Retry-After": "3"}),
           Resp(503, text="overloaded", headers={}), _ok("done")]
    _patch_post(monkeypatch, lambda *a, **k: seq.pop(0))
    client = LLMClient(_cfg(load_wait_seconds=0))       # load deadline already spent
    assert client.chat([{"role": "user", "content": "x"}], 10) == "done"
    assert sleeps[0] == 3.0 and sleeps[1] == 4.0         # Retry-After, then backoff


def test_persistent_rate_limit_gives_up_with_a_clear_error(monkeypatch):
    monkeypatch.setattr("pedacito.llm.time.sleep", lambda s: None)
    n = []
    _patch_post(monkeypatch, lambda *a, **k: n.append(1) or Resp(429, text="no"))
    with pytest.raises(LLMError) as e:
        LLMClient(_cfg()).chat([{"role": "user", "content": "x"}], 10)
    assert "rate-limiting" in str(e.value)
    assert len(n) == llm_mod._MAX_RATE_LIMIT_RETRIES + 1


def test_circuit_breaker_fails_fast_after_repeated_server_errors(monkeypatch):
    calls = []
    _patch_post(monkeypatch, lambda *a, **k: calls.append(1) or Resp(
        500, text="boom"))
    cfg = _cfg(circuit_breaker=True, circuit_failures=3, circuit_cooldown=30)
    client = LLMClient(cfg)
    for _ in range(3):
        with pytest.raises(LLMError):
            client.chat([{"role": "user", "content": "x"}], 10)
    assert len(calls) == 3
    with pytest.raises(CircuitOpenError) as e:
        client.chat([{"role": "user", "content": "x"}], 10)
    assert "pausing requests" in str(e.value)
    assert len(calls) == 3                 # no request while open
    assert client.circuit_trips == 1

    # Cooldown elapsed: one probe is allowed, and a success closes the circuit.
    monkeypatch.setattr(llm_mod.time, "monotonic",
                        lambda: client._circuit_until + 1)
    _patch_post(monkeypatch, lambda *a, **k: _ok("fine"))
    assert client.chat([{"role": "user", "content": "x"}], 10) == "fine"
    assert client._circuit_until == 0 and client._failures == 0


def test_circuit_breaker_ignores_rate_limits_and_bad_requests(monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    posts = []
    _patch_post(monkeypatch, lambda *a, **k: posts.append(1) or Resp(
        429, text="slow down"))
    cfg = _cfg(circuit_breaker=True, circuit_failures=1)
    client = LLMClient(cfg)
    with pytest.raises(LLMError):
        client.chat([{"role": "user", "content": "x"}], 10)
    assert client.circuit_trips == 0      # a rate limit is not a dead server
    assert len(posts) > 1                 # the normal retry budget still ran

    _patch_post(monkeypatch, lambda *a, **k: Resp(400, text="bad request"))
    with pytest.raises(LLMError):
        client.chat([{"role": "user", "content": "x"}], 10)
    assert client.circuit_trips == 0      # a bad request will not heal by waiting


def test_circuit_breaker_can_be_disabled(monkeypatch):
    calls = []
    _patch_post(monkeypatch, lambda *a, **k: calls.append(1) or Resp(
        500, text="boom"))
    client = LLMClient(_cfg(circuit_breaker=False, circuit_failures=1))
    for _ in range(3):
        with pytest.raises(LLMError):
            client.chat([{"role": "user", "content": "x"}], 10)
    assert len(calls) == 3 and client.circuit_trips == 0


def test_model_loading_still_uses_the_load_deadline(monkeypatch):
    monkeypatch.setattr("pedacito.llm.time.sleep", lambda s: None)
    _patch_post(monkeypatch, lambda *a, **k: Resp(503, text="model is loading"))
    with pytest.raises(LLMError) as e:
        LLMClient(_cfg(load_wait_seconds=0)).chat([{"role": "user", "content": "x"}], 10)
    assert "still not ready" in str(e.value)


# --------------------------------------------------------------- request body
def test_sampler_cannot_override_structural_keys(monkeypatch):
    posts = []
    _patch_post(monkeypatch, lambda *a, **k: posts.append(k["json"]) or _ok())
    cfg = _cfg(sampler={"max_tokens": 10 ** 6, "response_format": {"type": "text"},
                        "tools": [1], "stream_options": {"x": 1}, "top_k": 7,
                        "chat_template_kwargs": {"preserve_thinking": True}},
               disable_thinking=True)
    LLMClient(cfg).chat([{"role": "user", "content": "x"}], 123)
    body = posts[0]
    assert body["max_tokens"] == 123
    assert "tools" not in body and "response_format" not in body
    assert body["top_k"] == 7
    # a user's chat_template_kwargs survive and merge with ours
    assert body["chat_template_kwargs"] == {"preserve_thinking": True,
                                            "enable_thinking": False,
                                            "thinking": False}


# ----------------------------------------------------- thinking compatibility
@pytest.mark.parametrize("text,expected", [
    ("<think>plan</think>answer", "answer"),
    ("<thinking>plan</thinking>answer", "answer"),
    ("<reasoning>plan</reasoning>answer", "answer"),
    ("<think>plan", ""),
    ("preamble\n<think>plan", "preamble"),
    ("answer</think>", "answer"),
    ("[THINK]plan[/THINK]answer", "answer"),
    ("[think]plan[/think]answer", "answer"),
    ("[THINK]plan", ""),
    ("answer[/THINK]", "answer"),
    ("<seed:think>plan</seed:think>answer", "answer"),
    ("<seed:think>plan", ""),
    ("<seed:cot_budget_reflect>spent</seed:cot_budget_reflect>answer",
     "answer"),
    ('<|channel|>analysis<|message|>plan<|end|><|start|>assistant'
     '<|channel|>final<|message|>answer<|end|>', "answer"),
    ("<|channel|>analysis<|message|>only thoughts<|end|>", ""),
    ("<|channel>thought\nplan<channel|>answer", "answer"),
    ("<|channel>thought\n<channel|>answer", "answer"),
    ("<|turn>model\n<|channel>thought\nplan<channel|>answer<turn|>",
     "answer"),
    ("<|channel>thought\nplan", ""),
])
def test_strip_reasoning_handles_family_markers(text, expected):
    assert strip_reasoning(text) == expected


def test_extract_content_reads_other_reasoning_fields():
    assert extract_content({"content": "", "thinking": "<think>x</think>ok"}) \
        == "ok"
    assert extract_content({"content": "",
                            "reasoning_details": [
                                {"text": "<think>x</think>ok"}]}) == "ok"


@pytest.mark.parametrize("model", [
    "Qwen3.8-27B-GGUF", "qwen3:32b", "gemma-4-31b-it",
    "Nanbeige4.2-3B", "LiquidAI/LFM2.5-1.2B-Thinking",
    "some-org/Unknown-Model-7B",
])
def test_local_disable_sends_every_common_template_switch(monkeypatch, model):
    posts = []
    _patch_post(monkeypatch, lambda *a, **k: posts.append(k["json"]) or _ok())
    LLMClient(_cfg(model=model, disable_thinking=True)).chat(
        [{"role": "user", "content": "x"}], 20)
    body = posts[0]
    assert body["chat_template_kwargs"]["enable_thinking"] is False
    assert body["chat_template_kwargs"]["thinking"] is False
    assert body["think"] is False


@pytest.mark.parametrize("model,where,key,value", [
    ("Magistral-Small-2509", "kwargs", "reasoning_effort", "none"),
    ("Seed-OSS-36B-Instruct", "kwargs", "thinking_budget", 0),
    ("gpt-oss-20b", "body", "reasoning_effort", "low"),
])
def test_local_family_specific_thinking_switches(
        monkeypatch, model, where, key, value):
    posts = []
    _patch_post(monkeypatch, lambda *a, **k: posts.append(k["json"]) or _ok())
    LLMClient(_cfg(model=model, disable_thinking=True)).chat(
        [{"role": "user", "content": "x"}], 20)
    body = posts[0]
    container = body["chat_template_kwargs"] if where == "kwargs" else body
    assert container[key] == value


def test_qwen_disables_with_hard_and_soft_switches_but_nanbeige_only_hard(
        monkeypatch):
    def body_for(model):
        posts = []
        _patch_post(monkeypatch,
                    lambda *a, **k: posts.append(k["json"]) or _ok())
        LLMClient(_cfg(model=model, disable_thinking=True)).chat(
            [{"role": "user", "content": "x"}], 20)
        return posts[0]

    qwen = body_for("Qwen3.8-27B-GGUF")
    assert qwen["messages"][-1]["content"].endswith("/no_think")
    nanbeige = body_for("Nanbeige4.2-3B")
    assert "/no_think" not in nanbeige["messages"][-1]["content"]


@pytest.mark.parametrize("base_url,model,check", [
    ("https://api.moonshot.ai/v1", "Kimi K2.6",
     {"thinking": {"type": "disabled"}}),
    ("https://api.moonshot.ai/v1", "kimi-k2.7-code", {}),
    ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen3-32b",
     {"enable_thinking": False}),
    ("https://api.inference.wandb.ai/v1", "google/gemma-4-31B-it",
     {"chat_template_kwargs": {"enable_thinking": False}}),
    ("https://api.deepseek.com/v1", "deepseek-chat", {}),
    ("https://example.com/v1", "deepseek-v3.2",
     {"chat_template_kwargs": {"thinking": False,
                               "enable_thinking": False}}),
    ("https://api.openai.com/v1", "gpt-5", {"reasoning_effort": "low"}),
    ("https://example.com/v1", "some-model", {}),
])
def test_hosted_disable_uses_only_known_provider_fields(
        monkeypatch, base_url, model, check):
    posts = []
    _patch_post(monkeypatch, lambda *a, **k: posts.append(k["json"]) or _ok())
    LLMClient(_cfg(base_url=base_url, model=model,
                   disable_thinking=True)).chat(
        [{"role": "user", "content": "x"}], 20)
    body = posts[0]
    for key, value in check.items():
        assert body[key] == value
    if not check:
        assert "chat_template_kwargs" not in body
        assert "think" not in body
        assert "thinking" not in body
        assert "reasoning_effort" not in body


def test_message_framing_overhead_is_counted():
    msgs = [{"role": "user", "content": "x" * 36}] * 10
    plain = sum(llm_mod.estimate_tokens(m["content"]) for m in msgs)
    assert estimate_messages(msgs) == plain + 4 * 10 + 3


def test_client_never_rewrites_history_to_fit_it_only_lowers_max_tokens(monkeypatch, capsys):
    posts = []
    _patch_post(monkeypatch, lambda *a, **k: posts.append(k["json"]) or _ok())
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": "map"}]
    for i in range(6):
        msgs += [{"role": "assistant", "content": f"{{'action': 'read', 'n': {i}}}"},
                 {"role": "user", "content": "RESULT:\n" + "line\n" * 300}]
    cfg = _cfg(context_tokens=4000)
    LLMClient(cfg).chat(msgs, max_tokens=2000)
    sent = posts[0]
    assert [m["content"] for m in sent["messages"]] == [m["content"] for m in msgs]
    assert sent["max_tokens"] < 2000
    assert "lowering the reply budget" in capsys.readouterr().err


def test_warmup_probes_the_system_role_before_the_first_real_turn(monkeypatch):
    posts = []

    def fake(*a, **k):
        posts.append(k["json"])
        if any(m["role"] == "system" for m in k["json"]["messages"]):
            return Resp(400, text="jinja error: system role not supported")
        return _ok("ok")

    _patch_post(monkeypatch, fake)
    monkeypatch.setattr(llm_mod.requests, "get",
                        lambda *a, **k: Resp(404), raising=False)
    monkeypatch.setattr(llm_mod.requests, "post", fake)
    client = LLMClient(_cfg(json_mode="object"))
    client.warmup()
    assert client._no_system_role is True
    chats = [p for p in posts if "messages" in p]
    assert any(m["role"] == "system" for m in chats[0]["messages"])   # the probe
    assert not any(m["role"] == "system" for m in chats[-1]["messages"])


# ------------------------------------------------------------------- streaming
def test_stream_callback_replaces_stdout_and_usage_is_captured(monkeypatch, capsys):
    posts = []
    lines = [_chunk("Hel"), _chunk("lo"), _chunk(finish="stop"),
             _chunk(usage={"prompt_tokens": 100, "completion_tokens": 7,
                           "prompt_tokens_details": {"cached_tokens": 80}}),
             "data: [DONE]"]
    resp = Resp(200, lines=lines)

    def fake(*a, **k):
        posts.append(k["json"])
        return resp

    _patch_post(monkeypatch, fake)
    client = LLMClient(_cfg(price_input_per_mtok=1.0, price_output_per_mtok=2.0,
                            price_cached_per_mtok=0.1))
    got, records = [], []
    client.stream_callback = got.append
    client.on_call = records.append
    client.site = "answer"
    text = client.chat([{"role": "user", "content": "x"}], 50, stream=True)
    assert text == "Hello" and got == ["Hel", "lo"]
    assert capsys.readouterr().out == ""                 # nothing echoed
    assert resp.closed
    assert posts[0]["stream_options"] == {"include_usage": True}
    s = client.stats
    assert (s["prompt_tokens"], s["completion_tokens"], s["cached_tokens"]) == (100, 7, 80)
    expected = (20 * 1.0 + 80 * 0.1 + 7 * 2.0) / 1e6
    assert s["cost"] == pytest.approx(expected)
    assert s["by_site"]["answer"]["cached_tokens"] == 80
    assert records[0]["site"] == "answer" and records[0]["cached_tokens"] == 80


def test_cached_token_fields_of_other_providers(monkeypatch):
    for usage, want in (({"prompt_tokens": 50, "completion_tokens": 1,
                          "prompt_cache_hit_tokens": 30}, 30),):
        _patch_post(monkeypatch, lambda *a, **k: _ok("x", usage))
        c = LLMClient(_cfg())
        c.chat([{"role": "user", "content": "x"}], 10)
        assert c.stats["cached_tokens"] == want


def test_prompt_cache_key_auto_is_sent_only_to_accepting_providers(monkeypatch):
    """C3: prompt_cache_key='auto' derives a stable key and is sent only where
    the provider table says the parameter is accepted (OpenAI, not DeepSeek)."""
    posts = []
    _patch_post(monkeypatch, lambda *a, **k: posts.append(k["json"]) or _ok())
    openai = LLMClient(_cfg(base_url="https://api.openai.com/v1",
                            prompt_cache_key="auto"))
    openai.cfg._root = "/proj/a"
    openai.chat([{"role": "user", "content": "hi"}], 10)
    assert "prompt_cache_key" in posts[-1]
    again = LLMClient(_cfg(base_url="https://api.openai.com/v1",
                           prompt_cache_key="auto"))
    again.cfg._root = "/proj/a"
    again.chat([{"role": "user", "content": "hi"}], 10)
    assert posts[-1]["prompt_cache_key"] == posts[-2]["prompt_cache_key"]
    posts.clear()
    ds = LLMClient(_cfg(base_url="https://api.deepseek.com/v1",
                        prompt_cache_key="auto"))
    ds.cfg._root = "/proj/a"
    ds.chat([{"role": "user", "content": "hi"}], 10)
    assert "prompt_cache_key" not in posts[-1]


@pytest.mark.parametrize("usage,want_read,want_write", [
    # DeepSeek: implicit cache hit/miss tokens at the top level.
    ({"prompt_tokens": 50, "completion_tokens": 1,
      "prompt_cache_hit_tokens": 30, "prompt_cache_miss_tokens": 20}, 30, 0),
    # OpenAI: cached_tokens inside prompt_tokens_details.
    ({"prompt_tokens": 50, "completion_tokens": 1,
      "prompt_tokens_details": {"cached_tokens": 30}}, 30, 0),
    # Anthropic-style: cache_read_input_tokens + cache_creation_input_tokens.
    ({"prompt_tokens": 50, "completion_tokens": 1,
      "cache_read_input_tokens": 30, "cache_creation_input_tokens": 15},
     30, 15),
    # DashScope-style: creation inside prompt_tokens_details.
    ({"prompt_tokens": 50, "completion_tokens": 1,
      "prompt_tokens_details": {"cache_creation_input_tokens": 15},
      "cache_read_input_tokens": 30}, 30, 15),
    # The generic cache_write_tokens spelling.
    ({"prompt_tokens": 50, "completion_tokens": 1,
      "cache_write_tokens": 15, "cached_tokens": 30}, 30, 15),
])
def test_cache_read_and_write_fields_of_each_provider_shape(
        monkeypatch, usage, want_read, want_write):
    _patch_post(monkeypatch, lambda *a, **k: _ok("x", usage))
    c = LLMClient(_cfg())
    c.chat([{"role": "user", "content": "x"}], 10)
    assert c.stats["cached_tokens"] == want_read
    assert c.stats["cache_write_tokens"] == want_write


def test_cache_writes_are_charged_at_the_write_price(monkeypatch):
    usage = {"prompt_tokens": 100, "completion_tokens": 10,
             "prompt_tokens_details": {
                 "cached_tokens": 30, "cache_creation_input_tokens": 40}}
    _patch_post(monkeypatch, lambda *a, **k: _ok("x", usage))
    client = LLMClient(_cfg(price_input_per_mtok=1.0, price_output_per_mtok=2.0,
                            price_cached_per_mtok=0.1,
                            price_cache_write_per_mtok=1.25))
    client.site = "answer"
    client.chat([{"role": "user", "content": "x"}], 10)
    s = client.stats
    assert s["cached_tokens"] == 30 and s["cache_write_tokens"] == 40
    # fresh = 100 - 30 - 40 = 30; read at 0.1, write at 1.25, out at 2.0.
    expected = (30 * 1.0 + 30 * 0.1 + 40 * 1.25 + 10 * 2.0) / 1e6
    assert s["cost"] == pytest.approx(expected)
    assert s["by_site"]["answer"]["cache_write_tokens"] == 40


def test_cache_writes_fall_back_to_the_input_price_when_unset(monkeypatch):
    usage = {"prompt_tokens": 100, "completion_tokens": 0,
             "cache_creation_input_tokens": 100}
    _patch_post(monkeypatch, lambda *a, **k: _ok("x", usage))
    client = LLMClient(_cfg(price_input_per_mtok=1.0))
    client.chat([{"role": "user", "content": "x"}], 10)
    # No write price: the write tokens are charged as plain input.
    assert client.stats["cost"] == pytest.approx(100 * 1.0 / 1e6)


def test_a_server_rejecting_stream_options_is_retried_without_them(monkeypatch):
    bodies = []

    def fake(*a, **k):
        bodies.append(dict(k["json"]))
        if "stream_options" in k["json"]:
            return Resp(400, text="unknown field: stream_options")
        return Resp(200, lines=[_chunk("ok"), _chunk(finish="stop"), "data: [DONE]"])

    _patch_post(monkeypatch, fake)
    client = LLMClient(_cfg())
    assert client.chat([{"role": "user", "content": "x"}], 10, stream=True) == "ok"
    assert "stream_options" in bodies[0] and "stream_options" not in bodies[1]
    assert client._stream_usage_ok is False
    client.chat([{"role": "user", "content": "x"}], 10, stream=True)
    assert "stream_options" not in bodies[-1]            # remembered


def test_multiline_and_comment_sse_frames_are_tolerated(monkeypatch):
    lines = [": keep-alive", "", _chunk("a"), "data:" + json.dumps(
        {"choices": [{"delta": {"content": "b"}, "finish_reason": None}]}),
        "event: ping", "data: {broken json", _chunk(finish="stop"), "data: [DONE]"]
    _patch_post(monkeypatch, lambda *a, **k: Resp(200, lines=lines))
    assert LLMClient(_cfg()).chat([{"role": "user", "content": "x"}], 10,
                                  stream=True) == "ab"


def test_cancel_stops_a_stream_and_keeps_partial_text(monkeypatch):
    lines = [_chunk("one "), _chunk("two "), _chunk("three"), "data: [DONE]"]
    resp = Resp(200, lines=lines)
    _patch_post(monkeypatch, lambda *a, **k: resp)
    client = LLMClient(_cfg())
    client.stream_callback = lambda d: client.cancel() if d == "two " else None
    with pytest.raises(CancelledError):
        client.chat([{"role": "user", "content": "x"}], 10, stream=True)
    assert client.partial_text == "one two"
    assert resp.closed


def test_dropped_stream_keeps_the_partial_reply_as_truncated(monkeypatch, capsys):
    err = requests.exceptions.ChunkedEncodingError("connection broken")
    resp = Resp(200, lines=[_chunk("partial answer")], raise_after=err)
    _patch_post(monkeypatch, lambda *a, **k: resp)
    client = LLMClient(_cfg())
    assert client.chat([{"role": "user", "content": "x"}], 10, stream=True) \
        == "partial answer"
    assert client.stream_truncated is True               # continuation machinery engages
    # with nothing received at all it is an error
    resp2 = Resp(200, lines=[], raise_after=err)
    _patch_post(monkeypatch, lambda *a, **k: resp2)
    monkeypatch.setattr("pedacito.llm.time.sleep", lambda s: None)
    with pytest.raises(LLMError):
        LLMClient(_cfg()).chat([{"role": "user", "content": "x"}], 10, stream=True)


# --------------------------------------------------------------------- budgets
def test_token_budget_stops_before_the_next_call(monkeypatch):
    _patch_post(monkeypatch, lambda *a, **k: _ok("x", {"prompt_tokens": 600,
                                                       "completion_tokens": 500}))
    client = LLMClient(_cfg(max_total_tokens=1000))
    client.chat([{"role": "user", "content": "x"}], 10)
    with pytest.raises(BudgetExceeded) as e:
        client.chat([{"role": "user", "content": "x"}], 10)
    assert "Token budget spent" in str(e.value)


def test_cost_budget_stops_before_the_next_call(monkeypatch):
    _patch_post(monkeypatch, lambda *a, **k: _ok("x", {"prompt_tokens": 1_000_000,
                                                       "completion_tokens": 0}))
    client = LLMClient(_cfg(max_cost=0.5, price_input_per_mtok=1.0))
    client.chat([{"role": "user", "content": "x"}], 10)
    assert client.stats["cost"] == pytest.approx(1.0)
    with pytest.raises(BudgetExceeded):
        client.chat([{"role": "user", "content": "x"}], 10)


def test_servers_that_report_no_usage_still_get_estimates(monkeypatch):
    _patch_post(monkeypatch, lambda *a, **k: _ok("y" * 360))
    client = LLMClient(_cfg())
    client.chat([{"role": "user", "content": "x" * 360}], 10)
    assert client.stats["prompt_tokens"] > 90 and client.stats["completion_tokens"] > 90


def test_temperature_override_applies_only_inside_the_context(monkeypatch):
    posts = []
    _patch_post(monkeypatch, lambda *a, **k: posts.append(k["json"]) or _ok())
    client = LLMClient(_cfg(temperature=0.2))
    with client.temperature(0.9):
        client.chat([{"role": "user", "content": "x"}], 10)
    client.chat([{"role": "user", "content": "x"}], 10)
    assert [p["temperature"] for p in posts] == [0.9, 0.2]


def _tool_resp(args: dict, finish="tool_calls"):
    return Resp(200, {"choices": [{"message": {
        "content": "", "tool_calls": [{"id": "1", "type": "function", "function": {
            "name": "step", "arguments": json.dumps(args)}}]},
        "finish_reason": finish}], "usage": {}})


def test_native_mode_sends_a_forced_tool_and_reads_its_arguments(monkeypatch):
    from pedacito.tools import schema_for
    seen = {}

    def fake(url, **kw):
        seen.update(kw["json"])
        return _tool_resp({"action": "grep", "pattern": "x", "reasoning": "r"})

    _patch_post(monkeypatch, fake)
    c = LLMClient(_cfg(tool_mode="native"))
    out = c.chat_json([{"role": "user", "content": "go"}],
                      schema_for(c.cfg), 200)
    assert out["action"] == "grep"
    assert seen["tools"][0]["function"]["name"] == "step"
    assert seen["tool_choice"]["function"]["name"] == "step"
    assert "response_format" not in seen


def test_native_mode_is_off_for_other_schemas_and_by_default(monkeypatch):
    seen = {}

    def fake(url, **kw):
        seen.clear()
        seen.update(kw["json"])
        return _ok('{"title": "t"}')

    _patch_post(monkeypatch, fake)
    c = LLMClient(_cfg(tool_mode="native"))
    c.chat_json([{"role": "user", "content": "go"}],
                {"type": "object", "properties": {"title": {"type": "string"}}}, 100)
    assert "tools" not in seen and "response_format" in seen
    c2 = LLMClient(_cfg())
    from pedacito.tools import schema_for
    c2.chat_json([{"role": "user", "content": "go"}], schema_for(c2.cfg), 100)
    assert "tools" not in seen


def test_native_mode_falls_back_when_the_server_rejects_tools(monkeypatch):
    from pedacito.tools import schema_for
    calls = []

    def fake(url, **kw):
        calls.append("tools" in kw["json"])
        if "tools" in kw["json"]:
            return Resp(400, text='{"error": "tools are not supported"}')
        return _ok('{"action": "done", "reasoning": "r"}')

    _patch_post(monkeypatch, fake)
    c = LLMClient(_cfg(tool_mode="native"))
    out = c.chat_json([{"role": "user", "content": "go"}], schema_for(c.cfg), 100)
    assert out["action"] == "done" and calls == [True, False]
    c.chat_json([{"role": "user", "content": "go"}], schema_for(c.cfg), 100)
    assert calls == [True, False, False]          # remembered


def test_calibrate_picks_native_only_when_it_measures_better():
    from pedacito import modelcard
    base = {"tool_choice": 0.6, "json_ok": 1.0, "edit_format": 1, "instructions": 1,
            "honesty": 1, "needle": 1, "continuation": 1}
    assert modelcard.choose({**base, "tool_choice_native": 0.9,
                             "json_ok_native": 1.0}, False)["tool_mode"] == "native"
    assert modelcard.choose({**base, "tool_choice_native": 0.65,
                             "json_ok_native": 1.0}, False)["tool_mode"] == "grammar"
    assert modelcard.choose({**base, "tool_choice_native": 0.9,
                             "json_ok_native": 0.8}, False)["tool_mode"] == "grammar"
    assert modelcard.choose(dict(base), False)["tool_mode"] == "grammar"
