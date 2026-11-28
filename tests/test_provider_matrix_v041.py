"""WP-4106B: provider registry, capability matrix, probes and reasoning state.

All tests are offline and deterministic: the one "network" test serves a fake
OpenAI-compatible endpoint on loopback and asserts the probe contract.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import FrozenInstanceError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import config as config_mod
from pedacito import discovery, llm as llm_mod, model_catalog
from pedacito import providers
from pedacito.config import Config
from pedacito.gui import settings_api
from pedacito.llm import LLMClient
from pedacito import transport_anthropic, transport_gemini, transport_openai


# --------------------------------------------------------------------- helpers

class Resp:
    def __init__(self, status=200, body=None, text="", lines=None,
                 headers=None):
        self.status_code = status
        self._body = body
        self.text = text or (json.dumps(body) if body is not None else "")
        self._lines = lines or []
        self.headers = headers or {}

    def json(self):
        return self._body

    def iter_lines(self, decode_unicode=False):
        yield from self._lines

    def close(self):
        pass


def _cfg(**kw):
    cfg = Config()
    cfg.base_url = "http://127.0.0.1:1/v1"
    cfg.model = "m"
    for key, value in kw.items():
        setattr(cfg, key, value)
    return cfg


@pytest.fixture
def cache_home(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "USER_CONFIG", tmp_path / "config.toml")
    providers.clear_capability_cache()
    yield tmp_path
    providers.clear_capability_cache()


@pytest.fixture
def fake_openai():
    state = {"calls": [], "payload": None, "status": 200}
    state["payload"] = {
        "choices": [{
            "message": {
                "content": '{"ok": true}',
                "reasoning_content": "hidden probe reasoning",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "pedacito_probe",
                                 "arguments": '{"ok": true}'},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            state["calls"].append({
                "path": self.path,
                "auth": self.headers.get("Authorization", ""),
                "x_api_key": self.headers.get("x-api-key", ""),
                "google_key": self.headers.get("x-goog-api-key", ""),
                "body": json.loads(raw.decode("utf-8") or "{}"),
            })
            if self.path.endswith("/messages"):
                payload = {
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            elif ":generateContent" in self.path:
                payload = {
                    "candidates": [{
                        "content": {"parts": [{"text": "ok"}]},
                        "finishReason": "STOP",
                    }],
                    "usageMetadata": {"promptTokenCount": 1,
                                      "candidatesTokenCount": 1},
                }
            else:
                payload = state["payload"]
            body = json.dumps(payload).encode("utf-8")
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# --------------------------------------------------------------------- matrix

def test_capabilities_default_to_unknown_and_are_frozen():
    caps = providers.ProviderCapabilities()
    for name in ("native_tools", "parallel_tools", "json_schema",
                 "reasoning_blocks", "vision", "cache_mode", "context_tokens",
                 "output_tokens", "idempotency", "stream_resume"):
        assert getattr(caps, name) is None, name
    assert caps.probed_at == 0.0 and caps.source == "unverified"
    with pytest.raises(FrozenInstanceError):
        caps.native_tools = True


def test_provider_additive_fields_are_defaulted():
    bare = providers.Provider(cache_mode="implicit")
    assert bare.id == "" and bare.transport == "openai"
    assert bare.auth_kind == "bearer" and bare.base_url_template == ""
    assert isinstance(bare.capabilities, providers.ProviderCapabilities)
    assert bare.capabilities.source == "unverified"


def test_catalog_has_twenty_hosted_and_five_local():
    hosted = providers.hosted_provider_ids()
    local = providers.local_runtime_ids()
    assert len(hosted) >= 20
    assert len(local) >= 5
    assert len(set(hosted)) == len(hosted)
    assert len(set(local)) == len(local)
    for name in ("groq", "cerebras", "fireworks", "together", "xai",
                 "mistral", "moonshot", "minimax", "zai", "nvidia",
                 "huggingface", "litellm"):
        assert name in hosted, name
    for name in ("mlx-lm", "sglang", "tabbyapi", "koboldcpp", "llamacpp"):
        assert name in local, name


def test_gateway_only_catalog_entries_do_not_claim_native_support():
    ids = providers.gateway_provider_ids()
    assert {"azure-openai", "aws-bedrock", "google-vertex"} <= set(ids)
    for provider_id in ids:
        entry = providers.provider_by_id(provider_id)
        assert entry is not None and entry.gateway_only
        assert entry.transport == "gateway"
        assert entry.capabilities.native_tools is None
        assert entry.capabilities.json_schema is None
        assert "requires configured OpenAI-compatible gateway" \
            in entry.capabilities.source
        assert provider_id not in providers.hosted_provider_ids()
        assert provider_id in providers.hosted_provider_ids(
            include_gateway=True)


def test_host_matching_is_exact_not_substring():
    assert providers.provider_for("https://api.openai.com/v1").id == "openai"
    assert providers.provider_for(
        "https://api.deepseek.com/v1").id == "deepseek"
    assert providers.provider_for(
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1").id == "qwen"
    assert providers.provider_for(
        "https://api.openai.com.attacker.test/v1").id == ""
    assert providers.provider_for(
        "https://notapi.openai.com/v1").id == ""
    assert providers.provider_for(
        "https://api.openai.com@evil.test/v1").id == ""
    assert providers.host_matches("api.openai.com", "api.openai.com")
    assert not providers.host_matches("api.openai.com.evil.test",
                                       "api.openai.com")
    assert not providers.host_matches("notapi.openai.com", "api.openai.com")
    assert providers.host_matches("foo.openai.azure.com",
                                  ".openai.azure.com")
    assert not providers.host_matches("fooopenai.azure.com",
                                      ".openai.azure.com")


def test_local_runtime_labels_are_exact_and_do_not_upgrade_lookalikes():
    assert providers.is_local_runtime_label("ollama")
    assert providers.is_local_runtime_label("LLAMA_CPP")
    assert providers.is_local_runtime_label("LlamaCpp")
    assert providers.local_runtime_for_label("LM Studio").id == "lmstudio"
    assert providers.local_runtime_for_label("ollama-proxy") is None
    assert providers.local_runtime_for_label("local-homelab") is not None
    assert config_mod._endpoint_is_local(
        "https://api.example.test/v1", "local-homelab") is True
    assert config_mod._endpoint_is_local(
        "https://api.example.test/v1", "ollama-proxy") is False
    assert config_mod._endpoint_is_local(
        "https://api.example.test/v1", "openai") is False
    # The legacy explicit-label contract is preserved for exact names.
    assert config_mod._endpoint_is_local(
        "https://api.example.test/v1", "ollama") is True
    # A loopback URL is local regardless of the provider label.
    assert config_mod._endpoint_is_local("http://127.0.0.1:1234/v1", "openai")


def test_model_catalog_uses_the_same_host_boundaries():
    pids = model_catalog.provider_ids_for_base_url(
        "https://api.openai.com/v1")
    assert pids[0] == "openai"
    lookalike = model_catalog.provider_ids_for_base_url(
        "https://api.openai.com.attacker.test/v1")
    assert "openai" not in lookalike
    assert lookalike == ("api.openai.com.attacker.test",)
    assert "groq" in model_catalog.provider_ids_for_base_url(
        "https://api.groq.com/openai/v1")


def test_discovery_runtime_ids_come_from_the_same_registry():
    assert discovery.runtime_preset_id("LM Studio") == "lmstudio"
    assert discovery.runtime_preset_id("llama.cpp") == "llamacpp"
    assert discovery.runtime_preset_id("SGLang") == "sglang"
    assert discovery.runtime_preset_id("unknown-runtime") == ""


def test_settings_api_uses_the_registry_presets():
    assert settings_api.CLOUD_PROVIDERS == providers.provider_presets()
    assert [p["id"] for p in settings_api.CLOUD_PROVIDERS] == [
        "anthropic", "openai", "deepseek", "qwen", "gemini", "openrouter",
        "strata"]
    for preset in settings_api.CLOUD_PROVIDERS:
        assert "api_key" not in preset
        assert preset["api_key_env"].endswith("_API_KEY")


# ---------------------------------------------------------------- capabilities

def test_get_capabilities_never_probes_without_consent(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("probe must not run without explicit consent")

    monkeypatch.setattr(providers, "probe_capabilities", boom)
    caps = providers.get_capabilities(
        "https://api.openai.com/v1", "gpt-5.4")
    assert caps.native_tools is True
    assert caps.json_schema is True
    assert caps.cache_mode == "implicit"
    assert caps.source.startswith("catalog")
    unknown = providers.get_capabilities(
        "https://unknown.example.test/v1", "m")
    assert unknown.native_tools is None and unknown.source == "unverified"


def test_runtime_refusals_are_merged_but_never_upgrade_unknown(monkeypatch,
                                                               cache_home):
    monkeypatch.setattr(providers, "_model_catalog_metadata",
                        lambda base, model: {})
    base = "https://api.openai.com/v1"
    model = "runtime-refusal-model"
    llm_mod._JSON_CAPS.clear()
    llm_mod._ACCEL_CAPS.clear()
    try:
        llm_mod._deny_json_cap(base, model, "schema")
        llm_mod._deny_accel_cap(base, model, "cache")
        caps = providers.get_capabilities(base, model)
        assert caps.json_schema is False
        assert "runtime" in caps.source
        assert caps.native_tools is True   # catalog evidence still stands
    finally:
        llm_mod._JSON_CAPS.clear()
        llm_mod._ACCEL_CAPS.clear()


def test_probe_is_bounded_cached_and_never_stores_the_secret(
        fake_openai, cache_home):
    url, state = fake_openai
    cfg = _cfg(base_url=url + "/v1", model="probe-model",
               provider="openai", api_key="sk-super-secret")
    caps = providers.probe_capabilities(
        url + "/v1", "probe-model", cfg=cfg,
        deadline=time.monotonic() + 10.0)
    assert caps.probed_at > 0
    assert caps.source == "probe"
    assert caps.json_schema is True
    assert caps.native_tools is True
    assert caps.reasoning_blocks is True
    assert len(state["calls"]) == 1
    call = state["calls"][0]
    assert call["path"] == "/v1/chat/completions"
    assert call["auth"] == "Bearer sk-super-secret"
    assert call["body"]["max_tokens"] <= 4
    assert "sk-super-secret" not in json.dumps(call["body"])

    # A second read is served from the 24 h cache; no second HTTP call.
    again = providers.get_capabilities(url + "/v1", "probe-model", cfg=cfg)
    assert again.source == "probe" and again.json_schema is True
    assert len(state["calls"]) == 1

    cache_text = providers.capability_cache_path().read_text(encoding="utf-8")
    assert "sk-super-secret" not in cache_text
    entry = next(iter(json.loads(cache_text)["entries"].values()))
    assert entry["url"] == url
    assert entry["model"] == "probe-model"
    assert entry["auth_hash"]

    # The exact model is part of the key: another model is not served the hit.
    other = providers.get_capabilities(url + "/v1", "other-model", cfg=cfg)
    assert other.source != "probe"


def test_probe_uses_the_native_anthropic_protocol(fake_openai, cache_home):
    url, state = fake_openai
    cfg = _cfg(base_url=url + "/v1", model="claude-probe",
               provider="anthropic", api_key="sk-ant")
    caps = providers.probe_capabilities(
        url + "/v1", "claude-probe", cfg=cfg,
        deadline=time.monotonic() + 10.0)
    assert caps.probed_at > 0 and caps.source == "probe"
    call = state["calls"][0]
    assert call["path"] == "/v1/messages"
    assert call["x_api_key"] == "sk-ant"
    assert call["auth"] == ""
    assert call["body"]["max_tokens"] == 1


def test_probe_uses_the_native_gemini_protocol(fake_openai, cache_home):
    url, state = fake_openai
    cfg = _cfg(base_url=url + "/v1beta", model="gemini-probe",
               provider="gemini", api_key="goog-key")
    caps = providers.probe_capabilities(
        url + "/v1beta", "gemini-probe", cfg=cfg,
        deadline=time.monotonic() + 10.0)
    assert caps.probed_at > 0 and caps.source == "probe"
    call = state["calls"][0]
    assert ":generateContent" in call["path"]
    assert call["google_key"] == "goog-key"
    assert call["body"]["generationConfig"]["maxOutputTokens"] == 1


def test_probe_uses_openai_responses_when_opted_in(fake_openai, cache_home):
    url, state = fake_openai
    cfg = _cfg(base_url=url + "/v1", model="gpt-5-probe",
               provider="openai-responses", api_key="sk-oai")
    caps = providers.probe_capabilities(
        url + "/v1", "gpt-5-probe", cfg=cfg,
        deadline=time.monotonic() + 10.0)
    assert caps.probed_at > 0 and caps.source == "probe"
    call = state["calls"][0]
    assert call["path"] == "/v1/responses"
    assert call["body"]["max_output_tokens"] == 4
    assert call["body"]["stream"] is False


def test_probe_cache_expires_after_24_hours(fake_openai, cache_home):
    url, _state = fake_openai
    cfg = _cfg(base_url=url + "/v1", model="ttl-model", provider="openai",
               api_key="k")
    providers.probe_capabilities(url + "/v1", "ttl-model", cfg=cfg,
                                 deadline=time.monotonic() + 10.0)
    path = providers.capability_cache_path()
    data = json.loads(path.read_text(encoding="utf-8"))
    key = next(iter(data["entries"]))
    data["entries"][key]["probed_at"] = time.time() - 25 * 3600
    path.write_text(json.dumps(data), encoding="utf-8")
    providers.clear_capability_cache()
    stale = providers.get_capabilities(url + "/v1", "ttl-model", cfg=cfg)
    assert stale.source != "probe"


def test_probe_refuses_redirects_and_errors(fake_openai, cache_home):
    url, state = fake_openai
    state["status"] = 302
    cfg = _cfg(base_url=url + "/v1", model="redirect-model",
               provider="openai", api_key="k")
    caps = providers.probe_capabilities(
        url + "/v1", "redirect-model", cfg=cfg,
        deadline=time.monotonic() + 10.0)
    assert caps.probed_at == 0.0
    assert "redirect-refused" in caps.source
    state["status"] = 401
    caps = providers.probe_capabilities(
        url + "/v1", "redirect-model", cfg=cfg,
        deadline=time.monotonic() + 10.0)
    assert "auth-failed" in caps.source


def test_probe_deadline_short_circuits(fake_openai, cache_home):
    url, state = fake_openai
    cfg = _cfg(base_url=url + "/v1", model="deadline-model")
    caps = providers.probe_capabilities(
        url + "/v1", "deadline-model", cfg=cfg,
        deadline=time.monotonic() - 1.0)
    assert "deadline" in caps.source
    assert state["calls"] == []


# ------------------------------------------------------- reasoning round-trip

def test_openai_compatible_reasoning_content_survives_tool_turns(monkeypatch):
    posts = []

    def fake_post(url, **kwargs):
        posts.append(dict(kwargs["json"]))
        return Resp(200, {
            "choices": [{
                "message": {
                    "content": "done",
                    "reasoning_content": "new hidden reasoning",
                },
                "finish_reason": "stop",
            }],
            "usage": {},
        })

    monkeypatch.setattr("pedacito.llm.requests.post", fake_post)
    client = LLMClient(_cfg())
    messages = [
        {"role": "assistant", "content": "",
         "reasoning_content": "prior hidden reasoning",
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "step", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        {"role": "user", "content": "continue"},
    ]
    assert client.chat(messages, 32) == "done"
    sent = posts[0]["messages"]
    assert sent[0]["reasoning_content"] == "prior hidden reasoning"
    assert "prior hidden reasoning" not in (sent[0].get("content") or "")
    assert client.last_reasoning_items == [
        {"type": "reasoning_content", "text": "new hidden reasoning"}]


def test_openai_responses_keeps_chat_reasoning_as_opaque_metadata():
    _instructions, items = transport_openai.translate_messages([{
        "role": "assistant",
        "content": "",
        "reasoning_content": "hidden chat reasoning",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "step", "arguments": "{}"},
        }],
    }])
    reasoning = [item for item in items if item.get("type") == "reasoning"]
    assert reasoning[0]["summary"] == [{
        "type": "summary_text", "text": "hidden chat reasoning"}]
    visible = [item for item in items if item.get("type") == "message"]
    assert not visible
    assert not any("hidden chat reasoning" in str(item.get("content") or "")
                   for item in items)


def test_reasoning_secret_is_dropped_before_external_request():
    secret = "sk-proj-abcdefghijklmnopqrstuvwxyz1234567890"
    cfg = _cfg(base_url="https://api.deepseek.com/v1", provider="openai",
               api_key="k")
    cfg.security_outbound_mode = "redact"
    client = LLMClient(cfg)
    out = client._apply_outbound_guard({"messages": [{
        "role": "assistant",
        "content": "visible answer",
        "reasoning_content": f'api_key = "{secret}"',
    }]})
    kept = out["messages"][0]
    assert kept["content"] == "visible answer"
    assert "reasoning_content" not in kept
    assert secret not in json.dumps(kept)


def test_anthropic_reasoning_blocks_round_trip():
    messages = [{
        "role": "assistant",
        "content": "",
        "reasoning_items": [{
            "type": "thinking",
            "thinking": "opaque thought",
            "signature": "sig-123",
        }],
        "tool_calls": [{
            "id": "toolu_1",
            "type": "function",
            "function": {"name": "step", "arguments": "{}"},
        }],
    }]
    _system, translated = transport_anthropic.translate_messages(messages)
    blocks = translated[0]["content"]
    assert blocks[0] == {"type": "thinking", "thinking": "opaque thought",
                         "signature": "sig-123"}
    assert any(block.get("type") == "tool_use" for block in blocks)
    assert all("opaque thought" not in (block.get("text") or "")
               for block in blocks)

    parsed = transport_anthropic.parse_json_response({
        "content": [
            {"type": "thinking", "thinking": "hidden answer reasoning",
             "signature": "sig-response"},
            {"type": "text", "text": "visible answer"},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 4, "output_tokens": 2},
    })
    assert parsed["text"] == "visible answer"
    assert parsed["reasoning"] == "hidden answer reasoning"
    assert parsed["reasoning_items"][0]["signature"] == "sig-response"


def test_anthropic_stream_preserves_thinking_signature():
    lines = [
        "data: " + json.dumps({
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "thinking", "thinking": ""}}),
        "data: " + json.dumps({
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "hidden"}}),
        "data: " + json.dumps({
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig-evt"}}),
        "data: " + json.dumps({"type": "message_stop"}),
    ]
    parsed = transport_anthropic.parse_sse_stream(Resp(200, lines=lines))
    assert parsed["reasoning"] == "hidden"
    assert parsed["reasoning_items"] == [{
        "type": "thinking", "thinking": "hidden", "signature": "sig-evt"}]
    assert parsed["text"] == ""


def test_gemini_thought_parts_and_signatures_round_trip():
    messages = [{
        "role": "assistant",
        "content": "",
        "reasoning_items": [{
            "text": "opaque gemini thought",
            "thought": True,
            "thoughtSignature": "gem-sig",
        }],
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "step", "arguments": "{}"},
            "thoughtSignature": "gem-call-sig",
        }],
    }]
    _system, contents = transport_gemini.translate_messages(messages)
    parts = contents[0]["parts"]
    assert parts[0] == {"text": "opaque gemini thought", "thought": True,
                        "thoughtSignature": "gem-sig"}
    assert parts[-1]["thoughtSignature"] == "gem-call-sig"
    assert all("opaque gemini thought" not in (part.get("text") or "")
               for part in parts if not part.get("thought"))

    parsed = transport_gemini.parse_response({
        "candidates": [{
            "content": {"parts": [
                {"text": "hidden", "thought": True,
                 "thoughtSignature": "sig-2"},
                {"text": "visible"},
            ]},
            "finishReason": "STOP",
        }],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1},
    })
    assert parsed["text"] == "visible"
    assert parsed["reasoning"] == "hidden"
    assert parsed["reasoning_items"][0]["thoughtSignature"] == "sig-2"
