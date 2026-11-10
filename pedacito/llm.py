"""OpenAI-compatible client: chat completions, JSON-schema tool calls, and
model warmup.

Usage
-----
    from .llm import LLMClient

    client = LLMClient(cfg)
    client.check()                       # verify the server and model
    text = client.chat(messages, max_tokens=500)
    step = client.chat_json(messages, schema, max_tokens=500)

Works against any server exposing an OpenAI-compatible /v1 surface (tested
against LM Studio, Lemonade, and Ollama). The tool loop uses
response_format={"type": "json_schema", ...}, which servers with grammar-
constrained decoding (llama.cpp, lemond, LM Studio, Ollama /v1, vLLM) enforce,
so a small model cannot emit unparseable JSON. `cfg.enforce_json_schema` gates
this engine-level enforcement (on by default); an endpoint that answers HTTP
400 for that mode is downgraded on the spot (json_object, then plain JSON
prompting) and the verdict is cached per (base_url, model), so no call ever
fails because of a server's schema support.

Also handles local-model failure modes: templates that reject certain message
shapes, replies truncated at the token limit, reasoning models that wrap their
answer in a family-specific thinking block, and a model that is still loading.

Only dependency is `requests`.
"""

from __future__ import annotations

import base64
import ipaddress
import json
import os
import random
import re
import sys
import threading
import time
from pathlib import Path
from urllib.parse import quote, urlparse

import requests


class LLMError(RuntimeError):
    """Base class for any failure talking to the model server. `kind` is a
    short taxonomy label used in logs and events."""

    kind = "error"


class ModelLoadError(LLMError):
    """The server could not load the model. Retrying rarely helps."""

    kind = "load"


class TemplateError(LLMError):
    """The model's chat template rejected the message sequence."""

    kind = "template"


class ContextOverflowError(LLMError):
    """The prompt is bigger than the server's context window. `chat()` trims
    the conversation and retries before giving up."""

    kind = "context"


class BudgetExceeded(LLMError):
    """A configured token or cost budget is spent; no further call is made."""

    kind = "budget"


class CancelledError(LLMError):
    """The caller cancelled the in-flight call (`client.cancel()`); partial
    output is kept in `client.partial_text`."""

    kind = "cancelled"


class CircuitOpenError(LLMError):
    """Repeated transport/server failures opened the circuit breaker; callers
    get a fast, actionable error instead of another timeout."""

    kind = "circuit"


class _Transient(LLMError):
    """Worth retrying: model still loading, or the server is busy.

    `rate_limited` separates "provider says slow down" (429, a hosted 503)
    from "local model is still loading": the two have separate wait budgets so
    a hosted 429 never burns the 240 s model-load deadline."""

    kind = "transient"

    def __init__(self, message: str, retry_after: float | None = None,
                 rate_limited: bool = False):
        super().__init__(message)
        self.retry_after = retry_after
        self.rate_limited = rate_limited


# Rate-limit retries (429/hosted 503) before giving up; separate from the
# local model-load wait.
_MAX_RATE_LIMIT_RETRIES = 5

# Mistral-family templates raise a Jinja exception on two user messages in a
# row; Gemma has no system role. The fix is to reshape the request.
_TEMPLATE_HINTS = (
    "jinja", "roles must alternate", "conversation roles",
    "system role not supported", "only user and assistant roles",
)

# LM Studio returns 400 for both "still loading" and "failed to load", so the
# status code alone can't distinguish transient from fatal.
_LOADING = ("is loading", "loading model", "model is not loaded", "please wait")
_LOAD_FAILED = ("failed to load model", "error loading model", "no model loaded")

# Shown on a plain connection failure; the fix differs per server.
CONNECT_HELP = """\
  Cannot reach the server at {url}.
  - Is a model server actually running there?
  - LM Studio: enable "Serve on Local Network" in the Developer tab, or it
    only listens on 127.0.0.1 and a remote machine gets refused.
  - Ollama: it binds to 127.0.0.1 by default. Set OLLAMA_HOST=0.0.0.0 (or the
    machine's address) in its environment before starting it, then restart.
  - If you're on a different machine from the server, check the URL and port
    in your config match where it's actually listening (LM Studio defaults to
    port 1234 and Ollama to 11434, but any port works; `pedacito models scan`
    finds a server wherever it listens)."""

LOAD_HELP = """\
  The server found the model but could not load it. Usual causes, in order:
  1. Not enough free VRAM. Another model is probably still resident --
     unload it in your model server, or lower this model's context length.
  2. The context length is set higher than the card can hold. Try 16384 with
     K/V cache at Q8_0 and flash attention on.
  3. An incomplete or corrupt download. Re-download the model.
  4. A runtime mismatch. Try switching between ROCm and Vulkan for this model.

  Fastest way to tell: load the model by hand in the server's own UI. Whatever
  error it shows there is the real one -- the API only reports that it failed."""


_GIB = 1024 ** 3


def _positive_int(value) -> int | None:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _usage_tokens(usage: dict, *names: str) -> int:
    """The first non-zero integer for any of `names`, looked up at the usage
    payload's top level and then inside `prompt_tokens_details` (providers put
    cache fields in either place)."""
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        details = {}
    for name in names:
        for source in (usage, details):
            n = _positive_int(source.get(name))
            if n:
                return n
    return 0


def _is_remote(base_url: str) -> bool:
    try:
        host = (urlparse(base_url).hostname or "").lower()
    except ValueError:
        return False
    return host not in ("localhost", "127.0.0.1", "0.0.0.0", "::1", "")


def _server_root(base_url: str) -> str:
    """Strip a `/v1` or `/api/v1` suffix from base_url, leaving the server root
    where the native endpoints (`/api/v0/models`, `/api/show`, ...) live."""
    base = base_url.rstrip("/")
    for suffix in ("/api/v1", "/v1"):
        if base.endswith(suffix):
            return base[:-len(suffix)]
    return base


def _api_key(cfg) -> str:
    """The API key for a request: `api_key_env` (the name of an environment
    variable) beats the plain `api_key` string, so a secret never has to be
    written to the config file."""
    env = getattr(cfg, "api_key_env", "") or ""
    if env:
        return os.environ.get(env, "")
    return getattr(cfg, "api_key", "") or ""


# --------------------------------------------------------------- thinking off
# Families spell "stop reasoning" differently. The request-side matrix:
#
#   enable_thinking=false   Qwen3/Qwen3.5, Gemma 4, Nanbeige, LiquidAI LFM3,
#                           GLM, Granite 4.2, most ChatML thinking templates
#   thinking=false          DeepSeek V3.1/V3.2, Granite 3.2, Holo2, Kimi K2.x
#                           (as a chat-template kwarg; Kimi's own API wants a
#                           top-level {"type": "disabled"} object)
#   reasoning_effort=...    Magistral/Mistral ("none" in the local template),
#                           GPT-OSS/OpenAI/Cerebras ("low", or "none" where
#                           the provider supports a true off)
#   thinking_budget=0       Seed-OSS
#   /no_think               Qwen3/SmolLM3 soft switch (prompt suffix)
#
# Local templates ignore kwargs they do not define, so a local request gets the
# union of the template spellings. Hosted APIs are stricter: only a recognised
# provider/family gets a field, and an unknown host gets none (safer than a 400).
_LOCAL_THINKING_OFF = {"enable_thinking": False, "thinking": False}


def _model_name(cfg) -> str:
    """Lower-case model id with spaces/underscores normalised to hyphens, so
    the same family check works for `LFM2_5`, `LFM2.5`, and `LFM2-5`."""
    raw = (getattr(cfg, "model", "") or "").lower()
    return re.sub(r"[\s_]+", "-", raw)


def _host_name(cfg) -> str:
    try:
        return (urlparse(getattr(cfg, "base_url", "")).hostname or "").lower()
    except ValueError:
        return ""


def _has(text: str, *needles: str) -> bool:
    return any(n in text for n in needles)


def _is_qwen_soft(model: str) -> bool:
    """Qwen3/SmolLM3 are the families trained on the /no_think soft switch."""
    return _has(model, "qwen3", "qwen-3", "smollm3", "smollm-3")


def _is_gemma_reasoning(model: str) -> bool:
    """Gemma 4 is the reasoning generation; Gemma 3 and older have no switch."""
    return _has(model, "gemma-4", "gemma4")


def _is_nanbeige(model: str) -> bool:
    return "nanbeige" in model


def _is_liquid_reasoning(model: str) -> bool:
    """LFM3 and LFM2.5-Thinking expose enable_thinking; older LFM2 does not."""
    return _has(model, "lfm3", "lfm-3", "lfm2.5", "lfm-2.5", "lfm2-5")


def _is_glm_reasoning(model: str) -> bool:
    return _has(model, "glm-4.5", "glm4.5", "glm-4.6", "glm4.6",
                "glm-4.7", "glm4.7", "glm-5", "glm5")


def _is_granite_reasoning(model: str) -> bool:
    return _has(model, "granite-3.2", "granite3.2",
                "granite-4", "granite4")


def _is_deepseek(model: str) -> bool:
    return "deepseek" in model


def _is_kimi(model: str) -> bool:
    return _has(model, "kimi", "moonshot")


def _is_kimi_reasoning(model: str) -> bool:
    return _has(model, "kimi-k2", "kimi-k3", "kimi-thinking")


def _is_magistral(model: str) -> bool:
    return _has(model, "magistral", "mistral-small-4", "mistral-small4",
                "mistral-medium-3.5", "mistral-medium3.5")


def _is_seed_oss(model: str) -> bool:
    return _has(model, "seed-oss", "seedoss")


def _is_gpt_oss(model: str) -> bool:
    return _has(model, "gpt-oss", "gptoss")


def _is_effort_model(model: str) -> bool:
    """Hosted models with a reasoning_effort control (not every GPT-4.x)."""
    return _has(model, "gpt-5", "gpt5", "gpt-6", "gpt6",
                "o1", "o3", "o4", "codex")


def _always_thinks(model: str) -> bool:
    """Families with no off switch. Hosted Kimi K2.7/K3 reject a disabled
    thinking object; LFM2.5-24B-A2B and GPT-OSS still reason at low effort."""
    return _has(model, "kimi-k2.7", "kimi-k3", "kimi-k2-thinking",
                "lfm2.5-24b-a2b", "gpt-oss", "gptoss")


def thinking_suffix_for(cfg) -> str:
    """Prompt suffix for a model whose soft switch we know (Qwen3/SmolLM3),
    or the user's explicit `thinking_suffix`. Empty when neither applies."""
    explicit = getattr(cfg, "thinking_suffix", "")
    if explicit:
        return explicit
    if _is_qwen_soft(_model_name(cfg)):
        return "/no_think"
    return ""


def _local_thinking_off(cfg) -> tuple[dict, dict]:
    model = _model_name(cfg)
    kwargs = dict(_LOCAL_THINKING_OFF)
    top = {"think": False}
    if _is_magistral(model):
        kwargs["reasoning_effort"] = "none"
    elif _is_seed_oss(model):
        kwargs["thinking_budget"] = 0
    elif _is_gpt_oss(model):
        top["reasoning_effort"] = "low"
    if _is_kimi(model) and _always_thinks(model):
        # K2.7/K3 always reason; a false `thinking` kwarg is wrong for them.
        kwargs.pop("thinking", None)
    return kwargs, top


def _hosted_thinking_off(cfg) -> tuple[dict, dict]:
    model, host = _model_name(cfg), _host_name(cfg)
    if _is_kimi(model) or "moonshot" in host:
        # K2.5/K2.6 accept {"type": "disabled"}; K2.7/K3 always think and
        # reject a disabled object. Kimi K1.x has no thinking switch.
        if not _is_kimi_reasoning(model) or _always_thinks(model):
            return {}, {}
        return {}, {"thinking": {"type": "disabled"}}
    if _is_gpt_oss(model):
        return {}, {"reasoning_effort": "low"}      # no true off
    if _is_deepseek(model) or "deepseek" in host:
        if "api.deepseek.com" in host:
            return {}, {}       # the model name chooses the mode
        return {"thinking": False, "enable_thinking": False}, {}
    if "cerebras" in host:
        # Cerebras exposes reasoning_effort rather than template kwargs.
        effort = "low" if _is_gpt_oss(model) else "none"
        if _is_effort_model(model) or _is_qwen_soft(model) \
                or _is_gemma_reasoning(model) or _is_gpt_oss(model):
            return {}, {"reasoning_effort": effort}
    if ("dashscope" in host or "aliyuncs" in host) and _is_qwen_soft(model):
        return {}, {"enable_thinking": False}      # DashScope's spelling
    if "openai" in host and _is_effort_model(model):
        return {}, {"reasoning_effort": "low"}
    if _is_qwen_soft(model) or _is_gemma_reasoning(model) \
            or _is_nanbeige(model) or _is_liquid_reasoning(model) \
            or _is_glm_reasoning(model) or _is_granite_reasoning(model):
        return {"enable_thinking": False}, {}
    return {}, {}


def thinking_off_fields(cfg) -> tuple[dict, dict]:
    """(chat_template_kwargs, top-level fields) asking for non-thinking mode.

    Local servers get every template spelling; hosted APIs get only the field
    their provider/family is known to accept. An unknown hosted model gets
    nothing, because a strict API would 400 on an unknown field."""
    if not _is_remote(getattr(cfg, "base_url", "")):
        return _local_thinking_off(cfg)
    return _hosted_thinking_off(cfg)


# A constrained-JSON verdict costs one round-trip to probe; remembering it per
# server+model keeps repeat runs consistent and skips the probe.
_MODE_CACHE_TTL = 7 * 24 * 3600


def _mode_cache_key(base_url: str, model: str) -> str:
    return f"{base_url.rstrip('/')}|{model}"


def _load_mode_cache(base_url: str, model: str) -> bool | None:
    """Cached `_schema_supported` verdict, or None when absent/stale/corrupt."""
    try:
        from . import config
        path = config.USER_CONFIG.parent / "server_modes.json"
        entry = json.loads(path.read_text(encoding="utf-8")).get(
            _mode_cache_key(base_url, model))
        if isinstance(entry, dict) and \
                isinstance(entry.get("schema_supported"), bool) and \
                time.time() - float(entry.get("ts", 0)) < _MODE_CACHE_TTL:
            return entry["schema_supported"]
    except Exception:
        pass
    return None


def _save_mode_cache(base_url: str, model: str, supported: bool) -> None:
    """Persist a probed verdict; an unwritable config dir never kills a run."""
    try:
        from . import config
        path = config.USER_CONFIG.parent / "server_modes.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}
        data[_mode_cache_key(base_url, model)] = {
            "schema_supported": supported, "ts": time.time()}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    except OSError:
        pass


# Sprint Q: engine-level JSON enforcement needs to know, per endpoint, which
# structured-output tiers that endpoint actually accepts. A 400 naming
# response_format flips the tier off for (base_url, model) for the rest of
# the process, so the retry and every later call start at the highest tier
# the server can honour: json_schema -> json_object -> plain prompt.
# (server_modes.json above remembers the probed grammar verdict across runs;
# this cache learns it mid-run, from live traffic, without any probe.)
_JSON_CAPS: dict[tuple[str, str], dict[str, bool]] = {}
_JSON_CAPS_LOCK = threading.Lock()


def _json_cap_key(base_url: str, model: str) -> tuple[str, str]:
    return (base_url.rstrip("/"), model)


def _json_caps(base_url: str, model: str) -> dict[str, bool]:
    """The mutable {schema: bool, object: bool} tier table for an endpoint."""
    with _JSON_CAPS_LOCK:
        return _JSON_CAPS.setdefault(
            _json_cap_key(base_url, model), {"schema": True, "object": True})


def _deny_json_cap(base_url: str, model: str, tier: str) -> None:
    """Remember that this endpoint refused one structured-output tier."""
    with _JSON_CAPS_LOCK:
        _JSON_CAPS.setdefault(
            _json_cap_key(base_url, model),
            {"schema": True, "object": True})[tier] = False


# Sprint R: local-inference payload acceleration. `cache_prompt` tells a
# llama.cpp-style server to keep the prompt's KV prefix resident between the
# step calls that replay an almost identical long gathering context, and
# `prompt_lookup_num_tokens` switches on n-gram speculative decoding. Both are
# top-level extension keys that only self-hosted endpoints understand; a 400/422
# arriving with one of them attached flips that capability off for
# (base_url, model) for the rest of the process and the request is re-sent
# without it (same learn-from-live-traffic shape as _JSON_CAPS above).
_ACCEL_CAPS: dict[tuple[str, str], dict[str, bool]] = {}
_ACCEL_CAPS_LOCK = threading.Lock()

# The two extension keys, and the capability name each one maps to.
_ACCEL_CACHE_KEY = "cache_prompt"
_ACCEL_LOOKUP_KEY = "prompt_lookup_num_tokens"


def _accel_cap_key(base_url: str, model: str) -> tuple[str, str]:
    return (base_url.rstrip("/"), model)


def _accel_caps(base_url: str, model: str) -> dict[str, bool]:
    """The mutable {cache: bool, lookup: bool} capability table for an endpoint."""
    with _ACCEL_CAPS_LOCK:
        return _ACCEL_CAPS.setdefault(
            _accel_cap_key(base_url, model), {"cache": True, "lookup": True})


def _deny_accel_cap(base_url: str, model: str, cap: str) -> None:
    """Remember that this endpoint refused one acceleration key."""
    with _ACCEL_CAPS_LOCK:
        _ACCEL_CAPS.setdefault(
            _accel_cap_key(base_url, model),
            {"cache": True, "lookup": True})[cap] = False


# Real self-host ranges: RFC1918 plus carrier-grade NAT (a homelab behind a
# CGNAT gateway is still the user's box), and IPv6 unique-local addresses.
_LAN_V4_NETS = tuple(ipaddress.ip_network(n) for n in
                    ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
                     "100.64.0.0/10"))
_LAN_V6_NETS = (ipaddress.ip_network("fc00::/7"),)


def _is_local_server(cfg) -> bool:
    """True for endpoints the user self-hosts, where the acceleration keys make
    sense: loopback or private-LAN addresses (the inverse of `_is_remote`, which
    only knows the loopback spellings, plus RFC1918/carve-out ranges), or a
    provider table entry that names a local runtime."""
    try:
        host = (urlparse(cfg.base_url).hostname or "").lower()
    except ValueError:
        return False
    if host.endswith(".local"):        # mDNS names, the usual LAN spelling
        return True
    if host in ("localhost", "127.0.0.1", "0.0.0.0", "::1", ""):
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        # Real self-host ranges: RFC1918 plus carrier-grade NAT (a homelab
        # behind a CGNAT gateway is still the user's box), loopback spellings,
        # link-local, and IPv6 unique-local. `is_private` alone is too broad:
        # the IANA carve-outs (203.0.113/24, 198.51.100/24, 192.0.2/24, ...)
        # read as "private" to the stdlib but route nowhere a user self-hosts.
        nets = (_LAN_V4_NETS if addr.version == 4 else _LAN_V6_NETS)
        if addr.is_loopback or addr.is_link_local \
                or any(addr in net for net in nets):
            return True
        # A numeric address outside every LAN range is only local if the
        # user's provider entry says so: choosing "llamacpp" for a tunneled
        # box is an explicit statement that the endpoint is self-hosted.
    name = str(getattr(cfg, "provider", "") or "").lower().replace("_", "")
    return any(w in name for w in ("local", "llamacpp", "llama.cpp", "lemond",
                                   "vllm", "lmstudio", "ollama"))


def _schema_example(schema: dict) -> str:
    """One well-formed dummy object for `schema`: the few-shot example shown
    when structured output degrades to plain JSON prompting."""
    ex = {}
    for k, p in (schema or {}).get("properties", {}).items():
        t = p.get("type")
        if p.get("enum"):
            ex[k] = p["enum"][0]
        elif t in ("integer", "number"):
            ex[k] = 0
        elif t == "boolean":
            ex[k] = False
        elif t == "array":
            ex[k] = []
        elif t == "object":
            ex[k] = {}
        else:
            ex[k] = ""
    return json.dumps(ex)


def schema_example_with(schema: dict, **overrides) -> str:
    """A well-formed object for `schema` with some fields filled in (used to
    build worked examples from the real project for servers without grammar
    support)."""
    ex = json.loads(_schema_example(schema))
    for k, v in overrides.items():
        if k in ex:
            ex[k] = v
    return json.dumps(ex)


# Tiny step-shaped schema for `config --doctor`'s structured-output probes.
_PROBE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["read", "done"]},
        "file": {"type": "string"},
        "reasoning": {"type": "string", "maxLength": 40},
    },
    "required": ["action", "file", "reasoning"],
    "additionalProperties": False,
}

# Representative prose+code sample for tokenizer calibration; token density
# differs per model family, so the server's own count beats a fixed 3.6.
_TOKENIZER_SAMPLE = (
    "def gather(self, task, verbose=True):\n"
    '    """Run the tool-calling loop until the model says done."""\n'
    "    budget = self.cfg.max_gathered_chars\n"
    "    while budget > 0:\n"
    "        step = self.client.chat_json(messages, schema, max_tokens=500)\n"
    "        if step.get('action') == 'read':\n"
    "            obs = tool_read(self.index, step['file'], step['start'])\n"
    "            budget -= len(obs)\n"
    "The quick brown fox jumps over the lazy dog. Structured output keeps "
    "tool calls parseable: a small model cannot emit unparseable JSON when "
    "the server constrains decoding with a grammar. Estimates matter too, "
    "because prompt plus reply must fit the context window or the server "
    "silently truncates the middle of the conversation.\n"
)


# Reasoning models emit a thinking block before the answer. Some runtimes move
# it to a separate field, some leave it inline in `content`; an unstripped
# block breaks JSON parsing and lands in the answer text. Families spell the
# block differently, so strip every known delimiter rather than assuming
# Qwen's <think>:
#   <think>...</think>             Qwen3, Phi-4-reasoning, distillations
#   <thinking> / <reasoning>       other ChatML templates
#   [THINK]...[/THINK]             Magistral / Mistral Small 4
#   <seed:think>...</seed:think>   Seed-OSS
#   <|channel|>analysis...         GPT-OSS harmony (handled separately)
#   <|channel>thought...           Gemma 4 (handled separately)
# The content may also arrive in reasoning_content / reasoning / thinking
# fields; extract_content() and the stream collector read those.
_THINK_PAIR_RE = re.compile(
    r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)
# An unterminated block: the model ran out of tokens mid-thought. Drop from
# the opening tag to EOF, keeping any prose before it.
_THINK_UNCLOSED_RE = re.compile(
    r"<(think|thinking|reasoning)>.*", re.DOTALL | re.IGNORECASE)
# Some templates emit only the closing marker when the runtime already routed
# the thinking text to a separate field.
_STRAY_THINK_CLOSE_RE = re.compile(
    r"</(think|thinking|reasoning)>", re.IGNORECASE)

# Magistral uses literal [THINK] special tokens; Seed-OSS uses XML-ish tags
# and may add a self-reflection marker inside the block.
_BRACKET_THINK_PAIR_RE = re.compile(
    r"\[THINK\].*?\[/THINK\]", re.DOTALL | re.IGNORECASE)
_BRACKET_THINK_UNCLOSED_RE = re.compile(
    r"\[THINK\].*", re.DOTALL | re.IGNORECASE)
_BRACKET_THINK_CLOSE_RE = re.compile(r"\[/THINK\]", re.IGNORECASE)
_SEED_THINK_PAIR_RE = re.compile(
    r"<seed:think>.*?</seed:think>", re.DOTALL | re.IGNORECASE)
_SEED_THINK_UNCLOSED_RE = re.compile(
    r"<seed:think>.*", re.DOTALL | re.IGNORECASE)
_SEED_REFLECT_PAIR_RE = re.compile(
    r"<seed:cot_budget_reflect>.*?</seed:cot_budget_reflect>",
    re.DOTALL | re.IGNORECASE)
_SEED_REFLECT_CLOSE_RE = re.compile(
    r"</seed:cot_budget_reflect>", re.IGNORECASE)

# GPT-OSS harmony: <|channel|>analysis<|message|>...<|end|>. Servers with a
# reasoning parser move the analysis channel to reasoning_content; a raw
# llama.cpp stream leaves it in content, so pull out the final channel.
_HARMONY_CHANNEL_RE = re.compile(
    r"<\|channel\|>(analysis|commentary|final)<\|message\|>"
    r"(.*?)(?=<\|end\|>|$)", re.DOTALL | re.IGNORECASE)
_HARMONY_TOKEN_RE = re.compile(r"<\|[^|>]{1,40}\|>")

# Gemma 4's thought channel uses <|channel>thought ... <channel|> (note the
# pipes differ from GPT-OSS harmony). A server without a Gemma parser leaves
# the whole block in content, and even non-thinking 12B/26B/31B models emit an
# empty one.
_GEMMA_THOUGHT_RE = re.compile(
    r"<\|channel>\s*thought\b.*?<channel\|>", re.DOTALL | re.IGNORECASE)
_GEMMA_THOUGHT_OPEN_RE = re.compile(
    r"<\|channel>\s*thought\b.*", re.DOTALL | re.IGNORECASE)
_GEMMA_ROLE_RE = re.compile(
    r"<\|turn>\s*(?:model|assistant|user|system)\s*", re.IGNORECASE)
# Gemma's delimiters are <|turn>/<turn|>, <|channel>/<channel|>, and
# <|think|>; do NOT use a generic <think> pattern here -- that is a separate,
# fully paired reasoning tag handled later.
_GEMMA_TOKEN_RE = re.compile(
    r"<\|(?:turn|channel|think)\|?>"
    r"|<(?:turn|channel)\|>|<eos>|<bos>", re.IGNORECASE)

# Native tool-call syntax in the answer phase means the model ignored "no
# tools here". A few are tolerated (recovery honours up to four reads), but a
# stream that keeps emitting them is a degenerate loop; stop it at six.
_NATIVE_TOOL_OPEN_RE = re.compile(
    r"<(?:tool_call|invoke|read)\b|<function\s*=", re.IGNORECASE)
_MAX_STREAM_TOOL_OPENS = 6


def _strip_harmony(text: str) -> str:
    """Text of the last `final` harmony channel, or "" when the reply contains
    only analysis/commentary. Plain text passes through unchanged."""
    if "<|channel|>" not in text.lower():
        return text
    finals = [m.group(2) for m in _HARMONY_CHANNEL_RE.finditer(text)
              if m.group(1).lower() == "final"]
    if not finals:
        return ""
    return _HARMONY_TOKEN_RE.sub("", finals[-1])


def _strip_gemma4(text: str) -> str:
    """Remove Gemma 4's `<|channel>thought ... <channel|>` block and the
    structural turn/channel tokens a raw server may leave behind."""
    out = _GEMMA_ROLE_RE.sub("", text)
    out = _GEMMA_THOUGHT_RE.sub("", out)
    out = _GEMMA_THOUGHT_OPEN_RE.sub("", out)
    return _GEMMA_TOKEN_RE.sub("", out)


def strip_reasoning(text: str) -> str:
    """Remove reasoning blocks from model output, whatever their family
    spelling. Text that is only an unterminated opening tag (cut mid-thought)
    becomes ""."""
    if not text:
        return ""
    out = _strip_harmony(text)
    out = _strip_gemma4(out)
    out = _THINK_PAIR_RE.sub("", out)
    out = _THINK_UNCLOSED_RE.sub("", out)
    out = _STRAY_THINK_CLOSE_RE.sub("", out)
    out = _BRACKET_THINK_PAIR_RE.sub("", out)
    out = _BRACKET_THINK_UNCLOSED_RE.sub("", out)
    out = _BRACKET_THINK_CLOSE_RE.sub("", out)
    out = _SEED_THINK_PAIR_RE.sub("", out)
    out = _SEED_THINK_UNCLOSED_RE.sub("", out)
    out = _SEED_REFLECT_PAIR_RE.sub("", out)
    out = _SEED_REFLECT_CLOSE_RE.sub("", out)
    return out.strip()


def _reasoning_text(message: dict) -> str:
    """Reasoning text from the separate fields providers use (a string or a
    list of `{text: ...}` parts)."""
    for key in ("reasoning_content", "reasoning", "thinking",
                "analysis", "reasoning_details"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list):
            parts = [p.get("text") for p in value
                     if isinstance(p, dict) and isinstance(p.get("text"), str)]
            if parts:
                return "".join(parts)
    return ""


def extract_content(message: dict) -> str:
    """Answer text of a response message (from `content`, else a reasoning
    field), with any thinking block stripped."""
    content = message.get("content") or ""
    if not content.strip():
        # Some runtimes put everything in a reasoning field; better than none.
        content = _reasoning_text(message)
    return strip_reasoning(content)


def _tool_call_arguments(message: dict) -> str:
    """The JSON arguments of the first native tool call in a response message
    ('' when the model answered in plain content)."""
    for call in message.get("tool_calls") or []:
        fn = (call or {}).get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, dict):
            return json.dumps(args)
        if isinstance(args, str) and args.strip():
            return args
    return ""


def _cut_field(text: str, obj: dict, schema: dict | None) -> str:
    """Which schema field a truncated reply died inside, or "" when benign.

    With `reasoning` last (the default step schema) the last opened key is the
    answer; callers treat "reasoning" as benign and refuse to run steps whose
    payload field was cut. With `reasoning` first, a reply can also die between
    arguments, inside a half-typed key that `_last_key_cut` never sees, so the
    first required argument missing from the salvage is the cut field.
    """
    field = _last_key_cut(text, obj, schema)
    props = (schema or {}).get("properties") or {}
    if field or next(iter(props), "") != "reasoning":
        return field
    required = set((schema or {}).get("required") or [])
    for key in props:
        if key in required and key != "reasoning" and key not in obj:
            return key
    return ""


def _last_key_cut(text: str, obj: dict, schema: dict | None) -> str:
    """The field the LAST opened `"key":` of a truncated reply names, when the
    salvage dropped it whole or kept an unterminated string fragment.

    An unescaped `"key":` in the raw reply is a real key opening, never payload
    text (quotes inside string values are escaped). The LAST opened key is where
    generation stopped; it counts as cut when repair_truncated_json dropped it
    whole (absent from obj) or kept an unterminated string fragment.
    """
    props = sorted((schema or {}).get("properties") or {})
    if not props:
        return ""
    rx = re.compile(r'(?<!\\)"(' + "|".join(re.escape(p) for p in props)
                    + r')"\s*:')
    matches = list(rx.finditer(text))
    if not matches:
        return ""
    last = matches[-1]
    if last.group(1) not in obj:
        return last.group(1)         # dropped whole by the salvage
    # Fragment-kept case: no closing quote before EOF means a partial payload.
    k = last.end()
    while k < len(text) and text[k] in " \t\r\n":
        k += 1
    if k < len(text) and text[k] == '"':
        j = k + 1
        while j < len(text):
            if text[j] == "\\":
                j += 2
                continue
            if text[j] == '"':
                return ""            # value completed; the cut landed later
            j += 1
        return last.group(1)         # unterminated: kept as a fragment
    return ""                        # number/bool present in obj: complete


def repair_truncated_json(text: str) -> str | None:
    """Salvage the complete top-level keys of a JSON object cut off mid-
    generation.

    Everything from the first incomplete token onward (unterminated string,
    bare key, dangling colon/comma, half-typed number) is dropped and the
    surviving pairs are closed into an object; nothing partial is guessed at,
    except that an unterminated string VALUE is kept as a fragment (see
    `_cut_field`). The default step schema puts `action` and its arguments
    before `reasoning`, so a cut inside the trailing explanation still yields a
    usable step; with `reasoning` first, a cut inside it yields no action and
    chat_json re-asks.

    Returns the repaired JSON text, or None if nothing salvageable survived.
    """
    start = text.find("{")
    if start == -1:
        return None
    buf = text[start:]
    n = len(buf)

    def _complete_value_end(i: int) -> int | None:
        """Index just after the value starting at i, or None if it is missing
        or cut off."""
        while i < n and buf[i] in " \t\r\n":
            i += 1
        if i >= n:
            return None
        ch = buf[i]
        if ch == '"':
            j = i + 1
            while j < n:
                if buf[j] == "\\":
                    j += 2
                    continue
                if buf[j] == '"':
                    return j + 1
                j += 1
            return None                      # string never closed
        if ch in "{[":
            depth, k, instr, esc = 0, i, False, False
            while k < n:
                c = buf[k]
                if esc:
                    esc = False
                elif instr:
                    if c == "\\":
                        esc = True
                    elif c == '"':
                        instr = False
                elif c == '"':
                    instr = True
                elif c in "{[":
                    depth += 1
                elif c in "}]":
                    depth -= 1
                    if depth == 0:
                        return k + 1         # nested structure survived whole
                    if depth < 0:
                        return None
                k += 1
            return None                      # container left open
        m = re.match(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?", buf[i:])
        if m:
            tok = m.group(0)
            j = i + len(tok)
            # A digit followed by '.', 'e', or an exponent sign was cut
            # mid-token; '.' or '-' can never legally follow a complete number.
            if j < n and (buf[j] == "." or (tok[-1].isdigit() and buf[j] in "eE+-")):
                return None
            # A bare integer at the very end of the buffer may be unfinished.
            if j == n and tok[-1].isdigit():
                return None
            return j
        for lit in ("true", "false", "null"):
            if buf.startswith(lit, i):
                return i + len(lit)
        return None

    # Keep only key:value pairs that arrived complete; stop at the first
    # fragment, the object's own close, or the end of the buffer.
    kept: list[str] = []
    i = 1                                      # past the opening brace
    while True:
        while i < n and buf[i] in " \t\r\n,":
            i += 1
        if i >= n or buf[i] == "}":
            break
        if buf[i] != '"':
            break                              # malformed; salvage what we have
        kend = _complete_value_end(i)
        if kend is None:
            break                              # dangling key
        j = kend
        while j < n and buf[j] in " \t\r\n":
            j += 1
        if j >= n or buf[j] != ":":
            break                              # key with no colon yet
        vend = _complete_value_end(j + 1)
        if vend is None:
            # An unterminated string value (reply hit max_tokens inside e.g. a
            # 'content' block) is kept as a fragment so callers can salvage
            # partial content; the quote and object are closed below.
            k = j + 1
            while k < n and buf[k] in " \t\r\n":
                k += 1
            if k < n and buf[k] == '"' and _complete_value_end(k) is None:
                # Keep the opening quote; escape only what follows it.
                frag = buf[k + 1:]
                frag = (frag.replace("\\", "\\\\")
                            .replace('"', '\\"').replace("\n", "\\n")
                            .replace("\r", "\\r").replace("\t", "\\t"))
                kept.append(buf[i:k] + '"' + frag + '"')
                break                          # nothing after the cut survives
            break                              # value missing or cut off
        kept.append(buf[i:vend])
        i = vend
    if not kept:
        return None
    return "{" + ",".join(kept) + "}"


def merge_consecutive(messages: list[dict]) -> list[dict]:
    """Collapse adjacent same-role messages into one; a stray repeat would
    otherwise surface as an opaque Jinja error from the server."""
    out: list[dict] = []
    for m in messages:
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"],
                       "content": f"{out[-1]['content']}\n\n{m['content']}"}
        else:
            out.append(dict(m))
    return out


def fold_system(messages: list[dict]) -> list[dict]:
    """Move system content into the first user turn, for templates without a
    system role (Gemma being the common one)."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [dict(m) for m in messages if m["role"] != "system"]
    if not system:
        return rest
    if rest and rest[0]["role"] == "user":
        rest[0]["content"] = f"{system}\n\n{rest[0]['content']}"
    else:
        rest.insert(0, {"role": "user", "content": system})
    return merge_consecutive(rest)


# One density constant for every estimate in the package (fit math, config
# warnings, chunk token guards); a calibrated client overrides it per model.
CHARS_PER_TOKEN = 3.6
# Chat templates wrap every message in role markers; ~4 tokens each plus a
# few for the reply primer. Counting only `len(content)` undercounts a long
# agent conversation by hundreds of tokens.
MESSAGE_OVERHEAD_TOKENS = 4


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~3.6 chars/token for source code)."""
    return int(len(text) / CHARS_PER_TOKEN) + 1


def estimate_messages(messages: list[dict], est=None) -> int:
    """Token estimate for a whole conversation, including per-message framing
    overhead. `est` is a text estimator (a client's calibrated one)."""
    est = est or estimate_tokens
    return sum(est(m.get("content") or "") + MESSAGE_OVERHEAD_TOKENS
               for m in messages) + 3


# Structural request keys a user `sampler` table must never override: they
# would defeat the output cap, the grammar, or the streaming protocol.
_PROTECTED_SAMPLER_KEYS = frozenset({
    "model", "messages", "stream", "max_tokens", "max_completion_tokens",
    "response_format", "tools", "tool_choice", "functions", "function_call",
    "stream_options", "n",
})

# A 400 body that says the PROMPT overflowed the window (as opposed to any
# unrelated 400 that merely contains the word "exceeds").
_CONTEXT_OVERFLOW_RE = re.compile(
    r"context[ _-]?(?:length|window|size)|\bn_ctx\b|ctx_len|too many tokens|"
    r"prompt is too long|input is too long|maximum (?:context|prompt)|"
    r"exceed\w*[^.\n]{0,60}\bcontext\b|\bcontext\b[^.\n]{0,60}exceed|"
    r"reduce the length of the (?:messages|prompt|input)|"
    r"context_length_exceeded|tokens? (?:in|of) (?:the )?(?:prompt|input|request)"
    r"[^.\n]{0,40}(?:exceed|over|more than)",
    re.IGNORECASE)

# Pool one connection set per client; tests (and users) may monkeypatch
# `requests.post`, so a patched function is honoured over the session.
_REAL_POST = requests.post


def _close(response) -> None:
    """Close a response on every path; mocks without close() are fine."""
    try:
        response.close()
    except Exception:
        pass


def _retry_after(response) -> float | None:
    """Seconds from a `Retry-After` header (delta-seconds form), clamped."""
    try:
        raw = (getattr(response, "headers", None) or {}).get("Retry-After")
        if raw is None:
            return None
        return max(0.0, min(float(raw), 120.0))
    except (TypeError, ValueError, AttributeError):
        return None


class LLMClient:
    """A thin client for one model server + model, handling retries, template
    quirks, and reasoning-model output transparently.

    Sampler parameters in `cfg.sampler` are merged into every request body
    verbatim (top_k, min_p, repeat_penalty, seed, ...); the accepted set
    differs per server, so they have no first-class config field.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.url = cfg.base_url.rstrip("/") + "/chat/completions"
        key = _api_key(cfg)
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        # Anthropic authenticates its OpenAI-compatible surface (and its
        # native /v1/messages API) with `x-api-key` + a version stamp. Send
        # both spellings when the server is Anthropic's so either accepted
        # auth path works; every other provider ignores these headers.
        if key and "anthropic.com" in (urlparse(cfg.base_url).hostname or ""):
            self.headers["x-api-key"] = key
            self.headers["anthropic-version"] = "2023-06-01"
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "cached_tokens": 0, "cache_write_tokens": 0,
                      "seconds": 0.0, "cost": 0.0, "by_site": {}}
        self._session = requests.Session()
        self._session_used = False
        self._stats_lock = threading.Lock()
        self._cancel = threading.Event()
        # Called with each streamed text delta instead of echoing to stdout
        # (GUI streaming, tests). None = echo to stdout.
        self.stream_callback = None
        self.echo_stream = True
        # Called with one dict per finished request (events.py attaches here).
        self.on_call = None
        # Label for the current call site (gather / answer / repair / ...);
        # set with `client.site = "answer"` and read by usage accounting.
        self.site = ""
        # Text streamed before a cancel or a mid-stream disconnect.
        self.partial_text = ""
        # Temporary sampling temperature (best-of-N candidates); None = config.
        self._temp_override: float | None = None
        # Worked step examples built from the real project (set by the agent):
        # shown instead of one dummy object when the server has no grammar.
        self.fewshot: list[dict] = []
        # None = untested; False after a server rejected stream_options.
        self._stream_usage_ok: bool | None = None
        # Learned once the template rejects a system role.
        self._no_system_role = False
        # Grammar-constrained JSON support: None = not probed (assume yes),
        # True = server advertises response_format/json_schema, False = it
        # answered the probe without advertising it (such requests fall back to
        # JSON prompting).
        self._schema_supported: bool | None = None
        # Native tool calling for step schemas (cfg.tool_mode = "native"):
        # None = not tried; False once the server rejected `tools`.
        # `native_override` lets `calibrate` A/B the two modes.
        self._native_ok: bool | None = None
        self.native_override: bool | None = None
        # Explicit prompt-cache markers (cache_control): False once the server
        # returned 400 on them; then this client falls back to implicit.
        self._cache_markers_rejected = False
        # "length" means cut off mid-answer: "valid JSON we never let it finish".
        self.last_finish_reason = ""
        # Calibrated chars-per-token from the server's tokenizer endpoint
        # (calibrate_tokenizer); 0.0 = uncalibrated, module heuristic applies.
        self.chars_per_token = 0.0
        # Truncation signal for streamed replies (finish_reason arrives on the
        # final SSE chunk); cleared per _chat_once call, sticky within it.
        self.stream_truncated = False
        # Chars of hidden reasoning in the last streamed attempt. chat()'s
        # empty-completion retry uses it to tell an instant-EOS flake (retry
        # identical) from a thinking runaway (retry with thinking disabled).
        self.last_reasoning_chars = 0
        # Circuit breaker: consecutive transport/server failures, and the
        # monotonic deadline before which calls fail fast. A success closes it.
        self._failures = 0
        self._circuit_until = 0.0
        self.circuit_trips = 0

    def _native_for(self, schema: dict | None, stream: bool) -> bool:
        """Whether this request should use the model's native tool-call channel
        instead of a JSON grammar: only for step schemas (an `action` field),
        never for streamed answers, and not after the server rejected tools."""
        if schema is None or stream or self._native_ok is False:
            return False
        if "action" not in (schema.get("properties") or {}):
            return False
        if self.native_override is not None:
            return self.native_override
        return getattr(self.cfg, "tool_mode", "grammar") == "native"

    def estimate_tokens(self, text: str) -> int:
        """Token estimate for fit math: the server's calibrated chars-per-token
        when warmup got one, else the module heuristic (~3.6)."""
        if self.chars_per_token:
            return int(len(text) / self.chars_per_token) + 1
        return estimate_tokens(text)

    def _context_too_long(self, body: str) -> bool:
        """True when a 400 body says the prompt overflowed the context window.
        llama.cpp/LM Studio say "exceeds the available context size" /
        "n_ctx"; Ollama says "context length exceeded" or "too many tokens";
        hosted APIs say "maximum context length" or `context_length_exceeded`.
        Bare "exceeds"/"out of range" are NOT enough: an unrelated 400 (a bad
        temperature, say) would otherwise trigger a pointless trim cascade."""
        return bool(_CONTEXT_OVERFLOW_RE.search(body or ""))

    def _http_post(self, url: str, **kw):
        """`requests.post` through this client's pooled session (a fresh TCP/TLS
        handshake per call is pure latency on hosted APIs), unless a caller
        replaced `requests.post`."""
        fn = requests.post
        if fn is not _REAL_POST:
            return fn(url, **kw)
        return self._session.post(url, **kw)

    def temperature(self, value: float):
        """Context manager: sample at `value` for the calls inside it (used to
        diversify best-of-N candidates)."""
        from contextlib import contextmanager

        @contextmanager
        def cm():
            prev = self._temp_override
            self._temp_override = value
            try:
                yield self
            finally:
                self._temp_override = prev
        return cm()

    def _provider(self):
        """The C0 provider entry for this client's server+model (cached)."""
        prov = getattr(self, "_prov", None)
        if prov is None:
            from .providers import provider_for
            prov = provider_for(self.cfg.base_url, self.cfg.model)
            self._prov = prov
        return prov

    def _reset_session(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass
        self._session = requests.Session()
        self._session_used = False

    def cancel(self) -> None:
        """Ask the in-flight (or next) streamed call to stop at the next
        chunk. Safe from any thread; cleared when the next call starts."""
        self._cancel.set()

    def _circuit_enabled(self) -> bool:
        return bool(getattr(self.cfg, "circuit_breaker", True))

    def _check_circuit(self) -> None:
        """Fail fast while the breaker is open; half-open after the cooldown."""
        if not self._circuit_enabled() or not self._circuit_until:
            return
        now = time.monotonic()
        if now < self._circuit_until:
            wait = self._circuit_until - now
            raise CircuitOpenError(
                f"The model server failed repeatedly; pausing requests for "
                f"another {wait:.0f}s (circuit breaker). Check that "
                f"{self.cfg.base_url} is reachable and the model is loaded, "
                "or set circuit_breaker = false to disable this guard.")
        # Cooldown elapsed: allow one probe. A failure re-opens immediately.
        self._circuit_until = 0.0

    def _record_success(self) -> None:
        self._failures = 0
        self._circuit_until = 0.0

    def _record_failure(self) -> None:
        if not self._circuit_enabled():
            return
        self._failures += 1
        limit = max(1, int(getattr(self.cfg, "circuit_failures", 4) or 4))
        if self._failures < limit:
            return
        cooldown = max(1.0, float(getattr(self.cfg, "circuit_cooldown", 60.0)
                                  or 60.0))
        self._failures = 0
        self._circuit_until = time.monotonic() + cooldown
        self.circuit_trips += 1
        print(f"[circuit breaker: {limit} consecutive model-server failures; "
              f"pausing requests for {cooldown:.0f}s]", file=sys.stderr)

    def _post(self, payload: dict, stream: bool = False):
        """POST to the chat completions endpoint, raising a specific LLMError
        subclass on failure (connection, timeout, load failure, template,
        transient/rate limit, context overflow, or other server error)."""
        self._check_circuit()
        try:
            try:
                r = self._http_post(
                    self.url, headers=self.headers, json=payload,
                    timeout=self.cfg.timeout, stream=stream)
            except requests.exceptions.ConnectionError:
                # A pooled keep-alive connection the server already closed
                # looks like a refused connection; retry once on a fresh
                # session before declaring the server unreachable.
                if not self._session_used:
                    raise
                self._reset_session()
                r = self._http_post(
                    self.url, headers=self.headers, json=payload,
                    timeout=self.cfg.timeout, stream=stream)
            self._session_used = True
        except requests.exceptions.ConnectionError as e:
            self._record_failure()
            raise LLMError(CONNECT_HELP.format(url=self.cfg.base_url)) from e
        except requests.exceptions.Timeout as e:
            self._record_failure()
            raise LLMError(
                f"The server at {self.cfg.base_url} did not answer within "
                f"{self.cfg.timeout:g}s (`timeout` in your config). A slow "
                "local model may need a larger timeout; a hosted API may be "
                "overloaded.") from e
        except requests.exceptions.RequestException as e:
            self._record_failure()
            raise LLMError(
                f"Request to {self.cfg.base_url} failed: "
                f"{type(e).__name__}: {str(e)[:200]}") from e
        if r.status_code >= 400:
            try:
                body = r.text[:800]
            finally:
                _close(r)
            low = body.lower()
            if any(k in low for k in _LOAD_FAILED):
                raise ModelLoadError(
                    f"The server could not load '{self.cfg.model}'.\n{LOAD_HELP}")
            if any(k in low for k in _TEMPLATE_HINTS):
                raise TemplateError(body[:400])
            loading = any(k in low for k in _LOADING)
            if r.status_code in (429, 503) or loading:
                raise _Transient(
                    f"{r.status_code}: {body[:200]}",
                    retry_after=None if loading else _retry_after(r),
                    rate_limited=not loading)
            if self._context_too_long(body):
                # chat() catches this, trims the conversation, and retries.
                raise ContextOverflowError(
                    f"Server says the request exceeds its context window: "
                    f"{body[:300]}")
            if r.status_code >= 500 and not loading:
                self._record_failure()
            raise LLMError(f"Server returned {r.status_code}: {body[:500]}")
        self._record_success()
        return r

    def embed(self, texts: list[str]) -> list[list[float]]:
        """One embedding vector per text via `<base_url>/embeddings`, in input
        order. Used only by optional semantic search (semantic.py); no chat
        retry/template handling."""
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []
        url = self.cfg.base_url.rstrip("/") + "/embeddings"
        model = getattr(self.cfg, "embedding_model", "") or self.cfg.model
        try:
            r = self._http_post(url, headers=self.headers,
                                json={"model": model, "input": list(texts)},
                                timeout=self.cfg.timeout)
        except requests.exceptions.ConnectionError as e:
            raise LLMError(CONNECT_HELP.format(url=self.cfg.base_url)) from e
        except requests.exceptions.RequestException as e:
            raise LLMError(f"Embeddings request failed: "
                           f"{type(e).__name__}: {str(e)[:200]}") from e
        if r.status_code >= 400:
            raise LLMError(
                f"Embeddings request failed ({r.status_code}): {r.text[:300]}. "
                f"Set embedding_model to a model your server can load, or turn "
                f"semantic_search off.")
        try:
            data = r.json()["data"]
            if isinstance(data, dict):        # single-input server shape
                data = [data]
            data = sorted(data, key=lambda d: d.get("index", 0))
            self.stats["calls"] += 1
            return [[float(x) for x in d["embedding"]] for d in data]
        except Exception as e:                  # pragma: no cover - server shape
            raise LLMError(
                f"Unexpected embeddings response: {r.text[:300]}") from e

    def vision(self, prompt: str, image_path, model: str | None = None,
               max_tokens: int = 400) -> str:
        """Ask a multimodal model to describe one local image (sent as an
        image_url data URI). A text-only model will reject it; the caller
        reports that as an observation."""
        path = Path(image_path)
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".webp": "image/webp"}.get(path.suffix.lower(), "image/png")
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        payload = {
            "model": model or getattr(self.cfg, "vision_model", "")
            or self.cfg.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]}],
            "max_tokens": max_tokens,
            "temperature": getattr(self.cfg, "temperature", 0.2),
        }
        r = self._post(payload)
        try:
            msg = r.json()["choices"][0]["message"]
            content = msg.get("content") or ""
            self.stats["calls"] += 1
            return strip_reasoning(content).strip()
        except Exception as e:                  # pragma: no cover - server shape
            raise LLMError(
                f"Unexpected vision response: {r.text[:300]}") from e

    def chat(self, messages: list[dict], max_tokens: int, schema: dict | None = None,
             stream: bool = False, disable_thinking: bool | None = None,
             json_schema: dict | None = None) -> str:
        """Send a chat completion request and return the response text.

        `json_schema` (Sprint Q) is the strict schema to attach as the
        engine-level `response_format` (constrained sampling); it defaults to
        `schema`, the shape the reply is parsed against. Servers that cannot
        honour it are downgraded automatically (see _chat_once).

        Retries while the model is still loading (and, separately, while a
        provider rate-limits), lowers `max_tokens` so prompt + reply fit the
        window, recovers from a server-reported context overflow, and folds
        the system prompt into the first user turn if the template rejects a
        system role.

        The client never rewrites conversation history on its own to make a
        prompt fit: that is the agent's job (one fold point, see
        agent.fold_middle) because every rewrite invalidates the server's
        prompt-prefix cache. The overflow path below is the last resort, taken
        only after the server itself refused the request.
        """
        messages = merge_consecutive(messages)
        if self._no_system_role:
            messages = fold_system(messages)
        delay = 2.0
        started = time.time()
        announced = False
        attempt = 0
        trimmed_once = False
        shrunk_once = False
        empty_streak = 0        # consecutive empty streamed replies
        rate_waits = 0          # rate-limit retries, budgeted apart from loading
        while True:
            attempt += 1
            # Fit the REPLY to the window: llama.cpp/LM Studio reject requests
            # whose prompt + max_tokens exceed it. Only max_tokens moves.
            window = self.cfg.context_tokens
            if window > 0 and not _is_remote(self.cfg.base_url):
                total = estimate_messages(messages, self.estimate_tokens)
                fit = window - total - 32
                if fit < max_tokens and fit >= 256:
                    print(f"[prompt ~{total:,} tokens in a {window:,}-token "
                          f"window; lowering the reply budget "
                          f"{max_tokens:,} -> {fit:,}]", file=sys.stderr)
                    max_tokens = fit
            try:
                text = self._chat_once(messages, max_tokens, schema, stream,
                                       disable_thinking, json_schema)
            except ContextOverflowError as e:
                # The server's window is smaller than our budgets assumed
                # (stock LM Studio 8192 / Ollama 4096). Halve the reply budget
                # first (no history rewrite), then shed stale middle history,
                # then fail.
                if not shrunk_once and max_tokens > 256:
                    shrunk_once = True
                    max_tokens = max(256, max_tokens // 2)
                    print("[server context overflow; halving the reply "
                          "budget and retrying once]", file=sys.stderr)
                    continue
                if not trimmed_once:
                    body_idx = [i for i, m in enumerate(messages)
                                if m["role"] == "user"
                                and m["content"].startswith("RESULT:\n")]
                    if len(body_idx) > 2:
                        drop = set(body_idx[: len(body_idx) // 2])
                        messages = merge_consecutive(
                            [m for i, m in enumerate(messages)
                             if i not in drop])
                        trimmed_once = True
                        print("[server context overflow: dropped older tool "
                              "results and retrying]", file=sys.stderr)
                        continue
                if any(m["role"] == "system" for m in messages) and \
                        len([m for m in messages if m["role"] != "system"]) > 2:
                    # No RESULT turns to shed (answer phase): drop the middle
                    # assistant turns, keeping the first user turn (map+task)
                    # and the last (latest instruction/result).
                    keep_first = next(i for i, m in enumerate(messages)
                                      if m["role"] == "user")
                    keep_last = max(i for i, m in enumerate(messages)
                                    if m["role"] == "user")
                    if keep_last - keep_first > 1:
                        mid = [i for i in range(keep_first + 1, keep_last)
                               if messages[i]["role"] == "assistant"]
                        if mid:
                            mid = set(mid)
                            messages = merge_consecutive(
                                [m for i, m in enumerate(messages)
                                 if i not in mid])
                            print("[server context overflow: shortened the "
                                  "conversation and retrying]", file=sys.stderr)
                            continue
                raise LLMError(
                    f"{e}\n\n"
                    "Pedacito already trimmed the conversation once and the "
                    "server still refused.\nThis model's configured context "
                    "window is too small for even a minimal run.\n"
                    "  * LM Studio: raise 'Context length' in the model's "
                    "runtime settings (8192 is\n"
                    "    the stock default; 16384-32768 suits a 27B on most "
                    "GPUs) and re-run.\n"
                    "  * Ollama: set num_ctx (e.g. OLLAMA_NUM_CTX=16384, or "
                    "options.num_ctx in a\n"
                    "    Modelfile). Pedacito reads it automatically.\n"
                    "  * Or lower context_tokens/max_gathered_chars in your "
                    "profile so auto-fit\n"
                    "    plans smaller prompts from the start.") from e
            except TemplateError as e:
                # Most likely no system role: fold it into the first user turn
                # and remember that. (warmup() resolves this before the first
                # real turn, so a mid-run flip is the rare case.)
                if any(m["role"] == "system" for m in messages):
                    if not self._no_system_role:
                        print(f"['{self.cfg.model}' has no system role; folding "
                              "it into the first user message]", file=sys.stderr)
                    self._no_system_role = True
                    messages = fold_system(messages)
                    continue
                raise LLMError(
                    f"The chat template for '{self.cfg.model}' rejected the "
                    f"request:\n{e}\n\n"
                    "This is a model-template incompatibility, not a Pedacito bug "
                    "in itself.\nTry a different quant or a different model; "
                    "Gemma and Qwen templates are\nmore permissive than "
                    "Mistral's.") from e
            except _Transient as e:
                if e.rate_limited:
                    # A provider asking us to slow down is not a model load:
                    # own budget, honour Retry-After, add jitter.
                    rate_waits += 1
                    if rate_waits > _MAX_RATE_LIMIT_RETRIES:
                        raise LLMError(
                            f"The server kept rate-limiting requests "
                            f"({_MAX_RATE_LIMIT_RETRIES} retries). Last "
                            f"response: {e}") from e
                    wait = e.retry_after if e.retry_after is not None \
                        else min(2.0 * 2 ** (rate_waits - 1), 30.0)
                    wait += random.uniform(0, 0.25 * max(wait, 1.0))
                    print(f"[rate limited by the server; waiting {wait:.0f}s "
                          f"({rate_waits}/{_MAX_RATE_LIMIT_RETRIES})]",
                          file=sys.stderr)
                    time.sleep(wait)
                    continue
                waited = time.time() - started
                if waited >= self.cfg.load_wait_seconds:
                    raise LLMError(
                        f"The model server was still not ready for "
                        f"'{self.cfg.model}' after {waited:.0f}s.\n"
                        "  If you are switching models, the server may be waiting "
                        "for the previous one\n"
                        "  to unload. Check its model-management UI, or raise "
                        "load_wait_seconds in your config.\n"
                        f"  Last response: {e}") from e
                if not announced:
                    print(f"[waiting for '{self.cfg.model}' to load -- switching "
                          f"models can take a minute or two]", file=sys.stderr)
                    announced = True
                elif attempt % 5 == 0:
                    print(f"[still waiting, {waited:.0f}s elapsed]", file=sys.stderr)
                time.sleep(delay + random.uniform(0, 0.5))
                delay = min(delay * 1.6, 10.0)
                continue

            if not stream or (text or "").strip():
                return text
            # An empty streamed answer is a provider flake or a thinking
            # runaway (budget spent on hidden reasoning), never a legitimate
            # answer. Retry up to twice here, then hand the "" back to the
            # callers' own fallbacks.
            empty_streak += 1
            if empty_streak > 2:
                return text
            effective_thinking_off = (disable_thinking
                                      if disable_thinking is not None
                                      else self.cfg.disable_thinking)
            if self.last_reasoning_chars and not effective_thinking_off:
                # Thinking runaway: re-rolling would think again; switch
                # thinking off instead.
                disable_thinking = True
                how = " with thinking disabled"
            else:
                how = ""
            print(f"[the model streamed an EMPTY answer (~"
                  f"{self.last_reasoning_chars:,} chars of hidden reasoning, "
                  f"finish '{self.last_finish_reason or 'stop'}'); "
                  f"retrying{how}]", file=sys.stderr)
            time.sleep(1.0)

    def _check_budget(self) -> None:
        """Raise BudgetExceeded before a call when a configured token/cost
        budget is already spent."""
        cap_tokens = int(getattr(self.cfg, "max_total_tokens", 0) or 0)
        used = self.stats["prompt_tokens"] + self.stats["completion_tokens"]
        if cap_tokens and used >= cap_tokens:
            raise BudgetExceeded(
                f"Token budget spent: {used:,} of max_total_tokens="
                f"{cap_tokens:,}. Raise it (or pass --max-tokens-budget) to "
                "continue; progress so far is saved.")
        cap_cost = float(getattr(self.cfg, "max_cost", 0) or 0)
        if cap_cost and self.stats["cost"] >= cap_cost:
            raise BudgetExceeded(
                f"Cost budget spent: ${self.stats['cost']:.4f} of max_cost="
                f"${cap_cost:.2f}. Raise it (or pass --max-cost) to continue.")

    def _account(self, usage: dict, timings: dict | None, seconds: float,
                 prompt_est: int, out_text: str = "") -> dict:
        """Fold one response's usage into the stats; returns the call record
        handed to `on_call`."""
        usage = usage or {}
        pt = int(usage.get("prompt_tokens") or 0)
        ct = int(usage.get("completion_tokens") or 0)
        if not pt and not ct:
            # The server reported nothing: keep estimates so budgets and the
            # per-task summary still mean something.
            pt = prompt_est
            ct = self.estimate_tokens(out_text) if out_text else 0
        # Cache reads: OpenAI `cached_tokens`, DeepSeek's implicit hit tokens,
        # Anthropic/DashScope `cache_read_input_tokens`, or a local server's KV
        # reuse. Cache writes are the explicit-cache creation tokens.
        cached = _usage_tokens(usage, "cached_tokens", "prompt_cache_hit_tokens",
                               "cache_read_input_tokens")
        cached = cached or (_positive_int((timings or {}).get("cache_n")) or 0)
        written = _usage_tokens(usage, "cache_creation_input_tokens",
                                "cache_write_tokens")
        cost = 0.0
        p_in = float(getattr(self.cfg, "price_input_per_mtok", 0) or 0)
        p_out = float(getattr(self.cfg, "price_output_per_mtok", 0) or 0)
        p_cached = float(getattr(self.cfg, "price_cached_per_mtok", 0) or 0)
        p_write = float(getattr(self.cfg, "price_cache_write_per_mtok", 0) or 0)
        if p_in or p_out:
            fresh = max(0, pt - cached - written)
            cost = (fresh * p_in + cached * (p_cached or p_in)
                    + written * (p_write or p_in)
                    + ct * p_out) / 1_000_000
        site = self.site or "call"
        with self._stats_lock:      # summaries may run on several threads
            self.stats["prompt_tokens"] += pt
            self.stats["completion_tokens"] += ct
            self.stats["cached_tokens"] += cached
            self.stats["cache_write_tokens"] += written
            self.stats["seconds"] += seconds
            self.stats["cost"] += cost
            row = self.stats["by_site"].setdefault(
                site, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                       "cached_tokens": 0, "cache_write_tokens": 0})
            row["calls"] += 1
            row["prompt_tokens"] += pt
            row["completion_tokens"] += ct
            row["cached_tokens"] += cached
            row["cache_write_tokens"] += written
        record = {"site": site, "prompt_tokens": pt, "completion_tokens": ct,
                  "cached_tokens": cached, "cache_write_tokens": written,
                  "seconds": round(seconds, 3),
                  "cost": round(cost, 6), "model": self.cfg.model,
                  "finish": self.last_finish_reason}
        cb = self.on_call
        if cb is not None:
            try:
                cb(record)
            except Exception:
                pass          # telemetry must never break a call
        return record

    def _chat_once(self, messages: list[dict], max_tokens: int,
                   schema: dict | None = None, stream: bool = False,
                   disable_thinking: bool | None = None,
                   json_schema: dict | None = None) -> str:
        """One HTTP request (no retry logic); returns the response text,
        buffered or streamed (to `stream_callback`, else stdout).

        Structured output ladder (Sprint Q, `cfg.enforce_json_schema`): the
        request asks the ENGINE to constrain sampling -- strict json_schema,
        else json_object, else plain JSON prompting. A 400 that names
        response_format flips the tried tier off for this (base_url, model)
        and the request is re-sent one tier down immediately, so no call ever
        fails over server schema support differences."""
        self._check_budget()
        fmt_schema = json_schema if json_schema is not None else schema
        msgs = [dict(m) for m in messages]
        want_off = (disable_thinking if disable_thinking is not None
                    else self.cfg.disable_thinking)
        # An explicit suffix is always applied (it may be a user's own
        # instruction). Otherwise use the family's soft switch only when we
        # are actually asking for non-thinking mode.
        suffix = getattr(self.cfg, "thinking_suffix", "")
        if not suffix and want_off:
            suffix = thinking_suffix_for(self.cfg)
        if suffix and msgs:
            msgs[-1]["content"] += "\n\n" + suffix
        if fmt_schema is not None and self.cfg.json_mode == "object" and msgs:
            # Hosted APIs require the word "json" in the prompt for
            # json_object mode; naming the fields improves compliance.
            fields = ", ".join(fmt_schema.get("properties", {}).keys())
            msgs[-1]["content"] += (
                f"\n\nReply with one JSON object only, no prose. "
                f"Fields: {fields}.")

        temp = (self._temp_override
                if self._temp_override is not None
                else self.cfg.temperature)
        payload = {
            "model": self.cfg.model,
            "messages": msgs,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        # Claude 4.7+ rejects any `temperature` below 1; the provider table
        # marks that with omit_temperature, so drop the key rather than 400.
        if not (self._provider().omit_temperature and float(temp) < 1.0):
            payload["temperature"] = temp
        # request_max_tokens_cap (0 = off): llama.cpp-based servers 400 when
        # max_tokens exceeds the context window, so never exceed the cap.
        cap = getattr(self.cfg, "request_max_tokens_cap", 0) or 0
        if cap and max_tokens > cap:
            payload["max_tokens"] = cap
        # Sampler parameters are merged verbatim (an unsupported one 400s the
        # request); structural keys (output cap, grammar, tools, streaming)
        # are protected so a sampler table cannot defeat them.
        for k, v in (self.cfg.sampler or {}).items():
            if k in _PROTECTED_SAMPLER_KEYS:
                continue
            payload[k] = v
        if self._temp_override is not None:
            # A phase temperature (temperature_answer/_repair, best_of) is more
            # specific than a `temperature` key in the sampler table.
            payload["temperature"] = self._temp_override
        if want_off:
            # Local templates get the union of known spellings; hosted
            # providers get only the field they are known to accept. Any
            # user-supplied chat_template_kwargs survive and merge.
            kwargs, top = thinking_off_fields(self.cfg)
            if kwargs:
                current = payload.get("chat_template_kwargs")
                merged = dict(current) if isinstance(current, dict) else {}
                merged.update(kwargs)
                payload["chat_template_kwargs"] = merged
            payload.update(top)
        if stream and self._stream_usage_ok is not False \
                and getattr(self.cfg, "stream_usage", True):
            # Without this, a streamed reply (the whole answer phase) reports
            # no usage at all and its cost/cache-hit data is lost.
            payload["stream_options"] = {"include_usage": True}
        native = self._native_for(schema, stream)
        if native:
            # One function whose parameters ARE the step schema, forced: the
            # model fills it in through its own tool-call training instead of a
            # sampling grammar. A server that ignores tool_choice may answer in
            # content; chat_json parses that the same way.
            payload["tools"] = [{"type": "function", "function": {
                "name": "step",
                "description": "Choose the next action for the task.",
                "parameters": schema}}]
            payload["tool_choice"] = {"type": "function",
                                      "function": {"name": "step"}}
        elif fmt_schema is not None:
            # Structured-output ladder for THIS endpoint. The engine tier
            # (`schema`) constrains sampling; `object` only asks for JSON;
            # `plain` keeps discipline in the prompt (and in the few-shot
            # examples the agent primed). A 400 rejection flips a tier off
            # in _json_caps, so the downgrade re-send lands one rung lower.
            caps = _json_caps(self.cfg.base_url, self.cfg.model)
            if self.cfg.json_mode == "object":
                tier = "object" if caps.get("object", True) else "plain"
            elif not bool(getattr(self.cfg, "enforce_json_schema", True)) \
                    or self._schema_supported is False:
                # Opt-out, or a probe found this server answers without
                # advertising constrained JSON: plain prompting, no grammar.
                tier = "plain"
            elif caps.get("schema", True):
                tier = "schema"
            elif caps.get("object", True):
                tier = "object"
            else:
                tier = "plain"
            if tier == "schema":
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "pedacito_action", "strict": True,
                                    "schema": fmt_schema},
                }
            elif tier == "object":
                payload["response_format"] = {"type": "json_object"}
            elif self.cfg.json_mode != "object":
                fields = ", ".join(fmt_schema.get("properties", {}).keys())
                shots = self.fewshot if "action" in (
                    fmt_schema.get("properties") or {}) else []
                if shots:
                    examples = "\n".join(
                        schema_example_with(fmt_schema, **s) for s in shots)
                    msgs[-1]["content"] += (
                        "\n\nReply with one JSON object only, no prose. "
                        f"Fields: {fields}.\nWorked examples (for THIS "
                        f"project; choose what the task needs):\n"
                        + examples)
                else:
                    msgs[-1]["content"] += (
                        "\n\nReply with one JSON object only, no prose. "
                        f"Fields: {fields}.\nExample of a well-formed "
                        "reply:\n" + _schema_example(fmt_schema))
        key = getattr(self.cfg, "prompt_cache_key", "") or ""
        if key:
            from .providers import cache_key, provider_for
            key = cache_key(self.cfg)
            if key and provider_for(self.cfg.base_url,
                                    self.cfg.model).cache_key_ok \
                    and _is_remote(self.cfg.base_url):
                payload["prompt_cache_key"] = key
        self._apply_acceleration(payload)

        prompt_est = estimate_messages(msgs, self.estimate_tokens)
        # Explicit prompt-cache markers go on AFTER the estimate and after the
        # replay digest (which hashes the original `messages`, not `msgs`), so
        # neither the budget math nor --replay matching sees them.
        self._apply_cache_markers(msgs)
        t0 = time.time()
        self.stats["calls"] += 1
        # Cleared per attempt so a stale "length" from an abandoned attempt
        # cannot trigger apply()'s continuation loop.
        self.stream_truncated = False
        self.partial_text = ""
        if stream:
            self._cancel.clear()

        if not stream:
            try:
                r = self._post(payload)
            except LLMError as e:
                if self._cache_markers_rejected_by(e):
                    self._cache_markers_rejected = True
                    print("[this server rejected explicit cache_control "
                          "markers; falling back to implicit caching]",
                          file=sys.stderr)
                    return self._chat_once(messages, max_tokens, schema,
                                           stream, disable_thinking,
                                           json_schema)
                if native and self._tools_rejected(e):
                    self._native_ok = False
                    print("[this server rejected native tool calls; using the "
                          "JSON grammar instead]", file=sys.stderr)
                    return self._chat_once(messages, max_tokens, schema, stream,
                                           disable_thinking, json_schema)
                if self._downgrade_response_format(payload, e):
                    return self._chat_once(messages, max_tokens, schema, stream,
                                           disable_thinking, json_schema)
                if self._deny_accel_caps(payload, e):
                    return self._chat_once(messages, max_tokens, schema, stream,
                                           disable_thinking, json_schema)
                raise
            try:
                data = r.json()
            finally:
                _close(r)
            try:
                self.last_finish_reason = data["choices"][0].get("finish_reason") or ""
            except (KeyError, IndexError, AttributeError, TypeError):
                self.last_finish_reason = ""
            try:
                msg = data["choices"][0]["message"]
                text = (_tool_call_arguments(msg) if native else "") \
                    or extract_content(msg)
            except (KeyError, IndexError, TypeError) as e:
                raise LLMError(f"Unexpected response shape: {json.dumps(data)[:500]}") from e
            if native and self._native_ok is None:
                self._native_ok = True
            self._account(data.get("usage") or {}, data.get("timings"),
                          time.time() - t0, prompt_est, text)
            return text

        # Streaming, so the user sees progress at local-model speeds.
        out: list[str] = []
        tool_opens = 0
        probe_tail = ""
        reasoning_chars = 0
        usage: dict = {}
        timings: dict | None = None
        # Per-attempt diagnostics for chat()'s empty-completion retry.
        self.last_finish_reason = ""
        self.last_reasoning_chars = 0
        try:
            r = self._post(payload, stream=True)
        except LLMError as e:
            if self._cache_markers_rejected_by(e):
                self._cache_markers_rejected = True
                print("[this server rejected explicit cache_control markers; "
                      "falling back to implicit caching]", file=sys.stderr)
                return self._chat_once(messages, max_tokens, schema, stream,
                                       disable_thinking, json_schema)
            if self._downgrade_response_format(payload, e):
                return self._chat_once(messages, max_tokens, schema, stream,
                                       disable_thinking, json_schema)
            if self._deny_accel_caps(payload, e):
                return self._chat_once(messages, max_tokens, schema, stream,
                                       disable_thinking, json_schema)
            low = str(e).lower()
            if "stream_options" in payload and (
                    "stream_options" in low or "include_usage" in low):
                # An older server that rejects the usage option: remember and
                # retry once without it.
                self._stream_usage_ok = False
                payload.pop("stream_options", None)
                r = self._post(payload, stream=True)
            else:
                raise
        sink = self.stream_callback
        cancelled = False
        dropped: Exception | None = None
        try:
            for raw in r.iter_lines(decode_unicode=True):
                if self._cancel.is_set():
                    cancelled = True
                    break
                if not raw:
                    continue
                # requests only decodes when the response declares a charset,
                # and text/event-stream often doesn't.
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", "replace")
                if raw.startswith(":"):
                    continue                    # SSE comment / keep-alive
                if raw.startswith("data:"):
                    body = raw[5:].lstrip()
                else:
                    continue
                if body.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(body)
                except Exception:
                    continue
                if isinstance(obj, dict) and obj.get("usage"):
                    usage = obj["usage"]
                if isinstance(obj, dict) and obj.get("timings"):
                    timings = obj["timings"]
                try:
                    chunk = obj["choices"][0]
                    d = chunk.get("delta") or {}
                    fr = chunk.get("finish_reason") or ""
                except Exception:
                    continue
                if fr:
                    self.last_finish_reason = fr
                if fr == "length":
                    self.stream_truncated = True
                delta = d.get("content")
                rc = _reasoning_text(d)
                if not delta and not rc:
                    continue
                if rc:
                    reasoning_chars += len(rc)       # counted, never echoed
                if delta:
                    out.append(delta)
                    if sink is not None:
                        try:
                            sink(delta)
                        except Exception:
                            pass
                    elif self.echo_stream:
                        sys.stdout.write(delta)
                        sys.stdout.flush()
                    # Degenerate native-tool-call loop guard; the 4-char
                    # overlap catches an opening tag split across chunks.
                    probe = probe_tail + delta
                    tool_opens += len(_NATIVE_TOOL_OPEN_RE.findall(probe))
                    probe_tail = probe[-4:]
                    if tool_opens >= _MAX_STREAM_TOOL_OPENS:
                        print("\n[stopped the reply: the model is repeating "
                              "tool-call syntax instead of answering; no tools "
                              "can run in this phase]", file=sys.stderr)
                        break
        except requests.exceptions.RequestException as e:
            dropped = e
        finally:
            _close(r)
        if sink is None and self.echo_stream:
            sys.stdout.write("\n")
        self.last_reasoning_chars = reasoning_chars
        text = strip_reasoning("".join(out))
        self.partial_text = text
        self._account(usage, timings, time.time() - t0, prompt_est, text)
        if cancelled:
            raise CancelledError("Cancelled by the user; the partial reply is "
                                 "kept in client.partial_text.")
        if dropped is not None:
            self._record_failure()
            if not text.strip():
                raise LLMError(
                    f"The connection to {self.cfg.base_url} dropped mid-reply "
                    f"({type(dropped).__name__}: {str(dropped)[:160]}).") \
                    from dropped
            # Partial answer survives: flag it truncated so the existing
            # continuation machinery asks for the rest instead of losing it.
            self.stream_truncated = True
            print(f"\n[connection dropped mid-reply "
                  f"({type(dropped).__name__}); treating the reply as cut "
                  f"off]", file=sys.stderr)
        return text

    @staticmethod
    def _tools_rejected(err: Exception) -> bool:
        low = str(err).lower()
        return any(w in low for w in ("tools", "tool_choice", "tool calling",
                                      "function call", "tool use"))

    @staticmethod
    def _cache_markers_rejected_by(err: Exception) -> bool:
        low = str(err).lower()
        return "cache_control" in low or "cache-control" in low

    @staticmethod
    def _response_format_rejected_by(err: Exception) -> bool:
        """True when a 400 says the SERVER cannot honour the structured-output
        request (not that our payload is otherwise malformed): llama.cpp-style
        `response_format` complaints, OpenAI's "Invalid schema for
        response_format", Ollama/vLLM "json_schema is not supported" or
        "unknown field" errors naming one of the two modes."""
        low = str(err).lower()
        if "response_format" in low:
            return True
        return (("json_schema" in low or "json_object" in low)
                and any(w in low for w in
                        ("unsupported", "not supported", "unknown", "invalid",
                         "unexpected", "unrecognized", "unrecognised",
                         "rejected", "does not", "cannot", "unable")))

    def _downgrade_response_format(self, payload: dict,
                                   err: Exception) -> bool:
        """Handle a structured-output 400 once: flip the tier this request
        used off the per-(base_url, model) capability cache and report that a
        lower tier exists, so the caller re-sends immediately. Tiers:
        json_schema -> json_object -> plain prompt. A second 400 on the same
        tier is a genuine error (the cache already says the tier is out),
        which keeps a server that 400s for an unrelated reason from looping."""
        rf = (payload.get("response_format") or {}).get("type") or ""
        if not rf or not self._response_format_rejected_by(err):
            return False
        tier = "schema" if rf == "json_schema" else "object"
        if not _json_caps(self.cfg.base_url, self.cfg.model).get(tier, True):
            return False
        _deny_json_cap(self.cfg.base_url, self.cfg.model, tier)
        if tier == "schema":
            print("[engine JSON mode: this server refused json_schema; "
                  "retrying with json_object]", file=sys.stderr)
        else:
            print("[engine JSON mode: this server refused json_object; "
                  "retrying with plain JSON prompting]", file=sys.stderr)
        return True

    def _apply_acceleration(self, payload: dict) -> None:
        """Attach the Sprint R local-inference speed-up keys when this endpoint
        is self-hosted, the settings ask for them, and a previous call has not
        already seen this server refuse them. Keys a `[sampler]` table set by
        hand win: the user's own spelling is left untouched."""
        if not _is_local_server(self.cfg):
            return
        caps = _accel_caps(self.cfg.base_url, self.cfg.model)
        if getattr(self.cfg, "enable_prompt_cache", True) \
                and caps.get("cache", True) \
                and _ACCEL_CACHE_KEY not in payload:
            payload[_ACCEL_CACHE_KEY] = True
        if getattr(self.cfg, "enable_prompt_lookup", False) \
                and caps.get("lookup", True) \
                and _ACCEL_LOOKUP_KEY not in payload:
            n = getattr(self.cfg, "prompt_lookup_num_tokens", 10)
            try:
                n = int(n)
            except (TypeError, ValueError):
                n = 0
            if n > 0:
                payload[_ACCEL_LOOKUP_KEY] = n

    def _deny_accel_caps(self, payload: dict, err: Exception) -> bool:
        """Handle a status rejection that may be caused by the acceleration
        keys: flip the attached-but-unproven ones off for this (base_url,
        model) and report that the caller should re-send without them. Only
        fires on 400/422 (an endpoint validating its request schema) while the
        payload still carries at least one acceleration key this cache has not
        yet refused; a second rejection of the same keys is a genuine error,
        which keeps a server that 400s for an unrelated reason from looping."""
        attached = [k for k in (_ACCEL_CACHE_KEY, _ACCEL_LOOKUP_KEY)
                    if k in payload]
        if not attached:
            return False
        low = str(err).lower()
        if not any(low.startswith(f"server returned {code}")
                   for code in ("400", "422")):
            return False
        caps = _accel_caps(self.cfg.base_url, self.cfg.model)
        names = {_ACCEL_CACHE_KEY: "cache", _ACCEL_LOOKUP_KEY: "lookup"}
        fresh = [k for k in attached if caps.get(names[k], True)]
        if not fresh:
            return False
        for k in fresh:
            _deny_accel_cap(self.cfg.base_url, self.cfg.model, names[k])
        print(f"[local acceleration: this server refused "
              f"{', '.join(fresh)}; retrying without it]", file=sys.stderr)
        return True

    def _apply_cache_markers(self, msgs: list[dict]) -> bool:
        """In explicit prompt-cache mode, wrap the breakpoint messages' content
        in a content-part array carrying a `cache_control` marker: system end,
        user0 end, and the last message. Returns True when any marker was
        added. No-op when the mode is not explicit, or a 400 rejection already
        disabled markers for this server+model."""
        if self._cache_markers_rejected:
            return False
        from .providers import cache_mode_for, provider_for
        if cache_mode_for(self.cfg) != "explicit":
            return False
        prov = provider_for(self.cfg.base_url, self.cfg.model)
        max_markers = prov.max_markers or 4
        ttl = (getattr(self.cfg, "prompt_cache_ttl", "5m")
               or "5m").strip().lower()
        if prov.ttls and ttl not in prov.ttls:
            ttl = ""       # an unsupported TTL: send a plain ephemeral marker
        idxs: list[int] = []
        for i, m in enumerate(msgs):
            if m.get("role") == "system":
                idxs.append(i)
                break
        for i, m in enumerate(msgs):
            if m.get("role") == "user" and i not in idxs:
                idxs.append(i)
                break
        if idxs and idxs[-1] != len(msgs) - 1:
            idxs.append(len(msgs) - 1)
        added = False
        for i in idxs[:max_markers]:
            content = msgs[i].get("content")
            if not isinstance(content, str):
                continue
            cc: dict = {"type": "ephemeral"}
            if ttl:
                cc["ttl"] = ttl
            msgs[i]["content"] = [{"type": "text", "text": content,
                                   "cache_control": cc}]
            added = True
        return added

    @staticmethod
    def _schema_has_big_fields(schema: dict | None) -> bool:
        """True when the schema has a string field large enough to hold real
        file content. STEP_SCHEMA caps `content` at 6000 chars, so a bigger
        budget cannot help a cut step reply (it died inside a short field);
        such schemas skip budget escalation."""
        BIG_FIELD_MIN = 8000   # maxLength above which a string field is "large"
        props = (schema or {}).get("properties", {})
        return any(
            p.get("type") == "string" and int(p.get("maxLength") or 0) > BIG_FIELD_MIN
            for p in props.values())

    def chat_json(self, messages: list[dict], schema: dict, max_tokens: int,
                  disable_thinking: bool | None = None) -> dict:
        """Send a grammar-constrained chat request and return the parsed
        JSON object.

        The common failure is not malformed output but a reply cut off at
        max_tokens, so a truncated attempt retries with a larger budget (see
        `_schema_has_big_fields` for when it does not), or salvages the
        complete prefix of the object.
        """
        retries = 2
        budget = max_tokens
        # Set when a truncated reply was salvaged; `truncated_field` names the
        # schema field the cut landed in ("" = nothing cut, or benign) so
        # agent.gather can refuse to dispatch a step whose payload was cut.
        self.step_truncated = False
        self.truncated_field = ""
        # Cuts of an answer-style schema; after the second, accept the salvage.
        step_cut = 0
        # One re-ask per call with an explicit no-reasoning instruction for
        # "whole budget spent on a preamble, no JSON at all".
        trunc_nudged = False
        # Ceiling for budget escalation, derived from the caller's config.
        from .config import is_hosted
        if is_hosted(self.cfg):
            # Hosted APIs have a provider-side output ceiling (~8k tokens on
            # deepseek-chat) that escalation cannot beat; honour it when known.
            mot = int(getattr(self.cfg, "max_output_tokens", 0) or 0)
            if mot > 0:
                cap = max(mot, max_tokens)
            else:
                cap = max(65536, self.cfg.max_tokens_answer,
                          self.cfg.max_tokens_step * 8, max_tokens * 8)
        else:
            # Never escalate past what the context window can hold.
            usable = self.cfg.context_tokens - self.cfg.reserve_tokens
            cap = min(max(2048, self.cfg.reserve_tokens,
                          self.cfg.max_tokens_answer, max_tokens * 4),
                      max(usable, max_tokens))
        cap = max(cap, max_tokens)
        for attempt in range(retries + 1):
            raw = self.chat(messages, max_tokens=budget, schema=schema,
                            disable_thinking=disable_thinking)
            truncated = self.last_finish_reason == "length"
            text = raw.strip()
            if text.startswith("```"):
                text = text.strip("`")
                text = text.split("\n", 1)[-1] if "\n" in text else text
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # Salvage a JSON object surrounded by prose. Scan every '{'
                # with raw_decode (thinking models emit several candidate
                # objects first); requiring an "action"/"edits" key keeps a
                # stray {"note": ...} from being taken for the reply.
                dec = json.JSONDecoder()
                start = 0
                while True:
                    i = text.find("{", start)
                    if i == -1:
                        break
                    try:
                        obj, _end = dec.raw_decode(text[i:])
                    except json.JSONDecodeError:
                        start = i + 1
                        continue
                    if isinstance(obj, dict) and ("action" in obj or "edits" in obj):
                        if i > 0 or _end + i < len(text):
                            print("[prose around JSON; salvaged the object]",
                                  file=sys.stderr)
                        return obj
                    start = i + 1
                i, j = text.find("{"), text.rfind("}")
                if i != -1 and j > i:
                    frag = text[i : j + 1]
                    try:
                        return json.loads(frag)
                    except json.JSONDecodeError:
                        pass
                    # Some servers (DeepSeek in json_object mode) emit
                    # single-quoted Python dicts. Convert only when the
                    # fragment has no double quotes, so apostrophes inside
                    # real strings are not mangled.
                    if '"' not in frag and "'" in frag:
                        cand = frag.replace("'", '"')
                        try:
                            obj = json.loads(cand)
                        except json.JSONDecodeError:
                            obj = None
                        if isinstance(obj, dict) and obj:
                            print("[reply was a single-quoted dict; "
                                  "converted to JSON]", file=sys.stderr)
                            return obj
                if truncated:
                    patched = repair_truncated_json(text)
                    if patched:
                        try:
                            obj = json.loads(patched)
                        except json.JSONDecodeError:
                            obj = None
                        # Only trust it if the part that matters survived.
                        if isinstance(obj, dict) and obj.get("action"):
                            if not self._schema_has_big_fields(schema):
                                # By default args precede 'reasoning', so a
                                # cut usually lands in the essay. A long
                                # `pattern` or `content` can also be cut (and,
                                # reasoning-first, any arg); record which
                                # field so agent.gather can refuse it.
                                self.truncated_field = _cut_field(text, obj,
                                                                  schema)
                                print("[reply was cut off; recovered the "
                                      "action from the partial reply]",
                                      file=sys.stderr)
                                self.step_truncated = True
                                return obj
                            # Answer-style schema: the salvage is partial, so
                            # do not act on half a file; escalate instead.
                            step_cut += 1
                            if step_cut >= 2:
                                print("[reply was cut off mid-content; last "
                                      "attempt will be salvaged whole]",
                                      file=sys.stderr)
                                self.step_truncated = True
                                self.truncated_field = _cut_field(text, obj,
                                                                  schema)
                                return obj
                if truncated:
                    # Step schemas: the reply died before any usable action,
                    # so a bigger budget cannot help. Re-ask once with a
                    # no-reasoning nudge, then fail.
                    if not self._schema_has_big_fields(schema):
                        if not trunc_nudged and attempt < retries:
                            trunc_nudged = True
                            print("[reply was cut off before any JSON was "
                                  "emitted; re-asking with a no-reasoning "
                                  "nudge]", file=sys.stderr)
                            messages = messages + [
                                {"role": "assistant", "content": raw},
                                {"role": "user", "content":
                                    "Your previous reply was cut off by the "
                                    "output token limit before you emitted "
                                    "the JSON object. Do not think or explain: "
                                    "output the JSON object itself now, with "
                                    "'reasoning' kept under 10 words."},
                            ]
                            continue
                        break   # fall through to the informative error below
                    nxt = min(budget * 2, cap)
                    if nxt > budget:
                        budget = nxt
                        print(f"[reply was cut off at the token limit; retrying "
                              f"with max_tokens={budget}] ", file=sys.stderr)
                    else:
                        print(f"[reply was cut off at the token limit and "
                              f"max_tokens is already at the ceiling "
                              f"({budget}); the answer will be salvaged from "
                              f"what survived]",
                              file=sys.stderr)
                    continue
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "That was not valid JSON. Reply with "
                                                "the JSON object only, and keep "
                                                "'reasoning' under 15 words."},
                ]
        # A cut-off last attempt, not a grammar failure, is what went wrong.
        if self.last_finish_reason == "length":
            raise LLMError(
                f"The model kept running past the token limit before finishing its "
                f"JSON (tried up to max_tokens={budget}, ceiling {cap}).\n"
                "  A reasoning (Qwen3/DeepSeek-R1) model spends this budget on its\n"
                "  <think> block before emitting any JSON -- the usual fix is a\n"
                "  larger budget, or disabling thinking so it writes JSON right away.\n"
                "  Raise the budget for THIS call in your active profile:\n"
                "    max_tokens_plan  (pedacito new planning)\n"
                "    max_tokens_step  (gather-loop tool calls)\n"
                "    max_tokens_answer (the final answer / edit reply)\n"
                "  ...and/or set disable_thinking = true. The retry ceiling follows\n"
                "  reserve_tokens / max_tokens_answer, it is not a fixed limit.\n"
                "  NOTE for hosted APIs (DeepSeek/OpenAI): their per-message OUTPUT\n"
                "  cap (~8k tokens on deepseek-chat) is provider-side and cannot be\n"
                "  raised from here -- keep max_tokens_step at/below it and let the\n"
                "  chunked-write protocol split big files across turns instead.\n"
                f"  Last partial reply: {raw[-160:]}")
        raise LLMError(
            f"Model would not produce valid JSON after {retries + 1} tries.\n"
            "  If this is a hosted OpenAI-compatible API (DeepSeek, OpenAI, "
            "...), set\n"
            "  json_mode = \"object\" under [server] -- hosted APIs reject "
            "json_schema.\n"
            "  If it's a local server, the runtime may be ignoring the JSON "
            "schema\n"
            "  (common with MLX builds; llama.cpp/GGUF supports it).\n"
            f"  Last reply: {raw[:300]}")

    def loaded_models(self) -> dict[str, str] | None:
        """{model_id: state} from LM Studio's native /api/v0/models (the
        OpenAI /v1/models lists everything downloaded, not what is resident).
        None when the server lacks the endpoint."""
        base = _server_root(self.cfg.base_url)
        try:
            r = requests.get(base + "/api/v0/models", headers=self.headers, timeout=10)
            if r.status_code != 200:
                return None
            data = r.json().get("data", [])
        except Exception:
            return None
        out = {m.get("id", "?"): m.get("state", "unknown") for m in data if isinstance(m, dict)}
        return out or None

    @staticmethod
    def _pick_model(entries, model: str) -> dict | None:
        """Find the entry for `model` in a server's model list.

        Servers spell the name differently (`id`/`key` with an optional
        `@q4_k_m` variant suffix on LM Studio, `name` on Ollama). Exact matches
        win, then a match ignoring the `@` suffix.
        """
        want = str(model or "").strip().lower()
        if not want:
            return None

        def names(entry: dict) -> list[str]:
            vals = (entry.get("id"), entry.get("key"), entry.get("name"),
                    entry.get("model"), entry.get("model_name"),
                    entry.get("display_name"))
            return [str(v).strip().lower() for v in vals if v]

        for entry in entries:
            if isinstance(entry, dict) and want in names(entry):
                return entry
        base = want.split("@", 1)[0]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if any(n.split("@", 1)[0] == base for n in names(entry)):
                return entry
        return None

    def _absorb_model_list(self, entries, source: str,
                           model: str | None = None) -> dict:
        """Extract size/context from the entry for `model` in a model list.

        Absorbs both extensions of the OpenAI /v1/models shape:
          * LM Studio v1: `size_bytes`, `loaded_instances[].config.context_length`,
            `max_context_length`.
          * Lemonade: `size` in GB, `context_length`, `max_context_window`.
        """
        m = self._pick_model(entries, model or self.cfg.model)
        if not m:
            return {}

        got: dict = {}
        raw_size = m.get("size_bytes")
        if raw_size is not None:
            try:
                got["size_bytes"] = int(raw_size)
            except (TypeError, ValueError):
                pass
        if not got.get("size_bytes") and m.get("size") is not None:
            # Lemonade reports model size as a GB number, not bytes.
            try:
                got["size_bytes"] = int(float(m["size"]) * _GIB)
            except (TypeError, ValueError):
                pass
        loaded = m.get("loaded_instances") or []
        loaded_ctx = None
        if loaded and isinstance(loaded[0], dict):
            loaded_ctx = _positive_int(
                (loaded[0].get("config") or {}).get("context_length"))
        # Lemonade's recipe_options carry the effective ctx_size (with saved
        # overrides); prefer it over the top-level context_length, which some
        # versions leave at the registry default.
        recipe = m.get("recipe_options") or {}
        ctx = (loaded_ctx or _positive_int(recipe.get("ctx_size"))
               or _positive_int(m.get("context_length"))
               or _positive_int(recipe.get("num_ctx")))
        if ctx:
            got["context_length"] = ctx
        max_ctx = _positive_int(m.get("max_context_length")
                                or m.get("max_context_window"))
        if max_ctx:
            got["max_context_length"] = max_ctx
        if got:
            if str(m.get("owned_by", "")).lower() == "lemonade" \
                    or m.get("recipe"):
                source = "lemonade"
            got["source"] = source
        return got

    def _lemonade_health_context(self, root: str,
                                 model: str | None = None) -> tuple[int | None, int | None]:
        """Context a currently loaded Lemonade model was launched with, from its
        health endpoint (the model list's `context_length` can lag behind).
        Tries both prefixes because older builds expose /api/v0/health."""
        for path in ("/api/v1/health", "/api/v0/health"):
            try:
                r = requests.get(root + path, headers=self.headers, timeout=5)
                if r.status_code != 200:
                    continue
                data = r.json()
            except Exception:
                continue
            entries = data.get("all_models_loaded") or []
            entry = self._pick_model(entries, model or self.cfg.model)
            if not entry:
                continue
            opts = entry.get("recipe_options") or {}
            ctx = (_positive_int(opts.get("ctx_size"))
                   or _positive_int(entry.get("context_length")))
            max_ctx = _positive_int(entry.get("max_context_window"))
            if ctx or max_ctx:
                return ctx, max_ctx
        return None, None

    def _lemonade_options_context(self, root: str,
                                  model: str | None = None) -> int | None:
        """The ctx_size Lemonade would use for a load right now, from
        /v1/models/{id}/options (`resolved_ctx_size` covers automatic sizing).
        Absent on older versions, hence the silent failure."""
        url = (root + "/v1/models/" + quote(str(model or self.cfg.model), safe="")
               + "/options")
        try:
            r = requests.get(url, headers=self.headers, timeout=5)
            if r.status_code != 200:
                return None
            data = r.json()
        except Exception:
            return None
        ctx = _positive_int(data.get("resolved_ctx_size"))
        if ctx:
            return ctx
        for key in ("effective", "saved"):
            opts = data.get(key) or {}
            if isinstance(opts, dict):
                ctx = _positive_int(opts.get("ctx_size"))
                if ctx:
                    return ctx
        return None

    def model_info(self, model: str | None = None) -> dict | None:
        """Metadata for a model ID (default `cfg.model`): any of `size_bytes`,
        `context_length`, `max_context_length`, plus `source`; None when nothing
        answers. Used by auto-fit to size the context window from the machine's
        VRAM, and by `models scan` to describe each listed model.

        Tried in order, stopping once both size and context are known:
          * LM Studio v1 `/api/v1/models`
          * OpenAI-compatible `/v1/models` (Lemonade adds `size` in GB,
            `context_length`, `max_context_window`)
          * LM Studio v0 `/api/v0/models` (older builds)
          * Ollama `/api/show` (also implemented by Lemonade)

        Every probe is best-effort; failures just yield nothing.
        """
        model = model or self.cfg.model
        root = _server_root(self.cfg.base_url)
        out: dict = {}

        def enough() -> bool:
            return bool(out.get("size_bytes") and (
                out.get("context_length") or out.get("max_context_length")))

        def merge(got: dict) -> None:
            for key, value in got.items():
                if key == "source":
                    continue
                if value and not out.get(key):
                    out[key] = value
            if got.get("source"):
                out.setdefault("source", got["source"])

        for url, source in (
                (root + "/api/v1/models", "lmstudio-v1"),
                (root + "/v1/models", "openai-models"),
                (root + "/api/v0/models", "lmstudio-v0")):
            if enough():
                break
            try:
                r = requests.get(url, headers=self.headers, timeout=5)
                if r.status_code != 200:
                    continue
                data = r.json()
                if isinstance(data, dict):
                    entries = data.get("models") or data.get("data") or []
                else:
                    entries = data if isinstance(data, list) else []
                merge(self._absorb_model_list(entries, source, model))
            except Exception:
                pass

        # Lemonade's loaded/saved recipe options beat a list entry that may
        # carry the registry default window.
        if out.get("source") == "lemonade":
            health_ctx, health_max = self._lemonade_health_context(root, model)
            if health_max and not out.get("max_context_length"):
                out["max_context_length"] = health_max
            if health_ctx:
                out["context_length"] = health_ctx
                out["context_source"] = "lemonade loaded options"
            else:
                options_ctx = self._lemonade_options_context(root, model)
                if options_ctx:
                    out["context_length"] = options_ctx
                    out["context_source"] = "lemonade saved options"

        # Ollama /api/show reports the weights size and context length.
        if not enough():
            try:
                r = requests.post(root + "/api/show",
                                  json={"name": model}, timeout=5)
                if r.status_code == 200:
                    d = r.json()
                    got: dict = {}
                    size = d.get("size")
                    if size is not None:
                        try:
                            got["size_bytes"] = int(size)
                        except (TypeError, ValueError):
                            pass
                    info = d.get("model_info") or {}
                    params = d.get("parameters") or {}
                    num_ctx = (params.get("num_ctx")
                               if isinstance(params, dict) else None)
                    # Keys are architecture-prefixed (llama., qwen2., gemma.,
                    # ...): accept any *.context_length, taking the largest.
                    max_ctx = max(
                        (v for v in (_positive_int(x)
                                     for k, x in info.items()
                                     if k.endswith(".context_length")) if v),
                        default=0) or None
                    ctx = _positive_int(num_ctx) or max_ctx
                    if ctx:
                        got["context_length"] = ctx
                    if max_ctx:
                        got["max_context_length"] = max_ctx
                    details = d.get("details") or {}
                    if not got.get("size_bytes") and details.get("parameter_size"):
                        got["parameter_size"] = details["parameter_size"]
                    if got:
                        got["source"] = "ollama"
                    merge(got)
            except Exception:
                pass

        if not (out.get("size_bytes") or out.get("context_length")
                or out.get("max_context_length")):
            return None
        return out

    def warmup(self) -> None:
        """Load the model with one tiny generation before real work begins, so
        a load failure or model-switch wait surfaces once, up front."""
        state = self.loaded_models()
        if state is not None:
            resident = [m for m, st in state.items() if st == "loaded"]
            if resident and self.cfg.model not in resident:
                print(f"[server currently has {', '.join(resident)} loaded; "
                      f"requesting '{self.cfg.model}']", file=sys.stderr)
        # LM Studio's native API lists `supported_parameters` per model; if
        # that list omits response_format/json_schema, use plain JSON
        # prompting instead of getting prose back from a "constrained" call.
        # A remembered verdict (server_modes.json) skips the probe.
        if self._schema_supported is None and self.cfg.json_mode != "object":
            cached = _load_mode_cache(self.cfg.base_url, self.cfg.model)
            if cached is not None:
                self._schema_supported = cached
                if not cached:
                    print("[remembered server capability: no constrained "
                          "JSON; using plain JSON prompting]", file=sys.stderr)
            else:
                try:
                    root = _server_root(self.cfg.base_url)
                    r = requests.get(root + "/api/v0/models",
                                     headers=self.headers, timeout=5)
                    if r.status_code == 200:
                        for m in r.json().get("data", []):
                            if not isinstance(m, dict):
                                continue
                            if m.get("id") == self.cfg.model or m.get("key") == self.cfg.model:
                                sp = [str(x).lower() for x in
                                      (m.get("supported_parameters") or [])]
                                if any("response_format" in x or "json_schema" in x
                                       for x in sp):
                                    self._schema_supported = True
                                elif sp:
                                    self._schema_supported = False
                                    print("[server does not support constrained "
                                          "JSON; using plain JSON prompting]",
                                          file=sys.stderr)
                                break
                except Exception:
                    pass   # best-effort probe
                if self._schema_supported is not None:
                    _save_mode_cache(self.cfg.base_url, self.cfg.model,
                                     self._schema_supported)
        t0 = time.time()
        # Include a system message so a template without a system role (Gemma)
        # is discovered HERE, before the first real turn -- a mid-run flip
        # would rewrite the head of the conversation and break the server's
        # prompt cache at the worst moment.
        self.chat([{"role": "system", "content": "You are a terse assistant."},
                   {"role": "user", "content": "Reply with the single word: ok"}],
                  max_tokens=5)
        took = time.time() - t0
        if took > 5:
            print(f"[model ready after {took:.0f}s]", file=sys.stderr)
        # The model is resident now: one cheap request can measure the real
        # chars-per-token so the fit math stops guessing.
        self.calibrate_tokenizer()

    def calibrate_tokenizer(self) -> None:
        """Derive chars-per-token from the server's tokenizer endpoint
        (LM Studio `/api/v0/tokenize`, llama.cpp `/tokenize`). Best-effort:
        with no endpoint the module heuristic (~3.6) stays in effect."""
        sample = _TOKENIZER_SAMPLE
        root = _server_root(self.cfg.base_url)
        for url, payload in (
                (root + "/api/v0/tokenize",
                 {"model": self.cfg.model, "prompt": sample,
                  "add_special_tokens": False}),
                (root + "/tokenize", {"content": sample})):
            try:
                r = requests.post(url, headers=self.headers, json=payload,
                                  timeout=10)
                if r.status_code != 200:
                    continue
                d = r.json()
                n = _positive_int(d.get("count")) or \
                    (len(d["tokens"]) if isinstance(d.get("tokens"), list) else 0)
                if n > 0:
                    cpt = len(sample) / n
                    if 1.0 <= cpt <= 8.0:
                        self.chars_per_token = cpt
                        return
            except Exception:
                continue

    def doctor_prompt_cache(self, delay: float = 2.0) -> str:
        """Send one >= 1.2k-token prompt twice, a few seconds apart, and report
        whether the second call hit a cache: 'observed', 'not observed', or
        'explicit-only' (the provider table says explicit markers are the only
        path). Never 'unsupported': DeepSeek-style caches persist
        asynchronously, so a single miss proves nothing."""
        from .providers import provider_for
        text = "\n".join(
            f"Line {i:03d}: the quick brown fox jumps over the lazy dog."
            for i in range(130))
        msgs = [{"role": "system", "content": text}]
        try:
            self.chat(msgs, max_tokens=8)
            time.sleep(delay)
            before = int(self.stats.get("cached_tokens", 0) or 0)
            self.chat(msgs, max_tokens=8)
        except Exception:
            return "not observed"
        if int(self.stats.get("cached_tokens", 0) or 0) > before:
            return "observed"
        prov = provider_for(self.cfg.base_url, self.cfg.model)
        if prov.cache_mode == "explicit":
            return "explicit-only"
        return "not observed"

    def probe_json_modes(self) -> list[tuple[str, bool, str]]:
        """`pedacito config --doctor`: one tiny request per structured-output
        mode. Returns [(mode, ok, detail)] for json_schema (strict grammar),
        json_object, and plain prompting; `ok` means the reply parsed as the
        expected JSON object. A json_schema verdict is remembered for runs."""
        user = ("Reply with one JSON object only, no prose: "
                '{"action": "done", "file": "", "reasoning": "probe"}')
        results: list[tuple[str, bool, str]] = []
        for mode, fmt in (
                ("json_schema",
                 {"type": "json_schema",
                  "json_schema": {"name": "step", "strict": True,
                                  "schema": _PROBE_SCHEMA}}),
                ("json_object", {"type": "json_object"}),
                ("prompt", None)):
            payload = {
                "model": self.cfg.model,
                "messages": [{"role": "user", "content": user}],
                "temperature": self.cfg.temperature,
                "max_tokens": 120,
                "stream": False,
            }
            if fmt is not None:
                payload["response_format"] = fmt
            try:
                data = self._post(payload).json()
                text = (extract_content(data["choices"][0]["message"]) or "").strip()
                i, j = text.find("{"), text.rfind("}")
                obj = json.loads(text[i:j + 1]) if i != -1 and j > i else None
                if isinstance(obj, dict) and obj.get("action") == "done":
                    results.append((mode, True, "valid JSON object"))
                else:
                    results.append((mode, False,
                                    f"reply was not the expected object: "
                                    f"{text[:80]!r}"))
            except LLMError as e:
                results.append((mode, False, str(e).split("\n")[0][:160]))
            except Exception as e:
                results.append((mode, False, f"{type(e).__name__}: {e}"[:160]))
        if self.cfg.json_mode != "object":
            for mode, ok, _detail in results:
                if mode == "json_schema":
                    self._schema_supported = ok
                    _save_mode_cache(self.cfg.base_url, self.cfg.model, ok)
        return results

    def check(self, warn: bool = True) -> str:
        """Verify the server is reachable and return its comma-joined model ids;
        with `warn`, print a warning if the configured model is not listed."""
        url = self.cfg.base_url.rstrip("/") + "/models"
        try:
            r = requests.get(url, headers=self.headers, timeout=15)
            r.raise_for_status()
        except Exception as e:
            raise LLMError(
                CONNECT_HELP.format(url=self.cfg.base_url) + "\n"
                "  - Set base_url in your config: `pedacito config --init`, then\n"
                "    `pedacito config --check` to test it."
            ) from e
        ids = [m.get("id", "?") for m in r.json().get("data", [])]
        if warn and self.cfg.model not in ids:
            print(f"[warn] model '{self.cfg.model}' not in loaded list: {ids}",
                  file=sys.stderr)
        return ", ".join(ids)
