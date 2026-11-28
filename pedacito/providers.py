"""One provider registry: endpoints, auth shapes, caching quirks and evidence.

The registry serves three jobs that used to live in separate tables:

* the hosted-provider compatibility table (prompt caching, JSON quirks);
* the GUI's quick-add catalog (one stable preset per major provider);
* the capability matrix used by the transports and the GUI.

Matching is deliberately strict.  A configured ``base_url`` matches a catalog
entry only when its hostname equals the entry's host or falls under an
explicit dot boundary (``.example.com``) -- never a substring.  A host such as
``api.openai.com.attacker.test`` therefore stays an unknown custom endpoint.

Capabilities are *evidence based*: ``None`` means "not known", never
"supported".  ``get_capabilities`` returns the built-in catalog evidence and
any live refusals remembered by the running transports; ``probe_capabilities``
sends one explicitly consented, bounded request and caches the result for 24
hours under a key that contains no secret.  Importing this module never opens a
socket.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlparse

from . import fileio

CAPABILITY_CACHE_TTL = 24 * 3600.0
CAPABILITY_CACHE_NAME = "provider_capabilities.json"

# Transports the client knows how to speak.  "gateway" marks a catalog entry
# whose native protocol is NOT implemented here: it only works when the user
# points it at a configured OpenAI-compatible gateway.
TRANSPORTS = ("openai", "anthropic", "gemini", "openai-responses", "gateway")


@dataclass(frozen=True)
class ProviderCapabilities:
    """What a provider/model is *known* to support, per capability.

    Every tri-state field uses ``None`` for unknown.  A catalog entry may
    leave a field unknown even when a particular model would support it.
    """

    native_tools: bool | None = None
    parallel_tools: bool | None = None
    json_schema: bool | None = None
    reasoning_blocks: bool | None = None
    vision: bool | None = None
    cache_mode: str | None = None
    context_tokens: int | None = None
    output_tokens: int | None = None
    idempotency: bool | None = None
    stream_resume: bool | None = None
    probed_at: float = 0.0
    source: str = "unverified"

    def to_dict(self) -> dict:
        """Serializable, redacted metadata (never prompts or credentials)."""
        return {
            "native_tools": self.native_tools,
            "parallel_tools": self.parallel_tools,
            "json_schema": self.json_schema,
            "reasoning_blocks": self.reasoning_blocks,
            "vision": self.vision,
            "cache_mode": self.cache_mode,
            "context_tokens": self.context_tokens,
            "output_tokens": self.output_tokens,
            "idempotency": self.idempotency,
            "stream_resume": self.stream_resume,
            "probed_at": float(self.probed_at or 0.0),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "ProviderCapabilities":
        """Rebuild from a cache entry; unknown keys are ignored."""
        data = data if isinstance(data, dict) else {}
        values = {}
        for name in (
            "native_tools", "parallel_tools", "json_schema",
            "reasoning_blocks", "vision", "cache_mode",
            "context_tokens", "output_tokens", "idempotency",
            "stream_resume",
        ):
            values[name] = data.get(name)
        try:
            probed_at = float(data.get("probed_at") or 0.0)
        except (TypeError, ValueError):
            probed_at = 0.0
        values["probed_at"] = probed_at
        values["source"] = str(data.get("source") or "unverified")
        return cls(**values)


UNKNOWN_CAPABILITIES = ProviderCapabilities()


@dataclass(frozen=True)
class Provider:
    # "implicit" (the provider caches on its own; send nothing extra),
    # "explicit" (opt-in `cache_control` markers), or "unsupported" (this
    # surface cannot cache -- the OpenAI-compatible Anthropic endpoint).
    cache_mode: str
    cache_key_ok: bool = False      # accepts `prompt_cache_key`
    max_markers: int = 0            # explicit-mode breakpoints available
    min_tokens: int = 0             # prompt floor for caching to apply
    ttls: tuple[str, ...] = ()      # supported `cache_control` TTLs
    usage_read: tuple[str, ...] = ()    # cache-read usage field names
    usage_write: tuple[str, ...] = ()   # cache-write usage field names
    # Compatibility quirks (C4): forced json_mode, whether `temperature` must
    # be omitted, whether a forced named tool_choice is accepted.
    json_mode: str = ""
    omit_temperature: bool = False
    native_tool_choice: bool = True
    # Additive registry fields (all defaulted so older construction sites and
    # tests that only pass ``cache_mode`` keep working).
    id: str = ""
    transport: str = "openai"
    auth_kind: str = "bearer"
    base_url_template: str = ""
    capabilities: ProviderCapabilities = field(
        default_factory=lambda: UNKNOWN_CAPABILITIES)
    hosts: tuple[str, ...] = ()
    label: str = ""
    local: bool = False
    gateway_only: bool = False

    @property
    def native(self) -> bool:
        """True when this client speaks the provider's own protocol."""
        return not self.gateway_only and self.transport in (
            "openai", "anthropic", "gemini", "openai-responses")

    @property
    def is_local_runtime(self) -> bool:
        return bool(self.local)


def _caps(**changes) -> ProviderCapabilities:
    """Catalog evidence with a consistent source label."""
    values = {"source": "catalog:provider-docs"}
    values.update(changes)
    return ProviderCapabilities(**values)


def _local_caps(**changes) -> ProviderCapabilities:
    values = {"cache_mode": "implicit", "source": "catalog:local-runtime"}
    values.update(changes)
    return ProviderCapabilities(**values)


# ------------------------------------------------------------------ catalog ---

PROVIDERS: tuple[Provider, ...] = (
    # --- hosted APIs with a directly implemented protocol -------------------
    Provider(
        cache_mode="explicit", max_markers=4, min_tokens=1024,
        ttls=("5m", "1h"),
        usage_read=("cache_read_input_tokens",),
        usage_write=("cache_creation_input_tokens",),
        json_mode="schema", omit_temperature=True, native_tool_choice=True,
        id="anthropic", label="Claude (Anthropic)", transport="anthropic",
        auth_kind="x-api-key", base_url_template="https://api.anthropic.com/v1",
        hosts=("api.anthropic.com",),
        capabilities=_caps(
            native_tools=True, parallel_tools=True, json_schema=False,
            reasoning_blocks=True, vision=True, cache_mode="explicit",
            context_tokens=200_000),
    ),
    Provider(
        cache_mode="implicit", cache_key_ok=True, min_tokens=1024,
        id="openai", label="OpenAI", transport="openai",
        auth_kind="bearer", base_url_template="https://api.openai.com/v1",
        hosts=("api.openai.com",),
        capabilities=_caps(
            native_tools=True, parallel_tools=True, json_schema=True,
            reasoning_blocks=True, vision=True, cache_mode="implicit",
            context_tokens=400_000),
    ),
    Provider(
        cache_mode="implicit",
        id="deepseek", label="DeepSeek", transport="openai",
        auth_kind="bearer", base_url_template="https://api.deepseek.com/v1",
        hosts=("api.deepseek.com",),
        capabilities=_caps(
            native_tools=True, json_schema=False, reasoning_blocks=True,
            cache_mode="implicit", context_tokens=128_000),
    ),
    Provider(
        cache_mode="explicit", max_markers=4, min_tokens=1024, ttls=("5m",),
        usage_read=("cache_read_input_tokens", "cached_tokens"),
        usage_write=("cache_creation_input_tokens", "cache_write_tokens"),
        id="qwen", label="Qwen (DashScope)", transport="openai",
        auth_kind="bearer",
        base_url_template=(
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
        hosts=("dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com",
               ".aliyuncs.com"),
        capabilities=_caps(
            native_tools=True, reasoning_blocks=True, vision=True,
            cache_mode="explicit", context_tokens=256_000),
    ),
    Provider(
        cache_mode="implicit",
        usage_read=("cachedContentTokenCount", "cached_tokens"),
        id="gemini", label="Google Gemini", transport="gemini",
        auth_kind="x-goog-api-key",
        base_url_template="https://generativelanguage.googleapis.com/v1beta",
        hosts=("generativelanguage.googleapis.com",),
        capabilities=_caps(
            native_tools=True, parallel_tools=True, json_schema=True,
            reasoning_blocks=True, vision=True, cache_mode="implicit",
            context_tokens=1_000_000),
    ),
    Provider(
        cache_mode="implicit",
        id="openrouter", label="OpenRouter", transport="openai",
        auth_kind="bearer", base_url_template="https://openrouter.ai/api/v1",
        hosts=("openrouter.ai",),
        capabilities=_caps(
            native_tools=True, reasoning_blocks=True, vision=True,
            cache_mode="implicit", context_tokens=200_000),
    ),
    Provider(
        cache_mode="implicit",
        id="groq", label="Groq", transport="openai",
        auth_kind="bearer", base_url_template="https://api.groq.com/openai/v1",
        hosts=("api.groq.com",),
        capabilities=_caps(
            native_tools=True, reasoning_blocks=True, cache_mode="implicit"),
    ),
    Provider(
        cache_mode="implicit",
        id="mistral", label="Mistral", transport="openai",
        auth_kind="bearer", base_url_template="https://api.mistral.ai/v1",
        hosts=("api.mistral.ai",),
        capabilities=_caps(
            native_tools=True, reasoning_blocks=True, vision=True,
            cache_mode="implicit"),
    ),
    Provider(
        cache_mode="implicit",
        id="moonshot", label="Moonshot / Kimi", transport="openai",
        auth_kind="bearer", base_url_template="https://api.moonshot.cn/v1",
        hosts=("api.moonshot.cn", "api.moonshot.ai"),
        capabilities=_caps(
            native_tools=True, reasoning_blocks=True, cache_mode="implicit"),
    ),
    Provider(
        cache_mode="implicit",
        id="xai", label="xAI", transport="openai",
        auth_kind="bearer", base_url_template="https://api.x.ai/v1",
        hosts=("api.x.ai",),
        capabilities=_caps(
            native_tools=True, reasoning_blocks=True, vision=True,
            cache_mode="implicit"),
    ),
    Provider(
        cache_mode="implicit",
        id="together", label="Together", transport="openai",
        auth_kind="bearer", base_url_template="https://api.together.xyz/v1",
        hosts=("api.together.xyz",),
        capabilities=_caps(
            native_tools=True, cache_mode="implicit"),
    ),
    Provider(
        cache_mode="implicit",
        id="cerebras", label="Cerebras", transport="openai",
        auth_kind="bearer", base_url_template="https://api.cerebras.ai/v1",
        hosts=("api.cerebras.ai",),
        capabilities=_caps(
            native_tools=True, reasoning_blocks=True, cache_mode="implicit"),
    ),
    Provider(
        cache_mode="implicit",
        id="fireworks", label="Fireworks", transport="openai",
        auth_kind="bearer", base_url_template="https://api.fireworks.ai/inference/v1",
        hosts=("api.fireworks.ai",),
        capabilities=_caps(
            native_tools=True, reasoning_blocks=True, cache_mode="implicit"),
    ),
    Provider(
        cache_mode="implicit",
        id="minimax", label="MiniMax", transport="openai",
        auth_kind="bearer", base_url_template="https://api.minimax.io/v1",
        hosts=("api.minimax.io", "api.minimaxi.com", "api.minimax.chat"),
        capabilities=_caps(
            native_tools=True, reasoning_blocks=True, cache_mode="implicit"),
    ),
    Provider(
        cache_mode="implicit",
        id="zai", label="Z.ai / GLM", transport="openai",
        auth_kind="bearer", base_url_template="https://api.z.ai/api/paas/v4",
        hosts=("api.z.ai",),
        capabilities=_caps(
            native_tools=True, reasoning_blocks=True, cache_mode="implicit"),
    ),
    Provider(
        cache_mode="implicit",
        id="nvidia", label="NVIDIA NIM", transport="openai",
        auth_kind="bearer",
        base_url_template="https://integrate.api.nvidia.com/v1",
        hosts=("integrate.api.nvidia.com",),
        capabilities=_caps(
            native_tools=True, cache_mode="implicit"),
    ),
    Provider(
        cache_mode="implicit",
        id="huggingface", label="Hugging Face", transport="openai",
        auth_kind="bearer", base_url_template="https://router.huggingface.co/v1",
        hosts=("router.huggingface.co", "api-inference.huggingface.co"),
        capabilities=_caps(cache_mode="implicit"),
    ),
    Provider(
        cache_mode="implicit",
        id="deepinfra", label="DeepInfra", transport="openai",
        auth_kind="bearer", base_url_template="https://api.deepinfra.com/v1/openai",
        hosts=("api.deepinfra.com",),
        capabilities=_caps(
            native_tools=True, cache_mode="implicit"),
    ),
    Provider(
        cache_mode="implicit",
        id="nebius", label="Nebius", transport="openai",
        auth_kind="bearer", base_url_template="https://api.studio.nebius.ai/v1",
        hosts=("api.studio.nebius.ai",),
        capabilities=_caps(
            native_tools=True, cache_mode="implicit"),
    ),
    Provider(
        cache_mode="implicit",
        id="sambanova", label="SambaNova", transport="openai",
        auth_kind="bearer", base_url_template="https://api.sambanova.ai/v1",
        hosts=("api.sambanova.ai",),
        capabilities=_caps(
            native_tools=True, cache_mode="implicit"),
    ),
    Provider(
        cache_mode="implicit",
        id="hyperbolic", label="Hyperbolic", transport="openai",
        auth_kind="bearer", base_url_template="https://api.hyperbolic.xyz/v1",
        hosts=("api.hyperbolic.xyz",),
        capabilities=_caps(
            native_tools=True, cache_mode="implicit"),
    ),
    Provider(
        cache_mode="implicit",
        id="lambda", label="Lambda", transport="openai",
        auth_kind="bearer", base_url_template="https://api.lambdalabs.com/v1",
        hosts=("api.lambdalabs.com",),
        capabilities=_caps(cache_mode="implicit"),
    ),
    Provider(
        cache_mode="implicit",
        id="perplexity", label="Perplexity", transport="openai",
        auth_kind="bearer", base_url_template="https://api.perplexity.ai",
        hosts=("api.perplexity.ai",),
        capabilities=_caps(
            native_tools=False, json_schema=False, reasoning_blocks=True,
            vision=False),
    ),
    Provider(
        cache_mode="implicit",
        id="litellm", label="LiteLLM gateway", transport="openai",
        auth_kind="bearer", base_url_template="http://127.0.0.1:4000/v1",
        hosts=(),
        capabilities=_caps(source="catalog:gateway-placeholder"),
    ),
    # --- gateway-only catalog entries: native auth is deliberately NOT
    # claimed.  These work only behind a configured OpenAI-compatible
    # gateway, and say so in every capability source. -----------------------
    Provider(
        cache_mode="implicit", id="azure-openai", label="Azure OpenAI",
        transport="gateway", auth_kind="gateway",
        base_url_template=(
            "https://{resource}.openai.azure.com/openai/deployments/"
            "{deployment}"),
        hosts=(".openai.azure.com", ".cognitiveservices.azure.com"),
        gateway_only=True,
        capabilities=_caps(source=(
            "unverified: requires configured OpenAI-compatible gateway")),
    ),
    Provider(
        cache_mode="implicit", id="aws-bedrock", label="AWS Bedrock",
        transport="gateway", auth_kind="gateway",
        base_url_template=(
            "https://bedrock-runtime.{region}.amazonaws.com"),
        hosts=("bedrock-runtime.*.amazonaws.com", "bedrock.*.amazonaws.com"),
        gateway_only=True,
        capabilities=_caps(source=(
            "unverified: requires configured OpenAI-compatible gateway")),
    ),
    Provider(
        cache_mode="implicit", id="google-vertex", label="Google Vertex AI",
        transport="gateway", auth_kind="gateway",
        base_url_template=(
            "https://{region}-aiplatform.googleapis.com/v1/projects/"
            "{project}/locations/{location}/endpoints/openapi"),
        hosts=(".aiplatform.googleapis.com",),
        gateway_only=True,
        capabilities=_caps(source=(
            "unverified: requires configured OpenAI-compatible gateway")),
    ),
    # --- local runtimes ----------------------------------------------------
    Provider(
        cache_mode="implicit", id="lmstudio", label="LM Studio",
        transport="openai", auth_kind="optional-bearer",
        base_url_template="http://127.0.0.1:1234/v1", hosts=(), local=True,
        capabilities=_local_caps()),
    Provider(
        cache_mode="implicit", id="ollama", label="Ollama",
        transport="openai", auth_kind="none",
        base_url_template="http://127.0.0.1:11434/v1", hosts=(), local=True,
        capabilities=_local_caps()),
    Provider(
        cache_mode="implicit", id="llamacpp", label="llama.cpp",
        transport="openai", auth_kind="none",
        base_url_template="http://127.0.0.1:8080/v1", hosts=(), local=True,
        capabilities=_local_caps()),
    Provider(
        cache_mode="implicit", id="vllm", label="vLLM",
        transport="openai", auth_kind="optional-bearer",
        base_url_template="http://127.0.0.1:8000/v1", hosts=(), local=True,
        capabilities=_local_caps()),
    Provider(
        cache_mode="implicit", id="lemonade", label="Lemonade",
        transport="openai", auth_kind="optional-bearer",
        base_url_template="http://127.0.0.1:8000/api/v1", hosts=(), local=True,
        capabilities=_local_caps()),
    Provider(
        cache_mode="implicit", id="lemond", label="Lemonade (lemond)",
        transport="openai", auth_kind="optional-bearer",
        base_url_template="http://127.0.0.1:8000/api/v1", hosts=(), local=True,
        capabilities=_local_caps()),
    Provider(
        cache_mode="implicit", id="strata", label="Strata",
        transport="openai", auth_kind="optional-bearer",
        base_url_template="http://127.0.0.1:8080/v1", hosts=(), local=True,
        capabilities=_local_caps()),
    Provider(
        cache_mode="implicit", id="avalon", label="Avalon",
        transport="openai", auth_kind="optional-bearer",
        base_url_template="http://127.0.0.1:8080/v1", hosts=(), local=True,
        capabilities=_local_caps()),
    Provider(
        cache_mode="implicit", id="mlx-lm", label="MLX-LM",
        transport="openai", auth_kind="none",
        base_url_template="http://127.0.0.1:8080/v1", hosts=(), local=True,
        capabilities=_local_caps()),
    Provider(
        cache_mode="implicit", id="sglang", label="SGLang",
        transport="openai", auth_kind="none",
        base_url_template="http://127.0.0.1:30000/v1", hosts=(), local=True,
        capabilities=_local_caps()),
    Provider(
        cache_mode="implicit", id="tabbyapi", label="TabbyAPI",
        transport="openai", auth_kind="optional-bearer",
        base_url_template="http://127.0.0.1:5000/v1", hosts=(), local=True,
        capabilities=_local_caps()),
    Provider(
        cache_mode="implicit", id="koboldcpp", label="KoboldCpp",
        transport="openai", auth_kind="none",
        base_url_template="http://127.0.0.1:5001/v1", hosts=(), local=True,
        capabilities=_local_caps()),
    Provider(
        cache_mode="implicit", id="localai", label="LocalAI",
        transport="openai", auth_kind="optional-bearer",
        base_url_template="http://127.0.0.1:8080/v1", hosts=(), local=True,
        capabilities=_local_caps()),
)

_PROVIDERS: dict[str, Provider] = {}
for _provider in PROVIDERS:
    for _pattern in _provider.hosts:
        _PROVIDERS.setdefault(_pattern, _provider)

_BY_ID: dict[str, Provider] = {p.id: p for p in PROVIDERS if p.id}

_PROVIDER_ID_ALIASES = {
    "azure": "azure-openai",
    "azure_openai": "azure-openai",
    "bedrock": "aws-bedrock",
    "aws": "aws-bedrock",
    "vertex": "google-vertex",
    "vertex-ai": "google-vertex",
    "google": "gemini",
    "dashscope": "qwen",
    "alibaba": "qwen",
    "kimi": "moonshot",
    "moonshot-ai": "moonshot",
    "glm": "zai",
    "z-ai": "zai",
    "nim": "nvidia",
    "hf": "huggingface",
    "hugging-face": "huggingface",
    "llama.cpp": "llamacpp",
    "lm-studio": "lmstudio",
    "mlx": "mlx-lm",
}

_LOCAL_FALLBACK = Provider(
    cache_mode="implicit", id="local", label="Local server",
    transport="openai", auth_kind="none", local=True,
    capabilities=_local_caps())

# Local runtime labels accepted by `config._endpoint_is_local`.  Matching is
# exact after normalisation, never a substring: "my-ollama-proxy" must not
# become local merely because it mentions a runtime.
_LOCAL_LABEL_ALIASES = {
    "local": "local", "localmoe": "local", "local-moe": "local",
    "llamacpp": "llamacpp", "llama.cpp": "llamacpp", "llamacpprouter":
    "llamacpp", "llama.cpp-router": "llamacpp",
    "lemonade": "lemonade", "lemond": "lemond",
    "vllm": "vllm", "lmstudio": "lmstudio", "lm-studio": "lmstudio",
    "ollama": "ollama", "strata": "strata", "avalon": "avalon",
    "mlx": "mlx-lm", "mlxlm": "mlx-lm", "mlx-lm": "mlx-lm",
    "sglang": "sglang", "tabbyapi": "tabbyapi", "tabby": "tabbyapi",
    "koboldcpp": "koboldcpp", "kobold": "koboldcpp",
    "localai": "localai",
}

# The GUI's long-standing quick-add list.  Keeping this subset stable avoids
# surprising existing profiles while the full registry above stays available
# to discovery, capability probing and documentation.
_PRESET_IDS = ("anthropic", "openai", "deepseek", "qwen", "gemini",
               "openrouter", "strata")
_PRESET_META = {
    "anthropic": {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY",
                  "suggest": "claude", "context_tokens": 200000,
                  "models": ["claude-sonnet-4-5", "claude-haiku-4-5",
                             "claude-opus-4-5", "claude-sonnet-4-6",
                             "claude-opus-4-6"]},
    "openai": {"provider": "openai", "api_key_env": "OPENAI_API_KEY",
               "label": "ChatGPT (OpenAI)",
               "suggest": "openai", "context_tokens": 400000,
               "models": ["gpt-5.4", "gpt-5.4-mini", "o4-mini", "o3",
                          "o3-mini", "gpt-4.1", "gpt-4.1-mini"]},
    "deepseek": {"provider": "openai", "api_key_env": "DEEPSEEK_API_KEY",
                 "suggest": "deepseek", "context_tokens": 128000,
                 "models": ["deepseek-chat", "deepseek-reasoner", "deepseek-v4",
                            "deepseek-v4-flash"]},
    "qwen": {"provider": "openai", "api_key_env": "DASHSCOPE_API_KEY",
             "suggest": "qwen", "context_tokens": 256000,
             "models": ["qwen3-coder-plus", "qwen3.8-flash", "qwen-max",
                        "qwen-plus", "qwen-flash"]},
    "gemini": {"provider": "gemini", "api_key_env": "GEMINI_API_KEY",
               "suggest": "gemini", "context_tokens": 1000000,
               "models": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-3-pro",
                          "gemini-3-flash"]},
    "openrouter": {"provider": "openai", "api_key_env": "OPENROUTER_API_KEY",
                   "suggest": "openrouter", "context_tokens": 200000,
                   "models": ["deepseek/deepseek-chat", "qwen/qwen3-coder",
                              "anthropic/claude-sonnet-4-5",
                              "google/gemini-2.5-pro"]},
    "strata": {"provider": "strata", "api_key_env": "STRATA_API_KEY",
               "label": "Strata / Avalon (Local MoE)",
               "suggest": "strata", "context_tokens": 131072, "models": []},
}


# --------------------------------------------------------------- matching ----

def _host(base_url: str) -> str:
    try:
        return (urlparse(str(base_url or "")).hostname or "").lower() \
            .rstrip(".")
    except ValueError:
        return ""


def normalize_base_url(base_url: str) -> str:
    """Normalize a base URL for identity comparisons (never for auth).

    Delegates to the project's discovery normalizer so trailing slashes, a
    ``/v1`` suffix and ``localhost`` spellings of one endpoint share one cache
    key.  Falls back to a local normalizer if discovery cannot be imported.
    """
    raw = str(base_url or "").strip()
    if not raw:
        return ""
    try:
        from . import discovery
        return discovery.normalize_url(raw)
    except Exception:
        pass
    try:
        parsed = urlparse(raw)
    except ValueError:
        return raw.rstrip("/")
    if parsed.scheme and parsed.netloc:
        host = (parsed.hostname or "").lower().rstrip(".")
        port = ""
        try:
            if parsed.port and not (
                    (parsed.scheme == "https" and parsed.port == 443)
                    or (parsed.scheme == "http" and parsed.port == 80)):
                port = f":{parsed.port}"
        except ValueError:
            port = ""
        path = (parsed.path or "").rstrip("/")
        return f"{parsed.scheme.lower()}://{host}{port}{path}"
    return raw.rstrip("/")


def host_matches(host: str, pattern: str) -> bool:
    """True for hostname equality, a dot-boundary suffix, or a ``*`` label.

    ``api.openai.com`` matches only the exact host or a catalog pattern like
    ``.openai.com`` / ``*.openai.com``.  It never matches
    ``api.openai.com.attacker.test`` or ``notapi.openai.com``.
    """
    h = str(host or "").strip().lower().rstrip(".")
    p = str(pattern or "").strip().lower().rstrip(".")
    if not h or not p:
        return False
    if p.startswith("."):
        return h.endswith(p) and len(h) > len(p)
    if p.startswith("*."):
        suffix = p[1:]
        return h.endswith(suffix) and len(h) > len(suffix)
    if "*" in p:
        h_labels = h.split(".")
        p_labels = p.split(".")
        return len(h_labels) == len(p_labels) and all(
            expected == "*" or expected == actual
            for expected, actual in zip(p_labels, h_labels))
    return h == p


def _provider_matches(provider: Provider, host: str) -> bool:
    return any(host_matches(host, pattern) for pattern in provider.hosts)


def provider_for(base_url: str, model: str = "") -> Provider:
    """The registry entry whose host matches `base_url`, else a safe fallback.

    `model` stays reserved for future per-model entries.  Unknown endpoints get
    a generic implicit-cache entry: nothing is forced on an API we do not know.
    """
    host = _host(base_url)
    if host:
        for provider in PROVIDERS:
            if provider.hosts and _provider_matches(provider, host):
                return provider
    return Provider(cache_mode="implicit", label="", id="", transport="openai")


def provider_by_id(provider_id: str) -> Provider | None:
    key = str(provider_id or "").strip().lower()
    key = _PROVIDER_ID_ALIASES.get(key, key)
    return _BY_ID.get(key)


def _provider_for_cfg(base_url: str, model: str, cfg) -> Provider:
    """Host match first, then an explicit provider label (local gateways)."""
    provider = provider_for(base_url, model)
    if cfg is not None:
        label = str(getattr(cfg, "provider", "") or "").strip().lower()
        responses = (label == "openai-responses"
                     or bool(getattr(cfg, "openai_responses", False)))
        if responses and provider.transport != "openai-responses":
            base = provider if provider.id else _BY_ID.get("openai", provider)
            return replace(base, transport="openai-responses")
        if not provider.id and label:
            explicit = provider_by_id(label)
            if explicit is not None:
                return explicit
    return provider


def catalog() -> tuple[Provider, ...]:
    """The complete registry (hosted, gateway placeholders and local runtimes)."""
    return PROVIDERS


def hosted_provider_ids(*, include_gateway: bool = False) -> tuple[str, ...]:
    """Distinct hosted provider ids; gateway-only placeholders excluded by default."""
    return tuple(
        p.id for p in PROVIDERS
        if p.id and not p.local and (include_gateway or not p.gateway_only))


def gateway_provider_ids() -> tuple[str, ...]:
    return tuple(p.id for p in PROVIDERS if p.gateway_only)


def local_runtime_ids() -> tuple[str, ...]:
    return tuple(p.id for p in PROVIDERS if p.local)


def local_runtimes() -> tuple[Provider, ...]:
    return tuple(p for p in PROVIDERS if p.local)


def _normalize_runtime_label(label: str) -> str:
    return str(label or "").strip().lower()


def local_runtime_for_label(label: str) -> Provider | None:
    """The local-runtime entry for an exact provider label, else None.

    An explicit ``local`` / ``local-*`` prefix is the user declaring the
    endpoint as self-hosted; a name that merely contains a runtime word
    (``my-ollama-proxy``) is not.
    """
    key = _normalize_runtime_label(label)
    if not key:
        return None
    if key == "local" or key.startswith(("local-", "local_", "local ")):
        return _LOCAL_FALLBACK
    key = key.replace("_", "-")
    runtime_id = _LOCAL_LABEL_ALIASES.get(key)
    if runtime_id is None:
        # Accept the compact spellings users already have in configs.
        compact = (key.replace("-", "").replace(".", "")
                   .replace(" ", ""))
        runtime_id = _LOCAL_LABEL_ALIASES.get(compact)
    if runtime_id is None:
        folded = key.replace("_", "").replace(" ", "")
        runtime_id = _LOCAL_LABEL_ALIASES.get(folded)
    if runtime_id is None:
        return None
    if runtime_id == "local":
        return _LOCAL_FALLBACK
    return _BY_ID.get(runtime_id)


def is_local_runtime_label(label: str) -> bool:
    return local_runtime_for_label(label) is not None


def provider_presets() -> tuple[dict, ...]:
    """The quick-add catalog consumed by the GUI (one source of truth)."""
    presets: list[dict] = []
    for preset_id in _PRESET_IDS:
        provider = _BY_ID[preset_id]
        meta = _PRESET_META[preset_id]
        presets.append({
            "id": provider.id,
            "label": meta.get("label") or provider.label,
            "provider": meta["provider"],
            "base_url": provider.base_url_template,
            "api_key_env": meta["api_key_env"],
            "suggest": meta["suggest"],
            "context_tokens": meta["context_tokens"],
            "models": list(meta["models"]),
        })
    return tuple(presets)


# --------------------------------------------------------------- caching ----

_CAPABILITY_CACHE: dict[str, dict] = {}
_CAPABILITY_CACHE_LOCK = threading.Lock()


def capability_cache_path() -> Path:
    """`provider_capabilities.json` beside config.toml (test-isolation aware)."""
    try:
        from . import config as config_mod
        cur_dir = Path(config_mod.user_config_dir())
        default_dir = Path(config_mod._user_config_dir())
        if cur_dir != default_dir:
            return cur_dir / CAPABILITY_CACHE_NAME
        if config_mod.USER_CONFIG.parent != default_dir:
            return config_mod.USER_CONFIG.parent / CAPABILITY_CACHE_NAME
        return cur_dir / CAPABILITY_CACHE_NAME
    except Exception:
        try:
            from . import config as config_mod
            return config_mod.USER_CONFIG.parent / CAPABILITY_CACHE_NAME
        except Exception:
            return Path.home() / ".config" / "pedacito" / CAPABILITY_CACHE_NAME


def _auth_identity(base_url: str, provider: Provider, cfg) -> str:
    """A stable hash of *identity* metadata, never of a secret."""
    parts = [provider.id or _host(base_url), provider.auth_kind,
             provider.transport]
    if cfg is not None:
        parts.append(str(getattr(cfg, "api_key_env", "") or ""))
        parts.append(str(getattr(cfg, "api_key_keyring", "") or ""))
        raw_key = str(getattr(cfg, "api_key", "") or "")
        parts.append("inline" if raw_key else "")
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]


def _cache_key(base_url: str, model: str, auth_hash: str) -> str:
    payload = "\0".join((normalize_base_url(base_url), str(model or ""),
                         auth_hash))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_cache() -> dict:
    try:
        path = capability_cache_path()
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    entries = data.get("entries")
    return entries if isinstance(entries, dict) else {}


def _write_cache(entries: dict) -> None:
    payload = json.dumps({"version": 1, "entries": entries}, indent=1,
                         sort_keys=True)
    try:
        fileio.write_bytes(capability_cache_path(), payload.encode("utf-8"))
    except OSError:
        pass


def _cache_load() -> None:
    with _CAPABILITY_CACHE_LOCK:
        if not _CAPABILITY_CACHE:
            _CAPABILITY_CACHE.update(_read_cache())


def _cache_get(key: str, *, now: float | None = None) -> ProviderCapabilities | None:
    _cache_load()
    now = time.time() if now is None else float(now)
    with _CAPABILITY_CACHE_LOCK:
        entry = _CAPABILITY_CACHE.get(key)
        if not isinstance(entry, dict):
            return None
        probed_at = entry.get("probed_at")
        try:
            stamp = float(probed_at if probed_at is not None
                          else entry.get("ts") or 0.0)
        except (TypeError, ValueError):
            stamp = 0.0
        if stamp <= 0 or now - stamp > CAPABILITY_CACHE_TTL:
            _CAPABILITY_CACHE.pop(key, None)
            return None
        return ProviderCapabilities.from_dict(entry.get("capabilities"))


def _cache_put(key: str, base_url: str, model: str, auth_hash: str,
               capabilities: ProviderCapabilities) -> None:
    _cache_load()
    entry = {
        "url": normalize_base_url(base_url),
        "model": str(model or ""),
        "auth_hash": auth_hash,
        "probed_at": float(capabilities.probed_at or time.time()),
        "capabilities": capabilities.to_dict(),
    }
    with _CAPABILITY_CACHE_LOCK:
        _CAPABILITY_CACHE[key] = entry
        snapshot = dict(_CAPABILITY_CACHE)
    _write_cache(snapshot)


def clear_capability_cache() -> None:
    """Drop the in-process cache (the on-disk cache is left untouched)."""
    with _CAPABILITY_CACHE_LOCK:
        _CAPABILITY_CACHE.clear()


# ------------------------------------------------ runtime compatibility -----

def capabilities_from_json_caps(caps: dict | None) -> dict:
    """Adapter for ``llm._JSON_CAPS``: merge only observed refusals."""
    caps = caps if isinstance(caps, dict) else {}
    out: dict = {}
    if caps.get("schema") is False:
        out["json_schema"] = False
    if caps.get("object") is False:
        out["json_object"] = False
    return out


def capabilities_from_accel_caps(caps: dict | None) -> dict:
    """Adapter for ``llm._ACCEL_CAPS``: merge only observed refusals."""
    caps = caps if isinstance(caps, dict) else {}
    out: dict = {}
    if caps.get("cache") is False:
        out["prompt_cache_accel"] = False
    if caps.get("lookup") is False:
        out["prompt_lookup"] = False
    return out


def _runtime_evidence(base_url: str, model: str) -> dict:
    try:
        from . import llm
        return dict(llm.runtime_capability_evidence(base_url, model) or {})
    except Exception:
        return {}


def _merge_evidence(caps: ProviderCapabilities, evidence: dict) -> tuple[
        ProviderCapabilities, bool]:
    """Apply live evidence without ever upgrading unknown to supported."""
    changes: dict = {}
    used = False
    if evidence.get("json_schema") is False:
        changes["json_schema"] = False
        used = True
    if evidence.get("json_object") is False:
        used = True
    if evidence.get("prompt_cache_accel") is False:
        used = True
    if evidence.get("prompt_lookup") is False:
        used = True
    if not used:
        return caps, False
    source = caps.source
    if "runtime" not in source:
        source = f"{source}+runtime"
    changes["source"] = source
    return replace(caps, **changes), True


def _model_catalog_metadata(base_url: str, model: str) -> dict:
    try:
        from . import model_catalog
        return model_catalog.lookup_model_meta(base_url, model,
                                               allow_fetch=False) or {}
    except Exception:
        return {}


def _static_capabilities(base_url: str, model: str,
                         provider: Provider) -> ProviderCapabilities:
    caps = provider.capabilities
    meta = _model_catalog_metadata(base_url, model)
    changes: dict = {}
    ctx = meta.get("context_tokens")
    if isinstance(ctx, int) and ctx > 0:
        changes["context_tokens"] = ctx
    out = meta.get("output_tokens")
    if isinstance(out, int) and out > 0:
        changes["output_tokens"] = out
    if changes:
        source = caps.source
        if "model_catalog" not in source:
            source = f"{source}+model_catalog"
        changes["source"] = source
        caps = replace(caps, **changes)
    return caps


def _auth_headers(provider: Provider, cfg) -> dict:
    try:
        from . import llm
        key = str(llm._api_key(cfg) or "") if cfg is not None else ""
    except Exception:
        key = ""
    if provider.auth_kind == "x-api-key":
        from . import transport_anthropic
        return transport_anthropic.build_headers(key)
    if provider.auth_kind == "x-goog-api-key":
        from . import transport_gemini
        return transport_gemini.build_headers(key)
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _probe_request(provider: Provider, base_url: str, model: str,
                   cfg) -> tuple[str, dict, dict]:
    """A bounded, minimal probe request for one transport."""
    prompt = "ping"
    if provider.transport == "anthropic":
        from . import transport_anthropic
        url = transport_anthropic.messages_endpoint(base_url)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "stream": False,
        }
    elif provider.transport == "gemini":
        from . import transport_gemini
        url = transport_gemini.endpoint(base_url, model)
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 1},
        }
    elif provider.transport == "openai-responses":
        from . import transport_openai
        url = transport_openai.responses_endpoint(base_url)
        payload = {
            "model": model,
            "input": [{"type": "message", "role": "user",
                       "content": [{"type": "input_text", "text": prompt}]}],
            "max_output_tokens": 4,
            "stream": False,
        }
    else:
        url = (base_url or "").rstrip("/") + "/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4,
            "stream": False,
        }
        # Ask for a tiny strict schema and one harmless tool.  A refusal is
        # evidence the surface does not accept them; a 200 alone is not proof
        # that the model will use them, so only positive tool calls upgrade
        # native_tools.
        if provider.capabilities.json_schema is not False:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "pedacito_probe",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"],
                        "additionalProperties": False,
                    },
                },
            }
        if provider.capabilities.native_tools is not False:
            payload["tools"] = [{
                "type": "function",
                "function": {
                    "name": "pedacito_probe",
                    "description": "Probe tool availability.",
                    "parameters": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"],
                        "additionalProperties": False,
                    },
                },
            }]
            payload["tool_choice"] = "auto"
    return url, payload, _auth_headers(provider, cfg)


def _probe_failure(caps: ProviderCapabilities, reason: str) -> ProviderCapabilities:
    source = caps.source
    if "probe" not in source:
        source = f"{source}+probe:{reason}"
    return replace(caps, source=source)


def probe_capabilities(base_url: str, model: str, *, cfg,
                       deadline: float) -> ProviderCapabilities:
    """One explicit, bounded capability probe.

    The caller is responsible for having obtained the user's consent; this
    function never runs by itself and never follows redirects (credentials must
    not cross a redirect boundary).  A failure is reported as an unverified
    source instead of breaking the caller's run.
    """
    provider = _provider_for_cfg(base_url, model, cfg)
    static = _static_capabilities(base_url, model, provider)
    try:
        now = time.monotonic()
    except Exception:
        now = 0.0
    try:
        target = float(deadline)
        # Callers in this codebase pass a monotonic deadline; accept an epoch
        # timestamp too rather than treating it as already expired.
        remaining = (target - time.time()) if target > 1_000_000_000 \
            else (target - now)
    except (TypeError, ValueError):
        remaining = 15.0
    if remaining <= 0:
        return _probe_failure(static, "deadline")
    if provider.gateway_only:
        return _probe_failure(static, "gateway-only")
    if not str(base_url or "").strip():
        return _probe_failure(static, "no-url")

    # Existing outbound secret scan (the probe prompt is fixed, but keep the
    # same discipline as every outbound path).
    try:
        from . import security
        findings = security.scan_messages_for_secrets(
            [{"role": "user", "content": "ping"}])
        if findings:
            return _probe_failure(static, "blocked-secrets")
    except Exception:
        pass

    url, payload, headers = _probe_request(provider, base_url, model, cfg)
    if not url:
        return _probe_failure(static, "no-url")
    timeout = max(0.1, min(15.0, remaining))
    try:
        import requests
        response = requests.post(
            url, headers=headers, json=payload,
            timeout=(min(5.0, timeout), timeout), allow_redirects=False)
    except Exception as exc:
        name = type(exc).__name__.lower()
        reason = "timeout" if "timeout" in name else "unreachable"
        return _probe_failure(static, reason)

    status = int(getattr(response, "status_code", 0) or 0)
    if status in (301, 302, 303, 307, 308):
        return _probe_failure(static, "redirect-refused")
    if status in (401, 403):
        return _probe_failure(static, "auth-failed")
    if status >= 400:
        body = str(getattr(response, "text", "") or "")[:1200].lower()
        changes: dict = {}
        if "response_format" in body or "json_schema" in body \
                or "schema" in body and "unsupported" in body:
            changes["json_schema"] = False
        if "tool" in body and ("unsupported" in body or "unknown" in body
                               or "not supported" in body):
            changes["native_tools"] = False
        if changes:
            caps = replace(static, **changes)
            return _probe_failure(caps, f"http-{status}")
        return _probe_failure(static, f"http-{status}")
    if status != 200:
        return _probe_failure(static, f"http-{status}")

    probed = replace(static, probed_at=time.time(), source="probe")
    try:
        data = response.json()
    except Exception:
        data = None
    if isinstance(data, dict):
        text, reasoning, tool_calls = _probe_response_parts(
            provider.transport, data)
        changes = {}
        if provider.transport == "openai" and isinstance(
                payload.get("response_format"), dict) and text:
            changes["json_schema"] = True
        if tool_calls:
            changes["native_tools"] = True
            if len(tool_calls) > 1:
                changes["parallel_tools"] = True
        if reasoning:
            changes["reasoning_blocks"] = True
        if changes:
            probed = replace(probed, **changes)
    auth_hash = _auth_identity(base_url, provider, cfg)
    _cache_put(_cache_key(base_url, model, auth_hash),
               base_url, model, auth_hash, probed)
    return probed


def _probe_response_parts(transport: str, data: dict) -> tuple[
        str, str, list]:
    """Visible text, reasoning text and tool calls from one probe response."""
    try:
        if transport == "anthropic":
            from . import transport_anthropic
            parsed = transport_anthropic.parse_json_response(data)
        elif transport == "gemini":
            from . import transport_gemini
            parsed = transport_gemini.parse_response(data)
        elif transport == "openai-responses":
            from . import transport_openai
            parsed = transport_openai.parse_response(data)
        else:
            parsed = {}
            choices = data.get("choices") or []
            choice = choices[0] if choices and isinstance(choices[0], dict) \
                else {}
            message = choice.get("message") or {}
            if isinstance(message, dict):
                parsed = {
                    "text": str(message.get("content") or ""),
                    "reasoning": str(message.get("reasoning_content") or ""),
                    "tool_calls": list(message.get("tool_calls") or []),
                }
        return (str(parsed.get("text") or ""),
                str(parsed.get("reasoning") or ""),
                list(parsed.get("tool_calls") or []))
    except Exception:
        return "", "", []


def _auth_identity_for(base_url: str, model: str, cfg) -> str:
    return _auth_identity(base_url, provider_for(base_url, model), cfg)


def get_capabilities(base_url: str, model: str, *, allow_probe: bool = False,
                     cfg=None) -> ProviderCapabilities:
    """Known capabilities for ``(base_url, model)``.

    Reads the 24 h probe cache first, then blends catalog evidence, offline
    model-catalog metadata and live transport refusals.  No network I/O
    happens unless ``allow_probe`` is true, which is the explicit consent
    signal; even then the probe is bounded to one minimal request.
    """
    provider = _provider_for_cfg(base_url, model, cfg)
    auth_hash = _auth_identity(base_url, provider, cfg)
    key = _cache_key(base_url, model, auth_hash)
    cached = _cache_get(key)
    if cached is not None:
        merged, _used = _merge_evidence(
            cached, _runtime_evidence(base_url, model))
        return merged
    caps = _static_capabilities(base_url, model, provider)
    evidence = _runtime_evidence(base_url, model)
    caps, _used = _merge_evidence(caps, evidence)
    if allow_probe:
        probed = probe_capabilities(base_url, model, cfg=cfg,
                                    deadline=time.monotonic() + 20.0)
        if probed.probed_at:
            return probed
        return probed
    return caps


# ------------------------------------------------------- compatibility API ---

_FRIENDLY_NAMES: dict[str, str] = {
    "api.deepseek.com": "DeepSeek",
    "api.openai.com": "OpenAI",
    "api.anthropic.com": "Anthropic",
    ".aliyuncs.com": "Alibaba DashScope",
    "api.moonshot.cn": "Moonshot",
    "api.moonshot.ai": "Moonshot",
    "generativelanguage.googleapis.com": "Google Gemini",
    "api.x.ai": "xAI",
    "api.groq.com": "Groq",
    "openrouter.ai": "OpenRouter",
    "api.together.xyz": "Together",
    "api.mistral.ai": "Mistral",
    "api.cerebras.ai": "Cerebras",
    "api.fireworks.ai": "Fireworks",
    "api.minimax.io": "MiniMax",
    "api.z.ai": "Z.ai",
    "integrate.api.nvidia.com": "NVIDIA NIM",
    "router.huggingface.co": "Hugging Face",
    "api.deepinfra.com": "DeepInfra",
    "api.studio.nebius.ai": "Nebius",
    "api.sambanova.ai": "SambaNova",
    "api.hyperbolic.xyz": "Hyperbolic",
    "api.lambdalabs.com": "Lambda",
    "api.perplexity.ai": "Perplexity",
}


def provider_label(base_url: str, provider: str = "") -> str:
    """A friendly name for the endpoint serving a call: the brand for a known
    hosted API, else the configured provider name, else "local server" for
    loopback / `.local` hosts, else the bare host."""
    host = _host(base_url)
    for pattern, name in _FRIENDLY_NAMES.items():
        if host_matches(host, pattern):
            return name
    if provider:
        return provider
    if not host or host in ("localhost", "127.0.0.1", "::1") \
            or host.endswith(".local"):
        return "local server"
    return host


def cache_mode_for(cfg) -> str:
    """The effective prompt_cache mode for `cfg`: an explicit setting always
    wins; "auto" means the provider table decides."""
    setting = (getattr(cfg, "prompt_cache", "auto") or "auto").strip().lower()
    if setting and setting != "auto":
        return setting
    return provider_for(getattr(cfg, "base_url", ""),
                        getattr(cfg, "model", "")).cache_mode


def cache_key(cfg) -> str:
    """The effective prompt_cache_key: an explicit non-"auto" value verbatim,
    "auto" as a stable hash of the project root + profile + model (so one
    project routes to a warm cache across runs, while two projects differ),
    "" for nothing."""
    raw = (getattr(cfg, "prompt_cache_key", "") or "").strip()
    if not raw:
        return ""
    if raw != "auto":
        return raw
    h = hashlib.sha256()
    for part in (getattr(cfg, "_root", None) or "",
                 getattr(cfg, "active_profile", None) or "",
                 getattr(cfg, "model", "") or ""):
        h.update((str(part) + "\0").encode("utf-8", "replace"))
    return h.hexdigest()[:32]


# Approximate USD list prices per million tokens: (input, cache read, output).
# Matched by substring on the lower-cased model name, most specific first;
# these are Tier 3 zero-network defaults for cost reporting / --max-cost, never
# a bill -- cached models.dev entries (Tier 2), provider /models metadata
# (Tier 1), and explicit price_* settings always win.
_MODEL_PRICES: tuple[tuple[str, float, float, float], ...] = (
    # Anthropic Claude 4.5 / 4.6 / 5.5
    ("claude-opus-5", 15.00, 1.50, 75.00),
    ("claude-sonnet-5", 3.00, 0.30, 15.00),
    ("claude-haiku-4-5", 1.00, 0.10, 5.00),
    ("claude-haiku-4", 0.80, 0.08, 4.00),
    ("claude-opus-4-6", 5.00, 0.50, 25.00),
    ("claude-opus-4-5", 5.00, 0.50, 25.00),
    ("claude-opus-4", 15.00, 1.50, 75.00),
    ("claude-sonnet-4-6", 3.00, 0.30, 15.00),
    ("claude-sonnet-4-5", 3.00, 0.30, 15.00),
    ("claude-sonnet-4", 3.00, 0.30, 15.00),
    # Google Gemini 2.5 / 3 / 4
    ("gemini-4-flash", 0.15, 0.0375, 0.60),
    ("gemini-4-pro", 1.25, 0.31, 10.00),
    ("gemini-3-flash", 0.15, 0.0375, 0.60),
    ("gemini-3-pro", 1.25, 0.31, 10.00),
    ("gemini-2.5-flash", 0.15, 0.0375, 0.60),
    ("gemini-2.5-pro", 1.25, 0.31, 10.00),
    # DeepSeek V4 / Flash / Reasoner / Chat
    ("deepseek-v4-flash", 0.14, 0.035, 0.55),
    ("deepseek-v4", 0.27, 0.07, 1.10),
    ("deepseek-reasoner", 0.55, 0.14, 2.19),
    ("deepseek-chat", 0.27, 0.07, 1.10),
    # Qwen 3.8 / 4 / Coder
    ("qwen3-coder-plus", 1.00, 0.20, 4.00),
    ("qwen3-coder", 0.50, 0.10, 2.00),
    ("qwen3.8", 0.80, 0.16, 3.20),
    ("qwen-3.8", 0.80, 0.16, 3.20),
    ("qwen-4", 1.60, 0.32, 6.40),
    ("qwen4", 1.60, 0.32, 6.40),
    ("qwen-max", 2.40, 0.48, 9.60),
    ("qwen-plus", 0.40, 0.08, 1.20),
    ("qwen-flash", 0.05, 0.01, 0.20),
    # OpenAI GPT-5 / o3 / o4 / 4.1 / 4o
    ("gpt-5.4-mini", 0.40, 0.10, 1.60),
    ("gpt-5.4", 2.50, 0.625, 10.00),
    ("gpt-5-mini", 0.25, 0.025, 2.00),
    ("gpt-5", 1.25, 0.125, 10.00),
    ("o4-mini", 1.10, 0.275, 4.40),
    ("o3-mini", 1.10, 0.55, 4.40),
    ("o3", 2.00, 0.50, 8.00),
    ("gpt-4.1-mini", 0.40, 0.10, 1.60),
    ("gpt-4.1", 2.00, 0.50, 8.00),
    ("gpt-4o-mini", 0.15, 0.075, 0.60),
    ("gpt-4o", 2.50, 1.25, 10.00),
)

# Per-host fallback for models the table does not name (gateways, aliases):
# the provider's flagship list price, clearly approximate.
_HOST_PRICES: dict[str, tuple[float, float, float]] = {
    "api.deepseek.com": (0.27, 0.07, 1.10),
    ".aliyuncs.com": (2.40, 0.48, 9.60),
    "api.openai.com": (2.50, 1.25, 10.00),
    "api.anthropic.com": (3.00, 0.30, 15.00),
    "generativelanguage.googleapis.com": (1.25, 0.31, 10.00),
}


def default_prices_with_source(base_url: str, model: str = "") -> tuple[
        dict, str]:
    """Return ``(prices_dict, source)`` where ``source`` is ``"catalog"``
    (Tier 2 on-disk cache), ``"builtin"`` (Tier 3 static fallback), or ``""``.

    Read-only on the hot path: never performs network I/O.
    """
    try:
        from . import model_catalog
        meta = model_catalog.lookup_model_meta(base_url, model, allow_fetch=False)
    except Exception:
        meta = {}
    if meta:
        p_in = float(meta.get("price_input_per_mtok") or 0.0)
        p_out = float(meta.get("price_output_per_mtok") or 0.0)
        p_cached = float(meta.get("price_cached_per_mtok") or 0.0)
        p_write = float(meta.get("price_cache_write_per_mtok") or 0.0)
        if p_in > 0 or p_out > 0 or p_cached > 0:
            out = {
                "price_input_per_mtok": p_in,
                "price_cached_per_mtok": p_cached,
                "price_output_per_mtok": p_out,
            }
            if p_write > 0:
                out["price_cache_write_per_mtok"] = p_write
            return out, "catalog"

    m = (model or "").lower()
    prices = None
    for needle, p_in, p_cached, p_out in _MODEL_PRICES:
        if needle in m:
            prices = (p_in, p_cached, p_out)
            break
    if prices is None:
        host = _host(base_url)
        for key, fallback in _HOST_PRICES.items():
            if host_matches(host, key):
                prices = fallback
                break
    if prices is None:
        return {}, ""
    p_in, p_cached, p_out = prices
    return {
        "price_input_per_mtok": p_in,
        "price_cached_per_mtok": p_cached,
        "price_output_per_mtok": p_out,
    }, "builtin"


def default_prices(base_url: str, model: str = "") -> dict:
    """Approximate price_* defaults for a hosted endpoint, keyed by the config
    field names; {} when nothing is known (local servers stay at 0)."""
    prices, _source = default_prices_with_source(base_url, model)
    return prices


def profile_overrides(base_url: str, model: str = "") -> dict:
    """The profile keys a hosted scan should write for this provider, so the
    profile carries the table's caching and compatibility choices (and the user
    can still override them). Never a secret -- only settings."""
    p = provider_for(base_url, model)
    out: dict = {}
    mode = {"implicit": "implicit", "explicit": "explicit",
            "unsupported": "off"}.get(p.cache_mode)
    if mode:
        out["prompt_cache"] = mode
    if p.json_mode:
        out["json_mode"] = p.json_mode
    if not p.native_tool_choice:
        out["tool_mode"] = "grammar"
    return out
