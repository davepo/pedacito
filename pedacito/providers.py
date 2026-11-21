"""Hosted-provider table: how each API handles prompt caching, plus the
compatibility quirks the transport must apply.

Local servers are not listed: they reuse their KV prefix automatically and need
no markers or keys. Matching is by the host of `base_url`; a provider not in
the table falls back to an "implicit" entry, so nothing is ever forced on an
unknown API (explicit `prompt_cache` / `prompt_cache_key` settings still win).
"""

from __future__ import annotations

from dataclasses import dataclass


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


_PROVIDERS: dict[str, Provider] = {
    # Automatic, best-effort. Reads surface as prompt_cache_hit_tokens and
    # prompt_tokens_details.cached_tokens.
    "api.deepseek.com": Provider(
        cache_mode="implicit",
        usage_read=("prompt_cache_hit_tokens", "cached_tokens")),
    # Automatic for prompts >= 1024 tokens; prompt_cache_key improves routing.
    "api.openai.com": Provider(
        cache_mode="implicit", cache_key_ok=True, min_tokens=1024),
    # Qwen on Alibaba Model Studio / DashScope. Implicit is automatic and
    # cannot be disabled; explicit (cache_control ephemeral) is opt-in.
    "dashscope": Provider(
        cache_mode="explicit", max_markers=4, min_tokens=1024, ttls=("5m",),
        usage_read=("cache_read_input_tokens", "cached_tokens"),
        usage_write=("cache_creation_input_tokens", "cache_write_tokens")),
    # Anthropic's native /v1/messages surface: explicit cache_control markers
    # (up to 4 breakpoints, 5m/1h TTLs) with usage read/write token fields.
    # json_mode stays "schema" (the native transport turns step schemas into
    # tools), temperature below 1 is rejected (Claude 4.7+), and a forced
    # named tool_choice is accepted.
    "api.anthropic.com": Provider(
        cache_mode="explicit", max_markers=4, min_tokens=1024,
        ttls=("5m", "1h"),
        usage_read=("cache_read_input_tokens",),
        usage_write=("cache_creation_input_tokens",),
        json_mode="schema", omit_temperature=True, native_tool_choice=True),
    # Reported as automatic/implicit by the providers and gateway docs.
    "api.moonshot.cn": Provider(cache_mode="implicit"),
    "generativelanguage.googleapis.com": Provider(cache_mode="implicit"),
    "api.x.ai": Provider(cache_mode="implicit"),
    "api.groq.com": Provider(cache_mode="implicit"),
}


def _host(base_url: str) -> str:
    from urllib.parse import urlparse
    try:
        return (urlparse(base_url or "").hostname or "").lower()
    except Exception:
        return ""


def provider_for(base_url: str, model: str = "") -> Provider:
    """The provider entry whose host matches `base_url` (model is reserved for
    future per-model entries), or a generic implicit-cache entry."""
    host = _host(base_url)
    for key, prov in _PROVIDERS.items():
        if key in host:
            return prov
    return Provider(cache_mode="implicit")


# Human-facing brand names for the heartbeat / telemetry label, matched by
# substring on the host (same convention as the provider table above).
_FRIENDLY_NAMES: dict[str, str] = {
    "api.deepseek.com": "DeepSeek",
    "api.openai.com": "OpenAI",
    "api.anthropic.com": "Anthropic",
    "dashscope": "Alibaba DashScope",
    "api.moonshot.cn": "Moonshot",
    "generativelanguage.googleapis.com": "Google Gemini",
    "api.x.ai": "xAI",
    "api.groq.com": "Groq",
}


def provider_label(base_url: str, provider: str = "") -> str:
    """A friendly name for the endpoint serving a call: the brand for a known
    hosted API, else the configured provider name, else "local server" for
    loopback / `.local` hosts, else the bare host."""
    host = _host(base_url)
    for key, name in _FRIENDLY_NAMES.items():
        if key in host:
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
    import hashlib
    h = hashlib.sha256()
    for part in (getattr(cfg, "_root", None) or "",
                 getattr(cfg, "active_profile", None) or "",
                 getattr(cfg, "model", "") or ""):
        h.update((str(part) + "\0").encode("utf-8", "replace"))
    return h.hexdigest()[:32]


# Approximate USD list prices per million tokens: (input, cache read, output).
# Matched by substring on the lower-cased model name, most specific first;
# these are defaults for cost reporting / --max-cost, never a bill -- an
# explicit price_* setting always wins.
_MODEL_PRICES: tuple[tuple[str, float, float, float], ...] = (
    ("deepseek-reasoner", 0.55, 0.14, 2.19),
    ("deepseek-chat", 0.27, 0.07, 1.10),
    ("qwen-max", 2.40, 0.48, 9.60),
    ("gpt-4o", 2.50, 1.25, 10.00),
)

# Per-host fallback for models the table does not name (gateways, aliases):
# the provider's flagship list price, clearly approximate.
_HOST_PRICES: dict[str, tuple[float, float, float]] = {
    "api.deepseek.com": (0.27, 0.07, 1.10),
    "dashscope": (2.40, 0.48, 9.60),
    "api.openai.com": (2.50, 1.25, 10.00),
}


def default_prices(base_url: str, model: str = "") -> dict:
    """Approximate price_* defaults for a hosted endpoint, keyed by the config
    field names; {} when nothing is known (local servers stay at 0)."""
    m = (model or "").lower()
    prices = None
    for needle, p_in, p_cached, p_out in _MODEL_PRICES:
        if needle in m:
            prices = (p_in, p_cached, p_out)
            break
    if prices is None:
        host = _host(base_url)
        for key, fallback in _HOST_PRICES.items():
            if key in host:
                prices = fallback
                break
    if prices is None:
        return {}
    p_in, p_cached, p_out = prices
    return {"price_input_per_mtok": p_in,
            "price_cached_per_mtok": p_cached,
            "price_output_per_mtok": p_out}


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
