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
    "generativelanguage.googleapis.com": Provider(
        cache_mode="implicit",
        usage_read=("cachedContentTokenCount", "cached_tokens")),
    "api.x.ai": Provider(cache_mode="implicit"),
    "api.groq.com": Provider(cache_mode="implicit"),
    "openrouter.ai": Provider(cache_mode="implicit"),
    "api.together.xyz": Provider(cache_mode="implicit"),
    "api.mistral.ai": Provider(cache_mode="implicit"),
    "api.cerebras.ai": Provider(cache_mode="implicit"),
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
    "openrouter.ai": "OpenRouter",
    "api.together.xyz": "Together",
    "api.mistral.ai": "Mistral",
    "api.cerebras.ai": "Cerebras",
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
    "dashscope": (2.40, 0.48, 9.60),
    "api.openai.com": (2.50, 1.25, 10.00),
    "api.anthropic.com": (3.00, 0.30, 15.00),
    "generativelanguage.googleapis.com": (1.25, 0.31, 10.00),
}


def default_prices_with_source(base_url: str, model: str = "") -> tuple[dict, str]:
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
            if key in host:
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
