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
    # Anthropic's OpenAI-compatible endpoint states "Prompt caching is not
    # supported" and returns empty prompt_tokens_details. It also rejects
    # json_object, temperature below 1 (Claude 4.7+), and forced tool_choice.
    "api.anthropic.com": Provider(
        cache_mode="unsupported",
        usage_read=("cache_read_input_tokens",),
        usage_write=("cache_creation_input_tokens",),
        json_mode="schema", omit_temperature=True, native_tool_choice=False),
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
