"""Quiet, offline-safe 24h-cached provider model catalog (models.dev).

Mirrors ``pedacito/update.py``'s zero-startup-cost, silent-fallback contract:

  * Cache path: ``config.user_config_dir() / "model_catalog.json"`` (TTL 24 h),
    respecting test isolation when ``config.USER_CONFIG`` is redirected.
  * Never blocks startup and never runs for local-only installs.
  * Fetches ``https://models.dev/api.json`` (timeout 3.0 s) ONLY when
    ``auto_refresh_model_catalog`` is True and a hosted provider is being
    probed/refreshed or explicitly requested.
  * Every failure mode (offline, timeout, non-200, malformed JSON, read-only
    config directory) is a silent no-op.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from . import config as config_mod
from . import fileio

CATALOG_URL = "https://models.dev/api.json"
FETCH_TIMEOUT = 3.0
CACHE_TTL = 24 * 3600.0
CACHE_NAME = "model_catalog.json"

# Map URL host patterns -> models.dev provider IDs in priority order.  Patterns
# are exact hostnames, dot-boundary suffixes (".example.com") or "*" labels;
# `providers.host_matches` enforces the boundary, so a look-alike host cannot
# inherit a real provider's catalog rows.
_HOST_TO_PROVIDER_IDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("api.anthropic.com", ("anthropic",)),
    ("api.openai.com", ("openai",)),
    ("generativelanguage.googleapis.com", ("google", "gemini", "google-vertex")),
    ("api.deepseek.com", ("deepseek",)),
    ("dashscope.aliyuncs.com", ("alibaba", "alibaba-cn", "dashscope", "qwen")),
    ("dashscope-intl.aliyuncs.com",
     ("alibaba", "alibaba-cn", "dashscope", "qwen")),
    (".aliyuncs.com", ("alibaba", "alibaba-cn", "dashscope", "qwen")),
    ("openrouter.ai", ("openrouter",)),
    ("api.groq.com", ("groq",)),
    ("api.together.xyz", ("togetherai", "together")),
    ("api.x.ai", ("xai",)),
    ("api.moonshot.cn", ("moonshotai", "moonshot")),
    ("api.moonshot.ai", ("moonshotai", "moonshot")),
    ("api.mistral.ai", ("mistral",)),
    ("api.cerebras.ai", ("cerebras",)),
    ("api.fireworks.ai", ("fireworks-ai",)),
    ("api.minimax.io", ("minimax",)),
    ("api.minimaxi.com", ("minimax",)),
    ("api.z.ai", ("zai", "z-ai", "zhipu")),
    ("integrate.api.nvidia.com", ("nvidia",)),
    ("router.huggingface.co", ("huggingface", "hf")),
    ("api-inference.huggingface.co", ("huggingface", "hf")),
    ("api.deepinfra.com", ("deepinfra",)),
    ("api.studio.nebius.ai", ("nebius",)),
    ("api.sambanova.ai", ("sambanova",)),
    ("api.hyperbolic.xyz", ("hyperbolic",)),
    ("api.lambdalabs.com", ("lambda",)),
    ("api.perplexity.ai", ("perplexity",)),
)

_DATE_SUFFIX_RE = re.compile(r"-(?:20\d{6}|\d{4}-\d{2}-\d{2})$")


def _host(base_url: str) -> str:
    try:
        return (urlparse(str(base_url or "")).hostname or "").lower()
    except ValueError:
        return ""


def is_local_base_url(base_url: str) -> bool:
    """True for empty, loopback, mDNS ``.local``, private/LAN, or CGNAT
    (100.64.0.0/10, Tailscale) addresses. Delegates to the one canonical
    predicate in config.py."""
    return config_mod._endpoint_is_local(base_url)


def cache_path() -> Path:
    """Path to ``model_catalog.json`` beside the user config."""
    try:
        cur_dir = Path(config_mod.user_config_dir())
        default_dir = Path(config_mod._user_config_dir())
        if cur_dir != default_dir:
            return cur_dir / CACHE_NAME
        if config_mod.USER_CONFIG.parent != default_dir:
            return config_mod.USER_CONFIG.parent / CACHE_NAME
        return cur_dir / CACHE_NAME
    except Exception:
        return config_mod.USER_CONFIG.parent / CACHE_NAME


def catalog_enabled(cfg=None) -> bool:
    """True unless disabled by ``PEDACITO_AUTO_REFRESH_MODEL_CATALOG`` or ``cfg``."""
    raw = os.environ.get("PEDACITO_AUTO_REFRESH_MODEL_CATALOG")
    if raw is not None and raw.strip():
        return raw.strip().lower() not in ("0", "false", "no", "off")
    if cfg is not None:
        return bool(getattr(cfg, "auto_refresh_model_catalog", True))
    return True


def normalize_model_id(model: str) -> str:
    """Canonicalize a model identifier for catalog lookup."""
    mid = str(model or "").strip().lower()
    if mid.startswith("models/"):
        mid = mid[len("models/"):].strip()
    return mid


def _candidate_model_ids(norm: str) -> list[str]:
    if not norm:
        return []
    out = [norm]
    no_tag = norm.split(":", 1)[0]
    if no_tag and no_tag not in out:
        out.append(no_tag)
    no_date = _DATE_SUFFIX_RE.sub("", no_tag or norm)
    if no_date and no_date not in out:
        out.append(no_date)
    return out


def provider_ids_for_base_url(base_url: str) -> tuple[str, ...]:
    """Candidate ``models.dev`` provider IDs / host keys for ``base_url``."""
    host = _host(base_url)
    if not host:
        return ()
    from . import providers
    out: list[str] = []
    for pattern, pids in _HOST_TO_PROVIDER_IDS:
        if providers.host_matches(host, pattern):
            for pid in pids:
                if pid not in out:
                    out.append(pid)
    if host not in out:
        out.append(host)
    return tuple(out)


def _num(val, default: float = 0.0) -> float:
    try:
        n = float(val)
    except (TypeError, ValueError):
        return default
    return n if n >= 0 else default


def _int_pos(val) -> int:
    try:
        n = int(val)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def _extract_model_record(entry: dict) -> dict | None:
    if not isinstance(entry, dict):
        return None
    limit = entry.get("limit") if isinstance(entry.get("limit"), dict) else {}
    cost = entry.get("cost") if isinstance(entry.get("cost"), dict) else {}
    ctx = _int_pos(limit.get("context") or entry.get("context_length")
                   or entry.get("context_tokens"))
    out_tok = _int_pos(limit.get("output") or entry.get("output_tokens")
                       or entry.get("max_output_tokens"))
    p_in = _num(cost.get("input") if "input" in cost else entry.get("price_input_per_mtok"))
    p_out = _num(cost.get("output") if "output" in cost else entry.get("price_output_per_mtok"))
    p_cached = _num(
        cost.get("cache_read") if "cache_read" in cost
        else entry.get("price_cached_per_mtok")
    )
    p_write = _num(
        cost.get("cache_write") if "cache_write" in cost
        else entry.get("price_cache_write_per_mtok")
    )
    if not (ctx or out_tok or p_in or p_out or p_cached):
        return None
    rec: dict = {
        "context_tokens": ctx,
        "output_tokens": out_tok,
        "price_input_per_mtok": p_in,
        "price_cached_per_mtok": p_cached,
        "price_output_per_mtok": p_out,
    }
    if p_write > 0:
        rec["price_cache_write_per_mtok"] = p_write
    return rec


def parse_models_dev(payload) -> dict[str, dict]:
    """Parse a ``models.dev/api.json`` payload into a flat
    ``{"<provider_id>|<normalized_model_id>": record}`` index."""
    index: dict[str, dict] = {}
    if not isinstance(payload, dict):
        return index
    for prov_key, prov_body in payload.items():
        if not isinstance(prov_body, dict):
            continue
        pid = str(prov_body.get("id") or prov_key or "").strip().lower()
        if not pid:
            continue
        api_host = _host(str(prov_body.get("api") or ""))
        models = prov_body.get("models")
        if isinstance(models, dict):
            items = list(models.items())
        elif isinstance(models, list):
            items = [
                (str(m.get("id") or m.get("name") or ""), m)
                for m in models if isinstance(m, dict)
            ]
        else:
            continue
        for raw_key, m_entry in items:
            if not isinstance(m_entry, dict):
                continue
            mid = normalize_model_id(str(m_entry.get("id") or raw_key or ""))
            if not mid:
                continue
            rec = _extract_model_record(m_entry)
            if rec is None:
                continue
            index[f"{pid}|{mid}"] = rec
            if api_host and api_host != pid:
                index.setdefault(f"{api_host}|{mid}", rec)
    return index


def read_cache(now: float | None = None, *, allow_stale: bool = False) -> dict | None:
    """Read ``model_catalog.json`` from disk.

    Returns ``{"fetched_at": float, "index": dict}`` or ``None`` when absent,
    corrupt, or stale (unless ``allow_stale=True``).
    """
    now = time.time() if now is None else now
    try:
        data = json.loads(cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    fetched_at = data.get("fetched_at")
    index = data.get("index")
    if not isinstance(fetched_at, (int, float)) or not isinstance(index, dict):
        return None
    if not allow_stale:
        if now - float(fetched_at) >= CACHE_TTL or float(fetched_at) > now + 60:
            return None
    return {"fetched_at": float(fetched_at), "index": index}


def write_cache(index: dict, now: float | None = None) -> None:
    """Best-effort atomic write of the parsed index to ``model_catalog.json``."""
    now = time.time() if now is None else now
    payload = {
        "version": 1,
        "fetched_at": float(now),
        "index": dict(index or {}),
    }
    try:
        fileio.write_bytes(
            cache_path(),
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        )
    except OSError:
        pass


def _fetch_payload(url: str = CATALOG_URL, timeout: float = FETCH_TIMEOUT):
    with urlopen(url, timeout=timeout) as resp:
        status = getattr(resp, "status", None)
        if status is None:
            status = getattr(resp, "code", 200)
        if status != 200:
            return None
        return json.loads(resp.read())


def refresh_catalog(
    force: bool = True,
    cfg=None,
    *,
    timeout: float = FETCH_TIMEOUT,
    url: str = CATALOG_URL,
) -> dict:
    """Refresh ``model_catalog.json`` from ``models.dev/api.json``.

    When ``force=False``:
      - Returns immediately if the cache is fresh (<24 h TTL).
      - Returns immediately with zero network calls if ``auto_refresh_model_catalog``
        is disabled.
    Every network/parse failure is caught and returned as ``{"ok": False, ...}``.
    """
    if not force:
        fresh = read_cache(allow_stale=False)
        if fresh is not None:
            return {
                "ok": True,
                "refreshed": False,
                "cached": True,
                "count": len(fresh["index"]),
                "fetched_at": fresh["fetched_at"],
                "path": str(cache_path()),
            }
        if not catalog_enabled(cfg):
            return {
                "ok": False,
                "refreshed": False,
                "cached": False,
                "detail": "disabled",
                "path": str(cache_path()),
            }

    try:
        payload = _fetch_payload(url=url, timeout=timeout)
    except Exception as exc:
        return {
            "ok": False,
            "refreshed": False,
            "cached": False,
            "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            "path": str(cache_path()),
        }
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "refreshed": False,
            "cached": False,
            "error": "empty or invalid catalog payload",
            "path": str(cache_path()),
        }

    index = parse_models_dev(payload)
    now = time.time()
    write_cache(index, now=now)
    return {
        "ok": True,
        "refreshed": True,
        "cached": False,
        "count": len(index),
        "fetched_at": now,
        "path": str(cache_path()),
    }


def lookup_model_meta(
    base_url: str,
    model: str,
    *,
    allow_fetch: bool = False,
    cfg=None,
) -> dict:
    """Look up cached model metadata (context window, output limit, pricing).

    Never performs network I/O unless ``allow_fetch=True``, ``base_url`` is a
    non-local hosted endpoint, ``auto_refresh_model_catalog`` is enabled, and
    the local cache is missing or stale.
    """
    norm = normalize_model_id(model)
    if not norm:
        return {}
    if is_local_base_url(base_url):
        return {}

    if allow_fetch and catalog_enabled(cfg):
        if read_cache(allow_stale=False) is None:
            refresh_catalog(force=False, cfg=cfg)

    cache = read_cache(allow_stale=True)
    if cache is None:
        return {}
    index = cache.get("index") or {}
    if not index:
        return {}

    pids = list(provider_ids_for_base_url(base_url))
    # If model has an org/vendor prefix (e.g. "anthropic/claude-sonnet-4-5" on
    # OpenRouter), check the endpoint provider first, then fall back to the
    # vendor prefix + bare model name.
    candidates = _candidate_model_ids(norm)
    for pid in pids:
        for cand in candidates:
            hit = index.get(f"{pid}|{cand}")
            if isinstance(hit, dict):
                return dict(hit)

    if "/" in norm:
        vendor, bare = norm.split("/", 1)
        for cand in _candidate_model_ids(bare):
            for pid in pids:
                hit = index.get(f"{pid}|{cand}")
                if isinstance(hit, dict):
                    return dict(hit)
            hit = index.get(f"{vendor}|{cand}")
            if isinstance(hit, dict):
                return dict(hit)

    return {}
