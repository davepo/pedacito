"""Model-server discovery.

`discover_servers` finds local model servers by probing candidate URLs, then
labels each one by what it answers (its fingerprint), not by the port it
listens on. Ports are user-configurable and collide: Lemonade may listen on
11434, the port that used to mean "Ollama", and LM Studio on 1234, so the old
fixed port -> label table is gone. A server is identified by its replies:

    Lemonade        `owned_by: "lemonade"`, or a `/api/v1/health` endpoint
    LM Studio       a native `/api/v0/models` endpoint (or a `/api/v1/models`
                    listing when nothing else answers)
    Ollama          `/api/version` plus `/api/tags`
    llama.cpp       a `/props` endpoint
    anything else   "OpenAI-compatible"

Candidates, in order:

    1. explicit URLs (e.g. from `--url`);
    2. the base_url of every existing profile, which re-finds custom ports;
    3. `ports` (default `DEFAULT_PORTS`), each tried on localhost;
    4. the `PEDACITO_DISCOVERY_URLS` environment variable.

Each candidate is tried with both `/v1` and `/api/v1` prefixes, so a server
that serves only one of them still shows up.
"""

from __future__ import annotations

import os
import re

import requests

# The well-known local ports, probed on localhost when nothing else points at a
# server. They are hints, not identities: the fingerprint below decides the
# label, so two servers may share a host and any port works.
DEFAULT_PORTS = (1234, 11434, 8000, 8080)

# Name fragments that mark an embedding model. The scan drops these: they
# cannot run the harness. Checked against the `id`/`name`/`model` field.
_EMBEDDING_NAME_MARKERS = (
    "embed", "bge-", "e5-", "gte-", "minilm", "nomic-embed", "all-mpnet",
)

_GIB = 1024 ** 3


_LOCALHOST_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.\-]*://)localhost(?=/|:|$)",
                           re.IGNORECASE)
_AUTHORITY_RE = re.compile(r"\A([a-zA-Z][a-zA-Z0-9+.\-]*://[^/]*)(.*)\Z")


def normalize_url(url) -> str:
    """The canonical form of a server URL for comparing candidates.

    One server is often spelled several ways (a profile's
    `http://localhost:11434/v1/` vs a scan's `http://127.0.0.1:11434`), and
    probing both spellings makes the same server show up twice. This folds
    those spellings into one key: trailing slashes dropped, the `localhost`
    host written as `127.0.0.1`, and a `/v1` (or `/api/v1`) suffix removed so
    the same endpoint is never probed -- or listed -- twice."""
    u = str(url or "").strip().rstrip("/")
    u = _LOCALHOST_RE.sub(r"\g<1>127.0.0.1", u, count=1)
    # scheme and host are case-insensitive; the path is not, so lower only the
    # `scheme://authority` part ("HTTP://LocalHost:9/" == "http://127.0.0.1:9")
    m = _AUTHORITY_RE.match(u)
    if m:
        u = m.group(1).lower() + m.group(2)
    for suffix in ("/api/v1", "/v1"):
        if u.endswith(suffix):
            u = u[: -len(suffix)].rstrip("/")
            break
    return u


def _root(base_url: str) -> str:
    """Strip a `/v1` or `/api/v1` suffix, leaving the server root where the
    native endpoints (`/api/v0/models`, `/api/tags`, ...) live."""
    base = (base_url or "").rstrip("/")
    for suffix in ("/api/v1", "/v1"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def _get(url: str, timeout: float, headers=None):
    """A 200 response from `url`, or None on any failure (wrong status, JSON
    error, timeout, refusal). Every probe is best-effort."""
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
    except Exception:
        return None
    return r if r.status_code == 200 else None


def _data(resp) -> dict | list | None:
    try:
        return resp.json()
    except Exception:
        return None


def _entries_from(data) -> list[dict] | None:
    """The model-entry list from an OpenAI-shape (`data`) or Ollama-shape
    (`models`) payload, or None when the payload carries no list."""
    if isinstance(data, dict):
        entries = data.get("data") or data.get("models") or []
    elif isinstance(data, list):
        entries = data
    else:
        entries = []
    entries = [m for m in entries if isinstance(m, dict)]
    return entries or None


def _list_entries(url: str, timeout: float, headers=None) -> list[dict] | None:
    resp = _get(url, timeout, headers)
    return _entries_from(_data(resp)) if resp else None


def _model_id(entry: dict) -> str | None:
    for key in ("id", "name", "model"):
        value = entry.get(key)
        if value:
            return str(value)
    return None


def _is_lemonade(entries: list[dict]) -> bool:
    return any(str(m.get("owned_by", "")).lower() == "lemonade" for m in entries)


def _fingerprint(root: str, timeout: float, headers=None) -> str:
    """Label a server at `root` from its native endpoints; "" when none of the
    server-specific endpoints answer (a bare OpenAI-compatible server)."""
    # Lemonade owns /api/v1/health, so it wins even on a port another server
    # historically used (e.g. Lemonade on 11434 must not read "Ollama").
    if _get(root + "/api/v1/health", timeout, headers):
        return "Lemonade"
    if _get(root + "/api/v0/models", timeout, headers):
        return "LM Studio"
    if _get(root + "/api/version", timeout, headers) and \
            _get(root + "/api/tags", timeout, headers):
        return "Ollama"
    if _get(root + "/props", timeout, headers):
        return "llama.cpp"
    return ""


def _ollama_entries(root: str, timeout: float, headers=None) -> list[dict] | None:
    """Ollama's native listing (`/api/tags`) spells models as `name`."""
    resp = _get(root + "/api/tags", timeout, headers)
    data = _data(resp) if resp else None
    models = (data or {}).get("models") if isinstance(data, dict) else None
    if not models:
        return None
    return [{"id": m.get("name")} for m in models
            if isinstance(m, dict) and m.get("name")]


def _probe_root(root: str, timeout: float, headers=None):
    """Identify and list one server at `root`. Returns (label, base_url,
    [model ids]) or None when the root answers nothing."""
    label = _fingerprint(root, timeout, headers)
    v1 = _list_entries(root + "/v1/models", timeout, headers)
    api = _list_entries(root + "/api/v1/models", timeout, headers)
    entries = v1 if v1 is not None else api
    base_url = (root + "/v1") if v1 is not None else \
               (root + "/api/v1") if api is not None else ""

    # Native listings when the OpenAI shim is absent.
    if entries is None:
        if label == "Ollama":
            entries = _ollama_entries(root, timeout, headers)
            base_url = root + "/v1"
        elif label == "LM Studio":
            entries = _list_entries(root + "/api/v0/models", timeout, headers)
            base_url = root + "/v1"

    if entries is None and label == "":
        return None

    if label == "" and entries is not None:
        if _is_lemonade(entries):
            label = "Lemonade"
        elif v1 is None and api is not None:
            # Only a native /api/v1 surface answered: LM Studio's layout,
            # once Lemonade (which also serves /api/v1) is ruled out.
            label = "LM Studio"

    ids = [mid for m in entries if (mid := _model_id(m))]
    return (label or "OpenAI-compatible", base_url or (root + "/v1"), ids)


def _candidate_roots(urls, profile_urls, ports) -> list[str]:
    """The deduplicated server roots to probe, in the documented order.

    Every candidate goes through `normalize_url` first, so `localhost` and
    `127.0.0.1` spellings (and trailing slashes or `/v1` suffixes) of one
    server collapse into a single root that is probed exactly once."""
    roots: list[str] = []

    def add(url: str) -> None:
        root = normalize_url(url)
        if root and root not in roots:
            roots.append(root)

    for url in urls:
        add(url)
    for url in profile_urls:
        add(url)
    for port in ports:
        add(f"http://localhost:{port}")
    for url in os.environ.get("PEDACITO_DISCOVERY_URLS", "").split():
        add(url)
    return roots


def discover_servers(timeout: float = 2.0, urls=(), ports=None,
                     profile_urls=(), headers=None) -> list[tuple[str, str, list[str]]]:
    """Probe candidates for local model servers.

    Returns `[(label, base_url, [model ids])]` for each server that answers,
    deduplicated by normalized server root (`normalize_url`). `ports=None`
    means the well-known local ports; pass `ports=()` to probe only the
    explicit `urls` / `profile_urls`."""
    roots = _candidate_roots(urls, profile_urls,
                             DEFAULT_PORTS if ports is None else ports)
    found: list[tuple[str, str, list[str]]] = []
    seen: set[str] = set()
    for root in roots:
        key = normalize_url(root)
        if key in seen:
            continue
        result = _probe_root(root, timeout, headers)
        if result is not None:
            seen.add(key)
            found.append(result)
    return found


def _quant_of(entry: dict) -> str:
    """The quantization for a model entry, from the `@q4_k_m` id suffix or
    Ollama's `details.quantization_level`; "" when unknown."""
    mid = _model_id(entry) or ""
    if "@" in mid:
        return mid.partition("@")[2]
    details = entry.get("details") or {}
    return str(details.get("quantization_level") or "")


def list_models(base_url: str, timeout: float = 5.0, headers=None) -> list[str]:
    """Model ids from a server's OpenAI-compatible listing (hosted or local),
    tried with both `/v1` and `/api/v1`. Hosted callers pass an Authorization
    header in `headers`."""
    root = _root(base_url)
    for prefix in ("/v1", "/api/v1"):
        entries = _list_entries(root + prefix + "/models", timeout, headers)
        if entries is not None:
            return [mid for m in entries if (mid := _model_id(m))]
    return []


def describe_models(base_url: str, timeout: float = 5.0, headers=None,
                    vram_gb: int = 0) -> list[dict]:
    """The scan's per-model view of one server.

    Lists the server's models, attaches the metadata the scan needs
    (`size_bytes`, `context_length`, `quant`) via `LLMClient.model_info(model)`,
    and marks `loaded` where the server reports resident state (`True` loaded,
    `False` not loaded, `None` unknown). Embedding models and models too big
    for `vram_gb` (0 = no filter) are dropped. Each record carries `id`,
    `base_url`, `loaded`.
    """
    from .config import Config
    from .llm import LLMClient

    root = _root(base_url)
    entries = None
    for prefix in ("/v1", "/api/v1"):
        got = _list_entries(root + prefix + "/models", timeout, headers)
        if got is not None:
            entries = got
            break
    if entries is None:
        label = _fingerprint(root, timeout, headers)
        if label == "Ollama":
            entries = _ollama_entries(root, timeout, headers)
        elif label == "LM Studio":
            entries = _list_entries(root + "/api/v0/models", timeout, headers)
    entries = entries or []

    cfg = Config()
    cfg.base_url = base_url
    client = LLMClient(cfg)
    states = client.loaded_models() or {}

    out: list[dict] = []
    for entry in entries:
        mid = _model_id(entry)
        if not mid or looks_embedding(entry):
            continue
        info: dict = {}
        try:
            info = client.model_info(mid) or {}
        except Exception:
            info = {}
        rec: dict = {"id": mid, "base_url": base_url}
        if mid in states:
            rec["loaded"] = states[mid] == "loaded"
        else:
            rec["loaded"] = None
        if info.get("size_bytes"):
            rec["size_bytes"] = int(info["size_bytes"])
        if info.get("context_length"):
            rec["context_length"] = int(info["context_length"])
        quant = _quant_of(entry)
        if quant:
            rec["quant"] = quant
        if vram_gb and too_big_for_vram(rec.get("size_bytes"), vram_gb):
            continue
        out.append(rec)
    return out


def looks_embedding(entry: dict) -> bool:
    """True when a model entry is an embedding model, from its `type` field or
    its name. Embedding models cannot run the harness, so the scan drops them."""
    if str(entry.get("type", "")).lower() in ("embedding", "embeddings"):
        return True
    name = str(_model_id(entry) or "").lower()
    return any(marker in name for marker in _EMBEDDING_NAME_MARKERS)


def too_big_for_vram(size_bytes: int, vram_gb: int) -> bool:
    """True when a model's weights cannot fit in `vram_gb`. Unknown size or
    VRAM (0) means "can't tell", so it is not flagged."""
    if not size_bytes or not vram_gb:
        return False
    return int(size_bytes) > int(vram_gb) * _GIB
