"""Unified local model-server and hardware control plane.

    from . import servers

    mem = servers.get_hardware_memory()          # RAM + ROCm VRAM, in GB
    states = servers.list_server_states(cfg)     # online + resident models
    servers.unload_server_model("strata", cfg)   # explicit eviction
    servers.evict_other_local_servers("strata", cfg)

Why this exists
---------------
A tiered MoE server (Strata, Avalon) keeps most of the GPU's VRAM *and* a
large DDR5 working set resident. Loading a dense model on Lemonade or
llama-server beside it does not "share" memory: both servers fight for the
same pages and one of them thrashes. Pedacito therefore treats local model
servers as one pool with explicit lifecycle controls:

* `get_hardware_memory()` reads /proc/meminfo and the DRM sysfs counters
  (`mem_info_vram_total` / `mem_info_vram_used`) for every card that exposes
  them (AMD amdgpu, Intel xe, and other drivers). `profiler.snapshot()` adds
  NVIDIA/macOS/Windows readers on top without changing this shape.
* `list_server_states(config)` probes every configured local endpoint with a
  fast 1.5 s timeout and reports whether it is online and which model(s) are
  currently resident.
* `load_server_model` / `unload_server_model` speak each runtime's native
  lifecycle API (Lemonade `/v1/load` + `/v1/unload`, llama.cpp
  `/models/load` + `/models/unload`, Strata/Avalon through the same generic
  endpoints) and fall back to an optional configured `load_cmd`/`unload_cmd`
  for a single-model daemon. `pin_server_model` pins or unpins a loaded
  Lemonade model through `/internal/pin` so LRU eviction skips it; other
  server kinds have no pin concept and treat it as a no-op.
* `evict_other_local_servers` is the auto-evict handshake: before a run
  starts on one local server, resident models on the *other* local servers are
  unloaded, then VRAM/RAM is polled briefly so the target's weights can claim
  the memory.

Every network call is best-effort: a sleeping server is a normal state, never
an exception. Nothing here runs against hosted APIs.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from . import safety

GIB = 1024 ** 3
MEMINFO_PATH = Path("/proc/meminfo")
DRM_ROOT = Path("/sys/class/drm")
PROBE_TIMEOUT = 1.5
# Loading can take minutes (weights from disk + a large prefill), unloading is
# quick. The fast probe timeout above is for status only.
LOAD_TIMEOUT = 600.0
ACTION_TIMEOUT = 30.0
COMMAND_TIMEOUT = 120.0
DEFAULT_EVICT_WAIT = 10.0

_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1", "")
_LEMONADE_ENV_KEYS = ("LEMONADE_API_KEY", "LEMONADE_KEY")


# --- small helpers ------------------------------------------------------------

def _read_int(path: Path) -> int:
    try:
        return int(str(Path(path).read_text(encoding="ascii")).strip())
    except (OSError, ValueError, UnicodeDecodeError):
        return 0


def _gb(value_bytes: float) -> float:
    return round(float(value_bytes) / GIB, 2)


def _pct(used: float, total: float) -> float:
    return round(used / total * 100.0, 1) if total > 0 else 0.0


def _mapping_get(obj, key: str, default=""):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _host(base_url: str) -> str:
    try:
        return (urlparse(str(base_url or "")).hostname or "").lower()
    except ValueError:
        return ""


def is_local_url(base_url: str) -> bool:
    """True for a loopback/private-LAN/mDNS endpoint (a self-hosted server)."""
    host = _host(base_url)
    if host in _LOCAL_HOSTS or host.endswith(".local"):
        return True
    try:
        import ipaddress
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return addr.is_loopback or addr.is_link_local or addr.is_private


def normalize_url(base_url: str) -> str:
    """One canonical key per server: no trailing slash, host lower-cased,
    localhost folded to 127.0.0.1, and a `/v1`/`/api/v1` suffix removed."""
    url = str(base_url or "").strip().rstrip("/")
    try:
        parts = urlparse(url)
    except ValueError:
        return url.lower()
    scheme = (parts.scheme or "http").lower()
    host = (parts.hostname or "").lower()
    if host == "localhost":
        host = "127.0.0.1"
    port = f":{parts.port}" if parts.port else ""
    path = parts.path.rstrip("/")
    for suffix in ("/api/v1", "/v1"):
        if path.endswith(suffix):
            path = path[: -len(suffix)].rstrip("/")
            break
    return f"{scheme}://{host}{port}{path}"


def _root(base_url: str) -> str:
    """The server root (the `/v1` or `/api/v1` suffix removed)."""
    base = str(base_url or "").rstrip("/")
    for suffix in ("/api/v1", "/v1"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def _server_key(server) -> str:
    """A URL or profile name from a state dict / config body / plain string."""
    if isinstance(server, str):
        return server
    return str(_mapping_get(server, "base_url", "")
               or _mapping_get(server, "name", "")
               or _mapping_get(server, "provider", ""))


def _headers_for(server, *, internal: bool = False) -> dict:
    """Bearer auth for the local server, when one is configured.

    Lemonade accepts `LEMONADE_API_KEY` from the environment; its `/internal/*`
    endpoints prefer `LEMONADE_ADMIN_API_KEY` when that is set. A profile's
    `api_key`/`api_key_env` is used too, but an absent key is normal: local
    servers require no credential and no `Authorization` header is sent."""
    kind = str(_mapping_get(server, "kind", "") or
               _mapping_get(server, "provider", "") or
               _mapping_get(server, "type", "")).lower()
    key = str(_mapping_get(server, "api_key", "") or "")
    env_name = str(_mapping_get(server, "api_key_env", "") or "")
    if env_name:
        key = os.environ.get(env_name, "") or key
    if kind in ("lemonade", "lemond"):
        env_names = _LEMONADE_ENV_KEYS + (
            ("LEMONADE_ADMIN_API_KEY",) if internal else ())
        for env_name in env_names:
            key = os.environ.get(env_name, "") or key
    return {"Authorization": f"Bearer {key}"} if key else {}


# --- hardware telemetry -------------------------------------------------------

def get_hardware_memory(meminfo_path=None, drm_root=None) -> dict:
    """Live system RAM and DRM sysfs VRAM, in GiB.

    Shape (JSON-safe)::

        {
          "ram":  {"total_gb", "used_gb", "available_gb", "percent"},
          "vram": {"total_gb", "used_gb", "percent", "cards": [
                     {"card", "total_gb", "used_gb", "percent"}, ...]},
          "ram_total_gb", "ram_used_gb", "ram_percent",
          "vram_total_gb", "vram_used_gb", "vram_percent",
          "gpus": [...], "available": bool
        }

    Missing files are normal (no AMD GPU, a container without sysfs, a
    non-Linux box): that section reports zeros and `available` is False,
    never an exception.
    """
    meminfo = Path(meminfo_path or MEMINFO_PATH)
    drm = Path(drm_root or DRM_ROOT)

    total_kb = available_kb = 0
    try:
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            name, _, rest = line.partition(":")
            if name == "MemTotal":
                total_kb = int(rest.strip().split()[0])
            elif name == "MemAvailable":
                available_kb = int(rest.strip().split()[0])
    except (OSError, ValueError, IndexError, UnicodeDecodeError):
        total_kb = available_kb = 0
    ram_total = total_kb * 1024 / GIB
    ram_available = available_kb * 1024 / GIB
    ram_used = max(0.0, ram_total - ram_available)
    ram = {
        "total_gb": round(ram_total, 2),
        "used_gb": round(ram_used, 2),
        "available_gb": round(ram_available, 2),
        "percent": _pct(ram_used, ram_total),
    }

    cards: list[dict] = []
    seen: set[str] = set()
    try:
        candidates = sorted(drm.glob("card*/device"))
    except OSError:
        candidates = []
    for device in candidates:
        # card0 and renderD128 may point at the same device; one entry per
        # real sysfs directory keeps the total honest.
        try:
            real = str(device.resolve())
        except OSError:
            real = str(device)
        if real in seen:
            continue
        seen.add(real)
        total_bytes = _read_int(device / "mem_info_vram_total")
        used_bytes = _read_int(device / "mem_info_vram_used")
        if total_bytes <= 0 and used_bytes <= 0:
            continue
        card_name = device.parent.name
        cards.append({
            "card": card_name,
            "total_gb": _gb(total_bytes),
            "used_gb": _gb(used_bytes),
            "percent": _pct(used_bytes, total_bytes),
        })
    vram_total = sum(c["total_gb"] for c in cards)
    vram_used = sum(c["used_gb"] for c in cards)
    vram = {
        "total_gb": round(vram_total, 2),
        "used_gb": round(vram_used, 2),
        "percent": _pct(vram_used, vram_total),
        "cards": cards,
    }
    return {
        "ram": ram,
        "vram": vram,
        # Flat aliases: convenient for the header badge and older clients.
        "ram_total_gb": ram["total_gb"],
        "ram_used_gb": ram["used_gb"],
        "ram_percent": ram["percent"],
        "vram_total_gb": vram["total_gb"],
        "vram_used_gb": vram["used_gb"],
        "vram_percent": vram["percent"],
        "gpus": cards,
        "available": bool(ram["total_gb"] or vram["total_gb"]),
        "time": time.time(),
    }


def memory_badge(mem: dict | None = None) -> str:
    """The compact one-line header string: `VRAM: X.X / Y.Y GB | RAM: ...`."""
    mem = mem or get_hardware_memory()
    vram, ram = mem.get("vram", {}), mem.get("ram", {})
    return (f"VRAM: {vram.get('used_gb', 0):.1f} / "
            f"{vram.get('total_gb', 0):.1f} GB | "
            f"RAM: {ram.get('used_gb', 0):.1f} / "
            f"{ram.get('total_gb', 0):.1f} GB")


# --- configured servers -------------------------------------------------------

def _provider_of(name: str, body) -> str:
    provider = str(_mapping_get(body, "provider", "") or
                   _mapping_get(body, "server_type", "") or "").lower()
    if provider:
        return provider
    return str(name or "").lower()


def _kind_for(name: str, provider: str, base_url: str) -> str:
    """Classify a configured local server for lifecycle endpoints."""
    text = " ".join((str(provider or "").lower(),
                     str(name or "").lower())).replace("_", "-")
    if "strata" in text:
        return "strata"
    if "avalon" in text:
        return "avalon"
    if "lemonade" in text or "lemond" in text:
        return "lemonade"
    if "llama-server" in text or "llama-cpp" in text or "llamacpp" in text \
            or "llama.cpp" in text:
        return "llama-server"
    if "local-moe" in text or "localmoe" in text:
        return "local-moe"
    if "local" in text:
        return "local"
    return "custom"


def configured_local_servers(config) -> list[dict]:
    """Every distinct local endpoint named by the active config or its
    profiles. One entry per normalized URL; the model ids and profile names
    that point there ride along."""
    if config is None:
        return []
    profiles = _mapping_get(config, "profiles", {}) or {}
    active_profile = str(_mapping_get(config, "active_profile", "") or "")
    servers: dict[str, dict] = {}

    def add(name: str, body, active: bool = False):
        base = str(_mapping_get(body, "base_url", "") or "")
        if not base:
            # A profile that only names a model lives on the config's server.
            base = str(_mapping_get(config, "base_url", "") or "")
        if not base or not is_local_url(base):
            return
        key = normalize_url(base)
        if key not in servers:
            provider = _provider_of(name, body)
            servers[key] = {
                "name": name or _host(base) or "local",
                "provider": provider,
                "kind": _kind_for(name, provider, base),
                "base_url": base,
                "api_key": str(_mapping_get(body, "api_key", "") or ""),
                "api_key_env": str(_mapping_get(body, "api_key_env", "") or ""),
                "load_cmd": str(_mapping_get(body, "load_cmd", "") or ""),
                "unload_cmd": str(_mapping_get(body, "unload_cmd", "") or ""),
                "models": [],
                "profiles": [],
                "active": False,
            }
        rec = servers[key]
        model = str(_mapping_get(body, "model", "") or "")
        if model and model not in rec["models"]:
            rec["models"].append(model)
        if name and name not in rec["profiles"]:
            rec["profiles"].append(name)
        for key_name in ("load_cmd", "unload_cmd"):
            value = str(_mapping_get(body, key_name, "") or "")
            if value and not rec.get(key_name):
                rec[key_name] = value
        if not rec.get("api_key"):
            rec["api_key"] = str(_mapping_get(body, "api_key", "") or "")
        if not rec.get("api_key_env"):
            rec["api_key_env"] = str(
                _mapping_get(body, "api_key_env", "") or "")
        if active:
            rec["active"] = True
            if not rec.get("load_cmd"):
                rec["load_cmd"] = str(
                    _mapping_get(config, "load_cmd", "") or "")
            if not rec.get("unload_cmd"):
                rec["unload_cmd"] = str(
                    _mapping_get(config, "unload_cmd", "") or "")

    add(active_profile or "active", config, active=True)
    for name, body in profiles.items():
        if isinstance(body, dict):
            add(str(name), body)
    return list(servers.values())


def _resolve_server(target, config=None) -> dict | None:
    """Find one configured server by name, profile, base_url, kind, or provider
    label. `target` may itself be a state dict."""
    if isinstance(target, dict) and target.get("base_url"):
        return target
    text = str(target or "").strip()
    states = configured_local_servers(config)
    if not text:
        return states[0] if len(states) == 1 else None
    key = normalize_url(text)
    for state in states:
        if key and normalize_url(state["base_url"]) == key:
            return state
        names = [state.get("name", ""), state.get("provider", ""),
                 state.get("kind", ""), *state.get("profiles", [])]
        if any(text.lower() == str(n).lower() for n in names if n):
            return state
    # A URL that is not in the config: still a legitimate explicit target.
    if text.lower().startswith(("http://", "https://")) \
            and is_local_url(text):
        provider = "custom"
        return {"name": _host(text) or text, "provider": provider,
                "kind": _kind_for("", provider, text), "base_url": text,
                "api_key": "", "api_key_env": "", "load_cmd": "",
                "unload_cmd": "", "models": [], "profiles": [],
                "active": False}
    return None


def _models_from_payload(payload, kind: str = "") -> tuple[list[str], list[str]]:
    """(all model ids, resident model ids) from an OpenAI/Ollama/Lemonade
    payload. A resident marker wins; a single-model runtime with no marker is
    treated as holding that one model."""
    if payload is None:
        return [], []
    entries = []
    explicit_loaded: list[str] = []
    if isinstance(payload, list):
        entries = [m for m in payload if isinstance(m, dict)]
    elif isinstance(payload, dict):
        for key in ("loaded_models", "resident_models", "models_loaded"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    text = (item.get("id") or item.get("model")
                            or item.get("name") if isinstance(item, dict)
                            else item)
                    if text:
                        explicit_loaded.append(str(text))
        raw = payload.get("data")
        if raw is None:
            raw = payload.get("models")
        if isinstance(raw, dict):
            raw = [raw]
        if isinstance(raw, list):
            entries = [m for m in raw if isinstance(m, dict)]
        else:
            for key in ("model", "id", "name"):
                if payload.get(key):
                    explicit_loaded.append(str(payload[key]))
                    break
    ids: list[str] = []
    loaded: list[str] = list(dict.fromkeys(explicit_loaded))
    for mid in loaded:
        if mid not in ids:
            ids.append(mid)
    for entry in entries:
        mid = str(entry.get("id") or entry.get("model")
                  or entry.get("name") or "")
        if not mid:
            continue
        if mid not in ids:
            ids.append(mid)
        marker = None
        for key in ("loaded_in_memory", "loaded", "resident", "in_memory",
                    "is_loaded"):
            if key in entry:
                marker = bool(entry.get(key))
                break
        state = str(entry.get("state") or entry.get("status") or "").lower()
        if marker is None and state in ("loaded", "resident", "ready"):
            marker = True
        if marker is None and state in ("unloaded", "idle", "not_loaded"):
            marker = False
        if marker:
            if mid not in loaded:
                loaded.append(mid)
    # A llama-server / Strata / Avalon daemon serves its one resident model
    # and exposes no loaded flag. Lemonade lists every downloaded recipe, so
    # an absent marker there must not mean "all resident".
    if not loaded and ids and (len(ids) == 1 or kind in
                               ("llama-server", "strata", "avalon",
                                "local-moe")):
        loaded = list(ids)
    return ids, loaded


def _probe_server(server: dict, timeout: float = PROBE_TIMEOUT) -> dict:
    """Probe one server for reachability and resident models. Never raises."""
    base = str(server.get("base_url") or "")
    root = _root(base)
    headers = _headers_for(server)
    deadline = time.monotonic() + max(0.05, float(timeout or PROBE_TIMEOUT))

    def left() -> float:
        return max(0.0, deadline - time.monotonic())

    base_v1 = base.rstrip("/")
    if not re.search(r"/(?:api/)?v1$", base_v1):
        base_v1 = root + "/v1"
    urls = [
        base_v1 + "/models",
        root + "/v1/models",
        root + "/models",
        root + "/api/v1/models",
    ]
    # Deduplicate while keeping the preferred order.
    urls = list(dict.fromkeys(urls))
    online = False
    ids: list[str] = []
    loaded: list[str] = []
    kind_raw = (server.get("kind") or server.get("provider")
                or server.get("type") or "")
    kind = (_kind_for("", str(kind_raw), base) if kind_raw else "custom")
    for url in urls:
        if left() <= 0:
            break
        try:
            resp = requests.get(url, headers=headers, timeout=left())
        except Exception:
            continue
        if getattr(resp, "status_code", 0) != 200:
            continue
        online = True
        try:
            payload = resp.json()
        except Exception:
            payload = None
        if kind in ("custom", "local") and isinstance(payload, dict):
            entries = payload.get("data") or payload.get("models") or []
            if isinstance(entries, list) and any(
                    str(m.get("owned_by", "")).lower() == "lemonade"
                    for m in entries if isinstance(m, dict)):
                kind = "lemonade"
        got_ids, got_loaded = _models_from_payload(payload, kind)
        for mid in got_ids:
            if mid not in ids:
                ids.append(mid)
        for mid in got_loaded:
            if mid not in loaded:
                loaded.append(mid)
        if ids or loaded:
            break
    if not online:
        # A health endpoint still proves the daemon is up even when it does
        # not list models.
        for path in ("/health", "/api/v1/health", "/v1/health"):
            if left() <= 0:
                break
            try:
                resp = requests.get(root + path, headers=headers,
                                    timeout=left())
            except Exception:
                continue
            if getattr(resp, "status_code", 0) == 200:
                online = True
                try:
                    got_ids, got_loaded = _models_from_payload(
                        resp.json(), kind)
                except Exception:
                    got_ids, got_loaded = [], []
                ids.extend(m for m in got_ids if m not in ids)
                loaded.extend(m for m in got_loaded if m not in loaded)
                break
    out = dict(server)
    out.update({
        "kind": kind,
        "online": bool(online),
        "status": "online" if online else "offline",
        "model_ids": ids,
        "models": ids,
        "loaded_models": loaded,
        "loaded_model": loaded[0] if len(loaded) == 1 else "",
        "resident_models": loaded,
        "loaded": bool(loaded),
    })
    if not online:
        out["error"] = "offline"
    return out


def list_server_states(config, timeout: float = PROBE_TIMEOUT) -> list[dict]:
    """Probe every configured local server in turn.

    Each row carries its name/kind/base_url, `online`, the model ids it
    advertises, and `loaded_models` (the resident set). Hosted endpoints and
    duplicate spellings of one server are not listed."""
    return [_probe_server(s, timeout=timeout)
            for s in configured_local_servers(config)]


# --- lifecycle actions --------------------------------------------------------

def _post_json(url: str, headers: dict, payload: dict | None,
               timeout: float = ACTION_TIMEOUT):
    try:
        resp = requests.post(url, headers=headers, json=payload,
                             timeout=timeout)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:200]}", None
    body = ""
    try:
        body = str(getattr(resp, "text", "") or "")[:300]
    except Exception:
        pass
    ok = 200 <= int(getattr(resp, "status_code", 0)) < 300
    return ok, body or str(getattr(resp, "status_code", "")), resp


def _retry_next_spelling(resp, detail: str) -> bool:
    """Whether a lifecycle attempt may try the alternate endpoint spelling.

    Only an absent route (404), a connection failure or a timeout qualifies;
    a reachable server's real error (400/500, ...) is reported as-is so a
    side-effecting action can never fire twice."""
    try:
        status = int(getattr(resp, "status_code", 0) or 0)
    except (TypeError, ValueError):
        status = 0
    if status == 404:
        return True
    text = str(detail or "")
    return "404" in text or "Connection" in text or "Timeout" in text


def _run_hook(command: str, server: dict, model: str, cfg=None
              ) -> tuple[bool, str]:
    """Run a configured load/unload command with shell=False.

    `{model}` and `{server}` are textual substitutions, never shell syntax;
    shlex.split + shell=False keeps a project config from doing anything a
    plain argv cannot. The child gets the same allow-list-scrubbed
    environment as every other model-chosen command (`safety.child_env`),
    plus two names that cannot alias a `PEDACITO_<KEY>` config override:
    `PEDACITO_SERVER_NAME` and `PEDACITO_MODEL_PATH`.
    """
    text = str(command or "").strip()
    if not text:
        return False, "no command configured"
    if isinstance(command, (list, tuple)):
        argv = [str(part).replace("{model}", model or "").replace(
            "{server}", str(server.get("base_url") or ""))
            for part in command]
    else:
        text = text.replace("{model}", model or "").replace(
            "{server}", str(server.get("base_url") or ""))
        try:
            argv = shlex.split(text)
        except ValueError as exc:
            return False, f"cannot parse command: {exc}"
    if not argv:
        return False, "empty command"
    env = safety.child_env(cfg, extra={
        "PEDACITO_SERVER_NAME": str(
            server.get("name") or server.get("base_url") or ""),
        "PEDACITO_MODEL_PATH": model or "",
    })
    try:
        proc = subprocess.run(
            argv, shell=False, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=COMMAND_TIMEOUT,
            env=env)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:200]}"
    detail = (proc.stdout or proc.stderr or "").strip()[:300]
    return proc.returncode == 0, detail or f"exit {proc.returncode}"


def _known_model(server: dict, model: str) -> str:
    model = str(model or "").strip()
    if model:
        return model
    models = [str(m) for m in (server.get("loaded_models")
                               or server.get("models") or []) if m]
    if len(models) == 1:
        return models[0]
    return ""


def _lemonade_action(server: dict, action: str,
                     model: str) -> tuple[bool, str, str]:
    root = _root(str(server.get("base_url") or ""))
    headers = _headers_for(server, internal=action in ("pin", "unpin"))
    if action in ("pin", "unpin"):
        # Lemonade's pin endpoint is internal and takes a boolean state;
        # `/api/v1/internal/pin` is the alternate spelling older builds use.
        payload = {"model_name": model, "pinned": action == "pin"}
        urls = (root + "/internal/pin", root + "/api/v1/internal/pin")
        timeout = ACTION_TIMEOUT
    else:
        payload = {"model_name": model} if model else {}
        urls = (root + f"/v1/{action}", root + f"/api/v1/{action}")
        timeout = LOAD_TIMEOUT if action == "load" else ACTION_TIMEOUT
    for url in urls:
        ok, detail, resp = _post_json(url, headers, payload, timeout=timeout)
        if ok:
            return True, f"POST {url}", detail
        if not _retry_next_spelling(resp, detail):
            # A reachable server answered with a real error: report it rather
            # than trying a different spelling that may load or pin twice.
            return False, f"POST {url}", detail
    return False, "lemonade", "no lifecycle endpoint answered"


def _llama_action(server: dict, action: str,
                  model: str) -> tuple[bool, str, str]:
    root = _root(str(server.get("base_url") or ""))
    headers = _headers_for(server)
    payload = {"model": model} if model else {}
    for url in (root + f"/models/{action}", root + f"/v1/models/{action}"):
        ok, detail, resp = _post_json(
            url, headers, payload,
            timeout=LOAD_TIMEOUT if action == "load" else ACTION_TIMEOUT)
        if ok:
            return True, f"POST {url}", detail
        if not _retry_next_spelling(resp, detail):
            return False, f"POST {url}", detail
    return False, "llama-server", "no lifecycle endpoint answered"


def _generic_action(server: dict, action: str,
                    model: str) -> tuple[bool, str, str]:
    """Strata/Avalon/custom: try the OpenAI-ish lifecycle endpoints first."""
    root = _root(str(server.get("base_url") or ""))
    headers = _headers_for(server)
    candidates = []
    if action == "unload":
        candidates = [("/v1/unload", {"model": model} if model else {}),
                      ("/models/unload", {"model": model} if model else {})]
    else:
        candidates = [("/v1/load", {"model": model} if model else {}),
                      ("/models/load", {"model": model} if model else {})]
    for path, payload in candidates:
        url = root + path
        ok, detail, _resp = _post_json(
            url, headers, payload,
            timeout=LOAD_TIMEOUT if action == "load" else ACTION_TIMEOUT)
        if ok:
            return True, f"POST {url}", detail
    return False, "generic", "no lifecycle endpoint answered"


def _server_action(server, action: str, model: str, config) -> dict:
    if not isinstance(server, dict) or not server.get("base_url"):
        server = _resolve_server(server, config)
    if not server:
        return {"ok": False, "action": action, "model": str(model or ""),
                "detail": "no such local server"}
    kind = str(server.get("kind") or server.get("provider")
               or server.get("type") or "custom")
    kind = _kind_for("", kind, str(server.get("base_url") or ""))
    if action in ("pin", "unpin") and kind != "lemonade":
        # Pinning is a Lemonade concept. Other runtimes have no equivalent,
        # so this is an ok no-op, never an invented endpoint or a hook.
        return {
            "ok": True,
            "action": action,
            "skipped": True,
            "server": server.get("name") or _host(
                server.get("base_url", "")),
            "kind": kind,
            "base_url": server.get("base_url", ""),
            "model": str(model or ""),
            "method": "none",
            "detail": "model pinning is Lemonade-only; nothing to do",
        }
    model = _known_model(server, model)
    if action in ("pin", "unpin"):
        if not model:
            return {
                "ok": False,
                "action": action,
                "server": server.get("name") or _host(
                    server.get("base_url", "")),
                "kind": kind,
                "base_url": server.get("base_url", ""),
                "model": "",
                "method": "none",
                "detail": f"no loaded model to {action}",
            }
        ok, method, detail = _lemonade_action(server, action, model)
    elif kind in ("lemonade", "lemond"):
        ok, method, detail = _lemonade_action(server, action, model)
    elif kind == "llama-server":
        ok, method, detail = _llama_action(server, action, model)
    else:
        ok, method, detail = _generic_action(server, action, model)
    hook = str(server.get(f"{action}_cmd") or "")
    if not ok and hook:
        hook_ok, hook_detail = _run_hook(hook, server, model, config)
        if hook_ok:
            ok, method, detail = True, f"command {action}_cmd", hook_detail
        else:
            detail = f"{detail}; {action}_cmd failed: {hook_detail}"
    return {
        "ok": bool(ok),
        "action": action,
        "server": server.get("name") or _host(server.get("base_url", "")),
        "kind": kind,
        "base_url": server.get("base_url", ""),
        "model": model,
        "method": method,
        "detail": detail,
    }


def load_server_model(server, model: str = "", config=None) -> dict:
    """Load `model` on one local server (or the server's configured model).

    Lemonade: POST /v1/load {"model_name": model}. llama.cpp: POST
    /models/load {"model": model}. Strata/Avalon/custom: /v1/load then
    /models/load, then the profile's optional `load_cmd`.

    After a successful load, the optional speed probe is scheduled once per
    (server, model) on a daemon thread unless `profile_probe_on_load` is
    false. It never blocks this call."""
    result = _server_action(server, "load", model, config)
    if (result.get("ok") and config is not None
            and bool(getattr(config, "profile_probe_on_load", True))):
        try:
            from . import profiler
            profiler.maybe_probe_async(
                config, base_url=str(result.get("base_url") or ""),
                model=str(result.get("model") or ""))
        except Exception:
            pass
    return result


def unload_server_model(server, model: str = "", config=None) -> dict:
    """Unload one model (or every resident model) from a local server.

    Lemonade: POST /v1/unload {"model_name": model}; without a model it asks
    the server to unload its resident set. llama.cpp: POST /models/unload
    {"model": model}. Strata/Avalon/custom: /v1/unload then /models/unload,
    then the optional `unload_cmd`. Returns a JSON-safe result dict."""
    if not model:
        # Resolve the resident set from a state dict/configured server so
        # Lemonade gets an explicit model_name per model where possible.
        if not isinstance(server, dict) or not server.get("loaded_models"):
            state = _resolve_server(server, config)
            if state is not None:
                try:
                    state = _probe_server(state)
                except Exception:
                    state = None
                if state is not None:
                    server = state
        loaded = list(_mapping_get(server, "loaded_models", []) or [])
        if "loaded_models" in (server or {}) and len(loaded) > 1:
            results = [unload_server_model(server, m, config) for m in loaded]
            return {"ok": all(r.get("ok") for r in results),
                    "action": "unload", "server": _mapping_get(server, "name"),
                    "models": loaded, "results": results,
                    "detail": f"unloaded {sum(1 for r in results if r.get('ok'))}"
                              f" of {len(loaded)} model(s)"}
    return _server_action(server, "unload", model, config)


def unload_all_local_servers(config, only_loaded: bool = True) -> list[dict]:
    """Unload every resident model on every configured local server."""
    results = []
    for state in list_server_states(config):
        if not state.get("online"):
            continue
        if only_loaded and not state.get("loaded_models"):
            continue
        results.append(unload_server_model(state, "", config))
    return results


def pin_server_model(server, model: str = "", pinned: bool = True,
                     config=None, kind: str = "") -> dict:
    """Pin or unpin a loaded model on a Lemonade server.

    Lemonade: POST /internal/pin with
    ``{"model_name": model, "pinned": true|false}``; when the old spelling is
    needed, ``/api/v1/internal/pin`` is tried after a 404, connection failure
    or timeout, exactly like the load/unload actions. The model must already
    be loaded; without an explicit id the server's single resident model is
    used. Other server kinds have no pin concept and return an ok no-op with
    ``skipped: true`` so GUI/TUI callers need no kind check of their own.

    `kind` overrides the configured kind when a probe already identified the
    server as Lemonade (the GUI's Servers panel does this); leave it empty to
    use the profile's own classification."""
    action = "pin" if pinned else "unpin"
    if kind and not (isinstance(server, dict) and server.get("base_url")):
        resolved = _resolve_server(server, config)
        if resolved:
            server = {**resolved, "kind": kind}
    result = _server_action(server, action, model, config)
    result["pinned"] = bool(pinned)
    return result


def _target_matches(state: dict, target: str) -> bool:
    if not target:
        return False
    key = normalize_url(target)
    if key and normalize_url(state.get("base_url", "")) == key:
        return True
    names = [state.get("name", ""), state.get("provider", ""),
             state.get("kind", ""), *state.get("profiles", [])]
    return any(str(target).lower() == str(n).lower() for n in names if n)


def _memory_used(mem: dict) -> float:
    return round(float(mem.get("vram", {}).get("used_gb", 0))
                 + float(mem.get("ram", {}).get("used_gb", 0)), 2)


def evict_other_local_servers(target_provider, config,
                              wait_seconds: float = DEFAULT_EVICT_WAIT,
                              poll: bool = True) -> dict:
    """Auto-evict handshake before a run starts on `target_provider`.

    When `auto_evict_servers` is on (the default), every *other* configured
    local server holding a resident model is unloaded. Then VRAM/RAM is
    polled (up to `wait_seconds`) until it drops, so the target's weights can
    load without fighting the previous server.

    Returns `{"ok", "target", "evicted": [...], "skipped": [...],
    "waited_s", "memory"}`. Doing nothing is a success, not an error."""
    if not bool(_mapping_get(config, "auto_evict_servers", True)):
        return {"ok": True, "evicted": [], "skipped": [], "waited_s": 0.0,
                "target": str(target_provider or ""), "disabled": True,
                "memory": get_hardware_memory()}
    states = list_server_states(config)
    target = str(target_provider or "")
    if not target:
        active = [s for s in states if s.get("active")]
        target = (active[0].get("name", "") if active
                  else (states[0].get("name", "") if states else ""))
    target_state = next((s for s in states if _target_matches(s, target)), None)
    target_url = normalize_url(target_state.get("base_url", "")) \
        if target_state else normalize_url(target)
    evicted: list[dict] = []
    skipped: list[str] = []
    for state in states:
        if target_url and normalize_url(state.get("base_url", "")) == target_url:
            continue
        if not state.get("online"):
            skipped.append(str(state.get("name") or "offline server"))
            continue
        if not state.get("loaded_models"):
            skipped.append(str(state.get("name") or state.get("base_url")))
            continue
        evicted.append(unload_server_model(state, "", config))
    if not evicted or not poll:
        return {"ok": all(r.get("ok") for r in evicted),
                "evicted": evicted, "skipped": skipped, "waited_s": 0.0,
                "target": target, "memory": get_hardware_memory()}
    before = get_hardware_memory()
    baseline = _memory_used(before)
    started = time.monotonic()
    deadline = started + max(0.0, float(wait_seconds or 0))
    dropped = False
    # No GPU sysfs means there is no VRAM signal to observe; the unload
    # acknowledgement is the only reliable signal available. (RAM alone
    # fluctuates more than a model unload and would stall the run.)
    if before.get("vram", {}).get("total_gb", 0) <= 0:
        return {"ok": all(r.get("ok") for r in evicted),
                "evicted": evicted, "skipped": skipped,
                "waited_s": 0.0, "target": target, "memory": before}
    while time.monotonic() < deadline:
        time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
        now = get_hardware_memory()
        if _memory_used(now) + 0.05 < baseline:
            dropped = True
            before = now
            break
    return {"ok": all(r.get("ok") for r in evicted),
            "evicted": evicted, "skipped": skipped,
            "waited_s": round(time.monotonic() - started, 2),
            "memory_dropped": dropped, "target": target,
            "memory": before}


def server_summary(config, timeout: float = PROBE_TIMEOUT,
                   memory: dict | None = None) -> dict:
    """One JSON payload for GUI/TUI: hardware plus every configured server.

    `memory` may be a :func:`profiler.snapshot` result (a superset with the
    same shape as get_hardware_memory); omitted, the existing reader is used."""
    memory = memory or get_hardware_memory()
    states = list_server_states(config, timeout=timeout)
    return {"ok": True, "memory": memory, "hardware": memory,
            "vram": memory["vram"], "ram": memory["ram"],
            "badge": memory_badge(memory),
            "servers": states,
            "auto_evict_servers": bool(
                _mapping_get(config, "auto_evict_servers", True))}


def format_server_states(states: list[dict], mem: dict | None = None) -> str:
    """Plain-text server/hardware report for the TUI's /servers."""
    mem = mem or get_hardware_memory()
    lines = ["Local servers", f"  hardware: {memory_badge(mem)}"]
    if not states:
        lines.append("  (no local servers configured)")
        return "\n".join(lines)
    for state in states:
        status = "online" if state.get("online") else "offline"
        models = state.get("loaded_models") or []
        loaded = ", ".join(models) if models else "(no model resident)"
        lines.append(
            f"  {state.get('name') or '?'} [{state.get('kind') or 'custom'}] "
            f"{status}  {state.get('base_url', '')}")
        lines.append(f"    {loaded}")
    return "\n".join(lines)


# A tiny compatibility namespace for callers that prefer attributes.
def config_probe(base_url: str, model: str = "", provider: str = "") -> bool:
    """True when one ad-hoc local endpoint is reachable."""
    state = _probe_server({"name": provider or _host(base_url),
                           "provider": provider,
                           "kind": _kind_for(provider, provider, base_url),
                           "base_url": base_url, "models": [model],
                           "loaded_models": [], "load_cmd": "",
                           "unload_cmd": ""})
    return bool(state.get("online"))


__all__ = [
    "GIB", "PROBE_TIMEOUT", "configured_local_servers", "evict_other_local_servers",
    "format_server_states", "get_hardware_memory", "is_local_url",
    "list_server_states", "load_server_model", "memory_badge",
    "normalize_url", "pin_server_model", "server_summary",
    "unload_all_local_servers", "unload_server_model",
]
