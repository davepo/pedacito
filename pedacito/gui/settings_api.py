"""The GUI's settings endpoints: read, edit, and write Pedacito's config files.

Thin glue over `settings.py` (pure text patching) -- this module owns the file
system: which file is meant, backups, atomic writes, secrets, and the live
server probe used by "test connection".
"""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from .. import config as config_mod
from .. import config_editor
from .. import fileio
from .. import i18n
from .. import imagegen
from ..config import USER_CONFIG, ConfigError
from . import settings
from .core import STATE

SECRET_MASK_KEYS = settings.SECRET_KEYS
_CONFIG_WRITE_LOCK = threading.Lock()


def config_path(which: str) -> Path:
    """The file an edit targets: the user config or the open project's."""
    if which == "user":
        return USER_CONFIG
    if which == "project":
        root = STATE.root
        if root is None:
            raise ConfigError("Open a project to edit its config.")
        return config_mod.find_project_config(root) or root / ".pedacito.toml"
    raise ConfigError(f"unknown config file {which!r}")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def write_config(path: Path, text: str) -> None:
    """Validate, back up the old file as `<name>.bak`, and write atomically."""
    settings.validate_text(text, str(path))
    with _CONFIG_WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                Path(str(path) + ".bak").write_bytes(path.read_bytes())
            except OSError:
                pass
        fileio.write_bytes(path, text.encode("utf-8"))


def _mask(body: dict) -> dict:
    out = dict(body)
    for key in SECRET_MASK_KEYS:
        if key in out:
            out[key] = ""
            out.setdefault("_secret_set", [])
            out["_secret_set"] = [*out["_secret_set"], key]
    return out


def file_info(which: str) -> dict:
    """One config file's state for the editor."""
    try:
        path = config_path(which)
    except ConfigError as e:
        return {"which": which, "available": False, "error": str(e)}
    text = _read_text(path)
    info = {"which": which, "available": True, "path": str(path),
            "exists": path.exists(), "text": settings.redact_secrets(text),
            "profiles": {}, "settings": {}, "error": ""}
    if text.strip():
        try:
            flat, profiles = settings.validate_text(text, str(path))
            info["settings"] = {
                k: (config_editor.redact_mcp_servers_dict(v)
                    if k == "mcp_servers" and isinstance(v, dict)
                    else (list(v) if isinstance(v, tuple) else v))
                for k, v in flat.items()
                if k not in SECRET_MASK_KEYS}
            info["settings_secret_set"] = [k for k in flat
                                           if k in SECRET_MASK_KEYS]
            info["profiles"] = {n: _mask(b) for n, b in profiles.items()}
        except ConfigError as e:
            info["error"] = str(e)
    if which == "project" and path.exists():
        info["trusted"] = config_mod.is_trusted(path)
        info["exec_fields"] = list(config_mod.EXEC_FIELDS)
    return info


def _merged_profile_view(cfg, out: dict):
    """The live config with every file-only profile layered in: the file
    bodies (project over user, per key) merged over the live profiles, so a
    profile that only exists in a file is judged by ITS model/base_url, not
    the default one. Shared by the specialty hints, the calibration badges,
    and the hosted-model checks."""
    import copy

    base = cfg if cfg is not None else config_mod.Config()
    file_profiles: dict = {}
    for which in ("user", "project"):      # project layer wins, per key
        for name, b in ((out.get(which) or {}).get("profiles") or {}).items():
            file_profiles.setdefault(name, {}).update(b or {})
    names = set(base.profiles or {}) | set(file_profiles)
    view = copy.copy(base)
    view.profiles = {n: dict((base.profiles or {}).get(n) or {})
                     for n in names}
    for n, b in file_profiles.items():
        view.profiles[n].update(b)
    return view


def _specialty_hints(cfg, out: dict) -> dict:
    """Per-profile specialty tags for the profile form: what the config sets
    explicitly, and what the model card / name heuristics detect."""
    from .. import roles
    view = _merged_profile_view(cfg, out)
    hints: dict = {}
    for name in sorted(view.profiles):
        body = view.profiles.get(name) or {}
        try:
            explicit = list(config_mod._coerce("specialties",
                                               body.get("specialties") or []))
        except ConfigError:
            explicit = []
        try:
            detected = list(roles.detected_specialties(view, name))
        except Exception:
            detected = []
        hints[name] = {"explicit": explicit, "detected": detected}
    return hints


def _calibration_map(cfg, out: dict) -> dict:
    """Per-profile calibration badges for Settings -> Profiles: the raw
    card's stamp (an expired card still reports stale) plus endpoint-based
    hosted-ness, so the row can say `Calibrated on ... (score: ...)` or
    `Not calibrated yet`, and hosted rows show the cloud badge instead."""
    from .. import modelcard
    view = _merged_profile_view(cfg, out)
    cal = modelcard.calibration_map(view)
    for name, meta in cal.items():
        body = view.profiles.get(name) or {}
        meta["specialties"] = [str(t) for t in (body.get("specialties") or [])]
    return cal


def hosted_profile_error(name: str) -> str:
    """The exact refusal text when `name` is a hosted provider profile,
    else "". Shared by the calibrate endpoint and the batch job so CLI and
    GUI refuse identically. The merged view covers file-only profiles."""
    from .. import modelcard, roles
    cfg = STATE.cfg
    view = _merged_profile_view(cfg, {"user": file_info("user"),
                                      "project": file_info("project")})
    try:
        hosted = roles.profile_is_hosted(view, name)
    except Exception:
        hosted = False
    if not hosted:
        return ""
    body = view.profiles.get(name) or {}
    model = body.get("model") or view.model or ""
    return modelcard.hosted_calibration_error(name, model)


def payload() -> dict:
    cfg = STATE.cfg
    if cfg is None and STATE.root is None:
        # A GUI session can start with no project open. The Settings dialog
        # still needs the user's global profiles and the active model, so read
        # the global config instead of falling back to bare Config defaults.
        try:
            cfg = STATE.global_config()
        except Exception:
            cfg = None
    out = {
        "schema": settings.schema(),
        # Universal settings must always carry a concrete effective value:
        # the open project's resolved one when there is a config, else the
        # built-in `Config()` defaults (a GUI session can start without a
        # project open, and the user file may not exist yet).
        "resolved": settings.resolved(cfg if cfg is not None
                                      else config_mod.Config()),
        "active_profile": cfg.active_profile if cfg is not None else "",
        "tiers": list(config_mod.PRESET_TIERS),
        "user": file_info("user"),
        "project": file_info("project"),
        "user_config_path": str(USER_CONFIG),
        "project_open": STATE.root is not None,
        "specialty_tags": [{"tag": t, "label": m["label"]}
                           for t, m in config_mod.SPECIALTIES.items()],
    }
    out["specialty_hints"] = _specialty_hints(cfg, out)
    out["calibration"] = _calibration_map(cfg, out)
    return out


def _changes_clean(changes: dict) -> dict:
    """Drop an empty secret (the form leaves it blank to mean 'unchanged')."""
    return {k: v for k, v in changes.items()
            if not (k in SECRET_MASK_KEYS and v in ("", None))}


def apply_changes(which: str, profile: str | None, changes: dict,
                  removes: list[str]) -> dict:
    path = config_path(which)
    text = _read_text(path)
    if not text and not path.exists() and which == "user":
        raise ConfigError("No user config file yet. Use \u201cCreate config\u201d "
                          "first.")
    new = settings.set_values(text, _changes_clean(changes), removes,
                              profile=profile or None)
    write_config(path, new)
    return payload()


def profile_action(which: str, action: str, name: str, values: dict,
                   source: str = "") -> dict:
    path = config_path(which)
    text = _read_text(path)
    if action == "create":
        new = settings.add_profile(text, name, _changes_clean(values))
    elif action == "delete":
        new = settings.remove_profile(text, name)
    elif action == "duplicate":
        body = settings.profile_body(text, source)
        if not body and source not in settings.profile_names(text):
            raise ConfigError(f"no profile named '{source}'")
        new = settings.add_profile(text, name, body)
    else:
        raise ConfigError(f"unknown profile action {action!r}")
    write_config(path, new)
    if action == "delete":
        with STATE.lock:
            was_selected = (
                STATE.profile_request == name
                or (STATE.cfg is not None
                    and STATE.cfg.active_profile == name)
            )
            if STATE.profile_request == name:
                STATE.profile_request = ""
            STATE._user_cfg_stamp = None
            STATE._user_raw_prefs = None
            STATE._global_cfg_stamp = None
        if was_selected:
            return reload_state_config(profile=None)
    return payload()


def rename_profile(which: str, old_name: str, new_name: str) -> dict:
    """Rename `[profiles.OLD]` to `[profiles.NEW]` in place (comments and
    every other key survive), retarget `default_profile` and `roles`
    references, and reload STATE so the header dropdown and the settings
    payload show the new name at once."""
    old_name = str(old_name or "").strip()
    raw_new = str(new_name or "").strip()
    new_name = _sanitize_profile_name(raw_new)
    # version dots never fail a rename either (`qwen3.8` -> `qwen3_8`), but a
    # truly wrong human name (an org `path/`) is refused, like the import.
    if not new_name or not PROFILE_NAME_RE.fullmatch(new_name):
        raise ConfigError(
            f"profile name {raw_new!r} must use letters, digits, '-' and "
            "'_' (starting with a letter or digit, max "
            f"{_MAX_PROFILE_NAME} characters)")
    path = config_path(which)
    text = _read_text(path)
    known = config_editor.profile_names(text)
    if old_name not in known:
        raise ConfigError(f"no profile named '{old_name}' in this file")
    if new_name == old_name:
        raise ConfigError(f"the profile is already named '{old_name}'")
    if new_name in known:
        raise ConfigError(f"a profile named '{new_name}' already exists")
    write_config(path, config_editor.rename_profile(text, old_name, new_name))
    # Keep the selection following the renamed profile: an explicit
    # user-chosen profile (STATE.profile_request) or the active one.
    with STATE.lock:
        requested = STATE.profile_request
        active = STATE.cfg.active_profile if STATE.cfg is not None else ""
    target = new_name if (requested == old_name or active == old_name) else None
    out = reload_state_config(profile=target)
    out["renamed"] = {"from": old_name, "to": new_name}
    out["profiles"] = sorted(set((out["user"].get("profiles") or {}))
                             | set((out["project"].get("profiles") or {})))
    return out


def save_raw(which: str, text: str) -> dict:
    """Save the raw editor's text. The view ships secrets masked as
    `***REDACTED***`; a save that still carries the sentinel restores the
    stored key (like the profile form's empty secret field), so editing any
    other line never wipes -- or leaks -- the API key."""
    path = config_path(which)
    text = settings.restore_secrets(text, _read_text(path))
    write_config(path, text)
    return payload()


def create_user_config(base_url: str, model: str, vram_gb) -> dict:
    if USER_CONFIG.exists():
        raise ConfigError(f"{USER_CONFIG} already exists.")
    if not base_url or not model:
        raise ConfigError("Give the server URL and the model name.")
    try:
        vram = int(vram_gb) if vram_gb not in (None, "") else None
    except (TypeError, ValueError):
        raise ConfigError("vram_gb must be a whole number of gigabytes.")
    write_config(USER_CONFIG, settings.new_user_config(base_url, model, vram))
    return payload()


def trust_project_config() -> dict:
    path = config_path("project")
    if not path.exists():
        raise ConfigError("This project has no config file to trust.")
    config_mod.trust_project(path)
    return payload()


def _url_origin(url: str) -> tuple[str, str, int] | None:
    """Return `(scheme, hostname, port)` for an http(s) URL, else None."""
    try:
        parsed = urlparse(str(url or "").strip())
    except ValueError:
        return None
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https") or not parsed.hostname:
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    if port is None:
        port = 443 if scheme == "https" else 80
    return (scheme, parsed.hostname.lower(), port)


def _stored_api_key(which: str, profile: str, probe_url: str = "") -> str:
    probe_origin = _url_origin(probe_url) if probe_url else None
    try:
        text = _read_text(config_path(which))
        body = settings.profile_body(text, profile) if profile else {}
        flat, _ = settings.validate_text(text)
        if body.get("api_key"):
            stored_url = body.get("base_url") or flat.get("base_url") or ""
            if probe_origin is None or _url_origin(stored_url) == probe_origin:
                return str(body["api_key"])
            return ""
        if flat.get("api_key"):
            stored_url = ((body.get("base_url") if profile else "")
                          or flat.get("base_url") or "")
            if probe_origin is None or _url_origin(stored_url) == probe_origin:
                return str(flat["api_key"])
        return ""
    except (ConfigError, OSError):
        return ""


def probe_server(base_url: str, api_key: str, which: str = "user",
                 profile: str = "") -> dict:
    """List the models a server reports (the "test connection" button)."""
    base_url = (base_url or "").strip()
    if not base_url:
        return {"ok": False, "error": "Enter the server URL first."}
    probe_origin = _url_origin(base_url)
    if probe_origin is None:
        return {"ok": False, "error": "Server URL must use http or https."}
    if not api_key:
        api_key = _stored_api_key(which, profile, base_url)
        if not api_key and STATE.cfg is not None and STATE.cfg.api_key:
            if _url_origin(getattr(STATE.cfg, "base_url", "")) == probe_origin:
                api_key = STATE.cfg.api_key
    try:
        r = requests.get(base_url.rstrip("/") + "/models",
                         headers={"Authorization": f"Bearer {api_key}"},
                         timeout=8)
        r.raise_for_status()
        ids = sorted({m.get("id", "?") for m in r.json().get("data", [])})
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    return {"ok": True, "models": ids}


def _endpoint_auth(flat: dict, profiles: dict) -> dict:
    """Per-server Authorization headers for a discovery scan.

    A profile's own `api_key`/`api_key_env` is attached to that profile's
    endpoint only; a profile that points at the universal endpoint may inherit
    `[universal]`'s key. Keys never cross origins, so scanning several local
    servers cannot leak one server's credential to another. Strata/Avalon also
    accept their conventional environment variables when the profile labels
    itself as one of those runtimes. Returns `{normalized_url: headers}` for
    `discovery.discover_servers(auth=...)` and `describe_models(auth=...)`.
    """
    from .. import discovery

    flat_url = str(flat.get("base_url") or "")

    def key_for(body: dict, fallback: bool) -> str:
        key = str(body.get("api_key") or "")
        env = str(body.get("api_key_env") or "")
        if env:
            key = os.environ.get(env, "") or key
        if not key and str(body.get("provider") or "").lower() in (
                "strata", "avalon", "local-moe", "local_moe", "localmoe"):
            for env in ("STRATA_API_KEY", "AVALON_API_KEY"):
                key = os.environ.get(env, "") or key
        if not key and fallback:
            key = str(flat.get("api_key") or "")
            env = str(flat.get("api_key_env") or "")
            if env:
                key = os.environ.get(env, "") or key
        return key

    auth: dict[str, dict] = {}
    for body in (profiles or {}).values():
        if not isinstance(body, dict):
            continue
        base = str(body.get("base_url") or flat_url or "")
        if not base:
            continue
        same_origin = bool(flat_url) and _url_origin(base) == _url_origin(flat_url)
        key = key_for(body, same_origin)
        if key:
            auth[discovery.normalize_url(base)] = {
                "Authorization": f"Bearer {key}"}
    if flat_url:
        key = key_for(flat, False)
        if key:
            auth.setdefault(discovery.normalize_url(flat_url),
                            {"Authorization": f"Bearer {key}"})
    return auth


def list_profiles() -> dict:
    """The profiles table (identity + measured scores + recommendations) for
    the settings page, from the same renderer the CLI uses."""
    from .. import profiles as profiles_mod
    cfg = STATE.cfg if STATE.cfg is not None else config_mod.Config()
    rows = profiles_mod.list_profiles(cfg)
    return {"profiles": rows, "recommendations": profiles_mod.recommend(rows)}


def _user_scan_config() -> tuple[dict, dict]:
    """The user config's flat settings and profile bodies for a scan."""
    try:
        return settings.validate_text(_read_text(config_path("user")))
    except ConfigError:
        return {}, {}


def _scan_profile_urls(flat: dict, profiles: dict) -> list[str]:
    """Every server URL the user config already names, deduplicated."""
    urls = [flat.get("base_url")] if flat.get("base_url") else []
    urls += [body.get("base_url") for body in (profiles or {}).values()
             if isinstance(body, dict) and body.get("base_url")]
    return list(dict.fromkeys(str(u) for u in urls if u))


def scan_servers() -> dict:
    """Discover servers and list each one's models with metadata (loaded,
    size, context, embedding/too-big already dropped).

    Existing profiles are probed too, with their own API key: a Strata server
    behind a key is not invisible just because it listens on a custom port or
    its OpenAI listing rejects an unauthenticated probe."""
    from .. import discovery
    flat, profiles = _user_scan_config()
    auth = _endpoint_auth(flat, profiles)
    servers = []
    seen: set[str] = set()
    for label, base_url, _ids in discovery.discover_servers(
            timeout=1.5, profile_urls=_scan_profile_urls(flat, profiles),
            auth=auth):
        key = _norm_url(base_url)
        if key in seen:
            continue          # the same server spelled two ways: probe once
        seen.add(key)
        servers.append({"label": label, "base_url": base_url,
                        "models": discovery.describe_models(base_url,
                                                            auth=auth)})
    return {"servers": servers}


# --- reset to baseline defaults ------------------------------------------------

# The sections of the baseline file: (section, blurb, (key, extra-comment)).
# Values always come from the built-in Config() defaults -- the recommended
# pre-configuration for small local models (reasoning-first steps, the brief
# edit plan, the incremental edit loop, tight read/gather budgets).
_BASELINE_SECTIONS = (
    ("universal", "Connection, generation, and shared defaults.", (
        ("base_url", "the model server; import real servers from Settings \u2192 Model Discovery & Profiles"),
        ("model", "the model id exactly as the server reports it"),
        ("json_mode", '"schema" for llama.cpp-based servers, "object" for hosted APIs'),
        ("tool_mode", '"grammar" (schema-constrained steps) or "native" (the model\'s tool calls)'),
        ("temperature", None),
        ("temperature_answer", "cool (not greedy) answers: escape repetition loops, still copy SEARCH text exactly"),
        ("timeout", None),
        ("load_wait_seconds", None),
        ("context_tokens", "fallback window; the server\'s real length wins with auto_detect_context"),
        ("reserve_tokens", None),
        ("auto_fit", None),
        ("auto_detect_context", None),
    )),
    ("agent", "Agent-loop budgets, tuned for small local models.", (
        ("max_steps", "0 = no lookup cap; the budgets below and the model\'s `done` bound it"),
        ("max_total_model_calls", "hard ceiling on model calls per task"),
        ("max_no_progress_steps", "ends gather after this many fruitless steps"),
        ("max_read_tokens", "token budget for one read during gather"),
        ("max_read_lines", None),
        ("max_gathered_chars", None),
        ("read_skeleton", None),
        ("max_rehydration_rounds", None),
        ("step_reasoning_first", "state the reason before picking the action"),
        ("edit_plan_brief", "1-3 lines naming the file, the lines, and why"),
        ("incremental_edit_loop", "one symbol per verified micro-loop"),
        ("repair_rounds", None),
        ("post_edit_check", None),
        ("auto_repair_rounds", None),
        ("auto_smoke", None),
        ("use_model_card", "let `pedacito calibrate` pick the remaining defaults"),
    )),
    ("editing", "Editing and workspace behaviour.", (
        ("backup", None),
        ("log_sessions", None),
        ("allow_git", None),
        ("manage_gitignore", None),
        ("create_ignore_file", None),
    )),
)


def baseline_text() -> str:
    """The clean baseline config.toml text: recommended Config() defaults,
    no model profiles and no per-role assignments."""
    cfg = config_mod.Config()
    lines = [
        "# Pedacito baseline configuration.",
        "#",
        "# \"Reset to Baseline\" restored the recommended defaults for small",
        "# local models and removed every saved profile and per-role model",
        "# assignment. The previous file was kept next to this one as a",
        "# config.toml.reset.<timestamp>.bak copy.",
        "#",
        "# Next step: open Settings \u2192 Model Discovery & Profiles and scan your",
        "# local servers (or run `pedacito models scan`), then import the models",
        "# you use. Every setting is documented in `pedacito config --reference`.",
    ]
    for section, blurb, keys in _BASELINE_SECTIONS:
        lines.append("")
        lines.append(f"[{section}]")
        lines.append(f"# {blurb}")
        if section == "universal":
            lines.append("vram_gb = 8   # whole GB of VRAM on the model machine;"
                         " auto_fit sizes the window from it (hosted APIs ignore it)")
        for key, note in keys:
            value = getattr(cfg, key)
            if value is None:
                continue  # TOML has no null: unset lets the card/default decide
            tail = f"   # {note}" if note else ""
            lines.append(f"{key} = {settings.literal(value)}{tail}")
    lines.append("")
    return "\n".join(lines)


def reset_to_baseline() -> dict:
    """Back the user config up with a timestamp, wipe all profiles and role
    assignments, and restore the recommended baseline defaults."""
    path = config_path("user")
    backup = ""
    text = baseline_text()
    settings.validate_text(text, str(path))
    with _CONFIG_WRITE_LOCK:
        if path.exists():
            stamp = time.strftime("%Y%m%d-%H%M%S")
            bp = Path(f"{path}.reset.{stamp}.bak")
            bp.write_bytes(path.read_bytes())
            backup = str(bp)
        path.parent.mkdir(parents=True, exist_ok=True)
        fileio.write_bytes(path, text.encode("utf-8"))
    out = reload_state_config()
    out["backup"] = backup
    return out


# --- GUI theme ------------------------------------------------------------------

def save_theme(name: str) -> dict:
    """Persist `gui_theme` in the user config and reflect it in the live
    theme state only. Purely cosmetic, so nothing is reloaded or restarted and
    live `STATE.client`/`STATE.agent` are never swapped (safe mid-job). When no
    user config exists yet it is created from the starter template first, so
    a project-less GUI session still remembers the choice."""
    name = str(name or "").strip().lower()
    themes = list(config_mod._ENUM_FIELDS["gui_theme"])
    if name not in themes:
        raise ConfigError(f"unknown theme {name!r}. Use one of: "
                          + ", ".join(themes))
    created = not USER_CONFIG.exists()
    text = _read_text(USER_CONFIG) if not created else config_mod.TEMPLATE
    write_config(USER_CONFIG, settings.set_values(text, {"gui_theme": name}))
    with STATE.lock:
        if STATE.cfg is not None:
            STATE.cfg.gui_theme = name
        if STATE._user_cfg is not None:
            STATE._user_cfg.gui_theme = name
        if STATE._global_cfg is not None:
            STATE._global_cfg.gui_theme = name
        STATE._user_cfg_stamp = None
        STATE._user_raw_prefs = None
        STATE._global_cfg_stamp = None
    return {"gui_theme": name, "themes": themes, "persisted": True,
            "created": created}


def save_language(lang: str) -> dict:
    """Persist `language` in the user config and reflect it in the live
    language state only. Purely UI/cosmetic, so nothing is reloaded or
    restarted and live `STATE.client`/`STATE.agent` are never swapped
    (safe mid-job). When no user config exists yet it is created from the
    starter template first."""
    raw = str(lang or "").strip().lower()
    langs = list(i18n.SUPPORTED_LANGUAGES)
    if raw not in langs:
        raise ConfigError(
            f"unknown language {raw!r}. Use one of: " + ", ".join(langs))
    with _CONFIG_WRITE_LOCK:
        res = i18n.save_user_language(raw)
    code = res["language"]
    i18n.set_current_lang(code)
    with STATE.lock:
        if STATE.cfg is not None:
            STATE.cfg.language = code
        if STATE._user_cfg is not None:
            STATE._user_cfg.language = code
        if STATE._global_cfg is not None:
            STATE._global_cfg.language = code
        STATE._user_cfg_stamp = None
        STATE._user_raw_prefs = None
        STATE._global_cfg_stamp = None
    return {"language": code, "supported": langs,
            "persisted": bool(res.get("persisted", True)),
            "created": bool(res.get("created", False))}


# --- dual-GPU Scout + Architect pairing ---------------------------------------

def configure_scout(body: dict) -> dict:
    """Save a Scout + Architect pairing through `configure_scout_architect`.

    `body`: {architect, scout?, architect_spec?, scout_spec?, create_scout?,
    scout_base_url?, scout_model?, scout_vram_gb?, scout_tier?}. With
    `create_scout` and no existing scout profile, a starter table on a
    secondary port is written so the pairing works on a one-GPU machine too.
    """
    body = body or {}
    with STATE.lock:
        cfg = STATE.cfg
    architect = str(body.get("architect")
                    or body.get("architect_profile") or "").strip()
    if not architect and cfg is not None:
        architect = str(cfg.active_profile or cfg.default_profile or "")
    scout = str(body.get("scout") or body.get("scout_profile") or "").strip()
    path = config_path("user")
    text = _read_config_text(path)
    known = config_editor.profile_names(text)

    architect_spec = body.get("architect_spec")
    if not isinstance(architect_spec, dict):
        architect_spec = None
    scout_spec = body.get("scout_spec")
    if not isinstance(scout_spec, dict):
        scout_spec = None
    if body.get("create_scout") and scout and scout not in known \
            and scout_spec is None:
        try:
            vram = int(body.get("scout_vram_gb") or 8)
        except (TypeError, ValueError):
            vram = 8
        scout_spec = {
            "base_url": str(body.get("scout_base_url")
                            or "http://localhost:11434/v1").strip(),
            "model": str(body.get("scout_model") or "qwen3:4b").strip(),
            "vram_gb": max(1, vram),
            "tier": str(body.get("scout_tier") or "small").strip(),
        }
    if not architect:
        raise ConfigError("Pick the architect profile (the planner/editor "
                          "model) first.")
    if not scout:
        raise ConfigError("Pick the scout profile (the fast gather model) or "
                          "create a starter one.")
    new = config_editor.configure_scout_architect(
        text, architect_profile=architect, scout_profile=scout,
        architect_spec=architect_spec, scout_spec=scout_spec)
    write_config(path, new)
    reload_state_config(profile=architect)
    out = payload()
    out["scout"] = {"architect": architect, "scout": scout,
                    "path": str(path)}
    out["status"] = STATE.status()
    return out


# --- named MCP servers ---------------------------------------------------------

def _mcp_specs(path: Path) -> dict[str, dict]:
    """The `[mcp_servers.<name>]` tables in one config file (never raises:
    the settings panel must render even while a hand edit is broken)."""
    text = _read_text(path)
    if not text.strip():
        return {}
    try:
        flat, _profiles = settings.validate_text(text, str(path))
    except ConfigError:
        return {}
    raw = flat.get("mcp_servers")
    if not isinstance(raw, dict):
        return {}
    return {str(k): dict(v) for k, v in raw.items() if isinstance(v, dict)}


def _mcp_config():
    """The live config, or the user config when no project is open."""
    with STATE.lock:
        cfg = STATE.cfg
    if cfg is not None:
        return cfg
    try:
        return config_mod.load(root=None, require_hardware=False,
                               warn_budgets=False, upgrade_user_config=False)
    except ConfigError:
        return config_mod.Config()


def mcp_servers_payload() -> dict:
    """Every named MCP server plus the tools each enabled one exposes.

    The user config is the primary list. A trusted project file may add
    servers (it layers over the user config), and the live tool listing comes
    from `mcp.list_mcp_tools`: a down server lists nothing but is still shown.
    """
    from .. import mcp
    user_path = USER_CONFIG
    specs: dict[str, dict] = {}
    sources: dict[str, str] = {}
    for name, body in _mcp_specs(user_path).items():
        specs[name] = body
        sources[name] = "user"
    project_path = None
    try:
        project_path = config_path("project")
    except ConfigError:
        project_path = None
    if project_path is not None and project_path.exists() \
            and config_mod.is_trusted(project_path):
        for name, body in _mcp_specs(project_path).items():
            specs[name] = body
            sources[name] = "project"
    cfg = _mcp_config()
    tools = mcp.list_mcp_tools(cfg)
    out = []
    for name in sorted(specs):
        body = specs[name]
        spec = mcp.normalise_named_spec(body) or {}
        out.append({
            "name": name,
            "source": sources.get(name, "user"),
            "transport": "http" if str(body.get("url") or "").strip()
            else "stdio",
            "command": str(body.get("command") or ""),
            "args": [str(a) for a in (body.get("args") or [])]
            if isinstance(body.get("args"), (list, tuple)) else [],
            "url": str(body.get("url") or ""),
            "env": config_editor.redact_mcp_env(body.get("env"))
            if isinstance(body.get("env"), dict) else {},
            "enabled": bool(body.get("enabled", True)),
            "trusted": bool(body.get("trusted", False)),
            "tools": [{"name": t["name"], "description": t["description"]}
                      for t in tools if t.get("server") == name],
            "live": bool(spec),
        })
    return {"servers": out, "tools": tools,
            "user_config": str(user_path),
            "project_config": str(project_path) if project_path else "",
            "enabled": bool(getattr(cfg, "enable_mcp", True))}


def _mcp_write_target(body: dict) -> tuple[Path, str]:
    """(path, which) for an MCP edit, enforcing the §21 trust boundary."""
    which = str(body.get("file") or "user").strip().lower()
    if which not in ("user", "project"):
        raise ConfigError(f"unknown config file {which!r}")
    if which == "user":
        return USER_CONFIG, "user"
    if STATE.root is None:
        # No project to host a trusted [mcp_servers.*] table: save globally
        # instead of failing the Add/Save button.
        return USER_CONFIG, "user"
    path = config_path("project")
    if not path.exists():
        raise ConfigError(
            f"{path} does not exist yet; create and trust it before adding "
            "execution-capable MCP servers there")
    if not config_mod.is_trusted(path):
        raise ConfigError(
            f"Refusing to write MCP servers to {path}: the project config is "
            "not trusted. Approve it in Settings, or save the server in the "
            "user config instead (the project file can run commands).")
    return path, "project"


def _mcp_body_from_request(body: dict) -> tuple[str, str, dict]:
    name = str(body.get("name") or "").strip()
    if not name:
        raise ConfigError("Give the MCP server a name.")
    transport = str(body.get("transport") or "").strip().lower()
    if transport not in ("stdio", "http"):
        transport = "http" if str(body.get("url") or "").strip() else "stdio"
    spec: dict = {"enabled": bool(body.get("enabled", True)),
                  "trusted": bool(body.get("trusted", False))}
    if transport == "http":
        url = str(body.get("url") or "").strip()
        if not url:
            raise ConfigError("An HTTP MCP server needs a url.")
        spec.update({"url": url, "command": None, "args": None})
    else:
        command = str(body.get("command") or "").strip()
        if not command:
            raise ConfigError(
                "A stdio MCP server needs a command (the executable to start).")
        args = body.get("args")
        if args is None:
            args = []
        spec.update({"command": command, "args": args, "url": None})
    if "env" in body:
        spec["env"] = body.get("env")
    return name, transport, spec


def _restore_mcp_env(env, stored_env: dict) -> dict | None:
    if env is None:
        return None
    norm = config_editor._normalise_mcp_env(env)
    if norm is None:
        return None
    out: dict[str, str] = {}
    for k, v in norm.items():
        if v == config_editor.REDACTED_SECRET:
            orig = stored_env.get(k) if isinstance(stored_env, dict) else None
            if orig in (None, ""):
                raise ConfigError(
                    f"MCP env `{k}` is masked as {config_editor.REDACTED_SECRET} "
                    "but has no stored value to restore.")
            out[k] = str(orig)
        else:
            out[k] = v
    return out


def save_mcp_server(body: dict) -> dict:
    """Add or update one named server. Writes USER_CONFIG, or a project file
    only when it is trusted; every write is validated and backed up."""
    from .. import mcp
    body = body or {}
    name, _transport, spec = _mcp_body_from_request(body)
    path, which = _mcp_write_target(body)
    text = _read_config_text(path)
    if "env" in spec and spec["env"] is not None:
        stored_env = (_mcp_specs(path).get(name) or {}).get("env") or {}
        spec["env"] = _restore_mcp_env(spec["env"], stored_env)
    new = config_editor.upsert_mcp_server(text, name, spec)
    write_config(path, new)
    if which == "project":
        # The user just approved this exact edit; re-mark it so the server is
        # live immediately instead of lapsed by the write's own hash change.
        config_mod.trust_project(path)
    mcp.forget_server(name)
    with STATE.lock:
        keep = STATE.profile_request or None
    reload_state_config(profile=keep)
    out = mcp_servers_payload()
    out["saved"] = name
    out["path"] = str(path)
    return out


def delete_mcp_server(body: dict) -> dict:
    """Remove one named server from the chosen config file."""
    from .. import mcp
    body = body or {}
    name = str(body.get("name") or "").strip()
    if not name:
        raise ConfigError("Name the MCP server to remove.")
    path, which = _mcp_write_target(body)
    text = _read_config_text(path)
    new = config_editor.remove_mcp_server(text, name)
    write_config(path, new)
    if which == "project":
        config_mod.trust_project(path)
    mcp.forget_server(name)
    with STATE.lock:
        keep = STATE.profile_request or None
    reload_state_config(profile=keep)
    out = mcp_servers_payload()
    out["deleted"] = name
    out["path"] = str(path)
    return out


def _stored_mcp_spec(name: str) -> dict:
    raw = _mcp_specs(USER_CONFIG).get(name) or {}
    if raw:
        return raw
    try:
        pp = config_path("project")
    except ConfigError:
        pp = None
    if pp is not None and pp.exists() and config_mod.is_trusted(pp):
        return _mcp_specs(pp).get(name) or {}
    return {}


def test_mcp_server(body: dict) -> dict:
    """Probe one server (saved or as edited in the dialog) for 5 seconds."""
    from .. import mcp
    body = body or {}
    name = str(body.get("name") or "").strip() or "test"
    if body.get("command") or body.get("url") or body.get("transport"):
        env = body.get("env") or {}
        if isinstance(env, dict) and any(
                v == config_editor.REDACTED_SECRET for v in env.values()):
            stored_env = _stored_mcp_spec(name).get("env") or {}
            env = _restore_mcp_env(env, stored_env) or {}
        raw = {"command": body.get("command"),
               "args": body.get("args") or [],
               "url": body.get("url"),
               "env": env,
               "enabled": bool(body.get("enabled", True)),
               "trusted": bool(body.get("trusted", False))}
    else:
        raw = _stored_mcp_spec(name)
        if not raw:
            raise ConfigError(f"no MCP server named '{name}'")
    return mcp.probe_named_server(name, raw, budget=5.0)


# --- Auto-Tune Universal Settings -----------------------------------------------

# The wizard answers are closed vocabularies; anything else is refused.
AUTOTUNE_QUESTIONS = {
    "hardware": ("lt8", "8to16", "16to24", "gt24", "hosted"),
    "model": ("small", "medium", "large", "hosted"),
    "codebase": ("small", "medium", "large"),
    "priority": ("token_saver", "balanced", "thorough"),
    # Optional fifth answer for a two-GPU box. A scout handles the gather
    # lookup loop on the second card, so the architect can spend its window
    # on planning and edits. "none" keeps the single-GPU recommendation.
    "dual_gpu": ("none", "dual24_8", "dual24_16"),
}

# Starting budgets per hardware tier: (max_read_tokens, max_gathered_chars,
# max_total_model_calls, max_no_progress_steps). A small VRAM box pays for
# every extra token with evictions and re-prefills, so gather stays lean.
_HW_TIERS = {
    "lt8": (800, 12000, 25, 3),
    "8to16": (1000, 16000, 30, 3),
    "16to24": (1400, 24000, 35, 4),
    "gt24": (2000, 32000, 45, 5),
    "hosted": (2000, 32000, 50, 5),
}


def _autotune_values(answers: dict) -> tuple[dict, dict]:
    """Deterministic (values, reasons) for the wizard answers."""
    hw = answers["hardware"]
    model = answers["model"]
    code = answers["codebase"]
    prio = answers["priority"]
    dual = answers.get("dual_gpu") or "none"
    read, gathered, calls, no_progress = _HW_TIERS[hw]
    if dual != "none":
        # The architect owns the big card; the scout does the reading. These
        # are the architect-side budgets, sized for a 24 GB planner.
        read, gathered, calls, no_progress = (
            (1200, 18000, 40, 4) if dual == "dual24_8"
            else (1600, 24000, 45, 5))
    weak_model = model in ("small", "medium")
    reasons: dict[str, str] = {}

    # Codebase size moves the budgets before the priority trade-off does: a
    # big repo needs bigger reads and more steps to map it, a small one is
    # mostly context already.
    if code == "large":
        calls = int(calls * 1.2 + 0.5)
        gathered = int(gathered * 1.25)
        if hw != "lt8":
            read = int(read * 1.15)
        reason_size = "a large codebase needs more lookup steps"
    elif code == "small":
        gathered = int(gathered * 0.75)
        reason_size = "a small codebase fits in fewer, smaller observations"
    else:
        reason_size = "a medium codebase sits at the tier defaults"

    # The priority answer decides where the remaining headroom goes.
    if prio == "token_saver":
        calls = int(calls * 0.8 + 0.001)
        no_progress = max(2, no_progress - 1)
        gathered = int(gathered * 0.8)
        verify = "off" if hw == "lt8" else "deterministic"
        best_of = 1
        reason_prio = "spending fewest tokens: tighter caps and no extra model rounds"
    elif prio == "thorough":
        calls = int(calls * 1.25 + 0.5)
        no_progress += 1
        verify = "full"
        best_of = 2 if weak_model else 1
        reason_prio = "quality first: more attempts, a verifier call, and two answer candidates on weaker models"
    else:
        verify = "deterministic"
        best_of = 1
        reason_prio = "balanced defaults for this tier"

    values: dict = {
        "max_read_tokens": read,
        "max_gathered_chars": gathered,
        "max_total_model_calls": calls,
        "max_no_progress_steps": no_progress,
        "verify_answer": verify,
        "best_of": best_of,
    }
    # Structured, incremental editing is the single biggest help for weaker
    # local models; big/hosted models hold a whole multi-file patch fine.
    if weak_model:
        values.update({
            "incremental_edit_loop": True,
            "step_reasoning_first": True,
            "edit_plan_brief": True,
            "dynamic_tool_pruning": True,
            "repair_rounds": 3,
            "post_edit_check": True,
        })
        reasons["incremental_edit_loop"] = (
            "one symbol per verified micro-loop keeps a "
            + ("small " if model == "small" else "mid-size ")
            + "model from emitting a broken multi-file patch")
        reasons["step_reasoning_first"] = (
            "state the reason before the action: fewer pointless steps on a "
            "model that does not plan ahead")
        reasons["edit_plan_brief"] = (
            "a 1-3 line plan before each edit block measurably improves block "
            "matching on weaker models")
        reasons["dynamic_tool_pruning"] = (
            "drop tools the task will not use: a small context spends its "
            "window on code, not on schema")
        reasons["repair_rounds"] = (
            "one more SEARCH/REPLACE retry round pays off when the model "
            "copies search text imperfectly")
        reasons["post_edit_check"] = (
            "a cheap per-file syntax/lint gate catches a broken hunk right "
            "after it lands, while the model still remembers the file")
    else:
        values["dynamic_tool_pruning"] = False
        values["step_reasoning_first"] = True
        reasons["dynamic_tool_pruning"] = (
            "a strong model with room to spare: keep every tool visible "
            "rather than pruning the list mid-task")
        reasons["step_reasoning_first"] = (
            "reasoning before the action still costs almost nothing and keeps "
            "the transcript auditable")

    reasons["max_read_tokens"] = (
        (f"Dual-GPU scout/architect: {read} tokens per architect gather read; "
         "the scout's own lookups stay on its card"
         if dual != "none" else
         f"{hw.upper() if hw != 'hosted' else 'hosted'} hardware: "
         + (f"{read} tokens per gather read keeps one observation cheap"
            if hw in ("lt8", "8to16") else
            f"{read} tokens per gather read fits the window comfortably")))
    reasons["max_gathered_chars"] = (
        f"cap on what one task accumulates: {reason_size}"
        + ("; the scout keeps lookup detail out of the architect's window"
           if dual != "none" else ""))
    reasons["max_total_model_calls"] = (
        f"hard ceiling per task: {calls} calls, {reason_prio}"
        + (" (architect-side; scout lookups have their own budget)"
           if dual != "none" else ""))
    reasons["max_no_progress_steps"] = (
        f"end gather after {no_progress} fruitless steps, {reason_prio}")
    reasons["verify_answer"] = (
        "check answers " + ("only with deterministic scans"
                             if verify == "deterministic" else
                             "with one verifier model call"
                             if verify == "full" else
                             "off: every extra round costs tokens"))
    reasons["best_of"] = (
        f"sample {best_of} candidate answer(s) and keep the best-scoring one")
    if prio == "thorough":
        reasons["max_read_tokens"] += "; thorough still caps one read"
    return values, reasons


def autotune(answers: dict, apply: bool = False) -> dict:
    """The \"Auto-Tune Universal Settings\" wizard: deterministic, offline
    recommendations for the universal budgets from four questions. Preview
    (`apply` false) writes nothing; applying stores the changes in the user
    config as `[universal]` values and reloads the live config."""
    clean: dict = {}
    for key, allowed in AUTOTUNE_QUESTIONS.items():
        val = str((answers or {}).get(key) or "").strip().lower()
        if key == "dual_gpu" and not val:
            val = "none"
        if val not in allowed:
            raise ConfigError(
                f"{key} must be one of: {', '.join(allowed)}")
        clean[key] = val
    values, reasons = _autotune_values(clean)
    cfg = STATE.cfg if STATE.cfg is not None else config_mod.Config()
    changes = [{"key": k, "value": v, "current": getattr(cfg, k),
                "reason": reasons.get(k, "")}
               for k, v in values.items()]
    out = {"answers": clean, "changes": changes, "applied": False}
    dual = clean.get("dual_gpu") or "none"
    if dual != "none":
        # The pairing itself is a separate one-click write (the scout profile
        # must exist first), so the preview recommends it without silently
        # creating a profile from a budget calculation.
        scout_gb = 8 if dual == "dual24_8" else 16
        scout_tier = "tiny" if scout_gb <= 8 else "small"
        out["dual_gpu"] = {
            "layout": dual,
            "architect_vram_gb": 24,
            "scout_vram_gb": scout_gb,
            "scout_tier": scout_tier,
            "suggested_scout_name": "scout",
            "scout_spec": {
                "base_url": "http://localhost:11434/v1",
                "model": "qwen3:4b" if scout_gb <= 8 else "qwen3:8b",
                "vram_gb": scout_gb,
                "tier": scout_tier,
            },
            "reason": (
                "Dual-GPU: keep the 24 GB card for planning, answers and "
                f"edits; a {scout_gb} GB scout serves the gather lookup loop "
                "on its own port so its reads never evict the architect's "
                "prefix."),
        }
    if not apply:
        return out
    path = config_path("user")
    text = _read_text(path)
    if not text and not path.exists():
        raise ConfigError("No user config file yet. Use \u201cCreate config\u201d "
                          "first.")
    write_config(path, settings.set_values(text, dict(values)))
    with STATE.lock:
        keep = STATE.profile_request or None
    out["payload"] = reload_state_config(profile=keep)
    out["applied"] = True
    return out


# --- immediate STATE config reload --------------------------------------------

def reload_state_config(profile: str | None = None) -> dict:
    """Re-read the config files into STATE so /api/state and /api/settings
    show the change at once (the profile dropdown included), without waiting
    for a project re-open job. Returns the fresh settings payload.

    `profile` forces the profile selection after the reload (the rename
    endpoint passes it when the renamed profile was the explicitly chosen
    one); None keeps the files' own selection and clears any stale request."""
    from ..agent import Agent
    from ..llm import LLMClient
    with STATE.lock:
        root = STATE.root
    try:
        cfg = config_mod.load(root=root, profile=profile,
                              require_hardware=False, warn_budgets=False,
                              upgrade_user_config=False)
    except ConfigError:
        cfg = None
    with STATE.lock:
        STATE.cfg = cfg
        STATE.client = LLMClient(cfg) if cfg is not None else None
        STATE.agent = (Agent(STATE.idx, STATE.client, cfg)
                       if cfg is not None and STATE.idx is not None else None)
        STATE.profile_request = profile or ""
        STATE.model_checked, STATE.model_status, STATE.model_ok = False, "", None
    return payload()


# --- selective model discovery and import ---------------------------------------

def _norm_url(url) -> str:
    """The canonical spelling of a server URL (see `discovery.normalize_url`):
    no trailing slash, `localhost` folded to `127.0.0.1`, `/v1` suffix gone."""
    from .. import discovery
    return discovery.normalize_url(url)


# What the profile editor (and the settings page) accept as a profile name.
PROFILE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}")
_MAX_PROFILE_NAME = 32


def _sanitize_profile_name(raw_name: str) -> str:
    """Harden a profile name so version numbers survive the round trip.

    Model ids carry things a profile name may not: `qwen3.8-27b` smuggles a
    dot, `glm-4.7-flash:latest` a colon, `mistral@7b` an at-sign. Each run of
    characters outside `[a-z 0-9 _ -]` becomes one `_` (`3.8` -> `3_8`), the
    name is lower-cased, repeated and dangling separators are collapsed and
    trimmed, a redundant `-gguf`/`_gguf` tail is dropped, and the result is
    capped at `_MAX_PROFILE_NAME`. A `/` is deliberately left untouched: it
    marks a name a human typed (an org path or a slip), and the import check
    should refuse that loudly instead of guessing a rewrite.
    """
    s = str(raw_name or "").strip().lower()
    s = re.sub(r"[^a-z0-9_/-]+", "_", s)
    s = re.sub(r"[-_]{2,}", lambda m: m.group(0)[0], s)
    s = s.strip("-_")
    while len(s) > 4 and s.endswith(("-gguf", "_gguf")):
        s = s[: -len("-gguf")].strip("-_")
    if len(s) > _MAX_PROFILE_NAME:
        s = s[:_MAX_PROFILE_NAME].strip("-_")
    return s


def _suggest_profile_name(model_id: str, taken: set, label: str = "") -> str:
    """The scan's suggested profile name: `config_editor.short_name`'s slug,
    sanitized (version dots and all), and made unique with `_2`, `_3`, ... if
    two ids collapse to the same slug."""
    base = _sanitize_profile_name(config_editor.short_name(model_id, taken,
                                                           label)) or "model"
    name = base
    i = 2
    while name in taken:
        tail = "_%d" % i
        name = base[:_MAX_PROFILE_NAME - len(tail)].strip("-_") + tail
        i += 1
    taken.add(name)
    return name


def scan_models(urls=(), vram_gb=0, budget=20.0) -> dict:
    """`pedacito models scan`'s discovery pass WITHOUT the writes: probe the
    well-known local ports, every configured profile's server, and `urls`;
    return one selectable row per model. Nothing is written to the config.

    Candidates arrive spelled many ways (`localhost` vs `127.0.0.1`, a
    trailing `/v1`, a trailing slash), so every server is keyed by its
    normalized URL: one server probes once, and the returned list holds one
    row per (normalized server URL, model id) -- never a duplicate.

    `budget` (seconds) is a wall-clock cap on the WHOLE scan, shared by the
    discovery pass and every per-server description: a config with many slow
    remote profile URLs returns the servers/models found so far instead of
    blocking the GUI request past a reasonable budget."""
    from .. import discovery
    budget = max(0.0, float(budget))
    deadline = time.monotonic() + budget
    # Discovery is the cheap pass; reserve half the budget for describing the
    # servers it finds, so one slow candidate cannot spend the whole allowance
    # probing and starve the useful output (the found servers' model rows).
    discovery_deadline = time.monotonic() + budget * 0.5
    explicit = [str(u).strip() for u in (urls or ()) if str(u).strip()]
    try:
        vram = int(vram_gb or 0)
    except (TypeError, ValueError):
        vram = 0
    flat, existing = _user_scan_config()
    # Re-find servers on custom ports from the same config the import reads:
    # one source of truth, not the CLI's own config global. The profiles'
    # API keys travel with their URLs, so a key-protected local server
    # (Strata/Avalon) is listed instead of answering 401 and disappearing.
    profile_urls = _scan_profile_urls(flat, existing)
    auth = _endpoint_auth(flat, existing)
    found = discovery.discover_servers(timeout=2.0, urls=explicit,
                                       profile_urls=profile_urls,
                                       deadline=discovery_deadline,
                                       auth=auth)
    imported = {(_norm_url(b.get("base_url", flat.get("base_url", ""))),
                 str(b.get("model") or "")) for b in existing.values()}
    taken = set(existing)
    servers_out, models_out = [], []
    seen_servers: set[str] = set()
    seen_models: set[tuple[str, str]] = set()
    for label, base_url, _ids in found:
        if time.monotonic() >= deadline:
            break                       # budget spent: keep the servers so far
        skey = _norm_url(base_url)
        if skey and skey in seen_servers:
            continue        # the same server spelled another way: probe once
        seen_servers.add(skey)
        try:
            recs = discovery.describe_models(base_url, vram_gb=vram,
                                             deadline=deadline, auth=auth)
        except Exception:
            recs = []
        rows = []
        for rec in recs:
            mid = str(rec.get("id") or "")
            if not mid or (skey, mid) in seen_models:
                continue    # one row per (normalized server URL, model id)
            seen_models.add((skey, mid))
            already = (skey, mid) in imported
            name = _suggest_profile_name(mid, taken, label)
            is_img = imagegen.is_image_model(
                mid, labels=rec.get("labels"), recipe=rec.get("recipe"))
            row = {"server_url": base_url, "provider": label, "model_id": mid,
                   "base_url": base_url, "loaded": rec.get("loaded"),
                   "context_length": int(rec.get("context_length") or 0),
                   "size_bytes": int(rec.get("size_bytes") or 0),
                   "quant": str(rec.get("quant") or ""),
                   "modality": "image" if is_img else "text",
                   "is_image_model": bool(is_img),
                   "suggested_profile_name": name,
                   "already_imported": bool(already)}
            rows.append(row)
            models_out.append(row)
        servers_out.append({"provider": label, "base_url": base_url,
                            "models": rows})
    return {"servers": servers_out, "models": models_out}


def _read_config_text(path: Path) -> str:
    """The text to add profiles to: the existing file, else the starter
    template (so a first import produces a valid config)."""
    text = _read_text(path)
    return text if text.strip() else config_mod.TEMPLATE


def import_models(models) -> dict:
    """Write ONLY the user-selected profiles into the user config (with a
    .bak backup), then reload STATE so they appear in the header dropdown.

    `models` is a list of {profile_name, base_url, model, context_length,
    vram_gb, tier, set_default}. A name that already exists is skipped, not
    duplicated."""
    if not isinstance(models, list) or not models:
        raise ConfigError("Pick at least one model to import.")
    path = config_path("user")
    text = _read_config_text(path)
    imported: list[str] = []
    skipped: list[str] = []
    default = ""
    default_is_image = False
    imported_text = False
    for sel in models:
        if not isinstance(sel, dict):
            continue
        name = str(sel.get("profile_name") or sel.get("name") or "").strip()
        mid = str(sel.get("model") or sel.get("model_id") or "").strip()
        base = str(sel.get("base_url") or sel.get("server_url") or "").strip()
        if not name or not mid or not base:
            raise ConfigError("Each imported model needs a profile name, a "
                              "model id, and a server URL.")
        # version numbers never fail an import: `qwen3.8` is auto-fixed to
        # `qwen3_8`. A name a human typed truly wrong (an org `path/`, say)
        # stays invalid and is refused rather than silently rewritten.
        fixed = _sanitize_profile_name(name)
        if not fixed or not PROFILE_NAME_RE.fullmatch(fixed):
            raise ConfigError(
                f"profile name {name!r} must use letters, digits, '-' and "
                "'_' (starting with a letter or digit, max "
                f"{_MAX_PROFILE_NAME} characters)")
        name = fixed
        is_img = bool(sel.get("is_image_model")) \
            or str(sel.get("modality") or "").strip() == "image" \
            or imagegen.is_image_model(mid)
        if name in config_editor.profile_names(text):
            skipped.append(name)
            if sel.get("set_default"):
                default = default or name
            continue
        values: dict = {"base_url": base, "model": mid}
        if is_img:
            values["modality"] = "image"
        else:
            imported_text = True
        tier = str(sel.get("tier") or "").strip()
        if not tier:
            tier = config_mod._guess_tier(name, values) or ""
        if tier:
            values["tier"] = tier
        for key, raw in (("vram_gb", sel.get("vram_gb")),
                         ("context_tokens", sel.get("context_length"))):
            try:
                n = int(raw or 0)
            except (TypeError, ValueError):
                n = 0
            if n > 0:
                values[key] = n
        text = config_editor.add_profile(text, name, values)
        imported.append(name)
        if sel.get("set_default"):
            default = name
            default_is_image = is_img
    if not imported and not skipped:
        raise ConfigError("Nothing to import: the selection was empty.")
    # An image generator can never run the chat agent, so when a text model is
    # imported in the same batch an image model must not become the default.
    if default and default_is_image and imported_text:
        default = ""
    if default:
        if default not in config_editor.profile_names(text):
            raise ConfigError(f"default profile '{default}' was not imported")
        text = settings.set_values(text, {"default_profile": default})
    write_config(path, text)
    out = reload_state_config()
    out["imported"] = imported
    out["skipped"] = skipped
    out["default_profile"] = default or out["active_profile"]
    return out


def known_profiles() -> list[str]:
    """Profile names the calibration picker may offer: the live config's, or
    the user file's when no project (and no loaded config) is open."""
    with STATE.lock:
        cfg = STATE.cfg
    if cfg is not None and cfg.profiles:
        return sorted(cfg.profiles)
    try:
        return sorted(config_editor.profile_names(
            _read_text(config_path("user"))))
    except (ConfigError, OSError):
        return []


# --- pre-configured hosted provider profiles -------------------------------------

# The major hosted providers, spelled the way Pedacito reaches them: every
# entry is an OpenAI-compatible chat-completions server (`llm.py` speaks that
# one protocol, and Anthropic publishes their API behind the same
# `/v1/chat/completions` surface). `provider` names the vendor for the UI.
# A preset carries only public facts -- base URL, the env var that holds the
# key, curated model ids, and a generous context budget. Nothing secret.
CLOUD_PROVIDERS: tuple[dict, ...] = (
    {
        "id": "anthropic", "label": "Claude (Anthropic)", "provider": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY", "suggest": "claude",
        "context_tokens": 200000,
        "models": ["claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-5"],
    },
    {
        "id": "openai", "label": "ChatGPT (OpenAI)", "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY", "suggest": "openai",
        "context_tokens": 272000,
        "models": ["gpt-5.4", "gpt-5.4-mini", "o3-mini", "gpt-4.1",
                   "gpt-4.1-mini"],
    },
    {
        "id": "deepseek", "label": "DeepSeek", "provider": "openai",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY", "suggest": "deepseek",
        "context_tokens": 128000,
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
    {
        "id": "qwen", "label": "Qwen (DashScope)", "provider": "openai",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY", "suggest": "qwen",
        "context_tokens": 256000,
        "models": ["qwen3-coder-plus", "qwen-max", "qwen-plus", "qwen-flash"],
    },
    {
        "id": "gemini", "label": "Google Gemini", "provider": "openai",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env": "GEMINI_API_KEY", "suggest": "gemini",
        "context_tokens": 200000,
        "models": ["gemini-2.5-pro", "gemini-2.5-flash"],
    },
    {
        "id": "openrouter", "label": "OpenRouter", "provider": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY", "suggest": "openrouter",
        "context_tokens": 200000,
        "models": ["deepseek/deepseek-chat", "qwen/qwen3-coder",
                   "anthropic/claude-3.7-sonnet"],
    },
)

# Generous gather reads for cloud windows (the `hosted` tier's own budgets are
# the floor; a pasted key should never have to fight tiny local-model caps).
_CLOUD_READ_TOKENS = 3000


def _preset_by_id(preset_id: str) -> dict:
    pid = str(preset_id or "").strip().lower()
    for p in CLOUD_PROVIDERS:
        if p["id"] == pid:
            return p
    raise ConfigError(f"unknown provider preset {preset_id!r}. Use one of: "
                      + ", ".join(p["id"] for p in CLOUD_PROVIDERS))


def provider_presets() -> dict:
    """`GET /api/providers/presets`: the quick-add catalog for the UI. The
    route wraps this in `{"ok": True, ...}`. Every key here must exist on the
    JS mirror `PDC.FALLBACK_PROVIDER_CATALOG` in static/app.js (and vice
    versa): the UI seeds from the fallback and replaces it with this answer."""
    return {"presets": [dict(p) for p in CLOUD_PROVIDERS]}


# Tokens that add nothing to an auto-suggested profile name: the vendor's own
# words (the preset's family prefix already says it) and pure version numbers.
# Keep style words (sonnet, mini, reasoner...) -- `claude-sonnet-4-5` should
# suggest `claude_sonnet`, not bare `claude`.
_FAMILY_WORDS = {"claude", "anthropic", "gpt", "chatgpt", "openai",
                 "deepseek", "qwen", "gemini", "google", "dashscope",
                 "openrouter", "o3", "o4"}


def _preset_profile_suggestion(preset: dict, model: str) -> str:
    """`claude_sonnet`, `deepseek_chat`, `gemini_flash`...: the preset's short
    family word plus up to two meaningful tokens of the model id (org paths
    like `qwen/qwen3-coder` lose the path, version dots lose the numbers)."""
    family = str(preset.get("suggest") or preset["id"])
    tail: list[str] = []
    for tok in re.split(r"[^A-Za-z0-9]+", model):
        low = tok.lower()
        if not tok or low in _FAMILY_WORDS or low == family or low.isdigit() \
                or (low != family and low.startswith(family)):
            continue
        tail.append(low)
        if len(tail) == 2:
            break
    if not tail:
        tail = [re.sub(r"[^A-Za-z0-9]+", "_", model).lower()]
    return _sanitize_profile_name("_".join([family] + tail)) or family


def add_provider_preset(body: dict) -> dict:
    """One-click hosted provider profile: the preset supplies the
    base URL, tier, context budgets and provider quirks; the user supplies an
    API key or the name of an environment variable that holds one. Writes the
    profile into the config file, reloads STATE, and returns the fresh
    settings payload so the topbar Profile dropdown can offer it immediately.

    `body`: {preset_id, model?, profile_name?, api_key?, api_key_env?,
             set_default?, file?}. A pasted key wins over the env var name
    (`api_key_env` beats `api_key` at request time, so only one is stored).

    The default profile name is derived from the MODEL (`_preset_profile_
    suggestion`), never from the vendor alone; the hosted-provider dialog
    passes an explicit model-id-based `profile_name` (step B of the probe
    flow), so every profile the UI creates reads as `claude_sonnet_4_5`,
    `deepseek_chat`, `gemini_2_5_flash`..."""
    body = body or {}
    preset = _preset_by_id(body.get("preset_id"))
    model = str(body.get("model") or "").strip() or preset["models"][0]
    raw_name = str(body.get("profile_name") or "").strip()
    name = _sanitize_profile_name(
        raw_name or _preset_profile_suggestion(preset, model))
    if not name or not PROFILE_NAME_RE.fullmatch(name):
        raise ConfigError(
            f"profile name {raw_name!r} must use letters, digits, '-' and "
            "'_' (starting with a letter or digit, max "
            f"{_MAX_PROFILE_NAME} characters)")
    api_key = str(body.get("api_key") or "").strip()
    api_key_env = str(body.get("api_key_env") or "").strip()
    if not api_key:
        api_key_env = api_key_env or preset["api_key_env"]
    values: dict = {"base_url": preset["base_url"], "model": model,
                    "tier": "hosted", "json_mode": "object",
                    "context_tokens": int(preset.get("context_tokens") or 0),
                    "max_read_tokens": _CLOUD_READ_TOKENS}
    # The provider table wins for the endpoints whose compatibility differs
    # from the generic hosted default (Anthropic's OpenAI-compatible surface
    # rejects json_object and cannot cache).
    from .. import providers
    values.update(providers.profile_overrides(preset["base_url"], model))
    if api_key:
        values["api_key"] = api_key
    elif api_key_env:
        values["api_key_env"] = api_key_env
    which = "project" if str(body.get("file") or "user") == "project" else "user"
    path = config_path(which)
    text = _read_config_text(path)
    if name in config_editor.profile_names(text):
        raise ConfigError(f"a profile named '{name}' already exists in {path}")
    text = config_editor.add_profile(text, name, values)
    set_default = bool(body.get("set_default"))
    if set_default:
        text = settings.set_values(text, {"default_profile": name})
    write_config(path, text)
    with STATE.lock:
        requested = STATE.profile_request
    # Selecting the new profile is the point of a one-click add only when it
    # is made the default; otherwise keep the profile the user was running.
    out = reload_state_config(profile=name if set_default
                              else (requested or None))
    out["added"] = name
    out["preset"] = preset["id"]
    out["model"] = model
    out["uses_env"] = bool(api_key_env and not api_key)
    return out


# --- live provider probe and model-named multi-import --------------------------
#
# Pedacito stays local-first: the hosted-provider dialog never trusts a key it
# has not used. `probe_provider` authenticates against the provider's own
# `/models` endpoint from this machine (one GET, 8s timeout), and every model
# the provider reports comes back with a profile name derived from the MODEL
# id (`deepseek_chat`, `gemini_2_5_flash`), uniquified against the profiles
# already in the config. On an auth or network failure the curated preset ids
# stay importable, so the dialog works even while the key needs fixing.

# Obvious non-chat model families: embeddings, speech, images, moderation.
# Dropping them keeps a coding agent's model picker honest about its size.
_NON_CHAT_HINTS = ("embedding", "whisper", "tts", "dall-e", "moderation",
                   "audio", "speech", "transcribe", "imagen", "image")

# Anthropic's OpenAI-compatible surface (and its native /v1/messages) take the
# key as `x-api-key` with a version stamp; sending both spellings costs nothing
# and lets either authentication path verify the key.
_ANTHROPIC_VERSION = "2023-06-01"


def _clean_model_id(raw) -> str:
    """`models/gemini-2.5-flash` (how Google's OpenAI endpoint spells its ids)
    becomes `gemini-2.5-flash`; everything else just trims."""
    mid = str(raw or "").strip()
    if mid.startswith("models/"):
        mid = mid[len("models/"):].strip()
    return mid


def _probe_headers(preset_id: str, base_url: str, key: str) -> dict:
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    if preset_id == "anthropic" or "anthropic.com" in str(base_url or "").lower():
        headers["x-api-key"] = key
        headers["anthropic-version"] = _ANTHROPIC_VERSION
    return headers


def _existing_profile_index(which: str) -> tuple[set, set]:
    """(profile names taken, imported (normalized URL, model) pairs) for one
    config file -- the uniqueness and 'already imported' bookkeeping."""
    try:
        flat, profiles = settings.validate_text(_read_text(config_path(which)))
    except (ConfigError, OSError):
        return set(), set()
    imported = {(_norm_url(b.get("base_url", flat.get("base_url", ""))),
                 str(b.get("model") or "")) for b in profiles.values()}
    return set(profiles), imported


def _provider_rows(items, base_url: str, taken: set, imported: set) -> list:
    """Probe payloads -> dialog rows: cleaned ids, obvious non-chat models
    dropped, one row per id, each carrying a unique model-derived profile
    name and whether this (server, model) pair is already configured."""
    skey = _norm_url(base_url)
    rows: list = []
    seen: set = set()
    for it in items or []:
        if isinstance(it, str):
            raw = it
        elif isinstance(it, dict):
            raw = it.get("id") or it.get("name") or ""
        else:
            raw = ""
        mid = _clean_model_id(raw)
        if not mid or mid in seen:
            continue
        if any(tok in mid.lower() for tok in _NON_CHAT_HINTS):
            continue        # embeddings/audio/moderation never lead a picker
        seen.add(mid)
        rows.append({"id": mid,
                     "suggested_profile_name": _suggest_profile_name(mid, taken),
                     "already_imported": (skey, mid) in imported})
    return rows


_KNOWN_PROVIDER_ENV_VARS = frozenset(
    p["api_key_env"] for p in CLOUD_PROVIDERS if p.get("api_key_env"))
_PROVIDER_ENV_KEY_RE = re.compile(r"^[A-Z0-9_]+_(?:API_KEY|KEY|TOKEN)$")


def _allowed_provider_env_var(name: str) -> bool:
    return bool(name and (name in _KNOWN_PROVIDER_ENV_VARS
                          or _PROVIDER_ENV_KEY_RE.fullmatch(name)))


def probe_provider(body: dict) -> dict:
    """`POST /api/providers/probe`: verify a hosted provider key live against
    `<base_url>/models` and return the model list the dialog imports from.

    `body`: {preset_id?, base_url?, api_key?, api_key_env?, file?}. The
    effective key is the pasted `api_key` when non-empty, else
    `os.environ[api_key_env]` (the preset's own env var when none was named).
    An unknown `preset_id` is refused; `custom` (or none) means "any
    OpenAI-compatible URL". Success: {"ok": True, "verified": True, "models":
    [{"id", "suggested_profile_name", "already_imported"}]}. Failure: {"ok":
    False, "error": "...", "models": <curated preset ids>} so the user can
    still pick a model and fix the key later."""
    body = body or {}
    pid = str(body.get("preset_id") or "").strip().lower()
    preset = _preset_by_id(pid) if pid and pid != "custom" else None
    base_url = str(body.get("base_url")
                   or (preset or {}).get("base_url") or "").strip()
    if not base_url:
        raise ConfigError("Give the provider base URL first.")
    if _url_origin(base_url) is None:
        raise ConfigError("Provider base URL must use http or https.")
    api_key = str(body.get("api_key") or "").strip()
    env = str(body.get("api_key_env") or "").strip() \
        or str((preset or {}).get("api_key_env") or "")
    key = api_key or (os.environ.get(env, "") if _allowed_provider_env_var(env) else "")
    which = "project" if str(body.get("file") or "user") == "project" else "user"
    taken, imported = _existing_profile_index(which)
    error = ""
    items: list = []
    try:
        r = requests.get(base_url.rstrip("/") + "/models",
                         headers=_probe_headers(pid, base_url, key), timeout=8)
        r.raise_for_status()
        payload = r.json()
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            items = payload.get("data") or payload.get("models") or []
        if not isinstance(items, list):
            items = []
    except Exception as e:
        error = f"{type(e).__name__}: {str(e)[:200]}"
    if error:
        return {"ok": False, "verified": False, "error": error,
                "base_url": base_url,
                "models": _provider_rows([{"id": m} for m in
                                          (preset or {}).get("models", [])],
                                         base_url, taken, imported)}
    return {"ok": True, "verified": True, "base_url": base_url, "preset": pid,
            "models": _provider_rows(items, base_url, taken, imported)}


def import_hosted_models(body: dict) -> dict:
    """`POST /api/providers/import`: the hosted-provider dialog's "import N
    models in one click". Every selected model becomes its own profile, named
    after the MODEL (sanitized and uniquified against the file's profiles --
    never after the vendor), all sharing the dialog's base URL and key (or
    env var) and the preset's hosted budgets.

    `body`: {preset_id?, base_url?, api_key?, api_key_env?, file?,
             models: [{model, profile_name?, set_default?}]}. Names that
    already exist are skipped, not duplicated."""
    body = body or {}
    pid = str(body.get("preset_id") or "").strip().lower()
    preset = _preset_by_id(pid) if pid and pid != "custom" else None
    base_url = str(body.get("base_url")
                   or (preset or {}).get("base_url") or "").strip()
    if not base_url:
        raise ConfigError("Give the provider base URL first.")
    sels = body.get("models")
    if not isinstance(sels, list) or not sels:
        raise ConfigError("Pick at least one model to import.")
    api_key = str(body.get("api_key") or "").strip()
    api_key_env = "" if api_key else (
        str(body.get("api_key_env") or "").strip()
        or str((preset or {}).get("api_key_env") or ""))
    ctx = int((preset or {}).get("context_tokens") or 0)
    which = "project" if str(body.get("file") or "user") == "project" else "user"
    path = config_path(which)
    text = _read_config_text(path)
    known = set(config_editor.profile_names(text))
    imported: list[str] = []
    skipped: list[str] = []
    default = ""
    for sel in sels:
        if not isinstance(sel, dict):
            continue
        mid = _clean_model_id(sel.get("model") or sel.get("model_id") or "")
        if not mid:
            raise ConfigError("Each imported model needs its exact model id.")
        raw_name = str(sel.get("profile_name") or "").strip()
        # The profile name defaults to the sanitized MODEL id; a name typed
        # in the dialog's slug column wins, and collisions get `_2`, `_3`...
        # (`_suggest_profile_name` consumes its `taken` set, so it gets a
        # copy: the suggestion must not look "already taken" on the next
        # line — `known` itself grows only through real imports below.)
        name = _sanitize_profile_name(raw_name) if raw_name \
            else _suggest_profile_name(mid, set(known))
        if not name or not PROFILE_NAME_RE.fullmatch(name):
            raise ConfigError(
                f"profile name {raw_name!r} must use letters, digits, '-' and "
                f"'_' (starting with a letter or digit, max "
                f"{_MAX_PROFILE_NAME} characters)")
        if name in known:
            skipped.append(name)
            if sel.get("set_default"):
                default = default or name
            continue
        values: dict = {"base_url": base_url, "model": mid, "tier": "hosted",
                        "json_mode": "object",
                        "max_read_tokens": _CLOUD_READ_TOKENS}
        if ctx > 0:
            values["context_tokens"] = ctx
        # The provider table wins where a surface deviates from the generic
        # hosted default (Anthropic rejects json_object and cannot cache).
        from .. import providers
        values.update(providers.profile_overrides(base_url, mid))
        if api_key:
            values["api_key"] = api_key
        elif api_key_env:
            values["api_key_env"] = api_key_env
        text = config_editor.add_profile(text, name, values)
        known.add(name)
        imported.append(name)
        if sel.get("set_default"):
            default = name
    if not imported and not skipped:
        raise ConfigError("Nothing to import: the selection was empty.")
    if default:
        if default not in config_editor.profile_names(text):
            raise ConfigError(f"default profile '{default}' was not imported")
        text = settings.set_values(text, {"default_profile": default})
    write_config(path, text)
    with STATE.lock:
        requested = STATE.profile_request
    # As with a one-click add: the dialog's chosen default wins, otherwise
    # the profile the user was running stays selected.
    out = reload_state_config(profile=default or requested or None)
    out["imported"] = imported
    out["skipped"] = skipped
    out["default_profile"] = default or out["active_profile"]
    out["preset"] = pid
    out["base_url"] = base_url
    out["uses_env"] = bool(api_key_env and not api_key)
    return out
