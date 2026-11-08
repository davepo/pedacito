"""The GUI's settings endpoints: read, edit, and write Pedacito's config files.

Thin glue over `settings.py` (pure text patching) -- this module owns the file
system: which file is meant, backups, atomic writes, secrets, and the live
server probe used by "test connection".
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import requests

from .. import config as config_mod
from .. import config_editor
from ..config import USER_CONFIG, ConfigError
from . import settings
from .core import STATE

SECRET_MASK_KEYS = settings.SECRET_KEYS


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
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            Path(str(path) + ".bak").write_bytes(path.read_bytes())
        except OSError:
            pass
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="")
    os.replace(tmp, path)


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
            "exists": path.exists(), "text": text, "profiles": {},
            "settings": {}, "error": ""}
    if text.strip():
        try:
            flat, profiles = settings.validate_text(text, str(path))
            info["settings"] = {k: (list(v) if isinstance(v, tuple) else v)
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


def payload() -> dict:
    cfg = STATE.cfg
    return {
        "schema": settings.schema(),
        "resolved": settings.resolved(cfg) if cfg is not None else {},
        "active_profile": cfg.active_profile if cfg is not None else "",
        "tiers": list(config_mod.PRESET_TIERS),
        "user": file_info("user"),
        "project": file_info("project"),
        "user_config_path": str(USER_CONFIG),
        "project_open": STATE.root is not None,
    }


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
    return payload()


def save_raw(which: str, text: str) -> dict:
    write_config(config_path(which), text)
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


def _stored_api_key(which: str, profile: str) -> str:
    try:
        text = _read_text(config_path(which))
        body = settings.profile_body(text, profile) if profile else {}
        if body.get("api_key"):
            return str(body["api_key"])
        flat, _ = settings.validate_text(text)
        return str(flat.get("api_key", ""))
    except (ConfigError, OSError):
        return ""


def probe_server(base_url: str, api_key: str, which: str = "user",
                 profile: str = "") -> dict:
    """List the models a server reports (the "test connection" button)."""
    base_url = (base_url or "").strip()
    if not base_url:
        return {"ok": False, "error": "Enter the server URL first."}
    if not api_key:
        api_key = _stored_api_key(which, profile) or (
            STATE.cfg.api_key if STATE.cfg is not None else "")
    try:
        r = requests.get(base_url.rstrip("/") + "/models",
                         headers={"Authorization": f"Bearer {api_key}"},
                         timeout=8)
        r.raise_for_status()
        ids = sorted({m.get("id", "?") for m in r.json().get("data", [])})
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    return {"ok": True, "models": ids}


def list_profiles() -> dict:
    """The profiles table (identity + measured scores + recommendations) for
    the settings page, from the same renderer the CLI uses."""
    from .. import profiles as profiles_mod
    cfg = STATE.cfg if STATE.cfg is not None else config_mod.Config()
    rows = profiles_mod.list_profiles(cfg)
    return {"profiles": rows, "recommendations": profiles_mod.recommend(rows)}


def scan_servers() -> dict:
    """Discover servers and list each one's models with metadata (loaded,
    size, context, embedding/too-big already dropped)."""
    from .. import discovery
    servers = []
    for label, base_url, _ids in discovery.discover_servers(timeout=1.5):
        servers.append({"label": label, "base_url": base_url,
                        "models": discovery.describe_models(base_url)})
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
        ("temperature_answer", "greedy answers, so SEARCH text is copied exactly"),
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
    if path.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        bp = Path(f"{path}.reset.{stamp}.bak")
        bp.write_bytes(path.read_bytes())
        backup = str(bp)
    text = baseline_text()
    settings.validate_text(text, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="")
    os.replace(tmp, path)
    out = reload_state_config()
    out["backup"] = backup
    return out


# --- immediate STATE config reload --------------------------------------------

def reload_state_config() -> dict:
    """Re-read the config files into STATE so /api/state and /api/settings
    show the change at once (the profile dropdown included), without waiting
    for a project re-open job. Returns the fresh settings payload."""
    from ..agent import Agent
    from ..llm import LLMClient
    with STATE.lock:
        root = STATE.root
    try:
        cfg = config_mod.load(root=root, require_hardware=False,
                              warn_budgets=False)
    except ConfigError:
        cfg = None
    with STATE.lock:
        STATE.cfg = cfg
        STATE.client = LLMClient(cfg) if cfg is not None else None
        STATE.agent = (Agent(STATE.idx, STATE.client, cfg)
                       if cfg is not None and STATE.idx is not None else None)
        STATE.profile_request = ""
        STATE.model_checked, STATE.model_status, STATE.model_ok = False, "", None
    return payload()


# --- selective model discovery and import ---------------------------------------

def _norm_url(url) -> str:
    return str(url or "").rstrip("/")


def scan_models(urls=(), vram_gb=0) -> dict:
    """`pedacito models scan`'s discovery pass WITHOUT the writes: probe the
    well-known local ports, every configured profile's server, and `urls`;
    return one selectable row per model. Nothing is written to the config."""
    from .. import cli as cli_mod
    from .. import discovery
    explicit = [str(u).strip() for u in (urls or ()) if str(u).strip()]
    try:
        profile_urls = cli_mod._profile_base_urls()
    except Exception:
        profile_urls = []
    try:
        vram = int(vram_gb or 0)
    except (TypeError, ValueError):
        vram = 0
    found = discovery.discover_servers(timeout=2.0, urls=explicit,
                                       profile_urls=profile_urls)
    try:
        flat, existing = settings.validate_text(_read_text(config_path("user")))
    except ConfigError:
        flat, existing = {}, {}
    imported = {(_norm_url(b.get("base_url", flat.get("base_url", ""))),
                 str(b.get("model") or "")) for b in existing.values()}
    taken = set(existing)
    servers_out, models_out = [], []
    for label, base_url, _ids in found:
        try:
            recs = discovery.describe_models(base_url, vram_gb=vram)
        except Exception:
            recs = []
        rows = []
        for rec in recs:
            mid = str(rec.get("id") or "")
            if not mid:
                continue
            already = (_norm_url(base_url), mid) in imported
            name = config_editor.short_name(mid, taken, label)
            taken.add(name)
            row = {"server_url": base_url, "provider": label, "model_id": mid,
                   "base_url": base_url, "loaded": rec.get("loaded"),
                   "context_length": int(rec.get("context_length") or 0),
                   "size_bytes": int(rec.get("size_bytes") or 0),
                   "quant": str(rec.get("quant") or ""),
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
    for sel in models:
        if not isinstance(sel, dict):
            continue
        name = str(sel.get("profile_name") or sel.get("name") or "").strip()
        mid = str(sel.get("model") or sel.get("model_id") or "").strip()
        base = str(sel.get("base_url") or sel.get("server_url") or "").strip()
        if not name or not mid or not base:
            raise ConfigError("Each imported model needs a profile name, a "
                              "model id, and a server URL.")
        if name in config_editor.profile_names(text):
            skipped.append(name)
            if sel.get("set_default"):
                default = default or name
            continue
        values: dict = {"base_url": base, "model": mid}
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
    if not imported and not skipped:
        raise ConfigError("Nothing to import: the selection was empty.")
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
