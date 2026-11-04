"""The GUI's settings endpoints: read, edit, and write Pedacito's config files.

Thin glue over `settings.py` (pure text patching) -- this module owns the file
system: which file is meant, backups, atomic writes, secrets, and the live
server probe used by "test connection".
"""

from __future__ import annotations

import os
from pathlib import Path

import requests

from .. import config as config_mod
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
