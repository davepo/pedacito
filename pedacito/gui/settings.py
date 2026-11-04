"""Editing the Pedacito config files from the GUI.

Pure functions (no HTTP, no shared state), so they are unit-testable:

    schema()                          every setting: type, default, group, help
    resolved(cfg)                     effective value and origin per setting

The comment-preserving editor itself lives in `config_editor.py`, so the CLI can
edit configs without importing the GUI package; those names are re-exported
below (`validate_text`, `set_values`, `add_profile`, ...) for the settings page.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import MISSING, fields

from .. import config as config_mod
from ..config import Config

# The comment-preserving editor, moved out of gui/ so the CLI can use it.
from ..config_editor import (  # noqa: F401
    add_profile, literal, new_user_config, profile_body, profile_names,
    remove_profile, set_values, table_name, upsert_profiles, validate_text,
)

SECRET_KEYS = ("api_key",)
# Not editable one-by-one here: tables (use the raw file), selectors handled by
# the profile UI, and bookkeeping.
HIDDEN_KEYS = {"active_tier", "default_profile"}

_GROUP_RE = re.compile(r"^\s{4}#\s*---+\s*(.*?)\s*-+\s*$")
_FIELD_RE = re.compile(r"^\s{4}([A-Za-z_]\w*)\s*:\s*[^=]+?(?:=.*)?$")
_COMMENT_RE = re.compile(r"^\s{4}#\s?(.*)$")


def _type_name(annotation) -> str:
    text = str(annotation)
    if "tuple" in text or "list" in text:
        return "list"
    if "dict" in text:
        return "table"
    if "bool" in text:
        return "bool"
    if "int" in text:
        return "int"
    if "float" in text:
        return "float"
    return "str"


def _choices(name: str) -> list[str] | None:
    if name in config_mod._ENUM_FIELDS:
        return list(config_mod._ENUM_FIELDS[name])
    if name == "json_mode":
        return ["schema", "object"]
    if name == "tool_mode":
        return ["grammar", "native"]
    if name == "edit_style":
        return ["quote", "anchor"]
    if name == "tier":
        return ["", *config_mod.PRESET_TIERS]
    if name in ("k_cache_quant", "v_cache_quant"):
        return ["", *config_mod._KV_QUANT_BITS]
    return None


_SCHEMA_CACHE: list[dict] | None = None


def schema() -> list[dict]:
    """Every editable setting, in source order, with the help text taken from
    the comment block above its declaration in `Config`."""
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None:
        return _SCHEMA_CACHE
    help_by: dict[str, str] = {}
    group_by: dict[str, str] = {}
    group = "General"
    comment: list[str] = []
    for line in inspect.getsource(Config).splitlines():
        if not line.strip():
            comment = []
            continue
        g = _GROUP_RE.match(line)
        if g:
            group = re.sub(r"\s*\(.*?\)\s*$", "", g.group(1)).strip() or group
            comment = []
            continue
        c = _COMMENT_RE.match(line)
        if c:
            comment.append(c.group(1).rstrip())
            continue
        f = _FIELD_RE.match(line)
        if f:
            name = f.group(1)
            group_by[name] = group
            if comment:
                help_by[name] = " ".join(t for t in comment if t).strip()
            comment = []
            continue
        comment = []
    base = Config()
    out: list[dict] = []
    for f in fields(Config):
        name = f.name
        if name not in config_mod._FIELDS or name in HIDDEN_KEYS \
                or name.startswith("_"):
            continue
        if f.default is not MISSING:
            default = f.default
        elif f.default_factory is not MISSING:
            default = f.default_factory()
        else:
            default = getattr(base, name, "")
        if isinstance(default, tuple):
            default = list(default)
        out.append({
            "name": name,
            "type": _type_name(f.type),
            "default": default,
            "group": group_by.get(name, "General"),
            "help": help_by.get(name, ""),
            "choices": _choices(name),
            "secret": name in SECRET_KEYS,
        })
    _SCHEMA_CACHE = out
    return out


def resolved(cfg) -> dict:
    """{setting: {"value", "source"}} for the open project's effective config
    (secrets are never included)."""
    out = {}
    for item in schema():
        name = item["name"]
        value = getattr(cfg, name, item["default"])
        if isinstance(value, tuple):
            value = list(value)
        if item["secret"]:
            value = ""
        out[name] = {"value": value,
                     "source": (cfg.sources or {}).get(name, "default")}
    return out
