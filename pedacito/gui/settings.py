"""Editing the Pedacito config files from the GUI.

Pure functions (no HTTP, no shared state), so they are unit-testable:

    schema()                          every setting: type, default, group, help
    resolved(cfg)                     effective value and origin per setting

`FIELD_META` / `FieldMeta` carry curated label + description entries for
settings that deserve more than the comment scraped from their `Config` line.

The comment-preserving editor itself lives in `config_editor.py`, so the CLI can
edit configs without importing the GUI package; those names are re-exported
below (`validate_text`, `set_values`, `add_profile`, ...) for the settings page.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import MISSING, dataclass, fields

from .. import config as config_mod
from ..config import Config

# The comment-preserving editor, moved out of gui/ so the CLI can use it.
from ..config_editor import (  # noqa: F401
    add_profile, literal, new_user_config, profile_body, profile_names,
    reference_setting_help, remove_profile, rename_profile, set_values,
    table_name, upsert_profiles, validate_text,
)

SECRET_KEYS = ("api_key",)
# Not editable one-by-one here: bookkeeping. `default_profile` is editable (it
# is a real config key) even though the profile UI also offers Make default.
HIDDEN_KEYS = {"active_tier"}


@dataclass(frozen=True)
class FieldMeta:
    """GUI presentation for a setting shown in ⚙ Settings -> All settings and
    the Profiles override editor: a human `label` and a `description` that
    wins over the help text scraped from the Config source."""
    label: str
    description: str


# Settings with curated GUI metadata. Keys absent here keep their name as the
# label and the Config comment block as the description.
FIELD_META: dict[str, FieldMeta] = {
    "enable_web": FieldMeta(
        label="Enable Web Fetch & Search",
        description="Allow the agent to fetch URLs or search documentation "
                    "online. When disabled, web tool attempts trigger an "
                    "interactive permission prompt."),
    "enable_run_test_in_gather": FieldMeta(
        label="Scoped Test Runs in Gather",
        description="Let the agent run one scoped pytest check (a test file "
                    "or node id, never a shell command) during investigation "
                    "to see live tracebacks before editing. When disabled, "
                    "every attempt asks the permission broker."),
    "frontend_contracts": FieldMeta(
        label="Frontend DOM Contract Gate",
        description="Before tests run, deterministically check that local "
                    "<script src>/<link href> targets exist, that literal "
                    "getElementById/querySelector('#id') selectors match the "
                    "ids defined in HTML, and that changed JS files parse. "
                    "Violations feed the repair loop."),
    "static_gate": FieldMeta(
        label="Static Analysis Gate",
        description="Check changed Python files for undefined names, bad "
                    "`__all__` entries and import placement before any "
                    "test/lint subprocess. Only errors the current task "
                    "introduced trigger the cheap repair loop."),
    "auto_heal_imports": FieldMeta(
        label="Auto-Heal Unambiguous Imports",
        description="Before an undefined-name finding reaches the model, "
                    "insert the one import that provably fixes it (a stdlib "
                    "module, or a symbol exactly one local module exports). "
                    "Ambiguous names are never guessed."),
    "auto_green_ratchet": FieldMeta(
        label="Green Ratchet (Undo Failed Auto Steps)",
        description="Snapshot the project before every Auto step and undo "
                    "exactly that step when it still fails its check after "
                    "the repair rounds. Later steps then build only on the "
                    "last verified-green workspace."),
    "enforce_json_schema": FieldMeta(
        label="Engine-Enforced JSON Grammar",
        description="Ask the inference server to constrain every gather "
                    "step to the action grammar (response_format json_schema, "
                    "strict). Servers that refuse the constraint fall back "
                    "per model: json_object mode, then prompt-only. Keep on "
                    "unless a server misbehaves on constrained sampling."),
    "scout_profile": FieldMeta(
        label="Scout Profile (fast gather model)",
        description="Dual-model routing: the named profile answers gather's "
                    "per-tool lookup steps while the active (architect) "
                    "profile keeps planning, answers, and edits. Empty means "
                    "no scout; pick one like 'scout' in the dropdown. The "
                    "profile must already exist with its own base_url/model."),
    "enable_prompt_cache": FieldMeta(
        label="Enable Local Server KV Prompt Caching (cache_prompt)",
        description="On self-hosted llama.cpp-style servers, add the "
                    "cache_prompt request key so the KV prefix of the "
                    "replayed gather context stays resident between steps. "
                    "Only local endpoints receive it; a server that answers "
                    "400/422 with the key is downgraded automatically and "
                    "remembered per model."),
    "enable_prompt_lookup": FieldMeta(
        label="Enable N-Gram Prompt Lookup Speculation",
        description="On self-hosted llama.cpp-style servers, add "
                    "prompt_lookup_num_tokens to switch on n-gram "
                    "speculative decoding (draft tokens are matched from the "
                    "prompt itself, no extra model). Off by default; a server "
                    "that refuses the key is downgraded and remembered."),
    "prompt_lookup_num_tokens": FieldMeta(
        label="Prompt Lookup Draft Tokens",
        description="How many n-gram tokens prompt_lookup_num_tokens asks the "
                    "server to speculate per step. Higher drafts more per "
                    "verification pass; typical values are 4 to 16."),
    "enable_mcp": FieldMeta(
        label="Enable Named MCP Servers (mcp_call)",
        description="Expose the pruned, permission-gated `mcp_call` action "
                    "for servers configured as [mcp_servers.<name>] tables "
                    "(command/args for stdio or url for HTTP). Only the "
                    "tools matching the current task are shown, and a server "
                    "not marked trusted asks the permission broker first. "
                    "Independent of the legacy allow_mcp catalog path."),
}

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
    ref_help = reference_setting_help()
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
        meta = FIELD_META.get(name)
        out.append({
            "name": name,
            "label": meta.label if meta else name,
            "type": _type_name(f.type),
            "default": default,
            "group": group_by.get(name, "General"),
            "help": meta.description if meta else
                    (ref_help.get(name) or help_by.get(name, "")),
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
