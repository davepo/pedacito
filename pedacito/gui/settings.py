"""Editing the Pedacito config files from the GUI.

Pure functions (no HTTP, no shared state), so they are unit-testable:

    schema()                          every setting: type, default, group, help
    validate_text(text)               parse + check a config file's text
    set_values(text, changes, ...)    change keys in place, keeping comments
    remove_profile / add_profile      whole-profile edits
    resolved(cfg)                     effective value and origin per setting

Edits are textual patches rather than a parse-and-rewrite, so the comments in
the starter template survive. Every edit is validated with the same code the
loader uses before anything is written, and the old file is kept as `.bak`.
"""

from __future__ import annotations

import inspect
import re
import tomllib
from dataclasses import MISSING, fields
from pathlib import Path

from .. import config as config_mod
from ..config import Config, ConfigError

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


# --- validation ---------------------------------------------------------------

def validate_text(text: str, label: str = "config") -> tuple[dict, dict]:
    """Parse and check a config file's text with the loader's own rules.
    Returns (settings, profiles) or raises ConfigError with a readable message."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"invalid TOML: {e}") from e
    flat, profiles = config_mod._flatten(data, Path(label))
    for key, value in flat.items():
        config_mod._coerce(key, value, Path(label))
    for name, body in profiles.items():
        merged, _tier = config_mod._apply_preset(name, body, label)
        for key, value in merged.items():
            config_mod._coerce(key, value, Path(label))
    return flat, profiles


# --- textual TOML patching ----------------------------------------------------

# Joins the parts of a table path internally, so a profile named "a.b" stays
# one part.
SEP = "\x1f"
_HEADER_RE = re.compile(r"^\s*(\[\[?)\s*(.+?)\s*\]\]?\s*(?:#.*)?$")
_ASSIGN_RE = re.compile(r"^(\s*)([A-Za-z0-9_-]+|\"[^\"]*\"|'[^']*')\s*=\s*(.*)$")


def _split_dotted(name: str) -> list[str]:
    """`profiles."my.model".sampler` -> ['profiles', 'my.model', 'sampler']."""
    parts, cur, quote = [], "", ""
    for ch in name:
        if quote:
            if ch == quote:
                quote = ""
            else:
                cur += ch
        elif ch in "\"'":
            quote = ch
        elif ch == ".":
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    parts.append(cur.strip())
    return parts


def _bare(name: str) -> bool:
    return re.fullmatch(r"[A-Za-z0-9_-]+", name) is not None


def table_name(*parts: str) -> str:
    """Header text for a dotted table path, quoting parts that need it."""
    return ".".join(p if _bare(p) else '"' + p.replace('"', "") + '"'
                    for p in parts)


def _triple_toggles(line: str, state: str) -> str:
    """Track whether a line leaves us inside a multi-line string."""
    for q in ('"""', "'''"):
        count = line.count(q)
        if state == q and count % 2 == 1:
            state = ""
        elif not state and count % 2 == 1:
            state = q
    return state


def _scan(lines: list[str]) -> list[dict]:
    """Locate table headers: [{name, line}] plus an implicit top-level entry."""
    heads = [{"name": "", "line": -1}]
    state = ""
    for i, line in enumerate(lines):
        if state:
            state = _triple_toggles(line, state)
            continue
        m = _HEADER_RE.match(line)
        if m and not _ASSIGN_RE.match(line):
            name = SEP.join(_split_dotted(m.group(2)))
            heads.append({"name": name, "line": i, "array": m.group(1) == "[["})
        state = _triple_toggles(line, state)
    return heads


def _regions(lines: list[str]) -> list[dict]:
    heads = _scan(lines)
    for k, h in enumerate(heads):
        h["start"] = h["line"] + 1
        h["end"] = heads[k + 1]["line"] if k + 1 < len(heads) else len(lines)
    return heads


def _value_end(lines: list[str], i: int, col: int) -> tuple[int, int]:
    """Where the value that starts at lines[i][col] ends: (line, column just
    after its last character, before any trailing comment). Understands
    strings, nested []/{} and multi-line strings and arrays."""
    depth = 0
    quote = ""
    j, k = i, col
    while j < len(lines):
        line = lines[j]
        stop = len(line)
        while k < len(line):
            ch = line[k]
            if quote:
                if quote in ('"""', "'''"):
                    if line.startswith(quote, k):
                        quote, k = "", k + 3
                        continue
                elif quote == '"' and ch == "\\":
                    k += 2
                    continue
                elif ch == quote:
                    quote = ""
                k += 1
                continue
            if line.startswith('"""', k) or line.startswith("'''", k):
                quote = line[k:k + 3]
                k += 3
                continue
            if ch in "\"'":
                quote = ch
            elif ch in "[{":
                depth += 1
            elif ch in "]}":
                depth -= 1
            elif ch == "#":
                stop = k
                break
            k += 1
        if depth <= 0 and quote not in ('"""', "'''"):
            return j, len(line[:stop].rstrip())
        j += 1
        k = 0
    return len(lines) - 1, len(lines[-1])


def _find_key(lines: list[str], region: dict, key: str):
    """(line, start_col_of_value, value_end_line, value_end_col) of `key`'s
    uncommented assignment inside the region, or None."""
    state = ""
    i = region["start"]
    while i < region["end"]:
        line = lines[i]
        if state:
            state = _triple_toggles(line, state)
            i += 1
            continue
        m = _ASSIGN_RE.match(line)
        if m:
            name = m.group(2).strip("\"'")
            vstart = m.start(3)
            if name == key:
                j, endc = _value_end(lines, i, vstart)
                return i, vstart, j, endc
            # skip over a multi-line value
            j, _endc = _value_end(lines, i, vstart)
            i = j + 1
            continue
        state = _triple_toggles(line, state)
        i += 1
    return None


def _last_assignment_end(lines: list[str], region: dict) -> int:
    """Line index after which a new key can be inserted: the end of the last
    uncommented assignment, else the header line itself."""
    last = region["line"]
    i = region["start"]
    state = ""
    while i < region["end"]:
        line = lines[i]
        if state:
            state = _triple_toggles(line, state)
            i += 1
            continue
        m = _ASSIGN_RE.match(line)
        if m:
            j, _ = _value_end(lines, i, m.start(3))
            last = j
            i = j + 1
            continue
        state = _triple_toggles(line, state)
        i += 1
    return last


def literal(value) -> str:
    """A Python value as a TOML literal."""
    return config_mod._toml_literal(value)


def _is_profile(name: str) -> bool:
    return name.startswith("profiles" + SEP)


def set_values(text: str, changes: dict | None = None,
               removes: list[str] | tuple[str, ...] = (),
               profile: str | None = None) -> str:
    """Return `text` with `changes` set and `removes` deleted.

    With `profile`, only that profile's table is edited (created if absent).
    Without, the settings are global: a key replaces its existing assignment
    wherever it lives outside the profiles, else lands in `[universal]`
    (created if absent)."""
    changes = changes or {}
    table = ("profiles" + SEP + profile) if profile else ""
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.replace("\r\n", "\n").split("\n")
    trailing_nl = text.endswith("\n")
    if trailing_nl and lines and lines[-1] == "":
        lines.pop()

    def regions():
        return _regions(lines)

    def region_for(name: str):
        for r in regions():
            if r["name"] == name:
                return r
        return None

    def locate(key: str):
        """The region+position currently holding `key` for this edit scope."""
        if table:
            r = region_for(table)
            if r is None:
                return None, None
            return r, _find_key(lines, r, key)
        for r in regions():
            if _is_profile(r["name"]) or r.get("array"):
                continue
            hit = _find_key(lines, r, key)
            if hit:
                return r, hit
        return None, None

    for key in removes:
        _r, hit = locate(key)
        if hit:
            i, _vs, j, _ec = hit
            del lines[i:j + 1]

    for key, value in changes.items():
        lit = literal(value)
        r, hit = locate(key)
        if hit:
            i, vs, j, ec = hit
            tail = lines[j][ec:]
            lines[i:j + 1] = [lines[i][:vs] + lit + tail]
            continue
        new_line = f"{key} = {lit}"
        if table:
            if r is None:
                if lines and lines[-1].strip():
                    lines.append("")
                lines.extend([f"[{table_name('profiles', profile)}]", new_line])
            else:
                lines.insert(_last_assignment_end(lines, r) + 1, new_line)
        else:
            r = region_for("universal")
            if r is None:
                if lines and lines[-1].strip():
                    lines.append("")
                lines.extend(["[universal]", new_line])
            else:
                lines.insert(_last_assignment_end(lines, r) + 1, new_line)

    out = nl.join(lines)
    return out + nl if (trailing_nl or not out) else out


def profile_names(text: str) -> list[str]:
    return [r["name"].split(SEP, 1)[1] for r in _regions(text.split("\n"))
            if r["name"].count(SEP) == 1 and _is_profile(r["name"])]


def remove_profile(text: str, name: str) -> str:
    """Delete `[profiles.NAME]` and any of its sub-tables."""
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.replace("\r\n", "\n").split("\n")
    target = "profiles" + SEP + name
    drop: list[tuple[int, int]] = []
    for r in _regions(lines):
        if r["name"] == target or r["name"].startswith(target + SEP):
            drop.append((r["line"], r["end"]))
    if not drop:
        raise ConfigError(f"no profile named '{name}' in this file")
    for a, b in sorted(drop, reverse=True):
        del lines[a:b]
    while lines and not lines[-1].strip():
        lines.pop()
    return nl.join(lines) + nl


def add_profile(text: str, name: str, values: dict) -> str:
    """Append a new `[profiles.NAME]` table holding `values`."""
    if not re.fullmatch(r"[A-Za-z0-9_.\- ]+", name or ""):
        raise ConfigError("profile names use letters, digits, '-', '_' and '.'")
    if name in profile_names(text):
        raise ConfigError(f"a profile named '{name}' already exists")
    nl = "\r\n" if "\r\n" in text else "\n"
    body = [f"[{table_name('profiles', name)}]"]
    body += [f"{k} = {literal(v)}" for k, v in values.items()]
    base = text.rstrip("\r\n")
    sep = (nl + nl) if base else ""
    return base + sep + nl.join(body) + nl


def profile_body(text: str, name: str) -> dict:
    """The raw settings written under `[profiles.NAME]` in this text."""
    _flat, profiles = validate_text(text)
    return dict(profiles.get(name, {}))


def new_user_config(base_url: str, model: str, vram_gb: int | None) -> str:
    """The starter template with the connection filled in."""
    text = config_mod.TEMPLATE
    values: dict = {"base_url": base_url, "model": model}
    text = set_values(text, values, profile="default")
    if vram_gb:
        text = set_values(text, {"vram_gb": int(vram_gb)})
    return text
