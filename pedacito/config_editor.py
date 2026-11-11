"""Comment-preserving edits to the Pedacito config file.

Pure functions (no HTTP, no shared state) so they are unit-testable and the
CLI can use them without importing the GUI package:

    validate_text(text)               parse + check a config file's text
    set_values(text, changes, ...)    change keys in place, keeping comments
    remove_profile / add_profile      whole-profile edits
    rename_profile(text, old, new)    rename a table and its references
    short_name(model_id, ...)         derive a profile name from a model id
    upsert_profiles(text, found)      append identity-only profiles idempotently

Edits are textual patches rather than a parse-and-rewrite, so the comments in
the starter template survive. Every edit is validated with the same code the
loader uses before anything is written.
"""

from __future__ import annotations

import re
import tomllib
from datetime import date
from pathlib import Path

from . import config as config_mod
from .config import ConfigError


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
    """Track whether a line leaves us inside a multi-line string.

    A character scan, not a count: a '#' comment or a single-line string can
    quote the triple delimiter without opening anything, and counting those
    occurrences swallowed every later table header as 'inside a string'."""
    i, n = 0, len(line)
    while i < n:
        if state:
            if state == '"""' and line[i] == "\\":
                i += 2
                continue
            if line.startswith(state, i):
                state = ""
                i += 3
                continue
            i += 1
            continue
        ch = line[i]
        if ch == "#":
            break
        if line.startswith(('"""', "'''"), i):
            q = line[i:i + 3]
            end = line.find(q, i + 3)
            if end == -1:
                return q
            i = end + 3
            continue
        if ch in "\"'":
            i += 1
            while i < n:
                if ch == '"' and line[i] == "\\":
                    i += 2
                    continue
                if line[i] == ch:
                    i += 1
                    break
                i += 1
            continue
        i += 1
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


def _retarget_roles(lines: list[str], old_name: str, new_name: str) -> None:
    """Rewrite the quoted profile name `old_name` to `new_name` wherever the
    roles table stores it: an inline/multi-line `roles = {...}` assignment
    outside the profiles, or a `[roles]` section. Trailing comments survive."""
    swaps = (('"' + old_name + '"', '"' + new_name + '"'),
             ("'" + old_name + "'", "'" + new_name + "'"))

    def swap(seg: str) -> str:
        for a, b in swaps:
            seg = seg.replace(a, b)
        return seg

    for r in _regions(lines):
        if _is_profile(r["name"]) or r.get("array"):
            continue
        if r["name"] == "roles":
            i = r["start"]
            while i < r["end"]:
                m = _ASSIGN_RE.match(lines[i])
                if not m:
                    i += 1
                    continue
                j, ec = _value_end(lines, i, m.start(3))
                if j == i:
                    lines[i] = (lines[i][:m.start(3)]
                                + swap(lines[i][m.start(3):ec]) + lines[i][ec:])
                else:
                    lines[i] = lines[i][:m.start(3)] + swap(lines[i][m.start(3):])
                    for k in range(i + 1, j):
                        lines[k] = swap(lines[k])
                    lines[j] = swap(lines[j][:ec]) + lines[j][ec:]
                i = j + 1
            continue
        hit = _find_key(lines, r, "roles")
        if not hit:
            continue
        a, vs, b, ec = hit
        if a == b:
            lines[a] = lines[a][:vs] + swap(lines[a][vs:ec]) + lines[a][ec:]
        else:   # a multi-line inline table: swap inside each value line
            lines[a] = lines[a][:vs] + swap(lines[a][vs:])
            for k in range(a + 1, b):
                lines[k] = swap(lines[k])
            lines[b] = swap(lines[b][:ec]) + lines[b][ec:]


def rename_profile(text: str, old_name: str, new_name: str) -> str:
    """Rename `[profiles.OLD]` (and any of its sub-tables) to
    `[profiles.NEW]`, keeping every key and comment outside the changed lines.

    References are retargeted automatically: a top-level `default_profile`
    equal to the old name and any `roles` value that named it. Raises
    ConfigError when the old profile does not exist or the new name does."""
    names = profile_names(text)
    if old_name not in names:
        raise ConfigError(f"no profile named '{old_name}' in this file")
    if new_name == old_name:
        return text
    if new_name in names:
        raise ConfigError(f"a profile named '{new_name}' already exists")
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.replace("\r\n", "\n").split("\n")
    target = "profiles" + SEP + old_name
    for r in _regions(lines):
        if r["name"] != target and not r["name"].startswith(target + SEP):
            continue
        parts = r["name"].split(SEP)
        parts[1] = new_name
        line = lines[r["line"]]
        m = _HEADER_RE.match(line)
        s, e = m.span(2)
        lines[r["line"]] = line[:s] + table_name(*parts) + line[e:]
    _retarget_roles(lines, old_name, new_name)
    out = nl.join(lines)
    # `default_profile` lives outside the profiles: replace it in place.
    flat, _profiles = validate_text(out, "renamed config")
    if flat.get("default_profile") == old_name:
        out = set_values(out, {"default_profile": new_name})
    validate_text(out, "renamed config")
    return out


def new_user_config(base_url: str, model: str, vram_gb: int | None) -> str:
    """The starter template with the connection filled in."""
    text = config_mod.TEMPLATE
    values: dict = {"base_url": base_url, "model": model}
    text = set_values(text, values, profile="default")
    if vram_gb:
        text = set_values(text, {"vram_gb": int(vram_gb)})
    return text


# --- profile naming and the scan writer ----------------------------------------

def short_name(model_id: str, taken=None, server_label: str = "") -> str:
    """Derive a profile name from a model id, disambiguating collisions.

    Rules (from the plan):
      * drop any `org/` prefix and a `.gguf` suffix;
      * turn Ollama `name:tag` into `name-tag`, dropping a `latest` tag;
      * lower-case;
      * replace anything outside `[a-z0-9.-]` with `-`.
    On a collision, append the quant (`qwen3-14b-q4km`), then the server label
    (`qwen3-14b-ollama`), then a numeric suffix. `taken` is the set of names
    already in use; `server_label` is the discovery fingerprint label.
    """
    base, quant = _short_parts(model_id)
    taken = set(taken or ())
    if base not in taken:
        return base
    if quant and f"{base}-{quant}" not in taken:
        return f"{base}-{quant}"
    if server_label:
        label = re.sub(r"[^a-z0-9.-]+", "-", str(server_label).lower()).strip("-")
        if label and f"{base}-{label}" not in taken:
            return f"{base}-{label}"
    i = 2
    while f"{base}-{i}" in taken:
        i += 1
    return f"{base}-{i}"


def _short_parts(model_id: str) -> tuple[str, str]:
    """(base, quant) from a model id: the id with its org/ prefix, `.gguf`
    suffix and `@quant` removed, and the normalised quant (`q4_k_m` -> `q4km`)."""
    raw = str(model_id or "").strip().split("/")[-1]
    quant = ""
    if "@" in raw:
        raw, _, quant = raw.partition("@")
        quant = re.sub(r"[^a-z0-9]+", "", quant.lower())
    if raw.endswith(".gguf"):
        raw = raw[: -len(".gguf")]
    if ":" in raw:
        base, _, tag = raw.partition(":")
        raw = base if tag in ("", "latest") else f"{base}-{tag}"
    raw = re.sub(r"[^a-z0-9.-]+", "-", raw.lower()).strip("-")
    return raw, quant


def _norm_url(url: str) -> str:
    return (url or "").rstrip("/")


def _provenance(entry: dict, when: str) -> str:
    """The leading `# added by pedacito models scan, ...` comment."""
    parts = ["added by pedacito models scan", when]
    if entry.get("server_label"):
        parts.append(str(entry["server_label"]))
    meta: list[str] = []
    if entry.get("quant"):
        meta.append(str(entry["quant"]))
    size_gb = entry.get("size_gb")
    if size_gb:
        meta.append(f"{size_gb} GB")
    if meta:
        parts.append(" ".join(meta))
    return "# " + ", ".join(parts)


def _profile_block(name: str, values: dict, comment: str) -> str:
    lines = [comment, f"[{table_name('profiles', name)}]"]
    lines += [f"{k} = {literal(v)}" for k, v in values.items()]
    return "\n".join(lines)


def upsert_profiles(text: str, found, *, now: str | None = None) -> str:
    """Append an identity-only profile for each discovered model that does not
    already have one, matched on `(base_url, model)`.

    `found` is an iterable of dicts, each with `base_url` and `model` (and
    optional `server_label`, `tier`, `vram_gb`, `quant`, `size_gb`). Card
    choices are never written. Idempotent: a re-scan with the same `found`
    returns `text` unchanged, and existing keys and comments survive byte for
    byte, because profiles are only ever appended.
    """
    _flat, profiles = validate_text(text)
    global_base = _flat.get("base_url")
    existing: dict[tuple[str, str], str] = {}
    for name, body in profiles.items():
        model = body.get("model")
        if not model:
            continue
        base = body.get("base_url", global_base) or ""
        existing[(_norm_url(base), str(model))] = name

    taken = set(profiles)
    blocks: list[str] = []
    when = now or date.today().isoformat()
    for entry in found:
        model = str(entry.get("model") or "")
        base = entry.get("base_url") or ""
        if not model or not base:
            continue
        key = (_norm_url(base), model)
        if key in existing:
            continue
        name = short_name(model, taken, entry.get("server_label", ""))
        taken.add(name)
        existing[key] = name
        values: dict = {}
        # Identity only, plus the API-key reference; card choices are never
        # written. Provider compatibility keys (prompt_cache, json_mode,
        # tool_mode) are transport settings a hosted scan adds from the C0
        # provider table, and the user can still override them.
        for k in ("base_url", "model", "tier", "vram_gb", "api_key_env",
                  "prompt_cache", "json_mode", "tool_mode"):
            if entry.get(k):
                values[k] = entry[k]
        blocks.append(_profile_block(name, values, _provenance(entry, when)))

    if not blocks:
        return text

    nl = "\r\n" if "\r\n" in text else "\n"
    base = text.rstrip("\r\n")
    sep = (nl + nl) if base else ""
    return base + sep + nl.join(blocks) + nl
