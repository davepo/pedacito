"""One color-scheme engine for the GUI and the terminal UI.

A theme is exactly ten hex color tokens (`THEME_KEYS`).  The same palette
paints the browser (as CSS custom properties) and the TUI (as 24-bit or
256-color ANSI), so a scheme picked in one place reads the same everywhere.

Built-in presets live in this module.  User-created themes and the globally
selected theme are stored in ``themes.json`` beside the user's config file
(``~/.config/pedacito/themes.json`` by default, honouring
``$XDG_CONFIG_HOME``).  A project may pin its own theme in
``<root>/.pedacito/config.json``::

    {"theme": "nord"}

The helpers are intentionally small and dependency-free: the GUI handlers and
the TUI slash command both call the same functions.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Mapping

from . import config as config_mod

__all__ = [
    "THEME_KEYS", "BUILTIN_THEMES", "DEFAULT_THEME",
    "normalize_hex", "validate_theme",
    "list_themes", "get_active_theme", "get_active_theme_name",
    "set_active_theme", "save_custom_theme", "delete_custom_theme",
    "set_project_theme", "clear_project_theme", "themes_path",
    "hex_to_rgb", "nearest_ansi256", "supports_truecolor",
    "ansi_fg", "ansi_bg", "style_text",
]

THEME_KEYS = (
    "bg",           # primary background
    "surface",      # panel / card background
    "surface_alt",  # elevated / input / hover surface
    "text",         # primary foreground text
    "muted",        # secondary / muted text
    "accent",       # primary brand / action accent
    "border",       # dividers and borders
    "success",      # green / passing status
    "warning",      # amber / caution status
    "danger",       # red / error status
)

DEFAULT_THEME = "pedacito-dark"

# Every preset carries the complete ten-token schema.  Names are stable and
# are what the GUI and `/theme <name>` use.
BUILTIN_THEMES: dict[str, dict[str, str]] = {
    "pedacito-dark": {
        "bg": "#0d1117", "surface": "#161b22", "surface_alt": "#21262d",
        "text": "#e6edf3", "muted": "#8b949e", "accent": "#58a6ff",
        "border": "#30363d", "success": "#3fb950", "warning": "#d29922",
        "danger": "#f85149",
    },
    "catppuccin-mocha": {
        "bg": "#1e1e2e", "surface": "#181825", "surface_alt": "#313244",
        "text": "#cdd6f4", "muted": "#a6adc8", "accent": "#cba6f7",
        "border": "#45475a", "success": "#a6e3a1", "warning": "#fab387",
        "danger": "#f38ba8",
    },
    "tokyo-night": {
        "bg": "#1a1b26", "surface": "#16161e", "surface_alt": "#24283b",
        "text": "#c0caf5", "muted": "#565f89", "accent": "#7aa2f7",
        "border": "#292e42", "success": "#9ece6a", "warning": "#e0af68",
        "danger": "#f7768e",
    },
    "gruvbox-dark": {
        "bg": "#282828", "surface": "#1d2021", "surface_alt": "#3c3836",
        "text": "#ebdbb2", "muted": "#a89984", "accent": "#fe8019",
        "border": "#504945", "success": "#b8bb26", "warning": "#fabd2f",
        "danger": "#fb4934",
    },
    "nord": {
        "bg": "#2e3440", "surface": "#3b4252", "surface_alt": "#434c5e",
        "text": "#eceff4", "muted": "#a3aab8", "accent": "#88c0d0",
        "border": "#4c566a", "success": "#a3be8c", "warning": "#ebcb8b",
        "danger": "#bf616a",
    },
    "solarized-dark": {
        "bg": "#002b36", "surface": "#073642", "surface_alt": "#0d4654",
        "text": "#93a1a1", "muted": "#657b83", "accent": "#2aa198",
        "border": "#586e75", "success": "#859900", "warning": "#b58900",
        "danger": "#dc322f",
    },
}

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
_BAD_NAME_RE = re.compile(r"[\x00-\x1f\x7f]")
_PROJECT_OVERRIDE_KEYS = ("theme", "active_theme", "gui_theme")

# Resolved at import time for the common case.  `themes_path()` follows a
# patched `config.USER_CONFIG` (as the GUI tests do) and accepts an explicit
# `THEMES_FILE` or `$PEDACITO_THEMES_FILE` override.
_DEFAULT_USER_CONFIG = Path(config_mod.USER_CONFIG).expanduser()
_DEFAULT_THEMES_FILE = _DEFAULT_USER_CONFIG.parent / "themes.json"
THEMES_FILE = _DEFAULT_THEMES_FILE


# --------------------------------------------------------------------- paths

def themes_path() -> Path:
    """The user theme store, honouring a test/application override.

    Precedence: ``$PEDACITO_THEMES_FILE``, a monkeypatched module
    ``THEMES_FILE``, then ``themes.json`` beside ``config.USER_CONFIG`` (which
    already follows ``$XDG_CONFIG_HOME`` on Linux/macOS and ``%APPDATA%`` on
    Windows).
    """
    env = os.environ.get("PEDACITO_THEMES_FILE")
    if env:
        return Path(env).expanduser()
    current = Path(THEMES_FILE).expanduser()
    if current != _DEFAULT_THEMES_FILE:
        return current
    config_path = Path(config_mod.USER_CONFIG).expanduser()
    if config_path != _DEFAULT_USER_CONFIG:
        return config_path.parent / "themes.json"
    config_themes = Path(getattr(config_mod, "USER_THEMES",
                                 _DEFAULT_THEMES_FILE)).expanduser()
    if config_themes != _DEFAULT_THEMES_FILE:
        return config_themes
    # No explicit override: let a caller that changed $XDG_CONFIG_HOME at
    # runtime get the matching store instead of the import-time path.
    return Path(config_mod.user_config_dir()).expanduser() / "themes.json"


def _project_config_path(root) -> Path | None:
    if root is None:
        return None
    return Path(str(root)).expanduser() / ".pedacito" / "config.json"


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict) -> None:
    """Atomically replace `path`; a failed write leaves the old file intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# -------------------------------------------------------------- validation

def normalize_hex(value) -> str:
    """Return `value` as a lowercase ``#rrggbb`` string.

    Accepts the leading ``#`` optionally and expands 3-digit shorthand
    (``#abc`` -> ``#aabbcc``).  Anything else raises ``ValueError`` with a
    message naming the offending value.
    """
    if not isinstance(value, str):
        raise ValueError(f"color must be a hex string, got {type(value).__name__}")
    match = _HEX_RE.fullmatch(value.strip())
    if not match:
        raise ValueError(f"invalid hex color {value!r}; expected #rgb or #rrggbb")
    digits = match.group(1).lower()
    if len(digits) == 3:
        digits = "".join(ch * 2 for ch in digits)
    return "#" + digits


def validate_theme(tokens) -> dict[str, str]:
    """Validate one complete palette and return normalized tokens.

    Every key in `THEME_KEYS` is required; unknown keys are rejected so a
    typo cannot silently disappear from a saved theme.
    """
    if not isinstance(tokens, Mapping):
        raise ValueError("a theme must be a mapping of ten color tokens")
    normal = {str(k).strip().lower(): v for k, v in tokens.items()}
    missing = [key for key in THEME_KEYS if key not in normal]
    if missing:
        raise ValueError("theme is missing color token(s): " + ", ".join(missing))
    unknown = sorted(key for key in normal if key not in THEME_KEYS)
    if unknown:
        raise ValueError("unknown theme color token(s): " + ", ".join(unknown))
    return {key: normalize_hex(normal[key]) for key in THEME_KEYS}


# ------------------------------------------------------------------- store

def _store_path() -> Path:
    return themes_path()


def _load_store() -> dict:
    data = _read_json(_store_path())
    raw = data.get("custom_themes", data.get("themes", data.get("custom", {})))
    custom: dict[str, dict[str, str]] = {}
    if isinstance(raw, dict):
        for name, tokens in raw.items():
            try:
                clean = validate_theme(tokens)
            except (ValueError, TypeError):
                continue        # a hand-edited bad entry must not break the app
            key = str(name).strip()
            if key and key.casefold() not in {n.casefold() for n in BUILTIN_THEMES}:
                custom[key] = clean
    active = data.get("active_theme", data.get("theme", DEFAULT_THEME))
    if not isinstance(active, str) or not active.strip():
        active = DEFAULT_THEME
    return {"active_theme": active.strip(), "custom_themes": custom}


def _save_store(store: dict) -> None:
    _write_json(_store_path(), {
        "active_theme": str(store.get("active_theme") or DEFAULT_THEME),
        "custom_themes": dict(store.get("custom_themes") or {}),
    })


def _find_name(name: str, names) -> str | None:
    """Exact match first, then a case-insensitive one."""
    wanted = str(name or "").strip()
    if not wanted:
        return None
    if wanted in names:
        return wanted
    folded = wanted.casefold()
    for candidate in names:
        if candidate.casefold() == folded:
            return candidate
    return None


def _project_theme_name(root) -> str | None:
    path = _project_config_path(root)
    if path is None:
        return None
    data = _read_json(path)
    for key in _PROJECT_OVERRIDE_KEYS:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = data.get("pedacito")
    if isinstance(nested, dict):
        for key in _PROJECT_OVERRIDE_KEYS:
            value = nested.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _merged(store: dict) -> dict[str, dict[str, str]]:
    themes = {name: dict(tokens) for name, tokens in BUILTIN_THEMES.items()}
    builtin_folded = {name.casefold() for name in BUILTIN_THEMES}
    for name, tokens in (store.get("custom_themes") or {}).items():
        if name.casefold() not in builtin_folded:
            themes[name] = dict(tokens)
    return themes


def _active_name(store: dict, root=None, themes=None) -> str:
    themes = themes or _merged(store)
    project = _project_theme_name(root)
    if project:
        match = _find_name(project, themes)
        if match:
            return match
    match = _find_name(str(store.get("active_theme") or DEFAULT_THEME), themes)
    return match or DEFAULT_THEME


def list_themes(root=None) -> dict:
    """All built-in and custom palettes plus the resolved active theme.

    Returns ``{"themes": {name: tokens}, "active_theme": name,
    "builtin_names": [...], "custom_names": [...]}``.  `root` only affects
    the active theme: a valid `.pedacito/config.json` override wins there.
    """
    store = _load_store()
    themes = _merged(store)
    custom = [name for name in store.get("custom_themes", {})
              if name in themes]
    return {
        "themes": themes,
        "active_theme": _active_name(store, root, themes),
        "builtin_names": list(BUILTIN_THEMES),
        "custom_names": custom,
        "project_theme": _project_theme_name(root),
    }


def get_active_theme_name(root=None) -> str:
    """The active theme's name, project override first."""
    return list_themes(root)["active_theme"]


def get_active_theme(root=None) -> dict[str, str]:
    """The active palette as a fresh ten-token dict."""
    data = list_themes(root)
    return dict(data["themes"][data["active_theme"]])


def _validate_theme_name(name: str, themes) -> str:
    cleaned = str(name or "").strip()
    if not cleaned:
        raise ValueError("theme name is required")
    if len(cleaned) > 64:
        raise ValueError("theme name must be 64 characters or fewer")
    if _BAD_NAME_RE.search(cleaned):
        raise ValueError("theme name cannot contain control characters")
    match = _find_name(cleaned, themes)
    if match and match in BUILTIN_THEMES:
        raise ValueError(f"{match!r} is a built-in theme and cannot be overwritten")
    return match or cleaned


def set_active_theme(name: str, root=None) -> dict[str, str]:
    """Select `name` globally, or update a project's own override.

    When `root` has `.pedacito/config.json` with a theme key, only that
    project override changes.  Otherwise the global store is updated.  The
    selected palette is returned.  Unknown names raise ``ValueError``.
    """
    store = _load_store()
    themes = _merged(store)
    match = _find_name(name, themes)
    if not match:
        raise ValueError(f"unknown theme {name!r}. Available: "
                         + ", ".join(themes))
    if _project_theme_name(root):
        set_project_theme(match, root)
    else:
        store["active_theme"] = match
        _save_store(store)
    return dict(themes[match])


def save_custom_theme(name: str, tokens, root=None) -> dict[str, str]:
    """Validate and persist a custom palette; returns its normalized tokens.

    Built-in names are reserved.  `root` is accepted for symmetry with the
    project-aware helpers; custom themes are always global so they can be
    reused across projects.
    """
    del root
    store = _load_store()
    themes = _merged(store)
    clean_name = _validate_theme_name(name, themes)
    clean = validate_theme(tokens)
    store.setdefault("custom_themes", {})[clean_name] = clean
    _save_store(store)
    return dict(clean)


def delete_custom_theme(name: str) -> bool:
    """Delete a custom theme.  Built-in presets and unknown names return False."""
    wanted = str(name or "").strip()
    if not wanted:
        return False
    if _find_name(wanted, BUILTIN_THEMES):
        return False
    store = _load_store()
    custom = store.get("custom_themes") or {}
    match = _find_name(wanted, custom)
    if not match:
        return False
    del custom[match]
    if str(store.get("active_theme", "")).casefold() == match.casefold():
        store["active_theme"] = DEFAULT_THEME
    _save_store(store)
    return True


def set_project_theme(name: str, root) -> dict[str, str]:
    """Pin a theme to one project in `<root>/.pedacito/config.json`."""
    if root is None:
        raise ValueError("a project root is required")
    path = _project_config_path(root)
    if path is None:
        raise ValueError("a project root is required")
    store = _load_store()
    themes = _merged(store)
    match = _find_name(name, themes)
    if not match:
        raise ValueError(f"unknown theme {name!r}. Available: "
                         + ", ".join(themes))
    data = _read_json(path)
    key = next((k for k in _PROJECT_OVERRIDE_KEYS
                if isinstance(data.get(k), str) and data.get(k)), "theme")
    data[key] = match
    _write_json(path, data)
    return dict(themes[match])


def clear_project_theme(root) -> bool:
    """Remove a project's theme override.  True when a key was removed."""
    path = _project_config_path(root)
    if path is None or not path.is_file():
        return False
    data = _read_json(path)
    changed = False
    for key in _PROJECT_OVERRIDE_KEYS:
        if key in data:
            data.pop(key, None)
            changed = True
    nested = data.get("pedacito")
    if isinstance(nested, dict):
        for key in _PROJECT_OVERRIDE_KEYS:
            if key in nested:
                nested.pop(key, None)
                changed = True
    if changed:
        _write_json(path, data)
    return changed


# -------------------------------------------------------------------- ANSI

def hex_to_rgb(hex_str) -> tuple[int, int, int]:
    """``#abc`` / ``#aabbcc`` -> ``(r, g, b)``."""
    value = normalize_hex(hex_str)[1:]
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def nearest_ansi256(hex_str) -> int:
    """The closest xterm-256 palette index (6x6x6 cube or gray ramp)."""
    r, g, b = hex_to_rgb(hex_str)
    levels = (0, 95, 135, 175, 215, 255)
    best_index, best_distance = 16, 1 << 30
    for ri, rv in enumerate(levels):
        for gi, gv in enumerate(levels):
            for bi, bv in enumerate(levels):
                distance = (r - rv) ** 2 + (g - gv) ** 2 + (b - bv) ** 2
                if distance < best_distance:
                    best_distance = distance
                    best_index = 16 + 36 * ri + 6 * gi + bi
    for i in range(24):
        value = 8 + i * 10
        distance = (r - value) ** 2 + (g - value) ** 2 + (b - value) ** 2
        if distance < best_distance:
            best_distance = distance
            best_index = 232 + i
    return best_index


def supports_truecolor(environ=None) -> bool:
    """True when the terminal advertises 24-bit color via COLORTERM/TERM."""
    env = os.environ if environ is None else environ
    colorterm = str(env.get("COLORTERM") or "").lower()
    term = str(env.get("TERM") or "").lower()
    return (colorterm in ("truecolor", "24bit")
            or "truecolor" in term or "24bit" in term
            or "direct" in term)


def ansi_fg(hex_str, truecolor=True) -> str:
    """A foreground SGR sequence.

    `truecolor=True` emits ``\\x1b[38;2;R;G;Bm``; ``False`` emits the nearest
    ``\\x1b[38;5;Nm``.  Pass ``None`` to detect support from the environment.
    """
    if truecolor is None:
        truecolor = supports_truecolor()
    if truecolor:
        r, g, b = hex_to_rgb(hex_str)
        return f"\x1b[38;2;{r};{g};{b}m"
    return f"\x1b[38;5;{nearest_ansi256(hex_str)}m"


def ansi_bg(hex_str, truecolor=True) -> str:
    """A background SGR sequence; same truecolor/256-color rule as ansi_fg."""
    if truecolor is None:
        truecolor = supports_truecolor()
    if truecolor:
        r, g, b = hex_to_rgb(hex_str)
        return f"\x1b[48;2;{r};{g};{b}m"
    return f"\x1b[48;5;{nearest_ansi256(hex_str)}m"


def style_text(text, fg=None, bg=None, bold=False, truecolor=True) -> str:
    """Wrap `text` in the requested SGR styles.

    `NO_COLOR` (set to a non-empty value) disables styling entirely, as does
    an empty style request.  The caller decides when the terminal supports
    color; this function never adds escapes for an unstyled call.
    """
    if os.environ.get("NO_COLOR"):
        return str(text)
    codes: list[str] = []
    if bold:
        codes.append("\x1b[1m")
    if fg:
        codes.append(ansi_fg(fg, truecolor=truecolor))
    if bg:
        codes.append(ansi_bg(bg, truecolor=truecolor))
    if not codes:
        return str(text)
    return "".join(codes) + str(text) + "\x1b[0m"
