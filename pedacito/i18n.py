"""Runtime internationalisation: language detection, catalog loading, and
message lookup.

Pedacito ships its interface strings as flat, dot-delimited JSON catalogs under
`pedacito/locales/`. The rest of the code talks to this module through three
entry points:

    lang = i18n.detect_lang(cli_lang, cfg)      # which language to show
    i18n.set_current_lang(lang)                 # remember it for the session
    text = i18n.t("errors.not_found")           # look the string up

The current language is a process-global default (the GUI and TUI set it once
per session), but every helper accepts an explicit `lang` so a call site can
stay deterministic. Only two languages exist today, both shipped in the
package, so the whole module needs no external dependency and no data files
beyond the two JSON catalogs.
"""

from __future__ import annotations

import json
import os
import threading
from importlib import resources

SUPPORTED_LANGUAGES = ("en", "es")
DEFAULT_LANGUAGE = "en"

# Per-session language, read by `t()` when no explicit language is given.
_current_lang: str = DEFAULT_LANGUAGE
_current_lock = threading.Lock()

# Loaded catalogs, keyed by the canonical language code ("en"/"es").
_catalogs: dict[str, dict[str, str]] = {}
_catalog_lock = threading.Lock()


def normalize_lang(raw: str | None) -> str:
    """The canonical code for a language `raw` names, else ``"en"``.

    Accepts the two codes with any common separator or region subtag
    (``es-419``, ``es-MX``, ``es_ES``, ``en-US``) and their full names
    (``español``, ``spanish``, ``english``). Anything unrecognised falls back
    to the default, so a bogus value can never crash a run."""
    if raw is None:
        return DEFAULT_LANGUAGE
    text = str(raw).strip().lower().replace("_", "-")
    if not text:
        return DEFAULT_LANGUAGE
    base = text.split("-", 1)[0]
    if base in ("es", "español", "spanish"):
        return "es"
    if base in ("en", "english"):
        return "en"
    return DEFAULT_LANGUAGE


def is_valid_lang(raw: str | None) -> bool:
    """True when `raw` actually names a supported language (``en``/``es``)."""
    if raw is None:
        return False
    text = str(raw).strip().lower().replace("_", "-")
    if not text:
        return False
    return text.split("-", 1)[0] in ("en", "english", "es", "español", "spanish")


def load_catalog(lang: str) -> dict[str, str]:
    """The flat catalog for `lang` (normalised), cached for the process.

    Reads the JSON shipped inside the package with ``importlib.resources``, so
    an installed wheel and a plain checkout both work. A missing or unreadable
    catalog yields an empty dict rather than raising."""
    code = normalize_lang(lang)
    with _catalog_lock:
        catalog = _catalogs.get(code)
        if catalog is not None:
            return catalog
    path = resources.files("pedacito").joinpath("locales", f"{code}.json")
    data: dict = {}
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, AttributeError):
        raw = ""
    if raw:
        try:
            loaded = json.loads(raw)
        except ValueError:
            loaded = {}
        if isinstance(loaded, dict):
            data = {str(k): str(v) for k, v in loaded.items()}
    with _catalog_lock:
        _catalogs[code] = data
    return data


def all_catalogs() -> dict[str, dict[str, str]]:
    """Every shipped catalog, keyed by its canonical language code."""
    return {code: load_catalog(code) for code in SUPPORTED_LANGUAGES}


def get_current_lang() -> str:
    """The language `t()` uses when no explicit language is given."""
    with _current_lock:
        return _current_lang


def set_current_lang(lang: str) -> str:
    """Set the process-global language; returns the canonical code used."""
    code = normalize_lang(lang)
    global _current_lang
    with _current_lock:
        _current_lang = code
    return code


def detect_lang(cli_lang: str | None = None, cfg=None, env=None) -> str:
    """Which language to show, by precedence:

    ``cli_lang`` > ``PEDACITO_LANG`` > ``PEDACITO_LANGUAGE`` > ``cfg.language``
    > ``"en"``. `env` is an optional mapping used instead of ``os.environ``
    (handy for tests and for the GUI's in-process environment view)."""
    if cli_lang is not None and str(cli_lang).strip():
        return normalize_lang(cli_lang)
    environ = os.environ if env is None else env
    for var in ("PEDACITO_LANG", "PEDACITO_LANGUAGE"):
        value = environ.get(var)
        if value:
            return normalize_lang(value)
    if cfg is not None:
        value = getattr(cfg, "language", "") or ""
        if str(value).strip():
            return normalize_lang(value)
    return DEFAULT_LANGUAGE


class _SafeMap(dict):
    """A ``format_map`` mapping that stands in for missing keys instead of
    raising, so a translation whose placeholder the caller forgot can never
    crash the UI."""

    def __missing__(self, key):
        return "{" + str(key) + "}"


def _safe_format(text: str, kwargs: dict) -> str:
    """``text.format_map`` that never raises ``KeyError``/``ValueError``.

    Missing keys render as their literal placeholder, and a malformed format
    string (unbalanced braces, a bad conversion) falls back to the raw text."""
    try:
        return text.format_map(_SafeMap(kwargs))
    except (KeyError, ValueError, IndexError):
        return text


def t(key: str, lang: str | None = None, **kwargs) -> str:
    """The message for `key` in `lang` (default: the current language).

    Lookup falls back ``lang`` -> ``"en"`` -> the key itself, so a missing
    translation degrades to English, then to the raw key, but never raises."""
    code = normalize_lang(lang if lang is not None else get_current_lang())
    for candidate in (code, DEFAULT_LANGUAGE):
        catalog = load_catalog(candidate)
        if key in catalog:
            text = catalog[key]
            return _safe_format(text, kwargs) if kwargs else text
    return key


def save_user_language(lang: str) -> dict:
    """Persist `lang` as the user's interface language and report what changed.

    Validates the value (raising ``ValueError`` for anything unsupported),
    reads the user config (starting from ``config.TEMPLATE`` when none exists
    yet), patches ``language = "<code>"`` comment-preservingly, and writes the
    result atomically. Returns ``{"language", "persisted", "created"}``; the
    ``created`` flag is True only when the config file itself did not exist."""
    if not is_valid_lang(lang):
        raise ValueError(
            f"unsupported language {lang!r}; choose one of "
            + ", ".join(SUPPORTED_LANGUAGES))
    code = normalize_lang(lang)

    # Imported lazily: `config` pulls in kvcache, and `gui.settings` pulls in
    # the review server; neither belongs on the critical path of every import
    # of this module.
    from . import config as config_mod
    from .gui import settings as gui_settings

    path = config_mod.USER_CONFIG
    created = not path.exists()
    text = config_mod.TEMPLATE if created else path.read_text(encoding="utf-8")
    new_text = gui_settings.set_values(text, {"language": code})

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        tmp.write_text(new_text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    return {"language": code, "persisted": True, "created": created}
