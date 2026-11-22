"""WP-05: bilingual catalogs, documentation links/anchors, switcher and parity.

This module is the end-to-end guard for the v0.3.1 Spanish localization pass.
It deliberately checks the *shipped* artifacts rather than the engine:

* both locale catalogs are valid UTF-8 JSON with the same keys, no empty
  strings and the same `{placeholder}` names per key;
* every relative ``.md`` link and every ``#anchor`` fragment in the root
  guides and in ``docs/`` / ``docs/es/`` resolves to a real file and heading,
  using GitHub's GFM heading-slug algorithm (lowercase, inline markdown
  stripped, Unicode word characters kept, punctuation dropped, duplicates
  suffixed ``-1``, ``-2`` ...);
* every paired document carries a symmetric EN <-> ES switcher near the top;
* every public ``Config`` field is documented in the English and Spanish
  configuration reference;
* the packaging metadata ships the catalogs and the Spanish guides;
* offline Spanish ``how`` retrieval and mirror resolution still work.

Run with ``.venv/bin/pytest -q tests/test_i18n_and_docs_integrity.py``.
"""
from __future__ import annotations

import dataclasses
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import how, how_index
from pedacito.config import Config

ROOT = Path(__file__).resolve().parents[1]
LOCALES = ROOT / "pedacito" / "locales"

ROOT_GUIDES = ("README", "QUICKSTART", "MODEL_GUIDE", "TECHNIQUES")
DOCS_DIR = ROOT / "docs"
DOCS_ES_DIR = DOCS_DIR / "es"

# Every Markdown file the integrity audit covers, in a stable order.
DOC_FILES = [
    *sorted(ROOT.glob("*.md")),
    *sorted(DOCS_DIR.glob("*.md")),
    *sorted(DOCS_ES_DIR.glob("*.md")),
]

# EN -> ES mirror pairs. The five root guides and every docs/ page.
PAIRS = (
    [(f"{name}.md", f"{name}.es.md") for name in ROOT_GUIDES]
    + [(f"docs/{p.name}", f"docs/es/{p.name}")
       for p in sorted(DOCS_DIR.glob("*.md"))]
)


# --------------------------------------------------------------- locale tests --

def _read_json(name: str) -> dict:
    path = LOCALES / name
    raw = path.read_bytes()
    text = raw.decode("utf-8")  # raises UnicodeDecodeError on invalid UTF-8
    data = json.loads(text)
    assert isinstance(data, dict), f"{name} is not a JSON object"
    return data


def test_locale_catalogs_are_valid_utf8_json():
    for name in ("en.json", "es.json"):
        catalog = _read_json(name)
        assert catalog, f"{name} is empty"
        for key, value in catalog.items():
            assert isinstance(key, str) and key, f"{name}: empty key"
            assert isinstance(value, str), f"{name}: {key!r} is not a string"
            assert value.strip(), f"{name}: {key!r} is empty"


def test_locale_catalogs_have_identical_key_sets():
    en = _read_json("en.json")
    es = _read_json("es.json")
    missing_es = sorted(set(en) - set(es))
    missing_en = sorted(set(es) - set(en))
    assert not missing_es, f"es.json is missing keys: {missing_es}"
    assert not missing_en, f"en.json is missing keys: {missing_en}"
    assert len(en) == len(es) and len(en) > 0


_PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")


def test_locale_placeholders_match_per_key():
    en = _read_json("en.json")
    es = _read_json("es.json")
    mismatches = []
    for key in sorted(set(en) & set(es)):
        en_names = set(_PLACEHOLDER_RE.findall(en[key]))
        es_names = set(_PLACEHOLDER_RE.findall(es[key]))
        if en_names != es_names:
            mismatches.append((key, sorted(en_names), sorted(es_names)))
    assert not mismatches, f"placeholder mismatch: {mismatches}"


# The Profiles/Discovery polish (collapsible pairing, Strata onboarding,
# hosted-model context/pricing columns, Servers auth hint) ships its strings
# in both languages rather than falling back to English in the Spanish UI.
_POLISH_KEYS = {
    "settings.pair.title", "settings.pair.paired",
    "settings.pair.not_configured", "settings.pair.configure",
    "settings.pair.hide", "settings.pair.toggle_title",
    "settings.pair.description", "settings.pair.architect",
    "settings.pair.scout", "settings.pair.create_starter",
    "settings.pair.starter_name", "settings.pair.base_url",
    "settings.pair.model_id", "settings.pair.model_ph",
    "settings.pair.vram", "settings.pair.tier",
    "settings.pair.starter_hint", "settings.pair.save",
    "settings.pair.saving", "settings.pair.pick_error",
    "settings.pair.saved", "settings.pair.user_only",
    "settings.scan.hint", "settings.scan.api_key_required",
    "settings.scan.api_key_placeholder", "settings.scan.api_key_hint",
    "settings.hosted.title", "settings.hosted.refresh_catalog",
    "settings.hosted.refreshing", "settings.hosted.refreshed",
    "settings.hosted.refresh_failed", "settings.hosted.refresh_catalog_title",
    "settings.hosted.col_ctx", "settings.hosted.col_price",
    "settings.hosted.price_title",
    "gui.servers.auth_required", "gui.servers.auth_hint",
}


def test_profiles_polish_keys_are_translated_in_both_catalogs():
    en = _read_json("en.json")
    es = _read_json("es.json")
    for name, catalog in (("en.json", en), ("es.json", es)):
        missing = sorted(_POLISH_KEYS - set(catalog))
        assert not missing, f"{name} is missing the polish keys: {missing}"
    # The onboarding promise is explicit in both languages.
    assert "Strata" in en["settings.scan.hint"]
    assert "Strata" in es["settings.scan.hint"]
    assert "STRATA_API_KEY" in en["gui.servers.auth_hint"]
    assert "STRATA_API_KEY" in es["gui.servers.auth_hint"]


# ----------------------------------------------------- GFM slug / link audit --

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_ATX_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_LINK_RE = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)>\s]+)>?(?:\s+[\"'][^\"']*[\"'])?\s*\)")
_INLINE_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")


def _strip_fenced_code(text: str) -> str:
    """Blank fenced code blocks so shell comments and examples are not
    mistaken for headings or links."""
    out: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        match = _FENCE_RE.match(line)
        if match:
            token = match.group(1)[0]
            if fence is None:
                fence = token
            elif fence == token:
                fence = None
            out.append("")
            continue
        out.append("" if fence is not None else line)
    return "\n".join(out)


def _plain_heading(text: str) -> str:
    """Render a heading's text the way GitHub does before slugging: strip
    inline code ticks, links/images, and emphasis markers."""
    text = text.replace("`", "")
    text = _INLINE_LINK_RE.sub(r"\1", text)
    text = re.sub(r"\*\*|__|~~|\*", "", text)
    return text


def github_slug(text: str) -> str:
    """GitHub's GFM heading slug: lowercase, keep ``\\w`` (Unicode), spaces and
    hyphens, drop every other punctuation character, then turn each space into
    a hyphen. ``"a / b"`` -> ``"a--b"``; ``"a & b"`` -> ``"a--b"``."""
    lowered = _plain_heading(text).lower()
    kept = "".join(ch for ch in lowered
                   if ch.isalnum() or ch == "_" or ch in " -")
    return kept.replace(" ", "-")


def _headings(text: str) -> list[str]:
    headings: list[str] = []
    for line in _strip_fenced_code(text).splitlines():
        match = _ATX_RE.match(line)
        if match:
            headings.append(match.group(2))
    return headings


def _anchors(text: str) -> set[str]:
    """Every anchor GitHub generates for one file, including ``-1``/``-2``
    suffixes for duplicate headings."""
    seen: dict[str, int] = {}
    anchors: set[str] = set()
    for raw in _headings(text):
        slug = github_slug(raw)
        if slug in seen:
            seen[slug] += 1
            unique = f"{slug}-{seen[slug]}"
            while unique in anchors:
                seen[slug] += 1
                unique = f"{slug}-{seen[slug]}"
        else:
            seen[slug] = 0
            unique = slug
        anchors.add(unique)
    return anchors


def _links(text: str) -> list[str]:
    return _LINK_RE.findall(_strip_fenced_code(text))


_ANCHOR_CACHE: dict[Path, set[str]] = {}


def _anchors_of(path: Path) -> set[str]:
    resolved = path.resolve()
    if resolved not in _ANCHOR_CACHE:
        _ANCHOR_CACHE[resolved] = _anchors(
            resolved.read_text(encoding="utf-8"))
    return _ANCHOR_CACHE[resolved]


def _is_external(target: str) -> bool:
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target)) or \
        target.startswith("//")


def _check_anchor(doc: Path, target: str, problems: list[str]) -> None:
    """Validate one link target (file and optional fragment) from `doc`."""
    if not target or _is_external(target):
        return
    target = unquote(target)
    if target.startswith("#"):
        fragment = target[1:]
        if fragment and fragment not in _anchors_of(doc):
            problems.append(f"{doc.relative_to(ROOT)}: missing anchor {target!r}")
        return
    if target.startswith("/"):
        return  # site-absolute: not a repository relative link
    path_part, _, fragment = target.partition("#")
    if not path_part:
        return
    if not path_part.lower().endswith(".md"):
        return  # images and other assets are not part of this audit
    resolved = (doc.parent / path_part).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        problems.append(f"{doc.relative_to(ROOT)}: link escapes repo: {target!r}")
        return
    if not resolved.is_file():
        problems.append(f"{doc.relative_to(ROOT)}: missing file {target!r}")
        return
    if fragment and fragment not in _anchors_of(resolved):
        problems.append(f"{doc.relative_to(ROOT)}: missing anchor {target!r}")


@pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_markdown_links_and_anchors_resolve(doc):
    problems: list[str] = []
    for target in _links(doc.read_text(encoding="utf-8")):
        _check_anchor(doc, target, problems)
    assert not problems, "\n".join(problems)


def test_slugifier_matches_github_examples():
    assert github_slug("Language & localization (`language`, `llm_reply_language`)") \
        == "language--localization-language-llm_reply_language"
    assert github_slug("Project rules: `.pedacito/rules.md`") \
        == "project-rules-pedacitorulesmd"
    assert github_slug("a / b") == "a--b"
    assert github_slug("a & b") == "a--b"
    assert github_slug("Cómo cambiar el idioma") == "cómo-cambiar-el-idioma"
    assert github_slug("2. Reasoning *before* acting") == "2-reasoning-before-acting"
    # Duplicate headings get GitHub's numeric suffixes.
    anchors = _anchors("# Same\n## Same\n### Same\n")
    assert anchors == {"same", "same-1", "same-2"}


# ----------------------------------------------------------- switcher parity --

def _switcher_target(doc: Path, counterpart: Path) -> bool:
    """True when the first few lines link to `counterpart`."""
    head = "\n".join(doc.read_text(encoding="utf-8").splitlines()[:6])
    for target in _links(head):
        if _is_external(target):
            continue
        if (doc.parent / unquote(target).split("#", 1)[0]).resolve() \
                == counterpart.resolve():
            return True
    return False


@pytest.mark.parametrize("en_rel,es_rel", PAIRS,
                         ids=lambda value: value.replace("/", "_"))
def test_language_switcher_is_symmetric(en_rel, es_rel):
    en = ROOT / en_rel
    es = ROOT / es_rel
    assert en.is_file(), f"missing English doc {en_rel}"
    assert es.is_file(), f"missing Spanish mirror {es_rel}"
    assert _switcher_target(en, es), \
        f"{en_rel} has no near-top link to {es_rel}"
    assert _switcher_target(es, en), \
        f"{es_rel} has no near-top link to {en_rel}"


# ------------------------------------------------------------ config parity ---

def _config_fields() -> list[str]:
    return sorted(
        field.name for field in dataclasses.fields(Config)
        if not field.name.startswith("_") and field.name != "sources")


def _documents_field(text: str, name: str) -> bool:
    # The reference documents each setting as a backticked key, so an
    # incidental prose mention cannot satisfy the check, and `max_steps`
    # cannot be satisfied by `subagent_max_steps`.
    return f"`{name}`" in text


@pytest.mark.parametrize("doc_name", ["docs/CONFIG.md", "docs/es/CONFIG.md"])
def test_every_public_config_field_is_documented(doc_name):
    text = (ROOT / doc_name).read_text(encoding="utf-8")
    missing = [name for name in _config_fields()
               if not _documents_field(text, name)]
    assert not missing, f"{doc_name} does not document: {missing}"


# ----------------------------------------------------- packaging and runtime ---

def test_pyproject_ships_catalogs_and_spanish_guides():
    import tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = data["tool"]["setuptools"]["package-data"]["pedacito"]
    assert "locales/*.json" in package_data

    data_files = data["tool"]["setuptools"]["data-files"]
    root_share = data_files["share/pedacito"]
    for name in ("README.es.md", "QUICKSTART.es.md", "MODEL_GUIDE.es.md",
                 "TECHNIQUES.es.md"):
        assert name in root_share, f"data-files missing {name}"
    assert "docs/es/*.md" in data_files["share/pedacito/docs/es"]


def test_offline_spanish_how_retrieval_works():
    matches = how_index.search("¿cómo cambio el idioma?")
    gui_ids = [item["id"] for item in matches.get("gui") or []]
    assert "language-switch" in gui_ids, gui_ids


def test_how_resolves_spanish_doc_mirrors():
    english = dict(how._documentation_paths(ROOT, "en"))
    spanish = dict(how._documentation_paths(ROOT, "es"))
    assert english.keys() == spanish.keys()

    # Root guides resolve to their .es.md mirror.
    for name in ("README.md", "QUICKSTART.md", "MODEL_GUIDE.md"):
        assert spanish[name] == ROOT / name.replace(".md", ".es.md")
        assert spanish[name].is_file()

    # docs/ pages resolve to docs/es/ mirrors, keeping the English chunk name.
    for path in sorted(DOCS_DIR.glob("*.md")):
        rel = f"docs/{path.name}"
        assert spanish[rel] == DOCS_ES_DIR / path.name
        assert spanish[rel].is_file()

    assert english["README.md"] == ROOT / "README.md"
    assert english["docs/CONFIG.md"] == ROOT / "docs" / "CONFIG.md"
