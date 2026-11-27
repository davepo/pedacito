"""`pedacito how`: built-in help from a structured CLI/GUI/setting index.

Usage
-----
    pedacito how "how do I set up a config file?"
    pedacito how --offline "how do I disable thinking?"
    pedacito how --json "how do I import models?"
    pedacito how "how do I scroll back in pedacito tui?"
    pedacito how                                  # interactive help prompt

Unlike every other Pedacito command, `how` does not need a project or an
index. It first retrieves the relevant CLI commands/flags, exact GUI navigation
paths, and config settings deterministically (`how_index.py`). When a fast,
configured model server is reachable it may rewrite those retrieved facts into
a short answer with GUI, CLI and settings sections; otherwise -- or with
`--offline` -- the structured cards are printed directly, so help always works
before a model is configured. The `tui` card also covers PgUp/PgDn scrolling,
mouse-wheel modes (`?1007h` native selection vs. optional SGR), F2, Tab
completion and command history.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import site
import sys
import sysconfig
import unicodedata
from collections import Counter
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from . import config as config_mod
from . import how_index, i18n, prompts
from .chunker import Chunk, chunk_source
from .config import (Config, ConfigError, auto_fit, find_project_config,
                     load as load_config, missing_hardware)
from .llm import LLMClient, LLMError, estimate_tokens
from .references import chunk_reference

DOC_FILES = ("README.md", "QUICKSTART.md")
# Optional docs: older installs may not have these, so they are not required
# for find_docs_root(), but they join the help corpus when present.
OPTIONAL_DOC_FILES = ("MODEL_GUIDE.md",)
# The README split moved the long-form prose into docs/*.md. Those guides are
# optional too (a wheel may predate them), but when present they are part of
# the corpus under their checkout-relative name (for example docs/CONFIG.md).
DOCS_DIR = "docs"
# Split long documentation sections and source units so retrieval can pick the
# matching part instead of feeding a 20k-character section to a small model.
MAX_ITEM_CHARS = 3200
# Answer budget for help. Help answers are intentionally short; capping this
# leaves more of the window for the documentation/source excerpts.
MAX_HELP_ANSWER_TOKENS = 2400

_SECRET_KEYS = {"api_key"}
_CORE_KEYS = (
    "base_url", "model", "api_key", "vram_gb", "active_profile",
    "active_tier", "context_tokens", "max_steps", "max_tokens_answer",
    "json_mode",
)
_CONFIG_WORDS = {
    "config", "configuration", "configure", "setup", "install", "vram",
    "model", "api", "key", "profile", "profiles", "toml", "setting",
    "settings", "server", "url", "hosted", "deepseek", "openai",
}

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do",
    "does", "for", "from", "get", "how", "i", "if", "in", "is", "it", "me",
    "my", "of", "on", "or", "should", "so", "that", "the", "this", "to",
    "use", "using", "what", "when", "where", "which", "who", "why", "will",
    "with", "would", "you", "your",
}

# Spanish question filler (accent-folded), so Spanish questions rank offline.
_ES_STOP = {
    "como", "cual", "cuales", "que", "quien", "donde", "cuando", "por",
    "para", "con", "sin", "el", "la", "los", "las", "un", "una", "unos",
    "unas", "de", "del", "al", "en", "es", "se", "mi", "mis", "tu", "tus",
    "su", "puedo", "puedes", "quiero", "necesito", "hacer", "hay", "y", "o",
    "lo", "le", "me", "te", "cambio", "cambiar", "uso", "usar", "esta",
    "este", "eso", "muy", "mas", "ya", "no", "si",
}

_ES_SYNONYMS = {
    "idioma": ("language", "lang", "es", "spanish", "espanol"),
    "lenguaje": ("language", "lang"),
    "espanol": ("language", "lang", "es", "spanish"),
    "configurar": ("configure", "setup", "config", "settings"),
    "configuracion": ("configuration", "config", "settings", "toml"),
    "modelo": ("model", "models", "profile", "server"),
    "modelos": ("model", "models", "profile", "server"),
    "deshacer": ("undo", "restore", "backup", "snapshot"),
    "revertir": ("revert", "restore", "backup", "snapshot"),
    "tema": ("theme", "appearance", "gui"),
    "habilidad": ("skill", "skills"),
    "habilidades": ("skill", "skills"),
    "instalar": ("install", "pip", "setup"),
    "instalacion": ("install", "pip", "setup"),
    "comando": ("command", "commands", "cli", "usage"),
    "comandos": ("command", "commands", "cli", "usage"),
    "error": ("fail", "failure", "troubleshoot"),
    "ayuda": ("help", "how"),
    "herramienta": ("tool", "tools"),
    "herramientas": ("tool", "tools"),
    "indice": ("index", "indexing"),
    "ignorar": ("ignore", "exclude", "pedacitoignore"),
    "archivo": ("file", "files"),
    "archivos": ("file", "files"),
    "cambios": ("changes", "apply", "review"),
    "seguridad": ("security", "sandbox", "permissions"),
    "memoria": ("memory", "vram", "gpu", "context"),
    "lento": ("slow", "performance", "context"),
}


def _fold(text: str) -> str:
    """Strip accents so "cómo" and "como" match (English text is unchanged)."""
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))


# Question words that rarely appear verbatim in documentation. Expanding them
# makes "how do I set this up?" find Configure/--setup, "undo" find restore,
# and so on.
_SYNONYMS = {
    "setup": ("configure", "configuration", "init", "install", "config"),
    "configure": ("setup", "config", "configuration", "toml", "settings"),
    "config": ("configuration", "configure", "settings", "toml", "init"),
    "install": ("pip", "setup", "install"),
    "command": ("commands", "cli", "run", "usage"),
    "run": ("command", "execute", "usage"),
    "undo": ("restore", "backup", "snapshot"),
    "revert": ("restore", "backup", "snapshot"),
    "edit": ("apply", "review", "changes"),
    "start": ("new", "scaffold", "create"),
    "create": ("new", "scaffold", "generate"),
    "error": ("fail", "failure", "troubleshoot", "wrong", "limits"),
    "broke": ("error", "fail", "troubleshoot", "wrong"),
    "slow": ("cost", "tokens", "time", "context", "performance"),
    "memory": ("vram", "gpu", "context"),
    "api": ("key", "hosted", "deepseek", "openai"),
    "model": ("profile", "models", "server"),
    "index": ("indexing", "map", "summaries", "chunks"),
    "ignore": ("exclude", "excluded", "pedacitoignore"),
    "exclude": ("ignore", "pedacitoignore"),
}

_WORD_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*")
# "what does collapse_repair_turns do?", "what is small_context_tokens?",
# "explain auto_green_ratchet": the named identifier right after the question
# verb is the setting the user means.
_SETTING_QUERY_RE = re.compile(
    r"\b(?:what(?:'s| is| does)|explain|describe|tell me about|"
    r"meaning of|document(?:ation for|ed)?)\s+`?([A-Za-z_][A-Za-z0-9_]*)",
    re.I)


def named_setting(question: str) -> dict | None:
    """The exact setting a question names, or None.

    A setting is only pinned when the user names it unambiguously: after a
    "what is/does/explain" verb, as a snake_case key anywhere, or as the
    whole (trimmed) question. A generic word such as `model` inside "how do I
    import models?" is not a setting lookup and keeps the procedural cards.
    """
    keys = {item["key"]: item for item in how_index.settings_dictionary()}
    low = " ".join((question or "").lower().split())
    match = _SETTING_QUERY_RE.search(low)
    if match and match.group(1) in keys:
        return keys[match.group(1)]
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", low))
    exact = [name for name in keys if name in tokens]
    if not exact:
        return None
    underscored = [name for name in exact if "_" in name]
    if underscored:
        return keys[max(underscored, key=len)]
    if len(exact) == 1:
        extra = tokens - set(_STOP) - {
            exact[0], "setting", "option", "config", "toml", "value",
            "default", "mean", "means", "meaning", "change", "set", "use",
            "please", "about",
        }
        if not extra:
            return keys[exact[0]]
    return None


def _pin_named_setting(question: str, matches: dict) -> dict | None:
    """Force a named setting to rank #1 (retrieval is deterministic)."""
    item = named_setting(question)
    if item is None:
        return None
    others = [s for s in (matches.get("settings") or [])
              if s.get("key") != item["key"]]
    pinned = {"score": 999.0, **item}
    matches["settings"] = [pinned] + others[:4]
    return pinned


_WHEN_MARKERS = (
    "off by default", "on by default", "0 = off", "keep on", "unless",
    "only if", "raise", "lower", "turn it on", "when you", "for servers",
    "for a ", "use it when",
)


def _setting_when(item: dict, lang: str = "en") -> str:
    """A deterministic "why/when to change it" line from the setting's own
    documentation, with a type-aware fallback."""
    lang = i18n.normalize_lang(lang)
    help_text = " ".join(how_index.setting_help(item, lang).split())
    if lang == "en":
        for sentence in re.split(r"(?<=[.!?])\s+", help_text):
            if sentence and any(mark in sentence.lower() for mark in _WHEN_MARKERS):
                return sentence
    choices = item.get("choices")
    if choices:
        shown = ", ".join(f"`{c}`" for c in choices if c != "")
        return i18n.t("how.when_choices", lang=lang, choices=shown)
    kind = item.get("type")
    if kind == "bool":
        return i18n.t("how.when_bool", lang=lang)
    if kind in ("int", "float"):
        return i18n.t("how.when_number", lang=lang)
    return i18n.t("how.when_other", lang=lang)


def render_setting_answer(question: str, item: dict,
                          matches: dict | None = None,
                          lang: str = "en") -> str:
    """A direct setting card: what it does, when to change it, the default,
    the exact config.toml syntax and the GUI path (the GUI's How tab renders
    the same item with its Open in Settings button)."""
    lang = i18n.normalize_lang(lang)

    def t(key: str, **kw) -> str:
        return i18n.t(f"how.{key}", lang=lang, **kw)

    key = item["key"]
    help_text = " ".join(how_index.setting_help(item, lang).split())
    gui = how_index.es_card_text(item.get("gui", "⚙ Settings → All settings"), lang)
    group = how_index.es_card_text(
        item.get("group") or t("setting_group_default"), lang)
    lines = [] if lang != "es" else [t("banner"), ""]
    lines += [
        f"**`{key}`** \u2014 {group}",
        "",
        t("setting_what"),
        help_text or t("setting_nodoc"),
        "",
        t("setting_when"),
        _setting_when(item, lang),
        "",
        t("setting_default_line", value=item.get("default_text", ""),
          type=item.get("type", "str")),
        "",
        t("gui_heading"),
        f"- {gui}",
        f"- {t('setting_open', gui=gui)}",
        "",
        t("cli_heading"),
        f"- {t('setting_cli_line')}",
        "",
        t("settings_heading"),
        "",
        "```toml",
        str(item.get("toml") or f"{key} = ..."),
        "```",
    ]
    if item.get("profile_gui"):
        lines += ["", t("setting_profile", gui=item["profile_gui"])]
    lines += ["", t("setting_footer")]
    return "\n".join(lines).rstrip() + "\n"


_ES_CONFIG_KEY_RE = re.compile(r"`([a-z_][a-z0-9_]*)`")


@lru_cache(maxsize=4)
def _parse_spanish_settings(path: str) -> dict[str, str]:
    """Setting key -> Spanish description, from the setting tables of the
    Spanish configuration reference (`docs/es/CONFIG.md`)."""
    out: dict[str, str] = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        if not line.startswith("| `"):
            continue
        cells = re.split(r"(?<!\\)\|", line)
        if len(cells) < 4:
            continue
        meaning = cells[-2].strip().replace("\\|", "|")
        if not meaning:
            continue
        for key in _ES_CONFIG_KEY_RE.findall(cells[1]):
            out.setdefault(key, meaning)
    return out


def spanish_setting_help() -> dict[str, str]:
    """Spanish setting descriptions from the bundled Spanish reference; an
    empty mapping when the Spanish guide is not installed."""
    try:
        root = find_docs_root()
    except HowError:
        return {}
    path = root / DOCS_DIR / "es" / "CONFIG.md"
    if not path.is_file():
        return {}
    return _parse_spanish_settings(str(path))


class HowError(RuntimeError):
    """Raised when the help corpus cannot be located or built."""


def find_docs_root() -> Path:
    """Return the directory containing README.md and QUICKSTART.md: $PEDACITO_DOCS,
    the checkout around this package, the package directory, package-relative
    and interpreter `share/pedacito` data-files locations, then the current
    directory. A normal wheel install keeps the modular guides beside those
    files in `<share>/pedacito/docs/`."""
    here = Path(__file__).resolve().parent
    candidates: list[Path] = []
    env = os.environ.get("PEDACITO_DOCS")
    if env:
        candidates.append(Path(env).expanduser())
    candidates += [here.parent, here]
    # `pip install --target DIR` puts the package at DIR/pedacito and its
    # data-files at DIR/share/pedacito, while sysconfig still reports the
    # running interpreter's prefix. Check the package-relative layouts first
    # so an isolated target install finds its own docs.
    candidates += [
        here.parent / "share" / "pedacito",
        here.parent / "data" / "share" / "pedacito",
    ]
    try:
        data = sysconfig.get_path("data")
        if data:
            candidates.append(Path(data) / "share" / "pedacito")
    except Exception:
        pass
    try:
        candidates.append(Path(site.getuserbase()) / "share" / "pedacito")
    except Exception:
        pass
    candidates.append(Path(sys.prefix) / "share" / "pedacito")
    # Last, so a user project's own README cannot shadow Pedacito's.
    candidates.append(Path.cwd())

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if all((candidate / name).is_file() for name in DOC_FILES):
            return candidate
    raise HowError(
        "Could not find Pedacito's README.md and QUICKSTART.md.\n"
        "Run this from a Pedacito checkout, point PEDACITO_DOCS at the "
        "checkout, or reinstall Pedacito so its documentation is included.")


def _split_large(chunk: Chunk, limit: int = MAX_ITEM_CHARS) -> list[Chunk]:
    """Split an oversized chunk at line boundaries, keeping line ranges accurate."""
    source = chunk.source or ""
    if len(source) <= limit:
        return [chunk]
    lines = source.splitlines()
    parts: list[Chunk] = []
    cur: list[str] = []
    size = 0
    start = chunk.start
    for line in lines:
        add = len(line) + 1
        if cur and size + add > limit:
            body = "\n".join(cur)
            parts.append(replace(chunk, start=start,
                                 end=start + len(cur) - 1, source=body))
            start += len(cur)
            cur, size = [], 0
        cur.append(line)
        size += add
    if cur:
        parts.append(replace(chunk, start=start,
                             end=start + len(cur) - 1,
                             source="\n".join(cur)))
    return parts


def _string_constants(rel: str, source: str) -> list[Chunk]:
    """Extract config.TEMPLATE and REFERENCE_TEMPLATE, which chunk_source skips
    (module-level strings) but "how do I write a config?" answers should quote."""
    wanted = {"TEMPLATE", "REFERENCE_TEMPLATE"}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()
    out: list[Chunk] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in wanted:
            continue
        start = node.lineno
        end = node.end_lineno or start
        body = "\n".join(lines[start - 1:end])
        out.append(Chunk(file=rel, qualname=target.id, kind="constant",
                         start=start, end=end,
                         signature=f"{target.id} (config template)",
                         source=body))
    return out


def _spanish_mirror(name: str, path: Path, root: Path) -> Path:
    """The Spanish mirror of a doc file when it exists, else `path`."""
    rel = Path(name)
    if rel.parent.name == DOCS_DIR:
        mirror = root / DOCS_DIR / "es" / rel.name
    else:
        mirror = root / f"{rel.stem}.es{rel.suffix}"
    return mirror if mirror.is_file() else path


def _documentation_paths(root: Path, lang: str = "en") -> list[tuple[str, Path]]:
    """The root guides plus every docs/*.md, keyed by their checkout-relative
    name. The docs directory is optional: an older installation without it
    still gets the README/QUICKSTART/source corpus."""
    paths = [(name, root / name) for name in DOC_FILES + OPTIONAL_DOC_FILES]
    docs = root / DOCS_DIR
    if docs.is_dir():
        paths += [(f"{DOCS_DIR}/{path.name}", path)
                  for path in sorted(docs.glob("*.md"))]
    if lang == "es":
        # Chunks keep the English name (the ranking rules key on it) but read
        # the Spanish text, falling back per file.
        paths = [(name, _spanish_mirror(name, path, root))
                 for name, path in paths]
    return paths


def build_corpus(docs_root: Path | None = None,
                 lang: str = "en") -> list[Chunk]:
    """README, QUICKSTART, docs/*.md, the config templates and the package's
    Python modules as searchable chunks."""
    root = docs_root or find_docs_root()
    chunks: list[Chunk] = []
    for name, path in _documentation_paths(root, lang):
        if not path.is_file():
            continue
        try:
            found = chunk_reference(name, path)
        except OSError:
            continue
        for chunk in found:
            chunks.extend(_split_large(chunk))

    pkg = Path(__file__).resolve().parent
    for path in sorted(pkg.glob("*.py")):
        if path.name == "how.py":
            # Its text mentions every setting and would crowd out the real docs.
            continue
        rel = f"pedacito/{path.name}"
        try:
            source = path.read_text(encoding="utf-8")
            found = chunk_source(rel, source)
        except (OSError, SyntaxError):
            continue
        for chunk in found:
            chunks.extend(_split_large(chunk))
        if path.name == "config.py":
            for chunk in _string_constants(rel, source):
                chunks.extend(_split_large(chunk))
    return chunks


def _weighted_terms(text: str) -> dict[str, float]:
    """Lowercase question terms with weights: a compound identifier such as
    `max_steps` counts fully, its pieces much less."""
    raws: list[str] = []
    for raw in _WORD_RE.findall(_fold(text.lower())):
        raw = raw.strip(".-")
        if len(raw) >= 2 and raw not in _STOP and raw not in _ES_STOP:
            raws.append(raw)

    weights: dict[str, float] = {}
    for raw in raws:
        weights[raw] = 1.0
        if "_" in raw or "-" in raw:
            weights[raw] = 1.6          # identifier, not prose
    for raw in raws:
        for part in re.split(r"[_\-.]", raw):
            if len(part) >= 3 and part not in _STOP and part not in _ES_STOP:
                weights.setdefault(part, 0.35)
        if raw.endswith("s") and len(raw) > 4:
            weights.setdefault(raw[:-1], 0.5)

    for term in list(weights):
        for synonym in _SYNONYMS.get(term, ()) + _ES_SYNONYMS.get(term, ()):
            weights.setdefault(synonym, 0.6)
    return weights


def _search_text(chunk: Chunk) -> str:
    return " ".join((chunk.file, chunk.qualname, chunk.signature,
                     chunk.docline, chunk.source or ""))


def _token_counts(text: str) -> Counter:
    return Counter(_WORD_RE.findall(text.lower()))


def rank_chunks(question: str, chunks: list[Chunk]) -> list[tuple[float, Chunk]]:
    """Rank chunks against the question with a small BM25 variant; terms in a
    chunk's file name or heading score extra."""
    weights = _weighted_terms(question)
    if not weights:
        return [(0.0, c) for c in chunks]

    corpus = []
    for chunk in chunks:
        counts = _token_counts(_search_text(chunk))
        heading = set(_WORD_RE.findall(
            (chunk.file + " " + chunk.qualname).lower()))
        corpus.append((chunk, counts, heading, sum(counts.values())))

    n = len(corpus)
    avgdl = (sum(c[3] for c in corpus) / n) if n else 1
    df = {t: sum(1 for _, counts, _, _ in corpus if counts.get(t))
          for t in weights}

    ranked: list[tuple[float, Chunk]] = []
    for chunk, counts, heading, dl in corpus:
        score = 0.0
        for term, weight in weights.items():
            tf = counts.get(term, 0)
            if not tf:
                continue
            idf = math.log(1 + (n - df.get(term, 0) + 0.5) /
                           (df.get(term, 0) + 0.5))
            score += weight * idf * (tf * 2.2) / (
                tf + 1.2 * (0.25 + 0.75 * dl / max(avgdl, 1)))
            if term in heading:
                score += weight * idf * 2.5
        if chunk.file in DOC_FILES or chunk.file.startswith(DOCS_DIR + "/"):
            score *= 1.12
        ranked.append((score, chunk))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return ranked


def _format_ref(chunk: Chunk) -> str:
    return (f"=== {chunk.file} lines {chunk.start}-{chunk.end}: "
            f"{chunk.qualname} ===\n{chunk.source.strip()}")


def _context_budget(cfg: Config, question: str, snapshot: str,
                    history: list[tuple[str, str]]) -> int:
    """Characters available for excerpts, after system prompt, question,
    config summary, history, and the answer budget."""
    answer = min(max(256, int(cfg.max_tokens_answer)), MAX_HELP_ANSWER_TOKENS)
    used = estimate_tokens(QUESTION_SYSTEM) + estimate_tokens(question)
    used += estimate_tokens(snapshot)
    used += sum(estimate_tokens(q) + estimate_tokens(a[:1200])
                for q, a in history[-2:])
    used += 256                     # message framing / headings
    room = cfg.context_tokens - cfg.reserve_tokens - answer - used
    chars = int(max(room, 512) * 3.6)
    chars = min(chars, max(3000, int(cfg.max_gathered_chars)))
    # A few sections plus source is enough; keeps latency and cost down.
    chars = min(chars, 24000)
    return max(2200, chars)


def select_references(question: str, chunks: list[Chunk], cfg: Config,
                      snapshot: str = "",
                      history: list[tuple[str, str]] | None = None
                      ) -> tuple[str, list[Chunk]]:
    """Pick excerpts for one question: always the best README section, the best
    QUICKSTART section, the best relevant docs/ guide and the CLI preamble,
    then top-ranked source and docs until the budget is used."""
    history = history or []
    ranked = rank_chunks(question, chunks)
    by_file: dict[str, list[tuple[float, Chunk]]] = {}
    for score, chunk in ranked:
        by_file.setdefault(chunk.file, []).append((score, chunk))

    budget = _context_budget(cfg, question, snapshot, history)
    picked: list[Chunk] = []
    blocks: list[str] = []
    used = 0
    terms = set(_weighted_terms(question))

    def add(chunk: Chunk | None) -> bool:
        nonlocal used
        if chunk is None or chunk in picked:
            return False
        block = _format_ref(chunk)
        room = budget - used - 2
        if len(block) > room:
            if room < 900:
                return False
            # Keep the head of an over-budget excerpt instead of dropping it.
            block = block[:room].rstrip() + "\n[...truncated to fit]"
        picked.append(chunk)
        blocks.append(block)
        used += len(block) + 2
        return True

    # Best section from each root guide.
    for name in DOC_FILES:
        candidates = [(s, c) for s, c in by_file.get(name, []) if c.source]
        add(candidates[0][1] if candidates else None)

    # The best scored docs/ section, when one matched the question at all.
    docs_candidates = [(s, c) for _name, candidates in by_file.items()
                       if _name.startswith(DOCS_DIR + "/")
                       for s, c in candidates if c.source]
    docs_candidates.sort(key=lambda pair: pair[0], reverse=True)
    if docs_candidates and docs_candidates[0][0] > 0:
        add(docs_candidates[0][1])

    # cli.py's command list helps almost every "what command..." question.
    cli_preamble = next(
        (c for c in chunks
         if c.file == "pedacito/cli.py" and c.qualname == "module"), None)
    add(cli_preamble)

    # Starter config template for setup questions.
    if terms & _CONFIG_WORDS:
        template = next(
            (c for c in chunks
             if c.file == "pedacito/config.py" and c.qualname == "TEMPLATE"),
            None)
        add(template)

    # At least one source-code excerpt beyond the CLI preamble.
    code_added = sum(1 for c in picked if not c.file.endswith(".md"))
    for _score, chunk in ranked:
        if chunk.file.endswith(".md"):
            continue
        if chunk in picked:
            continue
        if add(chunk):
            code_added += 1
        if code_added >= 2:
            break

    # Second section from each doc file, then fill with the best remaining
    # hits until the budget is exhausted.
    for name in DOC_FILES:
        candidates = [(s, c) for s, c in by_file.get(name, []) if c.source]
        if len(candidates) > 1:
            add(candidates[1][1])
    for _score, chunk in ranked:
        if used >= budget * 0.92:
            break
        add(chunk)
    return "\n\n".join(blocks), picked


def config_snapshot(cfg: Config, root: Path | None = None) -> str:
    """Render the resolved settings for the help prompt, hiding secrets."""
    root = root or Path.cwd()
    lines = [
        "Resolved Pedacito configuration for this help session "
        "(secret values are hidden):",
    ]
    user_cfg = (str(config_mod.USER_CONFIG)
                if config_mod.USER_CONFIG.exists() else "(none)")
    project_cfg = find_project_config(root)
    lines.append(f"  user config: {user_cfg}")
    lines.append(f"  project config: {project_cfg or '(none)'}")
    if cfg.active_profile:
        lines.append(f"  active profile: {cfg.active_profile}")
    if cfg.active_tier:
        lines.append(f"  active tier: {cfg.active_tier}")

    shown: set[str] = set()
    for key in _CORE_KEYS:
        if key not in cfg.sources and not hasattr(cfg, key):
            continue
        shown.add(key)
        value = getattr(cfg, key, "")
        if key in _SECRET_KEYS:
            value = "(set)" if value else "(empty)"
        source = cfg.sources.get(key, "default")
        lines.append(f"  {key} = {value!r}   [{source}]")

    extras = [
        key for key in sorted(cfg.sources)
        if key not in shown and cfg.sources.get(key, "default") != "default"
        and key not in _SECRET_KEYS
    ]
    for key in extras[:24]:
        lines.append(f"  {key} = {getattr(cfg, key, '')!r}   "
                     f"[{cfg.sources[key]}]")
    return "\n".join(lines)


QUESTION_SYSTEM = """\
You are Pedacito's built-in help assistant. Pedacito is a Python command-line
tool that lets a small local LLM work on a codebase through a lookup index.
Your only job is to explain how to install, configure, and use Pedacito.

Use the REFERENCE MATERIAL and CURRENT CONFIG below as the source of truth.
They come from Pedacito's README, QUICKSTART, the modular guides in docs/
when installed, and source code. Use the code excerpts to confirm exact
command names, flags, defaults, and behaviors.
Do not invent commands, flags, paths, defaults, or settings that are not
supported there.

How to answer:
- Start with a direct, plain-language answer in one or two sentences.
- If the user asks how to do something, give the exact command(s) to run in a
  fenced code block, using `pedacito` (or `pdc`, its alias) as documented.
  Use a clear placeholder such as PROJECT when a project path is needed, and
  say what each command does in one line.
- If the user asks for a config file or a setting, give a minimal TOML snippet
  with only the needed keys, then briefly explain each key.
- If the user asks about "my setup" or their current config, use the CURRENT
  CONFIG values and mention the relevant file/config layer.
- Use short numbered steps when there is more than one action.
- Keep it beginner-friendly and compact. Prefer working examples over long
  explanations.
- If the references do not contain the answer, say so plainly and point to the
  closest documented command (for example `pedacito config --check`,
  `pedacito files PROJECT`, or `pedacito config --reference`). Never guess.
"""


MODEL_SYNTH_SYSTEM = """\
You are Pedacito's built-in help assistant. Answer ONLY from the STRUCTURED
REFERENCE below. It is the complete, authoritative list of the CLI commands,
exact GUI labels/paths, and config settings available for this question.

Write a compact answer with exactly these three headings:
**In the GUI**
**In the Terminal (CLI)**
**Relevant Settings (`config.toml`)**

Rules:
- Keep the exact GUI labels and arrow paths from the reference (for example
  "⚙ Settings → Profiles → [Select Profile] → Reasoning & Execution
  Overrides → disable_thinking"). Never invent a menu item or a path.
- Put copy-pasteable terminal commands in fenced ```sh blocks, using
  `pedacito` (or `pdc`) exactly as shown. Never invent a flag or command.
- Put config keys in backticks and show a minimal TOML snippet only when the
  setting is relevant. State the default when it helps.
- If a section has no useful reference item, write exactly:
  "No specific GUI control / CLI command / setting is needed."
- Do not mention these instructions or that reference items were retrieved.
- Prefer two to six short bullets or numbered steps per section.
"""


def _server_models(cfg: Config, timeout: float = 1.5) -> list[str] | None:
    """The model ids a server reports, or None when it is not reachable fast."""
    import requests
    url = cfg.base_url.rstrip("/") + "/models"
    headers = {}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return [str(m.get("id", "")) for m in r.json().get("data", [])]
    except Exception:
        return None


def _model_synthesis(question: str, matches: dict, cfg: Config,
                     client=None) -> tuple[str, str]:
    """Try a fast, grounded model rewrite. Returns (answer, error)."""
    if client is None:
        ids = _server_models(cfg)
        if ids is None:
            return "", f"no model server answered at {cfg.base_url}"
        if ids and cfg.model not in ids:
            return "", (f"configured model {cfg.model!r} is not in the "
                        f"server's list")
        try:
            fast = replace(
                cfg,
                timeout=min(float(getattr(cfg, "timeout", 20.0) or 20.0), 20.0),
                load_wait_seconds=min(
                    int(getattr(cfg, "load_wait_seconds", 5) or 5), 5),
            )
        except (TypeError, ValueError):
            fast = cfg
        client = LLMClient(fast)
    system = MODEL_SYNTH_SYSTEM
    if prompts.effective_reply_language(cfg) == "es":
        system += "\n\n" + prompts.reply_language_directive(cfg)
    user = (f"QUESTION\n{question}\n\n"
            f"STRUCTURED REFERENCE\n{how_index.model_context(matches)}")
    try:
        text = client.chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            max_tokens=min(max(256, int(cfg.max_tokens_answer)),
                           MAX_HELP_ANSWER_TOKENS),
            stream=False,
            disable_thinking=True,
        ).strip()
    except LLMError as e:
        return "", str(e)
    except Exception as e:                    # a help answer must never crash
        return "", f"{type(e).__name__}: {e}"
    if not text:
        return "", "the model returned an empty help answer"
    return text, ""


def answer_structured(question: str, cfg: Config | None = None,
                      use_model: bool = False, client=None,
                      lang: str | None = None) -> dict:
    """Retrieve CLI/GUI/setting facts and answer from them.

    `use_model=False` is the deterministic, instant path. With a reachable,
    fast model the answer is a grounded rewrite; any failure falls back to the
    formatted structured cards instead of surfacing an error. `lang` (default:
    the `--lang`/config/env language) localizes the deterministic cards.
    """
    question = " ".join((question or "").split())
    if not question:
        raise HowError("No question given.")
    lang = i18n.normalize_lang(lang or i18n.detect_lang(cfg=cfg))
    matches = how_index.search(question)
    pinned = _pin_named_setting(question, matches)
    if lang == "es":
        how_index.localize_matches(matches, lang, spanish_setting_help())
    reference = (render_setting_answer(question, pinned, matches, lang=lang)
                 if pinned
                 else how_index.render_answer(question, matches, lang=lang))
    result = {
        "ok": True,
        "query": question,
        "lang": lang,
        "mode": "deterministic",
        "used_model": False,
        "model_error": "",
        "answer": reference,
        "reference_answer": reference,
        "setting": pinned,
        "matches": matches,
        "common_flags": how_index.COMMON_FLAGS,
        "topics": how_index.topics(),
    }
    if not use_model:
        return result
    answer, error = _model_synthesis(question, matches, cfg or Config(),
                                     client=client)
    if answer:
        low = answer.lower()
        required = ("in the gui", "in the terminal", "relevant settings")
        if all(mark in low for mark in required):
            result.update(mode="model", used_model=True, answer=answer)
        else:
            result["model_error"] = (
                "the model answer did not follow the required three-section "
                "format")
    else:
        result["model_error"] = error
    return result


def build_reference_index() -> dict:
    """The full structured CLI/GUI/setting index used by `how`."""
    return how_index.build_reference_index()


def _history_messages(history: list[tuple[str, str]]) -> list[dict]:
    out: list[dict] = []
    for question, answer in history[-2:]:
        out.append({"role": "user", "content": f"Earlier question: {question}"})
        out.append({"role": "assistant", "content": answer[:1200]})
    return out


class HowSession:
    """One help conversation: corpus built once, config/model reused."""

    def __init__(self, cfg: Config, client: LLMClient,
                 docs_root: Path | None = None):
        self.cfg = cfg
        self.client = client
        self.docs_root = docs_root
        self.chunks: list[Chunk] | None = None
        self.history: list[tuple[str, str]] = []

    def _corpus(self) -> list[Chunk]:
        if self.chunks is None:
            self.chunks = build_corpus(
                self.docs_root, i18n.detect_lang(cfg=self.cfg))
        return self.chunks

    def answer(self, question: str, verbose: bool = True) -> str:
        """Answer one question from the docs/source corpus."""
        question = " ".join(question.split())
        if not question:
            raise HowError("No question given.")
        snapshot = config_snapshot(self.cfg)
        context, picked = select_references(
            question, self._corpus(), self.cfg, snapshot, self.history)
        if verbose:
            # Mention docs/ only when an excerpt actually came from it, so the
            # footer stays true for older installs without the modular guides.
            sources = ("README, QUICKSTART, docs/, and source"
                       if any(c.file.startswith(DOCS_DIR + "/") for c in picked)
                       else "README, QUICKSTART, and source")
            print(f"  [help references: {len(picked)} excerpt(s), "
                  f"~{estimate_tokens(context):,} tokens from {sources}]",
                  file=sys.stderr)

        user = (
            f"QUESTION\n{question}\n\n"
            f"CURRENT CONFIG\n{snapshot}\n\n"
            f"REFERENCE MATERIAL\n{context or '(no matching excerpts)'}"
        )
        system = QUESTION_SYSTEM
        if prompts.effective_reply_language(self.cfg) == "es":
            system += "\n\n" + prompts.reply_language_directive(self.cfg)
        messages = [{"role": "system", "content": system}]
        messages += _history_messages(self.history)
        messages.append({"role": "user", "content": user})
        answer = self.client.chat(
            messages,
            max_tokens=min(max(256, self.cfg.max_tokens_answer),
                           MAX_HELP_ANSWER_TOKENS),
            stream=False,
        ).strip()
        if not answer:
            raise LLMError("The model returned an empty help answer.")
        self.history.append((question, answer))
        return answer


def config_from_args(args, quiet: bool = False) -> Config:
    """Build the `how` config: normal layers, then direct flags. VRAM is optional
    here; without it the built-in token budgets are kept."""
    root = Path.cwd()
    cfg = load_config(root=root,
                      explicit=getattr(args, "config", None),
                      profile=getattr(args, "profile", None),
                      require_hardware=False,
                      warn_budgets=False)

    if getattr(args, "url", None):
        cfg.base_url = args.url
        cfg.sources["base_url"] = "--url/--server"
    if getattr(args, "api_key", None) is not None:
        cfg.api_key = args.api_key
        cfg.sources["api_key"] = "--api-key"
    if getattr(args, "model", None):
        cfg.model = args.model
        cfg.sources["model"] = "--model"
    if getattr(args, "vram_gb", None) is not None:
        vram = int(args.vram_gb)
        if vram <= 0:
            raise ConfigError("--vram must be a positive whole number of GB")
        cfg.vram_gb = vram
        cfg.sources["vram_gb"] = "--vram"

    if missing_hardware(cfg):
        if not quiet:
            print("[how] vram_gb is not set; using built-in token budgets. "
                  "Pass --vram GB or add vram_gb to your config to size them "
                  "to your GPU.", file=sys.stderr)
    elif not quiet:
        # Offline help must not contact a model server just to fit budgets.
        auto_fit(cfg, verbose=False)
    return cfg


def _read_question(args) -> str:
    positional = " ".join(getattr(args, "question", []) or []).strip()
    flagged = (getattr(args, "question_opt", None) or "").strip()
    return flagged or positional


def run_cli(args) -> int:
    """Entry point used by cli.main for `pedacito how`.

    The structured reference answers instantly. A reachable model server may
    optionally synthesize the same retrieved facts; with --offline (or no
    usable config/server) the deterministic cards are printed directly.
    """
    offline = bool(getattr(args, "offline", False)
                   or getattr(args, "no_model", False)
                   or getattr(args, "fast", False))
    try:
        cfg = config_from_args(args, quiet=offline)
    except ConfigError as e:
        # Help must work even while the config itself is broken.
        print(f"[how] ignoring the broken config ({e}); using built-in "
              "defaults for this help session.", file=sys.stderr)
        cfg = Config()
        offline = True

    as_json = bool(getattr(args, "json", False))

    lang = i18n.detect_lang(getattr(args, "lang", None), cfg=cfg)

    def respond(question: str) -> None:
        result = answer_structured(question, cfg=cfg, use_model=not offline,
                                   lang=lang)
        if result.get("model_error") and not offline:
            print(f"[how] model synthesis skipped: {result['model_error']}; "
                  "showing the built-in reference.", file=sys.stderr)
        if as_json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(result["answer"])
            # BUG-18: an informational tag, not an error -- keep it on stdout
            # so Windows PowerShell 5.1 does not paint it as a red error.
            print(f"[how: {'model synthesis' if result['used_model'] else 'deterministic reference'}]")

    question = _read_question(args)
    if question:
        respond(question)
    elif not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if not piped:
            print("Usage: pedacito how \"your question\"", file=sys.stderr)
            return 2
        respond(piped)
    else:
        state = {"offline": offline}
        print("Ask how to use Pedacito. Type /quit to leave.",
              file=sys.stderr)
        print("  /offline toggles deterministic answers, /model re-enables "
              "model synthesis.", file=sys.stderr)
        while True:
            try:
                line = input("\nhow> ").strip()
            except (EOFError, KeyboardInterrupt):
                print(file=sys.stderr)
                break
            if not line:
                continue
            low = line.lower()
            if low in {"/quit", "/exit", "quit", "exit"}:
                break
            if low == "/offline":
                state["offline"] = not state["offline"]
                print("  offline mode " + ("ON" if state["offline"] else "OFF"),
                      file=sys.stderr)
                continue
            if low == "/model":
                state["offline"] = False
                print("  model synthesis ON", file=sys.stderr)
                continue
            if low in {"/help", "help", "?"}:
                print("  Ask about any CLI command/flag, GUI path, or setting."
                      "\n  Examples: \"how do I disable thinking?\", "
                      "\"how do I import models?\",\n"
                      "  \"how do I reset my config?\", \"where are "
                      "backups?\"", file=sys.stderr)
                continue
            offline = state["offline"]
            print()
            respond(line)

    return 0
