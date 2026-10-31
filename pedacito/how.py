"""`pedacito how`: built-in help answered from Pedacito's own docs and source.

Usage
-----
    pedacito how "how do I set up a config file?"
    pedacito how --url http://box:1234/v1 --model qwen3-27b --vram 24 \
                 "what command indexes a project?"
    pedacito how                                  # interactive help prompt

Unlike every other Pedacito command, `how` does not need a project or an
index. It builds a small in-memory corpus from README.md, QUICKSTART.md and
Pedacito's Python source, finds the passages relevant to the question, and
asks the configured model to answer from those passages only. The current
resolved config is included (with secrets hidden) so questions like "what
should I run with my setup?" can be answered against the real settings.
"""

from __future__ import annotations

import ast
import math
import os
import re
import site
import sys
import sysconfig
from collections import Counter
from dataclasses import replace
from pathlib import Path

from . import config as config_mod
from .chunker import Chunk, chunk_source
from .config import (Config, ConfigError, auto_fit, find_project_config,
                     load as load_config, missing_hardware)
from .llm import LLMClient, LLMError, estimate_tokens
from .references import chunk_reference

DOC_FILES = ("README.md", "QUICKSTART.md")
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


class HowError(RuntimeError):
    """Raised when the help corpus cannot be located or built."""


def find_docs_root() -> Path:
    """Return the directory containing README.md and QUICKSTART.md: $PEDACITO_DOCS,
    the checkout around this package, the package directory, setuptools
    `data-files` locations, then the current directory."""
    here = Path(__file__).resolve().parent
    candidates: list[Path] = []
    env = os.environ.get("PEDACITO_DOCS")
    if env:
        candidates.append(Path(env).expanduser())
    candidates += [here.parent, here]
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


def build_corpus(docs_root: Path | None = None) -> list[Chunk]:
    """README, QUICKSTART, the config templates and the package's Python modules
    as searchable chunks."""
    root = docs_root or find_docs_root()
    chunks: list[Chunk] = []
    for name in DOC_FILES:
        path = root / name
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
    for raw in _WORD_RE.findall(text.lower()):
        raw = raw.strip(".-")
        if len(raw) >= 2 and raw not in _STOP:
            raws.append(raw)

    weights: dict[str, float] = {}
    for raw in raws:
        weights[raw] = 1.0
        if "_" in raw or "-" in raw:
            weights[raw] = 1.6          # identifier, not prose
    for raw in raws:
        for part in re.split(r"[_\-.]", raw):
            if len(part) >= 3 and part not in _STOP:
                weights.setdefault(part, 0.35)
        if raw.endswith("s") and len(raw) > 4:
            weights.setdefault(raw[:-1], 0.5)

    for term in list(weights):
        for synonym in _SYNONYMS.get(term, ()):
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
        if chunk.file in DOC_FILES:
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
    QUICKSTART section and the CLI preamble, then top-ranked source and docs
    until the budget is used."""
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

    # Best section from each documentation file.
    for name in DOC_FILES:
        candidates = [(s, c) for s, c in by_file.get(name, []) if c.source]
        add(candidates[0][1] if candidates else None)

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
They come from Pedacito's README, QUICKSTART, and source code. Use the code
excerpts to confirm exact command names, flags, defaults, and behaviors.
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
            self.chunks = build_corpus(self.docs_root)
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
            print(f"  [help references: {len(picked)} excerpt(s), "
                  f"~{estimate_tokens(context):,} tokens from README, "
                  f"QUICKSTART, and source]", file=sys.stderr)

        user = (
            f"QUESTION\n{question}\n\n"
            f"CURRENT CONFIG\n{snapshot}\n\n"
            f"REFERENCE MATERIAL\n{context or '(no matching excerpts)'}"
        )
        messages = [{"role": "system", "content": QUESTION_SYSTEM}]
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


def config_from_args(args) -> Config:
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
        print("[how] vram_gb is not set; using built-in token budgets. "
              "Pass --vram GB or add vram_gb to your config to size them "
              "to your GPU.", file=sys.stderr)
    else:
        auto_fit(cfg, verbose=False)
    return cfg


def _read_question(args) -> str:
    positional = " ".join(getattr(args, "question", []) or []).strip()
    flagged = (getattr(args, "question_opt", None) or "").strip()
    return flagged or positional


def run_cli(args) -> int:
    """Entry point used by cli.main for `pedacito how`."""
    cfg = config_from_args(args)
    client = LLMClient(cfg)
    client.check()
    client.warmup()
    session = HowSession(cfg, client)

    question = _read_question(args)
    if question:
        print(session.answer(question))
    elif not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if not piped:
            print("Usage: pedacito how \"your question\"", file=sys.stderr)
            return 2
        print(session.answer(piped))
    else:
        print("Ask how to use Pedacito. Type /quit to leave.",
              file=sys.stderr)
        while True:
            try:
                line = input("\nhow> ").strip()
            except (EOFError, KeyboardInterrupt):
                print(file=sys.stderr)
                break
            if not line:
                continue
            if line.lower() in {"/quit", "/exit", "quit", "exit"}:
                break
            if line.lower() in {"/help", "help", "?"}:
                print("  Ask any question about installing, configuring, or "
                      "using Pedacito.\n  Examples: \"how do I set up a "
                      "config?\", \"what command indexes a project?\",\n"
                      "  \"how do I undo an apply?\"", file=sys.stderr)
                continue
            print()
            print(session.answer(line))

    stats = client.stats
    print(f"\n[{stats['calls']} calls, {stats['completion_tokens']} tokens "
          f"out, {stats['seconds']:.0f}s]", file=sys.stderr)
    return 0
