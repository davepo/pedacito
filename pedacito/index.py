"""Index construction, persistence, and tiered rendering.

    idx = build(["src", "README.md"], cfg, client=llm_client, root=project_root)
    text, tier = idx.map_for_prompt(cfg)         # what goes in every prompt
    idx = reindex_file(idx, "src/app.py", cfg)   # after an edit

An Index has a static layer from the chunkers (chunker.py, chunker_ts.py,
references.py: exact, free) and a semantic layer of one-line LLM summaries
(slow, cached, rationed).

The map goes into every prompt, so rendering is tiered: the flat map (every
unit with its summary) when it fits the token budget, else one card per file,
else one card per directory. The model descends with `outline` and `read`.

Summarisation is rationed: only undocumented top-level units above a length
threshold are summarised (methods are covered by their class summary plus
signature), which cut a 570-unit stdlib sample from 442 calls to 76.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from . import fileio
from .chunker import Chunk, chunk_source
from .chunker_ts import chunk_file_ts, LANG_BY_EXT, TS_AVAILABLE
from .llm import LLMError, ModelLoadError, estimate_tokens
from .references import chunk_reference
from .select import CODE_EXTS, PY_EXTS, REFERENCE_EXTS, Matcher, select

INDEX_DIR = ".pedacito"
CACHE_FILE = "summaries.json"
INDEX_FILE = "index.json"
BACKUP_DIR = "backups"
SESSION_DIR = "sessions"


def file_hash(path: Path) -> str:
    """Short content hash of a file's raw bytes ("" if unreadable)."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def file_stat(path: Path) -> "list[int] | None":
    """[mtime_ns, size] of a file, or None when it cannot be stat-ed. The
    cheap freshness prefilter for `Index.stale()`: only a stat mismatch pays
    for a content hash."""
    try:
        st = path.stat()
        return [st.st_mtime_ns, st.st_size]
    except OSError:
        return None


def state_dir(root: Path, *parts: str) -> Path:
    """Path under <root>/.pedacito/ with optional subpath components."""
    d = Path(root) / INDEX_DIR
    for p in parts:
        d = d / p
    return d


def infer_root(paths: list[str]) -> Path:
    """Project root when --root wasn't given: the common ancestor of the
    given paths (so `pedacito ask ~/code/anki` doesn't use the cwd)."""
    resolved = []
    for p in paths:
        q = Path(p).resolve()
        resolved.append(q if q.is_dir() else q.parent)
    if not resolved:
        return Path.cwd().resolve()
    if len(resolved) == 1:
        return resolved[0]
    try:
        return Path(os.path.commonpath([str(r) for r in resolved]))
    except ValueError:
        # Different Windows drives have no common path. Fall back to the first.
        return resolved[0]


TOP_KINDS = ("function", "class", "classheader", "module")

SUMMARY_SYSTEM = (
    "You summarise source code for a lookup index. Given one unit of code, reply with "
    "ONE sentence (two at most) describing what it does and any non-obvious assumption, "
    "side effect, or external dependency it has. Do not restate the signature. Do not "
    "list parameters. Do not use bullet points, markdown, or a preamble. Start directly "
    "with a verb."
)

FILE_SUMMARY_SYSTEM = (
    "You summarise a source file for a lookup index. You are given the module's "
    "skeleton: its imports and the signatures of everything it defines. Reply with TWO "
    "sentences: what this module is responsible for, and what another part of the system "
    "would come here to do. Do not list the functions back. No markdown, no preamble."
)


class Index:
    """A project's chunk collection plus derived summaries, with query and
    rendering methods for the agent's tool loop and prompt construction."""

    def __init__(self, root: Path, chunks: list[Chunk],
                 file_summaries: dict[str, str] | None = None,
                 file_hashes: dict[str, str] | None = None,
                 file_stats: dict[str, list] | None = None):
        """Build an Index in memory from a flat list of chunks. Prefer
        `build()` or `Index.load()`."""
        self.root = root
        self.chunks = chunks
        self.file_summaries = file_summaries or {}
        # content hash per file at index time: line ranges are only valid
        # against the bytes they were computed from
        self.file_hashes = file_hashes or {}
        # [mtime_ns, size] per file at index time (fast path for stale())
        self.file_stats = file_stats or {}
        # set by stale() when stat entries were refreshed without content
        # changes; callers persist the index so the next run skips the hashes
        self.stats_refreshed = False
        self._by_file: dict[str, list[Chunk]] = defaultdict(list)
        for c in chunks:
            self._by_file[c.file].append(c)
        for v in self._by_file.values():
            v.sort(key=lambda c: c.start)
        self._files_sorted: "list[str] | None" = None
        self._build_symbol_caches()

    def _build_symbol_caches(self) -> None:
        """Lookup dicts that keep find_symbol/callers_of off a linear scan."""
        by_qualname: dict[str, list[Chunk]] = {}
        by_suffix: dict[str, list[Chunk]] = {}
        by_call: dict[str, list[Chunk]] = {}
        for c in self.chunks:
            by_qualname.setdefault(c.qualname, []).append(c)
            # every dotted suffix of the lower-cased qualname
            parts = c.qualname.lower().split(".")
            for i in range(len(parts)):
                by_suffix.setdefault(".".join(parts[i:]), []).append(c)
            for name in c.calls:
                by_call.setdefault(name, []).append(c)
        self._by_qualname = by_qualname
        self._by_suffix = by_suffix
        self._by_call = by_call

    @property
    def files(self) -> list[str]:
        """All indexed file paths (relative to root), sorted. The cached list
        is shared: callers must not mutate it."""
        if self._files_sorted is None:
            self._files_sorted = sorted(self._by_file)
        return self._files_sorted

    def in_file(self, rel: str) -> list[Chunk]:
        """All chunks belonging to one file, in line order."""
        return self._by_file.get(rel, [])

    def top_level(self, rel: str) -> list[Chunk]:
        """A file's top-level definitions (never methods), as named on a file
        card."""
        return [c for c in self.in_file(rel)
                if c.kind in ("function", "class", "classheader")]

    def stale(self) -> tuple[list[str], list[str]]:
        """Return (changed, missing): files whose content hash no longer
        matches what was indexed, and files deleted since.

        A file whose recorded [mtime_ns, size] still matches is skipped
        without hashing. When only the stat differs (a touch) the file is
        hashed and, if unchanged, its stat entry is refreshed and
        `stats_refreshed` is set."""
        changed, missing = [], []
        self.stats_refreshed = False
        for rel in self.files:
            p = self.root / rel
            if not p.exists():
                missing.append(rel)
                continue
            stat_now = file_stat(p)
            if stat_now is not None and self.file_stats.get(rel) == stat_now:
                continue                    # untouched since indexing
            known = self.file_hashes.get(rel)
            if known and file_hash(p) != known:
                changed.append(rel)
            elif stat_now is not None and self.file_stats.get(rel) != stat_now:
                self.file_stats[rel] = stat_now
                self.stats_refreshed = True
        return changed, missing

    def forget(self, rel: str) -> None:
        """Remove a file from the index (chunks, summary, hash, stat)."""
        self.chunks = [c for c in self.chunks if c.file != rel]
        self._by_file.pop(rel, None)
        self.file_summaries.pop(rel, None)
        self.file_hashes.pop(rel, None)
        self.file_stats.pop(rel, None)
        self._files_sorted = None
        self._build_symbol_caches()

    def is_reference(self, rel: str) -> bool:
        """Whether a file was indexed structurally (reference file) rather
        than as parsed code."""
        return all(c.kind == "reference" for c in self.in_file(rel))

    @property
    def code_files(self) -> list[str]:
        """Indexed files that were parsed as code, sorted."""
        return [f for f in self.files if not self.is_reference(f)]

    @property
    def reference_files(self) -> list[str]:
        """Indexed files that are reference material, sorted."""
        return [f for f in self.files if self.is_reference(f)]

    def resolve_file(self, name: str) -> str | None:
        """Resolve a user- or model-typed file reference ('db.py',
        'src/db.py', 'src\\db.py', or the exact indexed path) to the exact
        path stored in the index, or None if no match."""
        n = fileio.normalise_rel(name)
        if not n:
            return None
        for f in self.files:
            if f == n:
                return f
        for f in self.files:
            if f.endswith("/" + n) or Path(f).name == n:
                return f
        return None

    def find_symbol(self, symbol: str) -> list[Chunk]:
        """Find chunk(s) matching a symbol name: exact qualname match first,
        then case-insensitive qualname/suffix match, then (for methods of a
        class that was indexed whole) the owning class. Lookups hit the
        prebuilt dicts from _build_symbol_caches."""
        s = symbol.strip()
        hits = self._by_qualname.get(s)
        if hits:
            return list(hits)
        # suffix map: every dotted suffix of every lower-cased qualname.
        # Small classes are indexed whole, so a bare method name won't match
        # a qualname; fall back to the class that declares it.
        hits = self._by_suffix.get(s.lower())
        if hits:
            return list(hits)
        base = s.split(".")[-1]
        return [c for c in self.chunks if base in c.members]

    def callers_of(self, symbol: str) -> list[Chunk]:
        """Chunks whose call list contains this symbol's bare name. An
        approximate cross-reference (names, not resolved bindings)."""
        base = symbol.split(".")[-1]
        return list(self._by_call.get(base, []))

    def importers_of(self, rel: str) -> list[str]:
        """Indexed files whose imports name the module behind `rel`. Matches
        module names, not resolved symbols, so it is evidence rather than
        proof; complements `callers_of` for calls through an alias."""
        base = rel.rsplit(".", 1)[0]
        if base.endswith("/__init__"):
            base = base[:-len("/__init__")]
        mod = base.replace("/", ".")
        last = mod.rsplit(".", 1)[-1]
        targets = {mod, last}
        out: list[str] = []
        for f in self.files:
            if f == rel:
                continue
            for c in self.in_file(f):
                for imp in c.imports:
                    imp = (imp or "").strip().lstrip(".")
                    if not imp:
                        continue
                    if imp in targets or imp.endswith("." + last) \
                            or imp.startswith(mod + ".") \
                            or imp.startswith(last + "."):
                        out.append(f)
                        break
                else:
                    continue
                break
        return sorted(set(out))

    def render_files(self, line_ranges: bool = True) -> str:
        """Tier-0 map: one card per file naming what it is, imports, and
        definitions (with line ranges so the model can go straight to `read`)."""
        return self.render_cards(self.code_files + self.reference_files,
                                 line_ranges)

    def _test_maps(self) -> tuple[dict, dict]:
        """Stem -> [rels] maps of module files and test files, for the cards'
        `tested by:` / `tests:` links."""
        mods: dict[str, list[str]] = {}
        tests: dict[str, list[str]] = {}
        for f in self.files:
            if not f.endswith(".py"):
                continue
            is_t, stem = _test_stem(f)
            if not stem:
                continue
            (tests if is_t else mods).setdefault(stem, []).append(f)
        return mods, tests

    @staticmethod
    def _test_link(rel: str, mods: dict,
                   tests: dict) -> tuple[str, str] | None:
        """('tested by', test paths) for a module, ('tests', module path) for
        a test file, when the stem match is unambiguous. Advisory map hint:
        it saves a grep for the obvious `foo.py <-> test_foo.py` convention."""
        if not rel.endswith(".py"):
            return None
        is_t, stem = _test_stem(rel)
        if not stem:
            return None
        if is_t:
            cands = [m for m in mods.get(stem, ()) if m != rel]
            return ("tests", cands[0]) if len(cands) == 1 else None
        cands = sorted(t for t in tests.get(stem, ()) if t != rel)
        return ("tested by", ", ".join(cands[:3])) if cands else None

    def render_cards(self, rels: list[str], line_ranges: bool = True,
                     max_names: int = 60) -> str:
        """File cards for any subset of files: the tier-0 map's body, and what
        `outline` renders for a directory."""
        out: list[str] = []
        mods_by_stem, tests_by_stem = self._test_maps()
        for f in rels:
            units = self.in_file(f)
            if self.is_reference(f):
                out.append(self._reference_card(f, units, line_ranges))
                continue
            top = self.top_level(f)
            nlines = max((c.end for c in units), default=0)
            imports = sorted({i.split(".")[0].lstrip(".") for c in units for i in c.imports})
            card = [f"### {f}  ({nlines} lines, {len(units)} units)"]
            if imports:
                card.append(f"  imports: {', '.join(imports[:14])}")
            summary = self.file_summaries.get(f)
            if summary:
                card.append(f"  {summary}")
            link = self._test_link(f, mods_by_stem, tests_by_stem)
            if link:
                card.append(f"  {link[0]}: {link[1]}")
            if top:
                shown, hidden = _cap(top, max_names)
                names = []
                for c in shown:
                    end = c.scope_end or c.end   # full class span, not the header
                    names.append(f"{c.qualname}:L{c.start}-{end}" if line_ranges
                                 else c.qualname)
                extra = f" (+{hidden} hidden; use outline)" if hidden else ""
                card.append(f"  defines: {', '.join(names)}{extra}")
            out.append("\n".join(card))
        return "\n\n".join(out)

    def render_dirs(self) -> str:
        """Directory-tier map: one card per top-level directory naming its
        files, for projects whose file cards don't fit the prompt budget. The
        model descends with `outline <dir>` (see tools.tool_outline)."""
        max_dirs, max_files, summary_chars = 80, 12, 100
        groups: dict[str, list[str]] = {}
        for f in self.files:
            top = f.split("/", 1)[0] + "/" if "/" in f else "(root)"
            groups.setdefault(top, []).append(f)
        out: list[str] = []
        names = sorted(groups)
        for d in names[:max_dirs]:
            rels = groups[d]
            card = [f"### {d}  ({len(rels)} files)"]
            for f in rels[:max_files]:
                summ = self.file_summaries.get(f) or ""
                if len(summ) > summary_chars:
                    summ = summ[:summary_chars - 1].rstrip() + "..."
                card.append(f"  {f}" + (f"  -- {summ}" if summ else ""))
            if len(rels) > max_files:
                where = ("`outline` '.' to list every file" if d == "(root)"
                         else "`outline` this directory")
                card.append(f"  (+{len(rels) - max_files} more files; {where})")
            out.append("\n".join(card))
        if len(names) > max_dirs:
            out.append(f"### ...  (+{len(names) - max_dirs} more directories; "
                       f"`grep` or `outline` one to descend)")
        return "\n\n".join(out)

    def _reference_card(self, f: str, units: list[Chunk], line_ranges: bool) -> str:
        """Tier-0 card for a non-code file: shape, not contents."""
        nlines = max((c.end for c in units), default=0)
        card = [f"### {f}  ({nlines} lines, reference)"]
        summary = self.file_summaries.get(f)
        if summary:
            card.append(f"  {summary}")
        for c in units[:20]:
            bit = f"  {c.signature}"
            if line_ranges and c.end:
                bit += f"  L{c.start}-{c.end}"
            if c.docline:
                bit += f"\n      {c.docline}"
            card.append(bit)
        if len(units) > 20:
            card.append(f"  (+{len(units) - 20} more sections; use outline)")
        return "\n".join(card)

    def render_outline(self, rel: str, header: bool = True) -> str:
        """Every unit in one file with signature, line range, and
        summary/docline: the `outline` tool's output and the flat map's body."""
        rows: list[str] = [f"### {rel}"] if header else []
        for c in self.in_file(rel):
            loc = f"L{c.start}-{c.end}"
            if c.kind == "module":
                label = "module preamble" if c.qualname == "module" else "top-level code"
                imps = ", ".join(c.imports[:12])
                head = f"  {loc:<12} {label}" + (f"  [imports: {imps}]" if imps else "")
            elif c.kind == "classheader":
                span = f"L{c.start}-{c.scope_end or c.end}"
                head = (f"  {span:<12} {c.signature}  "
                        f"(header {c.start}-{c.end}; methods below)")
            else:
                indent = "    " if c.kind == "method" else "  "
                head = f"{indent}{loc:<12} {c.signature}"
                if c.decorators:
                    head += "  " + " ".join(c.decorators[:2])
            text = c.summary or c.docline
            if text:
                head += f"\n{' ' * 16}-- {text}"
            if c.kind == "class" and c.members:
                head += f"\n{' ' * 16}   methods: {', '.join(c.members)}"
            rows.append(head)
        return "\n".join(rows)

    def render(self) -> str:
        """The flat map: every unit in every file, code before reference
        files. Only used when it fits the token budget (see `map_for_prompt`)."""
        out: list[str] = []
        for f in self.code_files + self.reference_files:
            out.append(f"### {f}")
            out.append(self.render_outline(f, header=False))
            out.append("")
        return "\n".join(out)

    def map_for_prompt(self, cfg) -> tuple[str, str]:
        """Render the map at the richest tier that fits: flat within
        cfg.flat_map_token_budget, else file cards (up to twice the budget),
        else directory cards. Returns (map_text, tier_name)."""
        flat = self.render()
        if estimate_tokens(flat) <= cfg.flat_map_token_budget:
            return flat, "flat"
        cards = self.render_files(line_ranges=cfg.file_card_line_ranges)
        if estimate_tokens(cards) <= cfg.flat_map_token_budget * 2:
            return cards, "cards"
        return self.render_dirs(), "dirs"

    def save(self) -> Path:
        """Write the index to .pedacito/index.json under the project root."""
        d = self.root / INDEX_DIR
        d.mkdir(exist_ok=True)
        payload = {
            "root": str(self.root),
            "file_summaries": self.file_summaries,
            "file_hashes": self.file_hashes,
            "file_stats": self.file_stats,
            "chunks": [c.to_record() for c in self.chunks],
        }
        p = d / INDEX_FILE
        p.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        return p

    @classmethod
    def load(cls, root: Path) -> "Index | None":
        """Load .pedacito/index.json under `root`, re-reading each chunk's
        source from disk (reference chunks keep their small stored preview).
        Returns None if no index has been saved yet."""
        p = root / INDEX_DIR / INDEX_FILE
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        chunks = [Chunk(**rec) for rec in data["chunks"]]
        cache: dict[str, list[str]] = {}
        for c in chunks:
            # rehydrating reference files could pull a huge CSV into memory
            if c.kind == "reference":
                continue
            if c.file not in cache:
                fp = root / c.file
                cache[c.file] = fileio.read_lines(fp) if fp.exists() else []
            lines = cache[c.file]
            if lines:
                c.source = "\n".join(lines[c.start - 1 : c.end])
        return cls(root, chunks, data.get("file_summaries", {}),
                   data.get("file_hashes", {}), data.get("file_stats", {}))


def _test_stem(rel: str) -> tuple[bool, str]:
    """(is_test, module stem) for a Python file: `test_*.py`, `*_test.py` and
    anything under a `tests/` directory are test files; the stem is the module
    name a test file's affix hides. Non-test files return (False, stem)."""
    p = Path(rel)
    stem = p.stem
    if stem.startswith("test_"):
        return True, stem[len("test_"):]
    if stem.endswith("_test"):
        return True, stem[:-len("_test")]
    if p.parent.name in ("tests", "test"):
        return True, stem
    return False, stem


def _cap(units: list[Chunk], max_names: int) -> tuple[list[Chunk], int]:
    """Fit a file's definitions into a card's name limit, dropping private
    (underscore-prefixed) helpers before public ones."""
    if len(units) <= max_names:
        return units, 0
    public = [c for c in units if not c.qualname.startswith("_")]
    if len(public) <= max_names:
        return public, len(units) - len(public)
    return public[:max_names], len(units) - max_names


def _entry(value) -> tuple[str, str]:
    """Read one summary-cache entry (legacy plain string or {"text", "model"}
    dict). Returns (text, model_id)."""
    if isinstance(value, dict):
        return value.get("text", ""), value.get("model", "?")
    return value or "", "?"


def _put(cache: dict, key: str, text: str, model: str) -> None:
    """Write one cache entry, recording which model produced the summary."""
    cache[key] = {"text": text, "model": model}


def cache_models(cache: dict) -> dict[str, int]:
    """Count cache entries by the model that produced them."""
    counts: dict[str, int] = {}
    for v in cache.values():
        _, m = _entry(v)
        counts[m] = counts.get(m, 0) + 1
    return counts


def _load_cache(root: Path) -> dict:
    """Load the persisted summary cache from .pedacito/summaries.json."""
    p = root / INDEX_DIR / CACHE_FILE
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(root: Path, cache: dict) -> None:
    """Persist the summary cache to .pedacito/summaries.json."""
    d = root / INDEX_DIR
    d.mkdir(exist_ok=True)
    (d / CACHE_FILE).write_text(json.dumps(cache, indent=1),
                                encoding="utf-8")


def needs_summary(c: Chunk, cfg) -> bool:
    """Whether a chunk is worth an LLM call: too short, a method covered by
    its class, or already documented are excluded by default."""
    if c.kind == "reference":
        return False   # shape is already extracted; prose adds nothing
    if c.nlines < cfg.min_lines_to_summarise:
        return False
    if c.kind not in TOP_KINDS and not cfg.summarise_methods:
        return False
    if c.docline and not cfg.summarise_documented:
        return False
    return True


def summary_plan(idx: "Index", cfg) -> dict:
    """Counts of eligible/skipped/to-summarise chunks and files, without
    calling the model. Used by `pedacito cost`."""
    eligible = [c for c in idx.chunks if c.nlines >= cfg.min_lines_to_summarise]
    todo = [c for c in idx.chunks if needs_summary(c, cfg)]
    by_doc = sum(1 for c in eligible if c.docline and not cfg.summarise_documented)
    by_kind = sum(1 for c in eligible
                  if c.kind not in TOP_KINDS and not cfg.summarise_methods)
    files = idx.files if cfg.summarise_references else idx.code_files
    return {
        "units": len(idx.chunks),
        "eligible": len(eligible),
        "skipped_documented": by_doc,
        "skipped_methods": by_kind,
        "skipped_references": sum(1 for c in idx.chunks if c.kind == "reference"),
        "unit_calls": len(todo),
        "file_calls": len(files),
    }


def _skeleton(idx: Index, rel: str, limit: int = 4000) -> str:
    """Compact stand-in for a file when summarising it: imports plus every
    top-level signature (with docstring first lines and class members)."""
    units = idx.in_file(rel)
    if idx.is_reference(rel):
        rows = [f"file: {rel}"] + [f"  {c.signature} {c.docline}".rstrip() for c in units[:40]]
        return "\n".join(rows)[:limit]
    imports = sorted({i for c in units for i in c.imports})
    lines = [f"module: {rel}", f"imports: {', '.join(imports[:40])}"]
    for c in units:
        if c.kind in ("function", "class", "classheader"):
            line = f"  {c.signature or c.qualname}"
            if c.docline:
                line += f"  # {c.docline[:100]}"
            if c.members:
                line += f"  [methods: {', '.join(c.members[:20])}]"
            lines.append(line)
    return "\n".join(lines)[:limit]


class _Breaker:
    """Aborts an indexing batch after repeated consecutive failures (a model
    that cannot load fails identically for every chunk)."""

    def __init__(self, limit: int):
        """Track failures against `limit` consecutive occurrences."""
        self.limit = limit
        self.consecutive = 0
        self.total = 0

    def ok(self) -> None:
        """Reset the consecutive-failure counter after a successful call."""
        self.consecutive = 0

    def fail(self, label: str, err: Exception) -> None:
        """Record one failure; raises SystemExit if the batch should stop
        (a model-load error, or `limit` consecutive failures reached)."""
        self.consecutive += 1
        self.total += 1
        print(f"[warn] summary failed for {label}: {err}", file=sys.stderr)
        if isinstance(err, ModelLoadError) or self.consecutive >= self.limit:
            raise SystemExit(
                f"\nStopping: {self.consecutive} summaries in a row failed.\n"
                f"{err}\n\n"
                "Progress so far is cached, so fixing the problem and re-running\n"
                "`pedacito index` will resume rather than start over."
            )


def _ask(client, cfg, system: str, prompt: str, label: str,
         breaker: _Breaker) -> str:
    """Request one summary from the model, truncated to 400 chars. Failures
    go to `breaker` (which may raise SystemExit); returns "" on failure."""
    try:
        s = client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_tokens=cfg.max_tokens_summary,
        ).strip().replace("\n", " ")
    except LLMError as e:
        breaker.fail(label, e)
        return ""
    breaker.ok()
    if len(s) > 400:
        s = s[:400].rsplit(" ", 1)[0] + "..."
    return s


def _unique_qualnames(chunks: list[Chunk]) -> list[Chunk]:
    """Make every chunk's (file, qualname) key unique by suffixing later
    duplicates (`vram#2`): property getter/setter pairs, TS overloads, and
    conditional redefinitions share a name but are all real bodies."""
    used: set[str] = set()
    for c in chunks:
        base = c.qualname
        name = base
        n = 2
        while name in used:
            name = f"{base}#{n}"
            n += 1
        if name != base:
            c.qualname = name
        used.add(name)
    return chunks


def _chunk_file(rel: str, path: Path, cfg) -> list[Chunk]:
    """Chunk one file by extension: AST for Python, tree-sitter for the
    languages in LANG_BY_EXT (when importable), else structural
    reference-chunking (also the fallback when tree-sitter is missing, so the
    file stays visible to `read`/`grep`)."""
    suffix = path.suffix.lower()
    if suffix in PY_EXTS:
        src = fileio.read_text(path)
        return _unique_qualnames(chunk_source(
            rel, src, cfg.class_split_lines,
            cfg.max_unit_lines, cfg.max_unit_chars,
            cfg.max_unit_tokens, cfg.unit_overlap_lines))
    if suffix in LANG_BY_EXT and TS_AVAILABLE:
        src = fileio.read_text(path)
        return _unique_qualnames(chunk_file_ts(
            rel, src, suffix, cfg.class_split_lines,
            cfg.max_unit_lines, cfg.max_unit_chars,
            cfg.max_unit_tokens, cfg.unit_overlap_lines))
    return chunk_reference(rel, path, cfg.max_reference_bytes)


def build(paths: list[str], cfg, client=None, root: Path | None = None,
          summarise: bool = True, verbose: bool = True,
          matcher: Matcher | None = None, resummarise: bool = False) -> Index:
    """Build a full Index from scratch: select files, chunk each, and
    optionally generate LLM summaries (file summaries first, then rationed
    unit summaries) through the on-disk summary cache. Saves to
    .pedacito/index.json before returning.

    With `client` None or `summarise` False only the static index is built
    (`pedacito map`/`files`/`cost`), still reusing cached summaries.
    """
    root = (root or Path.cwd()).resolve()
    matcher = matcher if matcher is not None else Matcher.from_files(root)

    exts = set(CODE_EXTS)
    if cfg.include_references:
        exts |= set(REFERENCE_EXTS)
    exts |= {e if e.startswith(".") else "." + e for e in cfg.extra_extensions}

    files = select(paths, root, matcher, exts)
    if not files:
        raise SystemExit(
            "Nothing to index in: " + ", ".join(paths) +
            "\nEverything matched an ignore rule, or no files had an indexed "
            "extension. Try `pedacito files` to see what was considered.")

    chunks: list[Chunk] = []
    rels: list[str] = []
    for f in files:
        try:
            rel = f.resolve().relative_to(root).as_posix()
        except ValueError:
            rel = f.name
        rels.append(rel)
        try:
            chunks.extend(_chunk_file(rel, f, cfg))
        except SyntaxError as e:
            print(f"[skip] {rel}: syntax error line {e.lineno}", file=sys.stderr)

    idx = Index(root, chunks)
    for rel, f in zip(rels, files):
        idx.file_hashes[rel] = file_hash(f)
        st = file_stat(f)
        if st is not None:
            idx.file_stats[rel] = st

    # apply cached summaries even when summarise=False, so static rebuilds
    # don't write an index.json with blank summaries
    cache = {} if resummarise else _load_cache(root)
    for c in chunks:
        if c.sha in cache:
            c.summary = _entry(cache[c.sha])[0]
    skeletons: dict[str, tuple[str, str]] = {}   # rel -> (skeleton, cache key)
    for rel in idx.files:
        # same skip condition as the summarise loop below
        if idx.is_reference(rel) and not cfg.summarise_references:
            continue
        skel = _skeleton(idx, rel)
        key = "file:" + hashlib.sha256(skel.encode()).hexdigest()[:16]
        skeletons[rel] = (skel, key)
        if key in cache:
            idx.file_summaries[rel] = _entry(cache[key])[0]

    if not (summarise and client is not None):
        idx.save()
        return idx

    if verbose and cache:
        counts = cache_models(cache)
        foreign = {m: n for m, n in counts.items() if m not in ("?", cfg.model)}
        if foreign:
            listed = ", ".join(f"{n} from {m}" for m, n in sorted(foreign.items()))
            print(f"  reusing cached summaries ({listed}); "
                  f"--resummarise to regenerate with {cfg.model}", file=sys.stderr)

    breaker = _Breaker(cfg.max_consecutive_failures)

    # prove the model can generate before committing to a long batch
    if verbose:
        print("  checking the model responds...", file=sys.stderr)
    try:
        client.warmup()
    except LLMError as e:
        _save_cache(root, cache)
        raise SystemExit(f"\n{e}\n\nNothing was indexed.")

    # file summaries first: they carry the tier-0 map
    for rel in idx.files:
        if rel not in skeletons or rel in idx.file_summaries:
            continue   # skipped reference file, or already loaded from the cache
        skel, key = skeletons[rel]
        if verbose:
            print(f"  [file] {rel}", file=sys.stderr)
        s = _ask(client, cfg, FILE_SUMMARY_SYSTEM, skel, rel, breaker)
        if s:
            idx.file_summaries[rel] = s
            _put(cache, key, s, cfg.model)
    _save_cache(root, cache)

    # unit summaries, rationed
    todo = [c for c in chunks if not c.summary and needs_summary(c, cfg)]
    if verbose:
        plan = summary_plan(idx, cfg)
        print(f"  {len(todo)} units to summarise "
              f"({plan['skipped_documented']} have docstrings, "
              f"{plan['skipped_methods']} are methods covered by their class)",
              file=sys.stderr)

    for i, c in enumerate(todo, 1):
        if verbose:
            print(f"  [{i}/{len(todo)}] {c.key}", file=sys.stderr)
        prompt = (f"File: {c.file}\nUnit: {c.qualname} ({c.kind}), "
                  f"lines {c.start}-{c.end}\n\n"
                  f"```python\n{c.source[:cfg.max_chunk_chars]}\n```")
        try:
            s = _ask(client, cfg, SUMMARY_SYSTEM, prompt, c.key, breaker)
        except SystemExit:
            _save_cache(root, cache)   # keep what we already paid for
            idx.save()
            raise
        if s:
            c.summary = s
            _put(cache, c.sha, s, cfg.model)
        if i % 10 == 0:
            _save_cache(root, cache)
    _save_cache(root, cache)

    idx.save()
    return idx


def reindex_file(index: Index, rel: str, cfg, save: bool = True) -> Index:
    """Re-chunk one changed file and return an updated Index. Summaries
    carry over by content hash, so only changed units lose theirs. Pass
    save=False when batching several reindexes and saving once."""
    root = index.root
    kept = [c for c in index.chunks if c.file != rel]
    try:
        new = _chunk_file(rel, root / rel, cfg)
    except (SyntaxError, OSError) as e:
        print(f"[warn] {rel} could not be re-indexed ({e}); index left stale",
              file=sys.stderr)
        return index
    cache = _load_cache(root)
    for c in new:
        c.summary = _entry(cache[c.sha])[0] if c.sha in cache else ""
    merged = Index(root, kept + new, dict(index.file_summaries),
                   dict(index.file_hashes), dict(index.file_stats))
    merged.file_hashes[rel] = file_hash(root / rel)
    st = file_stat(root / rel)
    if st is not None:
        merged.file_stats[rel] = st
    else:
        merged.file_stats.pop(rel, None)
    if save:
        merged.save()
    return merged