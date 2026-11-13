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

import dataclasses
import hashlib
import json
import os
import re
import sys
import threading
from collections import defaultdict
from pathlib import Path

from . import fileio, testlinks
from .chunker import Chunk, chunk_source
from .chunker_ts import chunk_file_ts, LANG_BY_EXT, TS_AVAILABLE
from .imports import raw_imports
from .llm import LLMError, ModelLoadError, estimate_tokens
from .references import chunk_reference
from .select import CODE_EXTS, PY_EXTS, REFERENCE_EXTS, Matcher, select

INDEX_DIR = ".pedacito"
CACHE_FILE = "summaries.json"
INDEX_FILE = "index.json"
BACKUP_DIR = "backups"
SESSION_DIR = "sessions"
# Bumped when index.json's layout changes; a newer file than this build
# understands is rebuilt instead of mis-read.
INDEX_VERSION = 2
_CHUNK_FIELDS = {f.name for f in dataclasses.fields(Chunk)}


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
    if isinstance(paths, str):
        # A command whose positional is optional (`rpc`) hands one string;
        # iterating it as a sequence of characters resolved the root to `/`.
        paths = [paths]
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
    rendering methods for the agent's tool loop and prompt construction.

    Mutations (`replace_file`, `forget`) update the lookup tables, the BM25
    index and the generation counter incrementally; derived structures (graph,
    rendered maps) key off `generation`, so they are rebuilt only when the
    index really changed."""

    def __init__(self, root: Path, chunks: list[Chunk],
                 file_summaries: dict[str, str] | None = None,
                 file_hashes: dict[str, str] | None = None,
                 file_stats: dict[str, list] | None = None,
                 file_imports: dict[str, list[str]] | None = None):
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
        # full import specifiers per file (the graph's raw material); a file
        # missing here is read lazily by `imports_of`
        self.file_imports = file_imports or {}
        # set by stale() when stat entries were refreshed without content
        # changes; callers persist the index so the next run skips the hashes
        self.stats_refreshed = False
        # bumped by every mutation; derived caches key off it
        self.generation = 0
        # (file::qualname) of units whose summary predates their last edit
        self.stale_summaries: set[str] = set()
        self._by_file: dict[str, list[Chunk]] = defaultdict(list)
        for c in chunks:
            self._by_file[c.file].append(c)
        for v in self._by_file.values():
            v.sort(key=lambda c: c.start)
        self._files_sorted: "list[str] | None" = None
        self._bm25 = None
        self._graph = None
        self._graph_gen = -1
        self._map_cache: dict = {}
        self._build_symbol_caches()

    # ------------------------------------------------------------- lookups
    def _build_symbol_caches(self) -> None:
        """Lookup dicts that keep find_symbol/callers_of off a linear scan."""
        self._by_qualname: dict[str, list[Chunk]] = {}
        self._by_suffix: dict[str, list[Chunk]] = {}
        self._by_call: dict[str, list[Chunk]] = {}
        self._cache_chunks(self.chunks)

    def _cache_chunks(self, chunks) -> None:
        for c in chunks:
            self._by_qualname.setdefault(c.qualname, []).append(c)
            # every dotted suffix of the lower-cased qualname
            parts = c.qualname.lower().split(".")
            for i in range(len(parts)):
                self._by_suffix.setdefault(".".join(parts[i:]), []).append(c)
            for name in c.calls:
                self._by_call.setdefault(name, []).append(c)

    def _uncache_chunks(self, chunks) -> None:
        dead = {id(c) for c in chunks}
        touched: list[tuple[dict, str]] = []
        seen: set[tuple[int, str]] = set()

        def touch(table: dict, key: str) -> None:
            if (id(table), key) not in seen:
                seen.add((id(table), key))
                touched.append((table, key))

        for c in chunks:
            touch(self._by_qualname, c.qualname)
            parts = c.qualname.lower().split(".")
            for i in range(len(parts)):
                touch(self._by_suffix, ".".join(parts[i:]))
            for name in c.calls:
                touch(self._by_call, name)
        for table, key in touched:
            lst = table.get(key)
            if lst is None:
                continue
            lst[:] = [x for x in lst if id(x) not in dead]
            if not lst:
                del table[key]

    @property
    def files(self) -> list[str]:
        """All indexed file paths (relative to root), sorted. The cached list
        is shared: callers must not mutate it."""
        if self._files_sorted is None:
            self._files_sorted = sorted(f for f, v in self._by_file.items() if v)
        return self._files_sorted

    def in_file(self, rel: str) -> list[Chunk]:
        """All chunks belonging to one file, in line order."""
        return self._by_file.get(rel, [])

    def top_level(self, rel: str) -> list[Chunk]:
        """A file's top-level definitions (never methods), as named on a file
        card."""
        return [c for c in self.in_file(rel)
                if c.kind in ("function", "class", "classheader")]

    # ----------------------------------------------------------- mutation
    def replace_file(self, rel: str, new_chunks: list[Chunk],
                     imports: list[str] | None = None) -> None:
        """Swap one file's chunks for `new_chunks` in place (O(that file),
        plus one pass over the chunk list), keeping every lookup table, the
        BM25 index and the generation counter in step."""
        old = self._by_file.pop(rel, [])
        if old:
            self._uncache_chunks(old)
            dead = {id(c) for c in old}
            self.chunks[:] = [c for c in self.chunks if id(c) not in dead]
        if self._bm25 is not None:
            self._bm25.remove_file(rel)
        if new_chunks:
            self.chunks.extend(new_chunks)
            self._by_file[rel] = sorted(new_chunks, key=lambda c: c.start)
            self._cache_chunks(new_chunks)
            if self._bm25 is not None:
                self._bm25.add(new_chunks)
        if imports is not None:
            self.file_imports[rel] = imports
        elif not new_chunks:
            self.file_imports.pop(rel, None)
        self._files_sorted = None
        self.generation += 1

    def forget(self, rel: str) -> None:
        """Remove a file from the index (chunks, summary, hash, stat)."""
        self.replace_file(rel, [])
        self.file_summaries.pop(rel, None)
        self.file_hashes.pop(rel, None)
        self.file_stats.pop(rel, None)

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
        if n in self._by_file and self._by_file[n]:
            return n
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
        approximate cross-reference (names, not resolved bindings).

        A qualified query (`Class.method`) ranks the chunks that recorded the
        exact qualified entry first -- the ones where the receiver IS that
        class (self/constructor/annotation/static evidence at chunk time) --
        then the bare-name matches that carry no qualification either way.
        Sites explicitly qualified to a DIFFERENT class still appear in the
        bare list (the chunker keeps the bare name alongside, so unqualified
        `d.load()` stays findable); `tools.callers` drops the different-class
        ones after reading the actual call line."""
        s = symbol.strip()
        base = s.split(".")[-1]
        bare = list(self._by_call.get(base, []))
        if "." not in s:
            return bare
        exact = list(self._by_call.get(s, []))
        seen = {id(c) for c in exact}
        return exact + [c for c in bare if id(c) not in seen]

    # ------------------------------------------------- derived structures
    def imports_of(self, rel: str) -> list[str]:
        """Full import specifiers of one file (lazy for older indexes)."""
        got = self.file_imports.get(rel)
        if got is None:
            try:
                got = raw_imports(rel, fileio.read_text(self.root / rel))
            except OSError:
                got = []
            self.file_imports[rel] = got
        return got

    @property
    def bm25(self):
        """BM25 over the units, built on first use and kept in step."""
        if self._bm25 is None:
            from .search import BM25Index
            self._bm25 = BM25Index(self.chunks)
        return self._bm25

    @property
    def graph(self):
        """The import/call graph, rebuilt only when the index changed."""
        if self._graph is None or self._graph_gen != self.generation:
            from .graph import Graph
            self._graph = Graph(self)
            self._graph_gen = self.generation
        return self._graph

    def callers_ranked(self, symbol: str, def_file: str | None
                       ) -> tuple[list[Chunk], list[Chunk]]:
        """(confident, approximate) callers of `symbol`. A call is confident
        when its file is the definition's own file or imports it (so the bare
        name can really refer to it); a bare-name match from an unrelated file
        is approximate -- two different `run()` methods look identical by name."""
        callers = self.callers_of(symbol)
        if not def_file:
            return callers, []
        try:
            near = set(self.graph.importers_of(def_file)) | {def_file}
        except Exception:
            return callers, []
        sure = [c for c in callers if c.file in near]
        return sure, [c for c in callers if c.file not in near]

    def importers_of(self, rel: str) -> list[str]:
        """Indexed files that import `rel`, from the resolved import graph
        (evidence, not proof: dynamic imports never name the file)."""
        return self.graph.importers_of(rel)

    def hydrate(self, c: Chunk) -> Chunk:
        """Fill a chunk's source from disk on demand (loaded indexes keep only
        signatures and summaries in memory)."""
        if c.source or c.kind == "reference":
            return c
        try:
            lines = fileio.read_lines_cached(self.root / c.file)
        except OSError:
            return c
        c.source = "\n".join(lines[c.start - 1: c.end])
        return c

    # ----------------------------------------------------------- rendering
    def render_files(self, line_ranges: bool = True) -> str:
        """Tier-0 map: one card per file naming what it is, imports, and
        definitions (with line ranges so the model can go straight to `read`)."""
        return self.render_cards(self.code_files + self.reference_files,
                                 line_ranges)

    def _test_maps(self) -> tuple[dict, dict]:
        """({(stem, family): [modules]}, {(stem, family): [tests]}) for the
        cards' `tested by:` / `tests:` links, across languages."""
        key = ("tests", self.generation)
        hit = self._map_cache.get(key)
        if hit is None:
            hit = testlinks.build_maps(self.files)
            self._map_cache[key] = hit
        return hit

    @staticmethod
    def _test_link(rel: str, mods: dict, tests: dict) -> tuple[str, str] | None:
        """('tested by', test paths) for a module, ('tests', module path) for
        a test file, when the stem match is unambiguous. Advisory map hint:
        it saves a grep for the obvious naming conventions."""
        return testlinks.link_for(rel, mods, tests)

    def _blast_note(self, f: str) -> str:
        """`imported by: N files (a, b)` for a file card, or the labelled
        'no importers found' hint for a code file nothing visibly loads."""
        g = self.graph
        imp = sorted(g.importers.get(f, ()))
        if imp:
            names = ", ".join(Path(x).name for x in imp[:2])
            more = f", +{len(imp) - 2}" if len(imp) > 2 else ""
            return f"imported by: {len(imp)} file(s) ({names}{more})"
        stem = Path(f).stem.lower()
        is_test, _s, _fam = testlinks.classify(f)
        if (g.import_reliable(f) and not is_test and not self.is_reference(f)
                and stem not in _ENTRY_STEMS and f not in g.called_files):
            return ("no importers or callers found (evidence, not proof: "
                    "an entry point or dynamically loaded?)")
        return ""

    def render_cards(self, rels: list[str], line_ranges: bool = True,
                     max_names: int = 60) -> str:
        """File cards for any subset of files: the tier-0 map's body, and what
        `outline` renders for a directory."""
        out: list[str] = []
        mods_by_stem, tests_by_stem = self._test_maps()
        for f in rels:
            out.append(self._card(f, line_ranges, max_names, mods_by_stem,
                                  tests_by_stem))
        return "\n\n".join(out)

    def _card(self, f: str, line_ranges: bool, max_names: int,
              mods_by_stem: dict, tests_by_stem: dict) -> str:
        units = self.in_file(f)
        if self.is_reference(f):
            return self._reference_card(f, units, line_ranges)
        # a class header split into windows (`Name#2`...) overlaps its first
        # part, which already spans the whole class: list it once
        top = [c for c in self.top_level(f)
               if not (c.kind == "classheader" and "#" in c.qualname)]
        nlines = max((c.end for c in units), default=0)
        imports = sorted({i.split(".")[0].lstrip(".") for c in units[:3]
                          for i in c.imports} - {"", "__future__"})
        card = [f"### {f}  ({nlines} lines, {len(units)} units)"]
        if imports:
            card.append(f"  imports: {', '.join(imports[:14])}")
        blast = self._blast_note(f)
        if blast:
            card.append(f"  {blast}")
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
        return "\n".join(card)

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

    def render_tree(self, budget_tokens: int, order: list[str] | None = None
                    ) -> str:
        """Hierarchical directory skeleton within `budget_tokens`: top-level
        directories first, then the biggest ones expanded into subdirectories
        while the budget lasts. Each line names a directory's file count and
        its best-ranked files; the model descends with `outline <dir>`."""
        files = self.files
        rank_pos = {f: i for i, f in enumerate(order)} if order else {}
        pr = self.graph.pagerank()
        tree: dict[str, list[str]] = defaultdict(list)    # dir -> files below
        for f in files:
            parts = f.split("/")[:-1]
            if not parts:
                tree["."].append(f)
            for i in range(len(parts)):
                tree["/".join(parts[:i + 1])].append(f)

        def best(fl: list[str], n: int = 3) -> str:
            key = (lambda f: rank_pos.get(f, 10 ** 6)) if rank_pos \
                else (lambda f: -pr.get(f, 0.0))
            return ", ".join(Path(x).name for x in sorted(fl, key=key)[:n])

        def line(d: str) -> str:
            fl = tree[d]
            label = "(root files)" if d == "." else d + "/"
            return f"  {label} {len(fl)} files: {best(fl)}"

        depth_of = lambda d: 0 if d == "." else d.count("/") + 1
        shown = [d for d in tree if depth_of(d) <= 1]
        shown.sort(key=lambda d: (d != ".", d))
        used = sum(estimate_tokens(line(d)) for d in shown)
        # expand the largest directories into their children while budget allows
        frontier = sorted((d for d in shown if d != "." and len(tree[d]) > 8),
                          key=lambda d: -len(tree[d]))
        seen = set(shown)
        while frontier and used < budget_tokens:
            d = frontier.pop(0)
            kids = sorted((k for k in tree if k.startswith(d + "/")
                           and k.count("/") == d.count("/") + 1
                           and k not in seen),
                          key=lambda k: -len(tree[k]))[:10]
            for k in kids:
                cost = estimate_tokens(line(k))
                if used + cost > budget_tokens:
                    break
                shown.append(k)
                seen.add(k)
                used += cost
                if len(tree[k]) > 8:
                    frontier.append(k)
        shown.sort(key=lambda d: (d != ".", d))
        out: list[str] = []
        cap = budget_tokens
        used = 0
        for d in shown:
            ln = line(d)
            used += estimate_tokens(ln)
            if used > cap and out:
                out.append("  ... (more directories; `outline .` lists them)")
                break
            out.append(ln)
        return "\n".join(out)

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

    def estimate_flat_tokens(self) -> int:
        """Cheap upper-ish estimate of the flat map's size, without building
        its (potentially multi-megabyte) text."""
        chars = 0
        for c in self.chunks:
            chars += 28 + len(c.signature) + len(c.summary or c.docline)
        return int(chars / 3.6)

    def map_for_prompt(self, cfg, task: str = "") -> tuple[str, str]:
        """Render the map at the richest tier that fits: flat within
        cfg.flat_map_token_budget, else file cards (up to twice the budget),
        else -- for large projects -- a RANKED block (the files most likely
        relevant to `task`, as cards) over a hierarchical directory skeleton,
        else plain directory cards. Returns (map_text, tier_name).

        When the whole map fits it is listed in a stable (alphabetical) order
        so the server's prefix cache survives across tasks; ranking only
        decides WHAT to show when something must be left out."""
        budget = int(cfg.flat_map_token_budget)
        ranked_on = bool(getattr(cfg, "rank_map", True))
        key = (self.generation, budget, bool(cfg.file_card_line_ranges),
               ranked_on, hashlib.sha1((task or "").encode()).hexdigest()[:10])
        hit = self._map_cache.get(("map",) + key)
        if hit is not None:
            return hit
        result = self._render_tier(cfg, task, budget, ranked_on)
        self._map_cache = {k: v for k, v in self._map_cache.items()
                           if not (isinstance(k, tuple) and k and k[0] == "map")}
        self._map_cache[("map",) + key] = result
        return result

    def _render_tier(self, cfg, task: str, budget: int, ranked_on: bool
                     ) -> tuple[str, str]:
        if self.estimate_flat_tokens() <= budget * 1.3:
            flat = self.render()
            if estimate_tokens(flat) <= budget:
                return flat, "flat"
        cards = self.render_files(line_ranges=cfg.file_card_line_ranges)
        if estimate_tokens(cards) <= budget * 2:
            return cards, "cards"
        if ranked_on and budget >= 400:
            return self.render_ranked(cfg, task, budget), "ranked"
        return self.render_dirs(), "dirs"

    def render_ranked(self, cfg, task: str, budget: int) -> str:
        """The large-project map: file cards for the files most likely
        relevant to `task` (best first, each with why it was listed), then a
        hierarchical skeleton of the whole tree."""
        from .rank import rank_files
        ranked = rank_files(self, task, cfg)
        order = [r.file for r in ranked]
        block_budget = int(budget * 1.1)
        skeleton_budget = max(200, int(budget * 0.7))
        mods, tests = self._test_maps()
        cards_out: list[str] = []
        used = 0
        listed = 0
        for r in ranked:
            card = self._card(r.file, cfg.file_card_line_ranges, 12, mods, tests)
            if r.reasons and r.score > 0:
                card += f"\n  why listed: {r.why}"
            cost = estimate_tokens(card)
            if used + cost > block_budget and listed:
                break
            cards_out.append(card)
            used += cost
            listed += 1
        tree = self.render_tree(skeleton_budget, order)
        return (f"LIKELY RELEVANT FILES ({listed} of {len(self.files)}, "
                f"best first)\n\n" + "\n\n".join(cards_out)
                + f"\n\nDIRECTORY SKELETON (all {len(self.files)} files)\n"
                + tree)

    # --------------------------------------------------------- persistence
    def save(self) -> Path:
        """Write the index to .pedacito/index.json under the project root
        (temp file + rename, so a crash never leaves a half-written map)."""
        d = self.root / INDEX_DIR
        d.mkdir(exist_ok=True)
        payload = {
            "version": INDEX_VERSION,
            "root": str(self.root),
            "file_summaries": self.file_summaries,
            "file_hashes": self.file_hashes,
            "file_stats": self.file_stats,
            "file_imports": self.file_imports,
            "chunks": [c.to_record() for c in self.chunks],
        }
        p = d / INDEX_FILE
        tmp = d / (INDEX_FILE + ".tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":")),
                       encoding="utf-8")
        os.replace(tmp, p)
        return p

    @classmethod
    def load(cls, root: Path) -> "Index | None":
        """Load .pedacito/index.json under `root`. Chunk source is NOT read
        here -- it is hydrated lazily (`hydrate`) for the few units a task
        touches, so opening a huge project costs the JSON parse, not a read of
        every file. Returns None when there is no usable index (missing,
        corrupt, or written by a newer version)."""
        p = root / INDEX_DIR / INDEX_FILE
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if int(data.get("version", 1)) > INDEX_VERSION:
                print("[index] index.json was written by a newer Pedacito; "
                      "rebuilding", file=sys.stderr)
                return None
            chunks = [Chunk(**{k: v for k, v in rec.items()
                               if k in _CHUNK_FIELDS})
                      for rec in data["chunks"]]
        except (ValueError, KeyError, TypeError, OSError) as e:
            print(f"[index] index.json is unreadable ({type(e).__name__}); "
                  "rebuilding", file=sys.stderr)
            return None
        return cls(root, chunks, data.get("file_summaries", {}),
                   data.get("file_hashes", {}), data.get("file_stats", {}),
                   data.get("file_imports", {}))


# Entry-point style stems that legitimately have no importers.
_ENTRY_STEMS = {
    "main", "__main__", "__init__", "cli", "app", "server", "index", "manage",
    "run", "wsgi", "asgi", "setup", "conftest", "config", "settings", "mod",
    "lib", "build", "gulpfile", "webpack", "rollup", "vite", "program",
}


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


# (path, mtime_ns, size) -> parsed cache. `reindex_file` runs per file in hot
# loops; re-reading and re-parsing summaries.json each time was quadratic.
_cache_memo: dict[str, tuple[tuple, dict]] = {}
_cache_lock = threading.Lock()


def _cache_path(root: Path) -> Path:
    return root / INDEX_DIR / CACHE_FILE


def _load_cache(root: Path) -> dict:
    """Load the persisted summary cache from .pedacito/summaries.json (memoised
    by file stamp). The returned dict is shared: mutate it only through
    `_put`/`_save_cache`."""
    p = _cache_path(root)
    try:
        st = p.stat()
    except OSError:
        return {}
    stamp = (st.st_mtime_ns, st.st_size)
    with _cache_lock:
        hit = _cache_memo.get(str(p))
        if hit and hit[0] == stamp:
            return hit[1]
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except (json.JSONDecodeError, OSError):
        data = {}
    with _cache_lock:
        _cache_memo[str(p)] = (stamp, data)
    return data


def _save_cache(root: Path, cache: dict) -> None:
    """Persist the summary cache to .pedacito/summaries.json (atomic)."""
    d = root / INDEX_DIR
    d.mkdir(exist_ok=True)
    p = _cache_path(root)
    tmp = d / (CACHE_FILE + ".tmp")
    tmp.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    try:
        st = p.stat()
        with _cache_lock:
            _cache_memo[str(p)] = ((st.st_mtime_ns, st.st_size), cache)
    except OSError:
        pass


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
        self._lock = threading.Lock()

    def ok(self) -> None:
        """Reset the consecutive-failure counter after a successful call."""
        with self._lock:
            self.consecutive = 0

    def fail(self, label: str, err: Exception) -> None:
        """Record one failure; raises SystemExit if the batch should stop
        (a model-load error, or `limit` consecutive failures reached)."""
        with self._lock:
            self.consecutive += 1
            self.total += 1
            consecutive = self.consecutive
        print(f"[warn] summary failed for {label}: {err}", file=sys.stderr)
        if isinstance(err, ModelLoadError) or consecutive >= self.limit:
            raise SystemExit(
                f"\nStopping: {consecutive} summaries in a row failed.\n"
                f"{err}\n\n"
                "Progress so far is cached, so fixing the problem and re-running\n"
                "`pedacito index` will resume rather than start over."
            )


def _ask(client, cfg, system: str, prompt: str, label: str,
         breaker: _Breaker, max_tokens: int | None = None) -> str:
    """Request one summary from the model, truncated to 400 chars (a batched
    reply, `max_tokens` given, keeps its lines and is parsed by the caller).
    Failures go to `breaker` (which may raise SystemExit); returns "" on
    failure."""
    try:
        s = client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_tokens=max_tokens or cfg.max_tokens_summary,
        ).strip()
    except LLMError as e:
        breaker.fail(label, e)
        return ""
    breaker.ok()
    if max_tokens:
        return s
    s = s.replace("\n", " ")
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
    return _chunk_with_imports(rel, path, cfg)[0]


def _chunk_with_imports(rel: str, path: Path, cfg
                        ) -> tuple[list[Chunk], list[str]]:
    """(chunks, full import specifiers) for one file; one read of the file."""
    suffix = path.suffix.lower()
    if suffix in PY_EXTS:
        src = fileio.read_text(path)
        return (_unique_qualnames(chunk_source(
            rel, src, cfg.class_split_lines,
            cfg.max_unit_lines, cfg.max_unit_chars,
            cfg.max_unit_tokens, cfg.unit_overlap_lines)),
            raw_imports(rel, src))
    if suffix in LANG_BY_EXT and TS_AVAILABLE:
        src = fileio.read_text(path)
        return (_unique_qualnames(chunk_file_ts(
            rel, src, suffix, cfg.class_split_lines,
            cfg.max_unit_lines, cfg.max_unit_chars,
            cfg.max_unit_tokens, cfg.unit_overlap_lines)),
            raw_imports(rel, src))
    return _unique_qualnames(chunk_reference(rel, path,
                                             cfg.max_reference_bytes)), []


def _unit_prompt(c: Chunk, cfg) -> str:
    lang = Path(c.file).suffix.lstrip(".") or "text"
    return (f"File: {c.file}\nUnit: {c.qualname} ({c.kind}), "
            f"lines {c.start}-{c.end}\n\n"
            f"```{lang}\n{c.source[:cfg.max_chunk_chars]}\n```")


def _summary_workers(cfg) -> int:
    """Concurrent summary requests: `summary_workers` if set, else 4 for a
    hosted API (which serves requests in parallel) and 1 for a local server
    (a single GPU queue gains nothing, and a half-loaded model could stall)."""
    n = int(getattr(cfg, "summary_workers", 0) or 0)
    if n > 0:
        return min(n, 8)
    from .config import is_hosted
    return 4 if is_hosted(cfg) else 1


_BATCH_SYSTEM = (
    "You summarise source code for a lookup index. You get N numbered units. "
    "Reply with EXACTLY N lines, one per unit, in order, each formatted as "
    "`<number>: <one sentence starting with a verb>`. No markdown, no preamble, "
    "no blank lines.")


def _parse_batch(text: str, n: int) -> list[str]:
    """Per-unit summaries from a batched reply ('' for any unit that did not
    come back as `k: sentence`)."""
    out = [""] * n
    for line in (text or "").splitlines():
        m = re.match(r"^\s*(\d+)\s*[:.)-]\s*(.+?)\s*$", line)
        if m and 1 <= int(m.group(1)) <= n and not out[int(m.group(1)) - 1]:
            s = m.group(2).strip()
            out[int(m.group(1)) - 1] = (s[:400].rsplit(" ", 1)[0] + "...") \
                if len(s) > 400 else s
    return out


def _summarise_units(idx: "Index", todo: list[Chunk], client, cfg, cache: dict,
                     breaker: "_Breaker", verbose: bool,
                     save: bool = True) -> int:
    """Summarise `todo` through a bounded worker pool (sequential for one
    worker), storing results in the content-hash cache. With
    `summary_batch` > 1, short units share one request (fewer round-trips and
    prompt prefills; unparsed lines fall back to single requests). Returns how
    many summaries landed. A tripped breaker saves progress and re-raises."""
    root = idx.root
    batch_n = max(1, int(getattr(cfg, "summary_batch", 1) or 1))
    small = [c for c in todo if batch_n > 1 and c.nlines <= 25
             and len(c.source) <= 1500]
    big = [c for c in todo if c not in small]
    groups: list[list[Chunk]] = [[c] for c in big]
    for i in range(0, len(small), batch_n):
        groups.append(small[i:i + batch_n])
    workers = max(1, min(_summary_workers(cfg), len(groups) or 1))
    done = 0
    lock = threading.Lock()

    def work(group: list[Chunk]) -> list[str]:
        if len(group) == 1:
            c = group[0]
            return [_ask(client, cfg, SUMMARY_SYSTEM, _unit_prompt(c, cfg),
                         c.key, breaker)]
        body = "\n\n".join(
            f"### {k}. {c.file} :: {c.qualname}\n{c.source[:600]}"
            for k, c in enumerate(group, 1))
        text = _ask(client, cfg, _BATCH_SYSTEM, f"N = {len(group)}\n\n{body}",
                    group[0].key + f" (+{len(group) - 1})", breaker,
                    max_tokens=cfg.max_tokens_summary * len(group))
        got = _parse_batch(text, len(group))
        # whatever the batch did not return is asked for singly
        return [g or _ask(client, cfg, SUMMARY_SYSTEM, _unit_prompt(c, cfg),
                          c.key, breaker)
                for g, c in zip(got, group)]

    def land(i: int, group: list[Chunk], texts: list[str]) -> None:
        nonlocal done
        for c, text in zip(group, texts):
            if text:
                c.summary = text
                idx.stale_summaries.discard(c.key)
                _put(cache, c.sha, text, cfg.model)
                with lock:
                    done += 1
        if save and i % 10 == 0:
            _save_cache(root, cache)

    try:
        if workers == 1:
            for i, group in enumerate(groups, 1):
                if verbose:
                    print(f"  [{i}/{len(groups)}] {group[0].key}"
                          + (f" (+{len(group) - 1})" if len(group) > 1 else ""),
                          file=sys.stderr)
                land(i, group, work(group))
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(work, g): (i, g)
                        for i, g in enumerate(groups, 1)}
                for fut in as_completed(futs):
                    i, g = futs[fut]
                    if verbose:
                        print(f"  [{i}/{len(groups)}] {g[0].key}", file=sys.stderr)
                    land(i, g, fut.result())
    except SystemExit:
        if save:
            _save_cache(root, cache)   # keep what we already paid for
            idx.save()
        raise
    if save:
        _save_cache(root, cache)
    return done


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
    imports_by_file: dict[str, list[str]] = {}
    for f in files:
        try:
            rel = f.resolve().relative_to(root).as_posix()
        except ValueError:
            rel = f.name
        rels.append(rel)
        try:
            got, imps = _chunk_with_imports(rel, f, cfg)
            chunks.extend(got)
            imports_by_file[rel] = imps
        except SyntaxError as e:
            print(f"[skip] {rel}: syntax error line {e.lineno}", file=sys.stderr)

    idx = Index(root, chunks, file_imports=imports_by_file)
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

    _summarise_units(idx, todo, client, cfg, cache, breaker, verbose)
    _save_cache(root, cache)

    idx.save()
    return idx


def reindex_file(index: Index, rel: str, cfg, save: bool = True) -> Index:
    """Re-chunk one changed file and update `index` IN PLACE (returned for
    call-site compatibility). Summaries carry over by content hash; a changed
    unit that has no cached summary keeps its OLD summary (matched by
    qualname) flagged stale until `refresh_summaries` replaces it, instead of
    going blank. Pass save=False when batching several reindexes and saving
    once.

    Secret/credential files (`security.is_sensitive_path`) are NEVER folded
    in -- an explicitly approved read may show a file to the model for this
    step, but its contents must not persist into `.pedacito/index.json`."""
    root = index.root
    from . import security
    if security.is_sensitive_path(Path(root) / rel):
        if rel in index.files:            # an older index leaked one in: drop it
            index.forget(rel)
            if save:
                index.save()
        return index
    try:
        new, imports = _chunk_with_imports(rel, root / rel, cfg)
    except (SyntaxError, OSError) as e:
        print(f"[warn] {rel} could not be re-indexed ({e}); index left stale",
              file=sys.stderr)
        return index
    cache = _load_cache(root)
    old_by_name = {c.qualname: c.summary for c in index.in_file(rel)
                   if c.summary}
    for c in new:
        if c.sha in cache:
            c.summary = _entry(cache[c.sha])[0]
            index.stale_summaries.discard(c.key)
        else:
            inherited = old_by_name.get(c.qualname, "")
            c.summary = inherited
            if inherited and needs_summary(c, cfg):
                index.stale_summaries.add(c.key)
    index.replace_file(rel, new, imports)
    index.file_hashes[rel] = file_hash(root / rel)
    st = file_stat(root / rel)
    if st is not None:
        index.file_stats[rel] = st
    else:
        index.file_stats.pop(rel, None)
    if save:
        index.save()
    return index


def pending_summaries(index: Index, cfg) -> list[Chunk]:
    """Units that should be (re)summarised: eligible, and either blank or
    carrying a summary from before their last edit."""
    return [c for c in index.chunks
            if needs_summary(c, cfg)
            and (not c.summary or c.key in index.stale_summaries)]


def refresh_summaries(index: Index, client, cfg, limit: int = 12,
                      verbose: bool = True) -> int:
    """Regenerate up to `limit` pending unit summaries (changed or new code)
    so a file the user edited does not lose its map text until the next
    manual `pedacito index`. Bounded by `limit` so a big pull never stalls a
    task; the rest wait for `pedacito index`. Returns how many landed."""
    if client is None or limit <= 0:
        return 0
    todo = pending_summaries(index, cfg)[:limit]
    if not todo:
        return 0
    for c in todo:
        index.hydrate(c)
    cache = _load_cache(index.root)
    breaker = _Breaker(cfg.max_consecutive_failures)
    if verbose:
        print(f"  refreshing {len(todo)} unit summar"
              f"{'y' if len(todo) == 1 else 'ies'} for changed code...",
              file=sys.stderr)
    try:
        n = _summarise_units(index, todo, client, cfg, cache, breaker, False)
    except SystemExit:
        return 0
    if n:
        index.generation += 1
        index.save()
    return n
