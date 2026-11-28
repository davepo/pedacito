"""File dependency graph: resolved imports, call edges, reverse index, PageRank.

    g = index.graph                       # built lazily, cached per generation
    g.importers["src/db.py"]              # files that import it (O(1))
    g.pagerank()["src/db.py"]             # how central it is, 0..1
    g.impact("src/db.py", depth=2)        # transitive importers {file: hops}

Edges come from two evidence sources, both approximate by design:

  * imports -- full specifiers extracted at index time (imports.py) and
    resolved to indexed files; an unresolvable import (a third-party package)
    creates no edge;
  * calls   -- a bare call name that is defined in exactly ONE other file
    adds a weaker edge (covers Go packages and dynamic-import styles).

The reverse map is computed once, so `callers`/`importers_of` stop paying a
linear scan over every chunk per query.
"""

from __future__ import annotations

from collections import defaultdict

from . import testlinks
from .imports import FileSet, resolve

CALL_EDGE_WEIGHT = 0.5


class Graph:
    """Import + call graph over one Index's files."""

    def __init__(self, index):
        files = list(index.files)
        self.files = files
        fs = FileSet(files)
        self.imports: dict[str, set[str]] = {f: set() for f in files}
        self.importers: dict[str, set[str]] = {f: set() for f in files}
        self.call_edges: dict[str, dict[str, float]] = defaultdict(dict)
        self.resolved_files: set[str] = set()   # files whose imports resolved
        for f in files:
            for spec in index.imports_of(f):
                for target in resolve(f, spec, fs):
                    if target != f and target in self.imports:
                        self.imports[f].add(target)
                        self.importers[target].add(f)
                        self.resolved_files.add(f)
        self._add_call_edges(index)
        # files some other file reaches through a call edge
        self.called_files: set[str] = {t for tg in self.call_edges.values()
                                       for t in tg}
        self._pr: dict[str, float] | None = None
        self._sym_pr: dict[str, float] | None = None

    def _add_call_edges(self, index) -> None:
        defs: dict[str, set[str]] = defaultdict(set)
        for c in index.chunks:
            if c.kind in ("module", "reference") \
                    or c.qualname in ("module", "__trailing__"):
                continue
            base = c.qualname.split("#")[0].split(".")[-1]
            if len(base) > 2:
                defs[base].add(c.file)
        for c in index.chunks:
            for name in c.calls:
                owners = defs.get(name)
                if not owners or len(owners) != 1:
                    continue
                (target,) = owners
                if target != c.file:
                    self.call_edges[c.file][target] = CALL_EDGE_WEIGHT

    # ------------------------------------------------------------ queries
    def importers_of(self, rel: str) -> list[str]:
        return sorted(self.importers.get(rel, ()))

    def callers_files(self, rel: str) -> list[str]:
        """Files with a (weak) call edge into `rel`."""
        return sorted(src for src, tgts in self.call_edges.items()
                      if rel in tgts)

    def dependents(self, rel: str) -> set[str]:
        """Files affected if `rel` changes directly: importers + callers."""
        return set(self.importers.get(rel, ())) | set(self.callers_files(rel))

    def impact(self, rel: str, depth: int = 2) -> dict[str, int]:
        """Transitive dependents of `rel` up to `depth` hops: {file: hops}."""
        seen: dict[str, int] = {}
        frontier = {rel}
        for hop in range(1, depth + 1):
            nxt: set[str] = set()
            for f in frontier:
                for d in self.dependents(f):
                    if d != rel and d not in seen:
                        seen[d] = hop
                        nxt.add(d)
            frontier = nxt
            if not frontier:
                break
        return seen

    def neighbors(self, rel: str) -> set[str]:
        """Direct dependencies and dependents."""
        out = set(self.imports.get(rel, ())) | set(self.importers.get(rel, ()))
        out |= set(self.call_edges.get(rel, {}))
        out |= set(self.callers_files(rel))
        return out

    def import_cycles(self, only=None) -> list[list[str]]:
        """Strongly connected import components of size > 1, deterministic.

        `only`, when given, restricts the graph to those files (edges to a
        file outside the set are ignored). An import cycle is legal in Python
        and in several other languages, so callers treat this as evidence and
        a warning, never as a hard failure. The implementation is iterative
        Tarjan so a deep import chain cannot hit the recursion limit.
        """
        allowed = set(self.files)
        if only is not None:
            allowed &= {str(f).replace("\\", "/") for f in only}
        nodes = sorted(allowed)
        adj = {
            f: sorted(t for t in self.imports.get(f, ()) if t in allowed)
            for f in nodes
        }
        indices: dict[str, int] = {}
        low: dict[str, int] = {}
        on_stack: set[str] = set()
        stack: list[str] = []
        components: list[list[str]] = []
        counter = 0
        for root in nodes:
            if root in indices:
                continue
            work: list[tuple[str, int]] = [(root, 0)]
            while work:
                node, pos = work[-1]
                if pos == 0:
                    indices[node] = low[node] = counter
                    counter += 1
                    stack.append(node)
                    on_stack.add(node)
                if pos < len(adj[node]):
                    target = adj[node][pos]
                    work[-1] = (node, pos + 1)
                    if target not in indices:
                        work.append((target, 0))
                    elif target in on_stack:
                        low[node] = min(low[node], indices[target])
                    continue
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
                if low[node] == indices[node]:
                    component: list[str] = []
                    while True:
                        item = stack.pop()
                        on_stack.discard(item)
                        component.append(item)
                        if item == node:
                            break
                    if len(component) > 1:
                        components.append(sorted(component))
        return sorted(components, key=lambda c: (c[0], len(c), c))

    def import_reliable(self, rel: str) -> bool:
        """Whether 'no importers' means something for this file: the language
        resolves imports to files (not Go packages / C# namespaces), and at
        least one import in the project resolved at all."""
        return testlinks.family_of(rel) in ("py", "js", "rs", "rb", "java", "c") \
            and bool(self.resolved_files)

    # ------------------------------------------------------------ PageRank
    def pagerank(self, iterations: int = 25, damping: float = 0.85
                 ) -> dict[str, float]:
        """Centrality over import (weight 1) and call (weight 0.5) edges,
        flowing from importer to imported; normalised so the top file is 1."""
        if self._pr is not None:
            return self._pr
        files = self.files
        n = len(files)
        if n == 0:
            self._pr = {}
            return self._pr
        # Tests import everything; an import from a test file says little about
        # a module's importance, so those edges count a fifth as much.
        scale = {f: (0.2 if testlinks.classify(f)[0] else 1.0) for f in files}
        out_w: dict[str, dict[str, float]] = {f: {} for f in files}
        for f, tgts in self.imports.items():
            for t in tgts:
                out_w[f][t] = out_w[f].get(t, 0.0) + scale[f]
        for f, tgts in self.call_edges.items():
            for t, w in tgts.items():
                out_w[f][t] = out_w[f].get(t, 0.0) + w * scale[f]
        # raw (unscaled) out-weight: a test file's edges carry only `scale` of
        # its rank, and the rest is spread like dangling mass
        total_out = {f: sum(w / scale[f] for w in ws.values()) if ws else 0.0
                     for f, ws in out_w.items()}
        rank = {f: 1.0 / n for f in files}
        for _ in range(iterations):
            nxt = {f: (1.0 - damping) / n for f in files}
            lost = 0.0
            for f in files:
                tot = total_out[f]
                if not tot:
                    lost += rank[f]
                else:
                    lost += rank[f] * (1.0 - min(1.0, sum(out_w[f].values()) / tot))
            share = damping * lost / n
            for f in files:
                nxt[f] += share
            for f, tgts in out_w.items():
                tot = total_out[f]
                if not tot:
                    continue
                give = damping * rank[f] / tot
                for t, w in tgts.items():
                    nxt[t] += give * w
            rank = nxt
        top = max(rank.values()) or 1.0
        self._pr = {f: r / top for f, r in rank.items()}
        return self._pr


def symbol_pagerank(index, seeds=None, *, iterations: int = 25,
                    damping: float = 0.85) -> dict[str, float]:
    """Symbol-level PageRank over `Chunk.calls` (including receiver-qualified
    `Class.method` calls) and resolved `imports_of` edges.

    Returns `{f"{file}::{qualname}": score}` normalised so the top symbol is
    1.0. Unseeded calls are cached on `index.graph` for the current index
    generation. `seeds`, when supplied, may be an iterable of symbol keys
    (`file::qualname`), qualnames (`Class.method` / `fn`), bare names, or file
    paths, or a `{seed: weight}` mapping for personalized PageRank.
    """
    g = index.graph
    if seeds is None and g._sym_pr is not None:
        return g._sym_pr

    def_nodes: set[str] = set()
    node_file: dict[str, str] = {}
    by_qual: dict[str, list[str]] = defaultdict(list)
    by_bare: dict[str, list[str]] = defaultdict(list)
    file_top_defs: dict[str, list[str]] = defaultdict(list)

    for c in index.chunks:
        if c.kind in ("module", "reference"):
            continue
        base_q = c.qualname.split("#")[0]
        if base_q in ("module", "__trailing__"):
            continue
        key = f"{c.file}::{base_q}"
        if key in def_nodes:
            continue
        def_nodes.add(key)
        node_file[key] = c.file
        by_qual[base_q].append(key)
        bare = base_q.split(".")[-1]
        if len(bare) > 1:
            by_bare[bare].append(key)
        # Whole-class chunks (below class_split_lines) store their methods in
        # `c.members`: register `Class.member` -> `file::Class` so receiver-
        # qualified `Class.method` calls credit the owning class cleanly.
        if c.kind == "class" and c.members:
            for mem in c.members:
                if mem:
                    by_qual[f"{base_q}.{mem}"].append(key)
        if "." not in base_q and not base_q.startswith("_"):
            file_top_defs[c.file].append(key)

    if not def_nodes:
        if seeds is None:
            g._sym_pr = {}
        return {}

    for d in (by_qual, by_bare, file_top_defs):
        for k in d:
            d[k].sort()

    # Include per-file module source nodes so module-level calls and imports
    # flow rank into the symbols they invoke/import.
    mod_nodes: set[str] = {f"{f}::__module__" for f in index.files}
    for f in index.files:
        node_file[f"{f}::__module__"] = f
    all_nodes = sorted(def_nodes | mod_nodes)
    n = len(all_nodes)

    scale_file = {f: (0.2 if testlinks.classify(f)[0] else 1.0)
                  for f in index.files}
    out_w: dict[str, dict[str, float]] = {u: {} for u in all_nodes}

    # 1. Call edges from Chunk.calls (bare and receiver-qualified Class.method)
    for c in index.chunks:
        if c.kind == "reference" or not c.calls:
            continue
        base_q = c.qualname.split("#")[0]
        if c.kind == "module" or base_q in ("module", "__trailing__"):
            src = f"{c.file}::__module__"
        else:
            src = f"{c.file}::{base_q}"
        if src not in out_w:
            continue
        sc = scale_file.get(c.file, 1.0)
        imp_files = g.imports.get(c.file, ())
        for call_name in c.calls:
            if "." in call_name:
                cands = by_qual.get(call_name, [])
                if not cands:
                    parts = call_name.split(".")
                    if len(parts) >= 2:
                        cands = by_qual.get(".".join(parts[-2:]), [])
                if not cands:
                    continue
                scoped = [k for k in cands
                          if node_file[k] == c.file or node_file[k] in imp_files]
                targets = scoped or cands
                w = (1.5 * sc) / len(targets)
            else:
                cands = by_qual.get(call_name) or by_bare.get(call_name, [])
                if not cands:
                    continue
                scoped = [k for k in cands
                          if node_file[k] == c.file or node_file[k] in imp_files]
                if scoped:
                    targets = scoped
                    w = (1.0 * sc) / len(targets)
                elif len(cands) == 1:
                    targets = cands
                    w = 0.5 * sc
                else:
                    continue
            for t in targets:
                if t != src:
                    out_w[src][t] = out_w[src].get(t, 0.0) + w

    # 2. Structural reinforcement: method -> enclosing class so calling
    #    `Class.method` also lifts `Class` in file-level symbol cards.
    for key in sorted(def_nodes):
        f, qual = key.split("::", 1)
        if "." in qual:
            parent_key = f"{f}::{qual.rsplit('.', 1)[0]}"
            if parent_key in def_nodes and parent_key != key:
                out_w[key][parent_key] = out_w[key].get(parent_key, 0.0) + 0.35

    # 3. Resolved import edges from `imports_of` (via `index.graph.imports`)
    for f in sorted(index.files):
        src = f"{f}::__module__"
        sc = scale_file.get(f, 1.0)
        for tgt_f in sorted(g.imports.get(f, ())):
            targets = file_top_defs.get(tgt_f, [])
            if not targets:
                continue
            w = (0.25 * sc) / len(targets)
            for t in targets:
                out_w[src][t] = out_w[src].get(t, 0.0) + w

    # Personalization vector (uniform when `seeds` is None or matches nothing)
    teleport: dict[str, float] = {}
    if seeds is not None:
        seed_items = (seeds.items() if isinstance(seeds, dict)
                      else ((s, 1.0) for s in seeds))
        for raw, wt in seed_items:
            if not raw:
                continue
            w = float(wt)
            if w <= 0.0:
                continue
            s = str(raw).strip()
            matched: list[str] = []
            if s in def_nodes:
                matched = [s]
            elif s in by_qual:
                matched = by_qual[s]
            elif s in by_bare:
                matched = by_bare[s]
            elif s in file_top_defs:
                matched = file_top_defs[s]
            else:
                matched = [k for k in sorted(def_nodes) if node_file[k] == s]
            if matched:
                per = w / len(matched)
                for m in matched:
                    teleport[m] = teleport.get(m, 0.0) + per
    if teleport:
        tot_t = sum(teleport.values())
        teleport = {u: teleport.get(u, 0.0) / tot_t for u in all_nodes}
    else:
        teleport = {u: 1.0 / n for u in all_nodes}

    total_out = {u: sum(ws.values()) for u, ws in out_w.items()}
    rank = dict(teleport)
    for _ in range(iterations):
        nxt = {u: (1.0 - damping) * teleport[u] for u in all_nodes}
        lost = 0.0
        for u in all_nodes:
            if not total_out[u]:
                lost += rank[u]
        if lost:
            for u in all_nodes:
                nxt[u] += damping * lost * teleport[u]
        for u in all_nodes:
            tot = total_out[u]
            if not tot:
                continue
            give = damping * rank[u] / tot
            for t, w in out_w[u].items():
                nxt[t] += give * w
        rank = nxt

    def_scores = {u: rank[u] for u in sorted(def_nodes)}
    top = max(def_scores.values()) or 1.0
    result = {u: r / top for u, r in def_scores.items()}
    if seeds is None:
        g._sym_pr = result
    return result


def symbol_importers(index, rel: str, symbol: str) -> list[str]:
    """Importers of ``rel`` that import ``symbol`` from it via a from-import.

    Resolved through the import graph (``index.graph.importers_of``) and each
    importer's parsed import bindings, so a from-import of a same-named symbol
    from ANOTHER module is never mistaken for this one. Used by cross-file
    codemods (``move_symbol``) to find the import statements that must be
    rewritten. Returns project-relative POSIX paths, sorted.
    """
    from . import fileio
    from .imports import FileSet, python_import_bindings, resolve

    try:
        importers = index.graph.importers_of(rel)
    except Exception:
        return []
    if not importers:
        return []
    fs = FileSet(index.files)
    out: list[str] = []
    for f in importers:
        if not f.endswith(".py"):
            continue
        try:
            text = fileio.read_text(index.root / f)
        except (OSError, UnicodeDecodeError):
            continue
        for binding in python_import_bindings(text):
            if binding.kind != "from":
                continue
            spec = ("." * binding.level) + binding.module
            try:
                targets = resolve(f, spec, fs) if spec else []
            except Exception:
                targets = []
            if rel in targets and any(n == symbol for n, _a in binding.names):
                out.append(f)
                break
    return sorted(out)
