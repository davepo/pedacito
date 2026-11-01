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
