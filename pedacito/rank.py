"""Task-conditioned file ranking: what should the map show first?

    ranked = rank_files(index, "make the leech threshold configurable", cfg)
    ranked[0].file, ranked[0].reasons      # 'src/scheduler.py', ['named in task']

Four signals, blended and capped so no single one can drown the rest:

  task       BM25 of the task text against every unit (name > signature >
             summary), aggregated per file                          (search.py)
  proximity  files the task names, the files that define symbols it names,
             and THEIR imports, importers, tests and directory siblings
  graph      PageRank over the import/call graph: widely imported modules
             and entry points outrank leaves                         (graph.py)
  history    uncommitted and recently committed files, and files that
             co-changed with a named file                          (gitinfo.py)

A file the task NAMES always outranks everything. Test files rank below the
code they test (the map links them), and non-code reference files below code,
unless the task names or matches them.
"""

from __future__ import annotations

import posixpath
import re
from collections import defaultdict
from dataclasses import dataclass, field

from . import gitinfo, testlinks

_FILE_TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./\-]*")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}(?:\.[A-Za-z_][A-Za-z0-9_]*)*")

W_NAMED, W_SYMBOL, W_TASK, W_PROX, W_GRAPH, W_HIST = 4.0, 2.5, 2.0, 1.0, 0.7, 0.6


@dataclass
class Ranked:
    file: str
    score: float
    reasons: list[str] = field(default_factory=list)

    @property
    def why(self) -> str:
        return "; ".join(self.reasons[:2])


def task_files(index, task: str) -> list[str]:
    """Indexed files the task text names (a path, or a bare file name)."""
    out: list[str] = []
    seen: set[str] = set()
    for tok in _FILE_TOKEN_RE.findall(task or ""):
        if "/" not in tok and "." not in tok.strip("."):
            continue
        rel = index.resolve_file(tok.strip(".,:;)(\"'`"))
        if rel and rel not in seen:
            seen.add(rel)
            out.append(rel)
    return out


def task_symbols(index, task: str, limit: int = 6) -> list:
    """Definitions whose exact name appears in the task text (case-sensitive,
    so the prose word `parse` does not summon every `parse` function)."""
    out = []
    seen: set[int] = set()
    for tok in _IDENT_RE.findall(task or ""):
        # Only tokens that LOOK like code names: a plain lower-case English
        # word ("recent", "number") that happens to name a function says
        # nothing about the task; BM25 already weighs such words by name.
        looks_like_ident = ("_" in tok or "." in tok
                            or (tok != tok.lower() and tok != tok.capitalize())
                            or tok[0].isupper())
        if not looks_like_ident:
            continue
        hits = index._by_qualname.get(tok) or []
        if not hits and looks_like_ident:
            hits = index._by_suffix.get(tok.lower(), [])
        # an ambiguous common name tells us little; cap what one token adds
        for c in hits[:3]:
            if id(c) not in seen and c.kind not in ("module", "reference"):
                seen.add(id(c))
                out.append(c)
        if len(out) >= limit:
            break
    return out[:limit]


def rank_files(index, task: str = "", cfg=None, use_git: bool = True
               ) -> list[Ranked]:
    """Every indexed file, best first, with the reasons it ranked."""
    g = index.graph
    pr = g.pagerank()
    scores: dict[str, float] = defaultdict(float)
    reasons: dict[str, list[tuple[float, str]]] = defaultdict(list)

    def bump(f: str, amount: float, why: str) -> None:
        if f in pr:
            scores[f] += amount
            if why:
                reasons[f].append((amount, why))

    task = task or ""
    named = task_files(index, task)
    for f in named:
        bump(f, W_NAMED, "named in task")
    sym_files: dict[str, str] = {}
    for c in task_symbols(index, task):
        sym_files.setdefault(c.file, c.qualname)
    for f, sym in sym_files.items():
        bump(f, W_SYMBOL, f"defines {sym}")

    if task.strip():
        ts = index.bm25.file_scores(task)
        top = max(ts.values(), default=0.0)
        for f, s in ts.items():
            if top > 0:
                bump(f, W_TASK * (s / top), "matches the task's words")

    _, tests_map = testlinks.build_maps(index.files)
    prox: dict[str, float] = defaultdict(float)
    prox_why: dict[str, str] = {}
    anchors = list(dict.fromkeys(list(named) + list(sym_files)))
    for a in anchors:
        for nb in g.imports.get(a, ()):
            prox[nb] += 0.8
            prox_why.setdefault(nb, f"imported by {posixpath.basename(a)}")
        for nb in g.importers.get(a, ()):
            prox[nb] += 0.7
            prox_why.setdefault(nb, f"imports {posixpath.basename(a)}")
        for nb in g.call_edges.get(a, {}):
            prox[nb] += 0.4
            prox_why.setdefault(nb, f"called by {posixpath.basename(a)}")
        for t in testlinks.tests_of(a, tests_map):
            prox[t] += 0.9
            prox_why.setdefault(t, f"tests {posixpath.basename(a)}")
        d = posixpath.dirname(a)
        for sib in index.files:
            if posixpath.dirname(sib) == d and sib != a:
                prox[sib] += 0.15
    for f, v in prox.items():
        bump(f, W_PROX * min(v, 1.0), prox_why.get(f, "near a named file"))

    for f, v in pr.items():
        if v > 0.02:
            bump(f, W_GRAPH * v,
                 f"imported by {len(g.importers.get(f, ()))} file(s)"
                 if g.importers.get(f) else "")

    if use_git and (cfg is None or getattr(cfg, "git_context", True)) \
            and gitinfo.is_repo(index.root):
        st = gitinfo.status(index.root)
        dirty = set(st["dirty"]) | set(st["untracked"])
        recent = gitinfo.recent_files(index.root, 10)
        co = gitinfo.cochange(index.root)
        for f in dirty:
            bump(f, W_HIST, "uncommitted change")
        for i, f in enumerate(recent[:20]):
            bump(f, W_HIST * 0.5 * (1.0 - i / 25.0), "recently committed")
        for a in anchors:
            for f, n in sorted(co.get(a, {}).items(), key=lambda kv: -kv[1])[:6]:
                bump(f, W_HIST * min(1.0, n / 3.0), f"changes with {posixpath.basename(a)}")

    named_set = set(named) | set(sym_files)
    matched = {f for f, rs in reasons.items()
               if any(w == "matches the task's words" for _a, w in rs)}
    out: list[Ranked] = []
    for f in index.files:
        s = scores.get(f, 0.0)
        is_test, _stem, _fam = testlinks.classify(f)
        if is_test and f not in named_set and f not in prox:
            s *= 0.7
        if index.is_reference(f) and f not in named_set and f not in matched:
            s *= 0.6
        rs = [w for _a, w in sorted(reasons.get(f, ()), key=lambda t: -t[0])]
        out.append(Ranked(f, s, list(dict.fromkeys(rs))))
    out.sort(key=lambda r: (-r.score, r.file))
    return out
