"""BM25 over indexed units: one search core for `find`, ranking, and grep misses.

    bm = BM25Index(index.chunks)
    for score, chunk in bm.query("retry backoff sync client", limit=10): ...
    bm.remove_file("src/a.py"); bm.add(new_chunks)        # incremental update

Every unit is one document with weighted fields: its qualname (split into
camelCase / snake_case words), signature, path, summary + docline, call and
member names, and a small sample of identifiers from its source. A word found
in a unit's NAME counts for far more than one found in a docstring, so
`find retry` ranks `retry_with_backoff` above a function that merely mentions
retrying. Zero model calls: it replaces the embeddings round-trip that used to
be the only fuzzy search, and usually beats chat-model embeddings at it.

The index is built lazily in memory (a few hundred ms for 100k units) and
updated incrementally when one file is re-chunked.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*|\d+")
_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

STOPWORDS = frozenset("""
a an and are as at be by for from has have how i if in into is it its of on or
that the this to was were what when where which who why will with do does did
not no can could should would you your we our add make use using code file
function method class
""".split())

# Field weights: a hit in the name beats the signature beats prose.
FIELD_WEIGHTS = {"name": 3.0, "sig": 1.5, "path": 1.0, "text": 1.0,
                 "refs": 0.6, "body": 0.3}
K1 = 1.4
B = 0.6


def stem(w: str) -> str:
    """Tiny suffix stripper (retries/retry, running/run, handlers/handler):
    enough to match inflections without a stemming library."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ing"):
        base = w[:-3]
        if len(base) > 2 and base[-1] == base[-2] and base[-1] not in "ls":
            base = base[:-1]
        return base
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 3 and w.endswith("es") and w[-3] in "sxz":
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokenize(text: str, keep_whole: bool = True) -> list[str]:
    """Lower-cased, stemmed word tokens of `text`: identifiers are split at
    camelCase / snake_case boundaries, and (with `keep_whole`) the whole
    identifier is kept too so an exact name still scores."""
    out: list[str] = []
    for ident in _WORD_RE.findall((text or "").replace("_", " ")):
        parts = [p.lower() for p in _CAMEL_RE.findall(ident)] or [ident.lower()]
        for p in parts:
            if len(p) > 1 and p not in STOPWORDS:
                out.append(stem(p))
        if keep_whole and len(parts) > 1:
            out.append(ident.lower())
    return out


def query_terms(text: str) -> list[str]:
    """Distinct query tokens in order (stopwords dropped)."""
    seen: set[str] = set()
    out: list[str] = []
    for t in tokenize(text):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _body_identifiers(source: str, limit: int = 40) -> list[str]:
    """A capped sample of distinct identifiers from a unit's source."""
    seen: dict[str, None] = {}
    for m in _IDENT_RE.finditer(source or ""):
        seen.setdefault(m.group(0), None)
        if len(seen) >= limit * 3:
            break
    words: list[str] = []
    for ident in list(seen)[:limit * 3]:
        words.extend(tokenize(ident, keep_whole=False))
    return words[:limit * 2]


class BM25Index:
    """Weighted-field BM25 with incremental add/remove per file."""

    def __init__(self, chunks=()):
        self.postings: dict[str, dict[int, float]] = defaultdict(dict)
        self.docs: dict[int, tuple[object, float, list[str]]] = {}
        self.by_file: dict[str, list[int]] = defaultdict(list)
        self.total_len = 0.0
        self._next = 0
        self.add(chunks)

    # ------------------------------------------------------------- building
    def _fields(self, c) -> dict[str, list[str]]:
        sig = getattr(c, "signature", "") or ""
        name = getattr(c, "qualname", "") or ""
        members = " ".join(getattr(c, "members", ()) or ())
        calls = " ".join(getattr(c, "calls", ()) or ())
        return {
            "name": tokenize(name.replace(".", " ").replace("#", " ")),
            "sig": tokenize(sig),
            "path": tokenize((getattr(c, "file", "") or "").replace("/", " ")
                             .replace(".", " ")),
            "text": tokenize(f"{getattr(c, 'summary', '')} "
                             f"{getattr(c, 'docline', '')}", keep_whole=False),
            "refs": tokenize(f"{members} {calls}", keep_whole=False),
            # Source text is not indexed: a loaded Index hydrates source
            # lazily, and a ranking that depended on whether a chunk had been
            # read from disk yet would differ between runs.
            "body": [],
        }

    def add(self, chunks) -> None:
        for c in chunks:
            if getattr(c, "kind", "") == "reference" and not (
                    getattr(c, "docline", "") or getattr(c, "signature", "")):
                continue
            weighted: dict[str, float] = defaultdict(float)
            length = 0.0
            for field, toks in self._fields(c).items():
                w = FIELD_WEIGHTS[field]
                for t in toks:
                    weighted[t] += w
                length += w * len(toks)
            doc_id = self._next
            self._next += 1
            self.docs[doc_id] = (c, max(length, 1.0), list(weighted))
            self.by_file[c.file].append(doc_id)
            self.total_len += max(length, 1.0)
            for t, wtf in weighted.items():
                self.postings[t][doc_id] = wtf

    def remove_file(self, rel: str) -> None:
        for doc_id in self.by_file.pop(rel, []):
            entry = self.docs.pop(doc_id, None)
            if entry is None:
                continue
            _c, length, terms = entry
            self.total_len -= length
            for t in terms:
                post = self.postings.get(t)
                if post is not None:
                    post.pop(doc_id, None)
                    if not post:
                        del self.postings[t]

    # -------------------------------------------------------------- querying
    def query(self, text: str, limit: int = 20, kinds=None,
              files=None) -> list[tuple[float, object]]:
        """Top `limit` (score, chunk) for the query words. `kinds` /
        `files` restrict the candidates."""
        terms = query_terms(text)
        if not terms or not self.docs:
            return []
        n = len(self.docs)
        avg = self.total_len / n if n else 1.0
        scores: dict[int, float] = defaultdict(float)
        for t in terms:
            post = self.postings.get(t)
            if not post:
                continue
            idf = math.log(1.0 + (n - len(post) + 0.5) / (len(post) + 0.5))
            for doc_id, wtf in post.items():
                length = self.docs[doc_id][1]
                denom = wtf + K1 * (1 - B + B * length / avg)
                scores[doc_id] += idf * (wtf * (K1 + 1)) / denom
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        out: list[tuple[float, object]] = []
        for doc_id, sc in ranked:
            c = self.docs[doc_id][0]
            if kinds is not None and c.kind not in kinds:
                continue
            if files is not None and c.file not in files:
                continue
            out.append((sc, c))
            if len(out) >= limit:
                break
        return out

    def file_scores(self, text: str, limit: int = 50) -> dict[str, float]:
        """Per-file relevance for a task text: the best unit's score plus a
        damped sum of the rest, so a file with several relevant units
        outranks one lucky hit."""
        hits = self.query(text, limit=400)
        per: dict[str, list[float]] = defaultdict(list)
        for sc, c in hits:
            per[c.file].append(sc)
        out = {f: max(v) + 0.25 * sum(sorted(v, reverse=True)[1:4])
               for f, v in per.items()}
        top = sorted(out.items(), key=lambda kv: -kv[1])[:limit]
        return dict(top)

    def matched_terms(self, chunk, text: str) -> list[str]:
        """Query words that matched `chunk` (for one-line reasons)."""
        terms = query_terms(text)
        have = set()
        for field, toks in self._fields(chunk).items():
            if field != "body":
                have.update(toks)
        return [t for t in terms if t in have]
