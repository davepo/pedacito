"""Optional semantic search over the indexed units.

Usage
-----
    from . import semantic

    matches = semantic.search(index, cfg, client, "where do we retry?")
    matches = semantic.hybrid_search(index, cfg, client, query, candidates)
    text = semantic.render("where do we retry?", matches)

Opt-in (`semantic_search = true`). Uses the same OpenAI-compatible server's
`/v1/embeddings` endpoint and only runs as a last resort when `grep` found
nothing textually, so exact identifiers still get exact results.

The grep fallback is CANDIDATE-FIRST: the free lexical pass (BM25, symbol
names, call graph) assembles a bounded top-K short-list (K = 50), only the
query and those K units are embedded, and the two rankings are merged with
Reciprocal Rank Fusion. A no-hit grep costs K+1 embedding calls, never one
pass over the whole project. `search()` keeps the full-index mode for
callers that want plain by-meaning retrieval without a candidate set.

Embeddings are cached by chunk content hash in `.pedacito/embeddings.json`;
switching the embedding model invalidates the cache.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

from .llm import LLMError

EMBEDDINGS_FILE = "embeddings.json"
_BATCH = 32
_MAX_UNIT_CHARS = 2000

# Candidate-first hybrid retrieval: the free lexical pass supplies at most
# this many units to embed (never the whole project), and RRF_K is the
# smoothing constant of Reciprocal Rank Fusion -- a rank contributes
# 1/(RRF_K + rank), so the merge values agreement between the two rankings
# without needing their scores to be comparable.
CANDIDATE_LIMIT = 50
RRF_K = 60

# Endpoints that cannot serve embeddings (e.g. Lemonade's llama.cpp backend
# without --embeddings returns 501; a wrong embedding_model returns 400/404).
# Remembered per process, keyed by endpoint + model, so later greps fail fast
# instead of re-embedding everything and failing again.
_DISABLED: dict[tuple[str, str], str] = {}


def _endpoint_key(cfg, model: str) -> tuple[str, str]:
    return (str(getattr(cfg, "base_url", "") or "").rstrip("/"), model)


def _permanent_embedding_failure(message: str) -> bool:
    """True for failures that will recur on every retry (unsupported
    endpoint, unknown model, bad request), as opposed to a transient
    connection drop where a later grep might succeed."""
    low = (message or "").lower()
    if "not support" in low and "embedding" in low:
        return True
    m = re.search(r"embeddings request failed \((\d{3})\)", low)
    if m:
        code = int(m.group(1))
        return 400 <= code < 500 or code == 501
    return False


def model_name(cfg) -> str:
    """Embedding model id: the explicit one, else the chat model id."""
    return (getattr(cfg, "embedding_model", "") or cfg.model).strip()


def _cache_path(root) -> Path:
    return Path(root) / ".pedacito" / EMBEDDINGS_FILE


def _load_cache(root, model: str) -> dict[str, list[float]]:
    """Load cached vectors for `model`; a different model starts fresh."""
    p = _cache_path(root)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("model") != model:
        return {}
    vectors = data.get("vectors") or {}
    return {k: v for k, v in vectors.items()
            if isinstance(v, list) and v}


def _save_cache(root, model: str, vectors: dict[str, list[float]]) -> None:
    p = _cache_path(root)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"model": model, "vectors": vectors}),
                     encoding="utf-8")
    except OSError as e:                       # pragma: no cover - disk full
        print(f"[semantic] could not save embedding cache: {e}", file=sys.stderr)


def _unit_text(chunk) -> str:
    """Text embedded for one unit: identity (file/qualname/signature) first, so
    a query can match a unit whose body never uses the word, then a capped body."""
    bits = [chunk.file, chunk.qualname, chunk.signature or ""]
    if chunk.docline:
        bits.append(chunk.docline)
    if chunk.summary:
        bits.append(chunk.summary)
    src = (chunk.source or "").strip()
    if src:
        bits.append(src[:_MAX_UNIT_CHARS])
    return "\n".join(b for b in bits if b)


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    # Six decimals keeps the on-disk cache compact without changing ranking.
    return [round(x / norm, 6) for x in vector] if norm else vector


def _ensure(index, cfg, client, verbose: bool = False, units=None
           ) -> tuple[dict[str, list[float]], str]:
    """Return (vectors_by_sha, model), embedding any missing units.

    `units` restricts the embedding work to one candidate set: only those
    units are hydrated and embedded, so a hybrid lookup costs at most K
    unit embeddings plus the query, never a pass over the whole index.
    None means every indexed unit (the standalone `search()` mode)."""
    model = model_name(cfg)
    key = _endpoint_key(cfg, model)
    if key in _DISABLED:
        raise LLMError(_DISABLED[key])
    vectors = _load_cache(index.root, model)
    # Source is hydrated lazily: only units with no cached vector are read.
    missing = []
    for c in (index.chunks if units is None else units):
        if c.kind == "module" or not c.sha or c.sha in vectors:
            continue
        hydrate = getattr(index, "hydrate", None)
        if hydrate is not None:
            hydrate(c)
        if (c.source or "").strip():
            missing.append(c)
    if not missing:
        return vectors, model
    if verbose:
        print(f"[semantic] embedding {len(missing)} unit(s) with {model} "
              f"(cached afterwards)", file=sys.stderr)
    for start in range(0, len(missing), _BATCH):
        group = missing[start:start + _BATCH]
        try:
            embedded = client.embed([_unit_text(c) for c in group])
        except LLMError as e:
            if _permanent_embedding_failure(str(e)):
                _DISABLED[key] = (
                    "semantic search disabled for this run: the server "
                    f"cannot embed with {model} ({str(e)[:240]}). Start the "
                    "server with embeddings support (llama.cpp: "
                    "`--embeddings`) or set semantic_search = false.")
            raise
        if len(embedded) != len(group):
            raise LLMError(
                f"Embeddings server returned {len(embedded)} vectors for "
                f"{len(group)} texts.")
        for chunk, vector in zip(group, embedded):
            if vector:
                vectors[chunk.sha] = _normalise(vector)
    # Prune vectors for units that no longer exist.
    current = {c.sha for c in index.chunks if c.sha}
    vectors = {sha: v for sha, v in vectors.items() if sha in current}
    _save_cache(index.root, model, vectors)
    return vectors, model


def search(index, cfg, client, query: str, top_k: int | None = None,
           verbose: bool = False) -> list[tuple[float, object]]:
    """Return [(score, chunk)] for the units closest to `query` by meaning."""
    q = (query or "").strip()
    if not q or not getattr(client, "embed", None):
        return []
    vectors, _ = _ensure(index, cfg, client, verbose=verbose)
    if not vectors:
        return []
    embedded = client.embed([q])
    if not embedded or not embedded[0]:
        return []
    qv = _normalise(embedded[0])
    min_score = float(getattr(cfg, "semantic_min_score", 0.2) or 0.0)
    scored: list[tuple[float, object]] = []
    for chunk in index.chunks:
        if chunk.kind == "module":
            continue
        vector = vectors.get(chunk.sha)
        if not vector or len(vector) != len(qv):
            continue
        score = sum(a * b for a, b in zip(qv, vector))
        if score >= min_score:
            scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    k = int(top_k or getattr(cfg, "semantic_top_k", 5) or 5)
    return scored[:k]


def hybrid_search(index, cfg, client, query: str, candidates,
                  top_k: int | None = None,
                  verbose: bool = False) -> list[tuple[float, object, float]]:
    """Candidate-first hybrid retrieval with Reciprocal Rank Fusion.

    `candidates` is the bounded top-K short-list from the free lexical pass
    (BM25 over names/summaries/sidecars, symbol names, call graph), in that
    rank order. Only the query and those candidates are hydrated and
    embedded -- the whole index is never embedded for one no-hit grep. The
    lexical ranking and the embedding ranking are then merged with RRF:

        rrf(c) = 1 / (RRF_K + lex_rank) + 1 / (RRF_K + sem_rank)

    RRF needs only the ranks, not comparable scores, so a cosine of 0.83 and
    a BM25 of 11.4 fuse cleanly. Units whose cosine falls below
    `semantic_min_score` are dropped: this list is the by-meaning answer,
    and a unit with no semantic signal is better served by the plain
    lexical suggestions than by an embedding vote it never earned.

    Returns [(rrf, chunk, cosine)], best first."""
    q = (query or "").strip()
    if not q or not getattr(client, "embed", None):
        return []
    units = [c for c in candidates or []
             if getattr(c, "sha", "") and getattr(c, "kind", "") != "module"]
    units = units[:CANDIDATE_LIMIT]
    if not units:
        return []
    vectors, _ = _ensure(index, cfg, client, verbose=verbose, units=units)
    embedded = client.embed([q])
    if not embedded or not embedded[0]:
        return []
    qv = _normalise(embedded[0])
    min_score = float(getattr(cfg, "semantic_min_score", 0.2) or 0.0)
    semantic: list[tuple[float, int, object]] = []
    for rank, c in enumerate(units):
        vector = vectors.get(c.sha)
        if not vector or len(vector) != len(qv):
            continue
        cos = sum(a * b for a, b in zip(qv, vector))
        if cos >= min_score:
            semantic.append((cos, rank, c))
    if not semantic:
        return []
    semantic.sort(key=lambda item: (-item[0], item[1]))
    sem_pos = {id(c): pos for pos, (_cos, _r, c) in enumerate(semantic)}
    cos_by_id = {id(c): cos for cos, _r, c in semantic}
    fused: list[tuple[float, int, object]] = []
    for lex_rank, c in enumerate(units):
        pos = sem_pos.get(id(c))
        if pos is None:
            continue
        rrf = (1.0 / (RRF_K + lex_rank + 1)
               + 1.0 / (RRF_K + pos + 1))
        fused.append((rrf, lex_rank, c))
    fused.sort(key=lambda item: (-item[0], item[1]))
    k = int(top_k or getattr(cfg, "semantic_top_k", 5) or 5)
    return [(rrf, c, cos_by_id[id(c)]) for rrf, _lr, c in fused[:k]]


def render(query: str, matches) -> str:
    """Render search results as locations, not source.

    The model still has to `read` the chosen unit before editing, so quote
    only what identifies it (path, range, symbol, summary). `matches` items
    are (score, chunk) from search(), or (rrf, chunk, cosine) from
    hybrid_search()."""
    if not matches:
        return f"No semantic matches for: {query}"
    out = [f"Semantic matches for: {query}"]
    for item in matches:
        score, c = item[0], item[1]
        hybrid = len(item) > 2
        cos = item[2] if hybrid else score
        extra = f", fused rank {score:.3f}" if hybrid else ""
        out.append(f"  {c.file}:{c.start}-{getattr(c, 'scope_end', 0) or c.end}"
                   f"  {c.qualname}  (score {cos:.2f}{extra})")
        detail = c.summary or c.docline
        if detail:
            out.append(f"      {detail}")
    out.append("[These are locations by meaning, not exact text. `read` a "
               "region before editing or quoting it.]")
    return "\n".join(out)
