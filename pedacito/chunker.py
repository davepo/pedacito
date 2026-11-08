"""AST-based source chunking for Python files.

    chunks = chunk_source("src/app.py", source_text, class_split_lines=60)

Splits a file into semantic units (module preamble, functions, classes, and
the methods of large classes) rather than fixed line windows. Every chunk
carries its exact 1-based inclusive line range. Static facts (signatures,
decorators, docstrings, imports, call names) come from the AST; only the
one-line summaries are LLM-generated (see index.py).

Two retrieval facts are computed at chunking time and PERSISTED with the
chunk, so lexical ranking never depends on whether a loaded index has read
the file from disk yet:

* `lex` -- a bounded lexical sidecar (split identifier tokens, docstring and
  comment words, string-literal words, config/dict keys) that search.py
  scores as the BM25 body field;
* `parent` / `part_of` / `part` -- parent-child and continuation lineage, so
  readers can auto-merge the `name#2`, `name#3`, ... fragments of a split
  unit and print a class/module breadcrumb once without duplicating source.

This module also holds the splitting helpers shared with chunker_ts.py.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, field, asdict

# AST node types treated as "scopes" worth chunking on their own.
DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
SCOPE_NODES = DEF_NODES + (ast.ClassDef,)


@dataclass
class Chunk:
    """One indexed unit of source: a module preamble, function, class, or
    method, with its exact line range and the static facts extracted from it."""

    file: str          # path relative to project root
    qualname: str      # "module", "load_config", "Server.start"
    kind: str          # module | function | class | method | classheader
    start: int         # 1-based inclusive
    end: int           # 1-based inclusive
    scope_end: int = 0   # for classheader: end of the whole class, not the header
    signature: str = ""
    decorators: list[str] = field(default_factory=list)
    docline: str = ""          # first line of docstring, if any
    calls: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    members: list[str] = field(default_factory=list)  # method names, for whole-class chunks
    lex: list[str] = field(default_factory=list)  # persisted lexical sidecar (see lex_sidecar)
    parent: str = ""     # enclosing definition's qualname ("Server.start" -> "Server")
    part_of: str = ""    # base qualname when this chunk continues a split unit ("big" for "big#2")
    part: int = 1        # 1-based index within a split unit (continuation metadata)
    source: str = ""           # not persisted in the compact index
    sha: str = ""
    summary: str = ""

    @property
    def key(self) -> str:
        """Globally unique identifier for this chunk: "file::qualname"."""
        return f"{self.file}::{self.qualname}"

    @property
    def nlines(self) -> int:
        """Number of source lines this chunk spans."""
        return self.end - self.start + 1

    def to_record(self) -> dict:
        """Serialise to a plain dict for JSON storage, dropping the source text
        (which is re-read from disk on load rather than persisted)."""
        d = asdict(self)
        d.pop("source", None)
        return d


def split_lines(src: str) -> list[str]:
    """Split on "\\n" only, like the file reader and the parsers' line
    numbering (`str.splitlines` also breaks on form feeds, U+2028, etc., which
    would shift every later line range)."""
    lines = src.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _sha(text: str) -> str:
    """Short content hash used to detect whether a chunk's text has changed."""
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def _slice(lines: list[str], start: int, end: int) -> str:
    """Return the 1-based inclusive line range [start, end] joined as text."""
    return "\n".join(lines[start - 1 : end])


def _first_line(node: ast.AST) -> int:
    """Real first line of a node, counting decorators (node.lineno points at
    the `def`/`class` keyword, not the first decorator above it)."""
    decs = getattr(node, "decorator_list", [])
    if decs:
        return min(d.lineno for d in decs)
    return node.lineno


def _signature(node: ast.AST) -> str:
    """Render a function or class node's signature as source-like text, e.g.
    "def foo(x, y=1) -> int" or "class Foo(Base)"."""
    if isinstance(node, DEF_NODES):
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        try:
            args = ast.unparse(node.args)
        except Exception:
            args = "..."
        ret = ""
        if node.returns is not None:
            try:
                ret = " -> " + ast.unparse(node.returns)
            except Exception:
                pass
        return f"{prefix} {node.name}({args}){ret}"
    if isinstance(node, ast.ClassDef):
        bases = []
        for b in list(node.bases) + list(node.keywords):
            try:
                bases.append(ast.unparse(b))
            except Exception:
                pass
        return f"class {node.name}" + (f"({', '.join(bases)})" if bases else "")
    return ""


def _decorators(node: ast.AST) -> list[str]:
    """Return a node's decorators as source-like strings, e.g. ["@staticmethod"]."""
    out = []
    for d in getattr(node, "decorator_list", []):
        try:
            out.append("@" + ast.unparse(d))
        except Exception:
            pass
    return out


def _docline(node: ast.AST) -> str:
    """Return the first line of a node's docstring, truncated to 160 chars."""
    try:
        doc = ast.get_docstring(node)
    except Exception:
        doc = None
    if not doc:
        return ""
    return doc.strip().splitlines()[0][:160]


def _calls(*nodes: ast.AST) -> list[str]:
    """Collect names invoked via function calls inside the given nodes.
    Attribute calls (obj.method()) reduce to just the attribute name; this is
    an approximation sufficient to power a `callers` lookup without a full
    name resolver."""
    names: set[str] = set()
    for node in nodes:
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                f = n.func
                if isinstance(f, ast.Name):
                    names.add(f.id)
                elif isinstance(f, ast.Attribute):
                    names.add(f.attr)
    return sorted(names)


def _imports(tree: ast.Module) -> list[str]:
    """Collect fully-qualified names for every import statement in the module,
    including function-local, class-body, and `try:`/`if:` guarded ones (a
    top-level-only scan hid every lazily imported dependency)."""
    out: list[str] = []
    seen: set[str] = set()
    for n in ast.walk(tree):
        names: list[str] = []
        if isinstance(n, ast.Import):
            names = [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            mod = ("." * (n.level or 0)) + (n.module or "")
            names = [(f"{mod}{a.name}" if mod.endswith(".") else
                      f"{mod}.{a.name}") if mod else a.name for a in n.names]
        for name in names:
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


def _estimate_tokens(text: str) -> int:
    """Rough code-token estimate (same 3.6 chars/token ratio as llm.py)."""
    return int(len(text) / 3.6) + 1


def _fits(lines: list[str], start: int, end: int,
          max_lines: int, max_chars: int, max_tokens: int = 0) -> bool:
    """True when a 1-based inclusive line range fits the retrieval budget."""
    if end - start + 1 > max_lines:
        return False
    text = _slice(lines, start, end)
    if len(text) > max_chars:
        return False
    return not (max_tokens and _estimate_tokens(text) > max_tokens)


def _window_parts(lines: list[str], first_line: int, last_line: int,
                  max_lines: int, max_chars: int, max_tokens: int,
                  overlap: int) -> list[tuple[int, int, list]] | None:
    """Last-resort line windows (with `overlap` lines of context) for one
    oversized statement, e.g. a giant dict literal."""
    parts: list[tuple[int, int, list]] = []
    start = first_line
    while start <= last_line:
        end = start
        while end < last_line and _fits(lines, start, end + 1, max_lines,
                                        max_chars, max_tokens):
            end += 1
        parts.append((start, end, []))
        if end >= last_line:
            break
        start = max(start + 1, end + 1 - max(0, overlap))
    return parts if len(parts) > 1 else None


def _stmt_span(stmt) -> tuple[int, int]:
    """(first line incl. decorators, last line) of a Python AST statement."""
    s = _first_line(stmt)
    return s, stmt.end_lineno or s


def _split_statements(statements: list, lines: list[str], first_line: int,
                      last_line: int, max_lines: int,
                      max_chars: int, max_tokens: int = 0,
                      overlap_lines: int = 0, span=_stmt_span
                      ) -> list[tuple[int, int, list]] | None:
    """Group top-level statements into retrieval-sized parts.

    Returns None when the whole span already fits. Each part is
    (start, end, statements), contiguous: comments and blank lines between
    statements belong to the preceding part. A single oversized statement
    falls back to overlapping line windows; normal code is never cut inside an
    expression. `span(stmt)` gives a statement's (start, end) lines (the
    tree-sitter chunker passes its own).
    """
    if _fits(lines, first_line, last_line, max_lines, max_chars, max_tokens):
        return None
    if len(statements) < 2:
        return _window_parts(lines, first_line, last_line, max_lines,
                             max_chars, max_tokens, overlap_lines)
    parts: list[tuple[int, int, list]] = []
    cur_start = first_line
    cur_end = first_line - 1
    cur: list = []
    for stmt in statements:
        s, e = span(stmt)
        if cur and s > cur_end and _section_start(lines, s) \
                and _half_budget(lines, cur_start, cur_end, max_lines,
                                 max_chars, max_tokens):
            # prefer a blank-line/comment boundary over a hard budget edge
            parts.append((cur_start, cur_end, cur))
            cur_start, cur_end, cur = s, e, [stmt]
            continue
        if cur and s > cur_end and \
                not _fits(lines, cur_start, e, max_lines, max_chars, max_tokens):
            parts.append((cur_start, cur_end, cur))
            cur_start, cur_end, cur = s, e, [stmt]
        else:
            cur.append(stmt)
            cur_end = max(cur_end, e)
    if cur:
        parts.append((cur_start, cur_end, cur))
    if len(parts) < 2:
        return None
    # Contiguous parts; the last one owns the scope's closing lines.
    for i in range(len(parts) - 1):
        ps, _pe, pnodes = parts[i]
        parts[i] = (ps, parts[i + 1][0] - 1, pnodes)
    ps, _pe, pnodes = parts[-1]
    parts[-1] = (ps, last_line, pnodes)
    if overlap_lines > 0:
        for i in range(1, len(parts)):
            ps, pe, pnodes = parts[i]
            parts[i] = (max(first_line, ps - overlap_lines), pe, pnodes)
    return parts


def _part_signature(sig: str, index: int, count: int) -> str:
    """Label a split unit so the map/outline makes the continuation visible."""
    return f"{sig}  [part {index}/{count}]" if count > 1 else sig


_COMMENT_PREFIXES = ("#", "//", "/*", "*", "--", "<!--")


def _section_start(lines: list[str], line_no: int) -> bool:
    """True when a statement starts a logical section (blank/comment above)."""
    if line_no <= 1:
        return False
    prev = lines[line_no - 2].strip()
    return prev == "" or prev.startswith(_COMMENT_PREFIXES)


def _half_budget(lines: list[str], start: int, end: int, max_lines: int,
                 max_chars: int, max_tokens: int) -> bool:
    if end - start + 1 >= max(2, int(max_lines * 0.5)):
        return True
    text = _slice(lines, start, end)
    if len(text) >= max(2, int(max_chars * 0.5)):
        return True
    return bool(max_tokens and _estimate_tokens(text) >= max_tokens * 0.5)


# ---------------------------------------------------------------------------
# Persisted lexical sidecar
#
# A bounded, deterministic bag of the words that make a unit FINDABLE by
# lexical search even when its source text is not in memory: split
# identifier tokens, docstring/comment words, string-literal words (error
# messages, prompts, config values) and config/dict keys. Stored on the
# chunk and persisted inside index.json, so search.py can score the body
# field without ever hydrating source from disk -- the ranking is identical
# whether or not a loaded Index has read the file yet (100% hydration-
# stable, and it works with `--no-summaries`). Categories are capped
# separately so an identifier-heavy body cannot push the error strings and
# dict keys -- the words queries actually look for -- out of the budget.
# ---------------------------------------------------------------------------

LEX_LIMIT = 72  # hard cap on persisted terms per chunk

_LEX_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_LEX_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*|\d+")
_LEX_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
# `name:` / "key": at line start (dict/config keys) and quoted inline keys.
_LEX_KEY_RE = re.compile(r"""^\s*(?:["']([A-Za-z_][A-Za-z0-9_]{2,})["']"""
                         r"""|([A-Za-z_][A-Za-z0-9_]{2,}))\s*:""",
                         re.MULTILINE)
_LEX_INLINE_KEY_RE = re.compile(r"""["']([A-Za-z_][A-Za-z0-9_]{2,})["']\s*:""")
_LEX_STR_RE = re.compile(
    r"'''(.*?)'''|\"\"\"(.*?)\"\"\"|'([^'\n]*)'|\"([^\"\n]*)\"", re.DOTALL)

# Python keywords and self-evident names are noise in a lexical index.
_LEX_STOP = frozenset("""
and as assert async await break class continue def del elif else except
finally for from global if import in is lambda nonlocal not or pass raise
return try while with yield self cls none true false
""".split())


def split_ident(name: str) -> list[str]:
    """camelCase/snake_case words of one identifier, lower-cased; fragments
    shorter than 3 chars and Python keywords are dropped. search.tokenize
    applies stopword removal and stemming on top when BM25 is built; the
    sidecar keeps raw split words so an analyzer change never forces a
    re-index of semantics into persisted data."""
    out: list[str] = []
    for word in _LEX_WORD_RE.findall((name or "").replace("_", " ")):
        for part in _LEX_CAMEL_RE.findall(word):
            w = part.lower()
            if len(w) >= 3 and w not in _LEX_STOP:
                out.append(w)
    return out


def lex_sidecar(text: str, limit: int = LEX_LIMIT) -> list[str]:
    """Build the bounded lexical sidecar for one chunk's source text.

    Deterministic (fixed scan order, insertion order kept) and derived only
    from the text passed in -- never from disk state -- so a chunk's
    persisted sidecar is identical at build time and after a reload. Order:
    dict/config keys, string-literal words, comment words, identifier
    tokens, then truncated to `limit`."""
    keys: dict[str, None] = {}
    strings: dict[str, None] = {}
    comments: dict[str, None] = {}
    idents: dict[str, None] = {}

    def put(bucket: dict[str, None], word: str, cap: int) -> None:
        if len(bucket) < cap:
            bucket.setdefault(word, None)

    # 1. config/dict keys: `name:` and "name": (settings modules, schemas).
    for m in _LEX_KEY_RE.finditer(text):
        for w in split_ident(m.group(1) or m.group(2) or ""):
            put(keys, w, 16)
    for m in _LEX_INLINE_KEY_RE.finditer(text):
        for w in split_ident(m.group(1)):
            put(keys, w, 16)
    # 2. string-literal tokens: error messages, prompts, config values.
    for m in _LEX_STR_RE.finditer(text):
        value = next((g for g in m.groups() if g is not None), "")
        for w in _LEX_WORD_RE.findall(value):
            for part in split_ident(w):
                put(strings, part, 32)
    # 3. comment words (docstrings are string literals, captured above).
    for line in text.split("\n"):
        pos = line.find("#")
        if pos < 0:
            continue
        for w in _LEX_WORD_RE.findall(line[pos + 1:]):
            for part in split_ident(w):
                put(comments, part, 20)
    # 4. identifier tokens: names, attributes, kwargs (internal call surface).
    for m in _LEX_IDENT_RE.finditer(text):
        for part in split_ident(m.group(0)):
            put(idents, part, 24)

    terms: dict[str, None] = {}
    for bucket in (keys, strings, comments, idents):
        for w in bucket:
            if len(terms) >= limit:
                return list(terms)
            terms.setdefault(w, None)
    return list(terms)


# qualname of a continuation part: "big#2", "Server.run#3", "module#2"...
_PART_RE = re.compile(r"(.+)#(\d+)$")


def _attach_lex_and_lineage(chunks: list[Chunk]) -> None:
    """Fill the persisted lexical sidecar and parent/continuation metadata
    on every chunk of one file, from the in-memory source only (hydration-
    independent).

    parent: the enclosing definition (a method's class, inherited by its
    continuation parts). part_of/part: which split unit a `name#N` fragment
    continues, so tools can auto-merge adjacent fragments and print the
    class/module breadcrumb once instead of duplicating parent source."""
    byq: dict[str, Chunk] = {}
    for c in chunks:
        byq.setdefault(c.qualname, c)
    for c in chunks:
        m = _PART_RE.fullmatch(c.qualname)
        if m:
            c.part_of, c.part = m.group(1), int(m.group(2))
            base = byq.get(c.part_of)
            if base is not None:
                c.parent = base.parent
        else:
            c.part = 1
        if c.kind == "method" and "." in c.qualname:
            c.parent = c.qualname.rsplit(".", 1)[0]
        c.lex = lex_sidecar(c.source or "")


def chunk_source(rel_path: str, src: str, class_split_lines: int = 60,
                 max_unit_lines: int = 220, max_unit_chars: int = 12000,
                 max_unit_tokens: int = 3000,
                 unit_overlap_lines: int = 0) -> list[Chunk]:
    """Chunk one Python file's source into a list of Chunk objects.

    Produces, in order: a module-preamble chunk (code before the first
    def/class), one chunk per top-level function, one per class (whole, or a
    classheader chunk plus one chunk per method when the class exceeds
    `class_split_lines`), and a trailing chunk for code after the last
    def/class (e.g. a __main__ guard).

    Units larger than the max_unit_* limits are split at statement boundaries
    into `name`, `name#2`, ... parts. Raises SyntaxError on invalid Python.
    """
    tree = ast.parse(src)
    lines = split_lines(src)
    total = len(lines)
    chunks: list[Chunk] = []
    imports = _imports(tree)

    top_scopes = [n for n in tree.body if isinstance(n, SCOPE_NODES)]

    # module preamble: imports, constants, anything above the first def
    first_def_line = min((_first_line(n) for n in top_scopes), default=total + 1)
    if first_def_line > 1:
        end = min(first_def_line - 1, total)
        body = _slice(lines, 1, end)
        if body.strip():
            # only preamble nodes, or every call in the module lands here
            preamble = [n for n in tree.body if (n.end_lineno or 0) <= end]
            parts = _split_statements(preamble, lines, 1, end,
                                      max_unit_lines, max_unit_chars,
                                      max_unit_tokens, unit_overlap_lines)
            if parts:
                for i, (ps, pe, nodes) in enumerate(parts, 1):
                    part_body = _slice(lines, ps, pe)
                    chunks.append(Chunk(
                        file=rel_path,
                        qualname="module" if i == 1 else f"module#{i}",
                        kind="module", start=ps, end=pe, imports=imports,
                        docline=_docline(tree) if i == 1 else "",
                        calls=_calls(*nodes),
                        source=part_body, sha=_sha(part_body),
                    ))
            else:
                chunks.append(Chunk(
                    file=rel_path, qualname="module", kind="module",
                    start=1, end=end, imports=imports,
                    docline=_docline(tree), calls=_calls(*preamble),
                    source=body, sha=_sha(body),
                ))

    def add(node, qualname, kind, start=None, end=None, members=None, scope_end=0):
        """Build and append one Chunk for a function/class/method node."""
        s = start if start is not None else _first_line(node)
        e = end if end is not None else (node.end_lineno or s)
        if kind in ("function", "method") and isinstance(node, DEF_NODES):
            parts = _split_statements(node.body, lines, s, e,
                                      max_unit_lines, max_unit_chars,
                                      max_unit_tokens, unit_overlap_lines)
            if parts:
                sig = _signature(node)
                for i, (ps, pe, nodes) in enumerate(parts, 1):
                    part_body = _slice(lines, ps, pe)
                    chunks.append(Chunk(
                        file=rel_path,
                        qualname=qualname if i == 1 else f"{qualname}#{i}",
                        kind=kind, start=ps, end=pe,
                        signature=_part_signature(sig, i, len(parts)),
                        decorators=_decorators(node) if i == 1 else [],
                        docline=_docline(node) if i == 1 else "",
                        calls=_calls(*nodes), imports=imports,
                        source=part_body, sha=_sha(part_body),
                    ))
                return
        body = _slice(lines, s, e)
        chunks.append(Chunk(
            file=rel_path, qualname=qualname, kind=kind, start=s, end=e,
            scope_end=scope_end,
            signature=_signature(node), decorators=_decorators(node),
            docline=_docline(node), calls=_calls(node), members=members or [],
            imports=imports, source=body, sha=_sha(body),
        ))

    # top-level functions and classes
    for node in top_scopes:
        if isinstance(node, DEF_NODES):
            add(node, node.name, "function")
            continue

        # ClassDef
        start, end = _first_line(node), node.end_lineno or _first_line(node)
        methods = [m for m in node.body if isinstance(m, DEF_NODES)]
        if (end - start + 1) <= class_split_lines or not methods:
            if not methods:
                parts = _split_statements(node.body, lines, start, end,
                                          max_unit_lines, max_unit_chars,
                                          max_unit_tokens, unit_overlap_lines)
                if parts:
                    sig = _signature(node)
                    for i, (ps, pe, nodes) in enumerate(parts, 1):
                        part_body = _slice(lines, ps, pe)
                        chunks.append(Chunk(
                            file=rel_path,
                            qualname=(node.name if i == 1
                                      else f"{node.name}#{i}"),
                            kind="class", start=ps, end=pe,
                            signature=_part_signature(sig, i, len(parts)),
                            decorators=_decorators(node) if i == 1 else [],
                            docline=_docline(node) if i == 1 else "",
                            calls=_calls(*nodes), imports=imports,
                            source=part_body, sha=_sha(part_body),
                        ))
                    continue
            # small class stays whole; members let symbol lookup find it
            add(node, node.name, "class", members=[m.name for m in methods])
            continue

        # large class: header chunk (class line, docstring, attributes) plus
        # one chunk per method
        header_end = min(_first_line(m) for m in methods) - 1
        header_end = max(header_end, start)
        # scope_end carries the full class span
        header_nodes = [n for n in node.body
                        if (n.end_lineno or 0) <= header_end]
        header_parts = _split_statements(header_nodes, lines, start,
                                         header_end, max_unit_lines,
                                         max_unit_chars, max_unit_tokens,
                                         unit_overlap_lines)
        if header_parts:
            sig = _signature(node)
            for i, (ps, pe, nodes) in enumerate(header_parts, 1):
                part_body = _slice(lines, ps, pe)
                chunks.append(Chunk(
                    file=rel_path,
                    qualname=(node.name if i == 1
                              else f"{node.name}#{i}"),
                    kind="classheader", start=ps, end=pe,
                    scope_end=end if i == 1 else 0,
                    signature=_part_signature(sig, i, len(header_parts)),
                    decorators=_decorators(node) if i == 1 else [],
                    docline=_docline(node) if i == 1 else "",
                    calls=_calls(*nodes), imports=imports,
                    members=[m.name for m in methods] if i == 1 else [],
                    source=part_body, sha=_sha(part_body),
                ))
        else:
            add(node, node.name, "classheader", start=start, end=header_end,
                members=[m.name for m in methods], scope_end=end)
        for m in methods:
            add(m, f"{node.name}.{m.name}", "method")

    # trailing top-level code after the last scope (main guards, etc.)
    if top_scopes:
        last_end = max(n.end_lineno or 0 for n in top_scopes)
        if last_end < total:
            body = _slice(lines, last_end + 1, total)
            if body.strip():
                tail = [n for n in tree.body if n.lineno > last_end]
                parts = _split_statements(tail, lines, last_end + 1, total,
                                          max_unit_lines, max_unit_chars,
                                          max_unit_tokens, unit_overlap_lines)
                if parts:
                    for i, (ps, pe, nodes) in enumerate(parts, 1):
                        part_body = _slice(lines, ps, pe)
                        chunks.append(Chunk(
                            file=rel_path,
                            qualname=("__trailing__" if i == 1
                                      else f"__trailing__#{i}"),
                            kind="module", start=ps, end=pe, imports=imports,
                            calls=_calls(*nodes),
                            source=part_body, sha=_sha(part_body),
                        ))
                else:
                    chunks.append(Chunk(
                        file=rel_path, qualname="__trailing__", kind="module",
                        start=last_end + 1, end=total, imports=imports,
                        calls=_calls(*tail), source=body, sha=_sha(body),
                    ))

    _attach_lex_and_lineage(chunks)
    chunks.sort(key=lambda c: c.start)
    return chunks
