"""AST-based source chunking for Python files.

Usage
-----
    from .chunker import chunk_source

    chunks = chunk_source("src/app.py", source_text, class_split_lines=60)
    for c in chunks:
        print(c.qualname, c.start, c.end, c.signature)

Splits one Python file into semantic units -- module preamble, top-level
functions, classes, and (for large classes) individual methods -- rather than
fixed line windows, so a chunk never starts or ends mid-construct. Every chunk
carries its exact 1-based inclusive line range, which is what the rest of the
system uses to read the corresponding source back later.

Static facts (signatures, decorators, docstrings, imports, call names) are
extracted here from the AST and are never asked of the LLM: the parser is
exact and free, so only a chunk's one-line summary is worth spending a model
call on (see index.py).
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field, asdict

# AST node types treated as "scopes" worth chunking on their own.
DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
SCOPE_NODES = DEF_NODES + (ast.ClassDef,)


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# AST extraction helpers (internal)
# --------------------------------------------------------------------------
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
    """Collect fully-qualified names for every top-level import statement."""
    out: list[str] = []
    for n in tree.body:
        if isinstance(n, ast.Import):
            out += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            mod = ("." * (n.level or 0)) + (n.module or "")
            out += [f"{mod}.{a.name}" if mod else a.name for a in n.names]
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
    """Last-resort line windows for one oversized statement.

    A single 5,000-line dict literal cannot be split at a statement boundary.
    Line windows are not independently parseable, but they are a better
    retrieval unit than one chunk that the model will never read. `overlap`
    gives the next window a few lines of context.
    """
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


def _split_statements(statements: list, lines: list[str], first_line: int,
                      last_line: int, max_lines: int,
                      max_chars: int, max_tokens: int = 0,
                      overlap_lines: int = 0
                      ) -> list[tuple[int, int, list]] | None:
    """Group top-level statements into retrieval-sized parts.

    Returns None when the whole span already fits. Each part is
    (start, end, statements). Parts are contiguous: comments and blank lines
    between statements belong to the preceding part. A single oversized
    statement falls back to overlapping line windows so a giant data literal
    is still retrievable; normal code is never cut inside an expression.
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
        s = _first_line(stmt)
        e = stmt.end_lineno or s
        if cur and s > cur_end and _section_start(lines, s) \
                and _half_budget(lines, cur_start, cur_end, max_lines,
                                 max_chars, max_tokens):
            # Prefer a blank-line/comment boundary over a hard budget edge:
            # the summary and the model both reason better about a section
            # that starts where the author started it.
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
    # Make the parts contiguous: a comment or blank line between two
    # statements belongs to the preceding part. The last part owns the scope's
    # closing lines. Optional overlap gives a continuation part its context.
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


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def chunk_source(rel_path: str, src: str, class_split_lines: int = 60,
                 max_unit_lines: int = 220, max_unit_chars: int = 12000,
                 max_unit_tokens: int = 3000,
                 unit_overlap_lines: int = 0) -> list[Chunk]:
    """Chunk one Python file's source into a list of Chunk objects.

    Produces, in order: a module-preamble chunk (imports and top-level code
    before the first def/class), one chunk per top-level function, one chunk
    per class (whole, or split into a classheader chunk plus one chunk per
    method if the class exceeds `class_split_lines`), and a trailing chunk
    for any code after the last top-level def/class (e.g. a __main__ guard).

    Functions/methods and preamble/trailing code larger than `max_unit_lines`
    or `max_unit_chars` are split at top-level statement boundaries into
    `name#2`, `name#3`, ... parts. A 2,000-line function otherwise forces a
    small model to read the whole thing (or none of it), and reading it once
    can consume most of the gathered-source budget. Splitting never cuts
    inside an expression; a single oversized statement stays whole.

    Raises SyntaxError if `src` is not valid Python.
    """
    tree = ast.parse(src)
    lines = src.splitlines()
    total = len(lines)
    chunks: list[Chunk] = []
    imports = _imports(tree)

    top_scopes = [n for n in tree.body if isinstance(n, SCOPE_NODES)]

    # --- module preamble: imports, constants, anything above the first def ---
    first_def_line = min((_first_line(n) for n in top_scopes), default=total + 1)
    if first_def_line > 1:
        end = min(first_def_line - 1, total)
        body = _slice(lines, 1, end)
        if body.strip():
            # Only nodes actually in the preamble -- walking the whole module
            # here would attribute every call in every function to this chunk.
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

    # --- helper: build and append one Chunk for a function/class/method node ---
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

    # --- top-level functions and classes ---
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
            # Small class stays whole, but record its methods so symbol lookup
            # can still point at it.
            add(node, node.name, "class", members=[m.name for m in methods])
            continue

        # Large class: split into a header chunk (class line, docstring, class
        # attributes) plus one chunk per method, so a 900-line class never has
        # to be read into context whole.
        header_end = min(_first_line(m) for m in methods) - 1
        header_end = max(header_end, start)
        # scope_end carries the full class span. Without it a file summary
        # would report a 500-line class as a 7-line range.
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

    # --- trailing top-level code after the last scope (main guards, etc.) ---
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

    chunks.sort(key=lambda c: c.start)
    return chunks
