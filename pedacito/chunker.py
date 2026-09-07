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
from pathlib import Path

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


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def chunk_source(rel_path: str, src: str, class_split_lines: int = 60) -> list[Chunk]:
    """Chunk one Python file's source into a list of Chunk objects.

    Produces, in order: a module-preamble chunk (imports and top-level code
    before the first def/class), one chunk per top-level function, one chunk
    per class (whole, or split into a classheader chunk plus one chunk per
    method if the class exceeds `class_split_lines`), and a trailing chunk
    for any code after the last top-level def/class (e.g. a __main__ guard).

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
        body = _slice(lines, s, e)
        chunks.append(Chunk(
            file=rel_path, qualname=qualname, kind=kind, start=s, end=e,
            scope_end=scope_end,
            signature=_signature(node), decorators=_decorators(node),
            docline=_docline(node), calls=_calls(node), members=members or [],
            source=body, sha=_sha(body),
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
            # Small class stays whole, but record its methods so symbol lookup
            # can still point at it.
            add(node, node.name, "class", members=[m.name for m in methods])
            continue

        # Large class: split into a header chunk (class line, docstring, class
        # attributes) plus one chunk per method, so a 900-line class never has
        # to be read into context whole.
        header_end = min(_first_line(m) for m in methods) - 1
        # scope_end carries the full class span. Without it a file summary
        # would report a 500-line class as a 7-line range.
        add(node, node.name, "classheader", start=start, end=max(header_end, start),
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
                chunks.append(Chunk(
                    file=rel_path, qualname="__trailing__", kind="module",
                    start=last_end + 1, end=total, calls=_calls(*tail),
                    source=body, sha=_sha(body),
                ))

    chunks.sort(key=lambda c: c.start)
    return chunks
