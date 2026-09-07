"""Tree-sitter based source chunking for JavaScript, TypeScript, and Ruby.

Usage
-----
    from .chunker_ts import chunk_source_ts, LANG_BY_EXT, TS_AVAILABLE

    chunks = chunk_source_ts("src/app.ts", source_text, "typescript",
                              class_split_lines=60)
    for c in chunks:
        print(c.qualname, c.start, c.end, c.signature)

Mirrors chunker.py's contract exactly -- same Chunk dataclass, same kind
vocabulary ("module", "function", "class", "classheader", "method"), same
line-range guarantees -- so nothing downstream (index.py's map rendering,
tools.py's outline/read/callers) needs to know or care that a file was
parsed by tree-sitter instead of the `ast` module. Only index.py's two
dispatch points (build/reindex_file) decide which chunker to call, based on
file extension.

Differences from Python's AST chunking, driven by how the grammar actually
shapes the tree (verified against parsed samples, not assumed from docs):

  - `export` (and `export default`) wraps the thing it exports as a parent
    node, so scope detection unwraps one level before classifying. A bare
    `export default someIdentifier;` (re-exporting an existing name) isn't a
    new scope and is left in the surrounding preamble/trailing text.
  - A top-level `const`/`let` is only function-like if its initializer is a
    function or arrow function (`export const f = () => ...`); otherwise
    it's treated like any other module-level statement.
  - There is no docstring literal. A JSDoc-style `/** ... */` comment is a
    preceding *sibling* node, not something inside the scope, so `docline`
    is pulled from the nearest immediately-preceding comment sibling.
  - TypeScript's `interface`/`type`/`enum` declarations have no Python
    analogue; they're chunked like a small whole class (kind="class"),
    with their property/member names recorded the same way a class's
    method names are.

Requires the optional `tree-sitter` + `tree-sitter-language-pack` packages.
If they aren't installed, `TS_AVAILABLE` is False and callers should fall
back to reference-style chunking (see references.py) rather than fail.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .chunker import Chunk

try:
    from tree_sitter_language_pack import get_parser
    TS_AVAILABLE = True
except ImportError:                                    # pragma: no cover
    get_parser = None
    TS_AVAILABLE = False

# File extension -> tree-sitter grammar name. .ts deliberately does not
# parse JSX (confirmed: the "typescript" grammar rejects JSX syntax with a
# parse error), which is exactly the disambiguation a .ts/.tsx split implies.
LANG_BY_EXT = {
    ".js": "javascript",
    ".jsx": "javascript",   # the plain "javascript" grammar parses JSX fine
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rb": "ruby",
}

# Node types that introduce a function-like scope, keyed by nothing more
# than their tree-sitter type name -- these are the same across the JS,
# TypeScript, and TSX grammars.
FUNCTION_NODES = {
    "function_declaration", "generator_function_declaration",
}
CLASS_NODES = {"class_declaration"}
# TypeScript-only: chunked like small whole classes (see module docstring).
TS_DECL_NODES = {
    "interface_declaration", "type_alias_declaration", "enum_declaration",
}
METHOD_NODES = {"method_definition"}

# Ruby node types (see chunk_source_ruby's docstring for why these need
# their own recursive handling instead of reusing chunk_source_ts).
RUBY_CONTAINER_NODES = {"class", "module"}
RUBY_METHOD_NODES = {"method", "singleton_method"}
RUBY_CARVE_NODES = RUBY_CONTAINER_NODES | RUBY_METHOD_NODES
RUBY_IMPORT_CALLS = {"require", "require_relative", "load"}

_parsers: dict[str, "object"] = {}


# --------------------------------------------------------------------------
# Small internal helpers
# --------------------------------------------------------------------------
def _sha(text: str) -> str:
    """Short content hash, same scheme as chunker.py, kept local so this
    module has no private cross-imports."""
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def _get_parser(language: str):
    """Return a cached tree-sitter Parser for `language`, building it once."""
    p = _parsers.get(language)
    if p is None:
        p = get_parser(language)
        _parsers[language] = p
    return p


def _text(node, src: bytes) -> str:
    """Decode a node's exact source slice."""
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _line_span(node) -> tuple[int, int]:
    """1-based inclusive (start, end) line range for a node."""
    return node.start_point.row + 1, node.end_point.row + 1


def _unwrap_export(node):
    """If `node` is an export_statement (plain or `export default`), return
    the declaration it wraps, or None if it's a re-export/bare export with
    no declaration of its own (e.g. `export default foo;`, `export {a, b};`).
    Otherwise return `node` unchanged."""
    if node.type != "export_statement":
        return node
    for child in node.children:
        if child.type in (FUNCTION_NODES | CLASS_NODES | TS_DECL_NODES |
                           {"lexical_declaration", "variable_declaration"}):
            return child
    return None


def _leading_comment_doc(node) -> str:
    """First line of the nearest immediately-preceding sibling comment, i.e.
    a JSDoc block directly above this scope. Falls back to "" if the
    previous sibling isn't a comment (JS/TS have no docstring literal)."""
    parent = node.parent
    if parent is None:
        return ""
    # Walk the parent's children to find node's own preceding sibling,
    # unwrapping through export_statement since the comment sits above the
    # `export`, not above the declaration it wraps.
    target = node
    while target.parent is not None and target.parent.type != parent.type:
        break
    siblings = list(parent.children)
    try:
        idx = siblings.index(node)
    except ValueError:
        # node was unwrapped from an export_statement; find that instead
        idx = next((i for i, s in enumerate(siblings)
                    if s.type == "export_statement" and node in s.children), -1)
    if idx <= 0:
        return ""
    prev = siblings[idx - 1]
    if prev.type != "comment":
        return ""
    text = prev.text.decode("utf-8", "replace")
    # Strip /** ... */, //, or Ruby's # decoration, keep the first content
    # line. strip("/*#") handles all three comment styles this function
    # is used for (JS/TS block and line comments, Ruby line comments).
    text = text.strip("/*#").strip()
    for line in text.splitlines():
        line = line.strip().lstrip("*#").strip()
        if line:
            return line[:160]
    return ""


def _decorators(node, src: bytes) -> list[str]:
    """TS decorators (@Component, @Injectable(...)) preceding a class or
    method, rendered as source-like strings."""
    out = []
    for child in node.children:
        if child.type == "decorator":
            out.append(_text(child, src))
    return out


def _signature_of_function(node, src: bytes, name: str) -> str:
    """Render an approximate signature: the declared keywords plus the
    parameter list, taken verbatim from source rather than reconstructed --
    tree-sitter gives exact byte ranges, so there's no unparse step needed."""
    params = node.child_by_field_name("parameters")
    ret = node.child_by_field_name("return_type")
    is_async = any(c.type == "async" for c in node.children)
    is_gen = node.type == "generator_function_declaration"
    prefix = ("async " if is_async else "") + "function" + ("*" if is_gen else "")
    param_text = _text(params, src) if params else "()"
    ret_text = _text(ret, src) if ret else ""
    return f"{prefix} {name}{param_text}{ret_text}".strip()


def _signature_of_class(node, src: bytes, name: str) -> str:
    heritage = node.child_by_field_name("heritage")
    h = f" {_text(heritage, src)}" if heritage else ""
    return f"class {name}{h}"


def _walk_calls(node, out: set[str]) -> None:
    """Collect names invoked via call/new expressions under `node`. Bare
    identifiers are taken as-is; member calls (obj.method()) reduce to the
    property name -- the same bare-name approximation chunker.py uses for
    Python, kept consistent across languages."""
    if node.type == "call_expression":
        fn = node.child_by_field_name("function")
        if fn is not None:
            if fn.type == "identifier":
                out.add(fn.text.decode())
            elif fn.type == "member_expression":
                prop = fn.child_by_field_name("property")
                if prop is not None:
                    out.add(prop.text.decode())
    elif node.type == "new_expression":
        ctor = node.child_by_field_name("constructor")
        if ctor is not None and ctor.type == "identifier":
            out.add(ctor.text.decode())
    for child in node.children:
        _walk_calls(child, out)


def _module_label(spec: str) -> str:
    """Collapse an import specifier to a short display label for the file
    card's "imports:" line -- mirrors what index.py's rendering does for
    Python's dotted imports (take the meaningful top-level piece), since a
    raw relative path like "./foo" would otherwise render as an empty
    string once index.py strips a leading dot."""
    spec = spec.rstrip("/")
    if spec.startswith("."):
        return spec.rsplit("/", 1)[-1] or spec
    return spec


def _imports(root, src: bytes) -> list[str]:
    """Collect module specifiers from every top-level import_statement."""
    out: list[str] = []
    for node in root.children:
        if node.type != "import_statement":
            continue
        for child in node.children:
            if child.type == "string":
                frag = next((c for c in child.children
                             if c.type == "string_fragment"), None)
                if frag is not None:
                    out.append(_module_label(frag.text.decode()))
    return out


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def chunk_source_ts(rel_path: str, src: str, language: str,
                     class_split_lines: int = 60) -> list[Chunk]:
    """Chunk one JS/TS/TSX file's source into a list of Chunk objects.

    Produces, in order: a module-preamble chunk (imports and any top-level
    statements before the first function/class/interface/type/enum), one
    chunk per top-level function (including arrow functions assigned to a
    top-level const/let), one chunk per class or TS interface/type/enum
    (whole, or split into a classheader chunk plus one chunk per method if
    a class exceeds `class_split_lines`), and a trailing chunk for any code
    after the last top-level scope.

    Never raises on malformed source: tree-sitter is error-tolerant by
    design (confirmed against deliberately broken input) and produces a
    best-effort tree even around a syntax error, rather than refusing to
    parse the way `ast.parse` does for Python. A badly broken file typically
    degrades to fewer/coarser chunks (e.g. the whole file as one "module"
    chunk) rather than failing outright -- there is no equivalent of
    chunk_source's SyntaxError to catch here.
    """
    if not TS_AVAILABLE:
        raise RuntimeError(
            "tree-sitter and tree-sitter-language-pack are required for "
            "JS/TS indexing; install the 'js' extra or fall back to "
            "reference-style chunking for this file.")

    parser = _get_parser(language)
    src_bytes = src.encode("utf-8", "replace")
    tree = parser.parse(src_bytes)
    root = tree.root_node
    lines = src.splitlines()
    total = len(lines)
    chunks: list[Chunk] = []
    imports = _imports(root, src_bytes)

    # --- classify each top-level statement, unwrapping export wrappers ---
    scopes: list[tuple[object, object]] = []   # (outer_node, inner_decl_node)
    for outer in root.children:
        inner = _unwrap_export(outer)
        if inner is None:
            continue
        if inner.type in FUNCTION_NODES:
            scopes.append((outer, inner))
        elif inner.type in CLASS_NODES:
            scopes.append((outer, inner))
        elif inner.type in TS_DECL_NODES:
            scopes.append((outer, inner))
        elif inner.type in ("lexical_declaration", "variable_declaration"):
            # Only function-like if a declarator's value is a function.
            for decl in inner.children:
                if decl.type != "variable_declarator":
                    continue
                value = decl.child_by_field_name("value")
                if value is not None and value.type in (
                        "arrow_function", "function", "function_expression"):
                    scopes.append((outer, inner))
                    break

    if not scopes:
        if src.strip():
            chunks.append(Chunk(
                file=rel_path, qualname="module", kind="module",
                start=1, end=total, imports=imports,
                calls=sorted(_calls_of(root, src_bytes)),
                source=src, sha=_sha(src),
            ))
        return chunks

    # --- module preamble: everything before the first scope's outer node ---
    first_start = scopes[0][0].start_point.row + 1
    if first_start > 1:
        end = first_start - 1
        body = "\n".join(lines[:end])
        if body.strip():
            calls: set[str] = set()
            for node in root.children:
                if node.start_point.row + 1 > end:
                    break
                _walk_calls(node, calls)
            chunks.append(Chunk(
                file=rel_path, qualname="module", kind="module",
                start=1, end=end, imports=imports,
                calls=sorted(calls), source=body, sha=_sha(body),
            ))

    # --- each top-level scope ---
    for outer, inner in scopes:
        start, end = _line_span(outer)
        body = "\n".join(lines[start - 1:end])
        doc = _leading_comment_doc(outer)
        decs = _decorators(outer, src_bytes) + _decorators(inner, src_bytes)

        if inner.type in FUNCTION_NODES:
            name_node = inner.child_by_field_name("name")
            name = name_node.text.decode() if name_node else "<anonymous>"
            calls: set[str] = set()
            _walk_calls(inner, calls)
            chunks.append(Chunk(
                file=rel_path, qualname=name, kind="function",
                start=start, end=end, signature=_signature_of_function(inner, src_bytes, name),
                decorators=decs, docline=doc, calls=sorted(calls),
                source=body, sha=_sha(body),
            ))
            continue

        if inner.type in ("lexical_declaration", "variable_declaration"):
            # Arrow-function-as-const: pull the name from the declarator.
            decl = next(c for c in inner.children if c.type == "variable_declarator")
            name_node = decl.child_by_field_name("name")
            name = name_node.text.decode() if name_node else "<anonymous>"
            value = decl.child_by_field_name("value")
            calls: set[str] = set()
            _walk_calls(value, calls)
            params = value.child_by_field_name("parameters")
            sig = f"const {name} = {_text(params, src_bytes) if params else '()'} => ..."
            chunks.append(Chunk(
                file=rel_path, qualname=name, kind="function",
                start=start, end=end, signature=sig,
                decorators=decs, docline=doc, calls=sorted(calls),
                source=body, sha=_sha(body),
            ))
            continue

        if inner.type in TS_DECL_NODES:
            name_node = inner.child_by_field_name("name")
            name = name_node.text.decode() if name_node else "<anonymous>"
            body_node = inner.child_by_field_name("body")
            members = []
            if body_node is not None:
                for m in body_node.children:
                    nm = m.child_by_field_name("name") if hasattr(m, "child_by_field_name") else None
                    if nm is not None:
                        members.append(nm.text.decode())
            kind_word = inner.type.replace("_declaration", "").replace("_", " ")
            chunks.append(Chunk(
                file=rel_path, qualname=name, kind="class",
                start=start, end=end,
                signature=f"{kind_word} {name}",
                decorators=decs, docline=doc, members=members,
                source=body, sha=_sha(body),
            ))
            continue

        # ClassDef (class_declaration)
        name_node = inner.child_by_field_name("name")
        name = name_node.text.decode() if name_node else "<anonymous>"
        body_node = inner.child_by_field_name("body")
        methods = [m for m in (body_node.children if body_node else [])
                   if m.type in METHOD_NODES]
        if (end - start + 1) <= class_split_lines or not methods:
            member_names = []
            for m in methods:
                nm = m.child_by_field_name("name")
                if nm is not None:
                    member_names.append(nm.text.decode())
            calls: set[str] = set()
            _walk_calls(inner, calls)
            chunks.append(Chunk(
                file=rel_path, qualname=name, kind="class",
                start=start, end=end,
                signature=_signature_of_class(inner, src_bytes, name),
                decorators=decs, docline=doc, members=member_names,
                calls=sorted(calls), source=body, sha=_sha(body),
            ))
            continue

        # Large class: header chunk + one chunk per method, same split
        # strategy as chunker.py.
        member_names = []
        for m in methods:
            nm = m.child_by_field_name("name")
            if nm is not None:
                member_names.append(nm.text.decode())
        first_method_line = min(m.start_point.row + 1 for m in methods)
        header_end = max(first_method_line - 1, start)
        header_body = "\n".join(lines[start - 1:header_end])
        chunks.append(Chunk(
            file=rel_path, qualname=name, kind="classheader",
            start=start, end=header_end, scope_end=end,
            signature=_signature_of_class(inner, src_bytes, name),
            decorators=decs, docline=doc, members=member_names,
            source=header_body, sha=_sha(header_body),
        ))
        for m in methods:
            m_name_node = m.child_by_field_name("name")
            m_name = m_name_node.text.decode() if m_name_node else "<anonymous>"
            m_start, m_end = _line_span(m)
            m_body = "\n".join(lines[m_start - 1:m_end])
            m_calls: set[str] = set()
            _walk_calls(m, m_calls)
            m_params = m.child_by_field_name("parameters")
            m_sig = f"{m_name}{_text(m_params, src_bytes) if m_params else '()'}"
            chunks.append(Chunk(
                file=rel_path, qualname=f"{name}.{m_name}", kind="method",
                start=m_start, end=m_end, signature=m_sig,
                decorators=_decorators(m, src_bytes),
                calls=sorted(m_calls), source=m_body, sha=_sha(m_body),
            ))

    # --- trailing top-level code after the last scope ---
    last_end = scopes[-1][0].end_point.row + 1
    if last_end < total:
        body = "\n".join(lines[last_end:])
        if body.strip():
            calls: set[str] = set()
            for node in root.children:
                if node.start_point.row + 1 > last_end:
                    _walk_calls(node, calls)
            chunks.append(Chunk(
                file=rel_path, qualname="__trailing__", kind="module",
                start=last_end + 1, end=total, calls=sorted(calls),
                source=body, sha=_sha(body),
            ))

    chunks.sort(key=lambda c: c.start)
    return chunks


def _calls_of(node, src: bytes) -> set[str]:
    """Convenience wrapper used only for the "whole file is one chunk"
    (no top-level scopes found) case."""
    out: set[str] = set()
    _walk_calls(node, out)
    return out


# --------------------------------------------------------------------------
# Ruby (recursive containers -- see chunk_source_ruby's docstring)
# --------------------------------------------------------------------------
def _ruby_call_name(node) -> str | None:
    """The `method` field text for a Ruby `call` node (works for both
    `foo(1)` and `obj.foo(1)` -- confirmed the field is present either
    way), or None if `node` isn't a call at all."""
    if node.type != "call":
        return None
    m = node.child_by_field_name("method")
    return m.text.decode() if m else None


def _ruby_first_string_arg(call_node) -> str | None:
    """First plain string literal argument of a call node, used to pull
    the path out of a `require "./foo"` / `require_relative "./foo"`."""
    args = call_node.child_by_field_name("arguments")
    if args is None:
        args = next((c for c in call_node.children if c.type == "argument_list"), None)
    if args is None:
        return None
    for c in args.children:
        if c.type == "string":
            frag = next((s for s in c.children if s.type == "string_content"), None)
            if frag is not None:
                return frag.text.decode()
    return None


def _ruby_signature(node, src: bytes, name: str) -> str:
    """Render a def/class/module signature verbatim from source, same
    "slice, don't reconstruct" approach as the JS/TS signatures."""
    if node.type == "method":
        params = node.child_by_field_name("parameters")
        return f"def {name}{_text(params, src) if params else ''}"
    if node.type == "singleton_method":
        obj = node.child_by_field_name("object")
        params = node.child_by_field_name("parameters")
        obj_text = obj.text.decode() if obj else "self"
        return f"def {obj_text}.{name}{_text(params, src) if params else ''}"
    if node.type == "class":
        sup = node.child_by_field_name("superclass")
        return f"class {name}" + (f" {_text(sup, src)}" if sup else "")
    if node.type == "module":
        return f"module {name}"
    return name


def _ruby_walk_calls(node, out: set[str]) -> None:
    """Same bare-name approximation as _walk_calls, using Ruby's `call`
    node shape instead of JS's call_expression/member_expression."""
    name = _ruby_call_name(node)
    if name:
        out.add(name)
    for child in node.children:
        _ruby_walk_calls(child, out)


def _ruby_member_name(node) -> str:
    n = node.child_by_field_name("name")
    return n.text.decode() if n else "<anonymous>"


def _ruby_container(node, qualname_prefix: str, rel_path: str, lines: list[str],
                     src_bytes: bytes, class_split_lines: int,
                     chunks: list[Chunk]) -> None:
    """Emit chunks for one class/module node. If it's split (too big, or
    with methods/nested containers to carve out), recurse into any nested
    class/module children so multi-level namespacing -- the idiomatic way
    Ruby organises code (`module App; module Models; class Widget; ... end;
    end; end`) -- still yields a symbol per level instead of one opaque
    blob. A container that stays whole (short, or nothing to carve out) is
    NOT recursed into: exactly the same parity chunker.py has for Python,
    where a class under class_split_lines keeps any nested detail
    unrepresented as separate chunks rather than half-splitting it.
    """
    name = _ruby_member_name(node)
    qualname = f"{qualname_prefix}::{name}" if qualname_prefix else name
    start, end = _line_span(node)
    body_node = node.child_by_field_name("body")
    direct = list(body_node.children) if body_node else []
    carve = [c for c in direct if c.type in RUBY_CARVE_NODES]
    doc = _leading_comment_doc(node)

    if (end - start + 1) <= class_split_lines or not carve:
        member_names = [_ruby_member_name(c) for c in carve]
        body_text = "\n".join(lines[start - 1:end])
        calls: set[str] = set()
        _ruby_walk_calls(node, calls)
        chunks.append(Chunk(
            file=rel_path, qualname=qualname, kind="class",
            start=start, end=end, signature=_ruby_signature(node, src_bytes, name),
            docline=doc, members=member_names, calls=sorted(calls),
            source=body_text, sha=_sha(body_text),
        ))
        return

    header_end = max(min(c.start_point.row + 1 for c in carve) - 1, start)
    header_body = "\n".join(lines[start - 1:header_end])
    chunks.append(Chunk(
        file=rel_path, qualname=qualname, kind="classheader",
        start=start, end=header_end, scope_end=end,
        signature=_ruby_signature(node, src_bytes, name), docline=doc,
        members=[_ruby_member_name(c) for c in carve],
        source=header_body, sha=_sha(header_body),
    ))

    for c in carve:
        if c.type in RUBY_METHOD_NODES:
            m_name = _ruby_member_name(c)
            m_start, m_end = _line_span(c)
            m_body = "\n".join(lines[m_start - 1:m_end])
            m_calls: set[str] = set()
            _ruby_walk_calls(c, m_calls)
            chunks.append(Chunk(
                file=rel_path, qualname=f"{qualname}.{m_name}", kind="method",
                start=m_start, end=m_end,
                signature=_ruby_signature(c, src_bytes, m_name),
                calls=sorted(m_calls), source=m_body, sha=_sha(m_body),
            ))
        else:   # nested class/module -- recurse
            _ruby_container(c, qualname, rel_path, lines, src_bytes,
                             class_split_lines, chunks)


def chunk_source_ruby(rel_path: str, src: str, class_split_lines: int = 60) -> list[Chunk]:
    """Chunk one Ruby file's source into a list of Chunk objects.

    Unlike chunk_source_ts, this recurses (see _ruby_container's
    docstring) because nested `module`/`class` namespacing is Ruby's
    normal way of organising a file, not an edge case the way nested
    classes are in Python or JS.

    `require`/`require_relative`/`load` calls at the top level stand in
    for `imports`, since Ruby's grammar has no import-statement node --
    these are just ordinary method calls syntactically.
    """
    if not TS_AVAILABLE:
        raise RuntimeError(
            "tree-sitter and tree-sitter-language-pack are required for "
            "Ruby indexing; install the 'js' extra or fall back to "
            "reference-style chunking for this file.")

    parser = _get_parser("ruby")
    src_bytes = src.encode("utf-8", "replace")
    tree = parser.parse(src_bytes)
    root = tree.root_node
    lines = src.splitlines()
    total = len(lines)
    chunks: list[Chunk] = []
    top = list(root.children)

    imports: list[str] = []
    for node in top:
        if _ruby_call_name(node) in RUBY_IMPORT_CALLS:
            arg = _ruby_first_string_arg(node)
            if arg:
                imports.append(_module_label(arg))

    scopes = [n for n in top if n.type in RUBY_CARVE_NODES]

    if not scopes:
        if src.strip():
            calls: set[str] = set()
            for n in top:
                _ruby_walk_calls(n, calls)
            chunks.append(Chunk(
                file=rel_path, qualname="module", kind="module",
                start=1, end=total, imports=imports, calls=sorted(calls),
                source=src, sha=_sha(src),
            ))
        return chunks

    # --- module preamble: everything before the first top-level scope ---
    first_start = scopes[0].start_point.row + 1
    if first_start > 1:
        end = first_start - 1
        body = "\n".join(lines[:end])
        if body.strip():
            calls: set[str] = set()
            for n in top:
                if n.start_point.row + 1 > end:
                    break
                _ruby_walk_calls(n, calls)
            chunks.append(Chunk(
                file=rel_path, qualname="module", kind="module",
                start=1, end=end, imports=imports, calls=sorted(calls),
                source=body, sha=_sha(body),
            ))

    # --- each top-level scope: container (recursive) or bare def ---
    for node in scopes:
        if node.type in RUBY_CONTAINER_NODES:
            _ruby_container(node, "", rel_path, lines, src_bytes,
                             class_split_lines, chunks)
            continue
        name = _ruby_member_name(node)
        start, end = _line_span(node)
        body = "\n".join(lines[start - 1:end])
        doc = _leading_comment_doc(node)
        calls: set[str] = set()
        _ruby_walk_calls(node, calls)
        chunks.append(Chunk(
            file=rel_path, qualname=name, kind="function",
            start=start, end=end, signature=_ruby_signature(node, src_bytes, name),
            docline=doc, calls=sorted(calls), source=body, sha=_sha(body),
        ))

    # --- trailing top-level code after the last scope ---
    last_end = scopes[-1].end_point.row + 1
    if last_end < total:
        body = "\n".join(lines[last_end:])
        if body.strip():
            calls: set[str] = set()
            for n in top:
                if n.start_point.row + 1 > last_end:
                    _ruby_walk_calls(n, calls)
            chunks.append(Chunk(
                file=rel_path, qualname="__trailing__", kind="module",
                start=last_end + 1, end=total, calls=sorted(calls),
                source=body, sha=_sha(body),
            ))

    chunks.sort(key=lambda c: c.start)
    return chunks


# --------------------------------------------------------------------------
# Dispatch: extension -> the right chunker above
# --------------------------------------------------------------------------
def chunk_file_ts(rel_path: str, src: str, ext: str,
                   class_split_lines: int = 60) -> list[Chunk]:
    """Chunk one file using whichever tree-sitter chunker matches its
    extension. The single entry point index.py calls -- it doesn't need
    to know JS/TS and Ruby use genuinely different algorithms internally
    (export-unwrapping vs. recursive containers)."""
    lang = LANG_BY_EXT[ext.lower()]
    if lang == "ruby":
        return chunk_source_ruby(rel_path, src, class_split_lines)
    return chunk_source_ts(rel_path, src, lang, class_split_lines)


def has_syntax_error(text: str, language: str) -> tuple[bool, int | None]:
    """Return (has_error, line) for `text` parsed as `language` -- line is
    the first ERROR node's 1-based line, or None if there's no error.
    Tree-sitter is error-tolerant (see chunk_source_ts's docstring) but
    still marks the damaged region with an ERROR node, confirmed against
    deliberately broken input. This is edits.py's post-edit safety check
    for JS/TS, the same role `ast.parse` plays for Python."""
    if not TS_AVAILABLE:
        # Caller's responsibility to skip the check when unavailable; treat
        # as "can't tell" rather than silently claiming success.
        raise RuntimeError("tree-sitter is not installed; cannot check syntax")
    parser = _get_parser(language)
    tree = parser.parse(text.encode("utf-8", "replace"))
    if not tree.root_node.has_error:
        return False, None
    node = tree.root_node
    while True:
        child = next((c for c in node.children if c.has_error), None)
        if child is None:
            break
        node = child
    return True, node.start_point.row + 1
