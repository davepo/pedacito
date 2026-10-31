"""Tree-sitter based source chunking for JS/TS, Ruby, Go, Rust, Java, C/C++, C#.

    chunks = chunk_file_ts("src/app.ts", source_text, ".ts", class_split_lines=60)

Mirrors chunker.py's contract (same Chunk dataclass, kind vocabulary, and line
ranges), so nothing downstream cares which parser produced a file's chunks.
Where the grammar differs from Python's:

  - `export` wraps the declaration it exports, so scope detection unwraps one
    level. A bare `export default someIdentifier;` is not a scope.
  - A top-level `const`/`let` is a scope only if its initializer is a function,
    arrow function, or object literal.
  - There is no docstring literal: `docline` comes from the immediately
    preceding comment sibling (JSDoc or `//`).
  - TS `interface`/`type`/`enum` are chunked like small whole classes.
  - Ruby recurses into nested `module`/`class` namespaces.

Needs `tree-sitter` + `tree-sitter-language-pack`. If they are not importable,
`TS_AVAILABLE` is False and callers fall back to references.py chunking.
"""

from __future__ import annotations

import re

from .chunker import (
    Chunk, _part_signature, _sha, _split_statements, split_lines,
)

try:
    from tree_sitter_language_pack import get_parser
    TS_AVAILABLE = True
except ImportError:                                    # pragma: no cover
    get_parser = None
    TS_AVAILABLE = False

# File extension -> tree-sitter grammar name. The "typescript" grammar rejects
# JSX, hence the separate "tsx".
LANG_BY_EXT = {
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rb": "ruby",
    # handled by chunk_source_generic (see GENERIC_SPECS)
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "cpp",          # C++ grammar is a practical superset for headers
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".cs": "csharp",
}

# Function-like scope node types (same across the JS, TS, and TSX grammars).
FUNCTION_NODES = {
    "function_declaration", "generator_function_declaration",
}
CLASS_NODES = {"class_declaration"}
# TypeScript-only: chunked like small whole classes.
TS_DECL_NODES = {
    "interface_declaration", "type_alias_declaration", "enum_declaration",
}
METHOD_NODES = {"method_definition"}
FIELD_METHOD_NODES = {"field_definition", "public_field_definition"}
OBJECT_NODES = {"object"}

# Ruby node types
RUBY_CONTAINER_NODES = {"class", "module"}
RUBY_METHOD_NODES = {"method", "singleton_method"}
RUBY_CARVE_NODES = RUBY_CONTAINER_NODES | RUBY_METHOD_NODES
RUBY_IMPORT_CALLS = {"require", "require_relative", "load"}

_parsers: dict[str, "object"] = {}


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


def _split_nodes(stmts: list, lines: list[str], first_line: int, last_line: int,
                 max_lines: int, max_chars: int, max_tokens: int = 0,
                 overlap_lines: int = 0
                 ) -> list[tuple[int, int, list]] | None:
    """Group top-level statement nodes into retrieval-sized parts."""
    return _split_statements(stmts, lines, first_line, last_line, max_lines,
                             max_chars, max_tokens, overlap_lines,
                             span=_line_span)


def _split_body(body_node, lines: list[str], first_line: int, last_line: int,
                max_lines: int, max_chars: int, max_tokens: int = 0,
                overlap_lines: int = 0
                ) -> list[tuple[int, int, list]] | None:
    """Group a function/method body's statements into retrieval-sized parts.
    Returns None when the span fits or the body is not a block."""
    if body_node is None or (
            "block" not in getattr(body_node, "type", "")
            and getattr(body_node, "type", "") != "body_statement"):
        return None
    stmts = list(getattr(body_node, "named_children", []) or [])
    return _split_nodes(stmts, lines, first_line, last_line,
                        max_lines, max_chars, max_tokens, overlap_lines)


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
    # strip("/*#") covers JSDoc blocks, // comments, and Ruby # comments
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
    """Approximate signature: keywords plus the parameter list, verbatim from
    source."""
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


def _is_method_like(node) -> bool:
    if node.type in METHOD_NODES:
        return True
    if node.type in FIELD_METHOD_NODES:
        value = node.child_by_field_name("value")
        return value is not None and value.type in (
            "arrow_function", "function", "function_expression")
    return False


def _method_body_node(node):
    body = node.child_by_field_name("body")
    if body is not None:
        return body
    if node.type in FIELD_METHOD_NODES:
        value = node.child_by_field_name("value")
        if value is not None:
            return value.child_by_field_name("body")
    return None


def _signature_of_field_method(node, src: bytes, name: str) -> str:
    value = node.child_by_field_name("value")
    params = value.child_by_field_name("parameters") if value is not None else None
    return f"{name} = {_text(params, src) if params else '()'} => ..."


def _walk_calls(node, out: set[str]) -> None:
    """Collect names invoked via call/new expressions under `node`. Member
    calls (obj.method()) reduce to the property name: the same bare-name
    approximation chunker.py uses for Python. Iterative, so deeply nested
    trees (minified JS) cannot hit the recursion limit."""
    stack = [node]
    while stack:
        node = stack.pop()
        t = node.type
        if t == "call_expression":
            fn = node.child_by_field_name("function")
            if fn is not None:
                if fn.type == "identifier":
                    out.add(fn.text.decode())
                elif fn.type == "member_expression":
                    prop = fn.child_by_field_name("property")
                    if prop is not None:
                        out.add(prop.text.decode())
                elif fn.type in ("field_expression", "selector_expression"):
                    prop = fn.child_by_field_name("field")
                    if prop is not None:
                        out.add(prop.text.decode())
        elif t == "new_expression":
            ctor = node.child_by_field_name("constructor")
            if ctor is not None and ctor.type == "identifier":
                out.add(ctor.text.decode())
        elif t == "method_invocation":
            nm = node.child_by_field_name("name")
            if nm is not None:
                out.add(nm.text.decode())
        elif t == "object_creation_expression":
            ty = node.child_by_field_name("type")
            if ty is not None:
                out.add(ty.text.decode().split(".")[-1])
        elif t == "macro_invocation":
            mac = node.child_by_field_name("macro")
            if mac is not None:
                out.add(mac.text.decode().rstrip("!"))
        stack.extend(node.children)


def _module_label(spec: str) -> str:
    """Collapse an import specifier to a short label for the file card's
    "imports:" line ("./a/foo" -> "foo"; index.py strips leading dots, which
    would otherwise empty a bare relative path)."""
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


def chunk_source_ts(rel_path: str, src: str, language: str,
                     class_split_lines: int = 60,
                     max_unit_lines: int = 220,
                     max_unit_chars: int = 12000,
                     max_unit_tokens: int = 3000,
                     unit_overlap_lines: int = 0) -> list[Chunk]:
    """Chunk one JS/TS/TSX file's source into a list of Chunk objects: a
    module preamble, one chunk per top-level function (including arrow
    functions in a const/let), one per class or TS interface/type/enum (whole,
    or classheader plus one chunk per method past `class_split_lines`), and a
    trailing chunk.

    Never raises on malformed source: tree-sitter is error-tolerant, so a
    broken file just degrades to fewer/coarser chunks.
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
    lines = split_lines(src)
    total = len(lines)
    chunks: list[Chunk] = []
    imports = _imports(root, src_bytes)

    # classify each top-level statement, unwrapping export wrappers
    scopes: list[tuple[object, object]] = []   # (outer_node, inner_decl_node)
    for outer in root.children:
        inner = _unwrap_export(outer)
        if inner is None:
            continue
        if inner.type in FUNCTION_NODES | CLASS_NODES | TS_DECL_NODES:
            scopes.append((outer, inner))
        elif inner.type in ("lexical_declaration", "variable_declaration"):
            # Only function-like if a declarator's value is a function.
            for decl in inner.children:
                if decl.type != "variable_declarator":
                    continue
                value = decl.child_by_field_name("value")
                if value is not None and (value.type in (
                        "arrow_function", "function", "function_expression")
                        or value.type in OBJECT_NODES):
                    scopes.append((outer, inner))
                    break

    if not scopes:
        if src.strip():
            whole_calls: set[str] = set()
            _walk_calls(root, whole_calls)
            chunks.append(Chunk(
                file=rel_path, qualname="module", kind="module",
                start=1, end=total, imports=imports,
                calls=sorted(whole_calls),
                source=src, sha=_sha(src),
            ))
        return chunks

    # module preamble: everything before the first scope
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

    # each top-level scope
    for outer, inner in scopes:
        start, end = _line_span(outer)
        body = "\n".join(lines[start - 1:end])
        doc = _leading_comment_doc(outer)
        decs = _decorators(outer, src_bytes) + _decorators(inner, src_bytes)

        if inner.type in FUNCTION_NODES:
            name_node = inner.child_by_field_name("name")
            name = name_node.text.decode() if name_node else "<anonymous>"
            sig = _signature_of_function(inner, src_bytes, name)
            body_node = inner.child_by_field_name("body")
            parts = _split_body(body_node, lines, start, end,
                                max_unit_lines, max_unit_chars,
                                max_unit_tokens, unit_overlap_lines)
            if parts:
                for i, (ps, pe, nodes) in enumerate(parts, 1):
                    part_body = "\n".join(lines[ps - 1:pe])
                    p_calls: set[str] = set()
                    for n in nodes:
                        _walk_calls(n, p_calls)
                    chunks.append(Chunk(
                        file=rel_path,
                        qualname=name if i == 1 else f"{name}#{i}",
                        kind="function", start=ps, end=pe,
                        signature=_part_signature(sig, i, len(parts)),
                        decorators=decs if i == 1 else [],
                        docline=doc if i == 1 else "",
                        calls=sorted(p_calls), imports=imports,
                        source=part_body, sha=_sha(part_body),
                    ))
                continue
            calls: set[str] = set()
            _walk_calls(inner, calls)
            chunks.append(Chunk(
                file=rel_path, qualname=name, kind="function",
                start=start, end=end, signature=sig,
                decorators=decs, docline=doc, calls=sorted(calls),
                imports=imports, source=body, sha=_sha(body),
            ))
            continue

        if inner.type in ("lexical_declaration", "variable_declaration"):
            # Arrow-function-as-const: pull the name from the declarator.
            decl = next(c for c in inner.children if c.type == "variable_declarator")
            name_node = decl.child_by_field_name("name")
            name = name_node.text.decode() if name_node else "<anonymous>"
            value = decl.child_by_field_name("value")
            if value is not None and value.type in OBJECT_NODES:
                # Object-literal methods (`const api = { foo() {}, bar: ... }`)
                # are real callable units. Expose the object as a class-like
                # unit and each method as its own chunk, matching the class path.
                members: list[tuple[str, object]] = []
                for child in value.named_children:
                    if child.type == "method_definition":
                        nm = child.child_by_field_name("name")
                        if nm is not None:
                            members.append((nm.text.decode(), child))
                    elif child.type == "pair":
                        key = child.child_by_field_name("key")
                        val = child.child_by_field_name("value")
                        if key is not None and val is not None and val.type in (
                                "arrow_function", "function",
                                "function_expression"):
                            members.append((_text(key, src_bytes).strip(
                                "\"'"), child))
                if not members:
                    chunks.append(Chunk(
                        file=rel_path, qualname=name, kind="class",
                        start=start, end=end,
                        signature=f"const {name} = {{...}}",
                        decorators=decs, docline=doc, imports=imports,
                        source=body, sha=_sha(body),
                    ))
                    continue
                first_method = min(m.start_point.row + 1
                                   for _n, m in members)
                header_end = max(first_method - 1, start)
                if header_end >= start:
                    header_body = "\n".join(lines[start - 1:header_end])
                    chunks.append(Chunk(
                        file=rel_path, qualname=name, kind="classheader",
                        start=start, end=header_end, scope_end=end,
                        signature=f"const {name} = {{...}}",
                        decorators=decs, docline=doc,
                        members=[n for n, _m in members], imports=imports,
                        source=header_body, sha=_sha(header_body),
                    ))
                for m_name, m in members:
                    m_start, m_end = _line_span(m)
                    if m.type == "method_definition":
                        params = m.child_by_field_name("parameters")
                        m_sig = (f"{m_name}"
                                 f"{_text(params, src_bytes) if params else '()'}")
                        m_body_node = m.child_by_field_name("body")
                    else:
                        val = m.child_by_field_name("value")
                        params = (val.child_by_field_name("parameters")
                                  if val is not None else None)
                        m_sig = (f"{m_name} = "
                                 f"{_text(params, src_bytes) if params else '()'}"
                                 f" => ...")
                        m_body_node = (val.child_by_field_name("body")
                                       if val is not None else None)
                    parts = _split_body(m_body_node, lines, m_start, m_end,
                                        max_unit_lines, max_unit_chars,
                                        max_unit_tokens, unit_overlap_lines)
                    if parts:
                        for i, (ps, pe, nodes) in enumerate(parts, 1):
                            part_body = "\n".join(lines[ps - 1:pe])
                            p_calls: set[str] = set()
                            for n in nodes:
                                _walk_calls(n, p_calls)
                            chunks.append(Chunk(
                                file=rel_path,
                                qualname=(f"{name}.{m_name}" if i == 1
                                          else f"{name}.{m_name}#{i}"),
                                kind="method", start=ps, end=pe,
                                signature=_part_signature(m_sig, i,
                                                          len(parts)),
                                calls=sorted(p_calls), imports=imports,
                                source=part_body, sha=_sha(part_body),
                            ))
                        continue
                    m_text = "\n".join(lines[m_start - 1:m_end])
                    calls: set[str] = set()
                    _walk_calls(m, calls)
                    chunks.append(Chunk(
                        file=rel_path, qualname=f"{name}.{m_name}",
                        kind="method", start=m_start, end=m_end,
                        signature=m_sig, calls=sorted(calls),
                        imports=imports, source=m_text, sha=_sha(m_text),
                    ))
                continue
            calls: set[str] = set()
            _walk_calls(value, calls)
            params = value.child_by_field_name("parameters")
            sig = f"const {name} = {_text(params, src_bytes) if params else '()'} => ..."
            body_node = value.child_by_field_name("body") if value is not None else None
            parts = _split_body(body_node, lines, start, end,
                                max_unit_lines, max_unit_chars,
                                max_unit_tokens, unit_overlap_lines)
            if parts:
                for i, (ps, pe, nodes) in enumerate(parts, 1):
                    part_body = "\n".join(lines[ps - 1:pe])
                    p_calls: set[str] = set()
                    for n in nodes:
                        _walk_calls(n, p_calls)
                    chunks.append(Chunk(
                        file=rel_path,
                        qualname=name if i == 1 else f"{name}#{i}",
                        kind="function", start=ps, end=pe,
                        signature=_part_signature(sig, i, len(parts)),
                        decorators=decs if i == 1 else [],
                        docline=doc if i == 1 else "",
                        calls=sorted(p_calls), imports=imports,
                        source=part_body, sha=_sha(part_body),
                    ))
                continue
            chunks.append(Chunk(
                file=rel_path, qualname=name, kind="function",
                start=start, end=end, signature=sig,
                decorators=decs, docline=doc, calls=sorted(calls),
                imports=imports, source=body, sha=_sha(body),
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
                imports=imports, source=body, sha=_sha(body),
            ))
            continue

        # ClassDef (class_declaration)
        name_node = inner.child_by_field_name("name")
        name = name_node.text.decode() if name_node else "<anonymous>"
        body_node = inner.child_by_field_name("body")
        methods = [m for m in
                   (body_node.named_children if body_node else [])
                   if _is_method_like(m)]
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
                calls=sorted(calls), imports=imports,
                source=body, sha=_sha(body),
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
            imports=imports, source=header_body, sha=_sha(header_body),
        ))
        for m in methods:
            m_name_node = m.child_by_field_name("name")
            m_name = m_name_node.text.decode() if m_name_node else "<anonymous>"
            m_start, m_end = _line_span(m)
            if m.type in FIELD_METHOD_NODES:
                m_sig = _signature_of_field_method(m, src_bytes, m_name)
            else:
                m_params = m.child_by_field_name("parameters")
                m_sig = (f"{m_name}"
                         f"{_text(m_params, src_bytes) if m_params else '()'}")
            m_decs = _decorators(m, src_bytes)
            body_node = _method_body_node(m)
            parts = _split_body(body_node, lines, m_start, m_end,
                                max_unit_lines, max_unit_chars,
                                max_unit_tokens, unit_overlap_lines)
            if parts:
                for i, (ps, pe, nodes) in enumerate(parts, 1):
                    part_body = "\n".join(lines[ps - 1:pe])
                    p_calls: set[str] = set()
                    for n in nodes:
                        _walk_calls(n, p_calls)
                    chunks.append(Chunk(
                        file=rel_path,
                        qualname=(f"{name}.{m_name}" if i == 1
                                  else f"{name}.{m_name}#{i}"),
                        kind="method", start=ps, end=pe,
                        signature=_part_signature(m_sig, i, len(parts)),
                        decorators=m_decs if i == 1 else [],
                        calls=sorted(p_calls), imports=imports,
                        source=part_body, sha=_sha(part_body),
                    ))
                continue
            m_body = "\n".join(lines[m_start - 1:m_end])
            m_calls: set[str] = set()
            _walk_calls(m, m_calls)
            chunks.append(Chunk(
                file=rel_path, qualname=f"{name}.{m_name}", kind="method",
                start=m_start, end=m_end, signature=m_sig,
                decorators=m_decs, calls=sorted(m_calls),
                imports=imports, source=m_body, sha=_sha(m_body),
            ))

    # trailing top-level code after the last scope
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


def _ruby_call_name(node) -> str | None:
    """The `method` field text of a Ruby `call` node (`foo(1)` or
    `obj.foo(1)`), or None if `node` isn't a call."""
    if node.type != "call":
        return None
    m = node.child_by_field_name("method")
    return m.text.decode() if m else None


def _ruby_first_string_arg(call_node) -> str | None:
    """First plain string literal argument of a call node (the path in
    `require "./foo"`)."""
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
    """Render a def/class/module signature verbatim from source."""
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
    """Bare-name call collection like _walk_calls, for Ruby `call` nodes."""
    stack = [node]
    while stack:
        node = stack.pop()
        name = _ruby_call_name(node)
        if name:
            out.add(name)
        stack.extend(node.children)


def _ruby_member_name(node) -> str:
    n = node.child_by_field_name("name")
    return n.text.decode() if n else "<anonymous>"


def _ruby_container(node, qualname_prefix: str, rel_path: str, lines: list[str],
                     src_bytes: bytes, class_split_lines: int,
                     chunks: list[Chunk], max_unit_lines: int = 220,
                     max_unit_chars: int = 12000,
                     max_unit_tokens: int = 3000,
                     unit_overlap_lines: int = 0,
                     imports: list[str] | None = None) -> None:
    """Emit chunks for one class/module node. When it is split, recurse into
    nested class/module children so `module App; module Models; class Widget`
    yields a symbol per level. A container that stays whole (short, or nothing
    to carve out) is not recursed into.
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
            imports=list(imports or []), source=body_text, sha=_sha(body_text),
        ))
        return

    header_end = max(min(c.start_point.row + 1 for c in carve) - 1, start)
    header_body = "\n".join(lines[start - 1:header_end])
    chunks.append(Chunk(
        file=rel_path, qualname=qualname, kind="classheader",
        start=start, end=header_end, scope_end=end,
        signature=_ruby_signature(node, src_bytes, name), docline=doc,
        members=[_ruby_member_name(c) for c in carve],
        imports=list(imports or []),
        source=header_body, sha=_sha(header_body),
    ))

    for c in carve:
        if c.type in RUBY_METHOD_NODES:
            m_name = _ruby_member_name(c)
            m_start, m_end = _line_span(c)
            m_sig = _ruby_signature(c, src_bytes, m_name)
            body_node = c.child_by_field_name("body")
            parts = _split_body(body_node, lines, m_start, m_end,
                                max_unit_lines, max_unit_chars,
                                max_unit_tokens, unit_overlap_lines)
            if parts:
                for i, (ps, pe, nodes) in enumerate(parts, 1):
                    part_body = "\n".join(lines[ps - 1:pe])
                    p_calls: set[str] = set()
                    for n in nodes:
                        _ruby_walk_calls(n, p_calls)
                    chunks.append(Chunk(
                        file=rel_path,
                        qualname=(f"{qualname}.{m_name}" if i == 1
                                  else f"{qualname}.{m_name}#{i}"),
                        kind="method", start=ps, end=pe,
                        signature=_part_signature(m_sig, i, len(parts)),
                        calls=sorted(p_calls), imports=list(imports or []),
                        source=part_body, sha=_sha(part_body),
                    ))
                continue
            m_body = "\n".join(lines[m_start - 1:m_end])
            m_calls: set[str] = set()
            _ruby_walk_calls(c, m_calls)
            chunks.append(Chunk(
                file=rel_path, qualname=f"{qualname}.{m_name}", kind="method",
                start=m_start, end=m_end, signature=m_sig,
                calls=sorted(m_calls), imports=list(imports or []),
                source=m_body, sha=_sha(m_body),
            ))
        else:   # nested class/module -- recurse
            _ruby_container(c, qualname, rel_path, lines, src_bytes,
                             class_split_lines, chunks, max_unit_lines,
                             max_unit_chars, max_unit_tokens,
                             unit_overlap_lines, imports)


def chunk_source_ruby(rel_path: str, src: str, class_split_lines: int = 60,
                      max_unit_lines: int = 220,
                      max_unit_chars: int = 12000,
                      max_unit_tokens: int = 3000,
                      unit_overlap_lines: int = 0) -> list[Chunk]:
    """Chunk one Ruby file's source into a list of Chunk objects.

    Top-level `require`/`require_relative`/`load` calls stand in for
    `imports`, since Ruby's grammar has no import-statement node.
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
    lines = split_lines(src)
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

    # module preamble: everything before the first top-level scope
    first_start = scopes[0].start_point.row + 1
    if first_start > 1:
        end = first_start - 1
        body = "\n".join(lines[:end])
        if body.strip():
            preamble_nodes = [n for n in top
                              if n.start_point.row + 1 <= end]
            parts = _split_nodes(preamble_nodes, lines, 1, end,
                                 max_unit_lines, max_unit_chars,
                                 max_unit_tokens, unit_overlap_lines)
            if parts:
                for i, (ps, pe, nodes) in enumerate(parts, 1):
                    part_body = "\n".join(lines[ps - 1:pe])
                    calls: set[str] = set()
                    for n in nodes:
                        _ruby_walk_calls(n, calls)
                    chunks.append(Chunk(
                        file=rel_path,
                        qualname="module" if i == 1 else f"module#{i}",
                        kind="module", start=ps, end=pe,
                        imports=imports, calls=sorted(calls),
                        source=part_body, sha=_sha(part_body),
                    ))
            else:
                calls: set[str] = set()
                for n in preamble_nodes:
                    _ruby_walk_calls(n, calls)
                chunks.append(Chunk(
                    file=rel_path, qualname="module", kind="module",
                    start=1, end=end, imports=imports, calls=sorted(calls),
                    source=body, sha=_sha(body),
                ))

    # each top-level scope: container (recursive) or bare def
    for node in scopes:
        if node.type in RUBY_CONTAINER_NODES:
            _ruby_container(node, "", rel_path, lines, src_bytes,
                             class_split_lines, chunks, max_unit_lines,
                             max_unit_chars, max_unit_tokens,
                             unit_overlap_lines, imports)
            continue
        name = _ruby_member_name(node)
        start, end = _line_span(node)
        sig = _ruby_signature(node, src_bytes, name)
        doc = _leading_comment_doc(node)
        body_node = node.child_by_field_name("body")
        parts = _split_body(body_node, lines, start, end,
                            max_unit_lines, max_unit_chars,
                            max_unit_tokens, unit_overlap_lines)
        if parts:
            for i, (ps, pe, nodes) in enumerate(parts, 1):
                part_body = "\n".join(lines[ps - 1:pe])
                p_calls: set[str] = set()
                for n in nodes:
                    _ruby_walk_calls(n, p_calls)
                chunks.append(Chunk(
                    file=rel_path,
                    qualname=name if i == 1 else f"{name}#{i}",
                    kind="function", start=ps, end=pe,
                    signature=_part_signature(sig, i, len(parts)),
                    docline=doc if i == 1 else "",
                    calls=sorted(p_calls), imports=imports,
                    source=part_body, sha=_sha(part_body),
                ))
            continue
        body = "\n".join(lines[start - 1:end])
        calls: set[str] = set()
        _ruby_walk_calls(node, calls)
        chunks.append(Chunk(
            file=rel_path, qualname=name, kind="function",
            start=start, end=end, signature=sig,
            docline=doc, calls=sorted(calls), imports=imports,
            source=body, sha=_sha(body),
        ))

    # trailing top-level code after the last scope
    last_end = scopes[-1].end_point.row + 1
    if last_end < total:
        body = "\n".join(lines[last_end:])
        if body.strip():
            tail_nodes = [n for n in top
                          if n.start_point.row + 1 > last_end]
            parts = _split_nodes(tail_nodes, lines, last_end + 1, total,
                                 max_unit_lines, max_unit_chars,
                                 max_unit_tokens, unit_overlap_lines)
            if parts:
                for i, (ps, pe, nodes) in enumerate(parts, 1):
                    part_body = "\n".join(lines[ps - 1:pe])
                    calls: set[str] = set()
                    for n in nodes:
                        _ruby_walk_calls(n, calls)
                    chunks.append(Chunk(
                        file=rel_path,
                        qualname=("__trailing__" if i == 1
                                  else f"__trailing__#{i}"),
                        kind="module", start=ps, end=pe, imports=imports,
                        calls=sorted(calls), source=part_body,
                        sha=_sha(part_body),
                    ))
            else:
                calls: set[str] = set()
                for n in tail_nodes:
                    _ruby_walk_calls(n, calls)
                chunks.append(Chunk(
                    file=rel_path, qualname="__trailing__", kind="module",
                    start=last_end + 1, end=total, imports=imports,
                    calls=sorted(calls), source=body, sha=_sha(body),
                ))

    chunks.sort(key=lambda c: c.start)
    return chunks


# Generic tree-sitter languages (Go, Rust, Java, C, C++, C#): one node-name
# spec per language. Function-like nodes become function chunks, containers a
# class (or classheader + methods), and declarations (struct/enum/type)
# class-shaped chunks.
GENERIC_SPECS = {
    "go": {
        "functions": {"function_declaration", "method_declaration"},
        "containers": set(),
        "methods": set(),
        "decls": {"type_declaration"},
        "imports": {"import_declaration"},
    },
    "rust": {
        "functions": {"function_item"},
        "containers": {"impl_item", "trait_item"},
        "methods": {"function_item"},
        "decls": {"struct_item", "enum_item", "mod_item", "type_item"},
        "imports": {"use_declaration"},
    },
    "java": {
        "functions": {"method_declaration", "constructor_declaration"},
        "containers": {"class_declaration", "interface_declaration",
                       "enum_declaration", "record_declaration",
                       "annotation_type_declaration"},
        "methods": {"method_declaration", "constructor_declaration"},
        "decls": set(),
        "imports": {"import_declaration"},
    },
    "c": {
        "functions": {"function_definition"},
        "containers": {"struct_specifier", "union_specifier"},
        "methods": set(),
        "decls": {"enum_specifier", "type_definition"},
        "imports": {"preproc_include"},
    },
    "cpp": {
        "functions": {"function_definition"},
        "containers": {"class_specifier", "struct_specifier",
                       "namespace_definition"},
        "methods": {"function_definition"},
        "decls": {"enum_specifier", "type_definition", "template_declaration"},
        "imports": {"preproc_include"},
    },
    "csharp": {
        "functions": {"method_declaration", "constructor_declaration"},
        "containers": {"class_declaration", "interface_declaration",
                       "struct_declaration", "enum_declaration",
                       "record_declaration"},
        "methods": {"method_declaration", "constructor_declaration"},
        "decls": set(),
        "imports": {"using_directive"},
    },
}


def _c_declarator_name(node, src: bytes) -> str:
    """Dig through C/C++ pointer/function declarators to the identifier."""
    d = node.child_by_field_name("declarator")
    for _ in range(12):
        if d is None:
            return ""
        if d.type in ("identifier", "field_identifier", "operator_name",
                      "destructor_name", "type_identifier"):
            return _text(d, src).strip()
        nxt = d.child_by_field_name("declarator")
        if nxt is None:
            nxt = next((c for c in d.named_children
                        if c.type.endswith("declarator")
                        or c.type in ("identifier", "field_identifier",
                                      "qualified_identifier")), None)
        d = nxt
    return ""


def _generic_name(node, src: bytes) -> str:
    """Best-effort declaration name across the generic grammars."""
    if node.type == "function_definition":
        name = _c_declarator_name(node, src)
        if name:
            return name.split("::")[-1]
    for fld in ("name", "type"):
        n = node.child_by_field_name(fld)
        if n is not None:
            text = _text(n, src).strip()
            if fld == "type":
                text = text.split("<", 1)[0].split(".")[-1]
            if text:
                return text
    for child in node.named_children:
        for fld in ("name", "type"):
            n = child.child_by_field_name(fld)
            if n is not None:
                text = _text(n, src).strip()
                if text:
                    return text
        if child.type in ("type_spec", "type_identifier", "identifier"):
            return _text(child, src).strip()
    return "<anonymous>"


def _generic_signature(node, src: bytes) -> str:
    first = _text(node, src).splitlines()[0].strip()
    return first.split("{", 1)[0].strip()[:200]


def _generic_body(node):
    b = node.child_by_field_name("body")
    if b is not None:
        return b
    body_types = {
        "class_body", "declaration_list", "field_declaration_list",
        "enum_body", "interface_body", "struct_body", "block",
        "compound_statement", "body",
    }
    for c in node.named_children:
        if c.type in body_types:
            return c
    return None


def _generic_imports(root, spec, src: bytes) -> list[str]:
    out: list[str] = []
    for node in root.named_children:
        if node.type not in spec["imports"]:
            continue
        text = _text(node, src)
        found = re.findall(r"[\"']([^\"']+)[\"']", text)
        if not found:
            val = re.sub(r"^\s*(?:use|import|using|#include)\s*", "", text,
                         flags=re.IGNORECASE).strip()
            val = val.lstrip("<").rstrip(">").split("\n", 1)[0]
            val = val.strip().rstrip(";")
            if val:
                found = [val]
        for val in found:
            val = val.strip().replace("::", ".")
            if val:
                out.append(_module_label(val))
    return out


def chunk_source_generic(rel_path: str, src: str, language: str,
                         class_split_lines: int = 60,
                         max_unit_lines: int = 220,
                         max_unit_chars: int = 12000,
                         max_unit_tokens: int = 3000,
                         unit_overlap_lines: int = 0) -> list[Chunk]:
    """Chunk Go/Rust/Java/C/C++/C# with the shared generic algorithm."""
    if language not in GENERIC_SPECS:
        raise ValueError(f"no generic chunker spec for {language}")
    spec = GENERIC_SPECS[language]
    parser = _get_parser(language)
    src_bytes = src.encode("utf-8", "replace")
    root = parser.parse(src_bytes).root_node
    lines = split_lines(src)
    total = len(lines)
    chunks: list[Chunk] = []
    imports = _generic_imports(root, spec, src_bytes)

    scopes: list[tuple[object, object, str]] = []
    preamble: list = []
    seen_scope = False
    for node in root.named_children:
        inner = node
        if node.type == "template_declaration":
            inner = next((c for c in node.named_children
                          if c.type in spec["functions"]
                          or c.type in spec["containers"]
                          or c.type in spec["decls"]), node)
        if inner.type in spec["functions"]:
            scopes.append((node, inner, "function"))
            seen_scope = True
        elif inner.type in spec["containers"]:
            scopes.append((node, inner, "container"))
            seen_scope = True
        elif inner.type in spec["decls"]:
            scopes.append((node, inner, "decl"))
            seen_scope = True
        elif inner.type in spec["imports"]:
            continue
        elif not seen_scope:
            preamble.append(node)

    def emit_function(outer, inner, name):
        start, end = _line_span(outer)
        sig = _generic_signature(outer, src_bytes)
        body = inner.child_by_field_name("body")
        parts = _split_body(body, lines, start, end, max_unit_lines,
                            max_unit_chars, max_unit_tokens,
                            unit_overlap_lines)
        if parts:
            for i, (ps, pe, nodes) in enumerate(parts, 1):
                part_body = "\n".join(lines[ps - 1:pe])
                calls: set[str] = set()
                for n in nodes:
                    _walk_calls(n, calls)
                chunks.append(Chunk(
                    file=rel_path,
                    qualname=name if i == 1 else f"{name}#{i}",
                    kind="function", start=ps, end=pe,
                    signature=_part_signature(sig, i, len(parts)),
                    calls=sorted(calls), imports=imports,
                    source=part_body, sha=_sha(part_body),
                ))
            return
        calls: set[str] = set()
        _walk_calls(inner, calls)
        body_text = "\n".join(lines[start - 1:end])
        chunks.append(Chunk(
            file=rel_path, qualname=name, kind="function",
            start=start, end=end, signature=sig, calls=sorted(calls),
            imports=imports, source=body_text, sha=_sha(body_text),
        ))

    def emit_container(outer, inner, name):
        start, end = _line_span(outer)
        sig = _generic_signature(outer, src_bytes)
        body = _generic_body(inner)
        methods = [c for c in (body.named_children if body else [])
                   if c.type in spec["methods"]]
        if not methods or (end - start + 1) <= class_split_lines:
            if not methods and body is not None:
                parts = _split_nodes(list(body.named_children), lines, start,
                                     end, max_unit_lines, max_unit_chars,
                                     max_unit_tokens, unit_overlap_lines)
                if parts:
                    for i, (ps, pe, nodes) in enumerate(parts, 1):
                        part_body = "\n".join(lines[ps - 1:pe])
                        calls: set[str] = set()
                        for n in nodes:
                            _walk_calls(n, calls)
                        chunks.append(Chunk(
                            file=rel_path,
                            qualname=name if i == 1
                            else f"{name}#{i}",
                            kind="class", start=ps, end=pe,
                            signature=_part_signature(sig, i, len(parts)),
                            calls=sorted(calls), imports=imports,
                            source=part_body, sha=_sha(part_body),
                        ))
                    return
            calls: set[str] = set()
            _walk_calls(inner, calls)
            body_text = "\n".join(lines[start - 1:end])
            chunks.append(Chunk(
                file=rel_path, qualname=name, kind="class",
                start=start, end=end, signature=sig,
                members=[_generic_name(m, src_bytes) for m in methods],
                calls=sorted(calls), imports=imports,
                source=body_text, sha=_sha(body_text),
            ))
            return

        first_method = min(m.start_point.row + 1 for m in methods)
        header_end = max(first_method - 1, start)
        header_nodes = [c for c in body.named_children
                        if c.start_point.row + 1 <= header_end]
        header_parts = _split_nodes(header_nodes, lines, start, header_end,
                                    max_unit_lines, max_unit_chars,
                                    max_unit_tokens, unit_overlap_lines)
        if header_parts:
            for i, (ps, pe, nodes) in enumerate(header_parts, 1):
                part_body = "\n".join(lines[ps - 1:pe])
                calls: set[str] = set()
                for n in nodes:
                    _walk_calls(n, calls)
                chunks.append(Chunk(
                    file=rel_path,
                    qualname=name if i == 1 else f"{name}#{i}",
                    kind="classheader", start=ps, end=pe,
                    scope_end=end if i == 1 else 0,
                    signature=_part_signature(sig, i, len(header_parts)),
                    members=[_generic_name(m, src_bytes) for m in methods]
                    if i == 1 else [],
                    calls=sorted(calls), imports=imports,
                    source=part_body, sha=_sha(part_body),
                ))
        else:
            header_body = "\n".join(lines[start - 1:header_end])
            chunks.append(Chunk(
                file=rel_path, qualname=name, kind="classheader",
                start=start, end=header_end, scope_end=end,
                signature=sig,
                members=[_generic_name(m, src_bytes) for m in methods],
                imports=imports, source=header_body,
                sha=_sha(header_body),
            ))
        for m in methods:
            m_name = _generic_name(m, src_bytes)
            m_start, m_end = _line_span(m)
            m_sig = _generic_signature(m, src_bytes)
            m_body = m.child_by_field_name("body")
            parts = _split_body(m_body, lines, m_start, m_end,
                                max_unit_lines, max_unit_chars,
                                max_unit_tokens, unit_overlap_lines)
            if parts:
                for i, (ps, pe, nodes) in enumerate(parts, 1):
                    part_body = "\n".join(lines[ps - 1:pe])
                    calls: set[str] = set()
                    for n in nodes:
                        _walk_calls(n, calls)
                    chunks.append(Chunk(
                        file=rel_path,
                        qualname=(f"{name}.{m_name}" if i == 1
                                  else f"{name}.{m_name}#{i}"),
                        kind="method", start=ps, end=pe,
                        signature=_part_signature(m_sig, i, len(parts)),
                        calls=sorted(calls), imports=imports,
                        source=part_body, sha=_sha(part_body),
                    ))
                continue
            m_text = "\n".join(lines[m_start - 1:m_end])
            calls: set[str] = set()
            _walk_calls(m, calls)
            chunks.append(Chunk(
                file=rel_path, qualname=f"{name}.{m_name}", kind="method",
                start=m_start, end=m_end, signature=m_sig,
                calls=sorted(calls), imports=imports,
                source=m_text, sha=_sha(m_text),
            ))

    # preamble before the first scope
    if preamble:
        first_start = preamble[0].start_point.row + 1
        end = max(c.end_point.row + 1 for c in preamble)
        parts = _split_nodes(preamble, lines, first_start, end,
                             max_unit_lines, max_unit_chars,
                             max_unit_tokens, unit_overlap_lines)
        if parts:
            for i, (ps, pe, nodes) in enumerate(parts, 1):
                part_body = "\n".join(lines[ps - 1:pe])
                calls: set[str] = set()
                for n in nodes:
                    _walk_calls(n, calls)
                chunks.append(Chunk(
                    file=rel_path,
                    qualname="module" if i == 1 else f"module#{i}",
                    kind="module", start=ps, end=pe, imports=imports,
                    calls=sorted(calls), source=part_body,
                    sha=_sha(part_body),
                ))
        else:
            body_text = "\n".join(lines[first_start - 1:end])
            calls: set[str] = set()
            for n in preamble:
                _walk_calls(n, calls)
            chunks.append(Chunk(
                file=rel_path, qualname="module", kind="module",
                start=first_start, end=end, imports=imports,
                calls=sorted(calls), source=body_text, sha=_sha(body_text),
            ))

    for outer, inner, kind in scopes:
        name = _generic_name(inner, src_bytes)
        if kind == "function":
            emit_function(outer, inner, name)
        elif kind == "container":
            emit_container(outer, inner, name)
        else:
            start, end = _line_span(outer)
            body_text = "\n".join(lines[start - 1:end])
            calls: set[str] = set()
            _walk_calls(inner, calls)
            chunks.append(Chunk(
                file=rel_path, qualname=name, kind="class",
                start=start, end=end, signature=_generic_signature(
                    outer, src_bytes),
                calls=sorted(calls), imports=imports,
                source=body_text, sha=_sha(body_text),
            ))

    # trailing top-level code
    if scopes:
        last_end = max(_line_span(o)[1] for o, _i, _k in scopes)
        tail = [n for n in root.named_children
                if n.start_point.row + 1 > last_end]
        if tail:
            parts = _split_nodes(tail, lines, last_end + 1, total,
                                 max_unit_lines, max_unit_chars,
                                 max_unit_tokens, unit_overlap_lines)
            if parts:
                for i, (ps, pe, nodes) in enumerate(parts, 1):
                    part_body = "\n".join(lines[ps - 1:pe])
                    calls: set[str] = set()
                    for n in nodes:
                        _walk_calls(n, calls)
                    chunks.append(Chunk(
                        file=rel_path,
                        qualname=("__trailing__" if i == 1
                                  else f"__trailing__#{i}"),
                        kind="module", start=ps, end=pe, imports=imports,
                        calls=sorted(calls), source=part_body,
                        sha=_sha(part_body),
                    ))
            else:
                body_text = "\n".join(lines[last_end:])
                calls: set[str] = set()
                for n in tail:
                    _walk_calls(n, calls)
                chunks.append(Chunk(
                    file=rel_path, qualname="__trailing__", kind="module",
                    start=last_end + 1, end=total, imports=imports,
                    calls=sorted(calls), source=body_text,
                    sha=_sha(body_text),
                ))

    chunks.sort(key=lambda c: c.start)
    return chunks


def chunk_file_ts(rel_path: str, src: str, ext: str,
                   class_split_lines: int = 60,
                   max_unit_lines: int = 220,
                   max_unit_chars: int = 12000,
                   max_unit_tokens: int = 3000,
                   unit_overlap_lines: int = 0) -> list[Chunk]:
    """Chunk one file with the tree-sitter chunker for its extension; the
    entry point index.py calls."""
    lang = LANG_BY_EXT[ext.lower()]
    if lang == "ruby":
        return chunk_source_ruby(rel_path, src, class_split_lines,
                                 max_unit_lines, max_unit_chars,
                                 max_unit_tokens, unit_overlap_lines)
    if lang in GENERIC_SPECS:
        return chunk_source_generic(rel_path, src, lang, class_split_lines,
                                    max_unit_lines, max_unit_chars,
                                    max_unit_tokens, unit_overlap_lines)
    return chunk_source_ts(rel_path, src, lang, class_split_lines,
                           max_unit_lines, max_unit_chars,
                           max_unit_tokens, unit_overlap_lines)


def has_syntax_error(text: str, language: str) -> tuple[bool, int | None]:
    """Return (has_error, line) for `text` parsed as `language`; line is the
    1-based line of the first ERROR node, else None. edits.py's post-edit
    syntax check, the counterpart of `ast.parse` for Python."""
    if not TS_AVAILABLE:
        # callers must skip the check when unavailable
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