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

import gc
import re
from importlib.metadata import PackageNotFoundError, version as _dist_version

from .chunker import (
    Chunk, _part_signature, _sha, _split_statements, split_lines,
)


def _version_tuple(raw: str) -> tuple[int, ...]:
    """A comparable (major, minor, patch) tuple from a version string."""
    parts: list[int] = []
    for piece in (raw or "").split(".")[:3]:
        m = re.match(r"\d+", piece)
        parts.append(int(m.group()) if m else 0)
    return tuple(parts)


def _tree_sitter_is_safe(version: str) -> bool:
    """False for tree-sitter >= 0.26, whose Node lifetime regression
    (py-tree-sitter#472) segfaults CPython while a live tree is walked.
    Pedacito pins `<0.26`; this guard keeps an environment that ignored the
    pin from crashing the whole process instead of using the fallback
    chunker."""
    return _version_tuple(version) < (0, 26)


_TS_DISABLED_REASON = ""
try:
    from tree_sitter_language_pack import get_parser
    _ts_version = ""
    try:
        _ts_version = _dist_version("tree-sitter")
    except PackageNotFoundError:                       # pragma: no cover
        pass
    if _ts_version and not _tree_sitter_is_safe(_ts_version):
        get_parser = None
        TS_AVAILABLE = False
        _TS_DISABLED_REASON = (
            f"tree-sitter {_ts_version} is unsafe (py-tree-sitter#472: "
            "segfault on live Node objects); install tree-sitter<0.26")
    else:
        TS_AVAILABLE = True
except ImportError:                                    # pragma: no cover
    get_parser = None
    TS_AVAILABLE = False
    _TS_DISABLED_REASON = "tree-sitter-language-pack is not installed"

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
    ".php": "php",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
    ".lua": "lua",
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


def _callee_name(fn) -> str:
    """Bare callee name of a call's function expression, across grammars:
    `f`, `obj.f` (JS member/Go selector/Rust field/C# member access),
    `a::b::f` (Rust/C++ scoped), and generic instantiations `f::<T>`/`F<T>`."""
    for _ in range(6):
        t = fn.type
        if t in ("identifier", "field_identifier", "property_identifier",
                 "type_identifier", "simple_identifier", "name"):
            return fn.text.decode()
        if t in ("navigation_expression", "member_expression_kt"):
            # Kotlin/Swift `a.b.f`: the last simple identifier
            ids = [c for c in fn.named_children
                   if c.type in ("simple_identifier", "navigation_suffix")]
            if not ids:
                return ""
            last = ids[-1]
            if last.type == "navigation_suffix":
                inner = [c for c in last.named_children
                         if c.type == "simple_identifier"]
                return inner[-1].text.decode() if inner else ""
            return last.text.decode()
        if t == "member_expression":
            n = fn.child_by_field_name("property")
        elif t in ("field_expression", "selector_expression"):
            n = fn.child_by_field_name("field")
        elif t in ("member_access_expression", "scoped_identifier",
                   "qualified_identifier", "scoped_type_identifier",
                   "member_binding_expression", "conditional_access_expression"):
            n = fn.child_by_field_name("name") \
                or (fn.named_children[-1] if fn.named_children else None)
        elif t in ("generic_function", "generic_name"):
            n = fn.child_by_field_name("function") or (
                fn.named_children[0] if fn.named_children else None)
        else:
            return ""
        if n is None:
            return ""
        fn = n
    return ""


# Function-scope node types across the supported grammars: entering one
# restarts the local-type map (parameter annotations only).
_FUNC_SCOPE_NODES = {
    "function_declaration", "generator_function_declaration",
    "function_expression", "function", "arrow_function", "generator_function",
    "method_definition", "method_declaration", "function_item",
    "function_definition", "constructor_declaration", "closure_literal",
    "func_literal", "init_declaration", "protocol_function_declaration",
}


# Binding statement node types whose local type a following call receiver
# can pick up (`x = new Foo(); x.bar()`), per grammar.
_BINDING_NODES = {
    "variable_declarator", "assignment_expression", "assignment",
    "let_declaration", "short_var_declaration", "var_spec", "const_spec",
    "local_variable_declaration", "field_declaration",
    "property_declaration",
}


def _text_id(node) -> str:
    """Trimmed source text of a node ("" for None)."""
    return node.text.decode("utf-8", "replace").strip() if node else ""


_IDENT_RE = re.compile(r"[A-Za-z_$][\w$]*\Z")


def _type_leaf(text: str) -> str:
    """The class name an annotation/declared-type text carries: `*Server`,
    `Server?`, `&Foo`, `Arc<Store>`-free simple names; "" for containers,
    unions and generics (`Record<..>`, `Vec<Store>`, `A | B`)."""
    t = text.strip().rstrip("?").lstrip("*&").strip()
    if not t or "<" in t or "|" in t or "." in t or " " in t:
        return ""
    return t if _IDENT_RE.match(t) and t[:1].isupper() else ""


def _node_type(node) -> str:
    """Class name a type node carries, or ""."""
    if node is None:
        return ""
    if node.type == "type_annotation":
        inner = next((c for c in node.named_children
                      if c.type != "predefined_type"), None)
        return _node_type(inner) if inner is not None \
            else _type_leaf(_text_id(node))
    if node.type == "named_type":
        nm = node.child_by_field_name("name")
        if nm is not None:
            return _type_leaf(_text_id(nm))
    return _type_leaf(_text_id(node))


def _ctor_type(value) -> str:
    """Class name if the initializer node is constructor evidence
    (`new Foo()`, `Foo()`, `Foo::new()`, `Foo {}`, `&Foo{}`), else ""."""
    if value is None:
        return ""
    t = value.type
    if t == "new_expression":
        return _type_leaf(_callee_name(value.child_by_field_name("constructor")))
    if t == "object_creation_expression":          # Java/C#/PHP `new Foo()`
        return _type_leaf(_text_id(value.child_by_field_name("type")
                                    or value.child_by_field_name("name")))
    if t == "struct_expression":                   # Rust `Foo { .. }`
        return _type_leaf(_text_id(value.child_by_field_name("type")
                                   or value.child_by_field_name("name")))
    if t == "composite_literal":                   # Go `Foo{..}` / `&Foo{..}`
        return _type_leaf(_text_id(value.child_by_field_name("type")))
    if t in ("call_expression", "invocation_expression",
            "function_call_expression"):
        fn = value.child_by_field_name("function")
        if fn is None:
            return ""
        if fn.type in ("identifier", "type_identifier"):
            return _type_leaf(_text_id(fn))
        if fn.type in ("scoped_identifier", "generic_function"):
            # `Foo::new()` / `Foo::default()` -- the conventional ctors
            if _callee_name(fn) in ("new", "default"):
                path = fn.child_by_field_name("path")
                txt = _text_id(path).replace("::", ".")
                return _type_leaf(txt.rsplit(".", 1)[-1]) if txt else ""
        return ""
    if t in ("member_expression", "selector_expression"):
        return ""
    return ""


def _binding_types(node) -> dict[str, str]:
    """{variable: ClassName} contributed by one binding/assignment statement:
    `x = new Foo()`, `let x: Foo`, `x := &Foo{}`, `Foo x = new()`."""
    out: dict[str, str] = {}
    t = node.type

    def bind(name_txt: str, cls: str) -> None:
        if name_txt and cls and _IDENT_RE.match(name_txt):
            out[name_txt] = cls

    if t == "variable_declarator":                 # JS/TS/C# const/let/var
        name = _text_id(node.child_by_field_name("name"))
        ann = next((c for c in node.children if c.type == "type_annotation"),
                   None)
        val = node.child_by_field_name("value")
        if val is None:                            # C# carries it unfielded
            val = next((c for c in reversed(node.named_children)
                        if c.type not in ("identifier", "type_annotation",
                                          "explicit_type")), None)
        bind(name, _node_type(ann) or _ctor_type(val))
    elif t == "assignment_expression":             # JS `db = new Store()`
        bind(_text_id(node.child_by_field_name("left")),
             _ctor_type(node.child_by_field_name("right")))
    elif t == "let_declaration":                   # Rust `let x: Foo` / ctor
        pat = node.child_by_field_name("pattern")
        ty = node.child_by_field_name("type")
        val = node.child_by_field_name("value") or \
            node.child_by_field_name("initializer")
        bind(_text_id(pat), _node_type(ty) or _ctor_type(val))
    elif t == "short_var_declaration":             # Go `x := &Foo{}`
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        names = _text_id(left) if left is not None else ""
        ctor = _ctor_type(right) if right is not None else ""
        if not ctor and right is not None:
            m = re.match(r"&?([A-Z]\w*)\{", _text_id(right))
            ctor = m.group(1) if m else ""
        for nm in names.split(","):
            bind(nm.strip().lstrip("*"), ctor)
    elif t in ("local_variable_declaration", "field_declaration",
                "property_declaration"):           # Java/C#/Kotlin
        ty = node.child_by_field_name("type")
        cls = _node_type(ty)
        if not cls:
            cls = _ctor_type(node.child_by_field_name("value"))
        for c in node.named_children:
            if c.type in ("variable_declarator", "variable_declarators",
                          "declaration_specifiers"):
                bind(_text_id(c.child_by_field_name("name") or c
                              ).split("=")[0].strip(), cls)
            elif c.type == "declarator":
                bind(_text_id(c.child_by_field_name("declarator")), cls)
            elif c.type in ("simple_identifier", "identifier"):
                bind(_text_id(c), cls)
    elif t in ("var_spec", "const_spec"):          # Go `var s Server` / `=`
        bind(_text_id(node.child_by_field_name("name")),
             _node_type(node.child_by_field_name("type"))
             or _ctor_type(node.child_by_field_name("value")))
    elif t == "assignment":                        # PHP `$store = new Store()`
        m = re.match(r"\s*(\$?[A-Za-z_]\w*)\s*=\s*new\s+([A-Z]\w*)",
                     _text_id(node))
        if m:
            bind(m.group(1), m.group(2))
        else:
            m = re.match(r"\s*(\$?[A-Za-z_]\w*)\s*=\s*([A-Z]\w*)::\s*"
                         r"(?:new|create|getInstance)\b", _text_id(node))
            if m:
                bind(m.group(1), m.group(2))
    return out


def _param_types(node) -> dict[str, str]:
    """{parameter: ClassName} from typed parameters of a function-like node
    (`def f(store: Store)`, `fn(x Store)`, `void run(Store s)`)."""
    out: dict[str, str] = {}
    params = node.child_by_field_name("parameters") or \
        node.child_by_field_name("parameter_bindings")
    if params is None:
        return out

    def pair(p):
        nm = p.child_by_field_name("name") or \
            next((c for c in p.named_children
                  if c.type in ("identifier", "simple_identifier")), None)
        ty = p.child_by_field_name("type") or next(
            (c for c in p.named_children
             if c.type in ("type_annotation", "type_identifier", "generic_type",
                           "named_type", "pointer_type", "reference_type",
                           "trait_type")), None)
        if nm is not None and ty is not None:
            name = _text_id(nm)
            cls = _node_type(ty)
            if name and cls:
                out[name] = cls

    stack = list(params.named_children)
    while stack:
        c = stack.pop()
        if c.type in ("parameter", "required_parameter", "optional_parameter",
                      "typed_parameter", "parameter_declaration",
                      "formal_parameter", "receiver", "typed_pattern"):
            pair(c)
        else:
            stack.extend(c.named_children)
    return out


def _receiver_qual(fn, name, cls, self_ids, vars_) -> str:
    """`ClassName.method` for a member-call callee node whose receiver is
    decidable: a self-ref inside class context, a locally bound variable, or
    a capitalized static receiver."""
    recv = None
    t = fn.type if fn is not None else ""
    if t == "member_expression":                   # JS / C++ / Java field
        recv = fn.child_by_field_name("object")
    elif t == "selector_expression":               # Go `s.Stop()`
        recv = fn.child_by_field_name("operand")
    elif t == "field_expression":                  # Rust `store.load()`
        recv = fn.child_by_field_name("value")
    elif t in ("member_access_expression",):       # C# `Helper.Do()`
        recv = fn.child_by_field_name("expression")
    elif t in ("scoped_identifier",):              # Rust `Foo::new()`
        recv = fn.child_by_field_name("path")
    elif t in ("navigation_expression",):          # Kotlin/Swift `s.stop()`
        kids = [c for c in fn.named_children
                if c.type in ("simple_identifier", "navigation_suffix")]
        if len(kids) >= 2:
            last = kids[-1]
            if last.type == "navigation_suffix":
                inner = [c for c in last.named_children
                         if c.type == "simple_identifier"]
                if inner and _text_id(inner[-1]) == name:
                    recv = kids[-2]
    if recv is None:
        return ""
    return _qualify(_text_id(recv), name, cls, self_ids, vars_)


def _qualify(recv: str, name: str, cls: str, self_ids, vars_) -> str:
    """Class-qualify call `name` given its receiver text."""
    if not recv:
        return ""
    if recv in self_ids and cls:
        return f"{cls}.{name}"
    if _IDENT_RE.match(recv) and recv in vars_:
        return f"{vars_[recv]}.{name}"
    if _IDENT_RE.match(recv) and recv[:1].isupper():
        return f"{recv}.{name}"
    return ""


def _member_qual(sc, name, cls, self_ids, vars_) -> str:
    """Same, for Java/C# method_invocation scopes; a dotted static scope
    (`System.Console.WriteLine()`) qualifies by its last capitalized leaf."""
    if sc is None:
        return ""
    txt = _text_id(sc)
    q = _qualify(txt, name, cls, self_ids, vars_)
    if q:
        return q
    leaf = txt.replace("::", ".").rstrip(".").rsplit(".", 1)[-1]
    if leaf[:1].isupper() and _IDENT_RE.match(leaf):
        return f"{leaf}.{name}"
    return ""


def _walk_calls(node, out: set[str], cls: str = "",
                self_ids: tuple[str, ...] = (),
                vmap: dict[str, str] | None = None) -> None:
    """Collect names invoked via call/new expressions under `node`. Member
    calls (obj.method()) reduce to the property name: the same bare-name
    approximation chunker.py uses for Python -- and, Sprint 4, the SAME
    receiver-aware qualification: ``ClassName.method`` is recorded beside the
    bare name when the receiver is decidable at chunk time (`cls` + a
    self-receiver id from `self_ids`, a local bound by `x = new Foo()` /
    `x: Foo` / `x := &Foo{}` / `Foo::new()`-style evidence, or a receiver
    literally capitalized in the source). Iterative, so deeply nested trees
    (minified JS) cannot hit the recursion limit."""
    stack = [(node, cls, vmap if vmap is not None else {})]
    while stack:
        node, cls_ctx, vars_ = stack.pop()
        t = node.type
        qual = ""
        if t in ("call_expression", "invocation_expression"):
            fn = node.child_by_field_name("function")
            if fn is None and node.named_children:
                fn = node.named_children[0]       # Kotlin/Swift: no field
            name = _callee_name(fn) if fn is not None else ""
            if name:
                out.add(name)
                qual = _receiver_qual(fn, name, cls_ctx, self_ids, vars_)
        elif t in ("function_call_expression",):             # PHP
            fn = node.child_by_field_name("function")
            name = _callee_name(fn) if fn is not None else ""
            if name:
                out.add(name)
                qual = _receiver_qual(fn, name, cls_ctx, self_ids, vars_)
        elif t in ("member_call_expression", "scoped_call_expression",
                   "nullsafe_member_call_expression"):       # PHP
            nm = node.child_by_field_name("name")
            if nm is not None:
                out.add(nm.text.decode())
                obj = (node.child_by_field_name("object")
                       or node.child_by_field_name("scope"))   # `Store::make()`
                qual = _qualify(_text_id(obj), nm.text.decode(), cls_ctx,
                                self_ids, vars_)
        elif t == "function_call":                           # Lua
            nm = node.child_by_field_name("name")
            if nm is not None:
                txt = nm.text.decode()
                out.add(txt.replace(":", ".").split(".")[-1])
                if cls_ctx and txt.startswith(("self.", "self:")):
                    qual = f"{cls_ctx}.{txt.replace(':', '.').split('.')[-1]}"
        elif t == "new_expression":
            ctor = node.child_by_field_name("constructor")
            if ctor is not None and ctor.type == "identifier":
                out.add(ctor.text.decode())
        elif t == "method_invocation":                       # Java / C#
            nm = node.child_by_field_name("name")
            if nm is not None:
                out.add(nm.text.decode())
                sc = (node.child_by_field_name("scope")
                      or node.child_by_field_name("object"))
                qual = _member_qual(sc, nm.text.decode(), cls_ctx,
                                    self_ids, vars_)
        elif t == "object_creation_expression":
            ty = node.child_by_field_name("type")
            if ty is not None:
                out.add(ty.text.decode().split(".")[-1])
        elif t == "macro_invocation":
            mac = node.child_by_field_name("macro")
            if mac is not None:
                out.add(mac.text.decode().rstrip("!"))
        if qual:
            out.add(qual)
        nxt_cls, nxt_map = cls_ctx, vars_
        if t in _FUNC_SCOPE_NODES:
            # a new function body: locals restart with this signature's
            # parameter types; the class context (`this`/`self`) survives --
            # methods' nested functions/arrow functions still resolve it
            nxt_map = dict(_param_types(node))
        elif t in _BINDING_NODES:
            # in place: statement siblings pushed with this same dict see the
            # binding -- `x = new Foo(); x.bar()` qualifies like chunker.py
            nxt_map.update(_binding_types(node))
        stack.extend((ch, nxt_cls, nxt_map) for ch in reversed(node.children))


def _module_chunks(rel_path: str, nodes: list, lines: list[str], first: int,
                   last: int, imports: list[str], base_name: str,
                   max_unit_lines: int, max_unit_chars: int,
                   max_unit_tokens: int, unit_overlap_lines: int,
                   with_imports: bool = True) -> list[Chunk]:
    """Chunk(s) for a run of top-level statements (a preamble, a trailing
    block, or a whole script with no scopes). A block larger than the unit
    limits is split at statement boundaries, as Python's chunker does, so a
    plain script cannot become one unbounded `module` chunk."""
    out: list[Chunk] = []
    if last < first:
        return out
    parts = _split_nodes(nodes, lines, first, last, max_unit_lines,
                         max_unit_chars, max_unit_tokens, unit_overlap_lines)
    if not parts:
        body = "\n".join(lines[first - 1:last])
        if not body.strip():
            return out
        calls: set[str] = set()
        mv = {}
        for n in nodes:
            _walk_calls(n, calls, vmap=mv)
        out.append(Chunk(
            file=rel_path, qualname=base_name, kind="module", start=first,
            end=last, imports=imports if with_imports else [],
            calls=sorted(calls), source=body, sha=_sha(body)))
        return out
    for i, (ps, pe, part_nodes) in enumerate(parts, 1):
        part_body = "\n".join(lines[ps - 1:pe])
        if not part_body.strip():
            continue
        calls = set()
        mv = {}
        for n in part_nodes:
            _walk_calls(n, calls, vmap=mv)
        out.append(Chunk(
            file=rel_path, qualname=base_name if i == 1 else f"{base_name}#{i}",
            kind="module", start=ps, end=pe,
            imports=imports if with_imports else [], calls=sorted(calls),
            source=part_body, sha=_sha(part_body)))
    return out


def _module_label(spec: str) -> str:
    """Collapse an import specifier to a short label for the file card's
    "imports:" line ("./a/foo" -> "foo"; index.py strips leading dots, which
    would otherwise empty a bare relative path)."""
    spec = spec.rstrip("/")
    if spec.startswith("."):
        return spec.rsplit("/", 1)[-1] or spec
    return spec


def _imports(root, src: bytes) -> list[str]:
    """Module specifiers of every top-level import_statement, plus CommonJS
    `require("x")`, dynamic `import("x")` and `export ... from "x"` anywhere
    in the file (labels, de-duplicated)."""
    from .imports import raw_imports
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
    for spec in raw_imports("x.js", src.decode("utf-8", "replace")):
        label = _module_label(spec)
        if label not in out:
            out.append(label)
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
            chunks.extend(_module_chunks(
                rel_path, list(root.children), lines, 1, total, imports,
                "module", max_unit_lines, max_unit_chars, max_unit_tokens,
                unit_overlap_lines))
        return chunks

    # module preamble: everything before the first scope
    first_start = scopes[0][0].start_point.row + 1
    if first_start > 1:
        end = first_start - 1
        pre = [n for n in root.children if n.start_point.row + 1 <= end]
        chunks.extend(_module_chunks(
            rel_path, pre, lines, 1, end, imports, "module", max_unit_lines,
            max_unit_chars, max_unit_tokens, unit_overlap_lines))

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
                fv = {}
                for i, (ps, pe, nodes) in enumerate(parts, 1):
                    part_body = "\n".join(lines[ps - 1:pe])
                    p_calls: set[str] = set()
                    for n in nodes:
                        _walk_calls(n, p_calls, vmap=fv)
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
                        mv = {}
                        for i, (ps, pe, nodes) in enumerate(parts, 1):
                            part_body = "\n".join(lines[ps - 1:pe])
                            p_calls: set[str] = set()
                            for n in nodes:
                                _walk_calls(n, p_calls, cls=name,
                                            self_ids=("this",), vmap=mv)
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
                    _walk_calls(m, calls, cls=name, self_ids=("this",))
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
                fv = {}
                for i, (ps, pe, nodes) in enumerate(parts, 1):
                    part_body = "\n".join(lines[ps - 1:pe])
                    p_calls: set[str] = set()
                    for n in nodes:
                        _walk_calls(n, p_calls, vmap=fv)
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
            _walk_calls(inner, calls, cls=name, self_ids=("this",))
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
                mv: dict[str, str] = {}
                for i, (ps, pe, nodes) in enumerate(parts, 1):
                    part_body = "\n".join(lines[ps - 1:pe])
                    p_calls: set[str] = set()
                    for n in nodes:
                        _walk_calls(n, p_calls, cls=name, self_ids=("this",),
                                    vmap=mv)
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
            _walk_calls(m, m_calls, cls=name, self_ids=("this",))
            chunks.append(Chunk(
                file=rel_path, qualname=f"{name}.{m_name}", kind="method",
                start=m_start, end=m_end, signature=m_sig,
                decorators=m_decs, calls=sorted(m_calls),
                imports=imports, source=m_body, sha=_sha(m_body),
            ))

    # trailing top-level code after the last scope
    last_end = scopes[-1][0].end_point.row + 1
    if last_end < total:
        tail = [n for n in root.children if n.start_point.row + 1 > last_end]
        chunks.extend(_module_chunks(
            rel_path, tail, lines, last_end + 1, total, imports,
            "__trailing__", max_unit_lines, max_unit_chars, max_unit_tokens,
            unit_overlap_lines, with_imports=False))

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
        # `mod x { ... }` with a body: its items are chunked (qualified `x.`)
        "namespaces": {"mod_item"},
        "namespace_prefix": True,
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
    "php": {
        "functions": {"function_definition"},
        "containers": {"class_declaration", "interface_declaration",
                       "trait_declaration", "enum_declaration"},
        "methods": {"method_declaration"},
        "decls": set(),
        "imports": {"namespace_use_declaration"},
        "namespaces": {"namespace_definition"},
        "namespace_prefix": False,
    },
    "kotlin": {
        "functions": {"function_declaration"},
        "containers": {"class_declaration", "object_declaration"},
        "methods": {"function_declaration"},
        "decls": set(),
        "imports": {"import_list", "import_header"},
    },
    "swift": {
        "functions": {"function_declaration"},
        "containers": {"class_declaration", "protocol_declaration"},
        "methods": {"function_declaration", "init_declaration",
                    "protocol_function_declaration"},
        "decls": set(),
        "imports": {"import_declaration"},
    },
    "scala": {
        "functions": {"function_definition"},
        "containers": {"class_definition", "trait_definition",
                       "object_definition"},
        "methods": {"function_definition"},
        "decls": set(),
        "imports": {"import_declaration"},
    },
    "lua": {
        "functions": {"function_declaration"},
        "containers": set(),
        "methods": set(),
        "decls": set(),
        "imports": set(),
    },
    "csharp": {
        "functions": {"method_declaration", "constructor_declaration"},
        "containers": {"class_declaration", "interface_declaration",
                       "struct_declaration", "enum_declaration",
                       "record_declaration"},
        "methods": {"method_declaration", "constructor_declaration"},
        "decls": set(),
        "imports": {"using_directive"},
        # Block-scoped `namespace X { ... }`: the types inside are chunked
        # (file-scoped `namespace X;` already leaves them at top level).
        "namespaces": {"namespace_declaration"},
        "namespace_prefix": False,
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
        if child.type in ("type_spec", "type_identifier", "identifier",
                          "simple_identifier", "name"):
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
        "compound_statement", "body", "enum_declaration_list",
        "enum_class_body", "template_body", "protocol_body",
    }
    for c in node.named_children:
        if c.type in body_types:
            return c
    return None


def _namespace_body(node):
    """The declaration list of a namespace/mod node, or None (`mod x;`)."""
    b = node.child_by_field_name("body")
    if b is not None:
        return b
    return next((c for c in node.named_children
                 if c.type == "declaration_list"), None)


def _walk_namespaces(nodes, spec):
    """Yield every node in `nodes`, descending into namespace bodies."""
    for node in nodes:
        yield node
        if node.type in spec.get("namespaces", ()):
            body = _namespace_body(node)
            if body is not None:
                yield from _walk_namespaces(body.named_children, spec)


def _go_receiver(node, src: bytes) -> str:
    """Receiver type name of a Go method declaration ("" for functions)."""
    return _go_receiver_pair(node, src)[0]


def _go_receiver_pair(node, src: bytes) -> tuple[str, str]:
    """(receiver type, receiver variable) of a Go method declaration; Go has
    no `self` keyword -- the named receiver IS the self reference."""
    recv = node.child_by_field_name("receiver")
    if recv is None:
        return "", ""
    text = _text(recv, src).strip("() \t\n")
    m = re.match(r"^([A-Za-z_]\w*)\s+\*?([A-Za-z_]\w*)$", text)
    if m:
        return m.group(2), m.group(1)
    m = re.search(r"\*?\s*([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*\)?\s*$", text)
    return (m.group(1) if m else ""), ""


# The spelled self-reference of each generic language ("" where the receiver
# is implicit: Kotlin/Swift `stop()` on self never needs a qualifier).
_SELF_REFS = {
    "rust": ("self",),
    "java": ("this",),
    "c": ("this",),
    "cpp": ("this",),
    "csharp": ("this",),
    "php": ("$this",),
    "kotlin": ("this",),
    "swift": ("self",),
    "scala": ("this",),
}


def _rust_impl_label(node, src: bytes, name: str) -> str:
    """`impl Foo` / `impl Trait for Foo` as the impl chunk's qualname, so the
    struct, its inherent impls and its trait impls stay distinguishable."""
    trait = node.child_by_field_name("trait")
    if trait is not None:
        t = _text(trait, src).split("<", 1)[0].strip()
        return f"impl {t} for {name}"
    return f"impl {name}"


def _generic_imports(root, spec, src: bytes) -> list[str]:
    out: list[str] = []
    for node in _walk_namespaces(root.named_children, spec):
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

    scopes: list[tuple[object, object, str, str]] = []   # outer, inner, kind, prefix
    preamble: list = []
    seen_scope = False
    ns_types = spec.get("namespaces", ())

    def classify(nodes, prefix: str) -> None:
        nonlocal seen_scope
        for node in nodes:
            inner = node
            if node.type == "template_declaration":
                inner = next((c for c in node.named_children
                              if c.type in spec["functions"]
                              or c.type in spec["containers"]
                              or c.type in spec["decls"]), node)
            if inner.type in ns_types and _namespace_body(inner) is not None:
                ns_name = _generic_name(inner, src_bytes)
                nxt = (prefix + ns_name + ".") if spec.get(
                    "namespace_prefix") else prefix
                classify(_namespace_body(inner).named_children, nxt)
            elif inner.type in spec["functions"]:
                scopes.append((node, inner, "function", prefix))
                seen_scope = True
            elif inner.type in spec["containers"]:
                scopes.append((node, inner, "container", prefix))
                seen_scope = True
            elif inner.type in spec["decls"]:
                scopes.append((node, inner, "decl", prefix))
                seen_scope = True
            elif inner.type in spec["imports"]:
                continue
            elif not seen_scope:
                preamble.append(node)

    classify(root.named_children, "")

    self_refs = _SELF_REFS.get(language, ())

    def emit_function(outer, inner, name, kind="function", cls="",
                      self_ids=()):
        start, end = _line_span(outer)
        sig = _generic_signature(outer, src_bytes)
        body = inner.child_by_field_name("body")
        parts = _split_body(body, lines, start, end, max_unit_lines,
                            max_unit_chars, max_unit_tokens,
                            unit_overlap_lines)
        if parts:
            fv = {}
            for i, (ps, pe, nodes) in enumerate(parts, 1):
                part_body = "\n".join(lines[ps - 1:pe])
                calls: set[str] = set()
                for n in nodes:
                    _walk_calls(n, calls, cls=cls, self_ids=self_ids, vmap=fv)
                chunks.append(Chunk(
                    file=rel_path,
                    qualname=name if i == 1 else f"{name}#{i}",
                    kind=kind, start=ps, end=pe,
                    signature=_part_signature(sig, i, len(parts)),
                    calls=sorted(calls), imports=imports,
                    source=part_body, sha=_sha(part_body),
                ))
            return
        calls: set[str] = set()
        _walk_calls(inner, calls, cls=cls, self_ids=self_ids)
        body_text = "\n".join(lines[start - 1:end])
        chunks.append(Chunk(
            file=rel_path, qualname=name, kind=kind,
            start=start, end=end, signature=sig, calls=sorted(calls),
            imports=imports, source=body_text, sha=_sha(body_text),
        ))

    def emit_container(outer, inner, name, label=None):
        """`name` qualifies member methods (`Foo.bar`); `label` is the
        container chunk's own qualname (`impl Foo` for a Rust impl). The
        language's self-reference resolves against `name` while walking."""
        label = label or name
        cls_ctx = name
        self_ids = self_refs
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
                            _walk_calls(n, calls, cls=cls_ctx, self_ids=self_ids)
                        chunks.append(Chunk(
                            file=rel_path,
                            qualname=label if i == 1
                            else f"{label}#{i}",
                            kind="class", start=ps, end=pe,
                            signature=_part_signature(sig, i, len(parts)),
                            calls=sorted(calls), imports=imports,
                            source=part_body, sha=_sha(part_body),
                        ))
                    return
            calls: set[str] = set()
            _walk_calls(inner, calls, cls=cls_ctx, self_ids=self_ids)
            body_text = "\n".join(lines[start - 1:end])
            chunks.append(Chunk(
                file=rel_path, qualname=label, kind="class",
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
                    _walk_calls(n, calls, cls=cls_ctx, self_ids=self_ids)
                chunks.append(Chunk(
                    file=rel_path,
                    qualname=label if i == 1 else f"{label}#{i}",
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
                file=rel_path, qualname=label, kind="classheader",
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
                        _walk_calls(n, calls, cls=cls_ctx, self_ids=self_ids)
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
            _walk_calls(m, calls, cls=cls_ctx, self_ids=self_ids)
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

    for outer, inner, kind, prefix in scopes:
        name = _generic_name(inner, src_bytes)
        if kind == "function":
            if language == "go" and inner.type == "method_declaration":
                rtype, rval = _go_receiver_pair(inner, src_bytes)
                if rtype:
                    emit_function(outer, inner, f"{rtype}.{name}", "method",
                                  cls=rtype, self_ids=(rval,) if rval else ())
                else:
                    emit_function(outer, inner, name, "method")
            else:
                emit_function(outer, inner, prefix + name)
        elif kind == "container":
            label = _rust_impl_label(inner, src_bytes, name) \
                if inner.type == "impl_item" else None
            if label and prefix:
                label = prefix + label
            emit_container(outer, inner, prefix + name, label)
        else:
            start, end = _line_span(outer)
            body_text = "\n".join(lines[start - 1:end])
            calls: set[str] = set()
            _walk_calls(inner, calls)
            chunks.append(Chunk(
                file=rel_path, qualname=prefix + name, kind="class",
                start=start, end=end, signature=_generic_signature(
                    outer, src_bytes),
                calls=sorted(calls), imports=imports,
                source=body_text, sha=_sha(body_text),
            ))

    # trailing top-level code
    if scopes:
        last_end = max(_line_span(o)[1] for o, _i, _k, _p in scopes)
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
    entry point index.py calls.

    The cyclic garbage collector is paused while tree-sitter nodes are alive:
    py-tree-sitter 0.26 can segfault when a collection runs over live Node
    objects (upstream py-tree-sitter#472). The dependency is pinned to
    `<0.26`; this is the second line of defence for an environment that still
    has 0.26 installed. Nodes are freed by reference counting when the chunker
    returns, so nothing accumulates."""
    lang = LANG_BY_EXT[ext.lower()]
    was_enabled = gc.isenabled()
    gc.disable()
    try:
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
    finally:
        if was_enabled:
            gc.enable()


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