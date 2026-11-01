"""Import extraction and resolution: which indexed file does each import name?

    specs = raw_imports("src/app.ts", source_text)      # full specifiers
    target = resolve("src/app.ts", "./lib/util", files) # -> ["src/lib/util.ts"]

The chunkers keep only short module labels per chunk (enough for a file card).
The import GRAPH needs full specifiers and the importing file's location, so it
is extracted here, once per file at index time, and stored on the Index. Every
language uses plain regexes (and `ast` for Python, so function-local and nested
imports count): cheap, dependency-free, and good enough for evidence about
blast radius. Resolution is deliberately conservative -- an import that cannot
be tied to an indexed file is "external" and creates no edge.
"""

from __future__ import annotations

import ast
import posixpath
import re

# ---------------------------------------------------------------- extraction
_JS_RES = (
    re.compile(r"""\bimport\s+(?:[^'"()]*?\bfrom\s*)?['"]([^'"\n]+)['"]"""),
    re.compile(r"""\bexport\s+[^'";]*?\bfrom\s*['"]([^'"\n]+)['"]"""),
    re.compile(r"""\brequire\s*\(\s*['"]([^'"\n]+)['"]\s*\)"""),
    re.compile(r"""\bimport\s*\(\s*['"]([^'"\n]+)['"]\s*\)"""),
)
_GO_BLOCK_RE = re.compile(r"\bimport\s*\((.*?)\)", re.DOTALL)
_GO_SPEC_RE = re.compile(r'(?:[\w.]+\s+)?"([^"\n]+)"')
_GO_SINGLE_RE = re.compile(r'^\s*import\s+(?:[\w.]+\s+)?"([^"\n]+)"', re.MULTILINE)
_RUST_USE_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?use\s+([^;]+);", re.MULTILINE)
_RUST_MOD_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;", re.MULTILINE)
_JAVA_RE = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+?)(?:\.\*)?\s*;", re.MULTILINE)
_CS_RE = re.compile(r"^\s*(?:global\s+)?using\s+(?:static\s+)?([\w.]+)\s*;", re.MULTILINE)
_RB_RE = re.compile(
    r"""^\s*(require|require_relative|load|autoload)\b[ (]*(?::\w+\s*,\s*)?['"]([^'"\n]+)['"]""",
    re.MULTILINE)
_C_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"\n]+)[>"]', re.MULTILINE)
_PY_STRING_IMPORT_RE = re.compile(
    r"""importlib\.import_module\(\s*['"]([\w.]+)['"]""")


def _expand_rust_use(body: str) -> list[str]:
    """`a::b::{c, d::e}` -> ["a::b::c", "a::b::d::e"] (nested groups are
    flattened recursively)."""
    body = re.sub(r"\s+", "", body)
    m = re.match(r"^(.*?)::\{(.*)\}$", body)
    if not m:
        return [body.replace("::*", "")]
    prefix, inner = m.group(1), m.group(2)
    out: list[str] = []
    depth, cur = 0, ""
    for ch in inner + ",":
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "," and depth == 0:
            if cur:
                out.extend(_expand_rust_use(f"{prefix}::{cur}"))
            cur = ""
        else:
            cur += ch
    return out


def _python_imports(src: str) -> list[str]:
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return []
    out: list[str] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            mod = ("." * (n.level or 0)) + (n.module or "")
            if mod:
                out.append(mod)
            # `from pkg import submodule` may name a module, not a symbol
            for a in n.names:
                if a.name != "*":
                    out.append(f"{mod}{a.name}" if (not mod or mod.endswith("."))
                               else f"{mod}.{a.name}")
    out += _PY_STRING_IMPORT_RE.findall(src)
    return out


def raw_imports(rel: str, src: str) -> list[str]:
    """Every module specifier `src` imports, in source order and de-duplicated:
    full relative paths for JS/TS/Ruby/C, dotted names for Python/Java/C#,
    `a::b::c` for Rust (`mod x;` becomes `self::x`), package paths for Go."""
    ext = rel.rsplit(".", 1)[-1].lower() if "." in rel else ""
    found: list[str] = []
    if ext == "py":
        found = _python_imports(src)
    elif ext in ("js", "jsx", "mjs", "cjs", "ts", "tsx"):
        for rx in _JS_RES:
            found += rx.findall(src)
    elif ext == "go":
        for block in _GO_BLOCK_RE.findall(src):
            found += _GO_SPEC_RE.findall(block)
        found += _GO_SINGLE_RE.findall(src)
    elif ext == "rs":
        for body in _RUST_USE_RE.findall(src):
            found += _expand_rust_use(body)
        found += [f"self::{m}" for m in _RUST_MOD_RE.findall(src)]
    elif ext == "java":
        found = _JAVA_RE.findall(src)
    elif ext == "cs":
        found = _CS_RE.findall(src)
    elif ext == "rb":
        found = [("./" + p if kind == "require_relative"
                  and not p.startswith(".") else p)
                 for kind, p in _RB_RE.findall(src)]
    elif ext in ("c", "h", "cc", "cpp", "cxx", "hpp", "hh"):
        found = _C_RE.findall(src)
    seen: set[str] = set()
    out: list[str] = []
    for s in found:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out[:200]


# ---------------------------------------------------------------- resolution
_JS_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")


class FileSet:
    """Lookup structures over the indexed paths, built once per graph."""

    def __init__(self, files):
        self.files = set(files)
        self.noext: dict[str, list[str]] = {}
        self.by_dir: dict[str, list[str]] = {}
        for f in self.files:
            base = posixpath.basename(f)
            stem = f.rsplit(".", 1)[0] if "." in base else f
            self.noext.setdefault(stem, []).append(f)
            self.by_dir.setdefault(posixpath.dirname(f), []).append(f)

    def suffix_match(self, path_noext: str, exts: tuple[str, ...]) -> list[str]:
        """Indexed files whose extension-less path equals or ends with
        `/path_noext`, restricted to `exts`."""
        out: list[str] = []
        tail = "/" + path_noext
        for stem, fs in self.noext.items():
            if stem == path_noext or stem.endswith(tail):
                out += [f for f in fs if f.lower().endswith(exts)]
        return sorted(out)


def _norm_join(base_dir: str, spec: str) -> str:
    """`spec` joined onto `base_dir` and normalised, or "" when it climbs out
    of the project root."""
    p = posixpath.normpath(posixpath.join(base_dir, spec))
    return "" if p == "." or p.startswith("..") else p.lstrip("/")


def _resolve_relative_file(fs: FileSet, base: str, exts: tuple[str, ...]
                           ) -> list[str]:
    """`base` (a joined, normalised path, with or without extension) to indexed
    files: the file itself, `base + ext`, or `base/index|mod|__init__`."""
    if not base:
        return []
    if base in fs.files:
        return [base]
    for e in exts:
        if base + e in fs.files:
            return [base + e]
    for idx in ("index", "mod", "__init__"):
        for e in exts:
            cand = f"{base}/{idx}{e}"
            if cand in fs.files:
                return [cand]
    return []


def resolve(rel: str, spec: str, fs: FileSet) -> list[str]:
    """Indexed files that import specifier `spec` (as written in `rel`) names;
    [] when it is external or cannot be tied to the index."""
    ext = rel.rsplit(".", 1)[-1].lower() if "." in rel else ""
    here = posixpath.dirname(rel)
    if ext in ("js", "jsx", "mjs", "cjs", "ts", "tsx"):
        if spec.startswith("."):
            base = _norm_join(here, spec)
            # TS projects import "./x.js" for x.ts
            base_noext = re.sub(r"\.(?:js|jsx|mjs|cjs)$", "", base)
            return (_resolve_relative_file(fs, base, _JS_EXTS)
                    or _resolve_relative_file(fs, base_noext, _JS_EXTS))
        # bare / alias specifier: only an in-repo path (`src/x`, `@/x`)
        trimmed = re.sub(r"^[@~]/", "", spec)
        for pre in ("", "src/"):
            got = _resolve_relative_file(fs, pre + trimmed, _JS_EXTS)
            if got:
                return got
        return []
    if ext == "py":
        return _resolve_python(rel, spec, fs)
    if ext == "go":
        parts = spec.split("/")
        for n in range(len(parts), 0, -1):
            d = "/".join(parts[-n:])
            hits = [f for dd, fl in fs.by_dir.items()
                    if dd == d or dd.endswith("/" + d)
                    for f in fl
                    if f.endswith(".go") and not f.endswith("_test.go")]
            # A single trailing segment is accepted only for a module-style
            # path (`example.com/x/store`), never for a bare std package name.
            if hits and (n >= 2 or "." in parts[0]):
                return sorted(hits)
        return []
    if ext == "rs":
        return _resolve_rust(rel, spec, fs)
    if ext == "java":
        return fs.suffix_match(spec.replace(".", "/"), (".java",))[:1]
    if ext == "rb":
        if spec.startswith("."):
            return _resolve_relative_file(fs, _norm_join(here, spec), (".rb",))
        return fs.suffix_match(spec, (".rb",))[:1]
    if ext in ("c", "h", "cc", "cpp", "cxx", "hpp", "hh"):
        joined = _norm_join(here, spec)
        if joined in fs.files:
            return [joined]
        want = posixpath.basename(spec)
        return [f for f in fs.suffix_match(spec.rsplit(".", 1)[0],
                (".h", ".hpp", ".hh", ".c", ".cc", ".cpp", ".cxx"))
                if f.endswith(want)][:1]
    return []            # C# namespaces do not map to files


def _resolve_python(rel: str, spec: str, fs: FileSet) -> list[str]:
    level = len(spec) - len(spec.lstrip("."))
    name = spec.lstrip(".")
    if level:
        base = posixpath.dirname(rel)
        for _ in range(level - 1):
            base = posixpath.dirname(base)
        path = posixpath.join(base, name.replace(".", "/")) if name else base
        return _resolve_relative_file(fs, posixpath.normpath(path), (".py",))
    path = name.replace(".", "/")
    exact = _resolve_relative_file(fs, path, (".py",))
    if exact:
        return exact
    # src-layout and nested roots: match on the path suffix
    hits = fs.suffix_match(path, (".py",))
    if not hits:
        hits = fs.suffix_match(path + "/__init__", (".py",))
    return hits[:1]


def _resolve_rust(rel: str, spec: str, fs: FileSet) -> list[str]:
    parts = [p for p in spec.split("::") if p and p != "*"]
    if not parts:
        return []
    first = parts[0]
    if first in ("std", "core", "alloc"):
        return []
    here = posixpath.dirname(rel)
    base_dir = here
    if first == "self":
        parts = parts[1:]
        # `mod x;` in lib.rs/main.rs/mod.rs -> sibling; else a child dir
        stem = posixpath.basename(rel).rsplit(".", 1)[0]
        if stem not in ("lib", "main", "mod"):
            base_dir = posixpath.join(here, stem)
    elif first == "super":
        base_dir = posixpath.dirname(here)
        parts = parts[1:]
        while parts and parts[0] == "super":
            base_dir = posixpath.dirname(base_dir)
            parts = parts[1:]
    elif first == "crate":
        parts = parts[1:]
        # crate root = nearest ancestor holding lib.rs/main.rs
        d = here
        while True:
            if any(posixpath.join(d, f) in fs.files
                   for f in ("lib.rs", "main.rs")):
                break
            nd = posixpath.dirname(d)
            if nd == d:
                break
            d = nd
        base_dir = d
    for n in range(len(parts), 0, -1):
        base = posixpath.join(base_dir, *parts[:n]) if base_dir \
            else "/".join(parts[:n])
        got = _resolve_relative_file(fs, posixpath.normpath(base), (".rs",))
        if got:
            return got
    if first not in ("crate", "self", "super"):
        # `use mycrate::a::b` inside a workspace: suffix match
        for n in range(len(parts), 0, -1):
            got = fs.suffix_match("/".join(parts[:n]), (".rs",))
            if got:
                return got[:1]
    return []
