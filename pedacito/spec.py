"""Prompt/Spec Architect and cross-file interface contracts.

Two related jobs, both aimed at the way 9B-35B models lose their grip on
names over a multi-step plan:

  1. ``refine_prompt()`` turns a brief idea ("add CSV export") into a
     concrete, repo-aware specification -- Objective / Files to Create or
     Modify / Interface Contracts / Requirements & Edge Cases / Verification
     -- from one schema-constrained model call, with a deterministic fallback
     when no model answers (GUI "✦ Refine Prompt", CLI ``--refine``).

  2. ``extract_project_contracts()`` / ``verify_cross_file_contracts()`` are
     the contract-first interface lock for ``auto`` and ``new``: every step
     of a plan re-reads the exact public signatures the project already has
     (so Step K cannot forget Step 1's function name), and a deterministic
     AST gate checks every local ``from x import y`` and call arity across
     files, feeding the violation straight into the repair prompt.

All checks here are deterministic and cost zero model tokens: the contract
checks run in-process with ``ast`` and regexes, and the static analysis gate
prefers a bounded, five-rule ``ruff`` subprocess with a pure-AST fallback when
ruff cannot run. Same rule everywhere -- checks are deliberately conservative:
a false positive would send a working project into a repair round, so anything
ambiguous is skipped rather than reported (the same rule `apicheck` follows).
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import functools
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from . import fileio, webcheck
from .chunker import DEF_NODES, _docline, _signature

# Directories no project check should ever walk into.
SKIP_DIRS = {".git", ".pedacito", ".hg", ".svn", "node_modules", "__pycache__",
             ".venv", "venv", "env", "build", "dist", ".tox", ".mypy_cache",
             ".pytest_cache", ".ruff_cache", "site-packages", "bower_components",
             ".next", ".nuxt", "coverage", "htmlcov"}

PY_EXTS = {".py"}
TS_EXTS = {".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx"}
DOC_EXTS = {".md", ".rst", ".txt", ".html", ".htm", ".css", ".json", ".toml",
            ".yaml", ".yml", ".sql", ".sh"}

MAX_FILE_BYTES = 400_000       # a file larger than this is not scanned


def _get(cfg, name: str, default):
    """cfg.<name> when cfg provides it, else `default` (cfg may be None)."""
    if cfg is None:
        return default
    val = getattr(cfg, name, None)
    return default if val is None else val


def _skip_dir(name: str) -> bool:
    return name in SKIP_DIRS or name.startswith(".")


def _iter_code_files(root: Path, exts: set[str], cap: int = 400) -> list[str]:
    """Sorted project-relative paths with one of `exts`, bounded and never
    crossing into vendored/hidden directories. Empty when root is absent."""
    root = Path(root)
    if not root.is_dir():
        return []
    out: list[str] = []
    stack = [root]
    while stack and len(out) < cap:
        d = stack.pop()
        try:
            entries = sorted(d.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        dirs = [e for e in entries if e.is_dir() and not _skip_dir(e.name)]
        for e in dirs:
            stack.append(e)
        for e in entries:
            if e in dirs or not e.is_file():
                continue
            if e.suffix.lower() in exts and e.stat().st_size <= MAX_FILE_BYTES:
                rel = e.relative_to(root).as_posix()
                out.append(rel)
                if len(out) >= cap:
                    break
    out.sort()
    return out


def _read(rel: str, root: Path) -> str:
    try:
        return fileio.read_text(root / rel, errors="replace")
    except OSError:
        return ""


def _is_test_file(rel: str) -> bool:
    base = rel.rsplit("/", 1)[-1]
    parts = rel.split("/")
    return (base.startswith("test_") or base.endswith("_test.py")
            or base in ("conftest.py",) or "tests" in parts
            or bool(re.search(r"\.(test|spec)\.[jt]sx?$", base)))


# ---------------------------------------------------------------- outlines ---

def _py_outline(src: str) -> list[str] | None:
    """Public top-level signatures of one Python source, as
    ["def foo(x, y=1) -> int", ...]; None when the file does not parse."""
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return None
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, DEF_NODES):
            if node.name.startswith("_"):
                continue
            sig = _signature(node)
            if sig:
                out.append(sig)
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            sig = _signature(node)
            init = next((n for n in node.body
                         if isinstance(n, DEF_NODES) and n.name == "__init__"),
                        None)
            if init is not None:
                args = ", ".join(
                    a for a in ast.unparse(init.args).split(", ")
                    if a not in ("self", "cls") and not a.startswith(("self:", "cls:")))
                sig += f": __init__({args})"
            out.append(sig)
    return out


_JS_FUNC_RE = re.compile(
    r"^\s*(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s*\(?\s*"
    r"([A-Za-z_$][\w$]*)\s*\(([^)]{0,120})\)")
_JS_CLASS_RE = re.compile(
    r"^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)\s*"
    r"(?:extends\s+([A-Za-z_$][\w$.]*))?")
_JS_ARROW_RE = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
    r"(?:async\s*)?\(([^)]{0,120})\)\s*=>")
_JS_METHOD_RE = re.compile(
    r"^\s{2,}(?:static\s+)?(?:async\s+)?([A-Za-z_$][\w$]*)\s*\(([^)]{0,100})\)\s*\{")
_JS_SKIP_METHOD = {"if", "for", "while", "switch", "catch", "function", "return",
                   "constructor"}


def _ts_outline(src: str) -> list[str]:
    """Public top-level signatures of one JS/TS source (regex scan: exports,
    functions, classes, and the constructor of each class)."""
    out: list[str] = []
    lines = src.splitlines()
    i = 0
    in_class = False
    ctor_done = True
    while i < len(lines) and len(out) < 60:
        ln = lines[i]
        m = _JS_FUNC_RE.match(ln)
        if m and not ln.lstrip().startswith("//"):
            out.append(f"function {m.group(1)}({m.group(2).strip()})")
            i += 1
            continue
        m = _JS_CLASS_RE.match(ln)
        if m and not ln.lstrip().startswith("//"):
            name = m.group(1)
            if not name.startswith("_"):
                out.append(f"class {name}"
                           + (f" extends {m.group(2)}" if m.group(2) else ""))
                in_class = True
                ctor_done = False
            else:
                in_class = True
                ctor_done = True
            i += 1
            continue
        m = _JS_ARROW_RE.match(ln)
        if m and not ln.lstrip().startswith("//"):
            if not m.group(1).startswith("_"):
                out.append(f"const {m.group(1)} = ({m.group(2).strip()}) => {{...}}")
            i += 1
            continue
        if in_class and not ctor_done:
            m = _JS_METHOD_RE.match(ln)
            if m and m.group(1) == "constructor" and m.group(1) not in _JS_SKIP_METHOD:
                out[-1] += f": constructor({m.group(2).strip()})"
                ctor_done = True
                i += 1
                continue
        if ln and not ln.isspace() and not ln[0].isspace() and "{" not in ln:
            in_class = False
        i += 1
    return out


def _outline(rel: str, root: Path) -> list[str]:
    src = _read(rel, root)
    if not src.strip():
        return []
    ext = "." + rel.rsplit(".", 1)[-1].lower() if "." in rel else ""
    if ext in PY_EXTS:
        got = _py_outline(src)
        return got or []
    if ext in TS_EXTS:
        return _ts_outline(src)
    return []


# ------------------------------------------------- project interface contract
#
# extract_project_contracts: a compact "who exports what, exactly" map of the
# project's public Python/JS/TS surface, ~300 tokens. run_loop injects it (via
# contract_map_block) ahead of every Auto task so each step can see the
# signatures the earlier steps wrote.

_FIXTURE_DECORATOR_BASES = {"pytest", "pytest_asyncio"}
_FIXTURE_HEADER = "# pytest fixtures (conftest.py)"


def _fixture_decorator(node: ast.AST) -> tuple[bool, str]:
    """(is_fixture, scope) for one decorator node.

    Accepts `@fixture` and `@pytest.fixture` (plus the asyncio alias), with or
    without a call. Scope is the literal keyword when present; anything more
    elaborate is treated as unspecified rather than guessed."""
    call = node if isinstance(node, ast.Call) else None
    fn = call.func if call is not None else node
    ok = False
    if isinstance(fn, ast.Name):
        ok = fn.id == "fixture"
    elif isinstance(fn, ast.Attribute) and fn.attr == "fixture":
        ok = (isinstance(fn.value, ast.Name)
              and fn.value.id in _FIXTURE_DECORATOR_BASES)
    if not ok:
        return False, ""
    scope = ""
    if call is not None:
        for kw in call.keywords:
            if (kw.arg == "scope" and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, str)):
                scope = kw.value.value
                break
    return True, scope


def _fixture_hint(node: ast.AST) -> str:
    """One truncated line describing a fixture: its docstring, else a yield
    hint (`yields <expr>`) when it is a generator fixture."""
    doc = _docline(node)
    if doc:
        return " ".join(doc.split())[:80]
    for sub in ast.walk(node):
        if isinstance(sub, (ast.Yield, ast.YieldFrom)):
            value = getattr(sub, "value", None)
            if value is not None:
                try:
                    expr = ast.unparse(value)
                except Exception:
                    expr = ""
                if expr:
                    return ("yields " + " ".join(expr.split()))[:80]
            return "yields"
    return ""


def _conftest_files(root: Path, cap: int = 6) -> list[str]:
    """Project-relative conftest.py paths, bounded, root-level first."""
    rels = [rel for rel in _iter_code_files(root, PY_EXTS, cap=400)
            if rel.rsplit("/", 1)[-1] == "conftest.py"]
    rels.sort(key=lambda r: (r.count("/"), r))
    return rels[:cap]


def extract_pytest_fixtures(root: Path, max_fixtures: int = 8,
                            max_chars: int = 700) -> str:
    """Compact `# pytest fixtures (conftest.py)` subsection, or "".

    Walks every discovered conftest.py with `ast.parse` (zero subprocesses,
    zero model calls), captures each `@pytest.fixture` / `@fixture` function's
    exact signature, literal scope, and a one-line docstring/yield hint. The
    result is bounded and deterministic so it can ride inside CONTRACT_HEADER
    without evicting the project's real public surface."""
    root = Path(root)
    rows: list[tuple[str, int, str]] = []
    seen: set[tuple[str, str]] = set()
    for rel in _conftest_files(root):
        src = _read(rel, root)
        if not src.strip():
            continue
        try:
            tree = ast.parse(src)
        except (SyntaxError, ValueError):
            continue
        for node in tree.body:
            if not isinstance(node, DEF_NODES):
                continue
            fixture, scope = False, ""
            for dec in node.decorator_list:
                hit, dec_scope = _fixture_decorator(dec)
                if hit:
                    fixture = True
                    scope = scope or dec_scope
                    break
            if not fixture:
                continue
            sig = _signature(node)
            if not sig:
                continue
            key = (rel, sig)
            if key in seen:
                continue
            seen.add(key)
            hint = _fixture_hint(node)
            line = f"- {rel}: {sig}"
            if scope:
                line += f" [scope={scope}]"
            if hint:
                line += f" -- {hint}"
            rows.append((rel, node.lineno, line))
    if not rows:
        return ""
    rows.sort(key=lambda r: (r[0], r[1]))
    out: list[str] = [_FIXTURE_HEADER]
    used = len(_FIXTURE_HEADER)
    for _rel, _line, text in rows:
        if len(out) - 1 >= max_fixtures or used + len(text) + 1 > max_chars:
            break
        out.append(text)
        used += len(text) + 1
    return "\n".join(out) if len(out) > 1 else ""


def _task_mentions_tests(task: str) -> bool:
    return bool(re.search(
        r"(?i)\b(?:tests?|testing|pytest|fixtures?|conftest|test_[a-z0-9_]*|"
        r"[a-z0-9_]+_test)\b", str(task or "")))


def extract_project_contracts(root: Path, max_files: int = 25,
                              max_chars: int = 1400,
                              modified: list[str] | None = None,
                              task: str = "",
                              include_fixtures: bool | None = None) -> str:
    """A compact map of the project's public signatures as
    `rel: def foo(x) -> y | class Bar` lines, capped near 300 tokens.

    `modified` (paths or rel strings, in any mix) sort those files first so
    the freshest contracts survive the cap; test files and unparsable files
    are skipped. When the repository has conftest.py files (or the task names
    tests, and a conftest exists), a bounded `# pytest fixtures` subsection is
    appended *inside* the same char budget so the model reuses real fixtures
    instead of inventing or duplicating them. Returns "" when the project has
    nothing to lock."""
    root = Path(root)
    rels = _iter_code_files(root, PY_EXTS | TS_EXTS, cap=120)
    want: set[str] = set()
    for m in modified or ():
        name = fileio.normalise_rel(str(m))
        if name.startswith("./"):
            name = name[2:]
        want.add(name)
    try:
        root_res = root.resolve()
        for m in modified or ():
            p = Path(str(m))
            if p.is_absolute():
                want.add(p.resolve().relative_to(root_res).as_posix())
    except (OSError, ValueError):
        pass
    order = {rel: i for i, rel in enumerate(rels)}
    rels.sort(key=lambda r: (0 if r in want else 1, order[r]))

    # Fixtures are included when a conftest.py exists and the caller has not
    # explicitly disabled them. A test-shaped task (or a modified test file)
    # gets the fuller subsection; other tasks still see a compact one, because
    # reusing a real fixture beats importing one nowhere.
    fixture_block = ""
    test_shaped = (_task_mentions_tests(task)
                   or any(_is_test_file(r) for r in want))
    if include_fixtures is not False and _conftest_files(root):
        fixture_block = extract_pytest_fixtures(
            root, max_fixtures=8 if test_shaped else 5,
            max_chars=(min(700, max(200, max_chars // 2)) if test_shaped
                       else min(420, max(160, max_chars // 3))))
    fixture_cost = (len(fixture_block) + 1) if fixture_block else 0
    map_budget = max(0, max_chars - fixture_cost)

    lines: list[str] = []
    used = 0
    files = 0
    for rel in rels:
        if files >= max_files or used >= map_budget:
            break
        if _is_test_file(rel):
            continue
        sigs = _outline(rel, root)
        if not sigs:
            continue
        entry = f"{rel}: " + "; ".join(sigs[:10])
        if len(entry) > 300:
            entry = entry[:297] + "..."
        if used + len(entry) > map_budget and lines:
            break
        lines.append(entry)
        used += len(entry) + 1
        files += 1
    if fixture_block:
        lines.append(fixture_block)
    return "\n".join(lines)


CONTRACT_HEADER = ("[PROJECT INTERFACE CONTRACTS — DO NOT BREAK SIGNATURES]")
CONTRACT_FOOTER = "[END INTERFACE CONTRACTS]"


def contract_map_block(root: Path, modified: list[str] | None = None,
                       task: str = "") -> str:
    """The interface map as a ready-to-inject instruction block, or "" when
    the project exposes nothing yet (a near-empty tree gets no block)."""
    cmap = extract_project_contracts(root, modified=modified, task=task)
    if not cmap:
        return ""
    return (CONTRACT_HEADER + "\n"
            "These are the exact public signatures the project has right now. "
            "Import and call them by these names and arities; never rename or "
            "re-order a parameter of an existing one. If your task must change "
            "a signature, update every caller in the same task.\n"
            + cmap + "\n" + CONTRACT_FOOTER)


# --------------------------------------------------------- contract verifier
#
# verify_cross_file_contracts: deterministic AST pass over the project's .py
# files. Two checks, both deliberately narrow:
#
#   * `from <local> import <symbol>` where <local> is a module/package that
#     exists in the project but never defines <symbol> (or a relative import
#     whose target module is gone) -- the exact error shape scaffold's import
#     check only catches by running the code.
#   * a direct call of such an imported name whose positional/keyword count
#     cannot satisfy the target's def/__init__ (defaults, *args/**kwargs and
#     star-args at the call site respected, same rules as apicheck).
#
# Anything ambiguous is skipped: unknown absolute modules are treated as
# third-party, shadowed or reassigned names are trusted, classes without an
# explicit __init__ are never arity-checked.

_PY = PY_EXTS


class _Mod:
    """One local module: its file, the names it exports, and its defs."""

    __slots__ = ("rel", "exports", "defs")

    def __init__(self, rel: str):
        self.rel = rel
        self.exports: set[str] = set()
        self.defs: dict[str, ast.AST] = {}


def _collect_exports(info: _Mod, tree: ast.AST) -> None:
    """Every name the module could resolve at import time: defs, classes,
    module-level assignments, top-level imports (re-exports), and names bound
    in if/try blocks (compat shims). Walk one level deep per top-level
    statement so a function-local name never counts as an export."""
    def add_target(t) -> None:
        if isinstance(t, ast.Name):
            info.exports.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                add_target(e)

    def visit(node) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            info.exports.add(node.name)
            info.defs.setdefault(node.name, node)
            return                      # do not descend into the body
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                add_target(t)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                info.exports.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, (ast.If, ast.Try)):
            for sub in (getattr(node, "body", []) + getattr(node, "orelse", [])
                        + getattr(node, "finalbody", [])
                        + list(getattr(node, "handlers", ()))):
                if isinstance(sub, ast.ExceptHandler):
                    if sub.name:
                        info.exports.add(sub.name)
                    for b in sub.body:
                        visit(b)
                else:
                    visit(sub)
        elif isinstance(node, (ast.For, ast.While, ast.With)):
            for sub in node.body + getattr(node, "orelse", ()):
                visit(sub)

    for stmt in tree.body:
        visit(stmt)


def _local_py_modules(root: Path) -> dict[str, _Mod]:
    """{dotted module name: _Mod} for the project's .py files, plus package
    roots (a dir with __init__.py maps to the package name)."""
    root = Path(root)
    mods: dict[str, _Mod] = {}
    for rel in _iter_code_files(root, _PY, cap=300):
        modname = rel[:-3].replace("/", ".")
        if modname.endswith(".__init__"):
            modname = modname[: -len(".__init__")]
        src = _read(rel, root)
        try:
            tree = ast.parse(src) if src.strip() else None
        except (SyntaxError, ValueError):
            tree = None
        info = _Mod(rel)
        if tree is not None:
            _collect_exports(info, tree)
        mods[modname] = info
    return mods


def _package_children(mods: dict[str, _Mod], pkg: str) -> set[str]:
    """Submodule names exported by a package: `from pkg import x` is fine when
    `pkg.x` exists."""
    return {name[len(pkg) + 1:] for name in mods
            if name.startswith(pkg + ".")}


def _arity_sig(node: ast.AST):
    """(required positional names, max positional, all param names) for a def
    or a class's __init__; None when the arity cannot be checked."""
    from .apicheck import _sig
    if isinstance(node, DEF_NODES):
        args = node.args
        pos = args.posonlyargs + args.args
        kwonly = [a.arg for a in args.kwonlyargs]
        names = [a.arg for a in pos] + kwonly
        if args.vararg:
            names.append(args.vararg.arg)
        if args.kwarg:
            names.append(args.kwarg.arg)
        return _sig(args, False), names
    if isinstance(node, ast.ClassDef):
        init = next((n for n in node.body
                     if isinstance(n, DEF_NODES) and n.name == "__init__"), None)
        if init is None:
            return None            # inherited/generated constructor: unknown
        args = init.args
        pos = args.posonlyargs + args.args
        if pos and pos[0].arg in ("self", "cls"):
            pos = pos[1:]
        names = ([a.arg for a in pos] + [a.arg for a in args.kwonlyargs]
                 + ([args.vararg.arg] if args.vararg else [])
                 + ([args.kwarg.arg] if args.kwarg else []))
        return _sig(args, True), names
    return None


def _file_package(rel: str, mods: dict[str, _Mod]) -> str:
    """The package a file lives in ("" at the top level), trusted only while
    every directory level appears in the module map (so has an __init__)."""
    parts = rel.split("/")[:-1]
    for i in range(1, len(parts) + 1):
        if ".".join(parts[:i]) not in mods:
            return ".".join(parts[:i - 1])
    return ".".join(parts)


def _local_defs_names(info: _Mod) -> list[str]:
    """Short, sorted, deduped list of a module's most definition-like names
    for the 'only defines [...]' half of a diagnostic."""
    names = sorted(set(info.defs) | {e for e in info.exports
                                     if e[0].isupper() or "Error" in e})
    return names[:6]


# ------------------------------------------- JS/TS ES module contracts
# The browser-side twin of the Python import/arity gate. The tree-sitter
# grammar the indexer already ships parses both sides; relative ES imports
# must resolve to a real project file and their named/default bindings must
# exist in that file's exports, and a direct call of an imported local
# function is checked against its known parameter range. Bare packages
# (`react`, `alpinejs`), dynamic `import(...)`, namespace imports, `...rest`
# parameters, functions that touch `arguments`, and any ambiguous shape are
# skipped -- a false positive would cost a repair round on working code.

_JS_FUNCTION_NODES = {
    "function_declaration", "generator_function_declaration",
    "function_expression", "arrow_function", "function",
}
_JS_DECL_NODES = {
    "function_declaration", "generator_function_declaration",
    "class_declaration", "abstract_class_declaration",
    "interface_declaration", "type_alias_declaration",
    "enum_declaration", "internal_module", "module",
}
_JS_IMPORT_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".json")


def _ts_text(node) -> str:
    return node.text.decode("utf-8", "replace") if node is not None else ""


def _walk_ts(node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.named_children))


def _ts_parser_for(suffix: str):
    try:
        from . import chunker_ts
        language = chunker_ts.LANG_BY_EXT.get(suffix)
        if not language or not chunker_ts.TS_AVAILABLE:
            return None
        return chunker_ts._get_parser(language)
    except Exception:
        return None


def _ts_guard():
    """The chunker's GC-pause guard, or a no-op when tree-sitter is absent."""
    try:
        from .chunker_ts import ts_nodes
        return ts_nodes()
    except Exception:
        return contextlib.nullcontext()


def _ts_guarded(fn):
    """Keep the cyclic collector paused for a whole parse+traverse helper.

    A tree-sitter Node is only safe while its tree is alive; returning a root
    node and then walking it with the collector enabled would recreate the
    py-tree-sitter#472 window inside the spec contract passes."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with _ts_guard():
            return fn(*args, **kwargs)
    return wrapped


def _ts_parse(text: str, suffix: str):
    parser = _ts_parser_for(suffix)
    if parser is None:
        return None
    try:
        with _ts_guard():
            tree = parser.parse(text.encode("utf-8", "replace"))
            return tree.root_node if not tree.root_node.has_error else None
    except Exception:
        return None


def _js_pattern_names(node) -> set[str]:
    """Identifiers bound by a JS/TS binding pattern (never default values)."""
    if node is None:
        return set()
    k = node.type
    if k in ("identifier", "shorthand_property_identifier_pattern"):
        return {_ts_text(node)}
    if k in ("rest_pattern",):
        return _js_pattern_names(node.named_children[0]
                                 if node.named_children else None)
    if k in ("assignment_pattern", "required_parameter", "optional_parameter"):
        target = node.child_by_field_name("pattern") \
            or node.child_by_field_name("left") \
            or (node.named_children[0] if node.named_children else None)
        return _js_pattern_names(target)
    if k in ("object_pattern", "array_pattern", "pair_pattern"):
        out: set[str] = set()
        for child in node.named_children:
            if child.type == "pair_pattern":
                out |= _js_pattern_names(
                    child.child_by_field_name("value"))
            elif child.type == "property_identifier":
                continue
            else:
                out |= _js_pattern_names(child)
        return out
    return set()


def _ts_param_count(params) -> tuple[int, int] | None:
    """(min_required, max_args) for a parameter node, or None when arity
    cannot be checked (rest parameter, unknown syntax)."""
    if params is None:
        return (0, 0)
    if params.type == "identifier":
        return (1, 1)
    if params.type != "formal_parameters":
        return None
    required = 0
    total = 0
    for param in params.named_children:
        if param.type == "comment":
            continue
        kind = param.type
        if kind == "rest_pattern":
            return None
        if kind in ("required_parameter", "optional_parameter"):
            pattern = param.child_by_field_name("pattern") or param
            if _ts_text(pattern).strip() == "this":
                continue
            if any(n.type == "rest_pattern" for n in _walk_ts(pattern)):
                return None
            total += 1
            defaulted = param.child_by_field_name("value") is not None
            if not (defaulted or kind == "optional_parameter"):
                required += 1
        elif kind == "identifier":
            total += 1
            required += 1
        elif kind == "assignment_pattern":
            total += 1
        elif kind in ("object_pattern", "array_pattern"):
            total += 1
            required += 1
        else:
            return None                  # unknown param shape: never guess
    return (required, total)


def _js_function_spec(node, name: str = ""):
    """(min_args, max_args, shown_signature) for one function-like node, or
    None when the parameter range cannot be proven."""
    params = node.child_by_field_name("parameters")
    if params is None and node.type == "arrow_function":
        params = node.child_by_field_name("parameter")
    body = node.child_by_field_name("body")
    if body is not None and re.search(r"(?<![\w$.])arguments(?![\w$])",
                                      _ts_text(body)):
        return None                      # arguments[] makes arity opaque
    count = _ts_param_count(params)
    if count is None:
        return None
    param_text = _ts_text(params) if params is not None else "()"
    if not param_text.startswith("("):
        param_text = f"({param_text})"
    if node.type == "arrow_function":
        shown = f"const {name or '(anonymous)'} = {param_text} => {{...}}"
    else:
        shown = f"function {name or '(anonymous)'}{param_text}"
    return (count[0], count[1], shown[:140])


def _js_add_local(info: dict, node) -> set[str]:
    """Record one top-level declaration in `info["locals"]`; returns names."""
    added: set[str] = set()
    if node is None:
        return added
    if node.type in ("function_declaration", "generator_function_declaration"):
        name_node = node.child_by_field_name("name")
        name = _ts_text(name_node)
        if name:
            info["locals"][name] = _js_function_spec(node, name)
            added.add(name)
        return added
    if node.type in ("class_declaration", "abstract_class_declaration"):
        name = _ts_text(node.child_by_field_name("name"))
        if name:
            info["locals"][name] = None
            added.add(name)
        return added
    if node.type in ("interface_declaration", "type_alias_declaration",
                     "enum_declaration", "internal_module", "module"):
        name = _ts_text(node.child_by_field_name("name"))
        if name:
            info["locals"][name] = None
            added.add(name)
        return added
    if node.type in ("function_signature", "method_signature"):
        name = _ts_text(node.child_by_field_name("name"))
        if name:
            info["locals"][name] = _js_function_spec(node, name)
            added.add(name)
        return added
    if node.type == "ambient_declaration":
        for child in node.named_children:
            added |= _js_add_local(info, child)
        return added
    if node.type in ("lexical_declaration", "variable_declaration"):
        for child in node.named_children:
            if child.type != "variable_declarator":
                continue
            names = _js_pattern_names(child.child_by_field_name("name"))
            if not names:
                continue
            value = child.child_by_field_name("value")
            spec = (_js_function_spec(value, next(iter(names)))
                    if value is not None and value.type in _JS_FUNCTION_NODES
                    and len(names) == 1 else None)
            for name in names:
                info["locals"][name] = spec
                added.add(name)
    return added


def _js_add_export_info(info: dict, node, text: str) -> None:
    """Update exports/opaque from one export_statement node."""
    is_default = bool(re.match(r"^\s*export\s+default\b", text))
    decl = node.child_by_field_name("declaration")
    value = node.child_by_field_name("value")
    source = node.child_by_field_name("source")
    clause = next((c for c in node.named_children
                   if c.type == "export_clause"), None)
    namespace = next((c for c in node.named_children
                      if c.type == "namespace_export"), None)

    if is_default:
        if decl is not None:
            # `_js_add_local` also makes a named default declaration callable
            # by a later `export { name }`; it is not a named export itself.
            _js_add_local(info, decl)
            if decl.type in ("function_declaration",
                             "generator_function_declaration"):
                info["exports"]["default"] = _js_function_spec(
                    decl, _ts_text(decl.child_by_field_name("name")))
            else:
                info["exports"]["default"] = None
        elif value is not None:
            if value.type in _JS_FUNCTION_NODES:
                info["exports"]["default"] = _js_function_spec(value)
            elif value.type == "identifier":
                info["exports"]["default"] = info["locals"].get(
                    _ts_text(value))
            else:
                info["exports"]["default"] = None
        else:
            info["exports"]["default"] = None
        return

    if decl is not None:
        before = set(info["locals"])
        _js_add_local(info, decl)
        for name in set(info["locals"]) - before:
            info["exports"][name] = info["locals"][name]
        return
    if clause is not None:
        for spec in clause.named_children:
            if spec.type != "export_specifier":
                continue
            imported = _ts_text(spec.child_by_field_name("name"))
            alias = _ts_text(spec.child_by_field_name("alias"))
            exported = alias or imported
            if not exported:
                continue
            # An export alias (especially in minified/vendored code) no
            # longer proves the parameter list the caller sees: keep the
            # export name for existence checks but never arity-check it.
            if source is not None or (alias and alias != imported):
                info["exports"][exported] = None
            else:
                info["exports"][exported] = info["locals"].get(imported)
        return
    if namespace is not None:
        name = _ts_text(namespace.named_children[0]
                        if namespace.named_children else None)
        if name:
            info["exports"][name] = None
        return
    if source is not None:
        info["opaque"] = True             # `export * from` hides the surface
        return
    if re.match(r"^\s*export\s*(?:=|\bas\s+namespace\b)", text):
        info["opaque"] = True             # CommonJS / UMD-shaped surface


@_ts_guarded
def _js_module_info(text: str, suffix: str) -> dict | None:
    """Exports/locals/opaque/module facts for one JS/TS source, or None when
    the parser cannot read it (the syntax gate owns that failure)."""
    root = _ts_parse(text, suffix)
    if root is None:
        return None
    info = {"exports": {}, "locals": {}, "opaque": False, "module": False}
    for child in root.named_children:
        if child.type == "import_statement":
            info["module"] = True
        elif child.type == "export_statement":
            info["module"] = True
            _js_add_export_info(info, child, _ts_text(child))
        elif child.type in _JS_DECL_NODES or child.type in (
                "lexical_declaration", "variable_declaration"):
            _js_add_local(info, child)
        elif child.type == "expression_statement":
            for sub in child.named_children:
                if sub.type != "assignment_expression":
                    continue
                left = sub.child_by_field_name("left")
                if left is None or left.type != "member_expression":
                    continue
                obj = _ts_text(left.child_by_field_name("object"))
                prop = _ts_text(left.child_by_field_name("property"))
                if obj == "module" and prop == "exports" or obj == "exports":
                    info["module"] = True
                    info["opaque"] = True
    return info


def _js_imports_and_reexports(root) -> tuple[list[dict], list[dict]]:
    """Relative import and `export ... from` statements with exact lines."""
    imports: list[dict] = []
    reexports: list[dict] = []
    for node in root.named_children:
        if node.type == "import_statement":
            source = node.child_by_field_name("source")
            spec = _ts_text(source).strip().strip("'\"")
            if not spec:
                continue
            clause = next((c for c in node.named_children
                           if c.type == "import_clause"), None)
            default = ""
            named: list[tuple[str, str]] = []
            namespace = ""
            if clause is not None:
                for child in clause.children:
                    if child.type == "identifier":
                        default = _ts_text(child)
                    elif child.type == "named_imports":
                        for spec_node in child.named_children:
                            if spec_node.type != "import_specifier":
                                continue
                            imported = _ts_text(
                                spec_node.child_by_field_name("name"))
                            alias = _ts_text(
                                spec_node.child_by_field_name("alias"))
                            if imported:
                                named.append((alias or imported, imported))
                    elif child.type == "namespace_import":
                        ident = next((x for x in child.named_children
                                      if x.type == "identifier"), None)
                        namespace = _ts_text(ident)
            imports.append({
                "spec": spec, "line": node.start_point[0] + 1,
                "default": default, "named": named, "namespace": namespace,
                "type_only": bool(re.match(r"^\s*import\s+type\b",
                                           _ts_text(node))),
            })
            continue
        if node.type != "export_statement":
            continue
        source = node.child_by_field_name("source")
        spec = _ts_text(source).strip().strip("'\"")
        if not spec:
            continue
        clause = next((c for c in node.named_children
                       if c.type == "export_clause"), None)
        namespace = next((c for c in node.named_children
                          if c.type == "namespace_export"), None)
        entry = {"spec": spec, "line": node.start_point[0] + 1,
                 "named": [], "star": False}
        if clause is not None:
            for spec_node in clause.named_children:
                if spec_node.type != "export_specifier":
                    continue
                imported = _ts_text(spec_node.child_by_field_name("name"))
                alias = _ts_text(spec_node.child_by_field_name("alias"))
                if imported:
                    entry["named"].append((alias or imported, imported))
        elif namespace is not None:
            entry["star"] = False          # existence-only, like a namespace
        else:
            entry["star"] = True
        reexports.append(entry)
    return imports, reexports


def _js_resolve_import(root: Path, from_rel: str, spec: str) -> str:
    """Project-relative path an ES specifier resolves to, or "" when it is
    not a verifiable project-relative file."""
    spec = (spec or "").strip()
    if not spec.startswith(".") or spec in (".", ".."):
        return ""
    if any(ch in spec for ch in "?#\x00") or "\\" in spec:
        return ""
    root_res = root.resolve()
    base = (root / from_rel).parent
    try:
        absolute = (base / spec).resolve()
        absolute.relative_to(root_res)
    except (OSError, ValueError):
        return ""                           # escapes the project: not guessed
    candidates: list[Path] = []
    suffix = absolute.suffix.lower()
    if suffix:
        candidates.append(absolute)
        if suffix in (".js", ".jsx", ".mjs", ".cjs"):
            for alt in (".ts", ".tsx"):
                candidates.append(absolute.with_suffix(alt))
    else:
        candidates.append(absolute)
        candidates.extend(absolute.with_name(absolute.name + ext)
                          for ext in _JS_IMPORT_EXTS)
        candidates.extend(absolute / ("index" + ext)
                          for ext in _JS_IMPORT_EXTS)
    for cand in candidates:
        try:
            if cand.is_file():
                return cand.relative_to(root_res).as_posix()
        except (OSError, ValueError):
            continue
    return ""


def _js_shadowed_names(root, imported: set[str]) -> set[str]:
    """Imported locals that are redeclared anywhere in the importing file."""
    shadowed: set[str] = set()

    def visit(node) -> None:
        if node.type == "import_statement":
            return
        if node.type in ("function_declaration",
                         "generator_function_declaration",
                         "class_declaration", "interface_declaration",
                         "type_alias_declaration", "enum_declaration"):
            name = _ts_text(node.child_by_field_name("name"))
            if name in imported:
                shadowed.add(name)
        elif node.type == "variable_declarator":
            names = _js_pattern_names(node.child_by_field_name("name"))
            shadowed.update(names & imported)
        elif node.type == "formal_parameters":
            for param in node.named_children:
                names = _js_pattern_names(param)
                shadowed.update(names & imported)
            return                         # defaults are uses, not bindings
        elif node.type == "catch_clause":
            for child in node.named_children:
                if child.type in ("identifier", "object_pattern",
                                  "array_pattern"):
                    shadowed.update(_js_pattern_names(child) & imported)
        elif node.type == "assignment_expression":
            left = node.child_by_field_name("left")
            if left is not None and left.type == "identifier":
                name = _ts_text(left)
                if name in imported:
                    shadowed.add(name)
        for child in node.named_children:
            visit(child)

    visit(root)
    return shadowed


def _js_export_label(exports) -> list[str]:
    return sorted(exports)[:6]


@_ts_guarded
def _js_cross_file_errors(root: Path, cap: int = 60
                          ) -> list[tuple[str, str, str]]:
    """(importer_rel, target_rel, ContractError) for JS/TS module contracts.

    Deterministic, in-process, and deliberately quiet on every shape a
    static checker cannot decide (bare packages, dynamic imports, namespace
    imports, `export *`, CommonJS, unparsable files, shadowed locals,
    rest/arguments arity)."""
    root = Path(root)
    rels = _iter_code_files(root, TS_EXTS, cap=300)
    if not rels:
        return []
    info_cache: dict[str, dict | None] = {}

    def info_for(rel: str) -> dict | None:
        if rel not in info_cache:
            info_cache[rel] = _js_module_info(_read(rel, root),
                                              "." + rel.rsplit(".", 1)[-1].lower())
        return info_cache[rel]

    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    def emit(importer: str, target: str, text: str) -> None:
        if text in seen or len(out) >= cap:
            return
        seen.add(text)
        out.append((importer, target, text))

    for rel in rels:
        text = _read(rel, root)
        if not text.strip():
            continue
        suffix = "." + rel.rsplit(".", 1)[-1].lower()
        root_node = _ts_parse(text, suffix)
        if root_node is None:
            continue                       # syntax gate owns broken files
        imports, reexports = _js_imports_and_reexports(root_node)
        if not imports and not reexports:
            continue
        imported_funcs: dict[str, tuple[str, tuple[int, int, str]]] = {}
        import_counts: dict[str, int] = {}
        for item in imports:
            for local, _imported in item["named"]:
                import_counts[local] = import_counts.get(local, 0) + 1
            if item["default"]:
                import_counts[item["default"]] = \
                    import_counts.get(item["default"], 0) + 1
        for item in imports:
            target = _js_resolve_import(root, rel, item["spec"])
            if not target:
                if item["spec"].startswith("."):
                    emit(rel, "", f"ContractError in {rel}:{item['line']}: "
                         f"imports '{item['spec']}', but no such file exists "
                         f"in the project -- create it or fix the path")
                continue
            target_info = info_for(target)
            if target_info is None or not target_info["module"] \
                    or target_info["opaque"]:
                continue                   # not provable: only existence above
            exports = target_info["exports"]
            if item["default"] and "default" not in exports:
                emit(rel, target, f"ContractError in {rel}:{item['line']}: "
                     f"imports the default export from {target}, but {target} "
                     f"has no default export (exports: "
                     f"{_js_export_label(exports)})")
            for local, imported in item["named"]:
                if imported not in exports:
                    emit(rel, target, f"ContractError in {rel}:{item['line']}: "
                         f"imports '{imported}' from {target}, but {target} "
                         f"does not export it (exports: "
                         f"{_js_export_label(exports)})")
                    continue
                spec = exports.get(imported)
                if (spec is not None and not item["type_only"]
                        and import_counts.get(local, 0) == 1):
                    imported_funcs[local] = (target, spec)
            if item["default"] and "default" in exports and not item["type_only"]:
                spec = exports.get("default")
                if (spec is not None
                        and import_counts.get(item["default"], 0) == 1):
                    imported_funcs[item["default"]] = (target, spec)

        for item in reexports:
            target = _js_resolve_import(root, rel, item["spec"])
            if not target:
                if item["spec"].startswith("."):
                    emit(rel, "", f"ContractError in {rel}:{item['line']}: "
                         f"re-exports '{item['spec']}', but no such file exists "
                         f"in the project -- create it or fix the path")
                continue
            if item["star"]:
                continue
            target_info = info_for(target)
            if target_info is None or not target_info["module"] \
                    or target_info["opaque"]:
                continue
            for exported, imported in item["named"]:
                if imported not in target_info["exports"]:
                    emit(rel, target, f"ContractError in {rel}:{item['line']}: "
                         f"re-exports '{imported}' from {target}, but {target} "
                         f"does not export it (exports: "
                         f"{_js_export_label(target_info['exports'])})")

        if not imported_funcs:
            continue
        shadowed = _js_shadowed_names(root_node, set(imported_funcs))
        for node in _walk_ts(root_node):
            if node.type != "call_expression":
                continue
            fn = node.child_by_field_name("function")
            if fn is None or fn.type != "identifier":
                continue
            local = _ts_text(fn)
            if local in shadowed or local not in imported_funcs:
                continue
            target, (minimum, maximum, shown) = imported_funcs[local]
            args = node.child_by_field_name("arguments")
            if args is None:
                continue
            if any(c.type == "spread_element" for c in args.named_children):
                continue                   # spread can satisfy any arity
            argc = sum(1 for c in args.named_children if c.type != "comment")
            if argc < minimum or argc > maximum:
                line = node.start_point[0] + 1
                want = (f"exactly {minimum}" if minimum == maximum
                        else f"{minimum}..{maximum}")
                emit(rel, target, f"ContractError in {rel}:{line}: calls "
                     f"{local}(...) with {argc} argument(s) imported from "
                     f"{target}, where the function is defined as {shown} and "
                     f"expects {want} argument(s)")
    out.sort(key=lambda t: (t[0], t[2]))
    return out


# The name vocabulary lives in `webcheck`: one list of what the host, a CDN
# bundle or a test runner provides, shared by the regex advisor and this gate.
# Two lists drifted once, and the two readers disagreed about whether a bare
# name needed an import, so there is only one now and it cannot drift again.
# It is a suppression list, so the merge was curated rather than unioned:
# every real Web API either side knew went in, and what was never a global at
# all stayed out -- HTML tag names (`Menu`, `Table`, `Form`, `Title`) and
# misspelled interfaces. A call to a `Menu()` nobody defined is exactly the
# finding this check exists to make, and the old list had made it unspeakable.
_JS_GLOBAL_NAMES = webcheck.JS_GLOBAL_NAMES
_JS_GLOBAL_PREFIXES = webcheck.JS_GLOBAL_PREFIXES
_JS_NON_NAMES = webcheck.JS_NON_NAMES
_JS_LIBRARY_GLOBALS = webcheck.JS_LIBRARY_GLOBALS
_JS_TEST_GLOBALS = webcheck.JS_TEST_GLOBALS

# A name only ever defined inside another file's function body is not a global;
# this is the generous "does this word get defined ANYWHERE in the project"
# pool that keeps the undefined-name check from guessing.
_JS_ANY_DEF_RE = re.compile(
    r"\b(?:function|class)\s*\*?\s*([A-Za-z_$][\w$]*)"
    r"|\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)"
    r"|\b(?:window|globalThis|self|global)\s*(?:\.\s*|\[\s*['\"])([A-Za-z_$][\w$]*)"
    r"|\b([A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?(?:function\b|\([^()]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)"
)


def _js_top_level_bindings(root_node) -> list[tuple[str, int, str]]:
    """(name, line, kind) for every binding a file's *top level* introduces.

    Only program children: a declaration inside a block or a class body is a
    different scope and cannot collide with anything here. `kind` is `import`,
    `lexical` (const/let/class), `function` or `hoisted` (var). A classic
    script lets `function` and `hoisted` repeat; an ES module only lets two
    `hoisted` bindings of one name coexist."""
    out: list[tuple[str, int, str]] = []

    def declarator_names(node) -> list[str]:
        names: list[str] = []
        for child in node.named_children:
            if child.type != "variable_declarator":
                continue
            names += sorted(_js_pattern_names(child.child_by_field_name("name")))
        return names

    def add(name: str, node, kind: str) -> None:
        if name and name not in _JS_NON_NAMES:
            out.append((name, node.start_point[0] + 1, kind))

    for child in root_node.named_children:
        k = child.type
        if k == "import_statement":
            clause = next((c for c in child.named_children
                           if c.type == "import_clause"), None)
            if clause is None:
                continue
            for spec in clause.children:
                if spec.type == "identifier":
                    add(_ts_text(spec), child, "import")
                elif spec.type == "named_imports":
                    for node in spec.named_children:
                        if node.type != "import_specifier":
                            continue
                        alias = _ts_text(node.child_by_field_name("alias"))
                        add(alias or _ts_text(
                            node.child_by_field_name("name")),
                            node, "import")
                elif spec.type == "namespace_import":
                    ident = next((x for x in spec.named_children
                                  if x.type == "identifier"), None)
                    add(_ts_text(ident), child, "import")
        elif k in ("lexical_declaration", "variable_declaration"):
            kind = "lexical" if re.match(r"^\s*(?:const|let)\b",
                                         _ts_text(child)) else "hoisted"
            for name in declarator_names(child):
                add(name, child, kind)
        elif k in ("function_declaration", "generator_function_declaration",
                   "class_declaration", "abstract_class_declaration"):
            add(_ts_text(child.child_by_field_name("name")), child,
                "lexical" if "class" in k else "function")
        elif k == "export_statement":
            decl = child.child_by_field_name("declaration")
            if decl is None:
                continue
            if decl.type in ("lexical_declaration", "variable_declaration"):
                kind = "lexical" if re.match(r"^\s*(?:const|let)\b",
                                             _ts_text(decl)) else "hoisted"
                for name in declarator_names(decl):
                    add(name, decl, kind)
            elif decl.type in ("function_declaration",
                               "generator_function_declaration"):
                add(_ts_text(decl.child_by_field_name("name")), decl, "function")
            elif decl.type in ("class_declaration", "abstract_class_declaration"):
                add(_ts_text(decl.child_by_field_name("name")), decl, "lexical")
    return out


def _js_bound_names(root_node) -> set[str]:
    """Every identifier this file itself binds, at any depth.

    A reference to a name in this set is resolved locally, however deeply
    nested; the free-identifier check only ever looks at the rest."""
    bound: set[str] = set()

    def visit(node) -> None:
        k = node.type
        if k == "import_statement":
            clause = next((c for c in node.named_children
                           if c.type == "import_clause"), None)
            for child in (clause.children if clause is not None else []):
                if child.type == "identifier":
                    bound.add(_ts_text(child))
                elif child.type == "named_imports":
                    for spec in child.named_children:
                        if spec.type != "import_specifier":
                            continue
                        alias = _ts_text(spec.child_by_field_name("alias"))
                        bound.add(alias or _ts_text(
                            spec.child_by_field_name("name")))
                elif child.type == "namespace_import":
                    ident = next((x for x in child.named_children
                                  if x.type == "identifier"), None)
                    bound.add(_ts_text(ident))
            return
        if k in ("formal_parameters", "catch_clause"):
            for param in node.named_children:
                bound.update(_js_pattern_names(param))
            return
        if k == "variable_declarator":
            bound.update(_js_pattern_names(node.child_by_field_name("name")))
        elif k in _JS_DECL_NODES:
            bound.add(_ts_text(node.child_by_field_name("name")))
        elif k == "assignment_expression":
            left = node.child_by_field_name("left")
            if left is not None and left.type == "identifier":
                bound.add(_ts_text(left))
        elif k in ("member_expression", "subscript_expression"):
            # `window.NAME = ...` / `globalThis.NAME = ...` create a global
            # that later lines of this very file may call.
            obj = _ts_text(node.child_by_field_name("object"))
            if obj in ("window", "globalThis", "self", "global"):
                prop = node.child_by_field_name("property")
                if prop is not None and prop.type == "property_identifier":
                    bound.add(_ts_text(prop))
                elif (prop is not None and prop.type == "string"
                      and prop.text):
                    raw = prop.text.decode("utf-8", "replace").strip("'\"")
                    if re.fullmatch(r"[A-Za-z_$][\w$]*", raw):
                        bound.add(raw)
        elif k == "labeled_statement":
            label = node.child_by_field_name("label")
            bound.add(_ts_text(label))
        for child in node.named_children:
            visit(child)

    visit(root_node)
    bound.discard("")
    return bound


def _js_called_names(root_node) -> list[tuple[str, int]]:
    """(base identifier, line) for every call the file makes.

    `fn(...)`, `new fn(...)` and `fn.member(...)` are the forms a missing
    binding turns into a ReferenceError or a "cannot read property of
    undefined"; a member chain is judged on its leftmost name only."""
    out: list[tuple[str, int]] = []
    for node in _walk_ts(root_node):
        if node.type == "call_expression":
            callee = node.child_by_field_name("function")
        elif node.type == "new_expression":
            callee = node.child_by_field_name("constructor")
        else:
            continue
        if callee is None or callee.type not in ("identifier",
                                                 "member_expression"):
            continue
        while callee.type == "member_expression":
            inner = callee.child_by_field_name("object")
            if inner is None:
                break
            callee = inner
        if callee.type != "identifier":
            continue
        name = _ts_text(callee)
        if not name or name in _JS_NON_NAMES:
            continue
        ancestor = node.parent
        for _ in range(3):                  # `@Injectable()` is not a reference
            if ancestor is None:
                break
            if "decorator" in ancestor.type:
                name = ""
                break
            ancestor = ancestor.parent
        if not name:
            continue
        out.append((name, node.start_point[0] + 1))
    return out


def _js_sibling_spec(from_rel: str, target_rel: str) -> str:
    """The ES specifier `from_rel` must write to reach `target_rel`."""
    import posixpath
    spec = posixpath.relpath(target_rel, posixpath.dirname(from_rel) or ".")
    return spec if spec.startswith((".", "/")) else "./" + spec


def _js_is_host_name(name: str) -> bool:
    if (name in _JS_GLOBAL_NAMES or name in _JS_LIBRARY_GLOBALS
            or name in _JS_TEST_GLOBALS):
        return True
    if name in _JS_NON_NAMES:
        return True
    return any(name.startswith(p) for p in _JS_GLOBAL_PREFIXES)


def _js_page_externals(root: Path) -> bool:
    """True when some HTML page loads a script from outside the project.

    A CDN bundle (three.js, gsap, phaser) puts names in the global scope that
    no file in the tree defines, so the "nothing defines this name" check has
    no business guessing about it; it is skipped in that project."""
    for rel in _iter_code_files(root, _FE_HTML_EXTS):
        text = _read(rel, root)
        for m in _FE_SCRIPT_SRC_RE.finditer(text):
            if _fe_external(m.group(1)):
                return True
    return False


@_ts_guarded
def _js_module_link_errors(root: Path, cap: int = 40
                           ) -> list[tuple[str, str, str]]:
    """(rel, peer, ContractError) for ES module wiring a parse cannot see.

    Two deterministic failures, both silent to `has_syntax_error` and both
    fatal to the whole module graph in the browser:

      - a name bound twice at module level (the second `import { WAYPOINTS }`
        an incremental edit appends): `Identifier 'WAYPOINTS' has already been
        declared`, the file never evaluates;
      - a call whose callee is bound nowhere -- not in this file, not imported,
        not a host global. When another module exports the name the fix is the
        missing import; when it only declares it, the name is unreachable from
        this scope; when nothing in the project defines it at all, the call is
        a ReferenceError the moment the line runs.

    Names an import statement already binds are left to
    `_js_cross_file_errors`, which owns the import-vs-export contract, and
    anything the project might be loading from a CDN stays unreported."""
    root = Path(root)
    rels = _iter_code_files(root, TS_EXTS, cap=300)
    if not rels:
        return []
    parsed: dict[str, object] = {}
    info_cache: dict[str, dict | None] = {}

    def text_for(rel: str) -> str:
        return _read(rel, root)

    def tree_for(rel: str):
        if rel not in parsed:
            suffix = "." + rel.rsplit(".", 1)[-1].lower()
            try:
                parsed[rel] = _ts_parse(text_for(rel), suffix)
            except Exception:
                parsed[rel] = None
        return parsed[rel]

    def info_for(rel: str) -> dict | None:
        if rel not in info_cache:
            suffix = "." + rel.rsplit(".", 1)[-1].lower()
            try:
                info_cache[rel] = _js_module_info(text_for(rel), suffix)
            except Exception:
                info_cache[rel] = None
        return info_cache[rel]

    # --- the project-wide pools the free-identifier check reasons over -----
    exported_by: dict[str, list[str]] = {}     # name -> module files exporting it
    classic_top: set[str] = set()              # names a classic script exposes
    declared_in: dict[str, list[str]] = {}     # name -> module files hiding it
    defined_anywhere: set[str] = set()         # any project file, any depth
    html_defined: set[str] = set()
    for rel in rels:
        text = text_for(rel)
        if not text.strip():
            continue
        for m in _JS_ANY_DEF_RE.finditer(text):
            defined_anywhere.add(next(g for g in m.groups() if g))
        info = info_for(rel)
        if info is None:
            continue
        if not info["module"]:
            # a classic script's top level really is the page's global scope
            classic_top.update(info["locals"])
            continue
        for name in info["exports"]:
            if name != "default":
                exported_by.setdefault(name, []).append(rel)
        for name in info["locals"]:
            if name not in info["exports"]:
                declared_in.setdefault(name, []).append(rel)
    for rel in _iter_code_files(root, _FE_HTML_EXTS):
        text = text_for(rel)
        for m in _JS_ANY_DEF_RE.finditer(text):
            html_defined.add(next(g for g in m.groups() if g))
    defined_anywhere |= html_defined
    externals = _js_page_externals(root)

    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    def emit(rel: str, peer: str, text: str) -> None:
        if text in seen or len(out) >= cap:
            return
        seen.add(text)
        out.append((rel, peer, text))

    for rel in rels:
        node = tree_for(rel)
        if node is None:
            continue                       # the syntax gate owns broken files
        info = info_for(rel)
        # 1. a module-level name bound twice: one SyntaxError, whole file dead.
        # An ES module only lets `var` repeat; a classic script also allows
        # repeated functions and var/function mixes (last one wins).
        esm = bool(info and info["module"] and not info["opaque"])
        repeatable = ("hoisted",) if esm else ("hoisted", "function")
        first: dict[str, tuple[int, str]] = {}
        for name, line, kind in _js_top_level_bindings(node):
            prev = first.get(name)
            if prev is None:
                first[name] = (line, kind)
                continue
            if kind in repeatable and prev[1] in repeatable:
                continue                   # two `var f`s are one binding
            if prev[1] == "import":
                where = f"imported at line {prev[0]}"
            elif prev[1] == "lexical":
                where = f"declared at line {prev[0]}"
            else:
                where = f"defined at line {prev[0]}"
            if kind == "import":
                this = "imported again here"
            elif kind == "lexical":
                this = "declared again here"
            else:
                this = "defined again here"
            emit(rel, "", f"ContractError in {rel}:{line}: '{name}' is "
                 f"{this}, and it is already {where} -- the browser refuses "
                 f"the file with `SyntaxError: Identifier '{name}' has "
                 f"already been declared`, so none of its exports exist and "
                 f"every module that imports it fails to load; keep exactly "
                 f"one binding for '{name}'")
            first[name] = (line, kind)     # report each name once

        # 2. calls to a name this module never binds
        if info is None:
            continue
        bound = _js_bound_names(node) | set(info["locals"])
        told: set[str] = set()
        for name, line in _js_called_names(node):
            if name in bound or name in told:
                continue
            if _js_is_host_name(name):
                continue
            holders = exported_by.get(name)
            if holders:
                told.add(name)
                holder = holders[0]
                where = ", ".join(sorted(set(holders))[:3])
                hint = ("add `import { " + name + " } from '"
                        + _js_sibling_spec(rel, holder) + "'`")
                emit(rel, holder, f"ContractError in {rel}:{line}: calls "
                     f"{name}() but this module never imports it -- {where} "
                     f"exports {name}, and an ES module only sees names it "
                     f"declares or imports, so the call is a ReferenceError "
                     f"the moment this line runs (" + hint + ")")
                continue
            hidden = declared_in.get(name)
            if hidden and name not in classic_top:
                told.add(name)
                emit(rel, hidden[0], f"ContractError in {rel}:{line}: calls "
                     f"{name}(), which {hidden[0]} declares at its top level "
                     f"but never exports -- every module has its own scope, "
                     f"so this file cannot reach it (add `export` before it "
                     f"in {hidden[0]} and import it here)")
                continue
            if name in defined_anywhere or externals:
                continue  # a classic script's global, or a CDN bundle: not provable
            told.add(name)
            emit(rel, "", f"ContractError in {rel}:{line}: calls {name}() but "
                 f"nothing in the project defines it and no file in the "
                 f"module graph exports it -- the ReferenceError stops this "
                 f"module the first time the line runs (define it here, "
                 f"import it from the file that should own it, or remove the "
                 f"call)")
    out.sort(key=lambda t: (t[0], t[2]))
    return out


# --- runtime-fatal shapes a parse accepts (BUG-210 / BUG-211) --------------
# `const x = 1; x = 2;` and `engine.setBpm(...)` on a class that never defines
# setBpm both parse clean, pass every import/DOM check, and throw a TypeError
# the first time the line runs. Like the rest of the gate, these only report
# what the project's own sources prove; every ambiguous shape stays quiet.

_JS_SCOPE_NODES = {
    "program", "statement_block", "static_block",
    "function_declaration", "generator_function_declaration",
    "function_expression", "generator_function", "arrow_function",
    "function", "method_definition",
    "for_statement", "for_in_statement", "catch_clause",
}


def _js_scope_of(node):
    """The nearest ancestor (or the node itself) that introduces a scope."""
    while node is not None and node.type not in _JS_SCOPE_NODES:
        node = node.parent
    return node


def _js_scope_bindings(scope) -> dict[str, tuple[str, int]]:
    """name -> (kind, line) for the bindings DIRECTLY inside one scope node
    (const/let/var/param/function/class/import/catch). Nested scopes are not
    descended into: their bindings belong to them, not to this scope."""
    out: dict[str, tuple[str, int]] = {}

    def add(names, kind, node) -> None:
        for name in names:
            if name and name not in out:
                out[name] = (kind, node.start_point[0] + 1)

    def visit(node) -> None:
        k = node.type
        # Node wrappers are not identity-stable: compare the underlying id.
        if k in _JS_SCOPE_NODES and node.id != scope.id:
            return                       # a nested scope owns its bindings
        if k == "lexical_declaration":
            kind = ("const" if re.match(r"^\s*const\b", _ts_text(node))
                    else "let")
            for child in node.named_children:
                if child.type == "variable_declarator":
                    add(_js_pattern_names(child.child_by_field_name("name")),
                        kind, child)
        elif k == "variable_declaration":
            for child in node.named_children:
                if child.type == "variable_declarator":
                    add(_js_pattern_names(child.child_by_field_name("name")),
                        "var", child)
        elif k in ("function_declaration", "generator_function_declaration"):
            add({_ts_text(node.child_by_field_name("name"))}, "function", node)
        elif k in ("class_declaration", "abstract_class_declaration"):
            add({_ts_text(node.child_by_field_name("name"))}, "class", node)
        elif k == "import_statement":
            clause = next((c for c in node.named_children
                           if c.type == "import_clause"), None)
            for child in (clause.children if clause is not None else []):
                if child.type == "identifier":
                    add({_ts_text(child)}, "import", child)
                elif child.type == "named_imports":
                    for spec in child.named_children:
                        if spec.type != "import_specifier":
                            continue
                        alias = _ts_text(spec.child_by_field_name("alias"))
                        add({alias or _ts_text(
                            spec.child_by_field_name("name"))}, "import", spec)
                elif child.type == "namespace_import":
                    ident = next((x for x in child.named_children
                                  if x.type == "identifier"), None)
                    add({_ts_text(ident)}, "import", node)
        elif k == "formal_parameters":
            for param in node.named_children:
                add(_js_pattern_names(param), "param", param)
        elif k == "arrow_function":
            param = node.child_by_field_name("parameter")
            if param is not None and param.type != "formal_parameters":
                add(_js_pattern_names(param), "param", param)
        elif k == "catch_clause":
            param = node.child_by_field_name("parameter")
            if param is not None:
                add(_js_pattern_names(param), "catch", param)
        elif k == "for_in_statement":
            # `for (const y of ys)`: some grammar versions keep the keyword
            # as an unnamed child and the left a bare identifier.
            left = node.child_by_field_name("left")
            kw = next((c.type for c in node.children
                       if c.type in ("const", "let", "var")), "")
            if left is not None and left.type == "identifier" and kw:
                add({_ts_text(left)},
                    "const" if kw == "const" else
                    "let" if kw == "let" else "var", left)
        for child in node.named_children:
            visit(child)

    if scope is not None:
        visit(scope)
    return out


def _js_assignment_targets(root_node):
    """(name, node) for every write to a bare identifier: `x = ...`,
    `x += ...`, `x++`, `for (x of ...)`. Member writes (`obj.x = ...`) are a
    different contract and stay out."""
    for node in _walk_ts(root_node):
        k = node.type
        if k in ("assignment_expression", "augmented_assignment_expression"):
            left = node.child_by_field_name("left")
            if left is None:
                continue
            if left.type == "identifier":
                yield _ts_text(left), node
            elif left.type in ("object_pattern", "array_pattern"):
                for name in sorted(_js_pattern_names(left)):
                    yield name, node
        elif k == "update_expression":
            arg = node.child_by_field_name("argument")
            if arg is not None and arg.type == "identifier":
                yield _ts_text(arg), node
        elif k == "for_in_statement":
            left = node.child_by_field_name("left")
            declared = any(c.type in ("const", "let", "var")
                           for c in node.children)
            if left is not None and left.type == "identifier" and not declared:
                yield _ts_text(left), node   # `for (y of xs)` writes y


@_ts_guarded
def _js_const_reassign_errors(root: Path, cap: int = 40
                              ) -> list[tuple[str, str, str]]:
    """(rel, '', ContractError) for writes to a const or imported binding.

    `const spaces = steps - pulses; ... spaces -= remainder;` parses clean
    and raises `TypeError: Assignment to constant variable` the first time
    the line runs (BUG-210). The NEAREST binding decides, so a shadowing
    let/param in an inner scope is never blamed on an outer const."""
    root = Path(root)
    rels = _iter_code_files(root, TS_EXTS, cap=300)
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for rel in rels:
        if len(out) >= cap:
            break
        text = _read(rel, root)
        if not text.strip():
            continue
        suffix = "." + rel.rsplit(".", 1)[-1].lower()
        node = _ts_parse(text, suffix)
        if node is None:
            continue                     # the syntax gate owns broken files
        scopes: dict[int, dict] = {}

        def bindings(scope) -> dict:
            key = scope.id
            if key not in scopes:
                scopes[key] = _js_scope_bindings(scope)
            return scopes[key]

        for name, anode in _js_assignment_targets(node):
            if not name or name in _JS_NON_NAMES:
                continue
            # A write inside the declarator's own initializer is a TDZ
            # question, not a reassignment of an established binding.
            p = anode.parent
            own_init = False
            while p is not None and p.type not in _JS_SCOPE_NODES:
                if p.type == "variable_declarator" and name in \
                        _js_pattern_names(p.child_by_field_name("name")):
                    own_init = True
                    break
                p = p.parent
            if own_init:
                continue
            scope = _js_scope_of(anode)
            found = None
            while scope is not None:
                hit = bindings(scope).get(name)
                if hit is not None:
                    found = hit
                    break
                scope = _js_scope_of(scope.parent)
            if found is None or found[0] not in ("const", "import"):
                continue
            kind, dline = found
            line = anode.start_point[0] + 1
            what = ("an imported binding" if kind == "import"
                    else "a const binding")
            fix = ("import a mutable holder or restructure the code"
                   if kind == "import" else
                   f"declare '{name}' with let if its value must change")
            err = (f"ContractError in {rel}:{line}: assigns to '{name}', "
                   f"which is {what} (declared at line {dline}) -- JavaScript "
                   f"raises `TypeError: Assignment to constant variable` the "
                   f"moment this line runs; {fix}")
            if err in seen:
                continue
            seen.add(err)
            out.append((rel, "", err))
    out.sort(key=lambda t: (t[0], t[2]))
    return out


# Every object answers these, whatever its class defines.
_JS_OBJECT_PROTO = {
    "constructor", "hasOwnProperty", "isPrototypeOf", "propertyIsEnumerable",
    "toLocaleString", "toString", "valueOf", "__defineGetter__",
    "__defineSetter__", "__lookupGetter__", "__lookupSetter__", "__proto__",
}


def _js_class_parent(class_node) -> str:
    """The parent class name in `class X extends Y`, or '' (grammar-version
    proof: read the header text up to the body, not a field name)."""
    body = class_node.child_by_field_name("body")
    header = _ts_text(class_node)
    if body is not None:
        header = header[:body.start_byte - class_node.start_byte]
    m = re.search(r"\bextends\s+([A-Za-z_$][\w$]*)", header)
    return m.group(1) if m else ""


def _js_class_surface(class_node) -> tuple[set[str], bool]:
    """(instance member names, opaque) for one class node. Opaque when the
    surface cannot be proven: a computed member name, an index signature,
    a decorator, or a body the parser could not read."""
    methods: set[str] = set()
    body = class_node.child_by_field_name("body")
    if body is None or body.has_error:
        return methods, True
    opaque = False
    for child in body.named_children:
        k = child.type
        if k in ("method_definition", "method_signature",
                 "abstract_method_signature", "public_field_definition",
                 "field_definition"):
            if any(c.type == "decorator" for c in child.named_children):
                opaque = True
            name = child.child_by_field_name("name")
            if name is None:
                # some grammar versions expose the member name as the first
                # named child instead of a `name` field (class fields)
                name = next((x for x in child.named_children
                             if x.type in ("property_identifier",
                                           "private_property_identifier",
                                           "computed_property_name",
                                           "string", "number")), None)
            if name is None:
                continue
            if name.type in ("computed_property_name", "string",
                             "number"):
                if name.type == "computed_property_name":
                    opaque = True        # [expr]() could be anything
                continue
            methods.add(_ts_text(name))
        elif k == "index_signature":
            opaque = True
        elif k in ("comment", "static_block"):
            continue                     # static blocks shape the class, not
                                         # the instance surface
        else:
            opaque = True                # unexpected member: never guess
    return methods, opaque


@_ts_guarded
def _js_instance_method_errors(root: Path, cap: int = 40
                               ) -> list[tuple[str, str, str]]:
    """(rel, def_rel, ContractError) for `x.method()` where x is a project
    class instance and the class defines no such method (BUG-211).

    Only proven shapes are reported: `const x = new Class()` at module level
    (or `this.x = new Class()` inside the class itself), where Class is
    defined in the project, has a fully readable member surface, and x is
    never reassigned or shadowed between the binding and the call. Anything
    ambiguous -- an unknown parent, a computed member, a second binding of
    the name -- stays quiet."""
    root = Path(root)
    rels = _iter_code_files(root, TS_EXTS, cap=300)
    if not rels:
        return []
    parsed: dict[str, object] = {}

    def tree_for(rel: str):
        if rel not in parsed:
            suffix = "." + rel.rsplit(".", 1)[-1].lower()
            try:
                parsed[rel] = _ts_parse(_read(rel, root), suffix)
            except Exception:
                parsed[rel] = None
        return parsed[rel]

    # --- 1. every project class: name -> [(rel, methods, opaque, parent)] --
    classes: dict[str, list] = {}
    default_class: dict[str, tuple] = {}   # rel -> entry of `export default class`

    def register(name: str, entry: tuple, rel: str, node) -> None:
        classes.setdefault(name, []).append(entry)
        parent = node.parent
        if parent is not None and parent.type == "export_statement" \
                and re.match(r"^\s*export\s+default\b", _ts_text(parent)):
            default_class[rel] = entry

    for rel in rels:
        node = tree_for(rel)
        if node is None:
            continue
        for c in _walk_ts(node):
            if c.type in ("class_declaration", "abstract_class_declaration"):
                name = _ts_text(c.child_by_field_name("name"))
                if not name:
                    continue
                methods, opaque = _js_class_surface(c)
                register(name, (rel, methods, opaque, _js_class_parent(c)),
                         rel, c)
            elif c.type == "variable_declarator":
                value = c.child_by_field_name("value")
                name_node = c.child_by_field_name("name")
                if value is None or value.type != "class" \
                        or name_node is None \
                        or name_node.type != "identifier":
                    continue
                methods, opaque = _js_class_surface(value)
                register(_ts_text(name_node),
                         (rel, methods, opaque, _js_class_parent(value)),
                         rel, c)

    surface_cache: dict[int, set | None] = {}

    def surface(entry: tuple, stack: tuple = ()) -> set | None:
        """The full instance surface incl. project parents, or None when it
        cannot be proven (unknown/ambiguous parent, opaque member)."""
        key = id(entry)
        if key in surface_cache:
            return surface_cache[key]
        rel_c, methods, opaque, parent = entry
        result: set | None = None
        if not opaque:
            if not parent:
                result = set(methods)
            elif parent not in stack:
                cands = [e for e in classes.get(parent, [])
                         if e[0] == rel_c] or classes.get(parent, [])
                if len(cands) == 1:
                    ps = surface(cands[0], stack + (parent,))
                    if ps is not None:
                        result = set(methods) | ps
        surface_cache[key] = result
        return result

    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    def emit(rel: str, peer: str, text: str) -> None:
        if text in seen or len(out) >= cap:
            return
        seen.add(text)
        out.append((rel, peer, text))

    # --- 2. per file: instances, then the calls on them --------------------
    for rel in rels:
        if len(out) >= cap:
            break
        node = tree_for(rel)
        if node is None:
            continue
        imports, _reexports = _js_imports_and_reexports(node)
        imported_from: dict[str, tuple[str, str]] = {}
        for item in imports:
            target = _js_resolve_import(root, rel, item["spec"])
            if not target:
                continue
            for local, imported in item["named"]:
                imported_from[local] = (target, imported)
            if item["default"]:
                imported_from[item["default"]] = (target, "default")

        def lookup(cname: str) -> tuple | None:
            """The one class entry `new cname()` provably constructs here."""
            local = [e for e in classes.get(cname, []) if e[0] == rel]
            if len(local) == 1:
                return local[0]
            if local or cname not in imported_from:
                return None              # ambiguous or not a project class
            target, imported = imported_from[cname]
            if imported == "default":
                return default_class.get(target)
            hits = [e for e in classes.get(imported, []) if e[0] == target]
            return hits[0] if len(hits) == 1 else None

        # module-level instances: name -> (entry, declarator line)
        instances: dict[str, tuple] = {}
        ambiguous: set[str] = set()
        for c in node.named_children:
            decls = []
            if c.type in ("lexical_declaration", "variable_declaration"):
                decls = [d for d in c.named_children
                         if d.type == "variable_declarator"]
            elif c.type == "export_statement":
                d = c.child_by_field_name("declaration")
                if d is not None and d.type in ("lexical_declaration",
                                                "variable_declaration"):
                    decls = [x for x in d.named_children
                             if x.type == "variable_declarator"]
            for d in decls:
                name_node = d.child_by_field_name("name")
                value = d.child_by_field_name("value")
                if name_node is None or name_node.type != "identifier":
                    continue
                vname = _ts_text(name_node)
                if value is None or value.type != "new_expression":
                    continue
                ctor = value.child_by_field_name("constructor")
                if ctor is None or ctor.type != "identifier":
                    continue
                entry = lookup(_ts_text(ctor))
                if entry is None:
                    continue
                if vname in instances:
                    ambiguous.add(vname)
                instances[vname] = (entry, d.start_point[0] + 1)
        # a name re-assigned anywhere is not a stable instance
        for name, _anode in _js_assignment_targets(node):
            if name in instances:
                ambiguous.add(name)

        scopes: dict[int, dict] = {}

        def bindings(scope) -> dict:
            key = scope.id
            if key not in scopes:
                scopes[key] = _js_scope_bindings(scope)
            return scopes[key]

        def nearest(name: str, from_node):
            scope = _js_scope_of(from_node)
            while scope is not None:
                hit = bindings(scope).get(name)
                if hit is not None:
                    return hit, scope
                scope = _js_scope_of(scope.parent)
            return None, None

        # this.x instances, per class body (only when the class has no
        # parent: a parent constructor could assign this.x too)
        this_inst: dict[tuple[int, str], tuple] = {}
        this_bad: set[tuple[int, str]] = set()
        for c in _walk_ts(node):
            if c.type not in ("class_declaration",
                              "abstract_class_declaration", "class"):
                continue
            if _js_class_parent(c):
                continue
            for n in _walk_ts(c):
                if n.type not in ("assignment_expression",
                                  "augmented_assignment_expression"):
                    continue
                left = n.child_by_field_name("left")
                if left is None or left.type != "member_expression":
                    continue
                obj = left.child_by_field_name("object")
                prop = left.child_by_field_name("property")
                if obj is None or obj.type != "this" or prop is None \
                        or prop.type != "property_identifier":
                    continue
                key = (c.id, _ts_text(prop))
                value = n.child_by_field_name("right")
                entry = None
                if value is not None and value.type == "new_expression":
                    ctor = value.child_by_field_name("constructor")
                    if ctor is not None and ctor.type == "identifier":
                        entry = lookup(_ts_text(ctor))
                if entry is None:
                    this_bad.add(key)    # assigned something unprovable
                elif key in this_inst and this_inst[key][0] is not entry:
                    this_bad.add(key)    # two different classes
                else:
                    this_inst[key] = (entry, n.start_point[0] + 1)

        def class_of(from_node):
            p = from_node.parent
            while p is not None:
                if p.type in ("class_declaration",
                              "abstract_class_declaration", "class"):
                    return p
                p = p.parent
            return None

        for n in _walk_ts(node):
            if n.type != "call_expression":
                continue
            fn = n.child_by_field_name("function")
            if fn is None or fn.type != "member_expression":
                continue
            obj = fn.child_by_field_name("object")
            prop = fn.child_by_field_name("property")
            if obj is None or prop is None \
                    or prop.type != "property_identifier":
                continue
            method = _ts_text(prop)
            if not method or method in _JS_OBJECT_PROTO:
                continue
            entry = None
            receiver = ""
            if obj.type == "identifier":
                x = _ts_text(obj)
                if x in ambiguous or x not in instances:
                    continue
                hit, scope = nearest(x, n)
                # only the module-level binding itself proves the receiver
                if hit is None or scope.id != node.id \
                        or hit[1] != instances[x][1]:
                    continue
                entry, _line = instances[x]
                receiver = x
            elif obj.type == "member_expression":
                o2 = obj.child_by_field_name("object")
                p2 = obj.child_by_field_name("property")
                if o2 is None or o2.type != "this" or p2 is None \
                        or p2.type != "property_identifier":
                    continue
                cls = class_of(n)
                if cls is None:
                    continue
                key = (cls.id, _ts_text(p2))
                if key in this_bad or key not in this_inst:
                    continue
                entry, _line = this_inst[key]
                receiver = "this." + _ts_text(p2)
            if entry is None:
                continue
            methods = surface(entry)
            if methods is None or method in methods:
                continue
            def_rel = entry[0]
            cname = next((nm for nm, es in classes.items()
                          if entry in es), "the class")
            shown = ", ".join(sorted(methods)[:8]) or "(none)"
            line = n.start_point[0] + 1
            emit(rel, def_rel,
                 f"ContractError in {rel}:{line}: calls {receiver}.{method}"
                 f"(...), but class {cname} ({def_rel}) defines no such "
                 f"method (it defines: {shown}) -- `TypeError: "
                 f"{receiver}.{method} is not a function` the first time "
                 f"this line runs; add {method} to {cname} or call a method "
                 f"it defines")
    out.sort(key=lambda t: (t[0], t[2]))
    return out


def verify_cross_file_contracts(root: Path,
                                modified_files: list | None = None) -> list[str]:
    """Cross-file import/arity violations in the project's Python and JS/TS
    files, as `ContractError in <file>:<line>: <detail>` strings (empty =
    consistent).

    Every local ``from <module> import <name>`` is checked against what
    <module> actually defines, and every direct call of such an imported name
    is checked for argument-count compatibility. The same gate covers
    relative ES module imports and exported-name/arity contracts in JS/TS.
    `modified_files` only reorders diagnostics (files touched by the last step
    are reported first); the scan always covers the whole project, since a
    change in one file breaks contracts in its *importers*."""
    root = Path(root)
    if not root.is_dir():
        return []
    mods = _local_py_modules(root)
    touched: set[str] = set()
    for m in modified_files or ():
        name = fileio.normalise_rel(str(m))
        if name.startswith("./"):
            name = name[2:]
        try:
            p = Path(str(m))
            if p.is_absolute():
                name = p.relative_to(root).as_posix()
        except (OSError, ValueError):
            pass
        touched.add(name)

    out: list[str] = []
    seen: set[str] = set()

    def emit(text: str, rel: str) -> None:
        if text in seen:
            return
        seen.add(text)
        out.append((0 if rel in touched else 1, len(out), text))

    for _modname, info in sorted(mods.items()):
        rel = info.rel
        src = _read(rel, root)
        if not src.strip():
            continue
        try:
            tree = ast.parse(src)
        except (SyntaxError, ValueError):
            continue
        local_imports: dict[str, tuple[str, ast.AST | None]] = {}
        pkg = _file_package(rel, mods)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue                  # plain `import pkg.mod`: not guessed
            names = [a for a in node.names if a.name != "*"]
            if not names:
                continue
            base = node.module or ""
            prefix = pkg
            if node.level:
                if not pkg:
                    continue      # relative import outside a known package
                depth = len(pkg.split(".")) - (node.level - 1)
                if depth < 1:
                    continue      # escapes the project's package tree
                prefix = ".".join(pkg.split(".")[:depth])
                target = f"{prefix}.{base}" if base else prefix
                # `from . import x` lands here too: target is the package,
                # and each name is checked as one of its submodules/defs.
            else:
                target = base
            if target not in mods:
                if not node.level:
                    continue      # stdlib or third-party: never guessed
                emit(f"ContractError in {rel}:{node.lineno}: imports "
                     f"'{'.' * node.level}{base}' from package '{prefix}', "
                     f"but that module does not exist in the project", rel)
                continue
            minfo = mods[target]
            children = _package_children(mods, target)
            for a in names:
                alias = a.asname or a.name
                if a.name in minfo.exports or a.name in children:
                    local_imports[alias] = (minfo.rel,
                                            minfo.defs.get(a.name))
                    continue
                defines = _local_defs_names(minfo)
                emit(f"ContractError in {rel}:{node.lineno}: imports "
                     f"'{a.name}' from {minfo.rel}, but {minfo.rel} only "
                     f"defines {defines or '[]'}"
                     + (f" (submodules: {sorted(children)[:4]})"
                        if children and not defines else ""), rel)

        if not local_imports:
            continue
        # Shadowed names are trusted: a same-file def or assignment wins over
        # the import, and telling them apart from a real drift is guesswork.
        shadowed: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                shadowed.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) \
                    else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        shadowed.add(t.id)
        from .apicheck import _arity_error
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)):
                continue
            name = node.func.id
            if name in shadowed or name not in local_imports:
                continue
            src_rel, defnode = local_imports[name]
            if defnode is None:
                continue
            got = _arity_sig(defnode)
            if got is None:
                continue
            sig, _names = got
            msg = _arity_error(sig, node)
            if msg:
                if isinstance(defnode, ast.ClassDef):
                    kind = "class"
                    shown = _signature(next(
                        (n for n in defnode.body if isinstance(n, DEF_NODES)
                         and n.name == "__init__"), defnode))
                else:
                    kind = "function"
                    shown = _signature(defnode)
                emit(f"ContractError in {rel}:{node.lineno}: calls {name}(...)"
                     f" ({msg}) imported from {src_rel}, where the {kind} is "
                     f"defined as {shown or name}", rel)

    # JS/TS ES module contracts: relative import existence, exported names,
    # and imported-function arity. Same diagnostic shape, same touched-first
    # ordering, same conservative skips.
    for importer, _target, text in _js_cross_file_errors(root):
        emit(text, importer)

    # ... and the wiring a parse cannot see: a name bound twice at module
    # level, a call whose exporter was never imported.
    for importer, _target, text in _js_module_link_errors(root):
        emit(text, importer)

    # Test files calling into (changed) source through dotted
    # imports or bound instances -- the two spellings the from-import pass
    # above never sees. Surfaced here, BEFORE the loop spawns pytest.
    for rel, text in _incoming_test_caller_errors(root, mods):
        emit(text, rel)

    out.sort(key=lambda t: (t[0], t[1]))
    return [t for _o, _i, t in out][:20]


def _requires_text(sig) -> str:
    """`requires ...` clause for an arity message: exact count, a range when
    defaults pad the tail, or an open minimum when the def takes *args."""
    lo = len(sig.required)
    hi = sig.max_pos
    if hi is None:
        return f"at least {lo} positional argument(s)" if lo else \
            "any number of arguments"
    if hi == lo:
        return f"{lo} positional argument(s)"
    return f"{lo}-{hi} positional argument(s)"


def _test_arity_error(call: ast.Call, node: ast.AST, drop_self: bool):
    """(shown_count, requires_text) when `call` violates the arity of
    definition `node`, else None. apicheck's `_arity_error` decides whether
    the call is impossible; the counts come from the signature."""
    from .apicheck import _arity_error, _sig
    if isinstance(node, DEF_NODES):
        if getattr(node, "decorator_list", ()):
            return None          # staticmethod/classmethod: binding unknown
        sig = _sig(node.args, drop_self)
    elif isinstance(node, ast.ClassDef):
        got = _arity_sig(node)
        if got is None:
            return None
        sig = got[0]
    else:
        return None
    msg = _arity_error(sig, call)
    if not msg:
        return None
    n = len(call.args) + len(call.keywords)
    return n, _requires_text(sig)


def _incoming_test_caller_errors(root: Path,
                                 mods: dict[str, _Mod]) -> list[tuple[str, str]]:
    """Arity violations at TEST-file call sites of project source: dotted
    `import mod` + `mod.fn(...)`, `from pkg import submodule` + `sub.fn(...)`,
    and instance-method calls on variables bound by constructor (`store =
    Store()`) or annotation (`store: Store`).

    The existing from-import pass already sees bare `from x import y; y(...)`;
    these spellings -- how tests really call into changed code -- were
    invisible until pytest ran. Now they surface as the same
    `ContractError in file:line:` shape before the subprocess spawns.
    Conservative like everything here: unresolved or shadowed receivers,
    decorated defs, inherited methods and star-args calls are skipped."""
    out: list[tuple[str, str]] = []
    defs_by_rel = {info.rel: info.defs for info in mods.values()}
    test_rel_set = {info.rel for info in mods.values() if _is_test_file(info.rel)}

    def emit(rel: str, line: int, shown: str, src_file: str,
             n_args: int, req: str) -> None:
        if len(out) >= 40:
            return
        text = (f"ContractError in {rel}:{line}: calls '{shown}' "
                f"with {n_args} arg(s), but updated signature in "
                f"{src_file} requires {req}")
        if any(t == text for _r, t in out):
            return
        out.append((rel, text))

    from .chunker import _local_types

    def assigned_names(fn: ast.AST) -> set[str]:
        """Every name a function binds anywhere in its body: Python treats
        all of them as function-locals, so an outer binding of the same name
        is invisible under this scope (and must not qualify its calls)."""
        got = set()
        stack = list(ast.iter_child_nodes(fn))
        while stack:
            n = stack.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                got.add(n.name)      # a nested def shadows any outer binding
                continue             # its body is a separate scope
            if isinstance(n, ast.ClassDef):
                got.add(n.name)      # ditto for a class defined in here
                continue
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                got.add(n.id)
            elif isinstance(n, ast.arg):
                got.add(n.arg)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                got.add(n.name)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    got.add(a.asname or a.name.split(".")[0])
            stack.extend(ast.iter_child_nodes(n))
        return got

    for _modname, info in sorted(mods.items()):
        rel = info.rel
        if not _is_test_file(rel):
            continue
        src = _read(rel, root)
        if not src.strip():
            continue
        try:
            tree = ast.parse(src)
        except (SyntaxError, ValueError):
            continue
        pkg = _file_package(rel, mods)
        class_imports: dict[str, tuple[str, ast.AST]] = {}   # alias -> (rel, class)
        module_alias: dict[tuple[str, ...], str] = {}        # chain -> module rel
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    if not pkg:
                        continue
                    depth = len(pkg.split(".")) - (node.level - 1)
                    if depth < 1:
                        continue
                    prefix = ".".join(pkg.split(".")[:depth])
                    target = f"{prefix}.{base}" if base else prefix
                else:
                    target = base
                minfo = mods.get(target)
                if minfo is None:
                    continue
                children = _package_children(mods, target)
                for a in node.names:
                    alias = a.asname or a.name
                    defnode = minfo.defs.get(a.name)
                    if isinstance(defnode, ast.ClassDef) \
                            and minfo.rel not in test_rel_set:
                        class_imports[alias] = (minfo.rel, defnode)
                    elif a.name in children:
                        sub = mods.get(f"{target}.{a.name}")
                        if sub is not None and sub.rel not in test_rel_set:
                            module_alias[(alias,)] = sub.rel
            elif isinstance(node, ast.Import):
                for a in node.names:
                    minfo = mods.get(a.name)
                    if minfo is None or minfo.rel in test_rel_set:
                        continue
                    if a.asname:
                        module_alias[(a.asname,)] = minfo.rel
                    else:
                        module_alias[tuple(a.name.split("."))] = minfo.rel
        if not module_alias and not class_imports:
            continue

        # every call, with the receiver-type map of its innermost scope:
        # module bindings (s = Store()) plus the function's own parameters,
        # annotations and constructor assigns; names rebound locally drop out
        calls: list[tuple[ast.Call, dict[str, tuple[str, ast.AST]]]] = []

        def walk_scope(n: ast.AST, vmap: dict) -> None:
            stack = [(c, vmap) for c in ast.iter_child_nodes(n)]
            while stack:
                node, vm = stack.pop()
                if isinstance(node, DEF_NODES):
                    shadowed = assigned_names(node)   # incl. locally-defined
                    inner = {k: v for k, v in vm.items()
                             if k not in shadowed}    # classes/defs shadow
                    for var, cname in _local_types(node).items():
                        if cname in class_imports and cname not in shadowed:
                            inner[var] = class_imports[cname]
                    walk_scope(node, inner)
                    continue
                if isinstance(node, ast.ClassDef):
                    walk_scope(node, vm)
                    continue
                if isinstance(node, ast.Call):
                    calls.append((node, vm))
                stack.extend((c, vm) for c in ast.iter_child_nodes(node))

        base_map = {v: class_imports[c] for v, c in _local_types(tree).items()
                    if c in class_imports}
        walk_scope(tree, base_map)

        for node, vmap in sorted(calls, key=lambda t: t[0].lineno):
            f = node.func
            if not isinstance(f, ast.Attribute):
                continue          # bare from-import calls: existing pass
            chain: list[str] = []
            cur: ast.AST = f
            while isinstance(cur, ast.Attribute):
                chain.append(cur.attr)
                cur = cur.value
            if not isinstance(cur, ast.Name):
                continue
            chain.append(cur.id)
            chain.reverse()
            shown = ".".join(chain[-2:])
            matched_module = False
            if chain[0] in {k[0] for k in module_alias}:
                # longest module-prefix of the chain, e.g. `pkg.sub.fn(...)`:
                # only a direct module member call is checkable
                i = len(chain) - 1
                while i >= 1 and tuple(chain[:i]) not in module_alias:
                    i -= 1
                if i >= 1 and i == len(chain) - 1 \
                        and tuple(chain[:i]) in module_alias:
                    matched_module = True
                    src_rel2 = module_alias[tuple(chain[:i])]
                    defnode = defs_by_rel.get(src_rel2, {}).get(chain[-1])
                    if defnode is not None:
                        got = _test_arity_error(node, defnode, False)
                        if got:
                            emit(rel, node.lineno, shown, src_rel2, *got)
            if not matched_module and len(chain) == 2 and chain[0] in vmap:
                src_rel2, cnode = vmap[chain[0]]
                m = next((x for x in cnode.body if isinstance(x, DEF_NODES)
                          and x.name == chain[1]), None)
                if m is None:
                    continue      # inherited/dynamic: unknown
                got = _test_arity_error(node, m, True)
                if got:
                    emit(rel, node.lineno, shown, src_rel2, *got)
    return out


# ------------------------------------------------------------- refinement ---

REFINE_SCHEMA = {
    "type": "object",
    "properties": {
        "objective": {"type": "string", "maxLength": 700},
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "maxLength": 200},
                    "action": {"enum": ["create", "modify"]},
                    "responsibility": {"type": "string", "maxLength": 300},
                },
                "required": ["path", "action", "responsibility"],
                "additionalProperties": False,
            },
        },
        "contracts": {
            "type": "array",
            "items": {"type": "string", "maxLength": 240},
        },
        "requirements": {
            "type": "array",
            "items": {"type": "string", "maxLength": 400},
        },
        "edge_cases": {
            "type": "array",
            "items": {"type": "string", "maxLength": 400},
        },
        "verification": {"type": "string", "maxLength": 900},
    },
    "required": ["objective", "files", "contracts", "requirements",
                 "verification"],
    "additionalProperties": False,
}

REFINE_SYSTEM = """\
You are a spec architect for an autonomous coding agent. The agent sees ONLY
your output -- never this conversation -- and a small 9B-35B model will
implement your spec step by step in fresh contexts. Turn the user's brief
idea into one unambiguous, immediately actionable specification.

Rules:
- objective: 1-2 sentences stating the concrete end state.
- files: exact relative paths (forward slashes, no leading "./") with
  "create" or "modify" and the file's single responsibility. When the repo
  snapshot shows a matching file, MODIFY it -- never plan a duplicate.
- contracts: every public name later steps must use, as an exact signature
  line, e.g. `def export_csv(rows: list[Row], path: pathlib.Path) -> str` or
  `class TableParser(name: str)` or `function renderTable(rows, opts)`.
  Small models rename things mid-plan; one spelling, defined once, here.
  Reuse existing signatures from the repo snapshot verbatim.
- requirements: numbered, concrete, testable behavioral rules. Every number,
  format, flag name, and default the idea implies must appear explicitly.
- edge_cases: the empty/missing/wrong-type/limit situations a test will force.
- verification: the command(s) and the observable pass criteria.
- Never write code bodies; never add options the idea did not ask for. Keep
  the whole spec under 45 lines."""


def _clean(value, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _clean_list(values, limit: int) -> list[str]:
    return [s for s in (_clean(v, limit) for v in (values or [])) if s][:24]


def _norm_refined(raw: dict, prompt: str) -> dict:
    """A model reply squeezed into the architect's shape (paths normalised,
    strings flattened and length-capped)."""
    raw = raw if isinstance(raw, dict) else {}
    files = []
    seen: set[str] = set()
    for f in raw.get("files") or []:
        if not isinstance(f, dict):
            continue
        rel = _clean(f.get("path"), 200).lstrip("/")
        while rel.startswith("./"):
            rel = rel[2:]
        if not rel or rel in seen or ".." in rel.split("/"):
            continue
        seen.add(rel)
        files.append({"path": rel,
                      "action": ("modify" if f.get("action") == "modify"
                                 else "create"),
                      "responsibility": _clean(f.get("responsibility"), 300)})
    return {
        "objective": _clean(raw.get("objective"), 700),
        "files": files[:20],
        "contracts": _clean_list(raw.get("contracts"), 240)[:30],
        "requirements": _clean_list(raw.get("requirements"), 400),
        "edge_cases": _clean_list(raw.get("edge_cases"), 400),
        "verification": _clean(raw.get("verification"), 900),
    }


def _edge_case_fallback(data: dict) -> list[str]:
    """EDGE CASES for a spec nobody wrote them into: phrased for the stack the
    file list actually names.

    The one generic line this used to print -- "handle empty input, missing
    files and wrong types with a clear error" -- was written for Python and
    landed on browser games, where it asked for a traceback in a project with
    no tracebacks, and said nothing about the failure every web build really
    hits: a DOM node that is not there, a first frame before any input, a
    canvas that does not fit the window."""
    paths = [str(f.get("path") or "") if isinstance(f, dict) else str(f)
             for f in (data.get("files") or ())]
    exts = {p.rsplit(".", 1)[-1] for p in paths if "." in p}
    web = exts & {"html", "htm", "css", "js", "mjs", "cjs", "jsx", "ts", "tsx"}
    py = exts & {"py", "pyw"}
    if not paths:
        return ["Empty or missing input fails with a clear message, not a "
                "traceback."]
    out: list[str] = []
    if web:
        out += [
            "A DOM node or canvas context the code expects is missing or "
            "null (the element is not in the page, or the query runs before "
            "it is created): guard for it, never dereference it.",
            "First load, before any input: empty state, zero score, no "
            "selection, nothing chosen -- the page must render and the loop "
            "must run without a click having happened.",
            "A viewport smaller than the designed size, a window resize, and "
            "a devicePixelRatio other than 1: content stays reachable and "
            "drawn inside its canvas.",
        ]
    if py:
        out += [
            "Empty or missing input, an absent file and a wrong type all "
            "fail with a clear message, not a traceback.",
            "The state is reloaded after a crash or a restart: no orphaned "
            "file, no half-written record.",
        ]
    if not out:
        out.append("Empty or missing input fails with a clear message, not a "
                   "traceback.")
    return out[:4]


def _render_refined(data: dict, original: str, mode: str) -> str:
    """The one shared text shape for a refined spec (model or fallback)."""
    lines = ["OBJECTIVE",
             data["objective"] or f"Deliver: {original.strip()}",
             "",
             "FILES TO CREATE OR MODIFY"]
    if data["files"]:
        lines += [f"- [{f['action'].upper()}] {f['path']}"
                  + (f" -- {f['responsibility']}" if f["responsibility"] else "")
                  for f in data["files"]]
    else:
        lines.append("- (not stated -- inspect the repo and choose minimal "
                     "file targets before editing)")
    lines += ["", "INTERFACE CONTRACTS (every later step must use these "
              "EXACT names and argument counts)"]
    lines += [f"- {c}" for c in data["contracts"]] or \
        ["- (no cross-file contracts required by this change)"]
    lines += ["", "REQUIREMENTS"]
    if data["requirements"]:
        lines += [f"{i}. {r}" for i, r in enumerate(data["requirements"], 1)]
    else:
        lines.append("1. Implement exactly the objective above; add nothing "
                     "it does not ask for.")
    lines += ["", "EDGE CASES"]
    cases = data["edge_cases"] or _edge_case_fallback(data)
    lines += [f"{i}. {e}" for i, e in enumerate(cases, 1)]
    lines += ["", "VERIFICATION",
              data["verification"] or "Run the project's tests (or its entry "
              "point) and confirm the objective's observable end state."]
    lines += ["", "ORIGINAL REQUEST (verbatim)", original.strip() or "(empty)"]
    if mode:
        lines.append(f"\n(mode: {mode})")
    return "\n".join(lines).strip() + "\n"


def _fallback_spec(prompt: str, snapshot: str, mode: str) -> dict:
    """Deterministic architect: used when no model answers (offline server,
    JSON failure) and for unit tests. Shapes the idea into the same sections
    as the model path so downstream stages always get a structure."""
    first = next((ln.strip() for ln in prompt.splitlines() if ln.strip()), "")
    files: list[str] = []
    in_files = False
    for ln in snapshot.splitlines():
        s = ln.strip()
        if s.startswith("EXISTING PROJECT FILES:"):
            in_files = True
        elif s.startswith("OUTLINE"):
            in_files = False
        elif in_files and s and not s.startswith("("):
            files.append(s)
    data = {
        "objective": (f"Deliver exactly this request, repo-aware: {first}"
                      if first else "Deliver the request stated below."),
        "files": [{"path": f, "action": "modify",
                   "responsibility": "existing project file -- inspect it "
                   "before deciding whether to modify or create beside it"}
                  for f in files[:8]],
        "contracts": [],
        "requirements": [
            f"Implement the request: {first or prompt.strip()} -- verbatim "
            "scope, no extra features.",
            "Before editing any file, read it and keep every existing public "
            "signature exactly as it is.",
            "Any new public function or class must be given one exact "
            "signature here before a later step calls it.",
        ],
        "edge_cases": [],
        "verification": ("Run the project's test suite (or import/run the "
                         "entry point) after the change and confirm the "
                         "objective's observable end state."),
    }
    data["edge_cases"] = _edge_case_fallback(data)
    return data


# ------------------------------------------------------------- refine entry

def refine_prompt(prompt: str, root, cfg, llm_client,
                  mode: str = "auto") -> dict:
    """Expand a brief user idea into a concrete, repo-aware specification.

    Returns {"ok": True, "original": prompt, "refined": str, "contracts":
    [str, ...]} -- always shaped that way. The repo snapshot (paths plus AST
    outlines of the files the prompt names) costs no model call; exactly one
    schema-constrained call does the expansion, and any failure to reach or
    parse the model falls back to the deterministic architect so the button
    and --refine never dead-end."""
    prompt = str(prompt or "")
    root = Path(root) if root is not None else Path.cwd()

    # 1. zero-LLM structural snapshot --------------------------------------
    mentions = {tok.lower() for tok in re.findall(r"[\w./-]{3,}", prompt)}
    rels = _iter_code_files(root, PY_EXTS | TS_EXTS | DOC_EXTS, cap=120)
    entry = {"main.py", "app.py", "cli.py", "index.html", "README.md",
             "pyproject.toml", "package.json", "__init__.py"}
    scored = []
    for rel in rels:
        low = rel.lower()
        base = low.rsplit("/", 1)[-1]
        score = 0
        if any(tok in low for tok in mentions):
            score -= 3
        if base in entry or low in entry:
            score -= 2
        score += low.count("/")
        scored.append((score, rel))
    scored.sort()
    picked = [rel for _s, rel in scored[:40]]
    snap_lines = ["EXISTING PROJECT FILES:"]
    snap_lines += [f"  {rel}" for rel in picked] or ["  (empty project)"]
    outlined = 0
    for rel in picked:
        if outlined >= 3 or rel.rsplit(".", 1)[-1].lower() not in \
                {e[1:] for e in PY_EXTS | TS_EXTS}:
            continue
        sigs = _outline(rel, root)
        if not sigs:
            continue
        snap_lines.append(f"OUTLINE {rel}:")
        snap_lines += [f"  {s}" for s in sigs[:12]]
        outlined += 1
    snapshot = "\n".join(snap_lines)[:6000]

    # 2. one schema-constrained model call ----------------------------------
    data = None
    if llm_client is not None:
        user = (f"MODE: {mode}\n\nBRIEF IDEA (from the user)\n\n"
                f"{prompt.strip()[:4000]}\n\n{snapshot}\n\n"
                "Expand the idea into the actionable spec fields now.")
        messages = [{"role": "system", "content": REFINE_SYSTEM},
                    {"role": "user", "content": user}]
        try:
            raw = llm_client.chat_json(
                messages, REFINE_SCHEMA,
                max_tokens=int(_get(cfg, "max_tokens_answer", 3000)),
                disable_thinking=True)
            if isinstance(raw, dict):
                data = _norm_refined(raw, prompt)
                if not data["objective"] and not data["files"] \
                        and not data["requirements"]:
                    data = None       # a technically valid but useless reply
        except Exception:      # any failure to reach or parse the model --
            data = None        # the button/CLI must land on the fallback

    # 3. deterministic fallback --------------------------------------------
    if data is None:
        data = _norm_refined(_fallback_spec(prompt, snapshot, mode), prompt)

    refined = _render_refined(data, prompt, mode)
    return {"ok": True, "original": prompt, "refined": refined,
            "contracts": list(data["contracts"])}


# ---------------------------------------------------------------------------
# Fast static analysis gate (TECHNIQUES.md #19)
# ---------------------------------------------------------------------------
# The test subprocess is the most expensive way to learn that a line says
# `json.dumps` in a file that never imports json: a full interpreter start,
# collection, and often seconds of unrelated tests. This gate answers the
# same question in milliseconds and zero model tokens -- `ruff check` with a
# five-rule set when ruff is importable, a pure-stdlib scope resolver
# (`ast_static_errors`) when it is not, errors out, or exceeds its timeout.
# Callers diff the result against the project state from BEFORE the step
# (see loop._capture_static_baseline), so only errors the step itself
# introduced can trigger the cheap repair microloop.

_STATIC_GATE_CODES = ("F821", "F822", "F823", "F402", "F404")
# E999 was removed from selectable rules in recent ruff; syntax errors now
# arrive as `invalid-syntax`, which we normalize back to the E999 label so
# ruff and the AST fallback agree on the error key.
_STATIC_SYNTAX_CODE = "E999"

# The stdlib/typing names small models forget to import most often: the hint
# turns a dead-end message into a one-glance fix.
_COMMON_IMPORT_HINTS: dict[str, str] = {
    "json": "import json", "re": "import re", "os": "import os",
    "sys": "import sys", "math": "import math", "time": "import time",
    "datetime": "import datetime", "collections": "import collections",
    "itertools": "import itertools", "functools": "import functools",
    "subprocess": "import subprocess", "shutil": "import shutil",
    "tempfile": "import tempfile", "textwrap": "import textwrap",
    "hashlib": "import hashlib", "random": "import random",
    "pathlib": "import pathlib", "Path": "from pathlib import Path",
    "typing": "import typing", "Optional": "from typing import Optional",
    "Any": "from typing import Any", "Callable": "from typing import Callable",
    "Dict": "from typing import Dict", "List": "from typing import List",
    "Set": "from typing import Set", "Tuple": "from typing import Tuple",
    "Union": "from typing import Union", "Iterable": "from typing import Iterable",
    "Sequence": "from typing import Sequence", "Mapping": "from typing import Mapping",
    "pytest": "import pytest", "yaml": "import yaml",
    "argparse": "import argparse", "csv": "import csv",
    "sqlite3": "import sqlite3", "base64": "import base64",
    "urllib": "import urllib", "traceback": "import traceback",
    "unicodedata": "import unicodedata", "copy": "import copy",
    "warnings": "import warnings", "dataclass": "from dataclasses import dataclass",
    "dataclasses": "import dataclasses", "enum": "import enum",
    "annotations": "`from __future__ import annotations`",
}

# ------------------------------------------------- deterministic import heal
# TECHNIQUES.md #19's cheapest repair is no repair at all: when an F821 names
# a stdlib top-level module or a public symbol defined in exactly ONE project
# file, the import is mechanically known. Insert it, re-run the same AST gate
# in memory, and only keep the insertion when the target error disappears and
# no new error appears. Everything ambiguous falls through untouched.

_AUTO_HEAL_EXTRA_STDLIB = {"pytest"}
_F821_NAME_RE = re.compile(r"`([^`]+)`")


def _auto_heal_enabled(cfg) -> bool:
    return bool(_get(cfg, "auto_heal_imports", True))


def _stdlib_top_level_names() -> frozenset[str]:
    names = set(getattr(sys, "stdlib_module_names", ())) or set()
    names |= _AUTO_HEAL_EXTRA_STDLIB
    # Builtins are handled by the resolver; `__future__`/`__main__` are not
    # import targets for a bare name in normal code.
    return frozenset(n for n in names
                     if n and not n.startswith("_") and n != "__future__")


def f821_names(errors) -> list[str]:
    """Undefined names named by F821 gate errors, sorted and deduplicated."""
    out: set[str] = set()
    for err in errors or ():
        text = str(err or "")
        if ": F821 " not in text:
            continue
        m = _F821_NAME_RE.search(text)
        if m:
            name = m.group(1).strip()
            if name.isidentifier():
                out.add(name)
    return sorted(out)


def _bound_names(tree: ast.AST) -> set[str]:
    """Every name the file binds anywhere. Used as a conservative shadow
    check: if the name is bound at all, auto-healing is not attempted."""
    out: set[str] = set()

    def target(node) -> None:
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                target(elt)
        elif isinstance(node, ast.Starred):
            target(node.value)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.arg):
            out.add(node.arg)
        elif isinstance(node, ast.alias):
            out.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            out.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            out.update(node.names)
        elif isinstance(node, ast.Name) and isinstance(
                node.ctx, (ast.Store, ast.Del)):
            out.add(node.id)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target])
            for t in targets:
                target(t)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            target(node.target)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            target(node.optional_vars)
        elif isinstance(node, ast.MatchAs) and node.name:
            out.add(node.name)
        elif isinstance(node, ast.MatchStar) and node.name:
            out.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            out.add(node.rest)
        elif isinstance(node, ast.comprehension):
            target(node.target)
        for tp in getattr(node, "type_params", ()) or ():
            out.add(getattr(tp, "name", ""))
    return {n for n in out if n}


def _insert_after_headers(src: str, tree: ast.AST) -> int:
    """Zero-based line index where a new module import belongs: after the
    shebang/coding cookie, module docstring, and every `from __future__`
    import (in that order)."""
    lines = src.split("\n")
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("#!") or re.match(r"^#.*coding[:=]", line):
            index += 1
            continue
        break
    body = getattr(tree, "body", [])
    if body and isinstance(body[0], ast.Expr) \
            and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        index = max(index, int(body[0].end_lineno or body[0].lineno or 1))
    for stmt in body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
            index = max(index, int(stmt.end_lineno or stmt.lineno or 1))
    return max(0, min(index, len(lines)))


def _module_path(rel: str) -> str:
    """Dotted import path for a project-relative .py file, or "" when a path
    segment is not a valid Python identifier."""
    name = rel[:-3] if rel.lower().endswith(".py") else rel
    name = name.replace("\\", "/").strip("/")
    if name.endswith("/__init__"):
        name = name[: -len("/__init__")]
    elif name == "__init__":
        return ""
    parts = name.split("/")
    if not all(p.isidentifier() for p in parts):
        return ""
    return ".".join(parts)


def _unique_project_symbols(root: Path | None, names: set[str],
                            current_rel: str) -> dict[str, str]:
    """{name: dotted module} for public top-level functions/classes defined
    by exactly ONE non-test project file. Ambiguous names are omitted."""
    if root is None or not names:
        return {}
    root = Path(root)
    found: dict[str, list[str]] = {name: [] for name in names}
    for rel in _iter_code_files(root, PY_EXTS, cap=400):
        if rel == current_rel or _is_test_file(rel):
            continue
        src = _read(rel, root)
        if not src.strip():
            continue
        try:
            tree = ast.parse(src)
        except (SyntaxError, ValueError):
            continue
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef)):
                continue
            if node.name in found and not node.name.startswith("_"):
                found[node.name].append(rel)
    out: dict[str, str] = {}
    for name, rels in found.items():
        if len(rels) != 1:             # zero, or a duplicate definition
            continue
        module = _module_path(rels[0])
        if module:
            out[name] = module
    return out


def auto_heal_source(rel: str, src: str, root: Path | None = None,
                     cfg=None, names=None) -> tuple[str, list[str]]:
    """Insert imports that provably fix F821 names in one Python source.

    Returns `(healed_source, receipts)`. `names` restricts healing to the
    errors the caller already classified as NEW (so pre-existing breakage is
    never laundered into the patch); with None every F821 in `src` is a
    candidate. Each insertion is validated with `ast.parse` plus the same
    AST static resolver before it is kept: exact, bounded, no model call."""
    if not _auto_heal_enabled(cfg) or not src.strip():
        return src, []
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return src, []
    # A star import or a dynamic scope makes the name set unknowable: never
    # guess an import there (the same house rule the AST resolver follows).
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(
                a.name == "*" for a in node.names):
            return src, []
        if isinstance(node, ast.Name) and node.id in ("exec", "eval",
                                                      "locals"):
            return src, []
        if isinstance(node, ast.Attribute) and node.attr in ("exec", "eval"):
            return src, []
    bound = _bound_names(tree)
    candidates = set(names) if names is not None else set(
        f821_names(ast_static_errors(rel, src)))
    candidates = {n for n in candidates
                  if n.isidentifier() and n not in _BUILTIN_NAMES
                  and n not in bound}
    if not candidates:
        return src, []
    stdlib = _stdlib_top_level_names()
    statements: list[tuple[str, str]] = []
    for name in sorted(candidates):
        if name in stdlib:
            statements.append((name, f"import {name}"))
    repo_names = {n for n in candidates if n not in stdlib}
    unique = _unique_project_symbols(root, repo_names, rel)
    for name in sorted(repo_names):
        module = unique.get(name)
        if module:
            statements.append((name, f"from {module} import {name}"))
    if not statements:
        return src, []

    keys = {static_error_key(e) for e in ast_static_errors(rel, src)}
    healed = src
    receipts: list[str] = []
    inserted = 0
    for name, stmt in statements:
        try:
            current_tree = ast.parse(healed)
        except (SyntaxError, ValueError):
            break
        index = _insert_after_headers(healed, current_tree) + inserted
        lines = healed.split("\n")
        lines.insert(min(index, len(lines)), stmt)
        trial = "\n".join(lines)
        try:
            ast.parse(trial)
        except (SyntaxError, ValueError):
            continue
        trial_errors = ast_static_errors(rel, trial)
        trial_keys = {static_error_key(e) for e in trial_errors}
        if name in f821_names(trial_errors):
            continue                    # the insertion did not resolve it
        if trial_keys - keys:
            continue                    # introduced a different error
        healed = trial
        keys = trial_keys
        inserted += 1
        receipts.append(f"[AUTO-HEALED: added '{stmt}' in {rel}]")
    return healed, receipts


def auto_heal_file(root: Path, rel: str, cfg=None,
                   names=None) -> list[str]:
    """Apply `auto_heal_source` to one file on disk; receipts or []."""
    if not _auto_heal_enabled(cfg):
        return []
    root = Path(root)
    path = root / fileio.normalise_rel(rel)
    try:
        src, style = fileio.read(path)
    except OSError:
        return []
    healed, receipts = auto_heal_source(fileio.normalise_rel(rel), src, root,
                                        cfg, names=names)
    if receipts and healed != src:
        try:
            fileio.write(path, healed, style)
        except OSError:
            return []
    return receipts


def heal_static_gate_errors(root: Path, files, errors, cfg=None
                            ) -> tuple[list[str], list[str]]:
    """Heal NEW F821 gate errors on disk, then re-run the gate on `files`.

    Returns `(errors_after, receipts)`. When nothing can be healed the
    original error list is returned unchanged, so the normal repair path sees
    exactly what it would have seen without this feature."""
    if not _auto_heal_enabled(cfg) or not errors:
        return list(errors or []), []
    by_file: dict[str, set[str]] = {}
    for err in errors:
        text = str(err or "")
        if ": F821 " not in text:
            continue
        m = re.match(r"^(?P<file>.+?):\d+:\d+: ", text)
        name_m = _F821_NAME_RE.search(text)
        if m and name_m:
            by_file.setdefault(m.group("file"), set()).add(
                name_m.group(1).strip())
    receipts: list[str] = []
    for rel, names in sorted(by_file.items()):
        receipts.extend(auto_heal_file(root, rel, cfg, names=names))
    if not receipts:
        return list(errors or []), []
    timeout = float(_get(cfg, "static_gate_timeout", _RUFF_TIMEOUT))
    try:
        after = run_fast_static_gate(root, files, timeout=timeout)
    except Exception:
        after = list(errors or [])
    return after, receipts


_RUFF_TIMEOUT = 3.0
_MAX_GATE_FILES = 80            # files per gate run (one subprocess)
_MAX_FILE_ERRORS = 12           # errors per file in the AST fallback
_MAX_TOTAL_ERRORS = 60          # runaway cap for any single gate run
_AST_NODE_LIMIT = 40_000        # the AST fallback skips pathological files

# One concise ruff line: "bad.py:2:12: F821 Undefined name `json`" or the
# newer "broken.py:1:7: invalid-syntax: Expected a parameter...".
_RUFF_LINE_RE = re.compile(
    r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+): "
    r"(?:(?P<code>[A-Z][A-Z0-9]{1,4})\s|(?P<rule>[a-z][a-z0-9-]*):\s)"
    r"(?P<msg>.*)$")

_ruff_cmd: list[str] | None = None       # None = not probed; [] = unusable here


def _ruff_command() -> list[str]:
    """argv prefix for the ruff CLI, or [] when ruff cannot run here. Cached;
    tests monkeypatch this to force either branch of the gate."""
    global _ruff_cmd
    if _ruff_cmd is not None:
        return _ruff_cmd
    exe = shutil.which("ruff")
    if exe:
        _ruff_cmd = [exe]
        return _ruff_cmd
    probe = [sys.executable, "-m", "ruff"]
    try:
        rc = subprocess.run([*probe, "--version"], capture_output=True,
                            timeout=15).returncode
    except (OSError, subprocess.TimeoutExpired):
        rc = -1
    _ruff_cmd = probe if rc == 0 else []
    return _ruff_cmd


def static_error_key(err: str) -> str:
    """The identity of a gate error with the volatile line/column removed: a
    name whose only change after a repair is its LINE is not a new error."""
    m = re.match(r"^(?P<f>.+?):(?:\d+):(?:\d+): (?P<rest>.*)$", err)
    if m:
        return f"{m.group('f')}: {m.group('rest')}"
    return err


_CONTRACT_ERR_RE = re.compile(
    r"^(?P<kind>[A-Za-z]*ContractError) in "
    r"(?P<file>.+?):(?:\d+): (?P<rest>.*)$")


def contract_error_key(err: str) -> str:
    """Contract-error identity without the line number, for baseline diffing.
    A pre-existing violation that only shifted lines is still pre-existing."""
    m = _CONTRACT_ERR_RE.match(str(err or ""))
    if m:
        return f"{m.group('kind')} in {m.group('file')}: {m.group('rest')}"
    return str(err or "")


def contract_error_file(err: str) -> str:
    """The file part of a ContractError string ("" when unparseable)."""
    m = _CONTRACT_ERR_RE.match(str(err or ""))
    return m.group("file") if m else ""


def _decorate(rel: str, line: int, col: int, code: str, msg: str) -> str:
    """One gate error: `<file>:<line>:<col>: <CODE> <message> [hint]`. Same
    shape and wording from ruff and from the AST fallback, so baseline keys
    match whichever engine produced them."""
    msg = msg.strip().rstrip(".")
    if code in ("F821", "F822", "F823"):
        m = re.search(r"`([^`]+)`|'([^']+)'", msg)
        name = (m.group(1) or m.group(2)) if m else ""
        if name:
            if name in _COMMON_IMPORT_HINTS:
                msg += f" -- did you forget `{_COMMON_IMPORT_HINTS[name]}`?"
            elif code == "F823":
                msg += " -- move the assignment above its use"
            elif code == "F822":
                msg += " -- export it in this file, or drop it from `__all__`"
    return f"{rel}:{line}:{col}: {code} {msg}"


def list_python_files(root: Path, cap: int = 400) -> list[Path]:
    """Every scannable project ``.py`` file (bounded; never crossing into
    vendored or hidden directories), for the per-step baseline scan."""
    return [root / rel for rel in _iter_code_files(root, PY_EXTS, cap)]


def run_fast_static_gate(
    root: Path,
    modified_files: Iterable[Path],
    timeout: float = _RUFF_TIMEOUT,
) -> list[str]:
    """Gate the modified Python files: undefined names, bad ``__all__``
    entries, use-before-binding, misplaced imports, and syntax errors -- in
    milliseconds, zero model tokens.

    Prefers ``ruff check --isolated`` with the five pyflakes codes; falls back
    to the pure-AST resolver when ruff is missing, fails, or times out. Only
    existing ``.py`` files are considered; non-Python and missing paths are
    ignored. Returns ``<file>:<line>:<col>: <CODE> <message>`` strings; an
    empty list means the gate found nothing, not that it could not run."""
    files: list[Path] = []
    seen: set[str] = set()
    for f in modified_files:
        p = Path(f) if not isinstance(f, Path) else f
        if not p.is_absolute():
            p = root / p
        try:
            if p.suffix.lower() != ".py" or not p.is_file():
                continue
            key = str(p)
        except OSError:
            continue
        if key not in seen:
            seen.add(key)
            files.append(p)
        if len(files) >= _MAX_GATE_FILES:
            break
    if not files:
        return []
    errors = _ruff_gate(root, files, timeout)
    if errors is None:
        errors = []
        for p in files:
            errors.extend(_ast_gate_file(root, p))
            if len(errors) >= _MAX_TOTAL_ERRORS:
                break
    return errors[:_MAX_TOTAL_ERRORS]


def _ruff_gate(root: Path, files: list[Path],
               timeout: float) -> list[str] | None:
    """Parsed ruff output, or None when the subprocess is unusable (missing,
    timed out, unexpected exit): the caller then trusts the AST fallback."""
    cmd = _ruff_command()
    if not cmd:
        return None
    argv = [*cmd, "check", "--isolated", "--no-cache", "--quiet",
            "--output-format", "concise",
            "--select", ",".join(_STATIC_GATE_CODES), "--", *files]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout, cwd=root,
                              encoding="utf-8", errors="replace")
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode not in (0, 1):
        return None
    out: list[str] = []
    raw = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
    matched = 0
    for ln in raw:
        m = _RUFF_LINE_RE.match(ln.strip())
        if not m:
            continue
        matched += 1
        code = m.group("code") or _STATIC_SYNTAX_CODE   # invalid-syntax -> E999
        rel = _gate_label(root, m.group("file"))
        out.append(_decorate(rel, int(m.group("line")), int(m.group("col")),
                             code, m.group("msg")))
    if raw and matched == 0:
        return None          # output shaped unexpectedly: trust the AST instead
    return out


def _gate_label(root: Path, name: str) -> str:
    """Project-relative label for any engine's path, so ruff and the AST
    fallback (and the baseline) always agree on a key's file part.

    Windows-formatted diagnostics use backslashes; normalise them before
    Path() so `src\\app.py` and `src/app.py` produce the same key on every
    platform (on POSIX, `Path` would otherwise treat the backslashes as part
    of one filename)."""
    name = fileio.normalise_rel(name)
    p = Path(name)
    if not p.is_absolute():
        p = root / name
    try:
        return p.resolve().relative_to(Path(root).resolve()).as_posix()
    except (OSError, ValueError):
        return p.as_posix()


# --- the pure-AST fallback resolver ------------------------------------------
# The subset of pyflakes that pays for a repair round: F821 (undefined name),
# F822 (`__all__` entry missing at module level), F823 (local shadow used
# before its binding), E999 (syntax error). House rule as everywhere in this
# module: anything the resolver cannot decide with certainty is SKIPPED -- a
# false positive would burn a repair round on working code.

_BUILTIN_NAMES = frozenset(dir(builtins)) | {
    "_", "__file__", "__name__", "__doc__", "__package__", "__spec__",
    "__loader__", "__builtins__", "__debug__", "__path__", "__class__",
    "__dict__", "__version__", "__annotations__",
}

_TYPE_ALIAS = getattr(ast, "TypeAlias", None)     # 3.12+; None on 3.11


class _Scope:
    """One lexical scope (module / class / function / lambda / comprehension):
    the names it binds, and the Name-load nodes seen directly in it. Loads are
    resolved in a second pass, once every binding anywhere has been
    collected, so forward references inside function bodies stay quiet."""

    __slots__ = ("kind", "parent", "bound", "uses", "bind_lines")

    def __init__(self, kind: str, parent: "_Scope | None"):
        self.kind = kind
        self.parent = parent
        self.bound: set[str] = set()
        self.uses: list[ast.Name] = []
        self.bind_lines: dict[str, int] = {}

    def bind(self, name: str, line: int) -> None:
        self.bound.add(name)
        prev = self.bind_lines.get(name)
        if prev is None or line < prev:
            self.bind_lines[name] = line


def ast_static_errors(rel: str, src: str) -> list[str]:
    """Gate errors for one Python source string (used on file contents in the
    fallback and on dry-applied buffers by ``verify.check_static``)."""
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        line = max(int(exc.lineno or 1), 1)
        col = max(int(exc.offset or 1), 1)
        msg = (exc.msg or "invalid syntax").splitlines()[0].strip()
        return [_decorate(rel, line, col, _STATIC_SYNTAX_CODE,
                          f"invalid-syntax: {msg}")]
    except (ValueError, RecursionError, MemoryError):
        return []
    if sum(1 for _ in ast.walk(tree)) > _AST_NODE_LIMIT:
        return []
    return _StaticResolver(rel).run(tree)


class _StaticResolver:
    def __init__(self, rel: str):
        self.rel = rel
        self.errors: list[str] = []
        self.scopes: list[_Scope] = []
        self.declared: set[str] = set()       # global/nonlocal: never guess
        self.skip_all = False                 # star import / exec / eval seen
        self.module: _Scope | None = None
        self.all_node: ast.Assign | None = None
        self.import_lines: dict[str, int] = {}    # module-level import alias
        self.top_fors: list[ast.For] = []

    def emit(self, node: ast.AST, code: str, msg: str) -> None:
        if len(self.errors) >= _MAX_FILE_ERRORS:
            return
        line = max(int(getattr(node, "lineno", 1) or 1), 1)
        col = max(int(getattr(node, "col_offset", 0) or 0) + 1, 1)
        self.errors.append(_decorate(self.rel, line, col, code, msg))

    # ---- pass 1: build the scope tree, collect bindings and uses ----------
    def run(self, tree: ast.Module) -> list[str]:
        mod = _Scope("module", None)
        self.module = mod
        self.all_node = self._find_all(tree.body)
        self.future_annotations = any(
            isinstance(s, ast.ImportFrom) and s.module == "__future__"
            and any(a.name == "annotations" for a in s.names)
            for s in tree.body)
        self._stmts(tree.body, mod, top=True)
        if not (self.skip_all or self._uses_dynamic_scope(tree)):
            self._resolve_names()
        self._check_all(mod)
        if not (self.skip_all or self.declared):
            self._check_f823()
        self._check_f404(tree)
        self._check_f402()
        return self.errors

    def _uses_dynamic_scope(self, tree: ast.Module) -> bool:
        """exec()/eval()/locals() anywhere makes name availability opaque."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in ("exec", "eval",
                                                          "locals"):
                return True
            if isinstance(node, ast.Attribute) and node.attr in ("exec",
                                                                 "eval"):
                return True
        return False

    def _find_all(self, body: list[ast.stmt]) -> ast.Assign | None:
        for stmt in body:
            if (isinstance(stmt, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "__all__"
                            for t in stmt.targets)):
                return stmt
        return None

    def _stmts(self, body: list[ast.stmt], scope: _Scope,
               top: bool = False) -> None:
        for stmt in body:
            self._stmt(stmt, scope, top)

    def _stmt(self, node: ast.stmt, scope: _Scope, top: bool) -> None:
        k = type(node)
        if k is ast.FunctionDef or k is ast.AsyncFunctionDef:
            scope.bind(node.name, node.lineno)
            parent = scope if scope.kind != "class" else self._func_parent(scope)
            inner = _Scope("function", parent)
            self.scopes.append(inner)
            for d in node.decorator_list:
                self._expr(d, scope)
            a = node.args
            for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs]:
                inner.bind(arg.arg, node.lineno)
            if a.vararg is not None:
                inner.bind(a.vararg.arg, node.lineno)
            if a.kwarg is not None:
                inner.bind(a.kwarg.arg, node.lineno)
            for d in [*a.defaults, *[x for x in a.kw_defaults if x]]:
                self._expr(d, scope)
            for tp in getattr(node, "type_params", ()) or ():
                inner.bind(tp.name, node.lineno)
            self._stmts(node.body, inner)
            if not self.future_annotations:
                for ann in self._ann_exprs(node):
                    self._expr(ann, scope)
            return
        if k is ast.ClassDef:
            scope.bind(node.name, node.lineno)
            for d in node.decorator_list:
                self._expr(d, scope)
            for b in [*node.bases, *node.keywords]:
                self._expr(b, scope)
            inner = _Scope("class", scope)
            self.scopes.append(inner)
            for tp in getattr(node, "type_params", ()) or ():
                inner.bind(tp.name, node.lineno)
            self._stmts(node.body, inner)
            return
        if k is ast.Assign:
            for tgt in node.targets:
                self._bind_target(tgt, scope)
            self._expr(node.value, scope)
            return
        if k is ast.AnnAssign:
            if node.value is not None and isinstance(node.target, ast.Name):
                scope.bind(node.target.id, node.target.lineno)
            if node.value is not None:
                self._expr(node.value, scope)
            if not self.future_annotations and isinstance(node.annotation,
                                                          ast.AST):
                self._expr(node.annotation, scope)
            return
        if k is ast.AugAssign:
            if isinstance(node.target, ast.Name):
                self._use(node.target, scope)
            self._expr(node.value, scope)
            return
        if k is ast.Import:
            for a in node.names:
                name = (a.asname or a.name).split(".")[0]
                scope.bind(name, node.lineno)
                if top:
                    self.import_lines.setdefault(name, node.lineno)
            return
        if k is ast.ImportFrom:
            for a in node.names:
                if a.name == "*":
                    self.skip_all = True
                    continue
                name = a.asname or a.name
                scope.bind(name, node.lineno)
                if top:
                    self.import_lines.setdefault(name, node.lineno)
            return
        if k in (ast.For, ast.AsyncFor):
            if top and isinstance(node.target, ast.Name):
                self.top_fors.append(node)
            self._bind_target(node.target, scope)
            self._expr(node.iter, scope)
            self._stmts(node.body, scope)
            self._stmts(node.orelse, scope)
            return
        if k in (ast.With, ast.AsyncWith):
            for item in node.items:
                self._expr(item.context_expr, scope)
                if item.optional_vars is not None:
                    self._bind_target(item.optional_vars, scope)
            self._stmts(node.body, scope)
            return
        if k in (ast.Try, ast.TryStar):
            self._stmts(node.body, scope)
            for h in node.handlers:
                if h.type is not None:
                    self._expr(h.type, scope)
                if h.name:
                    scope.bind(h.name, node.lineno)
                self._stmts(h.body, scope)
            self._stmts(node.orelse, scope)
            self._stmts(node.finalbody, scope)
            return
        if k in (ast.If, ast.While):
            self._expr(node.test, scope)
            self._stmts(node.body, scope)
            self._stmts(node.orelse, scope)
            return
        if k is ast.Match:
            self._expr(node.subject, scope)
            for case in node.cases:
                self._match_pattern(case.pattern, scope)
                if case.guard is not None:
                    self._expr(case.guard, scope)
                self._stmts(case.body, scope)
            return
        if k is ast.Global:
            self.declared.update(node.names)
            if self.module is not None:
                for n in node.names:
                    self.module.bind(n, node.lineno)
            return
        if k is ast.Nonlocal:
            self.declared.update(node.names)
            return
        if k is ast.Expr:
            self._expr(node.value, scope)
            return
        if _TYPE_ALIAS is not None and k is _TYPE_ALIAS:   # 3.12+ only
            if isinstance(node.name, ast.Name):
                scope.bind(node.name.id, node.lineno)
            self._expr(node.value, scope)
            return
        # Return / Raise / Assert / Delete: sweep the value-bearing fields.
        for fld in ("test", "value", "msg", "exc", "cause"):
            v = getattr(node, fld, None)
            if isinstance(v, ast.AST):
                self._expr(v, scope)

    def _func_parent(self, scope: _Scope) -> _Scope:
        """Functions never see class bodies: skip class scopes upward."""
        s = scope
        while s is not None and s.kind == "class":
            s = s.parent
        return s or self.module            # type: ignore[return-value]

    def _ann_exprs(self, node) -> list[ast.AST]:
        out = []
        for a in [*node.args.posonlyargs, *node.args.args,
                  *node.args.kwonlyargs,
                  *([node.args.vararg] if node.args.vararg else []),
                  *([node.args.kwarg] if node.args.kwarg else [])]:
            if a.annotation is not None:
                out.append(a.annotation)
        if node.returns is not None:
            out.append(node.returns)
        return out

    def _bind_target(self, tgt: ast.AST, scope: _Scope) -> None:
        k = type(tgt)
        if k is ast.Name:
            scope.bind(tgt.id, tgt.lineno)
        elif k in (ast.Tuple, ast.List):
            for e in tgt.elts:
                self._bind_target(e, scope)
        elif k is ast.Starred:
            self._bind_target(tgt.value, scope)
        # Attribute / Subscript targets bind no new names.

    def _match_pattern(self, p: ast.pattern, scope: _Scope) -> None:
        k = type(p)
        if k is ast.MatchAs:
            if p.pattern is not None:
                self._match_pattern(p.pattern, scope)
            if p.name:
                scope.bind(p.name, getattr(p, "lineno", 1) or 1)
            return
        if k is ast.MatchStar:
            if p.name:
                scope.bind(p.name, getattr(p, "lineno", 1) or 1)
            return
        if k is ast.MatchMapping:
            if p.rest:
                scope.bind(p.rest, getattr(p, "lineno", 1) or 1)
            for v in p.keys:
                self._expr(v, scope)
            for v in p.patterns:
                self._match_pattern(v, scope)
            return
        if k is ast.MatchValue:
            self._expr(p.value, scope)
            return
        if k is ast.MatchSingleton:
            return
        for attr in ("cls", "patterns", "kwd"):
            v = getattr(p, attr, None)
            if isinstance(v, ast.AST):
                self._expr(v, scope)
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, ast.pattern):
                        self._match_pattern(x, scope)
                    elif isinstance(x, ast.AST):
                        self._expr(x, scope)
        if k is ast.MatchClass:
            for x in p.kwd_patterns:
                self._match_pattern(x, scope)

    # ---- expressions: lambda and comprehension scopes own their names ------
    def _expr(self, node: ast.AST | None, scope: _Scope) -> None:
        if node is None:
            return
        k = type(node)
        if k is ast.Name:
            if isinstance(node.ctx, ast.Load):
                self._use(node, scope)
            return
        if k is ast.Lambda:
            inner = _Scope("lambda", scope)
            self.scopes.append(inner)
            a = node.args
            for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs]:
                inner.bind(arg.arg, node.lineno)
            if a.vararg is not None:
                inner.bind(a.vararg.arg, node.lineno)
            if a.kwarg is not None:
                inner.bind(a.kwarg.arg, node.lineno)
            for d in [*a.defaults, *[x for x in a.kw_defaults if x]]:
                self._expr(d, scope)
            self._expr(node.body, inner)
            return
        if k in (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp):
            inner = _Scope("comprehension", scope)
            self.scopes.append(inner)
            for i, gen in enumerate(node.generators):
                self._expr(gen.iter, scope if i == 0 else inner)
                self._bind_target(gen.target, inner)
                for cond in gen.ifs:
                    self._expr(cond, inner)
            if k is ast.DictComp:
                self._expr(node.key, inner)
                self._expr(node.value, inner)
            else:
                self._expr(node.elt, inner)
            return
        if k is ast.NamedExpr:                     # walrus binds in both
            self._expr(node.value, scope)
            if isinstance(node.target, ast.Name):
                self._use(node.target, scope)
                scope.bind(node.target.id, node.target.lineno)
                if scope.kind in ("comprehension", "lambda"):
                    leak = scope.parent
                    while leak is not None and leak.kind in (
                            "comprehension", "lambda"):
                        leak = leak.parent
                    if leak is not None:
                        leak.bind(node.target.id, node.target.lineno)
            return
        if k is ast.Attribute:
            self._expr(node.value, scope)
            return
        if k is ast.Constant or k is ast.arguments:
            return
        for _fld, v in ast.iter_fields(node):
            if isinstance(v, ast.AST):
                self._expr(v, scope)
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, ast.AST):
                        self._expr(x, scope)

    def _use(self, node: ast.Name, scope: _Scope) -> None:
        scope.uses.append(node)

    # ---- pass 2: resolve uses ----------------------------------------------
    def _chain(self, scope: _Scope) -> list[_Scope]:
        out, s = [], scope
        while s is not None:
            out.append(s)
            s = s.parent
        return out

    def _any_func_binding(self, name: str) -> bool:
        """Bound inside some function/lambda/comprehension anywhere: a use
        elsewhere cannot prove F821 -- a lazy import or a late binding may
        still land first. ruff is exact here; the fallback errs silent.
        CLASS scopes do not count: a class attribute is never visible to a
        method at runtime, so that use is a genuine undefined name."""
        return any(name in s.bound for s in self.scopes
                   if s.kind in ("function", "lambda", "comprehension"))

    def _resolve_names(self) -> None:
        for scope in [self.module, *self.scopes]:
            if scope is None:
                continue
            for node in scope.uses:
                self._resolve(node, scope)

    def _resolve(self, node: ast.Name, scope: _Scope) -> None:
        name = node.id
        if name in _BUILTIN_NAMES or name in self.declared:
            return
        for s in self._chain(scope):
            if name in s.bound:
                return
        if self._any_func_binding(name):
            return
        self.emit(node, "F821", f"Undefined name `{name}`")

    def _check_all(self, mod: _Scope) -> None:
        if self.skip_all:
            return                                # star import: names unknown
        node = self.all_node
        if node is None or not isinstance(node.value, (ast.List, ast.Tuple)):
            return                                # built dynamically: opaque
        elts = node.value.elts
        for e in elts:
            if not (isinstance(e, ast.Constant) and isinstance(e.value, str)):
                return                            # not a plain name list
        for e in elts:
            if e.value not in mod.bound:
                self.emit(node, "F822",
                          f"Undefined name `{e.value}` in `__all__`")

    def _check_f823(self) -> None:
        """A name used in a function's own scope before its local binding,
        while a module-level binding shadows the read: UnboundLocalError."""
        mod = self.module
        if mod is None:
            return
        for s in self.scopes:
            if s.kind != "function":
                continue
            for node in s.uses:
                name = node.id
                if (name in _BUILTIN_NAMES or name not in s.bound
                        or name not in mod.bound or name in self.declared):
                    continue
                bline = s.bind_lines.get(name, 0)
                if bline and node.lineno < bline:
                    self.emit(node, "F823",
                              f"Local variable `{name}` referenced before "
                              "assignment")

    def _check_f404(self, tree: ast.Module) -> None:
        seen_real = False
        for i, stmt in enumerate(tree.body):
            if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
                if seen_real:
                    self.emit(stmt, "F404",
                              "`from __future__` imports must occur at the "
                              "beginning of the file")
                continue
            if (isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str) and i == 0):
                continue                          # module docstring
            if isinstance(stmt, (ast.Pass, ast.ImportFrom)):
                continue
            seen_real = True

    def _check_f402(self) -> None:
        """Module-level `import x`, then `for x in ...`: the loop variable
        silently shadows the import for the rest of the file."""
        for stmt in self.top_fors:
            name = stmt.target.id
            imp = self.import_lines.get(name)
            if imp is not None and stmt.lineno > imp:
                self.emit(stmt, "F402",
                          f"Import `{name}` from line {imp} shadowed by "
                          "loop variable")


# Per-file cache for the fallback: (mtime_ns, size) -> errors. The gate runs
# per repair round over the same changed files plus the whole-project
# baseline; unchanged files then cost nothing.
_AST_CACHE: dict[str, tuple[int, int, list[str]]] = {}


# --------------------------------------------- frontend DOM contracts gate
# The deterministic twin of the import/endpoint gate for the browser side.
# An HTML+JS project drifts silently when an asset link rots, a script
# queries a DOM id no template ever defines, or app.js stops parsing: none
# of it raises until a human opens the page. Each check below is a regex or
# a subprocess the harness runs itself, so it costs nothing and invents
# nothing. Errors render in the same `...ContractError in <path>:<line>:
# <what the two sides disagree on>` shape that verify_cross_file_contracts
# uses, so the loop's repair path reads them unchanged.

_FE_HTML_EXTS = {".html", ".htm"}
_FE_JS_EXTS = {".js", ".mjs", ".cjs"}
_FE_SCRIPT_SRC_RE = re.compile(
    r"<script\b[^>]*\bsrc\s*=\s*[\"']([^\"']+)[\"']", re.I)
_FE_LINK_RE = re.compile(r"<link\b[^>]*>", re.I)
_FE_HREF_RE = re.compile(r"\bhref\s*=\s*[\"']([^\"']+)[\"']", re.I)
_FE_REL_RE = re.compile(r"\brel\s*=\s*[\"']([^\"']*)[\"']", re.I)
_FE_ID_ATTR_RE = re.compile(r"\bid\s*=\s*[\"']([A-Za-z][\w:.\-]*)[\"']")
_FE_GETID_RE = re.compile(r"""getElementById\(\s*(['"])([^'"]+)\1\s*\)""")
_FE_QS_ID_RE = re.compile(
    r"""querySelector(?:All)?\(\s*(['"])#([A-Za-z][\w:.\-]*)\1\s*\)""")
_FE_SETID_RE = re.compile(r"""\.id\s*=\s*(['"])([^'"]+)\1""")
_FE_SETATTR_ID_RE = re.compile(
    r"""setAttribute\(\s*(['"])id\1\s*,\s*(['"])([^'"]+)\2\s*\)""")
# a template literal that INTERPOLATES a backtick-quoted id (` `#${x}-row` `)
# means the id space is generated at runtime: no static query can be verified
# against it, so the file's id queries are exempt rather than guessed at.
_FE_TEMPLATE_INTERP_ID_RE = re.compile(
    r"id\s*=\s*(?:\\?[\"'])?\s*\$\{|[\"'`]\s*(?:#)?\s*\$\{")
# placeholders that mean "a server or template layer fills this in at render
# time" -- static verification cannot see the real path, so skip it.
_FE_TEMPLATE_REF_RE = re.compile(r"\{\{|\{%|<%[=.]?|\$\{")
# a bare `#anchor` in a stylesheet link or script src is in-page, not a file
_FE_ASSET_EXTS = (".js", ".mjs", ".cjs", ".css", ".html", ".htm")
_FE_ERRORS_CAP = 20


def _fe_line(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _fe_external(ref: str) -> bool:
    ref = ref.strip()
    return (not ref or ref.startswith(("http://", "https://", "//", "data:",
                                       "blob:", "mailto:", "#"))
            or _FE_TEMPLATE_REF_RE.search(ref) is not None)


def _fe_js_syntax_problem(rel: str, text: str) -> str | None:
    """Return `line: message` when `rel` does not parse, else None.
    `node --check` is the ground truth when node is installed; otherwise the
    tree-sitter grammar the indexer already uses reports the first ERROR
    node. No parser available: the check stays silent, never a guess."""
    node = shutil.which("node")
    if node:
        payload = text.encode("utf-8", "replace")

        def _check(*options: str) -> subprocess.CompletedProcess:
            # `input` already wires stdin; passing `stdin=PIPE` too raises
            # ValueError on every platform. The file may be a browser module
            # (`import`/`export`) even though node checks stdin as CommonJS,
            # so a script-mode failure is retried as an ES module.
            return subprocess.run([node, *options, "--check", "-"],
                                  input=payload, capture_output=True,
                                  timeout=15)

        try:
            proc = _check()
            if proc.returncode != 0:
                stderr = (proc.stderr or b"").decode("utf-8", "replace")
                esm_only = ("Cannot use import statement outside a module"
                            in stderr
                            or "Unexpected token 'export'" in stderr
                            or "Unexpected token 'import'" in stderr
                            or "top level bodies of modules" in stderr)
                if esm_only:
                    proc = _check("--input-type=module")
                    stderr = (proc.stderr or b"").decode("utf-8", "replace")
            if proc.returncode == 0:
                return None
            line = 1
            msg = ""
            m = re.search(r"file\S*?:(\d+)", stderr)
            if m:
                line = int(m.group(1))
            for ln in stderr.splitlines():
                if ln.strip().endswith("Error: " + ln.strip()[ln.strip().find(":") + 1:]) \
                        or "Error:" in ln:
                    msg = ln.strip()
                    break
            return f"{line}: {msg[:200] or 'node --check failed'}"
        except (OSError, subprocess.TimeoutExpired):
            pass  # node unusable right now: fall through to tree-sitter
    try:
        from . import chunker_ts
        parser = chunker_ts._get_parser("javascript")
    except Exception:
        parser = None
    if parser is None:
        return None
    try:
        with _ts_guard():
            tree = parser.parse(text.encode("utf-8", "replace"))
            root_node = tree.root_node
            if not root_node.has_error:
                return None
            node = root_node
            while True:
                bad = next((c for c in node.children
                            if c.type == "ERROR" or c.is_missing), None)
                if bad is None:
                    break
                node = bad
            line = node.start_point[0] + 1
            where = node.text.decode("utf-8", "replace")[:60].replace("\n", " ")
            return f"{line}: syntax error near {where!r}"
    except Exception:
        return None


def verify_frontend_contracts(root: Path, modified_files=None) -> list[str]:
    """Deterministic DOM Contract verification of the HTML/JS in `root`.

    Seven checks, all mechanical: (1) every <script src> / stylesheet <link
    href> in every HTML file points at a file that exists; (2) every literal
    DOM query (getElementById('x') / querySelector('#x')) matches an id the
    HTML defines (or a literal id the JS itself creates); files whose ids are
    generated in template literals are exempt -- nothing is verified against
    a moving target; (3) every JS file parses, via `node --check` when node
    is installed and tree-sitter otherwise; (4) relative ES module imports
    resolve and name real exports, and imported calls satisfy the known
    arity range (the same `ContractError` diagnostics as the Python gate);
    (5) every module binds each name once and declares or imports each name
    it calls; (6) no assignment targets a const/import binding (a parse-clean
    TypeError at runtime); (7) every `x.method()` on a provable project-class
    instance names a method the class actually defines.

    `modified_files` (root-relative paths, as the loop reports them) scopes
    the REPORT: only errors inside those files come back, so a pre-existing
    rot elsewhere cannot block the current task. Pass None/empty to verify
    the whole frontend (scaffold and one-shot audits). Returns plain strings,
    sorted, capped -- deterministic, no LLM."""
    root = Path(root)
    html_rels = _iter_code_files(root, _FE_HTML_EXTS)
    js_rels = _iter_code_files(root, _FE_JS_EXTS)
    module_rels = _iter_code_files(root, TS_EXTS)
    if not html_rels and not js_rels and not module_rels:
        return []
    scope = None
    if modified_files:
        scope = set()
        for f in modified_files:
            try:
                scope.add(Path(str(f)).relative_to(root).as_posix())
            except (ValueError, OSError, TypeError):
                scope.add(str(f).replace("\\", "/"))
    html_text = {rel: _read(rel, root) for rel in html_rels}
    js_text = {rel: _read(rel, root) for rel in js_rels}
    errors: list[str] = []

    html_changed = scope is not None and any(
        rel in scope for rel in html_rels)
    if scope is not None and not html_changed \
            and not any(rel in scope for rel in js_rels + module_rels):
        return []   # an edit to a non-frontend file cannot break the page

    def note(rel: str, line: int, msg: str, peer: bool = False) -> None:
        if scope is None or rel in scope:
            errors.append(f"FrontendContractError in {rel}:{line}: {msg}")
        elif peer and html_changed:
            # the id registry moved: a query in UNTOUCHED JS can have been
            # broken by the HTML edit, and that is this task's breakage
            errors.append(f"FrontendContractError in {rel}:{line}: {msg}")

    # 1. asset link integrity: html references a script/css path, file absent
    for rel, text in html_text.items():
        refs = []
        for m in _FE_SCRIPT_SRC_RE.finditer(text):
            refs.append((m.group(1), m.start(), "script"))
        for m in _FE_LINK_RE.finditer(text):
            tag = m.group(0)
            if "stylesheet" not in (_FE_REL_RE.search(tag).group(1).lower()
                                    if _FE_REL_RE.search(tag) else ""):
                continue
            hm = _FE_HREF_RE.search(tag)
            if hm:
                refs.append((hm.group(1), m.start(), "stylesheet"))
        for ref, pos, kind in refs:
            if _fe_external(ref):
                continue
            clean = ref.split("?")[0].split("#")[0]
            if not clean or not clean.lower().endswith(_FE_ASSET_EXTS):
                continue  # an extensionless ref is a route, not an asset file
            base = (root / rel).parent
            cand = [(base / clean), (root / clean.lstrip("/"))]
            if not any(c.exists() for c in cand):
                note(rel, _fe_line(text, pos),
                     f"references missing {kind} asset '{ref}' -- create the "
                     f"file or fix the path; the browser would 404 it")

    # 2. DOM id cross-reference: JS queries vs ids the HTML/JS defines
    defined_ids: set[str] = set()
    html_names: list[str] = []
    for rel, text in html_text.items():
        found = set(_FE_ID_ATTR_RE.findall(text))
        if found:
            defined_ids |= found
            html_names.append(rel)
    dynamic_js = {rel for rel, text in js_text.items()
                  if _FE_TEMPLATE_INTERP_ID_RE.search(text)}
    if (html_text or defined_ids) and js_text:
        for rel, text in js_text.items():
            for m in _FE_SETID_RE.finditer(text):
                defined_ids.add(m.group(2))
            for m in _FE_SETATTR_ID_RE.finditer(text):
                defined_ids.add(m.group(3))
    for rel, text in js_text.items():
        if rel in dynamic_js:
            continue  # this file builds ids at runtime; nothing to verify
        if scope is not None and rel not in scope and not html_changed:
            continue  # neither this file nor the id registry was touched
        queries = [(m.group(2), m.start()) for m in _FE_GETID_RE.finditer(text)]
        queries += [(m.group(2), m.start())
                    for m in _FE_QS_ID_RE.finditer(text)]
        for ident, pos in sorted(set(queries), key=lambda t: t[1]):
            if ident in defined_ids or "%" in ident:
                continue
            if not html_text:
                break  # no templates at all: queries belong to a lib/test
            known = sorted(defined_ids)[:5]
            where = ", ".join(html_names[:3]) if html_names else "any script"
            note(rel, _fe_line(text, pos),
                 f"queries DOM element '#{ident}', but {where} only defines "
                 f"ids {known or '[]'} -- render it, or fix the selector",
                 peer=True)

    # 3. JS syntax: the file must parse or the browser aborts it entirely
    for rel, text in js_text.items():
        if not text.strip():
            continue
        problem = _fe_js_syntax_problem(rel, text)
        if problem:
            line, _, msg = problem.partition(": ")
            note(rel, int(line) if line.isdigit() else 1,
                 f"does not parse -- {msg[:220]}")

    # 4. ES module contracts: a relative import that points at no
    # file or names an export the target does not have, and an imported call
    # whose argument count cannot satisfy the real signature. Bare packages
    # and dynamic imports are deliberately absent from the helper's output.
    for importer, target, text in _js_cross_file_errors(root):
        if scope is None or importer in scope or (target and target in scope):
            errors.append(text)

    # 5. ES module WIRING: the two failures a parse accepts and the browser
    # aborts on -- a module-level name bound twice (`import { WAYPOINTS }`
    # appended by an incremental edit next to the import that already binds
    # it) and a call to a name this module neither declares nor imports.
    # Neither is visible to `has_syntax_error` or to the DOM checks above, and
    # both kill the whole module graph before a single line runs.
    for importer, target, text in _js_module_link_errors(root):
        if scope is None or importer in scope or (target and target in scope):
            errors.append(text)

    # 6. runtime-fatal assignments: a write to a const/import binding parses
    # clean and throws TypeError the first time the line runs (BUG-210).
    for rel, _peer, text in _js_const_reassign_errors(root):
        if scope is None or rel in scope:
            errors.append(text)

    # 7. instance-method contracts: `engine.setBpm(...)` on a project class
    # that defines no setBpm is a TypeError on the first slider move, and no
    # import/DOM check sees it (BUG-211).
    for rel, peer, text in _js_instance_method_errors(root):
        if scope is None or rel in scope or (peer and peer in scope):
            errors.append(text)
    return sorted(set(errors))[:_FE_ERRORS_CAP]


def _ast_gate_file(root: Path, path: Path) -> list[str]:
    try:
        st = path.stat()
    except OSError:
        return []
    key = str(path)
    hit = _AST_CACHE.get(key)
    if hit is not None and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
        return list(hit[2])
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    errs = ast_static_errors(_gate_label(root, key), src)
    if len(_AST_CACHE) > 3000:
        _AST_CACHE.clear()
    _AST_CACHE[key] = (st.st_mtime_ns, st.st_size, errs)
    return list(errs)
