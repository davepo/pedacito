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

All checks here run in-process with ``ast`` and regexes: no model call, no
subprocess, and deliberately conservative -- a false positive would send a
working project into a repair round, so anything ambiguous is skipped rather
than reported (the same rule `apicheck` follows).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from . import fileio
from .chunker import DEF_NODES, _signature

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

def extract_project_contracts(root: Path, max_files: int = 25,
                              max_chars: int = 1400,
                              modified: list[str] | None = None) -> str:
    """A compact map of the project's public signatures as
    `rel: def foo(x) -> y | class Bar` lines, capped near 300 tokens.

    `modified` (paths or rel strings, in any mix) sort those files first so
    the freshest contracts survive the cap; test files and unparsable files
    are skipped. Returns "" when the project has nothing to lock."""
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

    lines: list[str] = []
    used = 0
    files = 0
    for rel in rels:
        if files >= max_files or used >= max_chars:
            break
        if _is_test_file(rel):
            continue
        sigs = _outline(rel, root)
        if not sigs:
            continue
        entry = f"{rel}: " + "; ".join(sigs[:10])
        if len(entry) > 300:
            entry = entry[:297] + "..."
        if used + len(entry) > max_chars and lines:
            break
        lines.append(entry)
        used += len(entry) + 1
        files += 1
    return "\n".join(lines)


CONTRACT_HEADER = ("[PROJECT INTERFACE CONTRACTS — DO NOT BREAK SIGNATURES]")
CONTRACT_FOOTER = "[END INTERFACE CONTRACTS]"


def contract_map_block(root: Path, modified: list[str] | None = None) -> str:
    """The interface map as a ready-to-inject instruction block, or "" when
    the project exposes nothing yet (a near-empty tree gets no block)."""
    cmap = extract_project_contracts(root, modified=modified)
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


def verify_cross_file_contracts(root: Path,
                                modified_files: list | None = None) -> list[str]:
    """Cross-file import/arity violations in the project's Python files, as
    `ContractError in <file>:<line>: <detail>` strings (empty = consistent).

    Every local ``from <module> import <name>`` is checked against what
    <module> actually defines, and every direct call of such an imported name
    is checked for argument-count compatibility. `modified_files` only
    reorders diagnostics (files touched by the last step are reported first);
    the scan always covers the whole project, since a change in one file
    breaks contracts in its *importers*."""
    root = Path(root)
    if not root.is_dir():
        return []
    mods = _local_py_modules(root)
    if not mods:
        return []
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

    out.sort(key=lambda t: (t[0], t[1]))
    return [t for _o, _i, t in out][:20]


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
    if data["edge_cases"]:
        lines += [f"{i}. {e}" for i, e in enumerate(data["edge_cases"], 1)]
    else:
        lines.append("1. Handle empty input, missing files, and wrong types "
                     "with a clear error, never a crash.")
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
        "edge_cases": ["Empty or missing input fails with a clear message, "
                       "not a traceback."],
        "verification": ("Run the project's test suite (or import/run the "
                         "entry point) after the change and confirm the "
                         "objective's observable end state."),
    }
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
