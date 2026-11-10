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
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

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
                              timeout=timeout, cwd=root)
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
    fallback (and the baseline) always agree on a key's file part."""
    p = Path(name)
    if not p.is_absolute():
        p = root / p
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
        tree = parser.parse(text.encode("utf-8", "replace"))
    except Exception:
        return None
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


def verify_frontend_contracts(root: Path, modified_files=None) -> list[str]:
    """Deterministic DOM Contract verification of the HTML/JS in `root`.

    Three checks, all mechanical: (1) every <script src> / stylesheet <link
    href> in every HTML file points at a file that exists; (2) every literal
    DOM query (getElementById('x') / querySelector('#x')) matches an id the
    HTML defines (or a literal id the JS itself creates); files whose ids are
    generated in template literals are exempt -- nothing is verified against
    a moving target; (3) every JS file parses, via `node --check` when node
    is installed and tree-sitter otherwise.

    `modified_files` (root-relative paths, as the loop reports them) scopes
    the REPORT: only errors inside those files come back, so a pre-existing
    rot elsewhere cannot block the current task. Pass None/empty to verify
    the whole frontend (scaffold and one-shot audits). Returns plain strings,
    sorted, capped -- deterministic, no LLM."""
    root = Path(root)
    html_rels = _iter_code_files(root, _FE_HTML_EXTS)
    js_rels = _iter_code_files(root, _FE_JS_EXTS)
    if not html_rels and not js_rels:
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
            and not any(rel in scope for rel in js_rels):
        return []   # an edit to a non-frontend file cannot break a DOM contract

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
