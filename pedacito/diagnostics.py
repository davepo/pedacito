"""Project diagnostics: run the linter/type checker the repository already uses.

    ok, report = diagnostics.run(root, cfg, files=["src/app.py"])
    ok, report = diagnostics.run_changed(root, cfg, changed_files)

    baseline = diagnostics.quick_findings(root, files)   # before the task
    ...apply edits...
    new = diagnostics.new_findings(baseline, diagnostics.quick_findings(root, files))

A thin layer over tools that are already installed: nothing is bundled, and a
project with no detected checker is a no-op, not a failure. Checks run through
the bounded process runner (`proc`). Linting only the changed files keeps the
feedback short for small models. `quick_findings` is the cheap per-file pass
the post-edit feedback loop uses (one linter run per changed file; with no
linter installed, a syntax parse plus a conservative undefined-name check);
`new_findings` diffs it against a baseline
so only problems the edits introduced are reported back.

Detection (first match per language):

    Python   ruff check --output-format concise [files|.]; pyflakes, flake8;
             plus mypy or pyright when the project configures one
    JS/TS    package.json "lint" script, else tsc --noEmit when tsconfig
             exists, else eslint on the given files
    Rust     cargo check --message-format short
    Go       go vet ./...
    Ruby     rubocop (only when a Gemfile/.rubocop.yml is present)
"""

from __future__ import annotations

import ast
import builtins
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import proc


@dataclass(frozen=True)
class Check:
    label: str
    argv: list[str]


def _has(root: Path, name: str) -> bool:
    return (root / name).exists()


def _which(name: str) -> str:
    return shutil.which(name) or ""


def _py_typechecker_configured(root: Path, name: str) -> bool:
    if name == "mypy" and any(_has(root, n) for n in ("mypy.ini", ".mypy.ini")):
        return True
    if name == "pyright" and _has(root, "pyrightconfig.json"):
        return True
    for cfg in ("pyproject.toml", "setup.cfg", "tox.ini"):
        try:
            text = (root / cfg).read_text(encoding="utf-8")
        except OSError:
            continue
        if re.search(rf"^\[(tool\.)?{name}[\].]", text, re.M):
            return True
    return False


def _py_files(files: list[str] | None) -> list[str]:
    return [f for f in (files or []) if f.lower().endswith((".py", ".pyi"))]


def _js_files(files: list[str] | None) -> list[str]:
    return [f for f in (files or [])
            if f.lower().endswith((".js", ".jsx", ".mjs", ".cjs", ".ts",
                                   ".tsx"))]


def _package_scripts(root: Path) -> dict:
    try:
        data = json.loads((root / "package.json").read_text(encoding="utf-8"))
        return data.get("scripts") or {}
    except (OSError, ValueError):
        return {}


def detect(root, files: list[str] | None = None) -> list[Check]:
    """The checks applicable to this project (or these files)."""
    root = Path(root)
    checks: list[Check] = []
    files = list(files or [])

    py_targets = _py_files(files)
    python_project = bool(py_targets) or any(
        _has(root, n) for n in ("pyproject.toml", "setup.py", "setup.cfg",
                                "requirements.txt", "Pipfile"))
    if python_project:
        targets = py_targets or ["."]
        if _which("ruff"):
            checks.append(Check("ruff", ["ruff", "check", "--output-format",
                                         "concise", *targets]))
        elif _which("pyflakes"):
            checks.append(Check("pyflakes", ["pyflakes", *targets]))
        elif _which("flake8"):
            checks.append(Check("flake8", ["flake8", *targets]))
        # a type checker only when the project configured one (an unconfigured
        # run on a whole tree is mostly noise a small model then chases)
        if _py_typechecker_configured(root, "mypy") and _which("mypy"):
            checks.append(Check("mypy", ["mypy", "--no-error-summary",
                                         "--no-color-output", *targets]))
        elif _py_typechecker_configured(root, "pyright") and _which("pyright"):
            checks.append(Check("pyright", ["pyright", *targets]))

    if _has(root, "package.json"):
        scripts = _package_scripts(root)
        if scripts.get("lint") and _which("npm"):
            checks.append(Check("npm lint", ["npm", "run", "lint", "--silent"]))
        elif _has(root, "tsconfig.json") and _which("npx"):
            checks.append(Check("tsc", ["npx", "--no-install", "tsc",
                                        "--noEmit"]))
        elif _js_files(files) and _which("npx"):
            checks.append(Check("eslint", ["npx", "--no-install", "eslint",
                                           *_js_files(files)]))

    if _has(root, "Cargo.toml") and _which("cargo"):
        checks.append(Check("cargo check", ["cargo", "check",
                                            "--message-format", "short"]))
    if _has(root, "go.mod") and _which("go"):
        checks.append(Check("go vet", ["go", "vet", "./..."]))
    if (_has(root, "Gemfile") or _has(root, ".rubocop.yml")) \
            and _which("rubocop"):
        checks.append(Check("rubocop", ["rubocop", "--format", "simple"]))
    return checks


def run(root, cfg=None, files: list[str] | None = None) -> tuple[bool, str]:
    """Run every detected check. Returns (passed, report). No detected
    checker is a pass: absence of a linter must not fail a task."""
    root = Path(root)
    checks = detect(root, files)
    if not checks:
        return True, ("No linter/type checker detected (looked for ruff, "
                      "pyflakes, flake8, mypy, pyright, npm lint, tsc, cargo, go vet, "
                      "rubocop).")
    timeout = int(getattr(cfg, "diagnostics_timeout", 90) or 90)
    cap = int(getattr(cfg, "max_diagnostics_chars", 8000) or 8000)
    ok = True
    chunks: list[str] = []
    for check in checks:
        rc, out, stopped, secs = proc.run(
            check.argv, cwd=root, timeout=timeout, idle_timeout=0,
            max_output=cap)
        bad = stopped in ("timeout", "error") or (rc not in (0, None))
        if bad:
            ok = False
        status = "PASS" if not bad else (
            "TIMEOUT" if stopped == "timeout" else "FAIL")
        chunks.append(f"CHECK {check.label}: {status} ({secs:.1f}s)")
        body = (out or "").strip()
        if body:
            chunks.append(body)
        if bad and stopped == "timeout":
            chunks.append(f"(stopped after {timeout}s)")
    report = "\n".join(chunks).strip()
    if len(report) > cap:
        report = report[:cap].rstrip() + "\n[... diagnostics truncated]"
    return ok, report


def run_changed(root, cfg, changed: list[str] | None = None) -> tuple[bool, str]:
    """Convenience wrapper for the automatic post-edit check."""
    return run(root, cfg, files=list(changed or []))


# ---------------------------------------------------------------------------
# Quick per-file checks + baseline diff (the post-edit feedback loop)
# ---------------------------------------------------------------------------
# Per-file timeout: this runs inline after every applied edit, so it must stay
# far below the project-wide diagnostics_timeout.
_QUICK_TIMEOUT = 30

# "path:12:3:" / "path:12:" prefixes and bare "line 12" mentions: positions
# shift when lines are inserted above, so the baseline diff ignores them.
_POS_PREFIX_RE = re.compile(r"^\S+?:\d+(?::\d+)?:\s*")
_LINE_WORD_RE = re.compile(r"\bline \d+\b")


def _norm_finding(line: str) -> str:
    """A finding with its positions erased, for baseline comparison."""
    s = _POS_PREFIX_RE.sub("", line.strip())
    s = _LINE_WORD_RE.sub("line N", s)
    return s.strip() or line.strip()


# Names that always resolve: the builtins plus the module dunders the import
# system injects (__path__ only exists in a package's __init__, but a static
# check cannot know which file it is looking at).
_ALWAYS_DEFINED = set(vars(builtins)) | {
    "__name__", "__file__", "__doc__", "__package__", "__spec__",
    "__loader__", "__builtins__", "__debug__", "__path__",
}

# The undefined-name pass reports at most this many findings per file; a file
# over the cap is a fragment, not a lint problem, and the note says so.
_MAX_UNDEFINED = 40


class _Scope:
    """One Python namespace: module, function/lambda, class body, or the
    private scope of a comprehension."""
    __slots__ = ("bindings", "kind", "loads", "parent")

    def __init__(self, kind: str, parent: _Scope | None):
        self.kind = kind
        self.parent = parent
        self.bindings: set[str] = set()
        self.loads: list[tuple[str, int, int]] = []   # name, line, col


def _undefined_names(tree: ast.AST, rel: str) -> list[str]:
    """A mini-pyflakes limited to the highest-signal check: a name that is
    loaded but bound nowhere in its scope chain and is no builtin -- the
    NameError class `ast.parse` waves through. Only used when no real linter
    is installed.

    Deliberately conservative, because these findings gate edits and a false
    alarm costs a whole repair round: files with a star-import or an
    eval/exec are skipped outright (names can arrive dynamically there),
    augmented assignments and `del` count as bindings, argument annotations
    are never checked, and a `nonlocal` name is never reported anywhere.
    Resolution is deferred until the whole tree is walked, so forward
    references (a helper defined below its first caller) are fine. Finding
    lines keep the `rel:line:col:` prefix the linters use."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) \
                and any(a.name == "*" for a in node.names):
            return []
        if isinstance(node, ast.Name) and node.id in ("eval", "exec") \
                and isinstance(node.ctx, ast.Load):
            return []

    scopes: list[_Scope] = []
    nonlocal_pass: set[str] = set()
    module = _Scope("module", None)
    scopes.append(module)

    def _enclosing_func(sc: _Scope) -> _Scope:
        while sc is not None and sc.kind not in ("module", "function"):
            sc = sc.parent
        return sc or module

    class _Visitor(ast.NodeVisitor):
        def __init__(self):
            self.scope = module

        def _push(self, kind: str) -> _Scope:
            outer = self.scope
            self.scope = _Scope(kind, outer)
            scopes.append(self.scope)
            return outer                      # caller restores it on pop

        # --- bindings -----------------------------------------------------
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):
                self.scope.loads.append(
                    (node.id, node.lineno, node.col_offset))
            else:
                self.scope.bindings.add(node.id)      # Store and Del

        def visit_NamedExpr(self, node):
            self.visit(node.value)
            # a walrus inside a comprehension leaks into the enclosing
            # function scope (CPython semantics)
            if isinstance(node.target, ast.Name) and self.scope.kind == "comp":
                _enclosing_func(self.scope).bindings.add(node.target.id)
            self.visit(node.target)

        def visit_Global(self, node):
            module.bindings.update(node.names)

        def visit_Nonlocal(self, node):
            nonlocal_pass.update(node.names)

        def visit_Import(self, node):
            for a in node.names:
                self.scope.bindings.add(a.asname or a.name.split(".")[0])

        def visit_ImportFrom(self, node):
            for a in node.names:
                if a.name != "*":
                    self.scope.bindings.add(a.asname or a.name)

        def visit_ExceptHandler(self, node):
            if node.name:
                self.scope.bindings.add(node.name)
            self.generic_visit(node)

        def _bind_pattern(self, pattern):
            """Capture names of a match-statement pattern (py3.10+)."""
            name = getattr(pattern, "name", None)
            if isinstance(name, str):
                self.scope.bindings.add(name)
            rest = getattr(pattern, "rest", None)
            if isinstance(rest, str):
                self.scope.bindings.add(rest)
            for child in ast.iter_child_nodes(pattern):
                self._bind_pattern(child)

        def visit_Match(self, node):
            self.visit(node.subject)
            for case in node.cases:
                self._bind_pattern(case.pattern)
                self.generic_visit(case.pattern)   # value expressions load
                if case.guard:
                    self.visit(case.guard)
                for stmt in case.body:
                    self.visit(stmt)

        def _bind_type_params(self, node):
            for tp in getattr(node, "type_params", None) or []:
                name = getattr(tp, "name", None)
                if isinstance(name, str):
                    self.scope.bindings.add(name)
                self.generic_visit(tp)             # bounds/defaults load

        def _bind_args(self, args):
            for a in (*args.posonlyargs, *args.args, *args.kwonlyargs):
                self.scope.bindings.add(a.arg)
            if args.vararg:
                self.scope.bindings.add(args.vararg.arg)
            if args.kwarg:
                self.scope.bindings.add(args.kwarg.arg)

        # --- scope creators ------------------------------------------------
        def visit_FunctionDef(self, node):
            self._function(node)

        def visit_AsyncFunctionDef(self, node):
            self._function(node)

        def _function(self, node):
            self.scope.bindings.add(node.name)
            # decorators, defaults and the return annotation are evaluated in
            # the ENCLOSING scope, at def time
            for deco in node.decorator_list:
                self.visit(deco)
            for default in (*node.args.defaults,
                            *(d for d in node.args.kw_defaults if d)):
                self.visit(default)
            if node.returns:
                self.visit(node.returns)
            outer = self._push("function")
            self._bind_type_params(node)
            self._bind_args(node.args)
            for stmt in node.body:
                self.visit(stmt)
            self.scope = outer

        def visit_ClassDef(self, node):
            self.scope.bindings.add(node.name)
            for deco in node.decorator_list:
                self.visit(deco)
            for base in node.bases:
                self.visit(base)
            for kw in node.keywords:
                self.visit(kw.value)
            outer = self._push("class")
            self._bind_type_params(node)
            for stmt in node.body:
                self.visit(stmt)
            self.scope = outer

        def visit_Lambda(self, node):
            for default in (*node.args.defaults,
                            *(d for d in node.args.kw_defaults if d)):
                self.visit(default)
            outer = self._push("function")
            self._bind_args(node.args)
            self.visit(node.body)
            self.scope = outer

        def _comprehension(self, node, elts):
            outer = self._push("comp")
            for gen in node.generators:
                self.visit(gen.iter)
                self.visit(gen.target)
                for cond in gen.ifs:
                    self.visit(cond)
            for elt in elts:
                self.visit(elt)
            self.scope = outer

        def visit_ListComp(self, node):
            self._comprehension(node, [node.elt])

        def visit_SetComp(self, node):
            self._comprehension(node, [node.elt])

        def visit_GeneratorExp(self, node):
            self._comprehension(node, [node.elt])

        def visit_DictComp(self, node):
            self._comprehension(node, [node.key, node.value])

        def visit_TypeAlias(self, node):           # py3.12+ `type X = ...`
            self.visit(node.name)
            self._bind_type_params(node)
            self.visit(node.value)

    _Visitor().visit(tree)

    def _resolves(name: str, sc: _Scope) -> bool:
        if name in nonlocal_pass:
            return True
        cur: _Scope | None = sc
        while cur is not None:
            if name in cur.bindings:
                return True
            cur = cur.parent
            # a class body is not part of the lookup chain of the scopes
            # nested inside it (methods cannot see class attributes)
            while cur is not None and cur.kind == "class":
                cur = cur.parent
        return name in _ALWAYS_DEFINED

    hits: list[tuple[int, int, str]] = []
    for sc in scopes:
        for name, lineno, col in sc.loads:
            if not _resolves(name, sc):
                hits.append((lineno, col,
                             (f"{rel}:{lineno}:{col + 1}: undefined name "
                              f"'{name}'")))
    hits.sort()
    found = [h[2] for h in hits]
    if len(found) > _MAX_UNDEFINED:
        found = found[:_MAX_UNDEFINED] + [
            (f"{rel}:0: ... and {len(found) - _MAX_UNDEFINED} more undefined "
             "names -- the file looks like a fragment; re-read the original "
             "and restore the missing parts")]
    return found


def _quick_python(root: Path, rel: str) -> list[str]:
    """One Python file: the first installed linter run on just that file,
    else a plain ast.parse plus the conservative undefined-name pass.
    Finding lines keep their `rel:line:col:` prefix."""
    # --quiet suppresses ruff's "All checks passed!" banner: only diagnostics
    # belong in the result, and a clean file must produce zero findings.
    for argv in (("ruff", "check", "--quiet", "--output-format", "concise"),
                 ("pyflakes",),
                 ("flake8",)):
        if not _which(argv[0]):
            continue
        rc, out, stopped, _secs = proc.run(
            [*argv, rel], cwd=root, timeout=_QUICK_TIMEOUT,
            idle_timeout=0, max_output=20000)
        if stopped in ("timeout", "error"):
            return []          # an unusable linter must not block edits
        if rc == 0:
            return []          # exit 0 means no violations, whatever it logged
        return [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
    try:
        src = (root / rel).read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"{rel}:{e.lineno or 0}: syntax error: {e.msg} (line {e.lineno})"]
    except (OSError, ValueError):
        return []
    return _undefined_names(tree, rel)


def _quick_js(root: Path, rel: str) -> list[str]:
    """One JS/TS file: the tree-sitter syntax check used by edits._validate."""
    from .chunker_ts import LANG_BY_EXT, TS_AVAILABLE, has_syntax_error
    lang = LANG_BY_EXT.get(Path(rel).suffix.lower())
    if not lang or not TS_AVAILABLE:
        return []
    try:
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    bad, lineno = has_syntax_error(text, lang)
    if bad:
        return [(f"{rel}:{lineno or 0}: syntax error (parse failed, "
                 f"line {lineno})")]
    return []


def quick_findings(root, files: list[str]) -> dict[str, list[str]]:
    """Cheap per-file syntax/lint findings as {rel: [finding lines]}.

    Python files go to the first installed linter (ruff, pyflakes, flake8)
    scoped to that one file, falling back to ast.parse plus a conservative
    undefined-name pass; JS/TS files get the
    tree-sitter syntax check. Other languages have no cheap per-file checker
    and are skipped. Never raises; an unreadable file maps to no findings."""
    root = Path(root)
    out: dict[str, list[str]] = {}
    for rel in files or []:
        rel = str(rel).strip()
        if not rel:
            continue
        suffix = Path(rel).suffix.lower()
        try:
            if suffix in (".py", ".pyi"):
                out[rel] = _quick_python(root, rel)
            elif suffix in (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"):
                out[rel] = _quick_js(root, rel)
        except Exception:
            out[rel] = []
    return out


def new_findings(baseline: dict[str, list[str]],
                 current: dict[str, list[str]]) -> dict[str, list[str]]:
    """Findings in `current` whose normalised text is absent from `baseline`
    (per file): only problems the edits introduced, never pre-existing ones
    whose line numbers shifted."""
    out: dict[str, list[str]] = {}
    for rel, items in current.items():
        old = {_norm_finding(x) for x in baseline.get(rel, [])}
        extra = [x for x in items if _norm_finding(x) not in old]
        if extra:
            out[rel] = extra
    return out


def render_findings(found: dict[str, list[str]], cap: int = 4000) -> str:
    """The finding lines as feedback text, bounded."""
    lines = [it for items in found.values() for it in items]
    text = "\n".join(lines)
    if len(text) > cap:
        text = text[:cap].rstrip() + "\n[... more findings truncated]"
    return text
