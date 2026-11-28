"""Deterministic codemod recipes with preview and transactional apply.

A codemod is a small, provable source transformation -- adding an import,
turning ``print`` calls into logger calls, extracting a literal into a
constant, adding an annotation. Unlike a model-authored edit block, the
transformation is computed from Python's real AST and tokenizer, so comments,
string literals and unrelated formatting are never touched by accident.

    from pathlib import Path
    from pedacito.codemod import CodemodRequest, list_recipes, preview, apply

    req = CodemodRequest("add_import", "app.py", "",
                         {"module": "typing", "name": "Any"})
    result = preview(Path("."), req)       # unified diff, no writes
    result = apply(Path("."), req)         # one atomic transaction

``preview`` never writes. ``apply`` re-runs the same preparation, snapshots
every touched file through ``edits.apply_edits_atomic`` (durable preimages and
just-before-commit hash verification), and returns the same diff shape.

Safety rules:

* only languages a recipe explicitly declares are accepted; anything else is
  refused with an actionable warning and no writes;
* a symbol or binding must resolve uniquely -- ambiguity refuses rather than
  guessing at the wrong definition;
* the output is parsed again before it can be applied, so a recipe can never
  land syntactically broken source;
* paths are resolved through :func:`security.resolve_safe_path`, so traversal
  and symlink/junction escapes are refused, and protected/sensitive files are
  never read or written;
* file encoding, BOM and line endings are preserved by ``fileio``.

The catalog is the shared registry WP-4102B extends: each entry documents its
option names, supported languages and implementation status. Recipes that are
catalogued but not implemented in this release return ``ok=False`` with a
"not implemented" warning and never write.
"""

from __future__ import annotations

import ast
import copy
import difflib
import hashlib
import io
import keyword
import tokenize
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import fileio, safety, security
from .edits import Edit, apply_edits_atomic, syntax_error

__all__ = ["CodemodRequest", "CodemodResult", "apply", "list_recipes",
           "preview"]

# A source file larger than this is refused: a codemod preview must stay
# bounded, and a multi-megabyte source is far outside a surgical refactor.
_MAX_SOURCE_BYTES = 2_000_000

# Recipe language identifiers are ASCII protocol values, never localized.
_LANG_PYTHON = "python"


@dataclass(frozen=True)
class CodemodRequest:
    """One codemod operation request.

    ``file`` is project-relative (POSIX or native separators); ``symbol`` is a
    recipe-specific target and may be empty. ``options`` values are strings and
    only names documented in :func:`list_recipes` metadata are accepted.
    """

    recipe: str
    file: str
    symbol: str
    options: dict[str, str]


@dataclass(frozen=True)
class CodemodResult:
    """Outcome of a codemod preview or apply.

    ``diffs`` maps project-relative POSIX paths to deterministic unified diffs;
    ``changed`` is sorted and only names files whose bytes would change. On a
    refusal ``ok`` is False, ``diffs`` and ``changed`` are empty, and ``error``
    says why. ``warnings`` are non-fatal, actionable notices.
    """

    ok: bool
    diffs: dict[str, str]
    changed: tuple[str, ...]
    warnings: tuple[str, ...]
    error: str = ""


class _Refuse(Exception):
    """Internal refusal carrying a user-actionable, bounded message."""


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
def _opt(name: str, description: str, *, required: bool = False,
         default: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "name": name,
        "required": required,
        "description": description,
    }
    if default is not None:
        item["default"] = default
    return item


def _recipe(recipe_id: str, title: str, description: str, *,
            implemented: bool, symbol: str, options: list[dict[str, Any]],
            languages: tuple[str, ...] = (_LANG_PYTHON,)) -> dict[str, Any]:
    return {
        "id": recipe_id,
        "title": title,
        "description": description,
        "status": "implemented" if implemented else "unverified",
        "implemented": implemented,
        "languages": list(languages),
        "symbol": symbol,
        "options": options,
        "option_names": [o["name"] for o in options],
        "options_doc": {o["name"]: o["description"] for o in options},
    }


# The exact initial catalog from the v0.4.1 plan, in its documented order.
# WP-4102B adds the remaining eight handlers to _HANDLERS; the metadata keeps
# their option names stable for clients and help text.
_RECIPES: tuple[dict[str, Any], ...] = (
    _recipe(
        "add_param", "Add a parameter",
        "Add a parameter to a uniquely resolved function and provable callers.",
        implemented=False, symbol="required",
        options=[
            _opt("name", "Parameter name.", required=True),
            _opt("annotation", "Optional annotation source."),
            _opt("default", "Optional default-value source."),
        ],
    ),
    _recipe(
        "remove_param", "Remove a parameter",
        "Remove a parameter and update only provable call sites.",
        implemented=False, symbol="required",
        options=[_opt("name", "Parameter name.", required=True)],
    ),
    _recipe(
        "add_import", "Add an import",
        "Add a missing Python import after the existing import block.",
        implemented=True, symbol="ignored",
        options=[
            _opt("module", "Module to import, e.g. `os` or `pkg.mod`.",
                 required=True),
            _opt("name", "Name to import from the module; empty means "
                        "`import module`."),
            _opt("alias", "Optional local alias for the imported module or "
                          "name."),
        ],
    ),
    _recipe(
        "remove_import", "Remove an import",
        "Remove an unused import, refusing when its binding is still used.",
        implemented=True, symbol="ignored",
        options=[
            _opt("module", "Module to remove, e.g. `os` or `pkg.mod`.",
                 required=True),
            _opt("name", "Imported name to remove from a from-import; empty "
                        "removes `import module`."),
            _opt("alias", "Optional alias used to disambiguate the binding."),
        ],
    ),
    _recipe(
        "wrap_try", "Wrap in try/except",
        "Wrap a uniquely resolved statement or symbol in try/except.",
        implemented=False, symbol="required",
        options=[
            _opt("exception", "Exception type expression to catch.",
                 default="Exception"),
        ],
    ),
    _recipe(
        "wrap_context", "Wrap in a context manager",
        "Wrap a uniquely resolved statement or symbol in a with-block.",
        implemented=False, symbol="required",
        options=[_opt("manager", "Context-manager expression.", required=True)],
    ),
    _recipe(
        "print_to_logger", "Replace print with logging",
        "Convert print() calls in a scope to logger calls, preserving "
        "arguments and comments.",
        implemented=True, symbol="optional",
        options=[
            _opt("logger", "Logger variable name to call.",
                 default="logger"),
            _opt("level", "Log level: debug, info, warning, error or critical.",
                 default="info"),
            _opt("create_logger", "Create a module logger when the logger "
                                 "name is unbound (true/false).",
                 default="true"),
        ],
    ),
    _recipe(
        "extract_constant", "Extract a literal constant",
        "Replace matching literal occurrences in a scope with a module-level "
        "constant.",
        implemented=True, symbol="optional",
        options=[
            _opt("name", "New module-level constant name.", required=True),
            _opt("value", "Literal expression to extract, e.g. `30` or "
                         "`\"timeout\"`.", required=True),
        ],
    ),
    _recipe(
        "add_annotation", "Add an annotation",
        "Add a missing parameter, return or variable annotation.",
        implemented=True, symbol="required",
        options=[
            _opt("annotation", "Annotation expression, e.g. `int`.",
                 required=True),
            _opt("target", "Target kind: auto, param, return or variable.",
                 default="auto"),
            _opt("name", "Parameter or variable name when the symbol names "
                        "a function, class or scope."),
            _opt("scope", "Optional function or class scope for a variable."),
            _opt("overwrite", "Replace a different existing annotation "
                             "(true/false).", default="false"),
        ],
    ),
    _recipe(
        "move_symbol", "Move a symbol",
        "Move a uniquely resolved symbol and update resolved imports.",
        implemented=False, symbol="required",
        options=[_opt("to_file", "Destination project-relative path.",
                      required=True)],
    ),
    _recipe(
        "move_file", "Move a file",
        "Move a source file and update resolved imports.",
        implemented=False, symbol="ignored",
        options=[_opt("to_path", "Destination project-relative path.",
                      required=True)],
    ),
    _recipe(
        "replace_deprecated", "Replace a deprecated symbol",
        "Replace a deprecated API at resolved references.",
        implemented=False, symbol="required",
        options=[
            _opt("old", "Deprecated name.", required=True),
            _opt("new", "Replacement name.", required=True),
        ],
    ),
    _recipe(
        "add_reexport", "Add a re-export",
        "Re-export a name from a module without touching callers.",
        implemented=False, symbol="ignored",
        options=[
            _opt("name", "Name to re-export.", required=True),
            _opt("from_module", "Module that defines the name.", required=True),
        ],
    ),
)

_RECIPE_BY_ID = {r["id"]: r for r in _RECIPES}


def list_recipes() -> list[dict[str, Any]]:
    """The codemod catalog, in stable documented order.

    Every entry names its ``id``, supported ``languages``, ``symbol`` role,
    implementation ``status`` (``implemented`` or ``unverified``) and its
    ``options`` with explicit names, required flags and descriptions. The
    result is a deep copy: callers may mutate it without touching the registry.
    """
    return copy.deepcopy(list(_RECIPES))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def preview(root: Path, request: CodemodRequest, *, index=None,
            cfg=None) -> CodemodResult:
    """Compute a codemod result and unified diff without writing anything.

    ``index`` is accepted for the extension recipes WP-4102B adds; the
    single-file recipes here derive every fact from the source itself.
    """
    try:
        plan = _prepare(root, request, cfg=cfg, for_write=False, index=index)
    except _Refuse as exc:
        return CodemodResult(False, {}, (), (), str(exc))
    return plan.result


def apply(root: Path, request: CodemodRequest, *, index=None,
          cfg=None) -> CodemodResult:
    """Apply a codemod through one all-or-nothing atomic transaction.

    The preview is recomputed from current bytes, the file's preimage hash is
    verified again by ``apply_edits_atomic`` immediately before commit, and a
    policy/protection/sensitive-path failure leaves every file untouched.
    """
    allow = getattr(cfg, "allow_write", None) if cfg is not None else None
    try:
        plan = _prepare(root, request, cfg=cfg, for_write=True, index=index)
    except _Refuse as exc:
        return CodemodResult(False, {}, (), (), str(exc))
    result = plan.result
    if not result.ok or not result.changed:
        return result
    if allow is False:
        return CodemodResult(
            False, {}, (), result.warnings,
            "apply requires allow_write = true; use preview to inspect the "
            "change first")

    backup = bool(getattr(cfg, "backup", False)) if cfg is not None else False
    backup_dir = getattr(cfg, "_backup_dir", None) if cfg is not None else None
    if backup_dir is not None:
        backup_dir = Path(backup_dir)
    if backup and backup_dir is None:
        try:
            from . import workspace
            backup_dir = workspace.new_backup_dir(plan.root)
            try:
                cfg._backup_dir = backup_dir
            except AttributeError:
                pass
        except OSError as exc:
            return CodemodResult(
                False, {}, (), result.warnings,
                f"refused to apply: could not create the snapshot directory "
                f"({exc}); nothing was written")

    edit = Edit(plan.rel, plan.before, plan.after, exact=True)
    try:
        outcomes = apply_edits_atomic(
            plan.root, [edit], backup=backup, backup_dir=backup_dir, cfg=cfg,
            expected_sha256={plan.rel: plan.raw_sha})
    except OSError as exc:
        return CodemodResult(
            False, {}, (), result.warnings,
            f"apply failed before writing: {exc}")
    failures = [r for r in outcomes if not r.ok]
    if failures:
        message = "; ".join(r.message for r in failures if r.message)
        return CodemodResult(False, {}, (), result.warnings,
                             message or "codemod apply was refused")
    return result


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Plan:
    root: Path
    rel: str
    before: str
    after: str
    raw_sha: str
    result: CodemodResult


def _prepare(root: Path, request: CodemodRequest, *, cfg, for_write: bool,
             index=None) -> _Plan:
    del index                         # reserved for cross-file recipes
    if not isinstance(request, CodemodRequest):
        raise _Refuse("codemod request must be a CodemodRequest")
    recipe_id = str(request.recipe or "").strip()
    if not recipe_id:
        raise _Refuse("codemod recipe is required; use list_recipes() for ids")
    recipe = _RECIPE_BY_ID.get(recipe_id)
    if recipe is None:
        raise _Refuse(f"unknown codemod recipe '{recipe_id}'; known recipes: "
                      + ", ".join(r["id"] for r in _RECIPES))
    handler = _HANDLERS.get(recipe_id)
    if handler is None:
        raise _Refuse(
            f"recipe '{recipe_id}' is catalogued but not implemented in this "
            "release (unverified); nothing was written")
    options = _validate_options(recipe, request.options)

    loaded = _load(root, request.file, request.symbol, cfg=cfg,
                   for_write=for_write)
    if loaded.suffix not in (".py", ".pyi"):
        raise _Refuse(
            f"recipe '{recipe_id}' supports only Python in this release; "
            f"'{loaded.rel}' has unsupported extension "
            f"'{loaded.suffix or '(none)'}' (language support is unverified) "
            "-- nothing was written")
    if loaded.text.strip() == "":
        tree = ast.parse("", filename=loaded.rel)
    else:
        tree = _parse_python(loaded.text, loaded.rel)
    source = _Source(loaded.text, tree, loaded.rel)

    outcome = handler(source, str(request.symbol or "").strip(), options)
    warnings = list(outcome.warnings)
    if outcome.text == loaded.text:
        warnings.append("no change: the requested transformation is already "
                        "present")
        return _Plan(
            loaded.root, loaded.rel, loaded.text, loaded.text, loaded.raw_sha,
            CodemodResult(True, {}, (), tuple(warnings)))

    bad = syntax_error(loaded.path, outcome.text)
    if bad:
        raise _Refuse(
            f"refused: the codemod would leave {loaded.rel} with {bad}; "
            "nothing was written")

    diff = _unified_diff(loaded.rel, loaded.text, outcome.text)
    result = CodemodResult(
        True, {loaded.rel: diff}, (loaded.rel,), tuple(warnings))
    return _Plan(loaded.root, loaded.rel, loaded.text, outcome.text,
                 loaded.raw_sha, result)


def _validate_options(recipe: Mapping[str, Any],
                      options: Mapping[str, str] | None) -> dict[str, str]:
    if options is not None and not isinstance(options, Mapping):
        raise _Refuse("codemod options must be a mapping of string values")
    given = dict(options or {})
    allowed = {o["name"]: o for o in recipe["options"]}
    unknown = sorted(set(given) - set(allowed))
    if unknown:
        raise _Refuse(
            f"unknown option(s) for '{recipe['id']}': "
            + ", ".join(unknown) + "; known options: "
            + (", ".join(allowed) or "(none)"))
    out: dict[str, str] = {}
    for name, meta in allowed.items():
        raw = given.get(name)
        if raw is None:
            continue
        if not isinstance(raw, str):
            raise _Refuse(f"option '{name}' must be a string")
        if len(raw) > 2000:
            raise _Refuse(
                f"option '{name}' is longer than the 2000-character codemod "
                "option limit")
        out[name] = raw.strip()
    missing = [name for name, meta in allowed.items()
               if meta.get("required") and not out.get(name)]
    if missing:
        raise _Refuse(
            f"recipe '{recipe['id']}' requires option(s): "
            + ", ".join(missing))
    return out


@dataclass(frozen=True)
class _Loaded:
    root: Path
    path: Path
    rel: str
    suffix: str
    text: str
    raw_sha: str


def _load(root: Path, file: str, symbol: str, *, cfg,
          for_write: bool) -> _Loaded:
    del symbol                         # recipe handlers validate their own
    try:
        root_path = Path(root).expanduser()
        root_resolved = root_path.resolve()
    except OSError as exc:
        raise _Refuse(f"project root could not be resolved: {root} ({exc})")
    if not root_resolved.is_dir():
        raise _Refuse(f"project root is not a directory: {root}")
    raw = fileio.normalise_rel(file)
    if not raw:
        raise _Refuse("codemod request needs a project-relative file path")
    try:
        path = security.resolve_safe_path(root_resolved, raw)
    except security.SecurityError as exc:
        raise _Refuse(f"refused: {exc}; nothing was written")
    try:
        rel = path.relative_to(root_resolved).as_posix()
    except ValueError:
        raise _Refuse(
            f"refused: path escapes the project root: {file}; nothing was "
            "written")
    why = safety.is_protected(rel, cfg, write=for_write)
    if why:
        raise _Refuse(f"refused: {why}; nothing was written")
    if security.is_sensitive_path(rel):
        raise _Refuse(
            f"refused: {rel} is a sensitive file (secrets/credentials) and "
            "is never read or rewritten by a codemod")
    if not path.is_file():
        raise _Refuse(
            f"refused: {rel} is not a regular file in the project; nothing "
            "was written")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise _Refuse(f"could not stat {rel}: {exc}")
    if size > _MAX_SOURCE_BYTES:
        raise _Refuse(
            f"refused: {rel} is {size} bytes, above the "
            f"{_MAX_SOURCE_BYTES}-byte codemod limit; nothing was written")
    _enforce_read_policy(cfg, rel, root_resolved)
    try:
        raw_bytes = path.read_bytes()
        text, _style = fileio.read(path, errors="strict")
    except UnicodeDecodeError:
        raise _Refuse(f"refused: {rel} is not valid UTF-8 text; nothing was "
                      "written")
    except OSError as exc:
        raise _Refuse(f"could not read {rel}: {exc}")
    return _Loaded(root_resolved, path, rel, path.suffix.lower(), text,
                   hashlib.sha256(raw_bytes).hexdigest())


def _enforce_read_policy(cfg, rel: str, root: Path) -> None:
    if cfg is None:
        return
    try:
        from . import policy as policy_mod
        decision = policy_mod.enforce(cfg, "read", rel, root=root)
    except Exception as exc:                       # fail closed
        raise _Refuse(
            f"denied by policy: policy evaluation error: {exc}; nothing was "
            "written")
    if decision is not None and decision.decision == "deny":
        raise _Refuse(
            f"denied by policy ({decision.matched_rule}): "
            f"{decision.reason}; nothing was written")


def _parse_python(text: str, rel: str) -> ast.Module:
    try:
        return ast.parse(text, filename=rel)
    except SyntaxError as exc:
        where = f"line {exc.lineno}" if exc.lineno else "an unknown line"
        raise _Refuse(
            f"refused: {rel} does not parse ({where}: {exc.msg}); nothing "
            "was written")
    except ValueError as exc:
        raise _Refuse(f"refused: {rel} could not be parsed: {exc}")


def _unified_diff(rel: str, before: str, after: str) -> str:
    return "".join(difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        f"a/{rel}", f"b/{rel}", n=3))


# ---------------------------------------------------------------------------
# Source model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Token:
    type: int
    string: str
    start: int
    end: int


@dataclass(frozen=True)
class _Source:
    text: str
    tree: ast.Module
    rel: str

    @property
    def lines(self) -> list[str]:
        return self.text.splitlines(keepends=True)

    @property
    def starts(self) -> list[int]:
        out: list[int] = []
        pos = 0
        for line in self.lines:
            out.append(pos)
            pos += len(line)
        return out

    def line_start(self, lineno: int) -> int:
        if lineno < 1:
            raise _Refuse(f"invalid line number {lineno} in {self.rel}")
        starts = self.starts
        if lineno > len(starts):
            return len(self.text)
        return starts[lineno - 1]

    def line_end(self, lineno: int) -> int:
        starts = self.starts
        if 1 <= lineno < len(starts):
            return starts[lineno]
        return len(self.text)

    def offset(self, lineno: int, byte_col: int) -> int:
        lines = self.lines
        if not 1 <= lineno <= len(lines):
            raise _Refuse(f"invalid AST position {lineno} in {self.rel}")
        line = lines[lineno - 1]
        data = line.encode("utf-8")
        if byte_col < 0:
            byte_col = 0
        if byte_col > len(data):
            byte_col = len(data)
        col = len(data[:byte_col].decode("utf-8", "ignore"))
        return self.line_start(lineno) + col

    def span(self, node: ast.AST) -> tuple[int, int]:
        start_line = getattr(node, "lineno", None)
        end_line = getattr(node, "end_lineno", None)
        start_col = getattr(node, "col_offset", None)
        end_col = getattr(node, "end_col_offset", None)
        if None in (start_line, end_line, start_col, end_col):
            raise _Refuse(
                f"refused: {self.rel} has a node without source positions; "
                "nothing was written")
        return (self.offset(start_line, start_col),
                self.offset(end_line, end_col))

    def seg(self, node: ast.AST) -> str:
        start, end = self.span(node)
        return self.text[start:end]

    def tokens(self) -> list[_Token]:
        out: list[_Token] = []
        try:
            for tok in tokenize.generate_tokens(io.StringIO(self.text).readline):
                if tok.type in (tokenize.ENCODING, tokenize.ENDMARKER,
                                tokenize.NL, tokenize.NEWLINE,
                                tokenize.INDENT, tokenize.DEDENT,
                                tokenize.COMMENT):
                    continue
                out.append(_Token(tok.type, tok.string,
                                  self.line_start(tok.start[0]) + tok.start[1],
                                  self.line_start(tok.end[0]) + tok.end[1]))
        except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
            raise _Refuse(
                f"refused: could not tokenize {self.rel} ({exc}); nothing was "
                "written")
        return out


@dataclass(frozen=True)
class _Outcome:
    text: str
    warnings: tuple[str, ...] = ()


def _warn(*items: str) -> tuple[str, ...]:
    return tuple(i for i in items if i)


# ---------------------------------------------------------------------------
# Python structure helpers
# ---------------------------------------------------------------------------
def _is_docstring_stmt(stmt: ast.stmt) -> bool:
    return (isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str))


def _module_docstring(tree: ast.Module) -> ast.stmt | None:
    if tree.body and _is_docstring_stmt(tree.body[0]):
        return tree.body[0]
    return None


def _statement_start_line(stmt: ast.stmt) -> int:
    """First source line of a statement, including its decorators."""
    line = int(getattr(stmt, "lineno", 1) or 1)
    for dec in getattr(stmt, "decorator_list", ()) or ():
        line = min(line, int(getattr(dec, "lineno", line) or line))
    return line


def _leading_imports(tree: ast.Module) -> list[ast.stmt]:
    body = list(tree.body)
    if body and _is_docstring_stmt(body[0]):
        body = body[1:]
    out: list[ast.stmt] = []
    for stmt in body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            out.append(stmt)
        else:
            break
    return out


def _top_anchor(source: _Source) -> tuple[int, str]:
    """Where a new top-level import/constant belongs.

    Returns ``(offset, kind)``. Imports and constants go after the leading
    import block, after a module docstring, immediately before the first
    statement (after any header comments/shebang), or at EOF.
    """
    body = list(source.tree.body)
    doc = _module_docstring(source.tree)
    imports = _leading_imports(source.tree)
    if imports:
        return source.line_end(imports[-1].end_lineno or imports[-1].lineno), \
            "after_imports"
    if doc is not None:
        return source.line_end(doc.end_lineno or doc.lineno), "after_docstring"
    body = [s for s in body if s is not doc]
    if body:
        return source.line_start(_statement_start_line(body[0])), \
            "before_first"
    return len(source.text), "eof"


def _insert_top(source: _Source, lines: list[str], *, purpose: str) -> str:
    """Insert top-level lines with deliberate, minimal blank-line separation."""
    offset, kind = _top_anchor(source)
    before, after = source.text[:offset], source.text[offset:]
    block = "\n".join(lines) + "\n"
    if kind in ("after_imports", "after_docstring"):
        if before and not before.endswith("\n"):
            block = "\n" + block
        if purpose == "constant" or kind == "after_docstring":
            block = "\n" + block
        if after and not after.startswith("\n"):
            block += "\n"
    elif kind == "before_first":
        if after and not after.startswith("\n"):
            block += "\n"
    elif kind == "eof":
        if before and not before.endswith("\n"):
            block = "\n" + block
    return before + block + after


def _scope_bindings(stmts: list[ast.stmt]) -> dict[str, str]:
    """Names bound by a statement list, mapped to a short description.

    Nested function/class bodies are not traversed (their locals cannot shadow
    a binding in this scope), but control-flow blocks are: a name bound in an
    ``if``/``try`` is still a binding of the enclosing scope.
    """
    out: dict[str, str] = {}

    def bind(name: str, what: str) -> None:
        out.setdefault(name, what)

    def targets(node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            bind(node.id, "assignment")
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                targets(elt)
        elif isinstance(node, ast.Starred):
            targets(node.value)

    def visit(items: list[ast.stmt]) -> None:
        for stmt in items:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bind(stmt.name, "function definition")
                continue
            if isinstance(stmt, ast.ClassDef):
                bind(stmt.name, "class definition")
                continue
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    bind(alias.asname or alias.name.split(".")[0], "import")
                continue
            if isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    if alias.name == "*":
                        bind("*", "star import")
                    else:
                        bind(alias.asname or alias.name, "import")
                continue
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    targets(target)
            elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
                targets(stmt.target)
            elif isinstance(stmt, (ast.For, ast.AsyncFor)):
                targets(stmt.target)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                for item in stmt.items:
                    if item.optional_vars is not None:
                        targets(item.optional_vars)
            for named in ast.walk(stmt):
                if isinstance(named, ast.NamedExpr):
                    targets(named.target)
            for child in ast.iter_child_nodes(stmt):
                if isinstance(child, ast.stmt) and not isinstance(
                        child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                ast.ClassDef)):
                    visit([child])

    visit(stmts)
    return out


def _module_scope_bindings(tree: ast.Module) -> dict[str, str]:
    """Names bound at module scope, mapped to a short description."""
    return _scope_bindings(tree.body)


def _all_bindings(tree: ast.Module) -> dict[str, str]:
    """Every binding name anywhere in the module, for shadowing checks."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            out.setdefault(node.name, "definition")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in _function_args(node):
                    out.setdefault(arg.arg, "parameter")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out.setdefault(alias.asname or alias.name.split(".")[0],
                               "import")
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    out.setdefault(alias.asname or alias.name, "import")
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            out.setdefault(node.id, "assignment")
        elif isinstance(node, ast.arg):
            out.setdefault(node.arg, "parameter")
        elif isinstance(node, ast.ExceptHandler) and node.name:
            out.setdefault(node.name, "exception binding")
    return out


def _function_args(node: ast.FunctionDef | ast.AsyncFunctionDef
                   ) -> list[ast.arg]:
    args = node.args
    out: list[ast.arg] = []
    out.extend(args.posonlyargs)
    out.extend(args.args)
    if args.vararg is not None:
        out.append(args.vararg)
    out.extend(args.kwonlyargs)
    if args.kwarg is not None:
        out.append(args.kwarg)
    return out


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _definitions(tree: ast.Module) -> list[tuple[str, ast.AST]]:
    parents = _parent_map(tree)

    def qualname(node: ast.AST) -> str:
        parts: list[str] = []
        cur = node
        while cur is not tree:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef,
                                ast.ClassDef)):
                parts.append(cur.name)
            cur = parents.get(cur)
            if cur is None:
                break
        return ".".join(reversed(parts))

    out: list[tuple[str, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            out.append((qualname(node), node))
    return out


def _resolve_definition(tree: ast.Module, symbol: str) -> ast.AST:
    wanted = symbol.strip()
    if not wanted:
        raise _Refuse("this recipe needs a symbol naming a definition")
    matches = [(q, n) for q, n in _definitions(tree) if q == wanted]
    if not matches:
        known = sorted({q for q, _n in _definitions(tree)})[:8]
        raise _Refuse(
            f"symbol '{wanted}' was not found as a unique definition"
            + (f"; known definitions: {', '.join(known)}" if known else "")
            + "; nothing was written")
    if len(matches) > 1:
        where = ", ".join(
            f"line {getattr(n, 'lineno', '?')}" for _q, n in matches[:6])
        raise _Refuse(
            f"symbol '{wanted}' is ambiguous ({len(matches)} definitions: "
            f"{where}); refuse to guess which one to edit")
    return matches[0][1]


def _matching_scope(tree: ast.Module, symbol: str) -> ast.AST | None:
    if not symbol.strip():
        return tree
    matches = [(q, n) for q, n in _definitions(tree) if q == symbol.strip()]
    if len(matches) == 1:
        return matches[0][1]
    if len(matches) > 1:
        raise _Refuse(
            f"symbol '{symbol}' is ambiguous ({len(matches)} definitions); "
            "refuse to guess which scope to edit")
    return None


def _identifier(value: str, what: str) -> str:
    name = (value or "").strip()
    if not name or not name.isidentifier() or keyword.iskeyword(name):
        raise _Refuse(f"{what} '{name}' is not a valid Python identifier")
    return name


def _parse_annotation(value: str) -> str:
    text = (value or "").strip()
    if not text:
        raise _Refuse("annotation option is required")
    if "\n" in text or "\r" in text:
        raise _Refuse("annotation must be a single-line expression")
    try:
        ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise _Refuse(
            f"annotation '{text}' is not a valid expression: {exc.msg}")
    return text


def _same_expr(left: ast.AST, right: ast.AST) -> bool:
    try:
        return ast.dump(left, include_attributes=False) == ast.dump(
            right, include_attributes=False)
    except Exception:
        return False


def _annotation_equal(existing: ast.AST, new_text: str) -> bool:
    try:
        new = ast.parse(new_text, mode="eval").body
    except SyntaxError:
        return False
    return _same_expr(existing, new)


def _bool_option(options: Mapping[str, str], name: str, default: bool) -> bool:
    raw = options.get(name)
    if raw is None or raw == "":
        return default
    low = raw.strip().lower()
    if low in ("1", "true", "yes", "on"):
        return True
    if low in ("0", "false", "no", "off"):
        return False
    raise _Refuse(f"option '{name}' must be true or false, not '{raw}'")


def _literal_value(value: str) -> ast.AST:
    text = (value or "").strip()
    if not text:
        raise _Refuse("value option is required")
    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError as exc:
        raise _Refuse(
            f"value option is not a valid expression: {exc.msg}")
    if not _is_extractable_literal(node):
        raise _Refuse(
            "value option is not a literal constant this recipe can extract "
            "(use a string, number, bytes, bool, None or a signed number)")
    return node


def _is_extractable_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)):
        return (isinstance(node.operand, ast.Constant)
                and isinstance(node.operand.value, (int, float, complex)))
    return False


def _literal_key(node: ast.AST) -> tuple:
    if isinstance(node, ast.Constant):
        value = node.value
        return ("const", type(value).__name__, repr(value))
    if isinstance(node, ast.UnaryOp) and _is_extractable_literal(node):
        operand = node.operand
        sign = "-" if isinstance(node.op, ast.USub) else "+"
        return ("unary", sign, type(operand.value).__name__,
                repr(operand.value))
    return ("other",)


def _is_docstring_constant(node: ast.AST, parents: Mapping[ast.AST, ast.AST]
                           ) -> bool:
    parent = parents.get(node)
    if not isinstance(parent, ast.Expr):
        return False
    grand = parents.get(parent)
    if not isinstance(grand, (ast.Module, ast.ClassDef, ast.FunctionDef,
                              ast.AsyncFunctionDef)):
        return False
    body = getattr(grand, "body", [])
    return bool(body) and body[0] is parent


def _annotation_spans(tree: ast.Module) -> list[ast.AST]:
    roots: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.annotation is not None:
            roots.append(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.returns is not None:
            roots.append(node.returns)
        elif isinstance(node, ast.AnnAssign) and node.annotation is not None:
            roots.append(node.annotation)
    return roots  # resolved to offsets by the caller


def _reexport_roots(tree: ast.Module) -> list[ast.AST]:
    """Expressions whose string literals are API names, not values.

    Extracting a literal from ``__all__`` would replace an exported name with
    a constant NAME (a semantic break), so every such expression is skipped.
    """
    roots: list[ast.AST] = []
    for stmt in ast.walk(tree):
        if isinstance(stmt, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "__all__"
                   for t in stmt.targets):
                roots.append(stmt.value)
        elif isinstance(stmt, ast.AugAssign):
            if isinstance(stmt.target, ast.Name) \
                    and stmt.target.id == "__all__":
                roots.append(stmt.value)
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            if (isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "__all__"
                    and func.attr in ("append", "extend", "insert")):
                roots.extend(stmt.value.args)
    return roots


def _inside_span(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= pos < end for start, end in spans)


# ---------------------------------------------------------------------------
# add_import
# ---------------------------------------------------------------------------
def _module_text(node: ast.ImportFrom) -> str:
    return "." * int(node.level or 0) + (node.module or "")


def _validate_import_line(line: str, *, module: str, name: str,
                          alias: str) -> None:
    try:
        parsed = ast.parse(line)
    except SyntaxError as exc:
        raise _Refuse(
            f"refused: '{line.strip()}' is not a valid import statement "
            f"({exc.msg})")
    if len(parsed.body) != 1:
        raise _Refuse(
            f"refused: '{line.strip()}' is not a single import statement")
    stmt = parsed.body[0]
    if name:
        if not isinstance(stmt, ast.ImportFrom):
            raise _Refuse("refused: malformed from-import option values")
        if len(stmt.names) != 1:
            raise _Refuse("refused: malformed from-import option values")
        got = stmt.names[0]
        if (_module_text(stmt) != module or got.name != name
                or (got.asname or "") != alias):
            raise _Refuse(
                "refused: module, name and alias options do not form the "
                "requested import")
    else:
        if not isinstance(stmt, ast.Import) or len(stmt.names) != 1:
            raise _Refuse("refused: malformed import option values")
        got = stmt.names[0]
        if got.name != module or (got.asname or "") != alias:
            raise _Refuse(
                "refused: module and alias options do not form the requested "
                "import")


def _import_line(module: str, name: str, alias: str) -> tuple[str, str]:
    module = (module or "").strip()
    name = (name or "").strip()
    alias = (alias or "").strip()
    if not module:
        raise _Refuse("option 'module' is required and may not be empty")
    if alias:
        _identifier(alias, "alias")
    if name:
        if name == "*":
            raise _Refuse(
                "refused: star imports are ambiguous and are never added by a "
                "codemod; name the symbol explicitly")
        _identifier(name, "imported name")
        line = f"from {module} import {name}"
        if alias:
            line += f" as {alias}"
        local = alias or name
    else:
        if any(part and not part.isidentifier() for part in module.split(".")):
            raise _Refuse(
                f"refused: module '{module}' is not a dotted module name")
        line = f"import {module}"
        if alias:
            line += f" as {alias}"
        local = alias or module.split(".")[0]
    _validate_import_line(line, module=module, name=name, alias=alias)
    return line, local


def _existing_import(tree: ast.Module, module: str, name: str,
                     alias: str) -> tuple[bool, str]:
    """(already_present, error) for an exact requested import."""
    for node in ast.walk(tree):
        if name and isinstance(node, ast.ImportFrom):
            if _module_text(node) != module:
                continue
            for got in node.names:
                if got.name == name and (got.asname or "") == alias:
                    return True, ""
                local = got.asname or got.name
                wanted_local = alias or name
                if local == wanted_local and (got.name != name
                                              or (got.asname or "") != alias):
                    return False, (
                        f"refused: local name '{local}' is already imported "
                        f"as '{got.name}' from '{_module_text(node)}'")
        elif not name and isinstance(node, ast.Import):
            for got in node.names:
                if got.name == module and (got.asname or "") == alias:
                    return True, ""
                local = got.asname or got.name.split(".")[0]
                wanted_local = alias or module.split(".")[0]
                if local == wanted_local and (got.name != module
                                              or (got.asname or "") != alias):
                    return False, (
                        f"refused: local name '{local}' is already bound by "
                        f"'import {got.name}'")
    return False, ""


def _recipe_add_import(source: _Source, _symbol: str,
                       options: Mapping[str, str]) -> _Outcome:
    line, local = _import_line(options.get("module", ""),
                               options.get("name", ""),
                               options.get("alias", ""))
    module = options.get("module", "").strip()
    name = options.get("name", "").strip()
    alias = options.get("alias", "").strip()
    already, conflict = _existing_import(source.tree, module, name, alias)
    if conflict:
        raise _Refuse(conflict + "; nothing was written")
    if already:
        return _Outcome(source.text, _warn(
            f"import already present: {line}"))
    bindings = _module_scope_bindings(source.tree)
    if local in bindings:
        raise _Refuse(
            f"refused: adding '{line}' would rebind existing module name "
            f"'{local}' ({bindings[local]}); nothing was written")
    new_text = _insert_top(source, [line], purpose="import")
    return _Outcome(new_text)


# ---------------------------------------------------------------------------
# remove_import
# ---------------------------------------------------------------------------
def _usage_lines(tree: ast.Module, local: str) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == local:
            lines.append(getattr(node, "lineno", 0))
    return sorted(set(lines))


def _all_exports(tree: ast.Module) -> set[str]:
    out: set[str] = set()

    def add_values(node: ast.AST) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.add(node.value)
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    out.add(elt.value)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "__all__"
                   for t in node.targets):
                add_values(node.value)
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name) \
                    and node.target.id == "__all__":
                add_values(node.value)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            func = node.func
            if isinstance(func.value, ast.Name) and func.value.id == "__all__":
                if func.attr in ("append", "extend", "insert"):
                    for arg in node.args:
                        add_values(arg)
    return out


def _comma_tokens(tokens: list[_Token], start: int, end: int) -> list[_Token]:
    return [t for t in tokens
            if t.type == tokenize.OP and t.string == ","
            and start <= t.start < end]


def _remove_alias_span(source: _Source, alias_start: int, alias_end: int,
                       stmt_start: int, stmt_end: int) -> tuple[int, int]:
    """Character span to delete when removing one alias from an import."""
    commas = _comma_tokens(source.tokens(), stmt_start, stmt_end)
    after = [t for t in commas if t.start >= alias_end]
    if after:
        stop = after[0].end
        text = source.text
        while stop < stmt_end and text[stop] in " \t":
            stop += 1
        return alias_start, stop
    before = [t for t in commas if t.end <= alias_start]
    if before:
        return before[-1].start, alias_end
    return stmt_start, stmt_end


def _remove_alias(source: _Source, got: ast.alias, stmt: ast.stmt) -> str:
    """Remove one alias, dropping a whole parenthesized list line when the
    alias owns it, and keeping a trailing comment instead of deleting it."""
    text = source.text
    start, end = source.span(got)
    line_start = source.line_start(got.lineno)
    line_end = source.line_end(got.end_lineno or got.lineno)
    before = text[line_start:start]
    after = text[end:line_end]
    if not before.strip():
        rest = after.lstrip(" \t")
        if rest.startswith(","):
            rest = rest[1:].lstrip(" \t")
        if rest.startswith("#"):
            comment = rest[rest.index("#"):]
            return text[:line_start] + before + comment + text[line_end:]
        if rest == "" or rest.startswith("\n") or rest.startswith("\r"):
            return text[:line_start] + text[line_end:]
    rm_start, rm_end = _remove_alias_span(
        source, start, end, *source.span(stmt))
    return text[:rm_start] + text[rm_end:]


def _remove_whole_statement(source: _Source, stmt: ast.stmt) -> str:
    text = source.text
    start, end = source.span(stmt)
    line_start = source.line_start(stmt.lineno)
    line_end = source.line_end(stmt.end_lineno or stmt.lineno)
    before = text[line_start:start]
    after = text[end:line_end]
    if ";" in after.split("#", 1)[0]:
        raise _Refuse(
            f"refused: the import at line {stmt.lineno} shares its line with "
            "other code; remove it manually")
    if before.strip():
        raise _Refuse(
            f"refused: the import at line {stmt.lineno} does not start its "
            "line; remove it manually")
    stripped = after.lstrip()
    if stripped.startswith("#"):
        comment = after[after.index("#"):]
        replacement = before + comment
        return text[:line_start] + replacement + text[line_end:]
    # Drop exactly one following blank line when the import sat at the start
    # of the code (or after a blank line), so removing it does not leave a
    # stray leading/duplicated blank line. An import between imports or code
    # keeps the blank separator that follows it.
    stop = line_end
    if text[stop:stop + 1] == "\n":
        prev_blank = (line_start == 0
                      or (line_start >= 2
                          and text[line_start - 2:line_start] == "\n\n")
                      or (line_start == 1 and text[0:1] == "\n"))
        if prev_blank:
            stop += 1
    return text[:line_start] + text[stop:]


def _recipe_remove_import(source: _Source, _symbol: str,
                          options: Mapping[str, str]) -> _Outcome:
    module = options.get("module", "").strip()
    name = options.get("name", "").strip()
    alias = options.get("alias", "").strip()
    if not module:
        raise _Refuse("option 'module' is required and may not be empty")
    if alias:
        _identifier(alias, "alias")
    if name:
        _identifier(name, "imported name")

    matches: list[tuple[ast.stmt, ast.alias | None]] = []
    for node in ast.walk(source.tree):
        if name and isinstance(node, ast.ImportFrom):
            if _module_text(node) != module:
                continue
            for got in node.names:
                if got.name != name:
                    continue
                local = got.asname or got.name
                if alias and local != alias:
                    continue
                matches.append((node, got))
        elif not name and isinstance(node, ast.Import):
            for got in node.names:
                if got.name != module:
                    continue
                local = got.asname or got.name.split(".")[0]
                if alias and local != alias:
                    continue
                matches.append((node, got))
    if not matches:
        if not name:
            for node in ast.walk(source.tree):
                if isinstance(node, ast.ImportFrom) \
                        and _module_text(node) == module:
                    raise _Refuse(
                        f"refused: '{module}' is imported with a from-import; "
                        "name the imported symbol explicitly (for example "
                        "name='symbol') -- a bare module option removes only "
                        f"`import {module}`")
        return _Outcome(source.text, _warn(
            f"no matching import of {('name ' + name) if name else module} "
            "was found; nothing to remove"))
    if len(matches) > 1:
        raise _Refuse(
            f"refused: {len(matches)} import statements match; the binding is "
            "ambiguous. Add option alias=... or remove it manually")
    stmt, got = matches[0]
    local = (got.asname or got.name) if got is not None else (
        alias or module.split(".")[0])
    used = _usage_lines(source.tree, local)
    if used:
        shown = ", ".join(str(n) for n in used[:6])
        raise _Refuse(
            f"refused: '{local}' is still used at line(s) {shown}; removing "
            "the import would break the file")
    exported = _all_exports(source.tree)
    if local in exported:
        raise _Refuse(
            f"refused: '{local}' is listed in __all__; removing the import "
            "would break the re-export")

    aliases = [a for a in getattr(stmt, "names", [])]
    if got is not None and len(aliases) > 1:
        new_text = _remove_alias(source, got, stmt)
    else:
        new_text = _remove_whole_statement(source, stmt)
    bad = syntax_error(Path(source.rel), new_text)
    if bad:
        raise _Refuse(
            f"refused: removing the import would leave {source.rel} with "
            f"{bad}; nothing was written")
    return _Outcome(new_text, _warn(f"removed import binding '{local}'"))


# ---------------------------------------------------------------------------
# print_to_logger
# ---------------------------------------------------------------------------
def _print_binding(tree: ast.Module) -> str:
    binds = _all_bindings(tree)
    return binds.get("print", "")


def _scope_for_prints(tree: ast.Module, symbol: str) -> ast.AST:
    if not symbol.strip():
        return tree
    found = _matching_scope(tree, symbol)
    if found is None:
        raise _Refuse(
            f"symbol '{symbol}' was not found as a unique function or class; "
            "nothing was written")
    return found


def _module_logger_bound(tree: ast.Module, name: str) -> bool:
    return name in _module_scope_bindings(tree)


def _module_imports_logging(tree: ast.Module) -> bool:
    for stmt in _leading_imports(tree):
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                local = alias.asname or alias.name.split(".")[0]
                if local == "logging" and (
                        alias.name == "logging"
                        or alias.name.startswith("logging.")):
                    return True
    return False


def _logging_binding_kind(tree: ast.Module) -> str:
    """How a top-level ``logging`` name is bound: module import, other, none."""
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                local = alias.asname or alias.name.split(".")[0]
                if local == "logging":
                    if alias.name == "logging" or alias.name.startswith(
                            "logging."):
                        return "module"
                    return "other"
        elif isinstance(stmt, ast.ImportFrom):
            for alias in stmt.names:
                if (alias.asname or alias.name) == "logging":
                    return "other"
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign,
                               ast.For, ast.AsyncFor, ast.With, ast.AsyncWith,
                               ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            for name in _scope_bindings([stmt]):
                if name == "logging":
                    return "other"
    return ""


def _logger_covered(tree: ast.Module, parents: Mapping[ast.AST, ast.AST],
                    call: ast.Call, logger: str) -> bool:
    """True when `logger` resolves for this call at module or enclosing scope."""
    if _module_logger_bound(tree, logger):
        return True
    cur: ast.AST | None = call
    while cur is not None and cur is not tree:
        parent = parents.get(cur)
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if logger in {a.arg for a in _function_args(parent)} \
                    or logger in _scope_bindings(parent.body):
                return True
        cur = parent
    return False


def _recipe_print_to_logger(source: _Source, symbol: str,
                            options: Mapping[str, str]) -> _Outcome:
    logger = (options.get("logger") or "logger").strip()
    _identifier(logger, "logger")
    level = (options.get("level") or "info").strip().lower()
    allowed_levels = {"debug", "info", "warning", "error", "critical"}
    if level not in allowed_levels:
        raise _Refuse(
            f"option 'level' must be one of {', '.join(sorted(allowed_levels))}"
            f", not '{level}'")
    create = _bool_option(options, "create_logger", True)

    shadow = _print_binding(source.tree)
    if shadow:
        raise _Refuse(
            f"refused: `print` is rebound in this module ({shadow}); "
            "converting calls would hit the wrong callable. Restore the "
            "builtin or edit those calls manually")
    scope = _scope_for_prints(source.tree, symbol)

    calls: list[ast.Call] = []
    for node in ast.walk(scope):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "print"):
            calls.append(node)
    if not calls:
        return _Outcome(source.text, _warn(
            "found no print() call"
            + (f" in '{symbol}'" if symbol else " in the file")
            + "; nothing to change"))
    calls.sort(key=lambda c: source.span(c)[0])

    for call in calls:
        if call.keywords:
            names = ", ".join(k.arg or "**" for k in call.keywords)
            raise _Refuse(
                f"refused: print() at line {call.lineno} passes keyword "
                f"argument(s) {names}; logging calls have different semantics. "
                "Rewrite that call manually")
        if any(isinstance(a, ast.Starred) for a in call.args):
            raise _Refuse(
                f"refused: print() at line {call.lineno} uses *args; the "
                "argument count is not statically known. Rewrite it manually")

    parents = _parent_map(source.tree)
    module_bound = _module_logger_bound(source.tree, logger)
    need_logger = not module_bound and not all(
        _logger_covered(source.tree, parents, call, logger) for call in calls)
    if need_logger and not create:
        raise _Refuse(
            f"refused: '{logger}' is not defined in this module and "
            "create_logger is false; nothing was written")

    edits: list[tuple[int, int, str]] = []
    for call in calls:
        func_start, func_end = source.span(call.func)
        if not call.args:
            start, end = source.span(call)
            edits.append((start, end, f"{logger}.{level}(\"\")"))
            continue
        edits.append((func_start, func_end, f"{logger}.{level}"))
        if len(call.args) > 1:
            open_paren = source.text.find("(", func_end, source.span(call)[1])
            if open_paren < 0:
                raise _Refuse(
                    f"refused: could not locate the argument list of print() "
                    f"at line {call.lineno}")
            fmt = " ".join(["%s"] * len(call.args))
            next_char = source.text[open_paren + 1:open_paren + 2]
            spacer = "" if next_char in ("\n", "\r") else " "
            edits.append((open_paren + 1, open_paren + 1,
                          f'"{fmt}",{spacer}'))

    new_text = _apply_text_edits(source.text, edits)
    if need_logger:
        setup: list[str] = []
        logging_kind = _logging_binding_kind(source.tree)
        if logging_kind == "other":
            raise _Refuse(
                "refused: the module name 'logging' is already bound to "
                "something other than the logging module; cannot create a "
                "logger safely")
        if not _module_imports_logging(source.tree):
            setup.append("import logging")
        setup.append(f"{logger} = logging.getLogger(__name__)")
        new_text = _insert_top(
            _Source(new_text, _parse_python(new_text, source.rel), source.rel),
            setup, purpose="import")
    return _Outcome(new_text, _warn(
        f"converted {len(calls)} print() call(s) to {logger}.{level}"))


def _apply_text_edits(text: str, edits: list[tuple[int, int, str]]) -> str:
    """Apply non-overlapping character edits, right-to-left."""
    ordered = sorted(edits, key=lambda e: (e[0], e[1]))
    last_end = -1
    for start, end, _repl in ordered:
        if start < last_end:
            raise _Refuse(
                "refused: internal codemod edits overlap; nothing was written")
        last_end = end
    out = text
    for start, end, repl in reversed(ordered):
        out = out[:start] + repl + out[end:]
    return out


# ---------------------------------------------------------------------------
# extract_constant
# ---------------------------------------------------------------------------
def _recipe_extract_constant(source: _Source, symbol: str,
                             options: Mapping[str, str]) -> _Outcome:
    name = _identifier(options.get("name", ""), "constant name")
    wanted = _literal_value(options.get("value", ""))
    wanted_key = _literal_key(wanted)

    if symbol.strip():
        scope = _matching_scope(source.tree, symbol)
        if scope is None:
            raise _Refuse(
                f"symbol '{symbol}' was not found as a unique function or "
                "class scope; nothing was written")
    else:
        scope = source.tree

    existing = _existing_module_constant(source.tree, name)
    bindings = _all_bindings(source.tree)
    skip_spans: list[tuple[int, int]] = []
    if name in bindings:
        if isinstance(existing, ast.AST):
            if _literal_key(existing) != wanted_key:
                raise _Refuse(
                    f"refused: '{name}' already exists with a different "
                    "value; choose a new constant name")
            if _binding_occurrences(source.tree, name) != 1:
                raise _Refuse(
                    f"refused: '{name}' is also bound in another scope; a "
                    "replacement could resolve to the wrong name")
            skip_spans.append(source.span(existing))
        elif existing == "conflict" or _binding_occurrences(
                source.tree, name) > 0:
            raise _Refuse(
                f"refused: '{name}' is already bound in this module "
                f"({bindings[name]}); choose a new constant name")
    module_bindings = _module_scope_bindings(source.tree)
    if name in module_bindings and not isinstance(existing, ast.AST):
        raise _Refuse(
            f"refused: '{name}' is already bound at module scope; choose a "
            "new constant name")

    parents = _parent_map(source.tree)
    annotation_spans = [(source.span(n)[0], source.span(n)[1])
                        for n in _annotation_spans(source.tree)]
    annotation_spans += [(source.span(n)[0], source.span(n)[1])
                         for n in _reexport_roots(source.tree)]
    matches: list[ast.AST] = []
    for node in ast.walk(scope):
        if not _is_extractable_literal(node):
            continue
        if _literal_key(node) != wanted_key:
            continue
        if _is_docstring_constant(node, parents):
            continue
        start, end = source.span(node)
        if _inside_span(start, annotation_spans):
            continue
        if _inside_span(start, skip_spans):
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.UnaryOp) and parent.operand is node:
            # A bare number inside `-1` is not the same literal as `1`.
            continue
        skip_ancestors = (ast.JoinedStr, ast.FormattedValue, ast.Match,
                          ast.MatchValue, ast.MatchSingleton, ast.MatchClass,
                          ast.MatchMapping, ast.MatchSequence)
        cur = parent
        inside_match_or_fstring = False
        while cur is not None:
            if isinstance(cur, skip_ancestors):
                inside_match_or_fstring = True
                break
            cur = parents.get(cur)
        if inside_match_or_fstring:
            continue
        matches.append(node)
    if not matches:
        if isinstance(existing, ast.AST):
            return _Outcome(source.text, _warn(
                f"constant {name} already holds the requested literal"))
        return _Outcome(source.text, _warn(
            "found no occurrence of the requested literal"
            + (f" in '{symbol}'" if symbol else " in the file")
            + "; nothing to change"))

    edits = [(*source.span(node), name) for node in matches]
    new_text = _apply_text_edits(source.text, edits)
    if isinstance(existing, ast.AST):
        return _Outcome(new_text, _warn(
            f"reused existing constant {name} for "
            f"{len(matches)} occurrence(s)"))
    first = matches[0]
    literal_source = source.seg(first)
    line = f"{name} = {literal_source}"
    try:
        ast.parse(line)
    except SyntaxError:
        raise _Refuse(
            f"refused: could not form a valid constant assignment from "
            f"'{literal_source}'")
    new_text = _insert_top(
        _Source(new_text, _parse_python(new_text, source.rel), source.rel),
        [line], purpose="constant")
    return _Outcome(new_text, _warn(
        f"extracted {len(matches)} occurrence(s) into {name}"))


def _existing_module_constant(tree: ast.Module,
                              name: str) -> ast.AST | str | None:
    """The RHS of a direct module-level ``name = literal`` assignment.

    Returns an AST node for a reusable constant, the string ``"conflict"``
    when the name is bound in another way, or None when it is unbound.
    """
    direct: ast.AST | None = None
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) == 1 and isinstance(
                    stmt.targets[0], ast.Name) \
                    and stmt.targets[0].id == name:
                if direct is not None:
                    return "conflict"
                direct = stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id == name \
                    and stmt.value is not None:
                if direct is not None:
                    return "conflict"
                direct = stmt.value
    if direct is not None:
        return direct
    if name in _module_scope_bindings(tree):
        return "conflict"
    return None


def _binding_occurrences(tree: ast.Module, name: str) -> int:
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)) and node.name == name:
            count += 1
        elif isinstance(node, ast.Import):
            count += sum(1 for a in node.names
                         if (a.asname or a.name.split(".")[0]) == name)
        elif isinstance(node, ast.ImportFrom):
            count += sum(1 for a in node.names
                         if a.name != "*" and (a.asname or a.name) == name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) \
                and node.id == name:
            count += 1
        elif isinstance(node, ast.arg) and node.arg == name:
            count += 1
        elif isinstance(node, ast.ExceptHandler) and node.name == name:
            count += 1
    return count


# ---------------------------------------------------------------------------
# add_annotation
# ---------------------------------------------------------------------------
def _function_colon(source: _Source, fn: ast.AST) -> int:
    """Character offset of the colon that closes a def/async def header."""
    start = source.line_start(fn.lineno)
    depth = 0
    seen_def = False
    for tok in source.tokens():
        if tok.end <= start:
            continue
        if tok.type == tokenize.NAME and tok.string == "def":
            seen_def = True
            continue
        if not seen_def:
            continue
        if tok.type == tokenize.OP:
            if tok.string in "([{":
                depth += 1
            elif tok.string in ")]}":
                depth = max(0, depth - 1)
            elif tok.string == ":" and depth == 0:
                return tok.start
    raise _Refuse(
        f"refused: could not locate the header colon of "
        f"{getattr(fn, 'name', 'the function')} at line {fn.lineno}")


def _find_arg(fn: ast.AST, name: str) -> ast.arg:
    for arg in _function_args(fn):
        if arg.arg == name:
            return arg
    known = ", ".join(a.arg for a in _function_args(fn)) or "(none)"
    raise _Refuse(
        f"refused: '{name}' is not a parameter of "
        f"{getattr(fn, 'name', 'the function')}; parameters: {known}")


def _scope_symbol_for_param(source: _Source, symbol: str,
                            options: Mapping[str, str]) -> tuple[ast.AST, str]:
    name = (options.get("name") or "").strip()
    if name:
        fn = _resolve_definition(source.tree, symbol)
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise _Refuse(f"symbol '{symbol}' is not a function definition")
        return fn, _identifier(name, "parameter name")
    if "." not in symbol:
        fn = _resolve_definition(source.tree, symbol)
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise _Refuse(f"symbol '{symbol}' is not a function definition")
        raise _Refuse(
            f"refused: '{symbol}' is a function; name the parameter with "
            "option name=... or use symbol='func.param'")
    prefix, leaf = symbol.rsplit(".", 1)
    fn = _resolve_definition(source.tree, prefix)
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise _Refuse(f"symbol '{prefix}' is not a function definition")
    _find_arg(fn, leaf)                # validates the parameter name
    return fn, leaf


def _recipe_add_annotation(source: _Source, symbol: str,
                           options: Mapping[str, str]) -> _Outcome:
    annotation = _parse_annotation(options.get("annotation", ""))
    target = (options.get("target") or "auto").strip().lower()
    if target not in ("auto", "param", "return", "variable"):
        raise _Refuse(
            "option 'target' must be one of auto, param, return, variable")
    overwrite = _bool_option(options, "overwrite", False)
    name_option = (options.get("name") or "").strip()

    if target == "auto":
        if name_option and not symbol.strip():
            target = "variable"
        elif name_option:
            found = _matching_scope(source.tree, symbol)
            if isinstance(found, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and not any(a.arg == name_option
                                for a in _function_args(found)):
                try:
                    _find_variable_assignment(found, name_option)
                    target = "variable"
                except _Refuse as exc:
                    if "ambiguous" in str(exc):
                        raise
                    target = "param"
            elif isinstance(found, ast.ClassDef):
                try:
                    _find_variable_assignment(found, name_option)
                    target = "variable"
                except _Refuse as exc:
                    if "ambiguous" in str(exc):
                        raise
                    target = "param"
            else:
                target = "param"
        elif _is_variable_symbol(source, symbol, options):
            target = "variable"
        else:
            found = _matching_scope(source.tree, symbol)
            if found is not None and isinstance(
                    found, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and symbol and "." not in symbol:
                target = "return"
            elif "." in symbol:
                prefix, leaf = symbol.rsplit(".", 1)
                try:
                    fn = _resolve_definition(source.tree, prefix)
                except _Refuse:
                    fn = None
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and any(a.arg == leaf for a in _function_args(fn)):
                    target = "param"
                else:
                    target = "return"
            else:
                target = "param"
    if target == "param":
        fn, param = _scope_symbol_for_param(source, symbol, options)
        arg = _find_arg(fn, param)
        if arg.annotation is not None:
            if _annotation_equal(arg.annotation, annotation):
                return _Outcome(source.text, _warn(
                    f"parameter '{param}' is already annotated as "
                    f"'{annotation}'"))
            if not overwrite:
                raise _Refuse(
                    f"refused: parameter '{param}' is already annotated; "
                    "pass overwrite=true to replace it")
            start, end = source.span(arg.annotation)
            return _Outcome(source.text[:start] + annotation + source.text[end:],
                            _warn(f"replaced annotation on '{param}'"))
        start, end = source.span(arg)
        return _Outcome(
            source.text[:end] + f": {annotation}" + source.text[end:],
            _warn(f"annotated parameter '{param}' as {annotation}"))
    if target == "return":
        fn = _resolve_definition(source.tree, symbol)
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise _Refuse(f"symbol '{symbol}' is not a function definition")
        if fn.returns is not None:
            if _annotation_equal(fn.returns, annotation):
                return _Outcome(source.text, _warn(
                    f"{fn.name} already returns '{annotation}'"))
            if not overwrite:
                raise _Refuse(
                    f"refused: {fn.name} already has a return annotation; "
                    "pass overwrite=true to replace it")
            start, end = source.span(fn.returns)
            return _Outcome(source.text[:start] + annotation + source.text[end:],
                            _warn(f"replaced return annotation on {fn.name}"))
        colon = _function_colon(source, fn)
        return _Outcome(
            source.text[:colon] + f" -> {annotation}" + source.text[colon:],
            _warn(f"annotated the return of {fn.name} as {annotation}"))

    # variable
    scope_node, var_name = _variable_scope(source, symbol, options)
    ann, stmt = _find_variable_assignment(scope_node, var_name)
    if ann is not None:
        if _annotation_equal(ann, annotation):
            return _Outcome(source.text, _warn(
                f"variable '{var_name}' is already annotated as {annotation}"))
        if not overwrite:
            raise _Refuse(
                f"refused: variable '{var_name}' already has an annotation; "
                "pass overwrite=true to replace it")
        start, end = source.span(ann)
        return _Outcome(source.text[:start] + annotation + source.text[end:],
                        _warn(f"replaced annotation on '{var_name}'"))
    target_node = stmt.targets[0] if isinstance(stmt, ast.Assign) \
        else stmt.target
    start, end = source.span(target_node)
    return _Outcome(
        source.text[:end] + f": {annotation}" + source.text[end:],
        _warn(f"annotated variable '{var_name}' as {annotation}"))


def _is_variable_symbol(source: _Source, symbol: str,
                        options: Mapping[str, str]) -> bool:
    name = (options.get("name") or "").strip() or symbol.strip()
    if not name:
        return False
    try:
        scope_node, var_name = _variable_scope(source, symbol, options)
        _find_variable_assignment(scope_node, var_name)
        return True
    except _Refuse as exc:
        if "ambiguous" in str(exc):
            raise
        return False


def _variable_scope(source: _Source, symbol: str,
                    options: Mapping[str, str]) -> tuple[ast.AST, str]:
    scope_option = (options.get("scope") or "").strip()
    name = (options.get("name") or "").strip()
    if scope_option:
        scope_node = _matching_scope(source.tree, scope_option)
        if scope_node is None:
            raise _Refuse(
                f"symbol '{scope_option}' was not found as a unique scope")
        return scope_node, _identifier(name or symbol, "variable name")
    if "." in symbol and not name:
        prefix, leaf = symbol.rsplit(".", 1)
        scope_node = _matching_scope(source.tree, prefix)
        if isinstance(scope_node, (ast.ClassDef, ast.FunctionDef,
                                   ast.AsyncFunctionDef)):
            return scope_node, _identifier(leaf, "variable name")
        raise _Refuse(
            f"refused: '{symbol}' is not a class/function-qualified variable; "
            "pass option name=... and scope=... for a local variable")
    if not name:
        return source.tree, _identifier(symbol, "variable name")
    scope_node = _matching_scope(source.tree, symbol)
    if scope_node is not None:
        return scope_node, _identifier(name, "variable name")
    return source.tree, _identifier(name, "variable name")


def _find_variable_assignment(scope_node: ast.AST, name: str
                              ) -> tuple[ast.AST | None, ast.AST]:
    body = getattr(scope_node, "body", [])
    found: list[ast.AST] = []
    for stmt in body:
        if isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id == name:
                found.append(stmt)
        elif isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                found.append(stmt)
    if not found:
        raise _Refuse(
            f"refused: no direct assignment to variable '{name}' in the "
            "selected scope; annotate it manually")
    if len(found) > 1:
        raise _Refuse(
            f"refused: variable '{name}' has {len(found)} assignments in the "
            "selected scope; the target is ambiguous")
    stmt = found[0]
    if isinstance(stmt, ast.AnnAssign):
        return stmt.annotation, stmt
    return None, stmt


# ---------------------------------------------------------------------------
# Handler registry (WP-4102B extends this mapping)
# ---------------------------------------------------------------------------
_HANDLERS = {
    "add_import": _recipe_add_import,
    "remove_import": _recipe_remove_import,
    "print_to_logger": _recipe_print_to_logger,
    "extract_constant": _recipe_extract_constant,
    "add_annotation": _recipe_add_annotation,
}
