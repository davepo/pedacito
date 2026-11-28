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
import os
import tokenize
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import fileio, safety, security
from .edits import Edit, apply_edits_atomic, syntax_error
from .imports import (FileSet, module_name, python_import_bindings,
                      raw_imports, relative_specifier,
                      resolve as _import_resolve)

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
        "Add a parameter to a uniquely resolved function and update only "
        "provable call sites; a required parameter refuses when any provable "
        "caller exists (pass a default to keep them working).",
        implemented=True, symbol="required",
        options=[
            _opt("name", "Parameter name.", required=True),
            _opt("annotation", "Optional annotation source."),
            _opt("default", "Optional default-value source."),
        ],
    ),
    _recipe(
        "remove_param", "Remove a parameter",
        "Remove a parameter and update only provable call sites.",
        implemented=True, symbol="required",
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
        "Wrap the body of a uniquely resolved function or method in "
        "try/except that re-raises (an error-handling seam to edit further).",
        implemented=True, symbol="required",
        options=[
            _opt("exception", "Exception type expression to catch.",
                 default="Exception"),
        ],
    ),
    _recipe(
        "wrap_context", "Wrap in a context manager",
        "Wrap the body of a uniquely resolved function or method in a "
        "with-block.",
        implemented=True, symbol="required",
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
        "Move a uniquely resolved top-level function or class into another "
        "file and update only resolved imports.",
        implemented=True, symbol="required",
        options=[_opt("to_file", "Destination project-relative path.",
                      required=True)],
    ),
    _recipe(
        "move_file", "Move a file",
        "Move a Python source file to a new path and update only resolved "
        "imports; no delete until every staged path validates.",
        implemented=True, symbol="ignored",
        options=[_opt("to_path", "Destination project-relative path.",
                      required=True)],
    ),
    _recipe(
        "replace_deprecated", "Replace a deprecated symbol",
        "Replace a deprecated API name at resolved references in the target "
        "file (strings and comments untouched); refuse when the binding is "
        "ambiguous.",
        implemented=True, symbol="optional",
        options=[
            _opt("old", "Deprecated name.", required=True),
            _opt("new", "Replacement name.", required=True),
        ],
    ),
    _recipe(
        "add_reexport", "Add a re-export",
        "Re-export a name from a module by adding a from-import to the target "
        "file (usually a package __init__.py); verifies the name is defined "
        "in the source module.",
        implemented=True, symbol="ignored",
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

    The preview is recomputed from current bytes, every touched file's preimage
    hash is verified again by ``apply_edits_atomic`` immediately before commit,
    and a policy/protection/sensitive-path failure leaves every file untouched.
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

    edits: list[Edit] = []
    expected_sha256: dict[str, str] = {}
    for change in plan.changes:
        if change.delete:
            edits.append(Edit(change.rel, "", "", delete=True))
            if change.existed:
                expected_sha256[change.rel] = change.raw_sha
        elif not change.existed:
            edits.append(Edit(change.rel, "", change.after, exact=True))
        else:
            edits.append(Edit(change.rel, change.before, change.after,
                              exact=True))
            expected_sha256[change.rel] = change.raw_sha
    try:
        outcomes = apply_edits_atomic(
            plan.root, edits, backup=backup, backup_dir=backup_dir, cfg=cfg,
            expected_sha256=expected_sha256)
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
class _FileChange:
    """One file's planned transformation within a codemod batch.

    ``rel`` is a project-relative POSIX path. ``before``/``after`` are the
    LF-normalised preimage/postimage text; ``before`` is empty for a creation
    and ``after`` empty for a deletion. ``raw_sha`` pins the file's on-disk
    bytes (empty for a creation) so a stale apply can be refused.
    """

    rel: str
    before: str
    after: str
    raw_sha: str
    existed: bool
    delete: bool = False


@dataclass(frozen=True)
class _Plan:
    root: Path
    changes: tuple[_FileChange, ...]
    result: CodemodResult


def _prepare(root: Path, request: CodemodRequest, *, cfg, for_write: bool,
             index=None) -> _Plan:
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
    if handler is None and recipe_id not in _MULTI_HANDLERS:
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
    tree = (_parse_python(loaded.text, loaded.rel)
            if loaded.text.strip() else ast.parse("", filename=loaded.rel))
    source = _Source(loaded.text, tree, loaded.rel)

    if recipe_id in _MULTI_HANDLERS:
        multi = _MULTI_HANDLERS[recipe_id](
            root, request, loaded, source, options, cfg=cfg,
            for_write=for_write, index=index)
        return _finalize_plan(loaded, multi.changes, multi.warnings)

    outcome = handler(source, str(request.symbol or "").strip(), options)
    warnings = list(outcome.warnings)
    if outcome.text == loaded.text:
        warnings.append("no change: the requested transformation is already "
                        "present")
        return _finalize_plan(
            loaded, (), tuple(warnings), no_change=True)
    change = _FileChange(loaded.rel, loaded.text, outcome.text,
                         loaded.raw_sha, True)
    return _finalize_plan(loaded, (change,), tuple(warnings))


def _finalize_plan(loaded: _Loaded, changes: tuple[_FileChange, ...],
                   warnings: tuple[str, ...], *, no_change: bool = False
                   ) -> _Plan:
    """Validate a batch of file changes and build the result plan.

    Every produced text is syntax-checked, no-op changes are dropped, and the
    diffs/changed set are assembled deterministically (sorted by path).
    """
    root = loaded.root
    warnings = list(warnings)
    kept: list[_FileChange] = []
    for change in changes:
        if change.delete:
            if not change.existed:
                continue
            kept.append(change)
            continue
        if change.existed and change.before == change.after:
            continue
        if not change.existed and change.after == "":
            continue
        bad = syntax_error(root / change.rel, change.after)
        if bad:
            raise _Refuse(
                f"refused: the codemod would leave {change.rel} with {bad}; "
                "nothing was written")
        kept.append(change)
    if not kept:
        if no_change or not changes:
            warnings.append("no change: the requested transformation is "
                            "already present")
        else:
            warnings.append("no change: the requested transformation produced "
                            "nothing")
        return _Plan(root, (),
                     CodemodResult(True, {}, (), tuple(warnings)))

    diffs: dict[str, str] = {}
    changed: list[str] = []
    for change in kept:
        if change.delete:
            diff = _unified_diff(change.rel, change.before, "")
            diffs[change.rel] = diff
        else:
            diffs[change.rel] = _unified_diff(change.rel, change.before,
                                              change.after)
        changed.append(change.rel)
    changed.sort()
    result = CodemodResult(True, diffs, tuple(changed), tuple(warnings))
    return _Plan(root, tuple(kept), result)


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
# Cross-file machinery (WP-4102B)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _MultiOutcome:
    """A batch of file changes plus warnings, produced by a multi-file recipe."""

    changes: tuple[_FileChange, ...]
    warnings: tuple[str, ...] = ()


def _single(outcome: _Outcome, loaded: _Loaded) -> _MultiOutcome:
    change = _FileChange(loaded.rel, loaded.text, outcome.text,
                         loaded.raw_sha, True)
    return _MultiOutcome((change,), outcome.warnings)


_SKIP_DIRS = frozenset({
    "node_modules", "__pycache__", ".venv", "venv", "env", ".tox", ".nox",
    "dist", "build", "target", "site-packages", "dist-packages", "vendor",
    ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".idea", ".vscode",
})


def _iter_py_files(root: Path) -> list[str]:
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in _SKIP_DIRS and not d.startswith(".")
            and not d.endswith(".egg-info"))
        for f in sorted(filenames):
            if f.endswith(".py"):
                p = Path(dirpath) / f
                try:
                    out.append(p.relative_to(root).as_posix())
                except ValueError:
                    continue
    return out


def _resolve_project_path(root: Path, file: str, cfg, for_write: bool, *,
                          must_exist: bool = False) -> tuple[Path, str]:
    """Resolve a project-relative path safely, returning (path, rel)."""
    try:
        root_resolved = Path(root).expanduser().resolve()
    except OSError as exc:
        raise _Refuse(f"project root could not be resolved: {root} ({exc})")
    raw = fileio.normalise_rel(file)
    if not raw:
        raise _Refuse("a project-relative path is required")
    try:
        path = security.resolve_safe_path(root_resolved, raw)
    except security.SecurityError as exc:
        raise _Refuse(f"refused: {exc}; nothing was written")
    try:
        rel = path.relative_to(root_resolved).as_posix()
    except ValueError:
        raise _Refuse(f"refused: path escapes the project root: {file}")
    why = safety.is_protected(rel, cfg, write=for_write)
    if why:
        raise _Refuse(f"refused: {why}; nothing was written")
    if security.is_sensitive_path(rel):
        raise _Refuse(f"refused: {rel} is a sensitive file and is never "
                      "written by a codemod")
    if must_exist and not path.is_file():
        raise _Refuse(
            f"refused: {rel} is not a regular file in the project; nothing "
            "was written")
    return path, rel


def _load_extra(root: Path, rel: str, cfg, for_write: bool) -> _Loaded:
    return _load(root, rel, "", cfg=cfg, for_write=for_write)


def _importers(root: Path, rel: str, index) -> list[str]:
    """Files that import ``rel``'s module: from the resolved graph when there
    is an index, otherwise a deterministic textual scan of the project."""
    if index is not None:
        try:
            mapped = index.resolve_file(rel) or rel
            return sorted(f for f in index.importers_of(mapped) if f != mapped)
        except Exception:
            pass
    fs = FileSet(_iter_py_files(root))
    out: list[str] = []
    for cand in _iter_py_files(root):
        if cand == rel:
            continue
        try:
            text = fileio.read_text(root / cand)
        except (OSError, UnicodeDecodeError):
            continue
        for spec in raw_imports(cand, text):
            if rel in _import_resolve(cand, spec, fs):
                out.append(cand)
                break
    return sorted(out)


def _resolve_function_qual(source: _Source, symbol: str
                           ) -> tuple[ast.AST, str]:
    """Resolve a uniquely named function/method, returning (node, qualname)."""
    wanted = (symbol or "").strip()
    if not wanted:
        raise _Refuse("this recipe needs a symbol naming a function")
    matches = [(q, n) for q, n in _definitions(source.tree)
               if q == wanted
               and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not matches:
        known = sorted({q for q, n in _definitions(source.tree)
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))})[:8]
        raise _Refuse(
            f"symbol '{wanted}' was not found as a unique function"
            + (f"; known functions: {', '.join(known)}" if known else "")
            + "; nothing was written")
    if len(matches) > 1:
        where = ", ".join(f"line {getattr(n, 'lineno', '?')}"
                          for _q, n in matches[:6])
        raise _Refuse(
            f"symbol '{wanted}' is ambiguous ({len(matches)} functions: "
            f"{where}); refuse to guess which one to edit")
    return matches[0][1], matches[0][0]


# ---------------------------------------------------------------------------
# Parameter helpers (add_param / remove_param)
# ---------------------------------------------------------------------------
def _positional_params(fn: ast.AST) -> list[ast.arg]:
    return list(fn.args.posonlyargs) + list(fn.args.args)


def _open_paren_offset(source: _Source, fn: ast.AST) -> int:
    start = source.line_start(getattr(fn, "lineno", 1))
    seen_def = False
    for tok in source.tokens():
        if tok.end <= start:
            continue
        if tok.type == tokenize.NAME and tok.string == "def":
            seen_def = True
            continue
        if not seen_def:
            continue
        if tok.type == tokenize.OP and tok.string == "(":
            return tok.end
    raise _Refuse(
        f"refused: could not locate the parameter list of "
        f"{getattr(fn, 'name', 'the function')}")


def _param_text(name: str, annotation: str, default: str) -> str:
    if annotation:
        result = f"{name}: {annotation}"
        if default:
            result += f" = {default}"
        return result
    if default:
        return f"{name}={default}"
    return name


def _arg_has_default(fn: ast.AST, index: int, positional: list[ast.arg]) -> bool:
    ndefaults = len(fn.args.defaults)
    return index >= len(positional) - ndefaults


def _insert_param(source: _Source, fn: ast.AST, name: str, annotation: str,
                  default: str) -> str:
    """Return the source with `name` inserted at a valid parameter position."""
    if fn.args.vararg is not None or fn.args.kwarg is not None:
        raise _Refuse(
            "refused: the function has *args/**kwargs; the call-site arity is "
            "not statically known, so a parameter cannot be added or removed "
            "mechanically")
    positional = _positional_params(fn)
    has_default = bool(default)
    text = source.text
    if positional:
        if has_default:
            last = positional[-1]
            offset = source.span(last)[1]
            return text[:offset] + ", " + _param_text(name, annotation, default) \
                + text[offset:]
        first_default = None
        for idx, arg in enumerate(positional):
            if _arg_has_default(fn, idx, positional):
                first_default = arg
                break
        if first_default is not None:
            offset = source.span(first_default)[0]
            return text[:offset] + _param_text(name, annotation, default) + ", " \
                + text[offset:]
        last = positional[-1]
        offset = source.span(last)[1]
        return text[:offset] + ", " + _param_text(name, annotation, default) \
            + text[offset:]
    # no positional parameters: insert after the opening paren
    offset = _open_paren_offset(source, fn)
    suffix = ", " if fn.args.kwonlyargs else ""
    return text[:offset] + _param_text(name, annotation, default) + suffix \
        + text[offset:]


def _remove_param_def(source: _Source, fn: ast.AST, arg: ast.arg) -> str:
    """Return the source with one parameter (and its comma) removed."""
    text = source.text
    start, end = source.span(arg)
    tokens = source.tokens()
    commas = [t for t in tokens
              if t.type == tokenize.OP and t.string == ","
              and t.start >= end and t.start <= source.line_end(arg.lineno)]
    if commas:
        # delete the arg plus the following comma and any spaces
        stop = commas[0].end
        while stop < len(text) and text[stop] in " \t":
            stop += 1
        return text[:start] + text[stop:]
    commas_before = [t for t in tokens
                     if t.type == tokenize.OP and t.string == ","
                     and t.end <= start
                     and t.start >= _open_paren_offset(source, fn)]
    if commas_before:
        return text[:commas_before[-1].start] + text[end:]
    return text[:start] + text[end:]


def _expr_path(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    else:
        return ""
    return ".".join(reversed(parts))


def _inside_class(node: ast.AST, class_leaf: str,
                  parents: Mapping[ast.AST, ast.AST]) -> bool:
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, ast.ClassDef) and cur.name == class_leaf:
            return True
        cur = parents.get(cur)
    return False


def _local_shadowed(tree: ast.Module, parents: Mapping[ast.AST, ast.AST],
                    fn: ast.AST, name: str, node: ast.AST) -> bool:
    """True when `name` resolves to a local binding (not the target `fn`) at
    `node`'s position in a nested scope."""
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if cur is fn:
                if name in {a.arg for a in _function_args(cur)}:
                    return True
            else:
                if name in {a.arg for a in _function_args(cur)}:
                    return True
                if name in _scope_bindings(cur.body):
                    return True
        elif isinstance(cur, ast.ClassDef):
            if name in _scope_bindings(cur.body):
                return True
        cur = parents.get(cur)
    return False


def _module_rebound_before(tree: ast.Module, name: str, lineno: int) -> bool:
    for stmt in tree.body:
        if getattr(stmt, "lineno", 0) >= lineno:
            break
        if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) \
                else [stmt.target]
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                return True
    return False


def _find_call_sites(source: _Source, fn: ast.AST, qualname: str
                     ) -> tuple[list[tuple[ast.Call, int]], list[str],
                                list[str]]:
    """(calls, dynamic_lines, issues) for a function's provable call sites.

    Each entry in `calls` is ``(call, receiver_offset)``: the number of leading
    parameters that are implicit at that call site (1 for ``self.m``/``cls.m``
    on a method, 0 for a bare or static ``C.m`` call), so the caller can map a
    parameter index onto the call's positional argument list. `dynamic_lines`
    are bare references to the name that are not calls (aliases/callbacks);
    `issues` are sites that could not be proven (shadowing, dynamic receivers).
    """
    tree = source.tree
    leaf = qualname.rsplit(".", 1)[-1]
    is_method = "." in qualname
    class_leaf = qualname.rsplit(".", 1)[0].rsplit(".", 1)[-1] if is_method else ""
    parents = _parent_map(tree)
    calls: list[tuple[ast.Call, int]] = []
    dynamic: list[str] = []
    issues: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == leaf and not is_method:
                if _local_shadowed(tree, parents, fn, leaf, node) \
                        or _module_rebound_before(tree, leaf, node.lineno):
                    issues.append(f"line {node.lineno}: '{leaf}' is shadowed")
                else:
                    calls.append((node, 0))
            elif isinstance(func, ast.Name) and func.id == leaf and is_method:
                # A bare `m(...)` resolves to a module/global name, never the
                # method `C.m`, so it is not a call site for this method.
                continue
            elif isinstance(func, ast.Attribute) and func.attr == leaf \
                    and is_method:
                path = _expr_path(func.value)
                if path in ("self", "cls", "this"):
                    if _inside_class(node, class_leaf, parents):
                        calls.append((node, 1))
                    else:
                        issues.append(
                            f"line {node.lineno}: self.{leaf} outside "
                            f"{class_leaf}")
                elif path.endswith("." + class_leaf) or path == class_leaf:
                    calls.append((node, 0))
                else:
                    issues.append(
                        f"line {node.lineno}: dynamic receiver `{path or '(...)'}"
                        f".{leaf}` is not provable")
        elif isinstance(node, ast.Name) and node.id == leaf \
                and isinstance(node.ctx, ast.Load):
            parent = parents.get(node)
            if isinstance(parent, ast.Call) and parent.func is node:
                continue
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef)) and parent.name == leaf:
                continue
            if isinstance(parent, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(parent, ast.arg) and parent.annotation is node:
                continue
            dynamic.append(f"line {node.lineno}")

    return calls, dynamic, issues


def _call_positional_count_ok(call: ast.Call) -> str | None:
    """An error when a call site's arguments cannot be mapped by position."""
    if call.keywords:
        names = ", ".join(k.arg or "**" for k in call.keywords)
        return f"keyword argument(s) {names}"
    if any(isinstance(a, ast.Starred) for a in call.args):
        return "*args"
    return None


# ---------------------------------------------------------------------------
# wrap_try / wrap_context (shared body wrapper)
# ---------------------------------------------------------------------------
def _indent_unit(source: _Source, fn: ast.AST) -> str:
    """One level of indentation for the function body (4 spaces or a tab),
    derived from the gap between the `def` and its first body statement."""
    fn_line = source.lines[fn.lineno - 1]
    fn_indent = fn_line[: len(fn_line) - len(fn_line.lstrip())]
    for stmt in fn.body:
        if _is_docstring_stmt(stmt):
            continue
        line = source.lines[stmt.lineno - 1]
        lead = line[: len(line) - len(line.lstrip())]
        if lead.startswith(fn_indent) and len(lead) > len(fn_indent):
            unit = lead[len(fn_indent):]
            if "\t" in unit:
                return "\t"
            return unit
    return "    "


def _reindent_block(text: str, cur: str, new: str) -> str:
    out: list[str] = []
    for ln in text.split("\n"):
        if not ln.strip():
            out.append(ln)
        elif ln.startswith(cur):
            out.append(new + ln[len(cur):])
        else:
            out.append(new + ln.lstrip())
    return "\n".join(out)


def _wrap_fn_body(source: _Source, fn: ast.AST, opener: str,
                  footer: str) -> str:
    """Wrap a function's non-docstring body: `opener` (a single header line)
    goes at the body's base indent, the body shifts one level deeper, and
    `footer` (a multi-line string with its own relative indentation) closes
    the block at the base indent."""
    start, end, body_text, base, unit = _fn_body_info(source, fn)
    inner = base + unit
    opener_indented = base + opener + "\n"
    wrapped = opener_indented + _reindent_block(body_text, base, inner)
    if footer:
        footer_lines = [base + ln if ln.strip() else ln
                        for ln in footer.split("\n")]
        wrapped += "\n".join(footer_lines) + "\n"
    return source.text[:start] + wrapped + source.text[end:]


def _fn_body_info(source: _Source, fn: ast.AST
                  ) -> tuple[int, int, str, str, str]:
    """(start, end, body_text, base_indent, unit) for a function's body."""
    body = list(fn.body)
    start_idx = 1 if body and _is_docstring_stmt(body[0]) else 0
    if start_idx >= len(body):
        raise _Refuse(
            f"refused: the body of {getattr(fn, 'name', 'the function')} is "
            "empty or only a docstring; nothing to wrap")
    first = body[start_idx]
    last = body[-1]
    if first.lineno == getattr(fn, "lineno", first.lineno):
        raise _Refuse(
            f"refused: {getattr(fn, 'name', 'the function')} has a one-line "
            "body; wrap it manually")
    start = source.line_start(first.lineno)
    end = source.line_end(last.end_lineno or last.lineno)
    body_text = source.text[start:end]
    unit = _indent_unit(source, fn)
    fn_line = source.lines[fn.lineno - 1]
    fn_indent = fn_line[: len(fn_line) - len(fn_line.lstrip())]
    base = fn_indent + unit
    return start, end, body_text, base, unit


# ---------------------------------------------------------------------------
# wrap_try
# ---------------------------------------------------------------------------
def _recipe_wrap_try(source: _Source, symbol: str,
                     options: Mapping[str, str]) -> _Outcome:
    fn, _qual = _resolve_function_qual(source, symbol)
    exception = (options.get("exception") or "Exception").strip()
    if not exception or "\n" in exception or "\r" in exception:
        raise _Refuse("option 'exception' must be a single-line expression")
    try:
        ast.parse(exception, mode="eval")
    except SyntaxError as exc:
        raise _Refuse(f"option 'exception' is not a valid expression: {exc.msg}")
    unit = _indent_unit(source, fn)
    new_text = _wrap_fn_body(
        source, fn, "try:",
        f"except {exception}:\n{unit}raise")
    return _Outcome(new_text, _warn(
        f"wrapped the body of {fn.name} in try/except {exception}"))


# ---------------------------------------------------------------------------
# wrap_context
# ---------------------------------------------------------------------------
def _recipe_wrap_context(source: _Source, symbol: str,
                         options: Mapping[str, str]) -> _Outcome:
    fn, _qual = _resolve_function_qual(source, symbol)
    manager = (options.get("manager") or "").strip()
    if not manager or "\n" in manager or "\r" in manager:
        raise _Refuse("option 'manager' must be a single-line expression")
    try:
        ast.parse(manager, mode="eval")
    except SyntaxError as exc:
        raise _Refuse(f"option 'manager' is not a valid expression: {exc.msg}")
    new_text = _wrap_fn_body(source, fn, f"with {manager}:", "")
    return _Outcome(new_text, _warn(
        f"wrapped the body of {fn.name} in `with {manager}:`"))


# ---------------------------------------------------------------------------
# replace_deprecated
# ---------------------------------------------------------------------------
def _recipe_replace_deprecated(source: _Source, symbol: str,
                               options: Mapping[str, str]) -> _Outcome:
    old = (options.get("old") or "").strip()
    new = (options.get("new") or "").strip()
    if not old:
        raise _Refuse("option 'old' is required and may not be empty")
    if not new:
        raise _Refuse("option 'new' is required and may not be empty")
    if old == new:
        raise _Refuse("the old and new names are the same; nothing to replace")
    _identifier(old, "old name")
    _identifier(new, "new name")

    bindings = _all_bindings(source.tree)
    shadow = bindings.get(old, "")
    if shadow and shadow not in ("import", "definition"):
        raise _Refuse(
            f"refused: '{old}' is bound as {shadow} in this module; replacing "
            "it mechanically could hit a local variable, not the deprecated "
            "API")

    scope = source.tree
    if symbol.strip():
        found = _matching_scope(source.tree, symbol)
        if found is None:
            raise _Refuse(
                f"symbol '{symbol}' was not found as a unique scope")
        scope = found

    parents = _parent_map(source.tree)
    hits: list[tuple[int, int]] = []
    for node in ast.walk(scope):
        if isinstance(node, ast.Name) and node.id == old:
            parent = parents.get(node)
            if isinstance(parent, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef)) and parent.name == old:
                continue
            if isinstance(parent, ast.Attribute) and parent.attr == old \
                    and parent.value is node:
                continue
            hits.append(source.span(node))
    if not hits:
        return _Outcome(source.text, _warn(
            f"found no reference to '{old}'"
            + (f" in '{symbol}'" if symbol else " in the file")
            + "; nothing to change"))
    edits = [(s, e, new) for s, e in hits]
    new_text = _apply_text_edits(source.text, edits)
    return _Outcome(new_text, _warn(
        f"replaced {len(hits)} reference(s) to '{old}' with '{new}'"))


# ---------------------------------------------------------------------------
# add_reexport (multi: validates the source module)
# ---------------------------------------------------------------------------
def _recipe_add_reexport(root, request, loaded, source, options, *, cfg,
                         for_write, index):
    del request, index
    name = _identifier(options.get("name", ""), "name")
    from_module = (options.get("from_module") or "").strip()
    if not from_module or not all(part.isidentifier() for part in
                                  from_module.split(".")):
        raise _Refuse(f"option 'from_module' '{from_module}' is not a dotted "
                      "module name")
    fs = FileSet(_iter_py_files(root))
    targets = _import_resolve(loaded.rel, from_module, fs)
    if not targets:
        raise _Refuse(
            f"refused: module '{from_module}' could not be resolved to a "
            "project file; a re-export must point at a provable definition")
    if len(targets) > 1:
        raise _Refuse(
            f"refused: module '{from_module}' is ambiguous ({len(targets)} "
            "files); name it unambiguously")
    target_rel = targets[0]
    target_loaded = _load_extra(root, target_rel, cfg, for_write)
    target_tree = (_parse_python(target_loaded.text, target_rel)
                   if target_loaded.text.strip()
                   else ast.parse("", filename=target_rel))
    if name not in _module_scope_bindings(target_tree):
        raise _Refuse(
            f"refused: '{name}' is not defined at module scope in "
            f"{target_rel}; a re-export would be unverified")

    line, _local = _import_line(from_module, name, "")
    already, conflict = _existing_import(source.tree, from_module, name, "")
    if conflict:
        raise _Refuse(conflict + "; nothing was written")
    if already:
        return _MultiOutcome((), _warn(f"re-export already present: {line}"))
    new_text = _insert_top(source, [line], purpose="import")
    return _single(_Outcome(new_text, _warn(
        f"re-exported {name} from {from_module}")), loaded)


# ---------------------------------------------------------------------------
# add_param / remove_param (multi: definition + provable callers)
# ---------------------------------------------------------------------------
def _cross_file_callers(root: Path, def_rel: str, qualname: str, index, cfg,
                        for_write: bool) -> tuple[dict[str, list[ast.Call]],
                                                  list[str]]:
    """(calls_by_file, issues) for callers of `qualname` in files that import
    `def_rel` via a from-import of the function's bare name."""
    leaf = qualname.rsplit(".", 1)[-1]
    importers = _importers(root, def_rel, index)
    fs = FileSet(_iter_py_files(root))
    calls_by_file: dict[str, list[ast.Call]] = {}
    issues: list[str] = []
    for rel in importers:
        loaded = _load_extra(root, rel, cfg, for_write)
        tree = _parse_python(loaded.text, rel)
        local_names = set()
        for binding in python_import_bindings(loaded.text, tree):
            if binding.kind != "from":
                continue
            spec = ("." * binding.level) + binding.module
            try:
                targets = _import_resolve(rel, spec, fs) if spec else []
            except Exception:
                targets = []
            if def_rel in targets:
                for n, asn in binding.names:
                    if n == leaf:
                        local_names.add(asn or n)
        if not local_names:
            continue
        parents = _parent_map(tree)
        found: list[ast.Call] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id in local_names:
                if _local_shadowed(tree, parents, None, func.id, node):
                    issues.append(f"{rel}:{node.lineno}: '{func.id}' is shadowed")
                    continue
                found.append(node)
        if found:
            calls_by_file[rel] = found
    return calls_by_file, issues


def _recipe_add_param(root, request, loaded, source, options, *, cfg,
                      for_write, index):
    symbol = str(request.symbol or "").strip()
    fn, qualname = _resolve_function_qual(source, symbol)
    name = _identifier(options.get("name", ""), "parameter name")
    annotation = ""
    if options.get("annotation"):
        annotation = _parse_annotation(options["annotation"])
    default = (options.get("default") or "").strip()
    if default:
        if "\n" in default or "\r" in default:
            raise _Refuse("option 'default' must be a single-line expression")
        try:
            ast.parse(default, mode="eval")
        except SyntaxError as exc:
            raise _Refuse(f"option 'default' is not a valid expression: {exc.msg}")
    if name in {a.arg for a in _function_args(fn)}:
        raise _Refuse(f"refused: '{name}' is already a parameter of {fn.name}")

    calls, _dynamic, _issues = _find_call_sites(source, fn, qualname)
    xcalls, _xissues = _cross_file_callers(root, loaded.rel, qualname, index,
                                           cfg, for_write)
    all_calls = [c for c, _off in calls]
    all_calls += [c for cs in xcalls.values() for c in cs]

    # A caller that already passes the new name by keyword is a conflict.
    for call in all_calls:
        if any(k.arg == name for k in call.keywords):
            raise _Refuse(
                f"refused: a call at line {call.lineno} already passes "
                f"'{name}=' as a keyword argument; the new parameter would "
                "collide with it")

    total = len(all_calls)
    if not default and total:
        raise _Refuse(
            f"refused: adding a required parameter would break {total} "
            "provable call site(s); pass option default=... so callers keep "
            "working, or update them manually")

    new_text = _insert_param(source, fn, name, annotation, default)
    warnings = [f"added parameter '{name}' to {fn.name}"]
    if default:
        warnings.append(f"{total} provable call site(s) stay valid via the "
                        "default value")
    else:
        warnings.append("no provable call sites; the required parameter is "
                        "safe to add")
    return _single(_Outcome(new_text, _warn(*warnings)), loaded)


def _remove_item_span(source: _Source, start: int, end: int, scope_start: int,
                      scope_end: int) -> tuple[int, int]:
    """The delete span for `[start, end)` folded together with one adjacent
    comma inside `[scope_start, scope_end)`."""
    text = source.text
    commas = [t for t in source.tokens()
              if t.type == tokenize.OP and t.string == ","
              and scope_start <= t.start < scope_end]
    after = [t for t in commas if t.start >= end]
    if after:
        stop = after[0].end
        while stop < scope_end and text[stop] in " \t":
            stop += 1
        return start, stop
    before = [t for t in commas if t.end <= start]
    if before:
        return before[-1].start, end
    return start, end


def _remove_call_arg_span(source: _Source, call: ast.Call,
                          arg: ast.AST) -> tuple[int, int]:
    start, end = source.span(arg)
    cs, ce = source.span(call)
    return _remove_item_span(source, start, end, cs, ce)


def _recipe_remove_param(root, request, loaded, source, options, *, cfg,
                         for_write, index):
    symbol = str(request.symbol or "").strip()
    fn, qualname = _resolve_function_qual(source, symbol)
    name = _identifier(options.get("name", ""), "parameter name")
    if fn.args.vararg is not None or fn.args.kwarg is not None:
        raise _Refuse(
            "refused: the function has *args/**kwargs; the call-site arity is "
            "not statically known, so a parameter cannot be removed "
            "mechanically")
    positional = _positional_params(fn)
    leaf = qualname.rsplit(".", 1)[-1]
    index_in_pos = None
    arg_node = None
    for i, a in enumerate(positional):
        if a.arg == name:
            index_in_pos, arg_node = i, a
            break
    if arg_node is None:
        if any(a.arg == name for a in fn.args.kwonlyargs):
            raise _Refuse(
                f"refused: '{name}' is a keyword-only parameter; call sites "
                "pass it by keyword and cannot be updated mechanically")
        known = ", ".join(a.arg for a in _function_args(fn)) or "(none)"
        raise _Refuse(
            f"refused: '{name}' is not a parameter of {fn.name}; parameters: "
            f"{known}")
    has_default = _arg_has_default(fn, index_in_pos, positional)

    calls, dynamic, issues = _find_call_sites(source, fn, qualname)
    xcalls, xissues = _cross_file_callers(root, loaded.rel, qualname, index,
                                          cfg, for_write)
    if dynamic:
        raise _Refuse(
            f"refused: '{leaf}' is referenced as a value at {dynamic[0]}; that "
            "reference cannot be updated mechanically (the signature changed)")
    problems = list(issues) + list(xissues)
    if problems:
        shown = "; ".join(problems[:3])
        raise _Refuse(
            f"refused: {len(problems)} call site(s) are not provable "
            f"({shown}); removing a parameter needs every caller to be known")

    n_params = len(positional)

    def check_call(call: ast.Call, recv_off: int = 0) -> ast.AST | None:
        err = _call_positional_count_ok(call)
        if err:
            raise _Refuse(
                f"refused: a call at line {call.lineno} uses {err}; the "
                "argument cannot be mapped by position")
        n_args = len(call.args)
        call_idx = index_in_pos - recv_off
        if n_args > call_idx:
            return call.args[call_idx]
        if not has_default:
            raise _Refuse(
                f"refused: a call at line {call.lineno} passes {n_args} "
                f"argument(s) for {n_params - recv_off} parameter(s); the "
                "arity does not match, so the removed argument cannot be "
                "located")
        return None

    edits: list[tuple[int, int, str]] = []
    for call, recv_off in calls:
        target = check_call(call, recv_off)
        if target is not None:
            edits.append((*_remove_call_arg_span(source, call, target), ""))
    open_off = _open_paren_offset(source, fn)
    colon = _function_colon(source, fn)
    aspan = source.span(arg_node)
    edits.append((*_remove_item_span(source, aspan[0], aspan[1], open_off,
                                     colon), ""))
    new_text = _apply_text_edits(source.text, edits)

    changes: list[_FileChange] = []
    changes.append(_FileChange(loaded.rel, loaded.text, new_text,
                               loaded.raw_sha, True))
    removed = 1
    for rel, calls_in_file in sorted(xcalls.items()):
        imp = _load_extra(root, rel, cfg, for_write)
        isrc = _Source(imp.text, _parse_python(imp.text, rel), rel)
        iedits: list[tuple[int, int, str]] = []
        for call in calls_in_file:
            target = check_call(call)
            if target is not None:
                iedits.append((*_remove_call_arg_span(isrc, call, target), ""))
                removed += 1
        if iedits:
            changes.append(_FileChange(rel, imp.text,
                                       _apply_text_edits(imp.text, iedits),
                                       imp.raw_sha, True))
    warnings = [f"removed parameter '{name}' from {fn.name}; updated "
                f"{removed} call site(s)"]
    return _MultiOutcome(tuple(changes), _warn(*warnings))


# ---------------------------------------------------------------------------
# move_symbol / move_file (cross-file)
# ---------------------------------------------------------------------------
def _symbol_block(source: _Source, node: ast.AST
                  ) -> tuple[int, int, int]:
    """(move_start, move_end, remove_end) for a top-level symbol: the block to
    copy (decorators + definition) and the span to delete (plus one blank)."""
    start_line = _statement_start_line(node)
    end_line = node.end_lineno or node.lineno
    start = source.line_start(start_line)
    end = source.line_end(end_line)
    stop = end
    while stop < len(source.text) and source.text[stop:stop + 1] == "\n":
        stop += 1
    return start, end, stop


def _new_specifier(imp_rel: str, dest_rel: str, is_relative: bool) -> str:
    if is_relative:
        spec = relative_specifier(imp_rel, dest_rel)
        if spec is None:
            raise _Refuse(
                "refused: no clean relative import path exists from the "
                "importer to the destination")
        return spec
    return module_name(dest_rel) or dest_rel


def _import_module_span(source: _Source, node: ast.ImportFrom
                        ) -> tuple[int, int] | None:
    start, end = source.span(node)
    tokens = [t for t in source.tokens() if start <= t.start < end]
    from_tok = next((t for t in tokens
                     if t.type == tokenize.NAME and t.string == "from"), None)
    import_tok = next((t for t in tokens
                       if t.type == tokenize.NAME and t.string == "import"
                       and (from_tok is None or t.start > from_tok.start)),
                      None)
    if from_tok is None or import_tok is None:
        return None
    s, e = from_tok.end, import_tok.start
    text = source.text
    while s < e and text[s] in " \t":
        s += 1
    while e > s and text[e - 1] in " \t":
        e -= 1
    return (s, e) if s < e else None


def _import_name_span(source: _Source, node: ast.Import, name: str
                      ) -> tuple[int, int] | None:
    start, end = source.span(node)
    text = source.text
    tokens = [t for t in source.tokens() if start <= t.start < end]
    import_tok = next((t for t in tokens
                       if t.type == tokenize.NAME and t.string == "import"),
                      None)
    if import_tok is None:
        return None
    as_tok = next((t for t in tokens
                   if t.type == tokenize.NAME and t.string == "as"
                   and t.start > import_tok.start), None)
    s, e = import_tok.end, as_tok.start if as_tok is not None else end
    while s < e and text[s] in " \t":
        s += 1
    while e > s and text[e - 1] in " \t":
        e -= 1
    if s >= e or text[s:e] != name:
        return None
    return s, e


def _rewrite_move_imports(loaded: _Loaded, src_rel: str, dest_rel: str,
                          fs: FileSet) -> tuple[str, int]:
    source = _Source(loaded.text, _parse_python(loaded.text, loaded.rel),
                     loaded.rel)
    src_mod = module_name(src_rel)
    edits: list[tuple[int, int, str]] = []
    n = 0
    for node in ast.walk(source.tree):
        if isinstance(node, ast.ImportFrom):
            spec = "." * node.level + (node.module or "")
            try:
                targets = _import_resolve(loaded.rel, spec, fs) if spec else []
            except Exception:
                targets = []
            if src_rel not in targets:
                continue
            if any(a.name == "*" for a in node.names):
                raise _Refuse(f"star import at line {node.lineno}")
            span = _import_module_span(source, node)
            if span is None:
                raise _Refuse(
                    f"could not locate the import specifier at line "
                    f"{node.lineno}")
            edits.append((span[0], span[1],
                          _new_specifier(loaded.rel, dest_rel, node.level > 0)))
            n += 1
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == src_mod:
                    if len(node.names) != 1:
                        raise _Refuse(
                            f"multi-name `import` at line {node.lineno}; "
                            "qualified references cannot be rewritten")
                    if not alias.asname:
                        raise _Refuse(
                            f"bare `import {alias.name}` at line {node.lineno}; "
                            "qualified references cannot be updated "
                            "mechanically")
                    span = _import_name_span(source, node, alias.name)
                    if span is None:
                        raise _Refuse(
                            f"could not locate '{alias.name}' at line "
                            f"{node.lineno}")
                    edits.append((span[0], span[1], module_name(dest_rel)))
                    n += 1
    if not edits:
        return loaded.text, 0
    return _apply_text_edits(source.text, edits), n


def _symbol_importers(root: Path, src_rel: str, symbol: str, index
                      ) -> list[str]:
    if index is not None:
        try:
            from .graph import symbol_importers
            return symbol_importers(index, src_rel, symbol)
        except Exception:
            pass
    fs = FileSet(_iter_py_files(root))
    out: list[str] = []
    for cand in _iter_py_files(root):
        if cand == src_rel:
            continue
        try:
            text = fileio.read_text(root / cand)
        except (OSError, UnicodeDecodeError):
            continue
        tree = _parse_python(text, cand) if text.strip() else ast.parse("")
        for binding in python_import_bindings(text, tree):
            if binding.kind != "from":
                continue
            spec = "." * binding.level + binding.module
            try:
                targets = _import_resolve(cand, spec, fs) if spec else []
            except Exception:
                targets = []
            if src_rel in targets and any(n == symbol
                                          for n, _a in binding.names):
                out.append(cand)
                break
    return sorted(out)


def _rewrite_symbol_import(loaded: _Loaded, src_rel: str, dest_rel: str,
                           symbol: str, fs: FileSet) -> tuple[str, int]:
    source = _Source(loaded.text, _parse_python(loaded.text, loaded.rel),
                     loaded.rel)
    edits: list[tuple[int, int, str]] = []
    insertions: list[tuple[int, str]] = []      # (offset, line)
    n = 0
    for node in ast.walk(source.tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        spec = "." * node.level + (node.module or "")
        try:
            targets = _import_resolve(loaded.rel, spec, fs) if spec else []
        except Exception:
            targets = []
        if src_rel not in targets:
            continue
        names = [a for a in node.names if a.name == symbol]
        if not names:
            continue
        if any(a.name == "*" for a in node.names):
            raise _Refuse(f"star import at line {node.lineno}")
        if len(node.names) == 1 and names[0] is node.names[0]:
            span = _import_module_span(source, node)
            if span is None:
                raise _Refuse(
                    f"could not locate the import specifier at line "
                    f"{node.lineno}")
            edits.append((span[0], span[1],
                          _new_specifier(loaded.rel, dest_rel, node.level > 0)))
            n += 1
            continue
        # multi-name: remove the alias from the source import and add a new
        # from-import of the symbol from the destination.
        got = names[0]
        new_line = (f"from {_new_specifier(loaded.rel, dest_rel, node.level > 0)}"
                    f" import {symbol}")
        if got.asname:
            new_line += f" as {got.asname}"
        new_text = _remove_alias(source, got, node)
        insert_off = source.line_end(node.end_lineno or node.lineno)
        insertions.append((insert_off, new_line))
        edits.append((0, len(source.text), new_text))
        n += 1
    if not edits and not insertions:
        return loaded.text, 0
    text = _apply_text_edits(source.text, edits)
    # insert new lines right-to-left to keep offsets stable
    for offset, line in sorted(insertions, key=lambda t: -t[0]):
        if not line.endswith("\n"):
            line += "\n"
        text = text[:offset] + line + text[offset:]
    return text, n


def _recipe_move_symbol(root, request, loaded, source, options, *, cfg,
                        for_write, index):
    symbol = str(request.symbol or "").strip()
    to_file = (options.get("to_file") or "").strip()
    raw_dest = fileio.normalise_rel(to_file)
    if raw_dest.lower() == loaded.rel.lower() and raw_dest != loaded.rel:
        raise _Refuse("refused: the destination differs only by case; a "
                      "case-only move is rejected")
    dest_path, dest_rel = _resolve_project_path(root, to_file, cfg, for_write)
    if dest_rel == loaded.rel:
        raise _Refuse("refused: source and destination are the same file")
    if dest_rel.lower() == loaded.rel.lower():
        raise _Refuse("refused: the destination differs only by case; a "
                      "case-only move is rejected")
    if not dest_rel.endswith(".py"):
        raise _Refuse("refused: to_file must be a .py file")

    matches = [(q, n) for q, n in _definitions(source.tree) if q == symbol]
    if not matches:
        known = sorted({q for q, _n in _definitions(source.tree)})[:8]
        raise _Refuse(
            f"symbol '{symbol}' was not found as a unique definition"
            + (f"; known: {', '.join(known)}" if known else ""))
    if len(matches) > 1:
        raise _Refuse(f"symbol '{symbol}' is ambiguous")
    qual, node = matches[0]
    if "." in qual:
        raise _Refuse(
            f"refused: '{symbol}' is not a top-level symbol (it is '{qual}'); "
            "move only module-level functions and classes")
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
        raise _Refuse("refused: move_symbol supports top-level functions and "
                      "classes only")

    move_start, move_end, remove_end = _symbol_block(source, node)
    symbol_text = source.text[move_start:move_end]
    new_source = source.text[:move_start] + source.text[remove_end:]

    # destination text (append, or create)
    if dest_path.is_file():
        dest_loaded = _load_extra(root, dest_rel, cfg, for_write)
        dest_before = dest_loaded.text
        dest_text = dest_before
        if dest_text.strip() and not dest_text.endswith("\n"):
            dest_text += "\n"
        if dest_text.strip() and not dest_text.endswith("\n\n"):
            dest_text += "\n"
        dest_after = dest_text + symbol_text
        if not dest_after.endswith("\n"):
            dest_after += "\n"
        dest_existed = True
        dest_sha = dest_loaded.raw_sha
    else:
        dest_before = ""
        dest_after = symbol_text
        if not dest_after.endswith("\n"):
            dest_after += "\n"
        dest_existed = False
        dest_sha = ""

    fs = FileSet(_iter_py_files(root))
    importers = _symbol_importers(root, loaded.rel, qual, index)
    changes: list[_FileChange] = []
    for imp_rel in importers:
        imp = _load_extra(root, imp_rel, cfg, for_write)
        new_imp, _n = _rewrite_symbol_import(imp, loaded.rel, dest_rel, qual,
                                             fs)
        changes.append(_FileChange(imp_rel, imp.text, new_imp, imp.raw_sha,
                                   True))
    changes.append(_FileChange(loaded.rel, loaded.text, new_source,
                               loaded.raw_sha, True))
    if dest_existed:
        changes.append(_FileChange(dest_rel, dest_before, dest_after, dest_sha,
                                   True))
    else:
        changes.append(_FileChange(dest_rel, "", dest_after, "", False))
    return _MultiOutcome(tuple(changes), _warn(
        f"moved '{qual}' from {loaded.rel} to {dest_rel}; updated "
        f"{len(importers)} importer(s)"))


def _recipe_move_file(root, request, loaded, source, options, *, cfg,
                      for_write, index):
    del request, source
    to_path = (options.get("to_path") or "").strip()
    raw_dest = fileio.normalise_rel(to_path)
    if raw_dest.lower() == loaded.rel.lower() and raw_dest != loaded.rel:
        raise _Refuse("refused: the destination differs only by case; a "
                      "case-only move is rejected")
    dest_path, dest_rel = _resolve_project_path(root, to_path, cfg, for_write)
    if dest_rel == loaded.rel:
        raise _Refuse("refused: source and destination are the same path")
    if dest_rel.lower() == loaded.rel.lower():
        raise _Refuse("refused: the destination differs only by case; a "
                      "case-only move is rejected")
    if not dest_rel.endswith(".py"):
        raise _Refuse("refused: to_path must be a .py file")
    if dest_path.exists():
        raise _Refuse(
            f"refused: destination {dest_rel} already exists; nothing was "
            "written")

    fs = FileSet(_iter_py_files(root))
    importers = _importers(root, loaded.rel, index)
    changes: list[_FileChange] = []
    updated = 0
    updated_files = 0
    for imp_rel in importers:
        imp = _load_extra(root, imp_rel, cfg, for_write)
        new_text, n = _rewrite_move_imports(imp, loaded.rel, dest_rel, fs)
        if n:
            changes.append(_FileChange(imp_rel, imp.text, new_text,
                                       imp.raw_sha, True))
            updated += n
            updated_files += 1
    changes.append(_FileChange(dest_rel, "", loaded.text, "", False))
    changes.append(_FileChange(loaded.rel, loaded.text, "",
                               loaded.raw_sha, True, delete=True))
    return _MultiOutcome(tuple(changes), _warn(
        f"moved {loaded.rel} to {dest_rel}; rewrote {updated} import "
        f"statement(s) in {updated_files} file(s)"))


# ---------------------------------------------------------------------------
# Handler registries
# ---------------------------------------------------------------------------
_HANDLERS = {
    "add_import": _recipe_add_import,
    "remove_import": _recipe_remove_import,
    "print_to_logger": _recipe_print_to_logger,
    "extract_constant": _recipe_extract_constant,
    "add_annotation": _recipe_add_annotation,
    "wrap_try": _recipe_wrap_try,
    "wrap_context": _recipe_wrap_context,
    "replace_deprecated": _recipe_replace_deprecated,
}

_MULTI_HANDLERS = {
    "add_param": _recipe_add_param,
    "remove_param": _recipe_remove_param,
    "move_symbol": _recipe_move_symbol,
    "move_file": _recipe_move_file,
    "add_reexport": _recipe_add_reexport,
}
