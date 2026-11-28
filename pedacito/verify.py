"""Deterministic answer verification, a requirement ledger, and candidate scoring.

    findings = check(agent, task, reply, results, gathered)
    errors = [f for f in findings if f.severity == "error"]

The dominant failure of a small model is not bad syntax (the edit path already
refuses that) but a plausible edit to the wrong place: code it never read, a
renamed function whose callers were left behind, a `TODO` standing in for the
work, a requirement from the task's second half that quietly vanished. Every
check here is free -- no model call -- and runs on the DRY-RUN result of the
edit blocks, before anything is written:

  placeholder   the edit introduces `TODO` / `NotImplementedError` / a bare
                `pass` body the task did not ask for
  unread        the edit changes a file (or a region of it) the model never
                read: it is guessing, not editing
  stale refs    a definition was removed or renamed but is still referenced
  moved import  a moved file is still imported by its old path
  lost wiring   a whole-file replacement dropped imported-and-called names
                that are still defined in the project
  gui hazard    a Python GUI/game edit brings an import-time mainloop(), a
                pack/grid collision, or an uninstalled third-party GUI
                import (the first two hang or crash headless verification)
  citations     the prose cites `file:line` in files that were never read
  coverage      requirements extracted from the task that no edit or sentence
                of the reply touches (a warning: heuristic); each requirement
                also carries an evidence state (UNLOCATED -> LOCATED -> READ
                -> EDITED -> VERIFIED, or READ -> REPORTED for investigations)

That coverage ledger is a run-scoped state machine
(`RequirementTracker`): search results, displayed reads and clean patches move
each clause forward, and the agent uses the oldest unmet clause to pick its
next deterministic lookup. No transition costs a model call.

Errors feed one bounded repair round; warnings are shown and logged. A model
verifier (`verify_answer = "full"`) is one extra call, used only when the
deterministic pass found nothing to fix.

Two hooks hand source back WITHOUT ever
rewriting history (TECHNIQUES.md section 11): `find_needs` parses the
explicit `NEED path/to/file.py:120-168` directive from an answer, repair or
verifier reply, and `evidence_rehydrations` selects the unread / fuzzy
findings whose flagged region was recorded in the agent's EvidenceStore (or
sits within ±30 lines of a recorded span) so the repair prompt can append
those exact lines from disk at its tail.
"""

from __future__ import annotations

import ast
import copy
import difflib
import json
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from . import fileio
from .edits import (StructuredEdit, StructuredIntent, locate,
                    materialize_structured_edits, parse_edits)


@dataclass
class Finding:
    kind: str                  # placeholder | unread | stale | import | lost_wiring | gui_hazard | citation | coverage | model | lsp
    severity: str              # "error" (fed to repair) | "warn"
    file: str
    message: str
    line: int = 0
    # The 1-based inclusive (start, end) region the finding flags, when the
    # check knows one (unread regions, fuzzy-anchored blocks). The
    # repair auto-rehydration re-fetches exactly these lines.
    lines: tuple = ()

    def render(self) -> str:
        loc = f"{self.file}:{self.line}" if self.line else self.file
        return f"[{self.severity}] {self.kind}: {loc} -- {self.message}"


# ------------------------------------------------------- immediate syntax gate
# A candidate patch is parsed in memory BEFORE any linter or test subprocess is
# started. Python uses `ast.parse`; every other language chunker_ts knows uses
# its tree-sitter grammar. The check compares the dry-applied `after` text with
# the original `before` text, so a file that was already broken stays editable
# (the edit path's recovery case) while a NEW syntax error rejects the patch in
# microseconds. The repair prompt gets the parser diagnostic plus a single
# 10-line window, never the whole file or a subprocess transcript.
_SYNTAX_WINDOW_LINES = 10


def _file_suffix(rel: str) -> str:
    """Lowercase extension of a path without importing Path for every check."""
    name = str(rel or "").replace("\\", "/").rsplit("/", 1)[-1]
    return ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""


def _syntax_problem(rel: str, text: str) -> tuple[bool, int | None, str]:
    """(has_error, 1-based line, parser diagnostic) for `text` as `rel`'s
    language. Unsupported extensions and unavailable parsers are not errors."""
    suffix = _file_suffix(rel)
    body = text if isinstance(text, str) else ""
    if suffix == ".py":
        try:
            ast.parse(body)
        except SyntaxError as e:
            line = int(e.lineno or 0) or None
            col = int(e.offset or 0)
            return (True, line,
                    f"ast.parse: {e.msg} (line {line or '?'}"
                    + (f", column {col}" if col else "") + ")")
        except (ValueError, TypeError) as e:      # null bytes / bad source type
            return True, None, f"ast.parse: {type(e).__name__}: {e}"
        return False, None, ""

    try:
        from .chunker_ts import LANG_BY_EXT, TS_AVAILABLE, has_syntax_error
    except Exception:                              # pragma: no cover - import guard
        return False, None, ""
    language = LANG_BY_EXT.get(suffix)
    if not language or not TS_AVAILABLE:
        return False, None, ""
    try:
        bad, line = has_syntax_error(body, language)
    except Exception:                              # unsupported/broken grammar
        return False, None, ""
    if not bad:
        return False, None, ""
    line = int(line or 0) or None
    return (True, line,
            f"tree-sitter {language}: ERROR node near line {line or '?'}")


def _syntax_window(text: str, line: int | None,
                   max_lines: int = _SYNTAX_WINDOW_LINES
                   ) -> tuple[str, int, int]:
    """(numbered window, first line, last line) around `line`; at most
    `max_lines` source lines, with the error line marked by `>>`."""
    lines = str(text or "").split("\n")
    if lines and lines[-1] == "" and len(lines) > 1:
        lines.pop()
    if not lines:
        return "", 0, 0
    err = max(1, min(int(line or 1), len(lines)))
    before = max(1, max_lines // 2)
    start = max(1, err - before)
    end = min(len(lines), start + max_lines - 1)
    start = max(1, end - max_lines + 1)
    shown = []
    for i in range(start, end + 1):
        mark = ">>" if i == err else "  "
        shown.append(f"{mark}{i:>4}| {lines[i - 1]}")
    return "\n".join(shown), start, end


def check_syntax(results) -> list[Finding]:
    """New syntax errors in dry-applied results, checked in-process.

    `Result.before`/`Result.after` are the in-memory buffers before and after a
    file's blocks. A parse failure that already existed in `before` is not a
    new failure (repairing a broken file must stay possible); only a transition
    from parseable to unparseable rejects the patch. Each finding carries the
    parser diagnostic and exactly the error's 10-line window, so callers can
    feed repair without running a linter or test subprocess first."""
    out: list[Finding] = []
    for r in results or ():
        if not getattr(r, "ok", False) or not getattr(r, "file", ""):
            continue
        after = getattr(r, "after", None)
        if after is None:
            continue
        before = getattr(r, "before", None)
        if before is None:
            before = ""
        was_bad, _, _ = _syntax_problem(r.file, before)
        if was_bad:
            continue
        is_bad, line, diagnostic = _syntax_problem(r.file, after)
        if not is_bad:
            continue
        window, start, end = _syntax_window(after, line)
        message = (
            f"this edit introduces a syntax error at {r.file}:"
            f"{line or '?'}; {diagnostic}. No lint or test process was run. "
            f"Fix the syntax first; only the {_SYNTAX_WINDOW_LINES}-line "
            f"parser window is shown:\n"
            f"lines {start}-{end}:\n{window}")
        out.append(Finding("syntax", "error", r.file, message,
                           line=line or 0, lines=(start, end)))
    return out


# Readable alias for callers that describe the result rather than the check.
syntax_findings = check_syntax


def _new_static_errors(rel: str, after: str, before: str) -> list[str]:
    """Static errors in `after` that were not already present in `before`."""
    from . import spec as spec_mod
    errs = [e for e in spec_mod.ast_static_errors(rel, after)
            if ": E999 " not in e]
    if not errs:
        return []
    base_keys = {spec_mod.static_error_key(e)
                 for e in spec_mod.ast_static_errors(rel, before)}
    return [e for e in errs if spec_mod.static_error_key(e) not in base_keys]


def auto_heal_static(results, root=None, cfg=None,
                     persist: bool = False) -> list[str]:
    """Deterministically insert unambiguous missing imports into dry-applied
    Python buffers (TECHNIQUES.md #19).

    Only NEW F821 names are candidates -- the same `before`-vs-`after`
    baseline `check_static` uses -- and `spec.auto_heal_source` re-parses and
    re-runs the static resolver before accepting an insertion. Each result's
    `after` buffer is updated in place, so the caller's checks see the healed
    source. With `persist=True`, a buffer that is byte-identical to the file
    on disk is written back (this is how `agent.apply` makes the fix real
    after the edit has landed); otherwise nothing is written.

    Returns the exact one-line receipts, or [] when nothing was healed."""
    from . import spec as spec_mod
    if not bool(getattr(cfg, "static_gate", True)) \
            or not bool(getattr(cfg, "auto_heal_imports", True)):
        return []
    receipts: list[str] = []
    for r in results or ():
        if not getattr(r, "ok", False) or not getattr(r, "file", ""):
            continue
        if not str(r.file).lower().endswith(".py"):
            continue
        after = getattr(r, "after", None)
        if after is None:
            continue
        before = getattr(r, "before", None) or ""
        new = _new_static_errors(r.file, after, before)
        names = spec_mod.f821_names(new)
        if not names:
            continue
        healed, adds = spec_mod.auto_heal_source(
            r.file, after, Path(root) if root is not None else None, cfg,
            names=names)
        if not adds:
            continue
        # Belt and braces: the healed buffer must resolve every name it
        # touched and introduce no error the original buffer did not have.
        remaining = _new_static_errors(r.file, healed, before)
        if any(n in spec_mod.f821_names(remaining) for n in names):
            continue
        if persist:
            # A persisted heal must describe bytes that are actually on disk.
            # If the file moved under us, leave the buffer and the receipt
            # untouched so the normal repair path still sees the real error.
            if root is None:
                continue
            path = Path(root) / fileio.normalise_rel(r.file)
            try:
                current = fileio.read_text(path)
            except OSError:
                continue
            if current != after:          # disk is not the buffer we healed
                continue
            try:
                _text, style = fileio.read(path)
                fileio.write(path, healed, style)
            except OSError:
                continue
        r.after = healed
        if getattr(r, "diff", ""):
            r.diff = "".join(difflib.unified_diff(
                before.splitlines(keepends=True),
                healed.splitlines(keepends=True),
                fromfile=f"a/{r.file}", tofile=f"b/{r.file}"))
        receipts.extend(adds)
    return receipts


def check_static(results, root=None, cfg=None) -> list[Finding]:
    """New undefined-name / `__all__` / use-before-binding errors in the
    dry-applied buffers, checked in-process with the pure-AST static gate
    (TECHNIQUES.md #19) -- no model tokens, no linter subprocess.

    Like the syntax gate this compares the `after` buffer against the file's
    `before` buffer, so errors the file already had stay quiet and a repair
    round is never spent on pre-existing problems. Before an F821 error is
    returned, unambiguous missing imports are auto-healed in memory (see
    `auto_heal_static`); only what remains reaches the model repair round.
    Only Python files are considered; syntax errors belong to `check_syntax`
    (whose verdict also makes any static verdict on the same buffer
    unreliable)."""
    out: list[Finding] = []
    for r in results or ():
        if not getattr(r, "ok", False) or not getattr(r, "file", ""):
            continue
        if not str(r.file).lower().endswith(".py"):
            continue
        after = getattr(r, "after", None)
        if after is None:
            continue
        before = getattr(r, "before", None) or ""
        new = _new_static_errors(r.file, after, before)
        if not new:
            continue
        receipts = auto_heal_static([r], root=root, cfg=cfg)
        if receipts:
            out.append(Finding("auto_heal", "warn", r.file,
                               "\n".join(receipts)))
            new = _new_static_errors(r.file, r.after, before)
        if not new:
            continue
        m = re.match(r"^.+?:(\d+):\d+: ", new[0])
        line = int(m.group(1)) if m else 0
        out.append(Finding(
            "static", "error", r.file,
            "this edit introduces names that do not exist at the point they "
            "are used (a missing import, a typo, or a bad `__all__` entry); "
            "no linter or test process was run. Fix these before anything "
            "else:\n" + "\n".join(f"- {e}" for e in new[:15]),
            line=line))
    return out


# ------------------------------------------- JS lost-import gate (BUG-310)
# The Python static gate above has no JS analogue, so a hunk that deleted a
# file's `node:test` / `node:assert` imports while still calling describe /
# test / strictEqual passed "applied and verified" and only died at load time.
# This cheap, deterministic check compares the import bindings of the before
# and after buffers: a name that was imported, is no longer imported, and is
# still referenced (and not re-declared) in the after buffer is an undefined
# name the edit just introduced. No scope analysis, no subprocess.
_JS_EXTS = (".js", ".mjs", ".cjs", ".jsx")
_JS_NAMED_IMPORT_RE = re.compile(r"import\s*\{([^}]*)\}\s*from", re.S)
_JS_DEFAULT_IMPORT_RE = re.compile(
    r"import\s+([\w$]+)\s*(?:,\s*\{[^}]*\})?\s*from", re.S)
_JS_NS_IMPORT_RE = re.compile(r"import\s*\*\s*as\s+([\w$]+)\s+from", re.S)
_JS_NAMED_REQUIRE_RE = re.compile(
    r"(?:const|let|var)\s*\{([^}]*)\}\s*=\s*require\s*\(", re.S)
_JS_DEFAULT_REQUIRE_RE = re.compile(
    r"(?:const|let|var)\s+([\w$]+)\s*=\s*require\s*\(", re.S)


def _js_binding_names(part: str) -> list[str]:
    out = []
    for piece in (part or "").split(","):
        piece = piece.strip()
        if not piece:
            continue
        if piece.startswith("type "):
            piece = piece[5:].strip()
        if " as " in piece:
            piece = piece.split(" as ")[-1].strip()
        m = re.match(r"[\w$]+", piece)
        if m:
            out.append(m.group(0))
    return out


def _js_imported_names(text: str) -> set[str]:
    names: set[str] = set()
    for m in _JS_NAMED_IMPORT_RE.finditer(text or ""):
        names.update(_js_binding_names(m.group(1)))
    for m in _JS_DEFAULT_IMPORT_RE.finditer(text or ""):
        names.add(m.group(1))
    for m in _JS_NS_IMPORT_RE.finditer(text or ""):
        names.add(m.group(1))
    for m in _JS_NAMED_REQUIRE_RE.finditer(text or ""):
        names.update(_js_binding_names(m.group(1)))
    for m in _JS_DEFAULT_REQUIRE_RE.finditer(text or ""):
        names.add(m.group(1))
    return names


def check_static_js(results) -> list[Finding]:
    """New undefined names from a JS edit that dropped an import it still uses
    (BUG-310). Compares before/after import bindings; only names still used and
    not re-declared in the after buffer are reported."""
    out: list[Finding] = []
    for r in results or ():
        if not getattr(r, "ok", False) or not getattr(r, "file", ""):
            continue
        if not str(r.file).lower().endswith(_JS_EXTS):
            continue
        after = getattr(r, "after", None)
        before = getattr(r, "before", None)
        if after is None or before is None or before == after:
            continue
        lost = _js_imported_names(before) - _js_imported_names(after)
        if not lost:
            continue
        flagged = []
        for name in sorted(lost):
            use = re.search(r"(?<![\w$])" + re.escape(name) + r"(?![\w$])",
                            after)
            if not use:
                continue
            if re.search(r"\b(?:function|class)\s+" + re.escape(name) + r"\b",
                         after) or re.search(
                r"\b(?:const|let|var)\s+" + re.escape(name) + r"\b", after):
                continue
            line = after.count("\n", 0, use.start()) + 1
            flagged.append((name, line))
        if not flagged:
            continue
        first_line = min(ln for _n, ln in flagged)
        out.append(Finding(
            "static", "error", r.file,
            "this edit removes import(s) whose names are still used, so they "
            "are now undefined at runtime (no linter or test process was "
            "run). Restore the import or remove the usage:\n"
            + "\n".join(f"- `{n}` (used near line {ln})"
                        for n, ln in flagged[:15]),
            line=first_line))
    return out


# ------------------------------------------- empty JS test files (item 10)
# `node --test` exits 0 on a file that defines no test cases: a "test" file
# of helpers, constants or commented-out cases reads as a green suite while
# verifying nothing. Count real invocations only -- `function test(` is a
# helper declaration and `regex.test(` a method call, neither is a case.
_JS_TEST_EXTS = (".js", ".mjs", ".cjs", ".jsx", ".ts", ".mts", ".cts", ".tsx")
_JS_TEST_CALL_RE = re.compile(
    r"(?<![\w$.])(?:test|it|describe|suite)"
    r"(?:\.(?:only|skip|todo|concurrent))*\s*\(")
_JS_TEST_DECL_RE = re.compile(r"\bfunction\s+(?:test|it|describe|suite)\s*\(")


def _is_js_test_file(rel: str) -> bool:
    """A JS/TS file named or placed like a test (`*.test.js`, `*.spec.ts`,
    or under a test/tests/__tests__ directory)."""
    low = str(rel or "").replace("\\", "/").lower()
    if not low.endswith(_JS_TEST_EXTS):
        return False
    base = low.rsplit("/", 1)[-1]
    stem = base.rsplit(".", 1)[0]
    if stem.endswith((".test", ".spec")):
        return True
    return any(p in ("test", "tests", "__tests__")
               for p in low.split("/")[:-1])


def check_empty_js_tests(results) -> list[Finding]:
    """Warn on an edited JS/TS test file whose new content defines zero test
    cases: the runner reports 0 tests and passes, so the file verifies
    nothing. A warning, not an error -- naming conventions are heuristic."""
    out: list[Finding] = []
    for r in results or ():
        if not getattr(r, "ok", False) or not getattr(r, "file", ""):
            continue
        if not _is_js_test_file(r.file):
            continue
        after = getattr(r, "after", None)
        before = getattr(r, "before", None)
        if after is None or before == after:
            continue
        body = re.sub(r"/\*.*?\*/", " ", after, flags=re.S)
        body = re.sub(r"//[^\n]*", " ", body)
        body = _JS_TEST_DECL_RE.sub(" ", body)
        if _JS_TEST_CALL_RE.search(body):
            continue
        out.append(Finding(
            "empty_tests", "warn", r.file,
            "this test file defines no test cases (no `test(`/`it(`/"
            "`describe(`/`suite(` invocation): `node --test` reports 0 "
            "tests and exits 0, so it verifies nothing. Add real test "
            "cases for the behavior it names."))
    return out


# --------------------------------------- desktop GUI / game hazards
# A tkinter calculator or a pygame arcade ships import-time traps that hang
# or crash headless verification: a top-level mainloop() (pytest never
# returns), pack/grid mixed in one container (TclError at runtime), and a
# third-party GUI import that is not installed (collection fails). All three
# are cheap AST checks on the dry-applied buffer; only NEW hazards are
# reported, so a file that already had them stays editable.

_GUI_GAME_IMPORTS = frozenset({
    "PyQt5", "PyQt6", "PySide2", "PySide6", "customtkinter", "wx", "kivy",
    "pygame", "arcade",
})
_TRY_NODES = (ast.Try,) + ((ast.TryStar,) if hasattr(ast, "TryStar") else ())


def _is_main_guard(node) -> bool:
    """True for `if __name__ == "__main__":` (either operand order)."""
    if not isinstance(node, ast.If):
        return False
    t = node.test
    if not (isinstance(t, ast.Compare) and len(t.ops) == 1
            and isinstance(t.ops[0], ast.Eq)):
        return False
    for a, b in ((t.left, t.comparators[0]), (t.comparators[0], t.left)):
        if (isinstance(a, ast.Name) and a.id == "__name__"
                and isinstance(b, ast.Constant) and b.value == "__main__"):
            return True
    return False


def _toplevel_mainloops(tree) -> list[int]:
    """Lines of .mainloop() calls that run at import time: module-level
    statements outside any function/class definition and outside an
    `if __name__ == "__main__":` body. A call inside a def is fine -- it
    only runs when someone calls the def."""
    hits: list[int] = []

    def stmts(body, guarded):
        for s in body:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef)):
                continue
            if isinstance(s, ast.If):
                stmts(s.body, guarded or _is_main_guard(s))
                stmts(s.orelse, guarded)      # the else runs at import
                continue
            if isinstance(s, _TRY_NODES):
                stmts(s.body, guarded)
                for h in s.handlers:
                    stmts(h.body, guarded)
                stmts(s.orelse, guarded)
                stmts(s.finalbody, guarded)
                continue
            if isinstance(s, (ast.With, ast.AsyncWith, ast.For,
                              ast.AsyncFor, ast.While)):
                stmts(s.body, guarded)
                stmts(getattr(s, "orelse", []), guarded)
                continue
            if guarded:
                continue
            for n in _walk_scope(s):
                if (isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute)
                        and n.func.attr == "mainloop"):
                    hits.append(n.lineno)

    stmts(tree.body, False)
    return hits


def _walk_scope(scope):
    """The nodes of one function/module scope (or one statement), without
    descending into nested function/class/lambda bodies or generator
    expressions -- those run later, not at import time."""
    todo = list(ast.iter_child_nodes(scope))
    while todo:
        n = todo.pop()
        yield n
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                          ast.ClassDef, ast.Lambda, ast.GeneratorExp)):
            continue
        todo.extend(ast.iter_child_nodes(n))


def _geometry_collisions(tree) -> list[tuple[int, str]]:
    """(line, message) where one scope manages widgets with BOTH pack and
    grid: the same widget variable calling both, or two children of the
    same parent container split between the managers. Either crashes with
    _tkinter.TclError at runtime."""
    out: list[tuple[int, str]] = []
    scopes = [tree] + [n for n in ast.walk(tree)
                       if isinstance(n, (ast.FunctionDef,
                                         ast.AsyncFunctionDef))]
    for scope in scopes:
        nodes = list(_walk_scope(scope))
        # widget variable -> parent variable, from `w = Widget(parent, ...)`
        parent: dict[str, str] = {}
        for n in nodes:
            if not (isinstance(n, ast.Assign)
                    and isinstance(n.value, ast.Call)):
                continue
            call = n.value
            fname = (call.func.attr if isinstance(call.func, ast.Attribute)
                     else call.func.id if isinstance(call.func, ast.Name)
                     else "")
            if (fname[:1].isupper() and call.args
                    and isinstance(call.args[0], ast.Name)):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        parent[t.id] = call.args[0].id
        # widget variable -> {manager: first line}
        mgr: dict[str, dict[str, int]] = {}
        for n in nodes:
            if (isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr in ("pack", "grid")
                    and isinstance(n.func.value, ast.Name)):
                mgr.setdefault(n.func.value.id, {}).setdefault(
                    n.func.attr, n.lineno)
        for var, used in sorted(mgr.items()):
            if "pack" in used and "grid" in used:
                out.append((max(used["pack"], used["grid"]), (
                    f"widget '{var}' calls both .pack() (line "
                    f"{used['pack']}) and .grid() (line {used['grid']}): "
                    "tkinter raises TclError -- one geometry manager per "
                    "widget.")))
        by_parent: dict[str, dict[str, tuple[str, int]]] = {}
        for var, used in mgr.items():
            p = parent.get(var)
            if not p:
                continue
            for method, line in used.items():
                by_parent.setdefault(p, {}).setdefault(method, (var, line))
        for p, used in sorted(by_parent.items()):
            if "pack" not in used or "grid" not in used:
                continue
            (v1, l1), (v2, l2) = used["pack"], used["grid"]
            if v1 == v2:
                continue        # the same-widget case already flagged it
            out.append((max(l1, l2), (
                f"children of '{p}' mix geometry managers: '{v1}' packs "
                f"(line {l1}), '{v2}' grids (line {l2}) -- TclError: cannot "
                "use grid inside a container already managed by pack. Pick "
                "ONE manager per container.")))
    return out


def _missing_gui_imports(tree, path, root=None,
                         task: str = "") -> list[tuple[int, str]]:
    """(line, library) for third-party GUI/game imports that are neither
    installed in this interpreter nor declared in the project's
    requirements.txt / pyproject.toml, and that the task did not ask for."""
    wanted: dict[str, int] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names = [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            names = [n.module]
        else:
            continue
        for name in names:
            top = name.split(".")[0]
            if top in _GUI_GAME_IMPORTS:
                wanted.setdefault(top, n.lineno)
    if not wanted:
        return []
    dirs = {Path(path or ".").parent}
    if root:
        dirs.add(Path(root))
    declared = ""
    for d in dirs:
        for name in ("requirements.txt", "pyproject.toml"):
            try:
                declared += (d / name).read_text(
                    encoding="utf-8").lower() + "\n"
            except OSError:
                pass
    task_l = (task or "").lower()
    out: list[tuple[int, str]] = []
    import importlib.util
    for lib, line in sorted(wanted.items(), key=lambda kv: kv[1]):
        if re.search(rf"\b{re.escape(lib.lower())}\b", task_l):
            continue                    # the user asked for this library
        if lib.lower() in declared:
            continue                    # the project intends to install it
        try:
            if importlib.util.find_spec(lib) is not None:
                continue
        except Exception:
            continue                    # uncertain: never flag on a maybe
        out.append((line, lib))
    return out


_TK_VAR_ATTRS = frozenset({"StringVar", "IntVar", "BooleanVar", "DoubleVar"})


def _handles_import_error(handler) -> bool:
    """True for `except ImportError:` / `except ModuleNotFoundError:` (also
    in a tuple, also a bare `except:` -- it catches ImportError too)."""
    t = handler.type
    if t is None:
        return True
    names = t.elts if isinstance(t, ast.Tuple) else [t]
    for n in names:
        if isinstance(n, ast.Name) and n.id in ("ImportError",
                                                "ModuleNotFoundError"):
            return True
        if isinstance(n, ast.Attribute) and n.attr in ("ImportError",
                                                       "ModuleNotFoundError"):
            return True
    return False


def _tk_none_aliases(tree) -> set[str]:
    """Names set to None inside an `except ImportError:` handler -- the
    headless fallback pattern (`try: import tkinter as tk ... except
    ImportError: tk = None`). On a box without libtk the name really is
    None at runtime, so every attribute access on it must be guarded."""
    aliases: set[str] = set()
    for n in ast.walk(tree):
        if not isinstance(n, _TRY_NODES):
            continue
        for h in n.handlers:
            if not _handles_import_error(h):
                continue
            for s in h.body:
                if not (isinstance(s, ast.Assign)
                        and isinstance(s.value, ast.Constant)
                        and s.value.value is None):
                    continue
                for t in s.targets:
                    if isinstance(t, ast.Name):
                        aliases.add(t.id)
    return aliases


def _tk_guard_state(test, aliases: set[str]):
    """What an `if` test proves about a tk alias: 'present' when the body
    only runs with the alias non-None (`if tk is not None:` / `if tk:`),
    'absent' when only the ELSE does (`if tk is None:`), None otherwise."""
    if isinstance(test, ast.Name) and test.id in aliases:
        return "present"
    if (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)
            and isinstance(test.operand, ast.Name)
            and test.operand.id in aliases):
        return "absent"
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1):
        return None
    op = test.ops[0]
    sides = (test.left, test.comparators[0])
    for a, b in (sides, sides[::-1]):
        if (isinstance(a, ast.Name) and a.id in aliases
                and isinstance(b, ast.Constant) and b.value is None):
            if isinstance(op, (ast.IsNot, ast.NotEq)):
                return "present"
            if isinstance(op, (ast.Is, ast.Eq)):
                return "absent"
    return None


def _tk_none_var_hazards(tree, aliases: set[str]) -> list[tuple[int, str, str]]:
    """(line, alias, attr) for tk.StringVar/IntVar/BooleanVar/DoubleVar calls
    inside __init__ that run unguarded even though the module sets the alias
    to None on ImportError -- headless verification crashes with
    AttributeError: 'NoneType' object has no attribute 'StringVar'."""
    out: list[tuple[int, str, str]] = []

    def expr(n, guarded):
        # Deferred bodies (lambda/generator) do not run during __init__.
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                          ast.ClassDef, ast.Lambda, ast.GeneratorExp)):
            return
        if isinstance(n, ast.IfExp):    # `tk.StringVar() if tk else holder`
            state = _tk_guard_state(n.test, aliases)
            expr(n.body, guarded or state == "present")
            expr(n.orelse, guarded or state == "absent")
            return
        if (isinstance(n, ast.Call) and not guarded
                and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id in aliases
                and n.func.attr in _TK_VAR_ATTRS):
            out.append((n.lineno, n.func.value.id, n.func.attr))
        for c in ast.iter_child_nodes(n):
            expr(c, guarded)

    def scan(body, guarded):
        for s in body:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef)):
                continue        # runs later, not during __init__
            if isinstance(s, ast.If):
                state = _tk_guard_state(s.test, aliases)
                scan(s.body, guarded or state == "present")
                scan(s.orelse, guarded or state == "absent")
                continue
            if isinstance(s, _TRY_NODES):
                scan(s.body, guarded)
                for h in s.handlers:
                    scan(h.body, guarded)
                scan(s.orelse, guarded)
                scan(s.finalbody, guarded)
                continue
            if isinstance(s, (ast.With, ast.AsyncWith, ast.For,
                              ast.AsyncFor, ast.While)):
                scan(s.body, guarded)
                scan(getattr(s, "orelse", []), guarded)
                continue
            expr(s, guarded)

    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        for fn in cls.body:
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and fn.name == "__init__":
                scan(fn.body, False)
    return out


def check_python_gui_hazards(path, content, *, task: str = "",
                             root=None) -> list[str]:
    """Static hazard notes for one Python file's text (`path` only locates
    the project's dependency manifests):

    - a .mainloop() that runs at import time -- it hangs pytest and every
      verification import indefinitely;
    - pack/grid mixed on one widget or one parent's children -- TclError;
    - a tk.StringVar/IntVar/BooleanVar/DoubleVar call in __init__ that is
      not guarded by `if tk is not None:` even though the module sets
      tk = None on ImportError -- AttributeError on headless boxes;
    - a third-party GUI/game import that is not installed, not declared in
      requirements.txt/pyproject.toml and not asked for by the task.

    Each note starts with `line N:`. A buffer that does not parse is left
    to the syntax gate. The optional `task`/`root` sharpen the import
    check (a requested or declared library is never flagged)."""
    text = content if isinstance(content, str) else ""
    if not text.strip():
        return []
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, TypeError):
        return []
    out: list[str] = []
    for line in _toplevel_mainloops(tree):
        out.append(
            f"line {line}: top-level .mainloop() runs at import time and "
            "hangs pytest/verification indefinitely. Move it under "
            "if __name__ == \"__main__\": and expose a testable class "
            "(app.press(key) / game.step(dt)) headless tests can drive.")
    for line, msg in _geometry_collisions(tree):
        out.append(f"line {line}: {msg}")
    aliases = _tk_none_aliases(tree)
    if aliases:
        for line, name, attr in _tk_none_var_hazards(tree, aliases):
            out.append(
                f"line {line}: {name}.{attr}() runs in __init__ without an "
                f"`if {name} is not None:` guard, but this module sets "
                f"{name} = None on ImportError -- headless runs crash with "
                f"AttributeError: 'NoneType' object has no attribute "
                f"'{attr}'. Guard the call, or use a tiny .get()/.set() "
                f"holder when {name} is None.")
    for line, lib in _missing_gui_imports(tree, path, root=root, task=task):
        out.append(
            f"line {line}: imports '{lib}', which is not installed in this "
            "Python environment and not declared in requirements.txt/"
            "pyproject.toml. Prefer stdlib tkinter (tk.Canvas / styled "
            f"ttk) or a zero-build HTML5 <canvas> unless the task asks "
            f"for {lib}.")
    return out


def check_gui_hazards(results, root=None, task: str = "") -> list[Finding]:
    """Findings for NEW GUI/game hazards in the dry-applied Python buffers
    (check_python_gui_hazards): an import-time event loop or a pack/grid
    collision is an error (it hangs or crashes headless verification), an
    uninstalled third-party GUI import a warning nudging toward stdlib
    tkinter or browser Canvas. Hazards the file already had stay quiet,
    like the static gate."""
    out: list[Finding] = []
    for r in results or ():
        if not getattr(r, "ok", False):
            continue
        if not str(getattr(r, "file", "")).lower().endswith(".py"):
            continue
        after = getattr(r, "after", None)
        if after is None:
            continue
        before = getattr(r, "before", None) or ""
        path = (Path(root) / str(r.file)) if root else r.file
        old = {re.sub(r"^line \d+: ", "", m)
               for m in check_python_gui_hazards(path, before, task=task,
                                                 root=root)}
        for msg in check_python_gui_hazards(path, after, task=task,
                                            root=root):
            if re.sub(r"^line \d+: ", "", msg) in old:
                continue
            m = re.match(r"^line (\d+):", msg)
            severity = "warn" if "which is not installed" in msg else "error"
            out.append(Finding("gui_hazard", severity, r.file, msg,
                               line=int(m.group(1)) if m else 0))
    return out


# ------------------------------------------- README grounding (BUG-307)
# A generated README is a reference file, so no static/import gate reads it.
# One run shipped "# Pedacito" claiming "Tests are written in Jest" while
# package.json said `node --test` and the project was PulseForge-16. This cheap
# check compares an edited README against the project manifest: a title that
# shares nothing with the package name, or a test framework that contradicts
# scripts.test, is a warning (never an error -- prose is heuristic).
_RUNNER_WORDS = {
    "node": ("node --test", "node:test"),
    "jest": ("jest",),
    "mocha": ("mocha",),
    "vitest": ("vitest",),
    "pytest": ("pytest", "py.test"),
    "ava": ("ava",),
}


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def check_doc_grounding(agent, results) -> list[Finding]:
    root = getattr(getattr(agent, "index", None), "root", None)
    if root is None:
        return []
    try:
        pkg = json.loads((Path(root) / "package.json").read_text(
            encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(pkg, dict):
        return []
    name = str(pkg.get("name") or "")
    scripts = pkg.get("scripts") or {}
    test_cmd = str(scripts.get("test") or "") if isinstance(scripts, dict) else ""
    actual = next((k for k, words in _RUNNER_WORDS.items()
                   if any(w in test_cmd for w in words)), "")
    out: list[Finding] = []
    for r in results or ():
        if not getattr(r, "ok", False):
            continue
        rel = str(getattr(r, "file", "") or "")
        if rel.rsplit("/", 1)[-1].lower() not in ("readme.md", "readme.rst",
                                                  "readme.txt", "readme"):
            continue
        after = getattr(r, "after", None)
        if not after:
            continue
        problems = []
        m = re.search(r"(?m)^#\s+(.+?)\s*$", after)
        if m and name:
            title = _norm_name(m.group(1))
            want = _norm_name(name)
            if title and want and want not in title and title not in want:
                problems.append(
                    f"the title '{m.group(1)}' does not match the package "
                    f"name '{name}'")
        if actual:
            low = after.lower()
            mentions_actual = any(w in low for w in _RUNNER_WORDS[actual])
            for other, words in _RUNNER_WORDS.items():
                if other == actual or mentions_actual:
                    continue
                if any(re.search(r"(?<![\w])" + re.escape(w) + r"(?![\w])",
                                 low) for w in words):
                    problems.append(
                        f"it names the '{other}' test framework but "
                        f"package.json scripts.test uses '{actual}' "
                        f"({test_cmd})")
                    break
        if problems:
            out.append(Finding(
                "grounding", "warn", rel,
                "the README contradicts the project manifest: "
                + "; ".join(problems)))
    return out


# ------------------------------------------------------------ placeholders
_PLACEHOLDER_RES = (
    re.compile(r"\braise\s+NotImplementedError\b"),
    re.compile(r"\bthrow\s+new\s+(?:NotImplemented|UnsupportedOperation)\w*"),
    re.compile(r"\btodo!\s*\(|\bunimplemented!\s*\("),
    re.compile(r"(?i)(?:#|//|/\*|--)\s*(?:TODO|FIXME|implement me|to be implemented)\b"),
    re.compile(r"(?m)^\s*\.\.\.\s*$"),
)
_PASS_RE = re.compile(r"(?m)^\s*pass\s*(?:#.*)?$")
_DEF_RE = re.compile(r"(?m)^\s*(?:async\s+)?(?:def|class)\s+[A-Za-z_]")
_STUB_OK_RE = re.compile(
    r"(?i)\b(stub|placeholder|skeleton|scaffold|todo|not implemented|"
    r"unimplemented|abstract|interface|protocol)\b")


_NOOP_FUNC_NAMES = frozenset(("noop", "_noop", "log_message"))


def _is_bare_pass_or_ellipsis(body: list) -> bool:
    if body and isinstance(body[0], ast.Expr) \
            and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]                 # drop the docstring
    if len(body) != 1:
        return False
    stmt = body[0]
    if isinstance(stmt, ast.Pass):
        return True
    return (isinstance(stmt, ast.Expr)
            and isinstance(getattr(stmt, "value", None), ast.Constant)
            and stmt.value.value is Ellipsis)


def _pass_stub_count(text: str) -> int:
    """Bare `pass` / `...` bodies that are the WHOLE body of a definition.

    A `pass` is not automatically a stub: `BaseHTTPRequestHandler.log_message`
    is deliberately a no-op, empty `class MyError(Exception): pass` subclasses
    are valid, and `except OSError: pass` is best-effort cleanup. However, a
    top-level (module-scope) function whose entire body is `pass` or `...`
    (like a scaffolded `def get_wave_difficulty(wave): pass` left behind in a
    10-function module) is always an unfinished stub. Methods inside classes
    still use the >= 50% ratio so intentional no-op overrides like
    `Handler.log_message` beside real handlers are never flagged.
    """
    text = text or ""
    try:
        tree = ast.parse(textwrap.dedent(text))
    except (SyntaxError, ValueError):
        passes = len(_PASS_RE.findall(text))
        defs = len(_DEF_RE.findall(text))
        return passes if passes and passes * 2 >= max(1, defs) else 0
    top_stubs = 0
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in _NOOP_FUNC_NAMES:
                continue
            decs = [getattr(d, "id", "") or getattr(d, "attr", "")
                    for d in (node.decorator_list or ())]
            if "abstractmethod" in decs:
                continue
            if _is_bare_pass_or_ellipsis(list(node.body)):
                top_stubs += 1
    if top_stubs:
        return top_stubs
    defs = 0
    stubs = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
            continue
        defs += 1
        if _is_bare_pass_or_ellipsis(list(node.body)):
            stubs += 1
    if stubs and stubs * 2 >= max(1, defs):
        return stubs
    return 0


def _count(text: str) -> int:
    text = text or ""
    return (sum(len(rx.findall(text)) for rx in _PLACEHOLDER_RES)
            + _pass_stub_count(text))


def check_placeholders(task: str, edits) -> list[Finding]:
    if _STUB_OK_RE.search(task or ""):
        return []
    out: list[Finding] = []
    for e in edits:
        added = _count(e.replace) - _count(e.search)
        if added > 0 and e.replace.strip():
            out.append(Finding(
                "placeholder", "error", e.file,
                f"the edit adds {added} placeholder(s) (TODO / "
                "NotImplementedError / bare `pass` / `...`) instead of "
                "working code; the task did not ask for a stub"))
    return out


# --------------------------------------------------------------- unread edits
def _ledger(agent) -> dict:
    return getattr(agent, "_read_spans", None) or {}


def check_unread(agent, edits, results) -> list[Finding]:
    """Edits to files (or line regions) the gather phase never read."""
    spans = _ledger(agent)
    if not spans and not getattr(agent, "read_any", False):
        # nothing was read at all: the answer-phase refusal logic owns that
        return []
    # A file this run CREATED during lookup (a `write` step) was never
    # "read", but the model authored every byte of it, so a later edit block
    # for it is not an unread guess. The run's backup dir records creations.
    created: set[str] = set()
    bdir = getattr(getattr(agent, "cfg", None), "_backup_dir", None)
    if bdir:
        try:
            created = {fileio.path_key(x) for x in
                       fileio.created_manifest(bdir).read_text(
                           encoding="utf-8").splitlines() if x.strip()}
        except OSError:
            created = set()
    out: list[Finding] = []
    root = agent.index.root
    seen_files: set[str] = set()
    for e in edits:
        rel = fileio.normalise_rel(e.file)
        key = fileio.path_key(rel)
        path = root / rel
        if not path.is_file():
            continue                      # a creation: nothing to have read
        if key in created:
            continue                      # created by this run's own write
        read = spans.get(key) or next(
            (v for k, v in spans.items() if k.endswith("/" + key)
             or key.endswith("/" + k)), None)
        if not read:
            if key not in seen_files:
                out.append(Finding(
                    "unread", "error", rel,
                    "this file was never `read` in the lookup phase, so the "
                    "edit's SEARCH text cannot have come from the file"))
                seen_files.add(key)
            continue
        # region check for quoted (non-anchored) edits
        if e.lines:
            a, b = e.lines
        elif e.search.strip():
            try:
                text, _style = fileio.read(path, errors="replace")
            except OSError:
                continue
            span = locate(text, e.search, agent.cfg)
            if span is None:
                continue
            a = text.count("\n", 0, span.start) + 1
            b = text.count("\n", 0, max(span.start, span.stop)) + 1
        else:
            continue
        covered = any(s <= a and b <= t for s, t in _merge(read))
        if not covered:
            overlap = any(not (t < a or s > b) for s, t in read)
            out.append(Finding(
                "unread", "warn" if overlap else "error", rel,
                f"lines {a}-{b} were not (fully) among the lines read "
                f"({_fmt(_merge(read))}); the edit may be a guess",
                line=a, lines=(a, b)))
    return out


def _merge(spans) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _fmt(spans) -> str:
    return ", ".join(f"{a}-{b}" for a, b in spans[:4]) or "none"


# ------------------------------------------------------------ stale references
_DEF_RES = (
    re.compile(r"(?m)^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)"),
    re.compile(r"(?m)^\s*(?:export\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)"),
    re.compile(r"(?m)^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
               r"(?:async\s*)?(?:\([^)]*\)|[\w$]+)\s*=>"),
    re.compile(r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)"),
    re.compile(r"(?m)^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)"),
    re.compile(r"(?m)^\s*(?:class|struct|interface|trait|enum)\s+([A-Za-z_]\w*)"),
)
_IGNORE_NAMES = {"main", "init", "new", "test", "setup", "run", "get", "set",
                 "self", "cls", "constructor", "toString", "__init__"}


def _defs(text: str) -> set[str]:
    names: set[str] = set()
    for rx in _DEF_RES:
        names.update(rx.findall(text or ""))
    return {n for n in names if len(n) > 2 and n not in _IGNORE_NAMES
            and not (n.startswith("__") and n.endswith("__"))}


def check_stale_references(agent, results) -> list[Finding]:
    """Definitions present before an edit and gone after it that other code
    still references (callers, importers, same-file uses)."""
    out: list[Finding] = []
    index = agent.index
    # what each file will contain once ALL of this reply's edits land: a
    # correct multi-file rename updates the callers in the same answer
    after_by_file = {r.file: (r.after or "") for r in results
                     if r.ok and r.file and r.after is not None}
    for r in results:
        if not r.ok or not r.file:
            continue
        before, after = r.before or "", r.after or ""
        if not before:
            continue
        gone = sorted(_defs(before) - _defs(after))
        if not gone:
            continue
        # files that could still use them: the file itself + importers + callers
        candidates = {r.file}
        try:
            candidates |= set(index.graph.dependents(r.file))
        except Exception:
            pass
        for name in gone[:6]:
            rx = re.compile(r"(?<![\w$])" + re.escape(name) + r"(?![\w$])")
            # moved elsewhere and imported back in: not a loss
            if re.search(r"(?m)^\s*(?:from\s+\S+\s+import[^\n]*|import\s+[^\n]*|"
                         r"(?:const|let|var)[^\n]*require\([^\n]*|use\s+[^\n]*|"
                         r"import\s*\{[^\n]*)(?<![\w$])" + re.escape(name)
                         + r"(?![\w$])", after):
                continue
            hits: list[tuple[str, int]] = []
            for i, line in enumerate((after or "").splitlines(), 1):
                if rx.search(line):
                    hits.append((r.file, i))
                    break
            for f in sorted(candidates):
                if f == r.file:
                    continue
                if f in after_by_file:
                    lines = after_by_file[f].splitlines()
                else:
                    try:
                        lines = fileio.read_lines_cached(index.root / f)
                    except OSError:
                        continue
                for i, line in enumerate(lines, 1):
                    if rx.search(line):
                        hits.append((f, i))
                        break
            if hits:
                shown = ", ".join(f"{f}:{n}" for f, n in hits[:3])
                out.append(Finding(
                    "stale", "error", r.file,
                    f"`{name}` is defined in the original but not in the "
                    f"edited file, and is still referenced at {shown}; "
                    "update those references in the same answer (or keep "
                    "the definition)"))
    return out


def check_moved_imports(agent, results) -> list[Finding]:
    out: list[Finding] = []
    for r in results:
        for old in getattr(r, "renames_from", ()) or ():
            try:
                users = [f for f in agent.index.graph.importers_of(old)
                         if f != r.file]
            except Exception:
                users = []
            if users:
                out.append(Finding(
                    "import", "error", old,
                    f"moved to {r.file}, but still imported by "
                    f"{', '.join(users[:4])}; update those imports in the "
                    "same answer"))
    return out


# --------------------------------------------------------------- lost wiring
_JS_IMPORT_RE = re.compile(
    r"(?m)^[ \t]*import\s+(?:\{([^}]*)\}|([A-Za-z_$][\w$]*))\s+from\s+"
    r"['\"][^'\"]+['\"]")
# Same shape, but the module source is captured (group 3) so a check can tell
# WHICH file an import pulls a name from (BUG-214).
_JS_IMPORT_FROM_RE = re.compile(
    r"(?m)^[ \t]*import\s+(?:\{([^}]*)\}|([A-Za-z_$][\w$]*))\s+from\s+"
    r"['\"]([^'\"]+)['\"]")
_PY_IMPORT_RE = re.compile(r"(?m)^[ \t]*from\s+[.\w]+\s+import\s+([^\n]+)")
_REWRITE_RE = re.compile(
    r"\b(?:rewrite|re-?write|rebuild|recreate|overhaul)\b|"
    r"\breplace\s+(?:the\s+)?(?:whole|entire|full)\s+file\b|"
    r"\breplace\s+the\s+contents\b", re.IGNORECASE)


def _imported_names(text: str) -> set[str]:
    """Binding names introduced by ES-module or Python import lines."""
    names: set[str] = set()
    for m in _JS_IMPORT_RE.finditer(text or ""):
        if m.group(1) is not None:
            for part in m.group(1).split(","):
                part = re.split(r"\s+as\s+", part.strip())[-1].strip()
                if re.fullmatch(r"[A-Za-z_$][\w$]*", part):
                    names.add(part)
        elif m.group(2):
            names.add(m.group(2))
    for m in _PY_IMPORT_RE.finditer(text or ""):
        for part in m.group(1).split(","):
            part = re.split(r"\s+as\s+", part.strip().strip("()"))[-1].strip()
            if part and part != "*" and re.fullmatch(r"[A-Za-z_]\w*", part):
                names.add(part)
    return names


def _asks_file_rewrite(task: str, rel: str) -> bool:
    """True when the task names this file AND asks for a rewrite/replacement
    of it: the only sanctioned reason to drop its existing wiring."""
    low = (task or "").lower()
    name = (rel or "").rsplit("/", 1)[-1].lower()
    return bool(name and name in low and _REWRITE_RE.search(low))


def check_lost_wiring(agent, results, task: str = "") -> list[Finding]:
    """A whole-file replacement that drops imported-and-called names still
    defined elsewhere in the project deletes working wiring (BUG-216): the
    removed helpers have no callers left to trip `check_stale_references`,
    and the new module graph validates in isolation while the page quietly
    loses its grid/transport/scope. Error by default; a warn when the task
    explicitly asks to rewrite that very file."""
    out: list[Finding] = []
    index = getattr(agent, "index", None)
    for r in results:
        if not r.ok or not r.file or not r.before:
            continue
        if not (r.message or "").startswith("replaced contents"):
            continue
        before, after = r.before, r.after or ""
        dropped: list[str] = []
        for name in sorted(_imported_names(before)):
            rx = re.compile(r"(?<![\w$])" + re.escape(name) + r"(?![\w$])")
            # called/used in the original beyond its own import line?
            if not any(rx.search(ln) for ln in before.splitlines()
                       if not ln.lstrip().startswith(("import ", "from "))):
                continue
            if rx.search(after):
                continue          # still imported or defined: wiring kept
            try:
                still_defined = bool(index and index.find_symbol(name))
            except Exception:
                still_defined = False
            if still_defined:
                dropped.append(name)
        if not dropped:
            continue
        out.append(Finding(
            "lost_wiring",
            "warn" if _asks_file_rewrite(task, r.file) else "error",
            r.file,
            "the whole-file replacement dropped "
            + ", ".join(f"`{n}`" for n in dropped[:6])
            + ": imported and called in the original, still defined in the "
              "project, and absent from the new text. Keep the existing "
              "wiring unless the task asks to remove it"))
    return out


_JS_RESULT_EXTS = (".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".mts",
                   ".cts")
_JS_READER_EXTS = _JS_RESULT_EXTS + (".html", ".htm")
# CommonJS named surface: `exports.name =` / `module.exports.name =` and the
# keys of a `module.exports = { ... }` object literal.
_CJS_EXPORT_DOT_RE = re.compile(
    r"\b(?:module\.)?exports\.([A-Za-z_$][\w$]*)\s*=(?!=)")
_CJS_EXPORT_OBJ_RE = re.compile(r"\bmodule\.exports\s*=\s*\{")
# Classic-script globals: `window.name =` / `globalThis.name =` (dot or
# bracket spelling), the surface a page's other scripts read.
_WINDOW_ASSIGN_RE = re.compile(
    r"\b(?:window|globalThis)\.([A-Za-z_$][\w$]*)\s*=(?!=)"
    r"|\b(?:window|globalThis)\[\s*['\"]([A-Za-z_$][\w$]*)['\"]\s*\]\s*=(?!=)")
_REQUIRE_RE = re.compile(r"\brequire\(\s*['\"]([^'\"]+)['\"]\s*\)")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)


def _cjs_exported_names(text: str) -> set[str]:
    """Names a CommonJS module puts on `module.exports` / `exports.x`."""
    from . import webcheck
    text = webcheck._js_no_comments(text)
    names = {m.group(1) for m in _CJS_EXPORT_DOT_RE.finditer(text)}
    for m in _CJS_EXPORT_OBJ_RE.finditer(text):
        # Top-level keys of the object literal: walk the braces so a nested
        # value object does not donate its keys to the export surface.
        depth, i, n = 1, m.end(), len(text)
        seg_start = i
        segs: list[str] = []
        while i < n and depth:
            c = text[i]
            if c in "'\"`":
                i += 1
                while i < n and text[i] != c:
                    i += 2 if text[i] == "\\" else 1
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    segs.append(text[seg_start:i])
                    break
            elif c == "," and depth == 1:
                segs.append(text[seg_start:i])
                seg_start = i + 1
            i += 1
        for seg in segs:
            km = re.match(r"['\"]?([A-Za-z_$][\w$]*)['\"]?\s*(?::|$)",
                          seg.strip())
            if km:
                names.add(km.group(1))
    return names


def _window_assigned_names(text: str) -> set[str]:
    """Globals a classic script assigns onto `window` / `globalThis`."""
    from . import webcheck
    text = webcheck._js_no_comments(text)
    return {m.group(1) or m.group(2)
            for m in _WINDOW_ASSIGN_RE.finditer(text)}


def _lost_cjs_names(index, rel: str, base: str,
                    dropped: set[str]) -> set[str]:
    """Dropped CommonJS names that another project file still reads through
    its `require()` of this module (destructured, bound, or direct)."""
    lost: set[str] = set()
    for other in index.files:
        if other == rel or not other.lower().endswith(_JS_RESULT_EXTS):
            continue
        try:
            text = fileio.read_text(Path(index.root) / other)
        except OSError:
            continue
        if base not in text:
            continue
        for m in _REQUIRE_RE.finditer(text):
            if m.group(1).rsplit("/", 1)[-1] != base:
                continue
            pre = text[max(0, m.start() - 200):m.start()]
            dm = re.search(r"(?:const|let|var)\s*\{([^{}]*)\}\s*=\s*$", pre)
            if dm:
                for part in dm.group(1).split(","):
                    name = part.split(":")[0].strip()
                    if name in dropped:
                        lost.add(name)
            bm = re.search(
                r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*$", pre)
            if bm:
                bind = re.escape(bm.group(1))
                for name in dropped:
                    if re.search(r"\b" + bind + r"\s*\.\s*"
                                 + re.escape(name) + r"\b", text):
                        lost.add(name)
            post = text[m.end():m.end() + 120]
            pm = re.match(r"\s*\.\s*([A-Za-z_$][\w$]*)", post)
            if pm and pm.group(1) in dropped:
                lost.add(pm.group(1))
    return lost


def _lost_window_names(index, rel: str, dropped: set[str]) -> set[str]:
    """Dropped window globals that another project file still reads."""
    from . import webcheck
    lost: set[str] = set()
    pats = {
        n: re.compile(r"\b(?:window|globalThis)\s*\.\s*" + re.escape(n)
                      + r"\b|\b(?:window|globalThis)\[\s*['\"]"
                      + re.escape(n) + r"['\"]\s*\]")
        for n in dropped}
    for other in index.files:
        if other == rel or not other.lower().endswith(_JS_READER_EXTS):
            continue
        try:
            text = fileio.read_text(Path(index.root) / other)
        except OSError:
            continue
        if not any(n in text for n in dropped):
            continue
        if other.lower().endswith((".html", ".htm")):
            text = _HTML_COMMENT_RE.sub("", text)
        else:
            text = webcheck._js_no_comments(text)
        for n, pat in pats.items():
            if n not in lost and pat.search(text):
                lost.add(n)
    return lost


def check_lost_exports(agent, results, task: str = "") -> list[Finding]:
    """A whole-file replacement that drops an export another project file still
    imports breaks the module graph (BUG-214): the new file validates on its
    own, but every importer fails to link at load (`does not provide an export
    named X`). `check_lost_wiring` only catches imported-and-CALLED names that
    live elsewhere; a file's OWN dropped exports have no such caller to trip
    it. Covers every whole-file shape (empty-SEARCH overwrite, a LINES range
    covering the file, a SEARCH quoting all of it) and all three export
    idioms: ESM `export`, CommonJS `module.exports`, and `window.*` globals.
    Error by default; warn when the task explicitly asks to rewrite that
    very file."""
    from . import webcheck
    out: list[Finding] = []
    index = getattr(agent, "index", None)
    if index is None:
        return out
    for r in results:
        if not r.ok or not r.file or not r.before:
            continue
        whole = ((r.message or "").startswith("replaced contents")
                 or getattr(r, "whole_file", False))
        if not whole:
            continue
        if not r.file.lower().endswith(_JS_RESULT_EXTS):
            continue
        after = r.after or ""
        esm_dropped = (webcheck._js_exported_names(r.before)
                       - webcheck._js_exported_names(after))
        cjs_dropped = (_cjs_exported_names(r.before)
                       - _cjs_exported_names(after))
        win_dropped = (_window_assigned_names(r.before)
                       - _window_assigned_names(after))
        if not (esm_dropped or cjs_dropped or win_dropped):
            continue
        base = r.file.rsplit("/", 1)[-1]
        lost: set[str] = set()
        if esm_dropped:
            try:
                importers = index.importers_of(r.file)
            except Exception:
                importers = []
            for rel in importers:
                if rel == r.file:
                    continue
                try:
                    text = fileio.read_text(Path(index.root) / rel)
                except OSError:
                    continue
                for m in _JS_IMPORT_FROM_RE.finditer(text):
                    if m.group(3).rsplit("/", 1)[-1] != base:
                        continue    # the import points at a different module
                    names: set[str] = set()
                    if m.group(1) is not None:
                        for part in m.group(1).split(","):
                            part = re.split(r"\s+as\s+",
                                            part.strip())[-1].strip()
                            if re.fullmatch(r"[A-Za-z_$][\w$]*", part):
                                names.add(part)
                    elif m.group(2):
                        names.add(m.group(2))
                    lost |= esm_dropped & names
        if cjs_dropped:
            lost |= _lost_cjs_names(index, r.file, base, cjs_dropped)
        if win_dropped:
            lost |= _lost_window_names(index, r.file, win_dropped)
        if not lost:
            continue
        out.append(Finding(
            "lost_exports",
            "warn" if _asks_file_rewrite(task, r.file) else "error",
            r.file,
            "the whole-file replacement dropped "
            + ", ".join(f"`{n}`" for n in sorted(lost)[:6])
            + ": still imported by other project files, which fail to link "
              "at load (`does not provide an export named ...`). Keep these "
              "exports unless the task asks to remove them"))
    return out


def check_oversized_rewrite(agent, results, task: str = "") -> list[Finding]:
    """Context-pressure guard: on a small-window profile, a whole-file draft
    of a large file is how a context-pressed model deletes working wiring
    (the GLM-4.7-Flash Phase-5a shape). Refuse it BEFORE it lands -- error by
    default, warn only when the task explicitly asks to rewrite that very
    file -- and name the forms that fit a small window instead."""
    cfg = getattr(agent, "cfg", None)
    small = int(getattr(cfg, "small_context_tokens", 0) or 0)
    if small <= 0:
        return []
    window = int(getattr(cfg, "context_tokens", 0) or 0)
    if window <= 0 or window > small:
        return []
    cap = max(20, int(getattr(cfg, "max_whole_file_lines", 120) or 120))
    out: list[Finding] = []
    for r in results:
        if not r.ok or not r.file or not getattr(r, "whole_file", False):
            continue
        if getattr(r, "renames_from", ()):
            continue          # a rename moves real bytes, not a fresh draft
        lines = len((r.before or "").splitlines())
        if lines <= cap:
            continue
        # Creative/design bypass: a hand-authored page / stylesheet / doc, a
        # creative task, or a restructure/replace brief is restructured, not
        # "drafted from a partial read" -- a 150-line index.html that becomes a
        # 90-line layout is exactly the requested change. Do not refuse it on a
        # small-window profile (the 300-line page cap is dropped for these).
        from .prompts import whole_rewrite_allowed
        if whole_rewrite_allowed(task, r.file):
            continue
        out.append(Finding(
            "oversized_rewrite",
            "warn" if _asks_file_rewrite(task, r.file) else "error",
            r.file,
            f"this reply replaces the WHOLE {lines}-line file, but the "
            f"profile's context window is small ({window:,} tokens) and the "
            f"whole-file ceiling is {cap} lines: a draft this size gets cut "
            "off mid-file and drops working wiring. Re-emit the change as "
            "small SEARCH/REPLACE blocks, a SYMBOL anchor with the full new "
            "definition, or successive append chunks -- never the whole "
            "file in one REPLACE"))
    return out


# ------------------------------------------------- static-site build hijack
# A design task on a plain HTML/CSS workspace keeps getting "upgraded" into a
# Node project: the model creates package.json / tailwind.config.js /
# postcss.config.js, and the page then renders unstyled because nothing here
# runs a build. The prose rule (prompts.STATIC_SITE_RULES) alone did not stop
# it, so the apply path strips these creations deterministically -- unless
# the workspace already had a package.json before the task, or the task
# explicitly asks for npm/Node/Tailwind tooling.
BUILD_TOOL_BASENAMES = frozenset({
    "package.json",
    "tailwind.config.js", "tailwind.config.cjs", "tailwind.config.mjs",
    "tailwind.config.ts",
    "postcss.config.js", "postcss.config.cjs", "postcss.config.mjs",
    "postcss.config.ts",
})


def unrequested_build_creations(preexisting, task: str, rels) -> list[str]:
    """The subset of `rels` that are build-tool files this task may not
    CREATE: the workspace had no package.json before the turn (any depth)
    and the task text does not ask for npm/Node/Tailwind tooling. Edits to
    pre-existing files are never flagged -- only fresh creations. Returns
    the offending paths in their original spelling, deduplicated."""
    from . import prompts
    if prompts.wants_build_tools(task):
        return []
    keys = {fileio.path_key(str(f)) for f in (preexisting or ())}
    if any(k.rsplit("/", 1)[-1] == "package.json" for k in keys):
        return []
    out: list[str] = []
    for rel in rels:
        key = fileio.path_key(str(rel))
        if key in keys:
            continue              # existed before the task: an edit, not a hijack
        if key.rsplit("/", 1)[-1] in BUILD_TOOL_BASENAMES:
            out.append(str(rel))
    return list(dict.fromkeys(out))


# ------------------------------------------------------------------ citations
_CITE_RES = (
    re.compile(r"(?<![\w/.-])([\w./\\-]+\.[A-Za-z]{1,5}):(\d+)"),
    re.compile(r"(?<![\w/.-])([\w./\\-]+\.[A-Za-z]{1,5})\s+(?:lines?|L)\s*(\d+)",
               re.IGNORECASE),
)
_BLOCK_STRIP_RE = re.compile(r"<{5,}\s*SEARCH.*?>{5,}\s*REPLACE", re.DOTALL)


def _prose(reply: str) -> str:
    return _BLOCK_STRIP_RE.sub("", reply or "")


def check_citations(agent, reply: str) -> list[Finding]:
    """`file:line` claims in the reply about files that were never read (or
    lines outside what was read): the model is describing code it has not
    seen. An unresolvable path is reported too, rather than silently skipped."""
    spans = _ledger(agent)
    if not spans:
        return []
    out: list[Finding] = []
    seen: set[str] = set()
    for rx in _CITE_RES:
        for m in rx.finditer(_prose(reply)):
            name = m.group(1)
            rel = agent.index.resolve_file(name)
            if not rel:
                if name not in seen:
                    seen.add(name)
                    out.append(Finding(
                        "citation", "warn", name,
                        f"the reply cites {name}:{m.group(2)}, a file that does "
                        "not resolve in the project"))
                continue
            if rel in seen:
                continue
            seen.add(rel)
            key = fileio.path_key(rel)
            read = spans.get(key)
            line = int(m.group(2))
            if not read:
                out.append(Finding(
                    "citation", "warn", rel,
                    f"the reply cites line {line} of a file the lookup phase "
                    "never read", line=line))
            elif not any(s <= line <= t for s, t in read):
                out.append(Finding(
                    "citation", "warn", rel,
                    f"the reply cites line {line}, which was not among the "
                    "lines read", line=line))
    return out[:4]


# ------------------------------------------- shadow-mode prose verification
_SNAKE_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b")
_CAMEL_RE = re.compile(r"\b[a-z][a-z0-9]*[A-Z][A-Za-z0-9]*\b")


def check_ungrounded_identifiers(agent, reply: str, task: str) -> list[Finding]:
    """Identifier-shaped tokens (snake_case or camelCase) in the prose that
    resolve to nothing in the index and are not named in the task -- a signal
    of an invented symbol. Shadow only: findings are logged, never acted on."""
    index = getattr(agent, "index", None)
    if index is None:
        return []
    task_words = {w.lower() for w in _WORD_RE.findall(task or "")}
    out: list[Finding] = []
    seen: set[str] = set()
    for rx in (_SNAKE_RE, _CAMEL_RE):
        for w in rx.findall(_prose(reply)):
            low = w.lower()
            if low in seen or low in task_words:
                continue
            seen.add(low)
            if not index.find_symbol(w):
                out.append(Finding(
                    "identifier", "warn", "",
                    f"'{w}' looks like a symbol but is neither in the index "
                    "nor named in the task"))
    return out[:4]


def prose_shadow(agent, task: str, reply: str, edits=None) -> list[Finding]:
    """Shadow-mode prose verification (P1a): the checks run on a no-edit
    answer, and every finding is a `warn` that is logged for later labelling --
    never promoted to an error or fed to repair."""
    findings: list[Finding] = []
    try:
        findings += check_citations(agent, reply)
        findings += check_ungrounded_identifiers(agent, reply, task)
        findings += check_coverage(clean_task_for_coverage(task), reply,
                                   edits or [])
    except Exception:
        pass
    for f in findings:
        f.severity = "warn"
    return findings


# ----------------------------------------------------- requirement ledger
_REQ_SPLIT_RE = re.compile(
    r"(?:\n\s*(?:[-*]|\d+[.)])\s+|;\s+|\.\s+(?=[A-Z])|,?\s+and then\s+|"
    r",?\s+and\s+(?=(?:also\s+)?(?:add|make|fix|update|change|remove|rename|"
    r"create|write|ensure|check|handle|support|use|return|move|delete|"
    r"implement|show|print|log|replace|restyle|redesign|style|insert|"
    r"position|reorder|align|center|swap|convert|keep|preserve)\b)|"
    r",?\s+then\s+)", re.IGNORECASE)

# A comma-separated list of visual edits ("remove X, move Y, add Z") is
# several requirements even without "and" before each verb. This split only
# applies to a brief that already looks like page/layout work or quotes the
# exact copy; without that signal a compound code sentence ("..., pass
# onPlay to buildTransport, add presets, and drop...") is ONE requirement.
_REQ_COMMA_SPLIT_RE = re.compile(
    r",\s+(?=(?:also\s+)?(?:add|move|remove|replace|restyle|redesign|style|"
    r"insert|position|reorder|align|center|swap|convert|keep|preserve|"
    r"make|update|change|edit|modify|create)\b)", re.IGNORECASE)
_WEB_LAYOUT_TEXT_RE = re.compile(
    r"\b(?:html|css|stylesheet|website|webpage|web page|landing page|"
    r"personal site|bio|portfolio|layout|hero|card|theme|styling|restyle|"
    r"redesign|sub\s*heading|subheading|heading|links?|navbar|footer)\b",
    re.IGNORECASE)

# A lead-in preamble introduces the list of requirements; it is not R1. The
# colon-terminated forms were already dropped, but "please implement the
# following" / "you need to update the app so that" without a colon still ate
# the first slot and shifted every R-number (consolidated follow-up, item 2).
# Both a lead-in head AND an introduction tail are required so a real clause
# ("Fix the button below") is never mistaken for a preamble.
_PREAMBLE_HEAD_RE = re.compile(
    r"^(?:in this (?:repository|repo|project|codebase)\b[,:-]?\s*"
    r"|(?:please|kindly)\b\s*"
    r"|you (?:need|have|must|should|want) to\b\s*"
    r"|your task is to\b\s*"
    r"|(?:we|i) (?:need|want) you to\b\s*)", re.IGNORECASE)
_PREAMBLE_TAIL_RE = re.compile(
    r"\b(?:the following|as follows|so that|"
    r"these (?:changes|requirements|items|steps|tasks))\s*:?\s*$",
    re.IGNORECASE)
_PREAMBLE_BARE_RE = re.compile(
    r"^(?:please\s+|kindly\s+)?(?:implement|do|perform|complete|carry out|"
    r"make|apply)\s+(?:the following|all of the following|these)\b",
    re.IGNORECASE)


def _strip_lead_preamble(clauses: list[str]) -> list[str]:
    """Drop leading list-introduction clauses before R-numbering. Only when
    concrete clauses follow: a lone introduction IS the task."""
    out = list(clauses)
    while len(out) > 1:
        head = out[0].strip()
        intro = (_PREAMBLE_HEAD_RE.match(head)
                 and _PREAMBLE_TAIL_RE.search(head)) \
            or _PREAMBLE_BARE_RE.match(head)
        if not intro:
            break
        out.pop(0)
    return out
# A sentence-boundary dot that actually belongs to an abbreviation ("e.g.",
# "i.e.", "etc.") is not a clause boundary: "handle errors, i.e. Check the
# inputs" used to split into "...i.e" + "Check the inputs" (consolidated
# sweep, BUG-204/212). The trailing dot is part of the match, so the prefix
# up to and including it must end in one of these.
_REQ_ABBREV_RE = re.compile(
    r"\b(?:e\.g|i\.e|etc|vs|cf|viz|ibid|Mr|Mrs|Ms|Dr|Prof|approx|No)\.$",
    re.IGNORECASE)


def _split_clauses(text: str) -> list[tuple[str, bool]]:
    """Split task text at clause boundaries that sit at bracket depth zero,
    returning ``(clause, hard)`` pairs.

    The old `_REQ_SPLIT_RE.split` cut at every `;` -- including one inside a
    parenthetical ("(sharps allowed; return null for bad input)"), which
    became two plan tasks and a phantom coverage clause (BUG-202). Track
    ()/[]/{} depth and double/back quotes so a boundary inside them is text,
    not a split point. Single quotes are left alone: apostrophes in prose
    ("the model's output") are far commoner than single-quoted strings.

    `hard` marks a clause that begins at an explicit bullet/numbered marker
    (or the very first clause). A bulleted list is the author saying "these
    are separate requirements", so a hard clause is never merged into the one
    before it even when its verb is not in our list ("document it in the
    README"). Only soft-boundary fragments (a `;`, a `.`, an `and then`) are
    candidates for the verbless-continuation merge."""
    parts: list[tuple[str, bool]] = []
    depth = 0
    quote = ""
    start = i = 0
    n = len(text)
    hard = True                       # the first clause stands on its own
    # Only a visual/page brief (or one that quotes the exact copy) gets the
    # comma-separated-imperative split. A code sentence that happens to list
    # arguments with commas must stay one requirement.
    allow_comma = bool(_quoted_literals(text)
                       or _WEB_LAYOUT_TEXT_RE.search(text))
    while i < n:
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in "\"`":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif depth == 0:
            m = _REQ_SPLIT_RE.match(text, i)
            if m is None and allow_comma:
                m = _REQ_COMMA_SPLIT_RE.match(text, i)
            if m is not None and m.end() > i:
                if m.group(0).startswith(".") and \
                        _REQ_ABBREV_RE.search(text[:m.start() + 1]):
                    i += 1          # "e.g." / "i.e." -- text, not a boundary
                    continue
                parts.append((text[start:m.start()], hard))
                hard = bool(re.match(r"\n\s*(?:[-*]|\d+[.)])\s+", m.group(0)))
                i = start = m.end()
                continue
        i += 1
    parts.append((text[start:], hard))
    return parts
_VERBS = {"add", "make", "fix", "update", "change", "remove", "rename", "create",
          "write", "ensure", "check", "handle", "support", "use", "return",
          "move", "delete", "implement", "show", "print", "log", "should",
          "must", "also", "please", "when", "with", "that", "this", "into",
          "from", "have", "make", "need", "the", "and", "for", "all", "any",
          "each", "every", "not", "does", "will", "would", "then", "than",
          "your", "our", "its", "are", "can", "get", "set", "new", "old"}
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")
# An identifier-like term (snake_case, dotted/path, a digit, or camelCase) is
# the distinctive evidence a clause is about; plain prose words are not. Used
# to keep the tail of a long compound clause in the ledger (BUG-328) and to
# attribute a hunk to the clause it actually serves (BUG-305/330).
_IDENT_TERM_RE = re.compile(r"[_./]|\d|[a-z][A-Z]")
# Generic words that match almost any edit and must not, on their own, tie a
# hunk to a clause (the all-to-all provenance of BUG-305/330).
_GENERIC_TERMS = frozenset({
    "test", "tests", "testing", "public", "private", "code", "file", "files",
    "function", "functions", "method", "methods", "class", "classes", "module",
    "modules", "import", "imports", "export", "exports", "return", "returns",
    "value", "values", "data", "node", "nodes", "name", "names", "type",
    "types", "line", "lines", "page", "app", "application", "project",
    "signature", "signatures", "unit", "units", "npm", "pass", "fail",
    # Common edit verbs and failure nouns match almost any hunk; on their own
    # they are not provenance evidence (consolidated sweep, BUG-203/223).
    "update", "updates", "error", "errors", "fix", "fixes", "change",
    "changes", "ensure", "ensures", "handle", "handles", "support",
    "supports", "using", "work", "works", "working",
})

# Requirement evidence state machine. An edit requirement has to
# travel UNLOCATED -> LOCATED -> READ -> EDITED -> VERIFIED; an investigation /
# explain requirement travels UNLOCATED -> (LOCATED) -> READ -> REPORTED. A
# search/find/outline result supplies LOCATED evidence, a displayed read span
# supplies READ, a clean patch supplies EDITED/VERIFIED, and prose supplies
# REPORTED. The tracker is cheap (no model calls) and monotonic.
REQ_UNLOCATED = "UNLOCATED"
REQ_LOCATED = "LOCATED"
REQ_READ = "READ"
REQ_EDITED = "EDITED"
REQ_VERIFIED = "VERIFIED"
REQ_REPORTED = "REPORTED"

# Short aliases so callers can spell a state the way the plan does.
UNLOCATED = REQ_UNLOCATED
LOCATED = REQ_LOCATED
READ = REQ_READ
EDITED = REQ_EDITED
VERIFIED = REQ_VERIFIED
REPORTED = REQ_REPORTED

_REQ_ORDER = {
    REQ_UNLOCATED: 0, REQ_LOCATED: 1, REQ_READ: 2,
    REQ_EDITED: 3, REQ_REPORTED: 3, REQ_VERIFIED: 4,
}
_REQ_EDIT_RE = re.compile(
    r"\b(?:add|make|fix|update|change|remove|rename|create|write|ensure|"
    r"handle|support|use|return|move|delete|implement|refactor|replace|"
    r"edit|modify|adjust|correct|bump|upgrade|migrate|introduce|apply|"
    r"keep|allow|accept|raise|lower|increase|decrease|reduce|extend|"
    r"prevent|avoid|guard|validate|normalize|normalise|convert|switch|"
    r"wire|hook|register|append|insert|set)\b", re.IGNORECASE)
_REQ_REPORT_RE = re.compile(
    r"\b(?:explain|describe|summari[sz]e|investigate|trace|find|locate|"
    r"identify|report|why|how|what|where|which|who|tell|show|list|answer|"
    r"determine|clarify|review|audit)\b|\?", re.IGNORECASE)
_REQ_LOCATION_RE = re.compile(
    r"(?<![\w/.-])([A-Za-z0-9_./\\-]+\.[A-Za-z0-9_]{1,8}):(\d+)"
    r"(?:-(\d+))?")

# Literal text the task asks to insert, remove or replace. Double quotes and
# backticks are unambiguous; single quotes are only a literal when they wrap
# a phrase (so an apostrophe in "Here's" is never mistaken for a delimiter).
_QUOTED_LITERAL_RES = (
    re.compile(r'"([^"\n]{2,300})"'),
    re.compile(r"`([^`\n]{2,300})`"),
    re.compile(r"(?<![\w])'([^'\n]{3,300})'(?![\w])"),
)
_REMOVE_LITERAL_RE = re.compile(
    # "replace X with \"Y\"" INSERTS Y; only a genuine removal verb makes the
    # quoted copy something that must be GONE from the final page.
    r"\b(?:remove|delete|drop|erase|omit|take out|get rid of)\b",
    re.IGNORECASE)


def _quoted_literals(text: str) -> list[str]:
    """Every literal quoted in `text`, in order, without duplicates.

    A multi-clause layout brief carries its exact copy in quotes ("Here's a
    bit about me...", "Our Mission"). Those are the only parts of the request
    that can be checked verbatim against the edited file, so they are kept
    whole instead of being lost in the generic word bag.
    """
    out: list[str] = []
    for rx in _QUOTED_LITERAL_RES:
        for m in rx.finditer(text or ""):
            lit = m.group(1).strip()
            if len(lit) >= 2 and lit not in out:
                out.append(lit)
    return out


_REPLACE_VERB_RE = re.compile(r"\breplace\b", re.IGNORECASE)
_REPLACE_GAP_RE = re.compile(r"^[\s,]*(?:with|by)\b", re.IGNORECASE)


def _replace_pairs(text: str) -> tuple[list[str], list[str]]:
    """The (from, to) quoted copy of every `replace "X" with "Y"` pair.

    Only the explicit "replace" verb qualifies: "change"/"rename" do not
    say which side is the old copy. The verb must sit OUTSIDE quoted copy,
    the from-side is the first quote after it and the to-side the next,
    joined by a short "with"/"by" gap -- so `replace the bio text with
    "new"` (an unquoted from-side) stays a plain insertion (BUG-009)."""
    spans: list[tuple[int, int, str]] = []
    for rx in _QUOTED_LITERAL_RES:
        for m in rx.finditer(text or ""):
            lit = m.group(1).strip()
            if len(lit) >= 2:
                spans.append((m.start(), m.end(), lit))
    spans.sort(key=lambda s: (s[0], s[0] - s[1]))
    ordered: list[tuple[int, int, str]] = []
    end = -1
    for s, e, lit in spans:
        if s >= end:            # a quote inside a wider quoted run is copy
            ordered.append((s, e, lit))
            end = e
    froms: list[str] = []
    tos: list[str] = []
    for vm in _REPLACE_VERB_RE.finditer(text or ""):
        if any(s <= vm.start() < e for s, e, _lit in ordered):
            continue
        after = [sp for sp in ordered if sp[0] >= vm.end()]
        if len(after) < 2:
            continue
        (_s1, e1, lit1), (s2, _e2, lit2) = after[0], after[1]
        gap = text[e1:s2]
        if lit1 != lit2 and len(gap) <= 40 and _REPLACE_GAP_RE.match(gap):
            if lit1 not in froms:
                froms.append(lit1)
            if lit2 not in tos:
                tos.append(lit2)
    return froms, tos


def _literal_in(text: str, literal: str) -> bool:
    """True when `literal` appears in `text` (HTML entities normalised)."""
    if not literal:
        return False
    if literal in (text or ""):
        return True
    try:
        import html
        return literal in html.unescape(text or "")
    except Exception:
        return False


# A removal clause quotes the copy from memory ("remove the 'lifelong tinker'
# tagline") while the page carries an inflected, punctuation-shifted form
# ("Consultant, builder, and lifelong tinkerer."). Exact substring matching
# declared that removal done with the text still on the page. Token matching
# tolerates inserted stopwords and a bounded stem difference -- for REMOVAL
# checks only: an insertion stays exact, so a missing quote is never excused
# by a near neighbour.
_COPY_STOPWORDS = frozenset({
    "a", "an", "and", "the", "of", "or", "to", "in", "on", "at", "by",
    "for", "with", "as", "is", "it", "its", "be", "been", "am", "are",
})


def _copy_tokens(text: str) -> list[str]:
    """Lowercased word tokens of `text`, punctuation dropped."""
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _copy_token_match(a: str, b: str) -> bool:
    """One literal token against one file token: exact, or a bounded prefix
    (stem tolerance: "tinker" ~ "tinkerer"). Both must be at least 4 chars
    and differ by at most 3, so "and" never matches "android"."""
    if a == b:
        return True
    if len(a) < 4 or len(b) < 4 or abs(len(a) - len(b)) > 3:
        return False
    return a.startswith(b) or b.startswith(a)


def _literal_fuzzy_in(text: str, literal: str) -> bool:
    """True when `literal`'s content tokens occur in `text` as one contiguous
    run, tolerating stopwords between them and bounded stem differences.
    Needs at least two non-stopword literal tokens: a single common word is
    too weak to call a removal unfinished."""
    content = [t for t in _copy_tokens(literal) if t not in _COPY_STOPWORDS]
    if len(content) < 2:
        return False
    hay = _copy_tokens(text)
    width = len(content) + 4
    for i in range(len(hay)):
        if not _copy_token_match(content[0], hay[i]):
            continue
        j, k = i + 1, 1
        end = min(i + width, len(hay))
        while k < len(content) and j < end:
            if hay[j] in _COPY_STOPWORDS:
                j += 1
                continue
            if _copy_token_match(content[k], hay[j]):
                k += 1
                j += 1
            else:
                break
        if k == len(content):
            return True
    return False


@dataclass
class Requirement:
    """One task clause plus the evidence the harness has gathered for it."""
    text: str
    terms: list[str]
    kind: str = "edit"                 # edit | report | constraint
    state: str = REQ_UNLOCATED
    anchors: list[tuple] = field(default_factory=list)    # (file, start, end)
    read_spans: list[tuple] = field(default_factory=list)
    edit_spans: list[tuple] = field(default_factory=list)
    # Stable clause id ("R1", "R2", ...) assigned by the task splitter; the
    # review page uses it to show which clause each file serves (R8).
    req_id: str = ""
    # Transient verdict of the last coverage pass: the reply's PROSE addresses
    # this edit clause but no edit implements it (description is not
    # implementation). The repair prompt names the gap explicitly.
    prose_only: bool = False
    # Exact quoted copy the clause inserts (or asks to remove). Coverage
    # checks these against the FINAL edited text: ordinary key terms can match
    # an unrelated neighbor, while the user's literal copy cannot.
    literals: list[str] = field(default_factory=list)
    literal_remove: bool = False
    # The quoted copy a `replace "X" with "Y"` clause swaps OUT (X) and IN
    # (Y). X must be absent from the final text like a removal; Y must be
    # present like an insertion. Without this split the coverage pass
    # demanded the replaced-away copy back and seeded a follow-up that
    # undid a successful edit (BUG-009).
    replace_from: list[str] = field(default_factory=list)
    replace_to: list[str] = field(default_factory=list)
    # Literals the last coverage pass found missing/present when they should
    # be the opposite; rendered by check_coverage and the repair prompt.
    missing_literals: list[str] = field(default_factory=list)
    # Disk evidence for an unfinished REMOVAL: (rel, line, line_text) for
    # every line still carrying the quoted copy, so the follow-up names the
    # exact line to delete instead of re-guessing where it lives.
    literal_hits: list[tuple] = field(default_factory=list)

    def __post_init__(self):
        if self.kind not in ("edit", "report", "constraint"):
            self.kind = "edit"
        if self.kind == "constraint":
            # A prohibition has nothing to locate, read or implement; it is
            # satisfied by not violating it. Park it in a final state so the
            # gather gate and next_action never chase it (BUG-306).
            self.state = REQ_REPORTED
        elif self.state not in _REQ_ORDER:
            self.state = REQ_UNLOCATED
        self.terms = list(dict.fromkeys(self.terms))

    @property
    def is_edit(self) -> bool:
        return self.kind == "edit"

    @property
    def final(self) -> bool:
        if self.kind == "constraint":
            return True
        return self.state == (REQ_VERIFIED if self.is_edit else REQ_REPORTED)

    @property
    def open(self) -> bool:
        return not self.final

    def add_anchor(self, rel: str, start: int, end: int) -> None:
        span = (fileio.normalise_rel(rel or ""), int(start or 1),
                int(end or 0))
        if span[0] and span not in self.anchors:
            self.anchors.append(span)

    def add_read(self, rel: str, start: int, end: int) -> None:
        span = (fileio.normalise_rel(rel or ""), int(start or 1),
                int(end or 0))
        if span[0] and span not in self.read_spans:
            self.read_spans.append(span)

    def add_edit(self, rel: str) -> None:
        rel = fileio.normalise_rel(rel or "")
        if rel:
            span = (rel, 0, 0)
            if span not in self.edit_spans:
                self.edit_spans.append(span)

    def literal_must_go(self, lit: str) -> bool:
        """True when this quoted copy must be ABSENT from the final text:
        a removal clause's copy, or the from-side of a replace pair. The
        to-side of a replace stays an insertion even when the same clause
        also carries a removal verb."""
        return lit in self.replace_from or (
            self.literal_remove and lit not in self.replace_to)

    def search_term(self) -> str:
        """The most specific term to grep/find when this requirement has no
        located anchor yet (identifiers and file names win over prose)."""
        generic = {
            "explain", "describe", "summarize", "summarise", "investigate",
            "trace", "find", "locate", "identify", "report", "works", "work",
            "logic", "thing", "things", "stuff", "code", "file", "files",
            "update", "change", "make", "fix", "add", "new", "old", "the",
            "and", "for", "with", "that", "this", "into", "from",
        }

        def rank(t: str):
            score = len(t)
            if "_" in t:
                score += 8
            if "." in t or "/" in t:
                score += 6
            if any(c.isdigit() for c in t):
                score += 3
            if t.isupper():
                score += 2
            if t.isdigit():
                score -= 4
            if t.lower() in generic:
                score -= 10
            return score
        terms = [t for t in self.terms if len(t) >= 3]
        return max(terms, key=rank) if terms else ""


def _term_in(term: str, text: str) -> bool:
    """Whether `term` appears as a word (or a simple plural stem) in `text`."""
    term = (term or "").strip().lower()
    if not term or not text:
        return False
    low = text.lower()
    if re.search(r"(?<![\w])" + re.escape(term) + r"(?![\w])", low):
        return True
    stem = term.rstrip("s")
    return stem != term and bool(
        re.search(r"(?<![\w])" + re.escape(stem) + r"(?![\w])", low))


def _match_score(terms, text: str) -> int:
    return sum(1 for t in terms or () if _term_in(t, text))


def _specific_terms(terms) -> list[str]:
    """The distinctive terms of a clause: identifier-like or long words, minus
    generic vocabulary that matches almost any edit. Provenance must rest on
    these, not on words like `tests`/`public`/`npm` (BUG-305/330).

    A bare number ("4", "16") is NOT distinctive even though `_IDENT_TERM_RE`
    matches its digit: almost every hunk body contains some number, so letting
    one tie a hunk to a clause re-created the all-to-all provenance of
    BUG-204/223 (a theory.js patch claimed the pattern clauses because both
    mentioned `16`). A term must carry at least one letter to be evidence."""
    out = []
    for t in terms or ():
        low = t.lower()
        if low in _GENERIC_TERMS:
            continue
        if not re.search(r"[A-Za-z]", t):
            continue
        if _IDENT_TERM_RE.search(t) or len(t) >= 6:
            out.append(t)
    return out


def _clause_names_file(text: str, rel: str) -> bool:
    """Whether the clause explicitly names this file (path or base name)."""
    if not rel:
        return False
    base = rel.rsplit("/", 1)[-1]
    for name in {rel, base}:
        if name and re.search(r"(?<![\w])" + re.escape(name) + r"(?![\w])",
                              text or ""):
            return True
    return False


# A file-like token ("src/ui.js", "package.json"): the clause scopes itself to
# named files. The extension must start with a letter so version numbers
# ("Qwen3.5") are not mistaken for paths.
_FILE_TOKEN_RE = re.compile(
    r"(?<![\w])(?:[\w.-]+/)*[\w.-]+\.[A-Za-z][A-Za-z0-9]{0,7}(?![\w])")


# Prose/reference documents are indexed for grep, but a generic term matching
# one must not pull it into the gather context as an auto-read (BUG-329).
_REFERENCE_DOC_EXTS = frozenset({".md", ".rst", ".txt", ".adoc", ".org"})


def _is_reference_doc(rel: str) -> bool:
    return "." in (rel or "") and \
        rel[rel.rfind("."):].lower() in _REFERENCE_DOC_EXTS


def _resolve_rel(name: str, index=None) -> str:
    name = (name or "").strip().strip("`'\"")
    if not name:
        return ""
    if index is not None:
        try:
            rel = index.resolve_file(name)
            if rel:
                return rel
        except Exception:
            pass
    return fileio.normalise_rel(name)


def _anchors_from_text(text: str, index=None) -> list[tuple]:
    """Exact `file:line` / `file:start-end` anchors in a tool observation."""
    out: list[tuple] = []
    seen: set[tuple] = set()
    for m in _REQ_LOCATION_RE.finditer(text or ""):
        name = m.group(1)
        if name.lower().startswith("http"):
            continue
        start = int(m.group(2))
        end = int(m.group(3) or start)
        if start <= 0 or end < start:
            continue
        rel = _resolve_rel(name, index)
        if not rel:
            continue
        span = (rel, start, end)
        if span not in seen:
            seen.add(span)
            out.append(span)
        if len(out) >= 40:
            break
    return out


def _clause_kind(clause: str, default: str) -> str:
    if _REQ_EDIT_RE.search(clause or ""):
        return "edit"
    if _REQ_REPORT_RE.search(clause or ""):
        return "report"
    return default


# A prohibition / preservation clause ("Change nothing else", "do not weaken
# unrelated tests", "Keep every public signature ...") binds the model but can
# never produce positive edit evidence. Treating it as an edit requirement made
# coverage warn that it was "unaddressed" and let R4 seed a pointless follow-up
# from it (BUG-306). Such clauses are tracked as `constraint`: always final,
# never a plan step, never a coverage seed, never a provenance badge.
_CONSTRAINT_HEAD_RE = re.compile(
    r"^\s*(?:do not|don't|dont|never|avoid|change nothing|no other|"
    r"without\s+(?:changing|modifying|touching|altering|breaking))\b",
    re.IGNORECASE)
_CONSTRAINT_PHRASE_RE = re.compile(
    r"\b(?:change nothing else|nothing else|leave (?:everything|it|them|the "
    r"rest) (?:unchanged|alone|untouched|as-is|as is))\b", re.IGNORECASE)
_CONSTRAINT_KEEP_RE = re.compile(
    r"^\s*(?:keep|preserve|maintain|leave)\b", re.IGNORECASE)
_CONSTRAINT_KEEP_MARK_RE = re.compile(
    r"\b(?:do not|don't|unchanged|untouched|intact|as-is|as is|the same|"
    r"unmodified|not weaken|no weaker|without (?:changing|weakening))\b",
    re.IGNORECASE)


def _is_constraint_clause(clause: str) -> bool:
    t = clause or ""
    if _CONSTRAINT_HEAD_RE.search(t) or _CONSTRAINT_PHRASE_RE.search(t):
        return True
    if _CONSTRAINT_KEEP_RE.search(t):
        rest = _CONSTRAINT_KEEP_RE.sub("", t, count=1)
        if _CONSTRAINT_KEEP_MARK_RE.search(rest):
            return True
    return False


def _clip_clause(p: str, limit: int = 140) -> str:
    """A requirement line a step can actually read: clipped on a word
    boundary, never mid-token.

    The old `p[:140]` cut wherever the count landed, so a plan step inherited
    a title ending "...the restart button must ma" and one 27B run concluded
    from it that the contract was already satisfied. A clause longer than the
    limit is a clause the splitter should have opened: keep whole words and
    mark the cut, so what remains is still a sentence."""
    if len(p) <= limit:
        return p
    head = p[:limit]
    if " " in head[40:]:                    # only clip when there is a word
        cut = head.rfind(" ", 40, limit)    # boundary to fall back on
    else:
        cut = len(head)
    clipped = head[:cut].rstrip(" ,;:.-\t")
    if not clipped:
        return p[:limit]
    return clipped + " ..."


def _requirements_from_task(task: str, limit: int = 16) -> list[Requirement]:
    # Only the CURRENT task is mined. An injected handoff/checkpoint carries
    # earlier session notes above a `NEW TASK` / `CURRENT TASK` / `TASK`
    # marker; mining the whole blob turned stale, already-finished work into
    # mandatory requirements of a standalone Ask run (D3).
    text = re.split(r"\n(?:NEW TASK|CURRENT TASK|TASK)\n", task or "")[-1].strip()
    if not text:
        return []
    parts = [(p.strip(" \t\n.,;"), hard)
             for p, hard in _split_clauses(text) if p.strip(" \t\n.,;")]
    # A fragment with no verb of its own is a continuation of the clause
    # before it ("... with sharps allowed"), not a requirement of its own:
    # merge it back instead of letting it become a nonsense plan step. A
    # hard clause (an explicit bullet/numbered item) is never merged: the
    # author listed it as a separate requirement even if its verb ("document")
    # is not one we recognise.
    merged: list[str] = []
    for p, hard in parts:
        p = re.sub(r"^(?:[-*]|\d+[.)])\s+", "", p)   # drop the bullet marker
        # A COMPLETED checklist item ("[x] ..." from a stale PLAN.md or
        # handoff) is history, not a requirement of the current task: it
        # must never seed plan steps, coverage seeds or follow-ups (D3).
        if re.match(r"^\[[xX]\]", p):
            continue
        p = re.sub(r"^\[ \]\s*", "", p)              # unchecked box: keep text
        if not p:
            continue
        if merged and not hard and not (
                _REQ_EDIT_RE.search(p) or _REQ_REPORT_RE.search(p)):
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    # A clause that ends in ":" is a list introduction ("Required final public
    # API and behavior:"), not a requirement of its own: it became R1 and ate a
    # slot the concrete clauses needed (BUG-203/205). Drop it when concrete
    # clauses follow; the full goal still rides along as the task background.
    if len(merged) > 1:
        merged = [p for p in merged if not p.rstrip().endswith(":")]
        # The same introduction without its colon ("In this repository,
        # please implement the following", "You need to update the app so
        # that") is still a preamble, not R1 (consolidated sweep, item 2).
        merged = _strip_lead_preamble(merged)
    task_kind = "edit" if _REQ_EDIT_RE.search(text) else (
        "report" if _REQ_REPORT_RE.search(text) else "edit")
    reqs: list[Requirement] = []
    for p in merged:
        if len(p) < 8:
            continue
        words = [w for w in _WORD_RE.findall(p) if w.lower() not in _VERBS]
        words += re.findall(r"\b\d+\b", p)
        literals = _quoted_literals(p)
        # Quoted copy leads the terms: it is the exact thing the user asked to
        # see (or stop seeing) on the page.
        words = literals + words
        # BUG-328: the clause used to be clipped to 140 chars and its terms
        # capped at the first 8 in occurrence order, so the tail of a long
        # compound clause (handler names, the `presets:` argument, the
        # preset-change behaviour) vanished from the ledger and coverage could
        # never see it. Keep the FULL clause text and rank identifier-like
        # terms first so the distinctive tail survives the cap.
        ident = [w for w in words if _IDENT_TERM_RE.search(w) or w in literals]
        plain = [w for w in words if not _IDENT_TERM_RE.search(w)
                 and w not in literals]
        terms = list(dict.fromkeys(ident + plain))[:16]
        if terms:
            # BUG-809: a protective tail merged into its edit clause by the
            # fragment merger ("Change index.html. Touch nothing else." ->
            # one clause) still carries the edit verb and the file name: that
            # clause IS the task, and classifying it as a pure constraint
            # silently dropped the whole requirement — an edit task then had
            # zero tracked edit clauses, so the review provenance map, the
            # coverage seeds and the unaccepted-change guard all went blind.
            # A prohibition that LEADS its clause ("Do not modify game.js.",
            # "Keep index.html unchanged.") is still a pure constraint and
            # outranks the edit words it contains (BUG-306); only a trailing
            # phrase absorbed by the merger is overruled.
            edit_outranks = bool(
                _REQ_EDIT_RE.search(p) and _FILE_TOKEN_RE.search(p)
                and not _CONSTRAINT_HEAD_RE.search(p)
                and not _CONSTRAINT_KEEP_RE.search(p))
            constraint = _is_constraint_clause(p) and not edit_outranks
            kind = ("constraint" if constraint
                    else _clause_kind(p, task_kind))
            rep_from, rep_to = _replace_pairs(p)
            reqs.append(Requirement(
                p, terms, kind, req_id=f"R{len(reqs) + 1}",
                literals=literals,
                literal_remove=bool(_REMOVE_LITERAL_RE.search(p)),
                replace_from=rep_from, replace_to=rep_to))
    # BUG-203/212/223: the old default cap of five silently truncated a six-
    # bullet repair (and a three-clause task the splitter expanded to eleven
    # sentences), so the tail clauses never existed for coverage or provenance
    # to see and the run exited 0 after addressing only the head. Sixteen slots
    # hold a realistic multi-file brief (R6+ survived hunts #8-#12's twelve);
    # the cap still bounds a runaway split.
    return reqs[:limit]


def requirement_ledger(task: str, limit: int = 16) -> list[tuple[str, list[str]]]:
    """[(requirement text, key terms)] for up to `limit` clauses of the task.
    Heuristic and free: split on sentence/bullet/`and add ...` boundaries, keep
    each clause's distinctive words (identifiers, file names, numbers,
    quoted text). Constraint clauses ("change nothing else") are excluded: they
    bind the model but are not steps to implement (BUG-306)."""
    return [(r.text, r.terms) for r in _requirements_from_task(task, limit)
            if r.kind != "constraint"]


class RequirementTracker:
    """Run-scoped requirement evidence state machine.

    The tracker is deliberately deterministic: every transition comes from a
    tool result or a proposed patch the harness already has, never from another
    model call. It powers both the gather completion gate (oldest unmet
    requirement) and the answer-phase source packer (pinned located spans).
    """

    def __init__(self, task: str = "", limit: int = 16):
        self.task = task or ""
        self.items = _requirements_from_task(self.task, limit)

    # ---- container conveniences
    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    @property
    def requirements(self) -> list[Requirement]:
        return self.items

    def states(self) -> list[tuple[str, str]]:
        return [(r.text, r.state) for r in self.items]

    def unmet(self, states=(REQ_UNLOCATED, REQ_LOCATED)) -> list[Requirement]:
        return [r for r in self.items if r.state in states]

    def next_unmet(self, states=(REQ_UNLOCATED, REQ_LOCATED)):
        found = self.unmet(states)
        return found[0] if found else None

    # Alias kept explicit: the gather gate asks for the OLDEST requirement.
    oldest_unmet = next_unmet

    def _advance(self, req: Requirement, target: str) -> bool:
        """Promote `req` monotonically to `target`, filling in the states a
        stronger piece of evidence implies (a verified patch also proves the
        span was read). Returns True when the state changed."""
        if _REQ_ORDER.get(req.state, 0) >= _REQ_ORDER.get(target, 0):
            return False
        req.state = target
        return True

    def _locate(self, req: Requirement, anchors=(), files=()) -> None:
        for rel, a, b in anchors:
            req.add_anchor(rel, a, b)
        for rel in files:
            req.add_anchor(rel, 1, 0)
        self._advance(req, REQ_LOCATED)

    # ---- evidence transitions
    def observe_search(self, action: str, step: dict | None = None,
                       observation: str = "", index=None) -> list[Requirement]:
        """A grep/find/outline/callers/context result that names a requirement
        term (or a file/line anchor) advances it to LOCATED."""
        step = step or {}
        obs = observation or ""
        if obs.startswith("ERROR"):
            return []
        query = " ".join(str(step.get(k) or "") for k in
                         ("pattern", "symbol", "file", "content", "command"))
        anchors = _anchors_from_text(obs, index)
        changed = []
        for req in self.items:
            if req.state not in (REQ_UNLOCATED, REQ_LOCATED):
                continue
            q_hit = _match_score(req.terms, query)
            o_hit = _match_score(req.terms, obs)
            # A grep only locates when its RESULT mentions the term; find /
            # outline / callers may locate from the query itself because their
            # output is a definition list for exactly that query. Anchors alone
            # are generic to the observation, so they do not locate every
            # requirement in the task.
            evidence = o_hit or (
                q_hit and action in ("find", "outline", "callers", "usages",
                                     "context"))
            if not evidence:
                continue
            files = []
            for key in ("file", "pattern", "symbol"):
                val = step.get(key)
                if not isinstance(val, str) or not _match_score(req.terms, val):
                    continue
                rel = _resolve_rel(val, index)
                if rel and "." in val:
                    files.append(rel)
            self._locate(req, anchors, files)
            changed.append(req)
        return changed

    def observe_read(self, file: str, start: int, end: int, text: str = "",
                     index=None) -> list[Requirement]:
        """A displayed read/context span that touches requirement terms or a
        previously located anchor advances the requirement to READ."""
        rel = _resolve_rel(file, index)
        if not rel or (text or "").startswith("ERROR"):
            return []
        changed = []
        for req in self.items:
            if req.final:
                continue
            hit = _match_score(req.terms, rel + "\n" + (text or ""))
            overlap = any(
                _same_file(rel, a_rel) and _spans_overlap(start, end, a, b)
                for a_rel, a, b in req.anchors)
            if not (hit or overlap):
                continue
            req.add_read(rel, start, end)
            if req.state != REQ_READ and self._advance(req, REQ_READ):
                changed.append(req)
        return changed

    def observe_edit(self, edits=None, *, file: str = "", search: str = "",
                     replace: str = "", ok: bool = True) -> list[Requirement]:
        """A clean patch whose file/text touches a requirement advances it to
        EDITED (passing through the earlier states, which the patch proves)."""
        if edits is None:
            edits = [{"file": file, "search": search,
                      "replace": replace, "ok": ok}]
        elif isinstance(edits, (str, dict)) or hasattr(edits, "file"):
            edits = [edits]
        changed = []
        for e in edits or ():
            if isinstance(e, str):
                e = {"file": e, "search": "", "replace": "", "ok": True}
            if isinstance(e, dict):
                rel = e.get("file", "")
                before = e.get("search", "")
                after = e.get("replace", "")
                good = e.get("ok", True)
            else:
                rel = getattr(e, "file", "")
                before = getattr(e, "search", "") or getattr(e, "before", "")
                after = getattr(e, "replace", "") or getattr(e, "after", "")
                good = getattr(e, "ok", True)
            if not good or not rel:
                continue
            rel = _resolve_rel(rel)
            # BUG-305/330: a hunk used to be evidence for EVERY clause whose
            # generic terms matched the file path, or whose anchors/read spans
            # merely touched the file -- so each file was tagged with every
            # clause in the run. Attribute a hunk only to a clause that names
            # the file, or whose DISTINCTIVE terms appear in the hunk's own
            # search/replace text (not the path, not a read overlap).
            body = f"{before}\n{after}"
            for req in self.items:
                if not req.is_edit or req.final:
                    continue
                if _is_test_author_clause(req.text) \
                        and not _TEST_DECL_RE.search(after or ""):
                    # "Add unit tests for X" is not evidenced by an
                    # import-only hunk to a test file (BUG-205): only a
                    # declaration-bearing edit promotes it.
                    continue
                names_file = _clause_names_file(req.text, rel)
                if not names_file:
                    specific = _specific_terms(req.terms)
                    if _FILE_TOKEN_RE.search(req.text):
                        # The clause scopes itself to named file(s): a hunk
                        # in ANOTHER file serves it only through an
                        # identifier the clause names. A shared prose word
                        # ("pattern", "preset") is not evidence -- that was
                        # the residual all-to-all of BUG-305/330.
                        specific = [t for t in specific
                                    if _IDENT_TERM_RE.search(t)]
                    if not _match_score(specific, body):
                        continue
                req.add_edit(rel)
                if self._advance(req, REQ_EDITED):
                    changed.append(req)
        return changed

    def observe_patch(self, file: str, text: str = "", ok: bool = True
                      ) -> list[Requirement]:
        """A `write`/`append`/`move` tool result is an edit too."""
        return self.observe_edit(file=file, replace=text, ok=ok)

    def observe_verified(self, files=None) -> list[Requirement]:
        """Promote EDITED edit requirements to VERIFIED after the checks and
        tests accept the patch. `files` narrows it to the touched files."""
        changed = []
        wanted = {_resolve_rel(f) for f in files} if files else None
        for req in self.items:
            if not req.is_edit or req.state != REQ_EDITED:
                continue
            if wanted is not None and not any(
                    _same_file(f, rel) for rel in wanted
                    for f, _a, _b in req.edit_spans):
                continue
            if self._advance(req, REQ_VERIFIED):
                changed.append(req)
        return changed

    def observe_report(self, reply: str) -> list[Requirement]:
        """Prose that answers a READ investigation requirement reports it."""
        changed = []
        for req in self.items:
            if req.kind != "report" or req.state != REQ_READ:
                continue
            if _match_score(req.terms, _prose(reply)):
                self._advance(req, REQ_REPORTED)
                changed.append(req)
        return changed

    def observe_answer(self, reply: str, edits=None, results=None
                       ) -> list[Requirement]:
        changed = []
        if results is not None:
            changed += self.observe_edit(results)
        elif edits is not None:
            changed += self.observe_edit(edits)
        changed += self.observe_report(reply)
        return changed

    # ---- review provenance (R8)
    def clause_map(self) -> dict[str, list[str]]:
        """file -> [clause ids] for the edit clauses whose patches touched it.
        The review page attaches these to each FileDiff so a partial accept
        can show which clause a file serves."""
        out: dict[str, list[str]] = {}
        for req in self.items:
            if not req.is_edit or not req.req_id:
                continue
            for rel, _a, _b in req.edit_spans:
                ids = out.setdefault(rel, [])
                if req.req_id not in ids:
                    ids.append(req.req_id)
        return out

    def clause_texts(self) -> dict[str, str]:
        """clause id -> the clause text, for the review page's legend."""
        return {req.req_id: req.text for req in self.items if req.req_id}

    # ---- gather completion / packing helpers
    def next_action(self, index=None) -> tuple[dict | None, Requirement | None]:
        """The cheapest deterministic action for the oldest requirement that
        is still UNLOCATED or LOCATED, or (None, None). A located anchor is
        read directly; otherwise a specific term is searched."""
        for req in self.items:
            if req.state not in (REQ_UNLOCATED, REQ_LOCATED):
                continue
            if req.state == REQ_LOCATED:
                for rel, a, b in req.anchors:
                    if any(_same_file(rel, r) and _spans_overlap(a, b or a, x, y)
                           for r, x, y in req.read_spans):
                        continue
                    # BUG-329: a generic term can locate a clause in an
                    # unrelated prose/reference doc (a bug log, a README).
                    # Never auto-read such a file unless the task names it.
                    if _is_reference_doc(rel) \
                            and not _clause_names_file(self.task, rel):
                        continue
                    start, end = max(1, a), max(0, b)
                    # A grep hit names a line; when it lands inside an indexed
                    # definition, read that whole definition so the eventual
                    # SEARCH block can quote real body lines.
                    if index is not None:
                        try:
                            units = [c for c in index.in_file(rel)
                                     if c.kind not in ("module", "reference")
                                     and c.start <= start <= c.end]
                            if units:
                                unit = min(units, key=lambda c: c.nlines)
                                start = max(1, unit.start)
                                end = max(start, unit.scope_end or unit.end)
                        except Exception:
                            pass
                    return {"action": "read", "file": rel,
                            "start": start, "end": end}, req
            term = req.search_term()
            if term:
                return {"action": "grep", "pattern": term}, req
        return None, None

    # Explicit alias for callers that describe the choice, not the ordering.
    cheapest_action = next_action

    def prompt(self, req: Requirement | None = None) -> str:
        """A specific harness nudge naming the oldest unmet requirement."""
        req = req or self.next_unmet()
        if req is None:
            return ""
        if req.state == REQ_LOCATED and req.anchors:
            rel, a, b = req.anchors[0]
            span = f"{rel}:{a}-{b}" if b else f"{rel}:{a}"
            return (f"[Harness: requirement still {req.state}: '{req.text}'. "
                    f"You located it at {span} but have not read it; `read` "
                    f"that exact span before answering.]")
        term = req.search_term()
        how = f"`grep`/`find` for '{term}'" if term else "locate the code"
        return (f"[Harness: requirement still {req.state}: '{req.text}'. "
                f"Use {how} and then `read` the matching span before "
                f"answering.]")

    def pinned_spans(self) -> list[tuple]:
        """Exact spans the answer packer must never evict: located anchors and
        displayed reads for every requirement that got past UNLOCATED."""
        out: list[tuple] = []
        for req in self.items:
            if req.state == REQ_UNLOCATED:
                continue
            for span in req.anchors + req.read_spans:
                if span not in out:
                    out.append(span)
        return out

    def packing_terms(self) -> list[tuple[list[str], int]]:
        """(terms, weight) for the source packer; unread requirements weigh
        most because their evidence is what the answer still needs."""
        out = []
        for req in self.items:
            if req.final:
                continue
            weight = 3 if req.state in (REQ_UNLOCATED, REQ_LOCATED) else (
                2 if req.state == REQ_READ else 1)
            out.append((req.terms, weight))
        return out


def _same_file(a: str, b: str) -> bool:
    a, b = fileio.path_key(a or ""), fileio.path_key(b or "")
    return bool(a and b and (a == b or a.endswith("/" + b)
                             or b.endswith("/" + a)))


def _spans_overlap(a1: int, b1: int, a2: int, b2: int) -> bool:
    """Overlap for 1-based inclusive spans; an end of 0 means whole file."""
    if not a1 or not a2:
        return True
    b1 = b1 or 10 ** 9
    b2 = b2 or 10 ** 9
    return not (b1 < a2 or b2 < a1)


def _edit_evidence_text(e) -> str:
    """One edit's file name plus its new and old text, for term matching."""
    return (f"{getattr(e, 'file', '')}\n"
            f"{getattr(e, 'replace', '') or getattr(e, 'after', '')}\n"
            f"{getattr(e, 'search', '') or getattr(e, 'before', '')}")


# A test DECLARATION, not merely the word "test" in a file name or an import
# line: the test-clause exemption below must not fire for an import-only edit
# to a test file (BUG-205).
_TEST_DECL_RE = re.compile(
    r"\b(?:test|it|describe)\s*\(\s*['\"`]|\bdef\s+test_|\bassert\b|"
    r"\b(?:pytest|unittest)\b|node:test|\b(?:expect|should)\s*\(",
    re.IGNORECASE)
# A clause that asks for tests to be WRITTEN ("Add unit tests for ..."), as
# opposed to made to pass ("make node --test ... pass"): only the former
# demands declaration-bearing test edits as evidence.
_TEST_AUTHOR_RE = re.compile(
    r"\b(?:add|write|create|append|implement|include|new)\b[^.\n]{0,60}?"
    r"\btests?\b", re.IGNORECASE)


def _is_test_author_clause(text: str) -> bool:
    low = (text or "").lower()
    return "test" in low and bool(_TEST_AUTHOR_RE.search(low))


# Markers that begin harness-injected prose appended to the raw task text by
# run_loop / run_task. Coverage extraction must stop at the FIRST of these, so
# backticked tool names in the evidence hint / retry notes and signatures in
# the contract map never become "required literal" clauses.
_HARNESS_MARKERS = (
    "\n\nEVIDENCE:",
    "\n\nCONTEXT FROM PRIOR INVESTIGATION",
    "\n\n[PROJECT INTERFACE CONTRACTS",
    "\n\n### RETRY --",
)

_CONTRACT_BLOCK_RE = re.compile(
    r"\[PROJECT INTERFACE CONTRACTS[^\]]*\].*?\[END INTERFACE CONTRACTS\]",
    re.DOTALL)


def clean_task_for_coverage(task: str) -> str:
    """The user's task text with harness-injected prose removed.

    `run_loop` appends an evidence hint (with backticked `run`/`screenshot`/
    `browser` tool names), prior-investigation context notes, an interface-
    contract map, and one-shot retry notes to the raw task before the worker
    sees it. Those are harness prose, not the user's brief, so they must not
    be mined for coverage requirements; strip everything from the first
    harness marker (and the goal block's ``ORIGINAL USER GOAL`` prefix).
    """
    text = task or ""
    marker = "\nCURRENT TASK\n"
    if marker in text:
        text = text.split(marker, 1)[1]
    cut = len(text)
    for m in _HARNESS_MARKERS:
        i = text.find(m)
        if i != -1:
            cut = min(cut, i)
    if cut != len(text):
        text = text[:cut]
    text = _CONTRACT_BLOCK_RE.sub("", text)
    return text.strip()


def _unaddressed(task: str, reply: str, edits,
                 tracker: RequirementTracker | None = None) -> list[Requirement]:
    """The requirement clauses no edit or sentence addresses: their key terms
    appear nowhere in the edits, the edits' file names, or (for report
    clauses) the reply's prose, and the evidence state machine has not
    promoted them past READ. Shared by `check_coverage` (which renders them
    as findings) and `coverage_seeds` (which turns the edit ones into
    follow-up intents). Runs for prose-only replies too (P1a), not only when
    edit blocks are present."""
    if tracker is None or tracker.task != task:
        tracker = RequirementTracker(task)
    else:
        # A dry-run or candidate-scoring pass must not advance the run's
        # authoritative evidence: only an applied edit does that (the agent
        # calls observe_edit itself once the blocks land).
        tracker = copy.deepcopy(tracker)
    reqs = tracker.items
    if len(reqs) < 2:
        return []
    tracker.observe_edit(edits)
    tracker.observe_report(reply)
    # Only APPLIED edits are evidence. A rejected hunk (ok=False) still carries
    # the model's attempted `replace` text, which mentions every symbol the
    # clause names; counting it let a failed edit satisfy the clause and defeat
    # the R4 coverage follow-up in exactly the case it exists for (BUG-308).
    edit_hay = "\n".join(_edit_evidence_text(e) for e in edits
                         if getattr(e, "ok", True)).lower()
    # Only content-bearing test edits can address a test clause: an
    # import-only hunk to a test file mentions the symbol but adds no tests
    # (BUG-205), so its text is not evidence for one.
    test_hay = "\n".join(
        _edit_evidence_text(e) for e in edits
        if getattr(e, "ok", True) and _TEST_DECL_RE.search(
            getattr(e, "replace", "") or getattr(e, "after", "") or "")
    ).lower()
    prose_hay = _prose(reply).lower()
    out = []
    for req in reqs:
        if req.kind == "constraint":
            # A prohibition has no positive evidence to find; it is never
            # "unaddressed" and never seeds a follow-up (BUG-306).
            continue
        text, terms = req.text, req.terms
        is_test_clause = "test" in text.lower()
        # An EDIT clause is addressed by edit evidence only: prose that
        # merely describes the feature is not implementation (BUG-214).
        # Report clauses keep the prose haystack.
        if req.is_edit:
            hay = test_hay if _is_test_author_clause(text) else edit_hay
        else:
            hay = prose_hay + "\n" + edit_hay
        hit = sum(1 for t in terms if t.lower() in hay
                  or t.lower().rstrip("s") in hay)
        # Quoted copy is checked against the FINAL edited text, not just the
        # replacement hunk. A clause that inserts "Here's a bit about me..."
        # is satisfied only when that string exists on the page; a clause that
        # removes "Our Mission" -- or replaces it with "About Us" -- is
        # satisfied only when the old copy is gone AND the new copy landed.
        # Generic terms can match a neighbor, which is exactly how a
        # multi-clause layout brief ended after its first edit.
        req.missing_literals = []
        if req.literals:
            after_text = "\n".join(
                (getattr(r, "after", "") or "")
                for r in edits
                if getattr(r, "ok", True) and getattr(r, "file", ""))
            if not after_text:
                # Candidate scoring/dry runs may carry parsed Edit blocks, not
                # applied Result objects: judge the replacement text we have.
                after_text = "\n".join(
                    (getattr(e, "replace", "") or "") for e in edits)
            missing: list[str] = []
            for lit in req.literals:
                if req.literal_must_go(lit):
                    # The copy being removed/replaced must be GONE; judge it
                    # with the same stem tolerance as a removal ("lifelong
                    # tinker" ~ "lifelong tinkerer"). With no after-text the
                    # disk re-check (unmet_requirements) owns it: absence
                    # cannot be proven from an empty hunk (BUG-009).
                    if after_text and (_literal_in(after_text, lit)
                                       or _literal_fuzzy_in(after_text, lit)):
                        missing.append(lit)
                elif not after_text:
                    missing.append(lit)
                elif not _literal_in(after_text, lit):
                    missing.append(lit)
            if missing:
                req.missing_literals = missing
                hit = 0
        if is_test_clause and any(
                getattr(e, "ok", True)
                and "test" in (getattr(e, "file", "") or "").lower()
                and _TEST_DECL_RE.search(
                    getattr(e, "replace", "") or getattr(e, "after", "")
                    or "")
                for e in edits):
            continue
        if req.state in (REQ_EDITED, REQ_VERIFIED, REQ_REPORTED) \
                and not req.missing_literals:
            continue
        if hit == 0:
            # Name the prose-vs-edit gap: a clause whose terms appear in the
            # answer's prose but in no edit was DESCRIBED, not implemented.
            # The repair prompt and the follow-up intents say so explicitly
            # instead of a generic "unaddressed".
            req.prose_only = bool(req.is_edit) and any(
                t.lower() in prose_hay or t.lower().rstrip("s") in prose_hay
                for t in terms)
            out.append(req)
    # Do not stop at the first three: a four-part layout brief (remove a
    # heading, move two links, add a heading, replace the bio) left its last
    # clause unseen by both the warning and the R4 follow-up.
    return out[:16]


def check_coverage(task: str, reply: str, edits,
                   tracker: RequirementTracker | None = None) -> list[Finding]:
    """Requirements whose key terms appear nowhere in the edits, their file
    names, or the reply's prose. When a live RequirementTracker is supplied,
    its states are authoritative for what counts as addressed."""
    out: list[Finding] = []
    for req in _unaddressed(task, reply, edits, tracker):
        if getattr(req, "missing_literals", None):
            # One verb per literal: the from-side of a replace (or a removal)
            # is "still present", an insertion that never landed is "missing"
            # -- a mixed clause names both groups honestly (BUG-009).
            gone = [l for l in req.missing_literals if req.literal_must_go(l)]
            owed = [l for l in req.missing_literals
                    if not req.literal_must_go(l)]
            for verb, group in (("missing", owed), ("still present", gone)):
                if group:
                    out.append(Finding(
                        "coverage", "warn", "",
                        f"required literal text is {verb} in the edited file: "
                        + ", ".join(f'"{lit}"' for lit in group)
                        + f" -- the clause '{req.text}' is not finished"))
        elif getattr(req, "prose_only", False):
            out.append(Finding(
                "coverage", "warn", "",
                f"you described '{req.text}' in prose but no edit implements "
                f"it (still {req.state}); description is not implementation "
                "-- emit the edit for it"))
        else:
            out.append(Finding(
                "coverage", "warn", "",
                f"no edit or sentence addresses: '{req.text}' "
                f"(still {req.state})"))
    return out


def coverage_seeds(task: str, reply: str, edits,
                   tracker: RequirementTracker | None = None) -> list[Requirement]:
    """The unaddressed EDIT clauses of `task` after a batch: the deterministic
    trigger for one coverage-seeded follow-up pass. Report clauses are left to
    the answer prose; only edit clauses can seed another edit intent."""
    return [req for req in _unaddressed(task, reply, edits, tracker)
            if req.is_edit]


def unmet_requirements(root, task: str, edits,
                       tracker: RequirementTracker | None = None
                       ) -> list[Requirement]:
    """Removal and replace clauses whose quoted copy is still ON DISK after
    the batch.

    `_unaddressed` judges literals against the edits' after-text only: a
    removal whose file was never touched looks satisfied because no
    replacement text carries the copy. This re-checks the real files -- every
    file the batch touched plus every existing file the clause names -- and
    returns deep-copied requirements annotated with `literal_hits`
    ((rel, line, line_text)) naming each line still carrying the copy, so the
    follow-up can point at the exact line instead of re-guessing. For a
    replace clause only the from-side is re-checked: the to-side is an
    insertion the after-text check already owns (BUG-009)."""
    if tracker is None or tracker.task != task:
        tracker = RequirementTracker(task)
    else:
        tracker = copy.deepcopy(tracker)
    root = Path(root)
    touched = [fileio.normalise_rel(getattr(e, "file", "") or "")
               for e in edits if getattr(e, "ok", True)]
    out: list[Requirement] = []
    for req in tracker.items:
        gone_lits = [l for l in req.literals if req.literal_must_go(l)]
        if not gone_lits:
            continue
        cands = list(dict.fromkeys(touched))
        for m in _FILE_TOKEN_RE.finditer(req.text):
            rel = fileio.normalise_rel(m.group(0))
            if rel and (root / rel).is_file() and rel not in cands:
                cands.append(rel)
        hits: list[tuple] = []
        missing: list[str] = []
        for lit in gone_lits:
            found = False
            for rel in cands:
                p = root / rel
                if not p.is_file():
                    continue
                try:
                    text, _sty = fileio.read(p, errors="replace")
                except OSError:
                    continue
                if not (_literal_in(text, lit)
                        or _literal_fuzzy_in(text, lit)):
                    continue
                found = True
                for ln, line in enumerate(text.splitlines(), 1):
                    if _literal_in(line, lit) or _literal_fuzzy_in(line, lit):
                        hit = (rel, ln, line.strip())
                        if hit not in hits:
                            hits.append(hit)
            if found:
                missing.append(lit)
        if missing:
            req.missing_literals = missing
            req.literal_hits = hits[:8]
            out.append(req)
    return out[:16]


def merge_unmet(base: list[Requirement],
                on_disk: list[Requirement]) -> list[Requirement]:
    """Merge disk-verified unmet removal/replace clauses into the coverage
    seeds. A clause already in `base` gains the disk location (its text is
    annotated with the exact file:line still carrying the copy); a clause the
    edit evidence alone considered addressed is appended. Both lists hold
    deep-copied requirements, so annotating them never touches the run's
    authoritative tracker."""
    out = list(base)
    by_text = {r.text: r for r in out}
    for req in on_disk:
        have = by_text.get(req.text)
        if have is None:
            out.append(req)
            by_text[req.text] = req
            have = req
        hits = getattr(req, "literal_hits", [])
        if hits:
            have.literal_hits = hits
            loc = "; ".join(f"{rel}:{ln}: {txt}" for rel, ln, txt in hits[:2])
            suffix = f" -- still on disk at {loc}"
            if suffix not in have.text:
                have.text += suffix
        for lit in req.missing_literals:
            if lit not in have.missing_literals:
                have.missing_literals.append(lit)
    return out[:16]


def _fuzzy_regions(before: str, after: str) -> list[tuple[int, int]]:
    """1-based inclusive line ranges of `before` that `after` changes: where
    the approximately-anchored blocks landed, computed free with difflib so a
    fuzzy finding carries the region rehydration should re-fetch."""
    if not before or not after or before == after:
        return []
    sm = difflib.SequenceMatcher(
        None, before.splitlines(), after.splitlines(), autojunk=False)
    out: list[tuple[int, int]] = []
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        a = i1 + 1
        out.append((a, max(i2, a)))
        if len(out) >= 4:
            break
    return out


def check_fuzzy(results) -> list[Finding]:
    """A block anchored approximately (fuzzy_match) is a warning, not an
    error: it landed somewhere the SEARCH text did not exactly match, so the
    diff needs a human's eye before it is trusted. The finding carries the
    region it anchored on (near recorded evidence, the repair
    re-fetches exactly those lines)."""
    out: list[Finding] = []
    for r in (results or []):
        if not getattr(r, "fuzzy", False):
            continue
        a = b = 0
        try:
            regions = _fuzzy_regions(getattr(r, "before", ""),
                                     getattr(r, "after", ""))
            if regions:
                a, b = regions[0]
        except Exception:
            a = b = 0
        out.append(Finding(
            "fuzzy", "warn", r.file,
            "one or more blocks matched approximately (not exactly); "
            "review the diff carefully", line=a,
            lines=(a, b) if a else ()))
    return out


# ------------------------------------------------------ exact-span rehydration
# `NEED shop/pricing.py:120-168` -- the model asks the harness for exactly
# the lines it needs, and the harness APPENDS them at the conversation tail
# so the prompt-prefix cache survives the round (TECHNIQUES.md #2). A plain
# `file:line` citation is not a request: the path needs an extension and a
# `:` or space before the line numbers.
_NEED_RE = re.compile(
    r"(?im)^[ \t]*(?:[-*]|\d+[.)])?[ \t]*NEED[ \t]+[`'\"]?[ \t]*"
    r"((?:[A-Za-z]:)?[\w./\\-]+\.[A-Za-z][A-Za-z0-9]{0,7})"
    r"(?:[ \t]*:[ \t]*|[ \t]+)"
    r"(\d+)(?:[ \t]*(?:-|\u2013|[ \t]to)[ \t]*(\d+))?[ \t]*[`'\"]?[ \t]*$")


def find_needs(reply: str) -> list[tuple[str, int, int]]:
    """`(file, start, end)` for each `NEED path:120-168` directive line in a
    reply (deduplicated, order preserved, capped). A single line number asks
    for one line; a reversed range is swapped. Anything without an extension
    and a number is prose about needing something, not a directive."""
    out: list[tuple[str, int, int]] = []
    seen: set[tuple[str, int, int]] = set()
    for m in _NEED_RE.finditer(reply or ""):
        name = fileio.normalise_rel(m.group(1).strip())
        a = int(m.group(2))
        b = int(m.group(3)) if m.group(3) else a
        if a > b:
            a, b = b, a
        key = (name, a, b)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
        if len(out) >= 8:
            break
    return out


_MAX_REHYDRATE_SPANS = 4


def _finding_region(f) -> tuple[int, int]:
    """The line region a finding flags: its `lines` pair, else its single
    `line`, else (0, 0) when the check knows no position."""
    if len(getattr(f, "lines", ()) or ()) == 2:
        return int(f.lines[0]), int(f.lines[1])
    if getattr(f, "line", 0):
        a = int(f.line)
        return a, a
    return 0, 0


def evidence_rehydrations(agent, findings) -> list[tuple[str, int, int]]:
    """`(rel, start, end)` spans the repair prompt should re-fetch from disk.

    Only `unread` and `fuzzy` findings qualify, and only when the flagged
    region is covered by, or within the evidence window (±30 lines) of, a
    span the EvidenceStore recorded as DISPLAYED to the model: the model
    misquoted source it saw, so handing back its fresh exact bytes fixes the
    patch for free. An unread region far from any recorded span is source it
    never saw at all -- rehydrating there would dump an unseen file on the
    model (against token economy #1) instead of grounding a correction."""
    store = getattr(agent, "evidence", None)
    if store is None or not findings:
        return []
    out: list[tuple[str, int, int]] = []
    seen: set[tuple[str, int, int]] = set()
    for f in findings:
        if getattr(f, "kind", "") not in ("unread", "fuzzy") or not f.file:
            continue
        a, b = _finding_region(f)
        if a <= 0:
            continue
        try:
            rel = agent.index.resolve_file(f.file) or fileio.normalise_rel(f.file)
        except Exception:
            rel = fileio.normalise_rel(f.file)
        if not rel or not store.nearby(rel, a, b):
            continue
        key = (rel, a, b)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
        if len(out) >= _MAX_REHYDRATE_SPANS:
            break
    return out


# --------------------------------------------------------------- entry points
def dedupe_findings(findings) -> list:
    """Stable de-duplication by (kind, file, message): the same coverage
    clause used to surface once per hunk and again at the final gate,
    doubling the ledger's apparent damage (BUG-207). First occurrence wins;
    order is preserved."""
    seen: set[tuple] = set()
    out = []
    for f in findings or ():
        key = (getattr(f, "kind", ""), getattr(f, "file", ""),
               getattr(f, "message", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def check(agent, task: str, reply: str, results, gathered: str = ""
          ) -> list[Finding]:
    """All deterministic findings for a proposed reply (never raises)."""
    findings: list[Finding] = []
    try:
        edits = parse_edits(reply)
        findings += check_syntax(results)
        cfg = getattr(agent, "cfg", None)
        root = getattr(getattr(agent, "index", None), "root", None)
        if getattr(cfg, "static_gate", True):
            findings += check_static(results, root=root, cfg=cfg)
            findings += check_static_js(results)
            findings += check_gui_hazards(results, root=root, task=task)
        findings += check_placeholders(task, edits)
        findings += check_unread(agent, edits, results)
        findings += check_stale_references(agent, results)
        findings += check_moved_imports(agent, results)
        findings += check_lost_wiring(agent, results, task)
        findings += check_lost_exports(agent, results, task)
        findings += check_oversized_rewrite(agent, results, task)
        findings += check_citations(agent, reply)
        findings += check_doc_grounding(agent, results)
        tracker = getattr(agent, "requirements", None)
        clean_task = clean_task_for_coverage(task)
        if tracker is not None and getattr(tracker, "task", "") != clean_task:
            tracker = None
        if not getattr(agent, "_is_repair", False):
            # A repair prompt is harness diagnostic text, not a user brief:
            # its "clauses" are gate output and instructions (BUG-314). The
            # harness-injected evidence hint / contract map / retry notes are
            # stripped too, so backticked tool names never read as clauses.
            findings += check_coverage(clean_task, reply, edits, tracker=tracker)
        findings += check_fuzzy(results)
        findings += check_empty_js_tests(results)
    except Exception as e:                  # a checker bug must not block a task
        findings.append(Finding("model", "warn", "",
                                f"verification skipped ({type(e).__name__})"))
    return dedupe_findings(findings)


def render(findings, severity: str | None = None) -> str:
    return "\n".join(f.render() for f in findings
                     if severity is None or f.severity == severity)


def score_candidate(agent, task: str, reply: str, results, gathered: str = ""
                    ) -> tuple[float, list[Finding]]:
    """A comparable quality score for one candidate edit reply (higher is
    better): blocks that parse and dry-apply cleanly count for it; failed
    blocks, placeholder/unread/stale errors and unaddressed requirements
    count against it."""
    edits = parse_edits(reply)
    findings = check(agent, task, reply, results, gathered)
    if not edits:
        return (0.0 if reply.strip() else -5.0), findings
    ok = sum(1 for r in results if r.ok)
    bad = sum(1 for r in results if not r.ok)
    score = 10.0 + 4.0 * ok - 6.0 * bad
    score -= 8.0 * sum(1 for f in findings if f.severity == "error")
    score -= 2.0 * sum(1 for f in findings
                       if f.severity == "warn" and f.kind != "auto_heal")
    return score, findings


VERIFIER_SYSTEM = """\
You are a strict code reviewer. You are given a TASK, the SOURCE the author
read, and the author's proposed search/replace edits. Judge the edits ONLY
against that source.

Check: (1) every SEARCH region really appears in the SOURCE; (2) the edit
does what the TASK says -- including its LAST requirement; (3) nothing the
edit removes or renames is still used elsewhere in the SOURCE; (4) no claim
about code that is not in the SOURCE.

Reply with either the single word PASS, or one line per concrete defect in
exactly this form (at most 5 lines, no preamble):
DEFECT: <file>:<line> -- <what is wrong and what it should be>

If the SOURCE is missing a region you must inspect to judge it, you may put
`NEED <file>:<start>-<end>` lines (at most two) before your verdict: the
harness fetches those exact lines and asks you again."""

_DEFECT_RE = re.compile(r"(?im)^\s*DEFECT\s*:\s*(?P<where>[^\s]+)\s*(?:--|-|:)\s*(?P<what>.+)$")


def parse_verdict(text: str, resolve=None, root=None) -> list[Finding]:
    """Defects from a verifier reply ([] for PASS or anything unparseable).

    `resolve` is a file-name -> indexed-path resolver (``index.resolve_file``);
    `root` is the project directory. With both, a DEFECT is kept as an `error`
    only when the file resolves and the cited line is inside it; otherwise it
    is a `warn` -- a verifier hallucinating a path or a line number must not
    feed the repair re-ask. Without `resolve`, every defect is an error (the
    caller has no index to check against)."""
    out = []
    for m in _DEFECT_RE.finditer(text or ""):
        where = m.group("where")
        f, _, line = where.partition(":")
        name = f.strip("`")
        line_no = int(line) if line.isdigit() else 0
        severity = "error"
        file = name
        if resolve is not None:
            rel = resolve(name)
            if rel is None or line_no <= 0:
                severity = "warn"
            else:
                file = rel
                if root is not None:
                    try:
                        if line_no > len(fileio.read_lines_cached(root / rel)):
                            severity = "warn"
                    except OSError:
                        severity = "warn"
        out.append(Finding("model", severity, file,
                           m.group("what").strip()[:240], line_no))
    return out[:5]


# --------------------------------------------------- structured-edit guard
# `validate_structured_reply` is the deterministic preflight a structured-mode
# reply must clear before anything is written: strict schema parse, exact
# intent matching, containment/sensitive/protected/policy, symbol/search
# resolution, no-op/overlap/scope, the syntax gate, and the existing
# undefined-name static gate over the dry-applied buffers. WP-4101B feeds it
# before the atomic transaction.
def _structured_static_errors(mat, cfg) -> list[str]:
    """New undefined-name (F821/`__all__`) errors in the dry-applied buffers,
    the structured twin of `check_static` for materialized Results."""
    out: list[str] = []
    for r in mat.results:
        rel = str(getattr(r, "file", "") or "")
        if not rel.lower().endswith(".py"):
            continue
        if not getattr(cfg, "static_gate", True):
            continue
        before = getattr(r, "before", "") or ""
        after = getattr(r, "after", "") or ""
        new = _new_static_errors(rel, after, before)
        if new:
            out.append(
                f"{rel}: this edit introduces names that do not exist at the "
                "point they are used (a missing import or a typo): "
                + "; ".join(new[:5]))
    return out


def validate_structured_reply(root, intents: list[StructuredIntent],
                              edits: list[StructuredEdit], *,
                              allowed_files: set[str] | None = None,
                              read_spans: dict | None = None,
                              cfg=None) -> list[str]:
    """Preflight a structured reply; ``[]`` means every edit is valid to apply.

    Runs the strict mechanical guard (:func:`edits.materialize_structured_edits`:
    intent matching, containment, sensitive/protected/policy, symbol/search
    resolution, no-op/overlap/scope, and the syntax gate) and then the existing
    verifier gate (undefined-name static check) over the dry-applied buffers.
    Nothing is written; each returned string is one refusal.
    """
    mat = materialize_structured_edits(root, intents, edits,
                                       allowed_files=allowed_files,
                                       read_spans=read_spans, cfg=cfg)
    errors = list(mat.errors)
    if errors:
        return errors
    errors.extend(_structured_static_errors(mat, cfg))
    return errors
