"""Deterministic symbol rename: the multi-file edit small models get wrong.

    tool_rename(index, cfg, "apply_discount", "discounted_price")
    rename_symbol_across_project(root, "old", "new", dry_run=True)

Renaming across files is exactly where a small model drops one call site. This
does it mechanically, and only where it is safe:

  * the symbol must have ONE definition (another function with the same bare
    name anywhere makes the rename ambiguous: refused, with the candidates);
  * the new name must be a valid identifier that nothing else defines;
  * only the definition's file and the files that IMPORT it are rewritten --
    whole-word, outside string literals and comments (Python uses the real
    tokenizer; JS/TS/Go/Rust/... rename tree-sitter IDENTIFIER nodes, so
    strings and comments can never be touched by construction);
  * every rewritten file must still parse (same check the edit path uses);
  * references it did NOT touch (files that do not import the definition,
    docs, strings) are listed so the model can handle them with edit blocks.

`rename_symbol_across_project` is the pure engine: project-root scoped (or
scoped to one file and its importers), zero LLM tokens, all-or-nothing, with
a structured dict result ({ok, files_modified, total_replacements, diffs})
shared by the agent tool, the `pedacito rename` CLI, and the GUI's Project
menu. `tool_rename` is the older index-strict agent tool kept for its
ambiguity refusal; `tool_rename_symbol` wraps the engine as an agent action.

Gated by `allow_write`, snapshotted like every other write (undo with
`pedacito restore`), and the index is refreshed afterwards.
"""

from __future__ import annotations

import difflib
import io
import keyword
import os
import re
import tokenize
from pathlib import Path

from . import fileio, safety

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Directories that never hold project source the rename should touch.
_SKIP_DIR_NAMES = frozenset({
    "node_modules", "__pycache__", "bower_components", "web_modules",
    ".venv", "venv", "env", ".tox", ".nox", "dist", "build", "target",
    "dist-packages", "site-packages", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".next", ".nuxt", ".svelte-kit", ".idea", ".vscode",
    ".gradle", ".terraform", "vendor",
})

# tree-sitter node types that name an identifier (never a string or comment:
# those are separate node types, so the walker cannot hit one).
_TS_IDENT_TYPES = frozenset({
    "identifier", "type_identifier", "field_identifier",
    "property_identifier", "shorthand_property_identifier",
    "namespace_identifier",
})

_MAX_RENAME_BYTES = 4_000_000


def _rename_python(text: str, old: str, new: str) -> tuple[str, int]:
    """Replace NAME tokens equal to `old` (strings and comments untouched)."""
    lines = text.split("\n")
    hits: list[tuple[int, int]] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.NAME and tok.string == old:
                hits.append(tok.start)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return text, -1
    for row, col in sorted(hits, reverse=True):
        ln = lines[row - 1]
        lines[row - 1] = ln[:col] + new + ln[col + len(old):]
    return "\n".join(lines), len(hits)


def _rename_python_receiver(text: str, old: str, new: str, class_path: str
                            ) -> tuple[str, int, list[str]]:
    """Receiver-aware Python rename for the qualified query `class_path.old`.

    Renames only the sites that provably refer to THAT class's member:

      * the definition (`def old(...)` inside the target class);
      * `self.old` / `cls.old` / `this.old` within the target class body;
      * static access through the class name (`Store.old`, `pkg.Store.old`);
      * `x.old(...)` when `x` is deterministically bound to the class in an
        enclosing scope (`x = Store(...)`, `x: Store`, `def f(x: Store)`) --
        the same constructor/annotation evidence the Sprint 4 call graph uses.

    Everything else -- a bare `old` outside the class, `y.old` on a receiver
    typed to ANOTHER class, or an untyped/dynamic receiver -- is skipped and
    described in the returned list instead of being renamed on a guess.
    Returns `(new_text, count, skipped_descriptions)`; count -1 when the file
    does not parse. String literals and comments can never be hit: the AST
    does not contain them."""
    import ast
    from .chunker import _local_types

    leaf = class_path.rsplit(".", 1)[-1]
    target_def = f"{class_path}.{old}"
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return text, -1, []
    lines = text.split("\n")
    hits: list[tuple[int, int, int]] = []          # (row, col0, col1)
    skips: list[str] = []

    def add_attr(a: ast.Attribute) -> None:
        """Queue the attribute NAME token of `x.old` (right-aligned span)."""
        row, col1 = a.end_lineno, a.end_col_offset
        col0 = col1 - len(old)
        ln = lines[row - 1]
        if ln[col0:col1] == old and not (col0 > 0
                                         and (ln[col0 - 1].isalnum()
                                              or ln[col0 - 1] == "_")):
            hits.append((row, col0, col1))

    def add_def(fn) -> None:
        row = fn.lineno
        ln = lines[row - 1]
        col = ln.find(old, fn.col_offset)
        if col >= 0 and (ln[col + len(old):col + len(old) + 1]
                         in ("", "(", " ", ":")):
            hits.append((row, col, col + len(old)))

    def expr_path(node) -> str:
        """Dotted path of a pure Name/Attribute expression, else ''."""
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return ""
        parts.append(node.id)
        return ".".join(reversed(parts))

    class Walk(ast.NodeVisitor):
        def __init__(self):
            self.cls: list[str] = []          # class qualname stack
            self.vmaps: list[dict] = [_local_types(tree)]  # scope type maps

        def in_target(self) -> bool:
            return bool(self.cls) and self.cls[-1] == class_path

        def lookup(self, name: str) -> str:
            for m in reversed(self.vmaps):
                if name in m:
                    return m[name]
            return ""

        def visit_Attribute(self, node: ast.Attribute) -> None:
            self.generic_visit(node)
            if node.attr != old:
                return
            row = node.end_lineno
            path = expr_path(node.value)
            if path in ("self", "cls", "this"):
                if self.in_target():
                    add_attr(node)
                else:
                    skips.append(f"line {row}: `self.{old}` belongs to "
                                 f"{self.cls[-1] or 'module level'}, not "
                                 f"{class_path}")
                return
            if not path:
                skips.append(f"line {row}: dynamic receiver `(...).{old}`")
                return
            segs = path.split(".")
            if segs[-1] == leaf:
                add_attr(node)              # static: Store.old / pkg.Store.old
            elif len(segs) == 1:
                t = self.lookup(segs[0])
                if t == leaf:
                    add_attr(node)          # x typed to the class by binding
                elif t:
                    skips.append(f"line {row}: `{segs[0]}.{old}` is on "
                                 f"{t}, not {leaf}")
                else:
                    skips.append(f"line {row}: `{segs[0]}.{old}` has an "
                                 "untyped receiver (name it `x: "
                                 f"{leaf}` or rename with edit blocks)")
            else:
                skips.append(f"line {row}: `{path}.{old}` is not on "
                             f"{class_path}")

        def visit_Name(self, node: ast.Name) -> None:
            if node.id == old:
                if self.in_target():
                    hits.append((node.lineno, node.col_offset,
                                 node.col_offset + len(old)))
                else:
                    skips.append(f"line {node.lineno}: bare `{old}` outside "
                                 f"{class_path} (it names something else)")
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.cls.append((self.cls[-1] + "." if self.cls else "")
                             + node.name)
            self.vmaps.append(_local_types(node))
            self.generic_visit(node)
            self.cls.pop()
            self.vmaps.pop()

        def _fn(self, node) -> None:
            if node.name == old:
                qual = (self.cls[-1] + "." if self.cls else "") + node.name
                if qual == target_def:
                    add_def(node)
                else:
                    skips.append(f"line {node.lineno}: another `{old}` "
                                 f"definition ({qual})")
            self.vmaps.append(_local_types(node))
            self.generic_visit(node)
            self.vmaps.pop()

        visit_FunctionDef = _fn
        visit_AsyncFunctionDef = _fn

    Walk().visit(tree)
    n = 0
    for row, c0, c1 in sorted(hits, reverse=True):
        ln = lines[row - 1]
        if ln[c0:c1] != old:
            continue
        lines[row - 1] = ln[:c0] + new + ln[c1:]
        n += 1
    return "\n".join(lines), n, skips



_COMMENT_STARTS = {".py": "#", ".rb": "#", ".sh": "#"}


def _rename_generic(text: str, old: str, new: str, suffix: str) -> tuple[str, int]:
    """Whole-word replace outside quotes and `//`/`#` line comments."""
    comment = _COMMENT_STARTS.get(suffix, "//")
    word = re.compile(r"(?<![\w$])" + re.escape(old) + r"(?![\w$])")
    out, n = [], 0
    for line in text.split("\n"):
        res, i, quote = [], 0, ""
        while i < len(line):
            ch = line[i]
            if quote:
                res.append(ch)
                if ch == "\\" and i + 1 < len(line):
                    res.append(line[i + 1])
                    i += 1
                elif ch == quote:
                    quote = ""
                i += 1
                continue
            if ch in ("'", '"', "`"):
                quote = ch
                res.append(ch)
                i += 1
                continue
            if line.startswith(comment, i):
                res.append(line[i:])
                break
            m = word.match(line, i)
            if m:
                res.append(new)
                n += 1
                i = m.end()
                continue
            res.append(ch)
            i += 1
        out.append("".join(res))
    return "\n".join(out), n


def rename_text(rel: str, text: str, old: str, new: str) -> tuple[str, int]:
    """(new text, replacements); -1 when the file could not be tokenised."""
    if rel.endswith(".py"):
        return _rename_python(text, old, new)
    return _rename_generic(text, old, new, Path(rel).suffix.lower())


# --------------------------------------------------------------------------
# Zero-token rename engine (Sprint P): no index, no LLM, no ambiguity games
# --------------------------------------------------------------------------
def _rename_treesitter(text: str, old: str, new: str, suffix: str
                       ) -> tuple[str, int] | None:
    """(new text, replacements) renaming every IDENTIFIER node equal to `old`
    (JS/TS/Go/Rust/... via the same grammars the chunker uses). Strings and
    comments are separate node types and can never be hit. None when no
    grammar is available for this suffix (the caller falls back to the
    quote-and-comment-aware word matcher)."""
    from .chunker_ts import LANG_BY_EXT, TS_AVAILABLE, _get_parser

    lang = LANG_BY_EXT.get(suffix)
    if not (TS_AVAILABLE and lang):
        return None
    import gc
    data = text.encode("utf-8")
    needle = old.encode("utf-8")
    was = gc.isenabled()
    gc.disable()          # py-tree-sitter GC segfault guard (see chunker_ts)
    try:
        tree = _get_parser(lang).parse(data)
        hits: list[tuple[int, int]] = []
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if (node.type in _TS_IDENT_TYPES
                    and data[node.start_byte:node.end_byte] == needle):
                hits.append((node.start_byte, node.end_byte))
            stack.extend(node.children)
    except Exception:
        return None
    finally:
        tree = None
        if was:
            gc.enable()
    if not hits:
        return text, 0
    pieces: list[bytes] = []
    prev = len(data)
    replacement = new.encode("utf-8")
    for s, e in sorted(hits, reverse=True):
        pieces.append(data[e:prev])
        pieces.append(replacement)
        prev = s
    pieces.append(data[:prev])
    pieces.reverse()
    return b"".join(pieces).decode("utf-8", "replace"), len(hits)


def _iter_project_files(root: Path) -> list[str]:
    """Project-relative source files the rename may consider: Python plus
    every tree-sitter language, skipping VCS/tooling/dependency directories,
    gitignore/.pedacitoignore-matched paths, and files too large to be
    source."""
    from .chunker_ts import LANG_BY_EXT
    from .select import Matcher

    exts = {".py"} | set(LANG_BY_EXT)
    matcher = Matcher.from_files(root)
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in _SKIP_DIR_NAMES and not d.startswith(".")
            and not d.endswith(".egg-info")
            and not matcher.excluded(
                (Path(dirpath) / d).relative_to(root).as_posix() + "/"))
        for f in sorted(filenames):
            if Path(f).suffix.lower() not in exts:
                continue
            p = Path(dirpath) / f
            rel = p.relative_to(root).as_posix()
            if matcher.excluded(rel):
                continue
            try:
                if p.stat().st_size > _MAX_RENAME_BYTES:
                    continue
            except OSError:
                continue
            out.append(rel)
    return out


def _scoped_files(root: Path, scope_file: str) -> list[str]:
    """`scope_file` plus the files that import it: the resolved import graph
    from a saved index when there is one, otherwise a textual scan of the
    import statements for the module stem."""
    rel = fileio.normalise_rel(scope_file)
    if not (root / rel).is_file():
        raise ValueError(f"scope file not found in the project: {rel}")
    try:
        from .index import Index
        idx = Index.load(root)
    except Exception:
        idx = None
    if idx is not None:
        mapped = idx.resolve_file(rel) or rel
        return [mapped] + [f for f in idx.importers_of(mapped)
                           if f != mapped]
    stem = Path(rel).stem
    pat = re.compile(r"(?:from|import|require|include|use)[^\n]*"
                     + re.escape(stem) + r"\b")
    out = [rel]
    for cand in _iter_project_files(root):
        if cand == rel:
            continue
        try:
            text = fileio.read_text(root / cand)
        except (OSError, UnicodeDecodeError):
            continue
        if pat.search(text):
            out.append(cand)
    return out


def rename_symbol_across_project(root, old_name: str, new_name: str,
                                 scope_file: str | None = None,
                                 dry_run: bool = False,
                                 before_write=None) -> dict:
    """Deterministic, zero-token multi-file symbol rename.

    Replaces every EXACT identifier token `old_name` with `new_name` across
    the indexed project source (or `scope_file` plus its importers): Python
    through the real `tokenizer` and JS/TS/Go/Rust/... through tree-sitter
    IDENTIFIER nodes, so string literals and comments that merely CONTAIN
    the old name are never touched. Every candidate file is validated with
    the same AST / tree-sitter syntax gate the edit path uses BEFORE
    anything is written: one file that the rename would break refuses the
    whole run (nothing lands).

    Returns {ok, files_modified, total_replacements, diffs} plus dry_run /
    skipped / warnings; with `dry_run` nothing is written. `before_write`
    (rel, path) -> error-or-None lets a caller snapshot or permission-gate
    each file just before the write pass.
    """
    from .edits import _validate

    root = Path(root).resolve()
    old_name = (old_name or "").strip()
    new_name = (new_name or "").strip()
    empty = {"ok": False, "files_modified": [], "total_replacements": 0,
             "diffs": {}, "dry_run": bool(dry_run)}

    def fail(msg: str) -> dict:
        return {**empty, "error": msg}

    if not old_name or not old_name.isidentifier():
        return fail(f"`{old_name}` is not a valid identifier.")
    if not new_name or not new_name.isidentifier():
        return fail(f"`{new_name}` is not a valid identifier.")
    if keyword.iskeyword(old_name) or keyword.iskeyword(new_name):
        return fail("renaming a keyword is not a symbol rename; pick "
                    "identifier names.")
    if old_name == new_name:
        return fail("the old and new names are the same; nothing to rename.")

    try:
        files = (_scoped_files(root, scope_file) if scope_file
                 else _iter_project_files(root))
    except ValueError as e:
        return fail(str(e))

    changes: dict[str, tuple[str, str, int, object]] = {}
    skipped: list[str] = []
    warnings: list[str] = []
    for rel in files:
        why = safety.is_protected(rel)
        if why:
            skipped.append(f"{rel} (protected: {why})")
            continue
        path = root / rel
        try:
            text, style = fileio.read(path, errors="strict")
        except (OSError, UnicodeDecodeError):
            skipped.append(f"{rel} (unreadable)")
            continue
        suffix = Path(rel).suffix.lower()
        if suffix == ".py":
            new_text, n = _rename_python(text, old_name, new_name)
            if n < 0:
                skipped.append(f"{rel} (could not be tokenized)")
                continue
        else:
            res = _rename_treesitter(text, old_name, new_name, suffix)
            if res is None:
                new_text, n = _rename_generic(text, old_name, new_name,
                                              suffix)
                if n > 0:
                    warnings.append(f"{rel}: renamed outside quotes and "
                                    "comments by the word matcher (no "
                                    "tree-sitter grammar available)")
            else:
                new_text, n = res
        if n == 0:
            continue
        bad = _validate(path, new_text)
        if bad and not _validate(path, text):
            return fail(f"the rename would leave {rel} with {bad}; "
                        "nothing written.")
        if bad:
            warnings.append(f"{rel} does not parse -- it did not parse "
                            "before the rename either; the renamed text "
                            "is kept as-is")
        changes[rel] = (text, new_text, n, style)

    diffs = {rel: "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        f"a/{rel}", f"b/{rel}"))[:4000]
        for rel, (before, after, _n, _s) in changes.items()}

    if before_write is not None and not dry_run:
        for rel in changes:
            err = before_write(rel, root / rel)
            if err:
                return fail(f"{err}; nothing written.")

    if not dry_run:
        written: list[str] = []
        try:
            for rel, (_before, after, _n, style) in changes.items():
                fileio.write(root / rel, after, style)
                written.append(rel)
        except OSError as e:
            restored = 0
            for rel in written:
                try:
                    fileio.write(root / rel, changes[rel][0], changes[rel][3])
                    restored += 1
                except OSError:
                    pass
            return fail(f"write failed mid-rename ({e}); restored "
                        f"{restored} of {len(written)} already-written "
                        "file(s)"
                        + ("" if restored == len(written)
                           else " -- `pedacito restore` can finish the job"))

    return {"ok": True,
            "files_modified": sorted(changes),
            "total_replacements": sum(n for _b, _a, n, _s
                                      in changes.values()),
            "diffs": diffs,
            "dry_run": bool(dry_run),
            "per_file": {rel: n for rel, (_b, _a, n, _s)
                         in changes.items()},
            "skipped": skipped,
            "warnings": warnings}


def tool_rename_symbol(index, cfg, old: str, new: str, file: str = "",
                       dry_run: bool = False) -> str:
    """Agent/GUI/CLI wrapper around `rename_symbol_across_project`: permission
    gates, task snapshots, and an index refresh over the changed files."""
    from .tools import (
        _note_mutation,
        _protected_error,
        _snapshot,
        _task_allowlist_error,
    )

    if not dry_run and not getattr(cfg, "allow_write", False):
        return "ERROR: `rename_symbol` needs allow_write = true."
    root = Path(index.root).resolve()

    def _gate(rel: str, path: Path) -> str | None:
        return (_protected_error(cfg, rel, "rename_symbol")
                or _task_allowlist_error(index, cfg, rel, "rename_symbol")
                or _snapshot(index, cfg, rel, path))

    res = rename_symbol_across_project(
        root, old, new, scope_file=(file or None), dry_run=dry_run,
        before_write=_gate if not dry_run else None)
    if not res["ok"]:
        return "ERROR: " + str(res.get("error", "rename failed"))
    if not res["files_modified"]:
        return (f"NOTHING TO DO: no occurrence of `{old}` found in the "
                f"{'scoped' if file else 'project'} source files"
                + (f" (skipped: {'; '.join(res['skipped'])})"
                   if res.get("skipped") else "") + ".")
    head = (f"RENAME PREVIEW `{old}` -> `{new}`" if dry_run
            else f"RENAMED `{old}` -> `{new}`")
    out = [f"{head}: {res['total_replacements']} replacement(s) in "
           f"{len(res['files_modified'])} file(s): "
           + ", ".join(res["files_modified"]),
           "".join(res["diffs"].values())[:3000]]
    if res.get("warnings"):
        out.append("WARNINGS: " + "; ".join(res["warnings"]))
    if res.get("skipped"):
        out.append("NOT checked: " + "; ".join(res["skipped"][:6]))
    if dry_run:
        out.append("(dry run: nothing was written)")
    else:
        out.append("Backed up; `pedacito restore` undoes it."
                   if getattr(cfg, "backup", False) else "Backups are OFF.")
        from . import index as index_mod
        for rel in res["files_modified"]:
            index_mod.reindex_file(index, rel, cfg, save=False)
        _note_mutation(cfg, *res["files_modified"])
    return "\n".join(x for x in out if x)


def tool_rename(index, cfg, old: str, new: str, file: str = "") -> str:
    """Rename a definition and its high-confidence references."""
    from .edits import _validate
    from .tools import (
        _note_mutation,
        _protected_error,
        _snapshot,
        _task_allowlist_error,
    )

    old = (old or "").strip()
    new = (new or "").strip()
    if not getattr(cfg, "allow_write", False):
        return "ERROR: `rename` needs allow_write = true."
    if not old or not new:
        return "ERROR: rename needs `symbol` (the old name) and `to` (the new one)."
    base_old, base_new = old.split(".")[-1], new.split(".")[-1]
    if not _IDENT.match(base_new) or keyword.iskeyword(base_new):
        return f"ERROR: `{new}` is not a valid identifier."
    if base_old == base_new:
        return "ERROR: the old and new names are the same."
    qualified = "." in old
    class_path = old.rsplit(".", 1)[0] if qualified else ""
    if qualified:
        # the dotted query is a QUALNAME, and methods live in their class
        # chunk's `members` (not as chunks of their own) -- the class homes
        # resolved below supply `defs`.
        defs: list = []
    else:
        defs = [c for c in index.find_symbol(old)
                if c.kind != "module"
                and c.qualname.split("#")[0].split(".")[-1] == base_old]
        if file:
            rel = index.resolve_file(file)
            defs = [c for c in defs if c.file == rel]
        if not defs:
            return (f"ERROR: no definition named `{old}` in the index. "
                    "`find` it first.")
    warnings: list[str] = []
    if qualified:
        # a qualified query is disambiguated by RECEIVER, not by bare-name
        # uniqueness: `Store.save` is unique even when `Other.save` exists,
        # because the rename pass can tell the receivers apart. A module-
        # prefixed query (`pkg.Store.save`) falls back to its class suffix.
        for cp in ([class_path] +
                   ([class_path.rsplit(".", 1)[-1]]
                    if "." in class_path else [])):
            homes_q = {(c.file, cp) for c in index.chunks
                       if c.kind in ("class", "classheader")
                       and c.qualname.split("#")[0] == cp
                       and base_old in (getattr(c, "members", None) or [])}
            if homes_q:
                class_path = cp
                break
        else:
            homes_q = set()
        if len(homes_q) > 1:
            listing = ", ".join(f"{f}:{q}" for f, q in sorted(homes_q)[:6])
            return (f"ERROR: `{class_path}.{base_old}` is defined in "
                    f"{len(homes_q)} places ({listing}); a mechanical rename "
                    "would touch the wrong ones. Rename with edit blocks.")
        if not homes_q:
            return (f"ERROR: the index has no `{class_path}.{base_old}` "
                    "definition to rename (is the class name spelled as the "
                    "index sees it?).")
        defs = [c for c in index.chunks
                if c.kind in ("class", "classheader")
                and (c.file, c.qualname.split("#")[0]) in homes_q]
        if file:
            rel = index.resolve_file(file)
            defs = [c for c in defs if c.file == rel]
    else:
        # every definition sharing the bare name, anywhere -- methods count
        # too, and they live in the class chunk's `members`, never as chunks
        same = [c for c in index.chunks if c.kind not in ("module", "reference")
                and c.qualname.split("#")[0].split(".")[-1] == base_old
                and c.qualname.split("#")[0] != "module"]
        homes = {(c.file, c.qualname.split("#")[0]) for c in same}
        homes |= {(c.file, f"{c.qualname.split('#')[0]}.{base_old}")
                  for c in index.chunks if c.kind in ("class", "classheader")
                  and base_old in (getattr(c, "members", None) or [])}
        if len(homes) > 1:
            listing = ", ".join(f"{f}:{q}" for f, q in sorted(homes)[:6])
            return (f"ERROR: `{base_old}` is defined in {len(homes)} places "
                    f"({listing}); a mechanical rename would touch the wrong "
                    "ones. Rename with edit blocks, or pass `file` ... and use "
                    "unique names.")
    taken = any(c.qualname.split("#")[0].split(".")[-1] == base_new
                for c in index.chunks if c.kind not in ("module", "reference"))
    if not qualified and not taken:
        taken = any(base_new in (getattr(c, "members", None) or [])
                    for c in index.chunks
                    if c.kind in ("class", "classheader"))
    if taken:
        return f"ERROR: `{base_new}` is already defined; pick an unused name."
    if not defs:
        return (f"ERROR: no definition named `{old}` in the index"
                + (f" in `{file}`" if file else "") + ". `find` it first.")
    def_file = defs[0].file
    targets = [def_file] + [f for f in index.importers_of(def_file)
                            if f != def_file]
    root = Path(index.root).resolve()
    if qualified and def_file.endswith(".py"):
        # files that CONSTRUCT the class but never import its module still
        # hold call sites the receiver pass can see; include those too
        extra = []
        cpat = re.compile(r"(?<![\w.])" + re.escape(class_path.rsplit(".", 1)[-1])
                          + r"\s*\(")
        for rel in index.files:
            if rel in targets or not rel.endswith(".py"):
                continue
            try:
                _t = fileio.read_text(root / rel)
            except (OSError, UnicodeDecodeError):
                continue
            if cpat.search(_t):
                extra.append(rel)
        targets += extra
    changes: dict[str, tuple[str, str, int, object]] = {}
    skipped_sites: list[str] = []
    for rel in targets:
        err = _protected_error(cfg, rel, "rename") \
            or _task_allowlist_error(index, cfg, rel, "rename")
        if err:
            return err
        path = root / rel
        try:
            text, style = fileio.read(path, errors="strict")
        except (OSError, UnicodeDecodeError):
            continue
        site_skips: list[str] = []
        if qualified and rel.endswith(".py"):
            new_text, n, site_skips = _rename_python_receiver(
                text, base_old, base_new, class_path)
            skipped_sites += [f"{rel}:{s}" for s in site_skips]
        else:
            new_text, n = rename_text(rel, text, base_old, base_new)
            if n < 0:
                return f"ERROR: could not parse {rel}; rename it with edit blocks."
            if qualified and n > 0 and not rel.endswith(".py"):
                warnings.append(f"{rel}: receiver tracking is Python-only; "
                                f"`{base_old}` was renamed there by whole-"
                                "word match -- verify it is the same member")
        if n == 0:
            continue
        bad = _validate(path, new_text)
        if bad and not _validate(path, text):
            return f"ERROR: the rename would leave {rel} with {bad}; nothing written."
        changes[rel] = (text, new_text, n, style)
    if not changes:
        msg = f"ERROR: found no occurrence of `{base_old}` to rename."
        if skipped_sites:
            msg += (" every candidate site was ambiguous or bound to another "
                    f"type: {'; '.join(skipped_sites[:6])}")
        return msg
    # references we deliberately did not touch
    rx = re.compile(r"(?<![\w$])" + re.escape(base_old) + r"(?![\w$])")
    untouched: list[str] = []
    for rel in index.files:
        if rel in changes or rel in targets:
            continue
        try:
            lines = fileio.read_lines_cached(root / rel)
        except OSError:
            continue
        for i, ln in enumerate(lines, 1):
            if rx.search(ln):
                untouched.append(f"{rel}:{i}")
                break
    diffs = []
    for rel, (before, after, n, _s) in changes.items():
        d = "".join(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            f"a/{rel}", f"b/{rel}", n=0))
        diffs.append(d[:1500])
    written: list[str] = []
    abort: str | None = None
    try:
        for rel, (before, after, n, style) in changes.items():
            err = _snapshot(index, cfg, rel, root / rel)
            if err:
                abort = err
                break
            fileio.write(root / rel, after, style)
            written.append(rel)
    except OSError as e:
        abort = f"could not write ({e})"
    if abort is not None:
        for rel in written:                     # all-or-nothing, always
            try:
                fileio.write(root / rel, changes[rel][0], changes[rel][3])
            except OSError:
                pass
        return (f"ERROR: rename aborted: {abort}; rolled back {len(written)} "
                "already-written file(s); nothing was renamed.")
    from . import index as index_mod
    for rel in changes:
        index_mod.reindex_file(index, rel, cfg, save=False)
    _note_mutation(cfg, *changes)
    total = sum(n for *_x, n, _s in changes.values())
    out = [f"RENAMED `{base_old}` -> `{base_new}`: {total} replacement(s) in "
           f"{len(changes)} file(s): {', '.join(changes)}",
           ("Backed up; `pedacito restore` undoes it."
            if getattr(cfg, "backup", False) else "Backups are OFF."),
           "".join(diffs)[:3000]]
    if skipped_sites:
        out.append("NOT renamed (ambiguous or foreign receivers): "
                   + "; ".join(skipped_sites[:8])
                   + (f", +{len(skipped_sites) - 8} more"
                      if len(skipped_sites) > 8 else "")
                   + " -- annotate those receivers (`x: "
                   + class_path.rsplit(".", 1)[-1] + "`) or fix them with "
                     "edit blocks.")
    if warnings:
        out.append("WARNINGS: " + "; ".join(warnings))
    if untouched:
        out.append(f"NOT changed (files that do not import the definition, or "
                   f"prose): {', '.join(untouched[:8])}"
                   + (f", +{len(untouched) - 8} more" if len(untouched) > 8 else "")
                   + ". Check them with `grep` and fix with edit blocks if "
                     "they refer to this symbol.")
    return "\n".join(x for x in out if x)
