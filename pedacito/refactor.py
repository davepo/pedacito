"""Deterministic symbol rename: the multi-file edit small models get wrong.

    tool_rename(index, cfg, "apply_discount", "discounted_price")

Renaming across files is exactly where a small model drops one call site. This
does it mechanically, and only where it is safe:

  * the symbol must have ONE definition (another function with the same bare
    name anywhere makes the rename ambiguous: refused, with the candidates);
  * the new name must be a valid identifier that nothing else defines;
  * only the definition's file and the files that IMPORT it are rewritten --
    whole-word, outside string literals and comments (Python uses the real
    tokenizer);
  * every rewritten file must still parse (same check the edit path uses);
  * references it did NOT touch (files that do not import the definition,
    docs, strings) are listed so the model can handle them with edit blocks.

Gated by `allow_write`, snapshotted like every other write (undo with
`pedacito restore`), and the index is refreshed afterwards.
"""

from __future__ import annotations

import difflib
import io
import keyword
import re
import tokenize
from pathlib import Path

from . import fileio

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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


def tool_rename(index, cfg, old: str, new: str, file: str = "") -> str:
    """Rename a definition and its high-confidence references."""
    from .edits import _validate
    from .tools import _protected_error, _snapshot, _task_allowlist_error

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
    defs = [c for c in index.find_symbol(old)
            if c.kind != "module" and c.qualname.split("#")[0].split(".")[-1] == base_old]
    if file:
        rel = index.resolve_file(file)
        defs = [c for c in defs if c.file == rel]
    if not defs:
        return (f"ERROR: no definition named `{old}` in the index. `find` it "
                "first.")
    # every definition sharing the bare name, anywhere
    same = [c for c in index.chunks if c.kind not in ("module", "reference")
            and c.qualname.split("#")[0].split(".")[-1] == base_old
            and c.qualname.split("#")[0] != "module"]
    homes = {(c.file, c.qualname.split("#")[0]) for c in same}
    if len(homes) > 1:
        listing = ", ".join(f"{f}:{q}" for f, q in sorted(homes)[:6])
        return (f"ERROR: `{base_old}` is defined in {len(homes)} places "
                f"({listing}); a mechanical rename would touch the wrong "
                "ones. Rename with edit blocks, or pass `file` ... and use "
                "unique names.")
    if any(c.qualname.split("#")[0].split(".")[-1] == base_new
           for c in index.chunks if c.kind not in ("module", "reference")):
        return f"ERROR: `{base_new}` is already defined; pick an unused name."
    def_file = defs[0].file
    targets = [def_file] + [f for f in index.importers_of(def_file)
                            if f != def_file]
    root = Path(index.root).resolve()
    changes: dict[str, tuple[str, str, int, object]] = {}
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
        new_text, n = rename_text(rel, text, base_old, base_new)
        if n < 0:
            return f"ERROR: could not parse {rel}; rename it with edit blocks."
        if n == 0:
            continue
        bad = _validate(path, new_text)
        if bad and not _validate(path, text):
            return f"ERROR: the rename would leave {rel} with {bad}; nothing written."
        changes[rel] = (text, new_text, n, style)
    if not changes:
        return f"ERROR: found no occurrence of `{base_old}` to rename."
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
    for rel, (before, after, n, style) in changes.items():
        err = _snapshot(index, cfg, rel, root / rel)
        if err:
            return err
        fileio.write(root / rel, after, style)
    from . import index as index_mod
    for rel in changes:
        index_mod.reindex_file(index, rel, cfg, save=False)
    total = sum(n for *_x, n, _s in changes.values())
    out = [f"RENAMED `{base_old}` -> `{base_new}`: {total} replacement(s) in "
           f"{len(changes)} file(s): {', '.join(changes)}",
           ("Backed up; `pedacito restore` undoes it."
            if getattr(cfg, "backup", False) else "Backups are OFF."),
           "".join(diffs)[:3000]]
    if untouched:
        out.append(f"NOT changed (files that do not import the definition, or "
                   f"prose): {', '.join(untouched[:8])}"
                   + (f", +{len(untouched) - 8} more" if len(untouched) > 8 else "")
                   + ". Check them with `grep` and fix with edit blocks if "
                     "they refer to this symbol.")
    return "\n".join(x for x in out if x)
