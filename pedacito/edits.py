"""Edit application via search/replace blocks.

Usage
-----
    from .edits import parse_edits, apply_edits

    edits = parse_edits(model_reply_text)
    results = apply_edits(project_root, edits, backup=True, dry_run=False)
    for r in results:
        print(r.ok, r.file, r.message)

Why search/replace rather than unified diffs: small models get hunk headers
and line offsets wrong constantly, and a wrong @@ header silently corrupts a
file. Why not whole-file rewrites: at ~25 tok/s a 400-line file is minutes of
generation, and the model will quietly drop a function somewhere in the
middle.

Search/replace instead anchors on content the model just read via the `read`
tool. It either matches the file exactly or it doesn't, and a failed match is
a clean, recoverable error that can be fed back to the model to retry.

Expected format in a model's reply:

    --- FILE: path/to/file.py
    <<<<<<< SEARCH
    exact original lines
    =======
    replacement lines
    >>>>>>> REPLACE
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from . import fileio
from .chunker_ts import LANG_BY_EXT, TS_AVAILABLE, has_syntax_error

BLOCK_RE = re.compile(
    r"---\s*FILE:\s*(?P<file>[^\n]+?)\s*\n"
    r"[^\n]*<{5,}\s*SEARCH[^\n]*\n"
    r"(?P<search>.*?)"
    r"\n={5,}[^\n]*\n"
    r"(?P<replace>.*?)"
    r"\n>{5,}\s*REPLACE",
    re.DOTALL,
)


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass
class Edit:
    """One parsed search/replace block, not yet applied."""

    file: str
    search: str
    replace: str


@dataclass
class Result:
    """The outcome of attempting one file's edits."""

    ok: bool
    file: str
    message: str
    diff: str = ""
    # Populated on success. The reviewer renders before/after side by side, and
    # `sha` pins the exact bytes the preview was built from so a later apply can
    # refuse if the file changed underneath.
    before: str = ""
    after: str = ""
    sha: str = ""


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def parse_edits(text: str) -> list[Edit]:
    """Extract every search/replace block from a model's reply text.

    Normalises the model's own line endings first, then strips markdown
    fences so a model that wraps its blocks in ``` still parses.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"^\s*```[a-zA-Z]*\s*$", "", text, flags=re.MULTILINE)
    return [
        Edit(fileio.normalise_rel(m.group("file")), m.group("search"), m.group("replace"))
        for m in BLOCK_RE.finditer(cleaned)
    ]


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------
def _locate(haystack: str, needle: str) -> tuple[int, int] | None:
    """Find the unique span of `needle` inside `haystack`: exact match first,
    then a whitespace-tolerant retry (ignoring trailing whitespace per line).
    Ambiguous matches (needle appears more than once) are treated as a
    failure -- better to ask the model for more surrounding context than to
    edit the wrong occurrence."""
    if not needle.strip():
        return None
    n = haystack.count(needle)
    if n == 1:
        i = haystack.index(needle)
        return i, i + len(needle)
    if n > 1:
        return None

    # Retry ignoring trailing whitespace and blank-line differences.
    def norm(s: str) -> list[str]:
        """Split into lines with trailing whitespace stripped, for the
        whitespace-tolerant retry."""
        return [ln.rstrip() for ln in s.splitlines()]

    hay, need = norm(haystack), norm(needle)
    if not need:
        return None
    starts = [i for i in range(len(hay) - len(need) + 1) if hay[i : i + len(need)] == need]
    if len(starts) != 1:
        return None
    s = starts[0]
    lines = haystack.splitlines(keepends=True)
    seg = lines[s : s + len(need)]
    start = sum(len(x) for x in lines[:s])
    # Exclude the final line terminator so the span matches exact-match
    # semantics, where the needle has no trailing newline.
    tail = len(seg[-1]) - len(seg[-1].rstrip("\r\n"))
    end = start + sum(len(x) for x in seg) - tail
    return start, end


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def apply_edits(root: Path, edits: list[Edit], backup: bool = True,
                dry_run: bool = False, backup_dir: Path | None = None) -> list[Result]:
    """Apply a list of Edits, grouped by file.

    For each file: locate and apply every search/replace block, validate the
    result parses as Python (if a .py file), and either write it (with a
    backup of the original) or, in dry-run mode, just compute the diff and
    before/after text without touching disk.

    Backups go into a timestamped snapshot directory under .pedacito/backups/
    rather than dropping .bak files beside the source, so a whole editing
    session is restorable as a unit and nothing Pedacito writes ever lands in
    the working tree unexpectedly.
    """
    results: list[Result] = []
    by_file: dict[str, list[Edit]] = {}
    for e in edits:
        by_file.setdefault(e.file, []).append(e)

    for rel, group in by_file.items():
        path = (root / rel).resolve()
        if not path.exists():
            results.append(Result(False, rel, "file not found"))
            continue
        try:
            path.relative_to(root.resolve())
        except ValueError:
            results.append(Result(False, rel, "refusing to edit outside project root"))
            continue

        # LF-normalised for matching; style remembered so the file is written
        # back exactly as it was found.
        original, style = fileio.read(path, errors="strict")
        text = original
        failed = False
        for k, e in enumerate(group, 1):
            span = _locate(text, e.search)
            if span is None:
                n = text.count(e.search)
                why = ("search block not found -- it must match the file exactly"
                       if n == 0 else
                       f"search block appears {n} times -- include more surrounding lines")
                results.append(Result(False, rel, f"block {k}: {why}"))
                failed = True
                break
            text = text[: span[0]] + e.replace + text[span[1] :]

        if failed:
            continue

        if path.suffix == ".py":
            try:
                ast.parse(text)
            except SyntaxError as e:
                results.append(Result(False, rel,
                                      f"edit produced a syntax error at line {e.lineno}: {e.msg}; not written"))
                continue
        elif path.suffix.lower() in LANG_BY_EXT and TS_AVAILABLE:
            bad, lineno = has_syntax_error(text, LANG_BY_EXT[path.suffix.lower()])
            if bad:
                results.append(Result(False, rel,
                                      f"edit produced a syntax error near line {lineno}; not written"))
                continue

        diff = "".join(difflib.unified_diff(
            original.splitlines(keepends=True), text.splitlines(keepends=True),
            fromfile=f"a/{rel}", tofile=f"b/{rel}",
        ))

        sha = hashlib.sha256(original.encode("utf-8")).hexdigest()
        if dry_run:
            results.append(Result(True, rel, f"{len(group)} block(s) would apply",
                                  diff, original, text, sha))
            continue

        if backup:
            if backup_dir is not None:
                dest = backup_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                fileio.write(dest, original, style)
            else:
                fileio.write(path.with_suffix(path.suffix + ".bak"), original, style)
        fileio.write(path, text, style)
        results.append(Result(True, rel, f"applied {len(group)} block(s)",
                              diff, original, text, sha))

    return results
