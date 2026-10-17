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

An empty SEARCH section against a file that does not exist yet creates the
file with the REPLACE content -- the agent's "write a new file" move, so it
can add modules without a separate action vocabulary.
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
    # The separator is `=======` at the end of what it consumed: optionally
    # one newline plus trailing spaces/tabs. `[^\n]*` here used to let the
    # REPLACE body leak into the separator line when a model wrote `=======`
    # directly above `>>>>>>> REPLACE` (an empty replacement), silently
    # dropping whole edit blocks.
    r"={5,}[ \t]*(?:\n[ \t]*)?"
    # The replace body may be empty too: with no newline after the separator
    # the closer sits on the very next line (`>{5,}` after `\n?`). A negative
    # look-ahead keeps the group from swallowing the next block's header when
    # two blocks sit adjacent.
    r"(?P<replace>(?:(?!-{3}\s*FILE:).)*?)"
    r"\n?>{5,}\s*REPLACE",
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
    # True when at least one block in this file anchored approximately rather
    # than matching exactly. Reviewers use it to flag the risky ones.
    fuzzy: bool = False


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
# A bare ``` line at column zero toggles fenced-code state while scanning a
# reply; anything inside a fence is content and must never be rewritten.
_FENCE_RE = re.compile(r"^\s{0,3}```.*$")


def _strip_wrapping_fences(text: str) -> str:
    """Blank out only the markdown fences that *wrap* the edit blocks -- an
    opening ``` before the first `--- FILE:` header and a closing ``` after
    the last `>>>>>>> REPLACE`.

    The naive version of this (delete every line that looks like a fence)
    corrupted real edits: replacement code that opens or closes a docstring
    containing a fenced example lands exactly such a line inside the REPLACE
    section, and deleting it silently wrote broken code to disk. Fences
    between two `--- FILE:` headers are therefore left untouched; a model
    that fences each block individually is handled by the outermost pair,
    which also blanks any stray inter-block fences sitting outside every
    search/replace block."""
    lines = text.split("\n")
    first = text.find("--- FILE:")
    if first == -1:
        return text
    # Opening fence: the last bare ``` line before the first header. The scan
    # stops as soon as it reaches the line that actually holds the header --
    # comparing whole-line lengths to a character offset was wrong and could
    # stop early on a short preamble, leaving the real opening fence in place.
    open_idx = -1
    fenced = False
    consumed = 0
    for i, ln in enumerate(lines):
        if _FENCE_RE.match(ln):
            open_idx = -1 if fenced else i
            fenced = not fenced
        if consumed <= first < consumed + len(ln) + 1:
            break                        # this line contains the header
        consumed += len(ln) + 1          # +1 for the newline we split on
    # Closing fence: the first bare ``` line after the final REPLACE marker.
    close_idx = -1
    if "REPLACE" in text:
        tail_from = max(i for i in range(len(lines)) if "REPLACE" in lines[i])
        for j in range(tail_from + 1, len(lines)):
            if _FENCE_RE.match(lines[j]):
                close_idx = j
                break
    for k in (open_idx, close_idx):
        if k != -1:
            lines[k] = ""
    return "\n".join(lines)


def parse_edits(text: str) -> list[Edit]:
    """Extract every search/replace block from a model's reply text.

    Normalises the model's own line endings first, then strips the markdown
    fences wrapping the blocks so a model that answers inside a ``` still
    parses -- without touching fence lines that belong to the code itself.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _strip_wrapping_fences(text)
    # A fence line blanked by the stripper can leave the header glued to the
    # line above it; make sure every header starts its own line.
    cleaned = re.sub(r"(?<!\n)(---\s*FILE:)", r"\n\1", cleaned)
    edits = []
    for m in BLOCK_RE.finditer(cleaned):
        search, replace = m.group("search"), m.group("replace")
        # The separator/closer may sit on the same line as the last body line
        # (an empty replacement writes `=======` straight onto
        # `>>>>>>> REPLACE`). Drop exactly one trailing newline so a normal
        # block keeps its historical "no trailing newline" semantics while an
        # empty replacement stays empty instead of becoming "\n".
        if search.endswith("\n"):
            search = search[:-1]
        if replace.endswith("\n"):
            replace = replace[:-1]
        edits.append(Edit(fileio.normalise_rel(m.group("file")), search, replace))
    return edits


def merge_duplicate_files(edits: list[Edit]) -> list[Edit]:
    """Fold blocks that repeat the same `--- FILE:` header into one group.

    Small models habitually re-emit a file's header instead of appending a
    second block under it, and separate passes over one path are unsafe in
    both directions: in dry-run (review) mode each pass computes its diff
    from the *original* text, so the review page shows only the last version
    and `write_files` refuses with a misleading "(deleted)" or "(changed on
    disk)"; live, a creation block can run before an edit block for the same
    path and clobber what was just created. Concatenating the bodies keeps
    every intent in reply order -- empty SEARCH sections included.
    """
    seen: dict[str, int] = {}
    merged: list[Edit] = []
    for e in edits:
        if e.file in seen:
            prev = merged[seen[e.file]]
            prev.search += "\n" + e.search
            prev.replace += "\n" + e.replace
        else:
            seen[e.file] = len(merged)
            merged.append(e)
    return merged


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------
@dataclass
class Span:
    """One located search block: the half-open [start, stop) character span in
    the file text, what to write in its place, and how it matched."""

    start: int
    stop: int
    replacement: str
    fuzzy: bool = False


def locate(haystack: str, needle: str, cfg=None) -> "Span | None":
    """Locate one search block in a file's text and return a Span, or None if
    it cannot be applied.

    Tries, in order: an exact unique match; a whitespace-tolerant unique match
    (trailing whitespace and blank lines ignored), used only when
    `cfg.fuzzy_match` allows it; and finally a similarity match against every
    same-length window in the file, accepted only when the best window is both
    close enough (`cfg.fuzzy_min_ratio`) and clearly better than the runner-up.
    An ambiguous exact match is always a failure: editing the wrong occurrence
    silently is worse than asking the model for more context.
    """
    span = _locate(haystack, needle)
    if span is not None:
        return Span(*span, needle)
    fuzzy_on = bool(cfg is not None and getattr(cfg, "fuzzy_match", True))
    if not fuzzy_on:
        return None
    min_ratio = float(getattr(cfg, "fuzzy_min_ratio", 0.85))
    max_len = int(getattr(cfg, "max_read_lines", 200)) * 200
    return _locate_fuzzy(haystack, needle, min_ratio, max_len)


def _locate_fuzzy(haystack: str, needle: str, min_ratio: float,
                  max_needle: int) -> Span | None:
    """Best-effort anchor for a search block that does not match exactly.

    Compares the needle against every window of haystack with the same number
    of lines, using difflib.SequenceMatcher (no external dependency). A match
    is accepted only if the best window clears `min_ratio` and beats the second
    best by a clear margin, so a genuinely near-miss anchors while an ambiguous
    one still fails loudly. Refuses to run on very large needles: the scan is
    O(lines x length) and a runaway whole-file block should fail, not stall."""
    if not needle.strip() or "\x00" in haystack or len(needle) > max_needle:
        return None
    hlines = haystack.split("\n")
    nlines = needle.split("\n")
    if len(nlines) > len(hlines):
        return None
    target = "\n".join(nlines)
    sm = difflib.SequenceMatcher(None, nlines, [""])
    best_ratio, second_ratio, best_i = 0.0, 0.0, -1
    for i in range(len(hlines) - len(nlines) + 1):
        window = "\n".join(hlines[i:i + len(nlines)])
        if window == target:
            continue                     # an exact match would have been found
        sm.a = nlines
        sm.b = hlines[i:i + len(nlines)]
        r = sm.ratio()
        if r > best_ratio:
            second_ratio = best_ratio
            best_ratio, best_i = r, i
        elif r > second_ratio:
            second_ratio = r
    if best_i < 0 or best_ratio < min_ratio:
        return None
    if second_ratio >= best_ratio - 0.03:
        return None                      # two plausible anchors: too risky
    start = sum(len(x) + 1 for x in hlines[:best_i])
    seg = hlines[best_i:best_i + len(nlines)]
    stop = start + sum(len(x) for x in seg) + (len(seg) - 1)
    return Span(start, stop, needle, fuzzy=True)


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
                dry_run: bool = False, backup_dir: Path | None = None,
                cfg=None) -> list[Result]:
    """Apply a list of Edits, grouped by file.

    For each file: locate and apply every search/replace block, validate the
    result still parses (Python via ast, tree-sitter languages via
    chunker_ts), and either write it (with a backup of the original) or, in
    dry-run mode, just compute the diff and before/after text without
    touching disk. A block with an empty SEARCH against a file that does not
    exist creates that file instead of failing.

    Pass `cfg` (a Config) to enable approximate anchoring (`fuzzy_match`) and
    to control the repair loop's token budgets; without it, matching stays
    exact-only and defaults are used elsewhere.

    Backups go into a timestamped snapshot directory under .pedacito/backups/
    rather than dropping .bak files beside the source, so a whole editing
    session is restorable as a unit and nothing Pedacito writes ever lands in
    the working tree unexpectedly.
    """
    results: list[Result] = []
    # Fold repeated FILE headers into one group per path before anything
    # else: two independent passes over one file each diff against the
    # original text (breaking the review flow) and, live, a creation block
    # can clobber -- or be clobbered by -- an edit block for the same path.
    by_file: dict[str, list[Edit]] = {}
    for e in merge_duplicate_files(edits):
        by_file.setdefault(e.file, []).append(e)

    root_resolved = Path(root).resolve()

    def _validate(path: Path, text: str) -> str | None:
        """Return an error message if `text` does not parse as source of
        `path`'s language, else None. A new file that fails to parse is not
        created at all; an edit that breaks an existing file leaves it
        untouched."""
        if path.suffix == ".py":
            try:
                ast.parse(text)
            except SyntaxError as e:
                return f"a syntax error at line {e.lineno}: {e.msg}"
        elif path.suffix.lower() in LANG_BY_EXT and TS_AVAILABLE:
            bad, lineno = has_syntax_error(text, LANG_BY_EXT[path.suffix.lower()])
            if bad:
                return f"a syntax error near line {lineno}"
        return None

    def _finish(rel: str, path: Path, original: str | None, text: str, style,
                note: str, fuzzy: bool = False) -> Result:
        """Validate, diff, back up, and write one finished file's content."""
        err = _validate(path, text)
        if err:
            verb = "creation aborted" if original is None else "not written"
            return Result(False, rel, f"edit produced {err}; {verb}",
                          fuzzy=fuzzy)

        diff = "".join(difflib.unified_diff(
            (original or "").splitlines(keepends=True), text.splitlines(keepends=True),
            fromfile=f"a/{rel}", tofile=f"b/{rel}",
        ))
        sha = hashlib.sha256((original or "").encode("utf-8")).hexdigest()
        if dry_run:
            return Result(True, rel, note + " (dry run)", diff, original or "",
                          text, sha, fuzzy=fuzzy)

        if backup and original is not None:
            if backup_dir is not None:
                dest = backup_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                fileio.write(dest, original, style)
            else:
                fileio.write(path.with_suffix(path.suffix + ".bak"), original, style)
        fileio.write(path, text, style)
        return Result(True, rel, note, diff, original or "", text, sha,
                      fuzzy=fuzzy)

    for rel, group in by_file.items():
        path = (Path(root) / rel).resolve()
        try:
            path.relative_to(root_resolved)
        except ValueError:
            results.append(Result(False, rel, "refusing to edit outside project root"))
            continue

        exists = path.exists()
        if not exists:
            # New-file creation: only empty-SEARCH blocks can do it. After
            # merge_duplicate_files every path is one group, so a multi-block
            # group here means the model wrote several non-empty searches
            # against a file that isn't in the tree -- a failed edit, not a
            # creation. Fuzzy matching does not apply: with no existing
            # content there is nothing to be similar to. The message must say
            # which case it is, because "file not found" alone invites the
            # model to answer by *renaming* files -- and Pedacito has no
            # rename operation (see the note below).
            creators = [e for e in group if not e.search.strip()]
            if not creators or len(creators) != len(group):
                hint = ""
                if path.parent.is_dir():
                    try:
                        near = sorted(p.name for p in path.parent.iterdir()
                                      if p.is_file())[:8]
                    except OSError:
                        near = []
                    if near:
                        hint = f" Files here: {', '.join(near)}."
                results.append(Result(
                    False, rel,
                    "no edit matched: this path does not exist yet. To CREATE "
                    "the file, emit exactly one block for it with an EMPTY "
                    "SEARCH section (between <<<<<<< SEARCH and =======). "
                    "Pedacito cannot rename or move files -- to 'rename', "
                    "create the new path with an empty-SEARCH block and delete "
                    "the old file's contents yourself." + hint))
                continue
            text = "\n".join(e.replace for e in group)
            if text and not text.endswith("\n"):
                text += "\n"
            path.parent.mkdir(parents=True, exist_ok=True)
            results.append(_finish(rel, path, None, text, None,
                                   f"created with {len(text.splitlines())} line(s)"))
            continue

        # LF-normalised for matching; style remembered so the file is written
        # back exactly as it was found.
        original, style = fileio.read(path, errors="strict")
        text = original
        failed = False
        any_fuzzy = False
        fuzzy_blocks: list[int] = []
        for k, e in enumerate(group, 1):
            span = locate(text, e.search, cfg)
            if span is None:
                n = text.count(e.search)
                why = ("search block not found -- it must match the file exactly"
                       if n == 0 else
                       f"search block appears {n} times -- include more surrounding lines")
                results.append(Result(False, rel, f"block {k}: {why}"))
                failed = True
                break
            start, stop, replacement, fuzzy = span.start, span.stop, span.replacement, span.fuzzy
            text = text[:start] + replacement + text[stop:]
            if fuzzy:
                any_fuzzy = True
                fuzzy_blocks.append(k)

        if failed:
            continue

        note = f"applied {len(group)} block(s)"
        if any_fuzzy:
            note += (" [matched approximately on block"
                     + ("s " if len(fuzzy_blocks) > 1 else " ")
                     + ", ".join(str(b) for b in fuzzy_blocks)
                     + "; review the diff carefully]")
        results.append(_finish(rel, path, original, text, style, note,
                               fuzzy=any_fuzzy))

    return results
