"""Edit application via search/replace blocks.

    edits = parse_edits(model_reply_text)
    results = apply_edits(project_root, edits, backup=True, dry_run=False)
    for r in results:
        print(r.ok, r.file, r.message)

Search/replace rather than unified diffs (small models get hunk headers and
offsets wrong) or whole-file rewrites (slow, and the model drops code): a block
either matches the file exactly or fails with a recoverable error that is fed
back to the model.

Expected format in a model's reply:

    --- FILE: path/to/file.py
    <<<<<<< SEARCH
    exact original lines
    =======
    replacement lines
    >>>>>>> REPLACE

An empty SEARCH section creates the file (or replaces its whole content).

A whole file is removed with a DELETE block (or a one-line directive):

    --- FILE: path/to/file.html
    <<<<<<< DELETE
    >>>>>>> DELETE

    DELETE_FILE: path/to/file.html

The file is backed up into the run's snapshot first, so `undo` / `pedacito
restore` brings it back byte-for-byte. An empty REPLACE is NOT a deletion.

The FILE header may carry an anchor instead of a SEARCH body:

    --- FILE: path/to/file.py LINES 40-60
    --- FILE: path/to/file.py SYMBOL Server.start

For an anchored block the REPLACE body overwrites that 1-based inclusive line
range (LINES) or that definition (SYMBOL, resolved to a line range by the
caller -- the agent, which owns the index -- or, when no index resolved it,
by `chunker.locate_symbol_span` straight from the AST), and SEARCH is ignored.

Zero-anchor surgical splicing needs no FILE header and no SEARCH
body at all: the symbol's current span comes from the AST, and the block's
indent is re-aligned to the target automatically:

    <<<<<<< SYMBOL path/to/file.py :: ClassName.method_name
    def method_name(self, x: int) -> int:
        return x + 1
    >>>>>>> END_SYMBOL

`<<<<<<< AFTER_SYMBOL path :: symbol ... >>>>>>> END_SYMBOL` inserts the body
directly AFTER that definition instead of replacing it. The same two edits
exist as structured payloads (a JSON object in the reply):

    {"edit_action": "replace_symbol", "file": "...", "symbol": "...",
     "replacement": "..."}
    {"edit_action": "insert_after_symbol", "file": "...", "symbol": "...",
     "replacement": "..."}

"""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from . import fileio, safety, security
from .chunker import file_symbol_spans, locate_symbol_span
from .chunker_ts import LANG_BY_EXT, TS_AVAILABLE, has_syntax_error

# The `=======` separator must be a whole line, so a `/* ====` banner at the
# start of a SEARCH stays in the body. The REPLACE body may be empty, and the
# look-ahead keeps it from swallowing the next block's header.
BLOCK_RE = re.compile(
    r"---\s*FILE:\s*(?P<file>[^\n]+?)\s*\n"
    r"[^\n]*<{5,}\s*SEARCH[^\n]*\n"
    r"(?P<search>.*?)"
    r"^[ \t]*={5,}[ \t]*$\n"
    r"(?P<replace>(?:(?!-{3}\s*FILE:).)*?)"
    r"^[ \t]*>{5,}\s*REPLACE",
    re.DOTALL | re.MULTILINE,
)

# Optional anchor suffix on the FILE header: "path LINES 40-60" replaces that
# 1-based inclusive line range with the REPLACE body (SEARCH ignored);
# "path SYMBOL name" replaces that definition (resolved to lines by the agent).
_ANCHOR_RE = re.compile(
    r"^(?P<path>.*?)[ \t]+(?:LINES[ \t]+(?P<start>\d+)[ \t]*-[ \t]*(?P<end>\d+)"
    r"|SYMBOL[ \t]+(?P<symbol>\S+))[ \t]*$",
    re.IGNORECASE,
)

# Whole-file deletion: a FILE header plus `<<<<<<< DELETE` / `>>>>>>> DELETE`
# markers (the body between them, if any, is ignored), or a bare one-line
# `DELETE_FILE: path` directive. Deletion backs the file up before unlinking,
# so the run's undo restores it; an empty REPLACE never deletes a file.
DELETE_BLOCK_RE = re.compile(
    r"---\s*FILE:\s*(?P<file>[^\n]+?)\s*\n"
    r"[^\n]*<{5,}[ \t]*DELETE[^\n]*\n"
    r"(?P<body>.*?)"
    r"^[ \t]*>{5,}[ \t]*DELETE",
    re.DOTALL | re.MULTILINE,
)
DELETE_FILE_RE = re.compile(
    r"(?m)^[ \t]*DELETE_FILE:[ \t]*(?P<file>[^\n]+?)[ \t]*$")
# An opened-but-unclosed DELETE marker (for malformed_blocks reporting).
_DELETE_MARK_RE = re.compile(r"(?m)^[ \t]*<{5,}[ \t]*DELETE\b")
_DELETE_CLOSE_RE = re.compile(r">{5,}[ \t]*DELETE")

# Zero-anchor symbol splice blocks. The header names the file and
# the symbol; the body is the FULL new definition (replace) or the new text
# (insert after); no SEARCH section, no line numbers:
#
#     <<<<<<< SYMBOL src/parser.py :: TableParser.parse_row
#     def parse_row(self, line):
#         ...
#     >>>>>>> END_SYMBOL
#
# `AFTER_SYMBOL` / `INSERT_AFTER_SYMBOL` splice the body after the definition.
SYMBOL_BLOCK_RE = re.compile(
    r"^[ \t]*<{5,}[ \t]*(?P<kind>SYMBOL|AFTER_SYMBOL|INSERT_AFTER_SYMBOL)"
    r"[ \t]+(?P<file>[^\n:]+?)[ \t]*::[ \t]*(?P<symbol>[^\s<>/][^\s<>/]*?)"
    r"[ \t]*\n"
    r"(?P<replace>.*?)"
    r"^[ \t]*>{5,}[ \t]*(?:END_SYMBOL|INSERT_AFTER_SYMBOL|AFTER_SYMBOL"
    r"|SYMBOL)[ \t]*$",
    re.DOTALL | re.MULTILINE,
)
# A block that was started but never closed (for malformed_blocks reporting).
_SYMBOL_MARK_RE = re.compile(
    r"(?m)^[ \t]*<{5,}[ \t]*(?:AFTER_|INSERT_AFTER_)?SYMBOL[ \t]+(?P<file>[^\n]*)$")
# Strips the `<<<<<<< SYMBOL ` lead off a captured mark line, leaving the path.
_SYMBOL_MARK_LEAD_RE = re.compile(r"^\s*<{5,}\s*(?:AFTER_|INSERT_AFTER_)?SYMBOL\s+")


def _is_anchored(e: "Edit") -> bool:
    """True for a LINES/SYMBOL-anchored block: a line-range replacement, never
    a creation, a rename half, or a no-op."""
    return bool(e.lines or e.symbol or e.insert_after)

# Conventional folders for dead/archived code: a basename-only move into one
# of these counts as a rename without further proof.
_ARCHIVE_DIRS = frozenset({
    "graveyard", "graveyards", "archive", "archived", "archives",
    "trash", "deleted", "dead", "deadcode", "dead-code", "old",
    "old_code", "deprecated", "bak", "backup", "backups", "attic",
})


@dataclass
class Edit:
    """One parsed search/replace block, not yet applied."""

    file: str
    search: str
    replace: str
    # FILE-header anchors: `LINES a-b` (1-based inclusive) as parsed; a
    # `SYMBOL name` anchor keeps the name here until apply_edits resolves it
    # into `lines` (via the caller's index when given one, otherwise straight
    # from the file's AST with `locate_symbol_span`).
    lines: tuple[int, int] | None = None
    symbol: str = ""
    # Zero-anchor splice fields: `insert_after` names the definition the body
    # is spliced AFTER (from an AFTER_SYMBOL block or an
    # {"edit_action": "insert_after_symbol"} payload); a plain symbol splice
    # lands in `symbol`.
    insert_after: str = ""
    # A whole-file deletion (a `<<<<<<< DELETE` block or `DELETE_FILE: path`
    # line): the file is backed up, then unlinked. Never a rename half.
    delete: bool = False
    # An engine-computed exact whole-file replacement (codemods): SEARCH and
    # REPLACE are the complete preimage/postimage, so the already-applied
    # heuristic must not reinterpret a trailing-line deletion as "the REPLACE
    # content is already present". Model edits leave this False.
    exact: bool = False


@dataclass
class Result:
    """The outcome of attempting one file's edits."""

    ok: bool
    file: str
    message: str
    diff: str = ""
    # before/after feed the reviewer; `sha` pins the bytes the preview was
    # built from so a later apply can refuse if the file changed underneath.
    before: str = ""
    after: str = ""
    sha: str = ""
    # At least one block anchored approximately rather than exactly.
    fuzzy: bool = False
    # For a creation detected as a rename: the old paths whose real contents
    # were moved into this file (and removed by apply).
    renames_from: tuple[str, ...] = ()
    # The edit replaced the WHOLE existing file (empty-SEARCH overwrite or a
    # LINES range covering every line). The context-pressure gate reads this
    # to refuse oversized whole-file drafts on small-window profiles.
    whole_file: bool = False
    # The edit DELETED the file (a DELETE block / DELETE_FILE directive): it
    # was backed up, then unlinked. The index forgets it instead of reindexing.
    deleted: bool = False


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
# A bare ``` line toggles fenced-code state while scanning a reply.
_FENCE_RE = re.compile(r"^\s{0,3}```.*$")
_HEADER_START_RE = re.compile(r"(?<!\n)(---\s*FILE:)")


def _strip_wrapping_fences(text: str) -> str:
    """Blank out only the markdown fences that *wrap* the edit blocks: an
    opening ``` before the first `--- FILE:` header and a closing ``` after the
    last `>>>>>>> REPLACE`. Fences between headers belong to the replacement
    code (e.g. a docstring with a fenced example) and are left alone."""
    lines = text.split("\n")
    first = text.find("--- FILE:")
    if first == -1:
        return text
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
    close_idx = -1
    tail_marks = [i for i, ln in enumerate(lines)
                  if "REPLACE" in ln or _DELETE_CLOSE_RE.search(ln)]
    if tail_marks:
        tail_from = max(tail_marks)
        for j in range(tail_from + 1, len(lines)):
            if _FENCE_RE.match(lines[j]):
                close_idx = j
                break
    for k in (open_idx, close_idx):
        if k != -1:
            lines[k] = ""
    return "\n".join(lines)


_HEADER_LINE_RE = re.compile(r"(?m)^[ \t]*-{3}\s*FILE:[ \t]*(?P<file>[^\n]*)$")
_SEARCH_MARK_RE = re.compile(r"(?m)^[ \t]*<{5,}\s*SEARCH")
_WRONG_HEADER_RE = re.compile(
    r"(?im)^[ \t]*(?:\*{2})?(?:file|path)\s*[:=]\s*(?P<file>[^\n]+?)(?:\*{2})?\s*$"
    r"|^[ \t]*#{2,4}\s+(?P<file2>[\w./\\-]+\.\w+)\s*$")


def malformed_blocks(text: str, parsed: int) -> list[tuple[str, str]]:
    """[(file, hint)] for blocks the model STARTED but that did not parse:
    `<<<<<<< SEARCH` markers or `--- FILE:` headers with no complete block
    behind them (a missing `=======` or `>>>>>>> REPLACE`, a wrong header
    spelling), or `<<<<<<< SYMBOL file :: name` splice blocks without their
    `>>>>>>> END_SYMBOL` closer. Without this they are silently dropped and
    the model never learns its edit was ignored. `parsed` is the number of
    blocks that did parse, so a clean reply returns []."""
    text = text.replace("\r\n", "\n")
    headers = _HEADER_LINE_RE.findall(text)
    searches = len(_SEARCH_MARK_RE.findall(text))
    symbols = _SYMBOL_MARK_RE.findall(text)
    delete_marks = list(_DELETE_MARK_RE.finditer(text))
    delete_closers = [m.start() for m in _DELETE_CLOSE_RE.finditer(text)]
    # A symbol splice block normally sits under its own `--- FILE:` header, so
    # those headers are not search blocks that lost their markers. A DELETE
    # block owns its header the same way.
    plain_headers = max(len(headers) - len(symbols) - len(delete_marks), 0)
    started = max(plain_headers, searches) + len(symbols) + len(delete_marks)
    if started <= parsed:
        return []
    missing = started - parsed
    out: list[tuple[str, str]] = []
    if searches > plain_headers:
        hint = ("a `<<<<<<< SEARCH` block has no `--- FILE: path` header "
                "above it -- did you mean `--- FILE: relative/path.py`?")
        wrong = _WRONG_HEADER_RE.search(text)
        out.append(((wrong.group("file") or wrong.group("file2")).strip()
                    if wrong else "(unknown file)", hint))
        missing -= 1
    for dm in delete_marks:
        if missing <= 0:
            break
        # Only an UNCLOSED mark is a malformed block: pair each opener with
        # the next closer after it, in order.
        closer = next((p for p in delete_closers if p > dm.start()), None)
        if closer is not None:
            delete_closers.remove(closer)
            continue
        # The DELETE marker carries no path; the nearest header above it does.
        head = "(unknown file)"
        for hm in _HEADER_LINE_RE.finditer(text):
            if hm.start() < dm.start():
                head = hm.group("file").strip() or head
            else:
                break
        out.append((head,
                    "its DELETE block is incomplete: after `<<<<<<< DELETE` "
                    "it needs a `>>>>>>> DELETE` line on its own (or use the "
                    "one-line form `DELETE_FILE: path`)"))
        missing -= 1
    for line in symbols:
        if missing <= 0:
            break
        head = _SYMBOL_MARK_LEAD_RE.sub("", line.split("::")[0]).strip()
        out.append((head or "(unknown file)",
                    "its symbol splice block is incomplete: after the "
                    "`<<<<<<< SYMBOL path :: name` header it needs the full "
                    "new definition and a `>>>>>>> END_SYMBOL` line on its "
                    "own"))
        missing -= 1
    for f in headers:
        if missing <= 0:
            break
        out.append((f.strip() or "(unknown file)",
                    "its block is incomplete: it needs `<<<<<<< SEARCH`, "
                    "`=======`, and `>>>>>>> REPLACE` lines, each on its own "
                    "line, in that order"))
        missing -= 1
    return out


def parse_edit_actions(text: str) -> list[tuple[int, "Edit"]]:
    """(position, Edit) for structured `{"edit_action": ...}` JSON payloads
    embedded anywhere in a reply (with or around code fences). Supported
    actions: `replace_symbol` and `insert_after_symbol`; unknown actions are
    ignored so a future payload shape never corrupts an existing reply. A
    malformed or incomplete object is skipped, never guessed at."""
    if "edit_action" not in text:
        return []
    dec = json.JSONDecoder()
    out: list[tuple[int, Edit]] = []
    for m in re.finditer(r'"edit_action"\s*:\s*"(replace_symbol|insert_after_symbol)"',
                         text):
        obj = None
        start = -1
        # Walk every `{` before the key outward until one whole object
        # decodes: the innermost brace opens a string fragment, not the payload.
        probe = text.rfind("{", 0, m.start())
        tries = 0
        while probe >= 0 and tries < 64:
            try:
                cand, _end = dec.raw_decode(text[probe:])
            except ValueError:
                cand = None
            if isinstance(cand, dict) and cand.get("edit_action") in (
                    "replace_symbol", "insert_after_symbol"):
                obj, start = cand, probe
                break
            probe = text.rfind("{", 0, probe)
            tries += 1
        if obj is None:
            continue
        rel = str(obj.get("file") or "").strip()
        symbol = str(obj.get("symbol") or obj.get("name") or "").strip()
        repl = obj.get("replacement")
        if repl is None:
            repl = obj.get("content")
        if not isinstance(repl, str) or not rel or not symbol:
            continue
        repl = repl.replace("\r\n", "\n").replace("\r", "\n").removesuffix("\n")
        e = Edit(fileio.normalise_rel(rel), "", repl,
                 symbol=symbol if obj["edit_action"] == "replace_symbol" else "",
                 insert_after=symbol
                 if obj["edit_action"] == "insert_after_symbol" else "")
        out.append((start, e))
    return out


def parse_edits(text: str, discards: list | None = None) -> list[Edit]:
    """Extract every search/replace and symbol-splice block from a model's
    reply text, in reply order.

    Normalises line endings and strips the fences wrapping the blocks. A block
    whose SEARCH and REPLACE are identical changes nothing and is dropped; its
    file is appended to `discards` when given, so callers never count a null
    edit as work. Zero-anchor `<<<<<<< SYMBOL file :: name` blocks and
    `{"edit_action": ...}` JSON payloads parse into anchored Edits (no SEARCH,
    a symbol to resolve at apply time). `<<<<<<< DELETE` blocks and
    `DELETE_FILE: path` lines parse into `delete=True` Edits."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _strip_wrapping_fences(text)
    # A blanked fence line can leave a header glued to the line above it.
    cleaned = _HEADER_START_RE.sub(r"\n\1", cleaned)
    found: list[tuple[int, Edit]] = []
    # Spans owned by real edit blocks: a DELETE marker quoted inside a
    # SEARCH/REPLACE or symbol-splice body (docs showing the syntax) is text,
    # not an edit, and must not parse.
    owned: list[tuple[int, int]] = []
    for m in BLOCK_RE.finditer(cleaned):
        owned.append((m.start(), m.end()))
        # Drop exactly one trailing newline: the separator/closer may sit on
        # the same line as the last body line, so an empty replacement must
        # stay "" instead of becoming "\n".
        search = m.group("search").removesuffix("\n")
        replace = m.group("replace").removesuffix("\n")
        raw_file = m.group("file").strip()
        lines_anchor: tuple[int, int] | None = None
        symbol_anchor = ""
        am = _ANCHOR_RE.match(raw_file)
        if am:
            raw_file = am.group("path").strip()
            if am.group("start"):
                a, b = int(am.group("start")), int(am.group("end"))
                lines_anchor = (min(a, b), max(a, b))
            else:
                symbol_anchor = am.group("symbol").strip()
        fname = fileio.normalise_rel(raw_file)
        anchored = bool(lines_anchor or symbol_anchor)
        if search == replace and not anchored:
            print(f"  [discarded a no-op edit block for {fname}: SEARCH and "
                  f"REPLACE are identical; it changes nothing]",
                  file=sys.stderr)
            if discards is not None:
                discards.append(fname)
            continue
        found.append((m.start(), Edit(fname, search, replace,
                                      lines=lines_anchor,
                                      symbol=symbol_anchor)))
    def _in_owned(pos: int) -> bool:
        return any(a <= pos < b for a, b in owned)

    # Zero-anchor splice blocks: `<<<<<<< SYMBOL file :: name` (replace) and
    # `<<<<<<< AFTER_SYMBOL file :: name` (insert after), closed by
    # `>>>>>>> END_SYMBOL`. No SEARCH section exists to quote or mis-indent.
    for m in SYMBOL_BLOCK_RE.finditer(cleaned):
        if _in_owned(m.start()):
            continue
        owned.append((m.start(), m.end()))
        kind = m.group("kind").upper()
        fname = fileio.normalise_rel(m.group("file").strip())
        body = m.group("replace").removesuffix("\n")
        anchor = Edit(fname, "", body,
                      symbol="" if kind != "SYMBOL" else m.group("symbol").strip(),
                      insert_after=m.group("symbol").strip()
                      if kind != "SYMBOL" else "")
        found.append((m.start(), anchor))
    # Whole-file deletions: `--- FILE: path` + `<<<<<<< DELETE` / `>>>>>>>
    # DELETE`, or a bare `DELETE_FILE: path` line. The body between the
    # markers is ignored; a DELETE quoted inside another block's body is
    # documentation, not an edit (the `owned` spans above).
    for m in DELETE_BLOCK_RE.finditer(cleaned):
        if _in_owned(m.start()):
            continue
        owned.append((m.start(), m.end()))
        raw_file = m.group("file").strip()
        am = _ANCHOR_RE.match(raw_file)      # a stray anchor suffix: path only
        if am:
            raw_file = am.group("path").strip()
        fname = fileio.normalise_rel(raw_file)
        found.append((m.start(), Edit(fname, "", "", delete=True)))
    for m in DELETE_FILE_RE.finditer(cleaned):
        if _in_owned(m.start()):
            continue
        fname = fileio.normalise_rel(m.group("file").strip())
        found.append((m.start(), Edit(fname, "", "", delete=True)))
    for pos, act in parse_edit_actions(cleaned):
        if not _in_owned(pos):
            found.append((pos, act))
    found.sort(key=lambda t: t[0])
    return [e for _pos, e in found]


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


def locate(haystack: str, needle: str, cfg=None,
           replacement: str | None = None) -> Span | None:
    """Locate one search block in a file's text and return a Span, or None if
    it cannot be applied.

    `replacement` is the text to splice in (the block's REPLACE body); it
    defaults to the needle, a no-op, for callers that only want the span.

    Tries, in order: an exact unique match; a whitespace-tolerant unique match
    (trailing whitespace ignored); a deterministic dedent-normalized unique
    match (stripping common leading indent and re-indenting `replacement` to
    the matched window's base indentation); and, when `cfg.fuzzy_match`
    allows, a similarity match against every same-length window, accepted only
    if the best window clears `cfg.fuzzy_min_ratio` and clearly beats the
    runner-up. An ambiguous match is always a failure: editing the wrong
    occurrence silently is worse than asking the model for more context.
    """
    repl = needle if replacement is None else replacement
    hit = _locate_with_indent(haystack, needle)
    if hit is not None:
        start, stop, win_base, need_base = hit
        if win_base != need_base and repl.strip():
            win_w = len(win_base.expandtabs(4))
            need_w = len(need_base.expandtabs(4))
            delta = win_w - need_w
            rep_w = len(_common_indent(repl.split("\n")).expandtabs(4))
            if delta < 0 and rep_w < -delta:
                delta = win_w - rep_w
            repl = _shift_indent(repl, delta, win_base)
        return Span(start, stop, repl)
    if haystack.count(needle) > 1:
        return None
    if not (cfg is not None and getattr(cfg, "fuzzy_match", False)):
        return None
    min_ratio = float(getattr(cfg, "fuzzy_min_ratio", 0.85))
    max_len = int(getattr(cfg, "max_read_lines", 200)) * 200
    return _locate_fuzzy(haystack, needle, min_ratio, max_len, repl)


def _locate_fuzzy(haystack: str, needle: str, min_ratio: float,
                  max_needle: int, replacement: str) -> Span | None:
    """Best-effort anchor for a search block that does not match exactly.

    Compares the needle against every window of the haystack with the same
    number of lines (difflib.SequenceMatcher). Accepted only if the best
    window clears `min_ratio` and beats the second best by a clear margin.
    Refuses very large needles: the scan is O(lines x length)."""
    if not needle.strip() or "\x00" in haystack or len(needle) > max_needle:
        return None
    hlines = haystack.split("\n")
    nlines = needle.split("\n")
    if len(nlines) > len(hlines):
        return None
    k = len(nlines)
    sm = difflib.SequenceMatcher(None)
    sm.set_seq2(nlines)              # the fixed side is cached by difflib
    best_ratio, second_ratio, best_i = 0.0, 0.0, -1
    for i in range(len(hlines) - k + 1):
        window = hlines[i:i + k]
        if window == nlines:
            continue                     # an exact match would have been found
        sm.set_seq1(window)
        if sm.real_quick_ratio() <= second_ratio or sm.quick_ratio() <= second_ratio:
            continue                     # cannot change best or second best
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
    seg = hlines[best_i:best_i + k]
    stop = start + sum(len(x) for x in seg) + (len(seg) - 1)
    return Span(start, stop, replacement, fuzzy=True)


def _rstripped_lines(s: str) -> list[str]:
    return [ln.rstrip() for ln in s.splitlines()]


def _line_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Char span of 1-based inclusive lines [start, end], INCLUDING the last
    line's terminator, or None when the range is out of bounds."""
    lines = text.splitlines(keepends=True)
    if start < 1 or end < start or end > len(lines):
        return None
    off = sum(len(x) for x in lines[:start - 1])
    return off, off + sum(len(x) for x in lines[start - 1:end])


# A SEARCH body that is nothing but a line number: the model quoted the
# read's `  4| text` prefix instead of the text. Auto-reanchored to LINES n-n
# when the number does not occur literally in the file (see apply_edits).
_NUMERIC_SEARCH_RE = re.compile(r"[0-9]{1,6}")

# Closest-window hint on a failed block: a lower similarity floor than fuzzy
# application (this only REPORTS a location, never edits) and a hard cap on the
# quoted lines so the repair feedback stays small.
_CLOSEST_FLOOR = 0.5
_CLOSEST_MAX_LINES = 18


def _closest_window(haystack: str, needle: str,
                    max_needle: int) -> tuple[int, int, float] | None:
    """The window of `haystack` most similar to `needle`, as (first_line,
    last_line, ratio) with 1-based inclusive lines, or None when nothing
    reaches _CLOSEST_FLOOR. Same scan as _locate_fuzzy, without the
    ambiguity/quality gates: used only in failure messages."""
    if not needle.strip() or "\x00" in haystack or len(needle) > max_needle:
        return None
    hlines = haystack.split("\n")
    nlines = needle.split("\n")
    if len(nlines) > len(hlines):
        return None
    k = len(nlines)
    sm = difflib.SequenceMatcher(None)
    sm.set_seq2(nlines)
    best_ratio, best_i, best_char = 0.0, -1, -1.0
    for i in range(len(hlines) - k + 1):
        window = hlines[i:i + k]
        if window == nlines:
            continue                     # an exact match would have been found
        sm.set_seq1(window)
        # `<`, not `<=`: a window tied on the line-level bound can still win
        # the char-level tie-break below.
        if sm.real_quick_ratio() < best_ratio or sm.quick_ratio() < best_ratio:
            continue
        r = sm.ratio()
        if r > best_ratio:
            best_ratio, best_i, best_char = r, i, -1.0
        elif r == best_ratio and r >= _CLOSEST_FLOOR:
            # Line-level tie: prefer the window closer character-for-character
            # (a one-char change beats a shifted-by-one window).
            c = difflib.SequenceMatcher(None, "\n".join(window), needle).ratio()
            if best_char < 0.0:
                best_char = difflib.SequenceMatcher(
                    None, "\n".join(hlines[best_i:best_i + k]), needle).ratio()
            if c > best_char:
                best_i, best_char = i, c
    if best_i < 0 or best_ratio < _CLOSEST_FLOOR:
        return None
    return best_i + 1, best_i + k, best_ratio


def _closest_hint(text: str, needle: str, cfg=None) -> str:
    """Failure-message suffix showing where the SEARCH text most nearly
    matches, with real line numbers, so the model can re-anchor on facts
    instead of re-guessing."""
    max_needle = int(getattr(cfg, "max_read_lines", 200) or 200) * 200
    win = _closest_window(text, needle, max_needle)
    if win is None:
        return ""
    a, b, ratio = win
    seg = text.split("\n")[a - 1:b]
    shown = seg[:_CLOSEST_MAX_LINES]
    more = (f"\n  (... {len(seg) - len(shown)} more line(s) in the range)"
            if len(seg) > len(shown) else "")
    body = "\n".join(f"  {a + i} | {ln}" for i, ln in enumerate(shown))
    return (f" The closest text in the file right now is at lines {a}-{b} "
            f"({ratio:.0%} similar):\n{body}{more}\n"
            "Copy it exactly into SEARCH, or anchor the block with "
            f"`--- FILE: <path> LINES {a}-{b}` and put the full replacement "
            "text in REPLACE (SEARCH empty).")


def _common_indent(lines: list[str]) -> str:
    """Minimum leading whitespace (by 4-col tab width) across non-blank lines."""
    best: str | None = None
    best_w = -1
    for ln in lines:
        if not ln.strip():
            continue
        ind = ln[: len(ln) - len(ln.lstrip())]
        w = len(ind.expandtabs(4))
        if best is None or w < best_w:
            best, best_w = ind, w
    return best or ""


def _dedent_lines(lines: list[str]) -> list[str]:
    """Strip common leading indent (in 4-col tab columns) and trailing
    whitespace from every line, leaving blank lines empty."""
    widths = [len(ln[: len(ln) - len(ln.lstrip())].expandtabs(4))
              for ln in lines if ln.strip()]
    if not widths:
        return ["" for _ in lines]
    min_w = min(widths)
    out: list[str] = []
    for ln in lines:
        if not ln.strip():
            out.append("")
        else:
            out.append(ln.rstrip().expandtabs(4)[min_w:])
    return out


def _locate_with_indent(haystack: str, needle: str
                        ) -> tuple[int, int, str, str] | None:
    """Find the unique span of `needle` in `haystack` as
    `(start, end, win_base_indent, needle_base_indent)`:
    1. exact unique match;
    2. trailing-whitespace-tolerant unique match;
    3. dedent-normalized unique match (stripping common leading indent).
    Ambiguous matches fail at every tier."""
    if not needle.strip():
        return None
    n = haystack.count(needle)
    if n == 1:
        i = haystack.index(needle)
        return i, i + len(needle), "", ""
    if n > 1:
        return None

    hay, need = _rstripped_lines(haystack), _rstripped_lines(needle)
    if not need:
        return None

    def _window_char_span(s: int) -> tuple[int, int]:
        lines = haystack.splitlines(keepends=True)
        seg = lines[s : s + len(need)]
        start = sum(len(x) for x in lines[:s])
        # Exclude the final line terminator: exact-match needles carry none.
        tail = len(seg[-1]) - len(seg[-1].rstrip("\r\n"))
        end = start + sum(len(x) for x in seg) - tail
        return start, end

    starts = [i for i in range(len(hay) - len(need) + 1)
              if hay[i : i + len(need)] == need]
    if len(starts) == 1:
        start, end = _window_char_span(starts[0])
        return start, end, "", ""
    if len(starts) > 1:
        return None

    # Tier 3: deterministic dedent-normalized unique match.
    need_ded = _dedent_lines(need)
    if not any(need_ded):
        return None
    first_stripped = need[0].lstrip()
    ded_starts: list[int] = []
    k = len(need)
    for i in range(len(hay) - k + 1):
        if hay[i].lstrip() != first_stripped:
            continue
        if _dedent_lines(hay[i : i + k]) == need_ded:
            ded_starts.append(i)
            if len(ded_starts) > 1:
                return None
    if len(ded_starts) != 1:
        return None
    s = ded_starts[0]
    start, end = _window_char_span(s)
    return start, end, _common_indent(hay[s : s + k]), _common_indent(need)


def _locate(haystack: str, needle: str) -> tuple[int, int] | None:
    """Find the unique span of `needle` in `haystack`: exact match first,
    then trailing-whitespace tolerant, then dedent-normalized. Ambiguous
    matches fail."""
    hit = _locate_with_indent(haystack, needle)
    return None if hit is None else (hit[0], hit[1])


def _indent_of_line(text: str, lineno: int) -> str:
    """Leading whitespace of a 1-based line ("" when out of range)."""
    lines = text.split("\n")
    if not 1 <= lineno <= len(lines):
        return ""
    ln = lines[lineno - 1]
    return ln[: len(ln) - len(ln.lstrip())]


def _shift_indent(body: str, delta: int, base_indent: str) -> str:
    """Shift every non-blank line in `body` by `delta` 4-col tab columns,
    preserving relative indentation and handling tab-dedent overshoots."""
    if delta == 0:
        return body
    out: list[str] = []
    for ln in body.split("\n"):
        if not ln.strip():
            out.append(ln)
            continue
        if delta > 0:
            pad = ("\t" * (delta // 4) if "\t" in base_indent and delta % 4 == 0
                   else " " * delta)
            out.append(pad + ln)
        else:
            drop = -delta
            nlead = len(ln) - len(ln.lstrip())
            ind = ln[:nlead]
            if " \t" in ind:
                # A space before a tab makes the tab's column width depend on
                # its starting column; expand to exact columns before dedenting.
                ind = ind.expandtabs(4)
                out.append(ind[drop:] + ln[nlead:])
                continue
            j = 0
            while j < len(ind) and drop > 0:
                drop -= 4 if ind[j] == "\t" else 1
                j += 1
            if drop < 0:                     # overshot a tab stop: restore excess spaces
                j -= 1
                ind = ind[j + 1:] + (" " * -drop)
                out.append(ind + ln[nlead:])
                continue
            out.append(ind[j:] + ln[nlead:])
    return "\n".join(out)


def normalise_symbol_indent(body: str, base_indent: str) -> str:
    """Shift a symbol replacement so its top-level lines sit at `base_indent`
    (the target definition's real indent in the file).

    A small model asked to rewrite `TableParser.parse_row` habitually emits

        def parse_row(self, line):
            return line

    at column 0; the splice needs it at the method's 4 spaces. The body's own
    first non-blank line defines its base level; every line is shifted by the
    same amount, so relative structure inside the definition is preserved.
    Already-correct bodies come back untouched (no churn, byte-stable diffs).
    """
    lines = body.split("\n")
    cur = None
    for ln in lines:
        if ln.strip():
            cur = ln[: len(ln) - len(ln.lstrip())]
            break
    if cur is None or cur == base_indent:
        return body
    cur_w = len(cur.expandtabs(4))
    base_w = len(base_indent.expandtabs(4))
    return _shift_indent(body, base_w - cur_w, base_indent)


def _leading_decorators(original: str, a: int, b: int) -> list[str]:
    """The decorator lines at the head of the anchored span [a, b], exactly as
    they sit in the file (already at the target indent), or [] when the
    definition is bare. Small models rewrite a method and forget to re-echo
    `@staticmethod`; dropping a decorator silently changes behaviour, so the
    splice carries them over unless the new body starts with its own `@`."""
    lines = original.split("\n")
    out: list[str] = []
    for i in range(a - 1, min(b, len(lines))):
        s = lines[i].strip()
        if s.startswith("@"):
            out.append(lines[i].rstrip())
        elif not s:
            continue
        else:
            break
    return out


# --------------------------------------------------------------------------
# Rename placeholders
# --------------------------------------------------------------------------
_BLOCK_COMMENT_RE = re.compile(r"^/\*.*?\*/", re.DOTALL)
_HTML_COMMENT_RE = re.compile(r"^<!--.*?-->", re.DOTALL)


def _is_placeholder_only(text: str) -> bool:
    """True when `text` is nothing but a short comment -- the shape of a rename
    placeholder ("/* moved from sentences.js */"). Used to refuse half-renames
    that would otherwise write a few-byte stub in place of real content."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines or len(lines) > 3:
        return False
    joined = "\n".join(lines)
    if len(joined.encode("utf-8")) > 200:
        return False
    rest = joined
    m = _BLOCK_COMMENT_RE.match(rest)
    while m:
        rest = rest[m.end():].strip()
        m = _BLOCK_COMMENT_RE.match(rest)
    m = _HTML_COMMENT_RE.match(rest)
    while m:
        rest = rest[m.end():].strip()
        m = _HTML_COMMENT_RE.match(rest)
    if rest == "":
        return True
    return all(ln.startswith(("#", "//", "<!--")) for ln in lines)


def _placeholder_comment(ph: str, suffix: str) -> str:
    """Render a rename placeholder as a leading comment for a file of the given
    suffix, without double-wrapping text the model already wrote as a comment."""
    lines = ph.splitlines()
    if suffix in (".html", ".htm"):
        return "\n".join(lines) + "\n"
    if suffix == ".py":
        return "".join(
            (ln if ln.lstrip().startswith("#") else "# " + ln) + "\n"
            for ln in lines)
    if suffix in (".css", ".js", ".ts", ".scss"):
        return "".join(
            (ln if (ln.lstrip().startswith("//") or ln.lstrip().startswith("/*"))
             else "/* " + ln + " */") + "\n"
            for ln in lines)
    return ph + "\n"


_ORPHAN_CREATE_MSG = (
    "refusing to create this file: its content is only a placeholder "
    "comment and no rename source matched. If you are renaming a file, "
    "emit ONLY the empty-SEARCH creation block for the new path plus a "
    "block for the old path whose SEARCH is part of its content and whose "
    "REPLACE is empty -- Pedacito moves the real bytes; never copy them. "
    "If you mean to write this file's content yourself, put that content "
    "in REPLACE.")


def _validate(path: Path, text: str) -> str | None:
    """Return an error message if `text` does not parse as source of `path`'s
    language, else None."""
    if path.suffix == ".py":
        try:
            ast.parse(text)
        except SyntaxError as e:
            return f"a syntax error at line {e.lineno}: {e.msg}"
        except ValueError as e:               # null bytes (Python < 3.12)
            return f"a parse error: {e}"
    elif path.suffix.lower() in LANG_BY_EXT and TS_AVAILABLE:
        bad, lineno = has_syntax_error(text, LANG_BY_EXT[path.suffix.lower()])
        if bad:
            return f"a syntax error near line {lineno}"
    return None


def syntax_error(path, text: str) -> str | None:
    """Public syntax check: an actionable error message when `text` does not
    parse as source of `path`'s language, else None. The review's hunk-level
    sign-off uses it to refuse a partial acceptance that would leave a file
    syntactically broken (e.g. half of a bracket pair)."""
    return _validate(Path(path), text)


def _creation_text(group: list[Edit]) -> str:
    """Body of a new file: the REPLACE bodies of the empty-SEARCH blocks, in
    reply order (anchored blocks are line replacements, never creation text).
    parse_edits drops one trailing newline from every body, so joining with
    "\\n" never invents a blank line."""
    text = "\n".join(e.replace for e in group
                     if not e.search.strip() and not _is_anchored(e))
    if text and not text.endswith("\n"):
        text += "\n"
    return text


_DEF_NAME_RE = re.compile(
    r"^[ \t]*(?:export\s+)?(?:async\s+)?"
    r"(?:def|function|class|func|type)\s+"
    r"([A-Za-z_$][\w$]*)", re.M)


def _defined_names(text: str) -> set[str]:
    """Top-level declaration names in a source file, language-agnostic
    enough for the fragment guard (function/def/class/func/type/const...)."""
    return set(_DEF_NAME_RE.findall(text))


# The public surface of a JS/TS module in its three idioms: ESM `export`,
# CommonJS `module.exports` / `exports.x`, and classic-script `window.x` /
# `globalThis.x` globals. A whole-file rewrite that drops ALL of them breaks
# every importer and every page script that reads the global.
_EXPORT_FORM_RE = re.compile(
    r"\bexport\s+(?:default|const|let|var|function|class|async|type|"
    r"interface|enum|\{|\*)"
    r"|\bmodule\.exports\s*[.=]"
    r"|\bexports\.[A-Za-z_$][\w$]*\s*=(?!=)"
    r"|\b(?:window|globalThis)\.[A-Za-z_$][\w$]*\s*=(?!=)"
    r"|\b(?:window|globalThis)\[\s*['\"][A-Za-z_$][\w$]*['\"]\s*\]\s*=(?!=)")
_JS_FAMILY_EXT = (".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".mts",
                  ".cts")


_IMPLAUSIBLE_PATH_CHARS = set('`${};=<>|"*?\x00\n\r\t\'()[]:,#@')


def _plausible_new_path(rel: str) -> bool:
    """Whether `rel` looks like a file path rather than a copied code line.

    A model can emit the line it meant to edit as the `--- FILE:` header. The
    path is normalised and confined to the project, so that used to create a
    real file named after the code line (for example
    `tile.className = `tile tile-${value}`;`) while the target file stayed
    untouched. New paths must not carry code/prose punctuation or padded
    segments. The punctuation set covers the common code-line shapes --
    single-quoted calls (`console.log('hi')`), signatures (`def greet(name):`),
    indexing (`arr[0]`), comments (`# ...`) and decorators (`@app.route`) --
    while leaving ordinary names alone (`Q&A.md`, `a+b.txt`, `my file.txt`,
    unicode names). A multi-word basename with no extension (`import os`,
    `return x`) is a code line too: real extensionless files (Makefile,
    LICENSE) are single tokens.
    """
    if not rel or len(rel) > 240:
        return False
    if any(ch in _IMPLAUSIBLE_PATH_CHARS for ch in rel):
        return False
    parts = Path(rel).parts
    if not parts or not all(part and part == part.strip()
                            and part not in (".", "..") for part in parts):
        return False
    name = parts[-1]
    if " " in name and "." not in name:
        return False
    return True


def _wipe_refusal(rel: str) -> str:
    """The shared refusal for a whole-file replacement to nothing."""
    return (f"ERROR: refusing to overwrite {rel}: the replacement is empty, "
            "which would erase the whole file -- every definition, export "
            "and comment in it. An empty REPLACE is not how a file is "
            f"emptied: to remove the file emit `DELETE_FILE: {rel}` (or a "
            f"`--- FILE: {rel}` header with `<<<<<<< DELETE` / "
            "`>>>>>>> DELETE` lines), or keep the file and make targeted "
            "search/replace edits.")


def fragment_refusal(rel: str, original: str, new_text: str,
                     task: str = "") -> str | None:
    """Why replacing `original` with `new_text` looks like a fragment of a
    partial read rather than a whole-file rewrite, or None when it is safe.
    Shared by the `write` tool and the edit path's empty-SEARCH whole-file
    replacement, so a model cannot dodge the guard by picking the other
    shape. Two shrinking forms are refused when the file is big enough to
    matter:

      (1) a tiny fragment: the file has >= 20 non-blank lines and the new
          content has < 10 (deepclitest: a 42-line module replaced by the
          2-line helper the task asked to append);
      (2) a window re-quote: the content shrinks the file by more than half
          and every new line is a verbatim, in-order copy of the original
          (the game-session PED-4 write), or it reuses original lines but
          drops the file's first or last one (a lost head/tail fragment).

    A genuine rewrite (no copied lines) still goes through, as does a full
    re-quote that only deletes a line or two. Below the floor a shrink is
    ordinary small-file editing and stays allowed. Two wipe forms are
    refused at any size: a replacement with no content at all, and a JS/TS
    rewrite that drops the file's entire export surface."""
    old_ls = [l.strip() for l in original.splitlines() if l.strip()]
    new_ls = [l.strip() for l in new_text.splitlines() if l.strip()]
    # Every shrinking rule below is gated on `new_ls`, so an empty (or
    # whitespace-only) replacement used to pass them all and wipe the file.
    # A whole-file replacement to nothing is never an edit.
    if old_ls and not new_ls:
        return _wipe_refusal(rel)
    # A JS/TS module whose replacement keeps content but drops EVERY export
    # form breaks each importer (`does not provide an export named ...`)
    # and each classic script reading the global. Targeted edits can still
    # remove exports one at a time when the task really asks for it. This
    # guard runs BEFORE the creative bypass so a .jsx/.tsx design rewrite
    # keeps its export-surface protection.
    if (new_ls and rel.lower().endswith(_JS_FAMILY_EXT)
            and _EXPORT_FORM_RE.search(original)
            and not _EXPORT_FORM_RE.search(new_text)):
        return (f"ERROR: refusing to overwrite {rel}: the file exports "
                "symbols (export / module.exports / window.*) but the "
                "replacement has no exports at all -- every importer and "
                "page script using them would break. Keep the export "
                "surface in the rewrite, or remove individual exports "
                "with targeted search/replace edits.")
    # Creative/design bypass: a whole-file rewrite of a hand-authored page /
    # stylesheet / doc, a creative task, or a restructure/replace brief is the
    # requested outcome, not the signature of a partial read. The wipe refusal
    # and the export guard above still apply; only the fragment heuristics
    # below stand down.
    from .prompts import whole_rewrite_allowed
    if whole_rewrite_allowed(task, rel):
        return None
    # A chunked write that stops early keeps most of the file's text but
    # drops the definitions that lived in the tail (test_1a: a 182-line
    # module written back as its first 120 lines passed the >half-shrink
    # rule and deleted checkLanding, calculateScore, isGameOver and
    # restart). If the new content is largely an in-order copy of the old
    # and loses a top-level name, it is a fragment even when it is not a
    # tiny one and the file did not shrink by half.
    if new_ls and len(old_ls) >= 8:
        lost = _defined_names(original) - _defined_names(new_text)
        if lost:
            probe = iter(old_ls)
            copied = sum(1 for l in new_ls if l in probe)
            if copied >= max(5, int(0.8 * len(new_ls))):
                names = ", ".join(sorted(lost)[:4])
                more = "" if len(lost) <= 4 else f" (+{len(lost) - 4} more)"
                return (f"ERROR: refusing to overwrite {rel}: the new "
                        f"content is an in-order copy of the file but drops "
                        f"top-level definition(s) {names}{more} -- the "
                        "signature of writing back only the first chunk you "
                        "read (lost fragment). Keep the file and use "
                        "targeted search/replace edits, or `append` the "
                        "remaining lines.")
    # A handful of lines is never a full-file rewrite of a real module:
    # it is the fragment the model composed for its addition, not the
    # file. (deepclitest: a 42-line module was replaced by the 2-line
    # helper the task asked to append.)
    if new_ls and len(old_ls) >= 20 and len(new_ls) < 10:
        return (f"ERROR: refusing to overwrite {rel}: your content has only "
                f"{len(new_ls)} non-blank lines but the file has "
                f"{len(old_ls)} -- that is a fragment, not a whole file. "
                "Keep the existing file and add your lines with a "
                "search/replace edit block (or `append` for the end of "
                "the file).")
    # The floor for a head/tail loss is lower than the window re-quote
    # floor: test_1a's 7-line README was written back as its 2-line
    # Changelog tail, which is data loss even though the file is small.
    if new_ls and len(old_ls) >= 5 and 2 * len(new_ls) < len(old_ls):
        shared_small = set(old_ls) & set(new_ls)
        keeps_ends_small = old_ls[0] in new_ls and old_ls[-1] in new_ls
        if shared_small and not keeps_ends_small:
            return (f"ERROR: refusing to overwrite {rel}: your content has "
                    f"{len(new_ls)} non-blank lines but the file has "
                    f"{len(old_ls)}, reuses text from the file, and drops "
                    "its beginning or end -- the signature of a lost "
                    "fragment, which would erase the lines you never "
                    "included. Make a targeted search/replace edit "
                    "instead. If you truly mean to shrink the whole file, "
                    "read it completely and rewrite it without copying "
                    "pieces of the old text.")
    if new_ls and len(old_ls) >= 8 and 2 * len(new_ls) < len(old_ls):
        probe = iter(old_ls)
        if all(l in probe for l in new_ls):
            return (f"ERROR: refusing to overwrite {rel}: your content has "
                    f"{len(new_ls)} non-blank lines but the file has "
                    f"{len(old_ls)}, and every one of your lines is a "
                    "verbatim copy from the file in order -- the signature "
                    "of rewriting a file from a PARTIAL read, which would "
                    "erase the lines you never saw. Make a targeted "
                    "search/replace edit instead. If you truly mean to "
                    "shrink this file, `read` it completely first so your "
                    "new content can differ from what you keep.")
        shared = set(old_ls) & set(new_ls)
        keeps_ends = old_ls[0] in new_ls and old_ls[-1] in new_ls
        if shared and not keeps_ends:
            return (f"ERROR: refusing to overwrite {rel}: your content has "
                    f"{len(new_ls)} non-blank lines but the file has "
                    f"{len(old_ls)}, reuses text from the file, and drops "
                    "its beginning or end -- the signature of a lost "
                    "fragment, which would erase the lines you never "
                    "included. Make a targeted search/replace edit "
                    "instead. If you truly mean to shrink the whole file, "
                    "read it completely and rewrite it without copying "
                    "pieces of the old text.")
    return None


def _content_lines(text: str) -> list[str]:
    """Lines that carry content (blanks dropped, trailing whitespace
    stripped), for already-applied comparisons."""
    return [l.rstrip() for l in text.splitlines() if l.strip()]


def _replace_already_present(text: str, repl: str, start: int = -1,
                             allow_single: bool = False) -> bool:
    """True when `repl`'s content lines already occur in `text` as one
    contiguous run -- at offset `start` when given (the span the block
    matched), anywhere otherwise. Replaces shorter than two content lines
    never count by default: a single common line is too ambiguous to skip a
    real edit over. `allow_single` relaxes that only after an earlier block
    in the same file was already found on disk (gather-write drift), and
    then only when the line occurs exactly once. Catches two field failures:
    a block re-emitted after a gather write already landed it (the SEARCH no
    longer matches but the REPLACE content is on disk), and repair rounds
    that appended the same test pair again and again (the SEARCH still
    matches and the REPLACE sits verbatim at the span, so applying would
    duplicate it)."""
    want = _content_lines(repl)
    if len(want) < 2 and not allow_single:
        return False
    if not want:
        return False
    if start >= 0:
        hay = _content_lines(text[start:])
        return hay[:len(want)] == want
    hay = _content_lines(text)
    for i in range(len(hay) - len(want) + 1):
        if hay[i:i + len(want)] == want:
            if len(want) == 1 and hay.count(want[0]) != 1:
                return False
            return True
    return False


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------
def apply_edits(root: Path, edits: list[Edit], backup: bool = True,
                dry_run: bool = False, backup_dir: Path | None = None,
                cfg=None, allowlist: set[str] | None = None) -> list[Result]:
    """Apply a list of Edits, grouped by file.

    Each path's blocks are applied IN ORDER against the evolving text, the
    result is validated (Python via ast, tree-sitter languages via
    chunker_ts), and then written (with a byte-exact backup of the original)
    or, in dry-run mode, only diffed. One cumulative Result is returned per
    file. An empty SEARCH against a missing file creates it. Paths outside
    `root` are refused. A `delete=True` Edit backs its file up, then unlinks
    it (Result.deleted); secret/credential paths and directories are refused.

    `cfg` enables approximate anchoring (`fuzzy_match`) and rename pairing
    (`allow_rename`); without it matching is exact-only.

    `allowlist`, when given, is a set of normalised (lowercased, no leading
    ./) paths the task may modify. Creating a new file is always allowed;
    editing, moving or deleting an EXISTING file outside the set is refused
    before any pairing, so a blocked rename source cannot be moved away.

    Backups go to `backup_dir` (a snapshot under .pedacito/backups/); an
    existing copy there is never overwritten, so the snapshot holds each
    file's state before the run's FIRST change. Without `backup_dir`, a
    `<file>.bak` is written beside the source.
    """
    results: list[Result] = []
    # Blocks stay unmerged: rename detection and sequential application both
    # need the raw blocks, and gluing several SEARCH bodies together yields a
    # needle that appears nowhere in the file.
    groups: dict[str, list[Edit]] = {}
    for e in edits:
        groups.setdefault(e.file, []).append(e)

    root_resolved = Path(root).resolve()
    task_text = str(getattr(cfg, "_task_text", "") or "")
    policy = getattr(cfg, "policy", None)

    # ---- unified policy engine (WP-4005) -----------------------------------
    try:
        from . import policy as policy_mod
        for rel in list(groups):
            d = policy_mod.enforce(cfg, "write", rel, root=root_resolved)
            if d is not None and d.decision == "deny":
                results.append(Result(
                    False, rel,
                    f"denied by policy ({d.matched_rule}): {d.reason}"))
                del groups[rel]
    except Exception as exc:
        # FAIL CLOSED: a policy evaluation error must never read as "allowed";
        # every remaining edit is refused rather than applied un-gated.
        for rel in list(groups):
            results.append(Result(
                False, rel,
                f"denied by policy: policy evaluation error: {exc}"))
        groups.clear()

    # ---- protected paths (.git/, .pedacito/, the plan file, config) --------
    for rel in list(groups):
        why = safety.is_protected(rel, cfg)
        if why:
            results.append(Result(False, rel, f"refused: {why}"))
            del groups[rel]

    def _resolve_inside(rel: str) -> Path | None:
        """Resolved path of `rel` when it stays inside the root (following
        symlinks), else None."""
        try:
            p = (root_resolved / rel).resolve()
            p.relative_to(root_resolved)
        except (OSError, RuntimeError, ValueError):
            return None
        return p

    # ---- opt-in write allowlist --------------------------------------------
    # With `allowlist`, an EXISTING file may only be edited, moved or deleted
    # when its normalised path is in the set (the task title named it or the
    # lookup phase read it); creating new files is always allowed. Blocked
    # paths are refused here, before rename pairing, so a blocked source can
    # never be moved away by its pair.
    if allowlist is not None:
        allowed = {fileio.path_key(p) for p in allowlist}
        for rel in list(groups):
            key = fileio.path_key(rel)
            if key in allowed:
                continue
            path = _resolve_inside(rel)
            if path is not None and path.is_file():
                results.append(Result(
                    False, rel,
                    "blocked by the write allowlist (auto_allowlist): this "
                    "task never read this file and its title does not name "
                    "it, so it may not be modified; creating new files is "
                    "still allowed"))
                del groups[rel]

    # ---- whole-file deletions ----------------------------------------------
    # DELETE blocks split out AFTER the protected/allowlist gates (a refused
    # path stays refused) but BEFORE rename pairing: a deletion is never a
    # rename half and never creation text. They apply inside the batch's try
    # below, so a mid-batch OSError rolls them back like every other mutation.
    delete_groups: dict[str, list[Edit]] = {}
    for rel in list(groups):
        deletes = [e for e in groups[rel] if e.delete]
        if not deletes:
            continue
        delete_groups[rel] = deletes
        rest = [e for e in groups[rel] if not e.delete]
        if rest:
            groups[rel] = rest
        else:
            del groups[rel]

    # ---- rename detection ---------------------------------------------------
    # Rename protocol: an empty-SEARCH creation block for the NEW path (a
    # placeholder) plus a content-quoting SEARCH / empty-REPLACE "clear" block
    # for each OLD path. Pedacito moves the old file's real bytes to the new
    # path and removes it.
    #
    # Blocks are classified per raw block, because small models re-emit the
    # same header per block (a rename target can share its name with a source),
    # and creators are identified by shape alone (empty SEARCH, non-empty
    # REPLACE) since the target often exists already ("overwrite + archive").
    # A path that is both creator and remover is a pure overwrite: its clear
    # blocks are dropped. Pairing only links distinct paths, so nothing is lost.
    def _classify(rel: str, g: list[Edit]) -> tuple[bool, bool]:
        """Return (can_create, can_clear) for one path's raw blocks. Anchored
        blocks never count: they replace a line range, not a whole file."""
        path = _resolve_inside(rel)
        if path is None:
            return False, False
        g = [e for e in g if not _is_anchored(e)]
        can_create = any(not e.search.strip() and e.replace.strip() for e in g)
        can_clear = False
        if path.is_file():
            original, _style = fileio.read(path, errors="replace")

            def _reaches_eof(search: str, span) -> bool:
                # locate() excludes a trailing newline from its span, so
                # tolerate one optional terminator after a whole-file clear.
                tail = original[span.stop:]
                if not tail.strip():
                    return True
                return tail in ("\n", "\r\n") and bool(search.strip())

            for e in g:
                if e.replace.strip():
                    continue
                if not e.search.strip():
                    # A path that also creates itself is never a remover: it
                    # would pair with itself and delete the real file.
                    if not can_create:
                        can_clear = True     # explicit wipe of an existing file
                    continue
                span = locate(original, e.search, cfg)
                if span is None:
                    # A truncated whole-file SEARCH (quoted from memory) still
                    # identifies the file if it matches contiguously at the
                    # START and again at the very END. The SEARCH text is
                    # never used as content.
                    first_nl = e.search.find("\n")
                    head = (e.search if first_nl < 0 else e.search[:first_nl])
                    if original.startswith(head):
                        tail = e.search[first_nl + 1:] if first_nl >= 0 else ""
                        if tail.strip():
                            tspan = locate(original, tail, cfg)
                            if tspan is not None and _reaches_eof(tail, tspan):
                                can_clear = True
                                continue
                elif not span.fuzzy:
                    # A unique exact match anywhere names the file as the move
                    # source; the SEARCH is never used as content. Fuzzy
                    # matches are too weak an identifier.
                    can_clear = True
        return can_create, can_clear

    classified = {rel: _classify(rel, g) for rel, g in groups.items()}
    creators = {rel for rel, (c, _cl) in classified.items() if c}
    removers = {rel for rel, (_c, cl) in classified.items() if cl}

    renames: dict[str, str] = {}             # cleared rel -> created rel
    if cfg is None or getattr(cfg, "allow_rename", True):
        used: set[str] = set()

        def _is_archive_dir(path: str) -> bool:
            return any(part.lower() in _ARCHIVE_DIRS
                       for part in Path(path).parts[:-1])

        def _placeholder_names_source(new: str, old: str) -> bool:
            """True when a placeholder creation for `new` mentions `old` (the
            "moved from X" comment of the rename idiom)."""
            names = {Path(old).name.lower(), Path(old).as_posix().lower(),
                     Path(old).stem.lower()}
            for e in groups.get(new, []):
                if e.search.strip() or not e.replace.strip():
                    continue
                body = e.replace.lower()
                if any(n and n in body for n in names):
                    return True
            return False

        def _pair_score(old: str, new: str) -> int:
            """Rank candidate pairings; -1 rejects a pair.

            Same-directory renames are unambiguous. Cross-directory ones need
            the old relative path to survive as a suffix, or, for a basename-
            only move, an explicit signal (archive folder, or a placeholder
            naming the source). A bare basename match into an arbitrary
            sibling directory is rejected: the overwrite-and-archive trap."""
            same_dir = Path(old).parent == Path(new).parent
            old_posix = fileio.path_key(Path(old).as_posix())
            new_posix = fileio.path_key(Path(new).as_posix())
            old_name = Path(old).name.lower()
            new_name = Path(new).name.lower()
            if same_dir:
                return 3 if old_name in new_name else 2
            if "/" in old_posix and new_posix.endswith("/" + old_posix):
                return 2
            # Moving a file back out of a subdirectory to the same name.
            if "/" in old_posix and old_posix.endswith("/" + new_posix):
                return 1
            if new_name == old_name and (
                    _is_archive_dir(new) or _placeholder_names_source(new, old)):
                return 1
            return -1

        def _pair(old: str, cands: list[str]):
            scored = [(s, n) for n in cands if (s := _pair_score(old, n)) >= 0]
            if not scored:
                return None
            scored.sort(key=lambda t: (-t[0], t[1]))
            return scored[0][1]

        rem_list, cre_list = sorted(removers), sorted(creators)

        def _source_is_real(old: str) -> bool:
            """True when the would-be source holds content worth moving. A file
            that is itself a rename placeholder carries no real bytes; pairing
            it as a source would wipe a live file."""
            path = root_resolved / old
            if not path.is_file():
                return False
            try:
                text, _ = fileio.read(path, errors="replace")
            except OSError:
                return False
            return not _is_placeholder_only(text)

        def _placeholder_target(new: str) -> bool:
            """True when `new`'s create blocks are only a short placeholder.
            Such a target may pair only with a source that has a
            content-quoting clear block; pairing it with a bare wipe would
            move zero bytes and leave a half-rename."""
            bodies = [e.replace.strip() for e in groups.get(new, [])
                      if not e.search.strip() and e.replace.strip()]
            return bool(bodies) and all(_is_placeholder_only(b) for b in bodies)

        def _explicit_wipe_only(old: str) -> bool:
            """True when every clear block for `old` has an EMPTY search."""
            clears = [e for e in groups.get(old, []) if not e.replace.strip()]
            return bool(clears) and all(not e.search.strip() for e in clears)

        # Pass 1: same-directory pairs first, ranked by name affinity. With a
        # single candidate the pairing is unambiguous; with several, the
        # alphabetical first is not necessarily the intended target, so score
        # them exactly as pass 2 does instead of taking `min(cand)`.
        for old in rem_list:
            if not _source_is_real(old):
                continue
            suffix = Path(old).suffix.lower()
            cand = [n for n in cre_list if n != old and n not in used
                    and Path(n).suffix.lower() == suffix
                    and Path(n).parent == Path(old).parent
                    and not (_placeholder_target(n) and _explicit_wipe_only(old))]
            if cand:
                new = _pair(old, cand)
                if new:
                    renames[old] = new
                    used.add(new)
        # Pass 2: remaining sources pair with remaining targets by score.
        for old in rem_list:
            if old in renames or not _source_is_real(old):
                continue
            suffix = Path(old).suffix.lower()
            cand = [n for n in cre_list if n != old and n not in used
                    and Path(n).suffix.lower() == suffix
                    and not (_placeholder_target(n) and _explicit_wipe_only(old))]
            if not cand:
                continue
            new = _pair(old, cand)
            if new:
                renames[old] = new
                used.add(new)

    # Blocks to apply, per path, in reply order. A rename source loses its
    # clear blocks (they would fail to match after the move); a path that is
    # also a creator keeps its create blocks, a pure source drops out.
    by_file: dict[str, list[Edit]] = {}
    for rel, g in groups.items():
        if rel in renames:
            keep = [e for e in g
                    if not (not e.replace.strip()
                            and (e.search.strip() or rel in creators))]
            if not keep:
                continue
            by_file[rel] = keep
            continue
        by_file[rel] = list(g)

    # ---- orphan placement guard -------------------------------------------
    # A placeholder-only creation with no rename source is a half-rename (the
    # clear block never matched). Refuse the new path AND any content-quoting
    # clears in the same reply: applying a leftover clear would strip lines
    # off the live file. The repair loop then re-asks with the rename protocol.
    rename_targets = set(renames.values())
    orphan_placeholders = {
        rel for rel, g in by_file.items()
        if rel not in rename_targets
        and (plain := [e for e in g if not _is_anchored(e)])
        and all(not e.search.strip() for e in plain)
        and any(e.replace.strip() for e in plain)
        and _is_placeholder_only(_creation_text(plain))
        and not any(o for o, n in renames.items()
                    if n == rel and (root_resolved / o).is_file())
    }

    results_before = len(results)   # rollback truncates Results back to here
    undo: list = []                 # (rel, restore-fn) for EVERY mutation below

    def _reverter(p: Path, data: bytes | None):
        """Undo closure: put `p` back exactly as it was before a mutation --
        rewrite the captured bytes, or remove a file that did not exist."""
        def _undo():
            if data is None:
                if p.is_file():
                    p.unlink()
            else:
                # Atomic replace: the failed write already moved `p` onto a
                # fresh inode, but a revert must still never mutate bytes in
                # place (a user or the ratchet may link that inode).
                fileio.write_bytes(p, data)
        return _undo

    def _finish(rel: str, path: Path, original: str | None, text: str, style,
                note: str, fuzzy: bool = False,
                renames_from: tuple[str, ...] = (),
                whole_file: bool = False) -> Result:
        """Validate, diff, back up, and write one finished file's content."""
        # Rename sources move their REAL bytes into the new path; the model's
        # creation REPLACE was only a placeholder. `moved_original` is what the
        # review diffs against and pins by sha (for a pure creation, "").
        moved_original = original or ""
        if renames_from:
            texts = []
            movable = []
            for old in renames_from:
                try:
                    otext, _ = fileio.read(root_resolved / old, errors="strict")
                except (OSError, UnicodeDecodeError):
                    continue     # unreadable or binary: left untouched
                texts.append(otext)
                movable.append(old)
            renames_from = tuple(movable)
            if texts:
                moved_original = texts[0]
                if len(texts) > 1:
                    note += (" [multiple rename sources; the first's contents "
                             "were moved -- the others are backed up and "
                             "removed, merge them manually]")
                if original is not None:      # existing target: append, never clobber
                    base = original.rstrip("\n")
                    add = "\n".join(t.rstrip("\n") for t in texts)
                    text = base + "\n" + add + "\n"
                else:                         # pure creation: placeholder + moved bytes
                    ph = text.strip()
                    body = texts[0].rstrip("\n")
                    if ph:
                        text = _placeholder_comment(ph, path.suffix.lower()) \
                               + body + "\n"
                    else:
                        text = body + "\n"
                    moved_original = ""       # nothing existed at the target
                note += f" [{len(text.splitlines())} line(s) in the final file]"

        # Validate the FINAL text, after the moved bytes are folded in.
        err = _validate(path, text)
        if err and original is not None and _validate(path, original):
            # The file did not parse before this edit either; refusing would
            # lock a broken file in that state forever. Apply with a warning.
            err = None
            note += (" [WARNING: this file does not parse -- it did not "
                     "before this edit either, so the edit was applied "
                     "anyway; finish fixing its syntax before relying on it]")
        if err:
            verb = "creation aborted" if original is None else "not written"
            return Result(False, rel, f"edit produced {err}; {verb}",
                          fuzzy=fuzzy)

        diff = "".join(difflib.unified_diff(
            moved_original.splitlines(keepends=True), text.splitlines(keepends=True),
            fromfile=f"a/{rel}", tofile=f"b/{rel}",
        ))
        sha = hashlib.sha256(moved_original.encode("utf-8")).hexdigest()
        if dry_run:
            return Result(True, rel, note + " (dry run)", diff, moved_original,
                          text, sha, fuzzy=fuzzy, renames_from=renames_from,
                          whole_file=whole_file)

        def _backup(src: Path, old_rel: str) -> None:
            if backup_dir is not None:
                fileio.backup_copy(src, backup_dir / old_rel)
            else:
                fileio.backup_copy(src, src.with_suffix(src.suffix + ".bak"),
                                   keep_existing=False)

        if backup and original is not None:
            _backup(path, rel)
        elif backup and original is None and backup_dir is not None:
            fileio.note_created(backup_dir, rel)

        # Sources are backed up under their OLD path so a snapshot restores a
        # rename by moving files back.
        for old in renames_from:
            opath = root_resolved / old
            try:
                if backup:
                    _backup(opath, old)
                _pre = opath.read_bytes() if opath.is_file() else None
                opath.unlink()
                if _pre is not None:
                    undo.append((old, _reverter(opath, _pre)))
            except FileNotFoundError:
                pass

        path.parent.mkdir(parents=True, exist_ok=True)
        _pre = path.read_bytes() if path.is_file() else None
        fileio.write(path, text, style)
        undo.append((rel, _reverter(path, _pre)))
        return Result(True, rel, note, diff, moved_original, text, sha,
                      fuzzy=fuzzy, renames_from=renames_from,
                      whole_file=whole_file)

    try:
        # Deletions run first: a later edit block for the same path then fails
        # with "does not exist" instead of silently resurrecting the file.
        for rel in delete_groups:
            try:
                path = security.resolve_safe_path(root_resolved, rel)
            except security.SecurityError as e:
                results.append(Result(False, rel, f"refused: {e}"))
                continue
            allowed, sensitive_error = security.check_sensitive_write(
                root_resolved, rel, action="delete", policy=policy)
            if not allowed:
                results.append(Result(False, rel,
                                      f"refused: {sensitive_error}"))
                continue
            if path.is_dir():
                results.append(Result(False, rel,
                                      "this path is a directory, not a file"))
                continue
            if not path.is_file():
                results.append(Result(
                    True, rel, "deleted: nothing to delete (already absent)"))
                continue
            pre = path.read_bytes()
            try:
                original, _sty = fileio.read(path, errors="replace")
            except OSError:
                original = ""
            diff = "".join(difflib.unified_diff(
                original.splitlines(keepends=True), [],
                fromfile=f"a/{rel}", tofile="/dev/null"))
            sha = hashlib.sha256(original.encode("utf-8")).hexdigest()
            note = (f"deleted ({len(original.splitlines())} line(s)); backed "
                    "up first -- undo / `pedacito restore` brings it back")
            if dry_run:
                results.append(Result(True, rel, note + " (dry run)", diff,
                                      original, "", sha, deleted=True))
                continue
            if backup:
                if backup_dir is not None:
                    fileio.backup_copy(path, backup_dir / rel)
                else:
                    fileio.backup_copy(
                        path, path.with_suffix(path.suffix + ".bak"),
                        keep_existing=False)
            path.unlink()
            undo.append((rel, _reverter(path, pre)))
            results.append(Result(True, rel, note, diff, original, "", sha,
                                  deleted=True))

        for rel, group in by_file.items():
            path = _resolve_inside(rel)
            if path is None:
                results.append(Result(False, rel, "refusing to edit outside project root"))
                continue
            allowed, sensitive_error = security.check_sensitive_write(
                root_resolved, rel, action="edit", policy=policy)
            if not allowed:
                results.append(Result(False, rel,
                                      f"refused: {sensitive_error}"))
                continue
            # D10: a rename's SOURCE is as sensitive as its destination; the
            # target group carries the source paths that `_finish` unlinks.
            blocked_source = ""
            for old in (o for o, new in renames.items()
                        if new == rel and o != rel):
                src_allowed, src_error = security.check_sensitive_write(
                    root_resolved, old, action="rename", policy=policy)
                if not src_allowed:
                    blocked_source = src_error
                    break
            if blocked_source:
                results.append(Result(False, rel,
                                      f"refused: {blocked_source}"))
                continue
            if path.is_dir():
                results.append(Result(False, rel, "this path is a directory, not a file"))
                continue

            if rel in orphan_placeholders:
                results.append(Result(False, rel, _ORPHAN_CREATE_MSG))
                continue
            if orphan_placeholders and all(not e.replace.strip() for e in group) \
                    and any(e.search.strip() for e in group):
                # The other half of a refused placeholder pair: applying it would
                # delete lines from the live source with no destination written.
                results.append(Result(
                    False, rel,
                    "not applying this clearing block: it looks like the other "
                    "half of a rename whose new path was refused (placeholder "
                    "creation with no matching source). Emit the rename pair "
                    "again with both blocks -- the old path's SEARCH quoting a "
                    "few real lines, the new path's SEARCH empty."))
                continue

            # A rename source equal to the target itself would read the placeholder
            # back as "moved content" and delete it.
            srcs = tuple(o for o, n in renames.items() if n == rel and o != rel)

            # A group of only empty-SEARCH blocks is a creation or an overwrite
            # (a path can be both a rename source and a creation target). Anchored
            # blocks are line replacements and never part of this path.
            creating = [e for e in group
                        if not e.search.strip() and not _is_anchored(e)]
            if creating and len(creating) == len(group):
                original = None
                style = None
                if path.is_file():
                    original, style = fileio.read(path, errors="replace")
                if original is None and not _plausible_new_path(rel):
                    results.append(Result(
                        False, rel,
                        "refusing to create this file: its name looks like a "
                        "copied source line, not a project path"))
                    continue
                text = _creation_text(group)
                note = f"created with {len(text.splitlines())} line(s)"
                if original is not None:
                    note = f"replaced contents ({len(original.splitlines())} -> {len(text.splitlines())} line(s))"
                # Claim only sources that still exist at apply time.
                srcs_now = tuple(o for o in srcs if (root_resolved / o).is_file())
                gone = [o for o in srcs if o not in srcs_now]
                if original is not None and not srcs_now:
                    # Overwriting an existing file with model-composed whole-file
                    # content is the edit-path twin of the `write` action, so it
                    # gets the same fragment guard (a window re-quote from a
                    # partial read would silently erase the rest of the file).
                    # A rename target is exempt: its real content arrives from
                    # the moved source, not from the REPLACE text.
                    refusal = fragment_refusal(rel, original, text, task=task_text)
                    if refusal:
                        results.append(Result(False, rel, refusal))
                        continue
                if srcs_now:
                    note += (" -- renamed from " + ", ".join(srcs_now) +
                             " (the moved contents are appended after the new "
                             "text; the old paths are removed)")
                if gone:
                    note += (" [already absent, nothing to move: " +
                             ", ".join(gone) + "]")
                results.append(_finish(rel, path, original, text, style, note,
                                       renames_from=srcs_now,
                                       whole_file=original is not None
                                       and not srcs_now))
                continue

            if not path.exists():
                # A missing file can only be created by empty-SEARCH blocks; a
                # mix with real searches is a failed edit against a file that is
                # not in the tree. The message must say so: "file not found"
                # alone invites the model to answer by renaming files.
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
                    "To RENAME a file into this path, also emit a block for "
                    "the old path whose SEARCH is its entire content and "
                    "whose REPLACE is empty -- Pedacito will move the real "
                    "contents and remove the old file." + hint))
                continue

            # The path's blocks apply IN ORDER, each anchored against the text the
            # previous one produced; the file is written (or refused) once at the end.
            try:
                original, style = fileio.read(path, errors="strict")
            except UnicodeDecodeError:
                results.append(Result(False, rel, "file is not valid UTF-8; "
                                      "refusing to edit it as text"))
                continue
            except OSError as e:
                results.append(Result(False, rel, f"could not read file: {e}"))
                continue
            text = original
            failed = False
            fuzzy_blocks: list[int] = []
            already: list[int] = []       # blocks whose REPLACE is already on disk
            reanchored: list[tuple[int, int]] = []  # numeric SEARCH -> LINES n-n
            whole_file = False            # a block replaced the entire file
            # Anchored blocks resolve to (a, b) here for failure messages, and
            # are rewritten to exact-text needles taken from the ORIGINAL file,
            # so an earlier block in the same group cannot shift their lines and
            # an externally drifted file still matches via locate()/fuzzy.
            anchored: dict[int, tuple[int, int]] = {}
            for k, e in enumerate(group, 1):
                if _is_anchored(e):
                    converted = False
                    if not e.lines and (e.symbol or e.insert_after):
                        # Zero-anchor splice (or a SYMBOL header no index
                        # resolved): locate the definition's exact span from the
                        # AST/tree-sitter on the evolving in-memory `text` so
                        # sequential SYMBOL/AFTER_SYMBOL edits compose cleanly.
                        sym = e.symbol or e.insert_after
                        span = locate_symbol_span(path, sym, text=text)
                        if span is None:
                            results.append(Result(
                                False, rel,
                                f"block {k}: SYMBOL anchor '{sym}' could not be "
                                "located in this file (missing, ambiguous, or the "
                                "file does not parse); check the spelling against "
                                "an `outline`, or `read` the symbol and emit the "
                                "block with LINES <start>-<end> instead"))
                            failed = True
                            break
                        s0, e0, base_ind = span
                        body = normalise_symbol_indent(e.replace, base_ind)
                        lspan = _line_span(text, s0, e0)
                        if lspan is None:
                            results.append(Result(
                                False, rel,
                                f"block {k}: symbol '{sym}' resolved to lines "
                                f"{s0}-{e0}, which are out of bounds in this "
                                "file; `read` it again"))
                            failed = True
                            break
                        needle = text[lspan[0]:lspan[1]]
                        if not needle.strip():
                            results.append(Result(
                                False, rel,
                                f"block {k}: symbol '{sym}' spans only blank "
                                "lines; nothing to anchor an edit against"))
                            failed = True
                            break
                        if e.insert_after:
                            repl = needle
                            if body.strip():
                                if not repl.endswith("\n"):
                                    repl += "\n"
                                repl += body + "\n"
                            anchored[k] = (s0, e0)
                            e = Edit(e.file, needle, repl)
                            converted = True
                        else:
                            repl = body
                            decs = _leading_decorators(text, s0, e0)
                            if decs and repl.strip() \
                                    and not repl.lstrip().startswith("@"):
                                repl = "\n".join(decs) + "\n" + repl
                            if repl and needle.endswith("\n") and not repl.endswith("\n"):
                                repl += "\n"
                            total_lines = len(text.splitlines())
                            if s0 <= 1 and e0 >= total_lines:
                                whole_file = True
                                whole = repl if not repl or repl.endswith("\n") \
                                    else repl + "\n"
                                refusal = fragment_refusal(rel, text, whole, task=task_text)
                                if refusal:
                                    results.append(Result(False, rel,
                                                          f"block {k}: {refusal}"))
                                    failed = True
                                    break
                            anchored[k] = (s0, e0)
                            e = Edit(e.file, needle, repl)
                            converted = True
                    if not converted:
                        if not e.lines:
                            results.append(Result(
                                False, rel,
                                f"block {k}: SYMBOL anchor '{e.symbol}' could not "
                                "be resolved to a definition in this file; `read` "
                                "the symbol and emit the block with "
                                "LINES <start>-<end> instead"))
                            failed = True
                            break
                        a, b = e.lines
                        lspan = _line_span(original, a, b)
                        if lspan is None:
                            results.append(Result(
                                False, rel,
                                f"block {k}: line range {a}-{b} is out of bounds "
                                f"-- this file has {len(original.splitlines())} "
                                "line(s); `read` it again and emit current line "
                                "numbers"))
                            failed = True
                            break
                        anchored[k] = (a, b)
                        needle = original[lspan[0]:lspan[1]]
                        if not needle.strip():
                            # A blank-lines-only needle would fall into the
                            # empty-SEARCH whole-text branch below; refuse instead.
                            results.append(Result(
                                False, rel,
                                f"block {k}: line range {a}-{b} is only blank "
                                "lines -- widen the anchored range to include "
                                "real code so it can be located"))
                            failed = True
                            break
                        # BUG-315: a LINES block that also quotes a SEARCH body
                        # carries the model's real anchor text. When the line
                        # numbers have drifted (an earlier hunk grew the file),
                        # the range-extracted needle points at the wrong lines
                        # and an append lands mid-file, unbalancing braces --
                        # the run rejects a reply the quoted text alone applies
                        # cleanly. Prefer the quoted SEARCH when it locates
                        # uniquely (exactly or whitespace-tolerantly) and
                        # relocate the anchor to it; the line range stays the
                        # fallback when the SEARCH is empty or unfindable.
                        if e.search.strip() and e.search != needle:
                            s_span = locate(original, e.search, cfg,
                                            replacement=e.replace)
                            if s_span is not None and not s_span.fuzzy:
                                needle = original[s_span.start:s_span.stop]
                                a = original[:s_span.start].count("\n") + 1
                                b = a + len(needle.splitlines()) - 1
                                anchored[k] = (a, b)
                        repl = e.replace
                        if e.symbol:
                            # A SYMBOL-anchored body written at the wrong base
                            # indent (the habitual column-0 `def` for a class
                            # method) is re-aligned to the definition's real
                            # indent; the range's first line is the target level.
                            # Idempotent for a body already written at the right
                            # depth.
                            repl = normalise_symbol_indent(repl, _indent_of_line(
                                original, a))
                            # Decorators are part of the definition's identity:
                            # carry them over when the new body does not re-echo
                            # them, so forgetting `@staticmethod` cannot silently
                            # change behaviour. An empty REPLACE deletes the whole
                            # definition, decorators included.
                            decs = _leading_decorators(original, a, b)
                            if decs and repl.strip() \
                                    and not repl.lstrip().startswith("@"):
                                repl = "\n".join(decs) + "\n" + repl
                        if repl and needle.endswith("\n") and not repl.endswith("\n"):
                            repl += "\n"
                        # A LINES range that covers the whole file is a whole-file
                        # replacement, even though its SEARCH text is non-empty.
                        # The fragment guard must cover this shape too: the model
                        # can otherwise dodge it by sending `LINES 1-N` with an
                        # empty replacement instead of an empty SEARCH (test_1a
                        # O7: a 214-line src/main.js was reduced to 52 lines this
                        # way after the `write` action had already refused the
                        # same fragment).
                        total_lines = len(original.splitlines())
                        if a <= 1 and b >= total_lines:
                            whole_file = True
                            whole = repl if not repl or repl.endswith("\n") \
                                else repl + "\n"
                            refusal = fragment_refusal(rel, text, whole, task=task_text)
                            if refusal:
                                results.append(Result(False, rel,
                                                      f"block {k}: {refusal}"))
                                failed = True
                                break
                        e = Edit(e.file, needle, repl)
                if not e.search.strip():
                    # An empty SEARCH mixed in with real edits replaces the WHOLE
                    # text that later blocks anchor against. That replacement is
                    # model-composed whole-file content, so it gets the same
                    # fragment guard the `write` action applies.
                    whole_file = True
                    whole = e.replace if not e.replace or e.replace.endswith("\n") \
                        else e.replace + "\n"
                    refusal = fragment_refusal(rel, text, whole, task=task_text)
                    if refusal:
                        results.append(Result(False, rel, f"block {k}: {refusal}"))
                        failed = True
                        break
                    text = whole
                    continue
                span = locate(text, e.search, cfg, replacement=e.replace)
                if span is None:
                    n = text.count(e.search)
                    if n == 0 and _replace_already_present(
                            text, e.replace, allow_single=bool(already)) \
                            and not e.exact:
                        # The block was already applied -- typically a gather
                        # write landed it before the answer phase, so the SEARCH
                        # no longer matches but the REPLACE content is on disk.
                        # Score it clean instead of failing the whole file.
                        already.append(k)
                        continue
                    if n == 0 and k not in anchored:
                        # Numeric-SEARCH auto-reanchor: a SEARCH body that is
                        # just a line number ("4") is the tiny-profile slip of
                        # quoting the read's line-number prefix instead of the
                        # line's text. Interpret it as LINES n-n and apply the
                        # replacement to that exact line -- deterministic, and
                        # one whole retry round saved. A number that DOES occur
                        # literally in the file keeps the normal match semantics.
                        num = _NUMERIC_SEARCH_RE.fullmatch(e.search.strip())
                        if num:
                            lineno = int(num.group(0))
                            total = len(text.splitlines())
                            lspan = _line_span(text, lineno, lineno)
                            if lspan is not None:
                                seg = text[lspan[0]:lspan[1]]
                                repl = e.replace
                                if repl and seg.endswith("\n") \
                                        and not repl.endswith("\n"):
                                    repl += "\n"
                                text = (text[:lspan[0]] + repl
                                        + text[lspan[1]:])
                                reanchored.append((k, lineno))
                                continue
                            results.append(Result(
                                False, rel,
                                f"block {k}: the SEARCH body is just the "
                                f"number {lineno}, read as a line number, but "
                                f"this file has {total} line(s). `read` it "
                                "again and anchor the block with "
                                f"`--- FILE: {rel} LINES <start>-<end>` using "
                                "the real line numbers, or quote the line's "
                                "exact text in SEARCH."))
                            failed = True
                            break
                    if k in anchored:
                        a, b = anchored[k]
                        why = (f"anchored range LINES {a}-{b} cannot be applied: "
                               + ("an earlier block in this reply already changed "
                                  "those lines" if n == 0 else
                                  f"its text appears {n} times in the file -- "
                                  "widen the range until it is unique, or emit a "
                                  "SEARCH block with more context"))
                    else:
                        why = ("search block not found -- it must match the file "
                               "exactly. If this block was meant to replace the "
                               "WHOLE file, do not quote the old text at all: emit "
                               "one block for the file with an EMPTY SEARCH section "
                               "(nothing between <<<<<<< SEARCH and =======)"
                               if n == 0 else
                               f"search block appears {n} times -- include more "
                               "surrounding lines")
                        if n == 0:
                            # Show where the text most nearly matches, with real
                            # line numbers, so the re-anchor is copyable.
                            why += _closest_hint(text, e.search, cfg)
                    results.append(Result(False, rel, f"block {k}: {why}"))
                    failed = True
                    break
                if _replace_already_present(text, e.replace, span.start) \
                        and not e.exact:
                    # The replacement already sits verbatim at the matched span:
                    # a repair round re-emitted an append-style block whose
                    # content landed in an earlier round. Applying again would
                    # duplicate it (the triple-appended test pair), so score the
                    # block as already applied.
                    already.append(k)
                    continue
                if span.start == 0 and span.stop == len(text):
                    # A SEARCH quoting the ENTIRE file is a whole-file rewrite
                    # just like an empty SEARCH: mark it so the lost-export and
                    # context-pressure gates see this shape too.
                    whole_file = True
                text = text[:span.start] + span.replacement + text[span.stop:]
                if span.fuzzy:
                    fuzzy_blocks.append(k)

            if failed:
                continue

            if original.strip() and not text.strip() \
                    and not any(getattr(e, "exact", False) for e in group):
                # The blocks deleted every line (a SEARCH quoting the whole
                # file with an empty REPLACE): the same wipe the empty-SEARCH
                # guard refuses, reached through the anchored path. An exact
                # engine-computed codemod (remove the last import, say) may
                # deliberately produce an empty module.
                results.append(Result(False, rel, _wipe_refusal(rel)))
                continue

            if text == original and already:
                # Every block was already applied: nothing to write. Rewriting
                # the identical bytes would still count the file as changed
                # (backup, reindex, post-edit hooks) for nothing.
                results.append(Result(
                    True, rel,
                    "no change: block(s) "
                    + ", ".join(str(b) for b in already)
                    + " already applied -- their REPLACE content is present in "
                      "the file"))
                continue

            note = f"applied {len(group)} block(s)"
            if already:
                note += (" [block(s) " + ", ".join(str(b) for b in already)
                         + " already applied -- content present, left as-is]")
            if reanchored:
                note += (" [numeric SEARCH auto-reanchored to LINES "
                         + ", ".join(f"{ln}-{ln} on block {b}"
                                     for b, ln in reanchored)
                         + "; the number was read as a line number, not text]")
            if fuzzy_blocks:
                note += (" [matched approximately on block"
                         + ("s " if len(fuzzy_blocks) > 1 else " ")
                         + ", ".join(str(b) for b in fuzzy_blocks)
                         + "; review the diff carefully]")
            results.append(_finish(rel, path, original, text, style, note,
                                   fuzzy=bool(fuzzy_blocks),
                                   whole_file=whole_file))

    except OSError as exc:
        # all-or-nothing batch: without this, a disk or permission failure
        # mid-write left the files already written changed while the rest of
        # the batch never ran -- a half-applied working tree. Undo every
        # mutation this call made (newest first), then report ONE failed
        # Result: the per-file outcomes above no longer describe the tree.
        del results[results_before:]
        total, undone = len(undo), 0
        for _desc, fn in reversed(undo):
            try:
                fn()
                undone += 1
            except OSError:
                pass
        undo.clear()
        msg = (f"write batch failed: {exc}; rolled back {undone} of {total} "
               "file mutation(s) to their pre-batch state")
        if undone != total:
            msg += (f"; {total - undone} could not be restored -- run "
                    "`pedacito restore`")
        results.append(Result(False, "*", msg))

    return results


# --------------------------------------------------------------------------
# Staged atomic batch (WP-4105C)
# --------------------------------------------------------------------------
def _backup_one(src: Path, rel: str, backup_dir: Path | None) -> None:
    """Back up `src`'s exact bytes before an atomic write/delete, matching the
    ordinary `apply_edits` snapshot behaviour (first copy per run wins)."""
    if backup_dir is not None:
        fileio.backup_copy(src, backup_dir / rel)
    else:
        fileio.backup_copy(src, src.with_suffix(src.suffix + ".bak"),
                           keep_existing=False)


def apply_edits_atomic(root: Path, edits: list[Edit], *, backup: bool = True,
                       backup_dir: Path | None = None, cfg=None,
                       allowlist: set[str] | None = None,
                       expected_sha256: dict[str, str] | None = None
                       ) -> list[Result]:
    """Staged all-or-nothing multi-file edit transaction (WP-4105C).

    The atomic contract, distinct from the per-file legacy apply:

    * every candidate is validated (policy, protected/sensitive paths,
      allowlist, rename pairing, fragment/wipe guards, syntax) against the
      current tree BEFORE anything is written;
    * the preimages of every affected file are snapshotted to a durable WAL
      (``journal.begin_atomic``) before the first write;
    * preimage hashes are re-verified just before commit, so a file changed
      underneath the batch aborts it;
    * an in-process write failure rolls back every file already written;
    * a crash mid-commit is recovered on the next run by
      ``journal.recover_atomic`` (restore preimages / remove creations), not
      by instantaneous filesystem-wide atomic visibility.

    Returns results shaped like :func:`apply_edits`: on any preflight failure,
    the failed results plus one "held" result per clean file (nothing was
    written). ``cfg.atomic_edits`` semantics (explicit value vs tier default)
    are resolved by :func:`config.atomic_edits_enabled`, not here.

    ``expected_sha256`` optionally pins the raw bytes each project-relative
    path must still have before commit, so an engine that previewed a file can
    refuse a stale apply even when a later search/replace would still match.
    """
    from . import journal as journal_mod

    root = Path(root)
    root_resolved = root.resolve()

    # A crashed previous run may have left a "prepared" transaction mid-commit:
    # roll it back before staging anything new.
    try:
        journal_mod.recover_atomic(root_resolved)
    except OSError:
        pass

    # Preflight: validate every candidate, writing nothing.
    dry = apply_edits(root, edits, backup=False, dry_run=True, cfg=cfg,
                      allowlist=allowlist)
    bad = [r for r in dry if not r.ok]
    if bad:
        held = [Result(False, r.file,
                       "not written: atomic_edits is on and another file's "
                       "edit failed; fix that one and resend ALL the blocks")
                for r in dry if r.ok]
        return bad + held

    # Build the commit plan from the clean dry run. A no-op result (empty sha)
    # changes nothing; a rename folds its sources into the target's deletion
    # set.
    write_rels: dict[str, str] = {}
    delete_rels: set[str] = set()
    for r in dry:
        if not r.sha:
            continue
        if r.deleted:
            delete_rels.add(r.file)
            continue
        write_rels[r.file] = r.after
        delete_rels.update(r.renames_from)

    # Read the preimage bytes of every existing affected file; new files are
    # recorded as creations so recovery can remove them.
    preimages: dict[str, bytes] = {}
    created: list[str] = []
    for rel in sorted(set(write_rels) | delete_rels):
        try:
            path = security.resolve_safe_path(root_resolved, rel)
        except security.SecurityError as exc:
            return [Result(False, rel, f"refused: {exc}")]
        if path.is_file():
            preimages[rel] = path.read_bytes()
        elif rel in write_rels:
            created.append(rel)

    # A caller that previewed bytes can pin them: refuse a stale apply before
    # the durable transaction record is even created.
    if expected_sha256:
        for rel, wanted in expected_sha256.items():
            data = preimages.get(rel)
            got = hashlib.sha256(data).hexdigest() if data is not None else ""
            if data is None or got != str(wanted):
                return [Result(False, rel,
                               "atomic batch aborted: this file changed after "
                               "the preview; nothing was written")]

    # Snapshot + WAL durable before writes; a snapshot failure stops the batch.
    try:
        txn = journal_mod.begin_atomic(root_resolved, preimages, created)
    except OSError as exc:
        return [Result(False, "*",
                       f"atomic batch failed: could not snapshot preimages "
                       f"before writing ({exc}); nothing was written")]

    def _abort(rel: str, why: str) -> list[Result]:
        journal_mod.finish_atomic(root_resolved, txn.tx_id, "rolled_back")
        journal_mod.remove_atomic(root_resolved, txn.tx_id)
        return [Result(False, rel, why)]

    # Re-verify preimage hashes just before commit.
    for rel, data in preimages.items():
        try:
            path = security.resolve_safe_path(root_resolved, rel)
        except security.SecurityError as exc:
            return _abort(rel, f"atomic batch aborted: {exc}")
        if not path.is_file() or path.read_bytes() != data:
            return _abort(rel, "atomic batch aborted: this file changed on "
                               "disk after it was validated; resend the edit "
                               "blocks")

    # Commit, with an in-process rollback ledger.
    undo: list[tuple[str, bytes | None]] = []
    try:
        for rel in write_rels:
            after = write_rels[rel]
            data = preimages.get(rel)
            style = fileio.detect(data) if data is not None else fileio.TextStyle()
            path = security.resolve_safe_path(root_resolved, rel)
            existed = path.is_file()
            if backup and existed:
                _backup_one(path, rel, backup_dir)
            elif backup and not existed and backup_dir is not None:
                fileio.note_created(backup_dir, rel)
            pre = path.read_bytes() if existed else None
            path.parent.mkdir(parents=True, exist_ok=True)
            fileio.write(path, after, style)
            undo.append((rel, pre))
        for rel in sorted(delete_rels):
            if rel in write_rels:
                continue               # a target is never also a source
            path = security.resolve_safe_path(root_resolved, rel)
            if not path.is_file():
                continue
            if backup:
                _backup_one(path, rel, backup_dir)
            pre = path.read_bytes()
            path.unlink()
            undo.append((rel, pre))
    except (OSError, security.SecurityError) as exc:
        undone = 0
        for rel, pre in reversed(undo):
            try:
                path = security.resolve_safe_path(root_resolved, rel)
                if pre is None:
                    if path.is_file() or path.is_symlink():
                        path.unlink()
                else:
                    fileio.write_bytes(path, pre)
                undone += 1
            except (OSError, security.SecurityError):
                pass
        journal_mod.finish_atomic(root_resolved, txn.tx_id, "rolled_back")
        journal_mod.remove_atomic(root_resolved, txn.tx_id)
        msg = (f"write batch failed: {exc}; rolled back {undone} of "
               f"{len(undo)} file mutation(s) to their pre-batch state")
        if undone != len(undo):
            msg += (f"; {len(undo) - undone} could not be restored -- run "
                    "`pedacito restore`")
        return [Result(False, "*", msg)]

    journal_mod.finish_atomic(root_resolved, txn.tx_id, "committed")
    journal_mod.remove_atomic(root_resolved, txn.tx_id)

    # The dry-run already computed each diff/before/after/sha; strip the
    # dry-run marker and return it as the applied result set.
    out: list[Result] = []
    for r in dry:
        if not r.sha:
            out.append(r)
            continue
        out.append(Result(r.ok, r.file, r.message.replace(" (dry run)", ""),
                          r.diff, r.before, r.after, r.sha, fuzzy=r.fuzzy,
                          renames_from=r.renames_from,
                          whole_file=r.whole_file, deleted=r.deleted))
    return out


# --------------------------------------------------------------------------
# Structured edit parser and guard (WP-4101A)
# --------------------------------------------------------------------------
# Structured mode requests one strict JSON object and never salvages embedded
# JSON or applies a partial reply. The five actions and their field contract
# (unused strings are ""):
#
#     replace_symbol       symbol != "", search == "", replacement != ""
#     insert_after_symbol  symbol != "", search == "", replacement != ""
#     search_replace       symbol != "", search != "", replacement any
#     append               symbol == "", search == "", replacement != ""
#     delete_symbol        symbol != "", search == "", replacement == ""
#
# A reply of no change is the explicit {"version":1,"intents":[],"edits":[]}.

STRUCTURED_ACTIONS = (
    "replace_symbol", "insert_after_symbol", "search_replace",
    "append", "delete_symbol",
)
_MAX_STRUCTURED = 8
_WHY_MAX = 300
_INTENT_KEYS = frozenset({"file", "symbol", "why"})
_EDIT_KEYS = frozenset({"edit_action", "file", "symbol", "search",
                        "replacement"})

# The strict JSON schema requested in structured mode (WP-4101A/B). It mirrors
# `parse_structured_reply`: one object, `additionalProperties: false`, required
# `version`/`intents`/`edits`; `version` is the integer 1; `intents`/`edits`
# are 0-8 objects (the explicit no-change reply is empty arrays). The local
# guard still enforces every operation-specific field, so a server whose
# schema support is incomplete is never trusted on the schema alone.
STRUCTURED_SCHEMA_NAME = "structured_edits"
STRUCTURED_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["version", "intents", "edits"],
    "properties": {
        "version": {"type": "integer", "const": 1},
        "intents": {
            "type": "array",
            "maxItems": _MAX_STRUCTURED,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["file", "symbol", "why"],
                "properties": {
                    "file": {"type": "string"},
                    "symbol": {"type": "string"},
                    "why": {"type": "string", "minLength": 1,
                            "maxLength": _WHY_MAX},
                },
            },
        },
        "edits": {
            "type": "array",
            "maxItems": _MAX_STRUCTURED,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["edit_action", "file", "symbol", "search",
                             "replacement"],
                "properties": {
                    "edit_action": {"type": "string",
                                    "enum": list(STRUCTURED_ACTIONS)},
                    "file": {"type": "string"},
                    "symbol": {"type": "string"},
                    "search": {"type": "string"},
                    "replacement": {"type": "string"},
                },
            },
        },
    },
}


@dataclass(frozen=True)
class StructuredIntent:
    """One structured edit intent: what the model plans to change and why."""

    file: str
    symbol: str
    why: str


@dataclass(frozen=True)
class StructuredEdit:
    """One structured edit action. Unused string fields are ``""`` (see the
    action table in the module docstring)."""

    edit_action: str
    file: str
    symbol: str
    search: str
    replacement: str


class StructuredReplyError(ValueError):
    """A reply failed strict structured-mode schema parsing."""


@dataclass(frozen=True)
class StructuredMaterialization:
    """The dry-run product of validated structured edits; nothing is written.

    ``buffers`` maps each touched file (normalised relative path) to its final
    LF-normalised text; ``styles`` records each file's original encoding so an
    apply can write the bytes back unchanged (CRLF/BOM preserved). ``results``
    carries the per-file before/after/diff/sha for the reviewer and the
    existing verifier gates. ``edits`` holds one legacy whole-file ``Edit``
    per file for compatibility and atomic apply. ``errors`` lists every
    refusal; empty means clean.
    """

    files: tuple[str, ...]
    buffers: dict[str, str]
    styles: dict[str, fileio.TextStyle]
    results: tuple[Result, ...]
    edits: tuple[Edit, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _strip_structured_fences(text: str) -> str:
    """Strip one wrapping markdown fence around the whole reply (normalisation,
    not salvage); the body must still be one strict JSON object."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^\s*```[^\n]*\n?", "", raw, count=1)
        raw = re.sub(r"\n?\s*```\s*$", "", raw, count=1)
    return raw.strip()


def _reject_keys(item: dict, allowed: frozenset, kind: str) -> None:
    """Enforce ``additionalProperties: false`` plus required keys."""
    extra = sorted(set(item) - allowed)
    if extra:
        raise StructuredReplyError(
            f"{kind} has unknown field(s): {', '.join(extra)} "
            "(additionalProperties is false)")
    missing = sorted(allowed - set(item))
    if missing:
        raise StructuredReplyError(
            f"{kind} is missing required field(s): {', '.join(missing)}")


def _require_string(item: dict, key: str, kind: str) -> str:
    value = item.get(key)
    if not isinstance(value, str):
        raise StructuredReplyError(f"{kind} field {key!r} must be a string")
    return value


def _parse_intent(item) -> StructuredIntent:
    if not isinstance(item, dict):
        raise StructuredReplyError("each intent must be a JSON object")
    _reject_keys(item, _INTENT_KEYS, "intent")
    file = _require_string(item, "file", "intent")
    symbol = _require_string(item, "symbol", "intent")
    why = _require_string(item, "why", "intent")
    if not why.strip():
        raise StructuredReplyError("intent 'why' must be nonblank")
    if len(why) > _WHY_MAX:
        raise StructuredReplyError(
            f"intent 'why' is {len(why)} characters; it is capped at "
            f"{_WHY_MAX}")
    return StructuredIntent(file=file, symbol=symbol, why=why)


def _parse_edit(item) -> StructuredEdit:
    if not isinstance(item, dict):
        raise StructuredReplyError("each edit must be a JSON object")
    _reject_keys(item, _EDIT_KEYS, "edit")
    action = _require_string(item, "edit_action", "edit")
    if action not in STRUCTURED_ACTIONS:
        raise StructuredReplyError(
            f"unknown edit_action {action!r} (expected one of "
            f"{', '.join(STRUCTURED_ACTIONS)})")
    file = _require_string(item, "file", "edit")
    symbol = _require_string(item, "symbol", "edit")
    search = _require_string(item, "search", "edit")
    replacement = _require_string(item, "replacement", "edit")
    return StructuredEdit(action, file, symbol, search, replacement)


def parse_structured_reply(text: str) -> tuple[list[StructuredIntent],
                                                list[StructuredEdit]]:
    """Parse one strict structured-edit reply into ``(intents, edits)``.

    The reply must be exactly one JSON object (an optional wrapping code fence
    is tolerated) with ``additionalProperties: false`` and required
    ``version``, ``intents``, ``edits``. ``version`` must be the integer 1;
    ``intents`` and ``edits`` are arrays of at most 8 objects with the exact
    keys documented above. There is no JSON salvage and no partial
    application: any deviation raises :class:`StructuredReplyError`.
    """
    raw = _strip_structured_fences(text)
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise StructuredReplyError(
            f"the reply is not one valid JSON object: {exc}") from exc
    if not isinstance(obj, dict):
        raise StructuredReplyError("the reply is not a JSON object")
    _reject_keys(obj, frozenset({"version", "intents", "edits"}), "reply")
    version = obj["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise StructuredReplyError("version must be the integer 1")
    intents_raw = obj["intents"]
    edits_raw = obj["edits"]
    if not isinstance(intents_raw, list) or not isinstance(edits_raw, list):
        raise StructuredReplyError("'intents' and 'edits' must be arrays")
    if len(intents_raw) > _MAX_STRUCTURED or len(edits_raw) > _MAX_STRUCTURED:
        raise StructuredReplyError(
            f"at most {_MAX_STRUCTURED} intents and edits are allowed")
    intents = [_parse_intent(i) for i in intents_raw]
    edits = [_parse_edit(e) for e in edits_raw]
    return intents, edits


def _action_field_errors(e: StructuredEdit) -> list[str]:
    """Operation-specific field constraints, enforced locally even when the
    server's JSON-schema support is incomplete."""
    errs: list[str] = []
    action = e.edit_action
    if not e.file.strip():
        errs.append(f"{action}: 'file' must name a project file")
        return errs
    if action == "append":
        if e.symbol:
            errs.append("append: 'symbol' must be empty (append is whole-file)")
        if e.search:
            errs.append("append: 'search' must be empty")
        if not e.replacement.strip():
            errs.append("append: 'replacement' must be nonblank")
    elif action in ("replace_symbol", "insert_after_symbol"):
        if not e.symbol:
            errs.append(f"{action}: 'symbol' must name a symbol")
        if e.search:
            errs.append(f"{action}: 'search' must be empty")
        if not e.replacement.strip():
            errs.append(f"{action}: 'replacement' must be nonblank"
                        + ("" if action == "insert_after_symbol"
                           else " (use delete_symbol to remove a symbol)"))
    elif action == "search_replace":
        if not e.symbol:
            errs.append("search_replace: 'symbol' must name a symbol")
        if not e.search:
            errs.append("search_replace: 'search' must be nonblank")
    elif action == "delete_symbol":
        if not e.symbol:
            errs.append("delete_symbol: 'symbol' must name a symbol")
        if e.search:
            errs.append("delete_symbol: 'search' must be empty")
        if e.replacement:
            errs.append("delete_symbol: 'replacement' must be empty")
    return errs


def _empty_materialization(errors: list[str]) -> StructuredMaterialization:
    return StructuredMaterialization((), {}, {}, (), (), tuple(errors))


def _span_lines(text: str, start: int, end: int) -> tuple[int, int]:
    """1-based inclusive line range of a half-open char span."""
    a = text.count("\n", 0, start) + 1
    b = a if end <= start else text.count("\n", 0, end - 1) + 1
    return a, b


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _covered(a: int, b: int, read: list[tuple[int, int]]) -> bool:
    return any(s <= a and b <= t for s, t in _merge_spans(read))


def _lookup_read(read_spans, key: str):
    """The read ledger for `key`, or None when no ledger was supplied, or []
    when the file was never read."""
    if read_spans is None:
        return None
    if key in read_spans:
        return list(read_spans[key])
    for k, v in read_spans.items():
        if k.endswith("/" + key) or key.endswith("/" + k):
            return list(v)
    return []


def _symbol_missing(rel: str, e: StructuredEdit, vpath, original: str,
                    errors: list[str]) -> None:
    """Append a missing-vs-ambiguous diagnostic for an unresolvable symbol."""
    spans = file_symbol_spans(vpath, text=original)
    if spans is None:
        errors.append(f"{rel}: cannot parse the file to resolve symbol "
                      f"{e.symbol!r}")
        return
    matches = [q for (q, _s, _e) in spans
               if q == e.symbol or q.endswith("." + e.symbol)]
    if not matches:
        errors.append(f"{rel}: no definition named {e.symbol!r} (unknown "
                      "symbol); check the spelling against an `outline`")
    else:
        uniq = sorted({q for q in matches})
        errors.append(f"{rel}: symbol {e.symbol!r} is ambiguous "
                      f"({len(uniq)} definitions match: "
                      f"{', '.join(uniq[:5])})")


def _resolve_structured_edit(vpath, e: StructuredEdit, original: str,
                             exists: bool, task_text: str,
                             errors: list[str]):
    """Resolve one StructuredEdit against ``original`` (``""`` for a new file).

    Returns ``(start, end, new_text, kind)`` in original char coordinates
    (half-open), or None after appending an error. ``kind`` is one of
    ``create``, ``append``, ``replace``, ``delete``.
    """
    action = e.edit_action
    if not exists:
        # Only a whole-file append (symbol == "") may create a file. Every
        # other action needs an existing definition to resolve against.
        if action != "append":
            errors.append(f"{e.file}: cannot {action} a file that does not "
                          "exist yet; use append with an empty symbol to "
                          "create it")
            return None
        text = e.replacement
        if not text.endswith("\n"):
            text += "\n"
        return (0, 0, text, "create")

    if action == "append":
        sep = "" if not original or original.endswith("\n") else "\n"
        body = sep + e.replacement
        if not body.endswith("\n"):
            body += "\n"
        return (len(original), len(original), body, "append")

    hit = locate_symbol_span(vpath, e.symbol, text=original)
    if hit is None:
        _symbol_missing(e.file, e, vpath, original, errors)
        return None
    s0, e0, base_ind = hit
    sym_span = _line_span(original, s0, e0)
    if sym_span is None:
        errors.append(f"{e.file}: symbol {e.symbol!r} resolved to lines "
                      f"{s0}-{e0}, which are out of bounds")
        return None
    cs, ce = sym_span
    needle = original[cs:ce]

    if action == "replace_symbol":
        body = normalise_symbol_indent(e.replacement, base_ind)
        decs = _leading_decorators(original, s0, e0)
        if decs and body.strip() and not body.lstrip().startswith("@"):
            body = "\n".join(decs) + "\n" + body
        if body and needle.endswith("\n") and not body.endswith("\n"):
            body += "\n"
        if body == needle:
            errors.append(f"{e.file}: replace_symbol {e.symbol!r} is a no-op "
                          "(the replacement equals the current definition)")
            return None
        total_lines = len(original.splitlines())
        if s0 <= 1 and e0 >= total_lines:
            whole = body if not body or body.endswith("\n") else body + "\n"
            refusal = fragment_refusal(e.file, original, whole, task=task_text)
            if refusal:
                errors.append(f"{e.file}: {refusal}")
                return None
        return (cs, ce, body, "replace")

    if action == "insert_after_symbol":
        body = normalise_symbol_indent(e.replacement, base_ind)
        repl = needle
        if body.strip():
            if not repl.endswith("\n"):
                repl += "\n"
            repl += body + "\n"
        return (cs, ce, repl, "replace")

    if action == "search_replace":
        n = needle.count(e.search)
        if n != 1:
            errors.append(f"{e.file}: search text must occur exactly once "
                          f"within symbol {e.symbol!r} (found {n}); widen the "
                          "search text or narrow the symbol")
            return None
        if e.replacement == e.search:
            errors.append(f"{e.file}: search_replace is a no-op (replacement "
                          "equals the search)")
            return None
        idx = needle.index(e.search)
        return (cs + idx, cs + idx + len(e.search), e.replacement, "replace")

    if action == "delete_symbol":
        if not needle.strip():
            errors.append(f"{e.file}: delete_symbol {e.symbol!r} spans only "
                          "blank lines; nothing to delete")
            return None
        return (cs, ce, "", "delete")

    errors.append(f"{e.file}: unsupported edit_action {action!r}")
    return None


def materialize_structured_edits(root, intents: list[StructuredIntent],
                                 edits: list[StructuredEdit], *,
                                 allowed_files: set[str] | None = None,
                                 read_spans: dict | None = None,
                                 cfg=None) -> StructuredMaterialization:
    """Dry-run validated structured edits against evolving in-memory buffers.

    Performs the strict preflight and returns the final buffers, per-file
    legacy ``Result``/``Edit`` objects, and any refusals (in ``errors``), never
    touching the filesystem for writes. Checks, in order:

    * every edit's exact canonical ``(file, symbol)`` matches a declared
      intent, and operation-specific fields are valid;
    * paths are contained (``security.resolve_safe_path``), non-sensitive,
      non-protected, policy-allowed, and inside ``allowed_files``;
    * a ``symbol=""`` file intent only creates an authorized new file or an
      explicit whole-file append; existing symbols resolve uniquely and stay
      inside the read ledger (``read_spans``);
    * ``search`` occurs exactly once within its symbol, no edit is a no-op,
      and no two edits overlap;
    * the resulting buffers parse (no new syntax errors) and change bytes.

    ``allowed_files`` is a set of normalised paths the task may modify; pass
    None to skip the file allowlist. ``read_spans`` maps
    ``fileio.path_key(file)`` to ``[(start, end)]`` 1-based inclusive line
    spans; pass None to skip the read-scope gate.
    """
    intents = list(intents or ())
    edits = list(edits or ())
    errors: list[str] = []
    task_text = str(getattr(cfg, "_task_text", "") or "")
    policy = getattr(cfg, "policy", None)

    if (not intents) != (not edits):
        errors.append("intents and edits must both be empty (a no-change reply "
                      'is {"version":1,"intents":[],"edits":[]}) or both '
                      "non-empty")
        return _empty_materialization(errors)
    if not intents:
        return _empty_materialization([])

    for e in edits:
        errors.extend(_action_field_errors(e))
    if errors:
        # No partial application: a malformed edit rejects the whole reply.
        return _empty_materialization(errors)

    intent_keys = {(fileio.path_key(i.file), i.symbol) for i in intents}

    try:
        root_resolved = Path(root).resolve()
    except OSError as exc:
        errors.append(f"project root could not be resolved: {exc}")
        return _empty_materialization(errors)

    allowed = None
    if allowed_files is not None:
        allowed = {fileio.path_key(p) for p in allowed_files}

    ordered: list[str] = []
    by_file: dict[str, list[StructuredEdit]] = {}
    rel_of: dict[str, str] = {}
    for e in edits:
        rel = fileio.normalise_rel(e.file)
        key = fileio.path_key(rel)
        if key not in by_file:
            by_file[key] = []
            ordered.append(key)
            rel_of[key] = rel
        by_file[key].append(e)

    files: list[str] = []
    buffers: dict[str, str] = {}
    styles: dict[str, fileio.TextStyle] = {}
    results: list[Result] = []
    legacy_edits: list[Edit] = []

    for key in ordered:
        rel = rel_of[key]
        group = by_file[key]

        # ---- edit -> intent exact canonical match -------------------------
        for e in group:
            if (key, e.symbol) not in intent_keys:
                errors.append(f"{rel}: edit (file={e.file}, symbol="
                              f"{e.symbol or ''!r}) does not match any "
                              "declared intent")

        # ---- containment / sensitive / protected / policy / allowlist -----
        try:
            path = security.resolve_safe_path(root_resolved, rel)
        except security.SecurityError as exc:
            errors.append(f"{rel}: {exc}")
            continue
        s_ok, s_err = security.check_sensitive_write(
            root_resolved, rel, action="edit", policy=policy)
        if not s_ok:
            errors.append(f"{rel}: {s_err}")
            continue
        why = safety.is_protected(rel, cfg)
        if why:
            errors.append(f"{rel}: refused: {why}")
            continue
        try:
            from . import policy as policy_mod
            d = policy_mod.enforce(cfg, "write", rel, root=root_resolved)
            if d is not None and getattr(d, "decision", None) == "deny":
                errors.append(f"{rel}: denied by policy "
                              f"({getattr(d, 'matched_rule', '?')}): "
                              f"{getattr(d, 'reason', '')}")
                continue
        except Exception as exc:                       # fail closed
            errors.append(f"{rel}: denied by policy: policy evaluation error: "
                          f"{exc}")
            continue
        if allowed is not None and key not in allowed:
            errors.append(f"{rel}: not in the authorized file set "
                          "(allowed_files)")
            continue

        read = _lookup_read(read_spans, key)

        exists = path.is_file()
        if exists:
            try:
                original, style = fileio.read(path, errors="strict")
            except UnicodeDecodeError:
                errors.append(f"{rel}: file is not valid UTF-8; refusing to "
                              "edit it as text")
                continue
            except OSError as exc:
                errors.append(f"{rel}: could not read file: {exc}")
                continue
        else:
            if len(group) > 1:
                errors.append(f"{rel}: a new file may only be created by a "
                              "single append edit")
                continue
            original = ""
            style = fileio.TextStyle()

        vpath = Path(root) / rel

        ops: list[tuple[int, int, int, str, str]] = []
        bad = False
        for idx, e in enumerate(group):
            op = _resolve_structured_edit(vpath, e, original, exists, task_text,
                                          errors)
            if op is None:
                bad = True
                continue
            start, end, new, kind = op
            if read is not None and exists and kind in ("replace", "delete"):
                a, b = _span_lines(original, start, end)
                if not read:
                    errors.append(f"{rel}: {e.edit_action} {e.symbol or e.search!r} "
                                  "targets a file this run never read")
                    bad = True
                elif not _covered(a, b, read):
                    errors.append(f"{rel}: lines {a}-{b} are outside the lines "
                                  "read; the edit may be a guess")
                    bad = True
            elif read is not None and exists and kind == "append":
                if not read:
                    errors.append(f"{rel}: append targets a file this run "
                                  "never read")
                    bad = True
            ops.append((idx, start, end, new, kind))

        if bad:
            continue

        # ---- overlap: two content edits may not touch the same bytes ------
        spans = sorted(((s, e, i) for i, s, e, _n, _k in ops if e > s),
                       key=lambda t: t[0])
        max_end = -1
        overlap = False
        for s, e, i in spans:
            if s < max_end:
                errors.append(f"{rel}: edits overlap (edit #{i + 1} touches a "
                              "region an earlier edit already changed); merge "
                              "them or drop one")
                overlap = True
                break
            max_end = max(max_end, e)
        if overlap:
            continue

        # ---- apply in reverse order so original offsets stay valid --------
        final = original
        for _i, start, end, new, _k in sorted(ops, key=lambda t: (-t[1], -t[0])):
            final = final[:start] + new + final[end:]

        # ---- syntax gate (reject only NEW syntax errors) -------------------
        err = _validate(path, final)
        if err and (original == "" or not _validate(path, original)):
            errors.append(f"{rel}: edit produced {err}; not applied")
            continue

        if final == original:
            errors.append(f"{rel}: edits change no bytes (net no-op)")
            continue

        before = original if exists else ""
        whole_file = exists and any(
            s == 0 and e == len(original) for (_i, s, e, _n, _k) in ops)
        diff = "".join(difflib.unified_diff(
            before.splitlines(keepends=True), final.splitlines(keepends=True),
            fromfile=f"a/{rel}", tofile=f"b/{rel}"))
        sha = hashlib.sha256(before.encode("utf-8")).hexdigest()
        note = (f"applied {len(group)} structured edit(s)"
                if exists else f"created with {len(final.splitlines())} line(s)")
        results.append(Result(True, rel, note, diff, before, final, sha,
                              whole_file=whole_file))
        files.append(rel)
        buffers[rel] = final
        styles[rel] = style
        legacy_edits.append(Edit(rel, "", final))

    return StructuredMaterialization(tuple(files), buffers, styles,
                                     tuple(results), tuple(legacy_edits),
                                     tuple(errors))
