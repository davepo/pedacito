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

The FILE header may carry an anchor instead of a SEARCH body:

    --- FILE: path/to/file.py LINES 40-60
    --- FILE: path/to/file.py SYMBOL Server.start

For an anchored block the REPLACE body overwrites that 1-based inclusive line
range (LINES) or that definition (SYMBOL, resolved to a line range by the
caller -- the agent, which owns the index), and SEARCH is ignored.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from . import fileio
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


def _is_anchored(e: "Edit") -> bool:
    """True for a LINES/SYMBOL-anchored block: a line-range replacement, never
    a creation, a rename half, or a no-op."""
    return bool(e.lines or e.symbol)

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
    # `SYMBOL name` anchor keeps the name here until the caller resolves it
    # into `lines` (apply_edits has no index).
    lines: tuple[int, int] | None = None
    symbol: str = ""


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


def parse_edits(text: str, discards: list | None = None) -> list[Edit]:
    """Extract every search/replace block from a model's reply text.

    Normalises line endings and strips the fences wrapping the blocks. A block
    whose SEARCH and REPLACE are identical changes nothing and is dropped; its
    file is appended to `discards` when given, so callers never count a null
    edit as work."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _strip_wrapping_fences(text)
    # A blanked fence line can leave a header glued to the line above it.
    cleaned = _HEADER_START_RE.sub(r"\n\1", cleaned)
    edits = []
    for m in BLOCK_RE.finditer(cleaned):
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
        edits.append(Edit(fname, search, replace,
                          lines=lines_anchor, symbol=symbol_anchor))
    return edits


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
    (trailing whitespace and blank lines ignored); and, when `cfg.fuzzy_match`
    allows, a similarity match against every same-length window, accepted only
    if the best window clears `cfg.fuzzy_min_ratio` and clearly beats the
    runner-up. An ambiguous match is always a failure: editing the wrong
    occurrence silently is worse than asking the model for more context.
    """
    repl = needle if replacement is None else replacement
    span = _locate(haystack, needle)
    if span is not None:
        return Span(*span, repl)
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


def _locate(haystack: str, needle: str) -> tuple[int, int] | None:
    """Find the unique span of `needle` in `haystack`: exact match first, then
    a retry ignoring trailing whitespace per line. Ambiguous matches fail."""
    if not needle.strip():
        return None
    n = haystack.count(needle)
    if n == 1:
        i = haystack.index(needle)
        return i, i + len(needle)
    if n > 1:
        return None

    hay, need = _rstripped_lines(haystack), _rstripped_lines(needle)
    if not need:
        return None
    starts = [i for i in range(len(hay) - len(need) + 1) if hay[i : i + len(need)] == need]
    if len(starts) != 1:
        return None
    s = starts[0]
    lines = haystack.splitlines(keepends=True)
    seg = lines[s : s + len(need)]
    start = sum(len(x) for x in lines[:s])
    # Exclude the final line terminator: exact-match needles carry none.
    tail = len(seg[-1]) - len(seg[-1].rstrip("\r\n"))
    end = start + sum(len(x) for x in seg) - tail
    return start, end


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
    `root` are refused.

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
        allowed = {(p or "").strip().lower().lstrip("./") for p in allowlist}
        for rel in list(groups):
            key = (rel or "").strip().lower().lstrip("./")
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
            old_posix = Path(old).as_posix().lower().lstrip("./")
            new_posix = Path(new).as_posix().lower().lstrip("./")
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

        # Pass 1: same-directory pairs first -- they are unambiguous.
        for old in rem_list:
            if not _source_is_real(old):
                continue
            suffix = Path(old).suffix.lower()
            cand = [n for n in cre_list if n != old and n not in used
                    and Path(n).suffix.lower() == suffix
                    and Path(n).parent == Path(old).parent
                    and not (_placeholder_target(n) and _explicit_wipe_only(old))]
            if cand:
                new = min(cand)
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

    def _finish(rel: str, path: Path, original: str | None, text: str, style,
                note: str, fuzzy: bool = False,
                renames_from: tuple[str, ...] = ()) -> Result:
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
                          text, sha, fuzzy=fuzzy, renames_from=renames_from)

        def _backup(src: Path, old_rel: str) -> None:
            if backup_dir is not None:
                fileio.backup_copy(src, backup_dir / old_rel)
            else:
                fileio.backup_copy(src, src.with_suffix(src.suffix + ".bak"),
                                   keep_existing=False)

        if backup and original is not None:
            _backup(path, rel)

        # Sources are backed up under their OLD path so a snapshot restores a
        # rename by moving files back.
        for old in renames_from:
            opath = root_resolved / old
            try:
                if backup:
                    _backup(opath, old)
                opath.unlink()
            except FileNotFoundError:
                pass

        path.parent.mkdir(parents=True, exist_ok=True)
        fileio.write(path, text, style)
        return Result(True, rel, note, diff, moved_original, text, sha,
                      fuzzy=fuzzy, renames_from=renames_from)

    for rel, group in by_file.items():
        path = _resolve_inside(rel)
        if path is None:
            results.append(Result(False, rel, "refusing to edit outside project root"))
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
            if path.is_file():
                original, _sty = fileio.read(path, errors="replace")
            text = _creation_text(group)
            note = f"created with {len(text.splitlines())} line(s)"
            if original is not None:
                note = f"replaced contents ({len(original.splitlines())} -> {len(text.splitlines())} line(s))"
            # Claim only sources that still exist at apply time.
            srcs_now = tuple(o for o in srcs if (root_resolved / o).is_file())
            gone = [o for o in srcs if o not in srcs_now]
            if srcs_now:
                note += (" -- renamed from " + ", ".join(srcs_now) +
                         " (the moved contents are appended after the new "
                         "text; the old paths are removed)")
            if gone:
                note += (" [already absent, nothing to move: " +
                         ", ".join(gone) + "]")
            results.append(_finish(rel, path, original, text, None, note,
                                   renames_from=srcs_now))
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
        # Anchored blocks resolve to (a, b) here for failure messages, and
        # are rewritten to exact-text needles taken from the ORIGINAL file,
        # so an earlier block in the same group cannot shift their lines and
        # an externally drifted file still matches via locate()/fuzzy.
        anchored: dict[int, tuple[int, int]] = {}
        for k, e in enumerate(group, 1):
            if _is_anchored(e):
                if not e.lines:
                    results.append(Result(
                        False, rel,
                        f"block {k}: SYMBOL anchor '{e.symbol}' could not be "
                        "resolved to a definition in this file; `read` the "
                        "symbol and emit the block with LINES <start>-<end> "
                        "instead"))
                    failed = True
                    break
                a, b = e.lines
                lspan = _line_span(original, a, b)
                if lspan is None:
                    results.append(Result(
                        False, rel,
                        f"block {k}: line range {a}-{b} is out of bounds -- "
                        f"this file has {len(original.splitlines())} line(s); "
                        "`read` it again and emit current line numbers"))
                    failed = True
                    break
                anchored[k] = (a, b)
                needle = original[lspan[0]:lspan[1]]
                if not needle.strip():
                    # A blank-lines-only needle would fall into the
                    # empty-SEARCH whole-text branch below; refuse instead.
                    results.append(Result(
                        False, rel,
                        f"block {k}: line range {a}-{b} is only blank lines -- "
                        "widen the anchored range to include real code so it "
                        "can be located"))
                    failed = True
                    break
                repl = e.replace
                if repl and needle.endswith("\n") and not repl.endswith("\n"):
                    repl += "\n"
                e = Edit(e.file, needle, repl)
            if not e.search.strip():
                # An empty SEARCH mixed in with real edits replaces the WHOLE
                # text that later blocks anchor against.
                text = e.replace if not e.replace or e.replace.endswith("\n") \
                    else e.replace + "\n"
                continue
            span = locate(text, e.search, cfg, replacement=e.replace)
            if span is None:
                n = text.count(e.search)
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
            text = text[:span.start] + span.replacement + text[span.stop:]
            if span.fuzzy:
                fuzzy_blocks.append(k)

        if failed:
            continue

        note = f"applied {len(group)} block(s)"
        if fuzzy_blocks:
            note += (" [matched approximately on block"
                     + ("s " if len(fuzzy_blocks) > 1 else " ")
                     + ", ".join(str(b) for b in fuzzy_blocks)
                     + "; review the diff carefully]")
        results.append(_finish(rel, path, original, text, style, note,
                               fuzzy=bool(fuzzy_blocks)))

    return results
