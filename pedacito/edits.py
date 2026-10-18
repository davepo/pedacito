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
    # The `=======` separator must be a whole line of (mostly) equals signs.
    # Matching any run of five or more '=' used to swallow the '===' banner
    # inside a comment header that a SEARCH began with (`/* ====...`), which
    # truncated the SEARCH to `/* ` and dumped the rest into REPLACE -- the
    # "search block appears N times" failures on JS/CSS/HTML renames.
    # Anchoring to a full line leaves such banners in the SEARCH body.
    r"^[ \t]*={5,}[ \t]*$\n"
    # The replace body may be empty too: with no newline after the separator
    # the closer sits on the very next line (`>{5,}` after `^`). A negative
    # look-ahead keeps the group from swallowing the next block's header when
    # two blocks sit adjacent.
    r"(?P<replace>(?:(?!-{3}\s*FILE:).)*?)"
    r"^[ \t]*>{5,}\s*REPLACE",
    re.DOTALL | re.MULTILINE,
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
    # For a creation that was detected as a rename: the old path(s) whose real
    # contents were moved into this file (and which apply then removed). The
    # reviewer renders them as deleted so nothing about the move is hidden.
    renames_from: tuple[str, ...] = ()


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


def locate(haystack: str, needle: str, cfg=None,
           replacement: str | None = None) -> "Span | None":
    """Locate one search block in a file's text and return a Span, or None if
    it cannot be applied.

    `replacement` is the text to splice in -- normally the block's REPLACE
    body. It defaults to the needle itself ONLY as a safe no-op for callers
    that just want to find a span; apply_edits must pass the real replacement,
    because writing the matched text back over itself silently reports
    "applied 1 block(s)" while changing nothing on disk (this was the worst
    instance of the GUI's "nothing to review" bug: every plain edit parsed,
    matched, validated, and rewrote the file byte-for-byte unchanged).

    Tries, in order: an exact unique match; a whitespace-tolerant unique match
    (trailing whitespace and blank lines ignored), used only when
    `cfg.fuzzy_match` allows it; and finally a similarity match against every
    same-length window in the file, accepted only when the best window is both
    close enough (`cfg.fuzzy_min_ratio`) and clearly better than the runner-up.
    An ambiguous exact match is always a failure: editing the wrong occurrence
    silently is worse than asking the model for more context.
    """
    repl = needle if replacement is None else replacement
    span = _locate(haystack, needle)
    if span is not None:
        return Span(*span, repl)
    fuzzy_on = bool(cfg is not None and getattr(cfg, "fuzzy_match", True))
    if not fuzzy_on:
        return None
    min_ratio = float(getattr(cfg, "fuzzy_min_ratio", 0.85))
    max_len = int(getattr(cfg, "max_read_lines", 200)) * 200
    return _locate_fuzzy(haystack, needle, min_ratio, max_len, repl)


def _locate_fuzzy(haystack: str, needle: str, min_ratio: float,
                  max_needle: int, replacement: str | None = None) -> Span | None:
    """Best-effort anchor for a search block that does not match exactly.

    `replacement` is what the caller wants spliced in at the located span
    (see `locate`); it defaults to the needle, which makes the splice a
    no-op -- correct for pure "find the span" callers like rename
    classification, never for apply_edits, which passes the real REPLACE.

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
    repl = needle if replacement is None else replacement
    return Span(start, stop, repl, fuzzy=True)


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
_BLOCK_COMMENT_RE = re.compile(r"^/\*.*?\*/", re.DOTALL)


def _is_placeholder_only(text: str) -> bool:
    """True when `text` is nothing but a short comment -- the shape of a
    rename placeholder ("/* Contenido movido desde sentences.js */"). Used
    to refuse half-renames: a creation whose whole body is a placeholder and
    which matched no rename source would otherwise write a few-byte stub for
    a task that meant to move real content. A genuine new file either
    carries non-comment content or is clearly longer than a one-line note."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines or len(lines) > 3:
        return False
    joined = "\n".join(lines)
    if len(joined.encode("utf-8")) > 200:
        return False
    # C-style block comments (possibly several, or line-spans of one).
    rest = joined
    m = _BLOCK_COMMENT_RE.match(rest)
    while m:
        rest = rest[m.end():].strip()
        m = _BLOCK_COMMENT_RE.match(rest)
    if rest == "":
        return True
    # Hash / slash line comments only.
    return all(ln.startswith("#") or ln.startswith("//") for ln in lines)


def _placeholder_comment(ph: str, suffix: str) -> str:
    """Render a rename placeholder as a leading comment for a file of the
    given suffix, without double-wrapping one the model already wrote as a
    comment. Wrapping `/* ... */` in another `/* ... */` produced
    `/* /* ... */ */`, a syntax error that aborted the whole rename."""
    lines = ph.splitlines()
    if suffix in (".html", ".htm"):
        # HTML tolerates a bare line; keep the model's text verbatim.
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
    # Group blocks by path *before* merging, so rename detection sees each
    # SEARCH/REPLACE block on its own. merge_duplicate_files concatenates
    # bodies; when a model emits "create index.html" and "empty out
    # index.html" under the same header (the rename idiom), the merged
    # group's search is non-empty while its replace keeps the new content --
    # which previously made the whole path look like neither a creator nor a
    # remover, silently dropping the rename pair and leaving the old files
    # to fail as unmatched edits.
    groups: dict[str, list[Edit]] = {}
    for e in edits:
        groups.setdefault(e.file, []).append(e)

    root_resolved = Path(root).resolve()

    # ---- rename detection ---------------------------------------------------
    # The documented rename protocol is: an empty-SEARCH creation block for
    # the NEW path (a placeholder), plus a full-content SEARCH / empty REPLACE
    # "clear" block for each OLD path. Pedacito then moves the old file's real
    # bytes to the new path and removes it.
    #
    # Two structural facts make this tricky, and both were sources of silent
    # failures ("the review page stays blank, nothing gets renamed"):
    #
    #   1. Small models re-emit the same `--- FILE:` header per block --
    #      including when the target of a rename shares its name with a source
    #      (replace index.html with a new one, move the old ones aside). If
    #      duplicate headers are concatenated *before* classification, such a
    #      path looks like neither creator nor remover and the pair vanishes.
    #      Classification therefore happens per raw block below.
    #   2. A creation target exists on disk in the very common "overwrite +
    #      archive" pattern, so creators must NOT be required to be absent --
    #      they are identified by shape alone (empty SEARCH, non-empty
    #      REPLACE). Removers are identified by a clear block whose search
    #      actually reaches end-of-file.
    #
    # When a path is BOTH creator and remover (its own content archived
    # elsewhere), it is treated as pure overwrite-and-rename: its clear blocks
    # are dropped, because executing them would delete the freshly created
    # file. Pairing only ever links distinct paths, so no data can be lost.
    def _classify(rel: str, g: list[Edit]) -> tuple[bool, bool]:
        """Return (can_create, can_clear) for one path's raw blocks."""
        can_create = any(not e.search.strip() and e.replace.strip() for e in g)
        can_clear = False
        path = root_resolved / rel
        if path.is_file():
            original, _style = fileio.read(path, errors="replace")

            def _reaches_eof(search: str, span) -> bool:
                """True when the matched span extends to end-of-file.

                `locate()` deliberately excludes a trailing newline from its
                span (exact-match semantics: needles carry no final "\\n"), so
                a whole-file clear block whose REPLACE is empty would leave
                exactly one "\\n" in `original[span.stop:]` -- and a strict
                `.strip()` check used to reject it whenever the file ended
                WITH a newline while accepting it when it didn't. Rename
                pairing then silently dropped, which surfaced as the GUI's
                "nothing to review" on tasks as simple as renaming a file.
                Tolerate that one optional terminator instead of requiring
                byte-exact EOF.
                """
                tail = original[span.stop:]
                if not tail.strip():
                    return True
                if tail in ("\n", "\r\n") and search.strip():
                    return True
                return False

            # A path whose blocks also CREATE it (empty SEARCH + non-empty
            # REPLACE -- the placeholder half of a rename pair, or an
            # overwrite) must never be classified as a remover from a stray
            # empty-SEARCH/empty-REPLACE block: pairing it with itself would
            # move a zero-byte stub into the new path and delete the real
            # file. Such pairs are refused outright (see the renames loop).
            creates_here = any(not e.search.strip() and e.replace.strip()
                               for e in g)
            for e in g:
                if e.replace.strip():
                    continue
                if not e.search.strip():
                    if not creates_here:
                        can_clear = True     # explicit wipe of an existing file
                    continue
                span = locate(original, e.search, cfg)
                if span is None:
                    # Tolerate a truncated whole-file SEARCH: models that
                    # quote the old file's content from memory drop lines in
                    # the middle. If the quoted text matches contiguously at
                    # the START and again at the very END of the file (the
                    # tail reaching EOF), it still identifies this exact file
                    # -- accept it as a clear block so the rename pairs up and
                    # Pedacito moves the real bytes. A mid-file-only match
                    # stays rejected; that is an ordinary edit anchor, not a
                    # wipe. The SEARCH text itself is never used as content.
                    first_nl = e.search.find("\n")
                    head = (e.search if first_nl < 0 else e.search[:first_nl])
                    if original.startswith(head):
                        tail = e.search[first_nl + 1:] if first_nl >= 0 else ""
                        if tail.strip():
                            tspan = locate(original, tail, cfg)
                            if tspan is not None and _reaches_eof(tail, tspan):
                                can_clear = True
                                continue
                elif span is not None and not span.fuzzy:
                    # A clear block whose SEARCH matches uniquely anywhere in
                    # the file identifies it as the source of a move: the model
                    # anchors on any exact text it actually read -- the opening
                    # lines, a mid-file script tag, or the whole file -- since
                    # the SEARCH is only used to name the file, never as
                    # content. Fuzzy (similarity) matches stay rejected; a
                    # near-miss is too weak an identifier for a rename.
                    can_clear = True
        return can_create, can_clear

    classified = {rel: _classify(rel, g) for rel, g in groups.items()}
    creators = {rel for rel, (c, _cl) in classified.items() if c}
    removers = {rel for rel, (_c, cl) in classified.items() if cl}

    renames: dict[str, str] = {}             # cleared rel -> created rel
    if cfg is None or getattr(cfg, "allow_rename", True):
        used: set[str] = set()
        def _stem(rel: str) -> str:
            """Path without the final extension, for same-basename pairing."""
            return str(Path(rel).with_suffix("")).lower()

        def _pair_score(old: str, new: str) -> int:
            """Rank candidate pairings; -1 rejects a pair.

            A rename almost always stays in the SAME directory -- and when it
            doesn't, the old file's stem must survive in the new name (moving
            `notes.md` into `docs/notes.md` is legitimate; pairing
            root/index.html with subj-imp/index.html because both are named
            index.html would WIPE the home page to "rename" an unrelated
            placeholder). So cross-directory candidates are accepted only on
            an exact stem match, same-directory ones always."""
            same_dir = Path(old).parent == Path(new).parent
            if not same_dir and _stem(old) != _stem(new):
                return -1
            base = Path(old).name.lower()
            if base in Path(new).name.lower():
                return 3
            if same_dir:
                return 2
            return 1

        def _pair(old: str, cands: list[str]):
            scored = [(s, n) for n in cands if (s := _pair_score(old, n)) >= 0]
            if not scored:
                return None
            scored.sort(key=lambda t: (-t[0], t[1]))
            return scored[0][1]

        # Pass 1: same-directory pairs first -- they are unambiguous.
        rem_list, cre_list = sorted(removers), sorted(creators)

        def _source_is_real(old: str) -> bool:
            """True when the would-be rename source actually holds content
            worth moving. A file whose whole body is a rename placeholder
            comment (created by an earlier half-finished rename turn) carries
            no real bytes: pairing it as a SOURCE turns the protocol's own
            marker into a wipe of a live file -- exactly how root index.html
            lost its contents when a stale subjuntivo-imperativo-sentences.js
            stub matched the clearing block instead of sentences.js."""
            path = root_resolved / old
            if not path.is_file():
                return False
            try:
                text, _ = fileio.read(path, errors="replace")
            except OSError:
                return False
            return not _is_placeholder_only(text)

        def _placeholder_target(new: str) -> bool:
            """True when `new`'s create blocks in this reply are only a short
            rename placeholder. Such a target may pair ONLY with a source that
            has a NON-empty clearing block: an empty-SEARCH/empty-REPLACE wipe
            also matches any other existing path in the reply (the model's
            stray blank block against the file it just created), and pairing a
            placeholder with that wipe moves zero bytes while leaving the real
            file edited-but-not-renamed -- the exact silent half-rename from
            the DeepSeek test. The wipe-vs-placeholder case stays allowed for
            full-content targets (legit 'replace X with new content')."""
            g = groups.get(new, [])
            bodies = [e.replace.strip() for e in g
                      if not e.search.strip() and e.replace.strip()]
            return bool(bodies) and all(_is_placeholder_only(b) for b in bodies)

        def _explicit_wipe_only(old: str) -> bool:
            """True when every clearing block for `old` has an EMPTY search
            (a bare wipe) -- i.e. there is no content-quoting clear to anchor
            a placeholder pairing on."""
            g = groups.get(old, [])
            clears = [e for e in g if not e.replace.strip()]
            return bool(clears) and all(not e.search.strip() for e in clears)

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
        # Pass 2: remaining sources pair with remaining targets by stem, then
        # nearest common parent, then plain name order.
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

    # Now fold repeated headers into one group per path for application.
    # Rename sources lose their clear blocks: live, they would fail to match
    # after the move removed the file; in dry-run they would preview as an
    # empty stub next to the rename. A path that is also a creator keeps its
    # create blocks (see fact 3 above); a pure source drops out entirely.
    by_file: dict[str, list[Edit]] = {}
    for rel, g in groups.items():
        if rel in renames:
            keep = [e for e in g
                    if not (not e.replace.strip()
                            and (e.search.strip() or rel in creators))]
            if not keep:
                continue
            by_file[rel] = merge_duplicate_files(keep)
            continue
        by_file[rel] = merge_duplicate_files(g)

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
                note: str, fuzzy: bool = False,
                renames_from: tuple[str, ...] = ()) -> Result:
        """Validate, diff, back up, and write one finished file's content."""
        # Rename sources move their REAL bytes into the new path: the model's
        # creation REPLACE was only a placeholder (the system prompt says so --
        # retyping whole files wastes tokens and drops functions), so the
        # moved content wins. The placeholder survives as an HTML comment at
        # the top when it carries information the user should see.
        # `moved_original` is what the review page diffs against and pins by
        # sha for pure creations; for renames it is the first source's bytes.
        moved_original = original or ""
        if renames_from:
            texts = []
            for old in renames_from:
                try:
                    otext, _ = fileio.read(root_resolved / old, errors="replace")
                except OSError:
                    continue
                texts.append(otext)
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

        # Validate the FINAL text -- after a rename has folded the source's
        # real bytes in. Validating the placeholder alone used to reject
        # renames whose placeholder was a bare non-code comment (e.g. a plain
        # "moved from old.py" line), and skipped checking the moved content
        # entirely.
        err = _validate(path, text)
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
                          text, sha, fuzzy=fuzzy, renames_from=tuple(renames_from))

        if backup and original is not None:
            if backup_dir is not None:
                dest = backup_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                fileio.write(dest, original, style)
            else:
                fileio.write(path.with_suffix(path.suffix + ".bak"), original, style)

        # Renamed-in sources were backed up under their OLD path (below), so
        # the snapshot dir can restore a rename by moving files back. A new
        # file created from scratch has no bytes to preserve.
        if renames_from:
            for old in renames_from:
                opath = root_resolved / old
                try:
                    otext, ostyle = fileio.read(opath, errors="strict")
                except OSError:
                    continue
                if backup:
                    if backup_dir is not None:
                        bdest = backup_dir / old
                        bdest.parent.mkdir(parents=True, exist_ok=True)
                        fileio.write(bdest, otext, ostyle)
                    else:
                        fileio.write(opath.with_suffix(
                            opath.suffix + ".bak"), otext, ostyle)
                try:
                    opath.unlink()
                except FileNotFoundError:
                    pass

        fileio.write(path, text, style)
        return Result(True, rel, note, diff, moved_original, text, sha,
                      fuzzy=fuzzy, renames_from=tuple(renames_from))

    for rel, group in by_file.items():
        path = (Path(root) / rel).resolve()
        try:
            path.relative_to(root_resolved)
        except ValueError:
            results.append(Result(False, rel, "refusing to edit outside project root"))
            continue

        srcs = tuple(o for o, n in renames.items() if n == rel)
        # A rename whose SOURCE is the target itself (a self-pair that slipped
        # past pairing rules) would read the placeholder back as "moved
        # content" and then delete it -- a no-op at best, data loss at worst.
        srcs = tuple(o for o in srcs if o != rel)
        # A path can be BOTH a rename source and a creation target ("replace
        # index.html with a new page, archive the old ones"). Its clear blocks
        # were dropped above; only empty-SEARCH create blocks remain -- treat
        # that as an overwrite creation rather than falling into the match
        # loop below, which would fail against text the model never quoted.
        leftover_creators = [e for e in group if not e.search.strip()]
        if leftover_creators and len(leftover_creators) == len(group):
            original = None
            if path.is_file():
                original, _sty = fileio.read(path, errors="replace")
            text = "\n".join(e.replace for e in leftover_creators)
            if text and not text.endswith("\n"):
                text += "\n"
            path.parent.mkdir(parents=True, exist_ok=True)
            note = f"created with {len(text.splitlines())} line(s)"
            if original is not None:
                note = f"replaced contents ({len(original.splitlines())} -> {len(text.splitlines())} line(s))"
            srcs_now = ()
            if srcs:
                # Only claim a rename if the source still exists at apply
                # time; otherwise _finish's read fails silently and the note
                # would promise a move that never happens.
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

        exists = path.exists()
        if not exists:
            # New-file creation works on the *raw* blocks, before duplicate
            # headers were concatenated: a model that re-emits the same
            # `--- FILE:` header for each block leaves merged bodies glued
            # together ("\n" + new content), which would otherwise be written
            # with a stray leading blank line. Only empty-SEARCH blocks can
            # create; a group mixing them with real searches is a failed edit
            # against a file that isn't in the tree. Fuzzy matching does not
            # apply here: with no existing content there is nothing to be
            # similar to. The message must say which case it is, because
            # "file not found" alone invites the model to answer by renaming
            # files -- and Pedacito only honours that intent when the clear
            # blocks match (see rename detection above).
            raw = groups.get(rel, group)
            creators = [e for e in raw if not e.search.strip()]
            if not creators or len(creators) != len(raw):
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
            # One block per created file: a model that re-emits the same
            # header for several create blocks was writing successive parts
            # of ONE file, so concatenate their bodies in reply order (the
            # same rule merge_duplicate_files applies everywhere else).
            # Joining with "\n" never invents a blank line: parse_edits drops
            # exactly one trailing newline from every body, so a complete
            # block's terminator survives as the join character.
            text = "\n".join(e.replace for e in creators)
            if text and not text.endswith("\n"):
                text += "\n"
            path.parent.mkdir(parents=True, exist_ok=True)
            note = f"created with {len(text.splitlines())} line(s)"
            srcs_now = ()
            if srcs:
                # Same existence guard as the overwrite branch above.
                srcs_now = tuple(o for o in srcs if (root_resolved / o).is_file())
                gone = [o for o in srcs if o not in srcs_now]
                if srcs_now:
                    note += (" -- renamed from " + ", ".join(srcs_now) +
                            " (the new file gets their current contents; the "
                            "old paths are removed)")
                if gone:
                    note += (" [already absent, nothing to move: " +
                             ", ".join(gone) + "]")
            if _is_placeholder_only(text) and not srcs_now:
                # The rename protocol says: create the NEW path with a tiny
                # placeholder, then clear the OLD path with a full-content
                # SEARCH / empty REPLACE; Pedacito moves the real bytes. When
                # the clearing block is missing -- because the gather loop
                # died at the output cap before the model could emit it -- the
                # pairing above finds no source and this branch would write a
                # few-token stub where a 26k-char file belongs. A truncated
                # continuation reply can look EXACTLY like a placeholder
                # (one short line), so refuse rather than half-rename: the
                # old file stays untouched and the next turn re-asks.
                results.append(Result(
                    False, rel,
                    "refusing to create this file: its content is only a "
                    "placeholder comment and no rename source matched (no "
                    "clearing block for an old path survived). If you are "
                    "renaming a file, emit ONLY the empty-SEARCH creation "
                    "block for the new path plus a block for the old path "
                    "whose SEARCH is its entire content and whose REPLACE is "
                    "empty -- Pedacito moves the real bytes; never copy them. "
                    "If you mean to write this file's content yourself, put "
                    "that content in REPLACE."))
                continue
            results.append(_finish(rel, path, None, text, None, note,
                                   renames_from=srcs_now))
            continue

        # LF-normalised for matching; style remembered so the file is written
        # back exactly as it was found.
        original, style = fileio.read(path, errors="strict")
        text = original
        failed = False
        any_fuzzy = False
        fuzzy_blocks: list[int] = []
        for k, e in enumerate(group, 1):
            span = locate(text, e.search, cfg, replacement=e.replace)
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
