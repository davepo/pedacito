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
            for e in g:
                if e.replace.strip():
                    continue
                if not e.search.strip():
                    can_clear = True         # explicit wipe of an existing file
                    continue
                span = locate(original, e.search, cfg)
                if span is not None and not original[span.stop:].strip():
                    can_clear = True         # search reaches end-of-file
        return can_create, can_clear

    classified = {rel: _classify(rel, g) for rel, g in groups.items()}
    creators = {rel for rel, (c, _cl) in classified.items() if c}
    removers = {rel for rel, (_c, cl) in classified.items() if cl}

    renames: dict[str, str] = {}             # cleared rel -> created rel
    if cfg is None or getattr(cfg, "allow_rename", True):
        used: set[str] = set()
        for old in sorted(removers):
            suffix = Path(old).suffix.lower()
            for new in sorted(creators):
                if new == old or new in used:
                    continue
                if Path(new).suffix.lower() != suffix:
                    continue
                renames[old] = new
                used.add(new)
                break

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
        err = _validate(path, text)
        if err:
            verb = "creation aborted" if original is None else "not written"
            return Result(False, rel, f"edit produced {err}; {verb}",
                          fuzzy=fuzzy)

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
                    suffix = path.suffix.lower()
                    if ph and suffix in (".html", ".htm"):
                        text = f"{ph}\n{body}\n"
                    elif ph and suffix in (".py",):
                        text = "".join(f"# {ln}\n" for ln in ph.splitlines()) \
                               + body + "\n"
                    elif ph and suffix in (".css", ".js", ".ts", ".scss"):
                        text = "".join(f"/* {ln} */\n" for ln in ph.splitlines()) \
                               + body + "\n"
                    else:
                        text = body + "\n"
                    moved_original = ""       # nothing existed at the target

        diff = "".join(difflib.unified_diff(
            moved_original.splitlines(keepends=True), text.splitlines(keepends=True),
            fromfile=f"a/{rel}", tofile=f"b/{rel}",
        ))
        sha = hashlib.sha256(moved_original.encode("utf-8")).hexdigest()
        if dry_run:
            return Result(True, rel, note + " (dry run)", diff, moved_original,
                          text, sha, fuzzy=fuzzy)

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
            if srcs:
                note += (" -- renamed from " + ", ".join(srcs) +
                        " (the moved contents are appended after the new "
                        "text; the old paths are removed)")
            results.append(_finish(rel, path, original, text, None, note,
                                   renames_from=srcs))
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
            text = "\n".join(e.replace for e in creators)
            if text and not text.endswith("\n"):
                text += "\n"
            path.parent.mkdir(parents=True, exist_ok=True)
            note = f"created with {len(text.splitlines())} line(s)"
            if srcs:
                note += (" -- renamed from " + ", ".join(srcs) +
                        " (the new file gets their current contents; the old "
                        "paths are removed)")
            results.append(_finish(rel, path, None, text, None, note,
                                   renames_from=srcs))
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
