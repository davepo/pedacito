"""The agent's tool surface: read, grep, callers, outline, sessions, diff, write/append, run, screenshot, browser, done.

Usage
-----
    from .tools import STEP_SCHEMA, TOOL_HELP, dispatch

    step = client.chat_json(messages, STEP_SCHEMA, max_tokens=700)
    observation = dispatch(index, cfg, step)

Deliberately a small, fixed set of tools. Local models degrade fast as the
tool surface grows -- four to six is about the ceiling before a small model
starts calling the wrong one or inventing arguments. The navigation tools
(read, grep, callers, outline, sessions, diff, done) are the core surface.
`sessions` is read-only history and is how a fresh run sees prior tasks;
`diff` is read-only too -- it compares the working tree against Pedacito's
own pre-write snapshots (`allow_diff = false` drops it).
Everything destructive or machine-visible is behind an explicit gate:
`write`/`append`/`move`/`delete` need `cfg.allow_write`, `run` needs
`cfg.allow_shell`, `screenshot` needs `cfg.allow_screenshots`, and `browser`
needs `cfg.allow_browser`; both launch tools also need `allow_shell` when
given a command. A default run therefore exposes only
the navigation surface; a user who opts in accepts a larger surface because
they asked for those capabilities. Every tool returns plain text with line
numbers attached, since the model needs those numbers to write search/replace
edits later.

The navigation tools are deliberately forgiving, because small models make
argument mistakes that must not cost a whole lookup: `read` accepts a symbol
name instead of a line range, `grep` falls back to a literal search when the
"regex" does not compile, retries case-insensitively when a case-sensitive
search finds nothing, accepts a directory or glob as its scope, shows a line
of context around each hit when there are few hits, and suggests similarly
named symbols when there are none. `outline` accepts a directory and renders
its file cards. When `semantic_search` is on, a grep with no textual hits
queries the server's embeddings endpoint instead of returning an empty
observation. A step that would otherwise have failed can often still return
useful source.

The mutating actions are off by default because a mid-gather change skips the
edit path's syntax validation and review page entirely. They exist for the
case where an existing file must be replaced, grown, moved, or removed
wholesale -- which a 9B model often cannot express as a search/replace block.
`append` makes the chunked-file protocol explicit instead of asking the model
to re-emit a tail and hope the writer notices. `run` feeds test/linter output
back to the model but executes model-chosen commands, so it is a separate,
equally explicit trust decision.

STEP_SCHEMA is the JSON schema used with LM Studio's grammar-constrained
`response_format`, so the model's tool choice always parses. See agent.py
for how it is used in the gather loop.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shlex
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from . import browser, fileio, proc, sessions, vision

TOOL_HELP = """\
read     - read exact lines of a file.       args: file, start, end
           (or set symbol and leave file empty: read that definition)
grep     - regex or literal search. args: pattern, and optional file/dir/glob
           (with semantic_search on, falls back to search by meaning)
callers  - who calls a symbol, with the call sites. args: symbol
outline  - list every unit in one file, or the file cards in one directory.
           args: file (a path, a symbol, or a directory/glob)
sessions - list/search what prior runs did.  args: pattern, optional file
diff     - what changed since Pedacito's last snapshot. args: optional file
write    - overwrite a whole file with text. args: file, content
append   - add text to the END of a file.    args: file, content
move     - rename/move one file.             args: file (old), to (new)
delete   - delete one file.                  args: file
run      - run one command, no shell pipes.  args: command
           (a quiet GUI/server is stopped after shell_idle_seconds)
screenshot - launch app, wait, capture screen; returns text analysis.
           args: command, start (seconds to wait), optional file (old shot)
           (needs allow_screenshots; launching a command also needs allow_shell)
browser  - open a web page in headless Chromium, interact, wait, capture it.
           args: command (URL, or a command that prints a localhost URL),
           pattern (optional "press Enter; click #start; wait 2"),
           start (seconds to wait), optional file (old shot to compare)
           (needs allow_browser; launching a command also needs allow_shell)
           (a desktop screenshot cannot see a page that was never opened)
           (write/append/move/delete need allow_write; run needs allow_shell)
done     - stop gathering; you have enough.  args: none"""

# Field order is load-bearing. Grammar-constrained decoding emits keys in schema
# order, so the action and its arguments come first: if the response is cut off
# by the token limit, the only thing lost is the explanation, and the step is
# still usable. With `reasoning` first, one rambling sentence truncates the
# whole object and the step is unrecoverable.
STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["read", "grep", "callers", "outline", "sessions",
                     "diff", "write", "append", "move", "delete", "run",
                     "screenshot", "browser", "done"],
        },
        "file": {"type": "string",
                 "description": "For read/write/append/move/delete. For "
                                "outline, a file OR a directory/glob. For "
                                "grep, optionally a file, directory, or glob. "
                                "For diff, optionally one file. For screenshot/"
                                "browser, an older screenshot to compare. Empty "
                                "otherwise.",
                 "maxLength": 200},
        "start": {"type": "integer",
                  "description": "For read, or screenshot/browser wait seconds. "
                                 "0 otherwise."},
        "end": {"type": "integer", "description": "For read. 0 otherwise."},
        "pattern": {"type": "string",
                    "description": "For grep: regex, or plain text if it is not "
                                   "valid regex. For browser: optional actions "
                                   "like 'press Enter; click #start; wait 2'. "
                                   "Empty otherwise.",
                    "maxLength": 200},
        "symbol": {"type": "string",
                   "description": "For callers; or for read when file is empty "
                                  "(reads that definition). Empty otherwise.",
                   "maxLength": 200},
        "to": {"type": "string",
               "description": "For move: the new path. Empty otherwise.",
               "maxLength": 200},
        "command": {"type": "string",
                    "description": "For run/screenshot: executable and "
                                   "arguments, e.g. 'pytest -q'. For browser: "
                                   "a URL or a command that prints one. No "
                                   "pipes or redirection. Empty otherwise.",
                    "maxLength": 300},
        "content": {"type": "string",
                    "description": "For write/append: at most ~120 lines of file "
                                   "text (large files are built chunk by chunk over "
                                   "several steps). Empty otherwise.",
                    "maxLength": 6000},
        "reasoning": {
            "type": "string",
            "description": "Under 15 words: what you still need. Not a plan.",
            "maxLength": 160,
        },
    },
    # Flat + all-required beats nested oneOf/anyOf for small models: the grammar
    # stays simple and the model never has to pick a branch.
    # NOTE: `content`'s maxLength MUST stay below llm._BIG_FIELD_MIN (8000).
    # That threshold is what marks a schema "answer-style" (cut mid-content =>
    # partial, must escalate) vs "step-style" (args all precede `reasoning`, so
    # a salvaged action is complete and accepted on the first try). Raising it
    # past 8000 silently disables step-truncation salvage; file-size limits are
    # enforced by max_write_lines in dispatch()/tool_write(), not here.
    "required": ["action", "file", "start", "end", "pattern", "symbol",
                 "to", "command", "content", "reasoning"],
    "additionalProperties": False,
}

# The action list is kept in one place so the schema, the gate in `dispatch`,
# and the prompt help cannot drift apart.
ACTIONS = ("read", "grep", "callers", "outline", "sessions", "diff", "write",
           "append", "move", "delete", "run", "screenshot", "browser", "done")
WRITE_ACTION = "write"
APPEND_ACTION = "append"
MOVE_ACTION = "move"
DELETE_ACTION = "delete"
RUN_ACTION = "run"
SCREENSHOT_ACTION = "screenshot"
BROWSER_ACTION = "browser"
DIFF_ACTION = "diff"
WRITE_ACTIONS = (WRITE_ACTION, APPEND_ACTION)
MUTATING_ACTIONS = WRITE_ACTIONS + (MOVE_ACTION, DELETE_ACTION)


def schema_for(cfg) -> dict:
    """Return STEP_SCHEMA narrowed to the actions this run actually allows.

    Grammar-constrained decoding sees the enum, so leaving disabled actions in
    it invites a small model to call them and burn a step on an ERROR. Gating
    the schema itself (not just dispatch) keeps the default surface at the
    five navigation actions (`diff` adds a sixth, read-only one unless
    allow_diff = false). Fields only used by a disabled action are dropped
    too, so the model is not asked for `content`/`to`/`command` it cannot use.
    """
    import copy

    actions = ["read", "grep", "callers", "outline", "sessions"]
    if getattr(cfg, "allow_diff", True):
        actions.append("diff")
    if getattr(cfg, "allow_write", False):
        actions += ["write", "append", "move", "delete"]
    if getattr(cfg, "allow_shell", False):
        actions.append("run")
    if getattr(cfg, "allow_screenshots", False):
        actions.append("screenshot")
    if getattr(cfg, "allow_browser", False):
        actions.append("browser")
    actions.append("done")

    schema = copy.deepcopy(STEP_SCHEMA)
    schema["properties"]["action"]["enum"] = actions
    required = ["action", "file", "start", "end", "pattern", "symbol",
                "reasoning"]
    drop = set()
    if not getattr(cfg, "allow_write", False):
        drop |= {"content", "to"}
    if not (getattr(cfg, "allow_shell", False)
            or getattr(cfg, "allow_screenshots", False)
            or getattr(cfg, "allow_browser", False)):
        drop.add("command")
    for key in drop:
        schema["properties"].pop(key, None)
    for key in ("content", "to", "command"):
        if key not in drop:
            required.append(key)
    if getattr(cfg, "compact_step_schema", False):
        # EXPERIMENTAL token-saver: only `action` (and the trailing
        # `reasoning`) stay required, so grammar-constrained decoding may
        # omit the empty fields a step does not use -- worth ~20-40 output
        # tokens per step at ~25 tok/s. Field ORDER is untouched, so a
        # truncated reply still salvages to a complete action. Off by
        # default: all-required keeps the grammar maximally simple for
        # small models; measure it on your server before enabling.
        required = ["action", "reasoning"]
    # Keep the load-bearing field order: action+args, then content, then
    # reasoning last (a cut-off reply must still yield a usable action).
    ordered = ["action", "file", "start", "end", "pattern", "symbol",
               "to", "command", "content", "reasoning"]
    schema["properties"] = {k: schema["properties"][k] for k in ordered
                            if k in schema["properties"]}
    schema["required"] = [k for k in ordered if k in required]
    return schema


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------
def _as_text(value) -> str:
    """Best-effort string coercion for model-supplied step fields.

    The JSON schema says these are strings, but a salvaged or repaired step
    can still carry null/number/list values. One bad field must not raise
    out of `dispatch` and kill the whole run, so tools normalise at the edge.
    """
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _as_int(value, default: int = 0) -> int:
    """Coerce a model-supplied integer field without raising.

    JSON-schema decoding normally guarantees an integer, but plain-JSON
    prompting and truncated-JSON salvage do not: a model can send "12", 12.0,
    null, or a prose fragment. Out-of-range line numbers are harmless (the
    read tools clamp them); a TypeError/ValueError here would crash the step.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            num = float(value)
            return int(num) if num == num and abs(num) != float("inf") else default
        except (TypeError, ValueError, OverflowError):
            return default


def _root_rel(index, name) -> "str | None":
    """Resolve a model-supplied path to a project-relative path, or None.

    This is the single gate for paths that the tools will touch. It accepts
    absolute paths only when they point inside the project, collapses `..`,
    and follows symlinks so a link inside the project cannot read/write an
    outside file. Everything outside the root is rejected. Returning a
    normalised relative path (instead of the raw input) also keeps backup
    destinations and index keys sane when a model sends an absolute path.
    """
    raw = fileio.normalise_rel(_as_text(name))
    if not raw:
        return None
    root = Path(index.root).resolve()
    try:
        cand = Path(raw)
        if not cand.is_absolute():
            cand = root / cand
        return cand.resolve().relative_to(root).as_posix()
    except (OSError, ValueError):
        return None


def _numbered(lines: list[str], start: int) -> str:
    """Render a list of lines with right-aligned line numbers, starting at
    `start`, in the "NNN | text" format the model reads and later echoes
    back (minus the prefix) in search/replace edits."""
    width = len(str(start + len(lines) - 1))
    return "\n".join(f"{i:>{width}} | {ln}" for i, ln in enumerate(lines, start))


GREP_WARNING = ("[grep output is trimmed and may be truncated. It shows you WHERE "
                "things are.\n Never copy these lines into an edit -- `read` the "
                "region first.]")


def _grep_result(scope_note: str, hits: list[str]) -> str:
    """Assemble a grep reply: optional scope correction first, then the
    standing warning, then the hits."""
    head = (scope_note + "\n") if scope_note else ""
    return head + GREP_WARNING + "\n" + "\n".join(hits)


def _enclosing_chunk(index, rel: str, line: int):
    """Innermost indexed unit containing `line` in `rel`, or None.

    Used to prefix grep hits with the symbol they live in -- a small model
    navigating a 2,000-line file can act on `[Server.start]` immediately,
    where a bare line number leaves it guessing which function to outline.
    Picks the smallest matching span so a method wins over its class."""
    try:
        units = index.in_file(rel)
    except Exception:
        return None
    best = None
    for c in units:
        if c.kind == "module" or not (c.start <= line <= c.end):
            continue
        if best is None or c.nlines < best.nlines:
            best = c
    return best


def _symbol_matches(index, symbol: str, rel: "str | None" = None) -> list:
    """Definitions matching `symbol`, optionally restricted to one file."""
    try:
        matches = index.find_symbol(_as_text(symbol).strip())
    except Exception:
        return []
    if rel:
        matches = [c for c in matches if c.file == rel]
    return matches


def _file_hint(index, name: str) -> str:
    """Compact 'file not found' message: a few real paths plus close matches.

    The old message joined every indexed path, which on a real project is
    hundreds of paths -- a token sink that buries the correction. A small
    model only needs one or two plausible names and a hint at the next move."""
    name = _as_text(name)
    files = index.files
    if not files:
        return f"ERROR: '{name}' is not in the index (the index is empty)."
    import difflib
    close = difflib.get_close_matches(
        fileio.normalise_rel(name), files, n=3, cutoff=0.55)
    if not close:
        base = Path(fileio.normalise_rel(name)).name
        close = [f for f in files if Path(f).name == base][:3]
    bits = [f"ERROR: '{name}' is not in the index."]
    if close:
        bits.append("Did you mean: " + ", ".join(close) + "?")
    shown = ", ".join(files[:12])
    more = f" ... (+{len(files) - 12} more)" if len(files) > 12 else ""
    bits.append(f"{len(files)} indexed files include: {shown}{more}.")
    bits.append("Use `grep` to locate a file you cannot name exactly.")
    return " ".join(bits)


def _symbol_hint(index, name: str) -> str:
    """Compact 'symbol not found' message with close-name suggestions."""
    import difflib
    name = _as_text(name)
    try:
        names = sorted({c.qualname for c in index.chunks
                        if c.qualname not in ("module", "__trailing__")})
    except Exception:
        names = []
    close = difflib.get_close_matches(name, names, n=5, cutoff=0.6)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    return (f"ERROR: no definition of '{name}' was found.{hint} "
            f"Use `grep` with pattern '{name}' to find references.")


def _grep_scope(index, scope: str) -> tuple[list[str], str]:
    """Resolve a grep scope to (files, note).

    `scope` may be one file, a directory (`src/`, `src/pkg`), or a glob
    (`*.py`, `tests/*.py`). Small models routinely put a directory or a
    pattern in the `file` field; treating that as "not in the index, search
    everything" wastes the lookup. Only an unresolvable scope falls back to
    the whole index, with a note saying so."""
    files = index.files
    s = fileio.normalise_rel(_as_text(scope)).strip()
    if not s or s in (".", "./"):
        return files, ""
    one = index.resolve_file(s)
    if one:
        return [one], ""
    low = s.lower().lstrip("./").rstrip("/")
    if any(ch in low for ch in "*?["):
        matches = [f for f in files
                   if fnmatch.fnmatch(f.lower(), low)
                   or fnmatch.fnmatch(Path(f).name.lower(), low)]
    else:
        matches = [f for f in files
                   if f.lower().lstrip("./") == low
                   or f.lower().lstrip("./").startswith(low + "/")]
    if matches:
        return matches, ""
    return files, (f"[note: '{scope}' matched no indexed file or directory; "
                   f"searched all {len(files)} indexed files instead]")


def _hit_text(index, rel: str, i: int, line: str) -> str:
    """One formatted grep hit: `rel:lineno: [symbol] text` with the line
    trimmed to 200 chars. Indentation is preserved: a stripped line looks
    editable and is not, which is how a model ends up writing SEARCH blocks
    that can never match."""
    shown = line[:200].rstrip()
    cut = " ...[line truncated]" if len(line) > 200 else ""
    unit = _enclosing_chunk(index, rel, i)
    where = (f" [{unit.qualname}]" if unit is not None
             and unit.qualname != "module" else "")
    return f"{rel}:{i}:{where} {shown}{cut}"


# A grep with at most this many real hits gets one line of context around
# each: with so few matches the model is almost certainly looking at THE
# place, and seeing the neighbouring line saves a `read` round trip. Dense
# results skip it -- context there is mostly noise and doubles the tokens.
SPARSE_CONTEXT_MAX = 5


def _context_line(index, rel: str, n: int) -> "str | None":
    """Render line `n` of `rel` as a GNU-grep-style context line
    (`file-N- text`), or None when it does not exist or is blank (a blank
    neighbour carries no shape worth spending tokens on)."""
    try:
        lines = fileio.read_lines_cached(index.root / rel)
    except OSError:
        return None
    if n < 1 or n > len(lines):
        return None
    line = lines[n - 1]
    if not line.strip():
        return None
    shown = line[:200].rstrip()
    cut = " ...[line truncated]" if len(line) > 200 else ""
    return f"{rel}-{n}- {shown}{cut}"


def _with_context(index, entries) -> list[str]:
    """Flatten grep entries to lines, inserting one context line before and
    after each hit. Entries already shown (as another hit's context, or as a
    hit itself) are never repeated, so consecutive hits stay consecutive."""
    emitted = {(rel, n) for rel, n, _ in entries if n is not None}
    out: list[str] = []
    for rel, n, text in entries:
        if n is None:
            out.append(text)                    # a cap/stop note
            continue
        if n - 1 >= 1 and (rel, n - 1) not in emitted:
            cl = _context_line(index, rel, n - 1)
            if cl is not None:
                out.append(cl)
                emitted.add((rel, n - 1))
        out.append(text)
        emitted.add((rel, n))
        if (rel, n + 1) not in emitted:
            cl = _context_line(index, rel, n + 1)
            if cl is not None:
                out.append(cl)
                emitted.add((rel, n + 1))
    return out


def _scan_grep(index, files: list[str], rx: re.Pattern, max_hits: int,
               rx_ci: "re.Pattern | None" = None) -> tuple[list, list]:
    """Search `files` with `rx`, optionally collecting case-insensitive hits
    for `rx_ci` in the SAME pass.

    Returns (entries, ci_entries): lists of (rel, lineno, text) tuples, with
    lineno None for cap/stop notes. Single-pass is the point -- the
    case-insensitive fallback used to re-read every file from disk, doubling
    the I/O of exactly the steps that found nothing. ci_entries is only
    meaningful when `entries` has no real hits (the case-sensitive result
    always wins), and collection for it stops at its own cap.
    """
    entries: list[tuple] = []
    ci_entries: list[tuple] = []
    per_file_cap = max(5, max_hits // 4)
    ci_done = False                  # ci list hit its global cap
    for rel in files:
        path = index.root / rel
        if not path.is_file():
            continue
        try:
            lines = fileio.read_lines_cached(path)
        except OSError:
            continue                      # unreadable: not a fatal grep
        file_hits = 0
        ci_file_hits = 0
        ci_note_added = False
        for i, line in enumerate(lines, 1):
            if rx.search(line):
                if file_hits >= per_file_cap:
                    entries.append((rel, None,
                                    f"{rel}: [more matches in this file; "
                                    f"grep it directly with `file` set]"))
                    # Breaking the file's scan is safe for the ci list too:
                    # a per-file cap means real cs hits exist, and the ci
                    # list is discarded whenever any cs hit was found.
                    break
                entries.append((rel, i, _hit_text(index, rel, i, line)))
                file_hits += 1
                if len(entries) >= max_hits:
                    entries.append((rel, None,
                                    f"[stopped at {max_hits} hits; narrow "
                                    f"the pattern]"))
                    return entries, ci_entries
            elif rx_ci is not None and not ci_done and rx_ci.search(line):
                if ci_file_hits >= per_file_cap:
                    if not ci_note_added:
                        ci_entries.append((rel, None,
                                           f"{rel}: [more matches in this "
                                           f"file; grep it directly with "
                                           f"`file` set]"))
                        ci_note_added = True
                    continue
                ci_entries.append((rel, i, _hit_text(index, rel, i, line)))
                ci_file_hits += 1
                if len(ci_entries) >= max_hits:
                    ci_entries.append((rel, None,
                                       f"[stopped at {max_hits} hits; narrow "
                                       f"the pattern]"))
                    ci_done = True
    return entries, ci_entries


def _symbol_suggestions(index, pattern: str) -> str:
    """A free 'did you mean this symbol?' when a grep found nothing at all.

    The index already knows every defined name; a pattern that misses as text
    often IS a near-miss of a symbol name (typo, wrong case boundary, guessed
    spelling). Substring matches come first (grep 'thresh' -> 'leech_threshold'),
    then difflib's close matches. Returns '' when nothing is close, so a
    natural-language miss stays a clean 'No matches'."""
    p = (pattern or "").strip()
    if len(p) < 3:
        return ""
    import difflib
    try:
        names = sorted({c.qualname for c in index.chunks
                        if c.qualname not in ("module", "__trailing__")})
    except Exception:
        return ""
    if not names:
        return ""
    low = p.lower()
    subs = [n for n in names if low in n.lower()][:5]
    close = difflib.get_close_matches(p, names, n=5, cutoff=0.6)
    picks = list(dict.fromkeys(subs + close))[:5]
    if not picks:
        return ""
    return ("[no text matches, but the index has similar symbol names: "
            + ", ".join(picks)
            + ". `read` one with `symbol` set, or see its uses with "
              "`callers`.]")


# --------------------------------------------------------------------------
# Tool implementations
# --------------------------------------------------------------------------
def _fresh_index(index, cfg, rel: str):
    """Re-chunk a file that exists on disk but is not in the index yet.

    A brand-new file has no chunks, so `read` and `outline` would both fail on
    it ("not in the index") even though its contents are exactly what the model
    needs to see before extending it -- the common case when resuming work on a
    project whose index predates the file. Returns None if nothing could be
    indexed."""
    rel = _root_rel(index, rel)
    if not rel:
        return None
    try:
        from . import index as index_mod
        return index_mod.reindex_file(index, rel, cfg, save=False)
    except Exception:
        return None


def adopt_if_on_disk(index, cfg, file: str) -> "str | None":
    """If `file` names something that exists on disk but is missing from the
    index, fold it in and return its indexed path; otherwise return None.

    Shared by `read` and `outline`: a scaffolded or hand-added file should never
    cost the model a lookup step just because indexing ran before it existed.
    A path outside the project root is never adopted (that would make `read`
    a filesystem-escape)."""
    cand = _root_rel(index, file)
    if not cand:
        return None
    if not (Path(index.root) / cand).is_file():
        return None
    fresh = _fresh_index(index, cfg, cand)
    if fresh is None:
        return None
    index.__dict__.update(fresh.__dict__)
    return index.resolve_file(file) or cand


def _definition_list(symbol: str, matches: list) -> str:
    """Render a compact list of definition locations when a symbol is
    ambiguous (the same name in several files/classes)."""
    out = [f"'{symbol}' has {len(matches)} definitions; choose one with "
           f"`read` file+start+end:"]
    for c in matches[:20]:
        out.append(f"  {c.file}:{c.start}-{c.end}  {c.signature or c.qualname}")
    if len(matches) > 20:
        out.append(f"  (+{len(matches) - 20} more)")
    return "\n".join(out)


def tool_read(index, cfg, file: str, start: int, end: int,
              symbol: str = "") -> str:
    """Return exact, line-numbered source text for `file` between `start`
    and `end` (1-based, inclusive). This is the only tool whose output is
    safe to quote verbatim in a search/replace edit.

    When `file` is empty and `symbol` is given, look the symbol up in the
    index and read its definition -- a small model often knows the name of
    the function it needs but not which file or line it lives on. If the
    name is ambiguous, return the definition list rather than guessing."""
    file = _as_text(file)
    symbol = _as_text(symbol).strip()
    start = _as_int(start, 1)
    end = _as_int(end, 0)
    if not file.strip() and not symbol:
        return ("ERROR: read needs a `file` (optionally with start/end), or a "
                "`symbol` with `file` empty. Use `grep` or `outline` to locate "
                "code first.")
    rel = None
    if file.strip():
        rel = index.resolve_file(file)
        if rel is None:
            # Files created after the last indexing run are real, readable,
            # and invisible to the index. Fold them in rather than losing the
            # step.
            rel = adopt_if_on_disk(index, cfg, file)
        if rel is None and not symbol:
            # A small model sometimes puts a symbol name in `file` by
            # mistake. If it names a real definition, treat it as a symbol
            # read rather than spending the lookup on an error.
            if _symbol_matches(index, file):
                symbol = file.strip()
            else:
                return _file_hint(index, file)

    symbol_note = ""
    if symbol:
        matches = _symbol_matches(index, symbol, rel)
        if not matches:
            hint = _symbol_hint(index, symbol)
            if rel is None:
                return hint
            detail = hint.split(" was found.", 1)[-1].strip()
            return f"ERROR: no definition of '{symbol}' in {rel}. {detail}"
        if rel is None:
            if len(matches) > 1:
                return _definition_list(symbol, matches)
            rel = matches[0].file
        # A symbol-only read means "show me that definition", even when the
        # grammar filled start/end with a default range. When a file AND a
        # symbol were given, only override a placeholder range.
        if not file.strip() or (start <= 1 and end <= 1):
            c = matches[0]
            start, end = c.start, c.end
            symbol_note = (f"\n[symbol '{symbol}' resolved to {c.qualname} "
                           f"at {rel}:{c.start}-{c.end}]")

    if rel is None:
        return _file_hint(index, file)
    path = index.root / rel
    if not path.is_file():
        return f"ERROR: {rel} is missing on disk (or is not a file)."
    try:
        lines = fileio.read_lines(path)
    except OSError as e:
        return f"ERROR: could not read {rel}: {e}"
    # A trailing newline leaves an empty phantom "last line" after split.
    # Reporting it is how a model gets told a file has 685 lines, asks for
    # L320-685, and watches the read clamp or fail right at the boundary.
    if lines and lines[-1] == "" and len(lines) > 1:
        lines = lines[:-1]
    elif lines == [""]:
        lines = []                      # an empty file has no line 1
    if not lines:
        return f"{rel} is empty (0 lines)."

    max_lines = max(1, _as_int(getattr(cfg, "max_read_lines", 200), 200))
    start = max(1, start)
    end = end or len(lines)
    end = min(end, len(lines))
    if start > len(lines):
        return f"ERROR: {rel} has only {len(lines)} lines."
    if end < start:
        # The model asked for a backwards range. Read forward from `start`
        # instead of failing -- one lookup's budget is too precious to waste
        # on an argument-order slip.
        end = min(start + max_lines - 1, len(lines))
    if end - start + 1 > max_lines:
        end = start + max_lines - 1
        note = f"\n[truncated to {max_lines} lines; read again from {end + 1} if needed]"
    else:
        note = ""
    body = _numbered(lines[start - 1 : end], start)
    return f"{rel} lines {start}-{end}:\n{body}{note}{symbol_note}"


def tool_grep(index, pattern: str, scope: str = "", max_hits: int = 40,
              cfg=None, client=None) -> str:
    """Regex search across every indexed file, or a narrower `scope`.

    `scope` may be one file, a directory (`src/`, `src/pkg`), or a glob
    (`*.py`, `tests/*.py`); an unindexed file that exists on disk is folded
    into the index first, like `read` does. Returns matching lines with their
    file:line locations and the enclosing symbol, prefixed with a warning
    that this output is for locating code, not for quoting into an edit
    (lines may be truncated to 200 chars).

    Small models frequently send a literal string where a regex is expected
    (`def foo(`), or the wrong case (`RETURN`). Both are recovered instead of
    wasting the lookup: an invalid regex is searched as literal text, and a
    case-sensitive search that finds nothing falls back to case-insensitive
    hits collected in the same pass. A sparse result (a few hits) includes
    one line of context around each hit; a total miss suggests similarly
    named symbols from the index.
    """
    pattern = _as_text(pattern)
    scope = _as_text(scope)
    max_hits = max(1, _as_int(max_hits, 40))
    if not pattern.strip():
        return ("ERROR: grep needs a `pattern` (a regex or plain text). To look "
                "inside one file, use `outline`, or `read` it.")
    notes: list[str] = []
    needle = pattern
    try:
        re.compile(pattern)
    except re.error:
        # A small model's "regex" is usually a literal snippet with a stray
        # paren or dot. Searching it literally is what it meant.
        needle = re.escape(pattern)
        notes.append("[note: pattern was not valid regex; searched it as "
                     "literal text]")

    # Adopt an on-disk file named as the scope, so a file created after the
    # last index run is searchable in the same step that names it.
    if scope and cfg is not None and index.resolve_file(scope) is None:
        adopt_if_on_disk(index, cfg, scope)
    if scope:
        # Normalise an absolute/`./`/`..` scope to a project-relative one so
        # the directory/glob matcher below sees the same spelling as the
        # index (`adopt_if_on_disk` already rejects anything outside root).
        safe_scope = _root_rel(index, scope)
        if safe_scope is not None:
            scope = safe_scope

    files, scope_note = _grep_scope(index, scope)
    if scope_note:
        notes.append(scope_note)
    if not files:
        return f"No indexed files to search for /{pattern}/."

    rx = re.compile(needle)
    # The case-insensitive fallback rides along in the same pass over the
    # files: re-reading the whole tree for a second sweep doubled the cost of
    # exactly the greps that found nothing.
    rx_ci = None
    if not rx.flags & re.IGNORECASE:
        try:
            rx_ci = re.compile(needle, re.IGNORECASE)
        except re.error:
            rx_ci = None
    entries, ci_entries = _scan_grep(index, files, rx, max_hits, rx_ci)
    if not any(n is not None for _, n, _ in entries) \
            and any(n is not None for _, n, _ in ci_entries):
        entries = ci_entries
        notes.append("[note: no case-sensitive matches; retried "
                     "case-insensitively]")

    real = [e for e in entries if e[1] is not None]
    if real and len(real) <= SPARSE_CONTEXT_MAX:
        hits = _with_context(index, entries)
        notes.append("[note: few matches -- one line of context shown around "
                     "each; a context line uses '-' where a hit uses ':']")
    else:
        hits = [t for _, _, t in entries]

    if not hits and client is not None and cfg is not None \
            and getattr(cfg, "semantic_search", False):
        # Last resort: find code by meaning. Exact search already failed, so
        # a natural-language phrase ("where do we retry?") would otherwise
        # come back empty and waste the lookup.
        try:
            from . import semantic
            matches = semantic.search(index, cfg, client, pattern, verbose=True)
        except Exception as e:
            notes.append(f"[semantic search unavailable: {str(e)[:200]}]")
        else:
            if matches:
                notes.append("[no text matches; showing semantic matches by "
                             "meaning. `read` a region before editing.]")
                hits = semantic.render(pattern, matches).splitlines()
            else:
                notes.append("[semantic search found nothing above the "
                             "similarity threshold]")

    if not hits:
        # Total miss: hand the model the closest real symbol names instead of
        # a dead end -- the index already knows them, so this costs nothing.
        suggest = _symbol_suggestions(index, pattern)
        if suggest:
            notes.append(suggest)

    note_text = ("\n".join(notes) + "\n") if notes else ""
    if not hits:
        return note_text + f"No matches for /{pattern}/."
    return _grep_result("\n".join(notes), hits)


def _call_sites(index, chunk, base: str, limit: int = 2) -> list[str]:
    """The actual source lines inside `chunk` that mention `base`.

    Ranges alone make the model fetch every caller just to see how it is
    used. A couple of exact (trimmed) lines are enough to choose the right
    one -- and the grep warning already tells it not to edit from this."""
    path = index.root / chunk.file
    try:
        lines = fileio.read_lines_cached(path)
    except OSError:
        return []
    try:
        rx = re.compile(r"\b" + re.escape(base) + r"\b")
    except re.error:
        return []
    out: list[str] = []
    for i in range(max(1, chunk.start), min(chunk.end, len(lines)) + 1):
        line = lines[i - 1]
        if not rx.search(line):
            continue
        stripped = line.strip()
        if stripped.startswith(("def ", "async def ", "class ")):
            continue        # the definition itself, not a call
        out.append(f"      L{i}: {stripped[:160]}")
        if len(out) >= limit:
            break
    return out


def _indexed_callees(index, chunk, limit: int = 8) -> list[str]:
    """Definitions the chunk calls that exist in the index, with locations.

    This turns `callers` into a cheap both-directions view: a small model
    tracing an `explain` question can jump from a function to the helpers it
    uses without spending an `outline` + `read` on each guess."""
    out: list[str] = []
    seen: set[str] = set()
    for name in getattr(chunk, "calls", ()) or ():
        base = name.split(".")[-1]
        if not base or base in seen:
            continue
        seen.add(base)
        hits = _symbol_matches(index, base)
        if not hits:
            continue
        d = hits[0]
        out.append(f"  {base}: {d.file}:{d.start}-{d.end}")
        if len(out) >= limit:
            break
    return out


def tool_callers(index, symbol: str) -> str:
    """Report where a symbol is defined, what it calls, and (by matching bare
    call names) which chunks appear to call it, with call-site lines. An
    approximate cross-reference, not a sound one (see Index.callers_of in
    index.py)."""
    symbol = _as_text(symbol).strip()
    if not symbol:
        return "ERROR: symbol is required."
    defs = index.find_symbol(symbol)
    callers = [c for c in index.callers_of(symbol) if c not in defs]
    out: list[str] = []
    if defs:
        out.append("Defined at:")
        out += [f"  {c.file}:{c.start}-{c.end}  {c.signature or c.qualname}"
                for c in defs]
        callees = _indexed_callees(index, defs[0])
        if callees:
            out.append("Calls (indexed definitions):")
            out += callees
    if callers:
        out.append("Called from:")
        base = symbol.split(".")[-1]
        for c in callers[:20]:
            out.append(f"  {c.file}:{c.start}-{c.end}  {c.qualname}")
            out += _call_sites(index, c, base)
        if len(callers) > 20:
            out.append(f"  (+{len(callers) - 20} more callers)")
    elif defs:
        out.append(f"No callers of '{symbol}' found in the indexed files.")
    else:
        return _symbol_hint(index, symbol)
    return "\n".join(out)


def tool_outline(index, cfg, file: str) -> str:
    """Return every unit (function/class/method) in one file with its
    signature, line range, and summary -- or, when `file` names a directory
    or glob, the file cards of everything under it.

    This is the model's main navigation move on a large project: the
    always-resident map (file cards, or directory cards on the largest
    projects) says something exists; the outline says where to `read` inside
    it. The directory form is how the model descends a directory-tier map.
    """
    file = _as_text(file)
    rel = index.resolve_file(file)
    if rel is None:
        rel = adopt_if_on_disk(index, cfg, file)
    if rel is None:
        # If the model named a symbol rather than a file, outline the file
        # that defines it -- still one cheap navigation step, no error.
        hits = _symbol_matches(index, file)
        if hits:
            rel = hits[0].file
    if rel is None:
        # A directory or glob: list the file cards under it instead. "."
        # lists the whole project (truncated) -- the escape hatch from a
        # directory-tier map's "(root)" card.
        name = fileio.normalise_rel(file or "").strip()
        if name in (".", "./"):
            return _outline_dir(index, ".", list(index.files))
        if name:
            matches, note = _grep_scope(index, name)
            if not note and matches:
                # note == "" means `name` really resolved as a file/dir/glob;
                # a miss falls through to the file-not-found hint rather than
                # dumping every card in the project.
                if len(matches) == 1:
                    rel = matches[0]
                else:
                    return _outline_dir(index, name, matches)
    if rel is None:
        return _file_hint(index, file)
    return index.render_outline(rel)


# Bounds for `outline <directory>`: enough cards to be useful, small enough
# that one listing cannot eat the whole gathered-source budget.
_OUTLINE_DIR_MAX_CARDS = 40
_OUTLINE_DIR_MAX_CHARS = 8000


def _outline_dir(index, name: str, matches: list[str]) -> str:
    """File cards for one directory/glob -- `outline`'s second meaning, and
    the way down from a directory-tier map to line ranges the model can
    `read`."""
    shown = matches[:_OUTLINE_DIR_MAX_CARDS]
    text = index.render_cards(shown)
    if len(text) > _OUTLINE_DIR_MAX_CHARS:
        text = (text[:_OUTLINE_DIR_MAX_CHARS]
                + "\n... [truncated; outline a narrower directory, or grep]")
    extra = ""
    if len(matches) > len(shown):
        extra = (f"\n[+{len(matches) - len(shown)} more files not shown; "
                 f"outline a subdirectory or grep to narrow]")
    tail = ("\n[Directory listing: one card per file. `outline` a file for "
            "its units; `read` straight from the line ranges shown.]")
    return f"### {name} ({len(matches)} files)\n{text}{extra}{tail}"


def tool_sessions(index, cfg, pattern: str = "", file: str = "",
                  start: int = 1, end: int = 0) -> str:
    """List, search, or read prior task session logs.

    `sessions` with no arguments lists the most recent runs. A `pattern`
    searches all logs (or one, when `file` names it); `file` alone with a
    start/end range reads that log's numbered lines. This is how a model sees
    what earlier runs asked, looked at, and changed without keeping the old
    conversation in context -- particularly useful for follow-up tasks on a
    small model with no memory between runs.
    """
    pattern = _as_text(pattern)
    file = _as_text(file)
    start = _as_int(start, 1)
    end = _as_int(end, 0)
    if pattern.strip():
        return sessions.search(index.root, pattern, scope=file)
    if file.strip():
        max_lines = max(1, _as_int(getattr(cfg, "max_read_lines", 200), 200))
        return sessions.read(index.root, file, start, end, max_lines=max_lines)
    return sessions.recent(index.root)


DIFF_NOTE = ("[Diff against Pedacito's own pre-write snapshots, not git. "
             "Historical record -- `read` a region before editing it.]")


def _snapshot_copies(index) -> list:
    """(rel, stamp, snapshot_path) for every file Pedacito has ever backed
    up, newest snapshot copy per file.

    Walks the backup dirs newest-first, so the first copy found for a rel is
    its most recent 'before' state -- the diff then answers 'what changed
    since Pedacito last touched this file' even when the last write to it was
    several runs ago."""
    from . import workspace
    out: list = []
    seen: set = set()
    try:
        snaps = workspace.list_backups(index.root)
    except OSError:
        snaps = []
    for s in snaps:
        try:
            paths = sorted(s.rglob("*"))
        except OSError:
            continue
        for p in paths:
            if not p.is_file():
                continue
            rel = p.relative_to(s).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            out.append((rel, s.name, p))
    out.sort()
    return out


def _diff_one(index, cfg, rel: str, stamp: str, snap: Path,
              max_lines: int = 0) -> str:
    """Unified diff of one file: its snapshot copy vs the current working
    tree. Pure difflib -- deliberately no `git diff` subprocess, since git
    attribute filters can execute code on checkout-adjacent operations."""
    import difflib
    try:
        old, _style = fileio.read(snap, errors="replace")
    except OSError as e:
        return f"ERROR: could not read the snapshot of {rel}: {e}"
    path = index.root / rel
    if not path.is_file():
        n = len(old.splitlines())
        return (f"DIFF {rel} (snapshot {stamp} -> working tree): DELETED "
                f"since the snapshot ({n} lines gone).\n{DIFF_NOTE}")
    try:
        new, _style = fileio.read(path, errors="replace")
    except OSError as e:
        return f"ERROR: could not read {rel}: {e}"
    if old == new:
        return f"DIFF {rel}: unchanged since snapshot {stamp}."
    body = list(difflib.unified_diff(
        old.splitlines(), new.splitlines(),
        fromfile=f"snapshot/{stamp}/{rel}", tofile=f"working/{rel}",
        lineterm="", n=3))
    cap = max(1, _as_int(max_lines or getattr(cfg, "max_read_lines", 200),
                          200))
    note = ""
    if len(body) > cap:
        body = body[:cap]
        note = f"\n[diff truncated to {cap} lines]"
    return (f"DIFF {rel} (snapshot {stamp} -> working tree):\n"
            + "\n".join(body) + note + f"\n{DIFF_NOTE}")


def tool_diff(index, cfg, file: str = "") -> str:
    """Show what changed in the working tree since Pedacito last snapshotted
    each file.

    A read-only view of the tool's own footprint: every write path (`apply`,
    the review page, gather's `write`/`append`/`move`/`delete`) copies each
    original into `.pedacito/backups/<stamp>/` first, so snapshot-vs-working
    answers "what did the previous run actually change?" without invoking
    git. With no `file`, summarises every snapshotted file; with one, shows
    that file's full unified diff. Gated by `allow_diff` (on by default;
    turning it off shrinks the tool surface by one action)."""
    if not getattr(cfg, "allow_diff", True):
        return "ERROR: `diff` is disabled (config allow_diff = false)."
    copies = _snapshot_copies(index)
    if not copies:
        return ("No Pedacito snapshots exist yet (.pedacito/backups/ is "
                "empty). Snapshots are taken before Pedacito writes anything, "
                "so until then there is nothing to diff against.")
    want = _root_rel(index, file) if _as_text(file).strip() else ""
    if _as_text(file).strip() and want is None:
        return (f"ERROR: refusing to diff outside the project root: {file}")
    if want:
        rel = index.resolve_file(want) or want
        pick = next((c for c in copies if c[0] == rel), None)
        if pick is None:
            known = ", ".join(r for r, _, _ in copies[:8])
            more = f" (+{len(copies) - 8} more)" if len(copies) > 8 else ""
            return (f"No snapshot of '{rel}' exists -- Pedacito has never "
                    f"written that file, so there is nothing to diff. "
                    f"Snapshotted files: {known}{more}.")
        return _diff_one(index, cfg, *pick)
    # No file: every snapshotted file, changed ones only, on a char budget
    # (~40 diff lines per file) so a big apply batch cannot flood the
    # observation.
    budget = max(2000, _as_int(getattr(cfg, "max_read_lines", 200), 200) * 40)
    out: list[str] = []
    used = 0
    changed = 0
    skipped: list[str] = []
    for rel, stamp, snap in copies:
        text = _diff_one(index, cfg, rel, stamp, snap, max_lines=40)
        if text.startswith(f"DIFF {rel}: unchanged"):
            continue
        changed += 1
        if used + len(text) > budget and out:
            skipped.append(rel)
            continue
        out.append(text)
        used += len(text)
    if not changed:
        return ("No changes: every file Pedacito has snapshotted still "
                f"matches its snapshot.\n{DIFF_NOTE}")
    head = f"{changed} file(s) changed since Pedacito snapshotted them:"
    if skipped:
        out.append(f"[+{len(skipped)} more changed file(s) not shown: "
                   + ", ".join(skipped[:8])
                   + "; `diff` one with file=<path>]")
    return "\n\n".join([head] + out)


def _snapshot(index, cfg, rel: str, path: Path, original: str = "",
              style=None) -> "str | None":
    """Snapshot the file's bytes before a mutating action.

    Returns an error string if the snapshot was requested and failed; None on
    success or when backups are off. Shared by write/append/move/delete so the
    destructive actions all honour the same `backup` setting.

    The copy is byte-for-byte (`shutil.copy2`), not a decode/re-encode: a
    backup that "helpfully" normalises line endings or mangles a non-UTF-8
    file is worse than no backup, because `pedacito restore` promises the
    original bytes back. `original`/`style` remain accepted for callers that
    already read the file, but are no longer used for the copy.
    """
    if not getattr(cfg, "backup", False):
        return None
    bdir = getattr(cfg, "_backup_dir", None)
    try:
        if bdir is None:
            # Gather-phase writes used to fall back to a `.bak` file beside
            # the source. Nothing ever set `_backup_dir` on the config, so
            # that was the path every `write`/`move`/`delete` took -- and the
            # `.bak` could then be picked up by the index. Keep the snapshot
            # inside `.pedacito/backups/` like every other write path.
            from . import workspace
            bdir = workspace.new_backup_dir(index.root)
        dest = Path(bdir) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(path), str(dest))
        return None
    except OSError as e:
        return f"ERROR: could not back up {rel}: {e}"


def _syntax_note(path: Path, text: str) -> str:
    """Advisory parse check on freshly written content: '' when it parses (or
    the language is unknown), else a WARNING string.

    The edit path validates and REFUSES broken syntax; the write path cannot
    refuse -- a chunked file is legitimately unparseable until its last
    `append` -- so it warns instead. Immediate feedback beats discovering the
    breakage at the next test run, and the wording covers the mid-chunk case
    so the warning does not get ignored as noise."""
    suffix = path.suffix.lower()
    err = None
    if suffix == ".py":
        import ast
        try:
            ast.parse(text)
        except SyntaxError as e:
            err = f"line {e.lineno}: {e.msg}"
        except ValueError:
            return ""                # null bytes etc.; not worth nagging
    else:
        try:
            from .chunker_ts import LANG_BY_EXT, TS_AVAILABLE, has_syntax_error
            if suffix in LANG_BY_EXT and TS_AVAILABLE:
                bad, lineno = has_syntax_error(text, LANG_BY_EXT[suffix])
                if bad:
                    err = f"near line {lineno}"
        except Exception:
            return ""
    if err is None:
        return ""
    return ("\n[WARNING: the file does not parse (" + err + "). Expected only "
            "if more `append` chunks are coming; otherwise fix it before "
            "`done` -- code that does not parse cannot run or be edited "
            "reliably.]")


def tool_write(index, cfg, file: str, content: str,
               append: bool = False) -> str:
    """Replace a whole file with `content` (or append it when `append` is
    true), backing the original up first.

    Only reachable when `cfg.allow_write` is on (see dispatch). Deliberately
    minimal: it reuses the same safety rails as the edit path -- refuse to
    escape the project root, snapshot the original into the run's backup dir,
    preserve the file's encoding style -- but it cannot offer a review diff, so
    the observation tells the model what it changed and points at `restore`.
    New files are created; an empty write against an existing file is refused
    rather than silently truncating it.

    `append` is what makes chunked file creation work. The old protocol asked
    the model to repeat the file's tail and trust the writer to fold it in;
    that never happened, so every "second chunk" overwrote the first. An
    explicit append action has no such ambiguity: the model writes the first
    <=max_write_lines lines, then appends the next chunk, and so on."""
    verb = "append" if append else "write"
    if not _as_text(file).strip():
        return f"ERROR: {verb} needs a `file` path."
    rel = _root_rel(index, file)
    if rel is None:
        return f"ERROR: refusing to write outside the project root: {file}"
    if not isinstance(content, str):
        return (f"ERROR: {verb} needs `content` to be a string of file text, "
                f"not {type(content).__name__}.")
    if not content.strip():
        if append:
            return ("ERROR: append needs non-empty `content` (the next lines to "
                    "add to the end of the file).")
        return ("ERROR: write needs non-empty `content` (the complete new file "
                "text). To change part of a file, use search/replace edit blocks "
                "in your answer instead.")
    root = Path(index.root).resolve()
    path = (root / rel).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return f"ERROR: refusing to write outside the project root: {file}"
    if path.is_dir():
        return f"ERROR: {rel} is a directory, not a file."

    existed = path.is_file()
    original = None
    style = None
    if existed:
        try:
            original, style = fileio.read(path, errors="strict")
        except UnicodeDecodeError:
            return (f"ERROR: {rel} is not valid UTF-8. Refusing to rewrite it "
                    "as UTF-8; edit it with another tool instead.")
        except OSError as e:
            return f"ERROR: could not read {rel} before writing: {e}"

    new_text = content.replace("\r\n", "\n").replace("\r", "\n")
    if not new_text.endswith("\n"):
        new_text += "\n"
    if append and existed:
        base = original or ""
        if base and not base.endswith("\n"):
            base += "\n"
        text = base + new_text
    else:
        text = new_text
    max_chars = max(1, _as_int(getattr(cfg, "max_write_chars", 200_000),
                               200_000))
    if len(text) > max_chars:
        return (f"ERROR: the resulting file is {len(text):,} chars, over the "
                f"{max_chars:,} limit. Write a smaller file, or use edit "
                "blocks.")

    if existed:
        err = _snapshot(index, cfg, rel, path, original, style)
        if err:
            return err

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fileio.write(path, text, style or fileio.TextStyle())
    except OSError as e:
        return f"ERROR: could not write {rel}: {e}"

    # Fold it into the index immediately: the map and line ranges would
    # otherwise describe bytes that no longer exist, and the next `read` of the
    # session would show the model stale code.
    try:
        from . import index as index_mod
        fresh = index_mod.reindex_file(index, rel, cfg, save=False)
        index.__dict__.update(fresh.__dict__)
    except Exception as e:
        print(f"  [write] index refresh failed for {rel}: {e}", file=sys.stderr)

    n = len(text.splitlines())
    # Advisory only: a chunked file legitimately does not parse until its
    # last append, so this warns rather than refusing (see _syntax_note).
    syntax = _syntax_note(path, text)
    if append:
        added = len(new_text.splitlines())
        note = (f"APPENDED {added} line(s) to {rel} (now {n} lines total)."
                if existed else f"CREATED {rel} ({n} lines).")
        return (note + "\n[This change was applied directly, without a review "
                "page. Undo with `pedacito restore` if it was wrong. If the "
                "file is not complete yet, `append` the next chunk now; choose "
                "`done` only once every part of the task is written.]"
                + syntax)

    note = (f"WROTE {rel} ({n} lines), replacing {len(original.splitlines())} "
            f"previous line(s)." if existed else f"CREATED {rel} ({n} lines).")
    budget = max(1, _as_int(getattr(cfg, "max_write_lines", 0), 0))
    if budget and n > budget:
        return (note + f"\n[Chunking is ON: one write step may emit at most "
                f"{budget} lines. This file needs more. Do NOT re-write the "
                f"whole file. Instead choose `append` with the next "
                f"<={budget} lines of {rel}; repeat `append` until complete, "
                f"then choose `done`.]"
                + "\n[This overwrite was applied directly, without a review "
                "page. Undo with `pedacito restore` if it was wrong.]"
                + syntax)
    return (note + "\n[This overwrite was applied directly, without a review "
            "page. Undo with `pedacito restore` if it was wrong. Now choose "
            "`done` and summarise what you wrote.]"
            + syntax)


def tool_append(index, cfg, file: str, content: str) -> str:
    """Append `content` to the end of `file`, creating it if needed.

    Thin wrapper over tool_write so the two actions share the same root guard,
    backup, encoding handling, and index refresh. Gated by `allow_write` in
    dispatch, like `write`."""
    return tool_write(index, cfg, file, content, append=True)


def tool_move(index, cfg, file: str, to: str) -> str:
    """Rename/move one file, refusing to overwrite an existing destination.

    The answer phase can already express a rename with a paired set of edit
    blocks; this is the gather-phase equivalent so a model can reorganise a
    file it just read without leaving the tool loop. Gated by `allow_write`.
    """
    src_name = _as_text(file).strip()
    dst_name = _as_text(to).strip()
    if not src_name:
        return "ERROR: move needs `file` (the old path)."
    if not dst_name:
        return "ERROR: move needs `to` (the new path)."
    src_rel = index.resolve_file(src_name) or _root_rel(index, src_name)
    dst_rel = _root_rel(index, dst_name)
    if src_rel is None or dst_rel is None:
        return f"ERROR: refusing to move outside the project root: {file} -> {to}"
    root = Path(index.root).resolve()
    src = (root / src_rel).resolve()
    dst = (root / dst_rel).resolve()
    try:
        src.relative_to(root)
        dst.relative_to(root)
    except ValueError:
        return f"ERROR: refusing to move outside the project root: {file} -> {to}"
    if not src.is_file():
        return f"ERROR: {src_rel} is not an existing file."
    if dst.exists():
        return (f"ERROR: destination already exists: {dst_rel}. Choose another "
                f"path or read it first.")
    try:
        # Move handles arbitrary files, including non-UTF-8 ones; only the
        # line count in the observation needs decoded text.
        original, style = fileio.read(src, errors="replace")
    except OSError as e:
        return f"ERROR: could not read {src_rel}: {e}"
    err = _snapshot(index, cfg, src_rel, src, original, style)
    if err:
        return err
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    except OSError as e:
        return f"ERROR: could not move {src_rel} -> {dst_rel}: {e}"

    summary = index.file_summaries.get(src_rel)
    index.forget(src_rel)
    if summary:
        index.file_summaries[dst_rel] = summary
    try:
        from . import index as index_mod
        fresh = index_mod.reindex_file(index, dst_rel, cfg, save=False)
        index.__dict__.update(fresh.__dict__)
    except Exception as e:
        print(f"  [move] index refresh failed for {dst_rel}: {e}", file=sys.stderr)

    n = len(original.splitlines())
    return (f"MOVED {src_rel} -> {dst_rel} ({n} lines). "
            "[This change was applied directly, without a review page. Undo "
            "with `pedacito restore` if it was wrong.]")


def tool_delete(index, cfg, file: str) -> str:
    """Delete one file, after snapshotting it when backups are on.

    Refuses directories and anything outside the project root. The index entry
    is dropped immediately so the next prompt's map does not advertise a file
    that no longer exists. Gated by `allow_write`."""
    name = _as_text(file).strip()
    if not name:
        return "ERROR: delete needs a `file` path."
    rel = index.resolve_file(name) or _root_rel(index, name)
    if rel is None:
        return f"ERROR: refusing to delete outside the project root: {file}"
    root = Path(index.root).resolve()
    path = (root / rel).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return f"ERROR: refusing to delete outside the project root: {file}"
    if not path.is_file():
        return f"ERROR: {rel} is not an existing file."
    try:
        # Delete works for any file; decoded text is only used for the line
        # count and the observation.
        original, style = fileio.read(path, errors="replace")
    except OSError as e:
        return f"ERROR: could not read {rel}: {e}"
    err = _snapshot(index, cfg, rel, path, original, style)
    if err:
        return err
    try:
        path.unlink()
    except OSError as e:
        return f"ERROR: could not delete {rel}: {e}"
    index.forget(rel)
    n = len(original.splitlines())
    undo = ("backed up; use `pedacito restore` to undo."
            if getattr(cfg, "backup", False) else
            "backups are OFF, so this cannot be undone.")
    return f"DELETED {rel} ({n} lines) -- {undo}"


def _split_command(cmd: str) -> list[str]:
    """Split a command into argv without invoking a shell.

    POSIX mode is used everywhere except Windows, where it would eat
    backslashes in paths. Windows' non-POSIX splitter keeps the surrounding
    quotes, so strip a matching pair from each token -- the subprocess layer
    re-quotes arguments itself."""
    if os.name != "nt":
        return shlex.split(cmd)
    return [p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'"
            else p for p in shlex.split(cmd, posix=False)]


def _allowlist_error(cfg, argv: list[str]) -> str:
    """Return an allowlist error for `argv`, or '' when it may run."""
    allow = tuple(getattr(cfg, "shell_allowlist", ()) or ())
    if not allow:
        return ""
    names = {Path(a).name.lower() for a in allow}
    exe = Path(argv[0]).name.lower()
    if exe in names or argv[0].lower() in {a.lower() for a in allow}:
        return ""
    return (f"ERROR: `{argv[0]}` is not in shell_allowlist "
            f"({', '.join(allow)}).")


def tool_run(index, cfg, command: str) -> str:
    """Run one command in the project root and return its output.

    Opt-in (`allow_shell = false` by default) because this executes code the
    model wrote. The command is split into an argv and run WITHOUT a shell, so
    pipes/redirection/`&&` are inert arguments rather than executed syntax.
    `shell_allowlist`, when set, restricts the executable name. Output and
    runtime are both capped so a hung or chatty command cannot eat the task's
    context budget.

    A GUI/server launch is the problem case: it opens a window and never
    exits. `proc.run` watches output and stops the process (and its process
    group) after `shell_idle_seconds` of silence, returning "started, then
    stopped" instead of blocking until a person closes the window. Test
    runners keep printing, so they still get the full `shell_timeout`."""
    if not getattr(cfg, "allow_shell", False):
        return ("ERROR: `run` is disabled (config allow_shell = false). Enable "
                "it to run tests or linters from the gather loop, or run the "
                "command yourself.")
    cmd = _as_text(command).strip()
    if not cmd:
        return "ERROR: run needs a `command`, e.g. `pytest -q`."
    try:
        argv = _split_command(cmd)
    except ValueError as e:
        return f"ERROR: could not parse command ({e})."
    if not argv:
        return "ERROR: run needs a `command`."

    denied = _allowlist_error(cfg, argv)
    if denied:
        return denied

    timeout = max(1, _as_int(getattr(cfg, "shell_timeout", 120), 120))
    idle = max(0, _as_int(getattr(cfg, "shell_idle_seconds", 10), 10))
    cap = max(200, _as_int(getattr(cfg, "max_shell_output_chars", 8000), 8000))
    env = dict(os.environ, NO_COLOR="1", TERM="dumb", GIT_PAGER="cat",
               PAGER="cat")
    rc, out, stopped, _elapsed = proc.run(
        argv, cwd=index.root, timeout=timeout, idle_timeout=idle,
        max_output=cap, env=env)
    if stopped == "error":
        return f"ERROR: could not run command: {out}"

    # A command can rewrite source (snapshot updates, codegen, formatters).
    # Line ranges are only valid against the bytes they were indexed from, so
    # refresh what changed before the next `read` sees a stale map.
    try:
        from . import index as index_mod
        changed, missing = index.stale()
        for rel in missing:
            index.forget(rel)
        for rel in changed:
            fresh = index_mod.reindex_file(index, rel, cfg, save=False)
            index.__dict__.update(fresh.__dict__)
        if changed or missing:
            print(f"[run] index refreshed ({len(changed)} changed, "
                  f"{len(missing)} removed)", file=sys.stderr)
    except Exception as e:                     # stale index is not fatal
        print(f"[run] index refresh skipped: {e}", file=sys.stderr)

    out = out.strip() or "(no output)"
    if stopped == "idle":
        head = (f"RUN: {cmd}\nSTOPPED: no output for {idle}s; the process was "
                "still running and has been killed.")
        tail = ("\n[This usually means a GUI or a server. Startup is all this "
                "check can observe; do not re-run it just to watch it. Use a "
                "bounded test command for pass/fail.]")
    elif stopped == "timeout":
        head = f"RUN: {cmd}\nSTOPPED: timed out after {timeout}s (killed)."
        tail = ("\n[The command was still running at the hard limit; the output "
                "above is what it produced before being stopped.]")
    else:
        status = "OK" if rc == 0 else "FAILED"
        head = f"RUN: {cmd}\nEXIT: {rc} ({status})"
        tail = ""
    server_url = browser.find_server_url(out)
    if server_url:
        head += f"\nSERVER: {server_url}"
        if getattr(cfg, "allow_browser", False):
            tail += ("\n[This was a WEB SERVER, not a GUI window. `run` stops "
                     "a quiet server; use `browser` with command set to this "
                     "launch command or its URL to open the page, interact, "
                     "and capture it.]")
        else:
            tail += ("\n[This was a WEB SERVER. `run` stopped it after idle; "
                     "a desktop screenshot cannot see its page. Enable "
                     "allow_browser (and install the browser extra) to drive "
                     "it, or verify with a console/headless check.]")
    return f"{head}\nOUTPUT:\n{out}{tail}"


def _resolve_screenshot(index, name: str) -> "Path | None":
    """Resolve an existing screenshot safely under the project root.

    Accepts a path relative to the project, or a bare filename from
    `.pedacito/screenshots/`. Absolute paths outside the root are refused."""
    n = _as_text(name).strip().strip("`\"'")
    if not n:
        return None
    root = Path(index.root).resolve()
    cands = []
    p = Path(n)
    if p.is_absolute():
        cands.append(p)
    else:
        cands.append(root / n)
        cands.append(root / ".pedacito" / "screenshots" / Path(n).name)
    for c in cands:
        try:
            c = c.resolve()
            c.relative_to(root)
        except (OSError, ValueError):
            continue
        if c.is_file():
            return c
    return None


def tool_screenshot(index, cfg, client, command: str = "", file: str = "",
                    start: int = 0) -> str:
    """Capture the screen (optionally launching an app first) and report what
    a text-only model can know about it.

    A model cannot see a GUI. This tool turns the image into text: pixel
    statistics and a blank/black verdict (Pillow), OCR text (tesseract), and
    an optional description from a multimodal model (`vision_model`). The PNG
    is kept under `.pedacito/screenshots/` so a human can inspect it.

    Gated by `allow_screenshots`; when `command` is given, also by
    `allow_shell` because it launches code. The launched process is stopped
    after the capture, so it cannot block the task.
    """
    if not getattr(cfg, "allow_screenshots", False):
        return ("ERROR: `screenshot` is disabled (config allow_screenshots = "
                "false). Enable it to capture the screen for visual checks.")
    cmd = _as_text(command).strip()
    file = _as_text(file).strip()
    start = _as_int(start, 0)
    prev = _resolve_screenshot(index, file) if file else None
    if file and prev is None:
        return f"ERROR: no screenshot named '{file}' under the project root."

    argv = []
    if cmd:
        if not getattr(cfg, "allow_shell", False):
            return ("ERROR: launching a screenshot command needs allow_shell "
                    "= true as well as allow_screenshots.")
        try:
            argv = _split_command(cmd)
        except ValueError as e:
            return f"ERROR: could not parse command ({e})."
        if not argv:
            return "ERROR: screenshot command is empty."
        denied = _allowlist_error(cfg, argv)
        if denied:
            return denied

    root = Path(index.root).resolve()
    out_dir = root / ".pedacito" / "screenshots"
    out = out_dir / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".png")
    delay = start or _as_int(getattr(cfg, "screenshot_wait_seconds", 5), 5)
    delay = max(0, min(delay, 60))
    env = dict(os.environ, NO_COLOR="1", TERM="dumb", PAGER="cat",
               GIT_PAGER="cat")

    launched = None
    exit_code = None
    server_url = ""
    launch_output = ""
    if prev is not None and not argv:
        # File-only mode: analyse that image instead of capturing again.
        out = prev
        method = "existing screenshot"
    else:
        if argv:
            if delay:
                # Use the server-aware launcher so a dev server's printed URL
                # is recognised while we wait; a GUI simply waits the delay.
                try:
                    launched, server_url, launch_output = \
                        browser.launch_server(
                            argv, cwd=root, env=env, wait_seconds=delay)
                except Exception as e:
                    return f"ERROR: could not launch '{cmd}': {e}"
                if launched is None:
                    return f"ERROR: could not launch '{cmd}': {launch_output}"
            else:
                try:
                    launched = proc.spawn(argv, cwd=root, env=env)
                except OSError as e:
                    return f"ERROR: could not launch '{cmd}': {e}"
        elif delay:
            # Give the currently running app a moment to settle.
            time.sleep(delay)

        exit_code = launched.poll() if launched is not None else None
        ok, method = vision.capture(out)
        if launched is not None:
            proc.terminate(launched)
            try:
                launched.wait(timeout=5)
            except Exception:
                pass
        if not ok:
            return f"ERROR: could not capture the screen: {method}"

    # Only compare when a NEW capture was taken; in file-only analysis mode
    # `out` and `prev` are the same file, so "0% changed from itself" is
    # noise rather than evidence.
    report = vision.analyze(out, previous=prev if argv else None)
    desc = ""
    if (getattr(cfg, "vision_model", "") or "").strip():
        try:
            desc = vision.describe(client, cfg, out)
        except Exception as e:
            desc = f"(vision model failed: {str(e)[:200]})"

    try:
        rel_out = out.relative_to(root).as_posix()
    except ValueError:                            # pragma: no cover
        rel_out = str(out)
    bits = [f"SCREENSHOT: {rel_out}", f"CAPTURE: {method}"]
    if argv:
        status = (f" (process had already exited with code {exit_code} before "
                  f"the capture)" if exit_code is not None else
                  " (still running; stopped after capture)")
        bits.append(f"LAUNCHED: {cmd}{status}")
    if server_url:
        # A desktop capture cannot show a page that no browser ever opened.
        # Say that plainly: otherwise a small model reads a static/blank
        # desktop shot as proof the game runs and behaves correctly.
        bits.append(f"SERVER: {server_url}")
        if getattr(cfg, "allow_browser", False):
            bits.append(
                "[This command is a WEB SERVER. The screenshot above is the "
                "DESKTOP, not its page. Do not infer game/runtime behavior "
                "from it. Use `browser` with command set to the URL or this "
                "launch command, and pattern set to e.g. 'press Enter', to "
                "open the page, interact, wait, and capture it.]")
        else:
            bits.append(
                "[This command is a WEB SERVER. The screenshot above is the "
                "DESKTOP, not its page. `allow_browser` is off, so this run "
                "cannot open or drive the page. Do not claim runtime behavior "
                "from this image; enable allow_browser (and install the "
                "browser extra) or verify with a console/headless check.]")
    if prev is not None and argv:
        try:
            bits.append("COMPARED WITH: " + prev.relative_to(root).as_posix())
        except ValueError:                        # pragma: no cover
            bits.append("COMPARED WITH: " + str(prev))
    bits.append("ANALYSIS:\n" + report)
    if desc:
        bits.append("VISION DESCRIPTION:\n" + desc)
    else:
        bits.append("[No vision_model configured: the evidence above is "
                    "pixel/OCR text. Set vision_model to a multimodal model "
                    "for an image description. The saved PNG can also be "
                    "opened by a human.]")
    bits.append("[A text model cannot see the image directly. Treat a "
                "black/blank verdict or OCR text as evidence, not a guess "
                "about rendering quality.]")
    return "\n".join(bits)


def tool_browser(index, cfg, client, command: str = "", pattern: str = "",
                 start: int = 0, file: str = "") -> str:
    """Open a web page, optionally interact with it, wait, and capture it.

    `command` is either an http(s) URL or a command that starts a local web
    server and prints its URL (the common `python main.py` case). `pattern`
    holds a tiny interaction list -- `press Enter; click #start; wait 2` --
    so the model can reach a game that waits behind a start screen. `start`
    is the wait before capture. A desktop `screenshot` cannot do any of this;
    this tool captures the PAGE and reports its text, canvas sizes, console
    errors, and optional vision description.

    Gated by `allow_browser`; launching a command also needs `allow_shell`.
    """
    if not getattr(cfg, "allow_browser", False):
        return ("ERROR: `browser` is disabled (config allow_browser = false). "
                "Enable it and install the browser extra:\n"
                "  pip install \"pedacito-local[browser]\"\n"
                "  python -m playwright install chromium")
    cmd = _as_text(command).strip()
    actions = _as_text(pattern).strip()
    start = _as_int(start, 0)
    file = _as_text(file).strip()
    if not cmd:
        return ("ERROR: browser needs `command`: an http(s) URL, or a command "
                "that starts a local server and prints its URL.")
    prev = _resolve_screenshot(index, file) if file else None
    if file and prev is None:
        return f"ERROR: no screenshot named '{file}' under the project root."

    root = Path(index.root).resolve()
    env = dict(os.environ, NO_COLOR="1", TERM="dumb", PAGER="cat",
               GIT_PAGER="cat")
    server = None
    server_url = ""
    if re.match(r"^(?:https?://|localhost(?::\d+)?(?:/|$)|"
                r"127\.0\.0\.1(?::\d+)?(?:/|$))", cmd, re.IGNORECASE):
        url = cmd if "://" in cmd else "http://" + cmd
        url = browser.normalise_url(url)
    else:
        if not getattr(cfg, "allow_shell", False):
            return ("ERROR: launching a browser server command needs "
                    "allow_shell = true as well as allow_browser. Pass the "
                    "URL directly if the server is already running.")
        try:
            argv = _split_command(cmd)
        except ValueError as e:
            return f"ERROR: could not parse command ({e})."
        if not argv:
            return "ERROR: browser command is empty."
        denied = _allowlist_error(cfg, argv)
        if denied:
            return denied
        wait = max(1, _as_int(getattr(cfg, "browser_server_wait_seconds", 15),
                              15))
        try:
            server, url, launch_output = browser.launch_server(
                argv, cwd=root, env=env, wait_seconds=wait)
        except Exception as e:
            return f"ERROR: could not launch '{cmd}': {e}"
        if server is None:
            return f"ERROR: could not launch '{cmd}': {launch_output}"
        if not url:
            proc.terminate(server)
            try:
                server.wait(timeout=5)
            except Exception:
                pass
            detail = (launch_output or "").strip()[-1200:]
            hint = ""
            if getattr(cfg, "allow_shell", False):
                hint = (" Start the server yourself and pass its URL as "
                        "`command`, or run it with a console check.")
            return (f"ERROR: '{cmd}' ran but printed no http:// URL within "
                    f"{wait}s, so there is no page to open.{hint}\n"
                    f"OUTPUT:\n{detail or '(no output)'}")
        server_url = url

    out_dir = root / ".pedacito" / "screenshots"
    out = out_dir / ("browser-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                     + ".png")
    delay = start or _as_int(getattr(cfg, "browser_wait_seconds", 3), 3)
    delay = max(0, min(delay, 60))
    timeout = max(5, _as_int(getattr(cfg, "browser_timeout_seconds", 20), 20))
    try:
        ok, info = browser.run(
            url, actions=actions, wait=delay, out=out, timeout=timeout,
            headless=bool(getattr(cfg, "browser_headless", True)))
    finally:
        if server is not None:
            proc.terminate(server)
            try:
                server.wait(timeout=5)
            except Exception:
                pass
    if not ok:
        return f"ERROR: {info}"

    report = vision.analyze(out, previous=prev)
    desc = ""
    if (getattr(cfg, "vision_model", "") or "").strip():
        try:
            desc = vision.describe(client, cfg, out)
        except Exception as e:
            desc = f"(vision model failed: {str(e)[:200]})"

    try:
        rel_out = out.relative_to(root).as_posix()
    except ValueError:                            # pragma: no cover
        rel_out = str(out)
    bits = [f"BROWSER: {info.get('url') or url}",
            f"CAPTURE: {info.get('method') or 'browser'}"
            f" ({rel_out})"]
    if server_url:
        bits.append(f"LAUNCHED: {cmd} (server {server_url}; stopped after "
                    f"capture)")
    if info.get("title"):
        bits.append(f"TITLE: {info['title']}")
    performed = info.get("actions") or []
    if performed:
        bits.append("ACTIONS: " + "; ".join(performed))
    if actions and not performed:
        bits.append(f"ACTIONS REQUESTED: {actions}")
    if info.get("action_errors"):
        bits.append("ACTION ERRORS:\n" + "\n".join(
            f"- {e}" for e in info["action_errors"]))
    if info.get("canvas"):
        bits.append("CANVAS: " + str(info["canvas"]))
    text = (info.get("text") or "").strip()
    if text:
        bits.append("PAGE TEXT:\n" + text[:3000])
    else:
        bits.append("PAGE TEXT: (none visible -- the page may draw only to "
                    "a canvas)")
    if info.get("console"):
        bits.append("CONSOLE:\n" + "\n".join(
            f"- {e}" for e in info["console"][-8:]))
    if prev is not None:
        try:
            bits.append("COMPARED WITH: " + prev.relative_to(root).as_posix())
        except ValueError:                        # pragma: no cover
            bits.append("COMPARED WITH: " + str(prev))
    bits.append("ANALYSIS:\n" + report)
    if desc:
        bits.append("VISION DESCRIPTION:\n" + desc)
    else:
        bits.append("[No vision_model configured: the evidence above is the "
                    "page's text/canvas and pixel analysis. Set vision_model "
                    "for an image description.]")
    if not actions:
        bits.append("[If the game waits behind a start screen, run `browser` "
                    "again with pattern='press Enter' (or a click/type "
                    "action) so the capture shows the running game.]")
    bits.append("[A single capture is a snapshot. To judge time-dependent "
                "behavior, take another capture after a known wait and "
                "compare the two images/text.]")
    return "\n".join(bits)


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------
def dispatch(index, cfg, step: dict, client=None) -> str:
    """Route one parsed agent step (matching STEP_SCHEMA) to the
    corresponding tool_* function and return its text observation.

    `client` is only needed for optional semantic search; every other action
    works without it, so offline tests and CLI inspection callers can omit it.
    """
    if not isinstance(step, dict):
        return f"ERROR: malformed step (expected an object, got {type(step).__name__})."
    action = _as_text(step.get("action")).strip()
    if action == "read":
        return tool_read(index, cfg, step.get("file", ""),
                         step.get("start", 1), step.get("end", 0),
                         step.get("symbol", ""))
    if action == "grep":
        return tool_grep(index, step.get("pattern", ""), step.get("file", ""),
                         cfg=cfg, client=client)
    if action == "callers":
        return tool_callers(index, step.get("symbol", ""))
    if action == "outline":
        return tool_outline(index, cfg, step.get("file", ""))
    if action == "sessions":
        return tool_sessions(index, cfg, step.get("pattern", ""),
                             step.get("file", ""), step.get("start", 1),
                             step.get("end", 0))
    if action == DIFF_ACTION:
        return tool_diff(index, cfg, step.get("file", ""))
    if action == RUN_ACTION:
        return tool_run(index, cfg, step.get("command", ""))
    if action == SCREENSHOT_ACTION:
        return tool_screenshot(index, cfg, client, step.get("command", ""),
                               step.get("file", ""), step.get("start", 0))
    if action == BROWSER_ACTION:
        return tool_browser(index, cfg, client, step.get("command", ""),
                            step.get("pattern", ""), step.get("start", 0),
                            step.get("file", ""))
    if action in MUTATING_ACTIONS:
        if not getattr(cfg, "allow_write", False):
            return (f"ERROR: `{action}` is disabled (config allow_write = false). "
                    "Answer the task with search/replace edit blocks instead -- "
                    "`read` the lines you want to change, then emit them in your "
                    "final reply.")
        if action == MOVE_ACTION:
            return tool_move(index, cfg, step.get("file", ""),
                             step.get("to", ""))
        if action == DELETE_ACTION:
            return tool_delete(index, cfg, step.get("file", ""))
        # Guard a write/append BEFORE it happens when its content already
        # exceeds one step's budget (max_write_lines). A 791-line `content`
        # string can only have arrived here as a truncated reply -- executing
        # it would silently store half a file. Refuse and teach the chunked
        # protocol instead so the model writes <=budget-line steps that each
        # fit the provider's real output limit.
        budget = max(1, _as_int(getattr(cfg, "max_write_lines", 0), 0))
        raw_content = step.get("content", "")
        content = raw_content if isinstance(raw_content, str) else ""
        nlines = len(content.splitlines())
        if budget and nlines > budget:
            if action == WRITE_ACTION:
                advice = (f"first `write` the first <={budget} lines, then "
                          f"`append` the next <={budget} lines, repeating "
                          f"`append` until the file is complete.")
            else:
                advice = (f"`append` at most <={budget} lines now, then "
                          f"`append` the next <={budget} lines, repeating until "
                          f"the file is complete.")
            return (f"ERROR: that `{action}` is {nlines} lines -- over the "
                    f"{budget}-line budget for ONE step (max_write_lines). "
                    f"Writing it in one message gets cut off at the token "
                    f"limit. Do it in chunks instead: {advice} Choose `done` "
                    f"only after every part of the task is written.")
        if action == APPEND_ACTION:
            return tool_append(index, cfg, step.get("file", ""), content)
        return tool_write(index, cfg, step.get("file", ""), content)
    return f"ERROR: unknown action '{action}'."
