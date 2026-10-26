"""The agent's tool surface: read, grep, callers, outline, sessions, write/append, done.

Usage
-----
    from .tools import STEP_SCHEMA, TOOL_HELP, dispatch

    step = client.chat_json(messages, STEP_SCHEMA, max_tokens=700)
    observation = dispatch(index, cfg, step)

Deliberately a small, fixed set of tools. Local models degrade fast as the
tool surface grows -- four to six is about the ceiling before a small model
starts calling the wrong one or inventing arguments. The six navigation
tools (read, grep, callers, outline, sessions, done) are the core surface.
`sessions` is read-only history and is how a fresh run sees prior tasks.
Everything destructive is behind an explicit gate: `write`/`append`/`move`/
`delete` need `cfg.allow_write`, and `run` needs `cfg.allow_shell`. A default run
therefore exposes only the navigation surface; a user who opts in accepts a
larger surface because they asked for those capabilities. Every tool returns
plain text with line numbers attached, since the model needs those numbers
to write search/replace edits later.

The navigation tools are deliberately forgiving, because small models make
argument mistakes that must not cost a whole lookup: `read` accepts a symbol
name instead of a line range, `grep` falls back to a literal search when the
"regex" does not compile, retries case-insensitively when a case-sensitive
search finds nothing, and accepts a directory or glob as its scope. When
`semantic_search` is on, a grep with no textual hits queries the server's
embeddings endpoint instead of returning an empty observation. A step that
would otherwise have failed can often still return useful source.

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
import subprocess
import sys
from pathlib import Path

from . import fileio, sessions

TOOL_HELP = """\
read     - read exact lines of a file.       args: file, start, end
           (or set symbol and leave file empty: read that definition)
grep     - regex or literal search. args: pattern, and optional file/dir/glob
           (with semantic_search on, falls back to search by meaning)
callers  - who calls a symbol, with the call sites. args: symbol
outline  - list every unit in one file.      args: file
sessions - list/search what prior runs did.  args: pattern, optional file
write    - overwrite a whole file with text. args: file, content
append   - add text to the END of a file.    args: file, content
move     - rename/move one file.             args: file (old), to (new)
delete   - delete one file.                  args: file
run      - run one command, no shell pipes.  args: command
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
                     "write", "append", "move", "delete", "run", "done"],
        },
        "file": {"type": "string",
                 "description": "For read/outline/write/append/move/delete. For "
                                "grep, optionally a file, directory, or glob. "
                                "Empty otherwise.",
                 "maxLength": 200},
        "start": {"type": "integer", "description": "For read. 0 otherwise."},
        "end": {"type": "integer", "description": "For read. 0 otherwise."},
        "pattern": {"type": "string",
                    "description": "For grep: regex, or plain text if it is not "
                                   "valid regex. Empty otherwise.",
                    "maxLength": 200},
        "symbol": {"type": "string",
                   "description": "For callers; or for read when file is empty "
                                  "(reads that definition). Empty otherwise.",
                   "maxLength": 200},
        "to": {"type": "string",
               "description": "For move: the new path. Empty otherwise.",
               "maxLength": 200},
        "command": {"type": "string",
                    "description": "For run: executable and arguments, e.g. "
                                   "'pytest -q'. No pipes or redirection. "
                                   "Empty otherwise.",
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
ACTIONS = ("read", "grep", "callers", "outline", "sessions", "write",
           "append", "move", "delete", "run", "done")
WRITE_ACTION = "write"
APPEND_ACTION = "append"
MOVE_ACTION = "move"
DELETE_ACTION = "delete"
RUN_ACTION = "run"
WRITE_ACTIONS = (WRITE_ACTION, APPEND_ACTION)
MUTATING_ACTIONS = WRITE_ACTIONS + (MOVE_ACTION, DELETE_ACTION)


def schema_for(cfg) -> dict:
    """Return STEP_SCHEMA narrowed to the actions this run actually allows.

    Grammar-constrained decoding sees the enum, so leaving disabled actions in
    it invites a small model to call them and burn a step on an ERROR. Gating
    the schema itself (not just dispatch) keeps the default surface at the
    five navigation actions. Fields only used by a disabled action are dropped
    too, so the model is not asked for `content`/`to`/`command` it cannot use.
    """
    import copy

    actions = ["read", "grep", "callers", "outline", "sessions"]
    if getattr(cfg, "allow_write", False):
        actions += ["write", "append", "move", "delete"]
    if getattr(cfg, "allow_shell", False):
        actions.append("run")
    actions.append("done")

    schema = copy.deepcopy(STEP_SCHEMA)
    schema["properties"]["action"]["enum"] = actions
    required = ["action", "file", "start", "end", "pattern", "symbol",
                "reasoning"]
    drop = set()
    if not getattr(cfg, "allow_write", False):
        drop |= {"content", "to"}
    if not getattr(cfg, "allow_shell", False):
        drop.add("command")
    for key in drop:
        schema["properties"].pop(key, None)
    for key in ("content", "to", "command"):
        if key not in drop:
            required.append(key)
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
        matches = index.find_symbol(symbol)
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
    s = fileio.normalise_rel(scope or "").strip()
    if not s:
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


def _scan_grep(index, files: list[str], rx: re.Pattern,
               max_hits: int) -> tuple[list[str], bool]:
    """Search `files` with `rx`; return (hits, stopped_at_cap)."""
    hits: list[str] = []
    per_file_cap = max(5, max_hits // 4)
    for rel in files:
        path = index.root / rel
        if not path.exists():
            continue
        file_hits = 0
        for i, line in enumerate(fileio.read_lines(path), 1):
            if not rx.search(line):
                continue
            if file_hits >= per_file_cap:
                hits.append(f"{rel}: [more matches in this file; "
                            f"grep it directly with `file` set]")
                break
            # Indentation is preserved: a stripped line looks editable and
            # is not, which is how a model ends up writing SEARCH blocks
            # that can never match.
            shown = line[:200].rstrip()
            cut = " ...[line truncated]" if len(line) > 200 else ""
            unit = _enclosing_chunk(index, rel, i)
            where = (f" [{unit.qualname}]" if unit is not None
                     and unit.qualname != "module" else "")
            hits.append(f"{rel}:{i}:{where} {shown}{cut}")
            file_hits += 1
            if len(hits) >= max_hits:
                hits.append(f"[stopped at {max_hits} hits; narrow the pattern]")
                return hits, True
    return hits, False


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
    rel = fileio.normalise_rel(rel or "")
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
    cost the model a lookup step just because indexing ran before it existed."""
    cand = fileio.normalise_rel(file)
    if not cand:
        return None
    if not (Path(index.root) / cand).is_file():
        return None
    fresh = _fresh_index(index, cfg, cand)
    if fresh is None:
        return None
    index.__dict__.update(fresh.__dict__)
    return index.resolve_file(file)


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
    symbol = (symbol or "").strip()
    if not (file or "").strip() and not symbol:
        return ("ERROR: read needs a `file` (optionally with start/end), or a "
                "`symbol` with `file` empty. Use `grep` or `outline` to locate "
                "code first.")
    rel = None
    if (file or "").strip():
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
                symbol = (file or "").strip()
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
        # Only override the range when the model did not supply a real one:
        # schema-required fields arrive as 0, or as 1/1, on symbol reads.
        if int(start or 0) <= 1 and int(end or 0) <= 1:
            c = matches[0]
            start, end = c.start, c.end
            symbol_note = (f"\n[symbol '{symbol}' resolved to {c.qualname} "
                           f"at {rel}:{c.start}-{c.end}]")

    if rel is None:
        return _file_hint(index, file)
    path = index.root / rel
    if not path.exists():
        return f"ERROR: {rel} missing on disk."
    lines = fileio.read_lines(path)
    # A trailing newline leaves an empty phantom "last line" after split.
    # Reporting it is how a model gets told a file has 685 lines, asks for
    # L320-685, and watches the read clamp or fail right at the boundary.
    if lines and lines[-1] == "" and len(lines) > 1:
        lines = lines[:-1]

    start = max(1, int(start or 1))
    end = int(end or 0) or len(lines)
    end = min(end, len(lines))
    if start > len(lines):
        return f"ERROR: {rel} has only {len(lines)} lines."
    if end < start:
        # The model asked for a backwards range. Read forward from `start`
        # instead of failing -- one lookup's budget is too precious to waste
        # on an argument-order slip.
        end = min(start + cfg.max_read_lines - 1, len(lines))
    if end - start + 1 > cfg.max_read_lines:
        end = start + cfg.max_read_lines - 1
        note = f"\n[truncated to {cfg.max_read_lines} lines; read again from {end + 1} if needed]"
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
    case-sensitive search that finds nothing is retried case-insensitively.
    """
    if not (pattern or "").strip():
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

    files, scope_note = _grep_scope(index, scope)
    if scope_note:
        notes.append(scope_note)
    if not files:
        return f"No indexed files to search for /{pattern}/."

    rx = re.compile(needle)
    hits, _ = _scan_grep(index, files, rx, max_hits)
    if not hits:
        # Case-insensitive retry: the same code path, one cheap second pass.
        rx_ci = re.compile(needle, re.IGNORECASE)
        ci_hits, _ = _scan_grep(index, files, rx_ci, max_hits)
        if ci_hits:
            hits = ci_hits
            notes.append("[note: no case-sensitive matches; retried "
                         "case-insensitively]")

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
        lines = fileio.read_lines(path)
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
    if not symbol.strip():
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
    signature, line range, and summary.

    This is the model's main navigation move on a large project: the
    always-resident file card (see index.py) says a module exists; the
    outline says where inside it to `read`.
    """
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
        return _file_hint(index, file)
    return index.render_outline(rel)


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
    if (pattern or "").strip():
        return sessions.search(index.root, pattern, scope=file)
    if (file or "").strip():
        max_lines = max(1, int(getattr(cfg, "max_read_lines", 200) or 200))
        return sessions.read(index.root, file, start, end, max_lines=max_lines)
    return sessions.recent(index.root)


def _snapshot(index, cfg, rel: str, path: Path, original: str,
              style) -> "str | None":
    """Snapshot `original` before a mutating action.

    Returns an error string if the snapshot was requested and failed; None on
    success or when backups are off. Shared by write/append/move/delete so the
    destructive actions all honour the same `backup` setting."""
    if not getattr(cfg, "backup", False):
        return None
    bdir = getattr(cfg, "_backup_dir", None)
    dest = (Path(bdir) / rel) if bdir else path.with_suffix(path.suffix + ".bak")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        fileio.write(dest, original, style)
        return None
    except OSError as e:
        return f"ERROR: could not back up {rel}: {e}"


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
    rel = fileio.normalise_rel(file)
    if not rel:
        return f"ERROR: {verb} needs a `file` path."
    if not (content or "").strip():
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

    existed = path.exists()
    original = None
    style = None
    if existed:
        original, style = fileio.read(path, errors="strict")

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
    if len(text) > getattr(cfg, "max_write_chars", 200_000):
        return (f"ERROR: the resulting file is {len(text):,} chars, over the "
                f"{cfg.max_write_chars:,} limit. Write a smaller file, or use "
                "edit blocks.")

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
    if append:
        added = len(new_text.splitlines())
        note = (f"APPENDED {added} line(s) to {rel} (now {n} lines total)."
                if existed else f"CREATED {rel} ({n} lines).")
        return (note + "\n[This change was applied directly, without a review "
                "page. Undo with `pedacito restore` if it was wrong. If the "
                "file is not complete yet, `append` the next chunk now; choose "
                "`done` only once every part of the task is written.]")

    note = (f"WROTE {rel} ({n} lines), replacing {len(original.splitlines())} "
            f"previous line(s)." if existed else f"CREATED {rel} ({n} lines).")
    budget = max(1, int(getattr(cfg, "max_write_lines", 0) or 0))
    if budget and n > budget:
        return (note + f"\n[Chunking is ON: one write step may emit at most "
                f"{budget} lines. This file needs more. Do NOT re-write the "
                f"whole file. Instead choose `append` with the next "
                f"<={budget} lines of {rel}; repeat `append` until complete, "
                f"then choose `done`.]"
                + "\n[This overwrite was applied directly, without a review "
                "page. Undo with `pedacito restore` if it was wrong.]")
    return (note + "\n[This overwrite was applied directly, without a review "
            "page. Undo with `pedacito restore` if it was wrong. Now choose "
            "`done` and summarise what you wrote.]")


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
    src_rel = index.resolve_file(file) or fileio.normalise_rel(file)
    dst_rel = fileio.normalise_rel(to)
    if not src_rel:
        return "ERROR: move needs `file` (the old path)."
    if not dst_rel:
        return "ERROR: move needs `to` (the new path)."
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
        original, style = fileio.read(src, errors="strict")
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
    rel = index.resolve_file(file) or fileio.normalise_rel(file)
    if not rel:
        return "ERROR: delete needs a `file` path."
    root = Path(index.root).resolve()
    path = (root / rel).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return f"ERROR: refusing to delete outside the project root: {file}"
    if not path.is_file():
        return f"ERROR: {rel} is not an existing file."
    try:
        original, style = fileio.read(path, errors="strict")
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
    undo = ("backed up; use `pedacito restore` to undo." if cfg.backup else
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


def tool_run(index, cfg, command: str) -> str:
    """Run one command in the project root and return its output.

    Opt-in (`allow_shell = false` by default) because this executes code the
    model wrote. The command is split into an argv and run WITHOUT a shell, so
    pipes/redirection/`&&` are inert arguments rather than executed syntax.
    `shell_allowlist`, when set, restricts the executable name. Output and
    runtime are both capped so a hung or chatty command cannot eat the task's
    context budget."""
    if not getattr(cfg, "allow_shell", False):
        return ("ERROR: `run` is disabled (config allow_shell = false). Enable "
                "it to run tests or linters from the gather loop, or run the "
                "command yourself.")
    cmd = (command or "").strip()
    if not cmd:
        return "ERROR: run needs a `command`, e.g. `pytest -q`."
    try:
        argv = _split_command(cmd)
    except ValueError as e:
        return f"ERROR: could not parse command ({e})."
    if not argv:
        return "ERROR: run needs a `command`."

    allow = tuple(getattr(cfg, "shell_allowlist", ()) or ())
    if allow:
        names = {Path(a).name.lower() for a in allow}
        exe = Path(argv[0]).name.lower()
        if exe not in names and argv[0].lower() not in {a.lower() for a in allow}:
            return (f"ERROR: `{argv[0]}` is not in shell_allowlist "
                    f"({', '.join(allow)}).")

    timeout = max(1, int(getattr(cfg, "shell_timeout", 120) or 120))
    env = dict(os.environ, NO_COLOR="1", TERM="dumb", GIT_PAGER="cat",
               PAGER="cat")
    try:
        proc = subprocess.run(argv, cwd=str(index.root), capture_output=True,
                              text=True, errors="replace", timeout=timeout,
                              env=env, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {timeout}s: {cmd}"
    except FileNotFoundError:
        return f"ERROR: command not found: {argv[0]}"
    except OSError as e:
        return f"ERROR: could not run command: {e}"

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

    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    cap = max(200, int(getattr(cfg, "max_shell_output_chars", 8000) or 8000))
    if len(out) > cap:
        out = out[:cap] + f"\n[output truncated to {cap} chars]"
    if not out:
        out = "(no output)"
    status = "OK" if proc.returncode == 0 else "FAILED"
    return f"RUN: {cmd}\nEXIT: {proc.returncode} ({status})\nOUTPUT:\n{out}"


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------
def dispatch(index, cfg, step: dict, client=None) -> str:
    """Route one parsed agent step (matching STEP_SCHEMA) to the
    corresponding tool_* function and return its text observation.

    `client` is only needed for optional semantic search; every other action
    works without it, so offline tests and CLI inspection callers can omit it.
    """
    action = (step.get("action") or "").strip()
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
    if action == RUN_ACTION:
        return tool_run(index, cfg, step.get("command", ""))
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
        budget = max(1, int(getattr(cfg, "max_write_lines", 0) or 0))
        content = step.get("content", "") or ""
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
