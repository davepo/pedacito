"""The agent's tool surface: read, grep, callers, outline, write, done.

Usage
-----
    from .tools import STEP_SCHEMA, TOOL_HELP, dispatch

    step = client.chat_json(messages, STEP_SCHEMA, max_tokens=700)
    observation = dispatch(index, cfg, step)

Deliberately a small, fixed set of tools. Local models degrade fast as the
tool surface grows -- four to six is about the ceiling before a small model
starts calling the wrong one or inventing arguments. Every tool returns
plain text with line numbers attached, since the model needs those numbers
to write search/replace edits later.

`write` is the one action that changes the project rather than reading it, and
it is off by default (`cfg.allow_write`): a mid-gather overwrite skips the edit
path's syntax validation, backup, and review page entirely. It exists for the
case where an existing file must be replaced wholesale -- which a 9B model
often cannot express as a search/replace block -- and the gate keeps that
trade-off visible rather than silent.

STEP_SCHEMA is the JSON schema used with LM Studio's grammar-constrained
`response_format`, so the model's tool choice always parses. See agent.py
for how it is used in the gather loop.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from . import fileio

TOOL_HELP = """\
read     - read exact lines of a file.       args: file, start, end
grep     - regex search. args: pattern, and file to search one file only
callers  - who calls a function or method.   args: symbol
outline  - list every unit in one file.      args: file
write    - overwrite a whole file with text. args: file, content
           (disabled unless allow_write is on; prefer edit blocks)
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
            "enum": ["read", "grep", "callers", "outline", "write", "done"],
        },
        "file": {"type": "string", "description": "For read/outline/write. Empty otherwise.",
                 "maxLength": 200},
        "start": {"type": "integer", "description": "For read. 0 otherwise."},
        "end": {"type": "integer", "description": "For read. 0 otherwise."},
        "pattern": {"type": "string", "description": "For grep. Empty otherwise.",
                    "maxLength": 200},
        "symbol": {"type": "string", "description": "For callers. Empty otherwise.",
                   "maxLength": 200},
        "content": {"type": "string",
                    "description": "For write: the complete new file text. Empty otherwise.",
                    "maxLength": 16000},
        "reasoning": {
            "type": "string",
            "description": "Under 15 words: what you still need. Not a plan.",
            "maxLength": 160,
        },
    },
    # Flat + all-required beats nested oneOf/anyOf for small models: the grammar
    # stays simple and the model never has to pick a branch.
    "required": ["action", "file", "start", "end", "pattern", "symbol",
                 "content", "reasoning"],
    "additionalProperties": False,
}

# The action list is kept in one place so the schema, the gate in `dispatch`,
# and the prompt help cannot drift apart.
ACTIONS = ("read", "grep", "callers", "outline", "write", "done")
WRITE_ACTION = "write"


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


def tool_read(index, cfg, file: str, start: int, end: int) -> str:
    """Return exact, line-numbered source text for `file` between `start`
    and `end` (1-based, inclusive). This is the only tool whose output is
    safe to quote verbatim in a search/replace edit."""
    rel = index.resolve_file(file)
    if rel is None:
        # Files created after the last indexing run are real, readable, and
        # invisible to the index. Fold them in rather than losing the step.
        rel = adopt_if_on_disk(index, cfg, file)
    if rel is None:
        return (f"ERROR: '{file}' is not in the index. Indexed files: "
                + ", ".join(index.files))
    path = index.root / rel
    if not path.exists():
        return f"ERROR: {rel} missing on disk."
    lines = fileio.read_lines(path)

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
    return f"{rel} lines {start}-{end}:\n{body}{note}"


def tool_grep(index, pattern: str, scope: str = "", max_hits: int = 40) -> str:
    """Regex search across every indexed file, or one file if `scope` is
    given. Returns matching lines with their file:line locations, prefixed
    with a warning that this output is for locating code, not for quoting
    into an edit (lines may be truncated to 200 chars)."""
    if not (pattern or "").strip():
        return ("ERROR: grep needs a `pattern` (a regex). To look inside one file, "
                "use `outline`, or `read` it.")
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"ERROR: bad regex ({e})."
    files = index.files
    scope_note = ""
    if scope:
        one = index.resolve_file(scope)
        if one:
            files = [one]
        elif pattern.strip():
            # A scoped grep that names a file outside the index is almost
            # never a mistake worth a wasted step: the pattern was the point,
            # so run it across everything and say the scope did not apply.
            scope_note = (f"[note: '{scope}' is not in the index; searched all "
                          f"{len(files)} indexed files instead]")
        else:
            # No usable pattern *and* no real file: the step was malformed.
            return (f"ERROR: '{scope}' is not in the index, and grep needs a "
                    "`pattern`. To look inside one file, use `outline`, or `read` it.")
    hits: list[str] = []
    per_file_cap = max(5, max_hits // 4)
    for rel in files:
        path = index.root / rel
        if not path.exists():
            continue
        file_hits = 0
        for i, line in enumerate(fileio.read_lines(path), 1):
            if rx.search(line):
                if file_hits >= per_file_cap:
                    hits.append(f"{rel}: [more matches in this file; "
                                f"grep it directly with `file` set]")
                    break
                # Indentation is preserved: a stripped line looks editable and
                # is not, which is how a model ends up writing SEARCH blocks
                # that can never match.
                shown = line[:200].rstrip()
                cut = " ...[line truncated]" if len(line) > 200 else ""
                hits.append(f"{rel}:{i}: {shown}{cut}")
                file_hits += 1
                if len(hits) >= max_hits:
                    hits.append(f"[stopped at {max_hits} hits; narrow the pattern]")
                    return _grep_result(scope_note, hits)
    if not hits:
        return f"No matches for /{pattern}/."
    return _grep_result(scope_note, hits)


def tool_callers(index, symbol: str) -> str:
    """Report where a symbol is defined and, by matching bare call names,
    which chunks appear to call it. An approximate cross-reference, not a
    sound one (see Index.callers_of in index.py)."""
    if not symbol.strip():
        return "ERROR: symbol is required."
    defs = index.find_symbol(symbol)
    out: list[str] = []
    if defs:
        out.append("Defined at:")
        out += [f"  {c.file}:{c.start}-{c.end}  {c.signature or c.qualname}" for c in defs]
    callers = [c for c in index.callers_of(symbol) if c not in defs]
    if callers:
        out.append("Called from:")
        out += [f"  {c.file}:{c.start}-{c.end}  {c.qualname}" for c in callers]
    else:
        out.append(f"No callers of '{symbol}' found in the indexed files.")
    return "\n".join(out)


def tool_outline(index, file: str) -> str:
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
        return (f"ERROR: '{file}' is not in the index. Indexed files: "
                + ", ".join(index.files))
    return index.render_outline(rel)


def tool_write(index, cfg, file: str, content: str) -> str:
    """Replace a whole file with `content`, backing the original up first.

    Only reachable when `cfg.allow_write` is on (see dispatch). Deliberately
    minimal: it reuses the same safety rails as the edit path -- refuse to
    escape the project root, snapshot the original into the run's backup dir,
    preserve the file's encoding style -- but it cannot offer a review diff, so
    the observation tells the model what it overwrote and points at `restore`.
    New files are created; an empty write against an existing file is refused
    rather than silently truncating it."""
    rel = fileio.normalise_rel(file)
    if not rel:
        return "ERROR: write needs a `file` path."
    if not (content or "").strip():
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
        if not original.strip() and not content.strip():
            return f"ERROR: {rel} is already empty."

    text = content.replace("\r\n", "\n").replace("\r", "\n")
    if not text.endswith("\n"):
        text += "\n"
    if len(text) > getattr(cfg, "max_write_chars", 200_000):
        return (f"ERROR: content is {len(text):,} chars, over the "
                f"{cfg.max_write_chars:,} limit. Write a smaller file, or use "
                "edit blocks.")

    if existed and cfg.backup:
        bdir = getattr(cfg, "_backup_dir", None)
        dest = (Path(bdir) / rel) if bdir else path.with_suffix(path.suffix + ".bak")
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            fileio.write(dest, original, style)
        except OSError as e:
            return f"ERROR: could not back up {rel}: {e}"

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
    note = (f"WROTE {rel} ({n} lines), replacing {len(original.splitlines())} "
            f"previous line(s)." if existed else f"CREATED {rel} ({n} lines).")
    return (note + "\n[This overwrite was applied directly, without a review "
            "page. Undo with `pedacito restore` if it was wrong. Now choose "
            "`done` and summarise what you wrote.]")


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------
def dispatch(index, cfg, step: dict) -> str:
    """Route one parsed agent step (matching STEP_SCHEMA) to the
    corresponding tool_* function and return its text observation."""
    action = (step.get("action") or "").strip()
    if action == "read":
        return tool_read(index, cfg, step.get("file", ""), step.get("start", 1), step.get("end", 0))
    if action == "grep":
        return tool_grep(index, step.get("pattern", ""), step.get("file", ""))
    if action == "callers":
        return tool_callers(index, step.get("symbol", ""))
    if action == "outline":
        return tool_outline(index, step.get("file", ""))
    if action == WRITE_ACTION:
        if not getattr(cfg, "allow_write", False):
            return ("ERROR: `write` is disabled (config allow_write = false). "
                    "Answer the task with search/replace edit blocks instead -- "
                    "`read` the lines you want to change, then emit them in your "
                    "final reply.")
        return tool_write(index, cfg, step.get("file", ""), step.get("content", ""))
    return f"ERROR: unknown action '{action}'."
