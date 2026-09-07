"""The agent's tool surface: read, grep, callers, outline, done.

Usage
-----
    from .tools import STEP_SCHEMA, TOOL_HELP, dispatch

    step = client.chat_json(messages, STEP_SCHEMA, max_tokens=700)
    observation = dispatch(index, cfg, step)

Deliberately a small, fixed set of tools. Local models degrade fast as the
tool surface grows -- four or five is about the ceiling before a small model
starts calling the wrong one or inventing arguments. Every tool returns
plain text with line numbers attached, since the model needs those numbers
to write search/replace edits later.

STEP_SCHEMA is the JSON schema used with LM Studio's grammar-constrained
`response_format`, so the model's tool choice always parses. See agent.py
for how it is used in the gather loop.
"""

from __future__ import annotations

import re
from pathlib import Path

from . import fileio

TOOL_HELP = """\
read     - read exact lines of a file.        args: file, start, end
grep     - regex search. args: pattern, and file to search one file only
callers  - who calls a function or method.    args: symbol
outline  - list every unit in one file.       args: file
done     - stop gathering; you have enough.   args: none"""

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
            "enum": ["read", "grep", "callers", "outline", "done"],
        },
        "file": {"type": "string", "description": "For read/outline. Empty otherwise.",
                 "maxLength": 200},
        "start": {"type": "integer", "description": "For read. 0 otherwise."},
        "end": {"type": "integer", "description": "For read. 0 otherwise."},
        "pattern": {"type": "string", "description": "For grep. Empty otherwise.",
                    "maxLength": 200},
        "symbol": {"type": "string", "description": "For callers. Empty otherwise.",
                   "maxLength": 200},
        "reasoning": {
            "type": "string",
            "description": "Under 15 words: what you still need. Not a plan.",
            "maxLength": 160,
        },
    },
    # Flat + all-required beats nested oneOf/anyOf for small models: the grammar
    # stays simple and the model never has to pick a branch.
    "required": ["action", "file", "start", "end", "pattern", "symbol", "reasoning"],
    "additionalProperties": False,
}


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


# --------------------------------------------------------------------------
# Tool implepedacitoions
# --------------------------------------------------------------------------
def tool_read(index, cfg, file: str, start: int, end: int) -> str:
    """Return exact, line-numbered source text for `file` between `start`
    and `end` (1-based, inclusive). This is the only tool whose output is
    safe to quote verbatim in a search/replace edit."""
    rel = index.resolve_file(file)
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
    if scope:
        one = index.resolve_file(scope)
        if one:
            files = [one]
    hits: list[str] = []
    for rel in files:
        path = index.root / rel
        if not path.exists():
            continue
        for i, line in enumerate(fileio.read_lines(path), 1):
            if rx.search(line):
                # Indentation is preserved: a stripped line looks editable and
                # is not, which is how a model ends up writing SEARCH blocks
                # that can never match.
                shown = line[:200].rstrip()
                cut = " ...[line truncated]" if len(line) > 200 else ""
                hits.append(f"{rel}:{i}: {shown}{cut}")
                if len(hits) >= max_hits:
                    hits.append(f"[stopped at {max_hits} hits; narrow the pattern]")
                    return GREP_WARNING + "\n" + "\n".join(hits)
    if not hits:
        return f"No matches for /{pattern}/."
    return GREP_WARNING + "\n" + "\n".join(hits)


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
        return (f"ERROR: '{file}' is not in the index. Indexed files: "
                + ", ".join(index.files))
    return index.render_outline(rel)


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
    return f"ERROR: unknown action '{action}'."
