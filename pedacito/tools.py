"""The agent's tool surface: read, grep, callers, outline, sessions, diff, write/append, run, screenshot, browser, done.

    step = client.chat_json(messages, STEP_SCHEMA, max_tokens=700)
    observation = dispatch(index, cfg, step)

The set is deliberately small: local models start calling the wrong tool or
inventing arguments beyond roughly half a dozen. Navigation tools (read, grep,
callers, outline, sessions, diff, done) are always on; everything destructive
or machine-visible is gated by config: `write`/`append`/`move`/`delete` need
`allow_write`, `run`/`diagnose` need `allow_shell`, `screenshot` needs
`allow_screenshots`, `browser` needs `allow_browser` (launching a command also
needs `allow_shell`), `mcp` needs `allow_mcp`, and `diff` is dropped by
`allow_diff = false`. Mutating actions skip the edit path's syntax validation
and review page, so they are for wholesale replace/grow/move/remove of files a
small model cannot express as a search/replace block.

Every tool returns plain text with line numbers, since the model needs them to
write search/replace edits. Navigation tools are forgiving of argument slips:
`read` accepts a symbol name, `grep` falls back to literal text for an invalid
regex, retries case-insensitively, accepts a directory or glob scope, adds
context for sparse hits, suggests similar symbols on a miss, and (with
`semantic_search`) queries the server's embeddings endpoint when nothing
matches textually.

STEP_SCHEMA is the JSON schema used for the model server's grammar-constrained
`response_format`, so the model's tool choice always parses.
"""

from __future__ import annotations

import ast
import copy
import difflib
import fnmatch
import json
import os
import re
import shlex
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path, PureWindowsPath

from . import browser, fileio, proc, sessions, vision

TOOL_HELP = """\
read     - read exact lines of a file.       args: file, start, end
           (or set symbol and leave file empty: read that definition)
grep     - regex or literal search. args: pattern, and optional file/dir/glob
           (pattern = an existing filename returns its references/imports;
            with semantic_search on, falls back to search by meaning)
find     - search indexed definitions by name, signature, or summary.
           args: pattern (a fragment like 'retry' or 'Server.start'). Hits
           carry exact line ranges; no regex, no source-text guessing
callers  - who calls a symbol, with the call sites. args: symbol
outline  - list every unit in one file, or the file cards in one directory.
           args: file (a path, a symbol, or a directory/glob)
sessions - list/search what prior runs did.  args: pattern, optional file
diff     - what changed since Pedacito's last snapshot. args: optional file
diagnose - run the project's detected linter/typechecker. args: optional file
todo     - replace the task checklist shown in later prompts. args: content
           (one "- [ ]" / "- [x]" line per step; keep it short)
mcp      - call an external MCP tool (opt-in). args: symbol (server/tool or
           "server" to list), content (JSON arguments object)
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
           pattern (optional "press Enter; click #start; wait 2; eval <js>";
           `eval` goes last and consumes the rest of the pattern; its result
           comes back clipped at 2000 chars -- resolve compact JSON),
           start (seconds to wait), optional file (old shot to compare)
           (needs allow_browser; launching a command also needs allow_shell)
           (a desktop screenshot cannot see a page that was never opened)
           (write/append/move/delete need allow_write; run and diagnose need
            allow_shell; mcp needs allow_mcp)
done     - stop gathering; you have enough.  args: none"""

# Field order is load-bearing: grammar-constrained decoding emits keys in
# schema order, so a reply cut off by the token limit loses only the trailing
# `reasoning`, not the action.
STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["read", "grep", "find", "callers", "outline", "sessions",
                     "diff", "diagnose", "write", "append", "move", "delete",
                     "run", "screenshot", "browser", "todo", "mcp", "done"],
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
                                   "valid regex. For find: a name/signature/"
                                   "summary fragment. For browser: optional "
                                   "actions like 'press Enter; click #start; wait 2; "
                                   "eval <js>' -- eval goes LAST and consumes "
                                   "the rest (semicolons allowed inside); its "
                                   "RESULT is reported clipped at 2000 chars, "
                                   "so resolve compact or aggregated JSON. "
                                   "Empty otherwise.",
                    # Constrained sampling force-closes a string AT maxLength
                    # (valid JSON, no truncation flag), so a cap shorter than
                    # a real eval silently runs a fragment. agent.gather
                    # refuses to dispatch a payload arriving at exactly its cap.
                    "maxLength": 2000},
        "symbol": {"type": "string",
                   "description": "For callers; or for read when file is empty "
                                  "(reads that definition). For mcp: "
                                  "server/tool, or server to list its tools. "
                                  "Empty otherwise.",
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
                                   "several steps). For todo: the full checklist, "
                                   "one '- [ ]'/'- [x]' line per step. For mcp: "
                                   "a JSON object of tool arguments. Empty otherwise.",
                    "maxLength": 6000},
        "reasoning": {
            "type": "string",
            "description": "Under 15 words: what you still need. Not a plan.",
            "maxLength": 160,
        },
    },
    # Flat + all-required keeps the grammar simple for small models.
    # `content`'s maxLength MUST stay below llm._BIG_FIELD_MIN (8000): that
    # threshold marks a schema "step-style" (a salvaged action is accepted)
    # rather than "answer-style" (cut mid-content => partial). llm.chat_json
    # records which field a cut landed in (`truncated_field`) and agent.gather
    # refuses to dispatch a step whose payload was cut. File-size limits are
    # enforced by max_write_lines in dispatch(), not here.
    "required": ["action", "file", "start", "end", "pattern", "symbol",
                 "to", "command", "content", "reasoning"],
    "additionalProperties": False,
}

WRITE_ACTION = "write"
APPEND_ACTION = "append"
MOVE_ACTION = "move"
DELETE_ACTION = "delete"
RUN_ACTION = "run"
SCREENSHOT_ACTION = "screenshot"
BROWSER_ACTION = "browser"
DIFF_ACTION = "diff"
DIAGNOSE_ACTION = "diagnose"
TODO_ACTION = "todo"
MCP_ACTION = "mcp"
WRITE_ACTIONS = (WRITE_ACTION, APPEND_ACTION)
MUTATING_ACTIONS = WRITE_ACTIONS + (MOVE_ACTION, DELETE_ACTION)


def schema_for(cfg) -> dict:
    """Return STEP_SCHEMA narrowed to the actions this run allows, so the
    model cannot pick a disabled action. Fields only a disabled action uses
    (`to`, `command`) are dropped too."""
    actions = ["read", "grep", "find", "callers", "outline", "sessions", "todo"]
    if getattr(cfg, "allow_diff", True):
        actions.append("diff")
    if getattr(cfg, "allow_shell", False):
        actions += ["run", "diagnose"]
    if getattr(cfg, "allow_write", False):
        actions += ["write", "append", "move", "delete"]
    if getattr(cfg, "allow_screenshots", False):
        actions.append("screenshot")
    if getattr(cfg, "allow_browser", False):
        actions.append("browser")
    if getattr(cfg, "allow_mcp", False):
        actions.append("mcp")
    actions.append("done")

    schema = copy.deepcopy(STEP_SCHEMA)
    schema["properties"]["action"]["enum"] = actions
    required = ["action", "file", "start", "end", "pattern", "symbol",
                "reasoning"]
    drop = set()
    if not getattr(cfg, "allow_write", False):
        # `content` stays: the always-available todo action uses it. `to` is
        # only ever used by move.
        drop.add("to")
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
        # Experimental: only `action` and `reasoning` stay required, so empty
        # unused fields may be omitted (saves ~20-40 output tokens per step).
        required = ["action", "reasoning"]
    ordered = ["action", "file", "start", "end", "pattern", "symbol",
               "to", "command", "content", "reasoning"]
    schema["properties"] = {k: schema["properties"][k] for k in ordered
                            if k in schema["properties"]}
    schema["required"] = [k for k in ordered if k in required]
    return schema


def _as_text(value) -> str:
    """String coercion for model-supplied step fields (a salvaged step can
    carry null/number/list values; one bad field must not kill the run)."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _as_int(value, default: int = 0) -> int:
    """Integer coercion for model-supplied fields ("12", 12.0, null, prose)
    that never raises."""
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError, OverflowError):
            return default


def _root_rel(index, name) -> str | None:
    """Resolve a model-supplied path to a project-relative POSIX path, or None
    when it is empty or escapes the root (`..`, absolute, or via a symlink).
    This is the single gate for paths the tools touch."""
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
    """Render lines as right-aligned "NNN | text", numbered from `start`."""
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


def _symbol_matches(index, symbol: str, rel: str | None = None) -> list:
    """Definitions matching `symbol`, optionally restricted to one file."""
    try:
        matches = index.find_symbol(_as_text(symbol).strip())
    except Exception:
        return []
    if rel:
        matches = [c for c in matches if c.file == rel]
    return matches


def _file_hint(index, name: str) -> str:
    """Compact 'file not found' message: close matches plus a few real paths."""
    name = _as_text(name)
    files = index.files
    if not files:
        return f"ERROR: '{name}' is not in the index (the index is empty)."
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
    """Resolve a grep scope (a file, a directory like `src/pkg`, or a glob like
    `tests/*.py`) to (files, note). An unresolvable scope falls back to the
    whole index, with a note saying so."""
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
    """One formatted grep hit: `rel:lineno: [symbol] text`, line trimmed to 200
    chars. Indentation is preserved so the model does not treat a stripped
    line as quotable in a SEARCH block."""
    shown = line[:200].rstrip()
    cut = " ...[line truncated]" if len(line) > 200 else ""
    unit = _enclosing_chunk(index, rel, i)
    where = (f" [{unit.qualname}]" if unit is not None
             and unit.qualname != "module" else "")
    return f"{rel}:{i}:{where} {shown}{cut}"


# A grep with at most this many hits gets one line of context around each.
SPARSE_CONTEXT_MAX = 5
GREP_MAX_HITS = 40


def _context_line(index, rel: str, n: int) -> str | None:
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
               rx_ci: re.Pattern | None = None) -> tuple[list, list]:
    """Search `files` with `rx`, collecting case-insensitive hits for `rx_ci`
    in the same pass.

    Returns (entries, ci_entries): lists of (rel, lineno, text) tuples, with
    lineno None for cap/stop notes. ci_entries only matters when `entries`
    has no real hits."""
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
                    break    # real hits exist, so the ci list is discarded
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
    """'Did you mean this symbol?' text for a grep that found nothing:
    substring matches first, then difflib close matches; '' when none."""
    p = (pattern or "").strip()
    if len(p) < 3:
        return ""
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


def _fresh_index(index, cfg, rel: str):
    """Re-chunk a file that exists on disk but is not in the index yet; None
    if nothing could be indexed."""
    rel = _root_rel(index, rel)
    if not rel:
        return None
    try:
        from . import index as index_mod
        return index_mod.reindex_file(index, rel, cfg, save=False)
    except Exception:
        return None


def adopt_if_on_disk(index, cfg, file: str) -> str | None:
    """If `file` exists on disk inside the project but is missing from the
    index, fold it in and return its indexed path; otherwise None."""
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
    """Definition locations for a symbol defined more than once."""
    out = [f"'{symbol}' has {len(matches)} definitions; choose one with "
           f"`read` file+start+end:"]
    for c in matches[:20]:
        out.append(f"  {c.file}:{c.start}-{c.end}  {c.signature or c.qualname}")
    if len(matches) > 20:
        out.append(f"  (+{len(matches) - 20} more)")
    return "\n".join(out)


def tool_read(index, cfg, file: str, start: int, end: int,
              symbol: str = "") -> str:
    """Return exact, line-numbered source for `file` between `start` and `end`
    (1-based, inclusive): the only tool output safe to quote in an edit.

    When `file` is empty and `symbol` is given, read that symbol's definition
    (an ambiguous name returns the definition list)."""
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
            rel = adopt_if_on_disk(index, cfg, file)
        if rel is None and not symbol:
            # A symbol name in `file` is treated as a symbol read.
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
        # Symbol-only reads ignore any default start/end; with a file too,
        # only a placeholder range is overridden.
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
    # Drop the phantom empty line a trailing newline leaves after split.
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
        # Backwards range: read forward from `start` instead of failing.
        end = min(start + max_lines - 1, len(lines))
    if end - start + 1 > max_lines:
        end = start + max_lines - 1
        note = f"\n[truncated to {max_lines} lines; read again from {end + 1} if needed]"
    else:
        note = ""
    # Mid-file reads: surface the enclosing signature and module imports.
    context = ""
    if not symbol_note and start > 1 and getattr(cfg, "read_context_hint", True):
        chunks = index.in_file(rel)
        for c in chunks:
            if c.start <= start <= c.end:
                bits = []
                if c.kind in ("function", "method", "class", "classheader") \
                        and c.signature:
                    bits.append(c.signature)
                # A method's signature alone lacks its class context; show
                # the owner's header too (qualnames are "Class.method", so
                # the owner is the dotted prefix).
                if c.kind == "method" and "." in c.qualname:
                    owner = c.qualname.rsplit(".", 1)[0]
                    for k in chunks:
                        if k.qualname == owner \
                                and k.kind in ("class", "classheader") \
                                and k.signature:
                            bits.append("in " + k.signature)
                            break
                imports = sorted({i.split(".")[0].lstrip(".")
                                  for i in c.imports})
                if imports:
                    bits.append("module imports: " + ", ".join(imports[:10]))
                if bits:
                    context = "\n[context: " + "; ".join(bits) + "]"
                break
    body = _numbered(lines[start - 1 : end], start)
    return f"{rel} lines {start}-{end}:{context}\n{body}{note}{symbol_note}"


_REF_IMPORT_RES = (
    re.compile(r"^\s*(?:from|import)\b"),                       # Python
    re.compile(r"^\s*(?:import|export)\b.*\bfrom\b"),           # JS/TS
    re.compile(r"\brequire\s*\("),                              # CommonJS
    re.compile(r"<(?:script|link|img|a|iframe)\b[^>]*\b"
               r"(?:src|href)\s*=", re.IGNORECASE),             # HTML
)


def _file_reference_target(index, pattern: str, scope: str) -> str | None:
    """If `pattern` names an indexed file and no scope was given, return its
    indexed path: `grep X.py` is how models ask for references to X.py."""
    if scope.strip():
        return None
    raw = (pattern or "").strip()
    if not raw or not re.fullmatch(r"[A-Za-z0-9_./\\-]+", raw):
        return None
    name = fileio.normalise_rel(raw).strip()
    name = name.removeprefix("./")
    if not name or ".." in Path(name).parts:
        return None
    rel = index.resolve_file(name)
    if rel:
        return rel
    # Accept a unique path suffix too (models drop or add a directory).
    matches = [f for f in index.files
               if f.lower() == name.lower()
               or f.lower().endswith("/" + name.lower())]
    return matches[0] if len(matches) == 1 else None


def _file_references(index, rel: str) -> str:
    """Report textual references to one file (imports and mentions): evidence,
    not proof of liveness, since dynamic imports never name the file."""
    base = Path(rel).name.lower()
    full = rel.lower()
    stem = Path(rel).stem.lower()
    stem_rx = re.compile(r"\b" + re.escape(stem) + r"\b") if stem else None
    hits: list[tuple[str, int, str, str]] = []
    code_files: set[str] = set()
    for frel in index.files:
        if frel == rel:
            continue
        try:
            lines = fileio.read_lines_cached(index.root / frel)
        except OSError:
            continue
        is_doc = Path(frel).suffix.lower() in {".md", ".rst", ".txt"}
        for lineno, line in enumerate(lines, 1):
            low = line.lower()
            matched = base in low or full in low
            import_like = None
            if not matched:
                import_like = any(rx.search(line) for rx in _REF_IMPORT_RES)
                if not (import_like and stem_rx and stem_rx.search(low)):
                    continue
            if import_like is None:
                import_like = any(rx.search(line) for rx in _REF_IMPORT_RES)
            kind = "doc" if is_doc else ("import" if import_like else "text")
            hits.append((frel, lineno, kind, line.strip()[:200]))
            if not is_doc:
                code_files.add(frel)
            if len(hits) >= 80:
                break
        if len(hits) >= 80:
            break

    code_count = sum(1 for _f, _n, kind, _t in hits if kind != "doc")
    doc_count = len(hits) - code_count
    header = f"References to {rel} (textual evidence, not proof):"
    if not hits:
        return (f"No references to {rel} found in {len(index.files)} indexed "
                "file(s).\n[It is a dead-code candidate -- check dynamic "
                "imports and string-built paths before removing it.]")
    if not code_count:
        return (f"No CODE references to {rel} found in {len(index.files)} "
                f"indexed file(s); only {doc_count} documentation "
                "mention(s):\n"
                + "\n".join(f"  {frel}:{lineno}  {text}"
                            for frel, lineno, _k, text in hits)
                + "\n[It is a dead-code candidate -- check dynamic imports "
                  "and string-built paths before removing it.]")
    out = [header]
    for frel, lineno, kind, text in hits:
        out.append(f"  {frel}:{lineno}  {kind:<6} {text}")
    if len(hits) >= 80:
        out.append("  (+more; narrow the search with a scope)")
    out.append(f"Found {code_count} code reference(s) in {len(code_files)} "
               f"file(s)" + (f" (+{doc_count} documentation mention(s))"
                             if doc_count else "") + ".")
    out.append(f"[To search INSIDE {rel}, call grep with pattern=<text> and "
               f"file=\"{rel}\".]")
    return "\n".join(out)


def tool_grep(index, pattern: str, scope: str = "", cfg=None,
              client=None) -> str:
    """Regex search across every indexed file, or a narrower `scope` (a file,
    directory or glob; an unindexed file on disk is adopted first).

    Returns `file:line: [symbol] text` hits (lines trimmed to 200 chars) under
    a warning not to quote them in edits. An invalid regex is searched as
    literal text; a case-sensitive miss falls back to case-insensitive hits;
    a sparse result gets one context line per hit; a total miss tries
    semantic search (if enabled) and similar symbol names. A `pattern` that
    is itself an indexed file path (no scope) returns that file's references.
    """
    pattern = _as_text(pattern)
    scope = _as_text(scope)
    max_hits = GREP_MAX_HITS
    if not pattern.strip():
        return ("ERROR: grep needs a `pattern` (a regex or plain text). To look "
                "inside one file, use `outline`, or `read` it.")
    ref_target = _file_reference_target(index, pattern, scope)
    if ref_target:
        return _file_references(index, ref_target)
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

    if scope and cfg is not None and index.resolve_file(scope) is None:
        adopt_if_on_disk(index, cfg, scope)
    if scope:
        # Normalise an absolute/`./`/`..` scope to the index's spelling.
        safe_scope = _root_rel(index, scope)
        if safe_scope is not None:
            scope = safe_scope

    files, scope_note = _grep_scope(index, scope)
    if scope_note:
        notes.append(scope_note)
    if not files:
        return f"No indexed files to search for /{pattern}/."

    rx = re.compile(needle)
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
        # Last resort: find code by meaning.
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
        suggest = _symbol_suggestions(index, pattern)
        if suggest:
            notes.append(suggest)

    note_text = ("\n".join(notes) + "\n") if notes else ""
    if not hits:
        return note_text + f"No matches for /{pattern}/."
    return _grep_result("\n".join(notes), hits)


def _call_sites(index, chunk, base: str, limit: int = 2) -> list[str]:
    """Up to `limit` source lines inside `chunk` that mention `base`, so the
    model can pick a caller without reading each."""
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
    """Definitions the chunk calls that exist in the index, with locations."""
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
    """Report where a symbol is defined, what it calls, and which chunks appear
    to call it (bare-name matching; see Index.callers_of), with call sites."""
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
        importers = index.importers_of(defs[0].file)
        if importers:
            out.append("Imported by (no direct bare-name call found):")
            out += [f"  {f}" for f in importers[:12]]
            if len(importers) > 12:
                out.append(f"  (+{len(importers) - 12} more)")
    else:
        return _symbol_hint(index, symbol)
    return "\n".join(out)


# One line per hit; a broad pattern must not eat the whole gather budget.
FIND_MAX_HITS = 40


def tool_find(index, pattern: str) -> str:
    """Search indexed definitions by name, signature, or summary: navigation
    by concept, where grep needs the exact source spelling. Case-insensitive
    substring; name hits rank above signature hits above summary hits."""
    raw = _as_text(pattern).strip()
    q = raw.lower()
    if not q:
        return ("ERROR: `find` needs a pattern: a name fragment ('retry'), a "
                "qualified name ('Server.start'), or a concept word likely "
                "in a summary.")
    scored: list[tuple[int, str, int, object]] = []
    for c in index.chunks:
        if c.kind in ("module", "reference") \
                or c.qualname in ("module", "__trailing__"):
            continue
        qual = c.qualname.lower()
        # Methods of a class indexed whole live in its `members` list.
        member_hit = any(q in f"{qual}.{m.lower()}" for m in c.members)
        if q in qual or member_hit:
            rank = 0 if qual == q or qual.endswith("." + q) else 1
        elif q in (c.signature or "").lower():
            rank = 2
        elif q in f"{c.summary} {c.docline}".lower():
            rank = 3
        else:
            continue
        scored.append((rank, c.file, c.start, c))
    if not scored:
        names = sorted({c.qualname for c in index.chunks
                        if c.qualname not in ("module", "__trailing__")})
        close = difflib.get_close_matches(raw, names, n=5, cutoff=0.5)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        return (f"No definition matches '{raw}'.{hint} Try a shorter name "
                "fragment, or `grep` for the text itself.")
    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    out = [f"{len(scored)} definition(s) matching '{raw}' -- line ranges are "
           "exact; `read` them directly:"]
    for _rank, rel, _start, c in scored[:FIND_MAX_HITS]:
        sig = (c.signature or c.qualname)[:100]
        note = f" -- {c.summary}" if c.summary else ""
        end = c.scope_end or c.end    # a classheader stands for its class
        out.append(f"  {rel}:{c.start}-{end} [{c.kind}] {sig}{note}")
    if len(scored) > FIND_MAX_HITS:
        out.append(f"  (+{len(scored) - FIND_MAX_HITS} more; narrow the pattern)")
    return "\n".join(out)


def tool_outline(index, cfg, file: str) -> str:
    """Return every unit in one file with its signature and line range, or,
    when `file` names a directory or glob, the file cards under it."""
    file = _as_text(file)
    rel = index.resolve_file(file)
    if rel is None:
        rel = adopt_if_on_disk(index, cfg, file)
    if rel is None:
        # A symbol name outlines the file that defines it.
        hits = _symbol_matches(index, file)
        if hits:
            rel = hits[0].file
    if rel is None:
        # A directory or glob lists file cards; "." lists the whole project.
        name = fileio.normalise_rel(file or "").strip()
        if name in (".", "./"):
            return _outline_dir(index, ".", list(index.files))
        if name:
            matches, note = _grep_scope(index, name)
            if not note and matches:
                # No note means `name` resolved; a miss falls through to the
                # file-not-found hint instead of dumping every card.
                if len(matches) == 1:
                    rel = matches[0]
                else:
                    return _outline_dir(index, name, matches)
    if rel is None:
        return _file_hint(index, file)
    return index.render_outline(rel)


# Bounds for `outline <directory>`.
_OUTLINE_DIR_MAX_CARDS = 40
_OUTLINE_DIR_MAX_CHARS = 8000


def _outline_dir(index, name: str, matches: list[str]) -> str:
    """File cards for one directory/glob, bounded by the _OUTLINE_DIR_* caps."""
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
    """List recent session logs (no args), search them (`pattern`, optionally
    one log via `file`), or read one log's numbered lines (`file` + range)."""
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
    """(rel, stamp, snapshot_path) for every backed-up file, newest snapshot
    copy per file."""
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
    """Unified diff of one file: its snapshot copy vs the working tree. Pure
    difflib, no `git` subprocess (attribute filters can execute code)."""
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
    """Show what changed in the working tree since Pedacito's backup snapshots
    (`.pedacito/backups/<stamp>/`): every changed file, or one full diff when
    `file` is given. Gated by `allow_diff`."""
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
    # No file: changed files only, on a char budget (~40 diff lines each).
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


def tool_todo(index, cfg, content: str = "") -> str:
    """Read or replace the task checklist shown in later prompts (kept in
    `.pedacito/todo.md`, capped, cleared at the start of each task)."""
    from . import todo as todo_mod
    text = _as_text(content).strip()
    if not text:
        current = todo_mod.read(index.root)
        return ("CURRENT CHECKLIST:\n" + current if current else
                "No checklist is set. Pass content with one '- [ ]' / "
                "'- [x]' line per step to create one.")
    saved = todo_mod.write(index.root, text)
    if not saved:
        return "ERROR: could not write the checklist under .pedacito/."
    return "CHECKLIST UPDATED:\n" + saved


def tool_diagnose(index, cfg, file: str = "") -> str:
    """Run the project's detected linter/typechecker (see diagnostics.py),
    optionally on one file. Gated by allow_shell."""
    if not getattr(cfg, "allow_shell", False):
        return ("ERROR: `diagnose` is disabled (config allow_shell = false). "
                "Enable allow_shell to let Pedacito run the project's linter/"
                "typechecker, or run the check yourself.")
    from . import diagnostics
    rel = ""
    if _as_text(file).strip():
        rel = index.resolve_file(_as_text(file)) or _root_rel(index, file) or ""
        if not rel:
            return f"ERROR: refusing to diagnose outside the project root: {file}"
    ok, report = diagnostics.run_changed(
        index.root, cfg, [rel] if rel else None)
    if ok and not rel:
        return report + "\n[No blocking diagnostics.]"
    if ok:
        return report + f"\n[No blocking diagnostics in {rel}.]"
    return report + "\n[Fix these before claiming the task is complete.]"


def tool_mcp(index, cfg, symbol: str = "", content: str = "") -> str:
    """Invoke one opt-in MCP tool, or list the configured catalog."""
    if not getattr(cfg, "allow_mcp", False):
        return "ERROR: MCP is disabled (config allow_mcp = false)."
    from . import mcp
    target = _as_text(symbol).strip()
    args = {}
    raw = _as_text(content).strip()
    if raw:
        try:
            args = json.loads(raw)
        except ValueError as e:
            return f"ERROR: mcp content must be a JSON object ({e})."
        if not isinstance(args, dict):
            return "ERROR: mcp content must be a JSON object."
    return mcp.call(cfg, target, args)


def _snapshot(index, cfg, rel: str, path: Path) -> str | None:
    """Byte-exact copy of `path` into the run's backup dir before a mutating
    action. Returns an error string if the copy failed; None on success or
    when backups are off. An existing copy is kept: the first one is the
    original that `pedacito restore` brings back."""
    if not getattr(cfg, "backup", False):
        return None
    bdir = getattr(cfg, "_backup_dir", None)
    try:
        if bdir is None:
            from . import workspace
            bdir = workspace.new_backup_dir(index.root)
        fileio.backup_copy(path, Path(bdir) / rel)
        return None
    except OSError as e:
        return f"ERROR: could not back up {rel}: {e}"


def _syntax_note(path: Path, text: str) -> str:
    """Advisory parse check on freshly written content: '' when it parses (or
    the language is unknown), else a WARNING. It cannot refuse like the edit
    path: a chunked file is unparseable until its last `append`."""
    suffix = path.suffix.lower()
    err = None
    if suffix == ".py":
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


def _task_allowlist_error(index, cfg, rel: str, verb: str) -> str:
    """ERROR observation when the opt-in task write allowlist (cfg.auto_
    allowlist) refuses modifying an EXISTING file, else ''. A file is allowed
    when the task title names it or the gather phase read it (cfg._task_reads
    is the live read ledger); creating new files is always allowed."""
    if not getattr(cfg, "auto_allowlist", False):
        return ""
    reads = getattr(cfg, "_task_reads", None)
    if reads is None:
        return ""                  # no gather ever armed the ledger
    allowed = {(p or "").strip().lower().lstrip("./")
               for p in (getattr(cfg, "_task_allowlist", None) or ())}
    allowed |= {(p or "").strip().lower().lstrip("./") for p in reads}
    if (rel or "").strip().lower().lstrip("./") in allowed:
        return ""
    if not (Path(index.root).resolve() / rel).is_file():
        return ""                  # creation: always allowed
    return (f"ERROR: {verb} of {rel} blocked by the task write allowlist "
            "(auto_allowlist): this task never read the file and its title "
            "does not name it. `read` it first if the task really needs to "
            "change it; creating new files is always allowed.")


def tool_write(index, cfg, file: str, content: str,
               append: bool = False) -> str:
    """Replace a whole file with `content` (or append it when `append` is
    true), backing the original up first. Gated by `allow_write` in dispatch.

    Refuses paths outside the project root, non-UTF-8 files and empty content;
    preserves the file's line-ending/BOM style; creates missing files. There
    is no review diff, so the observation points at `restore`. `append` makes
    chunked creation explicit: write the first <=max_write_lines lines, then
    append the next chunk, and so on."""
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
    err = _task_allowlist_error(index, cfg, rel, verb)
    if err:
        return err

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
        err = _snapshot(index, cfg, rel, path)
        if err:
            return err

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fileio.write(path, text, style or fileio.TextStyle())
    except OSError as e:
        return f"ERROR: could not write {rel}: {e}"

    # Refresh the index so line ranges describe the bytes now on disk.
    try:
        from . import index as index_mod
        fresh = index_mod.reindex_file(index, rel, cfg, save=False)
        index.__dict__.update(fresh.__dict__)
    except Exception as e:
        print(f"  [write] index refresh failed for {rel}: {e}", file=sys.stderr)

    n = len(text.splitlines())
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


def tool_move(index, cfg, file: str, to: str) -> str:
    """Rename/move one file, refusing to overwrite an existing destination.
    Gated by `allow_write`."""
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
    err = _task_allowlist_error(index, cfg, src_rel, "move")
    if err:
        return err
    if dst.exists():
        return (f"ERROR: destination already exists: {dst_rel}. Choose another "
                f"path or read it first.")
    try:
        original = fileio.read_text(src)      # only for the line count
    except OSError as e:
        return f"ERROR: could not read {src_rel}: {e}"
    err = _snapshot(index, cfg, src_rel, src)
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
    """Delete one file, after snapshotting it when backups are on. Refuses
    directories and paths outside the root. Gated by `allow_write`."""
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
    err = _task_allowlist_error(index, cfg, rel, "delete")
    if err:
        return err
    try:
        original = fileio.read_text(path)     # only for the line count
    except OSError as e:
        return f"ERROR: could not read {rel}: {e}"
    err = _snapshot(index, cfg, rel, path)
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
    """Split a command into argv without a shell. Windows uses non-POSIX
    splitting (POSIX mode eats path backslashes) and strips the quotes it keeps."""
    if os.name != "nt":
        return shlex.split(cmd)
    return [p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'"
            else p for p in shlex.split(cmd, posix=False)]


def _allowlist_error(cfg, argv: list[str]) -> str:
    """Return an allowlist error for `argv`, or '' when it may run."""
    allow = tuple(getattr(cfg, "shell_allowlist", ()) or ())
    if not allow:
        return ""

    def _exe(name: str) -> str:
        # PureWindowsPath parses backslash paths on every host; pathlib.Path
        # would try to instantiate WindowsPath when os.name is patched, which
        # raises on POSIX (and inside tests that simulate Windows).
        base = (PureWindowsPath(name) if os.name == "nt"
                else Path(name)).name.lower()
        # Compare without `.exe` on Windows.
        if os.name == "nt" and base.endswith(".exe"):
            base = base[:-4]
        return base

    names = {_exe(a) for a in allow}
    exe = _exe(argv[0])
    if exe in names or argv[0].lower() in {a.lower() for a in allow}:
        return ""
    return (f"ERROR: `{argv[0]}` is not in shell_allowlist "
            f"({', '.join(allow)}).")


def tool_run(index, cfg, command: str) -> str:
    """Run one command in the project root and return its output. Gated by
    `allow_shell`.

    The command runs as an argv WITHOUT a shell (pipes and `&&` are inert);
    `shell_allowlist` restricts the executable; output and runtime are capped.
    `proc.run` stops a process that is silent for `shell_idle_seconds` (a
    GUI or server), while test runners get the full `shell_timeout`."""
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
    env = _child_env()
    rc, out, stopped, _elapsed = proc.run(
        argv, cwd=index.root, timeout=timeout, idle_timeout=idle,
        max_output=cap, env=env)
    if stopped == "error":
        return f"ERROR: could not run command: {out}"

    # A command can rewrite source; refresh the index so line ranges stay valid.
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


def _child_env() -> dict:
    """Environment for launched commands: no colour, no pagers."""
    return dict(os.environ, NO_COLOR="1", TERM="dumb", PAGER="cat",
                GIT_PAGER="cat")


def _stop_process(p) -> None:
    proc.terminate(p)
    try:
        p.wait(timeout=5)
    except Exception:
        pass


def _vision_text(client, cfg, image: Path) -> str:
    """Description from the configured vision model, or '' when none is set."""
    if not (getattr(cfg, "vision_model", "") or "").strip():
        return ""
    try:
        return vision.describe(client, cfg, image)
    except Exception as e:
        return f"(vision model failed: {str(e)[:200]})"


def _rel_or_abs(root: Path, p: Path) -> str:
    try:
        return p.relative_to(root).as_posix()
    except ValueError:                            # pragma: no cover
        return str(p)


def _resolve_screenshot(index, name: str) -> Path | None:
    """Resolve an existing screenshot under the project root: a project-relative
    path, or a bare filename from `.pedacito/screenshots/`. Paths outside the
    root are refused."""
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
    """Capture the screen (optionally launching an app first) and report it as
    text: pixel statistics and a blank verdict (Pillow), OCR text (tesseract),
    and an optional `vision_model` description. The PNG is kept under
    `.pedacito/screenshots/`.

    Gated by `allow_screenshots`; a `command` also needs `allow_shell`. The
    launched process is stopped after the capture.
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
    env = _child_env()

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
                # Server-aware launcher: recognises a dev server's printed URL.
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
            _stop_process(launched)
        if not ok:
            return f"ERROR: could not capture the screen: {method}"

    # Compare only against a NEW capture; in file-only mode out is prev.
    report = vision.analyze(out, previous=prev if argv else None)
    desc = _vision_text(client, cfg, out)

    rel_out = _rel_or_abs(root, out)
    bits = [f"SCREENSHOT: {rel_out}", f"CAPTURE: {method}"]
    if argv:
        status = (f" (process had already exited with code {exit_code} before "
                  f"the capture)" if exit_code is not None else
                  " (still running; stopped after capture)")
        bits.append(f"LAUNCHED: {cmd}{status}")
    if server_url:
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
        bits.append("COMPARED WITH: " + _rel_or_abs(root, prev))
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

    `command` is an http(s) URL or a command that starts a local web server
    and prints its URL. `pattern` is an interaction list (`press Enter; click
    #start; wait 2; eval <js>`, see browser.parse_actions); `eval` consumes
    the rest of the pattern and its result is clipped at 2,000 chars. `start`
    is the wait before capture. Reports the page's text, canvas sizes,
    console errors and an optional vision description.

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
    env = _child_env()
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
            _stop_process(server)
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
            _stop_process(server)
    if not ok:
        return f"ERROR: {info}"

    report = vision.analyze(out, previous=prev)
    desc = _vision_text(client, cfg, out)

    rel_out = _rel_or_abs(root, out)
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
        bits.append("COMPARED WITH: " + _rel_or_abs(root, prev))
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


def dispatch(index, cfg, step: dict, client=None) -> str:
    """Route one parsed agent step (matching STEP_SCHEMA) to its tool_*
    function and return the text observation. `client` is only needed for
    semantic search, screenshot and browser vision descriptions."""
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
    if action == "find":
        return tool_find(index, step.get("pattern", ""))
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
    if action == DIAGNOSE_ACTION:
        return tool_diagnose(index, cfg, step.get("file", ""))
    if action == TODO_ACTION:
        return tool_todo(index, cfg, step.get("content", ""))
    if action == MCP_ACTION:
        return tool_mcp(index, cfg, step.get("symbol", ""),
                        step.get("content", ""))
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
        # Refuse a write/append over one step's line budget: it can only have
        # arrived as a truncated reply, and executing it would store half a
        # file. Teach the chunked protocol instead.
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
        return tool_write(index, cfg, step.get("file", ""), content,
                          append=action == APPEND_ACTION)
    return f"ERROR: unknown action '{action}'."
