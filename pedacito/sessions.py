"""Read and search prior task session logs.

Usage
-----
    from . import sessions

    sessions.recent(root)                 # what has happened lately
    sessions.search(root, "retry")        # find a topic across runs
    sessions.read(root, "20260101-120000.md")

Logs live in `<root>/.pedacito/sessions/*.md` (written by
`workspace.write_session`). They are not in the code index -- `.pedacito/` is
ignored -- so this module is the agent's read-only view of what previous runs
asked, looked at, and changed. It matters most for small models starting a
follow-up task with no memory of the prior conversation.
"""

from __future__ import annotations

import re
from pathlib import Path

from .index import SESSION_DIR, state_dir

MAX_LIST = 10
MAX_HITS = 20
MAX_READ_LINES = 200


def session_dir(root) -> Path:
    """Path of the project's session directory."""
    return state_dir(Path(root), SESSION_DIR)


def paths(root) -> list[Path]:
    """Session log files, newest first."""
    d = session_dir(root)
    if not d.exists():
        return []
    return sorted((p for p in d.glob("*.md") if p.is_file()), reverse=True)


def _resolve(root, name: str) -> Path | None:
    """Resolve a session filename safely to a file under the sessions dir.

    Only a basename is accepted: session logs are flat, and this makes a
    model-supplied `../index.json` or absolute path harmless."""
    name = (name or "").strip().strip("`\"'").replace("\\", "/")
    if not name:
        return None
    base = Path(name).name
    if base in ("", ".", ".."):
        return None
    p = session_dir(root) / base
    if p.is_file():
        return p
    # A model may give the timestamp without the .md suffix.
    if not base.endswith(".md"):
        p = session_dir(root) / (base + ".md")
        if p.is_file():
            return p
    return None


def _task(text: str) -> str:
    """The `**Task:**` line from a session log, or a placeholder."""
    for line in text.splitlines()[:40]:
        if line.startswith("**Task:**"):
            return line[len("**Task:**"):].strip()
    return "(no task recorded)"


def _edit_count(text: str) -> int:
    return sum(1 for line in text.splitlines()
               if line.startswith("- applied ") or line.startswith("- FAILED "))


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def recent(root, limit: int = MAX_LIST) -> str:
    """List recent session logs with their task and edit count."""
    ps = paths(root)
    if not ps:
        return ("No prior sessions are recorded for this project. Session "
                "logging may be off (`log_sessions = false`).")
    shown = ps[:max(1, int(limit or MAX_LIST))]
    out = [f"Prior sessions (most recent first; {len(ps)} total):"]
    for p in shown:
        try:
            text = _read_text(p)
        except OSError:
            continue
        n = _edit_count(text)
        tail = f"  [{n} edit(s)]" if n else ""
        out.append(f"  {p.name}  {_task(text)}{tail}")
    if len(ps) > len(shown):
        out.append(f"  ... {len(ps) - len(shown)} older session(s) not shown")
    out.append("[Read one with `sessions` file=<name>, or search all with "
               "pattern=<text>. These are historical records, not current "
               "source -- verify with `read` before editing.]")
    return "\n".join(out)


def read(root, name: str, start: int = 1, end: int = 0,
         max_lines: int = MAX_READ_LINES) -> str:
    """Return numbered lines from one session log."""
    p = _resolve(root, name)
    if p is None:
        return (f"ERROR: no session named '{name}'. Use `sessions` with no "
                "`file` to list the available ones.")
    try:
        lines = _read_text(p).splitlines()
    except OSError as e:
        return f"ERROR: could not read {p.name}: {e}"
    if lines and lines[-1] == "" and len(lines) > 1:
        lines = lines[:-1]
    start = max(1, int(start or 1))
    end = int(end or 0) or len(lines)
    end = min(end, len(lines))
    if start > len(lines):
        return f"ERROR: {p.name} has only {len(lines)} lines."
    if end < start:
        end = min(start + max_lines - 1, len(lines))
    note = ""
    if end - start + 1 > max_lines:
        end = start + max_lines - 1
        note = f"\n[truncated to {max_lines} lines; read again from {end + 1}]"
    width = len(str(end))
    body = "\n".join(f"{i:>{width}} | {ln}"
                     for i, ln in enumerate(lines[start - 1:end], start))
    return (f"{p.name} (session log, not source) lines {start}-{end}:\n"
            f"{body}{note}")


def _compile(query: str) -> tuple[re.Pattern, list[str]]:
    """Compile a query, falling back to literal text like the grep tool."""
    notes: list[str] = []
    needle = query
    try:
        re.compile(query)
    except re.error:
        needle = re.escape(query)
        notes.append("[note: query was not valid regex; searched literally]")
    return re.compile(needle), notes


def search(root, query: str, scope: str = "", max_hits: int = MAX_HITS) -> str:
    """Search session logs and return grouped matching lines."""
    q = (query or "").strip()
    if not q:
        return recent(root)
    ps = paths(root)
    if not ps:
        return "No prior sessions are recorded for this project."
    notes: list[str] = []
    if scope:
        one = _resolve(root, scope)
        if one is not None:
            ps = [one]
        else:
            notes.append(f"[note: '{scope}' is not a session; searched all "
                         f"{len(ps)} session(s)]")

    rx, compile_notes = _compile(q)
    notes += compile_notes

    def scan(rx: re.Pattern) -> list[str]:
        out: list[str] = []
        per_file = max(3, max_hits // 4)
        for p in ps:
            try:
                lines = _read_text(p).splitlines()
            except OSError:
                continue
            file_hits = 0
            for i, line in enumerate(lines, 1):
                if not rx.search(line):
                    continue
                if file_hits >= per_file:
                    out.append(f"  {p.name}: [more matches; search this session]")
                    break
                text = line[:200].rstrip()
                cut = " ...[truncated]" if len(line) > 200 else ""
                out.append(f"  {p.name}:{i}: {text}{cut}")
                file_hits += 1
                if len(out) >= max_hits:
                    out.append(f"  [stopped at {max_hits} hits; narrow the query]")
                    return out
        return out

    hits = scan(rx)
    if not hits:
        rx_ci = re.compile(rx.pattern, re.IGNORECASE)
        hits = scan(rx_ci)
        if hits:
            notes.append("[note: no case-sensitive matches; retried "
                         "case-insensitively]")
    note_text = ("\n".join(notes) + "\n") if notes else ""
    if not hits:
        return note_text + f"No session matches for /{q}/."
    return (note_text
            + "[Session logs are historical. Verify current code with `read` "
              "or `grep` before editing.]\n"
            + "\n".join(hits))
