"""A small, persistent task scratchpad for the gather/answer loop.

Usage
-----
    from . import todo

    todo.write(root, "- [x] inspect parser\\n- [ ] add the retry")
    print(todo.render(root, cfg))

The checklist lives in `.pedacito/todo.md`, so it is visible to the user and
survives the gather -> answer transition (the answer phase has no tools). It is
cleared at the start of each new task: `auto` already has the long-lived
PLAN.md, and a stale checklist from the previous task is worse than none.
"""

from __future__ import annotations

from pathlib import Path

TODO_FILE = "todo.md"
MAX_CHARS = 4000


def path(root) -> Path:
    return Path(root) / ".pedacito" / TODO_FILE


def clear(root) -> None:
    try:
        path(root).unlink()
    except OSError:
        pass


def write(root, content: str) -> str:
    text = (content or "").strip()
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS].rstrip() + "\n[... checklist truncated]"
    p = path(root)
    if not text:
        clear(root)
        return ""
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n", encoding="utf-8")
    except OSError:
        return ""
    return text


def read(root) -> str:
    try:
        return path(root).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def render(root, cfg=None, limit: int = 1500) -> str:
    """Render the current checklist for a prompt, capped for small windows."""
    text = read(root)
    if not text:
        return ""
    if limit > 0 and len(text) > limit:
        text = text[:limit].rstrip() + "\n[... checklist truncated]"
    return "CURRENT TASK CHECKLIST:\n" + text
