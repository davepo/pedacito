"""Project instruction files: the conventions a repository wants every task to
follow (build commands, style rules, forbidden areas, verification steps),
following the `AGENTS.md` / `CLAUDE.md` convention.

Only the first file found is used, in the order of `CANDIDATE_FILES`. The
content is capped by `max_instruction_chars` so a long document cannot crowd
out the project map on a small context window.
"""

from __future__ import annotations

from pathlib import Path

from . import fileio

CANDIDATE_FILES = (
    "PEDACITO.md",
    "PEDACEITO.md",
    "AGENTS.md",
    "CLAUDE.md",
    ".pedacito/instructions.md",
    ".pedacito/PEDACITO.md",
    ".pedacito/PEDACEITO.md",
)

# (path, mtime_ns, size) -> text. A changed file gets a new key, so edits are
# picked up; the cache only avoids re-reading an unchanged file every turn.
_CACHE: dict[tuple[str, int, int], str] = {}


def find(root) -> tuple[str, str]:
    """Return (relative_path, text) for the first instruction file, or ("", "")."""
    base = Path(root)
    for rel in CANDIDATE_FILES:
        p = base / rel
        try:
            st = p.stat()
        except OSError:
            continue
        if not p.is_file():
            continue
        key = (str(p), st.st_mtime_ns, st.st_size)
        text = _CACHE.get(key)
        if text is None:
            try:
                text = fileio.read_text(p, errors="replace")
            except OSError:
                continue
            _CACHE[key] = text
        return rel, text
    return "", ""


def render(root, cfg=None) -> str:
    """Render the project's instructions for a prompt, or '' when disabled."""
    if cfg is not None and not getattr(cfg, "use_project_instructions", True):
        return ""
    rel, text = find(root)
    text = (text or "").strip()
    if not rel or not text:
        return ""
    limit = int(getattr(cfg, "max_instruction_chars", 2000) or 0)
    if limit > 0 and len(text) > limit:
        text = (text[:limit].rstrip()
                + f"\n[... truncated; read {rel} for the rest]")
    return f"PROJECT INSTRUCTIONS (from {rel}):\n{text}"
