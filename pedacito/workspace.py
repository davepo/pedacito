"""Project workspace setup: the .pedacito/ directory, ignore files, backups,
and session logs.

Usage
-----
    from . import workspace

    workspace.ensure(project_root, manage_gitignore=True, create_ignore=True)
    backup_dir = workspace.new_backup_dir(project_root)
    ...
    workspace.write_session(project_root, task, steps, reply, results, backup_dir)

Everything Pedacito writes for a project lives in one place:

    <project>/.pedacito/
        index.json          the map
        summaries.json      summary cache, keyed by content hash
        backups/<stamp>/    pre-edit snapshots, mirroring your tree
        sessions/<stamp>.md what was asked, looked up, and changed

Nothing is written beside your source files. On first run this also creates
a `.pedacitoignore` file to edit, and adds `.pedacito/` to `.gitignore` so the
index and backups never end up in a commit.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from . import fileio
from .index import BACKUP_DIR, INDEX_DIR, SESSION_DIR, state_dir

IGNORE_FILE = ".pedacitoignore"

IGNORE_TEMPLATE = """\
# Files and folders Pedacito should not index. Gitignore syntax.
# Your .gitignore is read too, along with built-in defaults for .git,
# __pycache__, .venv, node_modules, build, dist and similar.
#
# Examples:
#   deprecated/              a folder of old script versions
#   *_old.py
#   /scratch                 anchored to the project root only
#   !deprecated/still_used.py    negation; last match wins
#
# Naming a path directly on the command line always overrides these.

.pedacito/
"""


# --------------------------------------------------------------------------
# Timestamps
# --------------------------------------------------------------------------
def stamp() -> str:
    """Return the current time as a sortable filesystem-safe string, used to
    name backup snapshots and session log files."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


# --------------------------------------------------------------------------
# First-run scaffolding
# --------------------------------------------------------------------------
def ensure(root: Path, manage_gitignore: bool = True,
           create_ignore: bool = True, verbose: bool = True) -> Path:
    """Create .pedacito/ under `root` and, on first run, the ignore
    scaffolding (.pedacitoignore, and a .gitignore entry). Idempotent -- safe
    to call on every command."""
    root = Path(root).resolve()
    d = state_dir(root)
    first_run = not d.exists()
    d.mkdir(parents=True, exist_ok=True)

    if create_ignore:
        ig = root / IGNORE_FILE
        if not ig.exists():
            ig.write_text(IGNORE_TEMPLATE)
            if verbose:
                print(f"  created {ig}", file=sys.stderr)
        elif ".pedacito" not in ig.read_text():
            with ig.open("a") as fh:
                fh.write("\n.pedacito/\n")

    if manage_gitignore:
        _add_to_gitignore(root, verbose)

    if first_run and verbose:
        print(f"  state for this project lives in {d}", file=sys.stderr)
    return d


def _add_to_gitignore(root: Path, verbose: bool) -> None:
    """Add a `.pedacito/` entry to the project's .gitignore, keeping the index
    and backups out of commits. Only touches a repo that already has a
    .gitignore, or one that is clearly a git checkout (has a .git/ dir)."""
    gi = root / ".gitignore"
    if not gi.exists():
        if not (root / ".git").exists():
            return          # not a repo; leave the directory alone
        gi.write_text(f"{INDEX_DIR}/\n")
        if verbose:
            print(f"  created {gi} with {INDEX_DIR}/", file=sys.stderr)
        return
    text = gi.read_text()
    if any(line.strip().rstrip("/") == INDEX_DIR for line in text.splitlines()):
        return
    with gi.open("a") as fh:
        fh.write(("" if text.endswith("\n") else "\n") + f"{INDEX_DIR}/\n")
    if verbose:
        print(f"  added {INDEX_DIR}/ to {gi}", file=sys.stderr)


# --------------------------------------------------------------------------
# Backups and restore
# --------------------------------------------------------------------------
def new_backup_dir(root: Path, when: str | None = None) -> Path:
    """Create and return a new timestamped backup directory under
    .pedacito/backups/, ready to receive pre-edit file copies."""
    d = state_dir(root, BACKUP_DIR, when or stamp())
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_backups(root: Path) -> list[Path]:
    """List all backup snapshot directories under .pedacito/backups/, most
    recent first."""
    d = state_dir(root, BACKUP_DIR)
    return sorted((p for p in d.iterdir() if p.is_dir()), reverse=True) if d.exists() else []


def restore(root: Path, snapshot: Path, dry_run: bool = False) -> list[str]:
    """Copy every file in a backup snapshot back over the working tree
    byte-for-byte, restoring it to its pre-edit state. Returns the list of
    relative paths restored (or that would be restored, if `dry_run`)."""
    root = Path(root).resolve()
    restored: list[str] = []
    for src in sorted(snapshot.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(snapshot).as_posix()
        dest = root / rel
        restored.append(rel)
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read_bytes())   # byte-exact restore
    return restored


# --------------------------------------------------------------------------
# Session logging
# --------------------------------------------------------------------------
def write_session(root: Path, task: str, steps: list[str], reply: str,
                  results, backup: Path | None, when: str | None = None) -> Path:
    """Write a readable Markdown record of one task (what was asked, every
    lookup made, the model's reply, and which edits landed) to
    .pedacito/sessions/<timestamp>.md. Useful when an edit turns out wrong
    later and you need to see what the model was actually looking at."""
    d = state_dir(root, SESSION_DIR)
    d.mkdir(parents=True, exist_ok=True)
    when = when or stamp()
    lines = [
        f"# {when}", "", f"**Task:** {task}", "", "## Lookups", "",
        *(f"- {s}" for s in (steps or ["(none)"])),
        "", "## Response", "", reply.strip() or "(empty)", "", "## Edits", "",
    ]
    if results:
        for r in results:
            lines.append(f"- {'applied' if r.ok else 'FAILED'} `{r.file}`: {r.message}")
    else:
        lines.append("- none")
    if backup is not None and backup.exists() and any(backup.rglob("*")):
        lines += ["", f"Originals saved to `{backup.relative_to(root)}`.",
                  f"Restore with `pedacito restore {root} --snapshot {backup.name}`."]
    p = d / f"{when}.md"
    p.write_text("\n".join(lines) + "\n")
    return p
