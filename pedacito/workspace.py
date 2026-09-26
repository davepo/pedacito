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
from .index import BACKUP_DIR, SESSION_DIR, state_dir

IGNORE_FILE = ".pedacitoignore"

# Paths Pedacito writes into a project that should never be committed.
# `.pedacito/` covers the whole workspace tree (index, backups, sessions,
# reviews, cached summaries); the other three are the ignore file and the
# two project-config filenames find_project_config() looks for.
GITIGNORE_ENTRIES = (
    ".pedacito/",
    ".pedacitoignore",
    ".pedacito.toml",
    "pedacito.toml",
)

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
    """Add Pedacito's workspace entries to the project's .gitignore, keeping
    the index, backups, sessions, and per-project config out of commits.

    Creates the file in a git checkout that doesn't already have one;
    otherwise appends only the entries that aren't already present. Never
    invents a .gitignore in a directory that isn't a repo -- that would be
    presumptuous, and the README's contract says so.
    """
    gi = root / ".gitignore"
    if not gi.exists():
        if not (root / ".git").exists():
            return          # not a repo; leave the directory alone
        gi.write_text("# Pedacito\n" + "\n".join(GITIGNORE_ENTRIES) + "\n")
        if verbose:
            print(f"  created {gi} with Pedacito entries", file=sys.stderr)
        return

    text = gi.read_text()
    # Strip trailing slashes on both sides so ".pedacito" and ".pedacito/"
    # count as the same entry (gitignore treats them identically for a
    # directory), and ignore comment lines when checking for presence.
    existing = {
        ln.strip().rstrip("/")
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    }
    to_add = [e for e in GITIGNORE_ENTRIES if e.rstrip("/") not in existing]
    if not to_add:
        return

    with gi.open("a") as fh:
        if text and not text.endswith("\n"):
            fh.write("\n")
        if text:
            fh.write("\n")           # blank separator
        fh.write("# Pedacito\n")
        for e in to_add:
            fh.write(f"{e}\n")
    if verbose:
        n = len(to_add)
        print(f"  updated {gi} ({n} Pedacito entr"
              f"{'y' if n == 1 else 'ies'} added)", file=sys.stderr)


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
