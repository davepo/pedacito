"""Project workspace setup: the .pedacito/ directory, ignore files, backups,
session logs, and the run log.

Everything Pedacito writes for a project lives in one place:

    <project>/.pedacito/
        index.json          the map
        summaries.json      summary cache, keyed by content hash
        backups/<stamp>/    pre-edit snapshots, mirroring your tree
        sessions/<stamp>.md what was asked, looked up, and changed
        log.txt             operational run log

On first run this also creates a `.pedacitoignore` file and adds `.pedacito/`
to `.gitignore`.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from .index import BACKUP_DIR, SESSION_DIR, state_dir

IGNORE_FILE = ".pedacitoignore"

# Paths Pedacito writes into a project that should never be committed: the
# workspace tree, the ignore file, and the two project-config filenames
# find_project_config() looks for.
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


def stamp() -> str:
    """Current time as a sortable filesystem-safe string (backup and session
    names)."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


RUN_LOG_NAME = "log.txt"


def run_log_path(root: Path) -> Path:
    """Path of the operational run log for a project: `.pedacito/log.txt`."""
    return state_dir(root, RUN_LOG_NAME)


def log_event(root: Path, message: str, kind: str = "INFO") -> None:
    """Append one timestamped line to the run log: commands, phase
    transitions, statuses, and errors, but no model output (that lives in
    `.pedacito/sessions/*.md`). Appended across runs."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    p = run_log_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(f"{ts}  {kind:<6} {message}\n")


def ensure(root: Path, manage_gitignore: bool = True,
           create_ignore: bool = True) -> Path:
    """Create .pedacito/ under `root` plus the ignore scaffolding
    (.pedacitoignore, a .gitignore entry). Idempotent."""
    root = Path(root).resolve()
    d = state_dir(root)
    first_run = not d.exists()
    d.mkdir(parents=True, exist_ok=True)

    if create_ignore:
        ig = root / IGNORE_FILE
        if not ig.exists():
            ig.write_text(IGNORE_TEMPLATE, encoding="utf-8")
            print(f"  created {ig}", file=sys.stderr)
        elif ".pedacito" not in ig.read_text(encoding="utf-8", errors="replace"):
            with ig.open("a", encoding="utf-8") as fh:
                fh.write("\n.pedacito/\n")

    if manage_gitignore:
        _add_to_gitignore(root)

    if first_run:
        print(f"  state for this project lives in {d}", file=sys.stderr)
    return d


def _add_to_gitignore(root: Path) -> None:
    """Add Pedacito's entries to the project's .gitignore: create it in a git
    checkout that has none, else append the missing entries. Never creates one
    outside a repo."""
    gi = root / ".gitignore"
    if not gi.exists():
        if not (root / ".git").exists():
            return          # not a repo; leave the directory alone
        gi.write_text("# Pedacito\n" + "\n".join(GITIGNORE_ENTRIES) + "\n",
                      encoding="utf-8")
        print(f"  created {gi} with Pedacito entries", file=sys.stderr)
        return

    text = gi.read_text(encoding="utf-8", errors="replace")
    # ".pedacito" and ".pedacito/" count as the same entry
    existing = {
        ln.strip().rstrip("/")
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    }
    to_add = [e for e in GITIGNORE_ENTRIES if e.rstrip("/") not in existing]
    if not to_add:
        return

    with gi.open("a", encoding="utf-8") as fh:
        if text and not text.endswith("\n"):
            fh.write("\n")
        if text:
            fh.write("\n")           # blank separator
        fh.write("# Pedacito\n")
        for e in to_add:
            fh.write(f"{e}\n")
    n = len(to_add)
    print(f"  updated {gi} ({n} Pedacito entr"
          f"{'y' if n == 1 else 'ies'} added)", file=sys.stderr)


def new_backup_dir(root: Path, when: str | None = None) -> Path:
    """Create and return a timestamped directory under .pedacito/backups/."""
    d = state_dir(root, BACKUP_DIR, when or stamp())
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_backups(root: Path) -> list[Path]:
    """Backup snapshot directories under .pedacito/backups/, newest first."""
    d = state_dir(root, BACKUP_DIR)
    return sorted((p for p in d.iterdir() if p.is_dir()), reverse=True) if d.exists() else []


def restore(root: Path, snapshot: Path, dry_run: bool = False) -> list[str]:
    """Copy every file in a backup snapshot back over the working tree
    byte-for-byte. Returns the relative paths restored (or that would be, if
    `dry_run`). Files created since the snapshot have no copy in it and are
    left in place, never deleted."""
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
            dest.write_bytes(src.read_bytes())
    return restored


def write_session(root: Path, task: str, steps: list[str], reply: str,
                  results, backup: Path | None, when: str | None = None,
                  raw_reply: str | None = None) -> Path:
    """Write a Markdown record of one task (request, lookups, reply, edits)
    to .pedacito/sessions/<timestamp>.md. `raw_reply`, when given, is the
    reply as the model emitted it, before trimming or continuation rounds."""
    d = state_dir(root, SESSION_DIR)
    d.mkdir(parents=True, exist_ok=True)
    when = when or stamp()
    lines = [
        f"# {when}", "", f"**Task:** {task}", "", "## Lookups", "",
        *(f"- {s}" for s in (steps or ["(none)"])),
        "", "## Response", "", reply.strip() or "(empty)", "", "## Edits", "",
    ]
    if raw_reply is not None and raw_reply != reply:
        lines += ["", "## Raw reply (as emitted, before parsing/trimming)", "",
                  raw_reply.strip() or "(empty)"]
    if results:
        for r in results:
            lines.append(f"- {'applied' if r.ok else 'FAILED'} `{r.file}`: {r.message}")
    else:
        lines.append("- none")
    if backup is not None and backup.exists() and any(backup.rglob("*")):
        lines += ["", f"Originals saved to `{backup.relative_to(root)}`.",
                  f"Restore with `pedacito restore {root} --snapshot {backup.name}`."]
    p = d / f"{when}.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p
