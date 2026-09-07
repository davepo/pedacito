"""File selection: deciding which files on disk get indexed.

Usage
-----
    from .select import Matcher, select, CODE_EXTS, REFERENCE_EXTS

    matcher = Matcher.from_files(project_root)     # reads .pedacitoignore/.gitignore
    matcher.extend(["--exclude-style/pattern/"])   # optional extra patterns
    files = select(["src", "README.md"], project_root, matcher,
                   exts=CODE_EXTS | REFERENCE_EXTS)

Supports a `.pedacitoignore` file (and a project's existing `.gitignore`) using
a practical subset of gitignore syntax, plus programmatic include/exclude
patterns from the CLI's `--exclude` / `--include` flags:

    build/              directory and everything under it
    *.min.py            glob on the file name
    /scratch            anchored to the project root
    old/**/legacy.py    ** spans directories
    !keep/important.py  negation; last matching pattern wins

Not supported (documented rather than silently wrong): character ranges like
[abc], and gitignore's rule that a negation cannot resurrect a file whose
parent directory was excluded. Here a negation always wins if it matches
last, which is the behaviour most people expect when they write one.

Directory walking follows symlinks (so a vendored or shared directory
reached only through a symlink is actually indexed, not silently skipped --
`Path.rglob` does not descend into symlinked directories on its own), while
guarding against symlink cycles by tracking each directory's resolved
("real") path and never descending into one already visited. A file
reachable by more than one route (two sibling paths pointing at the same
real directory, or a symlink pointing back at an ancestor) is walked and
kept exactly once.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

from . import fileio

# Directories and files excluded from every project by default.
DEFAULT_IGNORE = [
    ".git/", "__pycache__/", ".venv/", "venv/", "env/", "node_modules/",
    ".mypy_cache/", ".pytest_cache/", ".ruff_cache/", ".tox/",
    ".pedacito/", "build/", "dist/", "*.egg-info/",
    # Pedacito's own files. Indexing them wastes map budget and invites the model
    # to "helpfully" edit its own configuration.
    ".pedacito.toml", "pedacito.toml", ".pedacitoignore",
    "*.pyc", "*.pyo", "*.so", "*.bak", ".DS_Store",
]

# Parsed by the AST and chunked semantically (see chunker.py).
PY_EXTS = {".py"}

# Parsed via tree-sitter and chunked semantically when the optional
# tree-sitter dependency is installed (see chunker_ts.py); falls back to
# structural reference-chunking (see references.py) if it isn't, rather
# than being excluded from the index outright.
JS_TS_EXTS = {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}
RUBY_EXTS = {".rb"}

CODE_EXTS = PY_EXTS | JS_TS_EXTS | RUBY_EXTS

# Indexed structurally, not parsed: the model can `read` and `grep` them, and
# the map shows their shape (CSV columns, JSON keys, markdown headings). See
# references.py.
REFERENCE_EXTS = {
    ".md", ".rst", ".txt", ".csv", ".tsv", ".json", ".toml",
    ".yaml", ".yml", ".ini", ".cfg", ".conf", ".env", ".sql", ".sh",
}


# --------------------------------------------------------------------------
# Pattern matcher
# --------------------------------------------------------------------------
class Matcher:
    """An ordered list of gitignore-style patterns. Last matching pattern
    wins, matching gitignore's own precedence rule."""

    def __init__(self, patterns: list[str] | None = None):
        """Build a Matcher from an initial list of pattern lines."""
        # Each rule: (pattern, negate, dir_only, anchored).
        self.rules: list[tuple[str, bool, bool, bool]] = []
        for p in patterns or []:
            self.add(p)

    def add(self, raw: str) -> None:
        """Parse and append one pattern line (blank lines and #comments skipped)."""
        p = raw.strip()
        if not p or p.startswith("#"):
            return
        negate = p.startswith("!")
        if negate:
            p = p[1:]
        dir_only = p.endswith("/")
        p = p.rstrip("/")
        anchored = p.startswith("/") or ("/" in p.rstrip("/") and not p.startswith("**/"))
        p = p.lstrip("/")
        if p:
            self.rules.append((p, negate, dir_only, anchored))

    def extend(self, patterns) -> "Matcher":
        """Append multiple pattern lines at once; returns self for chaining."""
        for p in patterns or []:
            self.add(p)
        return self

    @classmethod
    def from_files(cls, root: Path, names=(".pedacitoignore", ".gitignore"),
                   use_defaults: bool = True) -> "Matcher":
        """Build a Matcher from the built-in defaults plus any of the named
        ignore files that exist under `root`, read in order."""
        m = cls(DEFAULT_IGNORE if use_defaults else [])
        for name in names:
            f = root / name
            if f.exists():
                for line in fileio.read_lines(f):
                    m.add(line)
        return m

    def _hit(self, rel: str, pat: str, dir_only: bool, anchored: bool) -> bool:
        """Test whether one pattern matches a given relative path."""
        parts = rel.split("/")
        if dir_only:
            # Match any ancestor directory of this path.
            for i in range(len(parts) - 1):
                seg = "/".join(parts[: i + 1]) if anchored else parts[i]
                if fnmatch.fnmatch(seg, pat):
                    return True
            return False
        if anchored:
            return fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(rel, pat + "/*")
        # Unanchored: match the basename, or any path suffix.
        if fnmatch.fnmatch(parts[-1], pat):
            return True
        return any(fnmatch.fnmatch("/".join(parts[i:]), pat) for i in range(len(parts)))

    def excluded(self, rel: str) -> bool:
        """Return True if `rel` (a project-relative path) should be excluded,
        applying every rule in order so the last match wins."""
        rel = rel.replace("\\", "/")   # index paths are POSIX; inputs may not be
        verdict = False
        for pat, negate, dir_only, anchored in self.rules:
            if self._hit(rel, pat, dir_only, anchored):
                verdict = not negate
        return verdict


# --------------------------------------------------------------------------
# Cycle-safe directory walk
# --------------------------------------------------------------------------
def _walk(path: Path) -> list[Path]:
    """Return every file under `path`, descending into symlinked directories
    (unlike `Path.rglob`, which treats a symlinked directory as a leaf and
    never looks inside it) while never re-entering a directory already
    visited in this walk.

    A directory is identified by its resolved ("real") path, so a cycle --
    a symlink pointing back at an ancestor, or two different routes to the
    same real directory -- is visited once and then skipped, rather than
    recursing forever or listing its contents twice.
    """
    visited: set[Path] = set()
    stack: list[Path] = [path]
    files: list[Path] = []
    while stack:
        d = stack.pop()
        try:
            real = d.resolve()
        except OSError:
            continue
        if real in visited:
            continue
        visited.add(real)
        try:
            entries = sorted(d.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                stack.append(entry)
            elif entry.is_file():
                files.append(entry)
    return sorted(files)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def select(paths: list[str], root: Path, matcher: Matcher,
           exts: set[str], max_bytes: int = 0) -> list[Path]:
    """Expand a list of files/directories into a filtered, deduplicated file
    list.

    Directories are walked recursively; each candidate file must have an
    extension in `exts` and must not be excluded by `matcher`, unless it was
    named explicitly (as opposed to discovered by walking a directory) --
    a path typed directly on the command line is always included, since
    typing it is an explicit request regardless of ignore rules.
    """
    out: list[Path] = []
    seen: set[Path] = set()

    def keep(f: Path, explicit: bool) -> None:
        """Add `f` to the result if it passes the extension, ignore-rule,
        and size checks (ignore rules are skipped for explicit paths)."""
        r = f.resolve()
        if r in seen:
            return
        if f.suffix.lower() not in exts:
            return
        if not explicit:
            try:
                rel = r.relative_to(root).as_posix()
            except ValueError:
                rel = f.name
            if matcher.excluded(rel):
                return
        if max_bytes and f.stat().st_size > max_bytes:
            return
        seen.add(r)
        out.append(f)

    for p in paths:
        path = Path(p)
        if path.is_file():
            keep(path, explicit=True)
        elif path.is_dir():
            for f in _walk(path):
                keep(f, explicit=False)
    return out
