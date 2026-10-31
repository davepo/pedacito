"""File selection: deciding which files on disk get indexed.

Usage
-----
    from .select import Matcher, select, CODE_EXTS, REFERENCE_EXTS

    matcher = Matcher.from_files(project_root)     # reads .pedacitoignore/.gitignore
    matcher.extend(["--exclude-style/pattern/"])   # optional extra patterns
    files = select(["src", "README.md"], project_root, matcher,
                   exts=CODE_EXTS | REFERENCE_EXTS)

Supports a `.pedacitoignore` file (and a project's `.gitignore`) using a
practical subset of gitignore syntax, plus programmatic include/exclude
patterns from the CLI's `--exclude` / `--include` flags:

    build/              directory and everything under it
    *.min.py            glob on the file name
    /scratch            anchored to the project root
    old/**/legacy.py    ** spans directories
    !keep/important.py  negation; last matching pattern wins

Not supported: character ranges like [abc], and gitignore's rule that a
negation cannot resurrect a file under an excluded directory (here a later
negation always wins).

Directory walking follows symlinks (`Path.rglob` does not descend into
symlinked directories) and guards against cycles by tracking resolved paths;
a file reachable by several routes is kept once.
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
    # Pedacito's own files.
    ".pedacito.toml", "pedacito.toml", ".pedacitoignore",
    "*.pyc", "*.pyo", "*.so", "*.bak", ".DS_Store",
]

# Chunked by the Python AST (chunker.py).
PY_EXTS = {".py"}

# Chunked via tree-sitter (chunker_ts.py); falls back to structural
# reference-chunking (references.py) if tree-sitter is unavailable.
JS_TS_EXTS = {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}
RUBY_EXTS = {".rb"}

CODE_EXTS = PY_EXTS | JS_TS_EXTS | RUBY_EXTS

# Indexed structurally (references.py): readable and greppable, with the map
# showing their shape (CSV columns, JSON keys, markdown headings).
REFERENCE_EXTS = {
    ".md", ".rst", ".txt", ".csv", ".tsv", ".json", ".toml",
    ".yaml", ".yml", ".ini", ".cfg", ".conf", ".env", ".sql", ".sh",
    ".html", ".htm", ".xhtml",
}


class Matcher:
    """An ordered list of gitignore-style patterns; the last match wins."""

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

    def extend(self, patterns) -> Matcher:
        """Append multiple pattern lines at once; returns self for chaining."""
        for p in patterns or []:
            self.add(p)
        return self

    @classmethod
    def from_files(cls, root: Path, names=(".pedacitoignore", ".gitignore")
                   ) -> Matcher:
        """Build a Matcher from the built-in defaults plus any of the named
        ignore files that exist under `root`, read in order."""
        m = cls(DEFAULT_IGNORE)
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
        """True if the project-relative path `rel` is excluded."""
        rel = rel.replace("\\", "/")   # index paths are POSIX; inputs may not be
        verdict = False
        for pat, negate, dir_only, anchored in self.rules:
            if self._hit(rel, pat, dir_only, anchored):
                verdict = not negate
        return verdict


def is_venv_dir(path: Path) -> bool:
    """True when `path` is the root of a Python virtual environment (it holds
    a ``pyvenv.cfg``, whatever the directory is named), so its site-packages
    is never indexed as project code."""
    try:
        return (path / "pyvenv.cfg").is_file()
    except OSError:
        return False


def _walk(path: Path) -> list[Path]:
    """Return every file under `path`, following symlinked directories but
    visiting each real directory once (cycle-safe) and skipping virtualenvs."""
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
                if is_venv_dir(entry):
                    continue
                stack.append(entry)
            elif entry.is_file():
                files.append(entry)
    return sorted(files)


def select(paths: list[str], root: Path, matcher: Matcher,
           exts: set[str]) -> list[Path]:
    """Expand files/directories into a filtered, deduplicated file list.

    Directory walks keep files whose extension is in `exts` and that `matcher`
    does not exclude; a file named explicitly is exempt from ignore rules.
    """
    out: list[Path] = []
    seen: set[Path] = set()

    def keep(f: Path, explicit: bool) -> None:
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
