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

Nested `.gitignore`/`.pedacitoignore` files are honoured inside their own
directory (a deeper file overrides a shallower one). Excluded directories are
pruned before they are walked (`node_modules` is never enumerated) unless a
negation rule could re-include something beneath them.

Not supported: character ranges like [abc], and gitignore's rule that a
negation cannot resurrect a file under an excluded directory (here a later
negation always wins).

Directory walking follows symlinks (`Path.rglob` does not descend into
symlinked directories) and guards against cycles by tracking resolved paths;
a file reachable by several routes is kept once.
"""

from __future__ import annotations

import difflib
import fnmatch
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from . import fileio, security

# Directories and files excluded from every project by default.
DEFAULT_IGNORE = [
    ".git/", "__pycache__/", ".venv/", "venv/", "env/", "node_modules/",
    ".mypy_cache/", ".pytest_cache/", ".ruff_cache/", ".tox/",
    ".pedacito/", "build/", "dist/", "*.egg-info/",
    # Pedacito's own files.
    ".pedacito.toml", "pedacito.toml", ".pedacitoignore",
    "*.pyc", "*.pyo", "*.so", "*.bak", ".DS_Store",
    # Secret-bearing files are never indexed (so never sent to a model).
    # Re-include one deliberately with `!.env` in .pedacitoignore, or by
    # naming it on the command line.
    ".env", ".env.*", "*.env", "!.env.example", "!.env.sample",
    "!.env.template",
    "*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "id_rsa*", "id_dsa*",
    "id_ecdsa*", "id_ed25519*", "!*.pub", ".npmrc", ".pypirc", ".netrc",
    ".git-credentials", "credentials.json", "secrets.json", "secrets.yaml",
    "secrets.yml", "service-account*.json",
]

# Chunked by the Python AST (chunker.py).
PY_EXTS = {".py"}

# Chunked via tree-sitter (chunker_ts.py); falls back to structural
# reference-chunking (references.py) if tree-sitter is unavailable.
JS_TS_EXTS = {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}
RUBY_EXTS = {".rb"}
# Go, Rust, Java, C, C++, C#, PHP, Kotlin, Swift, Scala and Lua (chunker_ts.GENERIC_SPECS). Must mirror
# chunker_ts.LANG_BY_EXT: an extension missing here is never selected by a
# directory walk, so the language is silently invisible to the index.
COMPILED_EXTS = {".go", ".rs", ".java", ".c", ".h", ".cc", ".cpp", ".cxx",
                 ".hpp", ".hh", ".cs", ".php", ".kt", ".kts", ".swift",
                 ".scala", ".lua"}

CODE_EXTS = PY_EXTS | JS_TS_EXTS | RUBY_EXTS | COMPILED_EXTS

# Indexed structurally (references.py): readable and greppable, with the map
# showing their shape (CSV columns, JSON keys, markdown headings).
REFERENCE_EXTS = {
    ".md", ".rst", ".txt", ".csv", ".tsv", ".json", ".toml",
    ".yaml", ".yml", ".ini", ".cfg", ".conf", ".env", ".sql", ".sh",
    ".html", ".htm", ".xhtml", ".css",
}


class Matcher:
    """An ordered list of gitignore-style patterns; the last match wins."""

    def __init__(self, patterns: list[str] | None = None):
        """Build a Matcher from an initial list of pattern lines."""
        # Each rule: (pattern, negate, dir_only, anchored).
        self.rules: list[tuple[str, bool, bool, bool]] = []
        # (directory rel path, Matcher) for nested ignore files, shallow first.
        self.scoped: list[tuple[str, "Matcher"]] = []
        # Ignore-file names read again in every directory the walker enters.
        self.nested_names: tuple[str, ...] = ()
        # Leading rules that are built-in defaults: their negations
        # (`!.env.example`) never keep an excluded directory walkable.
        self._default_count = 0
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
        m._default_count = len(m.rules)
        m.nested_names = tuple(names)
        for name in names:
            f = root / name
            if f.exists():
                for line in fileio.read_lines(f):
                    m.add(line)
        return m

    def add_scoped(self, base: str, lines) -> None:
        """Register the lines of a nested ignore file living in directory
        `base` (project-relative, POSIX). Its patterns only see paths below
        `base`, and its verdict beats the shallower files'."""
        sub = Matcher()
        for line in lines:
            sub.add(line)
        if not sub.rules:
            return
        base = base.strip("/")
        # Re-reading the same file (a second walk) replaces, never stacks.
        self.scoped = [(b, m) for b, m in self.scoped if b != base]
        self.scoped.append((base, sub))
        self.scoped.sort(key=lambda t: t[0].count("/"))

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

    def _verdict(self, rel: str) -> bool | None:
        """Last matching rule's verdict, or None when no rule matched."""
        verdict = None
        for pat, negate, dir_only, anchored in self.rules:
            if self._hit(rel, pat, dir_only, anchored):
                verdict = not negate
        return verdict

    def excluded(self, rel: str) -> bool:
        """True if the project-relative path `rel` is excluded."""
        rel = rel.replace("\\", "/")   # index paths are POSIX; inputs may not be
        verdict = bool(self._verdict(rel))
        for base, sub in self.scoped:
            if rel.startswith(base + "/"):
                v = sub._verdict(rel[len(base) + 1:])
                if v is not None:
                    verdict = v
        return verdict

    def prunable_dir(self, rel_dir: str) -> bool:
        """True when nothing beneath directory `rel_dir` can be selected, so a
        walk may skip it without listing it. Conservative: any negation rule
        that could re-include a path under it (unanchored, or anchored at or
        below this directory) keeps the directory walkable."""
        rel_dir = rel_dir.replace("\\", "/").strip("/")
        if not rel_dir or not self.excluded(rel_dir + "/_"):
            return False
        prefix = rel_dir + "/"
        rule_sets = [self.rules[self._default_count:]] \
            + [sub.rules for _b, sub in self.scoped]
        for rules in rule_sets:
            for pat, negate, _dir_only, anchored in rules:
                if not negate:
                    continue
                if not anchored:
                    # `!.env.example` style: could match at any depth
                    return False
                if pat.startswith(prefix) or prefix.startswith(pat + "/") \
                        or any(ch in pat.split("/")[0] for ch in "*?["):
                    return False
        return True


def _rel_posix(p: Path, root: Path) -> str | None:
    """`p` relative to `root` as a POSIX string ("." for the root itself)."""
    try:
        r = p.relative_to(root).as_posix()
    except ValueError:
        return None
    return r or "."


def is_venv_dir(path: Path) -> bool:
    """True when `path` is the root of a Python virtual environment (it holds
    a ``pyvenv.cfg``, whatever the directory is named), so its site-packages
    is never indexed as project code."""
    try:
        return (path / "pyvenv.cfg").is_file()
    except OSError:
        return False


def _walk(path: Path, matcher: "Matcher | None" = None,
          root: Path | None = None) -> list[Path]:
    """Return every file under `path`, following symlinked directories but
    visiting each real directory once (cycle-safe) and skipping virtualenvs.

    With a `matcher`, directories it excludes are pruned before descending and
    nested ignore files are read as each directory is entered."""
    visited: set[Path] = set()
    root_r = Path(root).resolve() if root is not None else None
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
        if matcher is not None and root_r is not None:
            rel_d = _rel_posix(real, root_r)
            if rel_d is not None and rel_d != "." and d != path:
                if matcher.prunable_dir(rel_d):
                    continue
            if rel_d is not None and matcher.nested_names \
                    and rel_d not in ("", "."):
                for name in matcher.nested_names:
                    f = d / name
                    if f.is_file():
                        try:
                            matcher.add_scoped(rel_d, fileio.read_lines(f))
                        except OSError:
                            pass
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

    Directory walks keep files whose extension is in `exts` (plus the
    non-sensitive env-template and public-key carve-outs of BUG-19) and that
    `matcher` does not exclude; a file named explicitly is exempt from ignore
    rules. A sensitive file (secret/credential material) is NEVER selected --
    not even when named explicitly -- so no secret can reach the index or a
    model prompt (BUG-03)."""
    out: list[Path] = []
    seen: set[Path] = set()
    root_r = Path(root).resolve()

    def keep(f: Path, explicit: bool) -> None:
        r = f.resolve()
        if r in seen:
            return
        try:
            rel = r.relative_to(root_r).as_posix()
        except ValueError:
            # BUG-02: a file outside `root` must never be selected under a fake
            # root-relative basename (`f.name`). Skip it cleanly.
            return
        if not _selectable_suffix(f.name, f.suffix.lower(), exts):
            return
        # A secret/credential file is never selected, whatever the extension
        # filter or a `!.env` style re-inclusion says.
        if security.is_sensitive_path(rel):
            return
        if not explicit:
            if matcher.excluded(rel):
                return
        seen.add(r)
        out.append(f)

    for p in paths:
        path = Path(p)
        if path.is_file():
            keep(path, explicit=True)
        elif path.is_dir():
            for f in _walk(path, matcher, root):
                keep(f, explicit=False)
    return out


def _selectable_suffix(name: str, suffix: str, exts: set[str]) -> bool:
    """Whether a file named `name` with `suffix` may be selected even when its
    extension is not in `exts`: the documented env templates (`.env.example`,
    `.env.sample`, `.env.template`, `.env.dist`, `.env.defaults`) and public
    keys (`*.pub`) that are NOT sensitive (BUG-19)."""
    if suffix in exts:
        return True
    low = name.lower()
    if low in security.ENV_TEMPLATE_NAMES:
        return True
    if suffix == ".pub" and not security.is_sensitive_path(low):
        return True
    return False


# --------------------------------------------------------------------------
# Explicit context pinning: `@file` and `@Class.method` mentions in a prompt
# --------------------------------------------------------------------------
# The planner/model may want a file (or one definition) in context no matter
# what automatic retrieval scores. These helpers turn a prompt's @-mentions
# (and an API client's `pinned` list) into resolved, already-read spans that
# the gather and answer phases place ahead of every automatically chosen item.

# A mention starts at a boundary, so `user@example.com` is not one. The body
# accepts relative paths, qualnames and `file.py:symbol`; trailing sentence
# punctuation is trimmed after the match.
_MENTION_RE = re.compile(r"(?<![\w.])@([A-Za-z0-9_][A-Za-z0-9_./:\\-]*)")
_PY_FENCE_RE = re.compile(r"```(?:python|py|python3|py3)\b[^\n]*\n(.*?)```",
                          re.DOTALL | re.IGNORECASE)
_TRAILING_PUNCT = ".,;!?)]}>\"'`"
_MAX_PIN_SCAN_FILES = 4000
_MAX_SYMBOL_SCAN_FILES = 500


@dataclass
class PinnedContext:
    """One explicitly pinned file or AST symbol definition.

    `content` is the exact source text (already read from disk), so the prompt
    builder never has to hydrate an index chunk again. `start_line` and
    `end_line` are 1-based inclusive, matching `Chunk.start`/`Chunk.end` and
    the numbered SOURCE blocks elsewhere in the harness.
    """

    raw_token: str
    kind: str                    # "file" | "symbol"
    path: str                    # project-relative, POSIX
    symbol_name: str = ""
    start_line: int = 0
    end_line: int = 0
    content: str = ""


def _split_root_index(index_or_root) -> tuple[Path | None, object | None]:
    """(root, index) for an Index, a project root Path/str, or None."""
    if index_or_root is None:
        return None, None
    if isinstance(index_or_root, (str, os.PathLike)):
        return Path(index_or_root).expanduser().resolve(), None
    root = getattr(index_or_root, "root", None)
    if root is not None and (hasattr(index_or_root, "chunks")
                             or hasattr(index_or_root, "find_symbol")
                             or hasattr(index_or_root, "files")):
        return Path(root).expanduser().resolve(), index_or_root
    return None, None


def _workspace_files(root: Path | None, index=None) -> list[str]:
    """Project-relative files a pin may name, honouring ignore rules.

    An index already knows the selected files, so it is the fast path; a
    bounded workspace walk adds files created since the last index (and makes
    pinning work before the first index exists).
    """
    if root is None:
        return sorted(getattr(index, "files", []) or [])
    found: set[str] = set(getattr(index, "files", []) or [])
    matcher = Matcher.from_files(root)
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        base = "" if d == root else d.relative_to(root).as_posix()
        dirnames[:] = [
            name for name in dirnames
            if not is_venv_dir(d / name)
            and not matcher.prunable_dir(f"{base}/{name}" if base else name)
        ]
        for name in sorted(filenames):
            rel = f"{base}/{name}" if base else name
            if matcher.excluded(rel):
                continue
            found.add(rel)
            scanned += 1
            if scanned >= _MAX_PIN_SCAN_FILES:
                return sorted(found)
    return sorted(found)


def _read_lines(root: Path | None, rel: str) -> list[str]:
    if root is None or not rel:
        return []
    try:
        return fileio.read_lines(root / rel)
    except OSError:
        return []


def _has_known_extension(token: str) -> bool:
    suffix = Path(token).suffix.lower()
    return suffix in CODE_EXTS or suffix in REFERENCE_EXTS or suffix in {
        ".md", ".html", ".yaml", ".yml", ".toml", ".json", ".csv",
        ".txt", ".rst", ".css", ".scss", ".less",
    }


def _resolve_file(token: str, files: list[str]) -> str | None:
    """Resolve a typed file reference to one indexed/workspace path.

    Exact relative paths and unique basenames win. A case-insensitive prefix
    or close match is accepted only when it identifies exactly one file, so a
    bare `@__init__.py` never silently picks one package's file.
    """
    want = fileio.normalise_rel(token).lower().strip("/")
    if not want or not files:
        return None
    exact = [f for f in files if f.lower() == want]
    if exact:
        return exact[0]

    base = want.rsplit("/", 1)[-1]
    base_hits = [f for f in files if f.rsplit("/", 1)[-1].lower() == base]
    if len(base_hits) == 1:
        return base_hits[0]

    suffix = [f for f in files if f.lower().endswith("/" + want)]
    if len(suffix) == 1:
        return suffix[0]

    prefix = [f for f in files if f.lower().startswith(want)]
    if len(prefix) == 1:
        return prefix[0]

    base_prefix = [
        f for f in files if f.rsplit("/", 1)[-1].lower().startswith(base)
    ]
    if len(base_prefix) == 1:
        return base_prefix[0]

    # Fuzzy fallbacks: only an unambiguous close match is useful. Paths are
    # tried first, then basenames (the way a person types `@gui.py`).
    close = difflib.get_close_matches(want, [f.lower() for f in files],
                                      n=2, cutoff=0.72)
    if len(close) == 1:
        for f in files:
            if f.lower() == close[0]:
                return f
    close_base = difflib.get_close_matches(
        base, [f.rsplit("/", 1)[-1].lower() for f in files],
        n=2, cutoff=0.75)
    if len(close_base) == 1:
        for f in files:
            if f.rsplit("/", 1)[-1].lower() == close_base[0]:
                return f
    return None


_SYMBOL_CACHE_MAX = 512
_SYMBOL_CACHE: OrderedDict[str, tuple[tuple, list]] = OrderedDict()
_SYMBOL_CACHE_LOCK = threading.Lock()


def _symbol_records(root: Path | None, index=None) -> list:
    """Chunks (or lightweight AST/tree-sitter chunks for a root with no index)."""
    if index is not None:
        return list(getattr(index, "chunks", []) or [])
    if root is None:
        return []
    from .chunker import chunk_source
    from .chunker_ts import LANG_BY_EXT, TS_AVAILABLE, chunk_file_ts

    files = _workspace_files(root, None)
    code_exts = PY_EXTS | (set(LANG_BY_EXT) if TS_AVAILABLE else set())
    code_files = [f for f in files if Path(f).suffix.lower() in code_exts]
    code_files = code_files[:_MAX_SYMBOL_SCAN_FILES]
    sig_parts: list[tuple] = []
    for rel in code_files:
        try:
            st = (root / rel).stat()
            sig_parts.append((rel, st.st_mtime_ns, st.st_size))
        except OSError:
            sig_parts.append((rel, 0, 0))
    sig = tuple(sig_parts)
    key = str(root)
    with _SYMBOL_CACHE_LOCK:
        hit = _SYMBOL_CACHE.get(key)
        if hit is not None and hit[0] == sig:
            _SYMBOL_CACHE.move_to_end(key)
            return hit[1]
    records: list = []
    for rel in code_files:
        try:
            text = fileio.read_text(root / rel)
            if len(text) > 2_000_000:
                continue
            suffix = Path(rel).suffix.lower()
            if suffix == ".py":
                records.extend(chunk_source(rel, text))
            elif suffix in LANG_BY_EXT and TS_AVAILABLE:
                records.extend(chunk_file_ts(rel, text, suffix))
        except (OSError, SyntaxError, ValueError, RuntimeError):
            continue
    with _SYMBOL_CACHE_LOCK:
        _SYMBOL_CACHE[key] = (sig, records)
        _SYMBOL_CACHE.move_to_end(key)
        while len(_SYMBOL_CACHE) > _SYMBOL_CACHE_MAX:
            _SYMBOL_CACHE.popitem(last=False)
    return records


def _symbol_hits(token: str, file: str | None, root: Path | None,
                 index=None) -> list:
    """Chunks defining `token` (exact qualname, dotted suffix, or member)."""
    records = _symbol_records(root, index)
    if not records and index is not None and hasattr(index, "find_symbol"):
        try:
            records = list(index.find_symbol(token) or [])
        except Exception:
            records = []
    if file:
        key = fileio.path_key(file)
        records = [c for c in records
                   if fileio.path_key(getattr(c, "file", "")) == key]
    want = token.strip()
    if not want:
        return []
    low = want.lower()
    hits = [c for c in records if getattr(c, "qualname", "") == want]
    if not hits:
        hits = [c for c in records
                if getattr(c, "qualname", "").lower() == low]
    if not hits:
        hits = [c for c in records
                if getattr(c, "qualname", "").lower().endswith("." + low)]
    if not hits:
        base = low.rsplit(".", 1)[-1]
        hits = [c for c in records
                if base in {str(m).lower()
                            for m in getattr(c, "members", ())}]
    # A split unit's continuation chunks all match the base qualname; keep
    # the first part only, which carries the signature and docstring.
    out, seen = [], set()
    for c in hits:
        key = (getattr(c, "file", ""), getattr(c, "qualname", ""),
               getattr(c, "start", 0), getattr(c, "end", 0))
        if key in seen or getattr(c, "part_of", ""):
            continue
        seen.add(key)
        out.append(c)
    return out


def _ast_symbol_span(root: Path | None, rel: str, token: str
                     ) -> tuple[int, int] | None:
    """Exact source span of `Class.method`/`method` via `chunker.locate_symbol_span`.

    A small class is indexed as one whole-class chunk, so a pin for one of its
    methods would otherwise quote the entire class. Delegating to
    `locate_symbol_span` gives the method its own definition boundary for
    Python as well as JS/TS/Go/Rust and other tree-sitter languages.
    """
    if root is None or not rel or not token:
        return None
    from .chunker import locate_symbol_span
    hit = locate_symbol_span(root / rel, token)
    if hit is None:
        return None
    return hit[0], hit[1]


def _symbol_pin(raw_token: str, token: str, chunk, root: Path | None
                ) -> PinnedContext | None:
    rel = getattr(chunk, "file", "")
    if not rel:
        return None
    start = int(getattr(chunk, "start", 0) or 0)
    end = int(getattr(chunk, "end", start) or start)
    if getattr(chunk, "kind", "") == "classheader":
        end = max(end, int(getattr(chunk, "scope_end", 0) or 0))
    span = _ast_symbol_span(root, rel, token)
    if span:
        start, end = span
    lines = _read_lines(root, rel)
    if lines:
        start = max(1, min(start or 1, len(lines)))
        end = max(start, min(end or start, len(lines)))
        content = "\n".join(lines[start - 1:end])
    else:
        content = str(getattr(chunk, "source", "") or "")
        if not content:
            return None
    name = token.strip()
    return PinnedContext(raw_token=raw_token, kind="symbol", path=rel,
                         symbol_name=name, start_line=start,
                         end_line=end, content=content)


def _file_pin(raw_token: str, rel: str, root: Path | None) -> PinnedContext:
    lines = _read_lines(root, rel)
    while lines and lines[-1] == "":
        lines.pop()
    content = "\n".join(lines)
    return PinnedContext(raw_token=raw_token, kind="file", path=rel,
                         start_line=1 if lines else 0,
                         end_line=len(lines), content=content)


def _trim_token(token: str) -> str:
    token = (token or "").strip()
    while token and token[-1] in _TRAILING_PUNCT:
        token = token[:-1]
    return token


def _resolve_token(token: str, raw_token: str, index_or_root
                   ) -> list[PinnedContext]:
    root, index = _split_root_index(index_or_root)
    files = _workspace_files(root, index)
    token = _trim_token(token)
    if not token:
        return []

    # `file.py:symbol` -- scope the symbol lookup to that file.
    if ":" in token:
        left, right = token.split(":", 1)
        rel = _resolve_file(left, files)
        if rel:
            if not right.strip():
                return [_file_pin(raw_token, rel, root)]
            hits = _symbol_hits(right.strip(), rel, root, index)
            pins = [p for c in hits
                    for p in [_symbol_pin(raw_token, right.strip(), c, root)]
                    if p is not None]
            return pins or [_file_pin(raw_token, rel, root)]

    file_rel = _resolve_file(token, files)
    hits = _symbol_hits(token, None, root, index)
    file_like = ("/" in token or "\\" in token
                 or _has_known_extension(token)
                 or (file_rel is not None
                     and token.lower() == file_rel.lower()))
    if hits and not file_like:
        pins = [p for c in hits
                for p in [_symbol_pin(raw_token, token, c, root)]
                if p is not None]
        if pins:
            return pins
    if file_rel:
        return [_file_pin(raw_token, file_rel, root)]
    if hits:
        return [p for c in hits
                for p in [_symbol_pin(raw_token, token, c, root)]
                if p is not None]
    return []


def _decorator_spans(prompt: str) -> list[tuple[int, int]]:
    """Character spans of decorator lines inside fenced Python code."""
    spans: list[tuple[int, int]] = []
    for block in _PY_FENCE_RE.finditer(prompt):
        start = block.start(1)
        pos = 0
        for line in block.group(1).splitlines(True):
            if line.lstrip().startswith("@") and len(line.lstrip()) > 1:
                spans.append((start + pos, start + pos + len(line)))
            pos += len(line)
    return spans


def _pin_key(pin: PinnedContext) -> tuple:
    return (pin.kind, fileio.path_key(pin.path), pin.symbol_name,
            pin.start_line, pin.end_line)


def _dedupe_pins(pins: list[PinnedContext]) -> list[PinnedContext]:
    out: list[PinnedContext] = []
    seen: set[tuple] = set()
    for pin in pins:
        key = _pin_key(pin)
        if key in seen:
            continue
        seen.add(key)
        out.append(pin)
    return out


def parse_pinned_references(prompt: str, index_or_root
                            ) -> list[PinnedContext]:
    """Resolve `@file` and `@symbol` mentions in `prompt`.

    Files match an exact relative path, a unique basename, or an unambiguous
    case-insensitive prefix/fuzzy match. Symbols match an exact qualname
    (`Class.method`, `file.py:symbol`) or a unique indexed member name.
    Email addresses (`name@domain.com`) and decorator lines inside fenced
    Python blocks are ignored. Resolution order follows the prompt, and each
    resolved item is deduplicated by (kind, path, symbol, line span).
    """
    text = str(prompt or "")
    if not text:
        return []
    ignored = _decorator_spans(text)
    pins: list[PinnedContext] = []
    for match in _MENTION_RE.finditer(text):
        if any(a <= match.start() < b for a, b in ignored):
            continue
        token = _trim_token(match.group(1))
        if not token:
            continue
        pins.extend(_resolve_token(token, "@" + token, index_or_root))
    return _dedupe_pins(pins)


def resolve_pinned_references(values, index_or_root) -> list[PinnedContext]:
    """Resolve an explicit `pinned` request list (`["@a.py", "Class.run"]`).

    A dict item from the GUI's autocomplete uses its `insert`, `label` or
    `path` field. Leading `@` is optional; order is preserved and duplicate
    resolutions collapse.
    """
    if isinstance(values, str):
        values = [values]
    pins: list[PinnedContext] = []
    for value in values or ():
        if isinstance(value, dict):
            value = (value.get("insert") or value.get("label")
                     or value.get("path") or value.get("file") or "")
        token = _trim_token(str(value or "").strip().lstrip("@").strip())
        if not token:
            continue
        pins.extend(_resolve_token(token, "@" + token, index_or_root))
    return _dedupe_pins(pins)


def render_pinned_context(pins) -> str:
    """Prompt-ready `--- pinned ...` blocks, or "" when nothing resolved."""
    parts: list[str] = []
    for pin in pins or ():
        if pin.kind == "symbol":
            head = (f"--- pinned symbol {pin.symbol_name} "
                    f"({pin.path} L{pin.start_line}-{pin.end_line})")
        elif pin.kind == "skill":
            head = f"--- pinned skill {pin.symbol_name}"
            if pin.path and pin.path != "bundled":
                head += f" ({pin.path})"
        else:
            head = (f"--- pinned file {pin.path} "
                    f"(L{pin.start_line}-{pin.end_line})")
        parts.append(head + "\n" + (pin.content or ""))
    return "\n\n".join(parts)


def pinned_summary(pins) -> str:
    """`pedacito/llm.py, ScientificCalculator.evaluate_expression`."""
    labels: list[str] = []
    for pin in pins or ():
        label = pin.symbol_name if pin.kind == "symbol" else pin.path
        if label and label not in labels:
            labels.append(label)
    return ", ".join(labels)


def _match_score(query: str, text: str) -> float:
    """A fast, deterministic fuzzy score (0 = no match)."""
    if not query:
        return 1.0
    q, t = query.lower(), text.lower()
    if not q or not t:
        return 0.0
    base = t.rsplit("/", 1)[-1]
    if t == q:
        return 100.0
    if base == q:
        return 96.0
    if t.endswith("/" + q):
        return 92.0
    if base.startswith(q):
        return 84.0
    if t.startswith(q):
        return 78.0
    if q in base:
        return 64.0 - min(20.0, (len(base) - len(q)) * 0.2)
    if q in t:
        return 54.0 - min(20.0, (len(t) - len(q)) * 0.1)
    # subsequence: `pdcgui` finds `pedacito/gui/server.py`
    pos = 0
    for ch in q:
        pos = t.find(ch, pos)
        if pos < 0:
            break
        pos += 1
    else:
        return 34.0
    ratio = difflib.SequenceMatcher(None, q, t).ratio()
    return ratio * 40.0 if ratio >= 0.55 else 0.0


def suggest_context(index_or_root, query: str = "", limit: int = 20) -> list[dict]:
    """Fast autocomplete entries for the GUI's `@` menu.

    Returns up to `limit` dicts shaped `{"kind", "label", "insert", "detail"}`
    for workspace files and indexed AST symbols, sorted by a fuzzy score.
    """
    root, index = _split_root_index(index_or_root)
    q = str(query or "").strip().lstrip("@")
    try:
        limit = max(1, min(50, int(limit or 20)))
    except (TypeError, ValueError):
        limit = 20
    files = _workspace_files(root, index)

    scored: list[tuple[float, int, dict]] = []
    for rel in files:
        score = _match_score(q, rel)
        if score <= 0:
            continue
        scored.append((score, 0, {
            "kind": "file", "label": rel, "insert": "@" + rel,
            "detail": "workspace file"}))
    seen_symbols: set[str] = set()
    for chunk in _symbol_records(root, index):
        if getattr(chunk, "kind", "") in ("module", "reference"):
            continue
        if getattr(chunk, "part_of", ""):
            continue
        name = str(getattr(chunk, "qualname", "") or "")
        if not name or name in seen_symbols:
            continue
        score = _match_score(q, name)
        # A symbol's file is useful context in the query too.
        if score <= 0 and getattr(chunk, "file", ""):
            score = _match_score(q, str(chunk.file)) * 0.8
        seen_symbols.add(name)
        if score > 0:
            detail = (f"{chunk.file}:{chunk.start}-{chunk.end} "
                      f"{getattr(chunk, 'kind', 'symbol')}")
            scored.append((score, 1, {
                "kind": "symbol", "label": name, "insert": "@" + name,
                "detail": detail}))
        # A small class is indexed whole; its methods are still AST symbols a
        # user can pin (`@Class.method`). Offer them as first-class entries.
        members = getattr(chunk, "members", ()) or ()
        for member in members:
            child = f"{name}.{member}"
            if child in seen_symbols:
                continue
            child_score = _match_score(q, child)
            if child_score <= 0:
                continue
            seen_symbols.add(child)
            scored.append((child_score, 1, {
                "kind": "symbol", "label": child,
                "insert": "@" + child,
                "detail": f"{chunk.file} (method of {name})"}))
    scored.sort(key=lambda row: (-row[0], row[1], len(row[2]["label"]),
                                 row[2]["label"].lower()))
    return [row[2] for row in scored[:limit]]
