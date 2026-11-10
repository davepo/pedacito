"""Text file I/O that preserves a file's on-disk encoding.

Every read or write of a project file goes through this module:

    text, style = fileio.read(path)      # text is always LF-normalised
    fileio.write(path, new_text, style)  # written back in the original style

`Path.read_text()` / `write_text()` would silently convert line endings (and
break search/replace matching against what the model read). Here the raw bytes
are read, the line-ending style and BOM detected, all processing happens on
LF text, and writes restore the original style.
"""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from collections import OrderedDict
from pathlib import Path

BOM = "\ufeff"


class TextStyle:
    """Records a file's line-ending convention and BOM presence."""

    __slots__ = ("bom", "newline")

    def __init__(self, newline: str = "\n", bom: bool = False):
        self.newline = newline      # "\n", "\r\n" or "\r"
        self.bom = bom              # a UTF-8 byte-order mark led the file

    def __repr__(self) -> str:
        nl = {"\r\n": "CRLF", "\n": "LF", "\r": "CR"}.get(self.newline, repr(self.newline))
        return f"<{nl}{' +BOM' if self.bom else ''}>"


def detect(raw: bytes) -> TextStyle:
    """Inspect raw file bytes and return their line-ending style and BOM flag."""
    crlf = raw.count(b"\r\n")
    lf = raw.count(b"\n") - crlf
    cr = raw.count(b"\r") - crlf
    if crlf and crlf >= lf and crlf >= cr:
        nl = "\r\n"
    elif cr and cr > lf:
        nl = "\r"
    else:
        nl = "\n"
    return TextStyle(nl, raw.startswith(b"\xef\xbb\xbf"))


def read(path: Path, errors: str = "replace") -> tuple[str, TextStyle]:
    """Read a file and return (text normalised to LF, its original style)."""
    raw = Path(path).read_bytes()
    style = detect(raw)
    text = raw.decode("utf-8", errors)
    if text.startswith(BOM):
        text = text[1:]
    if style.newline != "\n":
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text, style


def read_text(path: Path, errors: str = "replace") -> str:
    """Read a file's LF-normalised text when the caller won't write it back."""
    return read(path, errors)[0]


def read_lines(path: Path) -> list[str]:
    """Read a file and split its LF-normalised text into lines."""
    return read_text(path).split("\n")


# Bounded LRU keyed by (path, mtime_ns, size) for the scan-heavy paths (grep
# sweeps, call-site peeks) that re-read the same files step after step.
# Callers that must see the newest bytes (tool_read, the edit path) use
# read_lines() directly.
_LINE_CACHE: OrderedDict[tuple, tuple[list[str], int]] = OrderedDict()
_LINE_CACHE_BYTES = 0
_LINE_CACHE_MAX_BYTES = 8_000_000


def read_lines_cached(path: Path) -> list[str]:
    """read_lines() backed by the LRU cache. The returned list is SHARED with
    the cache -- callers must not mutate it."""
    global _LINE_CACHE_BYTES
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return read_lines(p)          # let the caller see the real error
    key = (str(p), st.st_mtime_ns, st.st_size)
    hit = _LINE_CACHE.get(key)
    if hit is not None:
        _LINE_CACHE.move_to_end(key)
        return hit[0]
    lines = read_lines(p)
    _LINE_CACHE[key] = (lines, st.st_size)
    _LINE_CACHE_BYTES += st.st_size
    while _LINE_CACHE_BYTES > _LINE_CACHE_MAX_BYTES and len(_LINE_CACHE) > 1:
        _, (_, size) = _LINE_CACHE.popitem(last=False)
        _LINE_CACHE_BYTES -= size
    return lines


def write_bytes(path: Path, data: bytes) -> None:
    """Atomically replace `path` with `data`, never rewriting in place.

    Ratchet checkpoints are HARD LINKS to earlier inodes of the same file.
    An in-place `open(path, "wb")` truncates every link sharing that inode,
    quietly rewriting checkpoint history. A temp file plus `os.replace`
    swaps the path onto a fresh inode and leaves the old one (and its links)
    untouched; the original permission bits are preserved."""
    p = Path(path)
    if p.is_symlink():
        p = p.resolve()          # replace through the link, not the link itself
    try:
        mode = stat.S_IMODE(p.stat().st_mode)
    except OSError:
        mode = 0o644
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=p.name + ".", suffix=".pedacito-tmp",
                               dir=str(p.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.chmod(tmp, mode)
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write(path: Path, text: str, style: TextStyle | None = None) -> None:
    """Write LF-normalised text to disk, restoring the given encoding style.

    The actual replacement is `write_bytes`: an atomic temp-file + rename,
    never an in-place rewrite, so a hardlinked checkpoint keeps its bytes."""
    style = style or TextStyle()
    out = text
    if style.newline != "\n":
        out = out.replace("\n", style.newline)
    data = out.encode("utf-8")
    if style.bom:
        data = b"\xef\xbb\xbf" + data
    write_bytes(path, data)


def copy_file_atomic(src: Path, dest: Path) -> None:
    """Copy `src` to `dest` through a temp file, preserving mode and mtime.

    Restores (undo, `pedacito restore`) must never copy into an existing
    inode: the destination may be hardlinked to a ratchet checkpoint, and an
    in-place `shutil.copy2` would rewrite that checkpoint too."""
    src = Path(src)
    p = Path(dest)
    if p.is_symlink():
        p = p.resolve()          # replace through the link, not the link itself
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=p.name + ".", suffix=".pedacito-tmp",
                               dir=str(p.parent))
    os.close(fd)
    try:
        shutil.copy2(str(src), tmp)
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def backup_copy(src: Path, dest: Path, keep_existing: bool = True) -> None:
    """Copy `src`'s exact bytes to `dest`, creating parent directories.

    With `keep_existing`, an existing `dest` is left alone: backup snapshot
    directories are shared by every write of a run, and the FIRST copy of a
    file is the original that `restore` must bring back. Replacement is
    atomic, so this is also safe for the reverse direction (`undo` restoring
    a working file that may share an inode with a checkpoint)."""
    if keep_existing and Path(dest).exists():
        return
    copy_file_atomic(src, dest)


def created_manifest(backup_dir) -> Path:
    """Where a snapshot lists the files its run CREATED (a sibling file, so a
    restore of the snapshot's own tree never copies it back)."""
    d = Path(backup_dir)
    return d.parent / f"{d.name}.created"


def note_created(backup_dir, rel: str) -> None:
    """Record that this run created `rel` (so `restore --delete-created` can
    remove it). Best effort."""
    try:
        m = created_manifest(backup_dir)
        m.parent.mkdir(parents=True, exist_ok=True)
        have = m.read_text(encoding="utf-8").splitlines() if m.exists() else []
        if rel not in have:
            with m.open("a", encoding="utf-8") as fh:
                fh.write(rel + "\n")
    except OSError:
        pass


def normalise_rel(name: str) -> str:
    """Turn a user- or model-typed path (backslashes, quotes, backticks) into
    the forward-slash relative path the index stores."""
    return (name or "").strip().strip("`\"'").replace("\\", "/")


def path_key(name: str) -> str:
    """Comparison key for a project path: lower-case, forward slashes, and any
    leading `./` segments removed. (`str.lstrip("./")` strips CHARACTERS, so
    it turned `.hidden/x` into `hidden/x` and `../x` into `x`.)"""
    p = normalise_rel(name).lower()
    while p.startswith("./"):
        p = p[2:]
    return p
