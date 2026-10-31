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

import shutil
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


def write(path: Path, text: str, style: TextStyle | None = None) -> None:
    """Write LF-normalised text to disk, restoring the given encoding style."""
    style = style or TextStyle()
    out = text
    if style.newline != "\n":
        out = out.replace("\n", style.newline)
    data = out.encode("utf-8")
    if style.bom:
        data = b"\xef\xbb\xbf" + data
    Path(path).write_bytes(data)


def backup_copy(src: Path, dest: Path, keep_existing: bool = True) -> None:
    """Copy `src`'s exact bytes to `dest`, creating parent directories.

    With `keep_existing`, an existing `dest` is left alone: backup snapshot
    directories are shared by every write of a run, and the FIRST copy of a
    file is the original that `restore` must bring back."""
    if keep_existing and Path(dest).exists():
        return
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dest))


def normalise_rel(name: str) -> str:
    """Turn a user- or model-typed path (backslashes, quotes, backticks) into
    the forward-slash relative path the index stores."""
    return (name or "").strip().strip("`\"'").replace("\\", "/")
