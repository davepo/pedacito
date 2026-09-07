"""Text file I/O that preserves a file's on-disk encoding.

Usage
-----
Every read or write of a project file goes through this module instead of
`pathlib.Path.read_text` / `write_text` directly:

    from . import fileio

    text, style = fileio.read(path)      # text is always LF-normalised
    ...
    fileio.write(path, new_text, style)  # written back in the original style

Why this exists: `Path.read_text()` uses universal newlines, so a CRLF file
comes back as LF, and `Path.write_text()` on Windows translates LF back to
CRLF on the way out. That silently converts an LF file to CRLF on save,
rewriting every line of a file only three lines of which changed. It also
breaks search/replace matching, since the text a model read no longer matches
the bytes on disk.

The fix: read raw bytes, detect the line-ending style and BOM, normalise to
LF for all internal processing (chunking, matching, diffing), and write back
in the file's original style.
"""

from __future__ import annotations

from pathlib import Path

BOM = "\ufeff"


# --------------------------------------------------------------------------
# Encoding descriptor
# --------------------------------------------------------------------------
class TextStyle:
    """Records a file's line-ending convention and BOM presence."""

    __slots__ = ("newline", "bom")

    def __init__(self, newline: str = "\n", bom: bool = False):
        """Record a file's line-ending string and whether it has a BOM."""
        # newline: "\n", "\r\n", or "\r" -- whichever the file actually uses.
        # bom: whether a UTF-8 byte-order-mark was present at the start.
        self.newline = newline
        self.bom = bom

    def __repr__(self) -> str:
        """Compact human-readable form, e.g. "<CRLF +BOM>"."""
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


# --------------------------------------------------------------------------
# Public read/write API
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Path helpers
# --------------------------------------------------------------------------
def normalise_rel(name: str) -> str:
    """Convert a user- or model-typed path (possibly using backslashes, quotes,
    or backticks) into the plain forward-slash relative path the index uses.

    The index always stores POSIX-style paths so a project indexed on one OS
    reads correctly on another; this function is the single place that accepts
    whatever form a person or model actually typed and normalises it.
    """
    return (name or "").strip().strip("`\"'").replace("\\", "/")
