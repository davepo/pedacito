"""Structural chunking for non-Python reference files (config, data, docs).

Usage
-----
    from .references import chunk_reference

    chunks = chunk_reference("data/config.csv", Path("data/config.csv"))

These files are not parsed as code. The goal is that the model knows a
`config.csv` exists, what columns it has, and roughly how big it is, so it
can `read` or `grep` it when a task actually depends on it -- indexing the
full *contents* of a 50,000-row CSV would be pointless and would overwhelm
the map. So each reference file gets one or a few structural chunks: enough
shape to decide whether to look closer, each with a real line range so
`read` still works against the original file.

The chunking strategy is chosen by file extension: CSV/TSV get a column
summary and sample rows; JSON gets its top-level key/item shape; INI/TOML get
one chunk per [section]; Markdown/RST get one chunk per heading; anything
else falls back to fixed-size line blocks.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path

from . import fileio
from .chunker import Chunk

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
INI_SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]")
TOML_KEY_RE = re.compile(r"^\s*([A-Za-z_][\w.-]*)\s*=")
BLOCK_LINES = 150   # fallback chunk size, in lines, for unstructured text


# --------------------------------------------------------------------------
# Chunk construction helper
# --------------------------------------------------------------------------
def _sha(t: str) -> str:
    """Short content hash used to detect whether a chunk's text has changed."""
    return hashlib.sha256(t.encode("utf-8", "replace")).hexdigest()[:16]


def _mk(rel: str, qualname: str, start: int, end: int, sig: str,
        docline: str = "", src: str = "") -> Chunk:
    """Build one reference Chunk with a single-line docline (multi-line
    docstrings would break the one-line-per-entry layout of the file map)."""
    docline = " ".join(docline.split())
    return Chunk(file=rel, qualname=qualname, kind="reference", start=start, end=end,
                 signature=sig, docline=docline, source=src, sha=_sha(src or sig + docline))


# --------------------------------------------------------------------------
# Per-format chunkers (internal)
# --------------------------------------------------------------------------
def _tabular(rel: str, text: str, delim: str) -> list[Chunk]:
    """CSV/TSV: one chunk summarising column names, row count, and two sample
    rows, so the model can see the shape without reading the whole table."""
    lines = text.splitlines()
    if not lines:
        return []
    try:
        reader = csv.reader(io.StringIO("\n".join(lines[:50])), delimiter=delim)
        rows = list(reader)
    except csv.Error:
        rows = [lines[0].split(delim)]
    header = rows[0] if rows else []
    ncols = len(header)
    sig = f"table: {ncols} columns x {len(lines) - 1} data rows"
    doc = "columns: " + ", ".join(h.strip()[:40] for h in header[:25])
    if ncols > 25:
        doc += f", ... (+{ncols - 25})"
    sample = " / ".join(l.strip() for l in lines[1:3])
    if sample:
        doc += f" | sample: {sample[:200]}"
    return [_mk(rel, "table", 1, len(lines), sig, doc, "\n".join(lines[:3]))]


def _structured(rel: str, text: str) -> list[Chunk]:
    """JSON: one chunk describing the top-level shape (object keys, array
    item type, or scalar type)."""
    lines = text.splitlines()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return [_mk(rel, "content", 1, len(lines), f"json ({len(lines)} lines, unparseable)")]
    if isinstance(data, dict):
        keys = list(data.keys())
        sig = f"json object, {len(keys)} top-level keys"
        doc = "keys: " + ", ".join(str(k)[:40] for k in keys[:30])
        if len(keys) > 30:
            doc += f", ... (+{len(keys) - 30})"
    elif isinstance(data, list):
        sig = f"json array, {len(data)} items"
        first = data[0] if data else None
        doc = ("item keys: " + ", ".join(map(str, list(first.keys())[:20]))
               if isinstance(first, dict) else f"item type: {type(first).__name__}")
    else:
        sig, doc = f"json {type(data).__name__}", ""
    return [_mk(rel, "root", 1, len(lines), sig, doc)]


def _sectioned(rel: str, text: str, rx: re.Pattern, label: str) -> list[Chunk]:
    """INI/TOML-style files: one chunk per [section], with real line ranges."""
    lines = text.splitlines()
    marks = [(i, m.group(1)) for i, line in enumerate(lines, 1)
             if (m := rx.match(line))]
    if not marks:
        return [_mk(rel, "content", 1, len(lines), f"{label}, {len(lines)} lines")]
    out = []
    for j, (ln, name) in enumerate(marks):
        end = marks[j + 1][0] - 1 if j + 1 < len(marks) else len(lines)
        body = "\n".join(lines[ln - 1 : end])
        out.append(_mk(rel, name, ln, end, f"{label} section [{name}]", "", body))
    return out


def _headed(rel: str, text: str) -> list[Chunk]:
    """Markdown/RST: one chunk per heading, falling back to fixed-size blocks
    if no headings are found."""
    lines = text.splitlines()
    marks = [(i, m.group(1), m.group(2)) for i, line in enumerate(lines, 1)
             if (m := HEADING_RE.match(line))]
    if not marks:
        return _blocks(rel, lines, "text")
    out = []
    for j, (ln, hashes, title) in enumerate(marks):
        end = marks[j + 1][0] - 1 if j + 1 < len(marks) else len(lines)
        body = "\n".join(lines[ln - 1 : end])
        out.append(_mk(rel, title[:60], ln, end,
                       f"{'#' * len(hashes)} {title[:80]}", "", body))
    return out


def _blocks(rel: str, lines: list[str], label: str) -> list[Chunk]:
    """Fallback chunker: split unstructured text into fixed-size line blocks
    of BLOCK_LINES, each with a preview of its first non-blank line."""
    if not lines:
        return []
    out = []
    for s in range(0, len(lines), BLOCK_LINES):
        e = min(s + BLOCK_LINES, len(lines))
        body = "\n".join(lines[s:e])
        preview = next((l.strip() for l in lines[s:e] if l.strip()), "")[:80]
        out.append(_mk(rel, f"lines_{s + 1}", s + 1, e,
                       f"{label} block", preview, body))
    return out


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def chunk_reference(rel: str, path: Path, max_bytes: int = 2_000_000) -> list[Chunk]:
    """Produce structural chunks for one non-Python file, dispatching on its
    extension to the appropriate chunker above.

    Files larger than `max_bytes` are never read in full: only a header chunk
    noting the file's size is produced, since the model can still `read` or
    `grep` the file directly if a task needs it.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size > max_bytes:
        # Header only. Enough to know it exists and roughly what it contains.
        with path.open("rb") as fh:
            head = b"".join(next(fh, b"") for _ in range(5)).decode("utf-8", "replace")
        return [_mk(rel, "content", 1, 0,
                    f"{path.suffix.lstrip('.') or 'file'}, {size // 1024}KB "
                    f"(too large to index; use grep or read)",
                    head.splitlines()[0][:160] if head else "")]

    text = fileio.read_text(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        return _tabular(rel, text, ",")
    if ext == ".tsv":
        return _tabular(rel, text, "\t")
    if ext == ".json":
        return _structured(rel, text)
    if ext in (".ini", ".cfg", ".conf"):
        return _sectioned(rel, text, INI_SECTION_RE, "ini")
    if ext == ".toml":
        return _sectioned(rel, text, INI_SECTION_RE, "toml")
    if ext in (".md", ".rst"):
        return _headed(rel, text)
    return _blocks(rel, text.splitlines(), ext.lstrip(".") or "text")
