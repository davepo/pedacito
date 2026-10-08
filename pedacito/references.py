"""Structural chunking for non-Python reference files (config, data, docs, HTML).

Usage
-----
    from .references import chunk_reference

    chunks = chunk_reference("data/config.csv", Path("data/config.csv"))
    chunks = chunk_reference("site/index.html", Path("site/index.html"))

These files are not parsed as code. The goal is that the model knows a
`config.csv` exists, what columns it has, and roughly how big it is, so it
can `read` or `grep` it when a task actually depends on it -- indexing the
full *contents* of a 50,000-row CSV would be pointless and would overwhelm
the map. So each reference file gets one or a few structural chunks: enough
shape to decide whether to look closer, each with a real line range so
`read` still works against the original file.

The chunking strategy is chosen by file extension: CSV/TSV get a column
summary and sample rows; JSON gets its top-level key/item shape; INI/TOML get
one chunk per [section]; Markdown/RST get one chunk per heading; HTML gets
one chunk per top-level landmark element (header, nav, main, section,
article, aside, footer, form), with the id/class that identifies it and the
first heading text inside as a label; anything else falls back to fixed-size
line blocks.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from html.parser import HTMLParser
from pathlib import Path

from . import fileio
from .chunker import Chunk

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
INI_SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]")
TOML_KEY_RE = re.compile(r"^\s*([A-Za-z_][\w.-]*)\s*=")
BLOCK_LINES = 150   # fallback chunk size, in lines, for unstructured text

# HTML landmark elements: their presence at the top level of <body> gives the
# page a structure worth exposing. Nested landmarks are treated as detail
# within their containing section, not as peer sections -- a <section> inside
# a <main> belongs to the main section's chunk, not its own. That keeps the
# card readable and the chunk count sane on real pages.
HTML_LANDMARKS = {
    "header", "nav", "main", "section", "article", "aside",
    "footer", "form", "figure", "details", "dialog", "template",
}
HTML_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
# Elements whose text content is code, not prose: skip their data when
# collecting heading text (which otherwise would pick up a stray string
# from an inline <script>).
HTML_RAW = {"script", "style"}


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
# HTML (outline-based structural chunking)
# --------------------------------------------------------------------------
class _HtmlOutline(HTMLParser):
    """Streaming HTML walker that records, in order:

      * top-level landmark elements (header/nav/main/section/...) with their
        line range and their id or class, so a chunk can be labelled
        meaningfully
      * every heading, so a section's label can be its own first heading
      * external dependencies (script src, stylesheet href), which are the
        first thing you'd want to know about a page you haven't read

    Nothing about text content beyond the title is kept: this is the same
    "shape, not contents" contract the other reference chunkers honour.
    Standard-library `html.parser` is tolerant enough for the purpose --
    real-world HTML is rarely fully well-formed and this still produces a
    useful outline around the broken bits.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.sections: list[dict] = []       # {tag, line, end, id, class}
        self.headings: list[dict] = []       # {tag, line, text}
        self.deps: dict[str, list[str]] = {"script": [], "style": []}
        self.title = ""
        self._stack: list[dict] = []         # open landmark nodes
        self._in_title = False
        self._in_raw = 0                     # depth inside <script>/<style>
        self._heading: dict | None = None    # currently open heading
        self._heading_buf: list[str] = []

    # -- start tags --------------------------------------------------------
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        line = self.getpos()[0]
        if tag in HTML_RAW:
            self._in_raw += 1
            # Record the src/href of scripts and stylesheets as deps, but
            # don't descend into their text -- it's code, not outline.
            if tag == "script" and attrs.get("src"):
                self.deps["script"].append(attrs["src"])
            return
        if tag == "link" and attrs.get("rel") == "stylesheet" and attrs.get("href"):
            self.deps["style"].append(attrs["href"])
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in HTML_HEADINGS:
            self._heading = {"tag": tag, "line": line, "text": ""}
            self._heading_buf = []
        if tag in HTML_LANDMARKS and not self._stack:
            self._stack.append({
                "tag": tag, "line": line, "end": line,
                "id": attrs.get("id", ""),
                "class": attrs.get("class", ""),
            })

    # -- end tags ----------------------------------------------------------
    def handle_endtag(self, tag):
        if tag in HTML_RAW:
            self._in_raw = max(0, self._in_raw - 1)
            return
        if tag == "title":
            self._in_title = False
            return
        if tag in HTML_HEADINGS and self._heading is not None:
            self._heading["text"] = " ".join(self._heading_buf).strip()
            self.headings.append(self._heading)
            self._heading = None
            return
        # A landmark close: only pop if it matches the top of the stack, so
        # a stray </section> doesn't silently drop an <article>.
        if self._stack and self._stack[-1]["tag"] == tag:
            node = self._stack.pop()
            node["end"] = self.getpos()[0]
            self.sections.append(node)

    # -- text --------------------------------------------------------------
    def handle_data(self, data):
        if self._in_raw:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title = (self.title + " " + text).strip()
        if self._heading is not None:
            self._heading_buf.append(text)

    # -- finish ------------------------------------------------------------
    def close(self):
        super().close()
        # Landmarks left unclosed at EOF (a malformed or truncated file):
        # close them at the last line we know about rather than silently
        # losing the section. Same spirit as tree-sitter's error-tolerant
        # parse -- degrade, don't refuse.
        last = 1
        for s in self.sections:
            last = max(last, s.get("end", s["line"]))
        for h in self.headings:
            last = max(last, h["line"])
        while self._stack:
            node = self._stack.pop()
            node["end"] = last
            self.sections.append(node)
        self.sections.sort(key=lambda s: s["line"])


def _html(rel: str, text: str) -> list[Chunk]:
    """HTML: one chunk per top-level landmark element, plus a document
    header chunk carrying the title and external dependencies. Falls back to
    heading-based splitting when the file has headings but no landmarks, and
    to fixed-size line blocks when it has neither (a bare fragment, a
    minified one-liner, or a `<div>`-soup template).
    """
    lines = text.splitlines()
    nlines = len(lines)
    if nlines < 3:
        return _blocks(rel, lines, "html")

    parser = _HtmlOutline()
    try:
        parser.feed(text)
        parser.close()
    except Exception:
        # html.parser can raise on pathological input; fall back rather than
        # refusing to index the file at all.
        return _blocks(rel, lines, "html")

    if not parser.sections and parser.headings:
        return _html_heading_split(rel, text, parser.headings)
    if not parser.sections:
        return _blocks(rel, lines, "html")

    chunks: list[Chunk] = []
    headings = sorted(parser.headings, key=lambda h: h["line"])

    # -- document header chunk: title, deps, everything above the first
    # -- landmark. On a typical page that's <!doctype>, <head>, and the
    # -- opening <body> -- exactly the "what is this page" summary.
    first = parser.sections[0]["line"]
    head_end = max(1, first - 1)
    deps_bits = []
    if parser.deps["style"]:
        deps_bits.append("styles: " + ", ".join(parser.deps["style"][:6]))
    if parser.deps["script"]:
        deps_bits.append("scripts: " + ", ".join(parser.deps["script"][:6]))
    title = parser.title[:120]
    doc = " | ".join(b for b in ([title] + deps_bits) if b)[:200]
    chunks.append(_mk(
        rel, "document", 1, head_end,
        f"html document, {nlines} lines",
        doc,
        "\n".join(lines[:head_end]),
    ))

    # -- landmark sections --
    for sec in parser.sections:
        start = sec["line"]
        end = max(start, sec.get("end", start))
        # Label with the first heading inside the section, if any.
        label = ""
        for h in headings:
            if start <= h["line"] <= end:
                label = f"{h['tag'].upper()}: {h['text'][:80]}"
                break
        # Signature: the tag with its id (preferred) or first classes.
        sig = f"<{sec['tag']}"
        if sec["id"]:
            sig += f' id="{sec["id"]}"'
        elif sec["class"]:
            classes = sec["class"].split()[:2]
            sig += f' class="{",".join(classes)}"'
        sig += ">"
        qual = sec["id"] or sec["tag"]
        # Disambiguate when a page has two sections with the same id or two
        # anonymous <section> tags: chunk keys must be unique per file.
        if any(c.qualname == qual for c in chunks):
            qual = f"{qual}#{start}"
        body = "\n".join(lines[start - 1:end])
        chunks.append(_mk(rel, qual, start, end, sig, label, body))

    # -- trailing content after the last landmark --
    last_end = max(s.get("end", s["line"]) for s in parser.sections)
    if last_end < nlines:
        body = "\n".join(lines[last_end:])
        if body.strip():
            chunks.append(_mk(rel, "trailing", last_end + 1, nlines,
                              "html tail", "", body))
    return chunks


def _html_heading_split(rel: str, text: str, headings: list[dict]) -> list[Chunk]:
    """Fallback for landmark-less HTML: split on headings, the way markdown
    is split. A page that is a linear run of <h1>/<h2>/<p> has no landmark
    structure to show, but its heading hierarchy still tells the model where
    things are.
    """
    lines = text.splitlines()
    nlines = len(lines)
    headings = sorted(headings, key=lambda h: h["line"])
    out: list[Chunk] = []
    if headings and headings[0]["line"] > 1:
        end = headings[0]["line"] - 1
        body = "\n".join(lines[:end])
        if body.strip():
            out.append(_mk(rel, "preamble", 1, end, "html preamble", "", body))
    for j, h in enumerate(headings):
        start = h["line"]
        end = headings[j + 1]["line"] - 1 if j + 1 < len(headings) else nlines
        body = "\n".join(lines[start - 1:end])
        qual = h["text"][:60] or f"{h['tag']}#{start}"
        if any(c.qualname == qual for c in out):
            qual = f"{qual}#{start}"
        out.append(_mk(rel, qual, start, end,
                       f"<{h['tag']}>", h["text"][:120], body))
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
    if ext in (".html", ".htm", ".xhtml"):
        return _html(rel, text)
    return _blocks(rel, text.splitlines(), ext.lstrip(".") or "text")