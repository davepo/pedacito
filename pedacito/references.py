"""Structural chunking for non-Python reference files (config, data, docs, HTML).

    chunks = chunk_reference("data/config.csv", Path("data/config.csv"))

These files are not parsed as code. Each gets one or a few chunks that show its
shape (columns, keys, sections, headings) with real line ranges, so the model
can decide whether to `read` or `grep` it. Strategy by extension: CSV/TSV
column summary; JSON top-level shape; INI/TOML sections; Markdown/RST
headings; HTML landmark elements; YAML top-level keys; SQL statements; shell
functions; anything else fixed-size line blocks.
"""

from __future__ import annotations

import csv
import io
import json
import re
from html.parser import HTMLParser
from pathlib import Path

from . import fileio
from .chunker import Chunk, _sha, split_lines

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
INI_SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]")
BLOCK_LINES = 150   # fallback chunk size, in lines, for unstructured text

# HTML landmark elements. Only top-level ones become chunks; nested landmarks
# belong to their containing section.
HTML_LANDMARKS = {
    "header", "nav", "main", "section", "article", "aside",
    "footer", "form", "figure", "details", "dialog", "template",
}
HTML_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
# Elements whose text is code, not prose (skipped when collecting headings).
HTML_RAW = {"script", "style"}


def _mk(rel: str, qualname: str, start: int, end: int, sig: str,
        docline: str = "", src: str = "") -> Chunk:
    """Build one reference Chunk with a single-line docline (multi-line
    docstrings would break the one-line-per-entry layout of the file map)."""
    docline = " ".join(docline.split())
    return Chunk(file=rel, qualname=qualname, kind="reference", start=start, end=end,
                 signature=sig, docline=docline, source=src, sha=_sha(src or sig + docline))


def _tabular(rel: str, text: str, delim: str) -> list[Chunk]:
    """CSV/TSV: one chunk summarising column names, row count, and two sample
    rows, so the model can see the shape without reading the whole table."""
    lines = split_lines(text)
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
    lines = split_lines(text)
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
    lines = split_lines(text)
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
    lines = split_lines(text)
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


class _HtmlOutline(HTMLParser):
    """Streaming HTML walker recording top-level landmark elements (line
    range, id, class), every heading, external script/stylesheet
    dependencies, and the title. No other text content is kept."""

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

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        line = self.getpos()[0]
        if tag in HTML_RAW:
            self._in_raw += 1
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
        # only pop on a match, so a stray </section> can't drop an <article>
        if self._stack and self._stack[-1]["tag"] == tag:
            node = self._stack.pop()
            node["end"] = self.getpos()[0]
            self.sections.append(node)

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

    def close(self):
        super().close()
        # close landmarks left open at EOF at the last known line
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
    """HTML: one chunk per top-level landmark element plus a document header
    chunk (title, dependencies). Falls back to heading-based splitting, then
    to fixed-size line blocks when there are no landmarks/headings.
    """
    lines = split_lines(text)
    nlines = len(lines)
    if nlines < 3:
        return _blocks(rel, lines, "html")

    parser = _HtmlOutline()
    try:
        parser.feed(text)
        parser.close()
    except Exception:
        # html.parser can raise on pathological input
        return _blocks(rel, lines, "html")

    if not parser.sections and parser.headings:
        return _html_heading_split(rel, text, parser.headings)
    if not parser.sections:
        return _blocks(rel, lines, "html")

    chunks: list[Chunk] = []
    headings = sorted(parser.headings, key=lambda h: h["line"])

    # document header: title, deps, everything above the first landmark
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

    # landmark sections
    for sec in parser.sections:
        start = sec["line"]
        end = max(start, sec.get("end", start))
        # label: first heading inside the section
        label = ""
        for h in headings:
            if start <= h["line"] <= end:
                label = f"{h['tag'].upper()}: {h['text'][:80]}"
                break
        # signature: tag with its id (preferred) or first classes
        sig = f"<{sec['tag']}"
        if sec["id"]:
            sig += f' id="{sec["id"]}"'
        elif sec["class"]:
            classes = sec["class"].split()[:2]
            sig += f' class="{",".join(classes)}"'
        sig += ">"
        qual = sec["id"] or sec["tag"]
        # chunk keys must be unique per file
        if any(c.qualname == qual for c in chunks):
            qual = f"{qual}#{start}"
        body = "\n".join(lines[start - 1:end])
        chunks.append(_mk(rel, qual, start, end, sig, label, body))

    # trailing content after the last landmark
    last_end = max(s.get("end", s["line"]) for s in parser.sections)
    if last_end < nlines:
        body = "\n".join(lines[last_end:])
        if body.strip():
            chunks.append(_mk(rel, "trailing", last_end + 1, nlines,
                              "html tail", "", body))
    return chunks


def _html_heading_split(rel: str, text: str, headings: list[dict]) -> list[Chunk]:
    """Fallback for landmark-less HTML: split on headings, like markdown."""
    lines = split_lines(text)
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


_YAML_KEY_RE = re.compile(
    r"^([A-Za-z_][\w.-]*|\"[^\"]+\"|'[^']+')\s*:")
_SHELL_FUNC_RE = re.compile(
    r"^\s*(?:function\s+([A-Za-z_][\w-]*)|"
    r"([A-Za-z_][\w-]*)\s*\(\s*\))\s*\{?\s*$")
_SHELL_PS_FUNC_RE = re.compile(
    r"^\s*function\s+([A-Za-z_][\w-]*)\s*\{?\s*$", re.IGNORECASE)
_SQL_OBJECT_RE = re.compile(
    r"\bCREATE\s+(?:OR\s+REPLACE\s+)?"
    r"(TABLE|VIEW|INDEX|FUNCTION|PROCEDURE|TRIGGER|SCHEMA)\s+"
    r"([A-Za-z_][\w.\"]*)", re.IGNORECASE)


def _yaml(rel: str, text: str) -> list[Chunk]:
    """Top-level YAML keys as sections (no YAML parser dependency)."""
    lines = split_lines(text)
    nlines = len(lines)
    keys: list[tuple[int, str]] = []
    for i, line in enumerate(lines, 1):
        if not line or line[0].isspace() or line.lstrip().startswith("#"):
            continue
        m = _YAML_KEY_RE.match(line)
        if m:
            keys.append((i, m.group(1).strip("\"'")))
    if not keys:
        return _blocks(rel, lines, "yaml")
    out: list[Chunk] = []
    if keys[0][0] > 1:
        head = "\n".join(lines[:keys[0][0] - 1])
        if head.strip():
            out.append(_mk(rel, "document", 1, keys[0][0] - 1,
                           "yaml document header", "", head))
    for idx, (start, key) in enumerate(keys):
        end = (keys[idx + 1][0] - 1) if idx + 1 < len(keys) else nlines
        body = "\n".join(lines[start - 1:end])
        out.append(_mk(rel, key, start, end, f"{key}:", "", body))
    return out


def _sql(rel: str, text: str) -> list[Chunk]:
    """SQL statements split on a trailing semicolon (quotes respected enough
    for the common schema/query files Pedacito indexes)."""
    lines = split_lines(text)
    nlines = len(lines)
    spans: list[tuple[int, int]] = []
    start = 1
    in_single = in_double = False
    for i, line in enumerate(lines, 1):
        j = 0
        while j < len(line):
            ch = line[j]
            if ch == "'" and not in_double:
                if in_single and j + 1 < len(line) and line[j + 1] == "'":
                    j += 1
                else:
                    in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            elif ch == ";" and not in_single and not in_double:
                if line[j + 1:].strip() == "":
                    spans.append((start, i))
                    start = i + 1
                break
            j += 1
    if start <= nlines:
        tail = "\n".join(lines[start - 1:]).strip()
        if tail:
            spans.append((start, nlines))
    if not spans:
        return _blocks(rel, lines, "sql")
    out: list[Chunk] = []
    for start, end in spans:
        body = "\n".join(lines[start - 1:end])
        first = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
        m = _SQL_OBJECT_RE.search(first)
        qual = (f"create {m.group(1).lower()} {m.group(2)}"
                if m else first[:60])
        sig = first[:160]
        out.append(_mk(rel, qual, start, end, sig, "", body))
    return out


def _shell(rel: str, text: str) -> list[Chunk]:
    """Shell/PowerShell function definitions, plus header/trailing blocks."""
    lines = split_lines(text)
    nlines = len(lines)
    funcs: list[tuple[int, str]] = []
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith("#"):
            continue
        m = _SHELL_PS_FUNC_RE.match(line)
        if m:
            funcs.append((i, m.group(1)))
            continue
        m = _SHELL_FUNC_RE.match(line)
        if m:
            funcs.append((i, m.group(1) or m.group(2)))
    if not funcs:
        return _blocks(rel, lines, "shell")
    out: list[Chunk] = []
    if funcs[0][0] > 1:
        head = "\n".join(lines[:funcs[0][0] - 1])
        if head.strip():
            out.append(_mk(rel, "script", 1, funcs[0][0] - 1,
                           "shell script header", "", head))
    for idx, (start, name) in enumerate(funcs):
        end = (funcs[idx + 1][0] - 1) if idx + 1 < len(funcs) else nlines
        body = "\n".join(lines[start - 1:end])
        out.append(_mk(rel, name, start, end, f"{name}()", "", body))
    return out


def chunk_reference(rel: str, path: Path, max_bytes: int = 2_000_000) -> list[Chunk]:
    """Produce structural chunks for one non-Python file, dispatching on its
    extension. Files larger than `max_bytes` are never read in full: they get
    a single header chunk noting the size.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size > max_bytes:
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
    if ext in (".yaml", ".yml"):
        return _yaml(rel, text)
    if ext == ".sql":
        return _sql(rel, text)
    if ext in (".sh", ".bash", ".zsh", ".ksh", ".ps1"):
        return _shell(rel, text)
    return _blocks(rel, split_lines(text), ext.lstrip(".") or "text")