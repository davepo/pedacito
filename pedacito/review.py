"""The review page: side-by-side diffs you sign off on before anything is written.

    diffs = review.diffs_from_results(edit_results)   # from apply_edits(dry_run=True)
    nonce = review.new_nonce()
    page = review.build_page(diffs, task, model, profile, nonce, interactive=True, stamp=when)
    outcome, detail = review.run(page, nonce, writer_fn, open_browser=True)

`build_page` renders a self-contained HTML file (no network, no webfonts).
`run` serves it from a single-purpose server bound to 127.0.0.1 so the Apply and
Cancel buttons work; requests with a foreign Host/Origin are refused.

Each FileDiff carries the SHA-256 of the file as previewed; applying re-hashes
and refuses on a mismatch, so a file edited while the review sat open is never
overwritten with a stale preview.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import secrets
import threading
import webbrowser
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from http.server import BaseHTTPRequestHandler
from itertools import accumulate
from pathlib import Path

from . import fileio
from .index import state_dir
from .local_server import LocalHTTPServer

REVIEW_DIR = "reviews"
CONTEXT = 4          # unchanged lines kept either side of a change
COLLAPSE_OVER = 10   # runs of unchanged lines longer than this get folded
TOKEN_RE = re.compile(r"\w+|\s+|.")


@dataclass
class Row:
    """One rendered row of the side-by-side diff table."""

    kind: str                 # equal | replace | delete | insert | skip
    a_no: int | None = None
    a: str = ""
    b_no: int | None = None
    b: str = ""
    a_spans: list = field(default_factory=list)   # (start, end) char ranges
    b_spans: list = field(default_factory=list)
    hidden: int = 0           # for skip rows: how many lines are folded
    fold: str = ""            # ties folded rows to their "show N" button


@dataclass
class FileDiff:
    """A rendered diff for one file, ready to embed in the review page."""

    rel: str
    rows: list[Row]
    added: int
    removed: int
    sha: str            # file contents when the preview was built
    after_text: str     # what gets written on sign-off
    message: str
    # For a creation detected as a rename: the old path(s) moved into this file.
    renames_from: tuple[str, ...] = ()
    # Display-only row for a rename's source ("removed (renamed to X)"); never
    # written on sign-off, the rename-removal pass deletes the source instead.
    removed_source: bool = False


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _word_spans(a: str, b: str) -> tuple[list, list]:
    """Character ranges that differ between two versions of one line."""
    at, bt = TOKEN_RE.findall(a), TOKEN_RE.findall(b)
    sm = SequenceMatcher(None, at, bt, autojunk=False)
    a_spans, b_spans = [], []
    offs_a = [0, *accumulate(len(t) for t in at)]
    offs_b = [0, *accumulate(len(t) for t in bt)]
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete") and i2 > i1:
            a_spans.append((offs_a[i1], offs_a[i2]))
        if tag in ("replace", "insert") and j2 > j1:
            b_spans.append((offs_b[j1], offs_b[j2]))
    return a_spans, b_spans


def build_rows(before: str, after: str) -> tuple[list[Row], int, int]:
    """Row-by-row diff of two texts, with word-level spans on replaced lines
    and long unchanged runs folded. Returns (rows, lines_added, lines_removed)."""
    a = before.split("\n")
    b = after.split("\n")
    sm = SequenceMatcher(None, a, b, autojunk=False)
    rows: list[Row] = []
    folds = [0]
    added = removed = 0

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            n = i2 - i1

            def eq(k, fold=""):
                """Build one equal-line Row at offset k within this run."""
                return Row("equal", i1 + k + 1, a[i1 + k], j1 + k + 1, b[j1 + k],
                           fold=fold)

            # A run at the top or bottom of the file only needs context on the
            # side facing a change.
            head = 0 if not rows else CONTEXT
            tail = 0 if i2 >= len(a) else CONTEXT
            if n > COLLAPSE_OVER + head + tail:
                folds[0] += 1
                fid = f"f{folds[0]}"
                for k in range(head):
                    rows.append(eq(k))
                for k in range(head, n - tail):
                    rows.append(eq(k, fold=fid))          # rendered hidden
                rows.append(Row("skip", hidden=n - head - tail, fold=fid))
                for k in range(n - tail, n):
                    rows.append(eq(k))
            else:
                for k in range(n):
                    rows.append(eq(k))
        elif tag == "replace":
            pairs = max(i2 - i1, j2 - j1)
            for k in range(pairs):
                la = a[i1 + k] if i1 + k < i2 else None
                lb = b[j1 + k] if j1 + k < j2 else None
                if la is not None and lb is not None:
                    sa, sb = _word_spans(la, lb)
                    rows.append(Row("replace", i1 + k + 1, la, j1 + k + 1, lb, sa, sb))
                    added += 1
                    removed += 1
                elif la is not None:
                    rows.append(Row("delete", i1 + k + 1, la))
                    removed += 1
                else:
                    rows.append(Row("insert", None, "", j1 + k + 1, lb))
                    added += 1
        elif tag == "delete":
            for k in range(i1, i2):
                rows.append(Row("delete", k + 1, a[k]))
                removed += 1
        elif tag == "insert":
            for k in range(j1, j2):
                rows.append(Row("insert", None, "", k + 1, b[k]))
                added += 1

    return rows, added, removed


def diffs_from_results(results, root=None) -> list[FileDiff]:
    """Convert edits.Result objects (from apply_edits(dry_run=True)) into
    FileDiffs. Only successful results with computed before/after text count.

    `root` is only consulted for renames, where each source file's real
    contents are shown as a deletion."""
    out: list[FileDiff] = []
    for r in results:
        if not r.ok or not r.sha:
            continue
        rows, added, removed = build_rows(r.before, r.after)
        out.append(FileDiff(r.file, rows, added, removed, r.sha, r.after,
                            r.message, renames_from=tuple(r.renames_from)))
        if root is None:
            continue
        for old in r.renames_from:
            try:
                otext, _ = fileio.read(Path(root) / old, errors="replace")
            except OSError:
                continue
            orows, oa, om = build_rows(otext, "")
            out.append(FileDiff(old, orows, oa, om, _sha(otext), "",
                                "removed (renamed to " + r.file + ")",
                                removed_source=True))
    return out


def diffs_from_backup(root: Path, backup_dir: Path) -> list[FileDiff]:
    """Reconstruct diffs after files were already written, by pairing each file
    in a backup snapshot with its current content (used by `new --review`).
    The sha pins the post-write bytes. Files gone from the tree are skipped."""
    out: list[FileDiff] = []
    root = Path(root).resolve()
    if backup_dir is None or not Path(backup_dir).is_dir():
        return out
    for src in sorted(Path(backup_dir).rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(backup_dir).as_posix()
        cur = root / rel
        if not cur.is_file():
            continue
        before, _ = fileio.read(src, errors="replace")
        after, _ = fileio.read(cur, errors="replace")
        if before == after:
            continue
        rows, added, removed = build_rows(before, after)
        out.append(FileDiff(rel, rows, added, removed, _sha(after), after,
                            f"built ({added}+ / {removed}-)"))
    return out


def _mark(text: str, spans: list) -> str:
    """HTML-escape a line, wrapping the given character ranges in <mark>."""
    if not spans:
        return html.escape(text) or "&nbsp;"
    out, pos = [], 0
    for start, end in spans:
        out.append(html.escape(text[pos:start]))
        out.append('<mark>' + html.escape(text[start:end]) + '</mark>')
        pos = end
    out.append(html.escape(text[pos:]))
    return "".join(out) or "&nbsp;"


CSS = """
:root {
  --paper:  #e6ece4;   /* ledger stock */
  --card:   #f5f8f3;
  --band:   #eaf0e8;   /* ruled column */
  --rule:   #b9c8ba;   /* printed rules */
  --hair:   #dde6db;
  --ink:    #1d2a23;
  --ink-2:  #566259;
  --cut:    #8c3a32;   /* struck out */
  --cut-bg: #f6e6e3;
  --cut-mk: #e9c9c3;
  --set:    #1f5140;   /* newly set */
  --set-bg: #e3eee7;
  --set-mk: #bfdcc9;
  --stamp:  #9d3327;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: var(--paper); color: var(--ink);
  font: 15px/1.55 "Segoe UI", system-ui, -apple-system, sans-serif;
}
code, .ln, .tx { font-family: "Cascadia Mono", Consolas, "SF Mono", Menlo, monospace; }

header {
  padding: 22px 26px 15px; background: var(--card);
  border-bottom: 3px double var(--rule);   /* ledger head rule */
}
h1 {
  margin: 0; font-family: "Palatino Linotype", Palatino, "Iowan Old Style", Georgia, serif;
  font-size: 25px; font-weight: 600; letter-spacing: -0.01em;
}
h1 .sub { color: var(--ink-2); font-weight: 400; font-style: italic; }
.task { margin: 9px 0 0; max-width: 78ch; }
.meta {
  margin-top: 11px; display: flex; flex-wrap: wrap; align-items: baseline;
  gap: 8px 15px; font-size: 12.5px; color: var(--ink-2);
}
.meta .lbl { text-transform: uppercase; letter-spacing: 0.08em; }
.meta .who {
  font-family: "Cascadia Mono", Consolas, monospace; font-size: 12px;
  background: var(--band); border: 1px solid var(--hair);
  padding: 1px 7px; border-radius: 2px; color: var(--ink);
}
main { padding: 22px 26px 150px; }

.file { margin-bottom: 30px; border: 1px solid var(--rule); background: var(--card); }
.file > h2 {
  margin: 0; padding: 11px 14px; border-bottom: 1px solid var(--rule);
  font-size: 14px; font-weight: 600; display: flex; align-items: center; gap: 12px;
  font-family: "Cascadia Mono", Consolas, monospace;
  position: sticky; top: 0; z-index: 3; background: var(--card);
}
.file > h2::after {
  content: ""; position: absolute; left: 0; right: 0; bottom: -1px; height: 1px;
}
thead th {
  position: sticky; top: 41px; z-index: 2; background: var(--band);
  font: 500 11px/1 "Segoe UI", system-ui, sans-serif; color: var(--ink-2);
  text-transform: uppercase; letter-spacing: 0.09em; text-align: left;
  padding: 6px 10px; border-bottom: 1px solid var(--rule);
}
thead th.right { border-left: 2px solid var(--rule); }
.tally { margin-left: auto; font-size: 12.5px; letter-spacing: 0.02em; }
.tally .p { color: var(--set); }
.tally .m { color: var(--cut); }

table { width: 100%; border-collapse: collapse; table-layout: fixed; }
col.n { width: 3.6em; }
td { padding: 0; vertical-align: top; border-bottom: 1px solid var(--hair); }
td.ln {
  text-align: right; padding: 1px 9px 1px 6px; color: var(--ink-2);
  font-size: 12px; user-select: none; background: var(--band);
  border-right: 1px solid var(--hair);
}
td.tx {
  padding: 1px 10px; font-size: 13px; white-space: pre-wrap;
  overflow-wrap: anywhere; tab-size: 4;
}
td.gutter { border-left: 2px solid var(--rule); }

tr.replace .a, tr.delete .a { background: var(--cut-bg); }
tr.replace .b, tr.insert .b { background: var(--set-bg); }
tr.delete  .b, tr.insert .a { background: repeating-linear-gradient(
    -45deg, transparent, transparent 5px, #e4eae2 5px, #e4eae2 6px); }
mark { background: var(--cut-mk); color: inherit; padding: 0 1px; border-radius: 2px; }
tr.replace .b mark, tr.insert .b mark { background: var(--set-mk); }

tr.skip td {
  background: var(--band); color: var(--ink-2); font-size: 12px;
  padding: 3px 12px; text-align: center; letter-spacing: 0.05em;
}
tr.skip button {
  background: none; border: 0; color: var(--ink-2); cursor: pointer;
  font: inherit; text-decoration: underline dotted; padding: 2px 6px;
}

footer {
  position: fixed; left: 0; right: 0; bottom: 0; background: var(--card);
  border-top: 2px solid var(--rule); padding: 15px 26px;
  box-shadow: 0 -8px 22px rgba(29,42,35,0.07);
}
.signoff { display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
.attest { display: flex; align-items: baseline; gap: 9px; cursor: pointer; max-width: 62ch; }
.attest input { width: 17px; height: 17px; accent-color: var(--set); flex: none; }
.attest span { font-size: 13.5px; }
.rule-line {
  flex: 1 1 120px; border-bottom: 1px solid var(--rule); min-width: 40px;
  align-self: flex-end; margin-bottom: 6px;
}
.acts { display: flex; gap: 10px; }
button.act {
  font: 600 14px/1 "Segoe UI", system-ui, sans-serif; padding: 11px 18px;
  border: 1px solid var(--rule); background: #fff; color: var(--ink);
  cursor: pointer; border-radius: 2px;
}
button.act:hover { border-color: var(--ink-2); }
button.write {
  border-color: var(--set); color: #fff; background: var(--set);
  box-shadow: inset 0 -2px 0 rgba(0,0,0,0.18);
}
button.write:hover { background: #1a4535; }
button.write.confirm { background: var(--stamp); border-color: var(--stamp); }
button.write.confirm:hover { background: #85291f; }
button.act[disabled] { opacity: 0.4; cursor: not-allowed; box-shadow: none; }
button.act:focus-visible, .attest input:focus-visible,
tr.skip button:focus-visible { outline: 2px solid var(--stamp); outline-offset: 2px; }
.note { margin-top: 9px; font-size: 12.5px; color: var(--ink-2); }

/* Sign-off stamp shown on apply. */
.stamp {
  position: fixed; inset: 0; display: none; place-items: center;
  background: rgba(238,241,236,0.86); z-index: 20;
}
.stamp.on { display: grid; }
.stamp .mark {
  border: 3px solid var(--stamp); color: var(--stamp); padding: 16px 30px;
  font: 700 27px/1.15 "Palatino Linotype", Palatino, Georgia, serif;
  letter-spacing: 0.13em; text-transform: uppercase; text-align: center;
  transform: rotate(-6deg); background: rgba(247,249,245,0.94);
  box-shadow: 0 0 0 2px rgba(157,51,39,0.18);
  animation: press 260ms cubic-bezier(.2,1.5,.4,1);
}
.stamp .mark small {
  display: block; font-size: 12.5px; letter-spacing: 0.08em;
  font-weight: 400; margin-top: 9px; text-transform: none; font-style: italic;
}
@keyframes press {
  from { transform: rotate(-6deg) scale(2.1); opacity: 0; }
  to   { transform: rotate(-6deg) scale(1);   opacity: 1; }
}
@media (prefers-reduced-motion: reduce) { .stamp .mark { animation: none; } }
@media (max-width: 820px) {
  main { padding: 16px 12px 210px; }
  header { padding: 16px 14px 12px; }
  footer { padding: 13px 14px; }
  td.tx { font-size: 12px; }
  col.n { width: 3em; }
}
"""

# Fold toggles, two-step sign-off (tick, click to arm, click again within 6s),
# Apply/Cancel POSTs, and the sign-off stamp.
JS = """
document.querySelectorAll('tr.skip button').forEach(function (b) {
  b.addEventListener('click', function () {
    document.querySelectorAll('[data-fold="' + b.dataset.fold + '"]')
      .forEach(function (r) { r.hidden = false; });
    b.closest('tr').hidden = true;
  });
});

var box = document.getElementById('attest');
var write = document.getElementById('write');
var cancel = document.getElementById('cancel');
var note = document.getElementById('note');
var armed = false;
var LABEL = write ? write.textContent : '';

function disarm() {
  armed = false;
  if (!write) return;
  write.textContent = LABEL;
  write.classList.remove('confirm');
}

if (box) {
  box.addEventListener('change', function () {
    write.disabled = !box.checked;
    if (!box.checked) disarm();
  });
}

function finish(title, detail) {
  document.getElementById('stamp-title').textContent = title;
  document.getElementById('stamp-detail').textContent = detail;
  document.getElementById('stamp').classList.add('on');
}

function post(path, body) {
  return fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {})
  }).then(function (r) { return r.json(); });
}

if (write) {
  write.addEventListener('click', function () {
    if (!armed) {
      // Second click required. An accidental Apply should not be one keystroke.
      armed = true;
      write.textContent = 'Confirm \\u2014 write ' + write.dataset.count;
      write.classList.add('confirm');
      setTimeout(function () { if (armed) disarm(); }, 6000);
      return;
    }
    write.disabled = true;
    cancel.disabled = true;
    post('/apply', { nonce: window.NONCE }).then(function (res) {
      if (res.ok) {
        finish('Written', res.detail);
      } else {
        write.disabled = false;
        cancel.disabled = false;
        disarm();
        note.textContent = res.detail;
        note.style.color = 'var(--stamp)';
      }
    }).catch(function () {
      write.disabled = false; cancel.disabled = false; disarm();
      note.textContent = 'Could not reach Pedacito. Did the terminal session end?';
      note.style.color = 'var(--stamp)';
    });
  });
}

if (cancel) {
  cancel.addEventListener('click', function () {
    post('/cancel', {}).then(function () {
      finish('Discarded', 'No files were changed.');
    }).catch(function () { finish('Discarded', 'No files were changed.'); });
  });
}

// Enter must never trigger a write.
document.addEventListener('keydown', function (e) {
  if (e.key === 'Enter' && document.activeElement === write) { e.preventDefault(); }
});
"""


def _rows_html(fd: FileDiff, fid: int) -> str:
    """Render one FileDiff's rows as HTML <tr> elements (folded rows hidden)."""
    out = []
    for row in fd.rows:
        if row.kind == "skip":
            out.append(
                f'<tr class="skip" data-btn="{fid}-{row.fold}"><td colspan="4">'
                f'<button data-fold="{fid}-{row.fold}">show {row.hidden} unchanged '
                f'line{"s" if row.hidden != 1 else ""}</button></td></tr>')
            continue
        a_no = "" if row.a_no is None else row.a_no
        b_no = "" if row.b_no is None else row.b_no
        a_txt = _mark(row.a, row.a_spans) if row.kind == "replace" else (
            html.escape(row.a) or "&nbsp;")
        b_txt = _mark(row.b, row.b_spans) if row.kind == "replace" else (
            html.escape(row.b) or "&nbsp;")
        if row.kind == "insert":
            a_no, a_txt = "", "&nbsp;"
        if row.kind == "delete":
            b_no, b_txt = "", "&nbsp;"
        hid = f' hidden data-fold="{fid}-{row.fold}"' if row.fold else ""
        out.append(
            f'<tr class="{row.kind}"{hid}>'
            f'<td class="ln">{a_no}</td><td class="tx a">{a_txt}</td>'
            f'<td class="ln gutter">{b_no}</td><td class="tx b">{b_txt}</td></tr>')
    return "\n".join(out)


def build_page(diffs: list[FileDiff], task: str, model: str, profile: str,
               nonce: str, interactive: bool, stamp: str) -> str:
    """Assemble the self-contained review page. `interactive` adds the sign-off
    checkbox and Apply/Discard buttons wired to `nonce`; otherwise the page is a
    read-only preview."""
    files_html = []
    for i, fd in enumerate(diffs):
        files_html.append(f"""
<section class="file">
  <h2>{html.escape(fd.rel)}
    <span class="tally"><span class="p">+{fd.added}</span>
      &nbsp;<span class="m">&minus;{fd.removed}</span></span>
  </h2>
  <table>
    <colgroup><col class="n"><col><col class="n"><col></colgroup>
    <thead><tr>
      <th colspan="2">Now on disk</th>
      <th colspan="2" class="right">Proposed</th>
    </tr></thead>
    <tbody>
{_rows_html(fd, i)}
    </tbody>
  </table>
</section>""")

    total_files = len(diffs)
    total_add = sum(d.added for d in diffs)
    total_rem = sum(d.removed for d in diffs)
    noun = "file" if total_files == 1 else "files"

    if interactive:
        footer = f"""
<footer>
  <div class="signoff">
    <label class="attest">
      <input type="checkbox" id="attest">
      <span>I have read these changes and want them written to disk.</span>
    </label>
    <span class="rule-line"></span>
    <div class="acts">
      <button class="act" id="cancel" type="button">Discard</button>
      <button class="act write" id="write" type="button" disabled
              data-count="{total_files} {noun}">Sign off &amp; write</button>
    </div>
  </div>
  <p class="note" id="note">Originals are copied to
    <code>.pedacito/backups/{html.escape(stamp)}/</code> first, so
    <code>pedacito restore</code> undoes this.</p>
</footer>"""
    else:
        footer = """
<footer>
  <div class="signoff">
    <span>Read-only preview. Nothing here can write to disk.</span>
    <span class="rule-line"></span>
  </div>
  <p class="note">Re-run with <code>--review</code> to get an apply button, or
    <code>--apply</code> to write from the terminal.</p>
</footer>"""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review &mdash; {html.escape(task[:60])}</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>Proposed changes <span class="sub">awaiting sign-off</span></h1>
  <p class="task">{html.escape(task)}</p>
  <div class="meta">
    <span class="lbl">{total_files} {noun} &middot; +{total_add} &minus;{total_rem} lines</span>
    {f'<span class="who">{html.escape(profile)}</span>' if profile else ""}
    <span class="who">{html.escape(model)}</span>
    <span class="lbl">{html.escape(stamp)}</span>
  </div>
</header>
<main>
{"".join(files_html)}
</main>
{footer}
<div class="stamp" id="stamp">
  <div class="mark">
    <span id="stamp-title">Written</span>
    <small id="stamp-detail"></small>
  </div>
</div>
<script>window.NONCE = {json.dumps(nonce)};</script>
<script>{JS}</script>
</body>
</html>"""


def request_allowed(handler, post: bool) -> bool:
    """Guard for the loopback servers against DNS rebinding and cross-site
    requests: Host must be this server, Origin (if sent) must match, and POSTs
    must be JSON (a cross-site page cannot send that without a CORS preflight)."""
    port = handler.server.server_address[1]
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    if (handler.headers.get("Host") or "").lower() not in hosts:
        return False
    if not post:
        return True
    origin = handler.headers.get("Origin")
    if origin and origin.lower() not in {f"http://{h}" for h in hosts}:
        return False
    ctype = (handler.headers.get("Content-Type") or "").split(";")[0]
    return ctype.strip().lower() == "application/json"


def read_json_body(handler) -> dict:
    """Parse the request body as a JSON object; {} when absent or malformed."""
    try:
        n = max(0, int(handler.headers.get("Content-Length") or 0))
        body = json.loads(handler.rfile.read(n) or b"{}")
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


class _Handler(BaseHTTPRequestHandler):
    """Serves the review page on GET and handles /apply and /cancel POSTs."""

    server_version = "Pedacito"

    def log_message(self, *a):
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not request_allowed(self, post=False):
            self._send(403, b"forbidden", "text/plain")
            return
        if self.path.split("?")[0] != "/":
            self._send(404, b"not found", "text/plain")
            return
        self._send(200, self.server.page.encode("utf-8"), "text/html; charset=utf-8")

    def do_POST(self):
        if not request_allowed(self, post=True):
            self._send(403, b'{"ok":false}', "application/json")
            return
        path = self.path.split("?")[0]
        body = read_json_body(self)

        if path == "/cancel":
            self.server.outcome = ("cancelled", "No files were changed.")
            if self.server.on_discard is not None:
                try:
                    self.server.on_discard()
                except Exception as e:          # never hang the review loop
                    self.server.outcome = ("cancelled", f"Discard failed: {e}")
            self._send(200, b'{"ok":true}', "application/json")
            self.server.done.set()
            return

        if path != "/apply":
            self._send(404, b'{"ok":false}', "application/json")
            return

        if body.get("nonce") != self.server.nonce:
            self._send(403, json.dumps({
                "ok": False,
                "detail": "This review is no longer valid. Re-run the task."
            }).encode(), "application/json")
            return

        ok, detail = self.server.writer()
        self._send(200, json.dumps({"ok": ok, "detail": detail}).encode(),
                   "application/json")
        if ok:
            self.server.outcome = ("applied", detail)
            self.server.done.set()


def write_files(root: Path, diffs: list[FileDiff], backup_dir: Path | None):
    """Write the reviewed text for every FileDiff to disk. Returns (ok, message).

    Pass one refuses the whole batch if any existing file changed (or vanished)
    since its sha was recorded; a creation diff pins the sha of "" so its
    absence is expected. A creation whose path now exists with different bytes
    is refused too. Pass two writes, backing originals up to `backup_dir`
    (never overwriting an existing backup), then completes renames by removing
    their sources."""
    EMPTY_SHA = _sha("")

    def _digest(p: Path) -> str:
        # Hash the same LF-normalised text the preview sha was recorded from.
        # Hashing raw bytes here would make every CRLF file on Windows look
        # "changed on disk" even though its text is identical.  A missing file
        # is still reported as absent, not as an empty read.
        try:
            text, _ = fileio.read(p, errors="replace")
        except FileNotFoundError:
            return EMPTY_SHA
        return _sha(text)

    if not diffs:
        return False, ("Not written: this review contains no applicable "
                       "changes. Re-run the task to rebuild the review.")
    stale = []
    for fd in diffs:
        p = root / fd.rel
        if not p.exists():
            if fd.sha and fd.sha != EMPTY_SHA:
                stale.append(f"{fd.rel} (deleted)")
            continue
        # Bytes already equal to what we would write: applied, not stale.
        if _digest(p) not in (fd.sha, _sha(fd.after_text)):
            stale.append(f"{fd.rel} (changed on disk)")
    if stale:
        return False, ("Not written: " + ", ".join(stale) +
                       ". Re-run the task to rebuild the review.")

    for fd in diffs:
        p = root / fd.rel
        if not fd.sha and p.exists() and _digest(p) != _sha(fd.after_text):
            return False, (f"Not written: {fd.rel} was created by something "
                           "else while this review was open. Re-run the task "
                           "to rebuild the review.")

    default_style = fileio.TextStyle()
    existed_at_check = {fd.rel: (root / fd.rel).exists() for fd in diffs}
    written = []
    for fd in diffs:
        if fd.removed_source:
            continue
        p = root / fd.rel
        try:
            if p.exists():
                original, style = fileio.read(p, errors="replace")
            elif existed_at_check[fd.rel]:
                raise FileNotFoundError
            else:
                original, style = "", default_style     # new file
        except FileNotFoundError:
            return False, (f"Partially written: {fd.rel} vanished between "
                           "preview check and write; files already written "
                           "in this batch: " + (", ".join(written) or "none") +
                           ". Re-run the task to rebuild the review.")
        if backup_dir is not None and original:
            dest = backup_dir / fd.rel
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                fileio.write(dest, original, style)
        p.parent.mkdir(parents=True, exist_ok=True)
        fileio.write(p, fd.after_text, style)
        written.append(fd.rel)

    removed = []
    rename_notes = []
    root_resolved = root.resolve()

    def _remove_rename_source(old: str, expect_sha: str) -> None:
        op = root / old
        if not op.is_file() or not op.resolve().is_relative_to(root_resolved):
            return
        try:
            cur = _digest(op)
        except OSError:
            return
        if expect_sha and cur != expect_sha:
            rename_notes.append(f"{old} changed since preview; left in place")
            return
        if backup_dir is not None:
            dest = backup_dir / old
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    otext, ostyle = fileio.read(op, errors="replace")
                    fileio.write(dest, otext, ostyle)
                except OSError:
                    pass
        try:
            op.unlink()
            removed.append(old)
        except OSError:
            rename_notes.append(f"{old} could not be removed")

    for fd in diffs:
        for old in fd.renames_from:
            # The matching removal row pins the source's pre-rename bytes.
            pin = next((d.sha for d in diffs
                        if d.rel == old and d.removed_source), "")
            _remove_rename_source(old.strip(), pin)

    msg = f"{len(written)} file(s) written: " + ", ".join(written)
    if removed:
        msg += "; renamed away: " + ", ".join(removed)
    if rename_notes:
        msg += "; NOTE: " + ", ".join(rename_notes)
    return True, msg


def save(root: Path, page: str, stamp: str) -> Path:
    """Save the page to .pedacito/reviews/<stamp>.html and return its path."""
    d = state_dir(root, REVIEW_DIR)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{stamp}.html"
    p.write_text(page, encoding="utf-8")
    return p


def run(page: str, nonce: str, writer, port: int = 0, open_browser: bool = True,
        timeout: float = 1800.0, on_url=None, on_discard=None):
    """Serve the page on 127.0.0.1 and block until Apply/Discard or `timeout`
    seconds. `writer()` -> (ok, detail) is called on Apply; `on_discard()` on
    Discard, for flows whose files are already on disk (`new --review`).
    Returns (outcome, detail); outcome is "applied", "cancelled" or "timeout"."""
    srv = LocalHTTPServer(("127.0.0.1", port), _Handler)
    srv.page = page
    srv.nonce = nonce
    srv.writer = writer
    srv.on_discard = on_discard
    srv.done = threading.Event()
    srv.outcome = ("timeout", "Review window timed out; nothing was written.")
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    if on_url:
        on_url(url)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.done.wait(timeout)
    except KeyboardInterrupt:
        srv.outcome = ("cancelled", "Interrupted; nothing was written.")
    srv.shutdown()
    srv.server_close()
    return srv.outcome


def new_nonce() -> str:
    """A fresh token authorising one review's Apply request."""
    return secrets.token_urlsafe(16)
