"""The review page: side-by-side diffs you sign off on before anything is written.

Usage
-----
    from . import review

    diffs = review.diffs_from_results(edit_results)   # from apply_edits(dry_run=True)
    nonce = review.new_nonce()
    page = review.build_page(diffs, task, model, profile, nonce, interactive=True, stamp=when)
    outcome, detail = review.run(page, nonce, writer_fn, open_browser=True)

Two halves.

`build_page` turns edit results into a self-contained HTML file. No network,
no webfonts, no external dependencies -- it opens from disk and works
offline, which matters for a tool that talks to a model on your own hardware.

`run` (backed by `_Handler`/`ThreadingHTTPServer`) serves that page from a
single-purpose local server so its Apply and Cancel buttons do something. It
binds to 127.0.0.1 only. This endpoint writes files, so it has no business
listening on a network interface.

The safety property that matters: the page carries the SHA-256 of every file
as it was when the preview was built. Applying re-hashes and refuses on a
mismatch. Without that, editing a file in your editor while the review sits
open would let you write a preview built from content that no longer exists.
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import fileio
from .index import state_dir

REVIEW_DIR = "reviews"
CONTEXT = 4          # unchanged lines kept either side of a change
COLLAPSE_OVER = 10   # runs of unchanged lines longer than this get folded
TOKEN_RE = re.compile(r"\w+|\s+|.")


# --------------------------------------------------------------------------
# Diff data model
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Diff computation
# --------------------------------------------------------------------------
def _word_spans(a: str, b: str) -> tuple[list, list]:
    """Compute character ranges that differ between two versions of one
    line.

    Line-level highlighting tells you a line changed; this tells you which
    part, which is the whole reason to look at a diff side by side.
    """
    at, bt = TOKEN_RE.findall(a), TOKEN_RE.findall(b)
    sm = SequenceMatcher(None, at, bt, autojunk=False)
    a_spans, b_spans = [], []
    offs_a = [0]
    for t in at:
        offs_a.append(offs_a[-1] + len(t))
    offs_b = [0]
    for t in bt:
        offs_b.append(offs_b[-1] + len(t))
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete") and i2 > i1:
            a_spans.append((offs_a[i1], offs_a[i2]))
        if tag in ("replace", "insert") and j2 > j1:
            b_spans.append((offs_b[j1], offs_b[j2]))
    return a_spans, b_spans


def tag_is_last(sm, tag, i2, alen) -> bool:
    """Return True when an equal run reaches the end of the file, so there
    is no following change that needs trailing context."""
    return i2 >= alen


def build_rows(before: str, after: str) -> tuple[list[Row], int, int]:
    """Compute the full row-by-row diff between two versions of a file's
    text, with word-level highlight spans on replaced lines and long
    unchanged runs collapsed behind a fold marker. Returns
    (rows, lines_added, lines_removed)."""
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

            # How much context to keep at each end. A run at the very top or
            # bottom of the file only needs context on the side facing a change.
            head = 0 if not rows else CONTEXT
            tail = 0 if tag_is_last(sm, tag, i2, len(a)) else CONTEXT
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

    # Trim leading/trailing unchanged bulk that no collapse caught.
    return rows, added, removed


def diffs_from_results(results) -> list[FileDiff]:
    """Convert a list of edits.Result (from apply_edits(dry_run=True)) into
    FileDiff objects ready for rendering. Only successful results with
    computed before/after text produce a diff."""
    out: list[FileDiff] = []
    for r in results:
        if not r.ok or not r.sha:
            continue
        rows, added, removed = build_rows(r.before, r.after)
        out.append(FileDiff(r.file, rows, added, removed, r.sha, r.after, r.message))
    return out


# --------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------
def _mark(text: str, spans: list) -> str:
    """HTML-escape a line, wrapping the given character ranges in <mark> to
    highlight the specific words that changed."""
    if not spans:
        return html.escape(text) or "&nbsp;"
    out, pos = [], 0
    for start, end in spans:
        out.append(html.escape(text[pos:start]))
        out.append('<mark>' + html.escape(text[start:end]) + '</mark>')
        pos = end
    out.append(html.escape(text[pos:]))
    return "".join(out) or "&nbsp;"


# Page styling: a self-contained stylesheet (system fonts only, no CDN) giving
# the review page its two-column ledger appearance -- struck/set colour coding
# for removed/added text, a sticky column header, and the sign-off stamp
# animation shown after a successful write.
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
/* Column heads, so it is always obvious which side is which. */
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
/* The fold down the middle of the ledger. */
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

/* Signature element: the sign-off stamp, struck across the page on apply. */
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

# Page behaviour: fold/unfold of collapsed unchanged runs, the two-step
# sign-off (tick the checkbox, click once to arm, click again within 6s to
# confirm), the Apply/Cancel POST requests to this module's local server, and
# the sign-off stamp shown on success. Enter is explicitly blocked from
# triggering a write.
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
    """Render one FileDiff's rows as HTML <tr> elements, including hidden
    rows behind fold markers and word-level <mark> highlighting."""
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
    """Assemble the complete, self-contained review HTML page for a set of
    FileDiffs. When `interactive` is True the page includes the sign-off
    checkbox and Apply/Discard buttons wired to `nonce`; otherwise it is a
    read-only preview (used when a review server isn't running)."""
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


# --------------------------------------------------------------------------
# Local server
# --------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    """HTTP request handler for the review server: serves the page on GET,
    and handles /apply and /cancel POST requests from the page's JS."""

    server_version = "Pedacito"

    def log_message(self, *a):
        """Suppress default request logging to stderr."""
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        """Write one complete HTTP response."""
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        """Serve the review page at "/"; 404 for anything else."""
        if self.path.split("?")[0] != "/":
            self._send(404, b"not found", "text/plain")
            return
        self._send(200, self.server.page.encode("utf-8"), "text/html; charset=utf-8")

    def do_POST(self):
        """Handle /cancel (end the review with no changes) and /apply
        (verify the nonce, call the server's writer function, and report the
        outcome as JSON)."""
        path = self.path.split("?")[0]
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            body = {}

        if path == "/cancel":
            self.server.outcome = ("cancelled", "No files were changed.")
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
    """Write the reviewed text for every FileDiff to disk, refusing the
    whole operation if any file has changed (or been deleted) since the
    preview's sha was recorded. Backs up originals to `backup_dir` first if
    given. Returns (ok, message)."""
    stale = []
    for fd in diffs:
        p = root / fd.rel
        if not p.exists():
            stale.append(f"{fd.rel} (deleted)")
            continue
        current = hashlib.sha256(fileio.read_text(p, errors="replace")
                                 .encode("utf-8")).hexdigest()
        if current != fd.sha:
            stale.append(f"{fd.rel} (changed on disk)")
    if stale:
        return False, ("Not written: " + ", ".join(stale) +
                       ". Re-run the task to rebuild the review.")

    written = []
    for fd in diffs:
        p = root / fd.rel
        original, style = fileio.read(p, errors="replace")
        if backup_dir is not None:
            dest = backup_dir / fd.rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            fileio.write(dest, original, style)
        fileio.write(p, fd.after_text, style)
        written.append(fd.rel)
    return True, f"{len(written)} file(s) written: " + ", ".join(written)


def save(root: Path, page: str, stamp: str) -> Path:
    """Save the rendered review page to .pedacito/reviews/<stamp>.html for
    later reference, and return its path."""
    d = state_dir(root, REVIEW_DIR)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{stamp}.html"
    p.write_text(page, encoding="utf-8")
    return p


def run(page: str, nonce: str, writer, port: int = 0, open_browser: bool = True,
        timeout: float = 1800.0, on_url=None):
    """Start a local review server bound to 127.0.0.1, optionally open it in
    a browser, and block until the person clicks Apply/Discard or `timeout`
    seconds pass. `writer` is called on Apply to actually write files (e.g.
    `write_files`). Returns (outcome, detail) where outcome is one of
    "applied", "cancelled", or "timeout"."""
    srv = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    srv.page = page
    srv.nonce = nonce
    srv.writer = writer
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
    """Generate a fresh single-use token used to authorise one review's
    Apply request."""
    return secrets.token_urlsafe(16)
