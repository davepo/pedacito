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
import inspect
import json
import logging
import re
import secrets
import threading
import time
import webbrowser
from bisect import bisect_right
from contextlib import nullcontext
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from http.server import BaseHTTPRequestHandler
from itertools import accumulate
from pathlib import Path

from . import fileio, safety, security, session_ctx
from .index import state_dir
from .local_server import LocalHTTPServer

REVIEW_DIR = "reviews"
CONTEXT = 4          # unchanged lines kept either side of a change
COLLAPSE_OVER = 10   # runs of unchanged lines longer than this get folded
TOKEN_RE = re.compile(r"\w+|\s+|.")
MAX_JSON_BODY_BYTES = 10 * 1024 * 1024


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
    hunk: int = -1            # index into FileDiff.hunks; -1 = context/skip


@dataclass
class Hunk:
    """One independently sign-off-able change group within a file diff.

    `a_start`/`b_start` are 1-based line numbers (0 when the side is empty,
    unified-diff style), `a_count`/`b_count` the span lengths including the
    CONTEXT lines kept either side. `rows` holds the indices of this hunk's
    rows in `FileDiff.rows`."""

    index: int
    a_start: int
    a_count: int
    b_start: int
    b_count: int
    added: int
    removed: int
    rows: list[int] = field(default_factory=list)
    clauses: tuple[str, ...] = ()


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
    # A whole-file deletion (DELETE block / DELETE_FILE): sign-off backs the
    # file up and unlinks it instead of writing `after_text` (which is "").
    deleted: bool = False
    # Requirement clause ids ("R1", ...) this file's edits serve (R8); empty
    # when no tracker evidence is available (e.g. diffs rebuilt from backups).
    clauses: tuple[str, ...] = ()
    # The change segmented into hunks (build_diff); empty for legacy diffs
    # and display-only rename sources, which are accepted whole-file only.
    hunks: list[Hunk] = field(default_factory=list)
    # The file's exact pre-change text, for reconstruct_from_hunks. Empty
    # means "rebuild from the rows" (see apply_hunk_selection).
    before_text: str = ""
    # Set by apply_hunk_selection when only some hunks were accepted: the
    # accepted hunk indices. None = the whole file (legacy sign-off).
    hunk_selection: tuple[int, ...] | None = None


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


def _hunk_groups(opcodes) -> list[tuple[int, int, int, int, int, int]]:
    """Group the change opcodes into hunks: consecutive changes separated by
    an equal run of more than 2*CONTEXT lines start a new hunk. Each group is
    (first_opcode, last_opcode, a_lo, a_hi, b_lo, b_hi) with up to CONTEXT
    equal lines of context on either side; the a/b ranges of different
    groups never overlap (a split gap is wider than both contexts combined)."""
    spans: list[list[int]] = []
    for i, (tag, i1, i2, _j1, _j2) in enumerate(opcodes):
        if tag == "equal":
            continue
        if spans and i1 - opcodes[spans[-1][1]][2] <= 2 * CONTEXT:
            spans[-1][1] = i
        else:
            spans.append([i, i])
    out = []
    for s, e in spans:
        head = tail = 0
        if s > 0 and opcodes[s - 1][0] == "equal":
            head = min(CONTEXT, opcodes[s - 1][2] - opcodes[s - 1][1])
        if e + 1 < len(opcodes) and opcodes[e + 1][0] == "equal":
            tail = min(CONTEXT, opcodes[e + 1][2] - opcodes[e + 1][1])
        first, last = opcodes[s], opcodes[e]
        out.append((s, e, first[1] - head, last[2] + tail,
                    first[3] - head, last[4] + tail))
    return out


def _rows_from_opcodes(a: list[str], b: list[str], opcodes,
                       groups) -> tuple[list[Row], int, int]:
    """The row-by-row diff table (word spans on replaced lines, long equal
    runs folded), with every row tagged with its hunk index."""
    a_los = [g[2] for g in groups]
    a_his = [g[3] for g in groups]
    b_los = [g[4] for g in groups]
    b_his = [g[5] for g in groups]

    def hunk_of(a_no, b_no) -> int:
        if a_no is not None:
            gi = bisect_right(a_los, a_no - 1) - 1
            if gi >= 0 and a_no - 1 < a_his[gi]:
                return gi
        if b_no is not None:
            gi = bisect_right(b_los, b_no - 1) - 1
            if gi >= 0 and b_no - 1 < b_his[gi]:
                return gi
        return -1

    rows: list[Row] = []
    folds = [0]
    added = removed = 0

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            n = i2 - i1

            def eq(k, fold=""):
                """Build one equal-line Row at offset k within this run."""
                return Row("equal", i1 + k + 1, a[i1 + k], j1 + k + 1, b[j1 + k],
                           fold=fold, hunk=hunk_of(i1 + k + 1, j1 + k + 1))

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
                    rows.append(Row("replace", i1 + k + 1, la, j1 + k + 1, lb,
                                    sa, sb, hunk=hunk_of(i1 + k + 1, j1 + k + 1)))
                    added += 1
                    removed += 1
                elif la is not None:
                    rows.append(Row("delete", i1 + k + 1, la,
                                    hunk=hunk_of(i1 + k + 1, None)))
                    removed += 1
                else:
                    rows.append(Row("insert", None, "", j1 + k + 1, lb,
                                    hunk=hunk_of(None, j1 + k + 1)))
                    added += 1
        elif tag == "delete":
            for k in range(i1, i2):
                rows.append(Row("delete", k + 1, a[k], hunk=hunk_of(k + 1, None)))
                removed += 1
        elif tag == "insert":
            for k in range(j1, j2):
                rows.append(Row("insert", None, "", k + 1, b[k],
                                hunk=hunk_of(None, k + 1)))
                added += 1

    return rows, added, removed


def build_diff(before: str, after: str) -> tuple[list[Row], int, int, list[Hunk]]:
    """Like `build_rows`, but also segments the change into hunks a reviewer
    can accept independently. Returns (rows, added, removed, hunks); each
    row's `hunk` field indexes into `hunks` (-1 for folded/skip context)."""
    a = before.split("\n")
    b = after.split("\n")
    sm = SequenceMatcher(None, a, b, autojunk=False)
    opcodes = sm.get_opcodes()
    groups = _hunk_groups(opcodes)
    rows, added, removed = _rows_from_opcodes(a, b, opcodes, groups)
    stats: dict[int, list] = {gi: [0, 0, []] for gi in range(len(groups))}
    for ri, row in enumerate(rows):
        if row.hunk >= 0:
            st = stats[row.hunk]
            st[2].append(ri)
            if row.kind in ("replace", "insert"):
                st[0] += 1
            if row.kind in ("replace", "delete"):
                st[1] += 1
    hunks = []
    for gi, (_s, _e, a_lo, a_hi, b_lo, b_hi) in enumerate(groups):
        h_added, h_removed, h_rows = stats[gi]
        hunks.append(Hunk(gi,
                          a_lo + 1 if a_hi > a_lo else a_lo, a_hi - a_lo,
                          b_lo + 1 if b_hi > b_lo else b_lo, b_hi - b_lo,
                          h_added, h_removed, h_rows))
    return rows, added, removed, hunks


def build_rows(before: str, after: str) -> tuple[list[Row], int, int]:
    """Row-by-row diff of two texts, with word-level spans on replaced lines
    and long unchanged runs folded. Returns (rows, lines_added, lines_removed)."""
    rows, added, removed, _hunks = build_diff(before, after)
    return rows, added, removed


def reconstruct_from_hunks(before_text: str, after_text: str,
                           accepted_hunk_ids) -> str:
    """The after-text with only the accepted hunks applied: equal lines always
    come from `before_text`, an accepted change from `after_text`, an
    unaccepted delete/replace keeps the before lines and an unaccepted insert
    is dropped. Accepting every hunk reproduces `after_text` exactly;
    accepting none reproduces `before_text`."""
    accepted = set(accepted_hunk_ids or ())
    a = before_text.split("\n")
    b = after_text.split("\n")
    sm = SequenceMatcher(None, a, b, autojunk=False)
    opcodes = sm.get_opcodes()
    owner: dict[int, int] = {}
    for gi, (s, e, *_ranges) in enumerate(_hunk_groups(opcodes)):
        for oi in range(s, e + 1):
            owner[oi] = gi
    out: list[str] = []
    for oi, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        if tag == "equal":
            out.extend(a[i1:i2])
        elif owner.get(oi, -1) in accepted:
            out.extend(b[j1:j2])
        elif tag in ("delete", "replace"):
            out.extend(a[i1:i2])
        # an unaccepted insert contributes nothing
    return "\n".join(out)


def _before_from_rows(fd) -> str:
    """Rebuild a file's pre-change text from its diff rows (the fallback when
    FileDiff.before_text was not recorded)."""
    return "\n".join(r.a for r in getattr(fd, "rows", ())
                     if r.kind not in ("insert", "skip"))


def apply_hunk_selection(diffs: list[FileDiff],
                         accepted_hunks: dict) -> tuple[bool, str]:
    """Apply a {rel: [hunk indices]} selection to staged FileDiffs in place.

    A file absent from the map is accepted whole; an empty index list drops
    the file; a partial selection reconstructs its after-text from the
    accepted hunks and refuses (before mutating anything) when the result
    would not parse. Returns (ok, detail); on failure the diffs are untouched
    so the review stays retryable."""
    from . import edits                     # lazy: edits never imports review
    plans = []
    any_writable = False
    for fd in diffs:
        if fd.removed_source or fd.deleted:
            continue
        sel = accepted_hunks.get(fd.rel)
        if sel is None:
            any_writable = True             # whole-file acceptance
            continue
        hunks = getattr(fd, "hunks", None) or []
        if not hunks:
            if sel:                         # legacy diff: kept whole
                any_writable = True
            continue
        ids = sorted(set(sel))
        if not ids:
            continue                        # file fully deselected
        if any(i < 0 or i >= len(hunks) for i in ids):
            return False, f"Not written: unknown hunk index for {fd.rel}."
        before = getattr(fd, "before_text", "") or _before_from_rows(fd)
        text = reconstruct_from_hunks(before, fd.after_text, ids)
        if len(ids) < len(hunks):
            err = edits.syntax_error(Path(fd.rel), text)
            if err:
                return False, (f"Not written: accepting only part of {fd.rel} "
                               f"would leave invalid syntax ({err}). Accept "
                               "the whole file or adjust the hunk selection.")
        plans.append((fd, text, tuple(ids), hunks))
        any_writable = True
    if not any_writable:
        return False, "Select at least one file or hunk to accept."
    for fd, text, ids, hunks in plans:
        fd.after_text = text
        fd.hunk_selection = ids
        if len(ids) < len(hunks):
            fd.added = sum(hunks[i].added for i in ids)
            fd.removed = sum(hunks[i].removed for i in ids)
    return True, ""


# --------------------------------------------------------------------------
# Per-session review routing (D9)
# --------------------------------------------------------------------------
_log = logging.getLogger(__name__)


def is_sensitive_diff(fd) -> bool:
    """True when a FileDiff stages secret/credential material."""
    if security.is_sensitive_path(getattr(fd, "rel", "")):
        return True
    return any(security.is_sensitive_path(old)
               for old in (getattr(fd, "renames_from", ()) or ()))


def filter_sensitive_diffs(diffs) -> list:
    """Drop every FileDiff that names (or renames from) a sensitive path.

    When nothing is sensitive the ORIGINAL list object is returned, so a
    caller that captured the diffs list (the legacy review writer) keeps
    seeing the same in-place selection the GUI filters."""
    if not diffs:
        return diffs if isinstance(diffs, list) else []
    kept: list = []
    skipped: list[str] = []
    for fd in diffs:
        if is_sensitive_diff(fd):
            skipped.append(str(getattr(fd, "rel", "")))
            continue
        kept.append(fd)
    if skipped:
        _log.warning("review skipped sensitive path(s): %s",
                     ", ".join(skipped))
        return kept
    return diffs if isinstance(diffs, list) else kept


@dataclass
class StagedReview:
    """One review waiting in a session's queue."""

    review_id: str
    session_id: str
    task: str
    diffs: list[FileDiff]
    writer: "callable"
    on_discard: "callable | None" = None
    partial_ok: bool = True
    clause_texts: dict[str, str] = field(default_factory=dict)
    created_at: float = 0.0
    expires_at: float = 0.0
    outcome: str = ""               # "" | applied | cancelled | timeout | expired
    detail: str = ""
    done: threading.Event = field(default_factory=threading.Event)
    # GUI compatibility: the transcript message id, hand edits and accepted
    # hunk bookkeeping mirror the legacy `gui.core.Review` shape.
    edits: dict[str, str] = field(default_factory=dict)
    unaccepted: list[str] = field(default_factory=list)
    accepted_hunks: dict[str, list[int]] = field(default_factory=dict)
    mid: int = 0


class ReviewQueue:
    """Thread-safe per-session review queue (D9).

    Each session owns its own queue, so two concurrent sessions can stage
    reviews without sharing ids, pending lists or decisions. Review ids are
    ``rev_<session-short>_<seq>``; a review id from another session is never
    visible (and the GUI rejects it)."""

    def __init__(self, session_id: str, *,
                 default_ttl_s: float = 1800.0) -> None:
        self.session_id = str(session_id or "session")
        clean = re.sub(r"[^A-Za-z0-9]", "", self.session_id)
        if clean.lower().startswith("ses"):
            clean = clean[3:]
        self._short = (clean[-8:] or "default")
        self.default_ttl_s = float(default_ttl_s or 1800.0)
        self._lock = threading.RLock()
        self._reviews: dict[str, StagedReview] = {}
        self._seq = 0
        # A non-GUI host (TUI/RPC/daemon) sets this to be notified when a
        # review is staged; `review.run` then routes through this queue.
        self.listener = None

    # -- configuration -----------------------------------------------------
    def set_listener(self, listener) -> None:
        """Install (or clear, with None) the staging callback."""
        with self._lock:
            self.listener = listener

    def has_listener(self) -> bool:
        with self._lock:
            return callable(self.listener)

    # -- staging -----------------------------------------------------------
    def stage(self, *, task: str, diffs: list[FileDiff], writer,
              on_discard=None, partial_ok: bool = True,
              clause_texts: dict[str, str] | None = None,
              ttl_s: float | None = None) -> StagedReview:
        """Queue a review for this session; sensitive diffs are dropped."""
        filtered = list(filter_sensitive_diffs(diffs))
        now = time.time()
        ttl = self.default_ttl_s if ttl_s is None else max(0.0, float(ttl_s))
        with self._lock:
            self._seq += 1
            review = StagedReview(
                review_id=f"rev_{self._short}_{self._seq}",
                session_id=self.session_id,
                task=str(task or ""),
                diffs=filtered,
                writer=writer,
                on_discard=on_discard,
                partial_ok=bool(partial_ok),
                clause_texts=dict(clause_texts or {}),
                created_at=now,
                expires_at=now + ttl,
            )
            self._reviews[review.review_id] = review
            listener = self.listener
        if callable(listener):
            try:
                listener(review)
            except Exception:
                _log.exception("review queue listener failed")
        return review

    # -- lookup ------------------------------------------------------------
    def get(self, review_id: str) -> StagedReview | None:
        with self._lock:
            return self._reviews.get(str(review_id or ""))

    def list_pending(self) -> list[dict]:
        """JSON-safe summaries of every undecided review."""
        with self._lock:
            return [{
                "review_id": r.review_id,
                "session_id": r.session_id,
                "task": r.task,
                "files": [d.rel for d in r.diffs if not d.removed_source],
                "partial_ok": r.partial_ok,
                "created_at": r.created_at,
                "expires_at": r.expires_at,
            } for r in self._reviews.values() if not r.outcome]

    def pending_reviews(self) -> list[StagedReview]:
        """The live StagedReview objects, oldest first (GUI convenience)."""
        with self._lock:
            return sorted((r for r in self._reviews.values() if not r.outcome),
                          key=lambda r: r.created_at)

    # -- decisions ---------------------------------------------------------
    def decide(self, review_id: str, action: str, *,
               files: list[str] | None = None,
               accepted_hunks: dict[str, list[int]] | None = None
               ) -> tuple[bool, str]:
        """Apply or reject one review. Returns (ok, detail)."""
        with self._lock:
            review = self._reviews.get(str(review_id or ""))
            if review is None:
                return False, f"unknown review {review_id!r}"
            if review.outcome:
                return False, f"this review was already {review.outcome}"
            now = time.time()
            if review.expires_at and now >= review.expires_at:
                self._expire_locked(review)
                return False, "this review expired; stage it again"
            verb = str(action or "").lower()
            if verb in ("reject", "cancel", "discard"):
                detail = self._discard_locked(review, "cancelled",
                                              "No files were changed.")
                return True, detail or "cancelled"
            if verb != "apply":
                return False, f"unknown review action {action!r}"

            hmap = accepted_hunks if isinstance(accepted_hunks, dict) else {}
            if review.partial_ok and (files is not None or hmap):
                keep = (set(files) if files is not None
                        else {d.rel for d in review.diffs
                              if not d.removed_source}) | set(hmap)
                snapshot = list(review.diffs)
                fd_snap = [(d, d.after_text, d.added, d.removed,
                            getattr(d, "hunk_selection", None))
                           for d in review.diffs]
                review.diffs[:] = [d for d in review.diffs
                                   if d.removed_source or d.rel in keep]
                if not any(not d.removed_source for d in review.diffs):
                    review.diffs[:] = snapshot
                    return False, ("Select at least one file or hunk to accept."
                                   if hmap else
                                   "Select at least one file to accept.")
                ok, detail = apply_hunk_selection(review.diffs, hmap)
                if not ok:
                    review.diffs[:] = snapshot
                    for d, after, added, removed, sel in fd_snap:
                        d.after_text = after
                        d.added = added
                        d.removed = removed
                        d.hunk_selection = sel
                    return False, detail
                for d in review.diffs:
                    if d.rel in review.edits:
                        d.after_text = review.edits[d.rel]
                review.accepted_hunks = {}
                for d in review.diffs:
                    if d.removed_source:
                        continue
                    sel = getattr(d, "hunk_selection", None)
                    if sel is not None:
                        review.accepted_hunks[d.rel] = list(sel)
                    elif d.rel in hmap:
                        review.accepted_hunks[d.rel] = sorted(set(hmap[d.rel]))
            elif review.edits:
                for d in review.diffs:
                    if d.rel in review.edits:
                        d.after_text = review.edits[d.rel]

            try:
                ok, detail = _call_staged_writer(
                    review.writer, review.diffs, files, hmap)
            except Exception as exc:
                return False, f"write failed: {exc}"
            detail = str(detail or "")
            if not ok:
                review.detail = detail
                return False, detail
            review.outcome = "applied"
            review.detail = detail
            review.done.set()
            return True, detail

    def _discard_locked(self, review: StagedReview, outcome: str,
                        detail: str, reason: str = "") -> str:
        if review.on_discard is not None:
            try:
                review.on_discard()
            except Exception as exc:
                detail = f"Discard failed: {exc}"
        review.outcome = outcome
        review.detail = detail or reason or outcome
        review.done.set()
        return review.detail

    def _expire_locked(self, review: StagedReview) -> None:
        self._discard_locked(
            review, "expired",
            "Review window timed out; nothing was written.")

    def expire_stale(self, now: float | None = None) -> int:
        """Expire every undecided review whose TTL has elapsed."""
        moment = time.time() if now is None else float(now)
        with self._lock:
            stale = [r for r in self._reviews.values()
                     if not r.outcome and r.expires_at
                     and moment >= r.expires_at]
            for review in stale:
                self._expire_locked(review)
            return len(stale)

    def cancel_all(self, reason: str = "session cancelled") -> int:
        """Cancel every undecided review (a session teardown unblocks waiters)."""
        with self._lock:
            pending = [r for r in self._reviews.values() if not r.outcome]
            for review in pending:
                self._discard_locked(
                    review, "cancelled",
                    str(reason or "Cancelled; nothing was written."),
                    reason=reason)
            return len(pending)


def _call_staged_writer(writer, diffs: list[FileDiff],
                        files: list[str] | None,
                        accepted_hunks: dict | None = None):
    """Invoke a queue writer, supporting every existing signature.

    - a writer naming `files`/`accepted_hunks` receives those selections;
    - a writer with one required positional receives the accepted {rel: text}
      map (the legacy `_call_writer` contract);
    - a zero-argument writer (the GUI and standalone page writers) is called
      as-is."""
    try:
        params = inspect.signature(writer).parameters
    except (TypeError, ValueError):
        params = {}
    if "files" in params or "accepted_hunks" in params:
        kwargs: dict = {}
        if "files" in params:
            kwargs["files"] = list(files) if files is not None else None
        if "accepted_hunks" in params:
            kwargs["accepted_hunks"] = dict(accepted_hunks or {})
        return writer(**kwargs)
    required = [p for p in params.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                and p.default is p.empty]
    if len(required) >= 2:
        return writer(list(files) if files is not None else None,
                      dict(accepted_hunks or {}))
    if required and diffs is not None:
        return writer({fd.rel: fd.after_text for fd in diffs
                       if not fd.removed_source and not fd.deleted})
    return writer()


def hunk_clauses_for_file(fd: FileDiff,
                          clause_texts: dict | None = None) -> dict[int, tuple[str, ...]]:
    """Attribute a file's clauses to individual hunks when the file has
    multiple hunks. Returns {hunk_index: (clause_id, ...)}.

    If a hunk carries explicit `h.clauses`, those win. Otherwise, when
    `clause_texts` (or a RequirementTracker's terms) are available, each clause
    on `fd.clauses` is matched against the changed lines of each hunk; any
    clause that matches no specific hunk (or when `clause_texts` is empty and
    the file has a single clause) falls back to all hunks (or positional 1-to-1
    when len(clauses) == len(hunks) and no terms matched)."""
    hunks = getattr(fd, "hunks", None) or []
    f_clauses = tuple(getattr(fd, "clauses", ()) or ())
    if not hunks:
        return {}
    out: dict[int, list[str]] = {h.index: list(getattr(h, "clauses", ()) or ())
                                 for h in hunks}
    if any(out.values()) or not f_clauses:
        return {k: tuple(v) for k, v in out.items()}
    if len(hunks) == 1:
        return {hunks[0].index: f_clauses}

    rows = getattr(fd, "rows", None) or []
    hunk_bodies: dict[int, str] = {}
    for h in hunks:
        parts: list[str] = []
        for ri in h.rows:
            if 0 <= ri < len(rows):
                r = rows[ri]
                if r.kind != "equal":
                    parts.append(f"{r.a}\n{r.b}")
        if not parts:
            for ri in h.rows:
                if 0 <= ri < len(rows):
                    r = rows[ri]
                    parts.append(f"{r.a}\n{r.b}")
        hunk_bodies[h.index] = "\n".join(parts)

    clause_texts = clause_texts or {}
    unmatched: list[str] = []
    for cid in f_clauses:
        ctext = str(clause_texts.get(cid) or "")
        tokens = [w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", ctext)
                  if w.lower() not in {"the", "and", "for", "with", "from",
                                       "into", "that", "this", "add", "fix",
                                       "make", "update", "change", "set",
                                       "use", "file", "line", "lines", "def",
                                       "return", "function", "class"}]
        rel_base = fd.rel.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
        tokens = [w for w in tokens if w.lower() != rel_base
                  and w.lower() != fd.rel.lower()]
        matched_hunks: list[int] = []
        if tokens:
            best_score = 0
            scores: dict[int, int] = {}
            for h in hunks:
                body_low = hunk_bodies.get(h.index, "").lower()
                sc = sum(2 if ("_" in tok or any(c.isupper() for c in tok[1:]))
                         else 1
                         for tok in set(tokens)
                         if re.search(r"(?<![\w])" + re.escape(tok.lower()) + r"(?![\w])",
                                      body_low))
                scores[h.index] = sc
                if sc > best_score:
                    best_score = sc
            if best_score > 0:
                matched_hunks = [hi for hi, sc in scores.items()
                                 if sc == best_score]
        if matched_hunks:
            for hi in matched_hunks:
                if cid not in out[hi]:
                    out[hi].append(cid)
        else:
            unmatched.append(cid)

    if unmatched:
        if len(unmatched) == len(f_clauses) and len(f_clauses) == len(hunks):
            for h, cid in zip(hunks, f_clauses):
                if cid not in out[h.index]:
                    out[h.index].append(cid)
        else:
            empty_hunks = [h.index for h in hunks if not out[h.index]]
            targets = empty_hunks if len(unmatched) == 1 and len(empty_hunks) == 1 \
                else [h.index for h in hunks]
            for cid in unmatched:
                for hi in targets:
                    if cid not in out[hi]:
                        out[hi].append(cid)
    return {k: tuple(v) for k, v in out.items()}


def diffs_from_results(results, root=None, clauses=None,
                       clause_texts=None) -> list[FileDiff]:
    """Convert edits.Result objects (from apply_edits(dry_run=True)) into
    FileDiffs. Only successful results with computed before/after text count.

    `root` is only consulted for renames, where each source file's real
    contents are shown as a deletion. `clauses` is an optional file -> [clause
    ids] map (from the requirement tracker) attached to each FileDiff so the
    review can show which task clause a file serves (R8)."""
    clauses = clauses or {}
    out: list[FileDiff] = []
    for r in results:
        if not r.ok or not r.sha:
            continue
        if security.is_sensitive_path(r.file) or any(
                security.is_sensitive_path(old) for old in r.renames_from):
            _log.warning("review skipped sensitive path: %s", r.file)
            continue
        rows, added, removed, hunks = build_diff(r.before, r.after)
        ids = tuple(clauses.get(r.file, ()))
        fd = FileDiff(r.file, rows, added, removed, r.sha, r.after,
                      r.message, renames_from=tuple(r.renames_from),
                      clauses=ids, deleted=getattr(r, "deleted", False),
                      hunks=hunks, before_text=r.before)
        if ids and hunks:
            hc = hunk_clauses_for_file(fd, clause_texts)
            for h in hunks:
                h.clauses = hc.get(h.index, ())
        out.append(fd)
        if root is None:
            continue
        for old in r.renames_from:
            if security.is_sensitive_path(old):
                _log.warning("review skipped sensitive rename source: %s", old)
                continue
            try:
                otext, _ = fileio.read(Path(root) / old, errors="replace")
            except OSError:
                continue
            orows, oa, om = build_rows(otext, "")
            out.append(FileDiff(old, orows, oa, om, _sha(otext), "",
                                "removed (renamed to " + r.file + ")",
                                removed_source=True, clauses=ids))
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
        if src.is_symlink() or not src.is_file():
            continue
        rel = src.relative_to(backup_dir).as_posix()
        if security.is_sensitive_path(rel):
            _log.warning("review skipped sensitive backup path: %s", rel)
            continue
        try:
            cur = security.resolve_safe_path(root, rel)
        except (security.SecurityError, OSError, ValueError):
            continue
        if cur.is_symlink():
            continue
        if not cur.is_file():
            # Backed up but gone from the tree: the run deleted (or renamed
            # away) this file. Show the deletion instead of skipping it.
            before, _ = fileio.read(src, errors="replace")
            rows, added, removed = build_rows(before, "")
            out.append(FileDiff(rel, rows, added, removed, _sha(before), "",
                                "deleted", deleted=True))
            continue
        before, _ = fileio.read(src, errors="replace")
        after, _ = fileio.read(cur, errors="replace")
        if before == after:
            continue
        rows, added, removed, hunks = build_diff(before, after)
        out.append(FileDiff(rel, rows, added, removed, _sha(after), after,
                            f"built ({added}+ / {removed}-)",
                            hunks=hunks, before_text=before))
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
.clause {
  font: 500 11px/1 "Segoe UI", system-ui, sans-serif; color: var(--ink-2);
  background: var(--band); border: 1px solid var(--hair);
  padding: 2px 6px; border-radius: 2px; letter-spacing: 0.04em;
}

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

/* Hunk-level sign-off: a header row per hunk, a checkbox per file. */
.file > h2 input.file-box {
  width: 16px; height: 16px; accent-color: var(--set); flex: none; cursor: pointer;
}
tr.hunk-head td {
  background: var(--band); border-top: 1px solid var(--rule);
  padding: 4px 10px; font-size: 12px; color: var(--ink-2);
}
tr.hunk-head label {
  display: flex; align-items: center; gap: 9px; cursor: pointer;
}
tr.hunk-head input.hunk-box {
  width: 15px; height: 15px; accent-color: var(--set); flex: none; cursor: pointer;
}
tr.hunk-head .hunk-range {
  font-family: "Cascadia Mono", Consolas, monospace; font-size: 11.5px;
}
tr.hunk-head .tally { margin-left: auto; }
tr.hunk-head.off td { opacity: 0.55; }

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

// Hunk-level sign-off: a file box selects the whole file, hunk boxes select
// parts of it; a partially selected file shows an indeterminate file box.
function hunkBoxes(fid) {
  return Array.prototype.slice.call(
    document.querySelectorAll('.hunk-box[data-file="' + fid + '"]'));
}
function syncFileBox(fid) {
  var fb = document.querySelector('.file-box[data-file="' + fid + '"]');
  if (!fb) return;
  var hs = hunkBoxes(fid);
  if (!hs.length) return;
  var on = hs.filter(function (hb) { return hb.checked; }).length;
  fb.checked = on > 0;
  fb.indeterminate = on > 0 && on < hs.length;
}
document.querySelectorAll('.file-box').forEach(function (fb) {
  fb.addEventListener('change', function () {
    fb.indeterminate = false;
    hunkBoxes(fb.dataset.file).forEach(function (hb) {
      hb.checked = fb.checked;
      hb.closest('tr').classList.toggle('off', !fb.checked);
    });
  });
});
document.querySelectorAll('.hunk-box').forEach(function (hb) {
  hb.addEventListener('change', function () {
    hb.closest('tr').classList.toggle('off', !hb.checked);
    syncFileBox(hb.dataset.file);
  });
});
function selection() {
  var accepted = [], acceptedHunks = {};
  document.querySelectorAll('.file-box').forEach(function (fb) {
    var rel = fb.dataset.rel;
    var hs = hunkBoxes(fb.dataset.file);
    var on = hs.filter(function (hb) { return hb.checked; });
    if (!hs.length) { if (fb.checked) accepted.push(rel); return; }
    if (on.length === hs.length) { accepted.push(rel); }
    else if (on.length > 0) {
      accepted.push(rel);
      acceptedHunks[rel] = on.map(function (hb) {
        return parseInt(hb.dataset.hunk, 10);
      });
    }
  });
  return { accepted: accepted, accepted_hunks: acceptedHunks };
}

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
    var body = selection();
    body.nonce = window.NONCE;
    post('/apply', body).then(function (res) {
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
    post('/cancel', { nonce: window.NONCE }).then(function () {
      finish('Discarded', 'No files were changed.');
    }).catch(function () { finish('Discarded', 'No files were changed.'); });
  });
}

// Enter must never trigger a write.
document.addEventListener('keydown', function (e) {
  if (e.key === 'Enter' && document.activeElement === write) { e.preventDefault(); }
});
"""


def _rows_html(fd: FileDiff, fid: int, interactive: bool = False) -> str:
    """Render one FileDiff's rows as HTML <tr> elements (folded rows hidden).
    Interactive pages with hunks get a checkbox header row per hunk."""
    out = []
    hunks = getattr(fd, "hunks", None) or []
    show_hunks = (interactive and bool(hunks)
                  and not fd.removed_source and not fd.deleted)
    prev_hunk = -1
    for row in fd.rows:
        if row.kind == "skip":
            out.append(
                f'<tr class="skip" data-btn="{fid}-{row.fold}"><td colspan="4">'
                f'<button data-fold="{fid}-{row.fold}">show {row.hidden} unchanged '
                f'line{"s" if row.hidden != 1 else ""}</button></td></tr>')
            continue
        if show_hunks and row.hunk >= 0 and row.hunk != prev_hunk:
            prev_hunk = row.hunk
            h = hunks[row.hunk]
            out.append(
                f'<tr class="hunk-head"><td colspan="4"><label>'
                f'<input type="checkbox" class="hunk-box" data-file="{fid}" '
                f'data-hunk="{h.index}" checked>'
                f'Hunk {h.index + 1}'
                f'<span class="hunk-range">@@ -{h.a_start},{h.a_count} '
                f'+{h.b_start},{h.b_count} @@</span>'
                f'<span class="tally"><span class="p">+{h.added}</span> '
                f'&nbsp;<span class="m">&minus;{h.removed}</span></span>'
                f'</label></td></tr>')
        a_no = "" if row.a_no is None else row.a_no
        b_no = "" if row.b_no is None else row.b_no
        # Displayed text is redacted; the FileDiff's before_text/after_text
        # stay intact so partial hunk selection still writes real code.
        a_plain = safety.redact(row.a)
        b_plain = safety.redact(row.b)
        a_spans = row.a_spans if a_plain == row.a else []
        b_spans = row.b_spans if b_plain == row.b else []
        a_txt = _mark(a_plain, a_spans) if row.kind == "replace" else (
            html.escape(a_plain) or "&nbsp;")
        b_txt = _mark(b_plain, b_spans) if row.kind == "replace" else (
            html.escape(b_plain) or "&nbsp;")
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
               nonce: str, interactive: bool, stamp: str,
               clause_texts: dict | None = None) -> str:
    """Assemble the self-contained review page. `interactive` adds the sign-off
    checkbox and Apply/Discard buttons wired to `nonce`; otherwise the page is a
    read-only preview. `clause_texts` (clause id -> text) titles the per-file
    clause badges so a reviewer sees which task clause each file serves."""
    clause_texts = clause_texts or {}
    files_html = []
    for i, fd in enumerate(diffs):
        badges = "".join(
            f'<span class="clause" title="{html.escape(safety.redact(clause_texts.get(c, c)))}">'
            f'{html.escape(c)}</span>' for c in fd.clauses)
        file_box = ""
        if interactive and not fd.removed_source:
            file_box = (f'<input type="checkbox" class="file-box" '
                        f'data-file="{i}" data-rel="{html.escape(fd.rel)}" '
                        f'checked title="include this file in the sign-off">')
        files_html.append(f"""
<section class="file">
  <h2>{file_box}{html.escape(fd.rel)}
    {badges}
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
{_rows_html(fd, i, interactive)}
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
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🍰</text></svg>">
<title>Review &mdash; {html.escape(safety.redact(task[:60]))}</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>Proposed changes <span class="sub">awaiting sign-off</span></h1>
  <p class="task">{html.escape(safety.redact(task))}</p>
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
    """Parse the request body as a JSON object; {} when absent, oversized, or malformed."""
    try:
        n = max(0, int(handler.headers.get("Content-Length") or 0))
    except ValueError:
        return {}
    if n > MAX_JSON_BODY_BYTES:
        return {}
    try:
        body = json.loads(handler.rfile.read(n) or b"{}")
    except (ValueError, UnicodeDecodeError):
        return {}
    return body if isinstance(body, dict) else {}


def _apply_page_selection(diffs: list, accepted, accepted_hunks) -> tuple[bool, str]:
    """Filter staged diffs to the page's file selection and apply its hunk
    choices, in place. A refused selection restores the snapshot so the
    review stays retryable (the same contract as the GUI's decide_review)."""
    hmap = accepted_hunks if isinstance(accepted_hunks, dict) else {}
    snapshot = list(diffs)
    fd_snap = [(fd, fd.after_text, fd.added, fd.removed,
                getattr(fd, "hunk_selection", None))
               for fd in diffs]
    if isinstance(accepted, list):
        keep = set(accepted) | set(hmap)
        diffs[:] = [fd for fd in diffs
                    if fd.removed_source or fd.rel in keep]
    ok, detail = apply_hunk_selection(diffs, hmap)
    if not ok:
        diffs[:] = snapshot
        for fd, after, added, removed, sel in fd_snap:
            fd.after_text = after
            fd.added = added
            fd.removed = removed
            if hasattr(fd, "hunk_selection"):
                fd.hunk_selection = sel
    return ok, detail


def _call_writer(writer, diffs):
    """Call the sign-off writer. Legacy writers take no arguments and close
    over the diffs list (which _apply_page_selection just filtered); a writer
    that declares a required parameter receives the accepted {rel: text} map
    instead, so it never has to reach into review internals."""
    try:
        params = inspect.signature(writer).parameters.values()
        required = [p for p in params
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                    and p.default is p.empty]
    except (TypeError, ValueError):
        required = []
    if required and diffs is not None:
        return writer({fd.rel: fd.after_text for fd in diffs
                       if not fd.removed_source and not fd.deleted})
    return writer()


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
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("X-Content-Type-Options", "nosniff")
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
        if path not in ("/apply", "/cancel"):
            self._send(404, b'{"ok":false}', "application/json")
            return
        body = read_json_body(self)

        raw_nonce = body.get("nonce")
        expected_nonce = getattr(self.server, "nonce", None)
        valid_nonce = (isinstance(raw_nonce, str)
                       and isinstance(expected_nonce, str)
                       and secrets.compare_digest(raw_nonce, expected_nonce))
        if not valid_nonce:
            self._send(403, json.dumps({
                "ok": False,
                "detail": "This review is no longer valid. Re-run the task."
            }).encode(), "application/json")
            return

        lock = getattr(self.server, "lock", None) or nullcontext()
        with lock:
            if self.server.done.is_set():
                self._send(409, json.dumps({
                    "ok": False,
                    "detail": "This review has already been decided."
                }).encode(), "application/json")
                return

            if path == "/cancel":
                self.server.outcome = ("cancelled", "No files were changed.")
                if self.server.on_discard is not None:
                    try:
                        self.server.on_discard()
                    except Exception as e:          # never hang the review loop
                        self.server.outcome = ("cancelled", f"Discard failed: {e}")
                self.server.done.set()
                self._send(200, b'{"ok":true}', "application/json")
                return

            diffs = getattr(self.server, "diffs", None)
            if diffs is not None and ("accepted" in body
                                      or "accepted_hunks" in body):
                ok, detail = _apply_page_selection(
                    diffs, body.get("accepted"), body.get("accepted_hunks"))
                if not ok:
                    self._send(200, json.dumps({"ok": False,
                                                "detail": detail}).encode(),
                               "application/json")
                    return

            ok, detail = _call_writer(self.server.writer, diffs)
            if ok:
                self.server.outcome = ("applied", detail)
                self.server.done.set()
            self._send(200, json.dumps({"ok": ok, "detail": detail}).encode(),
                       "application/json")


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
    root = Path(root)
    safe_targets: dict[int, Path] = {}
    for idx, fd in enumerate(diffs):
        if security.is_sensitive_path(fd.rel) or any(
                security.is_sensitive_path(old) for old in fd.renames_from):
            return (False, "Not written: refused to write sensitive file: "
                    f"{fd.rel}")
        if Path(fd.rel).is_absolute():
            return False, f"Not written: unsafe path {fd.rel!r} (must be relative)."
        try:
            safe_targets[idx] = security.resolve_safe_path(root, fd.rel)
        except (security.SecurityError, OSError, ValueError) as exc:
            return False, f"Not written: unsafe path {fd.rel!r}: {exc}"
        for old in fd.renames_from:
            old_rel = old.strip()
            if not old_rel:
                continue
            if Path(old_rel).is_absolute():
                return False, f"Not written: unsafe path {old_rel!r} (must be relative)."
            try:
                security.resolve_safe_path(root, old_rel)
            except (security.SecurityError, OSError, ValueError) as exc:
                return False, f"Not written: unsafe path {old_rel!r}: {exc}"

    stale = []
    for idx, fd in enumerate(diffs):
        p = safe_targets[idx]
        if not p.exists():
            if fd.deleted:
                continue         # already gone: the deletion's end state
            if fd.sha and fd.sha != EMPTY_SHA:
                stale.append(f"{fd.rel} (deleted)")
            continue
        # Bytes already equal to what we would write: applied, not stale.
        if _digest(p) not in (fd.sha, _sha(fd.after_text)):
            stale.append(f"{fd.rel} (changed on disk)")
    if stale:
        return False, ("Not written: " + ", ".join(stale) +
                       ". Re-run the task to rebuild the review.")

    # Defense in depth: a partial hunk acceptance must never write a file
    # that no longer parses (apply_hunk_selection already checked; a caller
    # that set hunk_selection by hand gets the same refusal here).
    for idx, fd in enumerate(diffs):
        if getattr(fd, "hunk_selection", None) and not fd.deleted:
            from . import edits
            err = edits.syntax_error(safe_targets[idx], fd.after_text)
            if err:
                return False, (f"Not written: {fd.rel} would be left with "
                               f"invalid syntax ({err}). Accept the whole "
                               "file or adjust the hunk selection.")

    for idx, fd in enumerate(diffs):
        p = safe_targets[idx]
        if not fd.sha and p.exists() and _digest(p) != _sha(fd.after_text):
            return False, (f"Not written: {fd.rel} was created by something "
                           "else while this review was open. Re-run the task "
                           "to rebuild the review.")

    default_style = fileio.TextStyle()
    existed_at_check = {idx: safe_targets[idx].exists() for idx in range(len(diffs))}
    written = []
    deleted: list[str] = []
    for idx, fd in enumerate(diffs):
        if fd.removed_source:
            continue
        try:
            p = security.resolve_safe_path(root, fd.rel)
        except (security.SecurityError, OSError, ValueError) as exc:
            return False, f"Not written: unsafe path {fd.rel!r}: {exc}"
        if fd.deleted:
            # Sign-off on a deletion: back the exact bytes up, then unlink.
            if p.is_file() and p.resolve().is_relative_to(root.resolve()):
                if backup_dir is not None:
                    fileio.backup_copy(p, backup_dir / fd.rel)
                p.unlink()
                deleted.append(fd.rel)
            continue
        try:
            if p.exists():
                original, style = fileio.read(p, errors="replace")
            elif existed_at_check[idx]:
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
            if not dest.exists() and not dest.is_symlink():
                dest.parent.mkdir(parents=True, exist_ok=True)
                fileio.write(dest, original, style)
        p.parent.mkdir(parents=True, exist_ok=True)
        fileio.write(p, fd.after_text, style)
        written.append(fd.rel)

    removed = []
    rename_notes = []

    def _remove_rename_source(old: str, expect_sha: str) -> None:
        if not old or Path(old).is_absolute():
            return
        try:
            op = security.resolve_safe_path(root, old)
        except (security.SecurityError, OSError, ValueError):
            return
        if not op.is_file():
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

    msg = (f"{len(written)} file(s) written: " + ", ".join(written)
           if written else "0 file(s) written")
    if deleted:
        msg += "; deleted: " + ", ".join(deleted)
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


# When set (the GUI does), `run` hands the review to this callable instead of
# opening a second browser window. It receives the same inputs as keyword
# arguments, plus `diffs`, `task` and `clause_texts`, and returns the same
# (outcome, detail). A runner may drop entries from `diffs` in place before
# calling `writer()` (per-file accept); callers therefore re-read `diffs`
# afterwards.
RUNNER = None


def run(page: str, nonce: str, writer, port: int = 0, open_browser: bool = True,
        timeout: float = 1800.0, on_url=None, on_discard=None,
        diffs=None, task: str = "", clause_texts=None):
    """Serve the page on 127.0.0.1 and block until Apply/Discard or `timeout`
    seconds. `writer()` -> (ok, detail) is called on Apply; `on_discard()` on
    Discard, for flows whose files are already on disk (`new --review`).
    Returns (outcome, detail); outcome is "applied", "cancelled" or "timeout".
    `diffs` also backs the standalone page's per-file/per-hunk selection (the
    Apply POST filters this list before calling `writer`); `task` and
    `clause_texts` are only used by an installed `RUNNER`."""
    if RUNNER is not None:
        return RUNNER(page=page, nonce=nonce, writer=writer, timeout=timeout,
                      on_discard=on_discard, diffs=diffs, task=task,
                      clause_texts=clause_texts)
    # A session with its own queue listener (TUI/RPC/daemon) routes the review
    # through that session's ReviewQueue instead of opening a second browser.
    sess = session_ctx.current_session()
    queue = getattr(sess, "review_queue", None) if sess is not None else None
    if queue is not None and (
            (callable(getattr(queue, "has_listener", None))
             and queue.has_listener())
            or callable(getattr(queue, "listener", None))):
        staged = queue.stage(
            task=task, diffs=diffs or [], writer=writer,
            on_discard=on_discard, partial_ok=on_discard is None,
            clause_texts=clause_texts, ttl_s=timeout)
        deadline = time.time() + float(timeout or 0.0)
        while not staged.done.wait(0.5):
            if sess is not None and sess.cancel_event.is_set():
                queue.cancel_all("session cancelled")
                break
            if deadline and time.time() > deadline:
                queue.expire_stale()
                if not staged.done.is_set():
                    queue.expire_stale(now=staged.expires_at + 0.001)
                break
        return (staged.outcome or "timeout",
                staged.detail
                or "Review window timed out; nothing was written.")
    srv = LocalHTTPServer(("127.0.0.1", port), _Handler)
    thread = None
    registry = getattr(sess, "resource_registry", None) if sess is not None else None
    resource_id = None
    try:
        srv.page = page
        srv.nonce = nonce
        srv.writer = writer
        srv.diffs = diffs
        srv.on_discard = on_discard
        srv.lock = threading.Lock()
        srv.done = threading.Event()
        srv.outcome = ("timeout", "Review window timed out; nothing was written.")
        url = f"http://127.0.0.1:{srv.server_address[1]}/"

        def _shutdown_review_server() -> None:
            """Release the loopback server; unblock a waiting review first."""
            try:
                with srv.lock:
                    if not srv.done.is_set():
                        srv.outcome = (
                            "cancelled",
                            "Session cancelled; nothing was written.")
                        srv.done.set()
            except Exception:
                pass
            try:
                srv.shutdown()
            except Exception:
                pass
            try:
                srv.server_close()
            except Exception:
                pass

        if registry is not None:
            try:
                resource_id = registry.register(
                    "review_server", url, _shutdown_review_server)
            except Exception:
                resource_id = None
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
        return srv.outcome
    finally:
        if resource_id is not None and registry is not None:
            try:
                registry.unregister(resource_id)
            except Exception:
                pass
        srv.shutdown()
        srv.server_close()
        if thread is not None:
            thread.join(timeout=2)


def new_nonce() -> str:
    """A fresh token authorising one review's Apply request."""
    return secrets.token_urlsafe(16)
