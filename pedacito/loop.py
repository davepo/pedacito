"""The autonomous outer loop (`pedacito auto`): a Ralph loop over a task list.

``ask`` and ``chat`` run ONE task through the inner loop (gather -> answer ->
apply). ``auto`` wraps that in an outer loop: pick the next unchecked task in
the plan file, run it with a FRESH agent/context, optionally verify, tick it
off, repeat. State lives in the plan file (a plain markdown checklist a human
can edit mid-run), not in any conversation.

Usage
-----
    from . import loop

    titles = loop.generate_tasks(spec, client, cfg)       # LLM task breakdown
    loop.write_plan(path, goal, titles)                   # -> markdown checklist
    code = loop.run_loop(root, cfg, client, idx, path,
                         apply=True, verify="pytest -q")
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import checkpoints as checkpoints_mod
from . import checkpoint_gc
from . import diagnostics, events as events_mod, hooks as hooks_mod
from . import fileio
from . import safety, workspace
from . import index as index_mod
from . import instructions as instructions_mod
from . import proc as proc_mod
from . import prompts as prompts_mod
from . import ratchet as ratchet_mod
from . import review as review_mod
from . import security as security_mod
from . import test_classify
from . import testsel
from . import todo as todo_mod
from . import trajectories
from .edits import Result, parse_edits
from .llm import CancelledError, CircuitOpenError, LLMError


# ---------------------------------------------------------------- task journal
def _journal_root(journal) -> Path | None:
    """The project root a journal lives under (``<root>/.pedacito/journal/``),
    or None when it cannot be derived."""
    try:
        p = Path(journal.path)
        if not p.is_absolute():
            return None
        # <root>/.pedacito/journal/<stem>.jsonl  ->  up three levels
        parent = p
        for _ in range(3):
            parent = parent.parent
        return parent
    except Exception:
        return None


def _journal_surface_error(root, message: str) -> None:
    """Surface a journal disk error to the run log and stderr; never raise."""
    try:
        if root is not None:
            workspace.log_event(Path(root), message, "ERROR")
    except Exception:
        pass
    try:
        print(f"  [task journal] {message}", file=sys.stderr)
    except Exception:
        pass


def _journal_open(root, cfg, task: str):
    """Best-effort `TaskJournal` for this task, stored on ``cfg._task_journal``
    (and on the bound session, so subprocess helpers can append too). Returns
    the journal or None when the root is absent / read-only. A lost durability
    guarantee is surfaced, never swallowed."""
    try:
        from . import journal as journal_mod
        sid = ""
        try:
            from . import session_ctx
            sess = session_ctx.current_session()
            if sess is not None:
                sid = str(sess.session_id)
        except Exception:
            pass
        journal = journal_mod.TaskJournal(root, task, session_id=sid)
        journal.append("task_start", task=(task or "")[:400])
        if getattr(journal, "durable", True) is False:
            _journal_surface_error(root, journal.error)
        try:
            from . import session_ctx
            sess = session_ctx.current_session()
            if sess is not None:
                sess._task_journal = journal
        except Exception:
            pass
        return journal
    except Exception as exc:
        _journal_surface_error(root, f"could not open task journal: {exc}")
        return None


def _journal_append(cfg, kind: str, **data) -> None:
    """Append one journal record, surfacing (not swallowing) durability loss."""
    journal = getattr(cfg, "_task_journal", None)
    if journal is None:
        return
    try:
        journal.append(kind, **data)
    except Exception as exc:
        _journal_surface_error(_journal_root(journal),
                               f"journal append failed: {exc}")
        return
    if getattr(journal, "durable", True) is False:
        _journal_surface_error(
            _journal_root(journal),
            f"task journal lost its durability guarantee: {journal.error}")


def _journal_close(cfg, status: str = "completed") -> None:
    """Close (append task_end with `status`) and detach the task journal, best
    effort. A non-durable journal is surfaced, never silently claimed as a
    durable terminal outcome."""
    journal = getattr(cfg, "_task_journal", None)
    if journal is not None:
        try:
            journal.close(status=status)
            if getattr(journal, "durable", True) is False:
                _journal_surface_error(
                    _journal_root(journal),
                    f"task journal is not durable; terminal outcome "
                    f"'{status}' is unrecoverable: {journal.error}")
        except Exception as exc:
            _journal_surface_error(_journal_root(journal),
                                   f"could not close task journal: {exc}")
    try:
        cfg._task_journal = None
    except Exception:
        pass
    try:
        from . import session_ctx
        sess = session_ctx.current_session()
        if sess is not None:
            sess._task_journal = None
    except Exception:
        pass


def _recover_atomic(root, cfg) -> None:
    """Roll back any atomic edit transaction a previous run left mid-commit.

    Runs before this run snapshots or writes anything, so the pre-run
    checkpoint captures the rolled-back state. Best effort and surfaced: a
    leftover transaction that cannot be recovered must not stop the run, but
    it is reported."""
    try:
        from . import journal as journal_mod
        restored = journal_mod.recover_atomic(root)
        if restored:
            print(f"  [recovered {len(restored)} file(s) from an interrupted "
                  f"atomic edit transaction]", file=sys.stderr)
    except Exception as exc:
        try:
            print(f"  [could not recover a leftover atomic edit transaction: "
                  f"{exc}]", file=sys.stderr)
        except Exception:
            pass


# A task line in the plan file: `- [ ] do the thing` / `- [x] done thing`.
# `*` bullets are accepted too. Group 2 is the box character, group 3 the text.
_TASK_LINE = re.compile(r"^(\s*[-*]\s*)\[([ xX])\]\s*(.*)$")

PLAN_HEADER = "# Plan\n\n"

_NOOP_PHRASES = (
    "no change", "nothing to", "nothing needs", "already ", "no edit",
    "no code change", "no code edit", "leave as", "as-is",
)


def _cancel_requested(cfg, client=None) -> bool:
    """True when the GUI/TUI cancellation token for this run is set.

    Accepts a threading.Event, any object with `is_set()`, or a callable
    predicate. Checked between tasks so a Stop never starts another worker
    session, even when the abort landed between LLM calls.
    """
    sources = [getattr(cfg, "_cancel_event", None)]
    if client is not None:
        sources.append(getattr(client, "cancel_event", None))
    for source in sources:
        if source is None:
            continue
        checker = getattr(source, "is_set", None)
        try:
            if callable(checker):
                if checker():
                    return True
            elif callable(source):
                if source():
                    return True
            elif source:
                return True
        except Exception:
            continue
    return False


def _emit_run_cancelled(root, cfg, stage: str, **fields) -> None:
    """Record a user Stop in the event log (GUI activity + TUI status)."""
    try:
        events_mod.get(root, cfg).emit("run_cancelled", stage=stage, **fields)
    except Exception:
        pass


def _is_noop(reply: str) -> bool:
    """True when a zero-edit reply says there is nothing to do (the work is
    already complete). Only consulted when NO edit applied."""
    low = (reply or "").lower()
    return any(p in low for p in _NOOP_PHRASES)


def _structured_files(reply: str) -> list[str]:
    """The normalised file paths a structured-mode reply touches (from its
    intents and edits); [] when it is not a strict structured object. Used for
    the pre_edit hook and test baselines without a legacy `parse_edits` pass."""
    try:
        from .edits import parse_structured_reply
        intents, edits = parse_structured_reply(reply or "")
    except Exception:
        return []
    return sorted({e.file for e in edits if e.file}
                  | {i.file for i in intents if i.file})


def _normalized_reply(reply: str) -> str:
    """Lowercase with apostrophe styles folded to ASCII, so refusal/incomplete
    phrasing matches whatever quoting came back. Local servers sometimes
    double-encode UTF-8, turning U+2019 into "â" plus stray characters."""
    return ((reply or "").lower()
            .replace("\u2019", "'")
            .replace("â\u20ac\u2122", "'")
            .replace("â\x80\x99", "'")
            .replace("â", "'"))


# A reply that OPENS by declaring it made no edit ("No edit.", "No edits were
# made:", "I did not edit any file"). Narrow on purpose: the caller DISCARDS
# the edit blocks of such a reply. A qualified sentence ("No changes were
# needed in main.js; I fixed index.html instead.") is not a refusal.
_EDIT_REFUSAL_HEAD_RE = re.compile(
    r"^(?:[#*\-\s]*)"
    r"(?:no\s+(?:edits?|changes?|code\s+changes?)"
    r"(?:\s+were\s+(?:made|applied|needed|necessary|required))?[\s]*[.:!\r\n]"
    r"|i\s+(?:did\s+not|didn't|won't|will\s+not)\s+"
    r"(?:\w+\s+){0,2}?(?:edit|apply|make|emit|write)\b)")


def _edit_refusal_head(reply: str) -> str:
    """The matched refusal phrase when a reply opens by declaring it made no
    edit; "" otherwise. Checked against the normalized first 200 chars."""
    m = _EDIT_REFUSAL_HEAD_RE.match(
        _normalized_reply(reply or "").strip()[:200])
    return m.group(0).strip() if m else ""


# Zero-edit replies that DECLARE THE WORK NOT DONE ("I can't safely emit the
# move blocks yet ... to continue, the files must be read"), as opposed to
# _is_noop's "nothing to do". Matched on _normalized_reply.
_INCOMPLETE_VERBS = (r"(?:emit|apply|make|produce|write|complete|finish|move|"
                     r"create|edit|proceed|continue|do)\b")
_INCOMPLETE_RES = (
    re.compile(r"\bcan(?:not|'t)\s+(?:\w+\s+){0,3}?" + _INCOMPLETE_VERBS),
    re.compile(r"\bunable to\s+(?:\w+\s+){0,3}?" + _INCOMPLETE_VERBS),
    re.compile(r"\bto continue\b"),
    re.compile(r"\bremaining work\b"),
    re.compile(r"\bmust(?:\s+first)?\s+be\s+(?:read|done|made)\b"),
    re.compile(r"\bhave(?:n't| not)\s+(?:yet\s+)?"
               r"(?:read|made|applied|done|finished|emitted)\b"),
    re.compile(r"\bwon't guess\b"),
    re.compile(r"\bnot yet been\s+"
               r"(?:done|made|applied|performed|completed|moved)\b"),
    re.compile(r"\bwill need to be\b"),
)


def _declares_incomplete(reply: str) -> bool:
    """True when a zero-edit reply says the task is NOT done and could not be
    completed now. Such a reply is an incomplete task, never a no-op: the loop
    retries it once and fails it on a second deferral. Repair tasks are exempt
    at the call site (refusing to edit is a sanctioned repair outcome)."""
    low = _normalized_reply(reply)
    return any(r.search(low) for r in _INCOMPLETE_RES)


# Outcome-negative patterns for verification reports. Deliberately NOT
# matched: a bare "not ... confirmed" or a be-verb form ("is not yet
# confirmed"): honest reports list what they could not confirm as diligence
# while still confirming the asked-about symptoms. Inability verbs and the
# verify family are the unambiguous signals.
_UNVERIFIED_RES = (
    # Inability, tolerating up to 4 intervening words and a hyphenated verb:
    # "could not visually verify", "can't produce a live visual verification".
    # Both ' and \u2019 are accepted.
    re.compile(r"\b(?:can(?:not|['\u2019]t)|could(?: not|n['\u2019]t)|"
               r"unable to|failed to|impossible to)\s+"
               r"(?:[\w-]+\s+){0,4}?[\w-]*(?:verif|confirm)"),
    # The verdict itself: "verification failed", "the verify step fails".
    re.compile(r"\bverif\w*\s+(?:failed|fails)\b"),
    # "verification did not pass/succeed".
    re.compile(r"\bverif\w*\s+(?:did|does)\s*(?:not|n['\u2019]t)\s+"
               r"(?:pass|succeed|verify|confirm|hold)"),
    # "not visually verified", "not directly verified" -- the verify family
    # only; a bare "not confirmed" is too common in honest caveat prose.
    re.compile(r"\bnot\s+(?:\w+\s+){0,2}?verif(?:y|ied|ies|ying)\b"),
    # "no live run/verification/visual evidence".
    re.compile(r"\bno\s+live\s+(?:run|verification|visual|evidence)"),
    # "Still unknown: whether the sprites render": the report names the
    # task's own claim as unknown. The `whether` window with a 40-char leash
    # separates that from a passing caveat ("Still unknown: direct
    # confirmation from index.html ..."). Pessimistic on purpose.
    re.compile(r"\bstill unknown\b[^.\n]{0,40}?\bwhether\b"),
)


# A PASSED verdict is phrase-matched against the report with prior-attempt
# sentences stripped: retry flows narrate earlier failures ("the previous
# failure ... could not confirm X"), which must not read as a present
# confession. A sentence is historical when it attributes something to a
# prior attempt or continues a dropped sentence as a pronoun ("It did not
# count ... so it could not confirm ..."). Present-tense first-person
# confessions carry no attribution and survive.
_PRIOR_ATTEMPT_RE = re.compile(
    r"\b(?:"
    r"(?:earlier|previous|prior|first|initial|last|original)\s+"
    r"(?:\w+\s+){0,2}?"
    r"(?:attempt|attempts|try|eval|evals|run|runs|session|sessions|"
    r"check|checks|verification|failure|failures|report|reports|"
    r"response|round|measurement|measurements|scan|scans|profile|profiles)"
    r"|previously|initially|originally|before this|in the past|at first"
    r"|attempt(?:s)? (?:1|one)"
    r")\b",
    re.IGNORECASE)
_PRONOUN_START_RE = re.compile(
    r"^[\s\-*\u2022\d.)]*\b(it|its|it['\u2019]s|this|that|these|those|"
    r"they|them|their|there|the same|said|which|both)\b", re.IGNORECASE)


def _strip_prior_attempt_prose(reply: str) -> str:
    """The report minus sentences describing PRIOR attempts.

    Deliberately over-eager on pronoun continuations: dropping a harmless
    sentence costs nothing (the matcher only looks for negatives). Sentences
    are rejoined with newlines so pruning cannot glue two kept fragments into
    an accidental confession phrase."""
    sentences = re.split(r"(?<=[.!?\n])\s+", reply or "")
    kept, prev_dropped = [], False
    for s in sentences:
        if not s.strip():
            continue
        historical = bool(_PRIOR_ATTEMPT_RE.search(s))
        if not historical and prev_dropped and _PRONOUN_START_RE.match(s):
            historical = True          # "It did not count ... could not ..."
        if historical:
            prev_dropped = True
            continue
        prev_dropped = False
        kept.append(s)
    return "\n".join(kept)


# Substrings not covered by _UNVERIFIED_RES.
_UNVERIFIED_PHRASES = (
    "can't launch the browser", "cannot launch the browser",
    "could not launch the browser", "unable to launch the browser",
    "can't run the game", "cannot run the game", "could not run the game",
    "without running the", "without a live run",
    "did not run the", "didn't run the", "never ran the",
)


def _reply_says_unverified(reply: str) -> bool:
    """True when the report says the verification did NOT succeed: it confesses
    it could not perform the check ("I can't launch the browser", "could not
    visually verify") or reports a negative outcome ("Verification failed").
    Only consulted for tasks whose TITLE asks to verify/confirm/observe;
    `_evidence_failure` pairs it with the objective tool-use signal."""
    low = (reply or "").lower()
    if any(p in low for p in _UNVERIFIED_PHRASES):
        return True
    return any(r.search(low) for r in _UNVERIFIED_RES)


# Verification reports must END with a machine-readable verdict line (the
# brief is appended in run_investigation_task) and _evidence_failure reads it
# first. Last match wins: a report may quote the brief's format before
# stating its own verdict. The prefix class tolerates markdown bold/bullets
# ("**VERDICT: FAILED**", "- VERDICT: PASSED").
_VERDICT_RE = re.compile(
    r"(?im)^[*\s-]{0,4}VERDICT:\s*(PASSED|FAILED)\b:?\s*(.*?)\s*$")


def _verdict_of(reply: str) -> tuple[str, str]:
    """The report's last verdict line as (state, reason): ("FAILED", "the
    frog never rendered"), ("PASSED", ""), or ("", "") when absent."""
    found = _VERDICT_RE.findall(reply or "")
    if not found:
        return "", ""
    state, reason = found[-1]
    return state.upper(), reason.strip().strip("*").strip()


# Words that make a task title a verification checkpoint: its deliverable is
# evidence that the expected behavior was observed, not a code change.
# `check` is matched as a whole word so "checksum" does not qualify.
_VERIFY_WORDS = re.compile(
    r"\b(?:verif|confirm|observ|validat)\w*|\bcheck(?:s|ed|ing)?\b")


def _is_verification_task(task: str) -> bool:
    """True for a task whose title asks to verify/confirm/observe behavior."""
    return bool(_VERIFY_WORDS.search((task or "").lower()))


# Plan items that only re-run checks the loop already performs after every
# edit task. Live runs on static sites kept ending the plan with "Run the
# page and verify ..." / "Verify the page renders": the worker finds nothing
# to edit, and the zero-edit/evidence gates then fail the item ([!]) even
# though the workspace is already green. When the disk evidence below
# confirms it, such an item is ticked done without a worker session or a
# diff. Narrow leading phrases on purpose: "Run the GAME and verify ..."
# stays a live checkpoint with its full evidence gate.
_NOOP_VERIFY_PREFIXES = (
    "verify ",
    "run the page and verify",
    "check that ",
    "test that ",
)


def _is_noop_verify_item(task: str) -> bool:
    """True for a checklist item that only re-checks what the loop's
    per-task verification already covers."""
    return (task or "").strip().lower().startswith(_NOOP_VERIFY_PREFIXES)


def _noop_verify_already_green(root: Path, task: str, verify: str,
                               cfg=None) -> bool:
    """Objective disk evidence that a no-op verification item is already
    satisfied: webcheck finds nothing, the configured verify command (the
    tests) passes, and no requirement of the title is still owed on disk.
    At least one real gate must have run and passed: a tree with no web
    files and no verify command proves nothing, so the item keeps its
    normal evidence gate there."""
    from . import verify as verify_mod, webcheck
    try:
        if webcheck.run(root):
            return False
        htmls, jss, _csss = webcheck._collect(root)
    except Exception:
        return False
    gate_ran = bool(htmls or jss)
    if verify:
        ok, _out = run_verify(root, verify, cfg=cfg)
        if not ok:
            return False
        gate_ran = True
    if not gate_ran:
        return False
    try:
        return not verify_mod.unmet_requirements(root, task, [])
    except Exception:
        return False


def _task_title(task: str) -> str:
    """The title line out of a possibly-assembled worker prompt (ORIGINAL USER
    GOAL block + CURRENT TASK + notes). Title-keyed decisions must not read
    the goal block: its spec prose can contain words that change the verdict.
    With no goal block the title is the first line."""
    body = task or ""
    marker = "\nCURRENT TASK\n"
    if marker in body:
        body = body.split(marker, 1)[1]
    return body.split("\n", 1)[0]


def checkpoint_mode(cfg) -> str:
    """`auto` for a step owned by the outer loop, else `task`."""
    return "auto" if getattr(cfg, "_auto_step", 0) else "task"


def _transport_outage(e: Exception) -> bool:
    """True when an LLMError is a model-server reachability problem (a dropped
    connection, a timeout, a refused connection) rather than a code or content
    defect. Such failures must never roll back valid applied edits."""
    low = str(e or "").lower()
    return any(p in low for p in (
        "could not connect", "connection", "did not answer", "timed out",
        "unreachable", "connection refused", "connect", "failed to connect",
    ))


def _checkpoint_task(root: Path, cfg, task: str, phase: str) -> dict | None:
    """Record the pre-run baseline or the post-run checkpoint for one task.

    Checkpointing is best-effort: a full disk or a racing process must never
    turn a working task into a failed one.  Dry runs leave the tree untouched;
    a review task records its pre-sign-off baseline here and its accepted
    write in `_review`. The checkpoint is bound to the task's WAL UUID so
    recovery can verify the exact association before restoring.
    """
    try:
        mode = checkpoint_mode(cfg)
        title = _task_title(task).strip()[:200] or "task"
        wal_id = ""
        journal = getattr(cfg, "_task_journal", None)
        if journal is not None:
            wal_id = str(getattr(journal, "wal_id", "") or "")
        if phase == "pre":
            entry = checkpoints_mod.ensure_pre_run_checkpoint(
                root, f"baseline before {title}", mode=mode, wal_id=wal_id)
        else:
            entry = checkpoints_mod.create_checkpoint(
                root, title, mode=mode, wal_id=wal_id)
        if isinstance(entry, dict) and entry.get("id"):
            _journal_append(cfg, "checkpoint", checkpoint_id=str(entry["id"]),
                            phase=phase, mode=mode)
        return entry
    except Exception as e:
        try:
            workspace.log_event(root, f"checkpoint skipped: {e}", "WARN")
        except Exception:
            pass
        return None


# Clipping budget for investigation reports carried between tasks. Caveats can
# sit anywhere in a report, so only past the budget are head AND tail kept.
_REPORT_BUDGET = 8000


def _clip_report(reply: str, budget: int = _REPORT_BUDGET) -> str:
    """Clip a carried-over investigation report, keeping head and tail."""
    text = (reply or "").strip()
    if len(text) <= budget:
        return text
    head = int(budget * 0.6)
    tail = budget - head
    return (text[:head]
            + "\n[...report clipped to fit; full text in the session log...]\n"
            + text[-tail:])


# Tasks whose deliverable includes live evidence are told up front that the
# tools exist only in the lookup phase (models otherwise discover this in the
# answer phase, too late).
_EVIDENCE_WORDS = re.compile(
    r"\b(run|runs|running|launch|launches|verify|verifies|verification|"
    r"confirm|confirms|observe|observes|observing|reproduce|reproduces|"
    r"screenshot|screenshots|browser|visually|render|renders|rendering|"
    r"game)\b")


def _evidence_hint(task: str) -> str:
    """A task-text addendum for live-evidence tasks; empty for code tasks."""
    low = (task or "").lower()
    if not _EVIDENCE_WORDS.search(low):
        return ""
    hint = ("\n\nEVIDENCE: this task is about live behavior. The `run`, "
            "`screenshot`, and `browser` tools exist ONLY in the lookup "
            "phase; the final answer phase has no tools. Launch the app and "
            "capture the evidence while gathering, before you answer -- a "
            "verification postponed to the answer phase cannot happen, and "
            "reporting the task done without that evidence fails it.")
    if _VISUAL_TITLE.search(low):
        # The gate below demands an OBSERVATION for these titles, so say so
        # up front: models otherwise start the server with `run`, see it come
        # up, and report the page verified.
        hint += (" This title asks what the page/canvas SHOWED, so a `run` "
                 "that only starts the process is not the evidence: open the "
                 "URL with `browser` (or `screenshot` a desktop window) and "
                 "quote what it returned -- page text, canvas size, console "
                 "or network errors.")
    return hint


# A steps_log entry looks like "4. browser python main.py [wait 3]  (reason)".
# Anchor on the ACTION token: a `read` step whose reason mentions "run the
# game" is not live evidence.
_LIVE_STEP = re.compile(r"^\d+\.\s+(?:run(?:_test)?|browser|screenshot)\b")

# Title words that promise live observation. Narrower than _EVIDENCE_WORDS:
# paired with a verify word, such titles need evidence from the RUNNING app.
# "Check the render path" stays a source-reading task.
_LIVE_TITLE = re.compile(
    r"\b(run|runs|running|launch|launches|start|starts|game|app|browser|"
    r"visually|screenshot|screenshots|observe|observes|observing|"
    r"reproduce|reproduces|live)\b")


def _needs_live_evidence(task: str) -> bool:
    """True when the title promises verification by live observation."""
    low = (task or "").lower()
    return bool(_VERIFY_WORDS.search(low) and _LIVE_TITLE.search(low))


def _used_live_evidence(steps) -> bool:
    """True when some lookup step actually drove the app (run/browser/
    screenshot) -- the objective signal, immune to how the report words it."""
    return any(_LIVE_STEP.match(s or "") for s in (steps or ()))


# Steps that OBSERVED the app rather than merely executing it. Starting a
# server with `run` proves the process came up; it says nothing about what the
# page or canvas showed.
_VISUAL_STEP = re.compile(r"^\d+\.\s+(?:browser|screenshot)\b")
_VISUAL_ACTIONS = ("browser", "screenshot")

# Title words that ask about the rendered artifact itself. Only consulted for
# titles that already promise live observation (_needs_live_evidence), so
# "verify the template renders the report" -- no run/launch/browser promise --
# is unaffected. Deliberately NOT here: "render", "display", "screen", "html"
# and "css", which a terminal, a report or a source file can also satisfy.
_VISUAL_TITLE = re.compile(
    r"\b(browser|page|pages|webpage|ui|visual|visually|screenshot|"
    r"screenshots|dom|canvas|viewport|layout)\b")


def _needs_visual_evidence(task: str) -> bool:
    """True when the title asks what the running UI SHOWED, so a `run` that
    only launched the process is not evidence."""
    return _needs_live_evidence(task) and bool(
        _VISUAL_TITLE.search((task or "").lower()))


def _used_visual_evidence(steps) -> bool:
    """True when a browser/screenshot step appears in the steps log."""
    return any(_VISUAL_STEP.match(s or "") for s in (steps or ()))


def _observed_the_app(tr) -> bool:
    """True when a browser/screenshot step produced an observation. Prefers
    the action kinds the Agent recorded; falls back to the steps log for
    callers and tests predating that field."""
    kinds = getattr(tr, "live_kinds", None)
    if kinds is None:
        return _used_visual_evidence(getattr(tr, "steps", ()))
    return bool(set(_VISUAL_ACTIONS) & set(kinds))


def _live_kinds_of(agent):
    """The agent's successful live-action kinds as a tuple, or None when it
    does not track them (a test double): the gate then reads the steps log."""
    kinds = getattr(agent, "live_evidence_kinds", None)
    return None if kinds is None else tuple(sorted(kinds))


def _evidence_failure(task: str, tr) -> str:
    """A reason string when a verification checkpoint lacks evidence, else "".

    Pessimistic on ties. Checks, in order: the report is a real report (not
    the steps-log stand-in); its own VERDICT line is not FAILED; the prose
    does not say the verification did not succeed; for titles promising live
    observation, a run/browser/screenshot step actually happened; and, for
    titles asking what the UI SHOWED, that step was a browser/screenshot
    observation rather than a command that only launched the process."""
    if not _is_verification_task(task):
        return ""
    if (tr.reply or "").startswith("No prose report was produced"):
        return ("its report is empty (the answer phase produced no prose "
                "even after a retry); a steps log is not a verification")
    if _looks_like_unexecuted_promise(tr.reply):
        return ("its report only promises to run the check; the answer phase "
                "has no tools, so no observation was reported")
    state, reason = _verdict_of(tr.reply)
    if state == "FAILED":
        return ("its own verdict line says VERDICT: FAILED"
                + (f": {reason[:200]}" if reason else ""))
    # A clean PASSED verdict is phrase-matched with prior-attempt narration
    # stripped; verdict-less reports keep the full-text match.
    if _reply_says_unverified(tr.reply if state != "PASSED"
                              else _strip_prior_attempt_prose(tr.reply)):
        return ("its report says the verification did not succeed "
                "(could not be performed, or was performed and failed)")
    live_ok = getattr(tr, "live_ok", None)
    if live_ok is None:                 # callers/tests predating the flag
        live_ok = _used_live_evidence(getattr(tr, "steps", ()))
    if _needs_live_evidence(task) and not live_ok:
        return ("it never successfully drove the app (no run/browser/"
                "screenshot step produced an observation); reading source "
                "or a failed launch attempt is not live evidence")
    if live_ok and _needs_visual_evidence(task) and not _observed_the_app(tr):
        return ("it never OBSERVED the app: the title asks what the page/"
                "canvas showed, and only a `browser` or `screenshot` step "
                "does that -- a `run` that started the process (a server, a "
                "build, a test) proves it came up, not what it displayed")
    return ""


def _evidence_miss_is_procedural(task: str, tr) -> bool:
    """True when the missing evidence says HOW the check was run (nobody drove
    the app, or nobody opened it) rather than what the product did. Such a
    miss re-runs the check with the toolset spelled out; it must never send a
    repair task after a product nobody measured."""
    return (not _used_live_evidence(getattr(tr, "steps", ()))
            or (_needs_visual_evidence(task) and not _observed_the_app(tr)))


# Titles starting with one of these verbs are checkpoints answered in prose,
# not edit tasks (a successful "reproduce the failure" must not count as a
# failure for lacking an applied edit).
_INVESTIGATE_START = (
    "reproduce", "investigate", "diagnose", "analyze", "analyse", "inspect",
    "trace", "locate", "find", "confirm", "check", "verify", "review",
    "profile", "explain", "identify", "determine", "observe", "test",
    "run",
    "audit", "assess", "evaluate", "report", "summarize", "summarise",
    "list", "enumerate",
)
# Any of these anywhere in the title keeps it on the edit path ("Reproduce and
# fix the crash", "Run the build and update the docs"). The right-hand bound
# excludes a hyphen on purpose: the first half of a compound NOUN ("add-on",
# "create-react-app", "wire-frame", "patch-notes", "fix-up") names a thing, it
# does not ask for an edit, and a verification title built around one must
# keep its evidence gate instead of being routed to the edit path. A LEADING
# hyphen still counts ("re-add the null check"), so a title that really wants
# an edit is never misrouted the other way.
_EDIT_WORDS_RE = re.compile(r"(?<!\w)(?:" + "|".join((
    "fix", "implement", "add", "remove", "delete", "rename", "move",
    "update", "change", "refactor", "write", "replace", "create", "patch",
    "correct", "repair", "build", "wire", "integrate", "migrate",
)) + r")(?![\w-])")


def _is_investigation_task(task: str) -> bool:
    """True for a task whose deliverable is evidence, not a code change: a
    leading investigation verb and no edit verb anywhere in the title."""
    low = (task or "").strip().lower()
    first = re.match(r"[a-z]+", low)
    if not first or not first.group(0).startswith(_INVESTIGATE_START):
        return False
    return not _EDIT_WORDS_RE.search(low)


def _lint_plan_titles(root: Path, plan_path: Path) -> None:
    """Advisory check at loop start: a title bundling 2+ distinct edit verbs
    usually means several changes in one task, which blurs verification and
    rollback. Warns on stderr and in the run log; never edits the plan."""
    for ln in read_plan(plan_path):
        m = _is_unchecked(ln)
        if not m:
            continue
        title = m.group(3).strip()
        verbs = sorted(set(_EDIT_WORDS_RE.findall(title.lower())))
        if len(verbs) >= 2:
            print(f"  [plan title bundles {len(verbs)} changes "
                  f"({', '.join(verbs)}): \"{title[:70]}\" -- one change per "
                  f"title keeps verification and rollback honest]",
                  file=sys.stderr)
            workspace.log_event(
                root, f"plan title bundles edits ({', '.join(verbs)}): "
                      f"{title[:80]}", "WARN")


# ------------------------------------------- Task-vs-Auto suitability ------
# A deterministic, zero-LLM check that keeps a one-pass edit out of the
# multi-step Auto loop. The CLI/TUI ask before converting; the GUI shows a
# live banner and a pre-run choice. Deliberately conservative: only obvious
# single-pass work is flagged, and an explicit planning/architecture ask
# always keeps Auto.

_MODE_OVERRIDE_PHRASES = ("multi-step plan", "multi step plan",
                          "create a plan", "from scratch")

# Localized edit verbs: a prompt built around these is one cohesive pass,
# not a plan-and-verify loop. Bare "add" is too broad (scaffolding asks use
# it constantly), so only the content-level phrases count.
_MODE_EDIT_VERBS_RE = re.compile(
    r"(?<!\w)(?:remove|delete|change|replace|stack|move|rename|tweak|fix|"
    r"hide|update|restyle|convert)(?![\w-])"
    r"|(?<!\w)add\s+(?:a|an|the\s+)?(?:heading|link)(?![\w-])")

# Words naming a module/subsystem of a NEW application. Two or more next to
# a build verb read as an architecture ask; four or more override the nudge
# outright.
_MODE_MODULE_WORDS = (
    "backend", "frontend", "middleware", "database", "migration", "auth",
    "authentication", "cli", "api", "endpoint", "service", "worker",
    "scheduler", "dashboard", "plugin", "microservice", "module",
    "test suite", "pytest suite", "unittest suite",
)
_MODE_BUILD_VERBS_RE = re.compile(
    r"(?<!\w)(?:build|create|scaffold|implement|develop|architect|design|"
    r"set\s+up|write)(?![\w-])")

# A filename-shaped token: `index.html`, `src/app.py`, `input.css`.
_MODE_FILE_TOKEN_RE = re.compile(r"[A-Za-z0-9_./\\-]+\.[A-Za-z0-9]{1,6}")
_MODE_FILE_EXTS = {
    ".py", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".html", ".htm",
    ".css", ".scss", ".sass", ".less", ".vue", ".svelte", ".json", ".toml",
    ".yaml", ".yml", ".md", ".sql", ".sh", ".rb", ".go", ".rs", ".java",
    ".c", ".cpp", ".h", ".txt",
}
# Source files counted for the tiny-project rule (docs/config alone do not
# make a project "big").
_MODE_SOURCE_EXTS = {
    ".py", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".html", ".htm",
    ".css", ".scss", ".sass", ".less", ".vue", ".svelte", ".rb", ".go",
    ".rs", ".java", ".c", ".cpp", ".h", ".sh",
}
_MODE_CONTENT_WORDS_RE = re.compile(
    r"(?<!\w)(?:content|styling|style|styles|css|color|colors|colour|"
    r"colours|font|fonts|heading|headings|layout|wording|copy|margin|"
    r"padding|palette|theme|spacing|typography)(?![\w-])")
_MODE_SCAFFOLD_RE = re.compile(
    r"(?<!\w)(?:scaffold|backend|test\s+suite|pytest|unittest|boilerplate|"
    r"new\s+project|api\s+server|database\s+schema|from\s+scratch)(?![\w-])")


def _mode_check_files(root: Path | None) -> list[str]:
    """Non-ignored source files under root (relative paths, bounded)."""
    if root is None:
        return []
    root = Path(root)
    if not root.is_dir():
        return []
    from . import spec as spec_mod      # lazy: shared skip-list, no cycle
    out: list[str] = []
    stack = [root]
    while stack and len(out) < 400:
        d = stack.pop()
        try:
            entries = sorted(d.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        for e in entries:
            if e.is_dir():
                if not spec_mod._skip_dir(e.name):
                    stack.append(e)
            elif e.is_file() and e.suffix.lower() in _MODE_SOURCE_EXTS:
                out.append(e.relative_to(root).as_posix())
                if len(out) >= 400:
                    break
    return sorted(out)


def _mode_file_mentions(text: str) -> list[str]:
    """Distinct filename-shaped tokens with a known source/doc extension."""
    seen: list[str] = []
    for m in _MODE_FILE_TOKEN_RE.finditer(text):
        tok = m.group(0).strip("./\\")
        if "." not in tok:
            continue
        ext = "." + tok.rsplit(".", 1)[-1].lower()
        if ext in _MODE_FILE_EXTS and tok not in seen:
            seen.append(tok)
    return seen


def _mode_module_hits(low: str) -> set[str]:
    hits: set[str] = set()
    for word in _MODE_MODULE_WORDS:
        if " " in word:
            if word in low:
                hits.add(word)
        elif re.search(rf"(?<!\w){re.escape(word)}(?![\w-])", low):
            hits.add(word)
    return hits


# Multi-module phrasing: "multi-module", "across four files", "split into
# three modules", "spanning 5 files" -- an architecture ask in words even
# when no filenames are spelled out.
_COMPLEX_COUNT_WORDS = "three|four|five|six|seven|eight|nine|ten"
_COMPLEX_PHRASE_RE = re.compile(
    r"(?<!\w)multi[-\s]?module(?![\w-])"
    rf"|(?<!\w)(?:across|in|into|over|spanning)\s+"
    rf"(?:{_COMPLEX_COUNT_WORDS}|[3-9]|\d{{2,}})\s+(?:files|modules)(?![\w-])"
    rf"|(?<!\w)(?:{_COMPLEX_COUNT_WORDS})\s+(?:files|modules)(?![\w-])",
    re.IGNORECASE)


def detect_complex_scaffold(text: str) -> bool:
    """True when a New/Auto prompt describes a multi-module architecture:
    explicit multi-module phrasing, >=3 distinct source files named, or a
    build verb with >=3 module words. Deterministic, zero LLM calls; used to
    auto-enable Complex scaffolding (an explicit user choice always wins)."""
    low = str(text or "").lower()
    if not low.strip():
        return False
    if _COMPLEX_PHRASE_RE.search(low):
        return True
    names = {t.rsplit("/", 1)[-1] for t in _mode_file_mentions(low)
             if "." + t.rsplit(".", 1)[-1] in _MODE_SOURCE_EXTS}
    if len(names) >= 3:
        return True
    return bool(_MODE_BUILD_VERBS_RE.search(low)) \
        and len(_mode_module_hits(low)) >= 3


def evaluate_mode_suitability(
    task: str,
    root: Path | None,
    requested_mode: str = "auto",
) -> dict[str, Any]:
    """Deterministic Task-vs-Auto suitability check (zero LLM calls).

    Returns ``{"recommended_mode": "ask"|"auto", "mismatch": bool,
    "reason": str}``. ``mismatch`` is True only when Auto was requested but
    the prompt reads as a single-pass Task edit: it targets 1-2 existing
    files or uses localized edit verbs without an architecture ask, or the
    project is tiny (<=5 non-ignored source files) and the prompt modifies
    existing content/styling rather than scaffolding. An explicit planning
    ask ("multi-step plan", "create a plan", "from scratch") or >=4 distinct
    new files/modules always keeps Auto.
    """
    requested = "ask" if str(requested_mode).lower() in ("ask", "task") \
        else "auto"
    verdict: dict[str, Any] = {"recommended_mode": requested,
                               "mismatch": False, "reason": ""}
    text = str(task or "").strip()
    if requested != "auto" or not text:
        return verdict
    low = text.lower()
    files = _mode_check_files(root)
    basenames = {f.rsplit("/", 1)[-1] for f in files}
    mentions = _mode_file_mentions(text)
    existing = [t for t in mentions
                if t in files or t.rsplit("/", 1)[-1] in basenames]
    new_files = [t for t in mentions if t not in existing]
    modules = _mode_module_hits(low)
    builds = bool(_MODE_BUILD_VERBS_RE.search(low))
    # Explicit overrides: the user asked for planning or a big new build.
    if any(p in low for p in _MODE_OVERRIDE_PHRASES):
        return verdict
    # A multi-module architecture ask keeps Auto -- but naming >=3 files
    # that ALL already exist is a multi-file edit, not a new scaffold: the
    # upgrade needs new files, architecture phrasing or a build verb.
    if detect_complex_scaffold(text) and (
            new_files or not existing
            or _COMPLEX_PHRASE_RE.search(low) or builds):
        return verdict
    if len(new_files) >= 4 or (builds and len(modules) >= 4):
        return verdict
    architecture = builds and len(modules) >= 2
    reason = ""
    if not architecture:
        if 1 <= len(existing) <= 2:
            reason = ("This prompt targets existing files and can be "
                      "applied in one cohesive pass (~20s) in Task mode "
                      "instead of spinning up a multi-step PLAN.md.")
        elif _MODE_EDIT_VERBS_RE.search(low):
            reason = ("This looks like a localized edit that Task mode can "
                      "apply in one cohesive pass (~20s), instead of "
                      "spinning up a multi-step PLAN.md.")
    if not reason and len(files) <= 5 \
            and not _MODE_SCAFFOLD_RE.search(low) \
            and (_MODE_EDIT_VERBS_RE.search(low)
                 or _MODE_CONTENT_WORDS_RE.search(low)):
        reason = (f"This project has only {len(files)} source file(s), so a "
                  "single-pass Task edit (~20s) fits better than a "
                  "multi-step PLAN.md loop.")
    if reason:
        verdict.update(recommended_mode="ask", mismatch=True, reason=reason)
    return verdict


def read_plan(path: Path) -> list[str]:
    """Return the plan file's lines (empty list if it does not exist)."""
    if path.exists():
        return path.read_text(encoding="utf-8").split("\n")
    return []


def _is_unchecked(line: str) -> re.Match[str] | None:
    m = _TASK_LINE.match(line)
    return m if m and m.group(2).strip() == "" else None


def _find_unchecked(lines: list[str], task: str) -> int | None:
    """Index of the first unchecked line whose task text is `task`."""
    for i, ln in enumerate(lines):
        m = _is_unchecked(ln)
        if m and m.group(3).strip() == task:
            return i
    return None


def next_task(lines: list[str]) -> tuple[int, str] | None:
    """Return (line_index, task_text) of the first unchecked task, or None.
    File order is priority order (the plan is written dependencies-first)."""
    for i, ln in enumerate(lines):
        m = _is_unchecked(ln)
        if m:
            return i, m.group(3).strip()
    return None


def _set_box(path: Path, task: str, box: str, notify=None) -> bool:
    """Set the checkbox of the first unchecked task matching `task` to `box`
    ("x" = done, "!" = failed). Matching by text rather than line index means
    a plan file the model edited mid-run still resolves. Returns False if the
    task could not be found. `notify(path, reason)`, when given, fires right
    after the file lands so a live GUI can refresh the checklist."""
    changed = [False]

    def updater(text: str) -> str:
        lines = text.split("\n") if text else []
        i = _find_unchecked(lines, task)
        if i is None:
            return text
        m = _TASK_LINE.match(lines[i])
        lines[i] = f"{m.group(1)}[{box}] {m.group(3)}"
        changed[0] = True
        return "\n".join(lines)

    workspace.atomic_update_text(path, updater)
    if not changed[0]:
        return False
    if notify is not None:
        try:
            notify(path, "done" if box == "x" else "failed")
        except Exception:
            pass
    return True


def mark_done(path: Path, task: str, notify=None) -> bool:
    """Tick the first unchecked task whose text matches `task` as done."""
    return _set_box(path, task, "x", notify=notify)


def mark_failed(path: Path, task: str, notify=None) -> bool:
    """Flag the first unchecked task matching `task` as failed (`[!]`).

    A failed box is neither unchecked (so the loop never re-picks it and
    spins) nor done (so a human can see what to revisit)."""
    return _set_box(path, task, "!", notify=notify)


def insert_task_before(path: Path, task: str, new_title: str,
                       notify=None) -> bool:
    """Insert an unchecked ``- [ ] new_title`` line immediately before the
    first unchecked occurrence of ``task`` (the remediation path: the failed
    check stays unchecked and retries after the repair). Returns False when
    the anchor task is not found."""
    changed = [False]

    def updater(text: str) -> str:
        lines = text.split("\n") if text else []
        i = _find_unchecked(lines, task)
        if i is None:
            return text
        lines.insert(i, f"- [ ] {new_title}")
        changed[0] = True
        return "\n".join(lines)

    workspace.atomic_update_text(path, updater)
    if not changed[0]:
        return False
    if notify is not None:
        try:
            notify(path, "remediation")
        except Exception:
            pass
    return True


def emit_plan_updated(path: Path, root: Path | None = None, cfg=None,
                      reason: str = "updated") -> None:
    """Tell listeners (the GUI's live Plan panel) that the checklist changed.

    Written right after `PLAN.md` is created or a box is ticked, so the page
    can render the checklist while Task 1 is still starting instead of waiting
    for the whole Auto run to finish."""
    try:
        events_mod.get(root, cfg).emit("plan_updated", path=str(path),
                                       reason=reason)
    except Exception:
        pass


# Title of the repair task inserted when a live verification fails its
# evidence gate. It must route to the EDIT branch (no investigation start
# word) and contain no verify/confirm/observe/validate/check words, so the
# evidence gate does not demand browser work from the repair itself. The
# diagnosis travels in the FAILED LIVE TEST context note, never in the title
# (titles are mined for forced file reads). The final clause stops a repair
# from passing the check by breaking behavior the goal freezes.
_REMEDIATION_TITLE = (
    "Repair the defect described in the FAILED LIVE TEST note in your "
    "context: apply the smallest code fix that resolves it, keeping every "
    "other constraint from the goal -- if the fix would change behavior the "
    "goal freezes, do not edit; report the conflict instead")


def _prune_notes(notes: list[str]) -> None:
    """Trim the context-note window IN PLACE: keep the FIRST note (the run's
    reproduce/observe report, whose live-path findings anchor later tasks)
    plus the latest two."""
    if len(notes) > 3:
        del notes[1:-2]


def archive_plan(path: Path, root: Path, reason: str = "superseded") -> Path | None:
    """Copy a superseded plan into `.pedacito/sessions/` before replacing it.

    A finished plan is the record of what the previous Auto run did; starting
    a new goal must not silently erase it. Returns the archive path (or None
    when there was no plan / the copy failed). The live `PLAN.md` is left for
    the caller to overwrite.
    """
    path = Path(path)
    if not path.is_file():
        return None
    dest_dir = Path(root) / ".pedacito" / "sessions"
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    stem = f"{workspace.stamp()}-plan-{path.name}"
    dest = dest_dir / stem
    n = 1
    while dest.exists():
        dest = dest_dir / f"{stem.removesuffix('.md')}-{n}.md"
        n += 1
    try:
        shutil.copy2(path, dest)
    except OSError:
        return None
    try:
        events_mod.get(Path(root)).emit("plan_archived", path=str(dest),
                                        reason=reason)
    except Exception:
        pass
    return dest


def write_plan(path: Path, goal: str, titles: list[str]) -> None:
    """Write a fresh plan file: the goal text plus one unchecked task per
    title.

    A multi-line goal is persisted WHOLE: the first line rides the `Goal:`
    line, continuation lines get a `> ` quote prefix so they can never parse
    as task lines. The resume path reads it back with ``read_plan_goal`` --
    workers on a resumed run otherwise see no goal at all.
    """
    body = [PLAN_HEADER.rstrip()]
    gl = (goal or "").strip()
    if gl:
        first, *rest = gl.splitlines()
        body += ["", f"Goal: {first}"]
        body += [f"> {ln}" if ln else ">" for ln in rest]
    body += ["", *(f"- [ ] {t}" for t in titles)]
    path.parent.mkdir(parents=True, exist_ok=True)
    fileio.write_bytes(path, ("\n".join(body) + "\n").encode("utf-8"))


def read_plan_goal(path: Path) -> str:
    """The full goal text stored in a plan file, or "" when absent/legacy.

    Continuation lines are the `> `-prefixed lines that follow `Goal:`; the
    first task line (or any other content) ends the block. Quote-prefixed
    lines never match `_TASK_LINE`, so ticking boxes is unaffected.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return ""
    out: list[str] = []
    seen = False
    for ln in lines:
        if not seen:
            if ln.startswith("Goal: "):
                out.append(ln[len("Goal: "):])
                seen = True
            continue
        if ln.startswith("> "):
            out.append(ln[2:])
        elif ln == ">":
            out.append("")
        else:
            break
    return "\n".join(out).strip() if seen else ""


TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string", "maxLength": 400},
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    # Grammar-constrained sampling force-closes a string AT
                    # maxLength (valid JSON, no truncation flag), which can cut
                    # a title mid-sentence or derail the planner into ending
                    # the list early; generate_tasks re-asks when a title
                    # arrives at the cap.
                    "title": {"type": "string", "maxLength": 800},
                },
                "required": ["title"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["tasks"],
    "additionalProperties": False,
}

# Plan mode's variant: the same shape plus an optional `depends_on`. Kept apart
# so `auto` (and its one-field-per-task contract for small models) is untouched.
import copy as _copy
PLAN_SCHEMA = _copy.deepcopy(TASK_SCHEMA)
PLAN_SCHEMA["properties"]["tasks"]["items"]["properties"]["depends_on"] = {
    "type": "array", "items": {"type": "integer"},
    "description": "1-based numbers of EARLIER tasks this one depends on "
                   "(task 1 is the first); omit when none"}

_SIMPLE_WEB_GOAL_RE = re.compile(
    r"\b(?:website|webpage|web page|landing page|personal site|bio|"
    r"portfolio|static site|index\.html|styles?\.css|stylesheet|hero|"
    r"card|theme|layout|css|styling|restyle|redesign)\b", re.IGNORECASE)
_BUILD_STACK_GOAL_RE = re.compile(
    r"\b(?:react|next\.js|nuxt|sveltekit|vite|webpack|rollup|parcel|"
    r"angular|vue cli|django|flask|fastapi|express|backend|database|"
    r"api endpoint)\b", re.IGNORECASE)


def _simple_web_goal(text: str) -> bool:
    """A static page brief: visual polish, not an enterprise feature build."""
    text = text or ""
    return bool(_SIMPLE_WEB_GOAL_RE.search(text)) \
        and not _BUILD_STACK_GOAL_RE.search(text)


WEB_PLAN_NOTE = """\
STATIC WEB PAGE MODE -- the goal is a hand-authored page. Task titles must
preserve the user's exact requested copy, links and section order VERBATIM;
plan only visual craft (typography, spacing, colour, cards, buttons, link
badges, responsive layout). Forbid invented sections and boilerplate: no fake
contact forms, no mobile hamburger toggle on a one-page bio with two links, no
dummy JavaScript, no unrequested README.md deployment manual, no rigid
accessibility-test checklist, and no trailing "Run tests / Verify the page"
item -- the loop already runs webcheck and verification after every edit
task. A no-build site gets complete browser-valid CSS -- never raw
`@tailwind` build directives in a linked stylesheet (write real CSS, or use
the Tailwind CDN script with utility classes)."""


GREENFIELD_PLAN_NOTE = """\
GREENFIELD PROJECT -- the workspace has no source files yet; this goal builds
it from scratch. The one exception to the no-documentation-chores rule: include
exactly one task that writes a real README.md, placed AFTER the code tasks so
it documents what was actually built. That README must contain an Overview &
Architecture section, a Quick Start with exact copy-pasteable commands for
every CLI subcommand (concrete example arguments and the real flags), and a
Running Tests section with the exact test command (e.g. `python -m unittest
-v`). Any command-line entry point must print its help (`parser.print_help()`)
when run with no arguments instead of crashing with a traceback."""


PLAN_SYSTEM = """\
You break a software goal into a small, ordered list of single-focus
implementation tasks for an autonomous coding loop. The loop will implement
them one at a time, each in a fresh context window that sees only the goal,
the current task, and whatever source it looks up.

Rules:
- Order matters: foundations and dependencies first, integration last.
- Each task must be small enough to finish in one focused session and to
  verify by running a command (tests, a linter, an import, a script).
- Each title is a clear, imperative instruction ("Implement the markdown
  parser"), not a wish ("there should be a parser").
- One title, one change: never bundle distinct edits into one task ("Remove
  the old header and update the CSS" is two tasks). A bundled task cannot be
  verified or rolled back selectively when one of its changes breaks
  something.
- A title that hits its length cap is CUT mid-sentence and loses criteria:
  compress the goal's wording instead of copying long passages, and split a
  check with sub-items (a/b/c/d) into one task per sub-item.
- Use as many tasks as the work genuinely needs. Do not cap the count at a
  fixed number; a large repair may need a dozen or more real steps, while a
  small fix may need one. Do not pad the plan with tiny stub tasks.
- Do not include meta-tasks (planning, documentation-only chores, git work).
- Do not end the plan with a standalone re-check item ("Run the tests",
  "Verify the page renders", "Check that everything works"): the loop
  already runs verification (webcheck, the configured verify command,
  impacted tests) after EVERY edit task, so a trailing no-op checklist item
  adds no work. Where the rules above demand a live checkpoint, title it
  with the specific behavior to observe -- never a generic
  "verify everything" closer.
- If the goal is a BUG or a broken program (a traceback, a wrong result), do
  not return an empty list: produce concrete repair steps -- reproduce the
  failure, locate the cause, fix it, then verify. "Fix the bug" is a valid
  task title when that is genuinely all there is to do.
- If the goal describes observable behavior -- a UI, a game, rendering, or a
  complaint like "X was added but is not showing / still looks wrong" -- the
  FIRST task must run the program and observe the actual current behavior,
  and the LAST task must run the program and observe that the change took
  visible effect. Title them with how to observe and, for a size/layout
  claim, the number to measure ("Run the game and verify the canvas is at
  most 480x360 and scrollHeight equals innerHeight"): the loop FAILS a
  verification task that never drove the app.
- If the goal lists several independent visible changes, put a live
  verification checkpoint after EACH one, not a single checkpoint at the
  end: when one big final check fails, the failure cannot be attributed to
  a single change and the loop cannot repair selectively.
- A checkpoint that confirms a DEFECT ("observe the timing to confirm the
  12-second delay") asserts the BEFORE state: once the plan's own change
  lands, it can never pass again, and the loop will keep failing and
  repairing it. Title pre-change observations "Reproduce..." or
  "Investigate..." (they report findings; the loop does not gate them);
  phrase every Verify/Confirm/Observe checkpoint as the AFTER-change
  expectation ("Verify spawning begins 3 seconds into a round").
- If the goal says an earlier change never took effect, treat WHICH FILE the
  program actually runs as unknown: include a task to confirm which code is
  really loaded (who imports or instantiates it) before any edit task --
  dead or duplicated definitions are the usual culprit.
- Carry every concrete constraint from the goal into the titles of the tasks
  that must honour it ("Remove the mid-body head block, KEEPING the doubled
  30-row height and the width unchanged"): a worker's attention goes to its
  title, and a numeric or visible constraint that lives only in the goal
  prose is easily lost mid-edit.
- A task may be investigation-only ("Reproduce...", "Locate the cause...",
  "Check..."). Keep such a title free of edit verbs and put the actual code
  change in a later task; the loop treats it as a checkpoint and does not
  expect edits.
- NEVER return zero tasks. Every goal has at least one concrete step."""


def deps_from(raw, n: int) -> list[list[int]]:
    """Per-task `depends_on` lists (0-based indexes of EARLIER tasks), aligned
    with `_titles_from(raw)`; [] entries where none were given. The plan's
    `depends_on` values are 1-based task numbers, converted to indexes here."""
    entries = raw.get("tasks") if isinstance(raw, dict) else raw
    out: list[list[int]] = []
    for i, entry in enumerate(entries or []):
        deps: list[int] = []
        if isinstance(entry, dict):
            for d in entry.get("depends_on") or []:
                try:
                    k = int(d) - 1
                except (TypeError, ValueError):
                    continue
                if 0 <= k < n and k != i:
                    deps.append(k)
        out.append(deps)
    return out[:n] + [[] for _ in range(max(0, n - len(out)))]


def order_by_dependencies(titles: list[str], deps: list[list[int]]) -> list[str]:
    """Titles in an order that respects `deps` (stable: ties keep the
    planner's order; a cycle is broken by keeping the original order)."""
    n = len(titles)
    placed: list[int] = []
    left = list(range(n))
    while left:
        for i in left:
            if all(d in placed for d in deps[i] if d < n):
                placed.append(i)
                left.remove(i)
                break
        else:                       # a cycle: give up on ordering the rest
            placed.extend(left)
            break
    return [titles[i] for i in placed]


def replace_unchecked(path: Path, titles: list[str]) -> None:
    """Replace every still-unchecked task in a plan file with `titles`, keeping
    done/failed lines and the header (used when the planner revises the rest)."""
    def updater(text: str) -> str:
        lines = text.split("\n") if text else []
        out: list[str] = []
        inserted = False
        for ln in lines:
            if _is_unchecked(ln):
                if not inserted:
                    out += [f"- [ ] {t}" for t in titles]
                    inserted = True
                continue
            out.append(ln)
        if not inserted:
            out += [f"- [ ] {t}" for t in titles]
        return "\n".join(out) + "\n"

    workspace.atomic_update_text(path, updater)


def _titles_from(raw) -> list[str]:
    """Pull task titles out of a planning reply, tolerating a bare list or a
    list of plain strings as well as the schema's [{title: ...}] shape."""
    if isinstance(raw, dict):
        entries = raw.get("tasks") or []
    elif isinstance(raw, list):
        entries = raw
    else:
        return []
    titles = []
    for entry in entries:
        if isinstance(entry, dict):
            title = str(entry.get("title", "")).strip()
            if title:
                titles.append(title)
        elif isinstance(entry, str) and entry.strip():
            titles.append(entry.strip())
    return titles


def generate_tasks(spec_text: str, client, cfg,
                   root: Path | None = None, with_deps: bool = False):
    """Ask the model to break `spec_text` into an ordered task list.

    Re-asks once when a title arrives at exactly the schema's length cap
    (force-closed mid-sentence), retries once with a "you returned zero
    tasks" nudge on an empty reply, and finally falls back to one task built
    from the goal. When `root` is given, the project's instruction file is
    added to the planner prompt."""
    schema = PLAN_SCHEMA if with_deps else TASK_SCHEMA
    budget = getattr(cfg, "max_spec_chars", 60_000)
    body = spec_text.strip()[:budget]
    max_tokens = getattr(cfg, "max_tokens_plan", 2000)
    plan_system = PLAN_SYSTEM
    if _simple_web_goal(body):
        plan_system += "\n\n" + WEB_PLAN_NOTE
    if root is not None and not _mode_check_files(root):
        plan_system += "\n\n" + GREENFIELD_PLAN_NOTE
    if not (getattr(cfg, "allow_shell", False)
            or getattr(cfg, "allow_browser", False)
            or getattr(cfg, "allow_screenshots", False)):
        plan_system += (
            "\n\nTOOL AVAILABILITY NOTE: No interactive shell (`run`), "
            "headless browser (`browser`), or screenshot tool is enabled in "
            "this run. Do NOT emit live-observation tasks that ask the agent "
            "to launch or visually inspect the running app/game; verify every "
            "change with unit tests (`run_test`) and code/static checks "
            "instead.")
    try:
        from .config import effective_tier
        if effective_tier(cfg) in ("tiny", "small"):
            plan_system += (
                "\n\nSMALL-MODEL PLAN CAP: Keep the plan compact (at most "
                "6-8 concrete, single-file tasks). Do not explode one feature "
                "into dozens of tiny micro-steps.")
    except Exception:
        pass
    notes = instructions_mod.render(root, cfg) if root is not None else ""
    if notes:
        plan_system += "\n\n" + notes
    rules = prompts_mod.project_rules_block(cfg, root=root)
    if rules:
        plan_system += "\n\n" + rules
    base_user = (f"GOAL\n\n{body}\n\n"
                 "Break this into an ordered task list. Each task title must be "
                 "a single imperative sentence a coding agent can implement and "
                 "verify in isolation.")
    from . import roles as roles_mod
    specialty_note = roles_mod.specialty_planner_note(cfg)
    if specialty_note:
        base_user += "\n\n" + specialty_note
    raw = client.chat_json(
        [{"role": "system", "content": plan_system},
         {"role": "user", "content": base_user}],
        schema, max_tokens=max_tokens, disable_thinking=True)
    titles = _titles_from(raw)
    cap = TASK_SCHEMA["properties"]["tasks"]["items"]["properties"][
        "title"]["maxLength"]
    cut = [t for t in titles if len(t) >= cap]
    if titles and cut:
        # One bounded re-ask; titles still cut afterwards are accepted (the
        # full goal rides along in every task's background context).
        print(f"  [planner returned {len(cut)} title(s) at exactly the "
              f"{cap}-char schema cap (cut mid-sentence); re-asking]",
              file=sys.stderr)
        retry = (base_user + "\n\nYour previous reply was CUT: "
                 f"{len(cut)} task title(s) arrived at exactly the {cap}-"
                 "character schema cap, truncated mid-sentence"
                 + (" -- your reply ended right after this task; the rest "
                    "of the list was lost" if len(titles) == 1 else "")
                 + ". Re-emit the COMPLETE task list now. Compress each "
                   "title well under the cap (paraphrase the goal; never "
                   "copy a long passage), and split any check with "
                   "sub-items into one task per sub-item.")
        raw = client.chat_json(
            [{"role": "system", "content": plan_system},
             {"role": "user", "content": retry}],
            schema, max_tokens=max_tokens, disable_thinking=True)
        retried = _titles_from(raw)
        if retried:
            titles = retried
        if any(len(t) >= cap for t in titles):
            print("  [planner titles still at the cap after the re-ask; "
                  "accepting -- the full goal rides along with every task]",
                  file=sys.stderr)
    if not titles:
        nudge = (base_user + "\n\nYou returned ZERO tasks. Every goal has at "
                 "least one concrete step. For a bug, list the steps to "
                 "reproduce, locate, fix, and verify it. Return the tasks now.")
        raw = client.chat_json(
            [{"role": "system", "content": plan_system},
             {"role": "user", "content": nudge}],
            schema, max_tokens=max_tokens, disable_thinking=True)
        titles = _titles_from(raw)
    if not titles:
        first = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
        only = [f"Address: {first[:200] or 'the requested change'}"]
        return (only, [[]]) if with_deps else only
    if with_deps:
        return titles, deps_from(raw, len(titles))
    return titles


def run_verify(root: Path, command: str, timeout: int = 600, cfg=None
               ) -> tuple[bool, str]:
    """Run `command` (a shell string) in `root`. Returns (passed, output).

    The command is user-authored config, so it still runs through the shell,
    but its environment is the same allow-list scrub model-chosen commands
    get: ambient API keys and tokens never reach it. `proc.run` still prepends
    the project's venv bin directories to PATH via `with_venv_path`, so a bare
    `pytest`/`ruff` resolves as before. The whole process group is killed on
    timeout, so a GUI/launcher child cannot keep a window alive after the
    check gives up."""
    rc, out, stopped, _ = proc_mod.run(
        command, cwd=root, timeout=timeout, shell=True, max_output=200_000,
        env=safety.child_env(cfg))
    _journal_append(cfg, "subprocess", command=command, returncode=rc,
                    stopped=stopped)
    if stopped == "timeout":
        return False, f"(verification timed out after {timeout}s)"
    if stopped == "error":
        return False, f"(could not run verification: {out})"
    if cfg is not None:
        try:
            from . import memory
            memory.observe_command(root, command, rc == 0, cfg, kind="test")
        except Exception:
            pass
    if rc == 5 and _pytest_no_tests_ran(out) and not _project_has_tests(root):
        # pytest exit code 5 means "no tests collected". During early scaffold
        # tasks the test files do not exist yet, so this is not a product
        # failure and must not trigger a green-ratchet rollback of valid code.
        return True, "(pytest collected no tests yet -- nothing to verify)"
    return rc == 0, out.strip()


def _pytest_no_tests_ran(output: str) -> bool:
    """True when pytest's exit-code-5 output says it collected no tests."""
    text = (output or "").lower()
    return ("no tests ran" in text or "collected 0 items" in text
            or "collected no tests" in text)


def _project_has_tests(root: Path) -> bool:
    """True when any .py file under `root` defines a `def test_` function
    (bounded walk, so a large tree cannot stall verification)."""
    try:
        root = Path(root)
        count = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                d for d in dirnames
                if d not in {".git", ".venv", "venv", "node_modules",
                             "__pycache__", ".pedacito", ".tox", ".nox"}
                and not d.startswith("."))
            for name in filenames:
                if not name.endswith(".py"):
                    continue
                count += 1
                if count > 200:
                    return False
                p = Path(dirpath) / name
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                if re.search(r"^\s*def\s+test_", text, re.MULTILINE):
                    return True
        return False
    except OSError:
        return False


def _failure_score(output: str, diag: bool = False) -> int:
    """Severity of a failed check, for the rollback-if-worse guard: pytest-
    style failure/error counters when present, finding-line count for a
    diagnostics report, else a flat 1 (comparable across rounds, but no
    finer)."""
    text = output or ""
    m = re.search(r"(\d+) failed", text)
    if m:
        score = int(m.group(1))
        e = re.search(r"(\d+) errors?", text)
        return score + (int(e.group(1)) if e else 0)
    nm = _NODE_FAIL_COUNT_RE.search(text)   # Node `# fail N` / `ℹ fail N`
    if nm:
        return int(nm.group(1) or nm.group(2))
    if diag:
        return sum(1 for ln in text.splitlines()
                   if re.match(r"^\S+?:\d+(?::\d+)?:", ln)) or 1
    return 1


# ------------------------------------------- failure identity + traceback packet
# A scalar count cannot tell "one old failure fixed, one new failure added"
# from "nothing changed". These helpers key each failure by its stable pytest
# node id (falling back to a normalized assertion signature when no id exists)
# and diff two runs by identity. A truly new id repairs even when the totals
# are equal. The repair prompt gets a small structured packet instead of the
# tail of the raw test output.
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_PYTEST_SUMMARY_RE = re.compile(
    r"(?m)^(?P<kind>FAILED|ERROR)\s+(?P<id>.+?)"
    r"(?:\s+-\s+(?P<reason>.*))?$")
_PYTEST_SECTION_RE = re.compile(r"^=+\s*(?:FAILURES|ERRORS)\s*=+$")
_PYTEST_SHORT_RE = re.compile(r"^=+\s*short test summary info\s*=+$")
_PYTEST_BLOCK_RE = re.compile(r"^_{3,}\s*(?P<name>.+?)\s*_{3,}$")
_PYTEST_LOCATION_RE = re.compile(
    r"^(?P<file>[^\s:][^:]*):(?P<line>\d+):\s*(?P<rest>.*)$")
_PYTEST_FRAME_RE = re.compile(
    r'^\s*File\s+"(?P<file>[^"]+)",\s+line\s+(?P<line>\d+),'
    r'\s+in\s+(?P<func>.*)$', re.M)
_TRACEBACK_EXC_RE = re.compile(
    r"(?m)^(?P<exc>[A-Za-z_][\w.]*(?:Error|Exception|Failure|Exit))\b"
    r"(?::\s*(?P<msg>.*))?$")
# Node `--test` output (BUG-322): the TAP reporter prints `not ok N - name`
# with an indented YAML detail block; the default spec reporter prints a
# `✖ name` line with an indented AssertionError + stack. Neither matches the
# pytest regexes, so without these every Node failure classified as 0/0/0/0.
_NODE_TAP_NOTOK_RE = re.compile(r"^\s*not ok\s+\d+\s+-\s+(?P<name>.+?)\s*$")
_NODE_SPEC_FAIL_RE = re.compile(
    r"^\s*(?:✖|✗|×)\s+(?P<name>.+?)\s*(?:\(\d[\d.]*\s*ms\))?\s*$")
_NODE_FAIL_COUNT_RE = re.compile(
    r"(?m)^(?:#\s*fail\s+(\d+)|ℹ\s*fail\s+(\d+))\s*$")


@dataclass
class FailureIdentity:
    """One failing test, stable across runs by node id + assertion signature."""
    test_id: str
    signature: str = ""
    kind: str = "failed"          # "failed" | "error"
    file: str = ""                # location line from the traceback block
    line: int = 0
    block: str = ""
    frames: list[tuple[str, int, str]] = field(default_factory=list)


@dataclass
class FailureDiff:
    """Identity-level comparison of a pre-edit and post-edit test run.

    `new` drives repair. `fixed`, `still_failing` and `pre_existing` are kept
    for the packet and the run log: `pre_existing` is every failure the suite
    already had before the edit, and `still_failing` is the subset still red.
    """
    new: list[FailureIdentity] = field(default_factory=list)
    fixed: list[FailureIdentity] = field(default_factory=list)
    pre_existing: list[FailureIdentity] = field(default_factory=list)
    still_failing: list[FailureIdentity] = field(default_factory=list)

    @property
    def has_new(self) -> bool:
        return bool(self.new)


def _normalise_test_id(test_id: str) -> str:
    """Normalise path separators and a leading `./` so the same node id from
    two runs compares equal; keep the class/function/param suffix intact."""
    parts = str(test_id or "").strip().split("::")
    head = parts[0].replace("\\", "/")
    while head.startswith("./"):
        head = head[2:]
    return "::".join([head] + parts[1:])


def _normalise_failure_signature(text: str) -> str:
    """A stable assertion signature: whitespace, ANSI colour, object
    addresses, source paths and line numbers are normalised; the assertion and
    exception wording survives so two different failures stay distinguishable.
    """
    sig = _ANSI_ESCAPE_RE.sub("", str(text or ""))
    sig = re.sub(r"0x[0-9a-fA-F]+", "0x<addr>", sig)
    sig = re.sub(
        r"(?<![\w])(?:[A-Za-z]:)?(?:[/\\][^\s:]+)+"
        r"\.(?:py|pyw|js|jsx|ts|tsx|rb|go|rs|java|c|cc|cpp|h|hpp|cs|php|kt|"
        r"kts|swift|scala|lua)(?::\d+)?", "<path>", sig)
    sig = re.sub(
        r"\b[\w./\\-]+\.(?:py|pyw|js|jsx|ts|tsx|rb|go|rs|java|c|cc|cpp|h|"
        r"hpp|cs|php|kt|kts|swift|scala|lua):\d+\b", "<path>", sig)
    sig = re.sub(r"\bline \d+\b", "line <n>", sig, flags=re.IGNORECASE)
    sig = re.sub(r"\s+", " ", sig).strip()
    return sig[:240] or "(no assertion message)"


def _pytest_summary_rows(text: str) -> list[tuple[str, str, str]]:
    """(kind, node id, reason) rows from pytest's short summary."""
    rows = []
    for m in _PYTEST_SUMMARY_RE.finditer(text or ""):
        node_id = m.group("id").strip()
        if not node_id:
            continue
        rows.append((m.group("kind").lower(), _normalise_test_id(node_id),
                     _normalise_failure_signature(m.group("reason") or "")))
    return rows


def _pytest_failure_blocks(text: str) -> list[tuple[str, str]]:
    """(header name, block text) for each pytest FAILURES/ERRORS section."""
    blocks: list[tuple[str, str]] = []
    in_failures = False
    name = ""
    buf: list[str] = []

    def flush() -> None:
        nonlocal name, buf
        if name:
            body = "\n".join(buf).strip("\n")
            if body.strip():
                blocks.append((name, body))
        name, buf = "", []

    for line in (text or "").splitlines():
        if _PYTEST_SECTION_RE.match(line):
            flush()
            in_failures = True
            continue
        if _PYTEST_SHORT_RE.match(line):
            flush()
            in_failures = False
            continue
        if not in_failures:
            continue
        m = _PYTEST_BLOCK_RE.match(line)
        if m:
            flush()
            name = m.group("name").strip()
            continue
        if name:
            buf.append(line)
    flush()
    return blocks


def _block_failure(name: str, block: str,
                   node_id: str = "") -> FailureIdentity:
    """Build a FailureIdentity from one pytest traceback/assertion block."""
    lines = block.splitlines()
    reason = ""
    for ln in lines:
        stripped = ln.lstrip()
        if stripped.startswith("E ") or stripped == "E":
            reason = stripped[1:].strip()
            if reason:
                break
    loc_file, loc_line, loc_rest = "", 0, ""
    for ln in reversed(lines):
        m = _PYTEST_LOCATION_RE.match(ln.strip())
        if not m:
            continue
        candidate = m.group("file").strip()
        if "." in candidate or "/" in candidate or "\\" in candidate:
            loc_file = candidate
            loc_line = int(m.group("line"))
            loc_rest = m.group("rest").strip()
            break
    if not reason:
        reason = loc_rest
    if not reason:
        m = _TRACEBACK_EXC_RE.search(block)
        if m:
            reason = (m.group("exc")
                      + (": " + m.group("msg") if m.group("msg") else ""))
    frames = [(m.group("file"), int(m.group("line")), m.group("func").strip())
              for m in _PYTEST_FRAME_RE.finditer(block)]
    if not node_id:
        if "::" in name:
            node_id = name
        elif loc_file:
            node_id = f"{loc_file}::{name}"
        else:
            node_id = name
    node_id = _normalise_test_id(node_id)
    return FailureIdentity(
        node_id, _normalise_failure_signature(reason),
        "failed", loc_file, loc_line, block, frames)


def _block_match(node_id: str, blocks: list[tuple[str, str]],
                 used: set[int]) -> int | None:
    """Index of the failure block that belongs to `node_id`, if any."""
    tail = node_id.split("::")[-1]
    bare = tail.split("[", 1)[0]
    for i, (name, _body) in enumerate(blocks):
        if i in used:
            continue
        n_tail = name.split("::")[-1].split("[", 1)[0]
        if (name == node_id or name == tail
                or node_id.endswith("::" + name)
                or n_tail == bare):
            return i
    return None


# A file-level TAP wrapper (`not ok N - tests/foo.test.js`) aggregates the
# file's subtests; the real failing identities are the subtests inside it
# (BUG-221). Recognise the wrapper by its test-file name.
_NODE_TEST_FILE_RE = re.compile(r"\.(?:test|spec)\.[cm]?[jt]s$")
# The spec reporter's `✖ failing tests:` line introduces the summary re-list,
# not a test of its own; counting it inflated every red suite by one and
# invented a spurious new/fixed identity when the baseline used TAP (BUG-221).
_NODE_SPEC_HEADER_RE = re.compile(r"^failing tests:?$", re.IGNORECASE)


def _node_error_message(block: list[str]) -> str:
    """The assertion message from a TAP detail block. A quoted/plain
    `error: '...'` value is taken as-is; a YAML block scalar (`error: |-` /
    `>-`) carries its message on the following lines, so the literal `|-`
    indicator must not become the signature (BUG-221)."""
    for k, bl in enumerate(block):
        em = re.match(r"error:\s*(.*)$", bl)
        if not em:
            continue
        val = em.group(1).strip()
        if re.fullmatch(r"[|>][+-]?", val):
            msg: list[str] = []
            for nxt in block[k + 1:]:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*:(?:\s|$)", nxt) \
                        or nxt == "...":
                    break
                if nxt:
                    msg.append(nxt)
            return " ".join(msg).strip()
        return val.strip("'\"")
    return ""


def _node_subtests_from_block(btext: str) -> list[FailureIdentity]:
    """The nested `not ok` subtests inside a file-level wrapper's detail block,
    as identities keyed by the subtest name (BUG-221). The block's lines are
    already de-indented, so each subtest runs from its `not ok` line to the
    next one."""
    lines = btext.splitlines()
    marks = [i for i, ln in enumerate(lines) if _NODE_TAP_NOTOK_RE.match(ln)]
    out: list[FailureIdentity] = []
    for k, i in enumerate(marks):
        m = _NODE_TAP_NOTOK_RE.match(lines[i])
        name = re.sub(r"\s+", " ", m.group("name").strip())
        end = marks[k + 1] if k + 1 < len(marks) else len(lines)
        sub = [ln.strip() for ln in lines[i + 1:end] if ln.strip()]
        stext = "\n".join(sub)
        sig = _node_error_message(sub)
        if not sig:
            for bl in sub:
                if re.search(r"\b\w*(?:Error|Exception)\b", bl):
                    sig = bl
                    break
        file, lineno = "", 0
        lm = re.search(r"location:\s*'?([^':\n]+):(\d+)", stext)
        if lm:
            file, lineno = lm.group(1), int(lm.group(2))
        info = FailureIdentity("node::" + name,
                               _normalise_failure_signature(sig or name),
                               "failed")
        info.file, info.line = file, lineno
        info.block = stext
        out.append(info)
    return out


def _node_failure_identities(text: str) -> dict[str, FailureIdentity]:
    """Failing-test identities from Node `--test` output (TAP or spec reporter,
    BUG-322). Each `not ok N - name` / `✖ name` line plus its indented detail
    block becomes one identity keyed by the test name, with the assertion
    signature and source location parsed out of the block.

    BUG-221: the spec reporter's `✖ failing tests:` summary header is not a
    test, and a file-level TAP wrapper (`not ok N - tests/foo.test.js`) is an
    aggregate of its subtests, not a failure of its own. Counting either
    inflated the totals and invented spurious new/fixed identities when the
    baseline and the current run used different reporter/command shapes."""
    infos: dict[str, FailureIdentity] = {}
    lines = (text or "").splitlines()
    i, n = 0, len(lines)
    while i < n:
        m = (_NODE_TAP_NOTOK_RE.match(lines[i])
             or _NODE_SPEC_FAIL_RE.match(lines[i]))
        if not m:
            i += 1
            continue
        name = re.sub(r"\s+", " ", m.group("name").strip())
        if _NODE_SPEC_HEADER_RE.match(name):
            i += 1
            continue
        j = i + 1
        block: list[str] = []
        while j < n and (lines[j][:1] in (" ", "\t") or not lines[j].strip()):
            if lines[j].strip():
                block.append(lines[j].strip())
            j += 1
        btext = "\n".join(block)
        # A file-level wrapper stands for its subtests: emit those instead so
        # the identity is the actual test, stable across reporter shapes. A
        # wrapper with no subtests (a module-link failure) stays one identity.
        if _NODE_TEST_FILE_RE.search(name):
            subs = _node_subtests_from_block(btext)
            if subs:
                for info in subs:
                    infos[info.test_id] = info
                i = j
                continue
        sig = _node_error_message(block)
        if not sig:
            for bl in block:
                if re.search(r"\b\w*(?:Error|Exception)\b", bl):
                    sig = bl
                    break
        signature = _normalise_failure_signature(sig or name)
        file, lineno = "", 0
        lm = re.search(r"location:\s*'?([^':\n]+):(\d+)", btext)
        if lm:
            file, lineno = lm.group(1), int(lm.group(2))
        else:
            fm = re.search(r"at\s+[^\n(]*\(?([^()\s]+):(\d+):\d+\)?", btext)
            if fm:
                file, lineno = fm.group(1), int(fm.group(2))
        info = FailureIdentity("node::" + name, signature, "failed")
        info.file, info.line = file, lineno
        info.block = btext
        infos[info.test_id] = info
        i = j
    return infos


def failure_identities(output: str) -> dict[str, FailureIdentity]:
    """Stable failing-test identities from a pytest or Node `--test` run, keyed
    by node id / test name.

    The short summary supplies ids and reasons; the FAILURES/ERRORS blocks
    add the source location and project traceback frames. When a run has no
    pytest summary (truncated output, another runner) a bare traceback still
    yields one signature-keyed identity rather than silently losing the
    failure. No model call and no file I/O happen here."""
    text = _ANSI_ESCAPE_RE.sub("", output or "")
    blocks = _pytest_failure_blocks(text)
    infos: dict[str, FailureIdentity] = {}
    used: set[int] = set()
    for kind, node_id, reason in _pytest_summary_rows(text):
        info = FailureIdentity(node_id, reason, kind)
        i = _block_match(node_id, blocks, used)
        if i is not None:
            used.add(i)
            detail = _block_failure(blocks[i][0], blocks[i][1], node_id)
            info.file, info.line = detail.file, detail.line
            info.block, info.frames = detail.block, detail.frames
            if info.signature.startswith("(no assertion"):
                info.signature = detail.signature
        infos[node_id] = info
    for i, (name, block) in enumerate(blocks):
        if i in used:
            continue
        info = _block_failure(name, block)
        if not info.test_id:
            info.test_id = f"signature::{info.signature}"
        if info.test_id in infos:
            old = infos[info.test_id]
            old.file, old.line = info.file, info.line
            old.block, old.frames = info.block, info.frames
            if old.signature.startswith("(no assertion"):
                old.signature = info.signature
        else:
            infos[info.test_id] = info
    if not infos:
        # Node `--test` (TAP / spec) before the bare-traceback fallback, so a
        # red Node suite classifies by identity instead of 0/0/0/0 (BUG-322).
        infos = _node_failure_identities(text)
    if not infos:
        frames = [(m.group("file"), int(m.group("line")),
                   m.group("func").strip())
                  for m in _PYTEST_FRAME_RE.finditer(text)]
        m = None
        for m in _TRACEBACK_EXC_RE.finditer(text):
            pass
        if frames or m:
            reason = ((m.group("exc")
                       + (": " + m.group("msg") if m.group("msg") else ""))
                      if m else "test failed")
            signature = _normalise_failure_signature(reason)
            ident = ("output::" + signature)
            info = FailureIdentity(ident, signature, "error")
            if frames:
                info.frames = frames
                info.file, info.line = frames[-1][0], frames[-1][1]
            infos[ident] = info
    return infos


# Aliases used by callers that describe the parse step rather than the result.
parse_failure_identities = failure_identities
_failure_identities = failure_identities


def classify_failures(baseline_output: str,
                      current_output: str) -> FailureDiff:
    """Diff post-edit failures against the pre-edit baseline by identity.

    `new` is a current node id that was not failing before (or a signature
    when ids are unavailable); `fixed` is a baseline id that is gone;
    `still_failing` is the baseline intersection. A new failure is a repair
    trigger even when the total failure count is unchanged."""
    before = failure_identities(baseline_output)
    after = failure_identities(current_output)
    diff = FailureDiff(pre_existing=list(before.values()))
    for key, failure in after.items():
        if key in before:
            diff.still_failing.append(failure)
        else:
            diff.new.append(failure)
    diff.fixed = [failure for key, failure in before.items()
                  if key not in after]
    return diff


_classify_failures = classify_failures


def _project_rel(root: Path, name: str) -> str:
    """Project-relative path for a traceback frame, or "" for a path outside
    the project or inside an environment directory (`.venv`, site-packages,
    node_modules...). Those frames are never sent to repair."""
    if not name:
        return ""
    path = Path(name)
    if not path.is_absolute():
        path = Path(root) / path
    try:
        rel = path.resolve().relative_to(Path(root).resolve())
    except (OSError, RuntimeError, ValueError):
        return ""
    parts = [part.lower() for part in rel.parts]
    if any(part in {".venv", "venv", ".tox", "node_modules", "site-packages",
                    "dist-packages"} for part in parts):
        return ""
    if any("site-packages" in part or "dist-packages" in part
           for part in parts):
        return ""
    return rel.as_posix()


def _same_rel(a: str, b: str) -> bool:
    ka, kb = fileio.path_key(a or ""), fileio.path_key(b or "")
    return bool(ka and kb and (ka == kb or ka.endswith("/" + kb)
                               or kb.endswith("/" + ka)))


def _source_window(root: Path, rel: str, line: int,
                   max_lines: int = 15) -> tuple[str, int, int, str]:
    """(numbered window, first line, last line, rel) around `line`; 15 lines
    by default, with the implicated line marked `>>`."""
    project_rel = _project_rel(root, rel)
    if not project_rel:
        return "", 0, 0, ""
    try:
        text = (Path(root) / project_rel).read_text(
            encoding="utf-8", errors="replace")
    except OSError:
        return "", 0, 0, project_rel
    lines = text.split("\n")
    if lines and lines[-1] == "" and len(lines) > 1:
        lines.pop()
    if not lines:
        return "", 0, 0, project_rel
    hit = max(1, min(int(line or 1), len(lines)))
    before = max(1, max_lines // 2)
    start = max(1, hit - before)
    end = min(len(lines), start + max_lines - 1)
    start = max(1, end - max_lines + 1)
    shown = []
    for i in range(start, end + 1):
        mark = ">>" if i == hit else "  "
        shown.append(f"{mark}{i:>4}| {lines[i - 1]}")
    return "\n".join(shown), start, end, project_rel


def _test_def_line(root: Path, test_id: str) -> int:
    """Line of the test function named by a node id, when the traceback did
    not record one (used to anchor the assertion window)."""
    if "::" not in (test_id or ""):
        return 0
    file_part, _, tail = test_id.partition("::")
    func = tail.split("::")[-1].split("[", 1)[0].strip()
    rel = _project_rel(root, file_part)
    if not rel or not func:
        return 0
    try:
        lines = (Path(root) / rel).read_text(
            encoding="utf-8", errors="replace").split("\n")
    except OSError:
        return 0
    rx = re.compile(r"^\s*(?:async\s+)?def\s+" + re.escape(func) + r"\s*\(")
    for i, line in enumerate(lines, 1):
        if rx.match(line):
            return i
    return 0


def _assertion_anchor(root: Path, failure: FailureIdentity) -> tuple[str, int]:
    """(project file, line) of the failing assertion window."""
    test_file = (failure.test_id.split("::", 1)[0]
                 if "::" in (failure.test_id or "") else "")
    if failure.file and failure.line:
        if not test_file or _same_rel(failure.file, test_file):
            return failure.file, failure.line
    line = _test_def_line(root, failure.test_id)
    if line:
        return test_file, line
    return (failure.file or test_file), failure.line


def _project_frame(root: Path, failure: FailureIdentity,
                   test_rel: str = "") -> tuple[str, int, str] | None:
    """The innermost project-owned traceback frame outside the test file."""
    for file, line, func in reversed(failure.frames or ()):
        rel = _project_rel(root, file)
        if not rel:
            continue
        if test_rel and _same_rel(rel, test_rel):
            continue
        return rel, int(line or 0), func
    return None


def traceback_repair_packet(root: Path, diff: FailureDiff,
                            current=None) -> str:
    """A compact, structured repair payload for a failed post-edit test run.

    Only project-owned frames are quoted (`.venv`/site-packages are dropped).
    Each new failure gets its normalized signature, a ~15-line failing-test
    assertion window, and a ~15-line window around the innermost implicated
    project traceback frame. Fixed and pre-existing ids are listed so the
    model neither re-fixes old red tests nor chases unrelated ones."""
    current = current or {}
    focus = list(diff.new) or list(diff.still_failing) or list(current.values())
    out = ["TRACEBACK REPAIR PACKET",
           f"Failure identity diff: {len(diff.new)} new, {len(diff.fixed)} "
           f"fixed, {len(diff.still_failing)} still failing, "
           f"{len(diff.pre_existing)} pre-existing."]
    if diff.fixed:
        out.append("Fixed since the pre-edit baseline: "
                   + ", ".join(f.test_id for f in diff.fixed[:5]))
    if diff.still_failing:
        out.append("Pre-existing failures still red (NOT caused by this edit): "
                   + ", ".join(f.test_id for f in diff.still_failing[:5]))
    if not focus:
        out.append("No structured pytest node ids were parsed from the output.")
        return "\n".join(out)
    for failure in focus[:3]:
        out.append("")
        out.append(f"FAILING TEST: {failure.test_id}")
        out.append(f"NORMALIZED FAILURE SIGNATURE: {failure.signature}")
        anchor, line = _assertion_anchor(root, failure)
        window, start, end, rel = _source_window(root, anchor, line)
        if window:
            out.append(f"ASSERTION WINDOW ({rel}:{start}-{end}):")
            out.append(window)
        test_rel = (_project_rel(root, failure.test_id.split("::", 1)[0])
                    if "::" in (failure.test_id or "") else rel)
        frame = _project_frame(root, failure, test_rel)
        if frame:
            frame_rel, frame_line, func = frame
            fwindow, fs, fe, frel = _source_window(
                root, frame_rel, frame_line)
            if fwindow:
                out.append(
                    f"PROJECT TRACEBACK FRAME ({frel}:{frame_line}, in "
                    f"{func or '?'}) window {fs}-{fe}:")
                out.append(fwindow)
    text = "\n".join(out)
    return text[:9000]


# Internal names kept short where the call sites read better.
_traceback_packet = traceback_repair_packet


_MISSING_MODULE_RE = re.compile(
    r"(?:ModuleNotFoundError|ImportError):\s*No module named\s+['\"]([^'\"]+)['\"]")

# Common import-name -> pip-distribution-name mismatches.
_PIP_NAMES = {
    "pil": "Pillow", "yaml": "PyYAML", "cv2": "opencv-python",
    "sklearn": "scikit-learn", "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil", "dotenv": "python-dotenv",
    "pkg_resources": "setuptools", "opengl": "PyOpenGL",
}


def _is_local_module(root: Path, name: str) -> bool:
    """True when `name` resolves to this project's own source: a top-level
    module file or a package directory. Deliberately narrow -- a dotted or
    odd-cased name is not an import root and is not checked."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        return False
    try:
        return (root / f"{name}.py").is_file() \
            or (root / name / "__init__.py").is_file()
    except OSError:
        return False


_UNLAUNCHABLE_HINTS = (
    "command not found",
    "is not recognized as an internal or external command",
    "is not recognized as the name of a cmdlet",
)


def _verify_unlaunchable(output: str) -> bool:
    """The check command itself could not be started (rc=127 family): the
    binary is missing, so no product change can green this check. Narrowly
    matched on launcher phrasing so a program that PRINTS these words from
    within a test assertion is not mistaken for a failed exec."""
    low = (output or "").lower()
    return any(h in low for h in _UNLAUNCHABLE_HINTS)


def missing_modules(output: str, root=None) -> list[str]:
    """Top-level import names (``foo`` for ``foo.bar``) from
    ModuleNotFoundError/ImportError messages. Callers route these to
    dependency handling instead of the code-repair loop.

    With `root` given, names that resolve to the project's OWN source are
    excluded (BUG-1002): an auto-built project whose test imports a sibling
    module under a plain `pytest` invocation reported
    `pip install pattern_store` -- a module sitting right there on disk --
    and with --auto-install would have installed (and imported) whatever
    distribution holds that name upstream. A local name is a broken import
    path the repair loop can fix; it is never a missing dependency."""
    out: list[str] = []
    for m in _MISSING_MODULE_RE.finditer(output or ""):
        name = m.group(1).split(".")[0].strip()
        if not name or name in out:
            continue
        if root is not None and _is_local_module(Path(root), name):
            continue
        out.append(name)
    return out


def pip_name(module: str) -> str:
    """The pip distribution for a module import name (the same name unless it
    is one of the common mismatches, e.g. ``PIL`` -> ``Pillow``)."""
    return _PIP_NAMES.get(module.lower(), module)


def install_modules(packages: list[str], root: Path) -> tuple[bool, str]:
    """pip-install the distributions for `packages` into THIS interpreter's
    environment, returning (ok, output).

    Only called when the user opted in (``--auto-install``): the module names
    come from model-generated code."""
    dists = sorted({pip_name(p) for p in packages
                    if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", p or "")})
    if not dists:
        return False, "no valid package names to install"
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", *dists]
    try:
        proc = subprocess.run(cmd, cwd=str(root), capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=600)
    except Exception as e:                       # pragma: no cover - env specific
        return False, f"could not run pip: {e}"
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return proc.returncode == 0, out


@dataclass
class TaskResult:
    """What one task run produced: the edit results, the reply, whether any
    writes landed, the files re-indexed, and the (possibly refreshed) index.
    `steps` is the lookup phase's action log ("4. browser ..."), which lets
    the outer loop tell objectively whether a verification task drove the app."""
    results: list
    reply: str
    applied: bool
    changed: list[str]
    idx: index_mod.Index
    session: Path | None = None
    steps: list[str] = field(default_factory=list)
    # True when the reply's only edit blocks were null edits (SEARCH ==
    # REPLACE) and the reply did not itself declare the work complete; the
    # outer loop treats that as a failed attempt, not "already done".
    fabricated: bool = False
    # The run's backup snapshot dir (when cfg.backup is on): what the
    # rollback-if-worse guard restores to undo a repair that made the check
    # worse.
    backup_dir: Path | None = None
    # Whether a run/browser/screenshot action actually executed and produced
    # a non-error observation. None means the caller predates the flag; the
    # steps log is then used as the fallback. A merely ATTEMPTED live step
    # (bad URL, missing server) is not evidence.
    live_ok: bool | None = None
    # WHICH live actions produced an observation ("run", "run_test",
    # "browser", "screenshot"). A title that asks what the page/canvas showed
    # is not satisfied by a `run` that only started the process, so the gate
    # reads the kinds, not just the flag. None means the caller predates the
    # field; the steps log is then used as the fallback.
    live_kinds: tuple[str, ...] | None = None
    # Deterministic/verifier findings on the proposed edits (verify.Finding).
    findings: list = field(default_factory=list)
    # R8 provenance (BUG-215): file -> clause ids each result's patch serves,
    # the clause id -> text legend, and the edit clauses no patch evidenced
    # ("unaccepted"). The JSONL result frame and the RPC summary carry these
    # so machine clients can show what the review page shows.
    clauses: dict = field(default_factory=dict)
    clause_texts: dict = field(default_factory=dict)
    unaccepted: list = field(default_factory=list)
    # Per-task usage: tokens, cache-hit ratio, seconds, cost (see _task_stats).
    stats: dict = field(default_factory=dict)
    # True when the task stopped early because it hit max_task_seconds or
    # max_task_completion_tokens (partial/aborted work).
    budget_hit: bool = False
    # The post-edit impacted-tests verdict: None when the tests did not run
    # (verification off / no test command), else True/False from the last
    # verification round. `ask --strict` exits nonzero on False.
    tests_ok: bool | None = None
    # True when the impacted suite is red but EVERY remaining failure predates
    # this task (R2): no new identity and no worse count. `ask --strict` then
    # treats the run as clean -- a recorded pre-existing failure is not a task
    # failure (BUG-220). False whenever a gate or a new test failure is red.
    tests_preexisting_only: bool = False
    # Green-ratchet bookkeeping (`auto` only): True when this task's step
    # failed its check after all repair rounds and was UNDONE -- files
    # restored from the step's snapshot, step-created files removed -- and
    # `rollback_info` holds that rollback's report (restored/deleted/skipped).
    rolled_back: bool = False
    rollback_info: dict = field(default_factory=dict)


def _snapshot_stats(client) -> dict:
    """Copy of a client's cumulative usage counters (test doubles may have
    none)."""
    st = getattr(client, "stats", None)
    if not isinstance(st, dict):
        return {}
    return {k: v for k, v in st.items() if isinstance(v, (int, float))}


def _task_stats(client, before: dict, started: float) -> dict:
    """Usage consumed since `before`: calls, tokens, cached share, dollars."""
    now = _snapshot_stats(client)
    d = {k: now.get(k, 0) - before.get(k, 0) for k in now}
    pt = d.get("prompt_tokens", 0)
    return {"calls": d.get("calls", 0), "prompt_tokens": pt,
            "completion_tokens": d.get("completion_tokens", 0),
            "cached_tokens": d.get("cached_tokens", 0),
            "cache_write_tokens": d.get("cache_write_tokens", 0),
            "cache_hit": round(d.get("cached_tokens", 0) / pt, 3) if pt else 0.0,
            "cost": round(d.get("cost", 0.0), 6),
            "seconds": round(time.time() - started, 2)}


def safe_text(text: str, cfg=None) -> str:
    """`text` with secret-looking values masked when redaction is on (log
    lines and session files must not become a secret store)."""
    if cfg is not None and not getattr(cfg, "redact_secrets", True):
        return text
    return safety.redact(text)


def _fire(cfg, root, event: str, **fields):
    """Fire a user hook; a HookResult that allows when none is configured."""
    if not hooks_mod.has_hooks(cfg, event):
        return hooks_mod.HookResult()
    return hooks_mod.fire(cfg, root, event, **fields)


def _run_tests(idx, cfg, files: list[str]):
    """(passed, output, description) of the tests impacted by `files`; None
    when no test command exists or the run itself failed to start."""
    try:
        ok, out, desc = testsel.run_impacted(idx, cfg, files)
    except Exception as e:
        print(f"  [tests: could not run ({type(e).__name__}: {e})]",
              file=sys.stderr)
        return None
    if desc == "no test command detected":
        return None
    return ok, out, desc


@dataclass(frozen=True)
class _RedRun:
    """Routing verdict for one red test run (WP-4105D). `real` is what repair
    may see; `unverified` means the run stays red and is not repaired."""
    real: tuple[FailureIdentity, ...]
    unverified: bool
    category: str
    summary: str


def _emit_test_category(root, cfg, round_: int, c, quarantined=False) -> None:
    events_mod.emit_test_category(
        events_mod.get(root, cfg), round_=round_, category=c.kind,
        node_id=c.node_id, baseline=c.baseline, rerun=c.rerun,
        reason=c.reason, quarantined=quarantined)


def _classify_new_failures(root, cfg, idx, changed, out, base_out, diff,
                           reruns: dict, skip: set) -> dict:
    """{node id: TestClassification} for each new failure of a red run.

    A failing node is rerun at most once per verify run (`reruns` remembers
    every verdict across repair rounds) and at most `MAX_RERUN_NODES` nodes are
    rerun. Nodes without a stable pytest id, a file inside the project, or
    budget are classified `unknown` and are not rerun. `skip` holds quarantined
    ids, which are reported but never rerun or repaired."""
    base = testsel.impacted_base(root, cfg, changed)
    verdicts: dict = {}
    for fail in diff.new:
        nid = fail.test_id
        if nid in skip:
            continue
        if nid in reruns:
            verdicts[nid] = reruns[nid]
            continue
        bstatus = test_classify.baseline_status(
            base_out, failure_identities(base_out or ""), nid)
        cmd = test_classify.rerun_command(base, nid)
        if not cmd or not test_classify.node_file_inside(root, nid):
            verdicts[nid] = test_classify.TestClassification(
                "unknown", nid, bstatus, "not_run",
                "no stable node id to rerun; the failure is unverified")
            continue
        if len(reruns) >= test_classify.MAX_RERUN_NODES:
            verdicts[nid] = test_classify.TestClassification(
                "unknown", nid, bstatus, "not_run",
                "rerun budget for this verify run is spent")
            continue
        passed, rerun_out = testsel.run_command(root, cfg, cmd)
        verdict = test_classify.classify_test_failure(
            out, test_classify.NodeRerun(nid, bool(passed), rerun_out or ""),
            base_out)
        reruns[nid] = verdict
        verdicts[nid] = verdict
    return verdicts


def _triage_red_run(root, cfg, idx, changed, out, base_out, desc, diff,
                    current, reruns: dict, quarantine: dict,
                    round_: int) -> _RedRun:
    """Decide which failures of a red run may enter model repair (WP-4105D).

    Infrastructure stops (timeout, collection error, missing dependency,
    infrastructure) consume no rerun. A recorded-history pre-existing failure and
    an all-pre-existing red suite keep their legacy handling. Otherwise each new
    failure is classified and only `real_failure` is repairable."""
    infra = test_classify.classify_output_kind(
        out, has_node_failures=test_classify.has_test_node(current))
    if infra:
        note = test_classify.TestClassification(
            infra, "", "absent" if not base_out else "not_failing", "not_run",
            f"{infra}: the run stopped before a test verdict; not repaired")
        _emit_test_category(root, cfg, round_, note)
        return _RedRun((), True, infra, note.reason)
    if not diff.new:
        if not current:
            return _RedRun((), True, "unknown",
                           "no failing test could be identified")
        return _RedRun((), False, "", "")
    if not base_out and testsel.PRE_EXISTING_LABEL in desc:
        return _RedRun((), False, "", "")

    quarantined = {f.test_id for f in diff.new if f.test_id in quarantine}
    verdicts = _classify_new_failures(root, cfg, idx, changed, out, base_out,
                                      diff, reruns, quarantined)
    real: list[FailureIdentity] = []
    unverified: list[str] = []
    category = ""
    for fail in diff.new:
        nid = fail.test_id
        if nid in quarantined:
            q = quarantine[nid]
            events_mod.emit_test_category(
                events_mod.get(root, cfg), round_=round_,
                category="quarantined", node_id=nid, baseline="failed",
                rerun="not_run", reason=f"{q.reason} (until {q.expires})",
                quarantined=True)
            unverified.append(f"quarantined: {nid}")
            category = category or "quarantined"
            continue
        verdict = verdicts[nid]
        _emit_test_category(root, cfg, round_, verdict)
        if verdict.kind in test_classify.REPAIRABLE_KINDS:
            real.append(fail)
        else:
            unverified.append(f"{verdict.kind}: {nid}")
            category = category or verdict.kind
    if real:
        return _RedRun(tuple(real), False, "real_failure",
                       f"{len(real)} attributable failure(s)")
    return _RedRun((), True, category or "unknown",
                   "; ".join(unverified)[:300])


# A test command the TASK ITSELF names ("make node --test tests/x.test.js
# pass", "npm test", "pytest ..."): its guarantee is part of the task, so a
# run that never executed it must say so instead of ending silent (BUG-203).
_TASK_TEST_RE = re.compile(
    r"\bnode\s+--test(?:\s+[\w.*./-]+)*|\bnpm\s+(?:run\s+)?test\b|"
    r"\bpython\s+-m\s+pytest\b|\bpytest\b|\bmake\s+test\b|"
    r"\byarn\s+test\b|\bnpx\s+(?:jest|vitest|mocha)\b|"
    r"\b(?:jest|vitest|mocha)\b", re.IGNORECASE)


def _task_named_tests(task: str) -> str:
    """The test command/file the task text names, '' when it names none."""
    m = _TASK_TEST_RE.search(task or "")
    return " ".join(m.group(0).split()) if m else ""


def _test_run_consent(cfg) -> bool:
    """Did the USER name a test command for this run -- via `test_command`
    or a `--verify` that is itself a test command?

    BUG-1006: `allow_shell` is the general execution boundary, but the
    session/step final gate (`run_check` -> `run_verify`) already executes a
    user-typed `--verify "pytest -q"` on every task and records it in the
    ledger as kind="test" evidence, and the interface-lock checks run the
    same pytest binary -- while `_post_edit_tests` ANDed its test subprocess
    with `allow_shell` and printed "no test evidence gathered" mid-run. One
    command, two contradictory verdicts. A command the user typed at the CLI
    (or configured in trusted config) carries its own consent; task text,
    which a worker may have written, does not."""
    if bool((getattr(cfg, "test_command", "") or "").strip()):
        return True
    return bool(getattr(cfg, "_tests_user_consent", False))


def _note_tests_not_run(root, cfg, task: str,
                        granted: bool = False) -> None:
    """The honest verdict for a task that named a test command while the
    test subprocess was switched off: no test evidence was gathered.
    BUG-1008: `granted` says the execution boundary is open (allow_shell or
    a user-granted test command) and the only missing piece is a DETECTED
    command -- blaming allow_shell there sent the user to adjust a knob that
    was not the blocker (chiptune run 3, task 1)."""
    named = _task_named_tests(task)
    if not named:
        return
    if granted:
        print(f"  [tests: NOT RUN -- the task names '{named}' but no test "
              "command was detected for this project; set `test_command` "
              "or add runnable tests]", file=sys.stderr)
        workspace.log_event(
            root, f"tests not run although the task names '{named}' "
                  "(no test command detected)", "WARN")
        events_mod.get(root, cfg).emit("tests_not_run", named=named,
                                       reason="no_test_command")
        return
    print(f"  [tests: NOT RUN -- the task names '{named}' but "
          "verify_tests/allow_shell is off and no test command was granted "
          "(--verify or test_command); this run gathers no test "
          "evidence for it]", file=sys.stderr)
    workspace.log_event(
        root, f"tests not run although the task names '{named}' "
              "(verify_tests/allow_shell off)", "WARN")
    events_mod.get(root, cfg).emit("tests_not_run", named=named)


# A task that NAMES its verification has made that check part of its
# guarantee: "run npm test" with verify_tests off ended green without any
# test evidence (BUG-211), and "run domcheck" with verify_dom off shipped
# page errors (BUG-216). Enable the named check for the run; the execution
# boundary (allow_shell) is still the user's to grant, and a run that cannot
# execute keeps the honest tests_not_run verdict above.
_TASK_DOM_RE = re.compile(
    r"\bdom\s*check\b"
    r"|\bheadless\s+browser\b"
    r"|\bbrowser\s+(?:check|test|verification)\b"
    # "verify/test/load ... in a (headless) browser": a verb of checking,
    # then a short span, then "in [a/the] [headless] browser".
    r"|\b(?:verify|test|check|load|open|run|validate|confirm)\b"
    r"[^.\n]{0,40}?\bin\s+(?:a\s+|an\s+|the\s+)?(?:headless\s+)?browser\b",
    re.IGNORECASE)


def _infer_task_checks(root, cfg, task: str) -> None:
    """Turn on verify_tests/verify_dom when the task text names them.

    BUG-1003: both flags are ANDed with `allow_shell` at their consumers, so
    on a shell-less run this assignment changed nothing -- yet the harness
    printed "enabling verify_tests for this run" and minutes later retracted
    it with "verify_tests/allow_shell is off". Enable honestly: announce a
    real enablement as one, and announce the boundary block as the boundary
    block it is."""
    shell = bool(getattr(cfg, "allow_shell", False))
    named = _task_named_tests(task)
    if named and not getattr(cfg, "verify_tests", False):
        cfg.verify_tests = True
        if shell or _test_run_consent(cfg):
            print(f"  [task names '{named}'; enabling verify_tests for this "
                  "run]", file=sys.stderr)
            workspace.log_event(root, f"verify_tests enabled: the task names "
                                      f"'{named}'", "INFO")
        else:
            print(f"  [task names '{named}'; verify_tests is on but the test "
                  "subprocess also needs allow_shell or a granted test "
                  "command (--verify/test_command) -- this run will report "
                  "no test evidence]", file=sys.stderr)
            workspace.log_event(root, f"verify_tests enabled but blocked: "
                                      f"'{named}' named, allow_shell off",
                                "WARN")
    m = _TASK_DOM_RE.search(task or "")
    if m and not getattr(cfg, "verify_dom", False):
        cfg.verify_dom = True
        dom_named = " ".join(m.group(0).split())
        if shell or getattr(cfg, "_dom_user_consent", False):
            print(f"  [task names '{dom_named}'; enabling verify_dom for this "
                  "run]", file=sys.stderr)
            workspace.log_event(root, f"verify_dom enabled: the task names "
                                      f"'{dom_named}'", "INFO")
        else:
            print(f"  [task names '{dom_named}'; verify_dom is on but the "
                  "headless check also needs allow_shell (or ask "
                  "--verify-dom) -- this run will not load the page]",
                  file=sys.stderr)
            workspace.log_event(root, f"verify_dom enabled but blocked: "
                                      f"'{dom_named}' named, allow_shell off",
                                "WARN")


def _merge_repair_results(results, new):
    """Keep earlier good files, but let a repair's successful Result replace
    the stale pre-repair buffer for the same file (otherwise the syntax gate
    would keep re-reading an error the repair just fixed)."""
    replaced = {fileio_norm(r.file) for r in new
                if r.ok and getattr(r, "file", "")}
    return [r for r in results
            if not (r.ok and getattr(r, "file", "")
                    and fileio_norm(r.file) in replaced)] + new


def _static_enabled(cfg) -> bool:
    """The static gate is on unless explicitly ablated."""
    return bool(getattr(cfg, "static_gate", True))


def _static_errors(root, cfg, files: list[str]) -> list[str]:
    """Fast static gate (ruff or AST fallback) over project-relative paths.
    Returns [] when disabled; never raises -- a broken gate must not sink a
    task it cannot check."""
    if not _static_enabled(cfg) or not files:
        return []
    try:
        from . import spec as spec_mod
        return spec_mod.run_fast_static_gate(
            root, [root / f for f in files],
            timeout=float(getattr(cfg, "static_gate_timeout", 3.0)))
    except Exception:
        return []


def _new_static_errors(root, cfg, files: list[str]) -> list[str]:
    """Static gate errors MINUS the ones the project already had before this
    step (`cfg._static_baseline`); with no baseline recorded every error is
    treated as new -- the safe direction for a gate, since pre-existing
    breakage in a file about to be tested wastes the subprocess anyway."""
    errors = _static_errors(root, cfg, files)
    if not errors:
        return []
    from . import spec as spec_mod
    baseline = getattr(cfg, "_static_baseline", None)

    def fresh(items):
        if baseline is None:
            return list(items)
        try:
            return [e for e in items
                    if spec_mod.static_error_key(e) not in baseline]
        except Exception:
            return list(items)

    new = fresh(errors)
    if new and getattr(cfg, "auto_heal_imports", True):
        # Deterministic import auto-healing: F821 names that provably have
        # exactly one import are fixed on disk here, then the gate is re-run.
        # Only what still fails reaches the model repair round.
        try:
            paths = [Path(f) if Path(f).is_absolute() else root / f
                     for f in files]
            after, receipts = spec_mod.heal_static_gate_errors(
                root, paths, new, cfg)
            for receipt in receipts:
                print(f"  {receipt}", file=sys.stderr)
            if receipts:
                workspace.log_event(root, "import auto-heal: " +
                                    "; ".join(receipts), "INFO")
                new = fresh(after)
        except Exception:
            pass
    return new


def _capture_static_baseline(root, cfg) -> None:
    """Record the static errors the project ALREADY has (whole-tree scan), so
    only problems the current step introduces can trigger the fast repair
    microloop. Bounded and cheap on repeat steps: the gate caches results per
    file mtime/size, so untouched files cost nothing from step 2 onward."""
    if not _static_enabled(cfg):
        cfg._static_baseline = None
        return
    try:
        from . import spec as spec_mod
        errors = spec_mod.run_fast_static_gate(
            root, spec_mod.list_python_files(root),
            timeout=float(getattr(cfg, "static_gate_timeout", 3.0)))
        cfg._static_baseline = {spec_mod.static_error_key(e) for e in errors}
    except Exception:
        cfg._static_baseline = None


def _frontend_enabled(cfg) -> bool:
    """The DOM contract gate is on unless explicitly ablated. It is also a
    no-op for repos without HTML/JS, so enabling it by default is free
    there -- verify_frontend_contracts returns [] before opening a file."""
    return bool(getattr(cfg, "frontend_contracts", True))


def _frontend_errors(root, cfg, files: list[str]) -> list[str]:
    """Deterministic HTML/JS DOM contract check over the touched files
    (asset links, DOM id queries, JS parse). Never raises: a broken gate
    must not sink the task it cannot check."""
    if not _frontend_enabled(cfg) or not files:
        return []
    try:
        from . import spec as spec_mod
        return spec_mod.verify_frontend_contracts(root, files)
    except Exception:
        return []


def _new_frontend_errors(root, cfg, files: list[str]) -> list[str]:
    """Frontend contract errors MINUS the ones the project already had
    before this task (`cfg._frontend_baseline`); with no baseline recorded,
    every error counts as new -- the safe direction for a gate."""
    errors = _frontend_errors(root, cfg, files)
    if not errors:
        return []
    from . import spec as spec_mod
    baseline = getattr(cfg, "_frontend_baseline", None)

    def fresh(items):
        if baseline is None:
            return list(items)
        try:
            keys = {spec_mod.contract_error_key(e) for e in baseline}
            return [e for e in items
                    if spec_mod.contract_error_key(e) not in keys]
        except Exception:
            return [e for e in items if e not in baseline]

    new = fresh(errors)
    if new and getattr(cfg, "auto_heal_imports", True):
        # Deterministic ES module import auto-healing: unambiguous missing
        # sibling imports identified by _js_module_link_errors are healed on
        # disk (tree-sitter validated) and re-checked before spending a repair.
        try:
            after, receipts = spec_mod.heal_js_module_imports(
                root, files=files, errors=new, cfg=cfg)
            for receipt in receipts:
                print(f"  {receipt}", file=sys.stderr)
            if receipts:
                workspace.log_event(root, "js import auto-heal: " +
                                    "; ".join(receipts), "INFO")
                new = fresh(after)
        except Exception:
            pass
    return new


def _capture_frontend_baseline(root, cfg) -> None:
    """Record the frontend contract errors the project ALREADY has (whole
    tree scan), so only breakage the current task introduces can gate it.
    Cheap: regexes over a handful of HTML/JS files, one `node --check` each
    at most, and a repo without a frontend returns before any of it."""
    if not _frontend_enabled(cfg):
        cfg._frontend_baseline = None
        return
    try:
        from . import spec as spec_mod
        cfg._frontend_baseline = {
            spec_mod.contract_error_key(e)
            for e in spec_mod.verify_frontend_contracts(root, None)}
    except Exception:
        cfg._frontend_baseline = None


_DOM_EXT = (".html", ".htm", ".js", ".mjs", ".cjs", ".css")


def _dom_wanted(cfg) -> bool:
    """The headless DOM smoke check is opt-in: `verify_dom` says the project
    wants it, `allow_shell` grants the execution boundary (the check runs
    project code in a browser, like the test subprocess). One exception
    (BUG-1004): an explicit `ask --verify-dom` is per-run *user* consent to
    this one gate -- README/QUICKSTART both document the flag as sufficient
    ("or `ask --verify-dom`"), the same way a bare `pedacito domcheck`
    invocation is consent on its own -- so the CLI marks it on cfg."""
    if not bool(getattr(cfg, "verify_dom", False)):
        return False
    return bool(getattr(cfg, "allow_shell", False)) \
        or bool(getattr(cfg, "_dom_user_consent", False))


def _web_files(files: list[str]) -> bool:
    return any(str(f).lower().endswith(_DOM_EXT) for f in files or ())


def _capture_dom_baseline(root, cfg, edit_files: list[str]) -> None:
    """Snapshot the mounted DOM BEFORE this task's edits, so the gate can
    diff the surface the entry module used to mount (the BUG-216 shape).
    Also remembers which page errors already existed, so pre-existing rot
    is not blamed on the task. No-op unless the gate is on and web files
    are in play; a check that cannot run leaves no baseline (every error
    then counts as new -- the safe direction for a gate)."""
    cfg._dom_baseline = None
    if not _dom_wanted(cfg) or not _web_files(edit_files):
        return
    try:
        from . import domcheck
        ok, hint = domcheck.available()
        if not ok:
            print(f"  [domcheck: skipped -- {hint.splitlines()[0]}]",
                  file=sys.stderr)
            return
        res = domcheck.check(
            root, entry=getattr(cfg, "dom_entry", "") or "",
            timeout=int(getattr(cfg, "dom_timeout", 45) or 45))
        if res.error:
            print(f"  [domcheck baseline: {res.error.splitlines()[0]}]",
                  file=sys.stderr)
            return
        cfg._dom_baseline = {
            "snapshot": res.snapshot,
            "errors": {f.key for f in res.findings if f.severity == "error"},
        }
    except Exception:
        cfg._dom_baseline = None


def _new_dom_errors(root, cfg, files: list[str]) -> list[str]:
    """Headless DOM smoke errors MINUS the ones the page already had before
    this task. Never raises: a broken gate must not sink the task."""
    if not _dom_wanted(cfg) or not _web_files(files):
        return []
    try:
        from . import domcheck
        ok, hint = domcheck.available()
        if not ok:
            if not getattr(cfg, "_dom_hint_shown", False):
                cfg._dom_hint_shown = True
                print(f"  [domcheck: skipped -- {hint.splitlines()[0]}]",
                      file=sys.stderr)
            return []
        base = getattr(cfg, "_dom_baseline", None) or {}
        res = domcheck.check(
            root, entry=getattr(cfg, "dom_entry", "") or "",
            timeout=int(getattr(cfg, "dom_timeout", 45) or 45),
            baseline=base.get("snapshot"))
        if res.error:
            print(f"  [domcheck: {res.error.splitlines()[0]}]",
                  file=sys.stderr)
            return []
        base_keys = base.get("errors", set())
        return [f.message for f in res.findings
                if f.severity == "error" and f.key not in base_keys]
    except Exception:
        return []


def _contract_errors(root, cfg, files: list[str]) -> list[str]:
    """Cross-file Python/JS contract violations for the touched files (the
    scan itself covers the whole project, because an edit breaks importers)."""
    if not getattr(cfg, "interface_lock", True):
        return []
    try:
        from . import spec as spec_mod
        return spec_mod.verify_cross_file_contracts(root, files)
    except Exception:
        return []


def _new_contract_errors(root, cfg, files: list[str]) -> list[str]:
    """Contract errors MINUS pre-existing rot elsewhere. Line numbers are
    stripped from the identity, so a violation that merely shifted lines is
    not blamed on the current task. A pre-existing violation inside a file
    this task actually changed still reports: the task touched that file, and
    leaving it broken is not something the baseline excuses."""
    errors = _contract_errors(root, cfg, files)
    if not errors:
        return []
    baseline = getattr(cfg, "_contract_baseline", None)
    if baseline is None:
        return errors
    from . import fileio as fileio_mod
    touched = {fileio_mod.normalise_rel(str(f)) for f in files or ()}
    try:
        from . import spec as spec_mod
        return [e for e in errors
                if spec_mod.contract_error_key(e) not in baseline
                or spec_mod.contract_error_file(e) in touched]
    except Exception:
        return errors


def _capture_contract_baseline(root, cfg) -> None:
    """Record the cross-file contract errors already present before a task or
    step, so baseline diffing ignores pre-existing rot elsewhere."""
    if not getattr(cfg, "interface_lock", True):
        cfg._contract_baseline = None
        return
    try:
        from . import spec as spec_mod
        cfg._contract_baseline = {
            spec_mod.contract_error_key(e)
            for e in spec_mod.verify_cross_file_contracts(root, None)}
    except Exception:
        cfg._contract_baseline = None


# ------------------------------------------------ repair lesson distiller --
#
# A repair round that went broken -> fixed is the cheapest lesson the
# repository will ever produce, and it normally evaporates with the session.
# The distiller below is deliberately mechanical: it recognizes the failure
# SHAPES the gates and pytest actually print (a DOM query with no element, a
# link to a missing asset, an undefined name, a signature mismatch) and turns
# one into a single line for the ledger (`memory.record_anti_pattern`). No
# LLM prose: an invented lesson would teach the next session a fiction.

_LESSON_FILE_RE = re.compile(
    r"[\w./\\-]+\.(?:py|js|mjs|cjs|jsx|ts|tsx|html|htm|css)\b")


def _lesson_file(name: str) -> str:
    """A lesson's file path: forward slashes and leading `./` segments removed,
    case preserved (the ledger retires a lesson only when its file is gone, so
    the name must still match the file on disk). `str.lstrip("./")` would strip
    the leading dot of `.eslintrc.js` and lose the file."""
    p = fileio.normalise_rel(name)
    while p.startswith("./"):
        p = p[2:]
    return p


def distill_repair_lesson(failure_text: str, check_label: str = ""):
    """Reduce one gate/test failure blob to (kind, one-line lesson, files),
    or None when no known pattern explains it. `check_label` is the gate
    name the loop uses ("frontend contract", "static gate", "tests", ...);
    the first recognizable pattern wins, so findings text should put the
    root failure first. Deterministic: same text, same lesson, forever."""
    text = str(failure_text or "")
    if not text.strip():
        return None
    files = sorted({_lesson_file(m.group(0))
                    for m in _LESSON_FILE_RE.finditer(text)})[:6]

    def row(kind, msg):
        return (kind, " ".join(str(msg).split())[:200], files)

    m = re.search(r"queries DOM element '([^']+)'", text)
    if m:
        return row("dom_id_mismatch",
                   f"{m.group(1)} was queried by JS but no rendered element "
                   f"carried that id -- add the id to the template or query "
                   f"what exists ({', '.join(files[:3]) or 'see finding'})")
    m = re.search(r"references missing (?:script|stylesheet) asset "
                  r"'([^']+)'", text)
    if m:
        return row("asset_404",
                   f"HTML linked '{m.group(1)}' which did not exist -- "
                   f"create the file or fix the path before linking it")
    if re.search(r"does not parse|syntax error near", text):
        return row("js_syntax",
                   "an edit left a JS file unparsable -- a browser aborts "
                   "the whole file on one syntax error; re-check the parse "
                   "before calling a JS edit done")
    if re.search(r"SyntaxError", text):
        return row("syntax_error",
                   "an edit left a Python file unparsable -- the traceback "
                   "points at the line; fix the exact spot, not the outline")
    m = re.search(r"undefined name '([^']+)'|NameError: name '([^']+)'",
                  text)
    if m:
        name = m.group(1) or m.group(2)
        return row("avoid_missing_import",
                   f"'{name}' was used without being imported or defined at "
                   f"that point -- import it where it is used, do not patch "
                   f"around the name later")
    m = re.search(r"cannot import name '?([^'\s]+)'?"
                  r"|No module named '?([\w.\-]+)'?", text)
    if m:
        name = m.group(1) or m.group(2)
        return row("missing_export",
                   f"'{name}' could not be imported from the module that was "
                   f"expected to provide it -- check the exporter's real "
                   f"names (and the installed packages) before importing")
    m = re.search(r"unexpected keyword argument '?([^'\s]+)'?"
                  r"|(?:takes|accepts) \d+ positional arguments?", text)
    if m:
        detail = (f"keyword {m.group(1)}" if m.group(1)
                  else "the positional count")
        return row("signature_contract",
                   "a call site did not match the function's real signature "
                   f"(the caller disagreed on {detail}) -- reconcile caller "
                   f"and callee in one step, never by renaming around it")
    m = re.search(r"has no attribute '?([^'\s:]+)'?", text)
    if m:
        return row("missing_symbol",
                   f"the module had no attribute '{m.group(1)}' at runtime "
                   f"-- grep the exporter for the real name before calling it")
    m = re.search(r"(?:FAILED|ERROR) ([\w./]+\.py::[\w.\[\]\-:]+)", text)
    if m:
        return row("broke_a_test",
                   f"the last change made {m.group(1)} fail -- read what it "
                   f"pins before editing the code it guards")
    if re.search(r"^FAILED |^E +assert", text, re.M):
        return row("failing_assertion",
                   "an assertion the task was supposed to satisfy still "
                   "failed -- run the test after the edit, not just before")
    return None


def _note_repair_failure(pending: list, kind: str, text: str,
                         seen: set | None = None,
                         recordable: bool = True) -> None:
    """Remember a gate/test failure for later lesson distillation; the first
    entry per kind wins (root causes precede follow-on noise). `seen`, when
    given, collects the distilled failure kind even when no repair can follow
    (`recordable` false): the lesson-effectiveness counters (R5) count every
    failure a run observed, not only the ones it could mend."""
    if seen is not None:
        lesson = distill_repair_lesson(text, kind)
        if lesson is not None:
            seen.add(lesson[0])
    if not recordable:
        return
    if any(k == kind for k, _ in pending):
        return
    pending.append((kind, str(text or "")[:6000]))


def _note_lesson_run(root, cfg, kinds, injected=None) -> None:
    """One observed check run against the lesson ledger's effectiveness
    counters (R5); a retirement is surfaced as an event, and nothing here may
    raise into the loop."""
    try:
        from . import memory
        out = memory.note_lesson_run(root, kinds, cfg, injected=injected)
        if out.get("retired"):
            events_mod.get(root, cfg).emit(
                "lesson_retired", lessons=out["retired"],
                reason="no avoided failure in "
                       f"{memory._LESSON_RETIRE_RUNS} observed runs")
    except Exception:
        pass


def _note_baseline_lesson_run(root, cfg, baseline) -> None:
    """The pre-edit baseline is an observed check run too (BUG-213): count it
    against the lesson ledger even when the task later applies nothing and
    the post-edit path never runs. Kinds come from the same distillation the
    post-edit gates use; `injected` is the ledger as it stood before the run,
    so a lesson distilled mid-run is not counted against its own baseline."""
    if baseline is None:
        return
    ok, out, _desc = baseline
    kinds: set = set()
    if not ok:
        lesson = distill_repair_lesson(out, "tests")
        if lesson is not None:
            kinds.add(lesson[0])
    _note_lesson_run(root, cfg, kinds, _injected_lessons(root))


def _injected_lessons(root) -> set | None:
    """The lesson ids a run's prompt carries: the live ledger at run start.
    None when the ledger cannot be read (count every live lesson then)."""
    try:
        from . import memory
        return {r.get("lesson_id") for r in memory.load_anti_patterns(root)}
    except Exception:
        return None


def _record_repair_lesson(root, cfg, pending: list,
                          only_kinds=None, skip_kinds=None) -> int:
    """Distill each remembered failure whose kind is in scope into the
    ledger and remove it from `pending`. Gate findings are in scope once the
    deterministic re-check passes; test findings need a green test run.
    Returns the number of lessons recorded."""
    n = 0
    keep = []
    for kind, text in pending:
        k = str(kind).strip().lower()
        if (only_kinds is not None and k not in only_kinds
                or skip_kinds is not None and k in skip_kinds):
            keep.append((kind, text))
            continue
        lesson = distill_repair_lesson(text, k)
        if lesson is not None:
            try:
                from . import memory
                if memory.record_anti_pattern(root, lesson[0], lesson[1],
                                              lesson[2], cfg) is not None:
                    n += 1
            except Exception:
                pass
    pending[:] = keep
    return n


def _restore_rejected_files(root, backup_dir, rels, idx, cfg):
    """BUG-218: put the files a failed gate rejected back to their pre-task
    bytes, so an unrepairable violation is not left sitting in the tree.

    A file the task MODIFIED is restored from the snapshot (first copy wins,
    so these are the pre-task bytes); a file the task CREATED has no snapshot
    copy and is removed instead. Returns (restored, deleted, idx). A None
    backup_dir (review/dry runs, or a caller with no snapshot) is a graceful
    no-op. Never raises: a rollback that cannot run must not sink the report
    of the failure that triggered it."""
    restored: list[str] = []
    deleted: list[str] = []
    if backup_dir is None:
        return restored, deleted, idx
    from . import security
    backup_dir = Path(backup_dir)
    root = Path(root)
    try:
        created = set(fileio.created_manifest(backup_dir).read_text(
            encoding="utf-8").splitlines())
    except OSError:
        created = set()
    for rel in dict.fromkeys(rels):
        rel = fileio.normalise_rel(rel)
        if not rel:
            continue
        try:
            dest = security.resolve_safe_path(root, rel)
        except Exception:
            continue
        src = backup_dir / rel
        if src.is_file():
            try:
                fileio.copy_file_atomic(src, dest)
                restored.append(rel)
            except OSError:
                continue
        elif rel in created and dest.is_file():
            try:
                dest.unlink()
                deleted.append(rel)
            except OSError:
                continue
            # A deleted file's stale index entry dies on the next refresh;
            # re-indexing a path that no longer exists only logs a warning.
            continue
        else:
            continue
        try:
            idx = index_mod.reindex_file(idx, rel, cfg)
        except Exception:
            pass
    return restored, deleted, idx


def _run_autofix_pass(root, cfg, results, reply, changed):
    """Run the opt-in §3.4 autofix ladder before the model repair rounds.

    A retained candidate is already committed to disk by `run_autofix`; here we
    only refresh the in-memory `Result.after` buffers for the kept files so the
    syntax gate and any later review diff see the fixed content, and note the
    outcome in the reply. A discarded candidate (or any exception) leaves
    `results`/`reply` untouched so the ordinary model repair proceeds.
    """
    try:
        from . import autofix as autofix_mod
        from . import env_resolve as env_resolve_mod
        env = env_resolve_mod.resolve_environment(root)
        deadline = getattr(cfg, "_task_deadline", 0) or None
        result = autofix_mod.run_autofix(root, changed, cfg=cfg, env=env,
                                         deadline=deadline)
    except Exception as exc:
        try:
            workspace.log_event(root, f"autofix skipped: {exc}", "WARN")
        except Exception:
            pass
        return results, reply
    if not result.verified:
        return results, reply
    kept = set(result.kept)
    refreshed: list = []
    for r in results:
        rel = fileio_norm(getattr(r, "file", ""))
        if getattr(r, "ok", False) and rel and rel in kept:
            try:
                path = security_mod.resolve_safe_path(root, rel)
                text, _style = fileio.read(path)
                r.after = text
                r.message = (str(getattr(r, "message", "") or "").rstrip()
                             + f" [autofix: {result.reason}]")
            except Exception:
                pass
        refreshed.append(r)
    note = (f"\n\n[Pedacito autofixed {len(kept)} file(s) before repair: "
            f"{result.reason}.]")
    return refreshed, (reply.rstrip() + note if reply.strip() else reply)


def _post_edit_tests(root, cfg, agent, idx, task, results, reply, backup_dir,
                     baseline, edit_files, collect=None):
    """After edits were written, run the impacted tests (`verify_tests`).

    Before every subprocess run the dry-applied buffers are parsed in-process:
    a NEW syntax error rejects the hunk immediately and only the parser
    diagnostic + 10-line window goes to repair. Once the patch parses, the
    fast static gate (TECHNIQUES.md #19) scans the changed files on disk for
    NEW undefined names / bad `__all__` entries / misplaced imports -- also
    zero subprocess, zero tokens. Then failures are diffed by stable test
    identity against the pre-edit baseline (`new` / `fixed` / `pre-existing` /
    `still_failing`), and any `new` failure repairs even when the total count
    did not rise. Repair receives a structured
    Traceback Repair Packet, not the raw output tail.

    Returns (results, reply, tests_ok); tests_ok is None when the tests did
    not run (verification off, or no test command), else the last run's
    pass/fail -- surfaced as TaskResult.tests_ok for `ask --strict` and the
    eval scorer. `collect`, when given, receives every gate finding the run
    could NOT repair: the edits still landed, so `applied` stays true, and a
    caller that reports the task finished has to be told the last gate verdict
    was red (one RPC `ask` answered `applied: true` for a module graph that
    could not link).

    The in-process gates are NOT gated on `allow_shell`: they parse and
    scan files in this process (TECHNIQUES.md #19/#23), and the only
    subprocess they would have protected -- pytest -- already declines to
    run without the flags. A config that disabled the shell used to disable
    every check too, so an `ask` build could report "applied and verified"
    while its ES module graph could not even link; now the gates always
    vet, and only the test subprocess honours the flags. BUG-1006 amended
    that last sentence: a test command the USER granted (`--verify` naming
    one, or `test_command`) carries its own consent -- the final gate and
    the ledger already run and record exactly that command every task, so
    the ladder may too.
    """
    tests_wanted = bool(getattr(cfg, "verify_tests", False)) \
        and (bool(getattr(cfg, "allow_shell", False))
             or _test_run_consent(cfg))
    gates_wanted = (_static_enabled(cfg) or _frontend_enabled(cfg)
                    or bool(getattr(cfg, "interface_lock", True))
                    or _dom_wanted(cfg))
    if not (tests_wanted or gates_wanted):
        _note_tests_not_run(root, cfg, task)
        return results, reply, None, False
    from . import verify as verify_mod

    def unresolved(kind: str, errors) -> None:
        """Hand the caller the gate errors this run ended on."""
        if collect is None:
            return
        for e in list(errors)[:20]:
            text = getattr(e, "message", None) or str(e)
            collect.append(verify_mod.Finding(kind, "error", "", str(text)))

    def unresolved_tests(severity, diff, score, out, triage=None) -> None:
        """Record a red suite as a finding so task.findings / RPC blockers name
        it even without --strict (BUG-323). A purely pre-existing failure is a
        warning (R2: it predates this task); a new failure is an error. A red
        run that could not be attributed (WP-4105D) stays an error and says so."""
        if collect is None:
            return
        if triage is not None and triage.unverified:
            msg = (f"the test run is red and NOT verified "
                   f"({triage.category}): {triage.summary or 'no attributable failure'}")
        elif diff.new:
            msg = (f"{len(diff.new)} new test failure(s) introduced by this "
                   f"task; {len(diff.still_failing)} still failing")
        elif score > 0:
            msg = (f"the test suite is still red: {score} failure(s), "
                   f"{len(diff.pre_existing)} pre-existing")
        else:
            msg = "the test suite is still failing after this task"
        tail = [ln for ln in (out or "").strip().splitlines() if ln.strip()][-6:]
        if tail:
            msg += "\n" + "\n".join(tail)
        collect.append(verify_mod.Finding("tests", severity, "", msg))

    def rollback_frontend(errors) -> None:
        """BUG-218: the frontend gate ended unresolved -- restore the files it
        rejected to their pre-task bytes so a broken module graph is not left
        in the tree. The wording above stays honest (the edits DID land); this
        is the cleanup at the failure exit, not a claim the write was refused.
        """
        nonlocal idx
        from . import spec as spec_mod
        rels = [spec_mod.contract_error_file(e) for e in errors]
        rels = [r for r in rels if r]
        if not rels:
            return
        restored, deleted, idx = _restore_rejected_files(
            root, backup_dir, rels, idx, cfg)
        if restored or deleted:
            workspace.log_event(
                root, "frontend gate unresolved: restored "
                      f"{len(restored)} rejected file(s) to their pre-task "
                      f"bytes and removed {len(deleted)} created file(s) so "
                      "the broken contract is not left in the tree", "WARN")
            print(f"  [frontend gate: could not repair; rolled back "
                  f"{len(restored)} modified and removed {len(deleted)} "
                  "created file(s) to the pre-task state]", file=sys.stderr)

    changed = sorted({r.file for r in results if r.ok and r.file}) or edit_files
    if getattr(cfg, "autofix_enabled", False) and changed:
        # §3.4 autofix ladder: try the project's own deterministic fixers on an
        # independent shadow copy BEFORE the model repair rounds. A retained
        # candidate is committed to disk here, so the gates below re-check the
        # fixed content and the model is only asked to repair what remains.
        results, reply = _run_autofix_pass(root, cfg, results, reply, changed)
        changed = (sorted({fileio_norm(r.file) for r in results if r.ok and r.file})
                   or edit_files)
    base_out = (baseline[1] if baseline is not None and not baseline[0]
                else "")
    base_failures = failure_identities(base_out)
    base_score = _failure_score(base_out) if base_out else 0
    reruns: dict = {}              # node id -> verdict, one rerun per node (WP-4105D)
    quarantine, q_warnings = test_classify.load_quarantine(root)
    for warning in q_warnings:
        workspace.log_event(root, f"flakes: {warning}", "WARN")
    max_rounds = max(0, int(getattr(cfg, "repair_rounds", 2) or 0))
    tests_ok = None
    preexisting_only = False       # the red suite is entirely pre-existing (R2)
    last_out = ""                  # the final test subprocess output, if any
    last_worse = False             # ...and whether THIS task made it worse
    pending_lesson: list = []      # (kind, text) this task broke and can mend
    lesson_kinds: set = set()      # every distilled failure kind this run saw
    injected = _injected_lessons(root)   # lessons this run's prompt carried
    for round_ in range(max_rounds + 1):
        syntax = verify_mod.check_syntax(results)
        if syntax:
            tests_ok = False
            _note_repair_failure(pending_lesson, "syntax",
                                 "\n".join(str(f.message) for f in syntax),
                                 seen=lesson_kinds,
                                 recordable=round_ < max_rounds)
            print(f"  [syntax: {len(syntax)} new syntax error(s) -- "
                  f"test/lint subprocess skipped until the patch parses]",
                  file=sys.stderr)
            workspace.log_event(
                root, "syntax gate rejected the patch; subprocess checks "
                      "skipped", "ERROR")
            events_mod.get(root, cfg).emit(
                "tests", ok=False, score=len(syntax), baseline=base_score,
                round=round_, syntax=True, new=len(syntax), fixed=0,
                still_failing=0, pre_existing=len(base_failures),
                violations=[str(f.message)[:200] for f in syntax[:10]])
            if round_ >= max_rounds:
                unresolved('syntax', syntax)
                break
            fixed = agent._repair_from_findings(task, reply, syntax, results)
            if not fixed:
                unresolved('syntax', syntax)   # BUG-311: keep the blocker
                break
            new = agent._apply_once(fixed, False, backup_dir)
            if not any(r.ok for r in new):
                unresolved('syntax', syntax)
                break
            results = _merge_repair_results(results, new)
            reply = (reply.rstrip() + "\n\n" + fixed)
            changed = sorted({fileio_norm(r.file) for r in results if r.ok})
            continue

        # Static gate (TECHNIQUES.md #19): a NEW undefined name / bad
        # `__all__` / import-placement error costs a micro-repair round here
        # instead of seconds of test subprocess -- and the subprocess never
        # starts while the patch cannot even resolve its own names.
        static = _new_static_errors(root, cfg, changed)
        if static:
            tests_ok = False
            _note_repair_failure(pending_lesson, "static",
                                 "\n".join(static), seen=lesson_kinds,
                                 recordable=round_ < max_rounds)
            print(f"  [static gate: {len(static)} new undefined-name/import "
                  f"error(s) -- tests/lint subprocess skipped until they "
                  f"are fixed]", file=sys.stderr)
            workspace.log_event(
                root, "static gate rejected the patch; test/lint "
                      "subprocess skipped", "ERROR")
            events_mod.get(root, cfg).emit(
                "tests", ok=False, score=len(static), baseline=base_score,
                round=round_, static=True, new=len(static), fixed=0,
                still_failing=0, pre_existing=len(base_failures))
            if round_ >= max_rounds:
                unresolved('static', static)
                break
            finding = verify_mod.Finding(
                "static", "error", "",
                "after your edits the changed files fail the fast static "
                "gate: these names do not exist at the point they are used "
                "(a missing import, a typo, or a bad `__all__` entry). No "
                "test or lint process was run. Fix these first:\n"
                + "\n".join(f"- {e}" for e in static[:20]))
            fixed = agent._repair_from_findings(task, reply, [finding],
                                                results)
            if not fixed:
                unresolved('static', static)    # BUG-311: keep the blocker
                break
            new = agent._apply_once(fixed, False, backup_dir)
            if not any(r.ok for r in new):
                unresolved('static', static)
                break
            results = _merge_repair_results(results, new)
            reply = (reply.rstrip() + "\n\n" + fixed)
            changed = sorted({fileio_norm(r.file) for r in results if r.ok})
            continue

        # DOM contract gate (TECHNIQUES.md #23): a broken <script src>, a
        # query for an id no template defines, or a JS file that does not
        # parse never raises here -- the browser is the only thing that
        # notices, and it is not watching. The verifier checks exactly what
        # the browser would enforce at load, for free, before the
        # subprocess: an asset that 404s or an id typo costs a micro-repair
        # round now instead of an invisible runtime `null` later.
        fe = _new_frontend_errors(root, cfg, changed)
        if fe:
            tests_ok = False
            # a lesson only when a repair can follow; the kind still counts
            _note_repair_failure(pending_lesson, "frontend contract",
                                 "\n".join(fe), seen=lesson_kinds,
                                 recordable=round_ < max_rounds)
            print(f"  [frontend gate: {len(fe)} new DOM contract violation(s)"
                  f" -- tests/lint subprocess skipped until they are fixed]",
                  file=sys.stderr)
            # BUG-218: this gate runs AFTER the write (post-write gate +
            # repair, BUG-323): "rejected the patch" read as if nothing had
            # landed while the violating file sat on disk. Say what is true.
            workspace.log_event(
                root, "frontend contract gate found violations in the "
                      "applied edits; test/lint subprocess skipped (the "
                      "files stay written pending repair)", "ERROR")
            # BUG-318: name the violations, not just their count, so the run
            # log and event stream record what was rejected even when the
            # repair produces nothing.
            for v in fe[:10]:
                workspace.log_event(
                    root, f"frontend contract: {str(v)[:200]}", "ERROR")
            events_mod.get(root, cfg).emit(
                "tests", ok=False, score=len(fe), baseline=base_score,
                round=round_, frontend=True, new=len(fe), fixed=0,
                still_failing=0, pre_existing=len(base_failures),
                violations=[str(e)[:200] for e in fe[:10]])
            if round_ >= max_rounds:
                unresolved('frontend', fe)
                rollback_frontend(fe)
                break
            finding = verify_mod.Finding(
                "frontend", "error", "",
                "after your edits the HTML/JS breaks a DOM contract the "
                "browser enforces at page load (a missing asset, a query "
                "for an element id that is never rendered, or JS that does "
                "not parse). No test or lint process was run. Fix these "
                "first:\n"
                + "\n".join(f"- {e}" for e in fe[:20]))
            fixed = agent._repair_from_findings(task, reply, [finding],
                                                results)
            if not fixed:
                unresolved('frontend', fe)      # BUG-311: keep the blocker
                rollback_frontend(fe)
                break
            new = agent._apply_once(fixed, False, backup_dir)
            if not any(r.ok for r in new):
                unresolved('frontend', fe)
                rollback_frontend(fe)
                break
            results = _merge_repair_results(results, new)
            reply = (reply.rstrip() + "\n\n" + fixed)
            changed = sorted({fileio_norm(r.file) for r in results if r.ok})
            continue

        # Cross-file contract gate: a changed signature plus a
        # test caller that still passes the old arity is proven broken here,
        # statically -- running pytest only to watch it TypeError wastes the
        # whole subprocess, so the gate vetoes the spawn like the frontend
        # gate above does.
        ce = _new_contract_errors(root, cfg, changed)
        if ce:
            tests_ok = False
            # a lesson only when a repair can follow; the kind still counts
            _note_repair_failure(pending_lesson, "contract", "\n".join(ce),
                                 seen=lesson_kinds,
                                 recordable=round_ < max_rounds)
            print(f"  [contract gate: {len(ce)} new cross-file contract "
                  f"violation(s) -- tests/lint subprocess skipped until they "
                  f"are fixed]", file=sys.stderr)
            workspace.log_event(
                root, "interface contract gate rejected the patch; "
                      "test/lint subprocess skipped", "ERROR")
            events_mod.get(root, cfg).emit(
                "tests", ok=False, score=len(ce), baseline=base_score,
                round=round_, contract=True, new=len(ce), fixed=0,
                still_failing=0, pre_existing=len(base_failures),
                violations=[str(e)[:200] for e in ce[:10]])
            if round_ >= max_rounds:
                unresolved('contract', ce)
                break
            finding = verify_mod.Finding(
                "contract", "error", "",
                "after your edits a call site cannot satisfy the real "
                "signature in the module it imports from (a renamed, "
                "missing or arity-changed interface; test callers included). "
                "No test or lint process was run. Fix these first:\n"
                + "\n".join(f"- {e}" for e in ce[:20]))
            fixed = agent._repair_from_findings(task, reply, [finding],
                                                results)
            if not fixed:
                unresolved('contract', ce)      # BUG-311: keep the blocker
                break
            new = agent._apply_once(fixed, False, backup_dir)
            if not any(r.ok for r in new):
                unresolved('contract', ce)
                break
            results = _merge_repair_results(results, new)
            reply = (reply.rstrip() + "\n\n" + fixed)
            changed = sorted({fileio_norm(r.file) for r in results if r.ok})
            continue

        # Headless DOM smoke gate (TECHNIQUES.md #45, opt-in): the static
        # gates above prove the contracts the browser enforces at LOAD;
        # this one proves the page actually MOUNTS. It runs the entry page
        # in headless Chromium and diffs the mounted surface against the
        # pre-edit snapshot, so a replaced entry module that silently stops
        # rendering (BUG-216) costs a micro-repair round here instead of
        # shipping an empty page. Behind verify_dom AND allow_shell.
        de = _new_dom_errors(root, cfg, changed)
        if de:
            tests_ok = False
            _note_repair_failure(pending_lesson, "dom", "\n".join(de),
                                 seen=lesson_kinds,
                                 recordable=round_ < max_rounds)
            print(f"  [dom gate: {len(de)} headless DOM error(s) -- "
                  f"tests/lint subprocess skipped until they are fixed]",
                  file=sys.stderr)
            workspace.log_event(
                root, "headless DOM gate rejected the patch; test/lint "
                      "subprocess skipped", "ERROR")
            events_mod.get(root, cfg).emit(
                "tests", ok=False, score=len(de), baseline=base_score,
                round=round_, dom=True, new=len(de), fixed=0,
                still_failing=0, pre_existing=len(base_failures),
                violations=[str(e)[:200] for e in de[:10]])
            if round_ >= max_rounds:
                unresolved('dom', de)
                break
            finding = verify_mod.Finding(
                "dom", "error", "",
                "after your edits the entry page fails the headless DOM "
                "smoke check (an uncaught page error, an asset/module that "
                "no longer loads, or UI that was mounted before the edit "
                "is now gone). No test or lint process was run. Fix these "
                "first:\n"
                + "\n".join(f"- {e}" for e in de[:20]))
            fixed = agent._repair_from_findings(task, reply, [finding],
                                                results)
            if not fixed:
                unresolved('dom', de)           # BUG-311: keep the blocker
                break
            new = agent._apply_once(fixed, False, backup_dir)
            if not any(r.ok for r in new):
                unresolved('dom', de)
                break
            results = _merge_repair_results(results, new)
            reply = (reply.rstrip() + "\n\n" + fixed)
            changed = sorted({fileio_norm(r.file) for r in results if r.ok})
            continue

        if pending_lesson:
            # Every gate this task tripped is green again (the deterministic
            # re-check above just passed): distill what the failure was into
            # the lesson ledger while the evidence is fresh. Test failures
            # are deferred until a green test run proves the fix.
            _record_repair_lesson(root, cfg, pending_lesson,
                                  skip_kinds={"tests"})
        if not tests_wanted:
            # Every in-process gate is green; the test subprocess is not
            # configured to run (verify_tests/allow_shell off). Report
            # honestly that no tests executed rather than pretending the
            # green gates are tests.
            _note_lesson_run(root, cfg, lesson_kinds, injected)
            _note_tests_not_run(root, cfg, task)
            return results, reply, tests_ok, False
        got = _run_tests(idx, cfg, changed)
        if got is None:
            # Boundary open (tests_wanted), nothing to run: say so honestly.
            _note_lesson_run(root, cfg, lesson_kinds, injected)
            _note_tests_not_run(root, cfg, task, granted=True)
            return results, reply, tests_ok, False
        ok, out, desc = got
        last_out = out
        if not ok:
            _note_repair_failure(pending_lesson, "tests", out,
                                 seen=lesson_kinds,
                                 recordable=round_ < max_rounds)
        elif pending_lesson:
            # fail -> pass, proven by this green run: distill what it was,
            # including any gate finding that preceded the passing tests
            _record_repair_lesson(root, cfg, pending_lesson)
        tests_ok = ok
        score = 0 if ok else _failure_score(out)
        current = failure_identities(out)
        diff = classify_failures(base_out, out)
        detail = ""
        if diff.new or diff.fixed or diff.still_failing:
            detail = (f"; {len(diff.new)} new, {len(diff.fixed)} fixed, "
                      f"{len(diff.still_failing)} still failing")
        print(f"  [tests: {'passed' if ok else 'FAILED'} -- {desc}{detail}]",
              file=sys.stderr)
        workspace.log_event(root, f"tests {'passed' if ok else 'failed'}: "
                                  f"{desc}", "INFO" if ok else "ERROR")
        events_mod.get(root, cfg).emit(
            "tests", ok=ok, score=score, baseline=base_score, round=round_,
            new=len(diff.new), fixed=len(diff.fixed),
            still_failing=len(diff.still_failing),
            pre_existing=len(diff.pre_existing))
        if ok:
            break
        # Only a failing test that a rerun showed to be a real, attributable
        # failure of this change reaches repair (WP-4105D). Flakes, infra and
        # anything unattributable stop here as unverified: the run stays red.
        triage = _triage_red_run(root, cfg, idx, changed, out, base_out, desc,
                                 diff, current, reruns, quarantine, round_)
        repair_diff = FailureDiff(
            new=list(triage.real), fixed=list(diff.fixed),
            still_failing=list(diff.still_failing),
            pre_existing=list(diff.pre_existing))
        should_repair = bool(triage.real)
        last_worse = should_repair
        # Baseline-aware verdict (R2): with no in-run baseline to diff
        # against, the cross-run command history is the only baseline there
        # is. When it recorded this exact command failing before the run,
        # the failure pre-dates the edit -- do not spend a repair round on
        # it. An in-run baseline stays authoritative: classify_failures then
        # separates new from pre-existing identities precisely.
        if should_repair and not base_out \
                and testsel.PRE_EXISTING_LABEL in desc:
            should_repair = False
            print("  [tests: failure pre-dates this run (recorded command "
                  "history); not auto-repaired]", file=sys.stderr)
        if not should_repair:
            preexisting_only = not ok and not triage.unverified
            if triage.unverified:
                print(f"  [tests: not repaired -- {triage.category}: "
                      f"{triage.summary or 'no attributable failure'}; the run "
                      "stays red and unverified]", file=sys.stderr)
            elif not ok and score > 0:
                # Honest exit: the suite is still red, but every remaining
                # failure predates this task. Say so instead of ending on a
                # silent FAILED line.
                print(f"  [tests: {score} failure(s) remain; the suite "
                      f"already had {base_score} before this task, so they "
                      "predate it and were not auto-repaired]", file=sys.stderr)
            unresolved_tests("error" if triage.unverified else "warn",
                             diff, score, last_out, triage)   # BUG-323
            break
        if round_ >= max_rounds:
            unresolved_tests("error", diff, score, last_out)  # BUG-323
            break
        packet = traceback_repair_packet(root, repair_diff, current=current)
        intro = "after your edits these tests fail (they did not before):\n"
        finding = verify_mod.Finding("tests", "error", "", intro + packet)
        fixed = agent._repair_from_findings(task, reply, [finding], results)
        if not fixed:
            unresolved_tests("error", diff, score, last_out)  # BUG-323
            break
        new = agent._apply_once(fixed, False, backup_dir)
        if not any(r.ok for r in new):
            unresolved_tests("error", diff, score, last_out)  # BUG-323
            break
        results = _merge_repair_results(results, new)
        reply = (reply.rstrip() + "\n\n" + fixed)
        changed = sorted({fileio_norm(r.file) for r in results if r.ok})
    if tests_ok is False and last_worse and last_out \
            and _SEMANTIC_FAILURE_RE.search(last_out):
        # The patch parses but does not WORK, and the repair rounds could not
        # mend it: a stronger model is the next rung, not a third local retry.
        _suggest_escalation(
            root, cfg,
            "tests still fail with runtime/semantic errors (a TypeError, a "
            "failed assertion, or a loop that never ends) after the repair "
            "rounds; the edits apply but the code does not work")
    _note_lesson_run(root, cfg, lesson_kinds, injected)
    return results, reply, tests_ok, preexisting_only


_LSP_GATE_MAX_BYTES = 1_048_576


@dataclass(frozen=True)
class _LSPGate:
    """Baseline state of the opt-in LSP diagnostics gate for one symbol batch.
    `texts` holds each file's pre-edit text ("" for a file the edit creates,
    None when it could not be read or is too large to compare)."""
    rels: tuple[str, ...]
    env: Any
    texts: dict
    before_status: str
    before: tuple
    remaining: float


def _lsp_gate_texts(root: Path, rels) -> dict:
    out: dict = {}
    for rel in rels:
        try:
            path = security_mod.resolve_safe_path(root, rel)
        except security_mod.SecurityError:
            out[rel] = None
            continue
        if not path.exists():
            out[rel] = ""
            continue
        try:
            if not path.is_file() or path.stat().st_size > _LSP_GATE_MAX_BYTES:
                out[rel] = None
            else:
                out[rel] = path.read_bytes().decode("utf-8", "replace")
        except OSError:
            out[rel] = None
    return out


def _lsp_gate_begin(root, cfg, files) -> _LSPGate | None:
    """Baseline diagnostics before a symbol batch lands. Returns None when the
    gate is off (lsp_diagnostics_gate and allow_lsp are both required)."""
    if not (getattr(cfg, "lsp_diagnostics_gate", False)
            and getattr(cfg, "allow_lsp", False)):
        return None
    from . import env_resolve, lsp as lsp_mod
    root = Path(root).resolve()
    rels = tuple(sorted({fileio_norm(f) for f in files if f}))
    env = env_resolve.resolve_environment(root)
    texts = _lsp_gate_texts(root, rels)
    deadline = time.monotonic() + lsp_mod.GATE_BUDGET_SECONDS
    status, before = lsp_mod.collect_gate_diagnostics(
        root, list(rels), cfg=cfg, environment=env, deadline=deadline)
    remaining = max(0.0, deadline - time.monotonic())
    return _LSPGate(rels, env, texts, status, before, remaining)


def _lsp_gate_finish(gate, root, cfg) -> list:
    """Actionable gate findings after the batch landed, as verify Findings
    (severity "error" feeds repair). The gate spends only the budget left from
    the baseline, so the edit's own time does not count against it. Any
    status other than ok is unverified: nothing is reported and an event
    records why. Never raises."""
    if gate is None:
        return []
    from . import lsp as lsp_mod, verify as verify_mod
    root = Path(root).resolve()
    status = gate.before_status
    after: tuple = ()
    actionable: tuple = ()
    try:
        if status == "ok":
            deadline = time.monotonic() + gate.remaining
            status, after = lsp_mod.collect_gate_diagnostics(
                root, list(gate.rels), cfg=cfg, environment=gate.env,
                deadline=deadline)
        if status == "ok":
            changed: dict = {}
            maps: dict = {}
            now = _lsp_gate_texts(root, gate.rels)
            for rel in gate.rels:
                old, new = gate.texts.get(rel), now.get(rel)
                if old is None or new is None:
                    status = "unavailable"
                    break
                changed[rel], maps[rel] = lsp_mod.diff_line_maps(old, new)
            if status == "ok":
                actionable = lsp_mod.diagnostics_delta(
                    gate.before, after, changed, line_maps=maps)
    except Exception:
        status = "unavailable"
    findings = []
    for d in actionable:
        findings.append(verify_mod.Finding(
            kind="lsp",
            severity="error" if d.severity in (1, 2) else "warn",
            file=d.file,
            message=f"{d.code or 'diagnostic'}: {d.message}",
            line=d.line,
            lines=(d.line, d.end_line)))
    try:
        events_mod.get(root, cfg).emit(
            "lsp_gate", status=status, files=len(gate.rels),
            before=len(gate.before), after=len(after),
            actionable=len(actionable), budget_s=lsp_mod.GATE_BUDGET_SECONDS)
    except Exception:
        pass
    return findings


def _post_edit_lint_incremental(root, cfg, agent, task, results, reply,
                                backup_dir, baseline):
    """The lint half of the post-edit check after a symbol microloop.

    `agent.apply()` owns this loop on the ordinary path; the incremental path
    writes hunks itself, so the outer loop runs the same baseline diff here
    once ALL symbols have landed. Returns (results, reply)."""
    if not getattr(cfg, "post_edit_check", True):
        return results, reply
    max_rounds = max(0, int(getattr(cfg, "repair_rounds", 2) or 0))
    for _round in range(max_rounds):
        changed = sorted({fileio_norm(r.file) for r in results
                          if r.ok and r.file})
        if not changed:
            break
        try:
            current = diagnostics.quick_findings(root, changed)
        except Exception:
            break
        found = diagnostics.new_findings(baseline, current)
        agent.active_diagnostics = found or {}
        if not found:
            break
        n = sum(len(v) for v in found.values())
        print(f"\n[incremental post-edit lint: {n} new finding(s) in the "
              "changed files; asking for a fix]", file=sys.stderr)
        fix = agent._chat_at(
            "temperature_repair",
            [
                {"role": "system", "content": agent._answer_sys(task)},
                {"role": "user", "content":
                    f"TASK\n{task}\n\n"
                    "Your incremental symbol edits applied, but a quick lint "
                    "of the changed files reports NEW problems they "
                    "introduced:\n"
                    f"{diagnostics.render_findings(found)}\n\n"
                    "SOURCE (current contents around each finding)\n\n"
                    f"{agent._diagnostics_source(found)}\n\n"
                    "Fix ONLY these problems with minimal search/replace "
                    "blocks -- copy the SEARCH text exactly from the SOURCE "
                    "above, byte for byte. Do not repeat edits that already "
                    "worked and do not refactor anything else. If a finding "
                    "is a false positive, reply with a single line: NOTHING "
                    "MORE."},
            ],
            max_tokens=cfg.max_tokens_answer,
            stream=True,
        )
        if getattr(agent.client, "stream_truncated", False):
            fix = agent._trim_dangling_block(fix)
        if not parse_edits(fix):
            break
        fresh = sorted({fileio_norm(e.file) for e in parse_edits(fix)}
                       - set(baseline))
        if fresh:
            try:
                baseline.update(diagnostics.quick_findings(root, fresh))
            except Exception:
                pass
        # A lint repair is a gate repair too: lock pre-existing test files
        # (#46), exempting the files the lint itself proved broken.
        agent._arm_repair_test_lock(task, found)
        new = agent._apply_once(fix, False, backup_dir)
        results = _merge_repair_results(results, new)
        if not any(r.ok for r in new):
            break
        reply = (reply.rstrip() + "\n\n" + fix) if reply.strip() else fix
    return results, reply


def fileio_norm(name: str) -> str:
    from . import fileio
    return fileio.normalise_rel(name)


# A task that forbids edits ("report only", "do not edit any file") has
# already been answered by prose; re-asking for edit blocks spends a model
# call the task explicitly ruled out. The phrases must read as task-level
# directives, not mentions: an edit task may legitimately add a "read-only
# mode" or change one file "without editing" another, and skipping the
# recovery there would strand it with a prose answer. A false negative only
# costs the old re-ask (the model then declares "No changes"); a false
# positive loses edits.
_NO_EDIT_TASK_RE = re.compile(
    r"(?im)\bdo not\s+(?:write|edit|modify|change)"
    r"(?:\s+or\s+(?:write|edit|modify|change))?\s+any\s+(?:file|files|"
    r"anything)(?=\s*(?:$|[.,;:!?\n]))"
    r"|\bwithout\s+(?:editing|writing|changing)\s+(?:any\s+)?"
    r"(?:files?|anything)\b"
    r"|\bno\s+edits?\s+to\s+(?:any\s+)?(?:files?|anything)\b"
    r"|(?:^|[.,;:\n])\s*(?:no\s+edits?|read[- ]only|report\s+only)"
    r"(?=\s*(?:$|[.,;:!?\n]))"
    r"|\band\s+report\s+only(?=\s*(?:$|[.,;:!?\n]))")


def _report(results, dry_run: bool):
    """Print each edit Result to stderr, with its unified diff in dry-run mode."""
    if not results:
        print("\n[no edit blocks in the reply]", file=sys.stderr)
        return
    print("", file=sys.stderr)
    for r in results:
        mark = "ok " if r.ok else "FAIL"
        print(f"[{mark}] {r.file}: {r.message}", file=sys.stderr)
        if r.ok and dry_run and r.diff:
            print(r.diff)


# Answer-phase recovery. Small models sometimes emit another agent framework's
# tool-call syntax (Anthropic `<invoke name="read">`, `<tool_call>` /
# `<function=read>`) instead of search/replace blocks, usually because the
# task needs source the gather phase never read. The recovery parses those
# read requests, honours them from disk, and re-asks for the edit blocks.
_READ_INVOKE_RE = re.compile(
    r"<invoke\b[^>]*name\s*=\s*['\"]?read['\"]?[^>]*>(.*?)</invoke>",
    re.DOTALL | re.IGNORECASE)
# A reply can be cut off before the closing tag, so match up to the next
# opening tag or end-of-reply rather than requiring closure.
_READ_TOOLCALL_RE = re.compile(
    r"<tool_call>(.*?)(?=</tool_call>|<tool_call>|\Z)",
    re.DOTALL | re.IGNORECASE)
_TOOLCALL_IS_READ_RE = re.compile(
    r"function\s*=\s*read\b|<read\b|['\"]name['\"]\s*:\s*['\"]read['\"]",
    re.IGNORECASE)
# Qwen-style native tool calls often use a bare <read> wrapper and a
# <file_path> tag rather than Anthropic's <invoke>/<parameter> pair.
_READ_BLOCK_RE = re.compile(
    r"<read\b[^>]*>(.*?)(?=</read>|<read\b|\Z)",
    re.DOTALL | re.IGNORECASE)
_FILEPATH_PARAM_RE = re.compile(
    r"<parameter\b[^>]*name\s*=\s*['\"]file_path['\"][^>]*>\s*(.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE)
_FILEPATH_EQ_PARAM_RE = re.compile(
    r"<parameter\s*=\s*file_path>\s*(.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE)
_FILEPATH_TAG_RE = re.compile(
    r"<file_path\b[^>]*>\s*(.*?)\s*(?:</file_path>|(?=<)|\Z)",
    re.DOTALL | re.IGNORECASE)


def _file_param(body: str) -> str:
    """First file path in a tool-call body, across the common dialects."""
    for rx in (_FILEPATH_PARAM_RE, _FILEPATH_EQ_PARAM_RE, _FILEPATH_TAG_RE):
        m = rx.search(body or "")
        if m:
            path = m.group(1).strip().strip("`'\" \t\r\n")
            # A degenerate reply can repeat one path for thousands of
            # characters; that is not a real request.
            return path if len(path) <= 1000 else ""
    return ""


_MAX_RECOVERY_ROUNDS = 2
_MAX_RECOVERY_FILES = 4
_MAX_FILE_CHARS = 20000
_MAX_EXTRA_CHARS = 30000


def _read_requests(reply: str) -> list[str]:
    """Extract the file paths a reply asked to `read` via tool-call syntax,
    in reply order and de-duplicated. Empty when the reply has none."""
    found: list[tuple[int, str]] = []
    for m in _READ_INVOKE_RE.finditer(reply):
        path = _file_param(m.group(1))
        if path:
            found.append((m.start(), path))
    for m in _READ_TOOLCALL_RE.finditer(reply):
        body = m.group(1)
        # A body that names another tool is skipped so a requested write is
        # not mistaken for a read.
        if not _TOOLCALL_IS_READ_RE.search(body):
            continue
        path = _file_param(body)
        if path:
            found.append((m.start(), path))
    # A bare <read> block without a <tool_call> wrapper (Qwen's native shape).
    for m in _READ_BLOCK_RE.finditer(reply):
        path = _file_param(m.group(1))
        if path:
            found.append((m.start(), path))
    seen: set[str] = set()
    uniq = []
    for _, path in sorted(found):
        if path not in seen:
            seen.add(path)
            uniq.append(path)
    return uniq


def _honour_reads(root: Path, requests: list[str]) -> str:
    """Read the requested files from disk and return them as one SOURCE block.

    Paths are constrained to the project root: anything outside it (or
    unreadable) is reported as unavailable instead. Per-file and total
    character caps keep the re-ask inside the model's window."""
    try:
        base = root.resolve()
    except OSError:
        base = root.absolute()
    blocks: list[str] = []
    total = 0
    for rel in requests[:_MAX_RECOVERY_FILES]:
        name = str(rel).strip().strip("`'\"")
        if not name:
            continue
        try:
            p = (root / name).resolve()
            inside = p.is_relative_to(base)
        except (OSError, ValueError):
            inside = False
        if not inside or not p.is_file():
            blocks.append(f"### {name}\n[not readable]\n")
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            blocks.append(f"### {name}\n[not readable]\n")
            continue
        if len(text) > _MAX_FILE_CHARS:
            text = text[:_MAX_FILE_CHARS] + "\n[file truncated]\n"
        room = _MAX_EXTRA_CHARS - total
        if room <= 0:
            break
        text = text[:room]
        total += len(text)
        blocks.append(f"### {name}\n{text}\n")
    return "\n\n".join(blocks)


def _recover_no_edit_reply(root: Path, agent, task: str, gathered: str,
                           reply: str, *, dry_run: bool, backup_dir,
                           answer_system: str | None = None):
    """Recover an answer reply that parsed to zero edit blocks (at most two
    rounds): if the model asked to READ files via tool-call syntax, honour
    the reads from disk and re-ask with that source; otherwise re-ask with an
    explicit 'no tools here, emit edit blocks' instruction. Returns
    (results, reply) from the first re-ask that produces blocks or declares
    "No changes", or None when every round came back empty.

    A task that asks for no edits at all ("report only", "do not edit") is
    already answered: one more model call cannot produce blocks it forbade,
    so the recovery is skipped and the prose answer stands."""
    if _NO_EDIT_TASK_RE.search(task or ""):
        return None
    asked = _read_requests(reply)
    for _round in range(_MAX_RECOVERY_ROUNDS):
        extra = _honour_reads(root, asked) if asked else ""
        asked = []
        if extra:
            print("\n[reply asked to read more files; read them and "
                  "re-asking]", file=sys.stderr)
        else:
            print("\n[reply had no edit blocks; re-asking with the edit "
                  "protocol]", file=sys.stderr)
        base = answer_system or "You are a Python code assistant."
        system = (base.rstrip()
                  + ("\n\n" if answer_system else " ")
                  + "In this FINAL answer phase no tools are available: you "
                  + "cannot read or write files directly. Complete the task "
                  + "with search/replace edit blocks for the source provided "
                  + "below.")
        task2 = (task + "\n\n"
                 "Your previous reply contained no edit blocks"
                 + (" (it only asked to read more files, which cannot be "
                    "fetched in this phase)" if extra else "")
                 + ". Produce the edit blocks now. If -- and only if -- no "
                   "edit is the correct outcome (the work is already done, "
                   "or editing would violate the goal), reply with ONE line "
                   "starting 'No changes:' and the reason; never emit a "
                   "block whose SEARCH and REPLACE are identical.")
        src = (gathered or "").strip()
        if extra:
            src = src + ("\n\n" if src else "") + \
                "EXTRA SOURCE READ FOR YOU\n\n" + extra
        raw = agent.answer(task2, src, system=system)
        results2, reply2 = agent.apply(raw, dry_run=dry_run, task=task,
                                       gathered=src, backup_dir=backup_dir)
        if results2:
            return results2, reply2
        if _is_noop(reply2):
            # The re-ask took the "No changes: <reason>" escape hatch; hand
            # the declaration back so run_task/run_loop treat it as a
            # decision rather than pressing the model again.
            return [], reply2
        asked = _read_requests(raw)
    return None


def _review(root: Path, results, task: str, cfg, backup_dir, stamp: str,
            clauses=None, clause_texts=None) -> list:
    """Serve the side-by-side review page and return the results as they
    finally stand (the page writes on Apply, so this re-stamps each result).
    `clauses` (file -> clause ids) and `clause_texts` come from the requirement
    tracker so each file shows the task clause it serves (R8)."""
    diffs = review_mod.diffs_from_results(results, root, clauses=clauses,
                                          clause_texts=clause_texts)
    if not diffs:
        print("\n  No review page: no edit block applied cleanly.", file=sys.stderr)
        _report(results, dry_run=True)
        return results
    nonce = review_mod.new_nonce()
    page = review_mod.build_page(diffs, task, cfg.model, cfg.active_profile,
                                 nonce, interactive=True, stamp=stamp,
                                 clause_texts=clause_texts)
    saved = review_mod.save(root, page, stamp)

    def writer():
        return review_mod.write_files(root, diffs, backup_dir)

    print(f"  review page: {saved}", file=sys.stderr)
    outcome, detail = review_mod.run(
        page, nonce, writer, port=cfg.review_port,
        open_browser=cfg.open_browser, timeout=cfg.review_timeout,
        on_url=lambda u: print(f"  waiting for sign-off at {u}", file=sys.stderr),
        diffs=diffs, task=task, clause_texts=clause_texts)

    if outcome == "applied":
        print(f"  {detail}", file=sys.stderr)
        # The sign-off is the Task's write: record its checkpoint now, so the
        # next run branches from exactly what the user accepted.
        try:
            checkpoints_mod.create_checkpoint(
                root, f"review: {task[:180]}", mode="task")
        except Exception as e:
            workspace.log_event(root, f"review checkpoint skipped: {e}", "WARN")
        # A reviewer may sign off on only some files (the GUI can), dropping
        # the rest from `diffs`; only those were written.
        written = {d.rel for d in diffs}
        return [replace(r, message="applied via review")
                if r.ok and r.file in written
                else replace(r, message="not written (left out of the review)")
                if r.ok else r
                for r in results]
    print(f"  {detail}", file=sys.stderr)
    return [replace(r, message=f"not written ({outcome})") if r.ok else r
            for r in results]


INVESTIGATE_SYSTEM = """\
You are investigating a software problem, not editing code. You have a map of
the project and whatever source and command output the gather phase collected.
Report what you observed: what reproduces the problem, the exact evidence
(command output, file and line ranges), the likely cause, and the smallest next
fix someone should make. Before naming a file as the cause, confirm the running
program actually loads that code -- who imports it, who instantiates it;
projects often contain dead or duplicated definitions that look authoritative
but never execute. The reverse also holds: a file the entry point loads (a
script tag, an import) but whose exports nothing ever calls is dead code --
being loaded is not being used, and the load reference is itself part of what
dead-code removal cleans up, never proof of life. When runtime evidence is
unavailable, reading the bootstrap module for calls into that file IS valid
usage evidence: classify from it instead of deferring the question to a live
check. Prefer evidence from the running program over source
reading, and if the live path is unconfirmed, lead "still unknown" with that.
Do not call tools -- none are available in this phase.
Do not emit search/replace blocks. Be concrete and concise; if the evidence is
inconclusive, say what is still unknown and what to check next.
For a layout or styling diagnosis, confirm the selector actually matches the
live element (id/class in the markup) and that no script sets the value
directly -- a rule whose selector matches nothing explains nothing. Also
confirm which document the route actually serves: a template that is never
rendered, or a stylesheet the served HTML never links, cannot be the cause.
"""

# Appended to INVESTIGATE_SYSTEM for verification-titled investigations (see
# run_investigation_task).
_VERDICT_BRIEF = """

Your report must END with a verdict as its very last line, exactly one of
these two forms:
VERDICT: PASSED -- only when EVERY part of the task's claim was confirmed by
live evidence you gathered in this session.
VERDICT: FAILED: <one-sentence reason> -- in every other case. A partially
confirmed claim is FAILED. Reading source code is never live evidence. Do not
hedge: the loop reads this line mechanically and acts on it.
For a visual or layout claim, quote a concrete before/after measurement from
the running page (element size, computed style, or scrollHeight vs
innerHeight). "No scroll bars appeared" is not evidence that the play area
changed; a number that is identical before and after means the fix did
nothing."""

_TOOL_SYNTAX = ("<tool_call", "<invoke", "<function=", "<read", '"name":')


def _looks_like_tool_calls_only(reply: str) -> bool:
    """True when a prose answer is really just agent-framework tool syntax
    (`<tool_call>` / `<invoke>` blocks emitted despite "no tools here")."""
    low = (reply or "").lower()
    if not low.strip() or not any(t in low for t in _TOOL_SYNTAX):
        return False
    # Strip tags, paths, and JSON punctuation (a runaway reply can be mostly
    # one repeated path); then a few repeated tool-call opens are enough.
    opens = len(re.findall(r"<tool_call\b|<invoke\b|<function\s*=", reply,
                           re.IGNORECASE))
    plain = re.sub(r"<[^>]+>", " ", reply)
    plain = re.sub(r"\S*[/\\]\S*", " ", plain)      # paths
    plain = re.sub(r"[{}\[\]\"]", " ", plain)
    words = re.findall(r"[A-Za-z]{3,}", plain)
    if opens and len(words) < 40 and not _VERDICT_RE.search(reply):
        # One tool call wrapped in a sentence of intent ("I'll run the game
        # ...") is not a report. Real prose quoting tool syntax usually has
        # enough surrounding report or a verdict line; a lone call does not.
        return True
    return opens >= 3 or len(words) < 8


# A verification report still to be performed: it opens with a promise
# ("I'll run the game..."), names no observed result, and -- in the answer
# phase there are no tools -- can never happen. The harness used to accept
# exactly that shape as a PASSED check.
_PROMISE_HEAD_RE = re.compile(
    r"^[\s#*\-\u2022\d.)]*(?:i['\u2019]ll|i will|i['\u2019]m going to|"
    r"i am going to|let me|first,? i['\u2019]ll|next,? i['\u2019]ll)\b")
_EDIT_BLOCK_MARK_RE = re.compile(r"(?m)^<{5,}\s*SEARCH\b")
_REPORT_EVIDENCE_RE = re.compile(
    r"\b(?:observed|observes|measured|measures|captured|shows?|showed|"
    r"found|confirmed|verified|reported|rendered|loaded|screenshot|"
    r"canvas|pixels?|dimensions?)\b|\d", re.IGNORECASE)


def _report_prose(reply: str) -> str:
    """The reply minus tool-call payloads and fenced code, for checks that
    must not mistake JSON numbers or `<tool_call>` text for observations."""
    text = re.sub(r"<tool_call\b[^>]*>.*?(?:</tool_call>|\Z)", " ", reply or "",
                  flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<invoke\b[^>]*>.*?(?:</invoke>|\Z)", " ", text,
                  flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r"```.*?```", " ", text, flags=re.DOTALL)


def _looks_like_unexecuted_promise(reply: str) -> bool:
    """True for a report that only promises to run the check. The answer
    phase has no tools, so no observation can follow the promise; a verdict
    line or a concrete result marker makes it a real report."""
    text = _report_prose(reply).strip()
    if not text:
        return False
    if _EDIT_BLOCK_MARK_RE.search(text):
        return False        # a plan line ("I'll rename X ...") before blocks
    norm = _normalized_reply(text)
    if not _PROMISE_HEAD_RE.match(norm):
        return False
    if _VERDICT_RE.search(text):
        return False
    return not _REPORT_EVIDENCE_RE.search(norm)


def _unusable_report(reply: str) -> bool:
    """True when an investigation reply is not a report at all (empty,
    tool-call syntax, or a bare promise to run the check later)."""
    return (not (reply or "").strip()
            or _looks_like_tool_calls_only(reply)
            or _looks_like_unexecuted_promise(reply))


def _prose_collapse(reply: str) -> bool:
    """True when a prose reply's tail degenerated into a phrase loop.

    Signal: low vocabulary novelty (unique/total words) in the last 1500
    chars. Paths and code are deliberately NOT stripped: their uniqueness is
    what separates a parallel file table (~0.28) from a loop (~0.10). A hit
    only triggers a regeneration. Not applied to build replies, whose edit
    content (sprite grids...) is legitimately repetitive."""
    words = re.findall(r"\w+", (reply or "")[-1500:].lower())
    if len(words) < 60:
        return False
    return len(set(words)) / len(words) < 0.22


def run_investigation_task(root: Path, cfg, agent, idx, task: str, *,
                           answer_system: str | None = None) -> TaskResult:
    """Run a reproduction/diagnosis task whose deliverable is a report.

    No edits are expected or applied: the gather phase uses the normal tools
    (including `run`), then the answer phase reports findings in prose. If the
    model answers with tool-call syntax or nothing, the answer is retried once
    and then replaced by a stand-in built from the steps log and gathered
    evidence."""
    when = workspace.stamp()
    workspace.log_event(root, f"investigate: {safe_text(task, cfg)}")
    todo_mod.clear(root)
    print("\n--- looking things up ---", file=sys.stderr)
    gathered = agent.gather(task)
    print("--- reporting ---\n", file=sys.stderr)
    system = answer_system or INVESTIGATE_SYSTEM
    if answer_system is None and _is_verification_task(_task_title(task)):
        # Keyed off the clean title: the goal block quotes the spec, whose
        # verify words would otherwise put the brief on every investigation.
        system += _VERDICT_BRIEF
    reply = agent.answer(task, gathered,
                         system=system,
                         include_edit_format=False, stream=True)
    collapsed = _prose_collapse(reply)
    if collapsed:
        print("  [the answer collapsed into repeating phrases; "
              "regenerating it once]", file=sys.stderr)
        workspace.log_event(root, "answer collapsed into phrase repetition; "
                            "regenerating once", "WARN")
    if _unusable_report(reply) or collapsed:
        # One retry absorbs transient empty completions and regenerates a
        # phrase-loop collapse before it reaches the context notes every
        # later task reads.
        if not collapsed:
            print("  [the answer phase returned no usable prose; "
                  "retrying once]", file=sys.stderr)
            workspace.log_event(root, "answer phase produced no prose "
                                "(empty, tool-calls-only, or a promise to "
                                "run the check later); retrying once", "WARN")
        retry = agent.answer(task, gathered,
                             system=system,
                             include_edit_format=False, stream=True)
        if (retry or "").strip() and not _unusable_report(retry):
            reply = retry
    steps_log = list(getattr(agent, "steps_log", []) or [])
    if _unusable_report(reply):
        steps = "\n".join(steps_log or ["(no steps)"])
        reply = ("No prose report was produced; the lookup phase recorded:\n\n"
                 + steps)
        evidence = (gathered or "").strip()
        if evidence:
            reply += "\n\nEVIDENCE\n" + evidence[:8000]
        workspace.log_event(root, "answer phase produced no prose even "
                            "after the retry; synthesized a steps-log "
                            "stand-in", "ERROR")
    session = None
    if cfg.log_sessions:
        session = workspace.write_session(root, safe_text(task, cfg),
                                          steps_log, safe_text(reply, cfg),
                                          [], None, when)
        print(f"[session: {session.relative_to(root)}]", file=sys.stderr)
    return TaskResult([], reply, True, [], idx, session, steps_log,
                      live_ok=getattr(agent, "live_evidence_ok", None),
                      live_kinds=_live_kinds_of(agent))


def _suggest_escalation(root, cfg, reason: str, *, pending: bool = False) -> None:
    """Record ONE escalation reason (the first wins) and surface it as an
    `escalation_suggested` event even when no `escalate` role is configured --
    the signal is useful to a human or an outer orchestrator either way.
    `pending` stores the reason for the NEXT task (the per-profile zero-edit
    statistic is only known after this task's ladder has finished)."""
    slot = "_pending_escalation_reason" if pending else "_task_escalation_reason"
    if getattr(cfg, slot, ""):
        return
    setattr(cfg, slot, reason)
    try:
        events_mod.get(root, cfg).emit(
            "escalation_suggested", reason=reason,
            profile=getattr(cfg, "active_profile", "") or "")
    except Exception:
        pass
    print(f"  [escalation suggested: {reason}]", file=sys.stderr)


# Runtime/semantic failure shapes in test output: the patch PARSES but does
# not WORK (wrong types, missing names, failed assertions, a loop that never
# ends). SyntaxError is deliberately absent -- that is the syntax gate's job,
# and a parse failure is a different repair skill, not a stronger-model case.
_SEMANTIC_FAILURE_RE = re.compile(
    r"\bTypeError\b|\bReferenceError\b|\bRangeError\b|\bRecursionError\b"
    r"|\bAssertionError\b|\bassertion failed\b|\binfinite loop\b"
    r"|\bmaximum call stack\b|\bstack overflow\b", re.IGNORECASE)


def _emit_model_swap(root, cfg, swap: dict) -> None:
    try:
        events_mod.get(root, cfg).emit("model_swap", **swap)
    except Exception:
        pass


def _specialist_route(root, cfg, client, agent, idx, task, is_repair=False):
    """Task-boundary specialist routing (TECHNIQUES.md #50): pick the
    profile whose specialty tags fit the task, hand the VRAM over cleanly,
    and return the (possibly new) client and agent. Never raises: routing
    is an optimization, and any failure keeps the worker. `agent` may be
    None (run_loop builds it after routing); a replaced agent keeps its
    forced-read targets."""
    from . import roles as roles_mod
    try:
        if is_repair or not getattr(cfg, "auto_route_specialties", False):
            return client, agent
        resident = set()
        try:
            state = client.loaded_models() or {}
            resident = {m for m, st in state.items() if st == "loaded"}
        except Exception:
            pass
        resident.add(str(getattr(cfg, "model", "") or ""))
        name, tag = roles_mod.select_specialist_profile(
            task, idx, cfg, resident=resident)
        if not name:
            # No specialist for this task: give the card back to the worker
            # when a previous specialist still holds it.
            swap = roles_mod.restore_worker_client(cfg, client)
            if swap:
                _emit_model_swap(root, cfg, swap)
            return client, agent
        new_client = roles_mod.switch_specialist_client(cfg, client, name)
        if new_client is client:
            return client, agent
        model = str(getattr(new_client.cfg, "model", "") or "")
        role = getattr(new_client, "active_model_role", None)
        if isinstance(role, dict):
            role["specialty"] = str(tag or "")
        print(f"\n[specialist router: switching worker to profile '{name}' "
              f"({model}) for specialty '{tag}']", file=sys.stderr)
        workspace.log_event(root, f"specialist router: profile '{name}' "
                                  f"({model}) for specialty '{tag}'")
        try:
            events_mod.get(root, cfg).emit(
                "specialist_routed", profile=name, model=model, specialty=tag)
        except Exception:
            pass
        swap = getattr(new_client, "_swap_info", None)
        if swap:
            new_client._swap_info = None
            _emit_model_swap(root, cfg, swap)
        if agent is not None:
            from .agent import Agent
            fresh = Agent(idx, new_client, cfg)
            fresh.task_targets = getattr(agent, "task_targets", None)
            agent = fresh
        return new_client, agent
    except Exception as e:
        print(f"  [specialist routing skipped: {str(e)[:100]}]",
              file=sys.stderr)
        return client, agent


def _specialist_zero_edit_fallback(root, cfg, client, idx, task, tr, kw):
    """Retry a routed specialist's zero-edit task on the top coding specialist.

    BUG_HUNT #14 Finding #5: a specialist card (LFM2.5-24B-A2B for
    ``math_algorithms``) streamed unbounded hidden reasoning on a local server
    and finished Task 2 with 0 applied edits / 0 files written. The task then
    failed even though a strong coding specialist (Devstral-Small-2-24B,
    Qwen3-Coder-30B) was configured and idle. When the run was routed to a
    specialist profile and produced no edit at all, hand the task ONCE to the
    top-scoring local coding specialist that clears the ``edit_format``
    competence gate, then let the caller's normal failure path decide.

    Returns the original ``tr`` when routing did not happen, the task is not
    an edit task, the specialist did land edits, no coding fallback exists, or
    the fallback also produced nothing. Never raises."""
    from . import roles as roles_mod
    try:
        role = getattr(client, "active_model_role", None)
        if not isinstance(role, dict) or role.get("role") != "specialist":
            return tr
        if not (kw.get("apply") and not kw.get("review")
                and not kw.get("is_repair")
                and kw.get("answer_system") is None):
            return tr
        if any(r.ok for r in tr.results) or tr.changed:
            return tr
        if _is_noop(tr.reply) or task.strip() == _REMEDIATION_TITLE:
            # A coherent "nothing to change" and a repair's sanctioned
            # conflict report are not stalls: the loop's own no-op paths judge
            # their evidence, and overriding a deliberate refusal would be
            # worse than the fallback.
            return tr
        failed = str(role.get("profile") or "")
        fallback = roles_mod.select_coding_fallback(
            cfg, exclude=(failed,) if failed else ())
        if not fallback or fallback == failed:
            return tr
        fb = roles_mod.switch_specialist_client(cfg, client, fallback)
        if fb is client:
            return tr
        model = str(getattr(fb.cfg, "model", "") or "")
        print(f"\n[specialist router: profile '{failed}' finished with 0 "
              f"applied edits / 0 files written; retrying the task once on "
              f"the top-scoring coding specialist '{fallback}' ({model})]",
              file=sys.stderr)
        workspace.log_event(
            root, f"specialist zero-edit fallback: '{failed}' -> "
                  f"'{fallback}' ({model})", "WARN")
        try:
            events_mod.get(root, cfg).emit(
                "specialist_fallback", failed_profile=failed,
                profile=fallback, model=model)
        except Exception:
            pass
        swap = getattr(fb, "_swap_info", None)
        if swap:
            fb._swap_info = None
            _emit_model_swap(root, cfg, swap)
        from .agent import Agent
        tr2 = _run_ladder(root, cfg, fb, Agent(idx, fb, cfg), idx, task, **kw)
        if any(r.ok for r in tr2.results) or tr2.changed:
            return tr2
        print(f"  [coding-specialist fallback '{fallback}' produced no "
              f"applied edits either]", file=sys.stderr)
    except Exception as e:
        print(f"  [specialist zero-edit fallback skipped: {str(e)[:100]}]",
              file=sys.stderr)
    return tr


# ------------------------------------------- split-stage hybrid routing
# hybrid_routing runs a task on TWO models in sequence: a strong ARCHITECT
# plans first (Stage 1: root cause, target files, exact signatures, edge
# cases), then the fast EDITOR implements that plan (Stage 2). The blueprint
# rides ahead of the editor's gathered context, so the cheap model never has
# to rediscover the analysis. When the two models live on different LOCAL
# servers, the VRAM handoff evicts the outgoing server between stages
# (auto_evict_servers); hosted endpoints coexist and skip the handshake.

_HYBRID_SYSTEM = """\
You are the ARCHITECT in a split-stage coding pipeline. A faster editor model
will implement your plan verbatim in the next stage; it cannot ask you
questions, so the plan must be complete and precise. Do not write code edits
and do not use tools -- reason from the task text and the project map.

Reply with exactly one blueprint block in this shape:

HYBRID_BLUEPRINT
1. ROOT CAUSE / GOAL: what is actually wrong (or what must exist), in 1-3
   sentences.
2. TARGET FILES: the exact paths to change or create, and why each.
3. EXACT SIGNATURES & CONTRACTS: function/class signatures, argument and
   return types, and the invariants the edit must preserve.
4. EDGE CASES: failure modes, empty/None inputs, concurrency, encoding,
   backwards compatibility -- anything the editor could get wrong.
END_BLUEPRINT

Name real files and symbols from the map. Be specific enough that an editor
who has never seen this codebase can implement the change correctly."""


def hybrid_active(cfg) -> bool:
    """True when a distinct architect model is armed for split-stage runs:
    the toggle is on AND an architect model is selected that differs from
    the editor's (same-model 'hybrid' would just pay twice for one brain)."""
    model = str(getattr(cfg, "architect_model", "") or "").strip()
    return (bool(getattr(cfg, "hybrid_routing", False)) and bool(model)
            and model != str(getattr(cfg, "model", "") or "").strip())


def _architect_client(cfg, editor):
    """The cached Stage-1 architect client for (architect_provider,
    architect_model). Unreachable servers cache None and warn once, exactly
    like the scout: a dead architect degrades to a plain editor run, never a
    failed task. The architect keeps its OWN stats ledger (clean per-stage
    deltas) but reports calls into the editor's sink so run telemetry sees
    both stages."""
    from . import roles as roles_mod
    provider = str(getattr(cfg, "architect_provider", "") or "").strip()
    model = str(getattr(cfg, "architect_model", "") or "").strip()
    key = (provider, model)
    cache = cfg.__dict__.setdefault("_architect_clients", {})
    if key in cache:
        return cache[key]
    new_cfg = roles_mod.resolve_model_config(cfg, provider, model)
    if new_cfg is None:
        cache[key] = None
        return None
    from .llm import LLMClient
    try:
        client = LLMClient(new_cfg)
        client.check(warn=False)
        client.warmup()
    except LLMError as e:
        print(f"[hybrid] architect '{model}' is unavailable "
              f"({str(e).splitlines()[0][:100]}); the editor runs alone",
              file=sys.stderr)
        cache[key] = None
        return None
    client.active_model_role = {"profile": provider, "model": new_cfg.model,
                                "role": "architect", "specialty": ""}
    if hasattr(editor, "on_call"):
        client.on_call = editor.on_call
    cache[key] = client
    return client


def _hybrid_evict(cfg, from_url: str, to_url: str) -> None:
    """The VRAM handoff between two stages: when BOTH endpoints are local
    servers on different ports and auto_evict_servers is on, unload the
    outgoing server before the incoming model loads. Never raises."""
    from . import servers as servers_mod
    try:
        if not getattr(cfg, "auto_evict_servers", True):
            return
        if not (servers_mod.is_local_url(from_url)
                and servers_mod.is_local_url(to_url)):
            return
        if servers_mod.normalize_url(from_url) == servers_mod.normalize_url(to_url):
            return
        servers_mod.evict_other_local_servers(to_url, cfg)
    except Exception:
        pass


def _extract_blueprint(reply: str) -> str:
    """The blueprint body between the markers; a reply without markers is
    taken whole (a small architect that forgot the fence still planned)."""
    m = re.search(r"HYBRID_BLUEPRINT\s*\n(.*?)\n\s*END_BLUEPRINT",
                  reply or "", re.DOTALL)
    return (m.group(1) if m else (reply or "")).strip()


def _hybrid_stage1(root, cfg, client, agent, idx, task):
    """Stage 1: the architect writes the blueprint. Returns a dict with the
    blueprint text, the architect client and its per-stage usage, or None
    when the architect is unavailable/empty -- the caller then runs the
    ordinary single-model path."""
    arch = _architect_client(cfg, client)
    if arch is None:
        return None
    # The architect is built and cached raw; wrap only this run's use so its
    # planning exchange enters the transcript without giving a later run a
    # proxy bound to this run's (finished) capture.
    capture = getattr(client, "_capture", None)
    if capture is not None:
        arch = capture.wrap(arch)
    acfg = arch.cfg
    _hybrid_evict(cfg, str(getattr(cfg, "base_url", "") or ""),
                  str(getattr(acfg, "base_url", "") or ""))
    label = f"Hybrid Stage 1 (Architect): {acfg.model}"
    print(f"\n[{label}]", file=sys.stderr)
    workspace.log_event(root, label)
    try:
        events_mod.get(root, cfg).emit(
            "hybrid_stage", stage=1, role="architect", model=acfg.model,
            provider=str(getattr(cfg, "architect_provider", "") or ""),
            label=label)
    except Exception:
        pass
    try:
        if agent is not None:
            map_text, _tier = agent._map(task)
        else:
            map_text, _tier = idx.map_for_prompt(cfg, task=task)
    except Exception:
        map_text = ""
    user = (f"TASK:\n{task}\n\nPROJECT MAP:\n{map_text}\n\n"
            "Write the HYBRID_BLUEPRINT for this task now.")
    arch.site = "plan"
    before = _snapshot_stats(arch)      # after warmup: only the plan counts
    started = time.time()
    try:
        reply = arch.chat(
            [{"role": "system", "content": _HYBRID_SYSTEM},
             {"role": "user", "content": user}],
            max_tokens=int(getattr(cfg, "max_tokens_plan", 2000) or 2000),
            stream=False, cancel_event=getattr(cfg, "_cancel_event", None))
    except LLMError as e:
        print(f"[hybrid] architect planning failed "
              f"({str(e).splitlines()[0][:100]}); the editor runs alone",
              file=sys.stderr)
        return None
    blueprint = _extract_blueprint(reply)
    if not blueprint:
        print("[hybrid] architect returned an empty blueprint; the editor "
              "runs alone", file=sys.stderr)
        return None
    return {"blueprint": blueprint, "client": arch,
            "stats": _task_stats(arch, before, started),
            "model": str(acfg.model),
            "provider": str(getattr(cfg, "architect_provider", "") or "")}


def _hybrid_finish(root, cfg, client, hybrid: dict, tr: TaskResult) -> None:
    """Fold the architect's usage into the task's combined telemetry and
    emit the per-stage breakdown (the GUI result card and TUI receipt read
    tr.stats['hybrid'])."""
    a = hybrid.get("stats") or {}
    e = tr.stats or {}
    merged = dict(e)
    for k in ("calls", "prompt_tokens", "completion_tokens", "cached_tokens",
              "cache_write_tokens"):
        merged[k] = int(e.get(k, 0) or 0) + int(a.get(k, 0) or 0)
    merged["cost"] = round(float(e.get("cost", 0.0) or 0.0)
                           + float(a.get("cost", 0.0) or 0.0), 6)
    merged["seconds"] = round(float(e.get("seconds", 0.0) or 0.0)
                              + float(a.get("seconds", 0.0) or 0.0), 2)
    pt = merged.get("prompt_tokens", 0)
    merged["cache_hit"] = (round(merged.get("cached_tokens", 0) / pt, 3)
                           if pt else 0.0)
    merged["hybrid"] = {
        "architect": {"model": hybrid.get("model", ""),
                      "provider": hybrid.get("provider", ""), **a},
        "editor": {"model": str(getattr(cfg, "model", "") or ""),
                   "provider": str(getattr(cfg, "active_profile", "") or ""),
                   **{k: v for k, v in e.items() if k != "hybrid"}},
    }
    tr.stats = merged
    try:
        events_mod.get(root, cfg).emit("hybrid", **merged["hybrid"],
                                       calls=merged.get("calls", 0),
                                       prompt_tokens=merged.get("prompt_tokens", 0),
                                       completion_tokens=merged.get("completion_tokens", 0),
                                       cost=merged.get("cost", 0.0))
    except Exception:
        pass


# The interleaved engine spends one model call per turn -- reads, edits and
# test runs all ride the same conversation -- so the gather-era default of
# 30 calls is far too tight for it. When the user left max_total_model_calls
# at a fallback value, interleaved runs get this generous floor; an explicit
# user cap (file/profile/env/CLI) is always respected.
_INTERLEAVED_CALL_FLOOR = 120

_INTERLEAVED_LIVE_TOOLS = ("run", "run_test", "browser", "screenshot")


def _interleaved_steps(res) -> list[str]:
    """A lookup-style action log synthesized from the interleaved transcript,
    so session files, the failure-memory tail and the GUI keep their familiar
    numbered step lines ("1. read pedacito/loop.py"). Handles both the native
    tool-call channel and the JSON text protocol."""
    import json
    steps: list[str] = []
    for m in getattr(res, "transcript", None) or []:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        calls: list[tuple[str, dict]] = []
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") if isinstance(tc, dict) else None
            fn = fn if isinstance(fn, dict) else {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            calls.append((str(fn.get("name") or ""),
                          args if isinstance(args, dict) else {}))
        if not calls and isinstance(m.get("content"), str):
            try:
                body = json.loads(m["content"])
            except Exception:
                body = None
            if isinstance(body, dict):
                if isinstance(body.get("tool"), str):
                    args = body.get("arguments")
                    calls.append((body["tool"],
                                  args if isinstance(args, dict) else {}))
                for tc in body.get("tool_calls") or []:
                    if isinstance(tc, dict) and isinstance(tc.get("tool"), str):
                        args = tc.get("arguments")
                        calls.append((tc["tool"],
                                      args if isinstance(args, dict) else {}))
        for name, args in calls:
            if not name:
                continue
            target = str(args.get("file") or args.get("path")
                         or args.get("pattern") or args.get("command")
                         or args.get("query") or "")
            steps.append(f"{len(steps) + 1}. {name}"
                         + (f" {target[:120]}" if target else ""))
    return steps


def _run_interleaved_task(root: Path, cfg, client, agent, idx, task: str, *,
                          apply: bool = True, review: bool = False,
                          capture=None) -> TaskResult:
    """Run ONE task through the interleaved engine and adapt its result to
    the TaskResult contract run_loop, the CLI and the GUI already consume.

    run_task took the pre-run checkpoint already (checkpoint=None here);
    review/dry-run hand the staged edits to the SAME `_review` page the
    two_phase ladder uses, so sign-off, the GUI Review pane and the undo
    baseline behave identically. Events flow through `agent.events` (the
    JSONL log and the GUI feed both listen there); `on_event` stays None so
    nothing is emitted twice."""
    from . import config as config_mod
    from . import interleaved as interleaved_mod
    when = workspace.stamp()
    wants_write = apply or review
    backup_dir = (workspace.new_backup_dir(root, when, unique=True)
                  if wants_write and cfg.backup else None)
    t_started = time.time()
    usage0 = _snapshot_stats(client)
    workspace.log_event(root, f"task (interleaved): {safe_text(task, cfg)}")
    todo_mod.clear(root)          # a fresh checklist per task, like two_phase
    _fire(cfg, root, "pre_task", task=task[:500])
    try:
        if getattr(agent, "events", None) is None:
            agent.events = events_mod.get(root, cfg)
    except Exception:
        pass
    # Live evidence (run/run_test/browser/screenshot observations) comes from
    # the tool_result events; a temporary listener keeps the verdict without
    # parsing result text.
    live_kinds: list[str] = []

    def _live_note(kind, fields):
        if (kind == "tool_result" and fields.get("ok")
                and fields.get("name") in _INTERLEAVED_LIVE_TOOLS
                and fields["name"] not in live_kinds):
            live_kinds.append(fields["name"])

    elog = getattr(agent, "events", None)
    if elog is not None:
        elog.listeners.append(_live_note)
    # A generous call floor for fallback-valued caps; explicit caps stand.
    saved_calls = int(getattr(cfg, "max_total_model_calls", 0) or 0)
    sources = getattr(cfg, "sources", None)
    source = str(sources.get("max_total_model_calls", "default")) \
        if isinstance(sources, dict) else "default"
    raised = (not config_mod._is_explicit_limit(source)
              and saved_calls < _INTERLEAVED_CALL_FLOOR)
    if raised:
        cfg.max_total_model_calls = _INTERLEAVED_CALL_FLOOR
        if isinstance(sources, dict):
            sources["max_total_model_calls"] = "interleaved default"
    cancel_event = (getattr(cfg, "_cancel_event", None)
                    or getattr(client, "cancel_event", None))
    print("\n--- interleaved run ---", file=sys.stderr)
    try:
        res = interleaved_mod.run_interleaved(
            agent, task, apply=apply, review=review, checkpoint=None,
            cancel_event=cancel_event, on_event=None, backup_dir=backup_dir)
    finally:
        if elog is not None:
            try:
                elog.listeners.remove(_live_note)
            except ValueError:
                pass
        if raised:
            cfg.max_total_model_calls = saved_calls
            if isinstance(sources, dict):
                sources["max_total_model_calls"] = source
    if capture is not None:
        # _ClientProxy intercepts chat/chat_stream, not chat_tools: record the
        # whole conversation as one trajectory entry instead.
        try:
            capture.note("interleaved", res.transcript, res.summary)
        except Exception:
            pass
    results = list(res.edit_results or [])
    reply = (res.summary or "").strip()
    if res.error:
        reply = (reply + "\n\n" if reply else "") + f"[{res.error}]"
    applied = False
    if review:
        results = _review(root, results, task, cfg, backup_dir, when)
        applied = any(r.ok and "applied" in r.message for r in results)
    else:
        _report(results, dry_run=not apply)
        applied = bool(apply and any(r.ok for r in results))
        if apply and results and not applied:
            print(f"\n[NOTHING APPLIED: all {len(results)} edit target(s) "
                  "failed; zero files were changed]", file=sys.stderr)
            workspace.log_event(
                root, f"NOTHING APPLIED: all {len(results)} edit target(s) "
                      "failed", "ERROR")
    changed: list[str] = []
    if applied:
        for r in results:
            if not r.ok or str(r.message).startswith("not written"):
                continue
            if getattr(r, "deleted", False):
                try:
                    idx.forget(r.file)
                except Exception:
                    pass
                continue
            try:
                idx = index_mod.reindex_file(idx, r.file, cfg)
                changed.append(r.file)
            except Exception:
                pass               # a moved file re-indexes on next refresh
        agent.index = idx
    steps_log = _interleaved_steps(res)
    session = None
    if cfg.log_sessions:
        session = workspace.write_session(root, safe_text(task, cfg),
                                          steps_log, safe_text(reply, cfg),
                                          results, backup_dir, when)
        print(f"[session: {session.relative_to(root)}]", file=sys.stderr)
    stats = _task_stats(client, usage0, t_started)
    ok_n = sum(1 for r in results if r.ok)
    events_mod.get(root, cfg).emit(
        "task", task=task[:200], steps=len(steps_log), applied=applied,
        edits_ok=ok_n, edits_failed=len(results) - ok_n, changed=changed,
        findings=[], answer_mode="interleaved", folds=res.folds, **stats)
    events_mod.get(root, cfg).emit(
        "result", _full=True, task=task[:200], reply=reply, applied=applied,
        edits=[{"file": r.file, "ok": r.ok, "message": r.message,
                "diff": r.diff, "clauses": []} for r in results],
        clause_texts={}, unaccepted=[])
    try:
        security_mod.emit_post_run_notice(root, cfg, files=changed)
    except Exception:
        pass
    if stats.get("calls"):
        print(f"[task: {stats['calls']} calls, {stats['prompt_tokens']:,} "
              f"in / {stats['completion_tokens']:,} out tokens, cache hit "
              f"{stats['cache_hit']:.0%}"
              + (f", {stats['cache_write_tokens']:,} written to cache"
                 if stats.get("cache_write_tokens") else "")
              + (f", ${stats['cost']:.4f}" if stats.get("cost") else "")
              + f", {stats['seconds']:.0f}s]", file=sys.stderr)
    _fire(cfg, root, "post_task", task=task[:300], applied=applied,
          files=changed, ok=ok_n, failed=len(results) - ok_n)
    return TaskResult(results, reply, applied, changed, idx, session,
                      steps_log, backup_dir=backup_dir,
                      live_ok=bool(live_kinds) if live_kinds else None,
                      live_kinds=tuple(live_kinds) if live_kinds else None,
                      stats=stats, tests_ok=res.tests_green,
                      budget_hit=bool(getattr(res, "stop_reason", "") in (
                          "max_task_seconds", "max_task_completion_tokens",
                          "max_total_model_calls")))


def run_task(root: Path, cfg, client, agent, idx, task: str, *,
             apply: bool = True, review: bool = False,
             answer_system: str | None = None,
             is_repair: bool = False,
             pinned: list | None = None,
             images: list[dict] | None = None) -> TaskResult:
    """`_run_ladder`, plus failure memory: a task that ends without a usable
    edit is remembered (files looked at, why) so a similar later task is warned
    off the same dead end; a success clears the warning. Zero-edit runs are
    also counted per profile: once `escalation_zero_edit_runs` accumulate, the
    next task carries an escalation suggestion into its ladder.

    `pinned` is an optional explicit context list (`@file`, `Class.method`);
    the prompt's own @mentions are always parsed by Agent.gather.
    `images` is an optional list of normalized image attachment dicts.
    """
    from . import memory as memory_mod
    has_saved_pinned = hasattr(cfg, "_pinned_refs")
    saved_pinned = getattr(cfg, "_pinned_refs", None)
    if pinned is not None:
        cfg._pinned_refs = [str(p).lstrip("@").strip()
                            for p in pinned if str(p).strip()]
    has_saved_images = hasattr(cfg, "_task_images")
    saved_images = getattr(cfg, "_task_images", None)
    if images is not None:
        cfg._task_images = list(images)
    try:
        journal = None
        if bool(apply) and not review:
            # A dry run / review leaves the tree byte-identical, so it must not
            # create `.pedacito/journal/...` either.
            journal = _journal_open(root, cfg, task)
            if journal is not None:
                try:
                    cfg._task_journal = journal
                except Exception:
                    pass
        if _cancel_requested(cfg, client):
            _emit_run_cancelled(root, cfg, "task", task=(task or "")[:200])
            raise CancelledError("Stopped by user; no further work was done.")
        writes = bool(apply) and not review
        if apply or review:
            # A review task writes only after a human sign-off, but the state
            # before the proposal is exactly what an undo must restore, so the
            # baseline is recorded here and the accepted review adds its child.
            if writes:
                _recover_atomic(root, cfg)
            _checkpoint_task(root, cfg, task, "pre")
        if not getattr(cfg, "_auto_step", 0):
            # Standalone task (ask/chat/eval): no outer loop owns the static or
            # contract baseline, so drop any leftover one and let _run_task_once
            # record a fresh snapshot for this task.
            cfg._static_baseline = None
            cfg._contract_baseline = None
            cfg._dom_baseline = None
            cfg._frontend_baseline = None
            # Standalone tasks route here; run_loop routes before it builds the
            # agent (its _auto_step marker is set), so there is no double switch.
            client, agent = _specialist_route(root, cfg, client, agent, idx,
                                              task, is_repair)
        # Opt-in trajectory export: wrap the client the task will actually use
        # (and the agent that owns it) so every gather/answer/repair call is the
        # real transcript. Capture is diagnostics: any failure here is ignored.
        capture = None
        try:
            capture = trajectories.begin(root, cfg, task, client)
            if capture is not None:
                client, agent = capture.install(client, agent)
        except Exception:
            capture = None

        def _finish_capture(result):
            """Write the trajectory; every failure is the capture's, not the
            task's. Kept here so a broken hook can never skip the task's return."""
            try:
                trajectories.finish(capture, root, cfg, result)
            except Exception:
                pass
        # Split-stage hybrid: the architect plans before the editor runs. Repair
        # rounds and scaffold/answer_system tasks keep their single-model path.
        # The interleaved engine plans and edits in one conversation, so it
        # skips the hybrid handoff entirely.
        from . import config as config_mod
        interleaved = (answer_system is None
                       and config_mod.resolve_execution_mode(cfg) == "interleaved")
        hybrid = None
        if ((apply or review) and not is_repair and answer_system is None
                and not interleaved and hybrid_active(cfg)):
            hybrid = _hybrid_stage1(root, cfg, client, agent, idx, task)
        saved_mode = getattr(cfg, "answer_mode", "")
        if hybrid is not None:
            # Stage 1 -> 2 handoff: give the editor's server the VRAM, then
            # inject the blueprint. Continue mode reuses only the gather
            # conversation, so Stage 2 must rebuild its context to see the plan.
            _hybrid_evict(cfg, str(getattr(hybrid["client"].cfg, "base_url", "") or ""),
                          str(getattr(cfg, "base_url", "") or ""))
            label = f"Hybrid Stage 2 (Editor): {cfg.model}"
            print(f"\n[{label}]", file=sys.stderr)
            workspace.log_event(root, label)
            try:
                events_mod.get(root, cfg).emit(
                    "hybrid_stage", stage=2, role="editor",
                    model=str(getattr(cfg, "model", "") or ""),
                    provider=str(getattr(cfg, "active_profile", "") or ""),
                    label=label)
            except Exception:
                pass
            cfg._hybrid_blueprint = hybrid["blueprint"]
            cfg.answer_mode = "rebuild"
        try:
            if interleaved:
                tr = _run_interleaved_task(root, cfg, client, agent, idx, task,
                                           apply=apply, review=review,
                                           capture=capture)
            else:
                tr = _run_ladder(root, cfg, client, agent, idx, task, apply=apply,
                                 review=review, answer_system=answer_system,
                                 is_repair=is_repair)
        except BaseException:
            # A cancelled or failed run still leaves the exchanged messages that
            # did happen; write them before propagating the original exception.
            _finish_capture(None)
            raise
        finally:
            if hybrid is not None:
                cfg.answer_mode = saved_mode
                cfg._hybrid_blueprint = ""
        if hybrid is not None:
            _hybrid_finish(root, cfg, client, hybrid, tr)
        if not interleaved:
            # The zero-edit fallback re-runs _run_ladder on another model; the
            # interleaved engine already owns its own retries and escalation.
            tr = _specialist_zero_edit_fallback(
                root, cfg, client, idx, task, tr,
                dict(apply=apply, review=review, answer_system=answer_system,
                     is_repair=is_repair))
        if answer_system is not None or review or not getattr(cfg, "use_memory", True):
            if writes:
                _checkpoint_task(root, cfg, task, "post")
            _finish_capture(tr)
            return tr
        try:
            if any(r.ok for r in tr.results) or tr.changed or _is_noop(tr.reply):
                memory_mod.clear_attempts(root, task)
                memory_mod.clear_zero_edit(root, cfg)
                # The profile un-stuck itself: drop this task's lease strikes and
                # any escalation suggestion its earlier failures queued up.
                strikes = getattr(cfg, "_lease_strikes", None)
                if strikes:
                    strikes.pop(" ".join((task or "").split())[:200], None)
                if getattr(cfg, "_pending_escalation_reason", ""):
                    cfg._pending_escalation_reason = ""
            else:
                looked = [ln.split(None, 2)[-1][:60] for ln in tr.steps
                          if " read " in f" {ln} "][:4]
                why = ("fabricated edits" if tr.fabricated
                       else "no valid edit was produced")
                memory_mod.note_attempt(root, task, why, looked)
                threshold = int(getattr(cfg, "escalation_zero_edit_runs", 3) or 0)
                if threshold > 0:
                    n = memory_mod.note_zero_edit(root, cfg)
                    if n >= threshold:
                        profile = getattr(cfg, "active_profile", "") or "(default)"
                        _suggest_escalation(
                            root, cfg,
                            f"{n} consecutive zero-edit runs on profile "
                            f"'{profile}'; a stronger model may succeed where "
                            "this one keeps producing no usable edit",
                            pending=True)
        except Exception:
            pass
        if writes:
            _checkpoint_task(root, cfg, task, "post")
        _finish_capture(tr)
        return tr
    finally:
        # Record the terminal journal entry before the deadline is cleared, so
        # a task that ends (or crashes into this finally) is never left as an
        # incomplete task for crash recovery. The outcome reflects HOW the task
        # ended: a propagating cancellation is "cancelled", any other exception
        # "failed", and a clean return "completed".
        exc_type = sys.exc_info()[0]
        if exc_type is None:
            status = "completed"
        elif issubclass(exc_type, CancelledError):
            status = "cancelled"
        else:
            status = "failed"
        _journal_close(cfg, status=status)
        # The task's wall-clock budget is over: clear the deadline so the next
        # task's (or a post-task summary refresh's) requests are not clamped
        # by a stale deadline.
        try:
            cfg._task_deadline = 0.0
        except Exception:
            pass
        if pinned is not None:
            if has_saved_pinned:
                cfg._pinned_refs = saved_pinned
            elif hasattr(cfg, "_pinned_refs"):
                delattr(cfg, "_pinned_refs")
        if images is not None:
            if has_saved_images:
                cfg._task_images = saved_images
            elif hasattr(cfg, "_task_images"):
                delattr(cfg, "_task_images")


def _rollback_attempt(root: Path, cfg, attempt_tr: TaskResult | None,
                      idx=None):
    bdir = getattr(attempt_tr, "backup_dir", None) if attempt_tr else None
    cur_idx = idx if idx is not None else getattr(attempt_tr, "idx", None)
    if not bdir:
        return cur_idx
    bpath = Path(bdir)
    if not bpath.is_dir():
        return cur_idx
    restored = workspace.restore(root, bpath)
    workspace.delete_created(root, bpath)
    if cur_idx is not None:
        for rel in restored:
            try:
                cur_idx = index_mod.reindex_file(cur_idx, rel, cfg)
            except Exception:
                pass
    return cur_idx


def _run_ladder(root: Path, cfg, client, agent, idx, task: str, *,
                apply: bool = True, review: bool = False,
                answer_system: str | None = None,
                is_repair: bool = False) -> TaskResult:
    """`_run_task_once`, plus the bounded recovery ladder: an unreachable or
    failing worker falls over to the `fallback` role's model, and a task that
    ends with NO edit applied gets one more attempt on the `escalate` role's
    (stronger) model. A recorded escalation reason (two no-progress leases, a
    semantic test failure that outlived the repair rounds, or the per-profile
    zero-edit statistic) earns the same retry even when edits DID land -- a
    patch that applies but does not work is still a failed task. Each rung is
    used at most once per task, and only when that role is configured, so a
    local model's bad day cannot silently burn a hosted budget. When no
    distinct `escalate` role exists, a fired reason earns ONE same-model
    retry with the edit style flipped (#49) instead of nothing."""
    from . import roles as roles_mod
    from .llm import BudgetExceeded, CancelledError
    cfg._task_mutations = []      # a fresh gather-mutation ledger per task
    if apply and not review and not is_repair:
        # BUG-211/216: a task that names its test/DOM command owes that
        # check; repair prompts are harness diagnostics, not user text.
        _infer_task_checks(root, cfg, task)
    # Files that exist BEFORE this task's gather: the repair test-lock (#46)
    # protects exactly these test files, never ones the task itself creates.
    _files = getattr(idx, "files", None)
    cfg._task_preexisting = ({fileio.path_key(f) for f in _files}
                             if _files is not None else None)
    # A reason queued by the previous task's failure memory (the zero-edit
    # statistic) carries into this task; reasons recorded DURING this task
    # (lease strikes, semantic failures) join it, first reason wins.
    cfg._task_escalation_reason = getattr(cfg, "_pending_escalation_reason", "")
    cfg._pending_escalation_reason = ""
    kw = dict(apply=apply, review=review, answer_system=answer_system,
              is_repair=is_repair)
    # Proactive two-path escalation (arXiv:2611.16414): a structurally complex
    # task goes straight to the `escalate` role, skipping the worker attempt.
    # Off by default (cfg.proactive_escalation), so reactive-only escalation
    # is byte-for-byte unchanged.
    if getattr(cfg, "proactive_escalation", False):
        esc = roles_mod.client_for(cfg, "escalate", client)
        reason = roles_mod.proactive_escalation_reason(cfg, task, idx)
        if reason and esc is not client and apply and not review:
            print(f"\n[proactive escalation: {reason}; running on the "
                  "`escalate` role's model]", file=sys.stderr)
            workspace.log_event(root, f"proactive escalation to the escalate "
                                      f"role: {reason}", "WARN")
            from .agent import Agent
            cfg._task_mutations = []
            try:
                return _run_task_once(root, cfg, esc, Agent(idx, esc, cfg), idx,
                                      task, **kw)
            except LLMError as e:
                # The escalate model is unreachable mid-run: fall back to the
                # ordinary worker ladder rather than dropping the task.
                print(f"[proactive escalation failed ({str(e).splitlines()[0][:100]})"
                      f"; running on the worker]", file=sys.stderr)
        if reason and esc is not client:
            # A dry run / review still records the suggestion for telemetry.
            cfg._task_escalation_reason = reason
    try:
        tr = _run_task_once(root, cfg, client, agent, idx, task, **kw)
    except (BudgetExceeded, CancelledError):
        raise
    except LLMError as e:
        fb = roles_mod.client_for(cfg, "fallback", client)
        if fb is client:
            raise
        print(f"\n[worker failed ({str(e).splitlines()[0][:100]}); falling "
              f"back to the `fallback` role's model]", file=sys.stderr)
        workspace.log_event(root, f"fallback to role model after: {e}", "WARN")
        from .agent import Agent
        return _run_task_once(root, cfg, fb, Agent(idx, fb, cfg), idx, task,
                              **kw)
    if answer_system is not None or review:
        return tr
    if not getattr(cfg, "auto_escalate", True):
        return tr
    esc = roles_mod.client_for(cfg, "escalate", client)
    reason = getattr(cfg, "_task_escalation_reason", "")
    if reason and apply and esc is not client:
        print(f"\n[escalation suggested ({reason}); one more attempt on the "
              "`escalate` role's model]", file=sys.stderr)
        workspace.log_event(root, f"escalating task to the escalate role: "
                                  f"{reason}", "WARN")
        from .agent import Agent
        wrote1 = (tr.applied or bool(getattr(tr, "changed", None))
                  or any(r.ok for r in tr.results)
                  or bool(getattr(cfg, "_task_mutations", None)))
        base_idx = tr.idx
        if apply and wrote1 and getattr(tr, "tests_ok", None) is not True:
            base_idx = _rollback_attempt(root, cfg, tr, base_idx)
        cfg._task_mutations = []
        tr2 = _run_task_once(root, cfg, esc, Agent(base_idx, esc, cfg),
                             base_idx, task, **kw)
        landed2 = any(r.ok for r in tr2.results) or bool(getattr(tr2, "changed", None))
        if not landed2:
            _rollback_attempt(root, cfg, tr2)
            return tr
        landed1 = tr.applied and (any(r.ok for r in tr.results) or bool(getattr(tr, "changed", None)))
        # Keep the first attempt only when it landed a GREEN result and the
        # escalated one came back red; otherwise the stronger model's work
        # wins (it was asked precisely because the first verdict was broken).
        if landed1 and getattr(tr, "tests_ok", None) is True \
                and getattr(tr2, "tests_ok", None) is False:
            _rollback_attempt(root, cfg, tr2)
            return tr
        return tr2
    if reason and apply and esc is client:
        # #49: a reason fired but there is no distinct escalate profile.
        # Instead of dropping the reason, give the task ONE more attempt on
        # the SAME client and budgets with the edit style flipped: a model
        # stuck on exact-quote SEARCH blocks often lands SYMBOL splices, and
        # vice versa. Green runs never reach this branch.
        landed1 = tr.applied and (any(r.ok for r in tr.results) or bool(getattr(tr, "changed", None)))
        if getattr(tr, "tests_ok", None) is True:
            return tr                     # green: nothing to recover
        if landed1 and getattr(tr, "tests_ok", None) is not False:
            return tr                     # landed, no red verdict: leave it
        if not landed1 and _is_noop(tr.reply):
            return tr                     # a coherent "nothing to change"
        style = getattr(cfg, "edit_style", None)
        flipped = "quote" if style == "anchor" else "anchor"
        print(f"\n[escalation suggested ({reason}); no `escalate` role -- one "
              f"retry with the {flipped} edit style on the same model]",
              file=sys.stderr)
        workspace.log_event(root, f"strategy_shift: edit_style {style} -> "
                                  f"{flipped}: {reason}", "WARN")
        from .agent import Agent
        wrote1 = (tr.applied or bool(getattr(tr, "changed", None))
                  or any(r.ok for r in tr.results)
                  or bool(getattr(cfg, "_task_mutations", None)))
        base_idx = tr.idx
        if apply and wrote1:
            base_idx = _rollback_attempt(root, cfg, tr, base_idx)
        cfg._task_mutations = []
        cfg.edit_style = flipped
        try:
            tr2 = _run_task_once(root, cfg, client, Agent(base_idx, client, cfg),
                                 base_idx, task, **kw)
        finally:
            cfg.edit_style = style
        landed2 = tr2.applied and (any(r.ok for r in tr2.results) or bool(getattr(tr2, "changed", None)))
        if not landed2:
            _rollback_attempt(root, cfg, tr2)
            return tr                     # the retry landed nothing: keep first
        if getattr(tr, "tests_ok", None) is False \
                and getattr(tr2, "tests_ok", None) is False:
            _rollback_attempt(root, cfg, tr2)
            return tr                     # both proven red: keep the first
        return tr2
    if esc is client or (tr.applied
                         and (any(r.ok for r in tr.results) or bool(getattr(tr, "changed", None)))):
        return tr
    if any(r.ok for r in tr.results) and not apply:
        return tr                      # a dry run that produced a valid diff
    if _is_noop(tr.reply):
        return tr                      # a coherent "nothing to change"
    print("\n[no edit landed; one more attempt on the `escalate` role's "
          "model]", file=sys.stderr)
    workspace.log_event(root, "escalating a failed task to the escalate role",
                        "WARN")
    from .agent import Agent
    wrote1 = (tr.applied or bool(getattr(tr, "changed", None))
              or any(r.ok for r in tr.results)
              or bool(getattr(cfg, "_task_mutations", None)))
    base_idx = tr.idx
    if apply and wrote1:
        base_idx = _rollback_attempt(root, cfg, tr, base_idx)
    cfg._task_mutations = []
    tr2 = _run_task_once(root, cfg, esc, Agent(base_idx, esc, cfg), base_idx,
                         task, **kw)
    landed2 = any(r.ok for r in tr2.results) or (
        not tr2.results and bool(getattr(tr2, "changed", None)))
    if not landed2:
        _rollback_attempt(root, cfg, tr2)
        return tr
    return tr2


def _run_task_once(root: Path, cfg, client, agent, idx, task: str, *,
                   apply: bool = True, review: bool = False,
                   answer_system: str | None = None,
                   is_repair: bool = False) -> TaskResult:
    """Run ONE task through the inner loop: gather source, answer (or edit),
    apply or review, re-index the touched files, and write a session log.

    `agent` and `idx` are passed in so the caller controls freshness (`auto`
    hands each task a brand-new Agent; `ask`/`chat` reuse one). `answer_system`,
    when given, replaces the default edit brief (used to implement scaffold
    stubs rather than edit existing code)."""
    when = workspace.stamp()
    wants_write = apply or review
    if (_static_enabled(cfg) and not getattr(cfg, "_auto_step", 0)
            and getattr(cfg, "_static_baseline", None) is None):
        # Standalone ask/chat/eval run: diff the gate against the project as
        # it stands, so pre-existing undefined names are not blamed on this
        # task (the `auto` loop records this per step instead).
        _capture_static_baseline(root, cfg)
    if (_frontend_enabled(cfg) and not getattr(cfg, "_auto_step", 0)
            and getattr(cfg, "_frontend_baseline", None) is None):
        _capture_frontend_baseline(root, cfg)
    if (getattr(cfg, "interface_lock", True)
            and not getattr(cfg, "_auto_step", 0)
            and getattr(cfg, "_contract_baseline", None) is None):
        _capture_contract_baseline(root, cfg)
    backup_dir = (workspace.new_backup_dir(root, when, unique=True)
                  if wants_write and cfg.backup else None)
    t_started = time.time()
    usage0 = _snapshot_stats(client)
    workspace.log_event(root, f"task: {safe_text(task, cfg)}")
    todo_mod.clear(root)          # a fresh checklist per task; auto has PLAN.md
    _fire(cfg, root, "pre_task", task=task[:500])
    print("\n--- looking things up ---", file=sys.stderr)
    gathered = agent.gather(task, backup_dir=backup_dir, is_repair=is_repair)
    blueprint = str(getattr(cfg, "_hybrid_blueprint", "") or "")
    if blueprint:
        # Hybrid Stage 2: the architect's plan rides ahead of everything the
        # editor gathered, so the cheap model implements instead of guessing.
        gathered = ("HYBRID_BLUEPRINT (from the architect model -- implement "
                    "this plan; it already names the root cause, target "
                    "files, exact signatures and edge cases):\n"
                    + blueprint + "\n\n" + (gathered or ""))
    # Files a gather-phase action wrote/moved/deleted (write/append/move/
    # delete/rename). A worker that finishes the job with gather actions --
    # the deepclitest pattern -- produces no answer-phase edit blocks, and
    # its mutations used to bypass post-edit verification entirely.
    gather_mut = list(getattr(cfg, "_task_mutations", None) or ())
    tests_ok = None
    tests_preexisting_only = False
    print("--- answering ---\n", file=sys.stderr)
    from . import verify as verify_mod
    findings: list = []
    truncated = False
    refusal = ""
    contradicted = False
    fabricated = False       # reply's only blocks were no-ops
    fabricated_only = False  # ...and the reply did not declare "no changes"
    intents: list = []
    structured = (answer_system is None
                  and getattr(cfg, "edit_mode", "legacy") == "structured")
    if (answer_system is None and apply and not review
            and not structured
            and hasattr(agent, "incremental_eligible")
            and agent.incremental_eligible(task)):
        intents = agent.plan_edit_intents(task, gathered)
        if len(intents) < 2:
            intents = []          # single target: the ordinary reply is cheaper
    incremental = bool(intents) and not structured

    def _mark_fabricated(text: str) -> str:
        """Record null-only blocks; add the "No changes" marker only when the
        reply itself was not already an honest "no changes" declaration."""
        nonlocal fabricated, fabricated_only
        fabricated = True
        if _is_noop(text):
            return text
        fabricated_only = True
        return (text.rstrip() + "\n\n[No changes: every edit block in the "
                "reply was a no-op (SEARCH == REPLACE) and was discarded; "
                "nothing was written.]")

    if structured:
        # Structured mode (WP-4101B): one strict JSON reply, generated and
        # re-asked (bounded) inside agent.answer, then parsed, locally guarded
        # and applied atomically inside agent.apply. No legacy block parsing
        # and no salvage; a rejected reply lands as failed results, never a
        # half-written batch.
        raw_reply = agent.answer(task, gathered, system=answer_system)
        edit_files = _structured_files(raw_reply)
        pre = _fire(cfg, root, "pre_edit", files=edit_files, task=task[:300])
        baseline_tests = None
        if apply and not review and edit_files and getattr(
                cfg, "verify_tests", False) \
                and (getattr(cfg, "allow_shell", False)
                     or _test_run_consent(cfg)):
            baseline_tests = _run_tests(idx, cfg, edit_files)
            _note_baseline_lesson_run(root, cfg, baseline_tests)
        if apply and not review:
            _capture_dom_baseline(root, cfg, edit_files)
        if pre.denied:
            print(f"  [edits blocked by a pre_edit hook: {pre.message}]",
                  file=sys.stderr)
            results = [Result(False, f, f"blocked by a pre_edit hook: "
                                        f"{pre.message}") for f in edit_files]
            reply = raw_reply
        else:
            results, reply = agent.apply(
                raw_reply, dry_run=review or not apply, task=task,
                gathered=gathered, backup_dir=None if review else backup_dir)
            if apply and not review and (
                    any(r.ok for r in results) or gather_mut):
                _fire(cfg, root, "post_edit",
                      files=sorted({r.file for r in results if r.ok}
                                   | set(gather_mut)))
                results, reply, tests_ok, tests_preexisting_only = \
                    _post_edit_tests(
                        root, cfg, agent, idx, task, results, reply,
                        backup_dir, baseline_tests,
                        sorted(set(edit_files) | set(gather_mut)),
                        collect=findings)
    elif incremental:
        # One symbol hunk at a time, each dry-applied and verified
        # before it is written. The full impacted-test/lint pass runs once,
        # after the last hunk, exactly like the ordinary apply path.
        edit_files = sorted({
            fileio_norm(i.get("file", "")) for i in intents if i.get("file")})
        pre = _fire(cfg, root, "pre_edit", files=edit_files, task=task[:300])
        baseline_tests = None
        if edit_files and getattr(cfg, "verify_tests", False) \
                and (getattr(cfg, "allow_shell", False)
                     or _test_run_consent(cfg)):
            baseline_tests = _run_tests(idx, cfg, edit_files)
            _note_baseline_lesson_run(root, cfg, baseline_tests)
        _capture_dom_baseline(root, cfg, edit_files)
        lint_baseline: dict = {}
        if getattr(cfg, "post_edit_check", True):
            try:
                lint_baseline = diagnostics.quick_findings(root, edit_files)
            except Exception:
                lint_baseline = {}
        lsp_gate = None
        if pre.denied:
            print(f"  [edits blocked by a pre_edit hook: {pre.message}]",
                  file=sys.stderr)
            results = [Result(False, f, f"blocked by a pre_edit hook: "
                                        f"{pre.message}") for f in edit_files]
            reply = raw_reply = ""
        else:
            lsp_gate = _lsp_gate_begin(root, cfg, edit_files)
            results, reply, findings = agent.apply_edit_intents(
                task, gathered, intents, backup_dir=backup_dir,
                verbose=True)
            raw_reply = reply
            landed = sum(1 for r in results if r.ok)
            if results and landed and landed < len(results):
                # The same mixed state the model was told about, in the
                # session log: a later reader of this run must not take
                # "11 of 12 applied" for a finished feature.
                missing = ", ".join(sorted({r.file for r in results
                                            if not r.ok})[:8])
                workspace.log_event(
                    root, f"incremental batch PARTIALLY applied: {landed} of "
                          f"{len(results)} block(s) landed; not applied: "
                          f"{missing}", "WARN")
            findings = verify_mod.dedupe_findings(
                list(findings) + _lsp_gate_finish(
                    lsp_gate if any(r.ok for r in results) else None,
                    root, cfg))
            for f in findings:
                workspace.log_event(root, f"verify: {f.render()}",
                                    "WARN" if f.severity == "warn" else "ERROR")
            if any(r.ok for r in results) or gather_mut:
                _fire(cfg, root, "post_edit",
                      files=sorted({r.file for r in results if r.ok}
                                   | set(gather_mut)))
                results, reply, tests_ok, tests_preexisting_only = _post_edit_tests(
                    root, cfg, agent, idx, task, results, reply,
                    backup_dir, baseline_tests,
                    sorted(set(edit_files) | set(gather_mut)),
                    collect=findings)
                results, reply = _post_edit_lint_incremental(
                    root, cfg, agent, task, results, reply, backup_dir,
                    lint_baseline)
    else:
        raw_reply = agent.answer(task, gathered, system=answer_system)
        if answer_system is None and parse_edits(raw_reply) \
                and not _edit_refusal_head(raw_reply):
            # Several candidates, scored by the free deterministic checks (opt-in).
            if int(getattr(cfg, "best_of", 1) or 1) > 1 \
                    and hasattr(agent, "best_of"):
                raw_reply = agent.best_of(task, raw_reply, gathered)
            # Check the proposed edits BEFORE anything is written: unread code,
            # placeholders, stale references, unaddressed requirements.
            if hasattr(agent, "preflight"):
                raw_reply, findings = agent.preflight(task, raw_reply, gathered)
                findings = verify_mod.dedupe_findings(findings)
                for f in findings:
                    workspace.log_event(root, f"verify: {f.render()}",
                                        "WARN" if f.severity == "warn"
                                        else "ERROR")
        elif answer_system is None and not parse_edits(raw_reply):
            # P1a shadow mode: a no-edit reply gets the grounded-prose checks
            # logged (citations, ungrounded identifiers, unaddressed
            # requirements), never acted on. A repair prompt is harness
            # diagnostic text, not user requirements: mining it produced
            # coverage clauses like 'Fix these first:' (BUG-314).
            if not is_repair:
                try:
                    from . import verify as verify_mod
                    for f in verify_mod.prose_shadow(agent, task, raw_reply):
                        workspace.log_event(root, f"prose: {f.render()}", "WARN")
                except Exception:
                    pass
        truncated = bool(getattr(client, "stream_truncated", False))
        if truncated:
            workspace.log_event(root, "answer reply truncated at the output "
                                      "limit", "WARN")
        refusal = _edit_refusal_head(raw_reply)
        contradicted = bool(refusal) and bool(parse_edits(raw_reply))

        if contradicted:
            # A reply that OPENS by refusing yet contains blocks is a small-
            # model coherence failure, not intent: honor the refusal, discard
            # the blocks.
            print(f"  [reply opens by refusing edits (\"{refusal}\") but "
                  f"contains edit blocks; honoring the refusal and discarding "
                  f"them]", file=sys.stderr)
            workspace.log_event(root, f"reply refused edits ('{refusal}') but "
                                      f"contained edit blocks; blocks "
                                      f"discarded", "WARN")
            results, reply = [], raw_reply
        else:
            edit_files = sorted({e.file for e in parse_edits(raw_reply)})
            pre = _fire(cfg, root, "pre_edit", files=edit_files, task=task[:300])
            baseline_tests = None
            if apply and not review and edit_files and getattr(
                    cfg, "verify_tests", False) \
                    and (getattr(cfg, "allow_shell", False)
                         or _test_run_consent(cfg)):
                baseline_tests = _run_tests(idx, cfg, edit_files)
                _note_baseline_lesson_run(root, cfg, baseline_tests)
            if apply and not review:
                _capture_dom_baseline(root, cfg, edit_files)
            if pre.denied:
                print(f"  [edits blocked by a pre_edit hook: {pre.message}]",
                      file=sys.stderr)
                results = [Result(False, f, f"blocked by a pre_edit hook: "
                                            f"{pre.message}")
                           for f in edit_files]
                reply = raw_reply
            else:
                results, reply = agent.apply(
                    raw_reply, dry_run=review or not apply, task=task,
                    gathered=gathered,
                    backup_dir=None if review else backup_dir)
                if apply and not review and answer_system is None \
                        and not is_repair and hasattr(agent, "_coverage_pass"):
                    # BUG-213: the ordinary answer path owes the same
                    # batch-level coverage verdict and ONE bounded follow-up
                    # the incremental microloop runs; a zero-intent answer
                    # run used to skip both because it never entered
                    # apply_edit_intents. The guard must NOT require results:
                    # a zero-edit answer is exactly the case that owes the
                    # coverage verdict. Preflight findings were already
                    # logged above: only the new ones reach the log here.
                    logged = {(f.kind, f.file, f.message) for f in findings}
                    results, reply, findings = agent._coverage_pass(
                        task, gathered, results, reply, findings,
                        backup_dir=backup_dir, verbose=True)
                    findings = verify_mod.dedupe_findings(findings)
                    for f in findings:
                        key = (f.kind, f.file, f.message)
                        if key in logged:
                            continue
                        logged.add(key)
                        workspace.log_event(
                            root, f"verify: {f.render()}",
                            "WARN" if f.severity == "warn" else "ERROR")
                    edit_files = sorted(
                        set(edit_files) | {r.file for r in results if r.file})
                if apply and not review and (
                        any(r.ok for r in results) or gather_mut):
                    _fire(cfg, root, "post_edit",
                          files=sorted({r.file for r in results if r.ok}
                                       | set(gather_mut)))
                    results, reply, tests_ok, tests_preexisting_only = _post_edit_tests(
                        root, cfg, agent, idx, task, results, reply,
                        backup_dir, baseline_tests,
                        sorted(set(edit_files) | set(gather_mut)),
                        collect=findings)
            if not results and getattr(agent, "noop_discards", None):
                # The reply's only blocks were null edits (SEARCH == REPLACE)
                # dropped by parse_edits: the model had concluded no edit was
                # needed. Honor that instead of re-asking for "real" blocks
                # (the marker matches _is_noop, so the re-ask below is skipped).
                files = ", ".join(dict.fromkeys(agent.noop_discards))
                workspace.log_event(root, f"reply emitted only no-op edit "
                                          f"block(s) ({files}); treated as a "
                                          f"refusal to edit", "WARN")
                reply = _mark_fabricated(reply)
    if not incremental and not structured and not results \
            and not _is_noop(reply) \
            and not contradicted and not gather_mut:
        # Zero edit blocks: one bounded recovery (honour read tool-calls or
        # re-ask with the edit protocol). Skipped for no-op replies ("already
        # complete") and refusals, which are coherent about doing nothing,
        # and for gather-only runs: the mutation ledger proves the work
        # landed during the lookup phase, so pressing the reply for "real"
        # blocks would only invite fabricated no-ops.
        rec = _recover_no_edit_reply(
            root, agent, task, gathered, reply,
            dry_run=review or not apply,
            backup_dir=None if review else backup_dir,
            answer_system=answer_system)
        if rec is not None:
            results, reply = rec
            if results:
                workspace.log_event(
                    root, "answer re-asked; recovered edit blocks", "INFO")
                if apply and not review:
                    # The recovered reply applied edits on its own path; run
                    # the same post-edit test evidence the main apply gets.
                    _fire(cfg, root, "post_edit",
                          files=sorted({r.file for r in results if r.ok}))
                    results, reply, tests_ok, tests_preexisting_only = _post_edit_tests(
                        root, cfg, agent, idx, task, results, reply,
                        backup_dir, None,
                        sorted({e.file for e in parse_edits(reply)}),
                        collect=findings)
            else:
                # The "No changes:" declaration is now the reply (run_loop
                # keys its refusal path off _is_noop(tr.reply)).
                workspace.log_event(root, "re-ask declared no changes; "
                                          "honored as a refusal to edit",
                                    "WARN")
        elif not contradicted and getattr(agent, "noop_discards", None):
            # The re-ask itself produced only null edits: same treatment.
            files = ", ".join(dict.fromkeys(agent.noop_discards))
            workspace.log_event(root, f"re-ask produced only no-op edit "
                                      f"block(s) ({files}); treated as a "
                                      f"refusal to edit", "WARN")
            reply = _mark_fabricated(reply)
    if (not incremental and not structured and not results
            and not contradicted and not fabricated
            and not _is_noop(reply) and not gather_mut):
        # A "no changes" declaration is a decision, not a parse failure, and
        # a gather-only run wrote its changes with tool actions instead of
        # blocks -- neither belongs in the log as an ERROR.
        workspace.log_event(root, "no edit blocks parsed from the reply", "ERROR")
    elif results:
        for r in results:
            workspace.log_event(
                root, f"{'applied' if r.ok else 'FAILED'} {r.file}: {r.message}",
                "ERROR" if not r.ok else "INFO")
    # R8 provenance for the result frame (BUG-215): the tracker's clause map
    # used to reach only the review page; JSONL/RPC clients got bare edits.
    tracker = getattr(agent, "requirements", None)
    clauses: dict = {}
    clause_texts: dict = {}
    unaccepted: list = []
    if tracker is not None and getattr(tracker, "task", "") == task:
        clauses = tracker.clause_map()
        clause_texts = tracker.clause_texts()
        unaccepted = [req.req_id or req.text for req in tracker
                      if req.is_edit and req.state in (
                          verify_mod.REQ_UNLOCATED, verify_mod.REQ_LOCATED,
                          verify_mod.REQ_READ)]
    # One finding per clause, not per gate that re-checked it (BUG-207).
    findings = verify_mod.dedupe_findings(findings)
    applied = False
    if review:
        if not parse_edits(reply):
            print("\n  The model proposed no edits.", file=sys.stderr)
        results = _review(root, results, task, cfg, backup_dir, when,
                          clauses=clauses, clause_texts=clause_texts)
        applied = any(r.ok and "applied" in r.message for r in results)
    else:
        _report(results, dry_run=not apply)
        # `applied` means "at least one edit landed", not "apply mode was
        # on": a run whose every block failed wrote nothing (BUG-212).
        # Gather-phase mutations count as landed work too.
        applied = bool(apply and (any(r.ok for r in results) or gather_mut))
        if apply and results and not applied:
            print(f"\n[NOTHING APPLIED: all {len(results)} edit target(s) "
                  "failed; zero files were changed]", file=sys.stderr)
            workspace.log_event(
                root, f"NOTHING APPLIED: all {len(results)} edit target(s) "
                      "failed", "ERROR")

    changed: list[str] = []
    if applied:
        for r in results:
            if not r.ok:
                continue
            if getattr(r, "deleted", False):
                # A deleted file leaves the index for good; it must not join
                # `changed` either, or lint/diagnostics receive a dead path.
                try:
                    idx.forget(r.file)
                except Exception:
                    pass
                continue
            try:
                idx = index_mod.reindex_file(idx, r.file, cfg)
                changed.append(r.file)
            except Exception:
                pass               # a moved file re-indexes on next refresh
        for rel in gather_mut:
            # Gather-phase writes are re-indexed by the tools themselves;
            # only files that still exist join `changed` so lint and the
            # diagnostics check never receive a deleted path.
            if rel not in changed and (root / rel).exists():
                changed.append(rel)
        agent.index = idx

    if applied and changed and int(getattr(cfg, "refresh_summaries", 0) or 0):
        try:
            from . import roles as roles_mod
            index_mod.refresh_summaries(
                idx, roles_mod.client_for(cfg, "summarizer", client), cfg,
                limit=int(cfg.refresh_summaries), verbose=True)
        except Exception as e:                 # never fail a task over a summary
            print(f"  [summary refresh skipped: {e}]", file=sys.stderr)

    steps_log = list(getattr(agent, "steps_log", []) or [])
    session = None
    if cfg.log_sessions:
        session = workspace.write_session(root, safe_text(task, cfg),
                                          steps_log,
                                          safe_text(reply, cfg), results,
                                          backup_dir, when,
                                          raw_reply=safe_text(raw_reply, cfg)
                                          if truncated else None)
        print(f"[session: {session.relative_to(root)}]", file=sys.stderr)
    stats = _task_stats(client, usage0, t_started)
    ok_n = sum(1 for r in results if r.ok)
    events_mod.get(root, cfg).emit(
        "task", task=task[:200], steps=len(steps_log), applied=applied,
        edits_ok=ok_n, edits_failed=len(results) - ok_n, changed=changed,
        findings=[f.kind for f in findings],
        answer_mode=getattr(agent, "answer_mode_used", ""),
        folds=getattr(agent, "fold_count", 0), **stats)
    events_mod.get(root, cfg).emit(
        "result", _full=True, task=task[:200], reply=reply, applied=applied,
        edits=[{"file": r.file, "ok": r.ok, "message": r.message,
                "diff": r.diff,
                "clauses": list(clauses.get(fileio_norm(r.file), ()))}
               for r in results],
        clause_texts=clause_texts, unaccepted=unaccepted)
    # Post-run hygiene, non-blocking: scan the files this task touched plus
    # the workspace's sensitive-file ignore state, then emit a masked notice
    # for the GUI banner / TUI transcript. A scan failure never fails a run.
    try:
        security_mod.emit_post_run_notice(root, cfg, files=changed)
    except Exception:
        pass
    if stats.get("calls"):
        print(f"[task: {stats['calls']} calls, {stats['prompt_tokens']:,} "
              f"in / {stats['completion_tokens']:,} out tokens, cache hit "
              f"{stats['cache_hit']:.0%}"
              + (f", {stats['cache_write_tokens']:,} written to cache"
                 if stats.get("cache_write_tokens") else "")
              + (f", ${stats['cost']:.4f}" if stats.get("cost") else "")
              + f", {stats['seconds']:.0f}s]", file=sys.stderr)
    _fire(cfg, root, "post_task", task=task[:300], applied=applied,
          files=changed, ok=ok_n, failed=len(results) - ok_n)
    _journal_append(cfg, "step", steps=len(steps_log),
                    title=_task_title(task)[:200])
    if changed:
        _journal_append(cfg, "edit", files=changed[:200])
    _journal_append(cfg, "model_call", calls=int(stats.get("calls") or 0))
    return TaskResult(results, reply, applied, changed, idx, session,
                      steps_log, fabricated=fabricated_only,
                      backup_dir=backup_dir,
                      live_ok=getattr(agent, "live_evidence_ok", None),
                      live_kinds=_live_kinds_of(agent),
                      findings=findings, stats=stats, tests_ok=tests_ok,
                      tests_preexisting_only=tests_preexisting_only,
                      budget_hit=bool(getattr(agent, "_task_budget_hit", False)),
                      clauses=clauses, clause_texts=clause_texts,
                      unaccepted=unaccepted)


# Corrective re-runs are one-shot: a second identical failure is a verdict,
# not a transient. Each keeps the task unchecked so the next iteration picks
# it up again, with a one-shot note appended to its prompt.
_MAX_TASK_RETRIES = 1

_VERIFY_RETRY_NOTE = (
    "### RETRY -- THE PREVIOUS ATTEMPT NEVER RAN THE CHECK\n"
    "This exact verification was already attempted and ended without a "
    "single `run`, `screenshot`, or `browser` step: it declared live "
    "verification impossible and answered from source alone. That "
    "declaration was false -- those tools ARE available in this attempt's "
    "lookup phase. Launch the app with `run`, drive the page with "
    "`browser`, capture the evidence, and only then write the report with "
    "its VERDICT line. If a tool itself fails, quote the exact error "
    "output in the report.")

_VISUAL_RETRY_NOTE = (
    "### RETRY -- THE PREVIOUS ATTEMPT RAN THE APP BUT NEVER LOOKED AT IT\n"
    "This exact verification was already attempted: a `run` step executed, "
    "but no `browser` or `screenshot` step ever observed the result. "
    "Starting a server, a build or a test proves the process came up -- it "
    "says nothing about what the page or canvas displayed, which is what this "
    "task asks about. Open it: point the `browser` tool at the served URL "
    "(or take a `screenshot` for a desktop window), with actions that reach "
    "the state in question, and read the page text / canvas / console / "
    "network evidence it returns. Only then write the report with its VERDICT "
    "line. If the observation tool itself fails, quote its exact error.")

# An empty report over gathered evidence is a report-writer flake (e.g. a
# thinking runaway), not a product verdict.
_EMPTY_REPORT_RETRY_NOTE = (
    "### RETRY -- THE PREVIOUS ATTEMPT'S REPORT CAME BACK EMPTY\n"
    "This exact verification already ran once: its lookup phase completed "
    "(the app WAS driven and evidence WAS gathered), but the final answer "
    "phase returned no prose at all -- a model hiccup, not a verdict and "
    "not your fault. Gather the evidence again the same way, then WRITE "
    "THE REPORT: the observations with their exact numbers, any caveats, "
    "and the final VERDICT: PASSED/FAILED line.")

_INCOMPLETE_RETRY_NOTE = (
    "### RETRY -- THE PREVIOUS ATTEMPT DID NOT DO THE WORK\n"
    "This exact task was already attempted and ended with ZERO changes "
    "applied; its reply described what a future attempt would need instead "
    "of doing it. There is no future attempt after this one. The lookup "
    "phase has the full toolset: `read` opens any file in the project, "
    "`grep` finds references, and `run` executes a command directly (a test "
    "runner, `python`, or an app launcher). When allow_write is on, "
    "`move`/`write`/`delete` change files too. Spend this attempt's lookups "
    "on the task's actual work and finish it. If -- and only if -- the work "
    "is genuinely already complete, or genuinely impossible, say exactly "
    "that as your first line, with the evidence or the exact blocker.")

_NOOP_EDIT_RETRY_NOTE = (
    "### RETRY -- THE PREVIOUS ATTEMPT'S EDIT BLOCKS WERE NO-OPS\n"
    "This exact task was already attempted and every edit block it emitted "
    "had identical SEARCH and REPLACE text (a null edit), so nothing was "
    "written. That is not a way to finish the task. Do the work now. To move "
    "a file, emit a rename pair: one empty-SEARCH creation block for the NEW "
    "path, plus one block for the OLD path whose SEARCH quotes a few real "
    "lines and whose REPLACE is empty. Never retype the file's contents. If "
    "the task is genuinely already complete, write exactly "
    "`No changes: <reason>` as your first line and emit no blocks.")

_NOOP_EDIT_RETRY_NOTE_MOVE = (
    "### RETRY -- THE PREVIOUS ATTEMPT'S EDIT BLOCKS WERE NO-OPS\n"
    "This exact task was already attempted and every edit block it emitted "
    "had identical SEARCH and REPLACE text (a null edit), so nothing was "
    "written. That is not a way to finish the task. Do the work now. A file "
    "move has two supported forms: the gather `move` action (`file` = old "
    "path, `to` = new path), or a rename pair of edit blocks -- one "
    "empty-SEARCH creation block for the NEW path, plus one block for the OLD "
    "path whose SEARCH quotes a few real lines and whose REPLACE is empty. "
    "Never retype the file's contents. If the task is genuinely already "
    "complete, write exactly `No changes: <reason>` as your first line and "
    "emit no blocks.")

_NOOP_EVIDENCE_RETRY_NOTE = (
    "### RETRY -- THE PREVIOUS ATTEMPT CLAIMED 'ALREADY COMPLETE' WITH NO "
    "EVIDENCE\n"
    "This exact task was already attempted and answered that the work was "
    "already done -- without editing anything AND without reading the code "
    "the task names. A completion claim is only as good as the evidence "
    "behind it. Look up the task's actual targets now (read the file or "
    "function it names), then either do the work or -- only if the code you "
    "READ proves it -- write exactly `No changes: <reason>` as your first "
    "line, citing what you read.")


def ensure_run_budget(cfg, max_minutes: int = 0, *, now: float | None = None):
    """Establish the run's :class:`RunBudget` before any model work, if none is
    already bound.

    The monotonic deadline derives from ``max_minutes`` (the run's own
    wall-clock budget) and starts before any model load/lease, so leases,
    backoff, streaming, gates, tests, teardown and child launches are all
    clamped to the same deadline. ``now`` is the monotonic anchor for the
    deadline (defaults to ``time.monotonic()``); callers that already captured
    the run start pass it so the anchor is not re-read. A strict cost cap
    (``strict_cost_budget``) turns ``max_cost`` into a reservation-based cap;
    without it the run is uncapped and retains legacy estimate-only behavior.
    """
    from . import session_ctx
    existing = session_ctx.current_run_budget(cfg)
    if existing is not None:
        return existing
    minutes = int(max_minutes or 0)
    anchor = time.monotonic() if now is None else float(now)
    deadline = (anchor + minutes * 60.0) if minutes > 0 else None
    strict = bool(getattr(cfg, "strict_cost_budget", False))
    max_cost = float(getattr(cfg, "max_cost", 0) or 0)
    if not strict:
        max_cost = 0.0
    budget = session_ctx.RunBudget(
        deadline_monotonic=deadline, max_cost_usd=max_cost)
    session_ctx.set_run_budget(budget, cfg)
    return budget


def run_loop(root: Path, cfg, client, idx, plan_path: Path, *,
             apply: bool = True, verify: str = "", repair_rounds: int = 2,
             max_tasks: int = 0, max_minutes: int = 0,
             stop_on_fail: bool = True,
             answer_system: str | None = None, auto_install: bool = False,
             goal: str = "", on_task=None, strict: bool = False,
             verify_final: bool = False, validate_plan: bool = True,
             pinned: list | None = None) -> int:
    """Run the Ralph loop until the plan is empty (or a cap/stop is hit).

    Each iteration: read the plan, take the first unchecked task, run it in a
    fresh ``Agent`` (fresh context), optionally verify, then tick the task off
    and continue. Returns a process exit code (0 = no failed tasks).

    - ``apply``         write edits (callers pass True; the flag mirrors run_task).
    - ``verify``        shell command run after each task; failure triggers a
                        repair round-trip (up to repair_rounds) before the task
                        is given up on.
    - ``max_tasks``     0 = no cap; N = stop after N attempted tasks.
    - ``max_minutes``   0 = no cap; N = stop before starting a task once N
                        minutes have elapsed. The current task is not killed.
    - ``stop_on_fail``  True = halt on the first task that produces no applied
                        edit or still fails verification (the safe default);
                        False = log it and carry on to the next task.
    - ``answer_system`` replacement answer brief (used when the tasks implement
                        scaffold stubs rather than edit existing code).
    - ``auto_install``  when a verification fails on a missing third-party
                        module, pip-install it and retry (opt-in: the module
                        name comes from generated code). When off, the failure
                        is reported as a dependency problem and the code-repair
                        loop is skipped, since editing source cannot add a dep.
    - ``goal``          the user's original spec text, carried by every task
                        prompt as background so constraints the planner dropped
                        from a title still bind the worker. Forced file-reads
                        are mined from the task TITLE only (agent.task_targets).
    - ``on_task``       optional callback invoked with each completed task.
    - ``validate_plan`` check the checklist against the index before the first
                        task (WP-4104C). Warnings never block; a definite
                        containment/security/dependency error stops with 1 and
                        an invalid plan stops with 2. Scaffold's Complex path
                        validates once itself and passes False here.

    Task handling:
    - Investigation-only tasks ("Reproduce...", "Locate the cause...") are
      answered in prose and marked done without requiring edits.
    - No-op re-check items ("Verify ...", "Run the page and verify ...",
      "Check that ...", "Test that ...") are ticked done WITHOUT a worker
      session when the workspace is already green on disk: webcheck clean,
      the verify command passing, and no requirement of the title still
      owed. The loop verifies after every edit task, so a trailing re-check
      needs no diff; failing it for zero edits was the defect.
    - Verification checkpoints (titles asking to verify/confirm/observe) are
      FAILED, not ticked done, when their report says the verification did
      not succeed, when (for titles promising live observation) the session
      never drove the app (no run/browser/screenshot step), or when the answer
      phase produced no prose even after one retry. An applied edit is not
      evidence.
    - A failed checkpoint gets a FAILED LIVE TEST context note and a repair
      task inserted immediately before it; the check retries after the repair.
      Bounded per check by ``repair_rounds`` (an empty report gets no repair).
    - A verification that failed WITHOUT ever driving the app failed as a
      check RUN, not as a product test: it is re-run once with the toolset
      spelled out, then retired as unverified. Never remediated.
    - A task whose reply declares the work incomplete with zero edits applied
      is retried once, then failed -- never ticked done as a no-op.
    - ``strict`` exits 1 when a task landed edits but left a failed edit
      block (a partial application) or impacted tests still failing after
      the repair rounds; without it both are reported but the run exits 0.
    - ``verify_final`` runs the shell ``verify`` command ONCE, after the
      plan's last unchecked task, instead of after every task. A whole-goal
      check cannot pass mid-plan by definition, and the per-task gate then
      rolls back perfectly valid partial fixes; the cheap gates (syntax,
      static, frontend, contract) still veto every single step. If the run
      stops before the last task (cap, time limit, early failure) the check
      is still paid for once, for the record, after the loop.
    """
    from .agent import Agent

    if pinned is not None:
        cfg._pinned_refs = [str(p).lstrip("@").strip()
                            for p in pinned if str(p).strip()]
    elif goal:
        # An Auto goal's @file/@symbol mentions apply to every generated task,
        # even when the planner shortens a task title and drops them.
        try:
            from . import select as select_mod
            refs = [str(p).lstrip("@").strip()
                    for p in (getattr(cfg, "_pinned_refs", None) or ())
                    if str(p).strip()]
            for pin in select_mod.parse_pinned_references(
                    goal, idx if idx is not None else root):
                ref = pin.symbol_name if pin.kind == "symbol" else pin.path
                if ref and ref not in refs:
                    refs.append(ref)
            cfg._pinned_refs = refs
        except Exception:
            pass

    # WP-4104C: validate the checklist at the actual execution boundary, before
    # any task can write. Warnings are logged only; a definite plan error stops
    # the run, and an unreadable/escaping plan is invalid (exit 2).
    if validate_plan:
        try:
            from . import plan as plan_mod
            gate_code = plan_mod.gate_plan(root, Path(plan_path), idx, cfg=cfg)
        except Exception as exc:      # validation must not become a new failure
            gate_code = None
            try:
                workspace.log_event(root, f"plan check skipped: "
                                          f"{type(exc).__name__}", "WARN")
            except Exception:
                pass
        if gate_code is not None:
            return gate_code

    attempted = 0
    done = 0
    failed = 0
    strict_bad = False
    # Tasks that ticked done with SOME of their blocks not applied: each is
    # reported at the moment it happens and in the finish line -- a partial
    # edit is by design, silence about it was the defect.
    partial_tasks: list[tuple[str, int, int]] = []
    red_tasks: list[str] = []   # BUG-808: ticked done while the gate was red
    started_at = time.monotonic()
    # Establish the run's deadline + strict cost budget from the same anchor
    # the loop already uses, before any model load or lease, so gates, tests,
    # backoff and child launches are all clamped to it.
    ensure_run_budget(cfg, max_minutes, now=started_at)
    # Reports/edit summaries carried into later task prompts so a fresh
    # context still benefits from earlier findings.
    context_notes: list[str] = []
    workspace.log_event(root, f"auto loop: plan={plan_path.name}"
                              f" verify={verify or '(none)'}")
    _lint_plan_titles(root, plan_path)

    def plan_changed(path: Path, reason: str = "updated") -> None:
        """A box just changed on disk: push it to the page immediately."""
        emit_plan_updated(path, root=Path(root), cfg=cfg, reason=reason)

    # The plan may have been generated just before the loop started; announce
    # the current checklist so the Plan panel fills before Task 1's first
    # lookup, not when the whole run ends.
    plan_changed(plan_path, "loop_start")
    ratchet_skip: list[str] = []      # loop state the green ratchet must undo
    # The plan file is harness state: only the loop ticks its boxes, so the
    # model may not write it through any tool or edit block.
    try:
        plan_rel = plan_path.resolve().relative_to(Path(root).resolve()).as_posix()
        if plan_rel not in cfg.protected_paths:
            cfg.protected_paths = tuple(cfg.protected_paths) + (plan_rel,)
        ratchet_skip.append(plan_rel)
    except (ValueError, OSError):
        pass

    remediated: dict[str, int] = {}     # verify title -> repair rounds used
    # A check whose inserted repair applied NO edits was judged unfixable in
    # code (e.g. the repair would re-break behavior the goal froze). Track
    # each repair's outcome per check: False = no applied edits -> when the
    # check misses again, retire it as failed and CONTINUE with the plan.
    repair_outcome: dict[str, bool] = {}  # check title -> last repair edited?
    pending_check: list[str] = []     # check whose repair runs next (0 or 1)
    retried: dict[str, int] = {}      # task title -> corrective re-runs used
    retry_note: list[str] = []        # one-shot note for the next attempt

    # Green ratchet (TECHNIQUES.md #20): every attempted step snapshots the
    # project first; a step that fails its check after all repair rounds is
    # undone to the last verified-green state. The plan file itself is loop
    # state and must never be rolled back.
    ratchet_on = apply and bool(getattr(cfg, "auto_green_ratchet", True))
    step_no = 0
    step_of: dict[int, int] = {}      # attempt -> ratchet step number
    rolled_back = 0

    def mark_green(attempt_key: int, task: str) -> int:
        """Step *attempt_key* verified green: keep its snapshot as the
        rollback target for everything after it. Returns the step number."""
        k = step_of.pop(attempt_key, None)
        if ratchet_on and k:
            try:
                ratchet_mod.green(root, k)
                events_mod.get(root, cfg).emit("green_checkpoint", step=k,
                                               task=task[:200])
            except Exception:
                pass
        return k or 0

    def undo_step(attempt_key: int, task: str, why: str,
                  result) -> None:
        """Roll a failed step back to the last verified-green state: restore
        the files it modified, delete the files it created, re-index what
        survived, and leave a context note so later tasks know the code they
        are building on was NOT changed by this step. A step that changed
        nothing (the worker never reached an edit) leaves an empty rollback:
        quiet, not counted, nothing was undone."""
        nonlocal rolled_back, idx
        k = step_of.pop(attempt_key, None)
        if not (ratchet_on and k):
            return
        try:
            info = ratchet_mod.rollback(root, k)
        except Exception as e:
            workspace.log_event(root, f"green ratchet rollback failed: {e}",
                                "ERROR")
            return
        if result is not None:
            try:
                result.rollback_info = info
            except Exception:
                pass
        if info.get("error"):
            workspace.log_event(root, f"green ratchet: {info['error']}",
                                "WARN")
        touched = list(info["restored"]) + list(info["deleted"])
        if not touched and not info["skipped"]:
            return                    # the step never touched a file
        for rel in touched:
            if rel in info["deleted"] and not (root / rel).exists():
                continue    # a deleted file's stale entry dies on next refresh
            try:
                idx = index_mod.reindex_file(idx, rel, cfg)
            except Exception:
                pass
        rolled_back += 1
        if result is not None:
            try:
                result.rolled_back = True
            except Exception:
                pass
        events_mod.get(root, cfg).emit(
            "green_ratchet", step=k, task=task[:200],
            restored=info["restored"], deleted=info["deleted"],
            skipped=info["skipped"], reason=why[:200])
        workspace.log_event(
            root, f"green ratchet: step {k} rolled back to the last "
                  f"verified-green state ({len(info['restored'])} restored, "
                  f"{len(info['deleted'])} deleted): {why[:160]}", "WARN")
        print(f"  [green ratchet: the step failed after all repair rounds; "
              f"rolled back {len(info['restored'])} modified and deleted "
              f"{len(info['deleted'])} created file(s) -- the workspace is "
              f"back at the last verified-green checkpoint]", file=sys.stderr)
        context_notes.append(
            f"### ROLLED BACK: {task}\nThis step's changes were undone "
            f"(restored: {', '.join(info['restored'][:12]) or 'nothing'}; "
            f"deleted: {', '.join(info['deleted'][:12]) or 'nothing'}). The "
            "workspace is at the last verified-green state -- re-read files "
            "before building on them.")
        _prune_notes(context_notes)

    def offer_remediation(task: str, tr, miss: str) -> bool:
        """Record a failed verification's report as a context note and insert
        a repair task BEFORE the still-unchecked check, so the loop self-heals
        and the check retries afterwards. Bounded per check by repair_rounds.
        Returns True when a repair was inserted (caller must then continue
        instead of failing the task)."""
        if (tr.reply or "").startswith("No prose report was produced"):
            return False        # empty answer phase: no diagnosis to act on
        used = remediated.get(task, 0)
        if used >= repair_rounds:
            return False
        remediated[task] = used + 1
        context_notes.append(
            f"### FAILED LIVE TEST: {task}\n"
            f"The check did not pass: {miss}.\n"
            f"Its report follows -- treat it as the diagnosis, and re-read "
            f"the current files before acting on any line numbers in it. "
            f"The report's suggested fix is only a hypothesis: if it would "
            f"change behavior the goal freezes (timing, movement, rules), "
            f"do NOT apply it -- explain the conflict in your reply "
            f"instead:\n{_clip_report(tr.reply)}")
        _prune_notes(context_notes)     # run anchor + latest two
        insert_task_before(plan_path, task, _REMEDIATION_TITLE,
                           notify=plan_changed)
        pending_check[:] = [task]       # the repair's outcome belongs to THIS check
        print(f"  [verification task failed: {miss} -- inserted a repair "
              f"task ahead of it; the check retries after "
              f"({repair_rounds - used - 1} repair round(s) left)]",
              file=sys.stderr)
        workspace.log_event(root, f"verification failed ({miss}); "
                                  f"remediation task inserted", "WARN")
        return True

    def procedural_rerun(task: str, tr, miss: str) -> str:
        """Handle an evidence gap that is about HOW the check ran -- nobody
        drove the app, or somebody ran it and never looked at it -- rather
        than about what the product did: re-run the check once with the
        missing tool spelled out, then retire it as unverified (never a
        repair round; there is no measured defect to repair).

        Returns "rerun" when the caller must `continue` and let the loop pick
        the same, still-unchecked task up again, "retire" when the caller must
        count and mark the failure, and "" when the miss is NOT procedural
        and the normal remediation/failure path applies."""
        if not _evidence_miss_is_procedural(task, tr):
            return ""
        drove = _used_live_evidence(tr.steps)
        gap = ("it ran the app but never observed it" if drove
               else "it never drove the app")
        tries = retried.get(task, 0)
        if tries < _MAX_TASK_RETRIES:
            retried[task] = tries + 1
            retry_note[:] = [_VISUAL_RETRY_NOTE if drove
                             else _VERIFY_RETRY_NOTE]
            print(f"  [verification task failed: {miss} -- but {gap}, so the "
                  f"failure describes the check run, not the product; "
                  f"re-running it once]", file=sys.stderr)
            workspace.log_event(
                root, f"verification ended without live evidence ({gap}); "
                      f"re-running once, no remediation", "WARN")
            return "rerun"
        print(f"  [verification could not be executed: two attempts ended "
              f"with {gap} -- retiring the check as unverified (not a product "
              f"failure) and continuing]", file=sys.stderr)
        workspace.log_event(root, f"verification check retired: two attempts "
                                  f"ended with {gap}", "ERROR")
        return "retire"

    use_diag = (not verify) and bool(getattr(cfg, "auto_diagnostics", False))
    check_label = (f"$ {verify}" if verify else
                   "project diagnostics" if use_diag
                   else "interface contracts")
    spec_mod = None
    if getattr(cfg, "interface_lock", True):
        # Contract-first interface locking: every task sees the project's
        # exact public signatures, and a deterministic cross-file gate runs
        # after every task even with no --verify command configured.
        from . import spec as _spec_mod
        spec_mod = _spec_mod

    def _open_task_count() -> int:
        return sum(1 for ln in read_plan(plan_path) if _is_unchecked(ln))

    final_state = {"ran": False}     # did the real shell check ever run?
    if verify and verify_final:
        print(f"  [final-only verification: `{verify}` runs once, after the "
              f"last unchecked task; the syntax/static/frontend/contract "
              f"gates still run every step]", file=sys.stderr)
        workspace.log_event(
            root, "final-only verification: the shell check is deferred to "
                  "the last unchecked task (a whole-goal check is expected "
                  "to fail mid-plan; do not roll back valid partial fixes "
                  "for it)", "INFO")

    def run_check(changed):
        if verify:
            if verify_final and _open_task_count() > 1:
                # More unchecked tasks follow this one: the whole-goal check
                # is not due yet. Return as passed; the cheap gates inside
                # _gated_check still vetoed this step before we got here.
                return True, ""
            # cfg rides along so the check gets the configured environment
            # (env_passthrough / scrub_env) and run_verify records the outcome
            # in project memory exactly once.
            ok, out = run_verify(root, verify, cfg=cfg)
            final_state["ran"] = True
            if verify_final:
                # Tell `auto`'s post-run pass that the whole-goal check has
                # already run, so it does not run the same command a second
                # time after the loop.
                cfg._verify_final_ran = True
                cfg._verify_final_ok = ok
            return ok, out
        if use_diag:
            return diagnostics.run_changed(root, cfg, changed)
        return True, ""

    def notify(task: str) -> None:
        if on_task is not None:
            try:
                on_task(task)
            except Exception:
                pass

    while max_tasks == 0 or attempted < max_tasks:
        if _cancel_requested(cfg, client):
            print(f"\n[stopped by user; {plan_path.name} keeps its "
                  f"remaining unchecked tasks]", file=sys.stderr)
            workspace.log_event(root, "auto stopped by user", "WARN")
            _emit_run_cancelled(root, cfg, "loop", plan=str(plan_path))
            break
        nxt = next_task(read_plan(plan_path))
        if nxt is None:
            if failed:
                # Nothing unchecked is left, but the plan is NOT finished in
                # the sense a reader assumes: --keep-going ran past failures,
                # and those tasks sit marked [!]. Saying "all tasks complete"
                # here reads as a clean finish in the log and the terminal.
                print(f"\nPlan exhausted: no unchecked tasks remain, but "
                      f"{failed} task(s) FAILED and are marked [!] in "
                      f"{plan_path.name} -- this is not a clean finish.",
                      file=sys.stderr)
                workspace.log_event(
                    root, f"plan exhausted with {failed} failed task(s); "
                          f"not a clean finish", "ERROR")
            elif strict_bad:
                # BUG-313: every box is ticked, but --strict rejected at
                # least one task's edits/tests. "All tasks complete." read
                # as a clean finish while the exit code said 1.
                print("\nAll tasks are ticked, but the run is NOT clean "
                      "under --strict: at least one task left failed edit "
                      "blocks or failing tests (exit 1).", file=sys.stderr)
                workspace.log_event(
                    root, "all tasks ticked but --strict flagged unclean "
                          "task(s); not a clean finish", "ERROR")
            else:
                print("\nAll tasks complete.", file=sys.stderr)
                workspace.log_event(root, "all tasks complete")
            break
        if max_minutes and \
                time.monotonic() - started_at >= max_minutes * 60:
            print(f"\n[stopping: time limit reached "
                  f"({max_minutes} min); remaining tasks are still "
                  f"unchecked in {plan_path.name}]", file=sys.stderr)
            workspace.log_event(
                root, f"auto stopped: {max_minutes}-minute limit reached",
                "WARN")
            break
        _, task = nxt
        if _is_noop_verify_item(task) and \
                _noop_verify_already_green(root, task, verify, cfg):
            # A "Run tests / Verify the page" re-check the planner should
            # not have emitted: the loop already verified after every edit
            # task, and the disk evidence says nothing is owed. Tick it
            # without a worker session or a diff -- the zero-edit gate
            # would otherwise fail it ([!]) for doing nothing.
            print(f"\n=== task (already verified): {task} ===",
                  file=sys.stderr)
            print("  [no-op verification item: the workspace already passes "
                  "webcheck" + (f" and `{verify}`" if verify else "")
                  + " with every requirement met on disk -- ticking it done "
                  "without a worker session or a diff]", file=sys.stderr)
            workspace.log_event(
                root, "no-op verification item ticked done: workspace "
                      "already green on disk (no diff needed)")
            mark_done(plan_path, task, notify=plan_changed)
            done += 1
            notify(task)
            continue
        attempted += 1
        print(f"\n=== task {attempted}: {task} ===", file=sys.stderr)
        if ratchet_on:
            # Checkpoint quota (§3.7): free eligible history first, and stop
            # new snapshot-dependent writes (the green ratchet) when the quota
            # cannot be met without deleting protected history.
            quota_ok, quota_reason = checkpoint_gc.enforce_quota(root, cfg)
            if not quota_ok:
                msg = ("checkpoint quota "
                       f"({getattr(cfg, 'checkpoint_quota_mb', 0)} MB) cannot "
                       f"be met safely; stopping: {quota_reason}")
                workspace.log_event(root, msg, "ERROR")
                print(f"\n[stopped: {msg}]", file=sys.stderr)
                break
            step_no += 1
            try:
                snap = ratchet_mod.begin(
                    root, step_no, task,
                    max_file_bytes=int(getattr(
                        cfg, "auto_checkpoint_max_file_bytes",
                        ratchet_mod.DEFAULT_MAX_FILE_BYTES)),
                    max_files=int(getattr(cfg, "auto_checkpoint_max_files",
                                          ratchet_mod.DEFAULT_MAX_FILES)),
                    skip=ratchet_skip)
                if snap.get("truncated"):
                    workspace.log_event(
                        root, f"green checkpoint step {step_no} hit the "
                              f"file cap; its rollback is partial", "WARN")
            except Exception as e:
                workspace.log_event(root, f"green checkpoint failed: {e}",
                                    "WARN")
            step_of[attempted] = step_no
        # The static gate's baseline belongs to the STEP, not the task: it is
        # taken here (before the worker's first write, and once per step so
        # the step's own errors are repairable, not "pre-existing").
        cfg._auto_step = attempted
        _capture_static_baseline(root, cfg)
        _capture_frontend_baseline(root, cfg)
        _capture_contract_baseline(root, cfg)

        investigate = _is_investigation_task(task)
        task_for_agent = task + _evidence_hint(task)
        if goal:
            task_for_agent = (
                "ORIGINAL USER GOAL (background -- the CURRENT TASK below is "
                "the step to do now; every constraint in the goal still "
                "binds it):\n" + _clip_report(goal, budget=4000)
                + "\n\nCURRENT TASK\n" + task_for_agent)
        if context_notes:
            task_for_agent = (
                task_for_agent + "\n\nCONTEXT FROM PRIOR INVESTIGATION\n\n"
                + "\n\n".join(context_notes))
        if spec_mod is not None and not investigate:
            # Interface lock: the live signature map rides ahead of the
            # retry note so no step can forget what an earlier step wrote.
            try:
                cmap = spec_mod.contract_map_block(root, task=task)
            except Exception:
                cmap = ""
            if cmap:
                task_for_agent += "\n\n" + cmap
        if retry_note:
            # One-shot corrective note, appended last (freshest thing the
            # worker reads) and popped so it cannot leak into another task.
            task_for_agent += "\n\n" + retry_note.pop()
        try:
            # Specialist routing at the task boundary (#50): `client` stays
            # the worker for the next task; this task may run on a
            # specialist instead. Scored on the clean title, before the
            # goal/context blocks are layered on.
            task_client, _ = _specialist_route(root, cfg, client, None, idx,
                                               task)
            agent = Agent(idx, task_client, cfg)
            # Forced reads are mined from the clean title only: prior reports
            # mention files descriptively and would bloat the must-read list.
            agent.task_targets = task
            task_lessons = _injected_lessons(root)
            if investigate:
                tr = run_investigation_task(
                    root, cfg, agent, idx,
                    task_for_agent, answer_system=answer_system)
            else:
                tr = run_task(root, cfg, task_client, agent, idx,
                              task_for_agent, apply=apply, review=False,
                              answer_system=answer_system)
        except CancelledError:
            raise
        except LLMError as e:
            print(f"  task failed: {e}", file=sys.stderr)
            workspace.log_event(root, f"task failed: {e}", "ERROR")
            if isinstance(e, CircuitOpenError) \
                    or _transport_outage(e):
                # A model-server outage (circuit breaker open, connection
                # dropped) is not a code defect: whatever edits the task
                # applied are valid, and the run simply could not verify
                # them. Rolling the green ratchet back here would discard
                # real work over a broken wire, so leave the tree as-is and
                # fail the step without an undo.
                failed += 1
                if stop_on_fail:
                    break
                mark_failed(plan_path, task, notify=plan_changed)
                continue
            undo_step(attempted, task, f"task failed: {e}", None)
            failed += 1
            if stop_on_fail:
                break
            mark_failed(plan_path, task, notify=plan_changed)
            continue
        idx = tr.idx

        if investigate:
            # The report is the deliverable; there is no code change to apply
            # or verify. A failed/empty run is still a failure.
            if (tr.reply or "").strip():
                miss = _evidence_failure(task, tr)
                if miss:
                    # A check nobody executed -- or nobody OPENED -- says
                    # nothing about the product: re-run it once with the
                    # toolset spelled out, and only retire it (never remediate
                    # it) when the re-run stays source-only too.
                    outcome = procedural_rerun(task, tr, miss)
                    if outcome == "rerun":
                        continue
                    if outcome == "retire":
                        failed += 1
                        mark_failed(plan_path, task, notify=plan_changed)
                        repair_outcome.pop(task, None)
                        continue
                    if (tr.reply or "").startswith(
                            "No prose report was produced"):
                        # The stand-in is not a verdict and carries no
                        # diagnosis. Re-run the check once; a second empty
                        # answer retires it as unverified (a procedural
                        # failure) and the plan CONTINUES.
                        tries = retried.get(task, 0)
                        if tries < _MAX_TASK_RETRIES:
                            retried[task] = tries + 1
                            retry_note[:] = [_EMPTY_REPORT_RETRY_NOTE]
                            print(f"  [verification task failed: {miss} -- "
                                  f"but the evidence WAS gathered and the "
                                  f"answer phase was empty; that is a "
                                  f"report-writer flake, not a verdict -- "
                                  f"re-running it once]", file=sys.stderr)
                            workspace.log_event(
                                root, "verification report came back empty "
                                      "despite gathered evidence "
                                      "(procedural failure); re-running "
                                      "once, no remediation", "WARN")
                            continue
                        print("  [verification report was empty twice -- "
                              "retiring the check as unverified (not a "
                              "product failure) and continuing]",
                              file=sys.stderr)
                        workspace.log_event(
                            root, "verification check retired: the answer "
                                  "phase produced no prose on two attempts",
                            "ERROR")
                        failed += 1
                        mark_failed(plan_path, task, notify=plan_changed)
                        repair_outcome.pop(task, None)
                        continue
                    if repair_outcome.get(task) is False:
                        # The repair for THIS check applied no edits: another
                        # round would refuse again and stopping would strand
                        # every downstream task. Retire the check loudly and
                        # let the plan continue.
                        print(f"  [verification task failed: {miss} -- its "
                              f"repair applied no edits (a goal conflict or "
                              f"an already-satisfied check); retiring the "
                              f"check as failed and continuing]",
                              file=sys.stderr)
                        workspace.log_event(
                            root, "verification check retired after a "
                                  f"no-edit repair: {miss}", "ERROR")
                        failed += 1
                        mark_failed(plan_path, task, notify=plan_changed)
                        repair_outcome.pop(task, None)
                        continue
                    if offer_remediation(task, tr, miss):
                        continue    # the check stays unchecked; the repair
                                    # task runs next, then it retries
                    print(f"  [verification task failed: {miss} -- "
                          f"{'stopping' if stop_on_fail else 'skipping'}]",
                          file=sys.stderr)
                    workspace.log_event(
                        root, f"verification task gathered no evidence: {miss}",
                        "ERROR")
                    failed += 1
                    if stop_on_fail:
                        break
                    mark_failed(plan_path, task, notify=plan_changed)
                    continue
                context_notes.append(f"### {task}\n{_clip_report(tr.reply)}")
                _prune_notes(context_notes)   # run anchor + latest two
                mark_green(attempted, task)
                mark_done(plan_path, task, notify=plan_changed)
                done += 1
                workspace.log_event(root, "investigation done")
                notify(task)
                continue
            print("  [investigation produced no report -- "
                  f"{'stopping' if stop_on_fail else 'skipping'}]",
                  file=sys.stderr)
            workspace.log_event(root, "investigation produced no report", "ERROR")
            failed += 1
            if stop_on_fail:
                break
            mark_failed(plan_path, task, notify=plan_changed)
            continue

        ok_edits = [r for r in tr.results if r.ok]
        if not tr.applied or not (ok_edits or tr.changed):
            if task != _REMEDIATION_TITLE and _declares_incomplete(tr.reply):
                # A reply describing work still required is never a no-op,
                # even if it incidentally contains "already". The deferred
                # work was possible in the lookup phase, so retry once with
                # that spelled out, then fail. Repair tasks are exempt:
                # refusing to edit is their sanctioned conflict-report.
                tries = retried.get(task, 0)
                if tries < _MAX_TASK_RETRIES:
                    retried[task] = tries + 1
                    retry_note[:] = [_INCOMPLETE_RETRY_NOTE]
                    print("  [task declared itself incomplete with no edits "
                          "applied -- retrying it once]", file=sys.stderr)
                    workspace.log_event(
                        root, "task declared itself incomplete with no edits "
                              "applied; retrying once", "WARN")
                    continue
                print(f"  [task declared itself incomplete twice; marking "
                      f"it failed -- "
                      f"{'stopping' if stop_on_fail else 'continuing'}]",
                      file=sys.stderr)
                workspace.log_event(
                    root, "task declared itself incomplete on retry; "
                          "marking failed", "ERROR")
                failed += 1
                if stop_on_fail:
                    break
                mark_failed(plan_path, task, notify=plan_changed)
                continue
            if tr.fabricated and task != _REMEDIATION_TITLE:
                # The reply's only blocks were null edits and it did not
                # itself declare the work complete. Retry once with concrete
                # move/rename instructions, then fail.
                tries = retried.get(task, 0)
                if tries < _MAX_TASK_RETRIES:
                    retried[task] = tries + 1
                    retry_note[:] = [
                        _NOOP_EDIT_RETRY_NOTE_MOVE
                        if getattr(cfg, "allow_write", False)
                        else _NOOP_EDIT_RETRY_NOTE]
                    print("  [task emitted only no-op edit blocks -- "
                          "retrying it once]", file=sys.stderr)
                    workspace.log_event(
                        root, "task emitted only no-op edit blocks; "
                              "retrying once", "WARN")
                    continue
                print(f"  [task emitted only no-op edit blocks twice; "
                      f"marking it failed -- "
                      f"{'stopping' if stop_on_fail else 'continuing'}]",
                      file=sys.stderr)
                workspace.log_event(
                    root, "task emitted only no-op edit blocks on retry; "
                          "marking failed", "ERROR")
                failed += 1
                if stop_on_fail:
                    break
                mark_failed(plan_path, task, notify=plan_changed)
                continue
            if (_is_noop(tr.reply) and task != _REMEDIATION_TITLE
                    and _is_verification_task(task)):
                # "Nothing to change" is still a CLAIM about live behavior,
                # and this branch used to tick it done before any gate ran:
                # the same title with an edit applied goes through
                # _evidence_failure below. Gate the zero-edit reply the same
                # way, or a verification checkpoint can be closed by prose
                # alone. Repair tasks stay exempt -- refusing to edit is
                # their sanctioned conflict report, and the check that
                # retries after them does the observing.
                miss = _evidence_failure(task, tr)
                if miss:
                    outcome = procedural_rerun(task, tr, miss)
                    if outcome == "rerun":
                        continue
                    if outcome == "retire":
                        # The check was never performed, so it says nothing
                        # about the product: counted and marked exactly like
                        # the investigation path, and the plan carries on.
                        failed += 1
                        mark_failed(plan_path, task, notify=plan_changed)
                        repair_outcome.pop(task, None)
                        continue
                    if offer_remediation(task, tr, miss):
                        continue        # a repair was inserted; the check
                                        # stays unchecked and retries after it
                    print(f"  [verification task reported 'already complete' "
                          f"with no edits and no evidence: {miss} -- "
                          f"{'stopping' if stop_on_fail else 'skipping'}]",
                          file=sys.stderr)
                    workspace.log_event(
                        root, f"no-op verification rejected: {miss}", "ERROR")
                    failed += 1
                    if stop_on_fail:
                        break
                    mark_failed(plan_path, task, notify=plan_changed)
                    continue
            if _is_noop(tr.reply):
                # "Already complete" is a CLAIM about the code, and prose
                # alone is not evidence (BUG-202): a broken task title once
                # let "the function is already correct" tick a step whose
                # function was wrong. Demand at least one edit clause the
                # run actually READ before ticking it done. Report-only
                # tasks have no edit clauses; repairs stay exempt because
                # the check that retries after them does the observing.
                from . import verify as verify_mod
                tracker = getattr(agent, "requirements", None)
                edit_reqs = [r for r in (tracker or ()) if r.is_edit]
                read_states = (verify_mod.REQ_READ, verify_mod.REQ_EDITED,
                               verify_mod.REQ_VERIFIED, verify_mod.REQ_REPORTED)
                if (task != _REMEDIATION_TITLE and edit_reqs
                        and not any(r.state in read_states for r in edit_reqs)):
                    tries = retried.get(task, 0)
                    if tries < _MAX_TASK_RETRIES:
                        retried[task] = tries + 1
                        retry_note[:] = [_NOOP_EVIDENCE_RETRY_NOTE]
                        print("  [no-op claim with no read evidence -- "
                              "retrying it once]", file=sys.stderr)
                        workspace.log_event(
                            root, "no-op claim lacked requirement evidence; "
                                  "retrying once", "WARN")
                        continue
                    print(f"  [no-op claim with no read evidence twice; "
                          f"marking it failed -- "
                          f"{'stopping' if stop_on_fail else 'continuing'}]",
                          file=sys.stderr)
                    workspace.log_event(
                        root, "no-op claim lacked requirement evidence on "
                              "retry; marking failed", "ERROR")
                    failed += 1
                    if stop_on_fail:
                        break
                    mark_failed(plan_path, task, notify=plan_changed)
                    continue
                print("  [model reports the task is already complete; "
                      "marking it done]", file=sys.stderr)
                workspace.log_event(root, "task already complete (no-op)")
                if task == _REMEDIATION_TITLE and pending_check:
                    # The repair refused to edit: remember that for the check
                    # that retries next.
                    repair_outcome[pending_check.pop()] = False
                mark_green(attempted, task)
                mark_done(plan_path, task, notify=plan_changed)
                done += 1
                continue
            print(f"  [task produced no applied edits -- {'stopping' if stop_on_fail else 'skipping'}]",
                  file=sys.stderr)
            workspace.log_event(root, "task produced no applied edits", "ERROR")
            failed += 1
            if stop_on_fail:
                break
            mark_failed(plan_path, task, notify=plan_changed)
            continue

        if task == _REMEDIATION_TITLE and pending_check:
            # The repair edited code: a re-failing check stays on the normal
            # remediation path (retiring only applies to refusing repairs).
            repair_outcome[pending_check.pop()] = True

        # Verification gate: on failure, hand the model the command's output
        # to fix. With auto_diagnostics on and no explicit --verify, linter/
        # typechecker findings for the changed files play the same role.
        passed = True
        dep_blocked: list[str] = []
        verify_failures: list[str] = []   # failure blobs this task's check saw
        if (verify or use_diag or spec_mod is not None
                or _frontend_enabled(cfg)):
            from . import verify as verify_mod

            def _gated_check(result):
                """Parse the applied buffers before starting any subprocess;
                a new syntax error is an immediate, structured failure. The
                static, frontend and cross-file contract gates all get the
                same veto -- and the contract gate runs BEFORE the verify
                subprocess, so an import or test-call that cannot satisfy a
                changed signature fails the step without paying for pytest."""
                syntax = verify_mod.check_syntax(
                    getattr(result, "results", None))
                if not syntax:
                    static = _new_static_errors(
                        root, cfg, getattr(result, "changed", None) or [])
                    if static:
                        workspace.log_event(
                            root, "static gate rejected the step; verify "
                                  "subprocess skipped", "ERROR")
                        print(f"  [static gate: {len(static)} new "
                              f"undefined-name/import error(s); "
                              f"{check_label} subprocess skipped]",
                              file=sys.stderr)
                        return False, (
                            "FAST STATIC GATE -- the changed files name "
                            "things that do not exist at the point they are "
                            "used (a missing import, a typo, or a bad "
                            "`__all__` entry); no test or lint subprocess "
                            "was run. Fix these first:\n"
                            + "\n".join(f"- {e}" for e in static[:25]))
                    fe = _new_frontend_errors(
                        root, cfg, getattr(result, "changed", None) or [])
                    if fe:
                        # BUG-218: the step's edits are already on disk; the
                        # gate fails the step, it does not unwrite the files.
                        workspace.log_event(
                            root, "frontend contract gate found violations "
                                  "in the applied edits; verify subprocess "
                                  "skipped (the files stay written pending "
                                  "repair)", "ERROR")
                        print(f"  [frontend gate: {len(fe)} new DOM contract "
                              f"violation(s); {check_label} subprocess "
                              f"skipped]", file=sys.stderr)
                        return False, (
                            "FRONTEND DOM CONTRACT GATE -- the changed "
                            "HTML/JS breaks a contract the browser enforces "
                            "at page load (an asset that would 404, a query "
                            "for an element id nothing renders, or JS that "
                            "does not parse); no test or lint subprocess "
                            "was run. Fix these first:\n"
                            + "\n".join(f"- {e}" for e in fe[:25]))
                    bad: list[str] = []
                    if spec_mod is not None:
                        bad = _new_contract_errors(
                            root, cfg, result.changed or [])
                    if bad:
                        workspace.log_event(
                            root, "contract gate rejected the step; verify "
                                  "subprocess skipped", "ERROR")
                        print(f"  [contract gate: {len(bad)} new cross-file "
                              f"contract violation(s); {check_label} "
                              f"subprocess skipped]", file=sys.stderr)
                        return False, (
                            "CROSS-FILE INTERFACE CONTRACT VIOLATIONS\n"
                            "(the names below do not exist in the module "
                            "they are imported from, the ES import/export "
                            "surface does not agree, or the call cannot "
                            "satisfy the real signature -- fix the "
                            "exporter and every importer in one step; "
                            "do not rename around the problem)\n\n"
                            + "\n".join(f"- {e}" for e in bad))
                    ok, out = run_check(result.changed)
                    return ok, out
                print(f"  [syntax: {len(syntax)} new syntax error(s); "
                      f"{check_label} subprocess skipped until the patch "
                      f"parses]", file=sys.stderr)
                workspace.log_event(
                    root, "syntax gate rejected the patch; verify subprocess "
                          "skipped", "ERROR")
                return False, verify_mod.render(syntax)

            passed, output = _gated_check(tr)
            if not passed:
                verify_failures.append(str(output)[:6000])
            rounds = repair_rounds
            installed: set[str] = set()
            # Rollback-if-worse: the current failure's severity. A repair
            # round that raises it gets undone from its own snapshot (which
            # holds the state before that round's writes), keeping the
            # better earlier state, and repair stops.
            prev_score = None if passed else _failure_score(output, use_diag)
            lesson_seen = "" if passed else str(output)[:6000]
            while not passed:
                # BUG-1009: a check that cannot even LAUNCH -- the verify
                # binary is not on this machine's PATH ("command not found",
                # rc=127) -- is a config/environment fact, not a product
                # defect. Run 3 proved the old reading: every task spent
                # repair rounds asking the model to "fix" a PATH problem
                # (the model even offered a correct PLAN edit: use
                # `python -m pytest`), then the green ratchet rolled the
                # valid steps back over the broken check. Retire the check
                # once, loudly; the gates and the remaining tasks continue.
                if _verify_unlaunchable(output):
                    print(f"  [verification check retired: `{check_label}` "
                          "could not launch (the command is missing from "
                          "this environment) -- no code change can install "
                          "it; the rest of the run is UNVERIFIED. Set "
                          "--verify to a runnable command.]",
                          file=sys.stderr)
                    workspace.log_event(
                        root, "verify check retired: command not launchable "
                              "in this environment", "WARN")
                    verify = ""      # run_check takes the gates-only path
                    passed = True    # the step is not blamed on the code
                    break
                # A missing third-party package is NOT a code bug: editing
                # source cannot add a dependency. Install it if the user opted
                # in, otherwise report and stop.
                missing = missing_modules(output, root=root)
                if missing:
                    dists = ", ".join(pip_name(m) for m in missing)
                    if auto_install and not installed.issuperset(missing):
                        installed.update(missing)
                        print(f"  [verify needs missing dependencies: {dists}; "
                              f"installing]", file=sys.stderr)
                        workspace.log_event(
                            root, f"verify missing deps: {dists}; installing",
                            "WARN")
                        ok, _ = install_modules(missing, root)
                        workspace.log_event(
                            root, f"pip install {'ok' if ok else 'FAILED'}: {dists}",
                            "INFO" if ok else "ERROR")
                        if ok:
                            passed, output = _gated_check(tr)
                            if not passed:
                                verify_failures.append(str(output)[:6000])
                            continue          # a dependency is not a repair round
                    elif auto_install:
                        print(f"  [verify still reports missing dependencies "
                              f"after installing them: {dists}]",
                              file=sys.stderr)
                        workspace.log_event(
                            root, f"verify blocked by missing deps after "
                                  f"install: {dists}", "ERROR")
                    else:
                        print(f"  [verify cannot run: missing dependencies: "
                              f"{dists}. Install with `pip install {dists}`, "
                              f"or rerun with --auto-install.]", file=sys.stderr)
                        workspace.log_event(
                            root, f"verify blocked by missing deps: {dists}",
                            "ERROR")
                    dep_blocked = missing
                    break
                if rounds <= 0:
                    break
                print(f"  [verification failed; asking the model to fix it "
                      f"({rounds} round(s) left)]", file=sys.stderr)
                workspace.log_event(root, f"verify failed: {check_label}",
                                    "ERROR")
                fix_task = (
                    f"Your last change failed this check:\n\n"
                    f"{check_label}\n{output[:6000]}\n\n"
                    f"Fix the code so that check succeeds. Re-run it in your "
                    f"head before answering, and only then emit search/replace "
                    f"blocks for the files that need changing.")
                try:
                    tr2 = run_task(root, cfg, client, Agent(idx, client, cfg),
                                   idx, fix_task, apply=apply, review=False,
                                   answer_system=answer_system, is_repair=True)
                except CancelledError:
                    raise
                except LLMError as e:
                    print(f"  repair failed: {e}", file=sys.stderr)
                    workspace.log_event(root, f"repair failed: {e}", "ERROR")
                    break
                idx = tr2.idx
                passed, output = _gated_check(tr2)
                if not passed:
                    verify_failures.append(str(output)[:6000])
                if not passed and prev_score is not None:
                    score = _failure_score(output, use_diag)
                    if score > prev_score and tr2.backup_dir \
                            and tr2.backup_dir.is_dir():
                        restored = workspace.restore(root, tr2.backup_dir)
                        for f in restored:
                            try:
                                idx = index_mod.reindex_file(idx, f, cfg)
                            except Exception:
                                pass
                        print(f"  [repair made the check worse "
                              f"({prev_score} -> {score} problem(s)); "
                              f"rolled back {len(restored)} file(s) from "
                              f"{tr2.backup_dir.name} and stopped repairing]",
                              file=sys.stderr)
                        workspace.log_event(
                            root, f"repair round made {check_label} worse "
                                  f"({prev_score} -> {score}); restored "
                                  f"{len(restored)} file(s)", "WARN")
                        break
                    prev_score = score
                rounds -= 1

            if passed and lesson_seen and rounds < repair_rounds:
                # This task's own repair rounds turned a failing check green:
                # distill what it was into the lesson ledger, while the
                # evidence is fresh. One line, mechanical, no LLM prose.
                _record_repair_lesson(root, cfg, [("verify", lesson_seen)])
            lesson_seen = ""
            # One observed run against the lesson effectiveness counters
            # (R5): a clean check counts as "avoided" for every live lesson,
            # a recurring failure kind as "recurred".
            kinds = set()
            for blob in verify_failures:
                lesson = distill_repair_lesson(blob, "verify")
                if lesson is not None:
                    kinds.add(lesson[0])
            _note_lesson_run(root, cfg, kinds, task_lessons)

        if not passed:
            if dep_blocked:
                dists = ", ".join(pip_name(m) for m in dep_blocked)
                print(f"  [verification blocked by missing dependencies: "
                      f"{dists}]", file=sys.stderr)
                workspace.log_event(
                    root, f"verify still failing (missing deps: {dists}): "
                          f"{check_label}", "ERROR")
            else:
                print(f"  [verification still failing: {check_label}]",
                      file=sys.stderr)
                workspace.log_event(
                    root, f"verify still failing: {check_label}", "ERROR")
            # BUG-1005: this undo_step used to live only in the else branch,
            # so a step that failed verify on a missing third-party package
            # skipped the green ratchet and left its wreckage in the tree --
            # exactly the poisoned baseline #20 exists to prevent. A step
            # that never verified is a failed step whatever blocked it.
            undo_step(attempted, task,
                      (f"verification blocked by missing dependencies: "
                       f"{', '.join(pip_name(m) for m in dep_blocked)}"
                       if dep_blocked else
                       f"verification still failing: {check_label}"), tr)
            failed += 1
            if stop_on_fail:
                break
            mark_failed(plan_path, task, notify=plan_changed)
            continue

        miss = _evidence_failure(task, tr)
        if miss:
            if offer_remediation(task, tr, miss):
                continue        # the check stays unchecked; the repair
                                # task runs next, then it retries
            # An applied edit is not evidence, even with the shell verify green.
            print(f"  [verification task failed: {miss} -- "
                  f"{'stopping' if stop_on_fail else 'skipping'}]",
                  file=sys.stderr)
            workspace.log_event(
                root, f"verification task gathered no evidence: {miss}",
                "ERROR")
            failed += 1
            if stop_on_fail:
                break
            mark_failed(plan_path, task, notify=plan_changed)
            continue

        if tr.applied and tr.changed:
            # Mark earlier notes stale so later tasks do not trust pre-edit
            # reports. Structured list, no reply prose: the model can re-read
            # a file, but tokens spent re-summarising the change are wasted.
            files = "\n".join(f"- {f}" for f in sorted(set(tr.changed)))
            context_notes.append(
                f"### {task}\nFILES CHANGED:\n{files}\n"
                "Earlier notes predate these writes; their line numbers and "
                "defect descriptions may be stale -- re-read the current file "
                "state before quoting it.")
            _prune_notes(context_notes)   # run anchor + latest two

            if any(f.endswith((".html", ".htm", ".js", ".mjs", ".cjs"))
                   for f in tr.changed):
                # Contract lint for web files (webcheck): a notice board,
                # never a task failure -- regex-level checks can misread a
                # dynamically built id, but the warnings land in the run log
                # where a later task (or the user) can act on them.
                try:
                    from . import webcheck
                    notes = webcheck.run(root, tr.changed)[:8]
                except Exception:
                    notes = []
                for note in notes:
                    print(f"  [webcheck] {note}", file=sys.stderr)
                    workspace.log_event(root, f"webcheck: {note}", "WARN")

        bad_blocks = [r for r in tr.results if not r.ok]
        if bad_blocks and (ok_edits or tr.changed):
            # A partial application: the unapplied half is named here, in the
            # run log, and again in the finish line -- never silently. The
            # task is NOT ticked done over a half-applied batch: it gets ONE
            # retry to land the missing targets, and a second miss counts as
            # a failure (the plan marks it [!] instead of [x]).
            where = ", ".join(sorted({r.file for r in bad_blocks}))
            total = len(ok_edits) + len(bad_blocks)
            partial_tasks.append((task, len(ok_edits), len(bad_blocks)))
            print(f"  [partial edit: {len(bad_blocks)} of {total} block(s) "
                  f"did not apply ({where})]", file=sys.stderr)
            workspace.log_event(
                root, f"partial edit on task '{task[:60]}': {len(bad_blocks)}"
                      f" of {total} block(s) did not apply ({where})", "WARN")
            tries = retried.get(task, 0)
            if task != _REMEDIATION_TITLE and tries < _MAX_TASK_RETRIES:
                retried[task] = tries + 1
                retry_note[:] = [
                    "### RETRY -- PART OF YOUR PREVIOUS EDIT DID NOT APPLY\n"
                    f"Your previous attempt applied {len(ok_edits)} block(s) "
                    f"but {len(bad_blocks)} did NOT land ({where}). The files "
                    "already written stay written; re-read their current "
                    f"state and emit ONLY the missing blocks for: {where}."]
                print("  [partial edit -- retrying once to land the missing "
                      "block(s)]", file=sys.stderr)
                workspace.log_event(
                    root, "partial edit: retrying once to land the missing "
                          "targets", "WARN")
                continue
            # The retry (or a repair) still left blocks unapplied: fail it
            # instead of ticking it done.
            print("  [partial edit persisted after a retry -- marking the "
                  "task failed]", file=sys.stderr)
            workspace.log_event(
                root, f"partial edit persisted after a retry on "
                      f"'{task[:60]}': marking failed", "ERROR")
            failed += 1
            if stop_on_fail:
                break
            mark_failed(plan_path, task, notify=plan_changed)
            continue

        if strict and ((tr.tests_ok is False and not tr.tests_preexisting_only)
                       or any(not r.ok for r in tr.results)):
            # The work landed but is not clean: a failed edit block (a
            # partial application) or impacted tests still red after the
            # repair rounds. A purely pre-existing red suite (R2, BUG-220) is
            # not a task failure and stays a run-log note. Without --strict
            # this stays a run-log note.
            strict_bad = True
            workspace.log_event(root, "strict: a task applied edits but left "
                                "failed blocks or failing tests", "ERROR")

        if not strict and (tr.tests_ok is False
                           and not tr.tests_preexisting_only):
            # BUG-808: without --strict this task still ticks done and the
            # run still exits 0 -- but "All tasks complete. [8 done, 0
            # failed]" after "tests/lint subprocess skipped until they are
            # fixed" read as if the fix had happened. Name it, like the
            # partial-edit precedent above: never silently.
            red_tasks.append(task)
            print("  [verification red: this task ticked done while the "
                  "last gate round was RED -- the files stay written, the "
                  "checks did not pass]", file=sys.stderr)
            workspace.log_event(
                root, f"verification red on task '{task[:60]}': ticked done "
                      f"with the last gate round red", "WARN")

        mark_green(attempted, task)
        mark_done(plan_path, task, notify=plan_changed)
        done += 1
        workspace.log_event(root, f"task done ({done})")
        notify(task)

    if verify and verify_final and not final_state["ran"] and attempted:
        # The run stopped before the last unchecked task (a cap, a time
        # limit, an early failure): the whole-goal check never came due
        # inside the loop, so pay for it once now, for the record. Repair
        # and exit-code responsibility after the loop stays with the
        # caller (auto's post-run verification pass).
        ok, out = run_verify(root, verify, cfg=cfg)
        cfg._verify_final_ran = True
        cfg._verify_final_ok = ok
        print(f"  [final-only check was not reached inside the loop; running "
              f"`{verify}` once for the record: "
              f"{'passed' if ok else 'FAILED'}]", file=sys.stderr)
        workspace.log_event(
            root, f"final-only verification for the record (the run stopped "
                  f"before the last task): "
                  f"{'passed' if ok else 'FAILED'}",
            "INFO" if ok else "ERROR")

    cfg._auto_step = 0
    cfg._static_baseline = None
    cfg._contract_baseline = None
    if ratchet_on:
        try:
            ratchet_mod.prune(
                root, int(getattr(cfg, "auto_keep_checkpoints",
                                  ratchet_mod.DEFAULT_KEEP)))
        except Exception:
            pass
    print(f"\n[{done} task(s) done, {failed} failed"
          + (f", {rolled_back} step(s) rolled back to the last green "
             f"checkpoint" if rolled_back else "")
          + (", --strict: task(s) ticked done with failed blocks or "
             "failing tests -- NOT a clean finish" if strict_bad else "")
          + "]", file=sys.stderr)
    remaining = sum(1 for ln in read_plan(plan_path) if _is_unchecked(ln))
    if remaining:
        # Stopped with work outstanding (a cap, a time limit, or a failed
        # task): say how to resume.
        print(f"  [{remaining} task(s) are still unchecked in "
              f"{plan_path.name}; rerun the same command to resume them"
              + (", or add --keep-going to run through failures"
                 if failed and stop_on_fail else "")
              + ".]", file=sys.stderr)
    workspace.log_event(root, f"auto loop finished: {done} done, "
                              f"{failed} failed, {remaining} left"
                              + (f", {rolled_back} rolled back"
                                 if rolled_back else "")
                              + (", strict: unclean task(s)" if strict_bad
                                 else ""),
                        "ERROR" if failed or strict_bad else "INFO")
    if partial_tasks:
        # The reported defect was silence: say it plainly at the end too.
        print(f"  [{len(partial_tasks)} task(s) left edit blocks that did not "
              f"apply -- see the 'partial edit' lines above and in "
              f"{workspace.run_log_path(root).name} (run log)]",
              file=sys.stderr)
        workspace.log_event(root, f"auto loop finished with "
                                  f"{len(partial_tasks)} partially applied "
                                  f"task(s)", "WARN")
    if red_tasks:
        # BUG-808: the finish line must be as loud about a red gate as it
        # is about a partial application -- a run that exits 0 after the
        # last verification round said "skipped until they are fixed" has
        # to say whether anything ever fixed them.
        print(f"  [{len(red_tasks)} task(s) ticked done while the last "
              f"verification gate was RED -- the checks did not pass; rerun "
              f"the same command to keep repairing]", file=sys.stderr)
        workspace.log_event(root, f"auto loop finished with "
                                  f"{len(red_tasks)} verification-red "
                                  f"task(s)", "WARN")
    _fire(cfg, root, "notification", message=f"pedacito auto finished: "
          f"{done} done, {failed} failed, {remaining} left"
          + (f", {rolled_back} step(s) rolled back to the last green "
             f"checkpoint" if rolled_back else ""))
    return 0 if failed == 0 and not strict_bad else 1
