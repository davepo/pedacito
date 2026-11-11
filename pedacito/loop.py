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

import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import diagnostics, events as events_mod, hooks as hooks_mod
from . import safety, workspace
from . import index as index_mod
from . import instructions as instructions_mod
from . import proc as proc_mod
from . import ratchet as ratchet_mod
from . import review as review_mod
from . import testsel
from . import todo as todo_mod
from .edits import Result, parse_edits
from .llm import LLMError

# A task line in the plan file: `- [ ] do the thing` / `- [x] done thing`.
# `*` bullets are accepted too. Group 2 is the box character, group 3 the text.
_TASK_LINE = re.compile(r"^(\s*[-*]\s*)\[([ xX])\]\s*(.*)$")

PLAN_HEADER = "# Plan\n\n"

_NOOP_PHRASES = (
    "no change", "nothing to", "nothing needs", "already ", "no edit",
    "no code change", "no code edit", "leave as", "as-is",
)


def _is_noop(reply: str) -> bool:
    """True when a zero-edit reply says there is nothing to do (the work is
    already complete). Only consulted when NO edit applied."""
    low = (reply or "").lower()
    return any(p in low for p in _NOOP_PHRASES)


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


def _set_box(path: Path, task: str, box: str) -> bool:
    """Set the checkbox of the first unchecked task matching `task` to `box`
    ("x" = done, "!" = failed). Matching by text rather than line index means
    a plan file the model edited mid-run still resolves. Returns False if the
    task could not be found."""
    lines = read_plan(path)
    i = _find_unchecked(lines, task)
    if i is None:
        return False
    m = _TASK_LINE.match(lines[i])
    lines[i] = f"{m.group(1)}[{box}] {m.group(3)}"
    path.write_text("\n".join(lines), encoding="utf-8")
    return True


def mark_done(path: Path, task: str) -> bool:
    """Tick the first unchecked task whose text matches `task` as done."""
    return _set_box(path, task, "x")


def mark_failed(path: Path, task: str) -> bool:
    """Flag the first unchecked task matching `task` as failed (`[!]`).

    A failed box is neither unchecked (so the loop never re-picks it and
    spins) nor done (so a human can see what to revisit)."""
    return _set_box(path, task, "!")


def insert_task_before(path: Path, task: str, new_title: str) -> bool:
    """Insert an unchecked ``- [ ] new_title`` line immediately before the
    first unchecked occurrence of ``task`` (the remediation path: the failed
    check stays unchecked and retries after the repair). Returns False when
    the anchor task is not found."""
    lines = read_plan(path)
    i = _find_unchecked(lines, task)
    if i is None:
        return False
    lines.insert(i, f"- [ ] {new_title}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return True


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
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


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
    "type": "array", "items": {"type": "integer"}}

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
    with `_titles_from(raw)`; [] entries where none were given."""
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
    lines = path.read_text(encoding="utf-8").splitlines()
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
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


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
    notes = instructions_mod.render(root, cfg) if root is not None else ""
    if notes:
        plan_system += "\n\n" + notes
    base_user = (f"GOAL\n\n{body}\n\n"
                 "Break this into an ordered task list. Each task title must be "
                 "a single imperative sentence a coding agent can implement and "
                 "verify in isolation.")
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

    The whole process group is killed on timeout, so a GUI/launcher child
    cannot keep a window alive after the check gives up."""
    rc, out, stopped, _ = proc_mod.run(
        command, cwd=root, timeout=timeout, shell=True, max_output=200_000)
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
    return rc == 0, out.strip()


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


def failure_identities(output: str) -> dict[str, FailureIdentity]:
    """Stable failing-test identities from a pytest run, keyed by node id.

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
    a = str(a or "").replace("\\", "/").lstrip("./")
    b = str(b or "").replace("\\", "/").lstrip("./")
    return bool(a and b and (a == b or a.endswith("/" + b)
                             or b.endswith("/" + a)))


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


def missing_modules(output: str) -> list[str]:
    """Top-level import names (``foo`` for ``foo.bar``) from
    ModuleNotFoundError/ImportError messages. Callers route these to
    dependency handling instead of the code-repair loop."""
    out: list[str] = []
    for m in _MISSING_MODULE_RE.finditer(output or ""):
        name = m.group(1).split(".")[0].strip()
        if name and name not in out:
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
    # Per-task usage: tokens, cache-hit ratio, seconds, cost (see _task_stats).
    stats: dict = field(default_factory=dict)
    # The post-edit impacted-tests verdict: None when the tests did not run
    # (verification off / no test command), else True/False from the last
    # verification round. `ask --strict` exits nonzero on False.
    tests_ok: bool | None = None
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
    baseline = getattr(cfg, "_frontend_baseline", None)
    if baseline is None:
        return errors
    try:
        from . import spec as spec_mod
        keys = {spec_mod.contract_error_key(e) for e in baseline}
        return [e for e in errors
                if spec_mod.contract_error_key(e) not in keys]
    except Exception:
        return [e for e in errors if e not in baseline]


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


def distill_repair_lesson(failure_text: str, check_label: str = ""):
    """Reduce one gate/test failure blob to (kind, one-line lesson, files),
    or None when no known pattern explains it. `check_label` is the gate
    name the loop uses ("frontend contract", "static gate", "tests", ...);
    the first recognizable pattern wins, so findings text should put the
    root failure first. Deterministic: same text, same lesson, forever."""
    text = str(failure_text or "")
    if not text.strip():
        return None
    files = sorted({m.group(0).lstrip("./")
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


def _note_repair_failure(pending: list, kind: str, text: str) -> None:
    """Remember a gate/test failure for later lesson distillation; the first
    entry per kind wins (root causes precede follow-on noise)."""
    if any(k == kind for k, _ in pending):
        return
    pending.append((kind, str(text or "")[:6000]))


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


def _post_edit_tests(root, cfg, agent, idx, task, results, reply, backup_dir,
                     baseline, edit_files):
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
    eval scorer."""
    if not (getattr(cfg, "verify_tests", False)
            and getattr(cfg, "allow_shell", False)):
        return results, reply, None
    from . import verify as verify_mod
    changed = sorted({r.file for r in results if r.ok and r.file}) or edit_files
    base_out = (baseline[1] if baseline is not None and not baseline[0]
                else "")
    base_failures = failure_identities(base_out)
    base_score = _failure_score(base_out) if base_out else 0
    max_rounds = max(0, int(getattr(cfg, "repair_rounds", 2) or 0))
    tests_ok = None
    pending_lesson: list = []      # (kind, text) this task broke and can mend
    for round_ in range(max_rounds + 1):
        syntax = verify_mod.check_syntax(results)
        if syntax:
            tests_ok = False
            if round_ < max_rounds:
                _note_repair_failure(pending_lesson, "syntax",
                                     "\n".join(str(f.message)
                                               for f in syntax))
            print(f"  [syntax: {len(syntax)} new syntax error(s) -- "
                  f"test/lint subprocess skipped until the patch parses]",
                  file=sys.stderr)
            workspace.log_event(
                root, "syntax gate rejected the patch; subprocess checks "
                      "skipped", "ERROR")
            events_mod.get(root, cfg).emit(
                "tests", ok=False, score=len(syntax), baseline=base_score,
                round=round_, syntax=True, new=len(syntax), fixed=0,
                still_failing=0, pre_existing=len(base_failures))
            if round_ >= max_rounds:
                break
            fixed = agent._repair_from_findings(task, reply, syntax, results)
            if not fixed:
                break
            new = agent._apply_once(fixed, False, backup_dir)
            if not any(r.ok for r in new):
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
            if round_ < max_rounds:
                _note_repair_failure(pending_lesson, "static",
                                     "\n".join(static))
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
                break
            new = agent._apply_once(fixed, False, backup_dir)
            if not any(r.ok for r in new):
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
            if round_ < max_rounds:      # only a lesson if a repair can follow
                _note_repair_failure(pending_lesson, "frontend contract",
                                     "\n".join(fe))
            print(f"  [frontend gate: {len(fe)} new DOM contract violation(s)"
                  f" -- tests/lint subprocess skipped until they are fixed]",
                  file=sys.stderr)
            workspace.log_event(
                root, "frontend contract gate rejected the patch; "
                      "test/lint subprocess skipped", "ERROR")
            events_mod.get(root, cfg).emit(
                "tests", ok=False, score=len(fe), baseline=base_score,
                round=round_, frontend=True, new=len(fe), fixed=0,
                still_failing=0, pre_existing=len(base_failures))
            if round_ >= max_rounds:
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
                break
            new = agent._apply_once(fixed, False, backup_dir)
            if not any(r.ok for r in new):
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
            if round_ < max_rounds:      # only a lesson if a repair can follow
                _note_repair_failure(pending_lesson, "contract", "\n".join(ce))
            print(f"  [contract gate: {len(ce)} new cross-file contract "
                  f"violation(s) -- tests/lint subprocess skipped until they "
                  f"are fixed]", file=sys.stderr)
            workspace.log_event(
                root, "interface contract gate rejected the patch; "
                      "test/lint subprocess skipped", "ERROR")
            events_mod.get(root, cfg).emit(
                "tests", ok=False, score=len(ce), baseline=base_score,
                round=round_, contract=True, new=len(ce), fixed=0,
                still_failing=0, pre_existing=len(base_failures))
            if round_ >= max_rounds:
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
                break
            new = agent._apply_once(fixed, False, backup_dir)
            if not any(r.ok for r in new):
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
        got = _run_tests(idx, cfg, changed)
        if got is None:
            return results, reply, tests_ok
        ok, out, desc = got
        if not ok and round_ < max_rounds:
            _note_repair_failure(pending_lesson, "tests", out)
        elif ok and pending_lesson:
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
        # A new identity repairs even when the total count is flat or lower;
        # a genuinely worse count (including output with no parseable ids)
        # keeps the old score-based guard.
        should_repair = bool(diff.new) or score > base_score
        if not should_repair:
            if not ok and score > 0:
                # Honest exit: the suite is still red, but every remaining
                # failure predates this task (or no identity could be parsed
                # and it was no worse than the baseline). Say so instead of
                # ending on a silent FAILED line.
                print(f"  [tests: {score} failure(s) remain; the suite "
                      f"already had {base_score} before this task, so they "
                      "predate it and were not auto-repaired]", file=sys.stderr)
            break
        if round_ >= max_rounds:
            break
        packet = traceback_repair_packet(root, diff, current=current)
        intro = ("after your edits these tests fail (they did not before):\n"
                 if diff.new else
                 "after your edits the test suite is failing more severely:\n")
        finding = verify_mod.Finding("tests", "error", "", intro + packet)
        fixed = agent._repair_from_findings(task, reply, [finding], results)
        if not fixed:
            break
        new = agent._apply_once(fixed, False, backup_dir)
        if not any(r.ok for r in new):
            break
        results = _merge_repair_results(results, new)
        reply = (reply.rstrip() + "\n\n" + fixed)
        changed = sorted({fileio_norm(r.file) for r in results if r.ok})
    return results, reply, tests_ok


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
        new = agent._apply_once(fix, False, backup_dir)
        results = _merge_repair_results(results, new)
        if not any(r.ok for r in new):
            break
        reply = (reply.rstrip() + "\n\n" + fix) if reply.strip() else fix
    return results, reply


def fileio_norm(name: str) -> str:
    from . import fileio
    return fileio.normalise_rel(name)


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
    "No changes", or None when every round came back empty."""
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


def _review(root: Path, results, task: str, cfg, backup_dir, stamp: str) -> list:
    """Serve the side-by-side review page and return the results as they
    finally stand (the page writes on Apply, so this re-stamps each result)."""
    diffs = review_mod.diffs_from_results(results, root)
    if not diffs:
        print("\n  No review page: no edit block applied cleanly.", file=sys.stderr)
        _report(results, dry_run=True)
        return results
    nonce = review_mod.new_nonce()
    page = review_mod.build_page(diffs, task, cfg.model, cfg.active_profile,
                                 nonce, interactive=True, stamp=stamp)
    saved = review_mod.save(root, page, stamp)

    def writer():
        return review_mod.write_files(root, diffs, backup_dir)

    print(f"  review page: {saved}", file=sys.stderr)
    outcome, detail = review_mod.run(
        page, nonce, writer, port=cfg.review_port,
        open_browser=cfg.open_browser, timeout=cfg.review_timeout,
        on_url=lambda u: print(f"  waiting for sign-off at {u}", file=sys.stderr),
        diffs=diffs, task=task)

    if outcome == "applied":
        print(f"  {detail}", file=sys.stderr)
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


def run_task(root: Path, cfg, client, agent, idx, task: str, *,
             apply: bool = True, review: bool = False,
             answer_system: str | None = None) -> TaskResult:
    """`_run_ladder`, plus failure memory: a task that ends without a usable
    edit is remembered (files looked at, why) so a similar later task is warned
    off the same dead end; a success clears the warning."""
    from . import memory as memory_mod
    if not getattr(cfg, "_auto_step", 0):
        # Standalone task (ask/chat/eval): no outer loop owns the static or
        # contract baseline, so drop any leftover one and let _run_task_once
        # record a fresh snapshot for this task.
        cfg._static_baseline = None
        cfg._contract_baseline = None
    tr = _run_ladder(root, cfg, client, agent, idx, task, apply=apply,
                     review=review, answer_system=answer_system)
    if answer_system is not None or review or not getattr(cfg, "use_memory", True):
        return tr
    try:
        if any(r.ok for r in tr.results) or tr.changed or _is_noop(tr.reply):
            memory_mod.clear_attempts(root, task)
        else:
            looked = [ln.split(None, 2)[-1][:60] for ln in tr.steps
                      if " read " in f" {ln} "][:4]
            why = ("fabricated edits" if tr.fabricated
                   else "no valid edit was produced")
            memory_mod.note_attempt(root, task, why, looked)
    except Exception:
        pass
    return tr


def _run_ladder(root: Path, cfg, client, agent, idx, task: str, *,
                apply: bool = True, review: bool = False,
                answer_system: str | None = None) -> TaskResult:
    """`_run_task_once`, plus the bounded recovery ladder: an unreachable or
    failing worker falls over to the `fallback` role's model, and a task that
    ends with NO edit applied gets one more attempt on the `escalate` role's
    (stronger) model. Each rung is used at most once per task, and only when
    that role is configured, so a local model's bad day cannot silently burn a
    hosted budget."""
    from . import roles as roles_mod
    from .llm import BudgetExceeded, CancelledError
    cfg._task_mutations = []      # a fresh gather-mutation ledger per task
    kw = dict(apply=apply, review=review, answer_system=answer_system)
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
    esc = roles_mod.client_for(cfg, "escalate", client)
    if esc is client or (tr.applied
                         and (any(r.ok for r in tr.results) or tr.changed)):
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
    tr2 = _run_task_once(root, cfg, esc, Agent(tr.idx, esc, cfg), tr.idx, task,
                         **kw)
    return tr2 if any(r.ok for r in tr2.results) else tr


def _run_task_once(root: Path, cfg, client, agent, idx, task: str, *,
                   apply: bool = True, review: bool = False,
                   answer_system: str | None = None) -> TaskResult:
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
    gathered = agent.gather(task, backup_dir=backup_dir)
    # Files a gather-phase action wrote/moved/deleted (write/append/move/
    # delete/rename). A worker that finishes the job with gather actions --
    # the deepclitest pattern -- produces no answer-phase edit blocks, and
    # its mutations used to bypass post-edit verification entirely.
    gather_mut = list(getattr(cfg, "_task_mutations", None) or ())
    tests_ok = None
    print("--- answering ---\n", file=sys.stderr)
    findings: list = []
    truncated = False
    refusal = ""
    contradicted = False
    fabricated = False       # reply's only blocks were no-ops
    fabricated_only = False  # ...and the reply did not declare "no changes"
    intents: list = []
    if (answer_system is None and apply and not review
            and hasattr(agent, "incremental_eligible")
            and agent.incremental_eligible(task)):
        intents = agent.plan_edit_intents(task, gathered)
        if len(intents) < 2:
            intents = []          # single target: the ordinary reply is cheaper
    incremental = bool(intents)

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

    if incremental:
        # One symbol hunk at a time, each dry-applied and verified
        # before it is written. The full impacted-test/lint pass runs once,
        # after the last hunk, exactly like the ordinary apply path.
        edit_files = sorted({
            fileio_norm(i.get("file", "")) for i in intents if i.get("file")})
        pre = _fire(cfg, root, "pre_edit", files=edit_files, task=task[:300])
        baseline_tests = None
        if edit_files and getattr(cfg, "verify_tests", False) \
                and getattr(cfg, "allow_shell", False):
            baseline_tests = _run_tests(idx, cfg, edit_files)
        lint_baseline: dict = {}
        if getattr(cfg, "post_edit_check", True):
            try:
                lint_baseline = diagnostics.quick_findings(root, edit_files)
            except Exception:
                lint_baseline = {}
        if pre.denied:
            print(f"  [edits blocked by a pre_edit hook: {pre.message}]",
                  file=sys.stderr)
            results = [Result(False, f, f"blocked by a pre_edit hook: "
                                        f"{pre.message}") for f in edit_files]
            reply = raw_reply = ""
        else:
            results, reply, findings = agent.apply_edit_intents(
                task, gathered, intents, backup_dir=backup_dir,
                verbose=True)
            raw_reply = reply
            for f in findings:
                workspace.log_event(root, f"verify: {f.render()}",
                                    "WARN" if f.severity == "warn" else "ERROR")
            if any(r.ok for r in results) or gather_mut:
                _fire(cfg, root, "post_edit",
                      files=sorted({r.file for r in results if r.ok}
                                   | set(gather_mut)))
                results, reply, tests_ok = _post_edit_tests(
                    root, cfg, agent, idx, task, results, reply,
                    backup_dir, baseline_tests,
                    sorted(set(edit_files) | set(gather_mut)))
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
                for f in findings:
                    workspace.log_event(root, f"verify: {f.render()}",
                                        "WARN" if f.severity == "warn"
                                        else "ERROR")
        elif answer_system is None and not parse_edits(raw_reply):
            # P1a shadow mode: a no-edit reply gets the grounded-prose checks
            # logged (citations, ungrounded identifiers, unaddressed
            # requirements), never acted on.
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
                    and getattr(cfg, "allow_shell", False):
                baseline_tests = _run_tests(idx, cfg, edit_files)
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
                if apply and not review and (
                        any(r.ok for r in results) or gather_mut):
                    _fire(cfg, root, "post_edit",
                          files=sorted({r.file for r in results if r.ok}
                                       | set(gather_mut)))
                    results, reply, tests_ok = _post_edit_tests(
                        root, cfg, agent, idx, task, results, reply,
                        backup_dir, baseline_tests,
                        sorted(set(edit_files) | set(gather_mut)))
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
    if not incremental and not results and not _is_noop(reply) \
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
                    results, reply, tests_ok = _post_edit_tests(
                        root, cfg, agent, idx, task, results, reply,
                        backup_dir, None,
                        sorted({e.file for e in parse_edits(reply)}))
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
    if (not incremental and not results and not contradicted and not fabricated
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
    applied = False
    if review:
        if not parse_edits(reply):
            print("\n  The model proposed no edits.", file=sys.stderr)
        results = _review(root, results, task, cfg, backup_dir, when)
        applied = any(r.ok and "applied" in r.message for r in results)
    else:
        _report(results, dry_run=not apply)
        applied = apply

    changed: list[str] = []
    if applied:
        for r in results:
            if not r.ok:
                continue
            try:
                idx = index_mod.reindex_file(idx, r.file, cfg)
                changed.append(r.file)
            except Exception:
                pass               # a deleted/moved file re-indexes on next refresh
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
                "diff": r.diff} for r in results])
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
                      steps_log, fabricated=fabricated_only,
                      backup_dir=backup_dir,
                      live_ok=getattr(agent, "live_evidence_ok", None),
                      live_kinds=_live_kinds_of(agent),
                      findings=findings, stats=stats, tests_ok=tests_ok)


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


def run_loop(root: Path, cfg, client, idx, plan_path: Path, *,
             apply: bool = True, verify: str = "", repair_rounds: int = 2,
             max_tasks: int = 0, max_minutes: int = 0,
             stop_on_fail: bool = True,
             answer_system: str | None = None, auto_install: bool = False,
             goal: str = "", on_task=None, strict: bool = False,
             verify_final: bool = False) -> int:
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

    Task handling:
    - Investigation-only tasks ("Reproduce...", "Locate the cause...") are
      answered in prose and marked done without requiring edits.
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

    attempted = 0
    done = 0
    failed = 0
    strict_bad = False
    # Tasks that ticked done with SOME of their blocks not applied: each is
    # reported at the moment it happens and in the finish line -- a partial
    # edit is by design, silence about it was the defect.
    partial_tasks: list[tuple[str, int, int]] = []
    started_at = time.monotonic()
    # Reports/edit summaries carried into later task prompts so a fresh
    # context still benefits from earlier findings.
    context_notes: list[str] = []
    workspace.log_event(root, f"auto loop: plan={plan_path.name}"
                              f" verify={verify or '(none)'}")
    _lint_plan_titles(root, plan_path)
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
        insert_task_before(plan_path, task, _REMEDIATION_TITLE)
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
            # Call positionally: existing callers (and tests) monkeypatch
            # run_verify with a two-argument callable. Record the outcome here,
            # where cfg is available, rather than threading it into the call.
            ok, out = run_verify(root, verify)
            final_state["ran"] = True
            try:
                from . import memory
                memory.observe_command(root, verify, ok, cfg, kind="test")
            except Exception:
                pass
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
        attempted += 1
        print(f"\n=== task {attempted}: {task} ===", file=sys.stderr)
        if ratchet_on:
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
            agent = Agent(idx, client, cfg)
            # Forced reads are mined from the clean title only: prior reports
            # mention files descriptively and would bloat the must-read list.
            agent.task_targets = task
            if investigate:
                tr = run_investigation_task(
                    root, cfg, agent, idx,
                    task_for_agent, answer_system=answer_system)
            else:
                tr = run_task(root, cfg, client, agent, idx,
                              task_for_agent, apply=apply, review=False,
                              answer_system=answer_system)
        except LLMError as e:
            print(f"  task failed: {e}", file=sys.stderr)
            workspace.log_event(root, f"task failed: {e}", "ERROR")
            undo_step(attempted, task, f"task failed: {e}", None)
            failed += 1
            if stop_on_fail:
                break
            mark_failed(plan_path, task)
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
                        mark_failed(plan_path, task)
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
                        mark_failed(plan_path, task)
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
                        mark_failed(plan_path, task)
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
                    mark_failed(plan_path, task)
                    continue
                context_notes.append(f"### {task}\n{_clip_report(tr.reply)}")
                _prune_notes(context_notes)   # run anchor + latest two
                mark_green(attempted, task)
                mark_done(plan_path, task)
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
            mark_failed(plan_path, task)
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
                mark_failed(plan_path, task)
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
                mark_failed(plan_path, task)
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
                        mark_failed(plan_path, task)
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
                    mark_failed(plan_path, task)
                    continue
            if _is_noop(tr.reply):
                print("  [model reports the task is already complete; "
                      "marking it done]", file=sys.stderr)
                workspace.log_event(root, "task already complete (no-op)")
                if task == _REMEDIATION_TITLE and pending_check:
                    # The repair refused to edit: remember that for the check
                    # that retries next.
                    repair_outcome[pending_check.pop()] = False
                mark_green(attempted, task)
                mark_done(plan_path, task)
                done += 1
                continue
            print(f"  [task produced no applied edits -- {'stopping' if stop_on_fail else 'skipping'}]",
                  file=sys.stderr)
            workspace.log_event(root, "task produced no applied edits", "ERROR")
            failed += 1
            if stop_on_fail:
                break
            mark_failed(plan_path, task)
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
                        workspace.log_event(
                            root, "frontend contract gate rejected the step; "
                                  "verify subprocess skipped", "ERROR")
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
            rounds = repair_rounds
            installed: set[str] = set()
            # Rollback-if-worse: the current failure's severity. A repair
            # round that raises it gets undone from its own snapshot (which
            # holds the state before that round's writes), keeping the
            # better earlier state, and repair stops.
            prev_score = None if passed else _failure_score(output, use_diag)
            lesson_seen = "" if passed else str(output)[:6000]
            while not passed:
                # A missing third-party package is NOT a code bug: editing
                # source cannot add a dependency. Install it if the user opted
                # in, otherwise report and stop.
                missing = missing_modules(output)
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
                                   answer_system=answer_system)
                except LLMError as e:
                    print(f"  repair failed: {e}", file=sys.stderr)
                    workspace.log_event(root, f"repair failed: {e}", "ERROR")
                    break
                idx = tr2.idx
                passed, output = _gated_check(tr2)
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

        if not passed:
            if dep_blocked:
                dists = ", ".join(pip_name(m) for m in dep_blocked)
                print(f"  [verification blocked by missing dependencies: "
                      f"{dists}]", file=sys.stderr)
            else:
                print(f"  [verification still failing: {check_label}]",
                      file=sys.stderr)
                workspace.log_event(
                    root, f"verify still failing: {check_label}", "ERROR")
                undo_step(attempted, task,
                          f"verification still failing: {check_label}", tr)
            failed += 1
            if stop_on_fail:
                break
            mark_failed(plan_path, task)
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
            mark_failed(plan_path, task)
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
            # A partial application: the task stays ticked (by design), but
            # the unapplied half is named here, in the run log, and again in
            # the finish line -- never silently.
            where = ", ".join(sorted({r.file for r in bad_blocks}))
            total = len(ok_edits) + len(bad_blocks)
            partial_tasks.append((task, len(ok_edits), len(bad_blocks)))
            print(f"  [partial edit: {len(bad_blocks)} of {total} block(s) "
                  f"did not apply ({where}) -- the task is still ticked "
                  f"done]", file=sys.stderr)
            workspace.log_event(
                root, f"partial edit on task '{task[:60]}': {len(bad_blocks)}"
                      f" of {total} block(s) did not apply ({where})", "WARN")

        if strict and (tr.tests_ok is False
                       or any(not r.ok for r in tr.results)):
            # The work landed but is not clean: a failed edit block (a
            # partial application) or impacted tests still red after the
            # repair rounds. Without --strict this stays a run-log note.
            strict_bad = True
            workspace.log_event(root, "strict: a task applied edits but left "
                                "failed blocks or failing tests", "ERROR")

        mark_green(attempted, task)
        mark_done(plan_path, task)
        done += 1
        workspace.log_event(root, f"task done ({done})")
        notify(task)

    if verify and verify_final and not final_state["ran"] and attempted:
        # The run stopped before the last unchecked task (a cap, a time
        # limit, an early failure): the whole-goal check never came due
        # inside the loop, so pay for it once now, for the record. Repair
        # and exit-code responsibility after the loop stays with the
        # caller (auto's post-run verification pass).
        ok, out = run_verify(root, verify)
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
             f"checkpoint" if rolled_back else "") + "]", file=sys.stderr)
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
                                 if rolled_back else ""),
                        "ERROR" if failed else "INFO")
    if partial_tasks:
        # The reported defect was silence: say it plainly at the end too.
        print(f"  [{len(partial_tasks)} task(s) ticked done with blocks that "
              f"did not apply -- see the 'partial edit' lines above and in "
              f"{workspace.run_log_path(root).name} (run log)]",
              file=sys.stderr)
        workspace.log_event(root, f"auto loop finished with "
                                  f"{len(partial_tasks)} partially applied "
                                  f"task(s)", "WARN")
    _fire(cfg, root, "notification", message=f"pedacito auto finished: "
          f"{done} done, {failed} failed, {remaining} left"
          + (f", {rolled_back} step(s) rolled back to the last green "
             f"checkpoint" if rolled_back else ""))
    return 0 if failed == 0 and not strict_bad else 1
