"""The autonomous outer loop (`pedacito auto`): a Ralph loop over a task list.

This is the piece that turns Pedacito from a single-shot tool into a
full-featured coding agent. ``ask`` and ``chat`` run ONE task end to end
(the *inner* loop: gather -> answer -> apply). ``auto`` wraps that in the
*outer* loop the Ralph pattern describes:

    pick the next unchecked task -> run it in a FRESH agent/context ->
    verify (optional) -> mark it done in the plan file -> repeat.

State lives in a plan file on disk (a markdown checklist), not in any
conversation. Every task starts with a brand-new ``Agent`` and therefore a
clean context window; what survives between iterations is only what the
model wrote: the code, git history, and the checkbox it just ticked. That is
the whole trick -- and the reason long autonomous runs do not rot the way a
single ever-growing session does.

Nothing here re-implements the inner loop: it calls the same ``gather`` /
``answer`` / ``apply`` path as ``ask``, so an ``auto`` task gets the identical
edit anchoring, repair rounds, backups, and session logging as a hand-typed
one. The plan/task-list file is deliberately plain markdown so a human can
edit it mid-run, and the loop just picks up the next unchecked line on the
next pass.

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

from . import index as index_mod
from . import instructions as instructions_mod
from . import review as review_mod
from . import workspace
from .edits import parse_edits
from .llm import LLMError

# A task line in the plan file: `- [ ] do the thing` / `- [x] done thing`.
# `*` bullets are accepted too. The space in the box (or `x`) is captured as
# group 2; group 3 is the task text.
_TASK_LINE = re.compile(r"^(\s*[-*]\s*)\[([ xX])\]\s*(.*)$")

PLAN_HEADER = "# Plan\n\n"


def _is_noop(reply: str) -> bool:
    """True when a task produced no edit blocks because there is nothing to
    do -- the model read the file and reported it already complete. Only
    consulted when NO edit applied, so an accidental match cannot mask a real
    failure that did produce blocks; a genuinely empty/errored reply has none
    of this language and still counts as a failure."""
    low = (reply or "").lower()
    return any(p in low for p in (
        "no changes", "no change", "nothing to", "nothing needs",
        "already ", "is already", "no edit", "no code change",
        # "no code edit": run 13d's first repair reply -- "No code edit
        # is appropriate here" -- matched NONE of the patterns above
        # ("no edit" needs the words adjacent), so the honest refusal
        # fell into the re-ask loop instead of the no-op path.
        "no code edit", "leave as", "as-is", "already exists",
        "already correct",
    ))


def _normalized_reply(reply: str) -> str:
    """Lowercase with the apostrophe styles models emit folded to ASCII, so
    refusal/incomplete phrasing matches whatever quoting came back. Local
    servers often double-encode UTF-8: the right single quote (U+2019) shows
    up as "â" ("canât"), which naive regexes silently miss."""
    return ((reply or "").lower()
            .replace("\u2019", "'")
            .replace("â", "'"))


# A reply that OPENS by declaring it made no edit. Narrow on purpose -- the
# caller uses this to DISCARD edit blocks the same reply contains, so only
# bare refusals count: "No edit.", "No edits were made:", "I did not edit
# any file". Deliberately NOT matched: "No changes were needed in main.js;
# I fixed index.html instead." -- a qualifier after the phrase means the
# sentence is about one file, not a refusal to have edited at all.
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


# Zero-edit replies that DECLARE THE WORK NOT DONE -- as opposed to _is_noop's
# legitimate "nothing to do". Graveyard-run task 4 ("Move each file identified
# as dead code into the 'graveyard' folder") answered "I can't safely emit the
# move blocks yet ... To continue, the first 3-5 lines of each file must be
# read ... I won't guess" with ZERO edits -- and an incidental "graveyard/ has
# already been created" (a PRIOR task's work) matched _is_noop, ticking the
# move done. The plan then reported a move that never happened; only the third
# verify session caught the empty graveyard. Matched on _normalized_reply.
_INCOMPLETE_RES = (
    re.compile(r"\bcan(?:not|'t)\s+(?:\w+\s+){0,3}?"
               r"(?:emit|apply|make|produce|write|complete|finish|move|"
               r"create|edit|proceed|continue|do)\b"),
    re.compile(r"\bunable to\s+(?:\w+\s+){0,3}?"
               r"(?:emit|apply|make|produce|write|complete|finish|move|"
               r"create|edit|proceed|continue|do)\b"),
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
    completed now ("can't emit the blocks yet -- to continue, the files must
    be read"). Such a reply is an incomplete task, never a no-op: the work it
    defers was possible in the lookup phase, so the loop retries it once and
    fails it loudly on a second deferral. Repair tasks are exempt at the call
    site -- refusing to edit is a sanctioned repair outcome (see bug N)."""
    low = _normalized_reply(reply)
    return any(r.search(low) for r in _INCOMPLETE_RES)


# Outcome-negative regexes, backing the substring list below. Run 7 ticked
# two verify tasks done whose reports said the opposite: task 4's "could not
# *visually* verify" (an adverb split every listed substring) and task 7's
# literal first line "Verification failed in the captured run" -- an outcome
# statement, not an inability, and the browser HAD been driven, so the
# objective gate passed too. The run ended "7 done, 0 failed" on a game that
# still missed its spec.
# Deliberately NOT matched: a bare "not ... confirmed" or a be-verb form
# ("the executing code path is not yet confirmed") -- honest reproduce
# reports list what they could not confirm as diligence while still
# confirming the symptoms they were asked about, and failing those would
# stop every run at its first task. Inability verbs (can't/could not/unable/
# failed to) and the verify family are the unambiguous signals.
_UNVERIFIED_RES = (
    # Inability, tolerating up to 4 intervening words and a hyphenated verb:
    # "could not visually verify", "could not re-verify", "can't produce a
    # live visual verification". Curly apostrophes occur in real model
    # output, so both ' and \u2019 are accepted.
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
    # "Still unknown: whether the sprites render" -- run 10's twice-repeated
    # false green in verdict-less form: the report's FIRST LINE named the
    # task's own claim as unknown ("does not show a grown frog", "does not
    # provide ... evidence" all escaped the patterns above). The `whether`
    # window separates a core-claim unknown from a passing caveat ("Still
    # unknown: direct confirmation from index.html ..." -- a legitimately
    # PASSED report in the same run); the 40-char leash keeps that caveat
    # style out. Pessimism is deliberate: a peripheral "still unknown
    # whether" costs one bounded remediation round, a missed one costs the
    # run's truthfulness.
    re.compile(r"\bstill unknown\b[^.\n]{0,40}?\bwhether\b"),
)


# Fix ZE (run 13g): the frog check FINALLY passed clean -- full per-row
# white/black counts quoted, discrete 2x2 cells confirmed, bare "VERDICT:
# PASSED" -- and was then retired as FAILED because its "Likely cause of
# the previous failure" section said the EARLIER eval "could not confirm
# white eyes": the confession matcher read the whole report and could not
# tell HISTORY from CONFESSION. Retry flows invite exactly that narration
# (the check re-runs with the prior FAILED note in context, and a good
# report explains it), so a PASSED verdict is now phrase-matched against
# the report with prior-attempt sentences stripped. A sentence counts as
# historical when it attributes something to a prior attempt ("the earlier
# live eval", "the previous failure") or continues a dropped sentence as a
# pronoun ("It did not count ... so it could not confirm ..."). Present-
# tense first-person confessions ("I could not verify the tail point in
# the capture") carry no attribution and survive the pruning -- the
# pessimistic gate still fails them.
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
    re.I)
_PRONOUN_START_RE = re.compile(
    r"^[\s\-*\u2022\d.)]*\b(it|its|it['\u2019]s|this|that|these|those|"
    r"they|them|their|there|the same|said|which|both)\b", re.I)


def _strip_prior_attempt_prose(reply: str) -> str:
    """The report minus sentences describing PRIOR attempts (Fix ZE).

    Sentence-scoped, attribution-driven, and deliberately over-eager on
    pronoun continuations: dropping a kept-shape sentence costs nothing
    (the matcher only looks for negatives), while a single surviving
    historical "could not confirm" flips a passed check to failed. The
    VERDICT line never carries an attribution or a pronoun head, so it
    always survives -- and `_verdict_of` reads the FULL reply anyway;
    only the phrase gate sees the pruned text. Sentences are rejoined with
    newlines, never spaces, so pruning cannot glue two kept fragments
    into an accidental confession phrase.
    """
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


def _reply_says_unverified(reply: str) -> bool:
    """True when the report says the verification did NOT succeed: either it
    confesses it could not perform the check the task asked for ("I can't
    launch the browser in this final phase", "could not visually verify"), or
    it drove the app and reports a negative outcome ("Verification failed in
    the captured run"). A checkpoint that confesses either must not be ticked
    done: that is how an auto run ends all-green with its edits in the wrong
    file. Models reword these admissions constantly -- adverbs split phrases,
    outcomes arrive as results -- so the substrings are backed by the
    `_UNVERIFIED_RES` regexes, and both are only consulted for tasks whose
    TITLE asks to verify/confirm/observe, so an incidental phrase in a fix
    task's prose cannot fail it. Phrase matching alone is still brittle;
    `_evidence_failure` pairs it with the objective tool-use signal."""
    low = (reply or "").lower()
    if any(p in low for p in (
            "can't verify", "cannot verify", "could not verify",
            "couldn't verify", "unable to verify",
            "can't confirm", "cannot confirm", "could not confirm",
            "unable to confirm",
            "can't launch the browser", "cannot launch the browser",
            "could not launch the browser", "unable to launch the browser",
            "can't run the game", "cannot run the game",
            "could not run the game",
            "can't produce a live visual verification",
            "cannot produce a live visual verification",
            "no live visual verification", "not visually verified",
            # Rewordings observed in the wild: a locate-and-verify checkpoint
            # that only read source, and a color-fix task in the wrong file.
            "not directly verified", "was not verified", "is not verified",
            "no live run", "no live verification",
            "could not be verified", "cannot be verified", "can't be verified",
            "without running the", "without a live run",
            "did not run the", "didn't run the", "never ran the",
    )):
        return True
    return any(r.search(low) for r in _UNVERIFIED_RES)


# Run 10 ended all-green with two checks whose reports opened "Still unknown:
# whether <the task's own claim>" -- the phrase patterns above missed the
# wording, and prose rewording will always outrun a pattern list. So
# verification reports must END with a machine-readable verdict line (the
# brief is appended in run_investigation_task) and _evidence_failure reads it
# first. Last match wins: a report may quote the brief's format before
# stating its own verdict. The prefix class tolerates markdown bold/bullets
# ("**VERDICT: FAILED**", "- VERDICT: PASSED") -- small models decorate
# exactly those lines.
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
    """The clean title line out of a possibly-assembled worker prompt.

    run_loop hands workers the assembled text (ORIGINAL USER GOAL block +
    CURRENT TASK + context notes), but title-keyed decisions -- like the
    verdict brief in run_investigation_task -- must not read the goal block:
    it quotes the spec, whose prose can contain verify words that would put
    PASSED/FAILED on every locate/reproduce task. With no goal block the
    title is simply the first line."""
    body = task or ""
    marker = "\nCURRENT TASK\n"
    if marker in body:
        body = body.split(marker, 1)[1]
    return body.split("\n", 1)[0]


# Clipping budget for the investigation reports carried between tasks. Local
# models run tight contexts, so the notes are capped -- but the answer brief
# puts caveats ("Still unknown", "Next check", live-path doubts) ANYWHERE in
# a report, often after the confident conclusion. A 3,000-char head-only clip
# lost a mid-report "nothing shows this file is even loaded by the running
# app" warning and later tasks edited dead code; observed reports run 4-7k
# chars, and two notes at 8k chars (~4k tokens) fit every context tier this
# loop actually drives. Clip head AND tail only past the budget.
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


# Tasks whose deliverable includes live evidence (a running app, a
# screenshot, observed behavior) are told up front that tools exist only in
# the lookup phase. Models reliably discover this in the answer phase, when
# it is too late ("I can't launch the browser in this final phase"), and
# then pass the checkpoint on prose alone.
_EVIDENCE_WORDS = re.compile(
    r"\b(run|runs|running|launch|launches|verify|verifies|verification|"
    r"confirm|confirms|observe|observes|observing|reproduce|reproduces|"
    r"screenshot|screenshots|browser|visually|render|renders|rendering|"
    r"game)\b")


def _evidence_hint(task: str) -> str:
    """A task-text addendum for live-evidence tasks; empty for code tasks."""
    if not _EVIDENCE_WORDS.search((task or "").lower()):
        return ""
    return ("\n\nEVIDENCE: this task is about live behavior. The `run`, "
            "`screenshot`, and `browser` tools exist ONLY in the lookup "
            "phase; the final answer phase has no tools. Launch the app and "
            "capture the evidence while gathering, before you answer -- a "
            "verification postponed to the answer phase cannot happen, and "
            "reporting the task done without that evidence fails it.")


# A steps_log entry looks like "4. browser python main.py [wait 3]  (reason)".
# Anchor on the ACTION token: a `read` step whose reason mentions "run the
# game" is not live evidence.
_LIVE_STEP = re.compile(r"^\d+\.\s+(?:run|browser|screenshot)\b")

# Title words that promise live observation. Narrower than _EVIDENCE_WORDS
# on purpose: paired with a verify word below, these titles are only
# satisfied by evidence from the RUNNING app. "Check the render path" stays
# a code-archaeology task; "Run the game and verify ..." does not.
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


def _evidence_failure(task: str, tr) -> str:
    """A reason string when a verification checkpoint lacks evidence, else "".

    Three independent signals, pessimistic on ties: the report's own VERDICT
    line (verification reports must end with VERDICT: PASSED/FAILED -- see
    _VERDICT_BRIEF; a FAILED verdict fails the check no matter how
    optimistic the prose around it is, and a PASSED verdict never overrides
    the other two -- though for the phrase gate its prior-attempt narration
    is stripped first, Fix ZE), the report saying the verification did not
    succeed -- it
    could not be performed, or it was performed and came back negative
    (phrase/regex match; models reword admissions constantly) -- and the
    session never driving the app at all (objective: no
    run/browser/screenshot step, checked for titles that promise live
    observation). A confident report over read-only lookups still fails the
    last gate; a browser-driving report that says "Verification failed"
    still fails the second."""
    if not _is_verification_task(task):
        return ""
    if (tr.reply or "").startswith("No prose report was produced"):
        # The answer phase returned nothing (even after the retry in
        # run_investigation_task) and the loop synthesized this stand-in
        # from the steps log. A steps log is not a verification: the
        # checkpoint's deliverable is a report, and an empty one carries
        # neither evidence nor caveats.
        return ("its report is empty (the answer phase produced no prose "
                "even after a retry); a steps log is not a verification")
    state, reason = _verdict_of(tr.reply)
    if state == "FAILED":
        return ("its own verdict line says VERDICT: FAILED"
                + (f": {reason[:200]}" if reason else ""))
    # Fix ZE (run 13g): a clean PASSED verdict is phrase-matched against
    # the report minus its prior-attempt narration -- the frog check
    # passed with fully quoted measurements and was still retired as
    # failed because its "Likely cause of the previous failure" section
    # said the EARLIER eval "could not confirm white eyes". Verdict-less
    # and FAILED reports keep the full-text match (pessimistic on ties).
    if _reply_says_unverified(tr.reply if state != "PASSED"
                              else _strip_prior_attempt_prose(tr.reply)):
        return ("its report says the verification did not succeed "
                "(could not be performed, or was performed and failed)")
    if _needs_live_evidence(task) and not _used_live_evidence(
            getattr(tr, "steps", ())):
        return ("it never drove the app (no run/browser/screenshot step); "
                "reading source is not live evidence")
    return ""


# Tasks that are checkpoints rather than edits. The planner explicitly emits
# "reproduce the failure" as a step for bugs, but the loop used to demand an
# applied edit from every task -- so a successful reproduction was recorded as
# a failure and `stop_on_fail` ended the run before the fix task. These titles
# are recognised as investigation-only and answered in prose.
_INVESTIGATE_START = (
    "reproduce", "investigate", "diagnose", "analyze", "analyse", "inspect",
    "trace", "locate", "find", "confirm", "check", "verify", "review",
    "profile", "explain", "identify", "determine", "observe", "test",
    # "Run the game and verify the fix" is a checkpoint, not a code change:
    # on the edit path its success condition is "an edit applied", which let
    # a live-verification task pass on a cosmetic edit to the wrong file
    # with no evidence gathered at all. The edit-word guard below keeps
    # "Run the build and update the docs" on the edit path.
    "run",
    # Reporting/survey verbs the planner can emit for a "look at the code
    # first" step. "Audit ... to identify which of X are already working" is
    # the canonical case: it must be answered as a report, not forced through
    # the edit path (which, via the no-edit recovery, made the model rewrite
    # PLAN.md with its findings instead of moving on to the implementation
    # tasks). The edit-word guard below still keeps any title that also says
    # "fix"/"implement"/"add" on the edit path.
    "audit", "assess", "evaluate", "report", "summarize", "summarise",
    "list", "enumerate",
)
# If any of these appear, the task also changes code and must stay on the edit
# path ("Reproduce and fix the crash", "Locate and update the config").
_EDIT_WORDS = (
    "fix", "implement", "add", "remove", "delete", "rename", "move",
    "update", "change", "refactor", "write", "replace", "create", "patch",
    "correct", "repair", "build", "wire", "integrate", "migrate",
)


def _is_investigation_task(task: str) -> bool:
    """True for a task whose deliverable is evidence, not a code change.

    Heuristic by necessity: the plan is plain markdown the user can edit. A
    leading investigation verb with no edit verb anywhere keeps the task out
    of the apply/edit path; anything containing an edit verb stays there."""
    low = (task or "").strip().lower()
    first = re.match(r"[a-z]+", low)
    if not first or not first.group(0).startswith(_INVESTIGATE_START):
        return False
    return not re.search(r"\b(" + "|".join(_EDIT_WORDS) + r")\b", low)


# --------------------------------------------------------------------------
# The plan file: a markdown checklist that is the loop's external memory
# --------------------------------------------------------------------------
def read_plan(path: Path) -> list[str]:
    """Return the plan file's lines (empty list if it does not exist)."""
    if path.exists():
        return path.read_text(encoding="utf-8").split("\n")
    return []


def next_task(lines: list[str]) -> tuple[int, str] | None:
    """Return (line_index, task_text) of the first unchecked task, or None.

    Order is the order of the file, which is also priority order: the plan is
    written dependencies-first, so "highest priority" is simply "first line".
    """
    for i, ln in enumerate(lines):
        m = _TASK_LINE.match(ln)
        if m and m.group(2).strip() == "":
            return i, m.group(3).strip()
    return None


def _set_box(path: Path, task: str, box: str) -> bool:
    """Set the checkbox of the first unchecked task matching `task` to `box`
    ("x" = done, "!" = failed). Matching by text rather than line index means
    a plan file the model edited mid-run still resolves. Returns False if the
    task could not be found."""
    lines = read_plan(path)
    for i, ln in enumerate(lines):
        m = _TASK_LINE.match(ln)
        if m and m.group(2).strip() == "" and m.group(3).strip() == task:
            lines[i] = _TASK_LINE.sub(
                lambda mm: f"{mm.group(1)}[{box}] {mm.group(3)}", ln, count=1)
            path.write_text("\n".join(lines), encoding="utf-8")
            return True
    return False


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
    first unchecked occurrence of ``task``. Used by the remediation path:
    the failed check must stay unchecked (so it retries after the repair),
    and the plan runs top-down, so "before" is the position that makes the
    repair execute first. Returns False when the anchor task is not found."""
    lines = read_plan(path)
    for i, ln in enumerate(lines):
        m = _TASK_LINE.match(ln)
        if m and m.group(2).strip() == "" and m.group(3).strip() == task:
            lines.insert(i, f"- [ ] {new_title}")
            path.write_text("\n".join(lines), encoding="utf-8")
            return True
    return False


# The repair task title inserted when a live verification checkpoint fails
# its evidence gate. Bug H: run 7's failed verify carried an exact repair
# recipe ("the GBGBG rows at main.js L213/L245 are still live") and no task
# ever acted on it, because the plan is static. The title is deliberately
# crafted to (a) route to the EDIT branch (no investigation start word) and
# (b) carry no verify/confirm/observe/validate/check words, so the evidence
# gate does not demand browser work from the repair itself -- the retried
# check right after it does the observing. The diagnosis travels in the
# FAILED LIVE TEST context note, never in this title (titles are mined for
# forced file reads and must stay clean).
# Run 10 added the frozen-behavior clause: its first field remediation
# "fixed" a failed ~10s-survival check by moving enemy spawning from 3s to
# 12s (a later one offset the spawn timers by -12) -- passing the check by
# violating the goal's frozen "spawn timing unchanged" constraint. That is
# the Goodhart trap in its purest form: the loop optimizes the signal it
# reads (the gate) and discards the intent it cannot (the spec). A repair
# whose only route to green breaks frozen behavior must refuse and surface
# the conflict instead of re-tuning the program under test. Scan-clean:
# freezes/conflict/instead add no verify words and no start word.
_REMEDIATION_TITLE = (
    "Repair the defect described in the FAILED LIVE TEST note in your "
    "context: apply the smallest code fix that resolves it, keeping every "
    "other constraint from the goal -- if the fix would change behavior the "
    "goal freezes, do not edit; report the conflict instead")


def _prune_notes(notes: list[str]) -> None:
    """Trim the context-note window IN PLACE: keep the FIRST note plus the
    latest two.

    The first note is the run's reproduce/observe report (PLAN_SYSTEM always
    makes task 1 one), and its live-path findings are the anchor every later
    task needs: run 9's task 6 "replaced the frog" in the dead
    static/js/entities/player.js because a plain latest-two window had
    already evicted task 1's "player.js is not confirmed live" analysis --
    the exact warning that would have redirected the edit. Middle notes
    still roll off; only the anchor is pinned."""
    if len(notes) > 3:
        del notes[1:-2]


def write_plan(path: Path, goal: str, titles: list[str]) -> None:
    """Write a fresh plan file: a goal line plus one unchecked task per title."""
    body = [PLAN_HEADER.rstrip()]
    if goal:
        body += ["", f"Goal: {goal}"]
    body += ["", *(f"- [ ] {t}" for t in titles)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Planning: turn a goal into an ordered task list (one LLM call)
# --------------------------------------------------------------------------
TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string", "maxLength": 400},
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    # 800, and `title` is the ONLY field: the old 300 cap
                    # guillotined every spec-derived title mid-sentence
                    # (run 13's plan: "resolving {scale, pixe", "counts of
                    # non-b") because grammar-constrained sampling force-
                    # closes a string AT maxLength -- valid JSON, no
                    # truncation flag -- and run 13b showed the worse half:
                    # derailed by the forced close, the planner ended the
                    # array after task 1, five of six spec checks were never
                    # planned, and `auto` exited 0 with "all tasks
                    # complete". An `acceptance` sibling used to sit here
                    # and was silently DROPPED by _titles_from; one field is
                    # one place for the criterion, and generate_tasks
                    # re-asks when a title still arrives at the cap.
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
  visible effect. Title them with how to observe ("Run the game and verify
  the snake is twice as tall in the browser"): the loop FAILS a verification
  task that never drove the app.
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
                   root: Path | None = None) -> list[str]:
    """Ask the model to break `spec_text` into an ordered task list.

    Retries once with an explicit "you returned zero tasks" nudge when the
    first reply comes back empty (a model handed a bug-fix goal sometimes
    reads it as "nothing to build"), re-asks once when a title arrives at
    exactly the schema's length cap (grammar-constrained sampling force-
    closed it mid-sentence -- Bug Y, runs 13/13b), and finally falls back
    to one task built from the goal itself rather than aborting the whole
    run.

    When `root` is given, the project's instruction file is included in the
    planner prompt too -- a plan that ignores "never edit generated/" is
    wrong before the first edit happens."""
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
        TASK_SCHEMA, max_tokens=max_tokens, disable_thinking=True)
    titles = _titles_from(raw)
    cap = TASK_SCHEMA["properties"]["tasks"]["items"]["properties"][
        "title"]["maxLength"]
    cut = [t for t in titles if len(t) >= cap]
    if titles and cut:
        # Bug Y (runs 13/13b): the cap guillotine again, this time in the
        # planner. A title at exactly `cap` chars was force-closed
        # mid-sentence (valid JSON, no flag), and run 13b showed the
        # forced close can derail the model into ending the task array
        # right there: one truncated task got planned, five of six spec
        # checks were silently skipped, and the loop exited 0 with "all
        # tasks complete". One bounded re-ask; if titles are STILL cut,
        # accept them -- they remain workable (the full goal rides along
        # in every task's background context) and looping burns budget.
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
            TASK_SCHEMA, max_tokens=max_tokens, disable_thinking=True)
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
            TASK_SCHEMA, max_tokens=max_tokens, disable_thinking=True)
        titles = _titles_from(raw)
    if not titles:
        first = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
        return [f"Address: {first[:200] or 'the requested change'}"]
    return titles


# --------------------------------------------------------------------------
# Verification: the feedback loop the outer loop depends on
# --------------------------------------------------------------------------
def run_verify(root: Path, command: str, timeout: int = 600) -> tuple[bool, str]:
    """Run `command` (a shell string) in `root`. Returns (passed, output).

    A Ralph loop without a feedback signal is a chaos engine -- it keeps
    churning out code no one checked. This is the signal: pass/fail plus the
    command's output, fed back to the model for repair on failure. The whole
    process group is killed on timeout, so a GUI/launcher child cannot keep a
    window alive after the check gives up."""
    from . import proc as proc_mod
    rc, out, stopped, _ = proc_mod.run(
        command, cwd=root, timeout=timeout, shell=True, max_output=200_000)
    if stopped == "timeout":
        return False, f"(verification timed out after {timeout}s)"
    if stopped == "error":
        return False, f"(could not run verification: {out})"
    return rc == 0, out.strip()


# --------------------------------------------------------------------------
# Missing third-party dependencies
# --------------------------------------------------------------------------
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
    """Top-level import names from ModuleNotFoundError/ImportError messages.

    A failing verify often means a third-party dependency is absent rather
    than the code being wrong; editing source cannot add one, so callers use
    this to route the failure to dependency handling instead of the repair
    loop. Returns the top-level module (``foo`` for ``foo.bar``)."""
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
    come from model-generated code, so installing them is never automatic."""
    dists = sorted({pip_name(p) for p in packages
                    if re.fullmatch(r"[A-Za-z0-9_.-]+", p or "")})
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


# --------------------------------------------------------------------------
# One task, end to end (the shared inner loop)
# --------------------------------------------------------------------------
@dataclass
class TaskResult:
    """What one task run produced: the edit results, the reply, whether any
    writes landed, the files re-indexed, and the (possibly refreshed) index.
    `steps` is the lookup phase's action log ("4. browser ..."), kept so the
    outer loop can tell objectively whether a verification task ever drove
    the app instead of trusting the report's prose."""
    results: list
    reply: str
    applied: bool
    changed: list[str]
    idx: "index_mod.Index"
    session: Path | None = None
    steps: list[str] = field(default_factory=list)
    # True when the reply's only edit blocks were null edits (SEARCH ==
    # REPLACE) and the reply did NOT itself declare the work complete. The
    # outer loop must not treat that as "already done": it is a failed
    # attempt to act (the graveyard run's task 2 ticked itself off this way
    # after three discarded no-op blocks).
    fabricated: bool = False


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


# --------------------------------------------------------------------------
# Answer-phase recovery: a reply with no edit blocks
# --------------------------------------------------------------------------
# Small models in the ANSWER phase sometimes emit another agent framework's
# tool-call syntax (Anthropic `<invoke name="read">`, or the `<tool_call>` /
# `<function=read>` dialect) instead of search/replace blocks -- usually
# because the task needs source the gather phase never read. The recovery
# below parses those requests, honours the reads from disk, and re-asks for
# the edit blocks. Both dialects are matched so the exact syntax the model
# imitates does not matter.
_READ_INVOKE_RE = re.compile(
    r"<invoke\b[^>]*name\s*=\s*['\"]?read['\"]?[^>]*>(.*?)</invoke>",
    re.DOTALL | re.IGNORECASE)
# A reply can be cut off before the closing tag (the exact case that made a
# model spin: `<tool_call>read` repeated with no `</tool_call>`), so match up
# to the next opening tag or end-of-reply rather than requiring closure.
_READ_TOOLCALL_RE = re.compile(
    r"<tool_call>(.*?)(?=</tool_call>|<tool_call>|\Z)",
    re.DOTALL | re.IGNORECASE)
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
            # A degenerate reply can repeat one path for tens of thousands of
            # characters. That is not a real request; do not carry it into a
            # recovery prompt (and do not hand it to Path.resolve()).
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
        # Accept both `<function=read>`/JSON `"name": "read"` and the bare
        # `<read>` wrapper a degraded reply emits. A body that names another
        # tool is skipped so a requested write is not mistaken for a read.
        if not re.search(r"function\s*=\s*read\b|<read\b|"
                         r"['\"]name['\"]\s*:\s*['\"]read['\"]",
                         body, re.IGNORECASE):
            continue
        path = _file_param(body)
        if path:
            found.append((m.start(), path))
    # A bare <read> block without a <tool_call> wrapper (Qwen's native shape
    # in the reported runaway output).
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


def _recover_no_edit_reply(root: Path, cfg, agent, task: str, gathered: str,
                           reply: str, *, dry_run: bool, backup_dir,
                           answer_system: str | None = None):
    """Recover an answer reply that parsed to zero edit blocks.

    Two failure classes are worth one retry each (bounded at two rounds
    total): the model asked to READ more files via tool-call syntax (honour
    the reads from disk and re-ask with that source included), or it simply
    answered in prose (re-ask with an explicit 'no tools here, emit edit
    blocks' instruction). Returns (results, reply) from the first re-ask
    that produces blocks, or None when every round came back empty."""
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
            # Fix X2 (run 13d): the re-ask took the escape hatch task2
            # offers -- an honest "No changes: <reason>" line -- and the
            # old loop DISCARDED it: only block-producing rounds were
            # returned, so the model was pressed a second time, both
            # honest replies were thrown away (None kept the original
            # prose), and the run stopped on a task that had correctly
            # concluded no edit was warranted. Propagate the declaration
            # instead: run_task keeps the parse-failure ERROR silent (a
            # decision, not a parse failure) and run_loop's _is_noop path
            # treats the repair as the refusal it is (repair_outcome
            # False -> bug N retires the failing check instead of
            # looping).
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
        on_url=lambda u: print(f"  waiting for sign-off at {u}", file=sys.stderr))

    if outcome == "applied":
        print(f"  {detail}", file=sys.stderr)
        return [replace(r, message="applied via review") if r.ok else r
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
"""

# Appended to INVESTIGATE_SYSTEM for verification-titled investigations (see
# run_investigation_task). Wording rules learned from the field: "partially
# confirmed is FAILED" (run 10's final check ticked done on partial evidence
# plus honest unknowns), "reading source is never live evidence" (the
# perennial read-only pass), and an explicit do-not-hedge line because the
# gate resolves PASSED-with-confession to a miss anyway -- the model should
# not have to discover that asymmetry by watching its own runs.
_VERDICT_BRIEF = """

Your report must END with a verdict as its very last line, exactly one of
these two forms:
VERDICT: PASSED -- only when EVERY part of the task's claim was confirmed by
live evidence you gathered in this session.
VERDICT: FAILED: <one-sentence reason> -- in every other case. A partially
confirmed claim is FAILED. Reading source code is never live evidence. Do not
hedge: the loop reads this line mechanically and acts on it."""

_TOOL_SYNTAX = ("<tool_call", "<invoke", "<function=", "<read", '"name":')


def _looks_like_tool_calls_only(reply: str) -> bool:
    """True when a prose answer is really just agent-framework tool syntax.

    Small models occasionally ignore "no tools here" and emit `<tool_call>` /
    `<invoke>` blocks in the answer phase. For an investigation task that is
    not an answer, so the caller substitutes the gathered evidence."""
    low = (reply or "").lower()
    if not low.strip() or not any(t in low for t in _TOOL_SYNTAX):
        return False
    # A runaway reply can be mostly a giant repeated PATH, which the old
    # word-count heuristic mistook for prose. Strip tags, paths, and JSON
    # punctuation first; then a few repeated tool-call opens are enough.
    opens = len(re.findall(r"<tool_call\b|<invoke\b|<function\s*=", reply,
                           re.IGNORECASE))
    plain = re.sub(r"<[^>]+>", " ", reply)
    plain = re.sub(r"\S*[/\\]\S*", " ", plain)      # paths
    plain = re.sub(r"[{}\[\]\"]", " ", plain)
    words = re.findall(r"[A-Za-z]{3,}", plain)
    return opens >= 3 or len(words) < 8


def _prose_collapse(reply: str) -> bool:
    """True when a prose reply's tail degenerated into a phrase loop.

    The graveyard run streamed a task answer cycling through quoted fragments
    ("...loaded by the page / used by the game runtime / executed / after
    moving the dead code files...") -- a sampling collapse no existing guard
    covers: llm.py's stream guard counts tool-call syntax opens, not prose.
    The signal here is vocabulary novelty in the tail: a looping window reuses
    a tiny word set, while even highly parallel legitimate prose (file tables,
    per-module bullet lists) keeps unique paths and names on every line, so
    paths and code are deliberately NOT stripped -- their uniqueness is what
    separates a table from a loop. Detection only ever triggers a
    regeneration, never a mutation: a false positive costs one extra answer
    call, a false negative ships looping garbage into the context notes every
    later task reads. The 0.22 floor sits well under real collapse ratios
    (the field tail measured ~0.10) and well over legitimate parallel prose
    (a 23-row file table measures ~0.28 -- every row keeps its own path).
    Not applied to build replies: sprite grids and other legitimate
    repetitive edit content would false-positive, and their stream already
    has the truncation-continuation path."""
    words = re.findall(r"\w+", (reply or "")[-1500:].lower())
    if len(words) < 60:
        return False
    return len(set(words)) / len(words) < 0.22


def run_investigation_task(root: Path, cfg, client, agent, idx, task: str, *,
                           answer_system: str | None = None) -> TaskResult:
    """Run a reproduction/diagnosis task whose deliverable is a report.

    Unlike run_task(), no edits are expected or applied. The gather phase uses
    the normal tools (including `run`), then the answer phase reports findings
    in prose. This is what makes the planner's "reproduce the failure" step a
    success instead of a no-edit failure. If the model answers with tool-call
    syntax anyway, the gathered evidence is recorded as the report so the next
    fix task (and the session log) still has something concrete to work from.
    """
    when = workspace.stamp()
    workspace.log_event(root, f"investigate: {task}")
    print("\n--- looking things up ---", file=sys.stderr)
    gathered = agent.gather(task)
    print("--- reporting ---\n", file=sys.stderr)
    system = answer_system or INVESTIGATE_SYSTEM
    if answer_system is None and _is_verification_task(_task_title(task)):
        # Fix L: force the binary verdict line, keyed off the clean TITLE
        # (the assembled prompt's goal block quotes the spec, whose verify
        # words would otherwise put the brief on every investigation).
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
    if _looks_like_tool_calls_only(reply) or not (reply or "").strip() \
            or collapsed:
        # An empty answer phase leaves a verification checkpoint with no
        # report at all -- run 8's final verify ticked done on the steps-log
        # fallback below plus a 15-word `done` reason. One retry absorbs the
        # transient empty completions local models produce before we fall
        # back to synthesizing a report from the lookups. The same retry
        # regenerates a phrase-loop collapse (bug R, graveyard run): nothing
        # is poisoned yet -- catching it here keeps looping garbage out of
        # the context notes every later task reads.
        if not collapsed:
            # Fix ZD: run 13f's empty answer left NO trace in the event log
            # between "reporting" and the stand-in -- no way to tell a
            # model flake from a harness bug after the fact. (The collapsed
            # case prints and logs its own WARN above.)
            print("  [the answer phase returned no usable prose; "
                  "retrying once]", file=sys.stderr)
            workspace.log_event(root, "answer phase produced no prose "
                                "(empty or tool-calls-only completion); "
                                "retrying once", "WARN")
        retry = agent.answer(task, gathered,
                             system=system,
                             include_edit_format=False, stream=True)
        if (retry or "").strip() and not _looks_like_tool_calls_only(retry):
            reply = retry
    if _looks_like_tool_calls_only(reply) or not (reply or "").strip():
        steps = "\n".join(getattr(agent, "steps_log", []) or ["(no steps)"])
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
        session = workspace.write_session(root, task,
                                          getattr(agent, "steps_log", []),
                                          reply, [], None, when)
        print(f"[session: {session.relative_to(root)}]", file=sys.stderr)
    return TaskResult([], reply, True, [], idx, session,
                      list(getattr(agent, "steps_log", []) or []))


def run_task(root: Path, cfg, client, agent, idx, task: str, *,
             apply: bool = True, review: bool = False,
             answer_system: str | None = None) -> TaskResult:
    """Run ONE task through the inner loop: gather source, answer (or edit),
    apply or review, re-index the touched files, and write a session log.

    `agent` and `idx` are passed in (not created here) so the caller controls
    freshness -- the autonomous loop hands each task a brand-new Agent over
    the same warm index, while `ask`/`chat` reuse one. `answer_system`, when
    given, replaces the default edit brief (used to implement scaffold stubs
    rather than edit existing code). Returns a TaskResult."""
    when = workspace.stamp()
    wants_write = apply or review
    backup_dir = (workspace.new_backup_dir(root, when)
                  if wants_write and cfg.backup else None)
    workspace.log_event(root, f"task: {task}")
    print("\n--- looking things up ---", file=sys.stderr)
    gathered = agent.gather(task, backup_dir=backup_dir)
    print("--- answering ---\n", file=sys.stderr)
    raw_reply = agent.answer(task, gathered, system=answer_system)
    truncated = bool(getattr(client, "stream_truncated", False))
    if truncated:
        workspace.log_event(root, "answer reply truncated at the output limit",
                            "WARN")
    refusal = _edit_refusal_head(raw_reply)
    contradicted = bool(refusal) and bool(parse_edits(raw_reply))
    fabricated = False       # reply's only blocks were no-ops (Fix X)
    fabricated_only = False  # ...and the reply did not declare "no changes"
    def _mark_fabricated(text: str) -> str:
        """Record null-only blocks; stamp the refusal marker only when the
        reply itself was not already an honest "no changes" declaration.
        Stamping a genuinely complete reply as fabricated would make the
        outer loop retry work that is already done."""
        nonlocal fabricated, fabricated_only
        fabricated = True
        if _is_noop(text):
            return text
        fabricated_only = True
        return (text.rstrip() + "\n\n[No changes: every edit block in the "
                "reply was a no-op (SEARCH == REPLACE) and was discarded; "
                "nothing was written.]")
    if contradicted:
        # Bug U (graveyard run): the inserted repair's prose said "No edit."
        # and explained why acting would violate the goal's freeze -- then
        # the same reply emitted a FILE block stripping 22 script tags from
        # index.html, and the harness applied it. When a reply OPENS by
        # refusing, its blocks are a small-model coherence failure, not
        # intent: honor the refusal and discard them.
        print(f"  [reply opens by refusing edits (\"{refusal}\") but "
              f"contains edit blocks; honoring the refusal and discarding "
              f"them]", file=sys.stderr)
        workspace.log_event(root, f"reply refused edits ('{refusal}') but "
                                  f"contained edit blocks; blocks discarded",
                            "WARN")
        results, reply = [], raw_reply
    else:
        results, reply = agent.apply(raw_reply, dry_run=review or not apply,
                                     task=task, gathered=gathered,
                                     backup_dir=None if review else backup_dir)
        if not results and getattr(agent, "noop_discards", None):
            # Fix X (run 13): the reply's only blocks were null edits
            # (SEARCH == REPLACE), dropped by parse_edits -- a fabricated
            # compliance with the edit protocol by a model that had
            # concluded no edit was needed (run 13's last two repairs were
            # pressed into exactly this by the re-ask below, on a file the
            # goal froze). Honor that conclusion instead of re-asking for
            # "real" blocks: the marker matches _is_noop, so the re-ask is
            # skipped and the outer loop's no-op path treats the repair as
            # the refusal it was (repair_outcome False -> the failing check
            # retires instead of burning another verify round).
            files = ", ".join(dict.fromkeys(agent.noop_discards))
            workspace.log_event(root, f"reply emitted only no-op edit "
                                      f"block(s) ({files}); treated as a "
                                      f"refusal to edit", "WARN")
            reply = _mark_fabricated(reply)
    if not results and not _is_noop(reply) and not contradicted:
        # The answer phase produced zero edit blocks. Before declaring the
        # task dead, give the model one bounded recovery: honour any read
        # tool-calls it emitted, or re-ask with the edit protocol spelled
        # out -- both without touching the no-op path ("already complete"
        # replies must still mark the task done, not re-ask). A refused
        # reply is coherent about doing nothing: re-asking with "produce the
        # edit blocks now" would pressure the model into undoing its own
        # refusal.
        rec = _recover_no_edit_reply(
            root, cfg, agent, task, gathered, reply,
            dry_run=review or not apply,
            backup_dir=None if review else backup_dir,
            answer_system=answer_system)
        if rec is not None:
            results, reply = rec
            if results:
                workspace.log_event(
                    root, "answer re-asked; recovered edit blocks", "INFO")
            else:
                # Fix X2: the re-ask came back with an honest "No
                # changes:" declaration instead of blocks; it is now the
                # reply (run_loop keys its refusal path off
                # _is_noop(tr.reply)).
                workspace.log_event(root, "re-ask declared no changes; "
                                          "honored as a refusal to edit",
                                    "WARN")
        elif not contradicted and getattr(agent, "noop_discards", None):
            # The re-ask itself produced only fabricated null edits
            # (parse_edits dropped them): same treatment as the
            # first-answer case above.
            files = ", ".join(dict.fromkeys(agent.noop_discards))
            workspace.log_event(root, f"re-ask produced only no-op edit "
                                      f"block(s) ({files}); treated as a "
                                      f"refusal to edit", "WARN")
            reply = _mark_fabricated(reply)
    if (not results and not contradicted and not fabricated
            and not _is_noop(reply)):
        # Fix X2: a "no changes" declaration is a decision the loop
        # honors (run_loop marks the task done or retires the repair),
        # not a parse failure -- an ERROR here would only muddy the
        # event trail of a clean refusal.
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
        agent.index = idx

    session = None
    if cfg.log_sessions:
        session = workspace.write_session(root, task,
                                          getattr(agent, "steps_log", []),
                                          reply, results, backup_dir, when,
                                          raw_reply=raw_reply if truncated else None)
        print(f"[session: {session.relative_to(root)}]", file=sys.stderr)
    return TaskResult(results, reply, applied, changed, idx, session,
                      list(getattr(agent, "steps_log", []) or []),
                      fabricated=fabricated_only)


# --------------------------------------------------------------------------
# The outer loop itself
# --------------------------------------------------------------------------

# Corrective re-runs are one-shot: a second identical failure is a verdict,
# not a transient. Shared by the procedural-verify re-run (bug T) and the
# incomplete-task retry (bug V); both keep the task unchecked so the next
# iteration picks it up again, with a one-shot note appended to its prompt.
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

# Fix ZD (run 13f): the frog check's gather phase succeeded PERFECTLY --
# a flat async-IIFE eval resolved the full per-row profile and the
# done-step reasoning confirmed the criterion -- then BOTH answer attempts
# streamed zero visible content (Qwen3 thinking runaway on Ollama), the
# loop synthesized the steps-log stand-in, and NO branch caught it: the run
# stopped, stranding three healthy checks. An empty report over gathered
# evidence is a report-writer flake, not a product verdict.
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
             goal: str = "", on_task=None) -> int:
    """Run the Ralph loop until the plan is empty (or a cap/stop is hit).

    Each iteration: read the plan, take the first unchecked task, run it in a
    fresh ``Agent`` (fresh context), optionally verify, then tick the task off
    and continue. Returns a process exit code (0 = no failed tasks).

    - ``apply``         write edits (False = the loop would tick tasks it never
                        wrote, so callers pass True; the flag exists for symmetry
                        with run_task).
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
    - Investigation-only tasks ("Reproduce...", "Locate the cause...") are
      answered in prose and marked done without requiring edits.
    - Verification checkpoints (titles asking to verify/confirm/observe) are
      FAILED, not ticked done, when their report says the verification did
      not succeed -- it could not be performed ("could not visually verify")
      or was performed and came back negative ("Verification failed in the
      captured run") -- or, for titles promising live observation ("Run the
      game and verify ..."), when the session never drove the app (no
      run/browser/screenshot step), or when the answer phase produced no
      prose at all even after one retry (the synthesized steps-log stand-in
      is not a report). An applied edit is not evidence, even with the shell
      verify green; a driven browser is not a pass when the report says the
      claim failed.
    - When a verification checkpoint fails its evidence gate, the loop does
      not give up on the plan: the failed report becomes a FAILED LIVE TEST
      context note and a repair task is inserted into the plan immediately
      before the still-unchecked check, which retries after the repair
      lands. Bounded per check by ``repair_rounds`` (a check whose report
      was empty gets no repair -- there is no diagnosis to act on); once
      the budget is spent the failure ends the run as before.
    - A verification that fails WITHOUT ever driving the app (no
      run/browser/screenshot step) failed as a check RUN, not as a product
      test: it is re-run once with the toolset spelled out and, if the
      re-run stays source-only too, retired as unverified. Remediation is
      never offered on zero evidence (bug T).
    - A build task whose reply declares the work incomplete ("can't emit
      the blocks yet -- to continue, the files must be read") with zero
      edits applied is retried once, then failed -- never ticked done as a
      no-op, even when the reply incidentally contains "already" (bug V).
    - ``auto_install`` when a verification fails on a missing third-party
                        module, pip-install it and retry (opt-in: the module
                        name comes from generated code). When off, the failure
                        is reported as a dependency problem and the code-repair
                        loop is skipped, since editing source cannot add a dep.
    - ``goal``          the user's original spec text. Every task prompt
                        carries it as background, so constraints the planner
                        dropped from a title ("keeping the doubled height")
                        still bind the worker. Forced file-reads are mined
                        from the task TITLE only (agent.task_targets), never
                        from the goal or report prose.
    """
    from .agent import Agent

    attempted = 0
    done = 0
    failed = 0
    started_at = time.monotonic()
    # Investigation tasks produce a report, not an edit. Carry the latest
    # reports into subsequent task prompts so a fresh context still benefits
    # from the reproduction -- the plan file itself stays untouched.
    context_notes: list[str] = []
    workspace.log_event(root, f"auto loop: plan={plan_path.name}"
                              f" verify={verify or '(none)'}")

    def tick(count: int) -> bool:
        """Return False once max_tasks (if any) is reached."""
        return max_tasks == 0 or count < max_tasks

    remediated: dict[str, int] = {}     # verify title -> repair rounds used
    # Bug N (run 11): a check whose inserted repair applied NO edits is a
    # check the repair worker judged unfixable in code -- run 11's planner
    # titled task 1 "observe the spawn timing to CONFIRM the 12-second
    # delay", the inserted repair instead did the goal's work (restored the
    # 3-second schedule), and the next repair refused to re-break the code
    # ("would conflict with the goal"). Retrying such a check is futile,
    # stopping strands every downstream task, and a resume replays the same
    # deadlock forever. Track each repair's outcome per check: False =
    # completed with no applied edits -> when the check misses again,
    # retire it as failed and CONTINUE with the rest of the plan.
    repair_outcome: dict[str, bool] = {}  # check title -> last repair edited?
    pending_check: list[str] = []     # check whose repair runs next (0 or 1)
    retried: dict[str, int] = {}      # task title -> corrective re-runs used
    retry_note: list[str] = []        # one-shot note for the next attempt

    def offer_remediation(task: str, tr, miss: str) -> bool:
        """Bug H: a failed verification checkpoint should not just end the
        run -- its report is often an exact repair recipe, and a static plan
        gives no task ownership of it. Record the diagnosis as a context
        note and insert a repair task BEFORE the still-unchecked check: the
        loop self-heals within the run, and the check retries afterwards.
        Bounded per check by repair_rounds, so a verify that keeps failing
        still ends the run loudly. Returns True when a repair was inserted
        (caller must then continue instead of failing the task)."""
        if (tr.reply or "").startswith("No prose report was produced"):
            return False        # the answer phase was empty: there is no
                                # diagnosis to act on, and a repair task
                                # would be guessing. Fail loudly instead.
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
        pending_check[:] = [task]       # replace: the repair is inserted
                                        # immediately before its check, so
                                        # its outcome belongs to THIS check
        print(f"  [verification task failed: {miss} -- inserted a repair "
              f"task ahead of it; the check retries after "
              f"({repair_rounds - used - 1} repair round(s) left)]",
              file=sys.stderr)
        workspace.log_event(root, f"verification failed ({miss}); "
                                  f"remediation task inserted", "WARN")
        return True

    while tick(attempted):
        lines = read_plan(plan_path)
        nxt = next_task(lines)
        if nxt is None:
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

        investigate = _is_investigation_task(task)
        task_for_agent = task + _evidence_hint(task)
        if goal:
            # The workers used to see only their title and prior reports:
            # a constraint that lived solely in the spec ("keep the doubled
            # height") was lost the moment the planner's title dropped it,
            # and the edit silently violated the user's goal.
            task_for_agent = (
                "ORIGINAL USER GOAL (background -- the CURRENT TASK below is "
                "the step to do now; every constraint in the goal still "
                "binds it):\n" + _clip_report(goal, budget=4000)
                + "\n\nCURRENT TASK\n" + task_for_agent)
        if context_notes:
            task_for_agent = (
                task_for_agent + "\n\nCONTEXT FROM PRIOR INVESTIGATION\n\n"
                + "\n\n".join(context_notes))
        if retry_note:
            # One-shot corrective note (bug T's procedural verify re-run or
            # bug V's incomplete-task retry): appended LAST so it is the
            # freshest thing the worker reads, and popped so it can never
            # leak into an unrelated task.
            task_for_agent += "\n\n" + retry_note.pop()
        try:
            agent = Agent(idx, client, cfg)
            # Forced reads ("files the task will change") are mined from the
            # clean title only. Mining the assembled prompt turned prior
            # reports' descriptive file mentions -- even an explicit "do NOT
            # change snake_spawner.py yet" list -- into a must-read list that
            # buried verify tasks in dead-code reads until no steps were left
            # to drive the browser.
            agent.task_targets = task
            if investigate:
                tr = run_investigation_task(
                    root, cfg, client, agent, idx,
                    task_for_agent, answer_system=answer_system)
            else:
                tr = run_task(root, cfg, client, agent, idx,
                              task_for_agent, apply=apply, review=False,
                              answer_system=answer_system)
        except LLMError as e:
            print(f"  task failed: {e}", file=sys.stderr)
            workspace.log_event(root, f"task failed: {e}", "ERROR")
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
                    if not _used_live_evidence(tr.steps):
                        # Bug T (graveyard run): verify 1 and verify 2 both
                        # ended with zero run/browser/screenshot steps --
                        # "no live verification possible without tools", a
                        # confabulation primed by carried notes saturated
                        # with "no evidence was captured". The loop treated
                        # that WORKER failure as a PRODUCT defect and
                        # inserted repairs; the first one stripped 22 script
                        # tags from index.html while its prose said "No
                        # edit". A check nobody executed says nothing about
                        # the product: re-run it once with the toolset
                        # spelled out, and only retire it -- never
                        # remediate it -- when the re-run stays source-only
                        # too.
                        tries = retried.get(task, 0)
                        if tries < _MAX_TASK_RETRIES:
                            retried[task] = tries + 1
                            retry_note[:] = [_VERIFY_RETRY_NOTE]
                            print(f"  [verification task failed: {miss} -- "
                                  f"but it never drove the app, so the "
                                  f"failure describes the check run, not "
                                  f"the product; re-running it once]",
                                  file=sys.stderr)
                            workspace.log_event(
                                root, "verification ended without driving "
                                      "the app (procedural failure); "
                                      "re-running once, no remediation",
                                "WARN")
                            continue
                        print("  [verification could not be executed: two "
                              "attempts ended without driving the app -- "
                              "retiring the check as unverified (not a "
                              "product failure) and continuing]",
                              file=sys.stderr)
                        workspace.log_event(
                            root, "verification check retired: two attempts "
                                  "never drove the app", "ERROR")
                        failed += 1
                        mark_failed(plan_path, task)
                        repair_outcome.pop(task, None)
                        continue
                    if (tr.reply or "").startswith(
                            "No prose report was produced"):
                        # Fix ZD (run 13f): the gather phase succeeded --
                        # live evidence present, measurement complete --
                        # and the ANSWER phase came back empty even after
                        # its own retry (Qwen3 spent the whole 20k budget
                        # on invisible reasoning, twice). The stand-in is
                        # not a verdict and carries no diagnosis: nothing
                        # to remediate, nothing real to fail. Re-run the
                        # check once; a second empty answer retires it as
                        # unverified (a procedural failure, never a
                        # product verdict) and the plan CONTINUES -- run
                        # 13f stranded three healthy checks because this
                        # case fell through to the stop-on-fail tail.
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
                        # Bug N: the repair inserted for THIS check completed
                        # with no edits -- it judged the check unfixable in
                        # code (run 11: refusing to reintroduce the defect
                        # the goal had just removed). Another repair round
                        # would refuse again, and stopping strands every
                        # downstream task while a resume replays the same
                        # deadlock forever. Retire the check loudly (failed
                        # box, event log, stderr) and let the plan continue.
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
                mark_done(plan_path, task)
                done += 1
                workspace.log_event(root, "investigation done")
                if on_task is not None:
                    try:
                        on_task(task)
                    except Exception:
                        pass
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
        if not tr.applied or not ok_edits:
            if task != _REMEDIATION_TITLE and _declares_incomplete(tr.reply):
                # Bug V (graveyard run): task 4 "Move each file..." answered
                # "I can't safely emit the move blocks yet -- to continue,
                # the files must be read" with ZERO edits, and an incidental
                # "graveyard/ has already been created" (a prior task's
                # work) matched _is_noop: the move was ticked DONE without
                # ever happening, and only the third verify session caught
                # the empty graveyard. A reply describing work still
                # required is never a no-op. The deferred work WAS possible
                # in the lookup phase (read/run have the full toolset), so
                # retry once with that spelled out, then fail loudly.
                # Repair tasks are exempt: refusing to edit is their
                # sanctioned conflict-report (bug N's retire keys off it).
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
            if getattr(tr, "fabricated", False) and task != _REMEDIATION_TITLE:
                # The reply's only blocks were null edits (SEARCH == REPLACE)
                # and it did not itself declare the work complete. _is_noop
                # used to match the synthetic marker run_task appended and
                # tick the task done -- the graveyard run's collision.py task
                # "completed" without a single lookup or move. Retry once with
                # the concrete move/rename instructions, then fail loudly.
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
            if _is_noop(tr.reply):
                print("  [model reports the task is already complete; "
                      "marking it done]", file=sys.stderr)
                workspace.log_event(root, "task already complete (no-op)")
                if task == _REMEDIATION_TITLE and pending_check:
                    # The repair refused to edit (run 11: "Reintroducing the
                    # 12-second delay ... would conflict with the goal").
                    # Remember the refusal for the check that retries next.
                    repair_outcome[pending_check.pop()] = False
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
            # remediation path (bug N's retire rule only fires for repairs
            # that REFUSED to edit).
            repair_outcome[pending_check.pop()] = True

        # Verification: the loop's quality gate. On failure, hand the model
        # the command's output and let it fix its own work -- the feedback
        # loop that separates a Ralph loop from a file-generator.
        passed = True
        dep_blocked: list[str] = []
        if verify:
            passed, output = run_verify(root, verify)
            rounds = repair_rounds
            while not passed:
                # A missing third-party package is NOT a code bug: editing
                # source cannot add a dependency, so sending it to the repair
                # loop just burns rounds. Handle it as a dependency problem --
                # install it if the user opted in, otherwise report and stop.
                missing = missing_modules(output)
                if missing:
                    dists = ", ".join(pip_name(m) for m in missing)
                    if auto_install:
                        print(f"  [verify needs missing dependencies: {dists}; "
                              f"installing]", file=sys.stderr)
                        workspace.log_event(
                            root, f"verify missing deps: {dists}; installing",
                            "WARN")
                        ok, inst = install_modules(missing, root)
                        workspace.log_event(
                            root, f"pip install {'ok' if ok else 'FAILED'}: {dists}",
                            "INFO" if ok else "ERROR")
                        if ok:
                            passed, output = run_verify(root, verify)
                            continue          # a dependency is not a repair round
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
                workspace.log_event(root, f"verify failed: $ {verify}", "ERROR")
                fix_task = (
                    f"Your last change failed this check:\n\n"
                    f"$ {verify}\n{output[:6000]}\n\n"
                    f"Fix the code so that command succeeds. Re-run it in your "
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
                passed, output = run_verify(root, verify)
                rounds -= 1

        if not passed:
            if dep_blocked:
                dists = ", ".join(pip_name(m) for m in dep_blocked)
                print(f"  [verification blocked by missing dependencies: "
                      f"{dists}]", file=sys.stderr)
            else:
                print(f"  [verification still failing: $ {verify}]",
                      file=sys.stderr)
                workspace.log_event(root, f"verify still failing: $ {verify}",
                                    "ERROR")
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
            # An applied edit is not evidence: a task that asks to verify
            # live behavior has to show it drove the app, even with edit
            # blocks on the floor and the shell verify green.
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
            # Later tasks must not trust pre-edit reports: after a fix
            # landed, a verify task tried to reconcile the stale "duplicated
            # block at L212-216" note against post-fix source and half
            # concluded the browser was serving a cache. State the change
            # explicitly so the notes carry their own expiry.
            summary = (tr.reply or "").split("--- FILE:")[0].strip()
            context_notes.append(
                f"### {task}\nEDITS APPLIED to: "
                f"{', '.join(sorted(set(tr.changed)))}.\n"
                + (summary[:400] + "\n" if summary else "")
                + "These files changed on disk AFTER the reports above; line "
                  "numbers and defect descriptions in earlier notes may be "
                  "stale -- re-read the current file state before quoting it.")
            _prune_notes(context_notes)   # run anchor + latest two

        mark_done(plan_path, task)
        done += 1
        workspace.log_event(root, f"task done ({done})")
        if on_task is not None:
            try:
                on_task(task)
            except Exception:
                pass

    print(f"\n[{done} task(s) done, {failed} failed]", file=sys.stderr)
    remaining = 0
    for ln in read_plan(plan_path):
        m = _TASK_LINE.match(ln)
        if m and m.group(2).strip() == "":
            remaining += 1
    if remaining:
        # The loop ended with work outstanding -- a cap, a time limit, or (the
        # default) a stop on the first failed task. Say so plainly, with the
        # two ways forward; without this a stopped run looks like the tool
        # silently gave up after one task.
        print(f"  [{remaining} task(s) are still unchecked in "
              f"{plan_path.name}; rerun the same command to resume them"
              + (", or add --keep-going to run through failures"
                 if failed and stop_on_fail else "")
              + ".]", file=sys.stderr)
    workspace.log_event(root, f"auto loop finished: {done} done, "
                              f"{failed} failed, {remaining} left",
                        "ERROR" if failed else "INFO")
    return 0 if failed == 0 else 1
