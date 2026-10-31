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

from . import diagnostics, workspace
from . import index as index_mod
from . import instructions as instructions_mod
from . import proc as proc_mod
from . import review as review_mod
from . import todo as todo_mod
from .edits import parse_edits
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


def _evidence_failure(task: str, tr) -> str:
    """A reason string when a verification checkpoint lacks evidence, else "".

    Pessimistic on ties. Checks, in order: the report is a real report (not
    the steps-log stand-in); its own VERDICT line is not FAILED; the prose
    does not say the verification did not succeed; and, for titles promising
    live observation, a run/browser/screenshot step actually happened."""
    if not _is_verification_task(task):
        return ""
    if (tr.reply or "").startswith("No prose report was produced"):
        return ("its report is empty (the answer phase produced no prose "
                "even after a retry); a steps log is not a verification")
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
    if _needs_live_evidence(task) and not _used_live_evidence(
            getattr(tr, "steps", ())):
        return ("it never drove the app (no run/browser/screenshot step); "
                "reading source is not live evidence")
    return ""


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
# fix the crash", "Run the build and update the docs").
_EDIT_WORDS_RE = re.compile(r"\b(?:" + "|".join((
    "fix", "implement", "add", "remove", "delete", "rename", "move",
    "update", "change", "refactor", "write", "replace", "create", "patch",
    "correct", "repair", "build", "wire", "integrate", "migrate",
)) + r")\b")


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
    """Write a fresh plan file: a goal line plus one unchecked task per title."""
    body = [PLAN_HEADER.rstrip()]
    if goal:
        body += ["", f"Goal: {goal}"]
    body += ["", *(f"- [ ] {t}" for t in titles)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


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

    Re-asks once when a title arrives at exactly the schema's length cap
    (force-closed mid-sentence), retries once with a "you returned zero
    tasks" nudge on an empty reply, and finally falls back to one task built
    from the goal. When `root` is given, the project's instruction file is
    added to the planner prompt."""
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


def run_verify(root: Path, command: str, timeout: int = 600) -> tuple[bool, str]:
    """Run `command` (a shell string) in `root`. Returns (passed, output).

    The whole process group is killed on timeout, so a GUI/launcher child
    cannot keep a window alive after the check gives up."""
    rc, out, stopped, _ = proc_mod.run(
        command, cwd=root, timeout=timeout, shell=True, max_output=200_000)
    if stopped == "timeout":
        return False, f"(verification timed out after {timeout}s)"
    if stopped == "error":
        return False, f"(could not run verification: {out})"
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
# run_investigation_task).
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
    return opens >= 3 or len(words) < 8


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
    workspace.log_event(root, f"investigate: {task}")
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
    if _looks_like_tool_calls_only(reply) or not (reply or "").strip() \
            or collapsed:
        # One retry absorbs transient empty completions and regenerates a
        # phrase-loop collapse before it reaches the context notes every
        # later task reads.
        if not collapsed:
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
    steps_log = list(getattr(agent, "steps_log", []) or [])
    if _looks_like_tool_calls_only(reply) or not (reply or "").strip():
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
        session = workspace.write_session(root, task, steps_log,
                                          reply, [], None, when)
        print(f"[session: {session.relative_to(root)}]", file=sys.stderr)
    return TaskResult([], reply, True, [], idx, session, steps_log)


def run_task(root: Path, cfg, client, agent, idx, task: str, *,
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
    backup_dir = (workspace.new_backup_dir(root, when)
                  if wants_write and cfg.backup else None)
    workspace.log_event(root, f"task: {task}")
    todo_mod.clear(root)          # a fresh checklist per task; auto has PLAN.md
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
    fabricated = False       # reply's only blocks were no-ops
    fabricated_only = False  # ...and the reply did not declare "no changes"

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

    if contradicted:
        # A reply that OPENS by refusing yet contains blocks is a small-model
        # coherence failure, not intent: honor the refusal, discard the blocks.
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
            # The reply's only blocks were null edits (SEARCH == REPLACE)
            # dropped by parse_edits: the model had concluded no edit was
            # needed. Honor that instead of re-asking for "real" blocks (the
            # marker matches _is_noop, so the re-ask below is skipped).
            files = ", ".join(dict.fromkeys(agent.noop_discards))
            workspace.log_event(root, f"reply emitted only no-op edit "
                                      f"block(s) ({files}); treated as a "
                                      f"refusal to edit", "WARN")
            reply = _mark_fabricated(reply)
    if not results and not _is_noop(reply) and not contradicted:
        # Zero edit blocks: one bounded recovery (honour read tool-calls or
        # re-ask with the edit protocol). Skipped for no-op replies ("already
        # complete") and refusals, which are coherent about doing nothing.
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
    if (not results and not contradicted and not fabricated
            and not _is_noop(reply)):
        # A "no changes" declaration is a decision, not a parse failure.
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

    steps_log = list(getattr(agent, "steps_log", []) or [])
    session = None
    if cfg.log_sessions:
        session = workspace.write_session(root, task, steps_log,
                                          reply, results, backup_dir, when,
                                          raw_reply=raw_reply if truncated else None)
        print(f"[session: {session.relative_to(root)}]", file=sys.stderr)
    return TaskResult(results, reply, applied, changed, idx, session,
                      steps_log, fabricated=fabricated_only,
                      backup_dir=backup_dir)


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
             goal: str = "", on_task=None) -> int:
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
    """
    from .agent import Agent

    attempted = 0
    done = 0
    failed = 0
    started_at = time.monotonic()
    # Reports/edit summaries carried into later task prompts so a fresh
    # context still benefits from earlier findings.
    context_notes: list[str] = []
    workspace.log_event(root, f"auto loop: plan={plan_path.name}"
                              f" verify={verify or '(none)'}")
    _lint_plan_titles(root, plan_path)

    remediated: dict[str, int] = {}     # verify title -> repair rounds used
    # A check whose inserted repair applied NO edits was judged unfixable in
    # code (e.g. the repair would re-break behavior the goal froze). Track
    # each repair's outcome per check: False = no applied edits -> when the
    # check misses again, retire it as failed and CONTINUE with the plan.
    repair_outcome: dict[str, bool] = {}  # check title -> last repair edited?
    pending_check: list[str] = []     # check whose repair runs next (0 or 1)
    retried: dict[str, int] = {}      # task title -> corrective re-runs used
    retry_note: list[str] = []        # one-shot note for the next attempt

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

    use_diag = (not verify) and bool(getattr(cfg, "auto_diagnostics", False))
    check_label = f"$ {verify}" if verify else "project diagnostics"

    def run_check(changed):
        if verify:
            return run_verify(root, verify)
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
                        # A check nobody executed says nothing about the
                        # product: re-run it once with the toolset spelled
                        # out, and only retire it (never remediate it) when
                        # the re-run stays source-only too.
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
        if not tr.applied or not ok_edits:
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
            if _is_noop(tr.reply):
                print("  [model reports the task is already complete; "
                      "marking it done]", file=sys.stderr)
                workspace.log_event(root, "task already complete (no-op)")
                if task == _REMEDIATION_TITLE and pending_check:
                    # The repair refused to edit: remember that for the check
                    # that retries next.
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
            # remediation path (retiring only applies to refusing repairs).
            repair_outcome[pending_check.pop()] = True

        # Verification gate: on failure, hand the model the command's output
        # to fix. With auto_diagnostics on and no explicit --verify, linter/
        # typechecker findings for the changed files play the same role.
        passed = True
        dep_blocked: list[str] = []
        if verify or use_diag:
            passed, output = run_check(tr.changed)
            rounds = repair_rounds
            installed: set[str] = set()
            # Rollback-if-worse: the current failure's severity. A repair
            # round that raises it gets undone from its own snapshot (which
            # holds the state before that round's writes), keeping the
            # better earlier state, and repair stops.
            prev_score = None if passed else _failure_score(output, use_diag)
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
                            passed, output = run_check(tr.changed)
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
                passed, output = run_check(tr2.changed)
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

        mark_done(plan_path, task)
        done += 1
        workspace.log_event(root, f"task done ({done})")
        notify(task)

    print(f"\n[{done} task(s) done, {failed} failed]", file=sys.stderr)
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
                              f"{failed} failed, {remaining} left",
                        "ERROR" if failed else "INFO")
    return 0 if failed == 0 else 1
