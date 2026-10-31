"""Offline tests for the autonomous outer loop (`pedacito auto`).

Covers the plan-file checklist helpers, the goal->task planning call, and the
Ralph loop driver itself (task ordering, marking done/failed, verification
repair round-trips, and the stop-on-fail / keep-going paths). No LLM server
required: the inner loop is monkeypatched away so the OUTER loop logic is what
is under test.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import loop
from pedacito.config import Config


# ------------------------------------------------------------------- plan file
def test_write_and_read_plan_roundtrip(tmp_path):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "build a todo CLI", ["task one", "task two"])
    text = plan.read_text()
    assert "# Plan" in text and "Goal: build a todo CLI" in text
    assert "- [ ] task one" in text and "- [ ] task two" in text
    assert loop.read_plan(plan) == text.split("\n")


def test_next_task_and_mark_done(tmp_path):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["first", "second", "third"])
    lines = loop.read_plan(plan)
    idx, task = loop.next_task(lines)
    assert task == "first"
    assert loop.mark_done(plan, "first")
    idx, task = loop.next_task(loop.read_plan(plan))
    assert task == "second"
    # A task already done is never re-picked, and [x] boxes are not unchecked.
    assert "- [x] first" in plan.read_text()


def test_mark_failed_is_skipped_not_retried(tmp_path):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["doomed", "next"])
    assert loop.mark_failed(plan, "doomed")
    idx, task = loop.next_task(loop.read_plan(plan))
    assert task == "next", "a failed task must not be re-picked by the loop"
    assert "- [!] doomed" in plan.read_text()


def test_mark_unknown_task_returns_false(tmp_path):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["known"])
    assert loop.mark_done(plan, "not in the plan") is False


# ----------------------------------------------------------- task kinds
def test_investigation_tasks_are_recognised():
    assert loop._is_investigation_task(
        "Reproduce the black screen by running main.py")
    assert loop._is_investigation_task("Locate the cause of the crash")
    assert loop._is_investigation_task("Check the render path")
    assert loop._is_investigation_task(
        "Audit the current game code to identify which of the seven requested "
        "enhancements are already implemented and working")
    assert loop._is_investigation_task("Assess which endpoints are covered")
    assert loop._is_investigation_task("Report what the scheduler does")
    assert loop._is_investigation_task(
        "Run the game and verify the snake renders with a diamond head")
    assert not loop._is_investigation_task("Fix the black screen")
    assert not loop._is_investigation_task("Reproduce and fix the crash")
    assert not loop._is_investigation_task("Implement the parser")
    assert not loop._is_investigation_task("Audit and fix the collision logic")
    assert not loop._is_investigation_task(
        "Run the build and update the changelog")


def test_verification_tasks_are_recognised():
    assert loop._is_verification_task(
        "Run the game and verify the snake renders taller")
    assert loop._is_verification_task("Confirm the head colors changed")
    assert loop._is_verification_task("Observe the snake in the browser")
    assert loop._is_verification_task(
        "Change the head colors and verify in the browser")
    assert not loop._is_verification_task(
        "Double the snake's body height while keeping its width")
    # `check` only as a whole word: a checksum is not a checkpoint.
    assert not loop._is_verification_task("Fix the checksum padding")


def test_unverified_admissions_are_detected():
    """The reported failure: the final task said "I can't launch the browser
    in this final phase, so I can't produce a live visual verification" --
    and the loop ticked it done anyway because an edit applied. Models
    reword the admission constantly ("No live run ... not directly
    verified"), so the list has to stay broad."""
    assert loop._reply_says_unverified(
        "I can't launch the browser in this final phase, so I can't produce "
        "a live visual verification. I made the head frame use black-green-"
        "black; the draw method already doubles the height.")
    assert loop._reply_says_unverified(
        "The server would not start; unable to verify the rendering.")
    assert loop._reply_says_unverified(
        "No live run/screenshot output was available in the gathered "
        "evidence, so the rendered snake in the running app is not "
        "directly verified.")
    assert loop._reply_says_unverified(
        "I could not verify any yellow head pixels in the provided source.")
    assert not loop._reply_says_unverified(
        "Verified in the browser: the snake is 180px tall with a diamond "
        "head (screenshot browser-2026.png).")


def test_negative_outcome_reports_count_as_unverified():
    """Run 7 ticked two verify tasks done whose reports said the opposite:
    task 4's "could not visually verify" (an adverb split every listed
    substring) and task 7's literal first line "Verification failed in the
    captured run" (an outcome, not an inability -- the browser HAD been
    driven, so the objective gate passed too)."""
    assert loop._reply_says_unverified(
        "Verification failed in the captured run: after pressing Enter and "
        "waiting about 10 seconds, the live browser was not still in play.")
    assert loop._reply_says_unverified(
        "Still unknown: I could not visually verify a one-head snake in the "
        "running browser.")
    assert loop._reply_says_unverified(
        "I could not re-verify the live browser in this final phase.")
    assert loop._reply_says_unverified(
        "The verification did not pass: the sprite still shows two heads.")


def test_honest_caveats_are_not_failed_verifications():
    """The flip side, from the same run: task 1 confirmed the symptoms it
    was asked to confirm while honestly listing what it could NOT confirm
    ("the executing code path is not yet confirmed", "plausible but not
    unambiguously confirmed"). Failing caveat prose would stop every run at
    its reproduce step -- only inability verbs and the verify family count
    as negative signals."""
    assert not loop._reply_says_unverified(
        "The live browser behavior is confirmed, but the executing code path "
        "is not yet confirmed; the duplication is visually plausible but not "
        "unambiguously confirmed by the captured text alone.")
    assert not loop._reply_says_unverified(
        "Verified in the browser: one head at the top, score 987 after 10s.")
    assert not loop._reply_says_unverified(
        "This verifies the fix did not regress the palette.")


def test_synthesized_steps_log_report_fails_verification():
    """Run 8's final verify: the answer phase returned nothing, so the loop
    synthesized a "report" from the steps log, and the checkpoint ticked
    done on a 15-word `done` reason ("snake sprites visible with single
    head at top"). A steps log is not a verification -- but for a non-verify
    investigation title the fallback still stands, since its findings feed
    the next task's context notes."""
    tr = types.SimpleNamespace(
        reply="No prose report was produced; the lookup phase recorded:\n\n"
              "3. browser python main.py [press Enter; wait 10]  (Look.)",
        steps=["3. browser python main.py [press Enter; wait 10]  (Look.)"])
    assert "empty" in loop._evidence_failure(
        "Run the game and verify the snake has one head", tr)
    assert loop._evidence_failure("Locate the snake frames", tr) == ""


def test_verdict_of_parses_the_last_verdict_line():
    """Last match wins (a report may quote the brief's format before its own
    verdict), and markdown bold/bullets around the line must not defeat the
    parse -- small models decorate exactly those lines."""
    assert loop._verdict_of("All good.\nVERDICT: PASSED") == ("PASSED", "")
    assert loop._verdict_of(
        "The brief says end with VERDICT: PASSED or VERDICT: FAILED.\n"
        "Everything rendered.\n\n"
        "**VERDICT: FAILED: the tail never tapered**") == (
        "FAILED", "the tail never tapered")
    assert loop._verdict_of("no verdict anywhere") == ("", "")


def test_evidence_failure_reads_the_verdict_line():
    """Run 10's terminal false green: task 11's report opened "Still
    unknown: whether all five visual changes are actually visible" and was
    ticked done -- no phrase pattern matched the wording. A FAILED verdict
    must fail the check even when the surrounding prose is optimistic."""
    tr = loop.TaskResult(
        [], "The game runs and sprites were drawn; great progress overall.\n"
            "\nVERDICT: FAILED: the sprite details were never shown live",
        True, [], None,
        steps=["1. browser python main.py [wait 3]  (Look.)"])
    miss = loop._evidence_failure(
        "Run the game and verify the sprites render", tr)
    assert "VERDICT: FAILED" in miss
    assert "never shown live" in miss


def test_evidence_failure_pessimistic_on_contradictory_verdict():
    """A PASSED verdict never overrides the phrase gate: when the body
    confesses the check could not be performed, the contradiction resolves
    against the pass -- pessimistic ties are the whole point of the gate."""
    tr = loop.TaskResult(
        [], "I could not verify the tail point in the capture.\n\n"
            "VERDICT: PASSED",
        True, [], None,
        steps=["1. browser python main.py [wait 3]  (Look.)"])
    assert loop._evidence_failure(
        "Run the game and verify the tail tapers", tr)


def test_evidence_failure_accepts_a_clean_passed_verdict():
    tr = loop.TaskResult(
        [], "The capture shows the snake falling head-first, its tail "
            "tapering to a single pixel.\n\nVERDICT: PASSED",
        True, [], None,
        steps=["1. browser python main.py [wait 3]  (Look.)"])
    assert loop._evidence_failure(
        "Run the game and verify the snake orientation", tr) == ""


def test_prior_attempt_narration_does_not_fail_a_passed_verdict():
    """Fix ZE (run 13g): the frog check FINALLY passed clean -- per-row
    white/black counts quoted, discrete 2x2 cells confirmed, bare
    "VERDICT: PASSED" -- and was retired as FAILED because its "Likely
    cause of the previous failure" section (retry flows invite exactly
    that narration) said the EARLIER eval "could not confirm white eyes
    or black mouth pixels". The confession matcher must not read history
    as confession. Present-tense first-person admissions still fire (the
    pinned pessimism above): they carry no prior-attempt attribution."""
    frog_report = (
        "Observed live reproduction:\n\n"
        "- row 70: `4` white pixels\n- row 82: `4` black pixels\n\n"
        "Likely cause of the previous failure:\n\n"
        "- The earlier live eval only classified pixels as background "
        "versus non-background.\n"
        "- It did not count `rgb(255,255,255)` or `rgb(0,0,0)` separately, "
        "so it could not confirm white eyes or black mouth pixels.\n"
        "- This session's eval added color-specific counts and produced "
        "the required measurements.\n\n"
        "Smallest next fix:\n\n"
        "- No game-code fix is required.\n\n"
        "VERDICT: PASSED")
    pruned = loop._strip_prior_attempt_prose(frog_report)
    assert not loop._reply_says_unverified(pruned)
    assert "VERDICT: PASSED" in pruned          # the verdict always survives
    tr = loop.TaskResult([], frog_report, True, [], None,
                         steps=["3. browser python main.py [wait 1]  (M.)"])
    assert loop._evidence_failure(
        "Verify frog sprite details by measured pixels", tr) == ""
    # The pinned pessimism, restated against the pruner: a PRESENT-tense
    # confession under a PASSED verdict still fails the check.
    tr2 = loop.TaskResult(
        [], "I could not verify the tail point in the capture.\n\n"
            "VERDICT: PASSED",
        True, [], None, steps=["1. browser python main.py [w]  (L.)"])
    assert loop._evidence_failure(
        "Run the game and verify the tail tapers", tr2)
    # Documented trade-off: a CURRENT-attempt failure narrated with an
    # attempt noun ("the first eval could not confirm ...") is pruned too
    # -- accepted, because that shape under a PASSED verdict is a
    # self-contradiction, while the frog's shape recurs on every retry
    # that passes.
    assert not loop._reply_says_unverified(loop._strip_prior_attempt_prose(
        "The first eval could not confirm the tail.\n\nVERDICT: PASSED"))


def test_unverified_regex_catches_still_unknown_whether():
    """The verdict-less fallback must also catch run 10's phrasing, where
    the report's first line named the task's own claim as unknown -- while
    a PASSED report's peripheral caveat ("Still unknown: direct
    confirmation from index.html ...", same run, legitimately green) must
    not trip it."""
    assert loop._reply_says_unverified(
        "Still unknown: whether all five visual changes are actually "
        "visible in the running browser.")
    assert loop._reply_says_unverified(
        "Still unknown from the live capture: whether the frog visibly "
        "scales, because the only evidence is a Game Over screen.")
    assert not loop._reply_says_unverified(
        "Still unknown: direct confirmation from index.html that the page "
        "includes main.js, because that file was not gathered.")


def test_investigation_answer_adds_the_verdict_brief_for_verify_titles(
        tmp_path):
    """Only verification-titled investigations get the verdict brief -- and
    the decision must key off the clean TITLE, not the assembled prompt:
    the goal block quotes the spec, whose verify words would otherwise put
    PASSED/FAILED on every locate/reproduce task."""
    seen = []

    class StubAgent:
        steps_log = ["1. browser python main.py [wait 3]  (Look.)"]

        def gather(self, task, **kw):
            return "EVIDENCE"

        def answer(self, task, gathered, system=None, **kw):
            seen.append(system)
            return "The claim held.\n\nVERDICT: PASSED"

    idx = types.SimpleNamespace(root=tmp_path)
    loop.run_investigation_task(
        tmp_path, _cfg(), StubAgent(), idx,
        "ORIGINAL USER GOAL (background):\nKeep the sprite work verifiable.\n"
        "\nCURRENT TASK\nVerify the tail tapers to a point")
    assert "VERDICT: PASSED" in seen[0]
    assert "very last line" in seen[0]

    loop.run_investigation_task(
        tmp_path, _cfg(), StubAgent(), idx,
        "ORIGINAL USER GOAL (background):\nYou must verify nothing "
        "regressed.\n\nCURRENT TASK\nLocate the snake rendering code")
    assert "VERDICT" not in seen[1]


def test_clip_report_keeps_head_and_tail():
    """Caveats can sit ANYWHERE in a report: the run that motivated this
    lost a mid-report "nothing shows this file is loaded by the running
    app" warning to a 3,000-char clip. Real reports run 4-7k chars and now
    pass through whole; beyond the budget, clipping keeps head AND tail."""
    real = ("EVIDENCE. " + "x" * 5000
            + " Still unknown: whether the app loads this file at all.")
    assert loop._clip_report(real) == real
    huge = ("EVIDENCE. " + "x" * 9000
            + " Next check: inspect main.js to confirm which snake class "
              "is instantiated.")
    clipped = loop._clip_report(huge)
    assert clipped.startswith("EVIDENCE.")
    assert "confirm which snake class" in clipped
    assert "clipped" in clipped
    assert len(clipped) < len(huge)
    # An explicit small budget clips head AND tail, never head only.
    small = loop._clip_report(huge, budget=1000)
    assert small.startswith("EVIDENCE.") and "instantiated." in small
    assert loop._clip_report("short report") == "short report"


def test_evidence_hint_only_for_live_behavior_tasks():
    hint = loop._evidence_hint(
        "Run the game and verify the snake renders with a diamond head")
    assert "lookup" in hint.lower() and "browser" in hint.lower()
    assert loop._evidence_hint("Double the snake's body height") == ""


def test_live_evidence_gate_is_objective():
    """The run that motivated this: "Locate ... and verify" never drove the
    app, admitted it only in prose the phrase list had never seen, and was
    ticked done. The steps log is the signal that cannot be reworded."""
    assert loop._needs_live_evidence(
        "Run the game and verify the snake is twice as tall")
    assert loop._needs_live_evidence(
        "Reproduce the rendering issue by running the game and observing")
    assert not loop._needs_live_evidence("Locate the snake code")
    assert not loop._needs_live_evidence("Double the snake body height")
    # Step entries anchor on the ACTION token: a read step whose reason
    # mentions running the game is not live evidence.
    assert loop._used_live_evidence(
        ["2. browser python main.py [wait 3]  (Launch it.)"])
    assert loop._used_live_evidence(["3. run pytest -q  (Tests.)"])
    assert not loop._used_live_evidence(
        ["1. read main.py L1-10  (Need to know how to run the game.)"])


def test_plan_system_requires_live_checkpoints():
    """The run that motivated this: the planner dropped BOTH the reproduce
    and the final verify tasks for a "not rendering in the game" spec, so
    nothing in the plan ever looked at the live app."""
    assert "FIRST task must run the program" in loop.PLAN_SYSTEM
    assert "LAST task must run the program" in loop.PLAN_SYSTEM
    assert "dead or duplicated" in loop.PLAN_SYSTEM


def test_investigate_system_demands_live_path_confirmation():
    assert "actually loads" in loop.INVESTIGATE_SYSTEM
    assert "dead or duplicated" in loop.INVESTIGATE_SYSTEM


def test_investigate_system_separates_loaded_from_used():
    """Graveyard-run task 1 answered the load-based title literally ("all
    22 files are loaded by index.html; none are dead code") even though it
    knew loaded != used, and deferred the classification to a live check
    that never came -- contradicting `explain`, which applied the usage
    criterion and named all 23 files. The brief must state the criterion
    outright: a load reference is itself part of what dead-code removal
    cleans up, and when the app cannot run, reading the bootstrap module
    for calls into the file IS valid usage evidence."""
    assert "being loaded is not being used" in loop.INVESTIGATE_SYSTEM
    assert "bootstrap module" in loop.INVESTIGATE_SYSTEM


def test_plan_system_requires_constraints_in_titles():
    """Run 6's planner dropped "keeping the doubled height" from the fix
    task's title, and the worker -- whose attention goes to the title --
    deleted rows instead of replacing them, shrinking the sprite."""
    assert "Carry every concrete constraint" in loop.PLAN_SYSTEM


def test_plan_system_wants_a_checkpoint_per_visible_change():
    """A five-item sprite spec verified by one big final check would make a
    failure unattributable and unrepairable selectively -- the planner must
    interleave a live checkpoint after each independent visible change."""
    assert "after EACH one" in loop.PLAN_SYSTEM


def test_gather_base_requires_locate_diligence():
    """Run 7's locate task concluded "done (Found snake frame and draw
    code)" after reading ONLY the dead enemies.js -- the first name match.
    Run 6's equivalent was accidentally thorough because the forced-read
    bug dragged it through main.js too; removing that crutch has to be
    paired with telling the gather loop what diligence now means."""
    from pedacito.agent import GATHER_BASE
    assert "hypothesis, not an answer" in GATHER_BASE
    assert "entry point" in GATHER_BASE


def test_gather_base_warns_browser_waits_add():
    """Runs 7 and 8 both captured at 2x the requested wait: the model put
    `wait N` in the pattern AND N in `start`, and browser.py executes both
    ("Time: 20" after a "wait 10" step muddled two runs' timing analysis).
    The old TOOL_HELP example taught the duplication; the prompt must say
    the two waits add together."""
    from pedacito.agent import GATHER_BASE
    assert "add together" in GATHER_BASE


def test_gather_base_documents_the_eval_action():
    """State-dependent visual checks (does the frog SCALE when grown) are
    undrivable by clicks alone: run 9's verify had no way to force the
    grown state and could only recommend it in prose. The prompt must teach
    the `eval` escape hatch or small models will never use it -- and run 10
    showed the teaching must include the consumes-rest grammar, or the
    model's multi-statement JS gets shredded by the `;` action splitter."""
    from pedacito.agent import GATHER_BASE
    assert "eval <js>" in GATHER_BASE
    assert "put it LAST" in GATHER_BASE


def test_gather_base_teaches_eval_as_measurement():
    """Run 12's retired visual checks died the same way twice: eval forced
    the state correctly (scale 1 -> 3, snake frames 0/1), but the VERDICT
    needed "visibly larger"/"frames differ" from the vision descriptions,
    which came back hedged or truncated -- nothing quotable, so the checks
    could not pass and retired after no-edit repairs. The frog-growth
    report even prescribed the fix itself: "explicitly compare the frog's
    bounding box or a numeric rendered size, rather than relying only on
    total frame pixel difference". eval's RETURNED value is the
    defensible evidence; the bullet must teach measuring numbers in the
    page (pixel counts, bounding boxes, frame diffs)."""
    from pedacito.agent import GATHER_BASE
    assert "measuring instrument" in GATHER_BASE
    assert "getImageData" in GATHER_BASE
    assert "looks larger" in GATHER_BASE


def test_gather_base_warns_about_capture_drift():
    """Run 12's combined-frame verify forced a controlled scene (enemies
    cleared, one snake and one turtle placed at fixed positions) but its
    report found "by the time of the captures, Time: 4 shows that normal
    spawning had resumed and extra scorpion sprites were present" -- the
    page keeps simulating between the eval and the screenshot, so forced
    states must be built to survive the gap and re-asserted right before
    each capture."""
    from pedacito.agent import GATHER_BASE
    assert "keeps running between" in GATHER_BASE
    assert "re-assert" in GATHER_BASE


def test_gather_base_separates_canvas_pixels_from_coordinates():
    """The getImageData measurement above is only sound if the model knows
    device pixels != game coordinates: pitfall draws enemies as 6x6 pixel
    blocks and the frog as 2x2 cells under a live ctx.scale(d.scale)
    growth transform, so sampling at a logical position without deriving
    the mapping -- and without a generous margin -- misses the sprite."""
    from pedacito.agent import GATHER_BASE
    assert "not game coordinates" in GATHER_BASE
    assert "ctx.scale" in GATHER_BASE


def test_run_investigation_task_retries_an_empty_answer(tmp_path):
    """Run 8's task 6 had an empty answer phase and the loop fell straight
    to the steps-log synthesis. The retry absorbs transient empty
    completions from local models before that fallback is reached -- and
    when the retry is also empty, the fallback still records what happened
    (the gate fails verify titles on it)."""
    cfg = _cfg()
    calls = []

    class FakeAgent:
        steps_log = ["1. browser python main.py [wait 3]  (Look.)"]

        def gather(self, task, **kw):
            return "gathered source"

        def answer(self, task, gathered, **kw):
            calls.append(1)
            return "" if len(calls) == 1 else (
                "A real report with enough prose to pass the word heuristic.")

    tr = loop.run_investigation_task(
        tmp_path, cfg, FakeAgent(),
        types.SimpleNamespace(root=tmp_path), "Reproduce the bug")
    assert len(calls) == 2
    assert tr.reply.startswith("A real report")

    class AlwaysEmpty(FakeAgent):
        def answer(self, task, gathered, **kw):
            calls.append(1)
            return ""

    tr = loop.run_investigation_task(
        tmp_path, cfg, AlwaysEmpty(),
        types.SimpleNamespace(root=tmp_path), "Reproduce the bug")
    assert len(calls) == 4
    assert tr.reply.startswith("No prose report was produced")
    assert "1. browser python main.py" in tr.reply
    # Fix ZD visibility: run 13f's log.txt showed NOTHING between the
    # context-size print and the run-stopping error -- no way to tell a
    # model flake from a harness bug after the fact. Both the retry and
    # the stand-in synthesis must land in the event log.
    from pedacito import workspace
    log = workspace.run_log_path(tmp_path).read_text()
    assert "retrying once" in log
    assert "synthesized a steps-log stand-in" in log


def test_tool_call_only_answer_is_detected():
    assert loop._looks_like_tool_calls_only(
        '<tool_call>{"name": "run", "arguments": {}}</tool_call>')
    assert not loop._looks_like_tool_calls_only(
        "Likely cause: the surface is never cleared. "
        "The run produced no output, so the loop is spun before drawing.")


def test_prose_collapse_flags_phrase_loops_but_not_parallel_prose():
    """The graveyard run streamed a task answer cycling through quoted
    fragments -- a sampling collapse no existing guard covers (llm.py's
    stream guard counts tool-call syntax opens, not prose). The signal is
    vocabulary novelty in the tail: a looping window reuses a tiny word
    set (~0.13 measured), while even a 23-row file table keeps unique
    paths and symbols on every line (~0.31). Paths and code spans are
    deliberately NOT stripped -- their uniqueness is what separates a
    table from a loop (a stripped 4-gram design false-positived on file
    tables). Short replies are below the signal floor and exempt."""
    looped = ("the file is loaded by the page but it is not used by the "
              "game runtime after moving the dead code files ") * 6
    assert loop._prose_collapse(looped)
    table = "\n".join(
        f"| static/js/mod{i:02d}.js | defines sprite{i:02d} | unused |"
        for i in range(23))
    assert not loop._prose_collapse(table)
    assert not loop._prose_collapse("done. " * 30)      # < 60 words


def test_run_investigation_task_regenerates_a_collapsed_answer(tmp_path):
    """A collapsed answer is regenerated exactly once: catching it here
    keeps looping garbage out of the context notes every later task
    reads. When the regeneration is ALSO collapsed the retry is kept as
    the report -- bounded, never an infinite regeneration loop."""
    cfg = _cfg()
    calls = []
    COLLAPSED = ("the file is loaded by the page but it is not used by "
                 "the game runtime after moving the dead code files ") * 6
    CLEAN = ("The game launches and the snake renders with the new "
             "colors; the browser screenshot at t=2s shows the diamond "
             "head.\n\nVERDICT: PASSED")

    class FakeAgent:
        steps_log = ["1. browser http://localhost:8000 [wait 2]  (Look.)"]

        def gather(self, task, **kw):
            return "gathered source"

        def answer(self, task, gathered, **kw):
            calls.append(1)
            return COLLAPSED if len(calls) == 1 else CLEAN

    tr = loop.run_investigation_task(
        tmp_path, cfg, FakeAgent(),
        types.SimpleNamespace(root=tmp_path),
        "Run the game and verify the snake renders")
    assert len(calls) == 2
    assert tr.reply == CLEAN
    from pedacito import workspace
    assert "collapsed" in workspace.run_log_path(tmp_path).read_text()

    class AlwaysCollapsed(FakeAgent):
        def answer(self, task, gathered, **kw):
            calls.append(1)
            return COLLAPSED

    tr = loop.run_investigation_task(
        tmp_path, cfg, AlwaysCollapsed(),
        types.SimpleNamespace(root=tmp_path),
        "Run the game and verify the snake renders")
    assert len(calls) == 4          # one regeneration, then the retry stays
    assert tr.reply == COLLAPSED


def test_run_loop_marks_investigation_task_done_without_edits(tmp_path,
                                                             monkeypatch):
    """The reported failure: a reproduce/diagnose task is a success even when
    it changes no code. It must be ticked and the next fix task must still
    run, instead of stopping the whole loop on 'no applied edits'."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "fix the black screen",
                    ["Reproduce the black screen",
                     "Fix the rendering bug"])
    idx = types.SimpleNamespace(root=tmp_path)
    calls = []

    def fake_investigate(root, cfg, agent, idx_, task,
                         answer_system=None):
        calls.append(("investigate", task))
        return loop.TaskResult(
            [], "Reproduced: the window opens black and the process produces "
            "no output.", True, [], idx_)

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None):
        calls.append(("edit", task))
        return _result(idx_)

    monkeypatch.setattr(loop, "run_investigation_task", fake_investigate)
    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
    assert code == 0
    assert [kind for kind, _ in calls] == ["investigate", "edit"]
    fix_task = calls[1][1]
    assert "CONTEXT FROM PRIOR INVESTIGATION" in fix_task
    assert "window opens black" in fix_task
    assert "- [x] Reproduce the black screen" in plan.read_text()
    assert "- [x] Fix the rendering bug" in plan.read_text()


def test_run_loop_fails_verify_task_that_gathered_no_evidence(tmp_path,
                                                              monkeypatch):
    """The reported failure: "Run the game and verify ..." could not launch
    the browser, said so in its report, and was ticked done anyway. A
    checkpoint that admits it gathered no evidence must fail the run -- and
    since bug T (graveyard run), it first gets ONE procedural re-run: a
    failure with zero run/browser/screenshot steps describes the check run,
    not the product, so the loop re-runs it with the toolset spelled out
    (retry note in the prompt) instead of failing or remediating. When the
    re-run stays source-only too, the check retires as unverified."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "fix the snake",
                    ["Fix the snake colors",
                     "Run the game and verify the new colors render"])
    idx = types.SimpleNamespace(root=tmp_path)
    calls = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None):
        calls.append(("edit", task))
        return _result(idx_)

    def fake_investigate(root, cfg, agent, idx_, task,
                         answer_system=None):
        calls.append(("investigate", task))
        return loop.TaskResult(
            [], "I can't launch the browser in this final phase, so I "
                "can't produce a live visual verification.", True, [], idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    monkeypatch.setattr(loop, "run_investigation_task", fake_investigate)
    # repair_rounds=0 isolates the gate: with a budget, a failed check now
    # gets a remediation task first (pinned in the remediation tests below).
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         repair_rounds=0)
    assert code == 1
    assert [kind for kind, _ in calls] == \
        ["edit", "investigate", "investigate"]
    # The verify task is told up front that tools live in the lookup phase.
    assert "EVIDENCE" in calls[1][1]
    # The procedural re-run carries the corrective note; the first try
    # does not, and no remediation task was ever inserted.
    assert "RETRY" in calls[2][1]
    assert "RETRY" not in calls[1][1]
    text = plan.read_text()
    assert "- [x] Fix the snake colors" in text
    assert "- [x] Run the game" not in text
    assert "- [!]" in text                  # retired as unverified
    assert "Repair the defect" not in text  # never remediated


def test_run_loop_procedural_verify_retry_can_pass(tmp_path, monkeypatch):
    """The procedural re-run is a real second chance, not a delayed
    failure: an attempt that drives the app and reports VERDICT: PASSED
    ticks the check done with no repair ever inserted. Retiring as
    unverified only happens when BOTH attempts stay source-only."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "fix the snake",
                    ["Fix the snake colors",
                     "Run the game and verify the new colors render"])
    idx = types.SimpleNamespace(root=tmp_path)
    calls = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None):
        calls.append(("edit", task))
        return _result(idx_)

    def fake_investigate(root, cfg, agent, idx_, task,
                         answer_system=None):
        calls.append(("investigate", task))
        if len(calls) == 2:
            # First verify attempt: the field confabulation -- read-only
            # steps plus a report declaring live verification impossible.
            return loop.TaskResult(
                [], "No live verification possible without tools.\n\n"
                    "VERDICT: FAILED", True, [], idx_, None,
                ["1. read main.js L1-40  (Inspect the draw call.)"])
        return loop.TaskResult(
            [], "The page loads, the canvas renders, and the snake shows "
                "the new colors.\n\nVERDICT: PASSED", True, [], idx_, None,
            ["1. run python -m http.server 8000  (Launch the game.)",
             "2. browser http://localhost:8000 [wait 2]  (Look.)"])

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    monkeypatch.setattr(loop, "run_investigation_task", fake_investigate)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         repair_rounds=0)
    assert code == 0
    assert [kind for kind, _ in calls] == \
        ["edit", "investigate", "investigate"]
    # The re-run carries the procedural corrective note, and only it does.
    assert "NEVER RAN THE CHECK" in calls[2][1]
    assert "RETRY" not in calls[1][1]
    text = plan.read_text()
    assert "- [x] Run the game" in text     # the check passed on attempt 2
    assert "- [!]" not in text
    assert "Repair the defect" not in text  # never remediated


def test_run_loop_edit_task_needs_evidence_when_title_says_verify(tmp_path,
                                                                  monkeypatch):
    """An edit-path task whose title asks to verify is not done just because
    an edit applied while the reply admits the verification never ran."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal",
                    ["Change the head colors and verify in the browser"])
    idx = types.SimpleNamespace(root=tmp_path)

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None):
        return loop.TaskResult(
            [_ok()], "Applied the palette edit. I can't launch the browser "
                     "in this final phase, so I can't produce a live visual "
                     "verification.", True, ["a.py"], idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         repair_rounds=0)   # isolate the gate; remediation
                                            # has its own tests
    assert code == 1
    assert "- [x]" not in plan.read_text()


def test_run_loop_live_verify_needs_a_live_step(tmp_path, monkeypatch):
    """Confident prose over read-only lookups still fails the objective
    gate: "Run the game and verify ..." with no run/browser/screenshot step
    observed nothing, whatever the report claims."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal",
                    ["Run the game and verify the snake is taller"])
    idx = types.SimpleNamespace(root=tmp_path)

    def fake_investigate(root, cfg, agent, idx_, task,
                         answer_system=None):
        return loop.TaskResult(
            [], "Verified: the snake renders 180px tall with a diamond head.",
            True, [], idx_,
            steps=["1. read static/js/main.js L195-230  (Check frames.)"])

    monkeypatch.setattr(loop, "run_investigation_task", fake_investigate)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         repair_rounds=0)   # isolate the gate; remediation
                                            # has its own tests
    assert code == 1
    assert "- [x]" not in plan.read_text()


def test_run_loop_live_verify_passes_with_browser_step(tmp_path, monkeypatch):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal",
                    ["Run the game and verify the snake is taller"])
    idx = types.SimpleNamespace(root=tmp_path)

    def fake_investigate(root, cfg, agent, idx_, task,
                         answer_system=None):
        return loop.TaskResult(
            [], "Verified: the snake renders 180px tall with a diamond head.",
            True, [], idx_,
            steps=["1. browser python main.py [wait 3]  (Launch the game.)",
                   "2. read static/js/main.js L195-230  (Confirm frames.)"])

    monkeypatch.setattr(loop, "run_investigation_task", fake_investigate)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
    assert code == 0
    assert "- [x] Run the game" in plan.read_text()


def test_insert_task_before_keeps_the_failed_check_unchecked(tmp_path):
    """The repair must run BEFORE the retried check, and the check must
    stay unchecked so the top-down loop picks the repair first."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal", ["Fix the sprite rows",
                                   "Run the game and verify one head"])
    loop.mark_done(plan, "Fix the sprite rows")
    assert loop.insert_task_before(plan, "Run the game and verify one head",
                                   loop._REMEDIATION_TITLE)
    lines = [ln for ln in plan.read_text().splitlines()
             if ln.startswith("- [")]
    assert lines == ["- [x] Fix the sprite rows",
                     f"- [ ] {loop._REMEDIATION_TITLE}",
                     "- [ ] Run the game and verify one head"]
    assert not loop.insert_task_before(plan, "no such task", "Whatever")


def test_remediation_title_routes_to_edit_branch_without_gate():
    """The inserted repair task must not itself be an investigation task,
    nor a verification checkpoint: a repair cannot always drive the app,
    and the retried check right after it does the observing. A future
    word-list change that breaks either property would quietly deadlock
    the remediation path (repair fails its own gate -> more repairs)."""
    t = loop._REMEDIATION_TITLE
    assert not loop._is_investigation_task(t)
    assert not loop._is_verification_task(t)
    # Run 10's Goodhart: the field remediation "fixed" a failed survival
    # check by moving enemy spawning 3s -> 12s, violating the goal's frozen
    # spawn timing. The title must make frozen behavior off-limits and a
    # conflict a reportable outcome -- while staying scan-clean (the two
    # routing properties pinned above).
    assert "freezes" in t and "do not edit" in t


def test_run_loop_inserts_remediation_after_failed_verify(tmp_path, monkeypatch):
    """Bug H: a failed verification used to end the run with its diagnosis
    (often an exact repair recipe) left in the log and no task owning it.
    Now the report becomes a FAILED LIVE TEST context note, a repair task
    is inserted BEFORE the still-unchecked check, and -- within the same
    run -- the repair lands and the check retries."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal",
                    ["Run the game and verify the snake has one head"])
    idx = types.SimpleNamespace(root=tmp_path)
    verifies = []
    repairs = []

    def fake_investigate(root, cfg, agent, idx_, task,
                         answer_system=None):
        verifies.append(task)
        if len(verifies) == 1:
            return loop.TaskResult(
                [], "Verification failed in the captured run: each snake "
                    "still shows a second head block halfway down its body "
                    "(main.js L213/L245).", True, [], idx_,
                steps=["1. browser python main.py [wait 3]  (Look.)"])
        return loop.TaskResult(
            [], "Verified in the browser: every snake now shows a single "
                "head at the top of its body.", True, [], idx_,
            steps=["1. browser python main.py [wait 3]  (Look again.)"])

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None):
        repairs.append(task)
        return loop.TaskResult([_ok()], "Replaced the duplicated rows.",
                               True, ["a.py"], idx_,
                               steps=["1. read a.py L1-5  (Rows.)"])

    monkeypatch.setattr(loop, "run_investigation_task", fake_investigate)
    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
    assert code == 0
    assert len(verifies) == 2          # the check retried within the run
    assert len(repairs) == 1
    assert repairs[0].startswith(loop._REMEDIATION_TITLE)
    # the diagnosis itself must reach the repair worker: the failed check's
    # report rides along in the assembled prompt's context notes
    assert "FAILED LIVE TEST" in repairs[0]
    assert "Verification failed in the captured run" in repairs[0]
    # Fix M: the failed report's suggested fix must reach the repair worker
    # flagged as a hypothesis, not a mandate -- run 10's repair followed the
    # report's "delay the first spawn" recipe straight into a violation of
    # the goal's frozen spawn timing.
    assert "only a hypothesis" in repairs[0]
    assert "freezes" in repairs[0]
    text = plan.read_text()
    assert text.count("- [x]") == 2    # repair + retried check
    assert "- [ ]" not in text


def test_run_loop_remediation_rounds_are_bounded(tmp_path, monkeypatch):
    """Self-healing must not become self-deception: once repair_rounds
    repairs were spent on one check, the next failure ends the run as
    before, with the check's box left unchecked for a resume."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal",
                    ["Run the game and verify the snake has one head"])
    idx = types.SimpleNamespace(root=tmp_path)
    verifies = []

    def fake_investigate(root, cfg, agent, idx_, task,
                         answer_system=None):
        verifies.append(task)
        return loop.TaskResult(
            [], "Verification failed: the snake still shows two heads.",
            True, [], idx_,
            steps=["1. browser python main.py [wait 3]  (Look.)"])

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None):
        return loop.TaskResult([_ok()], "Tried a fix.", True, ["a.py"],
                               idx_, steps=["1. read a.py L1-5  (Rows.)"])

    monkeypatch.setattr(loop, "run_investigation_task", fake_investigate)
    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         repair_rounds=2)
    assert code == 1
    assert len(verifies) == 3          # initial + two post-repair retries
    text = plan.read_text()
    assert text.count("- [x] Repair") == 2
    assert "- [ ] Run the game and verify" in text


def test_run_loop_retires_a_check_whose_repair_refused_to_edit(tmp_path,
                                                               monkeypatch):
    """Bug N, from run 11: the planner titled task 1 "observe the spawn
    timing to CONFIRM the 12-second delay" -- a defect-confirmation
    checkpoint. The inserted repair did the goal's work instead (restored
    the 3-second schedule), making the check permanently unpassable; the
    next repair refused to re-break the code ("would conflict with the
    goal"), the check failed again, and the run STOPPED with five
    downstream tasks never run -- and a resume replays the deadlock
    forever. A no-edit repair must retire the check as failed and let the
    rest of the plan run."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal",
                    ["Run the game and verify the delay is 12 seconds",
                     "Fix the sprite rows"])
    idx = types.SimpleNamespace(root=tmp_path)
    verifies = []
    repairs = []
    fixes = []

    def fake_investigate(root, cfg, agent, idx_, task,
                         answer_system=None):
        verifies.append(task)
        return loop.TaskResult(
            [], "The delay is gone; the goal's restore removed it.\n\n"
                "VERDICT: FAILED: the 12-second delay no longer exists",
            True, [], idx_,
            steps=["1. browser python main.py [wait 3]  (Look.)"])

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None):
        if task.startswith("Repair the defect"):
            repairs.append(task)
            return loop.TaskResult(
                [], "No code edit is needed: the restored schedule is "
                    "already in place, and reintroducing the delay would "
                    "conflict with the goal.", True, [], idx_, steps=[])
        fixes.append(task)
        return loop.TaskResult([_ok()], "Fixed the rows.", True, ["a.py"],
                               idx_, steps=["1. read a.py L1-5  (Rows.)"])

    monkeypatch.setattr(loop, "run_investigation_task", fake_investigate)
    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
    assert code == 1                     # the retired check stays a failure
    assert len(verifies) == 2            # original + one retry, no more
    assert len(repairs) == 1             # one repair, then retired
    assert len(fixes) == 1               # the DOWNSTREAM task still ran
    text = plan.read_text()
    assert "- [!] Run the game and verify the delay" in text
    assert "- [x] Fix the sprite rows" in text


def test_plan_system_forbids_defect_confirming_checkpoints():
    """Run 11's trap, planner-side: task 1 was "observe the current enemy
    spawn timing to CONFIRM the 12-second delay and -12 offset" -- a
    defect-confirmation checkpoint that the plan's own restore task makes
    permanently unpassable. Pre-change observations must be Reproduce/
    Investigate titles (ungated), and verify checkpoints must state the
    AFTER-change expectation."""
    assert "can never pass again" in loop.PLAN_SYSTEM
    assert "AFTER-change" in loop.PLAN_SYSTEM


def test_run_loop_empty_report_gets_no_remediation(tmp_path, monkeypatch):
    """A synthesized steps-log stand-in carries no diagnosis -- a repair
    task would be guessing at what to fix. Empty-report failures fail
    straight away; a resume retries the whole check, and the answer-phase
    retry in run_investigation_task may well produce a real report."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal",
                    ["Run the game and verify the snake has one head"])
    idx = types.SimpleNamespace(root=tmp_path)

    def fake_investigate(root, cfg, agent, idx_, task,
                         answer_system=None):
        return loop.TaskResult(
            [], "No prose report was produced; the lookup phase recorded:\n\n"
                "3. browser python main.py [wait 10]  (Look.)",
            True, [], idx_,
            steps=["3. browser python main.py [wait 10]  (Look.)"])

    monkeypatch.setattr(loop, "run_investigation_task", fake_investigate)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
    assert code == 1
    assert "Repair the defect" not in plan.read_text()


def test_run_loop_fails_verify_task_reporting_a_failed_verification(
        tmp_path, monkeypatch):
    """Run 7's terminal false green: the browser HAD been driven, so the
    objective gate passed, but the report's first line said "Verification
    failed in the captured run" -- and the loop ticked it done, ending
    "7 done, 0 failed" on a game that still missed its spec. A driven
    browser is not a pass when the report says the claim failed."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal",
                    ["Run the game and verify the snake has one head"])
    idx = types.SimpleNamespace(root=tmp_path)

    def fake_investigate(root, cfg, agent, idx_, task,
                         answer_system=None):
        return loop.TaskResult(
            [], "Verification failed in the captured run: each snake still "
                "shows a second head block halfway down its body.",
            True, [], idx_,
            steps=["1. browser python main.py [press Enter; wait 3]  (Look.)"])

    monkeypatch.setattr(loop, "run_investigation_task", fake_investigate)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         repair_rounds=0)   # isolate the gate; remediation
                                            # has its own tests
    assert code == 1
    assert "- [x]" not in plan.read_text()


def test_run_loop_fails_verify_task_with_no_prose_report(tmp_path, monkeypatch):
    """Run 8's terminal green: task 6's answer phase came back empty, the
    loop synthesized a steps-log fallback whose embedded `done` reason
    claimed success, and the final verify checkpoint ticked done with no
    report at all -- zero model-authored evidence, marked passed. The
    synthesized stand-in must fail verify titles."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal",
                    ["Run the game and verify the snake has one head"])
    idx = types.SimpleNamespace(root=tmp_path)

    def fake_investigate(root, cfg, agent, idx_, task,
                         answer_system=None):
        return loop.TaskResult(
            [], "No prose report was produced; the lookup phase recorded:\n\n"
                "3. browser python main.py [press Enter; wait 10]  (Look.)",
            True, [], idx_,
            steps=["3. browser python main.py [press Enter; wait 10]  (Look.)"])

    monkeypatch.setattr(loop, "run_investigation_task", fake_investigate)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         repair_rounds=0)   # isolate the gate; remediation
                                            # has its own tests
    assert code == 1
    assert "- [x]" not in plan.read_text()


def test_run_loop_edit_verify_task_needs_live_step(tmp_path, monkeypatch):
    """An edit-path task whose title promises browser verification fails on
    the objective gate too: edits applied, but the session never drove the
    app, so nothing observed the claimed effect."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal",
                    ["Change the head colors and verify in the browser"])
    idx = types.SimpleNamespace(root=tmp_path)

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None):
        return loop.TaskResult([_ok()], "Applied the new palette.", True,
                               ["a.py"], idx_,
                               steps=["1. read a.py L1-5  (Palette.)"])

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         repair_rounds=0)   # isolate the gate; remediation
                                            # has its own tests
    assert code == 1
    assert "- [x]" not in plan.read_text()


def test_run_loop_carries_goal_into_task_prompt(tmp_path, monkeypatch):
    """Run 6's task 3 deleted rows instead of replacing them: the spec's
    "keeping the doubled height" constraint existed only in the goal, which
    no task prompt carried, and the planner's title had dropped it. Every
    task prompt must include the goal as binding background -- while forced
    reads stay mined from the clean title only."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal", ["Fix the sprite rows"])
    idx = types.SimpleNamespace(root=tmp_path)
    seen = {}

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None):
        seen["task"] = task
        seen["targets"] = getattr(agent, "task_targets", None)
        return loop.TaskResult([_ok()], "reply", True, ["a.py"], idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         goal="Remove the mid-body head, KEEPING the "
                              "doubled 30-row height.")
    assert code == 0
    assert "ORIGINAL USER GOAL" in seen["task"]
    assert "KEEPING the doubled 30-row height" in seen["task"]
    assert "CURRENT TASK\nFix the sprite rows" in seen["task"]
    assert seen["targets"] == "Fix the sprite rows"


def test_run_loop_flags_stale_notes_after_edits(tmp_path, monkeypatch):
    """Run 6's verify task reconciled task 2's pre-edit report ("duplicated
    block at L212-216") against post-edit source and half concluded the
    browser served a cache. After an edit applies, the notes it feeds must
    announce which files changed and that older descriptions are stale."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal", ["Fix the duplicated head block",
                                   "Adjust the palette"])
    idx = types.SimpleNamespace(root=tmp_path)
    seen = {}

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None):
        seen["task"] = task
        return loop.TaskResult([_ok()], "Removed the middle rows.", True,
                               ["static/js/main.js"], idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
    assert code == 0
    # The second task's prompt carries the staleness marker for the first.
    assert "FILES CHANGED:\n- static/js/main.js" in seen["task"]
    assert "stale" in seen["task"]


def test_gather_mines_forced_reads_from_targets_only(tmp_path):
    """Run 6: prior reports' "do NOT change snake_spawner.py yet" prose was
    mined as files the task "will change", forcing ~14 dead-code reads per
    task and leaving verify tasks no budget to drive the browser. With
    task_targets set (the clean title), descriptive mentions are not
    must-reads; without it, behavior is unchanged for ask/chat flows."""
    from pedacito import index as index_mod
    from pedacito.agent import Agent

    (tmp_path / "snake_spawner.py").write_text("SNAKE = 1\n",
                                               encoding="utf-8")
    cfg = Config()
    cfg.log_sessions = False
    idx = index_mod.build([str(tmp_path)], cfg, client=None,
                          summarise=False, root=tmp_path)

    class DoneClient:
        stream_truncated = False
        step_truncated = False
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            return {"action": "done", "file": "", "pattern": "",
                    "symbol": "", "start": 0, "end": 0, "content": "",
                    "reasoning": "done"}
        def chat(self, messages, max_tokens, schema=None, stream=False, **kw):
            return ""

    assembled = ("Fix the sprite rows\n\nCONTEXT FROM PRIOR INVESTIGATION\n\n"
                 "Do NOT change snake_spawner.py yet; it is not loaded.")
    plain = Agent(idx, DoneClient(), cfg)
    plain.gather(assembled, verbose=False)
    assert plain._unread_files, "legacy behavior: prose mentions force reads"

    targeted = Agent(idx, DoneClient(), cfg)
    targeted.task_targets = "Fix the sprite rows"
    targeted.gather(assembled, verbose=False)
    assert targeted._unread_files == []


def test_gather_never_dispatches_a_step_whose_payload_was_cut(tmp_path, monkeypatch):
    """Bug W (run 13): all five pixel checks retired because the browser
    ran eval payloads that arrived cut off. Two deterministic cut classes:
    the token-limit salvage leaves client.truncated_field naming the half
    field, and grammar-constrained sampling force-closes a string at
    exactly its schema maxLength with NO truncation flag at all. Either
    way, dispatching the step runs half a program whose SyntaxError looks
    like a product failure. gather() must discard such a step WITHOUT
    logging a live browser step (that would falsely satisfy the
    live-evidence gate), tell the model to minify or split, and stop after
    _MAX_PAYLOAD_CUTS."""
    from pedacito import agent as agent_mod
    from pedacito import index as index_mod
    from pedacito.agent import Agent
    from pedacito.tools import schema_for

    (tmp_path / "main.js").write_text("var a = 1;\n", encoding="utf-8")
    cfg = Config()
    cfg.log_sessions = False
    cfg.allow_browser = True
    idx = index_mod.build([str(tmp_path)], cfg, client=None,
                          summarise=False, root=tmp_path)
    dispatched = []
    monkeypatch.setattr(agent_mod, "dispatch",
                        lambda index, cfg_, step, client=None:
                        dispatched.append(step.get("action")) or "ok")
    cap = schema_for(cfg)["properties"]["pattern"]["maxLength"]

    class Base:
        stream_truncated = False
        step_truncated = False
        truncated_field = ""
        def chat(self, messages, max_tokens, schema=None, stream=False, **kw):
            return ""

    class TokenCutClient(Base):
        """The salvage kept half a `pattern`; truncated_field names it."""
        step_truncated = True
        truncated_field = "pattern"
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            return {"action": "browser", "file": "", "start": 0, "end": 0,
                    "pattern": "eval new Promise(res => { const img = ctx.",
                    "symbol": "", "to": "", "command": "python main.py",
                    "content": "", "reasoning": "measure the frog"}

    class GuillotineClient(Base):
        """The grammar closed `pattern` at exactly its cap: valid JSON."""
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            return {"action": "browser", "file": "", "start": 0, "end": 0,
                    "pattern": "x" * cap, "symbol": "", "to": "",
                    "command": "python main.py", "content": "",
                    "reasoning": "measure the frog"}

    class ReasoningCutClient(Base):
        """A reasoning cut is benign: the step must NOT be discarded."""
        step_truncated = True
        truncated_field = "reasoning"
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            return {"action": "done", "file": "", "start": 0, "end": 0,
                    "pattern": "", "symbol": "", "to": "", "command": "",
                    "content": "", "reasoning": ""}

    for client in (TokenCutClient(), GuillotineClient()):
        dispatched.clear()
        agent = Agent(idx, client, cfg)
        agent.gather("Verify frog pixels by measurement", verbose=False)
        assert dispatched == []                 # the half payload never ran
        discarded = [s for s in agent.steps_log if "(discarded" in s]
        assert len(discarded) == agent_mod._MAX_PAYLOAD_CUTS
        assert all(not loop._LIVE_STEP.match(s) for s in agent.steps_log)

    agent = Agent(idx, ReasoningCutClient(), cfg)
    agent.gather("Verify frog pixels by measurement", verbose=False)
    assert not any("(discarded" in s for s in agent.steps_log)


def test_gather_base_warns_step_payloads_have_a_budget():
    """Run 13: the model composed the Promise+rAF+getImageData recipe
    correctly -- and the 600-char schema cap destroyed it. The cap got
    headroom and the guards catch the rest, but the prompt must also teach
    compact payloads and the shorten-or-split response to a cut report."""
    from pedacito.agent import GATHER_BASE
    assert "token budget" in GATHER_BASE
    assert "minified" in GATHER_BASE


def test_context_notes_carry_report_tail(tmp_path, monkeypatch):
    """Investigation reports put "Still unknown"/"Next check" last; the clip
    carried into later tasks must keep that tail, not just the confident
    conclusion at the top."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal",
                    ["Locate the snake rendering code",
                     "Fix the snake colors"])
    idx = types.SimpleNamespace(root=tmp_path)
    calls = []

    def fake_investigate(root, cfg, agent, idx_, task,
                         answer_system=None):
        calls.append(("investigate", task))
        report = ("Likely cause: enemies.js defines the snake. "
                  + "x" * 4000
                  + " Next check: inspect main.js to confirm which snake "
                    "class is actually instantiated.")
        return loop.TaskResult([], report, True, [], idx_)

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None):
        calls.append(("edit", task))
        return _result(idx_)

    monkeypatch.setattr(loop, "run_investigation_task", fake_investigate)
    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
    assert code == 0
    fix_task = calls[1][1]
    assert "CONTEXT FROM PRIOR INVESTIGATION" in fix_task
    assert "Likely cause: enemies.js" in fix_task
    assert "confirm which snake class" in fix_task


def test_prune_notes_pins_first_keeps_latest_two():
    """Fix K unit pin: the window keeps note[0] (the reproduce anchor) plus
    the latest two; only the middle rolls off."""
    notes = ["n0", "n1", "n2"]
    loop._prune_notes(notes)
    assert notes == ["n0", "n1", "n2"]        # window not exceeded yet
    notes.append("n3")
    loop._prune_notes(notes)
    assert notes == ["n0", "n2", "n3"]        # middle evicted, anchor kept
    notes += ["n4", "n5"]
    loop._prune_notes(notes)
    assert notes == ["n0", "n4", "n5"]


def test_run_loop_pins_the_reproduce_note_into_later_prompts(tmp_path,
                                                             monkeypatch):
    """Run 9's task 6 edited the dead player.js because task 1's reproduce
    report -- the one that named the file as unconfirmed-live -- had been
    evicted from the two-note window. The first note must ride along in
    every later prompt."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal",
                    [f"Investigate thing {i}" for i in range(1, 6)])
    idx = types.SimpleNamespace(root=tmp_path)
    prompts = []

    def fake_investigate(root, cfg, agent, idx_, task,
                         answer_system=None):
        prompts.append(task)
        n = len(prompts)
        return loop.TaskResult(
            [], f"NOTE-{n}: found evidence about thing {n}.", True, [], idx_)

    monkeypatch.setattr(loop, "run_investigation_task", fake_investigate)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
    assert code == 0
    last = prompts[4]                     # task 5's assembled prompt
    assert "NOTE-1" in last               # the pinned reproduce anchor
    assert "NOTE-3" in last and "NOTE-4" in last    # the latest two
    assert "NOTE-2" not in last           # middle notes still evict


# ------------------------------------------------------------------- planning
def test_generate_tasks_extracts_titles():
    class Stub:
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            assert schema["required"] == ["tasks"]
            return {"goal": "g", "tasks": [
                {"title": "Implement the parser"},
                {"title": "Wire the CLI"},
                {"title": ""},                      # dropped: empty title
                {"nonsense": "x"},                  # dropped: not a dict/field
            ]}
    titles = loop.generate_tasks("build a tool", Stub(), Config())
    assert titles == ["Implement the parser", "Wire the CLI"]


def test_generate_tasks_retries_once_when_empty():
    """A model handed a bug-fix goal sometimes returns zero tasks. The planner
    must ask once more (with an explicit nudge) before giving up."""
    calls = {"n": 0}

    class Stub:
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"tasks": []}
            return {"tasks": [{"title": "Reproduce the crash"},
                              {"title": "Fix the level loader"}]}

    titles = loop.generate_tasks("main.py crashes", Stub(), Config())
    assert titles == ["Reproduce the crash", "Fix the level loader"]
    assert calls["n"] == 2


def test_generate_tasks_falls_back_to_a_single_task():
    """If the model stays empty even after the retry, `auto` still makes an
    attempt instead of aborting the run outright."""
    class Stub:
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            return {"tasks": []}

    titles = loop.generate_tasks(
        "main.py crashes: FileNotFoundError in level_loader", Stub(), Config())
    assert titles == ["Address: main.py crashes: FileNotFoundError in level_loader"]


def test_generate_tasks_tolerates_plain_string_list():
    class Stub:
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            return {"tasks": ["First step", "Second step"]}
    assert loop.generate_tasks("do it", Stub(), Config()) == \
        ["First step", "Second step"]


def test_task_schema_gives_titles_room_and_one_home():
    """Bug Y (runs 13/13b): the planner's 300-char title cap guillotined
    every spec-derived title mid-sentence -- grammar-constrained sampling
    force-closes a string AT maxLength, producing valid JSON with no
    truncation flag -- and run 13b's planner, derailed by the forced
    close, ended the task array after task 1: five of six spec checks were
    never planned and `auto` exited 0 with "all tasks complete". The cap
    must hold a compressed-but-complete check, and `title` must be the
    ONLY field: the old `acceptance` sibling was silently dropped by
    _titles_from, so any criterion a model put there was lost."""
    items = loop.TASK_SCHEMA["properties"]["tasks"]["items"]
    assert items["properties"]["title"]["maxLength"] == 800
    assert set(items["properties"]) == {"title"}
    assert items["required"] == ["title"]


def test_generate_tasks_reasks_when_a_title_hits_the_cap():
    """The guard half of Bug Y: a title arriving at exactly the schema cap
    was force-closed mid-sentence, and (run 13b) the forced close can
    derail the model into ending the array after task 1. One bounded
    re-ask must demand the COMPLETE list with compressed titles; a clean
    retry replaces the cut plan, and a stubborn planner is accepted after
    exactly one retry (cut titles still work -- the goal rides along)."""
    cap = loop.TASK_SCHEMA["properties"]["tasks"]["items"]["properties"][
        "title"]["maxLength"]

    class Recovery:
        def __init__(self):
            self.prompts = []

        def chat_json(self, messages, schema, max_tokens=0, **kw):
            self.prompts.append(messages[1]["content"])
            if len(self.prompts) == 1:
                return {"tasks": [{"title": "x" * cap}]}   # run 13b's shape
            return {"tasks": [{"title": "Verify the spawn schedule"},
                              {"title": "Verify frog growth by pixels"}]}

    c = Recovery()
    titles = loop.generate_tasks("six checks", c, Config())
    assert titles == ["Verify the spawn schedule",
                      "Verify frog growth by pixels"]
    assert len(c.prompts) == 2
    assert "CUT" in c.prompts[1] and "COMPLETE" in c.prompts[1]
    assert f"{cap}-character" in c.prompts[1]
    # The single-task derailment gets the explicit "the rest was lost" line.
    assert "the rest of the list was lost" in c.prompts[1]

    class Stubborn:
        def __init__(self):
            self.n = 0

        def chat_json(self, messages, schema, max_tokens=0, **kw):
            self.n += 1
            return {"tasks": [{"title": "y" * cap}, {"title": "short one"}]}

    s = Stubborn()
    titles2 = loop.generate_tasks("spec", s, Config())
    assert s.n == 2                             # bounded: ONE re-ask
    assert titles2 == ["y" * cap, "short one"]  # accepted, run can proceed


def test_plan_system_teaches_compression_over_copying():
    """The prompt side of Bug Y: PLAN_SYSTEM told the model to "Carry
    every concrete constraint from the goal into the titles" with zero
    length guidance -- small models complied by copying spec passages
    verbatim into a capped field. The compression/split rule must exist,
    and the carry-constraints rule must stay."""
    assert "length cap" in loop.PLAN_SYSTEM.lower()
    assert "compress" in loop.PLAN_SYSTEM.lower()
    assert "one task per sub-item" in loop.PLAN_SYSTEM
    assert "Carry every concrete constraint" in loop.PLAN_SYSTEM


# ---------------------------------------------------------------- run_loop
def _ok(file="a.py"):
    return types.SimpleNamespace(ok=True, file=file)


def _result(idx, applied=True, ok=True):
    return loop.TaskResult([_ok()] if ok else [],
                           "reply", applied, ["a.py"] if ok else [], idx)


def _cfg():
    c = Config()
    c.log_sessions = False
    return c


def test_run_loop_implements_each_task_in_order(tmp_path, monkeypatch):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal", ["task A", "task B", "task C"])
    idx = types.SimpleNamespace(root=tmp_path)
    calls = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True, review=False, answer_system=None):
        calls.append(task)
        return _result(idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
    assert code == 0
    # Titles only: tasks after an applied edit also carry the staleness note.
    assert [c.split("\n\n")[0] for c in calls] == ["task A", "task B", "task C"]
    text = plan.read_text()
    assert "- [x] task A" in text and "- [x] task B" in text and "- [x] task C" in text


def test_run_loop_respects_max_tasks(tmp_path, monkeypatch):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["a", "b", "c"])
    idx = types.SimpleNamespace(root=tmp_path)
    calls = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True, review=False, answer_system=None):
        calls.append(task)
        return _result(idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True, max_tasks=2)
    assert [c.split("\n\n")[0] for c in calls] == ["a", "b"]
    assert "- [ ] c" in plan.read_text()          # never reached


def test_run_loop_respects_max_minutes_between_tasks(tmp_path, monkeypatch):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["a", "b"])
    idx = types.SimpleNamespace(root=tmp_path)
    calls = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None):
        calls.append(task)
        return _result(idx_)

    # start, first-task check, second-task check (already over 1 minute).
    clock = iter([0.0, 0.0, 61.0])
    monkeypatch.setattr(loop.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         max_minutes=1)
    assert code == 0
    assert calls == ["a"]
    text = plan.read_text()
    assert "- [x] a" in text
    assert "- [ ] b" in text              # not started after the time limit


def test_run_loop_stops_on_failed_task(tmp_path, monkeypatch):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["broken", "never reached"])
    idx = types.SimpleNamespace(root=tmp_path)

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True, review=False, answer_system=None):
        return _result(idx_, ok=False)            # no applied edits

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
    assert code == 1
    assert "- [ ] broken" in plan.read_text()     # not marked (we stopped)


def test_run_loop_keep_going_flags_and_continues(tmp_path, monkeypatch):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["broken", "good"])
    idx = types.SimpleNamespace(root=tmp_path)
    calls = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True, review=False, answer_system=None):
        calls.append(task)
        return _result(idx_, ok=(task != "broken"))

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         stop_on_fail=False)
    assert calls == ["broken", "good"]
    assert code == 1
    text = plan.read_text()
    assert "- [!] broken" in text                 # flagged, never re-picked
    assert "- [x] good" in text


def test_run_loop_treats_noop_reply_as_done(tmp_path, monkeypatch):
    """A task the model reports as already complete (no edit blocks, but an
    explicit 'no changes needed') must be marked done and the loop moved on,
    not treated as a failure that halts the whole run."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["already done", "real work"])
    idx = types.SimpleNamespace(root=tmp_path)
    calls = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True, review=False, answer_system=None):
        calls.append(task)
        if task == "already done":
            return loop.TaskResult(
                [], "No changes are needed; the file is already correct.",
                True, [], idx_)
        return _result(idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
    assert code == 0
    assert calls == ["already done", "real work"]
    text = plan.read_text()
    assert "- [x] already done" in text
    assert "- [x] real work" in text


def test_run_loop_retries_a_fabricated_noop_task_instead_of_ticking_it(
        tmp_path, monkeypatch):
    """The graveyard run's collision.py task emitted three null edits and was
    ticked DONE because _is_noop matched the synthetic 'No changes' marker.
    A fabricated no-op is a failed attempt, not a completion: retry once, and
    only then fail."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["move the dead file"])
    idx = types.SimpleNamespace(root=tmp_path)
    calls = []
    notes = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None):
        calls.append(task)
        notes.append(task)
        if len(calls) == 1:
            return loop.TaskResult(
                [], "I'll first read the file.\n\n[No changes: every edit "
                "block in the reply was a no-op (SEARCH == REPLACE) and was "
                "discarded; nothing was written.]", True, [], idx_,
                fabricated=True)
        return _result(idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
    assert code == 0
    assert calls[0] == "move the dead file"
    assert calls[1].startswith("move the dead file")
    assert "NO-OPS" in calls[1]            # the retry carries the corrective note
    assert "- [x] move the dead file" in plan.read_text()


def test_run_loop_marks_a_twice_fabricated_noop_task_failed(tmp_path,
                                                            monkeypatch):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["move the dead file"])
    idx = types.SimpleNamespace(root=tmp_path)
    calls = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None):
        calls.append(task)
        return loop.TaskResult(
            [], "I'll first read the file.\n\n[No changes: every edit block "
            "was a no-op.]", True, [], idx_, fabricated=True)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         stop_on_fail=False)
    assert code == 1
    assert len(calls) == 2                 # one retry, then a verdict
    assert "- [!] move the dead file" in plan.read_text()


def test_run_loop_reruns_an_empty_report_once_then_retires_the_check(
        tmp_path, monkeypatch):
    """Fix ZD (run 13f): the frog check's gather phase succeeded -- its
    eval resolved the full per-row profile and the done-step reasoning
    confirmed the criterion -- then the answer phase came back empty twice
    (thinking runaway), the loop synthesized the steps-log stand-in, and
    NO branch caught it: the run STOPPED, stranding three healthy checks.
    An empty report over gathered evidence is a report-writer flake, not a
    verdict: re-run once, then retire as unverified and CONTINUE -- never
    remediate (the stand-in carries no diagnosis)."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", [
        "Verify the frog sprite by measured pixels in the live game",
        "real work"])
    idx = types.SimpleNamespace(root=tmp_path)
    calls = []
    stand_in = ("No prose report was produced; the lookup phase recorded:\n"
                "\n1. browser python main.py [wait 1]  (Measure.)")

    def fake_investigate(root, cfg, agent, idx_, task,
                         answer_system=None):
        calls.append(task)
        return loop.TaskResult(
            [], stand_in, True, [], idx_, None,
            ["1. browser python main.py [wait 1]  (Measure.)"])

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None):
        calls.append(task)
        return _result(idx_)

    monkeypatch.setattr(loop, "run_investigation_task", fake_investigate)
    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
    assert code == 1                        # the retired check is a failure
    assert len(calls) == 3
    assert calls[1].split("\n\n### RETRY")[0] == calls[0]   # same check
    assert ("RETRY -- THE PREVIOUS ATTEMPT'S REPORT CAME BACK EMPTY"
            in calls[1])
    assert calls[2] == "real work"          # the plan CONTINUED
    text = plan.read_text()
    assert "- [!] Verify the frog sprite" in text
    assert "- [x] real work" in text
    from pedacito import workspace
    log = workspace.run_log_path(tmp_path).read_text()
    assert "re-running once, no remediation" in log
    assert "produced no prose on two attempts" in log


def test_declares_incomplete_separates_deferral_from_noop():
    """Graveyard-run task 4 ("Move each file...") answered "I can't safely
    emit the move blocks yet ... To continue, the first 3-5 lines of each
    file must be read ... I won't guess" with ZERO edits -- and an
    incidental "graveyard/ has already been created" (a PRIOR task's work)
    matched _is_noop, ticking the move done. A reply describing work still
    required is never a no-op. Mojibake apostrophes ("can\u00e2t") are folded
    before matching; honest caveats, completion reports, and the canonical
    no-op reply must not fire."""
    field_reply = (
        "I can't safely emit the move blocks yet. To continue, the first "
        "3-5 lines of each file must be read so the destination paths are "
        "known; I won't guess. Note: graveyard/ has already been created "
        "by the earlier task.")
    assert loop._is_noop(field_reply)              # the trap that fired
    assert loop._declares_incomplete(field_reply)  # now outranked
    assert loop._declares_incomplete(
        field_reply.replace("can't", "can\u00e2t"))
    assert not loop._declares_incomplete(
        "No changes are needed; the file is already correct.")
    assert not loop._declares_incomplete(
        "Moved all 23 files into graveyard/ and updated the manifest.")
    assert not loop._declares_incomplete(
        "The fix is in place. I have not yet run the browser check, but "
        "the code path is clear.")


def test_run_loop_incomplete_move_task_is_not_a_noop(tmp_path, monkeypatch):
    """The field replay: a build task that defers its own work with zero
    edits applied is retried once with the toolset spelled out, then
    failed loudly -- never ticked done via _is_noop's incidental "already"
    match. The retry note lands in the second attempt's prompt only, and
    with stop_on_fail=False the rest of the plan still runs."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "clean out the dead code",
                    ["Move each file identified as dead code into the "
                     "graveyard folder",
                     "Update the manifest to list the moved files"])
    idx = types.SimpleNamespace(root=tmp_path)
    calls = []
    DEFERRAL = (
        "I can't safely emit the move blocks yet. To continue, the first "
        "3-5 lines of each file must be read; I won't guess. Note: "
        "graveyard/ has already been created by the earlier task.")

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None):
        calls.append(task)
        if "Move each file" in task:
            return loop.TaskResult([], DEFERRAL, True, [], idx_)
        return _result(idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         stop_on_fail=False)
    assert code == 1
    assert len(calls) == 3                  # move retried once + manifest
    assert "Move each file" in calls[0] and "RETRY" not in calls[0]
    assert "Move each file" in calls[1] and "DID NOT DO THE WORK" in calls[1]
    assert "Update the manifest" in calls[2]
    text = plan.read_text()
    assert "- [x] Move each file" not in text   # the fake-done never ticks
    assert "- [!] Move each file" in text       # failed loudly instead
    assert "- [x] Update the manifest" in text  # the plan carries on


def test_run_loop_verification_repair_round_trip(tmp_path, monkeypatch):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["needs tests"])
    idx = types.SimpleNamespace(root=tmp_path)
    tasks = []
    verifies = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True, review=False, answer_system=None):
        tasks.append(task)
        return _result(idx_)

    def fake_verify(root, command):
        verifies.append(command)
        # Fail the first check, pass the second (after the model "fixes" it).
        return (len(verifies) >= 2), "boom" if len(verifies) < 2 else ""

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    monkeypatch.setattr(loop, "run_verify", fake_verify)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         verify="pytest -q", repair_rounds=2)
    assert code == 0
    assert len(tasks) == 2                         # original + one repair
    assert tasks[0] == "needs tests"
    assert "failed this check" in tasks[1]
    assert "- [x] needs tests" in plan.read_text()


def test_run_loop_auto_diagnostics_feeds_repair_rounds(tmp_path, monkeypatch):
    """With auto_diagnostics on and no --verify, a failing linter check is
    treated exactly like a failed verification: one repair task, then the
    check re-runs. No checker/pass must leave the task alone."""
    from pedacito import diagnostics

    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["add the parser"])
    idx = types.SimpleNamespace(root=tmp_path)
    tasks = []
    checks = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None):
        tasks.append(task)
        return _result(idx_)

    def fake_diagnostics(root, cfg, changed):
        checks.append(tuple(changed or ()))
        return (len(checks) >= 2), "E501 line too long"

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    monkeypatch.setattr(diagnostics, "run_changed", fake_diagnostics)
    cfg = _cfg()
    cfg.auto_diagnostics = True
    code = loop.run_loop(tmp_path, cfg, None, idx, plan, apply=True,
                         repair_rounds=1)
    assert code == 0
    assert len(tasks) == 2
    assert "failed this check" in tasks[1]
    assert checks and checks[0] == ("a.py",)


def test_run_loop_verification_exhausts_repair_rounds(tmp_path, monkeypatch):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["always failing"])
    idx = types.SimpleNamespace(root=tmp_path)
    tasks = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True, review=False, answer_system=None):
        tasks.append(task)
        return _result(idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    monkeypatch.setattr(loop, "run_verify", lambda root, cmd: (False, "boom"))
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         verify="pytest -q", repair_rounds=2)
    assert code == 1
    assert len(tasks) == 3                         # initial + two repairs
    assert "- [ ] always failing" in plan.read_text()   # never marked done


def test_run_loop_stops_when_no_tasks(tmp_path, monkeypatch):
    plan = tmp_path / "PLAN.md"
    plan.write_text("# Plan\n\n- [x] already done\n", encoding="utf-8")
    idx = types.SimpleNamespace(root=tmp_path)
    calls = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True, review=False, answer_system=None):
        calls.append(task)
        return _result(idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
    assert code == 0
    assert calls == []


# ------------------------------------------------------ missing dependencies
def test_missing_modules_and_pip_names():
    out = ("Traceback (most recent call last):\n"
           "ModuleNotFoundError: No module named 'pygame'\n"
           "ImportError: No module named 'foo.bar'")
    assert loop.missing_modules(out) == ["pygame", "foo"]
    assert loop.missing_modules("nothing here") == []
    assert loop.pip_name("PIL") == "Pillow"
    assert loop.pip_name("sklearn") == "scikit-learn"
    assert loop.pip_name("pygame") == "pygame"


def test_run_loop_does_not_code_repair_missing_dependency(tmp_path, monkeypatch):
    """A verify failure caused by an absent package must NOT spawn repair
    tasks (editing source cannot install it); it is reported and the task
    fails."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["implement main"])
    idx = types.SimpleNamespace(root=tmp_path)
    tasks = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True, review=False, answer_system=None):
        tasks.append(task)
        return _result(idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    monkeypatch.setattr(loop, "run_verify", lambda root, cmd: (
        False, "ModuleNotFoundError: No module named 'pygame'"))
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         verify="python -c 'import main'", repair_rounds=3,
                         auto_install=False)
    assert code == 1
    assert tasks == ["implement main"], "must not spawn code-repair tasks"
    assert "- [ ] implement main" in plan.read_text()


def test_run_loop_auto_installs_missing_dependency(tmp_path, monkeypatch):
    """With auto_install on, the missing package is installed and the verify
    is retried; a pass then marks the task done."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["implement main"])
    idx = types.SimpleNamespace(root=tmp_path)
    installs = []
    verify_calls = {"n": 0}

    def fake_verify(root, cmd):
        verify_calls["n"] += 1
        if verify_calls["n"] == 1:
            return False, "ModuleNotFoundError: No module named 'pygame'"
        return True, ""

    monkeypatch.setattr(loop, "run_task", lambda *a, **k: _result(idx))
    monkeypatch.setattr(loop, "run_verify", fake_verify)
    monkeypatch.setattr(loop, "install_modules",
                        lambda pkgs, root: (installs.append(list(pkgs)), (True, ""))[1])
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         verify="python main.py", auto_install=True)
    assert code == 0
    assert installs == [["pygame"]]
    assert "- [x] implement main" in plan.read_text()


def test_run_loop_auto_install_does_not_loop_when_module_stays_missing(
        tmp_path, monkeypatch):
    """If the install "succeeds" but verify still reports the same missing
    module, the loop must stop instead of reinstalling forever."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["implement main"])
    idx = types.SimpleNamespace(root=tmp_path)
    installs = []

    monkeypatch.setattr(loop, "run_task", lambda *a, **k: _result(idx))
    monkeypatch.setattr(loop, "run_verify", lambda root, cmd: (
        False, "ModuleNotFoundError: No module named 'pygame'"))
    monkeypatch.setattr(loop, "install_modules",
                        lambda pkgs, root: (installs.append(list(pkgs)), (True, ""))[1])
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         verify="python main.py", auto_install=True)
    assert code == 1
    assert installs == [["pygame"]]


def test_install_modules_rejects_option_like_names(tmp_path):
    ok, out = loop.install_modules(["--index-url", "-r"], tmp_path)
    assert ok is False
    assert "no valid package names" in out


def test_normalized_reply_folds_double_encoded_apostrophes():
    for bad in ("can\u2019t", "can\u00e2\u20ac\u2122t", "can\u00e2\x80\x99t"):
        assert loop._normalized_reply(bad) == "can't"
    assert loop._declares_incomplete("I can\u00e2\u20ac\u2122t emit the blocks yet")


def test_run_verify_runs_in_project_root(tmp_path):
    ok, _ = loop.run_verify(tmp_path, f"{sys.executable} -c 'import sys; sys.exit(0)'")
    assert ok is True
    bad, out = loop.run_verify(tmp_path,
                               f"{sys.executable} -c 'import sys; sys.exit(3)'")
    assert bad is False


# ---------------------------------------------------------------- run_task
def test_run_task_applies_edits_and_reindexes(tmp_path):
    """The shared single-task executor must walk gather -> answer -> apply and
    actually write (the path `ask`/`chat`/`auto` all share after the refactor),
    using an offline stub client in place of a live server."""
    from pedacito import index as index_mod
    from pedacito.agent import Agent

    (tmp_path / "old_name.py").write_text('print("hi")\n', encoding="utf-8")
    cfg = Config()
    cfg.log_sessions = False
    idx = index_mod.build([str(tmp_path)], cfg, client=None, summarise=False, root=tmp_path)

    REPLY = (
        "Renaming the file.\n\n"
        "--- FILE: new_name.py\n"
        "<<<<<<< SEARCH\n"
        "=======\n"
        'print("hi")\n'
        ">>>>>>> REPLACE\n\n"
        "--- FILE: old_name.py\n"
        "<<<<<<< SEARCH\n"
        'print("hi")\n'
        "=======\n"
        ">>>>>>> REPLACE\n"
    )

    class StubClient:
        stream_truncated = False
        step_truncated = False
        def __init__(self):
            self.n = 0
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            if self.n == 0:
                self.n += 1
                return {"action": "read", "file": "old_name.py", "pattern": "",
                        "symbol": "", "start": 1, "end": 200, "content": "",
                        "reasoning": "read the file"}
            return {"action": "done", "file": "", "pattern": "", "symbol": "",
                    "start": 0, "end": 0, "content": "", "reasoning": "done"}
        def chat(self, messages, max_tokens, schema=None, stream=False, **kw):
            return REPLY

    client = StubClient()
    tr = loop.run_task(tmp_path, cfg, client, Agent(idx, client, cfg), idx,
                       "rename old_name.py to new_name.py", apply=True,
                       review=False)
    assert tr.applied, "shared executor did not apply the edit"
    assert "new_name.py" in tr.changed           # reindexed after the write
    assert (tmp_path / "new_name.py").is_file()
    assert not (tmp_path / "old_name.py").exists()
    assert 'print("hi")' in (tmp_path / "new_name.py").read_text()
    # The operational run log records the task and its result.
    from pedacito import workspace
    log = workspace.run_log_path(tmp_path).read_text()
    assert "task: rename old_name.py to new_name.py" in log
    assert "applied new_name.py" in log


def test_read_requests_parses_both_tool_call_dialects():
    """Answer-phase replies that ask for reads come in at least two dialects:
    the `<tool_call>`/`<function=read>` form and the Anthropic
    `<invoke name="read">` form. Both must yield their file paths."""
    reply = (
        "<tool_call>\n<function=read>\n<parameter=file_path>\nsrc/game.py\n"
        "</parameter>\n</function>\n</tool_call>\n"
        '<invoke name="read">\n<parameter name="file_path">src/ui.py</parameter>\n'
        "</invoke>\n"
        '<invoke name="write">\n<parameter name="file_path">src/x.py</parameter>\n'
        "</invoke>\n")
    assert loop._read_requests(reply) == ["src/game.py", "src/ui.py"]
    assert loop._read_requests("just prose, no tool calls") == []


def test_honour_reads_refuses_paths_outside_root(tmp_path):
    """Files requested by the model are read relative to the project root;
    anything outside it (or missing) is reported as unreadable instead of
    leaking the host filesystem into the prompt."""
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    out = loop._honour_reads(tmp_path, ["a.py", "../../etc/passwd", "nope.py"])
    assert "x = 1" in out
    assert "### ../../etc/passwd" in out and "not readable" in out
    assert "### nope.py" in out and "not readable" in out


def test_run_task_recovers_after_tool_call_reply(tmp_path):
    """The reported failure: the answer phase emitted `<tool_call>` reads for
    files gather never touched, so no edit blocks parsed and the task died.
    run_task must honour those reads from disk and re-ask -- the re-ask
    carries the requested source and the task then applies cleanly."""
    from pedacito import index as index_mod
    from pedacito.agent import Agent

    (tmp_path / "game.py").write_text("def tick():\n    pass\n", encoding="utf-8")
    (tmp_path / "player.py").write_text("x = 1\n", encoding="utf-8")
    cfg = Config()
    cfg.log_sessions = False
    idx = index_mod.build([str(tmp_path)], cfg, client=None, summarise=False,
                          root=tmp_path)

    TOOL_REPLY = (
        "<tool_call>\n<function=read>\n<parameter=file_path>\ngame.py\n"
        "</parameter>\n</function>\n</tool_call>\n")
    EDIT_REPLY = (
        "Implementing player.\n\n"
        "--- FILE: player.py\n"
        "<<<<<<< SEARCH\nx = 1\n=======\n"
        "def update():\n    pass\n>>>>>>> REPLACE\n")

    class StubClient:
        stream_truncated = False
        step_truncated = False

        def __init__(self):
            self.answers = 0

        def chat_json(self, messages, schema, max_tokens=0, **kw):
            return {"action": "done", "file": "", "pattern": "", "symbol": "",
                    "start": 0, "end": 0, "content": "", "reasoning": "done"}

        def chat(self, messages, max_tokens, schema=None, stream=False, **kw):
            self.answers += 1
            if self.answers == 1:
                return TOOL_REPLY
            user = "\n".join(m["content"] for m in messages
                             if m["role"] == "user")
            assert "EXTRA SOURCE READ FOR YOU" in user
            assert "def tick()" in user
            return EDIT_REPLY

    client = StubClient()
    tr = loop.run_task(tmp_path, cfg, client, Agent(idx, client, cfg), idx,
                       "implement player.py", apply=True, review=False)
    assert tr.applied, "recovery did not apply the re-asked edit"
    assert "def update()" in (tmp_path / "player.py").read_text()
    assert client.answers == 2


def test_edit_refusal_head_is_narrow():
    """Only a reply that OPENS by declaring it made no edit counts -- the
    match DISCARDS real edit blocks, so a qualified statement about one
    file must not sweep them up. Mojibake apostrophes ("didn\u00e2t") are
    folded before matching, and the canonical no-op reply must keep
    working (run_loop ticks it done)."""
    assert loop._edit_refusal_head("No edit.\n\nThe freeze forbids acting.")
    assert loop._edit_refusal_head("No edits were made: the goal conflicts.")
    assert loop._edit_refusal_head("I did not edit any file this round.")
    assert loop._edit_refusal_head("I didn\u00e2t emit the blocks.")
    assert loop._edit_refusal_head("# No changes.\nDetails follow.")
    assert not loop._edit_refusal_head(
        "No changes were needed in main.js; I fixed index.html instead.")
    assert not loop._edit_refusal_head("I removed the stale script tags.")
    assert not loop._edit_refusal_head(
        "No changes are needed; the file is already correct.")


def test_run_task_discards_edit_blocks_that_contradict_a_refusal(tmp_path):
    """Graveyard repair 1: the prose said "No edit." and explained why
    acting would violate the goal's freeze -- then the same reply emitted
    a FILE block stripping 22 script tags from index.html, and the harness
    APPLIED it. When a reply opens by refusing, its blocks are a
    small-model coherence failure, not intent: honor the refusal, discard
    the blocks, and never re-ask (the recovery path would pressure the
    model into undoing its own refusal)."""
    (tmp_path / "index.html").write_text(
        '<script src="/static/js/main.js"></script>\n', encoding="utf-8")
    idx = types.SimpleNamespace(root=tmp_path)
    REPLY = (
        "No edit.\n"
        "\n"
        "Restoring the tags would change what the page executes and "
        "conflicts with the freeze.\n"
        "\n"
        "--- FILE: index.html\n"
        "<<<<<<< SEARCH\n"
        '<script src="/static/js/main.js"></script>\n'
        "=======\n"
        '<script src="/static/js/main.js"></script>\n'
        '<script src="/static/js/player.js"></script>\n'
        ">>>>>>> REPLACE\n")

    class FakeAgent:
        steps_log = []

        def __init__(self):
            self.apply_calls = 0

        def gather(self, task, **kw):
            return "gathered source"

        def answer(self, task, gathered, **kw):
            return REPLY

        def apply(self, reply, **kw):
            self.apply_calls += 1
            return ([_ok("index.html")], reply)

    agent = FakeAgent()
    tr = loop.run_task(tmp_path, _cfg(), None, agent, idx,
                       "Restore the script tags the last repair removed",
                       apply=True)
    assert agent.apply_calls == 0          # blocks never touched the disk
    assert tr.results == []
    assert tr.reply == REPLY               # the refusal stays the record
    assert (tmp_path / "index.html").read_text() == \
        '<script src="/static/js/main.js"></script>\n'
    from pedacito import workspace
    log = workspace.run_log_path(tmp_path).read_text()
    assert "blocks discarded" in log
    # The generic no-blocks ERROR is suppressed: this is a decision the
    # harness honored, not a parse failure.
    assert "no edit blocks parsed" not in log


def test_parse_edits_drops_noop_blocks():
    """Fix X (run 13): a repair that had correctly concluded "no code
    defect" was pressed by the no-edit re-ask into fabricating a block
    whose SEARCH and REPLACE were identical (`window.__game = state;` over
    itself, on a file the goal froze). The harness logged "applied 1
    block(s)", took a backup, and counted it as an EDIT -- burning a full
    verify re-run instead of retiring the check. A null edit is never
    work: drop it at parse time so no caller can mistake it for work."""
    from pedacito.edits import parse_edits
    noop = ("--- FILE: main.js\n"
            "<<<<<<< SEARCH\n"
            "window.__game = state;\n"
            "=======\n"
            "window.__game = state;\n"
            ">>>>>>> REPLACE\n")
    discards = []
    assert parse_edits(noop, discards=discards) == []
    assert discards == ["main.js"]

    real = ("--- FILE: main.js\n"
            "<<<<<<< SEARCH\n"
            "window.__game = state;\n"
            "=======\n"
            "window.__game = state;\n"
            "window.__debug = true;\n"
            ">>>>>>> REPLACE\n")
    both = noop + "\n" + real
    discards2 = []
    got = parse_edits(both, discards=discards2)
    assert len(got) == 1 and "window.__debug" in got[0].replace
    assert discards2 == ["main.js"]          # only the null block


def test_run_task_treats_fabricated_noop_blocks_as_a_refusal(tmp_path):
    """Fix X end-to-end: the reply's only blocks were null edits, dropped
    by parse_edits. run_task must NOT re-ask for "real" blocks -- that
    pressure is what fabricated them in run 13 -- and must mark the reply
    so the outer loop's no-op path sees the refusal (repair_outcome False
    -> the failing check retires instead of burning another verify
    round)."""
    (tmp_path / "main.js").write_text("window.__game = state;\n",
                                      encoding="utf-8")
    idx = types.SimpleNamespace(root=tmp_path)
    REPLY = ("No code defect was found: every failure was an eval payload "
             "truncation, and the task forbids changes.\n"
             "\n"
             "--- FILE: main.js\n"
             "<<<<<<< SEARCH\n"
             "window.__game = state;\n"
             "=======\n"
             "window.__game = state;\n"
             ">>>>>>> REPLACE\n")

    class FakeAgent:
        steps_log = []

        def __init__(self):
            self.apply_calls = 0
            self.noop_discards = []

        def gather(self, task, **kw):
            return "gathered source"

        def answer(self, task, gathered, **kw):
            return REPLY

        def apply(self, reply, **kw):
            self.apply_calls += 1
            self.noop_discards = ["main.js"]   # parse_edits dropped the block
            return ([], reply)

    agent = FakeAgent()
    tr = loop.run_task(tmp_path, _cfg(), None, agent, idx,
                       "Repair the defect described in the FAILED LIVE TEST "
                       "note", apply=True)
    assert agent.apply_calls == 1              # never re-asked for blocks
    assert tr.results == []
    assert "[No changes:" in tr.reply
    assert loop._is_noop(tr.reply)             # the outer loop sees refusal
    assert (tmp_path / "main.js").read_text() == "window.__game = state;\n"
    from pedacito import workspace
    log = workspace.run_log_path(tmp_path).read_text()
    assert "treated as a refusal" in log
    # Suppressed: a fabricated null edit is a decision the harness honored,
    # not a parse failure.
    assert "no edit blocks parsed" not in log


def test_recover_no_edit_reply_offers_an_explicit_no_edit_way_out(tmp_path):
    """Fix X, pressure side: the old re-ask said only "Produce the edit
    blocks now." -- a model that had correctly concluded no edit was
    needed could comply only by fabricating one, and in run 13 it did
    (twice). The re-ask must name the honest exit (one 'No changes:' line)
    and forbid blocks that change nothing."""
    asked = []

    class FakeAgent:
        def answer(self, task, gathered, **kw):
            asked.append(task)
            return "still no blocks"

        def apply(self, reply, **kw):
            return ([], reply)

    rec = loop._recover_no_edit_reply(tmp_path, FakeAgent(),
                                      "Repair the defect", "src",
                                      "prose reply",
                                      dry_run=True, backup_dir=None)
    assert rec is None
    assert asked, "the re-ask never ran"
    assert "No changes:" in asked[0]
    assert "identical" in asked[0]


def test_recover_honors_the_no_changes_escape_hatch_reply(tmp_path):
    """Fix X2 (run 13d): the repair took the escape hatch the re-ask
    teaches -- an honest "No changes: <reason>" line -- and the old loop
    DISCARDED it: only block-producing rounds were returned, so the
    model was pressed a second time, both honest replies were thrown
    away (None kept the original prose), and the run stopped on a task
    that had correctly concluded no edit was warranted. The declaration
    must end the re-ask loop and propagate as the reply."""
    asked = []

    class FakeAgent:
        def answer(self, task, gathered, **kw):
            asked.append(task)
            return ("No changes: the goal freezes all game code and the "
                    "failure was a measurement error, not a defect.")

        def apply(self, reply, **kw):
            return ([], reply)

    rec = loop._recover_no_edit_reply(tmp_path, FakeAgent(),
                                      "Repair the defect", "src",
                                      "prose reply",
                                      dry_run=True, backup_dir=None)
    assert rec is not None
    results, reply = rec
    assert results == []
    assert reply.startswith("No changes:")
    assert len(asked) == 1               # honored at once, no second re-ask


def test_run_task_propagates_a_reask_no_changes_reply(tmp_path):
    """Fix X2, run_task side: the propagated declaration becomes the
    task reply -- run_loop keys its refusal path off _is_noop(tr.reply)
    (for a repair: repair_outcome False -> the failing check retires)
    -- and the "no edit blocks parsed" ERROR stays out of the log: an
    honest refusal is a decision, not a parse failure."""
    idx = types.SimpleNamespace(root=tmp_path)
    answers = []

    class FakeAgent:
        steps_log = []
        noop_discards = []

        def gather(self, task, **kw):
            return "gathered source"

        def answer(self, task, gathered, **kw):
            answers.append(task)
            if len(answers) == 1:
                return ("The failure was a measurement error, so editing "
                        "is off the table.")
            return "No changes: the goal freezes all game code."

        def apply(self, reply, **kw):
            return ([], reply)

    tr = loop.run_task(tmp_path, _cfg(), None, FakeAgent(), idx,
                       "Repair the defect described in the FAILED LIVE "
                       "TEST note", apply=True)
    assert len(answers) == 2             # first reply + exactly one re-ask
    assert tr.results == []
    assert tr.reply.startswith("No changes:")
    assert loop._is_noop(tr.reply)       # the outer loop sees the refusal
    from pedacito import workspace
    log = workspace.run_log_path(tmp_path).read_text()
    assert "no edit blocks parsed" not in log
    assert "honored as a refusal" in log


def test_is_noop_matches_the_refusal_phrasing_models_actually_use():
    """Fix X2, pattern side: run 13d's FIRST repair reply -- "No code
    edit is appropriate here" -- matched none of the _is_noop patterns
    ("no edit" needs the words adjacent), so the honest refusal fell
    into the re-ask loop at all. The phrasing must count as a no-op
    declaration the moment it arrives."""
    assert loop._is_noop("No code edit is appropriate here.")
    assert loop._is_noop("No changes: the goal freezes the code.")
    assert not loop._is_noop("I moved the spawn gate to time >= 3.")


def test_run_log_records_events(tmp_path):
    from pedacito import workspace
    workspace.log_event(tmp_path, "hello", "INFO")
    workspace.log_event(tmp_path, "boom", "ERROR")
    log = workspace.run_log_path(tmp_path).read_text()
    assert "hello" in log and "boom" in log
    assert "INFO" in log and "ERROR" in log
    # one line per event, with a timestamp and a padded kind column
    lines = log.splitlines()
    assert len(lines) == 2
    assert "  INFO   " in lines[0] and "  ERROR  " in lines[1]


# ------------------------------------------------------ auto plan regeneration
def test_auto_regenerates_when_plan_is_complete(tmp_path):
    """`auto -t "new goal"` on a project whose PLAN.md is all-checked must
    start a fresh plan from the new goal, not silently re-read the finished
    checklist and report 'all tasks complete' without doing anything."""
    from pedacito import cli
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "old goal", ["old task"])
    assert loop.mark_done(plan, "old task")

    class Stub:
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            return {"tasks": [{"title": "fix the bug"}, {"title": "add a test"}]}

    args = types.SimpleNamespace(task="fix the bug", spec=None, plan=None,
                                 apply=False, force=False, verify=None,
                                 repair=None, max_tasks=None, keep_going=False)
    code = cli._cmd_auto(args, Config(), Stub(), None, tmp_path)
    assert code == 0
    text = plan.read_text()
    assert "- [ ] fix the bug" in text
    assert "- [ ] add a test" in text
    assert "old task" not in text


def test_auto_resumes_plan_with_open_tasks(tmp_path):
    """The inverse: a plan that still has unchecked work is resumed, not
    overwritten, when the same goal (or any -t) is given without --force."""
    from pedacito import cli
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal", ["done task", "open task"])
    assert loop.mark_done(plan, "done task")

    class Stub:
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            raise AssertionError("generate_tasks must not be called to resume")

    args = types.SimpleNamespace(task="anything", spec=None, plan=None,
                                 apply=False, force=False, verify=None,
                                 repair=None, max_tasks=None, keep_going=False)
    code = cli._cmd_auto(args, Config(), Stub(), None, tmp_path)
    assert code == 0
    text = plan.read_text()
    assert "- [x] done task" in text
    assert "- [ ] open task" in text


def test_auto_runs_post_verify_after_apply(tmp_path, monkeypatch):
    """`auto --apply` must exercise the project once after the checklist is
    done and feed a failure back, rather than declaring success blindly."""
    from pedacito import cli, scaffold
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal", ["only task"])
    seen = {}

    monkeypatch.setattr(loop, "run_loop", lambda *a, **k: 0)
    monkeypatch.setattr(scaffold, "verify_and_fix",
                        lambda *a, **k: seen.update(k) or (True, ""))

    args = types.SimpleNamespace(task=None, spec=None, plan=None, apply=True,
                                 force=False, verify=None, repair=None,
                                 max_tasks=None, keep_going=False,
                                 auto_install=False)
    code = cli._cmd_auto(args, Config(), None, None, tmp_path)
    assert code == 0
    assert "smoke" in seen, "auto --apply did not run the post-verify"


# ------------------------------------------------------ `new --complex` chaining
def test_plan_tasks_lists_one_task_per_source_file():
    from pedacito import scaffold
    plan = scaffold.Plan("md2html", "y", files=[
        scaffold.PlannedFile("main.py", "entry"),
        scaffold.PlannedFile("parser.py", "parse"),
        scaffold.PlannedFile("game/__init__.py", "package"),    # skipped
        scaffold.PlannedFile("README.md", "docs"),              # excluded
        scaffold.PlannedFile("tests/test_parser.py", "t"),      # excluded
    ], spec=scaffold.Spec("y"))
    assert scaffold.plan_tasks(plan) == [
        "Implement main.py -- entry",
        "Implement parser.py -- parse",
    ]


def test_plan_tasks_splits_oversized_files(tmp_path):
    """A source file whose stub is longer than auto_split_lines must be split
    into several tasks (one per group of top-level definitions) so no single
    whole-file reply overflows the model's output budget."""
    from pedacito import scaffold

    stub = "".join(
        f"def f{i}():\n    \"\"\"doc\"\"\"\n    pass\n\n" for i in range(10)
    )
    (tmp_path / "big.py").write_text(stub, encoding="utf-8")
    cfg = Config()
    plan = scaffold.Plan("x", "y", files=[scaffold.PlannedFile("big.py", "big")],
                         spec=scaffold.Spec("y"))
    idx = scaffold.skeleton_index(tmp_path, plan, cfg)
    tasks = scaffold.plan_tasks(plan, idx=idx, max_lines=8)
    assert len(tasks) > 1, f"expected split, got {tasks}"
    assert all("big.py:" in t and "lines " in t for t in tasks), tasks
    # every function is covered exactly once across the split tasks
    names = " ".join(tasks)
    for i in range(10):
        assert f"f{i}" in names

    # A small file is left whole.
    (tmp_path / "small.py").write_text("def f():\n    pass\n", encoding="utf-8")
    small_plan = scaffold.Plan("x", "y",
                               files=[scaffold.PlannedFile("small.py", "s")],
                               spec=scaffold.Spec("y"))
    idx2 = scaffold.skeleton_index(tmp_path, small_plan, cfg)
    assert scaffold.plan_tasks(small_plan, idx=idx2, max_lines=120) == \
        ["Implement small.py -- s"]


def test_new_complex_chains_to_auto_loop(tmp_path, monkeypatch):
    """`new --complex --apply` must scaffold the skeleton, write a resumable
    task checklist, and hand implementation to the outer loop with the
    stub-implementing answer brief."""
    from pedacito import scaffold

    calls = {}
    def fake_run_loop(root, cfg, client, idx, plan_path, **kw):
        calls["root"], calls["plan_path"], calls["kw"] = root, plan_path, kw
        return 0
    monkeypatch.setattr(loop, "run_loop", fake_run_loop)

    class Stub:
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            if "project_name" in schema["properties"]:
                return {"project_name": "md2html", "summary": "s",
                        "files": [{"path": "main.py", "purpose": "entry"},
                                  {"path": "parser.py", "purpose": "parse"}]}
            return {"files": [
                {"path": "main.py", "code": "def main():\n    pass\n"},
                {"path": "parser.py",
                 "code": "def parse(text):\n    raise NotImplementedError\n"},
            ]}
        def chat(self, *a, **k):
            return ""

    cfg = Config()
    code = scaffold.run_new(tmp_path, cfg, Stub(), scaffold.Spec("md2html"),
                            apply=True, complex=True, verbose=False)
    assert code == 0
    assert calls["kw"]["answer_system"] == scaffold.AUTO_IMPLEMENT_SYSTEM
    assert calls["plan_path"].name == "PLAN.md"
    # The checklist has one task per source file.
    text = "\n".join(loop.read_plan(calls["plan_path"]))
    assert "Implement main.py -- entry" in text
    assert "Implement parser.py -- parse" in text
    # The skeleton (interface stubs) is on disk before the loop runs.
    assert (tmp_path / "main.py").is_file()
    assert "def parse(text)" in (tmp_path / "parser.py").read_text()


def test_new_complex_dry_run_writes_nothing(tmp_path, monkeypatch):
    """`new --complex` WITHOUT --apply stays a dry run: nothing on disk, no
    loop, and the task list is only previewed."""
    from pedacito import scaffold

    ran = []
    def fake_run_loop(root, cfg, client, idx, plan_path, **kw):
        ran.append(plan_path)
        return 0
    monkeypatch.setattr(loop, "run_loop", fake_run_loop)

    class Stub:
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            return {"project_name": "x", "summary": "s",
                    "files": [{"path": "main.py", "purpose": "entry"}]}
        def chat(self, *a, **k):
            return ""

    code = scaffold.run_new(tmp_path, Config(), Stub(), scaffold.Spec("x"),
                            complex=True, verbose=False)
    assert code == 0
    assert ran == []
    assert list(tmp_path.iterdir()) == []


def _complex_stub():
    class Stub:
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            if "project_name" in schema["properties"]:
                return {"project_name": "x", "summary": "s",
                        "files": [{"path": "main.py", "purpose": "entry"}]}
            return {"files": [{"path": "main.py",
                               "code": "def main():\n    pass\n"}]}
        def chat(self, *a, **k):
            return ""
    return Stub()


def test_new_complex_runs_import_repair(tmp_path, monkeypatch):
    """Complex mode implements each file in a fresh context, so cross-file
    names can drift. It must run the same import-repair pass the simple `new`
    build does, or it reports success while shipping an unimportable tree."""
    from pedacito import scaffold

    calls = {"fix": 0}
    seq = {"n": 0}

    def fake_first_error(root, plan):
        seq["n"] += 1
        # A code-level mismatch (not a missing package), then clean.
        return ("ImportError: cannot import name 'x' from 'y'"
                if seq["n"] == 1 else None)

    monkeypatch.setattr(loop, "run_loop", lambda *a, **k: 0)
    monkeypatch.setattr(scaffold, "fix_import_errors",
                        lambda *a, **k: calls.__setitem__("fix", calls["fix"] + 1) or 0)
    monkeypatch.setattr(scaffold, "_first_import_error", fake_first_error)

    code = scaffold.run_new(tmp_path, Config(), _complex_stub(),
                            scaffold.Spec("x"), apply=True, complex=True,
                            verbose=False)
    assert calls["fix"] == 1, "complex mode skipped the import-repair pass"
    assert code == 0


def test_new_complex_handles_missing_dependency(tmp_path, monkeypatch):
    """A missing third-party package must not be sent through the code-repair
    loop (editing source cannot install it). With --auto-install it is
    installed and the import re-checked."""
    from pedacito import scaffold

    state = {"installed": [], "fix": 0, "checks": 0}

    def fake_first_error(root, plan):
        state["checks"] += 1
        if state["installed"]:
            return None
        return "ModuleNotFoundError: No module named 'pygame'"

    monkeypatch.setattr(loop, "run_loop", lambda *a, **k: 0)
    monkeypatch.setattr(scaffold, "_first_import_error", fake_first_error)
    monkeypatch.setattr(scaffold, "fix_import_errors",
                        lambda *a, **k: state.__setitem__("fix", state["fix"] + 1) or 0)

    def fake_install(pkgs, root):
        state["installed"].extend(pkgs)
        return True, ""
    monkeypatch.setattr(loop, "install_modules", fake_install)

    code = scaffold.run_new(tmp_path, Config(), _complex_stub(),
                            scaffold.Spec("x"), apply=True, complex=True,
                            verbose=False, auto_install=True)
    assert state["installed"] == ["pygame"]
    assert state["fix"] == 0, "a missing dep must not trigger code repair"
    assert code == 0


def test_new_complex_reports_missing_dependency_without_install(tmp_path, monkeypatch):
    from pedacito import scaffold

    state = {"fix": 0}
    monkeypatch.setattr(loop, "run_loop", lambda *a, **k: 0)
    monkeypatch.setattr(scaffold, "_first_import_error",
                        lambda root, plan: "ModuleNotFoundError: No module named 'pygame'")
    monkeypatch.setattr(scaffold, "fix_import_errors",
                        lambda *a, **k: state.__setitem__("fix", state["fix"] + 1) or 0)

    code = scaffold.run_new(tmp_path, Config(), _complex_stub(),
                            scaffold.Spec("x"), apply=True, complex=True,
                            verbose=False, auto_install=False)
    assert state["fix"] == 0, "a missing dep must not trigger code repair"
    assert code == 1


def test_new_complex_flags_still_broken_import(tmp_path, monkeypatch):
    """If the tree still cannot import after repair, the run must report a
    non-zero exit instead of quietly claiming success."""
    from pedacito import scaffold

    monkeypatch.setattr(loop, "run_loop", lambda *a, **k: 0)
    monkeypatch.setattr(scaffold, "fix_import_errors", lambda *a, **k: 0)
    monkeypatch.setattr(scaffold, "_first_import_error",
                        lambda root, plan: "ImportError: cannot import name 'x'")

    code = scaffold.run_new(tmp_path, Config(), _complex_stub(),
                            scaffold.Spec("x"), apply=True, complex=True,
                            verbose=False)
    assert code == 1


def test_fix_api_mismatches_repairs_a_call(tmp_path):
    """The API repair pass must turn a wrong-arity call into an applied edit,
    driving the tree back to consistency."""
    from pedacito import scaffold
    from pedacito import index as index_mod

    (tmp_path / "main.py").write_text(
        "from core import Game\n\ndef main():\n    Game()\n", encoding="utf-8")
    (tmp_path / "core.py").write_text(
        "class Game:\n    def __init__(self, x):\n        self.x = x\n",
        encoding="utf-8")
    idx = index_mod.Index(tmp_path, [])
    plan = scaffold.Plan("x", "y", files=[
        scaffold.PlannedFile("main.py"), scaffold.PlannedFile("core.py")],
        spec=scaffold.Spec("y"))

    class Stub:
        def chat(self, messages, max_tokens, schema=None, stream=False, **kw):
            return ("--- FILE: main.py\n<<<<<<< SEARCH\n    Game()\n"
                    "=======\n    Game(1)\n>>>>>>> REPLACE\n")

    n = scaffold.fix_api_mismatches(Stub(), Config(), idx, plan, None,
                                    verbose=False)
    assert n == 1
    assert "Game(1)" in (tmp_path / "main.py").read_text()
    from pedacito import apicheck
    assert apicheck.find_mismatches(
        scaffold._py_sources(tmp_path, ["main.py", "core.py"])) == []


# ------------------------------------------------------- runtime verification
def test_smoke_run_clean_exit_passes(tmp_path):
    from pedacito import scaffold
    (tmp_path / "main.py").write_text("print('ok')\n", encoding="utf-8")
    ok, _ = scaffold.smoke_run(tmp_path, "main.py", timeout=10)
    assert ok


def test_smoke_run_crash_fails_with_traceback(tmp_path):
    from pedacito import scaffold
    (tmp_path / "main.py").write_text("raise SystemExit('boom')\n",
                                      encoding="utf-8")
    ok, out = scaffold.smoke_run(tmp_path, "main.py", timeout=10)
    assert not ok and "boom" in out


def test_smoke_run_timeout_means_started(tmp_path):
    """A game/server that keeps running is a SUCCESS: it started."""
    from pedacito import scaffold
    (tmp_path / "main.py").write_text("import time\ntime.sleep(5)\n",
                                      encoding="utf-8")
    ok, out = scaffold.smoke_run(tmp_path, "main.py", timeout=1)
    assert ok and "still running" in out


def test_verify_and_fix_repairs_smoke_failure(tmp_path, monkeypatch):
    """A smoke-run crash is fed back to the model and the check retried."""
    from pedacito import scaffold
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    idx = types.SimpleNamespace(root=tmp_path)
    calls = {"n": 0}

    def fake_smoke(root, entry="main.py", timeout=20):
        calls["n"] += 1
        return (calls["n"] >= 2), ("TypeError: boom" if calls["n"] < 2 else "")

    fixes = []
    monkeypatch.setattr(scaffold, "smoke_run", fake_smoke)
    monkeypatch.setattr(scaffold, "_fix_from_output",
                        lambda *a, **k: fixes.append(1) or 1)
    ok, _ = scaffold.verify_and_fix(None, Config(), idx, ["main.py"], "x", None,
                                    smoke=True, rounds=2)
    assert ok
    assert len(fixes) == 1, "a failing smoke run must trigger a repair round"


def test_verify_and_fix_ignores_headless_display_error(tmp_path, monkeypatch):
    """No display is the machine's problem, not the code's: do not repair it,
    do not fail the run."""
    from pedacito import scaffold
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    idx = types.SimpleNamespace(root=tmp_path)
    monkeypatch.setattr(scaffold, "smoke_run",
                        lambda root, entry="main.py", timeout=20: (
                            False, "pygame.error: No available video device"))
    fixes = []
    monkeypatch.setattr(scaffold, "_fix_from_output",
                        lambda *a, **k: fixes.append(1) or 1)
    ok, out = scaffold.verify_and_fix(None, Config(), idx, ["main.py"], "x", None,
                                      smoke=True)
    assert ok and fixes == []
    assert "environment" in out


def test_verify_and_fix_does_not_code_repair_missing_dependency(tmp_path, monkeypatch):
    from pedacito import scaffold
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    idx = types.SimpleNamespace(root=tmp_path)
    monkeypatch.setattr(scaffold, "smoke_run",
                        lambda root, entry="main.py", timeout=20: (
                            False, "ModuleNotFoundError: No module named 'pygame'"))
    fixes = []
    monkeypatch.setattr(scaffold, "_fix_from_output",
                        lambda *a, **k: fixes.append(1) or 1)
    ok, _ = scaffold.verify_and_fix(None, Config(), idx, ["main.py"], "x", None,
                                    smoke=True)
    assert not ok
    assert fixes == [], "a missing dep must not be sent to the repair loop"


# ------------------------------------------------- custom answer brief + format
def test_custom_answer_system_still_carries_edit_format(tmp_path):
    """A custom answer brief (scaffold stub implementation, resume) must still
    carry the exact search/replace FORMAT, or a small model emits fenced
    blocks without a `--- FILE:` header and the parser finds nothing."""
    from pedacito.agent import Agent, ANSWER_SYSTEM, EDIT_FORMAT

    captured = {}

    class StubClient:
        stream_truncated = False
        step_truncated = False
        def chat(self, messages, max_tokens, schema=None, stream=False, **kw):
            captured["system"] = messages[0]["content"]
            return "ok"

    idx = types.SimpleNamespace(
        root=tmp_path,
        map_for_prompt=lambda cfg: ("MAP", "flat"),
    )
    cfg = Config()
    agent = Agent(idx, StubClient(), cfg)
    agent.answer("do a thing", "some source", system="IMPLEMENT THE STUBS",
                 stream=False)
    sys_prompt = captured["system"]
    assert "IMPLEMENT THE STUBS" in sys_prompt
    assert "--- FILE:" in sys_prompt
    assert ">>>>>>> REPLACE" in sys_prompt
    # The default brief is unchanged: intro + the shared format.
    assert ANSWER_SYSTEM.startswith("You are a code assistant")
    assert ANSWER_SYSTEM.endswith(EDIT_FORMAT)


def test_read_requests_parses_malformed_qwen_tool_calls():
    """The reported runaway shape: `<tool_call>` + `<read>` + `<file_path>`,
    repeated without closing tags. It must still yield the requested paths so
    the answer-phase recovery can supply the source instead of re-asking the
    model to emit the same broken block forever."""
    reply = (
        "<tool_call>\n<read>\n<file_path>\ngame.py\n</file_path>\n</read>\n"
        "<tool_call>\n<read>\n<file_path>\nplayer.py"
    )
    assert loop._read_requests(reply) == ["game.py", "player.py"]
    assert loop._looks_like_tool_calls_only(reply)


def test_tool_call_only_detection_ignores_a_giant_repeated_path():
    """A degenerate reply can be almost entirely one huge path; the old
    word-count heuristic counted that path as prose and treated it as a
    valid investigation report."""
    reply = ("<tool_call>read\n</think>\n<read>\n<file_path>\n"
             + "/Users/x/Projects/" * 80 + "\n")
    assert loop._looks_like_tool_calls_only(reply)


# ------------------------------------------------------- rollback-if-worse
def test_failure_score_counts_pytest_failures_and_errors():
    assert loop._failure_score("== 2 failed, 30 passed in 1.2s ==") == 2
    assert loop._failure_score("1 failed, 2 errors during collection") == 3
    # no pytest counters: a flat 1, comparable across rounds but no finer
    assert loop._failure_score("3 errors") == 1
    assert loop._failure_score("boom") == 1
    assert loop._failure_score("") == 1


def test_failure_score_counts_diagnostic_finding_lines():
    report = ("app.py:3:1: E501 line too long\n"
              "app.py:9:12: F821 undefined name 'y'\n"
              "util.py:1:1: W291 trailing whitespace\n"
              "Found 3 problems.")
    assert loop._failure_score(report, diag=True) == 3
    assert loop._failure_score("no findings here", diag=True) == 1
    # pytest counters win even in diag mode
    assert loop._failure_score("2 failed, 1 passed", diag=True) == 2


def test_run_loop_rolls_back_a_repair_that_made_verify_worse(tmp_path,
                                                             monkeypatch):
    """A repair round that raises the failure count is undone from its own
    snapshot (the state before that round's writes), the better earlier state
    survives, and the loop stops repairing instead of digging deeper."""
    from pedacito import workspace

    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["needs tests"])
    idx = types.SimpleNamespace(root=tmp_path)
    (tmp_path / "a.py").write_text("GOOD = 1\n", encoding="utf-8")
    tasks = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None):
        tasks.append(task)
        if len(tasks) == 1:
            return loop.TaskResult([_ok()], "reply", True, ["a.py"], idx_)
        # the repair: snapshots a.py first, then makes things worse
        snap = workspace.new_backup_dir(root)
        (snap / "a.py").write_bytes((root / "a.py").read_bytes())
        (root / "a.py").write_text("GOOD = 1\nBAD = 2\nWORSE = 3\n",
                                   encoding="utf-8")
        return loop.TaskResult([_ok()], "reply", True, ["a.py"], idx_,
                               backup_dir=snap)

    checks = iter([(False, "1 failed"), (False, "3 failed")])
    monkeypatch.setattr(loop, "run_task", fake_run_task)
    monkeypatch.setattr(loop, "run_verify", lambda root, cmd: next(checks))
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         verify="pytest -q", repair_rounds=3,
                         stop_on_fail=False)
    assert code == 1
    assert len(tasks) == 2                       # gave up after the rollback
    assert "failed this check" in tasks[1]
    assert (tmp_path / "a.py").read_text() == "GOOD = 1\n"
    assert "- [!] needs tests" in plan.read_text()


def test_run_loop_keeps_repairing_while_the_score_does_not_rise(tmp_path,
                                                                monkeypatch):
    """Same count (or better) is not 'worse': the rounds continue until
    repair_rounds runs out, with no rollback."""
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["needs tests"])
    idx = types.SimpleNamespace(root=tmp_path)
    tasks = []

    def fake_run_task(root, cfg, client, agent, idx_, task, apply=True,
                      review=False, answer_system=None):
        tasks.append(task)
        return loop.TaskResult([_ok()], "reply", True, ["a.py"], idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    monkeypatch.setattr(loop, "run_verify", lambda root, cmd: (False,
                                                               "1 failed"))
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True,
                         verify="pytest -q", repair_rounds=2)
    assert code == 1
    assert len(tasks) == 3                       # initial + two repairs


# ---------------------------------------------------------- plan title lint
def test_lint_plan_titles_warns_on_bundled_changes(tmp_path, capsys):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["Fix the parser and update the docs",
                               "Add the retry helper"])
    loop._lint_plan_titles(tmp_path, plan)
    err = capsys.readouterr().err
    assert "plan title bundles 2 changes" in err
    assert "fix, update" in err
    assert "Add the retry helper" not in err     # one verb: no warning


def test_lint_plan_titles_ignores_checked_lines(tmp_path, capsys):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["Fix and update everything"])
    loop.mark_done(plan, "Fix and update everything")
    loop._lint_plan_titles(tmp_path, plan)
    assert "bundles" not in capsys.readouterr().err


def test_run_loop_lints_bundled_plan_titles_at_start(tmp_path, monkeypatch,
                                                     capsys):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "", ["Fix the parser and update the docs"])
    idx = types.SimpleNamespace(root=tmp_path)
    monkeypatch.setattr(loop, "run_task", lambda *a, **k: _result(idx))
    loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
    assert "plan title bundles" in capsys.readouterr().err


def test_plan_system_wants_one_change_per_title():
    assert "One title, one change" in loop.PLAN_SYSTEM
