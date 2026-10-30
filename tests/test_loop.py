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


def test_plan_system_requires_constraints_in_titles():
    """Run 6's planner dropped "keeping the doubled height" from the fix
    task's title, and the worker -- whose attention goes to the title --
    deleted rows instead of replacing them, shrinking the sprite."""
    assert "Carry every concrete constraint" in loop.PLAN_SYSTEM


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
        tmp_path, cfg, None, FakeAgent(),
        types.SimpleNamespace(root=tmp_path), "Reproduce the bug")
    assert len(calls) == 2
    assert tr.reply.startswith("A real report")

    class AlwaysEmpty(FakeAgent):
        def answer(self, task, gathered, **kw):
            calls.append(1)
            return ""

    tr = loop.run_investigation_task(
        tmp_path, cfg, None, AlwaysEmpty(),
        types.SimpleNamespace(root=tmp_path), "Reproduce the bug")
    assert len(calls) == 4
    assert tr.reply.startswith("No prose report was produced")
    assert "1. browser python main.py" in tr.reply


def test_tool_call_only_answer_is_detected():
    assert loop._looks_like_tool_calls_only(
        '<tool_call>{"name": "run", "arguments": {}}</tool_call>')
    assert not loop._looks_like_tool_calls_only(
        "Likely cause: the surface is never cleared. "
        "The run produced no output, so the loop is spun before drawing.")


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

    def fake_investigate(root, cfg, client, agent, idx_, task,
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
    checkpoint that admits it gathered no evidence must fail the run."""
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

    def fake_investigate(root, cfg, client, agent, idx_, task,
                         answer_system=None):
        calls.append(("investigate", task))
        return loop.TaskResult(
            [], "I can't launch the browser in this final phase, so I "
                "can't produce a live visual verification.", True, [], idx_)

    monkeypatch.setattr(loop, "run_task", fake_run_task)
    monkeypatch.setattr(loop, "run_investigation_task", fake_investigate)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
    assert code == 1
    assert [kind for kind, _ in calls] == ["edit", "investigate"]
    # The verify task is told up front that tools live in the lookup phase.
    assert "EVIDENCE" in calls[1][1]
    text = plan.read_text()
    assert "- [x] Fix the snake colors" in text
    assert "- [x] Run the game" not in text


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
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
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

    def fake_investigate(root, cfg, client, agent, idx_, task,
                         answer_system=None):
        return loop.TaskResult(
            [], "Verified: the snake renders 180px tall with a diamond head.",
            True, [], idx_,
            steps=["1. read static/js/main.js L195-230  (Check frames.)"])

    monkeypatch.setattr(loop, "run_investigation_task", fake_investigate)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
    assert code == 1
    assert "- [x]" not in plan.read_text()


def test_run_loop_live_verify_passes_with_browser_step(tmp_path, monkeypatch):
    plan = tmp_path / "PLAN.md"
    loop.write_plan(plan, "goal",
                    ["Run the game and verify the snake is taller"])
    idx = types.SimpleNamespace(root=tmp_path)

    def fake_investigate(root, cfg, client, agent, idx_, task,
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

    def fake_investigate(root, cfg, client, agent, idx_, task,
                         answer_system=None):
        return loop.TaskResult(
            [], "Verification failed in the captured run: each snake still "
                "shows a second head block halfway down its body.",
            True, [], idx_,
            steps=["1. browser python main.py [press Enter; wait 3]  (Look.)"])

    monkeypatch.setattr(loop, "run_investigation_task", fake_investigate)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
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

    def fake_investigate(root, cfg, client, agent, idx_, task,
                         answer_system=None):
        return loop.TaskResult(
            [], "No prose report was produced; the lookup phase recorded:\n\n"
                "3. browser python main.py [press Enter; wait 10]  (Look.)",
            True, [], idx_,
            steps=["3. browser python main.py [press Enter; wait 10]  (Look.)"])

    monkeypatch.setattr(loop, "run_investigation_task", fake_investigate)
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
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
    code = loop.run_loop(tmp_path, _cfg(), None, idx, plan, apply=True)
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
    assert "EDITS APPLIED to: static/js/main.js" in seen["task"]
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

    def fake_investigate(root, cfg, client, agent, idx_, task,
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
    assert ANSWER_SYSTEM.startswith("You are a Python code assistant")
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
