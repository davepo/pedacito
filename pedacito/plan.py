"""Plan mode: split a multi-part task into single-focus steps, run each in a
fresh context, and re-plan the rest when it goes wrong.

    pedacito ask . -t "add retries to sync and log them and update the docs" --plan

A small model that must change three things in one context window usually does
the first, half-does the second and forgets the third. Plan mode gives each
change its own window, its own lookups, its own evidence checks and rollback
(`loop.run_loop`), with the plan as a plain markdown checklist under
`.pedacito/plans/` that you can read and edit.

  * Splitting is deterministic first: a task with bullets or several clear
    requirements (`verify.requirement_ledger`) becomes one step per requirement
    with no model call. Only a task that does not split that way is sent to the
    planner role.
  * The planner may give `depends_on`; steps are ordered by it.
  * After a failed step the planner is asked ONCE to revise the remaining steps
    given what finished, what failed and why (`replan`).
  * `plan_threshold` (or the model card) turns this on automatically for tasks
    that name that many files or hold that many requirements; `--plan` forces it.
"""

from __future__ import annotations

import sys
from pathlib import Path

from . import loop, rank, verify, workspace


def requirement_titles(task: str, limit: int = 6) -> list[str]:
    """One step per clearly separate requirement, '' when it is one change."""
    reqs = verify.requirement_ledger(task, limit=limit)
    return [text for text, _terms in reqs] if len(reqs) >= 2 else []


def should_plan(task: str, idx, cfg) -> bool:
    """Whether to auto-plan: enough files named, or enough requirements."""
    n = int(getattr(cfg, "plan_threshold", 0) or 0)
    if n <= 0:
        return False
    try:
        named = len(rank.task_files(idx, task))
    except Exception:
        named = 0
    return named >= n or len(requirement_titles(task, limit=8)) >= n


def make_plan(task: str, idx, client, cfg, root: Path
              ) -> tuple[list[str], str]:
    """(titles, how) for `task`: 'requirements' (free) or 'planner'."""
    titles = requirement_titles(task)
    if titles:
        return titles, "requirements"
    from . import roles
    planner = roles.client_for(cfg, "planner", client)
    got, deps = loop.generate_tasks(task, planner, cfg, root=root,
                                    with_deps=True)
    return loop.order_by_dependencies(got, deps), "planner"


REPLAN_SYSTEM = """\
You revise the REMAINING steps of a coding plan after something went wrong.
Reply with the new ordered list of remaining steps only (the finished ones
stay finished). A failed step may need to be split smaller, preceded by an
investigation step, or replaced by a different approach; do not simply repeat
it unchanged. One change per step. Never return zero steps unless the goal is
already met."""


def replan(task: str, plan_path: Path, client, cfg, failed: str,
           notes: str = "") -> list[str]:
    """One planner call: revised remaining steps ([] when it cannot)."""
    lines = plan_path.read_text(encoding="utf-8").splitlines()
    done = [ln for ln in lines if ln.strip().startswith(("- [x]", "- [X]"))]
    left = [ln for ln in lines if loop._is_unchecked(ln)]
    user = (f"GOAL\n{task}\n\nFINISHED\n" + ("\n".join(done) or "(none)")
            + f"\n\nFAILED\n{failed}\n" + (f"WHY (from the run):\n{notes}\n"
                                           if notes else "")
            + "\nREMAINING (as planned)\n" + ("\n".join(left) or "(none)")
            + "\n\nWrite the revised remaining steps now.")
    from . import roles
    try:
        raw = roles.client_for(cfg, "planner", client).chat_json(
            [{"role": "system", "content": REPLAN_SYSTEM},
             {"role": "user", "content": user}],
            loop.TASK_SCHEMA, max_tokens=getattr(cfg, "max_tokens_plan", 2000),
            disable_thinking=True)
    except Exception as e:
        print(f"  [re-plan failed: {str(e)[:100]}]", file=sys.stderr)
        return []
    titles = loop._titles_from(raw)
    return [t for t in titles if t.strip() != failed.strip()] or titles


def run_planned(root: Path, cfg, client, idx, task: str, *, apply: bool,
                review: bool = False, confirm=None) -> int:
    """Plan `task`, show the plan, run it step by step (with one re-plan after
    a failure), and return an exit code. `confirm`, when given, is called with
    the plan path to let the user approve or edit it first (returns False to
    stop)."""
    titles, how = make_plan(task, idx, client, cfg, root)
    stamp = workspace.stamp()
    plan_path = Path(root) / ".pedacito" / "plans" / f"{stamp}.md"
    loop.write_plan(plan_path, task[:300], titles)
    print(f"\n--- plan ({how}; {len(titles)} step(s)) -> "
          f"{plan_path.relative_to(root)} ---", file=sys.stderr)
    for i, t in enumerate(titles, 1):
        print(f"  {i}. {t}", file=sys.stderr)
    workspace.log_event(root, f"plan mode: {len(titles)} step(s) via {how}")
    if not apply:
        print("\nDry run: the steps above would each run in a fresh context "
              "(add --apply to run them).", file=sys.stderr)
        return 0
    if confirm is not None and not confirm(plan_path):
        print("Plan not approved; nothing was run.", file=sys.stderr)
        return 1
    replans = max(0, int(getattr(cfg, "max_replans", 1) or 0))
    goal = task
    code = 1
    for attempt in range(replans + 1):
        code = loop.run_loop(root, cfg, client, idx, plan_path, apply=True,
                             stop_on_fail=True, goal=goal,
                             repair_rounds=cfg.auto_repair_rounds)
        remaining = sum(1 for ln in loop.read_plan(plan_path)
                        if loop._is_unchecked(ln))
        if code == 0 or not remaining or attempt >= replans:
            break
        failed = _last_failed(plan_path)
        print(f"\n[step failed -- asking the planner to revise the remaining "
              f"{remaining} step(s)]", file=sys.stderr)
        revised = replan(task, plan_path, client, cfg, failed)
        if not revised:
            break
        loop.replace_unchecked(plan_path, revised)
        for i, t in enumerate(revised, 1):
            print(f"  revised {i}. {t}", file=sys.stderr)
        workspace.log_event(root, f"re-planned: {len(revised)} step(s)")
    return code


def _last_failed(plan_path: Path) -> str:
    for ln in reversed(plan_path.read_text(encoding="utf-8").splitlines()):
        s = ln.strip()
        if s.startswith("- [!]") or s.startswith("- [-]"):
            return s[5:].strip()
    return "(the previous step)"

