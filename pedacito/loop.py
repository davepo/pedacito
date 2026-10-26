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
from dataclasses import dataclass, replace
from pathlib import Path

from . import index as index_mod
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
        "leave as", "as-is", "already exists", "already correct",
    ))


# Tasks that are checkpoints rather than edits. The planner explicitly emits
# "reproduce the failure" as a step for bugs, but the loop used to demand an
# applied edit from every task -- so a successful reproduction was recorded as
# a failure and `stop_on_fail` ended the run before the fix task. These titles
# are recognised as investigation-only and answered in prose.
_INVESTIGATE_START = (
    "reproduce", "investigate", "diagnose", "analyze", "analyse", "inspect",
    "trace", "locate", "find", "confirm", "check", "verify", "review",
    "profile", "explain", "identify", "determine", "observe", "test",
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
                    "title": {"type": "string", "maxLength": 300},
                    "acceptance": {"type": "string", "maxLength": 300},
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
- Prefer 3-8 tasks for a small feature; never enumerate many tiny stubs.
- Do not include meta-tasks (planning, documentation-only chores, git work).
- If the goal is a BUG or a broken program (a traceback, a wrong result), do
  not return an empty list: produce concrete repair steps -- reproduce the
  failure, locate the cause, fix it, then verify. "Fix the bug" is a valid
  task title when that is genuinely all there is to do.
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


def generate_tasks(spec_text: str, client, cfg) -> list[str]:
    """Ask the model to break `spec_text` into an ordered task list.

    Retries once with an explicit "you returned zero tasks" nudge when the
    first reply comes back empty (a model handed a bug-fix goal sometimes
    reads it as "nothing to build"), and finally falls back to one task built
    from the goal itself rather than aborting the whole run."""
    budget = getattr(cfg, "max_spec_chars", 60_000)
    body = spec_text.strip()[:budget]
    max_tokens = getattr(cfg, "max_tokens_plan", 2000)
    base_user = (f"GOAL\n\n{body}\n\n"
                 "Break this into an ordered task list. Each task title must be "
                 "a single imperative sentence a coding agent can implement and "
                 "verify in isolation.")
    raw = client.chat_json(
        [{"role": "system", "content": PLAN_SYSTEM},
         {"role": "user", "content": base_user}],
        TASK_SCHEMA, max_tokens=max_tokens, disable_thinking=True)
    titles = _titles_from(raw)
    if not titles:
        nudge = (base_user + "\n\nYou returned ZERO tasks. Every goal has at "
                 "least one concrete step. For a bug, list the steps to "
                 "reproduce, locate, fix, and verify it. Return the tasks now.")
        raw = client.chat_json(
            [{"role": "system", "content": PLAN_SYSTEM},
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
                              text=True, timeout=600)
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
    writes landed, the files re-indexed, and the (possibly refreshed) index."""
    results: list
    reply: str
    applied: bool
    changed: list[str]
    idx: "index_mod.Index"
    session: Path | None = None


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
_READ_TOOLCALL_RE = re.compile(
    r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
_FILEPATH_PARAM_RE = re.compile(
    r"<parameter\b[^>]*name\s*=\s*['\"]file_path['\"][^>]*>\s*(.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE)
_FILEPATH_EQ_PARAM_RE = re.compile(
    r"<parameter\s*=\s*file_path>\s*(.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE)

_MAX_RECOVERY_ROUNDS = 2
_MAX_RECOVERY_FILES = 4
_MAX_FILE_CHARS = 20000
_MAX_EXTRA_CHARS = 30000


def _read_requests(reply: str) -> list[str]:
    """Extract the file paths a reply asked to `read` via tool-call syntax,
    in reply order and de-duplicated. Empty when the reply has none."""
    found: list[tuple[int, str]] = []
    for m in _READ_INVOKE_RE.finditer(reply):
        p = _FILEPATH_PARAM_RE.search(m.group(1))
        if p:
            found.append((m.start(), p.group(1).strip()))
    for m in _READ_TOOLCALL_RE.finditer(reply):
        body = m.group(1)
        if not re.search(r"function\s*=\s*read\b", body, re.IGNORECASE):
            continue
        p = _FILEPATH_EQ_PARAM_RE.search(body)
        if p:
            found.append((m.start(), p.group(1).strip()))
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
                 + (" (it only emitted tool-call requests, which cannot run "
                    "in this phase)" if extra else "")
                 + ". Produce the edit blocks now.")
        src = (gathered or "").strip()
        if extra:
            src = src + ("\n\n" if src else "") + \
                "EXTRA SOURCE READ FOR YOU\n\n" + extra
        raw = agent.answer(task2, src, system=system)
        results2, reply2 = agent.apply(raw, dry_run=dry_run, task=task,
                                       gathered=src, backup_dir=backup_dir)
        if results2:
            return results2, reply2
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
fix someone should make. Do not call tools -- none are available in this phase.
Do not emit search/replace blocks. Be concrete and concise; if the evidence is
inconclusive, say what is still unknown and what to check next.
"""

_TOOL_SYNTAX = ("<tool_call", "<invoke", "<function=", '"name":')


def _looks_like_tool_calls_only(reply: str) -> bool:
    """True when a prose answer is really just agent-framework tool syntax.

    Small models occasionally ignore "no tools here" and emit `<tool_call>` /
    `<invoke>` blocks in the answer phase. For an investigation task that is
    not an answer, so the caller substitutes the gathered evidence."""
    low = (reply or "").lower()
    if not low.strip() or not any(t in low for t in _TOOL_SYNTAX):
        return False
    plain = re.sub(r"<[^>]+>", " ", reply)
    plain = re.sub(r"[{}\[\]\"]", " ", plain)
    return len(re.findall(r"[A-Za-z]{3,}", plain)) < 8


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
    reply = agent.answer(task, gathered,
                         system=answer_system or INVESTIGATE_SYSTEM,
                         include_edit_format=False, stream=True)
    if _looks_like_tool_calls_only(reply) or not (reply or "").strip():
        steps = "\n".join(getattr(agent, "steps_log", []) or ["(no steps)"])
        reply = ("No prose report was produced; the lookup phase recorded:\n\n"
                 + steps)
        evidence = (gathered or "").strip()
        if evidence:
            reply += "\n\nEVIDENCE\n" + evidence[:8000]
    session = None
    if cfg.log_sessions:
        session = workspace.write_session(root, task,
                                          getattr(agent, "steps_log", []),
                                          reply, [], None, when)
        print(f"[session: {session.relative_to(root)}]", file=sys.stderr)
    return TaskResult([], reply, True, [], idx, session)


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
    gathered = agent.gather(task)
    print("--- answering ---\n", file=sys.stderr)
    raw_reply = agent.answer(task, gathered, system=answer_system)
    truncated = bool(getattr(client, "stream_truncated", False))
    if truncated:
        workspace.log_event(root, "answer reply truncated at the output limit",
                            "WARN")
    results, reply = agent.apply(raw_reply, dry_run=review or not apply,
                                 task=task, gathered=gathered,
                                 backup_dir=None if review else backup_dir)
    if not results and not _is_noop(reply):
        # The answer phase produced zero edit blocks. Before declaring the
        # task dead, give the model one bounded recovery: honour any read
        # tool-calls it emitted, or re-ask with the edit protocol spelled
        # out -- both without touching the no-op path ("already complete"
        # replies must still mark the task done, not re-ask).
        rec = _recover_no_edit_reply(
            root, cfg, agent, task, gathered, reply,
            dry_run=review or not apply,
            backup_dir=None if review else backup_dir,
            answer_system=answer_system)
        if rec is not None:
            results, reply = rec
            workspace.log_event(root, "answer re-asked; recovered edit blocks",
                                "INFO")
    if not results:
        workspace.log_event(root, "no edit blocks parsed from the reply", "ERROR")
    else:
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
    return TaskResult(results, reply, applied, changed, idx, session)


# --------------------------------------------------------------------------
# The outer loop itself
# --------------------------------------------------------------------------
def run_loop(root: Path, cfg, client, idx, plan_path: Path, *,
             apply: bool = True, verify: str = "", repair_rounds: int = 2,
             max_tasks: int = 0, stop_on_fail: bool = True,
             answer_system: str | None = None, auto_install: bool = False,
             on_task=None) -> int:
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
    - ``stop_on_fail``  True = halt on the first task that produces no applied
                        edit or still fails verification (the safe default);
                        False = log it and carry on to the next task.
    - ``answer_system`` replacement answer brief (used when the tasks implement
                        scaffold stubs rather than edit existing code).
    - Investigation-only tasks ("Reproduce...", "Locate the cause...") are
      answered in prose and marked done without requiring edits.
    - ``auto_install`` when a verification fails on a missing third-party
                        module, pip-install it and retry (opt-in: the module
                        name comes from generated code). When off, the failure
                        is reported as a dependency problem and the code-repair
                        loop is skipped, since editing source cannot add a dep.
    """
    from .agent import Agent

    attempted = 0
    done = 0
    failed = 0
    # Investigation tasks produce a report, not an edit. Carry the latest
    # reports into subsequent task prompts so a fresh context still benefits
    # from the reproduction -- the plan file itself stays untouched.
    context_notes: list[str] = []
    workspace.log_event(root, f"auto loop: plan={plan_path.name}"
                              f" verify={verify or '(none)'}")

    def tick(count: int) -> bool:
        """Return False once max_tasks (if any) is reached."""
        return max_tasks == 0 or count < max_tasks

    while tick(attempted):
        lines = read_plan(plan_path)
        nxt = next_task(lines)
        if nxt is None:
            print("\nAll tasks complete.", file=sys.stderr)
            workspace.log_event(root, "all tasks complete")
            break
        _, task = nxt
        attempted += 1
        print(f"\n=== task {attempted}: {task} ===", file=sys.stderr)

        investigate = _is_investigation_task(task)
        task_for_agent = task
        if context_notes:
            task_for_agent = (
                task + "\n\nCONTEXT FROM PRIOR INVESTIGATION\n\n"
                + "\n\n".join(context_notes))
        try:
            if investigate:
                tr = run_investigation_task(
                    root, cfg, client, Agent(idx, client, cfg), idx,
                    task_for_agent, answer_system=answer_system)
            else:
                tr = run_task(root, cfg, client, Agent(idx, client, cfg), idx,
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
                context_notes.append(f"### {task}\n{(tr.reply or '')[:3000]}")
                del context_notes[:-2]       # keep only the latest two
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
            if _is_noop(tr.reply):
                print("  [model reports the task is already complete; "
                      "marking it done]", file=sys.stderr)
                workspace.log_event(root, "task already complete (no-op)")
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

        mark_done(plan_path, task)
        done += 1
        workspace.log_event(root, f"task done ({done})")
        if on_task is not None:
            try:
                on_task(task)
            except Exception:
                pass

    print(f"\n[{done} task(s) done, {failed} failed]", file=sys.stderr)
    workspace.log_event(root, f"auto loop finished: {done} done, {failed} failed",
                        "ERROR" if failed else "INFO")
    return 0 if failed == 0 else 1
