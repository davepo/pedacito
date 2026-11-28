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
    that name that many files or hold that many requirements; `--plan` forces
    it. A WIDE task (`WIDE_TASK_FILES` or more named files or requirements) is
    planned even when auto-planning is off -- unless the config explicitly set
    `plan_threshold`; `--no-plan` still opts one task out.

Before an Auto or Complex run executes a checklist, `check_plan` validates the
paths and symbols the plan names against the static index, checks dependency
order, and previews the conservative impact (callers and tests). `pedacito plan
--check` exposes the same inspection without running a model or writing files:

    pdc plan --check PLAN.md . --impact

Warnings (a missing unmarked path, an approximate caller, a test with no naming
link) do not block. Only definite errors (containment/security, a dependency
cycle or a later-task dependency) stop execution; an unreadable or escaping
plan path is invalid and exits 2.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from . import fileio, loop, rank, safety, security, testlinks, verify, workspace
from .select import CODE_EXTS, REFERENCE_EXTS


def _sparse_moe(cfg) -> bool:
    """True for a sparse Mixture-of-Experts profile (the same predicate
    prompts.py re-exports from `config`)."""
    from .prompts import is_sparse_moe
    try:
        return bool(is_sparse_moe(cfg))
    except Exception:
        return False


# A task this wide is planned even with auto-planning off (`plan_threshold` 0,
# the default and what a strong model's card picks): one context window cannot
# hold the lookups for that many files, and "did the first, half-did the
# second, forgot the third" is what plan mode exists to prevent -- the
# deepclitest runs kept losing the tail of exactly these tasks. Width is a
# capacity problem, not a skill problem, so the card's verdict does not skip
# it; an explicit `plan_threshold` in the config (any value, 0 included) still
# governs.
WIDE_TASK_FILES = 5


def requirement_titles(task: str, limit: int = 16) -> list[str]:
    """One step per clearly separate requirement, '' when it is one change.

    A leading goal-summary clause with no verb of its own ("Bring the
    feature to a working, tested state") is not a step: as step 1 it is
    free to implement the whole plan, and every later step becomes a no-op
    (BUG-312). Drop it when concrete clauses follow; the full goal still
    rides along as background context for every step.

    The cap matches the requirement ledger's (BUG-203/205/212): a six-step
    ceiling truncated a five-file repair to its head clauses and marked the
    dropped tail done. Sixteen slots hold a realistic multi-file brief."""
    # One extra slot: the umbrella clause must not eat a concrete step's
    # place under the cap (the src/main.js step lost to it in BUG-312).
    reqs = verify.requirement_ledger(task, limit=limit + 1)
    if len(reqs) >= 2:
        first = reqs[0][0]
        # A list introduction ("Use these exact contracts:") is an umbrella
        # even when it carries an edit verb ("Repair ... contracts:"): it
        # names no file of its own and the concrete clauses follow it
        # (BUG-205). Drop it like the verb-less goal summary above.
        if first.rstrip().endswith(":") or not (
                verify._REQ_EDIT_RE.search(first)
                or verify._REQ_REPORT_RE.search(first)):
            reqs = reqs[1:]
    if len(reqs) < 2:
        return []
    return [text for text, _terms in reqs][:limit]


def _is_refined_spec(task: str) -> bool:
    """True for the sectioned spec produced by `--refine`.

    That text is one task already: its OBJECTIVE / FILES / CONTRACTS /
    REQUIREMENTS headings are structure, not separate plan steps. Mining
    them with requirement_titles produced steps literally named "OBJECTIVE"
    and "INTERFACE CONTRACTS", then ran each one against the same file.
    """
    text = (task or "").lstrip()
    return (text.startswith("OBJECTIVE\n")
            and "\nREQUIREMENTS\n" in text
            and "\nVERIFICATION\n" in text)


def should_plan(task: str, idx, cfg) -> bool:
    """Whether to auto-plan: enough files named, or enough requirements.
    With auto-planning off, a wide-enough task is still planned unless the
    user explicitly set `plan_threshold` (the model card's 0 for a strong
    model does not count as an opt-out -- it measured skill on small probes,
    not window capacity on a five-file task)."""
    if _is_refined_spec(task):
        return False
    n = int(getattr(cfg, "plan_threshold", 0) or 0)
    try:
        named = len(rank.task_files(idx, task))
    except Exception:
        named = 0
    reqs = len(requirement_titles(task, limit=8))
    if n > 0:
        if _sparse_moe(cfg):
            # A sparse-MoE model executes directly: ignore the small
            # plan_threshold split and only pre-split truly multi-module
            # (multi-file) tasks -- a capacity decision, not a skill one.
            return named >= WIDE_TASK_FILES
        return named >= n or reqs >= n
    sources = getattr(cfg, "sources", None) or {}
    if sources.get("plan_threshold", "default") not in ("default",
                                                        "model card"):
        return False
    return named >= WIDE_TASK_FILES or reqs >= WIDE_TASK_FILES


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
    specialty_note = roles.specialty_planner_note(cfg)
    if specialty_note:
        user += "\n\n" + specialty_note
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
                review: bool = False, confirm=None,
                strict: bool = False, on_plan=None) -> int:
    """Plan `task`, show the plan, run it step by step (with one re-plan after
    a failure), and return an exit code. `confirm`, when given, is called with
    the plan path to let the user approve or edit it first (returns False to
    stop)."""
    titles, how = make_plan(task, idx, client, cfg, root)
    stamp = workspace.stamp()
    plan_path = Path(root) / ".pedacito" / "plans" / f"{stamp}.md"
    loop.write_plan(plan_path, task, titles)
    loop.emit_plan_updated(plan_path, root=root, cfg=cfg, reason="created")
    if on_plan is not None:
        try:
            on_plan(plan_path)
        except Exception:
            pass
    print(f"\n--- plan ({how}; {len(titles)} step(s)) -> "
          f"{plan_path.relative_to(root)} ---", file=sys.stderr)
    for i, t in enumerate(titles, 1):
        # Display-only clip: the plan file and the step's task text keep the
        # FULL clause (its tail is the requirement, BUG-328).
        print(f"  {i}. {verify._clip_clause(t)}", file=sys.stderr)
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
                             repair_rounds=cfg.auto_repair_rounds,
                             strict=strict)
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


# ===========================================================================
# Plan validation and impact (WP-4104C)
# ===========================================================================
#
# The validation is deliberately conservative in both directions:
#
#   * a missing file is a definite error only when the task reads like an edit
#     of it; a bullet that names a creation ("Create `src/new.py`", "Implement
#     `main.py`") is a valid plan, not a missing-file failure;
#   * symbols are resolved against the index, never imported or executed;
#   * a caller/test list built from static import and call edges is labelled
#     unknown, not complete -- dynamic imports, reflection and tests with
#     unrelated names are outside what the index can prove.

ERROR = "error"
WARNING = "warning"
INFO = "info"
_INVALID_CODES = ("invalid_root", "invalid_plan", "invalid_plan_path")

_MAX_READ_BYTES = 4 * 1024 * 1024
_MAX_IMPACT_ITEMS = 200
_MAX_UNKNOWNS = 64
_MAX_FIELD = 240
_MAX_MESSAGE = 500

_TASK_LINE_RE = re.compile(r"^(\s*[-*]\s*)\[([ xX!\-])\]\s*(.*)$")
_FILE_TOKEN_RE = re.compile(
    r"(?:/|\.{1,2}[\\/]|\.?[\w@])[\w@./\\\-]*")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_BACKTICK_RE = re.compile(r"`([^`\n]{1,160})`")
_FILE_EXTS = {str(e).lower() for e in (set(CODE_EXTS) | set(REFERENCE_EXTS))}
_EXTENSIONLESS_FILES = {
    "dockerfile", "containerfile", "makefile", "rakefile", "gemfile",
    "procfile", "brewfile", "justfile", "vagrantfile", "license",
    "licence", "changelog", "readme",
}
_CREATION_VERB_RE = re.compile(
    r"\b(?:create|creating|add|adding|new|scaffold|scaffolding|generate|"
    r"generating|introduce|introducing|write|writing|implement|"
    r"implementing|build|building|design|designing|extract|extracting|"
    r"move|moving|rename|renaming|split|splitting|port|porting)\b",
    re.IGNORECASE)
_EDIT_VERB_RE = re.compile(
    r"\b(?:update|updating|modify|modifying|edit|editing|fix|fixing|change|"
    r"changing|refactor|refactoring|remove|removing|delete|deleting|replace|"
    r"replacing|adjust|adjusting|repair|repairing|correct|correcting|patch|"
    r"patching|bump|bumping|upgrade|upgrading|rename|renaming|move|moving)\b",
    re.IGNORECASE)
_MOVE_TARGET_RE = re.compile(r"\b(?:into|to)\s+[`\"']?$", re.IGNORECASE)
_OBJECT_TARGET_RE = re.compile(
    r"\b(?:create|add|write|implement|generate|build)\b[^.;]{1,70}"
    r"\b(?:to|in|into|for)\s+[`\"']?$",
    re.IGNORECASE)
_TEST_REFERENCE_RE = re.compile(
    r"\b(?:add|adding|write|writing|create|creating|generate|generating|"
    r"implement|implementing)\b[^.;]{0,50}\btests?\b[^.;]{0,30}"
    r"\b(?:for|on|of|to)\s*$",
    re.IGNORECASE)
_SYMBOL_LIST = (r"[A-Za-z_][A-Za-z0-9_]*"
                r"(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
                r"(?:(?:\s*,\s*|\s+and\s+)"
                r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)*")
_SYMBOL_AFTER_COLON_RE = re.compile(r"\s*:\s*(" + _SYMBOL_LIST + r")")
_SYMBOL_SPLIT_RE = re.compile(r"\s*,\s*|\s+and\s+")
_SYMBOL_STOP = {
    "the", "a", "an", "this", "that", "these", "those", "it", "its", "and",
    "or", "in", "on", "to", "for", "with", "from", "by", "as", "is", "are",
    "be", "file", "files", "function", "functions", "method", "methods",
}
_DEP_PAREN_RE = re.compile(
    r"[\(\[]\s*(?:depends?(?:\s+on)?|depends_on|deps?)"
    r"\s*[:=]?\s*(?:tasks?\s*)?#?\s*([0-9][0-9,\s&]*)", re.IGNORECASE)
_DEP_KEY_RE = re.compile(
    r"\bdepends_on\s*[:=]\s*(?:tasks?\s*)?#?\s*([0-9][0-9,\s&]*)",
    re.IGNORECASE)


@dataclass(frozen=True)
class PlanIssue:
    """One plan-check finding. `task_index` is 1-based; 0 means plan-level."""

    code: str
    severity: str
    task_index: int
    file: str
    symbol: str
    message: str


@dataclass(frozen=True)
class PlanImpact:
    """Conservative preview of what a plan can touch, from the static index."""

    files: tuple[str, ...]
    symbols: tuple[str, ...]
    callers: tuple[str, ...]
    tests: tuple[str, ...]
    unknowns: tuple[str, ...]


@dataclass(frozen=True)
class _Mention:
    """One file path written in a task title.

    `rel` keeps the plan's spelling (so a case-only Windows collision stays
    visible); `resolved` is the canonical contained path the filesystem/index
    reports for it.
    """

    raw: str
    rel: str
    resolved: str
    start: int
    end: int
    symbols: tuple[str, ...]
    creation: bool
    edit_like: bool


def _bounded(value: str, limit: int = _MAX_FIELD) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def _issue(code: str, severity: str, task_index: int, *, file: str = "",
           symbol: str = "", message: str = "") -> PlanIssue:
    return PlanIssue(
        code=code,
        severity=severity,
        task_index=int(task_index or 0),
        file=_bounded(file),
        symbol=_bounded(symbol),
        message=_bounded(message, _MAX_MESSAGE),
    )


def _empty_impact() -> PlanImpact:
    return PlanImpact((), (), (), (), ())


def _index_files(index) -> set[str]:
    raw = getattr(index, "files", None)
    if raw is None:
        raw = getattr(index, "_by_file", None)
    files: set[str] = set()
    try:
        values = raw.keys() if isinstance(raw, dict) else raw
        for value in values or ():
            text = str(value).replace("\\", "/").strip()
            if text:
                files.add(text)
    except Exception:
        return set()
    return files


def _index_chunks(index) -> list:
    raw = getattr(index, "chunks", None)
    try:
        return list(raw or ())
    except Exception:
        return []


def _index_available(index) -> bool:
    return bool(_index_files(index) or _index_chunks(index))


def _is_windows() -> bool:
    return os.name == "nt" or sys.platform.startswith("win")


def _norm_rel(raw: str) -> str:
    text = str(raw or "").replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text


def _looks_codeish(name: str) -> bool:
    """A symbol spelling that a plan author probably means literally."""
    return "_" in name or "." in name or any(ch.isupper() for ch in name)


def _looks_like_file_token(raw: str) -> bool:
    text = _norm_rel(raw)
    if not text or "://" in text:
        return False
    if text.startswith(("/", "../")) or re.match(r"^[A-Za-z]:/", text):
        return True
    if text.startswith((".git/", ".pedacito/")):
        return True
    if security.is_sensitive_path(text):
        return True
    suffix = Path(text).suffix.lower()
    if suffix in _FILE_EXTS:
        return True
    name = text.rsplit("/", 1)[-1].lower()
    if (name.startswith(".") and len(name) > 1
            and any(ch.isalpha() or ch == "_" for ch in name[1:])):
        return True
    return "/" not in text and name in _EXTENSIONLESS_FILES


def _trim_mention(title: str, match) -> tuple[str, int, int]:
    raw = match.group(0)
    start, end = match.start(), match.end()
    while raw and raw[-1] in ".,:;!?)]}\"'`>":
        raw = raw[:-1]
        end -= 1
    while raw and raw[0] in "([{<\"'`":
        raw = raw[1:]
        start += 1
    while end < len(title) and title[end] in "`\"'":
        end += 1
    return raw, start, end


def _symbols_after(title: str, end: int) -> tuple[str, ...]:
    match = _SYMBOL_AFTER_COLON_RE.match(title[end:end + 500])
    if match is None:
        return ()
    parts = [p.strip() for p in _SYMBOL_SPLIT_RE.split(match.group(1))]
    return tuple(p for p in parts if p)


def _creation_marked(title: str, start: int, end: int) -> bool:
    """True when this path is introduced by a creation verb or marker."""
    before = title[max(0, start - 90):start]
    after = title[end:end + 60]
    target = _norm_rel(title[start:end].strip("`\"'"))
    try:
        if testlinks.classify(target)[0]:
            return True
    except Exception:
        pass
    if _TEST_REFERENCE_RE.search(before):
        return False
    if _OBJECT_TARGET_RE.search(before):
        return False
    if re.search(r"[\(\[]\s*(?:new|created?|creation|create)"
                 r"(?:\s+file)?\s*[\)\]]", after, re.IGNORECASE):
        return True
    if re.search(r"\b(?:new|created?|creation)\s+file\b", after,
                 re.IGNORECASE):
        return True
    if _MOVE_TARGET_RE.search(before):
        return True
    return bool(_CREATION_VERB_RE.search(before))


def _edit_marker(title: str, start: int) -> bool:
    before = title[max(0, start - 90):start]
    return bool(_EDIT_VERB_RE.search(before))


def _mentions_in(root: Path, title: str
                 ) -> tuple[list[_Mention], list[PlanIssue]]:
    """Every contained file mention in one task title."""
    mentions: list[_Mention] = []
    escapes: list[PlanIssue] = []
    seen: set[tuple] = set()
    for match in _FILE_TOKEN_RE.finditer(title or ""):
        raw, start, end = _trim_mention(title, match)
        if not _looks_like_file_token(raw):
            continue
        clean = raw.replace("\\", "/")
        if clean.lower().startswith(("http:", "https:", "file:")):
            continue
        if re.search(r"[A-Za-z][A-Za-z0-9+.-]*://\S*$",
                     title[max(0, start - 300):start]):
            continue
        if re.match(r"^[A-Za-z]:", clean) and not _is_windows():
            escapes.append(_issue(
                "path_escape", ERROR, 0, file=raw,
                message=f"the plan names a Windows absolute path outside a "
                        f"POSIX project: {raw}"))
            continue
        try:
            resolved = security.resolve_safe_path(root, clean)
        except security.SecurityError as exc:
            escapes.append(_issue(
                "path_escape", ERROR, 0, file=raw,
                message=f"the plan names a path that is not contained in the "
                        f"project: {exc}"))
            continue
        except OSError:
            continue
        try:
            resolved_rel = resolved.relative_to(root).as_posix()
        except ValueError:
            escapes.append(_issue(
                "path_escape", ERROR, 0, file=raw,
                message=f"the plan names a path outside the project: {raw}"))
            continue
        if not resolved_rel or resolved_rel == ".":
            continue
        rel = _norm_rel(clean)
        if not rel or rel == ".":
            rel = resolved_rel
        if Path(clean).is_absolute():
            rel = resolved_rel
        symbols = _symbols_after(title, end)
        creation = _creation_marked(title, start, end)
        key = (rel.casefold(), symbols, creation)
        if key in seen:
            continue
        seen.add(key)
        mentions.append(_Mention(
            raw=raw, rel=rel, resolved=resolved_rel, start=start, end=end,
            symbols=symbols, creation=creation,
            edit_like=_edit_marker(title, start),
        ))
    return mentions, escapes


def _case_collision(root: Path, rel: str, index_files: set[str]) -> str:
    """The existing path that differs from `rel` only by case, or ''."""
    folded = rel.casefold()
    for known in sorted(index_files):
        if known != rel and known.casefold() == folded:
            return known
    path = Path(rel)
    parent = root / path.parent
    try:
        entries = list(parent.iterdir())
    except OSError:
        return ""
    for entry in sorted(entries, key=lambda p: p.name):
        if entry.name != path.name and entry.name.casefold() == path.name.casefold():
            try:
                return entry.relative_to(root).as_posix()
            except ValueError:
                return entry.name
    return ""


def _exact_name_exists(root: Path, rel: str) -> bool:
    path = root / rel
    try:
        if not path.exists():
            return False
        return path.name in {p.name for p in path.parent.iterdir()}
    except OSError:
        return False


def _creation_parent_issues(root: Path, rel: str, task_index: int
                            ) -> list[PlanIssue]:
    issues: list[PlanIssue] = []
    parent_rel = Path(rel).parent.as_posix()
    try:
        security.resolve_safe_path(root, parent_rel or ".")
    except security.SecurityError as exc:
        issues.append(_issue(
            "creation_parent_escape", ERROR, task_index, file=rel,
            message=f"the planned creation's parent escapes the project: {exc}"))
        return issues
    probe = root
    for part in Path(rel).parts[:-1]:
        probe = probe / part
        try:
            if probe.exists() and not probe.is_dir():
                issues.append(_issue(
                    "creation_parent_not_directory", ERROR, task_index,
                    file=rel,
                    message=f"cannot create {rel}: {part} is a file, not a "
                            f"directory"))
                return issues
        except OSError:
            continue
    return issues


def _dynamic_symbols_possible(root: Path, rel: str) -> bool:
    """True when a file can provide symbols the static index cannot see.

    `__getattr__`, `setattr`, `exec`/`eval`, `globals()` and `importlib` can
    create callables at runtime. A missing symbol in such a file is unknown,
    never a definite failure; an unreadable file is also treated as unknown.
    """
    if not rel.endswith(".py"):
        return True
    try:
        text = fileio.read_text(root / rel, errors="replace")[:200_000]
    except OSError:
        return True
    return bool(re.search(
        r"__getattr__|__getattribute__|\bsetattr\s*\(|\bexec\s*\(|"
        r"\beval\s*\(|\bglobals\s*\(|\bimportlib\b|"
        r"\b(?:make_dataclass|namedtuple)\s*\(|"
        r"\bfrom\s+\S+\s+import\s+\*", text))


def _chunk_matches(chunk, symbol: str) -> bool:
    qual = str(getattr(chunk, "qualname", "") or "").split("#")[0]
    return (qual == symbol or qual.endswith("." + symbol)
            or symbol in (getattr(chunk, "members", None) or ()))


def _symbol_hits(index, symbol: str) -> list:
    finder = getattr(index, "find_symbol", None)
    if callable(finder):
        try:
            hits = list(finder(symbol) or ())
        except Exception:
            hits = []
    else:
        hits = [c for c in _index_chunks(index) if _chunk_matches(c, symbol)]
    return hits


def _symbol_hits_in_file(index, rel: str, symbol: str) -> list:
    return [c for c in _symbol_hits(index, symbol)
            if _norm_rel(getattr(c, "file", "")) == rel]


def _callers_for(index, rel: str, symbol: str) -> tuple[list, list]:
    ranked = getattr(index, "callers_ranked", None)
    if callable(ranked):
        try:
            sure, approx = ranked(symbol, rel)
            return list(sure or ()), list(approx or ())
        except Exception:
            return [], []
    callers = getattr(index, "callers_of", None)
    if callable(callers):
        try:
            return list(callers(symbol) or ()), []
        except Exception:
            return [], []
    return [], []


def _check_symbol(index, index_files: set[str], rel: str, symbol: str, *,
                  task_index: int, explicit: bool, creation: bool,
                  issues: list[PlanIssue], symbols: set[str],
                  symbol_refs: set[tuple[str, str]], unknowns: set[str],
                  dynamic: bool = False) -> None:
    if _symbol_hits_in_file(index, rel, symbol):
        symbols.add(symbol)
        symbol_refs.add((rel, symbol))
        return
    if not _index_available(index) or rel not in index_files:
        unknowns.add(f"{symbol} in {rel} is not indexed: symbol resolution "
                     f"is unverified")
        return
    elsewhere = sorted({_norm_rel(getattr(c, "file", ""))
                        for c in _symbol_hits(index, symbol)
                        if _norm_rel(getattr(c, "file", ""))})
    if elsewhere:
        issues.append(_issue(
            "symbol_other_file", WARNING, task_index, file=rel, symbol=symbol,
            message=f"{symbol} is not defined in {rel}; the index finds it in "
                    f"{', '.join(elsewhere[:3])}"))
        unknowns.add(f"{symbol} may be re-exported or dynamically provided by "
                     f"{rel}")
        return
    # A new symbol in an existing file is a normal edit when the task says it
    # creates it. Without that signal, an explicit `file: symbol` reference in
    # Python is a definite planning error; every other language stays a
    # warning because its symbol extraction is approximate.
    constant_like = symbol.replace("_", "").isupper()
    definite = bool(explicit and not creation and rel.endswith(".py")
                    and not dynamic and not constant_like)
    severity = ERROR if definite else WARNING
    suffix = "" if definite else " (dynamic or generated symbols are unverified)"
    issues.append(_issue(
        "missing_symbol", severity, task_index, file=rel, symbol=symbol,
        message=f"{symbol} is not defined in the indexed file {rel}" + suffix))
    if not definite:
        unknowns.add(f"{symbol} in {rel} is unresolved: dynamic or generated "
                     f"symbols are unverified")


def _dependency_numbers(text: str) -> set[int]:
    numbers: set[int] = set()
    for match in _DEP_PAREN_RE.finditer(text or ""):
        for part in re.split(r"[\s,&]+", match.group(1)):
            try:
                numbers.add(int(part))
            except (TypeError, ValueError):
                continue
    match = _DEP_KEY_RE.search(text or "")
    if match is not None:
        for part in re.split(r"[\s,&]+", match.group(1)):
            try:
                numbers.add(int(part))
            except (TypeError, ValueError):
                continue
    return numbers


def _task_rows(text: str) -> list[tuple[int, str, str]]:
    rows: list[tuple[int, str, str]] = []
    for line in (text or "").splitlines():
        match = _TASK_LINE_RE.match(line)
        if match is not None:
            rows.append((len(rows) + 1, match.group(2), match.group(3).strip()))
    return rows


def _dependency_cycles(deps: dict[int, tuple[int, ...]]
                       ) -> list[tuple[int, ...]]:
    color: dict[int, int] = {}
    stack: list[int] = []
    found: set[tuple[int, ...]] = set()

    def visit(node: int) -> None:
        color[node] = 1
        stack.append(node)
        for dep in deps.get(node, ()):
            if dep not in deps:
                continue
            if color.get(dep, 0) == 1:
                start = stack.index(dep)
                found.add(tuple(sorted(stack[start:])))
            elif color.get(dep, 0) == 0:
                visit(dep)
        stack.pop()
        color[node] = 2

    for node in sorted(deps):
        if color.get(node, 0) == 0:
            visit(node)
    return sorted(found, key=lambda c: (c[0], len(c), c))


def _structured_errors(root: Path, text: str
                       ) -> tuple[list[PlanIssue],
                                  list[tuple[int, str, list[_Mention]]],
                                  list[int]]:
    """Dependency findings plus the unchecked tasks' parsed mentions."""
    issues: list[PlanIssue] = []
    tasks = _task_rows(text)
    if not tasks:
        issues.append(_issue(
            "invalid_plan", ERROR, 0,
            message="no `- [ ] task` checklist lines were found; this is not "
                    "an executable plan"))
        return issues, [], []

    deps: dict[int, tuple[int, ...]] = {}
    for number, _box, title in tasks:
        deps[number] = tuple(sorted(_dependency_numbers(title)))
    for number, _box, title in tasks:
        for dep in deps[number]:
            if dep < 1 or dep > len(tasks):
                issues.append(_issue(
                    "task_dependency_missing", ERROR, number,
                    message=f"task {number} depends on task {dep}, but the plan "
                            f"has {len(tasks)} task(s)"))
            elif dep != number and dep > number:
                issues.append(_issue(
                    "task_order", ERROR, number,
                    message=f"task {number} depends on later task {dep}; the "
                            f"loop executes top-down, so the dependency cannot "
                            f"be satisfied"))
    for cycle in _dependency_cycles(deps):
        label = ", ".join(str(n) for n in cycle)
        issues.append(_issue(
            "task_dependency_cycle", ERROR, min(cycle),
            message=f"dependency cycle between tasks {label}"))

    checked: list[int] = []
    mention_tasks: list[tuple[int, str, list[_Mention]]] = []
    for number, box, title in tasks:
        if box != " ":
            continue
        checked.append(number)
        got, escapes = _mentions_in(root, title)
        for issue in escapes:
            issues.append(PlanIssue(
                code=issue.code, severity=issue.severity,
                task_index=number, file=issue.file, symbol=issue.symbol,
                message=issue.message))
        mention_tasks.append((number, title, got))
    return issues, mention_tasks, checked


def _import_cycles(index, touched: set[str]) -> list[list[str]]:
    try:
        graph = getattr(index, "graph", None)
    except Exception:
        graph = None
    finder = getattr(graph, "import_cycles", None) if graph is not None else None
    if not callable(finder):
        return []
    try:
        cycles = finder()
    except Exception:
        return []
    out: list[list[str]] = []
    for cycle in cycles:
        members = sorted({_norm_rel(f) for f in cycle})
        if any(f in touched for f in members):
            out.append(members)
    return out


def _cap(values: set[str]) -> tuple[tuple[str, ...], bool]:
    ordered = sorted({_bounded(v) for v in values if _bounded(v)})
    return tuple(ordered[:_MAX_IMPACT_ITEMS]), len(ordered) > _MAX_IMPACT_ITEMS


def check_plan(root: Path, plan_path: Path, index
               ) -> tuple[list[PlanIssue], PlanImpact]:
    """Validate a checklist against `index` and preview its conservative impact.

    Warnings never block execution. A definite `error` (containment/security,
    a dependency cycle, a later-task dependency, or an explicit missing
    Python symbol) is what `gate_plan` stops on. An `invalid_*` finding means
    the plan itself could not be read or contained, and maps to exit code 2.
    """
    issues: list[PlanIssue] = []
    try:
        root_path = Path(root).expanduser().resolve()
    except OSError as exc:
        issues.append(_issue("invalid_root", ERROR, 0,
                             message=f"project root could not be resolved: {exc}"))
        return issues, _empty_impact()
    if not root_path.is_dir():
        issues.append(_issue("invalid_root", ERROR, 0,
                             message=f"project root is not a directory: "
                                     f"{root_path}"))
        return issues, _empty_impact()
    try:
        plan_file = security.resolve_safe_path(root_path, plan_path)
    except security.SecurityError as exc:
        issues.append(_issue(
            "invalid_plan_path", ERROR, 0, file=str(plan_path),
            message=f"the plan path is not contained in the project: {exc}"))
        return issues, _empty_impact()
    if not plan_file.is_file():
        issues.append(_issue(
            "invalid_plan_path", ERROR, 0,
            file=plan_file.relative_to(root_path).as_posix()
            if plan_file.is_relative_to(root_path) else str(plan_file),
            message=f"no plan file at {plan_file}"))
        return issues, _empty_impact()
    try:
        if plan_file.stat().st_size > _MAX_READ_BYTES:
            issues.append(_issue(
                "invalid_plan_path", ERROR, 0,
                file=plan_file.relative_to(root_path).as_posix(),
                message=f"plan file is larger than {_MAX_READ_BYTES} bytes"))
            return issues, _empty_impact()
        text, _style = fileio.read(plan_file, errors="strict")
    except (OSError, UnicodeDecodeError) as exc:
        issues.append(_issue(
            "invalid_plan_path", ERROR, 0,
            message=f"plan file could not be read: {exc}"))
        return issues, _empty_impact()

    index_files = _index_files(index)
    structural, mention_tasks, _checked = _structured_errors(root_path, text)
    issues.extend(structural)

    existing: set[str] = set()
    planned: set[str] = set()
    symbols: set[str] = set()
    symbol_refs: set[tuple[str, str]] = set()
    unknowns: set[str] = set()
    for number, title, got in mention_tasks:
        for mention in got:
            _check_mention(
                root_path, index, index_files, number, title, mention, issues,
                existing, planned, symbols, symbol_refs, unknowns)
        if len(got) == 1:
            rel = got[0].rel
            for sym, pos in _backtick_symbols(title, got):
                _check_symbol(
                    index, index_files, rel, sym, task_index=number,
                    explicit=False,
                    creation=got[0].creation or _creation_marked(
                        title, pos, pos),
                    issues=issues, symbols=symbols, symbol_refs=symbol_refs,
                    unknowns=unknowns,
                    dynamic=_dynamic_symbols_possible(root_path, rel))
        elif not got:
            # A bare backticked symbol (no file named) can still be resolved
            # against the index for the impact preview.
            for sym, _pos in _backtick_symbols(title, [], allow_plain=True):
                hits = _symbol_hits(index, sym)
                if hits:
                    symbols.add(sym)
                    for chunk in hits:
                        f = _norm_rel(getattr(chunk, "file", ""))
                        if f:
                            existing.add(f)
                            symbol_refs.add((f, sym))
                elif _index_available(index):
                    issues.append(_issue(
                        "missing_symbol", WARNING, number, symbol=sym,
                        message=f"{sym} is not defined in the index (dynamic "
                                f"or generated symbols are unverified)"))
                    unknowns.add(f"{sym} is unresolved: dynamic or generated "
                                 f"symbols are unverified")
                else:
                    unknowns.add(f"{sym} cannot be resolved without an index")

    for cycle in _import_cycles(index, existing | planned):
        issues.append(_issue(
            "import_cycle", WARNING, 0, file=cycle[0],
            message="these files import each other: " + " <-> ".join(cycle)))
        unknowns.add("an import cycle is legal in some languages but can hide "
                     "an order dependency")

    callers: set[str] = set()
    tests: set[str] = set()
    try:
        graph = getattr(index, "graph", None)
    except Exception:
        graph = None
    tests_map: dict = {}
    if _index_available(index):
        try:
            _mods, tests_map = testlinks.build_maps(sorted(index_files))
        except Exception:
            tests_map = {}
    for rel in sorted(existing):
        if graph is not None:
            try:
                callers.update(_norm_rel(f) for f in graph.dependents(rel))
            except Exception:
                unknowns.add(f"callers of {rel} are unverified: the import "
                             f"graph could not be read")
        else:
            unknowns.add(f"callers of {rel} are unverified: the import graph "
                         f"is unavailable")
        reliable = getattr(graph, "import_reliable", None) if graph else None
        try:
            if callable(reliable) and not reliable(rel):
                unknowns.add(f"callers of {rel} are not fully resolved for its "
                             f"language; dynamic imports are unknown")
        except Exception:
            pass
        if not _index_available(index):
            unknowns.add(f"{rel}: index unavailable, so callers and tests are "
                         f"unverified")
        try:
            linked = testlinks.tests_of(rel, tests_map) if tests_map else []
        except Exception:
            linked = []
        tests.update(linked)
        is_test = testlinks.classify(rel)[0]
        if is_test:
            tests.add(rel)
        elif (testlinks.family_of(rel) and not linked
                and _index_available(index)):
            unknowns.add(f"no indexed test is linked to {rel} by naming "
                         f"convention")
    for rel, symbol in sorted(symbol_refs):
        sure, approx = _callers_for(index, rel, symbol)
        for chunk in sure:
            callers.add(_norm_rel(getattr(chunk, "file", "")))
        for chunk in approx:
            callers.add(_norm_rel(getattr(chunk, "file", "")))
        if approx:
            unknowns.add(f"callers of {symbol} are approximate: bare-name "
                         f"matches in unrelated files may be included")
    callers.discard("")
    if plan_file.is_relative_to(root_path):
        callers.discard(_norm_rel(plan_file.relative_to(root_path).as_posix()))

    for rel in sorted(planned):
        if testlinks.classify(rel)[0]:
            tests.add(rel)
    for rel in sorted(callers):
        if testlinks.classify(rel)[0]:
            tests.add(rel)
    tests.discard("")
    if plan_file.is_relative_to(root_path):
        tests.discard(_norm_rel(plan_file.relative_to(root_path).as_posix()))

    if existing or planned:
        unknowns.add("caller lists come from static import and call edges; "
                     "dynamic imports and reflection may add callers")
        unknowns.add("test links use file naming conventions; tests with "
                     "unrelated names are not listed")
    for rel in sorted(planned):
        unknowns.add(f"{rel} is a planned creation: its symbols, callers and "
                     f"tests are unverified")
    for rel in sorted(existing):
        if _index_available(index) and rel not in index_files:
            unknowns.add(f"{rel} exists on disk but is not indexed: symbol, "
                         f"caller and test evidence is unverified")

    files, files_cut = _cap(existing | planned)
    symbol_items, symbols_cut = _cap(symbols)
    caller_items, callers_cut = _cap(callers)
    test_items, tests_cut = _cap(tests)
    if files_cut or symbols_cut or callers_cut or tests_cut:
        unknowns.add(f"impact lists are capped at {_MAX_IMPACT_ITEMS} entries")
    unknown_items, _unknown_cut = _cap(unknowns)
    impact = PlanImpact(files, symbol_items, caller_items, test_items,
                        unknown_items[:_MAX_UNKNOWNS])
    return issues, impact


def _backtick_symbols(title: str, mentions: list[_Mention],
                      *, allow_plain: bool = False
                      ) -> list[tuple[str, int]]:
    spans = [(m.start, m.end) for m in mentions]
    out: list[tuple[str, int]] = []
    for match in _BACKTICK_RE.finditer(title or ""):
        raw = match.group(1).strip()
        if not raw or _looks_like_file_token(raw):
            continue
        if set(raw) <= {"-"} or raw.startswith("-"):
            continue
        if any(start <= match.start() < end for start, end in spans):
            continue
        if not _IDENT_RE.fullmatch(raw):
            continue
        if raw.lower() in _SYMBOL_STOP:
            continue
        if not allow_plain and not _looks_codeish(raw):
            continue
        out.append((raw, match.start()))
    return out


def _check_mention(root: Path, index, index_files: set[str], task_index: int,
                   title: str, mention: _Mention, issues: list[PlanIssue],
                   existing: set[str], planned: set[str], symbols: set[str],
                   symbol_refs: set[tuple[str, str]], unknowns: set[str]
                   ) -> None:
    rel = mention.rel
    path = root / rel
    try:
        is_dir = path.is_dir()
    except OSError:
        is_dir = False
    if is_dir:
        return

    write_intent = mention.creation or mention.edit_like
    blocked = False
    for candidate in dict.fromkeys((rel, mention.resolved)):
        if security.is_sensitive_path(candidate):
            severity = ERROR if write_intent else WARNING
            blocked = blocked or severity == ERROR
            issues.append(_issue(
                "sensitive_path", severity, task_index, file=candidate,
                message=f"{candidate} holds secrets or credentials; a plan may "
                        f"not be allowed to write it"))
        protected = safety.is_protected(candidate)
        if protected:
            severity = ERROR if write_intent else WARNING
            blocked = blocked or severity == ERROR
            issues.append(_issue(
                "protected_path", severity, task_index, file=candidate,
                message=f"{candidate} is protected: {protected}"))

    case_hit = _case_collision(root, rel, index_files)
    exists = _exact_name_exists(root, rel)
    planned_hit = ""
    if not exists and not mention.creation:
        planned_hit = next(
            (p for p in sorted(planned) if p.casefold() == rel.casefold()), "")
    canonical = rel
    checked_symbols = True
    if planned_hit:
        # An earlier task in this same plan creates the file. Its later edit
        # is valid; the symbols, callers and tests remain unverified until the
        # creation actually lands.
        planned.add(rel)
        canonical = planned_hit
        unknowns.add(f"{rel} is created by an earlier task in this plan: its "
                     f"symbols, callers and tests are unverified")
    elif mention.creation:
        planned.add(rel)
        issues.extend(_creation_parent_issues(root, rel, task_index))
        if case_hit:
            if _is_windows():
                issues.append(_issue(
                    "case_collision", ERROR, task_index, file=rel,
                    message=f"cannot create {rel}: {case_hit} already exists "
                            f"under different casing"))
            else:
                issues.append(_issue(
                    "case_collision", WARNING, task_index, file=rel,
                    message=f"{rel} differs only by case from {case_hit}; both "
                            f"can coexist on this filesystem"))
        unknowns.add(f"{rel} is a planned creation: its symbols, callers and "
                     f"tests are unverified")
    elif exists or (case_hit and _is_windows()):
        canonical = case_hit or mention.resolved
        existing.add(canonical)
        if case_hit and case_hit != rel:
            issues.append(_issue(
                "case_collision", WARNING, task_index, file=rel,
                message=f"{rel} differs only by case from the existing "
                        f"{case_hit}"))
        if canonical not in index_files:
            unknowns.add(f"{canonical} exists on disk but is not indexed: "
                         f"symbol resolution is unverified")
    else:
        if not blocked:
            severity = ERROR if mention.edit_like else WARNING
            issues.append(_issue(
                "missing_file", severity, task_index, file=rel,
                message=f"{rel} does not exist and is not marked as a creation"))
        unknowns.add(f"{rel} does not exist: callers and tests are unverified")
        checked_symbols = False

    if checked_symbols:
        for symbol in mention.symbols:
            if symbol.lower() in _SYMBOL_STOP:
                continue
            _check_symbol(
                index, index_files, canonical, symbol, task_index=task_index,
                explicit=_looks_codeish(symbol),
                creation=mention.creation or not (exists or case_hit),
                issues=issues, symbols=symbols, symbol_refs=symbol_refs,
                unknowns=unknowns,
                dynamic=_dynamic_symbols_possible(root, canonical))


def blocking_issues(issues: list[PlanIssue]) -> list[PlanIssue]:
    """Findings that must stop execution: `error` severity (invalid included)."""
    return [i for i in issues if i.severity == ERROR]


def invalid_issues(issues: list[PlanIssue]) -> list[PlanIssue]:
    return [i for i in issues if i.code in _INVALID_CODES]


def check_plan_exit_code(issues: list[PlanIssue]) -> int:
    """0 clean/warnings, 1 definite errors, 2 invalid plan."""
    if invalid_issues(issues):
        return 2
    if any(i.severity == ERROR for i in issues):
        return 1
    return 0


def render_issue(issue: PlanIssue) -> str:
    """One human-readable line for a finding (CLI and gate output)."""
    where = issue.file
    if where and issue.symbol:
        where += f":{issue.symbol}"
    prefix = f"task {issue.task_index}: " if issue.task_index else ""
    middle = f"{where}: " if where else ""
    return f"  [{issue.severity}] {prefix}{middle}{issue.message}"


def render_impact(impact: PlanImpact, limit: int = 12) -> list[str]:
    """Lines of the conservative preview, bounded for terminal output."""
    def _join(items: tuple[str, ...]) -> str:
        shown = list(items[:limit])
        text = ", ".join(shown) if shown else "(none)"
        if len(items) > len(shown):
            text += f" (+{len(items) - len(shown)} more)"
        return text

    return [
        f"  files ({len(impact.files)}): {_join(impact.files)}",
        f"  symbols ({len(impact.symbols)}): {_join(impact.symbols)}",
        f"  callers ({len(impact.callers)}): {_join(impact.callers)}",
        f"  tests ({len(impact.tests)}): {_join(impact.tests)}",
        f"  unknowns ({len(impact.unknowns)}): {_join(impact.unknowns)}",
    ]


def gate_plan(root: Path, plan_path: Path, index, *, cfg=None,
              stream=None) -> int | None:
    """Run `check_plan` at an execution boundary.

    Returns None when execution may proceed (clean or warnings), otherwise the
    exit code (1 definite errors, 2 invalid). A validator crash is logged as a
    warning and never blocks a run: validation must not become a new failure
    mode for an otherwise valid plan.
    """
    try:
        issues, _impact = check_plan(root, plan_path, index)
    except Exception as exc:  # pragma: no cover - defensive boundary
        try:
            workspace.log_event(Path(root),
                                f"plan check skipped: {type(exc).__name__}",
                                "WARN")
        except Exception:
            pass
        return None
    warnings = [i for i in issues if i.severity == WARNING]
    if warnings:
        try:
            workspace.log_event(Path(root),
                                f"plan check: {len(warnings)} warning(s)",
                                "WARN")
        except Exception:
            pass
    # An empty checklist is a valid no-op for the loop (the CLI still reports
    # it as an invalid plan): execution must stop on definite errors only.
    blockers = [i for i in blocking_issues(issues) if i.code != "invalid_plan"]
    if not blockers:
        return None
    out = stream or sys.stderr
    for issue in blockers[:12]:
        print(render_issue(issue), file=out)
    extra = len(blockers) - 12
    if extra > 0:
        print(f"  ... and {extra} more plan issue(s)", file=out)
    code = check_plan_exit_code(blockers)
    reason = ("the plan is invalid" if code == 2
              else f"{len(blockers)} definite plan error(s)")
    print(f"  [plan check] execution stopped: {reason}. Run "
          f"`pedacito plan --check` for the full report.", file=out)
    try:
        workspace.log_event(Path(root),
                            f"plan check blocked execution: {reason}", "ERROR")
    except Exception:
        pass
    return code
