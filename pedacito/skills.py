"""Skills: named procedures the model loads on demand.

    skills.catalog(root, cfg)          # 'SKILLS ...' block: names + one line
    skills.load(root, cfg, "add-a-test")  # full text (progressive disclosure)
    skills.select_skills(root, task)   # the <=2 best trigger matches for a task
    skills.save_skill(root, name=..., description=..., triggers=[...], body=...)
    skills.delete_skill(root, "my-skill")
    skills.draft_skill_with_llm(client, "Skill for Alpine.js dark-mode components")

A skill is a markdown file with frontmatter:

    ---
    name: add-a-test
    description: Add a focused test for a function that has none
    triggers: test, coverage, pytest
    context: fork            # optional: run as a sub-agent (needs allow_subagents)
    ---
    1. `callers` the target ... (the procedure)

Search order (first name wins): `<root>/.pedacito/skills/<name>/SKILL.md` or
`<root>/.pedacito/skills/<name>.md`, the same two layouts under
`<config dir>/skills/`, then the bundled library of sixteen starters. The
catalog costs a few dozen tokens; a skill's text enters the conversation only
when the model asks for it (or `select_skills` matches its triggers, capped at
two skills / 1,200 chars so a prompt never bloats). Skills are PROMPT TEXT:
Pedacito never executes anything a skill folder contains, and a skill cannot
widen the tools a run has.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .memory import _frontmatter

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")
MAX_SKILL_CHARS = 3500
# Built-in body budget: crisp checklists only. Custom skills may use the full
# MAX_SKILL_CHARS, but a draft aims for the same concision.
MAX_BUILTIN_BODY_CHARS = 550
# select_skills injects at most this many skills, and at most this many
# characters of skill text in total -- the prompt-size contract.
MAX_SELECTED_SKILLS = 2
MAX_SELECTED_CHARS = 1200

# name -> (description, checklist body, trigger keywords). Triggers are matched
# against the task text by select_skills; the catalog stays one line per skill.
BUILTIN_SKILLS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "add-a-test": (
        "Add a focused test for a function that has none",
        "1. `context` the target function: note its callers, callees and any "
        "existing `tested by:` file.\n"
        "2. `read` the nearest existing test file to copy its imports, "
        "fixtures and naming style.\n"
        "3. Write ONE test for the main behaviour and ONE for a failure/edge "
        "case. Use real values from the code you read, not invented ones.\n"
        "4. If `run` is enabled, run just that test before and after.\n"
        "5. Do not change the code under test.",
        ("test", "tests", "coverage", "untested", "pytest", "add a test")),
    "fix-a-failing-test": (
        "Diagnose and fix a failing test",
        "1. `run` the failing test first and quote the exact failure line.\n"
        "2. `read` the test, then the code it exercises (`context` finds it).\n"
        "3. Decide whether the TEST or the CODE is wrong, and say why in one "
        "sentence before editing.\n"
        "4. Make the smallest edit that fixes that one failure.\n"
        "5. `run` the same test again; report the result honestly.",
        ("failing test", "failed test", "test failure", "broken test",
         "assertionerror", "red")),
    "refactor-safely": (
        "Rename or restructure code without breaking callers",
        "1. `callers` the symbol: every caller and importer is in scope.\n"
        "2. `read` each caller's call site before changing the signature.\n"
        "3. Change the definition and ALL call sites in the same answer, one "
        "file per block group.\n"
        "4. Never leave the old name behind unless asked for a shim.\n"
        "5. Re-run the impacted tests if `run` is enabled.",
        ("refactor", "restructure", "signature", "callers", "extract")),
    "review-my-diff": (
        "Review the current uncommitted changes for defects",
        "1. `git` with `diff` (or the `diff` action) to see what changed.\n"
        "2. For each changed function: `read` the whole function, not only "
        "the hunk.\n"
        "3. Look for: unhandled errors, off-by-one, missing await, changed "
        "behaviour without a test, stale references (`callers`).\n"
        "4. Report findings as `file:line -- problem`. Say PASS only if you "
        "found none. Do not edit.",
        ("review", "diff", "uncommitted", "defect", "audit")),
    "find-dead-code": (
        "Check whether a file or function is unused",
        "1. `callers` the symbol, and `grep` the file name for imports.\n"
        "2. Check dynamic uses: string-built paths, entry-point tables, "
        "plugin registries, config files (`grep` them).\n"
        "3. Confirm the program's entry point does or does not load it.\n"
        "4. Report evidence, not certainty: 'no references found in N files'.",
        ("dead code", "unused", "unreferenced", "orphan")),
    "surgical-symbol-edit": (
        ("Replace or insert functions and methods in large files using "
         "zero-anchor SYMBOL splices or LINES ranges without echoing old "
         "code."),
        ("1. `outline` the file first. Copy the EXACT symbol names and line "
         "ranges it prints; do not guess them from memory.\n"
         "2. For a >120-line file, or whenever an exact SEARCH block fails to "
         "match, do NOT paste the old code back. Use a splice instead.\n"
         "3. Replace or insert one definition with a zero-anchor SYMBOL "
         "splice: `<<<<<<< SYMBOL path/to/file.py :: Class.method`, the new "
         "body, then `>>>>>>> END_SYMBOL`.\n"
         "4. Insert immediately after a definition with "
         "`<<<<<<< AFTER_SYMBOL path/to/file.py :: symbol` ... "
         "`>>>>>>> END_SYMBOL`.\n"
         "5. As a last resort use a line-range whole-file block: "
         "`--- FILE: path LINES a-b` with an EMPTY `SEARCH` section and the "
         "replacement text in `REPLACE`. Take a-b from `outline`.\n"
         "6. Keep edits minimal: one symbol or one range per block. Re-read "
         "the changed region only to confirm, not to re-quote the whole "
         "file."),
        ("symbol", "splice", "large file", "outline", "lines range")),
    "frontend-dom-audio-debug": (
        ("Verify and fix web UIs, HTML/JS module contracts, canvas "
         "rendering, and silent Web Audio failures."),
        ("1. DOM contract: list every `getElementById(\"...\")` and "
         "`querySelector(\"#...\")` in the JS, then check each id exists "
         "exactly once in the HTML. Fix the mismatch where it is cheapest; "
         "never duplicate an id to make two lookups pass.\n"
         "2. ES modules: import each symbol once per module. A duplicate "
         "top-level import (or a second `type=\"module\"` script re-declaring "
         "the same name) throws at load and leaves the page dead.\n"
         "3. Reproduce with `browser`: load the page, then use "
         "`hold <Key> <seconds>` for a fixed-timestep game loop and `eval` to "
         "return numbers -- element bounding boxes, computed styles, canvas "
         "pixel values or state counters. Quote the numbers as evidence.\n"
         "4. Web Audio: create/resume the `AudioContext` inside a real user "
         "gesture (click/keydown), connect the source through gain to "
         "`ctx.destination`, and assert gain is finite and greater than 0. "
         "`NaN`, 0 gain, a suspended context or a missing connection is "
         "silent -- check those four before changing the sound code.\n"
         "5. Re-run the browser check after the fix and report the observed "
         "numbers, not 'it should work now'."),
        ("dom", "getelementbyid", "web audio", "audiocontext", "es module",
         "web ui")),
    "safe-multi-file-rename": (
        ("Rename a function, class, method, or variable across the project "
         "deterministically without manual multi-file edits."),
        ("1. `usages` the old name first: it lists every reference and "
         "importer, so you can see the true scope before touching anything.\n"
         "2. If `allow_write` is enabled, call `rename` (or `rename_symbol`) "
         "with the old and new names instead of editing files by hand. For a "
         "Python method, use the receiver-qualified form `ClassName.method` "
         "so same-named methods on other classes are left alone.\n"
         "3. Preview first when unsure; apply only after the scope matches "
         "the `usages` list.\n"
         "4. Do not hand-edit strings or comments containing the old name; "
         "the deterministic rename already leaves them untouched.\n"
         "5. Verify with `run_test` (or the impacted tests the harness names) "
         "and `grep` for the old identifier to prove nothing was missed."),
        ("rename", "multi-file", "usages", "identifier")),
    # --- the second eight: domain checklists, each body <= 550 chars ---------
    "fastapi-rest-endpoint": (
        "Add or fix a FastAPI REST endpoint with Pydantic validation and tests",
        "1. `read` the router and its Pydantic models first; reuse existing "
        "schemas, response_model and dependencies.\n"
        "2. Type every path/query/body param; set explicit status_code (201 "
        "create, 204 delete); raise HTTPException with a clear detail.\n"
        "3. Keep handlers thin: validation in Pydantic, logic in a service "
        "function, no I/O in the schema layer.\n"
        "4. Add TestClient tests: happy path, a 422 validation error, one "
        "not-found/auth case.\n"
        "5. `run` the new tests; confirm /docs still loads the schema.",
        ("fastapi", "pydantic", "apireouter", "testclient", "endpoint",
         "status_code", "httpexception")),
    "canvas-svg-2d-3d": (
        "Debug and build canvas/SVG/WebGL rendering and animation loops",
        "1. `read` the render code and its HTML container; the canvas/svg id "
        "must exist once with non-zero width/height before drawing.\n"
        "2. SVG: set viewBox and scale coordinates from it, not client "
        "pixels. Canvas: scale by devicePixelRatio; reset transforms per "
        "frame.\n"
        "3. One requestAnimationFrame loop, delta-time movement; cancel it "
        "on teardown, never stack two loops.\n"
        "4. WebGL/three.js: check shader compile logs; polar/radar math "
        "converts degrees to radians once, at the boundary.\n"
        "5. Verify with `browser` eval: sample pixels or boxes, quote numbers.",
        ("canvas", "svg", "viewbox", "webgl", "three.js",
         "requestanimationframe", "shader", "polar", "radar")),
    "sqlite-sql-migrations": (
        "Write safe SQLite schema migrations and correct SQL queries",
        "1. `read` the current schema (sqlite_master or migration files) "
        "before writing SQL; type and constrain every column.\n"
        "2. Migrations are forward-only and idempotent: IF NOT EXISTS, "
        "guarded ALTER TABLE ADD COLUMN; never rewrite a shipped one.\n"
        "3. PRAGMA foreign_keys = ON per connection; index every foreign key "
        "and every WHERE/JOIN/ORDER BY column.\n"
        "4. GROUP BY every non-aggregated column; explicit JOIN ... ON; "
        "parameterize values (?), never f-string SQL.\n"
        "5. `run` the migration on a database copy; compare row counts.",
        ("sqlite", "sqlite3", "sql", "schema", "migration", "foreign_key",
         "group by", "join")),
    "async-concurrency-safety": (
        "Fix asyncio/threading races, deadlocks and subprocess handling",
        "1. `grep` shared mutable state first: any variable touched by two "
        "tasks/threads is a race candidate; list them before editing.\n"
        "2. asyncio: never block the loop (time.sleep, requests, heavy CPU); "
        "await every coroutine; gather with return_exceptions for partial "
        "failure.\n"
        "3. threading: one Lock per invariant, fixed acquisition order "
        "against deadlock; prefer queue.Queue over shared lists.\n"
        "4. subprocess: always a timeout, captured output, reaped process; "
        "no shell with user input.\n"
        "5. Test the concurrent path twice; assert the invariant.",
        ("asyncio", "async", "await", "threading", "lock", "deadlock",
         "race condition", "subprocess")),
    "security-ssrf-xss-hardening": (
        "Harden code against SSRF, XSS, CSRF, injection and path traversal",
        "1. Treat every external string as hostile: URL, path, HTML, SQL, "
        "header. `grep` where it enters and where it is used.\n"
        "2. SSRF: allowlist scheme and host; reject private/loopback IPs "
        "after DNS resolution; re-check redirects.\n"
        "3. XSS: escape at the output boundary (textContent, autoescape); "
        "never innerHTML with user data; allowlist tags if HTML is needed.\n"
        "4. Path traversal/symlink: resolve() and assert the path stays "
        "under its root; reject escaping symlinks.\n"
        "5. CSRF: token or SameSite cookie on state changes. One regression "
        "test per fixed hole.",
        ("ssrf", "xss", "csrf", "sanitize", "path traversal", "symlink",
         "injection", "allowlist")),
    "i18n-localization-catalog": (
        "Externalize UI strings and keep translation catalogs in sync",
        "1. `grep` the UI for hardcoded strings; move each to the catalog "
        "under a stable key, never the English text as key.\n"
        "2. One source-of-truth locale (UTF-8, no BOM); other locales mirror "
        "its keys; a missing key falls back to the source, not the raw key.\n"
        "3. Named placeholders ({name}) identical across locales; plurals "
        "via the locale's rule, not English -s.\n"
        "4. Locale APIs (Intl, babel) for dates/numbers; layout tolerant of "
        "30% longer strings (Spanish, German).\n"
        "5. Test: load every catalog, assert key and placeholder parity.",
        ("i18n", "locale", "translate", "translation", "spanish", "catalog",
         "placeholder", "utf-8")),
    "parser-lexer-state-machine": (
        "Build lexers, parsers, ASTs and state machines that cannot loop",
        "1. `read` the grammar or existing parser; write down tokens, "
        "precedence and associativity before coding.\n"
        "2. Lexer: one pass, longest match, explicit token kinds, "
        "line/column errors; never swallow unknown characters.\n"
        "3. Parser: Pratt climbing for expressions, recursive descent for "
        "statements; every recursion must consume a token.\n"
        "4. Real typed AST nodes; RPN via a stack with an arity check per "
        "operator.\n"
        "5. State machines: one transition table, invalid transitions "
        "rejected. Test empty input, trailing garbage, deep nesting, each "
        "precedence level.",
        ("lexer", "tokenizer", "parser", "ast", "rpn", "precedence",
         "grammar", "state machine")),
    "performance-memory-profiling": (
        "Find and fix bottlenecks, unbounded caches and memory leaks",
        "1. Measure first: timeit/cProfile the hot path and record the "
        "baseline; a bottleneck is a measurement, not a guess.\n"
        "2. Fix complexity first: O(n^2) scans become dict/set lookups; "
        "hoist invariant work; batch I/O.\n"
        "3. Bound every cache: LRU maxsize or TTL plus eviction; an "
        "unbounded dict cache is a leak. Test the eviction path.\n"
        "4. Stream large files/responses in chunks; close handles and drop "
        "references so GC can collect.\n"
        "5. Re-measure the same workload; report before/after numbers and "
        "watch RSS across repeats.",
        ("lru", "ttl", "cache", "evict", "complexity", "bottleneck",
         "memory leak", "stream")),
}

# Historical name kept for callers and drift guards that say "bundled".
BUNDLED = BUILTIN_SKILLS


def _dirs(root, cfg) -> list[Path]:
    dirs = [Path(root) / ".pedacito" / "skills"] if root is not None else []
    try:
        from . import config as config_mod
        dirs.append(config_mod.USER_CONFIG.parent / "skills")
    except Exception:
        pass
    return dirs


def _skill_files(d: Path) -> list[Path]:
    """Both supported layouts, folder form first: <name>/SKILL.md, <name>.md."""
    return sorted(d.glob("*/SKILL.md")) + sorted(
        p for p in d.glob("*.md") if p.is_file())


def _parse_triggers(raw: str) -> list[str]:
    out: list[str] = []
    for part in str(raw or "").split(","):
        # A trigger is one frontmatter VALUE: an embedded newline/control
        # character could inject another metadata key (`x\nname: builtin`)
        # and let a saved skill claim a different (even built-in) name.
        t = re.sub(r"[\x00-\x1f\x7f]+", " ", part)
        t = " ".join(t.strip().strip("\"'").lower().split())
        if t and t not in out:
            out.append(t)
    return out


def _scan(root, cfg) -> dict[str, dict]:
    """name -> {description, body, meta, source, triggers}, earlier sources win."""
    found: dict[str, dict] = {}
    for d in _dirs(root, cfg):
        if not d.is_dir():
            continue
        for p in _skill_files(d):
            try:
                meta, body = _frontmatter(p.read_text(encoding="utf-8"))
            except OSError:
                continue
            fallback = p.parent.name if p.name == "SKILL.md" else p.stem
            name = (meta.get("name") or fallback).strip().lower()
            if not _NAME_RE.match(name) or name in found:
                continue
            found[name] = {"description": meta.get("description", "")[:120],
                           "body": body, "meta": meta,
                           "triggers": _parse_triggers(meta.get("triggers", "")),
                           "source": str(p)}
    for name, (desc, body, triggers) in BUILTIN_SKILLS.items():
        found.setdefault(name, {"description": desc, "body": body, "meta": {},
                                "triggers": list(triggers),
                                "source": "bundled"})
    return found


def names(root, cfg=None) -> list[str]:
    return sorted(_scan(root, cfg))


def _scope_of(info: dict, root) -> str:
    source = str(info.get("source") or "")
    if source == "bundled":
        return "bundled"
    if root is not None:
        # Path containment, not a string prefix: on Windows `str(Path)`
        # uses backslashes, so `source.startswith(root + "/")` labelled every
        # project skill "user" there.
        try:
            Path(source).resolve().relative_to(Path(root).resolve())
            return "project"
        except (ValueError, OSError):
            pass
    return "user"


def list_skills(root, cfg=None) -> list[dict]:
    """Every skill with its scope badge, for `pedacito skills list` and the
    GUI library: [{name, description, triggers, body, scope, source}]."""
    rows = []
    for name, info in sorted(_scan(root, cfg).items()):
        rows.append({"name": name,
                     "description": info["description"],
                     "triggers": list(info.get("triggers") or []),
                     "body": info["body"],
                     "scope": _scope_of(info, root),
                     "source": info["source"]})
    return rows


def catalog(root, cfg=None, limit: int | None = None, task: str = "") -> str:
    """The skills block ('' when disabled): names with one-line descriptions,
    skills whose name or description share words with `task` listed first, and
    capped to `max_skill_catalog_chars`."""
    if cfg is not None and not getattr(cfg, "use_skills", True):
        return ""
    limit = int(limit or getattr(cfg, "max_skill_catalog_chars", 900) or 900)
    found = _scan(root, cfg)
    if not found:
        return ""
    words = set(re.findall(r"[a-z]{4,}", (task or "").lower()))

    def rel(item):
        name, info = item
        text = f"{name} {info['description']}".lower()
        return -sum(1 for w in words if w in text)

    rows = []
    head = "SKILLS (load one with the `skill` action when the task matches)"
    used = len(head)
    for name, info in sorted(found.items(), key=lambda it: (rel(it), it[0])):
        row = f"- {name}: {info['description']}"
        if used + len(row) + 1 > limit:
            break
        rows.append(row)
        used += len(row) + 1
    return head + "\n" + "\n".join(rows) if rows else ""


def _trigger_score(triggers, text: str) -> float:
    """How well a skill's triggers match the task: a multi-word phrase is the
    strongest signal, a whole word next, a bare substring least."""
    score = 0.0
    for t in triggers or ():
        t = str(t).strip().lower()
        if not t:
            continue
        if " " in t:
            if t in text:
                score += 2.0
        elif re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", text):
            score += 1.0
        elif len(t) >= 5 and t in text:
            # Substring only for longer triggers: "ast" must not match
            # inside "fastapi", while "requestanimationframe" still matches
            # camelCase identifiers.
            score += 0.5
    return score


def select_skills(root, task, cfg=None, limit: int = MAX_SELECTED_SKILLS,
                  max_chars: int = MAX_SELECTED_CHARS) -> list[dict]:
    """The best trigger matches for one task, capped so the prompt never
    bloats: at most `limit` skills (default 2) and `max_chars` characters of
    body text in total (default 1,200). Highest score first; ties by name.
    Skills with no trigger match are never returned."""
    if cfg is not None and not getattr(cfg, "use_skills", True):
        return []
    text = " ".join((task or "").lower().split())
    if not text:
        return []
    scored = []
    for name, info in _scan(root, cfg).items():
        score = _trigger_score(info.get("triggers"), text)
        if score > 0:
            scored.append((score, name, info))
    scored.sort(key=lambda row: (-row[0], row[1]))
    picked: list[dict] = []
    used = 0
    limit = max(0, int(limit))
    max_chars = max(0, int(max_chars))
    for score, name, info in scored[:limit]:
        body = info["body"].strip()
        room = max_chars - used
        if room <= 0:
            break
        if len(body) > room:
            body = body[:room].rsplit("\n", 1)[0].rstrip() or body[:room]
        picked.append({"name": name, "description": info["description"],
                       "triggers": list(info.get("triggers") or []),
                       "body": body, "source": info["source"],
                       "score": score})
        used += len(body)
    return picked


def render_selected(root, task, cfg=None) -> str:
    """The matched-checklist block for a task's first user turn ('' when
    nothing matches). Whole block capped at MAX_SELECTED_CHARS."""
    picked = select_skills(root, task, cfg)
    if not picked:
        return ""
    head = ("SKILLS FOR THIS TASK (matched checklists -- follow them; load "
            "another with the `skill` action):")
    blocks = [head]
    used = len(head)
    for row in picked:
        block = f"## {row['name']}\n{row['body']}"
        if used + len(block) + 2 > MAX_SELECTED_CHARS:
            break
        blocks.append(block)
        used += len(block) + 2
    return "\n\n".join(blocks) if len(blocks) > 1 else ""


def load(root, cfg, name: str) -> tuple[str, dict | None]:
    """(text, info) for skill `name`; text is an ERROR line when unknown."""
    key = (name or "").strip().lower()
    found = _scan(root, cfg)
    if key not in found:
        avail = ", ".join(sorted(found)) or "(none)"
        return (f"ERROR: no skill named '{name}'. Available: {avail}.", None)
    info = found[key]
    body = info["body"].strip()[:MAX_SKILL_CHARS]
    return (f"SKILL {key} ({info['source']}):\n{body}", info)


def tool_skill(index, cfg, name: str) -> str:
    if not getattr(cfg, "use_skills", True):
        return "ERROR: skills are disabled (use_skills = false)."
    text, _info = load(index.root, cfg, name)
    return text


# ------------------------------------------------- custom skill management

def _scope_dir(root, scope: str) -> Path:
    scope = (scope or "project").strip().lower()
    if scope in ("user", "global"):
        from . import config as config_mod
        return config_mod.USER_CONFIG.parent / "skills"
    if scope != "project":
        raise ValueError(f"unknown skill scope {scope!r} (project or user)")
    if root is None:
        raise ValueError("a project root is required for scope='project'")
    return Path(root) / ".pedacito" / "skills"


def _validate_slug(name: str) -> str:
    key = (name or "").strip().lower()
    if not _NAME_RE.match(key):
        raise ValueError(
            f"invalid skill name {name!r}: use a slug like 'my-skill' "
            "(lowercase letters, digits, '-' and '_', starting with a "
            "letter or digit, at most 41 characters)")
    return key


def _skill_text(name: str, description: str, triggers: list[str],
                body: str) -> str:
    trig = ", ".join(_parse_triggers(", ".join(triggers or [])))
    return ("---\n"
            f"name: {name}\n"
            f"description: {description}\n"
            f"triggers: {trig}\n"
            "---\n\n"
            + body.strip() + "\n")


def save_skill(root: Path | None, *, name: str, description: str,
               triggers: list[str], body: str,
               scope: str = "project") -> Path:
    """Write a custom skill file and return its path.

    `scope="project"` writes `<root>/.pedacito/skills/<name>.md`;
    `scope="user"` (or "global") writes `~/.config/pedacito/skills/<name>.md`.
    An existing folder-form skill (`<name>/SKILL.md`) in the same scope is
    updated in place so edits keep their layout. The name must be a safe
    slug; description and body must be non-empty.
    """
    key = _validate_slug(name)
    description = " ".join((description or "").split())[:120]
    body = (body or "").strip()
    if not description:
        raise ValueError("a skill needs a one-line description")
    if not body:
        raise ValueError("a skill needs a checklist body")
    if len(body) > MAX_SKILL_CHARS:
        raise ValueError(f"skill body is {len(body)} chars; the cap is "
                         f"{MAX_SKILL_CHARS}")
    d = _scope_dir(root, scope)
    legacy = d / key / "SKILL.md"
    path = legacy if legacy.is_file() else d / f"{key}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(_skill_text(key, description, list(triggers or []), body),
                   encoding="utf-8")
    tmp.replace(path)
    return path


def delete_skill(root: Path | None, name: str,
                 scope: str = "project") -> bool:
    """Delete a custom skill file (either layout) from one scope. Built-in
    skills are immutable: with no custom file of that name, returns False.
    A custom file that shadows a built-in IS deletable (the built-in then
    shows through again)."""
    key = _validate_slug(name)
    d = _scope_dir(root, scope)
    for path in (d / key / "SKILL.md", d / f"{key}.md"):
        if path.is_file():
            path.unlink()
            if path.parent != d:
                try:
                    path.parent.rmdir()     # only when the folder is now empty
                except OSError:
                    pass
            return True
    return False


_DRAFT_SYSTEM = """\
You write Pedacito skills: short procedural checklists a coding assistant
follows for one kind of task. Reply with ONLY one JSON object, no prose and
no code fences, with exactly these keys:
  "name": a slug (lowercase letters/digits/'-'/'_', <= 41 chars)
  "description": one line, <= 120 chars, saying when to use the skill
  "triggers": 4-8 lowercase keywords or short phrases that identify the task
  "body": a numbered checklist of 3-6 concrete steps, <= 550 chars total
Rules: steps must be actionable and specific to the described domain; prefer
exact tool/API names over vague advice; never invent project facts; the body
is prompt text, not code to execute."""


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)[:41].strip("-")
    if not slug or not _NAME_RE.match(slug):
        return "custom-skill"
    return slug


def draft_skill_with_llm(client, prompt: str) -> dict:
    """Turn a one-sentence description into {name, description, triggers,
    body} using the active model. Raises ValueError on an empty prompt or a
    reply that is not usable skill JSON."""
    prompt = " ".join((prompt or "").split())
    if not prompt:
        raise ValueError("describe what the skill should teach first")
    text = client.chat(
        [{"role": "system", "content": _DRAFT_SYSTEM},
         {"role": "user", "content": f"Write a skill for: {prompt}"}],
        max_tokens=700, stream=False, disable_thinking=True)
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("the model did not return skill JSON; try again")
    try:
        data = json.loads(raw[start:end + 1])
    except ValueError as e:
        raise ValueError(f"the model returned invalid skill JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("the model returned invalid skill JSON (not an object)")
    name = _slugify(str(data.get("name") or prompt[:41]))
    description = " ".join(str(data.get("description") or "").split())[:120]
    if not description:
        description = f"Procedure for: {prompt}"[:120]
    triggers = data.get("triggers") or []
    if isinstance(triggers, str):
        triggers = triggers.split(",")
    triggers = _parse_triggers(", ".join(str(t) for t in triggers))[:12]
    body = str(data.get("body") or "").strip()
    if not body:
        raise ValueError("the model returned an empty skill body; try again")
    if len(body) > MAX_BUILTIN_BODY_CHARS:
        body = body[:MAX_BUILTIN_BODY_CHARS].rsplit("\n", 1)[0].rstrip() \
            or body[:MAX_BUILTIN_BODY_CHARS]
    return {"name": name, "description": description,
            "triggers": triggers, "body": body}
