"""Skills: named procedures the model loads on demand.

    skills.catalog(root, cfg)          # 'SKILLS ...' block: names + one line
    skills.load(root, cfg, "add-a-test")  # full text (progressive disclosure)

A skill is a markdown file with frontmatter:

    ---
    name: add-a-test
    description: Add a focused test for a function that has none
    context: fork            # optional: run as a sub-agent (needs allow_subagents)
    ---
    1. `callers` the target ... (the procedure)

Search order (first name wins): `<root>/.pedacito/skills/<name>/SKILL.md`,
`<config dir>/skills/<name>/SKILL.md`, then the small bundled starter set. The
catalog costs a few dozen tokens; a skill's text enters the conversation only
when the model asks for it. Skills are PROMPT TEXT: Pedacito never executes
anything a skill folder contains, and a skill cannot widen the tools a run has.
"""

from __future__ import annotations

import re
from pathlib import Path

from .memory import _frontmatter

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")
MAX_SKILL_CHARS = 3500

BUNDLED: dict[str, tuple[str, str]] = {
    "add-a-test": (
        "Add a focused test for a function that has none",
        "1. `context` the target function: note its callers, callees and any "
        "existing `tested by:` file.\n"
        "2. `read` the nearest existing test file to copy its imports, "
        "fixtures and naming style.\n"
        "3. Write ONE test for the main behaviour and ONE for a failure/edge "
        "case. Use real values from the code you read, not invented ones.\n"
        "4. If `run` is enabled, run just that test before and after.\n"
        "5. Do not change the code under test."),
    "fix-a-failing-test": (
        "Diagnose and fix a failing test",
        "1. `run` the failing test first and quote the exact failure line.\n"
        "2. `read` the test, then the code it exercises (`context` finds it).\n"
        "3. Decide whether the TEST or the CODE is wrong, and say why in one "
        "sentence before editing.\n"
        "4. Make the smallest edit that fixes that one failure.\n"
        "5. `run` the same test again; report the result honestly."),
    "refactor-safely": (
        "Rename or restructure code without breaking callers",
        "1. `callers` the symbol: every caller and importer is in scope.\n"
        "2. `read` each caller's call site before changing the signature.\n"
        "3. Change the definition and ALL call sites in the same answer, one "
        "file per block group.\n"
        "4. Never leave the old name behind unless asked for a shim.\n"
        "5. Re-run the impacted tests if `run` is enabled."),
    "review-my-diff": (
        "Review the current uncommitted changes for defects",
        "1. `git` with `diff` (or the `diff` action) to see what changed.\n"
        "2. For each changed function: `read` the whole function, not only "
        "the hunk.\n"
        "3. Look for: unhandled errors, off-by-one, missing await, changed "
        "behaviour without a test, stale references (`callers`).\n"
        "4. Report findings as `file:line -- problem`. Say PASS only if you "
        "found none. Do not edit."),
    "find-dead-code": (
        "Check whether a file or function is unused",
        "1. `callers` the symbol, and `grep` the file name for imports.\n"
        "2. Check dynamic uses: string-built paths, entry-point tables, "
        "plugin registries, config files (`grep` them).\n"
        "3. Confirm the program's entry point does or does not load it.\n"
        "4. Report evidence, not certainty: 'no references found in N files'."),
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
        "file.")),
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
        "numbers, not 'it should work now'.")),
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
        "and `grep` for the old identifier to prove nothing was missed.")),
}


def _dirs(root, cfg) -> list[Path]:
    dirs = [Path(root) / ".pedacito" / "skills"]
    try:
        from . import config as config_mod
        dirs.append(config_mod.USER_CONFIG.parent / "skills")
    except Exception:
        pass
    return dirs


def _scan(root, cfg) -> dict[str, dict]:
    """name -> {description, body, meta, source}, earlier sources win."""
    found: dict[str, dict] = {}
    for d in _dirs(root, cfg):
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*/SKILL.md")):
            try:
                meta, body = _frontmatter(p.read_text(encoding="utf-8"))
            except OSError:
                continue
            name = (meta.get("name") or p.parent.name).strip().lower()
            if not _NAME_RE.match(name) or name in found:
                continue
            found[name] = {"description": meta.get("description", "")[:120],
                           "body": body, "meta": meta,
                           "source": str(p)}
    for name, (desc, body) in BUNDLED.items():
        found.setdefault(name, {"description": desc, "body": body, "meta": {},
                                "source": "bundled"})
    return found


def names(root, cfg=None) -> list[str]:
    return sorted(_scan(root, cfg))


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
