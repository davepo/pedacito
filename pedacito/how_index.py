"""Structured reference index behind `pedacito how` and the GUI How tab.

The index has three deterministic sources of truth:

* every CLI subcommand, its flags and copy-pasteable examples;
* a GUI navigation map (exact labels and paths through the page);
* every `Config` setting, its default, help text, GUI location and TOML form.

`search()` ranks those items against a question and `render_answer()` formats
the same items without a model, so help always works offline.  A reachable
model server may optionally rewrite the retrieved facts into a shorter answer
(see `how.answer_structured`), but it never supplies missing facts.
"""

from __future__ import annotations

import re
from functools import lru_cache

from .config import Config

# ------------------------------------------------------------------ tokenising

_WORD_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-/]*")

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do",
    "does", "for", "from", "get", "how", "i", "if", "in", "is", "it", "me",
    "my", "of", "on", "or", "should", "so", "that", "the", "this", "to",
    "use", "using", "what", "when", "where", "which", "who", "why", "will",
    "with", "would", "you", "your", "please", "want", "need", "there",
    "option", "options", "preference", "preferences", "set", "up",
}

_SYNONYMS = {
    "setup": ("configure", "configuration", "init", "install", "config"),
    "configure": ("setup", "config", "configuration", "toml", "settings"),
    "config": ("configuration", "configure", "settings", "toml", "init"),
    "configuration": ("config", "settings", "toml"),
    "install": ("pip", "setup", "install"),
    "command": ("cli", "terminal", "usage"),
    "cli": ("command", "terminal", "shell", "pedacito"),
    "run": ("command", "execute", "start", "usage"),
    "undo": ("restore", "backup", "revert", "rollback", "snapshot"),
    "revert": ("restore", "backup", "undo", "rollback"),
    "restore": ("undo", "backup", "revert", "snapshot"),
    "edit": ("apply", "review", "changes", "write"),
    "start": ("new", "scaffold", "create", "gui"),
    "create": ("new", "scaffold", "generate"),
    "error": ("fail", "failure", "troubleshoot", "wrong", "limits"),
    "broke": ("error", "fail", "troubleshoot", "wrong"),
    "slow": ("cost", "tokens", "time", "context", "performance"),
    "memory": ("vram", "gpu", "context"),
    "api": ("key", "hosted", "deepseek", "openai", "token"),
    "key": ("api", "secret", "token", "hosted"),
    "model": ("profile", "models", "server", "llm"),
    "models": ("model", "profile", "profiles", "llm"),
    "profile": ("model", "models", "default"),
    "index": ("indexing", "map", "summaries", "chunks"),
    "ignore": ("exclude", "excluded", "pedacitoignore"),
    "exclude": ("ignore", "pedacitoignore"),
    "thinking": ("reasoning", "think", "disable_thinking", "no_think"),
    "reasoning": ("thinking", "think", "disable_thinking"),
    "think": ("thinking", "reasoning", "disable_thinking"),
    "disable": ("off", "toggle", "turn"),
    "off": ("disable", "toggle", "turn"),
    "switch": ("change", "choose", "select", "active"),
    "change": ("switch",),
    "import": ("scan", "discover", "add", "models"),
    "scan": ("discover", "import", "models"),
    "discover": ("scan", "import", "find"),
    "reset": ("baseline", "defaults", "wipe", "restore", "clear"),
    "baseline": ("reset", "defaults"),
    "defaults": ("default", "reset", "baseline"),
    "setting": ("config", "configuration", "option", "preference"),
    "settings": ("config", "configuration", "options", "preferences"),
    "reference": ("file", "files", "attach", "mention"),
    "chat": ("ask", "question", "conversation", "thread"),
    "ask": ("question", "explain", "chat"),
    "tool": ("tools", "shell", "command", "permission"),
    "backup": ("restore", "undo", "snapshot"),
    "small": ("tiny", "weak", "local", "budget"),
    "tiny": ("small", "weak", "local"),
    "cheat": ("cheatsheet", "comparison", "reference"),
}

_PHRASE_HINTS = {
    "turn off thinking": ("disable_thinking", "thinking"),
    "turn thinking off": ("disable_thinking", "thinking"),
    "disable thinking": ("disable_thinking", "thinking"),
    "disable reasoning": ("disable_thinking", "reasoning"),
    "switch model": ("profile", "switch"),
    "switch models": ("profile", "switch"),
    "change model": ("profile", "switch"),
    "import model": ("models scan", "import"),
    "import models": ("models scan", "import"),
    "reset settings": ("reset", "baseline"),
    "reset config": ("reset", "baseline"),
    "reset my config": ("reset", "baseline"),
    "reset to baseline": ("reset", "baseline"),
    "undo run": ("restore", "backup"),
    "undo auto": ("restore", "backup"),
    "reference file": ("@", "files"),
    "reference a file": ("@", "files"),
    "mention file": ("@", "files"),
    "small model": ("small", "budget"),
    "small models": ("small", "budget"),
}


def _terms(question: str) -> dict[str, float]:
    """Weighted lowercase terms, with light synonym expansion."""
    raw: list[str] = []
    for word in _WORD_RE.findall(question.lower()):
        word = word.strip(".-/")
        if len(word) >= 2 and word not in _STOP:
            raw.append(word)
    weights: dict[str, float] = {}
    for word in raw:
        weights[word] = 1.6 if ("_" in word or "-" in word) else 1.0
    for word in raw:
        for part in re.split(r"[_\-./]", word):
            if len(part) >= 3 and part not in _STOP:
                weights.setdefault(part, 0.35)
        if word.endswith("es") and len(word) > 4:
            weights.setdefault(word[:-2], 0.6)
        if word.endswith("s") and len(word) > 4:
            weights.setdefault(word[:-1], 0.5)
    for term in list(weights):
        for synonym in _SYNONYMS.get(term, ()):
            weights.setdefault(synonym, 0.55)
    return weights


def _norm(text: str) -> str:
    return " ".join(_WORD_RE.findall(str(text or "").lower()))


def _word_set(text: str) -> set[str]:
    """Exact tokens plus the pieces of identifiers (vram_gb -> vram, gb)."""
    words = set(_norm(text).split())
    parts: set[str] = set()
    for word in words:
        parts.update(re.split(r"[_\-./]", word))
    return words | {p for p in parts if p}


# --------------------------------------------------------------- CLI reference

COMMON_FLAGS = [
    {"flag": "PATH...", "help": "files or directories to index"},
    {"flag": "--exclude PATTERN",
     "help": "gitignore-style pattern to skip; repeatable"},
    {"flag": "--include PATTERN",
     "help": "re-include something an ignore rule excluded; repeatable"},
    {"flag": "--ext .EXT",
     "help": "extra extension to index as a reference file"},
    {"flag": "--no-refs",
     "help": "Python only; skip .md/.csv/.json/.toml and friends"},
    {"flag": "--no-gitignore",
     "help": "do not read .gitignore (.pedacitoignore is still read)"},
    {"flag": "--url URL",
     "help": "server base URL, e.g. http://localhost:1234/v1"},
    {"flag": "--model ID", "help": "model id/name as the server reports it"},
    {"flag": "--root DIR", "help": "project root (default: inferred)"},
    {"flag": "--config FILE", "help": "use this config file as well"},
    {"flag": "--profile NAME", "help": "apply a [profiles.NAME] block"},
    {"flag": "--steps N",
     "help": "hard lookup limit for gather; 0 = no limit (the default)"},
    {"flag": "--max-tokens-budget N",
     "help": "stop before the next call at N prompt+completion tokens"},
    {"flag": "--role ROLE=PROFILE",
     "help": "route one job to another profile; repeatable"},
    {"flag": "--record",
     "help": "save model responses to .pedacito/replay/<stamp>.jsonl"},
    {"flag": "--replay FILE",
     "help": "run against a recording instead of a model server"},
    {"flag": "--strict-replay",
     "help": "with --replay: exit 1 when prompts differ from the recording"},
    {"flag": "--strict",
     "help": "exit 1 when a failed edit block or impacted test remains"},
    {"flag": "--jsonl",
     "help": "one JSON event per line on stdout; human text on stderr"},
    {"flag": "--max-cost USD",
     "help": "stop before the next call at this estimated spend"},
]

CLI_COMMANDS = [
    {
        "name": "index",
        "usage": "pedacito index PROJECT [--no-summaries | --resummarise]",
        "summary": "Build or refresh the project index (map + summaries) that "
                   "ask, explain and auto use.",
        "keywords": ["build index", "refresh index", "map", "summaries"],
        "examples": [
            "pedacito index .",
            "pedacito index . --no-summaries",
            "pedacito index . --resummarise",
        ],
        "flags": [
            {"flag": "--no-summaries",
             "help": "static index only; instant, no model calls"},
            {"flag": "--resummarise, --resummarize",
             "help": "discard cached summaries and regenerate them"},
        ],
        "common": True,
    },
    {
        "name": "ask",
        "usage": 'pedacito ask PROJECT -t "TASK" [--apply | --review]',
        "summary": "Run one coding task end to end. Without --apply it is a "
                   "dry run that prints the proposed edits.",
        "keywords": ["task", "edit", "apply", "write", "review", "dry run"],
        "examples": [
            'pedacito ask . -t "add retries to the sync client"',
            'pedacito ask . -t "add retries" --apply',
            'pedacito ask . -t "add retries" --review',
            'pedacito ask . -t "refactor pricing and billing" --plan --review-plan',
            'pedacito ask . -t "finish the previous task" --continue',
        ],
        "flags": [
            {"flag": "-t, --task TEXT", "help": "the task (required)"},
            {"flag": "--apply", "help": "write edits to disk (backup first)"},
            {"flag": "--review",
             "help": "open the side-by-side review page before writing"},
            {"flag": "--no-review",
             "help": "never open the reviewer, even if config enables it"},
            {"flag": "--reindex", "help": "force a fresh index first"},
            {"flag": "--refine",
             "help": "expand a one-line idea into a repo-aware spec first"},
            {"flag": "--continue",
             "help": "resume the newest ask checkpoint as context"},
            {"flag": "--plan",
             "help": "split a multi-part task into fresh-context steps"},
            {"flag": "--no-plan", "help": "never auto-plan this task"},
            {"flag": "--review-plan",
             "help": "show the plan and ask before running it"},
        ],
        "common": True,
    },
    {
        "name": "explain",
        "usage": 'pedacito explain PROJECT -q "QUESTION"',
        "summary": "Answer questions about an already-indexed project in prose. "
                   "Read-only: no edits.",
        "keywords": ["question", "read only", "ask about code", "understand"],
        "examples": [
            'pedacito explain . -q "how does the retry backoff work?"',
            'pedacito explain .',
        ],
        "flags": [
            {"flag": "-q, --question TEXT",
             "help": "your question; omit for an interactive prompt"},
        ],
        "common": True,
    },
    {
        "name": "auto",
        "usage": 'pedacito auto PROJECT -t "GOAL" [--apply] [--verify CMD]',
        "summary": "Turn a goal into a checklist, then implement each task in "
                   "a fresh context. Without --apply it only writes/prints the "
                   "plan.",
        "keywords": ["autonomous", "plan", "checklist", "goal", "loop"],
        "examples": [
            'pedacito auto . -t "add CSV export with tests"',
            'pedacito auto . -t "add CSV export" --apply --verify "pytest -q"',
            "pedacito auto . --plan PLAN.md --apply --max-tasks 5",
            'pedacito auto . -t "risky refactor" --apply --worktree risky',
        ],
        "flags": [
            {"flag": "-t, --task TEXT", "help": "one-line goal"},
            {"flag": "--spec FILE", "help": "larger spec document"},
            {"flag": "--plan FILE",
             "help": "use this checklist (default: PLAN.md)"},
            {"flag": "--apply", "help": "implement the tasks and write edits"},
            {"flag": "--verify CMD",
             "help": "run this command after every task; failures repair"},
            {"flag": "--repair N",
             "help": "fix rounds per failing verification"},
            {"flag": "--diagnostics",
             "help": "run the detected linter/typechecker after each task"},
            {"flag": "--worktree NAME",
             "help": "run in an isolated git worktree"},
            {"flag": "--max-tasks N", "help": "stop after N tasks"},
            {"flag": "--max-minutes N", "help": "stop after N minutes"},
            {"flag": "--keep-going",
             "help": "continue past failed tasks instead of stopping"},
            {"flag": "--auto-install",
             "help": "pip-install a missing Python package and retry"},
            {"flag": "--force",
             "help": "regenerate the plan, discarding the existing one"},
            {"flag": "--refine",
             "help": "expand a one-line idea into a repo-aware spec first"},
        ],
        "common": True,
    },
    {
        "name": "new",
        "usage": 'pedacito new DIR -t "PROJECT IDEA" [--apply]',
        "summary": "Start a project from scratch: plan, scaffold, then build. "
                   "The default is a dry run.",
        "keywords": ["scaffold", "create project", "start", "generate"],
        "examples": [
            'pedacito new my-app -t "CLI todo manager with sqlite" --plan-only',
            'pedacito new my-app -t "CLI todo manager" --apply',
            'pedacito new my-app -t "CLI todo manager" --apply --review',
            'pedacito new my-app -t "todo app" --apply --complex --verify "pytest -q"',
        ],
        "flags": [
            {"flag": "DIR",
             "help": "new project directory (created; must be empty)"},
            {"flag": "-t, --task TEXT", "help": "one-line project description"},
            {"flag": "--spec FILE", "help": "larger spec (.md/.txt)"},
            {"flag": "--refine",
             "help": "expand a one-line idea into a repo-aware spec first"},
            {"flag": "--plan-only",
             "help": "stop after the plan and header skeleton"},
            {"flag": "--design",
             "help": "design interfaces, data shapes and tests first"},
            {"flag": "--apply",
             "help": "write the generated files (default: dry run)"},
            {"flag": "--review",
             "help": "build, then review everything before writing"},
            {"flag": "--force",
             "help": "allow scaffolding into a non-empty directory"},
            {"flag": "--complex",
             "help": "hand implementation to the autonomous loop"},
            {"flag": "--verify CMD",
             "help": "with --complex: command run after each task"},
            {"flag": "--repair N",
             "help": "with --complex: fix rounds per failed verification"},
            {"flag": "--max-tasks N", "help": "with --complex: task cap"},
            {"flag": "--max-minutes N", "help": "with --complex: minute cap"},
            {"flag": "--keep-going",
             "help": "with --complex: continue past failures"},
            {"flag": "--auto-install",
             "help": "with --complex --verify: pip-install missing modules"},
            {"flag": "--no-review", "help": "never open the reviewer"},
            {"flag": "--url URL", "help": "server base URL override"},
            {"flag": "--model ID", "help": "model id override"},
            {"flag": "--config FILE", "help": "use this config file as well"},
            {"flag": "--profile NAME", "help": "apply a profile block"},
        ],
    },
    {
        "name": "chat",
        "usage": "pedacito chat PROJECT [--apply] [--resume]",
        "summary": "Interactive session: keep the index warm and switch "
                   "profiles without reloading the project.",
        "keywords": ["interactive", "repl", "session", "slash commands"],
        "examples": [
            "pedacito chat . --apply",
            "pedacito chat . --review",
            "pedacito chat . --resume",
        ],
        "flags": [
            {"flag": "--apply", "help": "write edits to disk"},
            {"flag": "--review", "help": "open the review page before writing"},
            {"flag": "--no-review", "help": "never open the reviewer"},
            {"flag": "--resume",
             "help": "seed the first task with the newest chat checkpoint"},
        ],
        "notes": [
            "/profile NAME switches model profile; /model ID switches model",
            "/apply on|off toggles writes; /steps N sets the lookup cap",
            "/status, /cost, /memory, /skills, /profiles, /quit",
        ],
        "common": True,
    },
    {
        "name": "config",
        "usage": "pedacito config [--init | --setup | --check | --doctor | "
                 "--reference]",
        "summary": "Show every resolved setting and where it came from; create, "
                   "test or print the config.",
        "keywords": ["config file", "config.toml", "setup", "check", "doctor",
                     "reference", "settings"],
        "examples": [
            "pedacito config",
            "pedacito config --init",
            "pedacito config --init --force",
            "pedacito config --setup",
            "pedacito config --check",
            "pedacito config --doctor",
            "pedacito config --reference",
            "pedacito config --profile big",
        ],
        "flags": [
            {"flag": "--init",
             "help": f"write a starter config to the user config path"},
            {"flag": "--path FILE",
             "help": "with --init/--reference: use this file"},
            {"flag": "--force", "help": "overwrite an existing file"},
            {"flag": "--check",
             "help": "connect to the server and verify the model loads"},
            {"flag": "--doctor",
             "help": "like --check plus structured-output and context probes"},
            {"flag": "--setup",
             "help": "interactive first-run setup"},
            {"flag": "--reference",
             "help": "print (or write with --path) the annotated template"},
            {"flag": "--root DIR", "help": "project root"},
            {"flag": "--profile NAME", "help": "show one profile's resolution"},
            {"flag": "--config FILE", "help": "use this config file as well"},
        ],
    },
    {
        "name": "calibrate",
        "usage": "pedacito calibrate [--quick] [--recalibrate] [--repeats N]",
        "summary": "Measure the configured model (tool choice, edit format, "
                   "honesty, long context) and save a model card that tunes the "
                   "defaults.",
        "keywords": ["model card", "measure", "benchmark", "tune", "quick"],
        "examples": [
            "pedacito calibrate",
            "pedacito calibrate --quick",
            "pedacito calibrate --recalibrate --repeats 3",
            "pedacito calibrate --profile big",
        ],
        "flags": [
            {"flag": "--recalibrate",
             "help": "measure again even if a fresh card exists"},
            {"flag": "--quick",
             "help": "skip the long-context and capability probes"},
            {"flag": "--repeats N",
             "help": "run each probe N times (default 3; 1 with --quick)"},
            {"flag": "--url URL", "help": "server base URL override"},
            {"flag": "--model ID", "help": "model id override"},
            {"flag": "--config FILE", "help": "use this config file as well"},
            {"flag": "--profile NAME", "help": "apply a profile block"},
            {"flag": "--root DIR", "help": "project root"},
        ],
    },
    {
        "name": "models scan",
        "usage": "pedacito models scan [--url URL] [--match PATTERN] "
                 "[--calibrate none|quick|full]",
        "summary": "Discover local model servers and/or probe a hosted API, "
                   "write one profile per selected model, then optionally "
                   "calibrate them.",
        "keywords": ["import models", "model discovery", "profiles", "scan",
                     "lm studio", "ollama", "hosted"],
        "examples": [
            "pedacito models scan",
            "pedacito models scan --url http://localhost:1234/v1 --vram-gb 12",
            "pedacito models scan --match qwen --all -y",
            'pedacito models scan --url https://api.example.com/v1 '
            '--api-key-env OPENAI_API_KEY --match "gpt-*" --calibrate none',
        ],
        "flags": [
            {"flag": "scan", "help": "the only action: discover and import"},
            {"flag": "--url URL", "help": "server base URL to probe; repeatable"},
            {"flag": "--api-key-env VAR",
             "help": "environment variable holding a hosted API key"},
            {"flag": "--match PATTERN",
             "help": "only models whose id matches (fnmatch)"},
            {"flag": "--all", "help": "include models the server has not loaded"},
            {"flag": "--loaded", "help": "only models the server reports loaded"},
            {"flag": "--calibrate none|quick|full",
             "help": "calibrate selected models (local default: quick)"},
            {"flag": "-y, --yes",
             "help": "non-interactive: select all matching models"},
            {"flag": "--vram-gb GB",
             "help": "GPU VRAM of the server machine (asked if omitted)"},
            {"flag": "--config FILE",
             "help": "write profiles to this file (default: user config)"},
        ],
    },
    {
        "name": "profiles",
        "usage": "pedacito profiles [--recommend | --json] [--apply-roles]",
        "summary": "List every model profile with its measured harness-fit, "
                   "context and card status; recommend one per role.",
        "keywords": ["list profiles", "models", "recommend", "roles", "json"],
        "examples": [
            "pedacito profiles",
            "pedacito profiles --recommend",
            "pedacito profiles --recommend --apply-roles",
            "pedacito profiles --json",
        ],
        "flags": [
            {"flag": "--recommend",
             "help": "suggest a profile per role, with reasons"},
            {"flag": "--apply-roles",
             "help": "write the recommended roles = {...} (asks first)"},
            {"flag": "--json", "help": "print profiles as JSON"},
            {"flag": "-y, --yes",
             "help": "with --apply-roles: do not ask for confirmation"},
            {"flag": "--config FILE", "help": "use this config file as well"},
            {"flag": "--profile NAME", "help": "apply a profile block"},
        ],
    },
    {
        "name": "rename",
        "usage": "pedacito rename OLD NEW [--file PATH] [--apply] [--root DIR]",
        "summary": "Rename a symbol across the project without a model call: "
                   "Python identifiers move through the real tokenizer, "
                   "JavaScript/TypeScript/Go/Rust through tree-sitter "
                   "identifier nodes, so strings and comments that merely "
                   "contain the old name are never touched. Default is a "
                   "dry-run diff preview; --apply writes and snapshots the "
                   "originals, so restore undoes it.",
        "keywords": ["rename", "refactor", "symbol", "identifier", "variable",
                     "function name", "rename across files"],
        "examples": [
            "pedacito rename apply_discount discounted_price",
            "pedacito rename apply_discount discounted_price --file pkg/core.py",
            "pedacito rename apply_discount discounted_price --apply --root .",
            "pedacito restore . --dry-run",
        ],
        "flags": [
            {"flag": "OLD", "help": "the existing identifier"},
            {"flag": "NEW", "help": "the new identifier"},
            {"flag": "--file PATH",
             "help": "scope: that file plus the files importing it "
                     "(default: all indexed project source)"},
            {"flag": "--apply",
             "help": "write the rename (default: dry-run with diffs)"},
            {"flag": "--root DIR", "help": "project root (default: cwd)"},
        ],
    },
    {
        "name": "restore",
        "usage": "pedacito restore PROJECT [--list | --snapshot NAME] [--dry-run]",
        "summary": "Undo an --apply run by copying a backup snapshot back over "
                   "the working tree.",
        "keywords": ["undo", "backup", "revert", "rollback", "snapshot",
                     "restore"],
        "examples": [
            "pedacito restore . --list",
            "pedacito restore . --dry-run",
            "pedacito restore .",
            "pedacito restore . --snapshot 20260830-020724",
            "pedacito restore . --delete-created",
        ],
        "flags": [
            {"flag": "PATH...", "help": "the project (default: .)"},
            {"flag": "--snapshot NAME",
             "help": "which snapshot; default is the newest"},
            {"flag": "--list", "help": "list snapshots and exit"},
            {"flag": "--dry-run", "help": "show what would be restored"},
            {"flag": "--delete-created",
             "help": "also delete files that run created"},
            {"flag": "--root DIR", "help": "project root"},
            {"flag": "--config FILE", "help": "use this config file as well"},
        ],
    },
    {
        "name": "reset",
        "usage": "pedacito reset PROJECT [--dry-run] [--keep-memory] "
                 "[--reindex]",
        "summary": "Wipe this project's Pedacito state (index, plans, caches, "
                   "sessions, backups); code and config files are untouched. "
                   "This is not a config reset.",
        "keywords": ["wipe project state", "clear index", "clean", "start over",
                     "not config"],
        "examples": [
            "pedacito reset . --dry-run",
            "pedacito reset . --yes",
            "pedacito reset . --keep-memory --keep-config",
            "pedacito reset . --reindex",
        ],
        "flags": [
            {"flag": "PATH...", "help": "the project (default: .)"},
            {"flag": "--root DIR", "help": "project root"},
            {"flag": "--dry-run", "help": "show what would be removed"},
            {"flag": "-y, --yes", "help": "do not ask for confirmation"},
            {"flag": "--keep-memory", "help": "keep .pedacito/memory.md"},
            {"flag": "--keep-config, --keep-authored",
             "help": "keep authored rules, skills, prompts and instructions"},
            {"flag": "--force",
             "help": "also remove git worktrees under .pedacito/worktrees"},
            {"flag": "--reindex", "help": "rebuild the index after the wipe"},
            {"flag": "--no-summaries",
             "help": "with --reindex: static index only (no model calls)"},
            {"flag": "--url URL", "help": "server URL (with --reindex)"},
            {"flag": "--model ID", "help": "model id (with --reindex)"},
            {"flag": "--config FILE", "help": "use this config file as well"},
            {"flag": "--profile NAME", "help": "apply a profile block"},
        ],
    },
    {
        "name": "prune",
        "usage": "pedacito prune PROJECT [--keep-sessions N] [--keep-backups N] "
                 "[--dry-run]",
        "summary": "Delete old sessions, backups and event files and trim "
                   "log.txt.",
        "keywords": ["cleanup", "retention", "delete old", "logs", "backups"],
        "examples": [
            "pedacito prune . --dry-run",
            "pedacito prune . --keep-sessions 50 --keep-backups 20",
        ],
        "flags": [
            {"flag": "PATH...", "help": "the project (default: .)"},
            {"flag": "--keep-sessions N", "help": "keep the newest N sessions"},
            {"flag": "--keep-backups N", "help": "keep the newest N backups"},
            {"flag": "--dry-run", "help": "show what would be deleted"},
            {"flag": "--root DIR", "help": "project root"},
            {"flag": "--config FILE", "help": "use this config file as well"},
            {"flag": "--profile NAME", "help": "apply a profile block"},
        ],
    },
    {
        "name": "eval",
        "usage": "pedacito eval [--task ID] [--out FILE] [--baseline FILE]",
        "summary": "Score the harness on built-in fixture repos (locate "
                   "precision, edit success, tokens, cache hits).",
        "keywords": ["benchmark", "score", "harness", "baseline", "regression"],
        "examples": [
            "pedacito eval --list",
            "pedacito eval --out report.json",
            "pedacito eval --task locate_symbol --task edit_small",
            "pedacito eval --baseline old-report.json --max-token-growth 0.2",
        ],
        "flags": [
            {"flag": "--task ID", "help": "run only this task id; repeatable"},
            {"flag": "--out FILE", "help": "write the JSON report here"},
            {"flag": "--baseline FILE",
             "help": "A/B gate against an earlier --out report"},
            {"flag": "--max-token-growth FRAC",
             "help": "allowed token growth vs baseline (default 0.20)"},
            {"flag": "--max-time-growth FRAC",
             "help": "allowed wall-time growth vs baseline (default 0.50)"},
            {"flag": "--repeats N",
             "help": "run the whole task set N times"},
            {"flag": "--seed S", "help": "base sampler seed for repeats"},
            {"flag": "--list", "help": "list tasks and exit"},
            {"flag": "--url URL", "help": "server URL"},
            {"flag": "--model ID", "help": "model id"},
            {"flag": "--config FILE", "help": "use this config file"},
            {"flag": "--profile NAME", "help": "apply a profile block"},
            {"flag": "--root DIR", "help": "project root"},
        ],
    },
    {
        "name": "gui",
        "usage": "pedacito gui [PROJECT] [--profile NAME] [--port N] [--no-open]",
        "summary": "Open the local browser control panel (127.0.0.1 only).",
        "keywords": ["browser", "interface", "ui", "control panel", "web"],
        "examples": [
            "pedacito gui",
            "pedacito gui .",
            "pedacito gui /path/to/project --profile qwen --port 8756",
            "pedacito gui . --no-open",
        ],
        "flags": [
            {"flag": "PROJECT",
             "help": "project folder to open at startup"},
            {"flag": "--profile NAME", "help": "model profile for that project"},
            {"flag": "--port N", "help": "fixed port (default: OS-chosen)"},
            {"flag": "--no-open",
             "help": "print the URL instead of opening a browser"},
        ],
    },
    {
        "name": "map",
        "usage": "pedacito map PROJECT [--tier auto|cards|dirs|flat|ranked] "
                 "[-t TASK]",
        "summary": "Print the index/map exactly as the model would receive it.",
        "keywords": ["index map", "print map", "tiers"],
        "examples": [
            "pedacito map .",
            "pedacito map . --tier cards",
            'pedacito map . -t "fix the retry policy"',
        ],
        "flags": [
            {"flag": "--tier auto|cards|dirs|flat|ranked",
             "help": "map rendering tier"},
            {"flag": "-t, --task TEXT",
             "help": "rank the map for this task"},
        ],
        "common": True,
    },
    {
        "name": "files",
        "usage": "pedacito files PROJECT",
        "summary": "Show which files would be indexed, and why (ignored files "
                   "and reasons included).",
        "keywords": ["select files", "ignored", "what is indexed"],
        "examples": ["pedacito files ."],
        "flags": [],
        "common": True,
    },
    {
        "name": "cost",
        "usage": "pedacito cost PROJECT [--tps 25]",
        "summary": "Estimate indexing cost without spending it.",
        "keywords": ["cost", "tokens", "estimate", "price"],
        "examples": ["pedacito cost . --tps 25"],
        "flags": [
            {"flag": "--tps N", "help": "your tokens per second (default 25)"},
        ],
        "common": True,
    },
    {
        "name": "how",
        "usage": 'pedacito how "QUESTION" [--offline] [--json]',
        "summary": "Ask how to use Pedacito. Searches the built-in CLI, GUI "
                   "and setting reference; works without a model with "
                   "--offline.",
        "keywords": ["help", "documentation", "how to", "guide", "offline"],
        "examples": [
            'pedacito how "how do I disable thinking?"',
            'pedacito how --offline "how do I import models?"',
            'pedacito how -q "how do I reset my config?" --json',
        ],
        "flags": [
            {"flag": "QUESTION",
             "help": "your question; omit for an interactive prompt"},
            {"flag": "-q, --question TEXT", "help": "alternative to positional"},
            {"flag": "--offline, --no-model, --fast",
             "help": "deterministic answer only; no model server needed"},
            {"flag": "--json",
             "help": "print structured matches plus the formatted answer"},
            {"flag": "--url URL", "help": "server base URL override"},
            {"flag": "--api-key KEY", "help": "API key override"},
            {"flag": "--model ID", "help": "model id override"},
            {"flag": "--vram GB", "help": "GPU VRAM in whole gigabytes"},
            {"flag": "--config FILE", "help": "use this config file as well"},
            {"flag": "--profile NAME", "help": "apply a profile block"},
        ],
    },
    {
        "name": "worktree",
        "usage": "pedacito worktree create|list|remove NAME [--base REV]",
        "summary": "Create, list or remove an isolated git worktree for risky "
                   "runs.",
        "keywords": ["git worktree", "isolated branch", "sandbox branch"],
        "examples": [
            "pedacito worktree create risky",
            "pedacito worktree list",
            "pedacito worktree remove risky --force",
        ],
        "flags": [
            {"flag": "create|list|remove", "help": "the action"},
            {"flag": "NAME", "help": "worktree name"},
            {"flag": "--root DIR", "help": "the repository"},
            {"flag": "--base REV",
             "help": "commit/branch for a new worktree (default HEAD)"},
            {"flag": "--force",
             "help": "with remove: discard modified/untracked files"},
        ],
    },
    {
        "name": "diagnose",
        "usage": "pedacito diagnose [--root DIR] [--file FILE]",
        "summary": "Run the project's detected linter/typechecker.",
        "keywords": ["lint", "typecheck", "static analysis"],
        "examples": ["pedacito diagnose .", "pedacito diagnose . --file app.py"],
        "flags": [
            {"flag": "--root DIR", "help": "the project"},
            {"flag": "--file FILE", "help": "restrict to one project file"},
        ],
    },
    {
        "name": "serve",
        "usage": "pedacito serve [--name NAME] [--wait SECONDS] -- COMMAND...",
        "summary": "Start a background dev server, keep it alive, capture its "
                   "log, and stop it later with procs.",
        "keywords": ["dev server", "background process", "run server"],
        "examples": [
            'pedacito serve --name web -- python -m http.server 8000',
            "pedacito serve -- npm run dev",
        ],
        "flags": [
            {"flag": "COMMAND...", "help": "the command to run"},
            {"flag": "--name NAME",
             "help": "registry name (default: derived from the command)"},
            {"flag": "--wait SECONDS",
             "help": "seconds to wait for a printed URL"},
            {"flag": "--root DIR", "help": "project root"},
            {"flag": "--config FILE", "help": "use this config file"},
            {"flag": "--profile NAME", "help": "apply a profile block"},
        ],
    },
    {
        "name": "procs",
        "usage": "pedacito procs [list | logs NAME | stop NAME] [--tail N]",
        "summary": "List, show logs from, or stop background servers started "
                   "with serve or the model's `serve` action.",
        "keywords": ["background processes", "server logs", "stop server"],
        "examples": [
            "pedacito procs",
            "pedacito procs logs web --tail 2000",
            "pedacito procs stop web",
        ],
        "flags": [
            {"flag": "list|logs|stop", "help": "the action (default: list)"},
            {"flag": "NAME", "help": "registry name"},
            {"flag": "--tail N",
             "help": "with logs: last N characters (default 4000)"},
            {"flag": "--root DIR", "help": "project root"},
            {"flag": "--config FILE", "help": "use this config file"},
            {"flag": "--profile NAME", "help": "apply a profile block"},
        ],
    },
    {
        "name": "lsp",
        "usage": "pedacito lsp --file FILE [--op definition|references|hover|"
                 "diagnostics]",
        "summary": "Exact symbols/diagnostics through an optional language "
                   "server (allow_lsp).",
        "keywords": ["language server", "symbols", "definition", "hover"],
        "examples": [
            "pedacito lsp --file app.py --op definition --line 12 --column 5",
        ],
        "flags": [
            {"flag": "--file FILE", "help": "project-relative file"},
            {"flag": "--op OP",
             "help": "definition, references, hover or diagnostics"},
            {"flag": "--line N", "help": "line (default 1)"},
            {"flag": "--column N", "help": "column (default 1)"},
            {"flag": "--root DIR", "help": "project root"},
            {"flag": "--config FILE", "help": "use this config file"},
            {"flag": "--profile NAME", "help": "apply a profile block"},
        ],
    },
    {
        "name": "memory",
        "usage": "pedacito memory PROJECT [--add TEXT | --forget TEXT]",
        "summary": "List, add or forget the facts Pedacito remembers about a "
                   "project.",
        "keywords": ["remember", "facts", "notes", "memory.md"],
        "examples": [
            "pedacito memory .",
            'pedacito memory . --add "tests run with pytest -q"',
            'pedacito memory . --forget "old fact"',
            "pedacito memory . --global",
        ],
        "flags": [
            {"flag": "--add TEXT", "help": "remember a fact"},
            {"flag": "--forget TEXT", "help": "remove a fact"},
            {"flag": "--global",
             "help": "use the user-level memory instead of the project's"},
            {"flag": "--root DIR", "help": "project root"},
            {"flag": "--config FILE", "help": "use this config file"},
            {"flag": "--profile NAME", "help": "apply a profile block"},
        ],
    },
    {
        "name": "trust",
        "usage": "pedacito trust PROJECT [--yes]",
        "summary": "Approve this project's .pedacito.toml so its hooks, MCP "
                   "servers and verify commands may run.",
        "keywords": ["project config", "approve", "execution", "security"],
        "examples": ["pedacito trust .", "pedacito trust . --yes"],
        "flags": [
            {"flag": "--yes", "help": "do not ask for confirmation"},
            {"flag": "--root DIR", "help": "project root"},
            {"flag": "--config FILE", "help": "use this config file"},
            {"flag": "--profile NAME", "help": "apply a profile block"},
        ],
    },
    {
        "name": "tests",
        "usage": "pedacito tests PROJECT [--files FILE...]",
        "summary": "Show the detected test command and which tests a change to "
                   "the given files can break.",
        "keywords": ["test command", "impacted tests", "pytest"],
        "examples": [
            "pedacito tests .",
            "pedacito tests . --files src/app.py",
        ],
        "flags": [
            {"flag": "--files FILE...", "help": "changed files"},
            {"flag": "--root DIR", "help": "project root"},
            {"flag": "--config FILE", "help": "use this config file"},
            {"flag": "--profile NAME", "help": "apply a profile block"},
        ],
    },
    {
        "name": "webcheck",
        "usage": "pedacito webcheck PROJECT [--files FILE...]",
        "summary": "Static contract checks for a web project: DOM ids, ESM "
                   "syntax, duplicate globals, undeclared imports.",
        "keywords": ["web", "javascript", "dom", "html check"],
        "examples": ["pedacito webcheck ."],
        "flags": [
            {"flag": "--files FILE...", "help": "changed files"},
            {"flag": "--root DIR", "help": "project root"},
            {"flag": "--config FILE", "help": "use this config file"},
            {"flag": "--profile NAME", "help": "apply a profile block"},
        ],
    },
    {
        "name": "skills",
        "usage": "pedacito skills PROJECT",
        "summary": "List the available skills (project, user and bundled "
                   "starters).",
        "keywords": ["skills", "capabilities", "catalog"],
        "examples": ["pedacito skills ."],
        "flags": [
            {"flag": "--root DIR", "help": "project root"},
            {"flag": "--config FILE", "help": "use this config file"},
            {"flag": "--profile NAME", "help": "apply a profile block"},
        ],
    },
]

# --------------------------------------------------------------- GUI map ------

GUI_TOPICS = [
    {
        "id": "switch-model",
        "title": "Switch the active model or profile",
        "summary": "Pick another profile in the topbar and click Switch; the "
                   "open project reloads against that model.",
        "steps": [
            "Topbar Row 2 → Profile dropdown → choose a profile → click Switch",
            "Or ⚙ Settings → Profiles → click a profile → Make default",
            "⚙ Settings → Model discovery imports new models; Profiles edits "
            "the existing ones",
        ],
        "keywords": ["switch model", "change model", "choose model", "profile",
                     "active profile", "default profile", "provider"],
        "settings": ["default_profile", "model", "base_url", "tier"],
        "cli": ["profiles", "models scan", "chat"],
    },
    {
        "id": "import-models",
        "title": "Import models from a local server",
        "summary": "Scan the local servers, tick the models you want, and "
                   "import them as profiles.",
        "steps": [
            "Topbar → ⚙ Settings → Model discovery",
            "Click Scan local servers (or type another server URL and its "
            "VRAM GB)",
            "Tick the models to import, optionally choose a default radio, "
            "then Import Selected (N)",
            "Edit the new entries under ⚙ Settings → Profiles",
        ],
        "keywords": ["import models", "scan models", "discover models",
                     "lm studio", "ollama", "add model", "model discovery"],
        "settings": ["base_url", "model", "vram_gb", "default_profile"],
        "cli": ["models scan", "calibrate", "profiles"],
    },
    {
        "id": "disable-thinking",
        "title": "Disable model thinking / reasoning",
        "summary": "Turn thinking off per profile in Settings; the topbar "
                   "model chip then shows no-think.",
        "steps": [
            "Topbar → ⚙ Settings → Profiles",
            "Select the profile in the left list (the active one is tagged "
            "active)",
            "Reasoning & Execution Overrides → disable_thinking → choose on",
            "Optional: set thinking_suffix to /no_think for models that use "
            "the soft switch",
            "Click Save changes; the topbar chip changes from thinking to "
            "no-think",
        ],
        "keywords": ["disable thinking", "turn off thinking", "reasoning",
                     "no think", "no_think", "disable_thinking", "toggle "
                     "reasoning"],
        "settings": ["disable_thinking", "thinking_suffix"],
        "cli": ["config", "calibrate"],
    },
    {
        "id": "reset-config",
        "title": "Reset settings / config to the baseline",
        "summary": "Reset the user config to pedacito's recommended defaults; "
                   "the old file is backed up first.",
        "steps": [
            "Topbar → ⚙ Settings → Raw file / reset",
            "Make sure Editing: User config is selected",
            "Click Reset to baseline defaults (Reset to Baseline) and confirm",
            "The old file is saved as config.toml.reset.<timestamp>.bak; "
            "project settings are not touched",
        ],
        "keywords": ["reset settings", "reset config", "reset my config",
                     "restore defaults", "baseline", "reset to baseline",
                     "wipe config", "start over"],
        "settings": ["default_profile", "base_url", "model", "vram_gb"],
        "cli": ["config", "reset"],
    },
    {
        "id": "undo-run",
        "title": "Undo an Auto or Edit run / restore a backup",
        "summary": "Every write is snapshotted first; restore the newest or a "
                   "chosen snapshot from the Workspace panel.",
        "steps": [
            "Bottom-right Workspace panel → Backups tab",
            "Click Restore on the snapshot you want (or Restore last backup)",
            "Confirm; files changed since then are overwritten, files created "
            "since stay on disk",
            "Use the CLI (restore --delete-created) if you also want the "
            "run's new files removed",
        ],
        "keywords": ["undo", "restore", "backup", "revert", "rollback",
                     "undo auto", "undo edit", "snapshot"],
        "settings": ["backup", "keep_backups"],
        "cli": ["restore", "prune"],
    },
    {
        "id": "rename-symbol",
        "title": "Rename a symbol across the project",
        "summary": "Project ▾ → Rename symbol… renames every exact identifier "
                   "deterministically (no model tokens): strings and comments "
                   "that merely contain the old name are left alone.",
        "steps": [
            "Project ▾ → Rename symbol… (needs a project open and no run in "
            "progress)",
            "Type the current name, the new name, and optionally one scope file",
            "Preview shows the per-file unified diffs; nothing is written yet",
            "Rename writes the change and snapshots the originals to "
            ".pedacito/backups/",
            "Undo it with Project ▾ → Restore from a backup… (or the CLI "
            "restore)",
            "Without a scope file every indexed source file is considered; "
            "with one, only that file and the files importing it",
        ],
        "keywords": ["rename", "refactor", "symbol", "identifier",
                     "rename symbol", "variable", "function name",
                     "rename across files"],
        "settings": ["allow_write", "backup"],
        "cli": ["rename", "restore"],
    },
    {
        "id": "reference-file",
        "title": "Reference a file in chat",
        "summary": "Insert an @path mention from the Files tab, or type it "
                   "directly in the composer.",
        "steps": [
            "Bottom-right Workspace panel → Files tab",
            "Click @ next to any file to insert @path/to/file into the chat "
            "input",
            "Or type @path/to/file directly in the message",
            "Click a file name to preview it; the preview has Use in chat too",
        ],
        "keywords": ["reference file", "mention file", "at file", "@", "attach",
                     "include file", "file in chat"],
        "settings": [],
        "cli": ["ask", "explain"],
    },
    {
        "id": "index-project",
        "title": "Build or refresh the index",
        "summary": "The topbar Index button builds the map; Shift-click builds "
                   "a quick index with no model summaries.",
        "steps": [
            "Topbar → Index (Shift-click for Quick, no summaries)",
            "Watch the index chip for files/units and the index timestamp",
            "Ask, Task/Edit and Auto need an index; New does not",
        ],
        "keywords": ["index", "build index", "refresh index", "index a "
                     "project", "map", "indexed", "no index"],
        "settings": ["include_references", "summarise_references",
                     "summary_batch"],
        "cli": ["index", "map", "files"],
    },
    {
        "id": "review-changes",
        "title": "Review changes before they are written",
        "summary": "Task/Edit in Review mode stages a diff in the middle pane; "
                   "accept or reject it there.",
        "steps": [
            "Composer → Task/Edit → Options → Writing: Review before writing",
            "Run the task; the Chat header shows Review staged: expand (or "
            "the middle Changes pane opens)",
            "Untick files to accept only some; use ✎ Edit proposed or "
            "Side by side / Unified",
            "Click Accept (or Accept N of M) / Reject",
        ],
        "keywords": ["review", "diff", "accept", "reject", "staged changes",
                     "before writing"],
        "settings": ["review", "backup", "atomic_edits"],
        "cli": ["ask"],
    },
    {
        "id": "write-direct",
        "title": "Write directly without the review page",
        "summary": "Choose direct writes in Task/Edit options, or pass --apply "
                   "on the CLI; backups are still taken first.",
        "steps": [
            "Composer → Task/Edit → Options → Writing: Write directly (backup "
            "first)",
            "Run the task; edits are applied and listed in the chat",
            "Bottom-right Workspace → Backups if you need to undo them",
        ],
        "keywords": ["write directly", "no review", "apply", "skip review",
                     "auto apply"],
        "settings": ["review", "backup", "atomic_edits"],
        "cli": ["ask"],
    },
    {
        "id": "run-auto",
        "title": "Run an autonomous Auto task",
        "summary": "Give Auto a goal; it writes a checklist and works through "
                   "it, optionally verifying every task.",
        "steps": [
            "Composer → Auto → describe the goal",
            "Options: Plan only, existing plan file, verify command, repair "
            "rounds, max tasks/minutes, git worktree, keep going, pip-install, "
            "diagnostics, force new plan",
            "Click Apply & run; the Plan panel shows progress and lets you "
            "Edit or Run the checklist",
        ],
        "keywords": ["auto", "autonomous", "plan", "checklist", "goal",
                     "verify", "worktree", "keep going"],
        "settings": ["auto_verify", "auto_repair_rounds", "auto_max_tasks",
                     "auto_max_minutes", "auto_install_deps",
                     "auto_diagnostics", "auto_smoke",
                     "auto_green_ratchet"],
        "cli": ["auto", "worktree"],
    },
    {
        "id": "plan-only",
        "title": "Only generate or edit the plan",
        "summary": "Tick Plan only in Auto options, or omit --apply on the "
                   "CLI; edit the checklist in the Plan panel.",
        "steps": [
            "Composer → Auto → Options → Plan only (write the checklist, "
            "change no files)",
            "Click Plan only to generate PLAN.md",
            "Plan panel → Edit → Save, or Run to implement it",
        ],
        "keywords": ["plan only", "generate plan", "edit plan", "checklist",
                     "PLAN.md"],
        "settings": ["auto_plan_file"],
        "cli": ["auto"],
    },
    {
        "id": "new-project",
        "title": "Start a new project from scratch",
        "summary": "Use New mode to plan, scaffold and optionally build a "
                   "brand-new project folder.",
        "steps": [
            "Composer → New → New project folder (Browse… for a parent)",
            "What to do: Preview the plan, Build it, Build then review, or "
            "Header skeleton only",
            "Optional: Complex hands implementation to the autonomous loop; "
            "set verify, repair, task/minute caps",
            "Click Create",
        ],
        "keywords": ["new project", "scaffold", "create project", "start",
                     "generate app"],
        "settings": ["max_new_files", "max_spec_chars", "auto_split_lines"],
        "cli": ["new"],
    },
    {
        "id": "calibrate-model",
        "title": "Calibrate a model / save its model card",
        "summary": "Calibrate from the topbar or Settings; progress streams to "
                   "the Logs tab.",
        "steps": [
            "Topbar Row 2 → Calibrate (uses the active profile)",
            "Or ⚙ Settings → Model discovery / Profiles → Calibrate",
            "Watch Bottom-right Workspace → Logs; the model card tunes the "
            "remaining defaults",
        ],
        "keywords": ["calibrate", "model card", "measure model", "tune model",
                     "benchmark"],
        "settings": ["use_model_card"],
        "cli": ["calibrate", "models scan", "profiles"],
    },
    {
        "id": "tune-small-models",
        "title": "Tune settings for small or weak models",
        "summary": "Auto-Tune sets the universal budgets; per-model guard "
                   "rails live in the profile's Reasoning & Execution "
                   "Overrides.",
        "steps": [
            "⚙ Settings → Auto-Tune → answer hardware, model class, codebase "
            "size and priority → Preview recommendation → Apply",
            "⚙ Settings → Profiles → select the small model → Reasoning & "
            "Execution Overrides: incremental_edit_loop, edit_plan_brief, "
            "step_reasoning_first, dynamic_tool_pruning, repair_rounds",
            "Reduce max_read_tokens / max_gathered_chars if the window is "
            "small; keep best_of at 1 to save tokens",
        ],
        "keywords": ["small model", "small models", "weak model", "tiny",
                     "budget", "tune settings", "autotune", "auto-tune",
                     "reasoning overrides"],
        "settings": ["max_read_tokens", "max_gathered_chars",
                     "max_total_model_calls", "incremental_edit_loop",
                     "edit_plan_brief", "step_reasoning_first",
                     "dynamic_tool_pruning", "repair_rounds",
                     "post_edit_check", "static_gate", "best_of"],
        "cli": ["calibrate", "profiles"],
    },
    {
        "id": "change-theme",
        "title": "Change the GUI colour scheme",
        "summary": "Pick a theme in the topbar; it is stored as gui_theme in "
                   "the user config.",
        "steps": [
            "Topbar → theme dropdown (Light / Dark / 1990s)",
            "The choice is saved as gui_theme in the user config when one "
            "exists",
        ],
        "keywords": ["theme", "color scheme", "colour scheme", "dark mode",
                     "light mode", "gui_theme"],
        "settings": ["gui_theme"],
        "cli": ["config"],
    },
    {
        "id": "api-key-hosted",
        "title": "Configure a hosted API key / server",
        "summary": "Put the server URL, model and API key in the profile; for "
                   "hosted APIs prefer api_key_env so the secret stays out of "
                   "the file.",
        "steps": [
            "⚙ Settings → Profiles → select or create a profile",
            "Connection & Capabilities → base_url, model, api_key; click Test "
            "connection to list models",
            "For env-var secrets: ⚙ Settings → All settings → Apply to: "
            "Profile: NAME → api_key_env → Save",
            "Set json_mode = object for hosted APIs that reject json_schema",
        ],
        "keywords": ["api key", "hosted", "openai", "deepseek", "anthropic",
                     "server url", "base_url", "api_key_env", "authentication"],
        "settings": ["base_url", "api_key", "api_key_env", "model",
                     "json_mode", "tier"],
        "cli": ["config", "models scan", "calibrate"],
    },
    {
        "id": "find-logs",
        "title": "Find logs and terminal output",
        "summary": "The Workspace Logs tab carries everything Pedacito prints; "
                   "session transcripts live under .pedacito/sessions/.",
        "steps": [
            "Bottom-right Workspace panel → Logs tab",
            "Use follow to auto-scroll; Clear empties the visible terminal",
            "Project ▾ → Clear terminal does the same from the topbar",
            "CLI: session records are .pedacito/sessions/*.md; log.txt has the "
            "event trace",
        ],
        "keywords": ["logs", "terminal", "output", "session record", "log.txt",
                     "debug"],
        "settings": ["log_events", "log_sessions", "log_max_bytes",
                     "keep_events", "keep_sessions"],
        "cli": ["prune", "procs"],
    },
    {
        "id": "backups-location",
        "title": "Find or manage backups",
        "summary": "Snapshots appear in Workspace → Backups and live under "
                   ".pedacito/backups/<stamp>/.",
        "steps": [
            "Bottom-right Workspace panel → Backups tab",
            "The list shows snapshot name, file count and time; Restore copies "
            "one back",
            "CLI: pedacito restore . --list shows the same snapshots",
        ],
        "keywords": ["backups", "snapshots", "where are backups", "restore",
                     "undo"],
        "settings": ["backup", "keep_backups"],
        "cli": ["restore", "prune"],
    },
    {
        "id": "conversation-threads",
        "title": "Switch, start or delete chat conversations",
        "summary": "The Chat header holds saved threads for this project.",
        "steps": [
            "Chat header → thread dropdown to switch conversations",
            "+ New starts another thread without losing the current one",
            "Delete removes the open thread; Clear empties the transcript",
            "Threads are saved under .pedacito/gui/threads/",
        ],
        "keywords": ["thread", "conversation", "chat history", "new chat",
                     "delete chat", "switch chat"],
        "settings": ["log_sessions"],
        "cli": ["chat"],
    },
    {
        "id": "steer-task",
        "title": "Steer a running task or stop it",
        "summary": "While a Task/Edit runs, type a note and press Enter to "
                   "deliver it at the next step; Stop cancels.",
        "steps": [
            "While Task/Edit is running, type into the composer and press "
            "Enter (the send hint says delivered at the next lookup step)",
            "Use Stop in the chat header or Activity panel to cancel",
            "Steering needs allow_steering (on by default)",
        ],
        "keywords": ["steer", "stop", "cancel", "running task", "interrupt"],
        "settings": ["allow_steering"],
        "cli": ["chat"],
    },
    {
        "id": "attach-file",
        "title": "Attach a file as the task, question or spec",
        "summary": "Attach a text file in the composer or drag one in; it "
                   "travels with the run.",
        "steps": [
            "Composer → 📎 Attach (or drag a text file into the composer)",
            "The attachment chip appears above the input; remove it with ✕",
            "CLI: pass a larger spec with --spec FILE to new/auto",
        ],
        "keywords": ["attach", "upload", "drop file", "spec file", "text file"],
        "settings": ["max_spec_chars"],
        "cli": ["new", "auto", "ask"],
    },
    {
        "id": "allow-tools",
        "title": "Allow shell, write, browser or screenshot tools",
        "summary": "These are opt-in trust settings in All settings; they let "
                   "the model run commands or write during gather.",
        "steps": [
            "⚙ Settings → All settings → search for allow_shell / allow_write "
            "/ allow_browser / allow_screenshots",
            "Apply to: Global (every profile), choose on, then Save",
            "Review the safety companions too: shell_allowlist, shell_deny, "
            "sandbox, sandbox_network",
        ],
        "keywords": ["allow shell", "run commands", "allow write", "browser",
                     "screenshots", "permissions", "tools", "sandbox", "mcp",
                     "model context protocol", "lsp", "language server"],
        "settings": ["allow_shell", "allow_write", "allow_browser",
                     "allow_screenshots", "shell_allowlist", "shell_deny",
                     "sandbox", "sandbox_network", "scrub_env", "allow_mcp",
                     "mcp_servers", "mcp_timeout", "allow_lsp", "lsp_servers"],
        "cli": ["config"],
    },
    {
        "id": "verify-tests",
        "title": "Run tests or verification after edits",
        "summary": "Auto mode can run a verify command after every task; "
                   "Task/Edit can run the project's tests after applied edits.",
        "steps": [
            "Composer → Auto → Options → Verify command after each task "
            "(e.g. pytest -q)",
            "⚙ Settings → All settings → search verify_tests → on; set "
            "test_command if detection guesses wrong",
            "CLI: pedacito auto ... --apply --verify \"pytest -q\"; "
            "pedacito tests . --files src/app.py shows impacted tests",
        ],
        "keywords": ["run tests", "verify", "pytest", "verification",
                     "test command", "impacted tests", "after edits"],
        "settings": ["verify_tests", "test_command", "test_timeout",
                     "auto_verify", "auto_repair_rounds",
                     "auto_diagnostics", "static_gate"],
        "cli": ["auto", "tests", "ask"],
    },
    {
        "id": "set-up-config",
        "title": "Set up config.toml for the first time",
        "summary": "Use the first-run Settings wizard, or create and test a "
                   "minimal config from the CLI.",
        "steps": [
            "First run: ⚙ Settings opens the setup wizard",
            "Enter Server URL, click Find servers or Test, pick the model, "
            "set VRAM (blank for hosted), then Create config",
            "Later edits: ⚙ Settings → Profiles / All settings / Raw file",
        ],
        "keywords": ["set up config", "first run", "create config", "config.toml",
                     "setup", "install", "server"],
        "settings": ["base_url", "model", "vram_gb", "default_profile"],
        "cli": ["config"],
    },
    {
        "id": "check-model",
        "title": "Check that the model server works",
        "summary": "Click the topbar model chip or use Test connection / "
                   "pedacito config --check.",
        "steps": [
            "Topbar Row 2 → click the model chip (or Project ▾ → Check model "
            "(loads it))",
            "⚙ Settings → Profiles → Test connection lists the server's models",
            "CLI: pedacito config --check; --doctor also probes structured "
            "output and context",
        ],
        "keywords": ["check model", "test connection", "server down",
                     "model not found", "connection", "warmup"],
        "settings": ["base_url", "model", "timeout", "load_wait_seconds",
                     "max_consecutive_failures"],
        "cli": ["config", "calibrate"],
    },
    {
        "id": "find-settings",
        "title": "Find or change any setting",
        "summary": "All settings is a searchable form for every config key, "
                   "applied globally or to one profile.",
        "steps": [
            "⚙ Settings → All settings",
            "Type in search settings… (the key or words from its help)",
            "Apply to: Global (every profile) or Profile: NAME",
            "Change the control, then Save N changes; the Raw file tab edits "
            "TOML directly",
        ],
        "keywords": ["find setting", "change setting", "all settings", "search "
                     "settings", "config option", "preference"],
        "settings": [],
        "cli": ["config"],
    },
    {
        "id": "cli-gui-cheatsheet",
        "title": "CLI vs GUI cheat sheet",
        "summary": "The GUI and CLI drive the same engine; use whichever is "
                   "closer to the task.",
        "steps": [
            "Open/index: Topbar Open + Index ↔ pedacito gui / pedacito index .",
            "Ask: Ask mode ↔ pedacito explain . -q \"...\"",
            "Edit: Task/Edit ↔ pedacito ask . -t \"...\" --apply (or --review)",
            "Automate: Auto mode ↔ pedacito auto . -t \"...\" --apply",
            "Models: Profile dropdown / Settings ↔ pedacito profiles / pedacito "
            "models scan",
            "Undo: Workspace → Backups ↔ pedacito restore .",
            "Settings: ⚙ Settings ↔ edit config.toml / pedacito config",
        ],
        "keywords": ["cheat sheet", "cheatsheet", "cli vs gui", "gui vs cli",
                     "comparison", "which command", "same"],
        "settings": [],
        "cli": ["gui", "ask", "explain", "auto", "index", "profiles",
                "models scan", "restore", "config"],
    },
    {
        "id": "refine-prompt",
        "title": "Turn a one-sentence idea into a full spec (✦ Refine Prompt)",
        "summary": "Refine Prompt rewrites a brief request into a repo-aware spec "
                   "with locked file targets and interface signatures. Undo "
                   "Refine restores your original text.",
        "steps": [
            "Type your idea in the composer (Task/Edit, Auto or New)",
            "Click ✦ Refine Prompt. The composer now holds a spec: Objective, "
            "Files to Create or Modify, Interface Contracts, Requirements and "
            "Edge Cases, Verification",
            "Drag the top edge of the composer to resize it, then review the "
            "spec before you press Run",
            "Click ↶ Undo Refine to put back the text you wrote",
            "CLI: add --refine to pedacito ask, new or auto",
        ],
        "keywords": ["refine", "refine prompt", "spec", "architect", "expand idea",
                     "one sentence", "undo refine", "brief", "idea", "prompt"],
        "settings": [],
        "cli": ["ask", "new", "auto"],
    },
    {
        "id": "permission-dialog",
        "title": "Approve or deny a Permission requested dialog",
        "summary": "When the model asks for something gated (web access, a package "
                   "install, an unapproved shell command or a sensitive file), the "
                   "run pauses until you answer.",
        "steps": [
            "Read the badge, the action, the reason and the exact target shown in "
            "the dialog",
            "Allow Once: this single request only",
            "Allow for This Run: no more prompts for this kind of request until "
            "the run ends",
            "Always Allow: never asks again for that request. It is saved to your "
            "config when one exists. Web access flips enable_web, shell commands "
            "flip allow_shell",
            "Deny & Redirect: refuses the request. The model is told to continue "
            "with local project files only. Stop, or letting the dialog time out, "
            "also denies",
            "The CLI has no dialog: a gated tool returns its 'disabled in config' "
            "error unless the matching setting is on",
        ],
        "keywords": ["permission", "permissions", "allow", "deny", "approve",
                     "approval", "always allow", "allow once", "popup", "dialog",
                     "blocked", "waiting", "redirect"],
        "settings": ["enable_web", "allow_shell", "allow_fetch"],
        "cli": ["config"],
    },
    {
        "id": "web-access",
        "title": "Let the model read web pages and search (enable_web)",
        "summary": "Web access is off by default. fetch_url and web_search run "
                   "when enable_web is on, or when you approve each call in the "
                   "permission dialog.",
        "steps": [
            "⚙ Settings → All settings → search enable_web → Apply to: Global → "
            "on → Save",
            "Or leave it off and approve calls one at a time in the Permission "
            "requested dialog",
            "Loopback, link-local and private-network addresses are refused unless "
            "you list them in fetch_allowed_hosts",
            "Config form: enable_web = true in config.toml",
        ],
        "keywords": ["web", "internet", "online", "fetch", "fetch url", "web search",
                     "enable web", "url", "documentation", "browse", "download"],
        "settings": ["enable_web", "fetch_allowed_hosts", "allow_fetch"],
        "cli": ["config"],
    },
    {
        "id": "green-ratchet",
        "title": "See an Auto step rolled back (green ratchet)",
        "summary": "Auto saves a snapshot before each step. A step that still fails "
                   "after its repair rounds is undone, so the next step builds on "
                   "the last passing state.",
        "steps": [
            "Nothing to click: it runs in every Auto run while auto_green_ratchet "
            "is on",
            "A rollback shows as ROLLED BACK in the Activity log and as a chat note "
            "listing the restored and deleted files",
            "Old snapshots are pruned when the run ends (auto_keep_checkpoints)",
            "Turn it off in All settings (auto_green_ratchet) if you want failed "
            "steps left on disk",
        ],
        "keywords": ["rollback", "roll back", "green ratchet", "checkpoint",
                     "undo step", "failed step", "restore step", "auto rollback",
                     "step failed", "revert step"],
        "settings": ["auto_green_ratchet", "auto_keep_checkpoints",
                     "auto_checkpoint_max_files", "auto_checkpoint_max_file_bytes"],
        "cli": ["auto"],
    },
    {
        "id": "run-benchmarks",
        "title": "Compare models with the six diagnostic benchmarks",
        "summary": "▦ Benchmarks builds one small project with the chosen profile and "
                   "grades it with a hidden rubric. The pelican SVG benchmark also "
                   "shows the drawing the model produced.",
        "steps": [
            "Top bar → ▦ Benchmarks",
            "Pick a profile and one of the six benchmarks, then Run Benchmark",
            "Read the holdout score, the shortcomings and the report. The pelican "
            "benchmark shows its SVG inline",
            "Open Generated Project keeps the folder on disk to inspect or run",
        ],
        "keywords": ["benchmark", "benchmarks", "score", "rubric", "compare models",
                     "pelican", "svg", "bicycle", "holdout", "test model", "grade"],
        "settings": [],
        "cli": ["calibrate", "eval"],
    },
    {
        "id": "run-test-gather",
        "title": "Let the model run one scoped test while it investigates",
        "summary": "During gather the model can run a single pytest check (one test "
                   "file or node id) and read the live traceback before it writes a "
                   "fix. It is on by default and never runs a shell command line.",
        "steps": [
            "Nothing to click: the run_test action is offered when the project has "
            "tests or the task names a failure",
            "⚙ Settings → All settings → search enable_run_test_in_gather (on by default)",
            "When it is off, each run_test request appears in the Permission "
            "requested dialog under Test Execution",
            "Allow Once or Allow for This Run lets that request through; Always "
            "Allow turns the setting on",
        ],
        "keywords": ["run test", "run_test", "pytest", "traceback", "live test",
                     "test in gather", "investigate", "failing test", "scoped test",
                     "run tests during investigation"],
        "settings": ["enable_run_test_in_gather"],
        "cli": ["tests", "ask"],
    },
    {
        "id": "frontend-contracts",
        "title": "Catch broken page assets and DOM id mismatches",
        "summary": "For HTML and JavaScript projects Pedacito checks, before the tests "
                   "run, that every local script and stylesheet exists, that literal "
                   "getElementById and querySelector('#id') lookups match an id in the "
                   "HTML, and that changed JavaScript parses.",
        "steps": [
            "Nothing to click: it runs after each applied edit in Task/Edit and Auto",
            "A finding is reported as a FrontendContractError that names the file, "
            "the line and the missing id or asset; the repair round receives it",
            "⚙ Settings → All settings → search frontend_contracts to turn it off "
            "(on by default)",
            "For an on-demand check of the whole project, run pedacito webcheck PROJECT",
        ],
        "keywords": ["dom", "html", "javascript", "js", "getelementbyid",
                     "queryselector", "script src", "stylesheet", "asset",
                     "broken link", "missing id", "frontend", "page does not load",
                     "contract", "webcheck"],
        "settings": ["frontend_contracts", "post_edit_check"],
        "cli": ["webcheck", "auto"],
    },
    {
        "id": "repair-lessons",
        "title": "Repo gotchas and repair lessons (micro_facts.json)",
        "summary": "When a repair round fixes a failed check (syntax, static, frontend "
                   "or test), Pedacito writes one short lesson to "
                   ".pedacito/micro_facts.json under anti_patterns. Later runs in that "
                   "repository see the lessons as hints.",
        "steps": [
            "Nothing to click: a lesson is written only after the failing check "
            "passes again in the same task",
            "The file keeps at most eight lessons, de-duplicated; a lesson about a "
            "file that was deleted is dropped",
            "To read or forget them, open .pedacito/micro_facts.json in your editor; "
            "deleting the file forgets everything it recorded",
            "⚙ Settings → All settings → search micro_facts to turn recording and "
            "injection off, or max_micro_fact_tokens to change the budget",
        ],
        "keywords": ["lessons", "gotchas", "anti patterns", "anti-patterns",
                     "repair lesson", "micro facts", "micro_facts", "repo facts",
                     "remember fix", "learned", "what it learned"],
        "settings": ["micro_facts", "max_micro_fact_tokens"],
        "cli": ["memory", "auto"],
    },
    {
        "id": "scout-model",
        "title": "Use a fast scout model for lookups (scout_profile)",
        "summary": "The scout profile runs the gather lookups (grep, read, outline, find "
                   "and similar) on a small, fast model while the active profile keeps "
                   "planning, answering and editing. Built for two GPUs. A scout that is "
                   "missing or unreachable falls back to the active profile.",
        "steps": [
            "⚙ Settings → Profiles: create or select a second profile for the small "
            "model, with its own base_url and model",
            "⚙ Settings → All settings → search scout_profile → choose that profile in "
            "the dropdown, or None (use active profile) to turn it off",
            "While it is active, the topbar shows a Scout: <profile> badge beside the "
            "model chip",
            "Keep the large model as the active profile so answers and edits stay on it",
        ],
        "keywords": ["scout", "scout profile", "scout model", "architect", "two gpus",
                     "dual gpu", "second gpu", "fast model", "lookup model",
                     "gather model", "split models", "small model for search"],
        "settings": ["scout_profile", "default_profile"],
        "cli": ["profiles", "config", "ask"],
    },
    {
        "id": "json-schema-enforcement",
        "title": "Engine-enforced JSON for lookup steps (enforce_json_schema)",
        "summary": "Pedacito asks the model server to constrain every lookup step to the "
                   "action JSON, so a malformed step cannot be produced. A server that "
                   "refuses the request is downgraded automatically, per model.",
        "steps": [
            "Nothing to click: it is on by default, and it is best left on",
            "⚙ Settings → All settings → search enforce_json_schema to change it; "
            "false sends plain prompts only",
            "If a server reports errors on structured output, run pedacito config "
            "--doctor to see which modes it supports",
            "Native tool calls (tool_mode = native) bypass this and use the model's "
            "tool-call channel instead",
        ],
        "keywords": ["json", "json schema", "grammar", "structured output",
                     "response_format", "malformed json", "json error", "json mode",
                     "syntax retries", "constrained decoding", "llama.cpp", "lm studio",
                     "ollama", "vllm", "lemonade"],
        "settings": ["enforce_json_schema", "json_mode", "tool_mode"],
        "cli": ["config"],
    },
    {
        "id": "symbol-edits",
        "title": "Replace or add a whole function or class without retyping it",
        "summary": "An edit can name a definition, such as Order.total, and supply only "
                   "the new body. Pedacito finds the exact lines with the file's own "
                   "parser, so the old text is never copied and cannot drift.",
        "steps": [
            "Nothing to click: the model uses this edit shape by itself in Task/Edit "
            "and Auto",
            "Review the proposed code in the Changes pane (or with --review on the CLI) "
            "before it is written",
            "⚙ Settings → All settings → search edit_style to prefer symbol edits "
            "(anchor) or search/replace edits (quote)",
        ],
        "keywords": ["symbol", "function edit", "replace function", "replace method",
                     "splice", "surgical edit", "edit function", "edit method",
                     "whole function", "ast edit", "after_symbol", "insert method"],
        "settings": ["edit_style", "post_edit_check"],
        "cli": ["ask"],
    },
]

TOPICS = [
    ("Switch or import models",
     "How do I switch models or import a new model?"),
    ("Tune settings for small models",
     "How do I tune settings for small models?"),
    ("Toggle reasoning / thinking",
     "How do I disable model thinking or reasoning?"),
    ("CLI vs GUI cheat sheet",
     "Show me a CLI vs GUI cheat sheet"),
    ("Undo / restore backups",
     "How do I undo an Auto or Edit run and restore a backup?"),
    ("Reset config to baseline",
     "How do I reset my settings or config to the baseline?"),
    ("Reference a file in chat",
     "How do I reference a file in chat?"),
]


# ------------------------------------------------------------ settings map ----

# Exact GUI locations for the settings users ask about most. Everything else
# gets the generic All settings path below.
_SETTING_GUI = {
    "vram_gb": "⚙ Settings → Model discovery → VRAM (GB), or Profiles → "
               "[Select Profile] → Connection & Capabilities → vram_gb",
    "base_url": "⚙ Settings → Profiles → [Select Profile] → Connection & "
                "Capabilities → base_url",
    "api_key": "⚙ Settings → Profiles → [Select Profile] → Connection & "
               "Capabilities → api_key",
    "api_key_env": "⚙ Settings → All settings → Apply to: Profile: NAME → "
                   "api_key_env",
    "model": "⚙ Settings → Profiles → [Select Profile] → Connection & "
             "Capabilities → model",
    "tier": "⚙ Settings → Profiles → [Select Profile] → Connection & "
            "Capabilities → tier",
    "json_mode": "⚙ Settings → Profiles → [Select Profile] → Connection & "
                 "Capabilities → json_mode",
    "timeout": "⚙ Settings → Profiles → [Select Profile] → Connection & "
               "Capabilities → timeout",
    "allow_write": "⚙ Settings → Profiles → [Select Profile] → Connection & "
                   "Capabilities → allow_write",
    "allow_shell": "⚙ Settings → Profiles → [Select Profile] → Connection & "
                   "Capabilities → allow_shell",
    "allow_browser": "⚙ Settings → Profiles → [Select Profile] → Connection & "
                     "Capabilities → allow_browser",
    "allow_screenshots": "⚙ Settings → Profiles → [Select Profile] → "
                         "Connection & Capabilities → allow_screenshots",
    "semantic_search": "⚙ Settings → Profiles → [Select Profile] → Connection "
                       "& Capabilities → semantic_search",
    "review": "⚙ Settings → Profiles → [Select Profile] → Connection & "
              "Capabilities → review",
    "default_profile": "⚙ Settings → Profiles → [Select Profile] → Make default",
    "gui_theme": "Topbar → theme dropdown, or ⚙ Settings → All settings → "
                 "Graphical interface → gui_theme",
    "disable_thinking": "⚙ Settings → Profiles → [Select Profile] → Reasoning "
                        "& Execution Overrides → disable_thinking (Thinking)",
    "thinking_suffix": "⚙ Settings → Profiles → [Select Profile] → Reasoning "
                       "& Execution Overrides → thinking_suffix",
    "step_reasoning_first": "⚙ Settings → Profiles → [Select Profile] → "
                            "Reasoning & Execution Overrides → "
                            "step_reasoning_first",
    "incremental_edit_loop": "⚙ Settings → Profiles → [Select Profile] → "
                             "Reasoning & Execution Overrides → "
                             "incremental_edit_loop",
    "edit_plan_brief": "⚙ Settings → Profiles → [Select Profile] → Reasoning "
                       "& Execution Overrides → edit_plan_brief",
    "dynamic_tool_pruning": "⚙ Settings → Profiles → [Select Profile] → "
                            "Reasoning & Execution Overrides → "
                            "dynamic_tool_pruning",
    "prompt_pack": "⚙ Settings → Profiles → [Select Profile] → Reasoning & "
                   "Execution Overrides → prompt_pack",
    "answer_mode": "⚙ Settings → Profiles → [Select Profile] → Reasoning & "
                   "Execution Overrides → answer_mode",
    "edit_style": "⚙ Settings → Profiles → [Select Profile] → Reasoning & "
                  "Execution Overrides → edit_style",
    "tool_mode": "⚙ Settings → Profiles → [Select Profile] → Reasoning & "
                 "Execution Overrides → tool_mode",
    "best_of": "⚙ Settings → Profiles → [Select Profile] → Reasoning & "
               "Execution Overrides → best_of",
    "fuzzy_match": "⚙ Settings → Profiles → [Select Profile] → Reasoning & "
                   "Execution Overrides → fuzzy_match",
    "temperature": "⚙ Settings → Profiles → [Select Profile] → Reasoning & "
                   "Execution Overrides → temperature",
    "temperature_answer": "⚙ Settings → Profiles → [Select Profile] → "
                          "Reasoning & Execution Overrides → "
                          "temperature_answer",
    "max_output_tokens": "⚙ Settings → Profiles → [Select Profile] → "
                         "Reasoning & Execution Overrides → "
                         "max_output_tokens",
    "context_tokens": "⚙ Settings → Profiles → [Select Profile] → Reasoning & "
                      "Execution Overrides → context_tokens",
}

_REASONING_KEYS = {
    "disable_thinking", "thinking_suffix", "step_reasoning_first",
    "incremental_edit_loop", "edit_plan_brief", "dynamic_tool_pruning",
    "prompt_pack", "answer_mode", "edit_style", "tool_mode", "best_of",
    "fuzzy_match", "temperature", "temperature_answer", "max_output_tokens",
    "context_tokens",
}
_CONNECTION_KEYS = {
    "base_url", "api_key", "model", "tier", "vram_gb", "json_mode", "timeout",
    "allow_write", "allow_shell", "allow_browser", "allow_screenshots",
    "semantic_search", "review",
}
_PROFILE_GROUPS = _REASONING_KEYS | _CONNECTION_KEYS

_SETTING_KEYWORDS = {
    "disable_thinking": ["thinking", "reasoning", "turn off thinking",
                         "disable reasoning", "no think", "no_think"],
    "thinking_suffix": ["/no_think", "thinking suffix", "qwen3"],
    "api_key": ["api key", "token", "secret", "authentication", "hosted"],
    "api_key_env": ["api key env", "environment variable", "secret",
                    "hosted"],
    "base_url": ["server", "url", "endpoint", "lm studio", "ollama",
                 "llama.cpp", "hosted"],
    "model": ["model", "models", "model id", "profile"],
    "vram_gb": ["vram", "gpu", "memory", "hardware", "context window"],
    "default_profile": ["default model", "default profile", "startup model"],
    "gui_theme": ["theme", "colour scheme", "color scheme", "dark mode",
                  "light mode"],
    "review": ["review page", "diff review", "approve changes"],
    "backup": ["backup", "snapshot", "undo", "restore"],
    "allow_shell": ["run commands", "shell", "tool permission", "allow run"],
    "allow_write": ["write files", "gather write", "allow write"],
    "allow_browser": ["browser", "playwright", "web page", "screenshots"],
    "allow_screenshots": ["screenshot", "screen capture", "vision"],
    "auto_verify": ["verify command", "run tests after task", "auto verify"],
    "incremental_edit_loop": ["small model edits", "incremental edit",
                              "symbol edits"],
    "edit_plan_brief": ["edit plan", "brief before edits"],
    "step_reasoning_first": ["reasoning first", "reason before action"],
    "max_read_tokens": ["read budget", "token budget", "small context"],
    "max_gathered_chars": ["gather budget", "context budget", "small window"],
    "max_total_model_calls": ["call cap", "model call budget"],
    "post_edit_check": ["syntax check", "lint after edit", "check edit"],
    "static_gate": ["undefined name", "missing import", "ruff", "static "
                    "analysis", "fast check", "syntax gate", "typo check"],
    "static_gate_timeout": ["ruff timeout", "static analysis timeout"],
    "auto_green_ratchet": ["rollback", "roll back", "green checkpoint",
                           "undo failed step", "snapshot", "last good "
                           "state", "auto safety"],
    "auto_checkpoint_max_file_bytes": ["checkpoint size", "snapshot limit",
                                       "big files"],
    "auto_checkpoint_max_files": ["checkpoint file count", "snapshot limit"],
    "auto_keep_checkpoints": ["checkpoint retention", "keep snapshots",
                              "disk space"],
    "repair_rounds": ["edit repair", "retry matching", "fix edits"],
    "best_of": ["best of n", "sample answers", "candidates"],
    "verify_tests": ["run tests", "test after edit", "pytest"],
    "test_command": ["test command", "pytest", "npm test"],
    "auto_plan_file": ["plan file", "PLAN.md", "checklist file"],
    "include_references": ["reference files", "markdown", "csv", "json",
                           "non-python"],
    "extra_extensions": ["extra extensions", "custom file types"],
    "log_events": ["events", "telemetry", "jsonl"],
    "log_sessions": ["session logs", "transcript", "sessions"],
    "keep_backups": ["retention", "old backups", "prune backups"],
    "keep_sessions": ["retention", "old sessions", "prune sessions"],
    "prompt_cache": ["prompt cache", "caching", "cache control"],
    "temperature": ["temperature", "sampling", "creativity"],
    "max_output_tokens": ["output limit", "reply cap", "hosted limit"],
    "context_tokens": ["context window", "window size", "context length"],
    "use_model_card": ["model card", "calibrate", "tuned defaults"],
    "sandbox": ["sandbox", "bubblewrap", "safety", "isolated commands"],
    "auto_install_deps": ["pip install", "missing module", "dependency"],
    "enable_run_test_in_gather": ["run tests in gather", "pytest during "
                                  "investigation", "live traceback", "run_test"],
    "frontend_contracts": ["dom check", "asset check", "script src",
                           "getElementById", "frontend gate", "html js ids"],
    "enforce_json_schema": ["json schema", "grammar", "structured output",
                            "json grammar", "response_format"],
    "scout_profile": ["scout", "two gpus", "fast lookup model", "architect"],
    "micro_facts": ["lessons", "gotchas", "anti patterns", "repo facts",
                    "remember commands", "micro facts"],
    "max_micro_fact_tokens": ["lesson budget", "micro fact budget"],
    "edit_style": ["symbol edit", "anchor edit", "splice", "search replace"],
}


def _fallback_schema() -> list[dict]:
    """A minimal schema when the GUI package cannot be imported."""
    from dataclasses import MISSING, fields
    out = []
    base = Config()
    for field in fields(Config):
        if not hasattr(base, field.name) or field.name.startswith("_"):
            continue
        if field.default is not MISSING:
            default = field.default
        elif field.default_factory is not MISSING:
            default = field.default_factory()
        else:
            default = getattr(base, field.name, "")
        if isinstance(default, tuple):
            default = list(default)
        out.append({"name": field.name, "type": "str", "default": default,
                    "group": "General", "help": "", "choices": None,
                    "secret": False})
    return out


@lru_cache(maxsize=1)
def setting_schema() -> list[dict]:
    """Every editable setting, with the GUI's own help/group extraction."""
    try:
        from .gui import settings as gui_settings
        return list(gui_settings.schema())
    except Exception:
        return _fallback_schema()


def _toml_value(value) -> str:
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        return "{ " + ", ".join(f"{k} = {_toml_value(v)}"
                                for k, v in value.items()) + " }"
    return str(value)


def _default_text(value) -> str:
    if value is None:
        return "unset (model card / global value decides)"
    if isinstance(value, str):
        return f'"{value}"' if value else '""'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(str(v) for v in value) + "]" if value else "[]"
    if isinstance(value, dict):
        return "{}" if not value else _toml_value(value)
    return str(value)


@lru_cache(maxsize=1)
def settings_dictionary() -> list[dict]:
    """Every `Config` key, plus the hidden selectors, as searchable records."""
    schema = setting_schema()
    known = {item["name"] for item in schema}
    entries: list[dict] = []
    for item in schema:
        name = item["name"]
        group = item.get("group") or "General"
        default = item.get("default")
        profile = name in _PROFILE_GROUPS
        gui = _SETTING_GUI.get(
            name, f"⚙ Settings → All settings → {group} → {name}")
        if profile:
            section = ("Reasoning & Execution Overrides"
                       if name in _REASONING_KEYS
                       else "Connection & Capabilities")
            profile_gui = (f"⚙ Settings → Profiles → [Select Profile] → "
                           f"{section} → {name}")
        else:
            profile_gui = (f"⚙ Settings → All settings → Apply to: Profile: "
                           f"NAME → {name}")
        example = f"{name} = {_toml_value(default)}" if default is not None \
            else f"# {name} is unset by default; add it only to override"
        if profile:
            example = f"[profiles.NAME]\n{name} = {_toml_value(default)}" \
                if default is not None else \
                f"# [profiles.NAME] {name} is unset by default"
        keywords = [name, *re.split(r"[_\-./]+", name), group,
                    *_SETTING_KEYWORDS.get(name, [])]
        if item.get("choices"):
            keywords += [str(c) for c in item["choices"] if c]
        entries.append({
            "key": name,
            "title": name.replace("_", " "),
            "group": group,
            "type": item.get("type", "str"),
            "default": default,
            "default_text": _default_text(default),
            "help": item.get("help", ""),
            "choices": item.get("choices"),
            "secret": bool(item.get("secret")),
            "gui": gui,
            "profile_gui": profile_gui if profile else "",
            "toml": example,
            "keywords": keywords,
        })
    # The selectors are hidden from the one-by-one editor but users ask for
    # them constantly.
    if "default_profile" not in known:
        entries.append({
            "key": "default_profile", "title": "default profile",
            "group": "Profiles", "type": "str", "default": "",
            "default_text": '""', "help": "Profile used when --profile is not "
            "given. Set it with Make default in the profile editor.",
            "choices": None, "secret": False,
            "gui": _SETTING_GUI["default_profile"],
            "profile_gui": "",
            "toml": 'default_profile = "my-profile"',
            "keywords": ["default model", "default profile", "startup model",
                         "profile"],
        })
    return entries


# ------------------------------------------------------------------- search ---

def _doc_body(doc: dict) -> str:
    parts = [doc.get("title", ""), doc.get("summary", ""),
             doc.get("group", ""), doc.get("help", "")]
    parts += [str(x) for x in doc.get("steps", [])]
    parts += [str(x) for x in doc.get("keywords", [])]
    parts += [str(x) for x in doc.get("notes", [])]
    if doc.get("usage"):
        parts.append(doc["usage"])
    if doc.get("examples"):
        parts.extend(doc["examples"])
    if doc.get("flags"):
        parts.extend(f"{f['flag']} {f['help']}" for f in doc["flags"])
    parts.append(doc.get("key", ""))
    parts.append(doc.get("toml", ""))
    return _norm(" ".join(parts))


def _score(doc: dict, terms: dict[str, float], question_norm: str) -> float:
    title = _norm(doc.get("title") or doc.get("name") or doc.get("key", ""))
    keys = _norm(" ".join(str(k) for k in doc.get("keywords", [])))
    body = _doc_body(doc)
    heading_words = _word_set(title + " " + keys)
    body_words = _word_set(body)
    score = 0.0
    name = _norm(doc.get("name", ""))
    if (name and name not in _STOP
            and f" {name} " in f" {question_norm} "):
        # "models scan options" should outrank a generic model-profile hit.
        score += 15.0
    for term, weight in terms.items():
        if term == doc.get("key"):
            score += 18.0 * weight
        elif term == _norm(doc.get("name", "")):
            score += 14.0 * weight
        if term in heading_words:
            score += 3.4 * weight
        elif term in body_words:
            score += 0.75 * weight
    for phrase, hints in _PHRASE_HINTS.items():
        if phrase not in question_norm:
            continue
        for hint in hints:
            words = _norm(hint).split()
            # A phrase bonus only applies when the hint is part of the
            # item's heading/keywords. Otherwise "import" in a long help
            # paragraph would pull in every setting that mentions imports.
            if words and all(word in heading_words for word in words):
                score += 5.0
                break
    return score * float(doc.get("_boost", 1.0))


def _top(docs, terms, question_norm, limit, minimum=1.0, rel=0.18):
    scored = []
    for order, doc in enumerate(docs):
        score = _score(doc, terms, question_norm)
        if score >= minimum:
            scored.append((score, order, doc))
    scored.sort(key=lambda row: (-row[0], row[1]))
    if not scored:
        return []
    floor = max(minimum, scored[0][0] * rel)
    kept = [row for row in scored if row[0] >= floor]
    return [{"score": round(score, 3), **doc}
            for score, _order, doc in kept[:limit]]


def _expand(matches: dict, terms, question_norm) -> dict:
    """Pull in the items a strong GUI topic explicitly references."""
    by_cli = {c["name"]: c for c in CLI_COMMANDS}
    by_setting = {s["key"]: s for s in settings_dictionary()}
    gui = list(matches["gui"])
    cli = {item["name"]: item for item in matches["cli"]}
    settings = {item["key"]: item for item in matches["settings"]}
    for topic in gui[:1]:
        base = topic["score"] * 0.55
        for name in topic.get("cli", []):
            doc = by_cli.get(name)
            if doc is None:
                continue
            score = max(base * 0.8, _score(doc, terms, question_norm))
            current = cli.get(name)
            if current is None or current["score"] < score:
                cli[name] = {"score": round(score, 3), **doc}
        for key in topic.get("settings", []):
            doc = by_setting.get(key)
            if doc is None:
                continue
            score = max(base * 1.30, _score(doc, terms, question_norm))
            current = settings.get(key)
            if current is None or current["score"] < score:
                settings[key] = {"score": round(score, 3), **doc}
    return {
        "gui": gui,
        "cli": sorted(cli.values(), key=lambda d: -d["score"]),
        "settings": sorted(settings.values(), key=lambda d: -d["score"]),
    }


def search(question: str, limit_gui: int = 3, limit_cli: int = 3,
           limit_settings: int = 5) -> dict:
    """Rank the structured reference against one question."""
    question = " ".join((question or "").split())
    terms = _terms(question)
    question_norm = _norm(question)
    if not terms:
        return {"gui": [], "cli": [], "settings": []}
    matches = {
        "gui": _top(GUI_TOPICS, terms, question_norm, limit_gui, 1.4,
                    rel=0.60),
        "cli": _top(CLI_COMMANDS, terms, question_norm, limit_cli, 1.2,
                    rel=0.40),
        "settings": _top(settings_dictionary(), terms, question_norm,
                         limit_settings, 1.2, rel=0.40),
    }
    return _expand(matches, terms, question_norm)


def topics() -> list[dict]:
    return [{"label": label, "query": query} for label, query in TOPICS]


def build_reference_index() -> dict:
    """The complete deterministic reference (handy for tests and tooling)."""
    return {
        "cli": CLI_COMMANDS,
        "gui": GUI_TOPICS,
        "settings": settings_dictionary(),
        "common_flags": COMMON_FLAGS,
        "topics": topics(),
    }


# ---------------------------------------------------------------- formatting --

def _first_sentence(text: str, limit: int = 280) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for mark in (". ", "; ", ", "):
        pos = cut.rfind(mark)
        if pos >= limit // 2:
            return cut[:pos + 1].strip()
    return cut.rstrip() + "…"


def _flag_text(doc: dict, limit: int = 8) -> str:
    flags = [f['flag'] for f in doc.get("flags", [])]
    if not flags:
        return ""
    shown = ", ".join(f"`{f}`" for f in flags[:limit])
    if len(flags) > limit:
        shown += f", … ({len(flags) - limit} more; run `pedacito " \
                 f"{doc['name']} --help`)"
    return shown


def render_answer(question: str, matches: dict) -> str:
    """Format retrieved facts without a model. Always returns useful text."""
    gui = matches.get("gui") or []
    cli = matches.get("cli") or []
    settings = matches.get("settings") or []
    lines: list[str] = []
    lines.append("**Built-in reference**: no model or project needed.")
    lines.append("")
    best = (gui[0]["summary"] if gui else
            cli[0]["summary"] if cli else
            "I could not match that question to the built-in reference.")
    if best:
        lines.append(best)
        lines.append("")
    if gui:
        lines.append("**In the GUI**")
        for item in gui[:2]:
            lines.append(f"1. **{item['title']}**")
            for step in item.get("steps", [])[:6]:
                lines.append(f"   - {step}")
            if item.get("notes"):
                for note in item["notes"][:3]:
                    lines.append(f"   - {note}")
        lines.append("")
    else:
        lines.append("**In the GUI**")
        lines.append("- No dedicated control matches this question; use the "
                     "CLI or edit `config.toml`.")
        lines.append("")
    lines.append("**In the Terminal (CLI)**")
    if cli:
        for item in cli[:2]:
            lines.append(f"- **`pedacito {item['name']}`**: "
                         f"{_first_sentence(item['summary'], 220)}")
            lines.append(f"  - Usage: `{item['usage']}`")
            examples = item.get("examples") or []
            if examples:
                lines.append("")
                lines.append("  ```sh")
                for example in examples[:3]:
                    lines.append(f"  {example}")
                lines.append("  ```")
            flags = _flag_text(item)
            if flags:
                lines.append(f"  - Flags: {flags}")
            if item.get("common"):
                lines.append("  - Common project flags: `--exclude`, "
                             "`--include`, `--ext`, `--url`, `--model`, "
                             "`--profile`, `--steps`, `--role`, `--record`, "
                             "`--replay`, `--jsonl`, `--max-cost` "
                             "(see `--help`).")
            if item.get("notes"):
                for note in item["notes"][:4]:
                    lines.append(f"  - {note}")
            lines.append("")
    else:
        lines.append("- No CLI command is needed for this; the GUI setting is "
                     "the direct route.")
        lines.append("")
    lines.append("**Relevant Settings (`config.toml`)**")
    if settings:
        for item in settings[:4]:
            default = item.get("default_text", "")
            lines.append(f"- **`{item['key']}`** (default: `{default}`): "
                         f"{_first_sentence(item.get('help', ''), 240)}")
            if item.get("gui"):
                lines.append(f"  - Where: {item['gui']}")
            if item.get("toml"):
                lines.append("")
                lines.append("    ```toml")
                for row in item["toml"].splitlines():
                    lines.append(f"    {row}")
                lines.append("    ```")
            lines.append("")
    else:
        lines.append("- No specific setting is required; use the GUI control "
                     "or CLI command above.")
    lines.append("")
    lines.append("_Ask with `--offline` (CLI) or turn off “Use model” in the "
                 "How tab for this deterministic answer every time._")
    return "\n".join(lines).rstrip() + "\n"


def model_context(matches: dict) -> str:
    """The retrieved items, verbatim, for an optional model synthesis."""
    lines: list[str] = []
    gui = matches.get("gui") or []
    if gui:
        lines.append("GUI REFERENCE (exact labels; do not invent others):")
        for item in gui[:3]:
            lines.append(f"- {item['title']}: {item['summary']}")
            for step in item.get("steps", []):
                lines.append(f"  path step: {step}")
            if item.get("cli"):
                lines.append("  related CLI: " + ", ".join(item["cli"]))
            if item.get("settings"):
                lines.append("  settings: " + ", ".join(item["settings"]))
    cli = matches.get("cli") or []
    if cli:
        lines.append("CLI REFERENCE (exact command names/flags only):")
        for item in cli[:4]:
            lines.append(f"- pedacito {item['name']}: {item['summary']}")
            lines.append(f"  usage: {item['usage']}")
            for example in (item.get("examples") or [])[:4]:
                lines.append(f"  example: {example}")
            for flag in (item.get("flags") or [])[:20]:
                lines.append(f"  flag: {flag['flag']}: {flag['help']}")
            if item.get("common"):
                lines.append("  also supports the common project flags: "
                             + ", ".join(f["flag"] for f in COMMON_FLAGS))
    settings = matches.get("settings") or []
    if settings:
        lines.append("SETTING REFERENCE (key, default, exact GUI location):")
        for item in settings[:6]:
            lines.append(
                f"- {item['key']} (default {item['default_text']}): "
                f"{_first_sentence(item.get('help', ''), 300)}")
            lines.append(f"  GUI: {item.get('gui', '')}")
            lines.append(f"  TOML: {item.get('toml', '')}")
    return "\n".join(lines)
