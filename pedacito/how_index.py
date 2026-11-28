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
import unicodedata
from functools import lru_cache

from . import i18n
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

# Spanish question filler (accent-folded), so Spanish questions rank offline.
_ES_STOP = {
    "como", "cual", "cuales", "que", "quien", "donde", "cuando", "por",
    "para", "con", "sin", "el", "la", "los", "las", "un", "una", "unos",
    "unas", "de", "del", "al", "en", "es", "se", "mi", "mis", "tu", "tus",
    "su", "puedo", "puedes", "quiero", "necesito", "hacer", "hay", "y", "o",
    "lo", "le", "me", "te", "cambio", "cambiar", "uso", "usar", "esta",
    "este", "eso", "muy", "mas", "ya", "no", "si",
}

_ES_SYNONYMS = {
    "idioma": ("language", "lang", "es", "spanish", "espanol"),
    "lenguaje": ("language", "lang"),
    "espanol": ("language", "lang", "es", "spanish"),
    "configurar": ("configure", "setup", "config", "settings"),
    "configuracion": ("configuration", "config", "settings", "toml"),
    "modelo": ("model", "models", "profile", "server"),
    "modelos": ("model", "models", "profile", "server"),
    "deshacer": ("undo", "restore", "backup", "snapshot"),
    "revertir": ("revert", "restore", "backup", "snapshot"),
    "tema": ("theme", "appearance", "gui"),
    "habilidad": ("skill", "skills"),
    "habilidades": ("skill", "skills"),
    "instalar": ("install", "pip", "setup"),
    "instalacion": ("install", "pip", "setup"),
    "comando": ("command", "commands", "cli", "usage"),
    "comandos": ("command", "commands", "cli", "usage"),
    "error": ("fail", "failure", "troubleshoot"),
    "ayuda": ("help", "how"),
    "herramienta": ("tool", "tools"),
    "herramientas": ("tool", "tools"),
    "indice": ("index", "indexing"),
    "ignorar": ("ignore", "exclude", "pedacitoignore"),
    "archivo": ("file", "files"),
    "archivos": ("file", "files"),
    "cambios": ("changes", "apply", "review"),
    "seguridad": ("security", "sandbox", "permissions"),
    "memoria": ("memory", "vram", "gpu", "context"),
    "lento": ("slow", "performance", "context"),
    "politica": ("policy", "permission", "permissions", "permission_mode",
                 "approval_policy", "sandbox", "security"),
    "politicas": ("policy", "permission", "permissions", "policy_file"),
    "permisos": ("permission", "permissions", "permission_mode"),
    "permiso": ("permission", "permissions", "permission_mode"),
    "aprobacion": ("approval", "approve", "approval_policy"),
    "doctor": ("environment", "environment resolver", "diagnose", "probe"),
    "entorno": ("environment", "doctor", "interpreter", "virtualenv", "venv"),
}


def _fold(text: str) -> str:
    """Strip accents so "cómo" and "como" match (English text is unchanged)."""
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))


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
    "recover": ("recovery", "interrupted", "crash", "journal", "rollback",
                "resume", "cleanup"),
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
    "doctor": ("environment", "diagnose", "check", "probe", "config"),
    "environment": ("venv", "virtualenv", "interpreter", "python", "doctor",
                    "package manager", "cargo", "go"),
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
    "permission mode": ("permission_mode", "permission"),
    "permission modes": ("permission_mode", "permission"),
    "approval policy": ("approval_policy", "approval"),
    "policy file": ("policy_file", "policy"),
}


def _terms(question: str) -> dict[str, float]:
    """Weighted lowercase terms, with light synonym expansion."""
    raw: list[str] = []
    for word in _WORD_RE.findall(_fold(question.lower())):
        word = word.strip(".-/")
        if len(word) >= 2 and word not in _STOP and word not in _ES_STOP:
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
        for synonym in _SYNONYMS.get(term, ()) + _ES_SYNONYMS.get(term, ()):
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
    {"flag": "--image PATH",
     "help": "attach a PNG/JPEG/WebP/GIF image to the task; repeatable "
             "(max 4, 10 MB each)"},
    {"flag": "--config FILE", "help": "use this config file as well"},
    {"flag": "--lang {en,es}",
     "help": "interface language for this run (also PEDACITO_LANG)"},
    {"flag": "--profile NAME", "help": "apply a [profiles.NAME] block"},
    {"flag": "--steps N",
     "help": "hard lookup limit for gather; 0 = no limit (the default)"},
    {"flag": "--max-tokens-budget N",
     "help": "stop before the next call at N prompt+completion tokens"},
    {"flag": "--role ROLE=PROFILE",
     "help": "route one job to another profile; repeatable"},
    {"flag": "--auto-route",
     "help": "route each task to the specialist profile whose specialties "
             "tags match it best (task boundaries only)"},
    {"flag": "--no-auto-route",
     "help": "keep every task on the active profile, even when "
             "auto_route_specialties is on in the config"},
    {"flag": "--creative",
     "help": "creative/design (Builder) mode for every task: the designer "
             "persona, warmer answer sampling, an >=8192 token reply budget "
             "and whole-file HTML/CSS rewrites (design tasks are "
             "auto-detected even without it)"},
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
            'pedacito ask . -t "fix the parser tests" --apply --json --verify-cmd "pytest -q"',
        ],
        "flags": [
            {"flag": "-t, --task TEXT", "help": "the task (required)"},
            {"flag": "--apply", "help": "write edits to disk (backup first)"},
            {"flag": "--review",
             "help": "open the side-by-side review page before writing"},
            {"flag": "--no-review",
             "help": "never open the reviewer, even if config enables it"},
            {"flag": "--json",
             "help": "one JSON summary object on stdout at run end (human "
                     "output to stderr); turns = model calls; implies "
                     "--no-review"},
            {"flag": "--verify-cmd COMMAND",
             "help": "run this verification command once after an applied "
                     "task; overrides auto_verify for this run only (never "
                     "written to config)"},
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
            {"flag": "--execution-mode MODE",
             "help": "task engine for this run: two_phase, interleaved or "
                     "auto (overrides execution_mode in the config)"},
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
        "usage": 'pedacito auto PROJECT -t "GOAL" [--apply] [--verify CMD] '
                 '[--force-auto]',
        "summary": "Turn a goal into a checklist, then implement each task in "
                   "a fresh context. Without --apply it only writes/prints the "
                   "plan.",
        "keywords": ["autonomous", "plan", "checklist", "goal", "loop",
                     "task mode", "single pass", "suitability"],
        "examples": [
            'pedacito auto . -t "add CSV export with tests"',
            'pedacito auto . -t "add CSV export" --apply --verify "pytest -q"',
            "pedacito auto . --plan PLAN.md --apply --max-tasks 5",
            'pedacito auto . -t "risky refactor" --apply --worktree risky',
            'pedacito auto . -t "restyle the buttons in index.html" --force-auto',
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
            {"flag": "--force-auto",
             "help": "skip the Task-vs-Auto suitability nudge and run the "
                     "multi-step loop even for single-pass edits"},
            {"flag": "-y, --yes",
             "help": "assume yes: skip interactive confirmations, including "
                     "the Task-vs-Auto nudge"},
        ],
        "notes": [
            "Task-vs-Auto suitability: a prompt that targets 1-2 existing "
            "files, uses localized edit verbs (fix, restyle, rename, ...), "
            "or tweaks content/styling in a tiny project (<=5 source files) "
            "is flagged as single-pass Task work. On a TTY, auto asks "
            "'Run as a single-pass Task instead? [Y/n]' and Enter/y runs it "
            "as one ask pass; a non-TTY stdin only logs a note and keeps "
            "Auto. --force-auto (or --yes) bypasses the check; prompts that "
            "say 'multi-step plan', 'create a plan', 'from scratch', or "
            "name >=4 new files/modules always keep Auto.",
        ],
        "common": True,
    },
    {
        "name": "plan",
        "usage": "pedacito plan --check [PLAN] [PROJECT] [--impact] [--json]",
        "summary": "Validate a checklist against the index before Auto or "
                   "Complex execution: contain every named path, resolve "
                   "explicit symbols, check dependency order and preview the "
                   "callers and tests. Warnings never block; definite errors "
                   "exit 1 and an invalid plan exits 2.",
        "keywords": ["plan check", "validate plan", "impact", "callers",
                     "tests", "dependency order", "cycle", "preview",
                     "before auto", "before complex"],
        "examples": [
            "pedacito plan --check .",
            "pedacito plan --check PLAN.md .",
            "pedacito plan --check PLAN.md . --impact",
            "pedacito plan --check PLAN.md . --json",
        ],
        "flags": [
            {"flag": "PLAN", "help": "the checklist (default: PLAN.md in the "
                                     "project root)"},
            {"flag": "PROJECT", "help": "the project (default: .)"},
            {"flag": "--check", "help": "validate paths, symbols, dependency "
                                        "order and impact; read-only"},
            {"flag": "--impact", "help": "print the conservative caller/test "
                                         "preview"},
            {"flag": "--plan FILE", "help": "use this checklist instead of the "
                                            "positional PLAN"},
            {"flag": "--json", "help": "print issues and impact as JSON"},
            {"flag": "--root DIR", "help": "project root"},
            {"flag": "--config FILE", "help": "use this config file as well"},
            {"flag": "--profile NAME", "help": "apply a profile block"},
        ],
        "notes": [
            "Exit codes: 0 clean or warnings only, 1 definite errors, 2 an "
            "invalid or escaping plan path. A file may be marked as a planned "
            "creation ('Create `src/new.py`'); its parent must stay inside "
            "the project. Dynamic callers and tests without a naming link "
            "are reported as unknown, never as failures.",
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
            {"flag": "--image PATH",
             "help": "attach a reference image (PNG/JPEG/WebP/GIF); "
                     "repeatable, max 4 at 10 MB each"},
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
            {"flag": "--max-minutes N",
             "help": "wall-clock minute cap for the whole run (checked between "
                     "calls and files)"},
            {"flag": "--keep-going",
             "help": "with --complex: continue past failures"},
            {"flag": "--auto-install",
             "help": "with --complex --verify: pip-install missing modules"},
            {"flag": "--no-review", "help": "never open the reviewer"},
            {"flag": "--creative",
             "help": "creative/design (Builder) mode: designer persona, warmer "
                     "sampling, whole-file HTML/CSS rewrites"},
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
            {"flag": "--rpc", "help": "speak the JSONL-RPC protocol instead "
                                      "of prose (see `pedacito rpc`)"},
        ],
        "notes": [
            "/profile NAME switches model profile; /model ID switches model",
            "/apply on|off toggles writes; /steps N sets the lookup cap",
            "--creative starts the session in creative/design (Builder) "
            "mode; design tasks are auto-detected even without it "
            "(/creative on|off|auto lives in the TUI)",
            "/status, /cost, /memory, /skills, /profiles, /quit",
            "/consolidate (alias /compact) compacts the session into a "
            "handoff checkpoint; the next task starts from the handoff alone",
            "--rpc turns the same warm session into a machine protocol",
        ],
        "common": True,
    },
    {
        "name": "tui",
        "usage": "pedacito tui [PROJECT] [-t \"task\"] [--no-apply] "
                 "[--headless]",
        "summary": "Split-screen terminal UI: a transcript pane, a live "
                   "activity/plan sidebar, a two-row telemetry footer with "
                   "the Context Meter, Burn Rate (token burn rate) and "
                   "tok/s, plus PgUp/PgDn and mouse-wheel scrollback with "
                   "native text selection.",
        "keywords": ["tui", "terminal ui", "split screen", "context meter",
                     "telemetry", "burn rate", "token burn", "tok/s",
                     "sidebar", "plan",
                     "slash commands", "live activity", "scroll", "pgup",
                     "pgdn", "mouse wheel", "keybindings", "tab completion",
                     "command history", "/image", "/attach", "multi-line",
                     "bracketed paste", "shift+enter"],
        "examples": [
            "pedacito tui .",
            "pedacito tui ~/code/anki --apply",
            "pedacito tui . -t \"add retries to the sync client\"",
            "pedacito tui . --headless < commands.txt",
        ],
        "flags": [
            {"flag": "-t, --task TASK",
             "help": "run this task immediately after the screen opens"},
            {"flag": "--no-apply",
             "help": "start in dry-run mode; the TUI defaults to writing with "
                     "an undo backup"},
            {"flag": "--headless",
             "help": "scriptable mode: read commands from stdin and write "
                     "plain-text frames (also automatic when stdin is not a "
                     "TTY)"},
            {"flag": "--width COLS", "help": "frame width for headless runs"},
            {"flag": "--height ROWS", "help": "frame height for headless runs"},
            {"flag": "--profile NAME", "help": "apply a profile block"},
            {"flag": "--auto-route", "help": "turn specialist routing on"},
            {"flag": "--no-think", "help": "start with thinking disabled"},
            {"flag": "--creative",
             "help": "start in creative/design (Builder) mode; /creative "
                     "switches on|off|auto inside the TUI"},
        ],
        "notes": [
            "Zones: header (project/profile/model/calibration/Auto-Route/"
            "Scout/git), transcript + activity/plan split, telemetry footer, "
            "prompt bar",
            "The footer shows Ctx: 6,420 / 32,768 (19.6%) with a green/amber/"
            "red bar, Turn: +in/+out, Session totals, Speed/avg tok/s, KV "
            "cache, folds and Active model/role",
            "Slash commands: /help, /keys, /status (/telemetry), /auto GOAL, "
            "/auto! GOAL (force the multi-step loop, skipping the Task-mode "
            "suggestion), /plan, /consolidate (/compact), /profile, /model ID, "
            "/auto-route, /think, /creative, /apply, /verify on|off|status, "
            "/skills, /memory, /procs, /diff, /undo, /image, /attach, "
            "/update, /reset, /sidebar, /scroll, /mouse, /lang, /picard, /clear, /quit",
            "/attach [PATH | list | clear] (or @image.png) stages "
            "PNG/JPEG/WebP/GIF images for the next task (max 4, 10 MB each)",
            "Multi-line input: Shift+Enter, Alt+Enter, Ctrl+J or a trailing "
            "\\ then Enter insert a newline; bracketed paste (?2004h) keeps "
            "a multi-line paste as one block; Enter submits",
            "/lang [en|es] switches the interface language; F3 toggles "
            "EN/ES",
            "/auto GOAL nudges single-pass edits: when the goal looks like a "
            "one-pass Task (existing-file tweaks, localized edit verbs, a "
            "tiny project), the TUI asks 'Run as a single-pass Task instead? "
            "[Y/n]' -- Enter/y runs it as one Task, n keeps Auto. Headless "
            "sessions only log a note; /auto! forces Auto.",
            "Keys: PgUp/PgDn scroll the transcript, Shift+Up/Down scroll 3 "
            "lines, Alt+PgUp/PgDn scroll the sidebar, Up/Down recall command "
            "history, Left/Right/Home/End edit the prompt, and Tab completes "
            "slash commands and profile names",
            "Mouse wheel scrolls 3 lines by default through xterm Alternate "
            "Scroll (?1007h), preserving native left-click selection and OS "
            "copy/paste; /mouse on (or F2) enables optional SGR wheel capture "
            "(?1000h/?1006h) and /mouse off restores native selection",
            "Scrolled-back views stay stationary as new log lines arrive; "
            "the transcript shows '[ ↑ Scrolled back N lines ... ]' and "
            "Esc/End or /scroll bottom jumps to the live bottom",
            "Interactive input is a cbreak/msvcrt raw reader with a prompt "
            "buffer and synchronized DEC 2026 repaints, so PgUp and arrows "
            "never echo as ^[[5~ and telemetry repaints keep the cursor in "
            "place",
            "The right sidebar appears at 100+ columns and toggles with "
            "/sidebar; it shows PLAN.md checkboxes, gather steps, specialist "
            "swaps and modified files",
            "Permission prompts appear in the prompt bar: y = once, "
            "a = session, n = deny",
            "Rich zero-dependency view: edits show as inline syntax-highlighted "
            "unified diffs, and each tool call is a collapsible one-line card; "
            "Ctrl+O toggles the latest card, /expand [N|all] and /collapse [N|all] "
            "work on any card",
            "@ opens an interactive fuzzy picker (six rows) over project files, "
            "symbols and images; Up/Down choose, Tab or Enter inserts the path, "
            "and Esc dismisses it",
            "SGR mouse zones: click [EN|es] to switch language, the sidebar tabs, "
            "a tool card to expand it, or the scrollbar to jump; /mouse on or F2 "
            "enables capture and /mouse off restores native selection",
            "--headless is deterministic for CI: no PTY, no model server and "
            "explicit UTF-8 streams",
        ],
        "common": True,
    },
    {
        "name": "rpc",
        "usage": "pedacito rpc [PROJECT] [--permission-timeout SECONDS]",
        "summary": "Interactive JSONL-RPC over stdin/stdout for editors and "
                   "other agents: one JSON request per line "
                   '({"id":1,"method":"status"}), every stdout line is one '
                   "JSON frame (human diagnostics go to stderr), and the "
                   "index, the profiles and the session ledger stay warm "
                   "across requests. Methods: ask (streams the task's "
                   "events tagged with req_id), explain, steer (a note for "
                   "the running task), permission_reply, cancel, "
                   "consolidate (compact the session into a handoff "
                   "checkpoint; alias compact), switch_profile, status. "
                   "`pedacito chat --rpc` runs the same protocol inside the "
                   "chat command.",
        "keywords": ["rpc", "jsonl", "json", "stdio", "stdin", "stdout",
                     "editor", "machine client", "protocol", "integrate",
                     "stream events", "permission request", "steer",
                     "warm session"],
        "examples": [
            "pedacito rpc ~/code/app",
            "echo '{\"id\":1,\"method\":\"status\"}' | pedacito rpc ~/code/app",
            "echo '{\"id\":2,\"op\":\"consolidate\"}' | pedacito rpc ~/code/app",
            "pedacito chat ~/code/app --rpc",
        ],
        "flags": [
            {"flag": "--permission-timeout SECONDS",
             "help": "how long a gated tool waits for a permission_reply "
                     "before it is denied (default 120)"},
        ],
        "notes": [
            "stdout is 100% valid JSONL; nothing human-facing ever lands there",
            "one task at a time: a second ask gets a busy: error, while "
            "steer, cancel and permission_reply still reach the running one",
            "requests may spell the method as \"method\" or \"op\"; "
            "consolidate answers with before/after token counts and the "
            "handoff, and the next ask starts from that handoff alone",
            "stdin EOF flushes the active task's reply before the process "
            "exits; cancel keeps the session warm",
            "protocol details: pedacito/rpc.py; rationale: TECHNIQUES.md §31",
        ],
        "common": True,
    },
    {
        "name": "acp",
        "usage": "pedacito acp [PROJECT] [--profile NAME] "
                 "[--execution-mode MODE] [--permission-timeout SECONDS]",
        "summary": "Agent Client Protocol (JSON-RPC 2.0) over stdin/stdout "
                   "for editors and agent hosts. The bridge wraps the same "
                   "warm rpc engine: initialize advertises the agent "
                   "capabilities, session/new binds a project and optional "
                   "profile/execution mode, and session/prompt accepts text, "
                   "`@file` resource and image blocks. Tasks stream "
                   "session/update notifications (agent_message_chunk, "
                   "agent_thought_chunk, tool_call, tool_call_update, plan) "
                   "and permission prompts round-trip through "
                   "session/request_permission. session/cancel maps to the "
                   "rpc cancel and never kills the process.",
        "keywords": ["acp", "agent client protocol", "json-rpc", "jsonrpc",
                     "stdio", "editor", "zed", "agent host", "protocol",
                     "session", "prompt", "image", "resource", "permission"],
        "examples": [
            "pedacito acp ~/code/app",
            "pedacito acp ~/code/app --profile fast --execution-mode interleaved",
            "echo '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\","
            "\"params\":{}}' | pedacito acp ~/code/app",
        ],
        "flags": [
            {"flag": "--profile NAME",
             "help": "apply a [profiles.NAME] block for the session"},
            {"flag": "--execution-mode MODE",
             "help": "two_phase, interleaved or auto"},
            {"flag": "--permission-timeout SECONDS",
             "help": "how long a gated tool waits for a "
                     "session/request_permission answer before it is denied "
                     "(default 120)"},
        ],
        "notes": [
            "stdout is 100% JSON-RPC 2.0; every human line goes to stderr",
            "the project directory is bound at session/new; a foreign cwd is "
            "refused instead of being read from",
            "`@file` mentions and resource blocks go through "
            "security.resolve_safe_path, so traversal and symlink escapes are "
            "refused",
            "stdin EOF drains the active task and exits 0, with no pipe wait",
            "zero duplicated engine logic: it is pedacito/rpc.py underneath",
        ],
        "common": True,
    },
    {
        "name": "daemon",
        "usage": "pedacito daemon [start|stop|status|sessions] [--port N] "
                 "[--foreground] [--json]",
        "summary": "Headless loopback-only daemon: a warm index and model "
                   "leases shared by isolated workspace sessions. Every "
                   "endpoint requires the per-run bearer token and nothing "
                   "is exposed beyond 127.0.0.1. `status` (the default) exits "
                   "0 cleanly when no daemon is running, and `sessions` "
                   "lists attached clients.",
        "keywords": ["daemon", "background", "sessions", "workspace sessions",
                     "http", "sse", "attach", "detach", "bearer token",
                     "loopback", "warm index", "shared lease"],
        "examples": [
            "pedacito daemon status",
            "pedacito daemon start --foreground --port 8765",
            "pedacito daemon sessions --json",
            "pedacito daemon stop",
        ],
        "flags": [
            {"flag": "--port N",
             "help": "loopback port; 0 lets the OS choose an ephemeral one"},
            {"flag": "--foreground",
             "help": "keep the daemon in this process instead of detaching"},
            {"flag": "--json", "help": "print one JSON object"},
        ],
        "notes": [
            "discovery file: the per-user state folder's daemon.json (0o600 "
            "where the platform honours it), removed on stop when its pid "
            "matches",
            "HTTP/SSE control surface: /v1/health, /v1/sessions, "
            "/v1/sessions/<id>/events, /v1/quiesce and /v1/shutdown",
            "every request needs X-Pedacito-Token or Authorization: Bearer",
            "remote/WAN binding is refused by construction (loopback only)",
        ],
        "common": False,
    },
    {
        "name": "config",
        "usage": "pedacito config [--init | --setup | --check | --doctor | "
                 "--upgrade [--dry-run] | --reference]",
        "summary": "Show every resolved setting and where it came from; create, "
                   "test, print, or add newly documented options to the config.",
        "keywords": ["config file", "config.toml", "setup", "check", "doctor",
                     "reference", "settings", "upgrade", "up to date",
                     "environment", "venv", "interpreter", "package manager"],
        "examples": [
            "pedacito config",
            "pedacito config --init",
            "pedacito config --init --force",
            "pedacito config --setup",
            "pedacito config --check",
            "pedacito config --doctor",
            "pedacito config --upgrade",
            "pedacito config --upgrade --dry-run",
            "pedacito config --reference",
            "pedacito config --profile big",
        ],
        "flags": [
            {"flag": "--init",
             "help": "write a starter config to the user config path"},
            {"flag": "--path FILE",
             "help": "with --init/--reference/--upgrade: use this file"},
            {"flag": "--force", "help": "overwrite an existing file"},
            {"flag": "--check",
             "help": "connect to the server and verify the model loads"},
            {"flag": "--doctor",
             "help": "like --check plus structured-output, context and "
                     "resolved-environment probes"},
            {"flag": "--setup",
             "help": "interactive first-run setup"},
            {"flag": "--upgrade",
             "help": "append newly documented options commented out, keeping "
                     "every existing setting and comment"},
            {"flag": "--dry-run",
             "help": "with --upgrade: list the options that would be added "
                     "without writing the file"},
            {"flag": "--reference",
             "help": "print (or write with --path) the annotated template"},
            {"flag": "--root DIR", "help": "project root"},
            {"flag": "--profile NAME", "help": "show one profile's resolution"},
            {"flag": "--config FILE", "help": "use this config file as well"},
        ],
    },
    {
        "name": "doctor",
        "usage": "pedacito doctor [PROJECT] [--profile NAME] [--config FILE]",
        "summary": "Alias of `pedacito config --doctor`: connects to the "
                   "model, probes structured output, reports the sandbox and "
                   "the detected test command, and prints the offline-resolved "
                   "project environment (interpreter, PATH directories, "
                   "package manager and evidence). Resolution only reads local "
                   "metadata and executable presence: it never runs an "
                   "activation script, uv/poetry/conda/tox/nox, npx, an "
                   "install, or a network request.",
        "keywords": ["doctor", "environment", "environment resolver",
                     "venv", "virtualenv", "uv", "poetry", "pipenv", "conda",
                     "tox", "nox", "pnpm", "yarn", "bun", "cargo", "go",
                     "interpreter", "python", "health check", "diagnose"],
        "examples": [
            "pedacito doctor .",
            "pedacito doctor --profile big",
            "pedacito config --doctor",
        ],
        "flags": [
            {"flag": "PATH...", "help": "the project (default: .)"},
            {"flag": "--root DIR", "help": "project root"},
            {"flag": "--profile NAME",
             "help": "apply a [profiles.NAME] block from your config"},
            {"flag": "--config FILE", "help": "use this config file as well"},
            {"flag": "--lang en|es", "help": "interface language"},
        ],
    },
    {
        "name": "calibrate",
        "usage": "pedacito calibrate [--quick] [--recalibrate] [--repeats N] "
                 "| --all [-y]",
        "summary": "Measure the configured model (tool choice, edit format, "
                   "honesty, long context, six specialty micro-probes) and save "
                   "a model card that tunes the defaults, including the "
                   "architecture library's recommended sampler and window for "
                   "local models. Hosted provider models are refused: they use "
                   "provider-table defaults automatically. --all walks every "
                   "local profile in turn.",
        "keywords": ["model card", "measure", "benchmark", "tune", "quick",
                     "calibrate all", "hosted", "architecture defaults",
                     "calibration badge", "stale"],
        "examples": [
            "pedacito calibrate",
            "pedacito calibrate --quick",
            "pedacito calibrate --recalibrate --repeats 3",
            "pedacito calibrate --profile big",
            "pedacito calibrate --all -y",
        ],
        "flags": [
            {"flag": "--recalibrate",
             "help": "measure again even if a fresh card exists"},
            {"flag": "--all",
             "help": "calibrate every LOCAL profile in turn (hosted profiles "
                     "are skipped; fresh cards are skipped unless "
                     "--recalibrate)"},
            {"flag": "-y, --yes",
             "help": "skip the confirmation prompt (with --all)"},
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
                   "calibrate the local ones. Hosted imports are never "
                   "calibrated: they use provider-table defaults automatically.",
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
             "help": "calibrate selected LOCAL models (default: quick); "
                     "hosted models are skipped with a note"},
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
        "name": "models show",
        "usage": "pedacito models [show] [--json]",
        "summary": "Print the active model's tuned numbers next to its "
                   "profile info: hardware snapshot, footprint confidence, "
                   "last speed probe, context/timeout/max_concurrency and the "
                   "per-field auto-tune sources. The default when no action "
                   "is given; `models scan` is unchanged.",
        "keywords": ["models", "show models", "tuned numbers", "auto tune",
                     "context tokens", "timeout", "vram", "profile info"],
        "examples": [
            "pedacito models",
            "pedacito models show",
            "pedacito models --json",
        ],
        "flags": [
            {"flag": "show", "help": "print the active profile's tuning "
                                     "(the default)"},
            {"flag": "--json", "help": "print structured JSON"},
            {"flag": "--config FILE", "help": "use this config file"},
            {"flag": "--profile NAME", "help": "apply a profile block"},
        ],
    },
    {
        "name": "models refresh",
        "usage": "pedacito models refresh [--json] [--config FILE]",
        "summary": "Force-refresh the cached models.dev catalog now (context "
                   "windows and list prices for hosted providers, 24 h TTL). "
                   "Every failure is quiet: a failed fetch keeps the previous "
                   "cache, and nothing else changes.",
        "keywords": ["models refresh", "refresh catalog", "models.dev",
                     "pricing", "context window", "provider catalog",
                     "24 hours", "cache", "offline"],
        "examples": [
            "pedacito models refresh",
            "pedacito models refresh --json",
        ],
        "flags": [
            {"flag": "refresh", "help": "the action: refresh model_catalog.json "
                                      "from models.dev"},
            {"flag": "--json", "help": "print the refresh result as JSON"},
            {"flag": "--config FILE", "help": "use this config file as well"},
        ],
    },
    {
        "name": "profile",
        "usage": "pedacito profile [--show | --run | --reset] [--json]",
        "summary": "Show the vendor-neutral hardware snapshot (RAM/VRAM with a "
                   "source per number), the last cached speed probe and the "
                   "active auto-tune. --run forces one prefill/generation "
                   "probe; --reset drops cached measurements and turns off an "
                   "explicit auto_tune opt-in.",
        "keywords": ["profile", "profiling", "hardware snapshot", "auto tune",
                     "auto_tune", "vram", "ram", "nvidia-smi", "vm_stat",
                     "winapi", "prefill", "tokens per second", "confidence",
                     "kv cache", "footprint"],
        "examples": [
            "pedacito profile",
            "pedacito profile --show",
            "pedacito profile --run",
            "pedacito profile --reset",
            "pedacito profile --show --json --profile strata",
        ],
        "flags": [
            {"flag": "--show", "help": "snapshot + last probe + active tuning"},
            {"flag": "--run", "help": "force a speed/prefill probe now"},
            {"flag": "--reset", "help": "drop cached measurements and the "
                                        "auto_tune opt-in"},
            {"flag": "--json", "help": "print structured JSON"},
            {"flag": "--config FILE", "help": "use this config file"},
            {"flag": "--profile NAME", "help": "apply a profile block"},
        ],
    },
    {
        "name": "image",
        "usage": 'pedacito image PROMPT [--out FILE] [--size WxH] '
                 '[--steps N] [--seed N] [--profile NAME]',
        "summary": "Generate one image with a configured image-model "
                   "profile (modality = \"image\": Lemonade sd-cpp, Stable "
                   "Diffusion, Flux or a hosted image endpoint) and save it "
                   "inside the workspace, by default under "
                   ".pedacito/images/. No chat model is involved. The same "
                   "call is the /image slash command in the TUI and the "
                   "Project menu's Generate image dialog in the GUI.",
        "keywords": ["image", "generate image", "draw", "picture",
                     "stable diffusion", "flux", "sd-cpp", "lemonade",
                     "text to image", "image generation", "txt2img"],
        "examples": [
            'pedacito image "a lighthouse at dusk, watercolor"',
            'pedacito image "app icon, flat" --size 512x512 --steps 4 '
            '--out assets/icon.png',
            'pedacito image "a red fox" --seed 7 --json',
        ],
        "flags": [
            {"flag": "PROMPT", "help": "description of the image"},
            {"flag": "--out FILE",
             "help": "output path inside the workspace (default "
                     ".pedacito/images/...)"},
            {"flag": "--size WxH",
             "help": "dimensions, e.g. 1024x1024 (default image_default_size)"},
            {"flag": "--steps N",
             "help": "diffusion steps (default image_default_steps; 0 = "
                     "the server's own)"},
            {"flag": "--cfg-scale F", "help": "classifier-free guidance scale"},
            {"flag": "--seed N", "help": "random seed for reproducible output"},
            {"flag": "--negative TEXT", "help": "what to avoid"},
            {"flag": "--profile NAME", "help": "image profile to use"},
            {"flag": "--json", "help": "print the result as a JSON object"},
        ],
        "notes": [
            "Needs an image profile: modality = \"image\" on the profile "
            "(auto-detected by models scan) and image_profile naming it; "
            "empty image_profile uses the first image profile found",
            "allow_image_gen = true additionally exposes the generate_image "
            "agent action; it is an execution switch, ignored from a "
            "project's own config until pedacito trust",
            "pedacito calibrate on an image profile runs a single 512x512 "
            "readiness probe instead of the text probes",
        ],
    },
    {
        "name": "rename",
        "usage": "pedacito rename OLD NEW [--file PATH] [--strings] "
                 "[--include-labels] [--apply] [--root DIR]",
        "summary": "Rename a symbol across the project without a model call: "
                   "Python identifiers move through the real tokenizer, "
                   "JavaScript/TypeScript/Go/Rust through tree-sitter "
                   "identifier nodes, so strings and comments that merely "
                   "contain the old name are never touched by default. "
                   "--strings also renames whole-word occurrences inside "
                   "string literals (comments stay untouched); "
                   "--include-labels is the narrow middle option that moves "
                   "only test-label strings (test/it/describe/suite). Default "
                   "is a dry-run diff preview; --apply writes and snapshots "
                   "the originals, so restore undoes it.",
        "keywords": ["rename", "refactor", "symbol", "identifier", "variable",
                     "function name", "rename across files", "rename strings",
                     "string literals", "test labels"],
        "examples": [
            "pedacito rename apply_discount discounted_price",
            "pedacito rename apply_discount discounted_price --file pkg/core.py",
            "pedacito rename apply_discount discounted_price --strings",
            "pedacito rename toggleStep flipStep --include-labels",
            "pedacito rename apply_discount discounted_price --apply --root .",
            "pedacito restore . --dry-run",
        ],
        "flags": [
            {"flag": "OLD", "help": "the existing identifier"},
            {"flag": "NEW", "help": "the new identifier"},
            {"flag": "--file PATH",
             "help": "scope: that file plus the files importing it "
                     "(default: all indexed project source)"},
            {"flag": "--strings",
             "help": "also rename whole-word occurrences inside string "
                     "literals (test labels, docs that name the API); "
                     "comments stay untouched"},
            {"flag": "--include-labels",
             "help": "also rename test-label strings only (the first "
                     "argument of test/it/describe/suite); every other "
                     "string literal stays untouched"},
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
        "name": "undo",
        "usage": "pedacito undo [PROJECT] [--list]",
        "summary": "Undo the last run: restore the workspace to the state "
                   "immediately before the active checkpoint (or the newest "
                   "backup snapshot on older projects).",
        "keywords": ["undo", "revert", "last run", "checkpoint", "rollback"],
        "examples": [
            "pedacito undo .",
            "pedacito undo . --list",
        ],
        "flags": [
            {"flag": "PATH...", "help": "the project (default: .)"},
            {"flag": "--root DIR", "help": "project root"},
            {"flag": "--list", "help": "show the checkpoint tree instead"},
        ],
    },
    {
        "name": "checkpoints",
        "usage": "pedacito checkpoints PROJECT [--limit N] [--json]",
        "summary": "List the time-travel checkpoint tree: ids, parents, "
                   "timestamps, modes, prompts and changed-file counts.",
        "keywords": ["checkpoint", "timeline", "history", "snapshot",
                     "time travel", "branch"],
        "examples": [
            "pedacito checkpoints .",
            "pedacito checkpoints . --limit 10",
            "pedacito checkpoints . --json",
        ],
        "flags": [
            {"flag": "PATH...", "help": "the project (default: .)"},
            {"flag": "--root DIR", "help": "project root"},
            {"flag": "--limit N", "help": "show only the newest N checkpoints"},
            {"flag": "--json", "help": "print structured metadata"},
        ],
    },
    {
        "name": "checkout",
        "usage": "pedacito checkout ID [PROJECT] [--file PATH]",
        "summary": "Restore the whole workspace, or one file, from a "
                   "checkpoint. The restored checkpoint becomes the active "
                   "head, so the next run branches from it.",
        "keywords": ["checkout", "restore checkpoint", "time travel",
                     "branch", "undo"],
        "examples": [
            "pedacito checkout cp_0003_a1b2c3 .",
            "pedacito checkout cp_0003 . --file app.py",
        ],
        "flags": [
            {"flag": "ID", "help": "checkpoint id; a unique prefix works"},
            {"flag": "PATH...", "help": "the project (default: .)"},
            {"flag": "--root DIR", "help": "project root"},
            {"flag": "--file PATH",
             "help": "restore only this file; repeatable"},
        ],
    },
    {
        "name": "recover",
        "usage": "pedacito recover list [PROJECT] | pedacito recover "
                 "resume|rollback|cleanup TASK_ID [PROJECT] [--apply]",
        "summary": "List interrupted task journals by their UUID and preview "
                   "or apply a recovery action. Everything is read-only "
                   "without --apply; resume and cleanup refuse when Pedacito "
                   "cannot prove a durable boundary or authoritative "
                   "ownership.",
        "keywords": ["recover", "recovery", "interrupted task", "crashed run",
                     "journal", "wal", "resume", "rollback", "cleanup",
                     "task id", "uuid", "legacy journal"],
        "examples": [
            "pedacito recover list .",
            "pedacito recover rollback 6f1c... .",
            "pedacito recover rollback 6f1c... . --apply",
            "pedacito recover resume 6f1c... .",
            "pedacito recover cleanup 6f1c... .",
        ],
        "flags": [
            {"flag": "list | resume | rollback | cleanup",
             "help": "list is read-only; the others preview by default"},
            {"flag": "TASK_ID", "help": "the journal UUID from recover list; "
                                        "legacy titles are read-only"},
            {"flag": "PATH...", "help": "the project (default: .)"},
            {"flag": "--root DIR", "help": "project root"},
            {"flag": "--apply", "help": "perform a safely bound rollback; "
                                        "without it nothing is written"},
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
        "usage": "pedacito eval [--task ID] [--out FILE] [--baseline FILE] "
                 "[--scorecard FILE]",
        "summary": "Score the harness on built-in fixture repos (locate "
                   "precision, edit success, tokens, cache hits). With "
                   "--scorecard it also writes a schema-v2 scorecard, the "
                   "comparable artifact for `pedacito bench compare`.",
        "keywords": ["benchmark", "score", "harness", "baseline", "regression",
                     "scorecard"],
        "examples": [
            "pedacito eval --list",
            "pedacito eval --out report.json",
            "pedacito eval --task locate_symbol --task edit_small",
            "pedacito eval --baseline old-report.json --max-token-growth 0.2",
            "pedacito eval --seed 1 --scorecard card-before.json",
        ],
        "flags": [
            {"flag": "--task ID", "help": "run only this task id; repeatable"},
            {"flag": "--out FILE", "help": "write the JSON report here"},
            {"flag": "--scorecard FILE",
             "help": "write a schema-v2 scorecard; needs --seed or --repeats "
                     "greater than 1 and the full task set"},
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
            {"flag": "--lang {en,es}",
             "help": "interface language for the GUI (also the EN | ES "
                     "top-bar switcher)"},
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
            {"flag": "--url URL, --server URL", "help": "server base URL override"},
            {"flag": "--api-key KEY", "help": "API key override"},
            {"flag": "--model ID", "help": "model id override"},
            {"flag": "--vram GB, --vram-gb GB", "help": "GPU VRAM in whole gigabytes"},
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
        "usage": "pedacito diagnose [DIR] [--root DIR] [--file FILE]",
        "summary": "Run the project's detected linter/typechecker.",
        "keywords": ["lint", "typecheck", "static analysis"],
        "examples": ["pedacito diagnose .", "pedacito diagnose . --file app.py"],
        "flags": [
            {"flag": "--root DIR", "help": "the project"},
            {"flag": "--file FILE", "help": "restrict to one project file"},
        ],
    },
    {
        "name": "diag",
        "usage": "pedacito diag [PATH] [--root DIR] [--file FILE]",
        "summary": "Alias for `pedacito diagnose`: run the project's detected "
                   "linter/typechecker.",
        "keywords": ["diag", "lint", "typecheck", "static analysis", "alias"],
        "examples": ["pedacito diag .", "pedacito diag . --file app.py"],
        "flags": [
            {"flag": "--root DIR", "help": "the project"},
            {"flag": "--file FILE", "help": "restrict to one project file"},
        ],
    },
    {
        "name": "commit",
        "usage": "pedacito commit [PATH] [--all] [-m MSG] [--no-model] [-y] "
                 "[--allow-secrets]",
        "summary": "Review the staged diff and create a local git commit. The "
                   "message is drafted by the active model (or deterministically "
                   "with --no-model). A secret pre-flight scan runs on the "
                   "staged diff before every commit and refuses a match unless "
                   "--allow-secrets is given. It never pushes and never "
                   "force-pushes.",
        "keywords": ["commit", "git commit", "commit message", "stage", "staged",
                     "secret scan", "secrets", "gitops", "guided git", "local commit"],
        "examples": [
            "pedacito commit . --all -m \"fix retry backoff\"",
            "pedacito commit . --no-model -y",
            "pedacito commit . --all --profile fast",
        ],
        "flags": [
            {"flag": "PATH...", "help": "the project (default: .)"},
            {"flag": "--root DIR", "help": "project root"},
            {"flag": "--all", "help": "stage every change first (git add -A)"},
            {"flag": "-m, --message MSG",
             "help": "use this message instead of drafting one"},
            {"flag": "--no-model",
             "help": "draft the message deterministically, with no model call"},
            {"flag": "-y, --yes", "help": "commit without asking for confirmation"},
            {"flag": "--allow-secrets",
             "help": "commit even when the staged diff matches a secret pattern "
                     "(explicit override)"},
            {"flag": "--config FILE", "help": "use this config file as well"},
            {"flag": "--profile NAME", "help": "apply a profile block"},
        ],
        "notes": [
            "nothing here pushes: the commit stays local until you push it yourself",
            "argv-only git calls; no shell is involved",
            "the TUI twin is /git; the GUI twin is the Git drawer",
        ],
    },
    {
        "name": "pr",
        "usage": "pedacito pr [PATH] [--base BRANCH] [--draft] [--title T] "
                 "[--body B] [--dry-run] [-y] [--no-model]",
        "summary": "Draft a pull request title and body from the branch's commits "
                   "and open it with the GitHub CLI (gh). It never pushes the "
                   "branch itself and never force-pushes; --dry-run prints the "
                   "draft and never calls gh.",
        "keywords": ["pr", "pull request", "gh", "github", "draft pr", "open pr",
                     "merge request", "review request", "guided git"],
        "examples": [
            "pedacito pr . --base main --draft",
            "pedacito pr . --dry-run",
            "pedacito pr . --title \"Retry backoff\" --body \"Adds jitter\" -y",
        ],
        "flags": [
            {"flag": "PATH...", "help": "the project (default: .)"},
            {"flag": "--root DIR", "help": "project root"},
            {"flag": "--base BRANCH", "help": "base branch (default: main)"},
            {"flag": "--draft", "help": "open the pull request as a draft"},
            {"flag": "--title TEXT", "help": "use this title instead of drafting one"},
            {"flag": "--body TEXT", "help": "use this body instead of drafting one"},
            {"flag": "--dry-run",
             "help": "print the drafted title and body; never call gh"},
            {"flag": "-y, --yes", "help": "open the pull request without confirming"},
            {"flag": "--no-model",
             "help": "draft the title and body deterministically, no model call"},
            {"flag": "--config FILE", "help": "use this config file as well"},
            {"flag": "--profile NAME", "help": "apply a profile block"},
        ],
        "notes": [
            "push the branch first yourself: pr opens the pull request, it does "
            "not publish your commits",
            "the TUI twin is /git; the GUI twin is the Git drawer",
        ],
    },
    {
        "name": "parsers",
        "usage": "pedacito parsers prefetch [--languages LANG...]",
        "summary": "Tree-sitter grammar tools for offline and air-gapped machines. "
                   "`prefetch` warms the common grammars and reports ok or failed "
                   "per language. Without a grammar, the tree-sitter path falls "
                   "back to the built-in parsers and the run goes on.",
        "keywords": ["parsers", "tree-sitter", "tree sitter", "grammar", "grammars",
                     "offline", "air-gapped", "prefetch", "retry", "fallback"],
        "examples": [
            "pedacito parsers prefetch",
            "pedacito parsers prefetch --languages python javascript go",
        ],
        "flags": [
            {"flag": "prefetch", "help": "the only action: pre-warm the grammars"},
            {"flag": "--languages LANG...",
             "help": "grammar names to fetch (default: the common set)"},
        ],
        "notes": [
            "run it once while online, then work offline: a missing grammar "
            "retries and then falls back instead of failing the whole run",
        ],
    },
    {
        "name": "update",
        "usage": "pedacito update [--run] [--config FILE]",
        "summary": "Check PyPI for a newer Pedacito release and upgrade this "
                   "install. One anonymous GET (2 s timeout, cached 24 h); "
                   "offline is a quiet no-op. The upgrade command matches "
                   "how Pedacito was installed (pipx, venv, pip --user); "
                   "editable checkouts get git-pull instructions instead.",
        "keywords": ["update", "upgrade", "new version", "release", "pipx",
                     "check for updates", "offline"],
        "examples": [
            "pedacito update",
            "pedacito update --run",
        ],
        "flags": [
            {"flag": "--run",
             "help": "run the detected upgrade command without asking"},
            {"flag": "--config FILE",
             "help": "config file to read check_for_updates from"},
        ],
        "notes": [
            "Opt out entirely with check_for_updates = false in config.toml "
            "(or PEDACITO_CHECK_FOR_UPDATES=false): zero network calls and "
            "zero upgrade subprocesses, everywhere (GUI banner, TUI /update, "
            "this command).",
            "The GUI shows a slim dismissible banner when a newer release "
            "exists; Update streams the upgrade into an output drawer and "
            "offers Restart on success. The TUI command is /update [run].",
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
        "usage": "pedacito procs [list | logs NAME | stop NAME] [--tail N] "
                 "[--all]",
        "summary": "List, show logs from, or stop background servers started "
                   "with serve or the model's `serve` action. The list hides "
                   "stopped entries superseded by a newer running instance "
                   "of the same server; --all shows them.",
        "keywords": ["background processes", "server logs", "stop server"],
        "examples": [
            "pedacito procs",
            "pedacito procs --all",
            "pedacito procs logs web --tail 2000",
            "pedacito procs stop web",
        ],
        "flags": [
            {"flag": "list|logs|stop", "help": "the action (default: list)"},
            {"flag": "NAME", "help": "registry name"},
            {"flag": "--tail N",
             "help": "with logs: last N characters (default 4000)"},
            {"flag": "--all",
             "help": "with list: also show superseded stopped entries"},
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
        "name": "mcp",
        "usage": "pedacito mcp [list]",
        "summary": "List every tool the named MCP servers "
                   "([mcp_servers.<name>] tables) expose, with the trust "
                   "flag each call would ask about.",
        "keywords": ["mcp", "model context protocol", "external tools",
                     "mcp servers", "mcp tools"],
        "examples": [
            "pedacito mcp list",
            "pedacito mcp list --root ~/projects/app",
        ],
        "flags": [
            {"flag": "list", "help": "the only action (default)"},
            {"flag": "--root DIR", "help": "project root"},
            {"flag": "--config FILE", "help": "use this config file"},
            {"flag": "--profile NAME", "help": "apply a profile block"},
        ],
    },
    {
        "name": "memory",
        "usage": "pedacito memory PROJECT [--add TEXT | --forget TEXT | "
                 "--consolidate]",
        "summary": "List, add or forget the facts Pedacito remembers about a "
                   "project, or consolidate the recent session history into "
                   "a handoff checkpoint.",
        "keywords": ["remember", "facts", "notes", "memory.md",
                     "consolidate", "compact", "handoff", "session summary"],
        "examples": [
            "pedacito memory .",
            'pedacito memory . --add "tests run with pytest -q"',
            'pedacito memory . --forget "old fact"',
            "pedacito memory . --global",
            "pedacito memory . --consolidate",
        ],
        "flags": [
            {"flag": "--add TEXT", "help": "remember a fact"},
            {"flag": "--forget TEXT", "help": "remove a fact"},
            {"flag": "--consolidate",
             "help": "compact recent session history into a consolidated "
                     "handoff checkpoint (.pedacito/consolidated_session.md)"},
            {"flag": "--global",
             "help": "use the user-level memory instead of the project's"},
            {"flag": "--root DIR", "help": "project root"},
            {"flag": "--config FILE", "help": "use this config file"},
            {"flag": "--profile NAME", "help": "apply a profile block"},
        ],
        "notes": [
            "--consolidate distils the newest session transcripts into four "
            "sections (Goal & Context, Files Modified & Key Decisions, "
            "Verified Test Status, Open Next Steps) under 400 tokens, and "
            "pins durable decisions to memory.md",
            "in chat the same operation is /consolidate (alias /compact); "
            "over RPC it is the consolidate method; in the GUI the "
            "Consolidate Session button",
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
        "keywords": ["test command", "impacted tests", "pytest", "flaky test",
                     "flake", "quarantine", "flakes.json"],
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
        "name": "domcheck",
        "usage": "pedacito domcheck PROJECT [--entry FILE] [--snapshot FILE] "
                 "[--baseline FILE]",
        "summary": "Headless DOM smoke check: load the entry page in "
                   "Chromium, prove the module graph mounts, report page "
                   "errors, failed loads, unbound controls and silent Web "
                   "Audio failures (never-resumed context, disconnected "
                   "graph, dead transport, zeroed gains, invalid param "
                   "automation); --baseline diffs the mounted surface "
                   "against an earlier --snapshot. Needs Playwright (pip "
                   "install playwright; playwright install chromium).",
        "keywords": ["web", "dom", "headless", "browser", "playwright",
                     "mount", "smoke check", "runtime", "audio", "webaudio",
                     "sound", "silent", "synth", "sequencer"],
        "examples": [
            "pedacito domcheck .",
            "pedacito domcheck . --snapshot /tmp/before.json",
            "pedacito domcheck . --baseline /tmp/before.json",
        ],
        "flags": [
            {"flag": "--entry FILE", "help": "entry HTML to load"},
            {"flag": "--timeout SECONDS", "help": "cap one page load"},
            {"flag": "--snapshot FILE", "help": "save the mounted-DOM snapshot"},
            {"flag": "--baseline FILE", "help": "diff against an earlier snapshot"},
            {"flag": "--root DIR", "help": "project root"},
            {"flag": "--config FILE", "help": "use this config file"},
            {"flag": "--profile NAME", "help": "apply a profile block"},
        ],
    },
    {
        "name": "bench",
        "usage": "pedacito bench [BENCHMARK|all] [--list | --recent] "
                 "[--dir DIR] [--json] [--strict] | pedacito bench compare "
                 "BASELINE CURRENT [--json]",
        "summary": "Run the six-project diagnostic benchmark suite with its "
                   "hidden holdout rubric, or list the benchmarks and recent "
                   "runs. A run builds the project through the normal loop, "
                   "then grades the model's own tests and 71 holdout checks.",
        "keywords": ["benchmark", "bench", "suite", "holdout", "rubric",
                     "score", "compare models", "recent runs"],
        "examples": [
            "pedacito bench --list",
            "pedacito bench --recent",
            "pedacito bench rpn_expression_calc",
            "pedacito bench all --json --strict",
            "pedacito bench rpn_expression_calc --dir ~/benchmarks --json",
            "pedacito bench compare before.json after.json",
        ],
        "flags": [
            {"flag": "BENCHMARK",
             "help": "one benchmark id, or 'all' for all six (default: all)"},
            {"flag": "compare BASELINE CURRENT",
             "help": "gate two eval scorecards: exit 0 pass, 1 regression, "
                     "2 invalid or incomparable"},
            {"flag": "--max-prompt-growth FRAC",
             "help": "compare: allowed prompt-token growth (default 0)"},
            {"flag": "--max-time-growth FRAC",
             "help": "compare: allowed seconds growth (default 0.50)"},
            {"flag": "--list", "help": "list the six benchmarks and exit"},
            {"flag": "--recent",
             "help": "list recent runs under the benchmarks root and exit"},
            {"flag": "--dir DIR / --out-dir DIR",
             "help": "base directory for benchmark project folders"},
            {"flag": "--json",
             "help": "print structured JSON on stdout for CI"},
            {"flag": "--strict",
             "help": "exit 1 unless model tests and holdout rubric both pass"},
            {"flag": "--model-timeout SEC",
             "help": "timeout for the generated project's own tests"},
            {"flag": "--holdout-timeout SEC",
             "help": "timeout for the hidden holdout rubric"},
            {"flag": "--url URL", "help": "server URL"},
            {"flag": "--model ID", "help": "model id"},
            {"flag": "--config FILE", "help": "use this config file"},
            {"flag": "--profile NAME", "help": "apply a profile block"},
            {"flag": "--root DIR", "help": "project root"},
        ],
    },
    {
        "name": "skills",
        "usage": "pedacito skills [list | show NAME | add NAME | draft "
                 "\"PROMPT\" | rm NAME] [PROJECT] [--global]",
        "summary": "Browse and manage the skill library: nineteen bundled "
                   "starters plus your own user/project skills. list (the "
                   "default) shows every skill with its scope badge; show "
                   "prints one skill's triggers and checklist; add writes a "
                   "custom skill from flags; draft asks the active model to "
                   "write one from a sentence; rm deletes a custom skill "
                   "(built-ins are immutable).",
        "keywords": ["skills", "capabilities", "catalog", "custom skill",
                     "create skill", "draft skill", "delete skill",
                     "skill library", "triggers"],
        "examples": [
            "pedacito skills .",
            "pedacito skills show add-a-test",
            'pedacito skills add deploy-checklist --description "Ship a '
            'release safely" --triggers "deploy,release" --body "1. ..."',
            'pedacito skills draft "Skill for Alpine.js dark-mode components"',
            "pedacito skills rm deploy-checklist",
            "pedacito skills list --global",
        ],
        "flags": [
            {"flag": "--description TEXT",
             "help": "one-line description (skills add)"},
            {"flag": "--triggers K1,K2,...",
             "help": "comma-separated trigger keywords (skills add)"},
            {"flag": "--body TEXT", "help": "the checklist body (skills add)"},
            {"flag": "--global",
             "help": "use the user-level skill library instead of the "
                     "project's"},
            {"flag": "--url URL", "help": "server base URL override (draft)"},
            {"flag": "--model ID", "help": "model id override (draft)"},
            {"flag": "--root DIR", "help": "project root"},
            {"flag": "--config FILE", "help": "use this config file"},
            {"flag": "--profile NAME", "help": "apply a profile block"},
        ],
        "notes": [
            "custom skills live in .pedacito/skills/<name>.md (project) or "
            "~/.config/pedacito/skills/<name>.md (user); a project skill "
            "shadows a bundled one of the same name",
            "select_skills injects the top-2 trigger matches for a task, "
            "capped at 1,200 chars of checklist text",
            "the GUI twin is Project ▾ → Skills library & creator…",
        ],
    },
]

# --------------------------------------------------------------- GUI map ------

GUI_TOPICS = [
    {
        "id": "image-input",
        "title": "Send an image with your task (multimodal input)",
        "summary": "Attach PNG/JPEG/WebP/GIF images (max 4, 10 MB each) to a "
                   "task for a vision-capable model: paste, drag and drop or "
                   "use 📎 Attach in the GUI, /attach in the TUI, --image "
                   "PATH on the CLI.",
        "steps": [
            "GUI: paste with Ctrl+V / Cmd+V, drag and drop onto the "
            "composer, or click 📎 Attach",
            "TUI: /attach [PATH | list | clear] or @image.png in the prompt",
            "CLI: --image PATH (repeatable) on ask, auto, chat or new",
            "Use a vision-capable model; images over 10 MB or beyond four "
            "are refused",
        ],
        "keywords": ["image", "attach", "attachment", "screenshot", "paste "
                     "image", "drag and drop", "multimodal", "vision",
                     "image input", "send image", "upload", "picture",
                     "/attach", "--image"],
        "settings": [],
        "cli": ["ask", "new", "tui"],
    },
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
            "VRAM GB); the scan is capped at 20 s and returns what it found "
            "when the budget is spent",
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
        "id": "install-pedacito",
        "title": "Install Pedacito",
        "summary": "Install from PyPI with `pip install pedacito` (or "
                   "`pipx install pedacito` for an isolated CLI install); "
                   "Python 3.11+ is required. Both put `pedacito` and the "
                   "short alias `pdc` on your PATH.",
        "steps": [
            "pip install pedacito   (Windows: py -m pip install pedacito)",
            "pipx install pedacito   isolated install, recommended for a tool",
            "Optional extras: pip install \"pedacito[browser,vision]\" adds "
            "browser verification (Playwright) and screenshot analysis "
            "(Pillow); with pipx use pipx install \"pedacito[browser,vision]\"",
            "After the browser extra: python -m playwright install chromium "
            "(pipx: pipx run --spec playwright playwright install chromium)",
            "From source: git clone https://github.com/davepo/pedacito.git, "
            "then pip install -e \".[dev]\" inside a virtualenv",
            "Command not found? `python -m pedacito` is identical. Next "
            "step after installing: pedacito config --init",
        ],
        "keywords": ["install", "install pedacito", "installation", "pip",
                     "pip install", "pipx", "pypi", "setup pedacito",
                     "getting started", "python 3.11", "playwright",
                     "browser extra", "vision extra", "from source"],
        "settings": ["base_url", "model", "vram_gb"],
        "cli": ["config", "update"],
    },
    {
        "id": "update-pedacito",
        "title": "Update Pedacito to a newer release",
        "summary": "A slim banner appears when a newer release is on PyPI; "
                   "Update runs the command for your install method and "
                   "streams its output, then offers Restart. Fully opt-out.",
        "steps": [
            "When a newer release exists, a banner appears under the topbar: "
            "Pedacito vX.Y.Z is available (you have A.B.C)",
            "Click Update: the detected upgrade command (pipx upgrade / pip "
            "install --upgrade) runs in the background and its output "
            "streams into a small drawer",
            "Exit 0 → click Restart Pedacito; the GUI reopens on the same "
            "port with the new version",
            "A failed upgrade (or an editable/source install) shows the "
            "output plus the manual command instead",
            "Dismiss hides the banner until an even newer release appears",
            "Opt out entirely: ⚙ Settings → Updates → check_for_updates → "
            "off (zero network calls; the banner never appears); "
            "show_update_reminder = false hides the active-development update "
            "reminder on interactive exit and in the GUI",
        ],
        "keywords": ["update", "upgrade", "new version", "new release",
                     "update pedacito", "upgrade pedacito", "check for "
                     "updates", "check_for_updates", "show_update_reminder",
                     "update banner", "pipx upgrade", "restart"],
        "settings": ["check_for_updates", "show_update_reminder"],
        "cli": ["update"],
    },
    {
        "id": "profiling-auto-tune",
        "title": "Profile the hardware and auto-tune model budgets",
        "summary": "`pedacito profile --show` combines a vendor-neutral hardware "
                   "snapshot (free VRAM/RAM with a source per number), a cached "
                   "prefill/generation probe and the active profile's tuning. "
                   "Set auto_tune = true to apply it; the default is off.",
        "steps": [
            "Run `pedacito profile` (or `pedacito models`) to see RAM/VRAM, "
            "the reader that answered (drm-sysfs, nvidia-smi, vm_stat, "
            "winapi), the footprint confidence and the last probe",
            "Run `pedacito profile --run` to force one tiny completion plus a "
            "2k-token prefill (512 when the window is smaller); the result is "
            "cached for a week per server+model",
            "Set auto_tune = true in the profile (Settings → Profiles → "
            "Profiling and auto-tuning) to adjust context_tokens, retrieval "
            "budgets and timeout from free memory",
            "A tight pool shrinks to the 4096-token safety floor; spare "
            "headroom is only proposed as growth, never forced",
            "Tiered local-moe servers keep max_concurrency = 1 and the 600 s "
            "timeout; hosted APIs get only their 30 s timeout default",
            "Missing telemetry (for example AMD/Intel on Windows) makes "
            "auto-tune abstain and leaves every value untouched",
            "`pedacito profile --reset` drops the cached measurements and the "
            "optimistic opt-in; the TUI uses /profile show|run|reset",
        ],
        "keywords": ["profile", "profiling", "auto tune", "auto_tune",
                     "hardware", "vram", "ram", "nvidia", "amd", "intel",
                     "apple silicon", "windows", "macos", "prefill",
                     "tokens per second", "confidence", "kv cache",
                     "footprint", "context_tokens"],
        "settings": ["auto_tune", "context_tokens", "timeout",
                     "max_concurrency", "model_params_b",
                     "model_active_params_b", "model_layers",
                     "model_kv_heads", "model_head_dim", "model_quant",
                     "model_arch"],
        "cli": ["profile", "models scan", "profiles"],
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
            "Hunk-level sign-off: each change block inside a file is a Hunk "
            "N row with an accept-this-hunk-only checkbox; accepting a subset "
            "writes only the accepted hunks, and the file's SHA-256 is still "
            "re-checked before anything is written",
            "Each file shows clause badges (R1, R2, ...) for the parts of the "
            "task it serves; accepting only some files flags any clause left "
            "unaccepted",
            "Click Accept (or Accept N of M) / Reject",
        ],
        "keywords": ["review", "diff", "accept", "reject", "staged changes",
                     "before writing", "hunk", "hunks", "partial accept",
                     "sign off", "sign-off"],
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
                   "the Logs tab. Every profile carries a calibration badge "
                   "(✓ Calibrated, ⚠ stale, ○ Not calibrated, ☁ Hosted API). "
                   "Calibration is only for local models: hosted buttons are "
                   "disabled because hosted models use provider-table defaults "
                   "automatically. Calibrate All Local Models… measures the "
                   "whole local pool as one job.",
        "steps": [
            "Topbar Row 2 → Calibrate (uses the active profile); the badge "
            "beside the picker shows its calibration state, and clicking "
            "○ Not calibrated opens Settings → Model discovery, where the "
            "Calibrate buttons live",
            "Or ⚙ Settings → Model discovery / Profiles → Calibrate; the "
            "profile row says Calibrated on DATE (score, specialties) or Not "
            "calibrated yet",
            "Project ▾ → Calibrate All Local Models… (or the button in Model "
            "discovery): confirm the listed local profiles, optionally tick "
            "Re-calibrate models that already have a fresh card, and watch "
            "the rows tick off as each card is saved; hosted profiles are "
            "excluded",
            "Watch Bottom-right Workspace → Logs; the model card tunes the "
            "remaining defaults, including the architecture library's "
            "recommended temperature, sampler and context window for local "
            "models",
            "A calibrated temperature replaces only default-level values: a "
            "temperature you set on a profile, a PEDACITO_* environment "
            "variable or a CLI flag always wins over the card",
        ],
        "keywords": ["calibrate", "model card", "measure model", "tune model",
                     "benchmark", "calibrate all", "calibration badge",
                     "stale card", "hosted calibration", "architecture "
                     "defaults"],
        "settings": ["use_model_card"],
        "cli": ["calibrate", "models scan", "profiles"],
    },
    {
        "id": "live-telemetry",
        "title": "Read the live telemetry footer / CLI ribbon",
        "summary": "A footer under the GUI workspace tracks the context window, "
                   "turn and session tokens, generation speed, KV cache hits "
                   "and folds while a task runs; the CLI prints the same "
                   "numbers as a one-line ribbon on stderr after each task.",
        "steps": [
            "GUI: the telemetry footer shows Context: used / window (percent) "
            "with a green (<70%), amber (70-85%) or red (>85%) bar, turn and "
            "session tokens, tok/s (last call and session average), KV cache "
            "hit %, fold count, and the active model with its specialty plus "
            "the orchestrator profile",
            "Click the footer to expand the drawer with the full per-call "
            "numbers; while a specialist holds the card the last-call numbers "
            "are the specialist's",
            "CLI: after each task a ribbon prints on stderr, e.g. [telemetry "
            "| ctx: ████░░░░░░ 38% (6,210/16,384) | turn: 6,210 in / 185 out "
            "| 43.5 tok/s | cache: 82%]; show_telemetry = false hides it, and "
            "--jsonl / rpc modes never print it",
            "In chat, /status and /cost show the same context bar and speeds; "
            "the rpc status reply carries context_utilization_pct, "
            "last_tok_per_s and avg_tok_per_s",
            "GUI hardware badge: RAM and VRAM totals refresh every 3 s from "
            "/api/hardware and are also pushed live over SSE; server probes "
            "run in parallel, so one slow server does not stall the badge. "
            "It shows totals only, with no per-process GPU attribution",
        ],
        "keywords": ["telemetry", "context bar", "context usage", "tokens per "
                     "second", "tok/s", "speed", "cache hit", "footer",
                     "ribbon", "folds", "show_telemetry", "hardware", "ram",
                     "vram", "memory badge", "gpu usage", "api/hardware"],
        "settings": ["show_telemetry", "context_tokens"],
        "cli": ["ask", "chat", "rpc"],
    },
    {
        "id": "tune-small-models",
        "title": "Tune settings for small or weak models",
        "summary": "Auto-Tune sets the universal budgets; per-model guard "
                   "rails live in the profile's Reasoning & Execution "
                   "Overrides.",
        "steps": [
            "⚙ Settings → Auto-Tune → answer hardware, model class, codebase "
            "size, priority and the dual-GPU question → Preview recommendation "
            "→ Apply; a dual-GPU answer also offers Create starter Scout & "
            "pair… to split lookups onto a second card",
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
        "title": "Change the GUI color scheme",
        "summary": "Open the Theme Manager from the top bar. The choice is "
                   "saved in themes.json beside your config.",
        "steps": [
            "Top bar → palette button (theme manager and custom color editor)",
            "Pick one of the nine built-in schemes: slate, light, daylight, "
            "nord, midnight, jazz_solo, trapper_keeper, memphis_arcade or "
            "windbreaker (the TUI's /theme offers the same set)",
            "Or save a custom theme with ten color tokens; a project can pin "
            "its own theme in .pedacito/config.json",
        ],
        "keywords": ["theme", "color scheme", "colour scheme", "dark mode",
                     "light mode", "theme manager", "palette"],
        "settings": [],
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
            "The Raw file / reset tab never shows a stored key: it is masked "
            "as ***REDACTED***, and saving the mask unchanged keeps the stored "
            "value (a mask for a key the file does not have is refused)",
            "Set json_mode = object for hosted APIs that reject json_schema",
            "api.anthropic.com uses the native /v1/messages transport by "
            "default (anthropic_native = true): explicit cache_control "
            "breakpoints with a 5m/1h prompt_cache_ttl, "
            "cache_read_input_tokens / cache_creation_input_tokens usage, "
            "native extended thinking, tools and images; set "
            "anthropic_native = false for the OpenAI-compatible layer",
            "Google Gemini (generativelanguage.googleapis.com) uses the native "
            "generateContent transport by default (gemini_native = true; false "
            "uses the OpenAI-compatible shim). OpenAI o3/o4/GPT-5 reasoning "
            "models can opt into the Responses API with openai_responses = "
            "true, or provider = \"openai-responses\"",
            "Keep the key out of the file: api_key = \"keyring:<service>/"
            "<account>\" or api_key_keyring = \"<service>/<account>\" reads the "
            "OS keychain once the optional extra is installed: pip install "
            "\"pedacito[keyring]\"",
        ],
        "keywords": ["api key", "hosted", "openai", "deepseek", "anthropic",
                     "server url", "base_url", "api_key_env", "authentication",
                     "anthropic_native", "claude", "cache_control",
                     "messages api", "cache_read_input_tokens", "gemini",
                     "google", "generatecontent", "responses api", "o3",
                     "gpt-5", "keyring", "keychain"],
        "settings": ["base_url", "api_key", "api_key_env", "api_key_keyring",
                     "model", "json_mode", "tier", "anthropic_native",
                     "gemini_native", "openai_responses", "prompt_cache",
                     "prompt_cache_ttl"],
        "cli": ["config", "models scan", "calibrate"],
    },
    {
        "id": "provider-registry",
        "title": "Provider catalog, capabilities and opt-in probes",
        "summary": "One registry in pedacito/providers.py lists the hosted "
                   "providers and local runtimes, matches hosts exactly, and "
                   "reports only evidenced capabilities: unknown stays "
                   "unverified, never green.",
        "steps": [
            "⚙ Settings → Model discovery → Add Hosted Provider keeps the "
            "seven-entry quick-add catalog; the full registry also covers "
            "Groq, Cerebras, Fireworks, Together, xAI, Mistral, Moonshot/Kimi, "
            "MiniMax, Z.ai/GLM, NVIDIA NIM, Hugging Face, LiteLLM and more",
            "Local presets include LM Studio, Ollama, llama.cpp, vLLM, "
            "Lemonade, Strata/Avalon, MLX-LM, SGLang, TabbyAPI, KoboldCpp "
            "and LocalAI; an exact local runtime label counts as local, a "
            "look-alike provider name does not",
            "Host matching is equality or a dot-boundary suffix: "
            "api.openai.com.attacker.test and notapi.openai.com stay custom",
            "Capabilities are tri-state (None = unknown). A probe runs only "
            "with explicit consent, sends one minimal bounded request, refuses "
            "redirects and caches for 24 h beside config.toml; import and "
            "ordinary calls never probe",
            "Azure OpenAI, AWS Bedrock and Google Vertex are gateway-only: "
            "they need a configured OpenAI-compatible gateway and report "
            "unverified rather than claiming native auth",
            "Reasoning state survives tool turns: reasoning_content, signed "
            "Anthropic thinking blocks and Gemini thoughtSignature are replayed "
            "as protocol metadata, never as visible content",
        ],
        "keywords": ["provider catalog", "provider registry", "capabilities",
                     "capability matrix", "probe", "unverified", "azure",
                     "bedrock", "vertex", "gateway", "litellm", "groq",
                     "cerebras", "fireworks", "together", "xai", "mistral",
                     "moonshot", "kimi", "minimax", "z.ai", "glm", "nvidia",
                     "hugging face", "lm studio", "ollama", "llama.cpp", "vllm",
                     "mlx", "sglang", "tabbyapi", "koboldcpp", "localai",
                     "reasoning_content", "thoughtsignature", "host matching"],
        "settings": ["base_url", "prompt_cache", "json_mode", "tier",
                     "anthropic_native", "gemini_native", "openai_responses"],
        "cli": ["models scan", "config"],
    },
    {
        "id": "find-logs",
        "title": "Find logs and terminal output",
        "summary": "The Workspace Logs tab carries everything Pedacito prints; "
                   "session transcripts live under .pedacito/sessions/.",
        "steps": [
            "Bottom-right Workspace panel → Logs tab",
            "Use follow to auto-scroll; Clear empties the visible terminal",
            "Project ▾ → Clear terminal does the same from the topbar; "
            "Project ▾ → Background servers… tails a dev server's own log",
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
        "id": "project-upkeep",
        "title": "Project ▾ upkeep: memory, skills, background servers, "
                 "checks, prune, reset state",
        "summary": "The Project ▾ menu holds the upkeep dialogs: remembered "
                   "facts and repo lessons, the skills library & creator, "
                   "background dev servers, on-demand project checks, "
                   "pruning old sessions/backups, and a previewed reset of "
                   "the project's runtime state.",
        "steps": [
            "Project ▾ → Memory & repo lessons…: list, add or forget "
            "remembered facts (this project or all projects) and read the "
            "repair lessons; the same store as pedacito memory",
            "Project ▾ → Skills library & creator…: browse the nineteen "
            "built-in starters plus user/project skills, preview a task's "
            "trigger matches, and write or auto-draft custom skills "
            "(pedacito skills)",
            "Project ▾ → Background servers…: start a named dev server "
            "(e.g. npm run dev), tail its logs, stop it; the same engine as "
            "pedacito serve / pedacito procs",
            "Project ▾ → Run project checks…: run diagnose (the project's "
            "linter/typecheck), webcheck (static web contracts) or domcheck "
            "(headless DOM smoke, needs Playwright) on demand",
            "Project ▾ → Prune old sessions/backups…: Preview then Prune the "
            "oldest sessions/backups beyond the keep limits (pedacito prune)",
            "Project ▾ → Reset project state…: preview, then wipe "
            ".pedacito/ runtime state with Keep remembered facts / Keep "
            "authored rules, skills & prompts checkboxes and an optional "
            "reindex; source code and .pedacito.toml are never touched "
            "(pedacito reset)",
        ],
        "keywords": ["project menu", "upkeep", "memory dialog",
                     "skills library", "background servers", "dev server",
                     "project checks", "run checks", "prune sessions",
                     "prune backups", "reset project", "reset project state",
                     "wipe state", "repo lessons"],
        "settings": ["keep_sessions", "keep_backups", "micro_facts"],
        "cli": ["memory", "skills", "serve", "procs", "diagnose", "webcheck",
                "domcheck", "prune", "reset"],
    },
    {
        "id": "skills-library",
        "title": "Skills library & creator: browse, match, write skills",
        "summary": "Project ▾ → Skills library & creator… (also ⚙ Settings → "
                   "Quick links) browses the nineteen built-in starters plus "
                   "your user and project skills, previews which checklists "
                   "a task injects, and creates custom skills by hand or "
                   "with a model draft.",
        "steps": [
            "Project ▾ → Skills library & creator… opens the library; each "
            "row carries a scope badge (bundled / user / project)",
            "Type a task in the preview box and press Enter: the matching "
            "skills light up (top-2 trigger matches, ≤1,200 chars injected)",
            "Click a skill to read its triggers and checklist body",
            "+ New Custom Skill opens the editor: name (slug), description, "
            "comma-separated triggers and the checklist body",
            "✦ Auto-Draft from Description asks the active model to write "
            "the whole skill from your one-line description; review the "
            "draft before saving",
            "Save to Project writes .pedacito/skills/<name>.md; Save to "
            "Global writes ~/.config/pedacito/skills/<name>.md; Edit and "
            "Delete manage custom skills (built-ins are immutable)",
        ],
        "keywords": ["skill", "skills", "custom skill", "create skill",
                     "skill library", "skill creator", "draft skill",
                     "auto-draft", "triggers", "checklist", "delete skill"],
        "settings": ["use_skills", "max_skill_catalog_chars"],
        "cli": ["skills"],
    },
    {
        "id": "consolidate-session",
        "title": "Consolidate a long session into a handoff",
        "summary": "The Consolidate Session button in the telemetry footer "
                   "(or Project ▾ → Consolidate session…) distils the chat "
                   "transcript into a structured handoff under 400 tokens, "
                   "saves .pedacito/consolidated_session.md, and restarts "
                   "the chat from that handoff so the next turn starts "
                   "small. The Context Meter updates immediately.",
        "steps": [
            "Click Consolidate Session in the telemetry bar (footer) and "
            "confirm; a running job must be stopped first",
            "The handoff keeps four sections: Goal & Context, Files "
            "Modified & Key Decisions, Verified Test Status, Open Next Steps",
            "The transcript resets to the single handoff card; durable "
            "decisions are pinned to memory.md",
            "CLI twins: /consolidate (alias /compact) in pedacito chat, "
            "pedacito memory --consolidate, and the RPC consolidate method",
        ],
        "keywords": ["consolidate", "compact", "handoff", "session summary",
                     "context full", "long session", "compress session",
                     "checkpoint", "context meter"],
        "settings": ["use_memory", "context_tokens"],
        "cli": ["memory", "chat", "rpc"],
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
            "In a project's own .pedacito.toml these toggles stay ignored "
            "until you read that file and run pedacito trust PROJ; your user "
            "config and --config FILE need no approval",
        ],
        "keywords": ["allow shell", "run commands", "allow write", "browser",
                     "screenshots", "permissions", "tools", "sandbox", "mcp",
                     "model context protocol", "lsp", "language server"],
        "settings": ["allow_shell", "allow_write", "allow_browser",
                     "allow_screenshots", "shell_allowlist", "shell_deny",
                     "sandbox", "sandbox_network", "scrub_env", "allow_mcp",
                     "mcp_servers", "mcp_timeout", "allow_lsp", "lsp_servers",
                     "lsp_diagnostics_gate"],
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
            "Composer row → Post-edit verify checkbox (or TUI /verify on) detects "
            "the project runner and stores it in auto_verify; /verify status "
            "shows the gates, rounds and ratchet",
            "⚙ Settings → All settings → search verify_tests → on; set "
            "test_command if detection guesses wrong",
            "CLI: pedacito auto ... --apply --verify \"pytest -q\"; "
            "pedacito tests . --files src/app.py shows impacted tests",
            "A failure the command ledger already recorded before this run is "
            "labelled pre-existing and does not trigger repair",
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
            "Allow '<pattern>' in this project (GUI) or p in the TUI: a shell "
            "command matching a narrow rule skips the prompt in that project "
            "only. The rule is saved to shell_allowlist_by_project; package "
            "installs, shells and chained commands are never covered",
            "Deny & Redirect: refuses the request. The model is told to continue "
            "with local project files only. Stop, or letting the dialog time out, "
            "also denies",
            "The CLI has no dialog: a gated tool returns its 'disabled in config' "
            "error unless the matching setting is on",
        ],
        "keywords": ["permission", "permissions", "allow", "deny", "approve",
                     "approval", "always allow", "allow once", "popup", "dialog",
                     "blocked", "waiting", "redirect", "allow in this project",
                     "project allow rule", "shell allowlist", "allow rule"],
        "settings": ["enable_web", "allow_shell", "allow_fetch",
                     "shell_allowlist_by_project"],
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
            "A public name that resolves to a private address is refused too "
            "(DNS rebinding); the connection is pinned to the checked IP",
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
        "summary": "During gather the model can run a single scoped test (one test "
                   "file or node id) and read the live traceback before it writes a "
                   "fix. Python runs under pytest; *.test.* / *.spec.* files under "
                   "Vitest or Node's runner, *_test.go under go test, tests/*.rs "
                   "under cargo test. It is on by default and never runs a shell "
                   "command line.",
        "steps": [
            "Nothing to click: the run_test action is offered when the project has "
            "tests or the task names a failure",
            "The runner follows the target: pytest for Python, Vitest (else "
            "node --test) for *.test.* / *.spec.*, go test for *_test.go, "
            "cargo test --test NAME for tests/*.rs; every branch is argv-only, "
            "workspace-contained and capped at 15 seconds",
            "⚙ Settings → All settings → search enable_run_test_in_gather (on by default)",
            "When it is off, each run_test request appears in the Permission "
            "requested dialog under Test Execution",
            "Allow Once or Allow for This Run lets that request through; Always "
            "Allow turns the setting on",
        ],
        "keywords": ["run test", "run_test", "pytest", "traceback", "live test",
                     "test in gather", "investigate", "failing test", "scoped test",
                     "run tests during investigation", "vitest", "node test",
                     "go test", "cargo test"],
        "settings": ["enable_run_test_in_gather"],
        "cli": ["tests", "ask"],
    },
    {
        "id": "api-oracle",
        "title": "Look up exact library signatures before writing code",
        "summary": "When a task names a library that the project's manifests "
                   "declare exactly, the read-only api gather action returns "
                   "the package version and exact public signatures from static "
                   "Python .pyi/.py files plus distribution metadata, and from "
                   "contained node_modules *.d.ts declarations. It never imports "
                   "package code or uses the network by default. A miss is "
                   "unknown, not proof the API is missing.",
        "steps": [
            "Nothing to click: the api action appears only when the task names "
            "a declared dependency such as requests or react",
            "The model calls action=api with symbol=<library> and an optional "
            "pattern such as Session or useState; the answer lists exact "
            "signatures and the declaration source label",
            "⚙ Settings → All settings → search api_oracle_exec (off by default) "
            "to permit optional go doc, cargo metadata --offline --no-deps or "
            "Python inspect enrichment; it also needs allow_shell and an "
            "allowing policy",
            "Static results are cached under .pedacito/cache/api by package, "
            "environment, version and declaration hashes; a version or "
            "declaration change invalidates the old entry",
            "When the answer says unknown, confirm with read/grep or the "
            "library's own docs; do not assume the API does not exist",
        ],
        "keywords": ["api", "api oracle", "library", "signature", "signatures",
                     "stub", "pyi", "d.ts", "typescript declarations", "go doc",
                     "cargo metadata", "inspect", "api_oracle_exec", "offline",
                     "hallucination", "exact api"],
        "settings": ["api_oracle_exec", "allow_shell"],
        "cli": ["ask", "chat"],
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
            "webcheck is static (regex, notice-level). To prove the page actually "
            "MOUNTS, run pedacito domcheck PROJECT (headless Chromium; needs "
            "Playwright), or set verify_dom = true with allow_shell to gate edits on it",
        ],
        "keywords": ["dom", "html", "javascript", "js", "getelementbyid",
                     "queryselector", "script src", "stylesheet", "asset",
                     "broken link", "missing id", "frontend", "page does not load",
                     "contract", "webcheck", "domcheck", "mount", "headless"],
        "settings": ["frontend_contracts", "post_edit_check", "verify_dom",
                     "dom_entry", "dom_timeout"],
        "cli": ["webcheck", "domcheck", "auto"],
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
            "Lessons carry runs/recurred/avoided counters from observed checks; one "
            "that never avoids its failure kind across five runs moves to "
            "retired_lessons",
            "A lesson confirmed twice whose files share one directory is promoted "
            "into an auto-managed path-scoped rule (.pedacito/rules/auto_<dir>.md) "
            "shown only when the model reads a file in that directory",
            "To read or forget them: Project ▾ → Memory & repo lessons…, or "
            "open .pedacito/micro_facts.json in your editor; deleting the file "
            "forgets everything it recorded",
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
            "Quick path: ⚙ Settings → Profiles → Dual-GPU / Scout + Architect "
            "Pairing → pick the Architect (planner/editor) and a Scout, or "
            "➕ Create a starter Scout profile… (second server URL, model id, "
            "VRAM, tier) → Save pairing; it writes default_profile + "
            "scout_profile in one comment-preserving edit",
            "The Dual-GPU / Scout + Architect Pairing panel is collapsed by "
            "default; its header shows Paired or Not configured, and the "
            "Configure button opens the form",
            "⚙ Settings → Auto-Tune → answer the dual-GPU question; when a "
            "second card fits, the preview offers Create starter Scout & pair…",
            "Manual path: ⚙ Settings → Profiles: create or select a second "
            "profile for the small model, with its own base_url and model",
            "⚙ Settings → All settings → search scout_profile → choose that profile in "
            "the dropdown, or None (use active profile) to turn it off",
            "While it is active, the topbar shows a Scout: <profile> badge beside the "
            "model chip",
            "Keep the large model as the active profile so answers and edits stay on it",
        ],
        "keywords": ["scout", "scout profile", "scout model", "architect", "two gpus",
                     "dual gpu", "second gpu", "fast model", "lookup model",
                     "gather model", "split models", "small model for search",
                     "pairing", "scout pairing", "architect pairing"],
        "settings": ["scout_profile", "default_profile"],
        "cli": ["profiles", "config", "ask"],
    },
    {
        "id": "image-generation",
        "title": "Generate an image with a local image model",
        "summary": "Draw an image with an image-model profile (Lemonade "
                   "sd-cpp, Stable Diffusion, Flux or a hosted image "
                   "endpoint). The dialog, the CLI and the TUI all use the "
                   "image profile and save into the project under "
                   ".pedacito/images/ unless you choose a path.",
        "steps": [
            "Project ▾ → 🎨 Generate image…",
            "Type the prompt; optionally set Size, Steps, Seed, Negative "
            "prompt and a Save path, then click Generate",
            "No image profile yet? ⚙ Settings → Model discovery → Scan "
            "local servers; image generators get a 🎨 Image badge. Click "
            "Probe image to verify the endpoint with one fast 512x512 call",
            "Set image_profile to pick the profile and allow_image_gen = "
            "true to let the agent call generate_image on its own",
            "CLI: pedacito image \"PROMPT\"; TUI: /image PROMPT",
        ],
        "keywords": ["generate image", "image generation", "draw image",
                     "picture", "stable diffusion", "flux", "sd-cpp",
                     "lemonade", "image model", "image profile",
                     "text to image", "generate_image", "probe image"],
        "settings": ["image_profile", "allow_image_gen", "image_default_size",
                     "image_default_steps", "modality"],
        "cli": ["image", "models scan", "calibrate"],
    },
    {
        "id": "specialist-routing",
        "title": "Route tasks to specialist profiles (auto_route_specialties)",
        "summary": "With auto-routing on, each task runs on the profile whose "
                   "specialty tags fit it best: a coder model for code tasks, a "
                   "vision model for UI/SVG work, a reasoner for math. Tags come "
                   "from the model card (calibrate probes) or an explicit "
                   "specialties list, which always wins. Switches happen at task "
                   "boundaries only -- never mid-gather -- and on one local "
                   "server the outgoing model is unloaded first so the "
                   "specialist fits in VRAM. Profiles on different servers stay "
                   "warm together. Hosted (remote API) profiles are excluded by "
                   "default (auto_route_allow_hosted = false), so a local pool "
                   "never jumps to a paid cloud model; set it true to include "
                   "them.",
        "steps": [
            "Tag your profiles: ⚙ Settings → Profiles → [Select Profile] → "
            "Connection & Capabilities → specialties → toggle the tag pills "
            "(or type a custom lowercase slug and press Enter)",
            "Or let pedacito calibrate detect the tags: six micro-probes "
            "(SVG/UI, math, translation, SQL, testing, planning) plus the "
            "model name; the card shows them under specialties",
            "Turn routing on: the ✦ Auto-route checkbox in the composer row, or "
            "auto_route_specialties = true in config.toml, or --auto-route "
            "per run (--no-auto-route overrides the config)",
            "Plans can prefix a task with [specialty: TAG] to force a "
            "specialist; the planner groups same-specialty tasks to avoid "
            "extra VRAM swaps",
            "The Activity panel and the topbar show which specialist is "
            "running; an unreachable specialist falls back to the worker "
            "with one warning",
        ],
        "keywords": ["specialist", "specialties", "auto route", "auto-route",
                     "specialty routing", "route tasks", "coder model",
                     "model per task", "vram handoff", "model swap",
                     "unload model", "task routing"],
        "settings": ["auto_route_specialties", "auto_route_allow_hosted",
                     "specialties", "use_model_card"],
        "cli": ["profiles", "calibrate", "ask", "auto"],
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
        "id": "execution-mode",
        "title": "Choose the task engine: two_phase, interleaved or auto "
                 "(execution_mode)",
        "summary": "two_phase gathers context then answers with edit blocks; "
                   "interleaved runs one agentic conversation where the model "
                   "reads, edits, runs tests and fixes until the task is done. "
                   "auto picks interleaved for hosted APIs, local-MoE servers "
                   "and calibrated large local models whose card chose native "
                   "tool calls; tiny/small/medium local models stay two_phase.",
        "steps": [
            "⚙ Settings → Profiles → [Select Profile] → Reasoning & Execution "
            "Overrides → execution_mode",
            "Or per run: pedacito ask . -t \"...\" --execution-mode interleaved",
            "In the TUI: /mode interleaved (alias /exec-mode); /status shows "
            "the active and the resolved mode",
            "The GUI Activity pane shows live tool-call cards while an "
            "interleaved run works; review and dry-run use the same sign-off "
            "page as two_phase",
        ],
        "keywords": ["execution mode", "interleaved", "two phase", "agentic",
                     "agent loop", "engine", "tool loop", "auto mode"],
        "settings": ["execution_mode", "tool_mode"],
        "cli": ["ask", "auto"],
    },
    {
        "id": "local-acceleration",
        "title": "Speed up local servers: KV prompt cache and n-gram "
                 "speculation (cache_prompt, prompt_lookup_num_tokens)",
        "summary": "For self-hosted llama.cpp-style servers Pedacito adds "
                   "cache_prompt to every request so the KV prefix of the "
                   "replayed gather context stays warm between steps, and "
                   "can switch on n-gram speculative decoding. A server "
                   "that answers 400/422 with one of these keys is "
                   "downgraded on the spot and remembered per model.",
        "steps": [
            "Nothing to click: cache_prompt is on by default for local "
            "endpoints (loopback or a private network address)",
            "⚙ Settings → All settings → Enable N-Gram Prompt Lookup "
            "Speculation to also try prompt_lookup_num_tokens",
            "⚙ Settings → All settings → Prompt Lookup Draft Tokens sets "
            "the draft size (default 10)",
            "Hosted APIs never receive these keys; no fallback config is "
            "needed for them",
        ],
        "keywords": ["cache_prompt", "prompt lookup", "n-gram",
                     "speculative decoding", "speculation", "kv cache",
                     "local server speed", "llama.cpp", "lm studio",
                     "ollama", "vllm", "faster gather", "prompt caching "
                     "local"],
        "settings": ["enable_prompt_cache", "enable_prompt_lookup",
                     "prompt_lookup_num_tokens"],
        "cli": ["config", "calibrate"],
    },
    {
        "id": "named-mcp-servers",
        "title": "Connect an MCP server and use its tools safely "
                 "(mcp_call)",
        "summary": "Named [mcp_servers.<name>] tables describe a stdio "
                   "server (command + args) or a Streamable HTTP one (url). "
                   "Gather shows the model only the tools matching the "
                   "current task, and a server not marked trusted asks the "
                   "permission broker before anything runs.",
        "steps": [
            "In the GUI: ⚙ Settings → MCP Servers → add a server (stdio "
            "command + args, or an http(s) url), Test it, then enable or "
            "trust it; servers are written to the user config, and edit or "
            "delete them from the same tab",
            "Or add a table to the config: [mcp_servers.docs] with "
            "command = \"python\" and args = [\"server.py\"] (or url = "
            "\"http://localhost:8000/mcp\")",
            "Set trusted = true only for a server you fully approve; "
            "otherwise every call shows a permission dialog first",
            "pedacito mcp list shows what each server exposes",
            "During a run the permission dialog arrives in the chat; "
            "GET /api/mcp/tools serves the same listing to the interface",
            "The legacy allow_mcp + mcp_servers command list still works "
            "for the old `mcp` catalog action",
        ],
        "keywords": ["mcp", "model context protocol", "external tool",
                     "mcp server", "mcp_call", "mcp tools", "plugin",
                     "connector", "stdio server", "http mcp",
                     "mcp servers tab", "add mcp server"],
        "settings": ["enable_mcp", "mcp_servers", "mcp_timeout",
                     "allow_mcp"],
        "cli": ["mcp", "trust", "config"],
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
    {
        "id": "creative-mode",
        "title": "Creative / design mode: the Builder persona "
                 "(creative_mode, creative_auto)",
        "summary": "Design tasks -- websites, landing pages, dashboards, "
                   "restyles, .html/.css/.svg/.jsx/.tsx/.vue/.svelte work -- "
                   "run as a Senior Product Engineer & Visual Designer: "
                   "warmer answer sampling (>= 0.6 on local servers), an "
                   ">= 8192-token reply budget, and whole-file HTML/CSS "
                   "rewrites instead of the smallest-diff discipline. "
                   "creative_auto detects them by default; creative_mode "
                   "forces it for every task. Hosted APIs keep their own "
                   "sampling and receive no temperature key.",
        "steps": [
            "In the GUI: the ✨ Creative button above the composer forces the "
            "Builder persona for the next Task/Auto/New run; with it off, "
            "design tasks are still auto-detected",
            "In the TUI: /creative on (every task), /creative auto (detect "
            "design tasks, the default), /creative off (surgical edits "
            "everywhere); /status shows the current state",
            "On the CLI: --creative forces it for the run (ask, auto, new, "
            "chat, tui)",
            "Creative sway: creative_temperature_delta adds a relative extra "
            "temperature on top of the answer baseline (unset = +0.20 for "
            "tiny/small, +0.30 otherwise; explicit values 0.0-0.8). It never "
            "drops below the baseline, so a calibrated 0.6 model reaches 0.9",
            "Or set creative_mode = true / creative_auto = false in "
            "config.toml (⚙ Settings → All settings → search creative_mode)",
            "The fragment veto and the small-window oversized-rewrite brake "
            "stand down for whole .html/.htm/.css/.md rewrites and creative "
            "tasks; the wipe refusal and the JS export guard still apply",
        ],
        "keywords": ["creative", "creative mode", "design", "design mode",
                     "builder", "builder mode", "ui polish", "designer persona",
                     "website", "landing page", "portfolio", "dashboard",
                     "restyle", "redesign", "whole file rewrite",
                     "html rewrite", "css rewrite", "visual design"],
        "settings": ["creative_mode", "creative_auto", "temperature_answer"],
        "cli": ["ask", "auto", "new", "chat", "tui"],
    },
    {
        "id": "strata-onboarding",
        "title": "Connect Strata / Avalon (local MoE) on port 8080",
        "summary": "Strata and Avalon serve an OpenAI-compatible API, by default "
                   "at http://127.0.0.1:8080/v1. Scan finds them, asks for an "
                   "API key only when the server requires one, and reads "
                   "STRATA_API_KEY (or AVALON_API_KEY) when the field is empty. "
                   "A key-protected server shows an API key required badge.",
        "steps": [
            "⚙ Settings → Model discovery → Scan local servers: the list "
            "includes the Strata / Avalon (Local MoE) preset at "
            "http://127.0.0.1:8080/v1",
            "If the server answers API key required, paste the key into the "
            "API key field next to the scan, or leave it empty and export "
            "STRATA_API_KEY before starting Pedacito",
            "Tick the models and import them: they become local-moe profiles, "
            "and the Servers list keeps the API key required badge until a "
            "key is available",
            "Loading and unloading models use the profile's api_key, so set it "
            "once in ⚙ Settings → Profiles, or rely on STRATA_API_KEY",
            "Lemonade and Strata can share one machine: with auto_evict_servers "
            "= true (the default), a run first unloads the other local "
            "server's model, so the target server has the memory it needs; "
            "set it to false to keep every model loaded",
            "CLI equivalent: pedacito models scan --url http://127.0.0.1:8080/v1 "
            "--api-key-env STRATA_API_KEY",
        ],
        "keywords": ["strata", "avalon", "local moe", "local-moe", "8080",
                     "port 8080", "lemonade", "auto evict", "auto_evict_servers",
                     "api key required", "auth required", "STRATA_API_KEY",
                     "onboarding", "scan strata", "tiered moe"],
        "settings": ["auto_evict_servers", "api_key", "api_key_env", "base_url"],
        "cli": ["models scan", "config"],
    },
    {
        "id": "model-catalog-refresh",
        "title": "Keep hosted context windows and prices current (models.dev)",
        "summary": "Hosted providers get their context window and $/Mtok prices "
                   "from three tiers: the provider's own /models metadata (tier "
                   "1), a 24-hour models.dev cache (tier 2) and a built-in "
                   "static price table (tier 3). Explicit price settings always "
                   "win.",
        "steps": [
            "Automatic: with auto_refresh_model_catalog = true (the default), "
            "probing or refreshing a hosted provider quietly refreshes the "
            "models.dev cache once it is older than 24 hours",
            "GUI: ⚙ Settings → Model discovery → Add Hosted Provider → Refresh "
            "catalog updates the cache now; the price column shows the input / "
            "output $/Mtok and the context window per model",
            "CLI: pedacito models refresh (add --json for the result)",
            "Strictly offline: auto_refresh_model_catalog = false. Tier 2 cache "
            "data and the built-in table still apply",
        ],
        "keywords": ["models refresh", "model catalog", "models.dev", "pricing",
                     "prices", "cost", "context window", "catalog sync",
                     "refresh catalog", "24 hours", "3-tier", "provider models",
                     "price table"],
        "settings": ["auto_refresh_model_catalog"],
        "cli": ["models refresh", "models scan"],
    },
    {
        "id": "language-switch",
        "title": "Switch the interface language: English / Spanish "
                 "(language, llm_reply_language)",
        "summary": "Pedacito's GUI, TUI and CLI speak English and Spanish. "
                   "Switch live in the GUI or TUI, per run on the CLI, or "
                   "permanently with the language setting; "
                   "llm_reply_language separately controls the language of "
                   "the model's prose explanations.",
        "steps": [
            "In the GUI: click the EN | ES switcher in the top bar; the "
            "choice is saved to the language setting",
            "In the TUI: press F3 to toggle EN/ES, or type /lang es "
            "(/lang en)",
            "On the CLI: pass --lang es (for example pedacito --lang es "
            "gui) or set PEDACITO_LANG=es (alias PEDACITO_LANGUAGE)",
            "Or set language = \"es\" in config.toml (⚙ Settings → All "
            "settings → search language)",
            "To choose the language of the model's explanations, set "
            "llm_reply_language to match_ui (follow the UI), auto "
            "(follow your prompt), en or es",
        ],
        "keywords": ["language", "idioma", "spanish", "español", "english",
                     "translate", "locale", "switch language", "lang",
                     "en es", "interface language", "localization",
                     "i18n", "reply language"],
        "settings": ["language", "llm_reply_language"],
        "cli": ["gui", "tui", "config"],
    },
    {
        "id": "live-preview",
        "title": "Live Preview: see the project's HTML as you build it "
                 "(🌐 Preview tab)",
        "summary": "When the project has .html pages, the workspace panel "
                   "grows a 🌐 Preview tab: a read-only iframe served from "
                   "/preview/ inside the project root. It reloads when a run "
                   "finishes -- and while a run writes -- so a website task "
                   "shows its result immediately.",
        "steps": [
            "Open a project with at least one .html file: the 🌐 Preview tab "
            "appears beside Files / Backups / Logs",
            "Pick the page in the dropdown (index.html is the default); "
            "relative CSS/JS/image links resolve through the same read-only "
            "route",
            "⟳ Refresh reloads the page; ↗ New tab opens it in a full browser "
            "tab",
            "After Ask/Auto/New/Apply finishes -- and while a run changes "
            "files -- the preview reloads automatically",
            "Safety: the route never leaves the project root (.. and symlinks "
            "are refused), never serves sensitive files, and only the app's "
            "own origin may frame it",
        ],
        "keywords": ["preview", "live preview", "iframe", "html preview",
                     "web preview", "see the page", "preview tab",
                     "refresh page", "website preview", "live view"],
        "settings": [],
        "cli": ["gui"],
    },
    {
        "id": "hosted-sampling",
        "title": "Hosted APIs keep their own sampling "
                 "(send_temperature_to_hosted, hosted_temperature)",
        "summary": "A hosted provider (any non-local base_url) tunes sampling "
                   "per model itself, so pedacito sends it NO temperature, "
                   "top_p, top_k, min_p or repeat_penalty by default -- "
                   "including keys from the sampler table. Local servers "
                   "(loopback or LAN) keep every configured value, and a "
                   "repetition retry gets one gentle presence_penalty = 0.15 "
                   "nudge instead of a cold temperature.",
        "steps": [
            "Nothing to click: hosted requests omit the five sampling keys by "
            "default, so the provider's tuned defaults apply",
            "To send the configured temperature anyway: "
            "send_temperature_to_hosted = true (⚙ Settings → All settings → "
            "search send_temperature_to_hosted); the other four keys stay "
            "omitted",
            "To pin one temperature for hosted endpoints (e.g. 1.0 for "
            "Claude): hosted_temperature = 1.0, globally or in a profile; it "
            "wins over phase overrides",
            "Local servers keep temperature/top_p/top_k/min_p/repeat_penalty; "
            "temperature_answer defaults to 0.3 (cool, not greedy) and "
            "creative tasks run at >= 0.6",
        ],
        "keywords": ["hosted temperature", "sampling", "temperature",
                     "top_p", "top_k", "min_p", "repeat_penalty",
                     "provider defaults", "api sampling", "send temperature",
                     "pinned temperature", "presence penalty", "repetition"],
        "settings": ["send_temperature_to_hosted", "hosted_temperature",
                     "temperature", "temperature_answer", "sampler"],
        "cli": ["config", "profiles"],
    },
    {
        "id": "git-commit-pr",
        "title": "Commit and open a pull request with guided git (commit, pr)",
        "summary": "pedacito commit reviews the staged diff, drafts the message and "
                   "commits locally; pedacito pr drafts the title and body and opens "
                   "the pull request with gh. A secret pre-flight scan runs before "
                   "every commit, and nothing here pushes or force-pushes.",
        "steps": [
            "CLI: pedacito commit . --all (add -m MSG to write the message yourself, "
            "--no-model for a deterministic draft)",
            "The scan refuses a staged diff that matches a secret pattern; remove the "
            "secret, or pass --allow-secrets only if you have checked it is safe",
            "Push the branch yourself, then pedacito pr . --base main --draft "
            "(--dry-run prints the title and body without calling gh)",
            "TUI: /git offers commit (never pushes) or creating a branch",
            "GUI: the Git drawer offers the same staged diff, secret scan and draft; "
            "it never pushes",
        ],
        "keywords": ["git", "commit", "commit message", "pull request", "pr", "gh",
                     "github", "stage", "secret scan", "guided git", "gitops",
                     "draft pr", "git drawer"],
        "settings": ["allow_git", "git_context"],
        "cli": ["commit", "pr"],
    },
    {
        "id": "acp-bridge",
        "title": "Use Pedacito from an editor over ACP (pedacito acp)",
        "summary": "pedacito acp speaks the Agent Client Protocol (JSON-RPC 2.0) on "
                   "stdin and stdout. It wraps the same warm engine as pedacito rpc, "
                   "so an editor or agent host can bind a project, send prompts with "
                   "@file and image blocks, and answer permission prompts.",
        "steps": [
            "Point the agent host at the command: pedacito acp /path/to/project",
            "Optional: --profile NAME and --execution-mode interleaved for the session",
            "The host sends initialize, then session/new (binds the project), then "
            "session/prompt; updates stream back as session/update notifications",
            "session/request_permission carries gated tool calls; session/cancel stops "
            "the running task without killing the process",
            "Need a simpler protocol? pedacito rpc --describe prints the JSONL method list",
        ],
        "keywords": ["acp", "agent client protocol", "json-rpc", "jsonrpc", "editor",
                     "zed", "agent host", "stdio", "session", "protocol", "bridge"],
        "settings": ["execution_mode", "allow_shell", "allow_write"],
        "cli": ["acp", "rpc"],
    },
    {
        "id": "parsers-prefetch",
        "title": "Prepare tree-sitter grammars for offline use (parsers prefetch)",
        "summary": "Tree-sitter grammars are fetched on first use. Run pedacito "
                   "parsers prefetch while online to warm them for air-gapped work; "
                   "when a grammar is missing the fetch is retried and the run falls "
                   "back to the built-in parser instead of failing.",
        "steps": [
            "Online: pedacito parsers prefetch (or --languages python javascript go)",
            "The output lists ok or failed for each language",
            "Offline: the same commands keep working; a missing grammar falls back "
            "quietly and is reported once",
        ],
        "keywords": ["parsers", "parser prefetch", "tree-sitter", "tree sitter",
                     "grammar", "offline", "air-gapped", "no internet", "retry"],
        "settings": [],
        "cli": ["parsers", "index"],
    },
    {
        "id": "tool-mode-plain",
        "title": "Plain-word gather format for small models (tool_mode = \"plain\")",
        "summary": "tool_mode = \"plain\" asks for each gather step as short plain "
                   "words (for example: read app.py 1 40) instead of JSON or native "
                   "tool calls. It suits small models that stumble on JSON. "
                   "calibrate A/B-tests it against the grammar format before a card "
                   "recommends it.",
        "steps": [
            "⚙ Settings → Profiles → [Select Profile] → Reasoning & Execution "
            "Overrides → tool_mode → plain",
            "Or in config.toml, per profile: tool_mode = \"plain\"",
            "Run pedacito calibrate on that profile to A/B the plain format; the "
            "card keeps the choice that scores better",
            "Valid values are grammar (default), native and plain",
        ],
        "keywords": ["tool mode", "plain", "plain words", "gather format",
                     "small model", "json", "tool_mode", "gather dsl",
                     "format", "calibrate"],
        "settings": ["tool_mode", "enforce_json_schema", "execution_mode"],
        "cli": ["calibrate", "config"],
    },
    {
        "id": "proactive-escalation",
        "title": "Escalate hard tasks before they fail (proactive_escalation)",
        "summary": "proactive_escalation = true adds a two-path router that runs once "
                   "per task. A zero-token structural gate (import fan-out, files "
                   "touched, module boundaries, open-ended architecture words) sends "
                   "a hard task straight to the escalate role; a Scout classifier "
                   "keeps well-defined single-file tasks local. It is off by default "
                   "and needs roles.escalate.",
        "steps": [
            "Configure a stronger model for the escalate role in [roles]",
            "Set proactive_escalation = true (⚙ Settings → All settings → search "
            "proactive_escalation)",
            "Leave it false and the reactive auto_escalate behaviour stays unchanged",
            "A task that escalates shows the escalate role in the Activity panel",
        ],
        "keywords": ["escalate", "escalation", "proactive", "router", "complexity",
                     "hard task", "stronger model", "roles", "architect",
                     "scout classifier"],
        "settings": ["proactive_escalation", "auto_escalate", "roles"],
        "cli": ["ask", "auto", "config"],
    },
    {
        "id": "scratch-offload",
        "title": "Oversized tool output is offloaded to .pedacito/scratch/",
        "summary": "A tool observation that is too large for the context window is "
                   "written to .pedacito/scratch/ and replaced by a short pointer. "
                   "The model reads it back with the normal read and grep actions, "
                   "and the scratch area stays inside the project.",
        "steps": [
            "Nothing to click: offloading happens when an observation exceeds "
            "max_observation_tokens",
            "Find a file the model offloaded: look under .pedacito/scratch/",
            "The model can read and grep these logs but never writes to them; only "
            "direct *.log files under .pedacito/scratch/ are readable this way",
            "Delete .pedacito/scratch/ at any time; it is rebuilt when needed",
        ],
        "keywords": ["scratch", "scratchpad", "offload", "observation", "large output",
                     "truncated output", "context", "tool output", "scratch files"],
        "settings": ["max_observation_tokens", "wrap_observations"],
        "cli": ["ask", "chat"],
    },
    {
        "id": "tui-rich-view",
        "title": "Rich terminal view: diffs, tool cards and the @ picker (tui)",
        "summary": "The TUI renders edits as inline syntax-highlighted unified diffs, "
                   "shows each tool call as a one-line card you can expand, and opens "
                   "a fuzzy file picker on @. It needs no extra dependency.",
        "steps": [
            "Tool calls appear as collapsible one-line cards: Ctrl+O toggles the "
            "latest one; /expand [N|all] and /collapse [N|all] work on any card",
            "Type @ in the prompt to open the six-row fuzzy picker over files, symbols "
            "and images; Up/Down move, Tab or Enter inserts, Esc dismisses",
            "Mouse: click [EN|es] to switch language, a sidebar tab, a tool card or the "
            "scrollbar; /mouse on enables SGR capture and /mouse off restores selection",
            "/status is localized under /lang es, and a skipped test run reads "
            "tests: skipped (needs allow_shell)",
        ],
        "keywords": ["tui", "diff", "highlight", "tool card", "collapse", "expand",
                     "ctrl+o", "/expand", "/collapse", "picker", "fuzzy", "@",
                     "mouse", "click", "sgr", "scrollbar", "terminal"],
        "settings": ["show_telemetry", "allow_shell"],
        "cli": ["tui"],
    },
    {
        "id": "local-network-endpoints",
        "title": "Local, LAN and Tailscale endpoints (CGNAT 100.64.0.0/10)",
        "summary": "Pedacito treats loopback, .local names, RFC 1918 LAN, Tailscale "
                   "(RFC 6598 CGNAT 100.64.0.0/10), link-local, IPv6 ULA and explicit "
                   "local provider labels as local. Local models get the local "
                   "sampling, cache and calibration rules, and a local card that "
                   "lacks calibrated temperatures is marked stale.",
        "steps": [
            "Point base_url at a Tailscale address, for example "
            "http://100.101.102.103:1234/v1 or http://box.tail1234.ts.net:11434/v1",
            "The endpoint counts as local, so cache_prompt and the local sampler apply",
            "A local model card without a calibrated temperature is marked stale: run "
            "pedacito calibrate --recalibrate once to refresh it",
        ],
        "keywords": ["tailscale", "cgnat", "100.64", "lan", "local network",
                     "private network", "rfc 6598", "ipv6", "ula", "link local",
                     "stale card", "recalibrate", "base_url", "remote server",
                     "connect", "connect tailscale", "over tailscale"],
        "settings": ["base_url", "use_model_card", "prompt_cache"],
        "cli": ["calibrate", "config", "models scan"],
    },
    {
        "id": "windows-stderr",
        "title": "Capture stderr on Windows PowerShell 5.1 (2>&1)",
        "summary": "Pedacito writes human text to stderr, and under --jsonl, rpc and "
                   "acp stdout must stay pure JSON. On Windows PowerShell 5.1, "
                   "2>&1 wraps each stderr line as an error record, which looks like "
                   "a failure and can corrupt piped JSON.",
        "steps": [
            "Windows PowerShell 5.1: pedacito ask . -t \"...\" 2>&1 shows stderr as "
            "red ErrorRecord lines, even when the run succeeds",
            "For clean JSON on stdout, discard stderr: pedacito ask . -t \"...\" --json "
            "2>$null",
            "PowerShell 7+ and cmd.exe merge stderr as plain text",
        ],
        "keywords": ["powershell", "powershell 5.1", "windows", "2>&1", "stderr",
                     "redirect", "error record", "nativecommanderror", "stdout json",
                     "pipe", "windows terminal"],
        "settings": [],
        "cli": ["ask", "rpc"],
    },
]

TOPICS = [
    ("Install Pedacito",
     "How do I install pedacito?"),
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
    ("Creative / design mode",
     "How do I turn on creative mode for design tasks?"),
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
    "specialties": "⚙ Settings → Profiles → [Select Profile] → Connection & "
                   "Capabilities → specialties (tag pills + custom tag)",
    "auto_route_specialties": "Composer row → ✦ Auto-route checkbox, or ⚙ Settings "
                              "→ All settings → search auto_route_specialties",
    "auto_verify": "Composer row → Post-edit verify checkbox, TUI /verify on, or "
                   "⚙ Settings → All settings → search auto_verify",
    "creative_mode": "✨ Creative button above the composer, or ⚙ Settings → "
                     "All settings → search creative_mode",
    "creative_auto": "⚙ Settings → All settings → search creative_auto "
                     "(auto-detect design tasks when creative_mode is off)",
    "auto_route_allow_hosted": "⚙ Settings → All settings → search "
                               "auto_route_allow_hosted (off = local-only "
                               "routing)",
    "gui_theme": "Topbar → theme dropdown, or ⚙ Settings → All settings → "
                 "Graphical interface → gui_theme",
    "picard_mode": "⚙ Settings → Captain Picard Mode checkbox, or TUI /picard "
                   "on|off, or ⚙ Settings → All settings → Graphical interface "
                   "→ picard_mode",
    "check_for_updates": "⚙ Settings → All settings → Updates → "
                         "check_for_updates (off = fully offline: no release "
                         "check, no update banner)",
    "show_update_reminder": "⚙ Settings → All settings → Updates → "
                            "show_update_reminder (off = hide the "
                            "active-development update reminder)",
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
    "execution_mode": "⚙ Settings → Profiles → [Select Profile] → Reasoning & "
                      "Execution Overrides → execution_mode (or "
                      "--execution-mode per run, TUI /mode live)",
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
    "prompt_pack", "answer_mode", "edit_style", "tool_mode", "execution_mode",
    "best_of", "fuzzy_match", "temperature", "temperature_answer",
    "max_output_tokens", "context_tokens",
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
    "check_for_updates": ["update", "upgrade", "new version", "new release",
                          "check for updates", "update check", "phone home",
                          "offline", "opt out", "update banner"],
    "show_update_reminder": ["update reminder", "active development",
                             "reminder", "exit reminder", "update tip"],
    "picard_mode": ["picard", "make it so", "hágalo así", "captain picard",
                    "star trek", "button labels"],
    "review": ["review page", "diff review", "approve changes"],
    "backup": ["backup", "snapshot", "undo", "restore"],
    "allow_shell": ["run commands", "shell", "tool permission", "allow run"],
    "allow_write": ["write files", "gather write", "allow write"],
    "allow_browser": ["browser", "playwright", "web page", "screenshots"],
    "allow_screenshots": ["screenshot", "screen capture", "vision"],
    "auto_verify": ["verify command", "run tests after task", "auto verify",
                    "post-edit verify"],
    "incremental_edit_loop": ["small model edits", "incremental edit",
                              "symbol edits"],
    "edit_plan_brief": ["edit plan", "brief before edits"],
    "step_reasoning_first": ["reasoning first", "reason before action"],
    "execution_mode": ["execution mode", "interleaved", "two phase",
                       "agentic loop", "agent loop", "task engine"],
    "max_read_tokens": ["read budget", "token budget", "small context"],
    "max_gathered_chars": ["gather budget", "context budget", "small window"],
    "max_total_model_calls": ["call cap", "model call budget"],
    "post_edit_check": ["syntax check", "lint after edit", "check edit"],
    "autofix_enabled": ["autofix", "auto fix", "lint fix", "auto repair",
                        "fix formatting", "format before repair"],
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
    "anthropic_native": ["anthropic", "claude", "messages api",
                         "native transport", "cache_control"],
    "gemini_native": ["gemini", "google", "generateContent",
                      "native transport", "x-goog-api-key"],
    "openai_responses": ["openai", "responses api", "o3", "o4", "gpt-5",
                         "reasoning", "previous_response_id"],
    "auto_refresh_model_catalog": ["model catalog", "models.dev", "pricing",
                                   "context window", "provider refresh"],
    "api_key_keyring": ["keyring", "keychain", "credential", "secret",
                        "api key", "password"],
    "shell_allowlist_by_project": ["allow in this project", "project allow",
                                   "shell rule", "always allow shell",
                                   "per project shell", "skip prompt"],
    "creative_temperature_delta": ["creative temperature", "design sway",
                                   "creativity", "warmer", "design mode"],
    "auto_evict_servers": ["evict", "unload other servers", "lemonade",
                           "strata", "memory", "vram", "coexist"],
    "enable_prompt_cache": ["cache_prompt", "kv cache", "local server "
                            "cache", "llama.cpp cache", "faster local"],
    "enable_prompt_lookup": ["prompt lookup", "n-gram", "speculative "
                             "decoding", "speculation", "draft tokens"],
    "prompt_lookup_num_tokens": ["lookup tokens", "draft tokens",
                                 "speculation size"],
    "enable_mcp": ["mcp", "model context protocol", "mcp_call",
                   "external tools", "mcp servers"],
    "temperature": ["temperature", "sampling", "creativity"],
    "max_output_tokens": ["output limit", "reply cap", "hosted limit"],
    "context_tokens": ["context window", "window size", "context length"],
    "use_model_card": ["model card", "calibrate", "tuned defaults"],
    "sandbox": ["sandbox", "bubblewrap", "safety", "isolated commands"],
    "permission_mode": ["permission mode", "policy", "approval mode",
                        "capability", "read only mode", "plan mode",
                        "accept edits", "workspace write", "danger",
                        "permission", "permissions", "politica"],
    "approval_policy": ["approval policy", "approval", "auto-safe",
                        "auto safe", "never ask", "unattended", "headless",
                        "approve"],
    "policy_file": ["policy file", "policy.toml", "capability policy",
                    "permission policy", "trust", "politica"],
    "subagent_max_parallel": ["parallel subagents", "subagent parallel",
                              "subagent concurrency"],
    "daemon_port": ["daemon port", "background daemon port"],
    "daemon_auto_start": ["daemon", "auto start daemon", "background daemon"],
    "auto_install_deps": ["pip install", "missing module", "dependency"],
    "enable_run_test_in_gather": ["run tests in gather", "pytest during "
                                  "investigation", "live traceback", "run_test"],
    "frontend_contracts": ["dom check", "asset check", "script src",
                           "getElementById", "frontend gate", "html js ids"],
    "enforce_json_schema": ["json schema", "grammar", "structured output",
                            "json grammar", "response_format"],
    "scout_profile": ["scout", "two gpus", "fast lookup model", "architect"],
    "specialties": ["specialty", "specialist", "tags", "coding model",
                    "what is this model good at", "model tags"],
    "auto_route_specialties": ["auto route", "auto-route", "specialist routing",
                               "route tasks", "model per task", "vram handoff",
                               "model swap"],
    "micro_facts": ["lessons", "gotchas", "anti patterns", "repo facts",
                    "remember commands", "micro facts"],
    "max_micro_fact_tokens": ["lesson budget", "micro fact budget"],
    "edit_style": ["symbol edit", "anchor edit", "splice", "search replace"],
    "creative_mode": ["creative", "creative mode", "design mode", "builder",
                      "ui polish", "designer persona", "website builder"],
    "creative_auto": ["creative auto", "detect design tasks", "auto creative",
                      "design detection"],
    "send_temperature_to_hosted": ["hosted temperature", "send temperature",
                                   "provider sampling", "api temperature",
                                   "temperature opt-in"],
    "hosted_temperature": ["hosted temperature", "pinned temperature",
                           "claude temperature", "provider temperature"],
}


def _fallback_schema() -> list[dict]:
    """A minimal schema when the GUI package cannot be imported."""
    from dataclasses import MISSING, fields
    from . import config_editor
    help_by = config_editor.reference_setting_help()
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
                    "group": "General", "help": help_by.get(field.name, ""),
                    "choices": None, "secret": False})
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


_ES_SUMMARIES = {
    "gui:permission-dialog": "Cuando el modelo pide algo que requiere permiso (acceso web, instalar un paquete, un comando de shell no aprobado o un archivo sensible), la ejecución se pausa hasta que respondas.",
    "gui:set-up-config": "Usa el asistente de configuración de la primera ejecución en Ajustes, o crea y prueba una configuración mínima desde la CLI.",
    "gui:find-settings": "Todos los ajustes es un formulario con búsqueda para cada clave de configuración, aplicado globalmente o a un solo perfil.",
    "gui:language-switch": "La GUI, la TUI y la CLI funcionan en inglés y español. Cambia el idioma en vivo en la GUI o la TUI, por ejecución en la CLI, o de forma permanente con el ajuste language; llm_reply_language controla por separado el idioma de las respuestas del modelo.",
    "gui:plan-only": "Marca Solo plan en las opciones de Auto, u omite --apply en la CLI; edita la lista de tareas en el panel Plan.",
    "gui:run-auto": "Dale a Auto un objetivo; escribe una lista de tareas y las trabaja una por una, con verificación opcional de cada tarea.",
    "gui:switch-model": "Elige otro perfil en la barra superior y pulsa Cambiar; el proyecto abierto se recarga con ese modelo.",
    "gui:reset-config": "Restablece la configuración del usuario a los valores recomendados de Pedacito; antes se respalda el archivo anterior.",
    "gui:undo-run": "Cada escritura se guarda antes en una instantánea; restaura la más reciente o una elegida desde el panel Espacio de trabajo.",
    "gui:allow-tools": "Son ajustes de confianza opcionales en Todos los ajustes; permiten que el modelo ejecute comandos o escriba durante la fase de recopilación.",
    "gui:web-access": "El acceso web está desactivado por defecto. fetch_url y web_search se ejecutan si enable_web está activo, o si apruebas cada llamada en el diálogo de permisos.",
    "gui:api-key-hosted": "Pon la URL del servidor, el modelo y la clave de API en el perfil; para APIs alojadas usa api_key_env para que el secreto no quede en el archivo.",
    "gui:provider-registry": "Un solo registro en pedacito/providers.py reúne proveedores alojados y runtimes locales, coincide con los hosts de forma exacta y solo informa capacidades con evidencia: desconocido queda como no verificado, nunca en verde. Los sondeos solo se ejecutan con consentimiento explícito, hacen una petición mínima, rechazan redirecciones y se guardan 24 horas; Azure, Bedrock y Vertex son solo pasarela y necesitan un gateway compatible con OpenAI.",
    "gui:index-project": "El botón Indexar de la barra superior construye el mapa; Mayús+clic crea un índice rápido sin resúmenes del modelo.",
    "gui:install-pedacito": "Instala desde PyPI con `pip install pedacito` (o `pipx install pedacito` para una CLI aislada); se requiere Python 3.11+. Ambos dejan `pedacito` y el alias corto `pdc` en tu PATH.",
    "gui:verify-tests": "El modo Auto puede ejecutar un comando de verificación tras cada tarea; Task/Edit puede correr las pruebas del proyecto tras las ediciones aplicadas.",
    "gui:green-ratchet": "Auto guarda una instantánea antes de cada paso. Un paso que sigue fallando tras sus rondas de reparación se deshace, así el siguiente paso parte del último estado que pasaba.",
    "gui:execution-mode": "two_phase recopila contexto y luego responde con bloques de edición; interleaved ejecuta una conversación agéntica donde el modelo lee, edita, ejecuta pruebas y corrige hasta terminar; auto elige interleaved cuando corresponde.",
    "gui:live-telemetry": "Un pie bajo el espacio de trabajo de la GUI muestra la ventana de contexto, los tokens del turno y de la sesión, la velocidad de generación, los aciertos de caché KV y los pliegues mientras corre una tarea; la CLI imprime los mismos números en una línea.",
    "gui:cli-gui-cheatsheet": "La GUI y la CLI usan el mismo motor; usa la que te quede más cerca de la tarea.",
    "gui:named-mcp-servers": "Las tablas [mcp_servers.<nombre>] describen un servidor stdio (command + args) o uno Streamable HTTP (url). La fase de recopilación solo muestra al modelo las herramientas que coinciden con la tarea actual.",
    "cli:daemon": "Daemon sin interfaz y solo de loopback: un índice caliente y concesiones de modelo compartidas por sesiones de espacio de trabajo aisladas. Cada endpoint exige el token de portador de cada ejecución y nada queda expuesto fuera de 127.0.0.1.",
    "cli:config": "Muestra cada ajuste resuelto y de dónde viene; crea, prueba, imprime o añade opciones documentadas al archivo de configuración.",
    "cli:trust": "Aprueba el .pedacito.toml de este proyecto para que sus hooks, servidores MCP y comandos de verificación puedan ejecutarse.",
    "cli:how": "Pregunta cómo usar Pedacito. Busca en la referencia integrada de la CLI, la GUI y los ajustes; funciona sin modelo con --offline.",
    "cli:tui": "Interfaz de terminal dividida: panel de transcripción, barra lateral de actividad y plan, pie de telemetría y desplazamiento con PgUp/PgDn o la rueda del ratón.",
    "cli:index": "Construye o actualiza el índice del proyecto (mapa y resúmenes) que usan ask, explain y auto.",
    "cli:map": "Imprime el índice o mapa exactamente como lo recibiría el modelo.",
    "cli:files": "Muestra qué archivos se indexarían y por qué, incluidos los archivos ignorados y sus motivos.",
    "cli:ask": "Ejecuta una tarea de código de principio a fin. Sin --apply es una ejecución de prueba que imprime las ediciones propuestas.",
    "cli:auto": "Convierte un objetivo en una lista de tareas y luego implementa cada una en un contexto nuevo. Sin --apply solo escribe o imprime el plan.",
    "cli:plan": "Valida una lista de tareas contra el índice antes de la ejecución Auto o Complex: contiene cada ruta nombrada, resuelve los símbolos explícitos, comprueba el orden de dependencias y previsualiza llamadores y pruebas. Las advertencias no bloquean; los errores definitivos salen con 1 y un plan inválido con 2.",
    "cli:reset": "Borra el estado de Pedacito de este proyecto (índice, planes, cachés, sesiones y respaldos); el código y los archivos de configuración no se tocan. No es un restablecimiento de la configuración.",
    "cli:undo": "Deshace la última ejecución: restaura el espacio de trabajo al estado anterior al checkpoint activo (o a la instantánea de respaldo más reciente en proyectos antiguos).",
    "cli:recover": "Lista los diarios de tareas interrumpidas por su UUID y previsualiza o aplica una recuperación. Sin --apply no se escribe nada; resume y cleanup se rechazan cuando no se puede probar un límite duradero o una propiedad autoritativa.",
    "cli:gui": "Abre el panel de control local en el navegador (solo 127.0.0.1).",
    "cli:calibrate": "Mide el modelo configurado (elección de herramientas, formato de edición, honestidad, contexto largo y sondas de especialidad) y guarda una tarjeta de modelo que ajusta los valores por defecto.",
    "cli:rpc": "JSONL-RPC interactivo por stdin/stdout para editores y otros agentes: una petición JSON por línea; cada línea de stdout es un marco JSON y los diagnósticos humanos van a stderr.",
}


# Spanish step, note and title text for the highest-traffic permission, tools
# and daemon cards, keyed by the English text they replace.
_ES_CARD_TEXT = {
    "Approve or deny a Permission requested dialog":
        "Aprobar o denegar un diálogo de permiso solicitado",
    "Read the badge, the action, the reason and the exact target shown in the dialog":
        "Lee la insignia, la acción, el motivo y el destino exacto que muestra el diálogo",
    "Allow Once: this single request only":
        "Permitir una vez: solo esta solicitud",
    "Allow for This Run: no more prompts for this kind of request until the run ends":
        "Permitir durante esta ejecución: no vuelve a preguntar por este tipo de solicitud hasta que termine la ejecución",
    "Always Allow: never asks again for that request. It is saved to your config when one exists. Web access flips enable_web, shell commands flip allow_shell":
        "Permitir siempre: no vuelve a preguntar por esa solicitud. Se guarda en tu configuración si existe. El acceso web activa enable_web y los comandos de shell activan allow_shell",
    "Allow '<pattern>' in this project (GUI) or p in the TUI: a shell command matching a narrow rule skips the prompt in that project only. The rule is saved to shell_allowlist_by_project; package installs, shells and chained commands are never covered":
        "Permitir '<patrón>' en este proyecto (GUI) o p en la TUI: un comando de shell que coincida con una regla estrecha omite el aviso solo en ese proyecto. La regla se guarda en shell_allowlist_by_project; las instalaciones de paquetes, los shells y los comandos encadenados nunca quedan cubiertos",
    "Deny & Redirect: refuses the request. The model is told to continue with local project files only. Stop, or letting the dialog time out, also denies":
        "Denegar y redirigir: rechaza la solicitud y se le indica al modelo que continúe solo con los archivos locales del proyecto. Detener, o dejar que el diálogo expire, también deniega",
    "The CLI has no dialog: a gated tool returns its 'disabled in config' error unless the matching setting is on":
        "La CLI no tiene diálogo: una herramienta restringida devuelve su error 'disabled in config' salvo que el ajuste correspondiente esté activado",
    "Allow shell, write, browser or screenshot tools":
        "Permitir herramientas de shell, escritura, navegador o capturas de pantalla",
    "⚙ Settings → All settings → search for allow_shell / allow_write / allow_browser / allow_screenshots":
        "⚙ Ajustes → Todos los ajustes → busca allow_shell / allow_write / allow_browser / allow_screenshots",
    "Apply to: Global (every profile), choose on, then Save":
        "Aplicar a: Global (todos los perfiles), elige on y luego Guardar",
    "Review the safety companions too: shell_allowlist, shell_deny, sandbox, sandbox_network":
        "Revisa también los ajustes de seguridad relacionados: shell_allowlist, shell_deny, sandbox, sandbox_network",
    "In a project's own .pedacito.toml these toggles stay ignored until you read that file and run pedacito trust PROJ; your user config and --config FILE need no approval":
        "En el .pedacito.toml de un proyecto estos interruptores se ignoran hasta que leas ese archivo y ejecutes pedacito trust PROJ; tu configuración de usuario y --config FILE no necesitan aprobación",
    "discovery file: the per-user state folder's daemon.json (0o600 where the platform honours it), removed on stop when its pid matches":
        "archivo de descubrimiento: daemon.json en la carpeta de estado del usuario (0o600 donde la plataforma lo respete), se elimina al detener si coincide su pid",
    "HTTP/SSE control surface: /v1/health, /v1/sessions, /v1/sessions/<id>/events, /v1/quiesce and /v1/shutdown":
        "superficie de control HTTP/SSE: /v1/health, /v1/sessions, /v1/sessions/<id>/events, /v1/quiesce y /v1/shutdown",
    "every request needs X-Pedacito-Token or Authorization: Bearer":
        "cada solicitud necesita X-Pedacito-Token o Authorization: Bearer",
    "remote/WAN binding is refused by construction (loopback only)":
        "la vinculación remota o WAN se rechaza por diseño (solo bucle local)",
}

# Remaining English navigation words in Spanish card text (a fallback for the
# GUI paths and setting groups that have no exact entry above).
_ES_UI_TERMS = (
    ("⚙ Settings", "⚙ Ajustes"),
    ("All settings", "Todos los ajustes"),
    ("→ search ", "→ buscar "),
    ("Apply to:", "Aplicar a:"),
    ("Profiles", "Perfiles"),
    ("Unified policy engine", "Motor de políticas unificado"),
    ("Connection & Capabilities", "Conexión y capacidades"),
    ("Memory, skills, sub-agents, git", "Memoria, habilidades, subagentes y git"),
    ("Agent loop", "Bucle del agente"),
    ("Safety", "Seguridad"),
)


def es_card_text(text: str, lang: str = "en") -> str:
    """Spanish text for an English card string (exact match first, then the
    navigation terms); English text is returned unchanged."""
    if i18n.normalize_lang(lang) != "es" or not text:
        return text
    if text in _ES_CARD_TEXT:
        return _ES_CARD_TEXT[text]
    out = text
    for english, spanish in _ES_UI_TERMS:
        out = out.replace(english, spanish)
    return out


def _card_summary(item: dict, lang: str = "en") -> str:
    if i18n.normalize_lang(lang) == "es":
        return str(item.get("summary_es") or item.get("summary") or "")
    return str(item.get("summary") or "")


def setting_help(item: dict, lang: str = "en") -> str:
    if i18n.normalize_lang(lang) == "es":
        return str(item.get("help_es") or item.get("help") or "")
    return str(item.get("help") or "")


def localize_matches(matches: dict, lang: str = "es",
                     setting_help: dict[str, str] | None = None) -> None:
    """Attach Spanish card text to the (copied) items of a `search()` result.

    `setting_help` maps a setting key to its Spanish description from the
    Spanish configuration reference; English text stays as the fallback."""
    if i18n.normalize_lang(lang) != "es":
        return
    for item in matches.get("gui") or []:
        summary = _ES_SUMMARIES.get(f"gui:{item.get('id')}")
        if summary:
            item["summary_es"] = summary
    for item in matches.get("cli") or []:
        summary = _ES_SUMMARIES.get(f"cli:{item.get('name')}")
        if summary:
            item["summary_es"] = summary
    for item in matches.get("settings") or []:
        text = (setting_help or {}).get(item.get("key"))
        if text:
            item["help_es"] = text


def _flag_text(doc: dict, limit: int = 8, lang: str = "en") -> str:
    flags = [f['flag'] for f in doc.get("flags", [])]
    if not flags:
        return ""
    shown = ", ".join(f"`{f}`" for f in flags[:limit])
    if len(flags) > limit:
        shown += i18n.t("how.flags_more", lang=lang,
                        n=len(flags) - limit, name=doc["name"])
    return shown


def render_answer(question: str, matches: dict, lang: str = "en") -> str:
    """Format retrieved facts without a model. Always returns useful text."""
    lang = i18n.normalize_lang(lang)

    def t(key: str, **kw) -> str:
        return i18n.t(f"how.{key}", lang=lang, **kw)

    gui = matches.get("gui") or []
    cli = matches.get("cli") or []
    settings = matches.get("settings") or []
    lines: list[str] = []
    lines.append(t("banner"))
    lines.append("")
    best = (_card_summary(gui[0], lang) if gui else
            _card_summary(cli[0], lang) if cli else
            t("cant_match"))
    if best:
        lines.append(best)
        lines.append("")
    if gui:
        lines.append(t("gui_heading"))
        for item in gui[:2]:
            lines.append(f"1. **{es_card_text(item['title'], lang)}**")
            for step in item.get("steps", [])[:6]:
                lines.append(f"   - {es_card_text(step, lang)}")
            if item.get("notes"):
                for note in item["notes"][:3]:
                    lines.append(f"   - {es_card_text(note, lang)}")
        lines.append("")
    else:
        lines.append(t("gui_heading"))
        lines.append(t("gui_none"))
        lines.append("")
    lines.append(t("cli_heading"))
    if cli:
        for item in cli[:2]:
            lines.append(f"- **`pedacito {item['name']}`**: "
                         f"{_first_sentence(_card_summary(item, lang), 220)}")
            lines.append(f"  - {t('usage_label')}: `{item['usage']}`")
            examples = item.get("examples") or []
            if examples:
                lines.append("")
                lines.append("  ```sh")
                for example in examples[:3]:
                    lines.append(f"  {example}")
                lines.append("  ```")
            flags = _flag_text(item, lang=lang)
            if flags:
                lines.append(f"  - {t('flags_label')}: {flags}")
            if item.get("common"):
                lines.append(f"  - {t('common_flags')}")
            if item.get("notes"):
                for note in item["notes"][:4]:
                    lines.append(f"  - {es_card_text(note, lang)}")
            lines.append("")
    else:
        lines.append(t("cli_none"))
        lines.append("")
    lines.append(t("settings_heading"))
    if settings:
        for item in settings[:4]:
            default = item.get("default_text", "")
            lines.append(f"- **`{item['key']}`** ({t('default_label')}: "
                         f"`{default}`): "
                         f"{_first_sentence(setting_help(item, lang), 240)}")
            if item.get("gui"):
                lines.append(f"  - {t('where_label')}: {item['gui']}")
            if item.get("toml"):
                lines.append("")
                lines.append("    ```toml")
                for row in item["toml"].splitlines():
                    lines.append(f"    {row}")
                lines.append("    ```")
            lines.append("")
    else:
        lines.append(t("settings_none"))
    lines.append("")
    lines.append(t("footer"))
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
