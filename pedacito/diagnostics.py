"""Project diagnostics: run the linter/type checker the repository already uses.

    ok, report = diagnostics.run(root, cfg, files=["src/app.py"])
    ok, report = diagnostics.run_changed(root, cfg, changed_files)

    baseline = diagnostics.quick_findings(root, files)   # before the task
    ...apply edits...
    new = diagnostics.new_findings(baseline, diagnostics.quick_findings(root, files))

A thin layer over tools that are already installed: nothing is bundled, and a
project with no detected checker is a no-op, not a failure. Checks run through
the bounded process runner (`proc`). Linting only the changed files keeps the
feedback short for small models. `quick_findings` is the cheap per-file pass
the post-edit feedback loop uses (one linter run per changed file, or a syntax
parse when no linter is installed); `new_findings` diffs it against a baseline
so only problems the edits introduced are reported back.

Detection (first match per language):

    Python   ruff check --output-format concise [files|.]; pyflakes, flake8
    JS/TS    package.json "lint" script, else tsc --noEmit when tsconfig
             exists, else eslint on the given files
    Rust     cargo check --message-format short
    Go       go vet ./...
    Ruby     rubocop (only when a Gemfile/.rubocop.yml is present)
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import proc


@dataclass(frozen=True)
class Check:
    label: str
    argv: list[str]


def _has(root: Path, name: str) -> bool:
    return (root / name).exists()


def _which(name: str) -> str:
    return shutil.which(name) or ""


def _py_files(files: list[str] | None) -> list[str]:
    return [f for f in (files or []) if f.lower().endswith((".py", ".pyi"))]


def _js_files(files: list[str] | None) -> list[str]:
    return [f for f in (files or [])
            if f.lower().endswith((".js", ".jsx", ".mjs", ".cjs", ".ts",
                                   ".tsx"))]


def _package_scripts(root: Path) -> dict:
    try:
        data = json.loads((root / "package.json").read_text(encoding="utf-8"))
        return data.get("scripts") or {}
    except (OSError, ValueError):
        return {}


def detect(root, files: list[str] | None = None) -> list[Check]:
    """The checks applicable to this project (or these files)."""
    root = Path(root)
    checks: list[Check] = []
    files = list(files or [])

    py_targets = _py_files(files)
    python_project = bool(py_targets) or any(
        _has(root, n) for n in ("pyproject.toml", "setup.py", "setup.cfg",
                                "requirements.txt", "Pipfile"))
    if python_project:
        targets = py_targets or ["."]
        if _which("ruff"):
            checks.append(Check("ruff", ["ruff", "check", "--output-format",
                                         "concise", *targets]))
        elif _which("pyflakes"):
            checks.append(Check("pyflakes", ["pyflakes", *targets]))
        elif _which("flake8"):
            checks.append(Check("flake8", ["flake8", *targets]))

    if _has(root, "package.json"):
        scripts = _package_scripts(root)
        if scripts.get("lint") and _which("npm"):
            checks.append(Check("npm lint", ["npm", "run", "lint", "--silent"]))
        elif _has(root, "tsconfig.json") and _which("npx"):
            checks.append(Check("tsc", ["npx", "--no-install", "tsc",
                                        "--noEmit"]))
        elif _js_files(files) and _which("npx"):
            checks.append(Check("eslint", ["npx", "--no-install", "eslint",
                                           *_js_files(files)]))

    if _has(root, "Cargo.toml") and _which("cargo"):
        checks.append(Check("cargo check", ["cargo", "check",
                                            "--message-format", "short"]))
    if _has(root, "go.mod") and _which("go"):
        checks.append(Check("go vet", ["go", "vet", "./..."]))
    if (_has(root, "Gemfile") or _has(root, ".rubocop.yml")) \
            and _which("rubocop"):
        checks.append(Check("rubocop", ["rubocop", "--format", "simple"]))
    return checks


def run(root, cfg=None, files: list[str] | None = None) -> tuple[bool, str]:
    """Run every detected check. Returns (passed, report). No detected
    checker is a pass: absence of a linter must not fail a task."""
    root = Path(root)
    checks = detect(root, files)
    if not checks:
        return True, ("No linter/type checker detected (looked for ruff, "
                      "pyflakes, flake8, npm lint, tsc, cargo, go vet, "
                      "rubocop).")
    timeout = int(getattr(cfg, "diagnostics_timeout", 90) or 90)
    cap = int(getattr(cfg, "max_diagnostics_chars", 8000) or 8000)
    ok = True
    chunks: list[str] = []
    for check in checks:
        rc, out, stopped, secs = proc.run(
            check.argv, cwd=root, timeout=timeout, idle_timeout=0,
            max_output=cap)
        bad = stopped in ("timeout", "error") or (rc not in (0, None))
        if bad:
            ok = False
        status = "PASS" if not bad else (
            "TIMEOUT" if stopped == "timeout" else "FAIL")
        chunks.append(f"CHECK {check.label}: {status} ({secs:.1f}s)")
        body = (out or "").strip()
        if body:
            chunks.append(body)
        if bad and stopped == "timeout":
            chunks.append(f"(stopped after {timeout}s)")
    report = "\n".join(chunks).strip()
    if len(report) > cap:
        report = report[:cap].rstrip() + "\n[... diagnostics truncated]"
    return ok, report


def run_changed(root, cfg, changed: list[str] | None = None) -> tuple[bool, str]:
    """Convenience wrapper for the automatic post-edit check."""
    return run(root, cfg, files=list(changed or []))


# ---------------------------------------------------------------------------
# Quick per-file checks + baseline diff (the post-edit feedback loop)
# ---------------------------------------------------------------------------
# Per-file timeout: this runs inline after every applied edit, so it must stay
# far below the project-wide diagnostics_timeout.
_QUICK_TIMEOUT = 30

# "path:12:3:" / "path:12:" prefixes and bare "line 12" mentions: positions
# shift when lines are inserted above, so the baseline diff ignores them.
_POS_PREFIX_RE = re.compile(r"^\S+?:\d+(?::\d+)?:\s*")
_LINE_WORD_RE = re.compile(r"\bline \d+\b")


def _norm_finding(line: str) -> str:
    """A finding with its positions erased, for baseline comparison."""
    s = _POS_PREFIX_RE.sub("", line.strip())
    s = _LINE_WORD_RE.sub("line N", s)
    return s.strip() or line.strip()


def _quick_python(root: Path, rel: str) -> list[str]:
    """One Python file: the first installed linter run on just that file, else
    a plain ast.parse. Finding lines keep their `rel:line:col:` prefix."""
    for argv in (("ruff", "check", "--output-format", "concise"),
                 ("pyflakes",),
                 ("flake8",)):
        if not _which(argv[0]):
            continue
        _rc, out, stopped, _secs = proc.run(
            [*argv, rel], cwd=root, timeout=_QUICK_TIMEOUT,
            idle_timeout=0, max_output=20000)
        if stopped in ("timeout", "error"):
            return []          # an unusable linter must not block edits
        return [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
    import ast
    try:
        src = (root / rel).read_text(encoding="utf-8", errors="replace")
        ast.parse(src)
    except SyntaxError as e:
        return [f"{rel}:{e.lineno or 0}: syntax error: {e.msg} (line {e.lineno})"]
    except (OSError, ValueError):
        return []
    return []


def _quick_js(root: Path, rel: str) -> list[str]:
    """One JS/TS file: the tree-sitter syntax check used by edits._validate."""
    from .chunker_ts import LANG_BY_EXT, TS_AVAILABLE, has_syntax_error
    lang = LANG_BY_EXT.get(Path(rel).suffix.lower())
    if not lang or not TS_AVAILABLE:
        return []
    try:
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    bad, lineno = has_syntax_error(text, lang)
    if bad:
        return [f"{rel}:{lineno or 0}: syntax error (parse failed, "
                f"line {lineno})"]
    return []


def quick_findings(root, files: list[str]) -> dict[str, list[str]]:
    """Cheap per-file syntax/lint findings as {rel: [finding lines]}.

    Python files go to the first installed linter (ruff, pyflakes, flake8)
    scoped to that one file, falling back to ast.parse; JS/TS files get the
    tree-sitter syntax check. Other languages have no cheap per-file checker
    and are skipped. Never raises; an unreadable file maps to no findings."""
    root = Path(root)
    out: dict[str, list[str]] = {}
    for rel in files or []:
        rel = str(rel).strip()
        if not rel:
            continue
        suffix = Path(rel).suffix.lower()
        try:
            if suffix in (".py", ".pyi"):
                out[rel] = _quick_python(root, rel)
            elif suffix in (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"):
                out[rel] = _quick_js(root, rel)
        except Exception:
            out[rel] = []
    return out


def new_findings(baseline: dict[str, list[str]],
                 current: dict[str, list[str]]) -> dict[str, list[str]]:
    """Findings in `current` whose normalised text is absent from `baseline`
    (per file): only problems the edits introduced, never pre-existing ones
    whose line numbers shifted."""
    out: dict[str, list[str]] = {}
    for rel, items in current.items():
        old = {_norm_finding(x) for x in baseline.get(rel, [])}
        extra = [x for x in items if _norm_finding(x) not in old]
        if extra:
            out[rel] = extra
    return out


def render_findings(found: dict[str, list[str]], cap: int = 4000) -> str:
    """The finding lines as feedback text, bounded."""
    lines = [it for items in found.values() for it in items]
    text = "\n".join(lines)
    if len(text) > cap:
        text = text[:cap].rstrip() + "\n[... more findings truncated]"
    return text
