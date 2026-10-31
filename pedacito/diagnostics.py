"""Project diagnostics: run the linter/type checker the repository already uses.

Usage
-----
    from . import diagnostics

    ok, report = diagnostics.run(root, cfg, files=["src/app.py"])
    ok, report = diagnostics.run_changed(root, cfg, changed_files)

This is deliberately a thin, deterministic layer over tools that are already
installed. Pedacito does not bundle a linter and does not guess a project's
style; it detects the common ones and runs them through the same bounded
process runner as `run`. A project with no detected checker is a no-op, not a
failure.

Detection (first match per language, cheapest/most focused first):

    Python   ruff check --output-format concise [files|.]
             pyflakes / flake8 as fallbacks
    JS/TS    package.json "lint" script, else tsc --noEmit when tsconfig exists
    Rust     cargo check --message-format short
    Go       go vet ./...
    Ruby     rubocop (only when a Gemfile/.rubocop.yml is present)

File targeting matters for small models: linting only the changed file keeps
the feedback short and avoids burying the actual error in pre-existing noise.
"""

from __future__ import annotations

import json
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
    """Return the checks applicable to this project (or these files)."""
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
            checks.append(Check("flake8", *targets))

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


def available(root) -> list[str]:
    """Labels of the checks that would run, without executing anything."""
    return [c.label for c in detect(root)]


def run(root, cfg=None, files: list[str] | None = None) -> tuple[bool, str]:
    """Run every detected check. Returns (passed, report).

    No detected checker is a pass with an explanatory report: absence of a
    linter must not fail a task.
    """
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
