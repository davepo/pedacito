"""Test discovery and test-impact analysis.

    cmd = detect_test_command(root, cfg)          # 'python -m pytest -q'
    tests = impacted_tests(index, ["src/db.py"])  # tests worth running first
    ok, out, how = run_impacted(index, cfg, ["src/db.py"])

`detect_test_command` finds the project's runner from its manifests, without
running anything. `impacted_tests` ranks the test files a change can break:
the tests named after the file, tests of the files that import or call it, and
test files that import it directly. `run_impacted` runs just those first (a
narrow `pytest FILE...` / `npm test -- FILE...` / `go test ./dir/...`), so a
weak model gets a verdict in seconds instead of minutes.
"""

from __future__ import annotations

import json
import posixpath
import re
import shlex
import shutil
import sys
from pathlib import Path

from . import proc, safety, sandbox, testlinks

_NPM_PLACEHOLDER = re.compile(r"no test specified", re.IGNORECASE)


def _has(root: Path, *names: str) -> bool:
    return any((root / n).exists() for n in names)


def detect_test_command(root, cfg=None) -> str:
    """The project's test command, or '' when none is evident. A configured
    `test_command` always wins."""
    explicit = (getattr(cfg, "test_command", "") or "").strip()
    if explicit:
        return explicit
    root = Path(root)
    pkg = root / "package.json"
    if pkg.is_file():
        try:
            scripts = (json.loads(pkg.read_text(encoding="utf-8"))
                       .get("scripts") or {})
        except (OSError, ValueError):
            scripts = {}
        test = str(scripts.get("test") or "")
        if test and not _NPM_PLACEHOLDER.search(test):
            if _has(root, "pnpm-lock.yaml"):
                return "pnpm test"
            if _has(root, "yarn.lock"):
                return "yarn test"
            return "npm test"
    if _has(root, "Cargo.toml"):
        return "cargo test"
    if _has(root, "go.mod"):
        return "go test ./..."
    py_markers = _has(root, "pytest.ini", "tox.ini", "conftest.py", "pyproject.toml",
                      "setup.cfg", "setup.py")
    has_py_tests = any(root.glob("tests/test_*.py")) or any(root.glob("test_*.py")) \
        or any(root.glob("tests/**/test_*.py"))
    if has_py_tests or (py_markers and (root / "tests").is_dir()):
        return "python -m pytest -q" if not shutil.which("pytest") else "pytest -q"
    if _has(root, "Makefile", "makefile", "GNUmakefile"):
        for name in ("Makefile", "makefile", "GNUmakefile"):
            mk = root / name
            if mk.is_file():
                try:
                    if re.search(r"^test\s*:", mk.read_text(encoding="utf-8",
                                 errors="replace"), re.MULTILINE):
                        return "make test"
                except OSError:
                    pass
    if _has(root, "noxfile.py"):
        return "nox"
    if _has(root, "tox.ini"):
        return "tox"
    if _has(root, "Gemfile") and (root / "spec").is_dir():
        return "bundle exec rspec"
    if _has(root, "pom.xml"):
        return "mvn -q test"
    if _has(root, "build.gradle", "build.gradle.kts"):
        return "gradle test"
    if any(root.glob("*.sln")) or any(root.glob("*.csproj")):
        return "dotnet test"
    return ""


def impacted_tests(index, changed: list[str], limit: int = 12) -> list[tuple[str, str]]:
    """[(test file, why)] most relevant first for a set of changed files."""
    g = index.graph
    _, tests_map = testlinks.build_maps(index.files)
    all_tests = {f for f in index.files if testlinks.classify(f)[0]}
    out: dict[str, str] = {}

    def add(t: str, why: str) -> None:
        if t in all_tests and t not in out:
            out[t] = why

    for f in changed:
        if f in all_tests:
            add(f, "changed itself")
    for f in changed:
        for t in testlinks.tests_of(f, tests_map):
            add(t, f"named after {posixpath.basename(f)}")
    for f in changed:
        for t in sorted(g.importers.get(f, ())):
            add(t, f"imports {posixpath.basename(f)}")
        for t in g.callers_files(f):
            add(t, f"calls into {posixpath.basename(f)}")
    for f in changed:
        for dep in sorted(g.dependents(f)):
            for t in testlinks.tests_of(dep, tests_map):
                add(t, f"tests {posixpath.basename(dep)}, which uses "
                       f"{posixpath.basename(f)}")
    return list(out.items())[:limit]


def narrow_command(command: str, tests: list[str]) -> tuple[str, bool]:
    """(command, narrowed) -- `command` restricted to `tests` when the runner
    supports it; unchanged otherwise."""
    if not tests:
        return command, False
    low = command.lower()
    files = " ".join(shlex.quote(t) for t in tests)
    if "pytest" in low:
        return f"{command} {files}", True
    if low.startswith(("npm test", "pnpm test", "yarn test")):
        sep = "" if low.startswith("yarn") else " --"
        return f"{command}{sep} {files}", True
    if low.startswith("go test"):
        dirs = sorted({"./" + posixpath.dirname(t) if posixpath.dirname(t)
                       else "." for t in tests if t.endswith(".go")})
        if dirs:
            return "go test " + " ".join(d + "/..." if d != "." else "." for d in dirs), True
    if low.startswith("bundle exec rspec"):
        return f"{command} {files}", True
    return command, False


def _argv(command: str) -> list[str]:
    try:
        return shlex.split(command) if sys.platform != "win32" else \
            [p.strip("\"'") for p in shlex.split(command, posix=False)]
    except ValueError:
        return command.split()


def run_command(root, cfg, command: str, timeout: int | None = None
                ) -> tuple[bool, str]:
    """Run a test command as an argv (no shell) with the scrubbed environment;
    (passed, output)."""
    argv = _argv(command)
    if not argv:
        return False, "(empty test command)"
    timeout = int(timeout or getattr(cfg, "test_timeout", 300) or 300)
    try:
        argv, box = sandbox.wrap(argv, cfg, root)
    except sandbox.SandboxRefused as e:
        return False, f"could not run `{command}`: {e}"
    rc, out, stopped, _el = proc.run(
        argv, cwd=Path(root), timeout=timeout, idle_timeout=0,
        max_output=int(getattr(cfg, "max_shell_output_chars", 8000) or 8000) * 2,
        env=safety.child_env(cfg))
    if box and rc not in (0, None) and not stopped:
        out += (f"\n[ran in a sandbox ({box}); a failure from a blocked write "
                "outside the project or no network is the sandbox, not the "
                "code]")
    if stopped == "error":
        return False, f"could not run `{command}`: {out}"
    if stopped:
        return False, f"`{command}` was stopped ({stopped}) after {timeout}s\n{out}"
    return rc == 0, out


def run_impacted(index, cfg, changed: list[str]) -> tuple[bool, str, str]:
    """Run the tests a change can break: impacted ones first, then (if they
    pass and the command could be narrowed) nothing more -- the full suite is
    the user's `--verify`. Returns (passed, output, description)."""
    base = detect_test_command(index.root, cfg)
    if not base:
        return True, "", "no test command detected"
    picked = impacted_tests(index, changed)
    files = [t for t, _why in picked]
    cmd, narrowed = narrow_command(base, files)
    why = ", ".join(f"{t} ({w})" for t, w in picked[:4])
    desc = (f"`{cmd}` -- impacted tests: {why}" if narrowed
            else f"`{cmd}` (full suite; could not narrow)")
    ok, out = run_command(index.root, cfg, cmd)
    return ok, out, desc
