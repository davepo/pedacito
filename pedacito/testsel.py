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

# Appended to a failing run's description when the command's recorded history
# shows it already failed before this run (baseline-aware verdicts, R2). The
# repair loop reads this marker to avoid chasing pre-existing breakage.
PRE_EXISTING_LABEL = "pre-existing failure (recorded before this run)"


def _has(root: Path, *names: str) -> bool:
    return any((root / n).exists() for n in names)


def _has_js_tests(root: Path) -> bool:
    """True when the tree has Node-style test files (`*.test.js`/`.mjs` or a
    `test*/` dir of them) that `node --test` can run."""
    for pattern in ("*.test.js", "*.test.mjs", "tests/*.test.js",
                    "test/*.test.js", "tests/*.test.mjs", "test/*.test.mjs",
                    "src/**/*.test.js", "tests/**/*.test.js",
                    "test/**/*.test.js"):
        if any(root.glob(pattern)):
            return True
    return False


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
    if _has_js_tests(root):
        # Zero-dependency Node projects (no package.json test script, or no
        # package.json at all) still have a runner: `node --test` discovers
        # *.test.js recursively since Node 18/20. Both game sessions had to
        # fabricate a package.json purely so detection would fire.
        return "node --test"
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
    # A test file CREATED by the apply that triggered this check is not in the
    # index yet (it is reindexed after the run). Classify the changed paths
    # too, so a brand-new test file still runs in the post-edit hook.
    all_tests |= {f for f in changed if testlinks.classify(f)[0]}
    out: dict[str, str] = {}

    def add(t: str, why: str) -> None:
        if t in all_tests and t not in out:
            # A deleted test file still rides in `changed` (and the stale
            # index); narrowing to it runs zero tests and exits 0 -- a false
            # green (BUG-222). Select only files that exist on disk.
            if not (Path(index.root) / t).is_file():
                return
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
    # Transitive dependents (COVERAGE-1): a change deep in a package re-export
    # chain (reporting.py <- commands/__init__.py <- cli.py <- test_cli.py) is
    # only two hops from the test that exercises it through the public API, so
    # a one-hop walk misses it and the hook runs a suite that cannot see the
    # regression. Walk the impact graph to depth 2 and add those tests LAST:
    # `add` keeps the first reason and `limit` truncates the tail, so direct
    # links always win the slots and this only widens when there is room.
    for f in changed:
        impact = g.impact(f, depth=2)
        for dep in sorted(impact, key=lambda d: (impact[d], d)):
            if dep in all_tests:
                add(dep, f"transitively imports {posixpath.basename(f)}")
            for t in testlinks.tests_of(dep, tests_map):
                add(t, f"tests {posixpath.basename(dep)}, which transitively "
                       f"uses {posixpath.basename(f)}")
    return list(out.items())[:limit]


def narrow_command(command: str, tests: list[str],
                   changed: list[str] | None = None) -> tuple[str, bool]:
    """(command, narrowed) -- `command` restricted to `tests` when the runner
    supports it; unchanged otherwise. `changed` (the files the task actually
    touched) is what makes cargo narrowing safe; see the cargo branch."""
    if not tests:
        return command, False
    low = command.lower()
    files = " ".join(shlex.quote(t) for t in tests)
    if "pytest" in low:
        return f"{command} {files}", True
    if low.startswith("node --test") or low == "node -t":
        # `node --test FILE...` runs exactly those files; non-JS entries in
        # `tests` (a mixed tree) would make the runner error, so filter.
        js = [t for t in tests if t.lower().endswith((".js", ".mjs", ".cjs"))]
        if js:
            return f"{command} " + " ".join(shlex.quote(t) for t in js), True
        return command, False
    if low.startswith(("npm test", "pnpm test", "yarn test")):
        sep = "" if low.startswith("yarn") else " --"
        return f"{command}{sep} {files}", True
    if low.startswith("go test"):
        dirs = sorted({"./" + posixpath.dirname(t) if posixpath.dirname(t)
                       else "." for t in tests if t.endswith(".go")})
        if dirs:
            return "go test " + " ".join(d + "/..." if d != "." else "." for d in dirs), True
    if low.startswith("cargo test"):
        # #48: `cargo test --test NAME` runs ONLY that integration target --
        # the src unit tests and every other target are skipped. Narrowing is
        # sound only when the change is confined to top-level integration
        # tests (tests/*.rs): each is its own crate, so no other target can
        # be affected and the changed targets ARE the whole change. Unknown
        # `changed` (None), any src/manifest/helper change, or a command that
        # already selects targets keeps the full suite.
        if not changed:
            return command, False
        if re.search(r"--(?:test|lib|bins?|doc|examples?|bench(?:es)?|all-targets)"
                     r"(?:[\s=]|$)", command):
            return command, False
        stems = set()
        for f in changed:
            f = f.replace("\\", "/")
            if f.endswith(".rs") and posixpath.dirname(f) == "tests":
                stems.add(posixpath.basename(f)[:-3])
            else:
                return command, False
        return (command + " "
                + " ".join(f"--test {shlex.quote(s)}" for s in sorted(stems)),
                True)
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
    try:
        from . import memory
        memory.observe_command(root, command, rc == 0, cfg, kind="test")
    except Exception:
        pass
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
    cmd, narrowed = narrow_command(base, files, changed=changed)
    why = ", ".join(f"{t} ({w})" for t, w in picked[:4])
    desc = (f"`{cmd}` -- impacted tests: {why}" if narrowed
            else f"`{cmd}` (full suite; could not narrow)")
    # Consult the recorded history BEFORE running: run_command updates the
    # ledger with this run's outcome, which would otherwise erase the very
    # baseline we are trying to compare against.
    was_failing = _was_already_failing(index.root, cmd)
    if was_failing and any(f in files for f in changed):
        # The narrowed command names a test file THIS run rewrote: the
        # recorded baseline describes the old contents, so its failures
        # cannot be assumed pre-existing (BUG-206). Keep the verdict
        # edit-caused so repair sees the failures this run created.
        was_failing = False
    ok, out = run_command(index.root, cfg, cmd)
    if not ok and was_failing:
        desc += f" -- {PRE_EXISTING_LABEL}"
    return ok, out, desc


def _was_already_failing(root, cmd: str) -> bool:
    """The command's recorded history said it failed before this run. A
    missing ledger or an unreadable root must never mask a real failure, so
    any error answers False (the verdict stays edit-caused)."""
    try:
        from . import memory
        return memory.command_was_failing(root, cmd)
    except Exception:
        return False
