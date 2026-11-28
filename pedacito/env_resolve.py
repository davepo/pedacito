"""Offline project environment resolution.

    from . import env_resolve

    resolved = env_resolve.resolve_environment(root)
    env = env_resolve.command_env(resolved, cfg)
    # run pytest / a linter / a language server with `env`

`resolve_environment` is **detect-only**: it reads project metadata
(``.venv``, ``uv.lock``, ``poetry.lock``, ``Pipfile``, ``conda-meta``,
``tox.ini``/``.tox``, ``noxfile.py``/``.nox``, pnpm/yarn/bun lockfiles,
``Cargo.toml``, ``go.mod``) and checks whether a local interpreter exists.
It never runs an activation script, ``uv``/``poetry``/``conda``/``tox``/
``nox``, ``npx``, an install, or a network request.  The chosen interpreter
is only a ``Path`` until a caller decides to execute it.

Precedence is deterministic:

1. an existing project interpreter (``.venv``, ``venv``, a project conda
   environment, then a ``.tox``/``.nox`` environment);
2. an explicitly active environment (``VIRTUAL_ENV`` / ``CONDA_PREFIX``);
3. package-manager metadata, ranked by a fixed evidence table;
4. ``system`` -- the ambient ``PATH`` is used unchanged.

When more than one plausible environment exists the winner is still chosen by
that table, and ``ResolvedEnvironment.evidence`` records the ambiguity (the
string starts with ``ambiguous:``) instead of silently picking one.

Every project path goes through :func:`pedacito.security.resolve_safe_path`,
so a ``.venv`` symlink/junction that points outside the project is rejected
and never becomes ``ResolvedEnvironment.python``.  An environment named by
the caller's own ``VIRTUAL_ENV``/``CONDA_PREFIX`` is explicit user state and
may live outside the project.

``command_env`` composes :func:`pedacito.safety.child_env` (the existing
scrub allow-list) and only prepends the resolved bin directories to ``PATH``;
it never copies raw ambient variables or re-adds a secret somebody scrubbed.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from . import safety, security

__all__ = [
    "ResolvedEnvironment",
    "command_env",
    "dependency_roots",
    "find_executable",
    "resolve_environment",
    "site_packages_dirs",
]

# A bound on metadata reads: manifests are small, but a generated file or a
# hostile checkout must not turn detection into a memory sink.
_MAX_METADATA_BYTES = 262_144
_MAX_EVIDENCE = 24
_MAX_EVIDENCE_CHARS = 240

# Fixed candidate order for score ties. Never iterate a set here.
_ORDER = {
    ".venv": 0,
    "venv": 1,
    "conda-project": 2,
    "tox": 3,
    "nox": 4,
    "active-venv": 5,
    "active-conda": 6,
    "uv": 7,
    "poetry": 8,
    "pipenv": 9,
    "conda": 10,
    "pnpm": 11,
    "bun": 12,
    "yarn": 13,
    "node": 14,
    "cargo": 15,
    "go": 16,
    "system": 17,
}

_PY_MANAGERS = frozenset({"uv", "poetry", "pipenv", "conda"})
_NODE_MANAGERS = frozenset({"pnpm", "yarn", "bun", "node"})


@dataclass(frozen=True)
class ResolvedEnvironment:
    """One detected project environment. All paths are real :class:`Path`s.

    ``kind`` and ``package_manager`` are ASCII protocol values (localized only
    when rendered). ``python`` is ``None`` when no project-local interpreter
    was found, which means "let ``PATH`` decide", not "there is no Python".
    """

    root: Path
    kind: str
    python: Path | None
    bin_dirs: tuple[Path, ...]
    package_manager: str
    workspace_root: Path
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class _Candidate:
    kind: str
    score: int
    python: Path | None
    bin_dirs: tuple[Path, ...]
    package_manager: str
    evidence: tuple[str, ...]
    order: int


# ---------------------------------------------------------------------------
# Path and executable helpers
# ---------------------------------------------------------------------------

def _bin_name() -> str:
    return "Scripts" if os.name == "nt" else "bin"


def _is_executable(path: Path) -> bool:
    """A regular file that the platform can launch: POSIX needs an x bit;
    Windows treats any existing file with a known extension as executable."""
    try:
        if not path.is_file():
            return False
        if os.name == "nt":
            return True
        return bool(path.stat().st_mode & 0o111)
    except OSError:
        return False


def _find_python(directory: Path) -> Path | None:
    """The first existing interpreter in a bin/Scripts directory."""
    names = ("python.exe", "python3.exe") if os.name == "nt" \
        else ("python", "python3")
    for name in names:
        candidate = directory / name
        if _is_executable(candidate):
            return candidate
    return None


def _safe_project_path(root: Path, rel: str) -> Path | None:
    """`rel` resolved inside `root`, or None for an escape/unreadable path."""
    try:
        return security.resolve_safe_path(root, rel)
    except (security.SecurityError, OSError, ValueError):
        return None


def _contained(root: Path, path: Path) -> Path | None:
    """`path` if its resolved form stays under `root`, else None."""
    try:
        resolved = path.resolve()
    except OSError:
        return None
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def _file(root: Path, rel: str) -> Path | None:
    path = _safe_project_path(root, rel)
    return path if path is not None and path.is_file() else None


def _directory(root: Path, rel: str) -> Path | None:
    path = _safe_project_path(root, rel)
    return path if path is not None and path.is_dir() else None


def _read_small(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        with path.open("rb") as handle:
            data = handle.read(_MAX_METADATA_BYTES)
    except OSError:
        return ""
    return data.decode("utf-8", "replace")


def _rel_text(root: Path, path: Path) -> str:
    """Project-relative POSIX text for evidence; absolute only if outside."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _clean_evidence(items) -> tuple[str, ...]:
    out: list[str] = []
    for item in items:
        text = " ".join(str(item or "").split())
        if not text:
            continue
        text = text[:_MAX_EVIDENCE_CHARS]
        if text not in out:
            out.append(text)
        if len(out) >= _MAX_EVIDENCE:
            break
    return tuple(out)


def _same_path(a: Path | None, b: Path | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return os.path.normcase(str(a.resolve())) == os.path.normcase(str(b.resolve()))
    except OSError:
        return os.path.normcase(str(a)) == os.path.normcase(str(b))


# ---------------------------------------------------------------------------
# Metadata detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Facts:
    root: Path
    env: dict[str, str]
    pyproject: str
    package_json: dict
    uv: tuple[str, ...]
    poetry: tuple[str, ...]
    pipenv: tuple[str, ...]
    conda: tuple[str, ...]
    pnpm: tuple[str, ...]
    yarn: tuple[str, ...]
    bun: tuple[str, ...]
    npm: tuple[str, ...]
    cargo: tuple[str, ...]
    go: tuple[str, ...]


def _package_manager_field(data: dict) -> str:
    raw = data.get("packageManager") if isinstance(data, dict) else None
    if not isinstance(raw, str):
        return ""
    return raw.split("@", 1)[0].strip().lower()


def _facts(root: Path, env: dict[str, str]) -> _Facts:
    pyproject_path = _file(root, "pyproject.toml")
    pyproject = _read_small(pyproject_path)
    pkg_path = _file(root, "package.json")
    package_json: dict = {}
    pkg_raw = _read_small(pkg_path)
    if pkg_raw:
        try:
            loaded = json.loads(pkg_raw)
            if isinstance(loaded, dict):
                package_json = loaded
        except ValueError:
            package_json = {}
    declared = _package_manager_field(package_json)

    uv = []
    if _file(root, "uv.lock"):
        uv.append("uv.lock")
    if _file(root, "uv.toml"):
        uv.append("uv.toml")
    if re.search(r"(?m)^\s*\[tool\.uv(?:\.|\])", pyproject):
        uv.append("pyproject.toml [tool.uv]")
    if not uv and _file(root, ".python-version") and declared == "uv":
        uv.append(".python-version")

    poetry = []
    if _file(root, "poetry.lock"):
        poetry.append("poetry.lock")
    if _file(root, "poetry.toml"):
        poetry.append("poetry.toml")
    if re.search(r"(?m)^\s*\[tool\.poetry(?:\.|\])", pyproject):
        poetry.append("pyproject.toml [tool.poetry]")
    if declared == "poetry":
        poetry.append("package.json packageManager")

    pipenv = []
    if _file(root, "Pipfile"):
        pipenv.append("Pipfile")
    if _file(root, "Pipfile.lock"):
        pipenv.append("Pipfile.lock")

    conda = []
    for name in ("environment.yml", "environment.yaml"):
        if _file(root, name):
            conda.append(name)
    if _directory(root, "conda-meta") is not None:
        conda.append("conda-meta/")

    pnpm = []
    for name in ("pnpm-lock.yaml", "pnpm-workspace.yaml"):
        if _file(root, name):
            pnpm.append(name)
    if declared == "pnpm":
        pnpm.append("package.json packageManager")

    yarn = []
    if _file(root, "yarn.lock"):
        yarn.append("yarn.lock")
    if declared == "yarn":
        yarn.append("package.json packageManager")

    bun = []
    for name in ("bun.lockb", "bun.lock", "bunfig.toml"):
        if _file(root, name):
            bun.append(name)
    if declared == "bun":
        bun.append("package.json packageManager")

    npm = []
    if _file(root, "package-lock.json"):
        npm.append("package-lock.json")
    if _file(root, "npm-shrinkwrap.json"):
        npm.append("npm-shrinkwrap.json")
    if package_json and not (pnpm or yarn or bun):
        npm.append("package.json")

    cargo = []
    if _file(root, "Cargo.toml"):
        cargo.append("Cargo.toml")
    if _file(root, "Cargo.lock"):
        cargo.append("Cargo.lock")

    go = []
    if _file(root, "go.mod"):
        go.append("go.mod")
    if _file(root, "go.work"):
        go.append("go.work")

    return _Facts(
        root=root,
        env=env,
        pyproject=pyproject,
        package_json=package_json,
        uv=tuple(uv),
        poetry=tuple(poetry),
        pipenv=tuple(pipenv),
        conda=tuple(conda),
        pnpm=tuple(pnpm),
        yarn=tuple(yarn),
        bun=tuple(bun),
        npm=tuple(npm),
        cargo=tuple(cargo),
        go=tuple(go),
    )


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------

def _venv_hint(facts: _Facts) -> tuple[str, str, tuple[str, ...]]:
    """(kind, package_manager, evidence) for a project virtualenv."""
    for kind, files in (("uv", facts.uv), ("poetry", facts.poetry),
                        ("pipenv", facts.pipenv)):
        if files:
            return kind, kind, tuple(f"manager metadata: {f}" for f in files)
    return "venv", "pip", ()


def _node_bin(root: Path) -> tuple[Path, ...]:
    bin_dir = _directory(root, "node_modules/.bin")
    return (bin_dir,) if bin_dir is not None else ()


def _candidate_venv(facts: _Facts, name: str) -> _Candidate | None:
    """A `.venv`/`venv` directory and its interpreter, if one exists."""
    directory = _directory(facts.root, name)
    if directory is None:
        return None
    raw_bin = directory / _bin_name()
    bin_dir = (_contained(facts.root, raw_bin)
               if raw_bin.is_dir() else None)
    python = _find_python(bin_dir) if bin_dir is not None else None
    kind, manager, hint = _venv_hint(facts)
    score = (100 if python is not None else 30) if name == ".venv" \
        else (95 if python is not None else 25)
    evidence = [f"{name}/ exists"]
    if python is not None:
        evidence.append(f"existing interpreter: {_rel_text(facts.root, python)}")
    else:
        evidence.append(f"{name}/ has no {_bin_name()}/python interpreter")
    evidence.extend(hint)
    return _Candidate(
        kind=kind,
        score=score,
        python=python,
        bin_dirs=(bin_dir,) if bin_dir is not None else (),
        package_manager=manager,
        evidence=_clean_evidence(evidence),
        order=_ORDER[name],
    )


def _conda_python(root: Path) -> Path | None:
    if os.name == "nt":
        for directory in (root, root / "Scripts", root / "Library" / "bin"):
            python = _find_python(directory)
            if python is not None:
                return python
        return None
    return _find_python(root / "bin")


def _conda_bins(root: Path) -> tuple[Path, ...]:
    names = ("Scripts", "Library/bin", "bin") if os.name == "nt" else ("bin",)
    out: list[Path] = []
    for name in names:
        raw = root / name
        if not raw.is_dir():
            continue
        safe = _contained(root, raw)
        if safe is not None:
            out.append(safe)
    return tuple(out)


def _candidate_conda_project(facts: _Facts) -> _Candidate | None:
    meta = _directory(facts.root, "conda-meta")
    if meta is None:
        return None
    python = _conda_python(facts.root)
    if python is not None:
        python = _contained(facts.root, python)
    evidence = ["conda-meta/ exists"]
    if python is not None:
        evidence.append(f"existing interpreter: {_rel_text(facts.root, python)}")
    return _Candidate(
        kind="conda",
        score=98 if python is not None else 32,
        python=python,
        bin_dirs=_conda_bins(facts.root),
        package_manager="conda",
        evidence=_clean_evidence(evidence),
        order=_ORDER["conda-project"],
    )


def _env_dir_python(root: Path, directory: Path
                    ) -> tuple[Path | None, Path | None]:
    """(python, bin_dir) for one tox/nox environment directory."""
    raw_bin = directory / _bin_name()
    if not raw_bin.is_dir():
        return None, None
    bin_dir = _contained(root, raw_bin)
    if bin_dir is None:
        return None, None
    return _find_python(bin_dir), bin_dir


def _candidate_env_manager(facts: _Facts, name: str, kind: str,
                           score: int) -> _Candidate | None:
    """A fixed-rank `.tox`/`.nox` environment or its manifest."""
    base = _directory(facts.root, name)
    found: list[tuple[Path, Path, Path]] = []
    if base is not None:
        try:
            entries = sorted((p for p in base.iterdir() if p.is_dir()),
                             key=lambda p: p.name)
        except OSError:
            entries = []
        for entry in entries:
            safe = _contained(facts.root, entry)
            if safe is None:
                continue
            python, bin_dir = _env_dir_python(facts.root, safe)
            if python is not None and bin_dir is not None:
                found.append((safe, python, bin_dir))
    manifest = _file(facts.root, "tox.ini" if kind == "tox" else "noxfile.py")
    if not found and manifest is None:
        return None
    evidence: list[str] = []
    if manifest is not None:
        evidence.append(f"{manifest.name} present")
    if base is not None:
        evidence.append(f"{name}/ exists")
    python = None
    bins: tuple[Path, ...] = ()
    if found:
        chosen_dir, python, bin_dir = found[0]
        bins = (bin_dir,)
        evidence.append(
            f"existing interpreter: {_rel_text(facts.root, python)}")
        if len(found) > 1:
            evidence.append(
                f"ambiguous: {len(found)} {kind} environments have "
                f"interpreters; {_rel_text(facts.root, chosen_dir)} ranked "
                "first")
    return _Candidate(
        kind=kind,
        score=score if python is not None else 24,
        python=python,
        bin_dirs=bins,
        package_manager=kind,
        evidence=_clean_evidence(evidence),
        order=_ORDER[kind],
    )


def _active_environment(facts: _Facts) -> list[_Candidate]:
    """Explicit user state from VIRTUAL_ENV / CONDA_PREFIX (may be outside)."""
    out: list[_Candidate] = []
    raw = str(facts.env.get("VIRTUAL_ENV") or "").strip()
    if raw:
        try:
            directory = Path(raw).expanduser().resolve()
        except (OSError, ValueError):
            directory = Path(raw).expanduser()
        python = _find_python(directory / _bin_name())
        if python is None:
            python = _find_python(directory)
        if python is not None:
            out.append(_Candidate(
                kind="venv",
                score=85,
                python=python,
                bin_dirs=(python.parent,),
                package_manager="pip",
                evidence=_clean_evidence([
                    "VIRTUAL_ENV names an existing interpreter"]),
                order=_ORDER["active-venv"],
            ))
    raw = str(facts.env.get("CONDA_PREFIX") or "").strip()
    if raw:
        try:
            directory = Path(raw).expanduser().resolve()
        except (OSError, ValueError):
            directory = Path(raw).expanduser()
        python = _conda_python(directory)
        if python is not None:
            out.append(_Candidate(
                kind="conda",
                score=84,
                python=python,
                bin_dirs=_conda_bins(directory),
                package_manager="conda",
                evidence=_clean_evidence([
                    "CONDA_PREFIX names an existing interpreter"]),
                order=_ORDER["active-conda"],
            ))
    return out


def _metadata_candidate(facts: _Facts, kind: str, files: tuple[str, ...],
                        score: int) -> _Candidate | None:
    """A package manager detected only from local metadata (no interpreter)."""
    if not files:
        return None
    evidence = [f"{f} present" for f in files]
    bins: tuple[Path, ...] = ()
    if kind in _PY_MANAGERS:
        for name in (".venv", "venv"):
            directory = _directory(facts.root, name)
            if directory is None:
                continue
            raw_bin = directory / _bin_name()
            if not raw_bin.is_dir():
                continue
            bin_dir = _contained(facts.root, raw_bin)
            if bin_dir is not None:
                bins = (bin_dir,)
                evidence.append(
                    f"{name}/ exists without a resolved interpreter")
                break
    if kind in _NODE_MANAGERS:
        bins = _node_bin(facts.root)
        if bins:
            evidence.append("node_modules/.bin exists")
    manager = "npm" if kind == "node" else kind
    return _Candidate(
        kind=kind,
        score=score,
        python=None,
        bin_dirs=bins,
        package_manager=manager,
        evidence=_clean_evidence(evidence),
        order=_ORDER[kind],
    )


def _ambiguous(candidates: list[_Candidate], winner: _Candidate) -> list[str]:
    """Human-readable ambiguity notes for evidence, deterministic order."""
    notes: list[str] = []
    interpreters = [c for c in candidates if c.python is not None]
    distinct: list[_Candidate] = []
    for candidate in interpreters:
        if not any(_same_path(candidate.python, seen.python)
                   for seen in distinct):
            distinct.append(candidate)
    if len(distinct) > 1:
        others = [c for c in distinct if c is not winner]
        names = ", ".join(c.kind for c in others[:3])
        notes.append(
            f"ambiguous: {len(distinct)} existing interpreters "
            f"({names}); {winner.kind} ranked first by deterministic "
            "precedence")

    py_managers = [c for c in candidates
                   if c.kind in _PY_MANAGERS and c.score >= 50
                   and c.kind != winner.kind]
    if py_managers and winner.kind in _PY_MANAGERS:
        names = ", ".join(c.kind for c in py_managers[:3])
        notes.append(
            f"ambiguous: multiple Python package managers detected "
            f"({winner.kind}, {names}); {winner.kind} ranked first")

    node_managers = [c for c in candidates
                     if c.kind in _NODE_MANAGERS and c.score >= 50
                     and c.kind != winner.kind]
    if node_managers and winner.kind in _NODE_MANAGERS:
        names = ", ".join(c.kind for c in node_managers[:3])
        notes.append(
            f"ambiguous: multiple JavaScript package managers detected "
            f"({winner.kind}, {names}); {winner.kind} ranked first")

    ordered = sorted(candidates, key=lambda c: (-c.score, c.order))
    if len(ordered) > 1 and ordered[0].score == ordered[1].score \
            and ordered[1].kind != winner.kind:
        notes.append(
            f"ambiguous: {ordered[0].kind} and {ordered[1].kind} share the "
            f"top evidence score; {winner.kind} ranked first by fixed order")
    return notes


def _also_detected(candidates: list[_Candidate], winner: _Candidate) -> list[str]:
    kinds: list[str] = []
    for candidate in sorted(candidates, key=lambda c: (-c.score, c.order)):
        if candidate is winner or candidate.kind == winner.kind:
            continue
        if candidate.score < 50:
            continue
        label = candidate.kind
        if candidate.kind == "node":
            label = "node/npm"
        if label not in kinds:
            kinds.append(label)
    return [f"also detected: {', '.join(kinds[:6])}"] if kinds else []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_environment(root: Path, *, env: dict[str, str] | None = None
                        ) -> ResolvedEnvironment:
    """Detect the project's environment without executing anything."""
    try:
        project_root = Path(root).expanduser().resolve()
    except (TypeError, ValueError, OSError):
        project_root = Path.cwd().resolve()
    source = dict(os.environ if env is None else env)
    facts = _facts(project_root, source)

    candidates: list[_Candidate] = []
    for name in (".venv", "venv"):
        candidate = _candidate_venv(facts, name)
        if candidate is not None:
            candidates.append(candidate)
    conda_project = _candidate_conda_project(facts)
    if conda_project is not None:
        candidates.append(conda_project)
    for name, kind, score in ((".tox", "tox", 90), (".nox", "nox", 89)):
        candidate = _candidate_env_manager(facts, name, kind, score)
        if candidate is not None:
            candidates.append(candidate)
    candidates.extend(_active_environment(facts))

    manager_specs = (
        ("uv", facts.uv, 62),
        ("poetry", facts.poetry, 61),
        ("pipenv", facts.pipenv, 60),
        ("conda", facts.conda, 58),
        ("pnpm", facts.pnpm, 57),
        ("bun", facts.bun, 56),
        ("yarn", facts.yarn, 55),
        ("node", facts.npm, 54),
        ("cargo", facts.cargo, 53),
        ("go", facts.go, 53),
    )
    for kind, files, score in manager_specs:
        candidate = _metadata_candidate(facts, kind, files, score)
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        candidates.append(_Candidate(
            kind="system",
            score=0,
            python=None,
            bin_dirs=(),
            package_manager="",
            evidence=_clean_evidence([
                "no project environment metadata or interpreter detected; "
                "the ambient PATH is used unchanged"]),
            order=_ORDER["system"],
        ))

    ordered = sorted(candidates, key=lambda c: (-c.score, c.order, c.kind))
    winner = ordered[0]

    bins: list[Path] = []
    for directory in (*winner.bin_dirs, *_node_bin(project_root)):
        if directory.is_dir() and directory not in bins:
            bins.append(directory)

    evidence = list(winner.evidence)
    evidence.extend(_ambiguous(candidates, winner))
    evidence.extend(_also_detected(candidates, winner))
    if winner.kind == "system" and not evidence:
        evidence.append("the ambient PATH is used unchanged")

    return ResolvedEnvironment(
        root=project_root,
        kind=winner.kind,
        python=winner.python,
        bin_dirs=tuple(bins),
        package_manager=winner.package_manager,
        workspace_root=project_root,
        evidence=_clean_evidence(evidence),
    )


def _path_key(env: dict[str, str]) -> str:
    """The environment's PATH key, preserving Windows' usual `Path` spelling."""
    if "PATH" in env:
        return "PATH"
    for key in env:
        if key.upper() == "PATH":
            return key
    return "PATH"


def _venv_root(python: Path) -> Path | None:
    if python.parent.name.lower() not in ("bin", "scripts"):
        return None
    candidate = python.parent.parent
    try:
        if candidate.is_dir():
            return candidate
    except OSError:
        pass
    return None


def _conda_prefix(resolved: ResolvedEnvironment) -> Path | None:
    if resolved.kind != "conda" or resolved.python is None:
        return None
    candidates = [resolved.python.parent]
    if resolved.python.parent.name.lower() in ("scripts", "library"):
        candidates.append(resolved.python.parent.parent)
    if resolved.python.parent.name.lower() == "bin":
        candidates.append(resolved.python.parent.parent)
    for candidate in candidates:
        try:
            if (candidate / "conda-meta").is_dir():
                return candidate
        except OSError:
            continue
    return None


def command_env(resolved: ResolvedEnvironment, cfg=None) -> dict[str, str]:
    """The scrubbed child environment with resolved bin dirs prepended.

    Composition rule: start from :func:`safety.child_env` so the existing
    allow-list, secret rejection and ``env_passthrough`` policy decide what
    survives; only then prepend directories and set the active-environment
    markers.  There is deliberately no path that copies ``os.environ`` into
    the result after the scrub.
    """
    env = safety.child_env(cfg)
    key = _path_key(env)
    raw_path = str(env.get(key) or "")
    parts = [p for p in raw_path.split(os.pathsep) if p]
    seen = {os.path.normcase(os.path.abspath(p)) for p in parts}
    prepend: list[str] = []
    for directory in resolved.bin_dirs:
        try:
            if not directory.is_dir():
                continue
        except OSError:
            continue
        try:
            marker = os.path.normcase(os.path.abspath(str(directory)))
        except (OSError, ValueError):
            continue
        if marker in seen:
            continue
        seen.add(marker)
        prepend.append(str(directory))
    if prepend:
        env[key] = os.pathsep.join(prepend + parts)

    # Preserve the one legacy fallback every `proc` child had: the directory
    # of the interpreter running Pedacito (pytest/ruff are core dependencies
    # there). It goes last, after the project's own resolved directories.
    try:
        runner_dir = Path(sys.executable).resolve().parent
        runner_marker = os.path.normcase(os.path.abspath(str(runner_dir)))
        if runner_dir.is_dir() and runner_marker not in seen:
            env[key] = os.pathsep.join(
                [p for p in (env.get(key) or "").split(os.pathsep) if p]
                + [str(runner_dir)])
    except (OSError, ValueError):
        pass

    if resolved.python is not None:
        if resolved.kind != "conda":
            venv = _venv_root(resolved.python)
            if venv is not None:
                env["VIRTUAL_ENV"] = str(venv)
        prefix = _conda_prefix(resolved)
        if prefix is not None:
            env["CONDA_PREFIX"] = str(prefix)
    return env


def _path_exts(env: dict[str, str] | None) -> tuple[str, ...]:
    source = dict(os.environ if env is None else env)
    raw = ""
    for key, value in source.items():
        if key.upper() == "PATHEXT":
            raw = str(value)
            break
    exts = [e.strip().lower() for e in raw.split(os.pathsep) if e.strip()]
    return tuple(exts) or (".exe", ".com", ".bat", ".cmd")


def find_executable(resolved: ResolvedEnvironment, name: str, *,
                    env: dict[str, str] | None = None) -> Path | None:
    """The first executable named `name` in the resolved bin dirs, or on PATH.

    This is detection only; it never runs the executable. On Windows the
    ``PATHEXT`` search is delegated to :func:`shutil.which`.
    """
    name = str(name or "").strip()
    if not name:
        return None
    names = [name]
    if os.name == "nt" and not Path(name).suffix:
        names.extend(name + ext for ext in _path_exts(env))
    for directory in resolved.bin_dirs:
        for candidate_name in names:
            candidate = directory / candidate_name
            if _is_executable(candidate):
                return candidate
    source = dict(os.environ if env is None else env)
    path_value = next(
        (str(value) for key, value in source.items() if key.upper() == "PATH"),
        "")
    if env is None:
        found = shutil.which(name)
    else:
        found = (shutil.which(name, path=path_value) if path_value
                 else shutil.which(name))
    return Path(found) if found else None


# ---------------------------------------------------------------------------
# Dependency roots (read-only material for the API oracle)
# ---------------------------------------------------------------------------

def _site_dirs_of(anchor: Path) -> list[Path]:
    """Existing `site-packages` dirs directly under one interpreter prefix.

    Both Python layouts are checked without executing anything:
    ``<prefix>/lib/pythonX.Y/site-packages`` on POSIX/conda and
    ``<prefix>/Lib/site-packages`` on Windows. Every candidate resolves
    through :func:`_contained`, so a site-packages symlink/junction that aims
    outside its own interpreter prefix is ignored.
    """
    out: list[Path] = []
    candidates = [anchor / "Lib" / "site-packages",
                  anchor / "lib" / "site-packages"]
    for lib_name in ("lib", "lib64", "Lib"):
        lib = anchor / lib_name
        try:
            if not lib.is_dir():
                continue
            children = sorted(lib.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        for child in children:
            if re.match(r"(?i)^python\d", child.name) and child.is_dir():
                candidates.append(child / "site-packages")
    for raw in candidates:
        safe = _contained(anchor, raw)
        if safe is not None and safe.is_dir() \
                and safe.name.lower() == "site-packages":
            if safe not in out:
                out.append(safe)
    return out


def site_packages_dirs(resolved: ResolvedEnvironment) -> tuple[Path, ...]:
    """Existing `site-packages` directories of the resolved interpreter.

    This is detection only: no import, no execution, no network. The
    interpreter's own prefix is the approval boundary -- an explicitly active
    outside ``VIRTUAL_ENV``/``CONDA_PREFIX`` is user state and remains
    allowed, while a site-packages link that leaves *that* prefix is skipped.
    """
    anchors: list[Path] = []
    if resolved.python is not None:
        anchors.append(_venv_root(resolved.python) or resolved.python.parent)
    for directory in resolved.bin_dirs:
        anchors.append(directory.parent)
    anchors.extend((resolved.workspace_root, resolved.root))
    out: list[Path] = []
    for anchor in anchors:
        try:
            if not anchor.is_dir():
                continue
        except OSError:
            continue
        for found in _site_dirs_of(anchor):
            if found not in out:
                out.append(found)
    if not out:
        # The resolver chose "system": the interpreter running Pedacito is
        # then the ambient interpreter the project would use. Its own
        # site-packages are still dependency material, read-only.
        try:
            import sys
            import sysconfig
            ambient = (resolved.python is None
                       or _same_path(resolved.python, Path(sys.executable)))
            if ambient:
                for key in ("purelib", "platlib"):
                    raw = str(sysconfig.get_path(key) or "").strip()
                    if not raw:
                        continue
                    candidate = Path(raw)
                    if candidate.is_dir() and candidate not in out:
                        out.append(candidate)
        except (OSError, ValueError, KeyError):
            pass
    return tuple(out)


def dependency_roots(resolved: ResolvedEnvironment) -> tuple[Path, ...]:
    """Read-only roots an API index may inspect for this environment.

    The project and workspace roots come first, then the resolved
    interpreter's site-packages directories and a contained ``node_modules``.
    Nothing outside these explicitly approved roots is traversed; callers must
    still reject links that escape a root while walking it.
    """
    out: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        try:
            safe = path.resolve()
        except OSError:
            return
        if not safe.is_dir():
            return
        marker = os.path.normcase(str(safe))
        if marker in seen:
            return
        seen.add(marker)
        out.append(safe)

    add(resolved.root)
    add(resolved.workspace_root)
    for directory in site_packages_dirs(resolved):
        add(directory)
    for base in (resolved.root, resolved.workspace_root):
        node_modules = base / "node_modules"
        safe = _contained(base, node_modules)
        if safe is not None:
            add(safe)
    return tuple(out)


def find_executables(resolved: ResolvedEnvironment, names,
                     *, env: dict[str, str] | None = None) -> dict[str, Path | None]:
    """``{name: executable_path | None}`` for several tool names at once.

    One pass over the resolved bin dirs and ``PATH`` instead of a separate
    scan per name, so the autofix ladder can ask "which of ruff / biome /
    eslint / gofmt / cargo is actually installed here?" without re-walking the
    directories. Detection only: nothing here executes a tool. The mapping is
    returned in the caller's name order (a dict preserves insertion order).
    """
    return {name: find_executable(resolved, name, env=env) for name in names}
