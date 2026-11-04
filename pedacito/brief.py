"""The repo brief: a deterministic 300-token orientation for the model.

    text = build_brief(index, cfg, max_tokens=500)

What a new engineer learns in the first five minutes, built from facts the
index and the manifests already hold -- no model call, no staleness (it is
rebuilt from the index each run, cached per index generation):

    stack         languages by file count, frameworks/manifests found
    entry points  main/cli/app files, console scripts, package.json "main"
    commands      test, lint, build, run -- from the manifests
    layout        top-level directories with their role and busiest file

It is shown ABOVE the project map, with its own small budget, and labelled as
orientation (a hint), never as truth about behaviour.
"""

from __future__ import annotations

import json
import posixpath
import re
import tomllib
from collections import Counter
from pathlib import Path

from . import testsel

_LANG_NAMES = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript",
    ".mjs": "JavaScript", ".cjs": "JavaScript", ".ts": "TypeScript",
    ".tsx": "TypeScript", ".go": "Go", ".rs": "Rust", ".java": "Java",
    ".cs": "C#", ".rb": "Ruby", ".c": "C", ".h": "C/C++", ".cc": "C++",
    ".cpp": "C++", ".cxx": "C++", ".hpp": "C++", ".hh": "C++",
    ".php": "PHP", ".kt": "Kotlin", ".kts": "Kotlin", ".swift": "Swift",
    ".scala": "Scala", ".lua": "Lua",
}
_DIR_ROLES = {
    "src": "source", "lib": "library code", "app": "application",
    "pkg": "packages", "cmd": "command entry points", "internal": "internal packages",
    "tests": "tests", "test": "tests", "spec": "tests", "__tests__": "tests",
    "docs": "documentation", "doc": "documentation", "scripts": "scripts",
    "bin": "executables", "tools": "tooling", "examples": "examples",
    "migrations": "DB migrations", "static": "static assets",
    "templates": "templates", "public": "static assets", "assets": "assets",
    "config": "configuration", "configs": "configuration", "web": "web frontend",
    "api": "API", "server": "server", "client": "client", "core": "core logic",
    "models": "data models", "utils": "utilities", "vendor": "vendored code",
    "build": "build output", "ci": "CI",
}
_ENTRY_NAMES = {
    "main", "__main__", "cli", "app", "server", "index", "manage", "run",
    "wsgi", "asgi", "program", "start",
}
_CACHE: dict[tuple, str] = {}


def _read_toml(p: Path) -> dict:
    try:
        return tomllib.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _manifest_facts(root: Path) -> tuple[list[str], list[str], list[str]]:
    """(stack notes, entry points, commands) from manifest files."""
    stack: list[str] = []
    entries: list[str] = []
    commands: list[str] = []
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        data = _read_toml(pyproject)
        proj = data.get("project") or {}
        deps = [re.split(r"[<>=!~\[ ;]", d, maxsplit=1)[0]
                for d in proj.get("dependencies", [])]
        note = "pyproject.toml"
        if proj.get("name"):
            note += f" ({proj['name']})"
        if deps:
            note += " deps: " + ", ".join(deps[:6])
        stack.append(note)
        for name, target in list((proj.get("scripts") or {}).items())[:3]:
            entries.append(f"{name} -> {target}")
        tool = data.get("tool") or {}
        if "ruff" in tool:
            commands.append("lint: ruff check .")
        if "mypy" in tool:
            commands.append("types: mypy .")
    elif (root / "setup.py").is_file() or (root / "requirements.txt").is_file():
        stack.append("Python (setup.py/requirements.txt)")
    pkg = root / "package.json"
    if pkg.is_file():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        deps = list((data.get("dependencies") or {}))[:6]
        note = "package.json"
        if data.get("name"):
            note += f" ({data['name']})"
        if deps:
            note += " deps: " + ", ".join(deps)
        stack.append(note)
        if data.get("main"):
            entries.append(f"main: {data['main']}")
        bins = data.get("bin")
        if isinstance(bins, dict):
            entries += [f"bin {k} -> {v}" for k, v in list(bins.items())[:2]]
        for k in ("build", "lint", "dev", "start"):
            if (data.get("scripts") or {}).get(k):
                commands.append(f"{k}: npm run {k}")
    cargo = root / "Cargo.toml"
    if cargo.is_file():
        data = _read_toml(cargo)
        stack.append("Cargo.toml" + (f" ({data['package']['name']})"
                                     if (data.get("package") or {}).get("name") else ""))
        commands.append("build: cargo build")
    if (root / "go.mod").is_file():
        try:
            first = (root / "go.mod").read_text(encoding="utf-8").splitlines()[0]
        except (OSError, IndexError):
            first = "go.mod"
        stack.append(first.strip()[:60])
        commands.append("build: go build ./...")
    for name in ("pom.xml", "build.gradle", "build.gradle.kts", "Gemfile"):
        if (root / name).is_file():
            stack.append(name)
    mk = next((root / n for n in ("Makefile", "makefile") if (root / n).is_file()), None)
    if mk is not None:
        try:
            targets = re.findall(r"^([A-Za-z][\w-]*)\s*:(?!=)",
                                 mk.read_text(encoding="utf-8", errors="replace"),
                                 re.MULTILINE)
        except OSError:
            targets = []
        if targets:
            commands.append("make targets: " + ", ".join(dict.fromkeys(targets[:8])))
    return stack, entries, commands


def build_brief(index, cfg=None, max_tokens: int = 500) -> str:
    """The repo brief text ('' when the project is empty), within roughly
    `max_tokens` (trimmed by dropping the layout lines last-first)."""
    # Identity by content, not `id(index)`: a garbage-collected index can hand
    # its address to a new one, and a stale brief would leak across projects.
    key = (str(index.root), index.generation, tuple(index.files), max_tokens,
           getattr(cfg, "test_command", ""))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    root = Path(index.root)
    files = index.files
    if not files:
        return ""
    lines: list[str] = ["REPO BRIEF (orientation only; verify against source)"]

    langs = Counter(_LANG_NAMES[posixpath.splitext(f)[1].lower()]
                    for f in files if posixpath.splitext(f)[1].lower() in _LANG_NAMES)
    stack_note, entries, commands = _manifest_facts(root)
    if langs:
        lines.append("stack: " + ", ".join(f"{n} ({c})" for n, c in langs.most_common(4)))
    for s in stack_note[:2]:
        lines.append(f"  {s}")

    pr = index.graph.pagerank()
    entry_files = [f for f in files
                   if posixpath.splitext(posixpath.basename(f))[0].lower() in _ENTRY_NAMES
                   and posixpath.splitext(f)[1].lower() in _LANG_NAMES]
    entry_files.sort(key=lambda f: (f.count("/"), -pr.get(f, 0.0), f))
    shown = entries[:2] + [f for f in entry_files[:4]]
    if shown:
        lines.append("entry points: " + ", ".join(shown[:5]))

    cmds: list[str] = []
    test_cmd = testsel.detect_test_command(root, cfg)
    if test_cmd:
        cmds.append(f"test: {test_cmd}")
    cmds += commands
    if cmds:
        lines.append("commands: " + "; ".join(cmds[:5]))

    groups: dict[str, list[str]] = {}
    for f in files:
        top = f.split("/", 1)[0] if "/" in f else "(root)"
        groups.setdefault(top, []).append(f)
    layout: list[str] = []
    for d, fl in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:8]:
        role = _DIR_ROLES.get(d.lower(), "")
        busiest = max(fl, key=lambda f: pr.get(f, 0.0))
        bit = f"{d}{'/' if d != '(root)' else ''} {len(fl)} files"
        if role:
            bit += f" [{role}]"
        if len(fl) > 3:
            bit += f", central: {posixpath.basename(busiest)}"
        layout.append(bit)
    if layout:
        lines.append("layout: " + "; ".join(layout))

    text = "\n".join(lines)
    # shrink to budget: drop layout detail, then stack notes
    from .llm import estimate_tokens
    while estimate_tokens(text) > max_tokens and len(lines) > 2:
        lines.pop()
        text = "\n".join(lines)
    if len(lines) <= 1:
        text = ""
    _CACHE[key] = text
    if len(_CACHE) > 16:
        _CACHE.pop(next(iter(_CACHE)))
    return text
