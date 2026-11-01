"""Read-only git context: status, recent history, co-change, and the `git` tool.

    st = status(root)                      # branch, dirty files, conflicts
    recent_files(root)                     # files touched by the last commits
    cochange(root)                         # {file: {neighbour: times}}
    tool_git(index, cfg, "log -n 5 src/app.py")

Everything runs `git` as an argv (never through a shell), with a short timeout
and prompts disabled, and degrades to "nothing known" when git is missing or
the project is not a repository. Nothing here writes to the repository.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

_TIMEOUT = 8

_cache: dict[tuple, object] = {}


def _git(root, *args: str, timeout: int = _TIMEOUT) -> tuple[int, str]:
    """Run `git <args>` in `root`; (returncode, stdout). Never raises."""
    exe = shutil.which("git")
    if not exe:
        return 127, ""
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_PAGER="cat",
               PAGER="cat", LC_ALL="C")
    try:
        r = subprocess.run([exe, "-c", "core.quotepath=off", *args],
                           cwd=str(root), capture_output=True, timeout=timeout,
                           env=env, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return 1, ""
    return r.returncode, r.stdout.decode("utf-8", "replace")


def is_repo(root) -> bool:
    """True when `root` is inside a git work tree."""
    key = ("repo", str(root))
    if key not in _cache:
        rc, out = _git(root, "rev-parse", "--is-inside-work-tree")
        _cache[key] = rc == 0 and out.strip() == "true"
    return bool(_cache[key])


def head(root) -> str:
    """Short HEAD hash ('' for no commits / no repo)."""
    rc, out = _git(root, "rev-parse", "--short", "HEAD")
    return out.strip() if rc == 0 else ""


def status(root) -> dict:
    """{branch, dirty: [paths], untracked: [paths], conflicts: [paths]}."""
    out = {"branch": "", "dirty": [], "untracked": [], "conflicts": []}
    if not is_repo(root):
        return out
    rc, text = _git(root, "status", "--porcelain=v1", "-b")
    if rc != 0:
        return out
    for i, line in enumerate(text.splitlines()):
        if i == 0 and line.startswith("## "):
            out["branch"] = line[3:].split("...")[0].split(" ")[0]
            continue
        code, path = line[:2], line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if code == "??":
            out["untracked"].append(path)
        elif "U" in code or code in ("AA", "DD"):
            out["conflicts"].append(path)
        else:
            out["dirty"].append(path)
    return out


def has_conflicts(root, rel: str | None = None) -> bool:
    """Whether the work tree is mid-merge (or `rel` has conflict markers)."""
    st = status(root)
    if rel is None:
        return bool(st["conflicts"])
    return rel in st["conflicts"]


def recent_commits(root, n: int = 5) -> list[tuple[str, str]]:
    """[(short hash, subject)] newest first."""
    if not is_repo(root):
        return []
    rc, out = _git(root, "log", f"-n{int(n)}", "--pretty=format:%h\t%s")
    if rc != 0:
        return []
    rows = []
    for line in out.splitlines():
        h, _, subj = line.partition("\t")
        if h:
            rows.append((h, subj[:100]))
    return rows


def recent_files(root, commits: int = 20) -> list[str]:
    """Files touched by the last `commits` commits, most recent first."""
    if not is_repo(root):
        return []
    key = ("recent", str(root), head(root), commits)
    if key in _cache:
        return _cache[key]            # type: ignore[return-value]
    rc, out = _git(root, "log", f"-n{int(commits)}", "--name-only",
                   "--pretty=format:")
    seen: dict[str, None] = {}
    if rc == 0:
        for line in out.splitlines():
            if line.strip():
                seen.setdefault(line.strip(), None)
    _cache[key] = list(seen)
    return _cache[key]                # type: ignore[return-value]


def cochange(root, commits: int = 200, max_files: int = 30
             ) -> dict[str, dict[str, int]]:
    """How often two files changed in the same commit (commits touching more
    than `max_files` files -- reformats, vendor drops -- are ignored). The
    history signal for map ranking and test selection."""
    if not is_repo(root):
        return {}
    key = ("cochange", str(root), head(root), commits)
    if key in _cache:
        return _cache[key]            # type: ignore[return-value]
    rc, out = _git(root, "log", f"-n{int(commits)}", "--name-only",
                   "--pretty=format:@@@", timeout=20)
    pairs: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    if rc == 0:
        for block in out.split("@@@"):
            files = [f.strip() for f in block.splitlines() if f.strip()]
            if len(files) < 2 or len(files) > max_files:
                continue
            for a in files:
                for b in files:
                    if a != b:
                        pairs[a][b] += 1
    result = {a: dict(b) for a, b in pairs.items()}
    _cache[key] = result
    return result


def preamble(root, cfg=None, max_chars: int = 420) -> str:
    """Branch, dirty files and the last commit subjects as a short block for
    the task prompt; '' outside a repository or when disabled."""
    if cfg is not None and not getattr(cfg, "git_context", True):
        return ""
    if not is_repo(root):
        return ""
    st = status(root)
    bits = []
    head_line = f"git: branch {st['branch'] or '(detached)'}"
    if st["conflicts"]:
        head_line += f", MERGE CONFLICTS in {', '.join(st['conflicts'][:3])}"
    if st["dirty"]:
        shown = ", ".join(st["dirty"][:4])
        more = f" +{len(st['dirty']) - 4}" if len(st["dirty"]) > 4 else ""
        head_line += f"; uncommitted: {shown}{more}"
    bits.append(head_line)
    commits = recent_commits(root, 3)
    if commits:
        bits.append("recent: " + "; ".join(f"{h} {s[:50]}" for h, s in commits))
    return "\n".join(bits)[:max_chars]


# ------------------------------------------------------------------ the tool
_ALLOWED = ("status", "diff", "log", "show", "blame", "branch", "ls-files")
_SAFE_ARG_RE = re.compile(r"^[\w@./:\-^~,=+]+$")


def tool_git(index, cfg, args: str) -> str:
    """Read-only git for the model: `status`, `diff [path]`, `log [-n N]
    [path]`, `show <rev>[:path]`, `blame <path> [-L a,b]`, `branch`. Anything
    else (and every flag that could write or run a program) is refused."""
    root = Path(index.root)
    if not getattr(cfg, "allow_git", True):
        return "ERROR: `git` is disabled (allow_git = false)."
    if shutil.which("git") is None:
        return "ERROR: git is not installed."
    if not is_repo(root):
        return "ERROR: this project is not a git repository."
    try:
        import shlex
        parts = shlex.split(args or "status")
    except ValueError as e:
        return f"ERROR: could not parse git arguments ({e})."
    if not parts:
        parts = ["status"]
    sub, rest = parts[0].lower(), parts[1:]
    if sub not in _ALLOWED:
        return (f"ERROR: `git {sub}` is not available; read-only commands: "
                f"{', '.join(_ALLOWED)}.")
    argv: list[str] = []
    if sub == "status":
        argv = ["status", "--short", "--branch"]
    elif sub == "branch":
        argv = ["branch", "--list", "-vv"]
    elif sub == "ls-files":
        argv = ["ls-files"] + _paths(rest)
    elif sub == "diff":
        argv = ["diff", "--stat", "--patch", "--no-color", "-U2"] + _paths(rest)
    elif sub == "log":
        n = 8
        paths: list[str] = []
        i = 0
        while i < len(rest):
            a = rest[i]
            if a in ("-n", "--max-count") and i + 1 < len(rest) \
                    and rest[i + 1].isdigit():
                n = min(int(rest[i + 1]), 30)
                i += 2
                continue
            m = re.fullmatch(r"-n?(\d+)", a)
            if m:
                n = min(int(m.group(1)), 30)
            elif _SAFE_ARG_RE.match(a) and not a.startswith("-"):
                paths.append(a)
            i += 1
        argv = ["log", f"-n{n}", "--date=short",
                "--pretty=format:%h %ad %an: %s"] + (["--"] + paths if paths else [])
    elif sub == "show":
        target = next((a for a in rest if _SAFE_ARG_RE.match(a)
                       and not a.startswith("-")), "HEAD")
        argv = ["show", "--stat", "--patch", "--no-color", "-U2", target]
    elif sub == "blame":
        path = next((a for a in rest if _SAFE_ARG_RE.match(a)
                     and not a.startswith("-") and not re.match(r"^\d", a)), "")
        if not path:
            return "ERROR: git blame needs a file path."
        argv = ["blame", "--date=short", "-w"]
        lr = next((a for a in rest if re.fullmatch(r"\d+(,\d+)?", a)), "")
        if "-L" in rest and lr:
            argv += ["-L", lr]
        argv += ["--", path]
    rc, out = _git(root, *argv, timeout=15)
    cap = max(400, int(getattr(cfg, "max_shell_output_chars", 8000) or 8000))
    out = out.strip() or "(no output)"
    if len(out) > cap:
        out = out[:cap] + f"\n[truncated at {cap} chars]"
    prefix = "" if rc == 0 else f"[git exited {rc}]\n"
    return f"git {' '.join(argv[:1])}:\n{prefix}{out}"


def _paths(rest: list[str]) -> list[str]:
    paths = [a for a in rest if _SAFE_ARG_RE.match(a) and not a.startswith("-")]
    return ["--"] + paths if paths else []
