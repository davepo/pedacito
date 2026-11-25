"""Guided, opt-in git and pull-request workflows.

    st = gitops.status_hunks(root)          # branch + staged/unstaged diffs
    gitops.stage_files(root, ["src/app.py"])
    msg = gitops.draft_commit_message(root, client, cfg)
    gitops.create_commit(root, msg)         # scans the staged diff first

Safety rules, by construction:

* every `git` and `gh` call is an argv list with ``shell=False``;
* nothing here ever pushes (no ``git push``, no ``--force``), so a commit is
  local until the user pushes it themselves;
* every path from a caller is validated with
  :func:`pedacito.security.resolve_safe_path` before it reaches `git`;
* the staged diff is scanned with
  :func:`pedacito.security.scan_text_for_secrets` before every commit, and the
  branch diff before a pull request, so a key cannot ride along unnoticed.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from . import security

__all__ = [
    "GitOpsError",
    "status_hunks",
    "stage_files",
    "stage_all_files",
    "unstage_files",
    "scan_staged_secrets",
    "draft_commit_message",
    "create_commit",
    "create_branch",
    "draft_pr",
    "create_pr",
]

_TIMEOUT = 60
_MAX_DIFF_CHARS = 12000
_BRANCH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
_CONVENTIONAL_RE = re.compile(
    r"^(?:feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
    r"(?:\([^)]+\))?: \S")
_DOC_EXTS = {".md", ".rst", ".txt", ".adoc"}


class GitOpsError(RuntimeError):
    """A guided git operation was refused or failed."""


# --- process plumbing ----------------------------------------------------------

def _run(argv: list[str], root: Path, timeout: int = _TIMEOUT):
    """Run `argv` as a list in `root`; never through a shell."""
    exe = shutil.which(argv[0])
    if not exe:
        raise GitOpsError(
            f"`{argv[0]}` is not installed or not on PATH.")
    env = dict(os.environ)
    env.update({
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_PAGER": "cat",
        "PAGER": "cat",
        "LC_ALL": "C",
        "NO_COLOR": "1",
        "GH_PROMPT_DISABLED": "1",
    })
    try:
        return subprocess.run(
            [exe, *argv[1:]], cwd=str(root), shell=False,
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, env=env,
            stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as exc:
        raise GitOpsError(f"`{argv[0]}` failed to start: {exc}") from exc


def _git(root: Path, *args: str, timeout: int = _TIMEOUT):
    return _run(["git", *args], root, timeout=timeout)


def _gh(root: Path, *args: str, timeout: int = _TIMEOUT):
    return _run(["gh", *args], root, timeout=timeout)


def _output(proc) -> str:
    return ((proc.stdout or "") + (proc.stderr or "")).strip()


def _require_repo(root) -> Path:
    """The resolved root, when it is inside a git work tree."""
    try:
        resolved = Path(root).expanduser().resolve()
    except OSError as exc:
        raise GitOpsError(f"project root could not be resolved: {root} ({exc})")
    if not resolved.is_dir():
        raise GitOpsError(f"project root is not a directory: {resolved}")
    proc = _git(resolved, "rev-parse", "--is-inside-work-tree")
    if proc.returncode != 0 or (proc.stdout or "").strip() != "true":
        raise GitOpsError(f"{resolved} is not a git repository.")
    return resolved


# --- status --------------------------------------------------------------------

def _parse_status(text: str) -> dict:
    branch = ""
    staged: list[str] = []
    unstaged: list[str] = []
    untracked: list[str] = []
    conflicts: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            head = line[3:].strip()
            if head.startswith("No commits yet on "):
                branch = head[len("No commits yet on "):].strip()
            elif head:
                branch = head.split("...")[0].split(" ")[0]
            continue
        if len(line) < 3:
            continue
        code, path = line[:2], line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1].strip()
        path = path.strip('"')
        if not path:
            continue
        if code == "??":
            untracked.append(path)
            continue
        if "U" in code or code in ("AA", "DD"):
            conflicts.append(path)
        if code[0] not in " ?":
            staged.append(path)
        if code[1] != " ":
            unstaged.append(path)
    return {
        "branch": branch,
        "staged": sorted(dict.fromkeys(staged)),
        "unstaged": sorted(dict.fromkeys(unstaged)),
        "untracked": sorted(dict.fromkeys(untracked)),
        "conflicts": sorted(dict.fromkeys(conflicts)),
    }


def _diff_path(header_line: str) -> str:
    text = header_line[len("diff --git "):].strip()
    match = re.match(r"^a/(.+?) b/(.+)$", text)
    return match.group(2) if match else text


def _parse_diffs(text: str) -> dict[str, str]:
    """{project-relative path: unified diff block}."""
    out: dict[str, str] = {}
    current = ""
    chunks: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if current:
                out[current] = "".join(chunks)
            current = _diff_path(line.rstrip("\r\n"))
            chunks = [line]
        elif current:
            chunks.append(line)
    if current:
        out[current] = "".join(chunks)
    return out


def status_hunks(root) -> dict:
    """Branch, dirty state and the per-file diffs of the working tree."""
    resolved = _require_repo(root)
    proc = _git(resolved, "status", "--porcelain=v1", "-b")
    if proc.returncode != 0:
        raise GitOpsError(_output(proc) or "`git status` failed.")
    parsed = _parse_status(proc.stdout or "")

    cached = _git(resolved, "diff", "--cached", "--no-color", "--unified=3")
    if cached.returncode != 0:
        raise GitOpsError(_output(cached) or "`git diff --cached` failed.")
    work = _git(resolved, "diff", "--no-color", "--unified=3")
    if work.returncode != 0:
        raise GitOpsError(_output(work) or "`git diff` failed.")

    staged_diffs = _parse_diffs(cached.stdout or "")
    unstaged_diffs = _parse_diffs(work.stdout or "")
    paths = sorted(set(staged_diffs) | set(unstaged_diffs)
                   | set(parsed["staged"]) | set(parsed["unstaged"]))
    hunks = {
        path: {"staged": staged_diffs.get(path, ""),
               "unstaged": unstaged_diffs.get(path, "")}
        for path in paths
    }
    dirty = bool(parsed["staged"] or parsed["unstaged"]
                 or parsed["untracked"] or parsed["conflicts"])
    return {
        "ok": True,
        "root": str(resolved),
        **parsed,
        "dirty": dirty,
        "hunks": hunks,
        "diffs": hunks,
        "staged_diff": cached.stdout or "",
        "unstaged_diff": work.stdout or "",
    }


# --- staging -------------------------------------------------------------------

def _safe_relative(root: Path, raw) -> str:
    """A project-relative POSIX path, or GitOpsError for anything unsafe."""
    try:
        safe = security.resolve_safe_path(root, raw)
        rel = safe.relative_to(root).as_posix()
    except (security.SecurityError, OSError, ValueError) as exc:
        raise GitOpsError(f"refusing unsafe path {raw!r}: {exc}") from exc
    if rel in ("", "."):
        raise GitOpsError(f"refusing the project root as a file path: {raw!r}")
    if rel == ".git" or rel.startswith(".git/"):
        raise GitOpsError("refusing to stage or unstage git internals (.git/).")
    return rel


def _safe_paths(root: Path, paths) -> list[str]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    rels: list[str] = []
    for raw in paths or []:
        rel = _safe_relative(root, raw)
        if rel not in rels:
            rels.append(rel)
    if not rels:
        raise GitOpsError("no file paths were given.")
    return rels


def stage_files(root, paths) -> dict:
    """Stage validated project-relative paths."""
    resolved = _require_repo(root)
    rels = _safe_paths(resolved, paths)
    proc = _git(resolved, "add", "--", *rels)
    if proc.returncode != 0:
        raise GitOpsError(_output(proc) or "`git add` failed.")
    return {"ok": True, "staged": rels}


def stage_all_files(root) -> dict:
    """Stage every change (including deletions) in the repository."""
    resolved = _require_repo(root)
    proc = _git(resolved, "add", "-A", "--", ".")
    if proc.returncode != 0:
        raise GitOpsError(_output(proc) or "`git add -A` failed.")
    return {"ok": True}


def unstage_files(root, paths) -> dict:
    """Unstage validated paths, keeping the working-tree changes."""
    resolved = _require_repo(root)
    rels = _safe_paths(resolved, paths)
    proc = _git(resolved, "restore", "--staged", "--", *rels)
    if proc.returncode != 0:
        # `git restore` is newer than the `git reset` fallback; an unborn
        # branch also needs the fallback's different semantics.
        proc = _git(resolved, "reset", "--", *rels)
    if proc.returncode != 0:
        raise GitOpsError(_output(proc) or "`git restore --staged` failed.")
    return {"ok": True, "unstaged": rels}


# --- secret pre-flight ---------------------------------------------------------

def scan_staged_secrets(root) -> list:
    """Secret findings in the staged diff (`git diff --cached`)."""
    resolved = _require_repo(root)
    proc = _git(resolved, "diff", "--cached", "--no-color", "--unified=0")
    if proc.returncode != 0:
        raise GitOpsError(_output(proc) or "`git diff --cached` failed.")
    return security.scan_text_for_secrets(
        proc.stdout or "", "<staged diff>")


def _describe_findings(findings, limit: int = 8) -> str:
    rows = [
        f"  - {finding.kind} at {finding.file}:{finding.line} "
        f"({finding.masked_preview})"
        for finding in findings[:limit]
    ]
    extra = len(findings) - len(rows)
    if extra > 0:
        rows.append(f"  - and {extra} more finding(s)")
    return "\n".join(rows)


def _secret_refusal(findings, action: str) -> GitOpsError:
    return GitOpsError(
        f"refusing to {action}: the diff contains possible secret material.\n"
        + _describe_findings(findings)
        + "\nRemove the secret from the change, or use the explicit "
          "allow-secrets override when you are sure it is not a real "
          "credential.")


# --- commit drafting -----------------------------------------------------------

def _scope(paths: list[str]) -> str:
    dirs = {str(Path(p).parent.as_posix()) for p in paths}
    dirs.discard(".")
    if len(dirs) == 1:
        return next(iter(dirs))
    return ""


def _offline_commit_message(status: dict) -> str:
    """Deterministic Conventional Commit subject from the staged change."""
    files = list(status.get("staged") or [])
    if not files:
        raise GitOpsError("nothing is staged; stage a file first.")
    diff = "\n".join(
        (block or {}).get("staged", "")
        for block in (status.get("hunks") or {}).values())
    is_new = "new file mode" in diff
    is_deleted = bool(re.search(r"^deleted file mode", diff, re.MULTILINE))
    kind = "docs" if all(Path(p).suffix.lower() in _DOC_EXTS for p in files) \
        else "test" if all("test" in Path(p).name.lower() for p in files) \
        else "feat" if is_new else "chore"
    scope = _scope(files)
    prefix = f"{kind}({scope})" if scope else kind
    if len(files) > 1:
        return f"{prefix}: update {len(files)} files"
    stem = Path(files[0]).stem or files[0]
    verb = "remove" if is_deleted else "add" if is_new else "update"
    return f"{prefix}: {verb} {stem}"


def _pick_conventional(text: str) -> str:
    for raw in str(text or "").replace("\r\n", "\n").splitlines():
        line = raw.strip().strip("`").strip()
        line = re.sub(r"^(?:[-*]|\d+[.)])\s+", "", line)
        if _CONVENTIONAL_RE.match(line):
            return line[:100]
    return ""


def _bounded_diff(status: dict, *, staged: bool = True) -> str:
    key = "staged" if staged else "unstaged"
    text = "\n".join(
        (block or {}).get(key, "")
        for block in (status.get("hunks") or {}).values())
    if len(text) > _MAX_DIFF_CHARS:
        text = text[:_MAX_DIFF_CHARS] + "\n[diff truncated]"
    return text


def draft_commit_message(root, client=None, cfg=None) -> str:
    """A concise Conventional Commit message for the staged change.

    `client` is an LLMClient-like object; when it is None (the `--no-model`
    path) or its call fails, the deterministic offline message is returned, so
    committing never depends on a running model server.
    """
    resolved = _require_repo(root)
    status = status_hunks(resolved)
    if not status["staged"]:
        raise GitOpsError("nothing is staged; stage a file first.")
    offline = _offline_commit_message(status)
    if client is None:
        return offline
    if cfg is not None:
        try:
            from . import roles
            client = roles.client_for(cfg, "summarizer", client)
        except Exception:
            pass
    if client is None:
        return offline
    files = "\n".join(f"- {path}" for path in status["staged"])
    messages = [
        {"role": "system", "content": (
            "You write one concise Conventional Commit subject line for the "
            "staged change. Reply with exactly one line, no code fence and no "
            "explanation: <type>(<scope>): <summary>. Types: feat, fix, docs, "
            "style, refactor, perf, test, build, ci, chore, revert. Use a "
            "scope only when it is obvious.")},
        {"role": "user", "content": (
            f"Branch: {status['branch'] or '(detached)'}\n"
            f"Staged files:\n{files}\n\n"
            f"Staged diff:\n{_bounded_diff(status)}")},
    ]
    try:
        raw = client.chat(messages, max_tokens=120)
    except Exception:
        return offline
    if isinstance(raw, dict):
        raw = (raw.get("message") or raw.get("subject")
               or raw.get("title") or "")
    return _pick_conventional(raw) or offline


# --- commits and branches ------------------------------------------------------

def create_commit(root, message, *, stage_all: bool = False,
                  allow_secrets: bool = False) -> dict:
    """Commit locally after a secret pre-flight; never pushes."""
    resolved = _require_repo(root)
    text = str(message or "").strip()
    if not text:
        raise GitOpsError("a commit message is required.")
    if stage_all:
        stage_all_files(resolved)
    findings = scan_staged_secrets(resolved)
    if findings and not allow_secrets:
        raise _secret_refusal(findings, "commit")
    proc = _git(resolved, "commit", "-m", text)
    if proc.returncode != 0:
        raise GitOpsError(_output(proc) or "`git commit` failed.")
    sha_proc = _git(resolved, "rev-parse", "HEAD")
    sha = (sha_proc.stdout or "").strip() if sha_proc.returncode == 0 else ""
    return {
        "ok": True,
        "commit": sha,
        "message": text,
        "secrets_allowed": bool(findings and allow_secrets),
        "findings": [finding.to_dict() for finding in findings],
    }


def _validate_branch(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        raise GitOpsError("a branch name is required.")
    if len(text) > 200:
        raise GitOpsError("the branch name is too long.")
    if (text.startswith("-") or text.startswith("/")
            or text.endswith("/") or text.endswith(".")
            or ".." in text or "//" in text or "@{" in text
            or not _BRANCH_RE.fullmatch(text)):
        raise GitOpsError(
            f"invalid branch name {text!r}; use letters, digits, '.', '_', "
            "'/' or '-', without '..' or a leading '-'.")
    return text


def create_branch(root, name, *, checkout: bool = True) -> dict:
    """Create `name`, switching to it with `checkout=True`."""
    resolved = _require_repo(root)
    branch = _validate_branch(name)
    args = ["checkout", "-b", branch] if checkout else ["branch", branch]
    proc = _git(resolved, *args)
    if proc.returncode != 0:
        raise GitOpsError(_output(proc) or f"could not create branch {branch!r}.")
    return {"ok": True, "branch": branch, "checked_out": bool(checkout)}


# --- pull requests -------------------------------------------------------------

def _require_gh() -> None:
    if not shutil.which("gh"):
        raise GitOpsError(
            "the GitHub CLI (`gh`) is not installed or not on PATH; install it "
            "from https://cli.github.com to open a pull request.")


def _branch_context(root: Path, base: str) -> dict:
    base_name = _validate_branch(base)
    status = status_hunks(root)
    branch = status["branch"]
    if not branch:
        raise GitOpsError("this repository has no current branch (detached HEAD).")
    check = _git(root, "rev-parse", "--verify", f"{base_name}^{{commit}}")
    if check.returncode != 0:
        raise GitOpsError(f"base branch {base_name!r} was not found.")
    diff_proc = _git(root, "diff", "--no-color", f"{base_name}...HEAD")
    if diff_proc.returncode != 0:
        raise GitOpsError(_output(diff_proc) or "could not read the branch diff.")
    log_proc = _git(root, "log", "--pretty=format:%h %s",
                    f"{base_name}..HEAD")
    names_proc = _git(root, "diff", "--name-status", f"{base_name}...HEAD")
    commits = [line for line in (log_proc.stdout or "").splitlines()
               if line.strip()]
    files = [line for line in (names_proc.stdout or "").splitlines()
             if line.strip()]
    return {
        "branch": branch,
        "base": base_name,
        "diff": diff_proc.stdout or "",
        "commits": commits,
        "files": files,
    }


def _checkpoint_summaries(root: Path, limit: int = 5) -> list[str]:
    try:
        from . import checkpoints
        items = checkpoints.list_checkpoints(root)[:limit]
    except Exception:
        return []
    out: list[str] = []
    for item in items:
        label = str(item.get("label") or "").strip()
        if label:
            out.append(f"{item.get('id', '')} {label}".strip())
    return out


def _offline_pr_draft(context: dict, checkpoint_labels: list[str]) -> tuple[str, str]:
    commits = context["commits"]
    branch = context["branch"]
    title = commits[0].split(" ", 1)[1] if commits and " " in commits[0] \
        else branch.replace("-", " ").replace("_", " ").strip().capitalize()
    title = (title or "Update branch")[:72]
    lines = ["## Summary", ""]
    if commits:
        lines.extend(f"- {line.split(' ', 1)[1] if ' ' in line else line}"
                     for line in commits[:10])
    else:
        lines.append("- No new commits on this branch against the base.")
    lines.extend(["", "## Changes", ""])
    if context["files"]:
        lines.extend(f"- {line}" for line in context["files"][:20])
    else:
        lines.append("- No file changes were detected.")
    if checkpoint_labels:
        lines.extend(["", "## Checkpoints", ""])
        lines.extend(f"- {label}" for label in checkpoint_labels)
    return title, "\n".join(lines)


def _parse_pr_reply(text: str) -> tuple[str, str]:
    title = ""
    body_lines: list[str] = []
    lines = str(text or "").replace("\r\n", "\n").split("\n")
    for index, raw in enumerate(lines):
        line = raw.strip()
        low = line.lower()
        if low.startswith("title:") and not title:
            title = line.split(":", 1)[1].strip().strip("`\"")
        elif low in ("body:", "description:"):
            body_lines = [part.rstrip() for part in lines[index + 1:]]
            break
    if not title:
        title = next((line.strip().strip("`") for line in lines
                      if line.strip()), "")
    body = "\n".join(body_lines).strip()
    if body.startswith("```"):
        body = body.strip("`").strip()
    return title[:120], body


def draft_pr(root, client=None, cfg=None, *, base: str = "main") -> dict:
    """A grounded PR title/body for the current branch against `base`."""
    resolved = _require_repo(root)
    _require_gh()
    context = _branch_context(resolved, base)
    findings = security.scan_text_for_secrets(
        context["diff"], f"<branch diff vs {context['base']}>")
    if findings:
        raise _secret_refusal(findings, "draft a pull request")
    checkpoints = _checkpoint_summaries(resolved)
    offline_title, offline_body = _offline_pr_draft(context, checkpoints)
    if client is None:
        return {
            "ok": True,
            "title": offline_title,
            "body": offline_body,
            "base": context["base"],
            "branch": context["branch"],
            "commits": context["commits"],
            "files": context["files"],
        }
    if cfg is not None:
        try:
            from . import roles
            client = roles.client_for(cfg, "summarizer", client)
        except Exception:
            pass
    if client is None:
        title, body = offline_title, offline_body
    else:
        diff = context["diff"]
        if len(diff) > _MAX_DIFF_CHARS:
            diff = diff[:_MAX_DIFF_CHARS] + "\n[diff truncated]"
        messages = [
            {"role": "system", "content": (
                "You write a pull-request title and body from the repository "
                "evidence provided. Do not invent files or behaviour. Reply "
                "exactly as:\nTITLE: <one line, at most 72 characters>\n"
                "BODY:\n<markdown with a short Summary and a Changes list>")},
            {"role": "user", "content": (
                f"Branch: {context['branch']}\nBase: {context['base']}\n"
                f"Commits:\n" + "\n".join(context["commits"][:20])
                + "\n\nChanged files:\n" + "\n".join(context["files"][:40])
                + ("\n\nRecent checkpoints:\n" + "\n".join(checkpoints)
                   if checkpoints else "")
                + f"\n\nBranch diff:\n{diff}")},
        ]
        try:
            raw = client.chat(messages, max_tokens=500)
        except Exception:
            title, body = offline_title, offline_body
        else:
            title, body = _parse_pr_reply(raw)
            if not title or not body:
                title, body = offline_title, offline_body
    return {
        "ok": True,
        "title": title,
        "body": body,
        "base": context["base"],
        "branch": context["branch"],
        "commits": context["commits"],
        "files": context["files"],
    }


def create_pr(root, title, body, *, base: str = "main",
              draft: bool = False) -> dict:
    """Open a pull request with `gh`; it never pushes the branch itself."""
    resolved = _require_repo(root)
    _require_gh()
    clean_title = str(title or "").strip()
    clean_body = str(body or "").strip()
    if not clean_title:
        raise GitOpsError("a pull-request title is required.")
    base_name = _validate_branch(base)
    context = _branch_context(resolved, base_name)
    if context["branch"] == base_name:
        raise GitOpsError(
            "the current branch is the base branch; create a feature branch "
            "before opening a pull request.")
    findings = security.scan_text_for_secrets(
        context["diff"], f"<branch diff vs {base_name}>")
    if findings:
        raise _secret_refusal(findings, "open a pull request")
    args = ["pr", "create", "--base", base_name,
            "--title", clean_title, "--body", clean_body]
    if draft:
        args.append("--draft")
    proc = _gh(resolved, *args)
    if proc.returncode != 0:
        raise GitOpsError(_output(proc) or "`gh pr create` failed.")
    output = _output(proc)
    url = next((word for word in output.split() if word.startswith("http")), "")
    return {
        "ok": True,
        "url": url,
        "output": output,
        "title": clean_title,
        "base": base_name,
        "draft": bool(draft),
    }
