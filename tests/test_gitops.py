"""Guided git workflows against real temporary repositories.

Every test builds a throwaway `git init` repo under `tmp_path`; nothing here
touches the developer's checkout, and no test pushes anywhere.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import gitops  # noqa: E402
from pedacito.gui import core, server  # noqa: E402

FAKE_GITHUB_TOKEN = "ghp_" + "a" * 36


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(root), capture_output=True, text=True,
        encoding="utf-8", errors="replace", shell=False)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=str(root),
                   check=True, capture_output=True, text=True,
                   encoding="utf-8", errors="replace")
    assert _git(root, "config", "user.email", "test@example.com").returncode == 0
    assert _git(root, "config", "user.name", "Pedacito Test").returncode == 0
    (root / "README.md").write_text("# demo\n", encoding="utf-8")
    assert _git(root, "add", "README.md").returncode == 0
    assert _git(root, "commit", "-m", "chore: initial commit").returncode == 0
    return root


def test_status_hunks_reports_branch_and_per_file_diffs(tmp_path):
    root = _repo(tmp_path)
    (root / "a.txt").write_text("one\n", encoding="utf-8")
    (root / "b.txt").write_text("two\n", encoding="utf-8")
    gitops.stage_files(root, ["a.txt"])
    (root / "a.txt").write_text("one\ntwo\n", encoding="utf-8")

    status = gitops.status_hunks(root)
    assert status["ok"] is True
    assert status["branch"] == "main"
    assert status["dirty"] is True
    assert "a.txt" in status["staged"]
    assert "a.txt" in status["unstaged"]
    assert "b.txt" in status["untracked"]
    assert "+one" in status["hunks"]["a.txt"]["staged"]
    assert "+two" in status["hunks"]["a.txt"]["unstaged"]


def test_stage_and_unstage_round_trip(tmp_path):
    root = _repo(tmp_path)
    (root / "new.txt").write_text("hello\n", encoding="utf-8")

    staged = gitops.stage_files(root, ["new.txt"])
    assert staged["staged"] == ["new.txt"]
    assert "new.txt" in gitops.status_hunks(root)["staged"]

    unstaged = gitops.unstage_files(root, ["new.txt"])
    assert unstaged["unstaged"] == ["new.txt"]
    after = gitops.status_hunks(root)
    assert "new.txt" not in after["staged"]
    assert "new.txt" in after["untracked"]


def test_unsafe_paths_are_refused(tmp_path):
    root = _repo(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("nope\n", encoding="utf-8")
    with pytest.raises(gitops.GitOpsError, match="unsafe path"):
        gitops.stage_files(root, ["../outside.txt"])
    with pytest.raises(gitops.GitOpsError, match="unsafe path"):
        gitops.unstage_files(root, ["../outside.txt"])


def test_staged_secret_blocks_commit_unless_explicitly_allowed(tmp_path):
    root = _repo(tmp_path)
    (root / "leak.py").write_text(
        f'TOKEN = "{FAKE_GITHUB_TOKEN}"\n', encoding="utf-8")
    gitops.stage_files(root, ["leak.py"])

    findings = gitops.scan_staged_secrets(root)
    assert findings
    assert findings[0].kind == "github_token"
    assert FAKE_GITHUB_TOKEN not in str(findings[0].to_dict())

    with pytest.raises(gitops.GitOpsError, match="secret"):
        gitops.create_commit(root, "feat: add leak")
    assert _git(root, "log", "-1", "--pretty=format:%s").stdout.strip() \
        == "chore: initial commit"

    result = gitops.create_commit(
        root, "feat: add leak", allow_secrets=True)
    assert result["ok"] is True
    assert len(result["commit"]) == 40
    assert result["secrets_allowed"] is True


def test_commit_creates_a_local_commit_and_never_pushes(tmp_path):
    root = _repo(tmp_path)
    (root / "app.py").write_text("print('hi')\n", encoding="utf-8")
    gitops.stage_files(root, ["app.py"])

    message = gitops.draft_commit_message(root)
    assert message.startswith("feat:") or message.startswith("chore:")

    result = gitops.create_commit(root, "feat: add app")
    assert result["ok"] is True and result["message"] == "feat: add app"
    assert _git(root, "log", "-1", "--pretty=format:%s").stdout.strip() \
        == "feat: add app"
    # A local repository has no remote here, so status/hunks still work after.
    assert gitops.status_hunks(root)["dirty"] is False


def test_commit_with_stage_all_picks_up_untracked_files(tmp_path):
    root = _repo(tmp_path)
    (root / "fresh.txt").write_text("fresh\n", encoding="utf-8")
    result = gitops.create_commit(
        root, "feat: add fresh file", stage_all=True)
    assert result["ok"] is True
    names = _git(root, "show", "--name-only", "--pretty=format:",
                 "HEAD").stdout.split()
    assert "fresh.txt" in names


def test_draft_commit_message_uses_a_client_then_falls_back(tmp_path):
    root = _repo(tmp_path)
    (root / "client.py").write_text("x = 1\n", encoding="utf-8")
    gitops.stage_files(root, ["client.py"])

    class Client:
        def __init__(self, reply):
            self.reply = reply

        def chat(self, messages, max_tokens=0):
            assert messages and messages[0]["role"] == "system"
            return self.reply

    assert gitops.draft_commit_message(
        root, client=Client("feat: add client module")) \
        == "feat: add client module"
    # A non-conventional or broken reply falls back to the offline subject.
    assert gitops.draft_commit_message(
        root, client=Client("Here is a commit message")) \
        .startswith(("feat:", "chore:", "docs:", "test:"))


def test_draft_commit_message_disables_thinking_and_retries_when_empty(tmp_path):
    root = _repo(tmp_path)
    (root / "think.py").write_text("x = 1\n", encoding="utf-8")
    gitops.stage_files(root, ["think.py"])

    class Client:
        def __init__(self):
            self.calls = []

        def chat(self, messages, max_tokens=0, disable_thinking=False):
            self.calls.append((max_tokens, disable_thinking))
            return "" if max_tokens < 512 else "feat: think module"

    client = Client()
    assert gitops.draft_commit_message(root, client=client) == "feat: think module"
    assert client.calls == [(120, True), (512, True)]


def test_draft_commit_message_falls_back_on_client_without_thinking_kwarg(tmp_path):
    root = _repo(tmp_path)
    (root / "legacy.py").write_text("x = 1\n", encoding="utf-8")
    gitops.stage_files(root, ["legacy.py"])

    class Legacy:
        def __init__(self):
            self.calls = []

        def chat(self, messages, max_tokens=0):
            self.calls.append(max_tokens)
            return "fix: legacy reply"

    client = Legacy()
    assert gitops.draft_commit_message(root, client=client) == "fix: legacy reply"
    assert client.calls == [120]


def test_create_branch_switches_and_invalid_names_are_rejected(tmp_path):
    root = _repo(tmp_path)
    result = gitops.create_branch(root, "feature/gitops")
    assert result == {"ok": True, "branch": "feature/gitops",
                      "checked_out": True}
    assert gitops.status_hunks(root)["branch"] == "feature/gitops"

    for bad in ("../escape", "-force", "has space", "a..b", "trailing/"):
        with pytest.raises(gitops.GitOpsError):
            gitops.create_branch(root, bad)


def test_missing_gh_is_a_clear_error(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    gitops.create_branch(root, "feature/no-gh")
    real_which = shutil.which

    def fake_which(name):
        return None if name == "gh" else real_which(name)

    monkeypatch.setattr(gitops.shutil, "which", fake_which)
    with pytest.raises(gitops.GitOpsError, match="GitHub CLI"):
        gitops.draft_pr(root)
    with pytest.raises(gitops.GitOpsError, match="GitHub CLI"):
        gitops.create_pr(root, "feat: thing", "body")


def test_create_pr_uses_argv_only_gh_call(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    gitops.create_branch(root, "feature/pr")
    (root / "pr.txt").write_text("change\n", encoding="utf-8")
    gitops.stage_files(root, ["pr.txt"])
    gitops.create_commit(root, "feat: add pr file")

    real_which = shutil.which
    monkeypatch.setattr(
        gitops.shutil, "which",
        lambda name: "/usr/bin/gh" if name == "gh" else real_which(name))
    calls = []

    class Proc:
        returncode = 0
        stdout = "https://github.com/acme/demo/pull/7\n"
        stderr = ""

    def fake_gh(root_arg, *args, timeout=60):
        calls.append((str(root_arg), list(args)))
        return Proc()

    monkeypatch.setattr(gitops, "_gh", fake_gh)
    result = gitops.create_pr(
        root, "feat: add pr file", "A short body.", base="main", draft=True)

    assert result["ok"] is True
    assert result["url"] == "https://github.com/acme/demo/pull/7"
    assert calls == [(
        str(root),
        ["pr", "create", "--base", "main", "--title", "feat: add pr file",
         "--body", "A short body.", "--draft"],
    )]


def test_draft_pr_is_grounded_in_commits_and_files(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    gitops.create_branch(root, "feature/draft")
    (root / "draft.txt").write_text("change\n", encoding="utf-8")
    gitops.stage_files(root, ["draft.txt"])
    gitops.create_commit(root, "feat: add draft file")

    real_which = shutil.which
    monkeypatch.setattr(
        gitops.shutil, "which",
        lambda name: "/usr/bin/gh" if name == "gh" else real_which(name))
    draft = gitops.draft_pr(root, base="main")
    assert "draft file" in draft["title"].lower()
    assert "draft.txt" in draft["body"]
    assert draft["branch"] == "feature/draft"


def test_draft_pr_disables_thinking_and_falls_back_without_kwarg(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    gitops.create_branch(root, "feature/think-pr")
    (root / "pr_think.txt").write_text("change\n", encoding="utf-8")
    gitops.stage_files(root, ["pr_think.txt"])
    gitops.create_commit(root, "feat: add pr think file")

    real_which = shutil.which
    monkeypatch.setattr(
        gitops.shutil, "which",
        lambda name: "/usr/bin/gh" if name == "gh" else real_which(name))

    class Thinking:
        def __init__(self):
            self.kwargs = []

        def chat(self, messages, max_tokens=0, disable_thinking=False):
            self.kwargs.append((max_tokens, disable_thinking))
            return "TITLE: feat: model title\nBODY:\nModel body."

    class Legacy:
        def chat(self, messages, max_tokens=0):
            return "TITLE: feat: legacy title\nBODY:\nLegacy body."

    thinking = Thinking()
    draft = gitops.draft_pr(root, client=thinking, base="main")
    assert thinking.kwargs == [(500, True)]
    assert draft["title"] == "feat: model title"

    draft = gitops.draft_pr(root, client=Legacy(), base="main")
    assert draft["title"] == "feat: legacy title"


# --- GUI endpoints -------------------------------------------------------------

def test_gui_git_status_stage_and_commit_endpoints(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    monkeypatch.setattr(core.STATE, "root", root)
    monkeypatch.setattr(core.STATE, "cfg", None)
    (root / "gui.txt").write_text("hello\n", encoding="utf-8")

    code, payload = server.GET_ROUTES["/api/git/status"]({})
    assert code == 200 and payload["ok"] is True
    assert payload["branch"] == "main"
    assert "gui.txt" in payload["untracked"]

    code, payload = server.POST_ROUTES["/api/git/stage"](
        {}, {"paths": ["gui.txt"]})
    assert code == 200 and payload["ok"] is True
    assert "gui.txt" in payload["staged"]

    code, payload = server.POST_ROUTES["/api/git/commit"](
        {}, {"message": "feat: add gui file"})
    assert code == 200 and payload["ok"] is True
    assert payload["commit"]
    assert payload["dirty"] is False

    (root / "next.txt").write_text("next\n", encoding="utf-8")
    server.POST_ROUTES["/api/git/stage"]({}, {"paths": ["next.txt"]})
    code, payload = server.POST_ROUTES["/api/git/commit"](
        {}, {"draft_only": True})
    assert code == 200 and payload["ok"] is True
    assert payload["message"].startswith(
        ("feat:", "fix:", "docs:", "test:", "chore:"))


def test_gui_git_commit_refuses_a_staged_secret(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    monkeypatch.setattr(core.STATE, "root", root)
    monkeypatch.setattr(core.STATE, "cfg", None)
    (root / "leak.txt").write_text(
        f"token={FAKE_GITHUB_TOKEN}\n", encoding="utf-8")
    server.POST_ROUTES["/api/git/stage"]({}, {"paths": ["leak.txt"]})

    code, payload = server.POST_ROUTES["/api/git/commit"](
        {}, {"message": "feat: oops"})
    assert code == 400 and payload["ok"] is False
    assert "secret" in payload["detail"].lower()
    assert _git(root, "log", "-1", "--pretty=format:%s").stdout.strip() \
        == "chore: initial commit"

    code, payload = server.POST_ROUTES["/api/git/commit"](
        {}, {"message": "feat: oops", "allow_secrets": True})
    assert code == 200 and payload["ok"] is True
