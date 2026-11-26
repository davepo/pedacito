"""WP-B2 (Bug-Hunt Remediation Wave 2): scaffold entry points/README/git,
worktree dirty handling and branch cleanup, malformed Windows filenames,
checkpoint counts, --json/--review, and PowerShell-friendly output.

Every test is offline and uses throwaway temp directories/repos.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import (  # noqa: E402
    checkpoints, cli, config as config_mod, events, gitops, scaffold, worktree,
)
from pedacito.config import Config  # noqa: E402
from pedacito import index as index_mod  # noqa: E402


pytestmark = pytest.mark.skipif(not shutil.which("git"),
                                reason="git is not installed")


def _git(root: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=str(root), check=False,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    return (proc.stdout or "") + (proc.stderr or "")


def _repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "a@b.c"], cwd=tmp_path,
                   check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path,
                   check=True)
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "a.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    return tmp_path


def _branches(root: Path) -> list[str]:
    return [ln.strip().lstrip("*+ ").strip()
            for ln in _git(root, "branch", "--list").splitlines()
            if ln.strip()]


def _create(root: Path, name: str, *extra: str) -> int:
    return cli.main(["worktree", "create", name, "--root", str(root),
                     *extra])


def _remove(root: Path, name: str, *extra: str) -> int:
    return cli.main(["worktree", "remove", name, "--root", str(root),
                     *extra])


# ============================================================================
# BUG-05 -- a generated main.py gets the __main__ guard
# ============================================================================

def test_ensure_entrypoint_adds_guard_and_python_main_runs(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "main.py").write_text(
        "def main():\n    print('datetool ok')\n", encoding="utf-8")

    assert scaffold._ensure_python_cli_entrypoint(root) is True
    text = (root / "main.py").read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in text
    assert "    main()" in text

    proc = subprocess.run([sys.executable, "main.py"], cwd=str(root),
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    assert proc.returncode == 0
    assert "datetool ok" in proc.stdout

    # Idempotent: a second pass neither duplicates the guard nor reports work.
    assert scaffold._ensure_python_cli_entrypoint(root) is False
    assert text.count('if __name__ == "__main__":') == 1


def test_ensure_entrypoint_skips_existing_guard_syntax_errors_and_helpers(
        tmp_path):
    guarded = tmp_path / "guarded.py"
    guarded.write_text(
        'def main():\n    return 0\n\n\n'
        'if __name__ == "__main__":\n    raise SystemExit(main())\n',
        encoding="utf-8")
    broken = tmp_path / "main.py"
    broken.write_text("def main(:\n", encoding="utf-8")
    helper = tmp_path / "helper.py"
    helper.write_text("def value():\n    return 1\n", encoding="utf-8")

    assert scaffold._ensure_python_cli_entrypoint(tmp_path) is False
    assert (tmp_path / "main.py").read_text(encoding="utf-8") == "def main(:\n"
    assert 'if __name__ == "__main__":' in guarded.read_text(encoding="utf-8")
    assert "__main__" not in helper.read_text(encoding="utf-8")


# ============================================================================
# BUG-17 -- Windows-friendly README commands and a real git repo
# ============================================================================

def test_readme_commands_are_rewritten_to_plain_python(tmp_path):
    (tmp_path / "README.md").write_text(
        "python3 main.py\npython3 -m pytest -q\npython3.12 --version\n",
        encoding="utf-8")
    assert scaffold._normalize_python_commands(tmp_path, "README.md") is True
    text = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "python main.py" in text
    assert "python -m pytest -q" in text
    assert "python3.12 --version" in text       # a version, not a command


def test_skeleton_readme_uses_python_main_not_python3(tmp_path):
    plan = scaffold.Plan(
        "Demo", "a tiny cli",
        [scaffold.PlannedFile("main.py", "entry point")],
        spec=scaffold.Spec("a tiny cli"))
    scaffold.apply_skeleton(tmp_path, plan, dry_run=False, verbose=False)
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "python main.py" in readme
    assert "python3" not in readme


def test_run_new_apply_initialises_a_git_repository(tmp_path, monkeypatch):
    plan = scaffold.Plan(
        "Demo", "a tiny cli",
        [scaffold.PlannedFile("main.py", "entry point")],
        spec=scaffold.Spec("a tiny cli"))
    monkeypatch.setattr(scaffold, "build_plan", lambda *a, **k: plan)
    monkeypatch.setattr(scaffold, "build_interface", lambda *a, **k: {})
    monkeypatch.setattr(scaffold, "build_project", lambda *a, **k: {
        "done": [], "noop": [], "failed": [], "skipped": []})
    monkeypatch.setattr(scaffold, "build_docs", lambda *a, **k: {
        "done": [], "noop": [], "failed": [], "skipped": []})
    monkeypatch.setattr(scaffold, "fix_import_errors", lambda *a, **k: 0)
    monkeypatch.setattr(scaffold, "verify_and_fix", lambda *a, **k: (True, ""))
    monkeypatch.setattr(scaffold, "_web_contract_gate", lambda *a, **k: [])

    cfg = config_mod.load(require_hardware=False)
    cfg.backup = False
    cfg.log_sessions = False
    root = tmp_path / "fresh"
    code = scaffold.run_new(root, cfg, None,
                            scaffold.Spec("a tiny cli"), apply=True,
                            verbose=False)
    assert code == 0
    assert (root / ".git").exists()


# ============================================================================
# BUG-11 -- dirty trees are refused by default, carried or stashed on request
# ============================================================================

def test_worktree_create_refuses_dirty_tree_by_default(tmp_path, capsys):
    root = _repo(tmp_path)
    (root / "a.txt").write_text("dirty\n", encoding="utf-8")
    (root / "new.txt").write_text("untracked\n", encoding="utf-8")

    code = _create(root, "qa-dirty")
    captured = capsys.readouterr()
    assert code == 2
    assert "uncommitted" in captured.err
    assert "a.txt" in captured.err and "new.txt" in captured.err
    assert "carry_over" in captured.err and "stash" in captured.err
    assert not (root / ".pedacito" / "worktrees" / "qa-dirty").exists()
    assert not (root / ".pedacito" / "worktrees" / "qa-dirty" / ".git").exists()


def test_worktree_create_carry_over_copies_changes(tmp_path, capsys):
    root = _repo(tmp_path)
    (root / "a.txt").write_text("carried\n", encoding="utf-8")
    (root / "new.txt").write_text("also carried\n", encoding="utf-8")

    code = _create(root, "qa-carry", "--dirty", "carry_over")
    out = capsys.readouterr().out
    assert code == 0
    assert "Carried 2 uncommitted file(s)" in out
    dest = root / ".pedacito" / "worktrees" / "qa-carry"
    assert (dest / "a.txt").read_text(encoding="utf-8") == "carried\n"
    assert (dest / "new.txt").read_text(encoding="utf-8") == "also carried\n"
    # The source checkout keeps its uncommitted work.
    assert (root / "a.txt").read_text(encoding="utf-8") == "carried\n"


def test_worktree_create_stash_cleans_root_and_uses_head(tmp_path, capsys):
    root = _repo(tmp_path)
    (root / "a.txt").write_text("stashed\n", encoding="utf-8")

    code = _create(root, "qa-stash", "--dirty", "stash")
    out = capsys.readouterr().out
    assert code == 0
    assert "Stashed 1 uncommitted file(s)" in out
    assert not worktree.is_dirty(root)
    dest = root / ".pedacito" / "worktrees" / "qa-stash"
    assert (dest / "a.txt").read_text(encoding="utf-8") == "hello\n"


# ============================================================================
# BUG-16 -- removing a worktree also removes its branch
# ============================================================================

def test_worktree_remove_deletes_the_branch_by_default(tmp_path, capsys):
    root = _repo(tmp_path)
    assert _create(root, "qa-wt") == 0
    assert "pedacito/qa-wt" in _branches(root)
    capsys.readouterr()

    assert _remove(root, "qa-wt") == 0
    out = capsys.readouterr().out
    assert "Removed worktree qa-wt" in out
    assert "Deleted branch pedacito/qa-wt" in out
    assert "pedacito/qa-wt" not in _branches(root)

    # Recreating starts from HEAD, not from a stale branch tip.
    clean = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
    assert clean.stdout.strip() == ""
    assert _create(root, "qa-wt") == 0
    assert "pedacito/qa-wt" in _branches(root)


def test_worktree_remove_keep_branch_flag_preserves_it(tmp_path, capsys):
    root = _repo(tmp_path)
    assert _create(root, "qa-keep") == 0
    capsys.readouterr()

    assert _remove(root, "qa-keep", "--keep-branch") == 0
    out = capsys.readouterr().out
    assert "Kept branch pedacito/qa-keep" in out
    assert "pedacito/qa-keep" in _branches(root)


# ============================================================================
# BUG-13 -- malformed Windows filenames are reported cleanly
# ============================================================================

def _trailing_space_file(root: Path, name: str = ".env ") -> Path:
    """Create an NTFS name Win32 normally strips, through the \\\\?\\ prefix."""
    raw = "\\\\?\\" + str(root / name)
    with open(raw, "w", encoding="utf-8") as fh:
        fh.write("TOKEN=not-a-real-secret\n")
    return root / name


@pytest.mark.skipif(os.name != "nt", reason="NTFS filename rule")
def test_commit_all_reports_trailing_space_filename_cleanly(tmp_path, capsys):
    root = _repo(tmp_path)
    _trailing_space_file(root)
    assert ".env " in _git(root, "status", "--porcelain=v1")

    code = cli.main(["commit", str(root), "--all", "--no-model",
                     "-m", "test: stage", "-y"])
    err = capsys.readouterr().err
    assert code == 1
    assert "malformed Windows filename" in err
    assert ".env" in err
    assert "fatal: adding files failed" not in err
    assert "Traceback" not in err


def test_stage_all_parses_the_offending_path_from_git_error():
    err = ('error: open(".env "): No such file or directory\n'
           'error: unable to index file \'.env \'\n'
           'fatal: adding files failed\n')
    if os.name != "nt":
        pytest.skip("NTFS filename rule")
    assert gitops._paths_from_git_error(err) == [".env "]
    message = str(gitops._malformed_stage_error([".env "]))
    assert "'.env '" in message
    assert "trailing space" in message


# ============================================================================
# BUG-14 -- snapshot vs changed file counts are separate and accurate
# ============================================================================

def test_checkpoint_counts_separate_snapshot_and_changes(tmp_path):
    (tmp_path / "a.py").write_text("A = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("B = 1\n", encoding="utf-8")
    first = checkpoints.create_checkpoint(tmp_path, "first")
    assert first["changed_file_count"] == 2
    assert first["snapshot_file_count"] == 2

    (tmp_path / "a.py").write_text("A = 2\n", encoding="utf-8")
    (tmp_path / "c.py").write_text("C = 1\n", encoding="utf-8")
    second = checkpoints.create_checkpoint(tmp_path, "second")
    assert second["files_changed"] == ["a.py", "c.py"]
    assert second["changed_file_count"] == 2
    assert second["snapshot_file_count"] == 3
    assert second["file_count"] == second["changed_file_count"]


def test_checkpoints_json_exposes_both_counts(tmp_path, capsys):
    (tmp_path / "a.py").write_text("A = 1\n", encoding="utf-8")
    checkpoints.create_checkpoint(tmp_path, "first")
    (tmp_path / "a.py").write_text("A = 2\n", encoding="utf-8")
    checkpoints.create_checkpoint(tmp_path, "second")

    code = cli._cmd_checkpoints(SimpleNamespace(
        root=str(tmp_path), json=True, limit=0, config=None, profile=None))
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    latest = payload["checkpoints"][-1]
    assert latest["changed_file_count"] == 1
    assert latest["snapshot_file_count"] == 1
    assert latest["file_count"] == 1
    assert latest["files_changed"] == ["a.py"]


# ============================================================================
# BUG-15 -- explicit --review + --json warns; plain --json stays quiet
# ============================================================================

class _StubClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.actions = 0
        self.calls = 0
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0, "reasoning_tokens": 0,
                      "cached_tokens": 0, "cache_write_tokens": 0,
                      "seconds": 0.0, "cost": 0.0}

    def check(self, warn: bool = True):
        return ""

    def warmup(self):
        return None

    def cancel(self):
        return None

    def chat_json(self, messages, schema, max_tokens=0, **kw):
        action = ({"action": "read", "file": "app.py", "pattern": "",
                   "symbol": "", "start": 1, "end": 200, "content": "",
                   "reasoning": "read it"}
                  if self.actions == 0 else
                  {"action": "done", "file": "", "pattern": "", "symbol": "",
                   "start": 0, "end": 0, "content": "", "reasoning": "done"})
        self.actions += 1
        self._record()
        return action

    def chat(self, messages, max_tokens, schema=None, stream=False, **kw):
        self._record()
        return "No changes: nothing to do.\n"

    def _record(self) -> None:
        self.calls += 1
        events.note_call({
            "site": "answer", "prompt_tokens": 10, "completion_tokens": 5,
            "total_tokens": 15, "reasoning_tokens": 0, "cached_tokens": 0,
            "cache_write_tokens": 0, "seconds": 0.01, "tok_per_sec": 80.0,
            "cost": 0.0, "model": self.cfg.model, "provider": "stub",
            "finish": "stop"})


def _ask_json_run(tmp_path, monkeypatch, *extra):
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    (proj / "app.py").write_text("LIMIT = 10\n", encoding="utf-8")
    cfg_file = tmp_path / "user.toml"
    cfg_file.write_text("vram_gb = 8\nrefresh_summaries = 0\n"
                        "log_sessions = false\nshow_telemetry = false\n",
                        encoding="utf-8")
    cfg = Config()
    cfg.base_url = "http://127.0.0.1:9/v1"
    cfg.model = "stub-model"
    cfg.refresh_summaries = 0
    idx = index_mod.build([str(proj)], cfg, client=None, summarise=False,
                          root=proj)
    monkeypatch.setattr(cli, "LLMClient", lambda run_cfg: _StubClient(run_cfg))
    monkeypatch.setattr(cli, "_get_index", lambda *a, **k: idx)
    return cli.main(["ask", str(proj), "-t", "raise LIMIT",
                     "--config", str(cfg_file), "--model", "stub-model",
                     "--url", "http://127.0.0.1:9/v1", "--json", *extra])


def test_explicit_review_with_json_prints_a_clear_stderr_note(
        tmp_path, monkeypatch, capsys):
    assert _ask_json_run(tmp_path, monkeypatch, "--review") == 0
    captured = capsys.readouterr()
    assert ("Note: --json disables interactive --review" in captured.err)
    assert "omit --json to open the review page" in captured.err
    assert json.loads(captured.out)["ok"] is True      # stdout stays JSON


def test_plain_json_never_prints_the_review_note(tmp_path, monkeypatch,
                                                 capsys):
    assert _ask_json_run(tmp_path, monkeypatch) == 0
    captured = capsys.readouterr()
    assert "Note: --json disables interactive --review" not in captured.err
    assert json.loads(captured.out)["ok"] is True


# ============================================================================
# BUG-18 -- informational lines go to stdout in human mode
# ============================================================================

def test_how_offline_info_tag_is_on_stdout_not_stderr(tmp_path, monkeypatch,
                                                     capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_mod, "USER_CONFIG", tmp_path / "missing.toml")
    for key in list(os.environ):
        if key.startswith("PEDACITO_"):
            monkeypatch.delenv(key, raising=False)

    assert cli.main(["how", "--offline", "how do I import models?"]) == 0
    captured = capsys.readouterr()
    assert "[how: deterministic reference]" in captured.out
    assert "deterministic reference" not in captured.err


def test_worktree_create_and_remove_info_lines_are_on_stdout(tmp_path, capsys):
    root = _repo(tmp_path)
    assert _create(root, "ps-info") == 0
    captured = capsys.readouterr()
    assert "Run there with" in captured.out
    assert captured.err == ""

    assert _remove(root, "ps-info") == 0
    captured = capsys.readouterr()
    assert "Removed worktree ps-info" in captured.out
    assert "Deleted branch pedacito/ps-info" in captured.out
    assert captured.err == ""
