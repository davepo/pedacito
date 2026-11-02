"""`pedacito reset`: wiping a project's .pedacito/ state, keeping code."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import cli, workspace
from pedacito import config as config_mod


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An isolated user-config directory for the --reindex path."""
    monkeypatch.setattr(config_mod, "USER_CONFIG",
                        tmp_path / "home" / "config.toml")
    (tmp_path / "home").mkdir()
    return tmp_path / "home"


def _project(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "p"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "core.py").write_text("def core():\n    return 1\n",
                                          encoding="utf-8")
    (root / ".pedacito.toml").write_text('auto_verify = "true"\n',
                                         encoding="utf-8")
    (root / ".pedacitoignore").write_text(".pedacito/\n", encoding="utf-8")
    d = workspace.state_dir(root)
    files = {
        "index.json": "{}",
        "summaries.json": "{}",
        "log.txt": "log\n",
        "todo.md": "- [ ] task\n",
        "memory.md": "remember this\n",
        "instructions.md": "be brief\n",
        "design.json": "{}",
        "plans/20260101-000000.md": "- [ ] task\n",
        "rules/r.md": "rule\n",
        "skills/demo/SKILL.md": "skill\n",
        "prompts/answer.md": "override\n",
        "sessions/s.md": "session\n",
        "events/e.jsonl": "{}\n",
        "backups/b1/old.py": "old\n",
        "fetch/cache.json": "{}",
    }
    for rel, text in files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return root, d


def test_reset_removes_runtime_state_and_keeps_code(tmp_path):
    root, d = _project(tmp_path)
    result = workspace.reset(root)

    assert result["files"] > 0
    assert {"index.json", "plans", "sessions", "backups", "log.txt",
            "memory.md", "rules", "skills", "prompts"} <= set(result["removed"])
    assert (root / "pkg" / "core.py").is_file()
    assert (root / ".pedacito.toml").is_file()
    assert (root / ".pedacitoignore").is_file()
    assert d.is_dir()
    assert not any(d.iterdir())


def test_reset_keep_flags_preserve_only_what_was_asked(tmp_path):
    root, d = _project(tmp_path)
    result = workspace.reset(root, keep_memory=True, keep_config=True)

    assert result["kept"] == ["instructions.md", "memory.md", "prompts",
                              "rules", "skills"]
    assert (d / "memory.md").is_file()
    assert (d / "rules" / "r.md").is_file()
    assert (d / "skills" / "demo" / "SKILL.md").is_file()
    assert (d / "prompts" / "answer.md").is_file()
    assert (d / "instructions.md").is_file()
    assert not (d / "index.json").exists()
    assert not (d / "plans").exists()


def test_reset_dry_run_changes_nothing(tmp_path):
    root, d = _project(tmp_path)
    result = workspace.reset(root, dry_run=True)

    assert result["files"] > 0
    assert (d / "index.json").is_file()
    assert (d / "memory.md").is_file()
    assert (d / "plans" / "20260101-000000.md").is_file()


def test_cli_reset_wipe(tmp_path, capsys):
    root, d = _project(tmp_path)
    code = cli.main(["reset", str(root), "--yes"])
    err = capsys.readouterr().err

    assert code == 0
    assert "cleared" in err
    assert (root / "pkg" / "core.py").is_file()
    assert (root / ".pedacito.toml").is_file()
    assert not (d / "index.json").exists()
    assert not (d / "memory.md").exists()


def test_cli_reset_keep_memory_and_reindex_static(tmp_path, home, capsys,
                                                 monkeypatch):
    monkeypatch.setenv("PEDACITO_VRAM_GB", "8")
    root, d = _project(tmp_path)
    code = cli.main(["reset", str(root), "--yes", "--keep-memory",
                     "--reindex", "--no-summaries"])

    assert code == 0
    assert (d / "memory.md").is_file()
    assert (d / "index.json").is_file()
    assert not (d / "plans").exists()


@pytest.mark.skipif(not shutil.which("git"), reason="git is not installed")
def test_cli_reset_refuses_live_worktree_without_force(tmp_path, capsys):
    from pedacito import worktree

    root = tmp_path / "repo"
    root.mkdir()
    for cmd in (["git", "init", "-q"],
                ["git", "config", "user.email", "a@b.c"],
                ["git", "config", "user.name", "test"]):
        subprocess.run(cmd, cwd=root, check=True)
    (root / "a.txt").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "add", "a.txt"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=root, check=True)
    wt = worktree.create(root, "demo")
    d = workspace.state_dir(root)
    (d / "index.json").write_text("{}", encoding="utf-8")

    assert cli.main(["reset", str(root), "--yes"]) == 2
    assert (d / "index.json").is_file()
    assert wt.is_dir()

    assert cli.main(["reset", str(root), "--yes", "--force"]) == 0
    assert not wt.exists()
    assert not (d / "index.json").exists()
