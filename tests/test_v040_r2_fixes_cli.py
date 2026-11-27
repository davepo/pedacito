"""Round 2 remediation (Track R2-B):

* R2-BUG-02 -- `pdc rpc/acp --root A` from CWD B indexes only A, never B.
* R2-BUG-03 -- `pdc worktree remove` on a dirty worktree gives a clean
  Pedacito hint (no raw git `fatal:`) and works with `--force`.
* R2-BUG-07 -- a UTF-16 `.pedacito.toml` prints "Config error:" exactly once.
* R2-BUG-08 -- `pdc undo` lists only genuinely changed files and folds the
  unchanged ones into the summary count.

Every test is offline and uses throwaway temp directories/repos.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import checkpoints, cli, index as index_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


class _StubClient:
    """A local-only stand-in: no request ever reaches a server."""

    def __init__(self, cfg):
        self.cfg = cfg
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


def _config_file(tmp_path: Path) -> Path:
    p = tmp_path / "user.toml"
    p.write_text("vram_gb = 8\nrefresh_summaries = 0\n"
                 "log_sessions = false\nshow_telemetry = false\n",
                 encoding="utf-8")
    return p


def _install_static_index_build(monkeypatch) -> list[list[str]]:
    """Record the paths `_get_index` passes to `index.build` and build the
    real static index (no summariser model needed)."""
    calls: list[list[str]] = []
    real_build = index_mod.build

    def fake_build(paths, cfg, client=None, **kw):
        calls.append([str(p) for p in paths])
        return real_build(paths, cfg, client=None, summarise=False,
                          root=kw.get("root"), matcher=kw.get("matcher"),
                          resummarise=kw.get("resummarise", False))

    monkeypatch.setattr(cli.index_mod, "build", fake_build)
    return calls


def _index_files(root: Path) -> set[str]:
    data = json.loads((root / ".pedacito" / "index.json")
                      .read_text(encoding="utf-8"))
    files = set(data.get("file_hashes") or {})
    files.update(c.get("file") for c in data.get("chunks", [])
                 if isinstance(c, dict) and c.get("file"))
    files.discard(None)
    return files


def _run_front_end(tmp_path, monkeypatch, cmd: str) -> tuple[int, list[str],
                                                             set[str], Path]:
    """Run `pdc rpc|acp --root A` from an unrelated CWD B and report what
    landed in A's index."""
    project = tmp_path / "project_a"
    project.mkdir()
    (project / "inside.py").write_text("INSIDE = 1\n", encoding="utf-8")
    other = tmp_path / "cwd_b"
    other.mkdir()
    (other / "outside_code.py").write_text("OUTSIDE = 1\n", encoding="utf-8")

    calls = _install_static_index_build(monkeypatch)
    monkeypatch.setattr(cli, "LLMClient", lambda cfg: _StubClient(cfg))
    if cmd == "rpc":
        monkeypatch.setattr(cli, "_cmd_rpc", lambda *a, **k: 0)
    else:
        monkeypatch.setattr(cli, "cmd_acp", lambda *a, **k: 0)

    monkeypatch.chdir(other)
    argv = [cmd, "--root", str(project),
            "--config", str(_config_file(tmp_path)),
            "--model", "stub-model", "--url", "http://127.0.0.1:9/v1"]
    if cmd == "rpc":
        argv += ["--events", "off"]
    code = cli.main(argv)
    return code, calls[0] if calls else [], _index_files(project), project


# ============================================================================
# R2-BUG-02 -- --root A + CWD B never indexes B into A
# ============================================================================

def test_rpc_root_from_other_cwd_indexes_only_the_root(tmp_path, monkeypatch):
    code, build_paths, files, project = _run_front_end(tmp_path, monkeypatch,
                                                       "rpc")
    assert code == 0
    assert build_paths == [str(project.resolve())]
    assert "inside.py" in files
    assert "outside_code.py" not in files


def test_acp_root_from_other_cwd_indexes_only_the_root(tmp_path, monkeypatch):
    code, build_paths, files, project = _run_front_end(tmp_path, monkeypatch,
                                                       "acp")
    assert code == 0
    assert build_paths == [str(project.resolve())]
    assert "inside.py" in files
    assert "outside_code.py" not in files


def test_explicit_positional_path_resolves_against_root(tmp_path, monkeypatch):
    """An explicit relative positional with --root A is A-relative too."""
    project = tmp_path / "project_a"
    sub = project / "pkg"
    sub.mkdir(parents=True)
    (sub / "inside.py").write_text("INSIDE = 1\n", encoding="utf-8")
    other = tmp_path / "cwd_b"
    other.mkdir()
    (other / "outside_code.py").write_text("OUTSIDE = 1\n", encoding="utf-8")

    calls = _install_static_index_build(monkeypatch)
    monkeypatch.setattr(cli, "LLMClient", lambda cfg: _StubClient(cfg))
    monkeypatch.setattr(cli, "_cmd_rpc", lambda *a, **k: 0)

    monkeypatch.chdir(other)
    code = cli.main(["rpc", "pkg", "--root", str(project),
                     "--events", "off",
                     "--config", str(_config_file(tmp_path)),
                     "--model", "stub-model",
                     "--url", "http://127.0.0.1:9/v1"])
    assert code == 0
    assert calls[0] == [str(sub.resolve())]
    files = _index_files(project)
    assert "pkg/inside.py" in files
    assert "outside_code.py" not in files


# ============================================================================
# R2-BUG-03 -- removing a --dirty carry_over worktree gives a clean hint
# ============================================================================

pytest_git = pytest.mark.skipif(not shutil.which("git"),
                                reason="git is not installed")


@pytest_git
def test_worktree_remove_dirty_worktree_hints_force(tmp_path, capsys):
    root = _repo(tmp_path)
    (root / "a.txt").write_text("carried\n", encoding="utf-8")
    (root / "new.txt").write_text("also carried\n", encoding="utf-8")
    assert cli.main(["worktree", "create", "r2-carry", "--root", str(root),
                     "--dirty", "carry_over"]) == 0
    capsys.readouterr()          # drop the creation chatter

    dest = root / ".pedacito" / "worktrees" / "r2-carry"
    assert (dest / "a.txt").read_text(encoding="utf-8") == "carried\n"
    assert (dest / "new.txt").is_file()

    code = cli.main(["worktree", "remove", "r2-carry", "--root", str(root)])
    captured = capsys.readouterr()
    assert code == 1
    assert "uncommitted changes" in captured.err
    assert "--force" in captured.err
    assert "fatal:" not in captured.err
    # Nothing was discarded and the branch was preserved.
    assert (dest / ".git").exists()
    assert "pedacito/r2-carry" in _git(root, "branch", "--list")

    assert cli.main(["worktree", "remove", "r2-carry", "--root", str(root),
                     "--force"]) == 0
    assert not dest.exists()
    assert "pedacito/r2-carry" not in _git(root, "branch", "--list")


@pytest_git
def test_worktree_remove_clean_worktree_stays_quiet_on_stderr(tmp_path, capsys):
    root = _repo(tmp_path)
    assert cli.main(["worktree", "create", "r2-clean", "--root",
                     str(root)]) == 0
    capsys.readouterr()
    assert cli.main(["worktree", "remove", "r2-clean", "--root",
                     str(root)]) == 0
    captured = capsys.readouterr()
    assert "Removed worktree r2-clean" in captured.out
    assert captured.err == ""


# ============================================================================
# R2-BUG-07 -- UTF-16 project config has exactly one "Config error:" prefix
# ============================================================================

def test_utf16_config_prints_config_error_prefix_once(tmp_path, capsys):
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".pedacito.toml").write_bytes(
        b"\xff\xfe" + 'permission_mode = "plan"\n'.encode("utf-16-le"))

    code = cli.main(["config", "--root", str(project)])
    err = capsys.readouterr().err
    assert code == 2
    assert "Config error:" in err
    assert err.count("Config error:") == 1
    assert "Config error: Config error:" not in err
    assert "UTF-16" in err


# ============================================================================
# R2-BUG-08 -- pdc undo reports only genuinely restored files
# ============================================================================

def test_undo_reports_only_changed_files_and_folds_unchanged(tmp_path, capsys):
    root = tmp_path / "proj"
    root.mkdir()
    for i in range(6):
        (root / f"f{i}.py").write_text(f"VALUE = {i}\n", encoding="utf-8")

    checkpoints.create_checkpoint(root, "base")
    (root / "f0.py").write_text("VALUE = 99\n", encoding="utf-8")
    checkpoints.create_checkpoint(root, "the run")

    code = cli.main(["undo", "--root", str(root)])
    err = capsys.readouterr().err
    assert code == 0
    assert (root / "f0.py").read_text(encoding="utf-8") == "VALUE = 0\n"
    # Only the one genuinely changed file gets a "restored" line ...
    assert err.count("restored f0.py") == 1
    # ... and unchanged files are neither listed as restored nor repeated as
    # one "unchanged <path>" line each.
    assert "restored f1.py" not in err
    assert "unchanged f1.py" not in err
    assert "1 file(s) restored" in err
    assert "5 unchanged" in err
    assert "undo: back to" in err
