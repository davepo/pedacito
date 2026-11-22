"""VirtualWorkspace: the shadow overlay behind the interleaved agent loop.

Covers both modes (apply=False overlay / apply=True disk), snapshot and
rollback, review/edit-result compatibility, the dry-run command mirror, and
the path-containment guarantees (resolve_safe_path, .git/, secrets).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import edits, review, verify
from pedacito.config import Config
from pedacito.workspace_overlay import (PathRefused, VirtualWorkspace,
                                        WorkspaceError)

PRICING = (
    "def bulk_discount(quantity):\n"
    "    if quantity > 10:\n"
    "        return 0.1\n"
    "    return 0.0\n"
    "\n"
    "\n"
    "def total(price, quantity):\n"
    "    return price * quantity * (1 - bulk_discount(quantity))\n"
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture
def proj(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "pricing.py").write_text(PRICING, encoding="utf-8")
    (root / "notes.txt").write_text("hello\nworld\n", encoding="utf-8")
    (root / "pkg").mkdir()
    (root / "pkg" / "util.py").write_text("def helper():\n    return 1\n",
                                          encoding="utf-8")
    return root


def tree(root: Path) -> dict:
    """Every path under root with its bytes (None for a directory)."""
    out = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        out[rel] = p.read_bytes() if p.is_file() else None
    return out


# ----------------------------------------------------------- overlay mode
def test_overlay_edit_is_visible_to_read_but_never_touches_disk(proj):
    before = tree(proj)
    ws = VirtualWorkspace(proj, apply=False)
    res = ws.edit("pricing.py", "if quantity > 10:", "if quantity >= 10:")
    assert res.ok, res.message
    assert "staged in memory" in res.message
    assert ">= 10" in ws.read("pricing.py")
    assert ws.exists("pricing.py")
    assert ws.changed_files() == ["pricing.py"]
    assert ws.is_staged("pricing.py")
    assert tree(proj) == before                       # 100% untouched
    assert "quantity > 10" in (proj / "pricing.py").read_text(encoding="utf-8")


def test_overlay_edits_compose_and_report_cumulatively(proj):
    ws = VirtualWorkspace(proj, apply=False)
    assert ws.edit("pricing.py", "if quantity > 10:", "if quantity >= 10:").ok
    assert ws.edit("pricing.py", "return 0.1", "return 0.15").ok
    text = ws.read("pricing.py")
    assert ">= 10" in text and "0.15" in text
    (r,) = ws.to_edit_results()
    assert r.file == "pricing.py" and r.before == PRICING and r.after == text
    assert r.sha == sha(PRICING)
    assert "applied 2 edit(s)" in r.message and "(dry run)" in r.message
    assert r.diff.count("\n-") >= 2


def test_overlay_create_and_delete(proj):
    before = tree(proj)
    ws = VirtualWorkspace(proj, apply=False)
    assert ws.write("pkg/new_mod.py", "X = 1\n").ok
    assert ws.exists("pkg/new_mod.py")
    assert ws.read("pkg/new_mod.py") == "X = 1\n"
    assert not (proj / "pkg" / "new_mod.py").exists()

    res = ws.delete("notes.txt")
    assert res.ok and res.deleted
    assert ws.is_deleted("notes.txt")
    assert not ws.exists("notes.txt")
    with pytest.raises(FileNotFoundError):
        ws.read("notes.txt")
    assert (proj / "notes.txt").exists()
    assert tree(proj) == before
    assert ws.changed_files() == ["notes.txt", "pkg/new_mod.py"]
    files = ws.list_files()
    assert "pkg/new_mod.py" in files and "notes.txt" not in files


def test_created_then_deleted_is_not_a_change(proj):
    ws = VirtualWorkspace(proj, apply=False)
    ws.write("scratch.py", "A = 1\n")
    ws.delete("scratch.py")
    assert ws.changed_files() == []
    assert ws.to_edit_results() == [] and ws.to_file_diffs() == []


def test_edit_then_restoring_original_text_is_not_a_change(proj):
    ws = VirtualWorkspace(proj, apply=False)
    ws.edit("notes.txt", "hello", "goodbye")
    ws.edit("notes.txt", "goodbye", "hello")
    assert ws.changed_files() == []
    assert not ws.has_changes()


def test_failed_edits_stage_nothing(proj):
    ws = VirtualWorkspace(proj, apply=False)
    miss = ws.edit("pricing.py", "no such text", "x")
    assert not miss.ok and "not found" in miss.message
    ambiguous = ws.edit("pricing.py", "return", "yield")
    assert not ambiguous.ok and "appears" in ambiguous.message
    missing_file = ws.edit("ghost.py", "a", "b")
    assert not missing_file.ok
    assert ws.changed_files() == []
    assert ws.revision == 0


def test_syntax_error_edit_is_refused_and_state_unchanged(proj):
    ws = VirtualWorkspace(proj, apply=False)
    res = ws.edit("pricing.py", "return 0.1", "return (0.1")
    assert not res.ok and "syntax error" in res.message
    assert ws.read("pricing.py") == PRICING
    assert ws.changed_files() == []


def test_symbol_and_line_anchored_edits(proj):
    ws = VirtualWorkspace(proj, apply=False, cfg=Config())
    res = ws.edit("pricing.py", symbol="bulk_discount",
                  replace="def bulk_discount(quantity):\n    return 0.5\n")
    assert res.ok, res.message
    assert "return 0.5" in ws.read("pricing.py")
    assert "def total" in ws.read("pricing.py")
    res = ws.edit("notes.txt", start_line=1, end_line=1, replace="HELLO")
    assert res.ok, res.message
    assert ws.read("notes.txt").startswith("HELLO\nworld")


def test_non_utf8_file_is_refused_not_mangled(proj):
    (proj / "blob.txt").write_bytes(b"\xff\xfe\x00bad")
    ws = VirtualWorkspace(proj, apply=False)
    with pytest.raises(WorkspaceError):
        ws.read("blob.txt")
    res = ws.edit("blob.txt", "bad", "good")
    assert not res.ok and "UTF-8" in res.message


# ------------------------------------------------- results and review parity
def test_to_edit_results_are_engine_results(proj):
    ws = VirtualWorkspace(proj, apply=False)
    ws.edit("pricing.py", "return 0.1", "return 0.2")
    ws.write("new.py", "A = 1\n")
    ws.delete("notes.txt")
    results = ws.to_edit_results()
    assert all(isinstance(r, edits.Result) for r in results)
    by = {r.file: r for r in results}
    assert by["new.py"].before == "" and by["new.py"].sha == sha("")
    assert by["notes.txt"].deleted and by["notes.txt"].after == ""
    assert by["pricing.py"].sha == sha(PRICING)
    # the same verifiers the two-phase loop runs accept them
    assert verify.check_syntax(results) == []


def test_to_file_diffs_are_valid_review_rows(proj):
    ws = VirtualWorkspace(proj, apply=False)
    ws.edit("pricing.py", "return 0.1", "return 0.2")
    ws.write("fresh.py", "A = 1\nB = 2\n")
    ws.delete("notes.txt")
    diffs = {fd.rel: fd for fd in ws.to_file_diffs()}
    assert set(diffs) == {"pricing.py", "fresh.py", "notes.txt"}
    assert all(isinstance(fd, review.FileDiff) for fd in diffs.values())
    p = diffs["pricing.py"]
    assert (p.added, p.removed) == (1, 1)
    assert p.sha == sha(PRICING) and "return 0.2" in p.after_text
    assert p.rows and any(r.kind == "replace" for r in p.rows)
    assert diffs["fresh.py"].added == 2 and diffs["fresh.py"].sha == sha("")
    assert diffs["notes.txt"].deleted and diffs["notes.txt"].after_text == ""
    # a real review page renders from them
    page = review.build_page(list(diffs.values()), "task", "model", "profile",
                             "nonce", False, "stamp")
    assert "pricing.py" in page and "fresh.py" in page


def test_review_signoff_writes_the_staged_edits(proj, tmp_path):
    ws = VirtualWorkspace(proj, apply=False)
    ws.edit("pricing.py", "return 0.1", "return 0.2")
    ws.write("fresh.py", "A = 1\n")
    ws.delete("notes.txt")
    backup = tmp_path / "bk"
    ok, msg = review.write_files(proj, ws.to_file_diffs(), backup)
    assert ok, msg
    assert "return 0.2" in (proj / "pricing.py").read_text(encoding="utf-8")
    assert (proj / "fresh.py").read_text(encoding="utf-8") == "A = 1\n"
    assert not (proj / "notes.txt").exists()
    assert (backup / "notes.txt").read_text(encoding="utf-8") == "hello\nworld\n"


def test_diff_text_and_line_counts(proj):
    ws = VirtualWorkspace(proj, apply=False)
    assert ws.diff_text() == ""
    ws.edit("pricing.py", "return 0.1", "return 0.2")
    ws.edit("notes.txt", "hello\n", "hello\nthere\n")
    text = ws.diff_text()
    assert "--- a/pricing.py" in text and "--- a/notes.txt" in text
    only = ws.diff_text("notes.txt")
    assert "pricing.py" not in only and "+there" in only
    assert ws.line_counts() == {"pricing.py": (1, 1), "notes.txt": (1, 0)}
    assert len(ws.diff_text(max_chars=40)) < len(text)


# ------------------------------------------------------- snapshot / rollback
def test_rollback_in_overlay_mode(proj):
    ws = VirtualWorkspace(proj, apply=False)
    ws.edit("pricing.py", "return 0.1", "return 0.2")
    snap = ws.snapshot()
    ws.edit("pricing.py", "return 0.2", "return 0.3")
    ws.write("later.py", "Z = 1\n")
    ws.delete("notes.txt")
    rev = ws.revision
    ws.rollback(snap)
    assert ws.revision > rev
    assert "return 0.2" in ws.read("pricing.py")
    assert not ws.exists("later.py") and ws.exists("notes.txt")
    assert ws.changed_files() == ["pricing.py"]
    with pytest.raises(WorkspaceError):
        VirtualWorkspace(proj, apply=False).rollback(snap)


def test_rollback_restores_disk_in_apply_mode(proj):
    ws = VirtualWorkspace(proj, apply=True, backup_dir=proj.parent / "bk")
    snap = ws.snapshot()
    ws.edit("pricing.py", "return 0.1", "return 0.2")
    ws.write("deep/er/new.py", "N = 1\n")
    ws.delete("notes.txt")
    assert "return 0.2" in (proj / "pricing.py").read_text(encoding="utf-8")
    assert (proj / "deep" / "er" / "new.py").exists()
    assert not (proj / "notes.txt").exists()
    ws.rollback(snap)
    assert (proj / "pricing.py").read_text(encoding="utf-8") == PRICING
    assert (proj / "notes.txt").read_text(encoding="utf-8") == "hello\nworld\n"
    assert not (proj / "deep").exists()               # empty dirs pruned
    assert ws.changed_files() == []


def test_rollback_is_per_edit_not_all_or_nothing(proj):
    ws = VirtualWorkspace(proj, apply=True, backup_dir=proj.parent / "bk")
    ws.edit("pricing.py", "return 0.1", "return 0.2")
    snap = ws.snapshot()
    ws.edit("pricing.py", "return 0.2", "return 0.9")
    ws.rollback(snap)
    assert "return 0.2" in (proj / "pricing.py").read_text(encoding="utf-8")
    assert ws.changed_files() == ["pricing.py"]


# ------------------------------------------------------------- apply mode
def test_apply_mode_writes_through_fileio_with_backup(proj, tmp_path):
    bk = tmp_path / "bk"
    ws = VirtualWorkspace(proj, apply=True, backup_dir=bk)
    res = ws.edit("pricing.py", "return 0.1", "return 0.2")
    assert res.ok and "staged in memory" not in res.message
    assert "(dry run)" not in res.message
    assert "return 0.2" in (proj / "pricing.py").read_text(encoding="utf-8")
    assert (bk / "pricing.py").read_text(encoding="utf-8") == PRICING
    ws.edit("pricing.py", "return 0.2", "return 0.3")
    assert (bk / "pricing.py").read_text(encoding="utf-8") == PRICING   # first copy wins
    ws.write("fresh.py", "A = 1\n")
    assert fileio_manifest(bk) == ["fresh.py"]
    (r,) = [x for x in ws.to_edit_results() if x.file == "pricing.py"]
    assert r.before == PRICING and "(dry run)" not in r.message


def fileio_manifest(backup_dir: Path) -> list[str]:
    from pedacito import fileio
    m = fileio.created_manifest(backup_dir)
    return m.read_text(encoding="utf-8").split() if m.exists() else []


def test_apply_mode_preserves_crlf_and_bom(proj):
    (proj / "win.txt").write_bytes(b"\xef\xbb\xbfone\r\ntwo\r\n")
    ws = VirtualWorkspace(proj, apply=True, backup_dir=proj.parent / "bk")
    assert ws.edit("win.txt", "two", "TWO").ok
    assert (proj / "win.txt").read_bytes() == b"\xef\xbb\xbfone\r\nTWO\r\n"
    assert ws.read("win.txt") == "one\nTWO\n"


def test_apply_mode_delete_backs_up_first(proj, tmp_path):
    bk = tmp_path / "bk"
    ws = VirtualWorkspace(proj, apply=True, backup_dir=bk)
    res = ws.delete("notes.txt")
    assert res.ok and "backed up" in res.message
    assert not (proj / "notes.txt").exists()
    assert (bk / "notes.txt").read_text(encoding="utf-8") == "hello\nworld\n"
    assert ws.is_deleted("notes.txt")


def test_apply_mode_without_backup_dir_makes_one_lazily(proj):
    ws = VirtualWorkspace(proj, apply=True)
    ws.edit("pricing.py", "return 0.1", "return 0.2")
    assert ws.backup_dir is not None and ws.backup_dir.is_dir()
    assert ".pedacito/backups" in ws.backup_dir.as_posix()


# ------------------------------------------------------ path containment
@pytest.mark.parametrize("bad", ["../outside.py", "pkg/../../outside.py",
                                 "pkg/../pricing.py"])
@pytest.mark.parametrize("apply", [False, True])
def test_traversal_is_refused_everywhere(proj, apply, bad):
    outside = proj.parent / "outside.py"
    ws = VirtualWorkspace(proj, apply=apply, backup_dir=proj.parent / "bk")
    before = tree(proj)
    assert not ws.write(bad, "x = 1\n").ok
    assert not ws.edit(bad, "a", "b").ok
    assert not ws.delete(bad).ok
    assert "traversal" in ws.write(bad, "x\n").message
    with pytest.raises(PathRefused):
        ws.read(bad)
    assert not ws.exists(bad)
    assert not outside.exists()
    assert tree(proj) == before


@pytest.mark.parametrize("apply", [False, True])
def test_absolute_and_symlink_escapes_are_refused(proj, tmp_path, apply):
    secret = tmp_path / "elsewhere.txt"
    secret.write_text("top secret\n", encoding="utf-8")
    (proj / "link.txt").symlink_to(secret)
    (proj / "dirlink").symlink_to(tmp_path, target_is_directory=True)
    ws = VirtualWorkspace(proj, apply=apply, backup_dir=proj.parent / "bk")
    for target in (str(secret), "link.txt", "dirlink/elsewhere.txt",
                   "dirlink/new.py"):
        with pytest.raises(PathRefused):
            ws.read(target)
        assert not ws.write(target, "overwritten\n").ok
        assert not ws.delete(target).ok
    assert secret.read_text(encoding="utf-8") == "top secret\n"
    assert not (tmp_path / "new.py").exists()


@pytest.mark.parametrize("apply", [False, True])
def test_git_and_pedacito_state_are_never_touched(proj, apply):
    (proj / ".git").mkdir()
    (proj / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (proj / ".pedacito").mkdir()
    ws = VirtualWorkspace(proj, apply=apply, backup_dir=proj.parent / "bk")
    for rel in (".git/config", ".git/hooks/pre-commit", ".pedacito/index.json",
                ".pedacito.toml", "sub/../.git/config"):
        assert not ws.write(rel, "x\n").ok, rel
        assert not ws.edit(rel, "a", "b").ok, rel
        assert not ws.delete(rel).ok, rel
    with pytest.raises(PathRefused):
        ws.read(".git/config")
    assert (proj / ".git" / "config").read_text(encoding="utf-8") == "[core]\n"
    assert not (proj / ".git" / "hooks").exists()
    msg = ws.write(".git/config", "x").message
    assert "refused" in msg and ".git" in msg


@pytest.mark.parametrize("apply", [False, True])
def test_secret_files_are_blocked_but_template_env_is_not(proj, apply):
    (proj / ".env").write_text("TOKEN=abc\n", encoding="utf-8")
    (proj / ".env.example").write_text("TOKEN=\n", encoding="utf-8")
    ws = VirtualWorkspace(proj, apply=apply, backup_dir=proj.parent / "bk")
    with pytest.raises(PathRefused) as err:
        ws.read(".env")
    assert err.value.sensitive
    assert ws.read(".env", allow_sensitive=True) == "TOKEN=abc\n"
    assert not ws.write(".env", "TOKEN=x\n").ok      # writes: never, even so
    assert not ws.write("deploy/id_rsa", "key").ok
    assert ws.read(".env.example") == "TOKEN=\n"
    assert ".env" not in ws.list_files()
    assert (proj / ".env").read_text(encoding="utf-8") == "TOKEN=abc\n"


def test_config_protected_paths_block_writes_only(proj):
    cfg = Config()
    cfg.protected_paths = ("pkg/util.py",)
    ws = VirtualWorkspace(proj, apply=False, cfg=cfg)
    assert not ws.edit("pkg/util.py", "return 1", "return 2").ok
    assert "protected_paths" in ws.edit("pkg/util.py", "a", "b").message
    assert "def helper" in ws.read("pkg/util.py")      # still readable


def test_project_root_and_directories_are_not_files(proj):
    ws = VirtualWorkspace(proj, apply=False)
    assert not ws.write(".", "x").ok
    res = ws.write("pkg", "x")
    assert not res.ok and "directory" in res.message
    assert not ws.delete("pkg").ok
    assert not ws.exists("pkg/does_not_exist.py")
    with pytest.raises(WorkspaceError):
        ws.read("pkg")


def test_workspace_needs_a_real_directory(tmp_path):
    with pytest.raises(WorkspaceError):
        VirtualWorkspace(tmp_path / "missing")


# ----------------------------------------------------------- exec mirror
def test_exec_root_is_a_mirror_with_the_overlay_applied(proj):
    (proj / ".git").mkdir()
    (proj / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (proj / ".env").write_text("TOKEN=abc\n", encoding="utf-8")
    before = tree(proj)
    ws = VirtualWorkspace(proj, apply=False)
    ws.edit("pricing.py", "return 0.1", "return 0.2")
    ws.write("new_mod.py", "N = 1\n")
    ws.delete("notes.txt")
    with ws.exec_root() as mirror:
        assert mirror != proj and mirror.resolve() != proj.resolve()
        assert "return 0.2" in (mirror / "pricing.py").read_text(encoding="utf-8")
        assert (mirror / "new_mod.py").read_text(encoding="utf-8") == "N = 1\n"
        assert not (mirror / "notes.txt").exists()
        assert (mirror / "pkg" / "util.py").exists()
        assert not (mirror / ".git").exists()            # VCS state not copied
        assert not (mirror / ".env").exists()            # secrets not copied
        assert ws.scrub_paths(f"File {mirror}/pricing.py") == \
            f"File {proj}/pricing.py"
        kept = mirror
    assert tree(proj) == before
    ws.close()
    assert not kept.exists()
    ws.close()                                           # idempotent
    with ws.exec_root() as again:                        # rebuilt on demand
        assert "return 0.2" in (again / "pricing.py").read_text(encoding="utf-8")


def test_mirror_follows_rollback(proj):
    ws = VirtualWorkspace(proj, apply=False)
    snap = ws.snapshot()
    ws.edit("pricing.py", "return 0.1", "return 0.2")
    ws.write("temp.py", "T = 1\n")
    with ws.exec_root() as m:
        assert "0.2" in (m / "pricing.py").read_text(encoding="utf-8") and (m / "temp.py").exists()
    ws.rollback(snap)
    with ws.exec_root() as m:
        assert (m / "pricing.py").read_text(encoding="utf-8") == PRICING
        assert not (m / "temp.py").exists()
    ws.close()


def test_apply_mode_exec_root_is_the_real_root(proj):
    ws = VirtualWorkspace(proj, apply=True, backup_dir=proj.parent / "bk")
    with ws.exec_root() as root:
        assert root == proj.resolve()


def test_context_manager_closes_the_mirror(proj):
    with VirtualWorkspace(proj, apply=False) as ws:
        with ws.exec_root() as m:
            kept = m
        assert kept.exists()
    assert not kept.exists()


def test_every_mutation_bumps_revision(proj):
    ws = VirtualWorkspace(proj, apply=False)
    assert ws.revision == 0
    ws.edit("pricing.py", "return 0.1", "return 0.2")
    assert ws.revision == 1
    ws.read("pricing.py")
    assert ws.revision == 1
