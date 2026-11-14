"""Green-ratchet hard-link inode safety (TECHNIQUES.md #20).

`ratchet.begin` snapshots the workspace with `os.link`, so a checkpoint and
the live file share one inode. Every Pedacito write must therefore REPLACE the
path atomically (temp file + `os.replace`) and never open it in `"w"`, `"a"`
or `"r+"`: an in-place write would mutate the checkpoint bytes too and make
`ratchet.rollback` restore wreckage without noticing.

This module audits every mutating surface named in the hardening brief --
edits, write/append/move/delete, auto_heal_static, rename_symbol -- against a
real hard-linked checkpoint, then proves rollback restores the tree exactly.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import (  # noqa: E402
    edits,
    fileio,
    index as index_mod,
    ratchet,
    refactor,
    tools,
    verify,
)
from pedacito.config import Config  # noqa: E402

FILES = {
    "README.md": "# Project\n\nNot a code file, but snapshotted too.\n",
    "src/app.py": "def compute_total(items):\n    return sum(items)\n",
    "src/heal.py": "def show():\n    return 1\n",
    "src/util.py": (
        "def helper(value):\n"
        "    return value * 2\n"
        "\n"
        "\n"
        "def call_helper(value):\n"
        "    return helper(value)\n"
    ),
    "web/app.js": "const el = document.getElementById('app');\n",
}


def _block(rel: str, search: str, replace: str) -> str:
    return (f"--- FILE: {rel}\n<<<<<<< SEARCH\n{search}\n"
            f"=======\n{replace}\n>>>>>>> REPLACE\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_state(root: Path) -> dict[str, tuple[str, int]]:
    """SHA-256 + inode of every snapshot file, manifest included."""
    d = ratchet.step_dir(root, 1)
    return {p.relative_to(d).as_posix(): (_sha(p), p.stat().st_ino)
            for p in sorted(d.rglob("*")) if p.is_file()}


def _workspace_state(root: Path) -> dict[str, bytes]:
    """Every project file outside the state directory."""
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*"))
            if p.is_file() and ".pedacito" not in p.relative_to(root).parts}


def _assert_checkpoints_pristine(root: Path,
                                 frozen: dict[str, tuple[str, int]]) -> None:
    now = _checkpoint_state(root)
    assert now == frozen, "a checkpoint file changed through its shared inode"
    for rel, (_sha, ino) in frozen.items():
        if rel == "manifest.json":
            continue
        path = ratchet.step_dir(root, 1) / rel
        assert path.stat().st_ino == ino, rel


def test_every_mutation_breaks_the_hard_link_not_the_checkpoint(tmp_path):
    for rel, text in FILES.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    initial = _workspace_state(tmp_path)

    cfg = Config()
    cfg.backup = False
    cfg.allow_write = True
    cfg.log_events = False
    cfg.max_write_lines = 10_000
    cfg.max_write_chars = 1_000_000
    idx = index_mod.build([str(tmp_path)], cfg, client=None, summarise=False,
                          root=tmp_path, verbose=False)

    info = ratchet.begin(tmp_path, 1, "inode-safety audit")
    assert info["error"] is None, info
    frozen = _checkpoint_state(tmp_path)
    assert "src/app.py" in frozen and "manifest.json" in frozen

    manifest = json.loads(
        (ratchet.step_dir(tmp_path, 1) / "manifest.json").read_text(encoding="utf-8"))
    linked = manifest["files"]["src/app.py"]["linked"]
    if linked:   # the common same-device case
        assert (ratchet.step_dir(tmp_path, 1) / "src/app.py").stat().st_ino \
            == (tmp_path / "src/app.py").stat().st_ino
    original_app_inode = frozen["src/app.py"][1]

    # 1a. a normal SEARCH/REPLACE edit
    res = edits.apply_edits(
        tmp_path,
        edits.parse_edits(_block("src/app.py", "return sum(items)",
                                 "return sum(items) + 1")),
        backup=False)
    assert res and res[0].ok, res
    _assert_checkpoints_pristine(tmp_path, frozen)
    if linked:
        # atomic replacement moved the live path onto a fresh inode; the
        # checkpoint still holds the old one (and its bytes).
        assert (tmp_path / "src/app.py").stat().st_ino != original_app_inode

    # 1b. zero-anchor SYMBOL splice
    body = "def helper(value):\n    return value * 3\n"
    reply = ("<<<<<<< SYMBOL src/util.py :: helper\n" + body
             + ">>>>>>> END_SYMBOL\n")
    res = edits.apply_edits(tmp_path, edits.parse_edits(reply), backup=False)
    assert res and res[0].ok, res
    _assert_checkpoints_pristine(tmp_path, frozen)

    # 2. an overwrite of an existing file through the write action
    out = tools.dispatch(idx, cfg, {"action": "write", "file": "web/app.js",
                                    "content": "const el = null;\n"})
    assert out.startswith("WROTE"), out
    _assert_checkpoints_pristine(tmp_path, frozen)

    # ... and a brand-new file (rollback must delete it)
    out = tools.dispatch(idx, cfg, {"action": "write", "file": "src/created.py",
                                    "content": "CREATED = True\n"})
    assert out.startswith("CREATED"), out
    _assert_checkpoints_pristine(tmp_path, frozen)

    # 3. APPEND to an existing, hard-linked file -- the "a" mode trap
    out = tools.dispatch(idx, cfg, {"action": "append", "file": "src/util.py",
                                    "content": "\n# appended tail\n"})
    assert "APPENDED" in out, out
    assert (tmp_path / "src/util.py").read_text(encoding="utf-8").endswith("# appended tail\n")
    _assert_checkpoints_pristine(tmp_path, frozen)

    # 4. move and delete
    out = tools.dispatch(idx, cfg, {"action": "move", "file": "src/app.py",
                                    "to": "src/moved.py"})
    assert "MOVED" in out, out
    assert not (tmp_path / "src/app.py").exists()
    assert (tmp_path / "src/moved.py").is_file()
    _assert_checkpoints_pristine(tmp_path, frozen)

    out = tools.dispatch(idx, cfg, {"action": "delete", "file": "web/app.js"})
    assert "DELETED" in out, out
    assert not (tmp_path / "web/app.js").exists()
    _assert_checkpoints_pristine(tmp_path, frozen)

    # 5. auto_heal_static(persist=True) rewrites the file with the missing
    #    stdlib import; it must go through the atomic writer too.
    unhealed = "def show():\n    return json.dumps({})\n"
    fileio.write(tmp_path / "src/heal.py", unhealed)   # the edit that landed
    _assert_checkpoints_pristine(tmp_path, frozen)
    result = edits.Result(True, "src/heal.py", "",
                          before="def show():\n    return 1\n",
                          after=unhealed)
    receipts = verify.auto_heal_static([result], root=tmp_path, cfg=cfg,
                                       persist=True)
    assert receipts, "expected an import auto-heal receipt"
    assert "import json" in fileio.read_text(tmp_path / "src/heal.py")
    _assert_checkpoints_pristine(tmp_path, frozen)

    # 6. deterministic multi-file rename, write pass included
    renamed = refactor.rename_symbol_across_project(
        tmp_path, "helper", "assist", dry_run=False)
    assert renamed.get("ok"), renamed
    assert "src/util.py" in renamed["files_modified"], renamed
    assert "def assist(value):" in fileio.read_text(tmp_path / "src/util.py")
    _assert_checkpoints_pristine(tmp_path, frozen)

    # Rollback: every original byte returns, every creation/move target goes.
    report = ratchet.rollback(tmp_path, 1)
    assert report["error"] is None, report
    assert "src/app.py" in report["restored"], report
    assert "web/app.js" in report["restored"], report
    assert "src/created.py" in report["deleted"], report
    assert "src/moved.py" in report["deleted"], report
    assert _workspace_state(tmp_path) == initial
    assert not ratchet.step_dir(tmp_path, 1).exists()
