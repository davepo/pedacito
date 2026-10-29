"""Offline tests for file selection (`pedacito select`).

Focuses on the directory walker and the Python-virtualenv skip, which keeps a
venv with a non-standard name (``pdc_venv``, ``myenv``, ...) inside the project
from dragging its ``site-packages`` (Flask etc.) into the index. No LLM server
required.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import select


def _make_venv(root: Path, name: str) -> Path:
    """Create a minimal Python-virtualenv-shaped directory under `root`."""
    venv = root / name
    (venv / "lib" / "python3.11" / "site-packages" / "flask").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    (venv / "lib" / "python3.11" / "site-packages" / "flask" / "__init__.py").write_text(
        "MARKER = True\n", encoding="utf-8")
    return venv


def test_is_venv_dir_detects_pyvenv_cfg(tmp_path):
    venv = _make_venv(tmp_path, "pdc_venv")
    assert select.is_venv_dir(venv)
    assert not select.is_venv_dir(tmp_path)
    assert not select.is_venv_dir(tmp_path / "does_not_exist")


def test_walk_skips_venv_by_any_name(tmp_path):
    (tmp_path / "app.py").write_text("print('project')\n", encoding="utf-8")
    # A venv with a name NOT covered by DEFAULT_IGNORE (.venv/venv/env).
    _make_venv(tmp_path, "pdc_venv")
    # A normal (non-venv) directory with a .py file must still be walked.
    (tmp_path / "src" / "mod.py").parent.mkdir(parents=True)
    (tmp_path / "src" / "mod.py").write_text("x = 1\n", encoding="utf-8")

    walked = {p.relative_to(tmp_path).as_posix() for p in select._walk(tmp_path)}

    assert "app.py" in walked
    assert "src/mod.py" in walked
    # Nothing under the venv -- not even pyvenv.cfg -- is walked.
    assert not any(p.startswith("pdc_venv/") for p in walked)


def test_select_excludes_venv_site_packages(tmp_path):
    from pedacito.select import Matcher, select as select_files

    (tmp_path / "app.py").write_text("print('project')\n", encoding="utf-8")
    _make_venv(tmp_path, "pdc_venv")

    matcher = Matcher.from_files(tmp_path)
    files = select_files([str(tmp_path)], tmp_path, matcher, {".py"})
    rels = sorted(f.resolve().relative_to(tmp_path).as_posix() for f in files)

    assert rels == ["app.py"]
