"""Release drift guards.

The version is written in exactly two files, and a release may only ship
when the changelog names that version:

* `pyproject.toml` `[project] version` (what the build stamps into the
  sdist/wheel metadata);
* `pedacito/__init__.py` `__version__` (what `pdc --version` and the
  auto-update checker read at runtime);
* `CHANGELOG.md` `## vX.Y.Z` heading (the release notes).

If either copy drifts, or the changelog has no section for the version,
this test fails before a broken artifact can be built or uploaded. See
RELEASE.md for the full checklist.
"""
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import __version__ as package_version

ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)["project"]["version"]


def test_version_is_identical_in_pyproject_and_package():
    pyproject_version = _pyproject_version()
    assert pyproject_version == package_version, (
        "Version drift: pyproject.toml says "
        f"{pyproject_version!r} but pedacito/__init__.py says "
        f"{package_version!r}. Bump both together (see RELEASE.md)."
    )


def test_changelog_has_a_heading_for_the_release_version():
    version = _pyproject_version()
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = re.compile(
        rf"^##\s+v?{re.escape(version)}(?:\s|$)", re.MULTILINE
    )
    assert heading.search(changelog), (
        f"CHANGELOG.md has no '## v{version}' heading. Add the release "
        "notes before tagging (see RELEASE.md)."
    )
