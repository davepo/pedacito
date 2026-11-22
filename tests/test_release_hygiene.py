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

import json
import re
import sys
import tomllib
from importlib import resources
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import __version__ as package_version
from pedacito import cli, i18n, imagegen

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


def test_readme_root_markdown_links_are_packaged():
    """An installed README lives in share/pedacito; every root-level .md it
    links must be shipped beside it, or the link 404s in a wheel install."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    linked = {target for target in re.findall(r"\]\(([^)#]+\.md)", readme)
              if "/" not in target and "\\" not in target}
    with (ROOT / "pyproject.toml").open("rb") as fh:
        data = tomllib.load(fh)
    shipped = set()
    for patterns in data["tool"]["setuptools"]["data-files"].values():
        for pattern in patterns:
            if "*" not in pattern:
                shipped.add(Path(pattern).name)
    missing = sorted(linked - shipped)
    assert not missing, (
        "README links root-level guides that are not in "
        f"[tool.setuptools.data-files]: {missing}")


def test_pdc_version_prints_the_package_version(capsys):
    """`pdc --version` used to exit 2; the release procedure uses it."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"pedacito {package_version}"


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


def test_package_data_ships_locales_and_gui_static():
    """The wheel must carry the i18n catalogs and GUI assets; a missing
    package-data pattern leaves an installed `pdc` speaking one language
    or serving a blank page."""
    with (ROOT / "pyproject.toml").open("rb") as fh:
        package_data = tomllib.load(fh)["tool"]["setuptools"]["package-data"]
    assert "locales/*.json" in package_data.get("pedacito", []), (
        "[tool.setuptools.package-data] does not ship pedacito/locales/*.json")
    assert "static/*" in package_data.get("pedacito.gui", []), (
        "[tool.setuptools.package-data] does not ship "
        "pedacito/gui/static/*")

    for code in ("en", "es"):
        raw = resources.files("pedacito").joinpath(
            "locales", f"{code}.json").read_text(encoding="utf-8")
        catalog = json.loads(raw)
        assert isinstance(catalog, dict) and catalog, (
            f"pedacito/locales/{code}.json is empty or not an object")
        assert len(catalog) == len(i18n.load_catalog(code))


def test_docs_screenshots_exist_and_are_valid_pngs():
    """The docs link the captured GUI/TUI screenshots; a truncated or
    placeholder file ships a broken image to every reader."""
    for name in ("gui-workspace.png", "tui-split.png"):
        path = ROOT / "docs" / "assets" / name
        data = path.read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n"), (
            f"docs/assets/{name} does not start with the PNG magic bytes")
        kind, width, height = imagegen.sniff_image_format(data)
        assert kind == "png", f"docs/assets/{name} is not sniffed as PNG"
        assert width >= 320 and height >= 200, (
            f"docs/assets/{name} is suspiciously small: {width}x{height}")

