"""``pedacito-package.toml`` validation and SHA-256 trust pinning."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import pkg_manifest as pkg  # noqa: E402


def _manifest_text(**overrides) -> str:
    data = {
        "schema_version": 1,
        "name": "acme-lint",
        "version": "1.2.0",
        "kind": "extension",
        "description": "Acme lint rules",
        "capabilities": ["read_workspace"],
        "entry_points": {"main": "acme_lint/plugin.py"},
    }
    data.update(overrides)
    lines = [
        f'schema_version = {data["schema_version"]}',
        f'name = {json.dumps(data["name"])}',
        f'version = {json.dumps(data["version"])}',
        f'kind = {json.dumps(data["kind"])}',
        f'description = {json.dumps(data["description"])}',
        "capabilities = [" + ", ".join(
            json.dumps(c) for c in data["capabilities"]) + "]",
        "min_pedacito_version = \"0.4.0\"",
        "",
        "[entry_points]",
    ]
    lines += [f"{key} = {json.dumps(value)}"
              for key, value in data["entry_points"].items()]
    return "\n".join(lines) + "\n"


def _package(tmp_path: Path, text: str | None = None) -> Path:
    root = tmp_path / "pkg"
    root.mkdir(exist_ok=True)
    (root / pkg.MANIFEST_FILENAME).write_text(text or _manifest_text(),
                                               encoding="utf-8")
    return root


@pytest.mark.parametrize("kind", pkg.PACKAGE_KINDS)
def test_load_manifest_accepts_every_kind(tmp_path, kind):
    root = _package(tmp_path, _manifest_text(kind=kind))
    manifest = pkg.load_manifest(root)
    assert manifest.kind == kind
    assert manifest.schema_version == 1
    assert manifest.name == "acme-lint"
    assert manifest.version == "1.2.0"
    assert manifest.capabilities == ("read_workspace",)
    assert manifest.entry_points == {"main": "acme_lint/plugin.py"}
    raw = (root / pkg.MANIFEST_FILENAME).read_bytes()
    assert manifest.sha256 == hashlib.sha256(raw).hexdigest()
    # The direct manifest path is accepted too.
    direct = pkg.load_manifest(root / pkg.MANIFEST_FILENAME)
    assert direct.sha256 == manifest.sha256


def test_unknown_kind_is_rejected(tmp_path):
    root = _package(tmp_path, _manifest_text(kind="plugin"))
    with pytest.raises(ValueError) as err:
        pkg.load_manifest(root)
    assert "kind must be one of" in str(err.value)


def test_unknown_capability_is_rejected(tmp_path):
    root = _package(tmp_path, _manifest_text(capabilities=["root_everything"]))
    with pytest.raises(ValueError) as err:
        pkg.load_manifest(root)
    assert "unknown capability" in str(err.value)


def test_bad_schema_version_is_rejected(tmp_path):
    root = _package(tmp_path, _manifest_text(schema_version=2))
    with pytest.raises(ValueError) as err:
        pkg.load_manifest(root)
    assert "schema_version" in str(err.value)


@pytest.mark.parametrize("entry", [
    "../evil.py",
    "../../etc/passwd",
    "/absolute/evil.py",
    "C:\\evil.py",
    "pkg/../../evil.py",
    "",
])
def test_entry_point_traversal_is_rejected(tmp_path, entry):
    root = _package(tmp_path, _manifest_text(entry_points={"main": entry}))
    with pytest.raises(ValueError) as err:
        pkg.load_manifest(root)
    assert "entry_points" in str(err.value)


def test_validate_manifest_reports_every_field():
    errors = pkg.validate_manifest({
        "schema_version": 9,
        "name": "bad name!",
        "version": "",
        "kind": "nope",
        "capabilities": ["nope"],
        "entry_points": {"main": "../x"},
    })
    text = "\n".join(errors)
    for needle in ("schema_version", "name", "version", "kind",
                   "unknown capability", "entry_points"):
        assert needle in text


def test_trust_pins_sha_and_mutation_invalidates(tmp_path):
    root = _package(tmp_path)
    manifest = root / pkg.MANIFEST_FILENAME
    assert pkg.is_manifest_trusted(root) is False
    returned = pkg.trust_manifest(root)
    assert returned == manifest
    assert pkg.is_manifest_trusted(root) is True
    assert pkg.is_manifest_trusted(manifest) is True

    # A single byte change must flip trust back to False.
    original = manifest.read_bytes()
    manifest.write_bytes(original + b"\n")
    assert pkg.is_manifest_trusted(root) is False
    assert pkg.is_manifest_trusted(manifest) is False

    assert pkg.untrust_manifest(root) is True
    assert pkg.untrust_manifest(root) is False
    assert pkg.is_manifest_trusted(root) is False


def test_trust_refuses_an_invalid_manifest(tmp_path):
    root = _package(tmp_path, _manifest_text(kind="nope"))
    with pytest.raises(ValueError):
        pkg.trust_manifest(root)
    assert pkg.is_manifest_trusted(root) is False


def test_entry_point_file_mutation_invalidates_trust(tmp_path):
    root = _package(tmp_path)
    (root / "acme_lint").mkdir()
    ep = root / "acme_lint" / "plugin.py"
    ep.write_text("x = 1\n", encoding="utf-8")
    pkg.trust_manifest(root)
    assert pkg.is_manifest_trusted(root) is True
    # Mutating a resolved entry-point file (not the manifest) revokes trust.
    ep.write_text("x = 2\n", encoding="utf-8")
    assert pkg.is_manifest_trusted(root) is False


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink escape test")
def test_entry_point_symlink_escape_is_rejected(tmp_path):
    root = _package(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("evil", encoding="utf-8")
    (root / "acme_lint").mkdir()
    link = root / "acme_lint" / "plugin.py"
    link.symlink_to(outside)
    with pytest.raises(ValueError) as err:
        pkg.load_manifest(root)
    assert "escapes the package" in str(err.value)
