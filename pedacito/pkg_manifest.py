"""Package and trust manifest for ``pedacito-package.toml`` (WP-4006, §1.2).

A package is a folder that ships an extension, skill, command, or model pack.
The manifest is the *declaration* a human reviews before any of it may run::

    schema_version = 1
    name = "acme-lint"
    version = "1.2.0"
    kind = "extension"          # extension | skill | command | model-pack
    description = "Acme lint rules for Pedacito"
    capabilities = ["read_workspace", "run_commands"]
    min_pedacito_version = "0.4.0"

    [entry_points]
    main = "acme_lint/plugin.py"

Only the stdlib is used (``tomllib`` + ``hashlib``). Trust is a pinned SHA-256
of the raw manifest bytes in the per-user state folder
(``trusted_packages.json``): changing even one byte after ``trust_manifest``
makes ``is_manifest_trusted`` answer ``False`` again. Nothing here executes a
package; loading/running extensions is a later milestone.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - pyproject requires 3.11+
    import tomli as tomllib  # type: ignore[no-redef]

MANIFEST_FILENAME = "pedacito-package.toml"
MANIFEST_SCHEMA_VERSION = 1
PACKAGE_KINDS = ("extension", "skill", "command", "model-pack")
DECLARED_CAPABILITIES = (
    "read_workspace",
    "write_workspace",
    "run_commands",
    "network",
    "mcp_server",
    "hooks",
    "prompts",
    "model_profiles",
)

TRUST_FILENAME = "trusted_packages.json"

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DRIVE_RE = re.compile(r"^[A-Za-z]:")


@dataclass(frozen=True)
class PackageManifest:
    """A validated manifest; ``sha256`` pins the exact bytes that were read,
    and ``entry_points_sha256`` pins a deterministic digest of the resolved
    entry-point files (empty when none exist on disk)."""

    schema_version: int
    name: str
    version: str
    kind: str
    description: str
    capabilities: tuple[str, ...]
    entry_points: dict[str, str]
    min_pedacito_version: str = "0.4.0"
    sha256: str = ""
    entry_points_sha256: str = ""


def _manifest_path(path: Path | str) -> Path:
    """Resolve a directory (containing the manifest) or the manifest itself."""
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / MANIFEST_FILENAME
    return candidate


def _raw_bytes(path: Path | str) -> bytes:
    return _manifest_path(path).read_bytes()


def sha256_manifest(path: Path | str) -> str:
    """SHA-256 of the raw manifest bytes (empty string when unreadable)."""
    try:
        return hashlib.sha256(_raw_bytes(path)).hexdigest()
    except OSError:
        return ""


def _invalid_entry_point(value: str) -> str:
    """Return why ``value`` is unsafe as an entry point, or ``""``."""
    if not value or not value.strip():
        return "entry point path is empty"
    text = value.replace("\\", "/")
    if _DRIVE_RE.match(value) or text.startswith("/"):
        return "entry point must be a relative path"
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        return "entry point may not traverse outside the package (..)"
    return ""


def validate_manifest(data: dict[str, Any]) -> list[str]:
    """Return every validation error for a parsed manifest (empty = valid)."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["manifest must be a TOML table"]

    schema_version = data.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        errors.append("schema_version must be the integer 1")
    elif schema_version != MANIFEST_SCHEMA_VERSION:
        errors.append(
            f"schema_version {schema_version} is unsupported; expected "
            f"{MANIFEST_SCHEMA_VERSION}")

    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        errors.append("name must be a non-empty string")
    elif not _NAME_RE.match(name.strip()):
        errors.append(
            f"name {name!r} is invalid; use letters, digits, '.', '_' or '-'")

    version = data.get("version")
    if not isinstance(version, str) or not version.strip():
        errors.append("version must be a non-empty string")

    kind = data.get("kind")
    if not isinstance(kind, str) or kind.strip() not in PACKAGE_KINDS:
        errors.append(
            f"kind must be one of {list(PACKAGE_KINDS)}; got {kind!r}")

    description = data.get("description", "")
    if description is None:
        description = ""
    if not isinstance(description, str):
        errors.append("description must be a string")

    capabilities = data.get("capabilities", [])
    if capabilities is None:
        capabilities = []
    if not isinstance(capabilities, (list, tuple)):
        errors.append("capabilities must be an array of strings")
    else:
        seen: set[str] = set()
        for cap in capabilities:
            if not isinstance(cap, str):
                errors.append(f"capability {cap!r} must be a string")
                continue
            if cap not in DECLARED_CAPABILITIES:
                errors.append(
                    f"unknown capability {cap!r}; expected one of "
                    f"{list(DECLARED_CAPABILITIES)}")
            elif cap in seen:
                errors.append(f"capability {cap!r} is declared twice")
            seen.add(cap)

    entry_points = data.get("entry_points", {})
    if entry_points is None:
        entry_points = {}
    if not isinstance(entry_points, dict):
        errors.append("entry_points must be a table of name = \"path\"")
    else:
        for key, value in entry_points.items():
            if not isinstance(key, str) or not key.strip():
                errors.append(f"entry_points key {key!r} must be a string")
            if not isinstance(value, str):
                errors.append(f"entry_points.{key} must be a string path")
                continue
            reason = _invalid_entry_point(value)
            if reason:
                errors.append(f"entry_points.{key}: {reason}")

    minimum = data.get("min_pedacito_version", "0.4.0")
    if minimum is not None and not isinstance(minimum, str):
        errors.append("min_pedacito_version must be a string")
    return errors


def _resolve_entry_point(pkg_root: Path, rel: str) -> Path | None:
    """Resolve an entry-point path inside the package root.

    Returns the resolved path when the file exists on disk, or ``None`` when
    it does not (minimal test manifests do not create dummy entry-point
    files). Raises ``ValueError`` for a traversal or a symlink that escapes
    the package root."""
    from . import security
    try:
        resolved = security.resolve_safe_path(pkg_root, rel)
    except (security.SecurityError, OSError, ValueError, RuntimeError) as exc:
        raise ValueError(
            f"entry point {rel!r} escapes the package: {exc}") from exc
    return resolved if resolved.is_file() else None


def _entry_points_digest(pkg_root: Path, entry_points: dict[str, str]) -> str:
    """A deterministic SHA-256 over the resolved entry-point files' bytes.

    Only entry points whose files exist on disk are hashed (in sorted name
    order); the digest is empty when there are no such files, so a manifest
    whose entry points are not present still loads and trusts cleanly. A
    symlink escape is rejected by `_resolve_entry_point`."""
    if not entry_points:
        return ""
    digest = hashlib.sha256()
    any_file = False
    for name in sorted(entry_points):
        rel = str(entry_points[name])
        target = _resolve_entry_point(pkg_root, rel)
        if target is None:
            continue
        try:
            data = target.read_bytes()
        except OSError:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(rel.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(data)
        digest.update(b"\x00")
        any_file = True
    return digest.hexdigest() if any_file else ""


def load_manifest(path: Path | str) -> PackageManifest:
    """Read, validate and return one manifest (raises ``ValueError``).

    ``path`` may be the package directory or the manifest file itself.
    """
    manifest_path = _manifest_path(path)
    try:
        raw = manifest_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {manifest_path}: {exc}") from exc
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"invalid TOML in {manifest_path}: {exc}") from exc
    errors = validate_manifest(data)
    if errors:
        raise ValueError(
            f"invalid package manifest {manifest_path}: " + "; ".join(errors))
    capabilities = data.get("capabilities") or []
    entry_points = {str(k): str(v) for k, v in (data.get("entry_points") or {}).items()}
    return PackageManifest(
        schema_version=int(data["schema_version"]),
        name=str(data["name"]).strip(),
        version=str(data["version"]).strip(),
        kind=str(data["kind"]).strip(),
        description=str(data.get("description") or ""),
        capabilities=tuple(str(cap) for cap in capabilities),
        entry_points=entry_points,
        min_pedacito_version=str(
            data.get("min_pedacito_version") or "0.4.0"),
        sha256=hashlib.sha256(raw).hexdigest(),
        entry_points_sha256=_entry_points_digest(
            manifest_path.parent, entry_points),
    )


# --------------------------------------------------------------------------- #
# trust pinning                                                               #
# --------------------------------------------------------------------------- #

def _trust_store() -> Path:
    """The per-user trust store path, resolved at call time (tests patch it)."""
    from . import config as config_mod
    return Path(config_mod.USER_STATE_DIR) / TRUST_FILENAME


def _path_key(path: Path | str) -> str:
    """Stable identity for a manifest path in the trust store."""
    try:
        return str(_manifest_path(path).resolve())
    except OSError:
        return str(_manifest_path(path))


def _read_store() -> dict[str, Any]:
    try:
        data = json.loads(_trust_store().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_store(data: dict[str, Any]) -> None:
    from . import fileio
    store = _trust_store()
    store.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=1, sort_keys=True).encode("utf-8")
    fileio.write_bytes(store, payload)


def is_manifest_trusted(path: Path | str) -> bool:
    """True only when the manifest's current bytes AND its resolved entry-point
    files match the pinned digests (verified from a single consistent read)."""
    key = _path_key(path)
    data = _read_store()
    entries = data.get("packages")
    if not isinstance(entries, dict):
        return False
    entry = entries.get(key)
    if not isinstance(entry, dict):
        return False
    pinned = str(entry.get("sha256") or "")
    if not pinned:
        return False
    try:
        manifest = load_manifest(path)     # one read, both digests
    except ValueError:
        return False
    if manifest.sha256 != pinned:
        return False
    pinned_eps = str(entry.get("entry_points_sha256") or "")
    return manifest.entry_points_sha256 == pinned_eps


def trust_manifest(path: Path | str) -> Path:
    """Pin the manifest's SHA-256 (and its entry-point files' digest) in the
    per-user trust store."""
    manifest = load_manifest(path)          # validate before trusting
    manifest_path = _manifest_path(path)
    data = _read_store()
    entries = data.get("packages")
    if not isinstance(entries, dict):
        entries = {}
    entries[_path_key(manifest_path)] = {
        "sha256": manifest.sha256,
        "entry_points_sha256": manifest.entry_points_sha256,
        "name": manifest.name,
        "version": manifest.version,
        "kind": manifest.kind,
    }
    data = {"schema_version": 1, "packages": entries}
    _write_store(data)
    return manifest_path


def untrust_manifest(path: Path | str) -> bool:
    """Drop a pinned manifest; returns whether an entry existed."""
    data = _read_store()
    entries = data.get("packages")
    if not isinstance(entries, dict):
        return False
    key = _path_key(path)
    if key not in entries:
        return False
    entries.pop(key, None)
    _write_store({"schema_version": 1, "packages": entries})
    return True


__all__ = [
    "DECLARED_CAPABILITIES",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "PACKAGE_KINDS",
    "PackageManifest",
    "is_manifest_trusted",
    "load_manifest",
    "sha256_manifest",
    "trust_manifest",
    "untrust_manifest",
    "validate_manifest",
]
