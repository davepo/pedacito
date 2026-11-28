"""Offline, versioned public-API index for the gather loop.

    from . import api_oracle

    environment = env_resolve.resolve_environment(root)
    index = api_oracle.index_api(
        root, "requests", environment=environment, cfg=cfg,
        deadline=time.monotonic() + 5)
    hits = api_oracle.lookup_api(index, "Session", limit=8)

The default path is **static**: it reads Python ``.pyi``/``.py`` files and
distribution metadata from the resolved interpreter's ``site-packages`` (or the
project itself), and ``.d.ts`` declaration files from a contained
``node_modules``. It never imports package code, runs a package script, or
touches the network. That makes it safe to call while assembling a prompt.

Execution-based enrichment is opt-in and double-gated by
``config.api_oracle_exec`` (an ``EXEC_FIELDS`` setting) *and* the existing
execution consent (``allow_shell`` plus the unified policy engine):

* Go: ``go doc -all PACKAGE``;
* Rust: ``cargo metadata --offline --no-deps --format-version 1``;
* Python: the resolved interpreter's ``inspect``, only when no static
  declaration produced symbols.

Those children are argv-only (``shell=False``), bounded by the caller's
monotonic deadline, capped in output, run with the scrubbed environment from
:func:`env_resolve.command_env`, and labelled in the result as execution
sources. ``api_oracle_exec = false`` (the default) reports ``unknown`` for
Go/Rust/Python-inspect instead of guessing. A missing package, a static miss or
an unknown result is never proof that a dynamic API does not exist: the tool
renderer says so in words.

The cache is a keyed JSON artifact under ``.pedacito/cache/api/``. Its key
includes the package, the resolved environment identity, the version, and
content hashes of the declaration and lock files that were inspected. Writes
use the existing atomic ``fileio.write_bytes`` replacement, so concurrent
processes see a complete old or new artifact, never a partial one; a changed
version or declaration simply misses the old key and prunes it. Records hold
only package/symbol metadata (never raw source, prompts, tool output or
credentials), are secret-redacted, and are capped in bytes and item count.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import env_resolve, fileio, policy, proc, sandbox, security

__all__ = [
    "ApiIndex",
    "ApiSymbol",
    "index_api",
    "lookup_api",
    "match_named_library",
    "named_libraries",
    "render_result",
]

_CACHE_SCHEMA = 1
_CACHE_REL = Path(".pedacito") / "cache" / "api"
_MAX_CACHE_BYTES = 512_000
_MAX_CACHE_SYMBOLS = 900
_MAX_CACHE_FILES_PER_PACKAGE = 6
_MAX_FILE_BYTES = 1_000_000
_MAX_HASH_BYTES = 512_000
_MAX_METADATA_BYTES = 64_000
_MAX_SYMBOLS = 1_200
_MAX_SYMBOLS_PER_FILE = 120
_MAX_SIGNATURE_CHARS = 300
_MAX_WARNINGS = 10
_MAX_WARNING_CHARS = 240
_MAX_SOURCE_FILES = 320
_MAX_EXEC_OUTPUT = 24_000
_MAX_EXEC_SECONDS = 12.0
_MAX_LOOKUP_LIMIT = 50
_EXEC_SOURCES = frozenset({"go-doc", "cargo-metadata", "python-inspect"})

_PACKAGE_RE = re.compile(r"^@?[A-Za-z0-9][A-Za-z0-9._@/+~-]*$")
_GO_REQUIRE_RE = re.compile(r"^\s*([A-Za-z0-9._~-]+\.[A-Za-z0-9._~/-]+)\s+v\S+")
_GO_FUNC_RE = re.compile(
    r"^func\s+\(\s*\w+\s+\*?([A-Za-z_]\w*)(?:\[[^\]]*\])?\s*\)\s+"
    r"([A-Za-z_]\w*)\s*(\(.*)$")
_GO_PLAIN_FUNC_RE = re.compile(r"^func\s+([A-Za-z_]\w*)\s*(\(.*)$")
_GO_TYPE_RE = re.compile(r"^type\s+([A-Za-z_]\w*)\s*(.*)$")
_TS_NAMES_RE = re.compile(
    r"^\s*(?:export\s+(?:default\s+)?)?(?:declare\s+)?"
    r"(function|class|interface|type|const|let|var|enum|namespace|module)\s+"
    r"([A-Za-z_$][\w$]*)")
_TS_EXPORT_LIST_RE = re.compile(r"^\s*export\s*\{([^}]*)\}\s*;?\s*$")

# Dependency declarations are read to decide which libraries a task may ask
# the oracle about. Parsing them never executes a manager or a script.
_REQ_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_EXPLICIT_BACKTICK_RE = re.compile(r"`([^`\n]{1,120})`")
_EXPLICIT_AT_RE = re.compile(r"(?<![\w.])@([A-Za-z0-9][A-Za-z0-9._/-]{1,120})")
_TASK_TOKEN_RE = re.compile(r"[A-Za-z0-9_@./+~-]+")
_RESERVED_LIBRARY_NAMES = frozenset({
    "action", "api", "app", "append", "browser", "callers", "chat", "class",
    "code", "command", "content", "context", "delete", "diagnose", "diff",
    "done", "file", "find", "function", "git", "grep", "import", "index",
    "lsp", "main", "mcp", "module", "move", "outline", "package", "pattern",
    "project", "read", "remember", "rename", "run", "screenshot", "serve",
    "sessions", "skill", "symbol", "task", "test", "tests", "todo", "tool",
    "undo", "usages", "write",
})

_PY_INSPECT_SCRIPT = """\
import importlib, inspect, json, sys
name = sys.argv[1]
out = {"ok": False, "symbols": [], "version": "", "error": ""}
try:
    mod = importlib.import_module(name)
except BaseException as exc:
    out["error"] = type(exc).__name__
    print(json.dumps(out))
    raise SystemExit(0)
try:
    from importlib import metadata
    out["version"] = str(metadata.version(name))
except BaseException:
    pass
count = 0
for member_name, member in inspect.getmembers(mod):
    if member_name.startswith("_") or count >= 200:
        continue
    try:
        if inspect.isfunction(member) or inspect.isbuiltin(member):
            kind = "function"
            signature = str(inspect.signature(member))
        elif inspect.isclass(member):
            kind = "class"
            signature = str(inspect.signature(member))
        else:
            continue
    except BaseException:
        continue
    out["symbols"].append({"name": member_name, "kind": kind,
                           "signature": signature})
    count += 1
out["ok"] = True
print(json.dumps(out))
"""


@dataclass(frozen=True)
class ApiSymbol:
    """One public symbol. `source` is an ASCII protocol label, not a path the
    model should try to read."""

    package: str
    version: str
    name: str
    signature: str
    source: str


@dataclass(frozen=True)
class ApiIndex:
    """The result of one package lookup. `status` is render-only ASCII:
    ``ok``, ``unknown``, ``unavailable``, ``denied`` or ``timeout``."""

    status: str
    symbols: tuple[ApiSymbol, ...]
    cache_key: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _PythonPlan:
    found: bool
    version: str
    files: tuple[tuple[str, str, str], ...]   # (path, module name, kind)
    metadata_files: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _TsPlan:
    found: bool
    version: str
    files: tuple[tuple[str, str], ...]        # (path, module name)
    metadata_files: tuple[str, ...]
    warnings: tuple[str, ...]


# ---------------------------------------------------------------------------
# Small text/path helpers
# ---------------------------------------------------------------------------

def _bounded(text, limit: int) -> str:
    value = " ".join(str(text if text is not None else "").split())
    return value[:limit]


def _clip(text, limit: int) -> str:
    """Cap rendered text while preserving its line structure."""
    value = str(text if text is not None else "").rstrip()
    if len(value) <= limit:
        return value
    cut = value[:limit]
    newline = cut.rfind("\n")
    if newline > limit // 2:
        cut = cut[:newline]
    return cut.rstrip() + "\n[truncated]"


def _clean(text, limit: int) -> str:
    """Redact secrets, collapse whitespace and cap one stored string."""
    try:
        scrubbed = security.redact_secrets_in_text(str(text or ""))
    except Exception:
        scrubbed = str(text or "")
    return _bounded(scrubbed, limit)


class _Warnings:
    def __init__(self) -> None:
        self.items: list[str] = []

    def add(self, text) -> None:
        value = _clean(text, _MAX_WARNING_CHARS)
        if value and value not in self.items and len(self.items) < _MAX_WARNINGS:
            self.items.append(value)


def _valid_package(package) -> str:
    raw = str(package or "").strip().strip("`\"'")
    if not raw or len(raw) > 160:
        return ""
    if "\\" in raw or ":" in raw or raw.startswith("/") or raw.endswith("/"):
        return ""
    parts = raw.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return ""
    if not _PACKAGE_RE.fullmatch(raw):
        return ""
    if raw.casefold() in _RESERVED_LIBRARY_NAMES:
        return ""
    return raw


def _expired(deadline) -> bool:
    try:
        return deadline is not None and float(deadline) <= time.monotonic()
    except (TypeError, ValueError):
        return False


def _deadline_seconds(deadline, default: float = 8.0) -> float:
    if deadline is None:
        return default
    try:
        return max(0.0, min(default, float(deadline) - time.monotonic()))
    except (TypeError, ValueError):
        return default


def _safe_child(base: Path, rel: str) -> Path | None:
    try:
        return security.resolve_safe_path(base, rel)
    except (security.SecurityError, OSError, ValueError):
        return None


def _read_bounded(path: Path, limit: int = _MAX_FILE_BYTES) -> str:
    try:
        with path.open("rb") as handle:
            data = handle.read(limit + 1)
    except OSError:
        return ""
    if len(data) > limit:
        data = data[:limit]
    text = data.decode("utf-8", "replace")
    return text[1:] if text.startswith("\ufeff") else text


def _file_bytes_hash(path: Path, limit: int = _MAX_HASH_BYTES) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            data = handle.read(limit + 1)
    except OSError:
        return "missing"
    if len(data) > limit:
        data = data[:limit]
    return (hashlib.sha256(data).hexdigest()
            + f":{size}")


def _is_link(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        return security.is_junction_or_reparse(path)
    except OSError:
        return True


def _walk_files(base: Path, pattern: str, *, deadline,
                limit: int = _MAX_SOURCE_FILES) -> list[Path]:
    """Deterministic recursive walk that never follows a link."""
    out: list[Path] = []
    stack = [base]
    while stack and len(out) < limit:
        if _expired(deadline):
            break
        directory = stack.pop()
        try:
            entries = sorted(directory.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        for entry in entries:
            if len(out) >= limit:
                break
            if _is_link(entry):
                continue
            try:
                if entry.is_dir():
                    if entry.name in ("__pycache__", ".git", "node_modules"):
                        continue
                    stack.append(entry)
                elif entry.is_file() and entry.match(pattern):
                    out.append(entry)
            except OSError:
                continue
    out.sort(key=lambda p: p.as_posix().casefold())
    return out


def _norm_dist(name) -> str:
    return re.sub(r"[-_.]+", "_", str(name or "").strip()).casefold()


def _metadata_header(text: str, key: str) -> str:
    for line in text.splitlines():
        if line.lower().startswith(key.lower() + ":"):
            return line.split(":", 1)[1].strip()
        if not line.strip():
            break
    return ""


def _source_rel(root: Path, path: Path, kind: str) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        rel = path.name
    return f"{kind}:{rel}"


# ---------------------------------------------------------------------------
# Lock files and the cache key
# ---------------------------------------------------------------------------

_LOCK_NAMES = (
    "pyproject.toml", "setup.cfg", "setup.py", "Pipfile", "Pipfile.lock",
    "uv.lock", "poetry.lock", "package.json", "package-lock.json",
    "npm-shrinkwrap.json", "pnpm-lock.yaml", "pnpm-workspace.yaml",
    "yarn.lock", "bun.lockb", "bun.lock", "Cargo.toml", "Cargo.lock",
    "go.mod", "go.sum",
)


def _lock_files(root: Path) -> tuple[Path, ...]:
    out: list[Path] = []
    for name in _LOCK_NAMES:
        safe = _safe_child(root, name)
        if safe is not None and safe.is_file() and not _is_link(safe):
            out.append(safe)
    for pattern in ("requirements*.txt", "requirements/*.txt"):
        for path in sorted(root.glob(pattern))[:20]:
            if _is_link(path) or not path.is_file():
                continue
            safe = _safe_child(root, path.relative_to(root).as_posix())
            if safe is not None and safe not in out:
                out.append(safe)
    return tuple(out)


def _environment_identity(environment: env_resolve.ResolvedEnvironment) -> str:
    def norm(path) -> str:
        if path is None:
            return ""
        try:
            return os.path.normcase(os.path.abspath(str(path)))
        except (OSError, ValueError):
            return str(path)

    parts = [
        "v1",
        str(environment.kind),
        str(environment.package_manager),
        norm(environment.python),
        "|".join(norm(p) for p in environment.bin_dirs),
        norm(environment.workspace_root),
        norm(environment.root),
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _fingerprint_key(root: Path, package: str, environment, version: str,
                     paths: tuple[Path, ...], *, extra: str = "") -> str:
    digest = hashlib.sha256()
    for part in ("v1", package, _environment_identity(environment),
                 str(version or ""), str(extra or "")):
        digest.update(part.encode("utf-8", "replace"))
        digest.update(b"\x1f")
    seen: set[str] = set()
    for path in sorted(paths, key=lambda p: os.path.normcase(str(p))):
        try:
            marker = os.path.normcase(os.path.abspath(str(path)))
        except (OSError, ValueError):
            marker = str(path)
        if marker in seen:
            continue
        seen.add(marker)
        digest.update(marker.encode("utf-8", "replace"))
        digest.update(b"\x00")
        digest.update(_file_bytes_hash(path).encode("ascii"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def _cache_path(root: Path, package: str, key: str) -> Path | None:
    safe = re.sub(r"[^a-z0-9]+", "-", package.casefold()).strip("-")[:40]
    safe = safe or "package"
    rel = (_CACHE_REL / f"{safe}-{key[:24]}.json").as_posix()
    path = _safe_child(root, rel)
    if path is None:
        return None
    try:
        path.parent.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return path


def _cache_read(root: Path, package: str, key: str) -> ApiIndex | None:
    path = _cache_path(root, package, key)
    if path is None or not path.is_file() or _is_link(path):
        return None
    try:
        data = json.loads(path.read_bytes()[:_MAX_CACHE_BYTES].decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("schema") != _CACHE_SCHEMA or data.get("cache_key") != key \
            or data.get("package") != package:
        return None
    status = str(data.get("status") or "")
    if status != "ok":
        return None
    symbols: list[ApiSymbol] = []
    for raw in data.get("symbols") or []:
        if not isinstance(raw, dict) or len(symbols) >= _MAX_CACHE_SYMBOLS:
            break
        name = _clean(raw.get("name"), 240)
        signature = _clean(raw.get("signature"), _MAX_SIGNATURE_CHARS)
        source = _clean(raw.get("source"), 160)
        version = _clean(raw.get("version"), 80)
        if not name or not source:
            continue
        symbols.append(ApiSymbol(
            package=package, version=version, name=name,
            signature=signature, source=source))
    warnings = tuple(_clean(w, _MAX_WARNING_CHARS)
                     for w in (data.get("warnings") or ())[:_MAX_WARNINGS]
                     if _clean(w, _MAX_WARNING_CHARS))
    return ApiIndex(status="ok", symbols=tuple(symbols), cache_key=key,
                    warnings=warnings)


def _cache_write(root: Path, package: str, key: str, index: ApiIndex) -> None:
    if index.status != "ok":
        return
    symbols = []
    for symbol in index.symbols[:_MAX_CACHE_SYMBOLS]:
        symbols.append({
            "name": _clean(symbol.name, 240),
            "signature": _clean(symbol.signature, _MAX_SIGNATURE_CHARS),
            "source": _clean(symbol.source, 160),
            "version": _clean(symbol.version, 80),
        })
    warnings = [_clean(w, _MAX_WARNING_CHARS)
                for w in index.warnings[:_MAX_WARNINGS]]
    payload = {
        "schema": _CACHE_SCHEMA,
        "package": package,
        "cache_key": key,
        "status": "ok",
        "symbols": symbols,
        "warnings": warnings,
    }
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    while len(raw) > _MAX_CACHE_BYTES and payload["symbols"]:
        payload["symbols"] = payload["symbols"][:-max(1, len(payload["symbols"]) // 8)]
        payload["warnings"] = (payload["warnings"] + [
            "cache payload trimmed to the byte cap"])[-_MAX_WARNINGS:]
        raw = json.dumps(payload, ensure_ascii=True, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    if len(raw) > _MAX_CACHE_BYTES:
        return
    path = _cache_path(root, package, key)
    if path is None:
        return
    try:
        fileio.write_bytes(path, raw)
    except OSError:
        return
    _prune_cache(path, package)


def _prune_cache(current: Path, package: str) -> None:
    safe = re.sub(r"[^a-z0-9]+", "-", package.casefold()).strip("-")[:40]
    safe = safe or "package"
    try:
        entries = sorted(current.parent.iterdir(), key=lambda p: p.name)
    except OSError:
        return
    keep = {current.name}
    for entry in entries:
        if len(keep) >= _MAX_CACHE_FILES_PER_PACKAGE:
            break
        if not entry.name.startswith(safe + "-") or entry.name in keep:
            continue
        if not entry.name.endswith(".json") or _is_link(entry):
            continue
        keep.add(entry.name)
    for entry in entries:
        if entry.name.startswith(safe + "-") and entry.name not in keep:
            if entry.name.endswith(".json") and not _is_link(entry):
                try:
                    entry.unlink()
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Static Python declarations
# ---------------------------------------------------------------------------

def _python_module_rel(package: str) -> str:
    return package.replace(".", "/")


def _python_roots(root: Path, environment) -> tuple[Path, ...]:
    roots = list(env_resolve.dependency_roots(environment))
    for extra in (root, root / "src"):
        try:
            safe = extra.resolve()
        except OSError:
            continue
        if safe.is_dir() and safe not in roots:
            roots.append(safe)
    return tuple(roots)


def _dist_metadata(package: str, environment, deadline,
                   warnings: _Warnings) -> tuple[str, tuple[str, ...],
                                                 tuple[str, ...]]:
    """(version, top-level module names, metadata files) for one distribution."""
    targets = {_norm_dist(package), _norm_dist(package.split(".")[0])}
    version = ""
    top_levels: list[str] = []
    metadata_files: list[str] = []
    matches: list[tuple[str, str, str, str]] = []
    for site in env_resolve.site_packages_dirs(environment):
        for pattern in ("*.dist-info", "*.egg-info"):
            try:
                entries = sorted(site.glob(pattern), key=lambda p: p.name)
            except OSError:
                continue
            for entry in entries[:2000]:
                if _expired(deadline):
                    break
                if _is_link(entry) or not entry.is_dir():
                    continue
                entry_name = entry.name
                for suffix in (".dist-info", ".egg-info"):
                    if entry_name.lower().endswith(suffix):
                        entry_name = entry_name[:-len(suffix)]
                entry_name = re.sub(r"-[0-9].*$", "", entry_name)
                # The directory prefix already names the distribution in
                # every real wheel/egg; skip before reading METADATA so a
                # 1,000-package environment costs one directory listing, not
                # 1,000 small file reads.
                if _norm_dist(entry_name) not in targets:
                    continue
                meta = entry / ("METADATA" if entry.suffix == ".dist-info"
                                else "PKG-INFO")
                if not meta.is_file() or _is_link(meta):
                    continue
                text = _read_bounded(meta, _MAX_METADATA_BYTES)
                name = _metadata_header(text, "Name")
                if _norm_dist(name) not in targets:
                    continue
                matches.append((str(site), entry.as_posix(), name,
                                _metadata_header(text, "Version")))
                metadata_files.append(str(meta))
                top = entry / "top_level.txt"
                if top.is_file() and not _is_link(top):
                    for line in _read_bounded(top, 16_000).splitlines()[:40]:
                        mod = line.strip()
                        if re.fullmatch(r"[A-Za-z_]\w*", mod) \
                                and mod not in top_levels:
                            top_levels.append(mod)
                break
            if matches:
                break
        if matches:
            break
    if matches:
        _site, _entry, _name, version = sorted(matches)[0]
    if not top_levels and version:
        top_levels.append(package.split(".")[0])
    return version, tuple(top_levels), tuple(metadata_files)


def _python_plan(root: Path, package: str, environment, deadline,
                 warnings: _Warnings) -> _PythonPlan:
    if "/" in package or "@" in package:
        return _PythonPlan(False, "", (), (), ())
    version, top_levels, meta_files = _dist_metadata(
        package, environment, deadline, warnings)
    roots = _python_roots(root, environment)
    rel = _python_module_rel(package)
    top_rel = _python_module_rel(package.split(".")[0])
    candidates: list[tuple[Path, str]] = []   # (path, module name)
    seen: set[str] = set()

    def add_candidate(path: Path, module: str) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            return
        marker = os.path.normcase(str(resolved))
        if marker in seen or not resolved.exists():
            return
        seen.add(marker)
        candidates.append((resolved, module))

    for base in roots:
        for prefix in (rel, f"src/{rel}"):
            for suffix in (".pyi", ".py"):
                safe = _safe_child(base, prefix + suffix)
                if safe is not None and safe.is_file() and not _is_link(safe):
                    add_candidate(safe, package)
            safe_dir = _safe_child(base, prefix)
            if safe_dir is not None and safe_dir.is_dir() \
                    and not _is_link(safe_dir):
                stubs = _walk_files(safe_dir, "*.pyi", deadline=deadline)
                pyfiles = _walk_files(safe_dir, "*.py", deadline=deadline)
                shadowed = set()
                if stubs:
                    for stub in stubs:
                        try:
                            shadowed.add(
                                stub.with_suffix(".py").relative_to(safe_dir))
                        except ValueError:
                            continue
                for found in stubs + [
                        p for p in pyfiles
                        if p.relative_to(safe_dir) not in shadowed]:
                    try:
                        rel_file = found.relative_to(safe_dir)
                    except ValueError:
                        continue
                    if rel_file.name == "__init__.pyi" \
                            or rel_file.name == "__init__.py":
                        parts = list(rel_file.parent.parts)
                        module = ".".join([package, *parts]) if parts else package
                    else:
                        parts = list(rel_file.with_suffix("").parts)
                        module = ".".join([package, *parts])
                    add_candidate(found, module)
    # A distribution's top level may be a namespace package whose requested
    # dotted path was not found; look at the top-level name as a fallback.
    if not candidates and top_levels and not version:
        for base in roots:
            safe_dir = _safe_child(base, top_rel)
            if safe_dir is not None and safe_dir.is_dir() \
                    and not _is_link(safe_dir):
                for found in _walk_files(safe_dir, "*.py", deadline=deadline)[:40]:
                    add_candidate(found, package.split(".")[0])
                break

    files: list[tuple[str, str, str]] = []
    for path, module in sorted(candidates, key=lambda p: p[1].casefold()):
        kind = "python-pyi" if path.suffix == ".pyi" else "python-py"
        files.append((str(path), module, kind))
    found = bool(files or version)
    return _PythonPlan(found, version, tuple(files), meta_files,
                       tuple(warnings.items))


def _py_annotation(node) -> str:
    if node is None:
        return ""
    try:
        text = ast.unparse(node)
    except Exception:
        text = "..."
    return _bounded(text, 160)


def _py_signature(args: ast.arguments, *, name: str = "",
                  drop_first: bool = False, returns=None,
                  is_async: bool = False) -> str:
    pos = list(args.posonlyargs) + list(args.args)
    defaults = list(args.defaults)
    first_default = len(pos) - len(defaults)
    if drop_first and pos and pos[0].arg in ("self", "cls"):
        pos = pos[1:]
        first_default -= 1
    pieces: list[str] = []
    for index, arg in enumerate(pos):
        piece = arg.arg
        if arg.annotation is not None:
            piece += ": " + _py_annotation(arg.annotation)
        if first_default >= 0 and index >= first_default:
            default_index = index - first_default
            if 0 <= default_index < len(defaults):
                piece += " = " + _py_annotation(defaults[default_index])
        pieces.append(piece)
    if args.vararg is not None:
        piece = "*" + args.vararg.arg
        if args.vararg.annotation is not None:
            piece += ": " + _py_annotation(args.vararg.annotation)
        pieces.append(piece)
    elif args.kwonlyargs:
        pieces.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        piece = arg.arg
        if arg.annotation is not None:
            piece += ": " + _py_annotation(arg.annotation)
        if default is not None:
            piece += " = " + _py_annotation(default)
        pieces.append(piece)
    if args.kwarg is not None:
        piece = "**" + args.kwarg.arg
        if args.kwarg.annotation is not None:
            piece += ": " + _py_annotation(args.kwarg.annotation)
        pieces.append(piece)
    prefix = "async def " if is_async else "def "
    signature = f"{prefix}{name}({', '.join(pieces)})"
    result = _py_annotation(returns)
    if result:
        signature += " -> " + result
    return _bounded(signature, _MAX_SIGNATURE_CHARS)


def _python_symbols(plan: _PythonPlan, package: str, root: Path, deadline,
                    warnings: _Warnings) -> list[ApiSymbol]:
    symbols: list[ApiSymbol] = []
    for raw_path, module, kind in plan.files:
        if len(symbols) >= _MAX_SYMBOLS or _expired(deadline):
            break
        path = Path(raw_path)
        # Source labels are made relative to the project when possible and to
        # the package file name otherwise; never leak an absolute home path.
        label_root = root if path.is_relative_to(root) else path.parent
        source = _source_rel(label_root, path, kind)
        text = _read_bounded(path)
        if not text:
            warnings.add(f"unreadable declaration: {source}")
            continue
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError, RecursionError):
            warnings.add(f"unparseable declaration: {source}")
            continue
        file_symbols = _py_symbols_from_text(
            tree, module, package, plan.version, source)
        symbols.extend(file_symbols)
    return symbols[:_MAX_SYMBOLS]


def _py_symbols_from_text(tree, module, package, version, source):
    """AST walker used by `_python_symbols`; kept separate so the source label
    is fixed before parsing (no cwd-relative surprise in cache entries)."""
    out: list[ApiSymbol] = []

    def add(name: str, signature: str) -> None:
        if len(out) >= _MAX_SYMBOLS_PER_FILE:
            return
        out.append(ApiSymbol(
            package=package, version=version,
            name=f"{module}.{name}" if module else name,
            signature=_bounded(signature, _MAX_SIGNATURE_CHARS),
            source=source))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            add(node.name, _py_signature(
                node.args, name=node.name, returns=node.returns,
                is_async=isinstance(node, ast.AsyncFunctionDef)))
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            bases = ", ".join(_py_annotation(base) for base in node.bases)
            decorators = " ".join("@" + _py_annotation(d)
                                  for d in node.decorator_list)
            head = (decorators + " " if decorators else "") \
                + f"class {node.name}" + (f"({bases})" if bases else "")
            add(node.name, head)
            for child in node.body:
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if child.name.startswith("_") and child.name != "__init__":
                    continue
                if child.name == "__init__":
                    add(f"{node.name}.__init__", _py_signature(
                        child.args, name="__init__", drop_first=True,
                        returns=None,
                        is_async=isinstance(child, ast.AsyncFunctionDef)))
                    continue
                is_property = any(
                    isinstance(item, ast.Name) and item.id == "property"
                    for item in child.decorator_list)
                signature = "property" if is_property else _py_signature(
                    child.args, name=child.name, drop_first=True,
                    returns=child.returns,
                    is_async=isinstance(child, ast.AsyncFunctionDef))
                add(f"{node.name}.{child.name}", signature)
    return out


# ---------------------------------------------------------------------------
# Static TypeScript declaration files
# ---------------------------------------------------------------------------

def _ts_package_plan(root: Path, package: str, deadline,
                     warnings: _Warnings) -> _TsPlan:
    if not package or ".." in package.split("/"):
        return _TsPlan(False, "", (), (), ())
    candidates: list[Path] = []
    for base in (root, root / "src"):
        try:
            base = base.resolve()
        except OSError:
            continue
        if not base.is_dir():
            continue
        for rel in (f"node_modules/{package}",
                    f"node_modules/@types/{package}"):
            safe = _safe_child(base, rel)
            if safe is not None and safe.is_dir() and not _is_link(safe):
                candidates.append(safe)
    if not candidates:
        return _TsPlan(False, "", (), (), ())
    package_dir = sorted(candidates, key=lambda p: p.as_posix().casefold())[0]
    manifest = package_dir / "package.json"
    version = ""
    entry: Path | None = None
    metadata_files: list[str] = []
    data: dict = {}
    if manifest.is_file() and not _is_link(manifest):
        metadata_files.append(str(manifest))
        text = _read_bounded(manifest, _MAX_METADATA_BYTES)
        try:
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                data = loaded
        except ValueError:
            warnings.add("package.json is not valid JSON")
        version = str(data.get("version") or "").strip()
        raw_entry = data.get("types") or data.get("typings")
        if not raw_entry and isinstance(data.get("exports"), dict):
            exports = data["exports"]
            raw_entry = exports.get("types") or exports.get("typings")
            if not raw_entry and isinstance(exports.get("."), dict):
                raw_entry = exports["."].get("types")
        if isinstance(raw_entry, str):
            candidate = _safe_child(package_dir, raw_entry.lstrip("./"))
            if candidate is not None and candidate.is_file() \
                    and candidate.suffix == ".ts" and not _is_link(candidate):
                entry = candidate
    files: list[Path] = []
    if entry is not None:
        files = [entry]
    else:
        files = _walk_files(package_dir, "*.d.ts", deadline=deadline,
                            limit=200)
    file_pairs: list[tuple[str, str]] = []
    for path in files:
        try:
            rel = path.relative_to(package_dir)
        except ValueError:
            continue
        rel_text = rel.as_posix()
        stem = rel_text[:-5] if rel_text.endswith(".d.ts") \
            else rel.with_suffix("").as_posix()
        if stem.endswith("/index"):
            stem = stem[: -len("/index")]
        if stem in ("", "index"):
            module = package.split("/")[-1]
        else:
            module = stem.replace("/", ".")
        file_pairs.append((str(path), module))
    return _TsPlan(bool(file_pairs or version), version, tuple(file_pairs),
                   tuple(metadata_files), tuple(warnings.items))


def _strip_ts_comments(text: str) -> str:
    out: list[str] = []
    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if in_block:
            if "*/" in stripped:
                in_block = False
                line = stripped.split("*/", 1)[1]
            else:
                continue
        if stripped.startswith("/*"):
            if "*/" not in stripped:
                in_block = True
            continue
        if stripped.startswith("//"):
            continue
        out.append(line)
    return "\n".join(out)


def _ts_symbols_from_file(path: Path, module: str, package: str, version: str,
                          deadline) -> list[ApiSymbol]:
    if _expired(deadline):
        return []
    text = _strip_ts_comments(_read_bounded(path, _MAX_FILE_BYTES))
    if not text:
        return []
    try:
        rel = path.name
        for marker in ("node_modules/", "node_modules\\"):
            if marker in str(path):
                rel = str(path).split(marker, 1)[1].replace("\\", "/")
                break
    except Exception:
        rel = path.name
    source = f"ts-dts:{rel}"
    lines = text.splitlines()
    out: list[ApiSymbol] = []

    def add(name: str, signature: str) -> None:
        if len(out) >= _MAX_SYMBOLS_PER_FILE:
            return
        out.append(ApiSymbol(
            package=package, version=version,
            name=f"{module}.{name}" if module else name,
            signature=_bounded(signature, _MAX_SIGNATURE_CHARS),
            source=source))

    index = 0
    while index < len(lines):
        if len(out) >= _MAX_SYMBOLS_PER_FILE or _expired(deadline):
            break
        line = lines[index]
        export_list = _TS_EXPORT_LIST_RE.match(line)
        if export_list:
            for item in export_list.group(1).split(","):
                item = item.strip()
                if not item or item.startswith("type "):
                    continue
                if " as " in item:
                    item = item.split(" as ", 1)[1].strip()
                if re.fullmatch(r"[A-Za-z_$][\w$]*", item):
                    add(item, "re-export")
            index += 1
            continue
        match = _TS_NAMES_RE.match(line)
        if not match:
            index += 1
            continue
        kind, name = match.group(1), match.group(2)
        collected = [line.strip()]
        cursor = index
        first = collected[0]
        if not (first.endswith(";") or "{" in first or first.endswith("=")):
            while cursor + 1 < len(lines) and len(collected) < 12:
                cursor += 1
                nxt = lines[cursor].strip()
                if not nxt:
                    break
                collected.append(nxt)
                if nxt.endswith(";") or "{" in nxt or nxt.endswith("="):
                    break
        signature = " ".join(collected)
        signature = signature.split("{", 1)[0].strip()
        if kind in ("class", "interface", "enum", "namespace", "module"):
            add(name, f"{kind} {name}"
                + (signature.split(name, 1)[1]
                   if name in signature else ""))
        elif kind == "function":
            add(name, signature)
        elif kind == "type":
            add(name, signature)
        else:
            add(name, signature)
        index = max(index + 1, cursor + 1)
    return out


# ---------------------------------------------------------------------------
# Optional gated execution
# ---------------------------------------------------------------------------

def _exec_allowed(cfg, root: Path) -> tuple[bool, str]:
    if not bool(getattr(cfg, "api_oracle_exec", False)):
        return False, ("api_oracle_exec = false: Go/Rust/Python-inspect "
                       "enrichment is off")
    if not bool(getattr(cfg, "allow_shell", False)):
        return False, ("api_oracle_exec is on but allow_shell is off: "
                       "execution consent is missing")
    return True, ""


def _policy_error(cfg, root: Path, capability: str, target: str) -> str:
    try:
        decision = policy.enforce(cfg, capability, target, root=root)
    except Exception as exc:
        return f"policy evaluation error: {exc}"
    if decision is not None and decision.decision == "deny":
        return f"denied by policy ({decision.matched_rule}): {decision.reason}"
    return ""


def _run_exec(root: Path, argv: list[str], environment, cfg, deadline, *,
              extra_env: dict[str, str] | None = None):
    remaining = _deadline_seconds(deadline, _MAX_EXEC_SECONDS)
    if remaining <= 0:
        return None, "", "timeout"
    try:
        from . import env_resolve as env_mod
        child_env = env_mod.command_env(environment, cfg)
    except Exception:
        return None, "child environment could not be built", "error"
    if not isinstance(child_env, dict):
        return None, "child environment could not be built", "error"
    if extra_env:
        child_env = dict(child_env)
        child_env.update({str(k): str(v) for k, v in extra_env.items()})
    try:
        rc, out, stopped, _secs = proc.run(
            argv, cwd=root, timeout=remaining,
            idle_timeout=0, max_output=_MAX_EXEC_OUTPUT,
            env=child_env, windows_sandbox=sandbox.windows_sandbox_active(cfg),
            bin_dirs=environment.bin_dirs)
    except (OSError, ValueError) as exc:
        return None, str(exc), "error"
    if stopped in ("timeout", "cancelled"):
        return rc, out, "timeout"
    if stopped == "error":
        return rc, out, "error"
    return rc, out, ""


def _go_version(root: Path, package: str) -> str:
    manifest = _safe_child(root, "go.mod")
    if manifest is None or not manifest.is_file():
        return ""
    text = _read_bounded(manifest, _MAX_METADATA_BYTES)
    target_last = package.rsplit("/", 1)[-1]
    for line in text.splitlines():
        match = _GO_REQUIRE_RE.match(line)
        if not match:
            continue
        name = match.group(1)
        if name == package or name.rsplit("/", 1)[-1] == target_last:
            parts = line.split()
            if len(parts) >= 2:
                return parts[1].lstrip("v")
    return ""


def _go_symbols(root: Path, package: str, environment, cfg, deadline,
                warnings: _Warnings) -> tuple[list[ApiSymbol], str]:
    allowed, reason = _exec_allowed(cfg, root)
    if not allowed:
        return [], reason
    denied = _policy_error(cfg, root, "run", f"go doc -all {package}")
    if denied:
        return [], denied
    rc, out, stopped = _run_exec(
        root, ["go", "doc", "-all", package], environment, cfg, deadline,
        extra_env={"GOPROXY": "off", "GOFLAGS": "-mod=mod"})
    if stopped == "timeout":
        return [], "go doc timed out"
    if stopped == "error":
        return [], "go doc could not start"
    if rc not in (0, None):
        return [], f"go doc exited {rc}: {_bounded(out, 160)}"
    version = _go_version(root, package)
    package_name = package.rsplit("/", 1)[-1]
    symbols: list[ApiSymbol] = []
    seen: set[str] = set()
    for line in out.splitlines():
        text = line.strip()
        if not text or text.startswith("//"):
            continue
        match = _GO_FUNC_RE.match(text)
        if match:
            receiver, name, rest = match.groups()
            if receiver[:1].isupper() and name[:1].isupper():
                full = f"{package_name}.{receiver}.{name}"
                if full not in seen:
                    seen.add(full)
                    symbols.append(ApiSymbol(
                        package=package, version=version, name=full,
                        signature=_bounded(text, _MAX_SIGNATURE_CHARS),
                        source="go-doc"))
            continue
        match = _GO_PLAIN_FUNC_RE.match(text)
        if match:
            name, rest = match.groups()
            if name[:1].isupper():
                full = f"{package_name}.{name}"
                if full not in seen:
                    seen.add(full)
                    symbols.append(ApiSymbol(
                        package=package, version=version, name=full,
                        signature=_bounded(text, _MAX_SIGNATURE_CHARS),
                        source="go-doc"))
            continue
        match = _GO_TYPE_RE.match(text)
        if match:
            name = match.group(1)
            if name[:1].isupper():
                full = f"{package_name}.{name}"
                if full not in seen:
                    seen.add(full)
                    symbols.append(ApiSymbol(
                        package=package, version=version, name=full,
                        signature=_bounded(text, _MAX_SIGNATURE_CHARS),
                        source="go-doc"))
    if not symbols:
        return [], "`go doc` returned no exported symbols; API is unknown, "\
                   "not missing"
    return symbols[:_MAX_SYMBOLS], ""


def _cargo_symbols(root: Path, package: str, environment, cfg, deadline,
                   warnings: _Warnings) -> tuple[list[ApiSymbol], str]:
    allowed, reason = _exec_allowed(cfg, root)
    if not allowed:
        return [], reason
    denied = _policy_error(
        cfg, root, "run", "cargo metadata --offline --no-deps")
    if denied:
        return [], denied
    rc, out, stopped = _run_exec(
        root, ["cargo", "metadata", "--offline", "--no-deps",
               "--format-version", "1"],
        environment, cfg, deadline)
    if stopped == "timeout":
        return [], "cargo metadata timed out"
    if stopped == "error":
        return [], "cargo metadata could not start"
    if rc not in (0, None):
        return [], f"cargo metadata exited {rc}: {_bounded(out, 160)}"
    try:
        data = json.loads(out)
    except ValueError:
        return [], "cargo metadata returned non-JSON output"
    target = package.casefold()
    symbols: list[ApiSymbol] = []
    packages = data.get("packages") if isinstance(data, dict) else None
    for item in packages or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if name.casefold() != target:
            continue
        version = str(item.get("version") or "")
        edition = str(item.get("edition") or "")
        for target_info in item.get("targets") or []:
            if not isinstance(target_info, dict):
                continue
            target_name = str(target_info.get("name") or name)
            kinds = ", ".join(str(k) for k in target_info.get("kind") or [])
            signature = f"crate target {target_name} ({kinds})"
            if edition:
                signature += f" edition {edition}"
            symbols.append(ApiSymbol(
                package=package, version=version,
                name=f"{name}::{target_name}",
                signature=_bounded(signature, _MAX_SIGNATURE_CHARS),
                source="cargo-metadata"))
        break
    if not symbols:
        return [], ("cargo metadata --no-deps did not list this package; "
                    "API is unknown, not missing")
    return symbols[:_MAX_SYMBOLS], ""


def _python_inspect_symbols(root: Path, package: str, environment, cfg,
                            deadline,
                            warnings: _Warnings) -> tuple[list[ApiSymbol], str]:
    allowed, reason = _exec_allowed(cfg, root)
    if not allowed:
        return [], reason
    denied = _policy_error(cfg, root, "run", f"python -c inspect {package}")
    if denied:
        return [], denied
    python = environment.python
    if python is None:
        python = Path(sys.executable)
    rc, out, stopped = _run_exec(
        root, [str(python), "-B", "-c", _PY_INSPECT_SCRIPT, package],
        environment, cfg, deadline)
    if stopped == "timeout":
        return [], "python inspect timed out"
    if stopped == "error":
        return [], "python inspect could not start"
    if rc not in (0, None):
        return [], f"python inspect exited {rc}: {_bounded(out, 160)}"
    try:
        data = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return [], "python inspect returned non-JSON output"
    if not isinstance(data, dict) or not data.get("ok"):
        return [], "python inspect could not import the package"
    version = str(data.get("version") or "")
    symbols: list[ApiSymbol] = []
    for item in data.get("symbols") or []:
        if not isinstance(item, dict) or len(symbols) >= _MAX_SYMBOLS:
            continue
        name = str(item.get("name") or "")
        if not name or name.startswith("_"):
            continue
        kind = str(item.get("kind") or "")
        signature = f"{kind} {name}{item.get('signature') or ''}"
        symbols.append(ApiSymbol(
            package=package, version=version, name=f"{package}.{name}",
            signature=_bounded(signature, _MAX_SIGNATURE_CHARS),
            source="python-inspect"))
    return symbols, ""


# ---------------------------------------------------------------------------
# Public indexing API
# ---------------------------------------------------------------------------

def _finalize(symbols: list[ApiSymbol], warnings: _Warnings) -> tuple[ApiSymbol, ...]:
    seen: set[tuple[str, str, str]] = set()
    out: list[ApiSymbol] = []
    for symbol in sorted(symbols, key=lambda s: (
            s.name.casefold(), s.source, s.signature)):
        key = (symbol.name, symbol.signature, symbol.source)
        if key in seen:
            continue
        seen.add(key)
        out.append(ApiSymbol(
            package=symbol.package,
            version=_clean(symbol.version, 80),
            name=_clean(symbol.name, 240),
            signature=_clean(symbol.signature, _MAX_SIGNATURE_CHARS),
            source=_clean(symbol.source, 160)))
        if len(out) >= _MAX_SYMBOLS:
            break
    if len(symbols) > _MAX_SYMBOLS:
        warnings.add(f"symbol list capped at {_MAX_SYMBOLS}")
    return tuple(out)


def index_api(root: Path, package: str, *,
              environment: env_resolve.ResolvedEnvironment,
              cfg, deadline: float | None) -> ApiIndex:
    """Index the public API of `package` without importing it by default.

    `environment` is an :class:`env_resolve.ResolvedEnvironment`; `deadline`
    is a monotonic time (or None). Execution enrichment requires both
    ``cfg.api_oracle_exec`` and ``cfg.allow_shell`` plus an allowing policy.
    """
    return _index_api(root, package, environment=environment, cfg=cfg,
                      deadline=deadline, allow_exec=True)


def _index_api(root: Path, package: str, *, environment, cfg, deadline,
               allow_exec: bool) -> ApiIndex:
    try:
        project_root = Path(root).expanduser().resolve()
    except (OSError, ValueError):
        project_root = Path.cwd().resolve()
    warnings = _Warnings()
    raw = str(package or "").strip()
    valid = _valid_package(raw)
    if not valid:
        return ApiIndex(
            status="unavailable", symbols=(), cache_key="",
            warnings=(_clean("invalid or unsafe package name", 80),))
    if _expired(deadline):
        return ApiIndex(status="timeout", symbols=(), cache_key="",
                        warnings=("deadline expired before indexing",))

    py_plan = _python_plan(project_root, valid, environment, deadline, warnings)
    ts_plan = _ts_package_plan(project_root, valid, deadline, warnings)
    lock_files = _lock_files(project_root)
    fingerprint_paths: list[Path] = list(lock_files)
    fingerprint_paths.extend(Path(p) for p in py_plan.metadata_files)
    fingerprint_paths.extend(Path(p) for p, _m, _k in py_plan.files)
    fingerprint_paths.extend(Path(p) for p in ts_plan.metadata_files)
    fingerprint_paths.extend(Path(p) for p, _m in ts_plan.files)
    version = py_plan.version or ts_plan.version
    exec_mode = "exec" if (
        allow_exec
        and bool(getattr(cfg, "api_oracle_exec", False))
        and bool(getattr(cfg, "allow_shell", False))) else "static"
    key = _fingerprint_key(project_root, valid, environment, version,
                           tuple(fingerprint_paths), extra=exec_mode)
    cached = _cache_read(project_root, valid, key)
    if cached is not None and allow_exec and any(
            symbol.source in _EXEC_SOURCES for symbol in cached.symbols):
        # Execution-derived results are never served after consent has been
        # withdrawn or a policy now denies the command that produced them.
        allowed, _reason = _exec_allowed(cfg, project_root)
        denied = ""
        if allowed:
            for target in ("go doc -all " + valid,
                           "cargo metadata --offline --no-deps",
                           "python -c inspect " + valid):
                denied = _policy_error(cfg, project_root, "run", target)
                if denied:
                    break
        if not allowed or denied:
            cached = None
    if cached is not None:
        return cached

    symbols: list[ApiSymbol] = []
    if py_plan.found:
        symbols.extend(_python_symbols(
            py_plan, valid, project_root, deadline, warnings))
    if ts_plan.found:
        for raw_path, module in ts_plan.files:
            if len(symbols) >= _MAX_SYMBOLS or _expired(deadline):
                break
            symbols.extend(_ts_symbols_from_file(
                Path(raw_path), module, valid,
                ts_plan.version or py_plan.version, deadline))
    if symbols:
        final = _finalize(symbols, warnings)
        index = ApiIndex("ok", final, key, tuple(warnings.items))
        _cache_write(project_root, valid, key, index)
        return index
    if _expired(deadline):
        return ApiIndex("timeout", (), key,
                        tuple(warnings.items) + ("deadline expired during "
                                                 "declaration scan",))

    exec_allowed, exec_reason = _exec_allowed(cfg, project_root)
    go_marker = _safe_child(project_root, "go.mod")
    if go_marker is not None and not go_marker.is_file():
        go_marker = None
    cargo_marker = _safe_child(project_root, "Cargo.toml")
    if cargo_marker is not None and not cargo_marker.is_file():
        cargo_marker = None
    py_marker = any(
        path is not None and path.is_file()
        for path in (_safe_child(project_root, name)
                     for name in ("pyproject.toml", "setup.py", "setup.cfg",
                                  "requirements.txt")))
    unknown_reason = ""
    if allow_exec and exec_allowed:
        py_candidate = "/" not in valid and "@" not in valid
        if (py_plan.found or py_marker) and py_candidate:
            symbols, unknown_reason = _python_inspect_symbols(
                project_root, valid, environment, cfg, deadline, warnings)
        if not symbols and go_marker is not None:
            symbols, unknown_reason = _go_symbols(
                project_root, valid, environment, cfg, deadline, warnings)
        if not symbols and cargo_marker is not None:
            symbols, unknown_reason = _cargo_symbols(
                project_root, valid, environment, cfg, deadline, warnings)
    elif allow_exec:
        if go_marker is not None:
            unknown_reason = f"Go API: {exec_reason}"
        elif cargo_marker is not None:
            unknown_reason = f"Rust API: {exec_reason}"
        elif py_plan.found or py_marker:
            unknown_reason = f"Python inspect: {exec_reason}"

    if symbols:
        final = _finalize(symbols, warnings)
        index = ApiIndex("ok", final, key, tuple(warnings.items))
        _cache_write(project_root, valid, key, index)
        return index
    if unknown_reason:
        warnings.add(unknown_reason)
        # Only a missing execution consent or an explicit policy denial is a
        # denial. The opt-in flag being off, a tool that could not run, or a
        # symbol-free result is unknown -- never proof of absence.
        denied = any(marker in unknown_reason for marker in (
            "allow_shell is off", "denied by policy",
            "policy evaluation error"))
        return ApiIndex("denied" if denied else "unknown", (), key,
                        tuple(warnings.items))
    return ApiIndex("unavailable", (), key, tuple(warnings.items))


def lookup_api(index: ApiIndex, query: str, *,
               limit: int = 12) -> tuple[ApiSymbol, ...]:
    """Rank symbols by exact/last-segment/prefix/substring name match.

    The search is deliberately simple and deterministic: `query` may be a full
    dotted name, a class member, or a short fragment. An empty query returns the
    first `limit` symbols in index order. `limit` is clamped to 0..50.
    """
    try:
        count = max(0, min(int(limit), _MAX_LOOKUP_LIMIT))
    except (TypeError, ValueError):
        count = 12
    if count == 0 or not index.symbols:
        return ()
    text = str(query or "").strip()
    if not text:
        return index.symbols[:count]
    needle = text.casefold()
    terms = [t for t in re.split(r"[.\s:#/()]+", needle) if t]
    scored: list[tuple[float, str, str, ApiSymbol]] = []
    for symbol in index.symbols:
        name = symbol.name.casefold()
        signature = symbol.signature.casefold()
        last = name.rsplit(".", 1)[-1]
        score = 0.0
        if name == needle:
            score = 100.0
        elif last == needle:
            score = 92.0
        elif name.endswith("." + needle):
            score = 88.0
        elif name.startswith(needle):
            score = 76.0
        elif needle in name:
            score = 60.0
        elif needle in signature:
            score = 26.0
        overlap = sum(1 for term in terms if term and term in name)
        score += min(overlap, 4) * 4.0
        if score <= 0:
            continue
        scored.append((-score, symbol.name, symbol.source, symbol))
    scored.sort(key=lambda row: (row[0], row[1], row[2]))
    return tuple(row[3] for row in scored[:count])


# ---------------------------------------------------------------------------
# Task-named libraries and prompt injection
# ---------------------------------------------------------------------------

def _read_toml(path: Path) -> dict:
    try:
        import tomllib
        with path.open("rb") as handle:
            data = tomllib.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, ImportError, TypeError):
        return {}


def _add_name(out: list[str], name, limit: int = 400) -> None:
    value = str(name or "").strip()
    if not value or len(value) > 160 or len(out) >= limit:
        return
    if value.casefold() in {item.casefold() for item in out}:
        return
    out.append(value)


def _requirement_name(raw: str) -> str:
    text = str(raw or "").strip()
    if not text or text.startswith(("-", "#", "git+", "http:", "https:")):
        return ""
    match = _REQ_NAME_RE.match(text)
    return match.group(1) if match else ""


def _declared_dependencies(root: Path) -> tuple[str, ...]:
    names: list[str] = []
    pyproject = _safe_child(root, "pyproject.toml")
    if pyproject is not None and pyproject.is_file():
        data = _read_toml(pyproject)
        project = data.get("project") if isinstance(data, dict) else None
        if isinstance(project, dict):
            for item in project.get("dependencies") or []:
                _add_name(names, _requirement_name(str(item)))
            optional = project.get("optional-dependencies")
            if isinstance(optional, dict):
                for items in optional.values():
                    for item in items or []:
                        _add_name(names, _requirement_name(str(item)))
        tool = data.get("tool") if isinstance(data, dict) else None
        poetry = tool.get("poetry") if isinstance(tool, dict) else None
        if isinstance(poetry, dict):
            for key in (poetry.get("dependencies") or {}):
                if str(key).casefold() != "python":
                    _add_name(names, key)
            groups = poetry.get("group")
            if isinstance(groups, dict):
                for group in groups.values():
                    deps = group.get("dependencies") if isinstance(group, dict) else None
                    if isinstance(deps, dict):
                        for key in deps:
                            _add_name(names, key)
    for pattern in ("requirements*.txt", "requirements/*.txt"):
        for path in sorted(root.glob(pattern))[:20]:
            if _is_link(path) or not path.is_file():
                continue
            for line in _read_bounded(path, 200_000).splitlines():
                name = _requirement_name(line.split(" #", 1)[0])
                _add_name(names, name)
    pipfile = _safe_child(root, "Pipfile")
    if pipfile is not None and pipfile.is_file():
        data = _read_toml(pipfile)
        for section in ("packages", "dev-packages"):
            table = data.get(section) if isinstance(data, dict) else None
            if isinstance(table, dict):
                for key in table:
                    _add_name(names, key)
    package_json = _safe_child(root, "package.json")
    if package_json is not None and package_json.is_file():
        try:
            data = json.loads(_read_bounded(package_json, _MAX_METADATA_BYTES))
        except ValueError:
            data = {}
        if isinstance(data, dict):
            for section in ("dependencies", "devDependencies",
                            "peerDependencies", "optionalDependencies"):
                table = data.get(section)
                if isinstance(table, dict):
                    for key in table:
                        _add_name(names, key)
    cargo = _safe_child(root, "Cargo.toml")
    if cargo is not None and cargo.is_file():
        data = _read_toml(cargo)
        for section in ("dependencies", "dev-dependencies", "build-dependencies"):
            table = data.get(section) if isinstance(data, dict) else None
            if isinstance(table, dict):
                for key in table:
                    _add_name(names, key)
    go_mod = _safe_child(root, "go.mod")
    if go_mod is not None and go_mod.is_file():
        for line in _read_bounded(go_mod, _MAX_METADATA_BYTES).splitlines():
            match = _GO_REQUIRE_RE.match(line)
            if match:
                _add_name(names, match.group(1))
    cargo_lock = _safe_child(root, "Cargo.lock")
    if cargo_lock is not None and cargo_lock.is_file():
        data = _read_toml(cargo_lock)
        packages = data.get("package") if isinstance(data, dict) else None
        for item in packages or []:
            if isinstance(item, dict):
                _add_name(names, item.get("name"))
    return tuple(names)


def _library_variants(name: str) -> tuple[str, ...]:
    base = str(name or "").strip()
    if not base:
        return ()
    variants = {base.casefold()}
    last = base.rsplit("/", 1)[-1]
    variants.add(last.casefold())
    variants.add(base.replace("-", "_").casefold())
    variants.add(base.replace("_", "-").casefold())
    variants.add(last.replace("-", "_").casefold())
    variants.add(last.replace("_", "-").casefold())
    return tuple(sorted(v for v in variants if v))


def _task_tokens(task: str) -> set[str]:
    tokens = {token.casefold() for token in _TASK_TOKEN_RE.findall(task or "")
              if token}
    # A dotted attribute names its library too: `requests.get` must match the
    # declared `requests` dependency while an undeclared root still matches
    # nothing. Keep the exact token as well for direct lookups.
    roots = {token.split(".", 1)[0] for token in tokens if "." in token}
    return {token for token in tokens | roots if token}


def _explicit_library_tokens(task: str) -> list[str]:
    out: list[str] = []
    for pattern in (_EXPLICIT_BACKTICK_RE, _EXPLICIT_AT_RE):
        for match in pattern.finditer(task or ""):
            token = match.group(1).strip().strip(".,;:")
            if not token or token.casefold() in _RESERVED_LIBRARY_NAMES:
                continue
            valid = _valid_package(token)
            if not valid:
                continue
            # A bare backticked word is only a library when it looks like one:
            # a separator, a dot, or a declared dependency the caller matched.
            if not any(ch in valid for ch in "._/-@") and len(valid) < 5:
                continue
            if valid not in out:
                out.append(valid)
    return out


@lru_cache(maxsize=128)
def named_libraries(root: Path, task: str, *,
                    limit: int = 4) -> tuple[str, ...]:
    """Declared libraries the task names exactly, in task/manifest order.

    `root` and `task` are the only inputs; this reads manifest metadata and
    never executes a manager or imports a package. An empty task (or a task
    naming no declared dependency) returns an empty tuple, which is what keeps
    the gather schema and prompt byte-identical on unaffected paths. The cache
    is keyed by the task text, so it is stable for one gather run and cannot
    leak a name from one project into another.
    """
    text = str(task or "")
    if not text.strip():
        return ()
    try:
        project_root = Path(root).expanduser().resolve()
    except (OSError, ValueError):
        return ()
    tokens = _task_tokens(text)
    declared = _declared_dependencies(project_root)
    out: list[str] = []
    for dependency in declared:
        if any(variant in tokens for variant in _library_variants(dependency)):
            if dependency not in out:
                out.append(dependency)
        if len(out) >= limit:
            break
    if len(out) < limit:
        for token in _explicit_library_tokens(text):
            candidate = match_named_library(declared, token) or token
            if candidate not in out:
                out.append(candidate)
            if len(out) >= limit:
                break
    return tuple(out[:limit])


def match_named_library(named, requested: str) -> str:
    """The canonical named library matching `requested`, or ""."""
    wanted = str(requested or "").strip().strip("`\"'")
    if not wanted:
        return ""
    variants = _library_variants(wanted)
    for item in named or ():
        if any(variant in variants for variant in _library_variants(item)):
            return str(item)
    return ""


_SIGNATURE_PREFIXES = ("def ", "async def ", "class ", "interface ", "type ",
                       "export ", "crate ", "func ", "function ", "re-export",
                       "property")


def _symbol_line(symbol: ApiSymbol, limit: int = _MAX_SIGNATURE_CHARS) -> str:
    signature = symbol.signature.strip()
    if len(signature) > limit:
        signature = signature[: limit - 3].rstrip() + "..."
    if signature.startswith(_SIGNATURE_PREFIXES):
        body = f"- {symbol.name}: {signature}"
    else:
        body = f"- {symbol.name}({signature})"
    return f"{body}  [{symbol.source}]"


def _render_symbol(symbol: ApiSymbol) -> str:
    return _symbol_line(symbol)


def render_result(index: ApiIndex, hits: tuple[ApiSymbol, ...], *,
                  package: str, query: str = "") -> str:
    """A bounded, model-facing observation. Never claims an absent API."""
    lines: list[str] = []
    version = next((s.version for s in index.symbols if s.version), "")
    heading = f"API {package}"
    if version:
        heading += f" {version}"
    if query:
        heading += f" -- query {query!r}"
    if hits:
        lines.append(heading)
        lines.extend(_render_symbol(symbol) for symbol in hits)
        if index.status == "ok" and len(index.symbols) > len(hits):
            lines.append(f"[{len(index.symbols) - len(hits)} more indexed "
                         "symbols; narrow `pattern` to see them]")
        if index.warnings:
            lines.append("warnings: " + "; ".join(index.warnings[:4]))
        return _clip("\n".join(lines), 4000)
    reason = "; ".join(index.warnings[:4]) or "no matching static symbol"
    lines.append(f"{heading}: unknown -- {reason}")
    if index.status == "ok":
        lines.append("This is not proof the API is missing: dynamic or "
                     "runtime-generated symbols may be absent from static "
                     "declarations. Use `read`/`grep` on the source, or enable "
                     "`api_oracle_exec` for optional documented enrichment.")
    else:
        lines.append("This is unknown, not proof the API is missing. Static "
                     "declarations may be absent, partial or generated at "
                     "runtime; confirm with `grep` or the library's own docs.")
    return _clip("\n".join(lines), 4000)


def injection(root: Path, task: str, cfg, *, max_libraries: int = 2,
              max_symbols: int = 8) -> str:
    """A bounded, deterministic named-library block for the gather prompt.

    Only called when `named_libraries` found an exact task-named library, so
    tasks that name no dependency keep their prompt byte-identical. Static
    indexing only: assembling a prompt never triggers execution enrichment.
    """
    libraries = named_libraries(root, task, limit=max_libraries)
    if not libraries:
        return ""
    try:
        environment = env_resolve.resolve_environment(root)
    except Exception:
        return ""
    lines = ["LIBRARY APIS NAMED IN THIS TASK (read-only static index; call "
             "action=api with symbol=<library> and pattern=<name fragment> "
             "for a fuller/other lookup; an absent symbol is unknown, not "
             "proof the API is missing):"]
    deadline = time.monotonic() + 2.0
    for library in libraries:
        index = _index_api(root, library, environment=environment, cfg=cfg,
                           deadline=deadline, allow_exec=False)
        if index.symbols:
            shown = index.symbols[:max_symbols]
            lines.append(f"- {library}:")
            for symbol in shown:
                lines.append("  " + _symbol_line(symbol, 160))
            if len(index.symbols) > len(shown):
                lines.append(f"  [{len(index.symbols)} more symbols; use the "
                             "api action with a narrower pattern]")
        else:
            lines.append(f"- {library}: unknown locally "
                         "(static declarations absent or not installed)")
    text = "\n".join(lines)
    return _clip(text, 2400)
