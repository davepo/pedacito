"""WP-4103B: offline, versioned API oracle and its gather surface."""
from __future__ import annotations

import dataclasses
import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import api_oracle, config, env_resolve, tools
from pedacito.config import Config
from pedacito.index import Index

BIN = "Scripts" if os.name == "nt" else "bin"
PYTHON = "python.exe" if os.name == "nt" else "python"
SITE = Path("Lib") / "site-packages" if os.name == "nt" \
    else Path("lib") / "python3.11" / "site-packages"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _exec(path: Path, text: str = "") -> Path:
    _write(path, text)
    if os.name != "nt":
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _venv(root: Path) -> Path:
    venv = root / ".venv"
    _exec(venv / BIN / PYTHON)
    (venv / SITE).mkdir(parents=True, exist_ok=True)
    return venv


def _python_fixture(root: Path, *, version: str = "1.2.3") -> tuple[Path, Path]:
    venv = _venv(root)
    site = venv / SITE
    dist = site / f"mylib-{version}.dist-info"
    dist.mkdir(parents=True, exist_ok=True)
    _write(dist / "METADATA",
           "Metadata-Version: 2.1\n"
           f"Name: mylib\nVersion: {version}\n")
    _write(dist / "top_level.txt", "mylib\n")
    pkg = site / "mylib"
    pkg.mkdir(parents=True, exist_ok=True)
    _write(pkg / "__init__.pyi",
           "def greet(name: str, times: int = 1) -> str: ...\n"
           "\n"
           "class Thing(Base):\n"
           "    def run(self, x: int = 0) -> bool: ...\n")
    _write(pkg / "core.pyi",
           "async def fetch(url: str, *, timeout: float = 5.0) -> bytes: ...\n")
    return venv, site


def _ts_fixture(root: Path) -> Path:
    pkg = root / "node_modules" / "widget"
    _write(pkg / "package.json",
           '{"name": "widget", "version": "2.0.0", '
           '"types": "dist/index.d.ts"}')
    _write(pkg / "dist" / "index.d.ts",
           "export declare function makeWidget(name: string, "
           "size?: number): Widget;\n"
           "export interface Widget { id: string }\n"
           "export class Panel extends Base {\n"
           "  render(): void;\n"
           "}\n"
           "export { helper, other as alias };\n")
    return pkg


def _env(root: Path):
    return env_resolve.resolve_environment(root, env={})


def _cfg(task: str = "", root: Path | None = None) -> Config:
    cfg = Config()
    cfg.log_events = False
    if task:
        cfg._task_text = task
    if root is not None:
        cfg._root = str(root)
    return cfg


def _index(root: Path, package: str, cfg: Config | None = None):
    return api_oracle.index_api(
        root, package, environment=_env(root), cfg=cfg or Config(),
        deadline=time.monotonic() + 8)


def _names(index) -> list[str]:
    return [symbol.name for symbol in index.symbols]


# --------------------------------------------------------------- dataclasses ---


def test_contract_dataclasses_are_frozen_and_lookup_is_bounded():
    symbols = tuple(
        api_oracle.ApiSymbol(
            package="pkg", version="1.0.0", name=f"pkg.fn{i}",
            signature=f"def fn{i}(x: int) -> int", source="python-pyi:pkg.pyi")
        for i in range(25))
    index = api_oracle.ApiIndex(
        status="ok", symbols=symbols, cache_key="k",
        warnings=("one warning",))
    with pytest.raises(dataclasses.FrozenInstanceError):
        index.status = "unknown"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        symbols[0].name = "other"  # type: ignore[misc]

    assert len(api_oracle.lookup_api(index, "", limit=5)) == 5
    assert api_oracle.lookup_api(index, "", limit=0) == ()
    assert len(api_oracle.lookup_api(index, "fn", limit=99)) == 25
    exact = api_oracle.lookup_api(index, "fn7")
    assert exact and exact[0].name == "pkg.fn7"
    assert "pkg.fn7" in api_oracle.render_result(
        index, exact, package="pkg", query="fn7")


# ------------------------------------------------------------- static Python ---


def test_static_python_stub_index_is_versioned(tmp_path):
    _python_fixture(tmp_path)
    index = _index(tmp_path, "mylib")
    assert index.status == "ok"
    assert "mylib.greet" in _names(index)
    assert "mylib.Thing.run" in _names(index)
    assert "mylib.core.fetch" in _names(index)
    greet = next(s for s in index.symbols if s.name == "mylib.greet")
    assert greet.version == "1.2.3"
    assert greet.signature.startswith("def greet(name: str")
    assert greet.source.startswith("python-pyi:")
    assert str(tmp_path) not in greet.source
    # The async declaration survives as a signature, not as imported code.
    fetch = next(s for s in index.symbols if s.name == "mylib.core.fetch")
    assert fetch.signature.startswith("async def fetch(")

    cache_dir = tmp_path / ".pedacito" / "cache" / "api"
    files = sorted(cache_dir.glob("*.json"))
    assert len(files) == 1
    cached = _index(tmp_path, "mylib")
    assert cached.cache_key == index.cache_key
    assert _names(cached) == _names(index)


def test_version_and_declaration_changes_invalidate_the_cache(tmp_path):
    venv, site = _python_fixture(tmp_path)
    first = _index(tmp_path, "mylib")
    first_key = first.cache_key

    # A version bump alone changes the key.
    dist = site / "mylib-1.2.3.dist-info"
    dist.rename(site / "mylib-2.0.0.dist-info")
    _write(site / "mylib-2.0.0.dist-info" / "METADATA",
           "Metadata-Version: 2.1\nName: mylib\nVersion: 2.0.0\n")
    second = _index(tmp_path, "mylib")
    assert second.cache_key != first_key
    assert next(s for s in second.symbols
                if s.name == "mylib.greet").version == "2.0.0"

    # A declaration-only change (same version) also changes the key.
    _write(site / "mylib" / "__init__.pyi",
           "def greet(name: str, loud: bool = False) -> str: ...\n")
    third = _index(tmp_path, "mylib")
    assert third.cache_key != second.cache_key
    greet = next(s for s in third.symbols if s.name == "mylib.greet")
    assert "loud" in greet.signature

    # Old keys are pruned: at most the documented per-package file cap.
    files = list((tmp_path / ".pedacito" / "cache" / "api").glob("*.json"))
    assert 1 <= len(files) <= 6


def test_static_python_module_file_and_project_package(tmp_path):
    # A plain module file and a project-local src/ package need no dist info.
    _write(tmp_path / "single.py",
           "def one(value: int) -> int:\n    return value\n")
    _write(tmp_path / "src" / "localpkg" / "__init__.py",
           "def local(value: str) -> str:\n    return value\n")
    single = _index(tmp_path, "single")
    assert single.status == "ok"
    assert "single.one" in _names(single)
    local = _index(tmp_path, "localpkg")
    assert local.status == "ok"
    assert "localpkg.local" in _names(local)


# ----------------------------------------------------------- static TypeScript ---


def test_static_typescript_declaration_index(tmp_path):
    _ts_fixture(tmp_path)
    index = _index(tmp_path, "widget")
    assert index.status == "ok"
    assert "dist.makeWidget" in _names(index) or \
        "widget.makeWidget" in _names(index)
    assert any(name.endswith("Widget") for name in _names(index))
    assert any(name.endswith("Panel") for name in _names(index))
    assert any(name.endswith("alias") for name in _names(index))
    make = next(s for s in index.symbols if s.name.endswith("makeWidget"))
    assert make.version == "2.0.0"
    assert "name: string" in make.signature
    assert make.source.startswith("ts-dts:")
    rendered = api_oracle.render_result(
        index, api_oracle.lookup_api(index, "makeWidget"), package="widget")
    assert "makeWidget" in rendered
    assert "export declare function" in rendered


def test_scoped_types_package_fallback(tmp_path):
    pkg = tmp_path / "node_modules" / "@types" / "leftpad"
    _write(pkg / "index.d.ts",
           "export declare function leftpad(text: string, len: number): string;\n")
    _write(pkg / "package.json", '{"name": "@types/leftpad", "version": "1.1.0"}')
    index = _index(tmp_path, "leftpad")
    assert index.status == "ok"
    assert any(s.name.endswith("leftpad") for s in index.symbols)
    assert next(s for s in index.symbols if s.name.endswith("leftpad")).version \
        == "1.1.0"


def test_environment_helpers_expose_approved_roots(tmp_path):
    venv, site = _python_fixture(tmp_path)
    (tmp_path / "node_modules").mkdir(exist_ok=True)
    env = _env(tmp_path)
    assert site.resolve() in env_resolve.site_packages_dirs(env)
    roots = env_resolve.dependency_roots(env)
    assert tmp_path.resolve() in roots
    assert site.resolve() in roots
    assert (tmp_path / "node_modules").resolve() in roots


# ------------------------------------------------------- Go / Rust execution ---


def _go_fixture(root: Path) -> None:
    _write(root / "go.mod",
           "module example.com/app\n\ngo 1.22\n\n"
           "require github.com/gin-gonic/gin v1.9.1\n")
    _write(root / "main.go", "package main\n")


def test_go_execution_is_gated_and_argv_only(tmp_path, monkeypatch):
    _go_fixture(tmp_path)
    calls: list[tuple[list[str], dict]] = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return (0, "package gin\n\nfunc New(opts ...Option) *Engine\n"
                   "func (e *Engine) Run(addr ...string) (err error)\n"
                   "type Engine struct\n", "", 0.01)

    monkeypatch.setattr(api_oracle.proc, "run", fake_run)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "x" * 30)
    package = "github.com/gin-gonic/gin"

    off = _index(tmp_path, package, _cfg(root=tmp_path))
    assert off.status == "unknown"
    assert "api_oracle_exec" in " ".join(off.warnings)
    assert calls == []

    no_consent = _cfg(root=tmp_path)
    no_consent.api_oracle_exec = True
    denied = _index(tmp_path, package, no_consent)
    assert denied.status == "denied"
    assert calls == []

    on = _cfg(root=tmp_path)
    on.api_oracle_exec = True
    on.allow_shell = True
    index = _index(tmp_path, package, on)
    assert index.status == "ok"
    assert calls, "go doc should be attempted with consent"
    argv, kwargs = calls[0]
    assert argv == ["go", "doc", "-all", package]
    assert kwargs["cwd"] == tmp_path.resolve()
    assert kwargs.get("shell", False) is False
    assert "OPENAI_API_KEY" not in (kwargs.get("env") or {})
    assert (kwargs.get("env") or {}).get("GOPROXY") == "off"
    assert {"gin.New", "gin.Engine.Run", "gin.Engine"} <= set(_names(index))
    assert all(s.source == "go-doc" for s in index.symbols)


def test_rust_execution_uses_offline_no_deps_metadata(tmp_path, monkeypatch):
    _write(tmp_path / "Cargo.toml",
           '[package]\nname = "app"\nversion = "0.1.0"\n\n'
           '[dependencies]\nserde_json = "1"\n')
    _write(tmp_path / "Cargo.lock",
           '[[package]]\nname = "serde_json"\nversion = "1.0.120"\n')
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return (0, json.dumps({
            "packages": [{
                "name": "serde_json",
                "version": "1.0.120",
                "edition": "2021",
                "targets": [{"name": "serde_json", "kind": ["lib"]}],
            }],
        }), "", 0.01)

    monkeypatch.setattr(api_oracle.proc, "run", fake_run)
    off = _index(tmp_path, "serde_json", _cfg(root=tmp_path))
    assert off.status == "unknown"
    assert calls == []

    cfg = _cfg(root=tmp_path)
    cfg.api_oracle_exec = True
    cfg.allow_shell = True
    index = _index(tmp_path, "serde_json", cfg)
    assert index.status == "ok"
    assert calls
    argv = calls[0]
    assert argv[:2] == ["cargo", "metadata"]
    assert "--offline" in argv
    assert "--no-deps" in argv
    assert "--format-version" in argv and "1" in argv
    assert index.symbols[0].version == "1.0.120"
    assert index.symbols[0].source == "cargo-metadata"

    # Withdrawal of consent must not serve the execution-derived cache.
    cfg.api_oracle_exec = False
    calls.clear()
    again = _index(tmp_path, "serde_json", cfg)
    assert again.status == "unknown"
    assert calls == []


def test_python_inspect_enrichment_is_gated(tmp_path, monkeypatch):
    _venv(tmp_path)
    site = tmp_path / ".venv" / SITE
    dist = site / "compiled-3.1.0.dist-info"
    dist.mkdir(parents=True)
    _write(dist / "METADATA",
           "Metadata-Version: 2.1\nName: compiled\nVersion: 3.1.0\n")
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return (0, json.dumps({
            "ok": True, "version": "3.1.0", "error": "",
            "symbols": [{"name": "build", "kind": "function",
                         "signature": "(name: str) -> object"}],
        }), "", 0.01)

    monkeypatch.setattr(api_oracle.proc, "run", fake_run)
    off = _index(tmp_path, "compiled", _cfg(root=tmp_path))
    assert off.status == "unknown"
    assert calls == []

    cfg = _cfg(root=tmp_path)
    cfg.api_oracle_exec = True
    cfg.allow_shell = True
    index = _index(tmp_path, "compiled", cfg)
    assert index.status == "ok"
    assert calls and calls[0][-1] == "compiled"
    assert "-c" in calls[0]
    assert index.symbols[0].source == "python-inspect"
    assert "compiled.build" in _names(index)


def test_execution_deadline_is_clamped(tmp_path, monkeypatch):
    _go_fixture(tmp_path)
    seen: list[float] = []

    def fake_run(argv, **kwargs):
        seen.append(kwargs.get("timeout"))
        return 0, "package gin\nfunc New() *Engine\n", "", 0.01

    monkeypatch.setattr(api_oracle.proc, "run", fake_run)
    cfg = _cfg(root=tmp_path)
    cfg.api_oracle_exec = True
    cfg.allow_shell = True
    index = api_oracle.index_api(
        tmp_path, "github.com/gin-gonic/gin", environment=_env(tmp_path),
        cfg=cfg, deadline=time.monotonic() + 0.4)
    assert index.status == "ok"
    assert seen and seen[0] is not None and seen[0] <= 0.4


# ------------------------------------------------------------- containment ---


def test_traversal_and_escaping_links_are_rejected(tmp_path):
    invalid = _index(tmp_path, "../secrets")
    assert invalid.status == "unavailable"
    assert invalid.symbols == ()

    outside = tmp_path / "outside"
    _write(outside / "mylib" / "__init__.pyi", "def stolen() -> None: ...\n")
    root = tmp_path / "project"
    site = _venv(root) / SITE
    try:
        os.symlink(outside / "mylib", site / "mylib",
                   target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink unavailable: {exc}")
    index = _index(root, "mylib")
    assert "mylib.stolen" not in _names(index)
    assert index.symbols == ()


def test_symlinked_node_module_escape_is_rejected(tmp_path):
    outside = tmp_path / "outside-pkg"
    _write(outside / "package.json", '{"name": "escape", "version": "9.9.9"}')
    _write(outside / "index.d.ts", "export declare function leak(): void;\n")
    root = tmp_path / "project"
    (root / "node_modules").mkdir(parents=True)
    try:
        os.symlink(outside, root / "node_modules" / "escape",
                   target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink unavailable: {exc}")
    index = _index(root, "escape")
    assert index.symbols == ()


def test_cache_writes_are_atomic_and_complete(tmp_path):
    import threading

    _python_fixture(tmp_path)
    env = _env(tmp_path)
    errors: list[Exception] = []

    def worker():
        try:
            index = api_oracle.index_api(
                tmp_path, "mylib", environment=env, cfg=Config(),
                deadline=time.monotonic() + 8)
            assert index.status == "ok"
        except Exception as exc:          # pragma: no cover - failure detail
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    directory = tmp_path / ".pedacito" / "cache" / "api"
    files = sorted(directory.glob("*.json"))
    assert files
    for path in files:
        json.loads(path.read_text(encoding="utf-8"))
    assert not list(directory.glob("*.pedacito-tmp"))


# ------------------------------------------------------------- tool integration ---


def test_schema_and_tool_api_only_for_exact_task_named_library(tmp_path):
    _python_fixture(tmp_path)
    _write(tmp_path / "pyproject.toml",
           '[project]\nname = "app"\nversion = "0.1.0"\n'
           'dependencies = ["mylib"]\n')

    default = tools.schema_for(Config())
    assert "api" not in default["properties"]["action"]["enum"]

    task_cfg = _cfg("Use mylib to parse the document", tmp_path)
    schema = tools.schema_for(task_cfg)
    assert "api" in schema["properties"]["action"]["enum"]
    assert "api" in tools.TOOL_HELP

    idx = Index(tmp_path, [])
    out = tools.tool_api(idx, task_cfg, "mylib", "greet")
    assert "mylib.greet" in out
    assert "def greet(" in out
    assert "python-pyi" in out

    miss = tools.tool_api(idx, task_cfg, "mylib", "does_not_exist")
    assert "unknown" in miss.lower()
    assert "not proof" in miss.lower()

    other_cfg = _cfg("Use mylib but ask for another library", tmp_path)
    error = tools.tool_api(idx, other_cfg, "requests", "Session")
    assert error.startswith("ERROR")
    assert "exact task-named libraries" in error

    unnamed = _cfg("Refactor the parser", tmp_path)
    assert "api" not in tools.schema_for(unnamed)["properties"]["action"]["enum"]
    assert tools.api_injection(unnamed) == ""
    assert tools.api_injection(task_cfg)


def test_named_libraries_requires_an_exact_declared_name(tmp_path):
    _write(tmp_path / "pyproject.toml",
           '[project]\nname = "app"\nversion = "0.1.0"\n'
           'dependencies = ["mylib", "python-dateutil>=2"]\n')
    api_oracle.named_libraries.cache_clear()
    assert api_oracle.named_libraries(tmp_path, "Use mylib please") == ("mylib",)
    assert api_oracle.named_libraries(
        tmp_path, "Use the `mylib` API") == ("mylib",)
    assert api_oracle.named_libraries(tmp_path, "Use mylibrary") == ()
    assert api_oracle.named_libraries(tmp_path, "Nothing relevant") == ()
    assert api_oracle.match_named_library(
        ("python-dateutil",), "dateutil") == ""
    assert api_oracle.match_named_library(
        ("python-dateutil",), "python-dateutil") == "python-dateutil"


def test_injection_is_static_only_and_bounded(tmp_path, monkeypatch):
    venv, site = _python_fixture(tmp_path)
    _write(tmp_path / "pyproject.toml",
           '[project]\nname = "app"\ndependencies = ["mylib"]\n')
    cfg = _cfg("Use mylib", tmp_path)
    cfg.api_oracle_exec = True
    cfg.allow_shell = True

    def boom(*args, **kwargs):
        raise AssertionError("prompt assembly must not execute")

    monkeypatch.setattr(api_oracle.proc, "run", boom)
    block = tools.api_injection(cfg)
    assert "mylib" in block
    assert len(block) <= 2400
    assert "mylib.greet" in block


# ------------------------------------------------------------- config/exec ---


def test_api_oracle_exec_is_an_exec_field_and_defaults_off():
    assert "api_oracle_exec" in config.EXEC_FIELDS
    assert Config().api_oracle_exec is False
    assert "api_oracle_exec = false" in config.TEMPLATE
    project = config.project_template()
    assert "# api_oracle_exec = false" in project


# --------------------------------------------------------------- deadline ---


def test_deadline_and_unknown_results_are_honest(tmp_path):
    expired = api_oracle.index_api(
        tmp_path, "mylib", environment=_env(tmp_path), cfg=Config(),
        deadline=time.monotonic() - 1)
    assert expired.status == "timeout"

    missing = _index(tmp_path, "not-installed-anywhere")
    assert missing.status == "unavailable"
    assert missing.symbols == ()
    rendered = api_oracle.render_result(
        missing, api_oracle.lookup_api(missing, "Thing"),
        package="not-installed-anywhere")
    assert "unknown" in rendered.lower()
    assert "not proof" in rendered.lower()


def test_cache_redacts_secrets_and_stores_no_absolute_paths(tmp_path):
    venv, site = _python_fixture(tmp_path)
    secret = "sk-" + "a" * 32
    _write(site / "mylib" / "secrets.pyi",
           f'def call(api_key: str = "{secret}") -> None: ...\n')
    index = _index(tmp_path, "mylib")
    assert index.status == "ok"
    payload = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / ".pedacito" / "cache" / "api").glob("*.json"))
    assert secret not in payload
    assert str(tmp_path) not in payload
    rendered = api_oracle.render_result(
        index, api_oracle.lookup_api(index, "call"), package="mylib")
    assert secret not in rendered
