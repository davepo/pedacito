"""WP-4103A: offline environment resolver and its execution integration."""
from __future__ import annotations

import dataclasses
import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import cli, diagnostics, env_resolve, lsp, proc, testsel, tools
from pedacito.config import Config
from pedacito.index import Index

BIN = "Scripts" if os.name == "nt" else "bin"
PYTHON = "python.exe" if os.name == "nt" else "python"


def _write_exec(path: Path, text: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if os.name != "nt":
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _venv(root: Path, name: str = ".venv", *, python: bool = True) -> Path:
    directory = root / name
    bindir = directory / BIN
    bindir.mkdir(parents=True, exist_ok=True)
    if python:
        _write_exec(bindir / PYTHON)
    return directory


def _path_value(env: dict[str, str]) -> str:
    for key, value in env.items():
        if key.upper() == "PATH":
            return value
    return ""


def _resolved(root: Path, **kwargs):
    return env_resolve.resolve_environment(root, env=kwargs.pop("env", {}))


# --------------------------------------------------------------- dataclass ---


def test_resolved_environment_is_frozen_and_has_the_contract_fields(tmp_path):
    resolved = _resolved(tmp_path)
    assert dataclasses.is_dataclass(resolved)
    assert resolved.root == tmp_path.resolve()
    assert resolved.workspace_root == tmp_path.resolve()
    assert resolved.kind == "system"
    assert resolved.python is None
    assert resolved.bin_dirs == ()
    assert resolved.package_manager == ""
    assert resolved.evidence
    with pytest.raises(dataclasses.FrozenInstanceError):
        resolved.kind = "venv"  # type: ignore[misc]


# ------------------------------------------------------------- interpreters ---


def test_project_venv_interpreter_wins_and_multiple_venvs_are_reported(tmp_path):
    _venv(tmp_path, ".venv")
    _venv(tmp_path, "venv")
    resolved = _resolved(tmp_path)
    assert resolved.kind == "venv"
    assert resolved.python == (tmp_path / ".venv" / BIN / PYTHON).resolve()
    assert resolved.bin_dirs[0] == (tmp_path / ".venv" / BIN).resolve()
    assert any("existing interpreter" in item for item in resolved.evidence)
    assert any(item.startswith("ambiguous:") for item in resolved.evidence)


def test_venv_without_interpreter_still_contributes_its_bin_dir(tmp_path):
    (tmp_path / ".venv" / BIN).mkdir(parents=True)
    (tmp_path / "uv.lock").write_text("", encoding="utf-8")
    resolved = _resolved(tmp_path)
    assert resolved.kind == "uv"
    assert resolved.python is None
    assert (tmp_path / ".venv" / BIN).resolve() in resolved.bin_dirs


@pytest.mark.parametrize("marker,kind,manager", [
    ("uv.lock", "uv", "uv"),
    ("poetry.lock", "poetry", "poetry"),
    ("Pipfile", "pipenv", "pipenv"),
])
def test_python_metadata_detection(tmp_path, marker, kind, manager):
    (tmp_path / marker).write_text("", encoding="utf-8")
    resolved = _resolved(tmp_path)
    assert resolved.kind == kind
    assert resolved.package_manager == manager
    assert resolved.python is None
    assert any(marker in item for item in resolved.evidence)


def test_venv_metadata_hint_names_the_manager(tmp_path):
    _venv(tmp_path)
    (tmp_path / "uv.lock").write_text("", encoding="utf-8")
    resolved = _resolved(tmp_path)
    assert resolved.kind == "uv"
    assert resolved.package_manager == "uv"
    assert resolved.python == (tmp_path / ".venv" / BIN / PYTHON).resolve()
    assert any("uv.lock" in item for item in resolved.evidence)


def test_conda_project_environment_and_active_conda(tmp_path):
    (tmp_path / "conda-meta").mkdir()
    if os.name == "nt":
        _write_exec(tmp_path / "python.exe")
    else:
        _write_exec(tmp_path / "bin" / "python")
    resolved = _resolved(tmp_path)
    assert resolved.kind == "conda"
    assert resolved.python is not None
    assert resolved.package_manager == "conda"

    outside = tmp_path / "outside-conda"
    (outside / "conda-meta").mkdir(parents=True)
    active = _write_exec(outside / BIN / PYTHON)
    got = _resolved(tmp_path / "empty", env={"CONDA_PREFIX": str(outside)})
    assert got.kind == "conda"
    assert got.python == active.resolve()


def test_active_virtual_env_outside_the_project_is_explicit_user_state(tmp_path):
    outside = tmp_path / "outside-venv"
    active = _write_exec(outside / BIN / PYTHON)
    root = tmp_path / "project"
    root.mkdir()
    resolved = _resolved(root, env={"VIRTUAL_ENV": str(outside)})
    assert resolved.kind == "venv"
    assert resolved.python == active.resolve()
    assert (outside / BIN).resolve() in resolved.bin_dirs


@pytest.mark.parametrize("directory,manifest,kind", [
    (".tox/py311", "tox.ini", "tox"),
    (".nox/tests", "noxfile.py", "nox"),
])
def test_tox_and_nox_environments(tmp_path, directory, manifest, kind):
    (tmp_path / manifest).write_text("", encoding="utf-8")
    python = _write_exec(tmp_path / directory / BIN / PYTHON)
    resolved = _resolved(tmp_path)
    assert resolved.kind == kind
    assert resolved.python == python.resolve()
    assert (tmp_path / directory / BIN).resolve() in resolved.bin_dirs


# ------------------------------------------------------------- other stacks ---


@pytest.mark.parametrize("marker,kind,manager", [
    ("pnpm-lock.yaml", "pnpm", "pnpm"),
    ("yarn.lock", "yarn", "yarn"),
    ("bun.lockb", "bun", "bun"),
])
def test_node_package_managers(tmp_path, marker, kind, manager):
    (tmp_path / marker).write_text("", encoding="utf-8")
    (tmp_path / "node_modules" / ".bin").mkdir(parents=True)
    resolved = _resolved(tmp_path)
    assert resolved.kind == kind
    assert resolved.package_manager == manager
    assert (tmp_path / "node_modules" / ".bin").resolve() in resolved.bin_dirs


def test_package_json_package_manager_field_and_npm_fallback(tmp_path):
    (tmp_path / "package.json").write_text(
        '{"packageManager": "pnpm@8.15.0"}', encoding="utf-8")
    assert _resolved(tmp_path).kind == "pnpm"
    (tmp_path / "package.json").write_text('{"scripts": {}}', encoding="utf-8")
    npm = _resolved(tmp_path)
    assert npm.kind == "node"
    assert npm.package_manager == "npm"


def test_cargo_go_and_existing_interpreter_precedence(tmp_path):
    (tmp_path / "Cargo.toml").write_text("", encoding="utf-8")
    assert _resolved(tmp_path).kind == "cargo"
    (tmp_path / "go.mod").write_text("", encoding="utf-8")
    both = _resolved(tmp_path)
    assert both.kind == "cargo"
    assert any("also detected: go" in item for item in both.evidence)
    _venv(tmp_path)
    python_first = _resolved(tmp_path)
    assert python_first.kind == "venv"
    assert python_first.python is not None
    joined = " ".join(python_first.evidence)
    assert "cargo" in joined and "go" in joined


# --------------------------------------------------------------- boundaries ---


def test_escaping_venv_symlink_is_rejected(tmp_path):
    outside = tmp_path / "outside"
    _venv(outside)
    root = tmp_path / "project"
    root.mkdir()
    try:
        os.symlink(outside, root / ".venv", target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink unavailable: {exc}")
    resolved = _resolved(root)
    assert resolved.python is None
    assert resolved.kind == "system"


def test_spaces_in_the_project_path_are_preserved(tmp_path):
    root = tmp_path / "my project with spaces"
    root.mkdir()
    python = _write_exec(root / ".venv" / BIN / PYTHON)
    resolved = _resolved(root)
    assert resolved.python == python.resolve()
    path = _path_value(env_resolve.command_env(resolved, Config()))
    assert str((root / ".venv" / BIN).resolve()) in path.split(os.pathsep)


def test_detection_never_executes_scripts_or_managers(tmp_path):
    marker = tmp_path / "executed.marker"
    script = f"#!/bin/sh\ntouch {marker}\n"
    _write_exec(tmp_path / ".venv" / BIN / PYTHON, script)
    _write_exec(tmp_path / ".venv" / BIN / "activate", script)
    _write_exec(tmp_path / ".venv" / BIN / "uv", script)
    _write_exec(tmp_path / ".venv" / BIN / "poetry", script)
    _write_exec(tmp_path / ".venv" / BIN / "npx", script)
    (tmp_path / "uv.lock").write_text("", encoding="utf-8")
    (tmp_path / "poetry.lock").write_text("", encoding="utf-8")
    resolved = _resolved(tmp_path)
    assert resolved.python is not None
    assert not marker.exists()


# -------------------------------------------------------------- command_env ---


def test_command_env_scrubs_secrets_and_prepends_bin_dirs(tmp_path, monkeypatch):
    _venv(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "x" * 30)
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")
    cfg = Config()
    cfg.env_passthrough = ("DATABASE_URL",)
    resolved = _resolved(tmp_path)
    child = env_resolve.command_env(resolved, cfg)
    assert "OPENAI_API_KEY" not in child
    assert child["DATABASE_URL"] == "postgresql://user:pass@localhost/db"
    first = _path_value(child).split(os.pathsep)[0]
    assert first == str((tmp_path / ".venv" / BIN).resolve())
    assert child["VIRTUAL_ENV"] == str((tmp_path / ".venv").resolve())
    assert child["NO_COLOR"] == "1"
    assert child["PYTHONDONTWRITEBYTECODE"] == "1"


def test_command_env_does_not_mutate_the_resolved_object(tmp_path):
    _venv(tmp_path)
    resolved = _resolved(tmp_path)
    before = resolved.bin_dirs
    env_resolve.command_env(resolved, Config())
    assert resolved.bin_dirs == before


# -------------------------------------------------------------- integration ---


def test_explicit_test_command_is_never_rewritten(tmp_path):
    _venv(tmp_path)
    (tmp_path / "test_a.py").write_text("def test_ok():\n    pass\n",
                                       encoding="utf-8")
    cfg = Config()
    cfg.test_command = "python -m pytest -q"
    assert testsel.detect_test_command(tmp_path, cfg) == cfg.test_command


def test_run_test_uses_the_resolved_interpreter_and_bin_dirs(tmp_path, monkeypatch):
    _venv(tmp_path)
    (tmp_path / "test_ok.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8")
    idx = Index(tmp_path, [])
    cfg = Config()
    cfg.allow_shell = True
    cfg.log_events = False
    calls: list[tuple[list[str], dict]] = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        if argv and argv[-1] == "--version":
            return 0, "pytest 8.0", "", 0.01
        return 0, "1 passed", "", 0.01

    monkeypatch.setattr(tools.proc, "run", fake_run)
    out = tools.tool_run_test(idx, cfg, "test_ok.py")
    assert "RUN TEST:" in out
    actual = [call for call in calls
              if "test_ok.py" in call[0] and "--version" not in call[0]]
    assert actual, calls
    argv, kwargs = actual[-1]
    assert argv[0] == str((tmp_path / ".venv" / BIN / PYTHON).resolve())
    resolved = env_resolve.resolve_environment(tmp_path, env={})
    assert kwargs["bin_dirs"] == resolved.bin_dirs
    assert _path_value(kwargs["env"]).split(os.pathsep)[0] \
        == str((tmp_path / ".venv" / BIN).resolve())


def test_diagnostics_uses_the_resolved_tool_and_scrubbed_env(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    _venv(tmp_path)
    ruff_name = "ruff.exe" if os.name == "nt" else "ruff"
    _write_exec(tmp_path / ".venv" / BIN / ruff_name)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "x" * 30)
    cfg = Config()
    seen: list[dict] = []

    def fake_run(argv, **kwargs):
        seen.append({"argv": list(argv), **kwargs})
        return 0, "All checks passed!\n", "", 0.01

    monkeypatch.setattr(diagnostics.proc, "run", fake_run)
    ok, report = diagnostics.run(tmp_path, cfg, ["app.py"])
    assert ok
    assert "CHECK ruff: PASS" in report
    assert seen and seen[0]["argv"][0] == "ruff"
    assert "OPENAI_API_KEY" not in seen[0]["env"]
    resolved = env_resolve.resolve_environment(tmp_path, env={})
    assert seen[0]["bin_dirs"] == resolved.bin_dirs


def test_lsp_prefers_a_server_from_the_resolved_environment(tmp_path):
    _venv(tmp_path)
    name = "pyright-langserver.exe" if os.name == "nt" else "pyright-langserver"
    server = _write_exec(tmp_path / ".venv" / BIN / name)
    cfg = Config()
    cfg.lsp_servers = {}
    command = lsp.command_for(tmp_path, "a.py", cfg)
    assert command and Path(command[0]) == server.resolve()


def test_proc_bin_dirs_override_the_legacy_venv_probe(tmp_path):
    _venv(tmp_path)
    env = {"PATH": "/usr/bin", "OTHER": "keep"}
    legacy = proc.with_venv_path(env, tmp_path)
    assert str(tmp_path / ".venv" / BIN) in _path_value(legacy)
    explicit = proc.with_venv_path(env, tmp_path, bin_dirs=())
    assert explicit["PATH"] == "/usr/bin"
    custom = tmp_path / "custom-bin"
    custom.mkdir()
    got = proc.with_venv_path(env, tmp_path, bin_dirs=(custom,))
    assert got["PATH"].split(os.pathsep)[0] == str(custom)
    assert env["PATH"] == "/usr/bin"


def test_doctor_alias_and_environment_report(tmp_path):
    _venv(tmp_path)
    parser = cli.build_parser()
    args = parser.parse_args(["doctor", str(tmp_path)])
    assert args.cmd == "doctor"
    report = cli._environment_report(tmp_path, Config())
    assert "Environment:" in report
    assert "Interpreter:" in report
    assert ".venv" in report
    keys = {"cli.cmd.doctor", "cli.doctor.env.header",
            "cli.doctor.env.interpreter", "cli.doctor.env.bins",
            "cli.doctor.env.evidence", "cli.doctor.env.none"}
    from pedacito import i18n
    for lang in ("en", "es"):
        catalog = i18n.load_catalog(lang)
        assert keys <= set(catalog)
