"""WP-4102D: the §3.4 autofix ladder and its exact retain/discard contract.

Every test runs without a real linter, model, network or sandbox backend:
tool availability comes from fake executables in a fake bin directory and the
fixer/reporter behaviour comes from a fake `autofix._run`, while the real
applicability, argv, shadow-copy, finding-identity, scope, preimage-hash and
commit logic all execute.
"""
from __future__ import annotations

import dataclasses
import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import autofix, diagnostics, env_resolve, workspace_overlay
from pedacito.config import EXEC_FIELDS, Config

BIN = "Scripts" if os.name == "nt" else "bin"
MARKER = "UNUSED_IMPORT_FIXME"


# --------------------------------------------------------------------------- #
# fixtures / helpers                                                           #
# --------------------------------------------------------------------------- #

def _write_exec(path: Path, text: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if os.name != "nt":
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _resolved(root: Path, bin_dir: Path) -> env_resolve.ResolvedEnvironment:
    return env_resolve.ResolvedEnvironment(
        root=root, kind="system", python=None, bin_dirs=(bin_dir,),
        package_manager="", workspace_root=root, evidence=())


def _cfg() -> Config:
    cfg = Config()
    cfg.autofix_enabled = True
    cfg.backup = False
    cfg.log_events = False
    return cfg


def _source(root: Path, rel: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"import {MARKER}\n\ndef f():\n    return 1\n", encoding="utf-8")
    return p


@pytest.fixture()
def no_emit(monkeypatch):
    monkeypatch.setattr(autofix, "emit_autofix_result",
                        lambda *a, **k: None)


def _classify(argv):
    if any(t in argv for t in ("--fix", "-w", "--write")):
        return "fix"
    if argv and argv[0] == "cargo" and "fmt" in argv:
        return "report" if "--check" in argv else "fix"
    return "report"


def _fake_run(files, *, out_of_scope=(), mutate_original=(), marker=MARKER):
    """A fake `autofix._run`: reports one finding per marker line and removes
    marker lines on a fix. `out_of_scope` are rels the fix also mutates (in the
    shadow); `mutate_original` are rels whose ORIGINAL is also rewritten."""
    calls: list[tuple[list[str], Path]] = []

    def run(argv, *, cwd=None, cfg=None, child_env=None, bin_dirs=None,
            deadline=None):
        argv = list(argv)
        cwd = Path(cwd) if cwd is not None else Path(".")
        calls.append((argv, cwd))
        if _classify(argv) == "report":
            out = []
            for rel in files:
                p = cwd / rel
                try:
                    if marker in p.read_text(encoding="utf-8"):
                        out.append(f"{rel}:1:1: F401 '{marker}' imported but "
                                   "unused")
                except OSError:
                    continue
            return (1 if out else 0), "\n".join(out), ""
        # fix: strip marker lines from the shadow files
        for rel in list(files) + list(out_of_scope):
            p = cwd / rel
            try:
                text = p.read_text(encoding="utf-8")
                if marker in text:
                    new = "".join(ln for ln in text.splitlines(keepends=True)
                                  if marker not in ln)
                    p.write_text(new, encoding="utf-8")
            except OSError:
                continue
        return 0, "", ""

    return run, calls


def _green_tests(monkeypatch, command="python -m pytest -q"):
    from pedacito import testsel
    monkeypatch.setattr(testsel, "detect_test_command",
                        lambda root, cfg=None: command)
    monkeypatch.setattr(testsel, "run_command",
                        lambda root, cfg, command, timeout=None: (True, ""))


def _fake_tools(monkeypatch, bin_dir: Path):
    """Tool availability only from the fake bin dir (never the system PATH)."""
    def find(resolved, names, *, env=None):
        out = {}
        for n in names:
            exe = bin_dir / (n + (".exe" if os.name == "nt" else ""))
            out[n] = exe if exe.is_file() else None
        return out

    monkeypatch.setattr(env_resolve, "find_executables", find)


# --------------------------------------------------------------------------- #
# contract shapes                                                             #
# --------------------------------------------------------------------------- #

def test_autofix_result_is_frozen_and_has_the_contract_fields():
    r = autofix.AutofixResult((), (), (), (), False, "")
    assert dataclasses.is_dataclass(r)
    assert (r.kept, r.rejected, r.before, r.after, r.verified, r.reason) \
        == ((), (), (), (), False, "")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.verified = True  # type: ignore[misc]


def test_autofix_enabled_is_opt_in_and_trust_gated():
    assert Config().autofix_enabled is False
    assert "autofix_enabled" in EXEC_FIELDS


def test_adapter_definitions_match_the_contract_order_and_argv():
    labels = [a.label for a in (
        autofix._ADAPTERS["ruff"], autofix._ADAPTERS["biome"],
        autofix._ADAPTERS["eslint"], autofix._ADAPTERS["gofmt"],
        autofix._ADAPTERS["cargo-fmt"], autofix._ADAPTERS["cargo-clippy"])]
    assert labels == ["ruff", "biome", "eslint", "gofmt", "cargo fmt",
                      "cargo clippy"]
    assert autofix._ADAPTERS["ruff"].fix_argv == ("ruff", "check", "--fix")
    assert autofix._ADAPTERS["ruff"].report_argv == (
        "ruff", "check", "--quiet", "--output-format", "concise")
    assert autofix._ADAPTERS["biome"].fix_argv == ("biome", "check", "--write")
    assert autofix._ADAPTERS["eslint"].fix_argv == ("eslint", "--fix")
    assert autofix._ADAPTERS["gofmt"].fix_argv == ("gofmt", "-w")
    assert autofix._ADAPTERS["gofmt"].report_argv == ("gofmt", "-l")
    assert autofix._ADAPTERS["cargo-fmt"].fix_argv == ("cargo", "fmt")
    assert autofix._ADAPTERS["cargo-clippy"].fix_argv[0:2] == ("cargo", "clippy")
    # no unsafe clippy options: only the --fix allow flags
    assert autofix._ADAPTERS["cargo-clippy"].fix_argv[2:] == (
        "--fix", "--allow-dirty", "--allow-staged", "--allow-no-vcs")


def test_finding_identities_are_position_erased_and_path_scoped():
    before = {"a.py": ["a.py:12:3: F401 'x' imported but unused"]}
    shifted = {"a.py": ["a.py:99:1: F401 'x' imported but unused"]}
    assert diagnostics.finding_identities(before) == \
        diagnostics.finding_identities(shifted)
    other = {"b.py": ["b.py:12:3: F401 'x' imported but unused"]}
    assert diagnostics.finding_identities(before) != \
        diagnostics.finding_identities(other)
    assert diagnostics.finding_identity("a.py:1:1: line 7 message") == \
        diagnostics.finding_identity("a.py:9:9: line 42 message")


# --------------------------------------------------------------------------- #
# each adapter, via a fake binary + fake runner                                 #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("tool,ext,manifest", [
    ("ruff", ".py", None),
    ("biome", ".js", None),
    ("eslint", ".js", None),
    ("gofmt", ".go", None),
    ("cargo", ".rs", "Cargo.toml"),
])
def test_adapter_fixes_and_commits_on_a_shadow_copy(
        tmp_path, monkeypatch, no_emit, tool, ext, manifest):
    bin_dir = tmp_path / "bin"
    name = tool + (".exe" if os.name == "nt" else "")
    _write_exec(bin_dir / name)
    root = tmp_path / "project"
    rel = f"src/app{ext}"
    _source(root, rel)
    if manifest:
        (root / manifest).write_text("[package]\nname='x'\nversion='0.1.0'\n",
                                     encoding="utf-8")

    files = [rel]
    run, calls = _fake_run(files)
    monkeypatch.setattr(autofix, "_run", run)
    _fake_tools(monkeypatch, bin_dir)
    _green_tests(monkeypatch)

    result = autofix.run_autofix(root, files, cfg=_cfg(),
                                 env=_resolved(root, bin_dir))
    assert result.verified, result.reason
    assert result.kept == (rel,)
    assert result.rejected == ()
    assert MARKER not in (root / rel).read_text(encoding="utf-8")
    # the fix ran on a temp shadow copy, never on the project root
    fix_cwds = [cwd for argv, cwd in calls if _classify(argv) == "fix"]
    assert fix_cwds and all(cwd != root.resolve() for cwd in fix_cwds)


def test_biome_preferred_over_eslint_when_both_installed(
        tmp_path, monkeypatch, no_emit):
    bin_dir = tmp_path / "bin"
    for tool in ("biome", "eslint"):
        _write_exec(bin_dir / (tool + (".exe" if os.name == "nt" else "")))
    root = tmp_path / "project"
    rel = "src/app.js"
    _source(root, rel)
    run, calls = _fake_run([rel])
    monkeypatch.setattr(autofix, "_run", run)
    _fake_tools(monkeypatch, bin_dir)
    _green_tests(monkeypatch)
    result = autofix.run_autofix(root, [rel], cfg=_cfg(),
                                 env=_resolved(root, bin_dir))
    assert result.verified
    first_fix = next((argv for argv, _ in calls if _classify(argv) == "fix"))
    assert first_fix[0] == "biome"


def test_missing_tool_discards(tmp_path, monkeypatch, no_emit):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    root = tmp_path / "project"
    rel = "src/app.py"
    _source(root, rel)
    _fake_tools(monkeypatch, bin_dir)
    _green_tests(monkeypatch)
    result = autofix.run_autofix(root, [rel], cfg=_cfg(),
                                 env=_resolved(root, bin_dir))
    assert not result.verified
    assert result.rejected == (rel,)
    assert "no autofix tool available" in result.reason


# --------------------------------------------------------------------------- #
# sandbox denial                                                               #
# --------------------------------------------------------------------------- #

def test_sandbox_refusal_discards(tmp_path, monkeypatch, no_emit):
    bin_dir = tmp_path / "bin"
    _write_exec(bin_dir / ("ruff" + (".exe" if os.name == "nt" else "")))
    root = tmp_path / "project"
    rel = "src/app.py"
    _source(root, rel)
    _green_tests(monkeypatch)

    def refuse(argv, cfg, root):
        raise autofix.sandbox.SandboxRefused("no backend")

    monkeypatch.setattr(autofix.sandbox, "wrap", refuse)
    _fake_tools(monkeypatch, bin_dir)
    result = autofix.run_autofix(root, [rel], cfg=_cfg(),
                                 env=_resolved(root, bin_dir))
    assert not result.verified
    assert MARKER in (root / rel).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# baseline-red / no-tests                                                      #
# --------------------------------------------------------------------------- #

def test_red_baseline_discards(tmp_path, monkeypatch, no_emit):
    from pedacito import testsel
    bin_dir = tmp_path / "bin"
    _write_exec(bin_dir / ("ruff" + (".exe" if os.name == "nt" else "")))
    root = tmp_path / "project"
    rel = "src/app.py"
    _source(root, rel)
    monkeypatch.setattr(testsel, "detect_test_command",
                        lambda root, cfg=None: "pytest -q")
    monkeypatch.setattr(testsel, "run_command",
                        lambda root, cfg, command, timeout=None: (False, "1 failed"))
    result = autofix.run_autofix(root, [rel], cfg=_cfg(),
                                 env=_resolved(root, bin_dir))
    assert not result.verified
    assert "baseline is not green" in result.reason


def test_no_test_command_discards(tmp_path, monkeypatch, no_emit):
    from pedacito import testsel
    bin_dir = tmp_path / "bin"
    _write_exec(bin_dir / ("ruff" + (".exe" if os.name == "nt" else "")))
    root = tmp_path / "project"
    rel = "src/app.py"
    _source(root, rel)
    monkeypatch.setattr(testsel, "detect_test_command",
                        lambda root, cfg=None: "")
    result = autofix.run_autofix(root, [rel], cfg=_cfg(),
                                 env=_resolved(root, bin_dir))
    assert not result.verified
    assert "no test command" in result.reason


def test_disabled_by_default_is_a_noop(tmp_path, monkeypatch, no_emit):
    root = tmp_path / "project"
    rel = "src/app.py"
    _source(root, rel)
    cfg = _cfg()
    cfg.autofix_enabled = False
    result = autofix.run_autofix(root, [rel], cfg=cfg)
    assert not result.verified
    assert "disabled" in result.reason


# --------------------------------------------------------------------------- #
# out-of-scope mutation + preimage hash mismatch                                #
# --------------------------------------------------------------------------- #

def test_out_of_scope_mutation_discards(tmp_path, monkeypatch, no_emit):
    bin_dir = tmp_path / "bin"
    _write_exec(bin_dir / ("ruff" + (".exe" if os.name == "nt" else "")))
    root = tmp_path / "project"
    rel = "src/app.py"
    other = "src/other.py"
    _source(root, rel)
    _source(root, other)
    _green_tests(monkeypatch)
    run, _calls = _fake_run([rel], out_of_scope=(other,))
    monkeypatch.setattr(autofix, "_run", run)
    _fake_tools(monkeypatch, bin_dir)
    result = autofix.run_autofix(root, [rel], cfg=_cfg(),
                                 env=_resolved(root, bin_dir))
    assert not result.verified
    assert "out-of-scope" in result.reason
    # nothing was committed to the original tree
    assert MARKER in (root / rel).read_text(encoding="utf-8")


def test_preimage_hash_mismatch_discards(tmp_path, monkeypatch, no_emit):
    bin_dir = tmp_path / "bin"
    _write_exec(bin_dir / ("ruff" + (".exe" if os.name == "nt" else "")))
    root = tmp_path / "project"
    rel = "src/app.py"
    _source(root, rel)
    _green_tests(monkeypatch)

    original = root / rel

    def run(argv, *, cwd=None, cfg=None, child_env=None, bin_dirs=None,
            deadline=None):
        argv = list(argv)
        cwd = Path(cwd) if cwd is not None else Path(".")
        if _classify(argv) == "report":
            p = cwd / rel
            has = MARKER in p.read_text(encoding="utf-8")
            return (1 if has else 0), \
                (f"{rel}:1:1: F401 '{MARKER}'" if has else ""), ""
        # fix the shadow, then rewrite the ORIGINAL out from under us
        p = cwd / rel
        text = p.read_text(encoding="utf-8")
        p.write_text("".join(ln for ln in text.splitlines(keepends=True)
                             if MARKER not in ln), encoding="utf-8")
        original.write_text(original.read_text(encoding="utf-8")
                            + "\n# concurrent edit\n", encoding="utf-8")
        return 0, "", ""

    monkeypatch.setattr(autofix, "_run", run)
    _fake_tools(monkeypatch, bin_dir)
    result = autofix.run_autofix(root, [rel], cfg=_cfg(),
                                 env=_resolved(root, bin_dir))
    assert not result.verified
    assert "preimage hash mismatch" in result.reason


# --------------------------------------------------------------------------- #
# no net improvement (findings did not shrink)                                 #
# --------------------------------------------------------------------------- #

def test_no_net_improvement_discards(tmp_path, monkeypatch, no_emit):
    bin_dir = tmp_path / "bin"
    _write_exec(bin_dir / ("ruff" + (".exe" if os.name == "nt" else "")))
    root = tmp_path / "project"
    rel = "src/app.py"
    _source(root, rel)
    _green_tests(monkeypatch)

    def run(argv, *, cwd=None, cfg=None, child_env=None, bin_dirs=None,
            deadline=None):
        # report always finds the marker, but the "fix" changes nothing
        if _classify(argv) == "report":
            return 1, f"{rel}:1:1: F401 '{MARKER}'", ""
        return 0, "", ""

    monkeypatch.setattr(autofix, "_run", run)
    _fake_tools(monkeypatch, bin_dir)
    result = autofix.run_autofix(root, [rel], cfg=_cfg(),
                                 env=_resolved(root, bin_dir))
    assert not result.verified
    assert "no net improvement" in result.reason


# --------------------------------------------------------------------------- #
# shadow copy: no links to the original                                        #
# --------------------------------------------------------------------------- #

def test_copy_tree_no_link_never_links_to_the_original(tmp_path):
    root = tmp_path / "project"
    rel = "src/app.py"
    _source(root, rel)
    (root / "node_modules" / "dep").mkdir(parents=True)
    (root / "node_modules" / "dep" / "index.js").write_text("x", encoding="utf-8")
    dest = tmp_path / "shadow"
    copied = workspace_overlay.copy_tree_no_link(root, dest)
    assert rel in copied
    dest_file = dest / rel
    # an independent copy, not a symlink/hard link
    assert dest_file.is_file()
    assert not dest_file.is_symlink()
    assert dest_file.stat().st_ino != (root / rel).stat().st_ino
    # dependency trees are never linked (and never copied): node_modules stays
    # absent, unlike the exec_root mirror which symlinks/junctions it
    assert not (dest / "node_modules").exists()


def test_copy_tree_no_link_skips_sensitive_files(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "app.py").write_text("x = 1\n", encoding="utf-8")
    (root / ".env").write_text("SECRET=1\n", encoding="utf-8")
    dest = tmp_path / "shadow"
    copied = workspace_overlay.copy_tree_no_link(root, dest)
    assert "app.py" in copied
    assert ".env" not in copied
    assert not (dest / ".env").exists()


def test_copy_tree_no_link_refuses_traversal(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    dest = tmp_path / "shadow"
    copied = workspace_overlay.copy_tree_no_link(root, dest, files=["../outside.txt"])
    assert copied == ()
    assert not (dest / "outside.txt").exists()


# --------------------------------------------------------------------------- #
# resolver: find_executables                                                    #
# --------------------------------------------------------------------------- #

def test_find_executables_reports_each_tool(tmp_path):
    bin_dir = tmp_path / "bin"
    ruff = _write_exec(bin_dir / ("ruff" + (".exe" if os.name == "nt" else "")))
    resolved = _resolved(tmp_path, bin_dir)
    # an empty PATH keeps the fallback from finding host-installed tools
    found = env_resolve.find_executables(resolved, ("ruff", "biome", "gofmt"),
                                         env={"PATH": ""})
    assert found["ruff"] == ruff.resolve()
    assert found["biome"] is None
    assert found["gofmt"] is None


# --------------------------------------------------------------------------- #
# emit metadata without raw output                                             #
# --------------------------------------------------------------------------- #

def test_emit_autofix_result_is_bounded_and_secret_free(tmp_path, monkeypatch):
    from pedacito import events as events_mod
    from pedacito import workspace as workspace_mod

    class Bus:
        def __init__(self):
            self.events = []

        def emit(self, kind, **fields):
            self.events.append((kind, fields))

    bus = Bus()
    monkeypatch.setattr(events_mod, "get", lambda root, cfg=None: bus)
    monkeypatch.setattr(workspace_mod, "log_event", lambda *a, **k: None)
    result = autofix.AutofixResult(("a.py",), (), ("f1", "f2"), (), True,
                                   "ruff fixed 1 file(s)")
    autofix.emit_autofix_result(tmp_path, _cfg(), result)
    assert bus.events and bus.events[-1][0] == "autofix"
    fields = bus.events[-1][1]
    assert fields["verified"] is True
    assert fields["kept"] == ["a.py"]
    assert "secret" not in str(fields)
