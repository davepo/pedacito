"""Cargo test narrowing, bounded to provably-safe cases (TECHNIQUES.md #48).

`cargo test --test NAME` runs ONLY that integration target: the src unit
tests and every other target are skipped. Narrowing is therefore sound only
when the change is confined to top-level integration test files (tests/*.rs)
-- each is its own crate, so no other target can be affected and the changed
targets ARE the whole change. Unknown changed files, any src/manifest/helper
change, or an already target-selected command keep the full suite.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import cli, testsel  # noqa: E402
from pedacito import index as index_mod  # noqa: E402
from pedacito.config import Config  # noqa: E402


def _proj(tmp_path, files):
    root = tmp_path / "proj"
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    cfg = Config()
    idx = index_mod.build([str(root)], cfg, client=None, summarise=False,
                          root=root, verbose=False)
    return root, idx, cfg


# ------------------------------------------------------------- narrow_command
def test_cargo_narrows_when_only_integration_tests_changed():
    assert testsel.narrow_command("cargo test", ["tests/a.rs"],
                                  changed=["tests/a.rs"]) \
        == ("cargo test --test a", True)


def test_cargo_narrow_sorts_and_dedupes_targets():
    assert testsel.narrow_command(
        "cargo test", ["tests/b.rs", "tests/a.rs"],
        changed=["tests/b.rs", "tests/a.rs", "tests/b.rs"]) \
        == ("cargo test --test a --test b", True)


@pytest.mark.parametrize("changed", [
    ["src/lib.rs"],                          # src unit tests would be skipped
    ["src/lib.rs", "tests/a.rs"],            # mixed: never narrow
    ["Cargo.toml"],                          # manifest: rebuild everything
    ["README.md"],                           # non-source: stay conservative
    ["tests/common/mod.rs"],                 # shared helper: other targets use it
    ["tests/foo/main.rs"],                   # nested target: not a plain --test
    [],                                      # nothing known
    None,                                    # unknown (legacy callers)
])
def test_cargo_does_not_narrow_unless_change_is_integration_only(changed):
    assert testsel.narrow_command("cargo test", ["tests/a.rs"],
                                  changed=changed) == ("cargo test", False)


@pytest.mark.parametrize("cmd", [
    "cargo test --test a",
    "cargo test --lib",
    "cargo test --bin foo",
    "cargo test --bins",
    "cargo test --doc",
    "cargo test --example ex",
    "cargo test --examples",
    "cargo test --bench b",
    "cargo test --benches",
    "cargo test --all-targets",
])
def test_cargo_keeps_an_already_target_selected_command(cmd):
    assert testsel.narrow_command(cmd, ["tests/a.rs"],
                                  changed=["tests/a.rs"]) == (cmd, False)


def test_cargo_narrow_keeps_other_flags():
    assert testsel.narrow_command("cargo test --release", ["tests/a.rs"],
                                  changed=["tests/a.rs"]) \
        == ("cargo test --release --test a", True)
    # --test-threads is NOT a target selector: narrowing stays allowed
    assert testsel.narrow_command("cargo test --test-threads=1", ["tests/a.rs"],
                                  changed=["tests/a.rs"]) \
        == ("cargo test --test-threads=1 --test a", True)


def test_cargo_narrow_needs_impacted_tests():
    assert testsel.narrow_command("cargo test", [],
                                  changed=["tests/a.rs"]) \
        == ("cargo test", False)


def test_other_runners_ignore_the_changed_argument():
    assert testsel.narrow_command("pytest -q", ["tests/test_a.py"],
                                  changed=["src/a.py"]) \
        == ("pytest -q tests/test_a.py", True)
    assert testsel.narrow_command("go test ./...", ["store/store_test.go"],
                                  changed=["store/store.go"]) \
        == ("go test ./store/...", True)


# --------------------------------------------------------------- run_impacted
def test_run_impacted_narrows_cargo_for_integration_only_changes(tmp_path,
                                                                 monkeypatch):
    root, idx, cfg = _proj(tmp_path, {
        "Cargo.toml": "[package]\nname = 'x'\nversion = '0.1.0'\n",
        "src/lib.rs": "pub fn add(a: i32, b: i32) -> i32 { a + b }\n",
        "tests/a.rs": "#[test]\nfn it_works() { assert_eq!(2 + 2, 4); }\n",
    })
    calls = []

    def stub_run(root_, cfg_, command, timeout=None):
        calls.append(command)
        return True, "test result: ok."

    monkeypatch.setattr(testsel, "run_command", stub_run)
    ok, _out, desc = testsel.run_impacted(idx, cfg, ["tests/a.rs"])
    assert ok
    assert calls == ["cargo test --test a"]
    assert "impacted tests" in desc


def test_run_impacted_keeps_full_cargo_suite_for_src_changes(tmp_path,
                                                             monkeypatch):
    root, idx, cfg = _proj(tmp_path, {
        "Cargo.toml": "[package]\nname = 'x'\nversion = '0.1.0'\n",
        "src/lib.rs": "pub fn add(a: i32, b: i32) -> i32 { a + b }\n",
        "tests/a.rs": "#[test]\nfn it_works() { assert_eq!(2 + 2, 4); }\n",
    })
    calls = []

    def stub_run(root_, cfg_, command, timeout=None):
        calls.append(command)
        return True, "test result: ok."

    monkeypatch.setattr(testsel, "run_command", stub_run)
    ok, _out, desc = testsel.run_impacted(idx, cfg, ["src/lib.rs"])
    assert ok
    assert calls == ["cargo test"]
    assert "could not narrow" in desc


# ------------------------------------------------------------------ CLI seat
def test_cmd_tests_reports_the_narrowed_cargo_command(tmp_path, monkeypatch,
                                                      capsys):
    root, idx, _cfg = _proj(tmp_path, {
        "Cargo.toml": "[package]\nname = 'x'\nversion = '0.1.0'\n",
        "tests/a.rs": "#[test]\nfn it_works() { assert_eq!(2 + 2, 4); }\n",
    })
    monkeypatch.setattr(cli.index_mod.Index, "load", staticmethod(lambda r: idx))
    args = argparse.Namespace(root=str(root), files=["tests/a.rs"],
                              config=None, profile=None, paths=[])
    assert cli._cmd_tests(args) == 0
    out = capsys.readouterr().out
    assert "narrowed: cargo test --test a" in out
