"""Deterministic verification upgrades.

Three zero-model-token checks:

1. Import auto-healing: an F821 whose fix is mechanically unique (a stdlib
   top-level module, or a public symbol defined by exactly one project file)
   is inserted and re-validated in memory before any repair round; ambiguous
   names still go to the model.
2. pytest fixture contracts: conftest.py fixtures are parsed with `ast` and
   appended to CONTRACT_HEADER so the model reuses real fixtures.
3. JS/TS ES module contracts: relative imports must resolve and name real
   exports; imported-function calls must satisfy the known arity. Bare
   packages, dynamic imports, `...rest`, `arguments`, and shadows are skipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import loop, spec, verify  # noqa: E402
from pedacito.config import Config  # noqa: E402
from pedacito.edits import Result  # noqa: E402


def _result(before: str, after: str, rel: str = "app.py") -> Result:
    return Result(True, rel, "applied", before=before, after=after)


def _force_ast(monkeypatch):
    monkeypatch.setattr(spec, "_ruff_command", lambda: [])


# ---------------------------------------------------------- import auto-heal

def test_auto_heal_stdlib_import_after_future_and_docstring(tmp_path):
    src = ('"""Module docs."""\n'
           "from __future__ import annotations\n\n\n"
           "def dump(value):\n"
           "    return json.dumps(value)\n")
    r = _result("", src)
    findings = verify.check_static([r], root=tmp_path)

    assert not [f for f in findings if f.kind == "static"]
    assert "import json" in r.after
    lines = r.after.splitlines()
    assert lines.index("import json") > lines.index(
        "from __future__ import annotations")
    assert any(f.kind == "auto_heal"
               and "[AUTO-HEALED: added 'import json' in app.py]" in f.message
               for f in findings)


def test_auto_heal_unique_repo_symbol_uses_its_module(tmp_path):
    (tmp_path / "lib.py").write_text(
        "def unique_helper(value):\n    return value\n", encoding="utf-8")
    src = "def use():\n    return unique_helper(1)\n"
    r = _result("", src)
    findings = verify.check_static([r], root=tmp_path)

    assert not [f for f in findings if f.kind == "static"]
    assert "from lib import unique_helper" in r.after
    assert any("[AUTO-HEALED: added 'from lib import unique_helper' in app.py]"
               in f.message for f in findings if f.kind == "auto_heal")


def test_auto_heal_skips_ambiguous_and_unknown_names(tmp_path):
    (tmp_path / "a.py").write_text("def shared_name(x):\n    return x\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def shared_name(y):\n    return y\n", encoding="utf-8")
    src = "def use():\n    return shared_name(1) + totally_missing\n"
    r = _result("", src)
    findings = verify.check_static([r], root=tmp_path)

    static = [f for f in findings if f.kind == "static"]
    assert static, findings
    assert "shared_name" in static[0].message
    assert "totally_missing" in static[0].message
    assert r.after == src                    # ambiguous: untouched buffer


def test_auto_heal_skips_already_pre_existing_error(tmp_path):
    src = "def use():\n    return json.dumps({})\n"
    r = _result(src, src)
    assert verify.check_static([r], root=tmp_path) == []
    assert r.after == src                    # baseline says it was already there


def test_auto_heal_can_be_disabled(tmp_path):
    src = "def use():\n    return json.dumps({})\n"
    cfg = Config()
    cfg.auto_heal_imports = False
    r = _result("", src)
    findings = verify.check_static([r], root=tmp_path, cfg=cfg)
    assert any(f.kind == "static" for f in findings)
    assert r.after == src


def test_auto_heal_persists_only_when_disk_matches_buffer(tmp_path):
    src = "def use():\n    return os.getcwd()\n"
    (tmp_path / "app.py").write_text(src, encoding="utf-8")
    r = _result("", src)
    receipts = verify.auto_heal_static([r], root=tmp_path, cfg=Config(),
                                       persist=True)
    assert receipts and "import os" in (tmp_path / "app.py").read_text(encoding="utf-8")
    assert "import os" in r.after


def test_gate_heals_disk_before_returning_new_errors(tmp_path, monkeypatch):
    _force_ast(monkeypatch)
    (tmp_path / "m.py").write_text("print(hashlib.sha256(b'x'))\n", encoding="utf-8")
    cfg = Config()
    cfg._static_baseline = set()
    assert loop._new_static_errors(tmp_path, cfg, ["m.py"]) == []
    assert "import hashlib" in (tmp_path / "m.py").read_text(encoding="utf-8")


# --------------------------------------------------------- pytest fixtures

def test_extract_project_contracts_includes_conftest_fixtures(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "conftest.py").write_text(
        "import pytest\n"
        "from pytest import fixture\n\n"
        "@pytest.fixture(scope='module')\n"
        "def db_session(tmp_path):\n"
        '    """A temporary database."""\n'
        "    return tmp_path\n\n"
        "@fixture\n"
        "def client(app):\n"
        "    yield app\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("def run(x):\n    return x\n", encoding="utf-8")

    got = spec.extract_project_contracts(tmp_path, task="add tests for app.py")
    assert "# pytest fixtures (conftest.py)" in got
    assert "def db_session(tmp_path)" in got
    assert "scope=module" in got
    assert "A temporary database." in got
    assert "def client(app)" in got and "yields app" in got
    assert got.count("# pytest fixtures") == 1

    # The same subsection respects the existing ~300-token budget.
    assert len(got) <= 1400
    assert spec.extract_project_contracts(
        tmp_path, task="add tests", include_fixtures=False).count(
            "# pytest fixtures") == 0


def test_fixture_section_survives_without_public_python_files(tmp_path):
    (tmp_path / "conftest.py").write_text(
        "@pytest.fixture\ndef db():\n    return 1\n", encoding="utf-8")
    block = spec.contract_map_block(tmp_path, task="write tests")
    assert "# pytest fixtures (conftest.py)" in block
    assert "def db()" in block


# --------------------------------------------------------- JS/TS contracts

def test_js_broken_named_import_and_wrong_arity(tmp_path):
    (tmp_path / "mod.js").write_text(
        "export function add(a, b = 1) {\n  return a + b\n}\n"
        "export const helper = (x) => x\n", encoding="utf-8")
    (tmp_path / "app.js").write_text(
        "import { add, missing, helper } from './mod.js'\n"
        "import { ext } from 'react'\n"
        "import def from './missing-file.js'\n"
        "const dyn = import('./dynamic.js')\n"
        "add()\n"
        "add(1, 2, 3)\n"
        "helper(1, 2)\n", encoding="utf-8")

    errs = spec.verify_cross_file_contracts(tmp_path)
    assert any("imports 'missing' from mod.js" in e for e in errs)
    assert any("imports './missing-file.js'" in e for e in errs)
    assert any("calls add(...)" in e and "1..2" in e for e in errs)
    assert any("calls helper(...)" in e and "exactly 1" in e for e in errs)
    assert all(e.startswith("ContractError in app.js:") for e in errs)
    assert not any("react" in e or "dynamic.js" in e for e in errs)


def test_js_skips_rest_arguments_and_external_packages(tmp_path):
    (tmp_path / "mod.js").write_text(
        "export function collect(a, ...rest) {\n  return a\n}\n"
        "export function argsy(a) {\n  return arguments.length\n}\n", encoding="utf-8")
    (tmp_path / "app.js").write_text(
        "import { collect, argsy } from './mod.js'\n"
        "import { something } from 'alpinejs'\n"
        "collect()\n"
        "argsy()\n"
        "something()\n", encoding="utf-8")
    assert spec.verify_cross_file_contracts(tmp_path) == []


def test_ts_import_resolves_js_specifier_and_optional_params(tmp_path):
    (tmp_path / "mod.ts").write_text(
        "export function greet(name: string, title?: string, punct = '!'): "
        "string { return name + punct }\n"
        "export type Shape = { x: number }\n", encoding="utf-8")
    (tmp_path / "app.ts").write_text(
        "import { greet, Shape } from './mod.js'\n"
        "const s: Shape = { x: 1 }\n"
        "greet('a')\n"
        "greet('a', 'b', 'c', 'd')\n", encoding="utf-8")

    errs = spec.verify_cross_file_contracts(tmp_path)
    assert len(errs) == 1, errs
    assert "calls greet(...)" in errs[0] and "1..3" in errs[0]


def test_frontend_gate_carries_js_module_errors_and_baseline(tmp_path):
    (tmp_path / "mod.js").write_text("export const one = 1\n", encoding="utf-8")
    (tmp_path / "app.js").write_text(
        "import { none } from './mod.js'\nconsole.log(none)\n", encoding="utf-8")

    assert any("imports 'none'" in e
               for e in spec.verify_frontend_contracts(tmp_path))

    cfg = Config()
    loop._capture_frontend_baseline(tmp_path, cfg)
    assert loop._new_frontend_errors(tmp_path, cfg, ["app.js"]) == []


def test_contract_baseline_ignores_pre_existing_untouched_error(tmp_path):
    (tmp_path / "lib.py").write_text("def helper(a):\n    return a\n", encoding="utf-8")
    (tmp_path / "user.py").write_text(
        "from lib import vanished\nvanished(1)\n", encoding="utf-8")
    cfg = Config()
    loop._capture_contract_baseline(tmp_path, cfg)

    # An unrelated edit must not inherit the pre-existing violation.
    assert loop._new_contract_errors(tmp_path, cfg, ["other.py"]) == []
    # Touching the offending file keeps it visible: the task is on that file.
    assert loop._new_contract_errors(tmp_path, cfg, ["user.py"])


def test_contract_error_key_ignores_only_the_line(tmp_path):
    a = "ContractError in app.js:4: imports 'x' from mod.js"
    b = "ContractError in app.js:99: imports 'x' from mod.js"
    c = "ContractError in app.js:99: imports 'y' from mod.js"
    assert spec.contract_error_key(a) == spec.contract_error_key(b)
    assert spec.contract_error_key(a) != spec.contract_error_key(c)
    assert spec.contract_error_file(a) == "app.js"
