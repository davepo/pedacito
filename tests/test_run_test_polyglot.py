"""`run_test` dispatches non-Python test targets without a shell (#22)."""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import tools
from pedacito.config import Config


def _setup(tmp_path, files):
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    cfg = Config()
    cfg.sandbox = False
    cfg._root = str(tmp_path)
    idx = types.SimpleNamespace(root=tmp_path)
    return idx, cfg


def test_python_target_falls_through_to_pytest(tmp_path):
    _idx, cfg = _setup(tmp_path, {"tests/test_app.py": "def test_x():\n "
                                                  "    assert True\n"})
    assert tools._polyglot_test_argv(tmp_path, cfg, "tests/test_app.py",
                                     "") is None


def test_node_builtin_runner_for_js_and_ts_tests(tmp_path):
    _idx, cfg = _setup(tmp_path, {"tests/app.test.js": "test('x',()=>{});\n",
                                  "tests/app.spec.js": "test('x',()=>{});\n",
                                  "tests/app.test.ts": "test('x',()=>{});\n"})
    argv, kind = tools._polyglot_test_argv(
        tmp_path, cfg, "tests/app.test.js", "")
    assert argv == ["node", "--test", "tests/app.test.js"] and kind == "js"
    argv, _kind = tools._polyglot_test_argv(
        tmp_path, cfg, "tests/app.test.ts", "save")
    assert argv == ["node", "--test", "tests/app.test.ts",
                    "--test-name-pattern", "save"]


def test_vitest_is_detected_from_package_json(tmp_path):
    _idx, cfg = _setup(tmp_path, {
        "package.json": '{"scripts": {"test": "vitest run"}, '
                        '"devDependencies": {"vitest": "^1"}}\n',
        "src/app.test.ts": "test('x',()=>{});\n",
    })
    argv, kind = tools._polyglot_test_argv(
        tmp_path, cfg, "src/app.test.ts", "save or retry")
    assert kind == "js"
    assert argv == ["npx", "--no-install", "vitest", "run",
                    "src/app.test.ts", "-t", "save or retry"]


def test_go_file_and_directory_targets(tmp_path):
    _idx, cfg = _setup(tmp_path, {"pkg/calc_test.go": "package pkg\n"})
    argv, kind = tools._polyglot_test_argv(
        tmp_path, cfg, "pkg/calc_test.go", "TestAdd")
    assert kind == "go"
    assert argv == ["go", "test", "./pkg", "-run", "TestAdd"]
    argv, _kind = tools._polyglot_test_argv(
        tmp_path, cfg, "pkg", "")
    assert argv == ["go", "test", "./pkg"]


def test_rust_integration_test_target(tmp_path):
    _idx, cfg = _setup(tmp_path, {"tests/integration.rs": "#[test]\n"
                                                     "fn ok() {}\n"})
    argv, kind = tools._polyglot_test_argv(
        tmp_path, cfg, "tests/integration.rs", "ok")
    assert kind == "rust"
    assert argv == ["cargo", "test", "--test", "integration", "ok"]


def test_go_targets_are_not_confused_with_rust(tmp_path):
    _idx, cfg = _setup(tmp_path, {"src/lib.rs": "#[test]\nfn x() {}\n",
                                  "pkg/a_test.go": "package pkg\n"})
    # A .rs outside tests/ is not a cargo integration target.
    assert tools._polyglot_test_argv(tmp_path, cfg, "src/lib.rs", "") is None


def test_option_injection_is_refused(tmp_path):
    _idx, cfg = _setup(tmp_path, {"tests/app.test.js": "test('x',()=>{});\n"})
    out = tools._polyglot_test_argv(tmp_path, cfg, "tests/app.test.js", "-v")
    assert isinstance(out, str) and out.startswith("ERROR")
    assert "cannot start with '-'" in out


def test_path_traversal_is_refused(tmp_path):
    _idx, cfg = _setup(tmp_path, {"tests/app.test.js": "test('x',()=>{});\n"})
    out = tools._polyglot_test_argv(
        tmp_path, cfg, "../../etc/passwd", "")
    assert isinstance(out, str) and out.startswith("ERROR")
    assert "refused the target" in out


def test_dispatch_runs_js_argv_sandboxed_and_capped(tmp_path, monkeypatch):
    idx, cfg = _setup(tmp_path, {
        "tests/app.test.js": "test('x',()=>{});\n",
        "package.json": '{"devDependencies": {"vitest": "^1"}}\n',
    })
    seen = {}

    def fake_run(argv, cwd, timeout=0, idle_timeout=0, max_output=0, env=None,
                 **kw):
        seen["argv"] = list(argv)
        seen["timeout"] = timeout
        seen["env"] = env
        return 0, "Tests  2 passed", "", 0.1

    monkeypatch.setattr(tools.proc, "run", fake_run)
    out = tools.dispatch(idx, cfg, {"action": "run_test",
                                    "target": "tests/app.test.js"})
    assert out.startswith("RUN TEST:")
    assert seen["argv"][:5] == ["npx", "--no-install", "vitest", "run",
                                "tests/app.test.js"]
    assert seen["timeout"] <= 15
    # scrubbed environment: an ambient API key never reaches the child
    assert "OPENAI_API_KEY" not in (seen["env"] or {})
    assert all(";" not in a and "|" not in a for a in seen["argv"])
