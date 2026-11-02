"""Diagnostics detection/run and the `diagnose` tool."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import diagnostics, tools
from pedacito.config import Config
from pedacito.index import Index


def test_detect_prefers_ruff_and_targets_python_files(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.setattr(diagnostics.shutil, "which",
                        lambda name: "/usr/bin/" + name
                        if name in ("ruff", "npm") else "")
    checks = diagnostics.detect(tmp_path, ["pkg/app.py", "web/app.ts"])
    assert checks
    assert checks[0].label == "ruff"
    assert "pkg/app.py" in checks[0].argv
    assert "web/app.ts" not in checks[0].argv


def test_flake8_fallback_argv_is_a_list(tmp_path, monkeypatch):
    (tmp_path / "setup.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(diagnostics.shutil, "which",
                        lambda name: "/usr/bin/flake8" if name == "flake8" else "")
    checks = diagnostics.detect(tmp_path, ["app.py"])
    assert [c.argv for c in checks] == [["flake8", "app.py"]]


def test_no_detected_checker_is_not_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: "")
    ok, report = diagnostics.run(tmp_path, Config())
    assert ok
    assert "No linter/type checker detected" in report


def test_run_reports_failures_and_respects_the_output_cap(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.setattr(diagnostics.shutil, "which",
                        lambda name: "/usr/bin/ruff" if name == "ruff" else "")

    def fake_run(argv, **kw):
        return 1, "E501 line too long\n" + "x" * 5000, "", 0.2

    monkeypatch.setattr(diagnostics.proc, "run", fake_run)
    cfg = Config()
    cfg.max_diagnostics_chars = 300
    ok, report = diagnostics.run(tmp_path, cfg, ["app.py"])
    assert not ok
    assert "CHECK ruff: FAIL" in report
    assert "E501" in report
    assert len(report) <= 360


def test_diagnose_tool_is_gated_by_allow_shell(tmp_path):
    idx = Index(tmp_path, [])
    obs = tools.tool_diagnose(idx, Config())
    assert obs.startswith("ERROR") and "allow_shell" in obs


def test_diagnose_tool_runs_detected_checks(tmp_path, monkeypatch):
    idx = Index(tmp_path, [])
    cfg = Config()
    cfg.allow_shell = True
    seen = []

    def fake_run_changed(root, cfg_, files):
        seen.extend(files or [])
        return True, "CHECK ruff: PASS"

    monkeypatch.setattr(diagnostics, "run_changed", fake_run_changed)
    obs = tools.tool_diagnose(idx, cfg)
    assert "CHECK ruff: PASS" in obs
    assert seen == []


# ---------------------------------------------- quick findings (post-edit loop)
def test_quick_findings_python_syntax_error_via_ast_fallback(tmp_path,
                                                             monkeypatch):
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: "")
    (tmp_path / "bad.py").write_text("def f(:\n    pass\n", encoding="utf-8")
    (tmp_path / "good.py").write_text("x = 1\n", encoding="utf-8")
    found = diagnostics.quick_findings(tmp_path, ["bad.py", "good.py"])
    assert found["good.py"] == []
    assert len(found["bad.py"]) == 1
    assert found["bad.py"][0].startswith("bad.py:1:")
    assert "syntax error" in found["bad.py"][0]


def test_quick_python_fallback_flags_undefined_names(tmp_path, monkeypatch):
    """P9: with no linter installed, the fallback still catches the
    NameError class `ast.parse` waves through -- a name loaded but bound
    nowhere (deepclitest workers kept shipping files that referenced
    helpers no longer present after a botched from-memory rewrite)."""
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: "")
    (tmp_path / "m.py").write_text(
        "import os\n"
        "\n"
        "def f(a):\n"
        "    return os.path.join(a, g(a))\n", encoding="utf-8")
    found = diagnostics.quick_findings(tmp_path, ["m.py"])["m.py"]
    assert len(found) == 1
    assert found[0].startswith("m.py:4:")
    assert "undefined name 'g'" in found[0]


def test_quick_python_fallback_keeps_quiet_on_sound_code(tmp_path,
                                                          monkeypatch):
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: "")
    (tmp_path / "ok.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "CONST = 1\n"
        "\n"
        "def helper(x):\n"
        "    return x * CONST\n"
        "\n"
        "class C:\n"
        "    attr = [helper(i) for i in range(3)]\n"
        "\n"
        "    def method(self):\n"
        "        total = sum(v for v in self.items)\n"
        "        return json.dumps({'t': total, 'p': str(Path('.'))})\n"
        "\n"
        "def outer():\n"
        "    count = 0\n"
        "    def inner():\n"
        "        nonlocal count\n"
        "        count += 1\n"
        "    inner()\n"
        "    squares = [y := n * n for n in range(3)]\n"
        "    return count, y, squares\n"
        "\n"
        "def uses_later():\n"
        "    return later_defined()      # forward reference: legal\n"
        "\n"
        "def later_defined():\n"
        "    return 42\n", encoding="utf-8")
    assert diagnostics.quick_findings(tmp_path, ["ok.py"])["ok.py"] == []


def test_quick_python_fallback_skips_dynamic_name_sources(tmp_path,
                                                           monkeypatch):
    """Star-imports and eval/exec can bring in any name at runtime, so the
    whole file is skipped rather than risking a false alarm."""
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: "")
    (tmp_path / "star.py").write_text(
        "from os.path import *\nprint(join('a', 'b'))\n", encoding="utf-8")
    (tmp_path / "dyn.py").write_text(
        "exec('x = 1')\nprint(x)\n", encoding="utf-8")
    found = diagnostics.quick_findings(tmp_path, ["star.py", "dyn.py"])
    assert found["star.py"] == []
    assert found["dyn.py"] == []


def test_quick_python_fallback_respects_class_scope_rules(tmp_path,
                                                           monkeypatch):
    """A class body is not part of the lookup chain of the scopes nested in
    it: a method (or a comprehension!) reaching for a class attribute is a
    genuine NameError and gets flagged; the class body itself is fine."""
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: "")
    (tmp_path / "c.py").write_text(
        "class C:\n"
        "    LIMIT = 10\n"
        "    doubled = [v * 2 for v in range(LIMIT)]\n"
        "    same = LIMIT * 2\n"
        "\n"
        "    def m(self):\n"
        "        return LIMIT\n", encoding="utf-8")
    found = diagnostics.quick_findings(tmp_path, ["c.py"])["c.py"]
    assert any(x.startswith("c.py:3:") and "undefined name 'LIMIT'" in x
               for x in found)
    assert any(x.startswith("c.py:7:") and "undefined name 'LIMIT'" in x
               for x in found)
    assert not any(x.startswith("c.py:4:") for x in found)


def test_quick_python_fallback_caps_runaway_findings(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: "")
    src = "".join(f"print(undefined_{i})\n" for i in range(45))
    (tmp_path / "many.py").write_text(src, encoding="utf-8")
    found = diagnostics.quick_findings(tmp_path, ["many.py"])["many.py"]
    assert len(found) == diagnostics._MAX_UNDEFINED + 1
    assert "more undefined names" in found[-1] and "fragment" in found[-1]


def test_new_findings_surfaces_an_introduced_undefined_name(tmp_path,
                                                             monkeypatch):
    """End-to-end through the post-edit diff: an undefined name the edit
    introduced is reported, pre-existing findings are not."""
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: "")
    (tmp_path / "a.py").write_text("def f():\n    return stale_thing\n",
                                   encoding="utf-8")
    base = diagnostics.quick_findings(tmp_path, ["a.py"])
    (tmp_path / "a.py").write_text(
        "def f():\n    return stale_thing + missing\n", encoding="utf-8")
    cur = diagnostics.quick_findings(tmp_path, ["a.py"])
    new = diagnostics.new_findings(base, cur)
    assert [x for x in new["a.py"] if "'missing'" in x]
    assert not [x for x in new["a.py"] if "'stale_thing'" in x]


def test_quick_findings_prefers_an_installed_linter(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics.shutil, "which",
                        lambda name: "/usr/bin/ruff" if name == "ruff" else "")
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        return 1, "app.py:2:5: F821 undefined name 'y'\n", "", 0.1

    monkeypatch.setattr(diagnostics.proc, "run", fake_run)
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    found = diagnostics.quick_findings(tmp_path, ["app.py"])
    assert calls == [["ruff", "check", "--output-format", "concise", "app.py"]]
    assert found["app.py"] == ["app.py:2:5: F821 undefined name 'y'"]


def test_quick_findings_js_uses_tree_sitter(tmp_path):
    from pedacito.chunker_ts import TS_AVAILABLE
    (tmp_path / "app.js").write_text("function f( {\n", encoding="utf-8")
    (tmp_path / "ok.js").write_text("function f() { return 1; }\n",
                                    encoding="utf-8")
    found = diagnostics.quick_findings(tmp_path, ["app.js", "ok.js"])
    if TS_AVAILABLE:
        assert found["app.js"] and "syntax error" in found["app.js"][0]
        assert found["ok.js"] == []
    else:
        assert found["app.js"] == [] and found["ok.js"] == []


def test_quick_findings_skips_uncheckable_and_missing_files(tmp_path,
                                                            monkeypatch):
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: "")
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
    found = diagnostics.quick_findings(tmp_path,
                                       ["notes.txt", "gone.py", "", None])
    assert "notes.txt" not in found      # no cheap checker for .txt
    assert found["gone.py"] == []        # missing on disk: no crash


def test_new_findings_ignores_shifted_positions():
    baseline = {"app.py": ["app.py:3:1: E501 line too long"],
                "util.py": ["util.py: line 12 unused"]}
    current = {"app.py": ["app.py:9:1: E501 line too long",
                          "app.py:10:5: F821 undefined name 'y'"],
               "util.py": ["util.py: line 14 unused"]}
    new = diagnostics.new_findings(baseline, current)
    assert list(new) == ["app.py"]
    assert new["app.py"] == ["app.py:10:5: F821 undefined name 'y'"]


def test_new_findings_keeps_files_missing_from_the_baseline():
    new = diagnostics.new_findings({}, {"fresh.py": ["fresh.py:1:1: boom"]})
    assert new == {"fresh.py": ["fresh.py:1:1: boom"]}
    assert diagnostics.new_findings({"a.py": ["x"]}, {}) == {}


def test_render_findings_joins_and_caps():
    found = {"a.py": [f"a.py:{i}:1: problem {i}" for i in range(400)]}
    text = diagnostics.render_findings(found, cap=500)
    assert text.endswith("[... more findings truncated]")
    assert len(text) <= 500 + len("\n[... more findings truncated]")
    assert diagnostics.render_findings({}, 4000) == ""
    both = diagnostics.render_findings({"a.py": ["one"], "b.py": ["two"]})
    assert both == "one\ntwo"
