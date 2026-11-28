"""R2-BUG-03 regression tests: `pdc new` must not ship a `[project.scripts]`
or `[project.gui-scripts]` entry whose module or top-level symbol does not
exist in the generated tree. Post-build verification used to run `pytest -q`
only, so a scaffold with `addition-module = "main:main"` and no `main.py`
reported green while the installed command would die with ModuleNotFoundError.
"""

import tomllib

from pedacito import config, scaffold


class _StubClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0, "seconds": 0.0}

    def check(self, warn=True):
        return {}

    def chat_json(self, messages, schema, max_tokens=0, **kw):
        return {"files": []}

    def chat(self, messages, **kw):
        return ""


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _pyproject(root, tables):
    return _write(
        root, "pyproject.toml",
        '[project]\nname = "demo"\nversion = "0.1.0"\n\n' + tables)


def test_missing_script_module_is_a_finding(tmp_path):
    _write(tmp_path, "addition.py", "def add(a, b):\n    return a + b\n")
    _pyproject(tmp_path, '[project.scripts]\n'
                         'addition-module = "main:main"\n')

    findings = scaffold.script_entry_findings(tmp_path)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.table == "scripts"
    assert finding.name == "addition-module"
    assert finding.target == "main:main"
    assert "module 'main' not found" in finding.reason
    assert "main.py" in finding.reason


def test_missing_script_function_is_a_finding(tmp_path):
    _write(tmp_path, "main.py", "def run(argv=None):\n    return 0\n")
    _pyproject(tmp_path, '[project.scripts]\ndemo = "main:main"\n')

    findings = scaffold.script_entry_findings(tmp_path)

    assert len(findings) == 1
    assert "does not define or import 'main'" in findings[0].reason
    assert "main.py" in findings[0].reason


def test_gui_script_targets_are_validated_too(tmp_path):
    _write(tmp_path, "main.py", "def main():\n    return 0\n")
    _pyproject(tmp_path, '[project.gui-scripts]\ndemo-gui = "main:missing"\n')

    findings = scaffold.script_entry_findings(tmp_path)

    assert len(findings) == 1
    assert findings[0].table == "gui-scripts"
    assert "does not define or import 'missing'" in findings[0].reason


def test_valid_script_targets_in_flat_and_src_layouts(tmp_path):
    flat = tmp_path / "flat"
    _write(flat, "main.py", "def main(argv=None):\n    return 0\n")
    _pyproject(flat, '[project.scripts]\ndemo = "main:main"\n')
    assert scaffold.script_entry_findings(flat) == []

    src = tmp_path / "src-layout"
    _write(src, "src/acme/cli.py", "def main():\n    return 0\n")
    _pyproject(src, '[project.scripts]\ndemo = "acme.cli:main"\n')
    assert scaffold.script_entry_findings(src) == []

    package = tmp_path / "src-package"
    _write(package, "src/acme/__init__.py", "from .cli import main\n")
    _write(package, "src/acme/cli.py", "def main():\n    return 0\n")
    _pyproject(package, '[project.scripts]\ndemo = "acme:main"\n')
    assert scaffold.script_entry_findings(package) == []


def test_imported_script_symbol_counts_as_bound(tmp_path):
    _write(tmp_path, "app.py", "def main():\n    return 0\n")
    _write(tmp_path, "main.py", "from app import main\n")
    _pyproject(tmp_path, '[project.scripts]\ndemo = "main:main"\n')

    assert scaffold.script_entry_findings(tmp_path) == []


def test_gate_removes_only_broken_entries(tmp_path, capsys):
    _write(tmp_path, "main.py", "def main():\n    return 0\n")
    _pyproject(tmp_path, """\
[project.scripts]
demo = "main:main"
ghost = "main:nope"

[project.gui-scripts]
ghost-gui = "missing_app:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
""")

    remaining = scaffold._script_entry_gate(tmp_path)

    assert remaining == []
    text = (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    data = tomllib.loads(text)
    assert data["project"]["scripts"] == {"demo": "main:main"}
    assert "gui-scripts" not in data["project"]
    assert "[project.gui-scripts]" not in text
    assert "[build-system]" in text
    err = capsys.readouterr().err
    assert "entry point removed from pyproject.toml" in err
    assert "ghost" in err and "ghost-gui" in err


def test_inline_script_table_is_reported_not_rewritten(tmp_path):
    _pyproject(tmp_path, 'scripts = { ghost = "main:main" }\n')

    findings = scaffold.script_entry_findings(tmp_path)
    remaining = scaffold._script_entry_gate(tmp_path)

    assert len(findings) == 1
    assert remaining == findings
    text = (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    assert 'ghost = "main:main"' in text


def test_run_new_removes_broken_starter_script(tmp_path, monkeypatch,
                                               capsys):
    """The R2-BUG-03 repro: the starter pyproject declares `main:main`, but
    the plan (and build) only provides `addition.py`; `pdc new` must not leave
    that broken console script behind, and must say it removed it."""
    root = tmp_path / "tiny"
    spec = scaffold.Spec("A tiny Python module with add(a, b) and one "
                         "pytest test")
    plan = scaffold.Plan(
        "addition-module", "a tiny Python module",
        [scaffold.PlannedFile("addition.py", "add(a, b)"),
         scaffold.PlannedFile("test_addition.py", "one pytest test")],
        spec=spec)

    def fake_build(client, cfg, idx, planned, **kw):
        _write(root, "addition.py", "def add(a, b):\n    return a + b\n")
        _write(root, "test_addition.py",
               "from addition import add\n\n\ndef test_add():\n"
               "    assert add(2, 3) == 5\n")
        return {"done": ["addition.py", "test_addition.py"], "noop": [],
                "failed": [], "skipped": []}

    monkeypatch.setattr(scaffold, "build_plan", lambda *a, **k: plan)
    monkeypatch.setattr(scaffold, "build_interface", lambda *a, **k: {})
    monkeypatch.setattr(scaffold, "build_project", fake_build)
    monkeypatch.setattr(scaffold, "build_docs",
                        lambda *a, **k: {"done": [], "noop": [],
                                         "failed": [], "skipped": []})
    monkeypatch.setattr(scaffold, "verify_and_fix", lambda *a, **k: (True, ""))
    monkeypatch.setattr(scaffold, "_ensure_git_repo", lambda *a, **k: False)

    cfg = config.load(require_hardware=False)
    cfg.backup = False
    cfg.log_sessions = False
    code = scaffold.run_new(root, cfg, _StubClient(cfg), spec,
                            apply=True, verbose=False)

    assert code == 0
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "main:main" not in text
    assert "[project.scripts]" not in text
    assert tomllib.loads(text)["project"]["name"] == "addition-module"
    assert "[tool.pytest.ini_options]" in text
    err = capsys.readouterr().err
    assert "entry point removed from pyproject.toml" in err
