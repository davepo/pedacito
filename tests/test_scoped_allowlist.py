"""Scoped persistent shell allow-rules (v0.3.2 M1.2 / WP-3203).

`shell_allowlist_by_project` maps project roots to command patterns a human
persisted through the GUI/TUI "allow in this project" answer. Covers the
pattern suggestion/matching engine (argv-aware, never matching chaining or
injection metacharacters, never covering installs or shells), the TOML
round-trip and trust gating, persistence into the user config, and the
end-to-end gates in `run` / `run_test` and both brokers.
"""
from __future__ import annotations

import io
import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pedacito import config as config_mod  # noqa: E402
from pedacito import config_editor, security, tools  # noqa: E402
from pedacito import index as index_mod  # noqa: E402
from pedacito.config import Config, ConfigError  # noqa: E402
from pedacito.gui import core  # noqa: E402
from pedacito.tui import TuiSession  # noqa: E402

from test_tui import FakeClient, _cfg  # noqa: E402


# ------------------------------------------------------ suggest_shell_pattern ---

@pytest.mark.parametrize("cmd, pattern", [
    ("pytest tests/test_foo.py -q", "pytest *"),
    ("cargo test --lib", "cargo test *"),
    ("npm run lint", "npm run lint *"),
    ("pnpm run build --filter x", "pnpm run build *"),
    ("go test ./...", "go test *"),
    ("python -m black .", "python -m black *"),
    ("git status", "git status *"),
    ("make build", "make build *"),
    ("ruff check pedacito/", "ruff *"),
])
def test_suggest_shell_pattern_narrow_rules(cmd, pattern):
    assert security.suggest_shell_pattern(cmd) == pattern
    assert security.suggest_shell_pattern(cmd.split()) == pattern


@pytest.mark.parametrize("cmd", [
    "rm -rf /", "sudo apt install x", "bash -c 'x'", "sh", "cmd /c dir",
    "powershell -c x", "pip install x", "npm install left-pad",
    "python -m pip install x", "python -m venv .venv",
    "pytest ; rm -rf /", "pytest && rm x", "echo $(whoami)", "",
])
def test_suggest_shell_pattern_refuses_dangerous_commands(cmd):
    assert security.suggest_shell_pattern(cmd) == ""


# ----------------------------------------------------- matches_shell_allowlist ---

def test_matches_shell_allowlist_basic_patterns():
    m = security.matches_shell_allowlist
    assert m("pytest tests/test_foo.py -q", ["pytest *"])
    assert m("pytest", ["pytest *"])                 # trailing * covers 0 args
    assert m("cargo test --lib", ["cargo test *"])
    assert not m("cargo build", ["cargo test *"])
    assert m("npm run lint", ["npm run lint *"])
    assert not m("npm run build", ["npm run lint *"])
    assert m("git status", ["git status"])           # exact: whole argv
    assert not m("git status -s", ["git status"])
    assert m("pytest -q", ["py*"])                   # single-token glob
    assert m("PyTest -Q", ["pytest *"])              # case-insensitive
    assert m(["python", "-c", "print(42)"], ["python *"])
    assert m("/usr/bin/pytest -q", ["pytest *"])     # path-qualified exe
    assert not m("pytest -q", [])
    assert not m("", ["pytest *"])


@pytest.mark.parametrize("cmd", [
    "pytest ; rm -rf /",
    "pytest && rm x",
    "pytest || rm x",
    "pytest | tee out",
    "pytest `whoami`",
    "pytest $(whoami)",
    "pytest > out.txt",
    "pytest\nrm x",
])
def test_matches_shell_allowlist_never_matches_injection(cmd):
    assert not security.matches_shell_allowlist(cmd, ["pytest *"])
    assert not security.matches_shell_allowlist(cmd, ["*"])


def test_matches_shell_allowlist_never_matches_injection_argv_forms():
    m = security.matches_shell_allowlist
    assert not m(["pytest", ";", "rm", "-rf", "/"], ["pytest *"])
    assert not m(["pytest", "&&", "rm"], ["pytest *"])
    assert not m(["pytest", "|", "tee"], ["pytest *"])
    assert not m(["pytest", "$(whoami)"], ["pytest *"])
    # A pattern carrying metacharacters is itself skipped.
    assert not m("pytest -q", ["pytest ; *"])


@pytest.mark.parametrize("cmd, patterns", [
    ("pip install x", ["pip *"]),
    ("python -m pip install x", ["python *"]),
    ("npm install left-pad", ["npm *"]),
    ("uv pip install x", ["uv *"]),
    ("cargo add serde", ["cargo *"]),
    ("rm -rf /", ["rm *"]),
    ("bash -c x", ["bash *"]),
    ("sudo pytest", ["sudo *"]),
])
def test_installs_and_never_allow_executables_never_match(cmd, patterns):
    assert not security.matches_shell_allowlist(cmd, patterns)


# ------------------------------------------------------------- TOML handling ---

def test_shell_allowlist_by_project_toml_roundtrip_and_coercion():
    text = ('[universal]\n'
            'shell_allowlist_by_project = { "/a/b" = ["pytest *"], '
            '"/c/d" = ["cargo test *", "npm run lint *"] }\n')
    flat, _profiles = config_editor.validate_text(text)
    coerced = config_mod._coerce("shell_allowlist_by_project",
                                 flat["shell_allowlist_by_project"])
    assert coerced == {"/a/b": ("pytest *",),
                       "/c/d": ("cargo test *", "npm run lint *")}
    # The literal renderer quotes path keys so the output re-parses.
    literal = config_mod._toml_literal(coerced)
    again, _ = config_editor.validate_text(
        f"[universal]\nshell_allowlist_by_project = {literal}\n")
    assert again["shell_allowlist_by_project"] == {
        "/a/b": ["pytest *"], "/c/d": ["cargo test *", "npm run lint *"]}
    # Bad shapes are refused with a readable error.
    with pytest.raises(ConfigError):
        config_mod._coerce("shell_allowlist_by_project", ["pytest *"])
    with pytest.raises(ConfigError):
        config_mod._coerce("shell_allowlist_by_project", {"/a/b": "pytest *"})


def test_project_rules_are_trust_gated_like_shell_allowlist(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "USER_CONFIG",
                        tmp_path / "home" / "config.toml")
    proj = tmp_path / "proj"
    proj.mkdir()
    key = json.dumps(str(proj))
    (proj / ".pedacito.toml").write_text(
        f'shell_allowlist_by_project = {{ {key} = ["pytest *"] }}\n'
        'max_steps = 7\n', encoding="utf-8")
    cfg = config_mod.load(root=proj, use_env=False, require_hardware=False,
                          warn_budgets=False)
    assert cfg.shell_allowlist_by_project == {}    # untrusted: exec field off
    assert cfg.max_steps == 7                      # harmless settings apply
    config_mod.trust_project(proj / ".pedacito.toml")
    cfg = config_mod.load(root=proj, use_env=False, require_hardware=False,
                          warn_budgets=False)
    assert cfg.shell_allowlist_by_project == {str(proj): ("pytest *",)}


# --------------------------------------------------------------- persistence ---

def test_add_project_shell_pattern_creates_merges_and_is_idempotent(tmp_path):
    user = config_mod.USER_CONFIG                  # isolated by conftest
    root = tmp_path / "proj"
    root.mkdir()
    assert not user.exists()
    assert config_mod.add_project_shell_pattern(root, "pytest *") is True
    text = user.read_text(encoding="utf-8")
    flat, _ = config_editor.validate_text(text)
    norm = security.normalize_project_root(root)
    assert flat["shell_allowlist_by_project"] == {norm: ["pytest *"]}
    # Idempotent: the same rule again changes nothing.
    before = user.read_bytes()
    assert config_mod.add_project_shell_pattern(root, "pytest *") is False
    assert user.read_bytes() == before
    # A second rule for the same root, then one for another root: merged.
    assert config_mod.add_project_shell_pattern(root, "cargo test *") is True
    other = tmp_path / "other"
    other.mkdir()
    assert config_mod.add_project_shell_pattern(other, "npm run lint *") is True
    flat, _ = config_editor.validate_text(user.read_text(encoding="utf-8"))
    mapping = flat["shell_allowlist_by_project"]
    assert mapping[norm] == ["pytest *", "cargo test *"]
    assert mapping[security.normalize_project_root(other)] == ["npm run lint *"]
    # The previous version was backed up before the rewrite.
    assert user.with_name(user.name + ".bak").exists()


def test_add_project_shell_pattern_preserves_comments(tmp_path):
    user = config_mod.USER_CONFIG
    user.parent.mkdir(parents=True, exist_ok=True)
    user.write_text("# my precious comment\n[universal]\nmax_steps = 9\n",
                    encoding="utf-8")
    assert config_mod.add_project_shell_pattern(tmp_path, "pytest *") is True
    text = user.read_text(encoding="utf-8")
    assert "# my precious comment" in text
    assert "max_steps = 9" in text
    flat, _ = config_editor.validate_text(text)
    assert flat["max_steps"] == 9
    assert flat["shell_allowlist_by_project"] == {
        security.normalize_project_root(tmp_path): ["pytest *"]}
    bak = user.with_name(user.name + ".bak").read_text(encoding="utf-8")
    assert "shell_allowlist_by_project" not in bak


# ------------------------------------------------------------- tools: `run` ---

def _run_cfg(tmp_path, **overrides) -> Config:
    cfg = Config()
    cfg.allow_shell = True
    cfg.log_events = False
    cfg.shell_allowlist = ("git",)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_run_project_rule_skips_allowlist_and_broker(tmp_path):
    idx = index_mod.Index(tmp_path, [])
    cfg = _run_cfg(tmp_path, shell_allowlist_by_project={
        str(tmp_path): ("python *",)})
    # A hook that would deny everything: the rule must skip it entirely.
    calls = []
    security.set_permission_handler(
        lambda info: (calls.append(info), (False, "denied"))[1])
    try:
        out = tools.tool_run(idx, cfg, f'{sys.executable} -c "print(42)"')
    finally:
        security.set_permission_handler(None)
    assert "42" in out and "EXIT: 0" in out
    assert calls == []


def test_run_project_rule_is_scoped_and_deny_still_wins(tmp_path):
    idx = index_mod.Index(tmp_path, [])
    other = tmp_path / "elsewhere"
    other.mkdir()
    # A rule for ANOTHER project does not apply here.
    cfg = _run_cfg(tmp_path, shell_allowlist_by_project={
        str(other): ("python *",)})
    out = tools.tool_run(idx, cfg, f'{sys.executable} -c "print(42)"')
    assert "not in shell_allowlist" in out
    # The right project's rule applies, but shell_deny is a hard block.
    cfg = _run_cfg(tmp_path,
                   shell_allowlist_by_project={str(tmp_path): ("python *",)},
                   shell_deny=("python *",))
    out = tools.tool_run(idx, cfg, f'{sys.executable} -c "print(42)"')
    assert "shell_deny" in out


def test_run_install_still_asks_even_with_a_project_rule(tmp_path):
    idx = index_mod.Index(tmp_path, [])
    cfg = _run_cfg(tmp_path, shell_allowlist_by_project={
        str(tmp_path): ("pip *",)})
    calls = []

    def hook(info):
        calls.append(info)
        return False, security.DENIED_MESSAGE

    security.set_permission_handler(hook)
    try:
        out = tools.tool_run(idx, cfg, "pip install left-pad")
    finally:
        security.set_permission_handler(None)
    assert out == security.DENIED_MESSAGE
    assert len(calls) == 1
    assert calls[0]["category"] == security.CAT_PACKAGE_INSTALL
    # Installs get no persistable suggestion.
    assert calls[0]["suggested_pattern"] == ""


# ---------------------------------------------------------- tools: `run_test` ---

def test_run_test_disabled_project_rule_runs_without_broker(tmp_path):
    (tmp_path / "test_ok.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8")
    idx = index_mod.Index(tmp_path, [])
    cfg = _run_cfg(tmp_path, enable_run_test_in_gather=False)
    # Without a rule and without a broker: the historical refusal.
    out = tools.tool_run_test(idx, cfg, "test_ok.py")
    assert out.startswith("ERROR:") and "disabled" in out
    # The persisted "pytest *" rule is the human's standing answer.
    cfg.shell_allowlist_by_project = {str(tmp_path): ("pytest *",)}
    out = tools.tool_run_test(idx, cfg, "test_ok.py")
    assert "passed" in out


# ------------------------------------------------------------- GUI broker ---

@pytest.fixture
def scoped_broker(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "RECENT_FILE", tmp_path / "gui_recent.json")
    core.STATE.__init__()
    core.install_permission_broker()
    yield core.STATE
    core.uninstall_permission_broker()
    core.STATE.bus.close()


def _wait_pending(st, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        with st.lock:
            pp = dict(st.pending_permission or {})
        if pp:
            return pp
        time.sleep(0.02)
    raise AssertionError("no pending permission request appeared")


def test_gui_broker_allow_project_persists_and_updates_live_cfg(
        scoped_broker, tmp_path):
    st = scoped_broker
    st.cfg = Config()
    proj = tmp_path / "proj"
    proj.mkdir()
    result = {}

    def ask():
        result["ans"] = security.request_permission(
            security.CAT_SHELL, "Run command: pytest -q", detail="pytest -q",
            key="pytest -q", project_root=str(proj))

    t = threading.Thread(target=ask)
    t.start()
    try:
        pp = _wait_pending(st)
        # The modal gets everything it needs for the scoped button.
        assert pp["suggested_pattern"] == "pytest *"
        assert pp["project_root"] == security.normalize_project_root(proj)
        ok, _ = st.decide_permission(pp["id"], "allow_project")
        assert ok
        t.join(5)
    finally:
        if t.is_alive():
            st.deny_all_permissions("test teardown")
            t.join(5)
    assert result["ans"] == (True, "")
    norm = security.normalize_project_root(proj)
    assert "pytest *" in st.cfg.shell_allowlist_by_project[norm]
    # Persisted to the (isolated) user config, valid TOML.
    flat, _ = config_editor.validate_text(
        config_mod.USER_CONFIG.read_text(encoding="utf-8"))
    assert flat["shell_allowlist_by_project"][norm] == ["pytest *"]
    # The exact command is session-allowed: no second modal.
    assert security.request_permission(
        security.CAT_SHELL, "Run command: pytest -q", detail="pytest -q",
        key="pytest -q", project_root=str(proj)) == (True, "")


# ------------------------------------------------------------- TUI broker ---

def test_tui_prompt_bar_project_answer_persists_rule(tmp_path):
    cfg = _cfg()
    out = io.StringIO()
    session = TuiSession(tmp_path, cfg, client=FakeClient(), headless=True,
                         stdin=io.StringIO("p\n"), stdout=out)
    try:
        allowed, message = security.request_permission(
            security.CAT_SHELL, "run the test suite", detail="pytest -q",
            project_root=str(tmp_path))
        assert allowed is True
        assert "Allowed in this project" in message
        text = out.getvalue()
        assert "y = once / a = session / n = deny" in text
        assert "p = allow 'pytest *' in this project" in text
        norm = security.normalize_project_root(tmp_path)
        assert cfg.shell_allowlist_by_project[norm] == ("pytest *",)
        flat, _ = config_editor.validate_text(
            config_mod.USER_CONFIG.read_text(encoding="utf-8"))
        assert flat["shell_allowlist_by_project"][norm] == ["pytest *"]
    finally:
        session.close()
    assert security.permission_handler() is None
