"""Regression tests for BUG-006 (DeepSeek bug hunt, v0.3.1).

``pedacito new`` logged the command line into the target BEFORE the
dry-run/refusal decision, so a dry run that promised "Nothing was written"
still created ``.pedacito/log.txt`` -- and a refusal (exit 2, non-empty
directory) left state in a directory the user never accepted. The command
log belongs to the write path: it is recorded inside ``run_new`` only once
the run has committed to a skeleton (``--apply``/``--plan-only``/``--review``),
after the non-empty refusal guard.
"""

import sys

import pytest

from pedacito import cli, config, scaffold


class _StubClient:
    """Just enough client for the plan/skeleton path: no server contact."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0, "seconds": 0.0}

    def check(self, warn=True):
        return {}

    def chat_json(self, messages, schema, max_tokens=0, **kw):
        return {"files": []}


def _empty_plan(spec, client, cfg, root, *a, **k):
    return scaffold.Plan("Pond", "a tiny game", spec=spec)


@pytest.fixture()
def new_env(tmp_path, monkeypatch):
    """Isolated user config + stub model + stub plan, as the CLI sees them."""
    home = tmp_path / "home"
    (home / ".config" / "pedacito").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PEDACITO_VRAM_GB", "8")
    monkeypatch.setattr(config, "USER_CONFIG",
                        config._user_config_dir() / "config.toml")
    cfg_file = tmp_path / "user.toml"
    cfg_file.write_text(
        'vram_gb = 8\nrefresh_summaries = 0\nlog_sessions = false\n'
        'show_telemetry = false\n', encoding="utf-8")
    monkeypatch.setattr(cli, "LLMClient", _StubClient)
    monkeypatch.setattr(scaffold, "build_plan", _empty_plan)
    return cfg_file


def _cli_new(target, cfg_file, *extra):
    return cli.main(["new", str(target), "-t", "a tiny pond game",
                     "--config", str(cfg_file), "--model", "stub-model",
                     "--url", "http://127.0.0.1:9/v1", *extra])


def test_new_dry_run_creates_nothing_in_the_target(tmp_path, new_env, capsys):
    target = tmp_path / "fresh"
    capsys.readouterr()
    code = _cli_new(target, new_env)
    assert code == 0
    assert "Nothing was written" in capsys.readouterr().err
    # The dry run must not even create the directory, let alone state in it.
    assert not target.exists()


def test_new_refusal_leaves_only_the_preexisting_files(tmp_path, new_env,
                                                       capsys):
    target = tmp_path / "busy"
    target.mkdir()
    (target / "index.html").write_text("<h1>real project</h1>\n",
                                       encoding="utf-8")
    capsys.readouterr()
    code = _cli_new(target, new_env)
    assert code == 2
    assert "is not empty (index.html)" in capsys.readouterr().err
    assert sorted(p.name for p in target.iterdir()) == ["index.html"]


def test_new_plan_only_still_logs_the_command(tmp_path, new_env, capsys,
                                              monkeypatch):
    target = tmp_path / "built"
    capsys.readouterr()
    # The CLI logs the real process argv; under pytest that is pytest's own
    # command line, so pin a realistic one instead of relying on the test
    # file's name happening to contain "new".
    monkeypatch.setattr(sys, "argv",
                        ["pdc", "new", str(target), "--plan-only"])
    code = _cli_new(target, new_env, "--plan-only")
    assert code == 0
    log = (target / ".pedacito" / "log.txt").read_text(encoding="utf-8")
    assert "command: pdc new" in log and "--plan-only" in log


def test_run_new_logs_the_command_only_on_the_write_path(tmp_path,
                                                         monkeypatch):
    # The same guarantee the TUI and GUI rely on when they hand run_new a
    # log_command: a dry run and a refusal record nothing.
    cfg = config.load(require_hardware=False)
    monkeypatch.setattr(scaffold, "build_plan", _empty_plan)
    dry = tmp_path / "dry"
    code = scaffold.run_new(dry, cfg, _StubClient(cfg),
                            scaffold.Spec("a tiny pond game"),
                            verbose=False, log_command="command: test new")
    assert code == 0
    assert not dry.exists()

    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "main.py").write_text("print('real')\n", encoding="utf-8")
    code = scaffold.run_new(busy, cfg, _StubClient(cfg),
                            scaffold.Spec("a tiny pond game"),
                            verbose=False, log_command="command: test new")
    assert code == 2
    assert sorted(p.name for p in busy.iterdir()) == ["main.py"]
