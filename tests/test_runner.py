"""Run-tab child environments: credentials are scrubbed, ordinary variables kept."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import runner  # noqa: E402

SECRETS = {
    "ANTHROPIC_API_KEY": "sk-ant-test-LEAK-ME-1",
    "OPENAI_API_KEY": "sk-test-LEAK-ME-2",
    "MY_APP_SECRET": "shh-LEAK-ME-3",
    "DB_PASSWORD": "hunter2-LEAK-ME-4",
    "GITHUB_TOKEN": "ghp_LEAK-ME-5",
    "PYPI_API_KEY": "pypi-LEAK-ME-6",
    "github_token": "ghp_LEAK-ME-7",
    "AWS_ACCESS_KEY_ID": "AKIALEAKME8",
    "CI_JOB_CREDENTIAL": "cred-LEAK-ME-9",
}
KEPT = {
    "PEDACITO_TEST_VISIBLE": "visible-value",
    "DATABASE_URL_HOST": "db.internal",
    "LANG_CHOICE": "en_US.UTF-8",
}


@pytest.fixture
def exported(monkeypatch):
    for name, value in {**SECRETS, **KEPT}.items():
        monkeypatch.setenv(name, value)
    return monkeypatch


@pytest.mark.parametrize("name", sorted(SECRETS))
def test_credential_names_are_classified_sensitive(name):
    assert runner._is_sensitive_env(name)


@pytest.mark.parametrize("name", ["PATH", "HOME", "LANG", "TERM",
                                  "PEDACITO_TEST_VISIBLE", "DATABASE_URL_HOST"])
def test_ordinary_names_are_not_sensitive(name):
    assert not runner._is_sensitive_env(name)


def test_child_env_drops_secrets_and_keeps_normal_variables(exported):
    env = runner._child_env()
    for name in SECRETS:
        assert name not in env, name
    for name, value in KEPT.items():
        assert env.get(name) == value, name
    assert "PATH" in env


def test_run_tab_child_process_never_sees_exported_credentials(exported, tmp_path):
    probe = ("import json, os; "
             "print(json.dumps(dict(os.environ)))")
    result = runner.run_project_command(
        tmp_path, [sys.executable, "-c", probe], timeout=30)
    assert result["ok"], result["stderr"]
    seen = json.loads(result["stdout"])
    for name, value in SECRETS.items():
        assert name not in seen, name
        assert value not in result["stdout"], name
    assert seen.get("PEDACITO_TEST_VISIBLE") == "visible-value"
