"""B5: hosted profiles -- api_key_env, provider-table overrides, opt-in
calibration, and no secret on disk."""
from __future__ import annotations

import io
import json
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import cli, config_editor, discovery, providers
from pedacito import config as config_mod


def test_api_key_env_beats_the_plain_key(monkeypatch):
    from pedacito.llm import _api_key
    cfg = config_mod.Config()
    cfg.api_key = "plain"
    cfg.api_key_env = "MY_KEY"
    monkeypatch.setenv("MY_KEY", "from-env")
    assert _api_key(cfg) == "from-env"
    cfg.api_key_env = ""
    assert _api_key(cfg) == "plain"


def test_interactive_setup_writes_api_key_env_not_a_secret(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "discover_servers", lambda timeout=2.0: [])
    answers = {"Is this a hosted": "y",
               "API base URL": "https://api.anthropic.com/v1",
               "Env var holding the API key": "ANTHROPIC_KEY",
               "Model name": "claude-x"}

    def ask(prompt, default=""):
        for k, v in answers.items():
            if k in prompt:
                return v
        return default

    monkeypatch.setattr(config_mod, "_ask", ask)
    p = tmp_path / "config.toml"
    config_mod.interactive_setup(path=p, force=True, stdout=io.StringIO())
    raw = p.read_text(encoding="utf-8")
    assert 'api_key_env = "ANTHROPIC_KEY"' in raw
    prof = tomllib.loads(raw)["profiles"]["default"]
    assert prof["api_key_env"] == "ANTHROPIC_KEY"
    assert "api_key" not in prof          # no raw secret written


def test_profile_overrides_carry_the_provider_table():
    anth = providers.profile_overrides("https://api.anthropic.com/v1")
    assert anth["json_mode"] == "schema"
    assert anth["tool_mode"] == "grammar"
    assert anth["prompt_cache"] == "off"
    ds = providers.profile_overrides("https://api.deepseek.com/v1")
    assert ds["prompt_cache"] == "implicit"
    assert "json_mode" not in ds and "tool_mode" not in ds


def test_scan_hosted_never_records_the_secret(monkeypatch):
    monkeypatch.setattr(discovery, "list_models",
                        lambda url, headers=None: ["claude-3", "gpt-4"])
    monkeypatch.setenv("HOSTED_KEY", "sk-super-secret")

    class A:
        api_key_env = "HOSTED_KEY"
        match = "claude*"

    out = cli._scan_hosted(A(), ["https://api.fake.example/v1"])
    assert [e["id"] for e in out] == ["claude-3"]
    assert out[0]["api_key_env"] == "HOSTED_KEY"
    assert out[0]["tier"] == "hosted"
    assert "sk-super-secret" not in json.dumps(out)


def test_scan_hosted_requires_match(monkeypatch, capsys):
    monkeypatch.setenv("HOSTED_KEY", "k")

    class A:
        api_key_env = "HOSTED_KEY"
        match = ""

    assert cli._scan_hosted(A(), ["https://api.fake.example/v1"]) == []
    assert "--match" in capsys.readouterr().err


def test_upsert_writes_api_key_env_and_provider_settings():
    text = "[universal]\nvram_gb = 24\n"
    out = config_editor.upsert_profiles(text, [{
        "model": "claude-x", "base_url": "https://api.anthropic.com/v1",
        "server_label": "api.anthropic.com", "tier": "hosted",
        "api_key_env": "ANTHROPIC_KEY", "json_mode": "schema",
        "tool_mode": "grammar", "prompt_cache": "off"}],
        now="2026-11-02")
    assert 'api_key_env = "ANTHROPIC_KEY"' in out
    assert 'json_mode = "schema"' in out
    assert 'tool_mode = "grammar"' in out
    assert 'prompt_cache = "off"' in out
    # no card choice leaks in, and the result validates
    assert "best_of" not in out and "prompt_pack" not in out
    flat, profiles = config_editor.validate_text(out)
    body = profiles["claude-x"]
    assert body["api_key_env"] == "ANTHROPIC_KEY"
    assert body["prompt_cache"] == "off"
