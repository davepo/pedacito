"""C0: the hosted-provider table, the cache settings, and the doctor probe."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from pedacito import providers
from pedacito.config import Config, ConfigError
from pedacito.gui import settings as S


def test_provider_lookup_by_host():
    assert providers.provider_for(
        "https://api.deepseek.com/v1").cache_mode == "implicit"
    assert providers.provider_for(
        "https://api.openai.com/v1").cache_key_ok is True
    assert providers.provider_for(
        "https://dashscope.aliyuncs.com/compatible-mode/v1").cache_mode \
        == "explicit"
    anth = providers.provider_for("https://api.anthropic.com/v1")
    assert anth.cache_mode == "unsupported"
    assert anth.omit_temperature and anth.json_mode == "schema"
    assert not anth.native_tool_choice
    # an unknown host gets the safe implicit fallback (never forced markers)
    assert providers.provider_for("http://localhost:1234/v1").cache_mode \
        == "implicit"


def test_cache_mode_setting_beats_the_table():
    cfg = Config()
    cfg.base_url = "https://api.anthropic.com/v1"
    assert providers.cache_mode_for(cfg) == "unsupported"
    cfg.prompt_cache = "explicit"
    assert providers.cache_mode_for(cfg) == "explicit"
    cfg.prompt_cache = "off"
    assert providers.cache_mode_for(cfg) == "off"


def test_cache_settings_are_validated_and_default_to_auto():
    S.validate_text('prompt_cache = "explicit"\nprompt_cache_ttl = "1h"\n')
    for bad in ('prompt_cache = "always"', 'prompt_cache_ttl = "3h"'):
        with pytest.raises(ConfigError):
            S.validate_text(bad + "\n")
    cfg = Config()
    assert cfg.prompt_cache == "auto" and cfg.prompt_cache_ttl == "5m"
    item = next(i for i in S.schema() if i["name"] == "prompt_cache")
    assert item["choices"] == ["auto", "implicit", "explicit", "off"]


def test_doctor_probe_reports_observed_on_a_hit():
    from pedacito.llm import LLMClient

    class C(LLMClient):
        def __init__(self):
            LLMClient.__init__(self, Config())
            self.n = 0

        def chat(self, messages, max_tokens, **kw):
            self.n += 1
            if self.n >= 2:
                self.stats["cached_tokens"] = 100
            return "ok"

    assert C().doctor_prompt_cache(delay=0) == "observed"


def test_doctor_probe_reports_explicit_only_for_dashscope():
    from pedacito.llm import LLMClient

    class C(LLMClient):
        def __init__(self):
            cfg = Config()
            cfg.base_url = "https://dashscope.aliyuncs.com/v1"
            LLMClient.__init__(self, cfg)

        def chat(self, messages, max_tokens, **kw):
            return "ok"

    assert C().doctor_prompt_cache(delay=0) == "explicit-only"


def test_doctor_probe_reports_not_observed_for_a_local_server():
    from pedacito.llm import LLMClient

    class C(LLMClient):
        def __init__(self):
            cfg = Config()
            cfg.base_url = "http://localhost:11434/v1"
            LLMClient.__init__(self, cfg)

        def chat(self, messages, max_tokens, **kw):
            return "ok"

    assert C().doctor_prompt_cache(delay=0) == "not observed"


def test_cache_key_is_stable_and_differs_across_projects():
    """C3: 'auto' is a stable hash of project root + profile + model, so one
    project routes to a warm cache across runs while two projects differ."""
    def cfg(root, profile="x", model="m"):
        c = Config()
        c._root = root
        c.active_profile = profile
        c.model = model
        c.prompt_cache_key = "auto"
        return c

    a, b = cfg("/proj/a"), cfg("/proj/a")
    assert providers.cache_key(a) == providers.cache_key(b)
    assert providers.cache_key(a) != providers.cache_key(cfg("/proj/b"))
    assert providers.cache_key(a) != providers.cache_key(cfg("/proj/a", "y"))
    explicit = Config()
    explicit.prompt_cache_key = "my-key"
    assert providers.cache_key(explicit) == "my-key"
    assert providers.cache_key(Config()) == ""


def test_claude_quirks_apply_only_for_anthropic_and_users_can_override(
        tmp_path, monkeypatch):
    """C4: api.anthropic.com gets json_mode='schema' and tool_mode='grammar'
    from the provider table (never json_object, never forced tool_choice), and
    an explicit user value still wins."""
    from pedacito import config as config_mod
    monkeypatch.setattr(config_mod, "USER_CONFIG", tmp_path / "none.toml")
    proj = tmp_path / "p"
    proj.mkdir()

    def load(text):
        (tmp_path / "cfg.toml").write_text(text, encoding="utf-8")
        return config_mod.load(root=proj, explicit=tmp_path / "cfg.toml",
                               use_env=False, require_hardware=False,
                               warn_budgets=False)

    anth = load('base_url = "https://api.anthropic.com/v1"\n'
                'model = "claude-4-7"\n')
    assert anth.json_mode == "schema" and anth.tool_mode == "grammar"
    explicit = load('base_url = "https://api.anthropic.com/v1"\n'
                    'model = "claude-4-7"\njson_mode = "object"\n')
    assert explicit.json_mode == "object"          # the user overrode the quirk
    oai = load('base_url = "https://api.openai.com/v1"\nmodel = "gpt-4"\n')
    assert oai.json_mode == "object"               # generic hosted switch
