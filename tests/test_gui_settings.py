"""GUI config editing: schema, validation, and comment-preserving TOML patches."""
import pytest

from pedacito import config
from pedacito.config import ConfigError
from pedacito.gui import settings as S

BASE = '''\
# my config
default_profile = "default"

[universal]
vram_gb = 8          # my card
temperature = 0.2
shell_allowlist = [
    "pytest",   # tests
    "ruff",
]
note_free = 1

[agent]
allow_write = false

[profiles.default]
base_url = "http://localhost:1234/v1"
model = "a"   # keep me
# max_steps = 3

[profiles."my.qwen"]
base_url = "http://h:1/v1"
model = "q"
sampler = { top_p = 0.9, top_k = 40 }

[profiles.my.qwen.extra]
x = 1
'''


def _clean(text):
    # `note_free` is not a real setting; drop it so validation passes.
    return text.replace("note_free = 1\n", "").replace(
        "\n[profiles.my.qwen.extra]\nx = 1\n", "")


def test_schema_covers_settings_with_help_and_types():
    items = {i["name"]: i for i in S.schema()}
    assert items["vram_gb"]["type"] == "int"
    assert items["vram_gb"]["group"] == "Hardware"
    assert "VRAM" in items["vram_gb"]["help"]
    assert items["allow_shell"]["type"] == "bool"
    assert items["json_mode"]["choices"] == ["schema", "object"]
    assert items["shell_allowlist"]["type"] == "list"
    assert items["sampler"]["type"] == "table"
    assert items["api_key"]["secret"] is True
    assert "active_tier" not in items and "sources" not in items
    # every schema entry is a real, loadable setting
    assert all(i["name"] in config._FIELDS for i in items.values())


def test_template_validates_and_new_user_config_fills_connection():
    S.validate_text(config.TEMPLATE)
    text = S.new_user_config("http://h:9/v1", "my-model", 16)
    flat, profiles = S.validate_text(text)
    assert flat["vram_gb"] == 16
    assert profiles["default"]["model"] == "my-model"
    assert profiles["default"]["base_url"] == "http://h:9/v1"
    assert "# Pedacito configuration." in text      # comments survive


def test_validate_reports_unknown_and_bad_values():
    with pytest.raises(ConfigError, match="did you mean 'temperature'"):
        S.validate_text("temprature = 1\n")
    with pytest.raises(ConfigError, match="positive whole number"):
        S.validate_text("vram_gb = -3\n")
    with pytest.raises(ConfigError, match="invalid TOML"):
        S.validate_text("[universal\nvram_gb = 1\n")
    with pytest.raises(ConfigError, match="tier must be one of"):
        S.validate_text('[profiles.x]\ntier = "huge"\n')


def test_set_global_replaces_in_place_and_keeps_comments():
    out = S.set_values(_clean(BASE), {"vram_gb": 24, "temperature": 0.7})
    assert "vram_gb = 24          # my card" in out
    assert "temperature = 0.7" in out
    assert "# my config" in out
    S.validate_text(out)


def test_set_global_replaces_multiline_array():
    out = S.set_values(_clean(BASE), {"shell_allowlist": ["make"]})
    assert 'shell_allowlist = ["make"]' in out
    assert "pytest" not in out and "# tests" not in out
    S.validate_text(out)


def test_set_global_new_key_goes_to_universal_after_last_key():
    out = S.set_values(_clean(BASE), {"max_tokens_step": 900})
    lines = out.splitlines()
    i = lines.index("max_tokens_step = 900")
    assert lines.index("[universal]") < i < lines.index("[agent]")
    S.validate_text(out)


def test_global_key_that_lives_in_another_section_is_edited_there():
    out = S.set_values(_clean(BASE), {"allow_write": True})
    assert out.count("allow_write") == 1
    assert "allow_write = true" in out


def test_set_global_creates_universal_when_absent():
    out = S.set_values('default_profile = "default"\n', {"temperature": 0.5})
    assert out.endswith("[universal]\ntemperature = 0.5\n")
    S.validate_text(out)


def test_set_profile_values_stay_inside_the_profile():
    out = S.set_values(_clean(BASE), {"model": "b", "tier": "medium"},
                       profile="default")
    section = out.split("[profiles.default]")[1].split("[profiles")[0]
    assert 'model = "b"   # keep me' in section
    assert 'tier = "medium"' in section
    # the other profile is untouched
    assert 'model = "q"' in out
    S.validate_text(out)


def test_profile_insert_skips_commented_examples():
    out = S.set_values(config.TEMPLATE, {"temperature": 0.4},
                       profile="default")
    lines = out.splitlines()
    i = lines.index("temperature = 0.4")
    assert lines[i - 1].startswith("model")
    S.validate_text(out)


def test_set_profile_creates_missing_profile_and_handles_dotted_name():
    out = S.set_values(_clean(BASE), {"model": "z"}, profile="new one")
    assert '[profiles."new one"]' in out
    out = S.set_values(_clean(BASE), {"model": "z2"}, profile="my.qwen")
    assert out.count("[profiles.") == 2 and 'model = "z2"' in out
    S.validate_text(out)


def test_remove_keys():
    out = S.set_values(_clean(BASE), removes=["shell_allowlist", "vram_gb"])
    assert "shell_allowlist" not in out and "ruff" not in out
    assert "vram_gb" not in out
    out = S.set_values(_clean(BASE), removes=["model"], profile="default")
    assert 'model = "a"' not in out and 'model = "q"' in out


def test_inline_table_value_replaced():
    out = S.set_values(_clean(BASE), {"sampler": {"top_p": 0.5}},
                       profile="my.qwen")
    assert "sampler = {top_p = 0.5}" in out
    S.validate_text(out)


def test_multiline_string_content_is_not_mistaken_for_a_table():
    text = ('[universal]\nthinking_suffix = """\n[profiles.fake]\n"""\n'
            'vram_gb = 8\n')
    out = S.set_values(text, {"vram_gb": 12})
    assert "vram_gb = 12" in out and "[profiles.fake]" in out
    assert S.profile_names(text) == []


def test_crlf_files_keep_their_line_endings():
    text = _clean(BASE).replace("\n", "\r\n")
    out = S.set_values(text, {"vram_gb": 10})
    assert "\r\n" in out and "\n" not in out.replace("\r\n", "")
    S.validate_text(out)


def test_add_and_remove_profile_roundtrip():
    base = _clean(BASE)
    out = S.add_profile(base, "fast", {"base_url": "http://x/v1", "model": "m"})
    assert "fast" in S.profile_names(out)
    S.validate_text(out)
    with pytest.raises(ConfigError, match="already exists"):
        S.add_profile(out, "fast", {})
    with pytest.raises(ConfigError, match="letters, digits"):
        S.add_profile(out, "bad/name", {})
    gone = S.remove_profile(out, "fast")
    assert gone.rstrip() == base.rstrip()
    gone = S.remove_profile(base, "my.qwen")
    assert S.profile_names(gone) == ["default"]
    with pytest.raises(ConfigError, match="no profile named"):
        S.remove_profile(base, "nope")


def test_resolved_hides_secrets():
    cfg = config.Config()
    cfg.api_key = "sk-secret"
    cfg.sources["vram_gb"] = "user config (x)"
    res = S.resolved(cfg)
    assert res["api_key"]["value"] == ""
    assert res["vram_gb"]["source"] == "user config (x)"
