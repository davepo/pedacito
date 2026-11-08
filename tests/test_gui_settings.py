"""GUI config editing: schema, validation, and comment-preserving TOML patches."""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pedacito import config
from pedacito.config import ConfigError
from pedacito.gui import settings as S
from pedacito.gui import settings_api
from test_gui import gui  # noqa: F401  (the shared server fixture)
from test_models_scan import _lmstudio, _serve

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


# --- Model Discovery: version dots in model ids -> valid profile names ------
# (the scan + import endpoints run against the fake LM Studio server from
# test_models_scan, using test_gui's `gui` server fixture)

GIB = 1024 ** 3

DOT_MODELS = [
    {"id": "qwen3.8-27b-gsq-rco-gguf", "size_bytes": 17 * GIB,
     "context_length": 40960, "quantization": "q4_k_m",
     "loaded_in_memory": True},
    {"id": "zai-org/glm-4.7-flash:latest", "size_bytes": 4 * GIB,
     "context_length": 32768},
]


@pytest.fixture
def dot_server():
    srv = _serve(_lmstudio(DOT_MODELS))
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()
    srv.server_close()


def test_sanitize_profile_name_rules():
    f = settings_api._sanitize_profile_name
    assert f("qwen3.8") == "qwen3_8"                       # the version dot
    assert f("glm-4.7-flash-gguf") == "glm-4_7-flash"      # and the gguf tail
    assert f("QWEN3.8-27B-GSQ-RCO-GGUF") == "qwen3_8-27b-gsq-rco"
    assert f("Mistral 7B @ Q4") == "mistral_7b_q4"         # spaces and @ fold in
    assert f("--weird--") == "weird"                      # trimmed, collapsed
    assert f("x" * 40) == "x" * 32                        # capped for the UI
    # an org `path/` is a name a human typed wrong: kept invalid on purpose so
    # the import refuses it loudly instead of silently renaming
    assert "/" in f("zai-org/glm-4.7-flash:latest")
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}",
                        f("qwen3.8-27b-gsq-rco-gguf"))


def test_suggested_names_get_unique_on_sanitizer_collisions():
    taken: set = set()
    a = settings_api._suggest_profile_name("qwen3.8b", taken)
    b = settings_api._suggest_profile_name("qwen3_8b", taken)   # same slug after sanitizing
    assert a == "qwen3_8b" and b != a
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", b)


def test_scan_suggests_dot_free_names(gui, dot_server):
    status, j = gui.post("/api/models/scan", {"urls": [dot_server],
                                              "vram_gb": 24})
    assert status == 200 and j["ok"]
    rows = [m for m in j["models"] if m["server_url"] == dot_server]
    assert sorted(r["suggested_profile_name"] for r in rows) == \
        ["glm-4_7-flash", "qwen3_8-27b-gsq-rco"]
    for r in rows:
        assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}",
                            r["suggested_profile_name"]), r


def test_import_auto_fixes_dotted_names(gui, dot_server):
    """A profile name still carrying the model id's version dots imports
    cleanly: the backend sanitizes it instead of failing on `3.8`."""
    base = dot_server
    status, j = gui.post("/api/models/import", {"models": [
        {"profile_name": "qwen3.8-27b-gsq-rco-gguf", "base_url": base,
         "model": "qwen3.8-27b-gsq-rco-gguf", "set_default": True},
        {"profile_name": "glm-4.7-flash", "base_url": base,
         "model": "zai-org/glm-4.7-flash:latest"}]})
    assert status == 200 and j["ok"], j
    assert sorted(j["imported"]) == ["glm-4_7-flash", "qwen3_8-27b-gsq-rco"]
    assert j["default_profile"] == "qwen3_8-27b-gsq-rco"
    text = gui.cfg_path.read_text(encoding="utf-8")
    assert "[profiles.qwen3_8-27b-gsq-rco]" in text
    assert "[profiles.glm-4_7-flash]" in text
    assert 'model = "qwen3.8-27b-gsq-rco-gguf"' in text   # the id keeps its dots
    status, st = gui.get("/api/state")
    assert set(st["profiles"]) >= {"glm-4_7-flash", "qwen3_8-27b-gsq-rco"}
    # a truly invalid name is still refused, and nothing extra was written
    before = gui.cfg_path.read_text(encoding="utf-8")
    status, j = gui.post("/api/models/import", {"models": [
        {"profile_name": "bad/name", "base_url": base, "model": "x"}]})
    assert status == 400 and "detail" in j
    assert gui.cfg_path.read_text(encoding="utf-8") == before
