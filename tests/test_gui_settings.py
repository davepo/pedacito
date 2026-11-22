"""GUI config editing: schema, validation, and comment-preserving TOML patches."""
import re
import sys
import tomllib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pedacito import config
from pedacito.config import Config, ConfigError
from pedacito.gui import core
from pedacito.gui import settings as S
from pedacito.gui import settings_api
from test_gui import gui as _shared_gui  # noqa: F401  (pytest re-export)
from test_models_scan import _lmstudio, _serve


@pytest.fixture
def gui(_shared_gui):  # noqa: F811  (re-export the fixture imported above)
    return _shared_gui

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


# --- Profile renaming -----------------------------------------------------------

def _config(gui):
    gui.post("/api/settings/setup",
             {"base_url": "http://x:1/v1", "model": "m", "vram_gb": "8"})


def test_rename_profile_text_keeps_body_comments_and_default_reference():
    base = _clean(BASE)
    out = S.rename_profile(base, "default", "primary")
    assert "[profiles.primary]" in out and "[profiles.default]" not in out
    assert 'model = "a"   # keep me' in out          # body + inline comment
    assert "# my config" in out                      # file header comment
    assert 'default_profile = "primary"' in out      # the reference follows
    S.validate_text(out)
    assert S.rename_profile(base, "default", "default") == base
    with pytest.raises(ConfigError, match="no profile named"):
        S.rename_profile(base, "ghost", "x")
    with pytest.raises(ConfigError, match="already exists"):
        S.rename_profile(base, "default", "my.qwen")


def test_rename_profile_moves_quoted_names_and_subtables():
    text = ('[profiles."my.qwen"]\nmodel = "q"\n\n'
            '[profiles."my.qwen".sampler]\ntop_p = 0.9\n')
    out = S.rename_profile(text, "my.qwen", "qwen2")
    assert "[profiles.qwen2]" in out and "[profiles.qwen2.sampler]" in out
    assert "my.qwen" not in out
    _flat, profiles = S.validate_text(out)
    assert profiles["qwen2"]["sampler"] == {"top_p": 0.9}


def test_rename_profile_updates_roles_whichever_form_the_file_uses():
    section = ('default_profile = "fast"\n\n[roles]\nplanner = "fast"\n'
               'worker = "tiny"\n\n[profiles.fast]\nmodel = "m"\n\n'
               '[profiles.tiny]\nmodel = "t"\n')
    out = S.rename_profile(section, "fast", "quick")
    assert 'planner = "quick"' in out and 'worker = "tiny"' in out
    assert 'default_profile = "quick"' in out
    assert "[profiles.quick]" in out and "[profiles.fast]" not in out
    S.validate_text(out)
    inline = ('roles = { planner = "fast", worker = "tiny" }  # keep me\n\n'
              '[profiles.fast]\nmodel = "m"\n\n[profiles.tiny]\nmodel = "t"\n')
    out = S.rename_profile(inline, "fast", "quick")
    assert 'roles = { planner = "quick", worker = "tiny" }  # keep me' in out
    S.validate_text(out)


def test_profiles_rename_endpoint_updates_references(gui):
    _config(gui)
    assert gui.post("/api/settings/profile", {
        "file": "user", "action": "create", "name": "fast",
        "values": {"base_url": "http://y/v1", "model": "tiny"}})[0] == 200
    assert gui.post("/api/settings/set", {"file": "user", "changes": {
        "default_profile": "fast",
        "roles": {"planner": "fast", "worker": "default"}}})[0] == 200
    status, body = gui.post("/api/profiles/rename", {
        "file": "user", "old_name": "fast", "new_name": "quick"})
    assert status == 200 and body["ok"], body
    assert "quick" in body["user"]["profiles"]
    assert "fast" not in body["user"]["profiles"]
    assert body["renamed"] == {"from": "fast", "to": "quick"}
    assert body["profiles"] == ["default", "quick"]
    text = gui.cfg_path.read_text(encoding="utf-8")
    assert "[profiles.quick]" in text and "[profiles.fast]" not in text
    assert 'model = "tiny"' in text
    assert re.search(r'^default_profile = "quick"', text, re.M)
    assert 'planner = "quick"' in text and 'worker = "default"' in text
    # STATE sees the new name immediately: the header dropdown's list too
    status, st = gui.get("/api/state")
    assert st["profiles"] == ["default", "quick"]
    assert st["profile"] == "quick" and st["default_profile"] == "quick"


def test_profiles_rename_endpoint_validation(gui):
    _config(gui)
    before = gui.cfg_path.read_text(encoding="utf-8")
    assert gui.post("/api/profiles/rename",
                    {"old_name": "ghost", "new_name": "x"})[0] == 400
    assert gui.post("/api/profiles/rename",
                    {"old_name": "default", "new_name": "default"})[0] == 400
    assert gui.post("/api/profiles/rename",
                    {"old_name": "default", "new_name": "bad/name"})[0] == 400
    assert gui.post("/api/profiles/rename",
                    {"old_name": "default", "new_name": ""})[0] == 400
    assert gui.cfg_path.read_text(encoding="utf-8") == before
    # version dots are sanitized like the import endpoint
    status, body = gui.post("/api/profiles/rename", {
        "old_name": "default", "new_name": "qwen3.8"})
    assert status == 200 and "qwen3_8" in body["user"]["profiles"], body
    text = gui.cfg_path.read_text(encoding="utf-8")
    assert "[profiles.qwen3_8]" in text
    assert 'default_profile = "qwen3_8"' in text   # the reference followed it


# --- Universal settings are always concrete -------------------------------------

def test_settings_payload_resolves_concrete_universal_values(gui):
    """No project open and no user config yet: every universal setting still
    gets a concrete effective value (the Config() default) in /api/settings —
    there is nothing above the universal layer to inherit from."""
    status, p = gui.get("/api/settings")
    assert status == 200
    by = {i["name"]: i for i in p["schema"]}
    assert set(by) == set(p["resolved"])           # nothing missing
    for name in by:
        assert "value" in p["resolved"][name] and "source" in p["resolved"][name]
    assert p["resolved"]["allow_shell"]["value"] is False
    assert p["resolved"]["auto_fit"]["value"] is True
    assert p["resolved"]["temperature"]["value"] == 0.2
    assert p["resolved"]["temperature_answer"]["value"] == 0.3
    assert p["resolved"]["json_mode"]["value"] == "schema"
    assert p["resolved"]["context_tokens"]["value"] == \
        by["context_tokens"]["default"] == 8192
    # written values still win over the defaults
    _config(gui)
    status, p = gui.get("/api/settings")
    assert p["user"]["settings"]["vram_gb"] == 8
    assert p["resolved"]["api_key"]["value"] == ""    # secrets stay masked


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


# --- gui_theme persistence, Auto-Tune, profile reasoning overrides --------------

GUITHEMES = ["slate", "light", "daylight", "nord", "midnight", "jazz_solo",
             "trapper_keeper", "memphis_arcade", "windbreaker"]


def test_gui_theme_is_an_enum_setting():
    items = {i["name"]: i for i in S.schema()}
    it = items["gui_theme"]
    assert it["type"] == "str" and it["group"] == "Graphical interface"
    assert it["choices"] == GUITHEMES and it["default"] == "slate"
    flat, _ = S.validate_text('gui_theme = "nord"\n')
    assert flat["gui_theme"] == "nord"
    with pytest.raises(ConfigError, match="gui_theme must be one of"):
        S.validate_text('gui_theme = "vaporwave"\n')


def test_reasoning_override_group_keys_exist_in_schema():
    """Every key the Profiles editor renders in its Reasoning & Execution
    Overrides box is a real setting with the control type the frontend
    expects (bool selects, enum dropdowns, numeric boxes)."""
    by = {i["name"]: i for i in S.schema()}
    for key in ("disable_thinking", "thinking_suffix", "step_reasoning_first",
                "incremental_edit_loop", "edit_plan_brief",
                "dynamic_tool_pruning", "prompt_pack", "answer_mode",
                "edit_style", "tool_mode", "best_of", "fuzzy_match",
                "temperature", "temperature_answer", "max_output_tokens",
                "context_tokens"):
        assert key in by, key
    assert by["disable_thinking"]["type"] == "bool"
    assert by["step_reasoning_first"]["type"] == "bool"   # None = card decides
    assert by["prompt_pack"]["choices"] == ["", "tiny", "small", "full"]
    assert by["tool_mode"]["choices"] == ["grammar", "native"]
    assert by["edit_style"]["choices"] == ["quote", "anchor"]


def test_theme_endpoint_persists_in_user_config(gui):
    _config(gui)
    status, body = gui.post("/api/settings/theme", {"gui_theme": "nord"})
    assert status == 200 and body["ok"] and body["persisted"] is True
    assert "reloaded" not in body         # cosmetic: never re-opens the project
    assert 'gui_theme = "nord"' in gui.cfg_path.read_text(encoding="utf-8")
    _status, st = gui.get("/api/state")
    assert st["gui_theme"] == "nord"
    # a second pick replaces the key in place instead of duplicating it
    assert gui.post("/api/settings/theme", {"gui_theme": "midnight"})[0] == 200
    text = gui.cfg_path.read_text(encoding="utf-8")
    assert _assigned(text, "gui_theme") == ['gui_theme = "midnight"']
    for bad in ("vaporwave", ""):
        status, j = gui.post("/api/settings/theme", {"gui_theme": bad})
        assert status == 400 and not j["ok"], bad
    _status, st = gui.get("/api/state")
    assert st["gui_theme"] == "midnight"  # the rejected pick changed nothing


def test_theme_endpoint_creates_a_user_config_when_none_exists(gui):
    """A project-less GUI still remembers the pick: the first theme change
    creates USER_CONFIG from the starter template and persists gui_theme."""
    status, body = gui.post("/api/settings/theme", {"gui_theme": "jazz_solo"})
    assert status == 200 and body["ok"] and body["persisted"] is True
    assert body["created"] is True
    assert gui.cfg_path.exists()
    assert 'gui_theme = "jazz_solo"' in gui.cfg_path.read_text(encoding="utf-8")
    _status, st = gui.get("/api/state")
    assert st["gui_theme"] == "jazz_solo"
    # a second pick replaces it in place, no duplicate key
    assert gui.post("/api/settings/theme",
                    {"gui_theme": "midnight"})[0] == 200
    assert _assigned(gui.cfg_path.read_text(encoding="utf-8"),
                     "gui_theme") == ['gui_theme = "midnight"']


def _assigned(text, key):
    """The real (non-commented) assignment lines for a key in config text.
    The setup file is the documented template: it mentions most keys in
    comments as well, so a plain count would lie."""
    return [ln.strip() for ln in text.splitlines()
            if re.match(rf"\s*{key}\s*=", ln) and not ln.strip().startswith("#")]


def _tune(**over):
    answers = {"hardware": "lt8", "model": "small", "codebase": "medium",
               "priority": "balanced"}
    answers.update(over)
    return answers


def test_autotune_endpoint_preview_is_deterministic(gui):
    _config(gui)
    before = gui.cfg_path.read_text(encoding="utf-8")
    status, body = gui.post("/api/settings/autotune", _tune())
    assert status == 200 and body["ok"] and body["applied"] is False
    got = {c["key"]: c["value"] for c in body["changes"]}
    assert got["max_read_tokens"] == 800
    assert got["max_total_model_calls"] == 25
    assert got["max_no_progress_steps"] == 3
    assert got["incremental_edit_loop"] is True
    assert got["step_reasoning_first"] is True
    assert got["edit_plan_brief"] is True
    assert got["dynamic_tool_pruning"] is True
    assert got["best_of"] == 1
    assert all(c["reason"] for c in body["changes"])       # every row explains
    # the same answers produce the identical recommendation
    _status, again = gui.post("/api/settings/autotune", _tune())
    assert again["changes"] == body["changes"]
    # the 16-24 GB balanced tier
    _status, big = gui.post("/api/settings/autotune", _tune(hardware="16to24"))
    gotb = {c["key"]: c["value"] for c in big["changes"]}
    assert (gotb["max_read_tokens"], gotb["max_total_model_calls"],
            gotb["max_no_progress_steps"]) == (1400, 35, 4)
    # thorough on a small local model buys a second answer candidate
    _status, th = gui.post("/api/settings/autotune", _tune(priority="thorough"))
    assert {c["key"]: c["value"] for c in th["changes"]}["best_of"] == 2
    # strong models keep every tool visible and skip the edit microloops
    _status, strong = gui.post("/api/settings/autotune",
                               _tune(hardware="hosted", model="large"))
    got2 = {c["key"]: c["value"] for c in strong["changes"]}
    assert "incremental_edit_loop" not in got2
    assert got2["dynamic_tool_pruning"] is False
    # any missing or unknown answer is refused
    status, j = gui.post("/api/settings/autotune", _tune(hardware="solarpunk"))
    assert status == 400 and "hardware must be one of" in j["detail"]
    assert gui.post("/api/settings/autotune", {"model": "small"})[0] == 400
    assert gui.cfg_path.read_text(encoding="utf-8") == before   # no writes


def test_autotune_apply_writes_universal_settings(gui):
    _config(gui)
    status, body = gui.post("/api/settings/autotune", _tune(apply=True))
    assert status == 200 and body["ok"] and body["applied"] is True
    text = gui.cfg_path.read_text(encoding="utf-8")
    assert "max_read_tokens = 800" in text
    assert "max_total_model_calls = 25" in text
    assert "incremental_edit_loop = true" in text
    assert "[profiles.default]" in text   # the profiles themselves survive
    S.validate_text(text)
    # the live config sees the new budgets at once
    _status, p = gui.get("/api/settings")
    assert p["resolved"]["max_read_tokens"]["value"] == 800
    assert p["resolved"]["max_no_progress_steps"]["value"] == 3
    # applying a second tier replaces the keys instead of duplicating them
    assert gui.post("/api/settings/autotune",
                    _tune(hardware="16to24", apply=True))[0] == 200
    text = gui.cfg_path.read_text(encoding="utf-8")
    assert _assigned(text, "max_read_tokens") == ["max_read_tokens = 1400"]


def test_effort_endpoint_validates_live_value_and_persists(gui, monkeypatch):
    """The Options-bar Effort selector: valid levels reach the live config,
    carry the GUI source label and persist to the user file; bad levels are
    refused without touching either. `default` must send no transport field."""
    _config(gui)
    with core.STATE.lock:
        core.STATE.cfg = Config(reasoning_effort="medium")
    status, j = gui.post("/api/effort", {"level": "extreme"})
    assert status == 400 and j["ok"] is False
    assert "default, low, medium or high" in j["detail"]

    status, j = gui.post("/api/effort", {"level": "high"})
    assert status == 200 and j["ok"] is True
    assert j["reasoning_effort"] == "high"
    assert "model_class" in j
    with core.STATE.lock:
        cfg = core.STATE.cfg
        assert cfg.reasoning_effort == "high"
        assert cfg.sources["reasoning_effort"] == "GUI Effort selector"
    text = gui.cfg_path.read_text(encoding="utf-8")
    assert _assigned(text, "reasoning_effort") == ['reasoning_effort = "high"']

    # The endpoint accepts the setting's own key as well as the UI's `level`.
    status, j = gui.post("/api/effort", {"reasoning_effort": "low"})
    assert status == 200 and j["reasoning_effort"] == "low"
    with core.STATE.lock:
        assert core.STATE.cfg.reasoning_effort == "low"

    status, j = gui.post("/api/effort", {"level": "default"})
    assert status == 200 and j["reasoning_effort"] == "default"
    assert _assigned(gui.cfg_path.read_text(encoding="utf-8"),
                     "reasoning_effort") == ['reasoning_effort = "default"']

    # At the transport, "default" omits reasoning_effort entirely (the
    # local-moe client would otherwise forward low/medium/high).
    payloads = []

    class Reply:
        status_code = 200
        text = "{}"

        def json(self):
            return {"choices": [{"message": {"content": "ok"},
                                 "finish_reason": "stop"}]}

    def fake_post(url, headers=None, timeout=None, json=None, **kw):
        payloads.append(dict(json or {}))
        return Reply()

    monkeypatch.setattr("pedacito.llm.requests.post", fake_post)
    from pedacito.llm import LLMClient
    transport = Config(base_url="http://127.0.0.1:8000/v1",
                       model="deepseek-v4.1-flash", provider="avalon",
                       reasoning_effort="default")
    LLMClient(transport).chat([{"role": "user", "content": "hi"}],
                              max_tokens=16)
    assert "reasoning_effort" not in payloads[-1]


def test_profile_reasoning_overrides_persist(gui):
    """What the Profiles editor's override box writes: thinking-mode and
    edit-loop keys land inside [profiles.*], not the universal layer."""
    _config(gui)
    status, body = gui.post("/api/settings/set", {
        "file": "user", "profile": "default",
        "changes": {"disable_thinking": True, "step_reasoning_first": False,
                    "prompt_pack": "tiny", "best_of": 2}})
    assert status == 200 and body["ok"], body
    text = gui.cfg_path.read_text(encoding="utf-8")
    seg = text[text.index("[profiles.default]"):]
    assert "disable_thinking = true" in seg
    assert "step_reasoning_first = false" in seg
    assert 'prompt_pack = "tiny"' in seg
    assert "best_of = 2" in seg
    _status, p = gui.get("/api/settings")
    prof = p["user"]["profiles"]["default"]
    assert prof["disable_thinking"] is True and prof["prompt_pack"] == "tiny"
    # the universal layer kept its own value (the profile override wins)
    assert p["user"]["settings"]["disable_thinking"] is False
    # an invalid enum override is refused with the schema's own error
    status, j = gui.post("/api/settings/set", {
        "file": "user", "profile": "default",
        "changes": {"prompt_pack": "giant"}})
    assert status == 400 and "prompt_pack must be one of" in j["detail"]


def test_post_edit_verify_round_trips_auto_verify(gui):
    """The Options-bar checkbox writes the COMMAND string, never a bool, and
    /api/state reports it back so the checkbox reflects reality."""
    _config(gui)
    items = {i["name"]: i for i in S.schema()}
    assert items["auto_verify"]["type"] == "str"

    status, body = gui.post("/api/settings/set", {
        "file": "user", "changes": {"auto_verify": "npm test"}})
    assert status == 200 and body["ok"], body
    text = gui.cfg_path.read_text(encoding="utf-8")
    assert 'auto_verify = "npm test"' in text
    _status, state = gui.get("/api/state")
    assert state["auto_verify"] == "npm test"
    assert state["detected_test_command"] == ""   # scratchpad: no project

    status, body = gui.post("/api/settings/set", {
        "file": "user", "changes": {"auto_verify": ""}})
    assert status == 200 and body["ok"], body
    assert 'auto_verify = ""' in gui.cfg_path.read_text(encoding="utf-8")
    assert gui.get("/api/state")[1]["auto_verify"] == ""


def test_config_without_auto_verify_is_untouched(gui):
    """Old configs gain no key from a state poll or an unrelated edit."""
    gui.cfg_path.parent.mkdir(parents=True, exist_ok=True)
    gui.cfg_path.write_text('[universal]\nmodel = "m"\n', encoding="utf-8")
    before = gui.cfg_path.read_bytes()
    assert gui.get("/api/state")[0] == 200
    assert gui.cfg_path.read_bytes() == before
    status, body = gui.post("/api/settings/set", {
        "file": "user", "changes": {"temperature": 0.3}})
    assert status == 200 and body["ok"], body
    assert "auto_verify" not in gui.cfg_path.read_text(encoding="utf-8")


def test_state_exposes_the_detected_test_command(gui):
    """The checkbox tooltip/enable-assist reads the detected runner from the
    same state payload as auto_verify."""
    root = gui.tmp / "detect-proj"
    root.mkdir()
    (root / "package.json").write_text(
        '{"scripts": {"test": "node --test"}}\n', encoding="utf-8")
    with core.STATE.lock:
        core.STATE.root = root
        core.STATE.cfg = config.Config()
        core.STATE._detect_test = ("", "", "", 0.0)
    try:
        state = gui.get("/api/state")[1]
        assert state["detected_test_command"] == "npm test"
        assert state["auto_verify"] == ""
    finally:
        with core.STATE.lock:
            core.STATE.root = None
            core.STATE.cfg = None
            core.STATE._detect_test = ("", "", "", 0.0)


# --- Pre-configured hosted provider profiles -----------------------------------
# `/api/providers/add` turns a catalog preset into a complete profile; the user
# only supplies an API key or the env var that holds one.

def test_provider_presets_endpoint_lists_the_major_clouds(gui):
    status, j = gui.get("/api/providers/presets")
    assert status == 200 and j["ok"] is True      # envelope the UI accepts
    assert [p["id"] for p in j["presets"]] == [
        "anthropic", "openai", "deepseek", "qwen", "gemini", "openrouter",
        "strata"]
    for p in j["presets"]:
        # the keys static/app.js's PDC.FALLBACK_PROVIDER_CATALOG mirrors: the
        # UI seeds from the fallback and merges this reply, so a key that
        # appears on only one side silently breaks one of the two
        assert {"id", "label", "provider", "base_url", "api_key_env",
                "suggest", "context_tokens", "models"} <= set(p)
        assert p["api_key_env"].endswith("_API_KEY")          # env fallback named
        assert p["context_tokens"] >= 128000                  # generous windows
        assert "api_key" not in p                             # catalog never holds secrets
        if p["id"] == "strata":
            # The local-MoE runtime is a first-class preset, not a cloud: its
            # endpoint is local, and its live /v1/models answer is the catalog.
            assert p["label"] == "Strata / Avalon (Local MoE)"
            assert p["provider"] == "strata"
            assert p["base_url"] == "http://127.0.0.1:8080/v1"
            assert p["suggest"] == "strata"
            assert p["context_tokens"] == 131072
            assert p["models"] == []
        else:
            assert p["base_url"].startswith("https://")       # real hosted endpoints
            assert p["models"]                            # curated ids, none empty
    anthropic = [p for p in j["presets"] if p["id"] == "anthropic"][0]
    assert anthropic["base_url"] == "https://api.anthropic.com/v1"
    assert anthropic["models"][0] == "claude-sonnet-4-5"


def test_add_preset_creates_a_ready_hosted_profile(gui):
    status, j = gui.post("/api/providers/add", {
        "preset_id": "anthropic", "model": "claude-sonnet-4-5",
        "api_key": "sk-proxy-session-123", "set_default": True})
    assert status == 200 and j["ok"] and j["added"] == "claude_sonnet"
    assert j["uses_env"] is False and j["preset"] == "anthropic"
    body = tomllib.loads(gui.cfg_path.read_text(encoding="utf-8"))
    assert body["default_profile"] == "claude_sonnet"          # set_default honored
    prof = body["profiles"]["claude_sonnet"]
    assert prof["base_url"] == "https://api.anthropic.com/v1"
    assert prof["model"] == "claude-sonnet-4-5"
    assert prof["api_key"] == "sk-proxy-session-123"
    assert "api_key_env" not in prof                           # a pasted key clears the env
    assert prof["tier"] == "hosted"
    assert prof["context_tokens"] == 200000
    assert prof["max_read_tokens"] == 3000                     # cloud gather budgets
    # Anthropic's native-surface quirks come from the provider table
    assert prof["json_mode"] == "schema"
    assert prof["prompt_cache"] == "explicit"
    assert "tool_mode" not in prof                # forced native tool_choice is accepted
    # STATE reloaded: the profile is selectable right away, no restart
    _status, st = gui.get("/api/state")
    assert st["profile"] == "claude_sonnet"
    assert st["model"] == "claude-sonnet-4-5"
    assert "claude_sonnet" in st["profiles"]
    # the settings payload came back too (the modal refreshes from it)
    assert "claude_sonnet" in j["user"]["profiles"]


def test_add_preset_uses_env_var_and_suggests_a_friendly_name(gui):
    # no key at all: the profile points at the preset's env var instead
    status, j = gui.post("/api/providers/add", {
        "preset_id": "openrouter", "model": "qwen/qwen3-coder"})
    assert status == 200 and j["ok"] and j["uses_env"] is True
    assert j["added"] == "openrouter_qwen3_coder"   # org path and vendor words drop
    prof = tomllib.loads(gui.cfg_path.read_text(encoding="utf-8")
                         )["profiles"]["openrouter_qwen3_coder"]
    assert prof["api_key_env"] == "OPENROUTER_API_KEY" and "api_key" not in prof
    assert prof["base_url"] == "https://openrouter.ai/api/v1"
    assert prof["model"] == "qwen/qwen3-coder"
    assert prof["json_mode"] == "object"          # generic hosted default stands
    # an explicit name wins over the suggestion; a non-default add keeps the
    # previously chosen profile
    _config(gui)
    status, j = gui.post("/api/providers/add", {
        "preset_id": "openai", "profile_name": "gpt-main",
        "model": "gpt-5.4", "api_key": "sk-openai", "set_default": True})
    assert status == 200 and j["added"] == "gpt-main"
    status, j = gui.post("/api/providers/add", {"preset_id": "deepseek", "api_key": "d"})
    assert status == 200 and j["added"] == "deepseek_chat"
    body = tomllib.loads(gui.cfg_path.read_text(encoding="utf-8"))
    assert body["default_profile"] == "gpt-main"              # untouched by the add
    assert body["profiles"]["deepseek_chat"]["model"] == "deepseek-chat"
    assert body["profiles"]["deepseek_chat"]["api_key"] == "d"
    _status, st = gui.get("/api/state")
    assert st["profile"] == "gpt-main"


def test_add_preset_refuses_bad_requests_without_writing(gui):
    status, j = gui.post("/api/providers/add", {"preset_id": "bogus"})
    assert status == 400 and j["ok"] is False
    assert "unknown provider preset" in j["detail"] and "anthropic" in j["detail"]
    status, j = gui.post("/api/providers/add", {
        "preset_id": "openai", "profile_name": "!!!"})
    assert status == 400 and j["ok"] is False
    assert "profile name" in j["detail"]
    status, j = gui.post("/api/providers/add", {"preset_id": "openai"})
    assert status == 200 and j["added"] == "openai_gpt_5_4"   # first: passes
    status, j = gui.post("/api/providers/add", {"preset_id": "openai"})
    assert status == 400 and "already exists" in j["detail"]  # duplicate suggestion
    assert not gui.cfg_path.read_text(encoding="utf-8").count(
        "[profiles.openai_gpt_5_4]") > 1                      # the refusal wrote nothing


def test_js_fallback_catalog_mirrors_the_python_one():
    """The hosted-provider dialog renders from PDC.FALLBACK_PROVIDER_CATALOG
    until (and unless) GET /api/providers/presets answers, so the two catalogs
    must stay a byte-for-byte mirror: same ids in order, same labels/URLs/env
    names, same curated model lists (the dialog's curated fallback when a
    live probe cannot reach the provider)."""
    js = (Path(__file__).resolve().parents[1] / "pedacito" / "gui" / "static"
          / "app.js").read_text(encoding="utf-8")
    start = js.index("PDC.FALLBACK_PROVIDER_CATALOG = [")
    block = js[start:js.index("\n];", start)]
    flat = re.sub(r"\s+", " ", block)
    ids = re.findall(r"id: '([a-z_]+)'", block)
    assert ids == [p["id"] for p in settings_api.CLOUD_PROVIDERS]
    for p in settings_api.CLOUD_PROVIDERS:
        quoted = ", ".join("'%s'" % m for m in p["models"])
        for frag in ("id: '%s'" % p["id"], "label: '%s'" % p["label"],
                    "provider: '%s'" % p["provider"],
                    "base_url: '%s'" % p["base_url"],
                    "api_key_env: '%s'" % p["api_key_env"],
                    "suggest: '%s'" % p["suggest"],
                    "context_tokens: %d" % p["context_tokens"],
                    "models: [%s]" % quoted):
            assert frag in flat, (p["id"], frag)


def _settings_js():
    return (Path(__file__).resolve().parents[1] / "pedacito" / "gui" / "static"
            / "settings.js").read_text(encoding="utf-8")


def test_profiles_pairing_panel_starts_collapsed_with_a_status_pill():
    js = _settings_js()
    assert "var PAIR = { open: false," in js
    scout = js[js.index("function scoutSection(f)"):
               js.index("function profilesTab()")]
    # The disclosure header is always there; the form itself only exists
    # while open, so the profile editor does not pay its vertical space.
    assert "'aria-expanded': PAIR.open ? 'true' : 'false'" in scout
    assert "Dual-GPU / Scout + Architect Pairing" in scout
    assert "Paired: " in scout and "Not configured" in scout
    assert "if (!PAIR.open) return box;" in scout
    assert "class: 'pair-body'" in scout


def test_hosted_dialog_shows_context_price_and_a_catalog_refresh():
    js = _settings_js()
    rows = js[js.index("function hcRenderRows()"):js.index("function hcRender()")]
    imp = js[js.index("function hcImport()"):js.index("function hcRenderRows()")]
    assert "fmtCtx(ctx)" in rows and "fmtPricePair(m)" in rows
    assert "price_input_per_mtok" in js and "price_output_per_mtok" in js
    # the selected model's detected window travels with the import
    assert "context_tokens: m.context_tokens || 0" in imp
    assert "'/api/providers/refresh'" in js
    assert "Refresh catalog" in js
    assert "Provider catalog refreshed." in js


def test_strata_preset_needs_an_explicit_model_for_a_one_click_add(gui):
    # The local-MoE preset has no curated ids (its live /v1/models answer is
    # the catalog), so an add without a model is refused with guidance
    # instead of raising IndexError.
    status, j = gui.post("/api/providers/add", {"preset_id": "strata"})
    assert status == 400 and j["ok"] is False
    assert "explicit model" in j["detail"]


def test_import_hosted_models_keeps_the_context_window_picked_in_the_dialog(gui):
    status, j = gui.post("/api/providers/import", {
        "preset_id": "openai", "base_url": "https://api.openai.com/v1",
        "api_key": "sk-openai",
        "models": [{"model": "gpt-5.4", "profile_name": "gpt_main",
                    "context_tokens": 123456}]})
    assert status == 200 and j["imported"] == ["gpt_main"]
    prof = tomllib.loads(gui.cfg_path.read_text(encoding="utf-8")
                         )["profiles"]["gpt_main"]
    assert prof["context_tokens"] == 123456



# --- hosted-provider dialog: live probe + model-named import ------------------
# The dialog never trusts a key it has not used: step A verifies it against
# the provider's own /models, step B imports the LIVE model list and names
# every profile after the model itself (`gpt-5_4`, `deepseek_chat`...), never
# after the vendor. `/api/providers/probe` writes nothing; the writes are
# `/api/providers/import`.

def _hosted_server():
    """A fake hosted API: requests mentioning 'sk-good' pass, the rest 401.

    Answers the OpenAI/Anthropic shape (`{"data": [...]}` from GET /models),
    includes the non-chat noise a real provider lists, and records headers so
    the tests can assert how the probe authenticated.
    """
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler

    from pedacito.local_server import LocalHTTPServer

    seen = []

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            creds = (self.headers.get("Authorization") or "") + "|" + (
                self.headers.get("x-api-key") or "")
            seen.append({"path": self.path,
                         "authorization": self.headers.get("Authorization"),
                         "x_api_key": self.headers.get("x-api-key"),
                         "anthropic_version": self.headers.get(
                             "anthropic-version")})
            body = _json.dumps({"data": [
                {"id": "openai/gpt-5.4"},          # org path -> short name
                {"id": "gpt-4o-mini"},
                {"id": "gpt-4o-audio-preview"},    # not chat models: dropped
                {"id": "text-embedding-3-small"},
                {"id": "dall-e-3"},
                {"id": "whisper-1"},
            ]}).encode()
            code = 200 if "sk-good" in creds else 401
            self.send_response(code)
            if code == 200:
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if code == 200:
                self.wfile.write(body)

        def log_message(self, *a):  # keep pytest output clean
            pass

    srv = LocalHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


def _dead_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_probe_verifies_the_key_live_and_names_rows_after_models(gui):
    srv, seen = _hosted_server()
    try:
        base = "http://127.0.0.1:%d/v1" % srv.server_address[1]
        status, j = gui.post("/api/providers/probe", {
            "preset_id": "openai", "base_url": base, "api_key": "sk-good"})
        assert status == 200 and j["ok"] and j["verified"]
        assert j["base_url"] == base and j["preset"] == "openai"
        rows = {m["id"]: m for m in j["models"]}
        assert set(rows) == {"openai/gpt-5.4", "gpt-4o-mini"}, \
            "chat models only — embeddings/audio never lead a picker"
        # one row per LIVE model id; the profile name derives from the model
        assert rows["openai/gpt-5.4"]["suggested_profile_name"] == "gpt-5_4"
        assert rows["gpt-4o-mini"]["suggested_profile_name"] == "gpt-4o-mini"
        assert all(not m["already_imported"] for m in j["models"])
        assert seen[0]["path"] == "/v1/models"
        assert seen[0]["authorization"] == "Bearer sk-good"
    finally:
        srv.shutdown()


def test_probe_reads_the_key_from_the_named_env_var(gui, monkeypatch):
    srv, seen = _hosted_server()
    monkeypatch.setenv("PROBE_TEST_KEY", "sk-good")
    try:
        base = "http://127.0.0.1:%d/v1" % srv.server_address[1]
        status, j = gui.post("/api/providers/probe", {
            "preset_id": "openai", "base_url": base,
            "api_key_env": "PROBE_TEST_KEY"})
        assert status == 200 and j["ok"]
        assert seen[0]["authorization"] == "Bearer sk-good"
        # a pasted key beats the env var — the dialog sends both, and the
        # one the user is looking at is the one that gets tested
        del seen[0]
        status, j = gui.post("/api/providers/probe", {
            "preset_id": "openai", "base_url": base,
            "api_key": "sk-good", "api_key_env": "PROBE_TEST_KEY"})
        assert j["ok"] and seen[0]["authorization"] == "Bearer sk-good"
    finally:
        srv.shutdown()


def test_probe_speaks_anthropic_authentication(gui):
    srv, seen = _hosted_server()
    try:
        base = "http://127.0.0.1:%d/v1" % srv.server_address[1]
        status, j = gui.post("/api/providers/probe", {
            "preset_id": "anthropic", "base_url": base, "api_key": "sk-good"})
        assert status == 200 and j["ok"]
        assert seen[0]["x_api_key"] == "sk-good"          # Claude's spelling
        assert seen[0]["anthropic_version"] == "2023-06-01"
    finally:
        srv.shutdown()


def test_probe_reports_failures_but_stays_importable(gui):
    srv, _seen = _hosted_server()
    try:
        base = "http://127.0.0.1:%d/v1" % srv.server_address[1]
        status, j = gui.post("/api/providers/probe", {
            "preset_id": "openai", "base_url": base, "api_key": "sk-wrong"})
        assert status == 200 and j["ok"] is False and j["verified"] is False
        assert j["error"].startswith("HTTPError: 401")
        # the curated preset list stays selectable while the key gets fixed
        assert [m["id"] for m in j["models"]] == \
            settings_api._preset_by_id("openai")["models"]
        assert all(m["suggested_profile_name"] for m in j["models"])
        status, j = gui.post("/api/providers/probe", {
            "preset_id": "openai", "base_url": "http://127.0.0.1:%d/v1"
            % _dead_port(), "api_key": "sk-good"})
        assert status == 200 and j["ok"] is False and j["models"]
        # unusable requests are the only errors that are 400s
        status, j = gui.post("/api/providers/probe", {"preset_id": "nope"})
        assert status == 400 and "unknown provider preset" in j["detail"]
        status, j = gui.post("/api/providers/probe", {"file": "user"})
        assert status == 400 and "base URL" in j["detail"]
    finally:
        srv.shutdown()


def test_probe_flags_models_the_config_already_imports(gui):
    srv, _seen = _hosted_server()
    base = "http://127.0.0.1:%d/v1" % srv.server_address[1]
    gui.cfg_path.parent.mkdir(parents=True, exist_ok=True)
    gui.cfg_path.write_text(
        '[profiles.gpt-4o-mini]\nmodel = "gpt-4o-mini"\nbase_url = "%s"\n'
        % base, encoding="utf-8")
    try:
        status, j = gui.post("/api/providers/probe", {
            "preset_id": "openai", "base_url": base, "api_key": "sk-good"})
        flags = {m["id"]: m["already_imported"] for m in j["models"]}
        assert flags == {"openai/gpt-5.4": False, "gpt-4o-mini": True}
    finally:
        srv.shutdown()


def test_import_names_every_profile_after_its_model(gui):
    preset = settings_api._preset_by_id("deepseek")
    base = "https://api.deepseek.com/v1"
    status, j = gui.post("/api/providers/import", {
        "preset_id": "deepseek", "base_url": base, "api_key": "sk-d",
        "models": [
            {"model": "deepseek-v4-flash", "profile_name": "deepseek_chat",
             "set_default": True},
            {"model": "DeepSeek-V4-Pro", "profile_name": "deepseek_v4_pro"},
            {"model": "glm-5"}]})   # no name: the live model id names it
    assert status == 200 and j["ok"]
    assert j["imported"] == ["deepseek_chat", "deepseek_v4_pro", "glm-5"]
    assert j["skipped"] == [] and j["default_profile"] == "deepseek_chat"
    assert j["uses_env"] is False and j["preset"] == "deepseek"
    body = tomllib.loads(gui.cfg_path.read_text(encoding="utf-8"))
    assert body["default_profile"] == "deepseek_chat"
    prof = body["profiles"]["deepseek_v4_pro"]
    assert prof["model"] == "DeepSeek-V4-Pro"            # the exact live id
    assert prof["base_url"] == base and prof["api_key"] == "sk-d"
    assert "api_key_env" not in prof
    assert prof["tier"] == "hosted" and prof["json_mode"] == "object"
    assert prof["max_read_tokens"] == 3000               # cloud gather budgets
    assert prof["context_tokens"] == preset["context_tokens"]
    assert prof["prompt_cache"] == "implicit"            # generic-hosted default
    assert body["profiles"]["glm-5"]["model"] == "glm-5"
    assert "deepseek_v4_pro" in j["user"]["profiles"]
    # selectable right away: STATE was reloaded
    _status, st = gui.get("/api/state")
    assert st["profile"] == "deepseek_chat" and st["model"] == "deepseek-v4-flash"
    assert {"deepseek_chat", "deepseek_v4_pro", "glm-5"} <= set(st["profiles"])


def test_import_never_renames_or_overwrites_existing_profiles(gui):
    gui.cfg_path.parent.mkdir(parents=True, exist_ok=True)
    gui.cfg_path.write_text(
        'default_profile = "glm-5"\n\n[profiles.glm-5]\nmodel = "glm-5"\n'
        'base_url = "http://a/v1"\n', encoding="utf-8")
    # an explicit name already taken -> skipped, the old profile untouched
    status, j = gui.post("/api/providers/import", {
        "base_url": "http://b/v1", "api_key": "k",
        "models": [{"model": "glm-5", "profile_name": "glm-5"}]})
    assert status == 200 and j["imported"] == [] and j["skipped"] == ["glm-5"]
    body = tomllib.loads(gui.cfg_path.read_text(encoding="utf-8"))
    assert body["profiles"]["glm-5"]["base_url"] == "http://a/v1"
    # no explicit name -> the model-derived suggestion collides and gets
    # uniquified against the file instead of clobbering the existing profile
    status, j = gui.post("/api/providers/import", {
        "base_url": "http://b/v1", "api_key": "k",
        "models": [{"model": "glm-5"}]})
    assert status == 200 and j["imported"] == ["glm-5-2"]
    body = tomllib.loads(gui.cfg_path.read_text(encoding="utf-8"))
    assert body["profiles"]["glm-5-2"]["base_url"] == "http://b/v1"
    assert body["profiles"]["glm-5"]["base_url"] == "http://a/v1"
    assert body["default_profile"] == "glm-5"            # the add did not steal it


def test_import_stores_the_key_or_the_env_var_that_holds_it(gui):
    preset = settings_api._preset_by_id("openrouter")
    # a pasted key wins and is what gets stored (the env var would shadow it
    # at request time, so only one of the two may exist in the profile)
    status, j = gui.post("/api/providers/import", {
        "preset_id": "openrouter", "api_key": "sk-real",
        "api_key_env": "SOMETHING_ELSE",
        "models": [{"model": "qwen/qwen3-coder"}]})
    assert status == 200 and j["uses_env"] is False
    prof = tomllib.loads(gui.cfg_path.read_text(encoding="utf-8"))["profiles"]
    assert prof["qwen3-coder"]["api_key"] == "sk-real"
    assert "api_key_env" not in prof["qwen3-coder"]
    assert prof["qwen3-coder"]["base_url"] == preset["base_url"]
    # no key pasted: the profile points at the preset's env var
    status, j = gui.post("/api/providers/import", {
        "preset_id": "openrouter",
        "models": [{"model": "x/y", "profile_name": "by_env"}]})
    assert j["uses_env"] is True
    prof = tomllib.loads(gui.cfg_path.read_text(encoding="utf-8"))["profiles"]
    assert prof["by_env"]["api_key_env"] == "OPENROUTER_API_KEY"
    assert "api_key" not in prof["by_env"]
    # and importing to the open project's file writes that file, not the user's
    from pedacito.gui import core
    root = gui.tmp / "proj"
    root.mkdir()
    core.STATE.root = root
    status, j = gui.post("/api/providers/import", {
        "file": "project", "base_url": "http://h/v1",
        "models": [{"model": "m", "profile_name": "in_project"}]})
    assert status == 200
    body = tomllib.loads((root / ".pedacito.toml").read_text(encoding="utf-8"))
    assert body["profiles"]["in_project"]["base_url"] == "http://h/v1"
    assert "in_project" not in tomllib.loads(
        gui.cfg_path.read_text(encoding="utf-8"))["profiles"]


def test_import_refuses_bad_requests_without_writing(gui):
    status, j = gui.post("/api/providers/import", {"base_url": "http://h/v1",
                                                   "models": []})
    assert status == 400 and "at least one model" in j["detail"]
    status, j = gui.post("/api/providers/import", {
        "base_url": "http://h/v1", "models": [{"model": "  "}]})
    assert status == 400 and "exact model id" in j["detail"]
    status, j = gui.post("/api/providers/import", {
        "base_url": "http://h/v1", "models": [{"model": "m",
                                               "profile_name": "bad name/"}]})
    assert status == 400 and "profile name" in j["detail"]
    status, j = gui.post("/api/providers/import", {"preset_id": "nope"})
    assert status == 400 and "unknown provider preset" in j["detail"]
    assert not gui.cfg_path.exists()       # the refusals wrote nothing


def test_probe_server_only_attaches_stored_key_on_matching_origin(gui):
    srv, seen = _hosted_server()
    try:
        good_base = "http://127.0.0.1:%d/v1" % srv.server_address[1]
        gui.cfg_path.parent.mkdir(parents=True, exist_ok=True)
        gui.cfg_path.write_text(
            '[profiles.stored]\n'
            f'base_url = "{good_base}"\n'
            'model = "m"\n'
            'api_key = "sk-good"\n',
            encoding="utf-8",
        )
        # Same (scheme, host, port): stored api_key is attached
        out = settings_api.probe_server(good_base, "", which="user",
                                        profile="stored")
        assert out["ok"] is True
        assert seen[-1]["authorization"] == "Bearer sk-good"

        # Different host/port: stored api_key is NOT sent
        srv2, seen2 = _hosted_server()
        try:
            other_base = "http://127.0.0.1:%d/v1" % srv2.server_address[1]
            out2 = settings_api.probe_server(other_base, "", which="user",
                                             profile="stored")
            assert out2["ok"] is False
            assert seen2[-1]["authorization"] != "Bearer sk-good"
        finally:
            srv2.shutdown()

        # Non-http(s) URL is rejected
        bad = settings_api.probe_server("file:///etc/passwd", "")
        assert bad["ok"] is False and "http" in bad["error"].lower()
    finally:
        srv.shutdown()


def test_probe_provider_enforces_http_scheme_and_env_allowlist(gui, monkeypatch):
    srv, seen = _hosted_server()
    monkeypatch.setenv("HOME", "sk-good")
    monkeypatch.setenv("CUSTOM_API_KEY", "sk-good")
    try:
        base = "http://127.0.0.1:%d/v1" % srv.server_address[1]
        # Arbitrary env var (HOME) is never read or sent
        status, j = gui.post("/api/providers/probe", {
            "preset_id": "custom", "base_url": base, "api_key_env": "HOME"})
        assert status == 200 and j["ok"] is False
        assert seen[-1]["authorization"] == "Bearer "

        # Valid *_API_KEY env var works
        status, j = gui.post("/api/providers/probe", {
            "preset_id": "custom", "base_url": base,
            "api_key_env": "CUSTOM_API_KEY"})
        assert status == 200 and j["ok"] is True
        assert seen[-1]["authorization"] == "Bearer sk-good"

        # Non-http(s) base_url is rejected with 400
        status, j = gui.post("/api/providers/probe", {
            "preset_id": "custom", "base_url": "file:///etc/passwd"})
        assert status == 400 and "http" in j["detail"].lower()
    finally:
        srv.shutdown()


def test_mcp_env_is_redacted_on_read_and_preserved_on_save(gui):
    spec = {
        "name": "srv", "transport": "stdio",
        "command": sys.executable, "args": ["-c", "pass"],
        "env": {"GITHUB_TOKEN": "ghp_secret123456", "LOG_LEVEL": "debug"},
        "trusted": False, "enabled": True,
    }
    status, body = gui.post("/api/mcp/servers", spec)
    assert status == 200 and body["ok"]
    row = next(s for s in body["servers"] if s["name"] == "srv")
    assert row["env"]["GITHUB_TOKEN"] == "***REDACTED***"
    assert row["env"]["LOG_LEVEL"] == "debug"

    # Re-saving with the redacted sentinel preserves the stored secret on disk
    status, body2 = gui.post("/api/mcp/servers", {
        **spec,
        "env": {"GITHUB_TOKEN": "***REDACTED***", "LOG_LEVEL": "info"},
        "enabled": False,
    })
    assert status == 200 and body2["ok"]
    on_disk = tomllib.loads(gui.cfg_path.read_text(encoding="utf-8"))
    assert on_disk["mcp_servers"]["srv"]["env"]["GITHUB_TOKEN"] == "ghp_secret123456"
    assert on_disk["mcp_servers"]["srv"]["env"]["LOG_LEVEL"] == "info"

    # An orphan sentinel for an unknown env key is refused
    status, bad = gui.post("/api/mcp/servers", {
        **spec,
        "env": {"NEW_SECRET_TOKEN": "***REDACTED***"},
    })
    assert status == 400 and bad["ok"] is False


def test_settings_js_shows_calibrated_temperature_over_global():
    """Settings -> Profiles must surface calibrated model-card choices (such as
    arch_family temperature / temperature_answer) in placeholders and inherit
    labels instead of displaying stale `(global)` values from [universal]."""
    js = (Path(__file__).resolve().parents[1]
          / "pedacito" / "gui" / "static" / "settings.js").read_text(
              encoding="utf-8")
    assert "function calibratedChoice(name)" in js
    assert "cal.card_choices" in js
    assert "name === 'temperature' || name === 'temperature_answer'" in js
    assert "var cal = calibratedChoice(item.name);" in js

