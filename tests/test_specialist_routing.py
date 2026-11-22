"""Specialist profile auto-routing (TECHNIQUES.md #50).

Pins the whole chain: the specialty taxonomy and config parsing, model-card
auto-tagging (name heuristics + three micro-probes), deterministic routing
selection, the clean VRAM handoff on one local server (and no handoff across
servers), graceful fallback, the planner note, and the CLI/GUI surface.

Hermetic: scripted clients, monkeypatched card storage and unload calls --
no network, no real model server.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import config as config_mod
from pedacito import llm as llm_mod
from pedacito import loop, modelcard, profiles, roles
from pedacito.config import Config, ConfigError

LOCAL = "http://127.0.0.1:11434/v1"
LOCAL2 = "http://127.0.0.1:11435/v1"
DEEPSEEK = "https://api.deepseek.com/v1"

GOOD_SVG = ('<svg viewBox="0 0 200 100">'
            '<circle id="wheel" cx="50" cy="50" r="20"/></svg>')
GOOD_ES = "Hola {name}, tu archivo tiene {count} errores."


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolated model-card storage (never the real user's cards)."""
    monkeypatch.setattr(modelcard, "_path",
                        lambda: tmp_path / "home" / "model_cards.json")
    (tmp_path / "home").mkdir()
    return tmp_path / "home"


def _card(model, url=LOCAL, overall=0.8, specialties=None):
    card = {"version": modelcard.CARD_VERSION,
            "key": f"{url.rstrip('/')}|{model}", "ts": time.time(),
            "model": model, "url": url,
            "measured": {"overall": overall},
            "choices": {}, "notes": [], "repeats": 3, "seconds": 1.0}
    if specialties is not None:
        card["measured"]["specialties"] = list(specialties)
    modelcard.save_card(card)
    return card


def _routed_cfg(**kw):
    cfg = Config()
    cfg.base_url = LOCAL
    cfg.model = "llama3.1-8b"
    cfg.vram_gb = 24
    cfg.sources = {}
    cfg.active_profile = "worker"
    cfg.auto_route_specialties = True
    cfg.profiles = {
        "worker": {"model": "llama3.1-8b"},
        "coder": {"model": "qwen2.5-coder-14b"},
        "ui": {"model": "gemma-3-27b"},
        "math": {"model": "deepseek-r1-14b"},
        "polyglot": {"model": "aya-35b",
                     "specialties": ["translation_i18n"]},
    }
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


class FakeClient:
    """The roles-test double: cfg, ledger, check/warmup, loaded_models."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.on_call = None
        self.site = ""
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "cached_tokens": 0, "cache_write_tokens": 0,
                      "seconds": 0.0, "cost": 0.0, "by_site": {}}
        self._stats_lock = threading.Lock()
        self.warmups = 0
        self.fail_check = False
        self.fail_warmup = False

    def check(self, warn=True):
        if self.fail_check:
            raise llm_mod.LLMError("unreachable")
        return ""

    def warmup(self):
        if self.fail_warmup:
            raise llm_mod.LLMError("out of memory")
        self.warmups += 1

    def loaded_models(self):
        return {self.cfg.model: "loaded"}


def _factory(fail_check=(), fail_warmup=()):
    """An LLMClient stand-in whose per-model failures are scripted."""
    built = []

    def make(cfg):
        c = FakeClient(cfg)
        c.fail_check = cfg.model in fail_check
        c.fail_warmup = cfg.model in fail_warmup
        built.append(c)
        return c

    make.built = built
    return make


# ------------------------------------------------- taxonomy + config parsing
def test_specialty_taxonomy_and_normalization():
    assert set(config_mod.SPECIALTIES) == {
        "general", "coding", "frontend_ui", "3d_graphics_svg",
        "math_algorithms", "testing_qa", "sql_data", "translation_i18n"}
    assert config_mod.normalize_specialty("Coding") == "coding"
    assert config_mod.normalize_specialty("My Tag") == "my_tag"
    assert config_mod.normalize_specialty("") == ""
    assert config_mod.normalize_specialty(None) == ""
    with pytest.raises(ConfigError):
        config_mod.normalize_specialty("bad tag!")
    # general is a label, not a routing target: no keywords, no extensions
    assert config_mod.SPECIALTIES["general"]["keywords"] == ()
    assert config_mod.SPECIALTIES["general"]["extensions"] == ()


def test_specialties_coercion_and_env():
    assert config_mod._coerce("specialties", ["CODING", "frontend_ui",
                                              "coding"]) == \
        ("coding", "frontend_ui")
    assert config_mod._coerce("specialties", []) == ()
    with pytest.raises(ConfigError):
        config_mod._coerce("specialties", "coding")      # not a list
    with pytest.raises(ConfigError):
        config_mod._coerce("specialties", ["no!pe"])
    assert config_mod._parse_env("specialties", "coding, sql_data") == \
        ("coding", "sql_data")
    assert config_mod._parse_env("auto_route_specialties", "true") is True


def test_specialties_survive_a_config_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "USER_CONFIG",
                        tmp_path / "home" / "config.toml")
    cfgfile = tmp_path / "config.toml"
    cfgfile.write_text(
        'auto_route_specialties = true\n'
        'specialties = ["coding", "testing_qa"]\n'
        '[universal]\nvram_gb = 24\n'
        '[profiles.ui]\nmodel = "gemma-3-27b"\n'
        'specialties = ["frontend_ui", "3d_graphics_svg"]\n',
        encoding="utf-8")
    cfg = config_mod.load(explicit=cfgfile, use_env=False,
                          require_hardware=False)
    assert cfg.auto_route_specialties is True
    assert cfg.specialties == ("coding", "testing_qa")
    assert cfg.sources["specialties"] != "default"
    assert cfg.profiles["ui"]["specialties"] == ["frontend_ui",
                                                 "3d_graphics_svg"]
    # the profile's explicit list wins over its model's name heuristics
    assert roles.specialties_for(cfg, "ui") == ("frontend_ui",
                                                "3d_graphics_svg")


# ------------------------------------------------------- name heuristics
@pytest.mark.parametrize("model,tags", [
    ("qwen2.5-coder-14b", ["coding", "testing_qa"]),
    ("codestral-22b", ["coding", "testing_qa"]),
    ("starcoder2-15b", ["coding", "testing_qa"]),
    ("devstral-small", ["coding", "testing_qa"]),
    ("gemma-3-27b", ["frontend_ui", "3d_graphics_svg"]),
    ("qwen2.5-vl-7b", ["frontend_ui", "3d_graphics_svg"]),
    ("pixtral-12b", ["frontend_ui", "3d_graphics_svg"]),
    ("llava-1.6-34b", ["frontend_ui", "3d_graphics_svg"]),
    ("deepseek-r1-14b", ["math_algorithms"]),
    ("qwq-32b", ["math_algorithms"]),
    ("mathstral-7b", ["math_algorithms"]),
    ("llama3.1-8b", ["general"]),
    ("", ["general"]),
])
def test_name_heuristics(model, tags):
    assert modelcard.infer_name_specialties(model) == tags


# ---------------------------------------------------------- micro-probes
def test_svg_probe_scoring():
    assert modelcard._score_svg(GOOD_SVG) == 1.0
    # parseable + right viewBox, but the wheel is wrong: still passes (0.7)
    assert modelcard._score_svg(
        '<svg viewBox="0 0 200 100"><circle id="x"/></svg>') == \
        pytest.approx(0.7)
    # right wheel, wrong viewBox
    assert modelcard._score_svg(
        '<svg viewBox="0 0 10 10">'
        '<circle id="wheel" cx="50" cy="50" r="20"/></svg>') == \
        pytest.approx(0.7)
    assert modelcard._score_svg("<svg>unclosed") == 0.0
    assert modelcard._score_svg("Here is your image: ...") == 0.0
    # a namespaced SVG still parses
    assert modelcard._score_svg(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100">'
        '<circle id="wheel" cx="50" cy="50" r="20"/></svg>') == 1.0


class ProbeClient:
    """Answers the three specialty probes; everything else gets 'ok'."""

    chars_per_token = 3.2

    def __init__(self, svg=GOOD_SVG, math="29", es=GOOD_ES):
        self.svg, self.math, self.es = svg, math, es
        self.prompts: list[str] = []
        self.kw: list[dict] = []

    def chat_json(self, messages, schema, max_tokens, **kw):
        return {"reasoning": "r", "action": "done"}

    def chat(self, messages, max_tokens, **kw):
        text = messages[-1]["content"]
        self.prompts.append(text)
        self.kw.append({"max_tokens": max_tokens, **kw})
        if "SVG image" in text:
            return self.svg
        if "Compute" in text:
            return self.math
        if "Translate" in text:
            return self.es
        return "ok"

    def probe_calls(self, needle):
        return [i for i, p in enumerate(self.prompts) if needle in p]


def test_specialty_probes_pass_and_fail():
    cfg = Config()
    good = ProbeClient()
    assert modelcard._t_svg_ui(good, cfg) == (1.0, [])
    assert modelcard._t_math_algo(good, cfg) == (1.0, [])
    assert modelcard._t_translation(good, cfg) == (1.0, [])
    # thinking off and small caps on every probe call
    for i in good.probe_calls("SVG image") + good.probe_calls("Compute") \
            + good.probe_calls("Translate"):
        assert good.kw[i]["disable_thinking"] is True
        assert good.kw[i]["max_tokens"] <= 200

    bad = ProbeClient(svg="sorry, no svg", math="512",
                      es="Hello {name}, your file has errors.")
    score, notes = modelcard._t_svg_ui(bad, cfg)
    assert score == 0.0 and notes
    score, notes = modelcard._t_math_algo(bad, cfg)
    assert score == 0.0 and notes
    score, notes = modelcard._t_translation(bad, cfg)
    assert score == 0.0 and notes          # placeholders AND wording lost

    # word boundary: 129 is not 29
    assert modelcard._t_math_algo(ProbeClient(math="129"), cfg)[0] == 0.0
    assert modelcard._t_math_algo(
        ProbeClient(math="The answer is 29."), cfg)[0] == 1.0
    # placeholders kept but wording wrong: half credit, below threshold
    score, _ = modelcard._t_translation(
        ProbeClient(es="Hola {name}, tu carpeta tiene {count} problemas."),
        cfg)
    assert score == pytest.approx(0.5)
    # fichero counts like archivo
    assert modelcard._t_translation(
        ProbeClient(es="Hola {name}, tu fichero tiene {count} errores."),
        cfg)[0] == 1.0


def test_specialty_probes_survive_a_dead_server():
    class Dead:
        def chat(self, messages, max_tokens, **kw):
            raise llm_mod.LLMError("connection refused")

    cfg = Config()
    for probe in (modelcard._t_svg_ui, modelcard._t_math_algo,
                  modelcard._t_translation):
        score, notes = probe(Dead(), cfg)
        assert score == 0.0 and notes


def test_tags_from_measurements_combine_name_probes_and_harness_skills():
    f = modelcard.specialty_tags_from_measurements
    # a plain name earns tags purely from the probes -- when the edit-skill
    # gate passes (item 15: edit_format >= 0.65 AND tool_choice >= 0.5)
    assert f("mystery-7b", {"svg_ui": 1.0, "math_algo": 1.0,
                            "translation": 1.0,
                            "edit_format": 0.7, "tool_choice": 0.6}) == \
        ["frontend_ui", "3d_graphics_svg", "math_algorithms",
         "translation_i18n"]
    # without measured edit skill the editing tags are gated away;
    # translation is prose, not editing, and survives
    assert f("mystery-7b", {"svg_ui": 1.0, "math_algo": 1.0,
                            "translation": 1.0}) == ["translation_i18n"]
    # edit_format AND tool_choice >= 0.75 earn coding + testing_qa
    assert "coding" in f("mystery-7b", {"edit_format": 0.8,
                                        "tool_choice": 0.8})
    assert "coding" not in f("mystery-7b", {"edit_format": 0.8,
                                            "tool_choice": 0.6})
    # a weak edit format strips the coder family's tags (item 15 gate)
    tags = f("qwen2.5-coder-14b", {"edit_format": 0.4})
    assert "coding" not in tags and "testing_qa" not in tags
    # general drops out as soon as anything specific is earned
    assert f("llama3.1-8b", {"math_algo": 1.0, "edit_format": 0.7,
                             "tool_choice": 0.6}) == ["math_algorithms"]
    assert f("llama3.1-8b", {}) == ["general"]
    # below threshold: nothing earned
    assert f("llama3.1-8b", {"svg_ui": 0.66, "math_algo": 0.5,
                             "edit_format": 0.7,
                             "tool_choice": 0.6}) == ["general"]


def test_calibrate_records_specialties_once(home):
    cfg = Config()
    cfg.base_url = LOCAL
    cfg.model = "mystery-7b"
    probe = ProbeClient()
    card = modelcard.calibrate(probe, cfg, verbose=False, quick=True)
    m = card["measured"]
    assert m["svg_ui"] == 1.0 and m["math_algo"] == 1.0
    assert m["translation"] == 1.0
    # the scripted client cannot edit: the edit-skill gate (item 15) strips
    # every editing tag; translation is prose and survives
    assert m["specialties"] == ["translation_i18n"]
    assert card["choices"]["specialties"] == m["specialties"]
    # each micro-probe ran exactly ONCE (not repeated like the skill probes)
    assert len(probe.probe_calls("SVG image")) == 1
    assert len(probe.probe_calls("Compute")) == 1
    assert len(probe.probe_calls("Translate")) == 1
    # and the card renders its tags
    assert "specialties" in modelcard.render(card)


def test_card_fills_specialties_only_when_unset(home):
    cfg = Config()
    cfg.base_url = LOCAL
    cfg.model = "mystery-7b"
    card = modelcard.calibrate(ProbeClient(), cfg, verbose=False, quick=True)
    modelcard.save_card(card)

    fresh = Config()
    fresh.base_url = LOCAL
    fresh.model = "mystery-7b"
    fresh.sources = {}
    modelcard.apply(fresh)
    assert fresh.specialties == tuple(card["choices"]["specialties"])
    assert fresh.sources["specialties"] == "model card"

    explicit = Config()
    explicit.base_url = LOCAL
    explicit.model = "mystery-7b"
    explicit.specialties = ("sql_data",)
    explicit.sources = {"specialties": "config file"}
    modelcard.apply(explicit)
    assert explicit.specialties == ("sql_data",)   # explicit always wins


# ------------------------------------------- detected vs explicit tags
def test_detected_specialties_prefer_the_card_then_the_name(home):
    cfg = _routed_cfg()
    # no card: name heuristics stand in
    assert roles.detected_specialties(cfg, "coder") == ("coding",
                                                        "testing_qa")
    card = _card("qwen2.5-coder-14b", specialties=["sql_data"])
    # item 15: saved tags are re-derived under the edit-skill gate, so the
    # card must carry the measurements that earn them
    card["measured"].update({"edit_format": 0.9, "tool_choice": 0.9,
                             "sql_data": 1.0})
    modelcard.save_card(card)
    assert roles.detected_specialties(cfg, "coder") == ("coding", "testing_qa",
                                                        "sql_data")
    cfg.use_model_card = False
    assert roles.detected_specialties(cfg, "coder") == ("coding",
                                                        "testing_qa")


def test_explicit_profile_specialties_beat_the_card(home):
    cfg = _routed_cfg()
    _card("qwen2.5-coder-14b", specialties=["math_algorithms"])
    cfg.profiles["coder"]["specialties"] = ["coding"]
    assert roles.specialties_for(cfg, "coder") == ("coding",)
    # the active config's own effective list counts for the active profile
    cfg.specialties = ("sql_data",)
    assert roles.specialties_for(cfg, "worker") == ("sql_data",)


# ------------------------------------------------------ routing selection
def test_routing_off_or_single_profile_never_routes():
    cfg = _routed_cfg(auto_route_specialties=False)
    assert roles.select_specialist_profile("refactor app.py", None, cfg) == \
        ("", "")
    cfg = _routed_cfg()
    cfg.profiles = {"worker": {"model": "llama3.1-8b"}}
    assert roles.select_specialist_profile("refactor app.py", None, cfg) == \
        ("", "")


@pytest.mark.parametrize("task,profile,tag", [
    ("refactor the parser module in app.py", "coder", "coding"),
    ("write tests for the parser", "coder", "testing_qa"),
    ("fix the button layout in styles.css", "ui", "frontend_ui"),
    ("draw an svg sprite animation", "ui", "3d_graphics_svg"),
    ("optimize the sorting algorithm", "math", "math_algorithms"),
    ("translate the locales to spanish", "polyglot", "translation_i18n"),
    ("migrate the database schema in init.sql", "coder", "sql_data"),
])
def test_routing_by_keywords_and_extensions(task, profile, tag):
    cfg = _routed_cfg()
    if tag == "sql_data":
        # nobody is tagged sql_data by name: give the coder that hat
        cfg.profiles["coder"]["specialties"] = ["coding", "sql_data"]
    assert roles.select_specialist_profile(task, None, cfg) == (profile, tag)


def test_routing_unmatched_task_stays_on_the_worker():
    cfg = _routed_cfg()
    assert roles.select_specialist_profile("hello there", None, cfg) == \
        ("", "")
    # a specialty nobody carries stays put too
    assert roles.select_specialist_profile(
        "write the sql query for the report", None, cfg) == ("", "")


def test_hosted_profiles_are_excluded_from_local_routing_by_default(home):
    """auto_route_allow_hosted defaults to False: a matching cloud profile
    never wins a task when local candidates exist (or at all)."""
    cfg = _routed_cfg()
    assert cfg.auto_route_allow_hosted is False
    cfg.profiles["flash"] = {"model": "deepseek-flash", "base_url": DEEPSEEK,
                             "specialties": ["frontend_ui", "coding"]}
    _card("deepseek-flash", url=DEEPSEEK, overall=1.0)
    assert roles.profile_is_hosted(cfg, "flash") is True
    assert roles.profile_is_hosted(cfg, "coder") is False
    # no local UI specialist left: the hosted one must NOT take the task
    cfg.profiles.pop("ui")
    assert roles.select_specialist_profile(
        "fix the button layout in styles.css", None, cfg) == ("", "")
    # opt in explicitly: now it may route there
    cfg.auto_route_allow_hosted = True
    assert roles.select_specialist_profile(
        "fix the button layout in styles.css", None, cfg) == \
        ("flash", "frontend_ui")


def test_local_profile_mislabeled_hosted_tier_still_routes(home):
    """The tier is display metadata; locality comes from the endpoint. A local
    GGUF named 'DeepSeek-Coder' with tier='hosted' stays in the pool."""
    cfg = _routed_cfg()
    cfg.profiles["dcoder"] = {"model": "DeepSeek-Coder-V2-Lite",
                              "base_url": LOCAL, "tier": "hosted",
                              "specialties": ["math_algorithms"]}
    _card("DeepSeek-Coder-V2-Lite", overall=0.9)
    assert roles.profile_is_hosted(cfg, "dcoder") is False
    assert roles.select_specialist_profile(
        "optimize the sorting algorithm", None, cfg) == \
        ("dcoder", "math_algorithms")


def test_guess_tier_uses_the_endpoint_not_the_model_name():
    assert config_mod._guess_tier(
        "dcoder", {"model": "DeepSeek-Coder-V2-Lite",
                   "base_url": LOCAL}) is None
    assert config_mod._guess_tier(
        "flash", {"model": "deepseek-flash",
                  "base_url": DEEPSEEK}) == "hosted"
    assert config_mod._guess_tier(
        "qwen", {"model": "qwen2.5-coder-14b",
                 "base_url": LOCAL}) == "medium"


def test_math_probe_retries_once_after_a_truncated_reasoning_reply():
    class ThinkingClient:
        chars_per_token = 3.2

        def __init__(self):
            self.calls = 0
            self.last_finish_reason = "length"

        def chat(self, messages, max_tokens, **kw):
            self.calls += 1
            if self.calls == 1:
                return "Let me work through the precedence: 3 ** 2 = 9, then "
            self.last_finish_reason = "stop"
            return "29"

    c = ThinkingClient()
    assert modelcard._t_math_algo(c, Config()) == (1.0, [])
    assert c.calls == 2

    # A complete wrong answer is not retried (no wasted calibration calls).
    bad = ProbeClient(math="512")
    assert modelcard._t_math_algo(bad, Config())[0] == 0.0
    assert len(bad.probe_calls("Compute")) == 1


def test_routing_ignores_a_profile_name_in_the_specialty_prefix():
    """Regression: the planner used to copy a PROFILE name into
    `[specialty: <profile>]`. That slug matched no card, so routing returned
    ("", "") and the orchestrator kept specialist work. A profile name is not
    a tag: fall back to the task's own keywords and leave the orchestrator."""
    cfg = _routed_cfg()
    cfg.active_profile = "orchestrator"
    cfg.model = "nemotron-orchestrator-8b"
    cfg.profiles["orchestrator"] = {"model": "nemotron-orchestrator-8b"}
    # the audit's real shape: a profile name inside the prefix
    assert roles.select_specialist_profile(
        "[specialty: coder] refactor the parser in app.py", None, cfg) == \
        ("coder", "coding")
    # an unknown custom tag also falls back to keyword scoring
    assert roles.select_specialist_profile(
        "[specialty: nosuchtag] fix the button layout in styles.css",
        None, cfg) == ("ui", "frontend_ui")


def test_planner_note_lists_the_tag_vocabulary_not_profile_names():
    cfg = _routed_cfg()
    note = roles.specialty_planner_note(cfg)
    assert "TAG vocabulary" in note
    assert "never a profile name" in note
    assert "math_algorithms" in note and "frontend_ui" in note


def test_routing_explicit_prefix_and_tag_override_keywords():
    cfg = _routed_cfg()
    task = "[specialty: math_algorithms] fix the button layout in styles.css"
    assert roles.select_specialist_profile(task, None, cfg) == \
        ("math", "math_algorithms")
    assert roles.select_specialist_profile(
        "fix the button layout", None, cfg,
        explicit_tag="math_algorithms") == ("math", "math_algorithms")
    # custom slugs route too
    cfg.profiles["robot"] = {"model": "arm-7b", "specialties": ["robotics"]}
    assert roles.select_specialist_profile(
        "[specialty: robotics] calibrate the arm", None, cfg) == \
        ("robot", "robotics")


def test_routing_stays_when_the_active_profile_already_fits():
    cfg = _routed_cfg()
    cfg.active_profile = "coder"
    assert roles.select_specialist_profile("refactor app.py", None, cfg) == \
        ("", "")


def test_routing_tie_breaks_resident_then_server_then_card(home):
    cfg = _routed_cfg()
    cfg.profiles = {
        "worker": {"model": "llama3.1-8b"},
        "a": {"model": "coder-a", "specialties": ["coding"]},
        "b": {"model": "coder-b", "specialties": ["coding"]},
    }
    task = "refactor the module"
    # a resident model wins the tie
    assert roles.select_specialist_profile(
        task, None, cfg, resident={"coder-b"}) == ("b", "coding")
    # no resident: the higher card overall wins
    _card("coder-b", overall=0.9)
    assert roles.select_specialist_profile(task, None, cfg) == ("b", "coding")
    # a separate server beats a higher card on the worker's own server
    cfg.profiles["a"]["base_url"] = LOCAL2
    _card("coder-a", url=LOCAL2, overall=0.95)
    assert roles.select_specialist_profile(task, None, cfg) == ("a", "coding")


def test_routing_test_file_shapes_boost_testing_qa():
    cfg = _routed_cfg()
    cfg.profiles = {
        "worker": {"model": "llama3.1-8b"},
        "dev": {"model": "coder-x", "specialties": ["coding"]},
        "qa": {"model": "tester-x", "specialties": ["testing_qa"]},
    }
    # .py alone would go to dev; the test-file shape tips it to qa
    assert roles.select_specialist_profile(
        "update tests/test_parser.py", None, cfg) == ("qa", "testing_qa")
    assert roles.select_specialist_profile(
        "update parser.py", None, cfg) == ("dev", "coding")


# ------------------------------------------------- clean VRAM handoff
def _handoff_setup(monkeypatch, profiles=None, base=LOCAL,
                   fail_check=(), fail_warmup=()):
    cfg = Config()
    cfg.base_url = base
    cfg.model = "worker-m"
    cfg.vram_gb = 24
    cfg.sources = {}
    cfg.active_profile = "worker"
    cfg.profiles = profiles or {
        "worker": {"model": "worker-m"},
        "coder": {"model": "coder-m"},
        "ui": {"model": "ui-m"},
    }
    factory = _factory(fail_check, fail_warmup)
    monkeypatch.setattr(llm_mod, "LLMClient", factory)
    unloaded: list[str] = []

    def fake_unload(client):
        unloaded.append(client.cfg.model)
        return client.cfg.model

    monkeypatch.setattr(llm_mod, "unload_model", fake_unload)
    worker = FakeClient(cfg)
    worker.on_call = lambda rec: None
    return cfg, worker, factory, unloaded


def test_same_server_handoff_unloads_then_loads(monkeypatch):
    cfg, worker, factory, unloaded = _handoff_setup(monkeypatch)
    c1 = roles.switch_specialist_client(cfg, worker, "coder")
    assert c1 is not worker and c1.cfg.model == "coder-m"
    assert unloaded == ["worker-m"]              # the worker was evicted
    assert c1._swap_info["unloaded"] == "worker-m"
    assert c1._swap_info["loaded"] == "coder-m"
    assert c1._swap_info["elapsed_s"] >= 0
    assert cfg.__dict__["_specialist_on_worker_server"] == "coder"
    # one usage ledger: warmup tokens land in the worker's totals
    assert c1.stats is worker.stats
    assert c1._stats_lock is worker._stats_lock
    assert c1.on_call is worker.on_call

    # second specialist: the PREVIOUS specialist is evicted, not the worker
    c2 = roles.switch_specialist_client(cfg, worker, "ui")
    assert c2.cfg.model == "ui-m"
    assert unloaded == ["worker-m", "coder-m"]

    # back to the cached coder: the card changed hands, so hand off again
    warms = c1.warmups
    c3 = roles.switch_specialist_client(cfg, worker, "coder")
    assert c3 is c1                              # cached client reused
    assert unloaded == ["worker-m", "coder-m", "ui-m"]
    assert c1.warmups == warms + 1               # re-loaded into VRAM
    assert c1._swap_info["unloaded"] == "ui-m"


def test_different_servers_keep_both_warm(monkeypatch):
    cfg, worker, factory, unloaded = _handoff_setup(
        monkeypatch,
        profiles={"worker": {"model": "worker-m"},
                  "gpu2": {"model": "big-m", "base_url": LOCAL2}})
    c = roles.switch_specialist_client(cfg, worker, "gpu2")
    assert c is not worker and c.cfg.model == "big-m"
    assert unloaded == []                        # no eviction across servers
    assert getattr(c, "_swap_info", None) is None
    assert cfg.__dict__.get("_specialist_on_worker_server", "") == ""


def test_hosted_specialist_never_unloads(monkeypatch):
    cfg, worker, factory, unloaded = _handoff_setup(
        monkeypatch, base=DEEPSEEK,
        profiles={"worker": {"model": "worker-m", "base_url": DEEPSEEK},
                  "flash": {"model": "flash-m", "base_url": DEEPSEEK}})
    c = roles.switch_specialist_client(cfg, worker, "flash")
    assert c is not worker
    assert unloaded == []
    assert getattr(c, "_swap_info", None) is None


def test_unreachable_specialist_falls_back_once(monkeypatch, capsys):
    cfg, worker, factory, unloaded = _handoff_setup(
        monkeypatch, fail_check=("coder-m",))
    assert roles.switch_specialist_client(cfg, worker, "coder") is worker
    assert "unavailable" in capsys.readouterr().err
    assert unloaded == []                        # nothing evicted on failure
    # the verdict is cached: no new construction on the second call
    # (derive_config's auto-fit probe may build one extra during the first)
    built = len(factory.built)
    assert roles.switch_specialist_client(cfg, worker, "coder") is worker
    assert len(factory.built) == built


def test_warmup_failure_falls_back_to_the_worker(monkeypatch, capsys):
    cfg, worker, factory, unloaded = _handoff_setup(
        monkeypatch, fail_warmup=("coder-m",))
    assert roles.switch_specialist_client(cfg, worker, "coder") is worker
    assert "failed to load" in capsys.readouterr().err


def test_unknown_specialist_profile_falls_back(monkeypatch, capsys):
    cfg, worker, factory, unloaded = _handoff_setup(monkeypatch)
    assert roles.switch_specialist_client(cfg, worker, "ghost") is worker
    assert "unknown profile" in capsys.readouterr().err


def test_restore_worker_client_hands_the_card_back(monkeypatch):
    cfg, worker, factory, unloaded = _handoff_setup(monkeypatch)
    roles.switch_specialist_client(cfg, worker, "coder")
    swap = roles.restore_worker_client(cfg, worker)
    assert swap is not None
    assert swap["unloaded"] == "coder-m"
    assert swap["loaded"] == "worker-m"
    assert worker.warmups == 1                   # worker re-warmed
    assert "_specialist_on_worker_server" not in cfg.__dict__
    assert roles.restore_worker_client(cfg, worker) is None   # idempotent


# ------------------------------------------------------- unload_model itself
def test_unload_model_lemonade_then_lmstudio_then_ollama(monkeypatch):
    cfg = Config()
    cfg.base_url = LOCAL
    cfg.model = "m"
    client = SimpleNamespace(cfg=cfg, headers={})
    calls = []

    class R:
        def __init__(self, code):
            self.status_code = code

    # Lemonade's native endpoint is tried FIRST and ends the handshake.
    def lemonade(url, headers=None, json=None, timeout=None):
        calls.append((url, json, timeout))
        if url.endswith("/api/v1/unload"):
            return R(200)
        raise AssertionError("native Lemonade unload should win first")

    monkeypatch.setattr(llm_mod.requests, "post", lemonade)
    assert llm_mod.unload_model(client) == "m"
    assert len(calls) == 1
    assert calls[0][0] == "http://127.0.0.1:11434/api/v1/unload"
    assert calls[0][1] == {"model": "m"}
    assert calls[0][2] == 2                      # short timeout, best-effort

    # Not Lemonade: LM Studio's native unload answers next.
    calls.clear()

    def lmstudio(url, **kw):
        calls.append(url)
        if url.endswith("/api/v0/models/unload"):
            return R(200)
        return R(404)

    monkeypatch.setattr(llm_mod.requests, "post", lmstudio)
    assert llm_mod.unload_model(client) == "m"
    assert calls == ["http://127.0.0.1:11434/api/v1/unload",
                     "http://127.0.0.1:11434/api/v0/models/unload"]

    # Ollama's keep_alive=0 is the last resort (its native /api/v1 is 404).
    calls.clear()

    def ollama(url, **kw):
        calls.append((url, kw.get("json")))
        if url.endswith("/api/generate"):
            return R(200)
        return R(404)

    monkeypatch.setattr(llm_mod.requests, "post", ollama)
    assert llm_mod.unload_model(client) == "m"
    assert calls[-1] == ("http://127.0.0.1:11434/api/generate",
                         {"model": "m", "keep_alive": 0})

    # every endpoint down: "" and the caller proceeds to warmup()
    def dead(url, **kw):
        raise OSError("down")

    monkeypatch.setattr(llm_mod.requests, "post", dead)
    assert llm_mod.unload_model(client) == ""


def test_unload_model_never_touches_hosted_servers(monkeypatch):
    cfg = Config()
    cfg.base_url = DEEPSEEK
    cfg.model = "m"

    def boom(url, **kw):
        raise AssertionError("hosted servers must never be unloaded")

    monkeypatch.setattr(llm_mod.requests, "post", boom)
    assert llm_mod.unload_model(SimpleNamespace(cfg=cfg, headers={})) == ""
    assert llm_mod.unload_model(SimpleNamespace(cfg=None)) == ""


# ------------------------------------------- loop hook: events + telemetry
def test_loop_route_emits_events_and_telemetry(tmp_path, monkeypatch, capsys):
    cfg, worker, factory, unloaded = _handoff_setup(monkeypatch)
    cfg.auto_route_specialties = True
    cfg.profiles["coder"]["model"] = "qwen2.5-coder-14b"
    client, agent = loop._specialist_route(
        tmp_path, cfg, worker, None, None,
        "refactor the parser module in app.py")
    assert client is not worker
    assert client.cfg.model == "qwen2.5-coder-14b"
    assert unloaded == ["worker-m"]
    err = capsys.readouterr().err
    assert "specialist router: switching worker to profile 'coder'" in err
    files = list((tmp_path / ".pedacito" / "events").glob("*.jsonl"))
    assert files
    recs = [json.loads(ln)
            for ln in files[0].read_text(encoding="utf-8").splitlines()]
    routed = [r for r in recs if r["kind"] == "specialist_routed"]
    swaps = [r for r in recs if r["kind"] == "model_swap"]
    assert routed and routed[0]["profile"] == "coder"
    assert routed[0]["specialty"] == "coding"
    assert swaps and swaps[0]["unloaded"] == "worker-m"
    assert swaps[0]["loaded"] == "qwen2.5-coder-14b"


def test_loop_route_restores_worker_for_unmatched_task(tmp_path, monkeypatch):
    cfg, worker, factory, unloaded = _handoff_setup(monkeypatch)
    cfg.auto_route_specialties = True
    cfg.profiles["coder"]["model"] = "qwen2.5-coder-14b"
    # first task routes to the coder...
    client, _ = loop._specialist_route(
        tmp_path, cfg, worker, None, None, "refactor app.py")
    assert client is not worker
    # ...the next unmatched task hands the card back to the worker
    client2, _ = loop._specialist_route(
        tmp_path, cfg, worker, None, None, "hello there")
    assert client2 is worker
    assert unloaded == ["worker-m", "qwen2.5-coder-14b"]
    assert worker.warmups == 1


def test_loop_route_never_raises_on_a_broken_router(tmp_path, monkeypatch):
    cfg, worker, factory, unloaded = _handoff_setup(monkeypatch)
    cfg.auto_route_specialties = True

    def boom(*a, **kw):
        raise RuntimeError("router exploded")

    monkeypatch.setattr(roles, "select_specialist_profile", boom)
    client, agent = loop._specialist_route(
        tmp_path, cfg, worker, None, None, "refactor app.py")
    assert client is worker                      # graceful fallback


def test_specialist_zero_edit_falls_back_to_top_coding_specialist(
        tmp_path, monkeypatch, home, capsys):
    """BUG_HUNT #14 Finding #5: a routed math specialist (LFM2.5-24B-A2B)
    streamed unbounded reasoning and wrote 0 files. Rather than fail the task,
    retry it once on the top-scoring local coding specialist."""
    from pedacito.edits import Result

    cfg, worker, factory, unloaded = _handoff_setup(
        monkeypatch,
        profiles={"worker": {"model": "worker-m"},
                  "math": {"model": "math-m",
                           "specialties": ["math_algorithms"]},
                  "coder": {"model": "coder-m",
                            "specialties": ["coding"]}})
    cfg.auto_route_specialties = True
    for model, tag, overall in (("math-m", "math_algorithms", 0.9),
                                ("coder-m", "coding", 1.0)):
        card = _card(model, overall=overall, specialties=[tag])
        card["measured"].update({"edit_format": 1.0, "tool_choice": 1.0})
        modelcard.save_card(card)

    calls = []

    def fake_ladder(root, cfg_, client, agent, idx_, task, **kw):
        calls.append(client.cfg.model)
        if client.cfg.model == "math-m":
            return loop.TaskResult([], "", False, [], idx_)   # 0 edits
        return loop.TaskResult([Result(True, "math_utils.py", "applied")],
                               "done", True, ["math_utils.py"], idx_)

    monkeypatch.setattr(loop, "_run_ladder", fake_ladder)
    idx = SimpleNamespace(root=tmp_path, files=[])
    tr = loop.run_task(tmp_path, cfg, worker, None, idx,
                       "optimize the sorting algorithm", apply=True)

    assert calls == ["math-m", "coder-m"]       # specialist, then the coder
    assert tr.applied and tr.changed == ["math_utils.py"]
    err = capsys.readouterr().err
    assert "top-scoring coding specialist 'coder'" in err
    # the failed math card was handed off cleanly before the coder loaded
    assert unloaded == ["worker-m", "math-m"]
    assert roles.select_coding_fallback(
        cfg, exclude=("math",)) == "coder"


def test_planner_note_lists_specialists_only_when_routing_is_on():
    cfg = _routed_cfg()
    note = roles.specialty_planner_note(cfg)
    assert "[specialty: TAG]" in note
    assert "coder: coding, testing_qa" in note
    assert "group tasks" in note.lower()
    assert roles.specialty_planner_note(
        _routed_cfg(auto_route_specialties=False)) == ""
    # hosted profiles are not advertised to the planner in a local-only pool
    cfg = _routed_cfg()
    cfg.profiles["flash"] = {"model": "deepseek-flash", "base_url": DEEPSEEK,
                             "specialties": ["coding"]}
    assert "flash" not in roles.specialty_planner_note(cfg)
    cfg.auto_route_allow_hosted = True
    assert "flash" in roles.specialty_planner_note(cfg)
    # nothing to route between: general profiles only
    cfg = _routed_cfg()
    cfg.profiles = {"a": {"model": "llama3.1-8b"},
                    "b": {"model": "mistral-7b"}}
    assert roles.specialty_planner_note(cfg) == ""


# ------------------------------------------------------------- CLI surface
def test_cli_auto_route_flags():
    from pedacito import cli
    ap = cli.build_parser()
    args = ap.parse_args(["ask", "-t", "fix it", ".", "--auto-route"])
    assert args.auto_route is True and args.no_auto_route is False
    args = ap.parse_args(["ask", "-t", "fix it", ".", "--no-auto-route"])
    assert args.no_auto_route is True
    args = ap.parse_args(["new", "proj", "--auto-route"])
    assert args.auto_route is True
    args = ap.parse_args(["auto", "-t", "goal", ".", "--auto-route"])
    assert args.auto_route is True


def test_cli_flag_overrides_config(tmp_path, monkeypatch):
    from pedacito import cli
    monkeypatch.setattr(config_mod, "USER_CONFIG",
                        tmp_path / "home" / "config.toml")
    monkeypatch.chdir(tmp_path)
    cfgfile = tmp_path / "config.toml"
    cfgfile.write_text('[universal]\nvram_gb = 24\n'
                       'auto_route_specialties = true\n', encoding="utf-8")
    ap = cli.build_parser()
    args = ap.parse_args(["ask", "-t", "x", ".", "--config", str(cfgfile),
                          "--no-auto-route"])
    cfg = cli._config(args)
    assert cfg.auto_route_specialties is False
    assert cfg.sources["auto_route_specialties"] == "--no-auto-route"
    args = ap.parse_args(["ask", "-t", "x", ".", "--config", str(cfgfile)])
    assert cli._config(args).auto_route_specialties is True


def test_profiles_table_shows_specialties(home):
    cfg = _routed_cfg()
    rows = profiles.list_profiles(cfg)
    by_name = {r["name"]: r for r in rows}
    assert by_name["coder"]["specialties"] == ("coding", "testing_qa")
    assert by_name["polyglot"]["specialties"] == ("translation_i18n",)
    out = profiles.render(rows)
    assert "specialties" in out
    assert "coding, testing_qa" in out


# ------------------------------------------------------------- GUI surface
def test_settings_payload_carries_specialty_tags_and_hints(tmp_path,
                                                           monkeypatch):
    from pedacito.gui import core, settings_api
    monkeypatch.setattr(settings_api, "USER_CONFIG",
                        tmp_path / "config.toml")
    monkeypatch.setattr(modelcard, "_path",
                        lambda: tmp_path / "model_cards.json")
    monkeypatch.setattr(core.STATE, "cfg", None)
    monkeypatch.setattr(core.STATE, "root", None)
    (tmp_path / "config.toml").write_text(
        '[universal]\nvram_gb = 24\n'
        '[profiles.worker]\nmodel = "llama3.1-8b"\n'
        '[profiles.coder]\nmodel = "qwen2.5-coder-14b"\n',
        encoding="utf-8")
    out = settings_api.payload()
    tags = [t["tag"] for t in out["specialty_tags"]]
    assert tags == list(config_mod.SPECIALTIES)
    assert out["specialty_hints"]["coder"]["detected"] == ["coding",
                                                           "testing_qa"]
    assert out["specialty_hints"]["coder"]["explicit"] == []
    assert out["resolved"]["auto_route_specialties"]["value"] is False


def test_settings_write_specialties_and_auto_route(tmp_path, monkeypatch):
    from pedacito.gui import core, settings_api
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setattr(settings_api, "USER_CONFIG", cfg_path)
    monkeypatch.setattr(modelcard, "_path",
                        lambda: tmp_path / "model_cards.json")
    monkeypatch.setattr(core.STATE, "cfg", None)
    monkeypatch.setattr(core.STATE, "root", None)
    cfg_path.write_text(
        '[universal]\nvram_gb = 24\n'
        '[profiles.coder]\nmodel = "qwen2.5-coder-14b"\n',
        encoding="utf-8")
    settings_api.apply_changes("user", "coder",
                               {"specialties": ["coding", "testing_qa"]}, [])
    settings_api.apply_changes("user", None,
                               {"auto_route_specialties": True}, [])
    data = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    assert data["profiles"]["coder"]["specialties"] == ["coding",
                                                        "testing_qa"]
    assert data["universal"]["auto_route_specialties"] is True
    out = settings_api.payload()
    assert out["specialty_hints"]["coder"]["explicit"] == ["coding",
                                                           "testing_qa"]


def test_gui_state_tracks_the_routed_specialist(tmp_path, monkeypatch):
    from pedacito.gui import core
    monkeypatch.setattr(core, "USER_CONFIG", tmp_path / "config.toml")
    monkeypatch.setattr(core, "RECENT_FILE", tmp_path / "recent.json")
    st = core.State()
    try:
        cfg = _routed_cfg()
        with st.lock:
            st.cfg = cfg
        assert st.status()["auto_route_specialties"] is True
        assert st.engine()["specialist"] == {}
        st.note_event("specialist_routed",
                      {"profile": "coder", "model": "qwen2.5-coder-14b",
                       "specialty": "coding"})
        eng = st.engine()
        assert eng["specialist"] == {"profile": "coder",
                                     "model": "qwen2.5-coder-14b",
                                     "specialty": "coding"}
        assert any(a["tool"] == "specialist" for a in eng["actions"])
        st.note_event("model_swap",
                      {"unloaded": "llama3.1-8b",
                       "loaded": "qwen2.5-coder-14b", "elapsed_s": 1.2})
        assert any(a["tool"] == "model-swap" for a in st.engine()["actions"])
        # the specialist stays shown while it holds the card...
        assert st.engine()["specialist"]["profile"] == "coder"
        # ...and clears when the worker's own model is loaded back
        st.note_event("model_swap",
                      {"unloaded": "qwen2.5-coder-14b",
                       "loaded": "llama3.1-8b", "elapsed_s": 0.9})
        assert st.engine()["specialist"] == {}
    finally:
        st.bus.close()


def test_role_and_scout_clients_share_ledger_cancel_and_streaming_callbacks(
        monkeypatch):
    cfg = _routed_cfg(
        scout_profile="coder",
        roles={"planner": "ui"},
        profiles={
            "worker": {"model": "worker-m"},
            "coder": {"model": "coder-m", "base_url": LOCAL2},
            "ui": {"model": "ui-m", "base_url": LOCAL2},
        },
    )
    orig_check, orig_warmup = llm_mod.LLMClient.check, llm_mod.LLMClient.warmup
    llm_mod.LLMClient.check = lambda self, warn=True: ""
    llm_mod.LLMClient.warmup = lambda self: None
    try:
        worker = llm_mod.LLMClient(cfg)
        cancel = threading.Event()
        worker.cancel_event = cancel
        worker.on_call = lambda rec: None
        worker.stream_callback = lambda d: None
        worker.on_stream = lambda ev: None
        worker.on_stream_event = lambda ev: None
        worker.on_start = lambda ev: None
        worker.echo_stream = False

        planner = roles.client_for(cfg, "planner", worker)
        scout = roles.scout_client(cfg, worker)
        spec = roles.switch_specialist_client(cfg, worker, "coder")
        for c in (planner, scout, spec):
            assert c is not worker
            assert c.stats is worker.stats
            assert c._stats_lock is worker._stats_lock
            assert c.cancel_event is worker.cancel_event
            assert c.on_call is worker.on_call
            assert c.stream_callback is worker.stream_callback
            assert c.on_stream is worker.on_stream
            assert c.on_stream_event is worker.on_stream_event
            assert c.on_start is worker.on_start
            assert c.echo_stream is False
    finally:
        llm_mod.LLMClient.check = orig_check
        llm_mod.LLMClient.warmup = orig_warmup


def test_specialist_warmup_failure_clears_slot_and_rewarms_worker(
        monkeypatch, capsys):
    cfg, worker, factory, unloaded = _handoff_setup(
        monkeypatch, fail_warmup=("coder-m",))
    cfg.__dict__["_specialist_on_worker_server"] = "stale"
    assert roles.switch_specialist_client(cfg, worker, "coder") is worker
    assert unloaded == ["worker-m"]
    assert "_specialist_on_worker_server" not in cfg.__dict__
    assert worker.warmups == 1
    assert "failed to load" in capsys.readouterr().err

    # Also test reload failure of a cached specialist after unloading another
    cfg2, worker2, factory2, unloaded2 = _handoff_setup(monkeypatch)
    c_coder = roles.switch_specialist_client(cfg2, worker2, "coder")
    c_ui = roles.switch_specialist_client(cfg2, worker2, "ui")
    assert c_ui is not worker2
    assert cfg2.__dict__["_specialist_on_worker_server"] == "ui"
    c_coder.fail_warmup = True
    assert roles.switch_specialist_client(cfg2, worker2, "coder") is worker2
    assert "_specialist_on_worker_server" not in cfg2.__dict__
    assert worker2.warmups == 1
    assert "failed to reload" in capsys.readouterr().err
