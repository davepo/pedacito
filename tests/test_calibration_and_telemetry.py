"""Calibration badges, hosted lockout, architecture defaults, the three new
specialty probes, `calibrate --all`, and live telemetry (the D1-D7 pass).

Hermetic: scripted clients, monkeypatched card storage and unload calls, and
the shared GUI server fixture -- no network, no real model server.
"""
# `gui` below is test_gui's server fixture: pytest hands it to each test as a
# parameter, which ruff otherwise reads as a redefinition.
# ruff: noqa: F811
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pedacito import arch_defaults, cli, modelcard, profiles, roles
from pedacito import config as config_mod
from pedacito import llm as llm_mod
from pedacito.config import Config
from pedacito.gui import core, jobs
from test_gui import gui  # noqa: F401  (the shared server fixture)

LOCAL = "http://127.0.0.1:11434/v1"
HOSTED = "https://api.deepseek.com/v1"

REFUSAL = ("Calibration is only for local models. Profile '{name}' ({model}) "
           "is a hosted provider model and uses provider-table defaults "
           "automatically.")


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolated model-card storage (never the real user's cards)."""
    monkeypatch.setattr(modelcard, "_path",
                        lambda: tmp_path / "home" / "model_cards.json")
    (tmp_path / "home").mkdir()
    return tmp_path / "home"


def _card(model, url=LOCAL, overall=0.88, ts=None, version=None):
    card = {"version": modelcard.CARD_VERSION if version is None else version,
            "key": f"{url.rstrip('/')}|{model}",
            "ts": time.time() if ts is None else ts,
            "model": model, "url": url,
            "measured": {"overall": overall},
            "choices": {"prompt_pack": "coder", "temperature": 0.1},
            "notes": [], "repeats": 3, "seconds": 1.0}
    modelcard.save_card(card)
    return card


def _cfg(model="qwen2.5-coder-14b", url=LOCAL, **kw):
    cfg = Config()
    cfg.base_url = url
    cfg.model = model
    cfg.vram_gb = 24
    cfg.sources = {}
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


# ------------------------------------------------------- D2: hosted lockout

def test_cli_calibrate_refuses_hosted_with_exit_2(capsys):
    cfg = _cfg(model="deepseek-chat", url=HOSTED, active_profile="cloud")
    args = SimpleNamespace(all=False, recalibrate=False, quick=True,
                           repeats=None)
    assert cli._cmd_calibrate(args, cfg) == 2
    err = capsys.readouterr().err
    assert REFUSAL.format(name="cloud", model="deepseek-chat") in err


def test_cli_calibrate_refusal_names_default_without_profile(capsys):
    cfg = _cfg(model="deepseek-chat", url=HOSTED)
    args = SimpleNamespace(all=False, recalibrate=False, quick=True,
                           repeats=None)
    assert cli._cmd_calibrate(args, cfg) == 2
    assert REFUSAL.format(name="(default)", model="deepseek-chat") \
        in capsys.readouterr().err


def test_hosted_predicate_is_endpoint_based_not_tier():
    # a local GGUF with a hosted-sounding name (and tier) stays local
    cfg = _cfg(model="deepseek-coder-6.7b", url=LOCAL, active_profile="box")
    cfg.profiles = {"box": {"model": "deepseek-coder-6.7b", "tier": "hosted"}}
    assert roles.profile_is_hosted(cfg, "box") is False
    cfg.profiles["cloud"] = {"model": "deepseek-chat", "base_url": HOSTED}
    assert roles.profile_is_hosted(cfg, "cloud") is True


def _write_cfg(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


MIXED_CFG = '''\
vram_gb = 24
auto_fit = false
base_url = "http://127.0.0.1:11434/v1"
model = "qwen2.5-coder-14b"

[profiles.localcoder]
base_url = "http://127.0.0.1:11434/v1"
model = "qwen2.5-coder-14b"

[profiles.localgemma]
base_url = "http://127.0.0.1:11434/v1"
model = "gemma-3-12b-it"

[profiles.cloud]
base_url = "https://api.deepseek.com/v1"
model = "deepseek-chat"
'''


class _FakeClient:
    """Stands in for LLMClient: no network, records its config."""

    def __init__(self, cfg):
        self.cfg = cfg

    def check(self, warn=True):
        pass

    def warmup(self):
        pass


def _patch_calibration(monkeypatch, calls):
    def fake_calibrate(client, cfg, verbose=True, quick=False, repeats=None):
        calls.append(cfg.model)
        return {"version": modelcard.CARD_VERSION,
                "key": modelcard.card_key(cfg), "ts": time.time(),
                "model": cfg.model, "url": cfg.base_url,
                "measured": {"overall": 0.9}, "choices": {}, "notes": [],
                "repeats": 1, "seconds": 0.1}

    monkeypatch.setattr(cli, "LLMClient", _FakeClient)
    monkeypatch.setattr(jobs, "LLMClient", _FakeClient)
    monkeypatch.setattr(modelcard, "calibrate", fake_calibrate)
    monkeypatch.setattr(llm_mod, "unload_model", lambda client: "")


def test_calibrate_all_skips_hosted_and_saves_each_card(tmp_path, monkeypatch,
                                                        capsys):
    cfgfile = tmp_path / "config.toml"
    _write_cfg(cfgfile, MIXED_CFG)
    calls: list[str] = []
    _patch_calibration(monkeypatch, calls)
    rc = cli.main(["calibrate", "--all", "-y", "--config", str(cfgfile)])
    assert rc == 0
    assert calls == ["qwen2.5-coder-14b", "gemma-3-12b-it"]
    assert "hosted" in capsys.readouterr().err.lower()   # the skip note
    saved = json.loads((config_mod.USER_CONFIG.parent / "model_cards.json")
                       .read_text(encoding="utf-8"))
    assert len(saved) == 2                   # one card per local profile
    assert not any("deepseek-chat" in k for k in saved)


def test_calibrate_all_skips_fresh_cards_unless_recalibrate(tmp_path,
                                                            monkeypatch):
    cfgfile = tmp_path / "config.toml"
    _write_cfg(cfgfile, MIXED_CFG)
    _card("qwen2.5-coder-14b")               # fresh card for the first profile
    calls: list[str] = []
    _patch_calibration(monkeypatch, calls)
    assert cli.main(["calibrate", "--all", "-y",
                     "--config", str(cfgfile)]) == 0
    assert calls == ["gemma-3-12b-it"]
    calls.clear()
    assert cli.main(["calibrate", "--all", "-y", "--recalibrate",
                     "--config", str(cfgfile)]) == 0
    assert calls == ["qwen2.5-coder-14b", "gemma-3-12b-it"]


def test_calibrate_all_confirmation_prompt_exact(tmp_path, monkeypatch,
                                                 capsys):
    cfgfile = tmp_path / "config.toml"
    _write_cfg(cfgfile, MIXED_CFG)
    calls: list[str] = []
    _patch_calibration(monkeypatch, calls)
    prompts: list[str] = []

    class Tty:
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdin", Tty())
    monkeypatch.setattr("builtins.input",
                        lambda prompt="": prompts.append(prompt) or "n")
    assert cli.main(["calibrate", "--all", "--config", str(cfgfile)]) == 0
    assert calls == []                        # declined: nothing ran
    assert prompts == [
        "Calibrate 2 local model profile(s) (localcoder, localgemma)? "
        "This loads each model in turn and may take several minutes. [y/N] "]
    # a 'y' proceeds
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    assert cli.main(["calibrate", "--all", "--config", str(cfgfile)]) == 0
    assert calls == ["qwen2.5-coder-14b", "gemma-3-12b-it"]


def test_calibrate_all_unloads_between_models(tmp_path, monkeypatch):
    cfgfile = tmp_path / "config.toml"
    _write_cfg(cfgfile, MIXED_CFG)
    calls: list[str] = []
    _patch_calibration(monkeypatch, calls)
    unloaded: list[str] = []
    monkeypatch.setattr(llm_mod, "unload_model",
                        lambda client: unloaded.append(client.cfg.model) or "")
    assert cli.main(["calibrate", "--all", "-y",
                     "--config", str(cfgfile)]) == 0
    assert unloaded == ["qwen2.5-coder-14b", "gemma-3-12b-it"]


def test_gui_calibrate_refuses_hosted(gui):
    _write_cfg(gui.cfg_path, MIXED_CFG)
    for path in ("/api/models/calibrate", "/api/calibrate"):
        status, j = gui.post(path, {"profile": "cloud", "mode": "quick"})
        assert status == 400 and j["ok"] is False
        assert j["detail"] == REFUSAL.format(name="cloud",
                                             model="deepseek-chat")
    # a local profile still starts the job
    status, j = gui.post("/api/calibrate", {"profile": "localcoder",
                                            "mode": "quick"})
    assert status in (202, 409)
    gui.wait_idle(30)


def test_gui_settings_payload_marks_hosted_and_calibrate_all_excludes(gui,
                                                                      monkeypatch):
    _write_cfg(gui.cfg_path, MIXED_CFG)
    _card("qwen2.5-coder-14b")
    status, j = gui.get("/api/settings")
    assert status == 200
    cal = j["calibration"]
    assert cal["cloud"]["is_hosted"] is True
    assert cal["localcoder"]["is_hosted"] is False
    assert cal["localcoder"]["calibrated"] is True
    assert cal["localgemma"]["calibrated"] is False
    # the batch job walks the local profiles only
    calls: list[str] = []
    _patch_calibration(monkeypatch, calls)
    status, j = gui.post("/api/calibrate/all", {"recalibrate": False})
    assert status == 202 and j["ok"]
    assert gui.wait_idle(30)
    assert calls == ["gemma-3-12b-it"]       # localcoder has a fresh card


def test_models_scan_never_calibrates_hosted(tmp_path, monkeypatch, capsys):
    from pedacito import discovery
    monkeypatch.setattr(discovery, "DEFAULT_PORTS", ())
    monkeypatch.setattr(cli, "USER_CONFIG", tmp_path / "no-config.toml")
    # The hosted catalogue is stubbed (no network): one model on a provider
    # endpoint. The endpoint -- not the tier -- decides it never calibrates.
    monkeypatch.setattr(cli, "_scan_hosted", lambda args, urls: [
        {"id": "deepseek-chat", "base_url": HOSTED, "tier": "hosted",
         "server_label": "api.deepseek.com"}])
    calibrated: list[str] = []
    monkeypatch.setattr(cli, "_scan_calibrate",
                        lambda models, quick: calibrated.extend(
                            m["id"] for m in models))
    cfgfile = tmp_path / "config.toml"
    rc = cli.main(["models", "scan", "--yes", "--calibrate", "quick",
                   "--url", HOSTED, "--api-key-env", "FAKE_KEY",
                   "--match", "deepseek*", "--config", str(cfgfile)])
    assert rc == 0
    assert calibrated == []                  # hosted: never calibrated
    assert "hosted" in capsys.readouterr().err.lower()
    assert "[profiles.deepseek-chat]" in cfgfile.read_text(encoding="utf-8")


# --------------------------------------------- D3: architecture library

@pytest.mark.parametrize("model,family", [
    ("Qwen2.5-Coder-32B-Instruct", "qwen2.5-coder"),
    ("qwen2_5_coder_14b", "qwen2.5-coder"),
    ("Qwen3-Coder-30B", "qwen2.5-coder"),
    ("Qwen3-8B", "qwen3"),
    ("qwen3.5-32b", "qwen3"),
    ("deepseek-coder-v2-lite", "deepseek-coder"),
    ("DeepSeek-R1-Distill-Qwen-14B", "deepseek-r1"),
    ("QwQ-32B", "deepseek-r1"),
    ("nemotron-4-340b", "nemotron"),
    ("gemma-3-27b-it", "gemma"),
    ("gemma-4-12b", "gemma"),
    ("codestral-22b", "mistral"),
    ("devstral-small", "mistral"),
    ("Mistral-7B-Instruct-v0.3", "mistral"),
    ("granite-3.3-8b-instruct", "granite"),
    ("phi-4-mini", "phi-4"),
    ("Llama-3.1-8B-Instruct", "llama-3"),
    ("mystery-model-7b", None),
])
def test_match_arch_families(model, family):
    entry = arch_defaults.match_arch(model)
    if family is None:
        assert entry is None
    else:
        assert entry is not None and entry["family"] == family


def test_arch_library_values():
    lib = {e["family"]: e for e in arch_defaults.ARCH_LIBRARY}
    q = lib["qwen2.5-coder"]
    assert (q["temperature"], q["temperature_answer"]) == (0.1, 0.0)
    assert q["sampler"] == {"top_p": 0.8, "top_k": 20, "repeat_penalty": 1.05}
    assert q["context_tokens"] == 32768
    assert set(q["specialties"]) == {"coding", "testing_qa"}
    r1 = lib["deepseek-r1"]
    assert (r1["temperature"], r1["temperature_answer"]) == (0.5, 0.1)
    assert r1["specialties"] == ("math_algorithms",)
    g = lib["gemma"]
    assert g["sampler"] == {"top_p": 0.95, "top_k": 64}
    assert set(g["specialties"]) == {"general", "frontend_ui",
                                     "3d_graphics_svg", "translation_i18n"}
    assert lib["granite"]["context_tokens"] == 16384
    assert lib["phi-4"]["context_tokens"] == 16384
    assert lib["mistral"]["temperature"] == 0.15
    assert lib["nemotron"]["specialties"] == ("general",)
    assert lib["llama-3"]["context_tokens"] == 32768


def test_arch_context_capped_by_server_and_vram():
    entry = arch_defaults.match_arch("qwen2.5-coder-32b")
    # the server's reported window caps the recommendation
    assert arch_defaults.arch_context_tokens(entry, {"context_length": 8192}) \
        == 8192
    # no signal: the recommendation stands
    assert arch_defaults.arch_context_tokens(entry, {}) == 32768
    # weights that fill the card leave little KV room: the estimate caps it
    small = arch_defaults.arch_context_tokens(entry, {}, vram_gb=24,
                                              size_bytes=20 * 1024 ** 3)
    assert 1024 < small < 32768
    # the floor keeps a usable window even when barely anything fits
    tiny = arch_defaults.arch_context_tokens(entry, {}, vram_gb=21,
                                             size_bytes=20 * 1024 ** 3)
    assert tiny == 1024
    # weights that do NOT fit report 0 (keep fallback budgets): no cap
    assert arch_defaults.arch_context_tokens(entry, {}, vram_gb=4,
                                             size_bytes=20 * 1024 ** 3) \
        == 32768


def test_needle_step_down_ladder():
    assert arch_defaults.needle_step_down(32768) == 16384
    assert arch_defaults.needle_step_down(16384) == 8192
    assert arch_defaults.needle_step_down(8192) is None


class _ArchProbeClient:
    """Answers the specialty probes; the needle misses above 16k."""

    chars_per_token = 3.2
    stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
             "cached_tokens": 0, "cache_write_tokens": 0,
             "seconds": 0.0, "cost": 0.0, "by_site": {}}

    def __init__(self, needle_ok_window=16384):
        self.needle_ok_window = needle_ok_window
        self.needle_windows: list[int] = []

    def chat_json(self, messages, schema, max_tokens, **kw):
        return {"reasoning": "r", "action": "done"}

    def chat(self, messages, max_tokens, **kw):
        text = messages[-1]["content"]
        if "SVG image" in text:
            return ('<svg viewBox="0 0 200 100">'
                    '<circle id="wheel" cx="50" cy="50" r="20"/></svg>')
        if "Compute" in text:
            return "29"
        if "Translate" in text:
            return "Hola {name}, tu archivo tiene {count} errores."
        if "SQLite SELECT" in text:
            return ("SELECT u.name, SUM(o.amount) FROM users u JOIN orders o "
                    "ON o.user_id = u.id GROUP BY u.name HAVING SUM(o.amount) "
                    "> 100 ORDER BY u.name;")
        if "avg(values)" in text:
            return '{"empty_raises": "ZeroDivisionError"}'
        if "CLI calculator" in text:
            return '["Implement the parser", "Wire up the CLI", "Add tests"]'
        if "needle" in text.lower():
            self.needle_windows.append(kw.get("_window", 0))
            return "FOUND" if max_tokens >= 0 else ""
        return "ok"

    def model_info(self):
        return {"context_length": 65536}


def test_calibrate_records_arch_choices_local_only(home):
    cfg = _cfg(model="Qwen2.5-Coder-14B")
    client = _ArchProbeClient()
    card = modelcard.calibrate(client, cfg, verbose=False, quick=True)
    ch = card["choices"]
    assert ch["arch_family"] == "qwen2.5-coder"
    assert ch["temperature"] == 0.1
    assert ch["temperature_answer"] == 0.0
    assert ch["sampler"] == {"top_p": 0.8, "top_k": 20, "repeat_penalty": 1.05}
    assert ch["context_tokens"] == 32768
    assert "translation_i18n" in ch["specialties"]
    # the scripted client cannot edit: the edit-skill gate (item 15) strips
    # the architecture's coding/testing_qa recommendations again
    assert "coding" not in ch["specialties"]
    assert "testing_qa" not in ch["specialties"]
    # hosted models keep provider-table defaults: no arch recommendation
    hosted_cfg = _cfg(model="Qwen2.5-Coder-14B", url=HOSTED)
    hcard = modelcard.calibrate(_ArchProbeClient(), hosted_cfg,
                                verbose=False, quick=True)
    assert "arch_family" not in hcard["choices"]
    assert "temperature" not in hcard["choices"]


def test_apply_fills_arch_settings_only_when_unset(home):
    cfg = _cfg(model="Qwen2.5-Coder-14B")
    card = modelcard.calibrate(_ArchProbeClient(), cfg, verbose=False,
                               quick=True)
    modelcard.save_card(card)

    fresh = _cfg(model="Qwen2.5-Coder-14B")
    modelcard.apply(fresh)
    assert fresh.temperature == 0.1
    assert fresh.temperature_answer == 0.0
    assert fresh.sampler == {"top_p": 0.8, "top_k": 20, "repeat_penalty": 1.05}
    assert fresh.context_tokens == 32768
    assert fresh.sources["temperature"] == "model card"
    assert fresh.sources["context_tokens"] == "model card"

    # explicit user values always win
    tuned = _cfg(model="Qwen2.5-Coder-14B")
    tuned.temperature = 0.7
    tuned.sources["temperature"] = "user config"
    tuned.temperature_answer = 0.4
    tuned.sources["temperature_answer"] = "user config"
    tuned.sampler = {"top_p": 0.5}
    tuned.sources["sampler"] = "user config"
    tuned.context_tokens = 4096
    tuned.sources["context_tokens"] = "user config"
    modelcard.apply(tuned)
    assert tuned.temperature == 0.7
    assert tuned.temperature_answer == 0.4
    assert tuned.sampler == {"top_p": 0.5}
    assert tuned.context_tokens == 4096

    # an auto-fit window is a machine fallback, not a user choice: it yields
    autofit = _cfg(model="Qwen2.5-Coder-14B")
    autofit.context_tokens = 8192
    autofit.sources["context_tokens"] = "auto-fit (VRAM estimate)"
    modelcard.apply(autofit)
    assert autofit.context_tokens == 32768


# ------------------------------------------------- D4: the three new probes

GOOD_SQL = ("SELECT u.name, SUM(o.amount) AS total FROM users u "
            "JOIN orders o ON o.user_id = u.id GROUP BY u.name "
            "HAVING SUM(o.amount) > 100 ORDER BY u.name")


def test_sql_probe_grades_by_execution():
    assert modelcard._grade_sql(GOOD_SQL) is True
    # missing the >100 filter returns bo and di too
    assert modelcard._grade_sql(
        "SELECT u.name, SUM(o.amount) FROM users u JOIN orders o "
        "ON o.user_id = u.id GROUP BY u.name ORDER BY u.name") is False
    assert modelcard._grade_sql("DROP TABLE users") is False
    assert modelcard._grade_sql("") is False
    # fences and a trailing semicolon/prose are stripped
    fenced = "```sql\n" + GOOD_SQL + ";\n```\nHope that helps!"
    assert modelcard._extract_sql(fenced) == GOOD_SQL


def test_sql_probe_scores_one(home):
    class SqlClient:
        def chat(self, messages, max_tokens, **kw):
            assert kw.get("disable_thinking") is True
            return GOOD_SQL + ";"

    score, notes = modelcard._t_sql_data(SqlClient(), Config())
    assert score == 1.0 and notes == []

    class BadClient:
        def chat(self, messages, max_tokens, **kw):
            return "SELECT name FROM users"

    score, notes = modelcard._t_sql_data(BadClient(), Config())
    assert score == 0.0 and notes


def test_testing_qa_probe_pins_zero_division():
    class Good:
        def chat(self, messages, max_tokens, **kw):
            return '{"empty_raises": "ZeroDivisionError"}'

    assert modelcard._t_testing_qa(Good(), Config()) == (1.0, [])

    class Wrong:
        def chat(self, messages, max_tokens, **kw):
            return '{"empty_raises": "ValueError"}'

    score, notes = modelcard._t_testing_qa(Wrong(), Config())
    assert score == 0.0 and notes

    class Prose:
        def chat(self, messages, max_tokens, **kw):
            return "It raises ZeroDivisionError."   # not the JSON object

    assert modelcard._t_testing_qa(Prose(), Config())[0] == 0.0


def test_orchestrator_plan_probe_orders_parser_cli_tests():
    class Good:
        def chat(self, messages, max_tokens, **kw):
            return '["Build the parser", "Wire the CLI", "Write the tests"]'

    assert modelcard._t_orchestrator_plan(Good(), Config()) == (1.0, [])

    class WrongOrder:
        def chat(self, messages, max_tokens, **kw):
            return '["Write tests", "Build parser", "Wire CLI"]'

    assert modelcard._t_orchestrator_plan(WrongOrder(), Config())[0] == 0.0

    class TwoSteps:
        def chat(self, messages, max_tokens, **kw):
            return '["parser and cli", "tests"]'

    assert modelcard._t_orchestrator_plan(TwoSteps(), Config())[0] == 0.0


def test_new_probe_tags_join_specialties():
    f = modelcard.specialty_tags_from_measurements
    # probe tags join when the edit-skill gate passes (item 15)
    tags = f("mystery-7b", {"sql_data": 1.0, "testing_qa": 1.0,
                            "orchestrator_plan": 1.0,
                            "edit_format": 0.9, "tool_choice": 0.9})
    assert "sql_data" in tags and "testing_qa" in tags
    # the orchestrator probe scores planning, not a specialty tag
    assert "orchestrator_plan" not in tags
    # without measured edit skill the same probes earn nothing (item 15)
    assert f("mystery-7b", {"sql_data": 1.0, "testing_qa": 1.0,
                            "orchestrator_plan": 1.0}) == ["general"]
    # below threshold: nothing earned
    assert f("mystery-7b", {"sql_data": 0.5, "testing_qa": 0.0,
                            "edit_format": 0.7,
                            "tool_choice": 0.6}) == ["general"]
    # the architecture library's tags union with the probe scores...
    tags = f("granite-8b", {"sql_data": 1.0, "edit_format": 0.7,
                            "tool_choice": 0.6},
             arch_tags=("coding", "sql_data"))
    assert "sql_data" in tags and "coding" in tags
    # ...but the edit-skill gate still strips them without the skill
    tags = f("granite-8b", {"sql_data": 1.0, "edit_format": 0.4,
                            "tool_choice": 0.9},
             arch_tags=("coding", "sql_data"))
    assert "coding" not in tags and "sql_data" not in tags
    tags = f("granite-8b", {"sql_data": 1.0, "edit_format": 1.0,
                            "tool_choice": 0.9},
             arch_tags=("coding", "sql_data"))
    assert "coding" in tags and "sql_data" in tags


def test_calibrate_runs_all_six_micro_probes_once(home):
    cfg = _cfg(model="mystery-7b")
    client = _ArchProbeClient()
    card = modelcard.calibrate(client, cfg, verbose=False, quick=True)
    m = card["measured"]
    assert m["sql_data"] == 1.0
    assert m["testing_qa"] == 1.0
    assert m["orchestrator_plan"] == 1.0
    # the scripted client cannot edit: the probes ran, but the edit-skill
    # gate (item 15) keeps their editing tags off the card
    assert "sql_data" not in m["specialties"]
    assert "testing_qa" not in m["specialties"]
    assert m["specialties"] == ["translation_i18n"]


# ------------------------------------------- D1: badge metadata everywhere

def test_card_meta_fresh_stale_and_absent(home):
    fresh = _card("m1", overall=0.88)
    meta = modelcard.card_meta(fresh)
    assert meta["calibrated"] is True
    assert meta["calibrated_age_days"] == 0
    assert meta["calibrated_stale"] is False
    assert meta["card_score"] == 0.88
    assert time.strftime("%Y-%m-%d %H:%M", time.localtime(fresh["ts"])) \
        == meta["calibrated_at"]
    # older than the TTL: stale
    old = _card("m2", ts=time.time() - 40 * 86400)
    assert modelcard.card_meta(old)["calibrated_stale"] is True
    assert modelcard.card_meta(old)["calibrated_age_days"] == 40
    # an old card version is stale too
    v1 = _card("m3", version=1)
    assert modelcard.card_meta(v1)["calibrated_stale"] is True
    # absent card: the empty form
    empty = modelcard.card_meta(None)
    assert empty["calibrated"] is False and empty["calibrated_at"] == ""
    assert empty["calibrated_age_days"] == 0 and empty["card_score"] == 0.0


def test_profile_calibration_reads_raw_expired_card(home):
    cfg = _cfg(model="m1", active_profile="p")
    cfg.profiles = {"p": {"model": "m1"}}
    _card("m1", ts=time.time() - 40 * 86400)
    # load() hides the expired card, but the badge must say STALE, not absent
    assert modelcard.load(cfg) is None
    meta = modelcard.profile_calibration(cfg)
    assert meta["calibrated"] is True and meta["calibrated_stale"] is True
    assert meta["is_hosted"] is False
    cfg.base_url = HOSTED
    assert modelcard.profile_calibration(cfg)["is_hosted"] is True


def test_profiles_listing_carries_badge_fields(home):
    cfg = _cfg(model="m1", active_profile="local")
    cfg.profiles = {
        "local": {"model": "m1"},
        "cloud": {"model": "deepseek-chat", "base_url": HOSTED},
    }
    _card("m1", overall=0.75)
    rows = {r["name"]: r for r in profiles.list_profiles(cfg)}
    assert rows["local"]["calibrated"] is True
    assert rows["local"]["card_score"] == 0.75
    assert rows["local"]["calibrated_stale"] is False
    assert rows["local"]["is_hosted"] is False
    assert rows["cloud"]["calibrated"] is False
    assert rows["cloud"]["is_hosted"] is True


def test_gui_state_and_settings_carry_calibration(gui):
    _write_cfg(gui.cfg_path, MIXED_CFG)
    _card("qwen2.5-coder-14b", overall=0.88)
    # the fixture starts project-less: open the config so status() can map it
    cfg = config_mod.load(explicit=gui.cfg_path, require_hardware=False,
                          warn_budgets=False, upgrade_user_config=False)
    with core.STATE.lock:
        core.STATE.cfg = cfg
    status, st = gui.get("/api/state")
    assert status == 200
    cal = st["calibration"]
    assert cal["localcoder"]["calibrated"] is True
    assert cal["localcoder"]["card_score"] == 0.88
    assert cal["cloud"]["is_hosted"] is True
    assert cal["localgemma"]["calibrated"] is False
    status, j = gui.get("/api/settings")
    assert status == 200
    sc = j["calibration"]["localcoder"]
    assert sc["calibrated_at"] and sc["card_score"] == 0.88
    assert "specialties" in sc


# --------------------------------------------------- D6: live telemetry

def test_llm_client_telemetry_math():
    cfg = Config()
    cfg.context_tokens = 16384
    client = llm_mod.LLMClient(cfg)
    assert client.last_prompt_tokens == 0
    assert client.context_utilization_pct == 0.0
    assert client.avg_tok_per_s == 0.0
    assert client.active_model_role["role"] == "worker"

    client._account({"prompt_tokens": 6210, "completion_tokens": 185,
                     "cached_tokens": 5000}, None, 4.2, 0)
    assert client.last_prompt_tokens == 6210
    assert client.last_completion_tokens == 185
    assert client.last_elapsed_s == pytest.approx(4.2)
    assert client.last_tok_per_s == pytest.approx(185 / 4.2)
    assert client.context_utilization_pct == round(6210 / 16384 * 100, 1)

    client._account({"prompt_tokens": 1000, "completion_tokens": 100},
                    None, 2.0, 0)
    # session average spans BOTH calls: 285 tokens over 6.2 seconds
    assert client.avg_tok_per_s == pytest.approx(285 / 6.2)
    t = client.telemetry()
    assert t["session_prompt_tokens"] == 7210
    assert t["session_completion_tokens"] == 285
    assert t["context_tokens"] == 16384
    assert t["cache_hit_pct"] == round(5000 / 7210 * 100, 1)
    assert t["active_model_role"]["role"] == "worker"
    # estimates keep the needle moving when the server reports no usage
    client._account({}, None, 1.0, 500, "some output text")
    assert client.last_prompt_tokens == 500


def test_gui_state_exposes_telemetry_and_folds(gui):
    cfg = Config()
    cfg.context_tokens = 32768
    client = llm_mod.LLMClient(cfg)
    client._account({"prompt_tokens": 6420, "completion_tokens": 210},
                    None, 3.0, 0)
    with core.STATE.lock:
        core.STATE.cfg = cfg
        core.STATE.client = client
    status, st = gui.get("/api/state")
    assert status == 200
    t = st["telemetry"]
    assert t["last_prompt_tokens"] == 6420
    assert t["context_tokens"] == 32768
    assert t["context_utilization_pct"] == round(6420 / 32768 * 100, 1)
    assert t["last_tok_per_s"] == pytest.approx(70.0)
    assert st["folds"] == 0
    assert "planner_profile" in st
    # fold events count; a fresh run resets the counter
    core.STATE.note_event("fold", {})
    core.STATE.note_event("fold", {})
    assert gui.get("/api/state")[1]["folds"] == 2
    job = core.STATE.begin_job("task", "test run")
    assert job is not None
    assert gui.get("/api/state")[1]["folds"] == 0
    core.STATE.end_job(job)


def test_gui_state_specialist_overrides_last_call(gui):
    cfg = Config()
    cfg.context_tokens = 32768
    worker = llm_mod.LLMClient(cfg)
    worker._account({"prompt_tokens": 100, "completion_tokens": 10},
                    None, 1.0, 0)
    spec_cfg = Config()
    spec_cfg.context_tokens = 16384
    spec_cfg.model = "gemma-3-12b"
    spec = llm_mod.LLMClient(spec_cfg)
    spec.active_model_role = {"profile": "ui", "model": "gemma-3-12b",
                              "role": "specialist", "specialty": "frontend_ui"}
    spec._account({"prompt_tokens": 9000, "completion_tokens": 300},
                  None, 5.0, 0)
    cfg._specialist_clients = {"ui": spec}
    with core.STATE.lock:
        core.STATE.cfg = cfg
        core.STATE.client = worker
        core.STATE.specialist = {"profile": "ui", "model": "gemma-3-12b",
                                 "specialty": "frontend_ui"}
    t = gui.get("/api/state")[1]["telemetry"]
    assert t["last_prompt_tokens"] == 9000
    assert t["context_tokens"] == 16384
    assert t["active_model_role"]["profile"] == "ui"
    assert t["active_model_role"]["specialty"] == "frontend_ui"
    # session totals stay the worker ledger's (the shared-ledger rule)
    assert t["session_prompt_tokens"] == 100


def test_role_clients_carry_their_labels():
    cfg = _cfg(model="worker-model", active_profile="worker")
    cfg.profiles = {"worker": {"model": "worker-model"},
                    "planner": {"model": "big-model"}}
    cfg.roles = {"planner": "planner"}
    worker = llm_mod.LLMClient(cfg)
    assert worker.active_model_role["role"] == "worker"
    # client_for builds the dedicated client without a live server check:
    # patch check/warmup so no network is touched
    orig_check, orig_warmup = llm_mod.LLMClient.check, llm_mod.LLMClient.warmup
    llm_mod.LLMClient.check = lambda self, warn=True: {}
    llm_mod.LLMClient.warmup = lambda self: None
    try:
        planner = roles.client_for(cfg, "planner", worker)
        assert planner is not worker
        assert planner.active_model_role["role"] == "planner"
        assert planner.active_model_role["profile"] == "planner"
    finally:
        llm_mod.LLMClient.check = orig_check
        llm_mod.LLMClient.warmup = orig_warmup


# ----------------------------------------------- D7: ribbon and parity

def test_show_telemetry_defaults_true_and_coerces():
    assert Config().show_telemetry is True
    assert config_mod._coerce("show_telemetry", False) is False
    assert config_mod._coerce("show_telemetry", True) is True
    # TOML supplies real bools; a string is a config error, not a guess
    with pytest.raises(config_mod.ConfigError):
        config_mod._coerce("show_telemetry", "false")


def test_ascii_bar():
    assert cli._ascii_bar(38) == "[####------]"
    assert cli._ascii_bar(0) == "[----------]"
    assert cli._ascii_bar(100) == "[##########]"
    assert cli._ascii_bar(150) == "[##########]"    # clamped


def test_telemetry_ribbon_shape():
    cfg = Config()
    cfg.context_tokens = 16384
    client = llm_mod.LLMClient(cfg)
    client._account({"prompt_tokens": 6210, "completion_tokens": 185,
                     "cached_tokens": 5092}, None, 4.25, 0)
    ribbon = cli._telemetry_ribbon(client)
    assert ribbon.startswith("[telemetry | ctx: ")
    assert "\u2588\u2588\u2588\u2588\u2591\u2591\u2591\u2591\u2591\u2591 38%" in ribbon
    assert "(6,210/16,384)" in ribbon
    assert "turn: 6,210 in / 185 out" in ribbon
    assert "43.5 tok/s" in ribbon
    assert "cache: 82%" in ribbon
    assert ribbon.endswith("]")


def test_ribbon_suppressed_for_jsonl_rpc_and_config():
    cfg = Config()
    client = llm_mod.LLMClient(cfg)
    ref = {"cfg": cfg, "client": client}
    printed: list[str] = []
    import builtins
    real_print = builtins.print

    def spy(*a, **kw):
        if kw.get("file") is sys.stderr:
            printed.append(" ".join(str(x) for x in a))
        else:
            real_print(*a, **kw)

    import unittest.mock as mock
    with mock.patch("builtins.print", spy):
        cli._print_telemetry_ribbon(SimpleNamespace(cmd="ask"), ref)
        assert len(printed) == 1
        cfg.show_telemetry = False
        cli._print_telemetry_ribbon(SimpleNamespace(cmd="ask"), ref)
        assert len(printed) == 1
        cfg.show_telemetry = True
        cli._print_telemetry_ribbon(SimpleNamespace(cmd="ask", jsonl=True), ref)
        cli._print_telemetry_ribbon(SimpleNamespace(cmd="rpc"), ref)
        cli._print_telemetry_ribbon(SimpleNamespace(cmd="ask", rpc=True), ref)
        assert len(printed) == 1


def test_status_and_cost_slash_commands_show_telemetry(capsys):
    cfg = Config()
    cfg.context_tokens = 16384
    client = llm_mod.LLMClient(cfg)
    client._account({"prompt_tokens": 6210, "completion_tokens": 185},
                    None, 4.2, 0)
    agent = SimpleNamespace(_answer_mode=lambda: "plain")
    ref = {"cfg": cfg, "client": client, "agent": agent, "idx": None}
    args = SimpleNamespace(apply=False)
    assert cli._chat_command("/status", args, cfg, ref) is True
    out = capsys.readouterr().err
    assert "[####------]" in out
    assert "37.9%" in out
    assert "tok/s" in out
    assert cli._chat_command("/cost", args, cfg, ref) is True
    out = capsys.readouterr().err
    assert "[####------]" in out and "tok/s" in out