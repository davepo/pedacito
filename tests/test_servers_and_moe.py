"""The cross-server memory manager and the local-MoE execution profile.

No GPU, model server or real port is required: the hardware files are
fixtures, the lifecycle HTTP calls are captured with a fake ``requests.post``,
and the GUI tests reuse ``test_gui``'s isolated server fixture.
"""
# `gui` is test_gui's server fixture: pytest hands it to each test as a
# parameter, which ruff otherwise reads as a redefinition.
# ruff: noqa: F811
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pedacito import config as config_mod  # noqa: E402
from pedacito import profiler as profiler_mod  # noqa: E402
from pedacito import servers  # noqa: E402
from pedacito.config import Config  # noqa: E402
from pedacito.llm import LLMClient  # noqa: E402
from test_gui import gui  # noqa: F401  (the shared GUI server fixture)

GIB = 1024 ** 3


# ---------------------------------------------------------------- hardware ---

def test_get_hardware_memory_parses_meminfo_and_rocm_sysfs(
        tmp_path, monkeypatch):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       16777216 kB\n"
        "MemFree:         1048576 kB\n"
        "MemAvailable:    8388608 kB\n",
        encoding="utf-8")
    drm = tmp_path / "drm"
    for card, total_gib, used_gib in (("card0", 8, 2), ("card1", 16, 4)):
        device = drm / card / "device"
        device.mkdir(parents=True)
        (device / "mem_info_vram_total").write_text(
            str(total_gib * GIB), encoding="ascii")
        (device / "mem_info_vram_used").write_text(
            str(used_gib * GIB), encoding="ascii")

    monkeypatch.setattr(servers, "MEMINFO_PATH", meminfo)
    monkeypatch.setattr(servers, "DRM_ROOT", drm)
    mem = servers.get_hardware_memory()

    assert mem["ram"]["total_gb"] == 16.0
    assert mem["ram"]["used_gb"] == 8.0
    assert mem["ram"]["available_gb"] == 8.0
    assert mem["ram"]["percent"] == 50.0

    assert mem["vram"]["total_gb"] == 24.0
    assert mem["vram"]["used_gb"] == 6.0
    assert mem["vram"]["percent"] == 25.0
    assert [c["card"] for c in mem["vram"]["cards"]] == ["card0", "card1"]
    assert mem["vram"]["cards"][1]["used_gb"] == 4.0
    # Flat aliases the header badge and older clients read.
    assert mem["vram_total_gb"] == 24.0 and mem["ram_total_gb"] == 16.0
    assert servers.memory_badge(mem) == \
        "VRAM: 6.0 / 24.0 GB | RAM: 8.0 / 16.0 GB"


def test_get_hardware_memory_falls_back_when_files_are_missing(
        tmp_path, monkeypatch):
    monkeypatch.setattr(servers, "MEMINFO_PATH", tmp_path / "absent")
    monkeypatch.setattr(servers, "DRM_ROOT", tmp_path / "absent-drm")
    mem = servers.get_hardware_memory()
    assert mem["ram"]["total_gb"] == 0.0
    assert mem["vram"]["total_gb"] == 0.0
    assert mem["vram"]["cards"] == []
    assert mem["available"] is False


# --------------------------------------------------------- server lifecycle ---

class _Resp:
    status_code = 200
    text = '{"status": "success"}'

    def json(self):
        return {"status": "success"}


def _cfg_with(profiles, active=""):
    cfg = Config()
    cfg.profiles = profiles
    cfg.active_profile = active
    body = profiles.get(active) or {}
    cfg.base_url = body.get("base_url", "http://127.0.0.1:9999/v1")
    cfg.model = body.get("model", "model")
    cfg.vram_gb = 24
    return cfg


def test_unload_lemonade_and_llama_payloads(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers or {}, "json": json})
        return _Resp()

    monkeypatch.setattr(servers.requests, "post", fake_post)
    cfg = _cfg_with({
        "lemonade": {"base_url": "http://127.0.0.1:8000/v1",
                     "model": "qwen3-8b", "provider": "lemonade"},
        "llama": {"base_url": "http://127.0.0.1:8080/v1",
                  "model": "gemma-3-27b", "provider": "llama-server"},
    }, active="lemonade")

    result = servers.unload_server_model("lemonade", "qwen3-8b", cfg)
    assert result["ok"] is True
    assert calls[-1]["url"] == "http://127.0.0.1:8000/v1/unload"
    assert calls[-1]["json"] == {"model_name": "qwen3-8b"}

    result = servers.unload_server_model("llama", "gemma-3-27b", cfg)
    assert result["ok"] is True
    assert calls[-1]["url"] == "http://127.0.0.1:8080/models/unload"
    assert calls[-1]["json"] == {"model": "gemma-3-27b"}


def test_unload_uses_lemonade_api_key_from_environment(monkeypatch):
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.update(headers or {})
        return _Resp()

    monkeypatch.setattr(servers.requests, "post", fake_post)
    monkeypatch.setenv("LEMONADE_API_KEY", "secret-lemonade")
    cfg = _cfg_with({
        "lemonade": {"base_url": "http://127.0.0.1:8000/v1",
                     "model": "qwen3-8b", "provider": "lemonade"},
    }, active="lemonade")
    assert servers.unload_server_model("lemonade", "qwen3-8b", cfg)["ok"]
    assert seen["Authorization"] == "Bearer secret-lemonade"


def test_load_and_strata_hook_payload(monkeypatch, tmp_path):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append((url, json))
        return _Resp()

    monkeypatch.setattr(servers.requests, "post", fake_post)
    cfg = _cfg_with({
        "strata": {"base_url": "http://127.0.0.1:8001/v1",
                   "model": "qwen3.8-flash", "provider": "strata"},
    }, active="strata")
    result = servers.load_server_model("strata", "qwen3.8-flash", cfg)
    assert result["ok"] is True
    assert calls[-1] == ("http://127.0.0.1:8001/v1/load",
                         {"model": "qwen3.8-flash"})

    # HTTP refused: the configured unload_cmd is the fallback, run without a
    # shell and with {model} substituted.
    def no_http(url, headers=None, json=None, timeout=None):
        return SimpleNamespace(status_code=404, text="not found")

    marker = tmp_path / "unloaded.txt"
    cfg.profiles["strata"]["unload_cmd"] = (
        f"{sys.executable} -c \"open(r'{marker}', 'w').write('{{model}}')\"")
    monkeypatch.setattr(servers.requests, "post", no_http)
    result = servers.unload_server_model("strata", "qwen3.8-flash", cfg)
    assert result["ok"] is True and result["method"] == "command unload_cmd"
    assert marker.read_text(encoding="utf-8") == "qwen3.8-flash"


_HOOK_ENV_DUMP = (
    "import json, os, sys\n"
    "with open(sys.argv[1], 'w', encoding='utf-8') as fh:\n"
    "    json.dump(dict(os.environ), fh)\n"
)


def _dump_hook_env(env_file, cfg, *, server=None, model="qwen3.8-flash"):
    server = server or {"name": "strata",
                        "base_url": "http://127.0.0.1:8001/v1"}
    ok, detail = servers._run_hook(
        [sys.executable, "-c", _HOOK_ENV_DUMP, str(env_file)],
        server, model, cfg)
    assert ok, detail
    return json.loads(env_file.read_text(encoding="utf-8"))


def test_lifecycle_hook_env_names_cannot_alias_config_overrides():
    """PEDACITO_<KEY> overrides config; the two hook names must not match a
    key (PEDACITO_MODEL would have changed the child's own profile)."""
    assert "server_name" not in config_mod._FIELDS
    assert "model_path" not in config_mod._FIELDS


def test_lifecycle_hook_env_is_scrubbed_and_uses_documented_names(
        monkeypatch, tmp_path):
    ambient = ("OPENAI_API_KEY", "GITHUB_TOKEN", "AWS_SECRET_ACCESS_KEY",
               "DB_PASSWORD", "PEDICITO_SERVER", "PEDICITO_MODEL")
    for name in ambient:
        monkeypatch.setenv(name, "ambient-should-not-leak")

    env = _dump_hook_env(tmp_path / "hook-env.json", Config())

    for name in ambient:
        assert name not in env, name
    assert not any(
        re.search(r"(?i)(?:api_?key|token|secret|passw|credential|access_?key)",
                  key) for key in env)
    assert env["PEDACITO_SERVER_NAME"] == "strata"
    assert env["PEDACITO_MODEL_PATH"] == "qwen3.8-flash"
    assert not any(key.startswith("PEDICITO_") for key in env)
    assert "PATH" in env


def test_lifecycle_hook_env_policy_scrub_off_and_passthrough(
        monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ambient")
    monkeypatch.setenv("DATABASE_URL", "postgres://user:pw@db/app")

    unscubbed = Config()
    unscubbed.scrub_env = False
    env = _dump_hook_env(tmp_path / "hook-env-off.json", unscubbed)
    assert env["OPENAI_API_KEY"] == "sk-ambient"

    passthrough = Config()
    passthrough.env_passthrough = ["DATABASE_URL"]
    env = _dump_hook_env(tmp_path / "hook-env-pass.json", passthrough)
    assert env["DATABASE_URL"] == "postgres://user:pw@db/app"
    assert "OPENAI_API_KEY" not in env


def test_unload_hook_uses_the_caller_config_env_policy(monkeypatch, tmp_path):
    """`_server_action` threads the caller's cfg into `_run_hook`, so
    `scrub_env = false` is honoured on the public unload path too."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ambient")
    env_file = tmp_path / "public-hook-env.json"
    dumper = tmp_path / "dump_env.py"
    dumper.write_text(_HOOK_ENV_DUMP, encoding="utf-8")
    cfg = _cfg_with({
        "strata": {"base_url": "http://127.0.0.1:8001/v1",
                   "model": "qwen3.8-flash", "provider": "strata"},
    }, active="strata")
    cfg.scrub_env = False
    cfg.profiles["strata"]["unload_cmd"] = (
        f'"{sys.executable}" "{dumper}" "{env_file}"')

    monkeypatch.setattr(
        servers.requests, "post",
        lambda *a, **k: SimpleNamespace(status_code=404, text="not found"))
    result = servers.unload_server_model("strata", "qwen3.8-flash", cfg)

    assert result["ok"] is True and result["method"] == "command unload_cmd"
    env = json.loads(env_file.read_text(encoding="utf-8"))
    assert env["OPENAI_API_KEY"] == "sk-ambient"
    assert env["PEDACITO_SERVER_NAME"] == "strata"
    assert env["PEDACITO_MODEL_PATH"] == "qwen3.8-flash"


def test_evict_other_local_servers_unloads_idle_local_server(monkeypatch):
    target = {"name": "strata", "kind": "strata", "provider": "strata",
              "base_url": "http://127.0.0.1:8001/v1", "online": True,
              "models": ["qwen3.8-flash"], "loaded_models": ["qwen3.8-flash"],
              "active": True}
    other = {"name": "lemonade", "kind": "lemonade", "provider": "lemonade",
             "base_url": "http://127.0.0.1:8000/v1", "online": True,
             "models": ["gemma-3-27b"], "loaded_models": ["gemma-3-27b"],
             "active": False}
    monkeypatch.setattr(servers, "list_server_states",
                        lambda cfg, timeout=servers.PROBE_TIMEOUT:
                        [target, other])
    monkeypatch.setattr(servers, "get_hardware_memory",
                        lambda *a, **k: {"ram": {"used_gb": 1.0, "total_gb": 2.0},
                                         "vram": {"used_gb": 1.0, "total_gb": 2.0}})
    unloaded = []

    def fake_unload(state, model="", config=None):
        unloaded.append((state.get("name"), model))
        return {"ok": True, "server": state.get("name"),
                "action": "unload", "detail": "done"}

    monkeypatch.setattr(servers, "unload_server_model", fake_unload)
    cfg = _cfg_with({}, active="")
    cfg.auto_evict_servers = True

    result = servers.evict_other_local_servers("strata", cfg, wait_seconds=0)
    assert result["ok"] is True
    assert unloaded == [("lemonade", "")]
    assert [r["server"] for r in result["evicted"]] == ["lemonade"]

    # The switch is off: nothing is touched.
    unloaded.clear()
    cfg.auto_evict_servers = False
    result = servers.evict_other_local_servers("strata", cfg, wait_seconds=0)
    assert result.get("disabled") is True and unloaded == []


def test_list_server_states_detects_resident_models(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        if url.endswith("/v1/models"):
            return SimpleNamespace(status_code=200, json=lambda: {"data": [
                {"id": "qwen3.8-flash", "loaded_in_memory": True},
                {"id": "gemma-3-27b", "loaded": False},
            ]})
        return SimpleNamespace(status_code=404, json=lambda: {})

    monkeypatch.setattr(servers.requests, "get", fake_get)
    cfg = _cfg_with({
        "strata": {"base_url": "http://127.0.0.1:8001/v1",
                   "model": "qwen3.8-flash", "provider": "strata"},
    }, active="strata")
    states = servers.list_server_states(cfg)
    assert len(states) == 1
    state = states[0]
    assert state["online"] is True and state["kind"] == "strata"
    assert state["models"] == ["qwen3.8-flash", "gemma-3-27b"]
    assert state["loaded_models"] == ["qwen3.8-flash"]


# ---------------------------------------------------------- local-MoE profile ---

@pytest.mark.parametrize("model", [
    "qwen3.8-flash", "deepseek-v4.1-flash", "qwen3-35b-a3b",
    "Qwen3_8_Flash", "some-moe-model",
])
def test_local_moe_detection_from_model_patterns(model):
    cfg = Config(base_url="http://localhost:8000/v1", model=model)
    assert config_mod.is_local_moe(cfg) is True
    assert config_mod.capability_profile(cfg) == "local-moe"


@pytest.mark.parametrize("provider", ["strata", "avalon", "local-moe"])
def test_local_moe_detection_from_provider(provider):
    cfg = Config(base_url="http://localhost:8000/v1", model="whatever",
                 provider=provider)
    assert config_mod.is_local_moe(cfg) is True
    assert config_mod.capability_profile(cfg) == "local-moe"


def test_local_moe_does_not_claim_remote_or_dense_models():
    remote = Config(base_url="https://api.deepseek.com/v1",
                    model="deepseek-v4.1-flash")
    assert config_mod.is_local_moe(remote) is False
    assert config_mod.capability_profile(remote) == "hosted"
    dense = Config(base_url="http://localhost:1234/v1", model="gemma-3-27b")
    assert config_mod.capability_profile(dense) == "local-dense"


def test_local_moe_defaults_budget_timeout_and_single_stream():
    cfg = Config(base_url="http://localhost:8000/v1",
                 model="qwen3.8-flash", provider="strata")
    client = LLMClient(cfg)
    assert client.local_moe is True
    assert cfg.max_tokens_step >= 8192
    assert cfg.max_tokens_answer >= 8192
    assert cfg.max_tokens_plan >= 16384
    assert cfg.timeout >= 600
    assert cfg.max_concurrency == 1
    assert config_mod.local_moe_output_budget() == 8192
    assert config_mod.local_moe_output_budget(complex_plan=True) == 16384


def test_profile_load_auto_selects_local_moe_tier(tmp_path, monkeypatch):
    cfg_path = tmp_path / "user-config.toml"
    cfg_path.write_text(
        'default_profile = "strata"\n'
        '[profiles.strata]\n'
        'provider = "strata"\n'
        'base_url = "http://127.0.0.1:8001/v1"\n'
        'model = "qwen3.8-flash"\n'
        'vram_gb = 24\n',
        encoding="utf-8")
    monkeypatch.setattr(config_mod, "USER_CONFIG", cfg_path)
    cfg = config_mod.load(require_hardware=False, warn_budgets=False)
    assert cfg.active_tier == "local-moe"
    assert cfg.max_tokens_step >= 8192
    assert cfg.timeout >= 600
    assert cfg.max_concurrency == 1


def test_local_moe_forwards_reasoning_effort_and_keeps_sampling(monkeypatch):
    payloads = []

    class Reply:
        status_code = 200
        text = "{}"

        def json(self):
            return {"choices": [{"message": {"content": "ok"},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                              "total_tokens": 2}}

    def fake_post(url, headers=None, timeout=None, json=None, **kw):
        payloads.append(dict(json or {}))
        return Reply()

    monkeypatch.setattr("pedacito.llm.requests.post", fake_post)
    cfg = Config(base_url="http://localhost:8000/v1",
                 model="deepseek-v4.1-flash", provider="avalon",
                 temperature=0.7, reasoning_effort="high")
    client = LLMClient(cfg)
    client.chat([{"role": "user", "content": "hello"}], max_tokens=64)
    assert payloads
    body = payloads[-1]
    assert body["reasoning_effort"] == "high"
    assert body["temperature"] == 0.7          # never clamped to 0.1
    assert body["max_tokens"] == 64

    # "default" omits the field entirely.
    payloads.clear()
    cfg.reasoning_effort = "default"
    client.chat([{"role": "user", "content": "hello"}], max_tokens=64)
    assert "reasoning_effort" not in payloads[-1]


def test_default_effort_is_not_sent_to_a_dense_local_model(monkeypatch):
    payloads = []

    class Reply:
        status_code = 200
        text = "{}"

        def json(self):
            return {"choices": [{"message": {"content": "ok"},
                                 "finish_reason": "stop"}]}

    monkeypatch.setattr("pedacito.llm.requests.post",
                        lambda url, **kw: (payloads.append(kw.get("json") or {}),
                                           Reply())[1])
    cfg = Config(base_url="http://localhost:1234/v1", model="gemma-3-27b",
                 reasoning_effort="high")
    LLMClient(cfg).chat([{"role": "user", "content": "hi"}], max_tokens=8)
    assert "reasoning_effort" not in payloads[-1]


# ----------------------------------------------------------------- GUI API ---

def _gui_state():
    return {
        "name": "strata", "kind": "strata", "provider": "strata",
        "base_url": "http://127.0.0.1:8001/v1", "online": True,
        "models": ["qwen3.8-flash"], "loaded_models": ["qwen3.8-flash"],
        "active": True,
    }


def _fake_memory():
    return {
        "ram": {"total_gb": 64.0, "used_gb": 32.0, "percent": 50.0},
        "vram": {"total_gb": 24.0, "used_gb": 20.0, "percent": 83.3,
                 "cards": []},
        "ram_total_gb": 64.0, "ram_used_gb": 32.0,
        "vram_total_gb": 24.0, "vram_used_gb": 20.0,
        "available": True,
    }


def test_gui_servers_endpoint_returns_memory_and_states(gui, monkeypatch):
    monkeypatch.setattr(profiler_mod, "snapshot",
                        lambda *a, **k: _fake_memory())
    monkeypatch.setattr(servers, "list_server_states",
                        lambda *a, **k: [_gui_state()])
    status, j = gui.get("/api/servers")
    assert status == 200 and j["ok"] is True
    assert j["memory"]["vram"]["used_gb"] == 20.0
    assert j["badge"] == "VRAM: 20.0 / 24.0 GB | RAM: 32.0 / 64.0 GB"
    assert j["servers"][0]["name"] == "strata"
    assert j["servers"][0]["loaded_models"] == ["qwen3.8-flash"]


def test_gui_unload_endpoint_updates_telemetry(gui, monkeypatch):
    memory = _fake_memory()
    monkeypatch.setattr(profiler_mod, "snapshot",
                        lambda *a, **k: memory)
    monkeypatch.setattr(servers, "list_server_states",
                        lambda *a, **k: [{**_gui_state(),
                                          "loaded_models": []}])
    calls = []

    def fake_unload(target, model="", config=None):
        calls.append((target, model))
        return {"ok": True, "action": "unload", "server": target,
                "model": model, "detail": "done"}

    monkeypatch.setattr(servers, "unload_server_model", fake_unload)
    status, j = gui.post("/api/servers/unload",
                         {"server": "strata", "model": "qwen3.8-flash"})
    assert status == 200 and j["ok"] is True
    assert calls == [("strata", "qwen3.8-flash")]
    assert j["memory"]["vram"]["total_gb"] == 24.0
    assert j["servers"][0]["loaded_models"] == []


def test_gui_load_endpoint_returns_telemetry(gui, monkeypatch):
    monkeypatch.setattr(servers, "get_hardware_memory",
                        lambda *a, **k: _fake_memory())
    monkeypatch.setattr(servers, "list_server_states",
                        lambda *a, **k: [_gui_state()])
    calls = []

    def fake_load(target, model="", config=None):
        calls.append((target, model))
        return {"ok": True, "action": "load", "server": target,
                "model": model, "detail": "loaded"}

    monkeypatch.setattr(servers, "load_server_model", fake_load)
    status, j = gui.post("/api/servers/load",
                         {"server": "strata", "model": "qwen3.8-flash"})
    assert status == 200 and j["ok"] is True
    assert calls == [("strata", "qwen3.8-flash")]
    assert j["servers"] and j["memory"]["ram"]["total_gb"] == 64.0
