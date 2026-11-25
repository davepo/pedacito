"""WP-3305: one canonical local-endpoint predicate.

Python 3.11.4+ changed ``ipaddress.is_private`` so that RFC 6598 CGNAT
(``100.64.0.0/10``, Tailscale) reads as NOT private, while the IANA
documentation carve-outs (``192.0.2.0/24``, ``198.51.100.0/24``,
``203.0.113.0/24``) still read as private. Local model servers on Tailscale
addresses were therefore misclassified as hosted (``json_mode = "object"``,
hosted thinking output ceilings, no calibrated temperatures). This pins the
single predicate every caller delegates to, plus the calibration/stale-card
behaviour it drives.

Hermetic: no network, scripted clients only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import cli, modelcard, servers
from pedacito import config as config_mod
from pedacito import llm as llm_mod
from pedacito import model_catalog
from pedacito.config import Config

# URLs on RFC 6598 CGNAT/Tailscale, RFC 1918, ULA, mDNS and loopback all mean
# "the user self-hosts this server".
LOCAL_URLS = {
    "tailscale_mid": "http://100.107.251.107:11434/v1",
    "cgnat_start": "http://100.64.0.1:1234/v1",
    "cgnat_end": "http://100.127.255.254:1234/v1",
    "rfc1918_192": "http://192.168.1.50:1234/v1",
    "rfc1918_10": "http://10.0.0.5:11434/v1",
    "rfc1918_172": "http://172.16.0.9:8080/v1",
    "ula": "http://[fd00::1]:11434/v1",
    "mdns": "http://mymac.local:1234/v1",
    "loopback": "http://127.0.0.1:11434/v1",
    "localhost": "http://localhost:1234/v1",
}

# Addresses that route nowhere a user self-hosts (or are clearly hosted APIs).
REMOTE_URLS = {
    "below_cgnat": "http://100.63.255.255:1234/v1",
    "above_cgnat": "http://100.128.0.1:1234/v1",
    "google_dns": "http://8.8.8.8:1234/v1",
    "iana_carveout": "http://203.0.113.5:1234/v1",
    "iana_carveout_2": "http://192.0.2.5:1234/v1",
    "iana_carveout_3": "http://198.51.100.5:1234/v1",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
}


@pytest.mark.parametrize("url", list(LOCAL_URLS.values()))
def test_local_endpoints_are_local_everywhere(url):
    assert config_mod._endpoint_is_local(url) is True
    assert servers.is_local_url(url) is True
    assert model_catalog.is_local_base_url(url) is True
    assert cli._is_hosted_url(url) is False
    assert llm_mod._is_local_server(
        SimpleNamespace(base_url=url, provider="")) is True


@pytest.mark.parametrize("url", list(REMOTE_URLS.values()))
def test_remote_endpoints_are_remote_everywhere(url):
    assert config_mod._endpoint_is_local(url) is False
    assert servers.is_local_url(url) is False
    assert model_catalog.is_local_base_url(url) is False
    assert cli._is_hosted_url(url) is True
    assert llm_mod._is_local_server(
        SimpleNamespace(base_url=url, provider="")) is False


@pytest.mark.parametrize("label", [
    "local", "llamacpp", "llama.cpp", "lemonade", "lemond", "vllm",
    "lmstudio", "ollama", "strata", "avalon",
])
def test_explicit_local_provider_labels_force_local(label):
    url = "http://8.8.8.8:1234/v1"
    assert config_mod._endpoint_is_local(url, label) is True, label
    assert servers.is_local_url(url, label) is True, label
    assert llm_mod._is_local_server(
        SimpleNamespace(base_url=url, provider=label)) is True, label


def test_provider_labels_are_case_insensitive_and_strip_underscores():
    url = "http://8.8.8.8:1234/v1"
    assert config_mod._endpoint_is_local(url, "LEMONADE") is True
    assert config_mod._endpoint_is_local(url, "LlamaCpp") is True
    assert config_mod._endpoint_is_local(url, "LLAMA_CPP") is True
    assert config_mod._endpoint_is_local(url, "llama_cpp") is True


def test_unknown_provider_does_not_force_local():
    url = "http://8.8.8.8:1234/v1"
    assert config_mod._endpoint_is_local(url, "openai") is False
    assert config_mod._endpoint_is_local(url, "anthropic") is False


def test_load_tailscale_lemonade_stays_local(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        '[universal]\nvram_gb = 24\n\n'
        '[profiles.default]\n'
        'base_url = "http://100.107.251.107:11434/v1"\n'
        'provider = "lemonade"\n'
        'model = "qwen3-8b"\n',
        encoding="utf-8")
    cfg = config_mod.load(explicit=cfg_path, require_hardware=False,
                          warn_budgets=False, upgrade_user_config=False)
    assert cfg.base_url == "http://100.107.251.107:11434/v1"
    assert cfg.provider == "lemonade"
    assert cfg.json_mode == "schema"          # NOT downgraded to "object"
    assert cfg._hosted_server is False
    assert config_mod.is_hosted(cfg) is False


def test_load_tailscale_lemonade_skips_hosted_price_fill(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        '[profiles.default]\n'
        'base_url = "http://100.107.251.107:11434/v1"\n'
        'provider = "lemonade"\n'
        'model = "qwen3-8b"\n',
        encoding="utf-8")
    cfg = config_mod.load(explicit=cfg_path, require_hardware=False,
                          warn_budgets=False, upgrade_user_config=False)
    # Local servers keep zero prices rather than inheriting a hosted table.
    assert float(cfg.price_input_per_mtok or 0) == 0.0
    assert float(cfg.price_output_per_mtok or 0) == 0.0


def test_missing_hardware_treats_tailscale_as_local():
    cfg = Config(base_url="http://100.107.251.107:11434/v1",
                 provider="lemonade", vram_gb=0, model="qwen3-8b")
    assert config_mod.is_hosted(cfg) is False
    assert config_mod.missing_hardware(cfg) == ["vram_gb"]


# ------------------------------------------------ calibration on a Tailscale box

class _ProbeClient:
    """Answers the calibration probes; the needle misses above 16k. A Tailscale
    Lemonade server is reached only through `cfg.base_url`, so this client is
    endpoint-agnostic and simply records that it was asked."""

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


def _tailscale_cfg(model, **kw):
    cfg = Config(base_url="http://100.107.251.107:11434/v1",
                 provider="lemonade", model=model, vram_gb=24)
    cfg.sources = {}
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_calibrate_tailscale_gemma_records_arch_temperatures():
    card = modelcard.calibrate(_ProbeClient(),
                               _tailscale_cfg("google_gemma-4-26B-A4B-it-GGUF-Q4_K_L"),
                               verbose=False, quick=True)
    ch = card["choices"]
    assert ch["arch_family"] == "gemma"
    assert "temperature" in ch
    assert "temperature_answer" in ch
    assert isinstance(ch["sampler"], dict)
    assert ch["context_tokens"] > 0
    assert "best_of" in ch
    assert "fold_tail_steps" in ch


def test_calibrate_tailscale_unlisted_finetune_gets_temperatures():
    card = modelcard.calibrate(
        _ProbeClient(),
        _tailscale_cfg("DaisyRose-27B-GGUF-Q4_K_XL"),
        verbose=False, quick=True)
    ch = card["choices"]
    # Unlisted fine-tune: no named family, but still calibrated temperatures.
    assert "arch_family" not in ch
    assert ch["temperature"] == 0.3
    assert ch["temperature_answer"] == 0.3
    assert "best_of" in ch
    assert "fold_tail_steps" in ch


# ------------------------------------------------ stale-card detection

@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolated model-card storage (never the real user's cards)."""
    monkeypatch.setattr(modelcard, "_path",
                        lambda: tmp_path / "home" / "model_cards.json")
    (tmp_path / "home").mkdir()
    return tmp_path / "home"


def _card(model, url, choices=None, version=None, modality=None,
          measured=None, ts=None):
    card = {"version": modelcard.CARD_VERSION if version is None else version,
            "key": f"{url.rstrip('/')}|{model}",
            "ts": time.time() if ts is None else ts,
            "model": model, "url": url,
            "measured": dict(measured or {"overall": 0.88}),
            "choices": {} if choices is None else choices,
            "notes": [], "repeats": 3, "seconds": 1.0}
    if modality is not None:
        card["modality"] = modality
    modelcard.save_card(card)
    return card


def test_stale_card_detection_local_tailscale_missing_temperatures(home):
    _card("m", "http://100.107.251.107:11434/v1", choices={"prompt_pack": "tiny"})
    cfg = Config(base_url="http://100.107.251.107:11434/v1", model="m")
    loaded = modelcard.load(cfg)
    assert loaded is not None
    assert loaded["calibrated_stale"] is True
    assert modelcard.card_meta(loaded)["calibrated_stale"] is True
    assert "re-calibrate" in modelcard.render(loaded)


def test_stale_card_detection_hosted_card_not_stale(home):
    _card("m", "https://api.deepseek.com/v1", choices={"prompt_pack": "tiny"})
    cfg = Config(base_url="https://api.deepseek.com/v1", model="m")
    loaded = modelcard.load(cfg)
    assert loaded is not None
    assert loaded["calibrated_stale"] is False


def test_stale_card_detection_image_card_not_stale(home):
    _card("m", "http://100.107.251.107:11434/v1", choices={},
          modality="image", measured={"overall": 10.0, "image_ok": True})
    cfg = Config(base_url="http://100.107.251.107:11434/v1", model="m")
    loaded = modelcard.load(cfg)
    assert loaded is not None
    assert loaded["calibrated_stale"] is False


def test_stale_card_detection_local_card_with_temperatures_not_stale(home):
    _card("m", "http://100.107.251.107:11434/v1",
          choices={"temperature": 0.3, "temperature_answer": 0.3})
    cfg = Config(base_url="http://100.107.251.107:11434/v1", model="m")
    loaded = modelcard.load(cfg)
    assert loaded is not None
    assert loaded["calibrated_stale"] is False


def test_stale_card_detection_no_url_card_not_stale(home):
    # A synthetic minimal card without a url (or an empty one) is never marked
    # missing-temperature stale: there is no endpoint to call local.
    card = {"version": modelcard.CARD_VERSION, "ts": time.time(),
            "model": "m", "measured": {"overall": 0.88}, "choices": {}}
    assert modelcard.card_meta(card)["calibrated_stale"] is False


def test_stale_card_detection_v1_card_stale_via_version(home):
    _card("m", "http://100.107.251.107:11434/v1", choices={}, version=1)
    cfg = Config(base_url="http://100.107.251.107:11434/v1", model="m")
    loaded = modelcard.load(cfg)
    assert loaded is not None
    assert loaded["calibrated_stale"] is True
    assert loaded["_needs_recalibrate"] is True
