"""B7: front ends -- setup's scan option and the GUI scan/list endpoints."""
from __future__ import annotations

import io
import sys
import tomllib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import config as config_mod


def test_setup_scan_adds_a_profile_per_model(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "discover_servers", lambda timeout=2.0: [
        ("Ollama", "http://localhost:11434/v1",
         ["qwen3:14b", "llama3:8b", "tiny:3b"])])

    def ask(prompt, default=""):
        if "Add a profile for every model" in prompt:
            return "y"
        if "GPU VRAM" in prompt:
            return "24"
        return default

    monkeypatch.setattr(config_mod, "_ask", ask)
    p = tmp_path / "config.toml"
    config_mod.interactive_setup(path=p, force=True, stdout=io.StringIO())
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    # the first model is the default; the rest get their own profiles
    assert set(data["profiles"]) == {"default", "llama3-8b", "tiny-3b"}
    assert data["default_profile"] == "default"
    assert data["profiles"]["default"]["model"] == "qwen3:14b"
    # the result loads cleanly
    cfg = config_mod.load(explicit=p, use_env=False, require_hardware=False,
                          warn_budgets=False)
    assert "llama3-8b" in cfg.profiles


def test_setup_single_model_still_works(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "discover_servers", lambda timeout=2.0: [
        ("Ollama", "http://localhost:11434/v1", ["qwen3:14b", "llama3:8b"])])

    def ask(prompt, default=""):
        if "Add a profile for every model" in prompt:
            return "n"                      # decline the scan
        if "Which model" in prompt:
            return "qwen3:14b"
        if "GPU VRAM" in prompt:
            return "24"
        if "tier" in prompt.lower():
            return "medium"
        return default

    monkeypatch.setattr(config_mod, "_ask", ask)
    p = tmp_path / "config.toml"
    config_mod.interactive_setup(path=p, force=True, stdout=io.StringIO())
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    prof = data["profiles"]["default"]
    assert prof["model"] == "qwen3:14b"


def test_settings_api_list_profiles(monkeypatch):
    from pedacito.gui import core, settings_api
    cfg = config_mod.Config()
    cfg.base_url = "http://localhost:11434/v1"
    cfg.model = "m"
    cfg.vram_gb = 24
    cfg.profiles = {"qwen": {"model": "qwen3:14b",
                             "base_url": "http://localhost:11434/v1"}}
    cfg.active_profile = "qwen"
    monkeypatch.setattr(core.STATE, "cfg", cfg)
    out = settings_api.list_profiles()
    assert out["profiles"][0]["name"] == "qwen"
    assert "recommendations" in out


def test_settings_api_scan_servers(monkeypatch):
    from pedacito import discovery
    from pedacito.gui import settings_api
    monkeypatch.setattr(discovery, "discover_servers",
                        lambda timeout=1.5: [
                            ("LM Studio", "http://localhost:1234/v1", ["m"])])
    monkeypatch.setattr(discovery, "describe_models",
                        lambda base_url: [{"id": "m", "base_url": base_url,
                                           "loaded": True}])
    out = settings_api.scan_servers()
    assert out["servers"][0]["label"] == "LM Studio"
    assert out["servers"][0]["models"][0]["id"] == "m"
