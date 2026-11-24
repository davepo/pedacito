"""Tests for 3-tier model catalog enrichment, CLI/GUI refresh, and optional OS keyring."""
# ruff: noqa: F811

from __future__ import annotations

import builtins
import json
import sys
import threading
import time
import tomllib
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pedacito import cli
from pedacito import config as config_mod
from pedacito import model_catalog, providers, secrets_store, servers
from pedacito.gui import settings_api
from pedacito.llm import _api_key
from pedacito.local_server import LocalHTTPServer  # noqa: E402
from test_gui import gui  # noqa: F401  (shared GUI fixture)


class _FakeUrlopenResp:
    def __init__(self, payload, status=200):
        self._data = json.dumps(payload).encode("utf-8")
        self.status = status

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


MODELS_DEV_SAMPLE = {
    "openai": {
        "id": "openai",
        "api": "https://api.openai.com/v1",
        "models": {
            "gpt-5.4": {
                "id": "gpt-5.4",
                "limit": {"context": 400000, "output": 32768},
                "cost": {"input": 2.5, "output": 10.0, "cache_read": 0.625},
            },
            "custom-catalog-only": {
                "id": "custom-catalog-only",
                "limit": {"context": 512000, "output": 16384},
                "cost": {"input": 1.2, "output": 4.8, "cache_read": 0.3},
            },
        },
    },
    "google": {
        "id": "google",
        "models": {
            "gemini-2.5-pro": {
                "id": "gemini-2.5-pro",
                "limit": {"context": 1048576, "output": 65536},
                "cost": {"input": 1.25, "output": 10.0, "cache_read": 0.31},
            },
        },
    },
}


# --------------------------------------------------------------- model_catalog

def test_model_catalog_refresh_cache_ttl_and_offline_safety(monkeypatch):
    calls = []

    def fake_urlopen(url, timeout=3.0):
        calls.append((url, timeout))
        return _FakeUrlopenResp(MODELS_DEV_SAMPLE)

    monkeypatch.setattr(model_catalog, "urlopen", fake_urlopen)
    res = model_catalog.refresh_catalog(force=True)
    assert res["ok"] is True and res["refreshed"] is True
    assert res["count"] >= 3
    assert model_catalog.cache_path().exists()

    # Subsequent force=False within 24h TTL uses the cache without hitting network
    res2 = model_catalog.refresh_catalog(force=False)
    assert res2["ok"] is True and res2["refreshed"] is False and res2["cached"] is True
    assert len(calls) == 1

    # Lookup never touches the network when allow_fetch=False
    meta = model_catalog.lookup_model_meta(
        "https://api.openai.com/v1", "custom-catalog-only", allow_fetch=False)
    assert meta["context_tokens"] == 512000
    assert meta["output_tokens"] == 16384
    assert meta["price_input_per_mtok"] == 1.2
    assert meta["price_cached_per_mtok"] == 0.3
    assert meta["price_output_per_mtok"] == 4.8

    # Local URLs never fetch and never return catalog entries
    assert model_catalog.lookup_model_meta(
        "http://127.0.0.1:1234/v1", "gpt-5.4", allow_fetch=True) == {}
    assert len(calls) == 1

    # Network failure on refresh is a quiet no-op
    monkeypatch.setattr(
        model_catalog, "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(OSError("offline")),
    )
    err_res = model_catalog.refresh_catalog(force=True)
    assert err_res["ok"] is False and err_res["refreshed"] is False


def test_model_catalog_respects_auto_refresh_disabled(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network must not be called when disabled")

    monkeypatch.setattr(model_catalog, "urlopen", boom)
    cfg = config_mod.Config(auto_refresh_model_catalog=False)
    res = model_catalog.refresh_catalog(force=False, cfg=cfg)
    assert res["ok"] is False and res["detail"] == "disabled"
    assert model_catalog.lookup_model_meta(
        "https://api.openai.com/v1", "gpt-5.4", allow_fetch=True, cfg=cfg) == {}


def test_three_tier_default_prices_prefers_catalog_over_builtin():
    # Tier 3 built-in table covers 2026 models offline with zero cache
    p_anth = providers.default_prices("https://api.anthropic.com/v1", "claude-sonnet-4-5")
    assert p_anth["price_input_per_mtok"] == 3.0
    p_gem = providers.default_prices(
        "https://generativelanguage.googleapis.com/v1beta", "gemini-2.5-pro")
    assert p_gem["price_input_per_mtok"] == 1.25
    p_o4 = providers.default_prices("https://api.openai.com/v1", "o4-mini")
    assert p_o4["price_input_per_mtok"] == 1.10

    # Tier 2 cached catalog overrides Tier 3 when present
    model_catalog.write_cache({
        "openai|gpt-5.4": {
            "context_tokens": 400000,
            "output_tokens": 32768,
            "price_input_per_mtok": 9.99,
            "price_cached_per_mtok": 2.50,
            "price_output_per_mtok": 29.99,
        },
    }, now=time.time())
    prices, src = providers.default_prices_with_source(
        "https://api.openai.com/v1", "gpt-5.4")
    assert src == "catalog"
    assert prices["price_input_per_mtok"] == 9.99
    assert prices["price_output_per_mtok"] == 29.99


# ------------------------------------------- probe_provider / import / refresh

def _start_probe_server(payload_by_path):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.server.seen_headers.append(dict(self.headers))
            for prefix, body in payload_by_path.items():
                if self.path.startswith(prefix):
                    raw = json.dumps(body).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
            self.send_response(404)
            self.end_headers()

    srv = LocalHTTPServer(("127.0.0.1", 0), H)
    srv.seen_headers = []
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, t


def test_probe_provider_three_tier_enrichment_and_gemini_limits(gui, monkeypatch):
    # Prevent live network calls to models.dev; populate catalog cache for Tier 2
    monkeypatch.setattr(
        model_catalog, "urlopen",
        lambda *a, **k: _FakeUrlopenResp(MODELS_DEV_SAMPLE),
    )
    model_catalog.write_cache({
        "catalog-only-model": {},
        "openai|catalog-hit": {
            "context_tokens": 333000,
            "output_tokens": 8192,
            "price_input_per_mtok": 1.5,
            "price_cached_per_mtok": 0.4,
            "price_output_per_mtok": 6.0,
        },
    }, now=time.time())

    # 1. OpenRouter-style payload (Tier 1 context_length + USD/token pricing)
    #    + Groq-style context_window + Gemini inputTokenLimit
    srv, t = _start_probe_server({
        "/v1/models": {
            "data": [
                {
                    "id": "anthropic/claude-sonnet-4-5",
                    "context_length": 200000,
                    "top_provider": {"max_completion_tokens": 64000},
                    "pricing": {"prompt": "0.000003", "completion": "0.000015",
                                "input_cache_read": "0.0000003"},
                },
                {
                    "id": "groq-mixtral",
                    "context_window": 131072,
                    "max_tokens": 16384,
                },
                {
                    "id": "gpt-5.4",
                },
            ],
        },
        "/v1beta/models": {
            "models": [
                {
                    "name": "models/gemini-2.5-pro",
                    "inputTokenLimit": 1048576,
                    "outputTokenLimit": 65536,
                    "supportedGenerationMethods": ["generateContent"],
                },
                {
                    "name": "models/text-embedding-004",
                    "inputTokenLimit": 2048,
                    "supportedGenerationMethods": ["embedContent"],
                },
            ],
        },
    })
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}/v1"
        status, j = gui.post("/api/providers/probe", {
            "preset_id": "openai", "base_url": base, "api_key": "sk-test",
        })
        assert status == 200 and j["ok"] is True
        by_id = {m["id"]: m for m in j["models"]}

        # Tier 1: provider /models metadata
        anth = by_id["anthropic/claude-sonnet-4-5"]
        assert anth["context_tokens"] == 200000
        assert anth["output_tokens"] == 64000
        assert anth["price_input_per_mtok"] == 3.0
        assert anth["price_output_per_mtok"] == 15.0
        assert anth["price_cached_per_mtok"] == 0.3
        assert anth["meta_source"] == "provider"

        groq = by_id["groq-mixtral"]
        assert groq["context_tokens"] == 131072
        assert groq["output_tokens"] == 16384
        assert groq["meta_source"] == "provider"

        # Tier 3: builtin preset context_tokens + builtin model prices
        gpt5 = by_id["gpt-5.4"]
        assert gpt5["context_tokens"] == 400000
        assert gpt5["price_input_per_mtok"] == 2.50
        assert gpt5["meta_source"] == "builtin"

        # Native Gemini /v1beta/models probe reads inputTokenLimit/outputTokenLimit
        g_base = f"http://127.0.0.1:{srv.server_address[1]}/v1beta"
        status_g, j_g = gui.post("/api/providers/probe", {
            "preset_id": "gemini", "base_url": g_base, "api_key": "AIza-test",
        })
        assert status_g == 200 and j_g["ok"] is True
        g_rows = {m["id"]: m for m in j_g["models"]}
        assert list(g_rows) == ["gemini-2.5-pro"]  # embedContent filtered out
        assert g_rows["gemini-2.5-pro"]["context_tokens"] == 1048576
        assert g_rows["gemini-2.5-pro"]["output_tokens"] == 65536
        assert g_rows["gemini-2.5-pro"]["meta_source"] == "provider"
        assert srv.seen_headers[-1].get("x-goog-api-key") == "AIza-test"
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(5)


def test_import_hosted_models_uses_per_model_resolved_context_tokens(gui):
    model_catalog.write_cache({
        "openai|gpt-mini-from-catalog": {
            "context_tokens": 196608,
            "output_tokens": 16384,
            "price_input_per_mtok": 0.4,
            "price_cached_per_mtok": 0.1,
            "price_output_per_mtok": 1.6,
        },
    }, now=time.time())
    status, j = gui.post("/api/providers/import", {
        "preset_id": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-test",
        "models": [
            {"model": "gpt-5.4", "context_tokens": 1050000},
            {"model": "gpt-mini-from-catalog"},
            {"model": "unknown-fallback-model"},
        ],
    })
    assert status == 200 and j["ok"] is True
    profs = tomllib.loads(gui.cfg_path.read_text(encoding="utf-8"))["profiles"]
    assert profs["gpt-5_4"]["context_tokens"] == 1050000
    assert profs["gpt-mini-from-catalog"]["context_tokens"] == 196608
    assert profs["unknown-fallback-model"]["context_tokens"] == 400000


def test_gui_and_cli_refresh_endpoints(gui, monkeypatch, capsys):
    monkeypatch.setattr(
        model_catalog, "urlopen",
        lambda *a, **k: _FakeUrlopenResp(MODELS_DEV_SAMPLE),
    )
    status, j = gui.post("/api/providers/refresh", {})
    assert status == 200 and j["ok"] is True and j["refreshed"] is True

    rc = cli.main(["models", "refresh", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True and out["refreshed"] is True


# --------------------------------------------------------------- OS keyring

def _install_fake_keyring(monkeypatch, initial: dict | None = None):
    store = dict(initial or {})

    class PasswordDeleteError(Exception):
        pass

    fake = SimpleNamespace(
        errors=SimpleNamespace(PasswordDeleteError=PasswordDeleteError),
        get_password=lambda svc, acc: store.get((svc, acc)),
        set_password=lambda svc, acc, val: store.__setitem__((svc, acc), val),
        delete_password=lambda svc, acc: (
            store.pop((svc, acc)) if (svc, acc) in store
            else (_ for _ in ()).throw(PasswordDeleteError("missing"))
        ),
    )
    monkeypatch.setitem(sys.modules, "keyring", fake)
    return store


def test_keyring_resolution_across_llm_servers_and_settings_api(monkeypatch):
    store = _install_fake_keyring(monkeypatch, {
        ("pedacito", "openai"): "sk-from-keyring",
        ("customsvc", "local-strata"): "sk-strata-kr",
    })

    # secrets_store helpers
    assert secrets_store.get_secret("pedacito", "openai") == "sk-from-keyring"
    secrets_store.set_secret("pedacito", "extra", "val-1")
    assert store[("pedacito", "extra")] == "val-1"
    secrets_store.delete_secret("pedacito", "extra")
    assert ("pedacito", "extra") not in store

    # llm._api_key() resolves keyring:<service>/<account> and api_key_keyring
    cfg1 = config_mod.Config(api_key="keyring:pedacito/openai", api_key_env="")
    assert _api_key(cfg1) == "sk-from-keyring"

    cfg2 = config_mod.Config(api_key="", api_key_env="", api_key_keyring="customsvc/local-strata")
    assert _api_key(cfg2) == "sk-strata-kr"

    # api_key_env still beats api_key_keyring and api_key
    monkeypatch.setenv("KR_TEST_ENV", "sk-from-env")
    cfg3 = config_mod.Config(
        api_key="keyring:pedacito/openai",
        api_key_env="KR_TEST_ENV",
        api_key_keyring="customsvc/local-strata",
    )
    assert _api_key(cfg3) == "sk-from-env"

    # servers._headers_for() resolves keyring references
    h1 = servers._headers_for({"api_key": "keyring:pedacito/openai"})
    assert h1 == {"Authorization": "Bearer sk-from-keyring"}
    h2 = servers._headers_for({"api_key_keyring": "customsvc/local-strata"})
    assert h2 == {"Authorization": "Bearer sk-strata-kr"}

    # settings_api._endpoint_auth() resolves keyring references
    auth = settings_api._endpoint_auth(
        {},
        {
            "p1": {"base_url": "http://127.0.0.1:8090/v1", "api_key": "keyring:pedacito/openai"},
            "p2": {"base_url": "http://127.0.0.1:8091/v1", "api_key_keyring": "customsvc/local-strata"},
        },
    )
    assert auth["http://127.0.0.1:8090"] == {"Authorization": "Bearer sk-from-keyring"}
    assert auth["http://127.0.0.1:8091"] == {"Authorization": "Bearer sk-strata-kr"}


def test_missing_keyring_gives_clear_install_hint(monkeypatch):
    monkeypatch.delitem(sys.modules, "keyring", raising=False)
    real_import = builtins.__import__

    def block_keyring(name, *args, **kwargs):
        if name == "keyring":
            raise ImportError("No module named 'keyring'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_keyring)

    with pytest.raises(secrets_store.SecretStoreError) as exc_llm:
        _api_key(config_mod.Config(api_key="keyring:pedacito/openai", api_key_env=""))
    assert "pedacito[keyring]" in str(exc_llm.value)

    with pytest.raises(secrets_store.SecretStoreError) as exc_srv:
        servers._headers_for({"api_key_keyring": "pedacito/openai"})
    assert "pedacito[keyring]" in str(exc_srv.value)

    with pytest.raises(secrets_store.SecretStoreError) as exc_api:
        settings_api._endpoint_auth(
            {},
            {"p1": {"base_url": "http://127.0.0.1:8090/v1",
                    "api_key": "keyring:pedacito/openai"}},
        )
    assert "pedacito[keyring]" in str(exc_api.value)
