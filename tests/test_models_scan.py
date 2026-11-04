"""B3: `pedacito models scan` -- discovery, selection, profile writing, and
the calibration orchestration."""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import cli, config_editor, discovery, modelcard
from pedacito import config as config_mod
from pedacito.local_server import LocalHTTPServer


def _serve(routes, port=0):
    """Run a fake server; `routes` maps a path to (status_code, payload)."""

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            path = self.path.split("?")[0]
            hit = routes.get(path)
            if hit is None:
                self.send_response(404)
                self.end_headers()
                return
            code, payload = hit
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = LocalHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.delenv("PEDACITO_DISCOVERY_URLS", raising=False)
    servers = []

    def start(routes, port=0):
        srv = _serve(routes, port)
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    yield start
    for srv in servers:
        srv.shutdown()
        srv.server_close()


def _lmstudio(models):
    return {
        "/api/v0/models": (200, {"data": [
            {"id": m["id"], "type": m.get("type", "llm"),
             "state": m.get("state", "loaded")} for m in models]}),
        "/v1/models": (200, {"data": models}),
    }


def test_describe_models_filters_embedding_and_too_big(fake):
    gib = 1024 ** 3
    url = fake(_lmstudio([
        {"id": "model-a", "size_bytes": 1_000_000, "context_length": 4096},
        {"id": "text-embedding-small", "type": "embeddings",
         "size_bytes": 1_000_000},
        {"id": "huge-70b", "size_bytes": 70 * gib, "context_length": 32768},
    ]))
    base = url + "/v1"
    out = discovery.describe_models(base, vram_gb=8)
    assert [m["id"] for m in out] == ["model-a"]
    assert out[0]["loaded"] is True
    assert out[0]["size_bytes"] == 1_000_000


def test_scan_calibrate_skips_fresh_and_orders_loaded_first(monkeypatch):
    monkeypatch.setattr(modelcard, "load",
                        lambda cfg: "card" if cfg.model == "fresh" else None)
    calls: list[str] = []

    def fake_calibrate(client, cfg, verbose, quick):
        calls.append(cfg.model)
        return {"version": modelcard.CARD_VERSION, "key": "k", "ts": 0,
                "model": cfg.model, "url": cfg.base_url, "measured": {},
                "choices": {}, "notes": [], "repeats": 1, "seconds": 0.0}

    monkeypatch.setattr(modelcard, "calibrate", fake_calibrate)
    monkeypatch.setattr(modelcard, "save_card", lambda card: Path("x"))

    class FakeClient:
        def __init__(self, cfg):
            self.cfg = cfg

        def check(self):
            pass

        def warmup(self):
            pass

    monkeypatch.setattr(cli, "LLMClient", FakeClient)
    models = [
        {"id": "unloaded", "base_url": "http://x/v1", "vram_gb": 8},
        {"id": "fresh", "base_url": "http://x/v1", "vram_gb": 8, "loaded": True},
        {"id": "loaded", "base_url": "http://x/v1", "vram_gb": 8, "loaded": True},
    ]
    cli._scan_calibrate(models, quick=True)
    # 'fresh' has a card (skipped); 'loaded' (loaded) precedes 'unloaded'
    assert calls == ["loaded", "unloaded"]


def test_models_scan_writes_profiles(tmp_path, fake, monkeypatch):
    monkeypatch.setattr(discovery, "DEFAULT_PORTS", ())       # no real ports
    monkeypatch.setattr(cli, "USER_CONFIG", tmp_path / "no-config.toml")
    gib = 1024 ** 3
    url = fake(_lmstudio([
        {"id": "model-a", "size_bytes": int(8.9 * gib), "context_length": 32768},
    ]))
    cfgfile = tmp_path / "config.toml"
    rc = cli.main(["models", "scan", "--yes", "--calibrate", "none",
                   "--url", url, "--config", str(cfgfile)])
    assert rc == 0
    text = cfgfile.read_text(encoding="utf-8")
    assert "[profiles.model-a]" in text
    assert f'base_url = "{url}/v1"' in text
    assert 'model = "model-a"' in text
    assert "# added by pedacito models scan," in text
    # the result is a valid, loadable config
    cfg = config_mod.load(explicit=cfgfile, use_env=False,
                          require_hardware=False, warn_budgets=False)
    assert "model-a" in cfg.profiles
    # identity only: the new profile never carries card-choice keys
    assert config_editor.profile_body(text, "model-a").get("prompt_pack") is None


def test_models_scan_is_idempotent(tmp_path, fake, monkeypatch):
    monkeypatch.setattr(discovery, "DEFAULT_PORTS", ())
    monkeypatch.setattr(cli, "USER_CONFIG", tmp_path / "no-config.toml")
    gib = 1024 ** 3
    url = fake(_lmstudio([
        {"id": "model-a", "size_bytes": int(8.9 * gib), "context_length": 32768},
    ]))
    cfgfile = tmp_path / "config.toml"
    args = ["models", "scan", "--yes", "--calibrate", "none",
            "--url", url, "--config", str(cfgfile)]
    assert cli.main(args) == 0
    first = cfgfile.read_text(encoding="utf-8")
    assert cli.main(args) == 0
    # a re-scan appends nothing (and the backup keeps the prior file)
    assert cfgfile.read_text(encoding="utf-8") == first
    assert first.count("[profiles.model-a]") == 1
