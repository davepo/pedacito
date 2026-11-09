"""B1: model-server discovery by fingerprint (their replies), not by port."""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import discovery
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
    """Starts fake servers on loopback and returns a `start(routes) -> url`."""
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


def _lmstudio():
    return {
        "/api/v0/models": (200, {"data": [{"id": "lm/model-a", "type": "llm"}]}),
        "/api/v1/models": (200, {"data": [{"id": "lm/model-a", "type": "llm"}]}),
        "/v1/models": (200, {"data": [{"id": "lm/model-a"}]}),
    }


def _ollama():
    return {
        "/api/version": (200, {"version": "0.5.0"}),
        "/api/tags": (200, {"models": [{"name": "qwen3:14b"}]}),
        "/v1/models": (200, {"data": [{"id": "qwen3:14b"}]}),
    }


def _openai():
    return {"/v1/models": (200, {"data": [{"id": "gpt-x"}]})}


def _lemonade():
    return {
        "/api/v1/health": (200, {"status": "ok"}),
        "/v1/models": (200, {"data": [
            {"id": "lemon/lemonade-14b", "owned_by": "lemonade"}]}),
    }


def _llamacpp():
    return {
        "/props": (200, {"default_generation_settings": {}}),
        "/v1/models": (200, {"data": [{"id": "llama-3"}]}),
    }


def test_lm_studio_is_fingerprinted(fake):
    url = fake(_lmstudio())
    found = discovery.discover_servers(urls=[url], ports=())
    assert [(lbl, base, ids) for lbl, base, ids in found] == \
        [("LM Studio", url + "/v1", ["lm/model-a"])]


def test_ollama_is_fingerprinted(fake):
    url = fake(_ollama())
    found = discovery.discover_servers(urls=[url], ports=())
    assert found[0][0] == "Ollama" and found[0][2] == ["qwen3:14b"]
    assert found[0][1] == url + "/v1"


def test_openai_shape_is_generic(fake):
    url = fake(_openai())
    found = discovery.discover_servers(urls=[url], ports=())
    assert found[0][0] == "OpenAI-compatible" and found[0][2] == ["gpt-x"]


def test_llamacpp_is_fingerprinted(fake):
    url = fake(_llamacpp())
    found = discovery.discover_servers(urls=[url], ports=())
    assert found[0][0] == "llama.cpp"


def test_lemonade_on_a_shared_port_is_not_mislabelled(fake):
    """A Lemonade server answers /api/v1/health and owned_by=lemonade; even on
    the port that used to mean "Ollama", the fingerprint reads Lemonade."""
    url = fake(_lemonade())
    found = discovery.discover_servers(urls=[url], ports=())
    assert [lbl for lbl, _b, _ids in found] == ["Lemonade"]


def test_profile_url_finds_a_nonstandard_port(fake):
    """An existing profile's base_url re-finds a server on a custom port."""
    url = fake(_ollama())
    found = discovery.discover_servers(profile_urls=[url + "/v1"], ports=())
    assert [lbl for lbl, _b, _ids in found] == ["Ollama"]


def test_two_servers_on_one_host_are_two_entries(fake):
    lm = fake(_lmstudio())
    oll = fake(_ollama())
    found = discovery.discover_servers(urls=[lm, oll], ports=())
    assert sorted(lbl for lbl, _b, _ids in found) == ["LM Studio", "Ollama"]


def test_api_v1_only_surface_is_labelled_lm_studio(fake):
    """A server exposing only /api/v1/models (no /v1, no /api/v0) is LM
    Studio's native layout; Lemonade is already ruled out via /api/v1/health."""
    url = fake({"/api/v1/models": (200, {"data": [{"id": "m", "type": "llm"}]})})
    found = discovery.discover_servers(urls=[url], ports=())
    assert found[0][0] == "LM Studio" and found[0][1] == url + "/api/v1"


def test_dedup_by_server_root(fake):
    url = fake(_openai())
    found = discovery.discover_servers(urls=[url, url + "/v1"], ports=())
    assert len(found) == 1


def test_normalize_url_collapses_server_spellings():
    """One server, every spelling: one key. Trailing slash, `/v1`, `/api/v1`
    and the `localhost` name are all cosmetic."""
    assert discovery.normalize_url("http://localhost:11434/v1/") == "http://127.0.0.1:11434"
    assert discovery.normalize_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434"
    assert discovery.normalize_url("http://localhost:8001/api/v1") == "http://127.0.0.1:8001"
    assert discovery.normalize_url("https://localhost:1234/v1") == "https://127.0.0.1:1234"
    assert discovery.normalize_url(" HTTP://LOCALHOST:9/ ") == "http://127.0.0.1:9"
    # anything past the version prefix stays part of the URL
    assert discovery.normalize_url("http://h.test:8080/v1/extra/") == "http://h.test:8080/v1/extra"
    # a host that merely starts with "localhost" is not the loopback alias
    assert discovery.normalize_url("http://localhost.example/v1") == "http://localhost.example"
    assert discovery.normalize_url("") == ""
    # idempotent: normalizing a normalized URL changes nothing
    assert discovery.normalize_url("http://127.0.0.1:11434") == \
        discovery.normalize_url("http://localhost:11434/v1/")


def test_candidate_roots_collapse_spellings():
    """localhost vs 127.0.0.1, with or without /v1 or a trailing slash, is
    one probe -- never two entries for one server."""
    roots = discovery._candidate_roots(
        ["http://localhost:9999/v1/", "http://127.0.0.1:9999"], [], [])
    assert roots == ["http://127.0.0.1:9999"]
    # the well-known port spellings land on the same key as a profile URL
    roots = discovery._candidate_roots([], ["http://localhost:11434/v1"], [11434])
    assert roots == ["http://127.0.0.1:11434"]


def test_one_server_three_spellings_answers_once(fake, monkeypatch):
    url = fake(_openai())
    port = url.rsplit(":", 1)[1]
    probed = []
    real_probe = discovery._probe_root

    def spy(root, *a, **kw):
        probed.append(root)
        return real_probe(root, *a, **kw)

    monkeypatch.setattr(discovery, "_probe_root", spy)
    found = discovery.discover_servers(
        urls=[f"http://localhost:{port}/v1/", url, f"http://127.0.0.1:{port}/"],
        ports=())
    assert len(found) == 1 and len(probed) == 1


def test_probe_root_does_not_http_probe_a_dead_port(monkeypatch):
    """A closed local port must cost one short TCP check, not one HTTP timeout
    per fingerprint endpoint. That multiplier made GUI scans exceed their
    request budget on Windows."""
    calls = []
    monkeypatch.setattr(discovery, "_listening",
                        lambda root, timeout: False)
    monkeypatch.setattr(discovery, "_get",
                        lambda *a, **k: calls.append((a, k)))
    assert discovery._probe_root("http://127.0.0.1:9", 2.0) is None
    assert calls == []


def test_listening_is_true_for_a_live_server_and_false_for_a_dead_port(fake):
    url = fake(_openai())
    assert discovery._listening(url, 1.0) is True
    assert discovery._listening("http://127.0.0.1:1", 0.5) is False


def test_looks_embedding_from_type_or_name():
    assert discovery.looks_embedding({"id": "text-embedding-3-small"})
    assert discovery.looks_embedding({"type": "embeddings", "id": "x"})
    assert discovery.looks_embedding({"name": "nomic-embed-text"})
    assert not discovery.looks_embedding({"id": "qwen3:14b"})


def test_too_big_for_vram():
    gib = 1024 ** 3
    assert discovery.too_big_for_vram(9 * gib, 8)
    assert not discovery.too_big_for_vram(7 * gib, 8)
    assert not discovery.too_big_for_vram(0, 8)      # unknown size: not flagged
    assert not discovery.too_big_for_vram(9 * gib, 0)  # unknown VRAM: not flagged


def test_model_info_takes_an_explicit_model_id(monkeypatch):
    """B1: model_info(model) describes the given id, not just cfg.model."""
    from pedacito import llm as llm_mod
    from pedacito.config import Config

    class Resp:
        status_code = 200

        def __init__(self, data):
            self._d = data

        def json(self):
            return self._d

    def fake_get(url, **k):
        return Resp({"data": [
            {"id": "model-a", "size_bytes": 1_000_000, "context_length": 4096},
            {"id": "model-b", "size_bytes": 2_000_000, "context_length": 8192},
        ]})

    monkeypatch.setattr(llm_mod.requests, "get", fake_get)
    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **k: Resp({}))
    cfg = Config()
    cfg.base_url = "http://127.0.0.1:1/v1"
    cfg.model = "model-a"
    client = llm_mod.LLMClient(cfg)
    assert client.model_info()["size_bytes"] == 1_000_000
    assert client.model_info("model-b")["size_bytes"] == 2_000_000
    assert client.model_info("model-b")["context_length"] == 8192
