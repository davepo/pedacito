"""B1: model-server discovery by fingerprint (their replies), not by port."""
from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import discovery
from pedacito.local_server import LocalHTTPServer


def _serve(routes, port=0, auth=None, protected=()):
    """Run a fake server; `routes` maps a path to (status_code, payload).

    With `auth`, the paths in `protected` answer 401 unless the request's
    Authorization header matches that value -- the shape a Strata server uses
    for `/v1/models`."""

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            path = self.path.split("?")[0]
            if auth and path in protected \
                    and self.headers.get("Authorization") != auth:
                code, payload = 401, {"error": {
                    "type": "authentication_error",
                    "message": "missing or wrong API key"}}
            else:
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

    def start(routes, port=0, **kw):
        srv = _serve(routes, port, **kw)
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


def test_ollama_with_zero_models_is_still_a_server(fake):
    """A fresh Ollama answers /api/tags with an empty model list. The probe
    must report the server with no models, not crash iterating None."""
    url = fake({
        "/api/version": (200, {"version": "0.5.0"}),
        "/api/tags": (200, {"models": []}),
    })
    found = discovery.discover_servers(urls=[url], ports=())
    assert found == [("Ollama", url + "/v1", [])]


def test_openai_shape_is_generic(fake):
    url = fake(_openai())
    found = discovery.discover_servers(urls=[url], ports=())
    assert found[0][0] == "OpenAI-compatible" and found[0][2] == ["gpt-x"]


def test_llamacpp_is_fingerprinted(fake):
    url = fake(_llamacpp())
    found = discovery.discover_servers(urls=[url], ports=())
    assert found[0][0] == "llama.cpp"


def _strata():
    return {
        "/health": (200, {"status": "ok", "service": "strata",
                          "model": "qwen3.8-flash", "max_context": 131072,
                          "loaded": True, "api_key": True}),
        "/v1/models": (200, {"data": [{"id": "qwen3.8-flash"}]}),
    }


def test_strata_is_fingerprinted_and_lists_from_public_health(fake):
    """A Strata server's /v1/models can be behind an API key, but its public
    /health names the service, the model and the context window. Discovery
    must find it (and its model) without credentials."""
    url = fake(_strata(), auth="Bearer secret", protected={"/v1/models"})
    found = discovery.discover_servers(urls=[url], ports=())
    assert found == [("Strata", url + "/v1", ["qwen3.8-flash"])]


def test_strata_entries_preserve_requires_api_key_and_provider(fake):
    """The public /health names the runtime and whether an API key is needed;
    `_strata_entries` must carry both so a key-protected Strata is still
    described (and importable) without its credential."""
    url = fake({
        "/health": (200, {"service": "avalon", "model": "qwen3.8-flash",
                          "max_context": 131072, "loaded": True,
                          "api_key": True}),
    })
    entries = discovery._strata_entries(url, 1.0)
    assert entries == [{
        "id": "qwen3.8-flash",
        "context_length": 131072,
        "loaded": True,
        "requires_api_key": True,
        "provider": "avalon",
    }]


def test_strata_entries_report_no_key_when_health_omits_api_key(fake):
    url = fake({
        "/health": (200, {"service": "strata", "model": "qwen3.8-flash",
                          "max_context": 32768, "loaded": False}),
    })
    entries = discovery._strata_entries(url, 1.0)
    assert entries == [{
        "id": "qwen3.8-flash",
        "context_length": 32768,
        "loaded": False,
        "requires_api_key": False,
        "provider": "strata",
    }]


def test_describe_models_carries_requires_api_key_from_health(fake):
    """A Strata whose /v1/models answers 401 but whose /health answers 200 is
    still described; `requires_api_key` rides the record so the GUI scan can
    import it with an api_key/api_key_env."""
    url = fake(_strata(), auth="Bearer secret", protected={"/v1/models"})
    rows = discovery.describe_models(url + "/v1")
    assert rows == [{"id": "qwen3.8-flash", "base_url": url + "/v1",
                     "requires_api_key": True, "context_length": 131072,
                     "loaded": True}]


def test_auth_mapping_unlocks_a_key_protected_listing(fake):
    """Per-server credentials from configured profiles are sent to that
    server only; without them the OpenAI listing answers 401 and the server
    is not discoverable."""
    url = fake(_openai(), auth="Bearer secret", protected={"/v1/models"})
    assert discovery.discover_servers(urls=[url], ports=()) == []
    found = discovery.discover_servers(
        urls=[url], ports=(),
        auth={url + "/v1": {"Authorization": "Bearer secret"}})
    assert found == [("OpenAI-compatible", url + "/v1", ["gpt-x"])]


def test_auth_headers_match_normalized_spellings_and_never_leak():
    auth = {"http://127.0.0.1:9000/v1": {"Authorization": "Bearer secret"}}
    assert discovery._auth_headers("http://localhost:9000", auth) == \
        {"Authorization": "Bearer secret"}
    assert discovery._auth_headers("http://127.0.0.1:9001", auth) == {}


def test_describe_models_uses_the_listing_context_and_loaded_status(fake):
    url = fake({
        "/v1/models": (200, {"data": [{
            "id": "qwen3.8-flash", "context_length": 131072,
            "status": {"value": "loaded"}}]}),
    })
    rows = discovery.describe_models(url + "/v1", vram_gb=0)
    assert rows == [{"id": "qwen3.8-flash", "base_url": url + "/v1",
                     "context_length": 131072, "loaded": True}]


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
                        lambda root, timeout, deadline=None: False)
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


# ------------------------------------------------- R3: wall-clock scan budget
def _serve_sleeping(delay, port=0):
    """A server that accepts connections promptly but sleeps `delay` seconds
    before answering any request -- a slow remote profile URL in miniature."""

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            time.sleep(delay)
            try:
                body = b'{"data": [{"id": "slow-model"}]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass                    # the client gave up on the budget

    srv = LocalHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_discovery_returns_partial_results_within_the_budget(fake):
    """R3: a fast server plus one that sleeps far past the budget. The scan
    must return within ~budget carrying the fast server, not hang on the slow
    one (the GUI request used to block on every remote profile URL)."""
    fast = fake(_openai())
    slow_srv = _serve_sleeping(5.0)
    slow = f"http://127.0.0.1:{slow_srv.server_address[1]}"
    try:
        t0 = time.monotonic()
        found = discovery.discover_servers(urls=[fast, slow], ports=(),
                                           budget=0.6)
        elapsed = time.monotonic() - t0
        assert elapsed < 2.5, f"scan blew its budget: {elapsed:.2f}s"
        bases = [b for _l, b, _ids in found]
        assert fast + "/v1" in bases          # the fast server was probed
        assert slow + "/v1" not in bases      # the slow one was cut off
    finally:
        slow_srv.shutdown()
        slow_srv.server_close()


def test_discovery_without_a_budget_still_probes_everything(fake):
    """The budget is opt-in: with none, a slow-but-answering server is still
    found (behaviour unchanged for callers that pass no budget)."""
    slow_srv = _serve_sleeping(0.05)
    slow = f"http://127.0.0.1:{slow_srv.server_address[1]}"
    try:
        found = discovery.discover_servers(urls=[slow], ports=())
        assert [b for _l, b, _ids in found] == [slow + "/v1"]
    finally:
        slow_srv.shutdown()
        slow_srv.server_close()


def test_slice_caps_the_timeout_by_the_remaining_budget():
    """_slice bounds a request by the time left, and answers 0 once spent."""
    assert discovery._slice(2.0, None) == 2.0
    now = time.monotonic()
    assert discovery._slice(2.0, now + 0.5) <= 0.5
    assert discovery._slice(2.0, now - 1.0) == 0.0
    # a short per-request timeout still wins over a large remaining budget
    assert discovery._slice(0.2, now + 10.0) == 0.2
