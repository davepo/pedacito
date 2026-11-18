"""Non-blocking secret & credential guard regression tests.

Covers the fast scanner, masked previews, fingerprint-only acknowledgments,
redaction, localhost-vs-cloud outbound policy, `.gitignore` hygiene, and the
GUI's security endpoints.
"""
import http.client
import json
import os
import re
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import config as config_mod  # noqa: E402
from pedacito import llm, security  # noqa: E402
from pedacito.config import Config  # noqa: E402
from pedacito.gui import core, server, settings_api  # noqa: E402

SECRETS = {
    "openai": "sk-proj-abcdefghijklmnopqrstuvwxyz1234567890",
    "pypi": "pypi-AgEIcHlwaS5vcmcCJDEyMzQ1Njc4OTA",
    "github": "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
    "aws": "AKIAIOSFODNN7EXAMPLE",
    "google": "AIza" + "A" * 35,
    "slack": "xoxb-1234567890-abcdefghij",
    "generic": "AbCdEf0123456789XYZ",
}
PEM = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "MIIEowIBAAKCAQEA1234567890abcdefghijklmnop\n"
    "-----END RSA PRIVATE KEY-----"
)


def _write_project_env(root: Path, secret: str = "") -> None:
    (root / "app.py").write_text(
        f'api_key = "{SECRETS["generic"]}"\n', encoding="utf-8")
    if secret:
        (root / "keys.py").write_text(
            f'OPENAI_API_KEY = "{secret}"\n', encoding="utf-8")


# --------------------------------------------------------------- scanner ---

def test_scan_text_detects_known_formats_and_masks_raw_values():
    text = "\n".join([
        f'openai = "{SECRETS["openai"]}"',
        f'pypi = "{SECRETS["pypi"]}"',
        f'github = "{SECRETS["github"]}"',
        f'aws = "{SECRETS["aws"]}"',
        f'google = "{SECRETS["google"]}"',
        f'slack = "{SECRETS["slack"]}"',
        PEM,
        f'api_key = "{SECRETS["generic"]}"',
    ])
    findings = security.scan_text_for_secrets(text, "demo.py")
    kinds = {f.kind for f in findings}
    assert {"api_key", "pypi_token", "github_token", "aws_access_key",
            "google_api_key", "slack_token", "private_key",
            "credential_assignment"} <= kinds
    assert all(f.file == "demo.py" for f in findings)
    assert all(f.line >= 1 for f in findings)
    assert all(re.fullmatch(r"[0-9a-f]{64}", f.fingerprint)
               for f in findings)
    # The masked payload never contains the complete secret.
    for finding in findings:
        assert finding.raw_match not in finding.masked_preview
    openai = next(f for f in findings if f.kind == "api_key")
    assert openai.masked_preview.startswith("sk-proj")
    assert openai.masked_preview.endswith("7890")
    assert SECRETS["openai"] not in openai.masked_preview
    # PEM findings expose only the header, never the body.
    pem = next(f for f in findings if f.kind == "private_key")
    assert "MIIE" not in pem.masked_preview
    assert pem.to_dict()["kind"] == "private_key"
    assert "raw_match" not in pem.to_dict()


def test_scan_ignores_placeholders_and_environment_lookups():
    text = "\n".join([
        'api_key = "YOUR_API_KEY_1234567890"',
        'token = "changeme_1234567890"',
        'secret = "example_1234567890"',
        'password = "xxxxxxxxxxxxxxxxxxxx"',
        'passwd = "***"',
        'api_key = os.environ["API_KEY"]',
        'authorization = os.environ.get("LONG_AUTH_TOKEN_NAME")',
        'api_key = "os.environ_API_KEY_123456789"',
        'token = "${LONG_SECRET_TOKEN_NAME}"',
    ])
    assert security.scan_text_for_secrets(text, "config.py") == []


# --------------------------------------------------------- acknowledgments ---

def test_acknowledge_findings_persists_only_hashes_and_suppresses(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "app.py").write_text(
        f'api_key = "{SECRETS["generic"]}"\n', encoding="utf-8")
    findings = security.scan_workspace_secrets(root)
    assert findings
    status = security.security_status(root)
    assert status["secrets_found"] is True

    acknowledged = security.acknowledge_findings(
        root, [f.fingerprint for f in findings])
    assert {f.fingerprint for f in findings} <= acknowledged
    ack_path = root / ".pedacito" / "security_ack.json"
    stored = ack_path.read_text(encoding="utf-8")
    data = json.loads(stored)
    assert data["version"] == 1
    assert data["acknowledged"] == sorted(acknowledged)
    assert all(re.fullmatch(r"[0-9a-f]{64}", fp)
               for fp in data["acknowledged"])
    assert SECRETS["generic"] not in stored
    assert SECRETS["generic"] not in json.dumps(data)

    assert security.load_acknowledged_fingerprints(root) == acknowledged
    assert security.security_status(root)["findings"] == []
    assert security.unacknowledged_findings(findings, root) == []


def test_redact_secrets_in_text_replaces_values_cleanly():
    text = (f'api_key = "{SECRETS["generic"]}"\n'
            f'OPENAI_KEY = "{SECRETS["openai"]}"\n')
    redacted = security.redact_secrets_in_text(text)
    assert SECRETS["generic"] not in redacted
    assert SECRETS["openai"] not in redacted
    assert redacted.count(security.REDACTED_SECRET) == 2
    assert 'api_key = "<REDACTED_SECRET>"' in redacted
    assert 'OPENAI_KEY = "<REDACTED_SECRET>"' in redacted


def test_redact_messages_skips_acknowledged_findings(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    text = f'api_key = "{SECRETS["generic"]}"'
    finding = security.scan_text_for_secrets(text, "<prompt>")[0]
    security.acknowledge_findings(root, [finding.fingerprint])
    messages = [{"role": "user", "content": text}]
    out, count, found = security.redact_messages(messages, root)
    assert count == 0 and found == []
    assert out[0]["content"] == text


# ------------------------------------------------------ outbound cloud guard ---

class _FakeResponse:
    status_code = 200
    headers = {}
    encoding = "utf-8"
    text = ""

    def close(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": "ok"}}]}


def _cfg(base_url: str, mode: str = "warn", root=None) -> Config:
    cfg = Config()
    cfg.base_url = base_url
    cfg.api_key = ""
    cfg.api_key_env = ""
    cfg.provider = ""
    cfg.security_outbound_mode = mode
    cfg.timeout = 2
    if root is not None:
        cfg._root = str(root)
    return cfg


def _post_with(monkeypatch, cfg):
    calls = []

    def fake_post(url, **kwargs):
        calls.append({"url": url, "kwargs": kwargs})
        return _FakeResponse()

    monkeypatch.setattr(llm.requests, "post", fake_post)
    return calls


def _payload(text: str) -> dict:
    return {"model": "test", "messages": [{"role": "user", "content": text}]}


def test_localhost_never_triggers_outbound_confirmation(monkeypatch):
    security.install_permission_hook(lambda info: (True, "allowed"))
    try:
        for url in ("http://127.0.0.1:1234/v1", "http://localhost:1234/v1"):
            cfg = _cfg(url, "warn")
            calls = _post_with(monkeypatch, cfg)
            client = llm.LLMClient(cfg)
            text = f'api_key = "{SECRETS["generic"]}"'
            client._post(_payload(text))
            assert calls
            sent = calls[0]["kwargs"]["json"]["messages"][0]["content"]
            assert SECRETS["generic"] in sent
    finally:
        security.install_permission_hook(None)


def test_external_warn_mode_redacts_when_non_interactive(monkeypatch):
    security.install_permission_hook(None)
    cfg = _cfg("https://api.deepseek.com/v1", "warn")
    calls = _post_with(monkeypatch, cfg)
    client = llm.LLMClient(cfg)
    text = f'api_key = "{SECRETS["generic"]}"'
    client._post(_payload(text))
    assert calls
    sent = json.dumps(calls[0]["kwargs"]["json"])
    assert SECRETS["generic"] not in sent
    assert security.REDACTED_SECRET in sent


def test_external_redact_mode_always_redacts(monkeypatch):
    security.install_permission_hook(None)
    cfg = _cfg("https://api.deepseek.com/v1", "redact")
    calls = _post_with(monkeypatch, cfg)
    client = llm.LLMClient(cfg)
    client._post(_payload(f'k = "{SECRETS["openai"]}"'))
    sent = json.dumps(calls[0]["kwargs"]["json"])
    assert SECRETS["openai"] not in sent
    assert security.REDACTED_SECRET in sent


def test_external_warn_mode_pauses_with_confirmation(monkeypatch):
    security.install_permission_hook(lambda info: (True, "allowed"))
    try:
        cfg = _cfg("https://api.deepseek.com/v1", "warn")
        calls = _post_with(monkeypatch, cfg)
        client = llm.LLMClient(cfg)
        with pytest.raises(llm.SecretConfirmationRequired) as exc:
            client._post(_payload(f'k = "{SECRETS["openai"]}"'))
        assert calls == []
        message = str(exc.value)
        assert "api_key" in message
        assert SECRETS["openai"] not in message
        assert exc.value.findings
    finally:
        security.install_permission_hook(None)


def test_explicit_acknowledge_flag_sends_as_is_and_persists(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    security.install_permission_hook(lambda info: (True, "allowed"))
    try:
        cfg = _cfg("https://api.deepseek.com/v1", "warn", root=root)
        cfg._outbound_acknowledge_secrets = True
        calls = _post_with(monkeypatch, cfg)
        client = llm.LLMClient(cfg)
        client._post(_payload(f'api_key = "{SECRETS["generic"]}"'))
        sent = calls[0]["kwargs"]["json"]["messages"][0]["content"]
        assert SECRETS["generic"] in sent
        assert security.load_acknowledged_fingerprints(root)
    finally:
        security.install_permission_hook(None)


# ------------------------------------------------------- gitignore hygiene ---

def test_check_git_secret_hygiene_and_ensure_gitignore_entries(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / ".env").write_text("SAFE=1\n", encoding="utf-8")
    (root / ".env.example").write_text("API_KEY=YOUR_API_KEY\n",
                                       encoding="utf-8")
    findings = security.check_git_secret_hygiene(root)
    assert [f.file for f in findings] == [".env"]
    assert findings[0].kind == "unignored_env"

    added = security.ensure_gitignore_entries(root)
    assert added == [".env", ".env.*"]
    text = (root / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in text and ".env.*" in text
    assert security.check_git_secret_hygiene(root) == []
    assert security.ensure_gitignore_entries(root) == []


def test_hygiene_ignores_sensitive_file_already_covered(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / ".env.local").write_text("SAFE=1\n", encoding="utf-8")
    (root / ".gitignore").write_text(".env*\n", encoding="utf-8")
    assert security.check_git_secret_hygiene(root) == []
    assert security.ensure_gitignore_entries(root) == []


# --------------------------------------------------------------- GUI routes ---

class _Client:
    def __init__(self, port):
        self.port = port

    def _req(self, method, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        data = None if body is None and method == "GET" else json.dumps(body or {})
        conn.request(method, path, body=data,
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        payload = response.read()
        conn.close()
        parsed = json.loads(payload) if payload[:1] in (b"{", b"[") else payload
        return response.status, parsed

    def get(self, path):
        return self._req("GET", path)

    def post(self, path, body=None):
        return self._req("POST", path, body or {})


@pytest.fixture
def gui(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg" / "config.toml"
    monkeypatch.setattr(config_mod, "USER_CONFIG", cfg_path)
    monkeypatch.setattr(core, "USER_CONFIG", cfg_path)
    monkeypatch.setattr(settings_api, "USER_CONFIG", cfg_path)
    monkeypatch.setattr(core, "RECENT_FILE", tmp_path / "cfg" / "gui_recent.json")
    for var in [k for k in os.environ if k.startswith("PEDACITO_")]:
        monkeypatch.delenv(var)
    security.clear_security_cache()
    core.STATE.__init__()
    srv = server.create_server(0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    client = _Client(srv.server_address[1])
    yield client, tmp_path
    core.STATE.bus.close()
    srv.shutdown()
    srv.server_close()
    thread.join(5)
    security.clear_security_cache()


def _open_project(root: Path) -> Config:
    cfg = Config()
    cfg.base_url = "http://127.0.0.1:1234/v1"
    cfg._root = str(root)
    with core.STATE.lock:
        core.STATE.root = root
        core.STATE.cfg = cfg
        core.STATE.client = None
        core.STATE.idx = None
    return cfg


def test_gui_security_status_ack_and_gitignore(gui):
    client, tmp_path = gui
    root = tmp_path / "proj"
    root.mkdir()
    _write_project_env(root)
    (root / ".env").write_text("SAFE=1\n", encoding="utf-8")
    _open_project(root)

    status, payload = client.get("/api/security/status")
    assert status == 200
    assert payload["secrets_found"] is True
    assert payload["env_unignored"] is True
    assert ".env" in payload["unignored_env"]
    assert SECRETS["generic"] not in json.dumps(payload)
    state_status, state_payload = client.get("/api/state")
    assert state_status == 200
    assert state_payload["security"]["secrets_found"] is True
    assert state_payload["security"]["unignored_env"] == [".env"]
    assert SECRETS["generic"] not in json.dumps(state_payload["security"])
    finding = next(f for f in payload["findings"] if f["file"] == "app.py")
    assert finding["masked_preview"].endswith("XYZ")
    assert "AbCdEf0123456789XYZ" not in finding["masked_preview"]

    status, payload = client.post(
        "/api/security/ack", {"fingerprints": [finding["fingerprint"]]})
    assert status == 200
    stored = (root / ".pedacito" / "security_ack.json").read_text(
        encoding="utf-8")
    assert SECRETS["generic"] not in stored
    assert finding["fingerprint"] in stored

    status, payload = client.get("/api/security/status")
    assert status == 200
    assert payload["secrets_found"] is False

    status, payload = client.post("/api/security/gitignore", {})
    assert status == 200
    assert ".env" in (root / ".gitignore").read_text(encoding="utf-8")
    status, payload = client.get("/api/security/status")
    assert status == 200
    assert payload["env_unignored"] is False
    assert payload["unignored_env"] == []


def test_gui_security_scan_is_local_aware(gui):
    client, tmp_path = gui
    root = tmp_path / "proj"
    root.mkdir()
    cfg = _open_project(root)
    secret_text = f'api_key = "{SECRETS["generic"]}"'

    status, payload = client.post("/api/security/scan",
                                  {"text": secret_text, "mode": "task"})
    assert status == 200
    assert payload["external_provider"] is False
    assert payload["needs_confirmation"] is False
    assert payload["findings"]

    cfg.base_url = "https://api.deepseek.com/v1"
    cfg.security_outbound_mode = "warn"
    status, payload = client.post("/api/security/scan",
                                  {"text": secret_text, "mode": "task"})
    assert status == 200
    assert payload["external_provider"] is True
    assert payload["needs_confirmation"] is True
    assert payload["findings"][0]["kind"] == "credential_assignment"

    cfg.security_outbound_mode = "redact"
    status, payload = client.post("/api/security/scan",
                                  {"text": secret_text, "mode": "task"})
    assert status == 200
    assert payload["needs_confirmation"] is False
