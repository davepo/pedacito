"""Sprint R: local-inference payload acceleration (cache_prompt +
prompt_lookup_num_tokens with a 400/422 strip-and-retry fallback) and the
pruned, permission-gated named-server MCP bridge (`mcp_call`).

Invariants: the acceleration keys only ever reach self-hosted endpoints
(loopback, private LAN, or a local-runtime provider name); a 400/422 landing
while either key is attached flips it off for the (base_url, model) pair,
re-sends the request once without it, and every later call starts stripped.
Named [mcp_servers.<name>] tables feed list_mcp_tools / call_mcp_tool /
relevant_tools; the gather schema only offers `mcp_call` when a listed tool
matches the task, and an untrusted server's call goes through the permission
broker with the fixed decline text. The legacy allow_mcp + command-list path
keeps working untouched."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pedacito import config as config_mod
from pedacito import llm as llm_mod
from pedacito import mcp, security, tools
from pedacito.config import Config
from pedacito.gui import settings as S
from pedacito.llm import LLMClient, LLMError
from test_llm_transport import Resp, _cfg, _ok, _patch_post

FAKE = Path(__file__).resolve().parent / "fake_mcp.py"


@pytest.fixture(autouse=True)
def _fresh_accel_caps():
    """The acceleration cache is process-wide; keep tests independent."""
    llm_mod._ACCEL_CAPS.clear()
    mcp._NAMED_TOOLS.clear()
    mcp._NAMED_TOOLS_TIME.clear()
    yield
    llm_mod._ACCEL_CAPS.clear()
    mcp.close_all()


def _no_schema_done_payload(text="hello"):
    return _ok(text)


# ------------------------------------------------------- acceleration keys
def test_local_payload_attaches_cache_prompt_by_default(monkeypatch):
    posts = []

    def fake(url, **kw):
        posts.append(dict(kw["json"]))
        return _no_schema_done_payload()

    _patch_post(monkeypatch, fake)
    out = LLMClient(_cfg()).chat([{"role": "user", "content": "go"}], 50)
    assert out == "hello"
    assert posts[0]["cache_prompt"] is True
    assert "prompt_lookup_num_tokens" not in posts[0]


def test_lookup_speculation_attaches_when_enabled(monkeypatch):
    posts = []

    def fake(url, **kw):
        posts.append(dict(kw["json"]))
        return _no_schema_done_payload()

    _patch_post(monkeypatch, fake)
    cfg = _cfg(enable_prompt_lookup=True, prompt_lookup_num_tokens=12)
    LLMClient(cfg).chat([{"role": "user", "content": "go"}], 50)
    assert posts[0]["cache_prompt"] is True
    assert posts[0]["prompt_lookup_num_tokens"] == 12


def test_hosted_endpoints_never_see_the_keys(monkeypatch):
    posts = []

    def fake(url, **kw):
        posts.append(dict(kw["json"]))
        return _no_schema_done_payload()

    _patch_post(monkeypatch, fake)
    cfg = _cfg(base_url="https://api.hosted.example/v1",
               enable_prompt_lookup=True)
    LLMClient(cfg).chat([{"role": "user", "content": "go"}], 50)
    assert "cache_prompt" not in posts[0]
    assert "prompt_lookup_num_tokens" not in posts[0]


def test_private_lan_counts_as_local(monkeypatch):
    posts = []

    def fake(url, **kw):
        posts.append(dict(kw["json"]))
        return _no_schema_done_payload()

    _patch_post(monkeypatch, fake)
    cfg = _cfg(base_url="http://192.168.4.20:8080/v1")
    LLMClient(cfg).chat([{"role": "user", "content": "go"}], 50)
    assert posts[0]["cache_prompt"] is True


def test_public_ip_with_local_provider_name_is_local():
    cfg = _cfg(base_url="http://203.0.113.9:1234/v1", provider="llamacpp")
    assert llm_mod._is_local_server(cfg)


def test_public_ip_without_local_provider_is_not_local():
    cfg = _cfg(base_url="http://203.0.113.9:1234/v1", provider="")
    assert not llm_mod._is_local_server(cfg)


def test_sampler_key_wins_over_acceleration(monkeypatch):
    posts = []

    def fake(url, **kw):
        posts.append(dict(kw["json"]))
        return _no_schema_done_payload()

    _patch_post(monkeypatch, fake)
    cfg = _cfg(sampler={"cache_prompt": False})
    LLMClient(cfg).chat([{"role": "user", "content": "go"}], 50)
    assert posts[0]["cache_prompt"] is False   # the user's own spelling kept


def test_400_refusal_strips_keys_retries_and_remembers(monkeypatch):
    posts = []

    def fake(url, **kw):
        p = dict(kw["json"])
        posts.append(p)
        if "cache_prompt" in p or "prompt_lookup_num_tokens" in p:
            return Resp(400, text='{"error": "unknown field cache_prompt"}')
        return _no_schema_done_payload("fine")

    _patch_post(monkeypatch, fake)
    cfg = _cfg(enable_prompt_lookup=True, model="accel-refuse")
    out = LLMClient(cfg).chat([{"role": "user", "content": "go"}], 50)
    assert out == "fine"
    assert len(posts) == 2                        # retried immediately
    assert "cache_prompt" in posts[0]
    assert "prompt_lookup_num_tokens" in posts[0]
    assert "cache_prompt" not in posts[1]
    assert "prompt_lookup_num_tokens" not in posts[1]
    caps = llm_mod._accel_caps(cfg.base_url, cfg.model)
    assert caps["cache"] is False and caps["lookup"] is False
    # Remembered: every later call starts stripped, one post only.
    posts.clear()
    LLMClient(cfg).chat([{"role": "user", "content": "go"}], 50)
    assert len(posts) == 1 and "cache_prompt" not in posts[0]


def test_422_also_falls_back(monkeypatch):
    posts = []

    def fake(url, **kw):
        p = dict(kw["json"])
        posts.append(p)
        if "cache_prompt" in p:
            return Resp(422, text='{"detail": "extra field not allowed"}')
        return _no_schema_done_payload()

    _patch_post(monkeypatch, fake)
    cfg = _cfg(model="accel-422")
    out = LLMClient(cfg).chat([{"role": "user", "content": "go"}], 50)
    assert out == "hello"
    assert len(posts) == 2 and "cache_prompt" not in posts[1]


def test_400_for_an_unrelated_reason_is_not_looped(monkeypatch):
    posts = []

    def fake(url, **kw):
        posts.append(dict(kw["json"]))
        return Resp(400, text='{"error": "body missing model field"}')

    _patch_post(monkeypatch, fake)
    cfg = _cfg(model="always-400")
    with pytest.raises(LLMError):
        LLMClient(cfg).chat([{"role": "user", "content": "go"}], 50)
    assert len(posts) == 2          # one strip-retry, then the error stands
    assert "cache_prompt" not in posts[1]


def test_streaming_refusal_falls_back_too(monkeypatch):
    posts = []
    lines = ['data: ' + json.dumps({"choices": [{"delta": {"content": "hi"},
                                                 "finish_reason": None}]}),
             'data: ' + json.dumps({"choices": [{"delta": {},
                                                 "finish_reason": "stop"}]}),
             "data: [DONE]"]

    def fake(url, **kw):
        p = dict(kw["json"])
        posts.append(p)
        if "cache_prompt" in p:
            return Resp(400, text='{"error": "unknown field cache_prompt"}')
        return Resp(200, lines=lines)

    _patch_post(monkeypatch, fake)
    cfg = _cfg(model="accel-stream")
    out = LLMClient(cfg).chat([{"role": "user", "content": "go"}], 50,
                              stream=True)
    assert out == "hi"
    assert len(posts) == 2 and "cache_prompt" not in posts[1]


# ------------------------------------------------------ named server config
def test_named_server_tables_parse():
    cfg = Config()
    flat = config_mod._coerce("mcp_servers", {
        "docs": {"command": "python", "args": ["s.py"], "trusted": True},
        "web": {"url": "http://127.0.0.1:9/mcp"},
        "off": {"command": "x", "enabled": False},
    })
    cfg.mcp_servers = flat
    servers = mcp.named_servers(cfg)
    assert servers["docs"]["trusted"] is True
    assert servers["docs"]["args"] == ["s.py"]
    assert servers["web"]["url"] == "http://127.0.0.1:9/mcp"
    assert servers["web"]["trusted"] is False       # default
    assert "off" not in servers                      # enabled = false
    assert servers["web"]["env"] == {}


def test_legacy_command_list_still_works():
    cfg = Config()
    cfg.mcp_servers = ("python one.py", "node two.js")
    assert mcp.named_servers(cfg) == {}              # no named path
    cfg.allow_mcp = True
    assert mcp._servers(cfg) == ("python one.py", "node two.js")


def test_mcp_servers_flatten_from_toml_tables():
    import tomllib
    data = tomllib.loads('[mcp_servers.docs]\ncommand = "python"\n'
                         'args = ["s.py"]\n')
    flat, _ = config_mod._flatten(data, Path("t"))
    assert config_mod._coerce("mcp_servers", flat["mcp_servers"]) == {
        "docs": {"command": "python", "args": ["s.py"]}}


def test_mcp_servers_env_both_formats():
    assert config_mod._parse_env(
        "mcp_servers", '{"a": {"url": "http://h"}}') == {"a": {"url": "http://h"}}
    assert config_mod._parse_env("mcp_servers", "x y, z") == ("x y", "z")


def test_enable_mcp_false_hides_the_named_path():
    cfg = Config()
    cfg.mcp_servers = {"a": {"command": "x"}}
    cfg.enable_mcp = False
    assert mcp.named_servers(cfg) == {}


# --------------------------------------------------------- tools.list/call
def _named_cfg(**kw):
    cfg = Config()
    cfg.enable_mcp = True
    cfg.mcp_servers = {"fake": {"command": sys.executable,
                                "args": [str(FAKE)]}}
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_list_mcp_tools_descriptors():
    cfg = _named_cfg()
    tools_ = mcp.list_mcp_tools(cfg)
    assert any(d["server"] == "fake" and d["name"] == "echo" for d in tools_)
    d = tools_[0]
    assert set(d) == {"server", "name", "description", "input_schema",
                      "trusted"}
    assert d["trusted"] is False
    assert isinstance(d["input_schema"], dict)


def test_list_mcp_tools_caches_per_session():
    cfg = _named_cfg()
    first = mcp.list_mcp_tools(cfg)
    assert "fake" in mcp._NAMED_TOOLS            # cached for the session
    second = mcp.list_mcp_tools(cfg)
    assert second == first


def test_call_mcp_tool_echo_and_schema_check():
    cfg = _named_cfg()
    out = mcp.call_mcp_tool(cfg, "fake", "echo", {"x": "hi"})
    assert out.startswith("echo:") and '"x": "hi"' in out
    assert mcp.call_mcp_tool(cfg, "fake", "ghost", {}).startswith("ERROR")
    assert mcp.call_mcp_tool(cfg, "nowhere", "echo", {}).startswith("ERROR")


def test_call_mcp_tool_caps_at_max_read_tokens():
    cfg = _named_cfg(max_read_tokens=300)
    big = {"blob": "x" * 20_000}
    out = mcp.call_mcp_tool(cfg, "fake", "echo", big)
    assert "capped at about 300 tokens" in out
    assert len(out) < 20_000


# -------------------------------------------------------------- relevance
def _descs():
    return [
        {"server": "docs", "name": "search_api",
         "description": "search the project documentation",
         "input_schema": {"type": "object"}, "trusted": False},
        {"server": "docs", "name": "fetch_page",
         "description": "fetch one documentation page",
         "input_schema": {"type": "object"}, "trusted": False},
        {"server": "jira", "name": "create_ticket",
         "description": "open a ticket in the tracker",
         "input_schema": {"type": "object"}, "trusted": False},
        {"server": "jira", "name": "list_projects",
         "description": "list tracker projects",
         "input_schema": {"type": "object"}, "trusted": False},
        {"server": "fs", "name": "read_file",
         "description": "read a file",
         "input_schema": {"type": "object"}, "trusted": True},
        {"server": "fs", "name": "stat_file",
         "description": "stat a file",
         "input_schema": {"type": "object"}, "trusted": True},
    ]


def test_relevant_tools_matches_task_and_caps_at_four():
    hits = mcp.relevant_tools(_descs(), "update the documentation page")
    names = [d["name"] for d in hits]
    assert "fetch_page" in names and "search_api" in names
    assert "create_ticket" not in names
    broad = mcp.relevant_tools(_descs(), "file a ticket and read a file "
                               "and search docs and list projects and more")
    assert len(broad) <= 4


def test_relevant_tools_empty_task_returns_nothing():
    assert mcp.relevant_tools(_descs(), "") == []


def test_compact_line_shape():
    line = mcp.compact_line(_descs()[0])
    assert line.startswith("  docs.search_api(")
    assert "-- search the project documentation" in line


# ------------------------------------------------------------- the schema
def test_mcp_call_schema_gates_on_relevance():
    # fake_mcp.py serves one tool, "echo", described "echo the arguments
    # back": only a task naming it is relevant enough to open the gate.
    cfg = _named_cfg(_task_text="please echo the arguments back",
                     _root=None)
    schema = tools.schema_for(cfg)
    enum = schema["properties"]["action"]["enum"]
    assert "mcp_call" in enum
    for field in ("server", "tool", "arguments"):
        assert field in schema["properties"]
        assert field in schema["required"]
        assert field in list(schema["properties"])   # ordered field set


def test_mcp_call_absent_for_unrelated_task_and_plain_config():
    assert "mcp_call" not in tools.schema_for(Config())["properties"]["action"]["enum"]
    cfg = _named_cfg(_task_text="rename the widget class in app.py")
    enum = tools.schema_for(cfg)["properties"]["action"]["enum"]
    assert "mcp_call" not in enum


def test_default_schema_stays_byte_identical():
    """The acceleration and mcp_call additions must not shift the schema a
    warm prompt cache was built for: with default config the fields server/
    tool/arguments are absent entirely."""
    plain = tools.schema_for(Config())
    assert not {"server", "tool", "arguments"} & set(plain["properties"])


# ------------------------------------------------------------ injection
def test_mcp_injection_lists_only_relevant_tools():
    cfg = _named_cfg(_task_text="echo something back please")
    notes = tools.mcp_injection(cfg)
    assert "action=mcp_call" in notes
    assert "fake.echo" in notes


def test_mcp_injection_empty_when_irrelevant():
    cfg = _named_cfg(_task_text="rename the widget class")
    assert tools.mcp_injection(cfg) == ""


# ------------------------------------------------------ permission broker
def _hook_result(allowed, message=""):
    seen = []

    def hook(info):
        seen.append(info)
        return allowed, message

    return hook, seen


@pytest.fixture
def _broker():
    security.install_permission_hook(None)
    yield
    security.install_permission_hook(None)


def test_untrusted_call_asks_the_broker(_broker):
    cfg = _named_cfg()
    hook, seen = _hook_result(True)
    security.install_permission_hook(hook)
    obs = tools.tool_mcp_call(None, cfg, "fake", "echo", '{"t": 1}')
    assert obs.startswith("echo:")
    info = seen[0]
    assert info["category"] == "mcp_tool"
    assert info["action_summary"] == "Run MCP tool fake.echo"
    assert json.loads(info["detail"]) == {"server": "fake", "tool": "echo",
                                          "arguments": {"t": 1}}
    assert info["key"] == "fake.echo"


def test_denied_untrusted_call_returns_the_fixed_text(_broker):
    cfg = _named_cfg()
    hook, _seen = _hook_result(False)
    security.install_permission_hook(hook)
    obs = tools.tool_mcp_call(None, cfg, "fake", "echo", "{}")
    assert obs == ("[PERMISSION DENIED BY USER: MCP tool execution was "
                   "declined. Proceed using local repository tools only.]")


def test_no_broker_denies_in_the_safe_direction(_broker):
    cfg = _named_cfg()
    obs = tools.tool_mcp_call(None, cfg, "fake", "echo", "{}")
    assert obs.startswith("[PERMISSION DENIED BY USER")


def test_trusted_server_runs_without_asking(_broker):
    cfg = _named_cfg()
    cfg.mcp_servers = {"fake": {"command": sys.executable,
                                "args": [str(FAKE)], "trusted": True}}
    asked = []
    security.install_permission_hook(
        lambda info: asked.append(info) or (True, ""))
    obs = tools.tool_mcp_call(None, cfg, "fake", "echo", '{"t": 1}')
    assert obs.startswith("echo:")
    assert asked == []


def test_mcp_call_disabled_and_unconfigured_are_errors(_broker):
    cfg = _named_cfg()
    cfg.enable_mcp = False
    assert tools.tool_mcp_call(None, cfg, "fake", "echo", "{}").startswith(
        "ERROR")
    assert tools.tool_mcp_call(None, Config(), "fake", "echo", "{}")\
        .startswith("ERROR")


def test_dispatch_routes_mcp_call(_broker):
    cfg = _named_cfg()
    cfg.mcp_servers = {"fake": {"command": sys.executable,
                                "args": [str(FAKE)], "trusted": True}}
    from pedacito.index import Index
    obs = tools.dispatch(Index(Path.cwd(), []), cfg,
                         {"action": "mcp_call", "server": "fake",
                          "tool": "echo", "arguments": '{"x": 2}'})
    assert obs.startswith("echo:") and '"x": 2' in obs


def test_bad_arguments_json_is_refused(_broker):
    cfg = _named_cfg()
    obs = tools.tool_mcp_call(None, cfg, "fake", "echo", "not json")
    assert obs.startswith("ERROR")


# ------------------------------------------------------- HTTP transport
class _FakeResp:
    def __init__(self, body, ctype="application/json", sid=""):
        self._body = body.encode()
        self.headers = {"Content-Type": ctype}
        if sid:
            self.headers["Mcp-Session-Id"] = sid

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_http_session_initialize_and_call(monkeypatch):
    sent = []

    class Req:
        def __init__(self, url, data=None, headers=None, method=None):
            self.url, self.data, self.headers = url, data, headers or {}

    def fake_urlopen(req, timeout=None):
        msg = json.loads(req.data.decode())
        sent.append((msg, dict(req.headers)))
        if msg["method"] == "initialize":
            body = json.dumps({"jsonrpc": "2.0", "id": msg["id"],
                               "result": {"protocolVersion": mcp
                                          .PROTOCOL_VERSION}})
            return _FakeResp(body, sid="sess-1")
        if msg["method"] == "notifications/initialized":
            return _FakeResp("", sid="")
        if msg["method"] == "tools/call":
            body = json.dumps({"jsonrpc": "2.0", "id": msg["id"],
                               "result": {"content": [
                                   {"type": "text", "text": "http-ok"}]}})
            return _FakeResp(body)
        raise AssertionError(msg["method"])

    monkeypatch.setattr(mcp.urllib.request, "Request", Req)
    monkeypatch.setattr(mcp.urllib.request, "urlopen", fake_urlopen)
    s = mcp._HttpSession("http://127.0.0.1:9/mcp", 5)
    s.initialize()
    out = s.request("tools/call", {"name": "x", "arguments": {}})
    assert out["content"][0]["text"] == "http-ok"
    # the session id from initialize is echoed on every later request
    assert sent[2][1]["Mcp-Session-Id"] == "sess-1"


def test_http_session_sse_reply(monkeypatch):
    class Req:
        def __init__(self, url, data=None, headers=None, method=None):
            self.data = data

    def fake_urlopen(req, timeout=None):
        msg = json.loads(req.data.decode())
        body = ("event: message\n"
                "data: " + json.dumps({"jsonrpc": "2.0", "method":
                                       "notifications/message"}) + "\n"
                "data: " + json.dumps({"jsonrpc": "2.0", "id": msg["id"],
                                       "result": {"ok": 1}}) + "\n")
        return _FakeResp(body, ctype="text/event-stream")

    monkeypatch.setattr(mcp.urllib.request, "Request", Req)
    monkeypatch.setattr(mcp.urllib.request, "urlopen", fake_urlopen)
    s = mcp._HttpSession("http://127.0.0.1:9/mcp", 5)
    assert s.request("ping") == {"ok": 1}


def test_http_named_server_listing(monkeypatch):
    class Req:
        def __init__(self, url, data=None, headers=None, method=None):
            self.data = data

    def fake_urlopen(req, timeout=None):
        msg = json.loads(req.data.decode())
        result = {}
        if msg["method"] == "tools/list":
            result = {"tools": [{"name": "t1", "description": "one",
                                 "inputSchema": {"type": "object"}}]}
        body = json.dumps({"jsonrpc": "2.0", "id": msg.get("id"),
                           "result": result})
        return _FakeResp(body)

    monkeypatch.setattr(mcp.urllib.request, "Request", Req)
    monkeypatch.setattr(mcp.urllib.request, "urlopen", fake_urlopen)
    cfg = Config()
    cfg.mcp_servers = {"api": {"url": "http://127.0.0.1:9/mcp"}}
    descs = mcp.list_mcp_tools(cfg)
    assert descs[0]["name"] == "t1" and descs[0]["server"] == "api"


# ---------------------------------------------------------------- layers
def test_child_config_disables_named_mcp():
    from pedacito import subagent
    cfg = _named_cfg()
    child = subagent.child_config(cfg)
    assert child.enable_mcp is False


def test_gui_field_meta_labels():
    meta = S.FIELD_META
    assert meta["enable_prompt_cache"].label == \
        "Enable Local Server KV Prompt Caching (cache_prompt)"
    assert meta["enable_prompt_lookup"].label == \
        "Enable N-Gram Prompt Lookup Speculation"
    assert meta["prompt_lookup_num_tokens"].label == \
        "Prompt Lookup Draft Tokens"
    names = {item["name"]: item for item in S.schema()}
    for k in ("enable_prompt_cache", "enable_prompt_lookup",
              "prompt_lookup_num_tokens", "enable_mcp"):
        assert k in names


def test_gui_route_lists_named_tools(monkeypatch):
    from pedacito.gui import server as gs
    cfg = _named_cfg()
    with gs.STATE.lock:
        old_cfg = gs.STATE.cfg
        gs.STATE.cfg = cfg
    try:
        status, payload = gs._mcp_tools({}, None)
    finally:
        with gs.STATE.lock:
            gs.STATE.cfg = old_cfg
    assert status == 200
    assert payload["enabled"] is True and "fake" in payload["servers"]
    assert any(t["name"] == "echo" for t in payload["tools"])


def test_cli_mcp_list(tmp_path, capsys, monkeypatch):
    from pedacito.cli import main
    (tmp_path / "srv.py").write_text(FAKE.read_text())
    user = tmp_path / "cfg.toml"
    user.write_text(
        '[server]\nbase_url = "http://127.0.0.1:1/v1"\nmodel = "m"\n'
        # json.dumps gives a TOML basic string with backslashes escaped:
        # a raw Windows path (C:\...) is invalid TOML ("Unescaped '\'").
        f'[mcp_servers.fake]\ncommand = {json.dumps(sys.executable)}\n'
        f'args = [{json.dumps(str(FAKE))}]\n')
    monkeypatch.setattr(config_mod, "USER_CONFIG", user)
    rc = main(["mcp", "list", "--root", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "fake.echo" in out and "asks permission" in out


def test_cli_mcp_list_without_servers(tmp_path, capsys, monkeypatch):
    from pedacito.cli import main
    monkeypatch.setattr(config_mod, "USER_CONFIG", tmp_path / "none.toml")
    rc = main(["mcp", "list", "--root", str(tmp_path)])
    assert rc == 0
    assert "No named MCP servers" in capsys.readouterr().out
