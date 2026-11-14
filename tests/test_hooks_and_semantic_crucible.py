"""Bug Hunt #12 crucible: lifecycle hooks and candidate-first semantic search.

Two opt-in subsystems with very different failure modes. A hook can deny a
tool call, so its exit-status contract and trust gate must be exact. Semantic
search can quietly turn one no-hit grep into an embedding pass over the whole
project, so its candidate cap, RRF fusion, cache and permanent-failure memory
are asserted against a real `/v1/embeddings` endpoint.
"""
from __future__ import annotations

import json
import shlex
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import config, hooks, index, llm, semantic, tools
from pedacito.config import Config


def _hook(root: Path, name: str, body: str) -> str:
    p = root / name
    p.write_text(body, encoding="utf-8")
    return shlex.join([sys.executable, str(p)])


# ------------------------------------------------------------------- hooks

def test_pre_hooks_exit_two_deny_tool_edit_and_run(tmp_path):
    cfg = Config()
    cfg.hook_timeout = 5
    deny = _hook(tmp_path, "deny.py", """\
import json, sys
ev = json.load(sys.stdin)
print("blocked " + ev["event"], file=sys.stderr)
sys.exit(2)
""")
    cfg.hooks = {
        "pre_tool": [deny],
        "pre_edit": [deny],
        "pre_run": [deny],
        "post_tool": [deny],        # exit 2 on post_* is a warning, not a deny
    }
    for event in ("pre_tool", "pre_edit", "pre_run"):
        r = hooks.fire(cfg, tmp_path, event, action="run", files=["a.py"])
        assert r.denied, event
        assert event in r.message, r.message
    post = hooks.fire(cfg, tmp_path, "post_tool", action="run")
    assert not post.denied


def test_json_decision_overrides_and_plain_stdout_is_context(tmp_path):
    cfg = Config()
    cfg.hook_timeout = 5
    cfg.hooks = {
        "pre_edit": [_hook(tmp_path, "json_deny.py", """\
import json, sys
json.load(sys.stdin)
print(json.dumps({"decision": "deny", "message": "secrets are off limits"}))
""")],
        "post_tool": [_hook(tmp_path, "json_ctx.py", """\
import json, sys
json.load(sys.stdin)
print(json.dumps({"context": "lint passed"}))
""")],
        "post_edit": [_hook(tmp_path, "plain.py", """\
import sys
sys.stdin.read()
print("formatted a.py")
""")],
        "post_compact": [_hook(tmp_path, "mixed.py", """\
import json
print("log: compacting")
print(json.dumps({"context": "summary kept"}))
""")],
        "pre_run": [_hook(tmp_path, "json_allow.py", """\
import json
print(json.dumps({"decision": "allow", "context": "runtime guard ready"}))
""")],
    }
    denied = hooks.fire(cfg, tmp_path, "pre_edit", files=["secrets.py"])
    assert denied.denied and denied.message == "secrets are off limits"

    assert hooks.fire(cfg, tmp_path, "post_tool").context == "lint passed"
    assert hooks.fire(cfg, tmp_path, "post_edit", files=["a.py"]).context == \
        "formatted a.py"
    # Log lines precede the JSON object; the last line still parses.
    assert hooks.fire(cfg, tmp_path, "post_compact").context == "summary kept"
    assert hooks.fire(cfg, tmp_path, "pre_run").context == "runtime guard ready"


def test_failing_and_slow_hooks_warn_without_denying(tmp_path, capsys):
    cfg = Config()
    cfg.hook_timeout = 1
    cfg.hooks = {
        "pre_tool": [
            _hook(tmp_path, "bad.py", "import sys; sys.exit(1)"),
            _hook(tmp_path, "slow.py", "import time; time.sleep(5)"),
            "definitely-not-a-real-program-bh12",
        ],
    }
    r = hooks.fire(cfg, tmp_path, "pre_tool", action="run")
    assert not r.denied
    err = capsys.readouterr().err
    assert "exited 1" in err
    assert "stopped" in err


def test_hook_env_is_scrubbed_and_event_is_injected(tmp_path, monkeypatch):
    cfg = Config()
    cfg.hook_timeout = 5
    cfg.env_passthrough = ("PEDASH_KEEP",)
    dump = tmp_path / "hook_env.json"
    cfg.hooks = {"post_tool": [_hook(tmp_path, "env.py", f"""\
import json, os
json.dump(dict(os.environ), open({str(dump)!r}, "w"))
""")]}
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-hook-must-not-see")
    monkeypatch.setenv("PEDASH_KEEP", "yes")
    monkeypatch.setenv("PEDASH_UNLISTED", "no")
    hooks.fire(cfg, tmp_path, "post_tool")
    env = json.loads(dump.read_text(encoding="utf-8"))
    assert "DEEPSEEK_API_KEY" not in env
    assert env.get("PEDASH_KEEP") == "yes"
    assert "PEDASH_UNLISTED" not in env
    assert env.get("PEDACITO_EVENT") == "post_tool"


def test_project_hooks_and_serve_are_ignored_until_trusted(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    cfg_path = root / ".pedacito.toml"
    cfg_path.write_text(
        'hooks = { post_task = ["echo done"] }\n'
        'allow_serve = true\n', encoding="utf-8")
    cfg = config.load(root, require_hardware=False,
                      upgrade_user_config=False, warn_budgets=False)
    assert not cfg.hooks
    assert cfg.allow_serve is False

    config.trust_project(cfg_path)
    cfg = config.load(root, require_hardware=False,
                      upgrade_user_config=False, warn_budgets=False)
    assert cfg.hooks.get("post_task") == ["echo done"]
    assert cfg.allow_serve is True


# ---------------------------------------------------- semantic / embeddings

QUERY = "zzz-nomatch-semantic"


class _EmbedState:
    def __init__(self):
        self.counts: list[int] = []
        self.fail_status = 0
        self.lock = threading.Lock()


def _embed_server(state: _EmbedState):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                payload = {}
            texts = payload.get("input") or []
            if isinstance(texts, str):
                texts = [texts]
            with state.lock:
                state.counts.append(len(texts))
            if state.fail_status:
                body = b'{"error": "embeddings unsupported"}'
                self.send_response(state.fail_status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            data = []
            for i, text in enumerate(texts):
                # The query and units from the deliberately first-sorted file
                # are the only ones with semantic signal.
                vec = [1.0, 0.0] if (text == QUERY or "aaa_semantic" in text) \
                    else [0.0, 1.0]
                data.append({"index": i, "embedding": vec})
            body = json.dumps({"object": "list", "data": data,
                               "model": "mock-embed"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread


@pytest.fixture(autouse=True)
def _fresh_semantic_memory():
    semantic._DISABLED.clear()
    yield
    semantic._DISABLED.clear()


def _fixture_project(root: Path, files: int = 60) -> None:
    aaa = ["def semantic_unit_%d(x):\n    return x + %d\n" % (i, i)
           for i in range(5)]
    (root / "aaa_semantic.py").write_text("\n\n".join(aaa), encoding="utf-8")
    for i in range(files):
        (root / f"mod_{i:02d}.py").write_text(
            f"def plain_unit_{i}(value):\n    return value * {i}\n",
            encoding="utf-8")


def _make_cfg(port: int) -> Config:
    cfg = Config()
    cfg.base_url = f"http://127.0.0.1:{port}/v1"
    cfg.model = "mock-chat"
    cfg.embedding_model = "mock-embed"
    cfg.semantic_search = True
    cfg.semantic_min_score = 0.5
    cfg.semantic_top_k = 5
    cfg.include_references = False
    cfg.timeout = 10
    return cfg


def test_candidate_first_cap_rrf_floor_and_content_sha_cache(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    _fixture_project(root)
    state = _EmbedState()
    httpd, thread = _embed_server(state)
    try:
        cfg = _make_cfg(httpd.server_address[1])
        idx = index.build([str(root)], cfg, client=None, root=root,
                          summarise=False, verbose=False)
        non_module = [c for c in idx.chunks if c.kind != "module"]
        assert len(non_module) > 60, len(non_module)
        client = llm.LLMClient(cfg)

        out = tools.tool_grep(idx, QUERY, cfg=cfg, client=client,
                              scope="")
        # Candidate-first: at most K (50) units plus the query, never the
        # whole project, even though >60 units exist.
        total = sum(state.counts)
        assert 1 < total <= semantic.CANDIDATE_LIMIT + 1, state.counts
        assert total < len(non_module)
        assert len(state.counts) >= 2          # batched units + query
        assert "showing hybrid matches" in out
        assert "aaa_semantic.py" in out
        assert "mod_" not in out, out[:1000]

        # The similarity floor dropped every cos=0 unit; RRF fuses the
        # lexical rank of the candidate list with the cosine rank.
        candidates = tools._hybrid_candidates(idx, QUERY)
        assert len(candidates) == semantic.CANDIDATE_LIMIT
        fused = semantic.hybrid_search(idx, cfg, client, QUERY, candidates)
        assert fused[0][1].file == "aaa_semantic.py"
        assert fused[0][2] == pytest.approx(1.0)
        units = [c for c in candidates
                 if getattr(c, "sha", "") and c.kind != "module"][:50]
        lex_rank = {id(c): i for i, c in enumerate(units)}
        sem_order = sorted((c for c in units if c.file == "aaa_semantic.py"),
                           key=lambda c: lex_rank[id(c)])
        sem_pos = {id(c): i for i, c in enumerate(sem_order)}
        for rrf, chunk, cos in fused:
            assert cos >= cfg.semantic_min_score
            assert chunk.file == "aaa_semantic.py"
            expected = (1.0 / (semantic.RRF_K + lex_rank[id(chunk)] + 1)
                        + 1.0 / (semantic.RRF_K + sem_pos[id(chunk)] + 1))
            assert rrf == pytest.approx(expected)

        # Content-SHA cache: a second different query only embeds itself.
        cache = root / ".pedacito" / semantic.EMBEDDINGS_FILE
        assert cache.is_file()
        state.counts.clear()
        out2 = tools.tool_grep(idx, "yyy-nomatch-semantic", cfg=cfg,
                               client=client, scope="")
        assert state.counts == [1], state.counts
        assert "showing hybrid matches" in out2
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)


def test_transient_rate_limit_is_not_remembered_as_permanent(tmp_path):
    """429/408 mean "try later", not "this server cannot embed". Treating them
    as permanent would disable semantic search for the rest of a run after one
    burst of rate limiting."""
    root = tmp_path / "proj"
    root.mkdir()
    _fixture_project(root, files=10)
    state = _EmbedState()
    state.fail_status = 429
    httpd, thread = _embed_server(state)
    try:
        cfg = _make_cfg(httpd.server_address[1])
        idx = index.build([str(root)], cfg, client=None, root=root,
                          summarise=False, verbose=False)
        client = llm.LLMClient(cfg)
        first = tools.tool_grep(idx, QUERY, cfg=cfg, client=client, scope="")
        assert "semantic search unavailable" in first
        key = (cfg.base_url.rstrip("/"), "mock-embed")
        assert key not in semantic._DISABLED

        before = sum(state.counts)
        second = tools.tool_grep(idx, QUERY, cfg=cfg, client=client, scope="")
        assert "semantic search unavailable" in second
        assert "disabled for this run" not in second
        assert sum(state.counts) > before, "a retryable failure was not retried"
        assert key not in semantic._DISABLED
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)


@pytest.mark.parametrize("status", [404, 501])
def test_permanent_embedding_failure_is_remembered_within_the_run(tmp_path,
                                                                 status):
    root = tmp_path / "proj"
    root.mkdir()
    _fixture_project(root, files=60)
    state = _EmbedState()
    state.fail_status = status     # endpoint/model permanently unsupported
    httpd, thread = _embed_server(state)
    try:
        cfg = _make_cfg(httpd.server_address[1])
        idx = index.build([str(root)], cfg, client=None, root=root,
                          summarise=False, verbose=False)
        client = llm.LLMClient(cfg)

        first = tools.tool_grep(idx, QUERY, cfg=cfg, client=client, scope="")
        assert "semantic search unavailable" in first
        attempts = sum(state.counts)
        assert attempts > 0

        second = tools.tool_grep(idx, "another-nomatch", cfg=cfg,
                                 client=client, scope="")
        assert "semantic search unavailable" in second
        assert "disabled for this run" in second
        # No retry: the failure is remembered for the endpoint+model.
        assert sum(state.counts) == attempts
        assert ("http://127.0.0.1:%d/v1" % httpd.server_address[1],
                "mock-embed") in semantic._DISABLED
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)
