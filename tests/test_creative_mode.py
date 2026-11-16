"""Task-Adaptive Creative/Builder Mode, hosted sampling restraint, Live Preview.

D1 -- hosted provider APIs tune sampling per model themselves: pedacito omits
temperature/top_p/top_k/min_p/repeat_penalty unless the user opts in
(send_temperature_to_hosted) or pins one (hosted_temperature). Local servers
keep every configured value, and a repetition retry gets one gentle
presence_penalty nudge instead of a cold temperature.

D2 -- creative/design tasks (auto-detected, or forced with creative_mode) run
the answer warm (>= 0.6 local), with a >= 8192-token reply floor, the Senior
Product Engineer & Visual Designer persona, and a whole-rewrite bypass for the
fragment veto and the small-window oversized-rewrite brake. The wipe refusal
and the JS export-surface guard are never bypassed.

D3 -- the GUI Live Preview tab is fed by the read-only /preview/<rel> route:
directory-contained, correct MIME, framable only by the app's own origin, and
never serving sensitive files."""
# ruff: noqa: F811  (the `gui` fixture is re-exported from test_gui)
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from a1_support import make_project  # noqa: E402
from pedacito import edits, prompts, verify  # noqa: E402
from pedacito.agent import Agent  # noqa: E402
from pedacito.config import Config  # noqa: E402
from pedacito.gui import core  # noqa: E402
from pedacito.llm import LLMClient  # noqa: E402
from test_gui import gui  # noqa: F401,E402  (the shared server fixture)
from test_llm_transport import _cfg, _ok, _patch_post  # noqa: E402
from test_phase_temperature import TempClient  # noqa: E402

HOSTED = "https://api.example.com/v1"
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "repeat_penalty")
SAMPLER = {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.05,
           "repeat_penalty": 1.1, "seed": 7}


# ============================================================ D1: hosted APIs
def _record(monkeypatch, cfg):
    posts = []
    _patch_post(monkeypatch, lambda *a, **k: posts.append(k["json"]) or _ok())
    return LLMClient(cfg), posts


def test_hosted_requests_omit_every_sampling_key(monkeypatch):
    client, posts = _record(monkeypatch, _cfg(base_url=HOSTED, sampler=SAMPLER))
    client.chat([{"role": "user", "content": "x"}], 10)
    p = posts[-1]
    assert all(k not in p for k in SAMPLING_KEYS)
    assert p["seed"] == 7                  # non-sampling keys still pass through


def test_hosted_opt_in_sends_temperature_only(monkeypatch):
    client, posts = _record(monkeypatch, _cfg(
        base_url=HOSTED, send_temperature_to_hosted=True,
        temperature=0.4, sampler=SAMPLER))
    client.chat([{"role": "user", "content": "x"}], 10)
    p = posts[-1]
    assert p["temperature"] == 0.4
    assert all(k not in p for k in SAMPLING_KEYS if k != "temperature")


def test_hosted_pinned_temperature_beats_phase_and_sampler(monkeypatch):
    client, posts = _record(monkeypatch, _cfg(
        base_url=HOSTED, hosted_temperature=1.0, sampler=SAMPLER))
    with client.temperature(0.0):          # a phase override must not win
        client.chat([{"role": "user", "content": "x"}], 10)
    client.chat([{"role": "user", "content": "x"}], 10)
    assert [p["temperature"] for p in posts] == [1.0, 1.0]
    assert all("top_k" not in p for p in posts)


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8080/v1", "http://localhost:8080/v1",
    "http://192.168.1.50:8080/v1", "http://10.0.0.7:8080/v1",
])
def test_local_endpoints_keep_the_configured_sampling(monkeypatch, url):
    client, posts = _record(monkeypatch, _cfg(base_url=url, sampler=SAMPLER))
    client.chat([{"role": "user", "content": "x"}], 10)
    p = posts[-1]
    assert p["temperature"] == 0.7         # the sampler table's own value
    assert p["top_k"] == 40 and p["repeat_penalty"] == 1.1
    assert p["top_p"] == 0.9 and p["min_p"] == 0.05


def test_repetition_nudge_is_gentle_local_and_one_shot(monkeypatch):
    client, posts = _record(monkeypatch, _cfg())
    client._nudge_presence = True
    client.chat([{"role": "user", "content": "x"}], 10)
    client.chat([{"role": "user", "content": "x"}], 10)
    assert posts[0]["presence_penalty"] == 0.15
    assert "presence_penalty" not in posts[1]
    assert client._nudge_presence is False
    # an existing stronger penalty is never lowered
    client2, posts2 = _record(monkeypatch, _cfg(sampler={"presence_penalty": 0.3}))
    client2._nudge_presence = True
    client2.chat([{"role": "user", "content": "x"}], 10)
    assert posts2[0]["presence_penalty"] == 0.3


def test_hosted_gets_no_nudge_key(monkeypatch):
    client, posts = _record(monkeypatch, _cfg(base_url=HOSTED))
    client._nudge_presence = True
    client.chat([{"role": "user", "content": "x"}], 10)
    assert "presence_penalty" not in posts[0]
    assert client._nudge_presence is False  # consumed, not carried over


def test_json_mode_probes_omit_hosted_temperature(monkeypatch):
    client, posts = _record(monkeypatch, _cfg(base_url=HOSTED, json_mode="object"))
    client.probe_json_modes()
    assert len(posts) == 3
    assert all("temperature" not in p for p in posts)
    local, lposts = _record(monkeypatch, _cfg(json_mode="object"))
    local.probe_json_modes()
    assert all("temperature" in p for p in lposts)


# ======================================================= D2: task detection
@pytest.mark.parametrize("task", [
    "build a website for my bakery",
    "create a landing page from scratch",
    "restyle the dashboard",
    "design a portfolio hero section",
    "fix the ui bug in the login form",
    "polish styles.css",
    "redesign the pricing page",
])
def test_design_tasks_are_detected(task):
    assert prompts.is_creative_or_design_task(task) is True


@pytest.mark.parametrize("task", [
    "build a parser for ini files",
    "refactor the database layer",
    "add a retry to the sync client",
    "the assigned task is unclear",
    "update the build pipeline",
    "fix the off-by-one in bulk_discount",
])
def test_plain_code_tasks_are_not_creative(task):
    assert prompts.is_creative_or_design_task(task) is False


def test_ui_file_targets_mark_a_task_creative():
    assert prompts.is_creative_or_design_task("tweak it", ["src/App.tsx"])
    assert prompts.is_creative_or_design_task("tweak it", ["web/index.html"])
    assert not prompts.is_creative_or_design_task("tweak it", ["main.py"])


@pytest.mark.parametrize("rel", ["page.html", "site.htm", "style.css", "NOTES.md"])
def test_hand_authored_files_always_allow_whole_rewrites(rel):
    assert prompts.whole_rewrite_allowed("", rel) is True


def test_whole_rewrite_allowed_for_creative_and_restructure_briefs():
    assert prompts.whole_rewrite_allowed("redesign the dashboard", "big.py")
    assert prompts.whole_rewrite_allowed("reorganize the module", "big.py")
    assert prompts.whole_rewrite_allowed("replace the config loader", "big.py")
    assert not prompts.whole_rewrite_allowed("fix the bug", "big.py")
    # "rewrite"/"from scratch" stay with the small-window brake's warn path
    assert not prompts.whole_rewrite_allowed("rewrite big.py from scratch", "big.py")


# ============================================ D2: warm, roomy, designer answers
class RecClient(TempClient):
    """TempClient that also records each chat's messages and reply budget."""

    def __init__(self, cfg, replies=(), steps=()):
        super().__init__(cfg, replies, steps)
        self.chats: list[list[dict]] = []
        self.budgets: list[int] = []

    def chat(self, messages, max_tokens, stream=False, **kw):
        self.chats.append([dict(m) for m in messages])
        self.budgets.append(max_tokens)
        return super().chat(messages, max_tokens, stream=stream, **kw)


def _agent(tmp_path, replies=("done",), steps=(), **cfg_attrs):
    _root, idx, cfg = make_project(
        tmp_path, **{"post_edit_check": False, **cfg_attrs})
    client = RecClient(cfg, replies=list(replies), steps=list(steps))
    return Agent(idx, client, cfg), client, cfg


def test_creative_answer_is_warm_roomy_and_designer_briefed(tmp_path):
    agent, client, cfg = _agent(tmp_path)
    agent.answer("redesign the landing page with a bold hero", "src", stream=False)
    assert agent._creative_task is True
    assert [t for s, t in client.log if s == "answer"] == [0.6]
    assert client.budgets == [max(8192, cfg.max_tokens_answer)]
    system = client.chats[0][0]["content"]
    assert "Senior Product Engineer & Visual Designer" in system
    assert "MAY rewrite a whole" in system          # the Builder brief
    assert "Do not output the whole file" not in system


def test_plain_answer_keeps_cool_temperature_and_configured_budget(tmp_path):
    agent, client, cfg = _agent(tmp_path)
    agent.answer("Fix bulk_discount", "src", stream=False)
    assert agent._creative_task is False
    assert [t for s, t in client.log if s == "answer"] == [0.3]
    assert client.budgets == [cfg.max_tokens_answer]  # byte-identical budget
    system = client.chats[0][0]["content"]
    assert "code assistant" in system
    assert "Do not output the whole file" in system


def test_creative_mode_forces_every_task_and_respects_the_cap(tmp_path):
    agent, client, _cfg = _agent(tmp_path, creative_mode=True,
                                 request_max_tokens_cap=4096)
    agent.answer("Fix bulk_discount", "src", stream=False)
    assert agent._creative_task is True
    assert [t for s, t in client.log if s == "answer"] == [0.6]
    assert client.budgets == [4096]        # the provider cap still bounds it


def test_creative_floor_never_lowers_a_bigger_budget(tmp_path):
    agent, client, _cfg = _agent(tmp_path, max_tokens_answer=12000)
    agent.answer("build a website for my bakery", "src", stream=False)
    assert client.budgets == [12000]


def test_creative_auto_off_keeps_the_surgical_discipline(tmp_path):
    agent, client, cfg = _agent(tmp_path, creative_auto=False)
    agent.answer("redesign the landing page", "src", stream=False)
    assert agent._creative_task is False
    assert [t for s, t in client.log if s == "answer"] == [0.3]
    assert client.budgets == [cfg.max_tokens_answer]


def test_creative_warmth_is_local_only(tmp_path):
    agent, client, _cfg = _agent(tmp_path, base_url=HOSTED)
    agent.answer("redesign the landing page", "src", stream=False)
    assert agent._creative_task is True    # persona/brief still apply ...
    assert [t for s, t in client.log if s == "answer"] == [0.3]  # ... no warmth


def test_gather_marks_the_task_creative(tmp_path):
    agent, _client, _cfg = _agent(tmp_path, steps=[{"action": "done"}])
    agent.gather("redesign the landing page", verbose=False)
    assert agent._creative_task is True


# ================================================== D2: the brakes stand down
def _window_requote():
    """A 40-line file written back as its first 15 lines: the partial-read
    signature the fragment veto exists for."""
    before = "\n".join(f"line {i}" for i in range(40)) + "\n"
    after = "\n".join(f"line {i}" for i in range(15)) + "\n"
    return before, after


def test_fragment_veto_bypassed_for_creative_tasks():
    before, after = _window_requote()
    why = edits.fragment_refusal("mod.py", before, after, task="fix the bug")
    assert why and "refusing to overwrite" in why
    assert edits.fragment_refusal(
        "mod.py", before, after, task="redesign the dashboard") is None


@pytest.mark.parametrize("rel", ["page.html", "site.htm", "style.css", "NOTES.md"])
def test_fragment_veto_bypassed_for_hand_authored_files(rel):
    before, after = _window_requote()
    assert edits.fragment_refusal(rel, before, after, task="") is None


def test_fragment_veto_bypassed_for_restructure_briefs():
    before, after = _window_requote()
    assert edits.fragment_refusal(
        "mod.py", before, after, task="reorganize the module") is None


def test_wipe_refusal_is_never_bypassed():
    before, _after = _window_requote()
    why = edits.fragment_refusal("page.html", before, "",
                                 task="redesign the landing page")
    assert why and "erase the whole file" in why


def test_export_guard_is_never_bypassed():
    before = "export function App() { return <h1>hi</h1>; }\n"
    after = "function App() { return <h1>hi</h1>; }\n"
    why = edits.fragment_refusal("App.jsx", before, after,
                                 task="redesign the component")
    assert why and "export" in why


def _small_agent():
    cfg = Config()
    cfg.small_context_tokens = 16384
    cfg.context_tokens = 8192
    cfg.max_whole_file_lines = 120
    return SimpleNamespace(cfg=cfg)


def _whole_result(rel, lines):
    before = "\n".join(f"line{i}" for i in range(lines))
    return edits.Result(True, rel, "applied", before=before, whole_file=True)


def test_oversized_brake_bypassed_for_pages_even_over_300_lines():
    # the old exemption capped pages at 300 lines; a 400-line redesign passes
    r = _whole_result("page.html", 400)
    assert verify.check_oversized_rewrite(_small_agent(), [r], task="") == []


def test_oversized_brake_bypassed_for_creative_and_restructure_tasks():
    r = _whole_result("big.py", 200)
    assert verify.check_oversized_rewrite(
        _small_agent(), [r], task="redesign the dashboard") == []
    assert verify.check_oversized_rewrite(
        _small_agent(), [r], task="restructure big.py") == []


def test_oversized_brake_still_gates_plain_code():
    r = _whole_result("big.py", 200)
    found = verify.check_oversized_rewrite(_small_agent(), [r], task="fix the bug")
    assert found and found[0].severity == "error"
    found = verify.check_oversized_rewrite(
        _small_agent(), [r], task="rewrite big.py from scratch")
    assert found and found[0].severity == "warn"


# ======================================================= D3: GUI Live Preview
def _preview_project(tmp_path):
    root = tmp_path / "site"
    (root / "sub").mkdir(parents=True)
    (root / "index.html").write_text(
        "<!doctype html><title>Home</title><h1>Hi</h1>", encoding="utf-8")
    (root / "sub" / "page.html").write_text("<p>deep</p>", encoding="utf-8")
    (root / "style.css").write_text("body { color: red; }", encoding="utf-8")
    (root / "secrets.json").write_text('{"key": "x"}', encoding="utf-8")
    core.STATE.root = root
    return root


def _headers(resp):
    return {k.lower(): v for k, v in resp[1].items()}


def test_preview_lists_only_html_pages(gui, tmp_path):
    _preview_project(tmp_path)
    status, body = gui.get("/api/preview")
    assert status == 200
    assert body["files"] == ["index.html", "sub/page.html"]


def test_preview_serves_pages_with_framing_headers_and_mime(gui, tmp_path):
    root = _preview_project(tmp_path)
    resp = gui.req("GET", "/preview/index.html")
    assert resp[0] == 200
    h = _headers(resp)
    assert h["content-type"] == "text/html; charset=utf-8"
    assert h["x-frame-options"] == "SAMEORIGIN"
    assert "frame-ancestors 'self'" in h["content-security-policy"]
    assert resp[2] == (root / "index.html").read_bytes()
    # nested pages and relative assets resolve through the same route
    assert gui.req("GET", "/preview/sub/page.html")[0] == 200
    css = gui.req("GET", "/preview/style.css")
    assert css[0] == 200
    assert _headers(css)["content-type"] == "text/css; charset=utf-8"


def test_preview_refuses_traversal_symlinks_and_absolute_paths(gui, tmp_path):
    root = _preview_project(tmp_path)
    outside = tmp_path / "outside.html"
    outside.write_text("<p>outside</p>", encoding="utf-8")
    (root / "link.html").symlink_to(outside)
    for path in ("/preview/../outside.html",
                 "/preview/%2e%2e%2foutside.html",
                 "/preview/link.html",
                 "/preview//etc/passwd"):
        assert gui.req("GET", path)[0] == 404, path
    assert outside.read_text(encoding="utf-8") == "<p>outside</p>"


def test_preview_never_serves_sensitive_files(gui, tmp_path):
    _preview_project(tmp_path)
    assert gui.req("GET", "/preview/secrets.json")[0] == 403


def test_preview_missing_file_and_no_project(gui, tmp_path):
    _preview_project(tmp_path)
    assert gui.req("GET", "/preview/nope.html")[0] == 404
    core.STATE.root = None
    status, body = gui.get("/api/preview")
    assert status == 200 and body["files"] == []
    assert gui.req("GET", "/preview/index.html")[0] == 404