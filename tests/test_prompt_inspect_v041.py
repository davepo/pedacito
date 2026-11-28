"""WP-4104A: prompt budget inspector.

Offline tests for the report contract, the capture boundaries, the CLI
explain/compare surface, and the read-only GUI route. Every assertion checks
metadata (IDs, hashes, counts, dispositions) -- never stored prompt text.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import threading
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import (  # noqa: E402
    agent as agent_mod,
    index as index_mod,
    llm,
    prompt_inspect,
    prompts,
    tools,
)
from pedacito.cli import main  # noqa: E402
from pedacito.config import Config  # noqa: E402
from pedacito.gui import server  # noqa: E402


# --------------------------------------------------------------- fixtures ---

class Scripted:
    """A tiny client: chat_json returns scripted steps, chat one reply."""

    step_truncated = False
    stream_truncated = False

    def __init__(self, steps=(), replies=("ok",)):
        self.steps = list(steps)
        self.replies = list(replies)
        self.json_calls: list[list[dict]] = []
        self.chat_calls: list[list[dict]] = []
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def chat_json(self, messages, schema, max_tokens, **kw):
        self.json_calls.append([dict(m) for m in messages])
        step = self.steps.pop(0) if self.steps else {"action": "done"}
        return {"reasoning": "r", **step}

    def chat(self, messages, max_tokens, stream=False, **kw):
        self.chat_calls.append([dict(m) for m in messages])
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def _agent(tmp_path, client, files=None, **cfg_kw):
    for rel, text in (files or {}).items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    cfg = Config()
    cfg.backup = False
    cfg.log_events = False
    cfg.max_gathered_chars = 10 ** 7
    for key, value in cfg_kw.items():
        setattr(cfg, key, value)
    idx = index_mod.build([str(tmp_path)], cfg, client=None, summarise=False,
                          root=tmp_path, verbose=False)
    return agent_mod.Agent(idx, client, cfg)


def _seed_report(root: Path, *, model: str = "m", run_id: str | None = None,
                 session_id: str = "s1",
                 sections=None,
                 prefix_hash: str = "a" * 64,
                 pruned: dict | None = None) -> prompt_inspect.PromptReport:
    cap = prompt_inspect.PromptCapture(
        model=model, root=root, run_id=run_id, session_id=session_id,
        schema_actions=("read", "done"))
    for name, text, disposition, reason in (sections or [(
            "system:gather", "SYSTEM TEXT", "kept", "pack=full")]):
        cap.add(name, text, disposition=disposition, reason=reason)
    cap.prefix_message_count = 2
    cap.prefix_hash = prefix_hash
    for key, value in (pruned or {}).items():
        cap.set_pruned(key, value)
    report = cap.report()
    prompt_inspect.save_report(root, report, session_id=session_id)
    return report


# --------------------------------------------------------------- contracts ---

def test_dataclasses_are_frozen_and_keep_the_exact_fields():
    assert [f.name for f in dataclasses.fields(prompt_inspect.PromptSection)] == [
        "name", "ordinal", "token_count", "count_method", "content_hash",
        "disposition", "reason"]
    assert [f.name for f in dataclasses.fields(prompt_inspect.PromptReport)] == [
        "version", "run_id", "model", "sections", "prefix_message_count",
        "prefix_hash", "schema_actions", "pruned"]
    section = prompt_inspect.PromptSection(
        "s", 0, 1, "estimate:chars-div-3.6", "h", "kept", "")
    report = prompt_inspect.PromptReport(
        1, "pr-1", "m", (section,), 2, "p", (), {})
    with pytest.raises(dataclasses.FrozenInstanceError):
        section.name = "x"       # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.run_id = "x"      # type: ignore[misc]


def test_estimate_never_claims_exact_and_explicit_exact_is_labelled():
    count, method = prompt_inspect.count_text("x" * 36)
    assert count == 10 and method.startswith("estimate:")
    exact, exact_method = prompt_inspect.count_text(
        "", token_count=42, count_method="exact:provider-usage")
    assert (exact, exact_method) == (42, "exact:provider-usage")
    # An "exact" number without a provider label is not accepted as exact.
    count, method = prompt_inspect.count_text(
        "x" * 36, token_count=42, count_method="heuristic")
    assert count == 10 and method.startswith("estimate:")


def test_section_order_and_metadata_are_deterministic():
    cap = prompt_inspect.PromptCapture(model="m")
    cap.add("b", "second", reason="r2")
    cap.add("a", "first", reason="r1", matched=("rule:x", "rule:x",
                                                "skill:y"))
    cap.add("c", None, disposition="disabled", reason="off")
    report = cap.report()
    assert [s.name for s in report.sections] == ["b", "a", "c"]
    assert [s.ordinal for s in report.sections] == [0, 1, 2]
    assert report.sections[1].reason == "r1; matched=rule:x,skill:y"
    assert report.sections[2].token_count is None
    assert report.sections[2].disposition == "disabled"


# --------------------------------------------------------------- storage ---

def test_saved_report_contains_hashes_counts_and_no_prompt_text(tmp_path):
    secret = "SECRET-PROMPT-TEXT-DO-NOT-STORE"
    report = _seed_report(tmp_path, sections=[
        ("system:gather", secret, "kept", "pack=full"),
        ("user0:task", secret + "-2", "kept", "task"),
    ])
    path = prompt_inspect.report_path(tmp_path, report.run_id)
    assert path is not None and path.is_file()
    raw = path.read_text(encoding="utf-8")
    assert secret not in raw
    assert report.sections[0].content_hash in raw
    loaded = prompt_inspect.load_report(tmp_path, report.run_id)
    assert loaded == report


def test_reports_are_session_scoped_and_traversal_is_refused(tmp_path):
    report = _seed_report(tmp_path, session_id="session-a")
    assert prompt_inspect.load_report(
        tmp_path, report.run_id, session_id="session-a") is not None
    assert prompt_inspect.load_report(
        tmp_path, report.run_id, session_id="session-b") is None
    assert prompt_inspect.latest_report(
        tmp_path, session_id="session-b") is None
    assert prompt_inspect.latest_report(
        tmp_path, session_id="session-a").run_id == report.run_id
    for bad in ("../escape", "a/b", "", "..", "x" * 200):
        assert prompt_inspect.valid_run_id(bad) is False
        assert prompt_inspect.report_path(tmp_path, bad) is None


def test_reports_are_capped_and_pruned_per_session(tmp_path):
    for i in range(prompt_inspect.MAX_REPORTS + 3):
        _seed_report(tmp_path, run_id=f"pr-{i:04d}", session_id="session-a")
        _seed_report(tmp_path, run_id=f"pr-{i:04d}b", session_id="session-b")
    dir_a = prompt_inspect.list_reports(tmp_path, session_id="session-a",
                                        limit=100)
    dir_b = prompt_inspect.list_reports(tmp_path, session_id="session-b",
                                        limit=100)
    assert len(dir_a) == prompt_inspect.MAX_REPORTS
    assert len(dir_b) == prompt_inspect.MAX_REPORTS
    assert "pr-0000" not in dir_a
    assert "pr-0000b" not in dir_b


def test_compare_reports_diffs_ids_hashes_and_counts_not_text(tmp_path):
    old = _seed_report(tmp_path, run_id="pr-old", prefix_hash="b" * 64,
                       sections=[
                           ("system:gather", "OLD", "kept", "pack=small"),
                           ("answer:source", "OLD-SOURCE", "kept", "kept"),
                           ("answer:notes", None, "disabled", "none"),
                       ], pruned={"answer:source": "1 item trimmed"})
    new = _seed_report(tmp_path, run_id="pr-new", prefix_hash="c" * 64,
                       sections=[
                           ("system:gather", "NEW", "kept", "pack=full"),
                           ("answer:source", "NEW-SOURCE", "truncated",
                            "items_dropped=3"),
                           ("answer:task", "TASK", "kept", "task"),
                       ], pruned={"answer:source": "3 items trimmed"})
    diff = prompt_inspect.compare_reports(new, old)
    assert diff["prefix"]["changed"] is True
    assert [s["name"] for s in diff["sections"]["added"]] == ["answer:task"]
    assert [s["name"] for s in diff["sections"]["removed"]] == ["answer:notes"]
    changed = {c["name"]: c for c in diff["sections"]["changed"]}
    assert "content_hash" in changed["system:gather"]["fields"]
    assert "disposition" in changed["answer:source"]["fields"]
    assert diff["totals"]["delta"] != 0
    rendered = "\n".join(prompt_inspect.render_diff(diff))
    assert "OLD-SOURCE" not in rendered and "NEW-SOURCE" not in rendered
    # No stored section has a text field, so text cannot leak through compare.
    payload = prompt_inspect.report_to_dict(new)
    assert "OLD" not in json.dumps(payload)
    assert "content_hash" in payload["sections"][0]
    assert "text" not in payload["sections"][0]


# ----------------------------------------------------------- capture hooks ---

def test_gather_and_answer_record_the_real_assembly(tmp_path):
    steps = [{"action": "read", "file": "app.py", "start": 1, "end": 10},
             {"action": "done"}]
    client = Scripted(steps, ["done"])
    agent = _agent(tmp_path, client, {"app.py": "def f():\n    return 1\n\n"
                                       "def g():\n    return f()\n"})
    gathered = agent.gather("change f to return 2", verbose=False)
    agent.answer("change f to return 2", gathered, stream=False)

    report = agent.prompt_report
    assert report is not None
    names = [s.name for s in report.sections]
    assert names[0] == "system:gather"
    assert "user0:map" in names and "user0:task" in names
    assert "answer:system" in names and "answer:source" in names
    assert "answer:task" in names
    # Counts are labelled, never silently token-exact.
    for section in report.sections:
        assert section.count_method.startswith(("estimate:", "exact:"))
    # The report on disk is loaded by run id and has no prompt text.
    loaded = prompt_inspect.load_report(tmp_path, report.run_id)
    assert loaded is not None and loaded.run_id == report.run_id
    raw = prompt_inspect.report_path(tmp_path, report.run_id).read_text(
        encoding="utf-8")
    assert "change f to return 2" not in raw
    assert "def f()" not in raw


def test_capture_adds_no_prompt_text_to_the_first_messages(tmp_path):
    client = Scripted([{"action": "done"}])
    agent = _agent(tmp_path, client, {"app.py": "x = 1\n"})
    agent.gather("noop task", verbose=False)
    system = agent._messages[0]["content"]
    user0 = agent._messages[1]["content"]
    assert "Prompt report" not in system + str(user0)
    assert "content_hash" not in system + str(user0)
    assert "estimate:" not in system + str(user0)
    # The two gather messages the model saw are exactly the stored pair.
    assert client.json_calls[0][:2] == agent._messages[:2]


def test_prompt_helpers_keep_the_same_text_bytes():
    cfg = Config()
    cfg.prompt_pack = "tiny"
    expected_help = prompts._compact_tool_help(cfg)
    assert prompts.tool_help_block(cfg, agent_mod.TOOL_HELP) == expected_help
    tiny = prompts.gather_system(cfg, agent_mod.GATHER_BASE, agent_mod.TOOL_HELP)
    extras = [text for key, text in prompts._EXTRA_BULLETS
              if key in prompts._enabled(cfg)
              and key in ("context", "remember", "git", "question")]
    expected = prompts.with_project_rules(
        prompts._TINY_GATHER.format(
            help=expected_help,
            extra="\n".join(extras + [prompts._TRUST_RULE])), cfg)
    assert tiny == expected


def test_llm_boundary_hashes_serialized_request_and_marks_usage_exact(
        tmp_path, monkeypatch):
    cfg = Config()
    cfg.base_url = "http://127.0.0.1:1234/v1"
    cfg.model = "m"
    client = llm.LLMClient(cfg)
    fake = types.SimpleNamespace(status_code=200, text="{}")
    monkeypatch.setattr(llm.requests, "post",
                        lambda url, **kw: fake)
    capture = prompt_inspect.begin_capture(
        model="m", root=tmp_path, session_id="s1",
        schema_actions=["read"])
    try:
        prompt_inspect.note_request(
            [{"role": "system", "content": "S"},
             {"role": "user", "content": "U"}],
            tools.schema_for(cfg))
        client._http_post("http://127.0.0.1:1234/v1/chat/completions",
                          json={"model": "m", "messages": [
                              {"role": "system", "content": "S"}]})
        client._account({"prompt_tokens": 123, "completion_tokens": 5},
                        None, 0.1, 100)
        # An embedding/vision POST is not the chat request and must not
        # replace its provenance.
        client._http_post("http://127.0.0.1:1234/v1/embeddings",
                          json={"model": "m", "input": ["x"]})
    finally:
        prompt_inspect.clear_current()
    report = capture.report()
    transport = [s for s in report.sections
                 if s.name == prompt_inspect.TRANSPORT_SECTION]
    assert transport and transport[0].disposition == "sent"
    assert transport[0].count_method == "exact:provider-usage"
    assert transport[0].token_count == 123
    assert report.prefix_message_count == 2
    assert report.prefix_hash


# ------------------------------------------------------------------ CLI ---

def _cli(root, *args, capsys):
    code = main(["--lang", "en", "prompt", *args, str(root)])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_explain_last_and_json(tmp_path, capsys):
    report = _seed_report(tmp_path)
    code, out, err = _cli(tmp_path, "--explain", "--last", capsys=capsys)
    assert code == 0
    assert report.run_id in out
    assert "system:gather" in out
    assert "estimate:chars-div-3.6" in out
    code, out, _err = _cli(tmp_path, "--explain", "--last", "--json",
                           capsys=capsys)
    payload = json.loads(out)
    assert payload["ok"] is True
    assert payload["report"]["run_id"] == report.run_id
    assert "SYSTEM TEXT" not in out


def test_cli_compare_prints_metadata_only(tmp_path, capsys):
    old = _seed_report(tmp_path, run_id="pr-20260101-120000-old",
                       sections=[("system:gather", "OLD-SECRET", "kept", "a")])
    new = _seed_report(tmp_path, run_id="pr-20260102-120000-new",
                       sections=[("system:gather", "NEW-SECRET", "kept", "b")])
    code, out, err = _cli(tmp_path, "--explain", "--compare", old.run_id,
                          capsys=capsys)
    assert code == 0, err
    assert new.run_id in out and old.run_id in out
    assert "OLD-SECRET" not in out and "NEW-SECRET" not in out
    assert "changed sections" in out
    code, out, err = _cli(tmp_path, "--explain", "--compare", old.run_id,
                          "--json", capsys=capsys)
    payload = json.loads(out)
    assert code == 0 and payload["diff"]["sections"]["changed"]
    assert payload["compare"]["run_id"] == old.run_id
    assert "OLD-SECRET" not in out and "NEW-SECRET" not in out


def test_cli_missing_report_and_usage_exit_codes(tmp_path, capsys):
    code, _out, err = _cli(tmp_path, "--explain", capsys=capsys)
    assert code == 1
    assert "No prompt report" in err
    code, _out, err = _cli(tmp_path, capsys=capsys)
    assert code == 2
    assert "usage:" in err
    code, _out, err = _cli(tmp_path, "--compare", "../bad", capsys=capsys)
    assert code == 2


# ------------------------------------------------------------------ GUI ---

def test_gui_route_is_read_only_and_session_scoped(tmp_path, monkeypatch):
    report = _seed_report(tmp_path, session_id="session-a")
    state = types.SimpleNamespace(root=tmp_path, session_id="session-a",
                                  lock=threading.RLock())
    monkeypatch.setattr(server, "STATE", state)
    status, payload = server._prompt_report({"run_id": []})
    assert status == 200 and payload["ok"] is True
    assert payload["report"]["run_id"] == report.run_id
    assert payload["report"]["sections"][0]["name"] == "system:gather"
    assert "SYSTEM TEXT" not in json.dumps(payload)

    # Another session sees no report and cannot fetch it by id.
    state.session_id = "session-b"
    status, payload = server._prompt_report({"run_id": []})
    assert status == 200 and payload["report"] is None
    status, payload = server._prompt_report({"run_id": [report.run_id]})
    assert status == 404
    status, payload = server._prompt_report({"run_id": ["../bad"]})
    assert status == 400
