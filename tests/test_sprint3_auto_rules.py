"""Path-scoped rule promotion from the anti-pattern ledger.

A repair lesson confirmed at least twice whose named files all live under
one subdirectory graduates out of the global 8-slot `anti_patterns` array in
`.pedacito/micro_facts.json` into an auto-managed, path-scoped rule file
`.pedacito/rules/auto_<scope>.md` (`paths: <scope>/*`, `auto_generated:
true`, max five deduplicated lesson lines). Promoted lessons reach the model
only appended to a `read`/`context` observation tail -- never the system
prompt -- and self-clean symmetrically with §24 retirement: a lesson whose
files are all deleted loses its line, an auto rule with no lines left is
deleted, and user-authored rule files are never touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import memory
from pedacito.config import Config


def _cfg(**over):
    c = Config()
    c.log_sessions = False
    c.log_events = False
    for k, v in over.items():
        setattr(c, k, v)
    return c


def _doc(root):
    return json.loads((root / ".pedacito" / "micro_facts.json")
                      .read_text(encoding="utf-8"))


def _touch(root, *rels):
    for rel in rels:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x\n", encoding="utf-8")


# ------------------------------------------------------------ promotion ----

def test_twice_confirmed_directory_lesson_promotes_out_of_the_global_ledger(
        tmp_path):
    cfg = _cfg()
    _touch(tmp_path, "src/api/users.py", "src/api/auth.py")
    first = memory.record_anti_pattern(
        tmp_path, "signature_contract",
        "caller and callee disagreed on the argument count",
        ["src/api/users.py"], cfg)
    assert first["confirmations"] == 1
    rows = memory.load_anti_patterns(tmp_path)
    assert len(rows) == 1                      # one confirmation: still global
    block = memory.render_anti_patterns(tmp_path, cfg)
    assert "signature_contract" in block
    again = memory.record_anti_pattern(
        tmp_path, "signature_contract",
        "caller and callee disagreed on the argument count",
        ["src/api/auth.py"], cfg)
    assert again["confirmations"] == 2
    # out of the global 8-slot array...
    assert memory.load_anti_patterns(tmp_path) == []
    doc = _doc(tmp_path)
    assert doc["anti_patterns"] == []
    assert len(doc["promoted"]) == 1
    assert doc["promoted"][0]["promoted_to"] == ".pedacito/rules/auto_src_api.md"
    # ...and no longer injected into the stable prefix...
    assert memory.render_anti_patterns(tmp_path, cfg) == ""
    assert "signature_contract" not in memory.render_anti_patterns(tmp_path, cfg)
    # ...but now visible as a path-scoped rule for the directory it names.
    rule = tmp_path / ".pedacito" / "rules" / "auto_src_api.md"
    text = rule.read_text(encoding="utf-8")
    assert "paths: src/api/*" in text
    assert "auto_generated: true" in text
    assert ("- signature_contract: caller and callee disagreed on the "
            "argument count") in text
    got = memory.rules_for(tmp_path, "src/api/users.py")
    assert got == [(".pedacito/rules/auto_src_api.md",
                    text.split("\n---", 1)[1].strip())]
    assert memory.rules_for(tmp_path, "frontend/app.js") == []


def test_root_scatter_and_single_confirmation_stay_global(tmp_path):
    cfg = _cfg()
    _touch(tmp_path, "a.js", "src/a.js", "docs/b.md", "src/c.js")
    for _ in range(3):
        memory.record_anti_pattern(tmp_path, "js_syntax",
                                   "an edit left a file unparsable",
                                   ["a.js"], cfg)
    memory.record_anti_pattern(tmp_path, "asset_404", "linked a missing asset",
                               ["a.js"], cfg)
    rows = memory.load_anti_patterns(tmp_path)
    assert len(rows) == 2                      # root file: never a scope
    for _ in range(2):
        memory.record_anti_pattern(tmp_path, "dom_id_mismatch",
                                   "queried an id no template renders",
                                   ["src/a.js"], cfg)
    # one confirmation on the docs pair, one dir only seen once in the pair:
    memory.record_anti_pattern(tmp_path, "missing_export",
                               "the name was never exported",
                               ["docs/b.md"], cfg)
    memory.record_anti_pattern(tmp_path, "missing_export",
                               "the name was never exported",
                               ["docs/b.md", "src/c.js"], cfg)
    rows = memory.load_anti_patterns(tmp_path)
    kinds = {r["kind"] for r in rows}
    assert "dom_id_mismatch" not in kinds      # promoted: src/ holds it twice
    assert "missing_export" in kinds           # no shared subdir: stays put
    assert (tmp_path / ".pedacito" / "rules" / "auto_src.md").exists()
    assert list((tmp_path / ".pedacito" / "rules").glob("auto_docs*")) == []


def test_re_recording_a_promoted_lesson_never_resurrects_it_globally(tmp_path):
    cfg = _cfg()
    _touch(tmp_path, "frontend/app.js")
    for _ in range(3):
        memory.record_anti_pattern(tmp_path, "js_syntax",
                                   "an edit left app.js unparsable",
                                   ["frontend/app.js"], cfg)
    assert memory.load_anti_patterns(tmp_path) == []
    assert len(memory.load_promoted_lessons(tmp_path)) == 1
    rule = tmp_path / ".pedacito" / "rules" / "auto_frontend.md"
    assert rule.read_text(encoding="utf-8").count("- js_syntax:") == 1


def test_auto_rule_caps_at_five_lesson_lines_and_dedupes(tmp_path):
    cfg = _cfg()
    _touch(tmp_path, "src/api/a.py", "src/api/b.py")
    for i in range(6):
        for _ in range(2):
            memory.record_anti_pattern(
                tmp_path, f"kind{i}", f"lesson number {i} for the api dir",
                ["src/api/a.py"], cfg)
        memory.record_anti_pattern(
            tmp_path, f"kind{i}", f"lesson number {i} for the api dir",
            ["src/api/b.py"], cfg)      # repeats bump, never duplicate lines
    rule = (tmp_path / ".pedacito" / "rules" / "auto_src_api.md")
    lines = [l for l in rule.read_text(encoding="utf-8").splitlines()
             if l.startswith("- ")]
    assert len(lines) == 5
    assert len(lines) == len(set(lines))
    overflow = memory.load_anti_patterns(tmp_path)
    assert [r["kind"] for r in overflow] == ["kind5"]


# ---------------------------------------------------------- self-cleaning --

def test_promoted_lessons_self_clean_when_their_files_disappear(tmp_path):
    cfg = _cfg()
    _touch(tmp_path, "src/api/a.py", "src/api/b.py", "src/api/c.py")
    for kind, fname in (("js_syntax", "src/api/a.py"),
                        ("asset_404", "src/api/b.py")):
        for _ in range(2):
            memory.record_anti_pattern(tmp_path, kind,
                                       f"the {kind} lesson line",
                                       [fname], cfg)
    rule = tmp_path / ".pedacito" / "rules" / "auto_src_api.md"
    assert len(memory.load_promoted_lessons(tmp_path)) == 2
    # one lesson's file dies: only its line goes, the rule survives
    (tmp_path / "src/api/a.py").unlink()
    memory.load_anti_patterns(tmp_path)        # the pruning entry point
    text = rule.read_text(encoding="utf-8")
    assert "the js_syntax lesson line" not in text
    assert "the asset_404 lesson line" in text
    assert len(memory.load_promoted_lessons(tmp_path)) == 1
    # the last file named by the last lesson dies: line AND file go
    (tmp_path / "src/api/b.py").unlink()
    memory.load_anti_patterns(tmp_path)
    assert not rule.exists()
    assert memory.load_promoted_lessons(tmp_path) == []
    assert _doc(tmp_path)["promoted"] == []


def test_user_authored_rule_files_are_never_touched(tmp_path):
    cfg = _cfg()
    _touch(tmp_path, "src/api/a.py")
    rules = tmp_path / ".pedacito" / "rules"
    rules.mkdir(parents=True)
    clash = rules / "auto_src_api.md"           # same name, human-authored
    clash.write_text("---\npaths: src/api/*\n---\nHUMAN RULE, DO NOT EDIT.\n",
                     encoding="utf-8")
    mine = rules / "style.md"                  # unrelated user rule
    mine.write_text("---\npaths: src/*\n---\nPrefer dataclasses.\n",
                    encoding="utf-8")
    for _ in range(3):
        memory.record_anti_pattern(tmp_path, "js_syntax",
                                   "an edit left a file unparsable",
                                   ["src/api/a.py"], cfg)
    assert clash.read_text(encoding="utf-8") == \
        "---\npaths: src/api/*\n---\nHUMAN RULE, DO NOT EDIT.\n"
    assert mine.read_text(encoding="utf-8") == \
        "---\npaths: src/*\n---\nPrefer dataclasses.\n"
    # no auto_generated flag: the lesson stays in the global ledger
    assert len(memory.load_anti_patterns(tmp_path)) == 1
    # pruning a dead promoted row must also never edit a human file
    doc = _doc(tmp_path)
    doc["promoted"] = [{"lesson_id": "ap:x:1", "kind": "js_syntax",
                        "text": "an edit left a file unparsable",
                        "files": ["src/api/a.py"], "confirmations": 2,
                        "promoted_to": ".pedacito/rules/auto_src_api.md"}]
    (tmp_path / ".pedacito" / "micro_facts.json").write_text(
        json.dumps(doc), encoding="utf-8")
    (tmp_path / "src/api/a.py").unlink()
    memory.load_anti_patterns(tmp_path)
    assert clash.read_text(encoding="utf-8").startswith(
        "---\npaths: src/api/*\n---\nHUMAN RULE")
    assert _doc(tmp_path)["promoted"] == []


def test_micro_facts_ablation_silences_promotion_too(tmp_path):
    cfg = _cfg(micro_facts=False)
    _touch(tmp_path, "src/api/a.py")
    for _ in range(3):
        memory.record_anti_pattern(tmp_path, "js_syntax",
                                   "an edit left a file unparsable",
                                   ["src/api/a.py"], cfg)
    assert memory.load_anti_patterns(tmp_path) == []
    assert not (tmp_path / ".pedacito" / "rules").exists()


# ------------------------------------------------- observation-tail wiring --

from pedacito import index as index_mod                    # noqa: E402
from pedacito.agent import Agent                            # noqa: E402


class Scripted:
    """chat_json pops scripted steps; records every message list it sees."""
    step_truncated = False
    stream_truncated = False

    def __init__(self, steps=(), replies=("ok",)):
        self.steps = list(steps)
        self.replies = list(replies)
        self.json_calls: list[list[dict]] = []
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def chat_json(self, messages, schema, max_tokens, **kw):
        self.json_calls.append([dict(m) for m in messages])
        step = self.steps.pop(0) if self.steps else {"action": "done"}
        return {"reasoning": "r", **step}

    def chat(self, messages, max_tokens, stream=False, **kw):
        self.chat_calls = getattr(self, "chat_calls", [])
        self.chat_calls.append([dict(m) for m in messages])
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


API_FILES = {
    "src/api/users.py": ("def get_user(db, uid):\n"
                         "    return db[uid]\n\n\n"
                         "def put_user(db, uid, row):\n"
                         "    db[uid] = row\n"),
    "src/api/util.py": "def pad(x):\n    return x + '.'\n",
}


def _agent(tmp_path, client, **cfg_kw):
    for rel, text in API_FILES.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    cfg = _cfg(backup=False, max_gathered_chars=10 ** 7,
               stuck_recovery=False, **cfg_kw)
    idx = index_mod.build([str(tmp_path)], cfg, client=None, summarise=False,
                          root=tmp_path, verbose=False)
    return Agent(idx, client, cfg)


def _promote_one_api_lesson(tmp_path):
    cfg = _cfg()
    for _ in range(2):
        memory.record_anti_pattern(
            tmp_path, "signature_contract",
            "the caller disagreed on the argument count -- reconcile in one "
            "step (src/api/users.py)", ["src/api/users.py"], cfg)
    assert (tmp_path / ".pedacito" / "rules"
            / "auto_src_api.md").exists()


def test_promoted_rule_rides_the_read_tail_never_the_system_prompt(tmp_path):
    _promote_one_api_lesson(tmp_path)
    client = Scripted([{"action": "read", "file": "src/api/users.py",
                        "start": 1, "end": 5},
                       {"action": "read", "file": "src/api/util.py",
                        "start": 1, "end": 2},
                       {"action": "done"}])
    agent = _agent(tmp_path, client)
    agent.gather("touch src/api/users.py", verbose=False)
    system_first = client.json_calls[0][0]["content"]
    system_last = client.json_calls[-1][0]["content"]
    assert system_first == system_last         # prefix byte-identical: cached
    assert "signature_contract" not in system_first   # never mid-gather
    tails = [m["content"] for m in client.json_calls[-1] if m["role"] == "user"]
    hits = [t for t in tails
            if "project rule from .pedacito/rules/auto_src_api.md" in t]
    assert len(hits) == 1                      # once per rule per task
    assert "signature_contract" in hits[0]     # at the observation tail


def test_context_file_mode_gets_the_rule_tail_too(tmp_path):
    _promote_one_api_lesson(tmp_path)
    client = Scripted([{"action": "context", "file": "src/api/users.py"},
                       {"action": "done"}])
    agent = _agent(tmp_path, client)
    agent.gather("inspect the users API", verbose=False)
    users = [m["content"] for m in client.json_calls[-1] if m["role"] == "user"]
    assert any("FILE CONTEXT src/api/users.py" in u for u in users)
    assert any("project rule from .pedacito/rules/auto_src_api.md" in u
               and "signature_contract" in u for u in users)


def test_lesson_ledger_is_empty_without_a_project_root():
    # A session with no project root (root=None) still renders the stable
    # prefix; the ledger must answer "no lessons", not raise on Path(None).
    assert memory.load_anti_patterns(None) == []
    assert memory.render_anti_patterns(None) == ""
    assert memory.render_anti_patterns(None, _cfg()) == ""


# ------------------------------------------- effectiveness counters (R5) ----

def _seed_lesson(root, cfg, kind="js_syntax", text="an edit left a file unparsable",
                 files=("a.js",)):
    _touch(root, *files)
    return memory.record_anti_pattern(root, kind, text, list(files), cfg)


def test_run_counters_track_recurrence_and_avoidance(tmp_path):
    cfg = _cfg()
    row = _seed_lesson(tmp_path, cfg)
    memory.note_lesson_run(tmp_path, {"js_syntax"}, cfg)
    memory.note_lesson_run(tmp_path, set(), cfg)
    memory.note_lesson_run(tmp_path, {"asset_404"}, cfg)
    live = memory.load_anti_patterns(tmp_path)
    assert len(live) == 1
    got = live[0]
    assert got["lesson_id"] == row["lesson_id"]
    assert got["runs"] == 3
    assert got["recurred"] == 1      # only the first run tripped its kind
    assert got["avoided"] == 2       # a clean run and an unrelated failure


def test_ineffective_lesson_retires_after_the_threshold(tmp_path):
    cfg = _cfg()
    row = _seed_lesson(tmp_path, cfg)
    for i in range(memory._LESSON_RETIRE_RUNS - 1):
        out = memory.note_lesson_run(tmp_path, {"js_syntax"}, cfg)
        assert out["retired"] == []
        assert len(memory.load_anti_patterns(tmp_path)) == 1
    out = memory.note_lesson_run(tmp_path, {"js_syntax"}, cfg)
    assert out["retired"] == [row["lesson_id"]]
    assert memory.load_anti_patterns(tmp_path) == []
    assert memory.render_anti_patterns(tmp_path, cfg) == ""
    retired = memory.load_retired_lessons(tmp_path)
    assert len(retired) == 1
    assert retired[0]["lesson_id"] == row["lesson_id"]
    assert retired[0]["runs"] == memory._LESSON_RETIRE_RUNS
    assert retired[0]["recurred"] == memory._LESSON_RETIRE_RUNS
    assert retired[0]["avoided"] == 0
    assert retired[0]["retired_at"] > 0
    # the retirement survives another writer's document round-trip
    memory.record_anti_pattern(tmp_path, "asset_404", "another lesson",
                               ["a.js"], cfg)
    assert len(memory.load_retired_lessons(tmp_path)) == 1


def test_one_avoided_run_keeps_a_lesson_past_the_threshold(tmp_path):
    cfg = _cfg()
    _seed_lesson(tmp_path, cfg)
    for _ in range(memory._LESSON_RETIRE_RUNS - 1):
        memory.note_lesson_run(tmp_path, {"js_syntax"}, cfg)
    memory.note_lesson_run(tmp_path, set(), cfg)      # one clean run
    for _ in range(memory._LESSON_RETIRE_RUNS):
        out = memory.note_lesson_run(tmp_path, {"js_syntax"}, cfg)
        assert out["retired"] == []
    live = memory.load_anti_patterns(tmp_path)
    assert len(live) == 1
    assert live[0]["avoided"] == 1
    assert live[0]["runs"] == 2 * memory._LESSON_RETIRE_RUNS


def test_promoted_lessons_are_exempt_from_effectiveness_retirement(tmp_path):
    cfg = _cfg()
    _touch(tmp_path, "src/api/users.py", "src/api/auth.py")
    for _ in range(2):
        memory.record_anti_pattern(
            tmp_path, "signature_contract",
            "caller and callee disagreed on the argument count",
            ["src/api/users.py"], cfg)
    assert memory.load_anti_patterns(tmp_path) == []   # promoted
    for _ in range(memory._LESSON_RETIRE_RUNS + 2):
        out = memory.note_lesson_run(tmp_path, {"signature_contract"}, cfg)
        assert out["retired"] == []
    assert len(memory.load_promoted_lessons(tmp_path)) == 1
    assert memory.load_retired_lessons(tmp_path) == []


def test_only_injected_lessons_are_counted(tmp_path):
    cfg = _cfg()
    old = _seed_lesson(tmp_path, cfg, text="the older injected lesson")
    # a lesson distilled mid-run was not in the run's prompt
    fresh = memory.record_anti_pattern(
        tmp_path, "asset_404", "born during this run", ["a.js"], cfg)
    out = memory.note_lesson_run(tmp_path, {"js_syntax", "asset_404"}, cfg,
                                 injected={old["lesson_id"]})
    assert out["counted"] == 1
    rows = {r["lesson_id"]: r for r in memory.load_anti_patterns(tmp_path)}
    assert rows[old["lesson_id"]]["runs"] == 1
    assert rows[fresh["lesson_id"]]["runs"] == 0


def test_note_lesson_run_respects_the_off_switch_and_missing_root(tmp_path):
    cfg = _cfg(micro_facts=False)
    _seed_lesson(tmp_path, _cfg())
    assert memory.note_lesson_run(tmp_path, {"js_syntax"}, cfg) == {
        "counted": 0, "retired": []}
    assert memory.note_lesson_run(None, {"js_syntax"}, _cfg()) == {
        "counted": 0, "retired": []}
    assert memory.load_anti_patterns(tmp_path)[0]["runs"] == 0
