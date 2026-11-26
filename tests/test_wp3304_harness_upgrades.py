"""WP-3304: six research-backed harness upgrades (roadmap section 2.4).

Covers, in order:
  1. plain-word gather tool format (tool_mode = "plain", arXiv:2611.17169)
  2. shadow-tested best_of + cross-family role ensembling (arXiv:2611.17610)
  3. proactive two-path capability escalation routing (arXiv:2611.16414)
  4. local sparse-MoE execution-direct prompting (arXiv:2611.05919)
  5. deterministic pre-stop verifier on the interleaved `finish`
     (AutoHarness & arXiv:2611.22299)
  6. scratchpad filesystem offloading for oversized tool outputs
     (Meta-Harness, arXiv:2603.28052)

Everything is offline and deterministic: no model server, no network.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import (config as config_mod, modelcard, plan, profiles, roles,
                      safety, tools, workspace)
from pedacito.config import Config


# =========================================================== helpers
def _cfg(**kw):
    cfg = Config()
    cfg.log_events = False
    cfg.log_sessions = False
    cfg.backup = False
    cfg.sources = {}
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _rec(name, fit, tok, ctx, tier="medium", repeats=3, spread=0.0,
         arch_family=""):
    rec = {"name": name, "model": name, "server": "http://x/v1", "tier": tier,
           "active": False, "fit": fit, "spread": spread, "tok_per_s": tok,
           "repeats": repeats, "context_fits": ctx, "status": "ok"}
    if arch_family:
        rec["card_choices"] = {"arch_family": arch_family}
    return rec


class _Idx:
    """Minimal index stand-in for the roles/plan structural gates."""

    def __init__(self, files=None, graph=None):
        self.files = list(files or [])
        self.graph = graph
        self.root = Path(".")


class _Graph:
    def __init__(self, imports=None, dependents=None):
        self.imports = imports or {}
        self.dependents = lambda f: set(dependents.get(f, ())) if dependents else set()


# ==================================================== 1. plain tool mode
class TestPlainToolMode:
    def test_tool_mode_enum_accepts_plain(self):
        assert config_mod._coerce("tool_mode", "plain") == "plain"
        assert config_mod._coerce("tool_mode", "grammar") == "grammar"
        assert config_mod._coerce("tool_mode", "native") == "native"
        with pytest.raises(config_mod.ConfigError):
            config_mod._coerce("tool_mode", "prose")

    def test_parse_plain_step_slash_and_key_value_are_equal(self):
        slash = tools.parse_plain_step(
            "action read / file src/main.py / lines 10 to 40")
        kv = tools.parse_plain_step(
            "action: read\nfile: src/main.py\nstart: 10\nend: 40\n"
            "reasoning: need the loop")
        assert slash == {"action": "read", "file": "src/main.py",
                         "start": 10, "end": 40}
        assert kv == {"action": "read", "file": "src/main.py",
                      "start": 10, "end": 40, "reasoning": "need the loop"}

    def test_parse_plain_step_done_and_ranges(self):
        assert tools.parse_plain_step("action done") == {"action": "done"}
        assert tools.parse_plain_step(
            "action read / file a.py / start 5 / end 9") == \
            {"action": "read", "file": "a.py", "start": 5, "end": 9}
        assert tools.parse_plain_step(
            "action grep / pattern MAX_RETRIES / file src/sync.py") == \
            {"action": "grep", "pattern": "MAX_RETRIES", "file": "src/sync.py"}
        assert tools.parse_plain_step("just some prose, no action") is None
        assert tools.parse_plain_step("") is None

    def test_plain_gather_echoes_through_the_same_step_dict(self):
        # `lines X to Y` must fold into start/end like a JSON step would.
        step = tools.parse_plain_step("action read / file p.py / lines 20 to 25")
        assert step["start"] == 20 and step["end"] == 25 and step["file"] == "p.py"

    def test_choose_selects_plain_only_when_it_beats_baseline(self):
        base = {"tool_choice": 0.5, "json_ok": 0.9, "edit_format": 0.8,
                "instructions": 1.0, "honesty": 1.0, "needle": 1.0,
                "continuation": 1.0}
        # clear gain, no adherence drop -> plain
        m = dict(base, tool_choice_plain=0.7, json_ok_plain=0.9)
        assert modelcard.choose(m, False)["tool_mode"] == "plain"
        # equal accuracy, meaningfully better adherence -> plain
        m = dict(base, tool_choice=0.6, json_ok=0.7,
                 tool_choice_plain=0.6, json_ok_plain=0.95)
        assert modelcard.choose(m, False)["tool_mode"] == "plain"
        # no gain -> stays grammar
        m = dict(base, tool_choice_plain=0.5, json_ok_plain=0.9)
        assert modelcard.choose(m, False)["tool_mode"] == "grammar"
        # native beats plain on a tie
        m = dict(base, tool_choice_native=0.8, json_ok_native=0.9,
                 tool_choice_plain=0.7, json_ok_plain=0.9)
        assert modelcard.choose(m, False)["tool_mode"] == "native"


# ==================================== 2. shadow best_of + cross-family roles
class TestCrossFamilyEnsembling:
    def test_recommend_prefers_a_different_arch_family(self):
        profs = [
            _rec("w", 0.70, 30, 32768, arch_family="qwen"),
            _rec("same", 0.90, 20, 32768, tier="hosted", arch_family="qwen"),
            _rec("diff", 0.82, 20, 32768, tier="hosted", arch_family="gemma"),
        ]
        recs = profiles.recommend(profs)
        assert recs["default_profile"]["name"] == "w"
        for role in ("escalate", "verifier", "planner"):
            # `same` scores higher but shares the worker's family: `diff` wins.
            assert recs[role]["name"] == "diff"
            assert "architecture family" in recs[role]["reason"]

    def test_recommend_unchanged_without_arch_family(self):
        profs = [_rec("w", 0.70, 30, 32768),
                 _rec("h", 0.85, 20, 32768, tier="hosted")]
        recs = profiles.recommend(profs)
        for role in ("escalate", "verifier", "planner"):
            assert recs[role]["name"] == "h"

    def test_recommend_declines_a_weak_specialist_still(self):
        profs = [_rec("w", 0.70, 30, 32768),
                 _rec("h", 0.74, 20, 32768, tier="hosted")]
        assert profiles.recommend(profs)["escalate"].get("name") is None

    def test_roles_prefer_second_opinion_profile(self):
        cfg = _cfg()
        cfg.active_profile = "worker"
        cfg.profiles = {"worker": {"model": "qwen-27b"},
                        "verifier": {"model": "qwen-27b"},
                        "gemma": {"model": "gemma-27b"}}
        card = lambda fam: {"choices": {"arch_family": fam}}
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(roles, "_profile_card",
                            lambda c, n: card("qwen") if n in ("worker", "verifier")
                            else card("gemma"))
        try:
            assert roles.arch_family_of(cfg, "worker") == "qwen"
            assert roles.prefer_second_opinion_profile(cfg, "verifier",
                                                       "verifier") == "gemma"
            # a different-family named profile is left alone
            assert roles.prefer_second_opinion_profile(cfg, "escalate",
                                                       "gemma") == "gemma"
            # non-second-opinion roles are never rewritten
            assert roles.prefer_second_opinion_profile(cfg, "planner",
                                                       "verifier") == "verifier"
        finally:
            monkeypatch.undo()


# =============================================== 3. proactive escalation
class TestProactiveEscalation:
    def test_config_default_is_false(self):
        assert Config().proactive_escalation is False

    def test_config_roundtrip(self):
        cfg = config_mod._coerce("proactive_escalation", True)
        assert cfg is True

    def test_structural_gate_architectural_keyword(self):
        cfg = _cfg(proactive_escalation=True)
        reason = roles.proactive_escalation_reason(
            cfg, "refactor the whole codebase into a layered architecture", _Idx())
        assert "structurally complex" in reason

    def test_structural_gate_multi_file_touch(self):
        cfg = _cfg(proactive_escalation=True)
        idx = _Idx(files=["a.py", "b.py", "c.py", "d.py"])
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr("pedacito.rank.task_files",
                            lambda i, t: ["a.py", "b.py", "c.py", "d.py"])
        try:
            reason = roles.proactive_escalation_reason(
                cfg, "update the pricing module", idx)
            assert "structurally complex" in reason
        finally:
            monkeypatch.undo()

    def test_structural_gate_import_fanout(self):
        cfg = _cfg(proactive_escalation=True)
        imports = {"a.py": {f"m{i}.py" for i in range(10)}}
        graph = _Graph(imports=imports)
        idx = _Idx(files=["a.py"], graph=graph)
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr("pedacito.rank.task_files", lambda i, t: ["a.py"])
        try:
            reason = roles.proactive_escalation_reason(
                cfg, "touch up a.py", idx)
            assert "structurally complex" in reason
        finally:
            monkeypatch.undo()

    def test_off_by_default_and_single_file_stays_local(self):
        cfg = _cfg()  # proactive_escalation defaults False
        assert roles.proactive_escalation_reason(
            cfg, "refactor the whole codebase", _Idx()) == ""
        cfg2 = _cfg(proactive_escalation=True)
        idx = _Idx(files=["a.py"])
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr("pedacito.rank.task_files", lambda i, t: ["a.py"])
        try:
            assert roles.proactive_escalation_reason(
                cfg2, "fix the off-by-one in a.py", idx) == ""
        finally:
            monkeypatch.undo()


# ================================================ 4. sparse-MoE prompting
class TestSparseMoE:
    def test_is_sparse_moe_detection(self):
        assert config_mod.is_sparse_moe(model="qwen3-30b-a3b")
        assert config_mod.is_sparse_moe(model="Mixtral-8x7B")
        assert config_mod.is_sparse_moe(model="some-moe-model")
        assert config_mod.is_sparse_moe(model="deepseek-v4.1-flash")
        assert not config_mod.is_sparse_moe(model="qwen2.5-7b")
        assert not config_mod.is_sparse_moe(model="gemma-3-27b")

    def test_should_plan_sparse_moe_skips_small_threshold(self):
        cfg = _cfg(plan_threshold=2)
        cfg.model = "qwen3-30b-a3b"
        idx = _Idx(files=["a.py"])
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr("pedacito.rank.task_files",
                            lambda i, t: list(i.files))
        try:
            assert plan.should_plan("fix a.py", idx, cfg) is False
            # a truly multi-module task is still planned
            idx2 = _Idx(files=[f"f{i}.py" for i in range(6)])
            assert plan.should_plan("rewrite everything", idx2, cfg) is True
        finally:
            monkeypatch.undo()

    def test_edit_format_skips_plan_brief_for_sparse_moe(self):
        from pedacito import prompts
        cfg = _cfg(edit_plan_brief=True)
        cfg.model = "gemma-3-27b"
        dense = prompts.edit_format(cfg, "Before the blocks, write a short plain "
                                     "explanation of what you changed and why.")
        assert "write 1-3 lines" in dense
        cfg.model = "qwen3-30b-a3b"
        sparse = prompts.edit_format(cfg, "Before the blocks, write a short plain "
                                     "explanation of what you changed and why.")
        assert "write 1-3 lines" not in sparse


# ================================================= 5. pre-stop verifier
class _Scripted:
    """A tiny native tool-call double for the interleaved loop."""

    def __init__(self, steps, loop_last=False):
        self.steps = list(steps)
        self.loop_last = loop_last
        self.requests = []

    def chat_tools(self, messages, tools, max_tokens, **kw):
        n = len(self.requests)
        self.requests.append(copy.deepcopy(messages))
        if n < len(self.steps):
            step = self.steps[n]
        elif self.loop_last:
            step = self.steps[-1]
        else:
            step = [("finish", {"summary": "script exhausted"})]
        return {"content": "", "usage": {"completion_tokens": 5}, "tool_calls": [
            {"id": f"c{n}_{i}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}
            for i, (name, args) in enumerate(step)]}


def _make_agent(tmp_path, client, files=None, **cfg_attrs):
    from pedacito import index as index_mod
    from pedacito.agent import Agent
    root = tmp_path / "proj"
    root.mkdir(parents=True, exist_ok=True)
    base = {"calc.py": "def add(a, b):\n    return a + b\n"}
    for rel, text in (base if files is None else files).items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    cfg = _cfg(**cfg_attrs)
    idx = index_mod.build([str(root)], cfg, client=None, summarise=False,
                          root=root, verbose=False)
    return root, Agent(idx, client, cfg)


def _finish_msgs(res, name="finish"):
    return [m["content"] for m in res.transcript
            if m["role"] == "tool" and m.get("name") == name]


class TestPreStopVerifier:
    def test_zero_edits_on_an_edit_task_is_intercepted(self, tmp_path):
        from pedacito.interleaved import run_interleaved
        client = _Scripted([[("finish", {"summary": "done"}),
                             ("finish", {"summary": "done for real"})]])
        root, agent = _make_agent(tmp_path, client)
        res = run_interleaved(agent, "fix the calc bug", apply=True)
        first = _finish_msgs(res)[0]
        assert first.startswith("NOT FINISHED") and "no file was changed" in first
        # the intercept reset the text nudge budget: still bounded
        assert res.stop_reason == "finish"

    def test_no_op_explanation_is_not_intercepted(self, tmp_path):
        from pedacito.interleaved import run_interleaved
        client = _Scripted([[("finish", {"summary": "no change needed: the code "
                                                    "is already correct"})]])
        root, agent = _make_agent(tmp_path, client)
        res = run_interleaved(agent, "fix the calc bug", apply=True)
        assert res.stop_reason == "finish"
        assert _finish_msgs(res)[0] == "Finished."

    def test_read_only_task_is_not_intercepted(self, tmp_path):
        from pedacito.interleaved import run_interleaved
        client = _Scripted([[("finish", {"summary": "the answer is add"})]])
        root, agent = _make_agent(tmp_path, client)
        res = run_interleaved(agent, "explain what add does", apply=True)
        assert res.stop_reason == "finish"
        assert _finish_msgs(res)[0] == "Finished."

    def test_dry_run_zero_edits_is_not_intercepted(self, tmp_path):
        from pedacito.interleaved import run_interleaved
        client = _Scripted([[("finish", {"summary": "nothing yet"})]])
        root, agent = _make_agent(tmp_path, client)
        res = run_interleaved(agent, "fix the calc bug", apply=False)
        assert _finish_msgs(res)[0] == "Finished."

    def test_pre_stop_intercept_is_bounded_to_two(self, tmp_path):
        from pedacito.interleaved import run_interleaved
        client = _Scripted([[("finish", {"summary": "done"})]], loop_last=True)
        root, agent = _make_agent(tmp_path, client, max_no_progress_steps=0)
        res = run_interleaved(agent, "fix the calc bug", apply=True)
        finished = _finish_msgs(res)
        # max 2 intercepts, then the third finish is accepted
        assert sum(m.startswith("NOT FINISHED") for m in finished) <= 2
        assert finished[-1] == "Finished."


# =============================================== 6. scratchpad offloading
class TestScratchpad:
    def test_spill_observation_writes_and_stubs(self, tmp_path):
        cfg = _cfg(_root=str(tmp_path))
        big = "line %04d " + "x" * 60
        obs = "\n".join(big % i for i in range(500))
        out = tools.spill_observation(cfg, obs, "run", 4000)
        assert out is not obs
        assert ".pedacito/scratch/" in out and ".log" in out
        assert "lines)" in out
        logs = list((tmp_path / ".pedacito" / "scratch").glob("*.log"))
        assert len(logs) == 1
        assert logs[0].read_text(encoding="utf-8") == obs

    def test_spill_under_cap_is_a_noop(self, tmp_path):
        cfg = _cfg(_root=str(tmp_path))
        obs = "short output"
        assert tools.spill_observation(cfg, obs, "run", 4000) == obs
        assert not (tmp_path / ".pedacito" / "scratch").exists()

    def test_is_scratch_log(self):
        assert tools.is_scratch_log(".pedacito/scratch/abc.log")
        assert not tools.is_scratch_log(".pedacito/config.json")
        assert not tools.is_scratch_log(".pedacito/scratch/../x.log")
        assert not tools.is_scratch_log("scratch/abc.log")

    def test_scratch_is_read_only_but_pedacito_stays_write_protected(self):
        assert safety.is_protected(".pedacito/scratch/x.log", write=False) == ""
        assert safety.is_protected(".pedacito/scratch/x.log", write=True) != ""
        assert safety.is_protected(".pedacito/config.json", write=False) != ""
        assert safety.is_protected(".pedacito/index.json", write=True) != ""

    def test_read_scratch_log(self, tmp_path):
        d = tmp_path / ".pedacito" / "scratch"
        d.mkdir(parents=True)
        (d / "a.log").write_text("one\ntwo\nthree\n", encoding="utf-8")
        out = tools.read_scratch_log(tmp_path, ".pedacito/scratch/a.log",
                                     start=2, end=3)
        assert "lines 2-3 of 3" in out and "two" in out and "three" in out
        assert "one" not in out.split("[showing")[0]

    def test_prune_clears_scratch(self, tmp_path):
        from pedacito.index import state_dir
        scratch = state_dir(tmp_path, "scratch")
        scratch.mkdir(parents=True)
        (scratch / "a.log").write_text("x", encoding="utf-8")
        (scratch / "b.log").write_text("y", encoding="utf-8")
        removed = workspace.prune(tmp_path, keep_sessions=1, dry_run=True)
        assert removed["scratch"] == 2
        removed = workspace.prune(tmp_path, keep_sessions=1)
        assert removed["scratch"] == 2
        assert not scratch.exists()
