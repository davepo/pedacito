"""WP-11 Small-Model Capability Multipliers:
1. Token Economy & Observation Capping
2. Robust, Composable AST Patching
3. Symbol-Level PageRank & ES Import Auto-Healing
4. Oscillation Detection & Harness Hypothesis Ledger
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import chunker, edits, fold, graph, index as index_mod, loop, prompts, spec, tools
from pedacito.agent import (
    EDIT_FORMAT,
    GATHER_BASE,
    Agent,
    _detect_oscillation,
    _oscillation_key,
)
from pedacito.config import Config
from pedacito.llm import estimate_tokens
from pedacito.tools import TOOL_HELP


class _Scripted:
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


def _make_index(tmp_path: Path, files: dict[str, str], cfg: Config | None = None):
    root = tmp_path / "proj"
    root.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    idx = index_mod.build([str(root)], cfg or Config(), client=None,
                          summarise=False, root=root, verbose=False)
    return root, idx


def _make_agent(tmp_path: Path, files: dict[str, str], client=None, **cfg_kw):
    cfg = Config()
    cfg.backup = False
    cfg.log_events = False
    cfg.max_gathered_chars = 10 ** 7
    cfg.post_edit_check = False
    for k, v in cfg_kw.items():
        setattr(cfg, k, v)
    root, idx = _make_index(tmp_path, files, cfg)
    return root, Agent(idx, client or _Scripted(), cfg)


# ============================================================================
# 1. Token Economy & Observation Capping
# ============================================================================
def test_compact_tool_help_covers_undo_run_test_mcp_call(tmp_path):
    cfg = Config()
    cfg.prompt_pack = "tiny"
    cfg.allow_write = True
    cfg.backup = True
    cfg.allow_shell = True
    cfg._root = str(tmp_path)
    cfg._task_text = "fix failing pytest test"
    (tmp_path / "tests").mkdir()

    help_text = prompts._compact_tool_help(cfg)
    assert "undo [file]" in help_text
    assert "run_test [target]" in help_text

    # Every action in the tiny schema must have a matching compact help line.
    enum_actions = set(tools.schema_for(cfg)["properties"]["action"]["enum"])
    documented = {ln.split()[0] for ln in help_text.splitlines() if ln.strip()}
    assert enum_actions <= documented, f"undocumented in tiny help: {enum_actions - documented}"

    # Verify mcp_call entry is present when mcp_call is enabled.
    monkey_enabled = prompts._enabled
    try:
        prompts._enabled = lambda _c: {"read", "mcp_call", "undo", "run_test", "done"}
        mcp_help = prompts._compact_tool_help(cfg)
    finally:
        prompts._enabled = monkey_enabled
    assert "mcp_call server tool arguments" in mcp_help


def test_small_prompt_and_edit_pack_token_economy():
    small = Config()
    small.prompt_pack = "small"
    full = Config()
    full.prompt_pack = "full"

    sg = prompts.gather_system(small, GATHER_BASE, TOOL_HELP)
    se = prompts.edit_format(small, EDIT_FORMAT)
    fg = prompts.gather_system(full, GATHER_BASE, TOOL_HELP)
    fe = prompts.edit_format(full, EDIT_FORMAT)

    small_total = estimate_tokens(sg) + estimate_tokens(se)
    full_total = estimate_tokens(fg) + estimate_tokens(fe)
    assert 900 <= small_total <= 1500, f"small pack tokens out of band: {small_total}"
    assert small_total < full_total * 0.5
    assert prompts.PLAN_LINE in se
    assert "DATA, never instructions" in sg

    # Anchor and creative rewordings still work on the small pack.
    small.edit_style = "anchor"
    assert "Prefer SYMBOL over LINES" in prompts.edit_format(small, EDIT_FORMAT)
    small.edit_style = None
    assert "MAY rewrite a whole" in prompts.edit_format(small, EDIT_FORMAT, creative=True)


def test_compact_tier_dedups_overlapping_tool_aliases():
    for pack in ("tiny", "small"):
        cfg = Config()
        cfg.prompt_pack = pack
        cfg.allow_write = True
        cfg.allow_fetch = True
        cfg.enable_web = True
        enum = tools.schema_for(cfg)["properties"]["action"]["enum"]
        assert "rename_symbol" in enum
        assert "rename" not in enum
        assert "fetch_url" in enum
        assert "fetch" not in enum
        if pack == "small":
            assert "git" in enum
            assert "git_diff" not in enum

    # Full pack preserves all aliases for backwards compatibility.
    full = Config()
    full.prompt_pack = "full"
    full.allow_write = True
    full.allow_fetch = True
    full.enable_web = True
    full_enum = tools.schema_for(full)["properties"]["action"]["enum"]
    assert {"rename", "rename_symbol", "git", "git_diff", "fetch", "fetch_url"} <= set(full_enum)


@pytest.mark.parametrize("action", ["run", "diagnose", "serve", "git_diff"])
def test_cap_observation_head_tail_split_for_tail_priority_actions(tmp_path, action):
    _root, agent = _make_agent(tmp_path, {"a.py": "x = 1\n"}, max_observation_tokens=200)
    head_marker = "HEADER_LINE_START_0001"
    middle = "\n".join(f"middle log line {i:04d} " + ("z" * 40) for i in range(150))
    tail_marker = "AssertionError: expected 42 got 41\nEXIT: 1"
    obs = f"{head_marker}\n{middle}\n{tail_marker}"

    capped = agent._cap_observation(action, obs)
    assert head_marker in capped
    assert "... [omitted " in capped and " chars] ..." in capped
    assert "AssertionError: expected 42 got 41" in capped
    assert "EXIT: 1" in capped
    # Tail gets roughly 75% (significantly larger than head 25%).
    before_mid, after_mid = capped.split("... [omitted ", 1)
    assert len(after_mid) > len(before_mid) * 2


def test_cap_observation_group_caps_grep_per_file(tmp_path):
    _root, agent = _make_agent(tmp_path, {"a.py": "x = 1\n"}, max_observation_tokens=300)
    noisy_hits = [f"noisy.py:{i}: match token padding " + ("a" * 35) for i in range(1, 35)]
    quiet_hits = [f"quiet.py:{i}: important match" for i in range(1, 4)]
    obs = "[grep warning]\n" + "\n".join(noisy_hits + quiet_hits)

    capped = agent._cap_observation("grep", obs)
    assert "noisy.py:5:" in capped
    assert "noisy.py:6:" not in capped
    assert "(+29 more hit(s) in noisy.py)" in capped
    # Because noisy.py was group-capped at 5, quiet.py survives!
    assert "quiet.py:1: important match" in capped
    assert "quiet.py:3: important match" in capped


# ============================================================================
# 2. Robust, Composable AST Patching
# ============================================================================
def test_replace_already_present_checks_only_at_start():
    text = (
        "def first():\n"
        "    return 1\n\n"
        "def second():\n"
        "    x = 10\n"
        "    return x\n"
    )
    repl = "    x = 10\n    return x"
    # Present somewhere in file when start < 0:
    assert edits._replace_already_present(text, repl) is True
    # NOT present at offset 0 (inside first()):
    assert edits._replace_already_present(text, repl, start=0) is False
    # Present at exact offset where second()'s body starts:
    at = text.index("    x = 10")
    assert edits._replace_already_present(text, repl, start=at) is True


def test_normalise_symbol_indent_tab_dedent_overshoot():
    # Dedenting 4 tabs (16 cols) by 6 cols -> target 10 cols: 2 tabs + 2 spaces.
    body = "\t\tdef f(self):\n\t\t\t\treturn 1"
    shifted = edits.normalise_symbol_indent(body, "  ")
    lines = shifted.split("\n")
    assert lines[0] == "  def f(self):"
    assert lines[1] == "\t\t  return 1"
    assert len(lines[1][: len(lines[1]) - len(lines[1].lstrip())].expandtabs(4)) == 10

    # Mixed spaces + tab overshoot preserves trailing indent after overshot tab.
    body2 = "  \tdef g(self):\n  \t    return 2"
    shifted2 = edits.normalise_symbol_indent(body2, "  ")
    lines2 = shifted2.split("\n")
    assert lines2[0] == "  def g(self):"
    assert lines2[1] == "      return 2"


def test_empty_search_overwrite_preserves_crlf_and_bom(tmp_path):
    p = tmp_path / "win.py"
    p.write_bytes(b"\xef\xbb\xbfa = 1\r\nb = 2\r\n")
    reply = "--- FILE: win.py\n<<<<<<< SEARCH\n=======\na = 10\nb = 20\nc = 30\n>>>>>>> REPLACE\n"
    res = edits.apply_edits(tmp_path, edits.parse_edits(reply), backup=False)
    assert [r.ok for r in res] == [True]
    raw = p.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" in raw
    assert raw == b"\xef\xbb\xbfa = 10\r\nb = 20\r\nc = 30\r\n"


def test_parse_edits_ignores_quoted_symbol_and_json_actions_inside_owned_blocks():
    reply = (
        "--- FILE: docs.md\n"
        "<<<<<<< SEARCH\n"
        "old docs\n"
        "=======\n"
        "Use this syntax:\n"
        "<<<<<<< SYMBOL fake.py :: ghost\n"
        "def ghost():\n"
        "    return 1\n"
        ">>>>>>> END_SYMBOL\n"
        'Or JSON: {"edit_action": "replace_symbol", "file": "fake.py", '
        '"symbol": "ghost", "replacement": "def ghost(): pass"}\n'
        ">>>>>>> REPLACE\n"
    )
    parsed = edits.parse_edits(reply)
    assert len(parsed) == 1
    assert parsed[0].file == "docs.md"


def test_locate_dedent_normalized_unique_match_and_reindent():
    hay = (
        "class Calc:\n"
        "    def compute(self, x):\n"
        "        if x < 0:\n"
        "            return 0\n"
        "        return x * 2\n"
    )
    needle = "if x < 0:\n    return 0\nreturn x * 2"
    repl = "if x < 0:\n    return -x\nreturn x * 3"
    # Deterministic without fuzzy_match enabled:
    span = edits.locate(hay, needle, cfg=None, replacement=repl)
    assert span is not None and not span.fuzzy
    assert span.replacement == "        if x < 0:\n            return -x\n        return x * 3"
    patched = hay[:span.start] + span.replacement + hay[span.stop:]
    assert "        if x < 0:\n            return -x\n        return x * 3" in patched

    # Ambiguous dedented matches must still fail!
    dup_hay = hay + "\n    def other(self, x):\n        if x < 0:\n            return 0\n        return x * 2\n"
    assert edits.locate(dup_hay, needle, cfg=None, replacement=repl) is None


def test_chunker_locate_symbol_span_accepts_in_memory_text(tmp_path):
    disk_path = tmp_path / "mod.py"
    disk_path.write_text("def alpha():\n    return 1\n", encoding="utf-8")
    in_mem = "def alpha():\n    x = 1\n    y = 2\n    return x + y\n\ndef beta():\n    return 9\n"
    assert chunker.locate_symbol_span(disk_path, "beta") is None
    span = chunker.locate_symbol_span(disk_path, "beta", text=in_mem)
    assert span == (6, 7, "")


def test_sequential_symbol_and_after_symbol_edits_compose_in_one_reply(tmp_path):
    p = tmp_path / "app.py"
    p.write_text("def alpha():\n    return 1\n\ndef omega():\n    return 9\n", encoding="utf-8")
    reply = (
        "<<<<<<< SYMBOL app.py :: alpha\n"
        "def alpha():\n"
        "    a = 10\n"
        "    b = 20\n"
        "    return a + b\n"
        ">>>>>>> END_SYMBOL\n\n"
        "<<<<<<< AFTER_SYMBOL app.py :: alpha\n"
        "def middle():\n"
        "    return alpha() + 1\n"
        ">>>>>>> END_SYMBOL\n\n"
        "<<<<<<< SYMBOL app.py :: middle\n"
        "def middle():\n"
        "    return alpha() + 2\n"
        ">>>>>>> END_SYMBOL\n"
    )
    res = edits.apply_edits(tmp_path, edits.parse_edits(reply), backup=False)
    assert [r.ok for r in res] == [True], res[0].message
    out = p.read_text(encoding="utf-8")
    assert "return a + b" in out
    assert "def middle():\n    return alpha() + 2" in out
    assert "def omega():\n    return 9" in out


def test_resolve_symbol_anchors_prefers_live_ast_and_respects_insert_after(tmp_path):
    root, agent = _make_agent(tmp_path, {
        "app.py": "def alpha():\n    return 1\n\ndef beta():\n    return 2\n",
    })
    # Mutate disk WITHOUT reindexing: beta shifts down by 4 lines.
    (root / "app.py").write_text(
        "# header 1\n# header 2\n# header 3\n# header 4\n"
        "def alpha():\n    return 1\n\ndef beta():\n    return 2\n",
        encoding="utf-8",
    )
    parsed = edits.parse_edits(
        "--- FILE: app.py SYMBOL beta\n<<<<<<< SEARCH\n=======\n"
        "def beta():\n    return 22\n>>>>>>> REPLACE\n"
    )
    failed, left = agent._resolve_symbol_anchors(parsed)
    assert failed == []
    assert left[0].lines == (8, 9)  # live AST span, NOT stale index (4, 5)

    # AFTER_SYMBOL keeps lines=None so apply_edits inserts instead of replacing.
    ins = edits.parse_edits(
        "<<<<<<< AFTER_SYMBOL app.py :: beta\ndef gamma():\n    return 3\n>>>>>>> END_SYMBOL\n"
    )
    failed2, left2 = agent._resolve_symbol_anchors(ins)
    assert failed2 == []
    assert left2[0].lines is None and left2[0].insert_after == "beta"
    applied = edits.apply_edits(root, left2, backup=False)
    assert [r.ok for r in applied] == [True]
    final = (root / "app.py").read_text(encoding="utf-8")
    assert "def beta():" in final and "def gamma():" in final


# ============================================================================
# 3. Symbol-Level PageRank & ES Import Auto-Healing
# ============================================================================
def test_symbol_pagerank_ranks_called_and_qualified_symbols_and_supports_seeds(tmp_path):
    cfg = Config()
    cfg.class_split_lines = 4
    _root, idx = _make_index(tmp_path, {
        "store.py": (
            "class Store:\n"
            "    def get(self, k):\n"
            "        return k\n\n"
            "    def unused_method(self):\n"
            "        return 0\n\n"
            "class ColdBox:\n"
            "    def idle(self):\n"
            "        return 0\n\n"
            "def hub():\n"
            "    return 1\n\n"
            "def cold():\n"
            "    return 2\n"
        ),
        "svc_a.py": (
            "from store import Store, hub\n\n"
            "def a():\n"
            "    s = Store()\n"
            "    return s.get('x') + hub()\n"
        ),
        "svc_b.py": (
            "from store import Store, hub\n\n"
            "def b():\n"
            "    s = Store()\n"
            "    return s.get('y') + hub()\n"
        ),
    }, cfg=cfg)
    spr = graph.symbol_pagerank(idx)
    assert spr["store.py::hub"] > spr["store.py::cold"]
    assert spr["store.py::Store.get"] > spr["store.py::Store.unused_method"]
    assert spr["store.py::Store"] > spr["store.py::ColdBox"]
    # Determinism across cache clears
    idx.graph._sym_pr = None
    assert graph.symbol_pagerank(idx) == spr

    # Personalized seeds lift seeded targets
    seeded = graph.symbol_pagerank(idx, seeds=["store.py::cold"])
    assert seeded["store.py::cold"] > spr["store.py::cold"]


def test_large_file_card_orders_symbols_by_pagerank_deterministically(tmp_path):
    # Create a file with 25 functions where the LAST function (`zzz_core`) is
    # heavily called by multiple importers.
    funcs = "\n\n".join(f"def helper_{i:02d}():\n    return {i}" for i in range(24))
    big_mod = funcs + "\n\n\ndef zzz_core():\n    return 999\n"
    _root, idx = _make_index(tmp_path, {
        "big.py": big_mod,
        "u1.py": "from big import zzz_core\ndef f1():\n    return zzz_core()\n",
        "u2.py": "from big import zzz_core\ndef f2():\n    return zzz_core()\n",
        "u3.py": "from big import zzz_core\ndef f3():\n    return zzz_core()\n",
    })
    card1 = idx.render_cards(["big.py"], max_names=8)
    assert "zzz_core" in card1
    # Must be right at the front of `defines:` (highest centrality)
    defines_line = [ln for ln in card1.splitlines() if "defines:" in ln][0]
    first_sym = defines_line.split("defines:", 1)[1].strip().split(",")[0]
    assert first_sym.startswith("zzz_core")
    # Byte-identical on repeat renders
    idx.graph._sym_pr = None
    assert idx.render_cards(["big.py"], max_names=8) == card1


def test_heal_js_module_imports_auto_heals_and_wires_frontend_gate(tmp_path):
    from pedacito.chunker_ts import TS_AVAILABLE
    if not TS_AVAILABLE:
        pytest.skip("tree-sitter unavailable")

    (tmp_path / "index.html").write_text(
        '<!doctype html><html><body><script type="module" src="main.js"></script></body></html>\n',
        encoding="utf-8",
    )
    (tmp_path / "math.js").write_text(
        "export function doubleVal(x) { return x * 2; }\n"
        "export function ambig() { return 1; }\n",
        encoding="utf-8",
    )
    (tmp_path / "other.js").write_text(
        "export function ambig() { return 2; }\n",
        encoding="utf-8",
    )
    (tmp_path / "main.js").write_text(
        "export function run() {\n"
        "  return doubleVal(21);\n"
        "}\n",
        encoding="utf-8",
    )

    # Disabled flag heals nothing.
    cfg_off = Config()
    cfg_off.auto_heal_imports = False
    before = spec.verify_frontend_contracts(tmp_path, ["main.js"])
    assert any("doubleVal" in e for e in before)
    after_off, receipts_off = spec.heal_js_module_imports(
        tmp_path, files=["main.js"], errors=before, cfg=cfg_off)
    assert receipts_off == [] and after_off == before

    # Enabled via frontend contract gate in loop._new_frontend_errors:
    cfg = Config()
    cfg.auto_heal_imports = True
    remaining = loop._new_frontend_errors(tmp_path, cfg, ["main.js"])
    assert remaining == []
    healed = (tmp_path / "main.js").read_text(encoding="utf-8")
    assert "import { doubleVal } from './math.js';" in healed

    # Ambiguous export (both math.js and other.js export `ambig`) is NOT healed.
    (tmp_path / "caller.js").write_text(
        "export function go() { return ambig(); }\n", encoding="utf-8")
    ambig_errs = spec.verify_frontend_contracts(tmp_path, ["caller.js"])
    assert any("ambig" in e for e in ambig_errs)
    after_ambig, receipts_ambig = spec.heal_js_module_imports(
        tmp_path, files=["caller.js"], errors=ambig_errs, cfg=cfg)
    assert receipts_ambig == []
    assert after_ambig == ambig_errs


# ============================================================================
# 4. Oscillation Detection & Harness Hypothesis Ledger
# ============================================================================
def test_detect_oscillation_period2_and_period3():
    a = _oscillation_key({"action": "read", "file": "a.py", "start": 1, "end": 20})
    b = _oscillation_key({"action": "grep", "file": "b.py", "pattern": "foo"})
    c = _oscillation_key({"action": "outline", "file": "c.py"})

    assert _detect_oscillation([a, b, a, b]) == 2
    assert _detect_oscillation([a, b, c, a, b, c]) == 3
    assert _detect_oscillation([a, a, a, a]) == 0
    assert _detect_oscillation([a, b, c, a]) == 0


def test_gather_triggers_stuck_redirect_on_period2_oscillation(tmp_path):
    files = {
        "a.py": "def alpha():\n    return 1\n",
        "b.py": "def beta():\n    return 2\n",
    }
    client = _Scripted(steps=[
        {"action": "read", "file": "a.py", "start": 1, "end": 2},
        {"action": "read", "file": "b.py", "start": 1, "end": 2},
        {"action": "read", "file": "a.py", "start": 1, "end": 2},
        {"action": "read", "file": "b.py", "start": 1, "end": 2},
        {"action": "done"},
    ])
    _root, agent = _make_agent(tmp_path, files, client=client,
                               stuck_recovery=True, max_no_progress_steps=10)
    agent.gather("inspect a and b", verbose=False)
    all_user = "\n".join(
        m["content"] for call in client.json_calls for m in call if m["role"] == "user"
    )
    assert "[STUCK: you are oscillating in a period-2 loop" in all_user


def test_gather_triggers_stuck_redirect_on_period3_oscillation(tmp_path):
    files = {
        "a.py": "def alpha():\n    return 1\n",
        "b.py": "def beta():\n    return 2\n",
        "c.py": "def gamma():\n    return 3\n",
    }
    client = _Scripted(steps=[
        {"action": "read", "file": "a.py", "start": 1, "end": 2},
        {"action": "read", "file": "b.py", "start": 1, "end": 2},
        {"action": "read", "file": "c.py", "start": 1, "end": 2},
        {"action": "read", "file": "a.py", "start": 1, "end": 2},
        {"action": "read", "file": "b.py", "start": 1, "end": 2},
        {"action": "read", "file": "c.py", "start": 1, "end": 2},
        {"action": "done"},
    ])
    _root, agent = _make_agent(tmp_path, files, client=client,
                               stuck_recovery=True, max_no_progress_steps=10)
    agent.gather("inspect a, b, and c", verbose=False)
    all_user = "\n".join(
        m["content"] for call in client.json_calls for m in call if m["role"] == "user"
    )
    assert "[STUCK: you are oscillating in a period-3 loop" in all_user


def test_hypothesis_ledger_caps_at_5_and_flows_into_repairs_and_folds(tmp_path):
    files = {"app.py": "def calc(x):\n    return x + 1\n"}
    bad_reply = (
        "--- FILE: app.py\n"
        "<<<<<<< SEARCH\n"
        "def calc(x):\n    return x + 999\n"
        "=======\n"
        "def calc(x):\n    return x + 2\n"
        ">>>>>>> REPLACE\n"
    )
    fixed_reply = (
        "--- FILE: app.py\n"
        "<<<<<<< SEARCH\n"
        "def calc(x):\n    return x + 1\n"
        "=======\n"
        "def calc(x):\n    return x + 2\n"
        ">>>>>>> REPLACE\n"
    )
    client = _Scripted(replies=[fixed_reply])
    _root, agent = _make_agent(tmp_path, files, client=client, repair_rounds=1)

    # Record 7 distinct failures -> capped at latest 5.
    for i in range(7):
        agent._record_failed_hypothesis(f"f{i}.py", f"error number {i}")
    assert len(agent._failed_hypotheses) == 5
    assert agent._failed_hypotheses[0][0] == "f2.py"
    assert agent._failed_hypotheses[-1][0] == "f6.py"

    # Repair prompt includes TRIED & FAILED ledger.
    results, _final = agent.apply(bad_reply, dry_run=False, task="change +1 to +2")
    assert any(r.ok for r in results)
    repair_prompt = client.chat_calls[0][1]["content"]
    assert "TRIED & FAILED:" in repair_prompt
    assert "app.py ->" in repair_prompt

    # STATE fold includes TRIED & FAILED and preserves it across a fold of a fold.
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "map + task"},
    ]
    for i in range(5):
        msgs.append({"role": "assistant",
                     "content": json.dumps({"action": "outline", "file": f"m{i}.py"})})
        msgs.append({"role": "user", "content": f"RESULT:\n### m{i}.py\n  L1-5 def f{i}()"})
    assert fold.fold_middle(msgs, keep_pairs=1, hypotheses=agent._failed_hypotheses)
    state_msg = msgs[3]["content"]
    assert "TRIED & FAILED:" in state_msg
    assert "app.py ->" in state_msg

    # Fold of a fold keeps carried-forward hypotheses intact.
    for i in range(5, 8):
        msgs.append({"role": "assistant",
                     "content": json.dumps({"action": "outline", "file": f"m{i}.py"})})
        msgs.append({"role": "user", "content": f"RESULT:\n### m{i}.py\n  L1-5 def f{i}()"})
    assert fold.fold_middle(msgs, keep_pairs=1)
    assert "TRIED & FAILED:" in msgs[3]["content"]
    assert "app.py ->" in msgs[3]["content"]
