"""Offline smoke tests: no LLM server required.

Covers the config-simplification surface (templates parse, tier presets
apply, setup error carries a hint), the edit-block parser/renamer bugs
("nothing to review" on rename tasks), and the coverage ledger in agent.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import config as config_mod
from pedacito import edits
from pedacito.agent import Agent
from pedacito.llm import LMStudio, LLMError


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


FAKE = Path(__file__).resolve().parent / "fake_llm.py"


# --------------------------------------------------------------------- config
def test_templates_are_valid_toml():
    for name in ("TEMPLATE", "REFERENCE_TEMPLATE"):
        text = getattr(config_mod, name)
        data = tomllib.loads(text)          # must not raise
        assert isinstance(data, dict)
    # The starter template must stay short -- that was the whole point of
    # the simplification.
    assert len(config_mod.TEMPLATE.splitlines()) < 80
    assert "base_url" in config_mod.TEMPLATE
    assert "model" in config_mod.TEMPLATE


def test_tier_preset_applies_without_profiles(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        '[server]\nbase_url = "http://localhost:9/v1"\n'
        'model = "m"\ntier = "small"\n'
        'vram_gb = 8\n', encoding="utf-8")
    cfg = config_mod.load(root=tmp_path, explicit=p, use_env=False)
    assert cfg.max_steps == config_mod.PRESETS["small"]["max_steps"]
    assert cfg.active_tier == "small"
    assert "tier 'small' preset" in cfg.sources["max_steps"]


def test_bad_tier_fails_loudly(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text('tier = "huge"\n', encoding="utf-8")
    with pytest.raises(config_mod.ConfigError):
        config_mod.load(root=tmp_path, explicit=p, use_env=False)


def test_default_profile_missing_gives_setup_hint(tmp_path, monkeypatch):
    fake = tmp_path / "user.toml"
    fake.write_text('default_profile = "gpu"\n', encoding="utf-8")
    monkeypatch.setattr(config_mod, "USER_CONFIG", fake)
    with pytest.raises(config_mod.ConfigError) as e:
        config_mod.load(root=tmp_path, use_env=False)
    assert "--setup" in str(e.value)


def test_discover_servers_and_interactive_setup_exist():
    assert callable(config_mod.discover_servers)
    assert callable(config_mod.interactive_setup)


# --------------------------------------------------------------------- edits
RENAME_REPLY = """\
Renaming the module:

--- FILE: new_name.py
<<<<<<< SEARCH
=======
print("hi")
>>>>>>> REPLACE

--- FILE: old_name.py
<<<<<<< SEARCH
print("hi")
=======
>>>>>>> REPLACE
"""


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_parse_edits_roundtrip():
    blocks = edits.parse_edits(RENAME_REPLY)
    assert [b.file for b in blocks] == ["new_name.py", "old_name.py"]
    assert blocks[0].search == "" and "hi" in blocks[0].replace
    assert "hi" in blocks[1].search and blocks[1].replace.strip() == ""


def test_parse_edits_keeps_equals_banner_in_search():
    """A SEARCH that begins with a `/* ====...` comment banner must survive
    parsing intact. The separator matcher used to match the '=' run inside the
    banner, truncating the SEARCH to `/* ` (which then matched every comment
    opener -- the 'search block appears N times' failure on JS renames)."""
    reply = (
        "--- FILE: sentences.js\n"
        "<<<<<<< SEARCH\n"
        "/* =========================================================\n"
        "   sentences.js — Banco de oraciones\n"
        "=======\n"
        ">>>>>>> REPLACE\n"
    )
    blocks = edits.parse_edits(reply)
    assert len(blocks) == 1
    assert blocks[0].replace == ""
    assert blocks[0].search.startswith("/* =="), repr(blocks[0].search)
    assert "Banco de oraciones" in blocks[0].search



def test_rename_pair_applies(tmp_path):
    """The exact regression behind 'nothing to review' on rename tasks:
    deleting the source file's full content (EOF block without trailing
    newline in SEARCH) must pair with the create block. The protocol moves
    the source's REAL bytes into the new path and removes the old path, so
    the source contributes no Result of its own -- asserting on disk state
    is the honest check."""
    _write(tmp_path / "old_name.py", 'print("hi")')     # no trailing \n!
    blocks = edits.parse_edits(RENAME_REPLY)
    results = edits.apply_edits(tmp_path, blocks, backup=False, dry_run=True)
    by_file = {r.file: r for r in results}
    assert by_file["new_name.py"].ok, by_file["new_name.py"].message
    assert by_file["new_name.py"].renames_from == ("old_name.py",), \
        "rename pairing dropped -- this is the 'nothing to review' bug"
    # And for real this time: the move happens, old path is removed.
    results = edits.apply_edits(tmp_path, blocks, backup=False, dry_run=False)
    assert all(r.ok for r in results), [r.message for r in results]
    assert (tmp_path / "new_name.py").is_file()
    assert not (tmp_path / "old_name.py").exists()
    assert 'print("hi")' in (tmp_path / "new_name.py").read_text()


def test_rename_with_opening_lines_search(tmp_path):
    """The answer prompt tells the model to quote only the FIRST few lines of
    the old file in the clearing block, never the whole file. That opening-
    lines search must still pair with the create block and move the source's
    real bytes -- requiring a search that reaches end-of-file here made the
    documented protocol fail and left the old file half-cleared."""
    _write(tmp_path / "old_name.py", "line1\nline2\nline3\nline4\nline5\n")
    reply = (
        "--- FILE: new_name.py\n"
        "<<<<<<< SEARCH\n"
        "=======\n"
        "moved from old_name.py\n"
        ">>>>>>> REPLACE\n\n"
        "--- FILE: old_name.py\n"
        "<<<<<<< SEARCH\n"
        "line1\nline2\nline3\n"
        "=======\n"
        ">>>>>>> REPLACE\n"
    )
    results = edits.apply_edits(tmp_path, edits.parse_edits(reply),
                                backup=False, dry_run=False)
    assert all(r.ok for r in results), [r.message for r in results]
    assert (tmp_path / "new_name.py").is_file()
    assert not (tmp_path / "old_name.py").exists()
    assert "line5" in (tmp_path / "new_name.py").read_text()


def test_rename_with_midfile_clear_block(tmp_path):
    """Models anchor the clearing block on any UNIQUE text they read -- often
    a script tag or snippet in the middle of a file, not its first lines. That
    mid-file clear must still pair with the create block and move the whole
    source, rather than deleting just the snippet and leaving a stub."""
    _write(tmp_path / "index.html",
           "<html>\n<head><title>x</title></head>\n<body>\n"
           "<script src=\"sentences.js\"></script>\n"
           "<p>real body content</p>\n</body>\n</html>\n")
    reply = (
        "--- FILE: renamed.html\n"
        "<<<<<<< SEARCH\n"
        "=======\n"
        "<!-- placeholder: moved from index.html -->\n"
        ">>>>>>> REPLACE\n\n"
        "--- FILE: index.html\n"
        "<<<<<<< SEARCH\n"
        "<script src=\"sentences.js\"></script>\n"
        "=======\n"
        ">>>>>>> REPLACE\n"
    )
    results = edits.apply_edits(tmp_path, edits.parse_edits(reply),
                                backup=False, dry_run=False)
    assert all(r.ok for r in results), [r.message for r in results]
    assert (tmp_path / "renamed.html").is_file()
    assert not (tmp_path / "index.html").exists()
    assert "real body content" in (tmp_path / "renamed.html").read_text()


def test_rename_placeholder_already_a_comment(tmp_path):
    """A rename placeholder the model already wrote as a comment (`/* ... */`)
    must be kept verbatim, not wrapped in a second `/* ... */` -- the
    double-wrap produced `/* /* ... */ */`, a syntax error that aborted the
    whole JS rename."""
    _write(tmp_path / "sentences.js",
           "/* =========================================================\n"
           "   sentences.js — Banco de oraciones\n"
           "   ========================================================= */\n"
           "var s = ['uno'];\n")
    reply = (
        "--- FILE: subjuntivo-imperativo.js\n"
        "<<<<<<< SEARCH\n"
        "=======\n"
        "/* placeholder — real contents are moved from sentences.js */\n"
        ">>>>>>> REPLACE\n\n"
        "--- FILE: sentences.js\n"
        "<<<<<<< SEARCH\n"
        "/* =========================================================\n"
        "   sentences.js — Banco de oraciones\n"
        "=======\n"
        ">>>>>>> REPLACE\n"
    )
    results = edits.apply_edits(tmp_path, edits.parse_edits(reply),
                                backup=False, dry_run=False)
    assert all(r.ok for r in results), [r.message for r in results]
    assert (tmp_path / "subjuntivo-imperativo.js").is_file()
    assert not (tmp_path / "sentences.js").exists()
    body = (tmp_path / "subjuntivo-imperativo.js").read_text()
    assert "/* placeholder" in body
    assert "var s = ['uno'];" in body




def test_review_write_deletes_rename_source(tmp_path):
    """The review sign-off path must delete a rename's source file, not write
    an empty file over it. Writing '' first clobbered the source's bytes so
    the removal loop's sha check refused, leaving an empty old_name.py next
    to the renamed copy."""
    from pedacito import review
    _write(tmp_path / "old_name.py", "print('hi')\n")
    reply = (
        "--- FILE: new_name.py\n"
        "<<<<<<< SEARCH\n"
        "=======\n"
        "moved from old_name.py\n"
        ">>>>>>> REPLACE\n\n"
        "--- FILE: old_name.py\n"
        "<<<<<<< SEARCH\n"
        "print('hi')\n"
        "=======\n"
        ">>>>>>> REPLACE\n"
    )
    results = edits.apply_edits(tmp_path, edits.parse_edits(reply),
                                backup=False, dry_run=True)
    diffs = review.diffs_from_results(results, root=tmp_path)
    backup_dir = tmp_path / "bak"
    backup_dir.mkdir()
    ok, detail = review.write_files(tmp_path, diffs, backup_dir)
    assert ok, detail
    assert (tmp_path / "new_name.py").is_file()
    assert not (tmp_path / "old_name.py").exists(), \
        "rename source left behind as an empty file: " + detail


def test_search_block_matches_at_eof_with_newline_tail(tmp_path):
    _write(tmp_path / "a.py", "x = 1\n")
    reply = ('--- FILE: a.py\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n'
             '>>>>>>> REPLACE\n')
    results = edits.apply_edits(tmp_path, edits.parse_edits(reply),
                                backup=False, dry_run=False)
    assert results[0].ok, results[0].message
    assert (tmp_path / "a.py").read_text() == "x = 2\n"


def test_empty_search_overwrites_existing_file(tmp_path):
    """An EMPTY-SEARCH block against an existing file is a whole-file replace
    (the 'implement a stub' convention), not a failed anchor: there is nothing
    to match, so the file is replaced outright. This is what --complex's
    stub-implementing tasks rely on instead of reproducing the stub byte-for-byte."""
    _write(tmp_path / "assets.py",
           '"""doc"""\n\n\ndef foo():\n    raise NotImplementedError\n')
    reply = (
        "--- FILE: assets.py\n"
        "<<<<<<< SEARCH\n"
        "=======\n"
        '"""doc"""\n\n\ndef foo():\n    return 42\n'
        ">>>>>>> REPLACE\n"
    )
    results = edits.apply_edits(tmp_path, edits.parse_edits(reply),
                                backup=False, dry_run=False)
    assert results[0].ok, results[0].message
    assert "return 42" in (tmp_path / "assets.py").read_text()
    assert "raise NotImplementedError" not in (tmp_path / "assets.py").read_text()


def test_apply_continues_when_model_defers(tmp_path):
    """A model that cleanly finishes with "the rest follows in the next turn"
    (rather than truncating) must trigger a follow-up turn -- otherwise a
    multi-file task stops at its first batch, 'only half the request was
    addressed'. The completion loop re-prompts and applies the remaining work
    until the model answers NOTHING MORE."""
    from pedacito import index as index_mod
    _write(tmp_path / "old.js", "var a = 1;\nvar b = 2;\n")
    idx = index_mod.Index(tmp_path, [])

    class StubClient:
        stream_truncated = False
        def __init__(self):
            self.prompts = []
        def chat(self, messages, max_tokens, schema=None, stream=False, **kw):
            self.prompts.append(messages[-1]["content"])
            return "NOTHING MORE"

    client = StubClient()
    agent = Agent(idx, client, config_mod.Config())
    reply = (
        "--- FILE: new.js\n"
        "<<<<<<< SEARCH\n"
        "=======\n"
        "// placeholder\n"
        ">>>>>>> REPLACE\n\n"
        "--- FILE: old.js\n"
        "<<<<<<< SEARCH\n"
        "var a = 1;\n"
        "=======\n"
        ">>>>>>> REPLACE\n\n"
        "The HTML renames will follow in the next turn.\n"
    )
    results, _ = agent.apply(reply, dry_run=False,
                             task="rename old.js to new.js", gathered="")
    assert all(r.ok for r in results), [r.message for r in results]
    assert (tmp_path / "new.js").is_file()
    assert not (tmp_path / "old.js").exists()
    assert client.prompts, "deferral did not trigger a follow-up turn"
    assert "EDITS ALREADY DELIVERED" in client.prompts[0]


def test_completion_loop_has_no_default_round_cap(tmp_path):
    """Deferred work used to stop after six turns regardless of progress.
    With the default (0 = no cap) a model that keeps delivering blocks must
    be allowed to finish; only NOTHING MORE / no blocks / the token-derived
    budget ends the loop."""
    from pedacito import index as index_mod
    _write(tmp_path / "a.py", "x = 1\n")
    idx = index_mod.Index(tmp_path, [])

    class StubClient:
        stream_truncated = False

        def __init__(self):
            self.n = 0

        def chat(self, messages, max_tokens, schema=None, stream=False, **kw):
            self.n += 1
            if self.n > 7:
                return "NOTHING MORE"
            return (f"--- FILE: f{self.n}.py\n<<<<<<< SEARCH\n=======\n"
                    f"x = {self.n}\n>>>>>>> REPLACE\n\n"
                    "More follows in the next turn.")

    client = StubClient()
    agent = Agent(idx, client, config_mod.Config())
    first = ("--- FILE: first.py\n<<<<<<< SEARCH\n=======\nx = 0\n"
             ">>>>>>> REPLACE\n\nMore follows in the next turn.")
    results, _ = agent.apply(first, dry_run=False, task="build files")
    assert all(r.ok for r in results), [r.message for r in results]
    assert client.n == 8, f"stopped after {client.n} completion turns"
    assert (tmp_path / "f7.py").is_file()


def test_apply_reasks_when_truncated_before_any_block(tmp_path):
    """A streamed reply cut off before completing even ONE block used to be
    trimmed to nothing and silently fail ('no edit blocks in the reply').
    It must instead re-ask the model for the file, so a whole-file write that
    overflowed the output limit is recovered rather than dropped."""
    from pedacito import index as index_mod
    _write(tmp_path / "a.py", '"""doc"""\n\n\ndef foo():\n    raise NotImplementedError\n')
    idx = index_mod.Index(tmp_path, [])

    class StubClient:
        stream_truncated = True          # the first reply is cut off
        def __init__(self):
            self.prompts = []
        def chat(self, messages, max_tokens, schema=None, stream=False, **kw):
            self.prompts.append(messages[-1]["content"])
            self.stream_truncated = False      # the re-ask completes cleanly
            return (
                "--- FILE: a.py\n"
                "<<<<<<< SEARCH\n"
                "=======\n"
                '"""doc"""\n\n\ndef foo():\n    return 42\n'
                ">>>>>>> REPLACE\n"
            )

    client = StubClient()
    agent = Agent(idx, client, config_mod.Config())
    reply = ('--- FILE: a.py\n<<<<<<< SEARCH\n=======\n"""doc"""\n\n\n'
             'def foo():\n    return 4')          # truncated: no closing REPLACE
    results, _ = agent.apply(reply, dry_run=False,
                             task="implement a.py", gathered="")
    assert all(r.ok for r in results), [r.message for r in results]
    assert "return 42" in (tmp_path / "a.py").read_text()
    assert client.prompts, "a cut-off reply with no complete block must re-ask"
    assert "CUT OFF" in client.prompts[0]


# ------------------------------------------------------------------ budgets
def test_default_budgets_fit_small_windows():
    """The historical failure: defaults sized for 32k+ models made every
    step overflow an 8k local server (and DeepSeek's real window). The
    default context_tokens must be small enough that the per-step prompt
    budget is sane without any user tuning."""
    cfg = config_mod.Config()
    assert cfg.context_tokens <= 8192
    usable = cfg.context_tokens - cfg.reserve_tokens
    assert cfg.max_tokens_answer <= usable
    assert cfg.max_tokens_step <= usable
    # Every shipped tier preset must survive its own sanity check.
    for name, vals in config_mod.PRESETS.items():
        c = config_mod.Config()
        for k, v in vals.items():
            setattr(c, k, v)
        assert not config_mod.budget_warnings(c), f"preset {name} is over-budget"


def test_auto_fit_clamps_hosted_profile_to_small_window():
    """The DeepSeek symptom, reproduced offline: a `tier = "hosted"` config
    pointed at an 8k local server asked for reply budgets (16k answer, 32k
    cap) that exceed the whole window -- llama.cpp answers every request
    with a flat 400 ("kept hitting limits"). auto_fit must rewrite every
    budget so prompt + max_tokens fits inside context_tokens, and the
    result must pass budget_warnings with nothing left to complain about."""
    cfg = config_mod.Config()
    cfg.vram_gb = 8
    for k, v in config_mod.PRESETS["hosted"].items():
        setattr(cfg, k, v)
    cfg.context_tokens = 8192
    cfg.sources["context_tokens"] = "explicit"   # pretend the user set it
    cfg.auto_detect_context = False              # no server in this test
    notes = config_mod.auto_fit(cfg, verbose=False)
    assert notes, "hosted budgets on an 8k window must trigger auto-fit"
    usable = cfg.context_tokens - cfg.reserve_tokens
    assert usable >= 512, "reserve must shrink first to keep usable space"
    for key in ("max_tokens_step", "max_tokens_summary", "max_tokens_answer",
                "max_tokens_build", "max_tokens_plan"):
        v = getattr(cfg, key)
        assert v <= cfg.context_tokens - 512, f"{key}={v} still overflows"
    assert cfg.request_max_tokens_cap <= usable
    # map + gathered source + one step must share the remaining window
    map_tok = cfg.flat_map_token_budget
    gathered_tok = cfg.max_gathered_chars // 4
    assert map_tok + gathered_tok + cfg.max_tokens_step <= usable + 1
    assert not config_mod.budget_warnings(cfg), \
        f"post-fit warnings: {config_mod.budget_warnings(cfg)}"


def test_vram_setting_mandatory_for_local_servers(tmp_path):
    """vram_gb is required for local runs (auto-fit needs it), but hosted
    APIs run elsewhere and must not demand local specs."""
    local = tmp_path / "local.toml"
    local.write_text('[server]\nbase_url = "http://localhost:1234/v1"\n'
                     'model = "m"\n', encoding="utf-8")
    with pytest.raises(config_mod.ConfigError) as e:
        config_mod.load(explicit=local, use_env=False)
    assert "vram_gb" in str(e.value)

    # The short alias lands on the canonical field.
    local.write_text('[server]\nbase_url = "http://localhost:1234/v1"\n'
                     'model = "m"\nvram = 8\n', encoding="utf-8")
    cfg = config_mod.load(explicit=local, use_env=False)
    assert cfg.vram_gb == cfg.vram == 8

    hosted = tmp_path / "hosted.toml"
    hosted.write_text('[server]\nbase_url = "https://api.deepseek.com/v1"\n'
                      'model = "deepseek-chat"\njson_mode = "object"\n',
                      encoding="utf-8")
    cfg = config_mod.load(explicit=hosted, use_env=False)
    assert config_mod.missing_hardware(cfg) == []


def test_vram_estimate_caps_the_context_window():
    """VRAM is a hard cap for the context window: planning for a larger
    tier/default window than the GPU can hold spills the KV cache and tanks
    tokens/sec. The fallback is used only when the size is unknown."""
    def cfg_for(vram: int) -> config_mod.Config:
        c = config_mod.Config()
        c.base_url = "http://127.0.0.1:1/v1"
        c.model = "m"
        c.vram_gb = vram
        c.context_tokens = 8192
        c.sources["context_tokens"] = "tier 'small' preset"
        c.auto_detect_context = False
        c.model_size_bytes = 16_500_000_000
        return c

    # 19 GB VRAM - 16.5 GB model - 0.5 GB reserve leaves only a small
    # window, so the 8k fallback is reduced to what actually fits.
    low = cfg_for(19)
    config_mod.auto_fit(low, verbose=False)
    limit = config_mod.hardware_context_tokens(low.vram_gb,
                                               low.model_size_bytes)
    assert 0 < limit < 8192
    assert low.context_tokens == limit

    # More VRAM raises the window past the fallback.
    high = cfg_for(48)
    config_mod.auto_fit(high, verbose=False)
    assert high.context_tokens > 8192

    # An explicit context_tokens is never replaced by the estimate.
    explicit = cfg_for(48)
    explicit.context_tokens = 12000
    explicit.sources["context_tokens"] = "config file"
    config_mod.auto_fit(explicit, verbose=False)
    assert explicit.context_tokens == 12000


def test_kv_cache_quantization_refines_the_vram_estimate():
    """The VRAM estimate assumes a 16-bit KV cache. When the runtime is using
    quantized K/V caches, matching hints must scale the per-token cache down
    instead of pretending the configured window does not fit."""
    model = 19_000_000_000
    base = config_mod.hardware_context_tokens(24, model)
    quantized = config_mod.hardware_context_tokens(24, model, "q8_0", "q5_1")
    measured = config_mod.hardware_context_tokens(
        24, model, kv_cache_bytes_per_token=96 * 1024)

    assert 0 < base < quantized < measured
    # f16 K + f16 V -> q8_0 K + q5_1 V is roughly half the cache.
    assert quantized > base * 1.8

    with pytest.raises(config_mod.ConfigError):
        config_mod.kv_quant_bits("q2_banana")


def test_auto_fit_uses_kv_cache_hints():
    """auto_fit must pass the profile's K/V cache hints into the estimate so
    budgets are sized for the context the GPU actually holds."""
    def cfg_for(k: str, v: str) -> config_mod.Config:
        c = config_mod.Config()
        c.base_url = "http://127.0.0.1:1/v1"
        c.model = "m"
        c.vram_gb = 24
        c.auto_detect_context = False
        c.model_size_bytes = 19_000_000_000
        c.k_cache_quant = k
        c.v_cache_quant = v
        return c

    plain = cfg_for("", "")
    quant = cfg_for("q8_0", "q5_1")
    config_mod.auto_fit(plain, verbose=False)
    config_mod.auto_fit(quant, verbose=False)
    assert quant.context_tokens > plain.context_tokens
    assert quant.max_tokens_answer > plain.max_tokens_answer


def test_kv_cache_config_aliases_and_spellings(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(
        '[hardware]\n'
        'vram_gb = 24\n'
        'k_cache_type = "Q8_0"\n'
        'v_cache_quantization = "Q5.1"\n'
        'kv_cache_bytes_per_token = 131072\n', encoding="utf-8")
    cfg = config_mod.load(explicit=path, use_env=False)
    assert cfg.k_cache_quant == "q8_0"
    assert cfg.v_cache_quant == "q5_1"
    assert cfg.kv_cache_bytes_per_token == 131072

    bad = tmp_path / "bad.toml"
    bad.write_text('[hardware]\nvram_gb = 24\nk_cache_quant = "nope"\n',
                   encoding="utf-8")
    with pytest.raises(config_mod.ConfigError):
        config_mod.load(explicit=bad, use_env=False)


def test_lemonade_loaded_options_win_over_list_default(monkeypatch):
    """Lemonade's model-list context_length may be the registry/default value
    even when the model was loaded (or saved) with a larger ctx_size. The
    health and per-model options endpoints report the real one."""
    from pedacito import llm as llm_mod

    def fake_get(url, **kwargs):
        class R:
            status_code = 200

            def json(self):
                if url.endswith("/options"):
                    return {"resolved_ctx_size": 32768,
                            "effective": {"ctx_size": 32768}}
                if url.endswith("/health"):
                    return {"all_models_loaded": [{
                        "model_name": "qwen_coder_25",
                        "recipe_options": {"ctx_size": 32768},
                        "max_context_window": 262144,
                    }]}
                return {"data": []}

        return R()

    monkeypatch.setattr(llm_mod.requests, "get", fake_get)
    cfg = config_mod.Config(base_url="http://localhost:13305/v1",
                            model="qwen_coder_25")
    client = llm_mod.LMStudio(cfg)
    assert client._lemonade_health_context("http://localhost:13305") == \
        (32768, 262144)
    assert client._lemonade_options_context("http://localhost:13305") == 32768

    got = client._absorb_model_list([{
        "id": "qwen_coder_25",
        "owned_by": "lemonade",
        "size": 18.5,
        "context_length": 16384,
        "recipe_options": {"ctx_size": 32768},
        "max_context_window": 262144,
    }], "openai-models")
    assert got["context_length"] == 32768


def test_explicit_context_sizes_budgets_above_the_vram_estimate():
    """A configured context_tokens is authoritative for budget fitting, just
    like a server-reported one. The VRAM estimate may warn about cache spill,
    but it must not shrink every reply budget to the estimate (the symptom
    that made a 32k profile behave like an 8k one)."""
    cfg = config_mod.Config()
    cfg.base_url = "http://127.0.0.1:1/v1"
    cfg.model = "m"
    cfg.vram_gb = 24
    for key, value in config_mod.PRESETS["small"].items():
        setattr(cfg, key, value)
        cfg.sources[key] = "tier 'small' preset"
    cfg.context_tokens = 32768
    cfg.sources["context_tokens"] = "config file"
    cfg.auto_detect_context = False
    cfg.model_size_bytes = 19_000_000_000

    cfg.max_tokens_answer = 1500
    cfg.sources["max_tokens_answer"] = "tier 'small' preset"
    notes = config_mod.auto_fit(cfg, verbose=False)

    hw = config_mod.hardware_context_tokens(24, 19_000_000_000)
    assert cfg.context_tokens == 32768
    assert cfg.max_tokens_answer > hw
    assert any("configured context" in n for n in notes)


def test_server_configured_context_wins_over_vram_estimate(monkeypatch):
    """A server's CONFIGURED context length is a fact -- it is exactly what
    the server will accept -- while the VRAM estimate is a heuristic with no
    knowledge of the model's KV architecture or quantization. Clamping to a
    wrong estimate is what converted a working 262k-token setup into a 12k
    window whose reply budgets no thinking model could emit JSON inside
    (the 'kept running past the token limit ... max_tokens=1500' failure).
    The estimate must stay a speed advisory: warn about KV spill, but keep
    the server's window and fit budgets to THAT."""
    import pedacito.llm as llm_mod

    class FakeServer:
        def __init__(self, cfg):
            self.cfg = cfg

        def model_info(self):
            return {"size_bytes": int(17.2 * 1024 ** 3),
                    "context_length": 262144,
                    "max_context_length": 262144,
                    "source": "lemonade"}

    monkeypatch.setattr(llm_mod, "LMStudio", FakeServer)
    cfg = config_mod.Config()
    cfg.base_url = "http://localhost:13305/v1"
    cfg.model = "qwen3-32b"
    cfg.vram_gb = 24
    for key, value in config_mod.PRESETS["large"].items():
        setattr(cfg, key, value)
    cfg.sources = {key: "tier 'large' preset"
                   for key in config_mod.PRESETS["large"]}
    notes = config_mod.auto_fit(cfg, verbose=False)
    assert cfg.context_tokens == 262144
    assert any("spill" in n for n in notes), notes
    # Budgets are fitted to the SERVER's window, not the tiny VRAM estimate.
    assert cfg.max_tokens_plan >= config_mod.PRESETS["large"]["max_tokens_plan"]
    assert cfg.max_tokens_answer >= config_mod.PRESETS["large"]["max_tokens_answer"]
    assert cfg.max_tokens_build >= config_mod.PRESETS["large"]["max_tokens_build"]
    assert cfg.request_max_tokens_cap > 12000


def test_auto_fit_respects_explicit_values_that_still_fit():
    """auto_fit only clamps numbers that physically cannot work; a user who
    explicitly sets a modest value keeps it untouched."""
    cfg = config_mod.Config()
    cfg.vram_gb = 8
    cfg.context_tokens = 8192
    cfg.auto_detect_context = False
    cfg.max_tokens_answer = 1000
    cfg.sources["max_tokens_answer"] = "config file"
    before = dict(vars(cfg))
    config_mod.auto_fit(cfg, verbose=False)
    assert cfg.max_tokens_answer == 1000
    assert cfg.sources.get("max_tokens_answer") == "config file"
    del before


def test_auto_detected_output_cap_does_not_override_explicit_budgets(tmp_path):
    """An explicit max_tokens_* value is a user instruction, not a guess.

    The hosted Qwen heuristic auto-sets max_output_tokens = 8000. That
    detected ceiling may govern tier/default budgets, but it must never
    rewrite budgets the user spelled out themselves (the reported bug:
    explicit 32k/32k/16k budgets were printed as 8000 by auto-fit).
    """
    path = tmp_path / "c.toml"
    path.write_text(
        '[server]\n'
        'base_url = "https://api.example.com/v1"\n'
        'model = "qwen3-32b"\n'
        'max_tokens_answer = 32000\n'
        'max_tokens_step = 32000\n'
        'max_tokens_plan = 16000\n'
        'max_tokens_build = 32000\n', encoding="utf-8")
    cfg = config_mod.load(explicit=path, use_env=False)
    assert cfg.max_output_tokens == 8000
    assert "auto-detected" in cfg.sources["max_output_tokens"]
    want = {
        "max_tokens_answer": 32000,
        "max_tokens_step": 32000,
        "max_tokens_plan": 16000,
        "max_tokens_build": 32000,
    }
    config_mod.auto_fit(cfg, verbose=False)
    for key, value in want.items():
        assert getattr(cfg, key) == value, \
            f"{key} was changed by an auto-detected output cap"
        assert cfg.sources[key].startswith("--config")

    # auto_fit = false must not perform that rewrite either.
    cfg2 = config_mod.load(explicit=path, use_env=False)
    cfg2.auto_fit = False
    config_mod.auto_fit(cfg2, verbose=False)
    for key, value in want.items():
        assert getattr(cfg2, key) == value


def test_explicit_output_cap_is_still_a_hard_cap(tmp_path):
    """When the user sets max_output_tokens themselves it is a deliberate
    physical limit, so it still bounds reply budgets."""
    path = tmp_path / "c.toml"
    path.write_text(
        '[server]\n'
        'base_url = "https://api.example.com/v1"\n'
        'model = "m"\n'
        'max_output_tokens = 8000\n'
        'max_tokens_answer = 32000\n', encoding="utf-8")
    cfg = config_mod.load(explicit=path, use_env=False)
    assert cfg.sources["max_output_tokens"].startswith("--config")
    assert cfg.sources["max_tokens_answer"].startswith("--config")
    config_mod.auto_fit(cfg, verbose=False)
    assert cfg.max_tokens_answer == 8000


def test_evict_overflow_turns_clips_oldest_result():
    """Gather-loop history grows every step; when it outgrows the window the
    oldest RESULT turn must be clipped (head kept, so the file/line header
    the coverage ledger parses survives) and the newest turns left intact --
    the run degrades gracefully instead of 400ing forever."""
    from pedacito.agent import _evict_overflow_turns
    from pedacito.llm import estimate_tokens

    cfg = config_mod.Config()
    cfg.context_tokens = 4096
    cfg.max_tokens_step = 512
    cfg.reserve_tokens = 1024      # room = 4096-512-512 = 3072 tokens
    msgs = [{"role": "system", "content": "map + task"},
            {"role": "assistant", "content": "thinking"},
            {"role": "user", "content": "RESULT:\n" + "\n".join(
                f"line {i} " + "x" * 60 for i in range(300))},
            {"role": "assistant", "content": "more thinking"},
            {"role": "user", "content": "RESULT:\nyoung " + "y" * 100}]
    room = cfg.context_tokens - cfg.max_tokens_step - cfg.reserve_tokens // 2
    total_before = sum(estimate_tokens(m["content"]) for m in msgs)
    assert total_before > room, "fixture must actually overflow"
    changed = _evict_overflow_turns(msgs, client=None, cfg=cfg)
    assert changed, "oversized history must be trimmed"
    room = cfg.context_tokens - cfg.max_tokens_step - cfg.reserve_tokens // 2
    assert sum(estimate_tokens(m["content"]) for m in msgs) <= room
    assert msgs[0]["content"] == "map + task"          # untouched
    old = next(m for m in msgs if m["content"].startswith("RESULT:\nline"))
    assert "older read dropped" in old["content"]       # clip notice present
    assert any("young" in m["content"] for m in msgs)   # newest survives


def test_evict_overflow_turns_makes_progress_on_short_turns():
    """A short RESULT turn can reach a fixed point where head+tail clipping is
    the same size as the original. Re-clipping it forever used to hang the
    gather loop right after the first trim notice (the model then looked
    "idle-unloaded" because no further request was ever sent). The eviction
    pass must drop such a turn instead of spinning."""
    from pedacito.agent import _evict_overflow_turns
    from pedacito.llm import estimate_tokens

    cfg = config_mod.Config()
    cfg.context_tokens = 4096
    cfg.max_tokens_step = 512
    cfg.reserve_tokens = 1024      # room = 4096-512-512 = 3072 tokens
    room = cfg.context_tokens - cfg.max_tokens_step - cfg.reserve_tokens // 2

    system = "x" * int((room - 310) * 3.6)
    old1 = "RESULT:\n" + "\n".join(f"line {i}" for i in range(10))
    old2 = "RESULT:\nshort"
    young = "RESULT:\nyoung " + "y" * 1000
    msgs = [{"role": "system", "content": system},
            {"role": "assistant", "content": "thinking"},
            {"role": "user", "content": old1},
            {"role": "assistant", "content": "more thinking"},
            {"role": "user", "content": old2},
            {"role": "assistant", "content": "yet more thinking"},
            {"role": "user", "content": young}]
    assert sum(estimate_tokens(m["content"]) for m in msgs) > room

    changed = _evict_overflow_turns(msgs, client=None, cfg=cfg)

    assert changed
    assert sum(estimate_tokens(m["content"]) for m in msgs) <= room
    assert any("young" in m["content"] for m in msgs)   # newest survives
    assert not any(m["content"] == old1 for m in msgs)  # stuck turn dropped


def test_gather_stops_on_repeated_no_new_source_read(tmp_path):
    """A small model can lock onto one range and emit the same `read` dozens
    of times. The duplicate adds no source, yet the old loop echoed the same
    numbered lines back and never reset progress, so it ran to the context
    budget. The gather phase must detect the no-op, stop feeding the pattern,
    and move on to the answer with the source already collected."""
    from pedacito import index as index_mod

    _write(tmp_path / "PLAN.md", "\n".join(
        f"line {i}: parkour control note" for i in range(1, 14)) + "\n")
    cfg = config_mod.Config()
    idx = index_mod.build([str(tmp_path)], cfg, client=None,
                          summarise=False, root=tmp_path, verbose=False)

    class StubClient:
        stream_truncated = False
        step_truncated = False

        def __init__(self):
            self.calls = 0

        def chat_json(self, messages, schema, max_tokens=0, **kw):
            self.calls += 1
            return {"action": "read", "file": "PLAN.md", "start": 1,
                    "end": 13, "reasoning": "Check plan for parkour control"}

    client = StubClient()
    agent = Agent(idx, client, cfg)
    gathered = agent.gather("fix parkour controls in PLAN.md", verbose=False)

    assert client.calls == 3, f"gather kept repeating reads: {client.calls}"
    assert len(agent.steps_log) == 3
    assert "parkour control note" in gathered


def test_apply_stops_after_repeated_truncation(tmp_path):
    """A model that re-emits one oversized block every round must not spin
    the continuation loop for the whole char budget (each round generates
    the full max_tokens_answer -- minutes of GPU per attempt, the 'system
    bogged down' failure). After the generic re-ask and ONE escalated
    single-chunk ask, the loop gives up."""
    from pedacito import index as index_mod
    _write(tmp_path / "a.py", "x = 1\n")
    idx = index_mod.Index(tmp_path, [])

    class StubClient:
        def __init__(self):
            self.prompts = []
            self.stream_truncated = True

        def chat(self, messages, max_tokens, schema=None, stream=False, **kw):
            self.prompts.append(messages[-1]["content"])
            # Every attempt is cut off before any complete block.
            return "--- FILE: a.py\n<<<<<<< SEARCH\n=======\n" + "y" * 200

    client = StubClient()
    agent = Agent(idx, client, config_mod.Config())
    reply = "--- FILE: a.py\n<<<<<<< SEARCH\n=======\n" + "y" * 100
    results, _ = agent.apply(reply, dry_run=False,
                             task="implement a.py", gathered="")
    assert results == [], results
    assert len(client.prompts) == 2, \
        f"expected generic re-ask + one escalated ask, got {client.prompts}"
    assert "CUT OFF" in client.prompts[0]
    assert "FIRST ~30 lines" in client.prompts[1]


def test_truncated_json_gets_token_limit_error():
    """A thinking model that spends its whole budget on reasoning returns
    no JSON at all and finish_reason='length'. chat_json must report THAT
    (with actionable advice), not the generic 'would not produce valid JSON'
    message that sends users hunting for a broken grammar.

    The salvage path is covered by test_truncated_action_is_recovered: when
    enough of the object survived to repair, the step is reused instead of
    costing another round trip."""
    p2 = _free_port()
    # No FAKE_ACTIONS: the server's default script is read-then-done, so the
    # first reply is truncated-but-recoverable; the retry escalation then
    # meets the plain `done` object, which parses fine. That proves the
    # recovery path end to end without needing a second server.
    env = dict(os.environ, FAKE_PORT=str(p2),
               FAKE_ACTIONS=json.dumps([{"action": "read", "file": "a.txt",
                                         "start": 1, "end": 2,
                                         "content": "trunc",
                                         "_finish": "length"},
                                        {"action": "done", "reasoning": ""}]),
               FAKE_REPLY="")
    proc = subprocess.Popen([sys.executable, str(FAKE)], env=env,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", p2), 0.5):
                    break
            except OSError:
                if proc.poll() is not None:
                    pytest.fail("fake_llm exited early")
                time.sleep(0.1)
        cfg = config_mod.Config()
        cfg.base_url = f"http://127.0.0.1:{p2}/v1"
        cfg.model = "fake-model"
        client = LMStudio(cfg)
        obj = client.chat_json([{"role": "user", "content": "go"}],
                               {"type": "object"}, max_tokens=64)
        assert obj.get("action") == "read", \
            f"salvage failed: {obj!r}"
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_unrecoverable_truncation_raises_token_limit_error():
    """When NOTHING parseable survives (a thinking model that emitted only
    prose before the cutoff), chat_json must raise the token-limit LLMError
    with 'token limit' and 'max_tokens_step' in the message -- not the
    generic grammar-failure text."""
    p2 = _free_port()
    s = socket.socket()
    s.bind(("127.0.0.1", p2))
    s.listen(8)
    import threading

    def serve():
        while True:
            try:
                conn, _ = s.accept()
            except OSError:
                return
            try:
                data = b""
                while b"\r\n\r\n" not in data:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                body_len = 0
                head, _, rest = data.partition(b"\r\n\r\n")
                for line in head.split(b"\r\n"):
                    if line.lower().startswith(b"content-length:"):
                        body_len = int(line.split(b":")[1])
                while len(rest) < body_len:
                    rest += conn.recv(4096)
                payload = {
                    "choices": [{"message": {
                        "content": "I need to think about this very carefully "
                                   "because the task requires reading several "
                                   "files and considering many options..."},
                        "finish_reason": "length"}], "usage": {}}
                b_ = json.dumps(payload).encode()
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json"
                             b"\r\nContent-Length: " + str(len(b_)).encode()
                             + b"\r\n\r\n" + b_)
            finally:
                conn.close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    try:
        cfg = config_mod.Config()
        cfg.base_url = f"http://127.0.0.1:{p2}/v1"
        cfg.model = "fake-model"
        client = LMStudio(cfg)
        with pytest.raises(LLMError) as exc:
            client.chat_json([{"role": "user", "content": "go"}],
                             {"type": "object"}, max_tokens=64)
        msg = str(exc.value)
        assert "token limit" in msg.lower(), msg
        assert "max_tokens_step" in msg, msg
    finally:
        s.close()


def test_truncated_no_json_gets_no_reasoning_reask():
    """A schema step that dies at the token limit BEFORE emitting any JSON
    (a thinking model spending its whole budget on the preamble -- the
    frequent local-server 'kept running past the token limit' failure) must
    be re-asked ONCE with an explicit no-reasoning instruction instead of
    erroring on the first doomed reply. A model that still cannot stop
    rambling falls through to the token-limit error."""

    from pedacito.tools import STEP_SCHEMA
    from pedacito.config import Config

    cfg = Config()
    cfg.json_mode = "object"

    class Fake(LMStudio):
        def __init__(self, replies):
            object.__init__(self)
            self.cfg = cfg
            self.replies = list(replies)
            self.n = 0
            self.seen = []

        def chat(self, messages, max_tokens, schema=None, stream=False, **kw):
            reply, fr = self.replies[min(self.n, len(self.replies) - 1)]
            self.n += 1
            self.last_finish_reason = fr
            self.seen.append(messages)
            return reply

    good = ('{"action":"done","file":"","start":0,"end":0,"pattern":"",'
            '"symbol":"","content":"","reasoning":"ok"}')
    c = Fake([("I need to think about this very carefully...", "length"),
              (good, "stop")])
    obj = c.chat_json([{"role": "user", "content": "t"}], STEP_SCHEMA,
                      max_tokens=700)
    assert obj["action"] == "done"
    assert c.n == 2                      # one nudge re-ask, then success
    nudge = [m["content"] for m in c.seen[1] if m["role"] == "user"][-1]
    assert "cut off" in nudge and "JSON" in nudge

    c2 = Fake([("prose prose prose", "length")] * 3)
    with pytest.raises(LLMError) as exc:
        c2.chat_json([{"role": "user", "content": "t"}], STEP_SCHEMA,
                     max_tokens=700)
    assert "token limit" in str(exc.value).lower()
    assert c2.n == 2                     # nudge once, then give up with advice


# --------------------------------------------------------------------- agent
class FakeIndex:
    def __init__(self, root, files):
        self.root = root
        self.files = list(files)


def test_coverage_ledger_normalises_paths(tmp_path):
    """A read whose output head carries a './'-prefixed or differently-cased
    path must still count as coverage for the index key 'pkg/mod.py' -- the
    ledger/lookup mismatch here made tasks look 'unread' and killed the
    review stage."""
    from pedacito import tools
    from pedacito.agent import _READ_HEAD_RX

    _write(tmp_path / "pkg" / "mod.py", "a = 1\nb = 2\n")
    index = FakeIndex(tmp_path, ["pkg/mod.py"])
    index.resolve_file = lambda f: "pkg/mod.py"   # minimal Index stub
    cfg = config_mod.Config()

    agent = Agent(index, client=None, cfg=cfg)
    agent._read_spans = {}
    obs = tools.tool_read(index, cfg, "./PKG/mod.py", 1, 2)
    m = _READ_HEAD_RX.match(obs)
    assert m, f"tool_read output not recognised: {obs!r}"
    key = m.group(1).strip().lower().lstrip("./")
    agent._read_spans[key] = [(int(m.group(2)), int(m.group(3)))]
    assert agent._compute_unread("edit pkg/mod.py") == []


def test_compute_unread_reports_missing(tmp_path):
    _write(tmp_path / "pkg" / "mod.py", "a = 1\nb = 2\n")
    agent = Agent(FakeIndex(tmp_path, ["pkg/mod.py"]), client=None,
                  cfg=config_mod.Config())
    agent._read_spans = {}
    missing = agent._compute_unread("edit pkg/mod.py")
    assert len(missing) == 1 and "not yet read" in missing[0]


def test_hosted_configs_are_never_clamped_or_warned():
    """The user-facing rule: hosted tier = no artificial token limits.

    A `tier = "hosted"` config (or any non-localhost base_url) must NOT get
    local-window treatment: budget_warnings stays silent, auto_fit is a
    no-op even when the declared context_tokens is tiny, and chat_json's
    truncation-retry ceiling is generous instead of being capped by
    (context - reserve). This is the regression test for deepseek runs that
    died at 'tried up to max_tokens=9600, ceiling 16384'.
    """
    import pathlib, tempfile
    from pedacito.config import load, auto_fit, budget_warnings, is_hosted, PRESETS

    def mk(text):
        p = pathlib.Path(tempfile.mkdtemp()) / "c.toml"
        p.write_text(text + '\n[hardware]\n'
                     'vram_gb = 8\n')
        return load(explicit=p)

    cfg = mk('tier = "hosted"\nbase_url = "https://api.deepseek.com/v1"'
             '\nmodel = "deepseek-chat"\njson_mode = "object"\n')
    assert is_hosted(cfg)
    assert budget_warnings(cfg) == []
    before = (cfg.max_tokens_step, cfg.max_tokens_answer, cfg.max_steps)
    assert auto_fit(cfg, verbose=False) == []
    assert (cfg.max_tokens_step, cfg.max_tokens_answer, cfg.max_steps) == before
    assert cfg.max_steps == 0           # no lookup-step limit by default

    # Even an absurdly small declared window must not shrink hosted budgets.
    cfg2 = mk('tier = "hosted"\nbase_url = "https://api.deepseek.com/v1"'
              '\nmodel = "m"\njson_mode = "object"\ncontext_tokens = 8192\n')
    assert auto_fit(cfg2, verbose=False) == []
    assert cfg2.max_tokens_answer == PRESETS["hosted"]["max_tokens_answer"]

    # Local profiles keep the old protective behaviour.
    cfg3 = mk('[server]\nbase_url = "http://localhost:1234/v1"\nmodel = "m"\n'
              'tier = "small"\ncontext_tokens = 8192\nmax_tokens_answer = 99999\n')
    assert not is_hosted(cfg3)
    assert budget_warnings(cfg3)
    auto_fit(cfg3, verbose=False)
    assert cfg3.max_tokens_answer <= cfg3.context_tokens - 512


def test_write_step_budget_and_chunking():
    """DeepSeek regression: one write step must fit the provider's real
    per-message output limit, and oversized writes are refused with chunking
    instructions instead of silently storing a truncated file.

    Covers max_write_lines (config + hosted preset), tools.dispatch's
    pre-flight guard, tool_write's post-write chunk note, and llm.repair_truncated_json
    salvaging an action from a mid-string-truncated reply.
    """
    import pathlib, tempfile
    from pedacito.config import load, PRESETS
    from pedacito import tools
    from pedacito.llm import repair_truncated_json

    def mk(text):
        p = pathlib.Path(tempfile.mkdtemp()) / "c.toml"
        p.write_text(text + '\n[hardware]\n'
                     'vram_gb = 8\n')
        return load(explicit=p)

    cfg = mk('tier = "hosted"\nbase_url = "https://api.deepseek.com/v1"'
             '\nmodel = "deepseek-chat"\njson_mode = "object"\nallow_write = true\n')
    assert cfg.allow_write is True
    assert cfg.max_write_lines == PRESETS["hosted"]["max_write_lines"] == 120
    # A step budget that fits under deepseek's ~8k-token message ceiling.
    assert cfg.max_tokens_step <= 8000

    class FakeIndex:
        root = pathlib.Path(tempfile.mkdtemp())
        files = []
        def resolve_file(self, f):
            return None

    big = "\n".join(f"line {i}" for i in range(300))
    obs = tools.dispatch(FakeIndex(), cfg, {"action": "write",
                                            "file": "a.txt", "content": big})
    assert obs.startswith("ERROR") and "300 lines" in obs and "chunks" in obs

    small = "\n".join(f"line {i}" for i in range(5))
    obs2 = tools.dispatch(FakeIndex(), cfg, {"action": "write",
                                             "file": "b.txt", "content": small})
    assert obs2.startswith("CREATED b.txt (5 lines)")

    # Post-write note when a written file still exceeds the step budget.
    cfgw = mk('tier = "hosted"\nbase_url = "https://api.deepseek.com/v1"'
              '\nmodel = "m"\njson_mode = "object"\nallow_write = true\n')
    idx = FakeIndex()
    idx.root.mkdir(parents=True, exist_ok=True)
    obs3 = tools.tool_write(idx, cfgw, "c.txt", big)
    assert "WROTE c.txt (300 lines)" in obs3 or "CREATED c.txt (300 lines)" in obs3
    assert "Chunking is ON" in obs3

    # Truncated-reply salvage: action + args survive a cut-off reasoning tail.
    text = ('{"action": "read", "file": "a.txt", "start": 1, "end": 10, '
            '"reasoning": "I will now read the whole file and then explain')
    fixed = repair_truncated_json(text)
    assert fixed is not None
    import json as _json
    obj = _json.loads(fixed)
    assert obj["action"] == "read" and obj["file"] == "a.txt"


def test_json_mode_aliases_and_deepseek_autodetect(tmp_path, monkeypatch):
    """DeepSeek's docs say response_format {"type": "json_object"}, so users
    reasonably write json_mode = "json_object" in their config -- every
    spelling must land on the internal "object" mode, and a DeepSeek target
    must never be left running with json_schema (which it rejects)."""
    for raw, want in (("json_object", "object"), ("JSON_OBJECT", "object"),
                      ("json", "object"), ("  object ", "object"),
                      ("schema", "schema"), ("json_schema", "schema")):
        assert config_mod._coerce("json_mode", raw) == want
    with pytest.raises(config_mod.ConfigError):
        config_mod._coerce("json_mode", "json-object")

    def load(text):
        p = tmp_path / f"c{abs(hash(text))}.toml"
        p.write_text(text + '\n[hardware]\n'
                     'vram_gb = 8\n', encoding="utf-8")
        for k in [k for k in os.environ if k.startswith("PEDACITO_")]:
            monkeypatch.delenv(k, raising=False)
        return config_mod.load(explicit=p, use_env=True)

    # remote host, no json_mode written -> auto-switch to object
    cfg = load('[server]\nbase_url = "https://api.deepseek.com/v1"\n'
               'model = "deepseek-chat"\napi_key = "sk-x"\n')
    assert cfg.json_mode == "object"
    assert "auto-detected" in cfg.sources["json_mode"]
    assert config_mod.is_hosted(cfg)
    assert config_mod.budget_warnings(cfg) == []

    # deepseek model served LOCALLY (localhost) -> still object mode
    cfg = load('[server]\nbase_url = "http://localhost:1234/v1"\n'
               'model = "deepseek-r1-distill-27b"\n')
    assert cfg.json_mode == "object"

    # explicit schema + deepseek -> loud switch, never silent failure
    cfg = load('[server]\nbase_url = "https://api.deepseek.com/v1"\n'
               'model = "deepseek-chat"\njson_mode = "schema"\n')
    assert cfg.json_mode == "object"

    # plain local model untouched
    cfg = load('[server]\nbase_url = "http://localhost:1234/v1"\n'
               'model = "qwen3-27b"\n')
    assert cfg.json_mode == "schema"
    assert not config_mod.is_hosted(cfg)


def test_single_quoted_dict_reply_salvage():
    """Some hosted models in json_object mode emit Python-style dicts with
    single quotes. chat_json must convert them instead of burning all three
    retries and dying with 'would not produce valid JSON'."""
    reply = ("{'action': 'read', 'file': 'subj-imp/index.html', "
             "'start': 405, 'end': 445, "
             "'reasoning': 'See subj-imp script tag'}")
    client = LMStudio.__new__(LMStudio)          # no network needed
    client.cfg = config_mod.Config(base_url="http://x.invalid/v1", model="m",
                        json_mode="object")
    client.stats = {"calls": 0, "prompt_tokens": 0,
                    "completion_tokens": 0, "seconds": 0}
    client.last_finish_reason = "stop"
    schema = {"properties": {"action": {}, "file": {}}}
    seen = {"n": 0}

    def fake_chat(messages, max_tokens, schema=None, **kw):
        seen["n"] += 1
        return reply
    client.chat = fake_chat
    obj = client.chat_json([{"role": "user", "content": "hi"}],
                           schema=schema, max_tokens=100)
    assert obj["action"] == "read"
    assert obj["file"] == "subj-imp/index.html"
    assert seen["n"] == 1                     # salvaged on the first try


def test_step_truncation_salvage_no_escalation():
    """A gather-step reply cut off inside 'reasoning' must be accepted on the
    FIRST salvage attempt (schema steps have no large fields to grow into),
    set client.step_truncated so gather() can warn the model, and never burn
    the old 7000->14000->28000 escalation loop that killed DeepSeek runs.
    A schema WITH a large field (answer-style) must still escalate-retry."""
    import json as _json
    from pedacito.llm import LMStudio
    from pedacito.tools import STEP_SCHEMA
    from pedacito.config import Config

    cfg = Config(); cfg.json_mode = "object"

    class Fake(LMStudio):
        def __init__(self, replies, finish="length"):
            object.__init__(self)
            self.cfg = cfg; self.replies = list(replies); self.n = 0
            self.last_finish_reason = finish
        def chat(self, messages, max_tokens, schema=None, stream=False, **kw):
            r = self.replies[min(self.n, len(self.replies) - 1)]; self.n += 1
            return r

    cut = ('{"action":"read","file":"a.txt","start":1,"end":40,'
           '"pattern":"","symbol":"","content":"","reasoning":"See the im')
    c = Fake([cut])
    obj = c.chat_json([{"role": "user", "content": "t"}], STEP_SCHEMA, max_tokens=7000)
    assert obj["action"] == "read" and obj["file"] == "a.txt"
    assert c.step_truncated is True
    assert c.n == 1                      # accepted immediately, no retry loop

    good = _json.dumps({"action": "done", "file": "", "start": 0, "end": 0,
                        "pattern": "", "symbol": "", "content": "",
                        "reasoning": "ok"})
    c2 = Fake([good], finish="stop")
    obj2 = c2.chat_json([{"role": "user", "content": "t"}], STEP_SCHEMA, max_tokens=7000)
    assert obj2["action"] == "done" and c2.step_truncated is False and c2.n == 1

    big = {"type": "object",
           "properties": {"action": {"type": "string"},
                          "content": {"type": "string", "maxLength": 16000}},
           "required": ["action", "content"], "additionalProperties": False}
    # content cut mid-string (closing quote present, like a real DeepSeek
    # cut-off): the salvage is PARTIAL content -- acting on half a file would
    # corrupt it -- so the reply is NOT accepted immediately. It escalates
    # once, and only at the retry ceiling is the last partial reply salvaged
    # whole (flagged step_truncated) rather than raising blind.
    cut_q = '{"action":"edit","content":"' + ("y" * 9000) + '"'
    c3 = Fake([cut_q])
    obj3 = c3.chat_json([{"role": "user", "content": "t"}], big, max_tokens=8000)
    assert obj3["action"] == "edit" and len(obj3["content"]) > 5000
    assert c3.n >= 2                     # escalated before accepting partials
    assert c3.step_truncated is True     # gather() will warn the model

    # A truncated whole-file write that survives salvage is PARTIAL content:
    # it must never be executed blind (half a file on disk). The step is
    # flagged step_truncated and the partial reply is returned so dispatch()
    # in agent.py can refuse it with the chunked-append recipe instead.
    c4 = Fake(['{"action":"edit","content":"' + ("x" * 9000)])
    obj4 = c4.chat_json([{"role": "user", "content": "t"}], big, max_tokens=8000)
    assert obj4["action"] == "edit" and len(obj4["content"]) > 5000
    assert c4.step_truncated is True     # dispatch refuses + teaches chunking
    assert c4.n >= 2                     # escalated before accepting partials


# ------------------------------------------------------------------- scaffold
def test_new_dry_run_writes_nothing(tmp_path):
    """A bare `pedacito new` (no --apply/--review/--plan-only) must be a true
    dry run: it prints the plan and a skeleton preview but leaves the target
    directory empty. The old code wrote the skeleton and the build passes
    unconditionally, so 'new' and 'new --apply' produced identical trees."""
    from pedacito import scaffold

    class Stub:
        stats = {"calls": 0, "completion_tokens": 0, "seconds": 0}
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            return {"project_name": "md2html", "summary": "markdown to html",
                    "files": [{"path": "main.py", "purpose": "CLI entry"},
                              {"path": "parser.py", "purpose": "parse markdown"}]}
        def chat(self, *a, **k):
            return ""

    code = scaffold.run_new(tmp_path, config_mod.Config(), Stub(),
                            scaffold.Spec("markdown to html"),
                            apply=False, review=False, plan_only=False,
                            verbose=False)
    assert code == 0
    assert list(tmp_path.iterdir()) == [], "dry run wrote files to disk"


def test_starter_pyproject_has_script_entry():
    """The starter pyproject must wire a console-script entry point so a
    generated CLI project is actually runnable after `pip install -e .`."""
    from pedacito.scaffold import STARTER_PYPROJECT
    data = tomllib.loads(STARTER_PYPROJECT.format(
        name="md2html", summary="markdown to html"))
    assert data["project"]["scripts"] == {"md2html": "main:main"}


def test_first_import_error_detects_cross_file_mismatch(tmp_path):
    """Files are generated independently, so one module can import a name
    another defines differently. The post-build import check must surface
    that mismatch instead of leaving the project to fail on first run."""
    from pedacito import scaffold
    pkg = tmp_path / "md2html"
    pkg.mkdir()
    (tmp_path / "main.py").write_text(
        "from md2html.parser import parse_markdown\n")
    (pkg / "__init__.py").write_text("from .parser import parse_markdown\n")
    (pkg / "parser.py").write_text("def parse(text):\n    return text\n")
    plan = scaffold.Plan("md2html", "y", files=[
        scaffold.PlannedFile("main.py", ""),
        scaffold.PlannedFile("md2html/__init__.py", ""),
        scaffold.PlannedFile("md2html/parser.py", ""),
    ], spec=scaffold.Spec("y"))
    err = scaffold._first_import_error(tmp_path, plan)
    assert err and "parse_markdown" in err, err


def test_build_interface_fixes_contract_in_one_call(tmp_path):
    """The interface step writes signature stubs for every source file in a
    single call, so the per-file build implements a fixed contract instead of
    inventing a different API per file (the cause of cross-file import bugs)."""
    from pedacito import scaffold

    class Stub:
        def chat_json(self, messages, schema, max_tokens=0, **kw):
            assert schema["required"] == ["files"]
            return {"files": [
                {"path": "parser.py",
                 "code": "def parse_markdown(text):\n    \"\"\"parse\"\"\"\n    raise NotImplementedError\n"},
                {"path": "renderer.py",
                 "code": "def render_html(blocks):\n    \"\"\"render\"\"\"\n    raise NotImplementedError\n"},
            ]}

    plan = scaffold.Plan("md2html", "y", files=[
        scaffold.PlannedFile("parser.py", "parse"),
        scaffold.PlannedFile("renderer.py", "render"),
    ], spec=scaffold.Spec("y"))
    stubs = scaffold.build_interface(plan, Stub(), config_mod.Config(),
                                     verbose=False)
    assert stubs["parser.py"].startswith("def parse_markdown")
    assert stubs["renderer.py"].startswith("def render_html")

    # Those stubs are what the skeleton writes for source files.
    scaffold.apply_skeleton(tmp_path, plan, dry_run=False,
                            verbose=False, stubs=stubs)
    assert "parse_markdown" in (tmp_path / "parser.py").read_text()





def test_lemonade_model_metadata_is_absorbed():
    """Lemonade extends OpenAI /v1/models with `size` (GB), context_length,
    and max_context_window instead of LM Studio's size_bytes. Those must
    feed the same hardware estimate, or Lemonade silently loses the feature."""
    from pedacito.llm import LMStudio, _server_root

    cfg = config_mod.Config(base_url="http://localhost:13305/api/v1",
                            model="Qwen3-0.6B-GGUF")
    got = LMStudio(cfg)._absorb_model_list([{
        "id": "Qwen3-0.6B-GGUF",
        "owned_by": "lemonade",
        "recipe": "llamacpp",
        "size": 0.38,
        "context_length": 8192,
        "max_context_window": 40960,
    }], "openai-models")
    assert got["source"] == "lemonade"
    assert got["size_bytes"] == int(0.38 * 1024 ** 3)
    assert got["context_length"] == 8192
    assert got["max_context_length"] == 40960
    # Both API prefixes are stripped before native endpoints are built.
    assert _server_root("http://host:13305/api/v1") == "http://host:13305"
    assert _server_root("http://host:13305/v1/") == "http://host:13305"


def test_tier_budgets_are_monotonic():
    """Each larger local tier must be at least as generous as the one below
    it. The old presets had large's step (1000) and build (8000) SMALLER than
    medium's (1500 / 16000), which made the tier name misleading."""
    keys = ("max_tokens_summary", "max_tokens_step", "max_tokens_answer",
            "max_tokens_plan", "max_tokens_build", "flat_map_token_budget",
            "max_gathered_chars", "max_read_lines", "max_spec_chars",
            "max_new_files", "max_steps", "context_tokens", "reserve_tokens",
            "repair_rounds")
    for lower, higher in (("small", "medium"), ("medium", "large"),
                          ("large", "hosted")):
        for key in keys:
            assert config_mod.PRESETS[higher][key] >= \
                config_mod.PRESETS[lower][key], \
                f"{key}: {higher} ({config_mod.PRESETS[higher][key]}) < " \
                f"{lower} ({config_mod.PRESETS[lower][key]})"


def test_hardware_budgets_override_tier_values():
    """A medium profile is only a fallback once the model size and VRAM are
    known: 19 GB of weights on 48 GB VRAM with a 32k window has enough room
    for larger reply budgets than the medium preset, and an explicit user
    value must still win."""
    cfg = config_mod.Config()
    cfg.base_url = "http://127.0.0.1:1/v1"
    cfg.model = "m"
    cfg.vram_gb = 48
    for key, value in config_mod.PRESETS["medium"].items():
        setattr(cfg, key, value)
        cfg.sources[key] = "tier 'medium' preset"
    cfg.context_tokens = 32768
    cfg.sources["context_tokens"] = "server"
    cfg.auto_detect_context = False
    cfg.model_size_bytes = 19_000_000_000
    config_mod.auto_fit(cfg, verbose=False)
    assert cfg.max_tokens_step > config_mod.PRESETS["medium"]["max_tokens_step"]
    assert cfg.max_tokens_answer > config_mod.PRESETS["medium"]["max_tokens_answer"]
    assert cfg.max_tokens_build > config_mod.PRESETS["medium"]["max_tokens_build"]
    assert cfg.max_gathered_chars > \
        config_mod.PRESETS["medium"]["max_gathered_chars"]

    cfg2 = config_mod.Config()
    cfg2.base_url = "http://127.0.0.1:1/v1"
    cfg2.model = "m"
    cfg2.vram_gb = 48
    for key, value in config_mod.PRESETS["medium"].items():
        setattr(cfg2, key, value)
        cfg2.sources[key] = "tier 'medium' preset"
    cfg2.context_tokens = 32768
    cfg2.sources["context_tokens"] = "server"
    cfg2.auto_detect_context = False
    cfg2.model_size_bytes = 19_000_000_000
    cfg2.max_tokens_answer = 2000
    cfg2.sources["max_tokens_answer"] = "config file"
    config_mod.auto_fit(cfg2, verbose=False)
    assert cfg2.max_tokens_answer == 2000
    assert cfg2.sources["max_tokens_answer"] == "config file"


def test_explicit_max_tokens_override_vram_calculation_and_tier():
    """The user's explicit budget wins over both the tier default and the
    VRAM calculation (as long as it fits the server window)."""
    cfg = config_mod.Config()
    cfg.base_url = "http://127.0.0.1:1/v1"
    cfg.model = "m"
    cfg.vram_gb = 24
    for key, value in config_mod.PRESETS["medium"].items():
        setattr(cfg, key, value)
        cfg.sources[key] = "tier 'medium' preset"
    cfg.context_tokens = 32768
    cfg.sources["context_tokens"] = "config file"
    cfg.auto_detect_context = False
    cfg.model_size_bytes = 19_000_000_000
    cfg.max_tokens_answer = 20000
    cfg.sources["max_tokens_answer"] = "config file"
    config_mod.auto_fit(cfg, verbose=False)
    assert cfg.max_tokens_answer == 20000
    assert cfg.sources["max_tokens_answer"] == "config file"
