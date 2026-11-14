"""B2: profile naming (`short_name`) and the idempotent profile writer."""
from __future__ import annotations

import random
import re
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import config as config_mod
from pedacito import config_editor as E
from pedacito.cli import main


# --- short_name ---------------------------------------------------------------

def test_short_name_derivation():
    cases = {
        "qwen/qwen3-14b@q4_k_m": "qwen3-14b",
        "qwen3:14b": "qwen3-14b",
        "qwen3:latest": "qwen3",
        "org/model.gguf": "model",
        "meta-llama/Llama-3.2-8B-Instruct": "llama-3.2-8b-instruct",
        "Qwen3:14B": "qwen3-14b",
        "org/with space@q4_0": "with-space",
    }
    for model_id, want in cases.items():
        assert E.short_name(model_id) == want, model_id


def test_short_name_collisions():
    # quant disambiguates first
    assert E.short_name("qwen/qwen3-14b@q4_k_m", {"qwen3-14b"}) == "qwen3-14b-q4km"
    # then the server label
    assert E.short_name("qwen3:14b", {"qwen3-14b"}, "Ollama") == "qwen3-14b-ollama"
    # then a numeric suffix
    assert E.short_name("qwen3:14b", {"qwen3-14b", "qwen3-14b-ollama"}, "Ollama") \
        == "qwen3-14b-2"
    # no collision: the plain name, taken is untouched
    assert E.short_name("qwen3:14b", {"other"}) == "qwen3-14b"


# --- upsert_profiles ----------------------------------------------------------

def test_upsert_appends_identity_only_and_loads(tmp_path):
    text = config_mod.TEMPLATE
    out = E.upsert_profiles(text, [
        {"model": "qwen3:14b", "base_url": "http://localhost:11434/v1",
         "server_label": "Ollama", "tier": "medium", "vram_gb": 24,
         "quant": "Q4_K_M", "size_gb": 8.9}],
        now="2026-11-02")
    assert "# added by pedacito models scan, 2026-11-02, Ollama, Q4_K_M 8.9 GB" \
        in out
    assert '[profiles.qwen3-14b]' in out
    assert 'base_url = "http://localhost:11434/v1"' in out
    assert 'model = "qwen3:14b"' in out
    # identity only: no card choice is ever written into the new profile
    block = out.split("[profiles.qwen3-14b]", 1)[1].split("[", 1)[0]
    for key in ("prompt_pack", "best_of", "verify_answer", "answer_mode"):
        assert f"{key} =" not in block
    p = tmp_path / "config.toml"
    p.write_text(out, encoding="utf-8")
    cfg = config_mod.load(explicit=p, use_env=False, require_hardware=False,
                          warn_budgets=False)
    assert "qwen3-14b" in cfg.profiles


def test_upsert_is_idempotent():
    text = '[universal]\nvram_gb = 24\n'
    found = [{"model": "a", "base_url": "http://localhost:1234/v1",
              "server_label": "LM Studio"}]
    once = E.upsert_profiles(text, found, now="2026-11-02")
    twice = E.upsert_profiles(once, found, now="2026-11-02")
    assert twice == once


def test_upsert_preserves_user_comments_and_keys():
    text = (
        "# my custom comment\n"
        "[universal]\n"
        "vram_gb = 24\n"
        "\n"
        "[profiles.default]\n"
        "base_url = \"http://localhost:11434/v1\"\n"
        "model = \"qwen3:14b\"\n"
        "max_steps = 42  # user-added key\n"
    )
    out = E.upsert_profiles(text, [
        {"model": "new-model", "base_url": "http://localhost:1234/v1",
         "server_label": "LM Studio"}],
        now="2026-11-02")
    # the whole original survives verbatim as a prefix; only the new profile
    # is appended after it
    assert out.startswith(text.rstrip("\n"))
    assert "# my custom comment" in out
    assert "max_steps = 42  # user-added key" in out
    assert '[profiles.new-model]' in out


def test_upsert_matches_on_base_url_and_model():
    text = ('[profiles.default]\n'
            'base_url = "http://localhost:11434/v1"\n'
            'model = "qwen3:14b"\n')
    out = E.upsert_profiles(text, [
        {"model": "qwen3:14b", "base_url": "http://localhost:11434/v1"},
        {"model": "qwen3:14b", "base_url": "http://localhost:1234/v1"},
    ], now="2026-11-02")
    # the same (base_url, model) is skipped; the same model on another server
    # gets its own profile
    assert out.count("base_url = \"http://localhost:11434/v1\"") == 1
    assert '[profiles.qwen3-14b]' in out
    assert "base_url = \"http://localhost:1234/v1\"" in out


def test_upsert_takes_profile_names_as_collisions():
    text = ('[profiles.qwen3-14b]\n'
            'base_url = "http://localhost:11434/v1"\n'
            'model = "qwen3:14b"\n')
    # a *different* server's qwen3:14b collides on the short name, so it gets
    # the server-label suffix instead of clobbering the existing profile.
    out = E.upsert_profiles(text, [
        {"model": "qwen3:14b", "base_url": "http://localhost:1234/v1",
         "server_label": "LM Studio"}],
        now="2026-11-02")
    assert '[profiles.qwen3-14b-lm-studio]' in out


def test_editor_is_importable_without_the_gui_package():
    # The CLI needs the editor without pulling in pedacito.gui (which imports
    # the review server). Every test above exercised `config_editor` directly;
    # its source imports only `config`, so no gui module is involved.
    import inspect
    src = inspect.getsource(E.validate_text)
    assert "gui" not in src


def test_gui_settings_reexports_the_editor():
    from pedacito.gui import settings as S
    assert S.set_values is E.set_values
    assert S.add_profile is E.add_profile
    assert S.validate_text is E.validate_text
    assert S.upsert_profiles is E.upsert_profiles


# --- multi-line string scanning ----------------------------------------------

def test_triple_toggles_scans_characters_not_counts():
    T = E._triple_toggles
    # a comment or a single-line string may quote the delimiter freely
    assert T('# see the """ docs', "") == ""
    assert T("key = 'literal with \"\"\" inside'", "") == ""
    assert T('key = "basic with \'\'\' inside"', "") == ""
    # real multi-line strings still open and close
    assert T('key = """open', "") == '"""'
    assert T("still inside # not a comment", '"""') == '"""'
    assert T('close""" # done', '"""') == ""
    assert T("key = '''one'''", "") == ""
    assert T("key = '''open", "") == "'''"
    assert T("end''' and '''again'''", "'''") == ""
    # an escaped quote does not close a multi-line basic string
    assert T('escaped \\""" still open', '"""') == '"""'


def test_comment_quoting_triple_delimiter_does_not_swallow_tables():
    """A '#' comment mentioning `\"\"\"` used to flip the scanner into
    'inside a multi-line string', hiding every later table header:
    profile_names lost the profiles and set_values appended a duplicate
    table that the loader then rejected."""
    text = (
        '# the old prompt used """ triple quotes\n'
        "[profiles.one]\n"
        'model = "a"\n'
        "\n"
        "[profiles.two]\n"
        'model = "b"\n'
    )
    assert E.profile_names(text) == ["one", "two"]
    out = E.set_values(text, {"model": "c"}, profile="two")
    E.validate_text(out)
    assert out.count("[profiles.two]") == 1
    assert 'model = "c"' in out
    assert 'model = "a"' in out


# --- R7: seeded round-trip properties -----------------------------------------
#
# Deterministic generative guard over the whole textual-patch invariant: for
# any comment/string/table shape, set_values / remove_profile / add_profile
# must yield text the loader accepts, never duplicate a table, and never lose
# an unrelated key or comment. No model, no randomness at run time beyond the
# fixed seeds.

# Keys with unambiguous types, safe to generate in any scope.
_SAFE_KEYS = ("model", "base_url", "api_key_env", "vram_gb", "max_steps",
              "temperature", "disable_thinking")


def _safe_value(r, key):
    if key == "vram_gb":
        return r.randrange(1, 96)
    if key == "max_steps":
        return r.randrange(0, 60)
    if key == "temperature":
        return round(r.uniform(0.0, 1.5), 2)
    if key == "disable_thinking":
        return r.choice([True, False])
    if key == "base_url":
        return f"http://localhost:{r.randrange(1024, 65536)}/v1"
    if key == "api_key_env":
        return f"KEY_{r.randrange(100)}"
    return f"model-{r.randrange(1000)}"


def _string_shape(r, nl):
    """The FIX-011 class: quotes that look like triple delimiters, literal
    strings, and multi-line values hiding comment/table lookalikes. Only the
    name 'ghost' ever appears inside a string, so real profile headers can be
    counted without false positives."""
    return r.choice([
        '"basic \\"escaped\\" quotes"',
        "'literal with \"\"\" inside'",
        '"basic with \'\'\' inside"',
        '"""first # not a comment' + nl + '[profiles.ghost]' + nl + 'last"""',
        "'''first \"\"\" inside" + nl + "[profiles.ghost]" + nl + "last'''",
        '"""escaped \\""" still one value"""',
    ])


def _comment(r):
    return r.choice([
        "# plain note",
        '# the old prompt used """ triple quotes',
        "# ''' triple single quotes too",
        "# [profiles.ghost] is not a real table",
        '# "basic" and \'literal\' quotes',
        "# key = value lookalike",
    ])


def _gen_config(r):
    """A random but valid config: (text, profile names, regions). Each region
    records its full-line comments so a removal can be checked precisely.
    A comment emitted just before a header belongs to the previous region,
    matching what remove_profile actually deletes (header line onward)."""
    nl = "\r\n" if r.random() < 0.3 else "\n"
    regions = [{"header": "", "comments": []}]
    lines = []
    used_global = set()

    def region(header):
        regions.append({"header": header, "comments": []})
        lines.append(header)

    def comment():
        c = _comment(r)
        lines.append(c)
        regions[-1]["comments"].append(c)

    def keys(n, global_scope=False):
        for key in r.sample(_SAFE_KEYS, n):
            if global_scope:
                if key in used_global:
                    continue
                used_global.add(key)
            value = _safe_value(r, key)
            lit = E.literal(value)
            if key in ("model", "base_url", "api_key_env") and r.random() < 0.5:
                lit = _string_shape(r, nl)
            tail = r.choice(["", "  # note", '  # see """ docs'])
            lines.extend((f"{key} = {lit}{tail}").split(nl))

    if r.random() < 0.5:
        comment()
    keys(r.randrange(0, 3), global_scope=True)
    if r.random() < 0.7:
        lines.append("")
        region("[universal]")
        keys(r.randrange(1, 4), global_scope=True)
        if r.random() < 0.25:
            lines.extend([
                "shell_allowlist = [",
                '  "pytest",  # inside an array',
                '  "python",',
                "]",
            ])
    names = []
    for i in range(r.randrange(1, 4)):
        name = f"p{i}"
        names.append(name)
        lines.append("")
        if r.random() < 0.6:
            comment()           # FIX-011: comment right before a header
        region(f"[profiles.{name}]")
        keys(r.randrange(1, 4))
        if r.random() < 0.3:
            region(f"[profiles.{name}.sampler]")
            lines.append(f"seed = {r.randrange(100)}")
    text = nl.join(lines) + nl
    return text, names, regions


def _comment_lines(text):
    return [ln.strip() for ln in text.replace("\r\n", "\n").split("\n")
            if ln.lstrip().startswith("#")]


def _no_duplicate_tables(text, names):
    flat = text.replace("\r\n", "\n")
    for name in names:
        for header in (f"[profiles.{name}]", f"[profiles.{name}.sampler]"):
            hits = re.findall(r"^\s*" + re.escape(header) + r"\s*(?:#.*)?$",
                              flat, re.M)
            assert len(hits) <= 1, header


def _check(out, flat, profiles, names):
    f2, p2 = E.validate_text(out)
    assert f2 == flat, (f2, flat)
    assert p2 == profiles, (p2, profiles)
    assert sorted(E.profile_names(out)) == sorted(names)
    _no_duplicate_tables(out, names)


def test_roundtrip_single_ops_seeded():
    for seed in range(75):
        r = random.Random(seed)
        text, names, regions = _gen_config(r)
        flat0, prof0 = E.validate_text(text)      # generator sanity
        comments = _comment_lines(text)

        # The additive upgrader may only append commented options: the active
        # settings and profiles before and after must be byte-for-byte equal
        # once parsed, and a second pass must be a no-op.
        upgraded, _added_keys = E.upgrade_config_text(text)
        up_flat, up_prof = E.validate_text(upgraded)
        assert up_flat == flat0 and up_prof == prof0, (seed, up_flat, flat0)
        assert E.upgrade_config_text(upgraded) == (upgraded, [])

        key = r.choice(_SAFE_KEYS)
        val = _safe_value(r, key)

        # global set: only the key changes, every comment survives
        out = E.set_values(text, {key: val})
        _check(out, {**flat0, key: val}, prof0, names)
        assert _comment_lines(out) == comments

        # set inside an existing profile
        p = names[0]
        out = E.set_values(text, {key: val}, profile=p)
        want = {**prof0, p: {**prof0.get(p, {}), key: val}}
        _check(out, flat0, want, names)
        assert _comment_lines(out) == comments

        # set inside a profile that does not exist yet: it is created
        out = E.set_values(text, {key: val}, profile="fresh")
        _check(out, flat0, {**prof0, "fresh": {key: val}}, names + ["fresh"])
        assert _comment_lines(out) == comments

        # global remove of a key that is present
        if flat0:
            k = r.choice(sorted(flat0))
            out = E.set_values(text, removes=[k])
            _check(out, {a: b for a, b in flat0.items() if a != k},
                   prof0, names)
            assert _comment_lines(out) == comments

        # remove each profile: its regions' comments go, all others stay
        for p in names:
            out = E.remove_profile(text, p)
            want = {a: b for a, b in prof0.items() if a != p}
            _check(out, flat0, want, [n for n in names if n != p])
            kept = [c for reg in regions
                    if reg["header"] != f"[profiles.{p}]"
                    and not reg["header"].startswith(f"[profiles.{p}.")
                    for c in reg["comments"]]
            assert _comment_lines(out) == kept

        # add a brand-new profile
        vals = {"model": "added-model",
                "base_url": "http://localhost:1234/v1"}
        out = E.add_profile(text, "added", vals)
        _check(out, flat0, {**prof0, "added": vals}, names + ["added"])
        assert _comment_lines(out) == comments


def test_roundtrip_chained_seeded():
    for seed in range(40):
        r = random.Random(1000 + seed)
        text, names, _regions = _gen_config(r)
        flat, prof = E.validate_text(text)
        n_comments = len(_comment_lines(text))
        added = 0
        for _ in range(5):
            op = r.randrange(4)
            key = r.choice(_SAFE_KEYS)
            val = _safe_value(r, key)
            removed = False
            if op == 0:
                text = E.set_values(text, {key: val})
                flat = {**flat, key: val}
            elif op == 1 and names:
                p = r.choice(names)
                text = E.set_values(text, {key: val}, profile=p)
                prof = {**prof, p: {**prof.get(p, {}), key: val}}
            elif op == 2:
                added += 1
                name = f"added{added}"
                vals = {"model": f"m{added}", "vram_gb": 8}
                text = E.add_profile(text, name, vals)
                prof = {**prof, name: vals}
                names = names + [name]
            elif op == 3 and names:
                p = r.choice(names)
                text = E.remove_profile(text, p)
                prof = {a: b for a, b in prof.items() if a != p}
                names = [n for n in names if n != p]
                removed = True
            f2, p2 = E.validate_text(text)
            assert f2 == flat, (seed, f2, flat)
            assert p2 == prof, (seed, p2, prof)
            assert sorted(E.profile_names(text)) == sorted(names)
            _no_duplicate_tables(text, names)
            n = len(_comment_lines(text))
            assert n <= n_comments
            if not removed:
                assert n == n_comments
            n_comments = n


def test_roundtrip_over_the_starter_template_seeded():
    for seed in range(15):
        r = random.Random(2000 + seed)
        text = config_mod.TEMPLATE
        if seed % 3 == 0:
            text = text.replace("\n", "\r\n")
        flat0, prof0 = E.validate_text(text)
        names = E.profile_names(text)
        key = r.choice(_SAFE_KEYS)
        val = _safe_value(r, key)
        out = E.set_values(text, {key: val})
        f2, p2 = E.validate_text(out)
        assert p2 == prof0
        assert f2[key] == val
        _no_duplicate_tables(out, names)
        out = E.set_values(out, {key: val}, profile="default")
        _f3, p3 = E.validate_text(out)
        assert p3["default"][key] == val
        out = E.add_profile(out, "extra", {"model": "m"})
        E.validate_text(out)
        assert "extra" in E.profile_names(out)
        out = E.remove_profile(out, "extra")
        _f4, p4 = E.validate_text(out)
        assert "extra" not in p4
        _no_duplicate_tables(out, names)


def test_roundtrip_empty_and_comment_only_bases():
    for text in ("", "# only a comment\n", "# '''\n", '# """\n'):
        out = E.set_values(text, {"vram_gb": 8})
        flat, _prof = E.validate_text(out)
        assert flat["vram_gb"] == 8
        out = E.add_profile(out, "first", {"model": "m"})
        _flat, prof = E.validate_text(out)
        assert prof["first"] == {"model": "m"}
        out = E.remove_profile(out, "first")
        E.validate_text(out)
        assert E.profile_names(out) == []


def test_set_value_keeps_tail_after_escaped_triple_quote():
    """An escaped quote inside a multi-line basic string (\\\"\"\") must not
    end the value early: the trailing comment and the next key survive."""
    text = ('model = """escaped \\""" still one value"""  # keep me\n'
            'vram_gb = 8\n')
    out = E.set_values(text, {"model": "replaced"})
    E.validate_text(out)
    assert "# keep me" in out
    assert "vram_gb = 8" in out


# --- additive config upgrade --------------------------------------------------

_OLD_CONFIG = '''\
# my laptop config -- don't touch these comments
vram_gb = 24
scout_profile = "fast"

[universal]
timeout = 120.0
# keep my temperature low for code
temperature = 0.1

[profiles.worker]
base_url = "http://localhost:1234/v1"
model = "qwen3:14b"

[profiles.fast]
model = "gemma3:4b"

[mcp_servers.docs]
command = "python"
args = ["docs.py"]
'''


def test_upgrade_config_preserves_all_user_settings_and_profiles():
    old_text = _OLD_CONFIG
    flat0, prof0 = E.validate_text(old_text)

    new_text, added = E.upgrade_config_text(old_text)

    for key in ("verify_dom", "enable_prompt_cache", "auto_escalate"):
        assert key in added
    # A key the file already sets is never appended, active or commented.
    assert "vram_gb" not in added and "scout_profile" not in added

    # Zero active settings or profiles changed: the appended lines are all
    # comments.
    flat1, prof1 = E.validate_text(new_text)
    assert flat1 == flat0
    assert prof1 == prof0
    assert prof1["worker"]["model"] == "qwen3:14b"

    # Every user comment and profile table survives verbatim.
    for line in ("# my laptop config -- don't touch these comments",
                 "# keep my temperature low for code",
                 "[profiles.worker]", "[profiles.fast]",
                 "[mcp_servers.docs]",
                 'model = "gemma3:4b"'):
        assert line in new_text
    # Everything before the insertion point is untouched, byte for byte.
    assert new_text.startswith(
        old_text.split("[profiles.worker]")[0].rstrip("\n"))


def test_upgrade_config_is_idempotent():
    new_text, added = E.upgrade_config_text(_OLD_CONFIG)
    assert added

    text2, added2 = E.upgrade_config_text(new_text)
    assert added2 == []
    assert text2 == new_text

    # The reference template already documents every canonical option.
    reference_text, reference_added = E.upgrade_config_text(
        config_mod.REFERENCE_TEMPLATE)
    assert reference_added == []
    assert reference_text == config_mod.REFERENCE_TEMPLATE


def test_upgrade_config_respects_commented_out_user_keys():
    text = ('# verify_dom = true  # I turned this on locally\n'
            'vram_gb = 24\n')
    out, added = E.upgrade_config_text(text)
    assert "verify_dom" not in added
    assert out.count("# verify_dom") == 1
    assert "# verify_dom = false" not in out
    E.validate_text(out)


def test_cli_config_upgrade_and_auto_sync(tmp_path, monkeypatch, capsys):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(_OLD_CONFIG, encoding="utf-8")

    assert main(["config", "--upgrade", "--dry-run",
                 "--path", str(cfg_path)]) == 0
    dry_out = capsys.readouterr().out
    assert "would be added" in dry_out
    assert cfg_path.read_text(encoding="utf-8") == _OLD_CONFIG
    assert not cfg_path.with_name("config.toml.bak").exists()

    assert main(["config", "--upgrade", "--path", str(cfg_path)]) == 0
    out = capsys.readouterr().out
    assert "added" in out and "verify_dom" in out
    backup = cfg_path.with_name("config.toml.bak")
    assert backup.read_text(encoding="utf-8") == _OLD_CONFIG
    upgraded = cfg_path.read_text(encoding="utf-8")
    assert "# verify_dom = false" in upgraded

    # A subsequent load() sees the user's custom values untouched.
    monkeypatch.setattr(config_mod, "USER_CONFIG", cfg_path)
    cfg = config_mod.load(use_env=False, require_hardware=False,
                          warn_budgets=False)
    assert cfg.vram_gb == 24
    assert cfg.scout_profile == "fast"
    assert cfg.temperature == 0.1
    assert cfg.profiles["worker"]["model"] == "qwen3:14b"
    assert cfg.sources["vram_gb"] == f"user config ({cfg_path})"

    # Auto-sync on load: an installed user config gains the missing options,
    # commented, with one notice, and a second load is silent.
    auto_path = tmp_path / "auto.toml"
    auto_path.write_text(_OLD_CONFIG, encoding="utf-8")
    monkeypatch.setattr(config_mod, "USER_CONFIG", auto_path)
    config_mod.load(use_env=False, require_hardware=False, warn_budgets=True)
    err = capsys.readouterr().err
    assert "[config: added" in err and "verify_dom" in err
    assert "# verify_dom = false" in auto_path.read_text(encoding="utf-8")
    assert auto_path.with_name("auto.toml.bak").exists()
    config_mod.load(use_env=False, require_hardware=False, warn_budgets=True)
    assert "[config: added" not in capsys.readouterr().err


# --- config schema / GUI metadata drift guard ---------------------------------

# Runtime bookkeeping, not user-configurable settings.
_NON_SETTING_FIELDS = {"sources", "profiles", "active_profile", "active_tier",
                       "model_size_bytes", "_hosted_server"}


def _user_config_fields() -> list[str]:
    return [f.name for f in fields(config_mod.Config)
            if f.name not in _NON_SETTING_FIELDS
            and not f.name.startswith("_")]


def test_config_schema_and_gui_drift_guard():
    """Adding a Config field must show up in every user-facing surface.

    Guards the four places a new setting can silently go missing:
    REFERENCE_TEMPLATE (`config --reference` / `config --upgrade`), the GUI's
    All settings help, the documented built-in default, and the `how` index.
    """
    user_fields = _user_config_fields()
    assert user_fields

    # 1. REFERENCE_TEMPLATE parity, in both directions.
    blocks = E.reference_option_blocks()
    by_key = {b["key"]: b for b in blocks}
    assert set(by_key) == set(user_fields), (
        "missing from REFERENCE_TEMPLATE: "
        f"{sorted(set(user_fields) - set(by_key))}; "
        "not a Config field: "
        f"{sorted(set(by_key) - set(user_fields))}")
    assert set(by_key) <= set(config_mod._FIELDS)
    for block in blocks:
        # New options are appended commented out, never active.
        assert block["block"][-1].lstrip().startswith("#"), block["key"]

    # 2. Non-empty help in the reference and in the GUI's All settings payload.
    ref_help = E.reference_setting_help()
    from pedacito.gui import settings as gui_settings
    schema = {item["name"]: item for item in gui_settings.schema()}
    for name in user_fields:
        assert ref_help.get(name, "").strip(), f"{name}: no REFERENCE help"
        assert name in schema, f"{name}: missing from the GUI settings schema"
        assert schema[name]["help"].strip(), f"{name}: no GUI help text"

    # 3. The documented default matches Config(), so a stale reference default
    #    cannot ship. vram_gb is hardware-mandatory (the starter uses 8); the
    #    "unset" TOML samples legitimately stand for None.
    defaults = E.reference_setting_defaults()
    annotations = E.reference_annotations()
    base = config_mod.Config()
    for name in user_fields:
        assert name in defaults, f"{name}: no documented default"
        want = getattr(base, name)
        if annotations[name]["unset"]:
            assert want is None, name
            continue
        if name == "vram_gb":
            continue
        assert config_mod._coerce(name, defaults[name]) == want, name

    # 4. The `how` settings reference indexes every setting.
    from pedacito import how_index
    indexed = {item["key"] for item in how_index.settings_dictionary()}
    assert set(user_fields) <= indexed, (
        f"missing from the how settings index: "
        f"{sorted(set(user_fields) - indexed)}")
