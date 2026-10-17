"""Offline smoke tests: no LLM server required.

Covers the config-simplification surface (templates parse, tier presets
apply, setup error carries a hint), the edit-block parser/renamer bugs
("nothing to review" on rename tasks), and the coverage ledger in agent.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from pedacito import config as config_mod
from pedacito import edits
from pedacito.agent import Agent


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
        'model = "m"\ntier = "small"\n', encoding="utf-8")
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


def test_search_block_matches_at_eof_with_newline_tail(tmp_path):
    _write(tmp_path / "a.py", "x = 1\n")
    reply = ('--- FILE: a.py\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n'
             '>>>>>>> REPLACE\n')
    results = edits.apply_edits(tmp_path, edits.parse_edits(reply),
                                backup=False, dry_run=False)
    assert results[0].ok, results[0].message
    assert (tmp_path / "a.py").read_text() == "x = 2\n"


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
