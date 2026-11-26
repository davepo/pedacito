"""Round 2 remediation (Track R2-A) -- core defense & robustness fixes.

Proves, with the real engine surfaces and no network, the six R2 fixes:

* R2-BUG-02 -- `select.select()` / `index.build()` never index an out-of-root
  file under a fake root-relative basename (`f.name`).
* R2-BUG-01 -- a UTF-8-BOM `.pedacito/policy.toml` trusted via `trust_project`
  parses cleanly and enforces `[paths].deny` (no `schema_error`).
* R2-BUG-04 -- `tool_outline` refuses an on-disk sensitive file with the
  explicit sensitive-file message (not "not in the index ... use grep").
* R2-BUG-05 -- `_is_prompt_echo_summary` catches whitespace/Unicode-mutated
  echoes while keeping real summaries `False`.
* R2-BUG-06 -- an interleaved run through `loop.run_task` emits exactly one
  `task_start` and a strictly monotonic `seq`.
* R2-UI-06 -- a UTF-8-BOM `.gitignore` is read as ignoring `.env` by both
  `select` and `security`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import config as config_mod  # noqa: E402
from pedacito import index as index_mod  # noqa: E402
from pedacito import journal as journal_mod  # noqa: E402
from pedacito import loop  # noqa: E402
from pedacito import policy  # noqa: E402
from pedacito import security  # noqa: E402
from pedacito import tools  # noqa: E402
from pedacito.config import Config  # noqa: E402
from pedacito.select import CODE_EXTS, REFERENCE_EXTS, Matcher, select  # noqa: E402

from test_interleaved import FIX_FLOW, ScriptedTools, make_agent  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def home(tmp_path, monkeypatch):
    """An isolated user config dir so trust/policy never touch the real one."""
    monkeypatch.setattr(config_mod, "USER_CONFIG",
                        tmp_path / "home" / "config.toml")
    (tmp_path / "home").mkdir()
    return tmp_path / "home"


def _w(root: Path, rel: str, text: str = "x = 1\n") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# ============================================================================
# R2-BUG-02 -- select() and index.build() skip out-of-root files
# ============================================================================

def test_select_skips_explicit_out_of_root_file(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    _w(outside, "outside.py", "y = 2\n")

    got = select([str(outside / "outside.py")], root,
                 Matcher.from_files(root), CODE_EXTS | REFERENCE_EXTS)
    assert got == []


def test_select_keeps_in_root_files(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    _w(root, "inside.py", "x = 1\n")

    got = select([str(root)], root, Matcher.from_files(root),
                 CODE_EXTS | REFERENCE_EXTS)
    rels = {f.resolve().relative_to(root.resolve()).as_posix() for f in got}
    assert "inside.py" in rels


def test_index_build_skips_out_of_root_files(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    _w(root, "inside.py", "def f():\n    return 1\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    _w(outside, "outside.py", "def g():\n    return 2\n")

    idx = index_mod.build([str(root), str(outside / "outside.py")], Config(),
                          client=None, summarise=False, root=root, verbose=False)
    files = set(idx.files)
    assert "inside.py" in files
    assert "outside.py" not in files
    assert all(c.file != "outside.py" for c in idx.chunks)


def test_index_build_defense_in_depth_never_falls_back_to_fname(
        tmp_path, monkeypatch):
    """Even if `select` handed `build` an out-of-root file, `build` drops it
    rather than indexing it under a fake `f.name` basename."""
    root = tmp_path / "proj"
    root.mkdir()
    _w(root, "inside.py", "def f():\n    return 1\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    _w(outside, "outside.py", "def g():\n    return 2\n")

    real_select = index_mod.select

    def fake_select(paths, root_, matcher, exts):
        return real_select(paths, root_, matcher, exts) + \
            [outside / "outside.py"]

    monkeypatch.setattr(index_mod, "select", fake_select)
    idx = index_mod.build([str(root)], Config(), client=None, summarise=False,
                          root=root, verbose=False)
    files = set(idx.files)
    assert "inside.py" in files
    assert "outside.py" not in files


# ============================================================================
# R2-BUG-01 -- UTF-8-BOM policy.toml parses and enforces
# ============================================================================

def test_utf8_bom_policy_toml_trusts_parses_and_enforces(tmp_path, home):
    root = tmp_path / "proj"
    root.mkdir()
    pol = policy.policy_path(root)
    pol.parent.mkdir(parents=True, exist_ok=True)
    pol.write_bytes(b"\xef\xbb\xbf"
                    + b'[paths]\ndeny = ["sub/secret.env"]\n')

    config_mod.trust_project(pol)
    assert config_mod.is_trusted(pol)

    data = policy.load_policy_file(root)
    assert "_schema_error" not in data
    assert data.get("paths", {}).get("deny") == ["sub/secret.env"]

    cfg = config_mod.load(root=root, require_hardware=False,
                          upgrade_user_config=False)
    assert policy.policy_active(cfg, root)
    d = policy.enforce(cfg, "write", "sub/secret.env", root=root)
    assert d.decision == "deny"
    assert d.matched_rule.startswith("policy.toml:paths.deny")


def test_utf16_bom_policy_toml_fails_closed_with_schema_error(tmp_path, home):
    root = tmp_path / "proj"
    root.mkdir()
    pol = policy.policy_path(root)
    pol.parent.mkdir(parents=True, exist_ok=True)
    pol.write_bytes(b"\xff\xfe" + 'permission_mode = "plan"\n'.encode("utf-16-le"))

    config_mod.trust_project(pol)
    data = policy.load_policy_file(root)
    assert "_schema_error" in data


# ============================================================================
# R2-BUG-04 -- tool_outline refuses a sensitive on-disk file
# ============================================================================

def test_tool_outline_refuses_on_disk_sensitive_file(tmp_path):
    root = tmp_path / "proj"
    _w(root, "sub/secret.env", "API_KEY=super-secret\n")
    idx = index_mod.Index(root, [])          # an empty index: not indexed yet

    out = tools.tool_outline(idx, Config(), "sub/secret.env")
    assert "sensitive" in out.lower()
    assert "not in the index" not in out
    assert "grep" not in out


# ============================================================================
# R2-BUG-05 -- prompt-echo detection survives whitespace/Unicode mutations
# ============================================================================

def test_prompt_echo_catches_whitespace_and_unicode_mutations():
    mutated = (
        "You summarise\nsource code",
        "You  summarise  source  code",
        "You summarise\u00a0source code",          # NBSP
        "You\u200bsummarise source code",          # zero-width space
        "You\u200csummarise source code",          # zero-width non-joiner
        "You\u200dsummarise source code",          # zero-width joiner
        "\ufeffYou summarise source code",         # BOM / format char
        "\uff39ou \uff53ummarise \uff53ource \uff43ode",  # fullwidth Latin
    )
    for text in mutated:
        assert index_mod._is_prompt_echo_summary(text), repr(text)


def test_prompt_echo_keeps_real_summaries_false():
    real = (
        "Parses TOML config files and merges them into one Config.",
        "Renames one identifier across the files that import it.",
        "Doubles each item in the input list and returns the result.",
    )
    for text in real:
        assert not index_mod._is_prompt_echo_summary(text), repr(text)


# ============================================================================
# R2-BUG-06 -- interleaved run through loop.run_task: one task_start, monotonic
# ============================================================================

def test_interleaved_via_run_task_journal_is_not_duplicated(tmp_path):
    client = ScriptedTools(FIX_FLOW)
    root, agent = make_agent(tmp_path, client, execution_mode="interleaved")
    tr = loop.run_task(root, agent.cfg, client, agent, agent.index,
                       "fix the bulk discount threshold", apply=True)
    assert tr.applied

    jdir = root / ".pedacito" / "journal"
    journal_files = sorted(jdir.glob("*.jsonl"))
    assert len(journal_files) == 1, journal_files
    records = journal_mod.read_journal(journal_files[0])

    starts = [r for r in records if r.kind == "task_start"]
    assert len(starts) == 1
    seqs = [r.seq for r in records]
    assert seqs and seqs[0] == 1
    # Strictly monotonic, no gaps, no reset: 1, 2, 3, ...
    assert seqs == list(range(1, len(seqs) + 1))
    assert records[-1].kind == "task_end"


# ============================================================================
# R2-UI-06 -- UTF-8-BOM .gitignore recognized by select and security
# ============================================================================

def test_utf8_bom_gitignore_matches_in_security_and_select(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / ".gitignore").write_bytes(b"\xef\xbb\xbf.env\n")
    _w(root, ".env", "SECRET=x\n")

    assert security._path_ignored(root, ".env") is True
    matcher = Matcher.from_files(root)
    assert matcher.excluded(".env") is True
    # A non-secret name is not caught by the BOM-prefixed first line.
    assert not matcher.excluded("app.py")
