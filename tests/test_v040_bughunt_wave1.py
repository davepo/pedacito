"""WP-B1 (Bug-Hunt Remediation Wave 1): security, policy trust, the rename
policy gate, D10 indexing/outline, UTF-16 config, and the prompt-echo guard.

Six bugs from V040_BUG_HUNT_REPORT.md, each proven with the real engine
surfaces (config loading, the policy engine, file selection, index build, the
`outline` tool and `pdc trust` / `pdc rename` CLI commands). Offline and
platform-neutral: no network, no model server.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import cli, config as config_mod, index as index_mod, policy, tools  # noqa: E402
from pedacito.config import Config, ConfigError  # noqa: E402
from pedacito.select import CODE_EXTS, REFERENCE_EXTS, Matcher, select  # noqa: E402


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An isolated user config dir: the trust ledger and user config.toml live
    under it, so `trust`/`load` never touch the developer's real config."""
    monkeypatch.setattr(config_mod, "USER_CONFIG",
                        tmp_path / "home" / "config.toml")
    (tmp_path / "home").mkdir()
    return tmp_path / "home"


def _trust(root: Path, yes: bool = True) -> int:
    args = argparse.Namespace(paths=[str(root)], root=str(root), yes=yes,
                              config=None, profile=None)
    return cli._cmd_trust(args)


def _rename(root: Path, old: str, new: str, *, apply: bool = True,
            file: str = "") -> int:
    args = argparse.Namespace(old=old, new=new, file=file, strings=False,
                              include_labels=False, apply=apply, root=str(root),
                              config=None, profile=None, paths=[])
    return cli._cmd_rename(args)


def _w(root: Path, rel: str, text: str = "x = 1\n") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# ============================================================================
# BUG-01 -- `pdc trust` trusts `.pedacito/policy.toml`
# ============================================================================

_POLICY = """\
[paths]
deny = ["sub/secret.env"]

[commands]
deny = ["git push*"]
"""


def test_trust_policy_toml_alone_activates_deny_rules(tmp_path, home, capsys):
    root = tmp_path / "proj"
    _w(root, ".pedacito/policy.toml", _POLICY)

    assert _trust(root, yes=True) == 0
    assert "Trusted" in capsys.readouterr().out

    pol_path = policy.policy_path(root)
    assert config_mod.is_trusted(pol_path)

    cfg = config_mod.load(root=root, require_hardware=False,
                          upgrade_user_config=False)
    assert policy.policy_active(cfg, root)

    d = policy.enforce(cfg, "write", "sub/secret.env", root=root)
    assert d.decision == "deny"
    assert d.matched_rule.startswith("policy.toml:paths.deny")
    d = policy.enforce(cfg, "run", "git push origin main", root=root)
    assert d.decision == "deny"
    assert d.matched_rule.startswith("policy.toml:commands.deny")


def test_trust_trusts_both_project_and_policy_files(tmp_path, home, capsys):
    root = tmp_path / "proj"
    _w(root, ".pedacito.toml", 'auto_verify = "echo hi"\n')
    _w(root, ".pedacito/policy.toml", _POLICY)

    assert _trust(root, yes=True) == 0
    out = capsys.readouterr().out
    assert "Trusted" in out

    assert config_mod.is_trusted(root / ".pedacito.toml")
    assert config_mod.is_trusted(policy.policy_path(root))


def test_trust_no_files_mentions_both_names(tmp_path, home, capsys):
    root = tmp_path / "proj"
    root.mkdir()
    assert _trust(root, yes=True) == 0
    out = capsys.readouterr().out
    assert ".pedacito.toml" in out and ".pedacito/policy.toml" in out


# ============================================================================
# BUG-02 -- `pdc rename --apply` is gated by the policy engine
# ============================================================================

_SRC = "def load(x):\n    return x\n\n\ndef use():\n    return load(1)\n"


def test_rename_apply_blocked_in_plan_mode(tmp_path, home, capsys):
    root = tmp_path / "proj"
    _w(root, "app.py", _SRC)
    _w(root, ".pedacito.toml", 'permission_mode = "plan"\n')
    config_mod.trust_project(root / ".pedacito.toml")

    before = (root / "app.py").read_text(encoding="utf-8")
    code = _rename(root, "load", "read_all", apply=True)
    assert code == 1
    assert "denied by policy" in capsys.readouterr().err
    assert (root / "app.py").read_text(encoding="utf-8") == before


def test_rename_apply_blocked_by_paths_deny(tmp_path, home, capsys):
    root = tmp_path / "proj"
    _w(root, "src/app.py", _SRC)
    _w(root, ".pedacito/policy.toml",
       '[paths]\ndeny = ["src/app.py"]\n')
    config_mod.trust_project(policy.policy_path(root))

    before = (root / "src" / "app.py").read_text(encoding="utf-8")
    code = _rename(root, "load", "read_all", apply=True)
    assert code == 1
    assert "denied by policy" in capsys.readouterr().err
    assert (root / "src" / "app.py").read_text(encoding="utf-8") == before


def test_rename_apply_blocked_by_approval_policy_never(tmp_path, home, capsys):
    root = tmp_path / "proj"
    _w(root, "app.py", _SRC)
    _w(root, ".pedacito.toml",
       'permission_mode = "ask"\napproval_policy = "never"\n')
    config_mod.trust_project(root / ".pedacito.toml")

    before = (root / "app.py").read_text(encoding="utf-8")
    code = _rename(root, "load", "read_all", apply=True)
    assert code == 1
    assert "denied by policy" in capsys.readouterr().err
    assert (root / "app.py").read_text(encoding="utf-8") == before


def test_rename_dry_run_is_not_gated_by_policy(tmp_path, home):
    # A dry-run preview writes nothing, so plan mode must not refuse it.
    root = tmp_path / "proj"
    _w(root, "app.py", _SRC)
    _w(root, ".pedacito.toml", 'permission_mode = "plan"\n')
    config_mod.trust_project(root / ".pedacito.toml")

    assert _rename(root, "load", "read_all", apply=False) == 0
    assert (root / "app.py").read_text(encoding="utf-8") == _SRC


# ============================================================================
# BUG-03 & BUG-19 -- sensitive files excluded; templates & pub keys kept
# ============================================================================

def test_select_excludes_sensitive_and_keeps_templates_and_pub(tmp_path):
    root = tmp_path / "proj"
    _w(root, "sub/secret.env", "# fake nested secret\n")
    _w(root, "prod.env", "TOKEN=live\n")
    _w(root, ".env.example", "API_KEY=your-key-here\n")
    _w(root, "id_rsa.pub", "ssh-rsa AAAA public\n")
    _w(root, "app.py", "x = 1\n")

    exts = CODE_EXTS | REFERENCE_EXTS
    got = select([str(root)], root, Matcher.from_files(root), exts)
    rels = {f.resolve().relative_to(root.resolve()).as_posix() for f in got}

    assert "sub/secret.env" not in rels
    assert "prod.env" not in rels
    assert ".env.example" in rels
    assert "id_rsa.pub" in rels
    assert "app.py" in rels


def test_select_never_selects_a_sensitive_file_even_named_explicitly(tmp_path):
    root = tmp_path / "proj"
    _w(root, ".env", "API_KEY=super-secret\n")

    exts = CODE_EXTS | REFERENCE_EXTS
    got = select([str(root / ".env")], root, Matcher.from_files(root), exts)
    assert got == []


def test_index_build_skips_sensitive_and_indexes_templates(tmp_path):
    root = tmp_path / "proj"
    _w(root, "sub/secret.env", "# fake nested secret\n")
    _w(root, ".env.example", "API_KEY=your-key-here\n")
    _w(root, "id_rsa.pub", "ssh-rsa AAAA public\n")
    _w(root, "app.py", "def f():\n    return 1\n")

    idx = index_mod.build([str(root)], Config(), client=None, summarise=False,
                          root=root, verbose=False)
    files = set(idx.files)
    assert "sub/secret.env" not in files
    assert ".env.example" in files
    assert "id_rsa.pub" in files
    assert "app.py" in files


def test_tool_outline_refuses_a_sensitive_file_from_a_stale_index(tmp_path):
    # Simulate an index built before the fix: `sub/secret.env` still holds a
    # reference chunk whose docline is the secret's comment line.
    from pedacito.references import chunk_reference

    root = tmp_path / "proj"
    secret = _w(root, "sub/secret.env", "# fake nested secret\n")
    chunks = chunk_reference("sub/secret.env", secret)
    idx = index_mod.Index(root, chunks)

    out = tools.tool_outline(idx, Config(), "sub/secret.env")
    assert "sensitive" in out.lower()
    assert "# fake nested secret" not in out


# ============================================================================
# BUG-04 -- UTF-16 config is refused cleanly; UTF-8 BOM parses
# ============================================================================

def test_trust_refuses_utf16_config_and_does_not_pin_it(tmp_path, home, capsys):
    root = tmp_path / "proj"
    cfg_path = root / ".pedacito.toml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_bytes(b"\xff\xfe"
                         + 'permission_mode = "plan"\n'.encode("utf-16-le"))

    assert _trust(root, yes=True) == 2
    assert "UTF-16" in capsys.readouterr().err
    assert config_mod.is_trusted(cfg_path) is False


def test_load_raises_clean_config_error_for_utf16(tmp_path, home):
    root = tmp_path / "proj"
    cfg_path = root / ".pedacito.toml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_bytes(b"\xfe\xff"
                         + 'permission_mode = "plan"\n'.encode("utf-16-be"))

    with pytest.raises(ConfigError) as exc:
        config_mod.load(root=root, require_hardware=False,
                        upgrade_user_config=False)
    assert "UTF-16" in str(exc.value)


def test_utf8_bom_config_parses_cleanly(tmp_path, home):
    root = tmp_path / "proj"
    root.mkdir()
    (root / ".pedacito.toml").write_bytes(
        b"\xef\xbb\xbf" + b'temperature = 0.5\n')

    cfg = config_mod.load(root=root, require_hardware=False,
                          upgrade_user_config=False)
    assert cfg.temperature == 0.5


def test_trust_refuses_broken_toml(tmp_path, home, capsys):
    root = tmp_path / "proj"
    root.mkdir()
    (root / ".pedacito.toml").write_text("this is = = not toml\n",
                                         encoding="utf-8")
    assert _trust(root, yes=True) == 2
    assert "invalid TOML" in capsys.readouterr().err
    assert config_mod.is_trusted(root / ".pedacito.toml") is False


# ============================================================================
# BUG-10 -- prompt-echo summaries are detected and never cached
# ============================================================================

ECHO_1 = ('We need to respond to user: "You summarise source code for a '
          'lookup index. Given one unit of code, reply with ONE sentence ..."')
ECHO_2 = ("We need answer user's request: summarize source code for lookup "
          "index. One sentence (two at most) ...")


def test_is_prompt_echo_summary_catches_the_observed_echoes():
    assert index_mod._is_prompt_echo_summary(ECHO_1)
    assert index_mod._is_prompt_echo_summary(ECHO_2)
    assert index_mod._is_prompt_echo_summary(
        "You summarise source code for a lookup index.")
    assert index_mod._is_prompt_echo_summary("")
    assert index_mod._is_prompt_echo_summary("   ")


def test_is_prompt_echo_summary_accepts_real_summaries():
    assert not index_mod._is_prompt_echo_summary(
        "Parses TOML config files and merges them into one Config.")
    assert not index_mod._is_prompt_echo_summary(
        "Renames one identifier across the files that import it.")


class _EchoClient:
    def __init__(self, reply: str):
        self.reply = reply

    def warmup(self):
        pass

    def chat(self, messages, **kwargs):
        return self.reply


def _summarisable(tmp_path):
    (tmp_path / "mod.py").write_text(
        "def process(data):\n"
        "    out = []\n"
        "    for item in data:\n"
        "        out.append(item * 2)\n"
        "    return out\n",
        encoding="utf-8")
    cfg = Config()
    idx = index_mod.build([str(tmp_path)], cfg, client=None, summarise=False,
                          root=tmp_path, verbose=False)
    todo = [c for c in idx.chunks if index_mod.needs_summary(c, cfg)]
    return idx, todo, cfg


def test_prompt_echo_is_not_cached(tmp_path):
    idx, todo, cfg = _summarisable(tmp_path)
    assert todo, "the fixture unit should need a summary"

    cache: dict = {}
    breaker = index_mod._Breaker(cfg.max_consecutive_failures)
    n = index_mod._summarise_units(
        idx, todo, _EchoClient(ECHO_1), cfg, cache, breaker, False, save=False)
    assert n == 0
    assert all(not c.summary for c in todo)
    assert cache == {}


def test_real_summary_is_cached(tmp_path):
    idx, todo, cfg = _summarisable(tmp_path)
    cache: dict = {}
    breaker = index_mod._Breaker(cfg.max_consecutive_failures)
    n = index_mod._summarise_units(
        idx, todo, _EchoClient("Doubles each item in the input list."),
        cfg, cache, breaker, False, save=False)
    assert n == 1
    assert todo[0].summary == "Doubles each item in the input list."
    assert cache
